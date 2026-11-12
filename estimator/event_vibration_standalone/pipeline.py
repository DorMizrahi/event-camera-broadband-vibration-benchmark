from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .core import (
    EventBinner,
    KinematicTriplet,
    KinematicsEstimator,
    SpectralEstimator,
    kinematic_summary,
)
from .evaluation import (
    compare_methods,
    evaluate_kinematics,
    expected_frequencies_from_metadata,
    load_ground_truth,
    load_metadata,
    match_peaks,
)
from .io import EventFileInfo, load_events
from .plotting import (
    plot_chirp_ridge,
    plot_event_accumulation,
    plot_kinematic_method_comparison,
    plot_method_metric_summary,
    plot_primary_signal,
    plot_spectrum_and_peaks,
    plot_vector_acceleration_comparison,
)
from .roi import ROI


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)


def _flatten_summary(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


@dataclass
class AnalysisResult:
    output_dir: Path
    config: AnalysisConfig
    file_info: EventFileInfo
    roi: ROI
    binned_signals: pd.DataFrame
    source_signals: Dict[str, str]
    dominant_frequencies: pd.DataFrame
    spectral_descriptors: pd.DataFrame
    vector_periodogram: pd.DataFrame
    axis_periodograms: Dict[str, pd.DataFrame]
    windowed_stability: pd.DataFrame
    chirp_ridge: pd.DataFrame
    kinematics: pd.DataFrame
    kinematic_metrics: pd.DataFrame
    method_comparison: pd.DataFrame
    ground_truth_metrics: pd.DataFrame
    peak_metrics: Dict[str, float]
    peak_matches: pd.DataFrame
    signal_quality: Dict[str, Any]
    summary: Dict[str, Any]


def _quality_metrics(
    binned: pd.DataFrame,
    descriptors: pd.DataFrame,
    source_signals: Dict[str, str],
    windowed: pd.DataFrame,
) -> Dict[str, Any]:
    quality: Dict[str, Any] = {
        "bin_count": int(len(binned)),
        "duration_sec": float(binned["t_sec"].iloc[-1] - binned["t_sec"].iloc[0]) if len(binned) > 1 else 0.0,
        "mean_events_per_bin": float(binned["event_count"].mean()),
        "median_events_per_bin": float(binned["event_count"].median()),
        "empty_bin_fraction": float(np.mean(binned["event_count"].to_numpy(dtype=float) == 0)),
    }
    for axis, source in source_signals.items():
        observed_column = f"{source}_observed"
        if observed_column in binned.columns:
            quality[f"{axis}_primary_observed_fraction"] = float(
                binned[observed_column].mean()
            )
        quality[f"{axis}_primary_signal"] = source
    if not descriptors.empty:
        for axis, source in source_signals.items():
            row = descriptors[descriptors["signal"] == source]
            if not row.empty:
                for key in (
                    "band_power_total",
                    "spectral_centroid_hz",
                    "spectral_bandwidth_hz",
                    "spectral_flatness",
                    "spectral_snr_db",
                    "signal_std",
                ):
                    if key in row.columns:
                        quality[f"{axis}_{key}"] = float(row.iloc[0][key])
    if not windowed.empty and "window_psd_log_corr_to_full" in windowed.columns:
        values = windowed["window_psd_log_corr_to_full"].dropna().to_numpy(dtype=float)
        quality["window_psd_log_corr_mean"] = float(np.mean(values)) if len(values) else np.nan
        quality["window_psd_log_corr_min"] = float(np.min(values)) if len(values) else np.nan
    quality["calibration_applied"] = False
    quality["calibration_note"] = (
        "No learned frequency-response or RMS calibration is used. Outputs are direct pixel-domain estimates."
    )
    return quality


def _assemble_kinematics_table(
    time: np.ndarray,
    method_results: Dict[str, Dict[str, KinematicTriplet]],
    active_axes: str,
    reference_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    table: Dict[str, np.ndarray] = {"t_sec": np.asarray(time, dtype=float)}
    for method_name, axis_results in method_results.items():
        for axis, triplet in axis_results.items():
            table[f"{axis}_displacement_{method_name}"] = triplet.displacement
            table[f"{axis}_velocity_{method_name}"] = triplet.velocity
            table[f"{axis}_acceleration_{method_name}"] = triplet.acceleration
        selected = [axis for axis in ("x", "y") if axis in active_axes and axis in axis_results]
        if selected:
            table[f"a_abs_{method_name}"] = np.sqrt(
                sum(axis_results[axis].acceleration**2 for axis in selected)
            )
    output = pd.DataFrame(table)
    if reference_table is not None and not reference_table.empty:
        reference = reference_table.copy()
        reference = reference.drop(columns=["t_sec"], errors="ignore")
        output = pd.concat([output.reset_index(drop=True), reference.reset_index(drop=True)], axis=1)
    return output


def _kinematic_metric_table(
    method_results: Dict[str, Dict[str, KinematicTriplet]],
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method_name, axis_results in method_results.items():
        for axis, triplet in axis_results.items():
            for quantity in ("displacement", "velocity", "acceleration"):
                values = getattr(triplet, quantity)
                summary = kinematic_summary(
                    values,
                    config.acceleration_percentile,
                    config.robust_peak_percentile,
                )
                rows.append(
                    {
                        "method": method_name,
                        "axis": axis,
                        "quantity": quantity,
                        **summary,
                    }
                )
        selected = [axis for axis in ("x", "y") if axis in config.active_axes and axis in axis_results]
        if selected:
            vector = np.sqrt(sum(axis_results[axis].acceleration**2 for axis in selected))
            summary = kinematic_summary(
                vector,
                config.acceleration_percentile,
                config.robust_peak_percentile,
            )
            rows.append(
                {
                    "method": method_name,
                    "axis": "vector",
                    "quantity": "acceleration",
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _write_report(result: AnalysisResult) -> None:
    lines = [
        "# Event-camera vibration analysis report",
        "",
        "## Scope",
        "",
        "This run uses the standalone no-calibration pipeline. No learned frequency-response model, fold-specific model, or RMS calibration is applied.",
        "",
        "## Input",
        "",
        f"- File: `{result.file_info.source_path}`",
        f"- Format: `{result.file_info.source_format}`",
        f"- Retained events: {result.file_info.event_count}",
        f"- Duration: {result.file_info.duration_sec:.6f} s" if result.file_info.duration_sec is not None else "- Duration: unavailable",
        f"- ROI: {result.roi.as_tuple()}",
        f"- Active axes: `{result.config.active_axes}`",
        "",
        "## Dominant frequencies",
        "",
    ]
    if result.dominant_frequencies.empty:
        lines.append("No dominant-frequency candidates were reported.")
    else:
        lines.append(result.dominant_frequencies.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Method comparison",
            "",
            "Frequency-domain and time-domain derivatives use the same broadband reconstructed displacement. Their difference isolates the derivative operator.",
            "",
        ]
    )
    if result.method_comparison.empty:
        lines.append("No method-comparison metrics were available.")
    else:
        lines.append(result.method_comparison.to_markdown(index=False))
    if not result.ground_truth_metrics.empty:
        lines.extend(["", "## Ground-truth evaluation", "", result.ground_truth_metrics.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "All displacement, velocity, and acceleration values remain in px, px/s, and px/s². Conversion to physical units requires an external optical scale factor. The absence of learned calibration reduces dependence on the synthetic generator, but it does not by itself guarantee transfer to a physical camera.",
        ]
    )
    (result.output_dir / "measurement_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    input_path: Path,
    *,
    roi: ROI,
    config: AnalysisConfig,
    output_name: Optional[str] = None,
    ground_truth_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    expected_frequencies_hz: Optional[Sequence[float]] = None,
    preview_images: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    run_chirp_analysis: Optional[bool] = None,
) -> AnalysisResult:
    """Run the complete no-calibration analysis on one event recording."""
    config.validate()
    input_path = Path(input_path)
    output_name = output_name or input_path.stem
    output_dir = Path(config.output_root) / output_name
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    event_data = load_events(
        input_path,
        roi=roi.as_tuple(),
        hdf5_chunk_rows=config.hdf5_chunk_rows,
        dat_chunk_duration_us=config.dat_chunk_duration_us,
    )
    binner = EventBinner(config)
    binned = binner.bin(event_data.events)

    spectral = SpectralEstimator(config)
    periodograms, welch_tables, all_signal_peaks, descriptors = spectral.analyze_all_signals(binned)
    vector_periodogram, axis_periodograms, source_signals = spectral.primary_motion_periodogram(
        periodograms,
        binned,
    )
    dominant_frequencies = spectral.extract_peaks(
        vector_periodogram,
        "primary_axis_vector",
    )
    windowed = spectral.windowed_stability(binned, source_signals)

    metadata = load_metadata(metadata_path)
    motion_type = str(metadata.get("motion_type", "")).lower()
    if expected_frequencies_hz is None:
        # A chirp is evaluated as a time-frequency ridge, not as two stationary
        # endpoint peaks. Explicit user-supplied frequencies still take precedence.
        expected_frequencies_hz = (
            [] if "chirp" in motion_type else expected_frequencies_from_metadata(metadata)
        )
    if run_chirp_analysis is None:
        run_chirp_analysis = "chirp" in motion_type
    chirp_table = pd.DataFrame()
    chirp_metrics: Dict[str, float] = {}
    chirp_plot_data = (np.array([]), np.array([]), np.empty((0, 0)))
    if run_chirp_analysis:
        chirp_table, chirp_plot_data = spectral.chirp_ridge(binned, source_signals)
        frequency_range = metadata.get("frequency_range_hz", [])
        if len(frequency_range) >= 2 and not chirp_table.empty:
            start_frequency = float(frequency_range[0])
            end_frequency = float(frequency_range[1])
            duration = float(metadata.get("duration_sec", binned["t_sec"].max()))
            normalized_time = np.clip(
                chirp_table["time_sec"].to_numpy(dtype=float) / max(duration, 1e-12),
                0.0,
                1.0,
            )
            envelope = str(metadata.get("spectral_envelope", "linear_chirp")).lower()
            if "log" in envelope:
                expected_ridge = start_frequency * (end_frequency / start_frequency) ** normalized_time
            else:
                expected_ridge = start_frequency + (end_frequency - start_frequency) * normalized_time
            errors = chirp_table["estimated_ridge_hz"].to_numpy(dtype=float) - expected_ridge
            chirp_table["expected_ridge_hz"] = expected_ridge
            chirp_table["error_hz"] = errors
            chirp_table["absolute_error_hz"] = np.abs(errors)
            chirp_metrics = {
                "chirp_ridge_rmse_hz": float(np.sqrt(np.mean(errors**2))),
                "chirp_ridge_mae_hz": float(np.mean(np.abs(errors))),
                "chirp_ridge_p95_abs_error_hz": float(np.percentile(np.abs(errors), 95)),
                "chirp_ridge_within_tolerance_fraction": float(
                    np.mean(np.abs(errors) <= config.chirp_tolerance_hz)
                ),
            }

    kinematics_estimator = KinematicsEstimator(config)
    method_results: Dict[str, Dict[str, KinematicTriplet]] = {
        "frequency_domain": {},
        "time_domain": {},
    }
    for axis, source in source_signals.items():
        raw_signal = binned[source].to_numpy(dtype=float)
        method_results["frequency_domain"][axis] = kinematics_estimator.frequency_domain(raw_signal)
        method_results["time_domain"][axis] = kinematics_estimator.time_domain(raw_signal)

    method_comparison = compare_methods(method_results, config.active_axes)
    kinematic_metrics = _kinematic_metric_table(method_results, config)

    ground_truth_metrics = pd.DataFrame()
    reference_table = pd.DataFrame()
    ground_truth_provenance: Dict[str, Any] = {}
    if ground_truth_path is not None:
        ground_truth = load_ground_truth(ground_truth_path)
        ground_truth_metrics, reference_table, ground_truth_provenance = evaluate_kinematics(
            method_results,
            ground_truth,
            binned["t_sec"].to_numpy(dtype=float),
            config.active_axes,
        )

    kinematics_table = _assemble_kinematics_table(
        binned["t_sec"].to_numpy(dtype=float),
        method_results,
        config.active_axes,
        reference_table,
    )

    peak_metrics: Dict[str, float] = {}
    peak_matches = pd.DataFrame()
    if expected_frequencies_hz:
        peak_metrics, peak_matches = match_peaks(
            expected_frequencies_hz,
            dominant_frequencies,
            config,
        )

    signal_quality = _quality_metrics(
        binned,
        descriptors,
        source_signals,
        windowed,
    )

    summary: Dict[str, Any] = {
        "analysis_name": output_name,
        "input_file": str(input_path),
        "roi": roi.to_dict(),
        "active_axes": config.active_axes,
        "primary_signals": source_signals,
        "event_count": int(event_data.info.event_count or 0),
        "duration_sec": event_data.info.duration_sec,
        "dominant_frequencies_hz": dominant_frequencies["frequency_hz"].tolist() if not dominant_frequencies.empty else [],
        "calibration_applied": False,
        "time_derivative_method": config.time_derivative_method,
        "ground_truth_used_for_estimation": False,
        "ground_truth_evaluation_enabled": ground_truth_path is not None,
        "expected_frequencies_hz": list(expected_frequencies_hz or []),
        "peak_metrics": peak_metrics,
        "chirp_metrics": chirp_metrics,
        "signal_quality": signal_quality,
        "ground_truth_provenance": ground_truth_provenance,
    }

    # Export numerical results.
    write_json(config.to_dict(), output_dir / "analysis_config_snapshot.json")
    write_json(event_data.info.to_dict(), output_dir / "input_file_info.json")
    write_json(roi.to_dict(), output_dir / "selected_roi.json")
    write_json(signal_quality, output_dir / "signal_quality_metrics.json")
    write_json(summary, output_dir / "analysis_summary.json")
    if peak_metrics:
        write_json(peak_metrics, output_dir / "peak_evaluation_metrics.json")
    if chirp_metrics:
        write_json(chirp_metrics, output_dir / "chirp_ridge_metrics.json")

    binned.to_csv(output_dir / "binned_event_signals.csv", index=False)
    dominant_frequencies.to_csv(output_dir / "dominant_frequencies.csv", index=False)
    all_signal_peaks.to_csv(output_dir / "all_signal_peaks.csv", index=False)
    descriptors.to_csv(output_dir / "all_signal_spectral_descriptors.csv", index=False)
    vector_periodogram.to_csv(output_dir / "primary_vector_periodogram.csv", index=False)
    for axis, table in axis_periodograms.items():
        table.to_csv(output_dir / f"{axis}_primary_periodogram.csv", index=False)
    windowed.to_csv(output_dir / "windowed_spectral_stability.csv", index=False)
    chirp_table.to_csv(output_dir / "chirp_ridge.csv", index=False)
    kinematics_table.to_csv(output_dir / "displacement_velocity_acceleration.csv", index=False)
    kinematic_metrics.to_csv(output_dir / "kinematic_metrics.csv", index=False)
    method_comparison.to_csv(output_dir / "frequency_vs_time_method_comparison.csv", index=False)
    ground_truth_metrics.to_csv(output_dir / "ground_truth_error_metrics.csv", index=False)
    peak_matches.to_csv(output_dir / "peak_matches.csv", index=False)

    result = AnalysisResult(
        output_dir=output_dir,
        config=config,
        file_info=event_data.info,
        roi=roi,
        binned_signals=binned,
        source_signals=source_signals,
        dominant_frequencies=dominant_frequencies,
        spectral_descriptors=descriptors,
        vector_periodogram=vector_periodogram,
        axis_periodograms=axis_periodograms,
        windowed_stability=windowed,
        chirp_ridge=chirp_table,
        kinematics=kinematics_table,
        kinematic_metrics=kinematic_metrics,
        method_comparison=method_comparison,
        ground_truth_metrics=ground_truth_metrics,
        peak_metrics=peak_metrics,
        peak_matches=peak_matches,
        signal_quality=signal_quality,
        summary=summary,
    )

    # Plots are intentionally separate figures rather than crowded multi-panel plots.
    if config.save_plots:
        if preview_images is not None:
            plot_event_accumulation(
                preview_images[0],
                preview_images[1],
                roi,
                plots_dir / "01_event_accumulation_roi.png",
                config.plot_dpi,
            )
        for axis, source in source_signals.items():
            plot_primary_signal(
                binned,
                source,
                axis,
                plots_dir / f"02_primary_signal_{axis}.png",
                config.plot_dpi,
            )
        plot_spectrum_and_peaks(
            vector_periodogram,
            dominant_frequencies,
            config,
            plots_dir / "03_dominant_frequencies.png",
            config.plot_dpi,
        )
        for axis in source_signals:
            for quantity in ("displacement", "velocity", "acceleration"):
                gt_column = f"{axis}_{quantity}_gt"
                ground_truth_values = (
                    kinematics_table[gt_column].to_numpy(dtype=float)
                    if gt_column in kinematics_table.columns
                    else None
                )
                plot_kinematic_method_comparison(
                    binned["t_sec"].to_numpy(dtype=float),
                    method_results["frequency_domain"][axis],
                    method_results["time_domain"][axis],
                    axis,
                    quantity,
                    plots_dir / f"04_{axis}_{quantity}_frequency_vs_time.png",
                    config.plot_dpi,
                    ground_truth=ground_truth_values,
                )

        selected_axes = [axis for axis in ("x", "y") if axis in config.active_axes and axis in source_signals]
        if selected_axes:
            frequency_vector = np.sqrt(
                sum(method_results["frequency_domain"][axis].acceleration**2 for axis in selected_axes)
            )
            time_vector = np.sqrt(
                sum(method_results["time_domain"][axis].acceleration**2 for axis in selected_axes)
            )
            gt_vector = (
                kinematics_table["a_abs_gt"].to_numpy(dtype=float)
                if "a_abs_gt" in kinematics_table.columns
                else None
            )
            plot_vector_acceleration_comparison(
                binned["t_sec"].to_numpy(dtype=float),
                frequency_vector,
                time_vector,
                plots_dir / "05_vector_acceleration_frequency_vs_time.png",
                config.plot_dpi,
                ground_truth=gt_vector,
            )
        if not chirp_table.empty:
            plot_chirp_ridge(
                chirp_plot_data[0],
                chirp_plot_data[1],
                chirp_plot_data[2],
                chirp_table,
                plots_dir / "06_chirp_ridge.png",
                config.plot_dpi,
            )
        plot_method_metric_summary(
            method_comparison,
            plots_dir / "07_method_comparison_summary.png",
            config.plot_dpi,
        )

    _write_report(result)
    return result
