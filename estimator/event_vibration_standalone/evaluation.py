from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import AnalysisConfig
from .core import KinematicTriplet, robust_corr


def load_ground_truth(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError("Ground truth must be supplied as CSV/TXT.")
    table = pd.read_csv(path)
    if "t_sec" not in table.columns:
        if "t_us" in table.columns:
            table["t_sec"] = table["t_us"].astype(float) * 1e-6
        elif "t" in table.columns:
            values = table["t"].to_numpy(dtype=float)
            table["t_sec"] = values * 1e-6 if np.nanmax(values) > 1000 else values
        else:
            raise ValueError("Ground truth needs t_sec, t_us, or t.")
    return table.sort_values("t_sec").reset_index(drop=True)


def load_metadata(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_frequencies_from_metadata(metadata: Dict[str, Any]) -> List[float]:
    values = metadata.get("active_frequencies_hz", [])
    if isinstance(values, (list, tuple, np.ndarray)):
        return [float(value) for value in values]
    return []


def resample_column(table: pd.DataFrame, target_time: np.ndarray, column: str) -> Optional[np.ndarray]:
    if column not in table.columns:
        return None
    source = table[["t_sec", column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(source) < 2:
        return None
    return np.interp(
        target_time,
        source["t_sec"].to_numpy(dtype=float),
        source[column].to_numpy(dtype=float),
    )


def _metric_block(estimate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    mask = np.isfinite(estimate) & np.isfinite(reference)
    if mask.sum() < 3:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "nrmse_by_reference_rms": np.nan,
            "time_corr": np.nan,
            "estimate_rms": np.nan,
            "reference_rms": np.nan,
            "rms_error_percent": np.nan,
            "estimate_p95": np.nan,
            "reference_p95": np.nan,
            "p95_error_percent": np.nan,
        }
    e = estimate[mask]
    r = reference[mask]
    residual = e - r
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    estimate_rms = float(np.sqrt(np.mean(e**2)))
    reference_rms = float(np.sqrt(np.mean(r**2)))
    nrmse = rmse / reference_rms if reference_rms > 0 else np.nan
    estimate_p95 = float(np.percentile(np.abs(e), 95))
    reference_p95 = float(np.percentile(np.abs(r), 95))
    rms_error = 100.0 * abs(estimate_rms - reference_rms) / reference_rms if reference_rms > 0 else np.nan
    p95_error = 100.0 * abs(estimate_p95 - reference_p95) / reference_p95 if reference_p95 > 0 else np.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse_by_reference_rms": float(nrmse),
        "time_corr": robust_corr(e, r),
        "estimate_rms": estimate_rms,
        "reference_rms": reference_rms,
        "rms_error_percent": float(rms_error),
        "estimate_p95": estimate_p95,
        "reference_p95": reference_p95,
        "p95_error_percent": float(p95_error),
    }


def build_ground_truth_kinematics(
    ground_truth: pd.DataFrame,
    target_time: np.ndarray,
    axis: str,
) -> Tuple[Optional[KinematicTriplet], Dict[str, str]]:
    """Build a reference triplet, preferring supplied physical derivatives."""
    position = resample_column(ground_truth, target_time, f"{axis}_px")
    velocity = resample_column(ground_truth, target_time, f"v{axis}_px_s")
    acceleration = resample_column(ground_truth, target_time, f"a{axis}_px_s2")
    provenance: Dict[str, str] = {}

    if position is None:
        return None, provenance
    position = position - np.mean(position)
    provenance["displacement"] = f"{axis}_px"

    dt = float(np.median(np.diff(target_time)))
    if velocity is None:
        velocity = np.gradient(position, dt, edge_order=2)
        provenance["velocity"] = "derived_from_displacement_with_gradient"
    else:
        provenance["velocity"] = f"v{axis}_px_s"
    if acceleration is None:
        acceleration = np.gradient(velocity, dt, edge_order=2)
        provenance["acceleration"] = "derived_from_velocity_with_gradient"
    else:
        provenance["acceleration"] = f"a{axis}_px_s2"

    frequencies = np.fft.rfftfreq(len(position), d=dt)
    return (
        KinematicTriplet(
            displacement=np.asarray(position, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            acceleration=np.asarray(acceleration, dtype=float),
            frequencies_hz=frequencies,
            spectral_weight=np.ones_like(frequencies),
        ),
        provenance,
    )


def evaluate_kinematics(
    method_results: Dict[str, Dict[str, KinematicTriplet]],
    ground_truth: pd.DataFrame,
    target_time: np.ndarray,
    active_axes: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compare frequency/time methods against optional supplied ground truth."""
    metric_rows: List[Dict[str, Any]] = []
    reference_columns: Dict[str, np.ndarray] = {"t_sec": target_time}
    provenance: Dict[str, Any] = {}
    reference_triplets: Dict[str, KinematicTriplet] = {}

    for axis in ("x", "y"):
        reference, axis_provenance = build_ground_truth_kinematics(
            ground_truth,
            target_time,
            axis,
        )
        if reference is None:
            continue
        reference_triplets[axis] = reference
        provenance[axis] = axis_provenance
        reference_columns[f"{axis}_displacement_gt"] = reference.displacement
        reference_columns[f"{axis}_velocity_gt"] = reference.velocity
        reference_columns[f"{axis}_acceleration_gt"] = reference.acceleration

        for method_name, axis_results in method_results.items():
            if axis not in axis_results:
                continue
            estimate = axis_results[axis]
            for quantity in ("displacement", "velocity", "acceleration"):
                block = _metric_block(
                    getattr(estimate, quantity),
                    getattr(reference, quantity),
                )
                block.update(
                    {
                        "method": method_name,
                        "axis": axis,
                        "quantity": quantity,
                    }
                )
                metric_rows.append(block)

    # Vector acceleration is evaluated only from active axes that have references.
    selected_axes = [axis for axis in ("x", "y") if axis in active_axes]
    if selected_axes and all(axis in reference_triplets for axis in selected_axes):
        reference_vector = np.sqrt(
            sum(reference_triplets[axis].acceleration**2 for axis in selected_axes)
        )
        reference_columns["a_abs_gt"] = reference_vector
        for method_name, axis_results in method_results.items():
            if not all(axis in axis_results for axis in selected_axes):
                continue
            estimate_vector = np.sqrt(
                sum(axis_results[axis].acceleration**2 for axis in selected_axes)
            )
            block = _metric_block(estimate_vector, reference_vector)
            block.update(
                {
                    "method": method_name,
                    "axis": "vector",
                    "quantity": "acceleration",
                }
            )
            metric_rows.append(block)

    return pd.DataFrame(metric_rows), pd.DataFrame(reference_columns), provenance


def compare_methods(
    method_results: Dict[str, Dict[str, KinematicTriplet]],
    active_axes: str,
) -> pd.DataFrame:
    """Compare spectral and time differentiation without ground truth."""
    if "frequency_domain" not in method_results or "time_domain" not in method_results:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for axis in ("x", "y"):
        if axis not in method_results["frequency_domain"] or axis not in method_results["time_domain"]:
            continue
        first = method_results["frequency_domain"][axis]
        second = method_results["time_domain"][axis]
        for quantity in ("displacement", "velocity", "acceleration"):
            block = _metric_block(getattr(second, quantity), getattr(first, quantity))
            rows.append(
                {
                    "axis": axis,
                    "quantity": quantity,
                    "reference_method": "frequency_domain",
                    "compared_method": "time_domain",
                    **block,
                }
            )

    selected_axes = [axis for axis in ("x", "y") if axis in active_axes]
    if selected_axes and all(
        axis in method_results["frequency_domain"] and axis in method_results["time_domain"]
        for axis in selected_axes
    ):
        fd_vector = np.sqrt(
            sum(method_results["frequency_domain"][axis].acceleration**2 for axis in selected_axes)
        )
        td_vector = np.sqrt(
            sum(method_results["time_domain"][axis].acceleration**2 for axis in selected_axes)
        )
        block = _metric_block(td_vector, fd_vector)
        rows.append(
            {
                "axis": "vector",
                "quantity": "acceleration",
                "reference_method": "frequency_domain",
                "compared_method": "time_domain",
                **block,
            }
        )
    return pd.DataFrame(rows)


def match_peaks(
    expected_frequencies: Sequence[float],
    predicted_peaks: pd.DataFrame,
    config: AnalysisConfig,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    expected = np.asarray(list(expected_frequencies), dtype=float)
    predicted = predicted_peaks.head(config.top_k_peaks)["frequency_hz"].to_numpy(dtype=float)
    tolerance = config.peak_match_tolerance_hz
    if len(expected) == 0 or len(predicted) == 0:
        metrics = {
            "peak_match_tolerance_hz": tolerance,
            "peak_precision": 0.0 if len(predicted) else np.nan,
            "peak_recall": 0.0 if len(expected) else np.nan,
            "peak_f1": 0.0,
            "matched_peak_count": 0,
            "mean_matched_frequency_error_hz": np.nan,
            "median_matched_frequency_error_hz": np.nan,
            "max_matched_frequency_error_hz": np.nan,
        }
        return metrics, pd.DataFrame()

    real_cost = np.abs(expected[:, None] - predicted[None, :])
    invalid_cost = 1e6
    constrained = np.where(real_cost <= tolerance, real_cost, invalid_cost)
    dummy = np.full((len(expected), len(expected)), tolerance + 1.0)
    rows, columns = linear_sum_assignment(np.hstack([constrained, dummy]))

    details: List[Dict[str, Any]] = []
    matched_errors: List[float] = []
    matched_prediction_indices = set()
    for expected_index, assignment_index in zip(rows, columns):
        if assignment_index < len(predicted) and real_cost[expected_index, assignment_index] <= tolerance:
            error = float(real_cost[expected_index, assignment_index])
            matched_errors.append(error)
            matched_prediction_indices.add(int(assignment_index))
            details.append(
                {
                    "expected_frequency_hz": float(expected[expected_index]),
                    "predicted_frequency_hz": float(predicted[assignment_index]),
                    "absolute_error_hz": error,
                    "matched": True,
                }
            )
        else:
            details.append(
                {
                    "expected_frequency_hz": float(expected[expected_index]),
                    "predicted_frequency_hz": np.nan,
                    "absolute_error_hz": np.nan,
                    "matched": False,
                }
            )

    matched = len(matched_errors)
    precision = matched / len(predicted) if len(predicted) else np.nan
    recall = matched / len(expected) if len(expected) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    metrics = {
        "peak_match_tolerance_hz": float(tolerance),
        "peak_precision": float(precision),
        "peak_recall": float(recall),
        "peak_f1": float(f1),
        "matched_peak_count": int(matched),
        "mean_matched_frequency_error_hz": float(np.mean(matched_errors)) if matched_errors else np.nan,
        "median_matched_frequency_error_hz": float(np.median(matched_errors)) if matched_errors else np.nan,
        "max_matched_frequency_error_hz": float(np.max(matched_errors)) if matched_errors else np.nan,
    }
    return metrics, pd.DataFrame(details)
