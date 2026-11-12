from __future__ import annotations

import json
import math
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

from event_converter import emulate_events_h5, v2e_available
from ground_truth import write_acceleration_labels, write_ground_truth
from plotting import save_experiment_plots, save_global_plots
from trajectories import FREQ_MAX_HZ, FREQ_MIN_HZ, evaluate_motion, expected_top_frequencies, frequency_bounds, make_motion_spec
from video_generator import write_video


AMPLITUDE_TARGETS_PX = {
    "low": 2.5,
    "medium": 6.0,
    "high": 11.0,
}


def build_experiment_designs() -> List[Dict[str, object]]:
    designs = []
    plan = [
        ("sparse_multi_sine", 8),
        ("dense_multi_sine", 8),
        ("band_limited_random_phase", 6),
        ("chirp", 4),
        ("mixed_bands", 4),
    ]
    amp_cycle = ["low", "medium", "high"]
    idx = 1
    for motion_type, count in plan:
        for local_idx in range(count):
            amp_level = amp_cycle[local_idx % len(amp_cycle)]
            two_axis = (idx % 5 == 0) or (motion_type in {"mixed_bands", "chirp"} and local_idx % 2 == 1)
            contrast_base = {"low": 0.52, "medium": 0.72, "high": 0.92}[amp_level]
            design = {
                "experiment": f"experiment_{idx:03d}",
                "motion_type": motion_type,
                "amplitude_level": amp_level,
                "target_peak_displacement_px": AMPLITUDE_TARGETS_PX[amp_level],
                "two_axis_motion": two_axis,
                "marker_contrast": float(np.clip(contrast_base + 0.03 * ((idx % 3) - 1), 0.45, 0.98)),
                "noise_rate_hz": float([0.0, 250.0, 800.0][idx % 3]),
                "random_seed": 7300 + idx,
                "spectral_variant": _spectral_variant(motion_type, local_idx),
                "num_components": _component_count(motion_type, local_idx),
            }
            designs.append(design)
            idx += 1
    return designs


def build_dataset(dataset_root: Path) -> Dict[str, object]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "experiments").mkdir(exist_ok=True)
    (dataset_root / "plots").mkdir(exist_ok=True)

    config = default_config()
    write_config(dataset_root / "config.yaml", config)
    rows = []
    for design in build_experiment_designs():
        rows.append(_build_one_experiment(dataset_root, design, config))

    summary = pd.DataFrame(rows)
    summary.to_csv(dataset_root / "dataset_summary.csv", index=False)
    save_global_plots(dataset_root, summary)
    write_readme(dataset_root, summary, config)
    zip_path = dataset_root.with_suffix(".zip")
    write_zip(dataset_root, zip_path)
    return {
        "dataset_root": str(dataset_root),
        "zip_path": str(zip_path),
        "summary": summary,
        "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
    }


def default_config() -> Dict[str, object]:
    return {
        "frequency_range_hz": [FREQ_MIN_HZ, FREQ_MAX_HZ],
        "samples_per_highest_cycle": 20,
        "video_fps": 40000.0,
        "duration_sec": 0.2,
        "resolution": [240, 180],
        "marker_radius_px": 5.0,
        "base_position_px": [120.0, 90.0],
        "event_format": "HDF5 dataset events with columns [t_us, x, y, p]",
        "event_fallback": "analytic_marker_edge_event_emulator",
        "v2e_available_at_generation": bool(v2e_available()),
    }


def _build_one_experiment(dataset_root: Path, design: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    exp_id = str(design["experiment"])
    exp_dir = dataset_root / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    row = _base_summary_row(design, config)

    try:
        rng = np.random.default_rng(int(design["random_seed"]))
        fps = float(config["video_fps"])
        duration = float(config["duration_sec"])
        frame_count = int(math.ceil(fps * duration))
        t_sec = np.arange(frame_count, dtype=float) / fps
        target = float(design["target_peak_displacement_px"])
        spec = make_motion_spec(
            str(design["motion_type"]),
            target,
            rng,
            two_axis_motion=bool(design["two_axis_motion"]),
            num_components=int(design["num_components"]) if design["num_components"] else None,
            spectral_variant=str(design["spectral_variant"]),
        )
        motion = evaluate_motion(spec, t_sec, target)
        expected = expected_top_frequencies(spec)
        f_min, f_max = frequency_bounds(spec.active_frequencies_hz)

        gt = write_ground_truth(exp_dir / "ground_truth.csv", t_sec, motion)
        write_acceleration_labels(exp_dir / "acceleration_labels.csv", gt)
        save_experiment_plots(exp_dir, gt, expected, fps)

        video_result = write_video(
            exp_dir / "video.mp4",
            exp_dir / "preview.png",
            motion["x_px"],
            motion["y_px"],
            fps,
            tuple(config["resolution"]),
            float(config["marker_radius_px"]),
            float(design["marker_contrast"]),
            tuple(config["base_position_px"]),
        )

        event_info = emulate_events_h5(
            exp_dir / "events.h5",
            motion["x_px"],
            motion["y_px"],
            t_sec,
            tuple(config["resolution"]),
            float(config["marker_radius_px"]),
            float(design["marker_contrast"]),
            tuple(config["base_position_px"]),
            random_seed=int(design["random_seed"]) + 10000,
            noise_rate_hz=float(design["noise_rate_hz"]),
            refractory_us=0,
        )

        metadata = {
            "experiment_id": exp_id,
            "motion_type": spec.motion_type,
            "amplitude_level": design["amplitude_level"],
            "duration_sec": duration,
            "video_fps": fps,
            "frame_count": frame_count,
            "resolution": config["resolution"],
            "marker_radius_px": config["marker_radius_px"],
            "marker_contrast": design["marker_contrast"],
            "frequency_range_hz": [FREQ_MIN_HZ, FREQ_MAX_HZ],
            "active_frequencies_hz": spec.active_frequencies_hz,
            "amplitudes_x_px": spec.amplitudes_x_px,
            "amplitudes_y_px": spec.amplitudes_y_px,
            "phases_x_rad": spec.phases_x_rad,
            "phases_y_rad": spec.phases_y_rad,
            "expected_top_frequencies_hz": expected,
            "spectral_envelope": spec.spectral_envelope,
            "target_peak_displacement_px": target,
            "total_rms_displacement_px": motion["total_rms_displacement_px"],
            "total_rms_velocity_px_s": motion["total_rms_velocity_px_s"],
            "total_rms_acceleration_px_s2": motion["total_rms_acceleration_px_s2"],
            "peak_displacement_px": motion["peak_displacement_px"],
            "peak_velocity_px_s": motion["peak_velocity_px_s"],
            "peak_acceleration_px_s2": motion["peak_acceleration_px_s2"],
            "random_seed": design["random_seed"],
            "event_conversion_method": event_info["event_conversion_method"],
            "v2e_available_at_generation": event_info["v2e_available"],
            "video_writer": video_result,
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        qc = inspect_experiment(exp_dir, metadata)
        row.update(
            {
                "duration_sec": duration,
                "video_fps": fps,
                "num_components": len(spec.active_frequencies_hz),
                "min_frequency_hz": f_min,
                "max_frequency_hz": f_max,
                "expected_top_frequencies_hz": json.dumps(expected),
                "target_peak_displacement_px": target,
                "rms_displacement_px": motion["total_rms_displacement_px"],
                "rms_velocity_px_s": motion["total_rms_velocity_px_s"],
                "rms_acceleration_px_s2": motion["total_rms_acceleration_px_s2"],
                "peak_displacement_px": motion["peak_displacement_px"],
                "peak_velocity_px_s": motion["peak_velocity_px_s"],
                "peak_acceleration_px_s2": motion["peak_acceleration_px_s2"],
                "total_events": event_info["total_events"],
                "event_rate_hz": event_info["event_rate_hz"],
                "status": "ok" if qc["ok"] else "failed",
                "failure_reason": qc["reason"],
            }
        )
    except Exception as exc:
        row.update({"status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return row


def inspect_experiment(exp_dir: Path, metadata: Dict[str, object]) -> Dict[str, object]:
    failures = []
    for name in ["events.h5", "ground_truth.csv", "acceleration_labels.csv", "metadata.json", "video.mp4"]:
        if not (exp_dir / name).exists() or (exp_dir / name).stat().st_size == 0:
            failures.append(f"missing_or_empty:{name}")
    try:
        with h5py.File(exp_dir / "events.h5", "r") as h5:
            if "events" not in h5 or h5["events"].shape[0] <= 0:
                failures.append("empty_h5_events")
    except Exception as exc:
        failures.append(f"h5_read_error:{exc}")
    try:
        pd.read_csv(exp_dir / "ground_truth.csv")
        pd.read_csv(exp_dir / "acceleration_labels.csv")
    except Exception as exc:
        failures.append(f"csv_read_error:{exc}")
    freqs = np.asarray(metadata.get("active_frequencies_hz", []), dtype=float)
    if len(freqs) and (np.nanmin(freqs) < FREQ_MIN_HZ - 1e-9 or np.nanmax(freqs) > FREQ_MAX_HZ + 1e-9):
        failures.append("frequency_out_of_range")
    for plot in ["displacement.png", "velocity.png", "acceleration.png", "ground_truth_psd.png", "trajectory.png"]:
        path = exp_dir / "plots" / plot
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing_plot:{plot}")
    return {"ok": not failures, "reason": "; ".join(failures)}


def write_zip(dataset_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in dataset_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(dataset_root.parent))


def write_config(path: Path, config: Dict[str, object]) -> None:
    lines = ["# Broadband synthetic event vibration dataset configuration"]
    for key, value in config.items():
        lines.append(f"{key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(dataset_root: Path, summary: pd.DataFrame, config: Dict[str, object]) -> None:
    ok = summary[summary["status"] == "ok"]
    total_events = int(pd.to_numeric(ok["total_events"], errors="coerce").fillna(0).sum()) if len(ok) else 0
    text = f"""# Broadband Synthetic Event-Camera Vibration Dataset

This standalone dataset evaluates event-camera broadband vibration monitoring.
It covers sparse multi-sine, dense multi-sine, random-phase band-limited,
chirp, and mixed-band motion from 50 Hz to 2000 Hz.

## Experiment Design

- Experiments: {len(summary)}
- Successful experiments: {len(ok)}
- Sampling/video FPS: {config["video_fps"]} Hz
- Duration: {config["duration_sec"]} s
- Resolution: {config["resolution"][0]} x {config["resolution"][1]}
- Event format: HDF5 dataset `events` with rows `[t_us, x, y, p]`
- Total events: {total_events}

Amplitude levels are systematic:

- low: weak displacement, lower event density
- medium: moderate displacement and event density
- high: stronger displacement and higher event density

## File Structure

Each `experiments/experiment_XXX/` folder contains:

- `video.mp4`
- `events.h5`
- `ground_truth.csv`
- `acceleration_labels.csv`
- `metadata.json`
- `plots/displacement.png`
- `plots/velocity.png`
- `plots/acceleration.png`
- `plots/ground_truth_psd.png`
- `plots/trajectory.png`

## Ground Truth Columns

`ground_truth.csv` contains:

- `t_sec`
- `x_px`, `y_px`
- `vx_px_s`, `vy_px_s`
- `ax_px_s2`, `ay_px_s2`
- `a_abs_px_s2`

`acceleration_labels.csv` contains:

- `t_sec`
- `ax_px_s2`, `ay_px_s2`, `a_abs_px_s2`

## Event Format

`events.h5` contains a dataset named `events`.
Rows are ordered as `[t_us, x, y, p]`, where `p=1` is ON and `p=0` is OFF.
The event files are compatible with notebooks that support either this dataset
layout or separate `t_us`, `x`, `y`, `p` datasets.

## Quality Control

The generator checks that:

- `events.h5` is non-empty and readable
- ground-truth CSV files are readable
- active frequencies are within 50-2000 Hz
- required plots exist
- `dataset_summary.csv` records status and failure reason per experiment

## Analysis Notebook

Use `synthetic_event_vibration_broadband_spectral_analysis_v5.ipynb` later by
pointing it to this folder. The expected top frequencies are stored in
`metadata.json` and `dataset_summary.csv` for top-k spectral recovery and
band-power evaluation.
"""
    (dataset_root / "README.md").write_text(text, encoding="utf-8")


def _base_summary_row(design: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    return {
        "experiment": design["experiment"],
        "motion_type": design["motion_type"],
        "amplitude_level": design["amplitude_level"],
        "duration_sec": config["duration_sec"],
        "video_fps": config["video_fps"],
        "num_components": design["num_components"],
        "min_frequency_hz": "",
        "max_frequency_hz": "",
        "expected_top_frequencies_hz": "[]",
        "target_peak_displacement_px": design["target_peak_displacement_px"],
        "rms_displacement_px": "",
        "rms_velocity_px_s": "",
        "rms_acceleration_px_s2": "",
        "peak_displacement_px": "",
        "peak_velocity_px_s": "",
        "peak_acceleration_px_s2": "",
        "total_events": 0,
        "event_rate_hz": 0,
        "marker_contrast": design["marker_contrast"],
        "two_axis_motion": design["two_axis_motion"],
        "status": "pending",
        "failure_reason": "",
    }


def _component_count(motion_type: str, local_idx: int) -> int:
    if motion_type == "sparse_multi_sine":
        return [5, 6, 7, 8, 9, 10, 6, 8][local_idx]
    if motion_type == "dense_multi_sine":
        return [20, 24, 28, 32, 36, 40, 45, 50][local_idx]
    if motion_type == "band_limited_random_phase":
        return [60, 70, 80, 90, 100, 75][local_idx]
    if motion_type == "mixed_bands":
        return [10, 12, 14, 16][local_idx]
    return 1


def _spectral_variant(motion_type: str, local_idx: int) -> str:
    if motion_type == "band_limited_random_phase":
        return ["flat", "low_tilt", "mid_bump", "flat", "low_tilt", "mid_bump"][local_idx]
    if motion_type == "chirp":
        return ["linear", "linear", "log", "linear"][local_idx]
    return "flat"
