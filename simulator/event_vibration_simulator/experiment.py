from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import resolve_parameters
from .motion import (
    MotionSpec,
    create_motion,
    cycle_information,
    expected_top_frequencies,
)
from .plotting import save_experiment_plots
from .rendering import validate_object_bounds, write_media
from .v2e_runner import convert_video_with_v2e, count_events


def generate_experiment(
    user_parameters: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    progress=print,
) -> dict[str, Any]:
    """Generate one complete experiment through the shared notebook/script engine."""
    root = Path(project_root or Path.cwd()).resolve()
    config, warnings = resolve_parameters(user_parameters, root)
    output_root = Path(config["experiment"]["resolved_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = _prepare_experiment_directory(config, output_root)
    progress(f"Experiment directory: {experiment_dir}")
    _write_json(
        experiment_dir / "generation_status.json",
        {
            "status": "running",
            "started_utc": _utc_now(),
            "events_required_for_base_success": False,
        },
    )

    try:
        _write_json(experiment_dir / "parameters.json", config)
        _write_environment(experiment_dir / "environment.json")
        frame_count = int(config["derived"]["frame_count"])
        fps = float(config["sampling"]["video_fps"])
        time_sec = np.arange(frame_count, dtype=float) / fps

        progress(f"Creating {config['motion']['type']} ground truth ...")
        spec, motion, metrics = create_motion(config, time_sec)
        validate_object_bounds(config, motion["x_px"], motion["y_px"])
        expected_frequencies = expected_top_frequencies(spec)
        cycles = cycle_information(spec, float(config["derived"]["duration_sec"]))
        ground_truth = _write_ground_truth(experiment_dir, time_sec, motion, config)
        plot_paths = save_experiment_plots(
            experiment_dir,
            ground_truth,
            expected_frequencies,
            fps,
            config,
        )
        media = write_media(
            experiment_dir,
            config,
            motion["x_px"],
            motion["y_px"],
            progress=progress,
        )
        event_conversion = convert_video_with_v2e(
            experiment_dir,
            config,
            root,
            progress=progress,
        )

        metadata = _build_metadata(
            experiment_dir=experiment_dir,
            config=config,
            warnings=warnings,
            spec=spec,
            normalization_scale=float(motion["normalization_scale"]),
            metrics=metrics,
            expected_frequencies=expected_frequencies,
            cycles=cycles,
            media=media,
            plot_paths=plot_paths,
            event_conversion=event_conversion,
        )
        _write_json(experiment_dir / "metadata.json", metadata)
        summary = _summary_row(
            experiment_dir,
            config,
            spec,
            metrics,
            expected_frequencies,
            cycles,
            event_conversion,
            warnings,
        )
        pd.DataFrame([summary]).to_csv(
            experiment_dir / "experiment_summary.csv", index=False
        )
        quality = _quality_control(
            experiment_dir, config, frame_count, event_conversion
        )
        _write_json(experiment_dir / "quality_control.json", quality)
        completion_status = (
            "completed"
            if event_conversion["status"] == "completed"
            else "completed_without_events"
        )
        _write_json(
            experiment_dir / "generation_status.json",
            {
                "status": completion_status,
                "completed_utc": _utc_now(),
                "quality_control": quality["overall_status"],
                "event_conversion_status": event_conversion["status"],
                "events_required_for_base_success": False,
            },
        )
        if bool(config["experiment"]["write_manifest"]):
            _write_manifest(experiment_dir)
        dataset_summary_path = _update_dataset_summary(output_root, summary)
        result = {
            "status": completion_status,
            "experiment_name": config["experiment"]["name"],
            "experiment_directory": str(experiment_dir),
            "video_path": str(experiment_dir / "video.mp4"),
            "frames_directory": (
                str(experiment_dir / "frames")
                if bool(config["sampling"]["save_frames"])
                else None
            ),
            "events_path": event_conversion.get("events_path"),
            "event_conversion_status": event_conversion["status"],
            "ground_truth_path": str(experiment_dir / "ground_truth.csv"),
            "acceleration_labels_path": str(
                experiment_dir / "acceleration_labels.csv"
            ),
            "metadata_path": str(experiment_dir / "metadata.json"),
            "dataset_summary_path": str(dataset_summary_path),
            "quality_control": quality["overall_status"],
            "warnings": warnings,
            "metrics": metrics,
        }
        progress(f"Generation finished: {completion_status}.")
        return result
    except Exception as exc:
        _write_json(
            experiment_dir / "generation_status.json",
            {
                "status": "failed",
                "failed_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "reason": str(exc),
            },
        )
        raise


def _prepare_experiment_directory(
    config: dict[str, Any], output_root: Path
) -> Path:
    name = str(config["experiment"]["name"])
    candidate = output_root / name
    policy = str(config["experiment"]["output_policy"])
    if not candidate.exists():
        candidate.mkdir(parents=False)
        return candidate
    if policy == "error":
        raise FileExistsError(
            f"Experiment directory already exists: {candidate}. "
            "Use output_policy='new' or 'overwrite'."
        )
    if policy == "overwrite":
        resolved_root = output_root.resolve()
        resolved_candidate = candidate.resolve()
        if resolved_candidate.parent != resolved_root or resolved_candidate == resolved_root:
            raise RuntimeError("Refusing to overwrite a directory outside output_root.")
        shutil.rmtree(resolved_candidate)
        resolved_candidate.mkdir()
        return resolved_candidate

    suffix = 1
    while True:
        numbered = output_root / f"{name}_{suffix:03d}"
        if not numbered.exists():
            numbered.mkdir()
            config["experiment"]["requested_name"] = name
            config["experiment"]["name"] = numbered.name
            return numbered
        suffix += 1


def _write_ground_truth(
    experiment_dir: Path,
    time_sec: np.ndarray,
    motion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    ground_truth = pd.DataFrame(
        {
            "t_sec": time_sec,
            "x_px": motion["x_px"],
            "y_px": motion["y_px"],
            "vx_px_s": motion["vx_px_s"],
            "vy_px_s": motion["vy_px_s"],
            "ax_px_s2": motion["ax_px_s2"],
            "ay_px_s2": motion["ay_px_s2"],
            "a_abs_px_s2": motion["a_abs_px_s2"],
        }
    )
    meters_per_pixel = config["calibration"].get("meters_per_pixel")
    acceleration_columns = ["t_sec", "ax_px_s2", "ay_px_s2", "a_abs_px_s2"]
    if meters_per_pixel is not None:
        scale = float(meters_per_pixel)
        gravity = float(config["calibration"]["standard_gravity_m_s2"])
        ground_truth["x_m"] = ground_truth["x_px"] * scale
        ground_truth["y_m"] = ground_truth["y_px"] * scale
        ground_truth["vx_m_s"] = ground_truth["vx_px_s"] * scale
        ground_truth["vy_m_s"] = ground_truth["vy_px_s"] * scale
        ground_truth["ax_m_s2"] = ground_truth["ax_px_s2"] * scale
        ground_truth["ay_m_s2"] = ground_truth["ay_px_s2"] * scale
        ground_truth["a_abs_m_s2"] = ground_truth["a_abs_px_s2"] * scale
        ground_truth["a_abs_g"] = ground_truth["a_abs_m_s2"] / gravity
        acceleration_columns.extend(
            ["ax_m_s2", "ay_m_s2", "a_abs_m_s2", "a_abs_g"]
        )
    ground_truth.to_csv(experiment_dir / "ground_truth.csv", index=False)
    ground_truth[acceleration_columns].to_csv(
        experiment_dir / "acceleration_labels.csv", index=False
    )
    return ground_truth


def _build_metadata(
    experiment_dir: Path,
    config: Mapping[str, Any],
    warnings: list[str],
    spec: MotionSpec,
    normalization_scale: float,
    metrics: Mapping[str, float],
    expected_frequencies: list[float],
    cycles: Mapping[str, Any],
    media: Mapping[str, Any],
    plot_paths: list[str],
    event_conversion: Mapping[str, Any],
) -> dict[str, Any]:
    motion_spec = spec.to_dict()
    motion_spec["normalization_scale"] = normalization_scale
    motion_spec["normalized_amplitudes_x_px"] = [
        float(value * normalization_scale) for value in spec.amplitudes_x_px
    ]
    motion_spec["normalized_amplitudes_y_px"] = [
        float(value * normalization_scale) for value in spec.amplitudes_y_px
    ]
    return {
        "schema_version": "1.0",
        "generator": "standalone_event_vibration_simulator",
        "generated_utc": _utc_now(),
        "experiment_id": config["experiment"]["name"],
        "experiment_directory": str(experiment_dir),
        "motion_type": spec.motion_type,
        "sampling": config["sampling"],
        "rendering": config["rendering"],
        "calibration": config["calibration"],
        "motion_configuration": config["motion"],
        "v2e_configuration": config["v2e"],
        "derived": config["derived"],
        "motion_specification": motion_spec,
        "expected_top_frequencies_hz": expected_frequencies,
        "cycle_information": cycles,
        "metrics": dict(metrics),
        "units": {
            "t_sec": "s",
            "x_px_y_px": "pixel",
            "vx_px_s_vy_px_s": "pixel/s",
            "ax_px_s2_ay_px_s2_a_abs_px_s2": "pixel/s^2",
            "physical_columns_present": config["calibration"].get("meters_per_pixel")
            is not None,
        },
        "media": dict(media),
        "plot_paths": plot_paths,
        "event_conversion": dict(event_conversion),
        "warnings": warnings,
    }


def _summary_row(
    experiment_dir: Path,
    config: Mapping[str, Any],
    spec: MotionSpec,
    metrics: Mapping[str, float],
    expected_frequencies: list[float],
    cycles: Mapping[str, Any],
    event_conversion: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    active = [float(value) for value in spec.active_frequencies_hz]
    row: dict[str, Any] = {
        "experiment": config["experiment"]["name"],
        "experiment_directory": experiment_dir.name,
        "motion_type": spec.motion_type,
        "duration_sec": float(config["derived"]["duration_sec"]),
        "video_fps": float(config["sampling"]["video_fps"]),
        "frame_count": int(config["derived"]["frame_count"]),
        "num_components": 1 if spec.motion_type == "chirp" else len(active),
        "min_frequency_hz": min(active),
        "max_frequency_hz": max(active),
        "expected_top_frequencies_hz": json.dumps(expected_frequencies),
        "minimum_cycle_count": float(cycles["minimum_cycle_count"]),
        "maximum_cycle_count": float(cycles["maximum_cycle_count"]),
        "normalization_mode": config["motion"]["normalization_mode"],
        "normalization_value": float(config["motion"]["normalization_value"]),
        "two_axis_motion": bool(config["motion"]["two_axis_motion"]),
        "event_conversion_status": event_conversion["status"],
        "used_v2e": bool(event_conversion["used_v2e"]),
        "total_events": event_conversion.get("event_count"),
        "event_rate_hz": event_conversion.get("event_rate_hz"),
        "base_generation_status": "ok",
        "warning_count": len(warnings),
    }
    row.update(metrics)
    return row


def _quality_control(
    experiment_dir: Path,
    config: Mapping[str, Any],
    expected_frames: int,
    event_conversion: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    required_files = [
        "video.mp4",
        "preview.png",
        "ground_truth.csv",
        "acceleration_labels.csv",
        "metadata.json",
        "parameters.json",
        "event_conversion_status.json",
    ]
    missing = [
        name
        for name in required_files
        if not (experiment_dir / name).is_file()
        or (experiment_dir / name).stat().st_size == 0
    ]
    checks["required_base_files"] = {"passed": not missing, "missing": missing}

    ground_truth_rows = len(pd.read_csv(experiment_dir / "ground_truth.csv"))
    label_rows = len(pd.read_csv(experiment_dir / "acceleration_labels.csv"))
    checks["ground_truth_row_count"] = {
        "passed": ground_truth_rows == expected_frames and label_rows == expected_frames,
        "ground_truth_rows": ground_truth_rows,
        "acceleration_label_rows": label_rows,
        "expected_rows": expected_frames,
    }
    if bool(config["sampling"]["save_frames"]):
        saved_frames = sum(1 for _ in (experiment_dir / "frames").glob("frame_*.png"))
        checks["png_frame_count"] = {
            "passed": saved_frames == expected_frames,
            "saved_frames": saved_frames,
            "expected_frames": expected_frames,
        }
    else:
        checks["png_frame_count"] = {
            "passed": True,
            "skipped": True,
            "reason": "sampling.save_frames is false",
        }

    expected_plots = [
        "displacement.png",
        "velocity.png",
        "acceleration.png",
        "trajectory.png",
        "ground_truth_psd.png",
    ]
    missing_plots = [
        name
        for name in expected_plots
        if not (experiment_dir / "plots" / name).is_file()
        or (experiment_dir / "plots" / name).stat().st_size == 0
    ]
    checks["plots"] = {"passed": not missing_plots, "missing": missing_plots}

    if event_conversion["status"] == "completed":
        events_path = experiment_dir / "events.h5"
        readable_count = count_events(events_path)
        checks["events"] = {
            "passed": readable_count > 0
            and readable_count == int(event_conversion["event_count"]),
            "event_count": readable_count,
            "expected_event_count": event_conversion["event_count"],
        }
    else:
        checks["events"] = {
            "passed": not (experiment_dir / "events.h5").exists(),
            "optional": True,
            "status": event_conversion["status"],
            "reason": event_conversion["reason"],
        }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "overall_status": "passed" if passed else "failed",
        "events_optional_when_v2e_unavailable": True,
        "checks": checks,
    }


def _update_dataset_summary(output_root: Path, row: Mapping[str, Any]) -> Path:
    path = output_root / "dataset_summary.csv"
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True, sort=False)
    if "experiment_directory" in updated.columns:
        updated = updated.drop_duplicates(
            subset=["experiment_directory"], keep="last"
        )
    updated.to_csv(path, index=False)
    return path


def _write_environment(path: Path) -> None:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "h5py",
        "matplotlib",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    _write_json(
        path,
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "operating_system": os.name,
            "packages": packages,
        },
    )


def _write_manifest(experiment_dir: Path) -> None:
    rows = []
    for path in sorted(experiment_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest_sha256.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "relative_path": path.relative_to(experiment_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(experiment_dir / "manifest_sha256.csv", index=False)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
