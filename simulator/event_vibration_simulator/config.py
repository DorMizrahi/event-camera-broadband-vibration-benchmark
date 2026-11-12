from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


SCENARIO_TYPES = (
    "single_sine",
    "sparse_multi_sine",
    "dense_multi_sine",
    "band_limited_random_phase",
    "chirp",
    "mixed_bands",
)

NORMALIZATION_MODES = (
    "peak_displacement_px",
    "rms_displacement_px",
    "peak_velocity_px_s",
    "rms_velocity_px_s",
    "peak_acceleration_px_s2",
    "rms_acceleration_px_s2",
    "peak_acceleration_m_s2",
    "rms_acceleration_m_s2",
)


def default_parameters() -> dict[str, Any]:
    """Return a fresh, fully populated single-experiment configuration."""
    return {
        "experiment": {
            "name": "single_sine_500hz_default",
            "output_root": "generated_experiments",
            "output_policy": "new",
            "random_seed": 7301,
            "write_manifest": True,
        },
        "sampling": {
            "video_fps": 40000.0,
            "duration_mode": "seconds",
            "duration_sec": 0.2,
            "number_of_cycles": 40.0,
            "cycle_reference_frequency_hz": 500.0,
            "width_px": 240,
            "height_px": 180,
            "save_frames": True,
            "recommended_samples_per_cycle": 20.0,
        },
        "rendering": {
            "object_shape": "circle",
            "marker_radius_px": 5.0,
            "rectangle_width_px": 10.0,
            "rectangle_height_px": 10.0,
            "base_position_x_px": 120.0,
            "base_position_y_px": 90.0,
            "background_intensity": 5,
            "marker_contrast": 0.72,
            "video_codec": "libx264",
            "video_crf": 28,
            "video_preset": "veryfast",
        },
        "calibration": {
            "meters_per_pixel": None,
            "standard_gravity_m_s2": 9.80665,
        },
        "motion": {
            "type": "single_sine",
            "frequency_min_hz": 50.0,
            "frequency_max_hz": 2000.0,
            "normalization_mode": "peak_displacement_px",
            "normalization_value": 6.0,
            "two_axis_motion": False,
            "y_amplitude_ratio": 0.25,
            "custom_frequencies_hz": [],
            "custom_weights": [],
            "custom_phases_x_rad": [],
            "custom_phases_y_rad": [],
            "single_sine": {
                "frequency_hz": 500.0,
                "phase_x_rad": 0.0,
                "phase_y_rad": 1.5707963267948966,
            },
            "sparse_multi_sine": {
                "num_components": 8,
                "minimum_frequency_separation_hz": 35.0,
                "weight_min": 0.6,
                "weight_max": 1.3,
            },
            "dense_multi_sine": {
                "num_components": 32,
                "weight_lognormal_mean": 0.0,
                "weight_lognormal_sigma": 0.45,
            },
            "band_limited_random_phase": {
                "num_components": 80,
                "spectral_variant": "flat",
                "mid_bump_center_hz": 800.0,
                "mid_bump_sigma_hz": 450.0,
            },
            "chirp": {
                "start_frequency_hz": 50.0,
                "end_frequency_hz": 2000.0,
                "kind": "linear",
                "phase_x_rad": 0.0,
                "phase_y_rad": 1.5707963267948966,
            },
            "mixed_bands": {
                "bands_hz": [[50.0, 200.0], [1000.0, 2000.0]],
                "components_per_band": [6, 6],
                "weight_ranges": [[0.8, 1.2], [0.6, 1.0]],
            },
        },
        "v2e": {
            "enabled": True,
            "command": "",
            "script_path": "",
            "python_executable": "",
            "disable_slomo": True,
            "pos_thres": 0.2,
            "neg_thres": 0.2,
            "sigma_thres": 0.03,
            "refractory_period_sec": 0.0001,
            "shot_noise_rate_hz": 0.0,
            "cutoff_hz": 0.0,
            "dvs_emulator_seed": None,
            "timeout_sec": 0,
        },
    }


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_parameter_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Parameter files must use JSON format.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The top level of a parameter file must be a JSON object.")
    return data


def resolve_parameters(
    user_parameters: Mapping[str, Any] | None,
    project_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    """Merge defaults, calculate derived values, and validate the experiment."""
    config = deep_merge(default_parameters(), user_parameters or {})
    project_root = Path(project_root).resolve()
    warnings: list[str] = []

    experiment = config["experiment"]
    sampling = config["sampling"]
    rendering = config["rendering"]
    calibration = config["calibration"]
    motion = config["motion"]
    v2e = config["v2e"]

    name = str(experiment["name"]).strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(
            "experiment.name must contain only letters, digits, '.', '_' or '-'."
        )
    experiment["name"] = name
    policy = str(experiment["output_policy"]).lower()
    if policy not in {"new", "error", "overwrite"}:
        raise ValueError("experiment.output_policy must be new, error, or overwrite.")
    experiment["output_policy"] = policy
    experiment["random_seed"] = int(experiment["random_seed"])

    output_root = Path(str(experiment["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    experiment["resolved_output_root"] = str(output_root.resolve())

    fps = _positive_float(sampling["video_fps"], "sampling.video_fps")
    duration_mode = str(sampling["duration_mode"]).lower()
    if duration_mode == "seconds":
        duration = _positive_float(sampling["duration_sec"], "sampling.duration_sec")
    elif duration_mode == "cycles":
        cycles = _positive_float(
            sampling["number_of_cycles"], "sampling.number_of_cycles"
        )
        reference_hz = float(sampling["cycle_reference_frequency_hz"])
        if reference_hz <= 0 and str(motion["type"]) == "single_sine":
            reference_hz = float(motion["single_sine"]["frequency_hz"])
        reference_hz = _positive_float(
            reference_hz, "sampling.cycle_reference_frequency_hz"
        )
        duration = cycles / reference_hz
    else:
        raise ValueError("sampling.duration_mode must be seconds or cycles.")
    frame_count = int(math.ceil(fps * duration))
    if frame_count < 2:
        raise ValueError("The selected sampling settings produce fewer than two frames.")
    sampling["video_fps"] = fps
    sampling["duration_mode"] = duration_mode
    sampling["resolved_duration_sec"] = duration
    sampling["frame_count"] = frame_count

    width = int(sampling["width_px"])
    height = int(sampling["height_px"])
    if width < 16 or height < 16:
        raise ValueError("sampling width and height must each be at least 16 pixels.")
    sampling["width_px"] = width
    sampling["height_px"] = height

    shape = str(rendering["object_shape"]).lower()
    if shape not in {"circle", "rectangle", "square"}:
        raise ValueError("rendering.object_shape must be circle, rectangle, or square.")
    rendering["object_shape"] = shape
    rendering["marker_radius_px"] = _positive_float(
        rendering["marker_radius_px"], "rendering.marker_radius_px"
    )
    rendering["rectangle_width_px"] = _positive_float(
        rendering["rectangle_width_px"], "rendering.rectangle_width_px"
    )
    rendering["rectangle_height_px"] = _positive_float(
        rendering["rectangle_height_px"], "rendering.rectangle_height_px"
    )
    contrast = float(rendering["marker_contrast"])
    if not 0.0 < contrast <= 1.0:
        raise ValueError("rendering.marker_contrast must be in (0, 1].")
    background = int(rendering["background_intensity"])
    if not 0 <= background <= 254:
        raise ValueError("rendering.background_intensity must be in [0, 254].")
    rendering["background_intensity"] = background

    meters_per_pixel = calibration.get("meters_per_pixel")
    if meters_per_pixel in {None, ""}:
        calibration["meters_per_pixel"] = None
    else:
        calibration["meters_per_pixel"] = _positive_float(
            meters_per_pixel, "calibration.meters_per_pixel"
        )
    calibration["standard_gravity_m_s2"] = _positive_float(
        calibration["standard_gravity_m_s2"],
        "calibration.standard_gravity_m_s2",
    )

    scenario = str(motion["type"])
    if scenario not in SCENARIO_TYPES:
        raise ValueError(
            f"motion.type must be one of: {', '.join(SCENARIO_TYPES)}."
        )
    motion["type"] = scenario
    f_min = _positive_float(motion["frequency_min_hz"], "motion.frequency_min_hz")
    f_max = _positive_float(motion["frequency_max_hz"], "motion.frequency_max_hz")
    if f_max <= f_min:
        raise ValueError("motion.frequency_max_hz must be greater than frequency_min_hz.")
    motion["frequency_min_hz"] = f_min
    motion["frequency_max_hz"] = f_max

    normalization_mode = str(motion["normalization_mode"])
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"motion.normalization_mode must be one of: {', '.join(NORMALIZATION_MODES)}."
        )
    if normalization_mode.endswith("_m_s2") and calibration["meters_per_pixel"] is None:
        raise ValueError(
            "A physical acceleration normalization requires calibration.meters_per_pixel."
        )
    motion["normalization_value"] = _positive_float(
        motion["normalization_value"], "motion.normalization_value"
    )
    ratio = float(motion["y_amplitude_ratio"])
    if ratio < 0:
        raise ValueError("motion.y_amplitude_ratio cannot be negative.")
    motion["y_amplitude_ratio"] = ratio

    max_active_frequency = _maximum_active_frequency(config)
    if fps <= 2.0 * max_active_frequency:
        raise ValueError(
            f"Nyquist violation: video_fps={fps:g} must exceed twice the maximum "
            f"active frequency ({max_active_frequency:g} Hz)."
        )
    samples_per_cycle = fps / max_active_frequency
    recommended = _positive_float(
        sampling["recommended_samples_per_cycle"],
        "sampling.recommended_samples_per_cycle",
    )
    if samples_per_cycle + 1e-12 < recommended:
        warnings.append(
            f"Only {samples_per_cycle:.3f} samples/highest-frequency cycle; "
            f"{recommended:g} is recommended."
        )

    for key in ("pos_thres", "neg_thres", "sigma_thres"):
        v2e[key] = _nonnegative_float(v2e[key], f"v2e.{key}")
    for key in ("refractory_period_sec", "shot_noise_rate_hz", "cutoff_hz"):
        v2e[key] = _nonnegative_float(v2e[key], f"v2e.{key}")
    emulator_seed = v2e.get("dvs_emulator_seed")
    if emulator_seed in {None, ""}:
        emulator_seed = int(experiment["random_seed"])
    emulator_seed = int(emulator_seed)
    if emulator_seed <= 0:
        raise ValueError("v2e.dvs_emulator_seed must be a positive integer or null.")
    v2e["resolved_dvs_emulator_seed"] = emulator_seed
    v2e["timeout_sec"] = int(v2e["timeout_sec"])
    if v2e["timeout_sec"] < 0:
        raise ValueError("v2e.timeout_sec cannot be negative.")

    config["derived"] = {
        "project_root": str(project_root),
        "duration_sec": duration,
        "frame_count": frame_count,
        "maximum_active_frequency_hz": max_active_frequency,
        "samples_per_highest_frequency_cycle": samples_per_cycle,
        "frame_period_sec": 1.0 / fps,
        "timestamp_unit": "seconds",
        "displacement_unit": "pixels",
        "velocity_unit": "pixels/second",
        "acceleration_unit": "pixels/second^2",
    }
    return config, warnings


def _maximum_active_frequency(config: Mapping[str, Any]) -> float:
    motion = config["motion"]
    scenario = motion["type"]
    if scenario == "single_sine":
        return _positive_float(
            motion["single_sine"]["frequency_hz"],
            "motion.single_sine.frequency_hz",
        )
    if scenario == "chirp":
        chirp = motion["chirp"]
        return max(
            _positive_float(chirp["start_frequency_hz"], "motion.chirp.start_frequency_hz"),
            _positive_float(chirp["end_frequency_hz"], "motion.chirp.end_frequency_hz"),
        )
    custom = [float(value) for value in motion.get("custom_frequencies_hz", [])]
    if custom:
        if min(custom) <= 0:
            raise ValueError("motion.custom_frequencies_hz must contain positive values.")
        return max(custom)
    if scenario == "mixed_bands":
        bands = motion["mixed_bands"]["bands_hz"]
        if not bands:
            raise ValueError("motion.mixed_bands.bands_hz cannot be empty.")
        return max(float(pair[1]) for pair in bands)
    return float(motion["frequency_max_hz"])


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return value
