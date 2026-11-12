from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_vibration_simulator import deep_merge, generate_experiment, load_parameter_file
from event_vibration_simulator.config import resolve_parameters


PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# USER PARAMETERS: edit this block, then run this file.
# One run always creates exactly one experiment.
# =============================================================================
USER_PARAMETERS = {
    "experiment": {
        "name": "single_sine_500hz_default",
        "output_root": "generated_experiments",
        "output_policy": "new",  # new | error | overwrite
        "random_seed": 7301,
        "write_manifest": True,
    },
    "sampling": {
        "video_fps": 40000.0,
        "duration_mode": "seconds",  # seconds | cycles
        "duration_sec": 0.2,
        "number_of_cycles": 40.0,
        "cycle_reference_frequency_hz": 500.0,
        "width_px": 240,
        "height_px": 180,
        "save_frames": True,
        "recommended_samples_per_cycle": 20.0,
    },
    "rendering": {
        "object_shape": "circle",  # circle | rectangle | square
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
        # Leave null/None for px, px/s, px/s^2 only. Set a scale to add SI columns.
        "meters_per_pixel": None,
        "standard_gravity_m_s2": 9.80665,
    },
    "motion": {
        # single_sine | sparse_multi_sine | dense_multi_sine |
        # band_limited_random_phase | chirp | mixed_bands
        "type": "single_sine",
        "frequency_min_hz": 50.0,
        "frequency_max_hz": 2000.0,
        # peak/rms displacement, velocity, or acceleration; see README.
        "normalization_mode": "peak_displacement_px",
        "normalization_value": 6.0,
        "two_axis_motion": False,
        "y_amplitude_ratio": 0.25,
        # Non-empty custom lists override generated frequencies/weights/phases.
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
            "spectral_variant": "flat",  # flat | low_tilt | mid_bump
            "mid_bump_center_hz": 800.0,
            "mid_bump_sigma_hz": 450.0,
        },
        "chirp": {
            "start_frequency_hz": 50.0,
            "end_frequency_hz": 2000.0,
            "kind": "linear",  # linear | log
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
        # Usually leave these blank for auto-detection. They can point to a
        # v2e executable, v2e.py, and the Python executable of its environment.
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
        # None uses experiment.random_seed, making V2E noise reproducible.
        "dvs_emulator_seed": None,
        "timeout_sec": 0,
    },
}
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one standalone synthetic event-camera vibration experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file whose values override USER_PARAMETERS.",
    )
    parser.add_argument("--name", help="Optional experiment name override.")
    parser.add_argument("--output-root", type=Path, help="Optional output root override.")
    parser.add_argument(
        "--no-v2e",
        action="store_true",
        help="Generate all base artifacts and intentionally skip V2E.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print resolved parameters without generating files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parameters = USER_PARAMETERS
    if args.config:
        parameters = deep_merge(parameters, load_parameter_file(args.config))
    cli_override: dict[str, object] = {}
    if args.name:
        cli_override.setdefault("experiment", {})["name"] = args.name
    if args.output_root:
        cli_override.setdefault("experiment", {})["output_root"] = str(args.output_root)
    if args.no_v2e:
        cli_override.setdefault("v2e", {})["enabled"] = False
    parameters = deep_merge(parameters, cli_override)

    if args.validate_only:
        resolved, warnings = resolve_parameters(parameters, PROJECT_ROOT)
        print(json.dumps({"parameters": resolved, "warnings": warnings}, indent=2))
        return 0

    result = generate_experiment(parameters, project_root=PROJECT_ROOT)
    print("\nRESULT")
    print(json.dumps(result, indent=2))
    if result["events_path"] is None:
        print(
            "\nNo events.h5 was created. See event_conversion_status.json and "
            "run_v2e_later scripts in the experiment directory."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
