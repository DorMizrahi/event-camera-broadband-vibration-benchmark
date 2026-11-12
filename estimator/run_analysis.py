from __future__ import annotations

import argparse
from pathlib import Path

from event_vibration_standalone import (
    AnalysisConfig,
    ROI,
    auto_roi_from_activity,
    build_event_accumulation,
    run_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone event-camera vibration analysis without learned calibration."
    )
    parser.add_argument("input", type=Path, help=".h5/.hdf5/.csv/.txt/.dat event file")
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("measurement_outputs"))
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--active-axes", choices=["x", "y", "xy"], default="xy")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--auto-roi", action="store_true")
    parser.add_argument("--time-derivative-method", choices=["gradient", "savgol"], default="gradient")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    on, off, info = build_event_accumulation(args.input)
    if args.roi:
        roi = ROI(*args.roi).clipped(info.width or on.shape[1], info.height or on.shape[0])
    elif args.auto_roi:
        roi = auto_roi_from_activity(on, off)
    else:
        roi = ROI(0, 0, info.width or on.shape[1], info.height or on.shape[0])

    config = AnalysisConfig(
        output_root=str(args.output_root),
        active_axes=args.active_axes,
        time_derivative_method=args.time_derivative_method,
    )
    result = run_analysis(
        args.input,
        roi=roi,
        config=config,
        output_name=args.output_name,
        ground_truth_path=args.ground_truth,
        metadata_path=args.metadata,
        preview_images=(on, off),
    )
    print(f"Results saved to: {result.output_dir}")
    print(result.dominant_frequencies.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
