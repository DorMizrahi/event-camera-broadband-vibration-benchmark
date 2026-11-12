"""Run after extracting the frozen estimator and minimal benchmark. See report.md."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("package", type=Path)
parser.add_argument("dataset", type=Path)
parser.add_argument("result", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.package.resolve()))
from event_vibration_standalone import AnalysisConfig, auto_roi_from_activity, build_event_accumulation, run_analysis

for folder in sorted((args.dataset / "experiments").glob("experiment_*")):
    metadata = json.loads((folder / "metadata.json").read_text())
    events = folder / "events.h5"
    on, off, _ = build_event_accumulation(events)
    config = AnalysisConfig(active_axes=metadata["active_axes"], chirp_tolerance_hz=25,
                            output_root=str(args.result), save_plots=False)
    result = run_analysis(events, roi=auto_roi_from_activity(on, off), config=config,
                          output_name=folder.name, metadata_path=folder / "metadata.json",
                          ground_truth_path=folder / "ground_truth.csv")
    for axis in metadata["active_axes"]:
        assert result.source_signals[axis] == axis + "_on_centroid"
        assert np.array_equal(result.kinematics[axis + "_displacement_frequency_domain"],
                              result.kinematics[axis + "_displacement_time_domain"])
    print(folder.name, flush=True)
