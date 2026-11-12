from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LEGACY_RUNTIME_DEPS = WORKSPACE_ROOT / "synthetic_event_vibration_dataset" / ".runtime_deps"
if LEGACY_RUNTIME_DEPS.exists():
    sys.path.insert(0, str(LEGACY_RUNTIME_DEPS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import h5py  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    summary_path = PROJECT_ROOT / "dataset_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    rows = []
    for _, row in summary.iterrows():
        exp = row["experiment"]
        exp_dir = PROJECT_ROOT / "experiments" / exp
        events_path = exp_dir / "events.h5"
        event_count = 0
        if events_path.exists():
            with h5py.File(events_path, "r") as h5:
                event_count = int(h5["events"].shape[0]) if "events" in h5 else 0
        png_ok = True
        for name in ["displacement.png", "velocity.png", "acceleration.png", "ground_truth_psd.png", "trajectory.png"]:
            try:
                imageio.imread(exp_dir / "plots" / name)
            except Exception:
                png_ok = False
        rows.append({
            "experiment": exp,
            "status": row.get("status", ""),
            "events_h5": events_path.exists(),
            "event_count": event_count,
            "ground_truth": (exp_dir / "ground_truth.csv").exists(),
            "acceleration_labels": (exp_dir / "acceleration_labels.csv").exists(),
            "metadata": (exp_dir / "metadata.json").exists(),
            "png_plots_open": png_ok,
        })
    checks = pd.DataFrame(rows)
    checks.to_csv(PROJECT_ROOT / "quality_control_report.csv", index=False)
    print(checks.to_string(index=False))
    print()
    print(summary.groupby("amplitude_level")["event_rate_hz"].agg(["min", "mean", "max"]).to_string())


if __name__ == "__main__":
    main()
