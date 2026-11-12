from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
LEGACY_RUNTIME_DEPS = WORKSPACE_ROOT / "synthetic_event_vibration_dataset" / ".runtime_deps"
if LEGACY_RUNTIME_DEPS.exists():
    sys.path.insert(0, str(LEGACY_RUNTIME_DEPS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_builder import build_dataset  # noqa: E402


def main() -> None:
    result = build_dataset(PROJECT_ROOT)
    summary = result["summary"]
    ok = summary[summary["status"] == "ok"]
    failed = summary[summary["status"] != "ok"]
    total_events = int(ok["total_events"].sum()) if len(ok) else 0
    print(f"dataset_root={result['dataset_root']}")
    print(f"zip_path={result['zip_path']}")
    print(f"successful_experiments={len(ok)}")
    print(f"failed_experiments={len(failed)}")
    print(f"total_events={total_events}")
    print(f"zip_size_bytes={result['zip_size_bytes']}")
    if len(ok):
        grouped = ok.groupby("amplitude_level")["event_rate_hz"].agg(["min", "mean", "max"]).reset_index()
        print(grouped.to_string(index=False))
    if len(failed):
        print(failed[["experiment", "failure_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
