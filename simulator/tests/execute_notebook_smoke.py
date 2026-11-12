from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "configurable_single_experiment_generator.ipynb"
)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_root = Path(temporary_directory) / "notebook_output"
        override = {
            "experiment": {
                "name": "notebook_execution_smoke",
                "output_root": str(output_root),
                "output_policy": "error",
                "write_manifest": False,
            },
            "sampling": {
                "video_fps": 1000.0,
                "duration_sec": 0.02,
                "width_px": 64,
                "height_px": 48,
                "recommended_samples_per_cycle": 20.0,
            },
            "rendering": {
                "base_position_x_px": 32.0,
                "base_position_y_px": 24.0,
                "marker_radius_px": 3.0,
            },
            "motion": {
                "type": "single_sine",
                "frequency_min_hz": 10.0,
                "frequency_max_hz": 50.0,
                "normalization_value": 2.0,
                "single_sine": {"frequency_hz": 50.0},
            },
            "v2e": {"enabled": False},
        }
        os.environ["EVENT_VIBRATION_CONFIG_JSON"] = json.dumps(override)
        os.environ["MPLCONFIGDIR"] = str(Path(temporary_directory) / "matplotlib")
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            allow_errors=False,
        )
        client.execute(cwd=str(PROJECT_ROOT))

        experiment_dir = output_root / "notebook_execution_smoke"
        summary = pd.read_csv(experiment_dir / "experiment_summary.csv")
        status = json.loads(
            (experiment_dir / "generation_status.json").read_text(encoding="utf-8")
        )
        frame_count = len(list((experiment_dir / "frames").glob("frame_*.png")))
        assert len(summary) == 1
        assert status["status"] == "completed_without_events"
        assert frame_count == 20
        assert not (experiment_dir / "events.h5").exists()
        print("NOTEBOOK_SMOKE_TEST=PASS")
        print(f"executed_cells={len(notebook.cells)}")
        print(f"frame_count={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
