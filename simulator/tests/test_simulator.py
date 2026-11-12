from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from event_vibration_simulator import default_parameters, generate_experiment
from event_vibration_simulator.config import SCENARIO_TYPES, resolve_parameters
from event_vibration_simulator.motion import create_motion


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MotionTests(unittest.TestCase):
    def test_all_scenarios_are_finite_and_normalized(self) -> None:
        for scenario in SCENARIO_TYPES:
            with self.subTest(scenario=scenario):
                parameters = default_parameters()
                parameters["motion"]["type"] = scenario
                parameters["sampling"]["duration_sec"] = 0.01
                resolved, _ = resolve_parameters(parameters, PROJECT_ROOT)
                time = np.arange(resolved["derived"]["frame_count"]) / resolved[
                    "sampling"
                ]["video_fps"]
                _, motion, metrics = create_motion(resolved, time)
                for key in (
                    "x_px",
                    "y_px",
                    "vx_px_s",
                    "vy_px_s",
                    "ax_px_s2",
                    "ay_px_s2",
                ):
                    self.assertTrue(np.all(np.isfinite(motion[key])), key)
                self.assertAlmostEqual(metrics["peak_displacement_px"], 6.0, places=10)

    def test_si_columns_require_a_scale(self) -> None:
        parameters = default_parameters()
        parameters["motion"]["normalization_mode"] = "peak_acceleration_m_s2"
        with self.assertRaisesRegex(ValueError, "meters_per_pixel"):
            resolve_parameters(parameters, PROJECT_ROOT)

    def test_nyquist_is_enforced(self) -> None:
        parameters = default_parameters()
        parameters["sampling"]["video_fps"] = 900.0
        parameters["motion"]["single_sine"]["frequency_hz"] = 500.0
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            resolve_parameters(parameters, PROJECT_ROOT)


class GenerationTests(unittest.TestCase):
    def test_complete_base_experiment_without_v2e(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parameters = default_parameters()
            parameters["experiment"].update(
                {
                    "name": "smoke_without_v2e",
                    "output_root": temporary_directory,
                    "output_policy": "error",
                    "write_manifest": True,
                }
            )
            parameters["sampling"].update(
                {
                    "video_fps": 1000.0,
                    "duration_sec": 0.02,
                    "width_px": 64,
                    "height_px": 48,
                    "recommended_samples_per_cycle": 20.0,
                }
            )
            parameters["rendering"].update(
                {
                    "base_position_x_px": 32.0,
                    "base_position_y_px": 24.0,
                    "marker_radius_px": 3.0,
                }
            )
            parameters["motion"].update(
                {
                    "normalization_value": 2.0,
                    "frequency_min_hz": 10.0,
                    "frequency_max_hz": 50.0,
                }
            )
            parameters["motion"]["single_sine"]["frequency_hz"] = 50.0

            with patch(
                "event_vibration_simulator.v2e_runner.locate_v2e", return_value=None
            ):
                result = generate_experiment(
                    parameters, project_root=PROJECT_ROOT, progress=lambda _: None
                )

            experiment_dir = Path(result["experiment_directory"])
            self.assertEqual(result["status"], "completed_without_events")
            self.assertEqual(result["event_conversion_status"], "skipped_not_installed")
            self.assertFalse((experiment_dir / "events.h5").exists())
            self.assertTrue((experiment_dir / "video.mp4").is_file())
            self.assertEqual(
                len(list((experiment_dir / "frames").glob("frame_*.png"))), 20
            )
            self.assertEqual(result["quality_control"], "passed")
            event_status = json.loads(
                (experiment_dir / "event_conversion_status.json").read_text()
            )
            self.assertFalse(event_status["fallback_used"])


class NotebookTests(unittest.TestCase):
    def test_notebook_is_a_wrapper_around_shared_engine(self) -> None:
        notebook_path = (
            PROJECT_ROOT
            / "notebooks"
            / "configurable_single_experiment_generator.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("generate_experiment(USER_PARAMETERS", code)
        self.assertNotIn("def create_motion", code)
        self.assertNotIn("def _evaluate", code)


if __name__ == "__main__":
    unittest.main()
