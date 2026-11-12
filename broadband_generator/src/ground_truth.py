from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def write_ground_truth(path: Path, t_sec, motion: Dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "t_sec": t_sec,
            "x_px": motion["x_px"],
            "y_px": motion["y_px"],
            "vx_px_s": motion["vx_px_s"],
            "vy_px_s": motion["vy_px_s"],
            "ax_px_s2": motion["ax_px_s2"],
            "ay_px_s2": motion["ay_px_s2"],
            "a_abs_px_s2": motion["a_abs_px_s2"],
        }
    )
    df.to_csv(path, index=False)
    return df


def write_acceleration_labels(path: Path, gt: pd.DataFrame) -> pd.DataFrame:
    labels = gt[["t_sec", "ax_px_s2", "ay_px_s2", "a_abs_px_s2"]].copy()
    labels.to_csv(path, index=False)
    return labels
