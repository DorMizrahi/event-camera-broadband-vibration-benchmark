from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_experiment_plots(
    experiment_dir: Path,
    ground_truth: pd.DataFrame,
    expected_frequencies_hz: Sequence[float],
    video_fps: float,
    config: Mapping[str, Any],
) -> list[str]:
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    time = ground_truth["t_sec"].to_numpy()
    paths = [
        _line_plot(
            plots_dir / "displacement.png",
            time,
            ground_truth[["x_px", "y_px"]],
            "Displacement",
            "px",
        ),
        _line_plot(
            plots_dir / "velocity.png",
            time,
            ground_truth[["vx_px_s", "vy_px_s"]],
            "Velocity",
            "px/s",
        ),
        _line_plot(
            plots_dir / "acceleration.png",
            time,
            ground_truth[["ax_px_s2", "ay_px_s2", "a_abs_px_s2"]],
            "Acceleration",
            "px/s^2",
        ),
        _trajectory_plot(plots_dir / "trajectory.png", ground_truth),
        _psd_plot(
            plots_dir / "ground_truth_psd.png",
            ground_truth["x_px"].to_numpy(),
            float(video_fps),
            expected_frequencies_hz,
            float(config["derived"]["maximum_active_frequency_hz"]),
        ),
    ]
    if config["calibration"].get("meters_per_pixel") is not None:
        paths.append(
            _line_plot(
                plots_dir / "acceleration_si.png",
                time,
                ground_truth[["ax_m_s2", "ay_m_s2", "a_abs_m_s2"]],
                "Acceleration (SI)",
                "m/s^2",
            )
        )
    return [str(path) for path in paths]


def _line_plot(
    path: Path,
    time: np.ndarray,
    values: pd.DataFrame,
    title: str,
    ylabel: str,
) -> Path:
    fig, axis = plt.subplots(figsize=(8, 3.6))
    stride = max(1, len(time) // 5000)
    for column in values.columns:
        axis.plot(
            time[::stride],
            values[column].to_numpy()[::stride],
            label=column,
            linewidth=0.9,
        )
    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _trajectory_plot(path: Path, ground_truth: pd.DataFrame) -> Path:
    fig, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.plot(ground_truth["x_px"], ground_truth["y_px"], linewidth=0.8)
    axis.set_title("2D trajectory")
    axis.set_xlabel("x displacement [px]")
    axis.set_ylabel("y displacement [px]")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _psd_plot(
    path: Path,
    signal: np.ndarray,
    fps: float,
    expected_frequencies_hz: Sequence[float],
    maximum_active_frequency_hz: float,
) -> Path:
    centered = np.asarray(signal, dtype=float) - float(np.mean(signal))
    window = np.hanning(len(centered))
    spectrum = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / fps)
    power = np.abs(spectrum) ** 2
    upper = min(fps / 2.0, max(1.1 * maximum_active_frequency_hz, 100.0))
    mask = (frequencies >= 0.0) & (frequencies <= upper)
    fig, axis = plt.subplots(figsize=(8, 3.8))
    axis.plot(frequencies[mask], 10.0 * np.log10(power[mask] + 1e-12), linewidth=0.8)
    for frequency in list(expected_frequencies_hz)[:8]:
        axis.axvline(float(frequency), color="tab:red", alpha=0.25, linewidth=0.8)
    axis.set_title("Ground-truth displacement PSD")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Power [dB]")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
