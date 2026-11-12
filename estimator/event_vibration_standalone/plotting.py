from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .core import KinematicTriplet
from .roi import ROI, total_activity_image


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_event_accumulation(
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    roi: ROI,
    path: Path,
    dpi: int,
) -> None:
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(total_activity_image(on_counts, off_counts), origin="upper", aspect="equal")
    ax.add_patch(
        Rectangle(
            (roi.x_min, roi.y_min),
            roi.width,
            roi.height,
            fill=False,
            linewidth=2,
        )
    )
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title("Event accumulation and selected ROI")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_primary_signal(
    binned: pd.DataFrame,
    source_signal: str,
    axis: str,
    path: Path,
    dpi: int,
) -> None:
    values = binned[source_signal].to_numpy(dtype=float)
    values = values - np.nanmean(values)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(binned["t_sec"], values, linewidth=0.9)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Relative {axis}-position [px]")
    ax.set_title(f"Primary event-derived motion signal: {source_signal}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_spectrum_and_peaks(
    vector_periodogram: pd.DataFrame,
    peaks: pd.DataFrame,
    config: AnalysisConfig,
    path: Path,
    dpi: int,
) -> None:
    frequencies = vector_periodogram["frequency_hz"].to_numpy(dtype=float)
    power = vector_periodogram["periodogram_psd"].to_numpy(dtype=float)
    mask = (frequencies >= 0) & (frequencies <= config.f_max_hz + 0.1 * config.f_max_hz)
    normalized = power / np.nanmax(power) if np.nanmax(power) > 0 else power

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(frequencies[mask], normalized[mask], linewidth=1.1, label="Primary/vector periodogram")
    ax.axvspan(config.f_min_hz, config.f_max_hz, alpha=0.10, label="Analysis band")
    if not peaks.empty:
        x = peaks["discrete_frequency_hz"].to_numpy(dtype=float)
        y = np.interp(x, frequencies, normalized)
        ax.scatter(x, y, s=40, zorder=3, label="Selected peaks")
        for refined, discrete, yy in zip(peaks["frequency_hz"], x, y):
            ax.annotate(
                f"{refined:.1f}",
                xy=(discrete, yy),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized spectral power")
    ax.set_title("Dominant-frequency estimation")
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_kinematic_method_comparison(
    time: np.ndarray,
    frequency_result: KinematicTriplet,
    time_result: KinematicTriplet,
    axis: str,
    quantity: str,
    path: Path,
    dpi: int,
    ground_truth: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(time, getattr(frequency_result, quantity), linewidth=1.2, label="Frequency-domain differentiation")
    ax.plot(time, getattr(time_result, quantity), linewidth=1.0, alpha=0.85, label="Time-domain differentiation")
    if ground_truth is not None:
        ax.plot(time, ground_truth, linewidth=1.1, linestyle="--", label="Ground truth")
    units = {
        "displacement": "px",
        "velocity": "px/s",
        "acceleration": "px/s²",
    }[quantity]
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{axis.upper()} {quantity} [{units}]")
    ax.set_title(f"{axis.upper()}-axis {quantity}: frequency versus time differentiation")
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_vector_acceleration_comparison(
    time: np.ndarray,
    frequency_vector: np.ndarray,
    time_vector: np.ndarray,
    path: Path,
    dpi: int,
    ground_truth: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(time, frequency_vector, linewidth=1.2, label="Frequency-domain differentiation")
    ax.plot(time, time_vector, linewidth=1.0, alpha=0.85, label="Time-domain differentiation")
    if ground_truth is not None:
        ax.plot(time, ground_truth, linewidth=1.1, linestyle="--", label="Ground truth")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vector acceleration [px/s²]")
    ax.set_title("Vector acceleration: frequency versus time differentiation")
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_chirp_ridge(
    times: np.ndarray,
    frequencies: np.ndarray,
    power: np.ndarray,
    ridge: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    if power.size == 0 or ridge.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    display_power = 10.0 * np.log10(power + 1e-30)
    mesh = ax.pcolormesh(times, frequencies, display_power, shading="auto")
    ax.plot(ridge["time_sec"], ridge["estimated_ridge_hz"], linewidth=1.5, label="Estimated ridge")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Chirp time-frequency ridge")
    ax.legend(framealpha=0.92)
    fig.colorbar(mesh, ax=ax, label="Power [dB]")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_method_metric_summary(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    if metrics.empty:
        return
    subset = metrics[
        (metrics["quantity"] == "acceleration")
        & metrics["axis"].isin(["x", "y", "vector"])
    ].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"{row.axis}" for row in subset.itertuples()]
    values = subset["nrmse_by_reference_rms"].to_numpy(dtype=float)
    ax.bar(labels, values)
    ax.set_ylabel("NRMSE relative to frequency-domain result")
    ax.set_title("Time-domain versus frequency-domain acceleration")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path, dpi)
