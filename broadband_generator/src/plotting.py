from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_experiment_plots(exp_dir: Path, gt: pd.DataFrame, expected_freqs: List[float], fps: float) -> None:
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    t = gt["t_sec"].to_numpy()

    _line_plot(plots_dir / "displacement.png", t, gt[["x_px", "y_px"]], "Displacement", "px")
    _line_plot(plots_dir / "velocity.png", t, gt[["vx_px_s", "vy_px_s"]], "Velocity", "px/s")
    _line_plot(plots_dir / "acceleration.png", t, gt[["ax_px_s2", "ay_px_s2", "a_abs_px_s2"]], "Acceleration", "px/s^2")
    _trajectory_plot(plots_dir / "trajectory.png", gt)
    _psd_plot(plots_dir / "ground_truth_psd.png", gt["x_px"].to_numpy(), fps, expected_freqs)


def save_global_plots(dataset_root: Path, summary: pd.DataFrame) -> None:
    plots_dir = dataset_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ok = summary[summary["status"] == "ok"].copy()
    if len(ok) == 0:
        return

    _box_scatter(plots_dir / "amplitude_level_vs_event_rate.png", ok, "amplitude_level", "event_rate_hz", "Amplitude level", "Event rate [events/s]")
    _box_scatter(plots_dir / "amplitude_level_vs_peak_acceleration.png", ok, "amplitude_level", "peak_acceleration_px_s2", "Amplitude level", "Peak acceleration [px/s^2]")
    _box_scatter(plots_dir / "motion_type_vs_event_rate.png", ok, "motion_type", "event_rate_hz", "Motion type", "Event rate [events/s]", rotate=True)
    _frequency_distribution(plots_dir / "expected_frequency_distribution.png", ok)
    _contact_sheet(plots_dir / "dataset_contact_sheet.png", dataset_root, ok["experiment"].tolist())


def _line_plot(path: Path, t: np.ndarray, values: pd.DataFrame, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    stride = max(1, len(t) // 5000)
    for col in values.columns:
        ax.plot(t[::stride], values[col].to_numpy()[::stride], label=col, linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _trajectory_plot(path: Path, gt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(gt["x_px"], gt["y_px"], linewidth=0.8)
    ax.set_title("2D trajectory")
    ax.set_xlabel("x displacement [px]")
    ax.set_ylabel("y displacement [px]")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _psd_plot(path: Path, x: np.ndarray, fps: float, expected_freqs: List[float]) -> None:
    y = np.asarray(x, dtype=float)
    y = y - np.mean(y)
    win = np.hanning(len(y))
    spec = np.fft.rfft(y * win)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / float(fps))
    psd = np.abs(spec) ** 2
    mask = (freqs >= 0) & (freqs <= 2200)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(freqs[mask], 10 * np.log10(psd[mask] + 1e-12), linewidth=0.8)
    for f in expected_freqs[:8]:
        ax.axvline(float(f), color="tab:red", alpha=0.25, linewidth=0.8)
    ax.set_title("Ground-truth displacement PSD")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power [dB]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _box_scatter(path: Path, df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, rotate: bool = False) -> None:
    groups = [(name, pd.to_numeric(g[y_col], errors="coerce").dropna().to_numpy()) for name, g in df.groupby(x_col)]
    labels = [g[0] for g in groups]
    values = [g[1] for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if values:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        for idx, vals in enumerate(values, start=1):
            ax.scatter(np.full(len(vals), idx), vals, s=32, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if rotate:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _frequency_distribution(path: Path, df: pd.DataFrame) -> None:
    freqs = []
    for value in df["expected_top_frequencies_hz"]:
        try:
            freqs.extend(float(v) for v in json.loads(value))
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(freqs, bins=np.linspace(50, 2000, 31), color="tab:blue", alpha=0.8)
    ax.set_xlabel("Expected top frequency [Hz]")
    ax.set_ylabel("Count")
    ax.set_title("Expected frequency distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _contact_sheet(path: Path, dataset_root: Path, experiments: Iterable[str]) -> None:
    experiments = list(experiments)
    cols = 5
    rows = int(np.ceil(len(experiments) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, experiment in zip(axes, experiments):
        preview = dataset_root / "experiments" / experiment / "preview.png"
        if preview.exists():
            ax.imshow(plt.imread(preview), cmap="gray")
        ax.set_title(experiment, fontsize=8)
        ax.set_axis_off()
    for ax in axes[len(experiments):]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
