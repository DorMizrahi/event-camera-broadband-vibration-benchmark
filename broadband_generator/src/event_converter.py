from __future__ import annotations

import importlib.util
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


def v2e_available() -> bool:
    return shutil.which("v2e") is not None or importlib.util.find_spec("v2e") is not None


def emulate_events_h5(
    events_path: Path,
    x_px: np.ndarray,
    y_px: np.ndarray,
    t_sec: np.ndarray,
    resolution: Tuple[int, int],
    marker_radius_px: float,
    marker_contrast: float,
    base_position: Tuple[float, float],
    random_seed: int,
    noise_rate_hz: float = 0.0,
    refractory_us: int = 0,
) -> Dict[str, float | int | str]:
    width, height = resolution
    rng = np.random.default_rng(random_seed)
    bg = 5.0
    marker = bg + marker_contrast * (255.0 - bg)
    log_delta = math.log1p(marker) - math.log1p(bg)
    event_keep_probability = float(np.clip((log_delta - 0.12) / 1.5, 0.10, 1.0))

    chunks = []
    total = 0
    last_event_time = {} if refractory_us > 0 else None

    prev_x = base_position[0] + float(x_px[0])
    prev_y = base_position[1] + float(y_px[0])
    for i in range(1, len(t_sec)):
        cx = base_position[0] + float(x_px[i])
        cy = base_position[1] + float(y_px[i])
        t_us = int(round(float(t_sec[i]) * 1e6))
        block = _edge_events_between_positions(
            prev_x,
            prev_y,
            cx,
            cy,
            t_us,
            width,
            height,
            marker_radius_px,
            event_keep_probability,
            rng,
            last_event_time,
            refractory_us,
        )
        if len(block):
            chunks.append(block)
            total += len(block)
        if noise_rate_hz > 0:
            noise = _noise_events(noise_rate_hz, t_us, width, height, rng, dt_sec=float(t_sec[i] - t_sec[i - 1]))
            if len(noise):
                chunks.append(noise)
                total += len(noise)
        prev_x, prev_y = cx, cy

    if chunks:
        events = np.concatenate(chunks, axis=0)
        order = np.lexsort((events[:, 1], events[:, 0]))
        events = events[order].astype(np.int64, copy=False)
    else:
        events = np.zeros((0, 4), dtype=np.int64)

    with h5py.File(events_path, "w") as h5:
        h5.create_dataset("events", data=events, compression="gzip", compression_opts=4)
        h5.attrs["columns"] = "[t_us, x, y, p]"
        h5.attrs["polarity"] = "1=ON, 0=OFF"
        h5.attrs["event_conversion_method"] = "analytic_marker_edge_event_emulator"

    duration = float(t_sec[-1] - t_sec[0]) if len(t_sec) > 1 else 0.0
    return {
        "event_conversion_method": "analytic_marker_edge_event_emulator",
        "total_events": int(len(events)),
        "event_rate_hz": float(len(events) / max(duration, 1e-12)),
        "v2e_available": bool(v2e_available()),
    }


def _edge_events_between_positions(
    prev_x: float,
    prev_y: float,
    curr_x: float,
    curr_y: float,
    t_us: int,
    width: int,
    height: int,
    radius: float,
    keep_probability: float,
    rng: np.random.Generator,
    last_event_time: dict | None,
    refractory_us: int,
) -> np.ndarray:
    r = float(radius)
    x0 = max(0, int(math.floor(min(prev_x, curr_x) - r - 2)))
    x1 = min(width, int(math.ceil(max(prev_x, curr_x) + r + 3)))
    y0 = max(0, int(math.floor(min(prev_y, curr_y) - r - 2)))
    y1 = min(height, int(math.ceil(max(prev_y, curr_y) + r + 3)))
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 4), dtype=np.int64)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    prev_inside = (xx - prev_x) ** 2 + (yy - prev_y) ** 2 <= r * r
    curr_inside = (xx - curr_x) ** 2 + (yy - curr_y) ** 2 <= r * r
    on_y, on_x = np.where(curr_inside & ~prev_inside)
    off_y, off_x = np.where(prev_inside & ~curr_inside)

    if keep_probability < 1.0:
        on_keep = rng.random(len(on_x)) <= keep_probability
        off_keep = rng.random(len(off_x)) <= keep_probability
        on_x, on_y = on_x[on_keep], on_y[on_keep]
        off_x, off_y = off_x[off_keep], off_y[off_keep]

    rows = []
    if len(on_x):
        xs = on_x + x0
        ys = on_y + y0
        rows.append(_pack_events(t_us, xs, ys, 1, last_event_time, refractory_us))
    if len(off_x):
        xs = off_x + x0
        ys = off_y + y0
        rows.append(_pack_events(t_us, xs, ys, 0, last_event_time, refractory_us))
    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    rows = [r for r in rows if len(r)]
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 4), dtype=np.int64)


def _pack_events(
    t_us: int,
    xs: np.ndarray,
    ys: np.ndarray,
    polarity: int,
    last_event_time: dict | None,
    refractory_us: int,
) -> np.ndarray:
    if last_event_time is not None and refractory_us > 0:
        keep = []
        for x, y in zip(xs, ys):
            key = (int(x), int(y), int(polarity))
            previous = last_event_time.get(key, -10**18)
            ok = (t_us - previous) >= refractory_us
            keep.append(ok)
            if ok:
                last_event_time[key] = t_us
        keep_arr = np.asarray(keep, dtype=bool)
        xs = xs[keep_arr]
        ys = ys[keep_arr]
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    out = np.empty((len(xs), 4), dtype=np.int64)
    out[:, 0] = t_us
    out[:, 1] = xs.astype(np.int64)
    out[:, 2] = ys.astype(np.int64)
    out[:, 3] = int(polarity)
    return out


def _noise_events(noise_rate_hz: float, t_us: int, width: int, height: int, rng: np.random.Generator, dt_sec: float) -> np.ndarray:
    expected = max(float(noise_rate_hz), 0.0) * max(float(dt_sec), 0.0)
    n = int(rng.poisson(expected))
    if n <= 0:
        return np.zeros((0, 4), dtype=np.int64)
    out = np.empty((n, 4), dtype=np.int64)
    out[:, 0] = t_us
    out[:, 1] = rng.integers(0, width, n)
    out[:, 2] = rng.integers(0, height, n)
    out[:, 3] = rng.integers(0, 2, n)
    return out
