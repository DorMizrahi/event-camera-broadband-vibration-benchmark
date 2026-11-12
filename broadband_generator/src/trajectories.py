from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


FREQ_MIN_HZ = 50.0
FREQ_MAX_HZ = 2000.0


@dataclass
class MotionSpec:
    motion_type: str
    active_frequencies_hz: List[float]
    amplitudes_x_px: List[float]
    amplitudes_y_px: List[float]
    phases_x_rad: List[float]
    phases_y_rad: List[float]
    spectral_envelope: str
    two_axis_motion: bool
    chirp_kind: str = ""
    chirp_start_hz: float = FREQ_MIN_HZ
    chirp_end_hz: float = FREQ_MAX_HZ


def make_motion_spec(
    motion_type: str,
    target_peak_displacement_px: float,
    rng: np.random.Generator,
    two_axis_motion: bool = False,
    num_components: int | None = None,
    spectral_variant: str = "flat",
) -> MotionSpec:
    if motion_type == "sparse_multi_sine":
        n = int(num_components or rng.integers(5, 11))
        freqs = _sample_separated_frequencies(rng, n)
        weights = rng.uniform(0.6, 1.3, n)
        envelope = "sparse_flat"
    elif motion_type == "dense_multi_sine":
        n = int(num_components or rng.integers(20, 51))
        freqs = np.sort(rng.uniform(FREQ_MIN_HZ, FREQ_MAX_HZ, n))
        weights = rng.lognormal(mean=0.0, sigma=0.45, size=n)
        envelope = "dense_lognormal_flat"
    elif motion_type == "band_limited_random_phase":
        n = int(num_components or rng.integers(60, 101))
        freqs = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, n)
        if spectral_variant == "low_tilt":
            weights = 1.0 / np.sqrt(freqs)
            envelope = "1_over_sqrt_f"
        elif spectral_variant == "mid_bump":
            weights = np.exp(-0.5 * ((freqs - 800.0) / 450.0) ** 2)
            envelope = "mid_band_gaussian"
        else:
            weights = np.ones(n)
            envelope = "flat_random_phase"
    elif motion_type == "mixed_bands":
        low_n = max(2, int((num_components or 12) // 2))
        high_n = max(2, int(num_components or 12) - low_n)
        low = rng.uniform(50.0, 200.0, low_n)
        high = rng.uniform(1000.0, 2000.0, high_n)
        freqs = np.sort(np.concatenate([low, high]))
        weights = np.concatenate([
            rng.uniform(0.8, 1.2, low_n),
            rng.uniform(0.6, 1.0, high_n),
        ])
        envelope = "low_50_200_plus_high_1000_2000"
    elif motion_type == "chirp":
        chirp_kind = spectral_variant if spectral_variant in {"linear", "log"} else "linear"
        return MotionSpec(
            motion_type=motion_type,
            active_frequencies_hz=[FREQ_MIN_HZ, FREQ_MAX_HZ],
            amplitudes_x_px=[float(target_peak_displacement_px)],
            amplitudes_y_px=[float(target_peak_displacement_px * (0.25 if two_axis_motion else 0.0))],
            phases_x_rad=[float(rng.uniform(0.0, 2.0 * math.pi))],
            phases_y_rad=[float(rng.uniform(0.0, 2.0 * math.pi))],
            spectral_envelope=f"{chirp_kind}_chirp",
            two_axis_motion=two_axis_motion,
            chirp_kind=chirp_kind,
            chirp_start_hz=FREQ_MIN_HZ,
            chirp_end_hz=FREQ_MAX_HZ,
        )
    else:
        raise ValueError(f"Unsupported motion_type: {motion_type}")

    phases_x = rng.uniform(0.0, 2.0 * math.pi, len(freqs))
    phases_y = rng.uniform(0.0, 2.0 * math.pi, len(freqs))
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(np.sqrt(np.sum(weights**2))), 1e-12)
    amplitudes_x = target_peak_displacement_px * weights
    y_scale = rng.uniform(0.18, 0.38) if two_axis_motion else 0.0
    amplitudes_y = target_peak_displacement_px * y_scale * weights

    return MotionSpec(
        motion_type=motion_type,
        active_frequencies_hz=[float(f) for f in freqs],
        amplitudes_x_px=[float(a) for a in amplitudes_x],
        amplitudes_y_px=[float(a) for a in amplitudes_y],
        phases_x_rad=[float(p) for p in phases_x],
        phases_y_rad=[float(p) for p in phases_y],
        spectral_envelope=envelope,
        two_axis_motion=two_axis_motion,
    )


def evaluate_motion(spec: MotionSpec, t_sec: np.ndarray, target_peak_displacement_px: float) -> Dict[str, np.ndarray | float]:
    if spec.motion_type == "chirp":
        raw = _evaluate_chirp(spec, t_sec)
    else:
        raw = _evaluate_multi_sine(spec, t_sec)

    peak = float(np.nanmax(np.sqrt(raw["x_px"] ** 2 + raw["y_px"] ** 2)))
    scale = target_peak_displacement_px / max(peak, 1e-12)
    out = {key: np.asarray(value, dtype=float) * scale for key, value in raw.items()}
    out["a_abs_px_s2"] = np.sqrt(out["ax_px_s2"] ** 2 + out["ay_px_s2"] ** 2)
    out["peak_displacement_px"] = float(np.nanmax(np.sqrt(out["x_px"] ** 2 + out["y_px"] ** 2)))
    out["peak_velocity_px_s"] = float(np.nanmax(np.sqrt(out["vx_px_s"] ** 2 + out["vy_px_s"] ** 2)))
    out["peak_acceleration_px_s2"] = float(np.nanmax(out["a_abs_px_s2"]))
    out["total_rms_displacement_px"] = float(np.sqrt(np.mean(out["x_px"] ** 2 + out["y_px"] ** 2)))
    out["total_rms_velocity_px_s"] = float(np.sqrt(np.mean(out["vx_px_s"] ** 2 + out["vy_px_s"] ** 2)))
    out["total_rms_acceleration_px_s2"] = float(np.sqrt(np.mean(out["a_abs_px_s2"] ** 2)))
    return out


def expected_top_frequencies(spec: MotionSpec, k: int = 5) -> List[float]:
    if spec.motion_type == "chirp":
        return [50.0, 500.0, 1000.0, 1500.0, 2000.0][:k]
    amps = np.sqrt(np.asarray(spec.amplitudes_x_px) ** 2 + np.asarray(spec.amplitudes_y_px) ** 2)
    order = np.argsort(amps)[::-1]
    return [float(spec.active_frequencies_hz[i]) for i in order[:k]]


def _evaluate_multi_sine(spec: MotionSpec, t: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.zeros_like(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    vx = np.zeros_like(t, dtype=float)
    vy = np.zeros_like(t, dtype=float)
    ax = np.zeros_like(t, dtype=float)
    ay = np.zeros_like(t, dtype=float)

    for freq, ax_amp, ay_amp, phase_x, phase_y in zip(
        spec.active_frequencies_hz,
        spec.amplitudes_x_px,
        spec.amplitudes_y_px,
        spec.phases_x_rad,
        spec.phases_y_rad,
    ):
        omega = 2.0 * math.pi * float(freq)
        sx = np.sin(omega * t + phase_x)
        cx = np.cos(omega * t + phase_x)
        sy = np.sin(omega * t + phase_y)
        cy = np.cos(omega * t + phase_y)
        x += ax_amp * sx
        y += ay_amp * sy
        vx += ax_amp * omega * cx
        vy += ay_amp * omega * cy
        ax += -ax_amp * omega * omega * sx
        ay += -ay_amp * omega * omega * sy

    return {
        "x_px": x,
        "y_px": y,
        "vx_px_s": vx,
        "vy_px_s": vy,
        "ax_px_s2": ax,
        "ay_px_s2": ay,
    }


def _evaluate_chirp(spec: MotionSpec, t: np.ndarray) -> Dict[str, np.ndarray]:
    duration = max(float(t[-1] - t[0]), 1e-12)
    f0 = float(spec.chirp_start_hz)
    f1 = float(spec.chirp_end_hz)
    phase0_x = float(spec.phases_x_rad[0])
    phase0_y = float(spec.phases_y_rad[0])
    amp_x = float(spec.amplitudes_x_px[0])
    amp_y = float(spec.amplitudes_y_px[0])

    if spec.chirp_kind == "log":
        ratio = max(f1 / f0, 1.0001)
        log_ratio = math.log(ratio)
        phase_cycles = f0 * duration / log_ratio * (np.exp(t / duration * log_ratio) - 1.0)
        inst_freq = f0 * np.exp(t / duration * log_ratio)
        inst_slope = inst_freq * log_ratio / duration
    else:
        slope = (f1 - f0) / duration
        phase_cycles = f0 * t + 0.5 * slope * t * t
        inst_freq = f0 + slope * t
        inst_slope = np.full_like(t, slope)

    phase_x = 2.0 * math.pi * phase_cycles + phase0_x
    phase_y = 2.0 * math.pi * phase_cycles + phase0_y
    omega = 2.0 * math.pi * inst_freq
    omega_dot = 2.0 * math.pi * inst_slope

    x = amp_x * np.sin(phase_x)
    y = amp_y * np.sin(phase_y)
    vx = amp_x * omega * np.cos(phase_x)
    vy = amp_y * omega * np.cos(phase_y)
    ax = amp_x * (omega_dot * np.cos(phase_x) - omega * omega * np.sin(phase_x))
    ay = amp_y * (omega_dot * np.cos(phase_y) - omega * omega * np.sin(phase_y))
    return {
        "x_px": x,
        "y_px": y,
        "vx_px_s": vx,
        "vy_px_s": vy,
        "ax_px_s2": ax,
        "ay_px_s2": ay,
    }


def _sample_separated_frequencies(rng: np.random.Generator, n: int, min_sep_hz: float = 35.0) -> np.ndarray:
    values: List[float] = []
    attempts = 0
    while len(values) < n and attempts < 2000:
        attempts += 1
        f = float(rng.uniform(FREQ_MIN_HZ, FREQ_MAX_HZ))
        if all(abs(f - old) >= min_sep_hz for old in values):
            values.append(f)
    while len(values) < n:
        values.append(float(rng.uniform(FREQ_MIN_HZ, FREQ_MAX_HZ)))
    return np.array(sorted(values), dtype=float)


def frequency_bounds(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmin(arr)), float(np.nanmax(arr))
