from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class MotionSpec:
    motion_type: str
    active_frequencies_hz: list[float]
    amplitudes_x_px: list[float]
    amplitudes_y_px: list[float]
    phases_x_rad: list[float]
    phases_y_rad: list[float]
    spectral_envelope: str
    two_axis_motion: bool
    chirp_kind: str = ""
    chirp_start_hz: float = 0.0
    chirp_end_hz: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_motion(
    config: Mapping[str, Any],
    t_sec: np.ndarray,
) -> tuple[MotionSpec, dict[str, Any], dict[str, float]]:
    """Create, evaluate, and normalize one deterministic motion trajectory."""
    motion_config = config["motion"]
    rng = np.random.default_rng(int(config["experiment"]["random_seed"]))
    seed_amplitude = _seed_amplitude(motion_config)
    spec = _make_motion_spec(motion_config, seed_amplitude, rng)
    raw = _evaluate(spec, np.asarray(t_sec, dtype=float))
    scale = _normalization_scale(raw, config)
    motion = {name: values * scale for name, values in raw.items()}
    motion["a_abs_px_s2"] = np.hypot(motion["ax_px_s2"], motion["ay_px_s2"])
    metrics = calculate_metrics(motion, config)
    motion["normalization_scale"] = float(scale)
    return spec, motion, metrics


def calculate_metrics(
    motion: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, float]:
    displacement = np.hypot(motion["x_px"], motion["y_px"])
    velocity = np.hypot(motion["vx_px_s"], motion["vy_px_s"])
    acceleration = np.hypot(motion["ax_px_s2"], motion["ay_px_s2"])
    metrics = {
        "peak_displacement_px": float(np.max(displacement)),
        "rms_displacement_px": float(np.sqrt(np.mean(displacement**2))),
        "peak_velocity_px_s": float(np.max(velocity)),
        "rms_velocity_px_s": float(np.sqrt(np.mean(velocity**2))),
        "peak_acceleration_px_s2": float(np.max(acceleration)),
        "rms_acceleration_px_s2": float(np.sqrt(np.mean(acceleration**2))),
        "acceleration_p95_px_s2": float(np.percentile(acceleration, 95)),
    }
    meters_per_pixel = config["calibration"].get("meters_per_pixel")
    if meters_per_pixel is not None:
        scale = float(meters_per_pixel)
        gravity = float(config["calibration"]["standard_gravity_m_s2"])
        metrics.update(
            {
                "peak_displacement_m": metrics["peak_displacement_px"] * scale,
                "rms_displacement_m": metrics["rms_displacement_px"] * scale,
                "peak_velocity_m_s": metrics["peak_velocity_px_s"] * scale,
                "rms_velocity_m_s": metrics["rms_velocity_px_s"] * scale,
                "peak_acceleration_m_s2": metrics["peak_acceleration_px_s2"] * scale,
                "rms_acceleration_m_s2": metrics["rms_acceleration_px_s2"] * scale,
                "acceleration_p95_m_s2": metrics["acceleration_p95_px_s2"] * scale,
                "peak_acceleration_g": metrics["peak_acceleration_px_s2"] * scale / gravity,
                "rms_acceleration_g": metrics["rms_acceleration_px_s2"] * scale / gravity,
            }
        )
    return metrics


def expected_top_frequencies(spec: MotionSpec, k: int = 5) -> list[float]:
    if spec.motion_type == "chirp":
        start = float(spec.chirp_start_hz)
        end = float(spec.chirp_end_hz)
        return [float(value) for value in np.linspace(start, end, k)]
    amplitudes = np.hypot(spec.amplitudes_x_px, spec.amplitudes_y_px)
    order = np.argsort(amplitudes)[::-1]
    return [float(spec.active_frequencies_hz[index]) for index in order[:k]]


def cycle_information(spec: MotionSpec, duration_sec: float) -> dict[str, Any]:
    duration = float(duration_sec)
    if spec.motion_type != "chirp":
        counts = [float(frequency * duration) for frequency in spec.active_frequencies_hz]
        return {
            "definition": "frequency_hz * requested_duration_sec",
            "cycle_counts_by_frequency": [
                {"frequency_hz": float(frequency), "cycles": count}
                for frequency, count in zip(spec.active_frequencies_hz, counts)
            ],
            "minimum_cycle_count": float(min(counts)),
            "maximum_cycle_count": float(max(counts)),
        }

    f0 = float(spec.chirp_start_hz)
    f1 = float(spec.chirp_end_hz)
    if spec.chirp_kind == "log":
        ratio = f1 / f0
        cycles = f0 * duration * (ratio - 1.0) / math.log(ratio)
    else:
        cycles = 0.5 * (f0 + f1) * duration
    return {
        "definition": "integral of instantaneous chirp frequency over requested duration",
        "chirp_total_cycles": float(cycles),
        "minimum_cycle_count": float(cycles),
        "maximum_cycle_count": float(cycles),
    }


def _seed_amplitude(motion_config: Mapping[str, Any]) -> float:
    if motion_config["normalization_mode"] in {
        "peak_displacement_px",
        "rms_displacement_px",
    }:
        return float(motion_config["normalization_value"])
    return 1.0


def _make_motion_spec(
    motion: Mapping[str, Any],
    seed_amplitude: float,
    rng: np.random.Generator,
) -> MotionSpec:
    scenario = str(motion["type"])
    f_min = float(motion["frequency_min_hz"])
    f_max = float(motion["frequency_max_hz"])
    two_axis = bool(motion["two_axis_motion"])
    y_ratio = float(motion["y_amplitude_ratio"]) if two_axis else 0.0
    custom_frequencies = np.asarray(motion.get("custom_frequencies_hz", []), dtype=float)

    if scenario == "single_sine":
        section = motion["single_sine"]
        frequency = float(section["frequency_hz"])
        return MotionSpec(
            motion_type=scenario,
            active_frequencies_hz=[frequency],
            amplitudes_x_px=[seed_amplitude],
            amplitudes_y_px=[seed_amplitude * y_ratio],
            phases_x_rad=[float(section["phase_x_rad"])],
            phases_y_rad=[float(section["phase_y_rad"])],
            spectral_envelope="single_frequency",
            two_axis_motion=two_axis,
        )

    if scenario == "chirp":
        section = motion["chirp"]
        kind = str(section["kind"]).lower()
        if kind not in {"linear", "log"}:
            raise ValueError("motion.chirp.kind must be linear or log.")
        start = float(section["start_frequency_hz"])
        end = float(section["end_frequency_hz"])
        if start <= 0 or end <= 0 or math.isclose(start, end):
            raise ValueError("Chirp start/end frequencies must be positive and different.")
        return MotionSpec(
            motion_type=scenario,
            active_frequencies_hz=[start, end],
            amplitudes_x_px=[seed_amplitude],
            amplitudes_y_px=[seed_amplitude * y_ratio],
            phases_x_rad=[float(section["phase_x_rad"])],
            phases_y_rad=[float(section["phase_y_rad"])],
            spectral_envelope=f"{kind}_chirp",
            two_axis_motion=two_axis,
            chirp_kind=kind,
            chirp_start_hz=start,
            chirp_end_hz=end,
        )

    if scenario == "sparse_multi_sine":
        section = motion[scenario]
        n = int(section["num_components"])
        frequencies = (
            np.sort(custom_frequencies)
            if custom_frequencies.size
            else _sample_separated_frequencies(
                rng,
                n,
                f_min,
                f_max,
                float(section["minimum_frequency_separation_hz"]),
            )
        )
        weights = _custom_or_generated_weights(
            motion,
            len(frequencies),
            lambda: rng.uniform(
                float(section["weight_min"]),
                float(section["weight_max"]),
                len(frequencies),
            ),
        )
        envelope = "sparse_flat"
    elif scenario == "dense_multi_sine":
        section = motion[scenario]
        n = int(section["num_components"])
        frequencies = (
            np.sort(custom_frequencies)
            if custom_frequencies.size
            else np.sort(rng.uniform(f_min, f_max, n))
        )
        weights = _custom_or_generated_weights(
            motion,
            len(frequencies),
            lambda: rng.lognormal(
                mean=float(section["weight_lognormal_mean"]),
                sigma=float(section["weight_lognormal_sigma"]),
                size=len(frequencies),
            ),
        )
        envelope = "dense_lognormal_flat"
    elif scenario == "band_limited_random_phase":
        section = motion[scenario]
        n = int(section["num_components"])
        frequencies = (
            np.sort(custom_frequencies)
            if custom_frequencies.size
            else np.linspace(f_min, f_max, n)
        )
        variant = str(section["spectral_variant"])
        if motion.get("custom_weights"):
            weights = _custom_or_generated_weights(
                motion, len(frequencies), lambda: np.ones(len(frequencies))
            )
            envelope = "custom"
        elif variant == "low_tilt":
            weights = 1.0 / np.sqrt(frequencies)
            envelope = "1_over_sqrt_f"
        elif variant == "mid_bump":
            center = float(section["mid_bump_center_hz"])
            sigma = float(section["mid_bump_sigma_hz"])
            weights = np.exp(-0.5 * ((frequencies - center) / sigma) ** 2)
            envelope = "mid_band_gaussian"
        elif variant == "flat":
            weights = np.ones(len(frequencies))
            envelope = "flat_random_phase"
        else:
            raise ValueError(
                "band_limited_random_phase.spectral_variant must be flat, low_tilt, or mid_bump."
            )
    elif scenario == "mixed_bands":
        section = motion[scenario]
        if custom_frequencies.size:
            frequencies = np.sort(custom_frequencies)
            weights = _custom_or_generated_weights(
                motion, len(frequencies), lambda: np.ones(len(frequencies))
            )
        else:
            bands = section["bands_hz"]
            counts = section["components_per_band"]
            ranges = section["weight_ranges"]
            if not (len(bands) == len(counts) == len(ranges)):
                raise ValueError(
                    "mixed_bands bands_hz, components_per_band, and weight_ranges must have equal lengths."
                )
            pairs: list[tuple[float, float]] = []
            for band, count, weight_range in zip(bands, counts, ranges):
                low, high = map(float, band)
                if low <= 0 or high <= low:
                    raise ValueError("Each mixed band must be [positive_low, larger_high].")
                count = int(count)
                if count < 1:
                    raise ValueError("Each mixed band must contain at least one component.")
                band_frequencies = rng.uniform(low, high, count)
                band_weights = rng.uniform(
                    float(weight_range[0]), float(weight_range[1]), count
                )
                pairs.extend(zip(band_frequencies.tolist(), band_weights.tolist()))
            pairs.sort(key=lambda pair: pair[0])
            frequencies = np.asarray([pair[0] for pair in pairs], dtype=float)
            weights = np.asarray([pair[1] for pair in pairs], dtype=float)
        envelope = "mixed_configured_bands"
    else:
        raise ValueError(f"Unsupported motion type: {scenario}")

    if len(frequencies) < 1:
        raise ValueError("At least one active frequency is required.")
    if np.any(frequencies <= 0):
        raise ValueError("All active frequencies must be positive.")
    weights = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Motion weights must be finite, non-negative, and not all zero.")
    weights = weights / max(float(np.sqrt(np.sum(weights**2))), 1e-12)
    amplitudes_x = seed_amplitude * weights
    amplitudes_y = seed_amplitude * y_ratio * weights
    phases_x = _phases(motion.get("custom_phases_x_rad", []), len(frequencies), rng)
    phases_y = _phases(motion.get("custom_phases_y_rad", []), len(frequencies), rng)

    return MotionSpec(
        motion_type=scenario,
        active_frequencies_hz=[float(value) for value in frequencies],
        amplitudes_x_px=[float(value) for value in amplitudes_x],
        amplitudes_y_px=[float(value) for value in amplitudes_y],
        phases_x_rad=[float(value) for value in phases_x],
        phases_y_rad=[float(value) for value in phases_y],
        spectral_envelope=envelope,
        two_axis_motion=two_axis,
    )


def _evaluate(spec: MotionSpec, t_sec: np.ndarray) -> dict[str, np.ndarray]:
    if spec.motion_type == "chirp":
        return _evaluate_chirp(spec, t_sec)

    x = np.zeros_like(t_sec)
    y = np.zeros_like(t_sec)
    vx = np.zeros_like(t_sec)
    vy = np.zeros_like(t_sec)
    ax = np.zeros_like(t_sec)
    ay = np.zeros_like(t_sec)
    for frequency, amp_x, amp_y, phase_x, phase_y in zip(
        spec.active_frequencies_hz,
        spec.amplitudes_x_px,
        spec.amplitudes_y_px,
        spec.phases_x_rad,
        spec.phases_y_rad,
    ):
        omega = 2.0 * math.pi * float(frequency)
        sin_x = np.sin(omega * t_sec + phase_x)
        cos_x = np.cos(omega * t_sec + phase_x)
        sin_y = np.sin(omega * t_sec + phase_y)
        cos_y = np.cos(omega * t_sec + phase_y)
        x += amp_x * sin_x
        y += amp_y * sin_y
        vx += amp_x * omega * cos_x
        vy += amp_y * omega * cos_y
        ax += -amp_x * omega**2 * sin_x
        ay += -amp_y * omega**2 * sin_y
    return {
        "x_px": x,
        "y_px": y,
        "vx_px_s": vx,
        "vy_px_s": vy,
        "ax_px_s2": ax,
        "ay_px_s2": ay,
    }


def _evaluate_chirp(spec: MotionSpec, t_sec: np.ndarray) -> dict[str, np.ndarray]:
    duration = max(float(t_sec[-1] - t_sec[0]), 1e-12)
    f0 = float(spec.chirp_start_hz)
    f1 = float(spec.chirp_end_hz)
    if spec.chirp_kind == "log":
        ratio = f1 / f0
        if ratio <= 0 or math.isclose(ratio, 1.0):
            raise ValueError("A logarithmic chirp requires unequal frequencies with a positive ratio.")
        log_ratio = math.log(ratio)
        phase_cycles = f0 * duration / log_ratio * (
            np.exp(t_sec / duration * log_ratio) - 1.0
        )
        instantaneous_frequency = f0 * np.exp(t_sec / duration * log_ratio)
        instantaneous_slope = instantaneous_frequency * log_ratio / duration
    else:
        slope = (f1 - f0) / duration
        phase_cycles = f0 * t_sec + 0.5 * slope * t_sec**2
        instantaneous_frequency = f0 + slope * t_sec
        instantaneous_slope = np.full_like(t_sec, slope)

    omega = 2.0 * math.pi * instantaneous_frequency
    omega_dot = 2.0 * math.pi * instantaneous_slope
    phase_x = 2.0 * math.pi * phase_cycles + spec.phases_x_rad[0]
    phase_y = 2.0 * math.pi * phase_cycles + spec.phases_y_rad[0]
    amp_x = spec.amplitudes_x_px[0]
    amp_y = spec.amplitudes_y_px[0]
    return {
        "x_px": amp_x * np.sin(phase_x),
        "y_px": amp_y * np.sin(phase_y),
        "vx_px_s": amp_x * omega * np.cos(phase_x),
        "vy_px_s": amp_y * omega * np.cos(phase_y),
        "ax_px_s2": amp_x
        * (omega_dot * np.cos(phase_x) - omega**2 * np.sin(phase_x)),
        "ay_px_s2": amp_y
        * (omega_dot * np.cos(phase_y) - omega**2 * np.sin(phase_y)),
    }


def _normalization_scale(
    raw: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> float:
    displacement = np.hypot(raw["x_px"], raw["y_px"])
    velocity = np.hypot(raw["vx_px_s"], raw["vy_px_s"])
    acceleration = np.hypot(raw["ax_px_s2"], raw["ay_px_s2"])
    metrics = {
        "peak_displacement_px": float(np.max(displacement)),
        "rms_displacement_px": float(np.sqrt(np.mean(displacement**2))),
        "peak_velocity_px_s": float(np.max(velocity)),
        "rms_velocity_px_s": float(np.sqrt(np.mean(velocity**2))),
        "peak_acceleration_px_s2": float(np.max(acceleration)),
        "rms_acceleration_px_s2": float(np.sqrt(np.mean(acceleration**2))),
    }
    meters_per_pixel = config["calibration"].get("meters_per_pixel")
    if meters_per_pixel is not None:
        metrics["peak_acceleration_m_s2"] = (
            metrics["peak_acceleration_px_s2"] * float(meters_per_pixel)
        )
        metrics["rms_acceleration_m_s2"] = (
            metrics["rms_acceleration_px_s2"] * float(meters_per_pixel)
        )
    mode = str(config["motion"]["normalization_mode"])
    target = float(config["motion"]["normalization_value"])
    source = float(metrics[mode])
    if source <= 1e-15:
        raise ValueError(f"Cannot normalize {mode}; the unscaled value is zero.")
    return target / source


def _sample_separated_frequencies(
    rng: np.random.Generator,
    count: int,
    minimum_hz: float,
    maximum_hz: float,
    minimum_separation_hz: float,
) -> np.ndarray:
    if count < 1:
        raise ValueError("num_components must be at least one.")
    if minimum_separation_hz < 0:
        raise ValueError("minimum_frequency_separation_hz cannot be negative.")
    if count > 1 and (count - 1) * minimum_separation_hz > maximum_hz - minimum_hz:
        raise ValueError("The requested sparse frequencies cannot fit with that separation.")
    values: list[float] = []
    attempts = 0
    while len(values) < count and attempts < 100000:
        attempts += 1
        candidate = float(rng.uniform(minimum_hz, maximum_hz))
        if all(abs(candidate - existing) >= minimum_separation_hz for existing in values):
            values.append(candidate)
    if len(values) != count:
        raise RuntimeError("Could not sample separated sparse frequencies; adjust the range or seed.")
    return np.asarray(sorted(values), dtype=float)


def _custom_or_generated_weights(
    motion: Mapping[str, Any],
    count: int,
    generator,
) -> np.ndarray:
    custom = np.asarray(motion.get("custom_weights", []), dtype=float)
    if custom.size:
        if len(custom) != count:
            raise ValueError("motion.custom_weights length must match the number of frequencies.")
        return custom
    return np.asarray(generator(), dtype=float)


def _phases(
    custom_values: list[float],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    custom = np.asarray(custom_values, dtype=float)
    if custom.size:
        if len(custom) != count:
            raise ValueError("Custom phase list length must match the number of frequencies.")
        return custom
    return rng.uniform(0.0, 2.0 * math.pi, count)
