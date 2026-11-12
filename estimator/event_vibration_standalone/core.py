from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import find_peaks, savgol_filter, spectrogram, welch

from .config import AnalysisConfig


def robust_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def safe_savgol(values: np.ndarray, window: int, polyorder: int, *, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    if len(values) < 5:
        result = values.copy()
        for _ in range(deriv):
            result = np.gradient(result, delta)
        return result
    window = int(window)
    if window % 2 == 0:
        window += 1
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window = min(window, maximum)
    if window <= polyorder or window < 3:
        result = values.copy()
        for _ in range(deriv):
            result = np.gradient(result, delta)
        return result
    return savgol_filter(
        values,
        window_length=window,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        mode="interp",
    )


class EventBinner:
    """Convert asynchronous events to fixed-width, polarity-aware motion signals."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    @property
    def bin_sec(self) -> float:
        return self.config.bin_sec

    @staticmethod
    def _quantile_midpoint(
        indices: np.ndarray,
        coordinates: np.ndarray,
        n_bins: int,
        coordinate_size: int,
        low_quantile: float = 0.10,
        high_quantile: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        integer_coordinates = np.clip(
            np.rint(coordinates).astype(int),
            0,
            coordinate_size - 1,
        )
        histogram = np.zeros((n_bins, coordinate_size), dtype=np.uint32)
        np.add.at(histogram, (indices, integer_coordinates), 1)
        cumulative = np.cumsum(histogram, axis=1)
        totals = cumulative[:, -1]

        low_targets = np.maximum(np.ceil(low_quantile * totals).astype(np.uint32), 1)
        high_targets = np.maximum(np.ceil(high_quantile * totals).astype(np.uint32), 1)
        low_values = (cumulative >= low_targets[:, None]).argmax(axis=1).astype(float)
        high_values = (cumulative >= high_targets[:, None]).argmax(axis=1).astype(float)
        missing = totals == 0
        low_values[missing] = np.nan
        high_values[missing] = np.nan
        midpoint = 0.5 * (low_values + high_values)
        return low_values, high_values, midpoint

    def bin(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            raise ValueError("The event table is empty.")
        required = {"t_sec", "x", "y", "p"}
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"Missing event columns for binning: {sorted(missing)}")

        time = events["t_sec"].to_numpy(dtype=float)
        n_bins = int(np.floor(time.max() / self.bin_sec)) + 1
        if n_bins < 4:
            raise ValueError("The recording is too short for the selected temporal bin.")
        indices = np.floor(time / self.bin_sec).astype(int)
        indices = np.clip(indices, 0, n_bins - 1)

        x = events["x"].to_numpy(dtype=float)
        y = events["y"].to_numpy(dtype=float)
        on_mask = events["p"].to_numpy() > 0
        off_mask = ~on_mask
        on_weights = on_mask.astype(float)
        off_weights = off_mask.astype(float)
        signed_weights = np.where(on_mask, 1.0, -1.0)

        counts = np.bincount(indices, minlength=n_bins).astype(float)
        positive = np.bincount(indices, weights=on_weights, minlength=n_bins)
        negative = np.bincount(indices, weights=off_weights, minlength=n_bins)
        polarity_sum = np.bincount(indices, weights=signed_weights, minlength=n_bins)

        x_sum = np.bincount(indices, weights=x, minlength=n_bins)
        y_sum = np.bincount(indices, weights=y, minlength=n_bins)
        x_on_sum = np.bincount(indices, weights=x * on_weights, minlength=n_bins)
        x_off_sum = np.bincount(indices, weights=x * off_weights, minlength=n_bins)
        y_on_sum = np.bincount(indices, weights=y * on_weights, minlength=n_bins)
        y_off_sum = np.bincount(indices, weights=y * off_weights, minlength=n_bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            x_centroid = np.where(counts > 0, x_sum / counts, np.nan)
            y_centroid = np.where(counts > 0, y_sum / counts, np.nan)
            x_on_centroid = np.where(positive > 0, x_on_sum / positive, np.nan)
            x_off_centroid = np.where(negative > 0, x_off_sum / negative, np.nan)
            y_on_centroid = np.where(positive > 0, y_on_sum / positive, np.nan)
            y_off_centroid = np.where(negative > 0, y_off_sum / negative, np.nan)

        sensor_width = max(1, int(np.nanmax(x)) + 1)
        sensor_height = max(1, int(np.nanmax(y)) + 1)
        x_q10, x_q90, x_robust_center = self._quantile_midpoint(
            indices, x, n_bins, sensor_width
        )
        y_q10, y_q90, y_robust_center = self._quantile_midpoint(
            indices, y, n_bins, sensor_height
        )

        output = pd.DataFrame(
            {
                "t_sec": (np.arange(n_bins) + 0.5) * self.bin_sec,
                "event_count": counts,
                "positive_count": positive,
                "negative_count": negative,
                "polarity_sum": polarity_sum,
                "x_centroid": x_centroid,
                "y_centroid": y_centroid,
                "x_on_centroid": x_on_centroid,
                "x_off_centroid": x_off_centroid,
                "y_on_centroid": y_on_centroid,
                "y_off_centroid": y_off_centroid,
                "x_polarity_difference": x_on_centroid - x_off_centroid,
                "y_polarity_difference": y_on_centroid - y_off_centroid,
                "x_on_off_midpoint": 0.5 * (x_on_centroid + x_off_centroid),
                "y_on_off_midpoint": 0.5 * (y_on_centroid + y_off_centroid),
                "x_q10": x_q10,
                "x_q90": x_q90,
                "x_robust_center": x_robust_center,
                "y_q10": y_q10,
                "y_q90": y_q90,
                "y_robust_center": y_robust_center,
            }
        )

        observed_columns = [
            "x_centroid",
            "y_centroid",
            "x_on_centroid",
            "x_off_centroid",
            "y_on_centroid",
            "y_off_centroid",
            "x_robust_center",
            "y_robust_center",
        ]
        for column in observed_columns:
            output[f"{column}_observed"] = np.isfinite(output[column]).astype(np.uint8)

        interpolation_columns = [
            "x_centroid",
            "y_centroid",
            "x_on_centroid",
            "x_off_centroid",
            "y_on_centroid",
            "y_off_centroid",
            "x_polarity_difference",
            "y_polarity_difference",
            "x_on_off_midpoint",
            "y_on_off_midpoint",
            "x_q10",
            "x_q90",
            "x_robust_center",
            "y_q10",
            "y_q90",
            "y_robust_center",
        ]
        for column in interpolation_columns:
            output[column] = output[column].interpolate(limit_direction="both")

        return output


class SpectralEstimator:
    """Frozen-style primary-axis spectral analysis without calibration."""

    SIGNAL_COLUMNS = (
        "x_on_centroid",
        "x_off_centroid",
        "y_on_centroid",
        "y_off_centroid",
        "x_polarity_difference",
        "y_polarity_difference",
        "x_on_off_midpoint",
        "y_on_off_midpoint",
        "x_robust_center",
        "y_robust_center",
        "x_centroid",
        "y_centroid",
        "event_count",
        "positive_count",
        "negative_count",
        "polarity_sum",
    )

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.fs_hz = config.sampling_rate_hz

    def prepare_signal(self, binned: pd.DataFrame, column: str, *, smooth: Optional[bool] = None) -> np.ndarray:
        if column not in binned.columns:
            raise ValueError(f"Signal column does not exist: {column}")
        values = binned[column].to_numpy(dtype=float)
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        values = values - np.nanmean(values)
        use_smoothing = self.config.enable_centroid_smoothing if smooth is None else smooth
        if (
            use_smoothing
            and any(token in column for token in ("centroid", "center", "difference", "midpoint"))
            and np.nanstd(values) > 0
        ):
            values = safe_savgol(
                values,
                self.config.smoothing_window_samples,
                self.config.smoothing_polyorder,
            )
            values = values - np.nanmean(values)
        return values

    def primary_signal_name(self, binned: pd.DataFrame, axis: str) -> str:
        preferred = (
            [
                self.config.primary_x_signal,
                "x_polarity_difference",
                "x_robust_center",
                "x_centroid",
            ]
            if axis == "x"
            else [
                self.config.primary_y_signal,
                "y_polarity_difference",
                "y_robust_center",
                "y_centroid",
            ]
        )
        for name in preferred:
            if name in binned.columns and np.nanstd(binned[name].to_numpy(dtype=float)) > 1e-12:
                return name
        raise ValueError(f"No usable {axis}-axis motion signal is available.")

    def full_periodogram(self, signal: np.ndarray) -> pd.DataFrame:
        signal = np.asarray(signal, dtype=float)
        signal = signal - np.nanmean(signal)
        n = len(signal)
        if n < 4:
            raise ValueError("Signal is too short for a periodogram.")
        window = np.hanning(n)
        transformed = rfft(signal * window)
        frequencies = rfftfreq(n, d=1.0 / self.fs_hz)
        psd = (np.abs(transformed) ** 2) / (self.fs_hz * np.sum(window**2))
        if len(psd) > 2:
            psd[1:-1] *= 2.0
        return pd.DataFrame({"frequency_hz": frequencies, "periodogram_psd": psd})

    def welch_psd(self, signal: np.ndarray) -> pd.DataFrame:
        nperseg = min(self.config.welch_nperseg, len(signal))
        noverlap = min(
            nperseg - 1,
            int(round(nperseg * self.config.welch_overlap_fraction)),
        )
        frequencies, psd = welch(
            signal,
            fs=self.fs_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        return pd.DataFrame({"frequency_hz": frequencies, "welch_psd": psd})

    def band_mask(self, frequencies: np.ndarray) -> np.ndarray:
        return (frequencies >= self.config.f_min_hz) & (
            frequencies <= self.config.f_max_hz
        )

    @staticmethod
    def parabolic_peak_frequency(
        frequencies: np.ndarray,
        powers: np.ndarray,
        index: int,
    ) -> Tuple[float, float]:
        if index <= 0 or index >= len(powers) - 1:
            return float(frequencies[index]), float(powers[index])
        eps = 1e-30
        alpha = np.log(powers[index - 1] + eps)
        beta = np.log(powers[index] + eps)
        gamma = np.log(powers[index + 1] + eps)
        denominator = alpha - 2.0 * beta + gamma
        if denominator == 0 or not np.isfinite(denominator):
            return float(frequencies[index]), float(powers[index])
        delta = 0.5 * (alpha - gamma) / denominator
        delta = float(np.clip(delta, -0.5, 0.5))
        spacing = float(np.median(np.diff(frequencies)))
        refined_frequency = float(frequencies[index] + delta * spacing)
        refined_power = float(np.exp(beta - 0.25 * (alpha - gamma) * delta))
        return refined_frequency, refined_power

    def extract_peaks(self, periodogram: pd.DataFrame, signal_name: str) -> pd.DataFrame:
        frequencies = periodogram["frequency_hz"].to_numpy(dtype=float)
        powers = periodogram["periodogram_psd"].to_numpy(dtype=float)
        mask = self.band_mask(frequencies)
        f_band = frequencies[mask]
        p_band = powers[mask]
        columns = [
            "signal",
            "rank",
            "frequency_hz",
            "discrete_frequency_hz",
            "power",
            "prominence",
            "normalized_power",
        ]
        if len(f_band) < 3 or np.nanmax(p_band) <= 0:
            return pd.DataFrame(columns=columns)

        resolution = float(np.median(np.diff(f_band)))
        distance_bins = max(
            1,
            int(round(self.config.min_peak_distance_hz / resolution)),
        )
        prominence_threshold = self.config.peak_prominence_fraction * float(
            np.nanmax(p_band)
        )
        indices, properties = find_peaks(
            p_band,
            prominence=prominence_threshold,
            distance=distance_bins,
        )
        if len(indices) == 0:
            indices = np.argsort(p_band)[::-1][: self.config.top_k_peaks]
            prominences = np.full(len(indices), np.nan)
        else:
            prominences = properties.get("prominences", np.full(len(indices), np.nan))

        entries: List[Dict[str, Any]] = []
        maximum = float(np.nanmax(p_band))
        for local_index, prominence in zip(indices, prominences):
            refined_frequency, refined_power = self.parabolic_peak_frequency(
                f_band,
                p_band,
                int(local_index),
            )
            entries.append(
                {
                    "signal": signal_name,
                    "frequency_hz": refined_frequency,
                    "discrete_frequency_hz": float(f_band[int(local_index)]),
                    "power": refined_power,
                    "prominence": float(prominence) if np.isfinite(prominence) else np.nan,
                    "normalized_power": float(refined_power / maximum),
                }
            )

        peaks = (
            pd.DataFrame(entries)
            .sort_values("power", ascending=False)
            .head(self.config.top_k_peaks)
            .reset_index(drop=True)
        )
        peaks.insert(1, "rank", np.arange(1, len(peaks) + 1))
        return peaks[columns]

    def spectral_descriptors(self, psd: pd.DataFrame, power_column: str) -> Dict[str, float]:
        frequencies = psd["frequency_hz"].to_numpy(dtype=float)
        powers = psd[power_column].to_numpy(dtype=float)
        mask = self.band_mask(frequencies)
        f = frequencies[mask]
        p = powers[mask]
        eps = 1e-30
        if len(f) == 0 or np.sum(p) <= 0:
            return {
                "band_power_total": np.nan,
                "spectral_centroid_hz": np.nan,
                "spectral_bandwidth_hz": np.nan,
                "spectral_flatness": np.nan,
                "spectral_snr_db": np.nan,
            }

        total = float(np.trapezoid(p, f))
        centroid = float(np.sum(f * p) / np.sum(p))
        bandwidth = float(np.sqrt(np.sum((f - centroid) ** 2 * p) / np.sum(p)))
        flatness = float(np.exp(np.mean(np.log(p + eps))) / np.mean(p + eps))
        snr = float(10.0 * np.log10((np.max(p) + eps) / (np.median(p) + eps)))
        output: Dict[str, float] = {
            "band_power_total": total,
            "spectral_centroid_hz": centroid,
            "spectral_bandwidth_hz": bandwidth,
            "spectral_flatness": flatness,
            "spectral_snr_db": snr,
        }
        for low, high in zip(
            self.config.band_edges_hz[:-1],
            self.config.band_edges_hz[1:],
        ):
            selection = (frequencies >= low) & (frequencies < high)
            output[f"band_power_{int(low)}_{int(high)}_hz"] = (
                float(np.trapezoid(powers[selection], frequencies[selection]))
                if np.any(selection)
                else np.nan
            )
        return output

    def analyze_all_signals(
        self,
        binned: pd.DataFrame,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        periodograms: Dict[str, pd.DataFrame] = {}
        welch_tables: Dict[str, pd.DataFrame] = {}
        peak_tables: List[pd.DataFrame] = []
        descriptor_rows: List[Dict[str, Any]] = []

        for signal_name in self.SIGNAL_COLUMNS:
            if signal_name not in binned.columns:
                continue
            signal = self.prepare_signal(binned, signal_name)
            if np.nanstd(signal) <= 1e-12:
                continue
            periodogram = self.full_periodogram(signal)
            welch_table = self.welch_psd(signal)
            peaks = self.extract_peaks(periodogram, signal_name)
            descriptors = self.spectral_descriptors(welch_table, "welch_psd")
            descriptors.update(
                {
                    "signal": signal_name,
                    "signal_std": float(np.nanstd(signal)),
                }
            )
            periodograms[signal_name] = periodogram
            welch_tables[signal_name] = welch_table
            peak_tables.append(peaks)
            descriptor_rows.append(descriptors)

        all_peaks = pd.concat(peak_tables, ignore_index=True) if peak_tables else pd.DataFrame()
        all_descriptors = pd.DataFrame(descriptor_rows)
        return periodograms, welch_tables, all_peaks, all_descriptors

    def primary_motion_periodogram(
        self,
        periodograms: Dict[str, pd.DataFrame],
        binned: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, str]]:
        axis_tables: Dict[str, pd.DataFrame] = {}
        source_signals: Dict[str, str] = {}

        if self.config.active_axes in {"x", "xy"}:
            x_name = self.primary_signal_name(binned, "x")
            axis_tables["x"] = periodograms[x_name].copy()
            source_signals["x"] = x_name
        if self.config.active_axes in {"y", "xy"}:
            y_name = self.primary_signal_name(binned, "y")
            axis_tables["y"] = periodograms[y_name].copy()
            source_signals["y"] = y_name

        if not axis_tables:
            raise ValueError("No active-axis periodogram is available.")
        first_axis = next(iter(axis_tables))
        frequencies = axis_tables[first_axis]["frequency_hz"].to_numpy(dtype=float)
        vector_power = np.zeros_like(frequencies)
        for table in axis_tables.values():
            vector_power += np.interp(
                frequencies,
                table["frequency_hz"].to_numpy(dtype=float),
                table["periodogram_psd"].to_numpy(dtype=float),
            )
        source_label = "+".join(source_signals.values())
        vector = pd.DataFrame(
            {
                "frequency_hz": frequencies,
                "periodogram_psd": vector_power,
                "source_signal": source_label,
            }
        )
        return vector, axis_tables, source_signals

    def windowed_stability(self, binned: pd.DataFrame, source_signals: Dict[str, str]) -> pd.DataFrame:
        signals = {
            axis: self.prepare_signal(binned, name)
            for axis, name in source_signals.items()
        }
        if not signals:
            return pd.DataFrame()
        time = binned["t_sec"].to_numpy(dtype=float)
        window_samples = max(64, int(round(self.config.window_sec * self.fs_hz)))
        step_samples = max(
            1,
            int(round(window_samples * (1.0 - self.config.window_overlap))),
        )
        if len(time) < window_samples:
            window_samples = len(time)
            step_samples = len(time)

        def vector_periodogram(segment_signals: Dict[str, np.ndarray]) -> pd.DataFrame:
            tables = {axis: self.full_periodogram(values) for axis, values in segment_signals.items()}
            first = next(iter(tables.values()))
            frequencies = first["frequency_hz"].to_numpy(dtype=float)
            power = np.zeros_like(frequencies)
            for table in tables.values():
                power += np.interp(
                    frequencies,
                    table["frequency_hz"].to_numpy(dtype=float),
                    table["periodogram_psd"].to_numpy(dtype=float),
                )
            return pd.DataFrame({"frequency_hz": frequencies, "periodogram_psd": power})

        full = vector_periodogram(signals)
        full_f = full["frequency_hz"].to_numpy(dtype=float)
        full_p = full["periodogram_psd"].to_numpy(dtype=float)
        band = self.band_mask(full_f)
        eps = 1e-30
        rows: List[Dict[str, Any]] = []

        for start in range(0, len(time) - window_samples + 1, step_samples):
            stop = start + window_samples
            segment = {axis: values[start:stop] for axis, values in signals.items()}
            table = vector_periodogram(segment)
            peaks = self.extract_peaks(table, "window_primary_vector")
            interpolated = np.interp(
                full_f,
                table["frequency_hz"].to_numpy(dtype=float),
                table["periodogram_psd"].to_numpy(dtype=float),
            )
            row: Dict[str, Any] = {
                "window_index": len(rows),
                "window_start_sec": float(time[start]),
                "window_end_sec": float(time[stop - 1]),
                "window_center_sec": float((time[start] + time[stop - 1]) / 2.0),
                "window_psd_log_corr_to_full": robust_corr(
                    np.log10(interpolated[band] + eps),
                    np.log10(full_p[band] + eps),
                ),
            }
            frequencies = table["frequency_hz"].to_numpy(dtype=float)
            power = table["periodogram_psd"].to_numpy(dtype=float)
            total_mask = self.band_mask(frequencies)
            total = float(np.trapezoid(power[total_mask], frequencies[total_mask]))
            for low, high in zip(
                self.config.band_edges_hz[:-1], self.config.band_edges_hz[1:]
            ):
                selection = (frequencies >= low) & (frequencies < high)
                value = (
                    float(np.trapezoid(power[selection], frequencies[selection]))
                    if np.any(selection)
                    else np.nan
                )
                row[f"band_power_share_{int(low)}_{int(high)}_hz"] = (
                    value / total if total > 0 else np.nan
                )
            for _, peak in peaks.head(5).iterrows():
                row[f"top{int(peak['rank'])}_frequency_hz"] = float(
                    peak["frequency_hz"]
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def chirp_ridge(
        self,
        binned: pd.DataFrame,
        source_signals: Dict[str, str],
    ) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        signals = [self.prepare_signal(binned, name) for name in source_signals.values()]
        if not signals:
            return pd.DataFrame(), (np.array([]), np.array([]), np.empty((0, 0)))
        nperseg = min(
            len(signals[0]),
            max(128, int(round(self.config.chirp_window_sec * self.fs_hz))),
        )
        noverlap = min(
            nperseg - 1,
            int(round(nperseg * self.config.chirp_overlap)),
        )
        powers = []
        times = frequencies = None
        for signal in signals:
            f, t, power = spectrogram(
                signal,
                fs=self.fs_hz,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=max(self.config.chirp_nfft, nperseg),
                detrend="constant",
                scaling="density",
                mode="psd",
            )
            band = (f >= self.config.f_min_hz) & (f <= self.config.f_max_hz)
            powers.append(power[band, :])
            frequencies = f[band]
            times = t
        vector_power = np.sum(np.stack(powers, axis=0), axis=0)
        ridge_indices = np.argmax(vector_power, axis=0)
        estimated = frequencies[ridge_indices]
        table = pd.DataFrame(
            {
                "time_sec": times,
                "estimated_ridge_hz": estimated,
                "ridge_power": vector_power[ridge_indices, np.arange(len(ridge_indices))],
            }
        )
        return table, (times, frequencies, vector_power)


@dataclass
class KinematicTriplet:
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    frequencies_hz: np.ndarray
    spectral_weight: np.ndarray


class KinematicsEstimator:
    """Uncalibrated displacement, velocity, and acceleration estimators."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.fs_hz = config.sampling_rate_hz
        self.dt = 1.0 / self.fs_hz

    def raised_cosine_band_weight(self, frequencies: np.ndarray) -> np.ndarray:
        low = self.config.f_min_hz
        high = self.config.f_max_hz
        transition = max(0.0, self.config.transition_width_hz)
        weight = np.zeros_like(frequencies, dtype=float)
        core = (frequencies >= low) & (frequencies <= high)
        weight[core] = 1.0
        if transition > 0:
            lower = (frequencies >= max(0.0, low - transition)) & (frequencies < low)
            upper = (frequencies > high) & (frequencies <= high + transition)
            weight[lower] = 0.5 * (
                1.0
                - np.cos(
                    np.pi
                    * (frequencies[lower] - (low - transition))
                    / transition
                )
            )
            weight[upper] = 0.5 * (
                1.0
                + np.cos(
                    np.pi * (frequencies[upper] - high) / transition
                )
            )
        return weight

    @staticmethod
    def prepare_displacement(displacement: np.ndarray) -> np.ndarray:
        values = pd.Series(np.asarray(displacement, dtype=float)).interpolate(
            limit_direction="both"
        ).to_numpy()
        return values - np.mean(values)

    def frequency_domain(self, displacement: np.ndarray) -> KinematicTriplet:
        values = self.prepare_displacement(displacement)
        frequencies = rfftfreq(len(values), d=self.dt)
        spectrum = rfft(values)
        weight = self.raised_cosine_band_weight(frequencies)
        filtered = spectrum * weight
        velocity_spectrum = 1j * 2.0 * np.pi * frequencies * filtered
        acceleration_spectrum = -((2.0 * np.pi * frequencies) ** 2) * filtered
        return KinematicTriplet(
            displacement=irfft(filtered, n=len(values)),
            velocity=irfft(velocity_spectrum, n=len(values)),
            acceleration=irfft(acceleration_spectrum, n=len(values)),
            frequencies_hz=frequencies,
            spectral_weight=weight,
        )

    def time_domain(self, displacement: np.ndarray) -> KinematicTriplet:
        """Differentiate in time after the same broadband displacement filter.

        Using the same band-limited displacement isolates the difference between
        the derivative operators. The default method uses second-order central
        differences through NumPy gradient. Savitzky-Golay derivatives are an
        optional comparison setting and are not part of the frozen v8.1 values.
        """
        frequency_result = self.frequency_domain(displacement)
        filtered_displacement = frequency_result.displacement
        if self.config.time_derivative_method == "gradient":
            velocity = np.gradient(filtered_displacement, self.dt, edge_order=2)
            acceleration = np.gradient(velocity, self.dt, edge_order=2)
        else:
            velocity = safe_savgol(
                filtered_displacement,
                self.config.time_derivative_savgol_window_samples,
                self.config.time_derivative_savgol_polyorder,
                deriv=1,
                delta=self.dt,
            )
            acceleration = safe_savgol(
                filtered_displacement,
                self.config.time_derivative_savgol_window_samples,
                self.config.time_derivative_savgol_polyorder,
                deriv=2,
                delta=self.dt,
            )
        return KinematicTriplet(
            displacement=filtered_displacement,
            velocity=velocity,
            acceleration=acceleration,
            frequencies_hz=frequency_result.frequencies_hz,
            spectral_weight=frequency_result.spectral_weight,
        )


def kinematic_summary(values: np.ndarray, percentile: float, robust_peak_percentile: float) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    rms = float(np.sqrt(np.mean(values**2)))
    p_value = float(np.percentile(np.abs(values), percentile))
    robust_peak = float(np.percentile(np.abs(values), robust_peak_percentile))
    absolute_max = float(np.max(np.abs(values)))
    crest_factor = robust_peak / rms if rms > 0 else np.nan
    return {
        "rms": rms,
        f"p{percentile:g}": p_value,
        f"robust_peak_p{robust_peak_percentile:g}": robust_peak,
        "absolute_max": absolute_max,
        "crest_factor": crest_factor,
    }
