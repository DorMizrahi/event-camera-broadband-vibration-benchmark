from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class AnalysisConfig:
    """All numerical and operational settings for one analysis run.

    The notebook constructs this object explicitly in a code cell. No external
    configuration file is loaded. Values marked as frozen reproduce the v8.1
    simulation-stage choices, except that learned calibration is deliberately
    omitted. Settings under the time-domain comparison heading are new and exist
    only to compare spectral differentiation with time-domain differentiation.
    """

    # ------------------------------------------------------------------
    # Frozen v8.1 scientific settings
    # ------------------------------------------------------------------
    f_min_hz: float = 50.0
    f_max_hz: float = 2000.0
    bin_us: int = 50

    welch_nperseg: int = 2048
    welch_overlap_fraction: float = 0.50

    top_k_peaks: int = 10
    peak_prominence_fraction: float = 0.01
    min_peak_distance_hz: float = 5.0
    peak_match_tolerance_hz: float = 10.0

    band_edges_hz: Tuple[float, ...] = (50, 100, 200, 500, 1000, 2000)

    window_sec: float = 0.10
    window_overlap: float = 0.50

    chirp_window_sec: float = 0.05
    chirp_overlap: float = 0.75
    chirp_nfft: int = 4096
    chirp_tolerance_hz: float = 50.0

    enable_centroid_smoothing: bool = True
    smoothing_window_samples: int = 11
    smoothing_polyorder: int = 3

    acceleration_percentile: float = 95.0
    robust_peak_percentile: float = 99.5
    transition_width_hz: float = 20.0

    primary_x_signal: str = "x_on_centroid"
    primary_y_signal: str = "y_on_centroid"
    active_axes: str = "xy"  # "x", "y", or "xy"

    # ------------------------------------------------------------------
    # New comparison-only settings: time-domain differentiation
    # ------------------------------------------------------------------
    time_derivative_method: str = "gradient"  # "gradient" or "savgol"
    time_derivative_savgol_window_samples: int = 11
    time_derivative_savgol_polyorder: int = 3

    # ------------------------------------------------------------------
    # I/O and output settings
    # ------------------------------------------------------------------
    hdf5_chunk_rows: int = 2_000_000
    dat_chunk_duration_us: int = 50_000
    output_root: str = "measurement_outputs"
    save_plots: bool = True
    plot_dpi: int = 180

    def validate(self) -> None:
        if self.f_min_hz <= 0:
            raise ValueError("f_min_hz must be positive.")
        if self.f_max_hz <= self.f_min_hz:
            raise ValueError("f_max_hz must be greater than f_min_hz.")
        if self.bin_us <= 0:
            raise ValueError("bin_us must be positive.")
        if not 0 <= self.welch_overlap_fraction < 1:
            raise ValueError("welch_overlap_fraction must be in [0, 1).")
        if self.top_k_peaks < 1:
            raise ValueError("top_k_peaks must be at least 1.")
        if self.peak_prominence_fraction < 0:
            raise ValueError("peak_prominence_fraction must be non-negative.")
        if self.min_peak_distance_hz <= 0:
            raise ValueError("min_peak_distance_hz must be positive.")
        if self.active_axes not in {"x", "y", "xy"}:
            raise ValueError("active_axes must be 'x', 'y', or 'xy'.")
        if self.time_derivative_method not in {"gradient", "savgol"}:
            raise ValueError("time_derivative_method must be 'gradient' or 'savgol'.")
        if len(self.band_edges_hz) < 2:
            raise ValueError("band_edges_hz must contain at least two edges.")
        if tuple(sorted(self.band_edges_hz)) != tuple(self.band_edges_hz):
            raise ValueError("band_edges_hz must be sorted in ascending order.")
        nyquist = 0.5 * self.sampling_rate_hz
        if self.f_max_hz + self.transition_width_hz >= nyquist:
            raise ValueError(
                "f_max_hz + transition_width_hz must be below Nyquist. "
                f"Current Nyquist is {nyquist:g} Hz."
            )

    @property
    def bin_sec(self) -> float:
        return self.bin_us * 1e-6

    @property
    def sampling_rate_hz(self) -> float:
        return 1.0 / self.bin_sec

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sampling_rate_hz"] = self.sampling_rate_hz
        data["bin_sec"] = self.bin_sec
        return data
