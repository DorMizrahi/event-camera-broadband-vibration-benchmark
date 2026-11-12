from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".h5", ".hdf5", ".csv", ".txt", ".dat"}


@dataclass
class EventFileInfo:
    source_path: Path
    source_format: str
    width: Optional[int]
    height: Optional[int]
    event_count: Optional[int]
    t_min_us: Optional[int]
    t_max_us: Optional[int]
    duration_sec: Optional[float]
    polarity_values: Tuple[int, ...]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "source_format": self.source_format,
            "width": self.width,
            "height": self.height,
            "event_count": self.event_count,
            "t_min_us": self.t_min_us,
            "t_max_us": self.t_max_us,
            "duration_sec": self.duration_sec,
            "polarity_values": list(self.polarity_values),
            "notes": list(self.notes),
        }


@dataclass
class EventData:
    events: pd.DataFrame
    info: EventFileInfo


def _read_delimited_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Read CSV or whitespace-delimited text without assuming a delimiter."""
    try:
        iterator = pd.read_csv(path, chunksize=chunksize)
        first = True
        for chunk in iterator:
            if first and len(chunk.columns) == 1:
                raise ValueError("single-column parse")
            first = False
            yield chunk
        return
    except Exception:
        pass

    for chunk in pd.read_csv(path, sep=r"\s+", engine="python", chunksize=chunksize):
        yield chunk


def _normalize_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an event chunk to integer [t_us, x, y, p]."""
    if frame.empty:
        return pd.DataFrame(columns=["t_us", "x", "y", "p"])

    frame = frame.copy()
    rename_map = {}
    for candidate in ("timestamp_us", "timestamp", "ts"):
        if candidate in frame.columns and "t_us" not in frame.columns:
            rename_map[candidate] = "t_us"
            break
    if "t" in frame.columns and "t_us" not in frame.columns:
        rename_map["t"] = "t_us"
    frame = frame.rename(columns=rename_map)

    if "t_us" not in frame.columns and "t_sec" in frame.columns:
        frame["t_us"] = frame["t_sec"].astype(float) * 1e6

    required = {"t_us", "x", "y", "p"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    frame = frame[["t_us", "x", "y", "p"]].replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return frame

    t = frame["t_us"].to_numpy(dtype=float)
    # Generic t columns may contain seconds. Integer-valued microseconds are kept.
    if np.nanmax(np.abs(t)) < 1000 and np.any(np.abs(t - np.rint(t)) > 1e-9):
        t = t * 1e6

    p = frame["p"].to_numpy()
    if np.any(p < 0):
        p = (p > 0).astype(np.uint8)
    else:
        p = (p != 0).astype(np.uint8)

    output = pd.DataFrame(
        {
            "t_us": np.rint(t).astype(np.int64),
            "x": np.rint(frame["x"].to_numpy(dtype=float)).astype(np.int32),
            "y": np.rint(frame["y"].to_numpy(dtype=float)).astype(np.int32),
            "p": p.astype(np.uint8),
        }
    )
    return output


def _metavision_iterator(path: Path, delta_t_us: int):
    try:
        from metavision_core.event_io import EventsIterator  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading .dat files requires the Prophesee Metavision SDK and its "
            "Python module 'metavision_core'. HDF5/CSV/TXT remain available "
            "without this optional dependency."
        ) from exc

    kwargs = {
        "input_path": str(path),
        "mode": "delta_t",
        "delta_t": int(delta_t_us),
        "relative_timestamps": False,
    }
    try:
        return EventsIterator(**kwargs)
    except TypeError:
        kwargs.pop("input_path")
        return EventsIterator(str(path), **kwargs)


def _structured_events_to_frame(chunk: np.ndarray) -> pd.DataFrame:
    if chunk is None or len(chunk) == 0:
        return pd.DataFrame(columns=["t_us", "x", "y", "p"])
    names = set(chunk.dtype.names or ())
    required = {"t", "x", "y", "p"}
    if not required.issubset(names):
        raise ValueError(f"Unexpected DAT event fields: {sorted(names)}")
    return _normalize_chunk(
        pd.DataFrame(
            {
                "t_us": chunk["t"],
                "x": chunk["x"],
                "y": chunk["y"],
                "p": chunk["p"],
            }
        )
    )


def iter_event_chunks(
    path: Path,
    *,
    hdf5_chunk_rows: int = 2_000_000,
    dat_chunk_duration_us: int = 50_000,
) -> Iterator[pd.DataFrame]:
    """Yield normalized chunks from HDF5, CSV/TXT, or Metavision DAT."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if suffix in {".csv", ".txt"}:
        for chunk in _read_delimited_chunks(path, hdf5_chunk_rows):
            normalized = _normalize_chunk(chunk)
            if not normalized.empty:
                yield normalized
        return

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            if "events" in handle:
                dataset = handle["events"]
                if dataset.dtype.names:
                    names = set(dataset.dtype.names)
                    time_name = "t_us" if "t_us" in names else "t" if "t" in names else None
                    if time_name is None or not {"x", "y", "p"}.issubset(names):
                        raise ValueError(
                            "Structured HDF5 'events' dataset must contain t_us/t, x, y, p."
                        )
                    for start in range(0, len(dataset), hdf5_chunk_rows):
                        values = np.asarray(dataset[start : start + hdf5_chunk_rows])
                        frame = pd.DataFrame(
                            {
                                "t_us": values[time_name],
                                "x": values["x"],
                                "y": values["y"],
                                "p": values["p"],
                            }
                        )
                        normalized = _normalize_chunk(frame)
                        if not normalized.empty:
                            yield normalized
                    return

                if dataset.ndim != 2 or dataset.shape[1] < 4:
                    raise ValueError("HDF5 dataset 'events' must be an N x 4 array.")
                for start in range(0, dataset.shape[0], hdf5_chunk_rows):
                    values = np.asarray(dataset[start : start + hdf5_chunk_rows, :4])
                    normalized = _normalize_chunk(
                        pd.DataFrame(values, columns=["t_us", "x", "y", "p"])
                    )
                    if not normalized.empty:
                        yield normalized
                return

            candidate_sets = [
                ("t_us", "x", "y", "p"),
                ("t", "x", "y", "p"),
            ]
            selected = next(
                (keys for keys in candidate_sets if all(key in handle for key in keys)),
                None,
            )
            if selected is None:
                raise ValueError(
                    "Unsupported HDF5 layout. Expected an 'events' dataset or "
                    "separate t_us/x/y/p (or t/x/y/p) arrays."
                )
            time_key, x_key, y_key, p_key = selected
            n_rows = len(handle[time_key])
            for start in range(0, n_rows, hdf5_chunk_rows):
                stop = min(n_rows, start + hdf5_chunk_rows)
                normalized = _normalize_chunk(
                    pd.DataFrame(
                        {
                            "t_us": np.asarray(handle[time_key][start:stop]),
                            "x": np.asarray(handle[x_key][start:stop]),
                            "y": np.asarray(handle[y_key][start:stop]),
                            "p": np.asarray(handle[p_key][start:stop]),
                        }
                    )
                )
                if not normalized.empty:
                    yield normalized
        return

    iterator = _metavision_iterator(path, dat_chunk_duration_us)
    for chunk in iterator:
        normalized = _structured_events_to_frame(chunk)
        if not normalized.empty:
            yield normalized


def _dat_sensor_size(path: Path, delta_t_us: int) -> Tuple[Optional[int], Optional[int]]:
    try:
        iterator = _metavision_iterator(path, delta_t_us)
        getter = getattr(iterator, "get_size", None)
        if callable(getter):
            size = getter()
            if isinstance(size, (tuple, list)) and len(size) == 2:
                height, width = int(size[0]), int(size[1])
                return width, height
    except Exception:
        pass
    return None, None


def inspect_event_file(
    path: Path,
    *,
    hdf5_chunk_rows: int = 2_000_000,
    dat_chunk_duration_us: int = 50_000,
    scan: bool = True,
) -> EventFileInfo:
    """Validate an event file and return basic metadata."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported event file extension: {suffix}")

    width = height = count = t_min = t_max = None
    polarity_values = set()
    notes = []

    if suffix == ".dat":
        width, height = _dat_sensor_size(path, dat_chunk_duration_us)
    elif suffix in {".h5", ".hdf5"}:
        try:
            with h5py.File(path, "r") as handle:
                for key in ("sensor_width", "width"):
                    if key in handle.attrs:
                        width = int(handle.attrs[key])
                for key in ("sensor_height", "height"):
                    if key in handle.attrs:
                        height = int(handle.attrs[key])
                if "events" in handle:
                    count = int(handle["events"].shape[0])
        except OSError as exc:
            raise ValueError(f"Unable to open HDF5 file: {path}") from exc

    if scan:
        count_accumulator = 0
        max_x = max_y = -1
        previous_t = None
        for chunk in iter_event_chunks(
            path,
            hdf5_chunk_rows=hdf5_chunk_rows,
            dat_chunk_duration_us=dat_chunk_duration_us,
        ):
            times = chunk["t_us"].to_numpy(dtype=np.int64)
            if np.any(np.diff(times) < 0):
                raise ValueError("Event timestamps are not monotonic within a chunk.")
            if previous_t is not None and times[0] < previous_t:
                raise ValueError("Event timestamps are not monotonic across chunks.")
            previous_t = int(times[-1])
            t_min = int(times[0]) if t_min is None else min(t_min, int(times[0]))
            t_max = int(times[-1]) if t_max is None else max(t_max, int(times[-1]))
            max_x = max(max_x, int(chunk["x"].max()))
            max_y = max(max_y, int(chunk["y"].max()))
            polarity_values.update(int(v) for v in np.unique(chunk["p"]))
            count_accumulator += len(chunk)
        count = count_accumulator
        width = max(width or 0, max_x + 1) if max_x >= 0 else width
        height = max(height or 0, max_y + 1) if max_y >= 0 else height

    duration = None
    if t_min is not None and t_max is not None:
        duration = max(0.0, (t_max - t_min) * 1e-6)
    if count == 0:
        notes.append("The file contains no readable events.")

    return EventFileInfo(
        source_path=path,
        source_format=suffix.lstrip("."),
        width=width,
        height=height,
        event_count=count,
        t_min_us=t_min,
        t_max_us=t_max,
        duration_sec=duration,
        polarity_values=tuple(sorted(polarity_values)),
        notes=tuple(notes),
    )


def load_events(
    path: Path,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    t_start_us: Optional[int] = None,
    t_end_us: Optional[int] = None,
    hdf5_chunk_rows: int = 2_000_000,
    dat_chunk_duration_us: int = 50_000,
) -> EventData:
    """Load events after optional temporal and ROI filtering.

    ROI uses exclusive upper bounds: (x_min, y_min, x_max, y_max).
    Filtering is performed chunk-by-chunk before concatenation.
    """
    frames = []
    scanned_count = 0
    max_x = max_y = -1
    min_t = max_t = None
    polarity_values = set()

    for chunk in iter_event_chunks(
        path,
        hdf5_chunk_rows=hdf5_chunk_rows,
        dat_chunk_duration_us=dat_chunk_duration_us,
    ):
        scanned_count += len(chunk)
        if t_start_us is not None:
            chunk = chunk[chunk["t_us"] >= int(t_start_us)]
        if t_end_us is not None:
            chunk = chunk[chunk["t_us"] <= int(t_end_us)]
        if roi is not None:
            x0, y0, x1, y1 = map(int, roi)
            chunk = chunk[
                (chunk["x"] >= x0)
                & (chunk["x"] < x1)
                & (chunk["y"] >= y0)
                & (chunk["y"] < y1)
            ]
        if chunk.empty:
            continue
        frames.append(chunk)
        max_x = max(max_x, int(chunk["x"].max()))
        max_y = max(max_y, int(chunk["y"].max()))
        min_t = int(chunk["t_us"].iloc[0]) if min_t is None else min(min_t, int(chunk["t_us"].iloc[0]))
        max_t = int(chunk["t_us"].iloc[-1]) if max_t is None else max(max_t, int(chunk["t_us"].iloc[-1]))
        polarity_values.update(int(v) for v in np.unique(chunk["p"]))

    if not frames:
        raise ValueError("No events remain after the requested time/ROI filters.")

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values("t_us", kind="mergesort").reset_index(drop=True)
    events["t_sec"] = (events["t_us"] - events["t_us"].iloc[0]) * 1e-6
    duration = float(events["t_sec"].iloc[-1]) if len(events) > 1 else 0.0

    source_info = inspect_event_file(
        path,
        hdf5_chunk_rows=hdf5_chunk_rows,
        dat_chunk_duration_us=dat_chunk_duration_us,
        scan=False,
    )
    width = source_info.width or (max_x + 1)
    height = source_info.height or (max_y + 1)
    notes = list(source_info.notes)
    if roi is not None:
        notes.append(
            f"Loaded ROI {tuple(map(int, roi))}; retained {len(events)} of "
            f"{scanned_count} scanned events."
        )

    info = EventFileInfo(
        source_path=Path(path),
        source_format=source_info.source_format,
        width=width,
        height=height,
        event_count=len(events),
        t_min_us=min_t,
        t_max_us=max_t,
        duration_sec=duration,
        polarity_values=tuple(sorted(polarity_values)),
        notes=tuple(notes),
    )
    return EventData(events=events, info=info)


def build_event_accumulation(
    path: Path,
    *,
    hdf5_chunk_rows: int = 2_000_000,
    dat_chunk_duration_us: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, EventFileInfo]:
    """Stream a recording into ON/OFF activity images for ROI selection."""
    hint = inspect_event_file(
        path,
        hdf5_chunk_rows=hdf5_chunk_rows,
        dat_chunk_duration_us=dat_chunk_duration_us,
        scan=False,
    )
    width = int(hint.width or 0)
    height = int(hint.height or 0)
    on = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint64)
    off = np.zeros_like(on)
    total = 0
    t_min = t_max = None
    polarities = set()

    def ensure_shape(required_width: int, required_height: int) -> None:
        nonlocal on, off
        pad_y = max(0, required_height - on.shape[0])
        pad_x = max(0, required_width - on.shape[1])
        if pad_x or pad_y:
            on = np.pad(on, ((0, pad_y), (0, pad_x)), mode="constant")
            off = np.pad(off, ((0, pad_y), (0, pad_x)), mode="constant")

    for chunk in iter_event_chunks(
        path,
        hdf5_chunk_rows=hdf5_chunk_rows,
        dat_chunk_duration_us=dat_chunk_duration_us,
    ):
        x = chunk["x"].to_numpy(dtype=np.int64)
        y = chunk["y"].to_numpy(dtype=np.int64)
        p = chunk["p"].to_numpy(dtype=np.uint8)
        valid = (x >= 0) & (y >= 0)
        x, y, p = x[valid], y[valid], p[valid]
        if len(x) == 0:
            continue
        ensure_shape(int(x.max()) + 1, int(y.max()) + 1)
        np.add.at(on, (y[p > 0], x[p > 0]), 1)
        np.add.at(off, (y[p == 0], x[p == 0]), 1)
        total += len(x)
        times = chunk["t_us"].to_numpy(dtype=np.int64)
        t_min = int(times[0]) if t_min is None else min(t_min, int(times[0]))
        t_max = int(times[-1]) if t_max is None else max(t_max, int(times[-1]))
        polarities.update(int(v) for v in np.unique(p))

    if total == 0:
        raise ValueError("The event file contains no readable events.")

    duration = max(0.0, (t_max - t_min) * 1e-6) if t_min is not None and t_max is not None else None
    info = EventFileInfo(
        source_path=Path(path),
        source_format=Path(path).suffix.lower().lstrip("."),
        width=on.shape[1],
        height=on.shape[0],
        event_count=total,
        t_min_us=t_min,
        t_max_us=t_max,
        duration_sec=duration,
        polarity_values=tuple(sorted(polarities)),
        notes=tuple(hint.notes),
    )
    return on, off, info
