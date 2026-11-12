from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


@dataclass(frozen=True)
class ROI:
    """Axis-aligned ROI with exclusive upper bounds."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("ROI upper bounds must exceed lower bounds.")

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def clipped(self, width: int, height: int) -> "ROI":
        x0 = max(0, min(int(width) - 1, self.x_min))
        y0 = max(0, min(int(height) - 1, self.y_min))
        x1 = max(x0 + 1, min(int(width), self.x_max))
        y1 = max(y0 + 1, min(int(height), self.y_max))
        return ROI(x0, y0, x1, y1)

    def to_dict(self):
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
        }


def signed_activity_image(on_counts: np.ndarray, off_counts: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(on_counts, dtype=float)) - np.log1p(
        np.asarray(off_counts, dtype=float)
    )


def total_activity_image(on_counts: np.ndarray, off_counts: np.ndarray) -> np.ndarray:
    return np.log1p(
        np.asarray(on_counts, dtype=float) + np.asarray(off_counts, dtype=float)
    )


def auto_roi_from_activity(
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    *,
    percentile: float = 99.0,
    margin_px: int = 10,
    minimum_size_px: int = 16,
) -> ROI:
    """Suggest a compact ROI around the most active pixels.

    This is a convenience initializer, not part of the scientific estimator.
    The proposed ROI should be inspected before use on a new physical recording.
    """
    activity = np.asarray(on_counts, dtype=float) + np.asarray(off_counts, dtype=float)
    if activity.ndim != 2 or activity.size == 0 or np.max(activity) <= 0:
        raise ValueError("Cannot infer an ROI from an empty activity image.")

    positive = activity[activity > 0]
    threshold = np.percentile(positive, percentile) if len(positive) else np.max(activity)
    mask = activity >= threshold
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        ys, xs = np.nonzero(activity > 0)

    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(activity.shape[1], int(xs.max()) + margin_px + 1)
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(activity.shape[0], int(ys.max()) + margin_px + 1)

    if x1 - x0 < minimum_size_px:
        center = (x0 + x1) // 2
        half = max(1, minimum_size_px // 2)
        x0 = max(0, center - half)
        x1 = min(activity.shape[1], x0 + minimum_size_px)
    if y1 - y0 < minimum_size_px:
        center = (y0 + y1) // 2
        half = max(1, minimum_size_px // 2)
        y0 = max(0, center - half)
        y1 = min(activity.shape[0], y0 + minimum_size_px)

    return ROI(x0, y0, x1, y1)


def plot_roi_preview(
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    roi: Optional[ROI] = None,
    *,
    signed: bool = False,
    title: str = "Event accumulation and selected ROI",
):
    image = (
        signed_activity_image(on_counts, off_counts)
        if signed
        else total_activity_image(on_counts, off_counts)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, origin="upper", aspect="equal")
    if roi is not None:
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
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def select_roi_interactive(
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    *,
    initial_roi: Optional[ROI] = None,
    signed: bool = False,
) -> ROI:
    """Select an ROI with Matplotlib's RectangleSelector.

    In JupyterLab, `%matplotlib widget` usually provides the best interaction.
    Close the figure after selecting the rectangle.
    """
    image = (
        signed_activity_image(on_counts, off_counts)
        if signed
        else total_activity_image(on_counts, off_counts)
    )
    selected = {"roi": initial_roi}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, origin="upper", aspect="equal")
    ax.set_title("Drag to select ROI, then close the figure")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    if initial_roi is not None:
        ax.add_patch(
            Rectangle(
                (initial_roi.x_min, initial_roi.y_min),
                initial_roi.width,
                initial_roi.height,
                fill=False,
                linewidth=2,
            )
        )

    def on_select(eclick, erelease):
        values = (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        if any(value is None for value in values):
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        selected["roi"] = ROI(
            int(np.floor(x0)),
            int(np.floor(y0)),
            int(np.ceil(x1)),
            int(np.ceil(y1)),
        ).clipped(image.shape[1], image.shape[0])

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
    )
    plt.show(block=True)
    selector.set_active(False)
    if selected["roi"] is None:
        raise RuntimeError("No ROI was selected.")
    return selected["roi"]
