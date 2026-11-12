from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def render_marker_frame(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    marker_radius_px: float,
    marker_intensity: int,
    background_intensity: int = 5,
) -> np.ndarray:
    frame = np.full((height, width), int(background_intensity), dtype=np.uint8)
    r = float(marker_radius_px)
    x0 = max(0, int(np.floor(center_x - r - 2)))
    x1 = min(width, int(np.ceil(center_x + r + 3)))
    y0 = max(0, int(np.floor(center_y - r - 2)))
    y1 = min(height, int(np.ceil(center_y + r + 3)))
    if x0 >= x1 or y0 >= y1:
        return frame
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= r * r
    patch = frame[y0:y1, x0:x1]
    patch[mask] = np.uint8(marker_intensity)
    return frame


def write_video(
    video_path: Path,
    preview_path: Path,
    x_px: np.ndarray,
    y_px: np.ndarray,
    fps: float,
    resolution: Tuple[int, int],
    marker_radius_px: float,
    marker_contrast: float,
    base_position: Tuple[float, float],
    max_frames: int | None = None,
) -> Dict[str, object]:
    width, height = resolution
    bg = 5
    marker_intensity = int(np.clip(bg + marker_contrast * (255 - bg), 8, 255))
    count = len(x_px) if max_frames is None else min(len(x_px), int(max_frames))

    imageio = _try_import_imageio()
    if imageio is None:
        video_path.write_text("imageio is unavailable; video was not encoded.\n", encoding="utf-8")
        frame = render_marker_frame(width, height, base_position[0], base_position[1], marker_radius_px, marker_intensity, bg)
        _write_png(preview_path, frame)
        return {"writer_backend": "placeholder", "writer_ok": False, "encoded_frames": 0}

    mid = min(count - 1, count // 2)
    preview = render_marker_frame(
        width,
        height,
        base_position[0] + float(x_px[mid]),
        base_position[1] + float(y_px[mid]),
        marker_radius_px,
        marker_intensity,
        bg,
    )
    _write_png(preview_path, preview)

    try:
        with imageio.get_writer(
            str(video_path),
            fps=float(fps),
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "28"],
        ) as writer:
            for i in range(count):
                frame = render_marker_frame(
                    width,
                    height,
                    base_position[0] + float(x_px[i]),
                    base_position[1] + float(y_px[i]),
                    marker_radius_px,
                    marker_intensity,
                    bg,
                )
                writer.append_data(frame)
        return {"writer_backend": "imageio-libx264", "writer_ok": True, "encoded_frames": count}
    except Exception as exc:
        video_path.write_text(f"Video encoding failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"writer_backend": "placeholder", "writer_ok": False, "encoded_frames": 0, "warning": str(exc)}


def _try_import_imageio():
    try:
        import imageio.v2 as imageio

        return imageio
    except Exception:
        return None


def _write_png(path: Path, frame: np.ndarray) -> None:
    imageio = _try_import_imageio()
    if imageio is not None:
        imageio.imwrite(str(path), frame)
        return
    from PIL import Image

    Image.fromarray(frame).save(path)
