from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def render_frame(
    config: Mapping[str, Any],
    displacement_x_px: float,
    displacement_y_px: float,
) -> np.ndarray:
    sampling = config["sampling"]
    rendering = config["rendering"]
    width = int(sampling["width_px"])
    height = int(sampling["height_px"])
    background = int(rendering["background_intensity"])
    marker_intensity = int(
        np.clip(
            background + float(rendering["marker_contrast"]) * (255 - background),
            background + 1,
            255,
        )
    )
    center_x = float(rendering["base_position_x_px"]) + float(displacement_x_px)
    center_y = float(rendering["base_position_y_px"]) + float(displacement_y_px)
    frame = np.full((height, width), background, dtype=np.uint8)

    shape = str(rendering["object_shape"])
    if shape == "circle":
        radius = float(rendering["marker_radius_px"])
        x0 = max(0, int(np.floor(center_x - radius - 2)))
        x1 = min(width, int(np.ceil(center_x + radius + 3)))
        y0 = max(0, int(np.floor(center_y - radius - 2)))
        y1 = min(height, int(np.ceil(center_y + radius + 3)))
        if x0 < x1 and y0 < y1:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
            frame[y0:y1, x0:x1][mask] = np.uint8(marker_intensity)
        return frame

    rectangle_width = float(rendering["rectangle_width_px"])
    rectangle_height = (
        rectangle_width
        if shape == "square"
        else float(rendering["rectangle_height_px"])
    )
    x0 = max(0, int(np.floor(center_x - rectangle_width / 2.0)))
    x1 = min(width, int(np.ceil(center_x + rectangle_width / 2.0)))
    y0 = max(0, int(np.floor(center_y - rectangle_height / 2.0)))
    y1 = min(height, int(np.ceil(center_y + rectangle_height / 2.0)))
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = np.uint8(marker_intensity)
    return frame


def validate_object_bounds(
    config: Mapping[str, Any],
    x_px: np.ndarray,
    y_px: np.ndarray,
) -> None:
    sampling = config["sampling"]
    rendering = config["rendering"]
    width = int(sampling["width_px"])
    height = int(sampling["height_px"])
    center_x = float(rendering["base_position_x_px"])
    center_y = float(rendering["base_position_y_px"])
    if rendering["object_shape"] == "circle":
        half_width = half_height = float(rendering["marker_radius_px"])
    else:
        half_width = float(rendering["rectangle_width_px"]) / 2.0
        half_height = (
            half_width
            if rendering["object_shape"] == "square"
            else float(rendering["rectangle_height_px"]) / 2.0
        )
    minimum_x = center_x + float(np.min(x_px)) - half_width
    maximum_x = center_x + float(np.max(x_px)) + half_width
    minimum_y = center_y + float(np.min(y_px)) - half_height
    maximum_y = center_y + float(np.max(y_px)) + half_height
    if minimum_x < 0 or maximum_x >= width or minimum_y < 0 or maximum_y >= height:
        raise ValueError(
            "The moving object leaves the image. Adjust the normalization, base position, "
            "object size, or resolution. Computed bounds: "
            f"x=[{minimum_x:.3f}, {maximum_x:.3f}], "
            f"y=[{minimum_y:.3f}, {maximum_y:.3f}]."
        )


def write_media(
    experiment_dir: Path,
    config: Mapping[str, Any],
    x_px: np.ndarray,
    y_px: np.ndarray,
    progress=print,
) -> dict[str, Any]:
    """Render every sample once, writing the MP4, preview, and optional PNG frames."""
    frames_dir = experiment_dir / "frames"
    save_frames = bool(config["sampling"]["save_frames"])
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    video_path = experiment_dir / "video.mp4"
    preview_path = experiment_dir / "preview.png"
    rendering = config["rendering"]
    fps = float(config["sampling"]["video_fps"])
    ffmpeg_params = [
        "-preset",
        str(rendering["video_preset"]),
        "-crf",
        str(int(rendering["video_crf"])),
    ]
    midpoint = len(x_px) // 2
    frame_count = 0
    progress(f"Rendering {len(x_px)} frames at {fps:g} FPS ...")
    try:
        with imageio.get_writer(
            str(video_path),
            fps=fps,
            codec=str(rendering["video_codec"]),
            macro_block_size=1,
            ffmpeg_params=ffmpeg_params,
        ) as writer:
            for index, (displacement_x, displacement_y) in enumerate(zip(x_px, y_px)):
                frame = render_frame(config, float(displacement_x), float(displacement_y))
                writer.append_data(frame)
                if save_frames:
                    Image.fromarray(frame).save(
                        frames_dir / f"frame_{index:06d}.png",
                        format="PNG",
                        optimize=False,
                    )
                if index == midpoint:
                    Image.fromarray(frame).save(preview_path, format="PNG")
                frame_count += 1
    except Exception as exc:
        raise RuntimeError(f"Video/frame rendering failed: {type(exc).__name__}: {exc}") from exc

    if not preview_path.exists():
        frame = render_frame(config, float(x_px[0]), float(y_px[0]))
        Image.fromarray(frame).save(preview_path, format="PNG")
    return {
        "video_path": str(video_path),
        "preview_path": str(preview_path),
        "frames_directory": str(frames_dir) if save_frames else None,
        "encoded_frames": frame_count,
        "saved_png_frames": frame_count if save_frames else 0,
        "writer_backend": f"imageio-{rendering['video_codec']}",
    }
