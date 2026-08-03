"""Fixed-scale diagnostic videos for dependent multiscale decompositions."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class _AtomicMp4Writer:
    def __init__(self, path: Path, shape_yx: tuple[int, int], fps: float) -> None:
        height, width = shape_yx
        self.path = path
        self.partial = path.with_name(path.stem + ".partial.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(float(fps)), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-threads", "2", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.partial),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin was not created")
        self.shape = (height, width, 3)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError("video frame shape or dtype changed")
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr[-2000:]}")
        self.partial.replace(self.path)

    def abort(self) -> None:
        self.process.kill()
        self.process.wait()


def _bounds(values: np.ndarray, *, signed: bool) -> tuple[float, float]:
    array = np.asarray(values)
    sample = array[::max(1, len(array) // 64), ::4, ::4].astype(np.float32)
    if signed:
        limit = max(float(np.quantile(np.abs(sample), 0.995)), 1e-6)
        return -limit, limit
    low, high = np.quantile(sample, [0.005, 0.995])
    return float(low), max(float(high), float(low) + 1e-6)


def _render(values: np.ndarray, bounds: tuple[float, float], *, signed: bool) -> np.ndarray:
    frame = np.asarray(values, dtype=np.float32)
    low, high = bounds
    normalized = np.clip((frame - low) / (high - low), 0, 1)
    if not signed:
        gray = np.rint(255 * normalized).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    centered = 2 * normalized - 1
    magnitude = np.abs(centered)
    base = np.rint(35 + 80 * (1 - magnitude)).astype(np.uint8)
    red = np.where(centered >= 0, 120 + 135 * magnitude, base)
    blue = np.where(centered < 0, 120 + 135 * magnitude, base)
    green = 45 + 75 * (1 - magnitude)
    return np.stack((red, green, blue), axis=2).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_decomposition_video(
    *,
    display_observation: np.ndarray,
    scientific_carrier: np.ndarray,
    channels: Mapping[str, np.ndarray],
    views: Mapping[str, np.ndarray],
    labels_xy: Sequence[tuple[float, float]],
    review_start_ui: int,
    destination: str | Path,
    fps: float = 10.0,
) -> dict[str, object]:
    """Write one fixed-scale ten-panel full-interval diagnostic MP4."""
    display = np.asarray(display_observation)
    carrier = np.asarray(scientific_carrier)
    panels = {
        "raw observation": (display, False),
        "scientific carrier": (carrier, True),
        "background": (channels["background"], True),
        "structured signal": (channels["structured_signal"], True),
        "structured artifact": (channels["structured_artifact"], True),
        "noise candidate": (channels["noise_candidate"], True),
        "closure residual": (channels["closure_residual"], True),
        "5 px view": (views["scale_5"], True),
        "7 px view": (views["scale_7"], True),
        "15 px view": (views["scale_15"], True),
    }
    shape = carrier.shape
    if display.shape != shape or any(np.asarray(value).shape != shape for value, _ in panels.values()):
        raise ValueError("all video panels must be aligned [T,Y,X]")
    bounds = {name: _bounds(values, signed=signed) for name, (values, signed) in panels.items()}
    panel_width, panel_height, label_height = 286, 170, 22
    columns, rows = 5, 2
    output_shape = (rows * (panel_height + label_height), columns * panel_width)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = _AtomicMp4Writer(target, output_shape, fps)
    font = ImageFont.load_default()
    scale_x = panel_width / shape[2]
    scale_y = panel_height / shape[1]
    try:
        for frame_index in range(shape[0]):
            canvas = Image.new("RGB", (output_shape[1], output_shape[0]), "black")
            draw = ImageDraw.Draw(canvas)
            for index, (name, (stack, signed)) in enumerate(panels.items()):
                row, column = divmod(index, columns)
                x0 = column * panel_width
                y0 = row * (panel_height + label_height)
                rendered = Image.fromarray(_render(stack[frame_index], bounds[name], signed=signed))
                rendered = rendered.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
                canvas.paste(rendered, (x0, y0 + label_height))
                draw.text((x0 + 5, y0 + 4), name, fill="white", font=font)
                if name in {"raw observation", "scientific carrier", "structured signal"}:
                    for x, y in labels_xy:
                        px, py = x0 + x * scale_x, y0 + label_height + y * scale_y
                        draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=(0, 255, 255), width=1)
            draw.text(
                (output_shape[1] - 145, 4),
                f"UI {review_start_ui + frame_index}", fill=(255, 230, 100), font=font,
            )
            writer.write(np.asarray(canvas, dtype=np.uint8))
        writer.close()
    except Exception:
        writer.abort()
        raise
    return {
        "path": str(target),
        "frames": int(shape[0]),
        "fps": float(fps),
        "shape_yx": list(output_shape),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "display_bounds": {key: list(value) for key, value in bounds.items()},
        "scaling_contract": "fixed per panel across all frames",
        "label_overlay": "cyan sparse-positive evaluation labels; unmatched pixels unknown",
        "scientific_status": "diagnostic_only_do_not_advance",
    }
