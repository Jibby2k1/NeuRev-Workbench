"""Interpretable monochrome review video for dependent multiscale outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dependent_multiscale_figures import _AtomicMp4Writer


def _sample_bounds(values: np.ndarray, *, signed: bool, positive: bool = False) -> tuple[float, float]:
    array = np.asarray(values)
    sample = array[::max(1, len(array) // 64), ::4, ::4].astype(np.float32)
    if positive:
        return 0.0, max(float(np.quantile(np.maximum(sample, 0), 0.995)), 1e-6)
    if signed:
        limit = max(float(np.quantile(np.abs(sample), 0.995)), 1e-6)
        return -limit, limit
    low, high = np.quantile(sample, (0.005, 0.995))
    return float(low), max(float(high), float(low) + 1e-6)


def _gray(values: np.ndarray, bounds: tuple[float, float], *, positive: bool = False) -> np.ndarray:
    frame = np.asarray(values, dtype=np.float32)
    if positive:
        frame = np.maximum(frame, 0)
    low, high = bounds
    scaled = np.clip((frame - low) / (high - low), 0, 1)
    return np.rint(255 * scaled).astype(np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_grayscale_review_video(
    *,
    display_observation: np.ndarray,
    scientific_carrier: np.ndarray,
    channels: Mapping[str, np.ndarray],
    labels_xy: Sequence[tuple[float, float]],
    review_start_ui: int,
    destination: str | Path,
    fps: float = 10.0,
) -> dict[str, object]:
    """Write a six-panel grayscale movie with an explicit value legend."""
    raw = np.asarray(display_observation)
    carrier = np.asarray(scientific_carrier)
    panels = (
        ("Raw observation", raw, False, False),
        ("Accepted carrier", carrier, True, False),
        ("Neural signal: positive only", channels["structured_signal"], False, True),
        ("Background", channels["background"], True, False),
        ("Structured artifact", channels["structured_artifact"], True, False),
        ("Noise candidate", channels["noise_candidate"], True, False),
    )
    shape = carrier.shape
    if raw.shape != shape or any(np.asarray(values).shape != shape for _, values, _, _ in panels):
        raise ValueError("all grayscale review panels must be aligned [T,Y,X]")
    bounds = {
        name: _sample_bounds(values, signed=signed, positive=positive)
        for name, values, signed, positive in panels
    }
    panel_width, panel_height, label_height, banner_height = 382, 226, 38, 42
    columns, rows = 3, 2
    output_shape = (
        banner_height + rows * (panel_height + label_height),
        columns * panel_width,
    )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = _AtomicMp4Writer(target, output_shape, fps)
    title_font, panel_font = _font(14), _font(13)
    scale_x, scale_y = panel_width / shape[2], panel_height / shape[1]
    try:
        for frame_index in range(shape[0]):
            canvas = Image.new("RGB", (output_shape[1], output_shape[0]), "black")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 5),
                "SIGNED PANELS: black = negative | mid-gray = zero | white = positive",
                fill="white", font=title_font,
            )
            draw.text(
                (8, 23), "RAW / POSITIVE PANELS: black = low | white = high",
                fill=(205, 205, 205), font=panel_font,
            )
            draw.text((output_shape[1] - 105, 13), f"UI {review_start_ui + frame_index}", fill="white", font=title_font)
            for index, (name, stack, signed, positive) in enumerate(panels):
                row, column = divmod(index, columns)
                x0 = column * panel_width
                y0 = banner_height + row * (panel_height + label_height)
                low, high = bounds[name]
                label = f"{name}   [{low:.3g}, {high:.3g}]"
                draw.text((x0 + 6, y0 + 10), label, fill="white", font=panel_font)
                gray = _gray(stack[frame_index], (low, high), positive=positive)
                rendered = Image.fromarray(gray, mode="L").resize(
                    (panel_width, panel_height), Image.Resampling.BILINEAR
                ).convert("RGB")
                image_y = y0 + label_height
                canvas.paste(rendered, (x0, image_y))
                if name in {"Raw observation", "Accepted carrier", "Neural signal: positive only"}:
                    overlay = ImageDraw.Draw(canvas)
                    for x, y in labels_xy:
                        px, py = x0 + x * scale_x, image_y + y * scale_y
                        overlay.ellipse((px - 3, py - 3, px + 3, py + 3), outline="black", width=3)
                        overlay.ellipse((px - 3, py - 3, px + 3, py + 3), outline="white", width=1)
            writer.write(np.asarray(canvas, dtype=np.uint8))
        writer.close()
    except Exception:
        writer.abort()
        raise
    return {
        "path": target.name,
        "frames": int(shape[0]),
        "fps": float(fps),
        "shape_yx": list(output_shape),
        "bytes": target.stat().st_size,
        "sha256": _digest(target),
        "display_bounds": {name: list(value) for name, value in bounds.items()},
        "palette": "grayscale_only",
        "signed_legend": "black negative; mid-gray zero; white positive",
        "label_overlay": "black-backed white rings are sparse-positive labels; other pixels unknown",
        "scientific_status": "diagnostic_only_do_not_advance",
    }


def write_grayscale_review_artifact(
    *,
    raw_npy: str | Path,
    carrier_npy: str | Path,
    reconstruction_dir: str | Path,
    labels_tsv: str | Path,
    output_dir: str | Path,
    review_start_ui: int,
    review_end_ui: int,
) -> dict[str, object]:
    """Create a collision-safe review root from completed immutable arrays."""
    target = Path(output_dir).resolve()
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("grayscale review output or partial root exists")
    partial.mkdir(parents=True)
    started = time.time()
    try:
        raw_source = np.load(raw_npy, mmap_mode="r", allow_pickle=False)
        carrier = np.load(carrier_npy, mmap_mode="r", allow_pickle=False)
        start, stop = int(review_start_ui) - 1, int(review_end_ui)
        raw = raw_source[start:stop] if len(raw_source) != len(carrier) else raw_source
        root = Path(reconstruction_dir)
        channels = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("background", "structured_signal", "structured_artifact", "noise_candidate")
        }
        with Path(labels_tsv).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        labels = tuple({(float(row["x_px"]), float(row["y_px"])) for row in rows})
        video = write_grayscale_review_video(
            display_observation=raw,
            scientific_carrier=carrier,
            channels=channels,
            labels_xy=labels,
            review_start_ui=review_start_ui,
            destination=partial / "grayscale_decomposition_review.mp4",
        )
        metrics = {
            "schema_version": 1,
            "status": "completed_diagnostic_only",
            "source_reconstruction_dir": str(root.resolve()),
            "review_interval_ui_inclusive": [int(review_start_ui), int(review_end_ui)],
            "video": video,
            "elapsed_seconds": time.time() - started,
        }
        (partial / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "README.md").write_text(
            "# Grayscale decomposition review\n\nAll panels are grayscale. Signed panels use black for negative, mid-gray for zero, and white for positive. Black-backed white rings are sparse-positive labels; other pixels are unknown. This is diagnostic only.\n",
            encoding="utf-8",
        )
        partial.replace(target)
        return metrics
    except Exception:
        raise
