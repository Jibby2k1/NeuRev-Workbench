"""Chunked spatial cropping for frame-first video stacks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np

from neurobench.data.video import iter_video_chunks, video_metadata


@dataclass(frozen=True)
class CropBox:
    """Half-open image crop bounds in source x/y pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    def as_dict(self) -> dict[str, int]:
        return {"x0": int(self.x0), "y0": int(self.y0), "x1": int(self.x1), "y1": int(self.y1), "width": self.width, "height": self.height}


def validate_crop_box(crop: CropBox, *, source_width: int, source_height: int) -> None:
    if crop.x0 < 0 or crop.y0 < 0:
        raise ValueError("Crop origin must be non-negative.")
    if crop.x1 > int(source_width) or crop.y1 > int(source_height):
        raise ValueError(
            f"Crop {crop.as_dict()} exceeds source bounds width={int(source_width)}, height={int(source_height)}."
        )
    if crop.width <= 0 or crop.height <= 0:
        raise ValueError("Crop width and height must be positive.")


def crop_video_stack(
    *,
    source_path: str | Path,
    output_path: str | Path,
    crop: CropBox,
    chunk_size_frames: int = 64,
) -> dict[str, Any]:
    """Write a cropped TIFF stack without loading the full source video eagerly."""
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Cropping TIFF stacks requires tifffile.") from exc

    source = Path(source_path)
    output = Path(output_path)
    meta = video_metadata(source)
    validate_crop_box(crop, source_width=int(meta["width"]), source_height=int(meta["height"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(meta["frames"])
    first_dtype = None
    written = 0
    with tifffile.TiffWriter(output, bigtiff=True) as writer:
        for chunk in iter_video_chunks(source, chunk_size=int(chunk_size_frames)):
            arr = np.asarray(chunk.data)
            cropped = arr[:, crop.y0 : crop.y1, crop.x0 : crop.x1]
            if first_dtype is None:
                first_dtype = str(cropped.dtype)
            for frame in cropped:
                writer.write(frame, photometric="minisblack", contiguous=True)
            written += int(cropped.shape[0])
    out_meta = video_metadata(output)
    summary = {
        "schema_version": 1,
        "source_path": str(source),
        "output_path": str(output),
        "crop_box": crop.as_dict(),
        "source_shape": [int(v) for v in meta["shape"]],
        "output_shape": [int(v) for v in out_meta["shape"]],
        "source_dtype": str(meta.get("dtype")),
        "output_dtype": str(out_meta.get("dtype") or first_dtype),
        "frames": frame_count,
        "written_frames": written,
        "chunk_size_frames": int(chunk_size_frames),
    }
    if written != frame_count:
        raise RuntimeError(f"Expected to write {frame_count} frames from {source}, wrote {written}.")
    if int(out_meta["height"]) != crop.height or int(out_meta["width"]) != crop.width:
        raise RuntimeError(f"Cropped output has unexpected shape: {out_meta['shape']} for crop {crop.as_dict()}.")
    return summary


def write_crop_manifest(*, summaries: list[Mapping[str, Any]], output_path: str | Path, crop: CropBox) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "crop_box": crop.as_dict(),
        "video_count": len(summaries),
        "videos": [dict(item) for item in summaries],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
