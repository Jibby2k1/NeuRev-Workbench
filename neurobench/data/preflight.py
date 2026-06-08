"""Preflight estimates for template-grid real-data pilots."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np


def build_template_grid_preflight(
    *,
    manifest: Mapping[str, Any],
    output_root: str | Path,
    rows: int = 32,
    cols: int = 32,
    feature_channels: int = 1,
    registered_dtype: str = "float32",
    chunk_size_frames: int = 64,
    expected_video_count: int | None = 27,
    disk_safety_multiplier: float = 1.25,
) -> dict[str, Any]:
    """Estimate output sizes and safety warnings without reading video pixels."""
    videos = list(manifest.get("videos") or [])
    output = Path(output_root).expanduser()
    disk_path = _nearest_existing_parent(output)
    disk = shutil.disk_usage(disk_path)
    registered_itemsize = int(np.dtype(registered_dtype).itemsize)
    grid_itemsize = int(np.dtype(np.float32).itemsize)
    rows = int(rows)
    cols = int(cols)
    feature_channels = int(feature_channels)
    registered_total = 0
    grid_total = 0
    input_total = 0
    largest_chunk_bytes = 0
    warnings: list[str] = []
    video_summaries: list[dict[str, Any]] = []
    shapes: Counter[tuple[int | None, int | None]] = Counter()
    labels: Counter[str] = Counter()
    for video in videos:
        path = Path(str(video.get("path") or "")).expanduser()
        frame_count = _optional_int(video.get("frame_count"))
        height = _optional_int(video.get("height"))
        width = _optional_int(video.get("width"))
        label = str(video.get("label") or "")
        labels[label] += 1
        shapes[(height, width)] += 1
        exists = path.is_file()
        file_bytes = int(path.stat().st_size) if exists else 0
        if not exists:
            warnings.append(f"missing video path: {path}")
        input_total += file_bytes
        registered_bytes = 0
        grid_bytes = 0
        chunk_bytes = 0
        if frame_count is None or height is None or width is None:
            warnings.append(f"missing dimensions for video_id={video.get('video_id')}")
        else:
            registered_bytes = frame_count * height * width * registered_itemsize
            grid_state_bytes = frame_count * rows * cols * feature_channels * grid_itemsize
            grid_bytes = grid_state_bytes * 2
            chunk_bytes = min(frame_count, int(chunk_size_frames)) * height * width * registered_itemsize * 3
            registered_total += registered_bytes
            grid_total += grid_bytes
            largest_chunk_bytes = max(largest_chunk_bytes, chunk_bytes)
        video_summaries.append(
            {
                "video_id": str(video.get("video_id") or ""),
                "label": label,
                "path": str(path),
                "exists": exists,
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "input_bytes": file_bytes,
                "estimated_registered_bytes": registered_bytes,
                "estimated_grid_npz_payload_bytes": grid_bytes,
                "estimated_peak_chunk_bytes": chunk_bytes,
            }
        )
    expected_output = registered_total + grid_total
    required_disk = int(expected_output * float(disk_safety_multiplier))
    if expected_video_count is not None and len(videos) != int(expected_video_count):
        warnings.append(f"expected {int(expected_video_count)} videos, found {len(videos)}")
    missing_labels = [label for label in ["left", "right", "neutral"] if labels.get(label, 0) == 0]
    if missing_labels:
        warnings.append(f"missing labels: {', '.join(missing_labels)}")
    known_shapes = [shape for shape in shapes if None not in shape]
    if len(known_shapes) > 1:
        warnings.append(f"videos have multiple image shapes: {dict((str(k), v) for k, v in shapes.items())}")
    if disk.free < required_disk:
        warnings.append(
            f"available disk at {disk_path} is {disk.free} bytes, below estimated safe requirement {required_disk} bytes"
        )
    ram = _available_ram_bytes()
    if ram is not None and largest_chunk_bytes > ram * 0.5:
        warnings.append(
            f"estimated peak chunk workspace {largest_chunk_bytes} bytes is more than half available RAM {ram} bytes"
        )
    return {
        "schema_version": 1,
        "dataset_id": str(manifest.get("dataset_id") or ""),
        "output_root": str(output),
        "video_count": len(videos),
        "label_counts": {str(k): int(v) for k, v in labels.items()},
        "grid": {"rows": rows, "cols": cols, "feature_channels": feature_channels},
        "chunk_size_frames": int(chunk_size_frames),
        "registered_dtype": str(np.dtype(registered_dtype)),
        "input_total_bytes": int(input_total),
        "estimated_registered_total_bytes": int(registered_total),
        "estimated_grid_total_bytes": int(grid_total),
        "estimated_output_total_bytes": int(expected_output),
        "disk": {"path": str(disk_path), "total_bytes": int(disk.total), "used_bytes": int(disk.used), "free_bytes": int(disk.free), "required_with_safety_bytes": int(required_disk)},
        "available_ram_bytes": ram,
        "estimated_largest_chunk_workspace_bytes": int(largest_chunk_bytes),
        "warnings": warnings,
        "videos": video_summaries,
    }


def _nearest_existing_parent(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore
    except ModuleNotFoundError:
        return None
    return int(psutil.virtual_memory().available)
