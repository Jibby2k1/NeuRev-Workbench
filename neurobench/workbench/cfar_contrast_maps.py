"""Streaming Gamma CFAR contrast-map export helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from neurobench.workbench.intermediates import frame_output_path, write_png_gray8


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("NumPy is required to export CFAR contrast maps.") from exc
    return np


def _load_uniform_filter():
    try:
        from scipy.ndimage import uniform_filter
    except ModuleNotFoundError as exc:
        raise SystemExit("SciPy is required to export Gamma CFAR contrast maps.") from exc
    return uniform_filter


def cfar_threshold(pfa: float) -> float:
    np = _load_numpy()
    return float(np.sqrt(max(0.0, -2.0 * np.log(float(pfa)))))


def compute_cfar_contrast_block(
    block: Any,
    *,
    guard_px: int,
    training_radius_px: int,
    epsilon: float = 1e-6,
) -> Any:
    """Return the continuous positive Gamma CFAR local-contrast score."""
    np = _load_numpy()
    uniform_filter = _load_uniform_filter()
    if training_radius_px <= guard_px:
        raise ValueError("training_radius_px must be larger than guard_px.")

    arr = np.asarray(block, dtype=np.float32)
    was_2d = arr.ndim == 2
    if was_2d:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D block, got shape {arr.shape}.")

    evidence = np.maximum(arr, 0.0).astype(np.float32, copy=False)
    outer_width = 2 * int(training_radius_px) + 1
    guard_width = 2 * int(guard_px) + 1
    outer_area = float(outer_width * outer_width)
    guard_area = float(guard_width * guard_width)
    training_area = outer_area - guard_area

    outer_mean = uniform_filter(evidence, size=(1, outer_width, outer_width), mode="nearest")
    outer_sq_mean = uniform_filter(evidence * evidence, size=(1, outer_width, outer_width), mode="nearest")
    guard_mean = uniform_filter(evidence, size=(1, guard_width, guard_width), mode="nearest")
    guard_sq_mean = uniform_filter(evidence * evidence, size=(1, guard_width, guard_width), mode="nearest")
    local_mean = ((outer_mean * outer_area) - (guard_mean * guard_area)) / training_area
    local_sq_mean = ((outer_sq_mean * outer_area) - (guard_sq_mean * guard_area)) / training_area
    local_var = np.maximum(local_sq_mean - (local_mean * local_mean), 0.0)
    local_std = np.sqrt(local_var + float(epsilon)).astype(np.float32, copy=False)
    score = np.maximum((evidence - local_mean) / (local_std + float(epsilon)), 0.0).astype(np.float32, copy=False)
    return score[0] if was_2d else score


def contrast_frame_to_gray8(frame: Any, *, lo: float, hi: float) -> bytes:
    np = _load_numpy()
    arr = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite) or hi <= lo:
        return bytes(arr.size)
    clipped = np.clip(np.where(finite, arr, lo), lo, hi)
    return np.round((clipped - lo) * 255.0 / (hi - lo)).astype(np.uint8).tobytes()


def estimate_contrast_normalization(
    stack: Any,
    *,
    guard_px: int,
    training_radius_px: int,
    epsilon: float = 1e-6,
    chunk_frames: int = 10,
    sample_stride: int = 10,
    percentile: float = 99.5,
) -> dict[str, Any]:
    np = _load_numpy()
    frame_count = int(stack.shape[0])
    sample_stride = max(1, int(sample_stride))
    chunk_frames = max(1, int(chunk_frames))
    sample_indices = list(range(0, frame_count, sample_stride)) or [0]
    p_values: list[float] = []
    max_values: list[float] = []

    for start in range(0, len(sample_indices), chunk_frames):
        indices = sample_indices[start : start + chunk_frames]
        block = np.asarray(stack[indices], dtype=np.float32)
        score = compute_cfar_contrast_block(
            block,
            guard_px=guard_px,
            training_radius_px=training_radius_px,
            epsilon=epsilon,
        )
        finite = score[np.isfinite(score)]
        if finite.size:
            p_values.append(float(np.percentile(finite, percentile)))
            max_values.append(float(np.max(finite)))

    hi = max(p_values) if p_values else 1.0
    if hi <= 0 and max_values:
        hi = max(max_values)
    if hi <= 0:
        hi = 1.0
    return {
        "lo": 0.0,
        "hi": float(hi),
        "percentile": float(percentile),
        "sample_stride": int(sample_stride),
        "sampled_frames": len(sample_indices),
        "sample_max": float(max(max_values)) if max_values else None,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def export_cfar_contrast_frames(
    *,
    source_npy: Path,
    out_dir: Path,
    guard_px: int,
    training_radius_px: int,
    epsilon: float = 1e-6,
    chunk_frames: int = 10,
    sample_stride: int = 10,
    frame_pattern: str = "frame_%03d.png",
    force: bool = False,
) -> dict[str, Any]:
    np = _load_numpy()
    source_npy = source_npy.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "contrast_summary.json"
    stack = np.load(source_npy, mmap_mode="r")
    if stack.ndim != 3:
        raise ValueError(f"Expected a TxHxW source stack, got shape {stack.shape}.")
    frame_count, height, width = (int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2]))
    first = frame_output_path(out_dir, frame_pattern, 1)
    last = frame_output_path(out_dir, frame_pattern, frame_count)
    if not force and summary_path.exists() and first.exists() and last.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    normalization = estimate_contrast_normalization(
        stack,
        guard_px=guard_px,
        training_radius_px=training_radius_px,
        epsilon=epsilon,
        chunk_frames=chunk_frames,
        sample_stride=sample_stride,
    )
    lo = float(normalization["lo"])
    hi = float(normalization["hi"])
    chunk_frames = max(1, int(chunk_frames))
    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        block = np.asarray(stack[start:stop], dtype=np.float32)
        score = compute_cfar_contrast_block(
            block,
            guard_px=guard_px,
            training_radius_px=training_radius_px,
            epsilon=epsilon,
        )
        for offset, frame in enumerate(score, start=start + 1):
            write_png_gray8(
                frame_output_path(out_dir, frame_pattern, offset),
                width,
                height,
                contrast_frame_to_gray8(frame, lo=lo, hi=hi),
            )

    summary = {
        "schema_version": 1,
        "artifact_kind": "cfar_contrast_map",
        "source": str(source_npy),
        "shape": [frame_count, height, width],
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "guard_px": int(guard_px),
        "training_radius_px": int(training_radius_px),
        "epsilon": float(epsilon),
        "normalization": normalization,
    }
    write_json_atomic(summary_path, summary)
    return summary
