"""Background estimation helpers for calcium-imaging video stacks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def kalman_positive_residual_stack(
    source_npy: Path,
    out_dir: Path,
    *,
    residual_name: str = "positive_residual.npy",
    baseline_init_frames: int = 50,
    kalman_gain: float = 0.01,
    positive_update_gain: float = 0.002,
    negative_update_gain: float = 0.08,
    chunk_frames: int = 64,
    write_baseline_stack: bool = False,
) -> dict[str, Any]:
    """Write an adaptive-baseline positive residual stack for a video.

    The baseline update is intentionally asymmetric. Positive innovations are
    learned slowly so fast calcium events stay in the residual, while negative
    innovations are learned more quickly so background drops do not linger.
    """
    import numpy as np

    source_path = Path(source_npy)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = np.load(source_path, mmap_mode="r")
    if video.ndim != 3:
        raise ValueError(f"Expected a 3-D video stack at {source_path}, got shape {video.shape}")
    if baseline_init_frames < 1:
        raise ValueError("baseline_init_frames must be at least 1")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be at least 1")
    for name, value in {
        "kalman_gain": kalman_gain,
        "positive_update_gain": positive_update_gain,
        "negative_update_gain": negative_update_gain,
    }.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    frame_count, height, width = (int(video.shape[0]), int(video.shape[1]), int(video.shape[2]))
    init_count = min(int(baseline_init_frames), frame_count)
    baseline = np.median(np.asarray(video[:init_count], dtype=np.float32), axis=0).astype(np.float32)
    initial_baseline = baseline.copy()

    residual_path = output_dir / residual_name
    residual = np.lib.format.open_memmap(residual_path, mode="w+", dtype=np.float32, shape=video.shape)
    baseline_path: Path | None = None
    baseline_stack = None
    if write_baseline_stack:
        baseline_path = output_dir / "baseline_stack.npy"
        baseline_stack = np.lib.format.open_memmap(baseline_path, mode="w+", dtype=np.float32, shape=video.shape)

    positive_projection_max = np.zeros((height, width), dtype=np.float32)
    positive_projection_sum = np.zeros((height, width), dtype=np.float64)
    frame_mean: list[float] = []
    frame_p99: list[float] = []
    positive_fraction: list[float] = []

    for start in range(0, frame_count, int(chunk_frames)):
        stop = min(start + int(chunk_frames), frame_count)
        block = np.asarray(video[start:stop], dtype=np.float32)
        for offset, frame in enumerate(block):
            index = start + offset
            innovation = frame - baseline
            positive = np.maximum(innovation, 0.0)
            residual[index] = positive
            if baseline_stack is not None:
                baseline_stack[index] = baseline
            positive_projection_max = np.maximum(positive_projection_max, positive)
            positive_projection_sum += positive
            frame_mean.append(float(np.mean(positive)))
            frame_p99.append(float(np.percentile(positive, 99.0)))
            positive_fraction.append(float(np.count_nonzero(positive > 0.0) / positive.size))

            gain = np.where(innovation > 0.0, float(positive_update_gain), float(negative_update_gain)).astype(np.float32)
            if kalman_gain:
                gain = np.minimum(gain + float(kalman_gain), 1.0)
            baseline += gain * innovation

    residual.flush()
    if baseline_stack is not None:
        baseline_stack.flush()

    final_baseline_path = output_dir / "final_baseline.npy"
    initial_baseline_path = output_dir / "initial_baseline.npy"
    projection_max_path = output_dir / "positive_residual_max_projection.npy"
    projection_mean_path = output_dir / "positive_residual_mean_projection.npy"
    np.save(final_baseline_path, baseline.astype(np.float32, copy=False))
    np.save(initial_baseline_path, initial_baseline.astype(np.float32, copy=False))
    np.save(projection_max_path, positive_projection_max.astype(np.float32, copy=False))
    np.save(projection_mean_path, (positive_projection_sum / max(frame_count, 1)).astype(np.float32))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "source_npy": str(source_path),
        "residual_npy": str(residual_path),
        "shape": [frame_count, height, width],
        "dtype": "float32",
        "params": {
            "baseline_init_frames": int(init_count),
            "kalman_gain": float(kalman_gain),
            "positive_update_gain": float(positive_update_gain),
            "negative_update_gain": float(negative_update_gain),
            "chunk_frames": int(chunk_frames),
            "write_baseline_stack": bool(write_baseline_stack),
        },
        "artifacts": {
            "initial_baseline_npy": str(initial_baseline_path),
            "final_baseline_npy": str(final_baseline_path),
            "positive_residual_max_projection_npy": str(projection_max_path),
            "positive_residual_mean_projection_npy": str(projection_mean_path),
        },
        "residual_stats": {
            "frame_mean_min": float(min(frame_mean)) if frame_mean else 0.0,
            "frame_mean_median": float(np.median(np.asarray(frame_mean, dtype=np.float32))) if frame_mean else 0.0,
            "frame_mean_max": float(max(frame_mean)) if frame_mean else 0.0,
            "frame_p99_median": float(np.median(np.asarray(frame_p99, dtype=np.float32))) if frame_p99 else 0.0,
            "positive_fraction_median": float(np.median(np.asarray(positive_fraction, dtype=np.float32))) if positive_fraction else 0.0,
        },
    }
    if baseline_path is not None:
        summary["artifacts"]["baseline_stack_npy"] = str(baseline_path)

    summary_path = output_dir / "background_residual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
