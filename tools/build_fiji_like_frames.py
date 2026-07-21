"""Render review-dashboard video frames with a stable Fiji-like display window."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.workbench.intermediates import write_png_gray8


DEFAULT_PERCENTILES = (0.2, 99.5)


def _frame_path(out_dir: Path, pattern: str, index: int) -> Path:
    if "%" in pattern:
        return out_dir / (pattern % index)
    return out_dir / pattern.replace("{frame:03d}", f"{index:03d}").replace("{frame}", str(index))


def _sample_indices(frame_count: int, sample_frames: int) -> np.ndarray:
    count = max(1, min(int(sample_frames), int(frame_count)))
    return np.unique(np.linspace(0, frame_count - 1, count, dtype=np.int64))


def _stack_percentiles(stack: np.ndarray, indices: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    values = np.asarray(stack[indices], dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute display limits from a stack with no finite values.")
    computed = np.percentile(values, percentiles)
    return {f"{p:g}": float(v) for p, v in zip(percentiles, computed)}


def _render_frame(frame: np.ndarray, low: float, high: float, gamma: float) -> bytes:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D frames, got shape {arr.shape}.")
    finite = np.isfinite(arr)
    safe = np.where(finite, arr, low)
    scaled = np.clip((safe - low) / (high - low), 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    return np.round(scaled * 255.0).astype(np.uint8).tobytes()


def render_fiji_like_frames(
    *,
    source_npy: Path,
    out_dir: Path,
    frame_pattern: str = "frame_%03d.png",
    low_percentile: float = DEFAULT_PERCENTILES[0],
    high_percentile: float = DEFAULT_PERCENTILES[1],
    sample_frames: int = 300,
    gamma: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    stack = np.load(source_npy, mmap_mode="r")
    if stack.ndim not in {3, 4}:
        raise ValueError(f"Expected a time stack with 3 or 4 dimensions, got shape {stack.shape}.")
    if high_percentile <= low_percentile:
        raise ValueError("--high-percentile must be greater than --low-percentile.")
    if gamma <= 0:
        raise ValueError("--gamma must be positive.")

    frame_count = int(stack.shape[0])
    height, width = map(int, stack.shape[1:3])
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _frame_path(out_dir, frame_pattern, frame_count)
    if existing.exists() and not force:
        summary_path = out_dir / "display_window.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"{existing} already exists; use --force to regenerate frames.")

    sample = _sample_indices(frame_count, sample_frames)
    diagnostic_percentiles = sorted(set([0.0, 0.1, low_percentile, 0.5, 1.0, 50.0, 95.0, 98.0, 99.0, high_percentile, 99.7, 99.8, 100.0]))
    percentiles = _stack_percentiles(stack, sample, diagnostic_percentiles)
    low = percentiles[f"{low_percentile:g}"]
    high = percentiles[f"{high_percentile:g}"]
    if high <= low:
        low = float(np.nanmin(np.asarray(stack[sample], dtype=np.float32)))
        high = float(np.nanmax(np.asarray(stack[sample], dtype=np.float32)))
    if high <= low:
        high = low + 1.0

    for index in range(frame_count):
        out = _frame_path(out_dir, frame_pattern, index + 1)
        write_png_gray8(out, width, height, _render_frame(stack[index], low, high, gamma))

    summary = {
        "schema_version": 1,
        "rendering": "fiji_like_stack_window",
        "source_npy": str(source_npy),
        "shape": [int(v) for v in stack.shape],
        "dtype": str(stack.dtype),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frame_pattern": frame_pattern,
        "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "low": float(low),
        "high": float(high),
        "gamma": float(gamma),
        "sample_frame_count": int(sample.size),
        "sample_first_frame": int(sample[0] + 1),
        "sample_last_frame": int(sample[-1] + 1),
        "percentiles": percentiles,
    }
    tmp = out_dir / "display_window.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / "display_window.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npy", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frame-pattern", default="frame_%03d.png")
    parser.add_argument("--low-percentile", type=float, default=DEFAULT_PERCENTILES[0])
    parser.add_argument("--high-percentile", type=float, default=DEFAULT_PERCENTILES[1])
    parser.add_argument("--sample-frames", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = render_fiji_like_frames(
        source_npy=args.source_npy,
        out_dir=args.out_dir,
        frame_pattern=args.frame_pattern,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        sample_frames=args.sample_frames,
        gamma=args.gamma,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
