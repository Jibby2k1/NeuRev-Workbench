"""Standalone diagnostic videos for fixed multi-hypothesis CFAR experts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core as v1
from .multihypothesis import (
    MultiCFARConfig,
    _legacy_config,
    _quiet_windows,
    _score_maps,
    _thresholds,
    build_kernel_bank,
    expert_matrix,
)


def _slug(expert_id: str) -> str:
    return (
        expert_id.replace("sector_censored", "censored")
        .replace("causal_coherence", "coherence")
        .replace("_", "-")
    )


def _colorize(score: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib

    finite = score[np.isfinite(score)]
    upper = float(np.percentile(finite, 99.95)) if finite.size else threshold + 1
    upper = max(upper, float(score.max()), threshold + 1e-6)
    lower = min(threshold, float(np.percentile(finite, 97.0)))
    strength = np.clip((score - lower) / max(upper - lower, 1e-6), 0, 1)
    rgba = matplotlib.colormaps["magma"](strength)
    return np.asarray(rgba[..., :3] * 255, dtype=np.uint8), strength.astype(np.float32)


def _render_frame(
    raw: np.ndarray,
    heat_rgb: np.ndarray,
    heat_strength: np.ndarray,
    *,
    display_lo: float,
    display_hi: float,
    labels: list[dict[str, Any]],
    peaks: list[tuple[float, int, int]],
    matches: list[tuple[int, float, int, int, float]],
    title: str,
    frame_ui: int,
) -> np.ndarray:
    gray = np.clip((raw.astype(np.float32) - display_lo) / max(display_hi - display_lo, 1), 0, 1)
    base = np.repeat((gray[..., None] * 255).astype(np.uint8), 3, axis=2)
    alpha = (0.62 * heat_strength)[..., None]
    blended = np.asarray(base * (1 - alpha) + heat_rgb * alpha, dtype=np.uint8)
    header_height = 50
    width = blended.shape[1] + blended.shape[1] % 2
    height = blended.shape[0] + header_height
    height += height % 2
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(Image.fromarray(blended), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    matched_peak_xy = {(x, y) for _, _, x, y, _ in matches}
    matched_label_indices = {index for index, _, _, _, _ in matches}

    for index, row in enumerate(labels):
        x, y = int(round(row["x_px"])), int(round(row["y_px"])) + header_height
        color = (80, 255, 120) if index in matched_label_indices else (40, 225, 255)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
    for _, x, y0 in peaks:
        y = y0 + header_height
        color = (80, 255, 120) if (x, y0) in matched_peak_xy else (255, 165, 30)
        draw.line((x - 6, y, x + 6, y), fill=color, width=2)
        draw.line((x, y - 6, x, y + 6), fill=color, width=2)

    draw.text((8, 6), f"{title}  |  UI frame {frame_ui}", fill="white", font=font)
    draw.text(
        (8, 25),
        f"cyan label   green match   orange candidate   detections {len(peaks)}   matches {len(matches)}/{len(labels)}",
        fill=(205, 205, 205),
        font=font,
    )
    return np.asarray(canvas)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("Cannot encode an empty diagnostic video")
    height, width = frames[0].shape[:2]
    temporary = path.with_name(path.stem + ".tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError("Diagnostic video frames changed shape")
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        return_code = process.wait()
    except Exception:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr[-2000:]}")
    temporary.replace(path)


def generate(
    config: MultiCFARConfig,
    *,
    results_json: Path,
    output_dir: Path,
    expert_id: str | None = None,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Regenerate one fixed expert and write one standalone MP4 per burst."""
    import torch

    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_dir}")
    if not results_json.is_file():
        raise FileNotFoundError(results_json)
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be in (0, 60]")
    results = json.loads(results_json.read_text(encoding="utf-8"))
    selected_id = expert_id or results["checkpoint"]["best_single_diagnostic"]
    all_specs = expert_matrix(config.radii_px)
    selected = next((spec for spec in all_specs if spec.expert_id == selected_id), None)
    if selected is None:
        raise ValueError(f"Unknown expert_id: {selected_id}")

    labels = v1.load_labels(config.labels_tsv)
    quiet, bursts, _ = v1._prepare_arrays(_legacy_config(config), labels)
    bank = build_kernel_bank([selected], config.support_px)
    quiet_maps = [_score_maps(window, [selected], bank, config) for window in _quiet_windows(quiet)]
    thresholds, _ = _thresholds(quiet_maps, config)
    regenerated_threshold = float(thresholds[0])
    expected = next(item for item in results["fixed_experts"] if item["expert_id"] == selected_id)
    expected_threshold = float(expected["folds"][0]["threshold"])
    threshold_delta = abs(regenerated_threshold - expected_threshold)
    if threshold_delta > 1e-4:
        raise RuntimeError(
            f"Regenerated threshold {regenerated_threshold} != recorded {expected_threshold}"
        )
    threshold = expected_threshold

    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    q0, q1 = config.quiet_start_ui - 1, config.quiet_end_ui
    display_lo, display_hi = np.percentile(np.asarray(source[q0:q1:4]), [1.0, 99.8])
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for burst_id in sorted(bursts):
        rows = [row for row in labels if row["burst_id"] == burst_id]
        score = _score_maps(bursts[burst_id], [selected], bank, config)[0]
        peaks = v1._peaks(score, config.nms_distance_px, threshold=threshold, limit=2000)
        matches = v1._match(peaks, rows, config.tolerance_px)
        heat_rgb, heat_strength = _colorize(score, threshold)
        start = rows[0]["start_frame_zero"]
        stop = rows[0]["stop_frame_zero_exclusive"]
        title = f"Burst {burst_id} | {_slug(selected_id)}"
        rendered = [
            _render_frame(
                source[index], heat_rgb, heat_strength,
                display_lo=float(display_lo), display_hi=float(display_hi), labels=rows,
                peaks=peaks, matches=matches, title=title, frame_ui=index + 1,
            )
            for index in range(start, stop)
        ]
        path = output_dir / f"{_slug(selected_id)}_b{burst_id:02d}.mp4"
        _write_mp4(path, rendered, fps)
        records.append({
            "burst_id": burst_id, "path": str(path), "frames": len(rendered),
            "fps": fps, "candidate_count": len(peaks), "matched": len(matches),
            "labels": len(rows), "threshold": threshold, "bytes": path.stat().st_size,
        })
    payload = {
        "schema_version": 1,
        "generated_at": v1.utc_now(),
        "mode": "standalone_raw_with_static_temporally_pooled_evidence",
        "not_side_by_side": True,
        "source_results": str(results_json),
        "source_video": str(config.source_video),
        "expert_id": selected_id,
        "threshold": threshold,
        "regenerated_threshold": regenerated_threshold,
        "threshold_absolute_delta": threshold_delta,
        "threshold_reproduction_tolerance": 1e-4,
        "legend": {"cyan": "known sparse-positive label", "green": "matched detection", "orange": "unmatched candidate; truth unknown"},
        "videos": records,
        "resources": {"gpu_name": torch.cuda.get_device_name(0), "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() // 2**20},
    }
    v1.atomic_json(output_dir / "manifest.json", payload)
    return payload
