"""Guarded label-free Raw -> multi-lag MSICA -> MSLN inference for TIFF stacks."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

from neurobench.algorithms.multilag_msica import TemporalMSICAFit
from neurobench.algorithms.multiscale_local_normalization import JointSTContext
from neurobench.algorithms.msln_msica_cuda import causal_joint_msln_cuda, cuda_device_summary
from neurobench.annotations import default_annotations_v3
from neurobench.experiments.msln_msica.annotation_dashboard import _attach_traces, _render_frames
from neurobench.experiments.msln_msica.artifacts import atomic_json, sha256_file
from neurobench.experiments.msln_msica.multilag_program import _sample_fit
from neurobench.metrics.sparse_detection import extract_local_maxima
from neurobench.workbench.annotation_revisions import initialize_revision_root
from neurobench.workbench.builder import build_workbench
from neurobench.workbench.model_proposals import build_model_proposal_package

FROZEN_CONFIG_ID = "multilag_2d__normalized_hsic__short__uniform__bandwidth_scale-0p5"
FROZEN_LANE = FROZEN_CONFIG_ID + "::persistence::joint_s5_g1_t31_g1"
CONTEXT = JointSTContext("joint_s5_g1_t31_g1", 5, 1, 31, 1, "mean_std", 10.0)
DISPLAY_FPS = 10.0
CHUNK_FRAMES = 96
EVENT_LIMIT = 8
EVENT_HALF_WIDTH = 20
EVENT_MIN_GAP = 100
OCCURRENCES_PER_EVENT = 58
AUDIT_IDENTITY_LIMIT = 32
NMS_DISTANCE = 6
BORDER = 8
VRAM_CAP = 8 * 2**30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _heartbeat(root: Path, stage: str, **extra: Any) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), "stage": stage, **extra}, sort_keys=True) + "\n")


def _fit_config(seed: int) -> dict[str, Any]:
    return {
        "design": {"seed": int(seed)},
        "fitting": {
            "estimator_sample_caps": {"normalized_hsic": 256},
            "surface_screen_samples": 4096, "surface_confirmation_samples": 16384,
            "coarse_angle_step_degrees": 3.0, "refine_half_width_degrees": 3.0,
            "refine_step_degrees": 0.25, "objective_sharpness_delta_degrees": 2.0,
            "eigenvalue_floor_ratio": 1e-6,
        },
    }


def _source(path: Path) -> np.memmap:
    values = tifffile.memmap(path)
    if values.ndim != 3 or values.shape[0] < 200 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"expected integer TYX TIFF, got {values.shape} {values.dtype}")
    return values


def _temporal_scores(values: np.ndarray, root: Path) -> tuple[np.ndarray, np.ndarray]:
    target = root / "frame_difference_scores.npy"
    if target.is_file():
        scores = np.load(target)
    else:
        scores = np.zeros(len(values), dtype=np.float32)
        previous = np.asarray(values[0, ::8, ::8], dtype=np.float32)
        for start in range(1, len(values), 256):
            stop = min(len(values), start + 256)
            block = np.asarray(values[start:stop, ::8, ::8], dtype=np.float32)
            joined = np.concatenate((previous[None], block), axis=0)
            scores[start:stop] = np.median(np.abs(np.diff(joined, axis=0)), axis=(1, 2))
            previous = block[-1]
            if stop % 1024 < 256 or stop == len(values):
                _heartbeat(root, "temporal_scores", completed=stop, total=len(values))
        np.save(target, scores)
    cutoff = float(np.percentile(scores[1:], 25.0))
    quiet = scores <= cutoff
    quiet[0] = False
    return scores, quiet


def _project_msica(values: np.ndarray, fit: TemporalMSICAFit, target: Path, root: Path) -> np.memmap:
    expected = (len(values) - 1, values.shape[1], values.shape[2])
    if target.is_file():
        out = np.load(target, mmap_mode="r+")
        if out.shape != expected:
            raise RuntimeError("MSICA cache shape mismatch")
        return out
    partial = target.with_suffix(".partial.npy")
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected)
    import cupy as cp
    center = cp.asarray(fit.center, dtype=cp.float32).reshape((2, 1))
    demixing = cp.asarray(fit.demixing, dtype=cp.float32)
    for start in range(0, expected[0], CHUNK_FRAMES):
        count = min(CHUNK_FRAMES, expected[0] - start)
        source = cp.asarray(np.asarray(values[start:start + count + 1]), dtype=cp.float32)
        stack = cp.stack((source[:-1], source[1:]), axis=0)
        transformed = (demixing @ (stack.reshape((2, -1)) - center)).reshape(
            (2, count, expected[1], expected[2])
        )
        out[start:start + count] = cp.asnumpy(transformed[fit.persistence_index])
        out.flush()
        del source, stack, transformed
        cp.get_default_memory_pool().free_all_blocks()
        _heartbeat(root, "msica_projection", completed=start + count, total=expected[0])
    del out
    partial.replace(target)
    return np.load(target, mmap_mode="r+")


def _pilot_floor(msica: np.ndarray, quiet: np.ndarray, root: Path) -> tuple[float, list[dict[str, Any]]]:
    starts = list(range(0, len(msica) - 95, 128))
    ranked = sorted(starts, key=lambda s: float(np.mean(quiet[s + 31:s + 95])), reverse=True)
    selected: list[int] = []
    for start in ranked:
        if all(abs(start - prior) >= 256 for prior in selected):
            selected.append(start)
        if len(selected) == 8:
            break
    floors: list[float] = []
    rows: list[dict[str, Any]] = []
    import cupy as cp
    for index, start in enumerate(sorted(selected), 1):
        stop = start + 95
        mask = np.asarray(quiet[start:stop], dtype=bool)
        result = causal_joint_msln_cuda(
            np.asarray(msica[start:stop], dtype=np.float32), CONTEXT,
            quiet_mask=mask, review_crop_frames=31, max_vram_bytes=VRAM_CAP,
        )
        floors.append(float(result.scale_floor))
        rows.append({"start_msica_zero": start, "stop_msica_half_open": stop,
                     "quiet_fraction": float(mask[31:].mean()), "scale_floor": float(result.scale_floor)})
        del result
        cp.get_default_memory_pool().free_all_blocks()
        _heartbeat(root, "msln_floor_pilot", completed=index, total=len(selected))
    floor = float(np.median(floors))
    if not np.isfinite(floor) or floor <= 0:
        raise RuntimeError("invalid global MSLN floor")
    return floor, rows


def _project_msln(
    msica: np.ndarray, quiet: np.ndarray, target: Path, score_target: Path,
    root: Path, floor: float,
) -> tuple[np.memmap, np.ndarray, list[dict[str, Any]]]:
    expected = (len(msica) - 31, msica.shape[1], msica.shape[2])
    if target.is_file() and score_target.is_file():
        return np.load(target, mmap_mode="r+"), np.load(score_target), []
    partial = target.with_suffix(".partial.npy")
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected)
    scores = np.zeros(expected[0], dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []
    import cupy as cp
    for start in range(0, expected[0], CHUNK_FRAMES):
        count = min(CHUNK_FRAMES, expected[0] - start)
        block = np.asarray(msica[start:start + 31 + count], dtype=np.float32)
        mask = np.asarray(quiet[start:start + 31 + count], dtype=bool)
        normalized = causal_joint_msln_cuda(
            block, CONTEXT, quiet_mask=mask, review_crop_frames=31,
            max_vram_bytes=VRAM_CAP, scale_floor_override=floor,
        )
        chunk = normalized.values
        out[start:start + count] = cp.asnumpy(chunk)
        scores[start:start + count] = cp.asnumpy(
            cp.percentile(cp.square(chunk[:, ::4, ::4], dtype=cp.float32), 99.5, axis=(1, 2))
        )
        diagnostics.append({**normalized.diagnostics, "output_start_zero": start,
                            "output_stop_half_open": start + count})
        out.flush()
        del normalized, chunk
        cp.get_default_memory_pool().free_all_blocks()
        _heartbeat(root, "msln_projection", completed=start + count, total=expected[0])
    del out
    partial.replace(target)
    np.save(score_target, scores)
    return np.load(target, mmap_mode="r+"), scores, diagnostics


def _event_windows(scores: np.ndarray) -> list[dict[str, Any]]:
    median = float(np.median(scores))
    scale = max(1.4826 * float(np.median(np.abs(scores - median))), float(np.finfo(np.float32).eps))
    threshold = median + 4.0 * scale
    selected: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if scores[index] < threshold and len(selected) >= 4:
            break
        if index < EVENT_HALF_WIDTH or index >= len(scores) - EVENT_HALF_WIDTH:
            continue
        if all(abs(int(index) - prior) >= EVENT_MIN_GAP for prior in selected):
            selected.append(int(index))
        if len(selected) == EVENT_LIMIT:
            break
    if not selected:
        selected = [int(np.argmax(scores))]
    rows = []
    for event_id, peak in enumerate(sorted(selected), 1):
        start = max(0, peak - EVENT_HALF_WIDTH)
        stop = min(len(scores), peak + EVENT_HALF_WIDTH + 1)
        rows.append({"event_id": event_id, "peak_index_zero": peak, "peak_ui": peak + 33,
                     "start_index_zero": start, "stop_index_half_open": stop,
                     "start_ui": start + 33, "end_ui": stop + 32,
                     "score": float(scores[peak]), "threshold": threshold,
                     "selection": "robust_global_energy_peak"})
    return rows


def _proposals(
    msln: np.ndarray, events: list[dict[str, Any]], root: Path,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    pooled: dict[int, np.ndarray] = {}
    pooled_root = root / "pooled_detection_maps"
    pooled_root.mkdir(exist_ok=True)
    for event in events:
        event_id = int(event["event_id"])
        start, stop = int(event["start_index_zero"]), int(event["stop_index_half_open"])
        image = np.square(np.asarray(msln[start:stop], dtype=np.float32), dtype=np.float32).max(axis=0)
        image[:BORDER] = 0; image[-BORDER:] = 0; image[:, :BORDER] = 0; image[:, -BORDER:] = 0
        peaks = extract_local_maxima(image, NMS_DISTANCE, limit=OCCURRENCES_PER_EVENT)
        np.save(pooled_root / f"event_{event_id:02d}.npy", image)
        pooled[event_id] = image
        for rank, (score, x, y) in enumerate(peaks, 1):
            rows.append({"event_id": event_id, "rank": rank, "score": float(score),
                         "x": int(x), "y": int(y), "start_ui": int(event["start_ui"]),
                         "end_ui": int(event["end_ui"])})
    return rows, pooled


def _consolidate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: -float(item["score"])):
        match = next((item for item in groups if
                      (float(row["x"]) - item["x"]) ** 2 + (float(row["y"]) - item["y"]) ** 2 <= NMS_DISTANCE ** 2), None)
        if match is None:
            match = {"x": float(row["x"]), "y": float(row["y"]), "members": [], "weight": 0.0}
            groups.append(match)
        weight = max(float(row["score"]), 1e-6)
        total = match["weight"] + weight
        match["x"] = (match["x"] * match["weight"] + float(row["x"]) * weight) / total
        match["y"] = (match["y"] * match["weight"] + float(row["y"]) * weight) / total
        match["weight"] = total
        match["members"].append(dict(row))
    groups.sort(key=lambda item: (-len({m["event_id"] for m in item["members"]}),
                                  -max(m["score"] for m in item["members"]), item["y"], item["x"]))
    models: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    for index, group in enumerate(groups[:AUDIT_IDENTITY_LIMIT], 1):
        model_id = f"model_roi_{index:03d}"
        members = sorted(group["members"], key=lambda item: (item["event_id"], item["rank"]))
        for member in members:
            member.update({"burst": int(member["event_id"]), "model_roi": model_id,
                           "matched_expert_roi": "", "match_distance_px": None,
                           "expert_supported": False})
            occurrences.append(member)
        intervals = [[int(m["start_ui"]), int(m["end_ui"])] for m in members]
        models.append({"id": model_id, "source_xy": [round(group["x"], 3), round(group["y"], 3)],
                       "ui_frame": intervals[0][0], "events": [x[0] for x in intervals],
                       "eventIntervals": intervals, "event_intervals": intervals,
                       "geometry": {"kind": "center"}, "status": "unknown", "members": members,
                       "recurrence_events": len({m["event_id"] for m in members}),
                       "best_score": max(float(m["score"]) for m in members)})
    return models, occurrences


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model_roi"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _limits(values: np.ndarray, kind: str) -> tuple[float, float]:
    indices = np.unique(np.linspace(0, len(values) - 1, min(24, len(values))).astype(int))
    sample = np.asarray(values[indices, ::8, ::8], dtype=np.float32)
    if kind == "signed":
        limit = max(float(np.percentile(np.abs(sample), 99.5)), 1e-6)
        return -limit, limit
    if kind == "energy":
        return 0.0, max(float(np.percentile(sample, 99.5)), 1e-6)
    low, high = map(float, np.percentile(sample, [1.0, 99.8]))
    return low, max(high, low + 1e-6)


def _gray(frame: np.ndarray, limits: tuple[float, float], size: tuple[int, int]) -> Image.Image:
    low, high = limits
    out = np.clip((np.asarray(frame, dtype=np.float32) - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L").resize(size, Image.Resampling.BILINEAR).convert("RGB")


def _encode(path: Path, frames: Iterable[np.ndarray], width: int, height: int, total: int, root: Path, stage: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(DISPLAY_FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    started = time.monotonic()
    try:
        for index, frame in enumerate(frames, 1):
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            if index % 250 == 0 or index == total:
                _heartbeat(root, stage, completed=index, total=total)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        code = process.wait()
    except Exception:
        process.kill()
        raise
    if code:
        raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")
    partial.replace(path)
    return {"path": str(path), "frames": total, "fps": DISPLAY_FPS,
            "duration_seconds": total / DISPLAY_FPS, "width": width, "height": height,
            "runtime_seconds": time.monotonic() - started}


def _draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, width: int, height: int,
                 source_width: int, source_height: int, x_offset: int, y_offset: int) -> None:
    cx = x_offset + int(round(x * width / source_width))
    cy = y_offset + int(round(y * height / source_height))
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=(255, 145, 40), width=2)


def _full_video(path: Path, raw: np.ndarray, msica: np.ndarray, msln: np.ndarray,
                events: list[dict[str, Any]], pooled: dict[int, np.ndarray],
                models: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    pw, ph, header = 240, 168, 42
    raw_lim, msica_lim, msln_lim = _limits(raw, "raw"), _limits(msica, "raw"), _limits(msln, "signed")
    evidence_lim = (0.0, msln_lim[1] ** 2)
    pooled_lim = (0.0, max(float(np.percentile(item, 99.8)) for item in pooled.values()))
    labels = ("Raw", "MSICA persistence", "Signed MSLN", "Squared MSLN", "Event-pooled map")
    font = ImageFont.load_default()
    def frames() -> Iterable[np.ndarray]:
        blank = np.zeros(raw.shape[1:], dtype=np.float32)
        for frame_index in range(len(raw)):
            event = next((item for item in events if item["start_index_zero"] <= frame_index < item["stop_index_half_open"]), None)
            pool = pooled[int(event["event_id"])] if event else blank
            arrays = (raw[frame_index], msica[frame_index], msln[frame_index],
                      np.square(np.asarray(msln[frame_index], dtype=np.float32)), pool)
            limits = (raw_lim, msica_lim, msln_lim, evidence_lim, pooled_lim)
            canvas = Image.new("RGB", (pw * 5, ph + header), "black")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 4), f"Raw -> MSICA -> MSLN | UI frame {frame_index + 33} | orange=model only", fill="white", font=font)
            active = [] if event is None else [
                model for model in models
                if any(int(member["event_id"]) == int(event["event_id"]) for member in model["members"])
            ]
            for column, (array, limit, label) in enumerate(zip(arrays, limits, labels, strict=True)):
                canvas.paste(_gray(array, limit, (pw, ph)), (column * pw, header))
                draw.text((column * pw + 5, 22), label, fill="white", font=font)
                for model in active:
                    _draw_circle(draw, *model["source_xy"], pw, ph, raw.shape[2], raw.shape[1], column * pw, header)
            yield np.asarray(canvas, dtype=np.uint8)
    return _encode(path, frames(), pw * 5, ph + header, len(raw), root, "full_field_video")


def _crop(x: float, y: float, width: int, height: int, radius: int = 40) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - (2 * radius + 1), int(round(x)) - radius))
    y0 = max(0, min(height - (2 * radius + 1), int(round(y)) - radius))
    return x0, y0, x0 + 2 * radius + 1, y0 + 2 * radius + 1


def _closeup(path: Path, raw: np.ndarray, msica: np.ndarray, msln: np.ndarray,
             model: dict[str, Any], root: Path) -> dict[str, Any]:
    panel, header = 180, 42
    x, y = map(float, model["source_xy"])
    x0, y0, x1, y1 = _crop(x, y, raw.shape[2], raw.shape[1])
    arrays = (raw, msica, msln)
    limits = (_limits(raw, "raw"), _limits(msica, "raw"), _limits(msln, "signed"))
    labels = ("Raw close-up", "MSICA persistence", "Signed MSLN")
    font = ImageFont.load_default()
    def frames() -> Iterable[np.ndarray]:
        for frame_index in range(len(raw)):
            canvas = Image.new("RGB", (panel * 3, panel + header), "black")
            draw = ImageDraw.Draw(canvas)
            active = any(start <= frame_index + 33 <= stop for start, stop in model["eventIntervals"])
            draw.text((8, 4), f"{model['id']} | UI frame {frame_index + 33} | {'event window' if active else 'context'}", fill="white", font=font)
            for column, (values, limit, label) in enumerate(zip(arrays, limits, labels, strict=True)):
                canvas.paste(_gray(values[frame_index, y0:y1, x0:x1], limit, (panel, panel)), (column * panel, header))
                draw.text((column * panel + 5, 22), label, fill="white", font=font)
                local_x = (x - x0) * panel / (x1 - x0)
                local_y = (y - y0) * panel / (y1 - y0)
                draw.ellipse((column * panel + local_x - 7, header + local_y - 7,
                              column * panel + local_x + 7, header + local_y + 7),
                             outline=(255, 145, 40), width=2)
            yield np.asarray(canvas, dtype=np.uint8)
    return _encode(path, frames(), panel * 3, panel + header, len(raw), root, "closeup_" + model["id"])


def _trace(path: Path, model: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    for axis, view, title in zip(
        axes, ("raw", "msica", "msln"),
        ("Raw exact-pixel intensity", "MSICA persistence exact-pixel", "Signed MSLN exact-pixel"),
        strict=True,
    ):
        values = model["traces"][view]["pixel"]
        axis.plot(np.arange(len(values)) + 33, values, color="#333333", linewidth=0.8)
        for start, stop in model["eventIntervals"]:
            axis.axvspan(start, stop, color="#f28e2b", alpha=0.15)
        axis.set_ylabel(title); axis.grid(alpha=0.2)
    axes[-1].set_xlabel("UI frame (one-based)")
    fig.suptitle(f"{model['id']} at x={model['source_xy'][0]:.1f}, y={model['source_xy'][1]:.1f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _instant(
    path: Path, raw: np.ndarray, msica: np.ndarray, msln: np.ndarray,
    occurrence: dict[str, Any], event: dict[str, Any], pooled: np.ndarray,
    limits: tuple[tuple[float, float], ...],
) -> None:
    panel, header = 240, 34
    index = int(event["peak_index_zero"])
    arrays = (raw[index], msica[index], msln[index],
              np.square(np.asarray(msln[index], dtype=np.float32)), pooled)
    labels = ("Raw", "MSICA persistence", "Signed MSLN", "Squared MSLN", "Event-pooled map")
    canvas = Image.new("RGB", (panel * 5, int(round(panel * raw.shape[1] / raw.shape[2])) + header), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    image_height = canvas.height - header
    draw.text((8, 5), f"{occurrence['model_roi']} event {occurrence['event_id']} rank {occurrence['rank']} | UI frame {event['peak_ui']}", fill="white", font=font)
    for column, (array, limit, label) in enumerate(zip(arrays, limits, labels, strict=True)):
        canvas.paste(_gray(array, limit, (panel, image_height)), (column * panel, header))
        draw.text((column * panel + 5, 19), label, fill="white", font=font)
        _draw_circle(draw, float(occurrence["x"]), float(occurrence["y"]), panel, image_height,
                     raw.shape[2], raw.shape[1], column * panel, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _source_app(root: Path, input_tiff: Path, raw: np.ndarray, msica: np.ndarray,
                msln: np.ndarray, models: list[dict[str, Any]], fit: TemporalMSICAFit,
                events: list[dict[str, Any]], video: dict[str, Any]) -> Path:
    app = root / "app"
    app.mkdir(exist_ok=True)
    _attach_traces(models, {"raw": raw, "msica": msica, "msln": msln})
    frames = app / "frames"
    ranges = {
        "raw": _render_frames(raw, frames / "raw", 33, signed=False, resume=True),
        "msica": _render_frames(msica, frames / "msica", 33, signed=False, resume=True),
        "msln": _render_frames(msln, frames / "msln", 33, signed=True, resume=True),
    }
    dataset_id = root.name.replace("_multilag_msica_msln_inference_v1", "")
    revision_id = dataset_id + "_model_proposal_draft_v1"
    revision = {"schema_version": 1, "revisionId": revision_id, "parentRevisionId": "unlabeled_seed",
                "state": "draft", "reviewerId": "reviewer_local_1", "frozenRunId": FROZEN_LANE,
                "sourceAnnotationsSha256": hashlib.sha256(b"NO_EXPERT_ANNOTATIONS\n").hexdigest(),
                "createdAt": _now(), "updatedAt": _now(),
                "revisionToken": 0, "operationCount": 0}
    shape = list(raw.shape)
    contracts = [
        {"schema_version":1,"view_id":"raw","source_video_id":dataset_id,"shape_tyx":shape,
         "source_to_view":{"kind":"identity"},"frame_mapping":{"kind":"identity","offset":32},
         "intensity_semantics":"raw_amplitude","frame_pattern":"frames/raw/frame_%04d.png"},
        {"schema_version":1,"view_id":"msica","source_video_id":dataset_id,"shape_tyx":shape,
         "source_to_view":{"kind":"identity"},"frame_mapping":{"kind":"identity","offset":32},
         "intensity_semantics":"normalized_unsigned_visualization","frame_pattern":"frames/msica/frame_%04d.png"},
        {"schema_version":1,"view_id":"msln","source_video_id":dataset_id,"shape_tyx":shape,
         "source_to_view":{"kind":"identity"},"frame_mapping":{"kind":"identity","offset":32},
         "intensity_semantics":"normalized_signed_visualization","frame_pattern":"frames/msln/frame_%04d.png"},
    ]
    review = {
        "dataset":{"dataset_id":dataset_id,"name":input_tiff.stem + " label-free proposals","frame_rate_hz":None},
        "video":{"name":input_tiff.name,"width":raw.shape[2],"height":raw.shape[1],"frames":raw.shape[0],
                 "fps":DISPLAY_FPS,"framePattern":"frames/raw/frame_%04d.png"},
        "parameters":{"purpose":"single-reviewer label-free model proposal review","frozen_lane":FROZEN_LANE,
                      "event_source":"model_proposed","acquisition_frame_rate":"not_present_in_tiff_metadata",
                      "playback_fps_visualization_only":DISPLAY_FPS,
                      "candidate_surrogate_panel_limit":AUDIT_IDENTITY_LIMIT,
                      "event_windows":events,"fit":fit.to_dict(),"display_ranges":ranges,"display_video":video},
        "rois":[],
        "annotationCorrection":{"schema_version":1,"source_video_id":dataset_id,"read_only":False,
                                "revision":revision,"view_contracts":contracts,"expert_rois":[],
                                "model_rois":models,"matches":[]},
    }
    atomic_json(app / "review_data.json", review)
    build_workbench(app_dir=app, review_data_path=app / "review_data.json", dataset_id=dataset_id)
    revision_root = app / "annotation_revisions" / revision_id
    initialize_revision_root(app / "annotation_revisions", revision=revision, annotations=default_annotations_v3())
    shutil.copyfile(revision_root / "annotations.json", app / "annotations.json")
    return app


def _audit(package: Path, raw: np.ndarray, msica: np.ndarray, msln: np.ndarray,
           models: list[dict[str, Any]], occurrences: list[dict[str, Any]],
           events: list[dict[str, Any]], pooled: dict[int, np.ndarray], root: Path) -> dict[str, Any]:
    audit = package / "audit"
    model_root = audit / "2_Model_Annotations"
    (model_root / "videos" / "closeups").mkdir(parents=True, exist_ok=True)
    (model_root / "figures" / "traces").mkdir(parents=True, exist_ok=True)
    (model_root / "figures" / "instants").mkdir(parents=True, exist_ok=True)
    (model_root / "metadata").mkdir(parents=True, exist_ok=True)
    full = _full_video(model_root / "videos" / "model_only_full_field_sequential_stages.mp4",
                       raw, msica, msln, events, pooled, models, root)
    for index, model in enumerate(models, 1):
        close = _closeup(model_root / "videos" / "closeups" / (model["id"] + ".mp4"),
                         raw, msica, msln, model, root)
        trace_path = model_root / "figures" / "traces" / (model["id"] + ".png")
        _trace(trace_path, model)
        atomic_json(model_root / "metadata" / ("roi_" + model["id"] + ".json"),
                    {"model_roi":model["id"],"source_xy":model["source_xy"],
                     "event_intervals_ui":model["eventIntervals"],"coordinate_convention":"x=column,y=row",
                     "frame_convention":"UI one-based inclusive","closeup_video":close["path"],
                     "trace_figure":str(trace_path)})
        _heartbeat(root, "roi_audit", completed=index, total=len(models))
    _write_csv(model_root / "model_occurrences.csv", occurrences)
    _write_csv(model_root / "metadata" / "all_occurrences.csv", occurrences)
    event_by_id = {int(item["event_id"]): item for item in events}
    raw_lim, msica_lim, msln_lim = _limits(raw, "raw"), _limits(msica, "raw"), _limits(msln, "signed")
    pooled_lim = (0.0, max(float(np.percentile(item, 99.8)) for item in pooled.values()))
    instant_limits = (raw_lim, msica_lim, msln_lim, (0.0, msln_lim[1] ** 2), pooled_lim)
    for index, occurrence in enumerate(occurrences, 1):
        event_id = int(occurrence["event_id"])
        _instant(
            model_root / "figures" / "instants" /
            f"{occurrence['model_roi']}_event_{event_id:02d}_rank_{int(occurrence['rank']):03d}.png",
            raw, msica, msln, occurrence, event_by_id[event_id], pooled[event_id], instant_limits,
        )
        if index % 25 == 0 or index == len(occurrences):
            _heartbeat(root, "occurrence_instants", completed=index, total=len(occurrences))
    (model_root / "README.md").write_text(
        "# Model Annotations\n\nOrange markers are frozen model proposals only. The full-field video shows Raw, MSICA persistence, signed MSLN, squared MSLN evidence, and the event-pooled ranking map. Every audited identity has a full-duration close-up and exact-pixel Raw/MSICA/MSLN trace. No expert annotations were available.\n",
        encoding="utf-8")
    (model_root / "REPORT.md").write_text(
        f"# Model-only candidate-surrogate audit\n\n{len(models)} frozen model identities and "
        f"{len(occurrences)} selected occurrences are audited. Orange denotes model proposals; "
        "there are no expert markers. Every occurrence remains unknown pending review.\n",
        encoding="utf-8")
    atomic_json(model_root / "status.json",
                {"status":"complete","applicable":True,"model_identities":len(models),
                 "model_occurrences":len(occurrences),"full_field_video":full,
                 "closeups":len(models),"traces":len(models)})
    atomic_json(audit / "1_Expert_Annotations" / "status.json",
                {"status":"not_applicable_pending_labels","applicable":False,"reason":"No expert labels were supplied."})
    atomic_json(audit / "3_Comparison" / "status.json",
                {"status":"not_applicable_pending_labels","applicable":False,"reason":"Expert/model comparison requires expert labels."})
    summary = {"status":"complete","scientific_status":"unlabeled_visual_review_only",
               "frozen_lane":FROZEN_LANE,
               "stage_sequence":["Raw","MSICA persistence","signed MSLN","squared MSLN evidence","event-pooled detection map"],
               "event_source":"model_proposed","events":len(events),"model_identities":len(models),
               "model_occurrences":len(occurrences),"expert_annotations":"not_applicable_pending_labels",
               "comparison":"not_applicable_pending_labels","unmatched_candidates":"unknown",
               "candidate_surrogate_panel_limit":AUDIT_IDENTITY_LIMIT}
    atomic_json(audit / "summary.json", summary)
    atomic_json(audit / "llm_context.json",
                {**summary,"coordinate_convention":"x=column,y=row","frame_convention":"UI one-based inclusive",
                 "marker_semantics":{"orange":"model proposal","green":"reserved for future expert annotation"},
                 "display_semantics":{"raw":"fixed grayscale","msica":"fixed grayscale",
                                      "msln":"fixed symmetric grayscale with zero mid-gray",
                                      "evidence":"fixed zero-based grayscale","pooled":"fixed zero-based grayscale"},
                 "primary_artifacts":{"full_field_video":"2_Model_Annotations/videos/model_only_full_field_sequential_stages.mp4",
                                      "occurrences":"2_Model_Annotations/model_occurrences.csv"},
                 "limitations":["No expert labels were available.",
                                "10 fps is visualization timing because TIFF metadata lacked acquisition timing.",
                                "Candidate precision cannot be estimated; unlabeled candidates are unknown."]})
    validation = {"status":"complete","expert_applicability":"not_applicable_pending_labels",
                  "comparison_applicability":"not_applicable_pending_labels",
                  "expected_model_closeups":len(models),
                  "observed_model_closeups":len(list((model_root/"videos"/"closeups").glob("*.mp4"))),
                  "expected_model_traces":len(models),
                  "observed_model_traces":len(list((model_root/"figures"/"traces").glob("*.png"))),
                  "expected_occurrence_instants":len(occurrences),
                  "observed_occurrence_instants":len(list((model_root/"figures"/"instants").glob("*.png"))),
                  "full_field_video_exists":(model_root/"videos"/"model_only_full_field_sequential_stages.mp4").is_file(),
                  "sequential_stage_count":5,"grayscale_backgrounds":True,
                  "model_marker_color":"orange","expert_markers_present":False}
    atomic_json(audit / "validation.json", validation)
    artifacts = [{"path":str(path.relative_to(audit)),"size_bytes":path.stat().st_size}
                 for path in sorted(audit.rglob("*")) if path.is_file()]
    atomic_json(audit / "artifact_index.json", {"status":"complete","artifacts":artifacts})
    atomic_json(audit / "status.json",
                {"status":"complete","scientific_status":"unlabeled_visual_review_only","completed_at":_now()})
    (audit / "REPORT.md").write_text(
        f"# Label-free scientific audit\n\nThe frozen {FROZEN_LANE} lane proposed {len(models)} bounded candidate-surrogate identities across {len(events)} model-proposed event windows. No expert labels were available, so Expert and Comparison are explicitly not applicable and no precision claim is made.\n",
        encoding="utf-8")
    return validation


def preflight(input_tiff: Path, output_root: Path, review_root: Path, *, seed: int) -> dict[str, Any]:
    input_tiff, output_root, review_root = input_tiff.resolve(), output_root.resolve(), review_root.resolve()
    if output_root.exists() or review_root.exists():
        raise FileExistsError("inference and review roots must both be new")
    values = _source(input_tiff)
    disk = shutil.disk_usage(output_root.parent if output_root.parent.exists() else output_root.parent.parent)
    device = cuda_device_summary()
    estimate = 2 * int(np.prod(values.shape)) * 4 + int(input_tiff.stat().st_size * 1.5) + 8 * 2**30
    if disk.free < estimate:
        raise RuntimeError(f"insufficient disk: guarded estimate {estimate}, free {disk.free}")
    if device["free_bytes"] < VRAM_CAP:
        raise RuntimeError("free GPU memory is below the 8 GiB cap")
    output_root.mkdir(parents=True)
    payload = {"ready":True,"source_read_only":True,"output_collision_checked":True,
               "input_tiff":str(input_tiff),"source_sha256":sha256_file(input_tiff),
               "shape_tyx":list(values.shape),"dtype":str(values.dtype),
               "axes_interpretation":"TYX despite ImageJ ZYX metadata; pages are sequential frames",
               "frame_rate_hz":None,"playback_fps_visualization_only":DISPLAY_FPS,
               "gpu":device,"vram_cap_bytes":VRAM_CAP,"ram_cap_bytes":24*2**30,
               "disk_free_bytes":disk.free,"estimated_total_output_headroom_bytes":estimate,
               "frozen_lane":FROZEN_LANE,"event_source":"model_proposed","seed":int(seed),
               "resource_strategy":"read-only TIFF memmap; one CUDA worker; 96-frame chunks; fixed global MSLN floor; atomic/resumable NPY stages",
               "review_root":str(review_root),"created_at":_now()}
    atomic_json(output_root / "preflight.json", payload)
    atomic_json(output_root / "status.json", {"status":"preflight_complete","scientific_status":"not_started"})
    return payload


def run(input_tiff: Path, output_root: Path, review_root: Path, *, seed: int) -> dict[str, Any]:
    input_tiff, root, package = input_tiff.resolve(), output_root.resolve(), review_root.resolve()
    pre = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    if not pre.get("ready") or pre["source_sha256"] != sha256_file(input_tiff):
        raise RuntimeError("preflight/input fingerprint mismatch")
    if package.exists():
        raise FileExistsError(package)
    atomic_json(root / "status.json",
                {"status":"running","scientific_status":"unlabeled_visual_review_only","started_at":_now()})
    started = time.monotonic()
    values = _source(input_tiff)
    _, quiet_raw = _temporal_scores(values, root)
    fit = _sample_fit(values, _fit_config(seed), formulation="multilag_2d",
                      lags=(0,1,2,4), objective="normalized_hsic",
                      parameter={"bandwidth_scale":0.5}, weight_decay=0.0)
    atomic_json(root / "msica_fit.json", fit.to_dict())
    msica_all = _project_msica(values, fit, root / "msica_persistence.npy", root)
    quiet_msica = quiet_raw[1:]
    floor, pilot = _pilot_floor(msica_all, quiet_msica, root)
    atomic_json(root / "msln_floor_calibration.json",
                {"global_scale_floor":floor,
                 "method":"median of eight separated quiet-rich CUDA pilot chunks",
                 "pilot_chunks":pilot})
    msln, scores, chunk_diagnostics = _project_msln(
        msica_all, quiet_msica, root / "msln_signed.npy",
        root / "msln_frame_energy_scores.npy", root, floor)
    atomic_json(root / "msln_chunk_diagnostics.json", {"chunks":chunk_diagnostics})
    events = _event_windows(scores)
    atomic_json(root / "model_proposed_events.json", {"event_source":"model_proposed","events":events})
    all_occurrences, pooled = _proposals(msln, events, root)
    models, occurrences = _consolidate(all_occurrences)
    _write_csv(root / "all_model_occurrences.csv", all_occurrences)
    atomic_json(root / "frozen_candidate_panel.json",
                {"limit":AUDIT_IDENTITY_LIMIT,"selection":"recurrence then peak score, label-free","models":models})
    raw = values[32:]
    msica = msica_all[31:]
    if raw.shape != msica.shape or raw.shape != msln.shape:
        raise RuntimeError("Raw/MSICA/MSLN alignment invariant failed")
    source_video = _full_video(root / "videos" / "model_only_full_field_sequential_stages.mp4",
                               raw, msica, msln, events, pooled, models, root)
    app = _source_app(root, input_tiff, raw, msica, msln, models, fit, events, source_video)
    package_result = build_model_proposal_package(
        source_app_dir=app, output_root=package,
        dataset_id=root.name.replace("_multilag_msica_msln_inference_v1", ""),
        event_source="model_proposed")
    audit_validation = _audit(package, raw, msica, msln, models, occurrences, events, pooled, root)
    validation = {"status":"complete","source_sha256":pre["source_sha256"],
                  "shape_input_tyx":list(values.shape),"shape_aligned_tyx":list(raw.shape),
                  "alignment_start_ui":33,
                  "finite_msica_sample":bool(np.isfinite(np.asarray(msica[::max(1,len(msica)//16),::8,::8])).all()),
                  "finite_msln_sample":bool(np.isfinite(np.asarray(msln[::max(1,len(msln)//16),::8,::8])).all()),
                  "event_windows":len(events),"all_occurrences":len(all_occurrences),
                  "audited_model_identities":len(models),"audited_model_occurrences":len(occurrences),
                  "audit":audit_validation}
    atomic_json(root / "validation.json", validation)
    summary = {"status":"complete","scientific_status":"unlabeled_visual_review_only",
               "input_tiff":str(input_tiff),"inference_root":str(root),"review_root":str(package),
               "frozen_lane":FROZEN_LANE,"event_source":"model_proposed",
               "events":len(events),"model_identities":len(models),
               "model_occurrences":len(occurrences),"all_screened_occurrences":len(all_occurrences),
               "elapsed_seconds":time.monotonic()-started,"labels_used":False,
               "unmatched_candidates":"unknown","playback_fps_visualization_only":DISPLAY_FPS,
               "acquisition_frame_rate":"not_present_in_tiff_metadata",
               "proposal_package":package_result}
    atomic_json(root / "run_summary.json", summary)
    (root / "REPORT.md").write_text(
        f"# {input_tiff.stem}: label-free Raw -> MSICA -> MSLN run\n\n"
        f"The fixed v5 architecture {FROZEN_LANE} was fit without labels and applied with bounded CUDA chunks. "
        f"The detector proposed {len(events)} event windows, screened {len(all_occurrences)} event-level local maxima, "
        f"and froze {len(models)} consolidated identities for audit. These are unknown proposals, not validated neurons. "
        "The TIFF did not include acquisition timing; 10 fps is review playback only.\n",
        encoding="utf-8")
    atomic_json(root / "status.json",
                {"status":"complete","scientific_status":"unlabeled_visual_review_only","completed_at":_now()})
    gc.collect()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        item = sub.add_parser(name)
        item.add_argument("--input-tif", type=Path, required=True)
        item.add_argument("--output-root", type=Path, required=True)
        item.add_argument("--review-root", type=Path, required=True)
        item.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args(argv)
    result = (preflight(args.input_tif, args.output_root, args.review_root, seed=args.seed)
              if args.command == "preflight"
              else run(args.input_tif, args.output_root, args.review_root, seed=args.seed))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
