#!/usr/bin/env python3
"""Build 64x64 grid-dynamics dashboard artifacts from a completed sweep."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench.dynamics.concept_tests import ResidualPixelGRU, _build_spatial_pixel_model, _encode_windows, _load_autoencoder, _torch
from neurobench.dynamics.models import ScalableTemporalCNNResidual
from neurobench.dynamics.scalable import predict_scalable_temporal_cnn
from neurobench.dynamics.visualize_sweep import generate_sweep_visuals


FONT_5X7 = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


@dataclass(frozen=True)
class ModelSelection:
    tag: str
    label: str
    row: dict[str, Any]


SUPPORTED_VIDEO_KINDS = {
    "residual_pixel",
    "convgru_pixel",
    "convlstm_pixel",
    "temporal_cnn_pixel",
    "unet_convgru_pixel",
    "scalable_temporal_cnn_pixel",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh, delimiter="\t"):
            if not raw.get("experiment_id"):
                continue
            row: dict[str, Any] = dict(raw)
            for key in (
                "rank",
                "seed",
                "val_decoded_prediction_mse",
                "val_persistence_mse",
                "val_improvement_over_persistence_mse",
                "test_decoded_prediction_mse",
                "test_persistence_mse",
                "test_improvement_over_persistence_mse",
            ):
                value = row.get(key)
                if value not in (None, ""):
                    row[key] = int(float(value)) if key in {"rank", "seed"} else float(value)
            rows.append(row)
    return rows


def _score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1e9
    return number if np.isfinite(number) else -1e9


def choose_models(rows: list[dict[str, Any]], *, selected_count: int = 3) -> list[ModelSelection]:
    eligible = [r for r in rows if str(r.get("kind")) in SUPPORTED_VIDEO_KINDS]
    if not eligible:
        raise ValueError(f"No supported video-capable model rows found. Supported kinds: {sorted(SUPPORTED_VIDEO_KINDS)}")
    ranked = sorted(
        eligible,
        key=lambda r: (
            -_score(r.get("test_improvement_over_persistence_mse")),
            -_score(r.get("val_improvement_over_persistence_mse")),
            str(r.get("experiment_id", "")),
        ),
    )
    selections: list[ModelSelection] = []
    for index, row in enumerate(ranked[: max(1, int(selected_count))], start=1):
        selections.append(ModelSelection(f"top_test_{index}", f"Top test {index}", row))
    return selections


def slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "input"


def video_split(video_id: str, dataset: Mapping[str, Any]) -> str:
    splits = dataset.get("splits") or {}
    for split in ("train", "val", "test"):
        if video_id in {str(v) for v in splits.get(f"{split}_video_ids", [])}:
            return split
    return "unknown"


def select_dynamic_segment(windows: np.ndarray, targets: np.ndarray, indices: np.ndarray, max_frames: int) -> tuple[np.ndarray, int, float]:
    if indices.shape[0] <= max_frames:
        motion = np.mean(np.abs(targets[indices] - windows[indices, -1]), axis=(1, 2, 3))
        return indices, 0, float(np.mean(motion)) if motion.size else 0.0
    motion = np.mean(np.abs(targets[indices] - windows[indices, -1]), axis=(1, 2, 3))
    kernel = np.ones((max_frames,), dtype=np.float32)
    start = int(np.argmax(np.convolve(motion, kernel, mode="valid")))
    selected = indices[start : start + max_frames]
    return selected, start, float(np.mean(motion[start : start + max_frames]))


def predict_residual(row: Mapping[str, Any], dataset: Mapping[str, Any], autoencoder_run: Mapping[str, Any], windows: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    metrics_path = Path(str(row["metrics_path"]))
    checkpoint_path = metrics_path.parent / "concept_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ae, latent_mean, latent_std, latent_dim = _load_autoencoder(autoencoder_run, input_channels=int(windows.shape[2]), device=device)
    ae.eval()
    xw = torch.from_numpy(windows.astype(np.float32, copy=False)).to(device)
    z_windows = _encode_windows(ae, xw, latent_mean, latent_std, latent_dim, batch_size=batch_size)
    model = ResidualPixelGRU.build(
        latent_dim=latent_dim,
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_shape=tuple(int(v) for v in windows.shape[2:]),
        residual_scale=float(checkpoint["residual_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, z_windows.shape[0], int(batch_size)):
            pred = model(z_windows[start : start + int(batch_size)], xw[start : start + int(batch_size), -1])
            chunks.append(pred.detach().cpu().numpy().astype(np.float32))
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, *windows.shape[2:]), dtype=np.float32)


def predict_spatial_pixel(row: Mapping[str, Any], windows: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    metrics_path = Path(str(row["metrics_path"]))
    checkpoint_path = metrics_path.parent / "concept_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_spatial_pixel_model(
        architecture=str(checkpoint["architecture"]),
        input_channels=int(checkpoint.get("input_channels", windows.shape[2])),
        window_frames=int(checkpoint.get("window_frames", windows.shape[1])),
        hidden_channels=int(checkpoint["hidden_channels"]),
        num_layers=int(checkpoint.get("num_layers", 1)),
        residual_scale=float(checkpoint["residual_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    xw = torch.from_numpy(windows.astype(np.float32, copy=False)).to(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, xw.shape[0], int(batch_size)):
            pred = model(xw[start : start + int(batch_size)])
            chunks.append(pred.detach().cpu().numpy().astype(np.float32))
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, *windows.shape[2:]), dtype=np.float32)


def predict_scalable_pixel(row: Mapping[str, Any], windows: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    metrics_path = Path(str(row["metrics_path"]))
    checkpoint_path = metrics_path.parent / "concept_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ScalableTemporalCNNResidual(
        input_channels=int(checkpoint.get("input_channels", windows.shape[2])),
        window_frames=int(checkpoint.get("window_frames", windows.shape[1])),
        architecture_spec=checkpoint["architecture_spec"],
        residual_scale=float(checkpoint["residual_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    pred = predict_scalable_temporal_cnn(model, windows.astype(np.float32, copy=False), batch_size=batch_size, device=device)
    if device == "cuda":
        torch.cuda.empty_cache()
    return pred


def predict_model(row: Mapping[str, Any], dataset: Mapping[str, Any], autoencoder_run: Mapping[str, Any] | None, windows: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    kind = str(row.get("kind", ""))
    if kind == "residual_pixel":
        return predict_residual(row, dataset, autoencoder_run, windows, batch_size=batch_size, device=device)
    if kind in {"convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel", "unet_convgru_pixel"}:
        return predict_spatial_pixel(row, windows, batch_size=batch_size, device=device)
    if kind == "scalable_temporal_cnn_pixel":
        return predict_scalable_pixel(row, windows, batch_size=batch_size, device=device)
    raise ValueError(f"Unsupported video prediction model kind: {kind}")


def robust_limits(*arrays: np.ndarray, low_pct: float, high_pct: float, fallback_high: float = 1.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays if np.asarray(a).size])
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, float(fallback_high)
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi <= lo + 1e-8:
        hi = lo + float(fallback_high)
    return lo, hi


def scale_gray(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = (np.asarray(frame, dtype=np.float32) - float(low)) / max(float(high) - float(low), 1e-8)
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def draw_text(canvas: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int] = (24, 32, 40), scale: int = 2) -> None:
    cx = int(x)
    for ch in text.upper():
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    y0 = y + gy * scale
                    x0 = cx + gx * scale
                    canvas[y0 : y0 + scale, x0 : x0 + scale] = color
        cx += 6 * scale


def upsample_rgb(gray: np.ndarray, scale: int) -> np.ndarray:
    panel = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)
    return np.stack([panel, panel, panel], axis=-1)


def motion_frames(frames: np.ndarray) -> np.ndarray:
    current = np.asarray(frames, dtype=np.float32)
    previous = np.concatenate([current[:1], current[:-1]], axis=0)
    return np.abs(current - previous)


def iter_video_frames(
    *,
    title: str,
    view_label: str,
    target: np.ndarray,
    pred: np.ndarray,
    persistence: np.ndarray,
    scale: int,
    intensity_limits: tuple[float, float],
    motion_high: float,
    error_high: float,
    target_raw_indices: np.ndarray,
    model_raw_indices: np.ndarray,
    model_source_raw_indices: np.ndarray,
):
    target_2d = target[:, 0]
    pred_2d = pred[:, 0]
    persist_2d = persistence[:, 0]
    error_2d = np.abs(target_2d - pred_2d)
    if view_label == "motion":
        panels = (
            motion_frames(target_2d),
            motion_frames(pred_2d),
            motion_frames(persist_2d),
            error_2d,
        )
        limits = ((0.0, motion_high), (0.0, motion_high), (0.0, motion_high), (0.0, error_high))
    else:
        panels = (target_2d, pred_2d, persist_2d, error_2d)
        limits = (intensity_limits, intensity_limits, intensity_limits, (0.0, error_high))
    panel_h = target_2d.shape[1] * scale
    panel_w = target_2d.shape[2] * scale
    gutter = 10
    header = 34
    footer = 42
    width = 4 * panel_w + 5 * gutter
    height = header + panel_h + footer
    for i in range(target_2d.shape[0]):
        canvas = np.full((height, width, 3), 248, dtype=np.uint8)
        draw_text(canvas, gutter, 9, title[:80], scale=2)
        draw_text(canvas, width - 170, 9, view_label, color=(74, 92, 105), scale=2)
        target_idx = int(target_raw_indices[i]) if i < len(target_raw_indices) else i
        model_idx = int(model_raw_indices[i]) if i < len(model_raw_indices) else i
        source_idx = int(model_source_raw_indices[i]) if i < len(model_source_raw_indices) else target_idx
        labels = (f"TARGET F{target_idx}", f"MODEL F{model_idx}", f"PERSIST F{model_idx}", f"ABS ERR F{target_idx}")
        sublabels = ("", f"FROM F{source_idx}", f"COPY F{source_idx}", f"VS MODEL F{model_idx}")
        for pi, panel in enumerate(panels):
            x = gutter + pi * (panel_w + gutter)
            y = header
            gray = scale_gray(panel[i], *limits[pi])
            canvas[y : y + panel_h, x : x + panel_w] = upsample_rgb(gray, scale)
            draw_text(canvas, x + 2, header + panel_h + 6, labels[pi], color=(54, 68, 80), scale=1)
            if sublabels[pi]:
                draw_text(canvas, x + 2, header + panel_h + 18, sublabels[pi], color=(93, 106, 118), scale=1)
        yield canvas

def write_mp4(path: Path, frames, *, width: int, height: int, fps: float, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
    stdout = proc.stdout.read() if proc.stdout is not None else b""
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr.decode('utf-8', 'replace')[-2000:]}")


def render_clip_pair(
    *,
    charts_dir: Path,
    model: ModelSelection,
    video_id: str,
    split: str,
    windows: np.ndarray,
    targets: np.ndarray,
    pred: np.ndarray,
    selected_start: int,
    fps: float,
    scale: int,
    force: bool,
    window_frames: int,
    prediction_horizon_frames: int,
    temporal_stride_frames: int,
) -> dict[str, Any]:
    persistence = windows[:, -1]
    raw_offset = int(prediction_horizon_frames) * int(temporal_stride_frames)
    shift = max(0, int(prediction_horizon_frames))
    first_last_input_sample = int(selected_start) + int(window_frames) - 1
    first_target_sample = first_last_input_sample + int(prediction_horizon_frames)
    first_last_input_raw = first_last_input_sample * int(temporal_stride_frames)
    first_target_raw = first_target_sample * int(temporal_stride_frames)
    raw_step = int(temporal_stride_frames)
    frame_steps = np.arange(int(targets.shape[0]), dtype=np.int64) * raw_step
    all_target_raw_indices = first_target_raw + frame_steps
    all_last_input_raw_indices = first_last_input_raw + frame_steps

    display_count = int(min(targets.shape[0], pred.shape[0] - shift, persistence.shape[0] - shift))
    if display_count <= 0:
        raise ValueError(f"Not enough frames to lag-align {video_id} by {shift} sampled frames.")
    target_display = targets[:display_count]
    pred_display = pred[shift : shift + display_count]
    persistence_display = persistence[shift : shift + display_count]
    target_display_raw_indices = all_target_raw_indices[:display_count]
    model_display_raw_indices = all_target_raw_indices[shift : shift + display_count]
    model_source_raw_indices = all_last_input_raw_indices[shift : shift + display_count]

    intensity_limits = robust_limits(target_display[:, 0], pred_display[:, 0], persistence_display[:, 0], low_pct=1.0, high_pct=99.2, fallback_high=1.0)
    display_error = np.abs(target_display - pred_display)
    _elo, ehi = robust_limits(display_error, low_pct=0.0, high_pct=99.5, fallback_high=0.02)
    motion_high = robust_limits(motion_frames(target_display[:, 0]), motion_frames(pred_display[:, 0]), motion_frames(persistence_display[:, 0]), low_pct=0.0, high_pct=99.5, fallback_high=0.01)[1]

    display_mse = float(np.mean((pred_display - target_display) ** 2))
    display_persistence_mse = float(np.mean((persistence_display - target_display) ** 2))
    forecast_mse = float(np.mean((pred - targets) ** 2))
    forecast_persistence_mse = float(np.mean((persistence - targets) ** 2))

    offset_label = f"+{raw_offset}F"
    base = f"original_vs_reconstruction_{model.tag}_{slug(video_id)}"
    title = f"{model.label} | {video_id} | {split} | shifted {offset_label}"
    probe = next(iter_video_frames(
        title=title,
        view_label="intensity",
        target=target_display,
        pred=pred_display,
        persistence=persistence_display,
        scale=scale,
        intensity_limits=intensity_limits,
        motion_high=motion_high,
        error_high=ehi,
        target_raw_indices=target_display_raw_indices,
        model_raw_indices=model_display_raw_indices,
        model_source_raw_indices=model_source_raw_indices,
    ))
    height, width = probe.shape[:2]
    intensity_path = charts_dir / f"{base}_intensity.mp4"
    motion_path = charts_dir / f"{base}_motion.mp4"
    write_mp4(
        intensity_path,
        iter_video_frames(title=title, view_label="intensity", target=target_display, pred=pred_display, persistence=persistence_display, scale=scale, intensity_limits=intensity_limits, motion_high=motion_high, error_high=ehi, target_raw_indices=target_display_raw_indices, model_raw_indices=model_display_raw_indices, model_source_raw_indices=model_source_raw_indices),
        width=width,
        height=height,
        fps=fps,
        force=force,
    )
    write_mp4(
        motion_path,
        iter_video_frames(title=title, view_label="motion", target=target_display, pred=pred_display, persistence=persistence_display, scale=scale, intensity_limits=intensity_limits, motion_high=motion_high, error_high=ehi, target_raw_indices=target_display_raw_indices, model_raw_indices=model_display_raw_indices, model_source_raw_indices=model_source_raw_indices),
        width=width,
        height=height,
        fps=fps,
        force=force,
    )
    return {
        "video_id": video_id,
        "split": split,
        "model_tag": model.tag,
        "model_label": model.label,
        "experiment_id": model.row["experiment_id"],
        "dataset_key": model.row["dataset_key"],
        "start_window_in_video": int(selected_start),
        "window_frames": int(window_frames),
        "prediction_horizon_frames": int(prediction_horizon_frames),
        "temporal_stride_frames": int(temporal_stride_frames),
        "target_offset_sampled_frames": int(prediction_horizon_frames),
        "target_offset_raw_frames": int(raw_offset),
        "playback_alignment": "lag_compensated_by_forecast_horizon",
        "alignment_shift_sampled_frames": int(shift),
        "first_clip_last_input_sampled_frame": int(first_last_input_sample),
        "first_clip_target_sampled_frame": int(first_target_sample),
        "first_clip_last_input_raw_frame": int(first_last_input_raw),
        "first_clip_target_raw_frame": int(target_display_raw_indices[0]),
        "first_clip_model_raw_frame": int(model_display_raw_indices[0]),
        "first_clip_model_source_raw_frame": int(model_source_raw_indices[0]),
        "frame_count": int(display_count),
        "source_window_count": int(targets.shape[0]),
        "fps": float(fps),
        "mean_motion": float(np.mean(np.abs(target_display - persistence_display))),
        "decoded_prediction_mse": display_mse,
        "persistence_mse": display_persistence_mse,
        "improvement_over_persistence_mse": float(display_persistence_mse - display_mse),
        "forecast_decoded_prediction_mse": forecast_mse,
        "forecast_persistence_mse": forecast_persistence_mse,
        "forecast_improvement_over_persistence_mse": float(forecast_persistence_mse - forecast_mse),
        "display_limits": {
            "intensity_low": float(intensity_limits[0]),
            "intensity_high": float(intensity_limits[1]),
            "motion_high": float(motion_high),
            "error_high": float(ehi),
        },
        "intensity_file": intensity_path.name,
        "motion_file": motion_path.name,
    }

def build_video_selector(
    *,
    sweep_dir: Path,
    mapping: Mapping[str, Any],
    models: list[ModelSelection],
    charts_dir: Path,
    max_frames: int,
    batch_size: int,
    device: str,
    fps: float,
    scale: int,
    force: bool,
) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    dataset_cache: dict[str, tuple[dict[str, Any], dict[str, Any] | None, dict[str, np.ndarray]]] = {}
    for model in models:
        dataset_key = str(model.row["dataset_key"])
        if dataset_key not in dataset_cache:
            item = mapping[dataset_key]
            dataset = load_json(Path(item["dataset"]))
            autoencoder_run = load_json(Path(item["autoencoder_run"])) if item.get("autoencoder_run") else None
            arrays_npz = np.load(dataset["array_path"], allow_pickle=False)
            arrays = {
                "windows": arrays_npz["windows"].astype(np.float32, copy=False),
                "targets": arrays_npz["targets"].astype(np.float32, copy=False),
                "window_video_ids": arrays_npz["window_video_ids"].astype(str),
            }
            arrays_npz.close()
            dataset_cache[dataset_key] = (dataset, autoencoder_run, arrays)
        dataset, autoencoder_run, arrays = dataset_cache[dataset_key]
        windows = arrays["windows"]
        targets = arrays["targets"]
        video_ids = arrays["window_video_ids"]
        source_videos = list(dataset.get("source_videos") or sorted(set(video_ids.tolist())))
        for video_id in source_videos:
            indices = np.flatnonzero(video_ids == str(video_id))
            if not indices.size:
                continue
            windowing = dataset.get("windowing") or {}
            shift = max(0, int(windowing.get("prediction_horizon_frames", 1)))
            _selected_base, start, _mean_motion = select_dynamic_segment(windows, targets, indices, int(max_frames))
            desired_count = int(max_frames) + shift
            if indices.shape[0] > desired_count and start + desired_count > indices.shape[0]:
                start = max(0, int(indices.shape[0]) - desired_count)
            selected = indices[start : min(int(indices.shape[0]), start + desired_count)]
            pred = predict_model(model.row, dataset, autoencoder_run, windows[selected], batch_size=batch_size, device=device)
            split = video_split(str(video_id), dataset)
            options.append(
                render_clip_pair(
                    charts_dir=charts_dir,
                    model=model,
                    video_id=str(video_id),
                    split=split,
                    windows=windows[selected],
                    targets=targets[selected],
                    pred=pred,
                    selected_start=start,
                    fps=fps,
                    scale=scale,
                    force=force,
                    window_frames=int(windowing.get("window_frames", windows.shape[1])),
                    prediction_horizon_frames=int(windowing.get("prediction_horizon_frames", 1)),
                    temporal_stride_frames=int(windowing.get("temporal_stride_frames", 1)),
                )
            )
    selector_path = charts_dir / "original_vs_reconstruction_selector.html"
    selector_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep_dir),
        "panel_order": ["target_frame", "model_prediction_shifted_by_horizon", "persistence_prediction_shifted_by_horizon", "lag_compensated_absolute_error"],
        "segment_selection": f"highest mean future-target-vs-last-input motion segment, up to {max_frames} windows per input",
        "models": [
            {
                "tag": model.tag,
                "label": model.label,
                "experiment_id": model.row["experiment_id"],
                "dataset_key": model.row["dataset_key"],
                "val_improvement_over_persistence_mse": model.row.get("val_improvement_over_persistence_mse"),
                "test_improvement_over_persistence_mse": model.row.get("test_improvement_over_persistence_mse"),
            }
            for model in models
        ],
        "options": options,
    }
    (charts_dir / "original_vs_reconstruction_selector.json").write_text(json.dumps(selector_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selector_path.write_text(selector_html(selector_payload), encoding="utf-8")
    return selector_payload


def selector_html(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Dynamics Forecast Selector</title>
<style>
body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background:#f7f9fb; color:#17212b; }}
main {{ max-width:1180px; margin:0 auto; padding:22px; }}
.bar {{ display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:16px; }}
label {{ display:grid; gap:5px; font-size:12px; font-weight:700; color:#3f4b57; }}
select, button {{ border:1px solid #c9d3dc; background:#fff; color:#17212b; border-radius:6px; padding:8px 10px; font:inherit; }}
button.active {{ background:#17212b; color:#fff; border-color:#17212b; }}
video {{ width:100%; background:#111; border:1px solid #d3dce4; border-radius:8px; }}
.meta {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:8px; margin:14px 0; }}
.metric {{ background:#fff; border:1px solid #dce4eb; border-radius:8px; padding:10px 12px; }}
.metric b {{ display:block; font-variant-numeric: tabular-nums; }}
.metric span {{ color:#5d6a76; font-size:12px; }}
.note {{ color:#5d6a76; font-size:13px; }}
</style>
</head>
<body>
<main>
  <h1>Dynamics Forecast</h1>
  <div class=\"bar\">
    <label>Model<select id=\"model\"></select></label>
    <label>Input<select id=\"input\"></select></label>
    <button id=\"intensity\" class=\"active\" type=\"button\">Intensity</button>
    <button id=\"motion\" type=\"button\">Motion</button>
  </div>
  <video id=\"video\" controls muted loop playsinline></video>
  <div class=\"meta\" id=\"meta\"></div>
  <p class=\"note\">Panel order in every clip: target frame F, model prediction at F + horizon generated from F, persistence prediction at F + horizon copied from F, and lag-compensated absolute error between the displayed target and displayed model. This is an offset-aligned visual diagnostic; sweep metrics remain the direct forecast metrics.</p>
</main>
<script>
const PAYLOAD = {data};
let currentView = 'intensity';
const modelSelect = document.getElementById('model');
const inputSelect = document.getElementById('input');
const video = document.getElementById('video');
const meta = document.getElementById('meta');
const models = PAYLOAD.models || [];
for (const model of models) {{
  const option = document.createElement('option');
  option.value = model.tag;
  option.textContent = `${{model.label}}: ${{model.experiment_id}}`;
  modelSelect.appendChild(option);
}}
function optionsForModel() {{
  return (PAYLOAD.options || []).filter(item => item.model_tag === modelSelect.value);
}}
function fillInputs() {{
  const prior = inputSelect.value;
  inputSelect.innerHTML = '';
  for (const item of optionsForModel()) {{
    const option = document.createElement('option');
    option.value = item.video_id;
    option.textContent = `${{item.video_id}} (${{item.split}})`;
    inputSelect.appendChild(option);
  }}
  if ([...inputSelect.options].some(option => option.value === prior)) inputSelect.value = prior;
}}
function fmt(value) {{
  const n = Number(value);
  return Number.isFinite(n) ? n.toExponential(3) : 'n/a';
}}
function selectedItem() {{
  return optionsForModel().find(item => item.video_id === inputSelect.value) || optionsForModel()[0];
}}
function render() {{
  const item = selectedItem();
  if (!item) return;
  const src = currentView === 'motion' ? item.motion_file : item.intensity_file;
  video.src = src;
  video.load();
  meta.innerHTML = [
    ['Dataset', item.dataset_key],
    ['Split', item.split],
    ['Frames', item.frame_count],
    ['Forecast offset', `${{item.target_offset_raw_frames}} raw frames`],
    ['Window', `${{item.window_frames}} frames`],
    ['First target frame', item.first_clip_target_raw_frame],
    ['First model/persist frame', item.first_clip_model_raw_frame],
    ['Model/persist source', item.first_clip_model_source_raw_frame],
    ['Display MSE', fmt(item.decoded_prediction_mse)],
    ['Persistence MSE', fmt(item.persistence_mse)],
    ['Improvement', fmt(item.improvement_over_persistence_mse)],
    ['Start window', item.start_window_in_video],
    ['Mean motion', fmt(item.mean_motion)]
  ].map(([label, value]) => `<div class=\"metric\"><b>${{value}}</b><span>${{label}}</span></div>`).join('');
}}
modelSelect.addEventListener('change', () => {{ fillInputs(); render(); }});
inputSelect.addEventListener('change', render);
for (const id of ['intensity', 'motion']) {{
  document.getElementById(id).addEventListener('click', () => {{
    currentView = id;
    document.getElementById('intensity').classList.toggle('active', id === 'intensity');
    document.getElementById('motion').classList.toggle('active', id === 'motion');
    render();
  }});
}}
fillInputs();
render();
</script>
</body>
</html>
"""


def dashboard_artifacts_from_summary(summary: Mapping[str, Any], selector: Mapping[str, Any], dashboard_prefix: str) -> list[dict[str, Any]]:
    artifacts = list(summary.get("artifacts") or [])
    input_options = []
    seen_inputs: set[str] = set()
    for option in selector.get("options", []):
        video_id = str(option.get("video_id"))
        if video_id in seen_inputs:
            continue
        seen_inputs.add(video_id)
        input_options.append({"video_id": video_id, "split": option.get("split")})
    selector_artifact = {
        "id": "original_vs_reconstruction_selector_64",
        "label": "64x64 original vs reconstruction selector",
        "description": "Input-switching offset-aligned selector for 64x64 target, horizon-shifted model output, horizon-shifted persistence prediction, and lag-compensated absolute-error MP4 clips.",
        "file": f"{dashboard_prefix}/charts/original_vs_reconstruction_selector.html",
        "path": "Outputs/GridModel/060126/overnight_sweep_64_v1/visuals/charts/original_vs_reconstruction_selector.html",
        "input_options": input_options,
    }
    return [selector_artifact] + [a for a in artifacts if a.get("id") != selector_artifact["id"]]


def update_dashboard_json(
    *,
    dashboard_app: Path,
    sweep_dir: Path,
    visual_summary: Mapping[str, Any],
    selector: Mapping[str, Any],
    dashboard_prefix: str,
) -> None:
    arch_path = dashboard_app / "architecture_runs.json"
    arch = load_json(arch_path)
    template = dict(arch.get("templateGrid") or {})
    old_sweep = template.get("overnight_sweep")
    if old_sweep and "overnight_sweep_32_v1" not in template:
        template["overnight_sweep_32_v1"] = old_sweep
    sweep_payload = dict(visual_summary)
    sweep_payload["artifacts"] = dashboard_artifacts_from_summary(visual_summary, selector, dashboard_prefix)
    sweep_payload["original_vs_reconstruction_video"] = {
        "dataset_key": selector.get("models", [{}])[0].get("dataset_key"),
        "experiment_id": selector.get("models", [{}])[0].get("experiment_id"),
        "artifact": sweep_payload["artifacts"][0],
        "input_count": len({item.get("video_id") for item in selector.get("options", [])}),
        "model_count": len(selector.get("models", [])),
        "segment_selection": selector.get("segment_selection"),
        "display": {
            "intensity_view": "Shared robust intensity contrast for target, horizon-shifted model output, and horizon-shifted persistence prediction panels; error panel scaled separately.",
            "motion_view": "Frame-to-frame absolute temporal change for target, horizon-shifted model output, and horizon-shifted persistence prediction panels; error panel scaled separately.",
        },
    }
    sweep_payload["source_manifest"] = str(sweep_dir / "sweep_manifest.json")
    template["overnight_sweep"] = sweep_payload
    template["overnight_sweep_64_v1"] = sweep_payload
    template["artifacts"] = sweep_payload["artifacts"]
    template["gridStateCount"] = template.get("gridStateCount", "streamed")
    template["split_unit"] = "video"
    template["spec"] = {**dict(template.get("spec") or {}), "rows": 64, "cols": 64, "grid_id": "grid_64x64_template_from_1 resting_v1"}
    arch["templateGrid"] = template
    arch["dataset_id"] = "060126_grid64_dynamics"
    run = {
        "schema_version": 1,
        "run_id": "overnight_sweep_64_v1",
        "label": "060126 64x64 overnight dynamics sweep",
        "dataset_id": "060126_grid64_dynamics",
        "execution": {"status": "completed"},
        "summary": {
            "experiment_count": sweep_payload.get("experiment_count"),
            "positive_validation_count": sweep_payload.get("positive_validation_count"),
            "positive_test_count": sweep_payload.get("positive_test_count"),
            "positive_validation_and_test_count": sweep_payload.get("positive_validation_and_test_count"),
        },
        "artifacts": {
            "app_url": "index.html#progress",
            "sweep_summary_tsv": f"{dashboard_prefix}/source/sweep_summary.tsv",
            "sweep_visual_summary": f"{dashboard_prefix}/sweep_visual_summary.json",
            "original_vs_reconstruction_selector": f"{dashboard_prefix}/charts/original_vs_reconstruction_selector.html",
        },
    }
    runs = [r for r in arch.get("runs", []) if r.get("run_id") != "overnight_sweep_64_v1"]
    arch["runs"] = [run] + runs
    write_json(arch_path, arch)
    source_path = dashboard_app / "architecture_runs.source.json"
    write_json(source_path, arch)


def copy_to_dashboard(*, sweep_dir: Path, dashboard_app: Path, dashboard_prefix: str) -> None:
    destination = dashboard_app / dashboard_prefix
    destination.mkdir(parents=True, exist_ok=True)
    visuals = sweep_dir / "visuals"
    shutil.copytree(visuals, destination, dirs_exist_ok=True)
    source = destination / "source"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_dir / "sweep_summary.tsv", source / "sweep_summary.tsv")
    shutil.copy2(sweep_dir / "sweep_summary.md", source / "sweep_summary.md")
    shutil.copy2(sweep_dir / "sweep_manifest.json", source / "sweep_manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("Outputs/GridModel/060126/overnight_sweep_64_v1"))
    parser.add_argument("--mapping-json", type=Path, default=Path("Outputs/GridModel/060126/improvement_attempts_64_v1/datasets_64_mapping.json"))
    parser.add_argument("--dashboard-app", type=Path, default=Path("Outputs/GridModel/060126/dashboard/app"))
    parser.add_argument("--dashboard-prefix", default="grid_dynamics/overnight_sweep_64_v1")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    summary_tsv = args.sweep_dir / "sweep_summary.tsv"
    visuals_dir = args.sweep_dir / "visuals"
    summary = generate_sweep_visuals(
        summary_tsv=summary_tsv,
        out_dir=visuals_dir,
        title="64x64 Overnight Dynamics Sweep",
        dashboard_prefix=args.dashboard_prefix,
        top_n=30,
    )
    rows = load_rows(summary_tsv)
    models = choose_models(rows)
    selector = build_video_selector(
        sweep_dir=args.sweep_dir,
        mapping=load_json(args.mapping_json),
        models=models,
        charts_dir=visuals_dir / "charts",
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        device=args.device,
        fps=args.fps,
        scale=args.scale,
        force=args.force,
    )
    summary = load_json(visuals_dir / "sweep_visual_summary.json")
    copy_to_dashboard(sweep_dir=args.sweep_dir, dashboard_app=args.dashboard_app, dashboard_prefix=args.dashboard_prefix)
    update_dashboard_json(
        dashboard_app=args.dashboard_app,
        sweep_dir=args.sweep_dir,
        visual_summary=summary,
        selector=selector,
        dashboard_prefix=args.dashboard_prefix,
    )
    print(json.dumps({
        "status": "built",
        "models": [m.row["experiment_id"] for m in models],
        "input_options": len({item["video_id"] for item in selector["options"]}),
        "clip_options": len(selector["options"]),
        "selector": str(args.dashboard_app / args.dashboard_prefix / "charts" / "original_vs_reconstruction_selector.html"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
