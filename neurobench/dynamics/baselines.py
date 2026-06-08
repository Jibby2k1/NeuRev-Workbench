"""Simple pixel-space baselines for grid dynamics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def evaluate_baselines_from_arrays(arrays: Mapping[str, Any], *, moving_average_window: int | None = None) -> dict[str, Any]:
    windows = np.asarray(arrays["windows"], dtype=np.float32)
    targets = np.asarray(arrays["targets"], dtype=np.float32)
    video_ids = np.asarray(arrays.get("window_video_ids", []), dtype=str)
    labels = np.asarray(arrays.get("window_labels", []), dtype=str)
    if windows.shape[0] == 0:
        empty = _metrics(np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        return {"persistence": empty, "moving_average": empty, "linear_extrapolation": empty, "mean_delta": empty}
    persistence = baseline_prediction(windows, "persistence")
    moving = baseline_prediction(windows, "moving_average", moving_average_window=moving_average_window)
    linear = baseline_prediction(windows, "linear_extrapolation")
    mean_delta = baseline_prediction(windows, "mean_delta")
    return {
        "persistence": _with_groups(persistence, targets, video_ids, labels),
        "moving_average": _with_groups(moving, targets, video_ids, labels),
        "linear_extrapolation": _with_groups(linear, targets, video_ids, labels),
        "mean_delta": _with_groups(mean_delta, targets, video_ids, labels),
    }


def baseline_prediction(windows: np.ndarray, baseline: str, *, moving_average_window: int | None = None) -> np.ndarray:
    """Return a clipped pixel-space baseline prediction for each window."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("windows must have shape (n, time, channels, height, width).")
    name = str(baseline).strip().lower()
    if name == "persistence":
        return values[:, -1].astype(np.float32, copy=False)
    if name == "moving_average":
        k = max(1, int(moving_average_window or values.shape[1]))
        return values[:, -k:].mean(axis=1).astype(np.float32, copy=False)
    if name == "linear_extrapolation":
        if values.shape[1] < 2:
            return np.clip(values[:, -1], 0.0, 1.0).astype(np.float32, copy=False)
        return np.clip(values[:, -1] + (values[:, -1] - values[:, -2]), 0.0, 1.0).astype(np.float32, copy=False)
    if name in {"mean_delta", "average_delta"}:
        if values.shape[1] < 2:
            return np.clip(values[:, -1], 0.0, 1.0).astype(np.float32, copy=False)
        delta = np.diff(values, axis=1).mean(axis=1)
        return np.clip(values[:, -1] + delta, 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported baseline prediction: {baseline}")


def write_baseline_metrics(dataset: Mapping[str, Any], out_path: str | Path) -> dict[str, Any]:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        metrics = evaluate_baselines_from_arrays(arrays)
    Path(out_path).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _with_groups(pred: np.ndarray, target: np.ndarray, video_ids: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    payload = _metrics(pred, target)
    payload["per_video"] = {vid: _metrics(pred[video_ids == vid], target[video_ids == vid]) for vid in sorted(set(video_ids.tolist()))}
    payload["per_label"] = {lab: _metrics(pred[labels == lab], target[labels == lab]) for lab in sorted(set(labels.tolist()))}
    return payload


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.size == 0 or target.size == 0:
        return {"mse": 0.0, "mae": 0.0, "count": 0}
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    return {"mse": float(np.mean(diff * diff)), "mae": float(np.mean(np.abs(diff))), "count": int(pred.shape[0])}
