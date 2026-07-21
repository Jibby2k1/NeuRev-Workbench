"""Simple pixel-space baselines for grid dynamics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_BASELINES = ("persistence", "moving_average", "linear_extrapolation", "mean_delta")
KINETICS_BASELINES = ("exponential_decay_10hz", "exponential_decay_30hz", "lowpass_10hz", "lowpass_30hz", "ar1_per_cell")


def evaluate_baselines_from_arrays(
    arrays: Mapping[str, Any],
    *,
    moving_average_window: int | None = None,
    baseline_names: Sequence[str] | None = None,
    prediction_horizon_frames: int = 1,
    frame_rate_hz: float = 50.0,
) -> dict[str, Any]:
    windows = np.asarray(arrays["windows"], dtype=np.float32)
    targets = np.asarray(arrays["targets"], dtype=np.float32)
    video_ids = np.asarray(arrays.get("window_video_ids", []), dtype=str)
    labels = np.asarray(arrays.get("window_labels", []), dtype=str)
    names = tuple(baseline_names or DEFAULT_BASELINES)
    if windows.shape[0] == 0:
        empty = _metrics(np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        return {name: empty for name in names}
    metrics: dict[str, Any] = {}
    for name in names:
        pred = baseline_prediction(
            windows,
            name,
            moving_average_window=moving_average_window,
            prediction_horizon_frames=prediction_horizon_frames,
            frame_rate_hz=frame_rate_hz,
        )
        metrics[str(name)] = _with_groups(pred, targets, video_ids, labels)
    return metrics


def baseline_prediction(
    windows: np.ndarray,
    baseline: str,
    *,
    moving_average_window: int | None = None,
    prediction_horizon_frames: int = 1,
    frame_rate_hz: float = 50.0,
    reaction_rate_hz: float | None = None,
    decay_alpha: float | None = None,
) -> np.ndarray:
    """Return a clipped pixel-space baseline prediction for each window."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("windows must have shape (n, time, channels, height, width).")
    name = str(baseline).strip().lower()
    horizon = max(1, int(prediction_horizon_frames))
    if name == "persistence":
        return values[:, -1].astype(np.float32, copy=False)
    if name == "moving_average":
        k = max(1, int(moving_average_window or values.shape[1]))
        return values[:, -k:].mean(axis=1).astype(np.float32, copy=False)
    if name == "linear_extrapolation":
        if values.shape[1] < 2:
            return np.clip(values[:, -1], 0.0, 1.0).astype(np.float32, copy=False)
        return np.clip(values[:, -1] + horizon * (values[:, -1] - values[:, -2]), 0.0, 1.0).astype(np.float32, copy=False)
    if name in {"mean_delta", "average_delta"}:
        if values.shape[1] < 2:
            return np.clip(values[:, -1], 0.0, 1.0).astype(np.float32, copy=False)
        delta = np.diff(values, axis=1).mean(axis=1)
        return np.clip(values[:, -1] + horizon * delta, 0.0, 1.0).astype(np.float32, copy=False)
    if name in {"ar1_per_cell", "per_cell_ar1", "ar1"}:
        return _ar1_per_cell_prediction(values, horizon=horizon)
    if name.startswith("exponential_decay") or name.startswith("calcium_decay"):
        rate = _baseline_rate_hz(name, reaction_rate_hz=reaction_rate_hz)
        alpha = _decay_alpha(rate_hz=rate, frame_rate_hz=frame_rate_hz, decay_alpha=decay_alpha)
        local_baseline = values[:, : max(1, values.shape[1] // 2)].mean(axis=1)
        return np.clip(local_baseline + (alpha**horizon) * (values[:, -1] - local_baseline), 0.0, 1.0).astype(np.float32, copy=False)
    if name.startswith("lowpass") or name.startswith("ema"):
        rate = _baseline_rate_hz(name, reaction_rate_hz=reaction_rate_hz)
        alpha = _decay_alpha(rate_hz=rate, frame_rate_hz=frame_rate_hz, decay_alpha=decay_alpha)
        state = _ema_state(values, alpha=alpha)
        return np.clip(state + (alpha**horizon) * (values[:, -1] - state), 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported baseline prediction: {baseline}")


def _baseline_rate_hz(name: str, *, reaction_rate_hz: float | None) -> float:
    if reaction_rate_hz is not None:
        return float(reaction_rate_hz)
    if "30hz" in name or "30_hz" in name:
        return 30.0
    if "10hz" in name or "10_hz" in name:
        return 10.0
    return 10.0


def _decay_alpha(*, rate_hz: float, frame_rate_hz: float, decay_alpha: float | None) -> float:
    if decay_alpha is not None:
        return float(np.clip(decay_alpha, 0.0, 1.0))
    frame_rate = max(float(frame_rate_hz), 1e-6)
    rate = max(float(rate_hz), 1e-6)
    return float(np.exp(-rate / frame_rate))


def _ema_state(values: np.ndarray, *, alpha: float) -> np.ndarray:
    state = values[:, 0].astype(np.float32, copy=True)
    update = 1.0 - float(alpha)
    for index in range(1, values.shape[1]):
        state = float(alpha) * state + update * values[:, index]
    return state.astype(np.float32, copy=False)


def _ar1_per_cell_prediction(values: np.ndarray, *, horizon: int) -> np.ndarray:
    if values.shape[1] < 2:
        return values[:, -1].astype(np.float32, copy=False)
    samples = int(values.shape[1] - 1)
    shape = values[:, -1].shape
    sum_prev = np.zeros(shape, dtype=np.float32)
    sum_next = np.zeros(shape, dtype=np.float32)
    sum_prev_sq = np.zeros(shape, dtype=np.float32)
    sum_prev_next = np.zeros(shape, dtype=np.float32)
    for index in range(samples):
        prev = values[:, index]
        nxt = values[:, index + 1]
        sum_prev += prev
        sum_next += nxt
        sum_prev_sq += prev * prev
        sum_prev_next += prev * nxt
    prev_mean = sum_prev / float(samples)
    next_mean = sum_next / float(samples)
    denom = sum_prev_sq - (sum_prev * sum_prev) / float(samples)
    numer = sum_prev_next - (sum_prev * sum_next) / float(samples)
    phi = np.divide(numer, denom, out=np.ones_like(numer, dtype=np.float32), where=denom > 1e-8)
    phi = np.clip(phi, -0.99, 0.99).astype(np.float32, copy=False)
    intercept = (next_mean - phi * prev_mean).astype(np.float32, copy=False)
    fixed_point = np.divide(intercept, 1.0 - phi, out=values[:, -1].copy(), where=np.abs(1.0 - phi) > 1e-6)
    pred = fixed_point + (phi ** int(horizon)) * (values[:, -1] - fixed_point)
    return np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)


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
