"""Spatial and temporal error diagnostics for grid dynamics predictions."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def structured_prediction_error_metrics(
    *,
    pred_diff: np.ndarray,
    persistence_diff: np.ndarray,
    targets: np.ndarray,
    last_frames: np.ndarray,
    video_ids: np.ndarray,
    splits: Mapping[str, Any] | None,
    active_percentile: float = 90.0,
    top_activity_percent: float = 5.0,
    high_change_percentile: float = 90.0,
) -> dict[str, dict[str, Any]]:
    """Compute split-aware active-cell and high-change error diagnostics."""
    pred = np.asarray(pred_diff, dtype=np.float32)
    persist = np.asarray(persistence_diff, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    last = np.asarray(last_frames, dtype=np.float32)
    ids = np.asarray(video_ids, dtype=str)
    if pred.shape != persist.shape or pred.shape != target.shape or pred.shape != last.shape:
        raise ValueError("pred_diff, persistence_diff, targets, and last_frames must have matching shapes.")
    train_mask = _split_mask(ids, splits, "train", default_all=True)
    reference = target[train_mask] if np.any(train_mask) else target
    active_threshold = _percentile_threshold(reference, active_percentile)
    top_threshold = _percentile_threshold(reference, 100.0 - float(top_activity_percent))
    change_reference = np.abs(target[train_mask] - last[train_mask]) if np.any(train_mask) else np.abs(target - last)
    high_change_threshold = _percentile_threshold(change_reference, high_change_percentile)
    out: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test", "all"):
        if split_name == "all":
            mask = np.ones(ids.shape[0], dtype=bool)
        else:
            mask = _split_mask(ids, splits, split_name, default_all=False)
        out[split_name] = _structured_split_metrics(
            pred[mask],
            persist[mask],
            target[mask],
            last[mask],
            active_threshold=active_threshold,
            top_threshold=top_threshold,
            high_change_threshold=high_change_threshold,
        )
    out["thresholds"] = {
        "active_percentile": float(active_percentile),
        "active_threshold": float(active_threshold),
        "top_activity_percent": float(top_activity_percent),
        "top_activity_threshold": float(top_threshold),
        "high_change_percentile": float(high_change_percentile),
        "high_change_threshold": float(high_change_threshold),
    }
    return out


def promote_structured_error_metrics(metrics: dict[str, Any], structured: Mapping[str, Mapping[str, Any]]) -> None:
    """Promote key structured diagnostics to flat metric keys for TSV/dashboard use."""
    for split_name in ("train", "val", "test", "all"):
        split = structured.get(split_name, {})
        for key in (
            "active_cell_mse",
            "active_cell_persistence_mse",
            "active_cell_improvement_over_persistence_mse",
            "inactive_cell_mse",
            "top_activity_mse",
            "top_activity_improvement_over_persistence_mse",
            "high_change_mse",
            "high_change_improvement_over_persistence_mse",
            "active_cell_fraction",
            "high_change_fraction",
        ):
            if key in split:
                metrics[f"{split_name}_{key}"] = split[key]


def _structured_split_metrics(
    pred: np.ndarray,
    persist: np.ndarray,
    target: np.ndarray,
    last: np.ndarray,
    *,
    active_threshold: float,
    top_threshold: float,
    high_change_threshold: float,
) -> dict[str, Any]:
    if pred.size == 0:
        return {
            "window_count": 0,
            "cell_count": 0,
            "active_cell_count": 0,
            "active_cell_fraction": 0.0,
            "active_cell_mse": None,
            "active_cell_persistence_mse": None,
            "active_cell_improvement_over_persistence_mse": None,
            "inactive_cell_mse": None,
            "top_activity_mse": None,
            "top_activity_improvement_over_persistence_mse": None,
            "high_change_mse": None,
            "high_change_improvement_over_persistence_mse": None,
            "high_change_fraction": 0.0,
        }
    active = target >= float(active_threshold)
    top = target >= float(top_threshold)
    change = np.abs(target - last)
    high_change = change >= float(high_change_threshold)
    inactive = ~active
    pred_sq = pred * pred
    persist_sq = persist * persist
    return {
        "window_count": int(pred.shape[0]),
        "cell_count": int(pred.size),
        "active_cell_count": int(active.sum()),
        "active_cell_fraction": float(active.mean()) if active.size else 0.0,
        "active_cell_mse": _masked_mean_sq(pred_sq, active),
        "active_cell_persistence_mse": _masked_mean_sq(persist_sq, active),
        "active_cell_improvement_over_persistence_mse": _improvement(pred_sq, persist_sq, active),
        "inactive_cell_mse": _masked_mean_sq(pred_sq, inactive),
        "inactive_cell_persistence_mse": _masked_mean_sq(persist_sq, inactive),
        "inactive_cell_improvement_over_persistence_mse": _improvement(pred_sq, persist_sq, inactive),
        "top_activity_cell_count": int(top.sum()),
        "top_activity_mse": _masked_mean_sq(pred_sq, top),
        "top_activity_persistence_mse": _masked_mean_sq(persist_sq, top),
        "top_activity_improvement_over_persistence_mse": _improvement(pred_sq, persist_sq, top),
        "high_change_cell_count": int(high_change.sum()),
        "high_change_fraction": float(high_change.mean()) if high_change.size else 0.0,
        "high_change_mse": _masked_mean_sq(pred_sq, high_change),
        "high_change_persistence_mse": _masked_mean_sq(persist_sq, high_change),
        "high_change_improvement_over_persistence_mse": _improvement(pred_sq, persist_sq, high_change),
    }


def _masked_mean_sq(squared: np.ndarray, mask: np.ndarray) -> float | None:
    count = int(mask.sum())
    if count == 0:
        return None
    return float(np.mean(squared[mask]))


def _improvement(pred_sq: np.ndarray, persist_sq: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    return float(np.mean(persist_sq[mask]) - np.mean(pred_sq[mask]))


def _percentile_threshold(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, float(np.clip(percentile, 0.0, 100.0))))


def _split_mask(video_ids: np.ndarray, splits: Mapping[str, Any] | None, split_name: str, *, default_all: bool) -> np.ndarray:
    if not isinstance(splits, Mapping):
        return np.ones(video_ids.shape[0], dtype=bool) if default_all else np.zeros(video_ids.shape[0], dtype=bool)
    candidates = [split_name, f"{split_name}_video_ids", f"{split_name}_videos"]
    selected: set[str] | None = None
    for key in candidates:
        value = splits.get(key)
        if isinstance(value, Mapping):
            nested = value.get("video_ids") or value.get("videos") or value.get("ids")
            if nested is not None:
                selected = {str(item) for item in nested}
                break
        elif isinstance(value, (list, tuple, set)):
            selected = {str(item) for item in value}
            break
    if selected is None:
        assignments = splits.get("assignments")
        if isinstance(assignments, Mapping):
            return np.asarray([str(assignments.get(str(video_id), "")) == split_name for video_id in video_ids], dtype=bool)
        return np.ones(video_ids.shape[0], dtype=bool) if default_all else np.zeros(video_ids.shape[0], dtype=bool)
    return np.asarray([str(video_id) in selected for video_id in video_ids], dtype=bool)
