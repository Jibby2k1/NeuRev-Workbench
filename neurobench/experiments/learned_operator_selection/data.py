"""Frozen Spon data, frame, label, and split helpers."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
from neurobench.experiments.learnable_contrast.core import load_labels
from neurobench.experiments.pairwise_separation.evaluation import event_intervals

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def ui_inclusive_to_zero_half_open(start_ui: int, end_ui: int) -> tuple[int, int]:
    if start_ui < 1 or end_ui < start_ui: raise ValueError("invalid one-based inclusive frame interval")
    return start_ui - 1, end_ui

def load_and_validate_labels(path: Path, summary_path: Path, shape_yx: tuple[int, int]) -> list[dict[str, Any]]:
    labels = load_labels(path); summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(labels) != int(summary["total_point_window_labels"]) or len({row["roi_identity"] for row in labels}) != int(summary["unique_roi_coordinates"]):
        raise ValueError("label TSV and label summary disagree")
    height, width = shape_yx
    if any(not (0 <= row["x_px"] < width and 0 <= row["y_px"] < height) for row in labels):
        raise ValueError("label coordinate outside source; x=column and y=row are required")
    return labels

def outer_burst_splits(labels: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    bursts = tuple(sorted({int(row["burst_id"]) for row in labels}))
    if bursts != (1, 2, 3, 4): raise ValueError("canonical evaluation requires bursts 1,2,3,4")
    return tuple({"outer_fold": heldout, "heldout_burst": heldout, "training_bursts": [b for b in bursts if b != heldout], "inner_rotations": [{"validation_burst": val, "fit_bursts": [b for b in bursts if b not in {heldout, val}]} for val in bursts if val != heldout]} for heldout in bursts)

def raw_direct_stack(source: np.ndarray, review_start_ui: int, review_end_ui: int, quiet_end_ui: int) -> tuple[np.ndarray, dict[str, float]]:
    start, stop = ui_inclusive_to_zero_half_open(review_start_ui, review_end_ui)
    raw = np.asarray(source[start:stop], dtype=np.float32); quiet_frames = quiet_end_ui - review_start_ui + 1
    baseline = np.median(raw[:quiet_frames], axis=0)
    low, high = np.percentile(raw[:quiet_frames, ::4, ::4], [1, 99.9]); scale = max(float(high - low), 1.0)
    return np.maximum((raw - baseline) / scale, 0).astype(np.float32), {"quiet_percentile_1": float(low), "quiet_percentile_99p9": float(high), "global_scale": scale}

def canonical_event_intervals(labels: list[dict[str, Any]], review_start_ui: int):
    return event_intervals(labels, review_start_ui)
