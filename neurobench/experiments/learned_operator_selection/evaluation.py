"""Canonical deterministic KPR@58 and Q1 evaluator wrapper."""
from __future__ import annotations
import hashlib, json
from typing import Any
import numpy as np
from neurobench.experiments.pairwise_separation.evaluation import QUIET_DURATIONS, QUIET_STARTS
from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one, temporal_pool
from .config import LearnedOperatorConfig
from .data import canonical_event_intervals

def deterministic_peaks(score: np.ndarray, distance: int, *, limit: int = 3000):
    return extract_local_maxima(score, distance, limit=limit, tie_breaker=np.zeros_like(score, dtype=np.uint8))

def _quiet_cutoff(stack: np.ndarray, config: LearnedOperatorConfig) -> float:
    values = sorted((peak[0] for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS) for peak in deterministic_peaks(temporal_pool(stack[start:start + duration], config.evaluation.temporal_pool), config.evaluation.nms_distance_px)), reverse=True)
    allowed = max(1, int(round(config.evaluation.quiet_false_peaks_per_map * len(QUIET_STARTS))))
    if len(values) <= allowed: raise RuntimeError("too few quiet peaks for Q1 calibration")
    return float(np.nextafter(values[allowed], np.inf))

def evaluate_stack(stack: np.ndarray, labels: list[dict[str, Any]], config: LearnedOperatorConfig) -> dict[str, Any]:
    if stack.ndim != 3 or not np.isfinite(stack).all(): raise ValueError("evaluation stack must be finite TYX")
    cutoff = _quiet_cutoff(stack, config); folds = []; rankings = {}
    for burst, (start, stop) in sorted(canonical_event_intervals(labels, config.frames.review_start_ui).items()):
        score = temporal_pool(stack[start:stop], config.evaluation.temporal_pool)
        ranked = deterministic_peaks(score, config.evaluation.nms_distance_px)
        fixed = ranked[:config.evaluation.fixed_candidates_per_burst]; q1 = [peak for peak in ranked if peak[0] >= cutoff]
        rows = [row for row in labels if int(row["burst_id"]) == burst]
        fixed_matches, _ = match_peaks_one_to_one(fixed, rows, config.evaluation.match_radius_px)
        q1_matches, _ = match_peaks_one_to_one(q1, rows, config.evaluation.match_radius_px)
        folds.append({"burst_id": burst, "labels": len(rows), "matches_at_58": len(fixed_matches), "recall_at_58": len(fixed_matches)/len(rows), "candidates_at_58": len(fixed), "matches_q1": len(q1_matches), "recall_q1": len(q1_matches)/len(rows), "candidates_q1": len(q1)})
        rankings[str(burst)] = [[float(v), int(x), int(y)] for v, x, y in fixed]
    return {
        "metric": "Macro-KPR@58", "outer_folds": folds,
        "macro_kpr_at_58": float(np.mean([r["recall_at_58"] for r in folds])),
        "pooled_kpr_at_58": sum(r["matches_at_58"] for r in folds)/sum(r["labels"] for r in folds),
        "matches_at_58": sum(r["matches_at_58"] for r in folds), "labels": sum(r["labels"] for r in folds),
        "per_burst_matches_at_58": [r["matches_at_58"] for r in folds], "per_burst_recall_at_58": [r["recall_at_58"] for r in folds],
        "macro_kpr_q1": float(np.mean([r["recall_q1"] for r in folds])), "matches_q1": sum(r["matches_q1"] for r in folds), "candidates_q1": sum(r["candidates_q1"] for r in folds),
        "q1_cutoff": cutoff, "top58_candidate_hash": hashlib.sha256(json.dumps(rankings, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "candidate_interpretation": "unmatched candidates are unknown, not false positives", "finite_output_fraction": float(np.mean(np.isfinite(stack))),
    }
