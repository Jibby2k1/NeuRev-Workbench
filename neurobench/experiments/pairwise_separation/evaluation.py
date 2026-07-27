"""Comparable quiet-calibrated evaluation for pairwise lanes and anchors."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one, temporal_pool

from .config import PairwiseSeparationConfig


QUIET_STARTS = (0, 24, 48, 53)
QUIET_DURATIONS = (24, 24, 28, 47)


def event_intervals(labels: list[dict[str, Any]], review_start_ui: int) -> dict[int, tuple[int, int]]:
    review_zero = review_start_ui - 1; result = {}
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        rows = [row for row in labels if int(row["burst_id"]) == burst]
        result[burst] = (int(rows[0]["start_frame_zero"])-review_zero,
                         int(rows[0]["stop_frame_zero_exclusive"])-review_zero)
    return result


def _maps(values: np.ndarray, labels: list[dict[str, Any]], config: PairwiseSeparationConfig, mode: str,
          tie_values: np.ndarray | None = None) -> tuple[list[tuple[np.ndarray,np.ndarray|None]], dict[int,tuple[np.ndarray,np.ndarray|None]]]:
    quiet = []
    for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS):
        frames = values[start:start+duration]; tie = None if tie_values is None else tie_values[start:start+duration].max(axis=0)
        quiet.append((temporal_pool(frames, mode), tie))
    events = {}
    for burst, (start, stop) in event_intervals(labels, config.frames.review_start_ui).items():
        frames = values[start:stop]; tie = None if tie_values is None else tie_values[start:stop].max(axis=0)
        events[burst] = (temporal_pool(frames, mode), tie)
    return quiet, events


def _calibrate_rank(quiet_maps, config):
    peaks = []
    for score, tie in quiet_maps:
        for value, x, y in extract_local_maxima(score, config.evaluation.nms_distance_px, limit=3000, tie_breaker=tie):
            peaks.append((value, float(tie[y,x]) if tie is not None else 0.0))
    peaks.sort(reverse=True); allowed=max(1,int(round(config.evaluation.quiet_false_peaks_per_map*len(quiet_maps))))
    if len(peaks)<=allowed: raise RuntimeError("Too few quiet candidates")
    return peaks[allowed]


def evaluate_lane(method_id: str, values: np.ndarray, labels: list[dict[str, Any]], config: PairwiseSeparationConfig,
                  *, binary: bool, tie_values: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str,np.ndarray]]:
    mode = "occupancy" if binary else "lme0.25"
    quiet, events = _maps(values, labels, config, mode, tie_values)
    cutoff = _calibrate_rank(quiet, config)
    folds=[]; candidates=[]; maps={}
    for burst,(score,tie) in events.items():
        ranked=extract_local_maxima(score,config.evaluation.nms_distance_px,limit=3000,tie_breaker=tie)
        peaks=[p for p in ranked if (p[0],float(tie[p[2],p[1]]) if tie is not None else 0.0)>cutoff]
        rows=[r for r in labels if int(r["burst_id"])==burst]
        matches, matched=match_peaks_one_to_one(peaks,rows,config.evaluation.primary_match_radius_px)
        folds.append({"burst_id":burst,"matched":len(matches),"labels":len(rows),"recall":len(matches)/len(rows),"candidates":len(peaks)})
        maps[f"burst_{burst}"]=score.astype(np.float32)
        for i,(value,x,y) in enumerate(peaks):
            nearest=min(math.hypot(x-r["x_px"],y-r["y_px"]) for r in rows)
            candidates.append({"lane":method_id,"frame_or_burst_id":burst,"score":value,"x_px":x,"y_px":y,
                "matched_known_label":i in matched,"nearest_known_label_px":nearest,"source_stratum":"event",
                "review_status":"unreviewed","review_label":"","review_note":"","interpretation":"known_match" if i in matched else "unknown_candidate"})
    return {"lane":method_id,"primary_match_radius_px":6,"quiet_rank_cutoff":{"score":cutoff[0],"tie":cutoff[1]},
        "outer_folds":folds,"mean_recall":float(np.mean([x["recall"] for x in folds])),
        "pooled_recall":sum(x["matched"] for x in folds)/sum(x["labels"] for x in folds),
        "total_matched":sum(x["matched"] for x in folds),"total_labels":sum(x["labels"] for x in folds),
        "total_event_candidates":sum(x["candidates"] for x in folds),
        "known_label_candidate_fraction_lower_bound":sum(x["matched"] for x in folds)/max(1,sum(x["candidates"] for x in folds)),
        "precision_identified":False}, candidates, maps
