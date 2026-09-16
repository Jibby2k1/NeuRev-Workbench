"""Threshold sweeps with exact synthetic active-frame truth and frozen proposals.

Maintained NMS and score-ranked matching both have threshold-prefix invariance:
raising a strict cutoff removes a suffix, so assignments of remaining rows are
unchanged. Extract once at tau=0, seal before joining truth, then evaluate the
nonnegative sweep. The seal binds the metric-relevant candidate fields; callers
must separately hash source files and record their actual scoring-before-truth
chronology. Nothing here estimates precision from sparse real annotations.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .two_stencil_evaluation import extract_frame_candidates, evaluate_occurrence_windows


TRUTH_MODES = ("fully_synthetic", "exhaustive_real", "sparse_real")


@dataclass(frozen=True)
class Candidate:
    proposal_id: str
    source_frame_ui: int
    score: float
    x_px: float
    y_px: float

    def as_row(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, "source_frame_ui": self.source_frame_ui,
                "score": self.score, "x_px": self.x_px, "y_px": self.y_px}


@dataclass(frozen=True)
class SealedCandidates:
    rows: tuple[Candidate, ...]
    source_frames_ui: tuple[int, ...]
    metric_candidate_sha256: str
    minimum_threshold_z: float = 0.0


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    x = float(value)
    if not math.isfinite(x) or x < 1 or x != math.floor(x):
        raise ValueError(f"{name} must be a positive integer")
    return int(x)


def _coordinate(value: Any) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError("coordinates must be finite and nonnegative")
    return x


def threshold_grid() -> tuple[float, ...]:
    """Strict cutoffs 0..10 by .25, 12,15,20 and infinity (no output)."""
    return tuple(i / 4 for i in range(41)) + (12.0, 15.0, 20.0, math.inf)


def _threshold(value: Any) -> float:
    x = float(value)
    if math.isnan(x) or x < 0:
        raise ValueError("threshold must be nonnegative, or positive infinity")
    return x


def _threshold_fields(tau: float) -> dict[str, Any]:
    return {"threshold_z": tau if math.isfinite(tau) else None,
            "threshold_label": format(tau, "g") if math.isfinite(tau) else "no_output",
            "no_output_endpoint": math.isinf(tau), "threshold_comparison": "score > threshold"}


def extract_nms_prefix(
    score: Any, *, source_frame_ui: int, cell_id: str = "cell",
    frame_interval_ms: float = 20.0,
) -> list[dict[str, Any]]:
    """Complete maintained 6px NMS at strict tau=0; no label arguments."""
    return extract_frame_candidates(score, source_frame_ui=source_frame_ui,
                                    threshold_z=0.0, cell_id=cell_id,
                                    frame_interval_ms=frame_interval_ms)


def filter_nms_prefix(rows: Iterable[Mapping[str, Any]], threshold_z: float) -> list[dict[str, Any]]:
    """Reuse tau=0 rows at a larger threshold, retaining original proposal IDs."""
    tau = _threshold(threshold_z)
    return [{**row, "threshold_z": tau} for row in rows if float(row["score"]) > tau]


def seal_candidates(
    rows: Iterable[Mapping[str, Any]], *, source_frames_ui: Sequence[int],
) -> SealedCandidates:
    """Copy candidate core fields into an immutable stream, before truth loading.

    The declared frames include frames with zero proposals. They are the exposure
    denominator; passing only frames containing candidates would inflate rates.
    """
    frames = tuple(_integer(frame, "source_frame_ui") for frame in source_frames_ui)
    if not frames or any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("source_frames_ui must be nonempty and strictly increasing")
    allowed = set(frames)
    ids: set[str] = set()
    copied = []
    for row in rows:
        identity = str(row["proposal_id"])
        frame = _integer(row["source_frame_ui"], "source_frame_ui")
        score = float(row["score"])
        if not identity or identity in ids or frame not in allowed:
            raise ValueError("candidate IDs must be unique and frames must be in the declared exposure")
        if not math.isfinite(score) or score <= 0:
            raise ValueError("tau=0 candidate prefix must have finite strictly positive scores")
        ids.add(identity)
        copied.append(Candidate(identity, frame, score, _coordinate(row["x_px"]), _coordinate(row["y_px"])))
    copied.sort(key=lambda row: (row.source_frame_ui, -row.score, row.y_px, row.x_px, row.proposal_id))
    payload = {"source_frames_ui": frames, "minimum_threshold_z": 0.0,
               "candidate_fields": [row.as_row() for row in copied]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return SealedCandidates(tuple(copied), frames, digest)


def _truth(
    sealed: SealedCandidates, active_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]], truth_mode: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    if not isinstance(sealed, SealedCandidates):
        raise TypeError("seal the proposal stream before joining truth")
    if truth_mode not in TRUTH_MODES:
        raise ValueError(f"truth_mode must be one of {TRUTH_MODES}")
    if truth_mode == "sparse_real" and (active_rows or event_rows):
        raise ValueError("sparse real windows are not active-frame truth; use evaluate_sparse_occurrences")
    frames = set(sealed.source_frames_ui)
    events: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        event_id = str(row.get("event_id", row.get("observation_id", "")))
        start = _integer(row["source_start_ui"], "source_start_ui")
        stop = _integer(row["source_stop_ui"], "source_stop_ui")
        roi = str(row["canonical_roi_id"])
        if not event_id or event_id in events or not roi or stop < start or any(
            frame not in frames for frame in range(start, stop + 1)
        ):
            raise ValueError("event IDs must be unique and exact active extents must lie in application")
        events[event_id] = {**row, "event_id": event_id, "source_start_ui": start,
                            "source_stop_ui": stop, "canonical_roi_id": roi}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    occupied: set[tuple[str, int]] = set()
    event_counts: dict[str, int] = defaultdict(int)
    for row in active_rows:
        frame = _integer(row["source_frame_ui"], "source_frame_ui")
        event_id = str(row.get("event_id", row.get("observation_id", "")))
        roi = str(row["canonical_roi_id"])
        event = events.get(event_id)
        if event is None or frame not in frames or roi != event["canonical_roi_id"] or not (
            event["source_start_ui"] <= frame <= event["source_stop_ui"]
        ):
            raise ValueError("each active row must bind to its event, ROI and exact active extent")
        if (roi, frame) in occupied:
            raise ValueError("duplicate active-region truth for a canonical ROI in one frame")
        occupied.add((roi, frame))
        event_counts[event_id] += 1
        by_frame[frame].append({**row, "source_frame_ui": frame, "event_id": event_id,
                                "canonical_roi_id": roi, "x_px": _coordinate(row["x_px"]),
                                "y_px": _coordinate(row["y_px"])})
    if any(event_counts[key] != row["source_stop_ui"] - row["source_start_ui"] + 1
           for key, row in events.items()):
        raise ValueError("active truth must contain exactly one ROI row in every declared event frame")
    return by_frame, events


def _classify(
    sealed: SealedCandidates, active_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]], *, truth_mode: str, match_radius_px: float,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    radius = float(match_radius_px)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("match_radius_px must be finite and positive")
    by_frame, events = _truth(sealed, active_rows, event_rows, truth_mode)
    used: dict[int, set[str]] = defaultdict(set)
    classified = []
    for candidate in sealed.rows:
        labels = by_frame.get(candidate.source_frame_ui, [])
        nearby = sorted((
            (math.hypot(candidate.x_px - row["x_px"], candidate.y_px - row["y_px"]),
             row["canonical_roi_id"], row["event_id"])
            for row in labels
        ), key=lambda item: item)
        nearby = [item for item in nearby if item[0] <= radius]
        available = [item for item in nearby if item[1] not in used[candidate.source_frame_ui]]
        matched = available[0] if available else None
        if matched is not None:
            used[candidate.source_frame_ui].add(matched[1])
        kind = ("true_positive" if matched else "unknown" if truth_mode == "sparse_real"
                else "duplicate_near_active_region" if nearby else "false_positive_other")
        classified.append({**candidate.as_row(), "classification": kind,
                           "is_true_positive": matched is not None,
                           "is_duplicate_near_active_region": bool(nearby and not matched),
                           "matched_event_id": matched[2] if matched else None,
                           "matched_canonical_roi_id": matched[1] if matched else None,
                           "match_distance_px": matched[0] if matched else None})
    return classified, by_frame, events


def _count_above(sorted_scores: np.ndarray, tau: float) -> int:
    return int(len(sorted_scores) - np.searchsorted(sorted_scores, tau, side="right"))


def evaluate_threshold_sweep(
    sealed: SealedCandidates, active_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]], *, thresholds: Sequence[float] | None = None,
    frame_interval_s: float = 0.02, truth_mode: str = "fully_synthetic",
    match_radius_px: float = 6.0,
) -> dict[str, Any]:
    """Exact active-frame precision/sensitivity and descriptive event coverage.

    Each frame's score-ranked proposals greedily claim the nearest unassigned
    active region within an inclusive 6px disk (ties: canonical ROI, event ID).
    Remaining proposals, including same-frame duplicates, are false positives
    only for exhaustive truth. Event recovery is window coverage; first delay
    measures the first matched active frame and is conditional on recovery.
    No count of true-negative image pixels, classical FPR or event precision is
    invented. For sparse_real with no truth rows, only unknown burden is returned.
    """
    dt = float(frame_interval_s)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("frame_interval_s must be finite and positive")
    cutoffs = tuple(_threshold(t) for t in (threshold_grid() if thresholds is None else thresholds))
    if not cutoffs or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("thresholds must be nonempty and distinct")
    classified, by_frame, events = _classify(sealed, active_rows, event_rows,
                                             truth_mode=truth_mode, match_radius_px=match_radius_px)
    exhaustive = truth_mode != "sparse_real"
    scores = np.sort(np.asarray([row["score"] for row in classified], dtype=np.float64))
    tp_scores = np.sort(np.asarray([row["score"] for row in classified if row["is_true_positive"]]))
    dup_scores = np.sort(np.asarray([row["score"] for row in classified if row["is_duplicate_near_active_region"]]))
    hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in classified:
        if row["is_true_positive"]:
            hits[row["matched_event_id"]].append(row)
    frame_count = len(sealed.source_frames_ui)
    exposure_s = frame_count * dt
    truth_count = sum(map(len, by_frame.values()))
    curve, event_output = [], []
    for tau in cutoffs:
        count, tp, duplicates = (_count_above(values, tau) for values in (scores, tp_scores, dup_scores))
        delays = []
        recovered = 0
        for event_id, event in events.items():
            surviving = [row for row in hits[event_id] if row["score"] > tau]
            first = min((row["source_frame_ui"] for row in surviving), default=None)
            delay = None if first is None else (first - event["source_start_ui"]) * dt * 1000
            if first is not None:
                recovered += 1
                delays.append(delay)
            event_output.append({**_threshold_fields(tau), "event_id": event_id,
                                 "canonical_roi_id": event["canonical_roi_id"],
                                 "source_start_ui": event["source_start_ui"],
                                 "source_stop_ui": event["source_stop_ui"],
                                 "active_frame_count": event["source_stop_ui"] - event["source_start_ui"] + 1,
                                 "matched_active_frame_count": len(surviving),
                                 "recovered": first is not None, "first_matched_source_frame_ui": first,
                                 "first_delay_ms": delay, "truth_mode": truth_mode})
        fp = count - tp if exhaustive else None
        curve.append({
            **_threshold_fields(tau), "truth_mode": truth_mode,
            "metric_candidate_sha256": sealed.metric_candidate_sha256,
            "application_frame_count": frame_count, "exposure_seconds": exposure_s,
            "proposal_count": count, "proposals_per_frame": count / frame_count,
            "proposals_per_second": count / exposure_s,
            "active_region_frame_count": truth_count if exhaustive else None,
            "true_positive_count": tp if exhaustive else None,
            "false_positive_count": fp, "false_negative_count": truth_count - tp if exhaustive else None,
            "precision": tp / count if exhaustive and count else None,
            "framewise_sensitivity": tp / truth_count if exhaustive and truth_count else None,
            "false_proposals_per_second": fp / exposure_s if exhaustive else None,
            "duplicate_near_active_region_count": duplicates if exhaustive else None,
            "unmatched_unknown_count": count if not exhaustive else None,
            "event_count": len(events) if exhaustive else None,
            "recovered_event_count": recovered if exhaustive else None,
            "event_window_coverage": recovered / len(events) if events else None,
            "first_delay_ms_mean_among_recovered": float(np.mean(delays)) if delays else None,
            "first_delay_ms_median_among_recovered": float(np.median(delays)) if delays else None,
            "first_delay_recovered_denominator": len(delays),
            "match_radius_px": float(match_radius_px), "fpr": None,
            "fpr_not_reported_reason": "no declared true-negative spacetime opportunity unit",
            "event_precision": None, "event_precision_not_reported_reason": "no causal event linker",
            "event_recovery_definition": "at least one matched active-region frame in its finite event window",
            "false_positive_scope": "unmatched frame proposals including duplicates" if exhaustive else "not_identifiable",
        })
    return {"curve_rows": curve, "event_rows": event_output,
            "definitions": {"matching": "per-frame score/y/x/id order; nearest unassigned active region within inclusive radius; ROI/event tie break",
                            "duplicate": "unmatched proposal within radius of an already assigned active region in the same frame",
                            "threshold_prefix_reused": True, "event_onset_classification": False,
                            "calibration_burden_implies_application_burden": False,
                            "metric_candidate_sha256": sealed.metric_candidate_sha256}}


def evaluate_framewise(
    sealed: SealedCandidates, active_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]], *, threshold_z: float = 0.0,
    frame_interval_s: float = 0.02, truth_mode: str = "fully_synthetic",
    match_radius_px: float = 6.0,
) -> dict[str, Any]:
    """One operating point with full proposal/frame/event audit tables."""
    tau = _threshold(threshold_z)
    sweep = evaluate_threshold_sweep(sealed, active_rows, event_rows, thresholds=(tau,),
                                     frame_interval_s=frame_interval_s, truth_mode=truth_mode,
                                     match_radius_px=match_radius_px)
    classified, by_frame, _ = _classify(sealed, active_rows, event_rows,
                                       truth_mode=truth_mode, match_radius_px=match_radius_px)
    proposals = [row for row in classified if row["score"] > tau]
    frame_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        frame_groups[row["source_frame_ui"]].append(row)
    frame_output = []
    for frame in sealed.source_frames_ui:
        rows = frame_groups[frame]
        tp = sum(row["is_true_positive"] for row in rows)
        exhaustive = truth_mode != "sparse_real"
        frame_output.append({"source_frame_ui": frame, "proposal_count": len(rows),
                             "active_region_count": len(by_frame.get(frame, [])) if exhaustive else None,
                             "true_positive_count": tp if exhaustive else None,
                             "false_positive_count": len(rows) - tp if exhaustive else None,
                             "false_negative_count": len(by_frame.get(frame, [])) - tp if exhaustive else None,
                             "unmatched_unknown_count": len(rows) if not exhaustive else None})
    return {"summary": sweep["curve_rows"][0], "frame_rows": frame_output,
            "event_rows": sweep["event_rows"], "proposal_rows": proposals,
            "definitions": sweep["definitions"]}


def evaluate_sparse_occurrences(
    sealed: SealedCandidates, positives: Sequence[Mapping[str, Any]], *,
    burst_intervals_ui: Mapping[Any, Sequence[int]], threshold_z: float = 0.0,
    frame_interval_s: float = 0.02,
) -> dict[str, Any]:
    """Real sparse broad windows retain the established occurrence-site endpoint.

    This function does not expand an uncertain burst window into active-frame
    truth. Unmatched consolidated sites remain unknown. It supplies no precision,
    sensitivity of all real events, FPR, active-frame recall or onset latency.
    """
    if not isinstance(sealed, SealedCandidates):
        raise TypeError("seal the proposal stream before joining annotations")
    tau = _threshold(threshold_z)
    rows = [row.as_row() for row in sealed.rows if row.score > tau]
    result = evaluate_occurrence_windows(rows, positives, burst_intervals_ui=burst_intervals_ui)
    result["truth_mode"] = "sparse_real"
    result["threshold"] = _threshold_fields(tau)
    result["metric_candidate_sha256"] = sealed.metric_candidate_sha256
    result["claim_boundaries"] = {
        "precision": None, "false_positive_count": None, "fpr": None,
        "framewise_sensitivity": None, "event_onset_latency": None,
        "unmatched_status": "unknown", "endpoint": "known-positive broad-window site coverage",
    }
    dt = float(frame_interval_s)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("frame_interval_s must be finite and positive")
    result["application_proposal_count"] = len(rows)
    result["application_proposals_per_second"] = len(rows) / (len(sealed.source_frames_ui) * dt)
    return result


def paired_summary(
    curve_rows: Sequence[Mapping[str, Any]], *, baseline_spec_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Paired scene-level differences, stratified by truth mode and template.

    Callers add scene_id, template_id, seed and spec_id to each curve row. Seeds
    within templates are replicates of a simulation design; frames or events are
    not independent animals/recordings. No mixed-truth precision is produced.
    """
    keyed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    metrics = ("precision", "framewise_sensitivity", "event_window_coverage",
               "false_proposals_per_second", "proposals_per_frame")
    for row in curve_rows:
        key = (row["truth_mode"], row["template_id"], row["scene_id"],
               int(row["seed"]), row["threshold_label"], row["spec_id"])
        if key in keyed:
            raise ValueError("duplicate scene/spec/threshold summary")
        keyed[key] = row
    paired = []
    for key, row in keyed.items():
        if key[-1] == baseline_spec_id:
            continue
        baseline = keyed.get((*key[:-1], baseline_spec_id))
        if baseline is None:
            raise ValueError("each comparison requires the same scene, seed, truth and threshold baseline")
        paired.append({"truth_mode": key[0], "template_id": key[1], "scene_id": key[2],
                       "seed": key[3], "threshold_label": key[4], "spec_id": key[5],
                       "baseline_spec_id": baseline_spec_id,
                       **{f"delta_{name}": None if row.get(name) is None or baseline.get(name) is None
                          else float(row[name]) - float(baseline[name]) for name in metrics}})
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(row["truth_mode"], row["template_id"], row["threshold_label"], row["spec_id"])].append(row)
    aggregate = []
    for key, rows in groups.items():
        item = {"truth_mode": key[0], "template_id": key[1], "threshold_label": key[2],
                "spec_id": key[3], "baseline_spec_id": baseline_spec_id,
                "paired_scene_seed_count": len(rows), "replicate_unit": "scene/seed within template"}
        for metric in metrics:
            values = [row[f"delta_{metric}"] for row in rows if row[f"delta_{metric}"] is not None]
            item[f"mean_delta_{metric}"] = float(np.mean(values)) if values else None
            item[f"valid_pairs_{metric}"] = len(values)
        aggregate.append(item)
    return {"paired_rows": paired, "aggregate_rows": aggregate}
