"""Framewise calibration and descriptive occurrence-window evaluation.

These helpers do not select a representation or infer event onsets. Calibration
accepts only calibration score frames. Annotation coordinates enter only the
separate occurrence-window evaluation after the proposal stream is frozen.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from neurobench.metrics.sparse_detection import match_peaks_one_to_one

from .evaluation import strict_separated_nms


TARGET_PROPOSALS_PER_FRAME = (0.25, 0.5, 1.0, 2.0, 5.0)
CALIBRATION_BURDEN_UNIT = "nms_proposals_per_calibration_score_frame"
NMS_DISTANCE_PX = 6
NMS_BORDER_PX = 6
MATCH_RADIUS_PX = 6.0
NMS_SEMANTICS = (
    "square_13x13_local_max_prefilter_then_score_y_x_sorted_greedy_"
    "euclidean_strictly_greater_than_6px_with_6px_border_exclusion"
)


@dataclass(frozen=True)
class FramewiseCalibration:
    """Calibration-only frozen operating points with exact selected counts."""

    operating_points: tuple[Mapping[str, Any], ...]
    threshold_candidates: tuple[float, ...]
    evaluated_thresholds: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]

    def threshold_for(self, target_proposals_per_frame: float) -> float:
        target = float(target_proposals_per_frame)
        for row in self.operating_points:
            if float(row["target_proposals_per_frame"]) == target:
                return float(row["threshold_z"])
        raise KeyError(f"undeclared target proposals/frame: {target}")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    number = float(value)
    if not math.isfinite(number) or number != math.floor(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _score_frame(score: Any) -> np.ndarray:
    values = np.asarray(score)
    if values.ndim != 2 or min(values.shape) <= 2 * NMS_BORDER_PX:
        raise ValueError("score must be a YX frame with a nonempty 6px interior")
    if not np.isfinite(values).all():
        raise ValueError("score frame must be finite")
    return values


def _calibration_quantiles(
    scores: np.ndarray, *, max_candidates: int, max_samples: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic, bounded-memory calibration-only threshold proposals."""
    if max_candidates < 16 or max_samples < 16:
        raise ValueError("threshold candidate and sample limits must be >= 16")
    interior_shape = (scores.shape[1] - 12, scores.shape[2] - 12)
    pixels_per_frame = math.prod(interior_shape)
    population = len(scores) * pixels_per_frame
    sample_count = min(population, max_samples)
    ordinals = np.linspace(0, population - 1, sample_count, dtype=np.int64)
    sample = np.empty(sample_count, dtype=np.float64)
    minimum, maximum = math.inf, -math.inf
    for index, frame in enumerate(scores):
        values = _score_frame(frame)
        interior = values[6:-6, 6:-6]
        minimum = min(minimum, float(interior.min()))
        maximum = max(maximum, float(interior.max()))
        left = int(np.searchsorted(ordinals, index * pixels_per_frame))
        right = int(np.searchsorted(ordinals, (index + 1) * pixels_per_frame))
        offsets = ordinals[left:right] - index * pixels_per_frame
        sample[left:right] = interior[
            offsets // interior_shape[1], offsets % interior_shape[1]
        ]
    # Include exact extrema even if the deterministic quantile sample omits them.
    # The upper endpoint always provides an exact zero-proposal operating point.
    broad_count = max(4, (max_candidates - 3) // 4)
    tail_count = max_candidates - broad_count - 3
    broad = np.linspace(0.0, 0.9, broad_count, endpoint=False)
    tail = 1.0 - np.geomspace(1.0 / sample_count, 0.1, tail_count)
    quantiles = np.unique(np.clip(np.concatenate((broad, tail, [1.0])), 0, 1))
    proposed = np.quantile(sample, quantiles, method="nearest")
    below_minimum = float(np.nextafter(np.float64(minimum), -np.inf))
    if not math.isfinite(below_minimum):
        below_minimum = minimum
    thresholds = np.unique(np.concatenate(([below_minimum, maximum], proposed)))
    return thresholds, {
        "threshold_grid_method": "deterministic_even_ordinal_sample_nearest_quantiles_plus_exact_extrema",
        "quantile_sample_count": sample_count,
        "eligible_calibration_pixel_count": population,
        "quantile_sampling_used": sample_count < population,
        "max_threshold_candidates": max_candidates,
        "threshold_candidate_count": len(thresholds),
        "threshold_candidates_from_calibration_only": True,
    }


def calibrate_tau(
    calibration_scores: Any,
    *,
    source_frames_ui: Sequence[int],
    target_proposals_per_frame: Sequence[float] = TARGET_PROPOSALS_PER_FRAME,
    max_threshold_candidates: int = 129,
    max_quantile_samples: int = 1_000_000,
) -> FramewiseCalibration:
    """Freeze cutoffs at explicit target NMS proposals/calibration score frame.

    The source frame sequence must identify only the supplied calibration data;
    the common adjacent-difference experiment passes UI 2..100 (99 scores).
    Initialization is not assumed event-free. For each target, choose the
    threshold with the largest exact count at or below target * frame_count;
    ties choose the lower threshold. Actual application burden is uncontrolled.

    Maintained exact NMS has threshold-prefix invariance. Its sorted prefix is
    extracted once per calibration frame and counts for all grid thresholds are
    obtained from that prefix. A saturated prefix is only a count lower bound;
    such a row can never be accepted as a calibrated operating point.
    """
    scores = np.asarray(calibration_scores)
    if scores.ndim != 3 or len(scores) < 1:
        raise ValueError("calibration_scores must be a nonempty TYX array")
    frames = tuple(_integer(x, "source_frame_ui", minimum=1) for x in source_frames_ui)
    if len(frames) != len(scores) or any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("source_frames_ui must be strictly increasing and align with scores")
    targets = tuple(float(q) for q in target_proposals_per_frame)
    if not targets or len(set(targets)) != len(targets) or any(
        q not in TARGET_PROPOSALS_PER_FRAME for q in targets
    ):
        raise ValueError(f"targets must be distinct members of {TARGET_PROPOSALS_PER_FRAME}")
    thresholds, grid_diagnostics = _calibration_quantiles(
        scores,
        max_candidates=_integer(max_threshold_candidates, "max_threshold_candidates", minimum=16),
        max_samples=_integer(max_quantile_samples, "max_quantile_samples", minimum=16),
    )
    prefix_limit = int(math.floor(max(targets) * len(scores))) + 1
    counts = np.zeros(len(thresholds), dtype=np.int64)
    exact = np.ones(len(thresholds), dtype=np.bool_)
    for frame in scores:
        peaks = strict_separated_nms(
            frame, distance_px=NMS_DISTANCE_PX,
            threshold=float(thresholds[0]), limit=prefix_limit,
        )
        peak_scores = np.asarray([peak[0] for peak in peaks], dtype=np.float64)
        for index, threshold in enumerate(thresholds):
            counts[index] += int(np.count_nonzero(peak_scores > threshold))
            if len(peaks) == prefix_limit and threshold < peak_scores[-1]:
                exact[index] = False
    evaluated = tuple({
        "threshold_z": float(tau),
        "calibration_proposal_count": int(count),
        "calibration_proposals_per_frame": float(count / len(scores)),
        "count_is_exact": bool(is_exact),
        "count_is_lower_bound": not bool(is_exact),
    } for tau, count, is_exact in zip(thresholds, counts, exact))
    operating = []
    for target in targets:
        feasible = [row for row in evaluated
                    if row["count_is_exact"]
                    and row["calibration_proposal_count"] <= target * len(scores)]
        if not feasible:
            raise RuntimeError("no exact feasible calibration threshold")
        selected = min(feasible, key=lambda row: (
            -row["calibration_proposal_count"], row["threshold_z"]
        ))
        operating.append({
            **selected,
            "target_proposals_per_frame": target,
            "calibration_burden_unit": CALIBRATION_BURDEN_UNIT,
            "calibration_score_frame_count": len(scores),
            "calibration_source_frames_ui": list(frames),
            "threshold_frozen_from_calibration_only": True,
        })
    return FramewiseCalibration(
        tuple(operating), tuple(float(t) for t in thresholds), evaluated,
        {**grid_diagnostics,
         "calibration_burden_unit": CALIBRATION_BURDEN_UNIT,
         "calibration_source_frames_ui": list(frames),
         "calibration_score_frame_count": len(scores),
         "calibration_assumed_event_free": False,
         "nms_semantics": NMS_SEMANTICS,
         "exact_nms_prefix_limit_per_frame": prefix_limit,
         "selected_operating_point_counts_exact": True,
         "application_proposal_rate_controlled": False,
         "annotation_fields_used": False},
    )


def extract_frame_candidates(
    score: Any,
    *,
    source_frame_ui: int,
    threshold_z: float,
    frame_interval_ms: float = 20.0,
    cell_id: str = "cell",
    target_proposals_per_frame: float | None = None,
) -> list[dict[str, Any]]:
    """Return the complete maintained NMS table for one scored source frame."""
    values = _score_frame(score)
    frame = _integer(source_frame_ui, "source_frame_ui", minimum=1)
    interval = float(frame_interval_ms)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("frame_interval_ms must be finite and positive")
    if target_proposals_per_frame is not None and float(target_proposals_per_frame) not in TARGET_PROPOSALS_PER_FRAME:
        raise ValueError("undeclared target proposals/frame")
    # The eligible-pixel bound cannot truncate a valid NMS candidate table.
    peaks = strict_separated_nms(
        values, distance_px=NMS_DISTANCE_PX, threshold=float(threshold_z),
        limit=(values.shape[0] - 12) * (values.shape[1] - 12),
    )
    target_token = "unspecified" if target_proposals_per_frame is None else format(float(target_proposals_per_frame), "g")
    return [{
        "proposal_id": f"{cell_id}__q{target_token}__ui{frame:06d}__r{rank:05d}",
        "cell_id": str(cell_id),
        "target_proposals_per_frame": target_proposals_per_frame,
        "calibration_burden_unit": CALIBRATION_BURDEN_UNIT,
        "threshold_z": float(threshold_z),
        "source_frame_ui": frame,
        "source_time_s": (frame - 1) * interval / 1000.0,
        "source_time_basis": "recording_relative_declared_frame_interval",
        "frame_interval_ms": interval,
        "candidate_rank_within_frame": rank,
        "score": value, "x_px": x, "y_px": y,
        "biological_status": "unknown_unreviewed_proposal",
        "temporal_linking_applied": False,
    } for rank, (value, x, y) in enumerate(peaks, start=1)]


def _proposal_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-float(row["score"]), int(row["y_px"]), int(row["x_px"]),
            int(row["source_frame_ui"]), str(row["proposal_id"]))


def evaluate_occurrence_windows(
    candidate_rows: Iterable[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    *,
    burst_intervals_ui: Mapping[Any, Sequence[int]],
) -> dict[str, Any]:
    """Evaluate frozen frame proposals against broad expert burst occurrences.

    Within each inclusive burst window, proposals are sorted by (-score,y,x,
    source_frame_ui,proposal_id). Each proposal joins the first already accepted
    representative within 6px, otherwise it becomes a new representative. This
    greedy spatial grouping is nontransitive and never averages coordinates.
    Score-ranked representatives are matched to the nearest unassigned expert
    within 6px, with observation-id tie breaking. This measures occurrence-window
    site recall, not frame recall, onset latency, or a count of biological events.
    """
    windows: dict[int, tuple[int, int]] = {}
    for key, bounds in burst_intervals_ui.items():
        burst = _integer(key, "burst_id", minimum=1)
        if burst in windows or len(bounds) != 2:
            raise ValueError("burst windows require unique ids and two inclusive bounds")
        start, stop = (_integer(x, "burst frame bound", minimum=1) for x in bounds)
        if stop < start:
            raise ValueError("burst stop must not precede its start")
        windows[burst] = (start, stop)
    if not windows:
        raise ValueError("at least one burst window is required")
    ordered_windows = sorted(windows.items(), key=lambda item: item[1])
    if any(right[1][0] <= left[1][1] for left, right in zip(ordered_windows, ordered_windows[1:])):
        raise ValueError("burst windows must not overlap")
    labels_by_burst: dict[int, list[dict[str, Any]]] = {b: [] for b in windows}
    observation_ids: set[str] = set()
    for positive in positives:
        row = dict(positive)
        burst = _integer(row["burst_id"], "positive burst_id", minimum=1)
        observation = str(row["observation_id"])
        if burst not in windows or not observation or observation in observation_ids:
            raise ValueError("positive requires a declared burst and unique observation_id")
        for coordinate in ("x_px", "y_px"):
            value = float(row[coordinate])
            if not math.isfinite(value) or value < 0:
                raise ValueError("positive coordinates must be finite and nonnegative")
            row[coordinate] = value
        row.update(burst_id=burst, observation_id=observation)
        observation_ids.add(observation)
        labels_by_burst[burst].append(row)
    groups: dict[int, list[dict[str, Any]]] = {b: [] for b in windows}
    proposal_ids: set[str] = set()
    dimensions: dict[str, set[Any]] = {key: set() for key in (
        "cell_id", "target_proposals_per_frame", "variant_id", "representation", "context_id"
    )}
    total_rows = 0
    for candidate in candidate_rows:
        row = dict(candidate)
        identifier = str(row["proposal_id"])
        if not identifier or identifier in proposal_ids:
            raise ValueError("proposal_id must be unique within the frozen stream")
        proposal_ids.add(identifier)
        row["proposal_id"] = identifier
        for name in ("source_frame_ui", "x_px", "y_px"):
            row[name] = _integer(row[name], name, minimum=1 if name == "source_frame_ui" else 0)
        row["score"] = float(row["score"])
        if not math.isfinite(row["score"]):
            raise ValueError("proposal score must be finite")
        for key, seen in dimensions.items():
            if key in row:
                seen.add(row[key])
                if len(seen) > 1:
                    raise ValueError(f"evaluate one frozen operating point at a time: mixed {key}")
        total_rows += 1
        for burst, (start, stop) in ordered_windows:
            if start <= row["source_frame_ui"] <= stop:
                groups[burst].append(row)
                break
    occurrence_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    burst_summaries: list[dict[str, Any]] = []
    for burst in sorted(windows):
        representatives: list[dict[str, Any]] = []
        memberships: list[list[dict[str, Any]]] = []
        for row in sorted(groups[burst], key=_proposal_key):
            site_index = next((index for index, representative in enumerate(representatives)
                if (row["x_px"] - representative["x_px"]) ** 2
                + (row["y_px"] - representative["y_px"]) ** 2 <= MATCH_RADIUS_PX ** 2), None)
            if site_index is None:
                site_index = len(representatives)
                representatives.append(row)
                memberships.append([])
            memberships[site_index].append(row)
        burst_sites = []
        for rank, (representative, members) in enumerate(zip(representatives, memberships), start=1):
            site_id = f"burst_{burst}__site_{rank:05d}"
            site = {
                **representative, "burst_id": burst, "site_id": site_id,
                "site_rank": rank,
                "representative_proposal_id": representative["proposal_id"],
                "member_proposal_count": len(members),
                "member_source_frame_count": len({row["source_frame_ui"] for row in members}),
                "first_member_source_frame_ui": min(row["source_frame_ui"] for row in members),
                "last_member_source_frame_ui": max(row["source_frame_ui"] for row in members),
                "site_interpretation": "spatial_representative_not_unique_event",
            }
            burst_sites.append(site)
            membership_rows.extend({
                "burst_id": burst, "site_id": site_id,
                "proposal_id": row["proposal_id"],
                "source_frame_ui": row["source_frame_ui"],
                "distance_to_representative_px": math.hypot(row["x_px"] - site["x_px"], row["y_px"] - site["y_px"]),
                "is_representative": row["proposal_id"] == site["representative_proposal_id"],
            } for row in members)
        labels = sorted(labels_by_burst[burst], key=lambda row: row["observation_id"])
        peaks = [(row["score"], row["x_px"], row["y_px"]) for row in burst_sites]
        matches, matched_peak_indices = match_peaks_one_to_one(peaks, labels, MATCH_RADIUS_PX)
        by_label = {
            match[0]: (burst_sites[peak_index], match[4])
            for match, peak_index in zip(matches, sorted(matched_peak_indices))
        }
        for label_index, positive in enumerate(labels):
            assigned = by_label.get(label_index)
            nearest = min(burst_sites, key=lambda row: (
                math.hypot(row["x_px"] - positive["x_px"], row["y_px"] - positive["y_px"]),
                row["site_rank"], row["site_id"]), default=None)
            occurrence_rows.append({
                **positive,
                "window_start_frame_ui": windows[burst][0],
                "window_stop_frame_ui": windows[burst][1],
                "matched": assigned is not None,
                "matched_site_id": None if assigned is None else assigned[0]["site_id"],
                "matched_proposal_id": None if assigned is None else assigned[0]["representative_proposal_id"],
                "matched_source_frame_ui": None if assigned is None else assigned[0]["source_frame_ui"],
                "match_distance_px": None if assigned is None else assigned[1],
                "nearest_site_id": None if nearest is None else nearest["site_id"],
                "nearest_site_distance_px": None if nearest is None else math.hypot(nearest["x_px"] - positive["x_px"], nearest["y_px"] - positive["y_px"]),
                "recall_unit": "known_positive_occurrence_within_inclusive_burst_window",
                "onset_latency_identified": False,
            })
        site_rows.extend(burst_sites)
        burst_summaries.append({
            "burst_id": burst, "window_start_frame_ui": windows[burst][0],
            "window_stop_frame_ui": windows[burst][1],
            "emitted_proposal_count_in_window": len(groups[burst]),
            "spatial_representative_count": len(burst_sites),
            "known_positive_count": len(labels), "matched_known_positive_count": len(matches),
            "known_positive_recall": len(matches) / len(labels) if labels else None,
        })
    matched = sum(bool(row["matched"]) for row in occurrence_rows)
    return {
        "occurrence_rows": occurrence_rows, "site_rows": site_rows,
        "membership_rows": membership_rows, "burst_summaries": burst_summaries,
        "summary": {
            "emitted_frame_proposal_count": total_rows,
            "emitted_proposal_count_in_declared_windows": sum(len(rows) for rows in groups.values()),
            "spatial_representative_count_in_declared_windows": len(site_rows),
            "known_positive_count": len(occurrence_rows),
            "matched_known_positive_count": matched,
            "known_positive_occurrence_window_recall": matched / len(occurrence_rows) if occurrence_rows else None,
            "site_consolidation": "score_y_x_frame_id_order_first_representative_within_6px_nontransitive",
            "matching": "score_ranked_representative_to_nearest_unassigned_expert_within_6px_observation_id_ties",
            "evaluation_scope": "descriptive_within_recording_frozen_frame_proposals",
            "precision_identified": False, "onset_latency_identified": False,
            "unique_event_count_identified": False,
            "unmatched_candidates": "unknown_not_negative",
        },
    }


__all__ = [
    "CALIBRATION_BURDEN_UNIT", "TARGET_PROPOSALS_PER_FRAME", "NMS_SEMANTICS",
    "FramewiseCalibration", "calibrate_tau", "extract_frame_candidates",
    "evaluate_occurrence_windows",
]
