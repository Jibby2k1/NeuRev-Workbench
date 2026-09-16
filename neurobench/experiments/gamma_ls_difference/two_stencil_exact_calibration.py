"""Calibration-only exact NMS order statistics for the two-stencil amendment.

This is a new method; the frozen pixel-quantile-grid implementation remains in
``two_stencil_evaluation``. Only the cutoff search changes. Score frames, frame
denominators, strict thresholding, and maintained spatial NMS are unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from .evaluation import strict_separated_nms
from .two_stencil_evaluation import (
    CALIBRATION_BURDEN_UNIT,
    FramewiseCalibration,
    NMS_DISTANCE_PX,
    NMS_SEMANTICS,
    TARGET_PROPOSALS_PER_FRAME,
    _integer,
    _score_frame,
)


EXACT_CALIBRATION_METHOD = "exact_maintained_nms_score_order_statistics"
PREVIOUS_CALIBRATION_METHOD = (
    "deterministic_even_ordinal_sample_nearest_quantiles_plus_exact_extrema"
)


def _finite_below(value: float) -> float:
    """Return a finite float64 cutoff strictly below an observed score."""
    with np.errstate(over="ignore"):
        below = float(np.nextafter(np.float64(value), -np.inf))
    if not math.isfinite(below):
        raise ValueError("score minimum has no finite float64 cutoff below it")
    return below


def calibrate_tau_exact_nms(
    calibration_scores: Any,
    *,
    source_frames_ui: Sequence[int],
    target_proposals_per_frame: Sequence[float] = TARGET_PROPOSALS_PER_FRAME,
) -> FramewiseCalibration:
    """Select the densest attainable strict cutoff within each integer budget.

    For N supplied calibration frames and target q, the budget is floor(q*N).
    Retain L=floor(max(q)*N)+1 maintained NMS peaks from each frame at a common
    finite cutoff below every calibration score. Sort their union descending.
    A budget B uses score[B] as its strict cutoff. Equal-score ties remain
    indivisible and can underfill the budget; this is not a top-k head.

    Exactness with bounded prefixes follows from NMS threshold-prefix
    invariance: the local-max prefilter does not depend on the cutoff, and
    lower-ranked peaks cannot suppress higher-ranked retained peaks. Every
    saturated frame contributes L>B scores at least as large as its final
    retained score s. Hence the global score[B] is >=s, and every omitted tail
    score is <=s<=cutoff and excluded. The selected counts are therefore exact.
    Lowering score[B] admits at least B+1 peaks, proving maximal attainable
    burden. If <=B retained peaks exist, no frame can be saturated because
    L>B: all peaks are known and a cutoff below their minimum accepts them all.

    An all-zero scene explicitly returns a zero-proposal cutoff at zero. A
    scene with no eligible interior NMS peak returns its maximum score. These
    guards and all threshold decisions depend only on the supplied calibration
    scores. No annotations or application outcomes are accepted by this API.
    Initialization is not assumed event-free; application burden is uncontrolled.
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

    minimum, maximum = math.inf, -math.inf
    for frame in scores:
        values = _score_frame(frame)
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    extraction_threshold = _finite_below(minimum)
    prefix_limit = math.floor(max(targets) * len(scores)) + 1
    retained_scores: list[float] = []
    prefix_rows: list[dict[str, Any]] = []
    for source_frame, frame in zip(frames, scores):
        peaks = strict_separated_nms(
            frame, distance_px=NMS_DISTANCE_PX,
            threshold=extraction_threshold, limit=prefix_limit,
        )
        retained_scores.extend(float(peak[0]) for peak in peaks)
        prefix_rows.append({
            "source_frame_ui": source_frame,
            "retained_nms_peak_count": len(peaks),
            "prefix_saturated": len(peaks) == prefix_limit,
            "last_retained_score": float(peaks[-1][0]) if peaks else None,
        })
    ranked = np.sort(np.asarray(retained_scores, dtype=np.float64))[::-1]
    ascending = ranked[::-1]
    saturated = [row for row in prefix_rows if row["prefix_saturated"]]
    zero_scene = minimum == 0.0 and maximum == 0.0
    no_peaks = len(ranked) == 0
    operating = []
    for target in targets:
        budget = math.floor(target * len(scores))
        boundary_rank = None
        if zero_scene:
            threshold, reason = maximum, "all_zero_scene_guard"
        elif no_peaks:
            threshold, reason = maximum, "no_eligible_nms_peaks"
        elif len(ranked) > budget:
            threshold = float(ranked[budget])
            boundary_rank = budget
            reason = "strict_global_nms_order_statistic"
        else:
            if saturated:
                raise RuntimeError("a saturated prefix cannot have <= budget retained peaks")
            threshold = _finite_below(float(ranked[-1]))
            reason = "all_nms_peaks_fit_budget"
        if any(threshold < float(row["last_retained_score"]) for row in saturated):
            raise RuntimeError("selected cutoff would include an unobserved NMS prefix tail")
        count = len(ranked) - int(np.searchsorted(ascending, threshold, side="right"))
        if count > budget:
            raise RuntimeError("exact NMS calibration exceeded its integer budget")
        operating.append({
            "threshold_z": threshold,
            "calibration_proposal_count": count,
            "calibration_proposals_per_frame": count / len(scores),
            "count_is_exact": True,
            "count_is_lower_bound": False,
            "target_proposals_per_frame": target,
            "calibration_burden_unit": CALIBRATION_BURDEN_UNIT,
            "calibration_score_frame_count": len(scores),
            "calibration_source_frames_ui": list(frames),
            "threshold_frozen_from_calibration_only": True,
            "calibration_integer_proposal_budget": budget,
            "calibration_budget_underfill_count": budget - count,
            "calibration_method": EXACT_CALIBRATION_METHOD,
            "selection_reason": reason,
            "boundary_zero_based_rank": boundary_rank,
            "strict_boundary_tie_underfills_budget": (
                boundary_rank is not None and count < budget
            ),
        })

    # These are the distinct selected exact boundaries, plus a zero-proposal
    # upper endpoint. There is no sampled grid and no need to enumerate or
    # rescore every possible scalar cutoff to certify the selected boundaries.
    thresholds = tuple(sorted({maximum, *(row["threshold_z"] for row in operating)}))
    evaluated = tuple({
        "threshold_z": threshold,
        "calibration_proposal_count": len(ranked) - int(
            np.searchsorted(ascending, threshold, side="right")
        ),
        "calibration_proposals_per_frame": (
            len(ranked) - int(np.searchsorted(ascending, threshold, side="right"))
        ) / len(scores),
        "count_is_exact": True,
        "count_is_lower_bound": False,
    } for threshold in thresholds)
    return FramewiseCalibration(
        tuple(operating), thresholds, evaluated,
        {
            "calibration_method": EXACT_CALIBRATION_METHOD,
            "threshold_grid_method": "none_exact_nms_order_statistics",
            "previous_calibration_method": PREVIOUS_CALIBRATION_METHOD,
            "amended_component": "scalar_cutoff_search_only",
            "threshold_candidates_semantics": "selected_exact_boundaries_plus_global_maximum",
            "threshold_candidate_count": len(thresholds),
            "threshold_candidates_from_calibration_only": True,
            "quantile_sampling_used": False,
            "calibration_burden_unit": CALIBRATION_BURDEN_UNIT,
            "calibration_source_frames_ui": list(frames),
            "calibration_score_frame_count": len(scores),
            "target_budget_rule": "floor(target_proposals_per_frame * calibration_score_frame_count)",
            "calibration_assumed_event_free": False,
            "nms_semantics": NMS_SEMANTICS,
            "exact_nms_prefix_limit_per_frame": prefix_limit,
            "calibration_nms_prefix_rows": prefix_rows,
            "retained_nms_peak_count": len(ranked),
            "saturated_prefix_frame_count": len(saturated),
            "complete_nms_inventory_known": not saturated,
            "nms_extraction_threshold": extraction_threshold,
            "global_calibration_score_minimum": minimum,
            "global_calibration_score_maximum": maximum,
            "selected_operating_point_counts_exact": True,
            "selected_cutoffs_exclude_all_unobserved_prefix_tails": True,
            "densest_attainable_strict_cutoff": not zero_scene,
            "all_zero_scene_guard_applied": zero_scene,
            "ties_split_to_fill_budget": False,
            "application_proposal_rate_controlled": False,
            "annotation_fields_used": False,
            "application_scores_used": False,
            "input_or_operator_selection_performed": False,
        },
    )
