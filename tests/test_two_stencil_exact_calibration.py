"""Exact maintained-NMS calibration, independent of annotation outcomes."""
from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.evaluation import strict_separated_nms
from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import (
    CALIBRATION_BURDEN_UNIT,
    FramewiseCalibration,
    TARGET_PROPOSALS_PER_FRAME,
    calibrate_tau,
)
from neurobench.experiments.gamma_ls_difference.two_stencil_exact_calibration import (
    EXACT_CALIBRATION_METHOD,
    calibrate_tau_exact_nms,
)


def _full_count(scores, threshold):
    return sum(len(strict_separated_nms(frame, threshold=threshold, limit=frame.size))
               for frame in scores)


def _assert_exact_and_densest(scores, calibration):
    """Reference every attainable NMS score boundary without a prefix cap."""
    below_minimum = float(np.nextafter(np.float64(scores.min()), -np.inf))
    all_peak_scores = [peak[0] for frame in scores for peak in strict_separated_nms(
        frame, threshold=below_minimum, limit=frame.size
    )]
    cutoffs = set(all_peak_scores) | {float(scores.max())}
    if all_peak_scores:
        cutoffs.add(float(np.nextafter(np.float64(min(all_peak_scores)), -np.inf)))
    attained = {cutoff: _full_count(scores, cutoff) for cutoff in cutoffs}
    for operating in calibration.operating_points:
        budget = math.floor(operating["target_proposals_per_frame"] * len(scores))
        count = _full_count(scores, operating["threshold_z"])
        assert count == operating["calibration_proposal_count"]
        assert count == max(count for count in attained.values() if count <= budget)
        assert operating["calibration_integer_proposal_budget"] == budget
        assert operating["calibration_budget_underfill_count"] == budget - count
        assert operating["count_is_exact"] and not operating["count_is_lower_bound"]
    for evaluated in calibration.evaluated_thresholds:
        assert evaluated["calibration_proposal_count"] == _full_count(
            scores, evaluated["threshold_z"]
        )
    json.dumps({"operating": calibration.operating_points,
                "evaluated": calibration.evaluated_thresholds,
                "diagnostics": calibration.diagnostics}, allow_nan=False)


def test_rare_peaks_missed_by_pixel_quantile_grid_are_calibrated_exactly():
    # All 16 deterministic pixel samples miss these rare peaks. The old grid
    # has only zero, the exact maximum, and a below-zero endpoint: it cannot
    # express a cutoff that retains four of the eight positive NMS peaks.
    scores = np.zeros((4, 45, 45), dtype=np.float32)
    for index in range(4):
        scores[index, 12, 12] = 2 * index + 1
        scores[index, 30, 30] = 2 * index + 2
    frames = [2, 3, 4, 5]
    previous = calibrate_tau(scores, source_frames_ui=frames,
                             max_threshold_candidates=16, max_quantile_samples=16)
    exact = calibrate_tau_exact_nms(scores, source_frames_ui=frames)
    old_q1 = next(row for row in previous.operating_points
                  if row["target_proposals_per_frame"] == 1)
    new_q1 = next(row for row in exact.operating_points
                  if row["target_proposals_per_frame"] == 1)
    assert previous.diagnostics["quantile_sampling_used"]
    assert old_q1["calibration_proposal_count"] == 0
    assert new_q1["threshold_z"] == 4
    assert new_q1["calibration_proposal_count"] == 4
    _assert_exact_and_densest(scores, exact)


def test_tied_boundary_is_indivisible_and_can_underfill_integer_budget():
    scores = np.zeros((1, 51, 51), dtype=np.float32)
    for (y, x), value in zip([(10, 10), (10, 30), (30, 10), (30, 30)], [9, 5, 5, 3]):
        scores[0, y, x] = value
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=[2])
    q2 = next(row for row in calibration.operating_points
              if row["target_proposals_per_frame"] == 2)
    assert q2["threshold_z"] == 5
    assert q2["calibration_proposal_count"] == 1
    assert q2["calibration_integer_proposal_budget"] == 2
    assert q2["strict_boundary_tie_underfills_budget"]
    assert _full_count(scores, float(np.nextafter(np.float64(5), -np.inf))) == 3
    _assert_exact_and_densest(scores, calibration)


def test_saturated_prefixes_still_certify_every_selected_exact_count():
    scores = np.zeros((2, 81, 81), dtype=np.float32)
    coordinates = [(y, x) for y in range(10, 72, 12) for x in range(10, 72, 12)]
    for frame in range(2):
        for index, (y, x) in enumerate(coordinates):
            scores[frame, y, x] = 100 * (2 - frame) - index
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3])
    assert calibration.diagnostics["exact_nms_prefix_limit_per_frame"] == 11
    assert calibration.diagnostics["saturated_prefix_frame_count"] == 2
    assert not calibration.diagnostics["complete_nms_inventory_known"]
    assert calibration.diagnostics["retained_nms_peak_count"] == 22
    assert all(row["calibration_proposal_count"] == math.floor(
        row["target_proposals_per_frame"] * 2
    ) for row in calibration.operating_points)
    for prefix in calibration.diagnostics["calibration_nms_prefix_rows"]:
        assert all(row["threshold_z"] >= prefix["last_retained_score"]
                   for row in calibration.operating_points)
    _assert_exact_and_densest(scores, calibration)


def test_exhausted_unsaturated_inventory_accepts_all_peaks_when_they_fit():
    scores = np.zeros((2, 13, 13), dtype=np.float32)
    scores[:, 6, 6] = [1, 2]
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3])
    q1 = next(row for row in calibration.operating_points
              if row["target_proposals_per_frame"] == 1)
    assert q1["selection_reason"] == "all_nms_peaks_fit_budget"
    assert q1["threshold_z"] == float(np.nextafter(np.float64(1), -np.inf))
    assert q1["calibration_proposal_count"] == 2
    assert calibration.diagnostics["complete_nms_inventory_known"]
    _assert_exact_and_densest(scores, calibration)


@pytest.mark.parametrize("shape", [(2, 13, 13), (2, 51, 51)])
def test_all_zero_scene_guard_yields_zero_at_every_budget(shape):
    scores = np.zeros(shape, dtype=np.float32)
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3])
    assert calibration.diagnostics["all_zero_scene_guard_applied"]
    assert not calibration.diagnostics["densest_attainable_strict_cutoff"]
    for row in calibration.operating_points:
        assert row["threshold_z"] == 0
        assert row["calibration_proposal_count"] == _full_count(scores, 0) == 0
        assert row["selection_reason"] == "all_zero_scene_guard"


def test_no_eligible_interior_maximum_uses_global_maximum():
    y, x = np.indices((21, 21))
    scores = (y + x)[None].astype(np.float32)
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=[2])
    assert calibration.diagnostics["retained_nms_peak_count"] == 0
    assert all(row["threshold_z"] == 40 and row["calibration_proposal_count"] == 0
               and row["selection_reason"] == "no_eligible_nms_peaks"
               for row in calibration.operating_points)
    _assert_exact_and_densest(scores, calibration)


def test_99_frame_denominator_floor_budgets_and_old_result_api_are_preserved():
    scores = np.zeros((99, 13, 13), dtype=np.float32)
    scores[:, 6, 6] = np.arange(1, 100)
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=list(range(2, 101)))
    assert isinstance(calibration, FramewiseCalibration)
    assert calibration.diagnostics["exact_nms_prefix_limit_per_frame"] == 496
    assert [row["calibration_integer_proposal_budget"] for row in calibration.operating_points] == [24, 49, 99, 198, 495]
    assert [row["calibration_proposal_count"] for row in calibration.operating_points] == [24, 49, 99, 99, 99]
    assert all(row["calibration_score_frame_count"] == 99 and
               row["calibration_source_frames_ui"] == list(range(2, 101)) and
               row["calibration_burden_unit"] == CALIBRATION_BURDEN_UNIT and
               row["calibration_method"] == EXACT_CALIBRATION_METHOD
               for row in calibration.operating_points)
    assert tuple(row["target_proposals_per_frame"] for row in calibration.operating_points) == TARGET_PROPOSALS_PER_FRAME
    assert calibration.threshold_for(.25) == 75
    with pytest.raises(KeyError):
        calibration.threshold_for(3)


def test_inputs_are_calibration_only_deterministic_and_fail_closed():
    assert set(inspect.signature(calibrate_tau_exact_nms).parameters) == {
        "calibration_scores", "source_frames_ui", "target_proposals_per_frame"
    }
    scores = np.zeros((2, 21, 21), dtype=np.float32)
    scores[:, 10, 10] = [3, 4]
    calibration = calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3])
    assert calibration == calibrate_tau_exact_nms(scores.copy(), source_frames_ui=[2, 3])
    assert not calibration.diagnostics["annotation_fields_used"]
    assert not calibration.diagnostics["application_scores_used"]
    assert not calibration.diagnostics["input_or_operator_selection_performed"]
    assert not calibration.diagnostics["calibration_assumed_event_free"]
    assert not calibration.diagnostics["application_proposal_rate_controlled"]
    with pytest.raises(TypeError, match="expert"):
        calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3], expert_occurrences=[])
    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_tau_exact_nms(scores, source_frames_ui=[2, 2])
    with pytest.raises(ValueError, match="align"):
        calibrate_tau_exact_nms(scores, source_frames_ui=[2])
    with pytest.raises(ValueError, match="distinct"):
        calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3], target_proposals_per_frame=[1, 1])
    with pytest.raises(ValueError, match="distinct"):
        calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3], target_proposals_per_frame=[3])
    scores[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrate_tau_exact_nms(scores, source_frames_ui=[2, 3])
