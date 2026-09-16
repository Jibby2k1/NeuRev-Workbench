"""Scientific-contract checks for framewise calibration and window recall."""
from __future__ import annotations

import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.evaluation import strict_separated_nms
from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import (
    CALIBRATION_BURDEN_UNIT,
    TARGET_PROPOSALS_PER_FRAME,
    calibrate_tau,
    evaluate_occurrence_windows,
    extract_frame_candidates,
)


def proposal(identifier, frame, x, y, score):
    return {
        "proposal_id": identifier, "source_frame_ui": frame,
        "x_px": x, "y_px": y, "score": score,
    }


def positive(identifier, x, y, *, burst=1):
    return {
        "observation_id": identifier, "canonical_roi_id": f"roi_{identifier}",
        "burst_id": burst, "x_px": x, "y_px": y,
    }


def test_complete_frame_table_matches_maintained_nms_and_source_time():
    rng = np.random.default_rng(92026)
    score = rng.normal(size=(41, 45)).astype(np.float32)
    score[12:15, 12:15] = 9.0
    rows = extract_frame_candidates(
        score, source_frame_ui=101, threshold_z=1.0,
        cell_id="two_stencil_a", target_proposals_per_frame=1.0,
    )
    expected = strict_separated_nms(score, threshold=1.0, limit=score.size)
    assert [(row["score"], row["x_px"], row["y_px"]) for row in rows] == expected
    assert [row["candidate_rank_within_frame"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["source_frame_ui"] == 101 and row["source_time_s"] == 2.0 for row in rows)
    assert all(row["calibration_burden_unit"] == CALIBRATION_BURDEN_UNIT for row in rows)
    assert all(not row["temporal_linking_applied"] for row in rows)
    json.dumps(rows, allow_nan=False)


def test_square_prefilter_border_and_strict_threshold_are_preserved():
    score = np.zeros((37, 37), dtype=np.float32)
    score[10, 10] = 5
    score[16, 16] = 4  # farther than 6 Euclidean, but inside the square prefilter
    score[5, 25] = 20  # excluded border site
    score[26, 26] = 3  # exactly the strict threshold
    rows = extract_frame_candidates(score, source_frame_ui=2, threshold_z=3.0)
    assert [(row["x_px"], row["y_px"]) for row in rows] == [(10, 10)]


def test_calibration_selected_counts_equal_full_nms_on_every_frame():
    rng = np.random.default_rng(19)
    scores = rng.normal(size=(5, 47, 51)).astype(np.float32)
    calibration = calibrate_tau(scores, source_frames_ui=[2, 3, 4, 5, 6], max_quantile_samples=100)
    assert tuple(row["target_proposals_per_frame"] for row in calibration.operating_points) == TARGET_PROPOSALS_PER_FRAME
    for operating in calibration.operating_points:
        tau = operating["threshold_z"]
        total = sum(len(strict_separated_nms(frame, threshold=tau, limit=frame.size)) for frame in scores)
        assert total == operating["calibration_proposal_count"]
        assert total <= operating["target_proposals_per_frame"] * len(scores)
        assert operating["count_is_exact"]
        assert not operating["count_is_lower_bound"]
        feasible = [row for row in calibration.evaluated_thresholds if row["count_is_exact"]
                    and row["calibration_proposal_count"] <= operating["target_proposals_per_frame"] * len(scores)]
        expected = min(feasible, key=lambda row: (-row["calibration_proposal_count"], row["threshold_z"]))
        assert tau == expected["threshold_z"]
    assert calibration.diagnostics["quantile_sampling_used"]
    assert not calibration.diagnostics["calibration_assumed_event_free"]
    assert not calibration.diagnostics["annotation_fields_used"]
    again = calibrate_tau(scores.copy(), source_frames_ui=[2, 3, 4, 5, 6], max_quantile_samples=100)
    assert calibration == again


def test_saturated_prefix_cannot_be_selected_as_an_exact_operating_point():
    scores = np.zeros((2, 51, 51), dtype=np.float32)
    calibration = calibrate_tau(scores, source_frames_ui=[2, 3])
    assert any(row["count_is_lower_bound"] for row in calibration.evaluated_thresholds)
    assert all(row["count_is_exact"] and row["calibration_proposal_count"] == 0
               for row in calibration.operating_points)
    assert calibration.threshold_for(1.0) == 0.0
    with pytest.raises(KeyError):
        calibration.threshold_for(3)


def test_calibration_source_alignment_and_nonfinite_scores_fail_closed():
    scores = np.ones((2, 21, 21), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_tau(scores, source_frames_ui=[2, 2])
    with pytest.raises(ValueError, match="align"):
        calibrate_tau(scores, source_frames_ui=[2])
    scores[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrate_tau(scores, source_frames_ui=[2, 3])
    with pytest.raises(ValueError, match="undeclared"):
        extract_frame_candidates(np.zeros((21, 21)), source_frame_ui=2,
                                 threshold_z=0, target_proposals_per_frame=3)


def test_window_sites_are_nontransitive_and_represent_actual_rows():
    rows = [proposal("c", 11, 20, 10, 3), proposal("b", 12, 15, 10, 4),
            proposal("a", 10, 10, 10, 5), proposal("outside", 9, 30, 30, 20)]
    result = evaluate_occurrence_windows(
        rows, [positive("a", 10, 10), positive("c", 20, 10)],
        burst_intervals_ui={1: (10, 12)},
    )
    assert [row["representative_proposal_id"] for row in result["site_rows"]] == ["a", "c"]
    assert [row["member_proposal_count"] for row in result["site_rows"]] == [2, 1]
    membership = {row["proposal_id"]: row["site_id"] for row in result["membership_rows"]}
    assert membership["a"] == membership["b"] != membership["c"]
    assert result["summary"]["emitted_frame_proposal_count"] == 4
    assert result["summary"]["emitted_proposal_count_in_declared_windows"] == 3
    assert result["summary"]["matched_known_positive_count"] == 2
    assert not result["summary"]["unique_event_count_identified"]
    assert not result["summary"]["onset_latency_identified"]
    json.dumps(result, allow_nan=False)
    assert result == evaluate_occurrence_windows(list(reversed(rows)), list(reversed([
        positive("a", 10, 10), positive("c", 20, 10)])), burst_intervals_ui={1: (10, 12)})


def test_repeated_frame_proposals_cannot_recover_two_nearby_experts():
    rows = [proposal("p2", 12, 10, 10, 4), proposal("p1", 10, 10, 10, 4)]
    result = evaluate_occurrence_windows(
        rows, [positive("z", 16, 10), positive("a", 4, 10)],
        burst_intervals_ui={1: (10, 12)},
    )
    assert result["summary"]["matched_known_positive_count"] == 1
    assert result["summary"]["known_positive_occurrence_window_recall"] == .5
    assert result["site_rows"][0]["representative_proposal_id"] == "p1"
    assert result["occurrence_rows"][0]["observation_id"] == "a"
    assert result["occurrence_rows"][0]["matched"]
    assert result["occurrence_rows"][0]["match_distance_px"] == 6
    assert not result["occurrence_rows"][1]["matched"]
    assert result["occurrence_rows"][1]["nearest_site_distance_px"] == 6


def test_first_accepted_representative_controls_membership_not_nearest():
    rows = [proposal("a", 10, 10, 10, 10), proposal("c", 10, 20, 10, 9),
            proposal("b", 11, 16, 10, 8)]
    result = evaluate_occurrence_windows(rows, [], burst_intervals_ui={1: (10, 11)})
    membership = {row["proposal_id"]: row["site_id"] for row in result["membership_rows"]}
    assert membership["b"] == membership["a"]  # distance6 to first, versus4 to second
    assert result["summary"]["known_positive_occurrence_window_recall"] is None


def test_window_evaluation_rejects_ambiguous_inputs_and_preserves_empty_windows():
    row = proposal("p", 10, 10, 10, 5)
    with pytest.raises(ValueError, match="unique"):
        evaluate_occurrence_windows([row, row], [], burst_intervals_ui={1: (10, 12)})
    with pytest.raises(ValueError, match="overlap"):
        evaluate_occurrence_windows([], [], burst_intervals_ui={1: (10, 12), 2: (12, 20)})
    with pytest.raises(ValueError, match="mixed"):
        evaluate_occurrence_windows([{**row, "cell_id": "a"},
            {**row, "proposal_id": "q", "cell_id": "b"}], [], burst_intervals_ui={1: (10, 12)})
    result = evaluate_occurrence_windows([], [positive("miss", 10, 10)],
                                         burst_intervals_ui={1: (10, 12), 2: (20, 30)})
    assert len(result["occurrence_rows"]) == 1
    assert result["occurrence_rows"][0]["matched"] is False
    assert result["occurrence_rows"][0]["nearest_site_id"] is None
    assert len(result["burst_summaries"]) == 2
    assert result["summary"]["known_positive_occurrence_window_recall"] == 0
