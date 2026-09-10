from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.evaluation import (
    CANDIDATE_BUDGETS_PER_BURST,
    QUIET_NMS_PEAK_BURDENS,
    calibrate_training_quiet_thresholds,
    duration_matched_quiet_windows,
    evaluate_sparse_positive_recall,
    extract_burst_candidates,
    fit_training_quiet_scale_floor,
    paired_representation_equivalence,
    strict_separated_nms,
    temporal_threshold_occupancy,
)


def test_scale_floor_uses_positive_training_quiet_values_only() -> None:
    local_std = np.asarray(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[1_000.0, 2_000.0]],
        ],
        dtype=np.float32,
    )
    quiet = np.asarray([True, True, False])
    fit = fit_training_quiet_scale_floor(local_std, quiet, percentile=50.0)
    assert fit.scale_floor == pytest.approx(2.5)
    assert fit.diagnostics["fit_scope"] == "training_quiet_frames_only"
    assert fit.diagnostics["training_quiet_frame_count"] == 2
    assert fit.diagnostics["positive_training_quiet_sample_count"] == 4
    assert fit.diagnostics["quiet_interval_used"] is True
    assert fit.diagnostics["sparse_positive_coordinates_used"] is False

    changed_event = local_std.copy()
    changed_event[2] = 1_000_000.0
    changed = fit_training_quiet_scale_floor(changed_event, quiet, percentile=50.0)
    assert changed.scale_floor == fit.scale_floor
    with pytest.raises(ValueError, match="no positive values"):
        fit_training_quiet_scale_floor(
            np.zeros((3, 1, 2), dtype=np.float32),
            quiet,
            percentile=10.0,
        )


def test_canonical_duration_matched_windows_are_quiet_and_deterministic() -> None:
    quiet = np.ones(100, dtype=bool)
    durations = {1: 24, 2: 24, 3: 28, 4: 47}
    windows = duration_matched_quiet_windows(quiet, durations)
    assert windows == {1: (0, 24), 2: (24, 48), 3: (48, 76), 4: (53, 100)}
    for key, (start, stop) in windows.items():
        assert stop - start == durations[key]
        assert np.all(quiet[start:stop])

    quiet[60] = False
    with pytest.raises(ValueError, match="not entirely training quiet"):
        duration_matched_quiet_windows(
            quiet,
            durations,
            starts={1: 0, 2: 24, 3: 48, 4: 53},
        )


def test_occupancy_aggregation_and_nms_are_strict_and_spatially_separated() -> None:
    scores = np.zeros((3, 25, 25), dtype=np.float32)
    scores[0, 8, 8] = 2.0
    scores[:2, 8, 18] = 2.0
    occupancy = temporal_threshold_occupancy(scores, threshold_z=1.0)
    assert occupancy[8, 8] == pytest.approx(1.0 / 3.0)
    assert occupancy[8, 18] == pytest.approx(2.0 / 3.0)
    assert not np.array_equal(occupancy, scores.max(axis=0))

    extracted = extract_burst_candidates(
        scores,
        {1: (0, 3)},
        threshold_z=1.0,
    )
    assert [(x, y) for _, x, y in extracted.peaks[1]] == [(18, 8), (8, 8)]
    assert extracted.diagnostics["temporal_max_pooling_used"] is False
    assert extracted.diagnostics["frame_decision"].startswith("score_strictly")

    plateau = np.zeros((25, 25), dtype=np.float32)
    plateau[12, 7] = plateau[12, 11] = plateau[12, 17] = 1.0
    peaks = strict_separated_nms(plateau, threshold=0.0)
    assert [(x, y) for _, x, y in peaks] == [(7, 12), (17, 12)]
    assert strict_separated_nms(plateau, threshold=1.0) == []


def test_nms_four_and_eight_are_descriptive_sensitivities() -> None:
    scores = np.zeros((2, 31, 31), dtype=np.float32)
    scores[:, 10, 10] = 3.0
    scores[:, 10, 16] = 2.0
    four = extract_burst_candidates(
        scores, {1: (0, 2)}, threshold_z=1.0, nms_distance_px=4
    )
    eight = extract_burst_candidates(
        scores, {1: (0, 2)}, threshold_z=1.0, nms_distance_px=8
    )
    assert len(four.peaks[1]) == 2
    assert len(eight.peaks[1]) == 1
    assert four.diagnostics["nms_role"] == "descriptive_sensitivity"
    assert eight.diagnostics["nms_role"] == "descriptive_sensitivity"
    with pytest.raises(ValueError, match="must be one of"):
        extract_burst_candidates(
            scores, {1: (0, 2)}, threshold_z=1.0, nms_distance_px=5
        )


def _quiet_calibration_scores(event_value: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((10, 27, 27), dtype=np.float32)
    coordinates_and_values = (
        ((7, 7), 10.0),
        ((7, 17), 8.0),
        ((17, 7), 6.0),
        ((17, 17), 4.0),
    )
    for frame in range(8):
        for (y, x), value in coordinates_and_values:
            scores[frame, y, x] = value
    scores[8:, 13, 13] = event_value
    quiet = np.asarray([True] * 8 + [False] * 2)
    return scores, quiet


def test_quiet_thresholds_hit_empirical_burdens_without_event_or_label_leakage() -> None:
    scores, quiet = _quiet_calibration_scores(1_000.0)
    durations = {1: 2, 2: 2, 3: 2, 4: 2}
    calibration = calibrate_training_quiet_thresholds(scores, quiet, durations)
    changed_scores, _ = _quiet_calibration_scores(-1_000.0)
    changed = calibrate_training_quiet_thresholds(changed_scores, quiet, durations)

    assert calibration.pseudo_burst_windows == {
        1: (0, 2),
        2: (2, 4),
        3: (4, 6),
        4: (6, 8),
    }
    assert calibration.operating_points == changed.operating_points
    assert calibration.threshold_for(1.0) == pytest.approx(8.0)
    assert calibration.threshold_for(2.0) == pytest.approx(6.0)
    assert tuple(
        row["target_nms_peaks_per_pseudo_burst"]
        for row in calibration.operating_points
    ) == QUIET_NMS_PEAK_BURDENS
    for row in calibration.operating_points:
        assert (
            row["achieved_nms_peaks_per_pseudo_burst"]
            <= row["target_nms_peaks_per_pseudo_burst"]
        )
        assert row["probability_of_false_alarm_claimed"] is False
    assert calibration.diagnostics["calibration_scope"] == "training_quiet_frames_only"
    assert calibration.diagnostics["threshold_candidates_derived_from_training_quiet_only"]
    assert calibration.diagnostics["probability_of_false_alarm_claimed"] is False
    assert calibration.diagnostics["quiet_interval_used"] is True
    assert calibration.diagnostics["sparse_positive_coordinates_used"] is False


def test_sparse_positive_recall_is_one_to_one_at_frozen_budgets() -> None:
    burst_peaks = {
        1: [(5.0, 5, 10), (4.0, 15, 10), (3.0, 20, 20)],
        2: [(6.0, 30, 30)],
    }
    positives = [
        {"observation_id": "b1_a", "burst_id": 1, "x_px": 10.0, "y_px": 10.0},
        {"observation_id": "b1_b", "burst_id": 1, "x_px": 20.0, "y_px": 20.0},
        {"observation_id": "b2_a", "burst_id": 2, "x_px": 30.0, "y_px": 30.0},
    ]
    result = evaluate_sparse_positive_recall(burst_peaks, positives)
    assert result["candidate_budgets_per_burst"] == list(CANDIDATE_BUDGETS_PER_BURST)
    assert result["one_to_one_matching"] is True
    assert result["match_radius_px"] == 6.0
    assert result["pooled_known_positive_recall_by_budget"]["58"] == 1.0
    assert result["matched_known_positives_by_budget"]["20"] == 3
    burst_one = next(
        row
        for row in result["rows"]
        if row["burst_id"] == 1 and row["candidate_budget"] == 58
    )
    assert burst_one["matched_known_positive_count"] == 2
    assert burst_one["unmatched_candidate_count"] == 1
    assert burst_one["unmatched_candidates"] == "unknown_not_negative"
    assert result["precision_identified"] is False

    with pytest.raises(ValueError, match="budgets must be exactly"):
        evaluate_sparse_positive_recall(burst_peaks, positives, budgets=(20, 58, 100))


def test_paired_equivalence_is_scale_aware_and_uses_absolute_top_one_percent() -> None:
    reference = np.linspace(-2.0, 2.0, 200, dtype=np.float64).reshape(20, 10)
    candidate = -3.0 * reference
    result = paired_representation_equivalence(reference, candidate)
    assert result["paired_value_count"] == 200
    assert result["correlation"] == pytest.approx(-1.0)
    assert result["pearson_correlation"] == pytest.approx(-1.0)
    assert result["spearman_correlation"] == pytest.approx(-1.0)
    assert result["least_squares_reference_to_candidate_scale"] == pytest.approx(-3.0)
    assert result["nrms"] == pytest.approx(0.0, abs=1e-15)
    assert result["top_count"] == 2
    assert result["top_1pct_jaccard"] == 1.0
    assert result["sparse_positive_coordinates_used"] is False
    assert result["sparse_positive_identities_used"] is False
