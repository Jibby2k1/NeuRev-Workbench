import numpy as np

from neurobench.experiments.hard_roi_adjudication.reevaluate import (
    _candidate_records,
    _failure_reason,
    _match_spatiotemporal,
)


def _label(roi: str, x: float, y: float, start: int = 2003, end: int = 2026):
    return {
        "roi_identity": roi,
        "x_px": x,
        "y_px": y,
        "start_frame_ui": start,
        "end_frame_ui": end,
    }


def test_candidate_peak_frame_uses_one_based_frozen_frame_contract() -> None:
    values = np.zeros((560, 20, 20), dtype=np.float32)
    values[5, 10, 10] = 5  # frozen UI frame 1805
    score = np.zeros((20, 20), dtype=np.float32)
    score[10, 10] = 5
    candidates = _candidate_records(
        values, score, 1800, 1810, distance=2, limit=10
    )
    assert candidates[0]["x_px"] == 10
    assert candidates[0]["peak_frame_ui"] == 1805


def test_spatiotemporal_matching_rejects_spatial_peak_outside_event() -> None:
    candidates = [
        {"rank": 1, "x_px": 10, "y_px": 10, "peak_frame_ui": 2031},
        {"rank": 2, "x_px": 10, "y_px": 10, "peak_frame_ui": 2045},
    ]
    labels = [_label("roi_a", 10, 10, 2040, 2063)]
    matches, used = _match_spatiotemporal(candidates, labels, 6.0)
    assert matches == {0: 1}
    assert used == {1}


def test_failure_decomposition_separates_timing_ranking_and_localization() -> None:
    label = _label("roi_a", 10, 10, 2040, 2063)
    temporal = [{"rank": 1, "x_px": 10, "y_px": 10, "peak_frame_ui": 2031}]
    reason = _failure_reason(
        0, [label], temporal, temporal, {}, budget=58, radius=6,
        relaxed_radius=10,
    )[0]
    assert reason == "temporal_miss"

    # Candidate records are score-ranked, so list order and the one-based rank
    # field must agree. The correct spatial event appears only at rank 70.
    ranking = [
        {"rank": index + 1, "x_px": 0, "y_px": 0, "peak_frame_ui": 2045}
        for index in range(69)
    ] + [{"rank": 70, "x_px": 10, "y_px": 10, "peak_frame_ui": 2045}]
    reason = _failure_reason(
        0, [label], ranking, ranking, {}, budget=58, radius=6,
        relaxed_radius=10,
    )[0]
    assert reason == "ranking_miss"

    localized = [{"rank": 5, "x_px": 18, "y_px": 10, "peak_frame_ui": 2045}]
    reason = _failure_reason(
        0, [label], localized, localized, {}, budget=58, radius=6,
        relaxed_radius=10,
    )[0]
    assert reason == "localization_miss"
