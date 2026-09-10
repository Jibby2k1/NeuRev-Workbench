import numpy as np

from neurobench.experiments.ica_whitening_evaluation.model import PatchObservations
from neurobench.experiments.ica_whitening_evaluation.real_controls import (
    evaluate_rank_matched_controls,
)
from neurobench.experiments.ica_whitening_evaluation.real_runner import (
    _candidate_event_observations, _candidate_observations, _label_metrics,
)


def test_candidate_observations_follow_family_geometry():
    movie = np.arange(9 * 7 * 7, dtype=np.float32).reshape(9, 7, 7)
    proposals = [{"candidate_id": "c", "burst_id": 1, "x_px": 3, "y_px": 3,
                  "peak_frame_review_zero": 4}]
    temporal = _candidate_observations(movie, proposals, {
        "family": "temporal", "temporal_width_frames": 3,
        "spatial_width_px": None, "causality": "centered",
    })
    spatial = _candidate_observations(movie, proposals, {
        "family": "spatial", "temporal_width_frames": None,
        "spatial_width_px": 3, "causality": "centered",
    })
    joint = _candidate_observations(movie, proposals, {
        "family": "joint_spatiotemporal", "temporal_width_frames": 3,
        "spatial_width_px": 3, "causality": "centered",
    })
    assert temporal.shape == (3, 1)
    assert spatial.shape == (9, 1)
    assert joint.shape == (27, 1)


def test_label_metrics_keep_unmatched_candidates_unknown():
    proposals = [
        {"candidate_id": "a", "burst_id": burst, "x_px": 2, "y_px": 2,
         "peak_frame_review_zero": 0}
        for burst in (1, 2, 3, 4)
    ]
    labels = [
        {"burst_id": burst, "x_px": 2.0, "y_px": 2.0}
        for burst in (1, 2, 3, 4)
    ]
    result = _label_metrics(proposals, np.ones(4), labels, (1, 58), 1)
    assert result["macro_known_positive_recall"] == 1.0
    assert result["precision_identified"] is False
    assert result["unmatched_candidates"] == "unknown_not_negative"


def test_candidate_event_observations_are_candidate_major_within_burst():
    movie = np.arange(8 * 5 * 5, dtype=np.float32).reshape(8, 5, 5)
    proposals = [
        {"candidate_id": "a", "burst_id": 1, "x_px": 2, "y_px": 2},
        {"candidate_id": "b", "burst_id": 1, "x_px": 3, "y_px": 2},
    ]
    values, groups = _candidate_event_observations(
        movie, proposals,
        {"family": "temporal", "temporal_width_frames": 3,
         "spatial_width_px": None, "causality": "centered"},
        {1: (3, 5)},
    )
    assert values.shape == (3, 4)
    assert groups[0]["duration"] == 2
    assert groups[0]["candidate_indices"].tolist() == [0, 1]
    np.testing.assert_array_equal(values[:, 0], movie[2:5, 2, 2])
    np.testing.assert_array_equal(values[:, 2], movie[2:5, 2, 3])


def test_rank_matched_controls_are_deterministic_and_label_safe():
    rng = np.random.default_rng(4)
    fit = PatchObservations(
        values=rng.normal(size=(3, 96)), times=np.arange(96) % 12,
        rows=np.zeros(96, dtype=np.int32), columns=np.zeros(96, dtype=np.int32),
    )
    proposals = []
    groups = []
    offset = 0
    for burst in (1, 2, 3, 4):
        indices = np.arange(len(proposals), len(proposals) + 2)
        proposals.extend([
            {"candidate_id": f"{burst}_a", "burst_id": burst, "x_px": 2, "y_px": 2},
            {"candidate_id": f"{burst}_b", "burst_id": burst, "x_px": 8, "y_px": 8},
        ])
        groups.append({"burst_id": burst, "candidate_indices": indices,
                       "column_start": offset, "column_stop": offset + 6,
                       "duration": 3})
        offset += 6
    candidates = rng.normal(size=(3, offset))
    labels = [{"burst_id": burst, "x_px": 2.0, "y_px": 2.0}
              for burst in (1, 2, 3, 4)]
    specification = {"fit_id": "x", "family": "temporal",
                     "whitening_geometry": "none", "rank": 2, "seed": 9}
    kwargs = dict(quiet_count=6, budgets=(1, 2, 58), match_radius_px=1)
    first = evaluate_rank_matched_controls(
        fit, candidates, groups, proposals, labels, specification, **kwargs
    )
    second = evaluate_rank_matched_controls(
        fit, candidates, groups, proposals, labels, specification, **kwargs
    )
    assert first == second
    assert first["label_safe_score_before_metrics"] is True
    assert {row["control_id"] for row in first["controls"]} == {
        "rank_matched_pca", "rank_matched_random_rotation"
    }
    assert all(row["unmatched_candidates"] == "unknown_not_negative"
               for row in first["controls"])
