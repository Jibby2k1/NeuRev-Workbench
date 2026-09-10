from __future__ import annotations

import numpy as np

from neurobench.experiments.ica_whitening_evaluation.independent_confirmation import (
    INDEPENDENT_CANDIDATE_CONTRACT, build_independent_candidate_universe,
    candidate_contract_digest, independent_score_digest,
    independent_sparse_positive_metrics, fit_and_score_independent_finalist,
)


def test_independent_runner_requests_read_only_tiff_mapping() -> None:
    source = __import__(
        "inspect"
    ).getsource(__import__(
        "neurobench.experiments.ica_whitening_evaluation.independent_confirmation",
        fromlist=["run_independent_confirmation"],
    ).run_independent_confirmation)
    assert 'memmap(video_path, mode="r")' in source


def test_independent_candidates_are_deterministic_and_label_blind() -> None:
    rng = np.random.default_rng(19)
    movie = rng.normal(size=(125, 16, 17)).astype(np.float32)
    movie[70:75, 8, 9] += 8
    first, first_digest = build_independent_candidate_universe(movie)
    second, second_digest = build_independent_candidate_universe(movie.copy())
    assert first == second
    assert first_digest == second_digest
    assert {row["block_id"] for row in first} == {1, 2, 3}
    assert all(0 <= row["peak_frame"] < 125 for row in first)
    assert {row["block_stop_frame_exclusive"] for row in first if row["block_id"] == 3} == {125}
    assert INDEPENDENT_CANDIDATE_CONTRACT["unmatched_candidates"] == "unknown_not_negative"
    assert len(candidate_contract_digest()) == 64


def test_independent_sparse_positive_metrics_require_frozen_score_hash() -> None:
    candidates = [
        {"candidate_id": "a", "block_id": 1, "peak_frame": 12, "x_px": 5, "y_px": 5},
        {"candidate_id": "b", "block_id": 1, "peak_frame": 20, "x_px": 12, "y_px": 12},
        {"candidate_id": "c", "block_id": 2, "peak_frame": 70, "x_px": 8, "y_px": 9},
    ]
    annotations = [{
        "annotation_id": "roi_1", "crop_x": 5.0, "crop_y": 5.0,
        "spike_intervals": [{"start_frame": 10, "end_frame": 15}],
    }, {
        "annotation_id": "roi_2", "crop_x": 8.0, "crop_y": 9.0,
        "spike_intervals": [{"start_frame": 68, "end_frame": 75}],
    }]
    scores = np.asarray([3.0, 2.0, 1.0])
    digest = independent_score_digest(scores)
    result = independent_sparse_positive_metrics(
        candidates, scores, annotations, score_sha256_before_label_join=digest,
    )
    assert result["known_positive_count"] == 2
    assert result["pooled_budgets"][0]["pooled_known_positive_recall"] == 1.0
    assert result["mean_reciprocal_rank"] == 1.0
    assert result["precision_specificity_and_false_positive_rate"] == "not_identified"
    try:
        independent_sparse_positive_metrics(
            candidates, scores, annotations, score_sha256_before_label_join="wrong",
        )
    except RuntimeError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("label join must fail without the exact score hash")


def test_independent_finalist_fit_scores_without_labels() -> None:
    rng = np.random.default_rng(33)
    movie = rng.normal(size=(120, 12, 13)).astype(np.float32)
    candidates = [
        {"candidate_id": "a", "peak_frame": 2, "x_px": 5, "y_px": 5},
        {"candidate_id": "b", "peak_frame": 118, "x_px": 7, "y_px": 6},
    ]
    specification = {
        "fit_id": "frozen", "family": "temporal", "spatial_width_px": None,
        "temporal_width_frames": 3, "causality": "centered",
        "objective": "fastica_logcosh", "objective_scale": 1.0,
        "rank": 2, "seed": 7, "whitening_geometry": "none",
        "raw_preserving_blend": 0.0, "covariance_scope": "global_quiet",
    }
    scores, summary = fit_and_score_independent_finalist(
        movie, candidates, specification, maximum_fit_samples=256,
        score_chunk_candidates=1,
    )
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert summary["candidate_score_sha256_before_label_join"] == independent_score_digest(scores)
    assert summary["labels_accessed"] is False
    assert summary["reconstruction_integrity"]["reconstruction_nmse_centered"] < 1
