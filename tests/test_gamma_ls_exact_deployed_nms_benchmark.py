from __future__ import annotations

import numpy as np
import pytest
import torch

from neurobench.experiments.gamma_ls_difference import (
    exact_deployed_nms_benchmark as exact,
)
from neurobench.metrics.sparse_detection import extract_separated_local_maxima


def _table_peaks(
    table: dict[str, np.ndarray], frame_index: int = 0
) -> list[tuple[float, int, int]]:
    rows = table["candidate_frame_index"] == frame_index
    return [
        (float(score), int(x), int(y))
        for score, x, y in zip(
            table["candidate_score"][rows],
            table["candidate_x"][rows],
            table["candidate_y"][rows],
            strict=True,
        )
    ]


def test_exact_cleanup_matches_maintained_plateau_and_tie_order() -> None:
    score = np.zeros((31, 35), dtype=np.float32)
    score[10:14, 10:14] = 5.0
    score[20, 24] = 5.0
    table = exact._candidate_table_from_score_maps(
        score, threshold_z=1.0, distance_px=3
    )
    expected = extract_separated_local_maxima(
        score.astype(np.float64),
        3,
        threshold=float(np.nextafter(np.float64(1.0), np.inf)),
    )
    assert _table_peaks(table) == expected
    assert _table_peaks(table)[:2] == [(5.0, 10, 10), (5.0, 13, 11)]
    assert table["candidate_rank"].tolist() == list(
        range(1, len(expected) + 1)
    )
    assert table["preliminary_count"][0] > table["count"][0]


def test_exact_cleanup_matches_random_continuous_and_strict_threshold() -> None:
    rng = np.random.default_rng(37)
    score = rng.normal(size=(31, 35)).astype(np.float32)
    score[8, 8] = 1.25
    threshold = 1.25
    table = exact._candidate_table_from_score_maps(
        score, threshold_z=threshold, distance_px=3
    )
    expected = extract_separated_local_maxima(
        score.astype(np.float64),
        3,
        threshold=float(np.nextafter(np.float64(threshold), np.inf)),
    )
    assert _table_peaks(table) == expected
    assert all(value > threshold for value, _, _ in _table_peaks(table))


def test_exact_cleanup_is_frame_local_and_records_complete_bounded_table() -> None:
    score = np.zeros((2, 31, 35), dtype=np.float32)
    score[0, 10:14, 10:14] = 5.0
    score[1, 8, 8] = 7.0
    score[1, 20, 24] = 6.0
    table = exact._candidate_table_from_score_maps(
        score, threshold_z=1.0, distance_px=3, limit_per_frame=1
    )
    assert table["count"].tolist() == [1, 1]
    assert table["limit_reached"].tolist() == [True, True]
    assert table["candidate_frame_index"].tolist() == [0, 1]
    assert table["candidate_rank"].tolist() == [1, 1]
    assert _table_peaks(table, 0) == [(5.0, 10, 10)]
    assert _table_peaks(table, 1) == [(7.0, 8, 8)]
    assert table["top_x"].tolist() == [10, 8]
    assert table["top_y"].tolist() == [10, 8]


def test_exact_cleanup_validates_sparse_candidate_inputs() -> None:
    with pytest.raises(ValueError, match="shape N x 3"):
        exact.cleanup_square_local_max_candidates(
            np.zeros((2, 2), dtype=np.int32),
            np.zeros(2, dtype=np.float32),
            batch_size=1,
        )
    with pytest.raises(ValueError, match="invalid frame"):
        exact.cleanup_square_local_max_candidates(
            np.asarray([[1, 2, 3]], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            batch_size=1,
        )
    with pytest.raises(ValueError, match="finite"):
        exact.cleanup_square_local_max_candidates(
            np.asarray([[0, 2, 3]], dtype=np.int32),
            np.asarray([np.nan], dtype=np.float32),
            batch_size=1,
        )


def test_matched_batch_plan_uses_same_cyclic_frame_and_candidate_multiset() -> None:
    batch_sizes = (1, 2, 4, 8)
    plan = exact.matched_cyclic_batch_plan(
        ring_frames=8,
        batch_sizes=batch_sizes,
        warmup_reference_iterations=2,
        timed_reference_iterations=3,
    )
    expected_warmup = np.tile(np.arange(8), 2)
    expected_timed = np.tile(np.arange(8), 3)
    candidate_burden_by_source_frame = np.asarray(
        [0, 3, 1, 5, 2, 7, 1, 4], dtype=np.int64
    )
    observed_burdens = []
    for batch_size in batch_sizes:
        workload = plan[batch_size]
        warmup = exact.cyclic_ring_frame_indices(
            ring_frames=8,
            batch_frames=batch_size,
            batch_count=workload["warmup_batches"],
        )
        timed = exact.cyclic_ring_frame_indices(
            ring_frames=8,
            batch_frames=batch_size,
            batch_count=workload["timed_batches"],
        )
        assert np.array_equal(warmup, expected_warmup)
        assert np.array_equal(timed, expected_timed)
        assert np.array_equal(np.bincount(timed, minlength=8), np.full(8, 3))
        observed_burdens.append(int(candidate_burden_by_source_frame[timed].sum()))
    assert len(set(observed_burdens)) == 1

    with pytest.raises(ValueError, match="divisor"):
        exact.matched_cyclic_batch_plan(
            ring_frames=8,
            batch_sizes=(1, 3),
            warmup_reference_iterations=2,
            timed_reference_iterations=3,
        )


def test_exact_parity_self_check_covers_plateaus_ties_batches_and_threshold() -> None:
    result = exact.exact_nms_parity_self_check()
    assert result["status"] == "passed_exact_maintained_extractor_parity"
    assert result["device"] == "cpu"
    assert {row["case_id"] for row in result["cases"]} == {
        "random_continuous",
        "flat_plateau_and_tie",
        "strict_threshold",
        "two_frame_batch",
    }
    assert all(row["passed"] for row in result["cases"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_square_mask_and_host_cleanup_match_maintained_extractor() -> None:
    result = exact.exact_nms_parity_self_check(device="cuda:0")
    assert result["status"] == "passed_exact_maintained_extractor_parity"
    assert result["device"] == "cuda:0"
