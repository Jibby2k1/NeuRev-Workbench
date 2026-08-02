from __future__ import annotations

import numpy as np

from neurobench.algorithms.proposal_ranking import (
    CandidateTable,
    annular_kernel,
    cut_morphology_basis,
    fit_bounded_pairwise_linear,
    fit_residual_mlp_ranker,
    merge_peak_proposals,
    normalize_map,
    robust_map_normalizer,
    sample_candidate_features,
    score_bounded_pairwise_linear,
    score_residual_mlp_ranker,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_program import (
    _evaluate_candidate_budget,
)


def test_annular_kernel_and_cut_basis_detect_expected_geometry() -> None:
    kernel = annular_kernel(4.5, 1.25)
    assert np.isclose(kernel.sum(), 1)
    assert kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] < kernel.max()
    yy, xx = np.mgrid[:25, :25]
    radius = np.sqrt((xx - 12) ** 2 + (yy - 12) ** 2)
    ring = np.exp(-0.5 * ((radius - 4.5) / 0.8) ** 2).astype(np.float32)
    basis = cut_morphology_basis(
        ring,
        center_sigmas_px=(1.5, 2.5, 3.5),
        ring_specs=((4.5, 1.25),),
        crowd_sigma_px=8,
    )
    assert set(basis) == {
        "cut_center_sigma1p5",
        "cut_center_sigma2p5",
        "cut_center_sigma3p5",
        "cut_ring_r4p5_t1p25",
        "cut_crowd_context",
    }
    assert basis["cut_ring_r4p5_t1p25"][12, 12] > 0


def test_proposal_union_deduplicates_sources_and_samples_features() -> None:
    first = np.zeros((24, 24), dtype=np.float32)
    second = np.zeros_like(first)
    first[10, 10] = 4
    second[11, 10] = 5
    first[4, 4] = 3
    quiet = [np.zeros_like(first) + index * 0.01 for index in range(4)]
    normalizers = {
        "first": robust_map_normalizer(quiet),
        "second": robust_map_normalizer(quiet),
    }
    positions, counts = merge_peak_proposals(
        {"first": first, "second": second},
        normalizers,
        nms_distance_px=3,
        per_source_limit=4,
        dedupe_radius_px=2,
        clip=8,
    )
    assert len(positions) >= 2
    assert counts.max() == 2
    features = sample_candidate_features(
        positions,
        {"first": first, "second": second},
        ("first", "second"),
        normalizers,
        clip=8,
    )
    table = CandidateTable(positions, features, counts)
    assert table.features.shape == (len(positions), 2)


def test_proposal_union_excludes_zero_evidence_plateaus() -> None:
    score = np.zeros((24, 24), dtype=np.float32)
    normalizer = {"zero": (0.0, 1.0)}
    positions, counts = merge_peak_proposals(
        {"zero": score},
        normalizer,
        nms_distance_px=3,
        per_source_limit=10,
        dedupe_radius_px=2,
        clip=8,
    )
    assert positions.shape == (0, 2)
    assert counts.shape == (0,)


def test_quiet_normalization_is_strictly_monotone_above_zero() -> None:
    values = np.asarray([[0.0, 1.0, 8.0, 80.0]], dtype=np.float32)
    normalized = normalize_map(values, (0.0, 1.0), clip=8.0)
    assert np.all(np.diff(normalized[0]) > 0)
    assert normalized[0, -1] > 1.0


def _training_fixture() -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(
        [[1.0, 0.9, 0.1], [0.8, 0.8, 0.2], [1.2, 0.7, 0.1]],
        dtype=np.float64,
    )
    negative = np.asarray(
        [[0.4, 0.1, 0.7], [0.3, 0.2, 0.8], [0.5, 0.1, 0.6]],
        dtype=np.float64,
    )
    return positive, negative


def test_linear_ranker_starts_from_carrier_and_respects_authority() -> None:
    positive, negative = _training_fixture()
    model = fit_bounded_pairwise_linear(
        positive,
        negative,
        carrier_column=0,
        auxiliary_columns=(1, 2),
        auxiliary_directions=(1, -1),
        learning_rate=0.02,
        epochs=100,
        l2=0.1,
        maximum_total=0.5,
    )
    assert np.sum(model["weights"]) <= 0.5 + 1e-12
    assert model["loss_final"] < model["loss_initial"]
    assert np.mean(score_bounded_pairwise_linear(positive, model)) > np.mean(
        score_bounded_pairwise_linear(negative, model)
    )


def test_residual_mlp_is_deterministic_and_bounded() -> None:
    positive, negative = _training_fixture()
    kwargs = dict(
        carrier_column=0,
        input_columns=(0, 1, 2),
        hidden_units=4,
        maximum_residual=0.25,
        learning_rate=0.003,
        epochs=60,
        weight_decay=0.01,
        seed=17,
    )
    first = fit_residual_mlp_ranker(positive, negative, **kwargs)
    second = fit_residual_mlp_ranker(positive, negative, **kwargs)
    first_score = score_residual_mlp_ranker(positive, first)
    second_score = score_residual_mlp_ranker(positive, second)
    assert np.allclose(first_score, second_score)
    assert np.max(np.abs(first_score - positive[:, 0])) <= 0.25 + 1e-12
    assert first["loss_final"] < first["loss_initial"]


def test_candidate_budget_returns_exact_recovered_label_indices() -> None:
    table = CandidateTable(
        positions=np.asarray([[5, 5], [15, 15], [25, 25]], dtype=np.int32),
        features=np.asarray([[0.1], [0.9], [0.8]], dtype=np.float64),
        source_count=np.ones(3, dtype=np.int32),
    )
    labels = [
        {"x_px": 15.0, "y_px": 15.0},
        {"x_px": 25.0, "y_px": 25.0},
    ]
    result = _evaluate_candidate_budget(
        table.features[:, 0], table, labels, budget=2, match_radius_px=2.0
    )
    assert result["matched"] == 2
    assert result["recall"] == 1.0
    assert result["label_indices"] == [0, 1]
