from __future__ import annotations

import json

import numpy as np
import pytest

from neurobench.experiments.neuron_identifiability import uncertainty_aware_models as model_core
from neurobench.experiments.neuron_identifiability.uncertainty_aware_models import (
    BaggedPURanker,
    EvaluationConfig,
    INCLUSIVE_LABEL_POLICY,
    PenalizedLogisticRanker,
    RobustFoldPreprocessor,
    STRICT_LABEL_POLICY,
    TinyTanhMLPRanker,
    cross_seed_rank_stability,
    fold_percentile_normalize,
    grouped_paired_bootstrap_rank_delta,
    grouped_spatial_folds,
    nested_spatial_pu_evaluation,
    positive_vs_unlabeled_rank_auc,
    rank_metrics,
    resolve_labels,
    spatial_group_permutation_null,
    spatial_group_score_permutation_null,
    spu_auc,
    summarize_folded_oof_ranking,
    validate_frozen_outer_folds,
    validate_raw_feature_names,
)


def _separable_reference(seed: int = 7, count: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat([1, 0], count // 2)
    X = rng.normal(size=(count, 4))
    X[:, 0] += np.where(y == 1, 2.0, -1.0)
    X[:, 1] += np.where(y == 1, 0.8, 0.0)
    return X, y


def _nested_fixture(seed: int = 13) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Forty-eight spatial groups provide ample nested P/U support."""

    rng = np.random.default_rng(seed)
    y = np.tile([1, 0], 24)
    groups = np.asarray([f"spatial_{index:02d}" for index in range(len(y))])
    labels = np.where(y == 1, "definite", "unlabeled").astype(object)
    X = rng.normal(size=(len(y), 5))
    X[:, 0] += np.where(y == 1, 1.8, -0.4)
    X[:, 1] += np.where(y == 1, 0.8, 0.0)
    X[:, 2] += np.where(y == 1, X[:, 3] ** 2, 0.0)
    X[3, 4] = np.nan
    feature_names = [
        "feature__carrier_signed",
        "feature__coherence_w15",
        "feature__recurrence",
        "feature__neighbor_context",
        "feature__quiet_normalized_upstream",
    ]
    baseline = X[:, 0].copy()
    baseline[np.isnan(baseline)] = 0.0
    return X, feature_names, labels, groups, baseline


def test_feature_contract_rejects_identity_review_rank_and_cohort_z_leakage() -> None:
    allowed = validate_raw_feature_names(
        ["feature__carrier_signed", "feature__quiet_normalized_upstream", "missing__feature__snr"]
    )
    assert allowed[1] == "feature__quiet_normalized_upstream"
    for forbidden in (
        "z_carrier",
        "carrier_z",
        "cohort_z_carrier",
        "candidate_rank",
        "priority_score",
        "expert_label",
        "x_int",
        "observation_site_id",
    ):
        with pytest.raises(ValueError, match="forbidden|leaky"):
            validate_raw_feature_names(["feature__carrier_signed", forbidden])
    with pytest.raises(ValueError, match="unique"):
        validate_raw_feature_names(["snr", "snr"])


def test_strict_and_inclusive_label_policies_keep_unknowns_out_of_negative_semantics() -> None:
    labels = ["definite", "probable", "uncertain", "unlabeled", "artifact"]
    strict = resolve_labels(labels, STRICT_LABEL_POLICY)
    inclusive = resolve_labels(labels, INCLUSIVE_LABEL_POLICY)
    assert strict.positive.tolist() == [True, False, False, False, False]
    assert strict.unlabeled.tolist() == [False, True, True, True, False]
    assert strict.excluded.tolist() == [False, False, False, False, True]
    assert inclusive.positive.tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError, match="unrecognised"):
        resolve_labels(["definite", "invented_state"], STRICT_LABEL_POLICY)


def test_robust_preprocessor_is_fold_local_missing_aware_and_positive_referenced() -> None:
    train = np.asarray(
        [
            [10.0, 1.0, np.nan],
            [11.0, 1.2, 2.0],
            [9.0, 0.8, 2.1],
            [1.0, -1.0, 8.0],
            [2.0, -0.8, 7.5],
            [0.0, -1.2, 8.5],
        ]
    )
    positive = np.asarray([True, True, True, False, False, False])
    processor = RobustFoldPreprocessor(clip=5.0).fit(
        train,
        positive,
        feature_names=["carrier", "coherence", "kinetics"],
    )
    learned_location = processor.location_.copy()
    transformed = processor.transform(np.asarray([[10.0, 1.0, 2.0], [100_000.0, -100_000.0, np.nan]]))
    assert np.array_equal(processor.location_, learned_location)
    assert np.all(np.isfinite(transformed))
    assert np.max(np.abs(transformed[:, :3])) <= 5.0
    assert "missing__kinetics" in processor.feature_names_out_
    assert "positive_reference_distance" not in processor.feature_names_out_
    distance = processor.positive_reference_distance(
        np.asarray([[10.0, 1.0, 2.0], [100_000.0, -100_000.0, np.nan]])
    )
    assert distance[0] < distance[1]
    with pytest.raises(ValueError, match="all-missing"):
        RobustFoldPreprocessor().fit(
            np.asarray([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [4.0, np.nan]]),
            [True, True, False, False],
            feature_names=["valid", "missing"],
        )


def test_penalized_logistic_rankers_are_deterministic_and_rank_signal() -> None:
    X, y = _separable_reference()
    weights = np.where(y == 1, 1.0, 1.0)
    first = PenalizedLogisticRanker(penalty_strength=0.05, l1_ratio=0.0).fit(X, y, sample_weight=weights)
    second = PenalizedLogisticRanker(penalty_strength=0.05, l1_ratio=0.0).fit(X, y, sample_weight=weights)
    elastic = PenalizedLogisticRanker(penalty_strength=0.1, l1_ratio=0.5).fit(X, y)
    assert np.array_equal(first.predict_score(X), second.predict_score(X))
    assert spu_auc(y == 1, first.predict_score(X)) > 0.95
    assert spu_auc(y == 1, elastic.predict_score(X)) > 0.9
    assert first.converged_ and elastic.converged_
    with pytest.raises(ValueError, match="positive and unlabeled"):
        PenalizedLogisticRanker().fit(X[:20], np.ones(20, dtype=int))


def test_bagged_pu_ranker_is_deterministic_and_uses_only_resampled_unlabeled_reference() -> None:
    X, y = _separable_reference(seed=11)
    first = BaggedPURanker(bags=8, unlabeled_ratio=1.5, penalty_strength=0.1, seed=91).fit(X, y)
    second = BaggedPURanker(bags=8, unlabeled_ratio=1.5, penalty_strength=0.1, seed=91).fit(X, y)
    assert np.array_equal(first.predict_score(X), second.predict_score(X))
    assert first.sampled_unlabeled_indices_ == second.sampled_unlabeled_indices_
    assert first.component_scores(X).shape == (8, len(X))
    assert spu_auc(y == 1, first.predict_score(X)) > 0.9
    unlabeled = set(np.flatnonzero(y == 0))
    assert all(set(bag).issubset(unlabeled) for bag in first.sampled_unlabeled_indices_)


def test_logistic_tolerance_is_fail_closed_and_honored_by_every_reference_path(monkeypatch) -> None:
    for invalid in (0.0, -1e-8, float("nan"), float("inf"), "not-a-number"):
        with pytest.raises(ValueError, match="finite and positive"):
            EvaluationConfig(logistic_tolerance=invalid)
        with pytest.raises(ValueError, match="finite and positive"):
            BaggedPURanker(tolerance=invalid)

    requested = 2.5e-8
    observed: list[float] = []

    class CapturingRanker:
        def __init__(self, *, tolerance: float, **_: object) -> None:
            observed.append(float(tolerance))

        def fit(self, X: np.ndarray, y: np.ndarray, **_: object) -> "CapturingRanker":
            assert len(X) == len(y)
            return self

        def predict_score(self, X: np.ndarray) -> np.ndarray:
            return np.linspace(0.25, 0.75, len(X), dtype=np.float64)

    monkeypatch.setattr(model_core, "PenalizedLogisticRanker", CapturingRanker)
    config = EvaluationConfig(
        logistic_tolerance=requested,
        pu_bags=3,
        permutation_draws=2,
    )
    X_train = np.arange(48, dtype=np.float64).reshape(12, 4)
    y_train = np.tile([1, 0], 6)
    X_test = np.arange(20, dtype=np.float64).reshape(5, 4)
    for model_name in ("logistic_l2", "logistic_elastic", "bagged_pu_logistic"):
        scores = model_core._fit_reference_model(
            model_name,
            X_train,
            y_train,
            X_test,
            penalty=0.1,
            config=config,
            seed=91,
        )
        assert scores.shape == (len(X_test),)
    assert observed == [requested, requested, requested, requested, requested]


def test_tiny_mlp_has_exact_architecture_deterministic_early_stopping_and_fixed_refit() -> None:
    X, y = _separable_reference(seed=23)
    train = np.concatenate([np.arange(30), np.arange(40, 70)])
    validation = np.concatenate([np.arange(30, 40), np.arange(70, 80)])
    model_a = TinyTanhMLPRanker(seed=5, l2=0.01, max_epochs=160, patience=12).fit(
        X[train],
        y[train],
        validation=(X[validation], y[validation], None),
    )
    model_b = TinyTanhMLPRanker(seed=5, l2=0.01, max_epochs=160, patience=12).fit(
        X[train],
        y[train],
        validation=(X[validation], y[validation], None),
    )
    assert model_a.hidden_units == 4
    assert model_a.parameter_count_ == X.shape[1] * 4 + 4 + 4 + 1
    assert model_a.best_epoch_ <= model_a.n_epochs_ <= 160
    assert np.array_equal(model_a.predict_score(X), model_b.predict_score(X))
    refit = TinyTanhMLPRanker(seed=5, l2=0.01).fit(X, y, fixed_epochs=model_a.best_epoch_)
    assert refit.n_epochs_ == model_a.best_epoch_
    assert np.all(np.isfinite(refit.predict_score(X)))


def test_twelve_complete_raw_features_produce_exactly_fifty_seven_mlp_parameters() -> None:
    rng = np.random.default_rng(88)
    X = rng.normal(size=(40, 12))
    y = np.tile([1, 0], 20)
    transformed = RobustFoldPreprocessor().fit_transform(
        X,
        y == 1,
        feature_names=[f"feature__f{index}" for index in range(12)],
    )
    assert transformed.shape == (40, 12)
    model = TinyTanhMLPRanker(seed=2, max_epochs=30, patience=4).fit(transformed, y, fixed_epochs=10)
    assert model.parameter_count_ == 57


def test_grouped_spatial_folds_are_reproducible_disjoint_and_fail_closed() -> None:
    y = np.tile([1, 0], 12)
    groups = np.asarray([f"g{index}" for index in range(24)])
    first = grouped_spatial_folds(y, groups, n_splits=4, seed=8)
    second = grouped_spatial_folds(y, groups, n_splits=4, seed=8)
    assert all(np.array_equal(a, b) and np.array_equal(c, d) for (a, c), (b, d) in zip(first, second))
    covered: list[int] = []
    for train, test in first:
        assert not (set(groups[train]) & set(groups[test]))
        assert set(y[train]) == {0, 1}
        assert set(y[test]) == {0, 1}
        covered.extend(test.tolist())
    assert sorted(covered) == list(range(len(y)))
    sparse_y = np.asarray([1, 1, 0, 0, 0, 0])
    sparse_groups = np.asarray(["p1", "p2", "u1", "u2", "u3", "u4"])
    with pytest.raises(ValueError, match="insufficient"):
        grouped_spatial_folds(sparse_y, sparse_groups, n_splits=3, seed=1)


def test_rank_metrics_use_spu_names_and_fixed_budget() -> None:
    positive = np.asarray([True, False, True, False, False])
    scores = np.asarray([0.9, 0.8, 0.7, 0.1, 0.0])
    metrics = rank_metrics(positive, scores, budget=2, tie_breakers=["a", "b", "c", "d", "e"])
    assert np.isclose(metrics["positive_vs_unlabeled_rank_auc"], 5 / 6)
    assert metrics["positive_recovered_at_budget"] == 1
    assert metrics["positive_recall_at_budget"] == 0.5
    assert "roc_auc" not in metrics
    assert "precision" not in metrics
    tied = spu_auc([True, False], [1.0, 1.0])
    assert tied == 0.5


def test_fold_percentile_oof_metrics_are_invariant_to_fold_logit_shifts() -> None:
    positive = np.asarray([True, False, True, False, True, False, True, False])
    folds = [np.arange(0, 4), np.arange(4, 8)]
    base = np.asarray([0.9, 0.8, 0.7, 0.1, 0.95, 0.85, 0.65, 0.2])
    shifted = base.copy()
    shifted[folds[1]] += 10_000.0
    assert np.array_equal(fold_percentile_normalize(base, folds), fold_percentile_normalize(shifted, folds))
    base_summary = summarize_folded_oof_ranking(positive, base, folds, budget=4)
    shifted_summary = summarize_folded_oof_ranking(positive, shifted, folds, budget=4)
    assert base_summary["metrics"] == shifted_summary["metrics"]
    assert base_summary["oof_scores_fold_percentile"] == shifted_summary["oof_scores_fold_percentile"]
    assert (
        base_summary["raw_score_global_sensitivity_metrics"]
        != shifted_summary["raw_score_global_sensitivity_metrics"]
    )


def test_seed_stability_and_spatial_group_permutation_are_deterministic() -> None:
    base = np.linspace(0.0, 1.0, 12)
    stability = cross_seed_rank_stability({0: base, 1: base + 1e-9, 2: base[::-1]})
    assert stability["pair_count"] == 3
    assert stability["minimum_pairwise_spearman"] == pytest.approx(-1.0)
    groups = np.asarray([f"g{index}" for index in range(24)])
    positive = np.tile([True, False], 12)
    scores = np.tile([1.0, 0.0], 12) + np.repeat(np.linspace(0, 0.1, 12), 2)
    first = spatial_group_permutation_null(positive, scores, groups, draws=50, seed=7)
    second = spatial_group_permutation_null(positive, scores, groups, draws=50, seed=7)
    assert first == second
    assert first == spatial_group_score_permutation_null(positive, scores, groups, draws=50, seed=7)
    assert first["unit"] == "spatial_group"
    assert first["draws"] == 50
    assert "models are not refitted" in first["null_semantics"]


def test_grouped_paired_bootstrap_rank_delta_is_paired_deterministic_and_grouped() -> None:
    positive = np.asarray([True] * 8 + [False] * 12)
    strong = np.asarray([1.0] * 8 + [0.0] * 12) + np.linspace(0, 0.01, 20)
    weak = np.linspace(0, 1, 20)
    positive_groups = [f"p{index // 2}" if positive[index] else "unused" for index in range(20)]
    unlabeled_groups = ["unused" if positive[index] else f"u{(index - 8) // 2}" for index in range(20)]
    first = grouped_paired_bootstrap_rank_delta(
        positive,
        strong,
        weak,
        positive_groups,
        unlabeled_groups,
        draws=100,
        seed=22,
        score_a_name="tiny_mlp",
        score_b_name="linear_pu",
    )
    second = grouped_paired_bootstrap_rank_delta(
        positive,
        strong,
        weak,
        positive_groups,
        unlabeled_groups,
        draws=100,
        seed=22,
        score_a_name="tiny_mlp",
        score_b_name="linear_pu",
    )
    assert first == second
    assert first["observed_delta_a_minus_b"] > 0
    assert first["positive_identity_group_count"] == 4
    assert first["unlabeled_spatial_group_count"] == 6


def test_nested_spatial_pu_evaluation_scores_every_row_without_group_leakage() -> None:
    X, names, labels, groups, baseline = _nested_fixture()
    labels[-1] = "artifact"
    config = EvaluationConfig(
        outer_splits=3,
        inner_splits=2,
        candidate_budget=12,
        seed=31,
        logistic_penalties=(0.1,),
        elastic_penalties=(0.1,),
        pu_penalties=(0.1,),
        mlp_l2_values=(0.01,),
        mlp_seeds=(0, 1),
        pu_bags=4,
        max_logistic_iterations=3000,
        mlp_max_epochs=70,
        mlp_patience=7,
        permutation_draws=25,
    )
    result = nested_spatial_pu_evaluation(
        X,
        labels,
        groups,
        feature_names=names,
        sample_ids=[f"row_{index:03d}" for index in range(len(X))],
        baseline_scores={"frozen_cfar": baseline},
        config=config,
    )
    assert result["status"] == "complete_engineering_evaluation"
    assert result["candidate_count"] == len(X) - 1
    assert result["excluded_count"] == 1
    assert set(result["models"]) == {
        "positive_reference_distance",
        "logistic_l2",
        "logistic_elastic",
        "bagged_pu_logistic",
        "tiny_mlp_4_tanh",
    }
    for model in result["models"].values():
        assert len(model["oof_scores"]) == result["candidate_count"]
        assert np.all(np.isfinite(model["oof_scores"]))
        assert 0.0 <= model["metrics"]["positive_vs_unlabeled_rank_auc"] <= 1.0
    assert len(result["models"]["tiny_mlp_4_tanh"]["seed_oof_scores"]) == 2
    assert all(fold["spatial_groups_disjoint"] for fold in result["outer_folds"])
    assert all(
        fold["learned_model_input_width"] == len(fold["learned_model_feature_names"])
        for fold in result["outer_folds"]
    )
    assert all(
        "positive_reference_distance" not in fold["learned_model_feature_names"]
        for fold in result["outer_folds"]
    )
    assert {fold["learned_model_input_width"] for fold in result["outer_folds"]}.issubset({5, 6})
    assert "frozen_cfar" in result["baselines"]
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_nested_evaluation_is_exactly_reproducible() -> None:
    X, names, labels, groups, baseline = _nested_fixture(seed=29)
    config = EvaluationConfig(
        outer_splits=3,
        inner_splits=2,
        candidate_budget=10,
        seed=4,
        logistic_penalties=(0.1,),
        elastic_penalties=(0.1,),
        pu_penalties=(0.1,),
        mlp_l2_values=(0.01,),
        mlp_seeds=(0, 1),
        pu_bags=3,
        mlp_max_epochs=40,
        mlp_patience=5,
        permutation_draws=10,
    )
    kwargs = dict(
        feature_names=names,
        sample_ids=[f"candidate_{index}" for index in range(len(X))],
        baseline_scores={"frozen": baseline},
        equal_weight_feature_names=names[:3],
        frozen_outer_folds=_contiguous_guarded_folds(len(X)),
        config=config,
    )
    first = nested_spatial_pu_evaluation(X, labels, groups, **kwargs)
    second = nested_spatial_pu_evaluation(X, labels, groups, **kwargs)
    assert first == second
    assert first["outer_fold_source"] == "caller_frozen_with_optional_training_guard_omissions"
    assert "equal_weight_feature_separation" in first["models"]
    assert any(row["training_guard_omitted_candidate_count"] > 0 for row in first["outer_folds"])


def _contiguous_guarded_folds(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    assert count == 48
    all_indices = np.arange(count)
    output = []
    for start, stop in ((0, 16), (16, 32), (32, 48)):
        test = np.arange(start, stop)
        train = np.setdiff1d(all_indices, test)
        guards = [index for index in (start - 1, stop) if 0 <= index < count]
        train = np.setdiff1d(train, guards)
        output.append((train, test))
    return output


def test_frozen_outer_folds_require_exact_test_coverage_and_allow_guard_omissions() -> None:
    y = np.tile([True, False], 24)
    groups = [f"g{index}" for index in range(48)]
    folds = validate_frozen_outer_folds(
        _contiguous_guarded_folds(48),
        included_input_indices=np.arange(48),
        positive_mask=y,
        spatial_groups=groups,
        expected_folds=3,
    )
    assert len(folds) == 3
    assert any(len(train) + len(test) < 48 for train, test in folds)
    broken = _contiguous_guarded_folds(48)
    broken[0] = (broken[0][0], broken[0][1][1:])
    with pytest.raises(ValueError, match="cover every"):
        validate_frozen_outer_folds(
            broken,
            included_input_indices=np.arange(48),
            positive_mask=y,
            spatial_groups=groups,
            expected_folds=3,
        )


def test_optional_nested_selection_uses_only_grouped_outer_training_rows() -> None:
    X, names, labels, groups, _ = _nested_fixture(seed=47)
    config = EvaluationConfig(
        outer_splits=2,
        inner_splits=2,
        candidate_budget=10,
        seed=17,
        hyperparameter_selection="nested",
        logistic_penalties=(0.01, 0.1),
        elastic_penalties=(0.1,),
        pu_penalties=(0.1,),
        mlp_l2_values=(0.01,),
        mlp_seeds=(0, 1),
        pu_bags=3,
        mlp_max_epochs=25,
        mlp_patience=3,
        permutation_draws=5,
    )
    result = nested_spatial_pu_evaluation(X, labels, groups, feature_names=names, config=config)
    assert all(len(fold["inner_selection"]["logistic_l2"]) == 2 for fold in result["outer_folds"])
    assert all(
        "inner_positive_vs_unlabeled_rank_auc" in row
        for fold in result["outer_folds"]
        for row in fold["inner_selection"]["logistic_l2"]
    )
    assert all(fold["spatial_groups_disjoint"] for fold in result["outer_folds"])


def test_zero_permutation_draws_skips_legacy_group_max_null() -> None:
    X, names, labels, groups, baseline = _nested_fixture(seed=53)
    config = EvaluationConfig(
        outer_splits=3,
        inner_splits=2,
        candidate_budget=10,
        seed=19,
        logistic_penalties=(0.1,),
        elastic_penalties=(0.1,),
        pu_penalties=(0.1,),
        mlp_l2_values=(0.01,),
        mlp_seeds=(0, 1),
        pu_bags=3,
        mlp_max_epochs=25,
        mlp_patience=3,
        permutation_draws=0,
    )
    result = nested_spatial_pu_evaluation(
        X,
        labels,
        groups,
        feature_names=names,
        baseline_scores={"frozen": baseline},
        config=config,
    )
    assert all(
        "spatial_group_score_association_null" not in payload
        for payload in (*result["models"].values(), *result["baselines"].values())
    )


def test_interleaved_excluded_rows_are_never_used_or_scored_by_local_fold_indices() -> None:
    X, names, labels, groups, baseline = _nested_fixture(seed=61)
    excluded = np.zeros(len(labels), dtype=bool)
    excluded[[1, 14, 33]] = True
    full_labels = labels.copy()
    full_labels[excluded] = "artifact"
    config = EvaluationConfig(
        outer_splits=3,
        inner_splits=2,
        candidate_budget=10,
        seed=12,
        logistic_penalties=(0.1,),
        elastic_penalties=(0.1,),
        pu_penalties=(0.1,),
        mlp_l2_values=(0.01,),
        mlp_seeds=(0, 1),
        pu_bags=3,
        mlp_max_epochs=25,
        mlp_patience=3,
        permutation_draws=5,
    )
    full = nested_spatial_pu_evaluation(
        X,
        full_labels,
        groups,
        feature_names=names,
        sample_ids=[f"row_{index}" for index in range(len(X))],
        baseline_scores={"frozen": baseline},
        config=config,
    )
    kept = ~excluded
    filtered = nested_spatial_pu_evaluation(
        X[kept],
        labels[kept],
        groups[kept],
        feature_names=names,
        sample_ids=[f"row_{index}" for index in np.flatnonzero(kept)],
        baseline_scores={"frozen": baseline[kept]},
        config=config,
    )
    assert full["included_sample_ids"] == filtered["included_sample_ids"]
    assert full["excluded_count"] == 3
    for model_name in full["models"]:
        assert full["models"][model_name]["oof_scores"] == filtered["models"][model_name]["oof_scores"]
        assert full["models"][model_name]["metrics"] == filtered["models"][model_name]["metrics"]
    assert full["baselines"] == filtered["baselines"]


def test_nested_evaluation_fails_before_fitting_when_group_support_is_insufficient() -> None:
    X = np.arange(24, dtype=float).reshape(8, 3)
    labels = ["definite", "definite", "unlabeled", "unlabeled", "unlabeled", "unlabeled", "unlabeled", "unlabeled"]
    groups = ["p1", "p2", "u1", "u2", "u3", "u4", "u5", "u6"]
    with pytest.raises(ValueError, match="cannot support nested|positive_groups"):
        nested_spatial_pu_evaluation(
            X,
            labels,
            groups,
            feature_names=["carrier", "coherence", "kinetics"],
            config=EvaluationConfig(
                outer_splits=3,
                inner_splits=2,
                logistic_penalties=(0.1,),
                elastic_penalties=(0.1,),
                pu_penalties=(0.1,),
                mlp_l2_values=(0.1,),
                mlp_seeds=(0, 1),
                pu_bags=3,
                permutation_draws=5,
            ),
        )
