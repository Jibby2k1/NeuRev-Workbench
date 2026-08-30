from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.information_source_separation.semi_synthetic import (
    SemiSyntheticFixture,
)
from neurobench.experiments.neuron_identifiability.jepa_evaluation import (
    InjectionEvaluationError,
    evaluate_paired_fixture,
    evaluate_paired_suite,
    footprint_centers_yx,
    hierarchical_grouped_bootstrap,
    hierarchical_strongest_comparator_bootstrap,
    local_maxima_yx,
    make_native_background_injection,
    match_injected_sources,
    paired_sign_flip_test,
)


def _fixture(identifier: str = "pair-1") -> SemiSyntheticFixture:
    frames, height, width = 12, 16, 16
    footprints = np.zeros((2, height, width), dtype=np.float32)
    footprints[0, 5, 6] = 1
    footprints[1, 10, 11] = 1
    traces = np.zeros((2, frames), dtype=np.float32)
    traces[0, 3:7] = np.asarray([1, 2, 2, 1], dtype=np.float32)
    traces[1, 6:10] = np.asarray([1, 2, 2, 1], dtype=np.float32)
    injected = np.einsum("st,shw->thw", traces, footprints).astype(np.float32)
    rows, columns = np.mgrid[:height, :width]
    background = np.broadcast_to((rows + columns)[None] / 50, (frames, height, width)).copy()
    return SemiSyntheticFixture(
        fixture_id=identifier,
        observation=(background + injected).astype(np.float32),
        native_background=background.astype(np.float32),
        injected_neural_signal=injected,
        footprints=footprints,
        traces=traces,
        metadata={"native_background_is_not_decomposed_truth": True},
    )


def _temporal_range(movie: np.ndarray) -> np.ndarray:
    return np.max(movie, axis=0) - np.min(movie, axis=0)


def test_footprint_centers_and_deterministic_maxima() -> None:
    fixture = _fixture()
    assert footprint_centers_yx(fixture.footprints) == ((5.0, 6.0), (10.0, 11.0))
    candidates = local_maxima_yx(_temporal_range(fixture.observation), budget=2)
    assert [(row, column) for row, column, _ in candidates] == [(5, 6), (10, 11)]
    recovery = match_injected_sources(candidates, footprint_centers_yx(fixture.footprints), radius_px=1)
    assert recovery.recall == 1
    assert recovery.unmatched_candidate_count == 0
    assert recovery.matched_source_indices == (0, 1)


def test_plateau_maxima_collapse_and_minimum_separation_is_enforced() -> None:
    score = np.zeros((16, 16), dtype=np.float64)
    score[4:8, 4:8] = 5.0
    score[12, 12] = 4.0
    candidates = local_maxima_yx(score, budget=4, minimum_distance_px=3)
    assert candidates[:2] == ((5, 5, 5.0), (12, 12, 4.0))
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            assert np.hypot(first[0] - second[0], first[1] - second[1]) >= 3


def test_radius_constrained_matching_maximizes_valid_cardinality_first() -> None:
    # Raw-distance Hungarian assignment prefers (10 + 31.8) and would drop
    # its second edge after matching.  The crossed assignment (30 + 29.2)
    # contains two valid <=30-pixel matches and therefore must win.
    candidates = ((0, 10, 2.0), (23, 18, 1.0))
    sources = ((0.0, 0.0), (0.0, 40.0))
    recovery = match_injected_sources(candidates, sources, radius_px=30.0)
    assert recovery.recovered_sources == 2
    assert recovery.unmatched_candidate_count == 0


def test_paired_fixture_reports_truth_bounded_recovery() -> None:
    result = evaluate_paired_fixture(
        _fixture(), _temporal_range, method="temporal_range", candidate_budget=2, match_radius_px=1
    )
    assert result["source_on_recovery"]["recall"] == 1
    assert result["intervention_recovery"]["recall"] == 1
    assert result["maximum_pair_closure_absolute"] < 1e-6
    assert "unknown" in result["interpretation"]["unmatched_candidates"]
    assert "unavailable" in result["interpretation"]["precision"]
    assert result["score_pair_policy"] == "stateless_frozen_scorer"


def test_pair_aware_scorer_fits_source_off_once() -> None:
    class PairAware:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, movie: np.ndarray) -> np.ndarray:
            raise AssertionError("pair-aware path must not call the stateless scorer")

        def fit_source_off(self, source_off: np.ndarray):
            self.calls += 1
            center = np.median(source_off, axis=0)

            def frozen(movie: np.ndarray) -> np.ndarray:
                return np.max(movie, axis=0) - center

            return frozen

    scorer = PairAware()
    result = evaluate_paired_fixture(
        _fixture(), scorer, method="pair-aware", candidate_budget=2
    )
    assert scorer.calls == 1
    assert result["score_pair_policy"] == "fit_source_off_then_frozen_apply_to_both_arms"


def test_suite_order_is_stable() -> None:
    rows = evaluate_paired_suite(
        [_fixture("b"), _fixture("a")],
        {"z": _temporal_range, "a": _temporal_range},
        candidate_budget=2,
    )
    assert [(row["fixture_id"], row["method"]) for row in rows] == [
        ("a", "a"), ("a", "z"), ("b", "a"), ("b", "z")
    ]


def test_invalid_closure_and_score_shape_fail_closed() -> None:
    fixture = _fixture()
    broken = SemiSyntheticFixture(
        fixture_id=fixture.fixture_id,
        observation=fixture.observation + 1,
        native_background=fixture.native_background,
        injected_neural_signal=fixture.injected_neural_signal,
        footprints=fixture.footprints,
        traces=fixture.traces,
        metadata=fixture.metadata,
    )
    with pytest.raises(InjectionEvaluationError, match="correctly rounded"):
        evaluate_paired_fixture(broken, _temporal_range, method="x", candidate_budget=2)
    with pytest.raises(InjectionEvaluationError, match="must return finite map"):
        evaluate_paired_fixture(fixture, lambda movie: movie[:, 0], method="x", candidate_budget=2)


def test_paired_sign_flip_is_exact_for_small_cluster_count() -> None:
    result = paired_sign_flip_test([1.0, 1.0, 1.0, 1.0])
    assert result["exact"] is True
    assert result["permutations"] == 16
    assert result["two_sided_p_value"] == pytest.approx(0.125)


@pytest.mark.parametrize("source_count", [1, 2, 4])
def test_native_background_injection_is_exact_and_deterministic(source_count: int) -> None:
    rng = np.random.default_rng(11)
    native = rng.normal(500, 4, size=(32, 64, 64)).astype(np.float32)
    first = make_native_background_injection(
        native,
        fixture_id=f"sources-{source_count}",
        source_count=source_count,
        seed=19,
    )
    repeated = make_native_background_injection(
        native,
        fixture_id=f"sources-{source_count}",
        source_count=source_count,
        seed=19,
    )
    assert first.footprints.shape == (source_count, 64, 64)
    assert first.traces.shape == (source_count, 32)
    assert np.array_equal(first.injected_neural_signal, repeated.injected_neural_signal)
    assert first.metadata["native_background_is_not_decomposed_truth"] is True
    assert first.metadata["maximum_closure_absolute"] < 1e-4
    assert first.metadata["maximum_closure_float32_ulp"] <= 0.51
    assert first.metadata["primary_localization_target"] == "footprint_peak_yx"
    assert len(first.metadata["placement_centers_yx"]) == source_count
    assert len(first.metadata["footprint_peak_centers_yx"]) == source_count
    evaluated = evaluate_paired_fixture(
        first, _temporal_range, method="temporal-range", candidate_budget=4
    )
    assert evaluated["primary_localization_target"] == "footprint_peak_yx"
    assert (
        evaluated["placement_center_sensitivity"]["status"]
        == "reported_secondary_not_primary_matching_truth"
    )
    assert evaluated["footprint_peak_morphology_sensitivity"]["status"] == "reported"
    assert evaluated["placement_center_sensitivity"]["morphology_stratified"] is not None


def test_registered_injection_seeds_cover_overlap_neighbor_and_separated_cases() -> None:
    native = np.random.default_rng(41).normal(size=(32, 64, 64)).astype(np.float32)
    observed = {
        make_native_background_injection(
            native,
            fixture_id=f"seed-{seed}",
            source_count=2,
            seed=seed,
        ).metadata["crowding_case"]
        for seed in (3101, 3102, 3103)
    }
    assert observed == {"overlap", "near_neighbor", "separated"}


def test_fixture_hash_substream_varies_patterns_without_changing_top_level_seed() -> None:
    native = np.random.default_rng(42).normal(size=(32, 64, 64)).astype(np.float32)
    first = make_native_background_injection(
        native, fixture_id="recording-a-window-low", source_count=2, seed=3102
    )
    second = make_native_background_injection(
        native, fixture_id="recording-b-window-low", source_count=2, seed=3102
    )
    assert first.metadata["top_level_injection_seed"] == 3102
    assert second.metadata["top_level_injection_seed"] == 3102
    assert first.metadata["crowding_case"] == second.metadata["crowding_case"] == "overlap"
    assert first.metadata["fixture_pattern_seed"] != second.metadata["fixture_pattern_seed"]
    assert not np.array_equal(first.injected_neural_signal, second.injected_neural_signal)


def test_injection_amplitude_uses_exact_native_mad_and_rejects_degenerate_background() -> None:
    native = np.random.default_rng(43).normal(size=(32, 64, 64)).astype(np.float32)
    fixture = make_native_background_injection(
        native,
        fixture_id="exact-native-mad",
        source_count=1,
        seed=3101,
        amplitude_multiplier=1.0,
    )
    assert fixture.metadata["source_peak_raw_units"] == pytest.approx(
        fixture.metadata["native_difference_mad"]
    )
    assert fixture.metadata["amplitude_floor_raw_units"] is None
    with pytest.raises(InjectionEvaluationError, match="does not permit an implicit"):
        make_native_background_injection(
            np.zeros((32, 64, 64), dtype=np.float32),
            fixture_id="degenerate-native-mad",
            source_count=1,
            seed=3101,
        )


def test_hierarchical_bootstrap_respects_nested_cluster_rows() -> None:
    rows = []
    for recording in ("a", "b", "c", "d"):
        for window in ("low", "high"):
            for seed in (1, 2, 3):
                rows.append(
                    {
                        "background_recording_id": recording,
                        "background_window_id": window,
                        "injection_seed": seed,
                        "difference": 0.1,
                    }
                )
    result = hierarchical_grouped_bootstrap(
        rows, value_key="difference", draws=1_000, seed=4
    )
    assert result["observed_mean"] == pytest.approx(0.1)
    assert result["confidence_interval_95"] == pytest.approx([0.1, 0.1])
    assert result["recording_count"] == 4
    assert result["cluster_count"] == 24


def test_strongest_comparator_is_reselected_inside_grouped_bootstrap() -> None:
    rows = [
        {
            "background_recording_id": "a",
            "background_window_id": "a1",
            "injection_seed": 1,
            "method_values": {"jepa": 0.8, "mae": 0.7, "random": 0.4},
        },
        {
            "background_recording_id": "b",
            "background_window_id": "b1",
            "injection_seed": 1,
            "method_values": {"jepa": 0.8, "mae": 0.5, "random": 0.6},
        },
    ]
    result = hierarchical_strongest_comparator_bootstrap(
        rows,
        primary_method="jepa",
        comparator_methods=("mae", "random"),
        draws=1_000,
        seed=17,
    )
    assert result["observed_strongest_comparator"] == "mae"
    assert result["observed_mean"] == pytest.approx(0.2)
    assert result["strongest_comparator_reselected_inside_each_draw"] is True
    assert sum(result["bootstrap_strongest_selection_fraction"].values()) == pytest.approx(1)
