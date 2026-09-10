from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from neurobench.experiments.gamma_ls_difference.grid import (
    EXPECTED_G1_EVALUATIONS,
    MAX_G2_EVALUATIONS,
    GammaGridError,
    RadiusGuardPair,
    SCREEN_REPRESENTATIONS,
    TRAINING_FOLDS,
    assert_screen_design_counts,
    enumerate_g1_contexts,
    enumerate_g2_contexts,
    legacy_exact_diagnostic,
    select_g1_radius_guard_pairs,
    select_g2_common_finalist,
    select_successive_halving,
    square_box_control_metadata,
)


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json"


@pytest.fixture
def manifest() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _rows(contexts, *, scores, stage: str, fold: int = 1):
    rows = []
    for context in contexts:
        contrast, runtime = scores[context.context_id]
        for representation_index, representation in enumerate(SCREEN_REPRESENTATIONS):
            quiet_tail = 2.0 + 0.01 * fold
            rows.append(
                {
                    "stage": stage,
                    "context_id": context.context_id,
                    "representation": representation,
                    "training_fold": fold,
                    "event_positive_tail": (
                        quiet_tail + contrast + 0.001 * representation_index
                    ),
                    "quiet_positive_tail": quiet_tail,
                    "runtime_ms_per_frame": runtime,
                    "positive_coordinates_used": False,
                    "positive_identities_used": False,
                    "burst_windows_used": True,
                }
            )
    return rows


def test_g1_is_exactly_nine_guarded_radial_contexts_with_stable_ids(manifest) -> None:
    contexts = enumerate_g1_contexts(manifest)

    assert len(contexts) == 9
    assert contexts[0].context_id == "gamma_h7_g1_n5_m0p75"
    assert contexts[-1].context_id == "gamma_h15_g5_n5_m0p75"
    assert {context.half_width_px for context in contexts} == {7, 11, 15}
    assert {context.guard_radius_px for context in contexts} == {1, 3, 5}
    assert all("scale_floor" not in context.as_dict() for context in contexts)
    assert {context.support for context in contexts} == {"radial_disk"}
    assert all(context.guard_radius_px > 0 and context.eligible_primary for context in contexts)


def test_scale_floor_is_not_embedded_in_shared_geometry(manifest) -> None:
    contexts = enumerate_g1_contexts(manifest)
    assert all(not hasattr(context, "scale_floor") for context in contexts)


def test_g2_is_dynamic_but_bounded_to_two_retained_pairs(manifest) -> None:
    pairs = (RadiusGuardPair(7, 1), RadiusGuardPair(15, 5))
    contexts = enumerate_g2_contexts(manifest, pairs)

    assert len(contexts) == 18
    assert {context.radius_guard_pair for context in contexts} == set(pairs)
    assert {context.shape for context in contexts} == {2.0, 5.0, 9.0}
    assert {context.mode_fraction_of_half_width for context in contexts} == {
        0.5,
        0.75,
        1.0,
    }
    assert "gamma_h7_g1_n2_m0p5" in {context.context_id for context in contexts}
    with pytest.raises(GammaGridError, match="exactly two distinct"):
        enumerate_g2_contexts(manifest, (pairs[0], pairs[0]))
    with pytest.raises(GammaGridError, match="frozen G1"):
        enumerate_g2_contexts(
            manifest,
            (pairs[0], RadiusGuardPair(13, 2)),
        )


def test_diagnostics_are_separate_and_never_primary(manifest) -> None:
    g1_ids = {
        context.context_id
        for context in enumerate_g1_contexts(manifest)
    }
    legacy = legacy_exact_diagnostic(manifest)
    square = square_box_control_metadata(manifest)

    assert not legacy.eligible_primary
    assert not square.eligible_primary
    assert legacy.control_id not in g1_ids
    assert square.control_id not in g1_ids
    assert legacy.metadata["padding"] == "reflect"
    assert square.metadata["role"] == "named_diagnostic_control_only"


def test_design_count_assertions_are_108_and_at_most_216(manifest) -> None:
    g1 = enumerate_g1_contexts(manifest)
    g2 = enumerate_g2_contexts(
        manifest,
        (RadiusGuardPair(7, 1), RadiusGuardPair(11, 3)),
    )
    counts = assert_screen_design_counts(g1, g2)

    assert counts["g1_evaluations"] == EXPECTED_G1_EVALUATIONS == 108
    assert counts["g2_evaluations"] == MAX_G2_EVALUATIONS == 216
    assert counts["g2_max_evaluations"] == 216


def test_successive_halving_is_deterministic_and_fold_local(manifest) -> None:
    g1 = enumerate_g1_contexts(manifest)
    g1_scores = {
        context.context_id: (1.0, 4.0 + index)
        for index, context in enumerate(g1)
    }
    first_id = "gamma_h11_g3_n5_m0p75"
    second_id = "gamma_h15_g5_n5_m0p75"
    g1_scores[first_id] = (8.0, 1.0)
    g1_scores[second_id] = (7.0, 0.5)
    # This context is strictly dominated by first_id despite tying its contrast.
    g1_scores["gamma_h7_g1_n5_m0p75"] = (8.0, 2.0)
    g1_rows = _rows(g1, scores=g1_scores, stage="g1")

    selected_g1 = select_g1_radius_guard_pairs(g1_rows, g1, training_fold=1)
    assert selected_g1.retained_context_ids == (first_id, second_id)
    assert selected_g1.retained_pairs == (
        RadiusGuardPair(11, 3),
        RadiusGuardPair(15, 5),
    )

    g2 = enumerate_g2_contexts(manifest, selected_g1.retained_pairs)
    g2_scores = {
        context.context_id: (2.0, 3.0 + index / 10)
        for index, context in enumerate(g2)
    }
    winner_id = "gamma_h15_g5_n9_m1"
    g2_scores[winner_id] = (9.0, 0.75)
    g2_rows = _rows(g2, scores=g2_scores, stage="g2")

    shuffled_g1 = list(g1_rows)
    shuffled_g2 = list(g2_rows)
    random.Random(20260908).shuffle(shuffled_g1)
    random.Random(20260909).shuffle(shuffled_g2)
    result = select_successive_halving(
        manifest,
        g1_rows=shuffled_g1,
        g2_rows=shuffled_g2,
        training_fold=1,
    )
    assert result.g1.retained_context_ids == (first_id, second_id)
    assert result.g2.finalist_context_id == winner_id
    assert result.g2.ranked_contexts[0].cell_count == 3


def test_runtime_pareto_layer_and_stable_id_break_exact_ties(manifest) -> None:
    contexts = enumerate_g1_contexts(manifest)
    scores = {context.context_id: (1.0, 10.0) for context in contexts}
    fast_id = "gamma_h15_g5_n5_m0p75"
    scores[fast_id] = (1.0, 0.25)
    ranked = select_g1_radius_guard_pairs(
        _rows(contexts, scores=scores, stage="g1"), contexts, training_fold=1
    ).ranked_contexts

    assert ranked[0].context_id == fast_id
    assert ranked[0].pareto_layer == 0
    tied_slow_ids = [row.context_id for row in ranked if row.mean_runtime_ms_per_frame == 10.0]
    assert tied_slow_ids == sorted(tied_slow_ids)


def test_public_selection_rejects_across_fold_pooling(manifest) -> None:
    contexts = enumerate_g1_contexts(manifest)
    scores = {context.context_id: (1.0, 1.0) for context in contexts}
    pooled = []
    for fold in TRAINING_FOLDS:
        pooled.extend(_rows(contexts, scores=scores, stage="g1", fold=fold))
    with pytest.raises(GammaGridError, match="only the requested outer training fold"):
        select_g1_radius_guard_pairs(pooled, contexts, training_fold=1)


def test_incomplete_or_sparse_positive_rows_fail_closed(manifest) -> None:
    contexts = enumerate_g1_contexts(manifest)
    scores = {context.context_id: (1.0, 1.0) for context in contexts}
    rows = _rows(contexts, scores=scores, stage="g1")

    with pytest.raises(GammaGridError, match="incomplete or unbalanced"):
        select_g1_radius_guard_pairs(rows[:-1], contexts, training_fold=1)

    contaminated = [dict(row) for row in rows]
    contaminated[0]["positive_coordinates"] = [[12, 34]]
    with pytest.raises(GammaGridError, match="unknown=.*positive_coordinates"):
        select_g1_radius_guard_pairs(contaminated, contexts, training_fold=1)

    coordinate_flagged = [dict(row) for row in rows]
    coordinate_flagged[0]["positive_coordinates_used"] = True
    with pytest.raises(GammaGridError, match="positive coordinates"):
        select_g1_radius_guard_pairs(coordinate_flagged, contexts, training_fold=1)

    wrong_representation = [dict(row) for row in rows]
    wrong_representation[0]["representation"] = "cs_parzen_two_frame"
    with pytest.raises(GammaGridError, match="accepts only raw"):
        select_g1_radius_guard_pairs(wrong_representation, contexts, training_fold=1)


def test_g2_selection_requires_a_complete_common_context_screen(manifest) -> None:
    contexts = enumerate_g2_contexts(
        manifest,
        (RadiusGuardPair(7, 1), RadiusGuardPair(11, 3)),
    )
    scores = {context.context_id: (1.0, 1.0) for context in contexts}
    rows = _rows(contexts, scores=scores, stage="g2")
    selected = select_g2_common_finalist(rows, contexts, training_fold=1)
    assert selected.finalist_context_id == min(context.context_id for context in contexts)
