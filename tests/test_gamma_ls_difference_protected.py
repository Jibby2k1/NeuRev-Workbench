from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.protected import (
    _relay_progress_heartbeat,
    aggregate_match_rows,
    clustered_bootstrap_contrasts,
    context_lane_from_id,
    load_fold_context_plan,
    observation_match_rows,
    pair_eligible_current_frames,
    select_fold_ica_fit,
    select_fold_pca_fit,
)
from neurobench.experiments.gamma_ls_difference.screen import FoldContract


def _folds() -> tuple[FoldContract, ...]:
    return tuple(
        FoldContract(
            training_fold=fold,
            heldout_burst=str(fold),
            training_bursts=tuple(str(item) for item in range(1, 5) if item != fold),
            heldout_guard_ui=(100 * fold, 100 * fold + 20),
            quiet_half_a_ui=(1, 50),
            quiet_half_b_ui=(51, 100),
        )
        for fold in range(1, 5)
    )


def test_progress_heartbeat_nests_child_stage_without_key_collision() -> None:
    calls = []

    def heartbeat(stage: str, **details: object) -> None:
        calls.append((stage, details))

    child = {"stage": "causal_history", "processed_chunks": 3, "status": "active"}
    _relay_progress_heartbeat(heartbeat, "causal_preprocessing", child)
    assert calls == [
        (
            "causal_preprocessing",
            {"upstream_progress": child},
        )
    ]


def test_context_identifier_reconstructs_exact_radial_geometry() -> None:
    lane = context_lane_from_id(
        "gamma_h23_g5_n9_m0p75",
        role="size_sufficient_candidate",
        selection_source="fixture.json",
    )
    assert lane.half_width_px == 23
    assert lane.guard_radius_px == 5
    assert lane.shape == 9.0
    assert lane.mode_fraction_of_half_width == 0.75
    assert lane.mode_radius_px == 17.25
    assert lane.spec().support_width_px == 47
    assert lane.spec().support_geometry == "disk"


def test_pair_guard_excludes_pairs_when_either_frame_touches_guard() -> None:
    frame_ui = np.arange(95, 126)
    eligible = pair_eligible_current_frames(frame_ui, (100, 120))
    retained = frame_ui[eligible]
    assert 99 in retained
    assert 100 not in retained
    assert 120 not in retained
    # Current 121 is outside, but its previous frame 120 touches the guard.
    assert 121 not in retained
    assert 122 in retained


def test_original_screen_context_plan_is_exactly_fold_local(tmp_path: Path) -> None:
    rows = []
    for fold in _folds():
        context_id = "gamma_h11_g5_n9_m1"
        rows.append(
            {
                "training_fold": fold.training_fold,
                "heldout_burst": fold.heldout_burst,
                "common_g2_finalist_context_id": context_id,
                "g2_ranked_contexts": [{"context_id": context_id}],
            }
        )
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "across_fold_pooling_used_for_protected_selection": False,
                "folds": rows,
            }
        ),
        encoding="utf-8",
    )
    plan, provenance = load_fold_context_plan(tmp_path, _folds())
    assert set(plan) == {1, 2, 3, 4}
    assert all(lanes[0].role == "original_screen_primary" for lanes in plan.values())
    assert provenance["positive_coordinates_used"] is False


def test_size_extension_context_plan_deduplicates_equal_training_best(
    tmp_path: Path,
) -> None:
    def context(context_id: str) -> dict[str, object]:
        lane = context_lane_from_id(
            context_id, role="fixture", selection_source="fixture"
        )
        return lane.as_dict()

    rows = []
    for fold in _folds():
        rows.append(
            {
                "training_fold": fold.training_fold,
                "heldout_burst": fold.heldout_burst,
                "original_screen_context": context("gamma_h11_g5_n9_m1"),
                "support_candidate_context": context("gamma_h19_g5_n9_m1"),
                "larger_support_comparator": context("gamma_h31_g5_n9_m1"),
                "training_best_context": context("gamma_h31_g5_n9_m1"),
            }
        )
    (tmp_path / "fold_contexts.json").write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
                "folds": rows,
            }
        ),
        encoding="utf-8",
    )
    plan, _ = load_fold_context_plan(tmp_path, _folds())
    assert [lane.role for lane in plan[1]] == [
        "original_screen_context",
        "size_sufficient_candidate",
        "larger_support_comparator",
    ]


def test_size_extension_accepts_explicit_support_ids_and_keeps_equal_roles(
    tmp_path: Path,
) -> None:
    original = {
        "context_id": "gamma_h11_g5_n9_m1",
        "half_width_px": 11,
        "guard_radius_px": 5,
        "shape": 9.0,
        "mode_fraction_of_half_width": 1.0,
        "mode_radius_px": 11.0,
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
        "eligible_primary": True,
    }
    candidate = {
        **original,
        "context_id": "support_support_a_h11_g5_n9_m1",
    }
    comparator = {
        **original,
        "context_id": "support_support_a_h31_g11_n9_m1",
        "half_width_px": 31,
        "guard_radius_px": 11,
        "mode_radius_px": 31.0,
    }
    rows = [
        {
            "training_fold": fold.training_fold,
            "heldout_burst": fold.heldout_burst,
            "original_screen_context": original,
            "support_candidate_context": candidate,
            "larger_support_comparator": comparator,
            "training_best_context": comparator,
        }
        for fold in _folds()
    ]
    (tmp_path / "fold_contexts.json").write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
                "folds": rows,
            }
        ),
        encoding="utf-8",
    )
    plan, _ = load_fold_context_plan(tmp_path, _folds())
    assert [lane.role for lane in plan[1]] == [
        "original_screen_context",
        "size_sufficient_candidate",
        "larger_support_comparator",
    ]
    assert plan[1][1].context_id == "support_support_a_h11_g5_n9_m1"


def test_fit_selection_requires_exact_three_by_three_label_free_grid() -> None:
    rows = []
    bandwidths = (0.25, 0.35, 0.5)
    seeds = (1, 2, 3)
    for bandwidth in bandwidths:
        for seed in seeds:
            rows.append(
                {
                    "training_fold": 1,
                    "context_role": "primary",
                    "bandwidth": bandwidth,
                    "sample_seed": seed,
                    "sample_identity_sha256": f"seed-{seed}",
                    "pca_mean_positive_tail_contrast": seed / 100,
                    "pca_minimum_quiet_swap_positive_tail_contrast": seed / 200,
                    "ica_mean_positive_tail_contrast": bandwidth + seed / 100,
                    "ica_minimum_quiet_swap_positive_tail_contrast": bandwidth,
                    "positive_coordinates_used": False,
                    "positive_identities_used": False,
                }
            )
    winner = select_fold_ica_fit(
        rows,
        fold=1,
        context_role="primary",
        bandwidths=bandwidths,
        seeds=seeds,
    )
    assert winner["bandwidth"] == 0.5
    assert winner["sample_seed"] == 3
    pca_winner = select_fold_pca_fit(
        rows,
        fold=1,
        context_role="primary",
        bandwidths=bandwidths,
        seeds=seeds,
    )
    assert pca_winner["sample_seed"] == 3
    assert pca_winner["bandwidth"] == 0.25
    with pytest.raises(ValueError, match="exact 3x3"):
        select_fold_ica_fit(
            rows[:-1],
            fold=1,
            context_role="primary",
            bandwidths=bandwidths,
            seeds=seeds,
        )
    rows[0]["positive_coordinates_used"] = True
    with pytest.raises(ValueError, match="positive coordinates"):
        select_fold_ica_fit(
            rows,
            fold=1,
            context_role="primary",
            bandwidths=bandwidths,
            seeds=seeds,
        )


def test_empty_candidate_group_remains_an_explicit_miss() -> None:
    candidates = [
        {
            "context_role": "primary",
            "representation": "raw",
            "quiet_swap": "a_train_b_test",
            "nms_distance_px": 6,
            "target_nms_peaks_per_pseudo_burst": 1.0,
            "burst_id": 1,
            "candidate_rank": 1,
            "occupancy_score": 0.8,
            "x_px": 3,
            "y_px": 4,
        }
    ]
    operating = [
        {
            "context_role": "primary",
            "representation": "raw",
            "quiet_swap": "a_train_b_test",
            "nms_distance_px": 6,
            "target_nms_peaks_per_pseudo_burst": 1.0,
        }
    ]
    positives = [
        {
            "observation_id": f"b{burst}",
            "canonical_roi_id": f"roi{burst}",
            "burst_id": burst,
            "x_px": 3.0,
            "y_px": 4.0,
        }
        for burst in range(1, 5)
    ]
    rows = observation_match_rows(
        candidates,
        positives,
        cohort="fixture",
        operating_rows=operating,
    )
    assert len(rows) == 4 * 5
    b1_b20 = next(
        row for row in rows if row["burst_id"] == 1 and row["candidate_budget"] == 20
    )
    b2_b20 = next(
        row for row in rows if row["burst_id"] == 2 and row["candidate_budget"] == 20
    )
    assert b1_b20["matched"] is True
    assert b2_b20["matched"] is False
    aggregate = aggregate_match_rows(rows)
    assert len(aggregate) == 4 * 5


def test_cluster_bootstrap_uses_26_identities_and_adds_crossfit_average() -> None:
    rows = []
    arms = ("cs_parzen_two_frame", "difference_signed")
    for identity_index in range(26):
        burst = identity_index % 4 + 1
        for swap in ("a_train_b_test", "b_train_a_test"):
            for arm in arms:
                for budget in (20, 40, 58, 80, 100):
                    rows.append(
                        {
                            "context_role": "primary",
                            "quiet_swap": swap,
                            "nms_distance_px": 6,
                            "target_nms_peaks_per_pseudo_burst": 1.0,
                            "representation": arm,
                            "candidate_budget": budget,
                            "burst_id": burst,
                            "canonical_roi_id": f"roi_{identity_index:02d}",
                            "matched": arm == "cs_parzen_two_frame",
                        }
                    )
    contrasts = clustered_bootstrap_contrasts(
        rows,
        controls=("difference_signed",),
        seed=9,
        replicates=50,
    )
    assert {row["quiet_swap"] for row in contrasts} == {
        "a_train_b_test",
        "b_train_a_test",
        "crossfit_average",
    }
    assert all(row["cluster_count"] == 26 for row in contrasts)
    assert all(row["budget_auc_delta"] == 1.0 for row in contrasts)
