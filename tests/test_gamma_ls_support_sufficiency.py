from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurobench.algorithms.cfar import _cfar_numpy
from neurobench.experiments.gamma_ls_difference.grid import (
    SCREEN_REPRESENTATIONS,
    TRAINING_FOLDS,
)
from neurobench.experiments.gamma_ls_difference.support_config import (
    GammaSupportConfig,
    GammaSupportConfigError,
)
from neurobench.experiments.gamma_ls_difference.support_grid import (
    CONTROLS,
    GammaSupportGridError,
    best_context_by_fold_and_width,
    design_counts,
    enumerate_stage_contexts,
    stage_b_required,
    support_boundary_status,
)
from neurobench.experiments.gamma_ls_difference import support_sufficiency as support


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/spon_ca_burst_gamma_ls_support_sufficiency_v1.example.json"


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> GammaSupportConfig:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    return GammaSupportConfig.load(EXAMPLE)


def _metric_rows(contexts, scores=None):
    rows = []
    scores = scores or {}
    for fold in TRAINING_FOLDS:
        for context in contexts:
            contrast = float(scores.get((fold, context.context_id), context.half_width_px / 1000))
            for index, representation in enumerate(SCREEN_REPRESENTATIONS):
                quiet = 2.0 + fold / 100
                rows.append(
                    {
                        "stage": context.stage,
                        "context_id": context.context_id,
                        "representation": representation,
                        "training_fold": fold,
                        "event_positive_tail": quiet + contrast + index / 10000,
                        "quiet_positive_tail": quiet,
                        "runtime_ms_per_frame": context.half_width_px / 100,
                        "positive_coordinates_used": False,
                        "positive_identities_used": False,
                        "burst_windows_used": True,
                    }
                )
    return rows


def test_manifest_freezes_adaptive_support_and_control_design(config) -> None:
    assert config.payload["stopping_rule"]["stage_b_max_half_width_px"] == 47
    assert config.payload["stopping_rule"]["final_claim_requires"] == [
        "training_window_contrast",
        "protected_recall_sensitivity",
        "repeated_latency",
    ]
    assert config.payload["scientific_audit"]["enabled"] is True


def test_manifest_rejects_unplanned_width(tmp_path: Path, monkeypatch) -> None:
    payload = json.loads(EXAMPLE.read_text())
    payload["design"]["stage_a_radius_guard_pairs"][-1] = [35, 11]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    with pytest.raises(GammaSupportConfigError, match="pairs changed"):
        GammaSupportConfig.load(path)


def test_context_counts_and_geometry_are_exact() -> None:
    stage_a = enumerate_stage_contexts("support_a")
    stage_b = enumerate_stage_contexts("support_b")
    assert len(stage_a) == 81
    assert len(stage_b) == 36
    assert {row.half_width_px for row in stage_a} == {11, 15, 19, 23, 31}
    assert {row.half_width_px for row in stage_b} == {39, 47}
    assert all(row.support == "radial_disk" and row.eligible_primary for row in stage_a + stage_b)
    assert design_counts(stage_b_run=False)["eligible_fold_metric_cells"] == 972
    assert design_counts(stage_b_run=True)["eligible_fold_metric_cells"] == 1404
    assert design_counts(stage_b_run=True)["diagnostic_control_fold_metric_cells"] == 36


def test_controls_are_named_executable_and_structurally_ineligible() -> None:
    assert [row.control_id for row in CONTROLS] == [
        "legacy_exact_n9_mode35_w23_eps64",
        "signed_square_annulus_ls_h11_g3",
        "maintained_positive_box_cfar_h11_g3",
    ]
    assert all(row.eligible_primary is False for row in CONTROLS)
    contexts = enumerate_stage_contexts("support_a")
    rows = _metric_rows(contexts)
    rows[0] = {**rows[0], "context_id": CONTROLS[0].control_id}
    with pytest.raises(GammaSupportGridError, match="noneligible or unknown"):
        best_context_by_fold_and_width(rows, contexts)


def test_fold_width_selection_is_complete_and_fold_local() -> None:
    contexts = enumerate_stage_contexts("support_a")
    rows = _metric_rows(contexts)
    best = best_context_by_fold_and_width(rows, contexts)
    assert set(best) == set(TRAINING_FOLDS)
    assert all(set(best[fold]) == {11, 15, 19, 23, 31} for fold in TRAINING_FOLDS)
    # Synthetic score rises with width, so the best context within h31 is stable-id first.
    assert best[1][31]["context"].context_id.startswith("support_support_a_h31_")


def test_selector_rejects_protected_fields() -> None:
    contexts = enumerate_stage_contexts("support_a")
    rows = _metric_rows(contexts)
    rows[0]["known_positive_recall"] = 1.0
    with pytest.raises(GammaSupportGridError, match="forbidden protected fields"):
        best_context_by_fold_and_width(rows, contexts)


def test_h31_near_optimal_triggers_stage_b_and_h47_can_leave_boundary_open() -> None:
    contexts_a = enumerate_stage_contexts("support_a")
    best_a = best_context_by_fold_and_width(_metric_rows(contexts_a), contexts_a)
    trigger, rows = stage_b_required(
        best_a, absolute_tolerance=0.005, relative_tolerance=0.05
    )
    assert trigger is True
    assert all(row["h31_is_best_or_near_optimal"] for row in rows)

    contexts_b = enumerate_stage_contexts("support_b")
    best_b = best_context_by_fold_and_width(_metric_rows(contexts_b), contexts_b)
    combined = {fold: {**best_a[fold], **best_b[fold]} for fold in TRAINING_FOLDS}
    boundary = support_boundary_status(
        combined,
        stage_b_run=True,
        absolute_tolerance=0.005,
        relative_tolerance=0.05,
    )
    assert boundary["status"] == "unresolved_at_predeclared_maximum"
    assert boundary["largest_tested_half_width_px"] == 47


def test_clear_h31_inferiority_stops_without_stage_b() -> None:
    contexts = enumerate_stage_contexts("support_a")
    scores = {}
    for fold in TRAINING_FOLDS:
        for context in contexts:
            scores[(fold, context.context_id)] = 0.10 if context.half_width_px == 23 else 0.01
            if context.half_width_px == 31:
                scores[(fold, context.context_id)] = 0.01
    best = best_context_by_fold_and_width(_metric_rows(contexts, scores), contexts)
    trigger, _ = stage_b_required(
        best, absolute_tolerance=0.005, relative_tolerance=0.05
    )
    assert trigger is False
    boundary = support_boundary_status(
        best,
        stage_b_run=False,
        absolute_tolerance=0.005,
        relative_tolerance=0.05,
    )
    assert boundary["status"] == "closed_by_clear_endpoint_inferiority"


def test_signed_square_annulus_preserves_constant_field_at_borders() -> None:
    values = torch.full((2, 29, 31), 7.0)
    mean, std = support._square_annulus_moments(values)
    torch.testing.assert_close(mean, values)
    torch.testing.assert_close(std, torch.zeros_like(std), atol=1e-6, rtol=0)


def test_gpu_native_positive_box_control_matches_maintained_numpy_semantics() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(2, 31, 33)).astype(np.float32)
    actual = support._maintained_positive_box_score(torch.from_numpy(values)).numpy()
    expected = _cfar_numpy(
        values,
        pfa=0.001,
        guard_px=3,
        training_radius_px=11,
        epsilon=1e-6,
    )["score"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_mixed_radial_and_control_latency_rows_have_one_atomic_tsv_schema(
    config, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        support,
        "_benchmark_operation",
        lambda device, operation, warmups, iterations: (
            {
                "p50_ms": 0.20,
                "p95_ms": 0.24,
                "p99_ms": 0.25,
                "max_ms": 0.26,
                "mean_ms": 0.21,
            },
            4096,
        ),
    )
    common = torch.zeros((3, 29, 31), dtype=torch.float32)
    rows = support._repeated_latency(
        common,
        contexts=enumerate_stage_contexts("support_a"),
        config=config,
    )
    assert len(rows) == 8  # five unique radial widths plus three controls
    assert all(list(row) == list(rows[0]) for row in rows)
    assert sum(bool(row["eligible_primary"]) for row in rows) == 5
    assert all("p50_ratio_vs_h11" in row for row in rows)
    path = tmp_path / "repeated_single_frame_latency.tsv"
    support._atomic_tsv(path, rows)
    assert path.is_file()
    assert not list(tmp_path.glob("*.partial"))


def test_run_fails_before_output_mutation_when_preflight_fails(config, tmp_path, monkeypatch) -> None:
    output = tmp_path / "support"

    def blocked(*args, **kwargs):
        raise ValueError("fixture stale preflight")

    monkeypatch.setattr(support, "_verify_support_preflight", blocked)
    with pytest.raises(ValueError, match="fixture stale preflight"):
        support.run_support_screen(
            config,
            base_preflight_dir=tmp_path / "base",
            support_preflight_dir=tmp_path / "support-preflight",
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".support.partial-*"))
