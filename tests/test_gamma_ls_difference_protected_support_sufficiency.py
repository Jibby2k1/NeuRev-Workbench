from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import protected_finalize
from neurobench.experiments.gamma_ls_difference import (
    protected_support_sufficiency as support,
)


def _write_index(root: Path) -> str:
    artifacts = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": support._sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    index = root / "artifact_index.json"
    index.write_text(
        json.dumps({"schema_version": 1, "artifacts": artifacts}, sort_keys=True),
        encoding="utf-8",
    )
    return support._sha256(index)


def _observation(
    *,
    role: str,
    identity: str,
    observation: str,
    budget: int,
    matched: bool,
    swap: str = "swap_a",
) -> support.Observation:
    return support.Observation(
        cohort="protected_v1",
        context_role=role,
        representation="raw",
        quiet_swap=swap,
        nms_distance_px=6,
        nms_role="primary",
        burden=0.25,
        burst_id=1,
        candidate_budget=budget,
        effective_candidate_count=2,
        observation_id=observation,
        canonical_roi_id=identity,
        x_px="10.0",
        y_px="20.0",
        matched=matched,
        unmatched_candidates="unknown_not_negative",
    )


def test_frozen_analysis_contract_has_all_required_grid_points() -> None:
    assert support.FIXED_ARMS == (
        "raw",
        "difference_signed",
        "difference_energy_normalized",
    )
    assert support.NMS_DISTANCE_PX == 6
    assert support.QUIET_BURDENS == (0.25, 0.5, 1.0, 2.0, 5.0)
    assert support.CANDIDATE_BUDGETS == (20, 40, 58, 80, 100)
    assert support.BOOTSTRAP_SEED == 20260908
    assert support.BOOTSTRAP_REPLICATES == 2000


def test_artifact_index_is_sealed_complete_and_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.txt").write_text("sealed\n", encoding="utf-8")
    expected = _write_index(root)

    verified = support.verify_artifact_index(
        root, expected_index_sha256=expected
    )
    assert verified.index_sha256 == expected
    assert set(verified.entries) == {"a.txt"}

    (root / "a.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(
        support.ProtectedSupportSufficiencyUnavailable,
        match="hash or size changed",
    ):
        support.verify_artifact_index(root, expected_index_sha256=expected)


def test_bootstrap_weights_are_exactly_the_finalized_r2_rng_contract() -> None:
    identities = [f"roi_{index:03d}" for index in range(1, 27)]
    observed = support._bootstrap_weight_matrix(
        identities, seed=20260908, replicates=37
    )
    reference = protected_finalize._bootstrap_weight_matrix(
        identities, seed=20260908, replicates=37
    )
    np.testing.assert_array_equal(observed, reference)
    np.testing.assert_array_equal(observed.sum(axis=1), np.full(37, 26.0))


def test_pairing_rejects_role_specific_truth_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "FIXED_ARMS", ("raw",))
    monkeypatch.setattr(support, "QUIET_SWAPS", ("swap_a",))
    monkeypatch.setattr(support, "QUIET_BURDENS", (0.25,))
    monkeypatch.setattr(support, "CANDIDATE_BUDGETS", (20, 58))
    monkeypatch.setattr(support, "BURSTS", (1,))
    rows = []
    for budget in (20, 58):
        for identity in ("roi_001", "roi_002"):
            observation = identity.replace("roi", "obs")
            rows.append(
                _observation(
                    role="size_sufficient_candidate",
                    identity=identity,
                    observation=observation,
                    budget=budget,
                    matched=True,
                )
            )
            rows.append(
                _observation(
                    role="larger_support_comparator",
                    identity=identity,
                    observation=observation,
                    budget=budget,
                    matched=True,
                )
            )

    selected, universe_hash = support._analysis_rows(
        rows, expected_identities=2, expected_occurrences=2
    )
    assert len(selected) == 8
    assert len(universe_hash) == 64

    bad = list(rows)
    changed = bad[-1]
    bad[-1] = support.Observation(
        **{**changed.__dict__, "x_px": "99.0"}
    )
    with pytest.raises(
        support.ProtectedSupportSufficiencyUnavailable,
        match="same observation truth",
    ):
        support._analysis_rows(bad, expected_identities=2, expected_occurrences=2)


def test_paired_metrics_are_deterministic_and_preserve_claim_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support, "FIXED_ARMS", ("raw",))
    monkeypatch.setattr(support, "QUIET_SWAPS", ("swap_a",))
    monkeypatch.setattr(support, "QUIET_SCOPES", ("swap_a", "crossfit_average"))
    monkeypatch.setattr(support, "QUIET_BURDENS", (0.25,))
    monkeypatch.setattr(support, "CANDIDATE_BUDGETS", (20, 58))
    monkeypatch.setattr(support, "BURSTS", (1,))
    rows = []
    for budget in (20, 58):
        for identity, candidate_match in (("roi_001", True), ("roi_002", False)):
            observation = identity.replace("roi", "obs")
            rows.extend(
                [
                    _observation(
                        role="size_sufficient_candidate",
                        identity=identity,
                        observation=observation,
                        budget=budget,
                        matched=candidate_match,
                    ),
                    _observation(
                        role="larger_support_comparator",
                        identity=identity,
                        observation=observation,
                        budget=budget,
                        matched=True,
                    ),
                ]
            )

    first = support.compute_paired_metrics(
        rows,
        expected_identities=2,
        inferential=True,
        seed=19,
        replicates=101,
    )
    second = support.compute_paired_metrics(
        rows,
        expected_identities=2,
        inferential=True,
        seed=19,
        replicates=101,
    )
    assert first == second
    budget_rows, contrast_rows, burst_rows = first
    assert {row["candidate_budget"] for row in budget_rows} == {20, 58}
    assert len(contrast_rows) == 2
    assert len(burst_rows) == 2
    assert all(row["precision_identified"] is False for row in budget_rows)
    assert all(
        row["unmatched_candidates"] == "unknown_not_negative"
        for row in contrast_rows
    )
    base = next(row for row in contrast_rows if row["quiet_scope"] == "swap_a")
    assert base["candidate_minus_larger_budget_auc"] == pytest.approx(-0.5)
    assert base["candidate_minus_larger_b58_macro_recall"] == pytest.approx(-0.5)
    assert base["larger_minus_candidate_b58_matched_occurrences"] == 1
    assert base["predeclared_occurrence_tolerances_applied"] is True


def _latency_row(width: int, *, p50: float, p99: float) -> dict[str, str]:
    return {
        "family": "radial_gamma_ls",
        "context_id": f"support_support_a_h{width}_g5_n2_m0p5",
        "half_width_px": str(width),
        "support_width_px": str(2 * width + 1),
        "warmup_iterations": "50",
        "timed_iterations": "200",
        "p50_ms": str(p50),
        "p95_ms": str((p50 + p99) / 2),
        "p99_ms": str(p99),
        "max_ms": str(p99 + 0.1),
        "mean_ms": str(p50 + 0.05),
        "max_memory_allocated_bytes": str(1000 + width),
        "p50_ratio_vs_h11": "1.0",
        "p50_increase_vs_h11_ms": "0.0",
        "latency_gate_pass": "False",
        "eligible_primary": "True",
    }


def _context(width: int, latency: dict[str, str], role: str) -> dict[str, object]:
    copied: dict[str, object] = dict(latency)
    copied["half_width_px"] = int(copied["half_width_px"])
    copied["support_width_px"] = int(copied["support_width_px"])
    copied["warmup_iterations"] = int(copied["warmup_iterations"])
    copied["timed_iterations"] = int(copied["timed_iterations"])
    copied["p50_ms"] = float(copied["p50_ms"])
    copied["p95_ms"] = float(copied["p95_ms"])
    copied["p99_ms"] = float(copied["p99_ms"])
    copied["max_ms"] = float(copied["max_ms"])
    copied["mean_ms"] = float(copied["mean_ms"])
    copied["max_memory_allocated_bytes"] = int(
        copied["max_memory_allocated_bytes"]
    )
    copied["latency_gate_pass"] = False
    return {
        "context_id": f"ctx_{role}_{width}",
        "half_width_px": width,
        "repeated_latency": copied,
    }


def test_latency_tradeoff_remains_unresolved_when_gate_fails_and_cost_is_nonmonotone() -> None:
    h11 = _latency_row(11, p50=2.0, p99=2.4)
    h15 = _latency_row(15, p50=1.0, p99=1.3)
    h19 = _latency_row(19, p50=1.5, p99=1.8)
    folds = []
    for fold in range(1, 5):
        candidate_width, larger_width = ((11, 15) if fold in (1, 3) else (15, 19))
        by_width = {11: h11, 15: h15, 19: h19}
        folds.append(
            {
                "training_fold": fold,
                "heldout_burst": str(fold),
                "support_candidate_context": _context(
                    candidate_width, by_width[candidate_width], "candidate"
                ),
                "larger_support_comparator": _context(
                    larger_width, by_width[larger_width], "larger"
                ),
            }
        )

    rows, summary = support.latency_pairs(
        fold_contexts={"folds": folds}, latency_rows=[h11, h15, h19]
    )
    assert len(rows) == 4
    assert summary["candidate_latency_gate_pass_all_folds"] is False
    assert summary["larger_support_justifies_incremental_latency"] is None
    assert summary["p50_cost_directions"] == ["larger_faster", "larger_slower"]
    assert all(row["whole_pipeline_1khz_claim_allowed"] is False for row in rows)


def test_stopping_decision_rejects_v7_and_separates_validity_from_result() -> None:
    row = {
        "cohort": "protected_v1",
        "quiet_scope": "crossfit_average",
        "representation": "difference_energy_normalized",
        "target_nms_peaks_per_pseudo_burst": 2.0,
        "candidate_minus_larger_budget_auc": -0.03,
        "budget_auc_delta_ci95_low": -0.06,
        "budget_auc_delta_ci95_high": -0.005,
        "candidate_minus_larger_b58_macro_recall": -0.03,
        "b58_delta_ci95_low": -0.06,
        "b58_delta_ci95_high": -0.005,
        "larger_minus_candidate_b58_matched_occurrences": 4,
        "maximum_larger_minus_candidate_b58_matches_in_one_burst": 2,
        "nominal_auc_detectably_worse": True,
        "nominal_b58_detectably_worse": True,
        "budget_auc_loss_within_0p02": None,
        "b58_total_match_loss_within_1": None,
        "b58_each_burst_match_loss_within_1": None,
    }
    decision = support._decision(
        [row],
        boundary={
            "stage_b_triggered": False,
            "support_boundary": {"status": "closed_by_clear_endpoint_inferiority"},
        },
        latency_summary={
            "candidate_latency_gate_pass_all_folds": False,
            "larger_latency_gate_pass_all_folds": False,
            "larger_support_justifies_incremental_latency": None,
            "larger_support_latency_justification_status": "unresolved",
            "p50_cost_directions": [],
            "latency_scope": "Gamma_stage_only_not_whole_pipeline",
        },
    )
    assert decision["support_sufficient"] is False
    assert decision["protected_sensitivity_gate_pass"] is False
    assert decision["v7_used_for_selection_or_decision"] is False
    assert decision["precision_identified"] is False

    with pytest.raises(
        support.ProtectedSupportSufficiencyUnavailable,
        match="protected_v1 contrasts only",
    ):
        support._decision(
            [{**row, "cohort": "latest_v7_sensitivity"}],
            boundary={
                "stage_b_triggered": False,
                "support_boundary": {
                    "status": "closed_by_clear_endpoint_inferiority"
                },
            },
            latency_summary={
                "candidate_latency_gate_pass_all_folds": False,
                "larger_latency_gate_pass_all_folds": False,
                "larger_support_justifies_incremental_latency": None,
                "larger_support_latency_justification_status": "unresolved",
                "p50_cost_directions": [],
                "latency_scope": "Gamma_stage_only_not_whole_pipeline",
            },
        )
