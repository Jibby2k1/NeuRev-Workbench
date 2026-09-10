from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from neurobench.research.registry import _legacy_story, _paper_story, compile_registry


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "NREV-EXP-0021"
FAILED_RUN_ID = "NREV-RUN-EXP-0021-SCREEN-20260830-A"
B_RUN_ID = "NREV-RUN-EXP-0021-SCREEN-20260830-B"
FAILED_PROVENANCE = ROOT / "research" / "run-provenance" / FAILED_RUN_ID
B_PROVENANCE = ROOT / "research" / "run-provenance" / B_RUN_ID


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v1_1_run_b_binding_preserves_v1_authorities() -> None:
    v1_config = ROOT / "examples" / "uncertainty_aware_feature_learning_v1.example.json"
    v1_1_config = ROOT / "examples" / "uncertainty_aware_feature_learning_v1_1.example.json"
    v1_protocol = ROOT / "docs" / "workflows" / "uncertainty_aware_feature_learning_v1.md"
    v1_1_protocol = ROOT / "docs" / "workflows" / "uncertainty_aware_feature_learning_v1_1.md"
    runner = (
        ROOT
        / "neurobench"
        / "experiments"
        / "neuron_identifiability"
        / "uncertainty_aware_learning.py"
    )

    assert _sha(v1_config) == "49878846d1c8dff898f4648d1af5f8e33215eb3d961f678c2edd723558591b27"
    assert _sha(v1_protocol) == "a9d637e50252d82d64e30530aeb257016dbb67eb798b456d59873e13b663b7ee"
    assert _sha(v1_1_config) == "489e5f448d72fb5325f0531558d5a531eaa31ca8e7ba70362b26e9070a410063"
    assert _sha(v1_1_protocol) == "99161431d3ba27192c3ffff85b926fa2a10871e869bd3ad3edb82fb50a4379b7"
    assert _sha(runner) == "9ee07cfe0549dc357f5adcdb70415983d955bf04bd552ece4e5678cb6632cf25"

    old = _json(v1_config)
    amended = _json(v1_1_config)
    assert old["run_id"] == FAILED_RUN_ID
    assert old["output_root"].endswith(FAILED_RUN_ID)
    assert old["implementation_hash_contract"]["sources"][-1]["sha256"] == (
        "3fa813f12ee2e0d61331d6fb4ef4c1a671fb7b24fa86ed5ac16b03d853b78360"
    )
    assert amended["run_id"] == B_RUN_ID
    assert amended["output_root"].endswith(B_RUN_ID)
    assert amended["implementation_hash_contract"]["sources"][-1]["sha256"] == _sha(runner)

    resolved = _json(B_PROVENANCE / "resolved_config.json")
    assert resolved["configuration_sha256"] == _sha(v1_1_config)
    assert resolved["runner_sha256"] == _sha(runner)
    assert resolved["configuration"]["run_id"] == B_RUN_ID
    assert resolved["configuration"]["output_root"].endswith(B_RUN_ID)


def test_failed_a_and_succeeded_b_are_registered_without_scientific_promotion() -> None:
    experiment = _yaml(
        ROOT / "research" / "registry" / "experiments" / f"{EXPERIMENT_ID}.yaml"
    )
    failed = _yaml(ROOT / "research" / "registry" / "runs" / f"{FAILED_RUN_ID}.yaml")
    completed = _yaml(ROOT / "research" / "registry" / "runs" / f"{B_RUN_ID}.yaml")
    decision = _yaml(ROOT / "research" / "registry" / "decisions" / "NREV-DEC-0024.yaml")

    assert experiment["lifecycle"] == "draft"
    assert experiment["outcome"] == "not_evaluated"
    assert experiment["evidence_tier"] == "none"
    assert experiment["claim_ids"] == []
    assert experiment["run_ids"] == [FAILED_RUN_ID, B_RUN_ID]
    assert experiment["decision_ids"] == ["NREV-DEC-0024"]
    assert "evidence_capsule_id" not in experiment

    assert failed["lifecycle"] == "failed"
    assert failed["validation"]["execution"] == "failed"
    assert failed["validation"]["outputs"] == "partial"
    assert failed["failure"]["recoverable"] is False
    failed_checks = {row["id"]: row for row in failed["validation"]["checks"]}
    assert failed_checks["paired_bootstrap_assembly"]["status"] == "failed"
    assert failed_checks["primary_model_evaluation"]["status"] == "partial"

    assert completed["lifecycle"] == "succeeded"
    assert completed["validation"]["preflight"] == "passed"
    assert completed["validation"]["execution"] == "passed"
    assert completed["validation"]["outputs"] == "partial"
    completed_checks = {row["id"]: row for row in completed["validation"]["checks"]}
    assert completed_checks["nonlinear_advance_signal"]["status"] == "failed"
    assert completed_checks["robustness_sensitivities"]["status"] == "not_run"
    assert completed_checks["scientific_audit"]["status"] == "partial"
    assert completed_checks["scientific_promotion"]["status"] == "passed"

    assert decision["action"] == "hold"
    assert decision["lifecycle"] == "draft"
    assert decision["claim_ids"] == []
    assert decision["evidence_capsule_ids"] == []
    assert not (ROOT / "research" / "evidence" / f"{EXPERIMENT_ID}.json").exists()


def test_portable_run_provenance_copies_are_exact_and_sanitized() -> None:
    failed_index = _json(FAILED_PROVENANCE / "failed_provenance_index.json")
    assert _sha(FAILED_PROVENANCE / "failed_provenance_index.json") == (
        "bf5f44063c0659bb5594d8c2188e82a59f8184e18c636f1af1cf0bce61aeba95"
    )
    assert failed_index["model_metric_or_result_summary_serialized"] is False
    for row in failed_index["artifacts"]:
        artifact = FAILED_PROVENANCE / row["path"]
        assert artifact.stat().st_size == row["bytes"]
        assert _sha(artifact) == row["sha256"]

    artifact_index_path = B_PROVENANCE / "artifact_index.json"
    artifact_index = _json(artifact_index_path)
    assert _sha(artifact_index_path) == (
        "763b188dcef96e1f9168748b6ed5a2824dec337a224d4e0dc56b0aa5844ac863"
    )
    assert artifact_index["artifact_count"] == 42
    assert len(artifact_index["artifacts"]) == 42
    assert sum(row["bytes"] for row in artifact_index["artifacts"]) == 11_586_631
    indexed = {row["path"]: row for row in artifact_index["artifacts"]}
    for name in (
        "REPORT.md",
        "advance_signals.json",
        "bootstrap.json",
        "llm_context.json",
        "resolved_config.json",
        "solver_completion.json",
        "status.json",
        "summary.json",
        "validation.json",
    ):
        artifact = B_PROVENANCE / name
        assert artifact.stat().st_size == indexed[name]["bytes"]
        assert _sha(artifact) == indexed[name]["sha256"]

    portable = B_PROVENANCE / "run_provenance.json"
    provenance = _json(portable)
    assert _sha(portable) == "3894dddd67e254df0e645857452c86b56920616d4becaecfedc8960d1d04f4e2"
    assert provenance["source_execution_provenance"]["sha256"] == (
        "ee1c51001e2b9163ee98a40f236a69eb4ce803dd6963f91170634bcf257162fd"
    )
    assert provenance["source_execution_provenance"]["committed_exact_copy"] is False
    assert "/home/" not in portable.read_text(encoding="utf-8")


def test_run_b_metrics_and_nonadvance_interpretation_are_exact() -> None:
    summary = _json(B_PROVENANCE / "summary.json")
    auc = summary["primary_macro_fold_spu_auc_by_method"]
    assert auc == {
        "bagged_pu_linear": 0.9754961257664261,
        "carrier_signed": 0.9335176617048931,
        "cfar_score": 0.764660274799249,
        "elastic_linear_spu": 0.9756013669803807,
        "expert_separation_equal_weight": 0.9501444137027111,
        "linear_spu": 0.9740959139759031,
        "positive_reference": 0.8946409844767345,
        "tiny_mlp_spu": 0.9740218518068329,
    }
    recovered = {
        method: row["positive_recovered_at_budget"]
        for method, row in summary["primary_budget_58_metrics_by_method"].items()
    }
    assert recovered == {
        "bagged_pu_linear": 34,
        "carrier_signed": 29,
        "cfar_score": 14,
        "elastic_linear_spu": 33,
        "expert_separation_equal_weight": 30,
        "linear_spu": 33,
        "positive_reference": 24,
        "tiny_mlp_spu": 34,
    }
    contrasts = summary["tiny_mlp_cluster_bootstrap_contrasts"]
    assert contrasts["linear_spu"] == {
        "cluster_bootstrap_ci95_high": 0.00591243402667829,
        "cluster_bootstrap_ci95_low": -0.004607419722739924,
        "interval_interpretation": "descriptive_cluster_bootstrap_interval_non_claim_bearing",
        "resampling_unit": "whole_leakage_group_with_all_positive_and_unlabeled_members_joint",
        "tiny_mlp_minus_comparator_observed_delta": -7.406216907024366e-05,
    }
    assert contrasts["cfar_score"]["tiny_mlp_minus_comparator_observed_delta"] == 0.20936157700758384
    assert contrasts["carrier_signed"]["cluster_bootstrap_ci95_low"] == 0.006468971017181765
    assert contrasts["expert_separation_equal_weight"]["cluster_bootstrap_ci95_low"] == -0.005543024853791158
    assert summary["primary_positive_anchors"] == 78
    assert summary["primary_unlabeled_candidates"] == 1541
    assert summary["unreserved_proposal_stage_misses"] == 0
    assert summary["matched_review_reserved_positive_anchors"] == 27
    assert summary["reserved_proposal_stage_misses"] == 1
    assert summary["tiny_mlp_representative_component_null_p"] == 1 / 2001
    assert summary["tiny_mlp_seed_median_pairwise_spearman"] == 0.9942944810229658
    assert summary["tiny_mlp_seed_recall_at_58_range"] == 0.0
    assert summary["all_preregistered_advance_signals_passed"] is False
    assert summary["scientific_completion"] is False
    assert summary["claim_promotion_allowed"] is False

    advance = _json(B_PROVENANCE / "advance_signals.json")
    assert advance["all_preregistered_advance_signals_passed"] is False
    assert advance["checks"]["tiny_mlp_minus_linear_grouped_ci_lower"]["passed"] is False
    assert advance["checks"]["tiny_mlp_minus_linear_macro_fold_spu_auc"]["passed"] is False
    assert advance["checks"]["single_burst_spatial_block_and_review_policy_explanation_excluded"]["passed"] is False
    assert advance["scientific_claim_promotion_allowed"] is False


def test_migrated_v1_story_is_unchanged_while_current_story_exposes_run_history() -> None:
    payload = compile_registry()
    snapshot = _yaml(ROOT / "research" / "registry" / "migration" / "research_story_v1.yaml")
    assert _legacy_story(payload) == snapshot

    current = _paper_story(payload)
    row = next(item for item in current["experiments"] if item["id"] == EXPERIMENT_ID)
    assert row["status"] == (
        "draft_not_evaluated_engineering_run_history_failed_then_succeeded; evidence_tier_none"
    )
    assert "evidence_capsule" not in row
    assert "non-claim-bearing engineering history" in row["limitation"]
    assert "did not improve the primary estimand" in row["next_decision"]
    assert all("Outputs" not in path for path in row["artifacts"])
    assert "../../docs/research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md" in row["artifacts"]

    experiment = next(item for item in payload["experiments"] if item["id"] == EXPERIMENT_ID)
    assert not experiment.get("evidence_capsule_id")
    assert experiment["claim_ids"] == []
    assert experiment not in [
        item for item in payload["experiments"] if item.get("evidence_capsule_id")
    ]


def test_registry_index_and_results_document_capture_the_hold_boundary() -> None:
    index = _yaml(ROOT / "research" / "registry" / "index.yaml")
    assert "NREV-DEC-0024" in index["decisions"]
    assert FAILED_RUN_ID in index["runs"]
    assert B_RUN_ID in index["runs"]

    results = (
        ROOT / "docs" / "research" / "UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md"
    ).read_text(encoding="utf-8")
    prose = " ".join(results.split())
    assert "did **not** improve" in results
    assert "should not replace the simpler linear fusion layer" in results
    assert "No claim or evidence capsule is created" in prose
    assert "U means unknown, not negative" in results
    assert "not an end-to-end CFAR detector replacement test" in results

    current_state = (ROOT / "paper" / "overleaf_jnm" / "CURRENT_RESEARCH_STATE.md").read_text(
        encoding="utf-8"
    )
    current_prose = " ".join(current_state.split())
    assert "Tiny-MLP macro-fold SPU-AUC was 0.97402 versus 0.97410 for linear SPU" in current_prose
    assert "the nonlinear replacement is therefore held" in current_prose
    assert "UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md" in current_prose
    assert "separately versioned automated robustness package for the frozen" in current_prose
