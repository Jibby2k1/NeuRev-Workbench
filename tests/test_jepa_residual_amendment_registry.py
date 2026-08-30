from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "NREV-EXP-0029"
FAILED_RUN_ID = "NREV-RUN-EXP-0029-SCREEN-20260830-A"
B_RUN_ID = "NREV-RUN-EXP-0029-SCREEN-20260830-B"
FAILED_PROVENANCE = ROOT / "research" / "run-provenance" / FAILED_RUN_ID
B_PROVENANCE = ROOT / "research" / "run-provenance" / B_RUN_ID


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v1_1_is_bound_to_fresh_run_without_mutating_v1() -> None:
    config_path = ROOT / "examples" / "conditional_background_residual_v1_1.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed = _yaml(ROOT / "research" / "registry" / "runs" / f"{B_RUN_ID}.yaml")
    resolved = json.loads(
        (B_PROVENANCE / "resolved_config.json").read_text(encoding="utf-8")
    )

    assert config["schema_version"] == 2
    assert config["runner_version"] == "1.1"
    assert config["run_id"] == B_RUN_ID
    assert config["output_root"].endswith(B_RUN_ID)
    assert completed["lifecycle"] == "succeeded"
    assert completed["configuration"]["path"].endswith(
        f"{B_RUN_ID}/resolved_config.json"
    )
    assert completed["configuration"]["sha256"] == _sha(
        B_PROVENANCE / "resolved_config.json"
    )
    assert resolved["config_sha256"] == _sha(config_path)
    assert resolved["config"]["run_id"] == B_RUN_ID
    assert resolved["config"]["output_root"].endswith(B_RUN_ID)
    assert "prediction_coverage.json" in config["required_outputs"]

    arms = {row["id"]: row for row in config["arms"]}
    assert arms["raw_hc"]["movie"] == "native_observed_video_float32_parent_anchor_domain"
    assert "normalized" in arms["jepa_conditional_residual_hc"]["movie"]
    assert "normalized" in arms["random_conditional_residual_hc"]["movie"]
    anchor = config["paired_evaluation"]["raw_hc_anchor_regression"]
    assert anchor["fixture_count"] == 108
    assert anchor["required_recovery_result_count"] == 216
    assert anchor["numeric_tolerance"] == 0.0
    assert anchor["candidate_coordinate_or_count_exception_allowed"] is False

    assert _sha(ROOT / "examples" / "conditional_background_residual_v1.example.json") == (
        "91c88152e55b6c77ebdd785fca386d42820f092efe39be7a3c5db997b602a8fd"
    )
    assert _sha(ROOT / "docs" / "workflows" / "conditional_background_residual_v1.md") == (
        "be63983317172d323d65d33462625c1ca0ca4b093ee99c9211994cd3919f3a7f"
    )
    assert _sha(
        ROOT / "neurobench" / "experiments" / "neuron_identifiability" / "jepa_residual_pilot.py"
    ) == "846ec07a342effccdf149a129a0fb2381eb3a3cef8bfbfaabfae0db0e65be29d"
    assert _sha(
        ROOT
        / "neurobench"
        / "experiments"
        / "neuron_identifiability"
        / "jepa_residual_pilot_v1_1.py"
    ) == "e3b4cb4cebe155b2a46c1c8b4ff810b28ec3bedb5415b5aed1e1097f01c531d9"


def test_failed_run_is_preserved_without_endpoint_evidence_or_capsule() -> None:
    failed = _yaml(ROOT / "research" / "registry" / "runs" / f"{FAILED_RUN_ID}.yaml")
    completed = _yaml(ROOT / "research" / "registry" / "runs" / f"{B_RUN_ID}.yaml")
    experiment = _yaml(
        ROOT / "research" / "registry" / "experiments" / f"{EXPERIMENT_ID}.yaml"
    )
    decision = _yaml(ROOT / "research" / "registry" / "decisions" / "NREV-DEC-0022.yaml")

    assert failed["lifecycle"] == "failed"
    assert failed["validation"]["execution"] == "failed"
    assert failed["failure"]["recoverable"] is False
    checks = {row["id"]: row for row in failed["validation"]["checks"]}
    assert checks["raw_hc_cross_run_anchor"]["status"] == "failed"
    assert checks["residual_endpoint_evidence"]["status"] == "not_run"
    assert "216" in checks["native_domain_diagnosis"]["summary"]
    assert completed["lifecycle"] == "succeeded"
    assert completed["validation"]["preflight"] == "passed"
    assert completed["validation"]["execution"] == "passed"
    assert completed["validation"]["outputs"] == "partial"
    completed_checks = {
        row["id"]: row for row in completed["validation"]["checks"]
    }
    assert completed_checks["raw_hc_exact_parent_anchor"]["status"] == "passed"
    assert completed_checks["scientific_audit"]["status"] == "partial"
    assert completed_checks["scientific_promotion"]["status"] == "passed"

    assert experiment["lifecycle"] == "draft"
    assert experiment["outcome"] == "not_evaluated"
    assert experiment["evidence_tier"] == "none"
    assert experiment["claim_ids"] == []
    assert experiment["run_ids"] == [FAILED_RUN_ID, B_RUN_ID]
    assert decision["action"] == "hold"
    assert decision["claim_ids"] == []
    assert decision["evidence_capsule_ids"] == []
    assert not (ROOT / "research" / "evidence" / f"{EXPERIMENT_ID}.json").exists()


def test_failed_provenance_copies_and_four_mismatches_are_exact() -> None:
    index_path = FAILED_PROVENANCE / "failed_provenance_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for row in index["artifacts"]:
        path = FAILED_PROVENANCE / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha(path) == row["sha256"]

    regression = json.loads(
        (FAILED_PROVENANCE / "raw_hc_cross_run_regression.json").read_text(
            encoding="utf-8"
        )
    )
    assert regression["matched_fixture_count"] == 104
    assert regression["mismatch_count"] == 4
    assert regression["numeric_tolerance"] == 0.0
    assert regression["mismatched_fixture_ids"] == [
        "060126_10_rest__window_2_median_mad__sources_1__seed_3102",
        "060126_12_left__window_1_low_mad__sources_1__seed_3102",
        "060126_15_right__window_3_high_mad__sources_1__seed_3102",
        "spon_ca_burst_3_hindbrain_to_tail_488_20ms__window_1_low_mad__sources_1__seed_3103",
    ]

    amendment = (
        ROOT / "docs" / "workflows" / "conditional_background_residual_v1_1.md"
    ).read_text(encoding="utf-8")
    for expected in ("`2 / 1`", "`3 / 2`", "`4 / 3`", "all 216 parent"):
        assert expected in amendment
    assert "The guard is not relaxed" in amendment


def test_run_b_portable_provenance_and_observed_screen_metrics_are_exact() -> None:
    run = _yaml(ROOT / "research" / "registry" / "runs" / f"{B_RUN_ID}.yaml")
    resolved_path = B_PROVENANCE / "resolved_config.json"
    artifact_index_path = B_PROVENANCE / "artifact_index.json"

    assert _sha(resolved_path) == (
        "1ff575fd0110e4d65e275ce4ced705a3aa47b0d9a83b01d10a2270bbc865218c"
    )
    assert _sha(artifact_index_path) == (
        "34bd983ffdf55cb0941bbd1443adf939f4cb86ac4ffe51a736b2d6d9383cfc1c"
    )
    assert run["configuration"]["sha256"] == _sha(resolved_path)
    assert run["artifacts"]["manifest_sha256"] == _sha(artifact_index_path)
    assert run["started_at"] == "2026-08-30T06:18:33.910042Z"
    assert run["ended_at"] == "2026-08-30T06:19:23.403796Z"

    artifact_index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    assert len(artifact_index["artifacts"]) == 27
    assert sum(row["bytes"] for row in artifact_index["artifacts"]) == 13_600_885

    checks = {row["id"]: row for row in run["validation"]["checks"]}
    assert "0.09027777777777778" in checks["paired_recall_screen"]["summary"]
    assert "-0.24305555555555552" in checks["paired_recall_screen"]["summary"]
    assert "0.9990785812365571" in checks["residual_signal_accounting"]["summary"]
    assert "1.7102741349511534" in checks["background_suppression_screen"]["summary"]
    assert "2.5786383127702" in checks["background_suppression_screen"]["summary"]

    results = (
        ROOT / "docs" / "research" / "CONDITIONAL_BACKGROUND_RESIDUAL_V1_1_RESULTS.md"
    ).read_text(encoding="utf-8")
    assert "bounded non-scientific engineering screen" in results
    assert "no scientific claim or evidence capsule" in results


def test_registry_index_links_amendment_decision_and_b_run() -> None:
    index = _yaml(ROOT / "research" / "registry" / "index.yaml")
    assert "NREV-DEC-0022" in index["decisions"]
    assert FAILED_RUN_ID in index["runs"]
    assert B_RUN_ID in index["runs"]


def test_stable_source_off_safety_diagnostic_is_portable_and_non_claim_bearing() -> None:
    derived = B_PROVENANCE / "derived"
    safety = derived / "source_off_residual_safety_v1_1"
    summary = json.loads((safety / "summary.json").read_text(encoding="utf-8"))
    decisions = json.loads(
        (safety / "source_off_safety.json").read_text(encoding="utf-8")
    )

    assert not (derived / "source_off_residual_safety_v1").exists()
    assert _sha(safety / "artifact_index.json") == (
        "a7d3045c34e53d571c0cb277a5f8edd43b75345c94393b7efe675538fec4fe74"
    )
    assert _sha(safety / "summary.json") == (
        "1d5132a99851d59606488c48c9d15d0e40a12fe08c5df8113fb74f9841520134"
    )
    assert summary["result_sha256"] == (
        "fcc739667ba6e80056402ae88e31e4a7c766dc8a94db8a226c8d053a8c1a9a80"
    )
    assert summary["implementation_sha256"] == (
        "135144627dccb567e654211699105c23691a988dbef7df9ea784170306906274"
    )
    assert summary["safety_policy_implementation_sha256"] == (
        "047d988b6158d67f89ac195a5fd2f7433a44ca17d43b13862fa870464c0559d0"
    )
    assert summary["scientific_completion"] is False
    assert summary["scientific_promotion_allowed"] is False

    for row in summary["methods"].values():
        assert row["safe_window_count"] == 0
        assert row["window_count"] == 12
    for row in summary["policy_results"].values():
        assert row["fixture_count"] == 108
        assert row["residual_admitted_fixture_count"] == 0
        assert row["macro_source_on_recall"] == 0.1875
        assert row["micro_recovered_sources"] == 45
        assert row["micro_injected_sources"] == 252

    rows = decisions["decisions"]
    assert len(rows) == 24
    assert sum(row["checks"]["background_rms_not_amplified"] for row in rows) == 2
    assert sum(row["checks"]["dynamic_mad_not_amplified"] for row in rows) == 0
    assert sum(row["checks"]["spatial_seam_within_tolerance"] for row in rows) == 18
    assert {row["selected_endpoint"] for row in rows} == {
        "raw_frozen_handcrafted_stack"
    }


def test_stable_rank_displacement_diagnostic_is_exact_and_portable() -> None:
    rank = B_PROVENANCE / "derived" / "rank_displacement_v1"
    index = json.loads((rank / "artifact_index.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (rank / "rank_displacement_summary.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (rank / "execution_provenance.json").read_text(encoding="utf-8")
    )
    status = json.loads((rank / "status.json").read_text(encoding="utf-8"))

    assert _sha(rank / "artifact_index.json") == (
        "ca1933513b224b735882de1196a234f2d1246649211855ac81c6d4cb3c1fdaf5"
    )
    assert _sha(rank / "rank_displacement_summary.json") == (
        "d8e1201a5c2657060ec93ee998b09dc619b68c80d8b8985779c9308915e6402b"
    )
    assert summary["diagnostic_id"] == (
        "NREV-DIAG-EXP-0029-RANK-DISPLACEMENT-20260830-F"
    )
    assert summary["result_sha256"] == (
        "4cc984641e17ab9f3eb5fd0a653b0051d16fe3f449ba45666366f5372444db2d"
    )
    assert summary["inputs"]["implementation_sha256"] == (
        "8407eb2926b55304ec4d1e409e9b5049e3db08b720ba3dc046c9461ece62ddac"
    )
    assert index["artifact_count"] == 11
    assert sum(row["bytes"] for row in index["artifacts"]) == 491_492
    for row in index["artifacts"]:
        artifact = rank / row["path"]
        assert artifact.stat().st_size == row["bytes"]
        assert _sha(artifact) == row["sha256"]

    assert execution["runtime"]["scipy_version"] == "1.17.1"
    assert execution["duration_seconds"] == 42.225716
    assert execution["primary_duration_contract"] == (
        "duration_seconds equals ended_at minus started_at using timezone-aware UTC "
        "timestamp arithmetic"
    )
    assert summary["coverage"] == {
        "all_324_recovery_objects_exactly_reproduced": True,
        "fixture_count": 108,
        "method_count": 3,
        "source_count": 252,
        "source_method_rows": 756,
    }

    methods = (
        "raw_frozen_handcrafted_stack",
        "jepa_conditional_pixel_residual_frozen_handcrafted_stack",
        "random_encoder_conditional_pixel_residual_frozen_handcrafted_stack",
    )
    assert [
        summary["recovery_totals"][method]["total_source_on_recovered"]
        for method in methods
    ] == [45, 19, 18]
    assert [
        summary["classification_counts"][method]["recovered_source_on"]
        for method in methods
    ] == [43, 18, 16]
    assert [
        summary["classification_counts"][method]["source_on_only_inconsistent"]
        for method in methods
    ] == [2, 1, 2]
    assert {
        (row["method_short"], row["source_count"], row["source_row_count"])
        for row in summary["strata"]["source_count"]
    } == {
        (method, source_count, rows)
        for method in ("raw", "jepa_residual", "random_residual")
        for source_count, rows in (("1", 36), ("2", 72), ("4", 144))
    }
    assert status["scientific_completion"] is False
    assert status["claim_promotion_allowed"] is False
