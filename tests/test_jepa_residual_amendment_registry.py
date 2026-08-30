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
