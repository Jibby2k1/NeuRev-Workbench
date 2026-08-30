from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "registry"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_motion_run_e_is_the_only_registered_motion_screen() -> None:
    experiment = _yaml(REGISTRY / "experiments" / "NREV-EXP-0025.yaml")
    run = _yaml(
        REGISTRY / "runs" / "NREV-RUN-EXP-0025-SCREEN-20260830-E.yaml"
    )
    index = _yaml(REGISTRY / "index.yaml")

    assert experiment["origin"] == {"kind": "native"}
    assert experiment["lifecycle"] == "draft"
    assert experiment["outcome"] == "not_evaluated"
    assert experiment["evidence_tier"] == "none"
    assert experiment["claim_ids"] == []
    assert "evidence_capsule_id" not in experiment
    assert experiment["run_ids"] == ["NREV-RUN-EXP-0025-SCREEN-20260830-E"]

    assert run["lifecycle"] == "succeeded"
    assert run["validation"]["preflight"] == "passed"
    assert run["validation"]["execution"] == "passed"
    assert run["validation"]["outputs"] == "partial"
    checks = {row["id"]: row for row in run["validation"]["checks"]}
    assert checks["local_field_reliability"]["status"] == "partial"
    assert checks["grouped_association_family"]["status"] == "partial"
    assert checks["scientific_audit_inventory"]["status"] == "partial"

    config = ROOT / run["configuration"]["path"]
    manifest = ROOT / run["artifacts"]["manifest_path"]
    assert _sha(config) == run["configuration"]["sha256"]
    assert _sha(manifest) == run["artifacts"]["manifest_sha256"]
    assert run["code"]["diff_sha256"] == (
        "0ce57730ee8cb63da41178b6f43ee83cee2bad52a5b41c0a78a47e2026e901f3"
    )

    registered_runs = set(index["runs"])
    assert "NREV-RUN-EXP-0025-SCREEN-20260830-E" in registered_runs
    for suffix in "ABCD":
        assert f"NREV-RUN-EXP-0025-SCREEN-20260830-{suffix}" not in registered_runs


def test_predictor_run_b_records_zero_eligible_subtractors_and_media_boundary() -> None:
    experiment = _yaml(REGISTRY / "experiments" / "NREV-EXP-0030.yaml")
    run = _yaml(
        REGISTRY / "runs" / "NREV-RUN-EXP-0030-SCREEN-20260830-B.yaml"
    )

    assert experiment["origin"] == {"kind": "native"}
    assert experiment["lifecycle"] == "draft"
    assert experiment["outcome"] == "not_evaluated"
    assert experiment["evidence_tier"] == "none"
    assert experiment["claim_ids"] == []
    assert "evidence_capsule_id" not in experiment
    assert experiment["run_ids"] == ["NREV-RUN-EXP-0030-SCREEN-20260830-B"]

    checks = {row["id"]: row for row in run["validation"]["checks"]}
    assert checks["predictor_feasibility"]["status"] == "partial"
    assert "zero of seven subtractors" in checks["predictor_feasibility"][
        "summary"
    ]
    assert checks["replay_environment_provenance"]["status"] == "partial"
    replay_summary = checks["replay_environment_provenance"]["summary"]
    for dependency in ("SciPy", "Matplotlib", "Pillow", "ffmpeg/ffprobe"):
        assert dependency in replay_summary
    assert run["validation"]["outputs"] == "passed"

    config = ROOT / run["configuration"]["path"]
    manifest = ROOT / run["artifacts"]["manifest_path"]
    assert _sha(config) == run["configuration"]["sha256"]
    assert _sha(manifest) == run["artifacts"]["manifest_sha256"]
    assert run["code"]["diff_sha256"] == (
        "280243c4224546e406798dc928658e2975da8b33dfad425eed1372de13f3415e"
    )


def test_joint_hold_links_only_non_claim_bearing_experiments() -> None:
    decision = _yaml(REGISTRY / "decisions" / "NREV-DEC-0023.yaml")
    assert decision["action"] == "hold"
    assert decision["lifecycle"] == "draft"
    assert decision["review_state"] == "not_requested"
    assert decision["experiment_ids"] == [
        "NREV-EXP-0025",
        "NREV-EXP-0029",
        "NREV-EXP-0030",
    ]
    assert decision["claim_ids"] == []
    assert decision["evidence_capsule_ids"] == []

    for experiment_id in decision["experiment_ids"]:
        experiment = _yaml(REGISTRY / "experiments" / f"{experiment_id}.yaml")
        assert experiment["lifecycle"] == "draft"
        assert experiment["outcome"] == "not_evaluated"
        assert experiment["evidence_tier"] == "none"
        assert experiment["claim_ids"] == []
        assert "NREV-DEC-0023" in experiment["decision_ids"]


def test_rank_f_portable_tree_is_complete_and_exact() -> None:
    root = (
        ROOT
        / "research"
        / "run-provenance"
        / "NREV-RUN-EXP-0029-SCREEN-20260830-B"
        / "derived"
        / "rank_displacement_v1"
    )
    index = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (root / "rank_displacement_summary.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (root / "execution_provenance.json").read_text(encoding="utf-8")
    )

    assert _sha(root / "artifact_index.json") == (
        "ca1933513b224b735882de1196a234f2d1246649211855ac81c6d4cb3c1fdaf5"
    )
    assert summary["diagnostic_id"] == (
        "NREV-DIAG-EXP-0029-RANK-DISPLACEMENT-20260830-F"
    )
    assert index["artifact_count"] == 11
    assert sum(row["bytes"] for row in index["artifacts"]) == 491_492
    assert execution["runtime"]["scipy_version"] == "1.17.1"
    assert execution["duration_seconds"] == 42.225716
    for row in index["artifacts"]:
        artifact = root / row["path"]
        assert artifact.stat().st_size == row["bytes"]
        assert _sha(artifact) == row["sha256"]


def test_analytical_report_sources_are_portable_and_claim_bounded() -> None:
    report_path = (
        ROOT
        / "docs"
        / "reports"
        / "neuron_identifiability_automated_diagnostics_20260830"
        / "artifact.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generated_at = report["manifest"]["generatedAt"]
    assert generated_at == report["snapshot"]["generatedAt"]
    assert datetime.fromisoformat(generated_at.replace("Z", "+00:00")) > datetime.fromisoformat(
        "2026-08-30T14:49:23.491905+00:00"
    )

    manifest_source_ids = {row["id"] for row in report["manifest"]["sources"]}
    assert "rank_f" in manifest_source_ids
    assert "rank_e" not in manifest_source_ids
    for source in report["sources"]:
        assert source["id"] in manifest_source_ids
        assert (ROOT / source["path"]).is_file(), source["path"]
        assert source["query"]["executed_at"] == generated_at

    visible_text = "\n".join(
        row.get("body", "") for row in report["manifest"]["blocks"]
    )
    visible_text += "\n" + "\n".join(
        column["label"]
        for table in report["manifest"]["tables"]
        for column in table["columns"]
    )
    assert "score-map mechanisms" not in visible_text
    assert "safety certificate" not in visible_text
    assert "Runs A-D" not in visible_text
    assert "introduced enough" not in visible_text
    assert "Not recovered in either map" in visible_text

    reports_index = (ROOT / "docs" / "reports" / "README.md").read_text(
        encoding="utf-8"
    )
    assert report_path.parent.name in reports_index
