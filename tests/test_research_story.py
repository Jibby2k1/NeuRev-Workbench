import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "overleaf_jnm"


def _paper_builder_module():
    path = PAPER / "scripts" / "build_research_story.py"
    spec = importlib.util.spec_from_file_location("neurev_paper_story_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_portable_artifact(path: str, capsules: list[str]) -> None:
    resolved = (PAPER / path).resolve()
    if "Outputs" in resolved.parts:
        assert capsules, f"ignored local artifact lacks a portable evidence capsule: {path}"
        repository_path = resolved.relative_to(ROOT).as_posix()
        registered_artifacts = set()
        for capsule in capsules:
            capsule_path = (PAPER / capsule).resolve()
            assert capsule_path.is_file(), path
            payload = json.loads(capsule_path.read_text())
            registered_artifacts.update(item["repository_path"] for item in payload["artifacts"])
        assert repository_path in registered_artifacts, path
    else:
        assert resolved.is_file(), path


def test_story_registry_has_unique_scoped_claims_and_portable_evidence() -> None:
    payload = yaml.safe_load((PAPER / "story" / "research_story.yaml").read_text())
    assert payload["schema_version"] == 1
    claim_ids = [row["id"] for row in payload["claims"]]
    experiment_ids = [row["id"] for row in payload["experiments"]]
    assert len(claim_ids) == len(set(claim_ids))
    assert len(experiment_ids) == len(set(experiment_ids))
    assert all(row["scope"] for row in payload["claims"])
    for row in payload["claims"]:
        for path in row["evidence"]:
            _assert_portable_artifact(path, row.get("evidence_capsules", []))
    for row in payload["experiments"]:
        capsules = [row["evidence_capsule"]] if row.get("evidence_capsule") else []
        for path in row["artifacts"]:
            _assert_portable_artifact(path, capsules)
    for row in payload["next_experiments"]:
        if row.get("plan"):
            assert (PAPER / row["plan"]).is_file()


def test_registered_engineering_runs_are_visible_without_scientific_promotion() -> None:
    payload = yaml.safe_load((PAPER / "story" / "research_story.yaml").read_text())
    experiments = {row["id"]: row for row in payload["experiments"]}

    feature_learning = experiments["NREV-EXP-0021"]
    jepa = experiments["NREV-EXP-0028"]
    residual = experiments["NREV-EXP-0029"]
    for row in (feature_learning, jepa, residual):
        assert row["status"].startswith("draft_not_evaluated_engineering_run_history_")
        assert "evidence_tier_none" in row["status"]
        assert "evidence_capsule" not in row
        assert "non-claim-bearing engineering history" in row["limitation"]
        assert all("Outputs" not in path for path in row["artifacts"])

    assert "failed_then_succeeded" in feature_learning["status"]
    assert "did not improve the primary estimand" in feature_learning["next_decision"]
    assert "34 of 78" in feature_learning["finding"]
    assert (
        "../../docs/research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md"
        in feature_learning["artifacts"]
    )
    assert "failed_then_succeeded" in residual["status"]
    assert "0.1875 raw" in residual["finding"]
    assert "Hold this implementation path" in residual["next_decision"]
    assert "design triage" in residual["next_decision"]


def test_report_history_preserves_program_and_audit_boundaries() -> None:
    from neurobench.research.registry import compile_registry, _llm_context, _paper_story

    registry = compile_registry()
    context = _llm_context(registry)
    results = {row["id"]: row for row in context["documented_results"]}
    paper = {row["id"]: row for row in _paper_story(registry)["experiments"]}

    # A result on another recording in a distinct program must not widen the
    # identity-safe manuscript's cohort or its generalization claims.
    assert results["NREV-EXP-0034"]["program_id"] != registry["index"]["flagship_program_id"]
    assert "6/51" in results["NREV-EXP-0034"]["finding"]
    assert "20/51" in results["NREV-EXP-0035"]["finding"]
    assert not {"NREV-EXP-0034", "NREV-EXP-0035", "NREV-EXP-0036"} & paper.keys()
    assert "NREV-EXP-0037" in paper

    # The renderer must not contradict a validated morphology outcome or
    # pretend an incomplete new-ranking audit passed.
    morphology = paper["NREV-EXP-0033"]
    assert "validated_supported" in morphology["status"]
    assert "scientific outcome was evaluated" not in morphology["finding"]
    assert "current_recording" in morphology["limitation"]
    assert "evidence tier is none" not in morphology["limitation"]
    assert "remain incomplete" in paper["NREV-EXP-0037"]["finding"]
    for row in registry["experiments"]:
        if row["id"] in {"NREV-EXP-0034", "NREV-EXP-0035", "NREV-EXP-0036", "NREV-EXP-0037"}:
            assert not row["claim_ids"]
            assert not row["run_ids"]
            assert "evidence_capsule_id" not in row


def test_clean_clone_story_check_uses_capsule_for_ignored_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _paper_builder_module()
    repository = tmp_path / "repository"
    paper = repository / "paper" / "overleaf_jnm"
    capsule_path = repository / "research" / "evidence" / "fixture.json"
    capsule_path.parent.mkdir(parents=True)
    paper.mkdir(parents=True)

    absent_output = "../../Outputs/__clean_clone_fixture__/summary.json"
    capsule = "../../research/evidence/fixture.json"
    capsule_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"repository_path": "Outputs/__clean_clone_fixture__/summary.json"}
                ]
            }
        )
    )
    monkeypatch.setattr(builder, "ROOT", paper)
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", repository)
    assert not (paper / absent_output).exists()

    builder._portable_artifact(
        absent_output,
        capsules=[capsule],
        verify_live_artifacts=False,
        owner="clean-clone fixture",
    )
    with pytest.raises(FileNotFoundError):
        builder._portable_artifact(
            absent_output,
            capsules=[capsule],
            verify_live_artifacts=True,
            owner="clean-clone fixture",
        )


def test_generated_story_views_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "neurobench.research.registry", "check"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    result = subprocess.run(
        [sys.executable, str(PAPER / "scripts" / "build_research_story.py"), "--check"],
        cwd=PAPER, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
