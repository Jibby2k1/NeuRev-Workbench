from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

import neurobench.programs.fish_control as fish_control

from neurobench.cli.main import build_parser, main
from neurobench.programs.fish_control import (
    ProgramManifestError,
    audit_program_manifest,
    load_program_manifest,
    write_program_audit,
)
from neurobench.validation.schemas import schema_path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fish_control_program.example.json"


def _healthy_local_preflight(_project_root, *, inspect_gpu):
    return {
        "free_disk_gib": 500.0,
        "disk_probe_error": None,
        "active_gpu_jobs": 0 if inspect_gpu else None,
        "gpu_probe_error": None,
    }


def test_example_validates_and_declares_exact_job_count():
    manifest = load_program_manifest(EXAMPLE)
    audit = audit_program_manifest(EXAMPLE, check_paths=False, generated_at="2026-07-21T00:00:00Z")

    assert manifest["program_id"] == "fish_inverse_control_v1"
    assert audit["summary"]["experiment_count"] == 8
    assert audit["summary"]["planned_compute_jobs"] == 68
    assert audit["summary"]["recommended_next_experiment"] == "fc00_activation_annotation_panel_v1"
    assert audit["experiments"][0]["readiness"] == "manual_action_required"
    assert not any(
        blocker["type"] == "resource"
        for experiment in audit["experiments"]
        for blocker in experiment["blockers"]
    )


def test_program_schema_alias_resolves():
    assert schema_path("fish_control_program").name == "fish_control_program.schema.json"
    assert schema_path("fish_program").name == "fish_control_program.schema.json"


def test_semantic_validation_rejects_job_count_mismatch(tmp_path):
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["experiments"][1]["design"]["expected_jobs"] = 99
    path = tmp_path / "program.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProgramManifestError, match="factor grid"):
        load_program_manifest(path)


def test_semantic_validation_rejects_dependency_cycle(tmp_path):
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["experiments"][0]["dependencies"] = ["fc01_frozen_detector_tournament_v1"]
    path = tmp_path / "program.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProgramManifestError, match="dependency cycle"):
        load_program_manifest(path)


def test_path_checks_block_missing_future_artifacts(monkeypatch):
    monkeypatch.setattr(
        fish_control, "_collect_local_preflight", _healthy_local_preflight
    )
    audit = audit_program_manifest(EXAMPLE, check_paths=True, generated_at="2026-07-21T00:00:00Z")
    tournament = next(
        item
        for item in audit["experiments"]
        if item["id"] == "fc01_frozen_detector_tournament_v1"
    )

    assert tournament["readiness"] == "blocked"
    assert "../Outputs/FishControl/activation_annotation_panel_v1/benchmark_manifest.json" in tournament["missing_inputs"]


def test_audit_writes_atomic_json_and_markdown(tmp_path):
    audit = audit_program_manifest(EXAMPLE, check_paths=False, generated_at="2026-07-21T00:00:00Z")
    paths = write_program_audit(audit, tmp_path)

    written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["summary"]["planned_compute_jobs"] == 68
    assert "# Fish-control program audit" in markdown
    assert "This audit reports readiness; it does not authorize GPU or stimulation work." in markdown


def test_program_cli_is_selective_and_audits_without_path_checks(capsys):
    args = build_parser(active_command="program").parse_args(
        [
            "program",
            "fish-control",
            "audit",
            "--manifest",
            str(EXAMPLE),
            "--no-path-checks",
        ]
    )
    assert callable(args.func)

    code = main(
        [
            "program",
            "fish-control",
            "audit",
            "--manifest",
            str(EXAMPLE),
            "--no-path-checks",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["summary"]["planned_compute_jobs"] == 68



def test_schema_rejects_cuda_without_explicit_approval(tmp_path):
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    experiment = payload["experiments"][0]
    experiment["resources"].update(
        device="cuda",
        gpu_memory_target_mib=8500,
        gpu_memory_hard_limit_mib=9600,
    )
    experiment["launch_policy"] = "safe_cpu"
    path = tmp_path / "program.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="explicit_approval"):
        load_program_manifest(path)


def test_running_experiment_is_not_launchable_or_recommended(tmp_path):
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["experiments"][0]["status"] = "running"
    path = tmp_path / "program.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_program_manifest(
        path, check_paths=False, generated_at="2026-07-21T00:00:00Z"
    )

    assert audit["experiments"][0]["readiness"] == "running"
    assert (
        audit["summary"]["recommended_next_experiment"]
        == "fc03_intent_dataset_readiness_v1"
    )


def test_local_preflight_enforces_disk_and_gpu_concurrency(monkeypatch):
    monkeypatch.setattr(
        fish_control,
        "_collect_local_preflight",
        lambda _root, *, inspect_gpu: {
            "free_disk_gib": 100.0,
            "disk_probe_error": None,
            "active_gpu_jobs": 1 if inspect_gpu else None,
            "gpu_probe_error": None,
        },
    )

    audit = audit_program_manifest(
        EXAMPLE, check_paths=True, generated_at="2026-07-21T00:00:00Z"
    )
    first = audit["experiments"][0]
    cuda = next(
        item
        for item in audit["experiments"]
        if item["id"] == "fc02_structured_background_pu_v1"
    )
    first_resource_details = [
        blocker["detail"]
        for blocker in first["blockers"]
        if blocker["type"] == "resource"
    ]
    cuda_resource_details = [
        blocker["detail"]
        for blocker in cuda["blockers"]
        if blocker["type"] == "resource"
    ]

    assert any("below required minimum" in detail for detail in first_resource_details)
    assert any("active GPU jobs=1" in detail for detail in cuda_resource_details)


def test_resume_atomic_requires_owned_resumable_marker(tmp_path, monkeypatch):
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    experiment = payload["experiments"][0]
    payload["experiments"] = [experiment]
    experiment["inputs"] = []
    experiment["output_root"] = "existing_output"
    experiment["resume_policy"] = "resume_atomic"
    output_root = tmp_path / "existing_output"
    output_root.mkdir()
    path = tmp_path / "program.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        fish_control, "_collect_local_preflight", _healthy_local_preflight
    )

    missing = audit_program_manifest(
        path, check_paths=True, generated_at="2026-07-21T00:00:00Z"
    )
    assert missing["experiments"][0]["readiness"] == "blocked"
    assert any(
        "missing program_run.json" in blocker["detail"]
        for blocker in missing["experiments"][0]["blockers"]
    )

    marker = {
        "schema_version": 1,
        "program_id": payload["program_id"],
        "experiment_id": experiment["id"],
        "status": "stopped",
    }
    marker_path = output_root / "program_run.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    resumable = audit_program_manifest(
        path, check_paths=True, generated_at="2026-07-21T00:00:00Z"
    )
    assert resumable["experiments"][0]["readiness"] == "manual_action_required"
    assert resumable["experiments"][0]["resume_marker"] == marker

    marker["status"] = "completed"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    completed = audit_program_manifest(
        path, check_paths=True, generated_at="2026-07-21T00:00:00Z"
    )
    assert completed["experiments"][0]["readiness"] == "blocked"
    assert any(
        "not safely resumable" in blocker["detail"]
        for blocker in completed["experiments"][0]["blockers"]
    )
