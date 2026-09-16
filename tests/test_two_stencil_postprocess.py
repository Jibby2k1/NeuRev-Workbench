"""Completion and linked numerical-state provenance must fail closed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import two_stencil_audit as renderer
from neurobench.experiments.gamma_ls_difference import two_stencil_postprocess as module


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_state(root, cell, revision=None):
    output = root / "scientific_audits"
    if revision is not None:
        output /= revision
    output /= cell
    output.mkdir(parents=True)
    source = root / f"{cell}_source.txt"
    source.write_text("sealed stage fixture")
    summary = {"status": "complete", "scientific_audit_complete": True, "video_count": 3}
    _write(output / "summary.json", summary)
    _write(output / "status.json", {"status": "complete", "scientific_audit_complete": True})
    _write(output / "run_contract.json", {"cell_id": cell})
    _write(output / "source_manifest.json", {"sources": [{"path": str(source), "sha256": _hash(source)}]})
    (output / "figure.png").write_bytes(b"previously verified artifact fixture")
    _write(output / "artifact_index.json", {"artifacts": [
        {"path": path.name, "sha256": _hash(path)} for path in sorted(output.iterdir())
    ]})
    config = root / "audit_configs" / f"{cell}.json"
    _write(config, {"output_root": str(output)})
    return {"cell_id": cell, "config_path": str(config), "audit_revision": revision}, summary, source


def test_nonselected_complete_states_are_verified_and_new_returns_are_not_rehashed(tmp_path, monkeypatch):
    selected, result, _ = _complete_state(tmp_path, "selected")
    older, _, _ = _complete_state(tmp_path, "older")
    calls = []
    def verify_existing(**config):
        output = Path(config["output_root"])
        calls.append(output.name)
        renderer.verify_completed_audit(output)
        return json.loads((output / "summary.json").read_text())
    monkeypatch.setattr(renderer, "run_two_stencil_audit", verify_existing)
    complete = module.collect_audit_completion(tmp_path, [selected, older], verified_this_run={"selected": result})
    assert calls == ["older"]
    assert [row["cell_id"] for row in complete] == ["selected", "older"]
    assert complete[0]["completion_verification"] == "successful_renderer_return_this_invocation"
    assert complete[1]["completion_verification"] == "existing_complete_contract_artifacts_and_sources_reverified"
    assert all(len(row["artifact_index_sha256"]) == 64 for row in complete)


@pytest.mark.parametrize("damage", ["artifact", "source", "status", "config"])
def test_complete_summary_cannot_bypass_old_state_damage(tmp_path, monkeypatch, damage):
    record, _, source = _complete_state(tmp_path, "older")
    output = tmp_path / "scientific_audits/older"
    if damage == "artifact":
        (output / "figure.png").write_bytes(b"changed")
    elif damage == "source":
        source.write_text("changed")
    elif damage == "status":
        _write(output / "status.json", {"status": "rendering", "scientific_audit_complete": False})
    else:
        _write(Path(record["config_path"]), {"output_root": str(tmp_path / "another_state")})
    def verify_existing(**config):
        current = Path(config["output_root"])
        renderer.verify_completed_audit(current)
        return json.loads((current / "summary.json").read_text())
    monkeypatch.setattr(renderer, "run_two_stencil_audit", verify_existing)
    with pytest.raises(ValueError):
        module.collect_audit_completion(tmp_path, [record])


def test_completion_uses_revision_configuration_without_counting_legacy_media(tmp_path, monkeypatch):
    record, _, _ = _complete_state(tmp_path, "cell", revision="encoding_v2")
    calls = []
    def verify_existing(**config):
        output = Path(config["output_root"])
        calls.append(output)
        renderer.verify_completed_audit(output)
        return json.loads((output / "summary.json").read_text())
    monkeypatch.setattr(renderer, "run_two_stencil_audit", verify_existing)
    complete = module.collect_audit_completion(tmp_path, [record])
    assert calls == [tmp_path / "scientific_audits/encoding_v2/cell"]
    assert complete[0]["audit_revision"] == "encoding_v2"
    assert complete[0]["output_root"] == str(calls[0])


def _numeric_fixture(root, monkeypatch):
    labels = root / "expert.tsv"
    labels.write_text("sealed label fixture")
    monkeypatch.setattr(module, "LABEL_PATH", labels)
    monkeypatch.setattr(renderer, "audit_inventory_plan", lambda *a, **k: {
        "model_roi_count": 1, "expected_video_count": 4,
        "expected_trace_figure_count": 3, "closeup_video_frame_count_total": 8,
    })
    targets = [.25, .5, 1., 2., 5.]
    cell = "cell"
    folder = root / "cells" / cell
    _write(root / "protocol.json", {
        "cells": [{"cell_id": cell, "input_representation": "signed"}],
        "target_proposals_per_frame": targets,
        "inputs": {"signed": {"path": "source.npy", "file_sha256": "source_hash"}},
        "source_movie": {"path": "raw.npy", "file_sha256": "raw_hash"},
    })
    _write(root / "preflight.json", {"label_sha256": _hash(labels)})
    _write(root / "report_artifacts/display_limits.json", {"limits": {"signed": {
        name: {"signed": False, "vmin": 0, "vmax": 1}
        for name in ("Raw", "X", "A", "M", "sigma", "contrast", "Z")
    }}})
    _write(folder / "operator.json", {})
    seal_rows, summaries = [], []
    for q in targets:
        q_root = folder / "readouts/Z" / f"q{q:g}"
        q_root.mkdir(parents=True)
        candidate = q_root / "candidates.tsv"
        candidate.write_text(f"sealed candidate fixture q={q}\n")
        seal_rows.append({"readout": "Z", "q": q, "path": str(candidate.relative_to(root)), "sha256": _hash(candidate)})
        summary = {"readout": "Z", "target_proposals_per_frame": q, "threshold": q}
        summaries.append(summary)
        _write(q_root / "summary.json", summary)
    _write(folder / "candidate_seal.json", {"candidate_files": seal_rows})
    _write(folder / "numeric_complete.json", {
        "candidate_seal_sha256": _hash(folder / "candidate_seal.json"),
        "metadata_sha256": {}, "summary_rows": summaries,
        "stage_paths": {}, "stage_sha256": {},
    })
    _write(root / "campaign_candidate_seal.json", {})
    return folder


@pytest.mark.parametrize("damage", ["none", "q2_candidate", "q5_summary"])
def test_every_linked_q_candidate_and_summary_is_sealed(tmp_path, monkeypatch, damage):
    folder = _numeric_fixture(tmp_path, monkeypatch)
    if damage == "q2_candidate":
        (folder / "readouts/Z/q2/candidates.tsv").write_text("changed")
    elif damage == "q5_summary":
        _write(folder / "readouts/Z/q5/summary.json", {"changed": True})
    if damage == "none":
        records = module.make_audit_configs(tmp_path)
        config = json.loads(Path(records[0]["config_path"]).read_text())
        assert set(config["fullnumeric_q_paths"]) == {"0.25", "0.5", "1", "2", "5"}
    else:
        with pytest.raises(ValueError, match="q2 candidate|q5 numeric summary"):
            module.make_audit_configs(tmp_path)


def test_revision_preserves_original_configuration_and_inventory(tmp_path, monkeypatch):
    _numeric_fixture(tmp_path, monkeypatch)
    original = module.make_audit_configs(tmp_path)
    original_config = Path(original[0]["config_path"])
    original_config_hash = _hash(original_config)
    original_inventory_hash = _hash(tmp_path / "audit_inventory_plan.json")
    records = module.make_audit_configs(tmp_path, audit_revision="encoding_v2")
    config_path = Path(records[0]["config_path"])
    config = json.loads(config_path.read_text())
    assert config_path == tmp_path / "audit_configs/encoding_v2/cell.json"
    assert config["output_root"] == str(tmp_path / "scientific_audits/encoding_v2/cell")
    assert config["source_binding"]["audit_revision"] == "encoding_v2"
    assert _hash(original_config) == original_config_hash
    assert _hash(tmp_path / "audit_inventory_plan.json") == original_inventory_hash
    assert (tmp_path / "audit_inventory_plan_encoding_v2.json").is_file()
    with pytest.raises(ValueError, match="revision"):
        module.make_audit_configs(tmp_path, audit_revision="../replace")
