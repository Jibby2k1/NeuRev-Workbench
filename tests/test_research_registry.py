from __future__ import annotations

import copy
from pathlib import Path

import pytest

from neurobench.research.hashing import sha256_file
from neurobench.research.registry import (
    RegistryError,
    _llm_context,
    _project_story,
    _require_repository_layout,
    build,
    compile_registry,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_normalized_registry_is_valid_and_generated_views_are_current() -> None:
    stale, warnings = build(check=True)
    assert stale == 0
    assert warnings == []


def test_migrated_records_bind_to_the_retained_source_without_fake_runs() -> None:
    payload = compile_registry()
    migrated_experiments = [
        row for row in payload["experiments"] if row["origin"]["kind"] == "migrated_legacy"
    ]
    migrated_decisions = [
        row for row in payload["decisions"] if row["origin"]["kind"] == "migrated_legacy"
    ]
    assert migrated_experiments
    assert migrated_decisions

    for row in [*migrated_experiments, *migrated_decisions]:
        origin = row["origin"]
        source = ROOT / origin["source_path"]
        assert source.is_file(), row["id"]
        assert sha256_file(source) == origin["source_sha256"], row["id"]
        assert origin["recorded_on"]
        assert origin["missing_fields"]

    completed = [row for row in migrated_experiments if row["lifecycle"] == "validated"]
    assert completed
    records_with_run_gaps = [
        row
        for row in completed
        if any("run-level" in gap for gap in row["origin"]["missing_fields"])
    ]
    assert records_with_run_gaps
    for row in records_with_run_gaps:
        assert not row.get("run_ids"), row["id"]


def test_native_executed_experiment_without_run_is_rejected() -> None:
    payload = copy.deepcopy(compile_registry())
    experiment = next(row for row in payload["experiments"] if row["lifecycle"] == "validated")
    experiment["origin"] = {"kind": "native"}
    experiment["design"].update(
        {
            "unit_of_analysis": "Registered occurrence",
            "comparison": "Registered comparison",
            "data_scope": "Registered data scope",
        }
    )
    experiment["falsifiers"] = ["The registered result fails its prospective gate."]
    experiment["gates"] = [
        {
            "id": "prospective_gate",
            "criterion": "The prospective criterion passes.",
            "stage": "analysis",
            "required": True,
        }
    ]

    with pytest.raises(RegistryError, match="native executed experiment has no registered run"):
        validate_registry(payload)


def test_native_draft_is_rendered_without_fabricated_legacy_priority() -> None:
    payload = copy.deepcopy(compile_registry())
    payload["experiments"].append(
        {
            "id": "NREV-EXP-9001",
            "origin": {"kind": "native"},
            "title": "Native future experiment",
            "objective": "Exercise the post-migration queue contract.",
            "lifecycle": "draft",
            "created_on": max(row.get("created_on", "") for row in payload["experiments"]),
        }
    )

    story = _project_story(payload)
    context = _llm_context(payload)
    rendered = next(row for row in context["next_experiments"] if row["id"] == "NREV-EXP-9001")

    assert "Native future experiment" in story
    assert "scientific priority not assigned" in story
    assert rendered["priority"] is None
    assert rendered["queue_basis"] == "registered_draft_without_assigned_priority"
    assert context["next_experiments"][-1] == rendered


def test_registry_compiler_explains_checkout_only_distribution_boundary(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="checkout-only command"):
        _require_repository_layout(tmp_path)
