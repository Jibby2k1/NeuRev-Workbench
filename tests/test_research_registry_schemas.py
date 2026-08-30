from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "research" / "schemas"
SCHEMA_NAMES = (
    "program",
    "claim",
    "experiment",
    "decision",
    "run",
    "evidence-capsule",
)
SHA256 = "a" * 64


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _validate(name: str, record: dict) -> None:
    jsonschema.Draft202012Validator(
        _schema(name),
        format_checker=jsonschema.FormatChecker(),
    ).validate(record)


@pytest.fixture(scope="module")
def valid_records() -> dict[str, dict]:
    return {
        "program": {
            "schema_version": 1,
            "record_type": "program",
            "id": "NREV-PRG-0001",
            "slug": "neuron-identifiability",
            "title": "Neuron identifiability",
            "summary": "Separate stable measurements from biological identity claims.",
            "lifecycle": "active",
            "scientific_scope": "Calcium-imaging source proposal and bounded identity evidence.",
            "objectives": ["Measure identity recovery without treating unknowns as negatives."],
            "boundaries": ["Current-recording evidence does not establish independent transfer."],
            "documentation": ["docs/programs/neuron-identifiability.md"],
            "visibility": "public",
        },
        "claim": {
            "schema_version": 1,
            "record_type": "claim",
            "id": "NREV-CLM-0001",
            "program_id": "NREV-PRG-0001",
            "title": "Spatial context improves simulated localization",
            "statement": "Spatial context improves localization in the frozen simulator.",
            "scope": "Paired exact-truth movies from the registered simulator only.",
            "limitations": ["Simulation is not biological external validation."],
            "claim_state": "supported",
            "evidence_tier": "computational_simulation",
            "review_state": "approved",
            "evidence_capsule_ids": ["NREV-EVC-EXP-0001"],
            "visibility": "public",
            "legacy_status": "established_computational_simulation",
        },
        "experiment": {
            "schema_version": 1,
            "record_type": "experiment",
            "id": "NREV-EXP-0001",
            "origin": {"kind": "native"},
            "program_id": "NREV-PRG-0001",
            "title": "Paired simulator localization stress test",
            "question": "Does frozen spatial context improve localization?",
            "objective": "Compare spatial context with the frozen variance baseline.",
            "rationale": "Exact-truth movies isolate spatial failure modes.",
            "lifecycle": "closed",
            "outcome": "supported",
            "evidence_tier": "computational_simulation",
            "review_state": "approved",
            "falsifiers": ["The paired localization interval includes no improvement."],
            "design": {
                "summary": "Run paired generator cells while varying only the score map.",
                "protocol_path": "docs/workflows/example.md",
                "unit_of_analysis": "Paired simulator cell",
                "comparison": "Spatial context versus pixel variance",
                "data_scope": "Frozen synthetic generator families",
            },
            "gates": [
                {
                    "id": "paired_gain",
                    "criterion": "Paired localization gain is positive.",
                    "stage": "analysis",
                    "required": True,
                }
            ],
            "claim_ids": ["NREV-CLM-0001"],
            "decision_ids": ["NREV-DEC-0001"],
            "run_ids": ["NREV-RUN-EXP-0001-A"],
            "evidence_capsule_id": "NREV-EVC-EXP-0001",
            "visibility": "public",
        },
        "decision": {
            "schema_version": 1,
            "record_type": "decision",
            "id": "NREV-DEC-0001",
            "origin": {"kind": "native"},
            "program_id": "NREV-PRG-0001",
            "title": "Retain spatial context as proposal backbone",
            "lifecycle": "active",
            "action": "retain",
            "rationale": "The frozen comparison passed its localization gate.",
            "evidence_summary": "Evidence supports simulator localization, not biological identity.",
            "effective_on": "2026-08-29",
            "review_state": "approved",
            "experiment_ids": ["NREV-EXP-0001"],
            "claim_ids": ["NREV-CLM-0001"],
            "evidence_capsule_ids": ["NREV-EVC-EXP-0001"],
            "constraints": ["Do not interpret spatial context as one-to-one source identity."],
            "visibility": "public",
        },
        "run": {
            "schema_version": 1,
            "record_type": "run",
            "id": "NREV-RUN-EXP-0001-A",
            "experiment_id": "NREV-EXP-0001",
            "program_id": "NREV-PRG-0001",
            "lifecycle": "succeeded",
            "planned_at": "2026-08-29T12:00:00Z",
            "started_at": "2026-08-29T12:05:00Z",
            "ended_at": "2026-08-29T12:10:00Z",
            "code": {
                "repository": "https://github.com/Jibby2k1/NeuRev-Workbench",
                "commit": "b" * 40,
                "branch": "main",
                "dirty": False,
            },
            "configuration": {"path": "examples/experiment.json", "sha256": SHA256},
            "inputs": [
                {
                    "id": "simulator-fixture",
                    "role": "data",
                    "sha256": SHA256,
                    "location": "examples/fixtures/simulator.json",
                }
            ],
            "environment": {
                "platform": "Linux x86_64",
                "runtime": "CPython 3.11",
                "dependency_lock": {"path": "pyproject.toml", "sha256": SHA256},
            },
            "randomization": {"deterministic": True, "seeds": [0, 1]},
            "artifacts": {
                "output_root": "Outputs/NeuronIdentifiability/NREV-EXP-0001/runs/NREV-RUN-EXP-0001-A",
                "manifest_path": "research/evidence/manifests/NREV-RUN-EXP-0001-A.json",
                "manifest_sha256": SHA256,
            },
            "validation": {
                "preflight": "passed",
                "execution": "passed",
                "outputs": "passed",
                "checks": [
                    {"id": "manifest", "status": "passed", "summary": "Manifest hashes resolve."}
                ],
            },
            "visibility": "public",
        },
        "evidence-capsule": {
            "schema_version": 1,
            "record_type": "evidence_capsule",
            "id": "NREV-EVC-EXP-0001",
            "experiment_id": "NREV-EXP-0001",
            "program_id": "NREV-PRG-0001",
            "generated_at": "2026-08-29T12:20:00Z",
            "lifecycle": "published",
            "outcome": "supported",
            "evidence_tier": "computational_simulation",
            "review_state": "approved",
            "summary": "The paired simulator localization gate passed.",
            "run_ids": ["NREV-RUN-EXP-0001-A"],
            "metrics": [
                {
                    "id": "localization_gain",
                    "label": "Mean paired localization gain",
                    "estimate": 0.1,
                    "unit": "pixels",
                    "primary": True,
                    "population": "Frozen paired simulator cells",
                }
            ],
            "gates": [
                {
                    "id": "paired_gain",
                    "criterion": "Paired localization gain is positive.",
                    "status": "passed",
                    "observed": "Mean gain was positive in the frozen test.",
                }
            ],
            "claim_impacts": [
                {
                    "claim_id": "NREV-CLM-0001",
                    "effect": "supports",
                    "summary": "Supports the bounded simulator claim.",
                    "scope": "Frozen simulator families only.",
                }
            ],
            "artifacts": [
                {
                    "id": "summary",
                    "role": "summary",
                    "media_type": "application/json",
                    "sha256": SHA256,
                    "access": "public",
                    "repository_path": "research/evidence/NREV-EXP-0001.json",
                }
            ],
            "artifact_manifest_sha256": SHA256,
            "validation": {
                "status": "passed",
                "checked_at": "2026-08-29T12:19:00Z",
                "checks": [
                    {"id": "schema", "status": "passed", "summary": "Capsule schema is valid."}
                ],
            },
            "boundaries": {
                "scope": "Frozen exact-truth simulator cells.",
                "limitations": ["The generator remains stylized."],
                "does_not_establish": ["Biological identity or independent-recording transfer."],
            },
            "source_records_sha256": SHA256,
            "visibility": "public",
        },
    }


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_registry_schema_is_valid_draft_2020_12(name: str) -> None:
    schema = _schema(name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    jsonschema.Draft202012Validator.check_schema(schema)


def test_representative_registry_records_validate(valid_records: dict[str, dict]) -> None:
    for name in SCHEMA_NAMES:
        _validate(name, valid_records[name])


@pytest.mark.parametrize("name", ("program", "experiment", "run", "evidence-capsule"))
def test_repository_paths_reject_absolute_or_parent_traversal(
    name: str,
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records[name])
    if name == "program":
        record["documentation"] = ["/" + "home" + "/reviewer/private.md"]
    elif name == "experiment":
        record["design"]["protocol_path"] = "../private/protocol.md"
    elif name == "run":
        record["configuration"]["path"] = "C:\\private\\config.json"
    else:
        record["artifacts"][0]["repository_path"] = "/tmp/private-summary.json"

    with pytest.raises(jsonschema.ValidationError):
        _validate(name, record)


def test_supported_claim_requires_evidence_capsule(valid_records: dict[str, dict]) -> None:
    record = copy.deepcopy(valid_records["claim"])
    record["evidence_capsule_ids"] = []
    with pytest.raises(jsonschema.ValidationError):
        _validate("claim", record)


def test_completed_run_requires_end_and_manifest(valid_records: dict[str, dict]) -> None:
    record = copy.deepcopy(valid_records["run"])
    del record["ended_at"]
    del record["artifacts"]["manifest_path"]
    del record["artifacts"]["manifest_sha256"]
    with pytest.raises(jsonschema.ValidationError):
        _validate("run", record)


def test_decision_must_target_claim_or_experiment(valid_records: dict[str, dict]) -> None:
    record = copy.deepcopy(valid_records["decision"])
    record["experiment_ids"] = []
    record["claim_ids"] = []
    with pytest.raises(jsonschema.ValidationError):
        _validate("decision", record)


def test_public_capsule_rejects_unregistered_fields(valid_records: dict[str, dict]) -> None:
    record = copy.deepcopy(valid_records["evidence-capsule"])
    record["private_randomization_key"] = "must never enter a public capsule"
    with pytest.raises(jsonschema.ValidationError):
        _validate("evidence-capsule", record)


@pytest.mark.parametrize(
    "field",
    ("summary", "unit_of_analysis", "comparison", "data_scope"),
)
def test_native_experiment_requires_complete_design(
    field: str,
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records["experiment"])
    del record["design"][field]
    with pytest.raises(jsonschema.ValidationError):
        _validate("experiment", record)


@pytest.mark.parametrize("field", ("falsifiers", "gates"))
def test_native_experiment_requires_nonempty_scientific_guards(
    field: str,
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records["experiment"])
    record[field] = []
    with pytest.raises(jsonschema.ValidationError):
        _validate("experiment", record)


def test_native_decision_requires_effective_date(valid_records: dict[str, dict]) -> None:
    record = copy.deepcopy(valid_records["decision"])
    del record["effective_on"]
    with pytest.raises(jsonschema.ValidationError):
        _validate("decision", record)


@pytest.mark.parametrize(
    "field",
    ("source_path", "source_sha256", "recorded_on", "missing_fields"),
)
def test_migrated_origin_requires_explicit_provenance(
    field: str,
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records["experiment"])
    record["origin"] = {
        "kind": "migrated_legacy",
        "source_path": "research/registry/migration/research_story_v1.yaml",
        "source_sha256": SHA256,
        "recorded_on": "2026-08-29",
        "missing_fields": ["Historical run-level provenance was not recorded."],
    }
    del record["origin"][field]
    with pytest.raises(jsonschema.ValidationError):
        _validate("experiment", record)


def test_migrated_origin_rejects_empty_provenance_gap_list(
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records["decision"])
    record["origin"] = {
        "kind": "migrated_legacy",
        "source_path": "research/registry/migration/research_story_v1.yaml",
        "source_sha256": SHA256,
        "recorded_on": "2026-08-29",
        "missing_fields": [],
    }
    del record["effective_on"]
    with pytest.raises(jsonschema.ValidationError):
        _validate("decision", record)


def test_migrated_decision_may_record_unknown_date_without_inventing_one(
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records["decision"])
    record["origin"] = {
        "kind": "migrated_legacy",
        "source_path": "research/registry/migration/research_story_v1.yaml",
        "source_sha256": SHA256,
        "recorded_on": "2026-08-29",
        "missing_fields": ["Historical decision effective date was not recorded."],
    }
    del record["effective_on"]
    _validate("decision", record)


def test_migrated_experiment_may_record_unknowns_without_fabricating_guards(
    valid_records: dict[str, dict],
) -> None:
    record = copy.deepcopy(valid_records["experiment"])
    record["origin"] = {
        "kind": "migrated_legacy",
        "source_path": "research/registry/migration/research_story_v1.yaml",
        "source_sha256": SHA256,
        "recorded_on": "2026-08-29",
        "missing_fields": [
            "Historical run-level provenance was not recorded.",
            "Preregistered gates and falsifiers were not recorded.",
        ],
    }
    record["design"] = {"summary": "Faithful migration of the recorded design."}
    record["falsifiers"] = []
    record["gates"] = []
    record.pop("run_ids")
    _validate("experiment", record)
