"""Compile and validate the repository-wide NeuRev research registry."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from neurobench.research.hashing import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
REGISTRY = RESEARCH / "registry"
EVIDENCE = RESEARCH / "evidence"
GENERATED = RESEARCH / "generated"
SCHEMAS = RESEARCH / "schemas"
MIGRATION = REGISTRY / "migration"
PAPER = ROOT / "paper" / "overleaf_jnm"
PAPER_STORY = PAPER / "story" / "research_story.yaml"

SCHEMA_BY_GROUP = {
    "programs": "program.schema.json",
    "claims": "claim.schema.json",
    "experiments": "experiment.schema.json",
    "decisions": "decision.schema.json",
    "runs": "run.schema.json",
    "evidence_capsules": "evidence-capsule.schema.json",
}


class RegistryError(ValueError):
    """Raised when canonical research records are inconsistent."""


def _require_repository_layout(root: Path | None = None) -> None:
    """Fail clearly when the repository-owned registry is not available.

    The Python wheel contains the compiler implementation, but canonical
    scientific records remain in the source repository so there is only one
    authority to update and regenerate.
    """
    checkout = ROOT if root is None else root
    required = (
        Path("research/registry/views.yaml"),
        Path("research/registry/index.yaml"),
        Path("research/schemas/experiment.schema.json"),
    )
    missing = [path.as_posix() for path in required if not (checkout / path).is_file()]
    if missing:
        raise RegistryError(
            "repository research registry is unavailable (missing "
            + ", ".join(missing)
            + "). This checkout-only command is not backed by records in the "
            "Python wheel; run it from an editable NeuRev source checkout or "
            "inspect the generated research views in the repository."
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise RegistryError(f"expected mapping: {path.relative_to(ROOT)}")
    return payload


def _load_group(name: str) -> list[dict[str, Any]]:
    directory = REGISTRY / name
    return sorted((_load_yaml(path) for path in directory.glob("*.yaml")), key=lambda row: row["id"])


def _load_evidence() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVIDENCE.glob("*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise RegistryError(f"expected mapping: {path.relative_to(ROOT)}")
        payload["_registry_filename"] = path.name
        rows.append(payload)
    return rows


def compile_registry() -> dict[str, Any]:
    """Load federated records into one deterministic canonical object."""
    _require_repository_layout()
    capsules = _load_evidence()
    for row in capsules:
        row.pop("_registry_filename", None)
    return {
        "schema_version": 1,
        "authority": "research/registry",
        "views": _load_yaml(REGISTRY / "views.yaml"),
        "index": _load_yaml(REGISTRY / "index.yaml"),
        "compatibility": _load_yaml(MIGRATION / "story_compatibility.yaml"),
        "programs": _load_group("programs"),
        "claims": _load_group("claims"),
        "experiments": _load_group("experiments"),
        "decisions": _load_group("decisions"),
        "runs": _load_group("runs"),
        "evidence_capsules": capsules,
    }


def _unique(rows: Iterable[dict[str, Any]], group: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str):
            errors.append(f"{group}: missing string id")
            continue
        if identifier in index:
            errors.append(f"{group}: duplicate id {identifier}")
        index[identifier] = row
    return index


def _schema_errors(payload: dict[str, Any], errors: list[str]) -> None:
    checker = FormatChecker()
    for group, filename in SCHEMA_BY_GROUP.items():
        schema = json.loads((SCHEMAS / filename).read_text())
        validator = Draft202012Validator(schema, format_checker=checker)
        for row in payload[group]:
            identifier = row.get("id", "<missing-id>")
            for error in sorted(validator.iter_errors(row), key=lambda item: list(item.absolute_path)):
                field = ".".join(map(str, error.absolute_path)) or "<record>"
                errors.append(f"{group} {identifier} [{field}]: {error.message}")


def _safe_repository_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", value):
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
        return False
    return ".." not in Path(value).parts


def _artifact_path(row: dict[str, Any]) -> Path | None:
    value = row.get("repository_path")
    if not _safe_repository_path(value):
        return None
    return ROOT / str(value)


def _validate_index(payload: dict[str, Any], indexes: dict[str, dict[str, Any]], errors: list[str]) -> None:
    expected = {
        "programs": list(indexes["programs"]),
        "claims": list(indexes["claims"]),
        "experiments": list(indexes["experiments"]),
        "decisions": list(indexes["decisions"]),
        "runs": list(indexes["runs"]),
        "evidence_capsules": list(indexes["capsules"]),
    }
    registered = payload.get("index", {})
    for field, identifiers in expected.items():
        if registered.get(field) != identifiers:
            errors.append(f"index.{field}: expected {identifiers!r}, found {registered.get(field)!r}")
    completed = [row["id"] for row in payload["experiments"] if row.get("evidence_capsule_id")]
    planned = [row["id"] for row in payload["experiments"] if row.get("lifecycle") == "draft"]
    if registered.get("completed_experiments") != completed:
        errors.append("index.completed_experiments does not match experiments carrying evidence capsules")
    if registered.get("planned_experiments") != planned:
        errors.append("index.planned_experiments does not match draft experiments")


def _validate_snapshot(payload: dict[str, Any], errors: list[str]) -> None:
    migration = payload.get("views", {}).get("migration", {})
    snapshot_hint = migration.get("snapshot")
    if not _safe_repository_path(snapshot_hint):
        errors.append(f"views.migration.snapshot: unsafe path {snapshot_hint!r}")
        return
    snapshot = ROOT / str(snapshot_hint)
    if not snapshot.is_file():
        errors.append(f"views.migration.snapshot: missing {snapshot_hint}")
        return
    if sha256_file(snapshot) != migration.get("snapshot_sha256"):
        errors.append("views.migration.snapshot_sha256 does not match the retained v1 story")


def validate_registry(payload: dict[str, Any], *, verify_live: bool = False) -> list[str]:
    """Return warnings; raise :class:`RegistryError` when records are invalid."""
    errors: list[str] = []
    warnings: list[str] = []
    _schema_errors(payload, errors)

    programs = _unique(payload["programs"], "programs", errors)
    claims = _unique(payload["claims"], "claims", errors)
    experiments = _unique(payload["experiments"], "experiments", errors)
    decisions = _unique(payload["decisions"], "decisions", errors)
    runs = _unique(payload["runs"], "runs", errors)
    capsules = _unique(payload["evidence_capsules"], "evidence_capsules", errors)
    indexes = {
        "programs": programs,
        "claims": claims,
        "experiments": experiments,
        "decisions": decisions,
        "runs": runs,
        "capsules": capsules,
    }
    _validate_index(payload, indexes, errors)
    _validate_snapshot(payload, errors)

    for row in payload["programs"]:
        location = f"program {row.get('id')}"
        for related in row.get("related_program_ids", []):
            if related not in programs:
                errors.append(f"{location}: unknown related program {related}")

    for row in payload["claims"]:
        location = f"claim {row.get('id')}"
        if row.get("program_id") not in programs:
            errors.append(f"{location}: unknown program {row.get('program_id')}")
        for identifier in row.get("evidence_capsule_ids", []):
            if identifier not in capsules:
                errors.append(f"{location}: unknown evidence capsule {identifier}")
        for identifier in row.get("decision_ids", []):
            if identifier not in decisions:
                errors.append(f"{location}: unknown decision {identifier}")

    for row in payload["experiments"]:
        location = f"experiment {row.get('id')}"
        if row.get("program_id") not in programs:
            errors.append(f"{location}: unknown program {row.get('program_id')}")
        for field, known in (("claim_ids", claims), ("decision_ids", decisions), ("run_ids", runs)):
            for identifier in row.get(field, []):
                if identifier not in known:
                    errors.append(f"{location}: unknown {field.removesuffix('_ids')} {identifier}")
        evidence_id = row.get("evidence_capsule_id")
        if evidence_id is not None and evidence_id not in capsules:
            errors.append(f"{location}: unknown evidence capsule {evidence_id}")
        if row.get("origin", {}).get("kind") == "native" and row.get("lifecycle") not in {"draft", "preregistered"} and not row.get("run_ids"):
            errors.append(f"{location}: native executed experiment has no registered run")

    for row in payload["decisions"]:
        location = f"decision {row.get('id')}"
        if row.get("program_id") not in programs:
            errors.append(f"{location}: unknown program {row.get('program_id')}")
        for field, known in (("experiment_ids", experiments), ("claim_ids", claims), ("evidence_capsule_ids", capsules)):
            for identifier in row.get(field, []):
                if identifier not in known:
                    errors.append(f"{location}: unknown {field.removesuffix('_ids')} {identifier}")

    for row in payload["runs"]:
        location = f"run {row.get('id')}"
        if row.get("program_id") not in programs:
            errors.append(f"{location}: unknown program {row.get('program_id')}")
        if row.get("experiment_id") not in experiments:
            errors.append(f"{location}: unknown experiment {row.get('experiment_id')}")

    for row in payload["evidence_capsules"]:
        location = f"evidence capsule {row.get('id')}"
        experiment = experiments.get(row.get("experiment_id"))
        if experiment is None:
            errors.append(f"{location}: unknown experiment {row.get('experiment_id')}")
            continue
        expected_filename = f"{row['experiment_id']}.json"
        if not (EVIDENCE / expected_filename).is_file():
            errors.append(f"{location}: canonical file is missing: research/evidence/{expected_filename}")
        if row.get("program_id") != experiment.get("program_id"):
            errors.append(f"{location}: program does not match experiment")
        if experiment.get("evidence_capsule_id") != row.get("id"):
            errors.append(f"{location}: experiment does not point back to this capsule")
        for run_id in row.get("run_ids", []):
            if run_id not in runs:
                errors.append(f"{location}: unknown run {run_id}")
        for impact in row.get("claim_impacts", []):
            claim_id = impact.get("claim_id")
            if claim_id not in claims:
                errors.append(f"{location}: unknown claim impact {claim_id}")
            elif claim_id not in experiment.get("claim_ids", []):
                errors.append(f"{location}: claim impact {claim_id} is not registered on the experiment")
        if canonical_sha256(row.get("artifacts", [])) != row.get("artifact_manifest_sha256"):
            errors.append(f"{location}: artifact_manifest_sha256 mismatch")
        linked_decisions = [decisions[x] for x in experiment.get("decision_ids", []) if x in decisions]
        source_records = {
            "experiment": experiment,
            "decisions": sorted(linked_decisions, key=lambda item: item["id"]),
            "claims": sorted(
                (claims[x] for x in experiment.get("claim_ids", []) if x in claims), key=lambda item: item["id"]
            ),
        }
        if canonical_sha256(source_records) != row.get("source_records_sha256"):
            errors.append(f"{location}: source_records_sha256 mismatch")
        for artifact in row.get("artifacts", []):
            path = _artifact_path(artifact)
            access = artifact.get("access")
            should_verify = access == "public" or verify_live
            if not should_verify:
                continue
            if path is None or not path.is_file():
                errors.append(f"{location}: unavailable {access} artifact {artifact.get('repository_path')}")
                continue
            if sha256_file(path) != artifact.get("sha256"):
                errors.append(f"{location}: artifact hash drift {artifact.get('repository_path')}")
            if artifact.get("bytes") is not None and path.stat().st_size != artifact.get("bytes"):
                errors.append(f"{location}: artifact size drift {artifact.get('repository_path')}")

    try:
        migrated_story = _legacy_story(payload)
        snapshot_hint = payload["views"]["migration"]["snapshot"]
        snapshot_story = _load_yaml(ROOT / snapshot_hint)
        if migrated_story != snapshot_story:
            errors.append("migration reconciliation: normalized records do not reproduce the retained v1 story")
    except (KeyError, TypeError, RegistryError) as exc:
        errors.append(f"migration reconciliation failed: {exc}")

    if errors:
        raise RegistryError("\n".join(errors))
    return warnings


def _paper_relative(repository_path: str) -> str:
    return Path(os.path.relpath(ROOT / repository_path, PAPER)).as_posix()


def _legacy_story(payload: dict[str, Any]) -> dict[str, Any]:
    compatibility = payload["compatibility"]
    claims = {row["legacy_id"]: row for row in payload["claims"] if row.get("legacy_id")}
    experiments = {row["legacy_id"]: row for row in payload["experiments"] if row.get("legacy_id")}
    decisions = {row["id"]: row for row in payload["decisions"]}
    capsules = {row["id"]: row for row in payload["evidence_capsules"]}
    rendered_claims = []
    for legacy_id in compatibility["claim_order"]:
        row = claims[legacy_id]
        rendered_claims.append(
            {
                "id": legacy_id,
                "statement": row["statement"],
                "status": row["legacy_status"],
                "scope": row["scope"],
                "evidence": compatibility["claim_evidence"][legacy_id],
                "evidence_capsules": [
                    f"../../research/evidence/{capsules[evidence_id]['experiment_id']}.json"
                    for evidence_id in row["evidence_capsule_ids"]
                ],
            }
        )
    rendered_experiments = []
    for legacy_id in compatibility["experiment_order"]:
        row = experiments[legacy_id]
        evidence = capsules[row["evidence_capsule_id"]]
        decision_rows = [decisions[item] for item in row["decision_ids"]]
        rendered_experiments.append(
            {
                "id": legacy_id,
                "question": row["question"],
                "design": row["design"]["summary"],
                "status": row["legacy_status"],
                "finding": evidence["summary"],
                "limitation": evidence["boundaries"]["limitations"][0],
                "artifacts": [_paper_relative(item["repository_path"]) for item in evidence["artifacts"]],
                "evidence_capsule": f"../../research/evidence/{row['id']}.json",
                "next_decision": decision_rows[0]["rationale"],
            }
        )
    rendered_planned = []
    for legacy_id in compatibility["planned_order"]:
        row = experiments[legacy_id]
        migration = compatibility["planned"][legacy_id]
        rendered_planned.append(
            {
                "id": legacy_id,
                "priority": migration["priority"],
                "objective": row["objective"],
                **({"plan": migration["plan"]} if migration.get("plan") else {}),
            }
        )
    return {
        "schema_version": 1,
        "canonical_document": compatibility["canonical_document"],
        "documents": compatibility["documents"],
        "claims": rendered_claims,
        "experiments": rendered_experiments,
        "next_experiments": rendered_planned,
    }


def _paper_story(payload: dict[str, Any]) -> dict[str, Any]:
    """Extend the frozen migration view with bounded native run history.

    `_legacy_story` remains an exact reconstruction of the retained v1
    snapshot. The Overleaf compatibility view may additionally expose native
    engineering executions, provided their draft/evidence boundaries remain
    explicit and no evidence capsule is manufactured.
    """
    story = _legacy_story(payload)
    decisions = {row["id"]: row for row in payload["decisions"]}
    runs = {row["id"]: row for row in payload["runs"]}
    for row in _ordered_experiments(payload):
        if row["origin"]["kind"] != "native":
            continue
        executed_runs = [
            runs[run_id]
            for run_id in row.get("run_ids", [])
            if runs[run_id]["lifecycle"] in {"succeeded", "failed"}
        ]
        if not executed_runs or row.get("evidence_capsule_id"):
            continue
        decision_rows = [decisions[item] for item in row.get("decision_ids", [])]
        latest_decision = decision_rows[-1] if decision_rows else None
        run_states = "_then_".join(run["lifecycle"] for run in executed_runs)
        artifact_paths = [row["design"]["protocol_path"]]
        for run in executed_runs:
            artifact_paths.extend(
                [
                    f"research/registry/runs/{run['id']}.yaml",
                    run["configuration"]["path"],
                    run["artifacts"]["manifest_path"],
                ]
            )
        if latest_decision:
            artifact_paths.append(
                f"research/registry/decisions/{latest_decision['id']}.yaml"
            )
        story["experiments"].append(
            {
                "id": row["id"],
                "question": row["question"],
                "design": row["design"]["summary"],
                "status": (
                    f"{row['lifecycle']}_{row['outcome']}_engineering_run_history_"
                    f"{run_states}; evidence_tier_{row['evidence_tier']}"
                ),
                "finding": (
                    latest_decision["evidence_summary"]
                    if latest_decision
                    else "A bounded engineering run is registered; no scientific outcome was evaluated."
                ),
                "limitation": (
                    "This is non-claim-bearing engineering history: scientific audit and promotion "
                    "remain unresolved, evidence tier is none, and no evidence capsule exists."
                ),
                "artifacts": list(
                    dict.fromkeys(_paper_relative(item) for item in artifact_paths)
                ),
                "next_decision": (
                    latest_decision["rationale"]
                    if latest_decision
                    else "Retain the registered draft state pending an explicit decision."
                ),
            }
        )
    return story


def _ordered_experiments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(payload["experiments"], key=lambda row: int(row["id"].rsplit("-", 1)[1]))


def _planned_queue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Order drafts without manufacturing priority for new native records.

    The retained v1 story contains an explicit priority for its eight planned
    experiments. A native scaffold has no such scientific ranking, so it is
    placed after that compatibility queue and ordered deterministically by its
    creation date and registry ID.
    """
    legacy_priorities = payload.get("compatibility", {}).get("planned", {})

    def sort_key(row: dict[str, Any]) -> tuple[int, int | str, str]:
        legacy = legacy_priorities.get(row.get("legacy_id"))
        if isinstance(legacy, dict) and isinstance(legacy.get("priority"), int):
            return (0, legacy["priority"], row["id"])
        return (1, str(row.get("created_on", "")), row["id"])

    return sorted(
        (row for row in payload["experiments"] if row["lifecycle"] == "draft"),
        key=sort_key,
    )


def _project_story(payload: dict[str, Any]) -> str:
    programs = sorted(payload["programs"], key=lambda row: row["id"])
    claims = sorted(payload["claims"], key=lambda row: row["id"])
    experiments = _ordered_experiments(payload)
    completed = [row for row in experiments if row.get("evidence_capsule_id")]
    planned = [row for row in experiments if row["lifecycle"] == "draft"]
    flagship = next(row for row in programs if row["id"] == payload["index"]["flagship_program_id"])
    priority = payload["compatibility"]["planned"]
    rows = [
        "# NeuRev research story",
        "",
        "Generated from the repository research registry; do not edit by hand.",
        "",
        "NeuRev turns difficult neural-imaging movies into auditable evidence: it builds interpretable measurements, proposes candidate activity, exposes failure modes, supports blinded review, and records which conclusions have or have not survived validation.",
        "",
        "## Programs",
        "",
        "| Program | Lifecycle | Registered scope |",
        "| --- | --- | --- |",
    ]
    rows += [f"| {row['title']} | {row['lifecycle']} | {row['scientific_scope']} |" for row in programs]
    rows += [
        "",
        "## Registry snapshot",
        "",
        f"- {len(claims)} atomic claims",
        f"- {len(completed)} completed historical experiments with portable evidence capsules",
        f"- {len(planned)} planned experiments",
        f"- {len(payload['runs'])} registered run records",
        "",
        "Historical v1 experiments intentionally have no fabricated run records; their capsules preserve results, hashes, and explicit provenance gaps.",
        "",
        "## Current flagship boundary",
        "",
        flagship["scientific_scope"],
        "",
        "Unresolved boundaries:",
        "",
    ]
    rows += [f"- {item}" for item in flagship["boundaries"]]
    rows += ["", "## Current decision queue", ""]
    for row in _planned_queue(payload):
        legacy = priority.get(row.get("legacy_id"))
        if isinstance(legacy, dict) and isinstance(legacy.get("priority"), int):
            rows.append(f"{legacy['priority']}. **{row['title']}:** {row['objective']}")
        else:
            rows.append(
                f"- **{row['title']}:** {row['objective']} "
                "_(registered draft; scientific priority not assigned)_"
            )
    rows += [
        "",
        "See [the claim ledger](CLAIM_LEDGER.md) and [experiment timeline](EXPERIMENT_TIMELINE.md) for the complete generated record.",
        "",
    ]
    return "\n".join(rows)


def _timeline(payload: dict[str, Any]) -> str:
    decisions = {row["id"]: row for row in payload["decisions"]}
    rows = [
        "# Experiment timeline",
        "",
        "Generated from the repository research registry. Order is explicit; historical execution dates and run provenance remain unknown when absent from the v1 source.",
        "",
        "| ID | Experiment | Lifecycle | Outcome | Evidence | Decision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in _ordered_experiments(payload):
        actions = [decisions[item]["action"] for item in row.get("decision_ids", [])]
        rows.append(
            f"| `{row['id']}` | {row['title']} | {row['lifecycle']} | {row['outcome']} | {row['evidence_tier']} | {', '.join(actions) or 'pending'} |"
        )
    rows.append("")
    return "\n".join(rows)


def _claim_ledger(payload: dict[str, Any]) -> str:
    experiments = _ordered_experiments(payload)
    linked: dict[str, list[str]] = {row["id"]: [] for row in payload["claims"]}
    for experiment in experiments:
        for claim_id in experiment.get("claim_ids", []):
            linked[claim_id].append(experiment["id"])
    rows = [
        "# Claim ledger",
        "",
        "Generated from the repository research registry; every scope and limitation is part of the claim.",
        "",
        "| ID | Claim | State | Evidence tier | Linked experiments |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(payload["claims"], key=lambda item: item["id"]):
        experiments_text = ", ".join(f"`{item}`" for item in linked[row["id"]]) or "none"
        rows.append(
            f"| `{row['id']}` | {row['statement']} **Scope:** {row['scope']} **Limit:** {row['limitations'][0]} | {row['claim_state']} | {row['evidence_tier']} | {experiments_text} |"
        )
    rows.append("")
    return "\n".join(rows)


def _llm_context(payload: dict[str, Any]) -> dict[str, Any]:
    planned_priority = payload["compatibility"]["planned"]
    return {
        "schema_version": 1,
        "authority": "research/registry",
        "generated_policy": "do_not_edit",
        "identity": {
            "repository": "NeuRev Workbench",
            "python_package": "neurobench",
            "flagship_program_id": payload["index"]["flagship_program_id"],
        },
        "programs": [
            {
                "id": row["id"],
                "slug": row["slug"],
                "title": row["title"],
                "lifecycle": row["lifecycle"],
                "scientific_scope": row["scientific_scope"],
                "boundaries": row["boundaries"],
                "documentation": row.get("documentation", []),
            }
            for row in sorted(payload["programs"], key=lambda item: item["id"])
        ],
        "counts": {
            "claims": len(payload["claims"]),
            "completed_experiments": sum(bool(row.get("evidence_capsule_id")) for row in payload["experiments"]),
            "planned_experiments": sum(row["lifecycle"] == "draft" for row in payload["experiments"]),
            "runs": len(payload["runs"]),
            "evidence_capsules": len(payload["evidence_capsules"]),
        },
        "claims": [
            {key: row[key] for key in ("id", "statement", "scope", "limitations", "claim_state", "evidence_tier", "review_state")}
            for row in sorted(payload["claims"], key=lambda item: item["id"])
        ],
        "next_experiments": [
            {
                "id": row["id"],
                "priority": (
                    planned_priority[row["legacy_id"]]["priority"]
                    if row.get("legacy_id") in planned_priority
                    else None
                ),
                "objective": row["objective"],
                "lifecycle": row["lifecycle"],
                **(
                    {}
                    if row.get("legacy_id") in planned_priority
                    else {"queue_basis": "registered_draft_without_assigned_priority"}
                ),
            }
            for row in _planned_queue(payload)
        ],
        "rules": [
            "Do not infer biological identity from a stable component or detected center.",
            "Do not treat sparse unmatched candidates as verified negatives.",
            "Do not promote a claim from run completion alone.",
            "Do not hand-edit generated research-story views.",
            "Do not require ignored Inputs or Outputs for clean-clone validation.",
        ],
    }


def _navigation(payload: dict[str, Any]) -> dict[str, Any]:
    flagship = next(row for row in payload["programs"] if row["id"] == payload["index"]["flagship_program_id"])
    program_paths = {row["slug"]: f"research/registry/programs/{row['slug']}.yaml" for row in payload["programs"]}
    return {
        "schema_version": 2,
        "generated_from": "research/registry",
        "generated_policy": "do_not_edit",
        "repository": {
            "name": "NeuRev-Workbench",
            "product": "NeuRev Workbench",
            "python_package": "neurobench",
            "checkout_rule": "Use the directory containing AGENTS.md; verify git status and do not infer identity from an external data path.",
            "python": ".venv-neurobench/bin/python",
            "maintained_code": "neurobench",
            "local_artifacts": "Outputs",
            "portable_evidence": "research/evidence",
        },
        "routes": [
            {
                "id": "project_story",
                "human_start": "research/generated/PROJECT_STORY.md",
                "authority": "research/registry",
                "compiled_view": "research/generated/canonical.json",
                "machine_context": "research/generated/llm_context.json",
                "validation": [".venv-neurobench/bin/python -m neurobench.research.registry check"],
            },
            {
                "id": "current_neuron_identifiability_state",
                "human_start": "paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md",
                "authority": program_paths["neuron-identifiability"],
                "implementation": "neurobench/experiments/neuron_identifiability",
                "validation": ["make -C paper/overleaf_jnm story-check"],
            },
            {
                "id": "experiment_timeline",
                "human_start": "research/generated/EXPERIMENT_TIMELINE.md",
                "authority": "research/registry/index.yaml",
                "validation": [".venv-neurobench/bin/python -m neurobench.research.registry check"],
            },
            {
                "id": "claim_ledger",
                "human_start": "research/generated/CLAIM_LEDGER.md",
                "authority": "research/registry/claims",
                "validation": [".venv-neurobench/bin/python -m neurobench.research.registry check"],
            },
            {
                "id": "scientific_experiment",
                "human_start": "docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md",
                "authority": "docs/decisions/0001-repository-wide-research-registry.md",
                "implementation": "neurobench/experiments",
                "validation": ["experiment-specific tests", "validation.json", "portable evidence capsule"],
            },
            {
                "id": "local_candidate_review",
                "human_start": "docs/HOW_TO_USE_DASHBOARD.md",
                "authority": "docs/NEURON_WORKBENCH.md",
                "implementation": "neurobench/workbench",
                "validation": ["workbench-focused tests", "browser render smoke when enabled"],
            },
            {
                "id": "maintained_codebase",
                "human_start": "docs/CODEBASE_NAVIGATION.md",
                "authority": "neurobench",
                "validation": ["focused package tests", "full portable test suite"],
            },
            {
                "id": "publication_release",
                "human_start": "docs/RELEASE_PROCESS.md",
                "authority": "docs/PUBLICATION_BOUNDARY.md",
                "validation": [
                    ".venv-neurobench/bin/python tools/audit_publication_boundary.py --fail-on warning",
                    ".venv-neurobench/bin/python -m neurobench.research.registry check",
                ],
            },
            {
                "id": "external_bounded_review_v2",
                "human_start": "docs/workflows/external_blinded_bounded_review_v2.md",
                "authority": "research/evidence/NREV-EXP-0017.json",
                "local_artifacts": "Outputs/NeuronIdentifiability/external_blinded_bounded_review_v2",
                "implementation": "neurobench/experiments/neuron_identifiability/external_bounded_review.py",
                "status": "implementation_complete_annotations_pending",
            },
            {
                "id": "fish_inverse_control",
                "human_start": "docs/programs/fish_inverse_control/README.md",
                "authority": program_paths["fish-inverse-control"],
                "implementation": "neurobench/programs",
            },
            {
                "id": "grid_dynamics",
                "human_start": "docs/GRID_LATENT_DYNAMICS.md",
                "authority": program_paths["grid-dynamics"],
                "implementation": "neurobench/dynamics",
            },
        ],
        "scientific_boundaries": {
            **flagship.get("study_scope", {}),
            "unchanged_claims": flagship["boundaries"],
        },
        "generated_files": {
            "policy": "do_not_edit",
            "source": "research/registry",
            "paths": [
                "research/generated/canonical.json",
                "research/generated/PROJECT_STORY.md",
                "research/generated/EXPERIMENT_TIMELINE.md",
                "research/generated/CLAIM_LEDGER.md",
                "research/generated/llm_context.json",
                "paper/overleaf_jnm/story/research_story.yaml",
                "paper/overleaf_jnm/STORY_INDEX.md",
                "paper/overleaf_jnm/EXPERIMENT_STORY.md",
                "paper/overleaf_jnm/generated/story_status_table.tex",
                "paper/overleaf_jnm/generated/experiment_story.tex",
                "docs/navigation.json",
                "llms.txt",
            ],
        },
        "generated_regions": {
            "policy": "edit_registry_source_and_rebuild",
            "source": "research/registry",
            "regions": [
                {
                    "path": "README.md",
                    "begin_marker": "<!-- BEGIN GENERATED RESEARCH SNAPSHOT -->",
                    "end_marker": "<!-- END GENERATED RESEARCH SNAPSHOT -->",
                },
                {
                    "path": "docs/REPOSITORY_GUIDE.md",
                    "begin_marker": "<!-- BEGIN GENERATED FLAGSHIP BOUNDARY -->",
                    "end_marker": "<!-- END GENERATED FLAGSHIP BOUNDARY -->",
                },
            ],
        },
    }


def _llms_text(_: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# NeuRev Workbench",
            "",
            "> Generated from `research/registry/`; do not edit by hand.",
            "",
            "> Evidence-first calcium-imaging source separation, neuron detection, scientific audit, and blinded review.",
            "",
            "## Canonical routes",
            "",
            "- [Repository research story](research/generated/PROJECT_STORY.md)",
            "- [Machine research context](research/generated/llm_context.json)",
            "- [Claim ledger](research/generated/CLAIM_LEDGER.md)",
            "- [Experiment timeline](research/generated/EXPERIMENT_TIMELINE.md)",
            "- [Repository guide](docs/REPOSITORY_GUIDE.md)",
            "- [Scientific audit standard](docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md)",
            "- [Maintained implementation](neurobench/)",
            "",
            "## Authority order",
            "",
            "1. Immutable manifests, evidence capsules, hashes, and validation records.",
            "2. Experiment protocols and decision records.",
            "3. Canonical records under `research/registry/`.",
            "4. Generated research-story and navigation views.",
            "5. Meetings, presentations, historical plans, and local output prose.",
            "",
            "Large artifacts under `Inputs/` and `Outputs/` are local and are not public source authority. Read compact JSON indexes before media.",
            "",
        ]
    )


def _readme_snapshot(payload: dict[str, Any]) -> str:
    programs = sorted(payload["programs"], key=lambda row: row["id"])
    flagship = next(row for row in programs if row["id"] == payload["index"]["flagship_program_id"])
    scope = flagship["study_scope"]
    links = {
        "neuron-identifiability": "paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md",
        "source-separation": "docs/research/README.md",
        "fish-inverse-control": "docs/programs/fish_inverse_control/README.md",
        "grid-dynamics": "docs/GRID_LATENT_DYNAMICS.md",
    }
    rows = [
        "## Current research boundary",
        "",
        f"The flagship neuron-identifiability program currently studies **{scope['occurrences']} confirmed occurrences at {scope['sites']} immutable sites in {scope['recordings']} recording**. Within that boundary, NeuRev has tested complete-trace features, identity-aware failure modes, truth-known movie stress tests, detector calibration, deblending alternatives, and a staged blinded-review package.",
        "",
        "The current evidence still leaves three boundaries unresolved:",
        "",
    ]
    rows += [f"- {item}" for item in flagship["boundaries"]]
    rows += [
        "",
        "Those are registered open questions, not footnotes. See the generated",
        "[claim ledger](research/generated/CLAIM_LEDGER.md),",
        "[experiment timeline](research/generated/EXPERIMENT_TIMELINE.md), and",
        "[current manuscript state](paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md).",
        "",
        "## What lives here",
        "",
        "| Program | Purpose | Lifecycle |",
        "| --- | --- | --- |",
    ]
    for row in programs:
        rows.append(f"| [{row['title']}]({links[row['slug']]}) | {row['summary']} | {row['lifecycle']} |")
    return "\n".join(rows)


def _replace_generated_block(path: Path, begin: str, end: str, content: str) -> str:
    if not path.is_file():
        raise RegistryError(f"missing generated-view host: {path.relative_to(ROOT)}")
    text = path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RegistryError(f"generated markers must occur exactly once in {path.relative_to(ROOT)}")
    start = text.find(begin)
    finish = text.find(end)
    if start < 0 or finish < 0 or finish < start:
        raise RegistryError(f"missing or invalid generated markers in {path.relative_to(ROOT)}")
    finish += len(end)
    replacement = f"{begin}\n\n{content}\n\n{end}"
    return text[:start] + replacement + text[finish:]


def _guide_boundary(payload: dict[str, Any]) -> str:
    flagship = next(
        row for row in payload["programs"] if row["id"] == payload["index"]["flagship_program_id"]
    )
    rows = [
        "## Current flagship boundary",
        "",
        flagship["scientific_scope"],
        "",
        "The registered unresolved boundaries are:",
        "",
    ]
    rows += [f"- {item}" for item in flagship["boundaries"]]
    rows += [
        "",
        "The exact claim states and next-experiment queue are generated from canonical",
        "records; use the [claim ledger](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/CLAIM_LEDGER.md) rather",
        "than copying status prose into another document.",
    ]
    return "\n".join(rows)


def generated_outputs(payload: dict[str, Any]) -> dict[Path, str]:
    paper_story = "# Generated from research/registry; do not edit by hand.\n" + yaml.safe_dump(
        _paper_story(payload), sort_keys=False, allow_unicode=True, width=120
    )
    readme = ROOT / "README.md"
    guide = ROOT / "docs" / "REPOSITORY_GUIDE.md"
    return {
        readme: _replace_generated_block(
            readme,
            "<!-- BEGIN GENERATED RESEARCH SNAPSHOT -->",
            "<!-- END GENERATED RESEARCH SNAPSHOT -->",
            _readme_snapshot(payload),
        ),
        guide: _replace_generated_block(
            guide,
            "<!-- BEGIN GENERATED FLAGSHIP BOUNDARY -->",
            "<!-- END GENERATED FLAGSHIP BOUNDARY -->",
            _guide_boundary(payload),
        ),
        GENERATED / "canonical.json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
        GENERATED / "PROJECT_STORY.md": _project_story(payload),
        GENERATED / "EXPERIMENT_TIMELINE.md": _timeline(payload),
        GENERATED / "CLAIM_LEDGER.md": _claim_ledger(payload),
        GENERATED / "llm_context.json": json.dumps(_llm_context(payload), indent=2, sort_keys=True) + "\n",
        ROOT / "docs" / "navigation.json": json.dumps(_navigation(payload), indent=2, sort_keys=False) + "\n",
        ROOT / "llms.txt": _llms_text(payload),
        PAPER_STORY: paper_story,
    }


def build(*, check: bool = False, verify_live: bool = False) -> tuple[int, list[str]]:
    payload = compile_registry()
    warnings = validate_registry(payload, verify_live=verify_live)
    expected = generated_outputs(payload)
    stale = [path for path, text in expected.items() if not path.is_file() or path.read_text() != text]
    if check:
        if stale:
            raise RegistryError("stale generated research views: " + ", ".join(str(path.relative_to(ROOT)) for path in stale))
    else:
        for path, text in expected.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return len(stale), warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="validate records and regenerate synchronized views")
    subparsers.add_parser("check", help="fail when records are invalid or generated views are stale")
    live = subparsers.add_parser("verify-live", help="also reconcile metadata-only local artifacts against captured hashes")
    live.add_argument("--check", action="store_true", help="also require generated views to be current")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            stale, warnings = build()
            print(f"research registry built: refreshed {stale} stale views")
        elif args.command == "check":
            _, warnings = build(check=True)
            payload = compile_registry()
            print(
                f"research registry current: {len(payload['programs'])} programs, {len(payload['claims'])} claims, "
                f"{sum(bool(row.get('evidence_capsule_id')) for row in payload['experiments'])} completed experiments, "
                f"{sum(row['lifecycle'] == 'draft' for row in payload['experiments'])} planned experiments, "
                f"{len(payload['runs'])} registered run records"
            )
        else:
            _, warnings = build(check=args.check, verify_live=True)
            print("live artifact reconciliation passed")
    except RegistryError as exc:
        parser.error(str(exc))
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
