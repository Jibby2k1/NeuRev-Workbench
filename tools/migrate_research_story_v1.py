#!/usr/bin/env python3
"""Migrate the manuscript-owned v1 story into the repository registry.

The migration is semantics preserving. It normalizes compound legacy statuses
into orthogonal fields, retains a normalized semantic snapshot and every original status, and never invents a
historical run, execution date, preregistration gate, or falsifier. The exact
v1 story is retained as migration provenance and rebuilt from the normalized
records by ``neurobench.research.registry``.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.research.hashing import canonical_sha256, sha256_file  # noqa: E402

PAPER = ROOT / "paper" / "overleaf_jnm"
PAPER_STORY = PAPER / "story" / "research_story.yaml"
DEFAULT_RESEARCH_ROOT = ROOT / "research"
MIGRATION_DATE = "2026-08-29"
MIGRATION_INSTANT = "2026-08-29T23:52:35Z"
ORIGINAL_SOURCE = "paper/overleaf_jnm/story/research_story.yaml"
ORIGINAL_SOURCE_SHA256 = "fbd0889c305963a02199a012beccfe2fa405a296a851f2f62525c49931ddd029"
SNAPSHOT_PATH = "research/registry/migration/research_story_v1.yaml"
FLAGSHIP_PROGRAM_ID = "NREV-PRG-0001"


PROGRAMS: tuple[dict[str, Any], ...] = (
    {
        "schema_version": 1,
        "record_type": "program",
        "id": "NREV-PRG-0001",
        "slug": "neuron-identifiability",
        "title": "Neuron Identifiability",
        "summary": "Evidence-first measurement, representation, detection, identity auditing, and blinded review in calcium imaging.",
        "lifecycle": "active",
        "scientific_scope": "An intensive within-recording methods study of 106 confirmed occurrences at 50 immutable sites in one calcium-imaging recording.",
        "study_scope": {
            "recordings": 1,
            "sites": 50,
            "occurrences": 106,
            "description": "Confirmed occurrences at immutable registered centers in the current recording.",
        },
        "objectives": [
            "Determine which interpretable temporal and spatial measurements recover registered neural-activity occurrences.",
            "Expose detector failure modes and distinguish measurement recovery from biological source identity.",
            "Prepare bounded, blinded review and independent-recording tests without treating sparse positives as verified negatives.",
        ],
        "boundaries": [
            "Full-field precision is unresolved.",
            "One-to-one biological source identity is unresolved.",
            "Generalization to an independent recording is unresolved.",
        ],
        "owner_roles": ["research_team"],
        "related_program_ids": ["NREV-PRG-0002", "NREV-PRG-0003", "NREV-PRG-0004"],
        "documentation": [
            "paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md",
            "docs/workflows/spon_ca_burst_neuron_identifiability_paper.md",
            "docs/research/README.md",
        ],
        "visibility": "public",
        "created_on": MIGRATION_DATE,
        "updated_on": MIGRATION_DATE,
    },
    {
        "schema_version": 1,
        "record_type": "program",
        "id": "NREV-PRG-0002",
        "slug": "source-separation",
        "title": "Source Separation and Representation",
        "summary": "Interpretable temporal, spatial, and multiscale representations for separating neural signal, structured artifact, and measurement noise.",
        "lifecycle": "active",
        "scientific_scope": "Foundational representation and source-separation methods used by the repository's imaging workflows.",
        "objectives": [
            "Develop interpretable representations that preserve event structure while reducing measurement noise.",
            "Audit numerical behavior and biological interpretation separately.",
        ],
        "boundaries": [
            "Latent or independent components do not by themselves establish one-to-one biological identity.",
            "Representation gains have not been established on independent recordings across the repository.",
            "Full-field specificity requires bounded exhaustive labels.",
        ],
        "related_program_ids": ["NREV-PRG-0001", "NREV-PRG-0004"],
        "documentation": ["docs/research/README.md", "docs/workflows/README.md"],
        "visibility": "public",
        "created_on": MIGRATION_DATE,
        "updated_on": MIGRATION_DATE,
    },
    {
        "schema_version": 1,
        "record_type": "program",
        "id": "NREV-PRG-0003",
        "slug": "fish-inverse-control",
        "title": "Fish Intent and Inverse Control",
        "summary": "Stage-gated measurement, intent decoding, action-conditioned system identification, simulation, and safety-bounded control research.",
        "lifecycle": "draft",
        "scientific_scope": "A downstream causal program that may proceed only after measurement, leakage, intervention, and safety gates are satisfied.",
        "objectives": [
            "Test pre-movement intent under leakage-resistant validation.",
            "Estimate action-conditioned dynamics from measured interventions before considering control.",
        ],
        "boundaries": [
            "Causal pre-movement intent has not been established.",
            "Action-conditioned dynamics have not been established from measured interventions.",
            "No command-capable deployment is authorized outside a validated safety envelope.",
        ],
        "related_program_ids": ["NREV-PRG-0001", "NREV-PRG-0004"],
        "documentation": [
            "docs/programs/fish_inverse_control/README.md",
            "docs/research/FISH_INVERSE_CONTROL_ROADMAP.md",
        ],
        "visibility": "public",
        "legacy_status": "gated",
        "created_on": MIGRATION_DATE,
        "updated_on": MIGRATION_DATE,
    },
    {
        "schema_version": 1,
        "record_type": "program",
        "id": "NREV-PRG-0004",
        "slug": "grid-dynamics",
        "title": "Grid and Latent Dynamics",
        "summary": "Template-aligned neural-state grids, latent representations, forecasting baselines, and video-level classifiers.",
        "lifecycle": "active",
        "scientific_scope": "Measurement and forecasting methods for template-aligned neural state, without causal or control interpretation.",
        "objectives": [
            "Compare transparent and learned forecasting baselines on identity-safe neural-state grids.",
            "Separate passive forecast quality from downstream causal or control claims.",
        ],
        "boundaries": [
            "Independent-video generalization of learned latent dynamics is unresolved.",
            "Passive neural-state forecasts do not establish causal interpretation.",
            "Predicted state has not been shown to invert into a safe control action.",
        ],
        "related_program_ids": ["NREV-PRG-0001", "NREV-PRG-0002", "NREV-PRG-0003"],
        "documentation": ["docs/GRID_LATENT_DYNAMICS.md", "docs/TEMPLATE_GRID_WORKFLOW.md"],
        "visibility": "public",
        "created_on": MIGRATION_DATE,
        "updated_on": MIGRATION_DATE,
    },
)


def display_title(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").title()


def dump_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def repository_path_from_story(value: str) -> tuple[Path, str]:
    path = (PAPER / value).resolve()
    try:
        relative = path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"v1 story artifact escapes the repository: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"v1 story artifact was unavailable during migration: {relative}")
    return path, relative.as_posix()


def artifact_role(relative: str) -> str:
    path = Path(relative)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if "summary" in name:
        return "summary"
    if "manifest" in name or "index" in name:
        return "manifest"
    if suffix in {".png", ".pdf", ".svg"}:
        return "figure"
    if suffix in {".tsv", ".csv"}:
        return "table"
    if suffix in {".md", ".tex", ".json"}:
        return "report"
    if suffix in {".npz", ".npy", ".parquet", ".h5", ".hdf5"}:
        return "data"
    return "other"


def media_type(path: Path) -> str:
    known = {".md": "text/markdown", ".tsv": "text/tab-separated-values", ".npz": "application/x-npz"}
    return known.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def capsule_artifacts(values: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values, 1):
        path, relative = repository_path_from_story(value)
        rows.append(
            {
                "id": f"artifact-{index:02d}",
                "role": artifact_role(relative),
                "media_type": media_type(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "access": "metadata_only" if relative.startswith("Outputs/") else "public",
                "repository_path": relative,
            }
        )
    return rows


def capsule_id(experiment_id: str) -> str:
    return experiment_id.replace("NREV-", "NREV-EVC-", 1)


def experiment_id_from_capsule_path(value: str) -> str:
    name = Path(value).name
    if not name.endswith(".json"):
        raise ValueError(f"invalid evidence-capsule path in v1 story: {value}")
    return name.removesuffix(".json")


def claim_axes(status: str) -> tuple[str, str, str]:
    if status == "unresolved":
        return "unresolved", "none", "not_requested"
    if status.startswith("provisional"):
        return "provisional", "reviewed_current_recording", "in_review"
    if status.startswith("implementation"):
        return "provisional", "descriptive", "pending"
    if status.startswith("exploratory"):
        return "provisional", "current_recording", "not_requested"
    tier = "current_recording"
    review = "not_requested"
    if "computational_simulation" in status or "computational_negative" in status or "computational_mixed" in status:
        tier = "computational_simulation"
    elif "reviewed" in status:
        tier = "reviewed_current_recording"
        review = "in_review"
    return "supported", tier, review


def experiment_axes(legacy_id: str, status: str) -> tuple[str, str, str, str]:
    simulation_ids = {
        "major_validation_next_steps_v1",
        "realistic_movie_detector_benchmark_v3",
        "generator_family_holdout_v4",
        "label_free_false_alarm_gated_fusion_v5",
        "automated_challenge_suite_v7",
        "targeted_automated_development_v8",
        "joint_generative_deblending_v10",
    }
    if "negative_result" in status:
        return "closed", "rejected", "computational_simulation", "not_requested"
    if "mixed_result" in status:
        return "closed", "mixed", "computational_simulation", "not_requested"
    if "provisional_single_reviewer" in status:
        return "reviewed", "inconclusive", "reviewed_current_recording", "in_review"
    if "reviewed_interpretation" in status:
        return "reviewed", "supported", "reviewed_current_recording", "in_review"
    if "implementation_human_gated" in status:
        return "validated", "not_evaluated", "descriptive", "pending"
    if "human_gated" in status:
        return "validated", "mixed", "evidence_synthesis", "pending"
    if "descriptive" in status or "exploratory" in status:
        return "validated", "inconclusive", "current_recording", "not_requested"
    tier = "computational_simulation" if legacy_id in simulation_ids else "current_recording"
    return "validated", "supported", tier, "not_requested"


def decision_action(status: str) -> str:
    if "human_gated" in status or "provisional_single_reviewer" in status:
        return "hold"
    if "negative_result" in status:
        return "stop"
    return "retain"


def claim_limitations(status: str, scope: str) -> list[str]:
    if status == "unresolved":
        return ["The cited evidence does not establish the registered claim."]
    return [f"No inference is registered beyond this scope: {scope}"]


def claim_links(story: dict[str, Any], experiment_ids: dict[str, str]) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    for claim in story["claims"]:
        explicit = [experiment_id_from_capsule_path(path) for path in claim.get("evidence_capsules", [])]
        if explicit:
            links[claim["id"]] = explicit
            continue
        evidence = set(claim["evidence"])
        links[claim["id"]] = [
            experiment_ids[item["id"]]
            for item in story["experiments"]
            if evidence.intersection(item["artifacts"])
        ]
    return links


def build_records(story: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    completed = story["experiments"]
    planned = story["next_experiments"]
    experiment_ids = {row["id"]: f"NREV-EXP-{index:04d}" for index, row in enumerate(completed, 1)}
    planned_ids = {
        row["id"]: f"NREV-EXP-{index:04d}" for index, row in enumerate(planned, len(completed) + 1)
    }
    claim_ids = {row["id"]: f"NREV-CLM-{index:04d}" for index, row in enumerate(story["claims"], 1)}
    links = claim_links(story, experiment_ids)
    reverse_claims: dict[str, list[str]] = {identifier: [] for identifier in experiment_ids.values()}
    for legacy_claim_id, linked in links.items():
        for experiment_id in linked:
            if experiment_id not in reverse_claims:
                raise ValueError(f"claim {legacy_claim_id} links unknown experiment {experiment_id}")
            reverse_claims[experiment_id].append(claim_ids[legacy_claim_id])

    decisions_by_experiment = {
        experiment_ids[row["id"]]: f"NREV-DEC-{index:04d}" for index, row in enumerate(completed, 1)
    }
    experiment_origin = {
        "kind": "migrated_legacy",
        "source_path": SNAPSHOT_PATH,
        "source_sha256": source_sha256,
        "recorded_on": MIGRATION_DATE,
        "missing_fields": [
            "historical execution date",
            "run-level code, configuration, input, environment, and randomization provenance",
            "preregistered gates",
            "experiment-specific falsifiers",
            "normalized unit of analysis, comparison, and data scope",
        ],
    }
    decision_origin = {
        "kind": "migrated_legacy",
        "source_path": SNAPSHOT_PATH,
        "source_sha256": source_sha256,
        "recorded_on": MIGRATION_DATE,
        "missing_fields": ["historical decision effective date"],
    }

    claims: list[dict[str, Any]] = []
    for item in story["claims"]:
        state, tier, review = claim_axes(item["status"])
        linked_experiments = links[item["id"]]
        claims.append(
            {
                "schema_version": 1,
                "record_type": "claim",
                "id": claim_ids[item["id"]],
                "program_id": FLAGSHIP_PROGRAM_ID,
                "title": display_title(item["id"]),
                "statement": item["statement"],
                "scope": item["scope"],
                "limitations": claim_limitations(item["status"], item["scope"]),
                "claim_state": state,
                "evidence_tier": tier,
                "review_state": review,
                "evidence_capsule_ids": [capsule_id(identifier) for identifier in linked_experiments],
                "decision_ids": [decisions_by_experiment[identifier] for identifier in linked_experiments],
                "tags": [part for part in item["id"].replace("_", "-").split("-") if len(part) > 1],
                "visibility": "public",
                "legacy_id": item["id"],
                "legacy_status": item["status"],
                "created_on": MIGRATION_DATE,
                "updated_on": MIGRATION_DATE,
            }
        )

    experiments: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    capsules: list[dict[str, Any]] = []
    claims_by_id = {row["id"]: row for row in claims}

    for index, item in enumerate(completed, 1):
        experiment_id = experiment_ids[item["id"]]
        decision_id = decisions_by_experiment[experiment_id]
        evidence_id = capsule_id(experiment_id)
        lifecycle, outcome, tier, review = experiment_axes(item["id"], item["status"])
        experiment = {
            "schema_version": 1,
            "record_type": "experiment",
            "id": experiment_id,
            "origin": experiment_origin,
            "program_id": FLAGSHIP_PROGRAM_ID,
            "title": display_title(item["id"]),
            "question": item["question"],
            "objective": item["question"],
            "rationale": "Migrated to preserve the v1 question-design-result-decision record; the v1 story did not encode a separate rationale.",
            "lifecycle": lifecycle,
            "outcome": outcome,
            "evidence_tier": tier,
            "review_state": review,
            "falsifiers": [],
            "design": {"summary": item["design"]},
            "gates": [],
            "claim_ids": reverse_claims[experiment_id],
            "decision_ids": [decision_id],
            "evidence_capsule_id": evidence_id,
            "visibility": "public",
            "legacy_id": item["id"],
            "legacy_status": item["status"],
            "created_on": MIGRATION_DATE,
            "updated_on": MIGRATION_DATE,
        }
        experiments.append(experiment)

        decision = {
            "schema_version": 1,
            "record_type": "decision",
            "id": decision_id,
            "origin": decision_origin,
            "program_id": FLAGSHIP_PROGRAM_ID,
            "title": f"Decision after {display_title(item['id'])}",
            "lifecycle": "active",
            "action": decision_action(item["status"]),
            "rationale": item["next_decision"],
            "evidence_summary": item["finding"],
            "review_state": review,
            "experiment_ids": [experiment_id],
            "claim_ids": reverse_claims[experiment_id],
            "evidence_capsule_ids": [evidence_id],
            "constraints": [
                item["limitation"],
                "The historical decision date was not encoded in the v1 story and was not guessed during migration.",
            ],
            "visibility": "public",
            "legacy_status": item["status"],
            "created_on": MIGRATION_DATE,
            "updated_on": MIGRATION_DATE,
        }
        decisions.append(decision)

        artifacts = capsule_artifacts(item["artifacts"])
        claim_impacts = []
        for claim_id in reverse_claims[experiment_id]:
            claim = claims_by_id[claim_id]
            unresolved = claim["claim_state"] == "unresolved"
            claim_impacts.append(
                {
                    "claim_id": claim_id,
                    "effect": "qualifies" if unresolved else "supports",
                    "summary": (
                        "The experiment bears on this registered claim without resolving it."
                        if unresolved
                        else "The experiment supports this claim only within its registered scope."
                    ),
                    "scope": claim["scope"],
                }
            )
        source_records = {
            "experiment": experiment,
            "decisions": [decision],
            "claims": sorted((claims_by_id[x] for x in reverse_claims[experiment_id]), key=lambda row: row["id"]),
        }
        capsules.append(
            {
                "schema_version": 1,
                "record_type": "evidence_capsule",
                "id": evidence_id,
                "experiment_id": experiment_id,
                "program_id": FLAGSHIP_PROGRAM_ID,
                "generated_at": MIGRATION_INSTANT,
                "lifecycle": "validated",
                "outcome": outcome,
                "evidence_tier": tier,
                "review_state": review,
                "summary": item["finding"],
                "run_ids": [],
                "metrics": [],
                "gates": [
                    {
                        "id": "legacy-artifact-capture",
                        "criterion": "Every artifact referenced by the v1 experiment story is available and checksum identified at migration.",
                        "status": "passed",
                        "observed": f"Captured {len(artifacts)} of {len(item['artifacts'])} referenced artifacts with SHA-256 identities.",
                    }
                ],
                "claim_impacts": claim_impacts,
                "artifacts": artifacts,
                "artifact_manifest_sha256": canonical_sha256(artifacts),
                "validation": {
                    "status": "partial",
                    "checked_at": MIGRATION_INSTANT,
                    "checks": [
                        {
                            "id": "portable-paths",
                            "status": "passed",
                            "summary": "Artifact references are repository-relative; ignored local outputs are metadata-only.",
                        },
                        {
                            "id": "historical-provenance",
                            "status": "not_run",
                            "summary": "The v1 story did not encode run-level provenance, so no historical run record was invented.",
                        },
                    ],
                },
                "boundaries": {
                    "scope": "This capsule preserves the registered v1 experiment finding and its linked claim scopes.",
                    "limitations": [item["limitation"]],
                    "does_not_establish": [
                        "Claims beyond the registered finding and linked claim scopes.",
                        "Reproducible run provenance that was absent from the v1 story.",
                    ],
                },
                "source_records_sha256": canonical_sha256(source_records),
                "visibility": "public",
                "legacy_status": item["status"],
            }
        )

    for offset, item in enumerate(planned, len(completed) + 1):
        experiment_id = planned_ids[item["id"]]
        design: dict[str, Any] = {"summary": item["objective"]}
        if item.get("plan"):
            _, protocol_path = repository_path_from_story(item["plan"])
            design["protocol_path"] = protocol_path
        experiments.append(
            {
                "schema_version": 1,
                "record_type": "experiment",
                "id": experiment_id,
                "origin": experiment_origin,
                "program_id": FLAGSHIP_PROGRAM_ID,
                "title": display_title(item["id"]),
                "question": item["objective"],
                "objective": item["objective"],
                "rationale": f"Priority {item['priority']} item in the migrated v1 next-experiment queue.",
                "lifecycle": "draft",
                "outcome": "not_evaluated",
                "evidence_tier": "none",
                "review_state": "not_requested",
                "falsifiers": [],
                "design": design,
                "gates": [],
                "claim_ids": [],
                "decision_ids": [],
                "visibility": "public",
                "legacy_id": item["id"],
                "legacy_status": "next_experiment",
                "created_on": MIGRATION_DATE,
                "updated_on": MIGRATION_DATE,
            }
        )

    compatibility = {
        "schema_version": 1,
        "canonical_document": story["canonical_document"],
        "documents": story["documents"],
        "claim_order": [row["id"] for row in story["claims"]],
        "experiment_order": [row["id"] for row in completed],
        "planned_order": [row["id"] for row in planned],
        "claim_evidence": {row["id"]: row["evidence"] for row in story["claims"]},
        "planned": {
            row["id"]: {"priority": row["priority"], **({"plan": row["plan"]} if row.get("plan") else {})}
            for row in planned
        },
    }
    return {
        "programs": list(PROGRAMS),
        "claims": claims,
        "experiments": experiments,
        "decisions": decisions,
        "runs": [],
        "capsules": capsules,
        "compatibility": compatibility,
        "maps": {
            "claim_id_map": claim_ids,
            "experiment_id_map": experiment_ids | planned_ids,
            "decision_id_map": {row["id"]: f"NREV-DEC-{index:04d}" for index, row in enumerate(completed, 1)},
        },
    }


def write_records(research_root: Path, story: dict[str, Any], source_path: Path) -> dict[str, Any]:
    registry = research_root / "registry"
    evidence = research_root / "evidence"
    migration = registry / "migration"
    for group in ("programs", "claims", "experiments", "decisions", "runs", "migration"):
        (registry / group).mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)

    dump_yaml(migration / "research_story_v1.yaml", story)
    snapshot_digest = sha256_file(migration / "research_story_v1.yaml")
    payload = build_records(story, snapshot_digest)
    dump_yaml(migration / "story_compatibility.yaml", payload["compatibility"])

    for record in payload["programs"]:
        dump_yaml(registry / "programs" / f"{record['slug']}.yaml", record)
    for group in ("claims", "experiments", "decisions", "runs"):
        for record in payload[group]:
            dump_yaml(registry / group / f"{record['id']}.yaml", record)
    for record in payload["capsules"]:
        dump_json(evidence / f"{record['experiment_id']}.json", record)

    views = {
        "schema_version": 1,
        "canonical_document": story["canonical_document"],
        "documents": story["documents"],
        "migration": {
            "original_source": ORIGINAL_SOURCE,
            "original_source_sha256": ORIGINAL_SOURCE_SHA256,
            "replay_source": os.path.relpath(source_path, ROOT),
            "snapshot": SNAPSHOT_PATH,
            "snapshot_sha256": snapshot_digest,
            "recorded_at": MIGRATION_DATE,
            "semantic_change": "none",
        },
    }
    dump_yaml(registry / "views.yaml", views)
    completed_ids = [row["id"] for row in payload["experiments"] if row.get("evidence_capsule_id")]
    planned_ids = [row["id"] for row in payload["experiments"] if row["lifecycle"] == "draft"]
    index = {
        "schema_version": 1,
        "flagship_program_id": FLAGSHIP_PROGRAM_ID,
        "programs": [row["id"] for row in payload["programs"]],
        "claims": [row["id"] for row in payload["claims"]],
        "experiments": [row["id"] for row in payload["experiments"]],
        "completed_experiments": completed_ids,
        "planned_experiments": planned_ids,
        "decisions": [row["id"] for row in payload["decisions"]],
        "runs": [],
        "evidence_capsules": [row["id"] for row in payload["capsules"]],
        "migration": {**payload["maps"], "recorded_at": MIGRATION_DATE, "snapshot_sha256": snapshot_digest},
    }
    dump_yaml(registry / "index.yaml", index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write normalized records")
    parser.add_argument("--force", action="store_true", help="allow replacement of this migration's generated records")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help="research root to populate (default: repository research/)",
    )
    args = parser.parse_args()
    snapshot = DEFAULT_RESEARCH_ROOT / "registry" / "migration" / "research_story_v1.yaml"
    source = snapshot if snapshot.is_file() else PAPER_STORY
    if not source.is_file():
        raise SystemExit(f"missing v1 research story: {source}")
    story = yaml.safe_load(source.read_text())
    if not args.write:
        print(
            f"migration ready: {len(story['claims'])} claims, {len(story['experiments'])} completed experiments, "
            f"{len(story['next_experiments'])} planned experiments"
        )
        return 0
    occupied = [
        path
        for path in (args.output_root / "registry" / "claims", args.output_root / "registry" / "experiments")
        if path.exists() and any(path.iterdir())
    ]
    if occupied and not args.force:
        raise SystemExit("refusing to overwrite migrated records without --force: " + ", ".join(map(str, occupied)))
    result = write_records(args.output_root, story, source)
    print(
        f"migrated {len(result['claims'])} claims, {len(result['completed_experiments'])} completed experiments, "
        f"{len(result['planned_experiments'])} planned experiments, and {len(result['evidence_capsules'])} capsules"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
