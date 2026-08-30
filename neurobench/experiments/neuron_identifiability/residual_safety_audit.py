"""Durable post-screen source-off residual safety audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import atomic_json, atomic_text, stable_hash
from .residual_safety import (
    JEPA_RESIDUAL_METHOD,
    RANDOM_RESIDUAL_METHOD,
    ResidualSafetyThresholds,
    apply_residual_safety_policy,
    evaluate_residual_safety,
)
from . import residual_safety as safety_module


REQUIRED_PARENT_INPUTS = (
    "residual_diagnostics.json",
    "paired_injection_results.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_parent_inputs(parent: Path) -> dict[str, Any]:
    index_path = parent / "artifact_index.json"
    if not index_path.is_file():
        raise ValueError("parent artifact_index.json is missing")
    index = _json(index_path)
    entries = {str(row["path"]): row for row in index.get("artifacts", [])}
    verified: list[dict[str, Any]] = []
    for relative in REQUIRED_PARENT_INPUTS:
        if relative not in entries:
            raise ValueError(f"parent artifact index does not register {relative}")
        path = parent / relative
        if not path.is_file():
            raise ValueError(f"parent input is missing: {relative}")
        observed_sha = _sha256_file(path)
        observed_bytes = path.stat().st_size
        expected = entries[relative]
        if observed_sha != expected["sha256"] or observed_bytes != int(expected["bytes"]):
            raise ValueError(f"parent input drifted: {relative}")
        verified.append(
            {
                "path": relative,
                "sha256": observed_sha,
                "bytes": observed_bytes,
            }
        )
    return {
        "artifact_index_sha256": _sha256_file(index_path),
        "verified_inputs": verified,
        "all_required_inputs_verified": True,
    }


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "artifact_index.json":
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def run_residual_safety_audit(
    parent_run_root: Path,
    output_root: Path,
    *,
    thresholds: ResidualSafetyThresholds = ResidualSafetyThresholds(),
) -> dict[str, Any]:
    """Write a non-colliding, non-claim-bearing guardrail package."""

    parent = Path(parent_run_root).resolve()
    output = Path(output_root).resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("residual safety output or partial root already exists")
    integrity = _verify_parent_inputs(parent)
    diagnostics = _json(parent / "residual_diagnostics.json")
    evaluations = _json(parent / "paired_injection_results.json")
    diagnostic_rows = diagnostics.get("rows")
    evaluation_rows = evaluations.get("rows")
    if not isinstance(diagnostic_rows, list) or not isinstance(evaluation_rows, list):
        raise ValueError("parent JSON inputs must contain row lists")

    safety = evaluate_residual_safety(diagnostic_rows, thresholds=thresholds)
    applications = {
        method: apply_residual_safety_policy(
            evaluation_rows,
            safety,
            residual_method=method,
        )
        for method in (JEPA_RESIDUAL_METHOD, RANDOM_RESIDUAL_METHOD)
    }
    partial.mkdir(parents=True)
    try:
        atomic_json(partial / "source_integrity.json", integrity)
        atomic_json(partial / "source_off_safety.json", safety)
        atomic_json(partial / "policy_application.json", applications)
        flattened_safety: list[dict[str, Any]] = []
        for row in safety["decisions"]:
            flattened_safety.append(
                {
                    **{key: value for key, value in row.items() if key != "checks"},
                    **row["checks"],
                }
            )
        _write_tsv(
            partial / "source_off_window_safety.tsv",
            flattened_safety,
            (
                "method",
                "background_recording_id",
                "background_window_id",
                "repeated_fixture_rows",
                "background_rms_ratio",
                "dynamic_mad_ratio",
                "seam_to_interior_jump_ratio",
                "background_rms_not_amplified",
                "dynamic_mad_not_amplified",
                "spatial_seam_within_tolerance",
                "safe_for_residual_use",
                "selected_endpoint",
            ),
        )
        policy_rows = [
            {"policy_residual_method": method, **row}
            for method, payload in applications.items()
            for row in payload["rows"]
        ]
        _write_tsv(
            partial / "policy_application.tsv",
            policy_rows,
            (
                "policy_residual_method",
                "fixture_id",
                "background_recording_id",
                "background_window_id",
                "selected_method",
                "residual_admitted",
                "source_on_recall",
                "recovered_sources",
                "injected_sources",
            ),
        )
        summary = {
            "schema_version": "neurobench.residual_safety_audit.v1",
            "parent_run_id": parent.name,
            "parent_artifact_index_sha256": integrity["artifact_index_sha256"],
            "implementation_sha256": _sha256_file(Path(__file__)),
            "safety_policy_implementation_sha256": _sha256_file(Path(safety_module.__file__)),
            "decision_grain": safety["decision_grain"],
            "thresholds": safety["thresholds"],
            "methods": safety["methods"],
            "policy_results": {
                method: {
                    key: payload[key]
                    for key in (
                        "fixture_count",
                        "residual_admitted_fixture_count",
                        "macro_source_on_recall",
                        "micro_recovered_sources",
                        "micro_injected_sources",
                    )
                }
                for method, payload in applications.items()
            },
            "outcome": (
                "No EXP-0029 residual window satisfied the post-screen source-off safety "
                "certificate; both fixed policies therefore selected raw for every fixture."
            ),
            "decision_uses_source_truth_or_labels": False,
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
            "claim_boundary": {
                "supported": "deterministic source-off guardrail behavior on frozen Run-B artifacts",
                "not_supported": [
                    "denoising",
                    "neuron identity",
                    "precision or specificity",
                    "independent biological generalization",
                    "motion robustness",
                ],
            },
        }
        summary["result_sha256"] = stable_hash(summary)
        atomic_json(partial / "summary.json", summary)
        validation = {
            "status": "passed_derived_engineering_audit_scientific_outputs_incomplete",
            "checks": {
                "parent_inputs_hash_verified": integrity["all_required_inputs_verified"],
                "source_off_decision_only": safety["decision_uses_source_truth_or_labels"] is False,
                "one_decision_per_method_window": len(safety["decisions"]) == 24,
                "expected_parent_fixture_count": all(
                    payload["fixture_count"] == 108 for payload in applications.values()
                ),
                "fallback_is_exact_raw_endpoint": all(
                    payload["residual_admitted_fixture_count"] == 0
                    and payload["macro_source_on_recall"] == 0.1875
                    for payload in applications.values()
                ),
                "scientific_completion_false": summary["scientific_completion"] is False,
            },
        }
        validation["all_engineering_checks_passed"] = all(validation["checks"].values())
        if not validation["all_engineering_checks_passed"]:
            raise ValueError("residual safety engineering validation failed")
        atomic_json(partial / "validation.json", validation)
        atomic_json(
            partial / "scientific_audit_status.json",
            {
                "status": "incomplete_derived_metrics_only",
                "expert_section": "not_applicable_no_new_expert_labels",
                "model_section": "not_produced_no_new_candidates",
                "comparison_section": "tables_only",
                "upstream_run": parent.name,
                "scientific_completion": False,
            },
        )
        atomic_json(
            partial / "llm_context.json",
            {
                "entry_point": "summary.json",
                "primary_table": "source_off_window_safety.tsv",
                "secondary_table": "policy_application.tsv",
                "validation": "validation.json",
                "unit_of_decision": "residual method by source-off background window",
                "unmatched_candidates": "unknown_not_false_positives",
            },
        )
        atomic_text(
            partial / "REPORT.md",
            "# Source-off residual safety certificate\n\n"
            "All 12 JEPA-residual and all 12 random-residual window decisions were unsafe "
            "under the post-screen no-amplification certificate. The fixed policy therefore "
            "fell back to the frozen raw endpoint for all 108 fixtures and reproduced raw "
            "macro source-on recall of `0.1875`. Selection used source-off RMS, dynamic MAD, "
            "and the explicitly stored source-off residual seam ratio only; injected truth "
            "was evaluated after the decision and never entered it.\n\n"
            "This is a derived engineering guardrail, not a preregistered Run-B gate or a "
            "scientific denoising result. Full scientific-audit media remain incomplete.\n",
        )
        atomic_json(
            partial / "status.json",
            {
                "status": "derived_engineering_audit_complete_scientific_outputs_incomplete",
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
            },
        )
        artifacts = _artifact_rows(partial)
        atomic_json(
            partial / "artifact_index.json",
            {
                "schema_version": 1,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "artifact_index_self_excluded_to_avoid_circular_hash": True,
            },
        )
        # Recompute the complete live tree immediately before atomic promotion.
        if artifacts != _artifact_rows(partial):
            raise ValueError("residual safety artifact tree changed before promotion")
        partial.replace(output)
        return summary
    except Exception:
        # Preserve the partial package for forensic inspection.
        raise


__all__ = ["run_residual_safety_audit"]
