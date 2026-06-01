"""LLM handoff helpers for architecture and experiment proposal planning."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

from neurobench.architecture_runs import as_run_manifest, build_planned_manifest
from neurobench.pipeline_catalog import catalog_as_dict, normalize_pipeline
from neurobench.validation.schemas import validate_dict


DEFAULT_MAX_COMBINATIONS = 4096


def slugify(value: str) -> str:
    """Return a stable local identifier fragment."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-").lower()
    return cleaned or "proposal"


def build_llm_context(
    *,
    dataset_manifest: Mapping[str, Any] | None = None,
    architecture_runs: Mapping[str, Any] | None = None,
    objective: str = "review_efficiency",
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    lab_notes: str = "",
) -> dict[str, Any]:
    """Build a provider-neutral context payload for an external LLM."""

    manifest = as_run_manifest(architecture_runs) if architecture_runs else {"schema_version": 1, "dataset_id": "", "runs": []}
    dataset_id = str((dataset_manifest or {}).get("dataset_id") or manifest.get("dataset_id") or "")
    stage_catalog = catalog_as_dict()
    return {
        "schema_version": 1,
        "kind": "neurobench_llm_architecture_context",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "objective": objective,
        "max_combinations_per_architecture": int(max_combinations),
        "llm_response_schema": "llm_architecture_proposal.schema.json",
        "constraints": [
            "Return JSON only.",
            "Use structured pipeline steps with unique id values.",
            "Sweep axes must reference concrete pipeline step ids.",
            "Do not propose arbitrary shell commands.",
            "Keep each architecture sweep at or below max_combinations_per_architecture.",
            "Represent multi-stage CFAR as explicit sequential CFAR steps with unique ids.",
        ],
        "dataset_manifest": deepcopy(dict(dataset_manifest or {})),
        "stage_catalog": stage_catalog,
        "current_runs": [_run_summary(run) for run in manifest.get("runs", [])],
        "saved_pipelines": deepcopy(list(manifest.get("saved_pipelines") or [])),
        "optimization_studies": deepcopy(list(manifest.get("optimization_studies") or [])),
        "lab_notes": lab_notes,
    }


def render_llm_prompt(context: Mapping[str, Any]) -> str:
    """Render a compact human-readable handoff prompt for an external LLM."""

    dataset_id = context.get("dataset_id", "")
    objective = context.get("objective", "review_efficiency")
    max_combinations = context.get("max_combinations_per_architecture", DEFAULT_MAX_COMBINATIONS)
    implemented = [
        stage["stage_id"]
        for stage in (context.get("stage_catalog") or {}).values()
        if stage.get("availability") == "implemented"
    ]
    return "\n".join(
        [
            "# Neurobench Architecture Proposal Request",
            "",
            f"Dataset: `{dataset_id}`",
            f"Objective: `{objective}`",
            f"Maximum combinations per architecture: `{max_combinations}`",
            "",
            "Return JSON matching `schemas/llm_architecture_proposal.schema.json`.",
            "Propose multiple compact architectures, each with a bounded parameter sweep.",
            "Use concrete pipeline step ids in all sweep axes.",
            "For multi-stage CFAR, use two explicit CFAR stages, such as `cfar_small_ref` then `cfar_large_ref`, and include cascade metadata.",
            "Do not include arbitrary shell commands or unregistered stage names.",
            "",
            "Implemented stage ids available for executable local tests:",
            ", ".join(implemented),
            "",
            "Use the attached context JSON for parameter ranges, saved baselines, real-time metadata, and current run summaries.",
        ]
    ).rstrip() + "\n"


def validate_proposal_set(
    proposal_set: Mapping[str, Any],
    *,
    max_combinations: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize an LLM architecture proposal set."""

    validate_dict(proposal_set, "llm_architecture_proposal")
    limit = int(max_combinations or proposal_set.get("max_combinations_per_architecture") or DEFAULT_MAX_COMBINATIONS)
    normalized = deepcopy(dict(proposal_set))
    normalized["max_combinations_per_architecture"] = limit
    seen: set[str] = set()
    reports: list[dict[str, Any]] = []
    proposals = []
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = slugify(str(proposal.get("id", "")))
        if proposal_id in seen:
            raise ValueError(f"Duplicate proposal id '{proposal_id}'.")
        seen.add(proposal_id)
        pipeline = normalize_pipeline(proposal.get("pipeline"), require_structured=True)
        sweep = _normalize_proposal_sweep(proposal.get("sweep"), pipeline)
        combinations = sweep_combination_count(sweep)
        if combinations > limit:
            raise ValueError(
                f"Proposal '{proposal_id}' sweep has {combinations} combinations, above limit {limit}."
            )
        item = deepcopy(dict(proposal))
        item["id"] = proposal_id
        item["pipeline"] = pipeline
        if sweep is not None:
            item["sweep"] = sweep
        proposals.append(item)
        reports.append(
            {
                "id": proposal_id,
                "label": item.get("label", proposal_id),
                "stage_count": len(pipeline),
                "sweep_combinations": combinations,
                "status": "valid",
            }
        )
    normalized["proposals"] = proposals
    normalized["validation_report"] = {
        "status": "valid",
        "proposal_count": len(proposals),
        "max_combinations_per_architecture": limit,
        "proposals": reports,
    }
    return normalized


def proposal_set_to_architecture_manifest(
    proposal_set: Mapping[str, Any],
    *,
    base_manifest: Mapping[str, Any] | None = None,
    max_combinations: int | None = None,
) -> dict[str, Any]:
    """Convert valid LLM proposals into architecture-run manifest entries."""

    validated = validate_proposal_set(proposal_set, max_combinations=max_combinations)
    base = as_run_manifest(base_manifest) if base_manifest else {
        "schema_version": 1,
        "dataset_id": validated["dataset_id"],
        "runs": [],
    }
    base["dataset_id"] = base.get("dataset_id") or validated["dataset_id"]
    saved = list(base.get("saved_pipelines") or [])
    runs = list(base.get("runs") or [])
    experiments = list(base.get("experiments") or [])
    generated_at = datetime.now(timezone.utc).isoformat()
    proposal_run_ids: list[str] = []

    for proposal in validated["proposals"]:
        template = _proposal_template(validated, proposal, generated_at)
        saved = [item for item in saved if item.get("id") != template["id"]]
        saved.append(template)

        spec = _proposal_spec(validated, proposal, template)
        planned = build_planned_manifest(spec)
        new_ids = {run["run_id"] for run in planned.get("runs", [])}
        if len(new_ids) != len(planned.get("runs", [])):
            raise ValueError(f"Proposal '{proposal['id']}' generated duplicate run_id values.")
        existing_current_ids = {item.get("run_id") for item in runs}
        duplicate_ids = sorted(new_ids & existing_current_ids)
        if duplicate_ids:
            runs = [item for item in runs if item.get("run_id") not in new_ids]
        runs.extend(planned.get("runs", []))
        proposal_run_ids.extend(run["run_id"] for run in planned.get("runs", []))

    proposal_set_id = slugify(str(validated["proposal_set_id"]))
    experiments.append(
        {
            "id": f"llm_{proposal_set_id}",
            "source": "llm_architecture_proposal",
            "mode": "architecture_sweep_pack",
            "createdAt": generated_at,
            "objective": validated.get("objective"),
            "proposal_set_id": proposal_set_id,
            "run_ids": proposal_run_ids,
        }
    )
    proposal_sets = [
        item for item in list(base.get("llm_proposal_sets") or []) if item.get("proposal_set_id") != proposal_set_id
    ]
    proposal_sets.append(
        {
            "proposal_set_id": proposal_set_id,
            "source": "llm_handoff",
            "objective": validated.get("objective"),
            "createdAt": generated_at,
            "max_combinations_per_architecture": validated.get("max_combinations_per_architecture"),
            "validation_report": validated.get("validation_report"),
        }
    )
    base["saved_pipelines"] = sorted(saved, key=lambda item: str(item.get("label") or item.get("id")))
    base["runs"] = runs
    base["experiments"] = experiments
    base["llm_proposal_sets"] = proposal_sets
    return base


def sweep_combination_count(sweep: Mapping[str, Any] | None) -> int:
    if not sweep:
        return 1
    total = 1
    for axis in list(sweep.get("parameters") or []):
        total *= len(list(axis.get("values") or []))
    return total


def _normalize_proposal_sweep(sweep: Any, pipeline: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if sweep is None:
        return None
    if not isinstance(sweep, Mapping):
        raise ValueError("Proposal sweep must be an object.")
    normalized: dict[str, Any] = {key: deepcopy(value) for key, value in dict(sweep).items() if key != "parameters"}
    axes = []
    step_ids = {str(step["id"]): step for step in pipeline}
    for axis in list(sweep.get("parameters") or []):
        if not isinstance(axis, Mapping):
            raise ValueError("Proposal sweep parameter entries must be objects.")
        step_id = str(axis.get("stage") or axis.get("step_id") or "")
        if step_id not in step_ids:
            raise ValueError(f"Proposal sweep references unknown pipeline step '{step_id}'. Use concrete step ids.")
        if "stage_id" in axis and axis["stage_id"] != step_ids[step_id]["stage_id"]:
            raise ValueError(f"Proposal sweep axis '{step_id}' stage_id does not match the pipeline step.")
        clean_axis = deepcopy(dict(axis))
        clean_axis["stage"] = step_id
        clean_axis["stage_id"] = step_ids[step_id]["stage_id"]
        axes.append(clean_axis)
    if axes:
        normalized["parameters"] = axes
    return normalized if axes else None


def _proposal_template(proposal_set: Mapping[str, Any], proposal: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    template_id = slugify(f"llm_{proposal_set['proposal_set_id']}_{proposal['id']}")
    return {
        "id": template_id,
        "label": proposal.get("label") or proposal["id"],
        "description": proposal.get("rationale") or proposal.get("hypothesis") or "",
        "dataset_id": proposal_set["dataset_id"],
        "createdAt": generated_at,
        "updatedAt": generated_at,
        "source": "llm_architecture_proposal",
        "proposal_set_id": proposal_set["proposal_set_id"],
        "proposal_id": proposal["id"],
        "objective": proposal_set.get("objective"),
        "pipeline": deepcopy(list(proposal.get("pipeline") or [])),
        "sweep": deepcopy(proposal.get("sweep")),
        "metadata": {
            "hypothesis": proposal.get("hypothesis", ""),
            "expected_tradeoffs": proposal.get("expected_tradeoffs", ""),
            "priority": proposal.get("priority"),
        },
    }


def _proposal_spec(proposal_set: Mapping[str, Any], proposal: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    run_id = slugify(f"planned_{proposal_set['proposal_set_id']}_{proposal['id']}")
    spec = {
        "schema_version": 1,
        "dataset_id": proposal_set["dataset_id"],
        "run_id": run_id,
        "label": proposal.get("label") or proposal["id"],
        "pipeline": deepcopy(list(proposal.get("pipeline") or [])),
        "artifacts": {
            "source": "llm_architecture_proposal",
            "proposal_set_id": proposal_set["proposal_set_id"],
            "proposal_id": proposal["id"],
        },
        "parameters": {
            "objective": proposal_set.get("objective"),
            "hypothesis": proposal.get("hypothesis"),
            "template_id": template["id"],
        },
    }
    if proposal.get("sweep"):
        spec["sweep"] = deepcopy(proposal["sweep"])
    return spec


def _run_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(run.get("summary") or {})
    execution = dict(run.get("execution") or {})
    return {
        "run_id": run.get("run_id", ""),
        "label": run.get("label", ""),
        "status": execution.get("status", ""),
        "roi_count": summary.get("roi_count"),
        "event_count": summary.get("event_count"),
        "suggestion_count": summary.get("suggestion_count"),
        "pipeline": [
            {"id": step.get("id"), "stage_id": step.get("stage_id", step.get("stage", step.get("name")))}
            for step in list(run.get("pipeline") or [])
        ],
    }
