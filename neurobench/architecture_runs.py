"""Helpers for combining Neurobench architecture-run manifests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import json
from typing import Any, Iterable, Mapping, Sequence

from neurobench.pipeline_catalog import get_stage, normalize_pipeline


def as_run_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    if "runs" in data:
        runs = [_validated_run(run) for run in list(data.get("runs") or [])]
        dataset_id = data.get("dataset_id") or (runs[0].get("dataset_id") if runs else "")
        manifest: dict[str, Any] = {"schema_version": 1, "dataset_id": dataset_id, "runs": runs}
        for key in ("sweep", "experiments", "saved_pipelines", "optimization_studies", "llm_proposal_sets"):
            if key in data:
                manifest[key] = deepcopy(data[key])
        return manifest
    if "run_id" in data:
        run = _validated_run(data)
        return {"schema_version": 1, "dataset_id": data.get("dataset_id", ""), "runs": [run]}
    raise ValueError("Expected an architecture-run manifest with 'runs' or a single run with 'run_id'.")


def _validated_run(run_like: Mapping[str, Any]) -> dict[str, Any]:
    run = dict(run_like)
    if "pipeline" in run:
        run["pipeline"] = normalize_pipeline(run.get("pipeline"))
    return run


def merge_run_manifests(manifests: Iterable[Mapping[str, Any]], *, replace: bool = False) -> dict[str, Any]:
    merged: dict[str, Any] = {"schema_version": 1, "dataset_id": "", "runs": []}
    seen: dict[str, int] = {}
    for manifest_like in manifests:
        manifest = as_run_manifest(manifest_like)
        if not merged["dataset_id"]:
            merged["dataset_id"] = manifest.get("dataset_id", "")
        elif manifest.get("dataset_id") and manifest.get("dataset_id") != merged["dataset_id"]:
            raise ValueError(f"Cannot merge dataset_id={manifest.get('dataset_id')} into dataset_id={merged['dataset_id']}")
        for run in manifest.get("runs", []):
            run_id = run.get("run_id")
            if not run_id:
                raise ValueError("Architecture run is missing run_id.")
            if run_id in seen:
                if not replace:
                    raise ValueError(f"Duplicate architecture run_id '{run_id}'. Use --replace to overwrite.")
                merged["runs"][seen[run_id]] = dict(run)
            else:
                seen[run_id] = len(merged["runs"])
                merged["runs"].append(dict(run))
    return merged


def build_planned_run(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic planned architecture-run from a pipeline spec."""

    pipeline = normalize_pipeline(spec.get("pipeline"), require_structured=True)
    dataset_id = spec.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Pipeline spec is missing required string 'dataset_id'.")

    run_id = spec.get("run_id")
    if run_id is None:
        run_id = _deterministic_run_id(spec)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Pipeline spec run_id must be a non-empty string when provided.")

    run: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "pipeline": pipeline,
        "execution": {"status": "planned"},
        "artifacts": dict(spec.get("artifacts") or {}),
    }
    for key in ("label", "parameters", "summary"):
        if key in spec:
            run[key] = spec[key]
    if "execution" in spec:
        execution = dict(spec.get("execution") or {})
        execution["status"] = "planned"
        run["execution"] = execution
    return run


def build_planned_runs(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build one or more deterministic planned runs from a pipeline spec."""

    runs, _ = _build_planned_runs_and_sweep(spec)
    return runs


def build_planned_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    runs, sweep = _build_planned_runs_and_sweep(spec)
    dataset_id = runs[0]["dataset_id"] if runs else spec.get("dataset_id", "")
    manifest: dict[str, Any] = {"schema_version": 1, "dataset_id": dataset_id, "runs": runs}
    if sweep is not None:
        manifest["sweep"] = sweep
    return manifest


def _deterministic_run_id(spec: Mapping[str, Any]) -> str:
    stable_spec = {key: value for key, value in spec.items() if key != "run_id"}
    encoded = json.dumps(stable_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"planned_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _build_planned_runs_and_sweep(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sweep_like = spec.get("sweep")
    if sweep_like is None:
        return [build_planned_run(spec)], None

    base_run_id = spec.get("run_id")
    if base_run_id is not None and (not isinstance(base_run_id, str) or not base_run_id):
        raise ValueError("Pipeline spec run_id must be a non-empty string when provided.")

    pipeline = normalize_pipeline(spec.get("pipeline"), require_structured=True)
    sweep = _normalize_sweep(sweep_like, pipeline)
    factors = sweep["parameters"]
    total_runs = 1
    for factor in factors:
        total_runs *= len(factor["values"])

    runs: list[dict[str, Any]] = []
    for index, values in enumerate(product(*(factor["values"] for factor in factors))):
        assignments = [
            {
                "stage": factor["stage"],
                "stage_id": factor["stage_id"],
                "param": factor["param"],
                "value": deepcopy(value),
            }
            for factor, value in zip(factors, values)
        ]
        run_sweep = _run_sweep_metadata(sweep, index=index, total_runs=total_runs, assignments=assignments)
        expanded_spec = dict(spec)
        expanded_spec["pipeline"] = _pipeline_with_sweep_values(pipeline, assignments)
        expanded_spec["sweep"] = run_sweep
        if base_run_id is not None:
            expanded_spec["run_id"] = f"{base_run_id}__sweep_{index + 1:03d}"

        run = build_planned_run(expanded_spec)
        run["sweep"] = run_sweep
        runs.append(run)

    manifest_sweep = dict(sweep)
    manifest_sweep["total_runs"] = total_runs
    return runs, manifest_sweep


def _normalize_sweep(sweep_like: Any, pipeline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(sweep_like, Mapping):
        raise ValueError("Pipeline spec sweep must be an object.")

    raw_factors = sweep_like.get("parameters")
    if not _is_non_string_sequence(raw_factors) or not raw_factors:
        raise ValueError("Pipeline spec sweep.parameters must be a non-empty array.")

    sweep: dict[str, Any] = {}
    for key in ("id", "label", "description"):
        if key in sweep_like:
            value = sweep_like[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Pipeline spec sweep.{key} must be a non-empty string when provided.")
            sweep[key] = value

    factors: list[dict[str, Any]] = []
    for index, factor_like in enumerate(raw_factors):
        if not isinstance(factor_like, Mapping):
            raise ValueError(f"Pipeline spec sweep parameter at index {index} must be an object.")
        step = _resolve_sweep_step(factor_like, pipeline, index)
        step_id = step["id"]
        stage_id = step["stage_id"]

        param = factor_like.get("param")
        if not isinstance(param, str) or not param:
            raise ValueError(f"Pipeline spec sweep parameter for stage '{step_id}' is missing required string 'param'.")
        _validate_sweep_param(step, param)

        values = factor_like.get("values")
        if not _is_non_string_sequence(values) or not values:
            raise ValueError(f"Pipeline spec sweep parameter '{step_id}.{param}' values must be a non-empty array.")

        normalized_values = [deepcopy(value) for value in values]
        for value in normalized_values:
            _validate_sweep_value(pipeline, step_id, param, value)

        factors.append({"stage": step_id, "stage_id": stage_id, "param": param, "values": normalized_values})

    sweep["parameters"] = factors
    return sweep


def _resolve_sweep_step(
    factor: Mapping[str, Any],
    pipeline: Sequence[Mapping[str, Any]],
    factor_index: int,
) -> Mapping[str, Any]:
    if "step_id" in factor:
        key = "step_id"
        target = factor.get("step_id")
        matches = [step for step in pipeline if step.get("id") == target]
    elif "stage" in factor:
        key = "stage"
        target = factor.get("stage")
        matches = [step for step in pipeline if step.get("id") == target]
        if not matches:
            matches = [step for step in pipeline if step.get("stage_id") == target]
    elif "stage_id" in factor:
        key = "stage_id"
        target = factor.get("stage_id")
        matches = [step for step in pipeline if step.get("stage_id") == target]
    else:
        raise ValueError(f"Pipeline spec sweep parameter at index {factor_index} is missing required string 'stage'.")

    if not isinstance(target, str) or not target:
        raise ValueError(f"Pipeline spec sweep parameter at index {factor_index} has invalid {key}.")
    if not matches:
        raise ValueError(f"Unknown sweep stage '{target}'.")
    if len(matches) > 1:
        raise ValueError(f"Sweep stage '{target}' matches multiple pipeline steps; use step_id.")
    return matches[0]


def _validate_sweep_param(step: Mapping[str, Any], param: str) -> None:
    stage_id = str(step["stage_id"])
    stage = get_stage(stage_id)
    known_params = set(stage.required_params)
    known_params.update(dict(stage.default_params or {}))
    known_params.update(dict(stage.param_ranges or {}))
    known_params.update(dict(step.get("params") or {}))
    if param not in known_params:
        raise ValueError(f"Sweep parameter '{param}' is not valid for pipeline stage '{stage_id}'.")


def _validate_sweep_value(
    pipeline: Sequence[Mapping[str, Any]],
    step_id: str,
    param: str,
    value: Any,
) -> None:
    normalize_pipeline(_pipeline_with_sweep_values(pipeline, [{"stage": step_id, "param": param, "value": value}]), require_structured=True)


def _pipeline_with_sweep_values(
    pipeline: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expanded = deepcopy(list(pipeline))
    by_step_id = {step["id"]: step for step in expanded}
    for assignment in assignments:
        step = by_step_id[assignment["stage"]]
        params = dict(step.get("params") or {})
        params[assignment["param"]] = deepcopy(assignment["value"])
        step["params"] = params
    return expanded


def _run_sweep_metadata(
    sweep: Mapping[str, Any],
    *,
    index: int,
    total_runs: int,
    assignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = {key: sweep[key] for key in ("id", "label", "description") if key in sweep}
    metadata.update(
        {
            "index": index,
            "total_runs": total_runs,
            "parameters": [dict(assignment) for assignment in assignments],
        }
    )
    return metadata


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
