"""Audit a stage-gated fish intent and inverse-control experiment program."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from neurobench.validation.schemas import validate_json


class ProgramManifestError(ValueError):
    """Raised when a program manifest is structurally or semantically invalid."""


_TERMINAL_STATUSES = {"completed", "failed", "stopped", "cancelled"}
_RUNNABLE_POLICIES = {"safe_cpu", "explicit_approval", "manual_action"}
_RESUME_MARKER_NAME = "program_run.json"
_RESUMABLE_MARKER_STATUSES = {"failed", "stopped"}


def load_program_manifest(path: str | Path) -> dict[str, Any]:
    """Load and semantically validate a fish-control program manifest."""
    source = Path(path).expanduser()
    payload = validate_json(source, "fish_control_program")
    _validate_semantics(payload)
    return payload


def _validate_semantics(payload: Mapping[str, Any]) -> None:
    gates = payload["gates"]
    experiments = payload["experiments"]
    gate_ids = [str(item["id"]) for item in gates]
    experiment_ids = [str(item["id"]) for item in experiments]
    if len(gate_ids) != len(set(gate_ids)):
        raise ProgramManifestError("gate ids must be unique")
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ProgramManifestError("experiment ids must be unique")
    known_gates = set(gate_ids)
    known_experiments = set(experiment_ids)

    priorities = [int(item["priority"]) for item in experiments]
    if len(priorities) != len(set(priorities)):
        raise ProgramManifestError("experiment priorities must be unique")

    dependency_graph: dict[str, list[str]] = {}
    for experiment in experiments:
        experiment_id = str(experiment["id"])
        dependencies = [str(value) for value in experiment.get("dependencies", [])]
        missing = sorted(set(dependencies) - known_experiments)
        if missing:
            raise ProgramManifestError(
                f"{experiment_id} references unknown dependencies: {', '.join(missing)}"
            )
        if experiment_id in dependencies:
            raise ProgramManifestError(f"{experiment_id} cannot depend on itself")
        dependency_graph[experiment_id] = dependencies

        if (
            experiment["resources"]["device"] == "cuda"
            and experiment["launch_policy"] != "explicit_approval"
        ):
            raise ProgramManifestError(
                f"{experiment_id} uses CUDA and requires launch_policy=explicit_approval"
            )

        for field in ("requires_gate", "advances_gate"):
            gate_id = experiment.get(field)
            if gate_id is not None and str(gate_id) not in known_gates:
                raise ProgramManifestError(
                    f"{experiment_id}.{field} references unknown gate {gate_id!r}"
                )

        design = experiment.get("design")
        if design is not None:
            computed = _computed_job_count(design)
            declared = int(design["expected_jobs"])
            if computed != declared:
                raise ProgramManifestError(
                    f"{experiment_id}.design.expected_jobs is {declared}, "
                    f"but the factor grid and repeats imply {computed}"
                )

    _validate_acyclic(dependency_graph)


def _computed_job_count(design: Mapping[str, Any]) -> int:
    factors = design.get("factor_grid", {})
    repeats = int(design.get("repeats", 1))
    if not factors:
        return repeats
    return math.prod(len(values) for values in factors.values()) * repeats


def _validate_acyclic(graph: Mapping[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ProgramManifestError(f"dependency cycle detected at {node}")
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for experiment_id in graph:
        visit(experiment_id)


def audit_program_manifest(
    manifest_path: str | Path,
    *,
    check_paths: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return readiness, resource, dependency, and job-count diagnostics."""
    source = Path(manifest_path).expanduser()
    payload = load_program_manifest(source)
    base_dir = source.resolve().parent
    project_root = Path(__file__).resolve().parents[2]
    gate_by_id = {str(item["id"]): item for item in payload["gates"]}
    experiment_by_id = {str(item["id"]): item for item in payload["experiments"]}
    hardware = payload["hardware_profile"]
    local_preflight = (
        _collect_local_preflight(
            project_root,
            inspect_gpu=any(
                item["resources"]["device"] == "cuda"
                for item in payload["experiments"]
            ),
        )
        if check_paths
        else None
    )
    results: list[dict[str, Any]] = []

    for experiment in sorted(payload["experiments"], key=lambda item: int(item["priority"])):
        blockers: list[dict[str, str]] = []
        experiment_id = str(experiment["id"])
        status = str(experiment["status"])

        for dependency_id in experiment.get("dependencies", []):
            dependency_status = str(experiment_by_id[dependency_id]["status"])
            if dependency_status != "completed":
                blockers.append(
                    {
                        "type": "dependency",
                        "detail": f"{dependency_id} is {dependency_status}, not completed",
                    }
                )

        required_gate = experiment.get("requires_gate")
        if required_gate is not None:
            gate_status = str(gate_by_id[str(required_gate)]["status"])
            if gate_status != "passed":
                blockers.append(
                    {
                        "type": "gate",
                        "detail": f"{required_gate} is {gate_status}, not passed",
                    }
                )

        missing_inputs: list[str] = []
        if check_paths:
            for item in experiment.get("inputs", []):
                if not bool(item.get("required_for_launch", True)):
                    continue
                resolved = _resolve_manifest_path(base_dir, item["path"])
                if not resolved.exists():
                    missing_inputs.append(str(item["path"]))
                    blockers.append(
                        {
                            "type": "input",
                            "detail": f"missing required input: {item['path']}",
                        }
                    )

        output_collision = False
        resume_marker: dict[str, Any] | None = None
        output_root = experiment.get("output_root")
        if check_paths and output_root and status not in _TERMINAL_STATUSES:
            resolved_output = _resolve_manifest_path(base_dir, output_root)
            output_collision = resolved_output.exists()
            if output_collision and experiment.get("resume_policy") == "refuse_existing":
                blockers.append(
                    {
                        "type": "output_collision",
                        "detail": f"output already exists: {output_root}",
                    }
                )
            elif output_collision and experiment.get("resume_policy") == "resume_atomic":
                resume_marker, marker_issue = _validate_resume_output(
                    resolved_output,
                    program_id=str(payload["program_id"]),
                    experiment_id=experiment_id,
                )
                if marker_issue is not None:
                    blockers.append(
                        {
                            "type": "output_collision",
                            "detail": marker_issue,
                        }
                    )

        resource_issues = _resource_issues(experiment["resources"], hardware)
        if check_paths and status not in _TERMINAL_STATUSES and status != "running":
            resource_issues.extend(
                _local_resource_issues(
                    experiment["resources"],
                    hardware,
                    local_preflight,
                )
            )
        blockers.extend({"type": "resource", "detail": value} for value in resource_issues)

        readiness = _readiness(
            status=status,
            launch_policy=str(experiment["launch_policy"]),
            blockers=blockers,
        )
        design = experiment.get("design")
        expected_jobs = int(design["expected_jobs"]) if design else 0
        decision_value = (
            2 * int(experiment["impact"])
            + 2 * int(experiment["information_gain"])
            - int(experiment["cost"])
            - int(experiment["risk"])
        )
        results.append(
            {
                "id": experiment_id,
                "title": experiment["title"],
                "stage": experiment["stage"],
                "priority": int(experiment["priority"]),
                "status": status,
                "readiness": readiness,
                "launch_policy": experiment["launch_policy"],
                "expected_jobs": expected_jobs,
                "decision_value": decision_value,
                "dependencies": list(experiment.get("dependencies", [])),
                "requires_gate": required_gate,
                "advances_gate": experiment.get("advances_gate"),
                "missing_inputs": missing_inputs,
                "output_collision": output_collision,
                "resume_marker": resume_marker,
                "blockers": blockers,
            }
        )

    readiness_counts = Counter(item["readiness"] for item in results)
    recommended = next(
        (
            item
            for item in results
            if item["readiness"] in {"ready", "manual_action_required", "approval_required"}
        ),
        None,
    )
    now = generated_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "program_id": payload["program_id"],
        "objective": payload["objective"],
        "manifest": str(source),
        "generated_at": now,
        "path_checks_enabled": bool(check_paths),
        "local_preflight": local_preflight,
        "hardware_profile": hardware,
        "summary": {
            "experiment_count": len(results),
            "planned_compute_jobs": sum(item["expected_jobs"] for item in results),
            "gate_counts": dict(Counter(item["status"] for item in payload["gates"])),
            "readiness_counts": dict(readiness_counts),
            "recommended_next_experiment": recommended["id"] if recommended else None,
        },
        "gates": list(payload["gates"]),
        "experiments": results,
    }


def _resolve_manifest_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _collect_local_preflight(
    project_root: Path,
    *,
    inspect_gpu: bool,
) -> dict[str, Any]:
    """Collect read-only local disk and GPU-concurrency telemetry."""
    result: dict[str, Any] = {
        "free_disk_gib": None,
        "disk_probe_error": None,
        "active_gpu_jobs": None,
        "gpu_probe_error": None,
    }
    try:
        free_bytes = shutil.disk_usage(project_root).free
        result["free_disk_gib"] = round(free_bytes / (1024**3), 3)
    except OSError as exc:
        result["disk_probe_error"] = str(exc)

    if not inspect_gpu:
        return result

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["gpu_probe_error"] = str(exc)
        return result

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        result["gpu_probe_error"] = detail[:300]
        return result

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    invalid = [line for line in lines if not line.isdigit()]
    if invalid:
        result["gpu_probe_error"] = (
            "unexpected nvidia-smi compute-process output: " + ", ".join(invalid[:3])
        )
        return result
    result["active_gpu_jobs"] = len(set(lines))
    return result


def _local_resource_issues(
    resources: Mapping[str, Any],
    hardware: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
) -> list[str]:
    """Apply fail-closed local resource gates to a prospective launch."""
    if preflight is None:
        return []

    issues: list[str] = []
    free_disk_gib = preflight.get("free_disk_gib")
    minimum_disk_gib = float(hardware["min_free_disk_gib"])
    if free_disk_gib is None:
        issues.append(
            "could not verify free disk headroom: "
            f"{preflight.get('disk_probe_error') or 'probe returned no value'}"
        )
    elif float(free_disk_gib) < minimum_disk_gib:
        issues.append(
            f"free disk {float(free_disk_gib):g} GiB is below required minimum "
            f"{minimum_disk_gib:g} GiB"
        )

    if resources["device"] != "cuda":
        return issues

    active_gpu_jobs = preflight.get("active_gpu_jobs")
    max_gpu_jobs = int(hardware["max_concurrent_gpu_jobs"])
    if active_gpu_jobs is None:
        issues.append(
            "could not verify active GPU job count: "
            f"{preflight.get('gpu_probe_error') or 'probe returned no value'}"
        )
    elif int(active_gpu_jobs) >= max_gpu_jobs:
        issues.append(
            f"active GPU jobs={int(active_gpu_jobs)} meets or exceeds concurrent "
            f"limit {max_gpu_jobs}"
        )
    return issues


def _validate_resume_output(
    output_root: Path,
    *,
    program_id: str,
    experiment_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify that an existing resumable root is owned and safely resumable."""
    marker_path = output_root / _RESUME_MARKER_NAME
    if not output_root.is_dir():
        return None, f"resume output is not a directory: {output_root}"
    if not marker_path.is_file():
        return None, (
            f"existing resume output is unverifiable; missing {_RESUME_MARKER_NAME}: "
            f"{output_root}"
        )
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {_RESUME_MARKER_NAME}: {exc}"

    expected_fields = {"schema_version", "program_id", "experiment_id", "status"}
    if not isinstance(marker, dict) or set(marker) != expected_fields:
        return None, (
            f"invalid {_RESUME_MARKER_NAME}: expected exactly "
            "schema_version, program_id, experiment_id, and status"
        )
    if marker['schema_version'] != 1:
        return marker, f"unsupported resume marker schema: {marker['schema_version']!r}"
    if not all(
        isinstance(marker[field], str)
        for field in ("program_id", "experiment_id", "status")
    ):
        return marker, "invalid resume marker: ownership and status fields must be strings"
    if marker['program_id'] != program_id:
        return marker, (
            f"resume marker program mismatch: expected {program_id}, "
            f"found {marker['program_id']}"
        )
    if marker['experiment_id'] != experiment_id:
        return marker, (
            f"resume marker experiment mismatch: expected {experiment_id}, "
            f"found {marker['experiment_id']}"
        )
    if marker['status'] not in _RESUMABLE_MARKER_STATUSES:
        return marker, (
            f"resume marker status {marker['status']!r} is not safely resumable; "
            "expected failed or stopped"
        )
    return marker, None


def _resource_issues(
    resources: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    cpu_threads = int(resources["cpu_threads"])
    max_cpu_threads = int(hardware["recommended_max_cpu_threads"])
    if cpu_threads > max_cpu_threads:
        issues.append(
            f"cpu_threads={cpu_threads} exceeds recommended cap {max_cpu_threads}"
        )
    max_ram_gib = float(resources["max_ram_gib"])
    recommended_ram_gib = float(hardware["recommended_max_ram_gib"])
    if max_ram_gib > recommended_ram_gib:
        issues.append(
            f"max_ram_gib={max_ram_gib:g} exceeds recommended cap "
            f"{recommended_ram_gib:g}"
        )
    if resources["device"] == "cuda":
        gpu_total = int(hardware["gpu_memory_mib"])
        reserve = int(hardware["gpu_memory_reserve_mib"])
        hard_limit = int(resources.get("gpu_memory_hard_limit_mib", 0))
        target = int(resources.get("gpu_memory_target_mib", 0))
        usable = gpu_total - reserve
        if hard_limit <= 0 or target <= 0:
            issues.append("CUDA jobs require positive target and hard GPU memory limits")
        if target > hard_limit:
            issues.append(
                f"GPU target {target} MiB exceeds hard limit {hard_limit} MiB"
            )
        if hard_limit > usable:
            issues.append(
                f"GPU hard limit {hard_limit} MiB exceeds usable budget {usable} MiB"
            )
    return issues


def _readiness(
    *,
    status: str,
    launch_policy: str,
    blockers: list[dict[str, str]],
) -> str:
    if status == "completed":
        return "completed"
    if status == "running":
        return "running"
    if status in {"failed", "stopped", "cancelled"}:
        return status
    if blockers:
        return "blocked"
    if launch_policy == "manual_action":
        return "manual_action_required"
    if launch_policy == "explicit_approval":
        return "approval_required"
    if launch_policy == "safe_cpu":
        return "ready"
    if launch_policy not in _RUNNABLE_POLICIES:
        return "blocked"
    return "ready"


def render_program_audit_markdown(audit: Mapping[str, Any]) -> str:
    """Render a compact, LLM-friendly program readiness report."""
    summary = audit["summary"]
    lines = [
        f"# Fish-control program audit: {audit['program_id']}",
        "",
        "## Decision",
        "",
        f"- Planned experiments: `{summary['experiment_count']}`.",
        f"- Planned compute jobs: `{summary['planned_compute_jobs']}`.",
        f"- Recommended next experiment: `{summary['recommended_next_experiment'] or 'none'}`.",
        f"- Readiness counts: `{json.dumps(summary['readiness_counts'], sort_keys=True)}`.",
        "",
        "This audit reports readiness; it does not authorize GPU or stimulation work.",
        "",
        "## Stage gates",
        "",
        "| Gate | Status | Evidence | Blockers |",
        "|---|---|---|---|",
    ]
    for gate in audit["gates"]:
        evidence = "<br>".join(gate.get("evidence", [])) or "None recorded"
        blockers = "<br>".join(gate.get("blockers", [])) or "None recorded"
        lines.append(
            f"| `{gate['id']}` | `{gate['status']}` | {evidence} | {blockers} |"
        )
    lines.extend(
        [
            "",
            "## Experiment queue",
            "",
            "| Priority | Experiment | Stage | Jobs | Readiness | Decision value |",
            "|---:|---|---|---:|---|---:|",
        ]
    )
    for item in audit["experiments"]:
        lines.append(
            f"| {item['priority']} | `{item['id']}` | {item['stage']} | "
            f"{item['expected_jobs']} | `{item['readiness']}` | "
            f"{item['decision_value']} |"
        )
        for blocker in item["blockers"]:
            lines.append(
                f"|  | ↳ blocker | {blocker['type']} |  | "
                f"{blocker['detail']} |  |"
            )
    lines.extend(
        [
            "",
            "## Hardware envelope",
            "",
            f"- CPU: {audit['hardware_profile']['cpu_model']}; "
            f"{audit['hardware_profile']['cpu_logical_count']} logical CPUs, "
            f"recommended experiment cap "
            f"{audit['hardware_profile']['recommended_max_cpu_threads']}.",
            f"- RAM: {audit['hardware_profile']['ram_total_gib']} GiB total, "
            f"recommended experiment cap "
            f"{audit['hardware_profile']['recommended_max_ram_gib']} GiB.",
            f"- GPU: {audit['hardware_profile']['gpu_name']}, "
            f"{audit['hardware_profile']['gpu_memory_mib']} MiB; reserve "
            f"{audit['hardware_profile']['gpu_memory_reserve_mib']} MiB.",
            f"- Concurrent GPU jobs: "
            f"{audit['hardware_profile']['max_concurrent_gpu_jobs']}.",
            "",
        ]
    )
    return "\n".join(lines)


def write_program_audit(
    audit: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Atomically write JSON and Markdown audit artifacts."""
    target = Path(output_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "program_audit.json"
    markdown_path = target / "program_audit.md"
    _atomic_text(json_path, json.dumps(dict(audit), indent=2, sort_keys=True) + "\n")
    _atomic_text(markdown_path, render_program_audit_markdown(audit))
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()

