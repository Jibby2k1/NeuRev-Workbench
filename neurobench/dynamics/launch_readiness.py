"""Build launch-readiness handoff artifacts for grid128 Stage B."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def build_grid128_stage_b_launch_readiness(
    *,
    root: Path,
    out_dir: Path,
    title: str = "Grid128 Stage B Launch Readiness",
) -> dict[str, Any]:
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sweep_dir = root / "sweeps" / "grid128_sequence_1day_v1"
    stage_b_dir = root / "plans" / "grid128_sequence_stage_b_v1"
    stop_review_path = root / "plans" / "grid128_stage_a_stop_review_v1" / "stage_a_stop_review.json"
    report_path = root / "reports" / "grid128_sequence_1day_partial_report_v1" / "dynamics_experiment_report.json"
    audit_path = root / "plans" / "grid128_artifact_audit_v1" / "grid128_artifact_audit.json"
    plan_path = stage_b_dir / "next_sweep_plan.json"
    manifest_path = stage_b_dir / "next_sweep_manifest.json"
    dry_run_path = stage_b_dir / "stage_b_sweep" / "sweep_manifest.json"

    active = _load_json(sweep_dir / "sweep_active.json")
    progress = _load_progress_log_summary(sweep_dir / "sweep_progress.jsonl")
    if not progress:
        raise ValueError(f"No valid progress records found in {sweep_dir / 'sweep_progress.jsonl'}")
    latest = progress["latest"]

    stop_review = _load_json(stop_review_path)
    plan = _load_json(plan_path)
    manifest = _load_json(manifest_path)
    dry_run = _load_json(dry_run_path)
    report = _load_json(report_path)
    audit = _load_json(audit_path)

    manifest_experiments = manifest.get("experiments") if isinstance(manifest.get("experiments"), list) else []
    dry_run_experiments = dry_run.get("experiments") if isinstance(dry_run.get("experiments"), list) else []
    plan_count = plan.get("planned_experiment_count")
    manifest_count = manifest.get("planned_experiment_count")
    dry_run_count = dry_run.get("experiment_count")
    dry_run_validated = (
        str(plan_count) == str(manifest_count)
        and str(manifest_count) == str(dry_run_count)
        and len(manifest_experiments) == len(dry_run_experiments)
    )

    recommendation = stop_review.get("recommendation") or (
        "Use the refreshed Stage B manifest as the default next GPU job unless the user "
        "explicitly asks to continue the original Stage A sweep from index 478."
    )
    report_audit = report.get("artifact_audit_summary") if isinstance(report.get("artifact_audit_summary"), Mapping) else {}
    stage_b_out_dir = stage_b_dir / "stage_b_sweep"
    suggested_command = (
        ".venv-neurobench/bin/python -m neurobench.dynamics.overnight_sweep "
        f"--manifest {manifest_path} "
        f"--out-dir {stage_b_out_dir} "
        "--device cuda --batch-size 4 --seeds 7,13 --time-limit-hours 48"
    )

    readiness: dict[str, Any] = {
        "schema_version": 1,
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": "ready_pending_user_approval",
        "decision": {
            "default_next_gpu_job": "stage_b_manifest",
            "requires_user_approval": True,
            "recommendation": recommendation,
            "explicit_alternative": "resume_stage_a_from_index_478",
        },
        "stage_a": {
            "state": "stopped",
            "current_index": latest.get("index"),
            "experiment_count": latest.get("experiment_count"),
            "progress_text": f"{latest.get('index')} / {latest.get('experiment_count')}",
            "record_count": progress.get("record_count"),
            "status_counts": progress.get("status_counts"),
            "active_status": active.get("status"),
            "active_experiment_id": active.get("experiment_id"),
            "active_finished_at": active.get("finished_at"),
            "former_pid": stop_review.get("former_pid"),
            "stop_review_generated_at": stop_review.get("generated_at"),
            "stop_review_path": str(stop_review_path),
        },
        "stage_b": {
            "plan_path": str(plan_path),
            "plan_created_at": plan.get("created_at"),
            "planned_experiment_count": plan_count,
            "manifest_path": str(manifest_path),
            "manifest_experiment_count": manifest_count,
            "manifest_list_count": len(manifest_experiments),
            "dry_run_manifest_path": str(dry_run_path),
            "dry_run_experiment_count": dry_run_count,
            "dry_run_list_count": len(dry_run_experiments),
            "dry_run_validated": dry_run_validated,
            "selection_counts": plan.get("selection_counts") if isinstance(plan.get("selection_counts"), Mapping) else {},
            "suggested_command": suggested_command,
        },
        "evidence": {
            "report_path": str(report_path),
            "report_generated_at": report.get("generated_at"),
            "report_embedded_audit_artifact_count": report_audit.get("artifact_count"),
            "report_embedded_audit_status_counts": report_audit.get("status_counts") if isinstance(report_audit.get("status_counts"), Mapping) else {},
            "audit_path": str(audit_path),
            "audit_artifact_count": audit.get("artifact_count"),
            "audit_status_counts": audit.get("status_counts") if isinstance(audit.get("status_counts"), Mapping) else {},
            "audit_ok": audit.get("ok"),
        },
        "pre_launch_checklist": _pre_launch_checklist(),
        "guardrails": _guardrails(),
    }

    json_path = out / "stage_b_launch_readiness.json"
    markdown_path = out / "stage_b_launch_readiness.md"
    readiness["json_path"] = str(json_path)
    readiness["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_grid128_stage_b_launch_readiness_markdown(readiness), encoding="utf-8")
    return readiness


def render_grid128_stage_b_launch_readiness_markdown(readiness: Mapping[str, Any]) -> str:
    decision = readiness.get("decision") if isinstance(readiness.get("decision"), Mapping) else {}
    stage_a = readiness.get("stage_a") if isinstance(readiness.get("stage_a"), Mapping) else {}
    stage_b = readiness.get("stage_b") if isinstance(readiness.get("stage_b"), Mapping) else {}
    evidence = readiness.get("evidence") if isinstance(readiness.get("evidence"), Mapping) else {}
    lines = [
        f"# {readiness.get('title') or 'Grid128 Stage B Launch Readiness'}",
        "",
        f"Generated: `{readiness.get('generated_at')}`",
        f"Status: `{readiness.get('status')}`",
        "",
        "## Decision",
        "",
        f"- Default next GPU job: `{decision.get('default_next_gpu_job')}`.",
        f"- Requires user approval before launch: `{str(decision.get('requires_user_approval')).lower()}`.",
        f"- Recommendation: {decision.get('recommendation')}",
        "- Explicit alternative: resume Stage A from index `478`.",
        "",
        "## Stage A Stop State",
        "",
        f"- Progress: `{stage_a.get('progress_text')}`.",
        f"- Progress records: `{stage_a.get('record_count')}`.",
        f"- Status counts: `{stage_a.get('status_counts')}`.",
        f"- Active marker: index `{stage_a.get('current_index')}`, `{stage_a.get('active_experiment_id')}`, status `{stage_a.get('active_status')}`.",
        f"- Stop review: `{stage_a.get('stop_review_path')}`.",
        "",
        "## Stage B Inputs",
        "",
        f"- Plan: `{stage_b.get('plan_path')}`.",
        f"- Plan created: `{stage_b.get('plan_created_at')}`.",
        f"- Planned experiments: `{stage_b.get('planned_experiment_count')}`.",
        f"- Manifest experiments: `{stage_b.get('manifest_experiment_count')}` declared, `{stage_b.get('manifest_list_count')}` listed.",
        f"- Dry run experiments: `{stage_b.get('dry_run_experiment_count')}` declared, `{stage_b.get('dry_run_list_count')}` listed.",
        f"- Selection counts: `{stage_b.get('selection_counts')}`.",
        f"- Dry run validated: `{stage_b.get('dry_run_validated')}`.",
        "",
        "## Evidence",
        "",
        (
            f"- Report: `{evidence.get('report_path')}` generated `{evidence.get('report_generated_at')}` "
            f"with embedded audit `{evidence.get('report_embedded_audit_artifact_count')}` artifacts "
            f"and status counts `{evidence.get('report_embedded_audit_status_counts')}`."
        ),
        (
            f"- Artifact audit: `{evidence.get('audit_path')}` with `{evidence.get('audit_artifact_count')}` "
            f"artifacts, status counts `{evidence.get('audit_status_counts')}`."
        ),
        "",
        "## Pre-Launch Checklist",
        "",
    ]
    for item in readiness.get("pre_launch_checklist") or []:
        if isinstance(item, Mapping):
            lines.append(f"- [ ] `{item.get('id')}`: {item.get('description')}")
    lines.extend(
        [
            "",
            "## Suggested Command",
            "",
            "```bash",
            str(stage_b.get("suggested_command") or ""),
            "```",
            "",
            "## Guardrails",
            "",
        ]
    )
    for item in readiness.get("guardrails") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _pre_launch_checklist() -> list[dict[str, Any]]:
    return [
        {
            "id": "confirm_user_gpu_choice",
            "required": True,
            "description": "Confirm whether the next GPU job is the default Stage B manifest or an explicit Stage A resume from index 478.",
        },
        {
            "id": "rerun_process_check",
            "required": True,
            "description": "Immediately before launch, rerun process/GPU checks to confirm no duplicate sweep process is active and GPU memory is available.",
        },
        {
            "id": "verify_artifact_audit_ok",
            "required": True,
            "description": "Confirm the latest grid128 artifact audit reports ok status for all required report, review, and plan artifacts.",
        },
        {
            "id": "launch_stage_b_manifest_or_explicit_stage_a_resume",
            "required": True,
            "description": "Launch the Stage B manifest command only after approval; use Stage A resume only if that alternative was explicitly selected.",
        },
    ]


def _guardrails() -> list[str]:
    return [
        "Do not silently resume Stage A; resume from index 478 only by explicit choice.",
        "Do not overwrite stopped Stage A outputs; Stage B output directory is the stage_b_sweep directory under the Stage B plan.",
        "Broad pixel families remain deferred because archived OOMs dominate them; included pixel models are scouts only.",
        "Run the manifest command only after explicit approval for a GPU job.",
        "If CUDA memory fails at batch size 4, retry Stage B with a smaller batch size before expanding pixel families.",
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_progress_log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    status_counts: Counter[str] = Counter()
    record_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            continue
        record_count += 1
        latest = dict(payload)
        if payload.get("status") is not None:
            status_counts[str(payload["status"])] += 1
    if not latest:
        return {}
    return {
        "latest": latest,
        "record_count": record_count,
        "status_counts": dict(status_counts),
    }
