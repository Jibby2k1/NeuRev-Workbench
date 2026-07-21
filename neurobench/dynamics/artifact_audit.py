"""Audit key grid-dynamics report, review, and planning artifacts."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    relative_path: str
    kind: str
    parse_json: bool = False


DEFAULT_GRID128_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("Sweep live status", "sweeps/grid128_sequence_1day_v1/sweep_live_status.md", "sweep_status"),
    ArtifactSpec("Sweep health report", "sweeps/grid128_sequence_1day_v1/sweep_health_report.md", "sweep_status"),
    ArtifactSpec("Comparison manifest", "comparison_grid128_sequence_1day_v1/comparison_manifest.json", "comparison", True),
    ArtifactSpec("Results intelligence", "comparison_grid128_sequence_1day_v1/results_intelligence.json", "comparison", True),
    ArtifactSpec("Comparison dashboard", "comparison_grid128_sequence_1day_v1/comparison_dashboard.html", "comparison"),
    ArtifactSpec("Partial experiment report", "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.md", "report"),
    ArtifactSpec("Partial experiment report JSON", "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json", "report", True),
    ArtifactSpec("Stage B plan", "plans/grid128_sequence_stage_b_v1/next_sweep_plan.md", "plan"),
    ArtifactSpec("Stage B plan JSON", "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json", "plan", True),
    ArtifactSpec("Stage B manifest", "plans/grid128_sequence_stage_b_v1/next_sweep_manifest.json", "plan", True),
    ArtifactSpec("Stage B dry-run manifest", "plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json", "plan", True),
    ArtifactSpec("Stage A stop review", "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.md", "plan"),
    ArtifactSpec("Stage A stop review JSON", "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.json", "plan", True),
    ArtifactSpec("Stage B launch readiness", "plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.md", "plan"),
    ArtifactSpec("Stage B launch readiness JSON", "plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.json", "plan", True),
    ArtifactSpec("Best-test video review", "reviews/grid128_video_error_review_best_test_v1/video_error_review.json", "review", True),
    ArtifactSpec("Best-test video review HTML", "reviews/grid128_video_error_review_best_test_v1/video_error_review.html", "review"),
    ArtifactSpec("Active-cell video review", "reviews/grid128_active_cell_review_v1/video_error_review.json", "review", True),
    ArtifactSpec("Active-cell video review HTML", "reviews/grid128_active_cell_review_v1/video_error_review.html", "review"),
    ArtifactSpec("Shared-horizon clip review", "reviews/shared_horizon_neural_clip_review_v1/video_error_review.json", "review", True),
    ArtifactSpec("Shared-horizon clip review HTML", "reviews/shared_horizon_neural_clip_review_v1/video_error_review.html", "review"),
    ArtifactSpec("Shared-horizon baseline comparison", "reports/shared_horizon_baseline_comparison_v1/shared_horizon_baseline_comparison.md", "report"),
    ArtifactSpec("Shared-horizon baseline comparison JSON", "reports/shared_horizon_baseline_comparison_v1/shared_horizon_baseline_comparison.json", "report", True),
    ArtifactSpec("Shared-horizon status", "plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_status.md", "plan"),
    ArtifactSpec("Shared-horizon status JSON", "plans/shared_horizon_neural_grid_v1/shared_horizon_neural_grid_status.json", "plan", True),
    ArtifactSpec("Active-cell rescue plan", "plans/active_cell_rescue_v1/active_cell_rescue_plan.md", "plan"),
    ArtifactSpec("Active-cell rescue plan JSON", "plans/active_cell_rescue_v1/active_cell_rescue_plan.json", "plan", True),
    ArtifactSpec("Latent objective plan", "plans/latent_objective_plan_v1/latent_objective_plan.md", "plan"),
    ArtifactSpec("Latent objective plan JSON", "plans/latent_objective_plan_v1/latent_objective_plan.json", "plan", True),
    ArtifactSpec("Latent-head smoke report", "plans/latent_head_smoke_v1/latent_classifier_report.md", "report"),
    ArtifactSpec("Latent-head smoke run JSON", "plans/latent_head_smoke_v1/latent_classifier_run.json", "report", True),
    ArtifactSpec("Current active-cell backfill preflight", "plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.json", "plan", True),
    ArtifactSpec("Current active-cell backfill preflight Markdown", "plans/grid128_backfill_preflight_v1/current_active_cell_leader_metric_backfill_preflight.md", "plan"),
    ArtifactSpec("Current learned-leader backfill preflight", "plans/grid128_backfill_preflight_v1/current_learned_leader_metric_backfill_preflight.json", "plan", True),
    ArtifactSpec("Current learned-leader backfill preflight Markdown", "plans/grid128_backfill_preflight_v1/current_learned_leader_metric_backfill_preflight.md", "plan"),
    ArtifactSpec("Current active-cell challenger backfill preflight", "plans/grid128_backfill_preflight_v1/current_active_cell_challenger_metric_backfill_preflight.json", "plan", True),
    ArtifactSpec("Current active-cell challenger backfill preflight Markdown", "plans/grid128_backfill_preflight_v1/current_active_cell_challenger_metric_backfill_preflight.md", "plan"),
)


def build_grid128_artifact_audit(
    *,
    root: Path,
    out_dir: Path,
    title: str = "Grid128 Artifact Audit",
    artifact_specs: Sequence[ArtifactSpec] = DEFAULT_GRID128_ARTIFACTS,
) -> dict[str, Any]:
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()

    rows = [_audit_one(root, spec) for spec in artifact_specs]
    status_counts = dict(Counter(row["status"] for row in rows))
    kind_counts = dict(Counter(row["kind"] for row in rows))
    report = {
        "schema_version": 1,
        "title": title,
        "created_at": created_at,
        "root": str(root),
        "artifact_count": len(rows),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "ok": all(row["status"] == "ok" for row in rows),
        "artifacts": rows,
    }
    json_path = out / "grid128_artifact_audit.json"
    markdown_path = out / "grid128_artifact_audit.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_grid128_artifact_audit_markdown(report), encoding="utf-8")
    return report


def render_grid128_artifact_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('title') or 'Grid128 Artifact Audit'}",
        "",
        f"Generated: `{report.get('created_at')}`",
        f"Root: `{report.get('root')}`",
        f"Artifacts checked: `{report.get('artifact_count')}`",
        f"Status counts: `{report.get('status_counts')}`",
        f"Overall OK: `{report.get('ok')}`",
        "",
        "## Artifacts",
        "",
        "| Label | Kind | Status | Size | Summary | Path |",
        "|---|---|---|---:|---|---|",
    ]
    for row in report.get("artifacts") or []:
        summary = _markdown_cell(row.get("summary_text") or "")
        lines.append(
            f"| {_markdown_cell(row.get('label'))} | {_markdown_cell(row.get('kind'))} | "
            f"{_markdown_cell(row.get('status'))} | {row.get('size_bytes') or 0} | "
            f"{summary} | `{row.get('relative_path')}` |"
        )
    missing = [row for row in report.get("artifacts") or [] if row.get("status") != "ok"]
    if missing:
        lines.extend(["", "## Attention", ""])
        for row in missing:
            detail = row.get("error") or row.get("status")
            lines.append(f"- `{row.get('relative_path')}`: {detail}")
    return "\n".join(lines) + "\n"


def _audit_one(root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    path = root / spec.relative_path
    row: dict[str, Any] = {
        "label": spec.label,
        "relative_path": spec.relative_path,
        "path": str(path),
        "kind": spec.kind,
        "parse_json": spec.parse_json,
        "exists": path.exists(),
        "status": "missing",
        "size_bytes": 0,
        "summary": {},
        "summary_text": "",
    }
    if not path.exists():
        return row
    row["size_bytes"] = path.stat().st_size
    if spec.parse_json:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - exact decoder varies by Python version
            row["status"] = "invalid_json"
            row["error"] = str(exc)
            return row
        row["summary"] = _summarize_json_payload(payload)
        active_summary = _summarize_report_active_consistency(payload, root=root, spec=spec)
        if active_summary:
            row["active_summary"] = active_summary
            row["summary"].update(active_summary)
            if active_summary.get("active_summary_matches_sweep") is False:
                row["status"] = "stale_active_summary"
                row["error"] = "report active_sweep_summary does not match current sweep status"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        embedded_audit_summary = _summarize_report_embedded_audit_consistency(payload, root=root, spec=spec)
        if embedded_audit_summary:
            row["embedded_audit_summary"] = embedded_audit_summary
            row["summary"].update(embedded_audit_summary)
            if embedded_audit_summary.get("embedded_audit_summary_matches_current_state") is False:
                row["status"] = "stale_embedded_audit_summary"
                row["error"] = "report artifact_audit_summary does not match current artifact state"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        stage_b_summary = _summarize_stage_b_manifest_consistency(payload, root=root, spec=spec)
        if stage_b_summary:
            row["stage_b_summary"] = stage_b_summary
            row["summary"].update(stage_b_summary)
            if stage_b_summary.get("stage_b_manifest_matches_plan") is False:
                row["status"] = "stale_stage_b_manifest"
                row["error"] = "Stage B plan does not match next_sweep_manifest.json"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        stage_b_source_progress_summary = _summarize_stage_b_source_progress_consistency(payload, root=root, spec=spec)
        if stage_b_source_progress_summary:
            row["stage_b_source_progress_summary"] = stage_b_source_progress_summary
            row["summary"].update(stage_b_source_progress_summary)
            if stage_b_source_progress_summary.get("stage_b_source_progress_matches_sweep") is False:
                row["status"] = "stale_stage_b_source_progress"
                row["error"] = "Stage B plan source progress does not match current sweep progress"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        stage_b_dry_run_summary = _summarize_stage_b_dry_run_consistency(payload, root=root, spec=spec)
        if stage_b_dry_run_summary:
            row["stage_b_dry_run_summary"] = stage_b_dry_run_summary
            row["summary"].update(stage_b_dry_run_summary)
            if stage_b_dry_run_summary.get("stage_b_dry_run_matches_manifest") is False:
                row["status"] = "stale_stage_b_dry_run"
                row["error"] = "Stage B dry-run sweep_manifest.json does not match next_sweep_manifest.json"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        stage_a_stop_review_summary = _summarize_stage_a_stop_review_consistency(payload, root=root, spec=spec)
        if stage_a_stop_review_summary:
            row["stage_a_stop_review_summary"] = stage_a_stop_review_summary
            row["summary"].update(stage_a_stop_review_summary)
            if stage_a_stop_review_summary.get("stage_a_stop_review_matches_current_state") is False:
                row["status"] = "stale_stage_a_stop_review"
                row["error"] = "Stage A stopped-run review does not match current sweep or Stage B state"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        stage_b_launch_readiness_summary = _summarize_stage_b_launch_readiness_consistency(payload, root=root, spec=spec)
        if stage_b_launch_readiness_summary:
            row["stage_b_launch_readiness_summary"] = stage_b_launch_readiness_summary
            row["summary"].update(stage_b_launch_readiness_summary)
            if stage_b_launch_readiness_summary.get("stage_b_launch_readiness_matches_current_state") is False:
                row["status"] = "stale_stage_b_launch_readiness"
                row["error"] = "Stage B launch readiness does not match current stopped-run or Stage B state"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        manifest_reference_summary = _summarize_comparison_manifest_references(payload, root=root, artifact_path=path, spec=spec)
        if manifest_reference_summary:
            row["manifest_reference_summary"] = manifest_reference_summary
            row["summary"].update(
                {
                    "referenced_file_count": manifest_reference_summary["referenced_file_count"],
                    "missing_referenced_file_count": manifest_reference_summary["missing_referenced_file_count"],
                    "referenced_metric_file_count": manifest_reference_summary["referenced_metric_file_count"],
                    "referenced_prediction_file_count": manifest_reference_summary["referenced_prediction_file_count"],
                }
            )
            if manifest_reference_summary["missing_referenced_file_count"]:
                row["status"] = "missing_reference"
                row["error"] = f"missing referenced files: {manifest_reference_summary['missing_referenced_file_count']}"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        review_count_summary = _summarize_review_count_consistency(payload, spec=spec)
        if review_count_summary:
            row["review_count_summary"] = review_count_summary
            row["summary"].update(review_count_summary)
            if review_count_summary.get("review_counts_match") is False:
                row["status"] = "stale_review_counts"
                row["error"] = "review count fields do not match review arrays"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        preflight_reference_summary = _summarize_backfill_preflight_references(payload, root=root, artifact_path=path)
        if preflight_reference_summary:
            row["preflight_reference_summary"] = preflight_reference_summary
            row["summary"].update(
                {
                    "referenced_file_count": preflight_reference_summary["referenced_file_count"],
                    "missing_referenced_file_count": preflight_reference_summary["missing_referenced_file_count"],
                    "preflight_input_reference_count": preflight_reference_summary["referenced_file_count"],
                    "missing_preflight_input_reference_count": preflight_reference_summary["missing_referenced_file_count"],
                }
            )
            if preflight_reference_summary["missing_referenced_file_count"]:
                row["status"] = "missing_reference"
                row["error"] = f"missing referenced files: {preflight_reference_summary['missing_referenced_file_count']}"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        report_reference_summary = _summarize_report_artifact_references(payload, root=root, artifact_path=path, spec=spec)
        if report_reference_summary:
            row["report_reference_summary"] = report_reference_summary
            row["summary"].update(
                {
                    "referenced_file_count": report_reference_summary["referenced_file_count"],
                    "missing_referenced_file_count": report_reference_summary["missing_referenced_file_count"],
                    "report_artifact_reference_count": report_reference_summary["referenced_file_count"],
                    "missing_report_artifact_reference_count": report_reference_summary["missing_referenced_file_count"],
                }
            )
            if report_reference_summary["missing_referenced_file_count"]:
                row["status"] = "missing_reference"
                row["error"] = f"missing referenced files: {report_reference_summary['missing_referenced_file_count']}"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        reference_summary = _summarize_referenced_files(payload, root=root, artifact_path=path)
        if reference_summary:
            row["reference_summary"] = reference_summary
            row["summary"].update(
                {
                    "referenced_file_count": reference_summary["referenced_file_count"],
                    "missing_referenced_file_count": reference_summary["missing_referenced_file_count"],
                }
            )
            if reference_summary["missing_referenced_file_count"]:
                row["status"] = "missing_reference"
                row["error"] = f"missing referenced files: {reference_summary['missing_referenced_file_count']}"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        row["summary_text"] = _summary_text(row["summary"])
    else:
        text_payload = path.read_text(encoding="utf-8", errors="replace")
        sweep_status_summary = _summarize_sweep_status_markdown_consistency(text_payload, root=root, spec=spec)
        if sweep_status_summary:
            row["sweep_status_summary"] = sweep_status_summary
            row["summary"].update(sweep_status_summary)
            if sweep_status_summary.get("sweep_status_matches_sweep") is False:
                row["status"] = "stale_sweep_status"
                row["error"] = "sweep status Markdown does not match current sweep state"
                row["summary_text"] = _summary_text(row["summary"])
                return row
        row["summary_text"] = _summary_text(row["summary"])
    row["status"] = "ok"
    return row


def _summarize_json_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"json_type": type(payload).__name__}
    summary: dict[str, Any] = {}
    for key in (
        "schema_version",
        "title",
        "created_at",
        "generated_at",
        "status_counts",
        "selected_model_count",
        "missing_visual_count",
        "temporal_clip_model_count",
        "planned_experiment_count",
        "completed_shared_neural_count",
        "pending_shared_neural_count",
        "active_cell_warning_count",
        "diagnosis",
        "dry_run",
        "architecture",
        "dataset_window_count",
        "estimated_metric_batches",
        "estimated_uncompressed_gib",
        "would_update_metrics",
        "would_backfill_metrics",
        "example_count",
    ):
        if key in payload:
            summary[key] = payload[key]
    if "models" in payload and isinstance(payload["models"], list):
        summary["model_count"] = len(payload["models"])
    if "rows" in payload and isinstance(payload["rows"], list):
        summary["row_count"] = len(payload["rows"])
    if "runs" in payload and isinstance(payload["runs"], list):
        summary["run_count"] = len(payload["runs"])
    if "horizon_summary" in payload and isinstance(payload["horizon_summary"], list):
        summary["horizon_summary_count"] = len(payload["horizon_summary"])
    if "active_cell_warnings" in payload and isinstance(payload["active_cell_warnings"], list):
        summary["active_cell_warning_count"] = len(payload["active_cell_warnings"])
    if "artifacts" in payload and isinstance(payload["artifacts"], list):
        summary["artifact_count"] = len(payload["artifacts"])
    if "would_write_files" in payload and isinstance(payload["would_write_files"], list):
        summary["would_write_file_count"] = len(payload["would_write_files"])
    if "recommended_candidates" in payload and isinstance(payload["recommended_candidates"], list) and payload["recommended_candidates"]:
        first = payload["recommended_candidates"][0]
        if isinstance(first, Mapping):
            summary["next_candidate"] = first.get("config_id")
    if "gate_summary" in payload and isinstance(payload["gate_summary"], Mapping):
        summary["gate_interpretation"] = payload["gate_summary"].get("interpretation")
    if "best_overall" in payload and isinstance(payload["best_overall"], Mapping):
        best_overall = payload["best_overall"]
        summary["best_overall_label"] = best_overall.get("label")
        summary["best_overall_improvement_over_persistence_mse"] = best_overall.get("improvement_over_persistence_mse")
    return summary


def _summary_text(summary: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, sort_keys=True)
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text}")
    return "; ".join(parts)


def _summarize_sweep_status_markdown_consistency(text: str, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path not in {
        "sweeps/grid128_sequence_1day_v1/sweep_live_status.md",
        "sweeps/grid128_sequence_1day_v1/sweep_health_report.md",
    }:
        return {}
    active_path = root / "sweeps/grid128_sequence_1day_v1/sweep_active.json"
    progress_path = root / "sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl"
    active = _load_json_if_exists(active_path)
    latest_progress = _load_latest_progress(progress_path)
    if not active and not latest_progress:
        return {"sweep_status_check": "not_available"}
    parsed = _parse_sweep_status_markdown(text)
    expected_progress = "unknown"
    if latest_progress:
        expected_progress = f"{latest_progress.get('index', 'unknown')} / {latest_progress.get('experiment_count', 'unknown')}"
    mismatches: list[str] = []
    if expected_progress != "unknown" and parsed.get("progress") != expected_progress:
        mismatches.append("progress")
    comparisons = {
        "active_index": active.get("index"),
        "active_experiment_id": active.get("experiment_id"),
        "active_status": active.get("status"),
    }
    observed = {
        "active_index": parsed.get("active_index"),
        "active_experiment_id": parsed.get("active_experiment_id"),
        "active_status": parsed.get("active_status"),
    }
    for key, expected in comparisons.items():
        if expected is None:
            continue
        if str(observed.get(key)) != str(expected):
            mismatches.append(key)
    stale_warning_present = "Progress file is stale" in text or "Progress appears stale" in text
    stopped_marker_present = "progress age reflects a stopped run" in text
    active_state = str(active.get("status") or "")
    if active_state in {"completed", "failed", "stopped"}:
        if stale_warning_present:
            mismatches.append("stopped_run_stale_warning")
        if spec.relative_path == "sweeps/grid128_sequence_1day_v1/sweep_health_report.md" and not stopped_marker_present:
            mismatches.append("stopped_run_marker")
    return {
        "sweep_status_check": "compared",
        "sweep_status_matches_sweep": not mismatches,
        "sweep_status_mismatches": mismatches,
        "sweep_status_stale_warning_present": stale_warning_present,
        "sweep_status_stopped_marker_present": stopped_marker_present,
        "report_progress": parsed.get("progress"),
        "expected_progress": expected_progress,
        "report_active_index": parsed.get("active_index"),
        "expected_active_index": active.get("index"),
        "report_active_experiment_id": parsed.get("active_experiment_id"),
        "expected_active_experiment_id": active.get("experiment_id"),
    }


def _parse_sweep_status_markdown(text: str) -> dict[str, Any]:
    progress_match = re.search(r"^Progress:\s*`?([^`\n]+?)`?\s*/\s*`?([^`\n]+?)`?\s*$", text, flags=re.MULTILINE)
    parsed: dict[str, Any] = {}
    if progress_match:
        parsed["progress"] = f"{progress_match.group(1).strip()} / {progress_match.group(2).strip()}"
    table_values: dict[str, str] = {}
    for match in re.finditer(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", text, flags=re.MULTILINE):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key in {"Field", "---"}:
            continue
        table_values[key] = value
    if table_values.get("status"):
        parsed["active_status"] = table_values["status"]
    if table_values.get("index"):
        parsed["active_index"] = table_values["index"]
    if table_values.get("experiment"):
        parsed["active_experiment_id"] = table_values["experiment"]
    return parsed


def _summarize_report_active_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    report_active = payload.get("active_sweep_summary") if isinstance(payload.get("active_sweep_summary"), Mapping) else {}
    active_path = root / "sweeps/grid128_sequence_1day_v1/sweep_active.json"
    progress_path = root / "sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl"
    active = _load_json_if_exists(active_path)
    latest_progress = _load_latest_progress(progress_path)
    if not report_active and not active:
        return {"active_summary_check": "not_available"}
    expected_progress = "unknown"
    if latest_progress:
        expected_progress = f"{latest_progress.get('index', 'unknown')} / {latest_progress.get('experiment_count', 'unknown')}"
    mismatches: list[str] = []
    comparisons = {
        "status": active.get("status"),
        "index": active.get("index"),
        "experiment_id": active.get("experiment_id"),
        "dataset_key": active.get("dataset_key"),
        "kind": active.get("kind"),
    }
    for key, expected in comparisons.items():
        if expected is None:
            continue
        observed = report_active.get(key)
        if str(observed) != str(expected):
            mismatches.append(key)
    if expected_progress != "unknown" and str(report_active.get("progress")) != expected_progress:
        mismatches.append("progress")
    return {
        "active_summary_check": "compared",
        "active_summary_matches_sweep": not mismatches,
        "active_summary_mismatches": mismatches,
        "report_active_progress": report_active.get("progress"),
        "expected_active_progress": expected_progress,
        "report_active_index": report_active.get("index"),
        "expected_active_index": active.get("index"),
        "report_active_experiment_id": report_active.get("experiment_id"),
        "expected_active_experiment_id": active.get("experiment_id"),
    }



def _summarize_report_embedded_audit_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    embedded = payload.get("artifact_audit_summary")
    if not isinstance(embedded, Mapping):
        return {}
    mismatches: list[str] = []
    latest_progress = _load_latest_progress(root / "sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl")
    progress_summary = _load_progress_log_summary(root / "sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl")
    expected_progress = "unknown"
    if latest_progress:
        expected_progress = f"{latest_progress.get('index', 'unknown')} / {latest_progress.get('experiment_count', 'unknown')}"

    check_details: dict[str, list[str]] = {}
    for check in embedded.get("consistency_checks") or []:
        if not isinstance(check, Mapping):
            continue
        key = str(check.get("check") or "")
        detail = str(check.get("detail") or "")
        check_details.setdefault(key, []).append(detail)

    expected_progress_detail = f"report={expected_progress} expected={expected_progress}"
    progress_detail_checks = ("active_sweep_summary",)
    for check_name in progress_detail_checks:
        for detail in check_details.get(check_name, []):
            if detail != expected_progress_detail:
                mismatches.append(check_name)
                break

    stage_b_plan = _load_json_if_exists(root / "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json")
    if stage_b_plan and progress_summary and check_details.get("stage_b_source_progress"):
        plan_progress = stage_b_plan.get("progress_summary") if isinstance(stage_b_plan.get("progress_summary"), Mapping) else {}
        latest = progress_summary.get("latest") if isinstance(progress_summary.get("latest"), Mapping) else {}
        expected_stage_b_detail = (
            f"plan={plan_progress.get('current_index')} "
            f"source={latest.get('index')} records={progress_summary.get('record_count')}"
        )
        for detail in check_details.get("stage_b_source_progress", []):
            if detail != expected_stage_b_detail:
                mismatches.append("stage_b_source_progress")
                break

    current_audit = _load_json_if_exists(root / "plans/grid128_artifact_audit_v1/grid128_artifact_audit.json")
    if current_audit:
        if str(embedded.get("artifact_count")) != str(current_audit.get("artifact_count")):
            mismatches.append("artifact_count")
        embedded_status_counts = embedded.get("status_counts") if isinstance(embedded.get("status_counts"), Mapping) else {}
        current_status_counts = current_audit.get("status_counts") if isinstance(current_audit.get("status_counts"), Mapping) else {}
        if dict(embedded_status_counts) != dict(current_status_counts):
            mismatches.append("status_counts")

    embedded_reference_rows = embedded.get("review_reference_counts") if isinstance(embedded.get("review_reference_counts"), list) else []
    embedded_comparison = next(
        (row for row in embedded_reference_rows if isinstance(row, Mapping) and row.get("label") == "Comparison manifest"),
        None,
    )
    expected_comparison = _current_comparison_manifest_reference_summary(root)
    if embedded_comparison and expected_comparison:
        for key in (
            "referenced_file_count",
            "missing_referenced_file_count",
            "referenced_metric_file_count",
            "referenced_prediction_file_count",
        ):
            if str(embedded_comparison.get(key)) != str(expected_comparison.get(key)):
                mismatches.append(f"comparison_manifest.{key}")

    return {
        "embedded_audit_check": "compared",
        "embedded_audit_summary_matches_current_state": not mismatches,
        "embedded_audit_summary_mismatches": mismatches,
        "embedded_audit_created_at": embedded.get("created_at"),
        "embedded_audit_expected_progress": expected_progress,
        "embedded_audit_artifact_count": embedded.get("artifact_count"),
        "expected_audit_artifact_count": current_audit.get("artifact_count") if current_audit else None,
        "embedded_audit_status_counts": embedded.get("status_counts") if isinstance(embedded.get("status_counts"), Mapping) else {},
        "expected_audit_status_counts": current_audit.get("status_counts") if isinstance(current_audit.get("status_counts"), Mapping) else {},
        "embedded_audit_comparison_reference_count": embedded_comparison.get("referenced_file_count") if embedded_comparison else None,
        "expected_comparison_reference_count": expected_comparison.get("referenced_file_count") if expected_comparison else None,
        "embedded_audit_comparison_metric_count": embedded_comparison.get("referenced_metric_file_count") if embedded_comparison else None,
        "expected_comparison_metric_count": expected_comparison.get("referenced_metric_file_count") if expected_comparison else None,
        "embedded_audit_comparison_prediction_count": embedded_comparison.get("referenced_prediction_file_count") if embedded_comparison else None,
        "expected_comparison_prediction_count": expected_comparison.get("referenced_prediction_file_count") if expected_comparison else None,
    }


def _current_comparison_manifest_reference_summary(root: Path) -> dict[str, Any]:
    manifest_path = root / "comparison_grid128_sequence_1day_v1/comparison_manifest.json"
    manifest = _load_json_if_exists(manifest_path)
    if not manifest:
        return {}
    spec = ArtifactSpec("Comparison manifest", "comparison_grid128_sequence_1day_v1/comparison_manifest.json", "comparison", True)
    return _summarize_comparison_manifest_references(manifest, root=root, artifact_path=manifest_path, spec=spec)


def _summarize_stage_b_manifest_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    manifest_value = payload.get("manifest_path")
    manifest_path = _resolve_reference_path(str(manifest_value), root=root, artifact_path=root / spec.relative_path) if manifest_value else root / "plans/grid128_sequence_stage_b_v1/next_sweep_manifest.json"
    manifest = _load_json_if_exists(manifest_path)
    if not manifest:
        return {
            "stage_b_manifest_check": "missing_or_invalid",
            "stage_b_manifest_matches_plan": False,
            "stage_b_manifest_path": str(manifest_path),
        }
    plan_count = payload.get("planned_experiment_count")
    manifest_count = manifest.get("planned_experiment_count")
    manifest_len = len(manifest.get("experiments") or []) if isinstance(manifest.get("experiments"), list) else None
    plan_progress = payload.get("progress_summary") if isinstance(payload.get("progress_summary"), Mapping) else {}
    manifest_progress = manifest.get("progress_summary") if isinstance(manifest.get("progress_summary"), Mapping) else {}
    mismatches: list[str] = []
    if str(plan_count) != str(manifest_count):
        mismatches.append("planned_experiment_count")
    if manifest_len is not None and str(plan_count) != str(manifest_len):
        mismatches.append("manifest_experiment_count")
    for key in ("current_index", "experiment_count", "last_experiment_id", "last_status"):
        plan_value = plan_progress.get(key)
        manifest_value = manifest_progress.get(key)
        if plan_value is not None and manifest_value is not None and str(plan_value) != str(manifest_value):
            mismatches.append(f"progress_summary.{key}")
    return {
        "stage_b_manifest_check": "compared",
        "stage_b_manifest_matches_plan": not mismatches,
        "stage_b_manifest_mismatches": mismatches,
        "stage_b_plan_count": plan_count,
        "stage_b_manifest_count": manifest_count,
        "stage_b_manifest_experiment_count": manifest_len,
        "stage_b_plan_progress_index": plan_progress.get("current_index"),
        "stage_b_manifest_progress_index": manifest_progress.get("current_index"),
    }



def _summarize_stage_b_source_progress_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    plan_progress = payload.get("progress_summary") if isinstance(payload.get("progress_summary"), Mapping) else {}
    source_value = payload.get("source_sweep_dir")
    source_sweep_dir = (
        _resolve_reference_path(str(source_value), root=root, artifact_path=root / spec.relative_path)
        if source_value
        else root / "sweeps/grid128_sequence_1day_v1"
    )
    source_progress_path = source_sweep_dir / "sweep_progress.jsonl"
    source_progress = _load_progress_log_summary(source_progress_path)
    if not source_progress:
        if not source_value and not source_progress_path.exists():
            return {
                "stage_b_source_progress_check": "not_available",
                "stage_b_source_sweep_dir": str(source_sweep_dir),
            }
        return {
            "stage_b_source_progress_check": "missing_or_invalid",
            "stage_b_source_progress_matches_sweep": False,
            "stage_b_source_sweep_dir": str(source_sweep_dir),
        }
    latest = source_progress["latest"]
    mismatches: list[str] = []
    comparisons = {
        "current_index": latest.get("index"),
        "experiment_count": latest.get("experiment_count"),
        "last_experiment_id": latest.get("experiment_id"),
        "last_status": latest.get("status"),
        "current_records": source_progress.get("record_count"),
    }
    for key, expected in comparisons.items():
        observed = plan_progress.get(key)
        if observed is not None and str(observed) != str(expected):
            mismatches.append(key)
    observed_counts = plan_progress.get("current_status_counts")
    expected_counts = source_progress.get("status_counts")
    if isinstance(observed_counts, Mapping) and dict(observed_counts) != dict(expected_counts):
        mismatches.append("current_status_counts")
    return {
        "stage_b_source_progress_check": "compared",
        "stage_b_source_progress_matches_sweep": not mismatches,
        "stage_b_source_progress_mismatches": mismatches,
        "stage_b_plan_progress_index": plan_progress.get("current_index"),
        "stage_b_source_progress_index": latest.get("index"),
        "stage_b_plan_progress_experiment_id": plan_progress.get("last_experiment_id"),
        "stage_b_source_progress_experiment_id": latest.get("experiment_id"),
        "stage_b_plan_progress_records": plan_progress.get("current_records"),
        "stage_b_source_progress_records": source_progress.get("record_count"),
    }


def _summarize_stage_b_dry_run_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    stage_b_manifest_path = root / "plans/grid128_sequence_stage_b_v1/next_sweep_manifest.json"
    stage_b_manifest = _load_json_if_exists(stage_b_manifest_path)
    if not stage_b_manifest:
        return {
            "stage_b_dry_run_check": "missing_or_invalid_source_manifest",
            "stage_b_dry_run_matches_manifest": False,
            "stage_b_source_manifest_path": str(stage_b_manifest_path),
        }
    dry_run_count = payload.get("experiment_count")
    dry_run_experiments = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
    source_count = stage_b_manifest.get("planned_experiment_count")
    source_experiments = stage_b_manifest.get("experiments") if isinstance(stage_b_manifest.get("experiments"), list) else []
    dry_run_ids = [item.get("experiment_id") for item in dry_run_experiments if isinstance(item, Mapping)]
    source_ids = [item.get("experiment_id") for item in source_experiments if isinstance(item, Mapping)]
    mismatches: list[str] = []
    if str(dry_run_count) != str(source_count):
        mismatches.append("experiment_count")
    if len(dry_run_experiments) != len(source_experiments):
        mismatches.append("experiment_list_length")
    if dry_run_ids != source_ids:
        mismatches.append("experiment_id_order")
    return {
        "stage_b_dry_run_check": "compared",
        "stage_b_dry_run_matches_manifest": not mismatches,
        "stage_b_dry_run_mismatches": mismatches,
        "stage_b_source_manifest_count": source_count,
        "stage_b_dry_run_count": dry_run_count,
        "stage_b_source_manifest_experiment_count": len(source_experiments),
        "stage_b_dry_run_experiment_count": len(dry_run_experiments),
    }

def _summarize_stage_a_stop_review_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    progress = payload.get("progress") if isinstance(payload.get("progress"), Mapping) else {}
    latest_status = payload.get("latest_active_status") if isinstance(payload.get("latest_active_status"), Mapping) else {}
    stage_b = payload.get("stage_b") if isinstance(payload.get("stage_b"), Mapping) else {}
    source_progress = _load_progress_log_summary(root / "sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl")
    active = _load_json_if_exists(root / "sweeps/grid128_sequence_1day_v1/sweep_active.json")
    stage_b_plan = _load_json_if_exists(root / "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json")
    stage_b_dry_run = _load_json_if_exists(root / "plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json")
    report = _load_json_if_exists(root / "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json")
    current_audit = _load_json_if_exists(root / "plans/grid128_artifact_audit_v1/grid128_artifact_audit.json")
    if not source_progress:
        return {
            "stage_a_stop_review_check": "missing_or_invalid_source_progress",
            "stage_a_stop_review_matches_current_state": False,
        }

    latest = source_progress["latest"]
    mismatches: list[str] = []
    progress_comparisons = {
        "current_index": latest.get("index"),
        "experiment_count": latest.get("experiment_count"),
        "record_count": source_progress.get("record_count"),
    }
    for key, expected in progress_comparisons.items():
        if str(progress.get(key)) != str(expected):
            mismatches.append(f"progress.{key}")
    expected_progress_text = f"{latest.get('index')} / {latest.get('experiment_count')}"
    if str(progress.get("progress_text")) != expected_progress_text:
        mismatches.append("progress.progress_text")
    if dict(progress.get("status_counts") or {}) != dict(source_progress.get("status_counts") or {}):
        mismatches.append("progress.status_counts")

    active_comparisons = {
        "index": active.get("index"),
        "status": active.get("status"),
        "experiment_id": active.get("experiment_id"),
        "finished_at": active.get("finished_at"),
    }
    for key, expected in active_comparisons.items():
        if expected is None:
            continue
        if str(latest_status.get(key)) != str(expected):
            mismatches.append(f"latest_active_status.{key}")

    plan_count = stage_b_plan.get("planned_experiment_count")
    dry_run_count = stage_b_dry_run.get("experiment_count")
    if plan_count is not None and str(stage_b.get("planned_experiment_count")) != str(plan_count):
        mismatches.append("stage_b.planned_experiment_count")
    if dry_run_count is not None and str(stage_b.get("dry_run_experiment_count")) != str(dry_run_count):
        mismatches.append("stage_b.dry_run_experiment_count")
    if stage_b_plan.get("created_at") and str(stage_b.get("created_at")) != str(stage_b_plan.get("created_at")):
        mismatches.append("stage_b.created_at")
    if isinstance(stage_b_plan.get("selection_counts"), Mapping) and dict(stage_b.get("selection_counts") or {}) != dict(stage_b_plan.get("selection_counts") or {}):
        mismatches.append("stage_b.selection_counts")

    return {
        "stage_a_stop_review_check": "compared",
        "stage_a_stop_review_matches_current_state": not mismatches,
        "stage_a_stop_review_mismatches": mismatches,
        "stage_a_stop_review_progress": progress.get("progress_text"),
        "expected_stage_a_stop_progress": expected_progress_text,
        "stage_a_stop_review_record_count": progress.get("record_count"),
        "expected_stage_a_stop_record_count": source_progress.get("record_count"),
        "stage_a_stop_review_active_index": latest_status.get("index"),
        "expected_stage_a_stop_active_index": active.get("index"),
        "stage_a_stop_review_stage_b_count": stage_b.get("planned_experiment_count"),
        "expected_stage_b_count": plan_count,
    }


def _summarize_stage_b_launch_readiness_consistency(payload: Any, *, root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    stage_a = payload.get("stage_a") if isinstance(payload.get("stage_a"), Mapping) else {}
    stage_b = payload.get("stage_b") if isinstance(payload.get("stage_b"), Mapping) else {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), Mapping) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else {}
    source_progress = _load_progress_log_summary(root / "sweeps/grid128_sequence_1day_v1/sweep_progress.jsonl")
    active = _load_json_if_exists(root / "sweeps/grid128_sequence_1day_v1/sweep_active.json")
    stop_review = _load_json_if_exists(root / "plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.json")
    stage_b_plan = _load_json_if_exists(root / "plans/grid128_sequence_stage_b_v1/next_sweep_plan.json")
    stage_b_manifest = _load_json_if_exists(root / "plans/grid128_sequence_stage_b_v1/next_sweep_manifest.json")
    stage_b_dry_run = _load_json_if_exists(root / "plans/grid128_sequence_stage_b_v1/stage_b_sweep/sweep_manifest.json")
    report = _load_json_if_exists(root / "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json")
    current_audit = _load_json_if_exists(root / "plans/grid128_artifact_audit_v1/grid128_artifact_audit.json")
    if not source_progress:
        return {
            "stage_b_launch_readiness_check": "missing_or_invalid_source_progress",
            "stage_b_launch_readiness_matches_current_state": False,
        }

    latest = source_progress["latest"]
    mismatches: list[str] = []
    expected_progress_text = f"{latest.get('index')} / {latest.get('experiment_count')}"
    stage_a_comparisons = {
        "current_index": latest.get("index"),
        "experiment_count": latest.get("experiment_count"),
        "progress_text": expected_progress_text,
        "record_count": source_progress.get("record_count"),
        "active_status": active.get("status"),
        "active_experiment_id": active.get("experiment_id"),
    }
    for key, expected in stage_a_comparisons.items():
        if expected is None:
            continue
        if str(stage_a.get(key)) != str(expected):
            mismatches.append(f"stage_a.{key}")
    if dict(stage_a.get("status_counts") or {}) != dict(source_progress.get("status_counts") or {}):
        mismatches.append("stage_a.status_counts")

    plan_count = stage_b_plan.get("planned_experiment_count")
    manifest_count = stage_b_manifest.get("planned_experiment_count")
    manifest_len = len(stage_b_manifest.get("experiments") or []) if isinstance(stage_b_manifest.get("experiments"), list) else None
    dry_run_count = stage_b_dry_run.get("experiment_count")
    stage_b_comparisons = {
        "plan_created_at": stage_b_plan.get("created_at"),
        "planned_experiment_count": plan_count,
        "manifest_experiment_count": manifest_count,
        "manifest_list_count": manifest_len,
        "dry_run_experiment_count": dry_run_count,
    }
    for key, expected in stage_b_comparisons.items():
        if expected is None:
            continue
        if str(stage_b.get(key)) != str(expected):
            mismatches.append(f"stage_b.{key}")
    if isinstance(stage_b_plan.get("selection_counts"), Mapping) and dict(stage_b.get("selection_counts") or {}) != dict(stage_b_plan.get("selection_counts") or {}):
        mismatches.append("stage_b.selection_counts")

    stop_recommendation = stop_review.get("recommendation")
    if stop_recommendation and str(decision.get("recommendation")) != str(stop_recommendation):
        mismatches.append("decision.recommendation")
    if decision.get("default_next_gpu_job") != "stage_b_manifest":
        mismatches.append("decision.default_next_gpu_job")
    if decision.get("requires_user_approval") is not True:
        mismatches.append("decision.requires_user_approval")

    checklist = payload.get("pre_launch_checklist") if isinstance(payload.get("pre_launch_checklist"), list) else []
    checklist_ids = {str(item.get("id")) for item in checklist if isinstance(item, Mapping)}
    required_checklist_ids = {
        "confirm_user_gpu_choice",
        "rerun_process_check",
        "verify_artifact_audit_ok",
        "launch_stage_b_manifest_or_explicit_stage_a_resume",
    }
    missing_checklist_ids = sorted(required_checklist_ids - checklist_ids)
    if missing_checklist_ids:
        mismatches.append("pre_launch_checklist")

    report_summary = report.get("artifact_audit_summary") if isinstance(report.get("artifact_audit_summary"), Mapping) else {}
    evidence_comparisons = {
        "report_generated_at": report.get("generated_at"),
        "report_embedded_audit_artifact_count": report_summary.get("artifact_count"),
        "audit_artifact_count": current_audit.get("artifact_count"),
        "audit_ok": current_audit.get("ok"),
    }
    for key, expected in evidence_comparisons.items():
        if expected is None:
            continue
        if str(evidence.get(key)) != str(expected):
            mismatches.append(f"evidence.{key}")
    if dict(evidence.get("report_embedded_audit_status_counts") or {}) != dict(report_summary.get("status_counts") or {}):
        mismatches.append("evidence.report_embedded_audit_status_counts")
    if dict(evidence.get("audit_status_counts") or {}) != dict(current_audit.get("status_counts") or {}):
        mismatches.append("evidence.audit_status_counts")

    return {
        "stage_b_launch_readiness_check": "compared",
        "stage_b_launch_readiness_matches_current_state": not mismatches,
        "stage_b_launch_readiness_mismatches": mismatches,
        "stage_b_launch_readiness_status": payload.get("status"),
        "stage_b_launch_readiness_progress": stage_a.get("progress_text"),
        "expected_stage_b_launch_readiness_progress": expected_progress_text,
        "stage_b_launch_readiness_plan_count": stage_b.get("planned_experiment_count"),
        "expected_stage_b_launch_readiness_plan_count": plan_count,
        "stage_b_launch_readiness_dry_run_count": stage_b.get("dry_run_experiment_count"),
        "expected_stage_b_launch_readiness_dry_run_count": dry_run_count,
        "stage_b_launch_readiness_report_generated_at": evidence.get("report_generated_at"),
        "expected_report_generated_at": report.get("generated_at"),
        "stage_b_launch_readiness_audit_artifact_count": evidence.get("audit_artifact_count"),
        "expected_audit_artifact_count": current_audit.get("artifact_count"),
        "stage_b_launch_readiness_report_embedded_audit_artifact_count": evidence.get("report_embedded_audit_artifact_count"),
        "expected_report_embedded_audit_artifact_count": report_summary.get("artifact_count"),
        "stage_b_launch_readiness_checklist_count": len(checklist),
        "stage_b_launch_readiness_missing_checklist_ids": missing_checklist_ids,
    }


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_latest_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            latest = dict(payload)
    return latest


def _load_progress_log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    status_counts: Counter[str] = Counter()
    record_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        record_count += 1
        latest = dict(payload)
        status = payload.get("status")
        if status is not None:
            status_counts[str(status)] += 1
    if not latest:
        return {}
    return {
        "latest": latest,
        "record_count": record_count,
        "status_counts": dict(status_counts),
    }


def _summarize_comparison_manifest_references(payload: Any, *, root: Path, artifact_path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if Path(spec.relative_path).name != "comparison_manifest.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    references: list[dict[str, Any]] = []
    metric_fields = {"metrics_path"}
    prediction_fields = {"prediction_examples_path", "prediction_clip_examples_path"}
    for row in payload.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        experiment_id = row.get("experiment_id") or row.get("row_id") or ""
        for field in sorted(metric_fields | prediction_fields):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                continue
            record = _reference_record(field, value, root=root, artifact_path=artifact_path)
            if experiment_id:
                record["experiment_id"] = experiment_id
            references.append(record)
    if not references:
        return {
            "referenced_file_count": 0,
            "missing_referenced_file_count": 0,
            "referenced_metric_file_count": 0,
            "referenced_prediction_file_count": 0,
            "missing_references": [],
        }
    missing = [record for record in references if not record["exists"]]
    return {
        "referenced_file_count": len(references),
        "missing_referenced_file_count": len(missing),
        "referenced_metric_file_count": sum(1 for record in references if record.get("field") in metric_fields),
        "referenced_prediction_file_count": sum(1 for record in references if record.get("field") in prediction_fields),
        "missing_references": missing[:20],
    }



def _summarize_review_count_consistency(payload: Any, *, spec: ArtifactSpec) -> dict[str, Any]:
    if Path(spec.relative_path).name != "video_error_review.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    missing_rows = payload.get("missing_visual_example_rows") if isinstance(payload.get("missing_visual_example_rows"), list) else []
    expected_missing = len(missing_rows)
    declared_selected = payload.get("selected_model_count")
    declared_missing = payload.get("missing_visual_example_count", payload.get("missing_visual_count"))
    declared_temporal = payload.get("temporal_clip_model_count")
    temporal_count = sum(1 for model in models if isinstance(model, Mapping) and model.get("clip_panel_png"))
    mismatches: list[str] = []
    if declared_selected is not None and int(declared_selected) != len(models):
        mismatches.append("selected_model_count")
    if declared_missing is not None and int(declared_missing) != expected_missing:
        mismatches.append("missing_visual_example_count")
    if declared_temporal is not None and int(declared_temporal) != temporal_count:
        mismatches.append("temporal_clip_model_count")
    return {
        "review_count_check": "compared",
        "review_counts_match": not mismatches,
        "review_count_mismatches": mismatches,
        "declared_selected_model_count": declared_selected,
        "actual_model_count": len(models),
        "declared_missing_visual_count": declared_missing,
        "actual_missing_visual_count": expected_missing,
        "declared_temporal_clip_model_count": declared_temporal,
        "actual_temporal_clip_model_count": temporal_count,
    }

def _summarize_backfill_preflight_references(payload: Any, *, root: Path, artifact_path: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    if not payload.get("dry_run") or not payload.get("would_backfill_metrics"):
        return {}
    references: list[dict[str, Any]] = []
    for field in ("run_dir", "checkpoint_path", "metrics_path", "dataset_array_path"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            references.append(_reference_record(field, value, root=root, artifact_path=artifact_path))
    if not references:
        return {}
    missing = [record for record in references if not record["exists"]]
    return {
        "referenced_file_count": len(references),
        "missing_referenced_file_count": len(missing),
        "missing_references": missing,
    }


def _summarize_report_artifact_references(payload: Any, *, root: Path, artifact_path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    if spec.relative_path != "reports/grid128_sequence_1day_partial_report_v1/dynamics_experiment_report.json":
        return {}
    if not isinstance(payload, Mapping):
        return {}
    references: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(field: str, value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        key = (field, value)
        if key in seen:
            return
        seen.add(key)
        references.append(_reference_record(field, value, root=root, artifact_path=artifact_path))

    for field in ("report_path", "markdown_path", "comparison_manifest_path", "results_intelligence_path"):
        add(field, payload.get(field))
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), Mapping) else {}
    for label, value in sorted(artifacts.items()):
        add(f"artifacts.{label}", value)
    if not references:
        return {}
    missing = [record for record in references if not record["exists"]]
    return {
        "referenced_file_count": len(references),
        "missing_referenced_file_count": len(missing),
        "missing_references": missing,
    }


def _summarize_referenced_files(payload: Any, *, root: Path, artifact_path: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    references: list[dict[str, Any]] = []
    for field in ("html_path",):
        value = payload.get(field)
        if isinstance(value, str) and value:
            references.append(_reference_record(field, value, root=root, artifact_path=artifact_path))
    for model in payload.get("models") or []:
        if not isinstance(model, Mapping):
            continue
        experiment_id = model.get("experiment_id") or model.get("row_id") or ""
        for field in ("panel_png", "clip_panel_png"):
            value = model.get(field)
            if isinstance(value, str) and value:
                record = _reference_record(field, value, root=root, artifact_path=artifact_path)
                if experiment_id:
                    record["experiment_id"] = experiment_id
                references.append(record)
    if not references:
        return {}
    missing = [record for record in references if not record["exists"]]
    return {
        "referenced_file_count": len(references),
        "missing_referenced_file_count": len(missing),
        "missing_references": missing,
    }


def _reference_record(field: str, value: str, *, root: Path, artifact_path: Path) -> dict[str, Any]:
    path = _resolve_reference_path(value, root=root, artifact_path=artifact_path)
    return {
        "field": field,
        "path": value,
        "resolved_path": str(path),
        "exists": path.exists(),
    }


def _resolve_reference_path(value: str, *, root: Path, artifact_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [path, root / path, artifact_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")
