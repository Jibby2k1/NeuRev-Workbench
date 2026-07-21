"""Meeting-ready reports for partial grid dynamics sweeps."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurobench.dynamics.comparison import build_comparison_dashboard
from neurobench.dynamics.supervisor import build_sweep_health_report


def build_dynamics_experiment_report(
    *,
    sweep_dirs: Sequence[str | Path],
    out_dir: str | Path,
    comparison_dir: str | Path | None = None,
    title: str = "Grid Dynamics Experiment Report",
    refresh_dashboard: bool = False,
) -> dict[str, Any]:
    """Build a JSON and Markdown report from partial or completed sweep artifacts."""
    if not sweep_dirs:
        raise ValueError("At least one sweep directory is required.")
    sweeps = [Path(path) for path in sweep_dirs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison = Path(comparison_dir) if comparison_dir is not None else out / "comparison"
    manifest_path = comparison / "comparison_manifest.json"
    intelligence_path = comparison / "results_intelligence.json"
    if refresh_dashboard or not manifest_path.exists() or not intelligence_path.exists():
        build_comparison_dashboard(sweep_dirs=sweeps, out_dir=comparison)
    manifest = _load_json(manifest_path)
    intelligence = _load_json(intelligence_path)
    health = []
    for sweep in sweeps:
        if (sweep / "sweep_progress.jsonl").exists() or (sweep / "sweep_manifest.json").exists():
            health.append(build_sweep_health_report(sweep_dir=sweep))
    report = {
        "schema_version": 1,
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_dirs": [str(path) for path in sweeps],
        "comparison_dir": str(comparison),
        "comparison_manifest_path": str(manifest_path),
        "results_intelligence_path": str(intelligence_path),
        "health": health,
        "dataset_summary": _dataset_summary(manifest.get("datasets", {})),
        "run_summary": _run_summary(manifest, intelligence, health),
        "active_sweep_summary": _active_sweep_summary(health),
        "best_models": _best_models(intelligence),
        "baseline_comparison": _baseline_comparison(intelligence),
        "runtime_summary": intelligence.get("runtime_summary", {}),
        "hyperparameter_findings": _hyperparameter_findings(manifest),
        "video_error_summary": _report_video_error_summary(manifest),
        "visual_review_summary": _visual_review_summary(comparison),
        "artifact_audit_summary": _artifact_audit_summary(comparison),
        "next_sweep_recommendation": _next_sweep_recommendation(comparison),
        "active_error_summary": _active_error_summary(manifest),
        "failure_summary": intelligence.get("failure_summary", {}),
        "recommendations": _recommendations(intelligence, health),
        "artifacts": {
            "comparison_dashboard": str(comparison / "comparison_dashboard.html"),
            "comparison_manifest": str(manifest_path),
            "results_intelligence": str(intelligence_path),
        },
    }
    report_path = out / "dynamics_experiment_report.json"
    markdown_path = out / "dynamics_experiment_report.md"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_dynamics_experiment_report_markdown(report), encoding="utf-8")
    return report


def render_dynamics_experiment_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Grid Dynamics Experiment Report')}",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "## Executive Summary",
        "",
    ]
    run = report.get("run_summary", {})
    lines.extend(
        [
            f"- Completed metric rows in dashboard: `{run.get('completed_metric_rows', 0)}`.",
            f"- Positive test-improvement rows: `{run.get('positive_test_count', 0)}`.",
            f"- Failed configurations represented: `{run.get('failure_count', 0)}`.",
            f"- Active sweep progress: `{run.get('active_progress', 'unknown')}`.",
        ]
    )
    best = report.get("best_models", {})
    if best.get("test_overall"):
        row = best["test_overall"]
        lines.append(f"- Best current test row: `{row.get('experiment_id')}` ({row.get('model_family')}, improve `{_fmt(row.get('improvement_over_persistence_mse'))}`).")
    if best.get("test_learned"):
        row = best["test_learned"]
        lines.append(f"- Best learned test row: `{row.get('experiment_id')}` ({row.get('model_family')}, improve `{_fmt(row.get('improvement_over_persistence_mse'))}`).")
    active_summary = report.get("active_sweep_summary", {})
    if isinstance(active_summary, Mapping) and active_summary.get("available"):
        lines.append(f"- Active spec: `{active_summary.get('experiment_id')}` at `{active_summary.get('progress')}` ({active_summary.get('status')}).")
    lines.extend(["", "## Active Sweep Liveness", ""])
    lines.append(_active_sweep_markdown(active_summary if isinstance(active_summary, Mapping) else {}))
    lines.extend(["", "## Dataset And Timing", "", _dataset_table(report.get("dataset_summary", [])), "", "## Best Test Models", ""])
    lines.append(_model_table(best.get("test_top", [])))
    lines.extend(["", "## Best Test Model Per Family", ""])
    lines.append(_model_table(best.get("test_by_family", [])))
    lines.extend(["", "## Runtime Summary", ""])
    lines.append(_runtime_table(report.get("runtime_summary", {})))
    lines.extend(["", "## Hyperparameter Findings", ""])
    lines.append(_hyperparameter_findings_markdown(report.get("hyperparameter_findings", {})))
    lines.extend(["", "## Per-Video Evidence", ""])
    lines.append(_video_evidence_table(report.get("video_error_summary", {})))
    lines.extend(["", "## Visual Examples", ""])
    lines.append(_visual_review_table(report.get("visual_review_summary", {})))
    lines.extend(["", "## Artifact Integrity Audit", ""])
    lines.append(_artifact_audit_markdown(report.get("artifact_audit_summary", {})))
    lines.extend(["", "## Recommended Next Sweep", ""])
    lines.append(_next_sweep_recommendation_markdown(report.get("next_sweep_recommendation", {})))
    lines.extend(["", "## Persistence And Kinetics Baseline Comparison", ""])
    lines.append(_baseline_table(report.get("baseline_comparison", {})))
    lines.extend(["", "## Active-Cell Error Check", ""])
    lines.append(_active_error_table(report.get("active_error_summary", {})))
    lines.extend(["", "## Failure Analysis", ""])
    failure = report.get("failure_summary", {})
    if failure.get("failure_count"):
        lines.append(f"Failed configurations represented: `{failure.get('failure_count')}`")
        lines.extend(["", _count_table("Failure class", failure.get("by_class", {})), "", _count_table("Model kind", failure.get("by_kind", {}))])
    else:
        lines.append("No failed configurations were represented in the dashboard intelligence artifact.")
    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Artifacts", ""])
    for label, path in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines).rstrip() + "\n"


def _run_summary(manifest: Mapping[str, Any], intelligence: Mapping[str, Any], health: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = next((item for item in health if item.get("experiment_count")), None)
    active_progress = "unknown"
    if active:
        active_progress = f"{active.get('current_index', 0)} / {active.get('experiment_count', 0)}"
    dist = intelligence.get("improvement_distribution", {}).get("test", {})
    return {
        "completed_metric_rows": int(manifest.get("row_count") or intelligence.get("completed_count") or 0),
        "positive_test_count": int(dist.get("positive_count") or 0),
        "failure_count": int(intelligence.get("failure_summary", {}).get("failure_count") or 0),
        "active_progress": active_progress,
        "health_flags": [flag for item in health for flag in item.get("health_flags", [])],
    }


def _active_sweep_summary(health: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for item in health:
        active = item.get("active_status") if isinstance(item.get("active_status"), Mapping) else {}
        if not active:
            continue
        sweep_dir = Path(str(item.get("sweep_dir") or ""))
        live_status_path = sweep_dir / "sweep_live_status.md" if str(sweep_dir) else None
        return {
            "available": True,
            "sweep_dir": item.get("sweep_dir"),
            "profile": item.get("profile"),
            "progress": f"{item.get('current_index', 'unknown')} / {item.get('experiment_count', 'unknown')}",
            "completion_fraction": item.get("completion_fraction"),
            "status_counts": dict(item.get("status_counts", {})) if isinstance(item.get("status_counts"), Mapping) else {},
            "status": active.get("status"),
            "index": active.get("index"),
            "experiment_id": active.get("experiment_id"),
            "dataset_key": active.get("dataset_key"),
            "kind": active.get("kind"),
            "updated_at": active.get("updated_at"),
            "health_flags": [str(flag) for flag in item.get("health_flags", [])],
            "health_report_path": item.get("report_path"),
            "live_status_path": str(live_status_path) if live_status_path and live_status_path.exists() else None,
        }
    return {
        "available": False,
        "limitation": "No active sweep status was found in the included sweep health reports.",
    }


def _dataset_summary(datasets: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for key, dataset in sorted(datasets.items()):
        windowing = dataset.get("windowing", {}) if isinstance(dataset, Mapping) else {}
        records.append(
            {
                "dataset_key": str(key),
                "window_frames": windowing.get("window_frames") or dataset.get("window_frames") if isinstance(dataset, Mapping) else None,
                "prediction_horizon_frames": windowing.get("prediction_horizon_frames"),
                "prediction_horizon_sec": windowing.get("prediction_horizon_sec"),
                "frame_rate_hz": windowing.get("effective_frame_rate_hz") or windowing.get("source_frame_rate_hz"),
                "train_videos": len(_split_ids(dataset, "train")),
                "val_videos": len(_split_ids(dataset, "val")),
                "test_videos": len(_split_ids(dataset, "test")),
            }
        )
    return records


def _best_models(intelligence: Mapping[str, Any]) -> dict[str, Any]:
    leaders = list(intelligence.get("leaderboards", {}).get("test", []))
    by_family = intelligence.get("best_by_family", {}).get("test", {})
    learned = _best_learned_model(by_family)
    return {
        "test_top": leaders[:10],
        "test_overall": leaders[0] if leaders else None,
        "test_learned": learned,
        "test_by_family": [dict(row, group_name=name) for name, row in sorted(by_family.items())],
    }


def _hyperparameter_findings(manifest: Mapping[str, Any], *, split: str = "test", max_groups_per_dimension: int = 8) -> dict[str, Any]:
    rows = [row for row in manifest.get("rows", []) if isinstance(row, Mapping)]
    completed = [row for row in rows if _num(row.get(f"{split}_improvement_over_persistence_mse")) is not None]
    dimensions = [
        ("model_family", "Model family"),
        ("hyperparameter_group", "Hyperparameter group"),
        ("loss_mode", "Loss mode"),
        ("prediction_target", "Prediction target / baseline"),
        ("learning_rate", "Learning rate"),
        ("hidden_channels", "Hidden channels"),
        ("hidden_dim", "Hidden dim"),
        ("model_dim", "Model dim"),
        ("num_layers", "Layers"),
        ("residual_scale", "Residual scale"),
    ]
    summaries: dict[str, dict[str, Any]] = {}
    for key, label in dimensions:
        buckets: dict[str, list[Mapping[str, Any]]] = {}
        for row in completed:
            value = _hyperparameter_value(row, key)
            if value is None or value == "":
                continue
            buckets.setdefault(str(value), []).append(row)
        groups = []
        for value, bucket_rows in buckets.items():
            values = [float(_num(row.get(f"{split}_improvement_over_persistence_mse"))) for row in bucket_rows if _num(row.get(f"{split}_improvement_over_persistence_mse")) is not None]
            if not values:
                continue
            best = max(bucket_rows, key=lambda row: _num(row.get(f"{split}_improvement_over_persistence_mse")) or float("-inf"))
            groups.append(
                {
                    "value": value,
                    "count": len(values),
                    "positive_count": sum(1 for value in values if value > 0),
                    "mean_improvement": _mean(values),
                    "median_improvement": _median(values),
                    "best_experiment_id": best.get("experiment_id"),
                    "best_model_family": best.get("model_family"),
                    "best_improvement": _num(best.get(f"{split}_improvement_over_persistence_mse")),
                }
            )
        groups.sort(key=lambda group: (_num(group.get("mean_improvement")) if _num(group.get("mean_improvement")) is not None else float("-inf"), group.get("count", 0)), reverse=True)
        if groups:
            summaries[key] = {"label": label, "groups": groups[: int(max_groups_per_dimension)]}
    return {
        "available": bool(summaries),
        "split": split,
        "row_count": len(completed),
        "dimensions": summaries,
    }


def _hyperparameter_value(row: Mapping[str, Any], key: str) -> Any:
    params = row.get("params", {}) if isinstance(row.get("params"), Mapping) else {}
    if key == "prediction_target":
        return row.get("prediction_target") or row.get("baseline_name") or params.get("prediction_target") or params.get("baseline_name")
    if key in row and row.get(key) not in (None, ""):
        return row.get(key)
    return params.get(key)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _report_video_error_summary(manifest: Mapping[str, Any], *, split: str = "test", max_models: int = 10) -> dict[str, Any]:
    rows = [row for row in manifest.get("rows", []) if isinstance(row, Mapping)]
    rows_with_summary = []
    for row in rows:
        summary = row.get("video_error_summary", {})
        if not isinstance(summary, Mapping) or not isinstance(summary.get(split), Mapping):
            continue
        split_summary = summary[split]
        rows_with_summary.append(
            {
                "experiment_id": row.get("experiment_id"),
                "model_family": row.get("model_family"),
                "dataset_key": row.get("dataset_key"),
                "prediction_target": row.get("prediction_target") or row.get("baseline_name"),
                "split": split,
                "improvement_over_persistence_mse": row.get(f"{split}_improvement_over_persistence_mse"),
                "video_count": split_summary.get("video_count", 0),
                "best_videos": list(split_summary.get("best_videos", []))[:5] if isinstance(split_summary.get("best_videos"), list) else [],
                "worst_videos": list(split_summary.get("worst_videos", []))[:5] if isinstance(split_summary.get("worst_videos"), list) else [],
                "label_summary": list(split_summary.get("label_summary", []))[:5] if isinstance(split_summary.get("label_summary"), list) else [],
            }
        )
    rows_with_summary.sort(key=lambda row: (_num(row.get("improvement_over_persistence_mse")) if _num(row.get("improvement_over_persistence_mse")) is not None else float("-inf")), reverse=True)
    return {
        "available": bool(rows_with_summary),
        "split": split,
        "row_count": len(rows_with_summary),
        "rows": rows_with_summary[: int(max_models)],
        "limitation": None if rows_with_summary else "No completed metric rows include per-video prediction diagnostics yet. Regenerate or continue runs through metric writers that attach split_metrics.<split>.per_video.",
    }


def _visual_review_summary(comparison_dir: Path) -> dict[str, Any]:
    roots = []
    if comparison_dir.parent.exists():
        roots.append(comparison_dir.parent / "reviews")
    if comparison_dir.exists():
        roots.append(comparison_dir / "reviews")
    paths = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/video_error_review.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    reviews = []
    for path in paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        models = [model for model in payload.get("models", []) if isinstance(model, Mapping)]
        missing = [row for row in payload.get("missing_visual_example_rows", []) if isinstance(row, Mapping)]
        reviews.append(
            {
                "title": payload.get("title") or path.parent.name,
                "selection_mode": payload.get("selection_mode"),
                "split": payload.get("split"),
                "dataset_key": payload.get("dataset_key"),
                "selected_model_count": int(payload.get("selected_model_count") or len(models)),
                "temporal_clip_model_count": int(payload.get("temporal_clip_model_count") or 0),
                "missing_visual_example_count": int(payload.get("missing_visual_example_count") or len(missing)),
                "html_path": payload.get("html_path") or str(path.with_suffix(".html")),
                "summary_path": str(path),
                "model_ids": [str(model.get("experiment_id")) for model in models[:5] if model.get("experiment_id")],
                "missing_model_ids": [str(row.get("experiment_id")) for row in missing[:5] if row.get("experiment_id")],
                "limitations": [str(item) for item in payload.get("limitations", [])[:5]],
            }
        )
    reviews.sort(key=lambda item: (-(item.get("temporal_clip_model_count") or 0), -(item.get("selected_model_count") or 0), str(item.get("title") or "")))
    return {
        "available": bool(reviews),
        "review_count": len(reviews),
        "reviews": reviews,
        "limitation": None if reviews else "No video error review artifacts were found near the comparison directory. Run review-video-errors to generate visual examples.",
    }


def _artifact_audit_summary(comparison_dir: Path) -> dict[str, Any]:
    roots = []
    if comparison_dir.parent.exists():
        roots.append(comparison_dir.parent / "plans")
    if comparison_dir.exists():
        roots.append(comparison_dir / "plans")
    audit = _load_latest_plan(_dedupe_paths(roots), "*/grid128_artifact_audit.json")
    if not audit:
        return {
            "available": False,
            "limitation": "No grid128 artifact audit was found near the comparison directory. Run audit-grid128-artifacts after refreshing reports and reviews.",
        }
    artifacts = [row for row in audit.get("artifacts", []) if isinstance(row, Mapping)]
    review_reference_counts = []
    consistency_checks = []
    for row in artifacts:
        summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
        referenced = summary.get("referenced_file_count")
        missing = summary.get("missing_referenced_file_count")
        if referenced is not None and missing is not None:
            review_reference_counts.append(
                {
                    "label": row.get("label"),
                    "status": row.get("status"),
                    "referenced_file_count": int(referenced or 0),
                    "missing_referenced_file_count": int(missing or 0),
                    "referenced_metric_file_count": int(summary.get("referenced_metric_file_count") or 0),
                    "referenced_prediction_file_count": int(summary.get("referenced_prediction_file_count") or 0),
                    "referenced_input_file_count": int(summary.get("preflight_input_reference_count") or 0),
                    "path": row.get("relative_path") or row.get("path"),
                }
            )
        consistency_checks.extend(_artifact_consistency_checks(row, summary))
        source_progress_check = _artifact_stage_b_source_progress_check(row, summary)
        if source_progress_check:
            consistency_checks.append(source_progress_check)
    issues = [row for row in artifacts if row.get("status") != "ok"]
    return {
        "available": True,
        "ok": bool(audit.get("ok")),
        "created_at": audit.get("created_at"),
        "artifact_count": int(audit.get("artifact_count") or len(artifacts)),
        "status_counts": dict(audit.get("status_counts", {})) if isinstance(audit.get("status_counts"), Mapping) else {},
        "json_path": audit.get("json_path") or audit.get("_artifact_path"),
        "markdown_path": audit.get("markdown_path"),
        "review_reference_counts": review_reference_counts,
        "consistency_checks": consistency_checks,
        "issue_count": len(issues),
        "issues": [
            {
                "label": row.get("label"),
                "status": row.get("status"),
                "path": row.get("relative_path") or row.get("path"),
                "error": row.get("error"),
            }
            for row in issues[:10]
        ],
    }


def _artifact_stage_b_source_progress_check(row: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any] | None:
    if not summary.get("stage_b_source_progress_check"):
        return None
    return {
        "label": row.get("label"),
        "path": row.get("relative_path") or row.get("path"),
        "status": row.get("status"),
        "check": "stage_b_source_progress",
        "ok": summary.get("stage_b_source_progress_matches_sweep"),
        "detail": (
            f"plan={summary.get('stage_b_plan_progress_index')} "
            f"source={summary.get('stage_b_source_progress_index')} "
            f"records={summary.get('stage_b_source_progress_records')}"
        ),
    }


def _artifact_consistency_checks(row: Mapping[str, Any], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    label = row.get("label")
    path = row.get("relative_path") or row.get("path")
    checks: list[dict[str, Any]] = []
    if summary.get("sweep_status_check"):
        checks.append(
            {
                "label": label,
                "path": path,
                "status": row.get("status"),
                "check": "sweep_status_markdown",
                "ok": summary.get("sweep_status_matches_sweep"),
                "detail": f"report={summary.get('report_progress')} expected={summary.get('expected_progress')}",
            }
        )
    if summary.get("active_summary_check"):
        checks.append(
            {
                "label": label,
                "path": path,
                "status": row.get("status"),
                "check": "active_sweep_summary",
                "ok": summary.get("active_summary_matches_sweep"),
                "detail": f"report={summary.get('report_active_progress')} expected={summary.get('expected_active_progress')}",
            }
        )
    if summary.get("embedded_audit_check"):
        checks.append(
            {
                "label": label,
                "path": path,
                "status": row.get("status"),
                "check": "embedded_artifact_audit_summary",
                "ok": summary.get("embedded_audit_summary_matches_current_state"),
                "detail": (
                    f"progress={summary.get('embedded_audit_expected_progress')} "
                    f"references={summary.get('embedded_audit_comparison_reference_count')}/"
                    f"{summary.get('expected_comparison_reference_count')}"
                ),
            }
        )
    if summary.get("stage_b_manifest_check"):
        checks.append(
            {
                "label": label,
                "path": path,
                "status": row.get("status"),
                "check": "stage_b_plan_manifest",
                "ok": summary.get("stage_b_manifest_matches_plan"),
                "detail": (
                    f"plan={summary.get('stage_b_plan_count')} "
                    f"manifest={summary.get('stage_b_manifest_count')} "
                    f"experiments={summary.get('stage_b_manifest_experiment_count')}"
                ),
            }
        )
    if summary.get("stage_b_dry_run_check"):
        checks.append(
            {
                "label": label,
                "path": path,
                "status": row.get("status"),
                "check": "stage_b_dry_run_manifest",
                "ok": summary.get("stage_b_dry_run_matches_manifest"),
                "detail": (
                    f"source={summary.get('stage_b_source_manifest_experiment_count')} "
                    f"dry_run={summary.get('stage_b_dry_run_experiment_count')}"
                ),
            }
        )
    return checks


def _next_sweep_recommendation(comparison_dir: Path) -> dict[str, Any]:
    roots = []
    if comparison_dir.parent.exists():
        roots.append(comparison_dir.parent / "plans")
    if comparison_dir.exists():
        roots.append(comparison_dir / "plans")
    roots = _dedupe_paths(roots)
    stage_b = _load_latest_plan(roots, "*/next_sweep_plan.json")
    active_cell = _load_latest_plan(roots, "*/active_cell_rescue_plan.json")
    summary = {
        "available": bool(stage_b or active_cell),
        "stage_b_plan": _stage_b_recommendation(stage_b) if stage_b else None,
        "active_cell_rescue": _active_cell_rescue_recommendation(active_cell) if active_cell else None,
        "limitation": None,
    }
    if not summary["available"]:
        summary["limitation"] = "No next-sweep or active-cell rescue plan artifacts were found near the comparison directory."
    return summary


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    result = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _load_latest_plan(roots: Sequence[Path], pattern: str) -> dict[str, Any] | None:
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob(pattern):
            if path.is_file():
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    for path in candidates:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        payload.setdefault("_artifact_path", str(path))
        return payload
    return None


def _stage_b_recommendation(plan: Mapping[str, Any]) -> dict[str, Any]:
    progress = plan.get("progress_summary", {}) if isinstance(plan.get("progress_summary"), Mapping) else {}
    return {
        "created_at": plan.get("created_at"),
        "planned_experiment_count": int(plan.get("planned_experiment_count") or 0),
        "selection_counts": dict(plan.get("selection_counts", {})) if isinstance(plan.get("selection_counts"), Mapping) else {},
        "dataset_counts": dict(plan.get("dataset_counts", {})) if isinstance(plan.get("dataset_counts"), Mapping) else {},
        "target_counts": dict(plan.get("target_counts", {})) if isinstance(plan.get("target_counts"), Mapping) else {},
        "progress": f"{progress.get('current_index', 'unknown')} / {progress.get('experiment_count', 'unknown')}",
        "last_completed_or_seen": progress.get("last_experiment_id"),
        "suggested_command": plan.get("suggested_command"),
        "summary_path": plan.get("summary_path") or plan.get("_artifact_path"),
        "markdown_path": plan.get("markdown_path"),
        "manifest_path": plan.get("manifest_path"),
        "source_sweep_dir": plan.get("source_sweep_dir"),
    }


def _active_cell_rescue_recommendation(plan: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [item for item in plan.get("recommended_candidates", []) if isinstance(item, Mapping)]
    next_candidate = candidates[0] if candidates else {}
    return {
        "title": plan.get("title"),
        "created_at": plan.get("created_at"),
        "best_overall_label": plan.get("best_overall_label"),
        "best_overall_improvement_over_persistence_mse": plan.get("best_overall_improvement_over_persistence_mse"),
        "active_cell_warning_count": int(plan.get("active_cell_warning_count") or 0),
        "completed_shared_neural_count": int(plan.get("completed_shared_neural_count") or 0),
        "pending_shared_neural_count": int(plan.get("pending_shared_neural_count") or 0),
        "next_candidate": {
            "config_id": next_candidate.get("config_id"),
            "model_family": next_candidate.get("model_family"),
            "priority": next_candidate.get("priority"),
            "rationale": next_candidate.get("rationale"),
            "command": next_candidate.get("command"),
            "out_dir": next_candidate.get("out_dir"),
        }
        if next_candidate
        else None,
        "recommendations": [str(item) for item in plan.get("recommendations", [])[:5]],
        "plan_path": plan.get("plan_path") or plan.get("_artifact_path"),
        "markdown_path": plan.get("markdown_path"),
        "grid_status": plan.get("grid_status"),
    }

def _baseline_comparison(intelligence: Mapping[str, Any]) -> dict[str, Any]:
    by_family = intelligence.get("best_by_family", {}).get("test", {})
    best_array = by_family.get("array_baseline")
    best_kinetics = by_family.get("kinetics_baseline")
    best_learned = _best_learned_model(by_family)
    return {
        "best_array_baseline": best_array,
        "best_kinetics_baseline": best_kinetics,
        "best_learned_model": best_learned,
        "learned_minus_kinetics_improvement": _diff_improve(best_learned, best_kinetics),
        "learned_minus_array_improvement": _diff_improve(best_learned, best_array),
    }


def _best_learned_model(by_family: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for family, row in by_family.items() if family not in {"array_baseline", "kinetics_baseline"} and row]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _num(row.get("improvement_over_persistence_mse")) if _num(row.get("improvement_over_persistence_mse")) is not None else float("-inf"))



def _active_error_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in manifest.get("rows", []) if row.get("test_active_cell_improvement_over_persistence_mse") is not None]
    if not rows:
        return {"available": False, "row_count": 0, "best_by_family": {}, "best_overall": None, "best_learned": None, "learned_minus_kinetics_active_improvement": None}
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row.get("model_family") or row.get("kind") or "unknown")
        current = by_family.get(family)
        if current is None or _num(row.get("test_active_cell_improvement_over_persistence_mse")) > _num(current.get("test_active_cell_improvement_over_persistence_mse")):
            by_family[family] = row
    sorted_active = sorted(rows, key=lambda row: _num(row.get("test_active_cell_improvement_over_persistence_mse")) or float("-inf"), reverse=True)
    learned_rows = [row for row in rows if str(row.get("model_family") or row.get("kind") or "") not in {"array_baseline", "kinetics_baseline"}]
    best_overall = sorted_active[0]
    best_learned = _best_learned_model(by_family)
    best_kinetics = by_family.get("kinetics_baseline")
    best_global_learned = max(learned_rows, key=lambda row: _num(row.get("test_improvement_over_persistence_mse") or row.get("improvement_over_persistence_mse")) or float("-inf")) if learned_rows else None
    active_global_tradeoff = _active_global_tradeoff(best_active=best_overall, best_global=best_global_learned)
    return {
        "available": True,
        "row_count": len(rows),
        "best_by_family": by_family,
        "best_overall": best_overall,
        "best_learned": best_learned,
        "best_kinetics": best_kinetics,
        "best_global_learned": best_global_learned,
        "active_global_tradeoff": active_global_tradeoff,
        "top_active_rows": sorted_active[:8],
        "learned_minus_kinetics_active_improvement": _diff_metric(best_learned, best_kinetics, "test_active_cell_improvement_over_persistence_mse"),
    }

def _active_global_tradeoff(*, best_active: Mapping[str, Any] | None, best_global: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not best_active or not best_global:
        return None
    active_global = _num(best_active.get("test_improvement_over_persistence_mse") or best_active.get("improvement_over_persistence_mse"))
    active_active = _num(best_active.get("test_active_cell_improvement_over_persistence_mse"))
    global_global = _num(best_global.get("test_improvement_over_persistence_mse") or best_global.get("improvement_over_persistence_mse"))
    global_active = _num(best_global.get("test_active_cell_improvement_over_persistence_mse"))
    return {
        "best_active_experiment_id": best_active.get("experiment_id"),
        "best_global_experiment_id": best_global.get("experiment_id"),
        "same_experiment": best_active.get("experiment_id") == best_global.get("experiment_id"),
        "best_active_global_improvement": active_global,
        "best_active_active_cell_improvement": active_active,
        "best_global_global_improvement": global_global,
        "best_global_active_cell_improvement": global_active,
        "active_cell_gain_of_best_active_over_best_global": _diff_numbers(active_active, global_active),
        "global_improvement_cost_of_best_active_vs_best_global": _diff_numbers(active_global, global_global),
    }


def _diff_numbers(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _recommendations(intelligence: Mapping[str, Any], health: Sequence[Mapping[str, Any]]) -> list[str]:
    recs: list[str] = []
    comparison = _baseline_comparison(intelligence)
    learned = comparison.get("best_learned_model")
    kinetics = comparison.get("best_kinetics_baseline")
    if learned and kinetics:
        delta = comparison.get("learned_minus_kinetics_improvement")
        if delta is not None and delta < 0:
            recs.append("Do not claim learned dynamics beat kinetics-aware baselines yet; current best kinetics baseline has higher test improvement.")
        elif delta is not None:
            recs.append("Current best learned model beats the best kinetics baseline on test improvement; verify this survives more completed runs and visual review.")
    if not learned:
        recs.append("No learned model has enough completed test metrics for a learned-vs-baseline claim yet.")
    failure_count = int(intelligence.get("failure_summary", {}).get("failure_count") or 0)
    if failure_count:
        recs.append("Keep archived failures in the report; they explain why batch-size reductions were necessary.")
    health_recs = [rec for item in health for rec in item.get("recommendations", [])]
    recs.extend(item for item in health_recs if item not in recs)
    if not recs:
        recs.append("Continue collecting results and regenerate this report after the sweep advances.")
    return recs


def _split_ids(dataset: Mapping[str, Any], split: str) -> list[Any]:
    splits = dataset.get("splits", {}) if isinstance(dataset, Mapping) else {}
    if not isinstance(splits, Mapping):
        return []
    value = splits.get(f"{split}_video_ids") or splits.get(split) or []
    if isinstance(value, Mapping):
        value = value.get("video_ids") or value.get("videos") or []
    return list(value) if isinstance(value, (list, tuple, set)) else []



def _active_sweep_markdown(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return str(summary.get("limitation") or "No active sweep status found.")
    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| status | `{summary.get('status')}` |",
        f"| progress | `{summary.get('progress')}` |",
        f"| active index | `{summary.get('index')}` |",
        f"| experiment | `{summary.get('experiment_id')}` |",
        f"| dataset | `{summary.get('dataset_key') or ''}` |",
        f"| kind | `{summary.get('kind') or ''}` |",
        f"| updated at | `{summary.get('updated_at') or ''}` |",
        f"| status counts | `{summary.get('status_counts')}` |",
    ]
    if summary.get("health_report_path"):
        lines.append(f"| health report | `{summary.get('health_report_path')}` |")
    if summary.get("live_status_path"):
        lines.append(f"| live status | `{summary.get('live_status_path')}` |")
    flags = summary.get("health_flags") or []
    if flags:
        lines.extend(["", "Health flags:"])
        lines.extend(f"- {flag}" for flag in flags)
    return "\n".join(lines)


def _active_error_table(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return "Active-cell structured metrics are not available for the currently included metric rows. Regenerate metrics with the structured error-analysis code path to populate this section."
    rows = []
    for label, row in (("Best overall", summary.get("best_overall")), ("Best learned", summary.get("best_learned")), ("Best kinetics", summary.get("best_kinetics"))):
        if row:
            rows.append((label, row))
    lines = ["| Role | Experiment | Family | Active-cell improve | Top-activity improve | High-change improve |", "|---|---|---|---:|---:|---:|"]
    for label, row in rows:
        lines.append(
            f"| {label} | `{row.get('experiment_id')}` | {row.get('model_family')} | {_fmt(row.get('test_active_cell_improvement_over_persistence_mse'))} | {_fmt(row.get('test_top_activity_improvement_over_persistence_mse'))} | {_fmt(row.get('test_high_change_improvement_over_persistence_mse'))} |"
        )
    top_rows = [row for row in summary.get("top_active_rows", []) if isinstance(row, Mapping)]
    tradeoff = summary.get("active_global_tradeoff") if isinstance(summary.get("active_global_tradeoff"), Mapping) else None
    if tradeoff and not tradeoff.get("same_experiment"):
        lines.extend(
            [
                "",
                f"Best active-cell row versus best global learned row: active-cell gain `{_fmt(tradeoff.get('active_cell_gain_of_best_active_over_best_global'))}`, global-improvement cost `{_fmt(tradeoff.get('global_improvement_cost_of_best_active_vs_best_global'))}`.",
            ]
        )
    if top_rows:
        lines.extend(["", "Top active-cell rows:", "", "| Experiment | Family | Dataset | Active-cell improve | Global improve | Top-activity improve | High-change improve |", "|---|---|---|---:|---:|---:|---:|"])
        for row in top_rows[:5]:
            global_improve = row.get("test_improvement_over_persistence_mse")
            if global_improve is None:
                global_improve = row.get("improvement_over_persistence_mse")
            lines.append(
                f"| `{row.get('experiment_id')}` | {row.get('model_family')} | {row.get('dataset_key')} | {_fmt(row.get('test_active_cell_improvement_over_persistence_mse'))} | {_fmt(global_improve)} | {_fmt(row.get('test_top_activity_improvement_over_persistence_mse'))} | {_fmt(row.get('test_high_change_improvement_over_persistence_mse'))} |"
            )
    lines.append("")
    lines.append(f"Rows with active-cell diagnostics: `{summary.get('row_count', 0)}`")
    lines.append(f"Learned minus kinetics active-cell improvement: `{_fmt(summary.get('learned_minus_kinetics_active_improvement'))}`")
    return "\n".join(lines)

def _diff_improve(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> float | None:
    return _diff_metric(a, b, "improvement_over_persistence_mse")


def _diff_metric(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, key: str) -> float | None:
    if not a or not b:
        return None
    av = _num(a.get(key))
    bv = _num(b.get(key))
    if av is None or bv is None:
        return None
    return float(av - bv)


def _dataset_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No dataset metadata found."
    lines = ["| Dataset | Window | Horizon frames | Horizon sec | Frame rate | Train/Val/Test videos |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset_key", "")),
                    str(row.get("window_frames") or ""),
                    str(row.get("prediction_horizon_frames") or ""),
                    _fmt(row.get("prediction_horizon_sec")),
                    _fmt(row.get("frame_rate_hz")),
                    f"{row.get('train_videos', 0)}/{row.get('val_videos', 0)}/{row.get('test_videos', 0)}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _model_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No model rows found."
    lines = ["| Experiment | Family | Dataset | Target/Baseline | Test improve | HParams |", "|---|---|---|---|---:|---|"]
    for row in rows:
        label = row.get("prediction_target") or row.get("baseline_name") or ""
        group = f"{row.get('group_name')} / " if row.get("group_name") else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{group}{row.get('experiment_id', '')}`",
                    str(row.get("model_family", "")),
                    str(row.get("dataset_key", "")),
                    str(label),
                    _fmt(row.get("improvement_over_persistence_mse")),
                    f"`{row.get('hyperparameter_summary') or ''}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _runtime_table(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return "No completed runtime records found."
    lines = [
        f"Timed completed rows: `{summary.get('row_count', 0)}`",
        f"Median runtime: `{_fmt_duration(summary.get('median_seconds'))}`",
        f"Total timed runtime: `{_fmt_duration(summary.get('total_seconds'))}`",
        "",
        "| Family | Timed rows | Median runtime | Max runtime |",
        "|---|---:|---:|---:|",
    ]
    by_family = summary.get("by_family", {}) if isinstance(summary.get("by_family"), Mapping) else {}
    for family, item in sorted(by_family.items(), key=lambda pair: _num(pair[1].get("median_seconds")) or 0, reverse=True):
        if isinstance(item, Mapping):
            lines.append(f"| {family} | {item.get('count', 0)} | {_fmt_duration(item.get('median_seconds'))} | {_fmt_duration(item.get('max_seconds'))} |")
    slowest = [row for row in summary.get("slowest_rows", []) if isinstance(row, Mapping)]
    if slowest:
        lines.extend(["", "Slowest completed rows:", ""])
        lines.extend(f"- `{row.get('experiment_id')}` ({row.get('model_family')}, {_fmt_duration(row.get('elapsed_seconds'))})" for row in slowest[:5])
    return "\n".join(lines)


def _fmt_duration(seconds: Any) -> str:
    value = _num(seconds)
    if value is None:
        return "n/a"
    if value < 90:
        return f"{value:.0f}s"
    if value < 7200:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.2f}h"


def _hyperparameter_findings_markdown(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return "No hyperparameter findings are available for the currently included metric rows."
    lines = [f"Completed rows with `{summary.get('split', 'test')}` improvement metrics: `{summary.get('row_count', 0)}`"]
    dimensions = summary.get("dimensions", {}) if isinstance(summary.get("dimensions"), Mapping) else {}
    preferred = ["model_family", "hyperparameter_group", "prediction_target", "loss_mode", "learning_rate", "hidden_channels", "hidden_dim", "model_dim", "num_layers", "residual_scale"]
    for key in preferred:
        item = dimensions.get(key)
        if not isinstance(item, Mapping):
            continue
        groups = [group for group in item.get("groups", []) if isinstance(group, Mapping)]
        if not groups:
            continue
        lines.extend(["", f"### {item.get('label') or key}", "", "| Value | Rows | Positive | Mean improve | Best row | Best improve |", "|---|---:|---:|---:|---|---:|"])
        for group in groups[:5]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(group.get("value", "")),
                        str(group.get("count", 0)),
                        str(group.get("positive_count", 0)),
                        _fmt(group.get("mean_improvement")),
                        f"`{group.get('best_experiment_id', '')}`",
                        _fmt(group.get("best_improvement")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _video_evidence_table(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return str(summary.get("limitation") or "No per-video prediction evidence is available for the current report.")
    lines = [
        "| Experiment | Family | Dataset | Videos | Labels | Best videos | Worst videos |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in summary.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('experiment_id', '')}`",
                    str(row.get("model_family", "")),
                    str(row.get("dataset_key", "")),
                    str(row.get("video_count", 0)),
                    _label_summary_list(row.get("label_summary", [])),
                    _video_list(row.get("best_videos", [])),
                    _video_list(row.get("worst_videos", [])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _label_summary_list(rows: Any) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return "n/a"
    parts = []
    for row in list(rows)[:5]:
        if not isinstance(row, Mapping):
            continue
        parts.append(f"{row.get('label', 'unknown')} {_fmt(row.get('improvement_over_persistence_mse'))}")
    return ", ".join(parts) if parts else "n/a"


def _video_list(rows: Any) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return "n/a"
    parts = []
    for row in list(rows)[:5]:
        if not isinstance(row, Mapping):
            continue
        parts.append(f"{row.get('video_id')} ({_fmt(row.get('improvement_over_persistence_mse'))})")
    return ", ".join(parts) if parts else "n/a"


def _visual_review_table(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return str(summary.get("limitation") or "No visual review artifacts found.")
    lines = [
        "| Review | Selection | Split | Models | Clips | Missing visuals | Top models | HTML |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for review in summary.get("reviews", []):
        if not isinstance(review, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(review.get("title", "")),
                    str(review.get("selection_mode") or ""),
                    str(review.get("split") or ""),
                    str(review.get("selected_model_count", 0)),
                    str(review.get("temporal_clip_model_count", 0)),
                    str(review.get("missing_visual_example_count", 0)),
                    _compact_model_ids(review.get("model_ids", [])),
                    f"`{review.get('html_path', '')}`",
                ]
            )
            + " |"
        )
    notes = []
    for review in summary.get("reviews", []):
        if not isinstance(review, Mapping):
            continue
        missing = review.get("missing_model_ids") or []
        if missing:
            notes.append(f"- `{review.get('title')}` omitted top-ranked rows without visual artifacts: {', '.join(str(item) for item in missing[:5])}.")
    if notes:
        lines.extend(["", *notes])
    return "\n".join(lines)


def _artifact_audit_markdown(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return str(summary.get("limitation") or "No artifact audit found.")
    lines = [
        "Embedded audit snapshot captured when this report was generated; the standalone audit may have a later timestamp after final validation.",
        f"Audit OK: `{summary.get('ok')}`",
        f"Snapshot generated: `{summary.get('created_at')}`",
        f"Artifacts checked: `{summary.get('artifact_count')}`",
        f"Status counts: `{summary.get('status_counts')}`",
    ]
    if summary.get("markdown_path"):
        lines.append(f"Audit report: `{summary.get('markdown_path')}`")
    consistency = [row for row in summary.get("consistency_checks", []) if isinstance(row, Mapping)]
    if consistency:
        lines.extend(["", "Consistency checks:", "", "| Artifact | Check | OK | Detail |", "|---|---|---|---|"])
        for row in consistency:
            lines.append(f"| {row.get('label')} | {row.get('check')} | {row.get('ok')} | {row.get('detail')} |")
    references = [row for row in summary.get("review_reference_counts", []) if isinstance(row, Mapping)]
    if references:
        lines.extend(["", "Referenced artifact files:", "", "| Artifact | Status | Referenced | Metrics | Predictions | Inputs | Missing |", "|---|---|---:|---:|---:|---:|---:|"])
        for row in references:
            lines.append(
                f"| {row.get('label')} | {row.get('status')} | {row.get('referenced_file_count', 0)} | "
                f"{row.get('referenced_metric_file_count', 0)} | {row.get('referenced_prediction_file_count', 0)} | "
                f"{row.get('referenced_input_file_count', 0)} | {row.get('missing_referenced_file_count', 0)} |"
            )
    issues = [row for row in summary.get("issues", []) if isinstance(row, Mapping)]
    if issues:
        lines.extend(["", "Audit issues:"])
        for row in issues:
            lines.append(f"- `{row.get('path')}`: {row.get('status')} {row.get('error') or ''}".rstrip())
    return "\n".join(lines)


def _next_sweep_recommendation_markdown(summary: Mapping[str, Any]) -> str:
    if not summary.get("available"):
        return str(summary.get("limitation") or "No next-sweep recommendation artifacts found.")
    lines = []
    stage_b = summary.get("stage_b_plan") if isinstance(summary.get("stage_b_plan"), Mapping) else None
    if stage_b:
        lines.extend(
            [
                "### Stage B Sweep Plan",
                "",
                f"- Planned experiments: `{stage_b.get('planned_experiment_count', 0)}`.",
                f"- Source progress when planned: `{stage_b.get('progress', 'unknown')}`.",
                f"- Selection counts: `{_compact_counts(stage_b.get('selection_counts', {}))}`.",
                f"- Plan markdown: `{stage_b.get('markdown_path') or 'n/a'}`.",
                f"- Plan manifest: `{stage_b.get('manifest_path') or 'n/a'}`.",
            ]
        )
        if stage_b.get("suggested_command"):
            lines.extend(["", "Suggested Stage B command:", "", "```bash", str(stage_b.get("suggested_command")), "```"])
    active_cell = summary.get("active_cell_rescue") if isinstance(summary.get("active_cell_rescue"), Mapping) else None
    if active_cell:
        if lines:
            lines.append("")
        lines.extend(
            [
                "### Active-Cell Rescue Candidate",
                "",
                f"- Current global shared-horizon control: `{active_cell.get('best_overall_label') or 'n/a'}` ({_fmt(active_cell.get('best_overall_improvement_over_persistence_mse'))}).",
                f"- Active-cell warnings: `{active_cell.get('active_cell_warning_count', 0)}`.",
                f"- Shared-neural grid completed/pending: `{active_cell.get('completed_shared_neural_count', 0)} / {active_cell.get('pending_shared_neural_count', 0)}`.",
                f"- Plan markdown: `{active_cell.get('markdown_path') or 'n/a'}`.",
            ]
        )
        candidate = active_cell.get("next_candidate") if isinstance(active_cell.get("next_candidate"), Mapping) else None
        if candidate:
            lines.extend(
                [
                    f"- Next candidate: `{candidate.get('config_id')}` ({candidate.get('model_family')}, priority `{candidate.get('priority')}`).",
                    f"- Rationale: {candidate.get('rationale') or 'n/a'}",
                ]
            )
            if candidate.get("command"):
                lines.extend(["", "Suggested active-cell rescue command:", "", "```bash", str(candidate.get("command")), "```"])
    return "\n".join(lines) if lines else "No next-sweep recommendation details found."


def _compact_counts(counts: Any) -> str:
    if not isinstance(counts, Mapping) or not counts:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))

def _compact_model_ids(values: Any) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return "n/a"
    parts = [f"`{item}`" for item in list(values)[:3] if item]
    if not parts:
        return "n/a"
    if len(values) > 3:
        parts.append("...")
    return ", ".join(str(item) for item in parts)


def _baseline_table(comparison: Mapping[str, Any]) -> str:
    rows = []
    for label, key in (("Best array baseline", "best_array_baseline"), ("Best kinetics baseline", "best_kinetics_baseline"), ("Best learned model", "best_learned_model")):
        row = comparison.get(key)
        if row:
            rows.append({"label": label, **row})
    if not rows:
        return "No baseline comparison rows found."
    lines = ["| Role | Experiment | Family | Test improve |", "|---|---|---|---:|"]
    for row in rows:
        lines.append(f"| {row.get('label')} | `{row.get('experiment_id')}` | {row.get('model_family')} | {_fmt(row.get('improvement_over_persistence_mse'))} |")
    lines.append("")
    lines.append(f"Learned minus kinetics improvement: `{_fmt(comparison.get('learned_minus_kinetics_improvement'))}`")
    lines.append(f"Learned minus array-baseline improvement: `{_fmt(comparison.get('learned_minus_array_improvement'))}`")
    return "\n".join(lines)


def _count_table(label: str, counts: Mapping[str, Any]) -> str:
    if not counts:
        return "No counts."
    lines = [f"| {label} | Count |", "|---|---:|"]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.4g}"


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
