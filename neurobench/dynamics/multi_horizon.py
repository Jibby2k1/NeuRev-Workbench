"""Multi-horizon comparison and planning utilities for grid dynamics sweeps."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence


LEARNED_FAMILIES = {"latent_gru", "latent_transformer", "linear_latent", "convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}


def build_multi_horizon_report(
    *,
    comparison_dir: str | Path,
    out_dir: str | Path,
    split: str = "test",
    max_candidates: int = 20,
    title: str = "Multi-Horizon Forecasting Report",
) -> dict[str, Any]:
    """Compare matching single-horizon runs and plan shared-horizon candidates."""
    comparison = Path(comparison_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(comparison / "comparison_manifest.json")
    rows = [dict(row) for row in manifest.get("rows", []) if isinstance(row, Mapping)]
    datasets = manifest.get("datasets", {}) if isinstance(manifest.get("datasets"), Mapping) else {}
    horizon_index = _horizon_index(datasets, rows)
    groups = _paired_groups(rows, horizon_index=horizon_index, split=split)
    candidates = _rank_candidates(groups, split=split, max_candidates=max_candidates)
    family_summary = _family_summary(groups)
    planned = _planned_shared_configs(candidates)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "comparison_dir": str(comparison),
        "split": str(split),
        "source_row_count": len(rows),
        "horizon_index": horizon_index,
        "paired_group_count": len(groups),
        "paired_groups": groups,
        "top_candidates": candidates,
        "family_summary": family_summary,
        "planned_shared_horizon_configs": planned,
        "recommendations": _recommendations(candidates, family_summary),
        "limitations": [
            "This report compares separately trained single-horizon runs; it does not train a shared multi-horizon model yet.",
            "Candidate rankings depend on the currently completed comparison manifest and should be regenerated as the active sweep completes.",
            "Cross-horizon consistency is approximated from aggregate metrics, not from matched per-window predictions.",
        ],
    }
    json_path = out / "multi_horizon_report.json"
    markdown_path = out / "multi_horizon_report.md"
    manifest_path = out / "multi_horizon_plan_manifest.json"
    report["report_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    report["plan_manifest_path"] = str(manifest_path)
    plan_manifest = {
        "schema_version": 1,
        "manifest_kind": "multi_horizon_shared_model_plan",
        "created_at": report["created_at"],
        "source_comparison_dir": str(comparison),
        "split": str(split),
        "horizon_keys": sorted(horizon_index, key=lambda key: horizon_index[key].get("prediction_horizon_frames") or 0),
        "planned_configs": planned,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_multi_horizon_markdown(report), encoding="utf-8")
    manifest_path.write_text(json.dumps(plan_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report



def build_shared_horizon_baseline_comparison(
    *,
    runs: Sequence[str | Path],
    out_dir: str | Path,
    title: str = "Shared-Horizon Baseline Comparison",
) -> dict[str, Any]:
    """Compare completed shared-horizon linear/GRU/Transformer metric artifacts."""
    if not runs:
        raise ValueError("At least one shared-horizon metrics artifact is required.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = [_shared_horizon_comparison_record(item) for item in runs]
    records = sorted(records, key=lambda row: _sort_num(row.get("improvement_over_persistence_mse")), reverse=True)
    horizon_summary = _shared_horizon_comparison_horizon_summary(records)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "run_count": len(records),
        "runs": records,
        "horizon_summary": horizon_summary,
        "best_overall": records[0] if records else None,
        "active_cell_warnings": _shared_horizon_active_warnings(records),
        "recommendations": _shared_horizon_comparison_recommendations(records),
    }
    json_path = out / "shared_horizon_baseline_comparison.json"
    markdown_path = out / "shared_horizon_baseline_comparison.md"
    report["report_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_shared_horizon_baseline_comparison_markdown(report), encoding="utf-8")
    return report


def build_shared_horizon_review_manifest(
    *,
    runs: Sequence[str | Path],
    out_dir: str | Path,
    comparison_dir: str | Path | None = None,
    datasets: Sequence[str | Path] | None = None,
    title: str = "Shared-Horizon Review Input",
) -> dict[str, Any]:
    """Write a comparison_manifest.json from shared-horizon neural metrics.

    The resulting manifest uses the same row shape as the single-horizon
    comparison artifacts consumed by the video-error review builder.
    """
    if not runs:
        raise ValueError("At least one shared-horizon metrics artifact is required.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    shared_runs = []
    for item in runs:
        label, metrics_path = _parse_labeled_path(item)
        metrics = _load_json(metrics_path)
        label = label or _shared_horizon_default_label(metrics, metrics_path)
        shared_runs.append(
            {
                "label": str(label),
                "metrics_path": str(metrics_path),
                "model_family": metrics.get("model_family"),
                "model_kind": metrics.get("model_kind"),
                "shared_horizons_frames": metrics.get("shared_horizons_frames"),
                "improvement_over_persistence_mse": metrics.get("improvement_over_persistence_mse"),
            }
        )
        for dataset_key, horizon_metrics in sorted((metrics.get("per_horizon_metrics") or {}).items(), key=lambda kv: _sort_num((kv[1] or {}).get("prediction_horizon_frames"))):
            if not isinstance(horizon_metrics, Mapping):
                continue
            rows.append(_shared_horizon_review_row(label=str(label), metrics_path=metrics_path, metrics=metrics, dataset_key=str(dataset_key), horizon_metrics=horizon_metrics))
    manifest = {
        "schema_version": 1,
        "manifest_kind": "shared_horizon_review_input",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "source": "shared_horizon_review_manifest",
        "comparison_dir": str(comparison_dir) if comparison_dir is not None else None,
        "run_count": len(shared_runs),
        "row_count": len(rows),
        "shared_runs": shared_runs,
        "datasets": _shared_horizon_review_datasets(comparison_dir=comparison_dir, datasets=datasets),
        "rows": rows,
    }
    manifest_path = out / "comparison_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _shared_horizon_review_row(
    *,
    label: str,
    metrics_path: Path,
    metrics: Mapping[str, Any],
    dataset_key: str,
    horizon_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    model_family = _shared_horizon_review_model_family(metrics)
    row = {
        "experiment_id": f"{_safe_slug(label)}_{dataset_key}",
        "row_id": f"shared-horizon-review:{label}:{dataset_key}",
        "dataset_key": dataset_key,
        "kind": model_family,
        "model_family": model_family,
        "shared_model_family": metrics.get("model_family"),
        "shared_model_kind": metrics.get("model_kind"),
        "prediction_target": metrics.get("prediction_target"),
        "hyperparameter_summary": _shared_horizon_review_hparams(label, metrics),
        "metrics_path": _shared_horizon_review_metrics_path(metrics_path, dataset_key, horizon_metrics),
        "prediction_examples_path": _shared_horizon_review_artifact_path(metrics_path, dataset_key, horizon_metrics, "prediction_examples_path", "prediction_examples.json"),
        "prediction_clip_examples_path": _shared_horizon_review_artifact_path(metrics_path, dataset_key, horizon_metrics, "prediction_clip_examples_path", "prediction_clip_examples.json"),
        "params": _shared_horizon_params(metrics),
    }
    for key in (
        "prediction_horizon_frames",
        "prediction_horizon_sec",
        "decoded_prediction_mse",
        "persistence_mse",
        "improvement_over_persistence_mse",
        "train_decoded_prediction_mse",
        "train_persistence_mse",
        "train_improvement_over_persistence_mse",
        "val_decoded_prediction_mse",
        "val_persistence_mse",
        "val_improvement_over_persistence_mse",
        "test_decoded_prediction_mse",
        "test_persistence_mse",
        "test_improvement_over_persistence_mse",
        "test_active_cell_decoded_prediction_mse",
        "test_active_cell_persistence_mse",
        "test_active_cell_improvement_over_persistence_mse",
        "test_top_activity_decoded_prediction_mse",
        "test_top_activity_persistence_mse",
        "test_top_activity_improvement_over_persistence_mse",
        "test_high_change_decoded_prediction_mse",
        "test_high_change_persistence_mse",
        "test_high_change_improvement_over_persistence_mse",
    ):
        if key in horizon_metrics:
            row[key] = horizon_metrics.get(key)
    return row


def _shared_horizon_review_model_family(metrics: Mapping[str, Any]) -> str:
    text = str(metrics.get("model_family") or metrics.get("model_kind") or "").lower()
    if "transformer" in text:
        return "latent_transformer"
    if "gru" in text:
        return "latent_gru"
    if "linear" in text:
        return "linear_latent"
    return _safe_slug(metrics.get("model_family") or metrics.get("model_kind") or "shared_horizon")


def _shared_horizon_review_hparams(label: str, metrics: Mapping[str, Any]) -> str:
    params = _shared_horizon_params(metrics)
    pieces = [str(label)]
    if params.get("prediction_target"):
        pieces.append(f"target={params.get('prediction_target')}")
    for key in ("hidden_dim", "model_dim", "num_heads", "num_layers", "learning_rate", "evaluation_batch_size"):
        if key in params:
            pieces.append(f"{key}={params.get(key)}")
    return ", ".join(pieces)


def _shared_horizon_review_metrics_path(metrics_path: Path, dataset_key: str, horizon_metrics: Mapping[str, Any]) -> str:
    explicit = horizon_metrics.get("per_horizon_metrics_for_review_path")
    if explicit:
        return str(explicit)
    inferred = metrics_path.parent / dataset_key / "per_horizon_metrics_for_review.json"
    return str(inferred if inferred.exists() else metrics_path)


def _shared_horizon_review_artifact_path(metrics_path: Path, dataset_key: str, horizon_metrics: Mapping[str, Any], key: str, filename: str) -> str | None:
    explicit = horizon_metrics.get(key)
    if explicit:
        return str(explicit)
    inferred = metrics_path.parent / dataset_key / filename
    return str(inferred) if inferred.exists() else None


def _shared_horizon_review_datasets(*, comparison_dir: str | Path | None, datasets: Sequence[str | Path] | None) -> dict[str, Any]:
    if comparison_dir is not None:
        manifest_path = Path(comparison_dir) / "comparison_manifest.json"
        manifest = _load_json(manifest_path)
        existing = manifest.get("datasets")
        if isinstance(existing, Mapping):
            return {str(key): value for key, value in existing.items()}
    out: dict[str, Any] = {}
    for path in datasets or []:
        dataset_path = Path(path)
        payload = _load_json(dataset_path)
        key = str(payload.get("dataset_key") or dataset_path.parent.name)
        out[key] = {
            "dataset_path": str(dataset_path),
            "windowing": payload.get("windowing") or payload.get("extras", {}).get("windowing"),
        }
    return out


def render_shared_horizon_baseline_comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Shared-Horizon Baseline Comparison')}",
        "",
        f"Generated: `{report.get('created_at')}`",
        f"Runs compared: `{report.get('run_count')}`",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Label | Family | Horizons | Improve | Decoded MSE | Persistence MSE | Selection latent MSE |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(report.get("runs", []), start=1):
        horizons = ", ".join(str(v) for v in row.get("shared_horizons_frames") or []) or "n/a"
        lines.append(
            f"| {idx} | `{row.get('label')}` | {row.get('model_family')} | {horizons} | {_fmt(row.get('improvement_over_persistence_mse'))} | {_fmt(row.get('decoded_prediction_mse'))} | {_fmt(row.get('persistence_mse'))} | {_fmt(row.get('selection_latent_code_mse'))} |"
        )
    lines.extend(["", "## Per-Horizon Test Metrics", ""])
    if report.get("horizon_summary"):
        for horizon in report.get("horizon_summary", []):
            lines.extend([
                f"### {horizon.get('dataset_key')}",
                "",
                "| Rank | Label | Test improve | Test MSE | Persistence MSE | Active-cell improve | High-change improve | Clip artifact |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ])
            for idx, row in enumerate(horizon.get("runs", []), start=1):
                clip = row.get("prediction_clip_examples_path") or ""
                lines.append(
                    f"| {idx} | `{row.get('label')}` | {_fmt(row.get('test_improvement_over_persistence_mse'))} | {_fmt(row.get('test_decoded_prediction_mse'))} | {_fmt(row.get('test_persistence_mse'))} | {_fmt(row.get('test_active_cell_improvement_over_persistence_mse'))} | {_fmt(row.get('test_high_change_improvement_over_persistence_mse'))} | `{clip}` |"
                )
            lines.append("")
    else:
        lines.append("No per-horizon metrics were available.")
    lines.extend(["## Active-Cell Warnings", ""])
    warnings = list(report.get("active_cell_warnings") or [])
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("No active-cell regressions were detected in the compared artifacts.")
    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Artifact Paths", ""])
    for row in report.get("runs", []):
        lines.append(f"- {row.get('label')}: `{row.get('metrics_path')}`")
    return "\n".join(lines).rstrip() + "\n"


def _shared_horizon_comparison_record(item: str | Path) -> dict[str, Any]:
    label, path = _parse_labeled_path(item)
    metrics = _load_json(path)
    label = label or _shared_horizon_default_label(metrics, path)
    per_horizon = []
    for key, horizon_metrics in sorted((metrics.get("per_horizon_metrics") or {}).items(), key=lambda kv: _sort_num((kv[1] or {}).get("prediction_horizon_frames"))):
        if not isinstance(horizon_metrics, Mapping):
            continue
        per_horizon.append(_shared_horizon_per_horizon_record(label, key, horizon_metrics))
    return {
        "label": str(label),
        "metrics_path": str(path),
        "model_family": metrics.get("model_family"),
        "model_kind": metrics.get("model_kind"),
        "objective": metrics.get("objective"),
        "shared_horizons_frames": metrics.get("shared_horizons_frames"),
        "decoded_prediction_mse": metrics.get("decoded_prediction_mse"),
        "persistence_mse": metrics.get("persistence_mse"),
        "improvement_over_persistence_mse": metrics.get("improvement_over_persistence_mse"),
        "selection_latent_code_mse": metrics.get("selection_latent_code_mse"),
        "training_window_count": metrics.get("training_window_count"),
        "evaluation_window_count": metrics.get("evaluation_window_count"),
        "decoded_evaluation_mode": metrics.get("decoded_evaluation_mode"),
        "evaluation_batch_size": metrics.get("evaluation_batch_size"),
        "params": _shared_horizon_params(metrics),
        "per_horizon_metrics": per_horizon,
    }


def _shared_horizon_per_horizon_record(label: str, dataset_key: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": str(label),
        "dataset_key": str(dataset_key),
        "prediction_horizon_frames": metrics.get("prediction_horizon_frames"),
        "prediction_horizon_sec": metrics.get("prediction_horizon_sec"),
        "decoded_prediction_mse": metrics.get("decoded_prediction_mse"),
        "persistence_mse": metrics.get("persistence_mse"),
        "improvement_over_persistence_mse": metrics.get("improvement_over_persistence_mse"),
        "test_decoded_prediction_mse": metrics.get("test_decoded_prediction_mse"),
        "test_persistence_mse": metrics.get("test_persistence_mse"),
        "test_improvement_over_persistence_mse": metrics.get("test_improvement_over_persistence_mse"),
        "test_active_cell_improvement_over_persistence_mse": metrics.get("test_active_cell_improvement_over_persistence_mse"),
        "test_top_activity_improvement_over_persistence_mse": metrics.get("test_top_activity_improvement_over_persistence_mse"),
        "test_high_change_improvement_over_persistence_mse": metrics.get("test_high_change_improvement_over_persistence_mse"),
        "prediction_clip_examples_path": metrics.get("prediction_clip_examples_path"),
    }


def _shared_horizon_comparison_horizon_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for horizon in record.get("per_horizon_metrics", []):
            item = dict(horizon)
            item["model_family"] = record.get("model_family")
            item["metrics_path"] = record.get("metrics_path")
            by_dataset[str(item.get("dataset_key"))].append(item)
    summary = []
    for dataset_key, rows in sorted(by_dataset.items(), key=lambda kv: _sort_num(kv[1][0].get("prediction_horizon_frames") if kv[1] else None)):
        ranked = sorted(rows, key=lambda row: _sort_num(row.get("test_improvement_over_persistence_mse")), reverse=True)
        summary.append(
            {
                "dataset_key": dataset_key,
                "prediction_horizon_frames": ranked[0].get("prediction_horizon_frames") if ranked else None,
                "prediction_horizon_sec": ranked[0].get("prediction_horizon_sec") if ranked else None,
                "best_label": ranked[0].get("label") if ranked else None,
                "runs": ranked,
            }
        )
    return summary


def _shared_horizon_active_warnings(records: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings = []
    for record in records:
        for horizon in record.get("per_horizon_metrics", []):
            value = _num(horizon.get("test_active_cell_improvement_over_persistence_mse"))
            if value is not None and value < 0:
                warnings.append(
                    f"{record.get('label')} {horizon.get('dataset_key')} has negative test active-cell improvement ({_fmt(value)}), despite overall improvement {_fmt(record.get('improvement_over_persistence_mse'))}."
                )
    return warnings


def _shared_horizon_comparison_recommendations(records: Sequence[Mapping[str, Any]]) -> list[str]:
    if not records:
        return ["No shared-horizon metric artifacts were available to compare."]
    best = max(records, key=lambda row: _sort_num(row.get("improvement_over_persistence_mse")))
    recs = [f"Best overall shared-horizon run by persistence improvement is `{best.get('label')}` with improvement `{_fmt(best.get('improvement_over_persistence_mse'))}`."]
    if _shared_horizon_active_warnings([best]):
        recs.append("Do not present the best overall run as an active-cell forecasting success; active-cell improvement is still negative for at least one horizon.")
    per_horizon = _shared_horizon_comparison_horizon_summary(records)
    if per_horizon:
        winners = ", ".join(f"{h.get('dataset_key')}=`{h.get('best_label')}`" for h in per_horizon)
        recs.append(f"Per-horizon test-improvement winners: {winners}.")
    recs.append("Use this comparison together with temporal clip review before selecting the next shared-GRU or Transformer follow-up run.")
    return recs


def _shared_horizon_params(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keep = ["hidden_dim", "num_layers", "model_dim", "num_heads", "learning_rate", "prediction_target", "decoded_evaluation_mode", "evaluation_batch_size"]
    return {key: metrics.get(key) for key in keep if key in metrics}


def _shared_horizon_default_label(metrics: Mapping[str, Any], path: Path) -> str:
    family = str(metrics.get("model_family") or metrics.get("model_kind") or path.parent.name)
    return _safe_slug(family).replace("_", "-")


def _parse_labeled_path(item: str | Path) -> tuple[str | None, Path]:
    text = str(item)
    if "=" in text:
        label, raw_path = text.split("=", 1)
        return (label.strip() or None), Path(raw_path)
    return None, Path(text)


def build_active_cell_rescue_plan(
    *,
    comparison_report: str | Path,
    grid_status: str | Path,
    out_dir: str | Path,
    title: str = "Active-Cell Rescue Plan",
) -> dict[str, Any]:
    """Plan the next shared-horizon step after active-cell regressions."""
    comparison_path = Path(comparison_report)
    status_path = Path(grid_status)
    comparison = _load_json(comparison_path)
    status = _load_json(status_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings = list(comparison.get("active_cell_warnings") or [])
    rows = [dict(row) for row in status.get("rows", []) if isinstance(row, Mapping)]
    pending = [row for row in rows if row.get("status") == "pending"]
    completed = [row for row in rows if row.get("status") == "completed"]
    candidates = _active_cell_rescue_candidates(pending)
    recommendations = _active_cell_rescue_recommendations(comparison, candidates)
    plan = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "comparison_report": str(comparison_path),
        "grid_status": str(status_path),
        "best_overall_label": (comparison.get("best_overall") or {}).get("label"),
        "best_overall_improvement_over_persistence_mse": (comparison.get("best_overall") or {}).get("improvement_over_persistence_mse"),
        "active_cell_warning_count": len(warnings),
        "active_cell_warnings": warnings,
        "completed_shared_neural_count": len(completed),
        "pending_shared_neural_count": len(pending),
        "recommended_candidates": candidates,
        "recommendations": recommendations,
        "non_goals": [
            "Do not run another same-objective GRU solely to lower global MSE before testing an architecture or objective that could plausibly help active cells.",
            "Do not claim active-cell success until test_active_cell_improvement_over_persistence_mse is positive for both h2 and h5.",
        ],
    }
    json_path = out / "active_cell_rescue_plan.json"
    markdown_path = out / "active_cell_rescue_plan.md"
    plan["plan_path"] = str(json_path)
    plan["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_active_cell_rescue_plan_markdown(plan), encoding="utf-8")
    return plan


def render_active_cell_rescue_plan_markdown(plan: Mapping[str, Any]) -> str:
    lines = [
        f"# {plan.get('title', 'Active-Cell Rescue Plan')}",
        "",
        f"Generated: `{plan.get('created_at')}`",
        f"Comparison report: `{plan.get('comparison_report')}`",
        f"Grid status: `{plan.get('grid_status')}`",
        "",
        "## Diagnosis",
        "",
        f"- Best shared-horizon global run: `{plan.get('best_overall_label')}` with improvement `{_fmt(plan.get('best_overall_improvement_over_persistence_mse'))}`.",
        f"- Active-cell warnings across compared runs: `{plan.get('active_cell_warning_count')}`.",
        f"- Completed shared-neural planned entries: `{plan.get('completed_shared_neural_count')}`.",
        f"- Pending shared-neural planned entries: `{plan.get('pending_shared_neural_count')}`.",
        "",
        "## Recommended Candidates",
        "",
    ]
    candidates = list(plan.get("recommended_candidates") or [])
    if candidates:
        lines.extend(["| Rank | Config | Family | Priority | Rationale |", "|---:|---|---|---|---|"])
        for idx, row in enumerate(candidates, start=1):
            lines.append(f"| {idx} | `{row.get('config_id')}` | {row.get('model_family')} | {row.get('priority') or ''} | {row.get('rationale')} |")
        first = candidates[0]
        if first.get("command"):
            lines.extend(
                [
                    "",
                    "## Next Candidate Command",
                    "",
                    "Run this only when CPU/RAM headroom is intentionally available and the active GPU sweep does not need intervention.",
                    "",
                    "```bash",
                    str(first.get("command")),
                    "```",
                ]
            )
            if first.get("out_dir"):
                lines.append(f"Expected output directory: `{first.get('out_dir')}`")
    else:
        lines.append("No pending candidates were available in the shared-neural grid status artifact.")
    lines.extend(["", "## Recommendations", ""])
    for item in plan.get("recommendations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Non-Goals", ""])
    for item in plan.get("non_goals", []):
        lines.append(f"- {item}")
    if plan.get("active_cell_warnings"):
        lines.extend(["", "## Active-Cell Warning Evidence", ""])
        for item in plan.get("active_cell_warnings", []):
            lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def _active_cell_rescue_candidates(pending_rows: Sequence[Mapping[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    candidates = []
    for row in pending_rows:
        family = str(row.get("model_family") or "")
        params = row.get("params") if isinstance(row.get("params"), Mapping) else {}
        if "transformer" in family:
            score = (0, int(params.get("model_dim") or 10**9), int(params.get("num_layers") or 10**9), int(params.get("num_heads") or 10**9))
            rationale = "Architecture-diverse next test after linear and GRU both regress on active cells; choose the smallest Transformer first to limit CPU cost."
        elif "gru" in family:
            score = (1, int(params.get("num_layers") or 10**9), int(params.get("hidden_dim") or 10**9), float(params.get("learning_rate") or 1e9))
            rationale = "Same latent-MSE GRU objective; only run after architecture-diverse candidates or if testing seed/layer sensitivity intentionally."
        else:
            score = (2, 0, 0, 0)
            rationale = "Lower priority because it does not directly address the active-cell failure hypothesis."
        candidates.append((score, row, rationale))
    selected = []
    for _, row, rationale in sorted(candidates, key=lambda item: item[0])[: max(0, int(limit))]:
        selected.append(
            {
                "config_id": row.get("config_id"),
                "model_family": row.get("model_family"),
                "model_kind": row.get("model_kind"),
                "priority": row.get("priority"),
                "params": row.get("params") if isinstance(row.get("params"), Mapping) else {},
                "out_dir": row.get("out_dir"),
                "command": row.get("command"),
                "rationale": rationale,
            }
        )
    return selected


def _active_cell_rescue_recommendations(comparison: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    recs = []
    best = comparison.get("best_overall") or {}
    if best:
        recs.append(f"Use `{best.get('label')}` as the current global shared-horizon control, but not as an active-cell success claim.")
    if comparison.get("active_cell_warnings"):
        recs.append("The next run should change model family or objective emphasis; all compared shared-horizon runs currently have negative active-cell improvement evidence.")
    first = candidates[0] if candidates else None
    if first:
        recs.append(f"Next candidate: `{first.get('config_id')}` because {first.get('rationale')}")
    recs.append("After any rescue run completes, regenerate shared_horizon_baseline_comparison_v1 and this plan before launching another candidate.")
    return recs


def build_shared_horizon_neural_grid_plan(
    *,
    datasets: Sequence[str | Path],
    autoencoder_run: str | Path,
    out_dir: str | Path,
    run_root: str | Path,
    device: str = "cpu",
    epochs: int = 25,
    batch_size: int = 64,
    evaluation_batch_size: int = 16,
    seeds: Sequence[int] = (7, 13),
    max_gru_configs: int = 16,
    include_transformer_placeholders: bool = True,
    title: str = "Shared-Horizon Neural Follow-Up Grid",
) -> dict[str, Any]:
    """Write an inspectable shared-horizon neural follow-up grid plan.

    The planner intentionally does not launch training. GRU and Transformer
    entries are directly executable with the current shared-horizon CLIs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_root_path = Path(run_root)
    dataset_paths = [Path(path) for path in datasets]
    if len(dataset_paths) < 2:
        raise ValueError("At least two dataset JSON paths are required for a shared-horizon neural grid plan.")
    if not dataset_paths:
        raise ValueError("No datasets were provided.")
    created_at = datetime.now(timezone.utc).isoformat()
    horizon_slug = _horizon_slug(dataset_paths)
    seed_values = [int(seed) for seed in seeds] or [7]
    gru_specs = _shared_gru_grid_specs(
        dataset_paths=dataset_paths,
        autoencoder_run=Path(autoencoder_run),
        run_root=run_root_path,
        horizon_slug=horizon_slug,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        evaluation_batch_size=int(evaluation_batch_size),
        seeds=seed_values,
        max_configs=int(max_gru_configs),
    )
    transformer_specs = _shared_transformer_placeholder_specs(
        horizon_slug=horizon_slug,
        dataset_paths=dataset_paths,
        autoencoder_run=Path(autoencoder_run),
        run_root=run_root_path,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        evaluation_batch_size=int(evaluation_batch_size),
        seeds=seed_values,
    ) if include_transformer_placeholders else []
    planned = [*gru_specs, *transformer_specs]
    manifest = {
        "schema_version": 1,
        "manifest_kind": "shared_horizon_neural_followup_grid",
        "created_at": created_at,
        "title": str(title),
        "dataset_paths": [str(path) for path in dataset_paths],
        "autoencoder_run": str(autoencoder_run),
        "run_root": str(run_root_path),
        "device": str(device),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "evaluation_batch_size": int(evaluation_batch_size),
        "seeds": seed_values,
        "planned_config_count": len(planned),
        "directly_executable_count": sum(1 for spec in planned if spec.get("status") == "ready"),
        "placeholder_count": sum(1 for spec in planned if spec.get("status") != "ready"),
        "planned_configs": planned,
        "recommendations": [
            "Run the smallest GRU or Transformer entries first and compare per-horizon test improvement to shared linear and the completed shared-GRU baseline.",
            "Keep device=cpu while the main CUDA overnight sweep is active unless the user explicitly changes resource allocation.",
            "Use the status report command after any launched entry to rank completed shared-neural results and detect partial runs.",
        ],
    }
    manifest_path = out / "shared_horizon_neural_grid_manifest.json"
    markdown_path = out / "shared_horizon_neural_grid_plan.md"
    script_path = out / "run_shared_horizon_neural_grid.sh"
    manifest["manifest_path"] = str(manifest_path)
    manifest["markdown_path"] = str(markdown_path)
    manifest["script_path"] = str(script_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_shared_horizon_neural_grid_markdown(manifest), encoding="utf-8")
    script_path.write_text(_render_shared_gru_script(manifest), encoding="utf-8")
    script_path.chmod(0o755)
    return manifest



def build_shared_horizon_neural_grid_status(
    *,
    manifest_path: str | Path,
    out_dir: str | Path | None = None,
    title: str = "Shared-Horizon Neural Grid Status",
) -> dict[str, Any]:
    """Summarize completion and metrics for a shared-horizon neural grid plan."""
    manifest_file = Path(manifest_path)
    manifest = _load_json(manifest_file)
    out = Path(out_dir) if out_dir is not None else manifest_file.parent
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in manifest.get("planned_configs", []):
        if not isinstance(spec, Mapping):
            continue
        row = _shared_grid_status_row(spec)
        rows.append(row)
    status_counts = dict(Counter(row["status"] for row in rows))
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        family_counts[str(row.get("model_family") or "unknown")][str(row.get("status") or "unknown")] += 1
    completed = [row for row in rows if row.get("status") == "completed"]
    completed_sorted = sorted(completed, key=lambda row: _sort_num(row.get("improvement_over_persistence_mse")), reverse=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "manifest_path": str(manifest_file),
        "source_plan_title": manifest.get("title"),
        "planned_config_count": len(rows),
        "status_counts": status_counts,
        "family_status_counts": {family: dict(counts) for family, counts in sorted(family_counts.items())},
        "completed_count": len(completed),
        "pending_count": status_counts.get("pending", 0),
        "started_count": status_counts.get("started", 0),
        "incomplete_count": status_counts.get("incomplete", 0),
        "best_completed": completed_sorted[:10],
        "rows": rows,
        "recommendations": _shared_grid_status_recommendations(rows),
    }
    json_path = out / "shared_horizon_neural_grid_status.json"
    markdown_path = out / "shared_horizon_neural_grid_status.md"
    report["status_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_shared_horizon_neural_grid_status_markdown(report), encoding="utf-8")
    return report


def render_shared_horizon_neural_grid_status_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Shared-Horizon Neural Grid Status')}",
        "",
        f"Generated: `{report.get('created_at')}`",
        f"Manifest: `{report.get('manifest_path')}`",
        f"Planned configs: `{report.get('planned_config_count')}`",
        f"Status counts: `{report.get('status_counts')}`",
        "",
        "## Family Status",
        "",
        "| Family | Completed | Started | Incomplete | Pending |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, counts in (report.get("family_status_counts") or {}).items():
        lines.append(f"| {family} | {counts.get('completed', 0)} | {counts.get('started', 0)} | {counts.get('incomplete', 0)} | {counts.get('pending', 0)} |")
    lines.extend(["", "## Best Completed", ""])
    if report.get("best_completed"):
        lines.extend(["| Rank | Config | Family | Improve | Test active min | Test high-change min | Active horizons |", "|---:|---|---|---:|---:|---:|---:|"])
        for idx, row in enumerate(report.get("best_completed", []), start=1):
            active = f"{row.get('test_active_cell_positive_horizon_count', 0)}/{row.get('test_active_cell_horizon_count', 0)}"
            lines.append(
                f"| {idx} | `{row.get('config_id')}` | {row.get('model_family')} | {_fmt(row.get('improvement_over_persistence_mse'))} | {_fmt(row.get('min_test_active_cell_improvement_over_persistence_mse'))} | {_fmt(row.get('min_test_high_change_improvement_over_persistence_mse'))} | {active} |"
            )
    else:
        lines.append("No planned shared-neural configs have completed metrics yet.")
    lines.extend(["", "## Recommendations", ""])
    for rec in report.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.extend(["", "## Rows", "", "| Config | Family | Status | Test active min | Test high-change min | Progress phase | Metrics |", "|---|---|---|---:|---:|---|---|"])
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('config_id')}` | {row.get('model_family')} | {row.get('status')} | {_fmt(row.get('min_test_active_cell_improvement_over_persistence_mse'))} | {_fmt(row.get('min_test_high_change_improvement_over_persistence_mse'))} | {row.get('progress_phase') or ''} | `{row.get('metrics_path') or ''}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _shared_grid_status_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = Path(str(spec.get("out_dir") or ""))
    family = str(spec.get("model_family") or "unknown")
    prefix = "multi_horizon_transformer" if "transformer" in family else "multi_horizon_gru"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    run_path = out_dir / f"{prefix}_run.json"
    latest_path = out_dir / f"{prefix}_progress_latest.json"
    progress = _load_json_if_exists(latest_path)
    metrics = _load_json_if_exists(metrics_path)
    if metrics:
        status = "completed"
    elif progress and progress.get("phase") == "complete":
        status = "incomplete"
    elif progress:
        status = "started"
    else:
        status = "pending"
    row = {
        "config_id": str(spec.get("config_id") or out_dir.name),
        "model_family": family,
        "model_kind": spec.get("model_kind"),
        "status": status,
        "priority": spec.get("priority"),
        "out_dir": str(out_dir),
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "run_path": str(run_path) if run_path.exists() else None,
        "progress_latest_path": str(latest_path) if latest_path.exists() else None,
        "progress_phase": progress.get("phase") if isinstance(progress, Mapping) else None,
        "progress_message": progress.get("message") if isinstance(progress, Mapping) else None,
        "command": spec.get("command"),
        "params": spec.get("params") if isinstance(spec.get("params"), Mapping) else {},
    }
    if metrics:
        for key in (
            "decoded_prediction_mse",
            "persistence_mse",
            "improvement_over_persistence_mse",
            "selection_latent_code_mse",
            "evaluation_window_count",
            "training_window_count",
            "decoded_evaluation_mode",
            "evaluation_batch_size",
        ):
            if key in metrics:
                row[key] = metrics.get(key)
        row["shared_horizons_frames"] = metrics.get("shared_horizons_frames")
        row.update(_shared_grid_per_horizon_summary(metrics))
    return row


def _shared_grid_per_horizon_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    per_horizon = metrics.get("per_horizon_metrics")
    if not isinstance(per_horizon, Mapping):
        return {
            "per_horizon_test_metrics": [],
            "test_active_cell_horizon_count": 0,
            "test_active_cell_positive_horizon_count": 0,
            "all_test_active_cell_positive": False,
            "min_test_active_cell_improvement_over_persistence_mse": None,
            "min_test_high_change_improvement_over_persistence_mse": None,
        }
    rows = []
    for dataset_key, item in sorted(per_horizon.items(), key=lambda kv: str(kv[0])):
        if not isinstance(item, Mapping):
            continue
        active = _num(item.get("test_active_cell_improvement_over_persistence_mse"))
        high_change = _num(item.get("test_high_change_improvement_over_persistence_mse"))
        top_activity = _num(item.get("test_top_activity_improvement_over_persistence_mse"))
        test_improve = _num(item.get("test_improvement_over_persistence_mse"))
        rows.append(
            {
                "dataset_key": str(item.get("dataset_key") or dataset_key),
                "prediction_horizon_frames": item.get("prediction_horizon_frames"),
                "test_improvement_over_persistence_mse": test_improve,
                "test_active_cell_improvement_over_persistence_mse": active,
                "test_high_change_improvement_over_persistence_mse": high_change,
                "test_top_activity_improvement_over_persistence_mse": top_activity,
            }
        )
    active_values = [row["test_active_cell_improvement_over_persistence_mse"] for row in rows if row.get("test_active_cell_improvement_over_persistence_mse") is not None]
    high_values = [row["test_high_change_improvement_over_persistence_mse"] for row in rows if row.get("test_high_change_improvement_over_persistence_mse") is not None]
    active_positive = sum(1 for value in active_values if value > 0)
    return {
        "per_horizon_test_metrics": rows,
        "test_active_cell_horizon_count": len(active_values),
        "test_active_cell_positive_horizon_count": active_positive,
        "all_test_active_cell_positive": bool(active_values and active_positive == len(active_values)),
        "min_test_active_cell_improvement_over_persistence_mse": min(active_values) if active_values else None,
        "min_test_high_change_improvement_over_persistence_mse": min(high_values) if high_values else None,
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _shared_grid_status_recommendations(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    counts = Counter(str(row.get("status")) for row in rows)
    recs = []
    if counts.get("completed", 0) == 0:
        recs.append("No shared-neural grid entries have completed yet; run one small CPU entry at a time when resources are intentionally allocated.")
    if counts.get("started", 0):
        recs.append("At least one entry has progress artifacts but no final metrics; inspect its progress_latest JSON before launching more work.")
    if counts.get("incomplete", 0):
        recs.append("Some entries reached a complete progress phase without metrics; inspect their output directories for export failures.")
    completed = [row for row in rows if row.get("status") == "completed"]
    active_rows = [row for row in completed if row.get("test_active_cell_horizon_count")]
    negative_active = [row for row in active_rows if _sort_num(row.get("min_test_active_cell_improvement_over_persistence_mse")) < 0]
    positive_active = [row for row in active_rows if row.get("all_test_active_cell_positive")]
    if counts.get("completed", 0):
        recs.append("Compare the best completed shared-neural result against shared linear, persistence, active-cell metrics, and high-change metrics before claiming an improvement.")
    if negative_active and not positive_active:
        recs.append("Every completed shared-neural entry with active-cell diagnostics still has at least one horizon below persistence on test active-cell improvement.")
    if positive_active:
        labels = ", ".join(str(row.get("config_id")) for row in positive_active[:3])
        recs.append(f"Completed entries with positive test active-cell improvement on every measured horizon: {labels}.")
    return recs


def render_shared_horizon_neural_grid_markdown(plan: Mapping[str, Any]) -> str:
    lines = [
        f"# {plan.get('title', 'Shared-Horizon Neural Follow-Up Grid')}",
        "",
        f"Generated: `{plan.get('created_at')}`",
        f"Datasets: `{', '.join(plan.get('dataset_paths', []))}`",
        f"Autoencoder: `{plan.get('autoencoder_run')}`",
        f"Run root: `{plan.get('run_root')}`",
        f"Device: `{plan.get('device')}`",
        f"Planned configs: `{plan.get('planned_config_count')}`; directly executable: `{plan.get('directly_executable_count')}`; placeholders: `{plan.get('placeholder_count')}`",
        "",
        "## Executable GRU Grid",
        "",
        "| Config | Hidden | Layers | LR | Seed | Output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for spec in plan.get("planned_configs", []):
        if spec.get("model_family") != "multi_horizon_latent_gru":
            continue
        params = spec.get("params", {})
        lines.append(f"| `{spec.get('config_id')}` | {params.get('hidden_dim')} | {params.get('num_layers')} | {params.get('learning_rate')} | {params.get('seed')} | `{spec.get('out_dir')}` |")
    transformer_ready = [spec for spec in plan.get("planned_configs", []) if spec.get("model_family") == "multi_horizon_latent_transformer" and spec.get("status") == "ready"]
    if transformer_ready:
        lines.extend(["", "## Executable Transformer Grid", "", "| Config | Model dim | Heads | Layers | LR | Seed | Output |", "|---|---:|---:|---:|---:|---:|---|"])
        for spec in transformer_ready:
            params = spec.get("params", {})
            lines.append(f"| `{spec.get('config_id')}` | {params.get('model_dim')} | {params.get('num_heads')} | {params.get('num_layers')} | {params.get('learning_rate')} | {params.get('seed')} | `{spec.get('out_dir')}` |")
    placeholders = [spec for spec in plan.get("planned_configs", []) if spec.get("status") != "ready"]
    if placeholders:
        lines.extend(["", "## Transformer Design Targets", "", "| Config | Model dim | Heads | Layers | LR | Status |", "|---|---:|---:|---:|---:|---|"])
        for spec in placeholders:
            params = spec.get("params", {})
            lines.append(f"| `{spec.get('config_id')}` | {params.get('model_dim')} | {params.get('num_heads')} | {params.get('num_layers')} | {params.get('learning_rate')} | {spec.get('status')} |")
    lines.extend(["", "## How To Launch", "", f"Executable neural commands were written to `{plan.get('script_path')}`.", "", "## Recommendations", ""])
    for rec in plan.get("recommendations", []):
        lines.append(f"- {rec}")
    return "\n".join(lines).rstrip() + "\n"


def _shared_gru_grid_specs(
    *,
    dataset_paths: Sequence[Path],
    autoencoder_run: Path,
    run_root: Path,
    horizon_slug: str,
    device: str,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    seeds: Sequence[int],
    max_configs: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, int, float, int, str]] = []
    for hidden_dim in (32, 64, 96):
        for num_layers in (1, 2):
            for learning_rate in (3e-4, 1e-3):
                for seed in seeds:
                    priority = "core" if hidden_dim in {32, 64} and num_layers == 1 else "expanded"
                    candidates.append((hidden_dim, num_layers, learning_rate, int(seed), priority))
    priority_order = {"core": 0, "expanded": 1}
    candidates = sorted(candidates, key=lambda item: (priority_order[item[4]], item[0], item[1], item[2], item[3]))[: max(0, int(max_configs))]
    specs = []
    for hidden_dim, num_layers, learning_rate, seed, priority in candidates:
        config_id = f"shgru_{horizon_slug}_delta_hd{hidden_dim}_l{num_layers}_lr{_lr_slug(learning_rate)}_s{seed}"
        out_dir = run_root / config_id
        command = _shared_gru_command(
            dataset_paths=dataset_paths,
            autoencoder_run=autoencoder_run,
            out_dir=out_dir,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            evaluation_batch_size=evaluation_batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        specs.append(
            {
                "config_id": config_id,
                "model_family": "multi_horizon_latent_gru",
                "model_kind": "shared_multi_horizon_latent_gru",
                "status": "ready",
                "priority": priority,
                "out_dir": str(out_dir),
                "params": {
                    "prediction_target": "delta",
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "evaluation_batch_size": evaluation_batch_size,
                    "seed": seed,
                    "device": device,
                    "decoded_evaluation_mode": "chunked",
                    "horizon_conditioning": "normalized_horizon_scalar_head_conditioning",
                },
                "command": command,
                "rationale": "Small shared-GRU follow-up around the completed hd64/l1/lr1e-3 baseline with lower-memory evaluation and heartbeat progress.",
            }
        )
    return specs


def _shared_transformer_placeholder_specs(
    *,
    horizon_slug: str,
    dataset_paths: Sequence[Path],
    autoencoder_run: Path,
    run_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    specs = []
    for model_dim, num_heads, num_layers, learning_rate in ((64, 2, 1, 1e-4), (64, 2, 2, 1e-4), (64, 4, 1, 3e-4), (128, 4, 1, 1e-4)):
        seed = int(seeds[0])
        config_id = f"shxfmr_{horizon_slug}_delta_md{model_dim}_h{num_heads}_l{num_layers}_lr{_lr_slug(learning_rate)}_s{seed}"
        out_dir = run_root / config_id
        command = _shared_transformer_command(
            dataset_paths=dataset_paths,
            autoencoder_run=autoencoder_run,
            out_dir=out_dir,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            evaluation_batch_size=evaluation_batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        specs.append(
            {
                "config_id": config_id,
                "model_family": "multi_horizon_latent_transformer",
                "model_kind": "shared_multi_horizon_latent_transformer",
                "status": "ready",
                "priority": "transformer_candidate",
                "out_dir": str(out_dir),
                "source_dataset_keys": [path.parent.name for path in dataset_paths],
                "params": {
                    "prediction_target": "delta",
                    "model_dim": model_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "dropout": 0.1,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "evaluation_batch_size": evaluation_batch_size,
                    "seed": seed,
                    "device": device,
                    "decoded_evaluation_mode": "chunked",
                    "horizon_conditioning": "normalized_horizon_scalar_head_conditioning",
                },
                "command": command,
                "rationale": "Shared-horizon Transformer candidate based on paired single-horizon Transformer evidence and now backed by a tested shared Transformer trainer.",
            }
        )
    return specs


def _shared_gru_command(
    *,
    dataset_paths: Sequence[Path],
    autoencoder_run: Path,
    out_dir: Path,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> str:
    parts = [".venv-neurobench/bin/python", "-m", "neurobench.cli.main", "dynamics", "train-shared-gru-horizons"]
    for dataset in dataset_paths:
        parts.extend(["--dataset", str(dataset)])
    parts.extend(
        [
            "--autoencoder-run", str(autoencoder_run),
            "--hidden-dim", str(hidden_dim),
            "--num-layers", str(num_layers),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--evaluation-batch-size", str(evaluation_batch_size),
            "--learning-rate", str(learning_rate),
            "--prediction-target", "delta",
            "--device", str(device),
            "--seed", str(seed),
            "--progress-interval-epochs", "1",
            "--out-dir", str(out_dir),
        ]
    )
    return " ".join(shlex.quote(part) for part in parts)



def _shared_transformer_command(
    *,
    dataset_paths: Sequence[Path],
    autoencoder_run: Path,
    out_dir: Path,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> str:
    parts = [".venv-neurobench/bin/python", "-m", "neurobench.cli.main", "dynamics", "train-shared-transformer-horizons"]
    for dataset in dataset_paths:
        parts.extend(["--dataset", str(dataset)])
    parts.extend(
        [
            "--autoencoder-run", str(autoencoder_run),
            "--model-dim", str(model_dim),
            "--num-heads", str(num_heads),
            "--num-layers", str(num_layers),
            "--dropout", "0.1",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--evaluation-batch-size", str(evaluation_batch_size),
            "--learning-rate", str(learning_rate),
            "--prediction-target", "delta",
            "--device", str(device),
            "--seed", str(seed),
            "--progress-interval-epochs", "1",
            "--out-dir", str(out_dir),
        ]
    )
    return " ".join(shlex.quote(part) for part in parts)


def _render_shared_gru_script(plan: Mapping[str, Any]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# Generated: {plan.get('created_at')}", "# Run one command at a time unless CPU/RAM headroom is confirmed.", ""]
    for spec in plan.get("planned_configs", []):
        if spec.get("status") != "ready" or not spec.get("command"):
            continue
        lines.append(f"# {spec.get('config_id')} - {spec.get('priority')}")
        lines.append(str(spec.get("command")))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _horizon_slug(dataset_paths: Sequence[Path]) -> str:
    parts = []
    for path in dataset_paths:
        key = path.parent.name or path.stem
        frames = _parse_horizon_frames(key)
        parts.append(f"h{frames}" if frames is not None else _safe_slug(key))
    return "_".join(parts)


def _lr_slug(value: float) -> str:
    text = f"{float(value):.0e}".replace("e-0", "em").replace("e-", "em").replace("e+0", "ep").replace("e+", "ep")
    return text.replace(".", "p")


def render_multi_horizon_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Multi-Horizon Forecasting Report')}",
        "",
        f"Generated: `{report.get('created_at')}`",
        f"Comparison directory: `{report.get('comparison_dir')}`",
        f"Split: `{report.get('split')}`",
        f"Source rows: `{report.get('source_row_count')}`",
        f"Paired h2/h5-style groups: `{report.get('paired_group_count')}`",
        "",
        "## Horizon Inputs",
        "",
        "| Dataset | Horizon frames | Horizon seconds | Frame rate |",
        "|---|---:|---:|---:|",
    ]
    for key, item in sorted((report.get("horizon_index") or {}).items(), key=lambda kv: kv[1].get("prediction_horizon_frames") or 0):
        lines.append(f"| `{key}` | {item.get('prediction_horizon_frames')} | {_fmt(item.get('prediction_horizon_sec'))} | {_fmt(item.get('frame_rate_hz'))} |")
    lines.extend(["", "## Top Shared-Horizon Candidates", "", "| Rank | Signature | Family | Target | Mean improve | Min improve | Long-short degradation | Horizons |", "|---:|---|---|---|---:|---:|---:|---|"])
    for idx, item in enumerate(report.get("top_candidates", []), start=1):
        horizons = ", ".join(f"{h.get('dataset_key')}={_fmt(h.get('improvement_over_persistence_mse'))}" for h in item.get("horizons", []))
        lines.append(
            f"| {idx} | `{item.get('signature_label')}` | {item.get('model_family')} | {item.get('prediction_target') or item.get('baseline_name') or 'n/a'} | {_fmt(item.get('mean_improvement_over_persistence_mse'))} | {_fmt(item.get('min_improvement_over_persistence_mse'))} | {_fmt(item.get('long_minus_short_improvement'))} | {horizons} |"
        )
    lines.extend(["", "## Family Summary", "", "| Family | Paired configs | Positive on all horizons | Best min improve |", "|---|---:|---:|---:|"])
    for row in report.get("family_summary", []):
        lines.append(f"| {row.get('model_family')} | {row.get('paired_config_count')} | {row.get('positive_all_horizon_count')} | {_fmt(row.get('best_min_improvement_over_persistence_mse'))} |")
    lines.extend(["", "## Planned Shared-Horizon Configs", ""])
    if report.get("planned_shared_horizon_configs"):
        lines.extend(["| ID | Family | Rationale | Source single-horizon rows |", "|---|---|---|---|"])
        for cfg in report.get("planned_shared_horizon_configs", []):
            lines.append(f"| `{cfg.get('config_id')}` | {cfg.get('model_family')} | {cfg.get('rationale')} | `{', '.join(cfg.get('source_experiment_ids', []))}` |")
    else:
        lines.append("No shared-horizon configs were planned from the current evidence.")
    lines.extend(["", "## Recommendations", ""])
    for rec in report.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.extend(["", "## Limitations", ""])
    for item in report.get("limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def _horizon_index(datasets: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = {str(row.get("dataset_key")) for row in rows if row.get("dataset_key")}
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        dataset = datasets.get(key, {}) if isinstance(datasets, Mapping) else {}
        windowing = dataset.get("windowing", {}) if isinstance(dataset, Mapping) and isinstance(dataset.get("windowing"), Mapping) else {}
        frames = _num(windowing.get("prediction_horizon_frames"))
        if frames is None:
            frames = _parse_horizon_frames(key)
        frame_rate = _num(windowing.get("effective_frame_rate_hz") or windowing.get("source_frame_rate_hz"))
        seconds = _num(windowing.get("prediction_horizon_sec"))
        if seconds is None and frames is not None and frame_rate:
            seconds = float(frames) / float(frame_rate)
        out[key] = {
            "dataset_key": key,
            "prediction_horizon_frames": int(frames) if frames is not None else None,
            "prediction_horizon_sec": seconds,
            "frame_rate_hz": frame_rate,
        }
    return out


def _paired_groups(rows: Sequence[Mapping[str, Any]], *, horizon_index: Mapping[str, Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _signature_key(row)
        buckets[key].append(row)
    groups = []
    for key, members in buckets.items():
        by_dataset: dict[str, Mapping[str, Any]] = {}
        for row in members:
            dataset_key = str(row.get("dataset_key"))
            if dataset_key not in horizon_index:
                continue
            current = by_dataset.get(dataset_key)
            if current is None or _metric(row, split, "improvement_over_persistence_mse") > _metric(current, split, "improvement_over_persistence_mse"):
                by_dataset[dataset_key] = row
        if len(by_dataset) < 2:
            continue
        horizons = []
        for dataset_key, row in sorted(by_dataset.items(), key=lambda item: horizon_index[item[0]].get("prediction_horizon_frames") or 0):
            horizons.append(
                {
                    "dataset_key": dataset_key,
                    "prediction_horizon_frames": horizon_index[dataset_key].get("prediction_horizon_frames"),
                    "prediction_horizon_sec": horizon_index[dataset_key].get("prediction_horizon_sec"),
                    "experiment_id": row.get("experiment_id"),
                    "decoded_prediction_mse": _metric(row, split, "decoded_prediction_mse"),
                    "persistence_mse": _metric(row, split, "persistence_mse"),
                    "improvement_over_persistence_mse": _metric(row, split, "improvement_over_persistence_mse"),
                }
            )
        imps = [h["improvement_over_persistence_mse"] for h in horizons if h.get("improvement_over_persistence_mse") is not None]
        short = horizons[0].get("improvement_over_persistence_mse") if horizons else None
        long = horizons[-1].get("improvement_over_persistence_mse") if horizons else None
        exemplar = dict(next(iter(by_dataset.values())))
        groups.append(
            {
                "signature_key": key,
                "signature_label": _signature_label(exemplar),
                "model_family": exemplar.get("model_family") or exemplar.get("kind"),
                "kind": exemplar.get("kind"),
                "prediction_target": exemplar.get("prediction_target"),
                "baseline_name": exemplar.get("baseline_name"),
                "hyperparameter_summary": _strip_dataset_text(str(exemplar.get("hyperparameter_summary") or "")),
                "params": _signature_params(exemplar),
                "horizons": horizons,
                "horizon_count": len(horizons),
                "mean_improvement_over_persistence_mse": float(sum(imps) / len(imps)) if imps else None,
                "min_improvement_over_persistence_mse": float(min(imps)) if imps else None,
                "max_improvement_over_persistence_mse": float(max(imps)) if imps else None,
                "long_minus_short_improvement": (float(long) - float(short)) if long is not None and short is not None else None,
                "positive_all_horizons": bool(imps and len(imps) == len(horizons) and all(v > 0 for v in imps)),
            }
        )
    return sorted(groups, key=lambda item: (_sort_num(item.get("min_improvement_over_persistence_mse")), _sort_num(item.get("mean_improvement_over_persistence_mse"))), reverse=True)


def _rank_candidates(groups: Sequence[Mapping[str, Any]], *, split: str, max_candidates: int) -> list[dict[str, Any]]:
    learned = [dict(g) for g in groups if str(g.get("model_family")) in LEARNED_FAMILIES]
    candidates = learned or [dict(g) for g in groups]
    candidates = sorted(candidates, key=lambda item: (_sort_num(item.get("min_improvement_over_persistence_mse")), _sort_num(item.get("mean_improvement_over_persistence_mse"))), reverse=True)
    return candidates[: max(int(max_candidates), 0)]


def _family_summary(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        by_family[str(group.get("model_family") or group.get("kind") or "unknown")].append(group)
    rows = []
    for family, items in sorted(by_family.items()):
        best_min = max((_sort_num(item.get("min_improvement_over_persistence_mse")) for item in items), default=float("-inf"))
        rows.append(
            {
                "model_family": family,
                "paired_config_count": len(items),
                "positive_all_horizon_count": sum(1 for item in items if item.get("positive_all_horizons")),
                "best_min_improvement_over_persistence_mse": None if best_min == float("-inf") else best_min,
            }
        )
    return rows


def _planned_shared_configs(candidates: Sequence[Mapping[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    planned = []
    for item in candidates:
        family = str(item.get("model_family") or item.get("kind") or "unknown")
        if family not in {"latent_gru", "latent_transformer", "linear_latent"}:
            continue
        params = dict(item.get("params") or {})
        cfg_id = "mh_" + _safe_slug(item.get("signature_label") or family)
        planned.append(
            {
                "config_id": cfg_id[:120],
                "model_family": family,
                "kind": item.get("kind"),
                "prediction_target": item.get("prediction_target"),
                "source_experiment_ids": [str(h.get("experiment_id")) for h in item.get("horizons", [])],
                "source_dataset_keys": [str(h.get("dataset_key")) for h in item.get("horizons", [])],
                "shared_horizons_frames": [h.get("prediction_horizon_frames") for h in item.get("horizons", [])],
                "recommended_params": params,
                "rationale": f"paired single-horizon min improvement {_fmt(item.get('min_improvement_over_persistence_mse'))}; long-short delta {_fmt(item.get('long_minus_short_improvement'))}",
            }
        )
        if len(planned) >= limit:
            break
    return planned


def _recommendations(candidates: Sequence[Mapping[str, Any]], family_summary: Sequence[Mapping[str, Any]]) -> list[str]:
    recs = []
    if not candidates:
        return ["No paired h2/h5-style learned candidates were available yet; regenerate after more sweep rows complete."]
    best = candidates[0]
    if not best.get("positive_all_horizons"):
        recs.append("Do not launch a shared multi-horizon learned model yet without stronger single-horizon evidence across both horizons.")
    else:
        recs.append(f"Use `{best.get('signature_label')}` as the first shared multi-horizon candidate because it is positive across the paired horizons.")
    recs.append("Report h2 and h5 metrics separately for any shared model, then compare its per-horizon improvement against the paired single-horizon rows in this report.")
    recs.append("Include persistence, moving-average, kinetics, and linear latent controls when presenting shared-horizon results.")
    return recs


def _signature_key(row: Mapping[str, Any]) -> str:
    return json.dumps(_signature_parts(row), sort_keys=True, separators=(",", ":"))


def _signature_parts(row: Mapping[str, Any]) -> dict[str, Any]:
    params = _signature_params(row)
    return {
        "model_family": row.get("model_family") or row.get("kind"),
        "kind": row.get("kind"),
        "prediction_target": row.get("prediction_target"),
        "baseline_name": row.get("baseline_name"),
        "params": params,
    }


def _signature_params(row: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(row.get("params") or {})
    keep = [
        "baseline_name",
        "prediction_target",
        "hidden_dim",
        "hidden_channels",
        "model_dim",
        "num_heads",
        "num_layers",
        "learning_rate",
        "residual_scale",
        "epochs",
        "batch_size",
        "loss_mode",
        "dropout",
        "alphas",
        "grid_size",
        "grid_pooling",
        "input_resolution",
        "model_label",
        "hyperparameter_group",
    ]
    out = {key: params.get(key) for key in keep if key in params}
    for key in ("hidden_dim", "hidden_channels", "model_dim", "num_heads", "num_layers", "learning_rate", "residual_scale", "epochs", "batch_size", "grid_size", "grid_pooling"):
        if key in row and row.get(key) not in (None, "") and key not in out:
            out[key] = row.get(key)
    return out


def _signature_label(row: Mapping[str, Any]) -> str:
    summary = _strip_dataset_text(str(row.get("hyperparameter_summary") or ""))
    if summary:
        return summary
    parts = _signature_parts(row)
    return ", ".join(f"{k}={v}" for k, v in parts.items() if v not in (None, "", {}))


def _strip_dataset_text(text: str) -> str:
    for token in ("w8_s1_h2", "w8_s1_h5"):
        text = text.replace(token, "w8_s1_h*")
    parts = []
    for part in text.split(","):
        stripped = part.strip()
        if stripped in {"h=2", "h=5"}:
            parts.append(part.replace(stripped, "h=*"))
        else:
            parts.append(part)
    return ",".join(parts)


def _parse_horizon_frames(dataset_key: str) -> int | None:
    marker = "_h"
    if marker not in dataset_key:
        return None
    tail = dataset_key.rsplit(marker, 1)[-1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _metric(row: Mapping[str, Any], split: str, metric_name: str) -> float | None:
    keys = [f"{split}_{metric_name}", metric_name] if split != "all" else [f"all_{metric_name}", metric_name]
    for key in keys:
        value = _num(row.get(key))
        if value is not None:
            return value
    return None


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _sort_num(value: Any) -> float:
    num = _num(value)
    return float(num) if num is not None else float("-inf")


def _fmt(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "n/a"
    return f"{num:.4g}"


def _safe_slug(value: Any) -> str:
    text = str(value or "config").lower()
    chars = []
    for ch in text:
        chars.append(ch if ch.isalnum() else "_")
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "config"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
