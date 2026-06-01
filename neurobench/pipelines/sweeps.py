"""First-class parameter sweep execution and lightweight reporting."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from neurobench.architecture_runs import build_planned_manifest
from neurobench.manifests import write_json
from neurobench.pipelines.executor import execute_pipeline


def execute_parameter_sweep(
    spec: Mapping[str, Any],
    *,
    run_root: str | Path,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Expand a sweep spec, execute each run, and write sweep summary artifacts."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    planned = build_planned_manifest(spec)
    runs = []
    for index, planned_run in enumerate(planned.get("runs", []) or [], start=1):
        run_id = str(planned_run.get("run_id") or f"sweep_run_{index:03d}")
        run_dir = root / f"{index:03d}_{_safe_name(run_id)}"
        record = _base_record(planned_run, run_dir, root)
        try:
            result = execute_pipeline(planned_run, run_root=run_dir)
            pipeline_run = result["pipeline_run"]
            record.update(
                {
                    "status": "completed",
                    "artifact_count": len(pipeline_run.get("artifacts", [])),
                    "parameter_hash": pipeline_run.get("parameter_hash", ""),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                runs.append(record)
                break
        runs.append(record)

    failed = sum(1 for run in runs if run["status"] == "failed")
    succeeded = sum(1 for run in runs if run["status"] == "completed")
    summary = {
        "schema_version": 1,
        "dataset_id": planned.get("dataset_id", spec.get("dataset_id", "")),
        "sweep": planned.get("sweep", {}),
        "status": "completed" if failed == 0 else "completed_with_failures",
        "total": len(runs),
        "succeeded": succeeded,
        "failed": failed,
        "runs": runs,
    }
    write_json(root / "sweep_summary.json", summary)
    (root / "sweep_report.md").write_text(render_sweep_summary_markdown(summary), encoding="utf-8")
    return summary


def render_sweep_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render a sweep summary payload as a compact Markdown report."""
    sweep = dict(summary.get("sweep") or {})
    lines = [
        "# Neurobench Sweep Summary",
        "",
        f"- Dataset ID: `{summary.get('dataset_id', '')}`",
        f"- Sweep ID: `{sweep.get('id', 'sweep')}`",
        f"- Status: `{summary.get('status', '')}`",
        f"- Runs: {summary.get('succeeded', 0)} succeeded, {summary.get('failed', 0)} failed",
        "",
        "## Runs",
        "",
        "| Run | Status | Parameters | Artifacts | Metrics | Output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in summary.get("runs", []) or []:
        parameters = ", ".join(f"{item['stage']}.{item['param']}={item['value']}" for item in run.get("sweep_parameters", []) or [])
        metrics = _metric_summary(run.get("metrics") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run.get('run_id', '')}`",
                    f"`{run.get('status', '')}`",
                    parameters or "none",
                    str(run.get("artifact_count", 0)),
                    metrics or "n/a",
                    f"`{run.get('run_root', '')}`",
                ]
            )
            + " |"
        )
    overview = dict(summary.get("metric_overview") or {})
    if overview:
        lines.extend(
            [
                "",
                "## Metric Overview",
                "",
                f"- Best recall run: `{overview.get('best_recall_run_id', 'n/a')}`",
                f"- Lowest burden run: `{overview.get('lowest_burden_run_id', 'n/a')}`",
                f"- Mean candidate count: {_fmt_metric(overview.get('mean_candidate_count'))}",
                f"- Mean event count: {_fmt_metric(overview.get('mean_event_count'))}",
            ]
        )
    failures = [run for run in summary.get("runs", []) or [] if run.get("status") == "failed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for run in failures:
            lines.append(f"- `{run.get('run_id', '')}`: {run.get('error_type', 'Error')}: {run.get('error', '')}")
    return "\n".join(lines).rstrip() + "\n"


def _base_record(planned_run: Mapping[str, Any], run_dir: Path, root: Path) -> dict[str, Any]:
    sweep = dict(planned_run.get("sweep") or {})
    return {
        "run_id": str(planned_run.get("run_id", "")),
        "run_root": _display_path(run_dir, root),
        "status": "planned",
        "sweep_index": sweep.get("index"),
        "sweep_total": sweep.get("total_runs"),
        "sweep_parameters": list(sweep.get("parameters") or []),
    }


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return name or "run"


def _metric_summary(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return ""
    fields = []
    for key, label in (
        ("candidate_count", "candidates"),
        ("event_count", "events"),
        ("active_fraction", "active"),
        ("object_recall", "obj recall"),
        ("object_precision", "obj precision"),
        ("event_onset_recall", "event recall"),
    ):
        if key in metrics:
            fields.append(f"{label}={_fmt_metric(metrics.get(key))}")
    return ", ".join(fields)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
