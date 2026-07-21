"""Adaptive second-stage planners for grid dynamics sweeps."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_BASELINES = {"persistence", "moving_average"}
LEARNED_FAMILIES = {"latent_gru", "latent_transformer", "linear_latent", "convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}
HEAVY_KINDS = {"convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel", "unet_convgru_pixel", "latent_gru", "latent_transformer"}


def build_adaptive_sweep_plan(
    *,
    sweep_dir: str | Path,
    comparison_dir: str | Path,
    out_dir: str | Path,
    max_experiments: int = 160,
    suggested_batch_size: int = 4,
) -> dict[str, Any]:
    """Create a smaller second-stage plan from partial first-stage evidence."""
    sweep = Path(sweep_dir)
    comparison = Path(comparison_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(sweep / "sweep_manifest.json")
    intelligence = _load_json(comparison / "results_intelligence.json")
    progress = _progress_summary(sweep)
    current_specs = [dict(item) for item in manifest.get("experiments", []) if isinstance(item, Mapping)]
    decisions = _family_decisions(intelligence)
    selected, selection_notes, deferred = _select_specs(
        specs=current_specs,
        intelligence=intelligence,
        max_experiments=int(max_experiments),
    )
    next_manifest = {
        "schema_version": 1,
        "manifest_kind": "adaptive_next_sweep_plan",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sweep_dir": str(sweep),
        "source_profile": manifest.get("profile"),
        "source_experiment_count": manifest.get("experiment_count"),
        "planned_experiment_count": len(selected),
        "max_experiments": int(max_experiments),
        "datasets": manifest.get("datasets", {}),
        "experiments": selected,
        "selection_notes": selection_notes,
        "family_decisions": decisions,
        "deferred_specs": deferred,
        "progress_summary": progress,
        "suggested_command": _suggested_command(out, suggested_batch_size=suggested_batch_size),
    }
    summary = {
        "schema_version": 1,
        "created_at": next_manifest["created_at"],
        "source_sweep_dir": str(sweep),
        "comparison_dir": str(comparison),
        "planned_experiment_count": len(selected),
        "family_decisions": decisions,
        "selection_counts": dict(Counter(str(spec.get("kind")) for spec in selected)),
        "dataset_counts": dict(Counter(str(spec.get("dataset_key")) for spec in selected)),
        "target_counts": dict(Counter(str((spec.get("params") or {}).get("prediction_target") or (spec.get("params") or {}).get("baseline_name") or "n/a") for spec in selected)),
        "deferred_counts": dict(Counter(str(spec.get("kind")) for spec in deferred)),
        "progress_summary": progress,
        "suggested_command": next_manifest["suggested_command"],
    }
    manifest_path = out / "next_sweep_manifest.json"
    summary_path = out / "next_sweep_plan.json"
    markdown_path = out / "next_sweep_plan.md"
    manifest_path.write_text(json.dumps(next_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    summary["summary_path"] = str(summary_path)
    summary["markdown_path"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_adaptive_sweep_plan_markdown(summary, next_manifest, intelligence), encoding="utf-8")
    return summary


def render_adaptive_sweep_plan_markdown(summary: Mapping[str, Any], manifest: Mapping[str, Any], intelligence: Mapping[str, Any]) -> str:
    lines = [
        "# Adaptive Next Sweep Plan",
        "",
        f"Generated: `{summary.get('created_at')}`",
        f"Source sweep: `{summary.get('source_sweep_dir')}`",
        f"Planned experiments: `{summary.get('planned_experiment_count')}`",
        f"Source progress: `{(summary.get('progress_summary') or {}).get('current_index', 0)}` / `{(summary.get('progress_summary') or {}).get('experiment_count', 0)}`",
        "",
        "## Stage B Objective",
        "",
        "Use the partial Stage A evidence to spend the next GPU window on models that are both biologically defensible and operationally stable: 128x128 max-pooled inputs, delta forecasting, horizons tied to 50 Hz sampling, and low learning rates around the best completed learned rows.",
        "",
        "## Rationale",
        "",
    ]
    lines.extend(f"- {note}" for note in manifest.get("selection_notes", []))
    lines.extend(["", "## Family Decisions", "", "| Family | Decision | Evidence |", "|---|---|---|"])
    for item in summary.get("family_decisions", []):
        lines.append(f"| {item.get('family')} | {item.get('decision')} | {item.get('evidence')} |")
    lines.extend(["", "## Selection Counts", "", _count_table("Kind", summary.get("selection_counts", {})), "", _count_table("Dataset", summary.get("dataset_counts", {})), "", _count_table("Target or baseline", summary.get("target_counts", {}))])
    if summary.get("deferred_counts"):
        lines.extend(["", "## Deferred Families", "", _count_table("Kind", summary.get("deferred_counts", {})), "", "These specs are not included in the default Stage B plan because the partial evidence says they are less likely to be productive than the conservative latent-search neighborhood."])
    lines.extend(["", "## Planned Experiment Examples", "", "| Experiment | Kind | Dataset | Reason | HParams |", "|---|---|---|---|---|"])
    for spec in list(manifest.get("experiments", []))[:20]:
        params = spec.get("params") or {}
        lines.append(f"| `{spec.get('experiment_id')}` | {spec.get('kind')} | {spec.get('dataset_key')} | {params.get('adaptive_stage_reason', '')} | `{params.get('hyperparameter_summary') or _compact_params(params)}` |")
    lines.extend(["", "## Current Best Test Rows", "", "| Family | Experiment | Test improve | HParams |", "|---|---|---:|---|"])
    for family, row in sorted(intelligence.get("best_by_family", {}).get("test", {}).items()):
        lines.append(f"| {family} | `{row.get('experiment_id')}` | {_fmt(row.get('improvement_over_persistence_mse'))} | `{row.get('hyperparameter_summary') or ''}` |")
    lines.extend(["", "## Suggested Command", "", "```bash", str(summary.get("suggested_command", "")), "```", "", "The generated manifest is executable through the manifest-aware overnight sweep runner. Use `--dry-run` first when validating a newly generated plan."])
    return "\n".join(lines).rstrip() + "\n"


def _family_decisions(intelligence: Mapping[str, Any]) -> list[dict[str, Any]]:
    family_stats = intelligence.get("family_comparison", {}).get("test", {})
    failures = intelligence.get("failure_summary", {}).get("by_kind", {})
    decisions: list[dict[str, Any]] = []
    for family in sorted(set(family_stats) | set(failures)):
        stats = family_stats.get(family, {})
        best = (stats.get("best") or {}).get("improvement_over_persistence_mse")
        fail_count = int(failures.get(family, 0) or 0)
        positive = int(stats.get("positive_count", 0) or 0)
        count = int(stats.get("count", 0) or 0)
        if family in {"array_baseline", "kinetics_baseline"}:
            decision = "keep as controls"
        elif positive > 0:
            decision = "shrink around positive configurations"
        elif fail_count > 0 and count == 0:
            decision = "defer or retry only at smaller batch size"
        else:
            decision = "drop unless needed for diversity"
        evidence = f"positive={positive}/{count}, best={_fmt(best)}, archived_failures={fail_count}"
        decisions.append({"family": family, "decision": decision, "evidence": evidence})
    return decisions


def _select_specs(*, specs: Sequence[Mapping[str, Any]], intelligence: Mapping[str, Any], max_experiments: int) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    notes: list[str] = []
    deferred: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(spec: Mapping[str, Any], reason: str) -> None:
        exp_id = str(spec.get("experiment_id"))
        if exp_id in seen or len(selected) >= max_experiments:
            return
        item = dict(spec)
        params = dict(item.get("params") or {})
        params["adaptive_stage_reason"] = reason
        if item.get("kind") in HEAVY_KINDS:
            params["recommended_batch_size"] = min(int(params.get("batch_size") or 4), 4)
        item["params"] = params
        selected.append(item)
        seen.add(exp_id)

    by_id = {str(spec.get("experiment_id")): spec for spec in specs}
    best_rows = intelligence.get("best_by_family", {}).get("test", {})
    best_dataset_keys = {str(row.get("dataset_key")) for row in best_rows.values() if row.get("dataset_key")}
    if not best_dataset_keys:
        best_dataset_keys = {str(spec.get("dataset_key")) for spec in specs}
    for spec in specs:
        params = spec.get("params", {}) if isinstance(spec.get("params"), Mapping) else {}
        if spec.get("kind") == "array_baseline" and params.get("baseline_name") in REQUIRED_BASELINES:
            add(spec, "required persistence/moving-average control")
    notes.append("Kept persistence and moving-average controls for each dataset.")

    for row in best_rows.values():
        spec = by_id.get(str(row.get("experiment_id")))
        if spec is not None:
            add(spec, "current best row for its family")
    notes.append("Seeded the next plan with current best rows by family, including any completed pixel scout with positive evidence.")

    for spec in specs:
        kind = str(spec.get("kind"))
        params = spec.get("params", {}) if isinstance(spec.get("params"), Mapping) else {}
        dataset = str(spec.get("dataset_key"))
        if dataset not in best_dataset_keys:
            continue
        if kind == "linear_latent" and params.get("prediction_target") == "delta":
            add(spec, "low-cost delta latent baseline")
        elif kind == "latent_gru" and params.get("prediction_target") == "delta" and params.get("hidden_dim") in {64, 128} and _lr(params) in {3e-5, 1e-4}:
            add(spec, "conservative delta GRU neighborhood")
        elif kind == "latent_transformer" and params.get("prediction_target") == "delta" and params.get("model_dim") == 64 and params.get("num_heads") in {2, 4} and params.get("num_layers") in {1, 2} and _lr(params) in {3e-5, 1e-4}:
            add(spec, "conservative delta Transformer neighborhood")
    notes.append("Focused learned search on delta targets, smaller hidden/model dimensions, and low learning rates that currently look most promising.")

    for spec in specs:
        kind = str(spec.get("kind"))
        params = spec.get("params", {}) if isinstance(spec.get("params"), Mapping) else {}
        if kind in {"convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}:
            item = dict(spec)
            dparams = dict(params)
            dparams["adaptive_stage_reason"] = "deferred: archived OOMs dominate this family in Stage A"
            dparams["recommended_batch_size_if_retried"] = min(int(dparams.get("batch_size") or 4), 2)
            item["params"] = dparams
            deferred.append(item)
    notes.append("Deferred broad pixel recurrent/CNN neighborhoods by default because archived OOMs dominate them; retain only the current best completed pixel scout unless a separate small-batch pixel plan is launched.")
    notes.append("Preserved diversity across horizons, seeds, array controls, linear latent controls, GRU, Transformer, and at most the current best pixel scout without spending Stage B on the full unstable pixel grid.")
    return selected, notes, deferred


def _progress_summary(sweep_dir: Path) -> dict[str, Any]:
    rows = _load_progress(sweep_dir / "sweep_progress.jsonl")
    archived = []
    for path in sorted(sweep_dir.glob("sweep_progress_*.jsonl")):
        archived.extend(_load_progress(path))
    all_failed = [row for row in rows + archived if row.get("status") == "failed"]
    last = rows[-1] if rows else {}
    return {
        "current_records": len(rows),
        "current_index": int(last.get("index") or 0),
        "experiment_count": int(last.get("experiment_count") or 0),
        "current_status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
        "archived_failure_count": len([row for row in archived if row.get("status") == "failed"]),
        "failure_count_including_archives": len(all_failed),
        "last_experiment_id": last.get("experiment_id"),
        "last_status": last.get("status"),
    }


def _load_progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _compact_params(params: Mapping[str, Any]) -> str:
    keep = ["baseline_name", "prediction_target", "hidden_dim", "hidden_channels", "model_dim", "num_heads", "num_layers", "learning_rate", "batch_size", "recommended_batch_size"]
    parts = []
    for key in keep:
        if key in params and params.get(key) is not None:
            parts.append(f"{key}={params.get(key)}")
    return ", ".join(parts)


def _suggested_command(out_dir: Path, *, suggested_batch_size: int) -> str:
    return (
        ".venv-neurobench/bin/python -m neurobench.dynamics.overnight_sweep "
        f"--manifest {out_dir / 'next_sweep_manifest.json'} "
        f"--out-dir {out_dir / 'stage_b_sweep'} "
        "--device cuda "
        f"--batch-size {int(suggested_batch_size)} --seeds 7,13 --time-limit-hours 48"
    )


def _lr(params: Mapping[str, Any]) -> float | None:
    value = params.get("learning_rate")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_table(label: str, counts: Mapping[str, Any]) -> str:
    if not counts:
        return "No rows selected."
    lines = [f"| {label} | Count |", "|---|---:|"]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
