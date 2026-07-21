"""Comparison manifests and static dashboards for grid dynamics sweeps."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neurobench.dynamics.overnight_sweep import collect_metric_rows
from neurobench.dynamics.supervisor import classify_failure


SPLIT_ORDER = {"test": 0, "val": 1, "validation": 1, "train": 2, "unknown": 3}


def build_comparison_dashboard(
    *,
    sweep_dirs: Sequence[str | Path],
    out_dir: str | Path,
    title: str = "Grid Dynamics Architecture Comparison",
    dashboard_prefix: str = "",
    selected_count: int = 3,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweeps = [Path(p) for p in sweep_dirs]
    rows: list[dict[str, Any]] = []
    video_collections: list[dict[str, Any]] = []
    dataset_index: dict[str, dict[str, Any]] = {}
    for sweep_dir in sweeps:
        manifest = _load_json(sweep_dir / "sweep_manifest.json") if (sweep_dir / "sweep_manifest.json").exists() else {}
        sweep_id = str(manifest.get("run_id") or manifest.get("profile") or sweep_dir.name)
        collection = _video_collection_from_selector(sweep_dir=sweep_dir, out_dir=out, sweep_id=sweep_id)
        if collection is not None:
            video_collections.append(collection)
        dataset_map = manifest.get("datasets", {}) if isinstance(manifest.get("datasets"), Mapping) else {}
        for dataset_key, cfg in dataset_map.items():
            dataset_index.setdefault(str(dataset_key), _dataset_record(str(dataset_key), cfg))
        for row in collect_metric_rows(sweep_dir):
            metrics_path = Path(str(row["metrics_path"]))
            metrics = _load_json(metrics_path)
            dataset_key = str(row.get("dataset_key", ""))
            cfg = dataset_map.get(dataset_key, {}) if isinstance(dataset_map, Mapping) else {}
            dataset_index.setdefault(dataset_key, _dataset_record(dataset_key, cfg))
            rows.append(_comparison_row(sweep_id=sweep_id, sweep_dir=sweep_dir, row=row, metrics=metrics))
    rows_sorted = sorted(rows, key=_row_sort_key)
    selected = [r for r in rows_sorted if r.get("kind") not in {"array_baseline"}][: int(selected_count)]
    videos = _ordered_videos(dataset_index)
    intelligence = build_results_intelligence(rows=rows_sorted, sweep_dirs=sweeps)
    payload = {
        "schema_version": 1,
        "title": str(title),
        "sweep_dirs": [str(p) for p in sweeps],
        "row_count": len(rows_sorted),
        "rows": rows_sorted,
        "intelligence": intelligence,
        "selected_models": selected,
        "datasets": dataset_index,
        "input_videos": videos,
        "video_collections": video_collections,
        "dashboard_prefix": dashboard_prefix,
    }
    manifest_path = out / "comparison_manifest.json"
    html_path = out / "comparison_dashboard.html"
    intelligence_path = out / "results_intelligence.json"
    intelligence_md_path = out / "results_intelligence.md"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    intelligence_path.write_text(json.dumps(intelligence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    intelligence_md_path.write_text(render_results_intelligence_markdown(intelligence), encoding="utf-8")
    html_path.write_text(_comparison_html(payload), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "title": str(title),
        "row_count": len(rows_sorted),
        "selected_model_ids": [r["row_id"] for r in selected],
        "video_collection_count": len(video_collections),
        "failure_count": intelligence.get("failure_summary", {}).get("failure_count", 0),
        "positive_test_count": intelligence.get("improvement_distribution", {}).get("test", {}).get("positive_count", 0),
        "manifest_path": str(manifest_path),
        "html_path": str(html_path),
        "intelligence_path": str(intelligence_path),
        "intelligence_md_path": str(intelligence_md_path),
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary



def build_results_intelligence(*, rows: Sequence[Mapping[str, Any]], sweep_dirs: Sequence[str | Path]) -> dict[str, Any]:
    """Summarize completed and failed sweep configurations for dashboard review."""
    row_list = [dict(row) for row in rows]
    failures = _collect_failed_progress_records(sweep_dirs)
    splits = ("test", "val", "all")
    return {
        "schema_version": 1,
        "completed_count": len(row_list),
        "leaderboards": {split: _leaderboard(row_list, split=split, limit=10) for split in splits},
        "best_by_family": {split: _best_grouped(row_list, split=split, group_key="model_family") for split in splits},
        "best_by_horizon": {split: _best_grouped(row_list, split=split, group_key="dataset_key") for split in splits},
        "best_by_target": {split: _best_grouped(row_list, split=split, group_key="prediction_target") for split in splits},
        "target_comparison": {split: _group_stats(row_list, split=split, group_key="prediction_target") for split in splits},
        "family_comparison": {split: _group_stats(row_list, split=split, group_key="model_family") for split in splits},
        "improvement_distribution": {split: _improvement_distribution(row_list, split=split) for split in splits},
        "runtime_summary": _runtime_summary(row_list),
        "failures": failures[:200],
        "failure_summary": _failure_dashboard_summary(failures),
    }


def render_results_intelligence_markdown(intelligence: Mapping[str, Any]) -> str:
    """Render a compact professor-facing sweep intelligence report."""
    lines = [
        "# Results Intelligence",
        "",
        f"Completed metric rows: `{intelligence.get('completed_count', 0)}`",
        f"Failed configurations represented: `{intelligence.get('failure_summary', {}).get('failure_count', 0)}`",
        "",
        "## Top Test Models",
        "",
        _markdown_model_table(intelligence.get("leaderboards", {}).get("test", []), split="test"),
        "",
        "## Best Test Model Per Family",
        "",
        _markdown_model_table(intelligence.get("best_by_family", {}).get("test", {}).values(), split="test"),
        "",
        "## Best Test Model Per Horizon",
        "",
        _markdown_model_table(intelligence.get("best_by_horizon", {}).get("test", {}).values(), split="test"),
        "",
        "## Delta vs Absolute Target",
        "",
        _markdown_group_table(intelligence.get("target_comparison", {}).get("test", {}), split="test"),
        "",
        "## Runtime Summary",
        "",
        _markdown_runtime_summary(intelligence.get("runtime_summary", {})),
        "",
        "## Failure Summary",
        "",
    ]
    failure_summary = intelligence.get("failure_summary", {})
    if failure_summary.get("failure_count"):
        lines.append(_markdown_count_table("Failure class", failure_summary.get("by_class", {})))
        lines.extend(["", "By model kind:", "", _markdown_count_table("Kind", failure_summary.get("by_kind", {}))])
    else:
        lines.append("No failed configurations were found in current or archived progress logs.")
    return "\n".join(lines).rstrip() + "\n"

def _video_collection_from_selector(*, sweep_dir: Path, out_dir: Path, sweep_id: str) -> dict[str, Any] | None:
    candidates = [
        sweep_dir / "visuals" / "charts" / "original_vs_reconstruction_selector.json",
        sweep_dir / "charts" / "original_vs_reconstruction_selector.json",
    ]
    selector_path = next((path for path in candidates if path.exists()), None)
    if selector_path is None:
        return None
    selector = _load_json(selector_path)
    source_charts = selector_path.parent
    dashboard_charts = out_dir.parent / sweep_dir.name / "charts"
    asset_dir = dashboard_charts if dashboard_charts.exists() else source_charts
    asset_base = Path(os.path.relpath(asset_dir, out_dir)).as_posix()
    options: list[dict[str, Any]] = []
    for item in selector.get("options", []):
        if not isinstance(item, Mapping):
            continue
        option = dict(item)
        intensity = str(option.get("intensity_file") or "")
        motion = str(option.get("motion_file") or "")
        if intensity:
            option["intensity_src"] = _url_join(asset_base, intensity)
        if motion:
            option["motion_src"] = _url_join(asset_base, motion)
        options.append(option)
    return {
        "sweep_id": str(sweep_id),
        "sweep_dir": str(sweep_dir),
        "selector_path": str(selector_path),
        "asset_base": asset_base,
        "label": f"{sweep_dir.name} forecast clips",
        "panel_order": selector.get("panel_order", []),
        "segment_selection": selector.get("segment_selection", ""),
        "models": selector.get("models", []),
        "options": options,
    }


def _url_join(base: str, file_name: str) -> str:
    return (Path(base) / file_name).as_posix()


def _comparison_row(*, sweep_id: str, sweep_dir: Path, row: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(row.get("params") or {})
    family = str(metrics.get("model_family") or row.get("model_family") or row.get("kind") or "unknown")
    primary_split = "test" if metrics.get("test_decoded_prediction_mse") is not None else "val"
    primary_improvement = _num(metrics.get(f"{primary_split}_improvement_over_persistence_mse"))
    if primary_improvement is None:
        primary_improvement = _num(metrics.get("improvement_over_persistence_mse"))
        primary_split = "all"
    experiment_id = str(row.get("experiment_id"))
    return {
        "row_id": f"{sweep_id}:{experiment_id}",
        "sweep_id": str(sweep_id),
        "sweep_dir": str(sweep_dir),
        "experiment_id": experiment_id,
        "kind": str(row.get("kind", "")),
        "model_family": family,
        "model_kind": str(metrics.get("model_kind") or row.get("model_kind") or row.get("kind") or ""),
        "dataset_key": str(row.get("dataset_key", "")),
        "seed": int(row.get("seed") or 0),
        "objective": str(metrics.get("objective") or row.get("objective") or ""),
        "loss_mode": metrics.get("loss_mode") or row.get("loss_mode") or params.get("loss_mode"),
        "baseline_name": metrics.get("baseline_name") or params.get("baseline_name"),
        "prediction_target": metrics.get("prediction_target") or params.get("prediction_target"),
        "hyperparameter_group": row.get("hyperparameter_group") or params.get("hyperparameter_group"),
        "hyperparameter_summary": row.get("hyperparameter_summary") or params.get("hyperparameter_summary") or _compact_params(params),
        "grid_size": row.get("grid_size") or params.get("grid_size"),
        "grid_pooling": row.get("grid_pooling") or params.get("grid_pooling"),
        "primary_split": primary_split,
        "primary_improvement_over_persistence_mse": primary_improvement,
        "val_decoded_prediction_mse": _num(metrics.get("val_decoded_prediction_mse")),
        "val_persistence_mse": _num(metrics.get("val_persistence_mse")),
        "val_improvement_over_persistence_mse": _num(metrics.get("val_improvement_over_persistence_mse")),
        "test_decoded_prediction_mse": _num(metrics.get("test_decoded_prediction_mse")),
        "test_persistence_mse": _num(metrics.get("test_persistence_mse")),
        "test_improvement_over_persistence_mse": _num(metrics.get("test_improvement_over_persistence_mse")),
        "test_active_cell_improvement_over_persistence_mse": _num(metrics.get("test_active_cell_improvement_over_persistence_mse")),
        "test_active_cell_mse": _num(metrics.get("test_active_cell_mse")),
        "test_active_cell_persistence_mse": _num(metrics.get("test_active_cell_persistence_mse")),
        "test_top_activity_improvement_over_persistence_mse": _num(metrics.get("test_top_activity_improvement_over_persistence_mse")),
        "test_high_change_improvement_over_persistence_mse": _num(metrics.get("test_high_change_improvement_over_persistence_mse")),
        "test_active_cell_fraction": _num(metrics.get("test_active_cell_fraction")),
        "all_decoded_prediction_mse": _num(metrics.get("decoded_prediction_mse")),
        "all_persistence_mse": _num(metrics.get("persistence_mse")),
        "all_improvement_over_persistence_mse": _num(metrics.get("improvement_over_persistence_mse")),
        "prediction_examples_path": metrics.get("prediction_examples_path"),
        "prediction_clip_examples_path": metrics.get("prediction_clip_examples_path"),
        "video_error_summary": _video_error_summary(metrics),
        "elapsed_seconds": _num(row.get("elapsed_seconds")),
        "progress_index": row.get("progress_index"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "metrics_path": str(row.get("metrics_path", "")),
        "params": params,
    }


def _compact_params(params: Mapping[str, Any]) -> str:
    keys = (
        "baseline_name",
        "architecture",
        "loss_mode",
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
        "grid_size",
        "grid_pooling",
    )
    parts: list[str] = []
    for key in keys:
        value = params.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _dataset_record(dataset_key: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset_path = Path(str(cfg.get("dataset", ""))) if cfg.get("dataset") else None
    record: dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_path": str(dataset_path) if dataset_path else "",
        "autoencoder_run": str(cfg.get("autoencoder_run", "")),
        "window_frames": cfg.get("window_frames"),
        "windowing": dict(cfg.get("windowing", {})) if isinstance(cfg.get("windowing"), Mapping) else {},
        "splits": dict(cfg.get("splits", {})) if isinstance(cfg.get("splits"), Mapping) else {},
        "videos": [],
    }
    if not dataset_path or not dataset_path.exists():
        return record
    dataset = _load_json(dataset_path)
    record["windowing"] = dataset.get("windowing", record.get("windowing", {}))
    record["splits"] = dataset.get("splits", record.get("splits", {}))
    array_path = Path(str(dataset.get("array_path", "")))
    if array_path.exists():
        record["videos"] = _videos_from_arrays(array_path, dataset.get("splits", {}), dataset_key)
    return record


def _videos_from_arrays(array_path: Path, splits: Mapping[str, Any], dataset_key: str) -> list[dict[str, Any]]:
    with np.load(array_path, allow_pickle=False) as arrays:
        ids = arrays["window_video_ids"].astype(str)
        labels = arrays["window_labels"].astype(str) if "window_labels" in arrays else np.asarray([""] * len(ids), dtype=str)
    label_by_id: dict[str, str] = {}
    count_by_id: dict[str, int] = {}
    for vid, label in zip(ids.tolist(), labels.tolist()):
        label_by_id.setdefault(str(vid), str(label))
        count_by_id[str(vid)] = count_by_id.get(str(vid), 0) + 1
    videos = []
    for video_id in sorted(label_by_id):
        videos.append(
            {
                "dataset_key": dataset_key,
                "video_id": video_id,
                "label": label_by_id[video_id],
                "split": _split_for_video(video_id, splits),
                "window_count": count_by_id.get(video_id, 0),
            }
        )
    return sorted(videos, key=lambda item: (SPLIT_ORDER.get(str(item["split"]), 9), str(item["video_id"])))


def _split_for_video(video_id: str, splits: Mapping[str, Any] | None) -> str:
    if not isinstance(splits, Mapping):
        return "unknown"
    for split in ("test", "val", "train"):
        candidates = [split, f"{split}_video_ids", f"{split}_videos"]
        for key in candidates:
            value = splits.get(key)
            if isinstance(value, Mapping):
                nested = value.get("video_ids") or value.get("videos") or value.get("ids")
                if nested and str(video_id) in {str(v) for v in nested}:
                    return split
            elif isinstance(value, (list, tuple, set)) and str(video_id) in {str(v) for v in value}:
                return split
    assignments = splits.get("assignments")
    if isinstance(assignments, Mapping):
        return str(assignments.get(str(video_id), "unknown"))
    return "unknown"


def _ordered_videos(dataset_index: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    videos: list[dict[str, Any]] = []
    for dataset_key, dataset in dataset_index.items():
        for video in dataset.get("videos", []):
            key = (str(dataset_key), str(video.get("video_id")))
            if key in seen:
                continue
            seen.add(key)
            videos.append(dict(video))
    return sorted(videos, key=lambda item: (SPLIT_ORDER.get(str(item.get("split")), 9), str(item.get("dataset_key")), str(item.get("video_id"))))


def _row_sort_key(row: Mapping[str, Any]):
    test = row.get("test_improvement_over_persistence_mse")
    val = row.get("val_improvement_over_persistence_mse")
    primary = row.get("primary_improvement_over_persistence_mse")
    return (-_finite_or_floor(test), -_finite_or_floor(val), -_finite_or_floor(primary), str(row.get("experiment_id", "")))


def _finite_or_floor(value: Any) -> float:
    number = _num(value)
    return number if number is not None else -1e9


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number



def _runtime_summary(rows: Sequence[Mapping[str, Any]], *, slowest_limit: int = 10) -> dict[str, Any]:
    timed = [row for row in rows if _num(row.get("elapsed_seconds")) is not None]
    if not timed:
        return {"available": False, "row_count": 0, "by_family": {}, "slowest_rows": []}
    values = [float(_num(row.get("elapsed_seconds"))) for row in timed]
    by_family: dict[str, list[float]] = {}
    rows_by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in timed:
        family = str(row.get("model_family") or row.get("kind") or "unknown")
        elapsed = float(_num(row.get("elapsed_seconds")))
        by_family.setdefault(family, []).append(elapsed)
        rows_by_family.setdefault(family, []).append(row)
    family_summary = {}
    for family, family_values in sorted(by_family.items()):
        slowest = sorted(rows_by_family[family], key=lambda row: _finite_or_floor(row.get("elapsed_seconds")), reverse=True)[:3]
        family_summary[family] = {
            "count": len(family_values),
            "mean_seconds": float(np.mean(family_values)),
            "median_seconds": float(np.median(family_values)),
            "max_seconds": float(np.max(family_values)),
            "slowest_rows": [_runtime_row_card(row) for row in slowest],
        }
    slowest_rows = sorted(timed, key=lambda row: _finite_or_floor(row.get("elapsed_seconds")), reverse=True)[: int(slowest_limit)]
    return {
        "available": True,
        "row_count": len(timed),
        "total_seconds": float(np.sum(values)),
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "max_seconds": float(np.max(values)),
        "by_family": family_summary,
        "slowest_rows": [_runtime_row_card(row) for row in slowest_rows],
    }


def _runtime_row_card(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": str(row.get("experiment_id", "")),
        "model_family": str(row.get("model_family") or row.get("kind") or ""),
        "dataset_key": str(row.get("dataset_key", "")),
        "elapsed_seconds": _num(row.get("elapsed_seconds")),
        "hyperparameter_summary": row.get("hyperparameter_summary"),
    }


def _video_error_summary(metrics: Mapping[str, Any], *, limit: int = 5) -> dict[str, Any]:
    split_metrics = metrics.get("split_metrics", {})
    if not isinstance(split_metrics, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for split_name in ("test", "val", "train", "all"):
        split = split_metrics.get(split_name, {})
        if not isinstance(split, Mapping):
            continue
        per_video = split.get("per_video", {})
        if not isinstance(per_video, Mapping) or not per_video:
            continue
        rows = [_video_metric_card(video_id, record) for video_id, record in per_video.items() if isinstance(record, Mapping)]
        rows = [row for row in rows if row.get("improvement_over_persistence_mse") is not None]
        if not rows:
            continue
        ranked = sorted(rows, key=lambda row: (-_finite_or_floor(row.get("improvement_over_persistence_mse")), str(row.get("video_id", ""))))
        summary[split_name] = {
            "video_count": len(rows),
            "best_videos": ranked[: int(limit)],
            "worst_videos": list(reversed(ranked[-int(limit) :])),
            "label_summary": _video_label_summary(rows),
        }
    return summary


def _video_metric_card(video_id: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    video_id_text = str(video_id)
    return {
        "video_id": video_id_text,
        "video_label": _video_label(video_id_text),
        "window_count": int(record.get("window_count") or 0),
        "decoded_prediction_mse": _num(record.get("decoded_prediction_mse")),
        "persistence_mse": _num(record.get("persistence_mse")),
        "improvement_over_persistence_mse": _num(record.get("improvement_over_persistence_mse")),
    }


def _video_label_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("video_label") or "unknown")
        group = groups.setdefault(label, {"video_count": 0, "window_count": 0, "decoded_prediction_mse": [], "persistence_mse": [], "improvement_over_persistence_mse": []})
        group["video_count"] += 1
        windows = max(int(row.get("window_count") or 0), 1)
        group["window_count"] += int(row.get("window_count") or 0)
        for key in ("decoded_prediction_mse", "persistence_mse", "improvement_over_persistence_mse"):
            value = _num(row.get(key))
            if value is not None:
                group[key].append((float(value), windows))
    summary = []
    for label, group in groups.items():
        summary.append(
            {
                "label": label,
                "video_count": int(group["video_count"]),
                "window_count": int(group["window_count"]),
                "decoded_prediction_mse": _weighted_mean(group["decoded_prediction_mse"]),
                "persistence_mse": _weighted_mean(group["persistence_mse"]),
                "improvement_over_persistence_mse": _weighted_mean(group["improvement_over_persistence_mse"]),
            }
        )
    return sorted(summary, key=lambda item: (-_finite_or_floor(item.get("improvement_over_persistence_mse")), str(item.get("label", ""))))


def _weighted_mean(values: Sequence[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return float(sum(value * weight for value, weight in values) / total_weight)


def _video_label(video_id: str) -> str:
    tokens = str(video_id).lower().replace("_", " ").replace("-", " ").split()
    for label in ("left", "right", "neutral"):
        if label in tokens:
            return label
    return "unknown"


def _leaderboard(rows: Sequence[Mapping[str, Any]], *, split: str, limit: int) -> list[dict[str, Any]]:
    ranked = [row for row in rows if _split_improvement(row, split) is not None]
    ranked.sort(key=lambda row: (-_finite_or_floor(_split_improvement(row, split)), str(row.get("experiment_id", ""))))
    return [_model_card(row, split=split) for row in ranked[: int(limit)]]


def _best_grouped(rows: Sequence[Mapping[str, Any]], *, split: str, group_key: str) -> dict[str, dict[str, Any]]:
    best: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        improve = _split_improvement(row, split)
        if improve is None:
            continue
        group = _group_value(row, group_key)
        if group not in best or _finite_or_floor(improve) > _finite_or_floor(_split_improvement(best[group], split)):
            best[group] = row
    return {group: _model_card(row, split=split) for group, row in sorted(best.items())}


def _group_stats(rows: Sequence[Mapping[str, Any]], *, split: str, group_key: str) -> dict[str, dict[str, Any]]:
    values_by_group: dict[str, list[float]] = {}
    rows_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        improve = _split_improvement(row, split)
        if improve is None:
            continue
        group = _group_value(row, group_key)
        values_by_group.setdefault(group, []).append(float(improve))
        rows_by_group.setdefault(group, []).append(row)
    stats: dict[str, dict[str, Any]] = {}
    for group in sorted(values_by_group):
        values = values_by_group[group]
        best = _leaderboard(rows_by_group[group], split=split, limit=1)
        stats[group] = {
            "count": len(values),
            "positive_count": sum(1 for value in values if value > 0),
            "mean_improvement": float(np.mean(values)) if values else None,
            "median_improvement": float(np.median(values)) if values else None,
            "best": best[0] if best else None,
        }
    return stats


def _improvement_distribution(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    values = [float(value) for row in rows if (value := _split_improvement(row, split)) is not None]
    if not values:
        return {"count": 0, "positive_count": 0, "negative_count": 0, "zero_count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "positive_count": sum(1 for value in values if value > 0),
        "negative_count": sum(1 for value in values if value < 0),
        "zero_count": sum(1 for value in values if value == 0),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _model_card(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    return {
        "row_id": str(row.get("row_id", "")),
        "experiment_id": str(row.get("experiment_id", "")),
        "sweep_id": str(row.get("sweep_id", "")),
        "kind": str(row.get("kind", "")),
        "model_family": str(row.get("model_family", "")),
        "dataset_key": str(row.get("dataset_key", "")),
        "seed": row.get("seed"),
        "split": split,
        "decoded_prediction_mse": _split_metric(row, split, "decoded_prediction_mse"),
        "persistence_mse": _split_metric(row, split, "persistence_mse"),
        "improvement_over_persistence_mse": _split_improvement(row, split),
        "loss_mode": row.get("loss_mode"),
        "baseline_name": row.get("baseline_name"),
        "prediction_target": row.get("prediction_target") or row.get("baseline_name") or "unspecified",
        "hyperparameter_group": row.get("hyperparameter_group"),
        "hyperparameter_summary": row.get("hyperparameter_summary"),
        "video_error_summary": row.get("video_error_summary"),
        "elapsed_seconds": _num(row.get("elapsed_seconds")),
        "metrics_path": row.get("metrics_path"),
    }


def _split_metric(row: Mapping[str, Any], split: str, metric_name: str) -> float | None:
    return _num(row.get(f"{split}_{metric_name}")) if split in {"test", "val", "all"} else None


def _split_improvement(row: Mapping[str, Any], split: str) -> float | None:
    return _split_metric(row, split, "improvement_over_persistence_mse")


def _group_value(row: Mapping[str, Any], group_key: str) -> str:
    value = row.get(group_key)
    if value is None or value == "":
        if group_key == "prediction_target":
            value = row.get("baseline_name") or "unspecified"
        else:
            value = "unknown"
    return str(value)


def _collect_failed_progress_records(sweep_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for sweep_dir in sweep_dirs:
        sweep = Path(sweep_dir)
        manifest = _load_json(sweep / "sweep_manifest.json") if (sweep / "sweep_manifest.json").exists() else {}
        sweep_id = str(manifest.get("run_id") or manifest.get("profile") or sweep.name)
        spec_index = _spec_index(manifest)
        for progress_path in _progress_paths(sweep):
            source = "current" if progress_path.name == "sweep_progress.jsonl" else "archive"
            for record in _load_jsonl(progress_path):
                if record.get("status") != "failed":
                    continue
                exp_id = str(record.get("experiment_id", ""))
                spec = spec_index.get(exp_id, {})
                params = spec.get("params", {}) if isinstance(spec.get("params"), Mapping) else {}
                failures.append(
                    {
                        "sweep_id": sweep_id,
                        "source": source,
                        "progress_file": str(progress_path),
                        "index": record.get("index"),
                        "experiment_count": record.get("experiment_count"),
                        "experiment_id": exp_id,
                        "kind": str(record.get("kind") or spec.get("kind") or "unknown"),
                        "dataset_key": str(record.get("dataset_key") or spec.get("dataset_key") or "unknown"),
                        "seed": record.get("seed") or spec.get("seed"),
                        "architecture": params.get("architecture") or record.get("kind") or spec.get("kind") or "unknown",
                        "prediction_target": params.get("prediction_target") or params.get("baseline_name") or "unspecified",
                        "hyperparameter_group": params.get("hyperparameter_group"),
                        "hyperparameter_summary": params.get("hyperparameter_summary") or _compact_params(params),
                        "failure_class": classify_failure(record.get("error", "")),
                        "error_excerpt": str(record.get("error", ""))[:500],
                    }
                )
    failures.sort(key=lambda item: (str(item.get("source")), str(item.get("progress_file")), int(item.get("index") or 0), str(item.get("experiment_id"))))
    return failures


def _failure_dashboard_summary(failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    by_architecture: dict[str, int] = {}
    examples: dict[str, dict[str, Any]] = {}
    for failure in failures:
        for key, bucket in (("failure_class", by_class), ("kind", by_kind), ("dataset_key", by_dataset), ("architecture", by_architecture)):
            value = str(failure.get(key) or "unknown")
            bucket[value] = bucket.get(value, 0) + 1
        cls = str(failure.get("failure_class") or "unknown")
        examples.setdefault(cls, dict(failure))
    return {
        "failure_count": len(failures),
        "by_class": dict(sorted(by_class.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_architecture": dict(sorted(by_architecture.items())),
        "examples": examples,
    }


def _progress_paths(sweep: Path) -> list[Path]:
    paths = []
    current = sweep / "sweep_progress.jsonl"
    if current.exists():
        paths.append(current)
    paths.extend(sorted(path for path in sweep.glob("sweep_progress_*.jsonl") if path.is_file()))
    return paths


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _spec_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    experiments = manifest.get("experiments", []) if isinstance(manifest, Mapping) else []
    if not isinstance(experiments, Sequence):
        return {}
    return {str(item.get("experiment_id")): dict(item) for item in experiments if isinstance(item, Mapping) and item.get("experiment_id")}


def _markdown_model_table(rows: Iterable[Mapping[str, Any]], *, split: str) -> str:
    row_list = list(rows)
    if not row_list:
        return "No completed metric rows for this split."
    lines = [
        "| Experiment | Family | Dataset | Target | Improvement | HParams |",
        "|---|---|---|---|---:|---|",
    ]
    for row in row_list:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('experiment_id', '')}`",
                    str(row.get("model_family", "")),
                    str(row.get("dataset_key", "")),
                    str(row.get("prediction_target") or row.get("baseline_name") or ""),
                    _fmt(row.get("improvement_over_persistence_mse")),
                    f"`{row.get('hyperparameter_summary') or ''}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _markdown_group_table(groups: Mapping[str, Mapping[str, Any]], *, split: str) -> str:
    if not groups:
        return "No grouped metric rows for this split."
    lines = ["| Group | Count | Positive | Mean improve | Best experiment | Best improve |", "|---|---:|---:|---:|---|---:|"]
    for name, stats in sorted(groups.items()):
        best = stats.get("best") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(name),
                    str(stats.get("count", 0)),
                    str(stats.get("positive_count", 0)),
                    _fmt(stats.get("mean_improvement")),
                    f"`{best.get('experiment_id', '')}`" if best else "",
                    _fmt(best.get("improvement_over_persistence_mse")) if best else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _markdown_runtime_summary(summary: Mapping[str, Any]) -> str:
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


def _markdown_count_table(label: str, counts: Mapping[str, Any]) -> str:
    if not counts:
        return "No counts."
    lines = [f"| {label} | Count |", "|---|---:|"]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.4g}"

def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _comparison_html(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    title = html.escape(str(payload.get("title", "Grid Dynamics Architecture Comparison")))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d8dee8; --panel:#f7f9fc; --accent:#0b6bcb; --good:#087443; --bad:#b42318; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#fff; }}
header {{ padding:24px 28px 18px; border-bottom:1px solid var(--line); background:#fbfcfe; }}
h1 {{ margin:0 0 8px; font-size:24px; line-height:1.2; letter-spacing:0; }}
.sub {{ color:var(--muted); font-size:14px; }}
main {{ padding:20px 28px 28px; }}
.controls {{ display:grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap:10px; margin-bottom:16px; }}
label {{ display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--muted); }}
select, input {{ min-height:34px; border:1px solid var(--line); border-radius:6px; padding:6px 8px; background:#fff; color:var(--ink); font-size:14px; }}
.video-panel {{ border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:16px; background:#fff; }}
.video-head {{ display:flex; justify-content:space-between; align-items:start; gap:12px; margin-bottom:12px; }}
.video-head h2 {{ margin:0 0 4px; font-size:17px; }}
.video-controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:end; margin-bottom:12px; }}
.video-controls label {{ min-width:150px; }}
.video-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }}
.video-card {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fbfcfe; }}
.video-card header {{ padding:9px 10px; border-bottom:1px solid var(--line); background:#fff; }}
.video-card h3 {{ margin:0; font-size:13px; line-height:1.25; }}
.video-card video {{ display:block; width:100%; background:#111; aspect-ratio:16/9; }}
.video-card .meta-line {{ padding:8px 10px; color:var(--muted); font-size:12px; line-height:1.35; }}
.summary {{ display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap:10px; margin-bottom:16px; }}
.metric {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--panel); }}
.intelligence {{ border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:16px; background:#fff; }}
.intelligence h2 {{ margin:0 0 4px; font-size:18px; }}
.intel-grid {{ display:grid; grid-template-columns: repeat(6, minmax(170px, 1fr)); gap:10px; margin-top:12px; }}
.intel-card {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; min-width:0; }}
.intel-card h3 {{ margin:0 0 8px; font-size:13px; color:#344054; }}
.intel-item {{ border-top:1px solid var(--line); padding:7px 0; }}
.intel-item:first-child {{ border-top:0; padding-top:0; }}
.intel-item .title {{ font-size:12px; line-height:1.3; overflow-wrap:anywhere; }}
.intel-item .meta {{ color:var(--muted); font-size:11px; line-height:1.35; margin-top:2px; }}
.metric b {{ display:block; font-size:19px; margin-bottom:2px; }}
.metric span {{ color:var(--muted); font-size:12px; }}
.compare {{ display:grid; grid-template-columns: minmax(0, 1fr) 340px; gap:16px; align-items:start; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 7px; text-align:left; vertical-align:top; overflow:hidden; text-overflow:ellipsis; }}
th {{ position:sticky; top:0; background:#fff; z-index:1; color:#344054; font-size:12px; }}
tr.selected {{ background:#eef6ff; }}
button {{ min-height:30px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.side {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; position:sticky; top:12px; }}
.side h2 {{ font-size:16px; margin:0 0 10px; }}
.slot {{ border-top:1px solid var(--line); padding:9px 0; }}
.slot:first-of-type {{ border-top:0; }}
.small {{ color:var(--muted); font-size:12px; line-height:1.35; }}
.good {{ color:var(--good); font-variant-numeric:tabular-nums; }}
.bad {{ color:var(--bad); font-variant-numeric:tabular-nums; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
@media (max-width: 1180px) {{ .intel-grid {{ grid-template-columns: repeat(2, minmax(180px, 1fr)); }} }}
@media (max-width: 980px) {{ .controls, .summary, .compare, .intel-grid {{ grid-template-columns:1fr; }} .side {{ position:static; }} .video-head {{ display:block; }} }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">Held-out-first comparison of model families, losses, horizons, seeds, and input videos. Select up to three model rows for side-by-side review metadata.</div>
</header>
<main>
  <div class="controls">
    <label>Model family<select id="familyFilter"></select></label>
    <label>Dataset / horizon<select id="datasetFilter"></select></label>
    <label>Metric split<select id="splitFilter"><option value="test">Test</option><option value="val">Validation</option><option value="all">All</option></select></label>
    <label>Input video<select id="inputVideoFilter"></select></label>
    <label>Search<input id="searchBox" type="search" placeholder="experiment, loss, target"></label>
  </div>
  <section class="video-panel" id="videoPanel">
    <div class="video-head">
      <div>
        <h2>Video Comparison</h2>
        <div class="small">Synchronized pre-rendered clips. Each clip contains target, model output, persistence, and absolute error panels.</div>
      </div>
      <div class="small" id="clipStatus"></div>
    </div>
    <div class="video-controls">
      <label>Clip set<select id="clipSet"></select></label>
      <label>Input video<select id="clipInput"></select></label>
      <label>View<select id="clipView"><option value="intensity">Intensity</option><option value="motion">Motion</option></select></label>
      <button id="syncVideos" type="button">Sync</button>
      <button id="playVideos" type="button">Play</button>
      <button id="pauseVideos" type="button">Pause</button>
    </div>
    <div class="video-grid" id="videoGrid"></div>
    <div class="small" id="clipNote"></div>
  </section>
  <div class="summary" id="summary"></div>
  <section class="intelligence" id="intelligencePanel">
    <h2>Results Intelligence</h2>
    <div class="small">Family winners, horizon winners, target-mode comparisons, and failed configurations for the currently selected metric split.</div>
    <div class="intel-grid">
      <article class="intel-card"><h3>Top Models</h3><div id="intelTop"></div></article>
      <article class="intel-card"><h3>Family Winners</h3><div id="intelFamily"></div></article>
      <article class="intel-card"><h3>Horizon Winners</h3><div id="intelHorizon"></div></article>
      <article class="intel-card"><h3>Delta vs Absolute</h3><div id="intelTarget"></div></article>
      <article class="intel-card"><h3>Runtime</h3><div id="intelRuntime"></div></article>
      <article class="intel-card"><h3>Failure Heatmap</h3><div id="intelFailures"></div></article>
    </div>
  </section>
  <div class="compare">
    <div>
      <table>
        <thead><tr><th style="width:42px"></th><th>Experiment</th><th style="width:120px">Family</th><th style="width:110px">Dataset</th><th style="width:80px">Loss</th><th style="width:85px">Target</th><th>HParams</th><th style="width:100px">Split MSE</th><th style="width:105px">Improve</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <aside class="side">
      <h2>Selected Models</h2>
      <div class="small" id="videoNote"></div>
      <div id="selected"></div>
    </aside>
  </div>
</main>
<script>
const payload = {data};
const intelligence = payload.intelligence || {{}};
let selected = new Map(payload.selected_models.map(row => [row.row_id, row]));
const families = ['All', ...Array.from(new Set(payload.rows.map(r => r.model_family || r.kind))).sort()];
const datasets = ['All', ...Array.from(new Set(payload.rows.map(r => r.dataset_key))).sort()];
const videos = [{{label:'All held-out first', value:'All'}}, ...payload.input_videos.map(v => ({{label:`${{v.split}} · ${{v.video_id}} (${{v.dataset_key}})`, value:`${{v.dataset_key}}|${{v.video_id}}`}}))];
const videoCollections = payload.video_collections || [];
const splitRank = {{test:0, val:1, validation:1, train:2, unknown:3}};
function fillSelect(id, options) {{
  const node = document.getElementById(id);
  node.innerHTML = options.map(o => typeof o === 'string' ? `<option value="${{escapeHtml(o)}}">${{escapeHtml(o)}}</option>` : `<option value="${{escapeHtml(o.value)}}">${{escapeHtml(o.label)}}</option>`).join('');
}}
function metric(row, split, name) {{ return row[`${{split}}_${{name}}`]; }}
function fmt(value) {{ return value === null || value === undefined ? 'n/a' : Number(value).toExponential(3); }}
function cls(value) {{ return Number(value || 0) >= 0 ? 'good' : 'bad'; }}
function escapeHtml(value) {{ return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function visibleRows() {{
  const family = document.getElementById('familyFilter').value;
  const dataset = document.getElementById('datasetFilter').value;
  const split = document.getElementById('splitFilter').value;
  const q = document.getElementById('searchBox').value.trim().toLowerCase();
  return payload.rows.filter(row => {{
    if (family !== 'All' && row.model_family !== family) return false;
    if (dataset !== 'All' && row.dataset_key !== dataset) return false;
    if (metric(row, split, 'decoded_prediction_mse') === null || metric(row, split, 'decoded_prediction_mse') === undefined) return false;
    if (q) {{
      const haystack = `${{row.experiment_id}} ${{row.kind}} ${{row.model_family}} ${{row.loss_mode || ''}} ${{row.prediction_target || ''}} ${{row.objective || ''}} ${{row.hyperparameter_summary || ''}} ${{JSON.stringify(row.params || {{}})}}`.toLowerCase();
      if (!haystack.includes(q)) return false;
    }}
    return true;
  }});
}}

function activeCollection() {{
  const id = document.getElementById('clipSet').value;
  return videoCollections.find((collection, index) => String(index) === id) || videoCollections[0];
}}
function fillClipSets() {{
  const panel = document.getElementById('videoPanel');
  if (!videoCollections.length) {{
    panel.style.display = 'none';
    return;
  }}
  panel.style.display = '';
  fillSelect('clipSet', videoCollections.map((collection, index) => ({{label: collection.label || collection.sweep_id || `Clip set ${{index + 1}}`, value: String(index)}})));
  fillClipInputs();
}}
function clipInputRecords(collection) {{
  const byVideo = new Map();
  for (const item of collection.options || []) {{
    if (!byVideo.has(item.video_id)) byVideo.set(item.video_id, {{video_id:item.video_id, split:item.split || 'unknown'}});
  }}
  return Array.from(byVideo.values()).sort((a, b) => (splitRank[a.split] ?? 9) - (splitRank[b.split] ?? 9) || String(a.video_id).localeCompare(String(b.video_id)));
}}
function fillClipInputs() {{
  const collection = activeCollection();
  const prior = document.getElementById('clipInput').value;
  const options = clipInputRecords(collection).map(item => ({{label:`${{item.split}} · ${{item.video_id}}`, value:item.video_id}}));
  fillSelect('clipInput', options);
  if (options.some(item => item.value === prior)) document.getElementById('clipInput').value = prior;
  renderVideoComparison();
}}
function renderVideoComparison() {{
  const collection = activeCollection();
  const grid = document.getElementById('videoGrid');
  if (!collection) return;
  const videoId = document.getElementById('clipInput').value;
  const view = document.getElementById('clipView').value;
  const cards = (collection.models || []).map(model => {{
    const item = (collection.options || []).find(option => option.model_tag === model.tag && option.video_id === videoId);
    if (!item) return '';
    const src = view === 'motion' ? item.motion_src : item.intensity_src;
    return `<article class="video-card"><header><h3>${{escapeHtml(model.label || model.tag)}} · ${{escapeHtml(item.experiment_id)}}</h3></header><video class="clipVideo" controls muted loop playsinline src="${{escapeHtml(src)}}"></video><div class="meta-line">${{escapeHtml(item.dataset_key)}} · ${{escapeHtml(item.split)}} · horizon ${{escapeHtml(item.target_offset_raw_frames)}} raw frames · display improve ${{fmt(item.improvement_over_persistence_mse)}}</div></article>`;
  }}).filter(Boolean);
  grid.innerHTML = cards.length ? cards.join('') : '<div class="small">No rendered clips are available for this input/model combination yet.</div>';
  document.getElementById('clipStatus').textContent = `${{cards.length}} rendered model clip${{cards.length === 1 ? '' : 's'}}`;
  document.getElementById('clipNote').textContent = collection.segment_selection || 'Rendered clips are offset-aligned visual diagnostics; sweep metrics remain direct forecast metrics.';
}}
function clipVideos() {{
  return Array.from(document.querySelectorAll('.clipVideo'));
}}
function syncClipVideos() {{
  const clips = clipVideos();
  if (!clips.length) return;
  const time = clips[0].currentTime || 0;
  for (const clip of clips) clip.currentTime = time;
}}
function playClipVideos() {{
  syncClipVideos();
  for (const clip of clipVideos()) clip.play().catch(() => {{}});
}}
function pauseClipVideos() {{
  for (const clip of clipVideos()) clip.pause();
}}

function renderIntelModelList(rows, groupLabel) {{
  if (!rows || !rows.length) return '<div class="small">No completed rows for this split.</div>';
  return rows.slice(0, 5).map(row => {{
    const label = groupLabel ? `${{groupLabel(row)}} · ${{row.experiment_id}}` : row.experiment_id;
    return `<div class="intel-item"><div class="title mono">${{escapeHtml(label)}}</div><div class="meta">${{escapeHtml(row.model_family || row.kind || '')}} · ${{escapeHtml(row.dataset_key || '')}} · ${{escapeHtml(row.prediction_target || row.baseline_name || '')}}</div><div class="meta">improve <span class="${{cls(row.improvement_over_persistence_mse)}} mono">${{fmt(row.improvement_over_persistence_mse)}}</span></div><div class="meta">${{escapeHtml(row.hyperparameter_summary || '')}}</div></div>`;
  }}).join('');
}}
function renderIntelGroups(groups) {{
  const entries = Object.entries(groups || {{}});
  if (!entries.length) return '<div class="small">No grouped rows for this split.</div>';
  return entries.slice(0, 8).map(([name, stats]) => {{
    const best = stats.best || {{}};
    return `<div class="intel-item"><div class="title">${{escapeHtml(name)}}</div><div class="meta">${{escapeHtml(stats.positive_count || 0)}} / ${{escapeHtml(stats.count || 0)}} positive · mean ${{fmt(stats.mean_improvement)}}</div><div class="meta mono">best ${{escapeHtml(best.experiment_id || '')}} · <span class="${{cls(best.improvement_over_persistence_mse)}}">${{fmt(best.improvement_over_persistence_mse)}}</span></div></div>`;
  }}).join('');
}}
function renderIntelFailures(summary) {{
  const failureCount = Number(summary?.failure_count || 0);
  if (!failureCount) return '<div class="small">No failed configurations found in current or archived progress logs.</div>';
  const byClass = Object.entries(summary.by_class || {{}}).map(([name, count]) => `<div class="intel-item"><div class="title">${{escapeHtml(name)}}</div><div class="meta">${{escapeHtml(count)}} failed configuration${{Number(count) === 1 ? '' : 's'}}</div></div>`).join('');
  const byKind = Object.entries(summary.by_kind || {{}}).slice(0, 5).map(([name, count]) => `${{escapeHtml(name)}}=${{escapeHtml(count)}}`).join(' · ');
  return `${{byClass}}<div class="small">By kind: ${{byKind || 'n/a'}}</div>`;
}}
function fmtDuration(seconds) {{
  const value = Number(seconds);
  if (!Number.isFinite(value)) return 'n/a';
  if (value < 90) return `${{value.toFixed(0)}}s`;
  if (value < 7200) return `${{(value / 60).toFixed(1)}}m`;
  return `${{(value / 3600).toFixed(2)}}h`;
}}
function renderIntelRuntime(summary) {{
  if (!summary?.available) return '<div class="small">No completed runtime records found.</div>';
  const family = Object.entries(summary.by_family || {{}}).sort((a, b) => Number(b[1].median_seconds || 0) - Number(a[1].median_seconds || 0)).slice(0, 4).map(([name, item]) => `<div class="intel-item"><div class="title">${{escapeHtml(name)}}</div><div class="meta">${{escapeHtml(item.count || 0)}} rows · median ${{fmtDuration(item.median_seconds)}} · max ${{fmtDuration(item.max_seconds)}}</div></div>`).join('');
  const slowest = (summary.slowest_rows || []).slice(0, 2).map(row => `<div class="meta mono">${{escapeHtml(row.experiment_id)}} · ${{fmtDuration(row.elapsed_seconds)}}</div>`).join('');
  return `${{family}}<div class="small">Timed rows: ${{escapeHtml(summary.row_count || 0)}} · median ${{fmtDuration(summary.median_seconds)}} · total ${{fmtDuration(summary.total_seconds)}}</div>${{slowest}}`;
}}
function renderVideoEvidence(row, split) {{
  const summary = row.video_error_summary?.[split];
  if (!summary) return '<div class="small">Per-video prediction metrics are not available for this row.</div>';
  const line = item => `${{escapeHtml(item.video_id)}} ${{fmt(item.improvement_over_persistence_mse)}}`;
  const best = (summary.best_videos || []).slice(0, 5).map(line).join(' · ') || 'n/a';
  const worst = (summary.worst_videos || []).slice(0, 5).map(line).join(' · ') || 'n/a';
  const labels = (summary.label_summary || []).slice(0, 4).map(item => `${{escapeHtml(item.label)}} ${{fmt(item.improvement_over_persistence_mse)}}`).join(' · ') || 'n/a';
  return `<div class="small">${{escapeHtml(split)}} videos: ${{escapeHtml(summary.video_count || 0)}} · best ${{best}} · worst ${{worst}} · labels ${{labels}}</div>`;
}}
function renderIntelligence() {{
  const split = document.getElementById('splitFilter').value;
  const leaders = intelligence.leaderboards?.[split] || [];
  const family = intelligence.best_by_family?.[split] || {{}};
  const horizon = intelligence.best_by_horizon?.[split] || {{}};
  const target = intelligence.target_comparison?.[split] || {{}};
  document.getElementById('intelTop').innerHTML = renderIntelModelList(leaders);
  document.getElementById('intelFamily').innerHTML = renderIntelModelList(Object.entries(family).map(([name, row]) => ({{...row, group_name:name}})), row => row.group_name);
  document.getElementById('intelHorizon').innerHTML = renderIntelModelList(Object.entries(horizon).map(([name, row]) => ({{...row, group_name:name}})), row => row.group_name);
  document.getElementById('intelTarget').innerHTML = renderIntelGroups(target);
  document.getElementById('intelRuntime').innerHTML = renderIntelRuntime(intelligence.runtime_summary || {{}});
  document.getElementById('intelFailures').innerHTML = renderIntelFailures(intelligence.failure_summary || {{}});
}}

function render() {{
  const split = document.getElementById('splitFilter').value;
  const rows = visibleRows();
  document.getElementById('summary').innerHTML = [
    ['Rows', rows.length],
    ['Positive improve', rows.filter(r => Number(metric(r, split, 'improvement_over_persistence_mse') || 0) > 0).length],
    ['Families', new Set(rows.map(r => r.model_family)).size],
    ['Input videos', payload.input_videos.length]
  ].map(([label, value]) => `<div class="metric"><b>${{escapeHtml(value)}}</b><span>${{escapeHtml(label)}}</span></div>`).join('');
  renderIntelligence();
  document.getElementById('rows').innerHTML = rows.slice(0, 500).map(row => {{
    const mse = metric(row, split, 'decoded_prediction_mse');
    const improve = metric(row, split, 'improvement_over_persistence_mse');
    const active = selected.has(row.row_id);
    return `<tr class="${{active ? 'selected' : ''}}"><td><button data-id="${{escapeHtml(row.row_id)}}" class="${{active ? 'primary' : ''}}">${{active ? 'On' : 'Add'}}</button></td><td><div class="mono">${{escapeHtml(row.experiment_id)}}</div><div class="small">${{escapeHtml(row.objective)}}</div></td><td>${{escapeHtml(row.model_family)}}</td><td>${{escapeHtml(row.dataset_key)}}</td><td>${{escapeHtml(row.loss_mode || row.baseline_name || '')}}</td><td>${{escapeHtml(row.prediction_target || '')}}</td><td><div class="small">${{escapeHtml(row.hyperparameter_summary || '')}}</div></td><td class="mono">${{fmt(mse)}}</td><td class="mono ${{cls(improve)}}">${{fmt(improve)}}</td></tr>`;
  }}).join('');
  document.querySelectorAll('button[data-id]').forEach(btn => btn.addEventListener('click', () => toggle(btn.dataset.id)));
  renderSelected();
}}
function toggle(id) {{
  if (selected.has(id)) selected.delete(id);
  else {{
    if (selected.size >= 3) selected.delete(Array.from(selected.keys())[0]);
    selected.set(id, payload.rows.find(r => r.row_id === id));
  }}
  render();
}}
function renderSelected() {{
  const video = document.getElementById('inputVideoFilter').value;
  document.getElementById('videoNote').textContent = video === 'All' ? 'Video filter is set to held-out-first ordering.' : `Video focus: ${{video.replace('|', ' / ')}}`;
  const rows = Array.from(selected.values());
  const split = document.getElementById('splitFilter').value;
  document.getElementById('selected').innerHTML = rows.length ? rows.map(row => `<div class="slot"><div class="mono">${{escapeHtml(row.experiment_id)}}</div><div class="small">${{escapeHtml(row.model_family)}} · ${{escapeHtml(row.dataset_key)}} · ${{escapeHtml(row.loss_mode || row.prediction_target || row.baseline_name || '')}}</div><div class="small">${{escapeHtml(row.hyperparameter_summary || '')}}</div><div class="small">test improve <span class="${{cls(row.test_improvement_over_persistence_mse)}} mono">${{fmt(row.test_improvement_over_persistence_mse)}}</span>, val improve <span class="${{cls(row.val_improvement_over_persistence_mse)}} mono">${{fmt(row.val_improvement_over_persistence_mse)}}</span></div>${{renderVideoEvidence(row, split)}}</div>`).join('') : '<div class="small">Select rows from the table.</div>';
}}
fillSelect('familyFilter', families);
fillSelect('datasetFilter', datasets);
fillSelect('inputVideoFilter', videos);
fillClipSets();
document.getElementById('clipSet').addEventListener('input', fillClipInputs);
document.getElementById('clipInput').addEventListener('input', renderVideoComparison);
document.getElementById('clipView').addEventListener('input', renderVideoComparison);
document.getElementById('syncVideos').addEventListener('click', syncClipVideos);
document.getElementById('playVideos').addEventListener('click', playClipVideos);
document.getElementById('pauseVideos').addEventListener('click', pauseClipVideos);
['familyFilter','datasetFilter','splitFilter','inputVideoFilter','searchBox'].forEach(id => document.getElementById(id).addEventListener('input', render));
render();
</script>
</body>
</html>
"""


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a static comparison dashboard for grid dynamics sweeps.")
    parser.add_argument("--sweep-dir", action="append", required=True, help="Sweep directory to include. Can be passed multiple times.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="Grid Dynamics Architecture Comparison")
    parser.add_argument("--dashboard-prefix", default="")
    parser.add_argument("--selected-count", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = build_comparison_dashboard(
        sweep_dirs=args.sweep_dir,
        out_dir=args.out_dir,
        title=args.title,
        dashboard_prefix=args.dashboard_prefix,
        selected_count=args.selected_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
