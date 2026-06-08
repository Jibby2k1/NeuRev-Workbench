#!/usr/bin/env python3
"""Analyze component-level effects across scalable temporal-CNN sweeps."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.dynamics.overnight_sweep import collect_metric_rows

NUMERIC_FACTORS = (
    "parameter_count",
    "conv_layer_count",
    "total_configured_blocks",
    "stack_depth",
    "encoder_depth",
    "decoder_depth",
    "bottleneck_layers",
    "max_channels",
    "dropout",
    "learning_rate",
    "residual_scale",
)
CATEGORICAL_FACTORS = (
    "grid_size",
    "dataset_key",
    "architecture_id",
    "topology",
    "loss_mode",
    "skip_connections",
    "normalization",
    "activation",
)
OUTPUT_COLUMNS = (
    "sweep_dir",
    "experiment_id",
    "kind",
    "dataset_key",
    "seed",
    "architecture_id",
    "topology",
    "grid_size",
    "loss_mode",
    "learning_rate",
    "residual_scale",
    "parameter_count",
    "conv_layer_count",
    "total_configured_blocks",
    "stack_depth",
    "encoder_depth",
    "decoder_depth",
    "bottleneck_layers",
    "max_channels",
    "skip_connections",
    "normalization",
    "activation",
    "dropout",
    "val_decoded_prediction_mse",
    "val_persistence_mse",
    "val_improvement_over_persistence_mse",
    "test_decoded_prediction_mse",
    "test_persistence_mse",
    "test_improvement_over_persistence_mse",
    "metrics_path",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_for_metric(sweep_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _load_json(row["metrics_path"])
    params = dict(row.get("params") or {})
    spec = dict(metrics.get("architecture_spec") or params.get("architecture_spec") or {})
    summary = dict(metrics.get("architecture_summary") or {})
    out = {
        "sweep_dir": str(sweep_dir),
        "experiment_id": row.get("experiment_id"),
        "kind": row.get("kind"),
        "dataset_key": row.get("dataset_key"),
        "seed": row.get("seed"),
        "architecture_id": metrics.get("architecture_id") or params.get("architecture_id") or spec.get("architecture_id"),
        "topology": metrics.get("topology") or summary.get("topology") or spec.get("topology"),
        "grid_size": metrics.get("grid_size") or summary.get("grid_size"),
        "loss_mode": metrics.get("loss_mode") or row.get("loss_mode") or params.get("loss_mode"),
        "learning_rate": metrics.get("learning_rate") or params.get("learning_rate"),
        "residual_scale": metrics.get("residual_scale") or params.get("residual_scale"),
        "parameter_count": metrics.get("parameter_count") or summary.get("parameter_count"),
        "conv_layer_count": metrics.get("conv_layer_count") or summary.get("conv_layer_count"),
        "total_configured_blocks": metrics.get("total_configured_blocks") or summary.get("total_configured_blocks"),
        "stack_depth": metrics.get("stack_depth") or summary.get("stack_depth"),
        "encoder_depth": metrics.get("encoder_depth") or summary.get("encoder_depth"),
        "decoder_depth": metrics.get("decoder_depth") or summary.get("decoder_depth"),
        "bottleneck_layers": metrics.get("bottleneck_layers") or summary.get("bottleneck_layers"),
        "max_channels": metrics.get("max_channels") or summary.get("max_channels"),
        "skip_connections": metrics.get("skip_connections") if "skip_connections" in metrics else summary.get("skip_connections", spec.get("skip_connections")),
        "normalization": metrics.get("normalization") or summary.get("normalization") or spec.get("normalization"),
        "activation": metrics.get("activation") or summary.get("activation") or spec.get("activation"),
        "dropout": metrics.get("dropout") or summary.get("dropout") or spec.get("dropout"),
        "val_decoded_prediction_mse": metrics.get("val_decoded_prediction_mse"),
        "val_persistence_mse": metrics.get("val_persistence_mse"),
        "val_improvement_over_persistence_mse": metrics.get("val_improvement_over_persistence_mse"),
        "test_decoded_prediction_mse": metrics.get("test_decoded_prediction_mse"),
        "test_persistence_mse": metrics.get("test_persistence_mse"),
        "test_improvement_over_persistence_mse": metrics.get("test_improvement_over_persistence_mse"),
        "metrics_path": row.get("metrics_path"),
    }
    return out


def load_component_rows(sweep_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep_dir_raw in sweep_dirs:
        sweep_dir = Path(sweep_dir_raw)
        for row in collect_metric_rows(sweep_dir):
            if str(row.get("kind")) != "scalable_temporal_cnn_pixel":
                continue
            rows.append(_row_for_metric(sweep_dir, row))
    return rows


def group_summary(rows: list[Mapping[str, Any]], factor: str, metric: str) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = _num(row.get(metric))
        if value is None:
            continue
        key = str(row.get(factor, "unknown"))
        groups.setdefault(key, []).append(value)
    out = []
    for key, values in groups.items():
        mean = sum(values) / len(values)
        ordered = sorted(values)
        median = ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
        out.append({"factor": factor, "level": key, "count": len(values), "mean": mean, "median": median, "positive_count": sum(v > 0 for v in values)})
    return sorted(out, key=lambda item: (-float(item["mean"]), str(item["level"])))


def numeric_correlations(rows: list[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    ys = [_num(row.get(metric)) for row in rows]
    for factor in NUMERIC_FACTORS:
        pairs = [(_num(row.get(factor)), y) for row, y in zip(rows, ys) if y is not None]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            continue
        xs = [p[0] for p in pairs]
        yy = [p[1] for p in pairs]
        mx = sum(xs) / len(xs)
        my = sum(yy) / len(yy)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in yy)
        corr = 0.0 if vx <= 0 or vy <= 0 else sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)
        out.append({"factor": factor, "count": len(pairs), "correlation": corr})
    return sorted(out, key=lambda item: -abs(float(item["correlation"])))


def write_rows_tsv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def _write_bar_svg(path: Path, title: str, rows: list[Mapping[str, Any]], *, max_items: int = 12) -> None:
    items = rows[:max_items]
    width = 920
    row_h = 28
    margin_left = 230
    margin_right = 110
    height = 58 + max(1, len(items)) * row_h
    values = [float(item.get("mean") or 0.0) for item in items]
    max_abs = max([abs(v) for v in values] + [1e-12])
    zero_x = margin_left + (width - margin_left - margin_right) / 2
    scale = (width - margin_left - margin_right) / (2 * max_abs)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(f'<text x="18" y="28" font-family="Arial" font-size="18" font-weight="700" fill="#172033">{_xml(title)}</text>')
    lines.append(f'<line x1="{zero_x:.1f}" y1="44" x2="{zero_x:.1f}" y2="{height - 12}" stroke="#98a2b3"/>')
    for index, item in enumerate(items):
        y = 54 + index * row_h
        value = float(item.get("mean") or 0.0)
        x0 = zero_x if value >= 0 else zero_x + value * scale
        bar_w = abs(value * scale)
        color = "#087443" if value >= 0 else "#b42318"
        label = str(item.get("level", ""))[:34]
        lines.append(f'<text x="18" y="{y + 16}" font-family="Arial" font-size="12" fill="#344054">{_xml(label)}</text>')
        lines.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{bar_w:.1f}" height="16" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{width - 96}" y="{y + 16}" font-family="Arial" font-size="12" fill="#344054">{value:.3e}</text>')
    lines.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _xml(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def analyze_architecture_components(*, sweep_dirs: Iterable[str | Path], out_dir: str | Path, metric: str = "test_improvement_over_persistence_mse") -> dict[str, Any]:
    out = Path(out_dir)
    rows = load_component_rows(sweep_dirs)
    write_rows_tsv(out / "architecture_component_rows.tsv", rows)
    factor_summaries = {factor: group_summary(rows, factor, metric) for factor in CATEGORICAL_FACTORS}
    correlations = numeric_correlations(rows, metric)
    charts_dir = out / "charts"
    for factor in ("grid_size", "architecture_id", "topology", "loss_mode", "skip_connections"):
        _write_bar_svg(charts_dir / f"{factor}_mean_{metric}.svg", f"Mean {metric} by {factor}", factor_summaries[factor])
    top_rows = sorted([row for row in rows if _num(row.get(metric)) is not None], key=lambda row: -float(row[metric]))[:20]
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sweep_dirs": [str(Path(p)) for p in sweep_dirs],
        "row_count": len(rows),
        "metric": metric,
        "positive_count": sum((_num(row.get(metric)) or 0.0) > 0 for row in rows),
        "factor_summaries": factor_summaries,
        "numeric_correlations": correlations,
        "top_rows": top_rows,
        "rows_tsv": str(out / "architecture_component_rows.tsv"),
        "charts_dir": str(charts_dir),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "architecture_component_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out / "architecture_component_report.md", summary)
    return summary


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Architecture Component Analysis",
        "",
        f"Rows analyzed: `{summary.get('row_count', 0)}`",
        f"Metric: `{summary.get('metric')}`",
        f"Positive rows: `{summary.get('positive_count', 0)}`",
        "",
        "## Top Runs",
        "",
        "| Rank | Experiment | Grid | Architecture | Loss | Test improve | Val improve |",
        "|---:|---|---:|---|---|---:|---:|",
    ]
    for index, row in enumerate(summary.get("top_rows", [])[:15], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row.get('experiment_id')}`",
                    str(row.get("grid_size", "")),
                    str(row.get("architecture_id", "")),
                    str(row.get("loss_mode", "")),
                    _fmt(row.get("test_improvement_over_persistence_mse")),
                    _fmt(row.get("val_improvement_over_persistence_mse")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Strongest Numeric Associations", "", "| Factor | Correlation | Count |", "|---|---:|---:|"])
    for item in summary.get("numeric_correlations", [])[:12]:
        lines.append(f"| `{item.get('factor')}` | {_fmt(item.get('correlation'))} | {item.get('count')} |")
    lines.extend(["", "## Categorical Summaries", ""])
    for factor, rows in dict(summary.get("factor_summaries") or {}).items():
        lines.extend([f"### {factor}", "", "| Level | Count | Mean improve | Median improve | Positive |", "|---|---:|---:|---:|---:|"])
        for item in rows[:12]:
            lines.append(f"| `{item.get('level')}` | {item.get('count')} | {_fmt(item.get('mean'))} | {_fmt(item.get('median'))} | {item.get('positive_count')} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.4e}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", action="append", required=True, help="Sweep directory to analyze. Can be passed multiple times.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metric", default="test_improvement_over_persistence_mse")
    args = parser.parse_args(argv)
    summary = analyze_architecture_components(sweep_dirs=args.sweep_dir, out_dir=args.out_dir, metric=args.metric)
    print(json.dumps({"status": "analyzed", "row_count": summary["row_count"], "summary_path": str(args.out_dir / "architecture_component_summary.json")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
