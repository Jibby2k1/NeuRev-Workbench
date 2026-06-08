'''Generate dependency-free SVG visuals for dynamics overnight sweeps.'''
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

KIND_COLORS = {
    "residual_pixel": "#0f8b8d",
    "latent_gru": "#c14953",
    "latent_transformer": "#5b5fc7",
}


def generate_sweep_visuals(
    *,
    summary_tsv: str | Path,
    out_dir: str | Path,
    title: str = "Overnight Dynamics Sweep",
    dashboard_prefix: str = "",
    top_n: int = 30,
) -> dict[str, Any]:
    summary_path = Path(summary_tsv)
    out = Path(out_dir)
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    rows = load_sweep_rows(summary_path)
    enrich_rows_with_configs(rows)
    artifacts = [
        {"id": "top_validation_improvement", "label": "Top validation improvement", "file": _join_prefix(dashboard_prefix, "charts/top_validation_improvement.svg"), "path": str(charts / "top_validation_improvement.svg"), "description": "Highest validation MSE improvement over split-aware persistence."},
        {"id": "val_test_scatter", "label": "Validation vs test improvement", "file": _join_prefix(dashboard_prefix, "charts/val_test_scatter.svg"), "path": str(charts / "val_test_scatter.svg"), "description": "Each experiment plotted by validation and test improvement over persistence."},
        {"id": "kind_dataset_summary", "label": "Architecture family summary", "file": _join_prefix(dashboard_prefix, "charts/kind_dataset_summary.svg"), "path": str(charts / "kind_dataset_summary.svg"), "description": "Mean validation improvement grouped by architecture family and dataset."},
        {"id": "residual_hyperparameter_matrix", "label": "Residual hyperparameter matrix", "file": _join_prefix(dashboard_prefix, "charts/residual_hyperparameter_matrix.svg"), "path": str(charts / "residual_hyperparameter_matrix.svg"), "description": "Mean validation improvement for residual-frame models by hidden size, learning rate, and residual scale."},
    ]
    write_top_bar_svg(charts / "top_validation_improvement.svg", rows, title=title, top_n=top_n)
    write_scatter_svg(charts / "val_test_scatter.svg", rows, title=title)
    write_kind_dataset_svg(charts / "kind_dataset_summary.svg", rows, title=title)
    write_residual_matrix_svg(charts / "residual_hyperparameter_matrix.svg", rows, title=title)
    summary = build_visual_summary(rows, artifacts=artifacts, source_tsv=summary_path, title=title)
    (out / "sweep_visual_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "sweep_visual_summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    return summary


def load_sweep_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for raw in reader:
            if not raw.get("experiment_id"):
                continue
            row: dict[str, Any] = dict(raw)
            for key in (
                "rank", "seed", "val_decoded_prediction_mse", "val_persistence_mse",
                "val_improvement_over_persistence_mse", "test_decoded_prediction_mse",
                "test_persistence_mse", "test_improvement_over_persistence_mse",
            ):
                row[key] = _num(row.get(key))
            if row.get("rank") is not None:
                row["rank"] = int(row["rank"])
            if row.get("seed") is not None:
                row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def enrich_rows_with_configs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        metrics_path = Path(str(row.get("metrics_path") or ""))
        config_path = _experiment_config_path(metrics_path)
        if config_path and config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
            row["params"] = dict(config.get("params") or {})
            row["kind"] = row.get("kind") or config.get("kind")
            row["dataset_key"] = row.get("dataset_key") or config.get("dataset_key")
        else:
            row["params"] = {}


def build_visual_summary(rows: list[dict[str, Any]], *, artifacts: list[dict[str, Any]], source_tsv: Path, title: str) -> dict[str, Any]:
    positive_val = [r for r in rows if _finite(r.get("val_improvement_over_persistence_mse")) and r["val_improvement_over_persistence_mse"] > 0]
    positive_test = [r for r in rows if _finite(r.get("test_improvement_over_persistence_mse")) and r["test_improvement_over_persistence_mse"] > 0]
    positive_both = [r for r in rows if _finite(r.get("val_improvement_over_persistence_mse")) and _finite(r.get("test_improvement_over_persistence_mse")) and r["val_improvement_over_persistence_mse"] > 0 and r["test_improvement_over_persistence_mse"] > 0]
    best_validation = max(rows, key=lambda r: r.get("val_improvement_over_persistence_mse") if _finite(r.get("val_improvement_over_persistence_mse")) else -1e9, default={})
    best_test = max(rows, key=lambda r: r.get("test_improvement_over_persistence_mse") if _finite(r.get("test_improvement_over_persistence_mse")) else -1e9, default={})
    best_both = max(positive_both, key=lambda r: r.get("val_improvement_over_persistence_mse", -1e9), default={})
    grouped = []
    for dataset in sorted({str(r.get("dataset_key")) for r in rows if r.get("dataset_key")}):
        for kind in sorted({str(r.get("kind")) for r in rows if r.get("kind")}):
            members = [r for r in rows if r.get("dataset_key") == dataset and r.get("kind") == kind]
            if not members:
                continue
            grouped.append({
                "dataset_key": dataset,
                "kind": kind,
                "count": len(members),
                "mean_val_improvement": _mean(r.get("val_improvement_over_persistence_mse") for r in members),
                "median_val_improvement": _median(r.get("val_improvement_over_persistence_mse") for r in members),
                "mean_test_improvement": _mean(r.get("test_improvement_over_persistence_mse") for r in members),
                "positive_val_count": sum(1 for r in members if _finite(r.get("val_improvement_over_persistence_mse")) and r["val_improvement_over_persistence_mse"] > 0),
                "positive_test_count": sum(1 for r in members if _finite(r.get("test_improvement_over_persistence_mse")) and r["test_improvement_over_persistence_mse"] > 0),
            })
    return {
        "schema_version": 1,
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_tsv": str(source_tsv),
        "experiment_count": len(rows),
        "positive_validation_count": len(positive_val),
        "positive_test_count": len(positive_test),
        "positive_validation_and_test_count": len(positive_both),
        "best_validation": _public_row(best_validation),
        "best_test": _public_row(best_test),
        "best_validation_and_test": _public_row(best_both),
        "top_experiments": [_public_row(r) for r in sorted(rows, key=lambda r: r.get("val_improvement_over_persistence_mse") if _finite(r.get("val_improvement_over_persistence_mse")) else -1e9, reverse=True)[:20]],
        "grouped_summary": grouped,
        "artifacts": artifacts,
    }


def write_top_bar_svg(path: Path, rows: list[dict[str, Any]], *, title: str, top_n: int) -> None:
    top = sorted(rows, key=lambda r: r.get("val_improvement_over_persistence_mse") if _finite(r.get("val_improvement_over_persistence_mse")) else -1e9, reverse=True)[:top_n]
    width = 1280
    row_h = 28
    top_pad = 72
    left = 430
    right = 80
    height = top_pad + row_h * len(top) + 64
    values = [float(r.get("val_improvement_over_persistence_mse") or 0.0) for r in top]
    min_v = min(values + [0.0])
    max_v = max(values + [0.0])
    scale = _scale(min_v, max_v, left, width - right)
    zero_x = scale(0.0)
    parts = [_svg_header(width, height), _title(title, "Top validation improvement over persistence", width)]
    parts.append(f'<line x1="{zero_x:.1f}" y1="56" x2="{zero_x:.1f}" y2="{height-42}" stroke="#3f4a52" stroke-width="1"/>')
    for i, row in enumerate(top):
        y = top_pad + i * row_h
        value = float(row.get("val_improvement_over_persistence_mse") or 0.0)
        x = scale(min(0.0, value))
        x2 = scale(max(0.0, value))
        color = KIND_COLORS.get(str(row.get("kind")), "#68737d") if value >= 0 else "#b54747"
        label = _short_id(str(row.get("experiment_id", "")))
        parts.append(f'<text x="24" y="{y+18}" class="small">{_esc(label)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y+5}" width="{max(1, x2-x):.1f}" height="18" rx="3" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{width-72}" y="{y+18}" text-anchor="end" class="small mono">{_fmt(value)}</text>')
    parts.append(_axis_labels(scale, min_v, max_v, height - 34))
    parts.append(_svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_scatter_svg(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    pts = [r for r in rows if _finite(r.get("val_improvement_over_persistence_mse")) and _finite(r.get("test_improvement_over_persistence_mse"))]
    width, height = 980, 760
    left, right, top, bottom = 86, 32, 72, 82
    xs = [float(r["val_improvement_over_persistence_mse"]) for r in pts] + [0.0]
    ys = [float(r["test_improvement_over_persistence_mse"]) for r in pts] + [0.0]
    xscale = _scale(min(xs), max(xs), left, width - right)
    yscale = _scale(min(ys), max(ys), height - bottom, top)
    parts = [_svg_header(width, height), _title(title, "Validation vs test improvement", width)]
    parts.append(_plot_frame(left, top, width - right, height - bottom))
    parts.append(f'<line x1="{xscale(0):.1f}" y1="{top}" x2="{xscale(0):.1f}" y2="{height-bottom}" stroke="#3f4a52" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{left}" y1="{yscale(0):.1f}" x2="{width-right}" y2="{yscale(0):.1f}" stroke="#3f4a52" stroke-dasharray="4 4"/>')
    for row in pts:
        kind = str(row.get("kind"))
        dataset = str(row.get("dataset_key"))
        color = KIND_COLORS.get(kind, "#68737d")
        radius = 4.8 if dataset == "w8_s1_h50" else 3.9
        x = xscale(float(row["val_improvement_over_persistence_mse"]))
        y = yscale(float(row["test_improvement_over_persistence_mse"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="0.62"><title>{_esc(row.get("experiment_id", ""))}</title></circle>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-30}" text-anchor="middle" class="axis">validation improvement over persistence</text>')
    parts.append(f'<text transform="translate(28 {height/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">test improvement over persistence</text>')
    parts.append(_legend(width - 255, top + 10, KIND_COLORS.items()))
    parts.append(_svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_kind_dataset_svg(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    datasets = sorted({str(r.get("dataset_key")) for r in rows if r.get("dataset_key")})
    kinds = [k for k in ("residual_pixel", "latent_gru", "latent_transformer") if any(r.get("kind") == k for r in rows)]
    width, height = 1120, 620
    left, right, top, bottom = 120, 50, 80, 92
    vals = []
    for dataset in datasets:
        for kind in kinds:
            vals.extend([r.get("val_improvement_over_persistence_mse") for r in rows if r.get("dataset_key") == dataset and r.get("kind") == kind and _finite(r.get("val_improvement_over_persistence_mse"))])
    vals = vals + [0.0]
    yscale = _scale(min(vals), max(vals), height - bottom, top)
    parts = [_svg_header(width, height), _title(title, "Mean validation improvement by dataset and architecture", width)]
    parts.append(_plot_frame(left, top, width - right, height - bottom))
    parts.append(f'<line x1="{left}" y1="{yscale(0):.1f}" x2="{width-right}" y2="{yscale(0):.1f}" stroke="#3f4a52"/>')
    cluster_w = (width - right - left) / max(1, len(datasets))
    bar_w = min(52, cluster_w / max(1, len(kinds) + 1))
    for di, dataset in enumerate(datasets):
        cx = left + cluster_w * di + cluster_w / 2
        parts.append(f'<text x="{cx:.1f}" y="{height-52}" text-anchor="middle" class="axis">{_esc(dataset)}</text>')
        for ki, kind in enumerate(kinds):
            members = [r for r in rows if r.get("dataset_key") == dataset and r.get("kind") == kind and _finite(r.get("val_improvement_over_persistence_mse"))]
            if not members:
                continue
            value = _mean(r.get("val_improvement_over_persistence_mse") for r in members)
            x = cx - (len(kinds) * bar_w) / 2 + ki * bar_w + 4
            y0 = yscale(0)
            y1 = yscale(value)
            y = min(y0, y1)
            h = max(1, abs(y1-y0))
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-8:.1f}" height="{h:.1f}" rx="3" fill="{KIND_COLORS.get(kind, "#68737d")}" opacity="0.86"><title>{_esc(kind)} mean {_fmt(value)} n={len(members)}</title></rect>')
    parts.append(_legend(width - 330, top + 12, ((k, KIND_COLORS[k]) for k in kinds)))
    parts.append(_svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_residual_matrix_svg(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    residual = [r for r in rows if r.get("kind") == "residual_pixel"]
    datasets = sorted({str(r.get("dataset_key")) for r in residual if r.get("dataset_key")})
    hidden = sorted({_int_param(r, "hidden_dim") for r in residual if _int_param(r, "hidden_dim") is not None})
    combos = sorted({(_float_param(r, "learning_rate"), _float_param(r, "residual_scale")) for r in residual if _float_param(r, "learning_rate") is not None and _float_param(r, "residual_scale") is not None})
    row_keys = [(d, h) for d in datasets for h in hidden]
    width = 1160
    cell_w, cell_h = 118, 34
    left, top = 190, 94
    height = top + len(row_keys) * cell_h + 86
    values = []
    for dataset, h in row_keys:
        for lr, rs in combos:
            vals = [r.get("val_improvement_over_persistence_mse") for r in residual if r.get("dataset_key") == dataset and _int_param(r, "hidden_dim") == h and _float_param(r, "learning_rate") == lr and _float_param(r, "residual_scale") == rs and _finite(r.get("val_improvement_over_persistence_mse"))]
            if vals:
                values.append(_mean(vals))
    vmax = max(abs(v) for v in values + [1e-12])
    parts = [_svg_header(width, height), _title(title, "Residual-frame hyperparameter matrix: mean validation improvement", width)]
    for ci, (lr, rs) in enumerate(combos):
        x = left + ci * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="64" text-anchor="middle" class="small">lr {_short_float(lr)}</text>')
        parts.append(f'<text x="{x:.1f}" y="80" text-anchor="middle" class="small">scale {_short_float(rs)}</text>')
    for ri, (dataset, h) in enumerate(row_keys):
        y = top + ri * cell_h
        parts.append(f'<text x="24" y="{y+22}" class="axis">{_esc(dataset)} · hd{h}</text>')
        for ci, (lr, rs) in enumerate(combos):
            vals = [r.get("val_improvement_over_persistence_mse") for r in residual if r.get("dataset_key") == dataset and _int_param(r, "hidden_dim") == h and _float_param(r, "learning_rate") == lr and _float_param(r, "residual_scale") == rs and _finite(r.get("val_improvement_over_persistence_mse"))]
            x = left + ci * cell_w
            value = _mean(vals) if vals else None
            color = _diverging_color(float(value or 0.0), vmax) if value is not None else "#edf0f2"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-6}" height="{cell_h-5}" rx="3" fill="{color}" stroke="#fff"/>')
            if value is not None:
                parts.append(f'<text x="{x+cell_w/2-3:.1f}" y="{y+21}" text-anchor="middle" class="small mono">{_fmt(value)}</text>')
    parts.append(f'<text x="{left}" y="{height-34}" class="small">Green cells improved over persistence; red cells underperformed. Values average seeds where available.</text>')
    parts.append(_svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_summary(summary: Mapping[str, Any]) -> str:
    best = summary.get("best_validation") or {}
    both = summary.get("best_validation_and_test") or {}
    lines = [
        f"# {summary.get('title', 'Sweep Visual Summary')}", "",
        f"Generated: `{summary.get('created_at')}`", "",
        f"- Experiments: {summary.get('experiment_count')}",
        f"- Positive validation improvement: {summary.get('positive_validation_count')}",
        f"- Positive test improvement: {summary.get('positive_test_count')}",
        f"- Positive validation and test improvement: {summary.get('positive_validation_and_test_count')}", "",
        "## Best Validation", "",
        f"- Experiment: `{best.get('experiment_id', 'n/a')}`",
        f"- Dataset: `{best.get('dataset_key', 'n/a')}`",
        f"- Kind: `{best.get('kind', 'n/a')}`",
        f"- Validation improvement: `{best.get('val_improvement_over_persistence_mse', 'n/a')}`",
        f"- Test improvement: `{best.get('test_improvement_over_persistence_mse', 'n/a')}`", "",
        "## Best Positive Validation And Test", "",
        f"- Experiment: `{both.get('experiment_id', 'n/a')}`",
        f"- Dataset: `{both.get('dataset_key', 'n/a')}`",
        f"- Kind: `{both.get('kind', 'n/a')}`",
        f"- Validation improvement: `{both.get('val_improvement_over_persistence_mse', 'n/a')}`",
        f"- Test improvement: `{both.get('test_improvement_over_persistence_mse', 'n/a')}`", "",
        "## Visuals", "",
    ]
    for artifact in summary.get("artifacts", []):
        lines.append(f"- [{artifact.get('label')}]({artifact.get('file')})")
    return "\n".join(lines) + "\n"


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ["rank", "experiment_id", "kind", "dataset_key", "seed", "val_decoded_prediction_mse", "val_persistence_mse", "val_improvement_over_persistence_mse", "test_decoded_prediction_mse", "test_persistence_mse", "test_improvement_over_persistence_mse", "metrics_path"]
    out = {key: row.get(key) for key in keys if row.get(key) is not None}
    if row.get("params"):
        out["params"] = dict(row["params"])
    return out


def _experiment_config_path(metrics_path: Path) -> Path | None:
    if not metrics_path.parts:
        return None
    candidates = [metrics_path.parent / "experiment_config.json", metrics_path.parent.parent / "experiment_config.json"]
    return next((p for p in candidates if p.exists()), candidates[-1])


def _scale(min_v: float, max_v: float, out_min: float, out_max: float):
    if abs(max_v - min_v) < 1e-18:
        min_v -= 1.0
        max_v += 1.0
    pad = (max_v - min_v) * 0.08
    min_v -= pad
    max_v += pad
    def inner(value: float) -> float:
        return out_min + (float(value) - min_v) / (max_v - min_v) * (out_max - out_min)
    return inner


def _svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 22px system-ui, -apple-system, Segoe UI, sans-serif; fill: #15202b; }}
  .subtitle {{ font: 500 13px system-ui, -apple-system, Segoe UI, sans-serif; fill: #5f6b75; }}
  .axis {{ font: 600 12px system-ui, -apple-system, Segoe UI, sans-serif; fill: #35424d; }}
  .small {{ font: 500 11px system-ui, -apple-system, Segoe UI, sans-serif; fill: #35424d; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
<rect width="100%" height="100%" fill="#f8fafb"/>'''


def _title(title: str, subtitle: str, width: int) -> str:
    return f'<text x="24" y="31" class="title">{_esc(title)}</text><text x="24" y="52" class="subtitle">{_esc(subtitle)}</text>'


def _plot_frame(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="#ffffff" stroke="#d9e0e5"/>'


def _legend(x: float, y: float, items: Iterable[tuple[str, str]]) -> str:
    parts = [f'<g transform="translate({x} {y})">']
    for i, (label, color) in enumerate(items):
        yy = i * 22
        parts.append(f'<rect x="0" y="{yy}" width="14" height="14" rx="2" fill="{color}" opacity="0.86"/><text x="22" y="{yy+11}" class="small">{_esc(label)}</text>')
    parts.append('</g>')
    return "".join(parts)


def _axis_labels(scale, min_v: float, max_v: float, y: float) -> str:
    return "".join(f'<text x="{scale(v):.1f}" y="{y}" text-anchor="middle" class="small mono">{_fmt(v)}</text>' for v in (min_v, 0.0, max_v))


def _diverging_color(value: float, vmax: float) -> str:
    ratio = max(-1.0, min(1.0, value / max(vmax, 1e-12)))
    if ratio >= 0:
        base, bg, t = (15, 139, 141), (238, 248, 246), ratio
    else:
        base, bg, t = (193, 73, 83), (252, 240, 241), -ratio
    rgb = tuple(round(bg[i] * (1 - t) + base[i] * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _svg_footer() -> str:
    return "</svg>"


def _join_prefix(prefix: str, file: str) -> str:
    prefix = str(prefix or "").strip("/")
    return f"{prefix}/{file}" if prefix else file


def _short_id(value: str) -> str:
    return value.replace("residual_", "res_").replace("transformer_", "tx_").replace("latent_", "lat_")[:68]


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def _mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return float(statistics.median(vals)) if vals else 0.0


def _int_param(row: Mapping[str, Any], key: str) -> int | None:
    value = (row.get("params") or {}).get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_param(row: Mapping[str, Any], key: str) -> float | None:
    value = (row.get("params") or {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if not _finite(value):
        return "n/a"
    return f"{float(value):.3g}"


def _short_float(value: float) -> str:
    return f"{value:.0e}" if abs(value) < 0.001 else f"{value:g}"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SVG visuals for a dynamics sweep summary TSV.")
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="Overnight Dynamics Sweep")
    parser.add_argument("--dashboard-prefix", default="")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)
    summary = generate_sweep_visuals(summary_tsv=args.summary_tsv, out_dir=args.out_dir, title=args.title, dashboard_prefix=args.dashboard_prefix, top_n=args.top_n)
    print(json.dumps({"summary": str(Path(args.out_dir) / "sweep_visual_summary.json"), "experiment_count": summary["experiment_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
