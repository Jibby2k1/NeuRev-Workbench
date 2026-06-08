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
    payload = {
        "schema_version": 1,
        "title": str(title),
        "sweep_dirs": [str(p) for p in sweeps],
        "row_count": len(rows_sorted),
        "rows": rows_sorted,
        "selected_models": selected,
        "datasets": dataset_index,
        "input_videos": videos,
        "video_collections": video_collections,
        "dashboard_prefix": dashboard_prefix,
    }
    manifest_path = out / "comparison_manifest.json"
    html_path = out / "comparison_dashboard.html"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(_comparison_html(payload), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "title": str(title),
        "row_count": len(rows_sorted),
        "selected_model_ids": [r["row_id"] for r in selected],
        "video_collection_count": len(video_collections),
        "manifest_path": str(manifest_path),
        "html_path": str(html_path),
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


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
        "primary_split": primary_split,
        "primary_improvement_over_persistence_mse": primary_improvement,
        "val_decoded_prediction_mse": _num(metrics.get("val_decoded_prediction_mse")),
        "val_persistence_mse": _num(metrics.get("val_persistence_mse")),
        "val_improvement_over_persistence_mse": _num(metrics.get("val_improvement_over_persistence_mse")),
        "test_decoded_prediction_mse": _num(metrics.get("test_decoded_prediction_mse")),
        "test_persistence_mse": _num(metrics.get("test_persistence_mse")),
        "test_improvement_over_persistence_mse": _num(metrics.get("test_improvement_over_persistence_mse")),
        "all_decoded_prediction_mse": _num(metrics.get("decoded_prediction_mse")),
        "all_persistence_mse": _num(metrics.get("persistence_mse")),
        "all_improvement_over_persistence_mse": _num(metrics.get("improvement_over_persistence_mse")),
        "metrics_path": str(row.get("metrics_path", "")),
        "params": params,
    }


def _dataset_record(dataset_key: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset_path = Path(str(cfg.get("dataset", ""))) if cfg.get("dataset") else None
    record: dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_path": str(dataset_path) if dataset_path else "",
        "autoencoder_run": str(cfg.get("autoencoder_run", "")),
        "window_frames": cfg.get("window_frames"),
        "videos": [],
    }
    if not dataset_path or not dataset_path.exists():
        return record
    dataset = _load_json(dataset_path)
    record["windowing"] = dataset.get("windowing", {})
    record["splits"] = dataset.get("splits", {})
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
@media (max-width: 980px) {{ .controls, .summary, .compare {{ grid-template-columns:1fr; }} .side {{ position:static; }} .video-head {{ display:block; }} }}
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
  <div class="compare">
    <div>
      <table>
        <thead><tr><th style="width:42px"></th><th>Experiment</th><th style="width:130px">Family</th><th style="width:120px">Dataset</th><th style="width:80px">Loss</th><th style="width:95px">Target</th><th style="width:110px">Split MSE</th><th style="width:120px">Improve</th></tr></thead>
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
      const haystack = `${{row.experiment_id}} ${{row.kind}} ${{row.model_family}} ${{row.loss_mode || ''}} ${{row.prediction_target || ''}} ${{row.objective || ''}}`.toLowerCase();
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

function render() {{
  const split = document.getElementById('splitFilter').value;
  const rows = visibleRows();
  document.getElementById('summary').innerHTML = [
    ['Rows', rows.length],
    ['Positive improve', rows.filter(r => Number(metric(r, split, 'improvement_over_persistence_mse') || 0) > 0).length],
    ['Families', new Set(rows.map(r => r.model_family)).size],
    ['Input videos', payload.input_videos.length]
  ].map(([label, value]) => `<div class="metric"><b>${{escapeHtml(value)}}</b><span>${{escapeHtml(label)}}</span></div>`).join('');
  document.getElementById('rows').innerHTML = rows.slice(0, 500).map(row => {{
    const mse = metric(row, split, 'decoded_prediction_mse');
    const improve = metric(row, split, 'improvement_over_persistence_mse');
    const active = selected.has(row.row_id);
    return `<tr class="${{active ? 'selected' : ''}}"><td><button data-id="${{escapeHtml(row.row_id)}}" class="${{active ? 'primary' : ''}}">${{active ? 'On' : 'Add'}}</button></td><td><div class="mono">${{escapeHtml(row.experiment_id)}}</div><div class="small">${{escapeHtml(row.objective)}}</div></td><td>${{escapeHtml(row.model_family)}}</td><td>${{escapeHtml(row.dataset_key)}}</td><td>${{escapeHtml(row.loss_mode || row.baseline_name || '')}}</td><td>${{escapeHtml(row.prediction_target || '')}}</td><td class="mono">${{fmt(mse)}}</td><td class="mono ${{cls(improve)}}">${{fmt(improve)}}</td></tr>`;
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
  document.getElementById('selected').innerHTML = rows.length ? rows.map(row => `<div class="slot"><div class="mono">${{escapeHtml(row.experiment_id)}}</div><div class="small">${{escapeHtml(row.model_family)}} · ${{escapeHtml(row.dataset_key)}} · ${{escapeHtml(row.loss_mode || row.prediction_target || row.baseline_name || '')}}</div><div class="small">test improve <span class="${{cls(row.test_improvement_over_persistence_mse)}} mono">${{fmt(row.test_improvement_over_persistence_mse)}}</span>, val improve <span class="${{cls(row.val_improvement_over_persistence_mse)}} mono">${{fmt(row.val_improvement_over_persistence_mse)}}</span></div></div>`).join('') : '<div class="small">Select rows from the table.</div>';
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
