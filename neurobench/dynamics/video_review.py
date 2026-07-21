"""Static visual review pages for grid dynamics prediction examples."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.workbench.intermediates import normalize_array_frame, write_png_gray8


LEARNED_FAMILIES = {"latent_gru", "latent_transformer", "linear_latent", "convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}
VIDEO_SELECTION_MODES = {"most_improved_video", "least_improved_video"}
HELDOUT_SELECTION_MODE = "heldout_first"


def build_video_error_review(
    *,
    comparison_dir: str | Path,
    out_dir: str | Path,
    selection_mode: str = "best_by_family",
    split: str = "test",
    max_models: int = 5,
    example_index: int = 0,
    dataset_key: str | None = None,
    title: str = "Grid Dynamics Video Error Review",
) -> dict[str, Any]:
    """Build a static HTML review page from saved prediction examples."""
    comparison = Path(comparison_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_review_panels(out)
    manifest = _load_json(comparison / "comparison_manifest.json")
    rows = [dict(row) for row in manifest.get("rows", []) if isinstance(row, Mapping)]
    if dataset_key:
        rows = [row for row in rows if str(row.get("dataset_key")) == str(dataset_key)]
    requested_rows = _select_rows(rows, selection_mode=selection_mode, split=split, max_models=max_models)
    missing_visual_rows = [_missing_visual_summary(row, split=split, selection_mode=selection_mode) for row in requested_rows if not _has_visual_examples(row)]
    visual_rows = [row for row in rows if _has_visual_examples(row)]
    selected = _select_rows(visual_rows, selection_mode=selection_mode, split=split, max_models=max_models)
    review_models: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        examples = _load_examples(_examples_path(row)) if _examples_path(row).exists() else []
        clips = _load_clips(_clip_examples_path(row)) if _clip_examples_path(row).exists() else []
        if not examples and not clips:
            continue
        selection_video = _selection_video_item(row, split=split, selection_mode=selection_mode)
        idx = _artifact_index_for_selection(examples, example_index=example_index, selection_video=selection_video, split=split, selection_mode=selection_mode) if examples else 0
        clip_idx = _artifact_index_for_selection(clips, example_index=example_index, selection_video=selection_video, split=split, selection_mode=selection_mode) if clips else None
        clip = clips[clip_idx] if clip_idx is not None else None
        example = examples[idx] if examples else _first_clip_frame(clip)
        if not example:
            continue
        panel_path = out / f"model_{rank:02d}_{_safe_name(row.get('experiment_id'))}_example_{idx}.png"
        panel = _write_error_panel(panel_path, example)
        clip_panel_path = None
        clip_panel = None
        if clip:
            clip_panel_path = out / f"model_{rank:02d}_{_safe_name(row.get('experiment_id'))}_clip_{clip_idx}.png"
            clip_panel = _write_clip_panel(clip_panel_path, clip)
        review_models.append(
            {
                "rank": rank,
                "experiment_id": row.get("experiment_id"),
                "row_id": row.get("row_id"),
                "kind": row.get("kind"),
                "model_family": row.get("model_family"),
                "dataset_key": row.get("dataset_key"),
                "prediction_target": row.get("prediction_target"),
                "hyperparameter_summary": row.get("hyperparameter_summary"),
                "metrics_path": row.get("metrics_path"),
                "artifact_mode": "temporal_clip" if clip_panel_path else "single_frame",
                "example_index": int(example.get("index", idx)),
                "example_video_id": example.get("video_id"),
                "example_split": example.get("split"),
                "target_frame_index": example.get("target_frame_index"),
                "example_abs_error_mean": _num(example.get("abs_error_mean")),
                "clip_index": int(clip_idx) if clip_idx is not None else None,
                "clip_frame_count": int(clip.get("frame_count") or len(clip.get("frames", []))) if isinstance(clip, Mapping) else None,
                "clip_start_target_frame_index": clip.get("start_target_frame_index") if isinstance(clip, Mapping) else None,
                "clip_end_target_frame_index": clip.get("end_target_frame_index") if isinstance(clip, Mapping) else None,
                "split_improvement_over_persistence_mse": _metric(row, split, "improvement_over_persistence_mse"),
                "selection_metric_name": _selection_metric_name(selection_mode),
                "selection_metric_value": _selection_metric_value(row, split=split, selection_mode=selection_mode),
                "selection_video_id": selection_video.get("video_id") if selection_video else None,
                "selection_video_window_count": selection_video.get("window_count") if selection_video else None,
                "selection_video_decoded_prediction_mse": _num(selection_video.get("decoded_prediction_mse")) if selection_video else None,
                "selection_video_persistence_mse": _num(selection_video.get("persistence_mse")) if selection_video else None,
                "split_decoded_prediction_mse": _metric(row, split, "decoded_prediction_mse"),
                "split_persistence_mse": _metric(row, split, "persistence_mse"),
                "panel_png": str(panel_path),
                "panel_shape": panel,
                "clip_panel_png": str(clip_panel_path) if clip_panel_path else None,
                "clip_panel_shape": clip_panel,
            }
        )
    datasets = manifest.get("datasets", {}) if isinstance(manifest.get("datasets"), Mapping) else {}
    selected_dataset = dataset_key or _dominant_dataset(review_models)
    dataset_record = datasets.get(selected_dataset, {}) if isinstance(datasets, Mapping) else {}
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "comparison_dir": str(comparison),
        "selection_mode": str(selection_mode),
        "split": str(split),
        "requested_max_models": int(max_models),
        "selected_model_count": len(review_models),
        "temporal_clip_model_count": sum(1 for model in review_models if model.get("artifact_mode") == "temporal_clip"),
        "missing_visual_example_count": len(missing_visual_rows),
        "missing_visual_example_rows": missing_visual_rows,
        "dataset_key": selected_dataset,
        "dataset_record": dataset_record,
        "example_index": int(example_index),
        "models": review_models,
        "limitations": _limitations(review_models, missing_visual_rows=missing_visual_rows, selection_mode=selection_mode),
    }
    summary_path = out / "video_error_review.json"
    html_path = out / "video_error_review.html"
    summary["summary_path"] = str(summary_path)
    summary["html_path"] = str(html_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_video_error_review_html(summary), encoding="utf-8")
    return summary


def render_video_error_review_html(summary: Mapping[str, Any]) -> str:
    title = str(summary.get("title") or "Grid Dynamics Video Error Review")
    models = [dict(row) for row in summary.get("models", []) if isinstance(row, Mapping)]
    dataset = summary.get("dataset_record", {}) if isinstance(summary.get("dataset_record"), Mapping) else {}
    windowing = dataset.get("windowing", {}) if isinstance(dataset.get("windowing"), Mapping) else {}
    limitation_items = list(summary.get("limitations", []))
    cards = []
    for model in models:
        cards.append(
            f"""
<section class="model-card">
  <div class="model-head">
    <div>
      <h2>{_e(model.get('experiment_id'))}</h2>
      <div class="muted">{_e(model.get('model_family'))} · {_e(model.get('dataset_key'))} · target {_e(model.get('prediction_target') or 'n/a')}</div>
    </div>
    <div class="score">{_e(_selection_score_label(model))}</div>
  </div>
  {_review_image_html(model)}
  <dl>
    <dt>Artifact</dt><dd>{_e(model.get('artifact_mode') or 'single_frame')}</dd>
    <dt>Example</dt><dd>{_e(model.get('example_index'))}</dd>
    <dt>Video</dt><dd>{_e(model.get('example_video_id') or 'not stored in this run')}</dd>
    <dt>Selection video</dt><dd>{_e(_selection_video_label(model))}</dd>
    <dt>Split</dt><dd>{_e(model.get('example_split') or 'not stored in this run')}</dd>
    <dt>Target frame</dt><dd>{_e(model.get('target_frame_index') if model.get('target_frame_index') is not None else 'not stored')}</dd>
    <dt>Clip frames</dt><dd>{_e(model.get('clip_frame_count') or 'n/a')}</dd>
    <dt>Clip span</dt><dd>{_e(_clip_span(model))}</dd>
    <dt>MSE</dt><dd>{_fmt(model.get('split_decoded_prediction_mse'))}</dd>
    <dt>Persistence MSE</dt><dd>{_fmt(model.get('split_persistence_mse'))}</dd>
  </dl>
  <p class="hparams">{_e(model.get('hyperparameter_summary') or '')}</p>
</section>"""
        )
    limitations = "".join(f"<li>{_e(item)}</li>" for item in limitation_items) or "<li>No limitations recorded.</li>"
    missing_rows = [dict(row) for row in summary.get("missing_visual_example_rows", []) if isinstance(row, Mapping)]
    missing_html = _missing_visual_rows_html(missing_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_e(title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ margin: 0; background: #f6f7f9; color: #1f2933; }}
header {{ background: #111827; color: white; padding: 28px 32px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
header p {{ margin: 4px 0; color: #d1d5db; }}
main {{ padding: 24px 32px 40px; max-width: 1320px; margin: 0 auto; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.metric {{ background: white; border: 1px solid #d8dee6; border-radius: 8px; padding: 12px 14px; }}
.metric div:first-child {{ color: #5b6675; font-size: 12px; text-transform: uppercase; }}
.metric div:last-child {{ font-size: 18px; margin-top: 4px; }}
.model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
.model-card {{ background: white; border: 1px solid #d8dee6; border-radius: 8px; padding: 14px; overflow: hidden; }}
.model-head {{ display: flex; align-items: start; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
h2 {{ font-size: 15px; margin: 0 0 4px; word-break: break-word; }}
.muted {{ color: #5b6675; font-size: 13px; }}
.score {{ white-space: nowrap; font-variant-numeric: tabular-nums; color: #0f766e; font-size: 13px; }}
img {{ width: 100%; image-rendering: pixelated; border: 1px solid #e5e7eb; background: #111; }}
dl {{ display: grid; grid-template-columns: 120px 1fr; gap: 4px 10px; font-size: 13px; }}
dt {{ color: #5b6675; }} dd {{ margin: 0; word-break: break-word; }}
.hparams {{ color: #374151; font-size: 13px; line-height: 1.45; }}
.limitations {{ margin-top: 20px; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 14px 18px; }}
.limitations h2 {{ font-size: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: left; border-top: 1px solid #fed7aa; padding: 7px 8px; vertical-align: top; }}
</style>
</head>
<body>
<header>
  <h1>{_e(title)}</h1>
  <p>Selection: {_e(summary.get('selection_mode'))} · split {_e(summary.get('split'))} · dataset {_e(summary.get('dataset_key') or 'mixed')}</p>
  <p>Panels show target, model prediction, persistence prediction, model absolute error, persistence absolute error, and model-minus-persistence error. Temporal clips stack consecutive forecast frames vertically when available.</p>
</header>
<main>
  <section class="summary">
    <div class="metric"><div>Models</div><div>{len(models)}</div></div>
    <div class="metric"><div>Temporal clips</div><div>{_e(summary.get('temporal_clip_model_count') or 0)}</div></div>
    <div class="metric"><div>Window frames</div><div>{_e(windowing.get('window_frames') or 'n/a')}</div></div>
    <div class="metric"><div>Horizon frames</div><div>{_e(windowing.get('prediction_horizon_frames') or 'n/a')}</div></div>
    <div class="metric"><div>Frame rate</div><div>{_e(windowing.get('effective_frame_rate_hz') or windowing.get('source_frame_rate_hz') or 'n/a')} Hz</div></div>
  </section>
  <section class="model-grid">
    {''.join(cards) if cards else '<p>No models with prediction examples were available.</p>'}
  </section>
  {missing_html}
  <section class="limitations">
    <h2>Limitations</h2>
    <ul>{limitations}</ul>
  </section>
</main>
</body>
</html>
"""


def _missing_visual_rows_html(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    items = []
    for row in rows:
        items.append(
            f"<tr><td>{_e(row.get('experiment_id'))}</td><td>{_e(row.get('model_family'))}</td><td>{_e(row.get('dataset_key'))}</td><td>{_fmt(row.get('split_improvement_over_persistence_mse'))}</td><td>{_e(row.get('reason'))}</td></tr>"
        )
    return f"""<section class="limitations">
    <h2>Top-ranked rows without visual examples</h2>
    <p>These rows ranked within the requested selection before visual-artifact filtering, but no prediction example or clip artifact was available.</p>
    <table><thead><tr><th>Experiment</th><th>Family</th><th>Dataset</th><th>Improve</th><th>Reason</th></tr></thead><tbody>{''.join(items)}</tbody></table>
  </section>"""


def _review_image_html(model: Mapping[str, Any]) -> str:
    clip_panel = model.get("clip_panel_png")
    if clip_panel:
        name = Path(str(clip_panel)).name
        alt = f"Temporal prediction clip for {model.get('experiment_id')}"
        return f"<img src=\"{_e(name)}\" alt=\"{_e(alt)}\">"
    panel = model.get("panel_png")
    name = Path(str(panel)).name
    alt = f"Prediction error panel for {model.get('experiment_id')}"
    return f"<img src=\"{_e(name)}\" alt=\"{_e(alt)}\">"


def _cleanup_stale_review_panels(out: Path) -> None:
    for pattern in ("model_*_example_*.png", "model_*_clip_*.png"):
        for path in out.glob(pattern):
            if path.is_file():
                path.unlink()


def _artifact_index_for_selection(
    items: Sequence[Mapping[str, Any]],
    *,
    example_index: int,
    selection_video: Mapping[str, Any] | None,
    split: str,
    selection_mode: str,
) -> int:
    fallback = min(max(int(example_index), 0), len(items) - 1)
    if selection_mode == HELDOUT_SELECTION_MODE:
        for index, item in enumerate(items):
            if isinstance(item, Mapping) and str(item.get("split") or "") == str(split):
                return index
        return fallback
    if not selection_video:
        return fallback
    selected_video_id = selection_video.get("video_id")
    if not selected_video_id:
        return fallback
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("video_id") or "") != str(selected_video_id):
            continue
        item_split = item.get("split")
        if item_split and str(item_split) != str(split):
            continue
        return index
    return fallback


def _selection_video_label(model: Mapping[str, Any]) -> str:
    video_id = model.get("selection_video_id")
    if not video_id:
        return "n/a"
    value = _fmt(model.get("selection_metric_value"))
    windows = model.get("selection_video_window_count")
    suffix = f" ({windows} windows)" if windows is not None else ""
    return f"{video_id} improve {value}{suffix}"


def _clip_span(model: Mapping[str, Any]) -> str:
    start = model.get("clip_start_target_frame_index")
    end = model.get("clip_end_target_frame_index")
    if start is None or end is None:
        return "n/a"
    return f"{start} to {end}"



def _selection_metric_name(selection_mode: str) -> str:
    if selection_mode == "best_active_cell":
        return "active_cell_improvement_over_persistence_mse"
    if selection_mode == "most_improved_video":
        return "best_video_improvement_over_persistence_mse"
    if selection_mode == "least_improved_video":
        return "worst_video_improvement_over_persistence_mse"
    return "improvement_over_persistence_mse"


def _selection_metric_value(row: Mapping[str, Any], *, split: str, selection_mode: str) -> float | None:
    if selection_mode in VIDEO_SELECTION_MODES:
        item = _selection_video_item(row, split=split, selection_mode=selection_mode)
        return _num(item.get("improvement_over_persistence_mse")) if item else None
    return _metric(row, split, _selection_metric_name(selection_mode))


def _selection_video_item(row: Mapping[str, Any], *, split: str, selection_mode: str) -> dict[str, Any] | None:
    if selection_mode not in VIDEO_SELECTION_MODES:
        return None
    summary = row.get("video_error_summary", {})
    if not isinstance(summary, Mapping):
        return None
    split_summary = summary.get(split, {})
    if not isinstance(split_summary, Mapping):
        return None
    key = "best_videos" if selection_mode == "most_improved_video" else "worst_videos"
    videos = split_summary.get(key, [])
    if not isinstance(videos, Sequence) or isinstance(videos, (str, bytes)):
        return None
    for item in videos:
        if isinstance(item, Mapping) and _num(item.get("improvement_over_persistence_mse")) is not None:
            return dict(item)
    return None


def _selection_score_label(model: Mapping[str, Any]) -> str:
    metric = str(model.get("selection_metric_name") or "improvement_over_persistence_mse")
    value = model.get("selection_metric_value")
    if metric == "active_cell_improvement_over_persistence_mse":
        return f"active-cell improve {_fmt(value)}"
    if metric == "best_video_improvement_over_persistence_mse":
        return f"best-video improve {_fmt(value)}"
    if metric == "worst_video_improvement_over_persistence_mse":
        return f"worst-video improve {_fmt(value)}"
    return f"improve {_fmt(value)}"


def _select_rows(rows: Sequence[Mapping[str, Any]], *, selection_mode: str, split: str, max_models: int) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in rows if _is_learned_row(row)]
    if not candidates:
        candidates = [dict(row) for row in rows]
    if selection_mode == "best_test":
        selected = sorted(candidates, key=lambda row: _sort_value(_metric(row, "test", "improvement_over_persistence_mse")), reverse=True)
    elif selection_mode == "best_val":
        selected = sorted(candidates, key=lambda row: _sort_value(_metric(row, "val", "improvement_over_persistence_mse")), reverse=True)
    elif selection_mode == "worst_over_persistence":
        selected = sorted(candidates, key=lambda row: _sort_value(_metric(row, split, "improvement_over_persistence_mse")))
    elif selection_mode == "best_active_cell":
        selected = sorted(candidates, key=lambda row: _sort_value(_metric(row, split, "active_cell_improvement_over_persistence_mse")), reverse=True)
    elif selection_mode == HELDOUT_SELECTION_MODE:
        selected = sorted(candidates, key=lambda row: _sort_value(_metric(row, "test", "improvement_over_persistence_mse")), reverse=True)
    elif selection_mode in VIDEO_SELECTION_MODES:
        evidence_rows = [row for row in candidates if _selection_metric_value(row, split=split, selection_mode=selection_mode) is not None]
        if selection_mode == "most_improved_video":
            selected = sorted(evidence_rows, key=lambda row: (-float(_selection_metric_value(row, split=split, selection_mode=selection_mode) or 0.0), str(row.get("experiment_id") or "")))
        else:
            selected = sorted(evidence_rows, key=lambda row: (float(_selection_metric_value(row, split=split, selection_mode=selection_mode) or 0.0), str(row.get("experiment_id") or "")))
    elif selection_mode == "best_by_family":
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_family[str(row.get("model_family") or row.get("kind") or "unknown")].append(row)
        selected = []
        for family in sorted(by_family):
            best = sorted(by_family[family], key=lambda row: _sort_value(_metric(row, split, "improvement_over_persistence_mse")), reverse=True)[0]
            selected.append(best)
        selected = sorted(selected, key=lambda row: _sort_value(_metric(row, split, "improvement_over_persistence_mse")), reverse=True)
    else:
        raise ValueError("selection_mode must be best_by_family, best_test, best_val, best_active_cell, most_improved_video, least_improved_video, heldout_first, or worst_over_persistence")
    return selected[: max(int(max_models), 0)]


def _is_learned_row(row: Mapping[str, Any]) -> bool:
    labels = {str(row.get("model_family") or ""), str(row.get("kind") or "")}
    if labels & LEARNED_FAMILIES:
        return True
    return "pixel_convgru" in labels or "pixel_convlstm" in labels or "pixel_temporal_cnn" in labels


def _write_error_panel(path: Path, example: Mapping[str, Any]) -> dict[str, Any]:
    target = _array(example.get("target_next"))
    pred = _array(example.get("predicted_next"))
    persistence = _array(example.get("input_last"))
    if target.shape != pred.shape or target.shape != persistence.shape:
        raise ValueError("Prediction example target, prediction, and persistence arrays must have the same shape.")
    model_error = np.abs(target - pred)
    persistence_error = np.abs(target - persistence)
    improvement = model_error - persistence_error
    panels = [target, pred, persistence, model_error, persistence_error, improvement]
    normalized = [_normalize_panel(arr) for arr in panels]
    gap = np.ones((target.shape[0], 2), dtype=np.float32)
    canvas_parts: list[np.ndarray] = []
    for index, panel in enumerate(normalized):
        if index:
            canvas_parts.append(gap)
        canvas_parts.append(panel)
    canvas = np.concatenate(canvas_parts, axis=1)
    write_png_gray8(path, int(canvas.shape[1]), int(canvas.shape[0]), normalize_array_frame(canvas))
    return {"height": int(canvas.shape[0]), "width": int(canvas.shape[1]), "panel_count": 6}


def _write_clip_panel(path: Path, clip: Mapping[str, Any]) -> dict[str, Any]:
    frames = [dict(frame) for frame in clip.get("frames", []) if isinstance(frame, Mapping)]
    if not frames:
        raise ValueError("Prediction clip artifact must contain at least one frame.")
    rows = []
    for frame in frames:
        target = _array(frame.get("target_next"))
        pred = _array(frame.get("predicted_next"))
        persistence = _array(frame.get("persistence_next", frame.get("input_last")))
        if target.shape != pred.shape or target.shape != persistence.shape:
            raise ValueError("Prediction clip target, prediction, and persistence arrays must have the same shape.")
        model_error = np.abs(target - pred)
        persistence_error = np.abs(target - persistence)
        improvement = model_error - persistence_error
        panels = [_normalize_panel(arr) for arr in (target, pred, persistence, model_error, persistence_error, improvement)]
        gap = np.ones((target.shape[0], 2), dtype=np.float32)
        row_parts: list[np.ndarray] = []
        for index, panel in enumerate(panels):
            if index:
                row_parts.append(gap)
            row_parts.append(panel)
        rows.append(np.concatenate(row_parts, axis=1))
    row_gap = np.ones((2, rows[0].shape[1]), dtype=np.float32)
    canvas_parts: list[np.ndarray] = []
    for index, row in enumerate(rows):
        if index:
            canvas_parts.append(row_gap)
        canvas_parts.append(row)
    canvas = np.concatenate(canvas_parts, axis=0)
    write_png_gray8(path, int(canvas.shape[1]), int(canvas.shape[0]), normalize_array_frame(canvas))
    return {"height": int(canvas.shape[0]), "width": int(canvas.shape[1]), "panel_count": 6, "frame_count": len(rows)}


def _normalize_panel(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr, dtype=np.float32)


def _array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Prediction example arrays must be 2-D frames.")
    return arr


def _load_examples(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        examples = payload.get("examples", [])
        return [dict(item) for item in examples if isinstance(item, Mapping)]
    return []


def _load_clips(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        clips = payload.get("clips", [])
        return [dict(item) for item in clips if isinstance(item, Mapping)]
    return []


def _first_clip_frame(clip: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(clip, Mapping):
        return None
    frames = clip.get("frames", [])
    if not isinstance(frames, Sequence):
        return None
    for frame in frames:
        if isinstance(frame, Mapping):
            return dict(frame)
    return None


def _examples_path(row: Mapping[str, Any]) -> Path:
    explicit = row.get("prediction_examples_path")
    if explicit:
        return Path(str(explicit))
    metrics = row.get("metrics_path")
    if not metrics:
        return Path("__missing_prediction_examples__")
    return Path(str(metrics)).parent / "prediction_examples.json"


def _clip_examples_path(row: Mapping[str, Any]) -> Path:
    explicit = row.get("prediction_clip_examples_path")
    if explicit:
        return Path(str(explicit))
    metrics = row.get("metrics_path")
    if not metrics:
        return Path("__missing_prediction_clip_examples__")
    return Path(str(metrics)).parent / "prediction_clip_examples.json"


def _has_visual_examples(row: Mapping[str, Any]) -> bool:
    return _examples_path(row).exists() or _clip_examples_path(row).exists()


def _missing_visual_summary(row: Mapping[str, Any], *, split: str, selection_mode: str = "best_test") -> dict[str, Any]:
    examples_path = _examples_path(row)
    clips_path = _clip_examples_path(row)
    return {
        "experiment_id": row.get("experiment_id"),
        "row_id": row.get("row_id"),
        "kind": row.get("kind"),
        "model_family": row.get("model_family"),
        "dataset_key": row.get("dataset_key"),
        "prediction_target": row.get("prediction_target"),
        "metrics_path": row.get("metrics_path"),
        "prediction_examples_path": str(examples_path),
        "prediction_clip_examples_path": str(clips_path),
        "split_improvement_over_persistence_mse": _metric(row, split, "improvement_over_persistence_mse"),
        "selection_metric_name": _selection_metric_name(selection_mode),
        "selection_metric_value": _selection_metric_value(row, split=split, selection_mode=selection_mode),
        "reason": "missing prediction_examples.json and prediction_clip_examples.json",
    }


def _dominant_dataset(models: Sequence[Mapping[str, Any]]) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for model in models:
        key = model.get("dataset_key")
        if key:
            counts[str(key)] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _limitations(models: Sequence[Mapping[str, Any]], *, missing_visual_rows: Sequence[Mapping[str, Any]] = (), selection_mode: str = "") -> list[str]:
    limits = []
    if selection_mode in VIDEO_SELECTION_MODES and not models and not missing_visual_rows:
        limits.append("No rows with visual examples had per-video prediction diagnostics for the requested split.")
    if missing_visual_rows:
        limits.append("Some top-ranked rows were omitted from the visual cards because they do not yet have prediction_examples.json or prediction_clip_examples.json artifacts.")
    if any(not model.get("example_video_id") for model in models):
        limits.append("Older prediction example artifacts do not store video IDs; regenerate runs through the updated writers for exact video metadata.")
    if any(not model.get("example_split") for model in models):
        limits.append("Older prediction example artifacts do not store split labels; current review uses saved example index alignment.")
    if any(model.get("artifact_mode") != "temporal_clip" for model in models):
        limits.append("Some selected runs only provide saved representative frames, not full temporal clips.")
    if models and all(model.get("artifact_mode") == "temporal_clip" for model in models):
        limits.append("Temporal clip panels are still sampled inspection artifacts, not exhaustive per-video movies.")
    return limits


def _metric(row: Mapping[str, Any], split: str, metric_name: str) -> float | None:
    if split == "all":
        keys = [f"all_{metric_name}", metric_name]
    else:
        keys = [f"{split}_{metric_name}", metric_name]
    for key in keys:
        value = _num(row.get(key))
        if value is not None:
            return value
    return None


def _sort_value(value: Any) -> float:
    num = _num(value)
    return float(num) if num is not None else float("-inf")


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _safe_name(value: Any) -> str:
    text = str(value or "model")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:160]


def _fmt(value: Any) -> str:
    num = _num(value)
    if num is None:
        return "n/a"
    return f"{num:.4g}"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
