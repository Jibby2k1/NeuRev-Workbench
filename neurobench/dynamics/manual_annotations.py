"""Manual ROI and spike annotation import/evaluation helpers."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"a": XLSX_MAIN_NS, "r": XLSX_REL_NS}
FRAME_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def import_manual_roi_spikes(
    *,
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    grid_size: int = 128,
    crop_size: int = 512,
    frame_rate_hz: float = 50.0,
    title: str = "Manual ROI Spike Annotations",
) -> dict[str, Any]:
    """Import compact Excel ROI/spike workbooks into JSON/TSV/HTML artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    annotations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    workbook_summaries: list[dict[str, Any]] = []
    for path_like in inputs:
        path = Path(path_like)
        rows = _read_first_sheet_rows(path)
        workbook_title = str(rows[0][0]).strip() if rows and rows[0] else ""
        video_id = _video_id_from_path(path)
        if workbook_title and _normalize_label(_video_id_from_path(Path(workbook_title))) != _normalize_label(video_id):
            warnings.append(
                {
                    "level": "warning",
                    "file": str(path),
                    "kind": "title_mismatch",
                    "message": f"Workbook title {workbook_title!r} maps to a different video than file stem {path.stem!r}.",
                }
            )
        workbook_count = 0
        for sheet_row_number, row in enumerate(rows[2:], start=3):
            if not row or not str(row[0]).strip():
                continue
            roi_id = str(row[0]).strip()
            try:
                x = float(row[1])
                y = float(row[2])
            except (IndexError, TypeError, ValueError):
                warnings.append(
                    {
                        "level": "warning",
                        "file": str(path),
                        "row": sheet_row_number,
                        "kind": "invalid_coordinate",
                        "message": f"Skipping ROI {roi_id!r}: x/y coordinates are missing or invalid.",
                    }
                )
                continue
            spike_intervals: list[dict[str, Any]] = []
            for col_number, value in enumerate(row[4:], start=5):
                text = str(value).strip()
                if not text:
                    continue
                match = FRAME_RANGE_RE.match(text)
                if not match:
                    warnings.append(
                        {
                            "level": "warning",
                            "file": str(path),
                            "row": sheet_row_number,
                            "column": col_number,
                            "roi_id": roi_id,
                            "kind": "invalid_frame_range",
                            "value": text,
                            "message": "Frame range is not in start-end form.",
                        }
                    )
                    continue
                start = int(match.group(1))
                end = int(match.group(2))
                if end < start:
                    warnings.append(
                        {
                            "level": "warning",
                            "file": str(path),
                            "row": sheet_row_number,
                            "column": col_number,
                            "roi_id": roi_id,
                            "kind": "reversed_frame_range",
                            "value": text,
                            "message": "Frame range end is before start; interval was not imported.",
                        }
                    )
                    continue
                duration = int(end - start + 1)
                if duration > 300:
                    warnings.append(
                        {
                            "level": "warning",
                            "file": str(path),
                            "row": sheet_row_number,
                            "column": col_number,
                            "roi_id": roi_id,
                            "kind": "unusually_long_frame_range",
                            "value": text,
                            "duration_frames": duration,
                            "message": "Frame range is unusually long and may be a typo; interval was imported but should be reviewed.",
                        }
                    )
                spike_intervals.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                        "center_frame": int(round((start + end) / 2.0)),
                        "duration_frames": duration,
                        "start_sec": float(start / frame_rate_hz),
                        "end_sec": float(end / frame_rate_hz),
                        "source_column": int(col_number),
                    }
                )
            grid_x = _coord_to_grid_index(x, crop_size=crop_size, grid_size=grid_size)
            grid_y = _coord_to_grid_index(y, crop_size=crop_size, grid_size=grid_size)
            annotations.append(
                {
                    "annotation_id": f"{video_id}:roi_{roi_id}",
                    "video_id": video_id,
                    "source_file": str(path),
                    "source_sheet_title": workbook_title,
                    "source_row_number": int(sheet_row_number),
                    "roi_id": roi_id,
                    "crop_x": float(x),
                    "crop_y": float(y),
                    "grid_col": int(grid_x),
                    "grid_row": int(grid_y),
                    "grid_size": int(grid_size),
                    "crop_size": int(crop_size),
                    "mean_plus_sd": _optional_float(row[3] if len(row) > 3 else ""),
                    "spike_intervals": spike_intervals,
                    "spike_interval_count": len(spike_intervals),
                    "spike_frame_count": int(sum(item["duration_frames"] for item in spike_intervals)),
                }
            )
            workbook_count += 1
        workbook_summaries.append(
            {
                "source_file": str(path),
                "video_id": video_id,
                "source_sheet_title": workbook_title,
                "roi_count": int(workbook_count),
            }
        )
    video_counts = defaultdict(int)
    interval_counts = defaultdict(int)
    for row in annotations:
        video_counts[str(row["video_id"])] += 1
        interval_counts[str(row["video_id"])] += int(row["spike_interval_count"])
    manifest = {
        "schema_version": 1,
        "title": str(title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotation_count": len(annotations),
        "spike_interval_count": int(sum(row["spike_interval_count"] for row in annotations)),
        "grid_size": int(grid_size),
        "crop_size": int(crop_size),
        "frame_rate_hz": float(frame_rate_hz),
        "workbooks": workbook_summaries,
        "video_roi_counts": dict(sorted(video_counts.items())),
        "video_spike_interval_counts": dict(sorted(interval_counts.items())),
        "warnings": warnings,
        "annotations": annotations,
    }
    manifest_path = out / "manual_roi_spike_annotations.json"
    tsv_path = out / "manual_roi_spike_annotations.tsv"
    intervals_path = out / "manual_roi_spike_intervals.tsv"
    html_path = out / "manual_roi_spike_annotations.html"
    manifest["manifest_path"] = str(manifest_path)
    manifest["roi_tsv_path"] = str(tsv_path)
    manifest["interval_tsv_path"] = str(intervals_path)
    manifest["html_path"] = str(html_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv_path.write_text(_roi_tsv(annotations), encoding="utf-8")
    intervals_path.write_text(_interval_tsv(annotations), encoding="utf-8")
    html_path.write_text(render_manual_roi_annotations_html(manifest), encoding="utf-8")
    return manifest


def evaluate_manual_roi_spikes_on_dataset(
    *,
    dataset: Mapping[str, Any] | str | Path,
    annotations: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    event_margin_frames: int = 0,
    title: str = "Manual ROI Spike Dataset Evaluation",
) -> dict[str, Any]:
    """Evaluate annotated ROI/event targets against persistence in a dynamics dataset."""
    dataset_payload = _load_json(dataset) if not isinstance(dataset, Mapping) else dict(dataset)
    annotation_payload = _load_json(annotations) if not isinstance(annotations, Mapping) else dict(annotations)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(str(dataset_payload["array_path"]), allow_pickle=False)
    targets = arrays["targets"].astype(np.float32)
    windows = arrays["windows"].astype(np.float32)
    video_ids = arrays["window_video_ids"].astype(str)
    target_frame_indices = arrays["target_frame_indices"] if "target_frame_indices" in arrays.files else _infer_target_indices(video_ids)
    splits = dataset_payload.get("splits", {}) if isinstance(dataset_payload.get("splits"), Mapping) else {}
    last = windows[:, -1]
    rows: list[dict[str, Any]] = []
    for roi in annotation_payload.get("annotations", []) or []:
        if not isinstance(roi, Mapping):
            continue
        mask_video = video_ids == str(roi.get("video_id"))
        grid_row = int(roi.get("grid_row"))
        grid_col = int(roi.get("grid_col"))
        if not (0 <= grid_row < targets.shape[-2] and 0 <= grid_col < targets.shape[-1]):
            continue
        roi_targets = targets[:, 0, grid_row, grid_col]
        roi_last = last[:, 0, grid_row, grid_col]
        event_mask = np.zeros(targets.shape[0], dtype=bool)
        for interval in roi.get("spike_intervals", []) or []:
            if not isinstance(interval, Mapping):
                continue
            start = int(interval.get("start_frame"))
            end = int(interval.get("end_frame"))
            event_mask |= mask_video & (target_frame_indices >= start - int(event_margin_frames)) & (target_frame_indices <= end + int(event_margin_frames))
        baseline_sq = np.square(roi_last[event_mask] - roi_targets[event_mask]) if np.any(event_mask) else np.asarray([], dtype=np.float32)
        all_mask = mask_video
        rows.append(
            {
                "annotation_id": roi.get("annotation_id"),
                "video_id": roi.get("video_id"),
                "split": _split_for_video(str(roi.get("video_id")), splits),
                "roi_id": roi.get("roi_id"),
                "grid_row": grid_row,
                "grid_col": grid_col,
                "spike_interval_count": int(roi.get("spike_interval_count") or 0),
                "matched_event_window_count": int(event_mask.sum()),
                "video_window_count": int(all_mask.sum()),
                "event_target_mean": _mean_or_none(roi_targets[event_mask]),
                "event_target_max": _max_or_none(roi_targets[event_mask]),
                "event_persistence_mse": _mean_or_none(baseline_sq),
                "video_target_mean": _mean_or_none(roi_targets[all_mask]),
                "video_target_max": _max_or_none(roi_targets[all_mask]),
            }
        )
    summary = _summarize_annotation_eval(rows)
    payload = {
        "schema_version": 1,
        "title": str(title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset if not isinstance(dataset, Mapping) else ""),
        "annotation_manifest_path": str(annotations if not isinstance(annotations, Mapping) else ""),
        "dataset_id": dataset_payload.get("dataset_id"),
        "array_path": dataset_payload.get("array_path"),
        "event_margin_frames": int(event_margin_frames),
        "roi_count": len(rows),
        "summary": summary,
        "rows": rows,
        "limitations": [
            "This report scores annotated target cells and persistence only; model-specific annotation scores require checkpoint prediction inference.",
            "ROI coordinates are point centroids projected from crop512 pixels to single grid128 cells.",
        ],
    }
    json_path = out / "manual_roi_spike_dataset_evaluation.json"
    md_path = out / "manual_roi_spike_dataset_evaluation.md"
    html_path = out / "manual_roi_spike_dataset_evaluation.html"
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    payload["html_path"] = str(html_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_manual_roi_dataset_evaluation_markdown(payload), encoding="utf-8")
    html_path.write_text(render_manual_roi_dataset_evaluation_html(payload), encoding="utf-8")
    return payload




def score_spatial_checkpoints_on_manual_roi_spikes(
    *,
    dataset: Mapping[str, Any] | str | Path,
    annotations: Mapping[str, Any] | str | Path,
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    device: str = "cuda",
    batch_size: int = 2,
    event_margin_frames: int = 0,
    title: str = "Manual ROI Spike Model Scores",
) -> dict[str, Any]:
    """Score spatial pixel checkpoints on annotated ROI/event windows."""
    from neurobench.dynamics.concept_tests import _build_spatial_pixel_model, _predict_convgru_pixel, _torch

    torch = _torch()
    dataset_payload = _load_json(dataset) if not isinstance(dataset, Mapping) else dict(dataset)
    annotation_payload = _load_json(annotations) if not isinstance(annotations, Mapping) else dict(annotations)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(str(dataset_payload["array_path"]), allow_pickle=False)
    video_ids = arrays["window_video_ids"].astype(str)
    target_frame_indices = arrays["target_frame_indices"] if "target_frame_indices" in arrays.files else _infer_target_indices(video_ids)
    event_indices = _manual_event_window_indices(video_ids, target_frame_indices, annotation_payload, margin_frames=int(event_margin_frames))
    if event_indices.size == 0:
        raise ValueError("No dataset windows matched the manual ROI spike intervals.")
    windows_all = np.asarray(arrays["windows"], dtype=np.float32)
    targets_all = np.asarray(arrays["targets"], dtype=np.float32)
    windows = np.clip(np.nan_to_num(windows_all[event_indices], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    targets = np.clip(np.nan_to_num(targets_all[event_indices], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    last = windows[:, -1]
    selected_video_ids = video_ids[event_indices]
    selected_target_frames = target_frame_indices[event_indices]
    del windows_all, targets_all
    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_dir_like in run_dirs:
        run_dir = Path(run_dir_like)
        checkpoint_path = run_dir / "concept_checkpoint.pt"
        metrics_path = run_dir / "concept_metrics.json"
        ckpt = torch.load(checkpoint_path, map_location=device)
        architecture = str(ckpt.get("architecture") or "").strip()
        model = _build_spatial_pixel_model(
            architecture=architecture,
            input_channels=int(ckpt.get("input_channels") or windows.shape[2]),
            window_frames=int(ckpt.get("window_frames") or windows.shape[1]),
            hidden_channels=int(ckpt.get("hidden_channels") or ckpt.get("hidden_dim") or 32),
            num_layers=int(ckpt.get("num_layers") or 1),
            residual_scale=float(ckpt.get("residual_scale") if ckpt.get("residual_scale") is not None else 0.25),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        xw = torch.from_numpy(windows).to(device)
        pred = _predict_convgru_pixel(model, xw, batch_size=int(batch_size))
        del xw, model
        if str(device).startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        run_rows = _score_prediction_on_manual_rois(
            pred=pred,
            targets=targets,
            last=last,
            event_indices=event_indices,
            selected_video_ids=selected_video_ids,
            selected_target_frames=selected_target_frames,
            annotations=annotation_payload,
            splits=dataset_payload.get("splits", {}),
            run_dir=run_dir,
            metrics=metrics,
            event_margin_frames=int(event_margin_frames),
        )
        rows.extend(run_rows)
        model_values = [row["event_model_mse"] for row in run_rows if row.get("event_model_mse") is not None]
        persist_values = [row["event_persistence_mse"] for row in run_rows if row.get("event_persistence_mse") is not None]
        improvements = [row["event_improvement_over_persistence_mse"] for row in run_rows if row.get("event_improvement_over_persistence_mse") is not None]
        run_summaries.append(
            {
                "run_dir": str(run_dir),
                "experiment_id": run_dir.parent.name,
                "architecture": architecture,
                "roi_count": len(run_rows),
                "event_window_count": int(sum(row.get("matched_event_window_count") or 0 for row in run_rows)),
                "mean_event_model_mse": float(np.mean(model_values)) if model_values else None,
                "mean_event_persistence_mse": float(np.mean(persist_values)) if persist_values else None,
                "mean_event_improvement_over_persistence_mse": float(np.mean(improvements)) if improvements else None,
                "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            }
        )
        del pred
    run_summaries.sort(key=lambda row: row.get("mean_event_improvement_over_persistence_mse") if row.get("mean_event_improvement_over_persistence_mse") is not None else float("-inf"), reverse=True)
    payload = {
        "schema_version": 1,
        "title": str(title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset if not isinstance(dataset, Mapping) else ""),
        "annotation_manifest_path": str(annotations if not isinstance(annotations, Mapping) else ""),
        "dataset_id": dataset_payload.get("dataset_id"),
        "array_path": dataset_payload.get("array_path"),
        "device": str(device),
        "batch_size": int(batch_size),
        "event_margin_frames": int(event_margin_frames),
        "union_event_window_count": int(event_indices.size),
        "run_count": len(run_summaries),
        "run_summaries": run_summaries,
        "rows": rows,
        "limitations": [
            "Spatial checkpoint scoring uses single grid128 cells projected from ROI centroids.",
            "Scores cover annotated event windows only, not full-video behavior generalization.",
        ],
    }
    json_path = out / "manual_roi_spike_model_scores.json"
    md_path = out / "manual_roi_spike_model_scores.md"
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_manual_roi_model_scores_markdown(payload), encoding="utf-8")
    return payload


def render_manual_roi_model_scores_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload.get('title')}",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"Device: `{payload.get('device')}`",
        f"Batch size: `{payload.get('batch_size')}`",
        f"Union event windows: `{payload.get('union_event_window_count')}`",
        "",
        "## Run Ranking",
        "",
        "| Rank | Experiment | Architecture | ROIs | Event windows | Model MSE | Persistence MSE | Improvement |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(payload.get("run_summaries", []) or [], start=1):
        lines.append(
            f"| {rank} | `{row.get('experiment_id')}` | {row.get('architecture')} | {row.get('roi_count')} | {row.get('event_window_count')} | {_fmt(row.get('mean_event_model_mse'))} | {_fmt(row.get('mean_event_persistence_mse'))} | {_fmt(row.get('mean_event_improvement_over_persistence_mse'))} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload.get("limitations", []) or [])
    return "\n".join(lines).rstrip() + "\n"


def _manual_event_window_indices(video_ids: np.ndarray, target_frame_indices: np.ndarray, annotations: Mapping[str, Any], *, margin_frames: int) -> np.ndarray:
    mask = np.zeros(video_ids.shape[0], dtype=bool)
    for roi in annotations.get("annotations", []) or []:
        if not isinstance(roi, Mapping):
            continue
        video_mask = video_ids == str(roi.get("video_id"))
        for interval in roi.get("spike_intervals", []) or []:
            if not isinstance(interval, Mapping):
                continue
            start = int(interval.get("start_frame")) - int(margin_frames)
            end = int(interval.get("end_frame")) + int(margin_frames)
            mask |= video_mask & (target_frame_indices >= start) & (target_frame_indices <= end)
    return np.flatnonzero(mask).astype(np.int64)


def _score_prediction_on_manual_rois(
    *,
    pred: np.ndarray,
    targets: np.ndarray,
    last: np.ndarray,
    event_indices: np.ndarray,
    selected_video_ids: np.ndarray,
    selected_target_frames: np.ndarray,
    annotations: Mapping[str, Any],
    splits: Mapping[str, Any],
    run_dir: Path,
    metrics: Mapping[str, Any],
    event_margin_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for roi in annotations.get("annotations", []) or []:
        if not isinstance(roi, Mapping):
            continue
        grid_row = int(roi.get("grid_row"))
        grid_col = int(roi.get("grid_col"))
        roi_mask = selected_video_ids == str(roi.get("video_id"))
        interval_mask = np.zeros(event_indices.shape[0], dtype=bool)
        for interval in roi.get("spike_intervals", []) or []:
            if not isinstance(interval, Mapping):
                continue
            start = int(interval.get("start_frame")) - int(event_margin_frames)
            end = int(interval.get("end_frame")) + int(event_margin_frames)
            interval_mask |= roi_mask & (selected_target_frames >= start) & (selected_target_frames <= end)
        model_sq = np.square(pred[interval_mask, 0, grid_row, grid_col] - targets[interval_mask, 0, grid_row, grid_col]) if np.any(interval_mask) else np.asarray([], dtype=np.float32)
        persist_sq = np.square(last[interval_mask, 0, grid_row, grid_col] - targets[interval_mask, 0, grid_row, grid_col]) if np.any(interval_mask) else np.asarray([], dtype=np.float32)
        model_mse = _mean_or_none(model_sq)
        persist_mse = _mean_or_none(persist_sq)
        rows.append(
            {
                "run_dir": str(run_dir),
                "experiment_id": run_dir.parent.name,
                "model_family": metrics.get("model_family"),
                "annotation_id": roi.get("annotation_id"),
                "video_id": roi.get("video_id"),
                "split": _split_for_video(str(roi.get("video_id")), splits),
                "roi_id": roi.get("roi_id"),
                "grid_row": grid_row,
                "grid_col": grid_col,
                "matched_event_window_count": int(interval_mask.sum()),
                "event_model_mse": model_mse,
                "event_persistence_mse": persist_mse,
                "event_improvement_over_persistence_mse": (float(persist_mse - model_mse) if model_mse is not None and persist_mse is not None else None),
            }
        )
    return rows



def score_latent_sequence_runs_on_manual_roi_spikes(
    *,
    annotations: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    latent_runs: Sequence[Mapping[str, Any]] | None = None,
    hybrid_runs: Sequence[str | Path] | None = None,
    left_dataset: Mapping[str, Any] | str | Path | None = None,
    right_dataset: Mapping[str, Any] | str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    event_margin_frames: int = 0,
    title: str = "Manual ROI Spike Latent Sequence Scores",
) -> dict[str, Any]:
    """Score latent GRU and shared directional hybrid runs on manual event windows."""
    annotation_payload = _load_json(annotations) if not isinstance(annotations, Mapping) else dict(annotations)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for spec in latent_runs or []:
        dataset_payload = _load_json(spec["dataset"])
        run_dir = Path(str(spec["run_dir"]))
        subset = _load_manual_event_subset(dataset_payload, annotation_payload, margin_frames=int(event_margin_frames))
        if subset["windows"].shape[0] == 0:
            continue
        pred = _predict_latent_rnn_subset(run_dir=run_dir, windows=subset["windows"], device=device, batch_size=int(batch_size))
        run_rows = _score_prediction_on_manual_rois(
            pred=pred,
            targets=subset["targets"],
            last=subset["windows"][:, -1],
            event_indices=subset["event_indices"],
            selected_video_ids=subset["video_ids"],
            selected_target_frames=subset["target_frame_indices"],
            annotations=annotation_payload,
            splits=dataset_payload.get("splits", {}),
            run_dir=run_dir,
            metrics=_load_json(run_dir / "latent_rnn_metrics.json") if (run_dir / "latent_rnn_metrics.json").exists() else {},
            event_margin_frames=int(event_margin_frames),
        )
        rows.extend(run_rows)
        run_summaries.append(_model_score_summary(run_dir=run_dir, architecture="latent_gru_predictor", rows=run_rows))
        del pred
    if hybrid_runs:
        if left_dataset is None or right_dataset is None:
            raise ValueError("left_dataset and right_dataset are required when scoring hybrid_runs.")
        left_payload = _load_json(left_dataset) if not isinstance(left_dataset, Mapping) else dict(left_dataset)
        right_payload = _load_json(right_dataset) if not isinstance(right_dataset, Mapping) else dict(right_dataset)
        hybrid_subset = _load_hybrid_manual_event_subset(left_payload, right_payload, annotation_payload, margin_frames=int(event_margin_frames))
        for run_dir_like in hybrid_runs:
            run_dir = Path(run_dir_like)
            if hybrid_subset["windows"].shape[0] == 0:
                continue
            pred = _predict_shared_hybrid_subset(run_dir=run_dir, windows=hybrid_subset["windows"], direction_ids=hybrid_subset["direction_ids"], device=device, batch_size=int(batch_size))
            run_rows = _score_prediction_on_manual_rois(
                pred=pred,
                targets=hybrid_subset["targets"],
                last=hybrid_subset["windows"][:, -1],
                event_indices=hybrid_subset["event_indices"],
                selected_video_ids=hybrid_subset["video_ids"],
                selected_target_frames=hybrid_subset["target_frame_indices"],
                annotations=annotation_payload,
                splits=hybrid_subset["splits"],
                run_dir=run_dir,
                metrics=_load_json(run_dir / "hybrid_rnn_metrics.json") if (run_dir / "hybrid_rnn_metrics.json").exists() else {},
                event_margin_frames=int(event_margin_frames),
            )
            rows.extend(run_rows)
            run_summaries.append(_model_score_summary(run_dir=run_dir, architecture="shared_directional_hybrid_gru", rows=run_rows))
            del pred
    run_summaries.sort(key=lambda row: row.get("mean_event_improvement_over_persistence_mse") if row.get("mean_event_improvement_over_persistence_mse") is not None else float("-inf"), reverse=True)
    payload = {
        "schema_version": 1,
        "title": str(title),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotation_manifest_path": str(annotations if not isinstance(annotations, Mapping) else ""),
        "device": str(device),
        "batch_size": int(batch_size),
        "event_margin_frames": int(event_margin_frames),
        "run_count": len(run_summaries),
        "run_summaries": run_summaries,
        "rows": rows,
        "limitations": [
            "Latent sequence scores decode one-step predictions back to grid space and score single centroid cells for manual ROIs.",
            "Directional runs are scored only on annotations present in their source dataset; opposite-direction labels appear only in shared hybrid scoring.",
        ],
    }
    json_path = out / "manual_roi_spike_latent_sequence_scores.json"
    md_path = out / "manual_roi_spike_latent_sequence_scores.md"
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_manual_roi_model_scores_markdown(payload), encoding="utf-8")
    return payload


def _load_manual_event_subset(dataset_payload: Mapping[str, Any], annotation_payload: Mapping[str, Any], *, margin_frames: int) -> dict[str, Any]:
    arrays = np.load(str(dataset_payload["array_path"]), allow_pickle=False)
    video_ids = arrays["window_video_ids"].astype(str)
    target_frame_indices = arrays["target_frame_indices"] if "target_frame_indices" in arrays.files else _infer_target_indices(video_ids)
    event_indices = _manual_event_window_indices(video_ids, target_frame_indices, annotation_payload, margin_frames=int(margin_frames))
    windows = np.clip(np.nan_to_num(arrays["windows"][event_indices].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    targets = np.clip(np.nan_to_num(arrays["targets"][event_indices].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    return {
        "windows": windows,
        "targets": targets,
        "video_ids": video_ids[event_indices],
        "target_frame_indices": target_frame_indices[event_indices],
        "event_indices": event_indices,
    }


def _load_hybrid_manual_event_subset(left_payload: Mapping[str, Any], right_payload: Mapping[str, Any], annotation_payload: Mapping[str, Any], *, margin_frames: int) -> dict[str, Any]:
    left = _load_manual_event_subset(left_payload, annotation_payload, margin_frames=int(margin_frames))
    right = _load_manual_event_subset(right_payload, annotation_payload, margin_frames=int(margin_frames))
    windows = np.concatenate([left["windows"], right["windows"]], axis=0) if left["windows"].size or right["windows"].size else np.zeros((0, 8, 1, 128, 128), dtype=np.float32)
    targets = np.concatenate([left["targets"], right["targets"]], axis=0) if left["targets"].size or right["targets"].size else np.zeros((0, 1, 128, 128), dtype=np.float32)
    video_ids = np.concatenate([left["video_ids"], right["video_ids"]]).astype("U64")
    target_frame_indices = np.concatenate([left["target_frame_indices"], right["target_frame_indices"]]).astype(np.int64)
    event_indices = np.arange(video_ids.shape[0], dtype=np.int64)
    direction_ids = np.concatenate([np.zeros(left["windows"].shape[0], dtype=np.int64), np.ones(right["windows"].shape[0], dtype=np.int64)])
    splits = {
        "split_unit": "video",
        "split_method": "manual_roi_shared_left_right_subset",
        "train_video_ids": list(left_payload.get("splits", {}).get("train_video_ids") or []) + list(right_payload.get("splits", {}).get("train_video_ids") or []),
        "val_video_ids": list(left_payload.get("splits", {}).get("val_video_ids") or []) + list(right_payload.get("splits", {}).get("val_video_ids") or []),
        "test_video_ids": list(left_payload.get("splits", {}).get("test_video_ids") or []) + list(right_payload.get("splits", {}).get("test_video_ids") or []),
    }
    return {"windows": windows, "targets": targets, "video_ids": video_ids, "target_frame_indices": target_frame_indices, "event_indices": event_indices, "direction_ids": direction_ids, "splits": splits}


def _predict_latent_rnn_subset(*, run_dir: Path, windows: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    from neurobench.dynamics.models import GridAutoencoder, LatentGRUPredictor
    from neurobench.dynamics.train import _checkpoint_latent_stats, _torch

    torch = _torch()
    run = _load_json(run_dir / "latent_rnn_run.json")
    ckpt = torch.load(run_dir / "latent_rnn_checkpoint.pt", map_location=device)
    ae_ckpt = torch.load(str(run["source_autoencoder_run"]), map_location=device)
    latent_dim = int(ae_ckpt["latent_dim"])
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ae_ckpt, latent_dim)
    ae = GridAutoencoder(input_channels=int(windows.shape[2]), latent_dim=latent_dim, base_channels=int(ae_ckpt.get("base_channels", 16)), input_shape=tuple(ae_ckpt.get("input_shape") or windows.shape[2:])).to(device)
    ae.load_state_dict(ae_ckpt["model_state"])
    ae.eval()
    model = LatentGRUPredictor(latent_dim=latent_dim, hidden_dim=int(ckpt.get("hidden_dim") or 64)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    pred_chunks = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], max(1, int(batch_size))):
            batch = torch.as_tensor(windows[start:start+int(batch_size)], dtype=torch.float32, device=device)
            b, w, c, h, ww = batch.shape
            z_raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
            z = (z_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_step = model(z)
            if str(ckpt.get("prediction_target") or run.get("prediction_target")) == "delta":
                pred_z = z[:, -1, :] + pred_step
            else:
                pred_z = pred_step
            pred_raw = pred_z * latent_std + latent_mean
            pred_chunks.append(np.clip(np.nan_to_num(ae.decode(pred_raw).detach().cpu().numpy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
    if str(device).startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros((0, 1, 128, 128), dtype=np.float32)


def _predict_shared_hybrid_subset(*, run_dir: Path, windows: np.ndarray, direction_ids: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    from scripts.run_shared_directional_hybrid_rnn_sweep import DirectionalHybridGRU, decode_predictions, read_json
    from neurobench.dynamics.train import _checkpoint_latent_stats, _torch
    from neurobench.dynamics.models import GridAutoencoder

    torch = _torch()
    run = read_json(run_dir / "hybrid_rnn_run.json")
    ckpt = torch.load(run_dir / "hybrid_rnn_checkpoint.pt", map_location=device)
    config = ckpt["config"]
    ae_run = {"checkpoint_path": run["source_autoencoder_run"]}
    ae_ckpt = torch.load(str(ae_run["checkpoint_path"]), map_location=device)
    latent_dim = int(ae_ckpt["latent_dim"])
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ae_ckpt, latent_dim)
    ae = GridAutoencoder(input_channels=int(windows.shape[2]), latent_dim=latent_dim, base_channels=int(ae_ckpt.get("base_channels", 16)), input_shape=tuple(ae_ckpt.get("input_shape") or windows.shape[2:])).to(device)
    ae.load_state_dict(ae_ckpt["model_state"])
    ae.eval()
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    model = DirectionalHybridGRU(latent_dim=latent_dim, hidden_dim=int(config["hidden_dim"]), num_layers=int(config["num_layers"]), direction_emb_dim=int(config["direction_emb_dim"]), dropout=float(config["dropout"]), mode=str(config["mode"]), gate_kind=str(config["gate_kind"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    pred_z_chunks = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], max(1, int(batch_size))):
            batch = torch.as_tensor(windows[start:start+int(batch_size)], dtype=torch.float32, device=device)
            dirs = torch.as_tensor(direction_ids[start:start+int(batch_size)], dtype=torch.long, device=device)
            b, w, c, h, ww = batch.shape
            z_raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
            z = (z_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_z_chunks.append(model(z, dirs)["pred"].detach().cpu().numpy().astype(np.float32))
    pred_z = np.concatenate(pred_z_chunks, axis=0) if pred_z_chunks else np.zeros((0, latent_dim), dtype=np.float32)
    pred = decode_predictions(ae_run, pred_z, batch_size=max(1, int(batch_size)), device=device)
    if str(device).startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return pred


def _model_score_summary(*, run_dir: Path, architecture: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_values = [row["event_model_mse"] for row in rows if row.get("event_model_mse") is not None]
    persist_values = [row["event_persistence_mse"] for row in rows if row.get("event_persistence_mse") is not None]
    improvements = [row["event_improvement_over_persistence_mse"] for row in rows if row.get("event_improvement_over_persistence_mse") is not None]
    if run_dir.parent.name in {"left", "right"}:
        experiment_id = f"{run_dir.parent.name}/{run_dir.name}"
    else:
        experiment_id = run_dir.name
    return {
        "run_dir": str(run_dir),
        "experiment_id": experiment_id,
        "architecture": str(architecture),
        "roi_count": len(rows),
        "event_window_count": int(sum(row.get("matched_event_window_count") or 0 for row in rows)),
        "mean_event_model_mse": float(np.mean(model_values)) if model_values else None,
        "mean_event_persistence_mse": float(np.mean(persist_values)) if persist_values else None,
        "mean_event_improvement_over_persistence_mse": float(np.mean(improvements)) if improvements else None,
    }

def render_manual_roi_annotations_html(manifest: Mapping[str, Any]) -> str:
    rows = []
    for item in manifest.get("annotations", []) or []:
        rows.append(
            "<tr>"
            f"<td>{_e(item.get('video_id'))}</td><td>{_e(item.get('roi_id'))}</td>"
            f"<td>{_e(item.get('crop_x'))}, {_e(item.get('crop_y'))}</td>"
            f"<td>{_e(item.get('grid_row'))}, {_e(item.get('grid_col'))}</td>"
            f"<td>{_e(item.get('spike_interval_count'))}</td><td>{_e(item.get('spike_frame_count'))}</td>"
            "</tr>"
        )
    warnings = "".join(f"<li>{_e(w.get('message') if isinstance(w, Mapping) else w)}</li>" for w in manifest.get("warnings", []) or [])
    return _html_page(
        str(manifest.get("title") or "Manual ROI Spike Annotations"),
        f"""
<section class="summary">
  <div><b>{_e(manifest.get('annotation_count'))}</b><span>ROIs</span></div>
  <div><b>{_e(manifest.get('spike_interval_count'))}</b><span>Spike intervals</span></div>
  <div><b>{_e(manifest.get('grid_size'))}</b><span>Grid size</span></div>
  <div><b>{_e(manifest.get('frame_rate_hz'))}</b><span>Frame rate Hz</span></div>
</section>
<section><h2>ROIs</h2><table><thead><tr><th>Video</th><th>ROI</th><th>Crop x,y</th><th>Grid row,col</th><th>Intervals</th><th>Frames</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Warnings</h2><ul>{warnings or '<li>None.</li>'}</ul></section>
""",
    )


def render_manual_roi_dataset_evaluation_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload.get('title')}",
        "",
        f"Dataset: `{payload.get('dataset_id')}`",
        f"ROIs: `{payload.get('roi_count')}`",
        f"Event margin frames: `{payload.get('event_margin_frames')}`",
        "",
        "## Summary",
        "",
        "| Split | ROIs | Matched event windows | Mean event target | Mean event persistence MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, row in sorted((payload.get("summary", {}) or {}).get("by_split", {}).items()):
        lines.append(
            f"| {split} | {row.get('roi_count', 0)} | {row.get('matched_event_window_count', 0)} | {_fmt(row.get('mean_event_target'))} | {_fmt(row.get('mean_event_persistence_mse'))} |"
        )
    lines.extend(["", "## ROI Rows", "", "| Video | Split | ROI | Grid row,col | Event windows | Event target max | Persistence MSE |", "| --- | --- | ---: | --- | ---: | ---: | ---: |"])
    for row in payload.get("rows", []) or []:
        lines.append(
            f"| {row.get('video_id')} | {row.get('split')} | {row.get('roi_id')} | {row.get('grid_row')},{row.get('grid_col')} | {row.get('matched_event_window_count')} | {_fmt(row.get('event_target_max'))} | {_fmt(row.get('event_persistence_mse'))} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload.get("limitations", []) or [])
    return "\n".join(lines).rstrip() + "\n"


def render_manual_roi_dataset_evaluation_html(payload: Mapping[str, Any]) -> str:
    body = html.escape(render_manual_roi_dataset_evaluation_markdown(payload))
    return _html_page(str(payload.get("title") or "Manual ROI Spike Dataset Evaluation"), f"<pre>{body}</pre>")


def _read_first_sheet_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall("a:sheetData/a:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                values[_column_number(str(cell.get("r") or "A1"))] = _cell_value(cell, shared)
            if values:
                rows.append([values.get(idx, "") for idx in range(1, max(values) + 1)])
        return rows


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join((text.text or "") for text in item.findall(".//a:t", NS)) for item in root.findall("a:si", NS)]


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> str:
    if cell.get("t") == "inlineStr":
        return "".join((text.text or "") for text in cell.findall(".//a:t", NS))
    value = cell.find("a:v", NS)
    if value is None:
        return ""
    text = value.text or ""
    if cell.get("t") == "s":
        try:
            return str(shared[int(text)])
        except (IndexError, ValueError):
            return text
    return text


def _column_number(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 1
    out = 0
    for char in match.group(1):
        out = out * 26 + ord(char) - 64
    return out


def _video_id_from_path(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^ROIs[_ -]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_ -]*crop512x512$", "", name, flags=re.IGNORECASE)
    return name.replace("_", " ").strip()


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _coord_to_grid_index(value: float, *, crop_size: int, grid_size: int) -> int:
    return int(np.clip(np.floor(float(value) * float(grid_size) / float(crop_size)), 0, int(grid_size) - 1))


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _roi_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    header = ["annotation_id", "video_id", "roi_id", "crop_x", "crop_y", "grid_row", "grid_col", "spike_interval_count", "spike_frame_count", "mean_plus_sd"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(key) if row.get(key) is not None else "") for key in header))
    return "\n".join(lines) + "\n"


def _interval_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    header = ["annotation_id", "video_id", "roi_id", "grid_row", "grid_col", "start_frame", "end_frame", "center_frame", "duration_frames", "start_sec", "end_sec"]
    lines = ["\t".join(header)]
    for row in rows:
        for interval in row.get("spike_intervals", []) or []:
            merged = {**row, **interval}
            lines.append("\t".join(str(merged.get(key) if merged.get(key) is not None else "") for key in header))
    return "\n".join(lines) + "\n"


def _load_json(path_or_payload: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, Mapping):
        return dict(path_or_payload)
    return json.loads(Path(path_or_payload).read_text(encoding="utf-8"))


def _infer_target_indices(video_ids: np.ndarray) -> np.ndarray:
    counters: dict[str, int] = defaultdict(int)
    out = np.zeros(video_ids.shape[0], dtype=np.int64)
    for idx, video_id in enumerate(video_ids.astype(str)):
        out[idx] = counters[video_id]
        counters[video_id] += 1
    return out


def _split_for_video(video_id: str, splits: Mapping[str, Any]) -> str:
    for split in ("train", "val", "test"):
        if video_id in {str(item) for item in splits.get(f"{split}_video_ids", []) or []}:
            return split
    return "unknown"


def _mean_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _max_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(np.max(arr))


def _summarize_annotation_eval(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row.get("split") or "unknown") for row in rows}):
        selected = [row for row in rows if str(row.get("split") or "unknown") == split]
        event_targets = [row.get("event_target_mean") for row in selected if row.get("event_target_mean") is not None]
        event_persist = [row.get("event_persistence_mse") for row in selected if row.get("event_persistence_mse") is not None]
        by_split[split] = {
            "roi_count": len(selected),
            "matched_event_window_count": int(sum(int(row.get("matched_event_window_count") or 0) for row in selected)),
            "mean_event_target": float(np.mean(event_targets)) if event_targets else None,
            "mean_event_persistence_mse": float(np.mean(event_persist)) if event_persist else None,
        }
    return {"by_split": by_split}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_e(title)}</title>
<style>
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f7f8fa; }}
header {{ background: #17202a; color: #fff; padding: 24px 32px; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 24px 32px 40px; }}
h1 {{ margin: 0; font-size: 24px; }}
h2 {{ font-size: 17px; }}
section {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; background: transparent; border: 0; padding: 0; }}
.summary div {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; padding: 12px; }}
.summary b {{ display: block; font-size: 20px; }}
.summary span {{ color: #586579; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-top: 1px solid #e5eaf0; padding: 7px 8px; text-align: left; }}
pre {{ white-space: pre-wrap; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body><header><h1>{_e(title)}</h1></header><main>{body}</main></body>
</html>
"""
