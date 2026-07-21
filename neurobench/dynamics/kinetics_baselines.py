"""Kinetics-aware pixel baselines for grid dynamics sweeps."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.dynamics.baselines import KINETICS_BASELINES, _baseline_rate_hz, baseline_prediction
from neurobench.dynamics.error_analysis import promote_structured_error_metrics, structured_prediction_error_metrics
from neurobench.dynamics.train import _prediction_split_metrics, _prepare_model_array, _promote_split_metrics


def evaluate_kinetics_baselines(
    *,
    datasets: Mapping[str, Mapping[str, Any]],
    out_dir: str | Path,
    baseline_names: Sequence[str] = KINETICS_BASELINES,
    frame_rate_hz: float | None = None,
) -> dict[str, Any]:
    """Evaluate kinetics-aware baselines and write sweep-compatible metrics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    dataset_records = {str(key): dict(value) for key, value in datasets.items()}
    for dataset_key, dataset in dataset_records.items():
        timing = _dataset_timing(dataset, frame_rate_hz=frame_rate_hz)
        for baseline_name in baseline_names:
            exp_id = f"kinetics_{_slug(dataset_key)}_{_slug(baseline_name)}"
            exp_dir = out / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            params = _baseline_params(str(baseline_name), timing)
            config = {
                "experiment_id": exp_id,
                "kind": "array_baseline",
                "dataset_key": str(dataset_key),
                "seed": 0,
                "params": params,
            }
            (exp_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            metrics_path = _write_metrics(dataset=dataset, dataset_key=str(dataset_key), out_dir=exp_dir, baseline_name=str(baseline_name), timing=timing)
            experiments.append(config)
            rows.append({"experiment_id": exp_id, "dataset_key": str(dataset_key), "baseline_name": str(baseline_name), "metrics_path": str(metrics_path)})
    manifest = {
        "schema_version": 1,
        "profile": "kinetics_baselines",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(experiments),
        "datasets": dataset_records,
        "experiments": experiments,
    }
    (out / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out),
        "baseline_names": [str(name) for name in baseline_names],
        "experiment_count": len(experiments),
        "rows": rows,
        "manifest_path": str(out / "sweep_manifest.json"),
    }
    (out / "kinetics_baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "kinetics_baseline_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _write_metrics(*, dataset: Mapping[str, Any], dataset_key: str, out_dir: Path, baseline_name: str, timing: Mapping[str, Any]) -> Path:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        video_ids = arrays["window_video_ids"].astype(str)
        pred = baseline_prediction(
            windows,
            baseline_name,
            prediction_horizon_frames=int(timing["prediction_horizon_frames"]),
            frame_rate_hz=float(timing["frame_rate_hz"]),
        )
        persistence = baseline_prediction(windows, "persistence")
    diff = pred - targets
    persistence_diff = persistence - targets
    zero_latent = np.zeros((int(windows.shape[0]), 1), dtype=np.float32)
    split_metrics = _prediction_split_metrics(diff, zero_latent, zero_latent, persistence_diff, video_ids, dataset.get("splits"))
    structured_error_metrics = structured_prediction_error_metrics(
        pred_diff=diff,
        persistence_diff=persistence_diff,
        targets=targets,
        last_frames=windows[:, -1],
        video_ids=video_ids,
        splits=dataset.get("splits"),
    )
    metrics = {
        "schema_version": 1,
        "objective": f"kinetics_{baseline_name}_baseline",
        "model_kind": "array_baseline",
        "model_family": "kinetics_baseline",
        "baseline_family": "kinetics_aware",
        "baseline_name": str(baseline_name),
        "dataset_key": str(dataset_key),
        "frame_rate_hz": float(timing["frame_rate_hz"]),
        "prediction_horizon_frames": int(timing["prediction_horizon_frames"]),
        "prediction_horizon_sec": float(timing["prediction_horizon_sec"]),
        "reaction_rate_hz": _reaction_rate_for_name(baseline_name),
        "decoded_prediction_mse": float(np.mean(diff * diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(diff))),
        "persistence_mse": float(np.mean(persistence_diff * persistence_diff)),
        "persistence_mae": float(np.mean(np.abs(persistence_diff))),
        "persistence_baseline": {
            "mse": float(np.mean(persistence_diff * persistence_diff)),
            "mae": float(np.mean(np.abs(persistence_diff))),
            "count": int(windows.shape[0]),
        },
        "split_metrics": split_metrics,
        "structured_error_metrics": structured_error_metrics,
        "training_window_count": 0,
        "evaluation_window_count": int(windows.shape[0]),
        "input_normalization": "finite_clipped_unit_interval",
        "decoded_output_normalization": "clipped_unit_interval",
    }
    _promote_split_metrics(metrics, split_metrics, ["decoded_prediction_mse", "decoded_prediction_mae", "persistence_mse", "window_count"])
    promote_structured_error_metrics(metrics, structured_error_metrics)
    metrics["improvement_over_persistence_mse"] = float(metrics["persistence_mse"] - metrics["decoded_prediction_mse"])
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        if split.get("decoded_prediction_mse") is not None and split.get("persistence_mse") is not None:
            metrics[f"{split_name}_improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    metrics_path = out_dir / "array_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "model_kind": "array_baseline",
        "model_family": "kinetics_baseline",
        "baseline_name": str(baseline_name),
        "source_dataset": str(dataset.get("array_path")),
        "metrics_path": str(metrics_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "array_baseline_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_path


def _dataset_timing(dataset: Mapping[str, Any], *, frame_rate_hz: float | None) -> dict[str, Any]:
    windowing = dataset.get("windowing", {}) if isinstance(dataset.get("windowing"), Mapping) else {}
    rate = float(frame_rate_hz or windowing.get("effective_frame_rate_hz") or windowing.get("source_frame_rate_hz") or 50.0)
    horizon_frames = int(windowing.get("prediction_horizon_frames") or windowing.get("prediction_horizon_source_frames") or 1)
    horizon_sec = float(windowing.get("prediction_horizon_sec") or (horizon_frames / rate if rate else 0.0))
    return {"frame_rate_hz": rate, "prediction_horizon_frames": horizon_frames, "prediction_horizon_sec": horizon_sec}


def _baseline_params(baseline_name: str, timing: Mapping[str, Any]) -> dict[str, Any]:
    rate = _reaction_rate_for_name(baseline_name)
    params = {
        "baseline_name": str(baseline_name),
        "baseline_family": "kinetics_aware",
        "frame_rate_hz": float(timing["frame_rate_hz"]),
        "prediction_horizon_frames": int(timing["prediction_horizon_frames"]),
        "prediction_horizon_sec": float(timing["prediction_horizon_sec"]),
    }
    if rate is not None:
        params["reaction_rate_hz"] = float(rate)
    params["hyperparameter_summary"] = _hyperparameter_summary(params)
    return params


def _reaction_rate_for_name(baseline_name: str) -> float | None:
    name = str(baseline_name).lower()
    if name in {"ar1_per_cell", "per_cell_ar1", "ar1"}:
        return None
    return _baseline_rate_hz(name, reaction_rate_hz=None)


def _hyperparameter_summary(params: Mapping[str, Any]) -> str:
    parts = [f"baseline={params.get('baseline_name')}", f"h={params.get('prediction_horizon_frames')}", f"rate={params.get('frame_rate_hz')}hz"]
    if params.get("reaction_rate_hz") is not None:
        parts.append(f"reaction={params.get('reaction_rate_hz')}hz")
    return ", ".join(parts)


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Kinetics Baseline Summary",
        "",
        f"Output directory: `{summary.get('out_dir')}`",
        f"Experiments: `{summary.get('experiment_count')}`",
        "",
        "| Experiment | Dataset | Baseline | Metrics |",
        "|---|---|---|---|",
    ]
    for row in summary.get("rows", []):
        lines.append(f"| `{row.get('experiment_id')}` | {row.get('dataset_key')} | {row.get('baseline_name')} | `{row.get('metrics_path')}` |")
    return "\n".join(lines).rstrip() + "\n"


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "item"
