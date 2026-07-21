"""Shared multi-horizon linear latent baselines."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.dynamics.error_analysis import promote_structured_error_metrics, structured_prediction_error_metrics
from neurobench.dynamics.linear import _decode_latents, _encode_latent_windows, _fit_ridge
from neurobench.dynamics.models import GridAutoencoder
from neurobench.dynamics.train import (
    _checkpoint_latent_stats,
    _normalize_prediction_target,
    _prediction_examples,
    _prediction_split_metrics,
    _prepare_model_array,
    _promote_split_metrics,
    _split_mask,
    _torch,
    _write_grid_preview,
)


def evaluate_shared_multi_horizon_linear_latent(
    *,
    datasets: Mapping[str, Mapping[str, Any]],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    prediction_target: str = "delta",
    alphas: Sequence[float] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    batch_size: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit one horizon-conditioned linear latent model across multiple horizons."""
    if len(datasets) < 2:
        raise ValueError("At least two horizon datasets are required for shared multi-horizon evaluation.")
    prediction_target = _normalize_prediction_target(prediction_target)
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_items = sorted(datasets.items(), key=lambda item: _horizon_frames(item[1], item[0]) or 0)
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    base_channels = int(ckpt.get("base_channels", 16))
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    first_dataset = dataset_items[0][1]
    with np.load(first_dataset["array_path"], allow_pickle=False) as arrays:
        input_channels = int(arrays["windows"].shape[2])
        input_shape = tuple(ckpt.get("input_shape") or arrays["windows"].shape[2:])
    ae = GridAutoencoder(input_channels=input_channels, latent_dim=latent_dim, base_channels=base_channels, input_shape=input_shape).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()

    encoded = []
    train_x_parts = []
    train_y_parts = []
    val_x_parts = []
    val_y_parts = []
    horizon_values = [_horizon_frames(dataset, key) or (idx + 1) for idx, (key, dataset) in enumerate(dataset_items)]
    horizon_scale = float(max(horizon_values) or 1.0)
    for idx, (dataset_key, dataset) in enumerate(dataset_items):
        bundle = _encode_dataset(
            dataset_key=dataset_key,
            dataset=dataset,
            autoencoder=ae,
            latent_mean_np=latent_mean_np,
            latent_std_np=latent_std_np,
            horizon_value=float(horizon_values[idx]) / horizon_scale,
            prediction_target=prediction_target,
            batch_size=int(batch_size),
            device=device,
        )
        encoded.append(bundle)
        train_x_parts.append(bundle["features"][bundle["train_mask"]])
        train_y_parts.append(bundle["target_y"][bundle["train_mask"]])
        if np.any(bundle["val_mask"]):
            val_x_parts.append(bundle["features"][bundle["val_mask"]])
            val_y_parts.append(bundle["target_y"][bundle["val_mask"]])
    train_x = np.concatenate(train_x_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    if train_x.shape[0] == 0:
        raise ValueError("Shared multi-horizon training split is empty.")
    if val_x_parts:
        selection_x = np.concatenate(val_x_parts, axis=0)
        selection_y = np.concatenate(val_y_parts, axis=0)
        selection_metric = "val_latent_code_mse"
    else:
        selection_x = train_x
        selection_y = train_y
        selection_metric = "train_latent_code_mse"
    alpha_records = []
    best: dict[str, Any] | None = None
    for alpha in [float(v) for v in alphas]:
        weights = _fit_ridge(train_x, train_y, alpha=alpha)
        pred = (selection_x @ weights).astype(np.float32)
        diff = pred - selection_y
        selection_mse = float(np.mean(diff * diff))
        alpha_records.append({"alpha": alpha, "selection_latent_code_mse": selection_mse})
        if best is None or selection_mse < float(best["selection_latent_code_mse"]):
            best = {"alpha": alpha, "selection_latent_code_mse": selection_mse, "weights": weights}
    assert best is not None
    weights = np.asarray(best["weights"], dtype=np.float64)
    per_horizon: dict[str, Any] = {}
    weighted_mse_sum = 0.0
    weighted_persistence_sum = 0.0
    total_windows = 0
    for bundle in encoded:
        horizon_metrics = _evaluate_encoded_dataset(
            bundle=bundle,
            dataset=bundle["dataset"],
            autoencoder=ae,
            weights=weights,
            latent_mean_np=latent_mean_np,
            latent_std_np=latent_std_np,
            prediction_target=prediction_target,
            batch_size=int(batch_size),
            device=device,
            out_dir=out / str(bundle["dataset_key"]),
        )
        per_horizon[str(bundle["dataset_key"])] = horizon_metrics
        n = int(horizon_metrics.get("evaluation_window_count") or 0)
        total_windows += n
        weighted_mse_sum += float(horizon_metrics.get("decoded_prediction_mse") or 0.0) * n
        weighted_persistence_sum += float(horizon_metrics.get("persistence_mse") or 0.0) * n
    overall_mse = weighted_mse_sum / max(total_windows, 1)
    overall_persistence = weighted_persistence_sum / max(total_windows, 1)
    metrics = {
        "schema_version": 1,
        "objective": "shared_multi_horizon_linear_delta_latent" if prediction_target == "delta" else "shared_multi_horizon_linear_absolute_latent",
        "model_kind": "shared_multi_horizon_linear_latent",
        "model_family": "multi_horizon_linear_latent",
        "prediction_target": prediction_target,
        "horizon_conditioning": "normalized_horizon_scalar_with_feature_interactions",
        "dataset_keys": [key for key, _ in dataset_items],
        "shared_horizons_frames": [int(v) for v in horizon_values],
        "best_alpha": float(best["alpha"]),
        "alpha_records": alpha_records,
        "selection_metric": selection_metric,
        "selection_latent_code_mse": float(best["selection_latent_code_mse"]),
        "per_horizon_metrics": per_horizon,
        "decoded_prediction_mse": float(overall_mse),
        "persistence_mse": float(overall_persistence),
        "improvement_over_persistence_mse": float(overall_persistence - overall_mse),
        "evaluation_window_count": int(total_windows),
        "latent_dim": int(latent_dim),
        "latent_code_normalization": "standard_score_per_dimension",
        "decoded_output_normalization": "sigmoid_unit_interval",
    }
    weights_path = out / "multi_horizon_linear_weights.npz"
    metrics_path = out / "multi_horizon_linear_metrics.json"
    run_path = out / "multi_horizon_linear_run.json"
    np.savez(weights_path, weights=weights.astype(np.float32), best_alpha=np.asarray(float(best["alpha"]), dtype=np.float32), horizon_scale=np.asarray(horizon_scale, dtype=np.float32))
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": out.name or "multi_horizon_linear_latent_v1",
        "model_kind": "shared_multi_horizon_linear_latent",
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "dataset_keys": [key for key, _ in dataset_items],
        "shared_horizons_frames": [int(v) for v in horizon_values],
        "prediction_target": prediction_target,
        "weights_path": str(weights_path),
        "metrics_path": str(metrics_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "warnings": [],
        "extras": {
            "horizon_conditioning": "normalized_horizon_scalar_with_feature_interactions",
            "train_window_count": int(train_x.shape[0]),
            "selection_metric": selection_metric,
        },
    }
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run


def _encode_dataset(
    *,
    dataset_key: str,
    dataset: Mapping[str, Any],
    autoencoder: GridAutoencoder,
    latent_mean_np: np.ndarray,
    latent_std_np: np.ndarray,
    horizon_value: float,
    prediction_target: str,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        window_video_ids = arrays["window_video_ids"].astype(str)
    z_window, target_z = _encode_latent_windows(
        autoencoder,
        windows,
        targets,
        latent_mean_np=latent_mean_np,
        latent_std_np=latent_std_np,
        batch_size=batch_size,
        device=device,
    )
    n = int(z_window.shape[0])
    base_x = z_window.reshape(n, -1).astype(np.float64)
    features = _horizon_features(base_x, horizon_value)
    target_y = (target_z - z_window[:, -1, :]).astype(np.float64) if prediction_target == "delta" else target_z.astype(np.float64)
    train_mask = _split_mask(window_video_ids, dataset.get("splits"), "train", default_all=True)
    val_mask = _split_mask(window_video_ids, dataset.get("splits"), "val", default_all=False)
    return {
        "dataset_key": dataset_key,
        "dataset": dict(dataset),
        "horizon_value": float(horizon_value),
        "features": features,
        "target_y": target_y,
        "z_window": z_window,
        "target_z": target_z,
        "window_video_ids": window_video_ids,
        "train_mask": train_mask,
        "val_mask": val_mask,
    }


def _evaluate_encoded_dataset(
    *,
    bundle: Mapping[str, Any],
    dataset: Mapping[str, Any],
    autoencoder: GridAutoencoder,
    weights: np.ndarray,
    latent_mean_np: np.ndarray,
    latent_std_np: np.ndarray,
    prediction_target: str,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
    z_window = np.asarray(bundle["z_window"], dtype=np.float32)
    target_z = np.asarray(bundle["target_z"], dtype=np.float32)
    pred_step = (np.asarray(bundle["features"], dtype=np.float64) @ weights).astype(np.float32)
    pred_z = z_window[:, -1, :] + pred_step if prediction_target == "delta" else pred_step
    pred_z_raw = pred_z * latent_std_np.reshape(1, -1) + latent_mean_np.reshape(1, -1)
    pred_x = _decode_latents(autoencoder, pred_z_raw, batch_size=batch_size, device=device)
    decoded_diff = pred_x - targets
    latent_diff = pred_z - target_z
    latent_raw_diff = pred_z_raw.astype(np.float32) - (target_z * latent_std_np.reshape(1, -1) + latent_mean_np.reshape(1, -1)).astype(np.float32)
    persistence_diff = windows[:, -1] - targets
    video_ids = np.asarray(bundle["window_video_ids"]).astype(str)
    split_metrics = _prediction_split_metrics(decoded_diff, latent_diff, latent_raw_diff, persistence_diff, video_ids, dataset.get("splits"))
    structured_error_metrics = structured_prediction_error_metrics(
        pred_diff=decoded_diff,
        persistence_diff=persistence_diff,
        targets=targets,
        last_frames=windows[:, -1],
        video_ids=video_ids,
        splits=dataset.get("splits"),
    )
    metrics = {
        "dataset_key": str(bundle["dataset_key"]),
        "prediction_horizon_frames": _horizon_frames(dataset, str(bundle["dataset_key"])),
        "prediction_horizon_sec": _horizon_sec(dataset),
        "decoded_prediction_mse": float(np.mean(decoded_diff * decoded_diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(decoded_diff))),
        "persistence_mse": float(np.mean(persistence_diff * persistence_diff)),
        "persistence_mae": float(np.mean(np.abs(persistence_diff))),
        "latent_code_mse": float(np.mean(latent_diff * latent_diff)),
        "latent_code_mae": float(np.mean(np.abs(latent_diff))),
        "latent_code_raw_mse": float(np.mean(latent_raw_diff * latent_raw_diff)),
        "latent_code_raw_mae": float(np.mean(np.abs(latent_raw_diff))),
        "split_metrics": split_metrics,
        "structured_error_metrics": structured_error_metrics,
        "evaluation_window_count": int(decoded_diff.shape[0]),
    }
    _promote_split_metrics(
        metrics,
        split_metrics,
        ["latent_code_mse", "latent_code_mae", "latent_code_raw_mse", "latent_code_raw_mae", "decoded_prediction_mse", "decoded_prediction_mae", "persistence_mse", "window_count"],
    )
    promote_structured_error_metrics(metrics, structured_error_metrics)
    metrics["improvement_over_persistence_mse"] = float(metrics["persistence_mse"] - metrics["decoded_prediction_mse"])
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        if split.get("decoded_prediction_mse") is not None and split.get("persistence_mse") is not None:
            metrics[f"{split_name}_improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    examples = _prediction_examples(windows, targets, pred_x, max_examples=3, video_ids=video_ids, splits=dataset.get("splits"), windowing=dataset.get("windowing"))
    (out_dir / "prediction_examples.json").write_text(json.dumps({"schema_version": 1, "examples": examples}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if pred_x.shape[0]:
        _write_grid_preview(out_dir / "prediction_examples.png", targets[0, 0], pred_x[0, 0], np.abs(targets[0, 0] - pred_x[0, 0]))
    return metrics


def _horizon_features(base_x: np.ndarray, horizon_value: float) -> np.ndarray:
    h = np.full((base_x.shape[0], 1), float(horizon_value), dtype=np.float64)
    return np.concatenate([base_x, base_x * h, h, np.ones((base_x.shape[0], 1), dtype=np.float64)], axis=1)


def _horizon_frames(dataset: Mapping[str, Any], dataset_key: str) -> int | None:
    windowing = dataset.get("windowing", {}) if isinstance(dataset.get("windowing"), Mapping) else {}
    value = windowing.get("prediction_horizon_frames") or windowing.get("prediction_horizon_source_frames")
    if value is not None:
        return int(value)
    if "_h" in dataset_key:
        tail = dataset_key.rsplit("_h", 1)[-1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)
    return None


def _horizon_sec(dataset: Mapping[str, Any]) -> float | None:
    windowing = dataset.get("windowing", {}) if isinstance(dataset.get("windowing"), Mapping) else {}
    value = windowing.get("prediction_horizon_sec")
    return float(value) if value is not None else None
