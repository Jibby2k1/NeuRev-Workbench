"""Training and evaluation utilities for grid latent dynamics smoke runs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench.dynamics.baselines import evaluate_baselines_from_arrays, write_baseline_metrics
from neurobench.dynamics.models import GridAutoencoder, LatentGRUPredictor
from neurobench.workbench.intermediates import normalize_array_frame, write_png_gray8


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for grid dynamics model training.") from exc
    return torch


def train_autoencoder(
    *,
    dataset: Mapping[str, Any],
    out_dir: str | Path,
    latent_dim: int = 32,
    base_channels: int = 16,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        frames = _prepare_model_array(arrays["frames"])
        frame_video_ids = arrays["frame_video_ids"].astype(str)
        frame_labels = arrays["frame_labels"].astype(str)
    train_mask = _split_mask(frame_video_ids, dataset.get("splits"), "train", default_all=True)
    if not np.any(train_mask):
        raise ValueError("Autoencoder training split is empty.")
    model = GridAutoencoder(input_channels=int(frames.shape[1]), latent_dim=int(latent_dim), base_channels=int(base_channels), input_shape=tuple(frames.shape[1:])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x_all = torch.from_numpy(frames).to(device)
    train_indices = torch.nonzero(torch.as_tensor(train_mask, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], int(batch_size)):
            batch = x_all[perm[start : start + int(batch_size)]]
            opt.zero_grad()
            recon, _z = model(batch)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    model.eval()
    recon_chunks = []
    latent_chunks = []
    with torch.no_grad():
        for start in range(0, x_all.shape[0], int(batch_size)):
            recon, z = model(x_all[start : start + int(batch_size)])
            recon_chunks.append(recon.detach().cpu().numpy().astype(np.float32))
            latent_chunks.append(z.detach().cpu().numpy().astype(np.float32))
    recon_np = _prepare_model_array(np.concatenate(recon_chunks, axis=0)) if recon_chunks else np.zeros(frames.shape, dtype=np.float32)
    latent_raw = np.concatenate(latent_chunks, axis=0).astype(np.float32) if latent_chunks else np.zeros((0, int(latent_dim)), dtype=np.float32)
    latent_mean, latent_std = _latent_standardization_stats(latent_raw[train_mask])
    latent = _standardize_latent(latent_raw, latent_mean, latent_std)
    metrics = _reconstruction_metrics(recon_np, frames, frame_video_ids, frame_labels)
    metrics["objective"] = "reconstruct_input"
    metrics["input_normalization"] = "finite_clipped_unit_interval"
    metrics["output_normalization"] = "sigmoid_unit_interval"
    metrics["latent_code_normalization"] = "standard_score_per_dimension"
    metrics["latent_stats_source"] = "train_split"
    metrics["training_loss"] = losses
    metrics["training_frame_count"] = int(train_mask.sum())
    metrics["evaluation_frame_count"] = int(frames.shape[0])
    metrics["split_metrics"] = _reconstruction_split_metrics(recon_np, frames, frame_video_ids, dataset.get("splits"))
    _promote_split_metrics(metrics, metrics["split_metrics"], ["mse", "mae", "frame_count"])
    metrics_path = out / "autoencoder_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = out / "autoencoder_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_dim": int(latent_dim),
            "base_channels": int(base_channels),
            "input_shape": list(frames.shape[1:]),
            "input_normalization": "finite_clipped_unit_interval",
            "output_normalization": "sigmoid_unit_interval",
            "latent_code_normalization": "standard_score_per_dimension",
            "latent_stats_source": "train_split",
            "latent_mean": torch.from_numpy(latent_mean.astype(np.float32)),
            "latent_std": torch.from_numpy(latent_std.astype(np.float32)),
        },
        checkpoint,
    )
    examples_path = out / "reconstruction_examples.json"
    examples = _grid_examples(frames, recon_np, max_examples=3)
    examples_path.write_text(json.dumps({"schema_version": 1, "examples": examples}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_grid_preview(out / "reconstruction_examples.png", frames[0, 0], recon_np[0, 0], np.abs(frames[0, 0] - recon_np[0, 0]))
    latent_path = out / "latent_codes.npz"
    np.savez(
        latent_path,
        latent_codes=latent,
        latent_codes_raw=latent_raw,
        latent_mean=latent_mean.astype(np.float32),
        latent_std=latent_std.astype(np.float32),
        frame_video_ids=frame_video_ids.astype("U64"),
        frame_labels=frame_labels.astype("U16"),
        latent_stats_source=np.asarray("train_split"),
    )
    split_method = str(dataset.get("splits", {}).get("split_method", ""))
    run = {
        "schema_version": 1,
        "run_id": out.name or "autoencoder_v1",
        "model_kind": "grid_autoencoder",
        "input_shape": [int(v) for v in frames.shape[1:]],
        "latent_dim": int(latent_dim),
        "base_channels": int(base_channels),
        "autoencoder_objective": "reconstruct_input",
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "base_channels": int(base_channels),
            "objective": "reconstruct_input",
            "input_normalization": "finite_clipped_unit_interval",
            "output_normalization": "sigmoid_unit_interval",
            "latent_code_normalization": "standard_score_per_dimension",
            "latent_stats_source": "train_split",
            "split_method": split_method,
        },
        "input_normalization": "finite_clipped_unit_interval",
        "output_normalization": "sigmoid_unit_interval",
        "latent_code_normalization": "standard_score_per_dimension",
        "latent_stats_source": "train_split",
        "source_dataset": str(dataset.get("array_path")),
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "reconstruction_examples_path": str(examples_path),
        "latent_codes_path": str(latent_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "device": str(device),
        "warnings": [],
        "extras": {
            "evaluation_mode": split_method,
            "train_frame_count": int(train_mask.sum()),
            "evaluation_frame_count": int(frames.shape[0]),
        },
    }
    (out / "autoencoder_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run


def train_latent_rnn(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    window_frames: int = 8,
    hidden_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    prediction_target: str = "absolute",
    lambda_latent: float = 0.1,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prediction_target = _normalize_prediction_target(prediction_target)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        window_video_ids = arrays["window_video_ids"].astype(str)
        baseline = evaluate_baselines_from_arrays(arrays)
    train_mask = _split_mask(window_video_ids, dataset.get("splits"), "train", default_all=True)
    if not np.any(train_mask):
        raise ValueError("Latent RNN training split is empty.")
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    base_channels = int(ckpt.get("base_channels", 16))
    ae = GridAutoencoder(input_channels=int(windows.shape[2]), latent_dim=latent_dim, base_channels=base_channels, input_shape=tuple(ckpt.get("input_shape") or windows.shape[2:])).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()
    model = LatentGRUPredictor(latent_dim=latent_dim, hidden_dim=int(hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    xw = torch.from_numpy(windows).to(device)
    yt = torch.from_numpy(targets).to(device)
    train_indices = torch.nonzero(torch.as_tensor(train_mask, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    encode_batch_frames = max(int(batch_size) * int(window_frames), int(batch_size))
    with torch.no_grad():
        target_z_raw = _encode_frames_batched(ae, yt, batch_size=encode_batch_frames)
        target_z = (target_z_raw - latent_mean) / latent_std
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            batch = xw[idx]
            b, w, c, h, ww = batch.shape
            with torch.no_grad():
                z_window_raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
                z_window = (z_window_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_step = model(z_window)
            if prediction_target == "delta":
                target_step = target_z[idx] - z_window[:, -1, :]
            else:
                target_step = target_z[idx]
            loss = torch.mean((pred_step - target_step) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    model.eval()
    pred_z_chunks = []
    pred_z_raw_chunks = []
    pred_x_chunks = []
    with torch.no_grad():
        b, w, c, h, ww = xw.shape
        for start in range(0, b, int(batch_size)):
            batch = xw[start : start + int(batch_size)]
            bb = batch.shape[0]
            z_window_raw = ae.encode(batch.reshape(bb * w, c, h, ww)).reshape(bb, w, latent_dim)
            z_window = (z_window_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_step_batch = model(z_window)
            if prediction_target == "delta":
                pred_z_batch = z_window[:, -1, :] + pred_step_batch
            else:
                pred_z_batch = pred_step_batch
            pred_z_raw_batch = pred_z_batch * latent_std + latent_mean
            pred_x_batch = _prepare_model_array(ae.decode(pred_z_raw_batch).detach().cpu().numpy())
            pred_z_chunks.append(pred_z_batch.detach().cpu().numpy().astype(np.float32))
            pred_z_raw_chunks.append(pred_z_raw_batch.detach().cpu().numpy().astype(np.float32))
            pred_x_chunks.append(pred_x_batch)
        pred_x = np.concatenate(pred_x_chunks, axis=0) if pred_x_chunks else np.zeros(targets.shape, dtype=np.float32)
        target_z_np = target_z.detach().cpu().numpy().astype(np.float32)
        pred_z_np = np.concatenate(pred_z_chunks, axis=0) if pred_z_chunks else np.zeros((0, latent_dim), dtype=np.float32)
        target_z_raw_np = target_z_raw.detach().cpu().numpy().astype(np.float32)
        pred_z_raw_np = np.concatenate(pred_z_raw_chunks, axis=0) if pred_z_raw_chunks else np.zeros((0, latent_dim), dtype=np.float32)
    diff = pred_x - targets
    latent_diff = pred_z_np - target_z_np
    latent_raw_diff = pred_z_raw_np - target_z_raw_np
    persistence_diff = windows[:, -1] - targets
    decoded_prediction_mse = float(np.mean(diff * diff))
    decoded_prediction_mae = float(np.mean(np.abs(diff)))
    latent_code_mse = float(np.mean(latent_diff * latent_diff))
    latent_code_mae = float(np.mean(np.abs(latent_diff)))
    latent_code_raw_mse = float(np.mean(latent_raw_diff * latent_raw_diff))
    latent_code_raw_mae = float(np.mean(np.abs(latent_raw_diff)))
    split_metrics = _prediction_split_metrics(diff, latent_diff, latent_raw_diff, persistence_diff, window_video_ids, dataset.get("splits"))
    metrics = {
        "objective": "next_delta_code_mse" if prediction_target == "delta" else "next_code_mse",
        "prediction_target": prediction_target,
        "training_loss": losses,
        "training_latent_mse": losses,
        "training_window_count": int(train_mask.sum()),
        "evaluation_window_count": int(windows.shape[0]),
        "latent_code_mse": latent_code_mse,
        "latent_code_mae": latent_code_mae,
        "latent_code_raw_mse": latent_code_raw_mse,
        "latent_code_raw_mae": latent_code_raw_mae,
        "input_normalization": "finite_clipped_unit_interval",
        "decoded_output_normalization": "sigmoid_unit_interval",
        "latent_code_normalization": "standard_score_per_dimension",
        "decoded_prediction_mse": decoded_prediction_mse,
        "decoded_prediction_mae": decoded_prediction_mae,
        "prediction_mse": decoded_prediction_mse,
        "prediction_mae": decoded_prediction_mae,
        "persistence_baseline": baseline["persistence"],
        "split_metrics": split_metrics,
    }
    _promote_split_metrics(
        metrics,
        split_metrics,
        [
            "latent_code_mse",
            "latent_code_mae",
            "latent_code_raw_mse",
            "latent_code_raw_mae",
            "decoded_prediction_mse",
            "decoded_prediction_mae",
            "persistence_mse",
            "window_count",
        ],
    )
    metrics["improvement_over_persistence_mse"] = float(metrics["persistence_baseline"]["mse"] - metrics["decoded_prediction_mse"])
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        if split.get("decoded_prediction_mse") is not None and split.get("persistence_mse") is not None:
            metrics[f"{split_name}_improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    baseline_path = out / "baseline_metrics.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path = out / "latent_rnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = out / "latent_rnn_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_dim": latent_dim,
            "hidden_dim": int(hidden_dim),
            "objective": "next_delta_code_mse" if prediction_target == "delta" else "next_code_mse",
            "prediction_target": prediction_target,
            "latent_code_normalization": "standard_score_per_dimension",
            "predicted_code_space": "standardized_latent_delta" if prediction_target == "delta" else "standardized_latent",
        },
        checkpoint,
    )
    examples_path = out / "prediction_examples.json"
    examples = _prediction_examples(windows, targets, pred_x, max_examples=3)
    examples_path.write_text(json.dumps({"schema_version": 1, "examples": examples}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if pred_x.shape[0]:
        _write_grid_preview(out / "prediction_examples.png", targets[0, 0], pred_x[0, 0], np.abs(targets[0, 0] - pred_x[0, 0]))
    split_method = str(dataset.get("splits", {}).get("split_method", ""))
    run = {
        "schema_version": 1,
        "run_id": out.name or "latent_rnn_v1",
        "model_kind": "latent_gru_predictor",
        "latent_dim": latent_dim,
        "window_frames": int(window_frames),
        "prediction_horizon_frames": int(dataset.get("windowing", {}).get("prediction_horizon_frames", 1)),
        "recurrent_unit": "gru",
        "hidden_dim": int(hidden_dim),
        "autoencoder_objective": str(autoencoder_run.get("autoencoder_objective", "reconstruct_input")),
        "rnn_objective": "predict_next_delta_latent_code" if prediction_target == "delta" else "predict_next_latent_code",
        "prediction_target": prediction_target,
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "objective": "next_delta_code_mse" if prediction_target == "delta" else "next_code_mse",
            "prediction_target": prediction_target,
            "latent_code_normalization": "standard_score_per_dimension",
            "split_method": split_method,
        },
        "input_normalization": "finite_clipped_unit_interval",
        "decoded_output_normalization": "sigmoid_unit_interval",
        "latent_code_normalization": "standard_score_per_dimension",
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "source_dataset": str(dataset.get("array_path")),
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "prediction_examples_path": str(examples_path),
        "rollout_examples_path": str(examples_path),
        "baseline_metrics_path": str(baseline_path),
        "seed": int(seed),
        "device": str(device),
        "warnings": [],
        "extras": {
            "evaluation_mode": split_method,
            "train_window_count": int(train_mask.sum()),
            "evaluation_window_count": int(windows.shape[0]),
            "decoded_prediction_used_for_training": False,
            "legacy_lambda_latent_ignored": float(lambda_latent),
            "prediction_target": prediction_target,
            "predicted_code_space": "standardized_latent_delta" if prediction_target == "delta" else "standardized_latent",
            "decoded_prediction_space": "unit_interval_grid",
        },
    }
    (out / "latent_rnn_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run


def _normalize_prediction_target(value: str) -> str:
    target = str(value or "absolute").strip().lower()
    aliases = {"next": "absolute", "code": "absolute", "absolute_code": "absolute", "delta_code": "delta", "residual": "delta"}
    target = aliases.get(target, target)
    if target not in {"absolute", "delta"}:
        raise ValueError("prediction_target must be 'absolute' or 'delta'.")
    return target


def _encode_frames_batched(model: GridAutoencoder, frames, *, batch_size: int):
    chunks = []
    total = int(frames.shape[0])
    step = max(1, int(batch_size))
    for start in range(0, total, step):
        chunks.append(model.encode(frames[start : start + step]))
    if not chunks:
        return frames.new_zeros((0, model.latent_dim))
    return chunks[0] if len(chunks) == 1 else _torch().cat(chunks, dim=0)


def _prepare_model_array(values: np.ndarray) -> np.ndarray:
    """Return finite float32 model data clipped to the normalized unit interval."""
    arr = np.asarray(values, dtype=np.float32)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32, copy=False)


def _latent_standardization_stats(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(latent, dtype=np.float32)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize_latent(latent: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(latent, dtype=np.float32) - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def _checkpoint_latent_stats(checkpoint: Mapping[str, Any], latent_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = _checkpoint_array(checkpoint.get("latent_mean", np.zeros((latent_dim,), dtype=np.float32))).reshape(-1)
    std = _checkpoint_array(checkpoint.get("latent_std", np.ones((latent_dim,), dtype=np.float32))).reshape(-1)
    if mean.shape[0] != int(latent_dim) or std.shape[0] != int(latent_dim):
        raise ValueError("Autoencoder checkpoint latent normalization shape does not match latent_dim.")
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std


def _checkpoint_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _split_mask(video_ids: np.ndarray, splits: Any, split_name: str, *, default_all: bool = False) -> np.ndarray:
    ids = np.asarray(video_ids).astype(str)
    if not isinstance(splits, Mapping):
        return np.ones(ids.shape[0], dtype=bool) if default_all else np.zeros(ids.shape[0], dtype=bool)
    values = splits.get(f"{split_name}_video_ids", [])
    if values is None:
        values = []
    selected = {str(value) for value in values}
    if not selected and default_all:
        return np.ones(ids.shape[0], dtype=bool)
    return np.isin(ids, list(selected))


def _split_masks(video_ids: np.ndarray, splits: Any) -> dict[str, np.ndarray]:
    ids = np.asarray(video_ids).astype(str)
    return {
        "all": np.ones(ids.shape[0], dtype=bool),
        "train": _split_mask(ids, splits, "train", default_all=True),
        "val": _split_mask(ids, splits, "val", default_all=False),
        "test": _split_mask(ids, splits, "test", default_all=False),
    }


def _metric_payload_from_diff(diff: np.ndarray, *, count_key: str) -> dict[str, Any]:
    if diff.size == 0 or diff.shape[0] == 0:
        return {count_key: 0, "mse": None, "mae": None}
    return {count_key: int(diff.shape[0]), "mse": float(np.mean(diff * diff)), "mae": float(np.mean(np.abs(diff)))}


def _reconstruction_split_metrics(recon: np.ndarray, target: np.ndarray, video_ids: np.ndarray, splits: Any) -> dict[str, dict[str, Any]]:
    diff = recon - target
    payload: dict[str, dict[str, Any]] = {}
    for split_name, mask in _split_masks(video_ids, splits).items():
        item = _metric_payload_from_diff(diff[mask], count_key="frame_count")
        payload[split_name] = item
    return payload


def _prediction_split_metrics(
    decoded_diff: np.ndarray,
    latent_diff: np.ndarray,
    latent_raw_diff: np.ndarray,
    persistence_diff: np.ndarray,
    video_ids: np.ndarray,
    splits: Any,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for split_name, mask in _split_masks(video_ids, splits).items():
        count = int(mask.sum())
        if count == 0:
            payload[split_name] = {
                "window_count": 0,
                "latent_code_mse": None,
                "latent_code_mae": None,
                "latent_code_raw_mse": None,
                "latent_code_raw_mae": None,
                "decoded_prediction_mse": None,
                "decoded_prediction_mae": None,
                "persistence_mse": None,
                "persistence_mae": None,
            }
            continue
        dd = decoded_diff[mask]
        ld = latent_diff[mask]
        lrd = latent_raw_diff[mask]
        pd = persistence_diff[mask]
        payload[split_name] = {
            "window_count": count,
            "latent_code_mse": float(np.mean(ld * ld)),
            "latent_code_mae": float(np.mean(np.abs(ld))),
            "latent_code_raw_mse": float(np.mean(lrd * lrd)),
            "latent_code_raw_mae": float(np.mean(np.abs(lrd))),
            "decoded_prediction_mse": float(np.mean(dd * dd)),
            "decoded_prediction_mae": float(np.mean(np.abs(dd))),
            "persistence_mse": float(np.mean(pd * pd)),
            "persistence_mae": float(np.mean(np.abs(pd))),
        }
    return payload


def _promote_split_metrics(metrics: dict[str, Any], split_metrics: Mapping[str, Mapping[str, Any]], fields: list[str]) -> None:
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        for field in fields:
            if field in split:
                metrics[f"{split_name}_{field}"] = split[field]


def _reconstruction_metrics(recon: np.ndarray, target: np.ndarray, video_ids: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    diff = recon - target
    payload = {"mse": float(np.mean(diff * diff)), "mae": float(np.mean(np.abs(diff))), "frame_count": int(target.shape[0])}
    payload["per_video"] = {}
    for vid in sorted(set(video_ids.tolist())):
        mask = video_ids == vid
        d = diff[mask]
        payload["per_video"][vid] = {"mse": float(np.mean(d * d)), "mae": float(np.mean(np.abs(d))), "frame_count": int(mask.sum())}
    payload["per_label"] = {}
    for lab in sorted(set(labels.tolist())):
        mask = labels == lab
        d = diff[mask]
        payload["per_label"][lab] = {"mse": float(np.mean(d * d)), "mae": float(np.mean(np.abs(d))), "frame_count": int(mask.sum())}
    return payload


def _grid_examples(x: np.ndarray, recon: np.ndarray, *, max_examples: int) -> list[dict[str, Any]]:
    items=[]
    for i in range(min(max_examples, x.shape[0])):
        items.append({"index": int(i), "input": x[i,0].round(5).tolist(), "reconstruction": recon[i,0].round(5).tolist(), "abs_error_mean": float(np.mean(np.abs(x[i]-recon[i])))})
    return items


def _prediction_examples(windows: np.ndarray, targets: np.ndarray, pred: np.ndarray, *, max_examples: int) -> list[dict[str, Any]]:
    items=[]
    for i in range(min(max_examples, targets.shape[0])):
        items.append({"index": int(i), "input_last": windows[i,-1,0].round(5).tolist(), "target_next": targets[i,0].round(5).tolist(), "predicted_next": pred[i,0].round(5).tolist(), "abs_error_mean": float(np.mean(np.abs(targets[i]-pred[i])))})
    return items


def _write_grid_preview(path: Path, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    canvas = np.concatenate([a, b, c], axis=1)
    write_png_gray8(path, int(canvas.shape[1]), int(canvas.shape[0]), normalize_array_frame(canvas))
