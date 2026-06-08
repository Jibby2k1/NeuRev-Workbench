"""Linear latent baselines for grid dynamics."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.dynamics.baselines import evaluate_baselines_from_arrays
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


def evaluate_linear_latent_baseline(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    prediction_target: str = "absolute",
    alphas: Sequence[float] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    batch_size: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    prediction_target = _normalize_prediction_target(prediction_target)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        window_video_ids = arrays["window_video_ids"].astype(str)
        baseline = evaluate_baselines_from_arrays(arrays)
    train_mask = _split_mask(window_video_ids, dataset.get("splits"), "train", default_all=True)
    val_mask = _split_mask(window_video_ids, dataset.get("splits"), "val", default_all=False)
    if not np.any(train_mask):
        raise ValueError("Linear latent baseline training split is empty.")
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    base_channels = int(ckpt.get("base_channels", 16))
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    ae = GridAutoencoder(input_channels=int(windows.shape[2]), latent_dim=latent_dim, base_channels=base_channels, input_shape=tuple(ckpt.get("input_shape") or windows.shape[2:])).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()
    z_window, target_z = _encode_latent_windows(
        ae,
        windows,
        targets,
        latent_mean_np=latent_mean_np,
        latent_std_np=latent_std_np,
        batch_size=int(batch_size),
        device=device,
    )
    n = int(z_window.shape[0])
    x = z_window.reshape(n, -1).astype(np.float64)
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    if prediction_target == "delta":
        y = (target_z - z_window[:, -1, :]).astype(np.float64)
    else:
        y = target_z.astype(np.float64)
    train_x = x_aug[train_mask]
    train_y = y[train_mask]
    alpha_records = []
    best: dict[str, Any] | None = None
    for alpha in [float(v) for v in alphas]:
        weights = _fit_ridge(train_x, train_y, alpha=alpha)
        pred_step = (x_aug @ weights).astype(np.float32)
        pred_z = z_window[:, -1, :] + pred_step if prediction_target == "delta" else pred_step
        latent_diff = pred_z - target_z
        val_source = val_mask if np.any(val_mask) else np.ones(n, dtype=bool)
        selection_mse = float(np.mean(latent_diff[val_source] * latent_diff[val_source]))
        record = {"alpha": alpha, "selection_latent_code_mse": selection_mse}
        alpha_records.append(record)
        if best is None or selection_mse < float(best["selection_latent_code_mse"]):
            best = {"alpha": alpha, "selection_latent_code_mse": selection_mse, "weights": weights, "pred_z": pred_z}
    assert best is not None
    pred_z = np.asarray(best["pred_z"], dtype=np.float32)
    pred_z_raw = pred_z * latent_std_np.reshape(1, -1) + latent_mean_np.reshape(1, -1)
    pred_x = _decode_latents(ae, pred_z_raw, batch_size=int(batch_size), device=device)
    decoded_diff = pred_x - targets
    latent_diff = pred_z - target_z
    latent_raw_diff = pred_z_raw.astype(np.float32) - (target_z * latent_std_np.reshape(1, -1) + latent_mean_np.reshape(1, -1)).astype(np.float32)
    persistence_diff = windows[:, -1] - targets
    split_metrics = _prediction_split_metrics(decoded_diff, latent_diff, latent_raw_diff, persistence_diff, window_video_ids, dataset.get("splits"))
    metrics = {
        "objective": "linear_delta_latent_baseline" if prediction_target == "delta" else "linear_absolute_latent_baseline",
        "prediction_target": prediction_target,
        "best_alpha": float(best["alpha"]),
        "alpha_records": alpha_records,
        "selection_latent_code_mse": float(best["selection_latent_code_mse"]),
        "latent_code_mse": float(np.mean(latent_diff * latent_diff)),
        "latent_code_mae": float(np.mean(np.abs(latent_diff))),
        "latent_code_raw_mse": float(np.mean(latent_raw_diff * latent_raw_diff)),
        "latent_code_raw_mae": float(np.mean(np.abs(latent_raw_diff))),
        "decoded_prediction_mse": float(np.mean(decoded_diff * decoded_diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(decoded_diff))),
        "persistence_baseline": baseline["persistence"],
        "split_metrics": split_metrics,
        "training_window_count": int(train_mask.sum()),
        "evaluation_window_count": n,
        "latent_code_normalization": "standard_score_per_dimension",
        "decoded_output_normalization": "sigmoid_unit_interval",
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
    metrics_path = out / "linear_latent_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    weights_path = out / "linear_latent_weights.npz"
    np.savez(weights_path, weights=np.asarray(best["weights"], dtype=np.float32), best_alpha=np.asarray(float(best["alpha"]), dtype=np.float32))
    examples_path = out / "prediction_examples.json"
    examples = _prediction_examples(windows, targets, pred_x, max_examples=3)
    examples_path.write_text(json.dumps({"schema_version": 1, "examples": examples}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if pred_x.shape[0]:
        _write_grid_preview(out / "prediction_examples.png", targets[0, 0], pred_x[0, 0], np.abs(targets[0, 0] - pred_x[0, 0]))
    run = {
        "schema_version": 1,
        "run_id": out.name or "linear_latent_baseline_v1",
        "model_kind": "linear_latent_baseline",
        "latent_dim": latent_dim,
        "window_frames": int(dataset.get("windowing", {}).get("window_frames", 0) or 0),
        "prediction_horizon_frames": int(dataset.get("windowing", {}).get("prediction_horizon_frames", 1)),
        "prediction_target": prediction_target,
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "source_dataset": str(dataset.get("array_path")),
        "weights_path": str(weights_path),
        "metrics_path": str(metrics_path),
        "prediction_examples_path": str(examples_path),
        "baseline_metrics_path": "",
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
        "extras": {
            "train_window_count": int(train_mask.sum()),
            "evaluation_window_count": n,
            "selection_metric": "val_latent_code_mse" if np.any(val_mask) else "latent_code_mse",
        },
    }
    (out / "linear_latent_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    rhs = x.T @ y
    try:
        return np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ rhs


def _encode_latent_windows(
    ae: GridAutoencoder,
    windows: np.ndarray,
    targets: np.ndarray,
    *,
    latent_mean_np: np.ndarray,
    latent_std_np: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    z_windows = []
    z_targets = []
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, -1)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        for start in range(0, int(windows.shape[0]), max(1, int(batch_size))):
            wb = torch.from_numpy(windows[start : start + int(batch_size)]).to(device)
            tb = torch.from_numpy(targets[start : start + int(batch_size)]).to(device)
            b, w, c, h, ww = wb.shape
            zw_raw = ae.encode(wb.reshape(b * w, c, h, ww)).reshape(b, w, -1)
            zt_raw = ae.encode(tb)
            zw = (zw_raw - latent_mean.reshape(1, 1, -1)) / latent_std.reshape(1, 1, -1)
            zt = (zt_raw - latent_mean) / latent_std
            z_windows.append(zw.detach().cpu().numpy().astype(np.float32))
            z_targets.append(zt.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(z_windows, axis=0), np.concatenate(z_targets, axis=0)


def _decode_latents(ae: GridAutoencoder, latent_raw: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    chunks = []
    with torch.no_grad():
        for start in range(0, int(latent_raw.shape[0]), max(1, int(batch_size))):
            z = torch.from_numpy(latent_raw[start : start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            chunks.append(_prepare_model_array(ae.decode(z).detach().cpu().numpy()))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1, 32, 32), dtype=np.float32)
