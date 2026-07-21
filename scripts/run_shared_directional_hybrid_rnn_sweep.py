#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from neurobench.dynamics.error_analysis import promote_structured_error_metrics, structured_prediction_error_metrics
from neurobench.dynamics.models import GridAutoencoder
from neurobench.dynamics.train import (
    _checkpoint_latent_stats,
    _prepare_model_array,
    _prediction_examples,
    _prediction_split_metrics,
    _promote_split_metrics,
    _torch,
    _write_grid_preview,
)

ROOT = Path("Outputs/GridModel/060126_crop512_grid128_max_v1")
LEFT_DATASET = ROOT / "datasets/w8_s1_h2_left_train_test_rnn_v2/dynamics_dataset.json"
RIGHT_DATASET = ROOT / "datasets/w8_s1_h2_right_train_test_rnn_v2/dynamics_dataset.json"
DEFAULT_AE_RUNS = [
    ROOT / "models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json",
    ROOT / "models/autoencoder128_s1_ld128_bc16_e80_lr0p0010_v1/autoencoder_run.json",
]
DEFAULT_OUT = ROOT / "shared_directional_hybrid_rnn_sweep_v1"
DIRECTION_TO_ID = {"left": 0, "right": 1}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def ae_label(run: Mapping[str, Any]) -> str:
    return f"ld{int(run.get('latent_dim') or 0)}_" + Path(str(run.get("checkpoint_path", "ae"))).parent.name


class DirectionalHybridGRU(_torch().nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        direction_emb_dim: int,
        dropout: float,
        mode: str,
        gate_kind: str,
    ):
        torch = _torch()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.direction_emb_dim = int(direction_emb_dim)
        self.dropout = float(dropout)
        self.mode = str(mode)
        self.gate_kind = str(gate_kind)
        if self.mode not in {"absolute", "residual", "hybrid_taylor_gated"}:
            raise ValueError("mode must be absolute, residual, or hybrid_taylor_gated")
        if self.gate_kind not in {"scalar", "vector"}:
            raise ValueError("gate_kind must be scalar or vector")
        if self.direction_emb_dim > 0:
            self.direction_embedding = torch.nn.Embedding(2, self.direction_emb_dim)
        else:
            self.direction_embedding = None
        input_dim = self.latent_dim + max(0, self.direction_emb_dim)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(self.hidden_dim)
        self.abs_head = torch.nn.Linear(self.hidden_dim, self.latent_dim)
        self.delta_head = torch.nn.Linear(self.hidden_dim, self.latent_dim)
        self.accel_head = torch.nn.Linear(self.hidden_dim, self.latent_dim)
        gate_dim = 1 if self.gate_kind == "scalar" else self.latent_dim
        self.abs_gate = torch.nn.Linear(self.hidden_dim, gate_dim)
        self.accel_gate = torch.nn.Linear(self.hidden_dim, gate_dim)

    def forward(self, z_window, direction_ids):
        torch = _torch()
        if self.direction_embedding is not None:
            emb = self.direction_embedding(direction_ids.long()).unsqueeze(1).expand(-1, z_window.shape[1], -1)
            x = torch.cat([z_window, emb], dim=-1)
        else:
            x = z_window
        out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        z_last = z_window[:, -1, :]
        z_prev = z_window[:, -2, :] if z_window.shape[1] > 1 else z_last
        abs_z = self.abs_head(h)
        delta = self.delta_head(h)
        accel = self.accel_head(h)
        abs_gate = torch.sigmoid(self.abs_gate(h))
        accel_gate = torch.sigmoid(self.accel_gate(h))
        if self.mode == "absolute":
            pred = abs_z
        elif self.mode == "residual":
            pred = z_last + delta
        else:
            taylor = z_last + delta + accel_gate * accel
            pred = abs_gate * abs_z + (1.0 - abs_gate) * taylor
        target_delta_proxy = delta
        target_accel_proxy = accel
        return {
            "pred": pred,
            "abs": abs_z,
            "delta": target_delta_proxy,
            "accel": target_accel_proxy,
            "abs_gate_mean": abs_gate.mean(),
            "accel_gate_mean": accel_gate.mean(),
            "last_velocity": z_last - z_prev,
        }


def load_directional_arrays(dataset_path: Path, direction: str) -> dict[str, Any]:
    dataset = read_json(dataset_path)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        video_ids = arrays["window_video_ids"].astype(str)
    return {"dataset": dataset, "direction": direction, "windows": windows, "targets": targets, "video_ids": video_ids}


def combined_dataset_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_ds = left["dataset"]
    right_ds = right["dataset"]
    splits = {
        "split_unit": "video",
        "split_method": "shared_left_right_train_test_no_resting_direction_token_v1",
        "train_video_ids": list(left_ds["splits"].get("train_video_ids") or []) + list(right_ds["splits"].get("train_video_ids") or []),
        "val_video_ids": [],
        "test_video_ids": list(left_ds["splits"].get("test_video_ids") or []) + list(right_ds["splits"].get("test_video_ids") or []),
    }
    return {
        "schema_version": 1,
        "dataset_id": "shared_left_right_train_test_rnn_v1",
        "array_path": "in_memory_combined_left_right",
        "input_shape": list(left_ds.get("input_shape") or [1, 128, 128]),
        "grid_id": left_ds.get("grid_id"),
        "normalization": left_ds.get("normalization"),
        "windowing": left_ds.get("windowing"),
        "splits": splits,
        "source_datasets": {"left": str(LEFT_DATASET), "right": str(RIGHT_DATASET)},
        "warnings": ["Resting videos excluded from RNN training; left and right are combined with a direction token."],
    }


def load_combined_arrays() -> dict[str, Any]:
    left = load_directional_arrays(LEFT_DATASET, "left")
    right = load_directional_arrays(RIGHT_DATASET, "right")
    windows = np.concatenate([left["windows"], right["windows"]], axis=0).astype(np.float32)
    targets = np.concatenate([left["targets"], right["targets"]], axis=0).astype(np.float32)
    video_ids = np.concatenate([left["video_ids"], right["video_ids"]]).astype("U64")
    direction_labels = np.concatenate([
        np.full(left["windows"].shape[0], "left", dtype="U8"),
        np.full(right["windows"].shape[0], "right", dtype="U8"),
    ])
    direction_ids = np.asarray([DIRECTION_TO_ID[str(v)] for v in direction_labels], dtype=np.int64)
    dataset = combined_dataset_payload(left, right)
    return {"dataset": dataset, "windows": windows, "targets": targets, "video_ids": video_ids, "direction_labels": direction_labels, "direction_ids": direction_ids}


def split_mask(video_ids: np.ndarray, splits: Mapping[str, Any], split_name: str, *, default_all: bool = False) -> np.ndarray:
    values = splits.get(f"{split_name}_video_ids") or []
    selected = {str(v) for v in values}
    if not selected and default_all:
        return np.ones(video_ids.shape[0], dtype=bool)
    return np.isin(video_ids.astype(str), list(selected))


def load_autoencoder(autoencoder_run: Mapping[str, Any], device: str):
    torch = _torch()
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    ae = GridAutoencoder(
        input_channels=1,
        latent_dim=latent_dim,
        base_channels=int(ckpt.get("base_channels", 16)),
        input_shape=tuple(ckpt.get("input_shape") or (1, 128, 128)),
    ).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()
    return ae, latent_dim, latent_mean_np, latent_std_np


def encode_latent_cache(*, combined: Mapping[str, Any], autoencoder_run: Mapping[str, Any], out_dir: Path, batch_size: int, device: str) -> Path:
    cache_dir = out_dir / "latent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    label = ae_label(autoencoder_run).replace(os.sep, "_")
    cache_path = cache_dir / f"{label}_latent_only.npz"
    meta_path = cache_dir / f"{label}.json"
    if cache_path.exists() and meta_path.exists():
        return cache_path
    torch = _torch()
    ae, latent_dim, latent_mean_np, latent_std_np = load_autoencoder(autoencoder_run, device)
    windows = combined["windows"]
    targets = combined["targets"]
    frame_batch = max(1, int(batch_size))
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    z_windows = []
    z_targets = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], max(1, int(batch_size))):
            batch = torch.as_tensor(windows[start:start+int(batch_size)], dtype=torch.float32, device=device)
            b, w, c, h, ww = batch.shape
            raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
            z_windows.append(((raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)).detach().cpu().numpy().astype(np.float32))
        for start in range(0, targets.shape[0], frame_batch):
            batch = torch.as_tensor(targets[start:start+frame_batch], dtype=torch.float32, device=device)
            raw = ae.encode(batch)
            z_targets.append(((raw - latent_mean) / latent_std).detach().cpu().numpy().astype(np.float32))
    np.savez_compressed(
        cache_path,
        z_windows=np.concatenate(z_windows, axis=0).astype(np.float32),
        z_targets=np.concatenate(z_targets, axis=0).astype(np.float32),
        video_ids=combined["video_ids"].astype("U64"),
        direction_ids=combined["direction_ids"].astype(np.int64),
        direction_labels=combined["direction_labels"].astype("U8"),
        latent_mean=latent_mean_np.astype(np.float32),
        latent_std=latent_std_np.astype(np.float32),
    )
    write_json(meta_path, {"created_at": now(), "autoencoder_run": autoencoder_run, "latent_dim": latent_dim, "cache_path": cache_path.as_posix()})
    return cache_path


def decode_predictions(autoencoder_run: Mapping[str, Any], pred_z: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    ae, latent_dim, latent_mean_np, latent_std_np = load_autoencoder(autoencoder_run, device)
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    chunks = []
    with torch.no_grad():
        for start in range(0, pred_z.shape[0], max(1, int(batch_size))):
            z = torch.as_tensor(pred_z[start:start+int(batch_size)], dtype=torch.float32, device=device)
            raw = z * latent_std + latent_mean
            chunks.append(_prepare_model_array(ae.decode(raw).detach().cpu().numpy()))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1, 128, 128), dtype=np.float32)


def train_one(config: Mapping[str, Any], *, combined: Mapping[str, Any], cache_path: Path, autoencoder_run: Mapping[str, Any], run_dir: Path, device: str) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache_path, allow_pickle=False) as arrays:
        z_windows_np = arrays["z_windows"].astype(np.float32)
        z_targets_np = arrays["z_targets"].astype(np.float32)
        video_ids = arrays["video_ids"].astype(str)
        direction_ids_np = arrays["direction_ids"].astype(np.int64)
        direction_labels = arrays["direction_labels"].astype(str)
    windows = np.asarray(combined["windows"], dtype=np.float32)
    targets = np.asarray(combined["targets"], dtype=np.float32)
    dataset = combined["dataset"]
    train_mask_np = split_mask(video_ids, dataset["splits"], "train", default_all=True)
    if not np.any(train_mask_np):
        raise ValueError("Shared hybrid train split is empty.")
    latent_dim = int(z_windows_np.shape[-1])
    z_windows = torch.as_tensor(z_windows_np, dtype=torch.float32, device=device)
    z_targets = torch.as_tensor(z_targets_np, dtype=torch.float32, device=device)
    direction_ids = torch.as_tensor(direction_ids_np, dtype=torch.long, device=device)
    train_indices = torch.nonzero(torch.as_tensor(train_mask_np, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    model = DirectionalHybridGRU(
        latent_dim=latent_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        direction_emb_dim=int(config["direction_emb_dim"]),
        dropout=float(config["dropout"]),
        mode=str(config["mode"]),
        gate_kind=str(config["gate_kind"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    losses: list[float] = []
    gate_abs: list[float] = []
    gate_accel: list[float] = []
    batch_size = int(config["batch_size"])
    aux_abs_weight = float(config.get("aux_abs_weight", 0.0))
    aux_delta_weight = float(config.get("aux_delta_weight", 0.0))
    aux_accel_weight = float(config.get("aux_accel_weight", 0.0))
    for _epoch in range(int(config["epochs"])):
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        epoch_abs_gate = []
        epoch_accel_gate = []
        for start in range(0, perm.shape[0], batch_size):
            idx = perm[start:start+batch_size]
            z_window = z_windows[idx]
            z_target = z_targets[idx]
            out = model(z_window, direction_ids[idx])
            target_delta = z_target - z_window[:, -1, :]
            target_accel = target_delta - out["last_velocity"]
            loss = torch.mean((out["pred"] - z_target) ** 2)
            if aux_abs_weight:
                loss = loss + aux_abs_weight * torch.mean((out["abs"] - z_target) ** 2)
            if aux_delta_weight:
                loss = loss + aux_delta_weight * torch.mean((out["delta"] - target_delta) ** 2)
            if aux_accel_weight:
                loss = loss + aux_accel_weight * torch.mean((out["accel"] - target_accel) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_abs_gate.append(float(out["abs_gate_mean"].detach().cpu()))
            epoch_accel_gate.append(float(out["accel_gate_mean"].detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        gate_abs.append(float(np.mean(epoch_abs_gate)) if epoch_abs_gate else 0.0)
        gate_accel.append(float(np.mean(epoch_accel_gate)) if epoch_accel_gate else 0.0)
    model.eval()
    pred_chunks = []
    abs_gate_eval = []
    accel_gate_eval = []
    with torch.no_grad():
        for start in range(0, z_windows.shape[0], batch_size):
            out = model(z_windows[start:start+batch_size], direction_ids[start:start+batch_size])
            pred_chunks.append(out["pred"].detach().cpu().numpy().astype(np.float32))
            abs_gate_eval.append(float(out["abs_gate_mean"].detach().cpu()))
            accel_gate_eval.append(float(out["accel_gate_mean"].detach().cpu()))
    pred_z = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros(z_targets_np.shape, dtype=np.float32)
    pred_x = decode_predictions(autoencoder_run, pred_z, batch_size=max(16, batch_size), device=device)
    decoded_diff = pred_x - targets
    latent_diff = pred_z - z_targets_np
    latent_raw_diff = latent_diff.copy()
    persistence_diff = windows[:, -1] - targets
    split_metrics = _prediction_split_metrics(decoded_diff, latent_diff, latent_raw_diff, persistence_diff, video_ids, dataset["splits"])
    structured = structured_prediction_error_metrics(
        pred_diff=decoded_diff,
        persistence_diff=persistence_diff,
        targets=targets,
        last_frames=windows[:, -1],
        video_ids=video_ids,
        splits=dataset["splits"],
    )
    metrics: dict[str, Any] = {
        "objective": "shared_directional_hybrid_latent_mse",
        "model_mode": str(config["mode"]),
        "training_loss": losses,
        "training_window_count": int(train_mask_np.sum()),
        "evaluation_window_count": int(z_windows_np.shape[0]),
        "latent_code_mse": float(np.mean(latent_diff * latent_diff)),
        "decoded_prediction_mse": float(np.mean(decoded_diff * decoded_diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(decoded_diff))),
        "persistence_mse": float(np.mean(persistence_diff * persistence_diff)),
        "improvement_over_persistence_mse": float(np.mean(persistence_diff * persistence_diff) - np.mean(decoded_diff * decoded_diff)),
        "abs_gate_mean_training": gate_abs,
        "accel_gate_mean_training": gate_accel,
        "abs_gate_mean_eval": float(np.mean(abs_gate_eval)) if abs_gate_eval else None,
        "accel_gate_mean_eval": float(np.mean(accel_gate_eval)) if accel_gate_eval else None,
        "split_metrics": split_metrics,
        "structured_error_metrics": structured,
        "per_direction": {},
    }
    _promote_split_metrics(metrics, split_metrics, ["latent_code_mse", "decoded_prediction_mse", "decoded_prediction_mae", "persistence_mse", "window_count"])
    promote_structured_error_metrics(metrics, structured)
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        if split.get("decoded_prediction_mse") is not None and split.get("persistence_mse") is not None:
            metrics[f"{split_name}_improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    for direction in ("left", "right"):
        mask = direction_labels == direction
        if not np.any(mask):
            continue
        d_pred = decoded_diff[mask]
        d_persist = persistence_diff[mask]
        metrics["per_direction"][direction] = {
            "window_count": int(mask.sum()),
            "decoded_prediction_mse": float(np.mean(d_pred * d_pred)),
            "persistence_mse": float(np.mean(d_persist * d_persist)),
            "improvement_over_persistence_mse": float(np.mean(d_persist * d_persist) - np.mean(d_pred * d_pred)),
        }
        test_mask = mask & split_mask(video_ids, dataset["splits"], "test", default_all=False)
        if np.any(test_mask):
            t_pred = decoded_diff[test_mask]
            t_persist = persistence_diff[test_mask]
            metrics["per_direction"][direction]["test_decoded_prediction_mse"] = float(np.mean(t_pred * t_pred))
            metrics["per_direction"][direction]["test_persistence_mse"] = float(np.mean(t_persist * t_persist))
            metrics["per_direction"][direction]["test_improvement_over_persistence_mse"] = float(np.mean(t_persist * t_persist) - np.mean(t_pred * t_pred))
    metrics_path = run_dir / "hybrid_rnn_metrics.json"
    write_json(metrics_path, metrics)
    checkpoint_path = run_dir / "hybrid_rnn_checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_kind": "shared_directional_hybrid_gru",
        "latent_dim": latent_dim,
        "config": dict(config),
        "direction_to_id": DIRECTION_TO_ID,
    }, checkpoint_path)
    examples = _prediction_examples(windows, targets, pred_x, max_examples=3, video_ids=video_ids, splits=dataset["splits"], windowing=dataset.get("windowing"))
    examples_path = run_dir / "prediction_examples.json"
    write_json(examples_path, {"schema_version": 1, "examples": examples})
    if pred_x.shape[0]:
        _write_grid_preview(run_dir / "prediction_examples.png", targets[0, 0], pred_x[0, 0], np.abs(targets[0, 0] - pred_x[0, 0]))
    run = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "model_kind": "shared_directional_hybrid_gru",
        "model_family": "shared_directional_hybrid_rnn",
        "latent_dim": latent_dim,
        "recurrent_unit": "gru",
        "direction_token": int(config["direction_emb_dim"]) > 0,
        "hybrid_mode": str(config["mode"]),
        "training_config": dict(config),
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "source_dataset": "combined left/right train-test no-resting datasets",
        "checkpoint_path": checkpoint_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "prediction_examples_path": examples_path.as_posix(),
        "created_at": now(),
        "seed": int(config["seed"]),
        "device": str(device),
    }
    write_json(run_dir / "hybrid_rnn_run.json", run)
    return run


def default_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    # Sparse but covers the architectural questions: absolute vs residual vs gated Taylor, token/no-token, and low/high capacity.
    for mode, direction_emb_dim, hidden_dim, lr, seed in product(
        ["residual", "hybrid_taylor_gated", "absolute"],
        [4, 8, 0],
        [64, 128, 256],
        [3e-4, 1e-4],
        [7, 13],
    ):
        if mode == "absolute" and direction_emb_dim == 0:
            continue
        configs.append({
            "mode": mode,
            "direction_emb_dim": int(direction_emb_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": 1 if hidden_dim <= 128 else 2,
            "dropout": 0.0 if hidden_dim <= 128 else 0.1,
            "gate_kind": "vector" if mode == "hybrid_taylor_gated" else "scalar",
            "learning_rate": float(lr),
            "weight_decay": 1e-5,
            "epochs": 120,
            "batch_size": 64,
            "seed": int(seed),
            "grad_clip": 1.0,
            "aux_abs_weight": 0.1 if mode == "hybrid_taylor_gated" else 0.0,
            "aux_delta_weight": 0.1 if mode in {"residual", "hybrid_taylor_gated"} else 0.0,
            "aux_accel_weight": 0.05 if mode == "hybrid_taylor_gated" else 0.0,
        })
    return configs


def config_id(config: Mapping[str, Any], ae_run: Mapping[str, Any]) -> str:
    return "__".join([
        ae_label(ae_run),
        str(config["mode"]),
        f"tok{config['direction_emb_dim']}",
        f"hd{config['hidden_dim']}",
        f"ly{config['num_layers']}",
        f"lr{slug_float(config['learning_rate'])}",
        f"s{config['seed']}",
    ]).replace("/", "_")


def random_configs(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    for _ in range(n):
        mode = rng.choice(["residual", "hybrid_taylor_gated", "hybrid_taylor_gated", "absolute"])
        hidden_dim = rng.choice([64, 96, 128, 192, 256])
        configs.append({
            "mode": mode,
            "direction_emb_dim": rng.choice([4, 8, 16, 0]),
            "hidden_dim": hidden_dim,
            "num_layers": rng.choice([1, 2]),
            "dropout": rng.choice([0.0, 0.05, 0.1]),
            "gate_kind": rng.choice(["scalar", "vector"]),
            "learning_rate": 10 ** rng.uniform(math.log10(5e-5), math.log10(1e-3)),
            "weight_decay": 10 ** rng.uniform(math.log10(1e-7), math.log10(1e-4)),
            "epochs": rng.choice([75, 120, 180]),
            "batch_size": rng.choice([32, 64]),
            "seed": rng.choice([7, 13, 29]),
            "grad_clip": rng.choice([0.5, 1.0, 2.0]),
            "aux_abs_weight": rng.choice([0.0, 0.05, 0.1, 0.2]) if mode == "hybrid_taylor_gated" else 0.0,
            "aux_delta_weight": rng.choice([0.0, 0.05, 0.1, 0.2]) if mode in {"residual", "hybrid_taylor_gated"} else 0.0,
            "aux_accel_weight": rng.choice([0.0, 0.02, 0.05, 0.1]) if mode == "hybrid_taylor_gated" else 0.0,
        })
    return configs


def optuna_configs(n: int, seed: int, storage: str | None, study_name: str | None) -> list[dict[str, Any]] | None:
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError:
        return None
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage, study_name=study_name, load_if_exists=True)
    configs = []
    for _ in range(n):
        trial = study.ask()
        mode = trial.suggest_categorical("mode", ["residual", "hybrid_taylor_gated", "absolute"])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 192, 256])
        config = {
            "mode": mode,
            "direction_emb_dim": trial.suggest_categorical("direction_emb_dim", [4, 8, 16, 0]),
            "hidden_dim": hidden_dim,
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.15),
            "gate_kind": trial.suggest_categorical("gate_kind", ["scalar", "vector"]),
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "epochs": trial.suggest_categorical("epochs", [75, 120, 180]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "seed": trial.suggest_categorical("seed", [7, 13, 29]),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0]),
            "aux_abs_weight": trial.suggest_categorical("aux_abs_weight", [0.0, 0.05, 0.1, 0.2]) if mode == "hybrid_taylor_gated" else 0.0,
            "aux_delta_weight": trial.suggest_categorical("aux_delta_weight", [0.0, 0.05, 0.1, 0.2]) if mode in {"residual", "hybrid_taylor_gated"} else 0.0,
            "aux_accel_weight": trial.suggest_categorical("aux_accel_weight", [0.0, 0.02, 0.05, 0.1]) if mode == "hybrid_taylor_gated" else 0.0,
            "optuna_trial_number": trial.number,
        }
        configs.append(config)
    return configs


def metric_value(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("test_improvement_over_persistence_mse")
    if value is None:
        value = metrics.get("improvement_over_persistence_mse")
    return float(value) if value is not None else float("-inf")


def write_summary(out_dir: Path, records: list[dict[str, Any]], *, created_at: str, search_config: Mapping[str, Any]) -> None:
    completed = [r for r in records if r.get("status") == "completed"]
    best = sorted(completed, key=lambda r: metric_value(r.get("metrics", {})), reverse=True)[:20]
    payload = {
        "schema_version": 1,
        "created_at": created_at,
        "updated_at": now(),
        "state": "running",
        "search_config": dict(search_config),
        "counts": dict(Counter(str(r.get("status")) for r in records)),
        "records": records,
        "best_by_test_improvement": best,
    }
    write_json(out_dir / "shared_directional_hybrid_summary.json", payload)
    fields = ["index", "status", "config_id", "latent_dim", "mode", "direction_emb_dim", "hidden_dim", "num_layers", "learning_rate", "epochs", "seed", "test_improvement_over_persistence_mse", "test_decoded_prediction_mse", "test_persistence_mse", "test_high_change_improvement_over_persistence_mse", "run_path", "error"]
    lines = ["\t".join(fields)]
    for r in records:
        c = r.get("config", {})
        m = r.get("metrics", {})
        row = {
            "index": r.get("index", ""),
            "status": r.get("status", ""),
            "config_id": r.get("config_id", ""),
            "latent_dim": r.get("latent_dim", ""),
            "mode": c.get("mode", ""),
            "direction_emb_dim": c.get("direction_emb_dim", ""),
            "hidden_dim": c.get("hidden_dim", ""),
            "num_layers": c.get("num_layers", ""),
            "learning_rate": c.get("learning_rate", ""),
            "epochs": c.get("epochs", ""),
            "seed": c.get("seed", ""),
            "test_improvement_over_persistence_mse": m.get("test_improvement_over_persistence_mse", ""),
            "test_decoded_prediction_mse": m.get("test_decoded_prediction_mse", ""),
            "test_persistence_mse": m.get("test_persistence_mse", ""),
            "test_high_change_improvement_over_persistence_mse": m.get("test_high_change_improvement_over_persistence_mse", ""),
            "run_path": r.get("run_path", ""),
            "error": r.get("error", ""),
        }
        lines.append("\t".join(str(row.get(f, "")).replace("\t", " ").replace("\n", " ") for f in fields))
    (out_dir / "shared_directional_hybrid_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT.as_posix())
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["grid", "random", "optuna"], default="random")
    ap.add_argument("--n-trials", type=int, default=48)
    ap.add_argument("--max-runs", type=int, default=0, help="0 means run all selected trials")
    ap.add_argument("--time-limit-hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--autoencoder-run", action="append", default=[])
    ap.add_argument("--optuna-storage", default=None)
    ap.add_argument("--optuna-study-name", default="shared_directional_hybrid_rnn_v1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = load_combined_arrays()
    autoencoder_paths = [Path(p) for p in args.autoencoder_run] if args.autoencoder_run else [p for p in DEFAULT_AE_RUNS if p.exists()]
    if not autoencoder_paths:
        raise ValueError("No autoencoder_run.json files found. Pass --autoencoder-run.")
    autoencoder_runs = [read_json(p) for p in autoencoder_paths]
    if args.mode == "grid":
        configs = default_grid()
        search_backend = "deterministic_grid"
    elif args.mode == "optuna":
        configs = optuna_configs(int(args.n_trials), int(args.seed), args.optuna_storage, args.optuna_study_name)
        if configs is None:
            configs = random_configs(int(args.n_trials), int(args.seed))
            search_backend = "random_fallback_optuna_not_installed"
        else:
            search_backend = "optuna"
    else:
        configs = random_configs(int(args.n_trials), int(args.seed))
        search_backend = "random"
    created_at = now()
    search_config = {
        "backend": search_backend,
        "mode_requested": args.mode,
        "n_trial_configs": len(configs),
        "autoencoder_runs": [str(p) for p in autoencoder_paths],
        "total_candidate_runs": len(configs) * len(autoencoder_runs),
        "ranking_metric": "test_improvement_over_persistence_mse",
        "architectures": ["absolute", "residual", "hybrid_taylor_gated"],
        "uses_direction_token": True,
        "resting_policy": "excluded from RNN train/test; allowed only in upstream autoencoder training",
    }
    summary_path = out_dir / "shared_directional_hybrid_summary.json"
    records: list[dict[str, Any]] = []
    if summary_path.exists():
        prior = read_json(summary_path)
        records = list(prior.get("records") or [])
        created_at = str(prior.get("created_at") or created_at)
    completed = {str(r.get("config_id")) for r in records if r.get("status") == "completed"}
    write_json(out_dir / "combined_dataset_manifest.json", combined["dataset"])
    write_summary(out_dir, records, created_at=created_at, search_config=search_config)
    start_time = time.time()
    run_counter = 0
    for ae_run in autoencoder_runs:
        cache_path = encode_latent_cache(combined=combined, autoencoder_run=ae_run, out_dir=out_dir, batch_size=int(args.encode_batch_size), device=str(args.device))
        latent_dim = int(read_json(Path(ae_run["metrics_path"])) .get("latent_dim", ae_run.get("latent_dim", 0)) or ae_run.get("latent_dim", 0) or np.load(cache_path)["z_windows"].shape[-1])
        for cfg in configs:
            cid = config_id(cfg, ae_run)
            if cid in completed:
                continue
            if args.max_runs and run_counter >= int(args.max_runs):
                write_summary(out_dir, records, created_at=created_at, search_config=search_config)
                return 0
            if (time.time() - start_time) / 3600.0 >= float(args.time_limit_hours):
                write_summary(out_dir, records, created_at=created_at, search_config=search_config)
                return 0
            run_counter += 1
            index = len(records) + 1
            run_dir = out_dir / "runs" / cid
            record = {"index": index, "status": "running", "config_id": cid, "config": dict(cfg), "latent_dim": latent_dim, "run_dir": run_dir.as_posix(), "started_at": now()}
            write_json(out_dir / "sweep_active.json", record)
            print(f"{now()} start {index} {cid}", flush=True)
            try:
                run = train_one(cfg, combined=combined, cache_path=cache_path, autoencoder_run=ae_run, run_dir=run_dir, device=str(args.device))
                metrics = read_json(run["metrics_path"])
                keep = {k: metrics.get(k) for k in [
                    "test_improvement_over_persistence_mse",
                    "test_decoded_prediction_mse",
                    "test_persistence_mse",
                    "test_active_cell_improvement_over_persistence_mse",
                    "test_top_activity_improvement_over_persistence_mse",
                    "test_high_change_improvement_over_persistence_mse",
                    "abs_gate_mean_eval",
                    "accel_gate_mean_eval",
                    "improvement_over_persistence_mse",
                ] if k in metrics}
                record.update({"status": "completed", "completed_at": now(), "run_path": (run_dir / "hybrid_rnn_run.json").as_posix(), "metrics_path": run["metrics_path"], "checkpoint_path": run["checkpoint_path"], "metrics": keep})
                print(f"{now()} done {cid} test_improvement={keep.get('test_improvement_over_persistence_mse')}", flush=True)
            except Exception as exc:
                record.update({"status": "failed", "completed_at": now(), "error": repr(exc)})
                print(f"{now()} failed {cid}: {exc!r}", flush=True)
            records.append(record)
            with (out_dir / "shared_directional_hybrid_progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            write_summary(out_dir, records, created_at=created_at, search_config=search_config)
    active = {"state": "finished", "completed_at": now(), "records": len(records), "candidate_runs": len(configs) * len(autoencoder_runs), "pid": os.getpid()}
    write_json(out_dir / "sweep_active.json", active)
    summary = read_json(out_dir / "shared_directional_hybrid_summary.json")
    summary["state"] = "finished"
    summary["completed_at"] = active["completed_at"]
    write_json(out_dir / "shared_directional_hybrid_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
