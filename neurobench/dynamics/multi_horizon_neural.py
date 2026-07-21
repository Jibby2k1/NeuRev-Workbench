"""Shared multi-horizon neural latent predictors."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from neurobench.dynamics.error_analysis import promote_structured_error_metrics
from neurobench.dynamics.linear import _decode_latents, _encode_latent_windows
from neurobench.dynamics.models import GridAutoencoder
from neurobench.dynamics.multi_horizon_linear import _horizon_frames, _horizon_sec
from neurobench.dynamics.train import (
    _checkpoint_latent_stats,
    _normalize_prediction_target,
    _prediction_examples,
    _prepare_model_array,
    _promote_split_metrics,
    _split_mask,
    _split_name_for_video_id,
    _torch,
    _write_grid_preview,
)


class SharedHorizonLatentGRUPredictor(_torch().nn.Module):
    """Shared latent GRU with a horizon-conditioned output head."""

    def __init__(self, *, latent_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        torch = _torch()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = torch.nn.GRU(input_size=self.latent_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(self, z_window, horizon_value):
        output, _hidden = self.gru(z_window)
        h = horizon_value.reshape(-1, 1).to(dtype=output.dtype, device=output.device)
        return self.head(_torch().cat([output[:, -1, :], h], dim=1))


class SharedHorizonLatentTransformerPredictor(_torch().nn.Module):
    """Shared latent Transformer with a horizon-conditioned output head."""

    def __init__(
        self,
        *,
        latent_dim: int,
        model_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_window_frames: int = 64,
    ):
        torch = _torch()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.max_window_frames = int(max_window_frames)
        self.input = torch.nn.Linear(self.latent_dim, self.model_dim)
        self.position = torch.nn.Parameter(torch.zeros(1, self.max_window_frames, self.model_dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.norm = torch.nn.LayerNorm(self.model_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.model_dim + 1, self.model_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.model_dim),
            torch.nn.Linear(self.model_dim, self.latent_dim),
        )

    def forward(self, z_window, horizon_value):
        if z_window.shape[1] > self.max_window_frames:
            raise ValueError(f"Window has {z_window.shape[1]} frames, max_window_frames={self.max_window_frames}.")
        x = self.input(z_window) + self.position[:, : z_window.shape[1], :]
        encoded = self.encoder(x)
        h = horizon_value.reshape(-1, 1).to(dtype=encoded.dtype, device=encoded.device)
        return self.head(_torch().cat([self.norm(encoded[:, -1, :]), h], dim=1))


def train_shared_multi_horizon_latent_gru(
    *,
    datasets: Mapping[str, Mapping[str, Any]],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    prediction_target: str = "delta",
    seed: int = 7,
    device: str = "cpu",
    evaluation_batch_size: int | None = None,
    progress: Callable[[str], None] | None = None,
    progress_interval_epochs: int = 1,
) -> dict[str, Any]:
    """Train one horizon-conditioned latent GRU across multiple horizon datasets."""
    if len(datasets) < 2:
        raise ValueError("At least two horizon datasets are required for shared multi-horizon GRU training.")
    torch = _torch()
    torch.manual_seed(int(seed))
    if str(device) == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    prediction_target = _normalize_prediction_target(prediction_target)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_interval_epochs = max(1, int(progress_interval_epochs))
    _write_progress_event(
        out,
        phase="start",
        message="starting shared multi-horizon latent GRU training",
        progress=progress,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        prediction_target=prediction_target,
        seed=int(seed),
        device=str(device),
        dataset_count=len(datasets),
        evaluation_batch_size=int(evaluation_batch_size or batch_size),
    )

    dataset_items = sorted(datasets.items(), key=lambda item: _horizon_frames(item[1], item[0]) or 0)
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    base_channels = int(ckpt.get("base_channels", 16))
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    with np.load(dataset_items[0][1]["array_path"], allow_pickle=False) as arrays:
        input_channels = int(arrays["windows"].shape[2])
        input_shape = tuple(ckpt.get("input_shape") or arrays["windows"].shape[2:])
    ae = GridAutoencoder(input_channels=input_channels, latent_dim=latent_dim, base_channels=base_channels, input_shape=input_shape).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()

    horizon_values = [_horizon_frames(dataset, key) or (idx + 1) for idx, (key, dataset) in enumerate(dataset_items)]
    horizon_scale = float(max(horizon_values) or 1.0)
    encoded = []
    train_z_parts = []
    train_y_parts = []
    train_h_parts = []
    val_z_parts = []
    val_y_parts = []
    val_h_parts = []
    for idx, (dataset_key, dataset) in enumerate(dataset_items):
        horizon_norm = float(horizon_values[idx]) / horizon_scale
        _write_progress_event(
            out,
            phase="encode",
            message=f"encoding dataset {dataset_key}",
            progress=progress,
            dataset_key=str(dataset_key),
            horizon_frames=int(horizon_values[idx]),
            horizon_value=float(horizon_norm),
        )
        bundle = _encode_dataset_for_shared_gru(
            dataset_key=dataset_key,
            dataset=dataset,
            autoencoder=ae,
            latent_mean_np=latent_mean_np,
            latent_std_np=latent_std_np,
            horizon_value=horizon_norm,
            prediction_target=prediction_target,
            batch_size=int(batch_size),
            device=device,
        )
        encoded.append(bundle)
        _write_progress_event(
            out,
            phase="encode_done",
            message=f"encoded dataset {dataset_key}",
            progress=progress,
            dataset_key=str(dataset_key),
            window_count=int(bundle["z_window"].shape[0]),
            train_window_count=int(bundle["train_mask"].sum()),
            val_window_count=int(bundle["val_mask"].sum()),
        )
        train_mask = bundle["train_mask"]
        train_z_parts.append(bundle["z_window"][train_mask])
        train_y_parts.append(bundle["target_y"][train_mask])
        train_h_parts.append(np.full((int(train_mask.sum()),), horizon_norm, dtype=np.float32))
        if np.any(bundle["val_mask"]):
            val_mask = bundle["val_mask"]
            val_z_parts.append(bundle["z_window"][val_mask])
            val_y_parts.append(bundle["target_y"][val_mask])
            val_h_parts.append(np.full((int(val_mask.sum()),), horizon_norm, dtype=np.float32))

    if not train_z_parts or sum(part.shape[0] for part in train_z_parts) == 0:
        raise ValueError("Shared multi-horizon GRU training split is empty.")
    train_z_np = np.concatenate(train_z_parts, axis=0).astype(np.float32)
    train_y_np = np.concatenate(train_y_parts, axis=0).astype(np.float32)
    train_h_np = np.concatenate(train_h_parts, axis=0).astype(np.float32)
    val_z_np = np.concatenate(val_z_parts, axis=0).astype(np.float32) if val_z_parts else train_z_np
    val_y_np = np.concatenate(val_y_parts, axis=0).astype(np.float32) if val_y_parts else train_y_np
    val_h_np = np.concatenate(val_h_parts, axis=0).astype(np.float32) if val_h_parts else train_h_np
    selection_metric = "val_latent_code_mse" if val_z_parts else "train_latent_code_mse"

    _write_progress_event(
        out,
        phase="train_start",
        message="starting GRU optimization",
        progress=progress,
        training_window_count=int(train_z_np.shape[0]),
        validation_window_count=int(val_z_np.shape[0]),
        selection_metric=selection_metric,
    )
    model = SharedHorizonLatentGRUPredictor(latent_dim=latent_dim, hidden_dim=int(hidden_dim), num_layers=int(num_layers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    train_z = torch.from_numpy(train_z_np).to(device)
    train_y = torch.from_numpy(train_y_np).to(device)
    train_h = torch.from_numpy(train_h_np).to(device)
    train_indices = torch.arange(train_z.shape[0], device=device)
    losses: list[float] = []
    selection_losses: list[float] = []
    best_state: dict[str, Any] | None = None
    best_selection = float("inf")
    for _epoch in range(int(epochs)):
        epoch_number = int(_epoch) + 1
        model.train()
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], max(1, int(batch_size))):
            idx = perm[start : start + int(batch_size)]
            pred_step = model(train_z[idx], train_h[idx])
            loss = torch.mean((pred_step - train_y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        selection = _latent_step_mse(model, val_z_np, val_y_np, val_h_np, batch_size=int(batch_size), device=device)
        selection_losses.append(selection)
        if selection < best_selection:
            best_selection = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch_number == 1 or epoch_number == int(epochs) or epoch_number % progress_interval_epochs == 0:
            _write_progress_event(
                out,
                phase="train_epoch",
                message=f"finished epoch {epoch_number}/{int(epochs)}",
                progress=progress,
                epoch=epoch_number,
                epochs=int(epochs),
                train_latent_code_mse=float(losses[-1]),
                selection_latent_code_mse=float(selection),
                best_selection_latent_code_mse=float(best_selection),
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    per_horizon: dict[str, Any] = {}
    weighted_mse_sum = 0.0
    weighted_persistence_sum = 0.0
    total_windows = 0
    for bundle in encoded:
        _write_progress_event(
            out,
            phase="evaluate",
            message=f"evaluating dataset {bundle['dataset_key']}",
            progress=progress,
            dataset_key=str(bundle["dataset_key"]),
            window_count=int(bundle["z_window"].shape[0]),
        )
        horizon_metrics = _evaluate_encoded_dataset(
            bundle=bundle,
            dataset=bundle["dataset"],
            autoencoder=ae,
            model=model,
            latent_mean_np=latent_mean_np,
            latent_std_np=latent_std_np,
            prediction_target=prediction_target,
            batch_size=int(batch_size),
            evaluation_batch_size=int(evaluation_batch_size or batch_size),
            device=device,
            out_dir=out / str(bundle["dataset_key"]),
        )
        per_horizon[str(bundle["dataset_key"])] = horizon_metrics
        _write_progress_event(
            out,
            phase="evaluate_done",
            message=f"evaluated dataset {bundle['dataset_key']}",
            progress=progress,
            dataset_key=str(bundle["dataset_key"]),
            decoded_prediction_mse=float(horizon_metrics.get("decoded_prediction_mse") or 0.0),
            persistence_mse=float(horizon_metrics.get("persistence_mse") or 0.0),
            improvement_over_persistence_mse=float(horizon_metrics.get("improvement_over_persistence_mse") or 0.0),
        )
        n = int(horizon_metrics.get("evaluation_window_count") or 0)
        total_windows += n
        weighted_mse_sum += float(horizon_metrics.get("decoded_prediction_mse") or 0.0) * n
        weighted_persistence_sum += float(horizon_metrics.get("persistence_mse") or 0.0) * n

    overall_mse = weighted_mse_sum / max(total_windows, 1)
    overall_persistence = weighted_persistence_sum / max(total_windows, 1)
    metrics = {
        "schema_version": 1,
        "objective": "shared_multi_horizon_gru_delta_latent" if prediction_target == "delta" else "shared_multi_horizon_gru_absolute_latent",
        "model_kind": "shared_multi_horizon_latent_gru",
        "model_family": "multi_horizon_latent_gru",
        "prediction_target": prediction_target,
        "horizon_conditioning": "normalized_horizon_scalar_head_conditioning",
        "dataset_keys": [key for key, _ in dataset_items],
        "shared_horizons_frames": [int(v) for v in horizon_values],
        "training_loss": losses,
        "selection_loss": selection_losses,
        "selection_metric": selection_metric,
        "selection_latent_code_mse": float(best_selection),
        "per_horizon_metrics": per_horizon,
        "decoded_prediction_mse": float(overall_mse),
        "persistence_mse": float(overall_persistence),
        "improvement_over_persistence_mse": float(overall_persistence - overall_mse),
        "evaluation_window_count": int(total_windows),
        "training_window_count": int(train_z_np.shape[0]),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "evaluation_batch_size": int(evaluation_batch_size or batch_size),
        "decoded_evaluation_mode": "chunked",
        "latent_code_normalization": "standard_score_per_dimension",
        "decoded_output_normalization": "sigmoid_unit_interval",
        "progress_log_path": str(out / "multi_horizon_gru_progress.jsonl"),
        "progress_latest_path": str(out / "multi_horizon_gru_progress_latest.json"),
    }
    checkpoint = out / "multi_horizon_gru_checkpoint.pt"
    metrics_path = out / "multi_horizon_gru_metrics.json"
    run_path = out / "multi_horizon_gru_run.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_dim": int(latent_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "prediction_target": prediction_target,
            "shared_horizons_frames": [int(v) for v in horizon_values],
            "horizon_scale": float(horizon_scale),
            "latent_code_normalization": "standard_score_per_dimension",
            "predicted_code_space": "standardized_latent_delta" if prediction_target == "delta" else "standardized_latent",
        },
        checkpoint,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": out.name or "multi_horizon_gru_v1",
        "model_kind": "shared_multi_horizon_latent_gru",
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "dataset_keys": [key for key, _ in dataset_items],
        "shared_horizons_frames": [int(v) for v in horizon_values],
        "prediction_target": prediction_target,
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "device": str(device),
        "warnings": [],
        "extras": {
            "horizon_conditioning": "normalized_horizon_scalar_head_conditioning",
            "train_window_count": int(train_z_np.shape[0]),
            "selection_metric": selection_metric,
            "decoded_prediction_used_for_training": False,
            "decoded_evaluation_mode": "chunked",
            "evaluation_batch_size": int(evaluation_batch_size or batch_size),
            "progress_log_path": str(out / "multi_horizon_gru_progress.jsonl"),
            "progress_latest_path": str(out / "multi_horizon_gru_progress_latest.json"),
        },
    }
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress_event(
        out,
        phase="complete",
        message="completed shared multi-horizon latent GRU training",
        progress=progress,
        metrics_path=str(metrics_path),
        run_path=str(run_path),
        checkpoint_path=str(checkpoint),
        decoded_prediction_mse=float(overall_mse),
        persistence_mse=float(overall_persistence),
        improvement_over_persistence_mse=float(overall_persistence - overall_mse),
    )
    return run


def train_shared_multi_horizon_latent_transformer(
    *,
    datasets: Mapping[str, Mapping[str, Any]],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    model_dim: int = 64,
    num_heads: int = 2,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    prediction_target: str = "delta",
    seed: int = 7,
    device: str = "cpu",
    evaluation_batch_size: int | None = None,
    progress: Callable[[str], None] | None = None,
    progress_interval_epochs: int = 1,
) -> dict[str, Any]:
    """Train one horizon-conditioned latent Transformer across multiple horizon datasets."""
    if len(datasets) < 2:
        raise ValueError("At least two horizon datasets are required for shared multi-horizon Transformer training.")
    torch = _torch()
    torch.manual_seed(int(seed))
    if str(device) == "cpu" and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    prediction_target = _normalize_prediction_target(prediction_target)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_interval_epochs = max(1, int(progress_interval_epochs))
    artifact_prefix = "multi_horizon_transformer"
    display_label = "shared-transformer"
    _write_progress_event(
        out,
        phase="start",
        message="starting shared multi-horizon latent Transformer training",
        progress=progress,
        artifact_prefix=artifact_prefix,
        display_label=display_label,
        model_dim=int(model_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        dropout=float(dropout),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        prediction_target=prediction_target,
        seed=int(seed),
        device=str(device),
        dataset_count=len(datasets),
        evaluation_batch_size=int(evaluation_batch_size or batch_size),
    )

    dataset_items = sorted(datasets.items(), key=lambda item: _horizon_frames(item[1], item[0]) or 0)
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    base_channels = int(ckpt.get("base_channels", 16))
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    with np.load(dataset_items[0][1]["array_path"], allow_pickle=False) as arrays:
        input_channels = int(arrays["windows"].shape[2])
        input_shape = tuple(ckpt.get("input_shape") or arrays["windows"].shape[2:])
    ae = GridAutoencoder(input_channels=input_channels, latent_dim=latent_dim, base_channels=base_channels, input_shape=input_shape).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()

    horizon_values = [_horizon_frames(dataset, key) or (idx + 1) for idx, (key, dataset) in enumerate(dataset_items)]
    horizon_scale = float(max(horizon_values) or 1.0)
    encoded = []
    train_z_parts = []
    train_y_parts = []
    train_h_parts = []
    val_z_parts = []
    val_y_parts = []
    val_h_parts = []
    for idx, (dataset_key, dataset) in enumerate(dataset_items):
        horizon_norm = float(horizon_values[idx]) / horizon_scale
        _write_progress_event(
            out,
            phase="encode",
            message=f"encoding dataset {dataset_key}",
            progress=progress,
            artifact_prefix=artifact_prefix,
            display_label=display_label,
            dataset_key=str(dataset_key),
            horizon_frames=int(horizon_values[idx]),
            horizon_value=float(horizon_norm),
        )
        bundle = _encode_dataset_for_shared_gru(
            dataset_key=dataset_key,
            dataset=dataset,
            autoencoder=ae,
            latent_mean_np=latent_mean_np,
            latent_std_np=latent_std_np,
            horizon_value=horizon_norm,
            prediction_target=prediction_target,
            batch_size=int(batch_size),
            device=device,
        )
        encoded.append(bundle)
        _write_progress_event(
            out,
            phase="encode_done",
            message=f"encoded dataset {dataset_key}",
            progress=progress,
            artifact_prefix=artifact_prefix,
            display_label=display_label,
            dataset_key=str(dataset_key),
            window_count=int(bundle["z_window"].shape[0]),
            train_window_count=int(bundle["train_mask"].sum()),
            val_window_count=int(bundle["val_mask"].sum()),
        )
        train_mask = bundle["train_mask"]
        train_z_parts.append(bundle["z_window"][train_mask])
        train_y_parts.append(bundle["target_y"][train_mask])
        train_h_parts.append(np.full((int(train_mask.sum()),), horizon_norm, dtype=np.float32))
        if np.any(bundle["val_mask"]):
            val_mask = bundle["val_mask"]
            val_z_parts.append(bundle["z_window"][val_mask])
            val_y_parts.append(bundle["target_y"][val_mask])
            val_h_parts.append(np.full((int(val_mask.sum()),), horizon_norm, dtype=np.float32))

    if not train_z_parts or sum(part.shape[0] for part in train_z_parts) == 0:
        raise ValueError("Shared multi-horizon Transformer training split is empty.")
    train_z_np = np.concatenate(train_z_parts, axis=0).astype(np.float32)
    train_y_np = np.concatenate(train_y_parts, axis=0).astype(np.float32)
    train_h_np = np.concatenate(train_h_parts, axis=0).astype(np.float32)
    val_z_np = np.concatenate(val_z_parts, axis=0).astype(np.float32) if val_z_parts else train_z_np
    val_y_np = np.concatenate(val_y_parts, axis=0).astype(np.float32) if val_y_parts else train_y_np
    val_h_np = np.concatenate(val_h_parts, axis=0).astype(np.float32) if val_h_parts else train_h_np
    selection_metric = "val_latent_code_mse" if val_z_parts else "train_latent_code_mse"

    _write_progress_event(
        out,
        phase="train_start",
        message="starting Transformer optimization",
        progress=progress,
        artifact_prefix=artifact_prefix,
        display_label=display_label,
        training_window_count=int(train_z_np.shape[0]),
        validation_window_count=int(val_z_np.shape[0]),
        selection_metric=selection_metric,
    )
    model = SharedHorizonLatentTransformerPredictor(
        latent_dim=latent_dim,
        model_dim=int(model_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        dropout=float(dropout),
        max_window_frames=max(64, int(train_z_np.shape[1])),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    train_z = torch.from_numpy(train_z_np).to(device)
    train_y = torch.from_numpy(train_y_np).to(device)
    train_h = torch.from_numpy(train_h_np).to(device)
    train_indices = torch.arange(train_z.shape[0], device=device)
    losses: list[float] = []
    selection_losses: list[float] = []
    best_state: dict[str, Any] | None = None
    best_selection = float("inf")
    for _epoch in range(int(epochs)):
        epoch_number = int(_epoch) + 1
        model.train()
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], max(1, int(batch_size))):
            idx = perm[start : start + int(batch_size)]
            pred_step = model(train_z[idx], train_h[idx])
            loss = torch.mean((pred_step - train_y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        selection = _latent_step_mse(model, val_z_np, val_y_np, val_h_np, batch_size=int(batch_size), device=device)
        selection_losses.append(selection)
        if selection < best_selection:
            best_selection = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch_number == 1 or epoch_number == int(epochs) or epoch_number % progress_interval_epochs == 0:
            _write_progress_event(
                out,
                phase="train_epoch",
                message=f"finished epoch {epoch_number}/{int(epochs)}",
                progress=progress,
                artifact_prefix=artifact_prefix,
                display_label=display_label,
                epoch=epoch_number,
                epochs=int(epochs),
                train_latent_code_mse=float(losses[-1]),
                selection_latent_code_mse=float(selection),
                best_selection_latent_code_mse=float(best_selection),
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    per_horizon: dict[str, Any] = {}
    weighted_mse_sum = 0.0
    weighted_persistence_sum = 0.0
    total_windows = 0
    for bundle in encoded:
        _write_progress_event(
            out,
            phase="evaluate",
            message=f"evaluating dataset {bundle['dataset_key']}",
            progress=progress,
            artifact_prefix=artifact_prefix,
            display_label=display_label,
            dataset_key=str(bundle["dataset_key"]),
            window_count=int(bundle["z_window"].shape[0]),
        )
        horizon_metrics = _evaluate_encoded_dataset(
            bundle=bundle,
            dataset=bundle["dataset"],
            autoencoder=ae,
            model=model,
            latent_mean_np=latent_mean_np,
            latent_std_np=latent_std_np,
            prediction_target=prediction_target,
            batch_size=int(batch_size),
            evaluation_batch_size=int(evaluation_batch_size or batch_size),
            device=device,
            out_dir=out / str(bundle["dataset_key"]),
        )
        per_horizon[str(bundle["dataset_key"])] = horizon_metrics
        _write_progress_event(
            out,
            phase="evaluate_done",
            message=f"evaluated dataset {bundle['dataset_key']}",
            progress=progress,
            artifact_prefix=artifact_prefix,
            display_label=display_label,
            dataset_key=str(bundle["dataset_key"]),
            decoded_prediction_mse=float(horizon_metrics.get("decoded_prediction_mse") or 0.0),
            persistence_mse=float(horizon_metrics.get("persistence_mse") or 0.0),
            improvement_over_persistence_mse=float(horizon_metrics.get("improvement_over_persistence_mse") or 0.0),
        )
        n = int(horizon_metrics.get("evaluation_window_count") or 0)
        total_windows += n
        weighted_mse_sum += float(horizon_metrics.get("decoded_prediction_mse") or 0.0) * n
        weighted_persistence_sum += float(horizon_metrics.get("persistence_mse") or 0.0) * n

    overall_mse = weighted_mse_sum / max(total_windows, 1)
    overall_persistence = weighted_persistence_sum / max(total_windows, 1)
    metrics = {
        "schema_version": 1,
        "objective": "shared_multi_horizon_transformer_delta_latent" if prediction_target == "delta" else "shared_multi_horizon_transformer_absolute_latent",
        "model_kind": "shared_multi_horizon_latent_transformer",
        "model_family": "multi_horizon_latent_transformer",
        "prediction_target": prediction_target,
        "horizon_conditioning": "normalized_horizon_scalar_head_conditioning",
        "dataset_keys": [key for key, _ in dataset_items],
        "shared_horizons_frames": [int(v) for v in horizon_values],
        "training_loss": losses,
        "selection_loss": selection_losses,
        "selection_metric": selection_metric,
        "selection_latent_code_mse": float(best_selection),
        "per_horizon_metrics": per_horizon,
        "decoded_prediction_mse": float(overall_mse),
        "persistence_mse": float(overall_persistence),
        "improvement_over_persistence_mse": float(overall_persistence - overall_mse),
        "evaluation_window_count": int(total_windows),
        "training_window_count": int(train_z_np.shape[0]),
        "latent_dim": int(latent_dim),
        "model_dim": int(model_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "evaluation_batch_size": int(evaluation_batch_size or batch_size),
        "decoded_evaluation_mode": "chunked",
        "latent_code_normalization": "standard_score_per_dimension",
        "decoded_output_normalization": "sigmoid_unit_interval",
        "progress_log_path": str(out / "multi_horizon_transformer_progress.jsonl"),
        "progress_latest_path": str(out / "multi_horizon_transformer_progress_latest.json"),
    }
    checkpoint = out / "multi_horizon_transformer_checkpoint.pt"
    metrics_path = out / "multi_horizon_transformer_metrics.json"
    run_path = out / "multi_horizon_transformer_run.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_dim": int(latent_dim),
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "prediction_target": prediction_target,
            "shared_horizons_frames": [int(v) for v in horizon_values],
            "horizon_scale": float(horizon_scale),
            "latent_code_normalization": "standard_score_per_dimension",
            "predicted_code_space": "standardized_latent_delta" if prediction_target == "delta" else "standardized_latent",
        },
        checkpoint,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": out.name or "multi_horizon_transformer_v1",
        "model_kind": "shared_multi_horizon_latent_transformer",
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "dataset_keys": [key for key, _ in dataset_items],
        "shared_horizons_frames": [int(v) for v in horizon_values],
        "prediction_target": prediction_target,
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "device": str(device),
        "warnings": [],
        "extras": {
            "horizon_conditioning": "normalized_horizon_scalar_head_conditioning",
            "train_window_count": int(train_z_np.shape[0]),
            "selection_metric": selection_metric,
            "decoded_prediction_used_for_training": False,
            "decoded_evaluation_mode": "chunked",
            "evaluation_batch_size": int(evaluation_batch_size or batch_size),
            "progress_log_path": str(out / "multi_horizon_transformer_progress.jsonl"),
            "progress_latest_path": str(out / "multi_horizon_transformer_progress_latest.json"),
        },
    }
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress_event(
        out,
        phase="complete",
        message="completed shared multi-horizon latent Transformer training",
        progress=progress,
        artifact_prefix=artifact_prefix,
        display_label=display_label,
        metrics_path=str(metrics_path),
        run_path=str(run_path),
        checkpoint_path=str(checkpoint),
        decoded_prediction_mse=float(overall_mse),
        persistence_mse=float(overall_persistence),
        improvement_over_persistence_mse=float(overall_persistence - overall_mse),
    )
    return run


def _progress_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_progress_event(
    out_dir: Path,
    *,
    phase: str,
    message: str,
    progress: Callable[[str], None] | None = None,
    artifact_prefix: str = "multi_horizon_gru",
    display_label: str = "shared-gru",
    **fields: Any,
) -> None:
    event = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": str(phase),
        "message": str(message),
    }
    for key, value in fields.items():
        if value is not None:
            event[str(key)] = _progress_json_value(value)
    safe_prefix = str(artifact_prefix).strip() or "multi_horizon_gru"
    log_path = out_dir / f"{safe_prefix}_progress.jsonl"
    latest_path = out_dir / f"{safe_prefix}_progress_latest.json"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    latest_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress is not None:
        progress(f"[{display_label}] {phase}: {message}")


def _encode_dataset_for_shared_gru(
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
    target_y = (target_z - z_window[:, -1, :]).astype(np.float32) if prediction_target == "delta" else target_z.astype(np.float32)
    train_mask = _split_mask(window_video_ids, dataset.get("splits"), "train", default_all=True)
    val_mask = _split_mask(window_video_ids, dataset.get("splits"), "val", default_all=False)
    return {
        "dataset_key": dataset_key,
        "dataset": dict(dataset),
        "horizon_value": float(horizon_value),
        "z_window": z_window.astype(np.float32),
        "target_z": target_z.astype(np.float32),
        "target_y": target_y,
        "window_video_ids": window_video_ids,
        "train_mask": train_mask,
        "val_mask": val_mask,
    }


def _latent_step_mse(model: SharedHorizonLatentGRUPredictor, z_window: np.ndarray, target_y: np.ndarray, horizon_values: np.ndarray, *, batch_size: int, device: str) -> float:
    torch = _torch()
    if z_window.shape[0] == 0:
        return float("inf")
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(z_window.shape[0]), max(1, int(batch_size))):
            z = torch.from_numpy(z_window[start : start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            y = torch.from_numpy(target_y[start : start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            h = torch.from_numpy(horizon_values[start : start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            pred = model(z, h)
            losses.append(float(torch.mean((pred - y) ** 2).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _predict_standardized_latents(model: SharedHorizonLatentGRUPredictor, z_window: np.ndarray, horizon_value: float, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(z_window.shape[0]), max(1, int(batch_size))):
            z = torch.from_numpy(z_window[start : start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            h = torch.full((z.shape[0],), float(horizon_value), dtype=torch.float32, device=device)
            chunks.append(model(z, h).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.latent_dim), dtype=np.float32)



def _evaluate_encoded_dataset(
    *,
    bundle: Mapping[str, Any],
    dataset: Mapping[str, Any],
    autoencoder: GridAutoencoder,
    model: SharedHorizonLatentGRUPredictor,
    latent_mean_np: np.ndarray,
    latent_std_np: np.ndarray,
    prediction_target: str,
    batch_size: int,
    evaluation_batch_size: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        stored_start_indices = arrays["window_start_indices"].astype(np.int64) if "window_start_indices" in arrays.files else None
        stored_end_indices = arrays["window_end_indices"].astype(np.int64) if "window_end_indices" in arrays.files else None
        stored_target_indices = arrays["target_frame_indices"].astype(np.int64) if "target_frame_indices" in arrays.files else None
    z_window = np.asarray(bundle["z_window"], dtype=np.float32)
    target_z = np.asarray(bundle["target_z"], dtype=np.float32)
    pred_step = _predict_standardized_latents(model, z_window, float(bundle["horizon_value"]), batch_size=batch_size, device=device)
    pred_z = z_window[:, -1, :] + pred_step if prediction_target == "delta" else pred_step
    pred_z_raw = pred_z * latent_std_np.reshape(1, -1) + latent_mean_np.reshape(1, -1)
    target_z_raw = target_z * latent_std_np.reshape(1, -1) + latent_mean_np.reshape(1, -1)
    latent_diff = pred_z - target_z
    latent_raw_diff = pred_z_raw.astype(np.float32) - target_z_raw.astype(np.float32)
    video_ids = np.asarray(bundle["window_video_ids"]).astype(str)
    window_start_indices, window_end_indices, target_frame_indices = _window_frame_indices(
        video_ids=video_ids,
        windowing=dataset.get("windowing"),
        stored_start_indices=stored_start_indices,
        stored_end_indices=stored_end_indices,
        stored_target_indices=stored_target_indices,
    )
    split_masks = {
        "train": _split_mask(video_ids, dataset.get("splits"), "train", default_all=True),
        "val": _split_mask(video_ids, dataset.get("splits"), "val", default_all=False),
        "test": _split_mask(video_ids, dataset.get("splits"), "test", default_all=False),
    }
    split_metrics = _latent_split_metric_shell(latent_diff, latent_raw_diff, split_masks)
    thresholds = _structured_thresholds(targets, windows[:, -1], split_masks["train"])
    structured_acc = {split: _new_structured_acc() for split in ("train", "val", "test", "all")}
    decoded_acc = {split: {"decoded_sq": 0.0, "decoded_abs": 0.0, "persistence_sq": 0.0, "persistence_abs": 0.0, "cell_count": 0} for split in ("train", "val", "test")}
    overall_decoded_sq = 0.0
    overall_decoded_abs = 0.0
    overall_persistence_sq = 0.0
    overall_persistence_abs = 0.0
    overall_cell_count = 0
    examples: list[dict[str, Any]] | None = None
    clip_rows: list[dict[str, Any]] = []
    preview_written = False
    eval_batch = max(1, int(evaluation_batch_size))
    for start in range(0, int(pred_z_raw.shape[0]), eval_batch):
        stop = min(start + eval_batch, int(pred_z_raw.shape[0]))
        pred_x = _decode_latents(autoencoder, pred_z_raw[start:stop].astype(np.float32), batch_size=batch_size, device=device)
        target_chunk = targets[start:stop]
        last_chunk = windows[start:stop, -1]
        decoded_diff = pred_x - target_chunk
        persistence_diff = last_chunk - target_chunk
        overall_decoded_sq += float(np.sum(decoded_diff * decoded_diff, dtype=np.float64))
        overall_decoded_abs += float(np.sum(np.abs(decoded_diff), dtype=np.float64))
        overall_persistence_sq += float(np.sum(persistence_diff * persistence_diff, dtype=np.float64))
        overall_persistence_abs += float(np.sum(np.abs(persistence_diff), dtype=np.float64))
        overall_cell_count += int(decoded_diff.size)
        for split_name, full_mask in split_masks.items():
            mask = full_mask[start:stop]
            if not np.any(mask):
                continue
            dd = decoded_diff[mask]
            pd = persistence_diff[mask]
            decoded_acc[split_name]["decoded_sq"] += float(np.sum(dd * dd, dtype=np.float64))
            decoded_acc[split_name]["decoded_abs"] += float(np.sum(np.abs(dd), dtype=np.float64))
            decoded_acc[split_name]["persistence_sq"] += float(np.sum(pd * pd, dtype=np.float64))
            decoded_acc[split_name]["persistence_abs"] += float(np.sum(np.abs(pd), dtype=np.float64))
            decoded_acc[split_name]["cell_count"] += int(dd.size)
            _update_structured_acc(structured_acc[split_name], dd, pd, target_chunk[mask], last_chunk[mask], thresholds)
        _update_structured_acc(structured_acc["all"], decoded_diff, persistence_diff, target_chunk, last_chunk, thresholds)
        if examples is None:
            example_count = min(3, int(pred_x.shape[0]))
            examples = _prediction_examples(
                windows[start : start + example_count],
                targets[start : start + example_count],
                pred_x[:example_count],
                max_examples=3,
                video_ids=video_ids[start : start + example_count],
                splits=dataset.get("splits"),
                windowing=dataset.get("windowing"),
                window_start_indices=window_start_indices[start : start + example_count],
                window_end_indices=window_end_indices[start : start + example_count],
                target_frame_indices=target_frame_indices[start : start + example_count],
            )
        _append_clip_rows(
            clip_rows,
            windows=windows[start:stop],
            targets=target_chunk,
            pred=pred_x,
            video_ids=video_ids[start:stop],
            splits=dataset.get("splits"),
            windowing=dataset.get("windowing"),
            window_start_indices=window_start_indices[start:stop],
            window_end_indices=window_end_indices[start:stop],
            target_frame_indices=target_frame_indices[start:stop],
            max_rows=24,
        )
        if not preview_written and pred_x.shape[0]:
            _write_grid_preview(out_dir / "prediction_examples.png", target_chunk[0, 0], pred_x[0, 0], np.abs(target_chunk[0, 0] - pred_x[0, 0]))
            preview_written = True
    _attach_decoded_split_metrics(split_metrics, decoded_acc)
    structured_error_metrics = _finalize_structured_metrics(structured_acc, thresholds)
    decoded_prediction_mse = overall_decoded_sq / max(overall_cell_count, 1)
    decoded_prediction_mae = overall_decoded_abs / max(overall_cell_count, 1)
    persistence_mse = overall_persistence_sq / max(overall_cell_count, 1)
    persistence_mae = overall_persistence_abs / max(overall_cell_count, 1)
    metrics = {
        "dataset_key": str(bundle["dataset_key"]),
        "prediction_horizon_frames": _horizon_frames(dataset, str(bundle["dataset_key"])),
        "prediction_horizon_sec": _horizon_sec(dataset),
        "decoded_prediction_mse": float(decoded_prediction_mse),
        "decoded_prediction_mae": float(decoded_prediction_mae),
        "persistence_mse": float(persistence_mse),
        "persistence_mae": float(persistence_mae),
        "latent_code_mse": float(np.mean(latent_diff * latent_diff)),
        "latent_code_mae": float(np.mean(np.abs(latent_diff))),
        "latent_code_raw_mse": float(np.mean(latent_raw_diff * latent_raw_diff)),
        "latent_code_raw_mae": float(np.mean(np.abs(latent_raw_diff))),
        "split_metrics": split_metrics,
        "structured_error_metrics": structured_error_metrics,
        "evaluation_window_count": int(z_window.shape[0]),
        "evaluation_batch_size": int(eval_batch),
        "decoded_evaluation_mode": "chunked",
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
    examples_path = out_dir / "prediction_examples.json"
    clips_path = out_dir / "prediction_clip_examples.json"
    review_metrics_path = out_dir / "per_horizon_metrics_for_review.json"
    examples_path.write_text(json.dumps({"schema_version": 1, "examples": examples or []}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clip_examples = _prediction_clip_examples_from_rows(clip_rows, max_clips=2, max_clip_frames=8)
    clips_path.write_text(
        json.dumps({"schema_version": 1, "clips": clip_examples, "clip_count": len(clip_examples)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics["prediction_examples_path"] = str(examples_path)
    metrics["prediction_clip_examples_path"] = str(clips_path)
    metrics["per_horizon_metrics_for_review_path"] = str(review_metrics_path)
    review_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _window_frame_indices(
    *,
    video_ids: np.ndarray,
    windowing: Mapping[str, Any] | None,
    stored_start_indices: np.ndarray | None,
    stored_end_indices: np.ndarray | None,
    stored_target_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(video_ids.shape[0])
    window_frames = int((windowing or {}).get("window_frames") or 1)
    horizon_frames = int((windowing or {}).get("prediction_horizon_frames") or 1)
    stride_frames = int((windowing or {}).get("stride_frames") or 1)
    if stored_start_indices is not None and stored_start_indices.shape[0] == count:
        starts = stored_start_indices.astype(np.int64, copy=False)
    else:
        seen: dict[str, int] = {}
        inferred = []
        for video_id in video_ids.astype(str):
            ordinal = seen.get(str(video_id), 0)
            inferred.append(ordinal * stride_frames)
            seen[str(video_id)] = ordinal + 1
        starts = np.asarray(inferred, dtype=np.int64)
    ends = stored_end_indices.astype(np.int64, copy=False) if stored_end_indices is not None and stored_end_indices.shape[0] == count else starts + max(0, window_frames - 1)
    targets = stored_target_indices.astype(np.int64, copy=False) if stored_target_indices is not None and stored_target_indices.shape[0] == count else ends + horizon_frames
    return starts, ends, targets


def _append_clip_rows(
    rows: list[dict[str, Any]],
    *,
    windows: np.ndarray,
    targets: np.ndarray,
    pred: np.ndarray,
    video_ids: np.ndarray,
    splits: Any,
    windowing: Mapping[str, Any] | None,
    window_start_indices: np.ndarray,
    window_end_indices: np.ndarray,
    target_frame_indices: np.ndarray,
    max_rows: int,
) -> None:
    remaining = max(0, int(max_rows) - len(rows))
    if remaining <= 0:
        return
    ids = np.asarray(video_ids).astype(str)
    for i in range(min(remaining, int(targets.shape[0]))):
        item = {
            "index": len(rows),
            "video_id": str(ids[i]) if i < ids.shape[0] else None,
            "split": _split_name_for_video_id(str(ids[i]), splits) if i < ids.shape[0] else None,
            "window_start_index": int(window_start_indices[i]),
            "window_end_index": int(window_end_indices[i]),
            "target_frame_index": int(target_frame_indices[i]),
            "input_last": windows[i, -1, 0].round(5).tolist(),
            "target_next": targets[i, 0].round(5).tolist(),
            "predicted_next": pred[i, 0].round(5).tolist(),
            "persistence_next": windows[i, -1, 0].round(5).tolist(),
            "abs_error_mean": float(np.mean(np.abs(targets[i] - pred[i]))),
            "persistence_abs_error_mean": float(np.mean(np.abs(targets[i] - windows[i, -1]))),
        }
        if isinstance(windowing, Mapping):
            for key in ("window_frames", "temporal_stride_frames", "prediction_horizon_frames", "prediction_horizon_sec", "effective_frame_rate_hz", "source_frame_rate_hz"):
                if key in windowing:
                    item[key] = windowing.get(key)
        rows.append(item)


def _prediction_clip_examples_from_rows(rows: list[dict[str, Any]], *, max_clips: int, max_clip_frames: int) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2 and len(clips) < int(max_clips):
            first = current[0]
            clips.append(
                {
                    "clip_index": len(clips),
                    "video_id": first.get("video_id"),
                    "split": first.get("split"),
                    "start_target_frame_index": first.get("target_frame_index"),
                    "end_target_frame_index": current[-1].get("target_frame_index"),
                    "frame_count": len(current),
                    "frames": current,
                }
            )
        current = []

    for row in rows:
        if len(clips) >= int(max_clips):
            break
        same_video = bool(current and row.get("video_id") == current[-1].get("video_id"))
        consecutive = bool(current and row.get("target_frame_index") == int(current[-1].get("target_frame_index", -10**9)) + 1)
        if current and (not same_video or not consecutive or len(current) >= int(max_clip_frames)):
            flush()
        current.append(row)
        if len(current) >= int(max_clip_frames):
            flush()
    if len(clips) < int(max_clips):
        flush()
    return clips


def _latent_split_metric_shell(latent_diff: np.ndarray, latent_raw_diff: np.ndarray, split_masks: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for split_name, mask in split_masks.items():
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
        ld = latent_diff[mask]
        lrd = latent_raw_diff[mask]
        payload[split_name] = {
            "window_count": count,
            "latent_code_mse": float(np.mean(ld * ld)),
            "latent_code_mae": float(np.mean(np.abs(ld))),
            "latent_code_raw_mse": float(np.mean(lrd * lrd)),
            "latent_code_raw_mae": float(np.mean(np.abs(lrd))),
            "decoded_prediction_mse": None,
            "decoded_prediction_mae": None,
            "persistence_mse": None,
            "persistence_mae": None,
        }
    return payload


def _attach_decoded_split_metrics(split_metrics: dict[str, dict[str, Any]], decoded_acc: Mapping[str, Mapping[str, float]]) -> None:
    for split_name, acc in decoded_acc.items():
        cell_count = int(acc.get("cell_count") or 0)
        if cell_count == 0:
            continue
        split_metrics[split_name]["decoded_prediction_mse"] = float(acc["decoded_sq"] / cell_count)
        split_metrics[split_name]["decoded_prediction_mae"] = float(acc["decoded_abs"] / cell_count)
        split_metrics[split_name]["persistence_mse"] = float(acc["persistence_sq"] / cell_count)
        split_metrics[split_name]["persistence_mae"] = float(acc["persistence_abs"] / cell_count)


def _structured_thresholds(targets: np.ndarray, last_frames: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    reference = targets[train_mask] if np.any(train_mask) else targets
    change_reference = np.abs(targets[train_mask] - last_frames[train_mask]) if np.any(train_mask) else np.abs(targets - last_frames)
    active_percentile = 90.0
    top_activity_percent = 5.0
    high_change_percentile = 90.0
    return {
        "active_percentile": active_percentile,
        "active_threshold": _percentile_threshold(reference, active_percentile),
        "top_activity_percent": top_activity_percent,
        "top_activity_threshold": _percentile_threshold(reference, 100.0 - top_activity_percent),
        "high_change_percentile": high_change_percentile,
        "high_change_threshold": _percentile_threshold(change_reference, high_change_percentile),
    }


def _percentile_threshold(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, float(np.clip(percentile, 0.0, 100.0))))


def _new_structured_acc() -> dict[str, float]:
    return {
        "window_count": 0,
        "cell_count": 0,
        "active_cell_count": 0,
        "active_pred_sq": 0.0,
        "active_persist_sq": 0.0,
        "inactive_cell_count": 0,
        "inactive_pred_sq": 0.0,
        "inactive_persist_sq": 0.0,
        "top_activity_cell_count": 0,
        "top_activity_pred_sq": 0.0,
        "top_activity_persist_sq": 0.0,
        "high_change_cell_count": 0,
        "high_change_pred_sq": 0.0,
        "high_change_persist_sq": 0.0,
    }


def _update_structured_acc(acc: dict[str, float], pred: np.ndarray, persist: np.ndarray, target: np.ndarray, last: np.ndarray, thresholds: Mapping[str, float]) -> None:
    if pred.size == 0:
        return
    active = target >= float(thresholds["active_threshold"])
    top = target >= float(thresholds["top_activity_threshold"])
    high_change = np.abs(target - last) >= float(thresholds["high_change_threshold"])
    inactive = ~active
    pred_sq = pred * pred
    persist_sq = persist * persist
    acc["window_count"] += int(pred.shape[0])
    acc["cell_count"] += int(pred.size)
    _add_masked_sq(acc, "active", pred_sq, persist_sq, active)
    _add_masked_sq(acc, "inactive", pred_sq, persist_sq, inactive)
    _add_masked_sq(acc, "top_activity", pred_sq, persist_sq, top)
    _add_masked_sq(acc, "high_change", pred_sq, persist_sq, high_change)


def _add_masked_sq(acc: dict[str, float], prefix: str, pred_sq: np.ndarray, persist_sq: np.ndarray, mask: np.ndarray) -> None:
    count = int(mask.sum())
    acc[f"{prefix}_cell_count"] += count
    if count:
        acc[f"{prefix}_pred_sq"] += float(np.sum(pred_sq[mask], dtype=np.float64))
        acc[f"{prefix}_persist_sq"] += float(np.sum(persist_sq[mask], dtype=np.float64))


def _mean_or_none(total: float, count: int) -> float | None:
    if count == 0:
        return None
    return float(total / count)


def _improvement_or_none(pred_total: float, persist_total: float, count: int) -> float | None:
    if count == 0:
        return None
    return float((persist_total - pred_total) / count)


def _finalize_structured_metrics(accumulators: Mapping[str, Mapping[str, float]], thresholds: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test", "all"):
        acc = accumulators[split_name]
        cell_count = int(acc["cell_count"])
        active_count = int(acc["active_cell_count"])
        inactive_count = int(acc["inactive_cell_count"])
        top_count = int(acc["top_activity_cell_count"])
        high_count = int(acc["high_change_cell_count"])
        out[split_name] = {
            "window_count": int(acc["window_count"]),
            "cell_count": cell_count,
            "active_cell_count": active_count,
            "active_cell_fraction": float(active_count / cell_count) if cell_count else 0.0,
            "active_cell_mse": _mean_or_none(float(acc["active_pred_sq"]), active_count),
            "active_cell_persistence_mse": _mean_or_none(float(acc["active_persist_sq"]), active_count),
            "active_cell_improvement_over_persistence_mse": _improvement_or_none(float(acc["active_pred_sq"]), float(acc["active_persist_sq"]), active_count),
            "inactive_cell_mse": _mean_or_none(float(acc["inactive_pred_sq"]), inactive_count),
            "inactive_cell_persistence_mse": _mean_or_none(float(acc["inactive_persist_sq"]), inactive_count),
            "inactive_cell_improvement_over_persistence_mse": _improvement_or_none(float(acc["inactive_pred_sq"]), float(acc["inactive_persist_sq"]), inactive_count),
            "top_activity_cell_count": top_count,
            "top_activity_mse": _mean_or_none(float(acc["top_activity_pred_sq"]), top_count),
            "top_activity_persistence_mse": _mean_or_none(float(acc["top_activity_persist_sq"]), top_count),
            "top_activity_improvement_over_persistence_mse": _improvement_or_none(float(acc["top_activity_pred_sq"]), float(acc["top_activity_persist_sq"]), top_count),
            "high_change_cell_count": high_count,
            "high_change_fraction": float(high_count / cell_count) if cell_count else 0.0,
            "high_change_mse": _mean_or_none(float(acc["high_change_pred_sq"]), high_count),
            "high_change_persistence_mse": _mean_or_none(float(acc["high_change_persist_sq"]), high_count),
            "high_change_improvement_over_persistence_mse": _improvement_or_none(float(acc["high_change_pred_sq"]), float(acc["high_change_persist_sq"]), high_count),
        }
    out["thresholds"] = dict(thresholds)
    return out
