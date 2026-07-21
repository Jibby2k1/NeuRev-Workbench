"""Conservative concept tests for grid latent dynamics variants.

These experiments are intentionally separate from the stable dynamics CLI. They
are small probes for ideas that may or may not graduate into the main pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile
from typing import Any, Iterable, Mapping

import numpy as np

from neurobench.dynamics.error_analysis import (
    promote_structured_error_metrics,
    structured_prediction_error_metrics,
)
from neurobench.dynamics.models import (
    GridAutoencoder,
    LatentGRUPredictor,
    PixelConvGRUResidual,
    PixelConvLSTMResidual,
    TemporalCNNResidual,
    UNetConvGRUResidual,
)


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for dynamics concept tests.") from exc
    return torch


class ResidualPixelGRU:
    """Lazy wrapper so importing this module does not require torch."""

    @staticmethod
    def build(*, latent_dim: int, hidden_dim: int, output_shape: tuple[int, int, int], residual_scale: float = 0.25):
        torch = _torch()

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = torch.nn.GRU(int(latent_dim), int(hidden_dim), batch_first=True)
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
                    torch.nn.ReLU(),
                    torch.nn.Linear(int(hidden_dim), int(np.prod(output_shape))),
                )
                self.output_shape = tuple(int(v) for v in output_shape)

            def forward(self, z_window, last_frame, *, return_residual: bool = False):
                _seq, hidden = self.gru(z_window)
                residual = float(residual_scale) * torch.tanh(self.head(hidden[-1]).reshape(last_frame.shape[0], *self.output_shape))
                pred = torch.clamp(last_frame + residual, 0.0, 1.0)
                if return_residual:
                    return pred, residual
                return pred

        return _Model()


def run_concept_tests(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    variants: Iterable[str] = ("residual_pixel_mse", "residual_pixel_weighted", "residual_pixel_residual_mse", "convgru_pixel_motion_weighted_huber", "unet_convgru_pixel_motion_weighted_huber", "convlstm_pixel_motion_weighted_huber", "temporal_cnn_pixel_motion_weighted_huber", "joint_latent_pixel_delta_weighted"),
    hidden_dim: int = 128,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    residual_scale: float = 0.25,
    conv_hidden_channels: int | None = None,
    conv_layers: int = 1,
    loss_mode: str = "auto",
    active_weight: float = 5.0,
    active_threshold: float | None = None,
    lambda_latent: float = 0.05,
    lambda_reconstruction: float = 0.10,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    windows, targets, video_ids = _load_windows(dataset)
    train_mask = _split_mask(video_ids, dataset.get("splits"), "train", default_all=True)
    if not np.any(train_mask):
        raise ValueError("Training split is empty.")
    threshold = _active_threshold(targets[train_mask], active_threshold)
    persistence = _persistence_metrics(windows, targets, video_ids, dataset.get("splits"))

    variants = tuple(str(v) for v in variants)
    results: dict[str, Any] = {}
    for variant in variants:
        variant_out = out / variant
        variant_out.mkdir(parents=True, exist_ok=True)
        variant_loss_mode = _variant_loss_mode(variant, default=loss_mode)
        if variant.startswith("residual_pixel"):
            metrics = _train_residual_pixel_variant(
                dataset=dataset,
                autoencoder_run=autoencoder_run,
                windows=windows,
                targets=targets,
                video_ids=video_ids,
                train_mask=train_mask,
                out_dir=variant_out,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                residual_scale=residual_scale,
                loss_mode=variant_loss_mode,
                active_weight=active_weight,
                active_threshold=threshold,
                seed=seed,
                device=device,
            )
        elif variant.startswith(("convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel", "unet_convgru_pixel")):
            metrics = _train_convgru_pixel_variant(
                dataset=dataset,
                windows=windows,
                targets=targets,
                video_ids=video_ids,
                train_mask=train_mask,
                out_dir=variant_out,
                variant=variant,
                hidden_channels=int(conv_hidden_channels or hidden_dim),
                num_layers=int(conv_layers),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                residual_scale=residual_scale,
                loss_mode=variant_loss_mode,
                active_weight=active_weight,
                active_threshold=threshold,
                seed=seed,
                device=device,
            )
        elif variant.startswith("joint_latent"):
            metrics = _train_joint_latent_variant(
                dataset=dataset,
                autoencoder_run=autoencoder_run,
                windows=windows,
                targets=targets,
                video_ids=video_ids,
                train_mask=train_mask,
                out_dir=variant_out,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weighted=("weighted" in variant),
                active_weight=active_weight,
                active_threshold=threshold,
                lambda_latent=lambda_latent,
                lambda_reconstruction=lambda_reconstruction,
                seed=seed,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported concept-test variant: {variant}")
        results[variant] = metrics

    summary = {
        "schema_version": 1,
        "run_id": out.name or "dynamics_concept_tests_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset.get("array_path")),
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "device": str(device),
        "seed": int(seed),
        "training_config": {
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "residual_scale": float(residual_scale),
            "conv_hidden_channels": int(conv_hidden_channels or hidden_dim),
            "conv_layers": int(conv_layers),
            "loss_mode": str(loss_mode),
            "active_weight": float(active_weight),
            "active_threshold": float(threshold),
            "lambda_latent": float(lambda_latent),
            "lambda_reconstruction": float(lambda_reconstruction),
        },
        "persistence": persistence,
        "variants": results,
    }
    (out / "concept_test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(out / "concept_test_summary.md", summary)
    return summary


def _train_residual_pixel_variant(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    windows: np.ndarray,
    targets: np.ndarray,
    video_ids: np.ndarray,
    train_mask: np.ndarray,
    out_dir: Path,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    residual_scale: float,
    loss_mode: str,
    active_weight: float,
    active_threshold: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    loss_mode = _normalize_pixel_loss_mode(loss_mode)
    ae, latent_mean, latent_std, latent_dim = _load_autoencoder(autoencoder_run, input_channels=int(windows.shape[2]), device=device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)

    xw = torch.from_numpy(windows).to(device)
    yt = torch.from_numpy(targets).to(device)
    z_windows = _encode_windows(ae, xw, latent_mean, latent_std, latent_dim, batch_size=batch_size)
    model = ResidualPixelGRU.build(latent_dim=latent_dim, hidden_dim=hidden_dim, output_shape=tuple(targets.shape[1:]), residual_scale=residual_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    train_indices = torch.nonzero(torch.as_tensor(train_mask, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            pred, residual = model(z_windows[idx], xw[idx, -1], return_residual=True)
            loss = _pixel_prediction_loss(
                pred,
                yt[idx],
                xw[idx, -1],
                loss_mode=loss_mode,
                residual=residual,
                active_weight=active_weight,
                threshold=active_threshold,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    model.eval()
    pred = _predict_residual_pixel(model, z_windows, xw, batch_size=batch_size)
    metrics = _prediction_metrics(
        pred=pred,
        targets=targets,
        windows=windows,
        video_ids=video_ids,
        splits=dataset.get("splits"),
        objective=f"residual_pixel_{loss_mode}",
        training_loss=losses,
        train_count=int(train_mask.sum()),
        active_threshold=active_threshold,
        active_weight=active_weight if _loss_uses_motion_weights(loss_mode) else 0.0,
    )
    metrics["model_kind"] = "latent_gru_pixel_residual"
    metrics["model_family"] = "residual_pixel_gru"
    metrics["loss_mode"] = loss_mode
    metrics["hidden_dim"] = int(hidden_dim)
    metrics["residual_scale"] = float(residual_scale)
    metrics["prediction_examples_path"] = str(
        _write_prediction_examples(
            out_dir / "prediction_examples.json",
            windows=windows,
            targets=targets,
            pred=pred,
            video_ids=video_ids,
            splits=dataset.get("splits"),
            windowing=dataset.get("windowing"),
        )
    )
    (out_dir / "concept_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_dim": int(hidden_dim),
            "residual_scale": float(residual_scale),
            "loss_mode": loss_mode,
            "objective": metrics["objective"],
        },
        out_dir / "concept_checkpoint.pt",
    )
    return metrics


def _train_convgru_pixel_variant(
    *,
    dataset: Mapping[str, Any],
    windows: np.ndarray,
    targets: np.ndarray,
    video_ids: np.ndarray,
    train_mask: np.ndarray,
    out_dir: Path,
    variant: str,
    hidden_channels: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    residual_scale: float,
    loss_mode: str,
    active_weight: float,
    active_threshold: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    loss_mode = _normalize_pixel_loss_mode(loss_mode)
    architecture, model_kind, model_family = _spatial_architecture_for_variant(variant)
    model = _build_spatial_pixel_model(
        architecture=architecture,
        input_channels=int(windows.shape[2]),
        window_frames=int(windows.shape[1]),
        hidden_channels=int(hidden_channels),
        num_layers=int(num_layers),
        residual_scale=float(residual_scale),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    xw = torch.from_numpy(windows).to(device)
    yt = torch.from_numpy(targets).to(device)
    train_indices = torch.nonzero(torch.as_tensor(train_mask, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        model.train()
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            pred, residual = model(xw[idx], return_residual=True)
            loss = _pixel_prediction_loss(
                pred,
                yt[idx],
                xw[idx, -1],
                loss_mode=loss_mode,
                residual=residual,
                active_weight=active_weight,
                threshold=active_threshold,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    model.eval()
    pred = _predict_convgru_pixel(model, xw, batch_size=batch_size)
    metrics = _prediction_metrics(
        pred=pred,
        targets=targets,
        windows=windows,
        video_ids=video_ids,
        splits=dataset.get("splits"),
        objective=f"{architecture}_{loss_mode}",
        training_loss=losses,
        train_count=int(train_mask.sum()),
        active_threshold=active_threshold,
        active_weight=active_weight if _loss_uses_motion_weights(loss_mode) else 0.0,
    )
    metrics["model_kind"] = model_kind
    metrics["model_family"] = model_family
    metrics["architecture"] = architecture
    metrics["variant"] = str(variant)
    metrics["loss_mode"] = loss_mode
    metrics["hidden_channels"] = int(hidden_channels)
    metrics["num_layers"] = int(num_layers)
    metrics["residual_scale"] = float(residual_scale)
    metrics["prediction_examples_path"] = str(
        _write_prediction_examples(
            out_dir / "prediction_examples.json",
            windows=windows,
            targets=targets,
            pred=pred,
            video_ids=video_ids,
            splits=dataset.get("splits"),
            windowing=dataset.get("windowing"),
        )
    )
    (out_dir / "concept_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "variant": str(variant),
            "model_kind": model_kind,
            "model_family": model_family,
            "input_channels": int(windows.shape[2]),
            "window_frames": int(windows.shape[1]),
            "hidden_channels": int(hidden_channels),
            "num_layers": int(num_layers),
            "residual_scale": float(residual_scale),
            "loss_mode": loss_mode,
            "objective": metrics["objective"],
        },
        out_dir / "concept_checkpoint.pt",
    )
    return metrics


def _train_joint_latent_variant(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    windows: np.ndarray,
    targets: np.ndarray,
    video_ids: np.ndarray,
    train_mask: np.ndarray,
    out_dir: Path,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weighted: bool,
    active_weight: float,
    active_threshold: float,
    lambda_latent: float,
    lambda_reconstruction: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    ae, latent_mean, latent_std, latent_dim = _load_autoencoder(autoencoder_run, input_channels=int(windows.shape[2]), device=device)
    model = LatentGRUPredictor(latent_dim=latent_dim, hidden_dim=int(hidden_dim)).to(device)
    opt = torch.optim.Adam(list(ae.parameters()) + list(model.parameters()), lr=float(learning_rate))
    xw = torch.from_numpy(windows).to(device)
    yt = torch.from_numpy(targets).to(device)
    train_indices = torch.nonzero(torch.as_tensor(train_mask, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    losses: list[float] = []
    pred_losses: list[float] = []
    for _epoch in range(int(epochs)):
        ae.train()
        model.train()
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        epoch_pred_losses = []
        for start in range(0, perm.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            batch = xw[idx]
            target = yt[idx]
            b, w, c, h, ww = batch.shape
            z_window_raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
            z_window = (z_window_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            target_z_raw = ae.encode(target)
            target_z = (target_z_raw - latent_mean.reshape(1, latent_dim)) / latent_std.reshape(1, latent_dim)
            pred_delta = model(z_window)
            pred_z = z_window[:, -1, :] + pred_delta
            pred_x = ae.decode(pred_z * latent_std + latent_mean)
            pred_loss = _weighted_mse(pred_x, target, batch[:, -1], weighted=weighted, active_weight=active_weight, threshold=active_threshold)
            latent_loss = torch.mean((pred_delta - (target_z - z_window[:, -1, :])) ** 2)
            recon_target, _ = ae(target)
            recon_loss = _weighted_mse(
                recon_target,
                target,
                batch[:, -1],
                weighted=weighted,
                active_weight=active_weight,
                threshold=active_threshold,
            )
            loss = pred_loss + float(lambda_latent) * latent_loss + float(lambda_reconstruction) * recon_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_pred_losses.append(float(pred_loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        pred_losses.append(float(np.mean(epoch_pred_losses)) if epoch_pred_losses else 0.0)

    ae.eval()
    model.eval()
    pred = _predict_joint_latent(ae, model, xw, latent_mean, latent_std, latent_dim, batch_size=batch_size)
    metrics = _prediction_metrics(
        pred=pred,
        targets=targets,
        windows=windows,
        video_ids=video_ids,
        splits=dataset.get("splits"),
        objective="joint_latent_delta_pixel_weighted_mse" if weighted else "joint_latent_delta_pixel_mse",
        training_loss=losses,
        train_count=int(train_mask.sum()),
        active_threshold=active_threshold,
        active_weight=active_weight if weighted else 0.0,
    )
    metrics["training_prediction_loss"] = pred_losses
    metrics["lambda_latent"] = float(lambda_latent)
    metrics["lambda_reconstruction"] = float(lambda_reconstruction)
    metrics["prediction_examples_path"] = str(
        _write_prediction_examples(
            out_dir / "prediction_examples.json",
            windows=windows,
            targets=targets,
            pred=pred,
            video_ids=video_ids,
            splits=dataset.get("splits"),
            windowing=dataset.get("windowing"),
        )
    )
    (out_dir / "concept_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "autoencoder_state": ae.state_dict(),
            "model_state": model.state_dict(),
            "hidden_dim": int(hidden_dim),
            "objective": metrics["objective"],
        },
        out_dir / "concept_checkpoint.pt",
    )
    return metrics



def backfill_spatial_prediction_examples(
    *,
    dataset: Mapping[str, Any],
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    out_path: str | Path | None = None,
    batch_size: int = 16,
    max_examples: int = 3,
    device: str = "cpu",
    update_metrics: bool = True,
    backfill_metrics: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write prediction examples and optionally full diagnostics for a spatial checkpoint."""
    torch = _torch()
    run = Path(run_dir)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else run / "concept_checkpoint.pt"
    metrics_file = Path(metrics_path) if metrics_path is not None else run / "concept_metrics.json"
    examples_path = Path(out_path) if out_path is not None else run / "prediction_examples.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing concept checkpoint: {checkpoint}")
    load_device = "cpu" if dry_run else device
    ckpt = torch.load(checkpoint, map_location=load_device)
    architecture = str(ckpt.get("architecture") or "").strip()
    if not architecture:
        variant = str(ckpt.get("variant") or ckpt.get("objective") or "")
        architecture, _model_kind, _model_family = _spatial_architecture_for_variant(variant)
    if architecture not in {"convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel", "unet_convgru_pixel"}:
        raise ValueError(f"Checkpoint architecture is not a spatial pixel concept model: {architecture}")
    array_summary = _prediction_dataset_array_summary(dataset)
    example_preview = _prediction_example_selection_preview(dataset, max_examples=int(max_examples)) if dry_run else []
    split_summary = _prediction_window_split_summary(dataset) if dry_run else {}
    if dry_run:
        would_write_files = [str(examples_path), str(examples_path.parent / "prediction_examples_backfill.json")]
        if update_metrics and metrics_file.exists():
            would_write_files.append(str(metrics_file))
        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
            "run_dir": str(run),
            "checkpoint_path": str(checkpoint),
            "metrics_path": str(metrics_file) if metrics_file.exists() else None,
            "prediction_examples_path": str(examples_path),
            "architecture": architecture,
            "dataset_array_path": str(dataset.get("array_path")),
            "dataset_arrays": array_summary.get("arrays", {}),
            "dataset_window_count": int(array_summary.get("dataset_window_count") or 0),
            "example_count": len(example_preview),
            "example_preview": example_preview,
            "source_indices": [int(item.get("index", 0)) for item in example_preview],
            "split_window_counts": split_summary.get("split_window_counts", {}),
            "split_video_counts": split_summary.get("split_video_counts", {}),
            "split_label_counts": split_summary.get("split_label_counts", {}),
            "split_top_videos": split_summary.get("split_top_videos", {}),
            "estimated_uncompressed_bytes": int(array_summary.get("estimated_uncompressed_bytes") or 0),
            "estimated_uncompressed_gib": _bytes_to_gib(array_summary.get("estimated_uncompressed_bytes") or 0),
            "estimated_compressed_bytes": int(array_summary.get("estimated_compressed_bytes") or 0),
            "estimated_compressed_gib": _bytes_to_gib(array_summary.get("estimated_compressed_bytes") or 0),
            "requested_batch_size": int(batch_size),
            "estimated_example_batches": _ceil_div(len(example_preview), int(batch_size)),
            "estimated_metric_batches": _ceil_div(int(array_summary.get("dataset_window_count") or 0), int(batch_size)) if update_metrics and backfill_metrics else 0,
            "metrics_updated": False,
            "prediction_metrics_backfilled": False,
            "would_update_metrics": bool(update_metrics and metrics_file.exists()),
            "would_backfill_metrics": bool(update_metrics and backfill_metrics),
            "would_write_files": would_write_files,
        }
    windows, targets, video_ids, source_indices, dataset_window_count = _load_prediction_example_subset(dataset, max_examples=int(max_examples))
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
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    _write_prediction_examples(
        examples_path,
        windows=windows,
        targets=targets,
        pred=pred,
        video_ids=video_ids,
        splits=dataset.get("splits"),
        windowing=dataset.get("windowing"),
        max_examples=int(max_examples),
        source_indices=source_indices,
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run),
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_file) if metrics_file.exists() else None,
        "prediction_examples_path": str(examples_path),
        "architecture": architecture,
        "example_count": int(targets.shape[0]),
        "dataset_window_count": int(dataset_window_count),
        "source_indices": [int(index) for index in source_indices],
    }
    metrics = _load_json(metrics_file) if metrics_file.exists() else None
    if update_metrics and metrics is not None:
        metrics["prediction_examples_path"] = str(examples_path)
        metrics["prediction_examples_backfilled_at"] = summary["created_at"]
        summary["metrics_updated"] = True
    else:
        summary["metrics_updated"] = False

    if update_metrics and backfill_metrics:
        if metrics is None:
            raise FileNotFoundError(f"Cannot backfill metrics because metrics file is missing: {metrics_file}")
        metric_windows, metric_targets, metric_video_ids = _load_prediction_metric_arrays(dataset)
        metric_xw = torch.from_numpy(metric_windows).to(device)
        metric_pred = _predict_convgru_pixel(model, metric_xw, batch_size=int(batch_size))
        train_mask = _split_mask(metric_video_ids, dataset.get("splits"), "train", default_all=True)
        active_threshold = metrics.get("active_threshold")
        if active_threshold is None:
            active_threshold = _active_threshold(metric_targets[train_mask], None)
        active_weight = float(metrics.get("active_weight") or 0.0)
        diagnostic_metrics = _prediction_metrics(
            pred=metric_pred,
            targets=metric_targets,
            windows=metric_windows,
            video_ids=metric_video_ids,
            splits=dataset.get("splits"),
            objective=str(metrics.get("objective") or ckpt.get("objective") or f"{architecture}_backfill"),
            training_loss=list(metrics.get("training_loss", [])) if isinstance(metrics.get("training_loss"), list) else [],
            train_count=int(train_mask.sum()),
            active_threshold=float(active_threshold),
            active_weight=float(active_weight),
        )
        metrics.update(diagnostic_metrics)
        metrics["prediction_examples_path"] = str(examples_path)
        metrics["prediction_metrics_backfilled_at"] = summary["created_at"]
        split_metrics = metrics.get("split_metrics", {})
        summary["prediction_metrics_backfilled"] = True
        summary["metrics_evaluation_window_count"] = int(metric_targets.shape[0])
        summary["metrics_split_window_counts"] = {
            str(split): int(payload.get("window_count", 0))
            for split, payload in split_metrics.items()
            if isinstance(payload, Mapping)
        }
        summary["metrics_per_video_counts"] = {
            str(split): len(payload.get("per_video", {}))
            for split, payload in split_metrics.items()
            if isinstance(payload, Mapping)
        }
    else:
        summary["prediction_metrics_backfilled"] = False

    if update_metrics and metrics is not None:
        metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (examples_path.parent / "prediction_examples_backfill.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary





def _ceil_div(value: int, divisor: int) -> int:
    if int(value) <= 0:
        return 0
    return int((int(value) + max(int(divisor), 1) - 1) // max(int(divisor), 1))


def _bytes_to_gib(value: Any) -> float:
    return float(round(float(value) / float(1024 ** 3), 3))


def _prediction_dataset_array_summary(dataset: Mapping[str, Any]) -> dict[str, Any]:
    array_path = Path(str(dataset["array_path"]))
    arrays: dict[str, dict[str, Any]] = {}
    total_uncompressed = 0
    total_compressed = 0
    with zipfile.ZipFile(array_path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            key = Path(info.filename).stem
            with archive.open(info) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    shape, _fortran_order, dtype = np.lib.format._read_array_header(handle, version)  # noqa: SLF001
            uncompressed_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize) if shape else int(np.dtype(dtype).itemsize)
            arrays[key] = {
                "shape": [int(dim) for dim in shape],
                "dtype": str(np.dtype(dtype)),
                "uncompressed_bytes": uncompressed_bytes,
                "compressed_bytes": int(info.compress_size),
            }
            total_uncompressed += uncompressed_bytes
            total_compressed += int(info.compress_size)
    return {
        "array_path": str(array_path),
        "arrays": arrays,
        "dataset_window_count": int((arrays.get("targets") or {}).get("shape", [0])[0]),
        "estimated_uncompressed_bytes": int(total_uncompressed),
        "estimated_compressed_bytes": int(total_compressed),
    }



def _prediction_window_split_summary(dataset: Mapping[str, Any], *, top_video_limit: int = 3) -> dict[str, Any]:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        video_ids = arrays["window_video_ids"].astype(str)
        labels = arrays["window_labels"].astype(str) if "window_labels" in arrays.files else np.full(video_ids.shape, "unknown", dtype=str)
    split_window_counts: dict[str, int] = {}
    split_video_windows: dict[str, dict[str, int]] = {}
    split_label_counts: dict[str, dict[str, int]] = {}
    for video_id, label in zip(video_ids, labels):
        split = _split_name_for_video_id(str(video_id), dataset.get("splits")) or "unknown"
        video_key = str(video_id)
        label_key = str(label) if str(label) else "unknown"
        split_window_counts[split] = split_window_counts.get(split, 0) + 1
        per_video = split_video_windows.setdefault(split, {})
        per_video[video_key] = per_video.get(video_key, 0) + 1
        per_label = split_label_counts.setdefault(split, {})
        per_label[label_key] = per_label.get(label_key, 0) + 1
    split_top_videos = {
        split: [
            {"video_id": video_id, "window_count": int(count)}
            for video_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(top_video_limit)]
        ]
        for split, counts in sorted(split_video_windows.items())
    }
    return {
        "split_window_counts": {key: int(split_window_counts[key]) for key in sorted(split_window_counts)},
        "split_video_counts": {key: int(len(split_video_windows[key])) for key in sorted(split_video_windows)},
        "split_label_counts": {
            split: {label: int(count) for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))}
            for split, counts in sorted(split_label_counts.items())
        },
        "split_top_videos": split_top_videos,
    }

def _prediction_example_selection_preview(dataset: Mapping[str, Any], *, max_examples: int) -> list[dict[str, Any]]:
    count = max(int(max_examples), 0)
    if count <= 0:
        return []
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        video_ids = arrays["window_video_ids"].astype(str)
    selected = _prediction_example_indices(video_ids, dataset.get("splits"), max_examples=count)
    return [
        {
            "index": int(index),
            "video_id": str(video_ids[int(index)]),
            "split": _split_name_for_video_id(str(video_ids[int(index)]), dataset.get("splits")),
        }
        for index in selected
    ]

def _load_prediction_metric_arrays(dataset: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_array(arrays["windows"][:])
        targets = _prepare_array(arrays["targets"][:])
        video_ids = arrays["window_video_ids"].astype(str)
    return windows, targets, video_ids


def _load_prediction_example_subset(dataset: Mapping[str, Any], *, max_examples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    count = max(int(max_examples), 0)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        dataset_window_count = int(arrays["targets"].shape[0])
        if count <= 0 or dataset_window_count <= 0:
            empty_windows = _prepare_array(arrays["windows"][:0])
            empty_targets = _prepare_array(arrays["targets"][:0])
            return empty_windows, empty_targets, np.asarray([], dtype=str), np.asarray([], dtype=np.int64), dataset_window_count
        all_video_ids = arrays["window_video_ids"].astype(str)
        selected = _prediction_example_indices(all_video_ids, dataset.get("splits"), max_examples=count)
        windows = _prepare_array(arrays["windows"][selected])
        targets = _prepare_array(arrays["targets"][selected])
        video_ids = all_video_ids[selected].astype(str)
    return windows, targets, video_ids, selected.astype(np.int64), dataset_window_count


def _prediction_example_indices(video_ids: np.ndarray, splits: Mapping[str, Any] | None, *, max_examples: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    priority = ("test", "val", "train", None)
    for split in priority:
        for index, video_id in enumerate(video_ids.astype(str)):
            if index in seen:
                continue
            if split is not None and _split_name_for_video_id(str(video_id), splits) != split:
                continue
            selected.append(index)
            seen.add(index)
            if len(selected) >= int(max_examples):
                return np.asarray(selected, dtype=np.int64)
    return np.asarray(selected, dtype=np.int64)


def _load_windows(dataset: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_array(arrays["windows"])
        targets = _prepare_array(arrays["targets"])
        video_ids = arrays["window_video_ids"].astype(str)
    return windows, targets, video_ids


def _prepare_array(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _load_autoencoder(autoencoder_run: Mapping[str, Any], *, input_channels: int, device: str):
    torch = _torch()
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    base_channels = int(ckpt.get("base_channels", 16))
    ae = GridAutoencoder(input_channels=int(input_channels), latent_dim=latent_dim, base_channels=base_channels, input_shape=tuple(ckpt.get("input_shape") or (int(input_channels), 32, 32))).to(device)
    ae.load_state_dict(ckpt["model_state"])
    latent_mean = _checkpoint_tensor(ckpt.get("latent_mean"), latent_dim, fill=0.0, device=device)
    latent_std = _checkpoint_tensor(ckpt.get("latent_std"), latent_dim, fill=1.0, device=device)
    latent_std = torch.clamp(latent_std, min=1e-6)
    return ae, latent_mean.reshape(1, latent_dim), latent_std.reshape(1, latent_dim), latent_dim


def _checkpoint_tensor(value: Any, latent_dim: int, *, fill: float, device: str):
    torch = _torch()
    if value is None:
        return torch.full((latent_dim,), float(fill), dtype=torch.float32, device=device)
    if hasattr(value, "detach"):
        return value.detach().to(device=device, dtype=torch.float32).reshape(-1)
    return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device).reshape(-1)


def _encode_windows(ae, xw, latent_mean, latent_std, latent_dim: int, *, batch_size: int):
    torch = _torch()
    chunks = []
    with torch.no_grad():
        n, w, c, h, ww = xw.shape
        for start in range(0, n, int(batch_size)):
            batch = xw[start : start + int(batch_size)]
            bb = batch.shape[0]
            z_raw = ae.encode(batch.reshape(bb * w, c, h, ww)).reshape(bb, w, latent_dim)
            chunks.append(((z_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)).detach())
    return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, int(xw.shape[1]), latent_dim), device=xw.device)


def _predict_residual_pixel(model, z_windows, xw, *, batch_size: int) -> np.ndarray:
    torch = _torch()
    chunks = []
    with torch.no_grad():
        for start in range(0, z_windows.shape[0], int(batch_size)):
            pred = model(z_windows[start : start + int(batch_size)], xw[start : start + int(batch_size), -1])
            chunks.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, *xw.shape[2:]), dtype=np.float32)


def _predict_convgru_pixel(model, xw, *, batch_size: int) -> np.ndarray:
    torch = _torch()
    chunks = []
    with torch.no_grad():
        for start in range(0, xw.shape[0], int(batch_size)):
            pred = model(xw[start : start + int(batch_size)])
            chunks.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, *xw.shape[2:]), dtype=np.float32)


def _spatial_architecture_for_variant(variant: str) -> tuple[str, str, str]:
    name = str(variant).strip().lower()
    if name.startswith("unet_convgru_pixel"):
        return "unet_convgru_pixel", "unet_convgru_pixel_residual", "unet_convgru_residual"
    if name.startswith("convlstm_pixel"):
        return "convlstm_pixel", "pixel_convlstm_residual", "convlstm_residual"
    if name.startswith("temporal_cnn_pixel"):
        return "temporal_cnn_pixel", "temporal_cnn_pixel_residual", "temporal_cnn_residual"
    if name.startswith("convgru_pixel"):
        return "convgru_pixel", "pixel_convgru_residual", "pixel_convgru"
    raise ValueError(f"Unsupported spatial pixel variant: {variant}")


def _build_spatial_pixel_model(
    *,
    architecture: str,
    input_channels: int,
    window_frames: int,
    hidden_channels: int,
    num_layers: int,
    residual_scale: float,
):
    if architecture == "convgru_pixel":
        return PixelConvGRUResidual(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            residual_scale=residual_scale,
        )
    if architecture == "convlstm_pixel":
        return PixelConvLSTMResidual(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            residual_scale=residual_scale,
        )
    if architecture == "temporal_cnn_pixel":
        return TemporalCNNResidual(
            input_channels=input_channels,
            window_frames=window_frames,
            hidden_channels=hidden_channels,
            residual_scale=residual_scale,
            num_blocks=max(1, int(num_layers)),
        )
    if architecture == "unet_convgru_pixel":
        return UNetConvGRUResidual(
            input_channels=input_channels,
            base_channels=max(8, hidden_channels // 2),
            hidden_channels=hidden_channels,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unsupported spatial pixel architecture: {architecture}")


def _predict_joint_latent(ae, model, xw, latent_mean, latent_std, latent_dim: int, *, batch_size: int) -> np.ndarray:
    torch = _torch()
    chunks = []
    with torch.no_grad():
        n, w, c, h, ww = xw.shape
        for start in range(0, n, int(batch_size)):
            batch = xw[start : start + int(batch_size)]
            bb = batch.shape[0]
            z_raw = ae.encode(batch.reshape(bb * w, c, h, ww)).reshape(bb, w, latent_dim)
            z = (z_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_delta = model(z)
            pred_z = z[:, -1, :] + pred_delta
            pred = ae.decode(pred_z * latent_std + latent_mean)
            chunks.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, *xw.shape[2:]), dtype=np.float32)


def _variant_loss_mode(variant: str, *, default: str = "auto") -> str:
    requested = str(default or "auto").strip().lower()
    if requested != "auto":
        return _normalize_pixel_loss_mode(requested)
    name = str(variant).strip().lower()
    if "residual_mse" in name or "delta_mse" in name:
        return "residual_mse"
    if "huber" in name:
        return "motion_weighted_huber"
    if "motion_weighted" in name or "weighted" in name:
        return "motion_weighted_mse"
    return "frame_mse"


def _normalize_pixel_loss_mode(value: str) -> str:
    name = str(value or "frame_mse").strip().lower()
    aliases = {
        "mse": "frame_mse",
        "pixel_mse": "frame_mse",
        "weighted": "motion_weighted_mse",
        "weighted_mse": "motion_weighted_mse",
        "delta_mse": "residual_mse",
        "residual_delta_mse": "residual_mse",
        "weighted_huber": "motion_weighted_huber",
    }
    name = aliases.get(name, name)
    allowed = {"frame_mse", "residual_mse", "motion_weighted_mse", "motion_weighted_huber"}
    if name not in allowed:
        raise ValueError(f"Unsupported pixel loss mode: {value}")
    return name


def _loss_uses_motion_weights(loss_mode: str) -> bool:
    return _normalize_pixel_loss_mode(loss_mode) in {"motion_weighted_mse", "motion_weighted_huber"}


def _pixel_prediction_loss(pred, target, last_frame, *, loss_mode: str, residual=None, active_weight: float, threshold: float):
    torch = _torch()
    mode = _normalize_pixel_loss_mode(loss_mode)
    if mode == "residual_mse":
        pred_residual = residual if residual is not None else pred - last_frame
        target_residual = target - last_frame
        return torch.mean((pred_residual - target_residual) ** 2)
    diff = pred - target
    if mode == "frame_mse":
        return torch.mean(diff * diff)
    active = (target > float(threshold)) | (torch.abs(target - last_frame) > max(float(threshold) * 0.5, 1e-4))
    weights = 1.0 + float(active_weight) * active.to(dtype=diff.dtype)
    if mode == "motion_weighted_mse":
        pixel_loss = diff * diff
    else:
        beta = 0.02
        abs_diff = torch.abs(diff)
        pixel_loss = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    return torch.sum(pixel_loss * weights) / torch.clamp(torch.sum(weights), min=1.0)


def _weighted_mse(pred, target, last_frame, *, weighted: bool, active_weight: float, threshold: float):
    return _pixel_prediction_loss(
        pred,
        target,
        last_frame,
        loss_mode="motion_weighted_mse" if weighted else "frame_mse",
        residual=None,
        active_weight=active_weight,
        threshold=threshold,
    )


def _write_prediction_examples(
    path: Path,
    *,
    windows: np.ndarray,
    targets: np.ndarray,
    pred: np.ndarray,
    video_ids: np.ndarray,
    splits: Mapping[str, Any] | None,
    windowing: Mapping[str, Any] | None,
    max_examples: int = 3,
    source_indices: np.ndarray | None = None,
) -> Path:
    examples = _prediction_examples(
        windows=windows,
        targets=targets,
        pred=pred,
        video_ids=video_ids,
        splits=splits,
        windowing=windowing,
        max_examples=max_examples,
        source_indices=source_indices,
    )
    path.write_text(json.dumps({"schema_version": 1, "examples": examples}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prediction_examples(
    *,
    windows: np.ndarray,
    targets: np.ndarray,
    pred: np.ndarray,
    video_ids: np.ndarray | None,
    splits: Mapping[str, Any] | None,
    windowing: Mapping[str, Any] | None,
    max_examples: int,
    source_indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    items = []
    ids = np.asarray(video_ids).astype(str) if video_ids is not None else None
    for i in range(min(max(int(max_examples), 0), int(targets.shape[0]))):
        source_index = int(source_indices[i]) if source_indices is not None and i < len(source_indices) else int(i)
        item = {
            "index": source_index,
            "input_last": _json_frame(windows[i, -1, 0]),
            "target_next": _json_frame(targets[i, 0]),
            "predicted_next": _json_frame(pred[i, 0]),
            "abs_error_mean": float(np.mean(np.abs(targets[i] - pred[i]))),
        }
        if ids is not None and i < ids.shape[0]:
            item["video_id"] = str(ids[i])
            item["split"] = _split_name_for_video_id(str(ids[i]), splits)
        if isinstance(windowing, Mapping):
            for key in ("window_frames", "temporal_stride_frames", "prediction_horizon_frames", "prediction_horizon_sec", "effective_frame_rate_hz", "source_frame_rate_hz"):
                if key in windowing:
                    item[key] = windowing.get(key)
        items.append(item)
    return items


def _json_frame(frame: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(frame, dtype=np.float64), 5).tolist()


def _split_name_for_video_id(video_id: str, splits: Mapping[str, Any] | None) -> str | None:
    if not isinstance(splits, Mapping):
        return None
    assignments = splits.get("assignments")
    if isinstance(assignments, Mapping) and video_id in assignments:
        return str(assignments[video_id])
    for split_name in ("train", "val", "test"):
        for key in (split_name, f"{split_name}_video_ids", f"{split_name}_videos"):
            values = splits.get(key)
            if isinstance(values, Mapping):
                nested = values.get("video_ids") or values.get("videos") or values.get("ids")
                if nested is not None and video_id in {str(item) for item in nested}:
                    return split_name
            elif isinstance(values, (list, tuple, set)) and video_id in {str(item) for item in values}:
                return split_name
    return None


def _active_threshold(train_targets: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    nonzero = train_targets[train_targets > 0]
    if nonzero.size == 0:
        return 0.02
    return float(max(0.02, np.percentile(nonzero, 90)))


def _prediction_metrics(
    *,
    pred: np.ndarray,
    targets: np.ndarray,
    windows: np.ndarray,
    video_ids: np.ndarray,
    splits: Mapping[str, Any] | None,
    objective: str,
    training_loss: list[float],
    train_count: int,
    active_threshold: float,
    active_weight: float,
) -> dict[str, Any]:
    diff = pred - targets
    persistence_diff = windows[:, -1] - targets
    split_metrics = _split_prediction_metrics(diff, persistence_diff, video_ids, splits)
    structured_error_metrics = structured_prediction_error_metrics(
        pred_diff=diff,
        persistence_diff=persistence_diff,
        targets=targets,
        last_frames=windows[:, -1],
        video_ids=video_ids,
        splits=splits,
    )
    metrics = {
        "schema_version": 1,
        "objective": objective,
        "training_loss": training_loss,
        "training_window_count": int(train_count),
        "evaluation_window_count": int(targets.shape[0]),
        "active_threshold": float(active_threshold),
        "active_weight": float(active_weight),
        "decoded_prediction_mse": float(np.mean(diff * diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(diff))),
        "persistence_mse": float(np.mean(persistence_diff * persistence_diff)),
        "persistence_mae": float(np.mean(np.abs(persistence_diff))),
        "split_metrics": split_metrics,
        "structured_error_metrics": structured_error_metrics,
    }
    promote_structured_error_metrics(metrics, structured_error_metrics)
    metrics["improvement_over_persistence_mse"] = float(metrics["persistence_mse"] - metrics["decoded_prediction_mse"])
    for split_name, split in split_metrics.items():
        prefix = f"{split_name}_"
        for key, value in split.items():
            metrics[prefix + key] = value
        if split["decoded_prediction_mse"] is not None and split["persistence_mse"] is not None:
            metrics[prefix + "improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    return metrics


def _persistence_metrics(windows: np.ndarray, targets: np.ndarray, video_ids: np.ndarray, splits: Mapping[str, Any] | None) -> dict[str, Any]:
    diff = windows[:, -1] - targets
    return {
        "decoded_prediction_mse": float(np.mean(diff * diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(diff))),
        "split_metrics": _split_prediction_metrics(diff, diff, video_ids, splits, persistence_only=True),
    }


def _split_prediction_metrics(
    diff: np.ndarray,
    persistence_diff: np.ndarray,
    video_ids: np.ndarray,
    splits: Mapping[str, Any] | None,
    *,
    persistence_only: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ids = np.asarray(video_ids).astype(str)
    pred_diff = persistence_diff if persistence_only else diff
    for split_name in ("train", "val", "test", "all"):
        if split_name == "all":
            mask = np.ones(ids.shape[0], dtype=bool)
        else:
            mask = _split_mask(ids, splits, split_name, default_all=False)
        item = _concept_prediction_metric_payload(pred_diff[mask], persistence_diff[mask])
        item["per_video"] = _concept_prediction_per_video_metrics(
            pred_diff=pred_diff,
            persistence_diff=persistence_diff,
            video_ids=ids,
            mask=mask,
        )
        out[split_name] = item
    return out


def _concept_prediction_metric_payload(pred_diff: np.ndarray, persistence_diff: np.ndarray) -> dict[str, Any]:
    count = int(pred_diff.shape[0]) if pred_diff.ndim else 0
    if count == 0:
        return {
            "decoded_prediction_mse": None,
            "decoded_prediction_mae": None,
            "persistence_mse": None,
            "persistence_mae": None,
            "window_count": 0,
            "improvement_over_persistence_mse": None,
        }
    decoded_mse = float(np.mean(pred_diff * pred_diff))
    persistence_mse = float(np.mean(persistence_diff * persistence_diff))
    return {
        "decoded_prediction_mse": decoded_mse,
        "decoded_prediction_mae": float(np.mean(np.abs(pred_diff))),
        "persistence_mse": persistence_mse,
        "persistence_mae": float(np.mean(np.abs(persistence_diff))),
        "window_count": count,
        "improvement_over_persistence_mse": float(persistence_mse - decoded_mse),
    }


def _concept_prediction_per_video_metrics(
    *,
    pred_diff: np.ndarray,
    persistence_diff: np.ndarray,
    video_ids: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, Any]]:
    selected_ids = np.asarray(video_ids)[mask].astype(str)
    if selected_ids.size == 0:
        return {}
    out: dict[str, dict[str, Any]] = {}
    ids = np.asarray(video_ids).astype(str)
    for video_id in sorted(set(selected_ids.tolist())):
        video_mask = mask & (ids == str(video_id))
        out[str(video_id)] = _concept_prediction_metric_payload(pred_diff[video_mask], persistence_diff[video_mask])
    return out


def _split_mask(video_ids: np.ndarray, splits: Mapping[str, Any] | None, split_name: str, *, default_all: bool) -> np.ndarray:
    if not splits:
        return np.ones(video_ids.shape[0], dtype=bool) if default_all else np.zeros(video_ids.shape[0], dtype=bool)
    ids = _split_video_ids(splits, split_name)
    if ids is None:
        assignments = splits.get("assignments") if isinstance(splits, Mapping) else None
        if isinstance(assignments, Mapping):
            return np.asarray([str(assignments.get(str(v), "")) == split_name for v in video_ids], dtype=bool)
        return np.ones(video_ids.shape[0], dtype=bool) if default_all else np.zeros(video_ids.shape[0], dtype=bool)
    return np.isin(video_ids.astype(str), np.asarray(sorted(ids), dtype=str))


def _split_video_ids(splits: Mapping[str, Any], split_name: str) -> set[str] | None:
    candidates = [split_name, f"{split_name}_video_ids", f"{split_name}_videos"]
    for key in candidates:
        if key not in splits:
            continue
        value = splits[key]
        if isinstance(value, Mapping):
            for nested in ("video_ids", "videos", "ids"):
                if nested in value:
                    return {str(v) for v in value[nested]}
        if isinstance(value, (list, tuple, set)):
            return {str(v) for v in value}
    for nested in ("video_ids_by_split", "videos_by_split", "split_video_ids"):
        value = splits.get(nested)
        if isinstance(value, Mapping) and split_name in value:
            return {str(v) for v in value[split_name]}
    return None


def _write_markdown_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Dynamics Concept Test Summary",
        "",
        f"Experiment root: `{path.parent}`",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{summary['source_dataset']}`",
        f"- Autoencoder: `{summary['source_autoencoder_run']}`",
        f"- Device: `{summary['device']}`",
        f"- Active threshold: `{summary['training_config']['active_threshold']}`",
        "",
        "## Results",
        "",
        "| Variant | Val pred MSE | Val persistence MSE | Val improvement | Test pred MSE | Test persistence MSE | Test improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(summary["variants"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    _fmt(metrics.get("val_decoded_prediction_mse")),
                    _fmt(metrics.get("val_persistence_mse")),
                    _fmt(metrics.get("val_improvement_over_persistence_mse")),
                    _fmt(metrics.get("test_decoded_prediction_mse")),
                    _fmt(metrics.get("test_persistence_mse")),
                    _fmt(metrics.get("test_improvement_over_persistence_mse")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Positive improvement means the model beat persistence. Negative improvement means persistence was better.",
            "",
            "## References",
            "",
            "- Residual learning: https://arxiv.org/abs/1512.03385",
            "- Spatiotemporal sequence prediction with ConvLSTM: https://arxiv.org/abs/1506.04214",
            "- Predictive recurrent models: https://arxiv.org/abs/2103.09504",
            "- Foreground/background imbalance and focal loss: https://arxiv.org/abs/1708.02002",
            "- Biomedical image weighting/segmentation context: https://arxiv.org/abs/1505.04597",
            "- Latent-space predictive representation learning: https://arxiv.org/abs/1807.03748",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.9g}"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_variants(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run conservative grid dynamics concept tests.")
    parser.add_argument("--dataset", required=True, help="Path to dynamics_dataset.json")
    parser.add_argument("--autoencoder-run", required=True, help="Path to autoencoder_run.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--variants",
        default="residual_pixel_mse,residual_pixel_weighted,residual_pixel_residual_mse,convgru_pixel_motion_weighted_huber,unet_convgru_pixel_motion_weighted_huber,convlstm_pixel_motion_weighted_huber,temporal_cnn_pixel_motion_weighted_huber,joint_latent_pixel_delta_weighted",
        help="Comma-separated variants to run",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--conv-hidden-channels", type=int, default=None)
    parser.add_argument("--conv-layers", type=int, default=1)
    parser.add_argument("--loss-mode", default="auto", choices=["auto", "frame_mse", "residual_mse", "motion_weighted_mse", "motion_weighted_huber"])
    parser.add_argument("--active-weight", type=float, default=5.0)
    parser.add_argument("--active-threshold", type=float, default=None)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lambda-reconstruction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_concept_tests(
        dataset=_load_json(args.dataset),
        autoencoder_run=_load_json(args.autoencoder_run),
        out_dir=args.out_dir,
        variants=_parse_variants(args.variants),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        residual_scale=args.residual_scale,
        conv_hidden_channels=args.conv_hidden_channels,
        conv_layers=args.conv_layers,
        loss_mode=args.loss_mode,
        active_weight=args.active_weight,
        active_threshold=args.active_threshold,
        lambda_latent=args.lambda_latent,
        lambda_reconstruction=args.lambda_reconstruction,
        seed=args.seed,
        device=args.device,
    )
    print(f"Concept test summary: {Path(args.out_dir) / 'concept_test_summary.json'}")
    print(f"Concept test report: {Path(args.out_dir) / 'concept_test_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

