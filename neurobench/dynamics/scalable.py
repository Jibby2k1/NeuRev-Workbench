"""Scalable temporal-CNN training and architecture specs for grid dynamics."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.dynamics.concept_tests import (
    _active_threshold,
    _load_windows,
    _loss_uses_motion_weights,
    _normalize_pixel_loss_mode,
    _pixel_prediction_loss,
    _prediction_metrics,
    _split_mask,
    _torch,
)
from neurobench.dynamics.models import ScalableTemporalCNNResidual


def architecture_catalog() -> list[dict[str, Any]]:
    """Return the fixed Stage-A architecture catalogue for high-resolution sweeps."""
    base = {"kernel_size": 3, "normalization": "group", "activation": "silu", "dropout": 0.0, "dilations": [1, 2, 4]}
    return [
        {**base, "architecture_id": "stack_tiny_24x3", "topology": "stack", "stack_channels": [24, 24, 24], "stack_blocks": [1, 1, 1], "skip_connections": False},
        {**base, "architecture_id": "stack_small_32x4", "topology": "stack", "stack_channels": [32, 32, 32, 32], "stack_blocks": [1, 1, 1, 1], "dilations": [1, 2, 4, 8], "skip_connections": False},
        {**base, "architecture_id": "stack_deep_32x6", "topology": "stack", "stack_channels": [32, 32, 32], "stack_blocks": [2, 2, 2], "dilations": [1, 2, 4, 8], "skip_connections": False},
        {**base, "architecture_id": "stack_wide_48x4", "topology": "stack", "stack_channels": [48, 48, 48, 48], "stack_blocks": [1, 1, 1, 1], "dilations": [1, 2, 4, 8], "skip_connections": False},
        {**base, "architecture_id": "stack_wide_64x4", "topology": "stack", "stack_channels": [64, 64, 64, 64], "stack_blocks": [1, 1, 1, 1], "dilations": [1, 2, 4, 8], "skip_connections": False},
        {**base, "architecture_id": "ed_tiny_16_32_64", "topology": "encoder_decoder", "encoder_channels": [16, 32, 64], "encoder_blocks": [1, 1, 1], "decoder_channels": [32, 16], "decoder_blocks": [1, 1], "bottleneck_channels": 64, "bottleneck_blocks": 1, "skip_connections": True},
        {**base, "architecture_id": "ed_small_24_48_96", "topology": "encoder_decoder", "encoder_channels": [24, 48, 96], "encoder_blocks": [1, 1, 1], "decoder_channels": [48, 24], "decoder_blocks": [1, 1], "bottleneck_channels": 96, "bottleneck_blocks": 1, "skip_connections": True},
        {**base, "architecture_id": "ed_small_deep_24_48_96", "topology": "encoder_decoder", "encoder_channels": [24, 48, 96], "encoder_blocks": [1, 2, 2], "decoder_channels": [48, 24], "decoder_blocks": [2, 1], "bottleneck_channels": 96, "bottleneck_blocks": 2, "skip_connections": True},
        {**base, "architecture_id": "ed_medium_32_64_128", "topology": "encoder_decoder", "encoder_channels": [32, 64, 128], "encoder_blocks": [1, 1, 1], "decoder_channels": [64, 32], "decoder_blocks": [1, 1], "bottleneck_channels": 128, "bottleneck_blocks": 1, "skip_connections": True},
        {**base, "architecture_id": "ed_medium_deep_32_64_128", "topology": "encoder_decoder", "encoder_channels": [32, 64, 128], "encoder_blocks": [2, 2, 2], "decoder_channels": [64, 32], "decoder_blocks": [2, 1], "bottleneck_channels": 128, "bottleneck_blocks": 2, "skip_connections": True},
        {**base, "architecture_id": "ed_medium_no_skip_32_64_128", "topology": "encoder_decoder", "encoder_channels": [32, 64, 128], "encoder_blocks": [1, 1, 1], "decoder_channels": [64, 32], "decoder_blocks": [1, 1], "bottleneck_channels": 128, "bottleneck_blocks": 1, "skip_connections": False},
        {**base, "architecture_id": "ed_wide_48_96_192", "topology": "encoder_decoder", "encoder_channels": [48, 96, 192], "encoder_blocks": [1, 1, 1], "decoder_channels": [96, 48], "decoder_blocks": [1, 1], "bottleneck_channels": 192, "bottleneck_blocks": 1, "skip_connections": True},
        {**base, "architecture_id": "ed_wide_deep_48_96_192", "topology": "encoder_decoder", "encoder_channels": [48, 96, 192], "encoder_blocks": [2, 2, 2], "decoder_channels": [96, 48], "decoder_blocks": [2, 1], "bottleneck_channels": 192, "bottleneck_blocks": 2, "skip_connections": True},
        {**base, "architecture_id": "ed_fourstage_16_32_64_128", "topology": "encoder_decoder", "encoder_channels": [16, 32, 64, 128], "encoder_blocks": [1, 1, 1, 1], "decoder_channels": [64, 32, 16], "decoder_blocks": [1, 1, 1], "bottleneck_channels": 128, "bottleneck_blocks": 1, "skip_connections": True},
        {**base, "architecture_id": "ed_fourstage_medium_24_48_96_192", "topology": "encoder_decoder", "encoder_channels": [24, 48, 96, 192], "encoder_blocks": [1, 1, 1, 1], "decoder_channels": [96, 48, 24], "decoder_blocks": [1, 1, 1], "bottleneck_channels": 192, "bottleneck_blocks": 1, "skip_connections": True},
    ]


def canonical_architecture_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    model = ScalableTemporalCNNResidual(input_channels=1, window_frames=8, architecture_spec=spec, residual_scale=0.1)
    return dict(model.architecture_spec)


def architecture_summary(
    spec: Mapping[str, Any],
    *,
    input_channels: int = 1,
    window_frames: int = 8,
    grid_size: int | None = None,
) -> dict[str, Any]:
    model = ScalableTemporalCNNResidual(input_channels=input_channels, window_frames=window_frames, architecture_spec=spec, residual_scale=0.1)
    canonical = dict(model.architecture_spec)
    parameter_count = sum(int(p.numel()) for p in model.parameters())
    conv_layer_count = sum(1 for module in model.modules() if module.__class__.__name__ == "Conv2d")
    encoder_depth = int(sum(canonical.get("encoder_blocks") or []))
    decoder_depth = int(sum(canonical.get("decoder_blocks") or []))
    stack_depth = int(sum(canonical.get("stack_blocks") or []))
    channels = []
    for key in ("stack_channels", "encoder_channels", "decoder_channels"):
        channels.extend(int(v) for v in canonical.get(key, []) or [])
    channels.append(int(canonical.get("bottleneck_channels") or 0))
    out = {
        "architecture_id": canonical["architecture_id"],
        "topology": canonical["topology"],
        "parameter_count": int(parameter_count),
        "conv_layer_count": int(conv_layer_count),
        "stack_depth": int(stack_depth),
        "encoder_depth": int(encoder_depth),
        "decoder_depth": int(decoder_depth),
        "bottleneck_layers": int(canonical.get("bottleneck_blocks", 0)),
        "total_configured_blocks": int(stack_depth + encoder_depth + decoder_depth + int(canonical.get("bottleneck_blocks", 0))),
        "max_channels": int(max(channels) if channels else 0),
        "skip_connections": bool(canonical.get("skip_connections")),
        "normalization": canonical.get("normalization"),
        "activation": canonical.get("activation"),
        "dropout": float(canonical.get("dropout", 0.0)),
        "dilations": list(canonical.get("dilations") or []),
        "input_channels": int(input_channels),
        "window_frames": int(window_frames),
    }
    if grid_size is not None:
        out["grid_size"] = int(grid_size)
    return out


def predict_scalable_temporal_cnn(model, windows: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(windows.shape[0]), int(batch_size)):
            batch_np = windows[start : start + int(batch_size)].astype(np.float32, copy=False)
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            pred = model(batch)
            chunks.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, *windows.shape[2:]), dtype=np.float32)


def train_scalable_temporal_cnn(
    *,
    dataset: Mapping[str, Any],
    out_dir: str | Path,
    architecture_spec: Mapping[str, Any],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    residual_scale: float = 0.1,
    loss_mode: str = "residual_mse",
    active_weight: float = 5.0,
    active_threshold: float | None = None,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if hasattr(torch, "set_num_threads") and device == "cpu":
        torch.set_num_threads(1)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    windows, targets, video_ids = _load_windows(dataset)
    train_mask = _split_mask(video_ids, dataset.get("splits"), "train", default_all=True)
    if not np.any(train_mask):
        raise ValueError("Scalable temporal CNN training split is empty.")
    threshold = _active_threshold(targets[train_mask], active_threshold)
    loss_mode = _normalize_pixel_loss_mode(loss_mode)

    model = ScalableTemporalCNNResidual(
        input_channels=int(windows.shape[2]),
        window_frames=int(windows.shape[1]),
        architecture_spec=architecture_spec,
        residual_scale=float(residual_scale),
    ).to(device)
    canonical_spec = dict(model.architecture_spec)
    summary = architecture_summary(
        canonical_spec,
        input_channels=int(windows.shape[2]),
        window_frames=int(windows.shape[1]),
        grid_size=int(windows.shape[-1]) if windows.ndim == 5 and windows.shape[-1] == windows.shape[-2] else None,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_indices = np.flatnonzero(train_mask).astype(np.int64)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        model.train()
        perm = rng.permutation(train_indices)
        epoch_losses = []
        for start in range(0, int(perm.shape[0]), int(batch_size)):
            batch_idx = perm[start : start + int(batch_size)]
            x_batch = torch.from_numpy(windows[batch_idx].astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
            y_batch = torch.from_numpy(targets[batch_idx].astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
            pred, residual = model(x_batch, return_residual=True)
            loss = _pixel_prediction_loss(
                pred,
                y_batch,
                x_batch[:, -1],
                loss_mode=loss_mode,
                residual=residual,
                active_weight=active_weight,
                threshold=threshold,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    pred = predict_scalable_temporal_cnn(model, windows, batch_size=batch_size, device=device)
    metrics = _prediction_metrics(
        pred=pred,
        targets=targets,
        windows=windows,
        video_ids=video_ids,
        splits=dataset.get("splits"),
        objective=f"scalable_temporal_cnn_{loss_mode}",
        training_loss=losses,
        train_count=int(train_mask.sum()),
        active_threshold=threshold,
        active_weight=active_weight if _loss_uses_motion_weights(loss_mode) else 0.0,
    )
    metrics.update(
        {
            "model_kind": "scalable_temporal_cnn_pixel_residual",
            "model_family": "scalable_temporal_cnn",
            "architecture": "scalable_temporal_cnn_pixel",
            "architecture_id": canonical_spec["architecture_id"],
            "architecture_spec": canonical_spec,
            "architecture_summary": summary,
            "loss_mode": loss_mode,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip_norm": float(gradient_clip_norm),
            "residual_scale": float(residual_scale),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "seed": int(seed),
            "device": str(device),
            "input_normalization": "finite_clipped_unit_interval",
            "decoded_output_normalization": "last_frame_plus_tanh_residual_clipped_unit_interval",
        }
    )
    for key, value in summary.items():
        if key not in metrics:
            metrics[key] = value
    metrics_path = out / "concept_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = out / "concept_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "scalable_temporal_cnn_pixel",
            "model_kind": "scalable_temporal_cnn_pixel_residual",
            "model_family": "scalable_temporal_cnn",
            "architecture_id": canonical_spec["architecture_id"],
            "architecture_spec": canonical_spec,
            "architecture_summary": summary,
            "input_channels": int(windows.shape[2]),
            "window_frames": int(windows.shape[1]),
            "residual_scale": float(residual_scale),
            "loss_mode": loss_mode,
            "objective": metrics["objective"],
        },
        checkpoint,
    )
    run = {
        "schema_version": 1,
        "run_id": out.parent.name or "scalable_temporal_cnn_v1",
        "model_kind": "scalable_temporal_cnn_pixel_residual",
        "model_family": "scalable_temporal_cnn",
        "architecture_id": canonical_spec["architecture_id"],
        "architecture_spec": canonical_spec,
        "architecture_summary": summary,
        "source_dataset": str(dataset.get("array_path")),
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip_norm": float(gradient_clip_norm),
            "loss_mode": loss_mode,
            "active_weight": float(active_weight),
            "active_threshold": float(threshold),
            "residual_scale": float(residual_scale),
        },
        "seed": int(seed),
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
    }
    (out / "scalable_temporal_cnn_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return run


def write_architecture_catalog(path: str | Path, *, input_channels: int = 1, window_frames: int = 8, grid_sizes: Sequence[int] = (64, 128)) -> Path:
    out = Path(path)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architectures": [
            {
                "spec": canonical_architecture_spec(spec),
                "summaries": {
                    str(grid_size): architecture_summary(spec, input_channels=input_channels, window_frames=window_frames, grid_size=int(grid_size))
                    for grid_size in grid_sizes
                },
            }
            for spec in architecture_catalog()
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
