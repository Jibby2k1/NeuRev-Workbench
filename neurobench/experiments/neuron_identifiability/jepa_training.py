"""Deterministic matched training for the bounded spatiotemporal JEPA pilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spatiotemporal_jepa import (
    EncoderConfig,
    FrozenRandomEncoderBaseline,
    MaskConfig,
    MaskedPixelAutoencoder,
    SpatiotemporalJEPA,
    count_parameters,
    representation_diagnostics,
    set_deterministic_seed,
)


class JEPATrainingError(ValueError):
    """Raised when a matched training run violates its frozen contract."""


@dataclass(frozen=True)
class TrainingSchedule:
    steps: int = 5_000
    hard_step_cap: int = 10_000
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    optimizer: str = "AdamW"
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1e-8
    learning_rate_schedule: str = "constant"
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.996
    log_interval: int = 100
    amp: bool = True
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= self.hard_step_cap <= 10_000:
            raise JEPATrainingError("steps must be positive and at or below the 10k hard cap")
        if self.batch_size < 1 or self.log_interval < 1:
            raise JEPATrainingError("batch_size and log_interval must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise JEPATrainingError("learning rate must be positive and weight decay nonnegative")
        if self.optimizer != "AdamW" or self.learning_rate_schedule != "constant":
            raise JEPATrainingError(
                "only AdamW with a constant learning-rate schedule is supported"
            )
        if (
            len(self.adamw_betas) != 2
            or not 0 <= self.adamw_betas[0] < 1
            or not 0 <= self.adamw_betas[1] < 1
            or self.adamw_epsilon <= 0
        ):
            raise JEPATrainingError(
                "AdamW betas must be in [0,1) and epsilon must be positive"
            )
        if self.gradient_clip_norm <= 0:
            raise JEPATrainingError("gradient clip norm must be positive")
        if not 0 <= self.ema_decay <= 1:
            raise JEPATrainingError("EMA decay must be in [0, 1]")


@dataclass
class MatchedTrainingResult:
    jepa: SpatiotemporalJEPA
    mae: MaskedPixelAutoencoder
    random: FrozenRandomEncoderBaseline
    metrics: dict[str, Any]
    optimizer_states: dict[str, dict[str, Any]]


def _validated_clip_bank(values: np.ndarray, *, name: str) -> np.ndarray:
    clips = np.asarray(values, dtype=np.float32)
    if clips.ndim != 4 or clips.shape[0] < 1 or not np.isfinite(clips).all():
        raise JEPATrainingError(f"{name} clips must be finite [clip,time,row,column]")
    return clips


def _batch_tensor(clips: np.ndarray, indices: np.ndarray, device: torch.device) -> Tensor:
    values = torch.from_numpy(np.ascontiguousarray(clips[indices])).unsqueeze(1)
    return values.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        # The raw-calcium dynamic range can overflow fp16 in the pixel-MAE
        # squared-error arm.  Bfloat16 preserves mixed-precision throughput
        # while retaining float32-like exponent range on supported CUDA GPUs.
        dtype=torch.bfloat16,
        enabled=enabled,
    )


def _optimizer_step(
    loss: Tensor,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
) -> float:
    if loss.dtype != torch.float32:
        raise JEPATrainingError(
            f"loss accumulation must be float32; observed {loss.dtype}"
        )
    if not bool(torch.isfinite(loss).item()):
        raise JEPATrainingError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not bool(torch.isfinite(gradient_norm).item()):
        raise JEPATrainingError("non-finite gradient norm")
    scaler.step(optimizer)
    scaler.update()
    return float(gradient_norm.detach().cpu())


@torch.no_grad()
def _validation_metrics(
    jepa: SpatiotemporalJEPA,
    mae: MaskedPixelAutoencoder,
    random: FrozenRandomEncoderBaseline,
    validation_clips: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    jepa.eval()
    mae.eval()
    random.eval()
    jepa_losses: list[float] = []
    mae_losses: list[float] = []
    jepa_embeddings: list[Tensor] = []
    mae_embeddings: list[Tensor] = []
    random_embeddings: list[Tensor] = []
    observed_loss_dtypes: dict[str, set[str]] = {"jepa": set(), "mae": set()}
    for batch_number, start in enumerate(range(0, len(validation_clips), batch_size)):
        indices = np.arange(start, min(start + batch_size, len(validation_clips)))
        batch = _batch_tensor(validation_clips, indices, device)
        mask_seed = seed + 50_000_000 + batch_number
        with _autocast(device, amp):
            jepa_output = jepa(batch, mask_seed=mask_seed)
            mae_output = mae(batch, mask_seed=mask_seed)
            jepa_embedding = jepa.encode_target(batch)
            mae_embedding = mae.encoder(batch)
            random_embedding = random(batch)
        observed_loss_dtypes["jepa"].add(str(jepa_output.loss.dtype).removeprefix("torch."))
        observed_loss_dtypes["mae"].add(str(mae_output.loss.dtype).removeprefix("torch."))
        if jepa_output.loss.dtype != torch.float32 or mae_output.loss.dtype != torch.float32:
            raise JEPATrainingError("validation losses must accumulate in float32")
        jepa_losses.append(float(jepa_output.loss.detach().float().cpu()))
        mae_losses.append(float(mae_output.loss.detach().float().cpu()))
        jepa_embeddings.append(jepa_embedding.detach().float().cpu())
        mae_embeddings.append(mae_embedding.detach().float().cpu())
        random_embeddings.append(random_embedding.detach().float().cpu())
    diagnostics = {
        "jepa": representation_diagnostics(torch.cat(jepa_embeddings)).as_dict(),
        "mae": representation_diagnostics(torch.cat(mae_embeddings)).as_dict(),
        "random": representation_diagnostics(torch.cat(random_embeddings)).as_dict(),
    }
    return {
        "jepa_masked_latent_loss": float(np.mean(jepa_losses)),
        "mae_masked_pixel_loss": float(np.mean(mae_losses)),
        "representation_diagnostics": diagnostics,
        "observed_loss_dtypes": {
            key: sorted(values) for key, values in observed_loss_dtypes.items()
        },
    }


def train_matched_representations(
    train_clips: np.ndarray,
    validation_clips: np.ndarray,
    *,
    encoder_config: EncoderConfig,
    mask_config: MaskConfig,
    schedule: TrainingSchedule,
    seed: int,
    device: str | torch.device,
) -> MatchedTrainingResult:
    """Train JEPA and pixel-MAE controls on identical clips/masks/step budget."""

    training = _validated_clip_bank(train_clips, name="training")
    validation = _validated_clip_bank(validation_clips, name="validation")
    if training.shape[1:] != validation.shape[1:]:
        raise JEPATrainingError("training and validation clip geometry differs")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise JEPATrainingError("CUDA was requested but is unavailable")
    set_deterministic_seed(seed, deterministic_algorithms=schedule.deterministic_algorithms)
    jepa = SpatiotemporalJEPA(
        encoder_config,
        mask_config,
        initialization_seed=seed,
        ema_decay=schedule.ema_decay,
        normalize_latents=True,
    ).to(resolved_device)
    mae = MaskedPixelAutoencoder(
        encoder_config, mask_config, initialization_seed=seed
    ).to(resolved_device)
    random = FrozenRandomEncoderBaseline(
        encoder_config, initialization_seed=seed
    ).to(resolved_device)
    jepa_optimizer = torch.optim.AdamW(
        (parameter for parameter in jepa.parameters() if parameter.requires_grad),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
        betas=schedule.adamw_betas,
        eps=schedule.adamw_epsilon,
    )
    mae_optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
        betas=schedule.adamw_betas,
        eps=schedule.adamw_epsilon,
    )
    amp = bool(schedule.amp and resolved_device.type == "cuda")
    # Gradient scaling is an fp16 remedy and is unnecessary for bfloat16.
    jepa_scaler = torch.amp.GradScaler("cuda", enabled=False)
    mae_scaler = torch.amp.GradScaler("cuda", enabled=False)
    rng = np.random.default_rng(seed + 17)
    curves: list[dict[str, float | int]] = []
    observed_training_loss_dtypes: dict[str, set[str]] = {"jepa": set(), "mae": set()}
    jepa.train()
    mae.train()
    for step_index in range(schedule.steps):
        step = step_index + 1
        indices = rng.integers(0, len(training), size=schedule.batch_size)
        batch = _batch_tensor(training, indices, resolved_device)
        mask_seed = seed + step_index
        with _autocast(resolved_device, amp):
            jepa_output = jepa(batch, mask_seed=mask_seed)
        observed_training_loss_dtypes["jepa"].add(
            str(jepa_output.loss.dtype).removeprefix("torch.")
        )
        jepa_gradient = _optimizer_step(
            jepa_output.loss,
            model=jepa,
            optimizer=jepa_optimizer,
            scaler=jepa_scaler,
            gradient_clip_norm=schedule.gradient_clip_norm,
        )
        jepa.update_target_encoder()
        with _autocast(resolved_device, amp):
            mae_output = mae(batch, mask_seed=mask_seed)
        observed_training_loss_dtypes["mae"].add(
            str(mae_output.loss.dtype).removeprefix("torch.")
        )
        mae_gradient = _optimizer_step(
            mae_output.loss,
            model=mae,
            optimizer=mae_optimizer,
            scaler=mae_scaler,
            gradient_clip_norm=schedule.gradient_clip_norm,
        )
        if step == 1 or step == schedule.steps or step % schedule.log_interval == 0:
            curves.append(
                {
                    "step": step,
                    "jepa_masked_latent_loss": float(jepa_output.loss.detach().float().cpu()),
                    "mae_masked_pixel_loss": float(mae_output.loss.detach().float().cpu()),
                    "jepa_gradient_norm": jepa_gradient,
                    "mae_gradient_norm": mae_gradient,
                }
            )
    validation_metrics = _validation_metrics(
        jepa,
        mae,
        random,
        validation,
        batch_size=schedule.batch_size,
        seed=seed,
        device=resolved_device,
        amp=amp,
    )
    jepa_trainable = count_parameters(jepa, trainable_only=True)
    mae_trainable = count_parameters(mae, trainable_only=True)
    capacity_difference = abs(jepa_trainable - mae_trainable) / max(jepa_trainable, 1)
    metrics = {
        "seed": int(seed),
        "device": str(resolved_device),
        "amp_used": amp,
        "amp_dtype": "bfloat16" if amp else None,
        "float16_used": any(
            values == "float16"
            for arm_values in observed_training_loss_dtypes.values()
            for values in arm_values
        ),
        "loss_accumulation_dtype": "float32",
        "metric_dtype": "float32",
        "observed_training_loss_dtypes": {
            key: sorted(values)
            for key, values in observed_training_loss_dtypes.items()
        },
        "schedule": asdict(schedule),
        "train_clip_count": int(len(training)),
        "validation_clip_count": int(len(validation)),
        "clip_shape_tyx": list(training.shape[1:]),
        "capacity": {
            "jepa_trainable_parameters": jepa_trainable,
            "mae_trainable_parameters": mae_trainable,
            "random_trainable_parameters": count_parameters(random, trainable_only=True),
            "relative_jepa_mae_difference": capacity_difference,
            "matched_within_one_percent": capacity_difference <= 0.01,
        },
        "training_curve": curves,
        "validation": validation_metrics,
    }
    return MatchedTrainingResult(
        jepa=jepa,
        mae=mae,
        random=random,
        metrics=metrics,
        optimizer_states={
            "jepa": jepa_optimizer.state_dict(),
            "mae": mae_optimizer.state_dict(),
        },
    )


@torch.no_grad()
def latent_temporal_change_score(
    embeddings: Tensor,
    *,
    output_shape_yx: tuple[int, int],
) -> np.ndarray:
    """Map frozen latent temporal change to pixels; not a neuron probability."""

    if embeddings.ndim != 5 or embeddings.shape[0] != 1 or embeddings.shape[2] < 2:
        raise JEPATrainingError("embeddings must be [1,feature,time,row,column] with time>=2")
    change = torch.linalg.vector_norm(torch.diff(embeddings.float(), dim=2), dim=1).mean(dim=1)
    upsampled = F.interpolate(
        change.unsqueeze(1), size=output_shape_yx, mode="bilinear", align_corners=False
    )[0, 0]
    score = upsampled.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(score).all():
        raise JEPATrainingError("latent temporal-change score is non-finite")
    return score


def model_latent_temporal_change_scorer(
    model: SpatiotemporalJEPA | MaskedPixelAutoencoder | FrozenRandomEncoderBaseline,
    *,
    device: str | torch.device,
) -> callable:
    """Build a raw-movie scorer from a frozen matched encoder."""

    resolved_device = torch.device(device)
    model.eval()

    def score(movie: np.ndarray) -> np.ndarray:
        values = np.asarray(movie, dtype=np.float32)
        if values.ndim != 3:
            raise JEPATrainingError("score movie must be TYX")
        tensor = torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0).unsqueeze(0).to(resolved_device)
        with torch.no_grad():
            if isinstance(model, SpatiotemporalJEPA):
                embedding = model.encode_target(tensor)
            elif isinstance(model, MaskedPixelAutoencoder):
                embedding = model.encoder(tensor)
            else:
                embedding = model(tensor)
        return latent_temporal_change_score(embedding, output_shape_yx=values.shape[1:])

    return score
