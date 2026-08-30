"""Frozen-latent pixel background prediction for calcium-video residuals.

This module turns a frozen masked-latent provider into a pixel-space background
estimate.  It never subtracts a latent representation directly.  For each
spatial tubelet, the complete temporal tube and its one-token spatial
neighbourhood are hidden, the central latent tube is predicted from the
remaining context, and a small transposed-convolution head decodes only that
central patch.  Tiling the central predictions covers every input voxel once.

The primary provider is the learned JEPA predictor applied to its masked
context encoding.  A checkpoint-matched frozen random encoder is exposed as a
negative control.  In both cases the provider is immutable and only the pixel
head is trainable.  Inputs and pixel-space outputs use ``[B,C,T,H,W]`` order.

Residuals are signed observations minus predictions.  They are diagnostics,
not automatic estimates of neural signal: paired retention, absorption,
background suppression, closure, and patch-seam measurements are provided so
a runner can test that interpretation instead of assuming it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .discovery import sha256_file
from .spatiotemporal_jepa import (
    EncoderConfig,
    FrozenRandomEncoderBaseline,
    MaskConfig,
    SpatiotemporalJEPA,
    apply_patch_mask,
    count_parameters,
    patch_grid_shape,
)


SMOOTH_L1_BETA = 1.0
PRODUCTION_PATCH_SIZE_TYX = (4, 8, 8)
PRODUCTION_GRID_YX = (8, 8)
PRODUCTION_LATENT_DIM = 64
PRODUCTION_DECODER_PARAMETERS = 16_385
BLIND_HALO_SPATIAL_TOKENS = 3
REFLECT_PADDING_SPATIAL_TOKENS = 1


class JEPABackgroundResidualError(ValueError):
    """Raised when the frozen-background contract cannot be preserved."""


def _validated_seed(seed: int) -> int:
    resolved = int(seed)
    if resolved < 0 or resolved >= 2**63:
        raise JEPABackgroundResidualError("seed must be in [0, 2**63 - 1]")
    return resolved


def _finite_float(value: float, *, name: str, positive: bool = False) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or (positive and resolved <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise JEPABackgroundResidualError(f"{name} must be {qualifier}")
    return resolved


def _shape3(values: Sequence[int], *, name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise JEPABackgroundResidualError(f"{name} must contain [time,row,column]")
    resolved = tuple(int(value) for value in values)
    if any(value <= 0 for value in resolved):
        raise JEPABackgroundResidualError(f"{name} values must be positive")
    return resolved  # type: ignore[return-value]


def _validated_video(
    video: Tensor,
    *,
    patch_size: Sequence[int],
    in_channels: int,
) -> Tensor:
    if not isinstance(video, Tensor) or video.ndim != 5:
        raise JEPABackgroundResidualError(
            "video must be a torch.Tensor with shape [B,C,T,H,W]"
        )
    if video.dtype == torch.bool or video.is_complex():
        raise JEPABackgroundResidualError("video must have a real numeric dtype")
    if int(video.shape[0]) < 1 or int(video.shape[1]) != int(in_channels):
        raise JEPABackgroundResidualError(
            f"video must have a nonempty batch and exactly {in_channels} channels"
        )
    if not bool(torch.isfinite(video).all()):
        raise JEPABackgroundResidualError("video must contain only finite values")
    grid = patch_grid_shape(video, patch_size)
    if grid[1] < 2 or grid[2] < 2:
        raise JEPABackgroundResidualError(
            "reflect-padded blind-halo prediction requires at least 2x2 spatial tokens"
        )
    return video if video.is_floating_point() else video.to(dtype=torch.float32)


def _module_state_sha256(module: nn.Module) -> str:
    """Hash a module state without changing devices or dtypes."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Return a stable tensor-level SHA-256 for caller-side freeze checks."""

    if not isinstance(module, nn.Module):
        raise JEPABackgroundResidualError("module must be a torch.nn.Module")
    return _module_state_sha256(module)


def _frozen(module: nn.Module) -> bool:
    return all(not parameter.requires_grad for parameter in module.parameters())


class FrozenMaskedLatentProvider(nn.Module, ABC):
    """Interface for immutable masked-video latent providers."""

    encoder_config: EncoderConfig
    mask_value: float
    method_id: str

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.encoder_config.patch_size

    @property
    def latent_dim(self) -> int:
        return int(self.encoder_config.latent_dim)

    @property
    def in_channels(self) -> int:
        return int(self.encoder_config.in_channels)

    @abstractmethod
    def masked_latents(self, video: Tensor, patch_mask: Tensor) -> Tensor:
        """Return immutable latent features from a structurally masked video."""

    def train(self, mode: bool = True) -> "FrozenMaskedLatentProvider":
        super().train(False)
        return self


class FrozenJEPALatentProvider(FrozenMaskedLatentProvider):
    """Frozen JEPA context encoder plus learned latent predictor."""

    method_id = "frozen_jepa_predicted_latent_pixel_background_v1"

    def __init__(self, model: SpatiotemporalJEPA) -> None:
        super().__init__()
        self.model = model
        self.encoder_config = model.encoder_config
        self.mask_value = float(model.mask_config.mask_value)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenJEPALatentProvider":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def masked_latents(self, video: Tensor, patch_mask: Tensor) -> Tensor:
        masked = apply_patch_mask(
            video,
            patch_mask,
            patch_size=self.patch_size,
            mask_value=self.mask_value,
        )
        return self.model.predictor(self.model.context_encoder(masked)).detach()


class FrozenRandomLatentProvider(FrozenMaskedLatentProvider):
    """Checkpoint-matched frozen random encoder negative control."""

    method_id = "frozen_random_encoder_pixel_background_control_v1"

    def __init__(
        self,
        model: FrozenRandomEncoderBaseline,
        *,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.encoder_config = model.encoder_config
        self.mask_value = _finite_float(mask_value, name="mask_value")
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenRandomLatentProvider":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def masked_latents(self, video: Tensor, patch_mask: Tensor) -> Tensor:
        return self.model(
            video,
            patch_mask=patch_mask,
            mask_value=self.mask_value,
        ).detach()


@dataclass(frozen=True)
class LoadedFrozenProvider:
    """A strictly reconstructed provider and verifiable checkpoint identity."""

    provider: FrozenMaskedLatentProvider
    checkpoint_path: Path
    checkpoint_sha256: str
    expected_sha256: str | None
    hash_verified: bool
    checkpoint_schema_version: int | None
    training_seed: int | None
    mode: str | None
    scientific_checkpoint: bool | None
    training_schedule: Mapping[str, Any]
    source_arm: str

    @property
    def model(self) -> nn.Module:
        """Expose the strict checkpoint module for independent tensor hashing."""

        return self.provider.model  # type: ignore[attr-defined,no-any-return]

    @property
    def module(self) -> nn.Module:
        """Compatibility alias for provenance-oriented runners."""

        return self.model

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "loaded_frozen_latent_provider_v1",
            "source_arm": self.source_arm,
            "method_id": self.provider.method_id,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "expected_sha256": self.expected_sha256,
            "hash_verified": self.hash_verified,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "training_seed": self.training_seed,
            "mode": self.mode,
            "scientific_checkpoint": self.scientific_checkpoint,
            "training_schedule": dict(self.training_schedule),
            "encoder_config": asdict(self.provider.encoder_config),
            "provider_frozen": _frozen(self.provider),
            "provider_state_sha256": _module_state_sha256(self.provider),
        }


def _checkpoint_payload(
    path: Path,
    *,
    expected_sha256: str | None,
    map_location: str | torch.device,
) -> tuple[Path, str, Mapping[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise JEPABackgroundResidualError(f"checkpoint does not exist: {resolved}")
    observed = sha256_file(resolved)
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise JEPABackgroundResidualError("expected_sha256 must be 64 lowercase hex characters")
        if observed != expected:
            raise JEPABackgroundResidualError(
                f"checkpoint SHA-256 mismatch: expected {expected}, observed {observed}"
            )
    try:
        payload = torch.load(resolved, map_location=map_location, weights_only=True)
    except Exception as exc:  # PyTorch exposes several serialization exceptions.
        raise JEPABackgroundResidualError(
            f"could not safely load checkpoint {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise JEPABackgroundResidualError("checkpoint root must be a mapping")
    return resolved, observed, payload


def _configs_from_checkpoint(payload: Mapping[str, Any]) -> tuple[EncoderConfig, MaskConfig]:
    encoder_payload = payload.get("encoder_config")
    mask_payload = payload.get("mask_config")
    if not isinstance(encoder_payload, Mapping) or not isinstance(mask_payload, Mapping):
        raise JEPABackgroundResidualError(
            "checkpoint must contain encoder_config and mask_config mappings"
        )
    try:
        encoder = EncoderConfig(**dict(encoder_payload))
        mask = MaskConfig(**dict(mask_payload))
    except (TypeError, ValueError) as exc:
        raise JEPABackgroundResidualError(f"invalid checkpoint architecture: {exc}") from exc
    return encoder, mask


def load_frozen_jepa_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedFrozenProvider:
    """Strictly reconstruct and freeze the JEPA arm from a pilot checkpoint."""

    resolved, observed, payload = _checkpoint_payload(
        path, expected_sha256=expected_sha256, map_location=map_location
    )
    encoder, mask = _configs_from_checkpoint(payload)
    state = payload.get("jepa_state_dict")
    if not isinstance(state, Mapping):
        raise JEPABackgroundResidualError("checkpoint is missing jepa_state_dict")
    schedule = payload.get("schedule")
    ema_decay = 0.996
    if isinstance(schedule, Mapping) and "ema_decay" in schedule:
        ema_decay = float(schedule["ema_decay"])
    model = SpatiotemporalJEPA(
        encoder,
        mask,
        initialization_seed=0,
        ema_decay=ema_decay,
        # The registered trainer always used normalized masked-latent loss.
        # This flag does not change the provider path below, but reconstructing
        # it preserves the complete checkpoint model semantics.
        normalize_latents=True,
    )
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise JEPABackgroundResidualError(
            f"strict JEPA state-dict reconstruction failed: {exc}"
        ) from exc
    provider = FrozenJEPALatentProvider(model)
    return LoadedFrozenProvider(
        provider=provider,
        checkpoint_path=resolved,
        checkpoint_sha256=observed,
        expected_sha256=expected_sha256,
        hash_verified=expected_sha256 is not None and observed == expected_sha256,
        checkpoint_schema_version=(
            int(payload["schema_version"]) if "schema_version" in payload else None
        ),
        training_seed=(
            int(payload["training_seed"]) if "training_seed" in payload else None
        ),
        mode=str(payload["mode"]) if "mode" in payload else None,
        scientific_checkpoint=(
            bool(payload["scientific_checkpoint"])
            if "scientific_checkpoint" in payload
            else None
        ),
        training_schedule=(
            dict(payload["schedule"])
            if isinstance(payload.get("schedule"), Mapping)
            else {}
        ),
        source_arm="jepa",
    )


def load_frozen_random_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedFrozenProvider:
    """Strictly reconstruct the checkpoint-matched frozen random control."""

    resolved, observed, payload = _checkpoint_payload(
        path, expected_sha256=expected_sha256, map_location=map_location
    )
    encoder, mask = _configs_from_checkpoint(payload)
    state = payload.get("random_state_dict")
    if not isinstance(state, Mapping):
        raise JEPABackgroundResidualError("checkpoint is missing random_state_dict")
    model = FrozenRandomEncoderBaseline(encoder, initialization_seed=0)
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise JEPABackgroundResidualError(
            f"strict random state-dict reconstruction failed: {exc}"
        ) from exc
    provider = FrozenRandomLatentProvider(model, mask_value=mask.mask_value)
    return LoadedFrozenProvider(
        provider=provider,
        checkpoint_path=resolved,
        checkpoint_sha256=observed,
        expected_sha256=expected_sha256,
        hash_verified=expected_sha256 is not None and observed == expected_sha256,
        checkpoint_schema_version=(
            int(payload["schema_version"]) if "schema_version" in payload else None
        ),
        training_seed=(
            int(payload["training_seed"]) if "training_seed" in payload else None
        ),
        mode=str(payload["mode"]) if "mode" in payload else None,
        scientific_checkpoint=(
            bool(payload["scientific_checkpoint"])
            if "scientific_checkpoint" in payload
            else None
        ),
        training_schedule=(
            dict(payload["schedule"])
            if isinstance(payload.get("schedule"), Mapping)
            else {}
        ),
        source_arm="random",
    )


@dataclass(frozen=True)
class BlindHaloTarget:
    target_index: int
    target_y: int
    target_x: int
    padded_center_y: int
    padded_center_x: int
    halo_token_bounds_yx: tuple[tuple[int, int], tuple[int, int]]
    output_voxel_bounds_yx: tuple[tuple[int, int], tuple[int, int]]
    full_time_masked_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_index": self.target_index,
            "target_token_yx": [self.target_y, self.target_x],
            "padded_center_token_yx": [self.padded_center_y, self.padded_center_x],
            "halo_token_bounds_yx_half_open": [
                list(self.halo_token_bounds_yx[0]),
                list(self.halo_token_bounds_yx[1]),
            ],
            "output_voxel_bounds_yx_half_open": [
                list(self.output_voxel_bounds_yx[0]),
                list(self.output_voxel_bounds_yx[1]),
            ],
            "full_time_masked_tokens": self.full_time_masked_tokens,
        }


@dataclass(frozen=True)
class BlindHaloManifest:
    input_shape_bcthw: tuple[int, int, int, int, int]
    patch_size_tyx: tuple[int, int, int]
    original_grid_tyx: tuple[int, int, int]
    padded_grid_tyx: tuple[int, int, int]
    reflect_padding_voxels_yx: tuple[int, int]
    blind_halo_tokens_tyx: tuple[int, int, int]
    targets: tuple[BlindHaloTarget, ...]

    @property
    def target_count(self) -> int:
        return len(self.targets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "full_time_spatial_blind_halo_v1",
            "input_shape_bcthw": list(self.input_shape_bcthw),
            "patch_size_tyx": list(self.patch_size_tyx),
            "original_grid_tyx": list(self.original_grid_tyx),
            "padded_grid_tyx": list(self.padded_grid_tyx),
            "spatial_padding": {
                "mode": "reflect",
                "tokens_each_side": REFLECT_PADDING_SPATIAL_TOKENS,
                "voxels_yx_each_side": list(self.reflect_padding_voxels_yx),
            },
            "blind_halo_tokens_tyx": list(self.blind_halo_tokens_tyx),
            "target_count": self.target_count,
            "central_target_only_decoded": True,
            "target_tube_masked_for_all_time_tokens": True,
            "original_halo_premasked_before_reflect": True,
            "reflect_alias_leakage_structurally_blocked": True,
            "targets": [target.as_dict() for target in self.targets],
        }


def blind_halo_manifest(video: Tensor, patch_size: Sequence[int]) -> BlindHaloManifest:
    """Describe the fixed reflect-pad/full-time 3x3 blind-halo schedule."""

    patch = _shape3(patch_size, name="patch_size")
    if video.ndim != 5:
        raise JEPABackgroundResidualError("video must have shape [B,C,T,H,W]")
    grid = patch_grid_shape(video, patch)
    if grid[1] < 2 or grid[2] < 2:
        raise JEPABackgroundResidualError("blind-halo grid must be at least 2x2")
    targets: list[BlindHaloTarget] = []
    for target_y in range(grid[1]):
        for target_x in range(grid[2]):
            padded_y = target_y + REFLECT_PADDING_SPATIAL_TOKENS
            padded_x = target_x + REFLECT_PADDING_SPATIAL_TOKENS
            targets.append(
                BlindHaloTarget(
                    target_index=len(targets),
                    target_y=target_y,
                    target_x=target_x,
                    padded_center_y=padded_y,
                    padded_center_x=padded_x,
                    halo_token_bounds_yx=(
                        (padded_y - 1, padded_y + 2),
                        (padded_x - 1, padded_x + 2),
                    ),
                    output_voxel_bounds_yx=(
                        (target_y * patch[1], (target_y + 1) * patch[1]),
                        (target_x * patch[2], (target_x + 1) * patch[2]),
                    ),
                    full_time_masked_tokens=grid[0]
                    * BLIND_HALO_SPATIAL_TOKENS
                    * BLIND_HALO_SPATIAL_TOKENS,
                )
            )
    return BlindHaloManifest(
        input_shape_bcthw=tuple(int(value) for value in video.shape),  # type: ignore[arg-type]
        patch_size_tyx=patch,
        original_grid_tyx=grid,
        padded_grid_tyx=(grid[0], grid[1] + 2, grid[2] + 2),
        reflect_padding_voxels_yx=(patch[1], patch[2]),
        blind_halo_tokens_tyx=(
            grid[0],
            BLIND_HALO_SPATIAL_TOKENS,
            BLIND_HALO_SPATIAL_TOKENS,
        ),
        targets=tuple(targets),
    )


def _reflect_pad(video: Tensor, patch_size: Sequence[int]) -> Tensor:
    patch = _shape3(patch_size, name="patch_size")
    return F.pad(
        video,
        (patch[2], patch[2], patch[1], patch[1], 0, 0),
        mode="reflect",
    )


def _premask_original_halo_then_reflect(
    video: Tensor,
    manifest: BlindHaloManifest,
    target: BlindHaloTarget,
    *,
    mask_value: float,
) -> Tensor:
    """Mask original halo before reflection so padded aliases cannot leak."""

    _, patch_height, patch_width = manifest.patch_size_tyx
    height, width = map(int, video.shape[-2:])
    y0 = max(0, (target.target_y - 1) * patch_height)
    y1 = min(height, (target.target_y + 2) * patch_height)
    x0 = max(0, (target.target_x - 1) * patch_width)
    x1 = min(width, (target.target_x + 2) * patch_width)
    masked = video.clone()
    masked[:, :, :, y0:y1, x0:x1] = float(mask_value)
    return _reflect_pad(masked, manifest.patch_size_tyx)


def _target_mask(
    manifest: BlindHaloManifest,
    target: BlindHaloTarget,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(
        (batch_size, *manifest.padded_grid_tyx),
        dtype=torch.bool,
        device=device,
    )
    (y0, y1), (x0, x1) = target.halo_token_bounds_yx
    mask[:, :, y0:y1, x0:x1] = True
    return mask


class FrozenLatentPixelBackgroundModel(nn.Module):
    """Immutable latent provider plus the sole trainable pixel decoder."""

    def __init__(
        self,
        provider: FrozenMaskedLatentProvider | LoadedFrozenProvider,
        *,
        decoder_initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(provider, LoadedFrozenProvider):
            provider = provider.provider
        if not isinstance(provider, FrozenMaskedLatentProvider):
            raise JEPABackgroundResidualError(
                "provider must implement FrozenMaskedLatentProvider"
            )
        provider.requires_grad_(False)
        provider.eval()
        self.provider = provider
        seed = _validated_seed(decoder_initialization_seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.pixel_decoder = nn.ConvTranspose3d(
                provider.latent_dim,
                provider.in_channels,
                kernel_size=provider.patch_size,
                stride=provider.patch_size,
                bias=True,
            )

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.provider.patch_size

    @property
    def decoder_parameter_count(self) -> int:
        return count_parameters(self.pixel_decoder)

    @property
    def decoder(self) -> nn.Module:
        """Expose only the trainable head to runner checkpointing code."""

        return self.pixel_decoder

    @property
    def trainable_parameter_count(self) -> int:
        return count_parameters(self, trainable_only=True)

    def train(self, mode: bool = True) -> "FrozenLatentPixelBackgroundModel":
        super().train(mode)
        self.provider.eval()
        return self

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.pixel_decoder.parameters()

    @torch.no_grad()
    def masked_target_latent(self, video: Tensor, target_yx: tuple[int, int]) -> Tensor:
        values = _validated_video(
            video,
            patch_size=self.patch_size,
            in_channels=self.provider.in_channels,
        )
        manifest = blind_halo_manifest(values, self.patch_size)
        target_y, target_x = map(int, target_yx)
        if not (0 <= target_y < manifest.original_grid_tyx[1]) or not (
            0 <= target_x < manifest.original_grid_tyx[2]
        ):
            raise JEPABackgroundResidualError(
                f"target_yx {target_yx} is outside spatial grid {manifest.original_grid_tyx[1:]}"
            )
        target = manifest.targets[target_y * manifest.original_grid_tyx[2] + target_x]
        padded = _premask_original_halo_then_reflect(
            values,
            manifest,
            target,
            mask_value=self.provider.mask_value,
        )
        mask = _target_mask(
            manifest,
            target,
            batch_size=int(values.shape[0]),
            device=values.device,
        )
        latent = self.provider.masked_latents(padded, mask)
        return latent[
            :,
            :,
            :,
            target.padded_center_y : target.padded_center_y + 1,
            target.padded_center_x : target.padded_center_x + 1,
        ].detach()

    def decode_target_latent(self, central_latent: Tensor) -> Tensor:
        if central_latent.ndim != 5 or tuple(central_latent.shape[-2:]) != (1, 1):
            raise JEPABackgroundResidualError(
                "central_latent must have shape [B,latent,Tg,1,1]"
            )
        if int(central_latent.shape[1]) != self.provider.latent_dim:
            raise JEPABackgroundResidualError("central_latent feature dimension differs")
        return self.pixel_decoder(central_latent)

    def predict_target_patch(
        self,
        video: Tensor,
        target_yx: tuple[int, int],
        *,
        amp: bool = False,
    ) -> Tensor:
        """Predict one spatial target patch for every batch item."""

        device = video.device
        with _autocast(device, enabled=amp):
            latent = self.masked_target_latent(video, target_yx)
            prediction = self.decode_target_latent(latent)
        return prediction.float()


@dataclass(frozen=True)
class DecoderTrainingConfig:
    """Bounded fixed-objective decoder training schedule."""

    steps: int = 500
    hard_step_cap: int = 2_000
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    log_interval: int = 25
    amp: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.steps) <= int(self.hard_step_cap) <= 2_000:
            raise JEPABackgroundResidualError(
                "steps must be positive and at or below the 2000-step hard cap"
            )
        if int(self.batch_size) < 1 or int(self.log_interval) < 1:
            raise JEPABackgroundResidualError("batch_size and log_interval must be positive")
        _finite_float(self.learning_rate, name="learning_rate", positive=True)
        if _finite_float(self.weight_decay, name="weight_decay") < 0.0:
            raise JEPABackgroundResidualError("weight_decay must be nonnegative")
        _finite_float(
            self.gradient_clip_norm,
            name="gradient_clip_norm",
            positive=True,
        )


@dataclass
class DecoderTrainingResult:
    model: FrozenLatentPixelBackgroundModel
    metrics: dict[str, Any]
    optimizer_state: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return the JSON-safe result surface; the optimizer state is separate."""

        return self.metrics


@contextmanager
def _autocast(device: torch.device, *, enabled: bool):
    use_amp = bool(enabled)
    if use_amp and device.type not in {"cpu", "cuda"}:
        raise JEPABackgroundResidualError(
            "bfloat16 autocast is supported only on CPU or CUDA"
        )
    if use_amp and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise JEPABackgroundResidualError("requested CUDA device does not support bfloat16")
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        yield


@contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _central_target_patch(video: Tensor, target_yx: tuple[int, int], patch: Sequence[int]) -> Tensor:
    patch_time, patch_height, patch_width = _shape3(patch, name="patch_size")
    target_y, target_x = map(int, target_yx)
    return video[
        :,
        :,
        : (int(video.shape[2]) // patch_time) * patch_time,
        target_y * patch_height : (target_y + 1) * patch_height,
        target_x * patch_width : (target_x + 1) * patch_width,
    ]


def train_pixel_decoder(
    model: FrozenLatentPixelBackgroundModel,
    training_clips: Tensor,
    config: DecoderTrainingConfig = DecoderTrainingConfig(),
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> DecoderTrainingResult:
    """Train only the pixel head with fixed Smooth-L1 loss.

    Clip and target-index draws come from a private CPU generator.  Spatial
    targets cycle through a seeded permutation before reshuffling, avoiding a
    hidden mutable mask RNG.  The latent provider is evaluated under
    ``no_grad`` and is checked byte-for-byte before and after optimization.
    """

    if not isinstance(model, FrozenLatentPixelBackgroundModel):
        raise JEPABackgroundResidualError("model must be FrozenLatentPixelBackgroundModel")
    resolved_seed = _validated_seed(seed)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise JEPABackgroundResidualError("CUDA was requested but is unavailable")
    clips = _validated_video(
        training_clips,
        patch_size=model.patch_size,
        in_channels=model.provider.in_channels,
    ).detach().cpu().float()
    model.to(resolved_device)
    model.train(True)
    model.provider.requires_grad_(False)
    model.provider.eval()
    if not _frozen(model.provider):
        raise JEPABackgroundResidualError("latent provider must be completely frozen")
    if model.trainable_parameter_count != model.decoder_parameter_count:
        raise JEPABackgroundResidualError("the pixel decoder must be the only trainable module")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved_seed)
    optimizer = torch.optim.AdamW(
        list(model.decoder_parameters()),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    provider_digest_before = _module_state_sha256(model.provider)
    decoder_digest_before = _module_state_sha256(model.pixel_decoder)
    grid = patch_grid_shape(clips, model.patch_size)
    target_count = grid[1] * grid[2]
    target_order = torch.randperm(target_count, generator=generator).tolist()
    order_cursor = 0
    curve: list[dict[str, Any]] = []
    observed_loss_dtypes: set[str] = set()
    observed_prediction_dtypes: set[str] = set()
    sampling_schedule_digest = hashlib.sha256()
    sampling_schedule_digest.update(b"NEUREV-PIXEL-DECODER-SAMPLING-V1\0")

    with _deterministic_algorithms():
        for step_index in range(int(config.steps)):
            if order_cursor == len(target_order):
                target_order = torch.randperm(target_count, generator=generator).tolist()
                order_cursor = 0
            target_index = int(target_order[order_cursor])
            order_cursor += 1
            target_yx = (target_index // grid[2], target_index % grid[2])
            indices = torch.randint(
                0,
                int(clips.shape[0]),
                (int(config.batch_size),),
                generator=generator,
            )
            sampling_schedule_digest.update(
                torch.tensor(
                    [step_index, target_index], dtype=torch.int64
                ).numpy().tobytes()
            )
            sampling_schedule_digest.update(indices.contiguous().numpy().tobytes())
            batch = clips.index_select(0, indices).to(
                device=resolved_device,
                dtype=torch.float32,
                non_blocking=resolved_device.type == "cuda",
            )
            target = _central_target_patch(batch, target_yx, model.patch_size).float()
            with _autocast(resolved_device, enabled=config.amp):
                latent = model.masked_target_latent(batch, target_yx)
                prediction = model.decode_target_latent(latent)
            observed_prediction_dtypes.add(str(prediction.dtype).removeprefix("torch."))
            loss = F.smooth_l1_loss(
                prediction.float(),
                target,
                reduction="mean",
                beta=SMOOTH_L1_BETA,
            )
            observed_loss_dtypes.add(str(loss.dtype).removeprefix("torch."))
            if loss.dtype != torch.float32 or not bool(torch.isfinite(loss).item()):
                raise JEPABackgroundResidualError(
                    "decoder loss must be finite float32"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(model.decoder_parameters()),
                float(config.gradient_clip_norm),
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise JEPABackgroundResidualError("decoder gradient norm is nonfinite")
            optimizer.step()
            step = step_index + 1
            if step == 1 or step == config.steps or step % config.log_interval == 0:
                curve.append(
                    {
                        "step": step,
                        "target_token_yx": [int(target_yx[0]), int(target_yx[1])],
                        "smooth_l1_loss": float(loss.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                    }
                )

    provider_digest_after = _module_state_sha256(model.provider)
    decoder_digest_after = _module_state_sha256(model.pixel_decoder)
    if provider_digest_before != provider_digest_after:
        raise JEPABackgroundResidualError("frozen latent provider changed during training")
    metrics = {
        "schema_version": "frozen_latent_pixel_decoder_training_v1",
        "seed": resolved_seed,
        "device": str(resolved_device),
        "provider_method_id": model.provider.method_id,
        "training_clip_count": int(clips.shape[0]),
        "clip_shape_bcthw": [int(value) for value in clips.shape],
        "spatial_target_count": target_count,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
        },
        "loss": {
            "name": "SmoothL1Loss",
            "beta": SMOOTH_L1_BETA,
            "reduction": "mean",
            "accumulation_dtype": "float32",
        },
        "amp": {
            "enabled": bool(config.amp),
            "dtype": "bfloat16" if config.amp else None,
            "observed_prediction_dtypes": sorted(observed_prediction_dtypes),
            "observed_loss_dtypes": sorted(observed_loss_dtypes),
        },
        "schedule": asdict(config),
        "sampling_schedule_sha256": sampling_schedule_digest.hexdigest(),
        "sampling_schedule_hashed_fields": [
            "zero_based_step_index",
            "target_token_flat_index",
            "ordered_training_clip_indices",
        ],
        "decoder_parameters": model.decoder_parameter_count,
        "trainable_parameters": model.trainable_parameter_count,
        "provider_frozen": _frozen(model.provider),
        "provider_state_sha256_before": provider_digest_before,
        "provider_state_sha256_after": provider_digest_after,
        "provider_unchanged": provider_digest_before == provider_digest_after,
        "decoder_state_sha256_before": decoder_digest_before,
        "decoder_state_sha256_after": decoder_digest_after,
        "decoder_changed": decoder_digest_before != decoder_digest_after,
        "training_curve": curve,
    }
    return DecoderTrainingResult(
        model=model,
        metrics=metrics,
        optimizer_state=optimizer.state_dict(),
    )


@dataclass
class BackgroundSweepOutput:
    background: Tensor
    coverage_counts: Tensor
    manifest: BlindHaloManifest

    @property
    def minimum_coverage(self) -> int:
        return int(self.coverage_counts.min().item())

    @property
    def maximum_coverage(self) -> int:
        return int(self.coverage_counts.max().item())

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "pixel_background_sweep_v1",
            "background_shape_bcthw": [int(value) for value in self.background.shape],
            "coverage_shape_bcthw": [int(value) for value in self.coverage_counts.shape],
            "minimum_coverage": self.minimum_coverage,
            "maximum_coverage": self.maximum_coverage,
            "exactly_once_coverage": bool(
                self.minimum_coverage == 1 and self.maximum_coverage == 1
            ),
            "finite": bool(torch.isfinite(self.background).all()),
            "mask_manifest": self.manifest.as_dict(),
        }


@torch.no_grad()
def predict_full_background(
    model: FrozenLatentPixelBackgroundModel,
    video: Tensor,
    *,
    target_batch_size: int = 8,
    amp: bool = False,
) -> BackgroundSweepOutput:
    """Tile central blind-halo predictions into a complete pixel background."""

    if int(target_batch_size) < 1:
        raise JEPABackgroundResidualError("target_batch_size must be positive")
    values = _validated_video(
        video,
        patch_size=model.patch_size,
        in_channels=model.provider.in_channels,
    )
    model.eval()
    model.provider.eval()
    manifest = blind_halo_manifest(values, model.patch_size)
    background = torch.zeros_like(values, dtype=torch.float32)
    coverage = torch.zeros_like(values, dtype=torch.int16)
    patch_time, patch_height, patch_width = model.patch_size
    batch_size = int(values.shape[0])

    for start in range(0, manifest.target_count, int(target_batch_size)):
        targets = manifest.targets[start : start + int(target_batch_size)]
        expanded = torch.cat(
            [
                _premask_original_halo_then_reflect(
                    values,
                    manifest,
                    target,
                    mask_value=model.provider.mask_value,
                )
                for target in targets
            ],
            dim=0,
        )
        masks = torch.cat(
            [
                _target_mask(
                    manifest,
                    target,
                    batch_size=batch_size,
                    device=values.device,
                )
                for target in targets
            ],
            dim=0,
        )
        with _autocast(values.device, enabled=amp):
            latents = model.provider.masked_latents(expanded, masks)
            central = torch.cat(
                [
                    latents[
                        target_offset * batch_size : (target_offset + 1) * batch_size,
                        :,
                        :,
                        target.padded_center_y : target.padded_center_y + 1,
                        target.padded_center_x : target.padded_center_x + 1,
                    ]
                    for target_offset, target in enumerate(targets)
                ],
                dim=0,
            )
            decoded = model.decode_target_latent(central).float()
        for target_offset, target in enumerate(targets):
            patch_prediction = decoded[
                target_offset * batch_size : (target_offset + 1) * batch_size
            ]
            (y0, y1), (x0, x1) = target.output_voxel_bounds_yx
            if tuple(patch_prediction.shape) != (
                batch_size,
                model.provider.in_channels,
                int(values.shape[2]),
                patch_height,
                patch_width,
            ):
                raise JEPABackgroundResidualError(
                    "decoded target patch does not match the declared patch geometry"
                )
            background[:, :, :, y0:y1, x0:x1] = patch_prediction
            coverage[:, :, :, y0:y1, x0:x1] += 1

    if bool((coverage != 1).any()):
        minimum = int(coverage.min().item())
        maximum = int(coverage.max().item())
        raise JEPABackgroundResidualError(
            f"full sweep did not cover each voxel exactly once: min={minimum}, max={maximum}"
        )
    if not bool(torch.isfinite(background).all()):
        raise JEPABackgroundResidualError("full background contains nonfinite values")
    return BackgroundSweepOutput(
        background=background,
        coverage_counts=coverage,
        manifest=manifest,
    )


def signed_residual(video: Tensor, background: Tensor) -> Tensor:
    """Return the signed pixel residual ``video - predicted_background``."""

    if not isinstance(video, Tensor) or not isinstance(background, Tensor):
        raise JEPABackgroundResidualError("video and background must be tensors")
    if video.ndim != 5 or video.shape != background.shape:
        raise JEPABackgroundResidualError(
            "video and background must share [B,C,T,H,W] shape"
        )
    if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(background).all()):
        raise JEPABackgroundResidualError("video and background must be finite")
    return video.float() - background.float()


def _rms(values: Tensor) -> float:
    return float(torch.sqrt(values.float().square().mean()).cpu())


def _robust_scale(values: Tensor) -> float:
    flattened = values.float().reshape(-1)
    median = flattened.median()
    return float((flattened - median).abs().median().cpu())


def _safe_ratio(numerator: float, denominator: float, *, name: str) -> float:
    if denominator <= torch.finfo(torch.float32).eps:
        raise JEPABackgroundResidualError(f"{name} denominator is numerically zero")
    return numerator / denominator


def background_suppression_diagnostics(
    source_off: Tensor,
    predicted_background: Tensor,
) -> dict[str, Any]:
    """Measure source-off temporal background suppression after centering."""

    residual = signed_residual(source_off, predicted_background)
    centered_input = source_off.float() - source_off.float().median(
        dim=2, keepdim=True
    ).values
    centered_residual = residual - residual.median(dim=2, keepdim=True).values
    input_rms = _rms(centered_input)
    residual_rms = _rms(centered_residual)
    input_robust = _robust_scale(centered_input)
    residual_robust = _robust_scale(centered_residual)
    return {
        "source_off_centering": "per_pixel_temporal_median",
        "centered_input_rms": input_rms,
        "centered_residual_rms": residual_rms,
        "background_suppression_fraction_rms": 1.0
        - _safe_ratio(residual_rms, input_rms, name="RMS suppression"),
        "centered_input_median_absolute_deviation": input_robust,
        "centered_residual_median_absolute_deviation": residual_robust,
        "background_suppression_fraction_robust": 1.0
        - _safe_ratio(residual_robust, input_robust, name="robust suppression"),
    }


def patch_seam_diagnostics(
    values: Tensor,
    patch_size: Sequence[int],
) -> dict[str, Any]:
    """Compare jumps at decoder-patch boundaries with within-patch jumps."""

    patch = _shape3(patch_size, name="patch_size")
    if values.ndim != 5 or not bool(torch.isfinite(values).all()):
        raise JEPABackgroundResidualError("values must be finite [B,C,T,H,W]")
    height, width = map(int, values.shape[-2:])
    if height % patch[1] or width % patch[2]:
        raise JEPABackgroundResidualError("values dimensions must align to patch_size")
    differences_y = torch.diff(values.float(), dim=-2).abs()
    differences_x = torch.diff(values.float(), dim=-1).abs()
    boundary_y = torch.zeros(height - 1, dtype=torch.bool, device=values.device)
    boundary_x = torch.zeros(width - 1, dtype=torch.bool, device=values.device)
    if height > patch[1]:
        boundary_y[patch[1] - 1 :: patch[1]] = True
    if width > patch[2]:
        boundary_x[patch[2] - 1 :: patch[2]] = True
    seam_parts: list[Tensor] = []
    interior_parts: list[Tensor] = []
    if bool(boundary_y.any()):
        seam_parts.append(differences_y[..., boundary_y, :].reshape(-1))
    if bool(boundary_x.any()):
        seam_parts.append(differences_x[..., boundary_x].reshape(-1))
    if bool((~boundary_y).any()):
        interior_parts.append(differences_y[..., ~boundary_y, :].reshape(-1))
    if bool((~boundary_x).any()):
        interior_parts.append(differences_x[..., ~boundary_x].reshape(-1))
    if not seam_parts or not interior_parts:
        raise JEPABackgroundResidualError(
            "seam diagnostics require at least 2x2 spatial decoder patches"
        )
    seam = torch.cat(seam_parts)
    interior = torch.cat(interior_parts)
    seam_mean = float(seam.mean().cpu())
    interior_mean = float(interior.mean().cpu())
    ratio = seam_mean / max(interior_mean, float(torch.finfo(torch.float32).eps))
    return {
        "patch_size_tyx": list(patch),
        "mean_absolute_boundary_jump": seam_mean,
        "mean_absolute_within_patch_jump": interior_mean,
        "boundary_to_within_jump_ratio": ratio,
        "boundary_difference_count": int(seam.numel()),
        "within_patch_difference_count": int(interior.numel()),
    }


def paired_residual_diagnostics(
    source_off: Tensor,
    source_on: Tensor,
    predicted_background_off: Tensor,
    predicted_background_on: Tensor,
    *,
    injected_signal: Tensor | None = None,
    patch_size: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Exact paired retention, absorption, closure, suppression, and seams."""

    tensors = (source_off, source_on, predicted_background_off, predicted_background_on)
    if any(not isinstance(value, Tensor) or value.ndim != 5 for value in tensors):
        raise JEPABackgroundResidualError("paired inputs must be BCTHW tensors")
    shape = source_off.shape
    if any(value.shape != shape for value in tensors[1:]):
        raise JEPABackgroundResidualError("all paired inputs must share one shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise JEPABackgroundResidualError("paired inputs must be finite")
    residual_off = signed_residual(source_off, predicted_background_off)
    residual_on = signed_residual(source_on, predicted_background_on)
    observed_delta = source_on.float() - source_off.float()
    signal = observed_delta if injected_signal is None else injected_signal.float()
    if signal.shape != shape or not bool(torch.isfinite(signal).all()):
        raise JEPABackgroundResidualError("injected_signal must be finite and match BCTHW")
    signal_energy = float(torch.sum(signal.square()).cpu())
    if signal_energy <= torch.finfo(torch.float32).eps:
        raise JEPABackgroundResidualError("injected_signal must have nonzero energy")
    prediction_delta = predicted_background_on.float() - predicted_background_off.float()
    residual_delta = residual_on - residual_off
    pair_closure = observed_delta - prediction_delta - residual_delta
    injection_additivity_error = observed_delta - signal
    alpha_retained = float(torch.sum(residual_delta * signal).cpu()) / signal_energy
    alpha_absorbed = float(torch.sum(prediction_delta * signal).cpu()) / signal_energy
    total_signal_error = _rms(residual_delta - signal)
    orthogonal_distortion = _rms(residual_delta - alpha_retained * signal)
    signal_rms = _rms(signal)
    result: dict[str, Any] = {
        "schema_version": "paired_pixel_residual_diagnostics_v1",
        "definitions": {
            "signed_residual": "observation_minus_predicted_background",
            "delta_residual": "source_on_residual_minus_source_off_residual",
            "delta_prediction": "source_on_prediction_minus_source_off_prediction",
        },
        "signal": {
            "injected_energy": signal_energy,
            "retained_projection_alpha": alpha_retained,
            "prediction_absorption_alpha": alpha_absorbed,
            "projection_sum": alpha_retained + alpha_absorbed,
            "projection_sum_error_from_one": alpha_retained + alpha_absorbed - 1.0,
            "total_signal_error_ratio": _safe_ratio(
                total_signal_error,
                signal_rms,
                name="total signal error",
            ),
            "orthogonal_distortion_ratio": _safe_ratio(
                orthogonal_distortion,
                signal_rms,
                name="orthogonal signal distortion",
            ),
        },
        "closure": {
            "pair_closure_max_abs": float(pair_closure.abs().max().cpu()),
            "pair_closure_rms": _rms(pair_closure),
            "observed_delta_minus_injected_signal_max_abs": float(
                injection_additivity_error.abs().max().cpu()
            ),
        },
        "background": background_suppression_diagnostics(
            source_off,
            predicted_background_off,
        ),
    }
    if patch_size is not None:
        result["seams"] = {
            "predicted_background_off": patch_seam_diagnostics(
                predicted_background_off, patch_size
            ),
            "predicted_background_on": patch_seam_diagnostics(
                predicted_background_on, patch_size
            ),
            "signed_residual_off": patch_seam_diagnostics(residual_off, patch_size),
            "signed_residual_on": patch_seam_diagnostics(residual_on, patch_size),
        }
    return result


@torch.no_grad()
def target_tube_invariance_preflight(
    model: FrozenLatentPixelBackgroundModel,
    video: Tensor,
    *,
    target_yx: tuple[int, int] = (0, 0),
    adversarial_delta: float = 10_000.0,
) -> dict[str, Any]:
    """Verify that changing the complete hidden halo cannot change its prediction."""

    delta = _finite_float(adversarial_delta, name="adversarial_delta")
    if delta == 0.0:
        raise JEPABackgroundResidualError("adversarial_delta must be nonzero")
    values = _validated_video(
        video,
        patch_size=model.patch_size,
        in_channels=model.provider.in_channels,
    )
    target_y, target_x = map(int, target_yx)
    _, patch_height, patch_width = model.patch_size
    height, width = map(int, values.shape[-2:])
    halo_y0 = max(0, (target_y - 1) * patch_height)
    halo_y1 = min(height, (target_y + 2) * patch_height)
    halo_x0 = max(0, (target_x - 1) * patch_width)
    halo_x1 = min(width, (target_x + 2) * patch_width)
    changed = values.clone()
    changed[
        :,
        :,
        :,
        halo_y0:halo_y1,
        halo_x0:halo_x1,
    ] += delta
    original_prediction = model.predict_target_patch(values, target_yx)
    changed_prediction = model.predict_target_patch(changed, target_yx)
    maximum_change = float((changed_prediction - original_prediction).abs().max().cpu())

    manifest = blind_halo_manifest(values, model.patch_size)
    target = manifest.targets[target_y * manifest.original_grid_tyx[2] + target_x]
    mask = _target_mask(
        manifest,
        target,
        batch_size=int(values.shape[0]),
        device=values.device,
    )
    original_masked = apply_patch_mask(
        _premask_original_halo_then_reflect(
            values,
            manifest,
            target,
            mask_value=model.provider.mask_value,
        ),
        mask,
        patch_size=model.patch_size,
        mask_value=model.provider.mask_value,
    )
    changed_masked = apply_patch_mask(
        _premask_original_halo_then_reflect(
            changed,
            manifest,
            target,
            mask_value=model.provider.mask_value,
        ),
        mask,
        patch_size=model.patch_size,
        mask_value=model.provider.mask_value,
    )
    context_identical = bool(torch.equal(original_masked, changed_masked))
    return {
        "schema_version": "target_tube_invariance_preflight_v1",
        "target_token_yx": [target_y, target_x],
        "adversarial_original_halo_voxel_bounds_yx_half_open": [
            [halo_y0, halo_y1],
            [halo_x0, halo_x1],
        ],
        "adversarial_scope": "full_time_intersection_of_3x3_token_halo_with_original_video",
        "adversarial_delta": delta,
        "masked_provider_inputs_exactly_identical": context_identical,
        "predicted_target_max_abs_change": maximum_change,
        "exact_invariance": bool(context_identical and maximum_change == 0.0),
        "passed": bool(context_identical and maximum_change == 0.0),
    }


def production_contract_preflight(
    model: FrozenLatentPixelBackgroundModel,
    video: Tensor,
) -> dict[str, Any]:
    """Fail-closed check for the registered 32x64x64 production geometry."""

    values = _validated_video(
        video,
        patch_size=model.patch_size,
        in_channels=model.provider.in_channels,
    )
    manifest = blind_halo_manifest(values, model.patch_size)
    checks = {
        "patch_size_is_4x8x8": tuple(model.patch_size) == PRODUCTION_PATCH_SIZE_TYX,
        "spatial_grid_is_8x8": tuple(manifest.original_grid_tyx[1:]) == PRODUCTION_GRID_YX,
        "latent_dim_is_64": model.provider.latent_dim == PRODUCTION_LATENT_DIM,
        "single_input_output_channel": model.provider.in_channels == 1,
        "decoder_parameter_count_is_16385": (
            model.decoder_parameter_count == PRODUCTION_DECODER_PARAMETERS
        ),
        "decoder_is_only_trainable_module": (
            model.trainable_parameter_count == model.decoder_parameter_count
        ),
        "provider_is_frozen": _frozen(model.provider),
        "target_count_is_64": manifest.target_count == 64,
        "reflect_padding_is_one_spatial_token": (
            manifest.reflect_padding_voxels_yx == model.patch_size[1:]
        ),
        "blind_halo_is_full_time_3x3": (
            manifest.blind_halo_tokens_tyx
            == (manifest.original_grid_tyx[0], 3, 3)
        ),
    }
    return {
        "schema_version": "jepa_pixel_background_production_preflight_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "provider_method_id": model.provider.method_id,
        "decoder_parameters": model.decoder_parameter_count,
        "manifest": manifest.as_dict(),
    }
