"""Compact spatiotemporal JEPA components for bounded calcium-video studies.

The module deliberately contains model primitives only.  It does not load data,
choose train/validation recordings, normalize intensities, augment clips, or
interpret learned features as neurons.  In particular, masking is applied to
the input video *before* the context encoder is called.  This makes the most
important anti-leakage invariant directly testable.

Tensors use ``[batch, channel, time, height, width]`` ordering.  Patch masks use
``[batch, patch_time, patch_height, patch_width]`` ordering.
"""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


Shape3D = tuple[int, int, int]


def _shape3d(values: Sequence[int], *, name: str) -> Shape3D:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly (time, height, width).")
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} values must all be positive; received {result}.")
    return result  # type: ignore[return-value]


def _validate_seed(seed: int) -> int:
    value = int(seed)
    if value < 0 or value >= 2**63:
        raise ValueError("seed must be in the inclusive range [0, 2**63 - 1].")
    return value


def set_deterministic_seed(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Set explicit process-wide RNG state for a reproducible training run.

    Model constructors below isolate their initialization RNG and therefore do
    not require this function.  A runner can call it when deterministic data
    loading and optimization are also desired.  The global side effect is
    intentional and is only made by this explicit function.
    """

    resolved = _validate_seed(seed)
    random.seed(resolved)
    np.random.seed(resolved % 2**32)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture for the shared compact 3-D tubelet encoder.

    The default encoder has approximately 1.50 million parameters.  A JEPA
    target encoder is an EMA copy and is not counted as trainable capacity.
    """

    in_channels: int = 1
    patch_size: Shape3D = (4, 8, 8)
    embed_dim: int = 96
    latent_dim: int = 64
    depth: int = 3
    norm_groups: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "patch_size", _shape3d(self.patch_size, name="patch_size")
        )
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.embed_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("embed_dim and latent_dim must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.norm_groups <= 0:
            raise ValueError("norm_groups must be positive.")
        if self.embed_dim % self.norm_groups:
            raise ValueError("embed_dim must be divisible by norm_groups.")
        if self.latent_dim % self.norm_groups:
            raise ValueError("latent_dim must be divisible by norm_groups.")


@dataclass(frozen=True)
class MaskConfig:
    """Declared contiguous-block masking policy in patch-grid coordinates.

    Defaults encode the prespecified primary ``32 x 64 x 64`` clip contract:
    four non-overlapping ``2 x 4 x 8`` cuboids mask exactly 50% of its
    ``8 x 8 x 8`` patch tokens, and every cuboid spans two temporal tubelets.
    """

    block_shape: Shape3D = (2, 4, 8)
    blocks_per_sample: int = 4
    coverage_range: tuple[float, float] | None = (0.40, 0.55)
    non_overlapping: bool = True
    minimum_temporal_tubelets: int = 2
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_shape", _shape3d(self.block_shape, name="block_shape")
        )
        if self.blocks_per_sample <= 0:
            raise ValueError("blocks_per_sample must be positive.")
        if self.minimum_temporal_tubelets <= 0:
            raise ValueError("minimum_temporal_tubelets must be positive.")
        if self.block_shape[0] < self.minimum_temporal_tubelets:
            raise ValueError(
                "block_shape must span at least minimum_temporal_tubelets in time."
            )
        if self.coverage_range is not None:
            if len(self.coverage_range) != 2:
                raise ValueError("coverage_range must contain (minimum, maximum).")
            lower, upper = map(float, self.coverage_range)
            if not 0.0 < lower <= upper < 1.0:
                raise ValueError(
                    "coverage_range must satisfy 0 < minimum <= maximum < 1."
                )
            object.__setattr__(self, "coverage_range", (lower, upper))
            if not self.non_overlapping:
                raise ValueError(
                    "coverage-bounded masks require non_overlapping=True."
                )
        if not np.isfinite(self.mask_value):
            raise ValueError("mask_value must be finite.")


def tiny_smoke_configuration() -> tuple[EncoderConfig, MaskConfig]:
    """Return a fast CPU-only architecture for contract and runner smoke tests.

    This configuration is not a scientific training default and should never
    be substituted for a registered experiment configuration implicitly.
    """

    return (
        EncoderConfig(
            patch_size=(2, 2, 2),
            embed_dim=8,
            latent_dim=4,
            depth=1,
            norm_groups=4,
        ),
        MaskConfig(
            block_shape=(1, 1, 1),
            blocks_per_sample=1,
            coverage_range=None,
            minimum_temporal_tubelets=1,
        ),
    )


def patch_grid_shape(video: Tensor, patch_size: Sequence[int]) -> Shape3D:
    """Return the patch-grid shape after validating a video tensor."""

    if video.ndim != 5:
        raise ValueError(
            "video must have shape [batch, channel, time, height, width]."
        )
    patch = _shape3d(patch_size, name="patch_size")
    dimensions = tuple(int(value) for value in video.shape[-3:])
    if any(dimension % step for dimension, step in zip(dimensions, patch)):
        raise ValueError(
            f"video dimensions {dimensions} must be divisible by patch_size {patch}."
        )
    return tuple(
        dimension // step for dimension, step in zip(dimensions, patch)
    )  # type: ignore[return-value]


def make_contiguous_patch_mask(
    *,
    batch_size: int,
    grid_shape: Sequence[int],
    block_shape: Sequence[int],
    blocks_per_sample: int = 1,
    seed: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create deterministic masks made from contiguous 3-D cuboids.

    Random origins are sampled on CPU from a private generator, so mask
    generation does not consume global training RNG state.  Each individual
    block is contiguous; callers requesting more than one block should record
    that choice because the union need not itself be one connected component.
    """

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    grid = _shape3d(grid_shape, name="grid_shape")
    block = _shape3d(block_shape, name="block_shape")
    blocks_per_sample = int(blocks_per_sample)
    if blocks_per_sample <= 0:
        raise ValueError("blocks_per_sample must be positive.")
    if any(block_value > grid_value for block_value, grid_value in zip(block, grid)):
        raise ValueError(
            f"block_shape {block} must fit within patch grid {grid}."
        )
    if block == grid:
        raise ValueError("block_shape cannot mask the entire context grid.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_validate_seed(seed))
    mask = torch.zeros((batch_size, *grid), dtype=torch.bool, device="cpu")
    for batch_index in range(batch_size):
        for _ in range(blocks_per_sample):
            origins = [
                int(
                    torch.randint(
                        low=0,
                        high=grid_value - block_value + 1,
                        size=(1,),
                        generator=generator,
                    ).item()
                )
                for grid_value, block_value in zip(grid, block)
            ]
            time, row, column = origins
            block_time, block_height, block_width = block
            mask[
                batch_index,
                time : time + block_time,
                row : row + block_height,
                column : column + block_width,
            ] = True
        if bool(mask[batch_index].all()):
            raise ValueError(
                "The requested blocks masked the complete context grid; "
                "reduce blocks_per_sample or block_shape."
            )
    return mask.to(device=device)


def make_coverage_bounded_patch_mask(
    *,
    batch_size: int,
    grid_shape: Sequence[int],
    block_shape: Sequence[int],
    blocks_per_sample: int,
    coverage_range: tuple[float, float],
    minimum_temporal_tubelets: int,
    seed: int,
    device: torch.device | str | None = None,
    maximum_placement_attempts: int = 256,
) -> Tensor:
    """Create deterministic, non-overlapping cuboids at bounded coverage.

    Coverage is checked before placement from the exact cuboid volume and then
    checked again on every returned sample.  Placement uses private seeded RNG
    and bounded randomized greedy restarts; failure is explicit rather than
    silently accepting overlap or a different masking fraction.
    """

    batch_size = int(batch_size)
    blocks_per_sample = int(blocks_per_sample)
    minimum_temporal_tubelets = int(minimum_temporal_tubelets)
    maximum_placement_attempts = int(maximum_placement_attempts)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if blocks_per_sample <= 0:
        raise ValueError("blocks_per_sample must be positive.")
    if minimum_temporal_tubelets <= 0:
        raise ValueError("minimum_temporal_tubelets must be positive.")
    if maximum_placement_attempts <= 0:
        raise ValueError("maximum_placement_attempts must be positive.")
    grid = _shape3d(grid_shape, name="grid_shape")
    block = _shape3d(block_shape, name="block_shape")
    if any(block_value > grid_value for block_value, grid_value in zip(block, grid)):
        raise ValueError(
            f"block_shape {block} must fit within patch grid {grid}."
        )
    if block[0] < minimum_temporal_tubelets:
        raise ValueError(
            f"Each block must span at least {minimum_temporal_tubelets} "
            "temporal tubelets."
        )
    if len(coverage_range) != 2:
        raise ValueError("coverage_range must contain (minimum, maximum).")
    lower, upper = map(float, coverage_range)
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError("coverage_range must satisfy 0 < minimum <= maximum < 1.")
    block_volume = int(np.prod(block))
    grid_volume = int(np.prod(grid))
    expected_masked_tokens = blocks_per_sample * block_volume
    expected_coverage = expected_masked_tokens / grid_volume
    tolerance = 8 * np.finfo(float).eps
    if expected_coverage < lower - tolerance or expected_coverage > upper + tolerance:
        raise ValueError(
            f"Non-overlapping blocks imply coverage {expected_coverage:.6f}, "
            f"outside declared range [{lower:.6f}, {upper:.6f}]."
        )

    possible_origins = list(
        itertools.product(
            range(grid[0] - block[0] + 1),
            range(grid[1] - block[1] + 1),
            range(grid[2] - block[2] + 1),
        )
    )
    if len(possible_origins) < blocks_per_sample:
        raise ValueError("Too few distinct cuboid origins for blocks_per_sample.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_validate_seed(seed))
    masks: list[Tensor] = []
    for sample_index in range(batch_size):
        placed: Tensor | None = None
        for _ in range(maximum_placement_attempts):
            candidate_mask = torch.zeros(grid, dtype=torch.bool)
            block_count = 0
            order = torch.randperm(len(possible_origins), generator=generator)
            for origin_index in order.tolist():
                time, row, column = possible_origins[origin_index]
                block_time, block_height, block_width = block
                region = candidate_mask[
                    time : time + block_time,
                    row : row + block_height,
                    column : column + block_width,
                ]
                if bool(region.any()):
                    continue
                region.fill_(True)
                block_count += 1
                if block_count == blocks_per_sample:
                    placed = candidate_mask
                    break
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError(
                "Could not place the declared non-overlapping cuboids after "
                f"{maximum_placement_attempts} attempts for sample {sample_index}."
            )
        observed_tokens = int(placed.sum().item())
        if observed_tokens != expected_masked_tokens:
            raise RuntimeError(
                "Internal mask-placement error: overlap changed masked-token count."
            )
        observed_coverage = observed_tokens / grid_volume
        if not lower - tolerance <= observed_coverage <= upper + tolerance:
            raise RuntimeError("Internal mask-placement error: coverage gate failed.")
        masks.append(placed)
    return torch.stack(masks, dim=0).to(device=device)


def make_patch_mask_from_config(
    *,
    batch_size: int,
    grid_shape: Sequence[int],
    config: MaskConfig,
    seed: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Resolve a declared masking policy without hidden RNG or overlap changes."""

    grid = _shape3d(grid_shape, name="grid_shape")
    if config.coverage_range is not None or config.non_overlapping:
        coverage_range = config.coverage_range
        if coverage_range is None:
            exact_coverage = (
                config.blocks_per_sample * int(np.prod(config.block_shape))
            ) / int(np.prod(grid))
            if not 0.0 < exact_coverage < 1.0:
                raise ValueError(
                    "Non-overlapping mask configuration must leave visible context."
                )
            coverage_range = (exact_coverage, exact_coverage)
        return make_coverage_bounded_patch_mask(
            batch_size=batch_size,
            grid_shape=grid,
            block_shape=config.block_shape,
            blocks_per_sample=config.blocks_per_sample,
            coverage_range=coverage_range,
            minimum_temporal_tubelets=config.minimum_temporal_tubelets,
            seed=seed,
            device=device,
        )
    return make_contiguous_patch_mask(
        batch_size=batch_size,
        grid_shape=grid,
        block_shape=config.block_shape,
        blocks_per_sample=config.blocks_per_sample,
        seed=seed,
        device=device,
    )


def _validated_patch_mask(
    patch_mask: Tensor,
    *,
    batch_size: int,
    grid_shape: Shape3D,
    device: torch.device,
) -> Tensor:
    expected = (batch_size, *grid_shape)
    if tuple(patch_mask.shape) != expected:
        raise ValueError(
            f"patch_mask must have shape {expected}; received {tuple(patch_mask.shape)}."
        )
    mask = patch_mask.to(device=device, dtype=torch.bool)
    counts = mask.reshape(batch_size, -1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every sample must contain at least one masked patch.")
    if bool((counts == mask[0].numel()).any()):
        raise ValueError("Every sample must retain at least one visible context patch.")
    return mask


def patch_mask_to_voxels(patch_mask: Tensor, patch_size: Sequence[int]) -> Tensor:
    """Expand ``[B, Tg, Hg, Wg]`` patch masks to voxel masks."""

    if patch_mask.ndim != 4:
        raise ValueError("patch_mask must have shape [batch, time, height, width].")
    patch_time, patch_height, patch_width = _shape3d(
        patch_size, name="patch_size"
    )
    return (
        patch_mask.to(dtype=torch.bool)
        .repeat_interleave(patch_time, dim=1)
        .repeat_interleave(patch_height, dim=2)
        .repeat_interleave(patch_width, dim=3)
    )


def apply_patch_mask(
    video: Tensor,
    patch_mask: Tensor,
    *,
    patch_size: Sequence[int],
    mask_value: float = 0.0,
) -> Tensor:
    """Apply an aligned patch mask without modifying or normalizing the input."""

    grid = patch_grid_shape(video, patch_size)
    mask = _validated_patch_mask(
        patch_mask,
        batch_size=int(video.shape[0]),
        grid_shape=grid,
        device=video.device,
    )
    voxel_mask = patch_mask_to_voxels(mask, patch_size).unsqueeze(1)
    return video.masked_fill(voxel_mask, float(mask_value))


class ResidualConv3DBlock(nn.Module):
    """A compact residual block that preserves the patch grid."""

    def __init__(self, channels: int, norm_groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.conv1(F.gelu(self.norm1(values)))
        values = self.conv2(F.gelu(self.norm2(values)))
        return residual + values


class TubeletEncoder(nn.Module):
    """Dense 3-D encoder producing one latent vector per tubelet patch."""

    def __init__(self, config: EncoderConfig = EncoderConfig()) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.patch_norm = nn.GroupNorm(config.norm_groups, config.embed_dim)
        self.blocks = nn.Sequential(
            *[
                ResidualConv3DBlock(config.embed_dim, config.norm_groups)
                for _ in range(config.depth)
            ]
        )
        self.output_norm = nn.GroupNorm(config.norm_groups, config.embed_dim)
        self.output_projection = nn.Conv3d(
            config.embed_dim, config.latent_dim, kernel_size=1, bias=False
        )

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(
                "video must have shape [batch, channel, time, height, width]."
            )
        if int(video.shape[1]) != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} channels; received {video.shape[1]}."
            )
        patch_grid_shape(video, self.config.patch_size)
        values = F.gelu(self.patch_norm(self.patch_embed(video)))
        values = self.blocks(values)
        return self.output_projection(F.gelu(self.output_norm(values)))


class LatentPredictor(nn.Module):
    """Local-context predictor shared by JEPA and pixel-MAE controls."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(config.norm_groups, config.latent_dim)
        self.context_conv = nn.Conv3d(
            config.latent_dim,
            config.embed_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(config.norm_groups, config.embed_dim)
        self.output = nn.Conv3d(
            config.embed_dim, config.latent_dim, kernel_size=1, bias=False
        )

    def forward(self, context: Tensor) -> Tensor:
        values = self.context_conv(F.gelu(self.norm1(context)))
        values = self.output(F.gelu(self.norm2(values)))
        return context + values


def _construct_with_seed(factory: Any, seed: int) -> Any:
    """Construct CPU modules deterministically without consuming global RNG."""

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(_validate_seed(seed))
        return factory()


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count model parameters for capacity-matching checks."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def jepa_latent_loss(
    prediction: Tensor,
    target: Tensor,
    patch_mask: Tensor,
    *,
    normalize_latents: bool = False,
) -> Tensor:
    """Mean squared latent error on masked patch positions only."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "prediction and target must share [batch, latent, time, height, width]."
        )
    mask = _validated_patch_mask(
        patch_mask,
        batch_size=int(prediction.shape[0]),
        grid_shape=tuple(int(value) for value in prediction.shape[-3:]),
        device=prediction.device,
    )
    predicted_vectors = prediction.permute(0, 2, 3, 4, 1)[mask]
    target_vectors = target.detach().permute(0, 2, 3, 4, 1)[mask]
    if normalize_latents:
        predicted_vectors = F.layer_norm(
            predicted_vectors, (predicted_vectors.shape[-1],)
        )
        target_vectors = F.layer_norm(target_vectors, (target_vectors.shape[-1],))
    return F.mse_loss(predicted_vectors, target_vectors)


def jepa_prediction_error_map(
    prediction: Tensor,
    target: Tensor,
    *,
    normalize_latents: bool = False,
) -> Tensor:
    """Return dense per-patch latent prediction error ``[B, Tg, Hg, Wg]``.

    The map is an unsupervised prediction-error diagnostic, not a calibrated
    neuron-probability map.  A runner must retain this distinction in metadata.
    """

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "prediction and target must share [batch, latent, time, height, width]."
        )
    predicted_vectors = prediction.permute(0, 2, 3, 4, 1)
    target_vectors = target.detach().permute(0, 2, 3, 4, 1)
    if normalize_latents:
        predicted_vectors = F.layer_norm(
            predicted_vectors, (predicted_vectors.shape[-1],)
        )
        target_vectors = F.layer_norm(target_vectors, (target_vectors.shape[-1],))
    return (predicted_vectors - target_vectors).square().mean(dim=-1)


@dataclass
class JEPAOutput:
    loss: Tensor
    prediction: Tensor
    target: Tensor
    context: Tensor
    patch_mask: Tensor
    normalize_latents: bool

    def prediction_error_map(
        self, *, normalize_latents: bool | None = None
    ) -> Tensor:
        """Return the dense latent error map without changing mask semantics."""

        resolved = self.normalize_latents if normalize_latents is None else bool(
            normalize_latents
        )
        return jepa_prediction_error_map(
            self.prediction,
            self.target,
            normalize_latents=resolved,
        )


@dataclass
class MaskSweepOutput:
    """Fully covered masked-token error map and its observation counts."""

    prediction_error: Tensor
    observation_counts: Tensor

    @property
    def minimum_observations(self) -> int:
        return int(self.observation_counts.min().item())


class SpatiotemporalJEPA(nn.Module):
    """Compact JEPA with an EMA target encoder and explicit stop-gradient."""

    def __init__(
        self,
        encoder_config: EncoderConfig = EncoderConfig(),
        mask_config: MaskConfig = MaskConfig(),
        *,
        initialization_seed: int = 0,
        ema_decay: float = 0.996,
        normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(ema_decay) <= 1.0:
            raise ValueError("ema_decay must be in [0, 1].")
        self.encoder_config = encoder_config
        self.mask_config = mask_config
        self.ema_decay = float(ema_decay)
        self.normalize_latents = bool(normalize_latents)

        def build_trainable() -> tuple[TubeletEncoder, LatentPredictor]:
            return TubeletEncoder(encoder_config), LatentPredictor(encoder_config)

        self.context_encoder, self.predictor = _construct_with_seed(
            build_trainable, initialization_seed
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "SpatiotemporalJEPA":
        super().train(mode)
        # The teacher never switches into a train-time stochastic mode.
        self.target_encoder.eval()
        return self

    def _resolve_mask(
        self,
        video: Tensor,
        *,
        patch_mask: Tensor | None,
        mask_seed: int | None,
    ) -> Tensor:
        grid = patch_grid_shape(video, self.encoder_config.patch_size)
        if patch_mask is None:
            if mask_seed is None:
                raise ValueError(
                    "Provide patch_mask or an explicit mask_seed; masks are never "
                    "advanced from hidden mutable RNG state."
                )
            patch_mask = make_patch_mask_from_config(
                batch_size=int(video.shape[0]),
                grid_shape=grid,
                config=self.mask_config,
                seed=mask_seed,
                device=video.device,
            )
        return _validated_patch_mask(
            patch_mask,
            batch_size=int(video.shape[0]),
            grid_shape=grid,
            device=video.device,
        )

    def forward(
        self,
        video: Tensor,
        *,
        patch_mask: Tensor | None = None,
        mask_seed: int | None = None,
    ) -> JEPAOutput:
        mask = self._resolve_mask(
            video, patch_mask=patch_mask, mask_seed=mask_seed
        )
        # This ordering is a scientific invariant: the context encoder never
        # receives target pixels from masked tubelets.
        masked_video = apply_patch_mask(
            video,
            mask,
            patch_size=self.encoder_config.patch_size,
            mask_value=self.mask_config.mask_value,
        )
        context = self.context_encoder(masked_video)
        prediction = self.predictor(context)
        with torch.no_grad():
            target = self.target_encoder(video).detach()
        loss = jepa_latent_loss(
            prediction,
            target,
            mask,
            normalize_latents=self.normalize_latents,
        )
        return JEPAOutput(
            loss=loss,
            prediction=prediction,
            target=target,
            context=context,
            patch_mask=mask,
            normalize_latents=self.normalize_latents,
        )

    def encode_context(self, video: Tensor, *, detach: bool = True) -> Tensor:
        """Encode an unmasked clip with the learned online/context branch.

        No normalization or augmentation is applied.  ``detach=True`` is the
        safe default for frozen-probe evaluation; training code can request a
        differentiable result explicitly.
        """

        embeddings = self.context_encoder(video)
        return embeddings.detach() if detach else embeddings

    @torch.no_grad()
    def encode_target(self, video: Tensor) -> Tensor:
        """Encode an unmasked clip with the EMA target branch."""

        return self.target_encoder(video).detach()

    @torch.no_grad()
    def masked_prediction_error_sweep(
        self,
        video: Tensor,
        patch_masks: Tensor | Sequence[Tensor],
    ) -> MaskSweepOutput:
        """Average error only when a token is masked, requiring full coverage.

        A single JEPA forward produces predictions at every patch coordinate,
        but unmasked coordinates have direct access to their input and are not
        valid masked-prediction scores.  This method accepts a declared mask
        schedule, evaluates the shared target once, accumulates masked errors,
        and fails closed unless every patch is observed at least once.

        A tensor schedule has shape ``[draw, batch, Tg, Hg, Wg]``.  A sequence
        may contain individual masks with shape ``[batch, Tg, Hg, Wg]``.
        """

        grid = patch_grid_shape(video, self.encoder_config.patch_size)
        if isinstance(patch_masks, Tensor):
            if patch_masks.ndim == 4:
                schedule = [patch_masks]
            elif patch_masks.ndim == 5:
                schedule = list(patch_masks.unbind(dim=0))
            else:
                raise ValueError(
                    "patch_masks tensor must have [batch,Tg,Hg,Wg] or "
                    "[draw,batch,Tg,Hg,Wg] shape."
                )
        else:
            schedule = list(patch_masks)
        if not schedule:
            raise ValueError("patch-mask sweep cannot be empty.")

        target = self.target_encoder(video).detach()
        error_sum = torch.zeros(
            (int(video.shape[0]), *grid), dtype=target.dtype, device=video.device
        )
        observation_counts = torch.zeros_like(error_sum, dtype=torch.int64)
        for proposed_mask in schedule:
            mask = _validated_patch_mask(
                proposed_mask,
                batch_size=int(video.shape[0]),
                grid_shape=grid,
                device=video.device,
            )
            masked_video = apply_patch_mask(
                video,
                mask,
                patch_size=self.encoder_config.patch_size,
                mask_value=self.mask_config.mask_value,
            )
            prediction = self.predictor(self.context_encoder(masked_video))
            error = jepa_prediction_error_map(
                prediction,
                target,
                normalize_latents=self.normalize_latents,
            )
            error_sum.add_(error * mask.to(dtype=error.dtype))
            observation_counts.add_(mask)
        if bool((observation_counts == 0).any()):
            uncovered = int((observation_counts == 0).sum().item())
            raise ValueError(
                f"patch-mask sweep left {uncovered} patch tokens unobserved."
            )
        prediction_error = error_sum / observation_counts.to(dtype=error_sum.dtype)
        return MaskSweepOutput(
            prediction_error=prediction_error,
            observation_counts=observation_counts,
        )

    @torch.no_grad()
    def update_target_encoder(self, decay: float | None = None) -> None:
        """EMA-update the stop-gradient target after an optimizer step."""

        resolved_decay = self.ema_decay if decay is None else float(decay)
        if not 0.0 <= resolved_decay <= 1.0:
            raise ValueError("EMA decay must be in [0, 1].")
        context_parameters = dict(self.context_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            source = context_parameters[name]
            target_parameter.mul_(resolved_decay).add_(
                source, alpha=1.0 - resolved_decay
            )
        context_buffers = dict(self.context_encoder.named_buffers())
        for name, target_buffer in self.target_encoder.named_buffers():
            target_buffer.copy_(context_buffers[name])


@dataclass
class MaskedPixelOutput:
    loss: Tensor
    reconstruction: Tensor
    patch_mask: Tensor


class MaskedPixelAutoencoder(nn.Module):
    """Capacity-matched masked-pixel reconstruction control.

    It shares the exact encoder and latent predictor architecture used by the
    JEPA trainable path.  The small transposed-convolution pixel head adds only
    one patch worth of weights per latent channel.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig = EncoderConfig(),
        mask_config: MaskConfig = MaskConfig(),
        *,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.mask_config = mask_config

        def build() -> tuple[TubeletEncoder, LatentPredictor, nn.ConvTranspose3d]:
            return (
                TubeletEncoder(encoder_config),
                LatentPredictor(encoder_config),
                nn.ConvTranspose3d(
                    encoder_config.latent_dim,
                    encoder_config.in_channels,
                    kernel_size=encoder_config.patch_size,
                    stride=encoder_config.patch_size,
                    bias=True,
                ),
            )

        self.encoder, self.predictor, self.pixel_head = _construct_with_seed(
            build, initialization_seed
        )

    def forward(
        self,
        video: Tensor,
        *,
        patch_mask: Tensor | None = None,
        mask_seed: int | None = None,
    ) -> MaskedPixelOutput:
        grid = patch_grid_shape(video, self.encoder_config.patch_size)
        if patch_mask is None:
            if mask_seed is None:
                raise ValueError(
                    "Provide patch_mask or an explicit mask_seed; masks are never "
                    "advanced from hidden mutable RNG state."
                )
            patch_mask = make_patch_mask_from_config(
                batch_size=int(video.shape[0]),
                grid_shape=grid,
                config=self.mask_config,
                seed=mask_seed,
                device=video.device,
            )
        mask = _validated_patch_mask(
            patch_mask,
            batch_size=int(video.shape[0]),
            grid_shape=grid,
            device=video.device,
        )
        masked_video = apply_patch_mask(
            video,
            mask,
            patch_size=self.encoder_config.patch_size,
            mask_value=self.mask_config.mask_value,
        )
        latent = self.predictor(self.encoder(masked_video))
        reconstruction = self.pixel_head(latent)
        voxel_mask = patch_mask_to_voxels(mask, self.encoder_config.patch_size)
        squared_error = (reconstruction - video).square()
        selected = squared_error.masked_select(voxel_mask.unsqueeze(1))
        loss = selected.mean()
        return MaskedPixelOutput(
            loss=loss, reconstruction=reconstruction, patch_mask=mask
        )


class FrozenRandomEncoderBaseline(nn.Module):
    """Deterministic, randomly initialized frozen-encoder negative control."""

    def __init__(
        self,
        encoder_config: EncoderConfig = EncoderConfig(),
        *,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.encoder = _construct_with_seed(
            lambda: TubeletEncoder(encoder_config), initialization_seed
        )
        self.encoder.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenRandomEncoderBaseline":
        # A frozen baseline remains frozen/evaluation-only even if a parent
        # training harness recursively calls train().
        super().train(False)
        return self

    def forward(
        self,
        video: Tensor,
        *,
        patch_mask: Tensor | None = None,
        mask_value: float = 0.0,
    ) -> Tensor:
        values = video
        if patch_mask is not None:
            values = apply_patch_mask(
                video,
                patch_mask,
                patch_size=self.encoder_config.patch_size,
                mask_value=mask_value,
            )
        with torch.no_grad():
            return self.encoder(values).detach()


@dataclass(frozen=True)
class RepresentationDiagnostics:
    """JSON-ready representation-collapse and effective-rank diagnostics."""

    sample_count: int
    feature_dimension: int
    finite_fraction: float
    global_std: float
    mean_feature_std: float
    effective_rank: float
    effective_rank_fraction: float
    first_principal_component_variance_fraction: float
    numerical_rank: int
    collapsed: bool
    minimum_mean_feature_std: float
    minimum_effective_rank_fraction: float
    maximum_first_principal_component_variance_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def representation_diagnostics(
    embeddings: Tensor,
    *,
    minimum_mean_feature_std: float = 1e-4,
    minimum_effective_rank_fraction: float = 0.1,
    maximum_first_principal_component_variance_fraction: float = 0.5,
) -> RepresentationDiagnostics:
    """Measure collapse using feature variance and covariance effective rank.

    ``embeddings`` may be ``[samples, features]`` or a dense encoder output
    ``[batch, features, time, height, width]``.  Effective rank is the
    exponentiated Shannon entropy of the non-negative covariance spectrum.
    """

    if minimum_mean_feature_std < 0:
        raise ValueError("minimum_mean_feature_std must be non-negative.")
    if not 0.0 <= minimum_effective_rank_fraction <= 1.0:
        raise ValueError("minimum_effective_rank_fraction must be in [0, 1].")
    if not 0.0 < maximum_first_principal_component_variance_fraction <= 1.0:
        raise ValueError(
            "maximum_first_principal_component_variance_fraction must be in (0, 1]."
        )
    if embeddings.ndim == 2:
        flattened = embeddings
    elif embeddings.ndim == 5:
        flattened = embeddings.permute(0, 2, 3, 4, 1).reshape(
            -1, embeddings.shape[1]
        )
    else:
        raise ValueError(
            "embeddings must have shape [samples, features] or "
            "[batch, features, time, height, width]."
        )
    sample_count, feature_dimension = map(int, flattened.shape)
    if sample_count == 0 or feature_dimension == 0:
        raise ValueError("embeddings must contain at least one sample and feature.")

    finite = torch.isfinite(flattened)
    finite_fraction = float(finite.to(dtype=torch.float32).mean().item())
    finite_rows = finite.all(dim=1)
    valid = flattened[finite_rows].to(dtype=torch.float64)
    if valid.shape[0] < 2:
        global_std = 0.0
        mean_feature_std = 0.0
        effective_rank = 0.0
        first_principal_component_variance_fraction = 1.0
        numerical_rank = 0
    else:
        global_std = float(valid.std(unbiased=False).item())
        feature_std = valid.std(dim=0, unbiased=False)
        mean_feature_std = float(feature_std.mean().item())
        centered = valid - valid.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(int(valid.shape[0]) - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        total = eigenvalues.sum()
        if float(total.item()) <= torch.finfo(eigenvalues.dtype).eps:
            effective_rank = 0.0
            first_principal_component_variance_fraction = 1.0
            numerical_rank = 0
        else:
            probabilities = eigenvalues / total
            first_principal_component_variance_fraction = float(
                probabilities.max().item()
            )
            positive = probabilities > 0
            entropy = -torch.sum(
                probabilities[positive] * torch.log(probabilities[positive])
            )
            effective_rank = float(torch.exp(entropy).item())
            rank_floor = (
                float(eigenvalues.max().item())
                * max(sample_count, feature_dimension)
                * torch.finfo(eigenvalues.dtype).eps
            )
            numerical_rank = int((eigenvalues > rank_floor).sum().item())
    effective_rank_fraction = effective_rank / feature_dimension
    collapsed = bool(
        finite_fraction < 1.0
        or mean_feature_std < minimum_mean_feature_std
        or effective_rank_fraction < minimum_effective_rank_fraction
        or first_principal_component_variance_fraction
        > maximum_first_principal_component_variance_fraction
    )
    return RepresentationDiagnostics(
        sample_count=sample_count,
        feature_dimension=feature_dimension,
        finite_fraction=finite_fraction,
        global_std=global_std,
        mean_feature_std=mean_feature_std,
        effective_rank=effective_rank,
        effective_rank_fraction=effective_rank_fraction,
        first_principal_component_variance_fraction=(
            first_principal_component_variance_fraction
        ),
        numerical_rank=numerical_rank,
        collapsed=collapsed,
        minimum_mean_feature_std=float(minimum_mean_feature_std),
        minimum_effective_rank_fraction=float(minimum_effective_rank_fraction),
        maximum_first_principal_component_variance_fraction=float(
            maximum_first_principal_component_variance_fraction
        ),
    )


__all__ = [
    "EncoderConfig",
    "FrozenRandomEncoderBaseline",
    "JEPAOutput",
    "MaskSweepOutput",
    "MaskConfig",
    "MaskedPixelAutoencoder",
    "MaskedPixelOutput",
    "RepresentationDiagnostics",
    "SpatiotemporalJEPA",
    "TubeletEncoder",
    "apply_patch_mask",
    "count_parameters",
    "jepa_latent_loss",
    "jepa_prediction_error_map",
    "make_contiguous_patch_mask",
    "make_coverage_bounded_patch_mask",
    "make_patch_mask_from_config",
    "patch_grid_shape",
    "patch_mask_to_voxels",
    "representation_diagnostics",
    "set_deterministic_seed",
    "tiny_smoke_configuration",
]
