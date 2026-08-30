"""Leakage-safe conditional background prediction and residualization.

The models in this module predict the *current* center of a calcium-video
patch from two explicitly separated sources of context:

* current-frame pixels in a spatial annulus; and
* pixels from strictly earlier frames inside the declared outer radius.

Current-frame target pixels are masked before either predictor is called.
Consequently, subtracting the returned pixel-space estimate is a structural
operation rather than a latent-space analogy.  Tensors use ``BCTHW`` for input
video and ``BNCYX`` for aligned target-frame outputs, where ``N = T - lags``.

This file contains reusable primitives only.  It does not choose recordings,
split data, train a neural model, rank candidates, or claim that a residual is
neural signal.  Source-off calibration and paired-injection diagnostics remain
explicit so a runner can preserve those scientific boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConditionalBackgroundError(ValueError):
    """Raised when a conditional-background contract cannot be preserved."""


def _finite_positive(value: float, *, name: str, allow_zero: bool = False) -> float:
    resolved = float(value)
    valid = resolved >= 0.0 if allow_zero else resolved > 0.0
    if not math.isfinite(resolved) or not valid:
        relation = "nonnegative" if allow_zero else "positive"
        raise ConditionalBackgroundError(f"{name} must be finite and {relation}")
    return resolved


def _validated_seed(seed: int) -> int:
    resolved = int(seed)
    if resolved < 0 or resolved >= 2**63:
        raise ConditionalBackgroundError("seed must be in [0, 2**63 - 1]")
    return resolved


def _json_float(value: Tensor | np.ndarray | float) -> float:
    resolved = float(np.asarray(value.detach().cpu() if isinstance(value, Tensor) else value))
    if not math.isfinite(resolved):
        raise ConditionalBackgroundError("a JSON summary value was nonfinite")
    return resolved


@dataclass(frozen=True)
class CausalAnnularContextConfig:
    """Geometry and lag contract for a center-background prediction.

    Radii are measured from the geometric center of the input patch.  The
    annulus must be separated from the target disk; past frames may contain the
    center because they are causally available at the prediction time.
    """

    past_frames: int = 4
    target_radius_px: float = 2.5
    annulus_inner_radius_px: float = 4.0
    annulus_outer_radius_px: float = 12.0
    include_geometry_channels: bool = True

    def __post_init__(self) -> None:
        if int(self.past_frames) != self.past_frames or self.past_frames < 1:
            raise ConditionalBackgroundError("past_frames must be a positive integer")
        target = _finite_positive(self.target_radius_px, name="target_radius_px")
        inner = _finite_positive(
            self.annulus_inner_radius_px, name="annulus_inner_radius_px"
        )
        outer = _finite_positive(
            self.annulus_outer_radius_px, name="annulus_outer_radius_px"
        )
        if not target < inner < outer:
            raise ConditionalBackgroundError(
                "radii must satisfy target_radius_px < annulus_inner_radius_px "
                "< annulus_outer_radius_px"
            )

    def context_channels(self, imaging_channels: int) -> int:
        channels = int(imaging_channels)
        if channels < 1:
            raise ConditionalBackgroundError("imaging_channels must be positive")
        geometry = 2 if self.include_geometry_channels else 0
        return channels * (self.past_frames + 1) + geometry

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalAnnularBatch:
    """Leakage-safe model inputs and their aligned center targets."""

    context: Tensor
    target: Tensor
    target_mask: Tensor
    annulus_mask: Tensor
    past_support_mask: Tensor
    target_frame_indices: tuple[int, ...]
    config: CausalAnnularContextConfig
    input_dtype: str

    def __post_init__(self) -> None:
        if self.context.ndim != 5:
            raise ConditionalBackgroundError("context must have shape [B,N,F,H,W]")
        if self.target.ndim != 5:
            raise ConditionalBackgroundError("target must have shape [B,N,C,H,W]")
        if self.context.shape[:2] != self.target.shape[:2]:
            raise ConditionalBackgroundError("context and target sample axes differ")
        if self.context.shape[-2:] != self.target.shape[-2:]:
            raise ConditionalBackgroundError("context and target spatial axes differ")
        spatial_shape = tuple(int(value) for value in self.target.shape[-2:])
        for name, mask in (
            ("target_mask", self.target_mask),
            ("annulus_mask", self.annulus_mask),
            ("past_support_mask", self.past_support_mask),
        ):
            if mask.dtype != torch.bool or tuple(mask.shape) != spatial_shape:
                raise ConditionalBackgroundError(
                    f"{name} must be a boolean mask with shape {spatial_shape}"
                )
        if bool((self.target_mask & self.annulus_mask).any()):
            raise ConditionalBackgroundError("target and current annulus overlap")
        if len(self.target_frame_indices) != int(self.target.shape[1]):
            raise ConditionalBackgroundError("target frame-index count is inconsistent")
        if not bool(torch.isfinite(self.context).all()) or not bool(
            torch.isfinite(self.target).all()
        ):
            raise ConditionalBackgroundError("context and target must be finite")

    @property
    def imaging_channels(self) -> int:
        return int(self.target.shape[2])

    def summary(self) -> dict[str, Any]:
        current_channels = self.context[:, :, : self.imaging_channels]
        leakage = current_channels[..., self.target_mask]
        return {
            "schema_version": "causal_annular_batch_v1",
            "context_shape": [int(value) for value in self.context.shape],
            "target_shape": [int(value) for value in self.target.shape],
            "input_dtype": self.input_dtype,
            "output_dtype": str(self.target.dtype).removeprefix("torch."),
            "target_frame_indices": [int(value) for value in self.target_frame_indices],
            "target_pixel_count": int(self.target_mask.sum().item()),
            "annulus_pixel_count": int(self.annulus_mask.sum().item()),
            "past_support_pixel_count": int(self.past_support_mask.sum().item()),
            "current_center_max_abs": _json_float(leakage.abs().max()),
            "current_center_leakage_structurally_blocked": bool(
                torch.count_nonzero(leakage).item() == 0
            ),
            "temporal_context": "strictly_past_frames_only",
            "context_channel_order": _context_channel_order(
                self.imaging_channels, self.config
            ),
            "config": self.config.to_json_dict(),
        }


def _context_channel_order(
    imaging_channels: int, config: CausalAnnularContextConfig
) -> list[str]:
    order = [f"current_annulus_c{channel}" for channel in range(imaging_channels)]
    for lag in range(1, config.past_frames + 1):
        order.extend(
            f"past_lag_{lag}_c{channel}" for channel in range(imaging_channels)
        )
    if config.include_geometry_channels:
        order.extend(("annulus_mask", "target_mask"))
    return order


def _validated_video(video: Tensor, config: CausalAnnularContextConfig) -> Tensor:
    if not isinstance(video, Tensor):
        raise ConditionalBackgroundError("video must be a torch.Tensor")
    if video.ndim != 5:
        raise ConditionalBackgroundError(
            "video must have shape [batch, channel, time, height, width]"
        )
    if video.dtype == torch.bool or video.is_complex():
        raise ConditionalBackgroundError("video must have a real numeric dtype")
    if int(video.shape[0]) < 1 or int(video.shape[1]) < 1:
        raise ConditionalBackgroundError("video batch and channel axes must be nonempty")
    if int(video.shape[2]) <= config.past_frames:
        raise ConditionalBackgroundError(
            "video must contain more frames than the declared past context"
        )
    if min(int(video.shape[-2]), int(video.shape[-1])) < 3:
        raise ConditionalBackgroundError("video spatial dimensions are too small")
    if not bool(torch.isfinite(video).all()):
        raise ConditionalBackgroundError("video must contain only finite values")
    if not video.is_floating_point():
        return video.to(dtype=torch.float32)
    return video


def _radial_masks(
    height: int,
    width: int,
    config: CausalAnnularContextConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    maximum_complete_radius = min(center_y, center_x)
    if config.annulus_outer_radius_px > maximum_complete_radius + 1e-12:
        raise ConditionalBackgroundError(
            "annulus_outer_radius_px does not fit completely inside the input patch"
        )
    rows = torch.arange(height, device=device, dtype=torch.float64) - center_y
    columns = torch.arange(width, device=device, dtype=torch.float64) - center_x
    squared_radius = rows[:, None].square() + columns[None, :].square()
    target = squared_radius <= config.target_radius_px**2
    annulus = (squared_radius >= config.annulus_inner_radius_px**2) & (
        squared_radius <= config.annulus_outer_radius_px**2
    )
    support = squared_radius <= config.annulus_outer_radius_px**2
    if not bool(target.any()) or not bool(annulus.any()):
        raise ConditionalBackgroundError("declared geometry produced an empty mask")
    if bool((target & annulus).any()):
        raise ConditionalBackgroundError("declared target and annulus overlap")
    return target, annulus, support


def build_causal_annular_context(
    video: Tensor,
    config: CausalAnnularContextConfig = CausalAnnularContextConfig(),
) -> CausalAnnularBatch:
    """Mask the current center before constructing any model-visible tensor."""

    original_dtype = str(video.dtype).removeprefix("torch.") if isinstance(video, Tensor) else "invalid"
    values = _validated_video(video, config)
    batch, channels, frames, height, width = map(int, values.shape)
    target_mask, annulus_mask, support_mask = _radial_masks(
        height, width, config, device=values.device
    )
    lag_count = config.past_frames
    output_frames = frames - lag_count
    annulus_values = (
        values[:, :, lag_count:] * annulus_mask.to(dtype=values.dtype)
    ).permute(0, 2, 1, 3, 4)
    inputs = [annulus_values]
    for lag in range(1, lag_count + 1):
        past = values[:, :, lag_count - lag : frames - lag]
        inputs.append(
            (past * support_mask.to(dtype=values.dtype)).permute(0, 2, 1, 3, 4)
        )
    if config.include_geometry_channels:
        geometry = torch.stack((annulus_mask, target_mask), dim=0).to(
            dtype=values.dtype
        )
        inputs.append(
            geometry.reshape(1, 1, 2, height, width).expand(
                batch, output_frames, 2, height, width
            )
        )
    context = torch.cat(inputs, dim=2)
    target = values[:, :, lag_count:].permute(0, 2, 1, 3, 4)
    target = target * target_mask.to(dtype=values.dtype)
    result = CausalAnnularBatch(
        context=context,
        target=target,
        target_mask=target_mask,
        annulus_mask=annulus_mask,
        past_support_mask=support_mask,
        target_frame_indices=tuple(range(lag_count, frames)),
        config=config,
        input_dtype=original_dtype,
    )
    if not result.summary()["current_center_leakage_structurally_blocked"]:
        raise RuntimeError("internal current-center leakage invariant failed")
    return result


@dataclass(frozen=True)
class ConditionalResidualOutput:
    """Pixel-space background estimate and exact target-supported residual."""

    target: Tensor
    background_estimate: Tensor
    raw_residual: Tensor
    target_mask: Tensor
    target_frame_indices: tuple[int, ...]
    context_contract: Mapping[str, Any]
    model_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.target.ndim != 5 or self.target.shape != self.background_estimate.shape:
            raise ConditionalBackgroundError(
                "target and background_estimate must share [B,N,C,H,W] shape"
            )
        if self.raw_residual.shape != self.target.shape:
            raise ConditionalBackgroundError("raw_residual shape does not match target")
        if self.target_mask.dtype != torch.bool or tuple(self.target_mask.shape) != tuple(
            self.target.shape[-2:]
        ):
            raise ConditionalBackgroundError("target_mask is inconsistent with outputs")
        for name, values in (
            ("target", self.target),
            ("background_estimate", self.background_estimate),
            ("raw_residual", self.raw_residual),
        ):
            if not bool(torch.isfinite(values).all()):
                raise ConditionalBackgroundError(f"{name} contains nonfinite values")
        outside = ~self.target_mask
        if any(
            bool(torch.count_nonzero(values[..., outside]).item())
            for values in (self.target, self.background_estimate, self.raw_residual)
        ):
            raise ConditionalBackgroundError("outputs must be zero outside target_mask")

    def summary(self) -> dict[str, Any]:
        closure = self.target - self.background_estimate - self.raw_residual
        valid_closure = closure[..., self.target_mask]
        return {
            "schema_version": "conditional_pixel_residual_v1",
            "shape": [int(value) for value in self.target.shape],
            "target_frame_indices": [int(value) for value in self.target_frame_indices],
            "target_pixel_count": int(self.target_mask.sum().item()),
            "background_estimate_space": "input_video_pixels_center_support",
            "residual_definition": "observed_current_center_minus_predicted_background",
            "residual_signed": True,
            "residual_squared_or_rectified": False,
            "pair_closure_max_abs": _json_float(valid_closure.abs().max()),
            "finite": True,
            "context_contract": dict(self.context_contract),
            "model_contract": dict(self.model_contract),
        }


def _assemble_output(
    batch: CausalAnnularBatch,
    background_estimate: Tensor,
    *,
    model_contract: Mapping[str, Any],
) -> ConditionalResidualOutput:
    if background_estimate.shape != batch.target.shape:
        raise ConditionalBackgroundError(
            "background estimate must match aligned target shape; received "
            f"{tuple(background_estimate.shape)} versus {tuple(batch.target.shape)}"
        )
    if not bool(torch.isfinite(background_estimate).all()):
        raise ConditionalBackgroundError("background estimate contains nonfinite values")
    mask = batch.target_mask.to(device=background_estimate.device)
    target = batch.target.to(
        device=background_estimate.device, dtype=background_estimate.dtype
    )
    estimate = background_estimate * mask.to(dtype=background_estimate.dtype)
    residual = (target - estimate) * mask.to(dtype=background_estimate.dtype)
    return ConditionalResidualOutput(
        target=target,
        background_estimate=estimate,
        raw_residual=residual,
        target_mask=mask,
        target_frame_indices=batch.target_frame_indices,
        context_contract=batch.summary(),
        model_contract=dict(model_contract),
    )


def conditional_background_loss(
    output: ConditionalResidualOutput,
    *,
    loss: str = "smooth_l1",
    beta: float = 1.0,
) -> Tensor:
    """Return a target-mask-only regression loss for neural-model training."""

    observed = output.target[..., output.target_mask]
    predicted = output.background_estimate[..., output.target_mask]
    if loss == "mse":
        return F.mse_loss(predicted, observed)
    if loss == "smooth_l1":
        resolved_beta = _finite_positive(beta, name="beta")
        return F.smooth_l1_loss(predicted, observed, beta=resolved_beta)
    raise ConditionalBackgroundError("loss must be 'mse' or 'smooth_l1'")


@dataclass(frozen=True)
class LowRankRidgeConfig:
    """Deterministic principal-component ridge baseline configuration."""

    rank: int = 8
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        if int(self.rank) != self.rank or self.rank < 1:
            raise ConditionalBackgroundError("rank must be a positive integer")
        _finite_positive(self.ridge, name="ridge", allow_zero=True)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicLowRankBackgroundPredictor:
    """Source-off-fitted low-rank linear reference with no random state.

    Context is centered, projected onto deterministic SVD components, and
    mapped to center pixels by ridge regression.  SVD sign ambiguity cancels
    between scores and coefficients and therefore does not affect predictions.
    """

    def __init__(
        self,
        context_config: CausalAnnularContextConfig = CausalAnnularContextConfig(),
        config: LowRankRidgeConfig = LowRankRidgeConfig(),
    ) -> None:
        self.context_config = context_config
        self.config = config
        self._state: dict[str, Any] | None = None

    @staticmethod
    def _matrices(batch: CausalAnnularBatch) -> tuple[np.ndarray, np.ndarray]:
        context = batch.context.detach().to(device="cpu", dtype=torch.float64).numpy()
        target = batch.target.detach().to(device="cpu", dtype=torch.float64)
        samples = int(context.shape[0] * context.shape[1])
        x_matrix = context.reshape(samples, -1)
        y_matrix = target[..., batch.target_mask.cpu()].numpy().reshape(samples, -1)
        if not np.isfinite(x_matrix).all() or not np.isfinite(y_matrix).all():
            raise ConditionalBackgroundError("linear baseline matrices must be finite")
        return x_matrix, y_matrix

    def fit_source_off(
        self,
        source_off_video: Tensor,
        *,
        source_off_id: str,
    ) -> dict[str, Any]:
        """Fit only the declared source-off movie and return JSON-ready metadata."""

        if not isinstance(source_off_id, str) or not source_off_id.strip():
            raise ConditionalBackgroundError("source_off_id must be a nonempty string")
        batch = build_causal_annular_context(source_off_video, self.context_config)
        x_matrix, y_matrix = self._matrices(batch)
        if x_matrix.shape[0] < 2:
            raise ConditionalBackgroundError("linear fit requires at least two samples")
        x_mean = np.mean(x_matrix, axis=0, dtype=np.float64)
        y_mean = np.mean(y_matrix, axis=0, dtype=np.float64)
        centered_x = x_matrix - x_mean
        _, singular_values, right_vectors = np.linalg.svd(
            centered_x, full_matrices=False
        )
        tolerance = np.finfo(np.float64).eps * max(centered_x.shape) * singular_values[0]
        numerical_rank = int(np.count_nonzero(singular_values > tolerance))
        if numerical_rank < 1:
            raise ConditionalBackgroundError("source-off context has zero numerical rank")
        fitted_rank = min(self.config.rank, numerical_rank)
        components = right_vectors[:fitted_rank]
        scores = centered_x @ components.T
        gram = scores.T @ scores
        gram.flat[:: fitted_rank + 1] += self.config.ridge
        coefficients = np.linalg.solve(gram, scores.T @ (y_matrix - y_mean))
        arrays = (x_mean, y_mean, components, coefficients)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ConditionalBackgroundError("linear baseline fit produced nonfinite state")
        self._state = {
            "source_off_id": source_off_id.strip(),
            "x_mean": x_mean,
            "y_mean": y_mean,
            "components": components,
            "coefficients": coefficients,
            "fitted_rank": fitted_rank,
            "numerical_rank": numerical_rank,
            "imaging_channels": batch.imaging_channels,
            "spatial_shape": tuple(int(value) for value in batch.target.shape[-2:]),
            "target_mask": batch.target_mask.detach().cpu().numpy(),
            "sample_count": int(x_matrix.shape[0]),
            "context_feature_count": int(x_matrix.shape[1]),
            "target_feature_count": int(y_matrix.shape[1]),
        }
        return self.fit_summary()

    def _require_state(self) -> dict[str, Any]:
        if self._state is None:
            raise ConditionalBackgroundError("linear baseline must be fit before prediction")
        return self._state

    def predict(self, video: Tensor) -> ConditionalResidualOutput:
        state = self._require_state()
        batch = build_causal_annular_context(video, self.context_config)
        if batch.imaging_channels != state["imaging_channels"] or tuple(
            batch.target.shape[-2:]
        ) != state["spatial_shape"]:
            raise ConditionalBackgroundError("prediction geometry differs from fitted geometry")
        if not np.array_equal(batch.target_mask.detach().cpu().numpy(), state["target_mask"]):
            raise ConditionalBackgroundError("prediction target mask differs from fitted mask")
        x_matrix, _ = self._matrices(batch)
        predicted = (
            (x_matrix - state["x_mean"])
            @ state["components"].T
            @ state["coefficients"]
            + state["y_mean"]
        )
        if not np.isfinite(predicted).all():
            raise ConditionalBackgroundError("linear prediction produced nonfinite values")
        batch_size, output_frames, channels, height, width = map(int, batch.target.shape)
        pixel_count = int(batch.target_mask.sum().item())
        center_values = predicted.reshape(batch_size, output_frames, channels, pixel_count)
        estimate = torch.zeros_like(batch.target, dtype=torch.float64, device="cpu")
        estimate[..., batch.target_mask.cpu()] = torch.from_numpy(center_values)
        estimate = estimate.to(device=video.device, dtype=batch.target.dtype)
        return _assemble_output(batch, estimate, model_contract=self.fit_summary())

    def fit_summary(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "schema_version": "deterministic_low_rank_background_predictor_v1",
            "model_family": "principal_component_ridge",
            "fit_scope": "source_off_only",
            "source_off_id": state["source_off_id"],
            "sample_count": state["sample_count"],
            "context_feature_count": state["context_feature_count"],
            "target_feature_count": state["target_feature_count"],
            "requested_rank": self.config.rank,
            "fitted_rank": state["fitted_rank"],
            "source_off_context_numerical_rank": state["numerical_rank"],
            "ridge": self.config.ridge,
            "deterministic": True,
            "current_center_available_to_model": False,
            "context_config": self.context_config.to_json_dict(),
        }

    def export_state_json(self) -> dict[str, Any]:
        """Return a complete JSON-serializable state (potentially large)."""

        state = self._require_state()
        return {
            "contract": self.fit_summary(),
            "x_mean": state["x_mean"].tolist(),
            "y_mean": state["y_mean"].tolist(),
            "components": state["components"].tolist(),
            "coefficients": state["coefficients"].tolist(),
            "target_mask": state["target_mask"].astype(bool).tolist(),
            "spatial_shape": list(state["spatial_shape"]),
            "imaging_channels": state["imaging_channels"],
        }


@dataclass(frozen=True)
class ConvBackgroundConfig:
    """Small two-dimensional predictor for a bounded accelerator screen."""

    hidden_channels: int = 24
    depth: int = 3
    kernel_size: int = 3
    norm_groups: int = 4

    def __post_init__(self) -> None:
        for name in ("hidden_channels", "depth", "kernel_size", "norm_groups"):
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 1:
                raise ConditionalBackgroundError(f"{name} must be a positive integer")
        if self.kernel_size % 2 != 1:
            raise ConditionalBackgroundError("kernel_size must be odd")
        if self.hidden_channels % self.norm_groups:
            raise ConditionalBackgroundError(
                "hidden_channels must be divisible by norm_groups"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.conv(F.silu(self.norm(values)))


class CausalAnnularConvPredictor(nn.Module):
    """Compact convolutional center predictor with internal leakage masking."""

    def __init__(
        self,
        *,
        imaging_channels: int = 1,
        context_config: CausalAnnularContextConfig = CausalAnnularContextConfig(),
        model_config: ConvBackgroundConfig = ConvBackgroundConfig(),
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        if int(imaging_channels) != imaging_channels or imaging_channels < 1:
            raise ConditionalBackgroundError("imaging_channels must be a positive integer")
        self.imaging_channels = int(imaging_channels)
        self.context_config = context_config
        self.model_config = model_config
        self.initialization_seed = _validated_seed(initialization_seed)
        input_channels = context_config.context_channels(self.imaging_channels)
        padding = model_config.kernel_size // 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.input_projection = nn.Conv2d(
                input_channels,
                model_config.hidden_channels,
                kernel_size=model_config.kernel_size,
                padding=padding,
            )
            self.blocks = nn.Sequential(
                *[
                    _SpatialResidualBlock(
                        model_config.hidden_channels,
                        model_config.norm_groups,
                        model_config.kernel_size,
                    )
                    for _ in range(model_config.depth)
                ]
            )
            self.output_head = nn.Conv2d(
                model_config.hidden_channels, self.imaging_channels, kernel_size=1
            )

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": "causal_annular_conv_predictor_v1",
            "model_family": "small_spatial_convolutional_background_predictor",
            "parameter_count": int(sum(parameter.numel() for parameter in self.parameters())),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            ),
            "initialization_seed": self.initialization_seed,
            "current_center_available_to_model": False,
            "future_frames_available_to_model": False,
            "background_estimate_space": "input_video_pixels_center_support",
            "context_config": self.context_config.to_json_dict(),
            "model_config": self.model_config.to_json_dict(),
        }

    def forward(self, video: Tensor) -> ConditionalResidualOutput:
        batch = build_causal_annular_context(video, self.context_config)
        if batch.imaging_channels != self.imaging_channels:
            raise ConditionalBackgroundError(
                f"model expects {self.imaging_channels} imaging channels, "
                f"received {batch.imaging_channels}"
            )
        batch_size, output_frames, features, height, width = map(
            int, batch.context.shape
        )
        parameter = next(self.parameters())
        context = batch.context.to(device=parameter.device, dtype=parameter.dtype)
        values = context.reshape(batch_size * output_frames, features, height, width)
        values = F.silu(self.input_projection(values))
        values = self.blocks(values)
        estimate = self.output_head(F.silu(values)).reshape(
            batch_size,
            output_frames,
            self.imaging_channels,
            height,
            width,
        )
        return _assemble_output(batch, estimate, model_contract=self.contract())


@dataclass(frozen=True)
class RobustResidualScaleConfig:
    """Robust source-off calibration with no silent noise-scale floor."""

    granularity: str = "per_channel"
    normal_consistency: float = 1.4826
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.granularity not in {"per_channel", "per_pixel"}:
            raise ConditionalBackgroundError(
                "granularity must be 'per_channel' or 'per_pixel'"
            )
        _finite_positive(self.normal_consistency, name="normal_consistency")
        _finite_positive(self.minimum_scale, name="minimum_scale", allow_zero=True)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StandardizedResidualOutput:
    standardized_residual: Tensor
    raw_output: ConditionalResidualOutput
    calibration_contract: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        values = self.standardized_residual[..., self.raw_output.target_mask]
        if not bool(torch.isfinite(values).all()):
            raise ConditionalBackgroundError("standardized residual contains nonfinite values")
        return {
            "schema_version": "robust_standardized_conditional_residual_v1",
            "shape": [int(value) for value in self.standardized_residual.shape],
            "finite": True,
            "maximum_absolute_z": _json_float(values.abs().max()),
            "calibration_contract": dict(self.calibration_contract),
            "raw_output": self.raw_output.summary(),
        }


class RobustResidualStandardizer:
    """Freeze robust location/noise scale on source-off residuals only."""

    def __init__(self, config: RobustResidualScaleConfig = RobustResidualScaleConfig()) -> None:
        self.config = config
        self._location: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._source_off_id: str | None = None
        self._target_mask: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None
        self._sample_count: int | None = None

    def fit_source_off(
        self,
        source_off: ConditionalResidualOutput,
        *,
        source_off_id: str,
    ) -> dict[str, Any]:
        if not isinstance(source_off_id, str) or not source_off_id.strip():
            raise ConditionalBackgroundError("source_off_id must be a nonempty string")
        residual = source_off.raw_residual.detach().to(device="cpu", dtype=torch.float64)
        mask = source_off.target_mask.detach().cpu()
        values = residual[..., mask].numpy()  # [B,N,C,P]
        if not np.isfinite(values).all() or values.size == 0:
            raise ConditionalBackgroundError("source-off residual sample is invalid")
        sample_count = int(values.shape[0] * values.shape[1])
        if sample_count < 3:
            raise ConditionalBackgroundError(
                "robust source-off calibration requires at least three target frames"
            )
        if self.config.granularity == "per_channel":
            flattened = values.transpose(2, 0, 1, 3).reshape(values.shape[2], -1)
            location = np.median(flattened, axis=1)
            scale = self.config.normal_consistency * np.median(
                np.abs(flattened - location[:, None]), axis=1
            )
        else:
            flattened = values.transpose(2, 3, 0, 1).reshape(
                values.shape[2], values.shape[3], -1
            )
            location = np.median(flattened, axis=2)
            scale = self.config.normal_consistency * np.median(
                np.abs(flattened - location[..., None]), axis=2
            )
        if not np.isfinite(location).all() or not np.isfinite(scale).all():
            raise ConditionalBackgroundError("robust calibration produced nonfinite values")
        minimum_observed = float(np.min(scale))
        if minimum_observed <= self.config.minimum_scale:
            raise ConditionalBackgroundError(
                "source-off robust noise scale is degenerate: minimum observed "
                f"{minimum_observed:.9g} <= required {self.config.minimum_scale:.9g}"
            )
        self._location = location
        self._scale = scale
        self._source_off_id = source_off_id.strip()
        self._target_mask = mask.numpy()
        self._shape = tuple(int(value) for value in residual.shape[2:])
        self._sample_count = sample_count
        return self.contract(include_parameters=False)

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self._location is None or self._scale is None:
            raise ConditionalBackgroundError("standardizer must be source-off fit first")
        return self._location, self._scale

    def transform(self, output: ConditionalResidualOutput) -> StandardizedResidualOutput:
        location, scale = self._require_fit()
        if tuple(int(value) for value in output.raw_residual.shape[2:]) != self._shape:
            raise ConditionalBackgroundError("residual geometry differs from calibration")
        mask = output.target_mask.detach().cpu().numpy()
        if not np.array_equal(mask, self._target_mask):
            raise ConditionalBackgroundError("residual target mask differs from calibration")
        residual = output.raw_residual
        if self.config.granularity == "per_channel":
            location_tensor = torch.as_tensor(
                location, dtype=residual.dtype, device=residual.device
            ).reshape(1, 1, -1, 1, 1)
            scale_tensor = torch.as_tensor(
                scale, dtype=residual.dtype, device=residual.device
            ).reshape(1, 1, -1, 1, 1)
        else:
            channels, height, width = self._shape
            full_location = np.zeros((channels, height, width), dtype=np.float64)
            full_scale = np.ones((channels, height, width), dtype=np.float64)
            full_location[:, mask] = location
            full_scale[:, mask] = scale
            location_tensor = torch.as_tensor(
                full_location, dtype=residual.dtype, device=residual.device
            ).reshape(1, 1, channels, height, width)
            scale_tensor = torch.as_tensor(
                full_scale, dtype=residual.dtype, device=residual.device
            ).reshape(1, 1, channels, height, width)
        standardized = (residual - location_tensor) / scale_tensor
        standardized = standardized * output.target_mask.to(dtype=residual.dtype)
        if not bool(torch.isfinite(standardized).all()):
            raise ConditionalBackgroundError("standardization produced nonfinite values")
        return StandardizedResidualOutput(
            standardized_residual=standardized,
            raw_output=output,
            calibration_contract=self.contract(include_parameters=False),
        )

    def contract(self, *, include_parameters: bool = False) -> dict[str, Any]:
        location, scale = self._require_fit()
        result: dict[str, Any] = {
            "schema_version": "source_off_robust_residual_calibration_v1",
            "fit_scope": "source_off_only",
            "source_off_id": self._source_off_id,
            "sample_count": self._sample_count,
            "granularity": self.config.granularity,
            "location_estimator": "median",
            "scale_estimator": "normal_consistent_median_absolute_deviation",
            "normal_consistency": self.config.normal_consistency,
            "minimum_scale_gate": self.config.minimum_scale,
            "minimum_fitted_scale": float(np.min(scale)),
            "maximum_fitted_scale": float(np.max(scale)),
            "source_on_refit_permitted": False,
        }
        if include_parameters:
            result["location"] = location.tolist()
            result["scale"] = scale.tolist()
            result["target_mask"] = self._target_mask.astype(bool).tolist()
        return result


def _valid_values(output: ConditionalResidualOutput, name: str) -> Tensor:
    values = getattr(output, name)[..., output.target_mask].detach().to(
        device="cpu", dtype=torch.float64
    )
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ConditionalBackgroundError(f"{name} values are empty or nonfinite")
    return values


def _normal_consistent_mad(values: Tensor) -> float:
    median = torch.median(values)
    scale = 1.4826 * torch.median(torch.abs(values - median))
    resolved = float(scale.item())
    if not math.isfinite(resolved) or resolved <= np.finfo(np.float64).eps:
        raise ConditionalBackgroundError("diagnostic reference noise scale is degenerate")
    return resolved


def _collapse_diagnostics(prediction: Tensor) -> dict[str, Any]:
    # Treat each batch-frame as one observation and center every predicted pixel.
    matrix = prediction.reshape(-1, prediction.shape[-1]).numpy()
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    total_variance = float(np.sum(matrix * matrix))
    maximum_rank = int(min(matrix.shape))
    if total_variance <= np.finfo(np.float64).eps or maximum_rank == 0:
        return {
            "effective_rank": 0.0,
            "effective_rank_fraction": 0.0,
            "first_component_variance_fraction": 1.0,
            "prediction_constant": True,
        }
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    weights = np.square(singular_values)
    weights = weights / np.sum(weights)
    positive = weights[weights > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return {
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / maximum_rank,
        "first_component_variance_fraction": float(weights[0]),
        "prediction_constant": False,
    }


def conditional_residual_diagnostics(
    source_off: ConditionalResidualOutput,
    *,
    source_on: ConditionalResidualOutput | None = None,
    injected_signal_target: Tensor | None = None,
) -> dict[str, Any]:
    """Compute background, signal-retention, and output-collapse diagnostics.

    Signal metrics require an exact source-off/source-on pair and an injected
    target tensor in the same supported pixel space.  Native background pixels
    are not assigned biological negative labels by this function.
    """

    observed = _valid_values(source_off, "target")
    predicted = _valid_values(source_off, "background_estimate")
    residual = _valid_values(source_off, "raw_residual")
    observed_rms = float(torch.sqrt(torch.mean(observed.square())).item())
    residual_rms = float(torch.sqrt(torch.mean(residual.square())).item())
    if observed_rms <= np.finfo(np.float64).eps:
        raise ConditionalBackgroundError("source-off target RMS is degenerate")
    observed_scale = _normal_consistent_mad(observed)
    residual_scale = _normal_consistent_mad(residual)
    target_variance = float(torch.var(observed, unbiased=False).item())
    prediction_variance = float(torch.var(predicted, unbiased=False).item())
    collapse = _collapse_diagnostics(
        predicted.reshape(
            int(predicted.shape[0] * predicted.shape[1]),
            int(np.prod(predicted.shape[2:])),
        )
    )
    result: dict[str, Any] = {
        "schema_version": "conditional_background_diagnostics_v1",
        "background": {
            "target_rms": observed_rms,
            "residual_rms": residual_rms,
            "residual_to_target_rms_ratio": residual_rms / observed_rms,
            "background_suppression_fraction_rms": 1.0 - residual_rms / observed_rms,
            "target_robust_scale": observed_scale,
            "residual_robust_scale": residual_scale,
            "residual_to_target_robust_scale_ratio": residual_scale / observed_scale,
            "background_suppression_fraction_robust": 1.0
            - residual_scale / observed_scale,
        },
        "collapse": {
            **collapse,
            "target_variance": target_variance,
            "prediction_variance": prediction_variance,
            "prediction_to_target_variance_ratio": (
                prediction_variance / target_variance if target_variance > 0 else 0.0
            ),
            "screen_flag": bool(
                collapse["prediction_constant"]
                or collapse["effective_rank_fraction"] < 0.05
                or collapse["first_component_variance_fraction"] > 0.90
            ),
            "screen_thresholds": {
                "minimum_effective_rank_fraction": 0.05,
                "maximum_first_component_variance_fraction": 0.90,
            },
        },
        "signal": None,
        "interpretation_boundary": {
            "native_background_truth": "unknown_not_negative",
            "residual": "signal_plus_noise_plus_background_model_error",
            "neuron_probability": False,
        },
    }
    paired_inputs = (source_on is not None, injected_signal_target is not None)
    if any(paired_inputs) and not all(paired_inputs):
        raise ConditionalBackgroundError(
            "source_on and injected_signal_target must be supplied together"
        )
    if source_on is None or injected_signal_target is None:
        return result
    if (
        source_on.target.shape != source_off.target.shape
        or source_on.target_frame_indices != source_off.target_frame_indices
        or not torch.equal(source_on.target_mask.cpu(), source_off.target_mask.cpu())
    ):
        raise ConditionalBackgroundError("source-on and source-off outputs are not aligned")
    if injected_signal_target.shape != source_off.target.shape:
        raise ConditionalBackgroundError("injected_signal_target shape does not match outputs")
    signal = injected_signal_target.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(signal).all()):
        raise ConditionalBackgroundError("injected signal contains nonfinite values")
    outside = ~source_off.target_mask.cpu()
    if bool(torch.count_nonzero(signal[..., outside]).item()):
        raise ConditionalBackgroundError("injected signal must be zero outside target mask")
    off_target = source_off.target.detach().to(device="cpu", dtype=torch.float64)
    on_target = source_on.target.detach().to(device="cpu", dtype=torch.float64)
    pair_difference = on_target - off_target
    pair_error = pair_difference - signal
    maximum_pair_error = float(pair_error.abs().max().item())
    signal_maximum = float(signal.abs().max().item())
    pair_tolerance = 1e-6 + 1e-5 * signal_maximum
    if maximum_pair_error > pair_tolerance:
        raise ConditionalBackgroundError(
            "injected signal does not close the source-off/source-on target pair"
        )
    off_residual = source_off.raw_residual.detach().to(device="cpu", dtype=torch.float64)
    on_residual = source_on.raw_residual.detach().to(device="cpu", dtype=torch.float64)
    residual_delta = (on_residual - off_residual)[..., source_off.target_mask.cpu()]
    signal_values = signal[..., source_off.target_mask.cpu()]
    signal_energy = float(torch.sum(signal_values.square()).item())
    if signal_energy <= np.finfo(np.float64).eps:
        raise ConditionalBackgroundError("injected signal energy is degenerate")
    residual_energy = float(torch.sum(residual_delta.square()).item())
    projection = float(torch.sum(residual_delta * signal_values).item()) / signal_energy
    centered_delta = residual_delta - torch.mean(residual_delta)
    centered_signal = signal_values - torch.mean(signal_values)
    correlation_denominator = float(
        torch.sqrt(
            torch.sum(centered_delta.square()) * torch.sum(centered_signal.square())
        ).item()
    )
    correlation = (
        float(torch.sum(centered_delta * centered_signal).item())
        / correlation_denominator
        if correlation_denominator > np.finfo(np.float64).eps
        else 0.0
    )
    result["signal"] = {
        "pair_closure_max_abs": maximum_pair_error,
        "pair_closure_tolerance": pair_tolerance,
        "retention_projection_on_injected_signal": projection,
        "retention_norm_ratio": math.sqrt(residual_energy / signal_energy),
        "retention_correlation": correlation,
        "prediction_absorption_projection": 1.0 - projection,
        "injected_signal_energy": signal_energy,
        "residual_delta_energy": residual_energy,
    }
    return result


__all__ = [
    "CausalAnnularBatch",
    "CausalAnnularContextConfig",
    "CausalAnnularConvPredictor",
    "ConditionalBackgroundError",
    "ConditionalResidualOutput",
    "ConvBackgroundConfig",
    "DeterministicLowRankBackgroundPredictor",
    "LowRankRidgeConfig",
    "RobustResidualScaleConfig",
    "RobustResidualStandardizer",
    "StandardizedResidualOutput",
    "build_causal_annular_context",
    "conditional_background_loss",
    "conditional_residual_diagnostics",
]
