"""Paired direct/serial spatial stencils for the prospective Gamma-LS study.

These operators do not change the frozen Gamma-LS implementation. ``direct``
estimates reference moments of the input X; ``serial`` estimates moments of
the target-smoothed field A. Their denominators describe local variability,
not the sampling uncertainty of A-M or a standard-normal test statistic.

Spatial boundaries use valid-pixel renormalization at *each* convolution.
Chunking divides time only. Spatial tiling requires a sufficient input halo,
including both stencil radii for the serial arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Literal


Design = Literal["point", "direct", "serial"]
ReferenceFamily = Literal["gamma", "uniform"]
SupportGeometry = Literal["square", "disk"]


def _torch():
    import torch

    return torch


def _odd_width(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an odd integer >= {minimum}")
    if value < minimum or value % 2 != 1:
        raise ValueError(f"{name} must be an odd integer >= {minimum}")
    return value


@lru_cache(maxsize=32)
def match_centered_gamma_mu(
    target_width_px: int = 31,
    *,
    gaussian_sigma_px: float = 1.0,
    gaussian_truncate: float = 4.0,
) -> float:
    """Match the discrete radial second moment of a truncated Gaussian.

    A normalized exp(-mu*r) stencil on a finite square is compared with the
    normalized separable Gaussian on the radius round(truncate*sigma) square.
    Ninety-six deterministic bisection iterations use only Python float64
    arithmetic; no device or data-dependent search is involved.
    """
    width = _odd_width(target_width_px, "target_width_px", minimum=3)
    sigma = float(gaussian_sigma_px)
    truncate = float(gaussian_truncate)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("gaussian_sigma_px must be finite and positive")
    if not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("gaussian_truncate must be finite and positive")
    radius = int(truncate * sigma + 0.5)
    gaussian = [math.exp(-0.5 * (x / sigma) ** 2) for x in range(-radius, radius + 1)]
    moment = 2.0 * math.fsum(
        x * x * weight for x, weight in zip(range(-radius, radius + 1), gaussian)
    ) / math.fsum(gaussian)
    half = width // 2
    radii = [math.hypot(x, y) for y in range(-half, half + 1) for x in range(-half, half + 1)]

    def gamma_moment(mu: float) -> float:
        weights = [math.exp(-mu * r) for r in radii]
        return math.fsum(r * r * w for r, w in zip(radii, weights)) / math.fsum(weights)

    if not 0.0 < moment < gamma_moment(0.0):
        raise ValueError("Gaussian second moment is outside the target stencil's attainable range")
    lower, upper = 0.0, 1.0
    while gamma_moment(upper) > moment:
        upper *= 2.0
    for _ in range(96):
        middle = (lower + upper) / 2.0
        if gamma_moment(middle) > moment:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


@dataclass(frozen=True)
class TwoStencilSpec:
    """One spatial operator; guard=0 means no explicit exclusion, including CUT.

    The uniform reference includes the center when guard=0. The Gamma profile
    is naturally zero there because reference_shape_n>1. The target is always
    a full-square n=1 profile except when design=point (or target_width_px=1).
    support_geometry applies only to the reference.
    """

    design: Design = "direct"
    reference_family: ReferenceFamily = "gamma"
    guard_radius_px: float = 0.0
    reference_width_px: int = 31
    reference_shape_n: float = 9.0
    reference_mode_radius_px: float = 7.5
    support_geometry: SupportGeometry = "square"
    target_width_px: int = 31
    target_mu: float | None = None
    scale_floor: float = 0.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.design not in {"point", "direct", "serial"}:
            raise ValueError("design must be point, direct, or serial")
        if self.reference_family not in {"gamma", "uniform"}:
            raise ValueError("reference_family must be gamma or uniform")
        if self.support_geometry not in {"square", "disk"}:
            raise ValueError("support_geometry must be square or disk")
        reference_width = _odd_width(self.reference_width_px, "reference_width_px", minimum=3)
        _odd_width(self.target_width_px, "target_width_px")
        for name in ("guard_radius_px", "scale_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("reference_mode_radius_px", "epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        shape = float(self.reference_shape_n)
        if not math.isfinite(shape) or shape <= 1:
            raise ValueError("reference_shape_n must be finite and > 1")
        maximum_radius = reference_width // 2
        if self.support_geometry == "square":
            maximum_radius *= math.sqrt(2.0)
        if float(self.guard_radius_px) >= maximum_radius:
            raise ValueError("guard_radius_px leaves no reference pixels")
        if self.target_mu is None:
            # A singleton target is delta and its decay is immaterial.
            mu = 1.0 if self.target_width_px == 1 else match_centered_gamma_mu(self.target_width_px)
            object.__setattr__(self, "target_mu", mu)
        if not math.isfinite(float(self.target_mu)) or float(self.target_mu) <= 0:
            raise ValueError("target_mu must be finite and positive")

    @property
    def reference_mu(self) -> float:
        return (float(self.reference_shape_n) - 1.0) / float(self.reference_mode_radius_px)


@dataclass(frozen=True)
class TwoStencilCell:
    cell_id: str
    input_representation: str
    spec: TwoStencilSpec
    is_deployed_anchor: bool = False


def deployed_anchor_cell() -> TwoStencilCell:
    """Exact spatial settings of the existing signed h15 deployment.

    The caller must supply the same conditioned signed input and calibrated
    scale floor for numerical parity. This object does not perform calibration.
    """
    return TwoStencilCell(
        "deployed_signed_point_gamma_g7_disk",
        "difference_signed",
        TwoStencilSpec(design="point", guard_radius_px=7.0, support_geometry="disk"),
        is_deployed_anchor=True,
    )


def build_two_stencil_factorial(*, include_deployed_anchor: bool = False) -> tuple[TwoStencilCell, ...]:
    """The fixed 24-cell square-support design, optionally plus its disk anchor."""
    mu = match_centered_gamma_mu()
    cells = tuple(
        TwoStencilCell(
            f"{representation}__{design}_{family}_g{guard}_square",
            representation,
            TwoStencilSpec(
                design=design,
                reference_family=family,
                guard_radius_px=float(guard),
                target_mu=mu,
            ),
        )
        for representation in ("conditioned_current_frame", "difference_signed")
        for design in ("point", "direct", "serial")
        for guard in (0, 7)
        for family in ("uniform", "gamma")
    )
    return cells + ((deployed_anchor_cell(),) if include_deployed_anchor else ())


def _radii(width: int, *, device: Any, dtype: Any) -> Any:
    torch = _torch()
    half = width // 2
    coordinate = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.sqrt(xx.square() + yy.square())


def build_two_stencil_kernels(
    spec: TwoStencilSpec, *, device: Any = None, dtype: Any = None
) -> tuple[Any, Any]:
    """Return (target, reference) normalized torch kernels on the requested device."""
    torch = _torch()
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("kernel dtype must be torch.float32 or torch.float64")
    if spec.design == "point" or spec.target_width_px == 1:
        target = torch.ones((1, 1), dtype=dtype, device=device)
    else:
        target = torch.exp(-float(spec.target_mu) * _radii(spec.target_width_px, device=device, dtype=dtype))
        target = target / target.sum()
    radius = _radii(spec.reference_width_px, device=device, dtype=dtype)
    if spec.reference_family == "gamma":
        reference = radius.pow(float(spec.reference_shape_n) - 1.0) * torch.exp(-spec.reference_mu * radius)
    else:
        reference = torch.ones_like(radius)
    include = torch.ones_like(radius, dtype=torch.bool)
    if spec.guard_radius_px > 0:
        include = include & (radius > float(spec.guard_radius_px))
    if spec.support_geometry == "disk":
        include = include & (radius <= spec.reference_width_px // 2)
    reference = torch.where(include, reference, torch.zeros_like(reference))
    total = reference.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError("reference stencil has no finite positive mass")
    return target, reference / total


@dataclass(frozen=True)
class TwoStencilResult:
    values: Any
    target_response: Any | None
    reference_mean: Any | None
    reference_std: Any | None
    contrast: Any | None
    target_kernel: Any
    reference_kernel: Any
    diagnostics: dict[str, Any]

    @property
    def local_mean(self) -> Any | None:
        return self.reference_mean

    @property
    def local_std(self) -> Any | None:
        return self.reference_std


def two_stencil_local_standardization(
    values: Any,
    spec: TwoStencilSpec,
    *,
    device: Any = None,
    chunk_frames: int | None = None,
    return_stages: bool = True,
) -> TwoStencilResult:
    """Return signed LS scores and optional stages for finite T,Y,X input.

    Integer/half-precision inputs are converted to float32; float64 is retained.
    Returned stages remain on the input/requested device. The floor must be
    fitted by the caller on calibration-only data, separately for each arm.
    """
    torch = _torch()
    import torch.nn.functional as functional

    video = torch.as_tensor(values, device=device)
    if video.is_complex():
        raise ValueError("values must be real")
    if video.dtype not in {torch.float32, torch.float64}:
        video = video.to(dtype=torch.float32)
    if video.ndim != 3 or video.numel() == 0:
        raise ValueError("values must be a nonempty T,Y,X array")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("values must be finite")
    if chunk_frames is not None and (
        isinstance(chunk_frames, bool) or not isinstance(chunk_frames, int) or chunk_frames < 1
    ):
        raise ValueError("chunk_frames must be a positive integer")
    frames_per_chunk = len(video) if chunk_frames is None else chunk_frames
    target, reference = build_two_stencil_kernels(spec, device=video.device, dtype=video.dtype)

    def convolution(field: Any, kernel: Any) -> Any:
        return functional.conv2d(
            field.unsqueeze(1), kernel.reshape(1, 1, *kernel.shape), padding=kernel.shape[-1] // 2
        ).squeeze(1)

    with torch.no_grad():
        ones = torch.ones((1, *video.shape[-2:]), dtype=video.dtype, device=video.device)
        reference_mass = convolution(ones, reference)
        if bool((reference_mass <= 0).any()):
            raise ValueError("reference stencil has zero valid mass at one or more image pixels")
        target_mass = convolution(ones, target) if target.numel() > 1 else None
        if target_mass is not None and bool((target_mass <= 0).any()):
            raise ValueError("target stencil has zero valid mass")
        parts: dict[str, list[Any]] = {name: [] for name in ("values", "target_response", "reference_mean", "reference_std", "contrast")}
        for start in range(0, len(video), frames_per_chunk):
            source = video[start:start + frames_per_chunk]
            a = source if target.numel() == 1 else convolution(source, target) / target_mass
            domain = a if spec.design == "serial" else source
            mean = convolution(domain, reference) / reference_mass
            second = convolution(domain.square(), reference) / reference_mass
            std = torch.sqrt((second - mean.square()).clamp_min(0.0))
            if not bool(torch.isfinite(std).all()):
                raise ValueError("reference moments overflowed the selected numerical precision")
            contrast = a - mean
            denominator = torch.maximum(std, torch.as_tensor(float(spec.scale_floor), dtype=video.dtype, device=video.device)) + float(spec.epsilon)
            score = contrast / denominator
            if not bool(torch.isfinite(score).all()):
                raise ValueError("two-stencil standardization produced non-finite scores")
            parts["values"].append(score)
            if return_stages:
                for name, stage in (("target_response", a), ("reference_mean", mean), ("reference_std", std), ("contrast", contrast)):
                    parts[name].append(stage)
        outputs = {
            name: torch.cat(chunks, dim=0) if chunks else None
            for name, chunks in parts.items()
        }

    target_half = target.shape[-1] // 2
    reference_half = spec.reference_width_px // 2
    diagnostics = {
        "operator": "gamma_two_stencil_local_standardization",
        "design": spec.design,
        "reference_family": spec.reference_family,
        "reference_domain": "target_response" if spec.design == "serial" else "input",
        "reference_variance_role": "local_field_variability_not_contrast_sampling_variance",
        "guard_radius_px": float(spec.guard_radius_px),
        "guard_semantics": "exclude_r_lte_guard_only_when_guard_positive_in_reference_sampling_coordinates",
        "support_geometry": spec.support_geometry,
        "reference_width_px": spec.reference_width_px,
        "target_width_px": int(target.shape[-1]),
        "target_mu": float(spec.target_mu),
        "reference_shape_n": float(spec.reference_shape_n),
        "reference_mode_radius_px": float(spec.reference_mode_radius_px),
        "reference_mu": spec.reference_mu,
        "boundary_mode": "valid_renormalized_zero_each_stage",
        "required_input_halo_px": target_half + reference_half if spec.design == "serial" else max(target_half, reference_half),
        "interior_mean_reference_width_px": 2 * (target_half + reference_half) + 1 if spec.design == "serial" else spec.reference_width_px,
        "scale_floor": float(spec.scale_floor),
        "epsilon": float(spec.epsilon),
        "chunk_frames": frames_per_chunk,
        "device": str(video.device),
        "dtype": str(video.dtype).removeprefix("torch."),
        "backend": "conv2d",
        "signed_output": True,
        "probability_of_false_alarm_claimed": False,
    }
    return TwoStencilResult(**outputs, target_kernel=target, reference_kernel=reference, diagnostics=diagnostics)
