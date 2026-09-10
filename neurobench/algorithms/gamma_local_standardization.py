"""Signed Gamma-weighted local standardization and CFAR thresholding.

This module deliberately keeps three concepts separate:

* the cell under test is the current pixel;
* a normalized radial Gamma kernel estimates local reference moments; and
* CFAR is a direct threshold on the resulting signed standardized score.

Unlike :mod:`neurobench.algorithms.cfar`, the reference weights here are not a
uniform square annulus.  The functions accept and return PyTorch tensors so a
representation, Gamma local standardization, and threshold can remain on one
device.  No probability-of-false-alarm interpretation is attached to the
threshold without an external calibration experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


SupportGeometry = Literal["disk", "square"]
BoundaryMode = Literal["valid_renormalized_zero", "reflect"]


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - repository runtime includes torch
        raise RuntimeError("PyTorch is required for Gamma local standardization.") from exc
    return torch


@dataclass(frozen=True)
class GammaReferenceSpec:
    """Finite radial-Gamma reference used to estimate local moments.

    The unnormalized radial profile is

    ``r ** (shape_n - 1) * exp(-rate_mu * r)``.

    Pixels with ``r <= guard_radius_px`` have zero reference weight.  A disk
    support also zeros pixels outside the radius ``(support_width_px - 1) / 2``;
    a square support retains the full square footprint.  The latter exists to
    reproduce the archived implementation exactly, not as the preferred
    guarded stencil.
    """

    context_id: str
    support_width_px: int
    shape_n: float
    rate_mu: float
    guard_radius_px: float = 0.0
    support_geometry: SupportGeometry = "disk"
    boundary_mode: BoundaryMode = "valid_renormalized_zero"
    epsilon: float = 1e-6
    scale_floor: float = 0.0

    def __post_init__(self) -> None:
        import math

        if not self.context_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in self.context_id
        ):
            raise ValueError(
                "context_id must use lowercase ASCII letters, digits, and underscores"
            )
        width = int(self.support_width_px)
        if width != self.support_width_px or width < 3 or width % 2 != 1:
            raise ValueError("support_width_px must be an odd integer >= 3")
        if not math.isfinite(float(self.shape_n)) or float(self.shape_n) < 1.0:
            raise ValueError("shape_n must be finite and >= 1")
        if not math.isfinite(float(self.rate_mu)) or float(self.rate_mu) <= 0.0:
            raise ValueError("rate_mu must be finite and positive")
        if (
            not math.isfinite(float(self.guard_radius_px))
            or float(self.guard_radius_px) < 0.0
        ):
            raise ValueError("guard_radius_px must be finite and nonnegative")
        if self.support_geometry not in {"disk", "square"}:
            raise ValueError("support_geometry must be disk or square")
        if self.boundary_mode not in {"valid_renormalized_zero", "reflect"}:
            raise ValueError(
                "boundary_mode must be valid_renormalized_zero or reflect"
            )
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(float(self.scale_floor)) or float(self.scale_floor) < 0.0:
            raise ValueError("scale_floor must be finite and nonnegative")

        half_width = (width - 1) / 2.0
        maximum_radius = (
            half_width
            if self.support_geometry == "disk"
            else (2.0**0.5) * half_width
        )
        if float(self.guard_radius_px) >= maximum_radius:
            raise ValueError("guard_radius_px leaves no reference pixels")

        legacy_geometry = (
            width == 23
            and float(self.shape_n) == 9.0
            and math.isclose(
                float(self.rate_mu), 8.0 / 35.0, rel_tol=0.0, abs_tol=1e-15
            )
            and float(self.guard_radius_px) == 0.0
            and self.support_geometry == "square"
        )
        if self.boundary_mode == "reflect" and not legacy_geometry:
            raise ValueError(
                "reflect is reserved for the archived legacy_exact geometry; "
                "modern references must use valid_renormalized_zero"
            )

    @classmethod
    def from_mode(
        cls,
        context_id: str,
        *,
        support_width_px: int,
        shape_n: float,
        mode_radius_px: float,
        guard_radius_px: float = 0.0,
        support_geometry: SupportGeometry = "disk",
        boundary_mode: BoundaryMode = "valid_renormalized_zero",
        epsilon: float = 1e-6,
        scale_floor: float = 0.0,
    ) -> "GammaReferenceSpec":
        """Construct a reference from the positive radial mode ``(n-1)/mu``."""
        import math

        shape = float(shape_n)
        mode = float(mode_radius_px)
        if not math.isfinite(shape) or shape <= 1.0:
            raise ValueError("from_mode requires finite shape_n > 1")
        if not math.isfinite(mode) or mode <= 0.0:
            raise ValueError("mode_radius_px must be finite and positive")
        return cls(
            context_id=context_id,
            support_width_px=support_width_px,
            shape_n=shape,
            rate_mu=(shape - 1.0) / mode,
            guard_radius_px=guard_radius_px,
            support_geometry=support_geometry,
            boundary_mode=boundary_mode,
            epsilon=epsilon,
            scale_floor=scale_floor,
        )

    @classmethod
    def legacy_exact(cls, *, epsilon: float = 64.0) -> "GammaReferenceSpec":
        """Archived 23x23, n=9, nominal-mode-35, center-zero operator.

        This is a square-truncated, center-suppressed Gamma reference.  Its
        nominal mode lies outside the sampled support, and it has no explicit
        multi-pixel guard band.
        """
        return cls.from_mode(
            "legacy_exact_n9_mode35_w23",
            support_width_px=23,
            shape_n=9.0,
            mode_radius_px=35.0,
            guard_radius_px=0.0,
            support_geometry="square",
            boundary_mode="reflect",
            epsilon=epsilon,
            scale_floor=0.0,
        )

    @property
    def nominal_mode_radius_px(self) -> float | None:
        """Return the positive radial mode, or ``None`` for ``shape_n == 1``."""
        if float(self.shape_n) == 1.0:
            return None
        return (float(self.shape_n) - 1.0) / float(self.rate_mu)


@dataclass(frozen=True)
class GammaLocalStandardizationResult:
    """Device-resident signed local-standardization output."""

    values: Any
    reference_kernel: Any
    local_mean: Any | None
    local_std: Any | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class GammaCFARResult:
    """Device-resident direct-threshold CFAR output."""

    score: Any
    mask: Any
    threshold_z: float
    reference_kernel: Any
    local_mean: Any | None
    local_std: Any | None
    diagnostics: dict[str, Any]


def gamma_reference_kernel(
    spec: GammaReferenceSpec,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build the normalized finite Gamma reference on ``device``."""
    torch = _torch()
    resolved_dtype = torch.float32 if dtype is None else dtype
    if resolved_dtype not in {torch.float32, torch.float64}:
        raise ValueError("Gamma reference dtype must be torch.float32 or torch.float64")
    width = int(spec.support_width_px)
    half = width // 2
    coordinates = torch.arange(
        -half,
        half + 1,
        dtype=resolved_dtype,
        device=device,
    )
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    weights = radius.pow(float(spec.shape_n) - 1.0) * torch.exp(
        -float(spec.rate_mu) * radius
    )
    include = radius > float(spec.guard_radius_px)
    if spec.support_geometry == "disk":
        include = include & (radius <= float(half))
    weights = torch.where(include, weights, torch.zeros_like(weights))
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("Gamma reference has no finite positive mass")
    return weights / total


def _as_video_tensor(values: Any, *, device: Any = None) -> Any:
    torch = _torch()
    if torch.is_tensor(values):
        tensor = values if device is None else values.to(device=device)
        if tensor.dtype not in {torch.float32, torch.float64}:
            tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim != 3 or tensor.numel() == 0:
        raise ValueError("values must be a non-empty T,Y,X array")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("values must be finite")
    return tensor


def _standardize_chunk(
    values: Any, kernel: Any, spec: GammaReferenceSpec
) -> tuple[Any, Any, Any]:
    torch = _torch()
    import torch.nn.functional as functional

    half = int(spec.support_width_px) // 2
    if spec.boundary_mode == "reflect" and (
        int(values.shape[-2]) <= half or int(values.shape[-1]) <= half
    ):
        raise ValueError(
            "reflect padding requires each spatial dimension to exceed the kernel half-width"
        )
    frames = values.unsqueeze(1)
    weight = kernel.reshape(1, 1, *kernel.shape)
    if spec.boundary_mode == "reflect":
        padding = (half, half, half, half)
        padded = functional.pad(frames, padding, mode="reflect")
        local_mean = functional.conv2d(padded, weight).squeeze(1)
        padded_square = functional.pad(frames.square(), padding, mode="reflect")
        local_second = functional.conv2d(padded_square, weight).squeeze(1)
    else:
        # Zero outside the field of view, then divide by the reference mass
        # that overlaps real pixels.  This preserves a constant field at the
        # border without reflecting the CUT into its own reference stencil.
        reference_mass = functional.conv2d(
            torch.ones(
                (1, 1, int(values.shape[-2]), int(values.shape[-1])),
                dtype=values.dtype,
                device=values.device,
            ),
            weight,
            padding=half,
        )
        if bool((reference_mass <= 0.0).any()):
            raise ValueError(
                "reference stencil has zero valid mass at one or more pixels"
            )
        local_mean = (
            functional.conv2d(frames, weight, padding=half) / reference_mass
        ).squeeze(1)
        local_second = (
            functional.conv2d(frames.square(), weight, padding=half)
            / reference_mass
        ).squeeze(1)
    local_variance = (local_second - local_mean.square()).clamp_min(0.0)
    local_std = torch.sqrt(local_variance)
    denominator = torch.maximum(
        local_std,
        torch.as_tensor(
            float(spec.scale_floor), dtype=values.dtype, device=values.device
        ),
    ) + float(spec.epsilon)
    standardized = (values - local_mean) / denominator
    return standardized, local_mean, local_std


def gamma_local_standardization(
    values: Any,
    spec: GammaReferenceSpec,
    *,
    device: Any = None,
    chunk_frames: int | None = None,
    return_statistics: bool = True,
) -> GammaLocalStandardizationResult:
    """Compute signed Gamma-weighted local standardization.

    ``values`` and every tensor in the result remain on the resolved input
    device.  Chunking is over independent frames and therefore changes memory
    use, not the mathematical operator.
    """
    torch = _torch()
    video = _as_video_tensor(values, device=device)
    frames_per_chunk = len(video) if chunk_frames is None else int(chunk_frames)
    if frames_per_chunk < 1:
        raise ValueError("chunk_frames must be positive when provided")
    kernel = gamma_reference_kernel(spec, device=video.device, dtype=video.dtype)

    score_parts = []
    mean_parts = []
    std_parts = []
    with torch.no_grad():
        for start in range(0, len(video), frames_per_chunk):
            stop = min(start + frames_per_chunk, len(video))
            score, local_mean, local_std = _standardize_chunk(
                video[start:stop], kernel, spec
            )
            score_parts.append(score)
            if return_statistics:
                mean_parts.append(local_mean)
                std_parts.append(local_std)

    standardized = torch.cat(score_parts, dim=0)
    local_mean_result = torch.cat(mean_parts, dim=0) if return_statistics else None
    local_std_result = torch.cat(std_parts, dim=0) if return_statistics else None
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("Gamma local standardization produced non-finite values")
    diagnostics = {
        "context_id": spec.context_id,
        "operator": "signed_gamma_local_standardization",
        "reference_family": "radial_gamma",
        "support_width_px": int(spec.support_width_px),
        "support_geometry": spec.support_geometry,
        "guard_radius_px": float(spec.guard_radius_px),
        "shape_n": float(spec.shape_n),
        "rate_mu": float(spec.rate_mu),
        "nominal_mode_radius_px": spec.nominal_mode_radius_px,
        "boundary_mode": spec.boundary_mode,
        "edge_reference_renormalized": (
            spec.boundary_mode == "valid_renormalized_zero"
        ),
        "epsilon": float(spec.epsilon),
        "scale_floor": float(spec.scale_floor),
        "chunk_frames": int(frames_per_chunk),
        "device": str(video.device),
        "dtype": str(video.dtype).removeprefix("torch."),
        "signed_output": True,
        "probability_of_false_alarm_claimed": False,
    }
    return GammaLocalStandardizationResult(
        values=standardized,
        reference_kernel=kernel,
        local_mean=local_mean_result,
        local_std=local_std_result,
        diagnostics=diagnostics,
    )


def gamma_cfar(
    values: Any,
    spec: GammaReferenceSpec,
    *,
    threshold_z: float,
    device: Any = None,
    chunk_frames: int | None = None,
    return_statistics: bool = True,
) -> GammaCFARResult:
    """Apply a one-sided direct threshold to signed Gamma-LS scores.

    ``threshold_z`` is an operating threshold, not an asserted false-alarm
    probability.  A false-alarm interpretation requires a separate, frozen
    calibration and an appropriate negative/reference population.
    """
    import math

    threshold = float(threshold_z)
    if not math.isfinite(threshold):
        raise ValueError("threshold_z must be finite")
    standardized = gamma_local_standardization(
        values,
        spec,
        device=device,
        chunk_frames=chunk_frames,
        return_statistics=return_statistics,
    )
    mask = standardized.values > threshold
    diagnostics = {
        **standardized.diagnostics,
        "decision": "one_sided_score_strictly_greater_than_threshold",
        "threshold_z": threshold,
    }
    return GammaCFARResult(
        score=standardized.values,
        mask=mask,
        threshold_z=threshold,
        reference_kernel=standardized.reference_kernel,
        local_mean=standardized.local_mean,
        local_std=standardized.local_std,
        diagnostics=diagnostics,
    )


__all__ = [
    "GammaCFARResult",
    "GammaLocalStandardizationResult",
    "GammaReferenceSpec",
    "gamma_cfar",
    "gamma_local_standardization",
    "gamma_reference_kernel",
]
