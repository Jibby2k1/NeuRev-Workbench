"""Dark-soma candidate geometry from a quiet two-dimensional baseline.

This module detects provisional anatomical anchors, not calcium events.  A
candidate is a dark local core relative to a broader neighbourhood.  Event
evidence (for example, positive CFAR activity) can later be measured in the
returned perisomatic ring without calling that evidence the soma itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


@dataclass(frozen=True)
class DarkSomaZoneConfig:
    """Parameters for quiet-baseline dark-core detection and zone geometry."""

    inner_sigma: float = 1.0
    outer_sigma: float = 5.0
    z_threshold: float = 3.0
    min_distance: float = 6.0
    border: int = 10
    max_zones: int = 300
    core_radius: float = 4.0
    ring_inner_radius: float = 4.0
    ring_outer_radius: float = 10.0
    saturation_threshold: float | None = None
    saturation_flag_fraction: float = 0.1
    saturation_penalty: float = 0.0
    robust_scale_epsilon: float = 1e-6

    def validate(self) -> None:
        numeric = {
            "inner_sigma": self.inner_sigma,
            "outer_sigma": self.outer_sigma,
            "z_threshold": self.z_threshold,
            "min_distance": self.min_distance,
            "core_radius": self.core_radius,
            "ring_inner_radius": self.ring_inner_radius,
            "ring_outer_radius": self.ring_outer_radius,
            "saturation_flag_fraction": self.saturation_flag_fraction,
            "saturation_penalty": self.saturation_penalty,
            "robust_scale_epsilon": self.robust_scale_epsilon,
        }
        for name, value in numeric.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.inner_sigma <= 0 or self.outer_sigma <= self.inner_sigma:
            raise ValueError("require 0 < inner_sigma < outer_sigma")
        if self.z_threshold < 0:
            raise ValueError("z_threshold must be non-negative")
        if self.min_distance < 0:
            raise ValueError("min_distance must be non-negative")
        if not isinstance(self.border, Integral) or isinstance(self.border, bool):
            raise ValueError("border must be an integer")
        if self.border < 0:
            raise ValueError("border must be non-negative")
        if not isinstance(self.max_zones, Integral) or isinstance(self.max_zones, bool):
            raise ValueError("max_zones must be an integer")
        if self.max_zones < 0:
            raise ValueError("max_zones must be non-negative")
        if self.core_radius <= 0:
            raise ValueError("core_radius must be positive")
        if self.ring_inner_radius < self.core_radius:
            raise ValueError("ring_inner_radius must be at least core_radius")
        if self.ring_outer_radius <= self.ring_inner_radius:
            raise ValueError("ring_outer_radius must exceed ring_inner_radius")
        if not 0 <= self.saturation_flag_fraction <= 1:
            raise ValueError("saturation_flag_fraction must be in [0, 1]")
        if self.saturation_penalty < 0:
            raise ValueError("saturation_penalty must be non-negative")
        if self.robust_scale_epsilon <= 0:
            raise ValueError("robust_scale_epsilon must be positive")
        if self.saturation_threshold is not None and not np.isfinite(
            self.saturation_threshold
        ):
            raise ValueError("saturation_threshold must be finite when set")


@dataclass(frozen=True)
class DarkSomaZone:
    """Metadata for one provisional dark-core anatomical anchor."""

    zone_id: int
    y: int
    x: int
    contrast: float
    robust_z: float
    selection_score: float
    baseline_intensity: float
    saturation_fraction: float
    saturation_flagged: bool
    core_pixel_count: int
    ring_pixel_count: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class DarkSomaZones:
    """Detected anatomy and union masks for downstream event association."""

    zones: tuple[DarkSomaZone, ...]
    contrast: np.ndarray
    robust_z: np.ndarray
    core_mask: np.ndarray
    ring_mask: np.ndarray
    robust_center: float
    robust_scale: float

    def metadata(self) -> list[dict[str, int | float | bool]]:
        return [zone.to_dict() for zone in self.zones]


def detect_dark_soma_zones(
    baseline: np.ndarray,
    config: DarkSomaZoneConfig | None = None,
) -> DarkSomaZones:
    """Detect dark-core candidates in one quiet-baseline projection.

    The returned full-frame arrays are two-dimensional only.  No temporal
    stack or per-zone image cube is created.
    """

    cfg = config or DarkSomaZoneConfig()
    cfg.validate()
    image = np.asarray(baseline)
    if image.ndim != 2:
        raise ValueError(f"baseline must be 2-D, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("baseline must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("baseline must have a numeric dtype")

    work = image.astype(np.float32, copy=False)
    if not np.all(np.isfinite(work)):
        raise ValueError("baseline must contain only finite values")

    contrast = gaussian_filter(work, cfg.outer_sigma) - gaussian_filter(
        work, cfg.inner_sigma
    )
    center = float(np.median(contrast))
    mad = float(np.median(np.abs(contrast - center)))
    scale = max(1.4826 * mad, cfg.robust_scale_epsilon)
    robust_z = (contrast - center) / scale

    core_mask = np.zeros(image.shape, dtype=bool)
    ring_mask = np.zeros(image.shape, dtype=bool)
    if cfg.max_zones == 0:
        return _result((), contrast, robust_z, core_mask, ring_mask, center, scale)

    local_max = robust_z == maximum_filter(robust_z, size=3, mode="nearest")
    candidate_mask = local_max & (robust_z >= cfg.z_threshold)
    _exclude_border(candidate_mask, cfg.border)
    coordinates = np.argwhere(candidate_mask)

    candidates: list[tuple[float, float, int, int, float, bool]] = []
    for y_value, x_value in coordinates:
        y, x = int(y_value), int(x_value)
        fraction = _saturation_fraction(work, y, x, cfg)
        flagged = fraction >= cfg.saturation_flag_fraction and fraction > 0.0
        score = float(robust_z[y, x] - cfg.saturation_penalty * fraction)
        candidates.append(
            (score, float(robust_z[y, x]), y, x, fraction, flagged)
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

    accepted: list[tuple[float, float, int, int, float, bool]] = []
    minimum_squared = cfg.min_distance**2
    for candidate in candidates:
        _, _, y, x, _, _ = candidate
        if all(
            (y - prior[2]) ** 2 + (x - prior[3]) ** 2 >= minimum_squared
            for prior in accepted
        ):
            accepted.append(candidate)
            if len(accepted) >= cfg.max_zones:
                break

    zones: list[DarkSomaZone] = []
    for zone_id, (score, z_value, y, x, fraction, flagged) in enumerate(accepted):
        core, ring = _zone_masks(image.shape, y, x, cfg)
        core_mask |= core
        ring_mask |= ring
        zones.append(
            DarkSomaZone(
                zone_id=zone_id,
                y=y,
                x=x,
                contrast=float(contrast[y, x]),
                robust_z=z_value,
                selection_score=score,
                baseline_intensity=float(work[y, x]),
                saturation_fraction=fraction,
                saturation_flagged=flagged,
                core_pixel_count=int(np.count_nonzero(core)),
                ring_pixel_count=int(np.count_nonzero(ring)),
            )
        )

    return _result(zones, contrast, robust_z, core_mask, ring_mask, center, scale)


def _result(
    zones: Sequence[DarkSomaZone],
    contrast: np.ndarray,
    robust_z: np.ndarray,
    core_mask: np.ndarray,
    ring_mask: np.ndarray,
    center: float,
    scale: float,
) -> DarkSomaZones:
    return DarkSomaZones(
        zones=tuple(zones),
        contrast=np.asarray(contrast, dtype=np.float32),
        robust_z=np.asarray(robust_z, dtype=np.float32),
        core_mask=core_mask,
        ring_mask=ring_mask,
        robust_center=center,
        robust_scale=scale,
    )


def _exclude_border(mask: np.ndarray, border: int) -> None:
    if border == 0:
        return
    height, width = mask.shape
    if border * 2 >= height or border * 2 >= width:
        mask.fill(False)
        return
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False


def _zone_masks(
    shape: tuple[int, int],
    y: int,
    x: int,
    config: DarkSomaZoneConfig,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    distance_squared = (yy - y) ** 2 + (xx - x) ** 2
    core = distance_squared <= config.core_radius**2
    ring = (distance_squared > config.ring_inner_radius**2) & (
        distance_squared <= config.ring_outer_radius**2
    )
    return core, ring


def _saturation_fraction(
    image: np.ndarray,
    y: int,
    x: int,
    config: DarkSomaZoneConfig,
) -> float:
    if config.saturation_threshold is None:
        return 0.0
    radius = int(np.ceil(config.ring_outer_radius))
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    neighbourhood = (yy - y) ** 2 + (xx - x) ** 2 <= config.ring_outer_radius**2
    values = image[y0:y1, x0:x1][neighbourhood]
    return float(np.mean(values >= config.saturation_threshold))
