"""Frozen, pair-safe interpretable comparators for the JEPA pilot."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter


class JEPAComparatorError(ValueError):
    """Raised when a frozen comparator cannot preserve its score contract."""


HANDCRAFTED_COMPARATOR_VERSION = (
    "carrier_context_kinetic_source_off_component_calibration_v1"
)


def _validated_movie(movie: np.ndarray) -> np.ndarray:
    # The historical implementation receives float32 clips.  Preserve that
    # arithmetic exactly so the regression comparator cannot drift with the
    # storage dtype of an otherwise identical uint16 source.
    video = np.asarray(movie, dtype=np.float32)
    if video.ndim != 3 or len(video) < 2 or not np.isfinite(video).all():
        raise JEPAComparatorError("handcrafted comparator requires a finite TYX movie")
    return video


def handcrafted_components(movie: np.ndarray) -> dict[str, np.ndarray]:
    """Return the frozen carrier, spatial-context, and kinetic maps.

    On the registered 32-frame clips these definitions are exactly the ones
    used by ``major_next_steps._score_maps``.  The shorter-kernel branch exists
    only so the tiny non-scientific smoke runner can exercise the interface.
    """

    video = _validated_movie(movie)
    carrier = np.std(video, axis=0)
    spatial_context = gaussian_filter(carrier, 1.2) - gaussian_filter(carrier, 4.0)
    centered = video - np.median(video, axis=0, keepdims=True)
    kernel_length = min(18, len(video))
    axis = np.arange(kernel_length, dtype=np.float64)
    kernel = np.exp(-axis / 5.0) * (1.0 - np.exp(-axis / 1.5))
    norm = float(np.linalg.norm(kernel))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise JEPAComparatorError("frozen kinetic kernel is degenerate")
    kernel /= norm
    kinetic = np.max(
        np.stack(
            [
                np.sum(
                    centered[start : start + kernel_length]
                    * kernel[:, None, None],
                    axis=0,
                )
                for start in range(len(video) - kernel_length + 1)
            ]
        ),
        axis=0,
    )
    return {
        "carrier": carrier,
        "spatial_context": spatial_context,
        "kinetic": kinetic,
    }


def _source_off_center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    # This 1e-6 lower bound is part of the historical comparator definition,
    # not the raw-video injection-amplitude contract.
    return center, max(scale, 1e-6)


@dataclass(frozen=True)
class _FittedHandcraftedComparator:
    calibration: tuple[tuple[str, float, float], ...]

    def __call__(self, movie: np.ndarray) -> np.ndarray:
        components = handcrafted_components(movie)
        normalized: list[np.ndarray] = []
        for name, center, scale in self.calibration:
            if name not in components:
                raise JEPAComparatorError(f"fitted component is missing: {name}")
            normalized.append((components[name] - center) / scale)
        return normalized[0] + normalized[1] + normalized[2]


@dataclass(frozen=True)
class FrozenHandcraftedComparator:
    """Historical combined score with source-off-only paired calibration."""

    version: str = HANDCRAFTED_COMPARATOR_VERSION

    def __call__(self, movie: np.ndarray) -> np.ndarray:
        """Return the historical self-calibrated map for regression checks."""

        components = handcrafted_components(movie)
        normalized: list[np.ndarray] = []
        for values in components.values():
            center, scale = _source_off_center_scale(values)
            normalized.append((values - center) / scale)
        # Retain the historical left-to-right NumPy promotion/rounding order.
        return normalized[0] + normalized[1] + normalized[2]

    def fit_source_off(self, source_off: np.ndarray) -> _FittedHandcraftedComparator:
        """Fit using source-off only, returning a frozen map callable."""

        off_components = handcrafted_components(source_off)
        calibration: list[tuple[str, float, float]] = []
        for name, off_values in off_components.items():
            center, scale = _source_off_center_scale(off_values)
            calibration.append((name, center, scale))
        return _FittedHandcraftedComparator(tuple(calibration))

    def contract(self) -> dict[str, object]:
        return {
            "version": self.version,
            "carrier": "temporal_population_standard_deviation_ddof_0",
            "spatial_context": "gaussian_sigma_1.2_minus_gaussian_sigma_4.0_of_carrier",
            "kinetic": "maximum_valid_correlation_with_normalized_exp_t5_times_one_minus_exp_t1.5_length_18",
            "component_calibration": "source_off_spatial_median_and_normal_consistent_mad_floor_1e-6",
            "source_on_self_calibration": False,
            "combination": "equal_sum_of_three_source_off_calibrated_components",
        }


frozen_handcrafted_comparator = FrozenHandcraftedComparator()


__all__ = [
    "FrozenHandcraftedComparator",
    "HANDCRAFTED_COMPARATOR_VERSION",
    "JEPAComparatorError",
    "frozen_handcrafted_comparator",
    "handcrafted_components",
]
