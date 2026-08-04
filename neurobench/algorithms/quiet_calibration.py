"""Training-only robust coordinate and empirical quiet-tail calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neurobench.algorithms.activity_feature_bank import bounded_square


@dataclass
class QuietRobustStandardizer:
    """Median/MAD calibration reusable across downstream energy mappings."""

    mode: Literal["global", "per_pixel"] = "per_pixel"
    floor_percentile: float = 10.0
    minimum_samples: int = 3
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    scale_floor_: float | None = None
    sample_count_: np.ndarray | int | None = None

    def fit(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray,
    ) -> "QuietRobustStandardizer":
        array = np.asarray(values, dtype=np.float64)
        if not array.size or not np.isfinite(array).all():
            raise ValueError("values must be finite and non-empty")
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape == (array.shape[0],):
            mask = np.broadcast_to(mask[:, None, None], array.shape)
        else:
            try:
                mask = np.broadcast_to(mask, array.shape)
            except ValueError as exc:
                raise ValueError("valid_mask does not align with values") from exc
        if not np.any(mask):
            raise ValueError("valid_mask selects no quiet samples")
        if self.mode == "global":
            selected = array[mask]
            center = np.asarray(float(np.median(selected)))
            raw_scale = np.asarray(
                float(1.4826 * np.median(np.abs(selected - center)))
            )
            counts: np.ndarray | int = int(selected.size)
        elif self.mode == "per_pixel":
            if array.ndim != 3:
                raise ValueError("per_pixel mode requires TYX values")
            masked = np.where(mask, array, np.nan)
            counts = np.sum(mask, axis=0)
            if np.any(counts < int(self.minimum_samples)):
                raise ValueError("insufficient quiet samples for at least one pixel")
            center = np.nanmedian(masked, axis=0)
            raw_scale = 1.4826 * np.nanmedian(
                np.abs(masked - center[None]), axis=0
            )
        else:
            raise ValueError("mode must be global or per_pixel")
        positive = np.asarray(raw_scale)[np.asarray(raw_scale) > 0]
        floor = (
            float(np.percentile(positive, float(self.floor_percentile)))
            if positive.size
            else 1.0
        )
        floor = max(floor, np.finfo(np.float32).eps)
        self.center_ = np.asarray(center, dtype=np.float32)
        self.scale_ = np.maximum(raw_scale, floor).astype(np.float32)
        self.scale_floor_ = floor
        self.sample_count_ = counts
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("standardizer must be fit before transform")
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("values must be finite")
        try:
            result = (array - self.center_) / self.scale_
        except ValueError as exc:
            raise ValueError("values do not align with fitted calibration") from exc
        return np.asarray(result, dtype=np.float32)

    def fit_transform(
        self, values: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        return self.fit(values, valid_mask).transform(values)

    def to_dict(self) -> dict[str, Any]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("standardizer has not been fit")
        counts = self.sample_count_
        return {
            "mode": self.mode,
            "floor_percentile": float(self.floor_percentile),
            "minimum_samples": int(self.minimum_samples),
            "scale_floor": float(self.scale_floor_),
            "center_shape": list(self.center_.shape),
            "scale_shape": list(self.scale_.shape),
            "sample_count_min": (
                int(np.min(counts)) if isinstance(counts, np.ndarray) else int(counts)
            ),
            "sample_count_max": (
                int(np.max(counts)) if isinstance(counts, np.ndarray) else int(counts)
            ),
        }


@dataclass
class EmpiricalQuietTail:
    """Finite add-one empirical survival and surprise calibration."""

    sorted_quiet_: np.ndarray | None = None

    def fit(self, quiet_energy: np.ndarray) -> "EmpiricalQuietTail":
        values = np.asarray(quiet_energy, dtype=np.float64).ravel()
        if not values.size or not np.isfinite(values).all():
            raise ValueError("quiet_energy must be finite and non-empty")
        self.sorted_quiet_ = np.sort(values)
        return self

    def survival_probability(self, values: np.ndarray) -> np.ndarray:
        if self.sorted_quiet_ is None:
            raise RuntimeError("tail calibrator must be fit before transform")
        query = np.asarray(values, dtype=np.float64)
        if not np.isfinite(query).all():
            raise ValueError("values must be finite")
        first_ge = np.searchsorted(self.sorted_quiet_, query, side="left")
        count_ge = len(self.sorted_quiet_) - first_ge
        return ((count_ge + 1.0) / (len(self.sorted_quiet_) + 1.0)).astype(
            np.float64
        )

    def surprise(
        self,
        values: np.ndarray,
        log_base: Literal["e", 10] = "e",
    ) -> np.ndarray:
        probability = self.survival_probability(values)
        if log_base == "e":
            result = -np.log(probability)
        elif log_base == 10:
            result = -np.log10(probability)
        else:
            raise ValueError("log_base must be 'e' or 10")
        return np.asarray(result, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.sorted_quiet_ is None:
            raise RuntimeError("tail calibrator has not been fit")
        return {
            "sample_count": int(len(self.sorted_quiet_)),
            "minimum": float(self.sorted_quiet_[0]),
            "maximum": float(self.sorted_quiet_[-1]),
            "add_one_smoothing": True,
            "maximum_natural_surprise": float(
                -np.log(1.0 / (len(self.sorted_quiet_) + 1.0))
            ),
        }


def raw_square(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return np.square(array, dtype=np.float32)


def huber_energy(values: np.ndarray, delta: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    threshold = float(delta)
    if threshold <= 0 or not np.isfinite(array).all():
        raise ValueError("finite values and positive delta are required")
    magnitude = np.abs(array)
    return np.where(
        magnitude <= threshold,
        0.5 * magnitude * magnitude,
        threshold * (magnitude - 0.5 * threshold),
    ).astype(np.float32)


def group_energy(
    coordinates: np.ndarray,
    *,
    axis: int = 0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(coordinates, dtype=np.float32)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("coordinates must be finite and non-empty")
    count = array.shape[axis]
    coefficients = (
        np.ones(count, dtype=np.float32)
        if weights is None
        else np.asarray(weights, dtype=np.float32)
    )
    if (
        coefficients.shape != (count,)
        or np.any(coefficients < 0)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("group weights must be finite nonnegative and aligned")
    shape = [1] * array.ndim
    shape[axis] = count
    return np.sum(
        np.square(array, dtype=np.float32) * coefficients.reshape(shape),
        axis=axis,
        dtype=np.float32,
    )


def energy_mapping_bank(
    standardized: np.ndarray,
    *,
    bounded_kappa: float,
    huber_delta: float,
) -> dict[str, np.ndarray]:
    """Return signed-preserving input plus declared scalar energy mappings."""
    values = np.asarray(standardized, dtype=np.float32)
    return {
        "signed": values.copy(),
        "absolute": np.abs(values).astype(np.float32),
        "raw_square": raw_square(values),
        "bounded_square": bounded_square(values, float(bounded_kappa)),
        "huber_energy": huber_energy(values, float(huber_delta)),
    }
