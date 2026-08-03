"""Whitening, fitting, and canonical orientation for event-weighted CS-Parzen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobench.algorithms.pairwise_separation import (
    SeparationFit,
    Whitening2D,
    cs_parzen_independence,
    fit_cs_parzen_ica,
)

from .sample_weights import WeightedPairBatch


@dataclass(frozen=True)
class CanonicalFit:
    fit: SeparationFit
    whitening: Whitening2D
    common_component: int
    innovation_component: int
    innovation_sign: int
    effective_common_direction: np.ndarray
    effective_innovation_direction: np.ndarray
    cosine_to_common: float
    cosine_to_derivative: float
    angle_degrees: float


def fit_weighted_whitening_2d(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    eigenvalue_floor_ratio: float = 1e-6,
) -> tuple[np.ndarray, Whitening2D]:
    x = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or w.shape != (len(x),):
        raise ValueError("samples [N,2] and weights [N] must align")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("samples/weights must be finite and weights nonnegative")
    total = float(w.sum())
    if total <= 0 or not 0 < eigenvalue_floor_ratio < 1:
        raise ValueError("positive weight sum and valid eigenvalue floor required")
    normalized = w / total
    mean = normalized @ x
    centered = x - mean
    covariance = (centered * normalized[:, None]).T @ centered
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    largest = max(float(eigenvalues[0]), np.finfo(float).eps)
    floor = largest * eigenvalue_floor_ratio
    floored = np.maximum(eigenvalues, floor)
    condition = largest / max(float(eigenvalues[-1]), np.finfo(float).eps)
    whitening = np.diag(1 / np.sqrt(floored)) @ vectors.T
    dewhitening = vectors @ np.diag(np.sqrt(floored))
    whitened = (whitening @ centered.T).astype(np.float64)
    fit = Whitening2D(
        mean=mean,
        covariance=covariance,
        whitening=whitening,
        dewhitening=dewhitening,
        eigenvalues=eigenvalues,
        condition_number=float(condition),
        identifiable=bool(eigenvalues[-1] > floor and condition <= 1e8),
    )
    if not np.isfinite(whitening).all():
        raise ValueError("whitening matrix is non-finite")
    return whitened, fit


def apply_whitening(samples: np.ndarray, whitening: Whitening2D) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("samples must have shape [N,2]")
    result = whitening.whitening @ (values.T - whitening.mean[:, None])
    if not np.isfinite(result).all():
        raise ValueError("whitened samples are non-finite")
    return result


def canonicalize_fit(fit: SeparationFit, whitening: Whitening2D) -> CanonicalFit:
    directions = fit.demixing @ whitening.whitening
    unit = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    common = np.asarray([1.0, 1.0]) / np.sqrt(2)
    derivative = np.asarray([-1.0, 1.0]) / np.sqrt(2)
    common_component = int(np.argmax(np.abs(unit @ common)))
    innovation_component = 1 - common_component
    innovation_sign = 1 if float(unit[innovation_component] @ derivative) >= 0 else -1
    innovation_direction = unit[innovation_component] * innovation_sign
    common_sign = 1 if float(unit[common_component] @ common) >= 0 else -1
    common_direction = unit[common_component] * common_sign
    angle = float(
        np.rad2deg(
            np.arctan2(innovation_direction[1], innovation_direction[0])
        )
        % 180
    )
    return CanonicalFit(
        fit=fit,
        whitening=whitening,
        common_component=common_component,
        innovation_component=innovation_component,
        innovation_sign=innovation_sign,
        effective_common_direction=common_direction,
        effective_innovation_direction=innovation_direction,
        cosine_to_common=float(abs(common_direction @ common)),
        cosine_to_derivative=float(innovation_direction @ derivative),
        angle_degrees=angle,
    )


def fit_weighted_batch(
    screen: WeightedPairBatch,
    confirmation: WeightedPairBatch,
    natural_whitening_samples: np.ndarray,
    *,
    whitening_mode: str,
    bandwidth: float,
    block_rows: int,
    coarse_step_degrees: float,
    refine_half_width_degrees: float,
    refine_step_degrees: float,
    eigenvalue_floor_ratio: float,
    kernel_dtype: np.dtype = np.float64,
) -> CanonicalFit:
    natural_weights = np.ones(len(natural_whitening_samples), dtype=np.float64)
    if whitening_mode == "natural_fixed":
        _, whitening = fit_weighted_whitening_2d(
            natural_whitening_samples,
            natural_weights,
            eigenvalue_floor_ratio=eigenvalue_floor_ratio,
        )
    elif whitening_mode == "weighted":
        _, whitening = fit_weighted_whitening_2d(
            confirmation.samples,
            confirmation.weights,
            eigenvalue_floor_ratio=eigenvalue_floor_ratio,
        )
    else:
        raise ValueError("whitening_mode must be natural_fixed or weighted")
    if not whitening.identifiable:
        raise ValueError(
            f"whitening covariance is unidentifiable (condition={whitening.condition_number})"
        )
    screen_z = apply_whitening(screen.samples, whitening)
    confirm_z = apply_whitening(confirmation.samples, whitening)
    fit = fit_cs_parzen_ica(
        screen_z,
        confirm_z,
        bandwidth=bandwidth,
        block_rows=block_rows,
        screen_step_degrees=coarse_step_degrees,
        refine_half_width_degrees=refine_half_width_degrees,
        refine_step_degrees=refine_step_degrees,
        screen_weights=screen.weights,
        confirm_weights=confirmation.weights,
        kernel_dtype=kernel_dtype,
    )
    return canonicalize_fit(fit, whitening)


def natural_objective(
    canonical: CanonicalFit,
    natural_samples: np.ndarray,
    *,
    bandwidth: float,
    block_rows: int,
    kernel_dtype: np.dtype = np.float64,
) -> float:
    whitened = apply_whitening(natural_samples, canonical.whitening)
    outputs = canonical.fit.demixing @ whitened
    objective, _ = cs_parzen_independence(
        outputs, bandwidth, block_rows=block_rows, kernel_dtype=kernel_dtype
    )
    return objective


def whitening_diagnostics(whitening: Whitening2D) -> dict[str, Any]:
    largest = max(float(whitening.eigenvalues[0]), np.finfo(float).eps)
    floor = largest * 1e-6
    return {
        "mean": whitening.mean.tolist(),
        "covariance": whitening.covariance.tolist(),
        "condition_number": whitening.condition_number,
        "eigenvalues": whitening.eigenvalues.tolist(),
        "eigenvalue_floor": floor,
        "identifiable": whitening.identifiable,
    }
