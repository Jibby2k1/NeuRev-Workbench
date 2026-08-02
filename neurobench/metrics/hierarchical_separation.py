"""Metrics for explicit background/signal/artifact/noise decomposition."""
from __future__ import annotations

from typing import Any

import numpy as np


def _aligned(*arrays: np.ndarray) -> list[np.ndarray]:
    values = [np.asarray(array, dtype=np.float64) for array in arrays]
    if not values or any(
        not value.size or not np.isfinite(value).all() for value in values
    ):
        raise ValueError("metric arrays must be non-empty and finite")
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("metric arrays must have identical shapes")
    return values


def _normalized_squared_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = max(float(np.sum(reference**2)), np.finfo(float).eps)
    return float(np.sum((reference - estimate) ** 2) / denominator)


def _correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    left = reference.ravel()
    right = estimate.ravel()
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def stage1_leakage_metrics(
    true_background: np.ndarray,
    true_signal: np.ndarray,
    estimated_background: np.ndarray,
    dynamic_residual: np.ndarray,
) -> dict[str, Any]:
    """Measure Stage-1 B/S leakage on a synthetic or semi-synthetic fixture."""
    background, signal, estimate, residual = _aligned(
        true_background, true_signal, estimated_background, dynamic_residual
    )
    observation = background + signal
    background_error = estimate - background
    residual_error = residual - signal
    signal_energy = max(float(np.sum(signal**2)), np.finfo(float).eps)
    background_energy = max(float(np.sum(background**2)), np.finfo(float).eps)
    closure = observation - estimate - residual
    return {
        "background_nmse": _normalized_squared_error(background, estimate),
        "background_correlation": _correlation(background, estimate),
        "signal_residual_nmse": _normalized_squared_error(signal, residual),
        "signal_residual_correlation": _correlation(signal, residual),
        "signal_leakage_into_background": float(
            np.sum(background_error**2) / signal_energy
        ),
        "background_leakage_into_residual": float(
            np.sum(residual_error**2) / background_energy
        ),
        "closure_normalized_squared_error": float(
            np.sum(closure**2)
            / max(float(np.sum(observation**2)), np.finfo(float).eps)
        ),
        "closure_max_absolute": float(np.max(np.abs(closure))),
        "axes": "same_as_inputs",
    }
