"""Metrics for explicit B/S/A/N dependent-multiscale decompositions."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


CHANNELS = ("background", "structured_signal", "structured_artifact", "noise_candidate")


def _channels(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(values) != set(CHANNELS):
        raise ValueError(f"channels must be exactly {CHANNELS}")
    arrays = {key: np.asarray(values[key], dtype=np.float64) for key in CHANNELS}
    first = arrays[CHANNELS[0]]
    if not first.size or any(value.shape != first.shape or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("all channels must be non-empty, finite, and aligned")
    return arrays


def attribution_metrics(
    truth: Mapping[str, np.ndarray], estimate: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Return normalized attribution and direct energy-leakage matrices."""
    true = _channels(truth)
    predicted = _channels(estimate)
    correlation_energy = np.empty((4, 4), dtype=np.float64)
    direct_energy = np.empty((4, 4), dtype=np.float64)
    for row, true_id in enumerate(CHANNELS):
        denominator = max(float(np.sum(true[true_id] ** 2)), np.finfo(float).eps)
        for column, estimated_id in enumerate(CHANNELS):
            numerator = float(np.sum(true[true_id] * predicted[estimated_id]))
            estimate_energy = max(float(np.sum(predicted[estimated_id] ** 2)), np.finfo(float).eps)
            correlation_energy[row, column] = numerator**2 / (denominator * estimate_energy)
            direct_energy[row, column] = float(np.sum((true[true_id] - predicted[estimated_id]) ** 2) / denominator)
    diagonal = float(np.mean(np.diag(correlation_energy)))
    off_diagonal = float((correlation_energy.sum() - np.trace(correlation_energy)) / 12)
    return {
        "channel_order": list(CHANNELS),
        "normalized_attribution_matrix": correlation_energy.tolist(),
        "direct_energy_error_matrix": direct_energy.tolist(),
        "diagonality_margin": diagonal - off_diagonal,
        "primary_signal_leakage": float(np.sum(correlation_energy[1, [0, 2, 3]])),
    }


def closure_metrics(observation: np.ndarray, estimate: Mapping[str, np.ndarray]) -> dict[str, float]:
    values = np.asarray(observation, dtype=np.float64)
    predicted = _channels(estimate)
    if any(value.shape != values.shape for value in predicted.values()) or not np.isfinite(values).all():
        raise ValueError("observation and decomposition channels must align")
    residual = values - sum(predicted.values())
    scale = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    normalized = np.abs(residual) / scale
    return {
        "normalized_median": float(np.median(normalized)),
        "normalized_p95": float(np.quantile(normalized, 0.95)),
        "normalized_p99": float(np.quantile(normalized, 0.99)),
        "normalized_maximum": float(np.max(normalized)),
        "finite_value_rate": float(np.mean(np.isfinite(residual))),
    }
