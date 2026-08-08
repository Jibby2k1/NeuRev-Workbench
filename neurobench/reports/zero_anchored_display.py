"""Default zero-anchored display transforms for signed activity evidence.

These functions are presentation utilities. They never modify source arrays and
must not be used implicitly as detector inputs.
"""
from __future__ import annotations

import numpy as np


ZERO_ANCHORED_DISPLAY_CONTRACT = {
    "signed_evidence": "clip(x, 0, +inf) / global_positive_max",
    "temperature": "max(exp2(alpha * x) - 1, 0) / (exp2(alpha * global_positive_max) - 1)",
    "zero_reference": "x = 0 renders black",
    "scope": "visualization_only",
}


def global_positive_max(values: np.ndarray) -> float:
    """Return the finite movie-wide positive maximum for a signed map."""
    maximum = float(np.max(values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"signed evidence requires a finite positive maximum, got {maximum}")
    return maximum


def zero_anchored_signed(values: np.ndarray, *, positive_max: float) -> np.ndarray:
    """Map signed evidence to [0, 1], with all nonpositive values at black."""
    if not np.isfinite(positive_max) or positive_max <= 0.0:
        raise ValueError("positive_max must be finite and greater than zero")
    array = np.asarray(values, dtype=np.float32)
    return np.clip(array, np.float32(0.0), np.float32(positive_max)) / np.float32(positive_max)


def zero_anchored_exp2(
    values: np.ndarray, *, alpha: float, positive_max: float
) -> np.ndarray:
    """Apply the default zero-anchored exp2 display at one temperature."""
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and greater than zero")
    if not np.isfinite(positive_max) or positive_max <= 0.0:
        raise ValueError("positive_max must be finite and greater than zero")
    array = np.asarray(values, dtype=np.float32)
    positive = np.clip(array, np.float32(0.0), np.float32(positive_max))
    scale = np.float64(alpha) * np.log(2.0)
    denominator = np.expm1(scale * np.float64(positive_max))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("zero-anchored exp2 denominator is not finite and positive")
    transformed = np.expm1(np.float32(scale) * positive)
    return np.clip(transformed / np.float32(denominator), 0.0, 1.0).astype(np.float32, copy=False)
