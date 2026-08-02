"""Label-free decomposition stability features for identifiability confidence."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .qualification import qualify_temporal_components


FitFunction = Callable[[np.ndarray, int], dict[str, Any]]


def _absolute_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    return np.abs(ac @ bc.T) / np.maximum(
        np.linalg.norm(ac, axis=1)[:, None] * np.linalg.norm(bc, axis=1)[None, :],
        np.finfo(float).eps,
    )


def _cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64).T
    b = np.asarray(right, dtype=np.float64).T
    return np.abs(a @ b.T) / np.maximum(
        np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :],
        np.finfo(float).eps,
    )


def _maximum_off_diagonal(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 0.0
    values = matrix.copy()
    np.fill_diagonal(values, 0.0)
    return float(values.max())


def decomposition_confidence_features(
    movie: np.ndarray,
    *,
    fit: FitFunction,
    spatial_shape: tuple[int, int],
    seed: int,
    perturbations: int = 2,
    perturbation_scale: float = 0.002,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure evidence plus component stability under label-free perturbations."""
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or perturbations < 1 or perturbation_scale <= 0:
        raise ValueError("invalid movie or perturbation contract")
    base = fit(values, int(seed))
    qualification = qualify_temporal_components(
        base["spatial_maps"], base["sources"], spatial_shape=spatial_shape
    )
    source_self = _absolute_correlations(base["sources"], base["sources"])
    mixing_self = _cosines(base["spatial_maps"], base["spatial_maps"])
    temporal_stabilities = []
    spatial_stabilities = []
    rng = np.random.default_rng(int(seed) + 900_001)
    noise_sigma = max(float(np.std(values)), 1e-8) * float(perturbation_scale)
    for index in range(perturbations):
        perturbed = values.astype(np.float64) + rng.normal(
            scale=noise_sigma, size=values.shape
        )
        candidate = fit(perturbed.astype(np.float32), int(seed) + index + 1)
        temporal = _absolute_correlations(base["sources"], candidate["sources"])
        rows, columns = linear_sum_assignment(-temporal)
        temporal_stabilities.extend(float(temporal[r, c]) for r, c in zip(rows, columns))
        spatial = _cosines(base["spatial_maps"], candidate["spatial_maps"])
        spatial_stabilities.extend(float(spatial[r, c]) for r, c in zip(rows, columns))
    features = {
        "qualification_top_score": float(qualification["top_score"]),
        "qualification_margin": float(qualification["score_margin"]),
        "temporal_stability_mean": float(np.mean(temporal_stabilities)),
        "temporal_stability_worst": float(np.min(temporal_stabilities)),
        "spatial_stability_mean": float(np.mean(spatial_stabilities)),
        "spatial_stability_worst": float(np.min(spatial_stabilities)),
        "maximum_source_pair_correlation": _maximum_off_diagonal(source_self),
        "maximum_mixing_pair_cosine": _maximum_off_diagonal(mixing_self),
        "relative_observation_residual": float(base["relative_observation_residual"]),
    }
    return features, {
        "base": base, "qualification": qualification,
        "perturbations": int(perturbations),
        "perturbation_scale": float(perturbation_scale),
        "noise_sigma": noise_sigma,
    }
