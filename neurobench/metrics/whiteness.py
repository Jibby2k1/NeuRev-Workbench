"""Numerical diagnostics for covariance whitening and tile blending."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _square_covariance(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.size
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("covariance must be a finite non-empty square matrix")
    return 0.5 * (matrix + matrix.T)


def covariance_identity_error(covariance: np.ndarray) -> float:
    """Return RMS matrix error from identity: ``||C-I||_F / sqrt(D)``."""
    matrix = _square_covariance(covariance)
    return float(np.linalg.norm(matrix - np.eye(len(matrix)), ord="fro") / np.sqrt(len(matrix)))


def normalized_off_diagonal_energy(covariance: np.ndarray) -> float:
    """Return off-diagonal Frobenius energy divided by total Frobenius energy."""
    matrix = _square_covariance(covariance)
    off = matrix - np.diag(np.diag(matrix))
    denominator = float(np.linalg.norm(matrix, ord="fro"))
    return float(np.linalg.norm(off, ord="fro") / denominator) if denominator else 0.0


def maximum_absolute_correlation(covariance: np.ndarray) -> float:
    """Return the largest absolute off-diagonal correlation implied by ``C``."""
    matrix = _square_covariance(covariance)
    scale = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    denominator = scale[:, None] * scale[None, :]
    correlation = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > np.finfo(np.float64).eps,
    )
    np.fill_diagonal(correlation, 0.0)
    return float(np.max(np.abs(correlation))) if len(matrix) > 1 else 0.0


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Return entropy effective rank ``exp(-sum(p log p))`` for nonnegative eigenvalues."""
    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("eigenvalues must be finite, nonnegative, and non-empty")
    total = float(values.sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0
    probability = values[values > 0] / total
    return float(np.exp(-np.sum(probability * np.log(probability))))


def fit_holdout_covariance_error(
    fit_covariance: np.ndarray,
    holdout_covariance: np.ndarray,
    *,
    eigenvalue_floor_ratio: float = 1e-12,
) -> float:
    """Whiten ``holdout_covariance`` with the fitted SPD matrix and return identity error."""
    fitted = _square_covariance(fit_covariance)
    heldout = _square_covariance(holdout_covariance)
    if fitted.shape != heldout.shape or not 0 < eigenvalue_floor_ratio < 1:
        raise ValueError("covariances must align and floor ratio must lie in (0,1)")
    values, vectors = np.linalg.eigh(fitted)
    floor = max(float(np.max(np.abs(values))) * eigenvalue_floor_ratio, np.finfo(np.float64).eps)
    whitening = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, floor))) @ vectors.T
    return covariance_identity_error(whitening @ heldout @ whitening.T)


def temporal_autocorrelation_summary(values: np.ndarray, max_lag: int) -> dict[str, object]:
    """Summarize per-coordinate Pearson autocorrelation for lags ``1..max_lag``.

    The first axis is time and all remaining axes are flattened into coordinates.
    Constant coordinates contribute zero rather than a nonfinite correlation.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("values must be finite with time on the first axis")
    if not 1 <= int(max_lag) < len(array):
        raise ValueError("max_lag must be positive and shorter than time")
    flat = array.reshape(len(array), -1)
    rows: list[dict[str, float | int]] = []
    for lag in range(1, int(max_lag) + 1):
        left = flat[:-lag]
        right = flat[lag:]
        left = left - left.mean(axis=0, keepdims=True)
        right = right - right.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
        correlations = np.divide(
            np.sum(left * right, axis=0),
            denominator,
            out=np.zeros(flat.shape[1], dtype=np.float64),
            where=denominator > np.finfo(np.float64).eps,
        )
        rows.append({
            "lag": lag,
            "mean_correlation": float(np.mean(correlations)),
            "mean_absolute_correlation": float(np.mean(np.abs(correlations))),
            "maximum_absolute_correlation": float(np.max(np.abs(correlations))),
        })
    return {"max_lag": int(max_lag), "coordinate_count": int(flat.shape[1]), "lags": rows}


def tile_boundary_discontinuity(
    values: np.ndarray,
    tile_geometry: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Compare jumps across internal tile starts with ordinary adjacent-pixel jumps.

    ``tile_geometry`` contains ``(y0, y1, x0, x1)`` bounds. The returned ratio is
    mean boundary jump divided by mean all-adjacent jump; one is seam-neutral.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("values must be a finite array with spatial trailing axes")
    height, width = array.shape[-2:]
    bounds = [tuple(map(int, item)) for item in tile_geometry]
    if not bounds or any(len(item) != 4 for item in bounds):
        raise ValueError("tile_geometry must contain y0,y1,x0,x1 bounds")
    y_edges = sorted({item[0] for item in bounds if 0 < item[0] < height})
    x_edges = sorted({item[2] for item in bounds if 0 < item[2] < width})
    all_jumps = np.concatenate(
        [np.abs(np.diff(array, axis=-2)).ravel(), np.abs(np.diff(array, axis=-1)).ravel()]
    )
    boundary_parts = [np.abs(array[..., y, :] - array[..., y - 1, :]).ravel() for y in y_edges]
    boundary_parts += [np.abs(array[..., :, x] - array[..., :, x - 1]).ravel() for x in x_edges]
    boundary = np.concatenate(boundary_parts) if boundary_parts else np.zeros(1)
    all_mean = float(np.mean(all_jumps)) if all_jumps.size else 0.0
    boundary_mean = float(np.mean(boundary))
    ratio = boundary_mean / max(all_mean, np.finfo(np.float64).eps)
    return {
        "boundary_mean_absolute_jump": boundary_mean,
        "boundary_maximum_absolute_jump": float(np.max(boundary)),
        "all_adjacent_mean_absolute_jump": all_mean,
        "boundary_to_all_ratio": float(ratio),
        "boundary_count": float(boundary.size),
    }
