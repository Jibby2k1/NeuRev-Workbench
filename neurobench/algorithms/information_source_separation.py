"""Bounded pure-array methods for information-theoretic source separation.

The methods in this module are deliberately qualified references.  The HSIC
and kNN-MI rotators use bounded pairwise Jacobi updates in a PCA-whitened
subspace; they are not unrestricted reproductions of KICA or MILCA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class WhiteningModel:
    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    eigenvalues: np.ndarray
    retained_rank: int
    explained_fraction: float
    condition_number: float


@dataclass(frozen=True)
class LinearSeparationResult:
    method_id: str
    sources: np.ndarray
    demixing: np.ndarray
    mixing: np.ndarray
    whitening: WhiteningModel
    converged: bool
    iterations: int
    objective: float
    diagnostics: dict[str, Any]


def _matrix(value: np.ndarray, name: str, *, min_rows: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] < min_rows
        or array.shape[1] < 8
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be finite [channels>=%d, samples>=8]" % min_rows
        )
    return array


def pca_whiten(
    observations: np.ndarray,
    *,
    rank: int | None = None,
    eigenvalue_floor_ratio: float = 1e-8,
) -> tuple[np.ndarray, WhiteningModel]:
    """Center and PCA-whiten a channels-by-samples observation matrix."""
    values = _matrix(observations, "observations")
    channels, count = values.shape
    retained = channels if rank is None else int(rank)
    if not 2 <= retained <= channels:
        raise ValueError("rank must be in [2, channel_count]")
    if not 0 < eigenvalue_floor_ratio < 1:
        raise ValueError("eigenvalue_floor_ratio must be in (0, 1)")
    mean = values.mean(axis=1, keepdims=True)
    centered = values - mean
    covariance = centered @ centered.T / count
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    leading = max(float(eigenvalues[0]), np.finfo(float).eps)
    floor = leading * eigenvalue_floor_ratio
    kept_values = np.maximum(eigenvalues[:retained], floor)
    kept_vectors = vectors[:, :retained]
    whitening = np.diag(1.0 / np.sqrt(kept_values)) @ kept_vectors.T
    dewhitening = kept_vectors @ np.diag(np.sqrt(kept_values))
    whitened = whitening @ centered
    positive_total = max(float(np.maximum(eigenvalues, 0).sum()), np.finfo(float).eps)
    model = WhiteningModel(
        mean=mean[:, 0],
        covariance=covariance,
        whitening=whitening,
        dewhitening=dewhitening,
        eigenvalues=eigenvalues,
        retained_rank=retained,
        explained_fraction=float(np.maximum(eigenvalues[:retained], 0).sum() / positive_total),
        condition_number=float(leading / max(float(eigenvalues[retained - 1]), floor)),
    )
    return whitened, model


def _rotation(rank: int, left: int, right: int, angle: float) -> np.ndarray:
    result = np.eye(rank, dtype=np.float64)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    result[left, left] = cosine
    result[left, right] = sine
    result[right, left] = -sine
    result[right, right] = cosine
    return result


def _finalize(
    method_id: str,
    observations: np.ndarray,
    whitened: np.ndarray,
    whitening: WhiteningModel,
    rotation: np.ndarray,
    *,
    converged: bool,
    iterations: int,
    objective: float,
    diagnostics: dict[str, Any],
) -> LinearSeparationResult:
    sources = rotation @ whitened
    demixing = rotation @ whitening.whitening
    mixing = np.linalg.pinv(demixing)
    reconstructed = mixing @ sources + whitening.mean[:, None]
    values = np.asarray(observations, dtype=np.float64)
    relative_closure = float(
        np.linalg.norm(values - reconstructed)
        / max(np.linalg.norm(values - whitening.mean[:, None]), np.finfo(float).eps)
    )
    return LinearSeparationResult(
        method_id=method_id,
        sources=sources,
        demixing=demixing,
        mixing=mixing,
        whitening=whitening,
        converged=bool(converged),
        iterations=int(iterations),
        objective=float(objective),
        diagnostics={**diagnostics, "relative_subspace_closure_error": relative_closure},
    )


def fit_multilag_sobi(
    observations: np.ndarray,
    *,
    rank: int | None = None,
    lags: Sequence[int] = (1, 2, 4, 8),
    covariance_shrinkage: float = 0.02,
    max_sweeps: int = 100,
    tolerance: float = 1e-7,
) -> LinearSeparationResult:
    """Fit a symmetric multi-lag SOBI reference by Jacobi diagonalization."""
    values = _matrix(observations, "observations")
    if not 0 <= covariance_shrinkage < 1:
        raise ValueError("covariance_shrinkage must be in [0, 1)")
    if max_sweeps < 1 or tolerance <= 0:
        raise ValueError("max_sweeps/tolerance must be positive")
    lag_values = tuple(sorted({int(lag) for lag in lags}))
    if not lag_values or lag_values[0] < 1 or lag_values[-1] >= values.shape[1]:
        raise ValueError("lags must be unique, positive, and shorter than samples")
    z, model = pca_whiten(values, rank=rank)
    dimension = z.shape[0]
    matrices = []
    for lag in lag_values:
        forward = z[:, lag:]
        backward = z[:, :-lag]
        covariance = forward @ backward.T / forward.shape[1]
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.diag(np.diag(covariance))
        matrices.append(
            (1.0 - covariance_shrinkage) * covariance
            + covariance_shrinkage * diagonal
        )
    rotation = np.eye(dimension, dtype=np.float64)
    converged = False
    history: list[float] = []
    for sweep in range(1, max_sweeps + 1):
        maximum_sine = 0.0
        for left in range(dimension - 1):
            for right in range(left + 1, dimension):
                off = np.asarray([matrix[left, right] for matrix in matrices])
                contrast = np.asarray([
                    0.5 * (matrix[right, right] - matrix[left, left])
                    for matrix in matrices
                ])
                gram = np.asarray([
                    [float(off @ off), float(off @ contrast)],
                    [float(off @ contrast), float(contrast @ contrast)],
                ])
                _, eigenvectors = np.linalg.eigh(gram)
                direction = eigenvectors[:, 0]
                angle = 0.5 * float(np.arctan2(direction[1], direction[0]))
                while angle > np.pi / 4:
                    angle -= np.pi / 2
                while angle < -np.pi / 4:
                    angle += np.pi / 2
                if abs(angle) <= tolerance:
                    continue
                jacobi = _rotation(dimension, left, right, angle)
                matrices = [jacobi @ matrix @ jacobi.T for matrix in matrices]
                rotation = jacobi @ rotation
                maximum_sine = max(maximum_sine, abs(float(np.sin(angle))))
        off_energy = float(sum(
            np.sum((matrix - np.diag(np.diag(matrix))) ** 2)
            for matrix in matrices
        ))
        history.append(off_energy)
        if maximum_sine <= tolerance:
            converged = True
            break
    return _finalize(
        "multilag_sobi", values, z, model, rotation,
        converged=converged,
        iterations=sweep,
        objective=history[-1],
        diagnostics={
            "lags": list(lag_values),
            "covariance_shrinkage": float(covariance_shrinkage),
            "off_diagonal_energy_history": history,
            "objective_direction": "lower_is_better",
        },
    )


def _median_bandwidth(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if len(sorted_values) > 512:
        indices = np.linspace(0, len(sorted_values) - 1, 512).astype(int)
        sorted_values = sorted_values[indices]
    distances = np.abs(sorted_values[:, None] - sorted_values[None, :])
    positive = distances[distances > 0]
    return max(float(np.median(positive)) if positive.size else 1.0, 1e-6)


def normalized_hsic(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bandwidth_scale: float = 1.0,
) -> float:
    """Return a bounded normalized RBF-HSIC dependence statistic."""
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape or len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("HSIC inputs must be aligned finite vectors of length >=8")
    if bandwidth_scale <= 0:
        raise ValueError("bandwidth_scale must be positive")
    hx = _median_bandwidth(x) * float(bandwidth_scale)
    hy = _median_bandwidth(y) * float(bandwidth_scale)
    kernel_x = np.exp(-0.5 * ((x[:, None] - x[None, :]) / hx) ** 2)
    kernel_y = np.exp(-0.5 * ((y[:, None] - y[None, :]) / hy) ** 2)
    kernel_x -= kernel_x.mean(axis=0, keepdims=True)
    kernel_x -= kernel_x.mean(axis=1, keepdims=True)
    kernel_x += kernel_x.mean()
    kernel_y -= kernel_y.mean(axis=0, keepdims=True)
    kernel_y -= kernel_y.mean(axis=1, keepdims=True)
    kernel_y += kernel_y.mean()
    numerator = float(np.sum(kernel_x * kernel_y))
    denominator = np.sqrt(float(np.sum(kernel_x**2) * np.sum(kernel_y**2)))
    return float(max(0.0, numerator / max(denominator, np.finfo(float).eps)))


def knn_mutual_information(left: np.ndarray, right: np.ndarray, *, neighbors: int = 5) -> float:
    """Estimate bivariate mutual information with the KSG-1 max-norm estimator."""
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape or len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("MI inputs must be aligned finite vectors of length >=8")
    if not 1 <= int(neighbors) < len(x) - 1:
        raise ValueError("neighbors must be in [1, sample_count-2]")
    points = np.column_stack([x, y])
    distances, _ = cKDTree(points).query(points, k=int(neighbors) + 1, p=np.inf)
    epsilon = np.nextafter(distances[:, -1], 0.0)
    sorted_x = np.sort(x)
    sorted_y = np.sort(y)
    nx = np.searchsorted(sorted_x, x + epsilon, side="right") - np.searchsorted(
        sorted_x, x - epsilon, side="left"
    ) - 1
    ny = np.searchsorted(sorted_y, y + epsilon, side="right") - np.searchsorted(
        sorted_y, y - epsilon, side="left"
    ) - 1
    estimate = (
        digamma(int(neighbors))
        + digamma(len(x))
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )
    return float(max(0.0, estimate))


def _pairwise_rotation_fit(
    observations: np.ndarray,
    *,
    method_id: str,
    dependence: Callable[[np.ndarray, np.ndarray], float],
    rank: int | None,
    angle_step_degrees: float,
    max_sweeps: int,
    improvement_tolerance: float,
    max_fit_samples: int,
    seed: int,
    extra_diagnostics: dict[str, Any],
) -> LinearSeparationResult:
    values = _matrix(observations, "observations")
    if not 0 < angle_step_degrees <= 15 or max_sweeps < 1:
        raise ValueError("invalid angle step or sweep count")
    if improvement_tolerance < 0 or max_fit_samples < 32:
        raise ValueError("invalid tolerance or fit-sample bound")
    z, model = pca_whiten(values, rank=rank)
    rng = np.random.default_rng(int(seed))
    if z.shape[1] > max_fit_samples:
        indices = np.sort(rng.choice(z.shape[1], size=max_fit_samples, replace=False))
        fit_values = z[:, indices]
    else:
        indices = np.arange(z.shape[1])
        fit_values = z.copy()
    dimension = fit_values.shape[0]
    rotation = np.eye(dimension, dtype=np.float64)
    angles = np.deg2rad(
        np.arange(-45.0, 45.0 + angle_step_degrees * 0.5, angle_step_degrees)
    )

    def total_pairwise(current: np.ndarray) -> float:
        return float(sum(
            dependence(current[left], current[right])
            for left in range(dimension - 1)
            for right in range(left + 1, dimension)
        ))

    history = [total_pairwise(fit_values)]
    converged = False
    accepted_updates = 0
    for sweep in range(1, max_sweeps + 1):
        start_objective = history[-1]
        for left in range(dimension - 1):
            for right in range(left + 1, dimension):
                pair = fit_values[[left, right]]
                candidates = []
                for angle in angles:
                    cosine, sine = np.cos(angle), np.sin(angle)
                    first = cosine * pair[0] + sine * pair[1]
                    second = -sine * pair[0] + cosine * pair[1]
                    candidates.append(dependence(first, second))
                best_index = int(np.argmin(candidates))
                best_angle = float(angles[best_index])
                current_index = int(np.argmin(np.abs(angles)))
                if candidates[current_index] - candidates[best_index] <= improvement_tolerance:
                    continue
                jacobi = _rotation(dimension, left, right, best_angle)
                fit_values = jacobi @ fit_values
                rotation = jacobi @ rotation
                accepted_updates += 1
        objective = total_pairwise(fit_values)
        history.append(objective)
        if start_objective - objective <= improvement_tolerance:
            converged = True
            break
    return _finalize(
        method_id, values, z, model, rotation,
        converged=converged,
        iterations=sweep,
        objective=history[-1],
        diagnostics={
            **extra_diagnostics,
            "angle_step_degrees": float(angle_step_degrees),
            "max_fit_samples": int(max_fit_samples),
            "fit_sample_count": int(len(indices)),
            "seed": int(seed),
            "accepted_pair_updates": int(accepted_updates),
            "pairwise_dependence_history": history,
            "objective_direction": "lower_is_better",
            "qualification": "bounded_pairwise_rotation_reference",
        },
    )


def fit_kernel_hsic_pairwise_rotation(
    observations: np.ndarray,
    *,
    rank: int | None = None,
    bandwidth_scale: float = 1.0,
    angle_step_degrees: float = 5.0,
    max_sweeps: int = 8,
    improvement_tolerance: float = 1e-4,
    max_fit_samples: int = 256,
    seed: int = 20260801,
) -> LinearSeparationResult:
    """Minimize bounded pairwise normalized HSIC by Jacobi rotations."""
    if bandwidth_scale <= 0:
        raise ValueError("bandwidth_scale must be positive")
    return _pairwise_rotation_fit(
        observations,
        method_id="kernel_hsic_pairwise_rotation",
        dependence=lambda a, b: normalized_hsic(
            a, b, bandwidth_scale=bandwidth_scale
        ),
        rank=rank,
        angle_step_degrees=angle_step_degrees,
        max_sweeps=max_sweeps,
        improvement_tolerance=improvement_tolerance,
        max_fit_samples=max_fit_samples,
        seed=seed,
        extra_diagnostics={"bandwidth_scale": float(bandwidth_scale)},
    )


def fit_knn_mi_pairwise_rotation(
    observations: np.ndarray,
    *,
    rank: int | None = None,
    neighbors: int = 5,
    angle_step_degrees: float = 5.0,
    max_sweeps: int = 8,
    improvement_tolerance: float = 1e-3,
    max_fit_samples: int = 512,
    seed: int = 20260801,
) -> LinearSeparationResult:
    """Minimize bounded pairwise KSG mutual information by Jacobi rotations."""
    return _pairwise_rotation_fit(
        observations,
        method_id="knn_mi_pairwise_rotation",
        dependence=lambda a, b: knn_mutual_information(
            a, b, neighbors=int(neighbors)
        ),
        rank=rank,
        angle_step_degrees=angle_step_degrees,
        max_sweeps=max_sweeps,
        improvement_tolerance=improvement_tolerance,
        max_fit_samples=max_fit_samples,
        seed=seed,
        extra_diagnostics={"neighbors": int(neighbors)},
    )
