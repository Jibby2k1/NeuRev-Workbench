"""Linear representation baselines for spatiotemporal calcium movies.

Pixel time courses are observations: ``X.shape == (pixels, frames)``.
Component scores therefore reshape into spatial maps and component bases are
temporal traces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PCAResult:
    spatial_scores: np.ndarray
    temporal_basis: np.ndarray
    singular_values: np.ndarray
    explained_energy_ratio: np.ndarray


@dataclass(frozen=True)
class ICAResult:
    spatial_sources: np.ndarray
    temporal_traces: np.ndarray
    unmixing: np.ndarray
    iterations: int
    converged: bool
    final_delta: float


def truncated_pca(pixel_traces: np.ndarray, rank: int) -> PCAResult:
    """Return an uncentered truncated SVD with pixels as observations."""
    values = np.asarray(pixel_traces, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("pixel_traces must be a finite two-dimensional array")
    rank = int(rank)
    if not 1 <= rank <= min(values.shape):
        raise ValueError("rank must be between one and the smaller array dimension")
    u, singular, vt = np.linalg.svd(values, full_matrices=False)
    total = float(np.square(singular).sum())
    selected = singular[:rank]
    return PCAResult(
        spatial_scores=(u[:, :rank] * selected).astype(np.float32),
        temporal_basis=vt[:rank].astype(np.float32),
        singular_values=selected.astype(np.float64),
        explained_energy_ratio=(np.square(selected) / max(total, np.finfo(float).tiny)).astype(np.float64),
    )


def whiten_spatial_scores(scores: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale PCA spatial scores for ICA."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite two-dimensional array")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.maximum(centered.std(axis=0, ddof=1), float(floor))
    return centered / scale, scale


def _symmetric_decorrelation(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    inverse_root = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
    return inverse_root @ matrix


def symmetric_fastica(
    whitened_scores: np.ndarray,
    temporal_basis: np.ndarray,
    score_scale: np.ndarray,
    *,
    seed: int,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
) -> ICAResult:
    """Fit deterministic-seed symmetric FastICA using the log-cosh contrast."""
    z = np.asarray(whitened_scores, dtype=np.float64)
    basis = np.asarray(temporal_basis, dtype=np.float64)
    scale = np.asarray(score_scale, dtype=np.float64)
    if z.ndim != 2 or basis.ndim != 2 or scale.ndim != 1:
        raise ValueError("invalid ICA input dimensions")
    rank = z.shape[1]
    if basis.shape[0] != rank or scale.shape != (rank,) or not all(
        np.isfinite(item).all() for item in (z, basis, scale)
    ):
        raise ValueError("ICA inputs must be finite and share the component rank")
    rng = np.random.default_rng(int(seed))
    unmixing = _symmetric_decorrelation(rng.standard_normal((rank, rank)))
    delta = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        projected = z @ unmixing.T
        nonlinear = np.tanh(projected)
        derivative_mean = (1.0 - nonlinear * nonlinear).mean(axis=0)
        update = nonlinear.T @ z / len(z) - derivative_mean[:, None] * unmixing
        update = _symmetric_decorrelation(update)
        delta = float(np.max(np.abs(np.abs(np.diag(update @ unmixing.T)) - 1.0)))
        unmixing = update
        if delta < float(tolerance):
            converged = True
            break
    sources = z @ unmixing.T
    traces = unmixing @ (scale[:, None] * basis)
    return ICAResult(
        spatial_sources=sources.astype(np.float32),
        temporal_traces=traces.astype(np.float32),
        unmixing=unmixing.astype(np.float64),
        iterations=iteration,
        converged=converged,
        final_delta=delta,
    )


def orient_components(
    spatial_maps: np.ndarray,
    temporal_traces: np.ndarray,
    quiet_frames: int,
    event_intervals: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient signs toward the strongest event-versus-quiet temporal change."""
    spatial = np.asarray(spatial_maps, dtype=np.float32).copy()
    temporal = np.asarray(temporal_traces, dtype=np.float32).copy()
    if spatial.ndim != 2 or temporal.ndim != 2 or spatial.shape[1] != temporal.shape[0]:
        raise ValueError("spatial_maps and temporal_traces must share component rank")
    quiet = temporal[:, : int(quiet_frames)].mean(axis=1)
    deltas = np.stack([temporal[:, start:stop].mean(axis=1) - quiet for start, stop in event_intervals], axis=1)
    strongest = deltas[np.arange(len(temporal)), np.argmax(np.abs(deltas), axis=1)]
    signs = np.where(strongest < 0, -1.0, 1.0).astype(np.float32)
    spatial *= signs[None, :]
    temporal *= signs[:, None]
    return spatial, temporal, signs


def reconstruction(spatial_scores: np.ndarray, temporal_traces: np.ndarray) -> np.ndarray:
    """Reconstruct pixel time courses from compatible spatial and temporal factors."""
    spatial = np.asarray(spatial_scores, dtype=np.float32)
    temporal = np.asarray(temporal_traces, dtype=np.float32)
    if spatial.ndim != 2 or temporal.ndim != 2 or spatial.shape[1] != temporal.shape[0]:
        raise ValueError("factor dimensions are incompatible")
    return spatial @ temporal


def matched_component_stability(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    """Match component maps by absolute correlation and summarize stability."""
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("component matrices must have the same pixels-by-components shape")
    left -= left.mean(axis=0)
    right -= right.mean(axis=0)
    left /= np.maximum(np.linalg.norm(left, axis=0, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=0, keepdims=True), 1e-12)
    similarity = np.abs(left.T @ right)
    rows, columns = linear_sum_assignment(-similarity)
    values = similarity[rows, columns]
    return {
        "matched_absolute_correlations": values.astype(float).tolist(),
        "mean_absolute_correlation": float(values.mean()),
        "median_absolute_correlation": float(np.median(values)),
        "fraction_at_least_0p9": float(np.mean(values >= 0.9)),
        "assignment": [{"reference": int(a), "candidate": int(b)} for a, b in zip(rows, columns)],
    }
