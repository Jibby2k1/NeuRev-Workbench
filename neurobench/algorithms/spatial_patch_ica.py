"""Spatial patch ICA and translation-shared reconstruction operators.

The training samples are image patches, not pixel time courses.  A fitted
model can therefore be applied either to a coarse overlapping patch lattice or
densely as a convolutional analysis/synthesis bank.  This distinction is useful
for testing whether translation sharing helps before changing the ICA
objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import (
    noisy_parzen_posterior_mean,
)
from neurobench.algorithms.representation_benchmark import symmetric_fastica


@dataclass(frozen=True)
class SpatialPatchICAModel:
    patch_size: int
    rank: int
    patch_mean: np.ndarray
    analysis_filters: np.ndarray
    synthesis_atoms: np.ndarray
    component_scale: np.ndarray
    explained_variance_ratio: np.ndarray
    fastica_iterations: int
    fastica_converged: bool
    fastica_final_delta: float

    def diagnostics(self) -> dict[str, Any]:
        gram = self.analysis_filters @ self.synthesis_atoms
        return {
            "patch_size": self.patch_size,
            "rank": self.rank,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "explained_variance_sum": float(self.explained_variance_ratio.sum()),
            "fastica_iterations": self.fastica_iterations,
            "fastica_converged": self.fastica_converged,
            "fastica_final_delta": self.fastica_final_delta,
            "analysis_synthesis_identity_error": float(
                np.linalg.norm(gram - np.eye(self.rank), ord="fro")
                / np.sqrt(self.rank)
            ),
        }


@dataclass(frozen=True)
class ParzenShrinkage:
    grid: np.ndarray
    posterior_mean: np.ndarray
    centers: np.ndarray
    bandwidth: float
    noise_variance: float

    def diagnostics(self) -> dict[str, Any]:
        return {
            "center_count": int(len(self.centers)),
            "zero_center_count": int(np.count_nonzero(self.centers == 0)),
            "bandwidth": self.bandwidth,
            "noise_variance": self.noise_variance,
            "grid_min": float(self.grid[0]),
            "grid_max": float(self.grid[-1]),
            "grid_points": int(len(self.grid)),
        }


def _finite_video(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or not array.size or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite TYX array")
    return array


def sample_spatial_patches(
    values: np.ndarray,
    *,
    patch_size: int,
    sample_count: int,
    seed: int,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Sample reproducible flattened spatial patches from a TYX movie."""
    video = _finite_video(values)
    patch = int(patch_size)
    count = int(sample_count)
    if patch < 3 or patch % 2 == 0 or patch > min(video.shape[1:]):
        raise ValueError("patch_size must be an odd integer within the image")
    if count < 2 * patch * patch:
        raise ValueError("sample_count is too small for a stable patch covariance")
    if frame_indices is None:
        frames = np.arange(len(video), dtype=np.int64)
    else:
        frames = np.asarray(frame_indices, dtype=np.int64)
        if (
            frames.ndim != 1
            or not len(frames)
            or np.any(frames < 0)
            or np.any(frames >= len(video))
        ):
            raise ValueError("frame_indices must select valid video frames")
    rng = np.random.default_rng(int(seed))
    selected_t = rng.choice(frames, size=count, replace=True)
    selected_y = rng.integers(0, video.shape[1] - patch + 1, size=count)
    selected_x = rng.integers(0, video.shape[2] - patch + 1, size=count)
    output = np.empty((count, patch * patch), dtype=np.float32)
    for index, (frame, y, x) in enumerate(
        zip(selected_t, selected_y, selected_x)
    ):
        output[index] = video[frame, y : y + patch, x : x + patch].reshape(-1)
    return output


def fit_spatial_patch_fastica(
    patches: np.ndarray,
    *,
    rank: int,
    seed: int,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
    eigenvalue_floor_ratio: float = 1e-6,
) -> SpatialPatchICAModel:
    """Fit PCA-whitened FastICA to spatial patch observations."""
    values = np.asarray(patches, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("patches must be a finite samples-by-features array")
    selected_rank = int(rank)
    if not 1 <= selected_rank <= min(values.shape):
        raise ValueError("rank must fit within the patch matrix")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:selected_rank]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    floor = max(
        float(eigenvalues[0]) * float(eigenvalue_floor_ratio),
        np.finfo(np.float64).eps,
    )
    retained = np.maximum(eigenvalues, floor)
    whitening = eigenvectors / np.sqrt(retained)[None, :]
    whitened = centered @ whitening
    fit = symmetric_fastica(
        whitened,
        np.eye(selected_rank, dtype=np.float64),
        np.ones(selected_rank, dtype=np.float64),
        seed=int(seed),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
    )
    unmixing = fit.unmixing
    analysis = unmixing @ whitening.T
    dewhitening = eigenvectors * np.sqrt(retained)[None, :]
    synthesis = dewhitening @ unmixing.T
    sources = centered @ analysis.T
    component_scale = np.maximum(
        np.std(sources, axis=0, ddof=1), np.finfo(np.float32).eps
    )
    total_variance = max(float(np.trace(covariance)), np.finfo(float).tiny)
    return SpatialPatchICAModel(
        patch_size=int(round(np.sqrt(values.shape[1]))),
        rank=selected_rank,
        patch_mean=mean.astype(np.float32),
        analysis_filters=analysis.astype(np.float32),
        synthesis_atoms=synthesis.astype(np.float32),
        component_scale=component_scale.astype(np.float32),
        explained_variance_ratio=(eigenvalues / total_variance).astype(np.float64),
        fastica_iterations=fit.iterations,
        fastica_converged=fit.converged,
        fastica_final_delta=fit.final_delta,
    )


def fit_parzen_shrinkage(
    standardized_component_samples: np.ndarray,
    *,
    maximum_centers: int,
    zero_fraction: float,
    active_threshold_z: float,
    bandwidth: float,
    noise_variance: float,
    lookup_points: int,
    lookup_abs_z: float,
) -> ParzenShrinkage:
    """Fit the same bounded noisy-Parzen posterior used by the local audit."""
    samples = np.asarray(standardized_component_samples, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    active = samples[np.abs(samples) >= float(active_threshold_z)]
    center_count = int(maximum_centers)
    zero_count = int(round(center_count * float(zero_fraction)))
    slab_count = center_count - zero_count
    if center_count < 4 or not 1 <= zero_count < center_count:
        raise ValueError("invalid Parzen dictionary allocation")
    if len(active) < slab_count:
        raise ValueError("not enough active component samples for Parzen centers")
    centers = np.concatenate(
        (
            np.zeros(zero_count, dtype=np.float64),
            np.quantile(active, np.linspace(0, 1, slab_count)),
        )
    )
    grid = np.linspace(
        -float(lookup_abs_z), float(lookup_abs_z), int(lookup_points)
    )
    posterior = noisy_parzen_posterior_mean(
        grid, centers, float(bandwidth), float(noise_variance)
    )
    return ParzenShrinkage(
        grid=grid.astype(np.float64),
        posterior_mean=np.asarray(posterior, dtype=np.float64),
        centers=centers,
        bandwidth=float(bandwidth),
        noise_variance=float(noise_variance),
    )


def shrink_components(
    components: np.ndarray,
    component_scale: np.ndarray,
    *,
    method: str,
    lambda_z: float = 1.0,
    parzen: ParzenShrinkage | None = None,
) -> np.ndarray:
    values = np.asarray(components, dtype=np.float32)
    scale = np.asarray(component_scale, dtype=np.float32)
    if values.shape[-1] != len(scale):
        raise ValueError("component_scale must match the last component dimension")
    standardized = values / np.maximum(scale, 1e-6)
    if method == "wiener":
        gain = standardized * standardized / (
            standardized * standardized + float(lambda_z) ** 2
        )
        clean = standardized * gain
    elif method == "parzen":
        if parzen is None:
            raise ValueError("parzen shrinkage requires a fitted posterior")
        clean = np.interp(
            np.clip(standardized, parzen.grid[0], parzen.grid[-1]),
            parzen.grid,
            parzen.posterior_mean,
        )
    else:
        raise ValueError("method must be wiener or parzen")
    return (clean * scale).astype(np.float32)
