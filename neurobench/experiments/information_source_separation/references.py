"""Frozen adapters for existing PCA and FastICA/Wiener references."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobench.algorithms.representation_benchmark import (
    symmetric_fastica,
    truncated_pca,
    whiten_spatial_scores,
)
from neurobench.algorithms.spatial_patch_ica import (
    fit_spatial_patch_fastica,
    sample_spatial_patches,
)
from neurobench.algorithms.spatial_patch_ica_reconstruction import (
    dense_convolutional_reconstruction,
)


@dataclass(frozen=True)
class TemporalReference:
    method_id: str
    spatial_maps: np.ndarray
    temporal_sources: np.ndarray
    reconstruction: np.ndarray
    converged: bool
    iterations: int
    diagnostics: dict[str, Any]


def fit_amplitude_pca_reference(
    movie: np.ndarray,
    *,
    rank: int = 8,
) -> TemporalReference:
    """Apply the existing uncentered amplitude-PCA contract to a TYX movie."""
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("movie must be a finite TYX array")
    matrix = values.reshape(len(values), -1).T
    fit = truncated_pca(matrix, int(rank))
    reconstructed = fit.spatial_scores @ fit.temporal_basis
    return TemporalReference(
        method_id="amplitude_pca_reference",
        spatial_maps=fit.spatial_scores,
        temporal_sources=fit.temporal_basis,
        reconstruction=reconstructed.T.reshape(values.shape),
        converged=True,
        iterations=0,
        diagnostics={
            "rank": int(rank),
            "explained_energy_fraction": float(fit.explained_energy_ratio.sum()),
            "fit_scope": "full_fixture_labels_excluded",
            "reference_contract": "existing_uncentered_amplitude_pca",
        },
    )


def fit_spatial_fastica_reference(
    movie: np.ndarray,
    *,
    rank: int,
    seed: int,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
) -> TemporalReference:
    """Apply the existing full-window spatial FastICA contract."""
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("movie must be a finite TYX array")
    matrix = values.reshape(len(values), -1).T
    pca = truncated_pca(matrix, int(rank))
    whitened, scale = whiten_spatial_scores(pca.spatial_scores)
    fit = symmetric_fastica(
        whitened, pca.temporal_basis, scale,
        seed=int(seed), max_iterations=int(max_iterations), tolerance=float(tolerance),
    )
    reconstructed = fit.spatial_sources @ fit.temporal_traces
    return TemporalReference(
        method_id="full_window_spatial_fastica_reference",
        spatial_maps=fit.spatial_sources,
        temporal_sources=fit.temporal_traces,
        reconstruction=reconstructed.T.reshape(values.shape),
        converged=fit.converged,
        iterations=fit.iterations,
        diagnostics={
            "rank": int(rank),
            "final_delta": float(fit.final_delta),
            "fit_scope": "full_fixture_labels_excluded",
            "reference_contract": "existing_symmetric_logcosh_fastica",
        },
    )


def fit_dense_patch_fastica_wiener_reference(
    movie: np.ndarray,
    *,
    quiet_frames: int,
    patch_size: int,
    rank: int,
    sample_count: int,
    seed: int,
    wiener_lambda_z: float = 1.0,
) -> dict[str, Any]:
    """Apply the existing dense translation-shared FastICA/Wiener operator.

    This returns a detector/reconstruction auxiliary movie. Component traces
    are patch-local and are not falsely presented as globally identified
    neuronal sources.
    """
    values = np.asarray(movie, dtype=np.float32)
    if (
        values.ndim != 3
        or not values.size
        or not np.isfinite(values).all()
        or not 16 <= int(quiet_frames) < len(values)
    ):
        raise ValueError("movie or quiet-frame contract is invalid")
    quiet = values[: int(quiet_frames)]
    center = np.median(quiet, axis=0)
    mad = 1.4826 * np.median(np.abs(quiet - center[None]), axis=0)
    positive = mad[mad > 0]
    floor = float(np.percentile(positive, 10)) if positive.size else 1.0
    scale = np.maximum(mad, max(floor, 1e-6)).astype(np.float32)
    standardized = (values - center[None]) / scale[None]
    patches = sample_spatial_patches(
        standardized,
        patch_size=int(patch_size),
        sample_count=int(sample_count),
        seed=int(seed),
    )
    model = fit_spatial_patch_fastica(
        patches, rank=int(rank), seed=int(seed)
    )
    quiet_patches = sample_spatial_patches(
        standardized,
        patch_size=int(patch_size),
        sample_count=int(sample_count),
        seed=int(seed) + 1,
        frame_indices=np.arange(int(quiet_frames)),
    )
    quiet_components = (
        quiet_patches - model.patch_mean[None]
    ) @ model.analysis_filters.T
    from dataclasses import replace

    model = replace(
        model,
        component_scale=np.maximum(
            np.std(quiet_components, axis=0, ddof=1), 1e-6
        ).astype(np.float32),
    )
    standardized_signal, application = dense_convolutional_reconstruction(
        standardized,
        model,
        shrinkage="wiener",
        lambda_z=float(wiener_lambda_z),
        device="cpu",
        frame_batch_size=1,
    )
    signal = standardized_signal * scale[None]
    return {
        "method_id": "dense_patch_fastica_wiener_reference",
        "signal": signal.astype(np.float32),
        "remainder": (values - signal).astype(np.float32),
        "model_diagnostics": model.diagnostics(),
        "application_diagnostics": application,
        "quiet_frames": int(quiet_frames),
        "scientific_trace_status": "auxiliary_not_globally_identified_sources",
        "exact_signal_plus_remainder_closure": True,
    }
