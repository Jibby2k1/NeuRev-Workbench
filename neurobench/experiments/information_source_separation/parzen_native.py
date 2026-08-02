"""Native spatial stochastic-Parzen ICA with a noisy-posterior reconstruction."""
from __future__ import annotations

from typing import Any

import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import (
    ParzenDictionaryConfig, fit_stochastic_parzen_ica,
)
from neurobench.algorithms.spatial_patch_ica import (
    SpatialPatchICAModel, fit_parzen_shrinkage, sample_spatial_patches,
)
from neurobench.algorithms.spatial_patch_ica_reconstruction import (
    dense_convolutional_reconstruction,
)


def fit_spatial_stochastic_parzen_noisy_posterior(
    movie: np.ndarray, *, quiet_frames: int, patch_size: int, rank: int,
    noise_scale: float, seed: int, device: str = "cuda",
    sample_count: int = 2048,
) -> dict[str, Any]:
    """Fit the qualified spatial Parzen-score/noisy-posterior native track.

    This is deliberately not named exact Infomax: the demixer uses a bounded
    stochastic Parzen score and the reconstruction uses a noise-convolved
    posterior mean.
    """
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not 16 <= int(quiet_frames) < len(values):
        raise ValueError("invalid movie or quiet-frame contract")
    quiet = values[:int(quiet_frames)]
    center = np.median(quiet, axis=0)
    mad = 1.4826*np.median(np.abs(quiet-center[None]), axis=0)
    positive = mad[mad > 0]
    floor = float(np.percentile(positive, 10)) if positive.size else 1.0
    scale = np.maximum(mad, max(floor, 1e-6)).astype(np.float32)
    standardized = (values-center[None])/scale[None]
    patches = sample_spatial_patches(
        standardized, patch_size=int(patch_size), sample_count=int(sample_count),
        seed=int(seed),
    ).astype(np.float64)
    mean = patches.mean(axis=0)
    centered = patches-mean
    covariance = centered.T@centered/max(len(centered)-1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:int(rank)]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    retained = np.maximum(eigenvalues, max(float(eigenvalues[0])*1e-6, np.finfo(float).eps))
    whitening = eigenvectors/np.sqrt(retained)[None]
    whitened = (centered@whitening).T
    dictionary = ParzenDictionaryConfig(
        maximum_centers=48, minimum_center_separation=0.03,
        bandwidth=0.5, bandwidth_min=0.15, bandwidth_max=1.5,
        update_rate=0.02, replacement_policy="deterministic_reservoir",
        warmup_samples=min(512, whitened.shape[1]), seed=int(seed),
    )
    fit, dictionaries = fit_stochastic_parzen_ica(
        whitened, dictionary, learning_rate=5e-4, gradient_clip=3.0,
        maximum_angle_update_degrees=0.5, batch_size=128,
        maximum_iterations=50, tolerance=1e-5,
    )
    analysis = fit.demixing@whitening.T
    dewhitening = eigenvectors*np.sqrt(retained)[None]
    synthesis = dewhitening@fit.demixing.T
    components = (fit.demixing@whitened).T
    component_scale = np.maximum(np.std(components, axis=0, ddof=1), 1e-6)
    model = SpatialPatchICAModel(
        patch_size=int(patch_size), rank=int(rank),
        patch_mean=mean.astype(np.float32),
        analysis_filters=analysis.astype(np.float32),
        synthesis_atoms=synthesis.astype(np.float32),
        component_scale=component_scale.astype(np.float32),
        explained_variance_ratio=(eigenvalues/max(float(np.trace(covariance)), np.finfo(float).eps)).astype(np.float64),
        fastica_iterations=int(fit.iterations), fastica_converged=bool(fit.converged),
        fastica_final_delta=float(fit.gradient_norm or 0.0),
    )
    pooled = (components/component_scale[None]).ravel()
    posterior = fit_parzen_shrinkage(
        pooled, maximum_centers=48, zero_fraction=0.5,
        active_threshold_z=0.5, bandwidth=0.5,
        noise_variance=float(noise_scale)**2, lookup_points=513,
        lookup_abs_z=8.0,
    )
    standardized_signal, application = dense_convolutional_reconstruction(
        standardized, model, shrinkage="parzen", parzen=posterior,
        device=device, frame_batch_size=4,
    )
    signal = standardized_signal*scale[None]
    return {
        "method_id": "spatial_stochastic_parzen_score_noisy_posterior",
        "requested_family": "spatial_noisy_parzen_infomax",
        "signal": signal.astype(np.float32),
        "remainder": (values-signal).astype(np.float32),
        "converged": bool(fit.converged), "iterations": int(fit.iterations),
        "model_diagnostics": {**model.diagnostics(),
                              "parzen_demixer": fit.diagnostics,
                              "dictionary_states": [state.diagnostics for state in dictionaries],
                              "posterior": posterior.diagnostics()},
        "application_diagnostics": application,
        "quiet_frames": int(quiet_frames),
        "scientific_trace_status": "native_spatial_reconstruction_not_global_temporal_sources",
        "exact_signal_plus_remainder_closure": True,
    }
