"""Common-input method adapters used by the conclusive generated stages."""
from __future__ import annotations

from typing import Any

import numpy as np

from neurobench.algorithms.grouped_information_separation import fit_group_energy_hsic_isa
from neurobench.algorithms.information_source_separation import (
    fit_knn_mi_pairwise_rotation, fit_multilag_sobi,
)

from .config import InformationSeparationConfig
from .gpu_screen import _execute_cuda_method
from .references import fit_amplitude_pca_reference, fit_spatial_fastica_reference


GENERATED_COMMON_INPUT_METHODS = {
    "amplitude_pca_reference", "multilag_sobi",
    "full_window_spatial_fastica_reference", "kernel_hsic_pairwise_rotation",
    "knn_mi_pairwise_rotation", "group_energy_hsic_isa",
}


def execute_common_input(
    movie: np.ndarray, *, method_id: str, parameters: dict[str, Any],
    scientific_config: InformationSeparationConfig, seed: int, device: str,
) -> dict[str, Any]:
    """Return a uniform temporal-decomposition result for a bounded method."""
    values = np.asarray(movie, dtype=np.float32)
    observations = values.reshape(len(values), -1).T
    if method_id == "amplitude_pca_reference":
        fitted = fit_amplitude_pca_reference(values, rank=int(parameters["rank"]))
        residual = values-fitted.reconstruction
        return {"method_id": method_id, "sources": fitted.temporal_sources,
                "spatial_maps": fitted.spatial_maps, "converged": fitted.converged,
                "iterations": fitted.iterations,
                "relative_observation_residual": float(np.linalg.norm(residual)/max(np.linalg.norm(values-values.mean(axis=0)), np.finfo(float).eps)),
                "execution_backend": "numpy_scipy_cpu", "diagnostics": fitted.diagnostics}
    if method_id == "full_window_spatial_fastica_reference":
        fitted = fit_spatial_fastica_reference(values, rank=int(parameters["rank"]), seed=int(seed))
        residual = values-fitted.reconstruction
        return {"method_id": method_id, "sources": fitted.temporal_sources,
                "spatial_maps": fitted.spatial_maps, "converged": fitted.converged,
                "iterations": fitted.iterations,
                "relative_observation_residual": float(np.linalg.norm(residual)/max(np.linalg.norm(values-values.mean(axis=0)), np.finfo(float).eps)),
                "execution_backend": "numpy_scipy_cpu", "diagnostics": fitted.diagnostics}
    if method_id == "kernel_hsic_pairwise_rotation":
        result = _execute_cuda_method(values, method_id, parameters, scientific_config, seed, device)
        result["method_id"] = method_id
        return result
    if method_id == "multilag_sobi":
        fitted = fit_multilag_sobi(observations, **parameters)
    elif method_id == "knn_mi_pairwise_rotation":
        settings = scientific_config.methods["knn_mi_pairwise_rotation"]
        fitted = fit_knn_mi_pairwise_rotation(
            observations, **parameters,
            angle_step_degrees=float(settings["angle_step_degrees"]),
            max_sweeps=int(settings["max_sweeps"]),
            max_fit_samples=int(settings["max_fit_samples"]), seed=int(seed),
        )
    elif method_id == "group_energy_hsic_isa":
        fitted = fit_group_energy_hsic_isa(
            observations, **parameters, angle_step_degrees=10.0,
            max_sweeps=5, max_fit_samples=192, seed=int(seed),
        )
    else:
        raise ValueError(f"method is not a generated common-input method: {method_id}")
    return {
        "method_id": method_id, "sources": fitted.sources,
        "spatial_maps": fitted.mixing, "converged": fitted.converged,
        "iterations": fitted.iterations,
        "relative_observation_residual": float(fitted.diagnostics.get(
            "relative_subspace_closure_error",
            fitted.diagnostics.get("relative_observation_residual", np.nan))),
        "execution_backend": "numpy_scipy_cpu", "diagnostics": fitted.diagnostics,
    }
