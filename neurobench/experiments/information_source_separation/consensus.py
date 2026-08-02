"""Bounded multistart component-consensus source separation."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import InformationSeparationConfig
from .conclusive_methods import execute_common_input


def _aligned_stability(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    correlation = np.abs(left@right.T)/np.maximum(
        np.linalg.norm(left, axis=1)[:, None]*np.linalg.norm(right, axis=1)[None],
        np.finfo(float).eps)
    rows, columns = linear_sum_assignment(-correlation)
    matched = correlation[rows, columns]
    return float(np.mean(matched)), float(np.min(matched))


def fit_multistart_consensus(
    movie: np.ndarray, *, base_method: str, rank: int, starts: int,
    scientific_config: InformationSeparationConfig, seed: int, device: str,
) -> dict[str, Any]:
    """Choose the label-free medoid decomposition across bounded starts."""
    if int(starts) < 3 or int(starts) > 9:
        raise ValueError("starts must be in [3,9]")
    if base_method == "kernel_hsic_pairwise_rotation":
        parameters = {"rank": int(rank), "bandwidth_scale": 2.0}
    elif base_method == "knn_mi_pairwise_rotation":
        parameters = {"rank": int(rank), "neighbors": 3}
    elif base_method == "multilag_sobi":
        parameters = {"rank": int(rank), "lags": [1,2,4,8,15],
                      "covariance_shrinkage": 0.1}
    else:
        raise ValueError("unsupported consensus base method")
    fits = [execute_common_input(
        movie, method_id=base_method, parameters=parameters,
        scientific_config=scientific_config, seed=int(seed)+104729*index,
        device=device,
    ) for index in range(int(starts))]
    pairwise = np.eye(len(fits), dtype=np.float64)
    worst = np.ones(len(fits), dtype=np.float64)
    for left in range(len(fits)-1):
        for right in range(left+1, len(fits)):
            mean, minimum = _aligned_stability(fits[left]["sources"], fits[right]["sources"])
            pairwise[left, right] = pairwise[right, left] = mean
            worst[left] = min(worst[left], minimum)
            worst[right] = min(worst[right], minimum)
    medoid = int(np.argmax(np.mean(pairwise, axis=1)))
    selected = fits[medoid]
    return {**selected, "method_id": "multistart_consensus",
            "reported_base_method": base_method,
            "converged": bool(all(fit["converged"] for fit in fits)),
            "diagnostics": {"base_method": base_method, "base_parameters": parameters,
                            "starts": int(starts), "selected_medoid_start": medoid,
                            "pairwise_mean_stability": pairwise.tolist(),
                            "mean_consensus_stability": float(np.mean(pairwise[np.triu_indices(len(fits), 1)])),
                            "worst_component_stability": float(np.min(worst)),
                            "all_starts_converged": bool(all(fit["converged"] for fit in fits))}}
