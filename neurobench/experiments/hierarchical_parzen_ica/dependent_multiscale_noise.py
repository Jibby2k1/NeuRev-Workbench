"""Joint quiet-noise modeling for aligned multiscale coordinates."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from neurobench.algorithms.dependent_multiscale import JointQuietNoiseModel


def _aligned_samples(views: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray]:
    ids = tuple(sorted(views))
    if not ids:
        raise ValueError("quiet views cannot be empty")
    arrays = [np.asarray(views[key], dtype=np.float64).reshape(-1) for key in ids]
    if any(len(value) != len(arrays[0]) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("quiet views must be finite and aligned")
    return ids, np.column_stack(arrays)


def fit_joint_quiet_noise_model(
    quiet_views: Mapping[str, np.ndarray],
    *,
    model_kind: str = "joint_covariance_plus_parzen",
    maximum_reference_samples: int = 2048,
    eigenvalue_floor_ratio: float = 1e-6,
) -> JointQuietNoiseModel:
    """Fit robust center, joint covariance, whitening, and bounded reference."""
    if model_kind not in {
        "joint_covariance_robust", "joint_covariance_plus_parzen",
        "independent_scale_noise",
    }:
        raise ValueError("unsupported quiet-noise model")
    ids, samples = _aligned_samples(quiet_views)
    center = np.median(samples, axis=0)
    centered = samples - center
    radius = np.linalg.norm(centered, axis=1)
    cutoff = max(float(np.quantile(radius, 0.9)), np.finfo(float).eps)
    weights = np.minimum(1.0, cutoff / np.maximum(radius, np.finfo(float).eps))
    weighted = centered * np.sqrt(weights[:, None])
    covariance = weighted.T @ weighted / max(float(weights.sum()), 1.0)
    if model_kind == "independent_scale_noise":
        covariance = np.diag(np.diag(covariance))
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    floor = max(float(eigenvalues[0]) * float(eigenvalue_floor_ratio), np.finfo(float).eps)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor))) @ vectors.T
    whitened = centered @ inverse_sqrt
    pair_distances = np.linalg.norm(np.diff(whitened, axis=0), axis=1)
    positive = pair_distances[pair_distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    step = max(1, len(whitened) // int(maximum_reference_samples))
    reference = whitened[::step][: int(maximum_reference_samples)].copy()
    correlation = np.corrcoef(centered, rowvar=False)
    return JointQuietNoiseModel(
        view_ids=ids,
        center=center,
        covariance=covariance,
        inverse_sqrt=inverse_sqrt,
        eigenvalues=eigenvalues,
        kernel_bandwidth=max(bandwidth, 1e-6),
        reference_samples=reference if model_kind.endswith("parzen") else None,
        model_kind=model_kind,
        diagnostics={
            "sample_count": len(samples),
            "covariance_condition_number": float(eigenvalues[0] / max(eigenvalues[-1], floor)),
            "scale_pair_correlations": correlation.tolist(),
            "eigenvalue_floor": floor,
        },
    )


def joint_cs_divergence(
    samples: np.ndarray,
    model: JointQuietNoiseModel,
    *,
    maximum_samples: int = 512,
) -> float:
    """Compute bounded joint Gaussian-Parzen Cauchy--Schwarz divergence."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(model.view_ids) or not np.isfinite(values).all():
        raise ValueError("samples must be finite [N,number_of_views]")
    if model.reference_samples is None:
        raise ValueError("joint Parzen reference samples are unavailable")
    p = (values - model.center) @ model.inverse_sqrt
    q = np.asarray(model.reference_samples, dtype=np.float64)
    p = p[::max(1, len(p) // int(maximum_samples))][: int(maximum_samples)]
    q = q[::max(1, len(q) // int(maximum_samples))][: int(maximum_samples)]
    variance = max(2.0 * model.kernel_bandwidth**2, np.finfo(float).eps)

    def interaction(left: np.ndarray, right: np.ndarray) -> float:
        distance = np.sum((left[:, None] - right[None]) ** 2, axis=2)
        return float(np.mean(np.exp(-distance / variance)))

    pp, qq, pq = interaction(p, p), interaction(q, q), interaction(p, q)
    cosine = np.clip(pq * pq / max(pp * qq, np.finfo(float).eps), np.finfo(float).eps, 1.0)
    return float(-np.log(cosine))


def qualify_noise_candidate(metrics: dict[str, float], limits: dict[str, float]) -> str:
    """Return the only allowed residual names under the declared checks."""
    required = {
        "joint_cs_divergence", "covariance_error", "temporal_acf_energy",
        "spatial_acf_energy", "event_locked_energy", "motion_edge_correlation",
        "closure_max_normalized",
    }
    if set(metrics) != required or set(limits) != required:
        raise ValueError("qualification metrics and limits must match the full contract")
    if not all(np.isfinite(list(metrics.values()))) or not all(np.isfinite(list(limits.values()))):
        raise ValueError("qualification values must be finite")
    return "qualified_measurement_noise" if all(metrics[key] <= limits[key] for key in required) else "noise_candidate"
