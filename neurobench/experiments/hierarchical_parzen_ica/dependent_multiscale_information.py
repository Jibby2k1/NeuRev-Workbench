"""Bounded groupwise information objectives for dependent multiscale W4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from neurobench.algorithms.dependent_multiscale import PatchDecomposition, residualize_nuisance


@dataclass(frozen=True)
class InformationRefinementResult:
    decomposition: PatchDecomposition
    objective_terms: dict[str, float]
    authority: float
    diagnostics: dict[str, Any]


def _samples(values: np.ndarray, maximum_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("group samples must be finite [samples,features]")
    step = max(1, len(array) // int(maximum_samples))
    return array[::step][: int(maximum_samples)]


def normalized_gaussian_gram(
    samples: np.ndarray,
    *,
    bandwidth: float | None = None,
    maximum_samples: int = 512,
) -> tuple[np.ndarray, float]:
    """Return a trace-one Gaussian Gram matrix in bounded sample coordinates."""
    values = _samples(samples, maximum_samples)
    values = values - np.median(values, axis=0, keepdims=True)
    scale = np.maximum(
        1.4826 * np.median(np.abs(values), axis=0, keepdims=True), 1e-8
    )
    values = values / scale
    squared = np.sum((values[:, None] - values[None]) ** 2, axis=2)
    if bandwidth is None:
        positive = squared[squared > 0]
        selected = float(np.sqrt(0.5 * np.median(positive))) if positive.size else 1.0
    else:
        selected = float(bandwidth)
    if not np.isfinite(selected) or selected <= 0:
        raise ValueError("kernel bandwidth must be finite and positive")
    gram = np.exp(-squared / max(2 * selected**2, np.finfo(float).eps))
    trace = float(np.trace(gram))
    return gram / max(trace, np.finfo(float).eps), selected


def matrix_renyi_entropy(normalized_gram: np.ndarray, *, alpha: float = 2.0) -> float:
    """Compute matrix-based Rényi entropy for a trace-one PSD Gram matrix."""
    matrix = np.asarray(normalized_gram, dtype=np.float64)
    order = float(alpha)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("normalized_gram must be a finite square matrix")
    if order <= 0 or np.isclose(order, 1.0):
        raise ValueError("alpha must be positive and differ from one")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.clip(np.linalg.eigvalsh(symmetric), 0.0, None)
    eigenvalues /= max(float(eigenvalues.sum()), np.finfo(float).eps)
    moment = max(float(np.sum(eigenvalues**order)), np.finfo(float).eps)
    return float(np.log(moment) / (1.0 - order))


def matrix_renyi_group_dependence(
    left: np.ndarray,
    right: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
    alpha: float = 2.0,
    maximum_samples: int = 512,
) -> dict[str, float]:
    """Estimate group dependence after optional declared nuisance residualization."""
    first = residualize_nuisance(np.asarray(left, dtype=np.float64), nuisance)
    second = residualize_nuisance(np.asarray(right, dtype=np.float64), nuisance)
    if len(first) != len(second):
        raise ValueError("semantic groups must share their sample axis")
    first = _samples(first, maximum_samples)
    second = _samples(second, maximum_samples)
    count = min(len(first), len(second))
    first, second = first[:count], second[:count]
    left_gram, left_bandwidth = normalized_gaussian_gram(
        first, maximum_samples=maximum_samples
    )
    right_gram, right_bandwidth = normalized_gaussian_gram(
        second, maximum_samples=maximum_samples
    )
    joint = left_gram * right_gram
    joint /= max(float(np.trace(joint)), np.finfo(float).eps)
    left_entropy = matrix_renyi_entropy(left_gram, alpha=alpha)
    right_entropy = matrix_renyi_entropy(right_gram, alpha=alpha)
    joint_entropy = matrix_renyi_entropy(joint, alpha=alpha)
    dependence = max(0.0, left_entropy + right_entropy - joint_entropy)
    return {
        "dependence": float(dependence),
        "left_entropy": left_entropy,
        "right_entropy": right_entropy,
        "joint_entropy": joint_entropy,
        "left_bandwidth": left_bandwidth,
        "right_bandwidth": right_bandwidth,
        "sample_count": float(count),
    }


def build_frame_nuisance(
    observation: np.ndarray,
    *,
    translation_yx: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build label-free frame nuisance variables with fixed orientation ``[T,K]``."""
    movie = np.asarray(observation)
    if movie.ndim != 3 or not np.isfinite(movie).all():
        raise ValueError("observation must be finite [T,Y,X]")
    t = len(movie)
    columns = [movie.mean(axis=(1, 2), dtype=np.float64), np.linspace(-1.0, 1.0, t)]
    names = ["global_intensity", "slow_drift"]
    if translation_yx is not None:
        motion = np.asarray(translation_yx, dtype=np.float64)
        if motion.shape != (t, 2) or not np.isfinite(motion).all():
            raise ValueError("translation_yx must be finite [T,2]")
        columns.extend((motion[:, 0], motion[:, 1]))
        names.extend(("translation_y", "translation_x"))
    nuisance = np.column_stack(columns)
    nuisance -= nuisance.mean(axis=0, keepdims=True)
    nuisance /= np.maximum(nuisance.std(axis=0, keepdims=True), 1e-8)
    return nuisance, tuple(names)


def _temporal_group(values: np.ndarray, rank: int = 4) -> np.ndarray:
    movie = np.asarray(values)
    if movie.ndim != 3 or not np.isfinite(movie[::max(1, len(movie) // 16), ::8, ::8]).all():
        raise ValueError("group movie must be finite [T,Y,X]")
    matrix = movie.reshape(len(movie), -1)
    step = max(1, matrix.shape[1] // 64)
    sample_columns = matrix[:, ::step][:, :64].astype(np.float64, copy=True)
    sample_columns -= sample_columns.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(sample_columns, full_matrices=False)
    selected = min(int(rank), u.shape[1])
    return u[:, :selected] * singular[:selected]


def _project_group(source: np.ndarray, target: np.ndarray, nuisance: np.ndarray | None) -> np.ndarray:
    source_movie = np.asarray(source, dtype=np.float32)
    source_matrix = source_movie.reshape(len(source_movie), -1)
    target_coordinates = _temporal_group(target)
    basis = residualize_nuisance(target_coordinates, nuisance)
    basis, _ = np.linalg.qr(basis, mode="reduced")
    basis = basis.astype(np.float32)
    # residualize_nuisance includes an intercept, making the temporal basis
    # orthogonal to the constant vector. Avoid a full-movie centered copy.
    projected = basis @ (basis.T @ source_matrix)
    return projected.reshape(source_movie.shape)


def refine_group_dependence(
    baseline: PatchDecomposition,
    *,
    observation: np.ndarray,
    nuisance: np.ndarray | None,
    authority: float,
    alpha: float = 2.0,
    maximum_information_samples: int = 256,
    in_place: bool = False,
) -> InformationRefinementResult:
    """Apply a bounded groupwise projection and report the Rényi objective.

    Dependence is removed only from the neural group and reassigned to the
    semantic group whose temporal subspace explained it. Within-neural
    coordinates are never orthogonalized or independently penalized.
    """
    selected = float(authority)
    if not 0 <= selected <= 1:
        raise ValueError("authority must be in [0,1]")
    signal = np.array(baseline.structured_signal, dtype=np.float32, copy=not in_place)
    background = np.array(baseline.background, dtype=np.float32, copy=not in_place)
    artifact = np.array(baseline.structured_artifact, dtype=np.float32, copy=not in_place)
    correction_background = selected * _project_group(signal, background, nuisance)
    background_energy = float(np.mean(correction_background**2))
    signal -= correction_background
    background += correction_background
    del correction_background
    correction_artifact = selected * _project_group(signal, artifact, nuisance)
    artifact_energy = float(np.mean(correction_artifact**2))
    signal -= correction_artifact
    artifact += correction_artifact
    del correction_artifact
    noise = np.asarray(baseline.noise_candidate, dtype=np.float32)
    observed = np.asarray(observation, dtype=np.float32)
    closure = np.array(baseline.closure_residual, dtype=np.float32, copy=not in_place)
    np.subtract(observed, background, out=closure)
    closure -= signal
    closure -= artifact
    closure -= noise
    groups = {
        "signal": _temporal_group(signal),
        "background": _temporal_group(background),
        "artifact": _temporal_group(artifact),
    }
    signal_background = matrix_renyi_group_dependence(
        groups["signal"], groups["background"], nuisance=nuisance,
        alpha=alpha, maximum_samples=maximum_information_samples,
    )
    signal_artifact = matrix_renyi_group_dependence(
        groups["signal"], groups["artifact"], nuisance=nuisance,
        alpha=alpha, maximum_samples=maximum_information_samples,
    )
    decomposition = PatchDecomposition(
        patch_id=baseline.patch_id,
        background=background,
        structured_signal=signal,
        structured_artifact=artifact,
        noise_candidate=noise,
        closure_residual=closure,
        posterior_uncertainty=None,
        diagnostics={
            **baseline.diagnostics,
            "method_id": "dependent_groups_only",
            "dependence_authority": selected,
            "conditional_estimation": "declared_nuisance_residualization_approximation",
        },
    )
    return InformationRefinementResult(
        decomposition=decomposition,
        objective_terms={
            "neural_background_dependence": signal_background["dependence"],
            "neural_artifact_dependence": signal_artifact["dependence"],
            "total_group_dependence": signal_background["dependence"] + signal_artifact["dependence"],
            "reassigned_background_energy": background_energy,
            "reassigned_artifact_energy": artifact_energy,
        },
        authority=selected,
        diagnostics={
            "signal_background": signal_background,
            "signal_artifact": signal_artifact,
            "nuisance_columns": 0 if nuisance is None else nuisance.shape[1],
            "within_neural_dependence_penalized": False,
        },
    )
