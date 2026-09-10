"""Matched patch extraction and ICA fits for all three model geometries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobench.algorithms.information_source_separation import (
    LinearSeparationResult,
    fit_kernel_hsic_pairwise_rotation,
    pca_whiten,
)


@dataclass(frozen=True)
class PatchObservations:
    values: np.ndarray
    times: np.ndarray
    rows: np.ndarray
    columns: np.ndarray


def activity_priority_order(movie: np.ndarray, quiet_frames: int) -> np.ndarray:
    """Return the exact stable activity ordering shared by all patch geometries."""
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not 1 <= quiet_frames < len(values):
        raise ValueError("movie/quiet_frames are invalid")
    baseline = np.median(values[:quiet_frames], axis=0)
    scores = np.abs(values - baseline[None]).ravel()
    return np.argsort(-scores, kind="stable")


def extract_patch_observations(
    movie: np.ndarray,
    *,
    family: str,
    spatial_width: int | None,
    temporal_width: int | None,
    causality: str,
    maximum_samples: int,
    seed: int,
    quiet_frames: int | None = None,
    activity_fraction: float = 0.5,
    activity_order: np.ndarray | None = None,
) -> PatchObservations:
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("movie must be finite TYX")
    if family not in {"temporal", "spatial", "joint_spatiotemporal"}:
        raise ValueError(f"unsupported model family: {family}")
    needs_spatial = family in {"spatial", "joint_spatiotemporal"}
    needs_temporal = family in {"temporal", "joint_spatiotemporal"}
    sw = int(spatial_width or 1)
    tw = int(temporal_width or 1)
    if needs_spatial and (sw < 3 or sw % 2 == 0):
        raise ValueError("spatial model requires odd width >=3")
    if needs_temporal and (
        (causality == "centered" and (tw < 3 or tw % 2 == 0))
        or (causality == "causal" and tw < 2)
    ):
        raise ValueError(
            "temporal model requires odd width >=3 when centered or width >=2 when causal"
        )
    if causality not in {"centered", "causal"}:
        raise ValueError("causality must be centered or causal")
    spatial_radius = sw // 2 if needs_spatial else 0
    if needs_temporal and causality == "centered":
        before = after = tw // 2
    elif needs_temporal:
        before, after = tw - 1, 0
    else:
        before = after = 0
    padded = np.pad(
        values,
        ((before, after), (spatial_radius, spatial_radius), (spatial_radius, spatial_radius)),
        mode="reflect",
    )
    total = int(np.prod(values.shape))
    count = min(maximum_samples, total)
    rng = np.random.default_rng(int(seed))
    if quiet_frames is None:
        flat = np.sort(rng.choice(total, size=count, replace=False))
    else:
        if not 1 <= quiet_frames < len(values) or not 0 <= activity_fraction <= 1:
            raise ValueError("quiet_frames/activity_fraction are invalid")
        activity_count = min(int(round(count * activity_fraction)), count)
        uniform_count = count - activity_count
        uniform = set(map(int, rng.choice(total, size=uniform_count, replace=False)))
        if activity_order is None:
            order = activity_priority_order(values, quiet_frames)
        else:
            order = np.asarray(activity_order)
            if order.ndim != 1 or len(order) != total or not np.issubdtype(order.dtype, np.integer):
                raise ValueError("activity_order must be an integer permutation of movie indices")
        active = []
        for candidate in order:
            value = int(candidate)
            if value not in uniform:
                active.append(value)
                if len(active) == activity_count:
                    break
        flat = np.asarray(sorted(uniform | set(active)), dtype=np.int64)
    times, rows, columns = np.unravel_index(flat, values.shape)
    temporal_offsets = np.arange(-before, after + 1) if needs_temporal else np.asarray([0])
    spatial_offsets = np.arange(-spatial_radius, spatial_radius + 1) if needs_spatial else np.asarray([0])
    tt = times[:, None, None, None] + before + temporal_offsets[None, :, None, None]
    yy = rows[:, None, None, None] + spatial_radius + spatial_offsets[None, None, :, None]
    xx = columns[:, None, None, None] + spatial_radius + spatial_offsets[None, None, None, :]
    patches = padded[tt, yy, xx].reshape(len(flat), -1)
    return PatchObservations(
        values=patches.T.astype(np.float64),
        times=times.astype(np.int32), rows=rows.astype(np.int32),
        columns=columns.astype(np.int32),
    )


def _symmetric_decorrelation(matrix: np.ndarray) -> np.ndarray:
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return (
        eigenvectors
        * (1.0 / np.sqrt(np.maximum(eigenvalues, np.finfo(float).eps)))
    ) @ eigenvectors.T @ matrix


def _fastica_logcosh(
    observations: np.ndarray,
    *,
    rank: int,
    seed: int,
    objective_scale: float,
    max_iterations: int = 300,
    tolerance: float = 1e-5,
) -> LinearSeparationResult:
    from neurobench.algorithms.information_source_separation import _finalize

    if objective_scale <= 0:
        raise ValueError("objective_scale must be positive")
    z, whitening = pca_whiten(observations, rank=rank)
    rng = np.random.default_rng(int(seed))
    rotation = _symmetric_decorrelation(rng.standard_normal((rank, rank)))
    delta = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        projected = rotation @ z
        nonlinear = np.tanh(objective_scale * projected)
        derivative = objective_scale * (1.0 - nonlinear**2).mean(axis=1)
        update = nonlinear @ z.T / z.shape[1] - derivative[:, None] * rotation
        update = _symmetric_decorrelation(update)
        delta = float(np.max(np.abs(np.abs(np.diag(update @ rotation.T)) - 1.0)))
        rotation = update
        if delta <= tolerance:
            converged = True
            break
    objective = float(np.mean(np.log(np.cosh(np.clip(objective_scale * (rotation @ z), -30, 30)))))
    return _finalize(
        "fastica_logcosh", observations, z, whitening, rotation,
        converged=converged, iterations=iteration, objective=objective,
        diagnostics={
            "objective_scale": float(objective_scale),
            "final_delta": delta,
            "tolerance": tolerance,
            "objective_direction": "descriptive",
        },
    )


def fit_ica(
    observations: np.ndarray,
    specification: dict[str, Any],
    *,
    maximum_fit_samples: int,
) -> LinearSeparationResult:
    objective = str(specification["objective"])
    rank = int(specification["rank"])
    seed = int(specification["seed"])
    scale = float(specification["objective_scale"])
    if objective == "fastica_logcosh":
        return _fastica_logcosh(
            observations, rank=rank, seed=seed, objective_scale=scale
        )
    if objective == "hsic_pairwise":
        return fit_kernel_hsic_pairwise_rotation(
            observations, rank=rank, bandwidth_scale=scale,
            angle_step_degrees=10.0, max_sweeps=3,
            improvement_tolerance=1e-4,
            max_fit_samples=min(maximum_fit_samples, 128), seed=seed,
        )
    raise ValueError(f"unsupported ICA objective: {objective}")


def model_summary(result: LinearSeparationResult) -> dict[str, Any]:
    return {
        "method_id": result.method_id,
        "converged": result.converged,
        "iterations": result.iterations,
        "objective": result.objective,
        "internal_whitening": {
            "mean": result.whitening.mean.tolist(),
            "retained_rank": result.whitening.retained_rank,
            "explained_fraction": result.whitening.explained_fraction,
            "condition_number": result.whitening.condition_number,
            "eigenvalues": result.whitening.eigenvalues.tolist(),
        },
        "demixing": result.demixing.tolist(),
        "mixing": result.mixing.tolist(),
        "diagnostics": result.diagnostics,
    }
