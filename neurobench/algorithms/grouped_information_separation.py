"""Bounded grouped-dependence reference for independent subspace analysis."""
from __future__ import annotations

from typing import Any

import numpy as np

from .information_source_separation import (
    LinearSeparationResult,
    normalized_hsic,
    pca_whiten,
)


def fit_group_energy_hsic_isa(
    observations: np.ndarray,
    *,
    rank: int,
    group_size: int = 2,
    bandwidth_scale: float = 1.0,
    angle_step_degrees: float = 10.0,
    max_sweeps: int = 5,
    improvement_tolerance: float = 1e-4,
    max_fit_samples: int = 192,
    seed: int = 20260801,
) -> LinearSeparationResult:
    """Minimize dependence between group energies while retaining within-group dependence.

    This is a bounded grouped-HSIC ISA reference. It is not an unrestricted
    implementation of FastISA or IVA and remains gated from full-data use.
    """
    values = np.asarray(observations, dtype=np.float64)
    selected_rank = int(rank)
    size = int(group_size)
    if selected_rank < 4 or size < 2 or selected_rank % size:
        raise ValueError("rank must be >=4 and divisible by group_size>=2")
    if bandwidth_scale <= 0 or not 0 < angle_step_degrees <= 20:
        raise ValueError("invalid bandwidth or angle step")
    if max_sweeps < 1 or max_fit_samples < 32:
        raise ValueError("invalid sweep or sample bound")
    z, model = pca_whiten(values, rank=selected_rank)
    rng = np.random.default_rng(int(seed))
    if z.shape[1] > max_fit_samples:
        indices = np.sort(rng.choice(z.shape[1], max_fit_samples, replace=False))
        fit_values = z[:, indices]
    else:
        fit_values = z.copy()
    groups = tuple(
        tuple(range(start, start + size))
        for start in range(0, selected_rank, size)
    )
    group_of = {
        component: group_index
        for group_index, group in enumerate(groups)
        for component in group
    }

    def objective(current: np.ndarray) -> float:
        energies = [np.sqrt(np.sum(current[list(group)] ** 2, axis=0) + 1e-12) for group in groups]
        return float(sum(
            normalized_hsic(
                energies[left], energies[right], bandwidth_scale=bandwidth_scale
            )
            for left in range(len(groups) - 1)
            for right in range(left + 1, len(groups))
        ))

    angles = np.deg2rad(
        np.arange(-45.0, 45.0 + angle_step_degrees * 0.5, angle_step_degrees)
    )
    rotation = np.eye(selected_rank, dtype=np.float64)
    history = [objective(fit_values)]
    accepted = 0
    converged = False
    for sweep in range(1, int(max_sweeps) + 1):
        initial = history[-1]
        for left in range(selected_rank - 1):
            for right in range(left + 1, selected_rank):
                if group_of[left] == group_of[right]:
                    continue
                best_objective = objective(fit_values)
                best_angle = 0.0
                pair = fit_values[[left, right]].copy()
                for angle in angles:
                    candidate = fit_values.copy()
                    cosine, sine = np.cos(angle), np.sin(angle)
                    candidate[left] = cosine * pair[0] + sine * pair[1]
                    candidate[right] = -sine * pair[0] + cosine * pair[1]
                    candidate_objective = objective(candidate)
                    if candidate_objective < best_objective:
                        best_objective = candidate_objective
                        best_angle = float(angle)
                current_objective = objective(fit_values)
                if current_objective - best_objective <= improvement_tolerance:
                    continue
                cosine, sine = np.cos(best_angle), np.sin(best_angle)
                fit_values[left] = cosine * pair[0] + sine * pair[1]
                fit_values[right] = -sine * pair[0] + cosine * pair[1]
                jacobi = np.eye(selected_rank)
                jacobi[left, left] = cosine
                jacobi[left, right] = sine
                jacobi[right, left] = -sine
                jacobi[right, right] = cosine
                rotation = jacobi @ rotation
                accepted += 1
        current = objective(fit_values)
        history.append(current)
        if initial - current <= improvement_tolerance:
            converged = True
            break
    sources = rotation @ z
    demixing = rotation @ model.whitening
    mixing = np.linalg.pinv(demixing)
    reconstructed = mixing @ sources + model.mean[:, None]
    observation_residual = float(
        np.linalg.norm(values - reconstructed)
        / max(np.linalg.norm(values - model.mean[:, None]), np.finfo(float).eps)
    )
    return LinearSeparationResult(
        method_id="group_energy_hsic_isa",
        sources=sources,
        demixing=demixing,
        mixing=mixing,
        whitening=model,
        converged=converged,
        iterations=sweep,
        objective=history[-1],
        diagnostics={
            "groups": [list(group) for group in groups],
            "group_size": size,
            "bandwidth_scale": float(bandwidth_scale),
            "angle_step_degrees": float(angle_step_degrees),
            "fit_sample_count": int(fit_values.shape[1]),
            "accepted_cross_group_updates": int(accepted),
            "group_dependence_history": history,
            "relative_observation_residual": observation_residual,
            "objective_direction": "lower_is_better",
            "qualification": "bounded_group_energy_hsic_isa_reference",
        },
    )
