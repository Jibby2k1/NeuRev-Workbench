"""Per-context two-frame ICA and bounded cross-context subspace models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from neurobench.algorithms.pairwise_separation import (
    SeparationFit,
    apply_linear_separation,
    center_and_whiten_2d,
    cs_parzen_objective,
    fit_cs_parzen_ica,
)
from neurobench.algorithms.quiet_calibration import group_energy


@dataclass(frozen=True)
class PerContextICAFit:
    context_id: str
    sample_count: int
    center: np.ndarray
    whitening: np.ndarray
    rotation_angle_degrees: float
    rotation: np.ndarray
    demixing: np.ndarray
    persistence_index: int
    innovation_index: int
    component_signs: tuple[int, int]
    objective_name: str
    objective_value: float
    baseline_objective_value: float
    derivative_angle_distance_degrees: float
    converged: bool
    ambiguous_alignment: bool
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            self.center.shape != (2,)
            or self.whitening.shape != (2, 2)
            or self.rotation.shape != (2, 2)
            or self.demixing.shape != (2, 2)
            or self.persistence_index == self.innovation_index
            or set((self.persistence_index, self.innovation_index)) != {0, 1}
        ):
            raise ValueError("invalid per-context ICA fit")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("center", "whitening", "rotation", "demixing"):
            payload[key] = np.asarray(payload[key]).tolist()
        payload["component_signs"] = list(self.component_signs)
        return payload


@dataclass(frozen=True)
class CrossContextFit:
    mode: str
    input_count: int
    component_count: int
    center: np.ndarray
    transform: np.ndarray
    inverse_transform: np.ndarray
    explained_variance: np.ndarray | None
    converged: bool
    seed: int
    sample_count: int
    diagnostics: dict[str, Any]

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        array = _matrix(values, "values")
        if array.shape[1] != self.input_count:
            raise ValueError("cross-context input dimension mismatch")
        return ((array - self.center) @ self.transform.T).astype(np.float32)

    def inverse_values(self, values: np.ndarray) -> np.ndarray:
        array = _matrix(values, "values")
        if array.shape[1] != self.component_count:
            raise ValueError("cross-context component dimension mismatch")
        return (array @ self.inverse_transform.T + self.center).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_count": self.input_count,
            "component_count": self.component_count,
            "center": self.center.tolist(),
            "transform": self.transform.tolist(),
            "inverse_transform": self.inverse_transform.tolist(),
            "explained_variance": (
                None
                if self.explained_variance is None
                else self.explained_variance.tolist()
            ),
            "converged": self.converged,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "diagnostics": self.diagnostics,
        }


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return array


def _pairs(values: np.ndarray, name: str) -> np.ndarray:
    array = _matrix(values, name)
    if array.shape[1] == 2:
        return array
    if array.shape[0] == 2:
        return array.T
    raise ValueError(f"{name} must have shape [N,2] or [2,N]")


def _rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_degrees))
    return np.asarray(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )


def _canonical_alignment(
    demixing: np.ndarray,
    whitening: np.ndarray,
    *,
    ambiguity_margin: float,
) -> tuple[np.ndarray, int, int, tuple[int, int], bool, dict[str, Any]]:
    effective = np.asarray(demixing) @ np.asarray(whitening)
    common = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    difference = np.asarray([-1.0, 1.0]) / np.sqrt(2.0)
    normalized = effective / np.maximum(
        np.linalg.norm(effective, axis=1, keepdims=True), 1e-12
    )
    common_corr = normalized @ common
    difference_corr = normalized @ difference
    score_direct = abs(common_corr[0]) + abs(difference_corr[1])
    score_swap = abs(common_corr[1]) + abs(difference_corr[0])
    if score_direct >= score_swap:
        order = (0, 1)
        persistence_index, innovation_index = 0, 1
    else:
        order = (1, 0)
        persistence_index, innovation_index = 1, 0
    signs_by_original = [1, 1]
    signs_by_original[persistence_index] = (
        1 if common_corr[persistence_index] >= 0 else -1
    )
    signs_by_original[innovation_index] = (
        1 if difference_corr[innovation_index] >= 0 else -1
    )
    aligned = np.stack(
        [
            signs_by_original[persistence_index] * demixing[persistence_index],
            signs_by_original[innovation_index] * demixing[innovation_index],
        ]
    )
    ambiguous = abs(score_direct - score_swap) <= float(ambiguity_margin)
    return (
        aligned,
        persistence_index,
        innovation_index,
        (int(signs_by_original[0]), int(signs_by_original[1])),
        ambiguous,
        {
            "common_correlations": common_corr.tolist(),
            "difference_correlations": difference_corr.tolist(),
            "direct_assignment_score": float(score_direct),
            "swapped_assignment_score": float(score_swap),
            "aligned_order": list(order),
        },
    )


def derivative_angle_distance(effective_direction: np.ndarray) -> float:
    vector = np.asarray(effective_direction, dtype=np.float64).reshape(2)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError("effective direction must be finite and nonzero")
    vector /= np.linalg.norm(vector)
    derivative = np.asarray([-1.0, 1.0]) / np.sqrt(2.0)
    cosine = float(np.clip(abs(vector @ derivative), 0.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def fit_per_context_ica(
    context_id: str,
    screen_pairs: np.ndarray,
    confirmation_pairs: np.ndarray | None = None,
    *,
    objective: Literal["cs_parzen", "fastica"] = "cs_parzen",
    parzen_bandwidth: float = 0.35,
    eigenvalue_floor_ratio: float = 1e-6,
    coarse_step_degrees: float = 3.0,
    refine_half_width_degrees: float = 3.0,
    refine_step_degrees: float = 0.25,
    kernel_block_rows: int = 256,
    kernel_dtype: np.dtype = np.float32,
    ambiguity_margin: float = 0.05,
    compute_backend: Literal["cpu", "cuda"] = "cpu",
) -> PerContextICAFit:
    """Fit and canonically align one two-frame context model."""
    screen = _pairs(screen_pairs, "screen_pairs")
    confirmation = (
        screen
        if confirmation_pairs is None
        else _pairs(confirmation_pairs, "confirmation_pairs")
    )
    whitened_screen, whitening = center_and_whiten_2d(
        screen.T, eigenvalue_floor_ratio=float(eigenvalue_floor_ratio)
    )
    whitened_confirmation = whitening.whitening @ (
        confirmation.T - whitening.mean[:, None]
    )
    if objective == "cs_parzen":
        fitted = fit_cs_parzen_ica(
            whitened_screen,
            whitened_confirmation,
            bandwidth=float(parzen_bandwidth),
            block_rows=int(kernel_block_rows),
            screen_step_degrees=float(coarse_step_degrees),
            refine_half_width_degrees=float(refine_half_width_degrees),
            refine_step_degrees=float(refine_step_degrees),
            kernel_dtype=kernel_dtype,
            backend=compute_backend,
        )
        angle = float(fitted.diagnostics["selected_angle_degrees"])
    elif objective == "fastica":
        import warnings

        from sklearn.decomposition import FastICA
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = FastICA(
                n_components=None,
                whiten=False,
                random_state=0,
                max_iter=1000,
                tol=1e-6,
            ).fit(whitened_screen.T)
        demixing = np.asarray(model.components_, dtype=np.float64)
        fitted = SeparationFit(
            method_id="sklearn_fastica",
            demixing=demixing,
            mixing=np.linalg.pinv(demixing),
            objective=None,
            converged=int(model.n_iter_) < int(model.max_iter),
            iterations=int(model.n_iter_),
            activity_component=None,
            activity_sign=None,
            diagnostics={
                "algorithm": "parallel_fixed_point_fastica",
                "nonlinearity": "logcosh",
                "random_state": 0,
                "iterations": int(model.n_iter_),
            },
        )
        angle = float(
            np.rad2deg(np.arctan2(fitted.demixing[0, 1], fitted.demixing[0, 0]))
            % 90.0
        )
    else:
        raise ValueError("objective must be cs_parzen or fastica")
    aligned, persistence, innovation, signs, ambiguous, alignment = (
        _canonical_alignment(
            fitted.demixing,
            whitening.whitening,
            ambiguity_margin=float(ambiguity_margin),
        )
    )
    analytic = (
        np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    ) @ whitened_confirmation
    baseline = cs_parzen_objective(
        analytic.T,
        float(parzen_bandwidth),
        block_rows=int(kernel_block_rows),
        kernel_dtype=kernel_dtype,
        backend=compute_backend,
    ).objective
    selected_outputs = aligned @ whitened_confirmation
    selected_objective = cs_parzen_objective(
        selected_outputs.T,
        float(parzen_bandwidth),
        block_rows=int(kernel_block_rows),
        kernel_dtype=kernel_dtype,
        backend=compute_backend,
    ).objective
    innovation_effective = aligned[1] @ whitening.whitening
    return PerContextICAFit(
        context_id=str(context_id),
        sample_count=int(len(confirmation)),
        center=whitening.mean.astype(np.float64),
        whitening=whitening.whitening.astype(np.float64),
        rotation_angle_degrees=angle,
        rotation=_rotation(angle),
        demixing=aligned.astype(np.float64),
        persistence_index=persistence,
        innovation_index=innovation,
        component_signs=signs,
        objective_name=objective,
        objective_value=float(selected_objective),
        baseline_objective_value=float(baseline),
        derivative_angle_distance_degrees=derivative_angle_distance(
            innovation_effective
        ),
        converged=bool(fitted.converged),
        ambiguous_alignment=ambiguous,
        diagnostics={
            "alignment": alignment,
            "whitening": {
                "covariance": whitening.covariance.tolist(),
                "eigenvalues": whitening.eigenvalues.tolist(),
                "condition_number": whitening.condition_number,
                "identifiable": whitening.identifiable,
            },
            "optimizer": fitted.diagnostics,
        },
    )


def apply_per_context_fit(
    pairs: np.ndarray,
    fit: PerContextICAFit,
) -> np.ndarray:
    """Apply a fit and return columns [persistence, innovation]."""
    values = _pairs(pairs, "pairs")
    output = apply_linear_separation(
        values.T, fit.center, fit.whitening, fit.demixing
    )
    return output.T


def contiguous_block_bootstrap(
    context_id: str,
    pairs: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
    fitter_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Refit deterministic contiguous-block bootstrap samples."""
    values = _pairs(pairs, "pairs")
    if block_length < 2 or block_length > len(values) or replicates < 1:
        raise ValueError("invalid block bootstrap dimensions")
    rng = np.random.default_rng(int(seed))
    rows = []
    starts = np.arange(0, len(values) - block_length + 1)
    blocks_needed = int(np.ceil(len(values) / block_length))
    kwargs = dict(fitter_kwargs or {})
    for replicate in range(int(replicates)):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in selected]
        )[: len(values)]
        fitted = fit_per_context_ica(
            context_id,
            values[indices],
            values[indices],
            **kwargs,
        )
        rows.append(
            {
                "replicate": replicate,
                "angle_degrees": fitted.rotation_angle_degrees,
                "derivative_angle_distance_degrees": (
                    fitted.derivative_angle_distance_degrees
                ),
                "objective_value": fitted.objective_value,
                "baseline_objective_value": fitted.baseline_objective_value,
                "component_swap": fitted.persistence_index == 1,
                "ambiguous_alignment": fitted.ambiguous_alignment,
            }
        )
    return rows


def bootstrap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("bootstrap rows are required")
    angles = np.deg2rad([2.0 * float(row["angle_degrees"]) for row in rows])
    resultant = abs(np.mean(np.exp(1j * angles)))
    circular_std = (
        float(np.rad2deg(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-12)), 0))) / 2.0)
    )
    raw = np.asarray([row["angle_degrees"] for row in rows], dtype=np.float64)
    return {
        "replicates": len(rows),
        "median_angle_degrees": float(np.median(raw)),
        "circular_std_degrees": circular_std,
        "angle_p05_degrees": float(np.percentile(raw, 5)),
        "angle_p95_degrees": float(np.percentile(raw, 95)),
        "component_swap_fraction": float(
            np.mean([bool(row["component_swap"]) for row in rows])
        ),
        "ambiguous_fraction": float(
            np.mean([bool(row["ambiguous_alignment"]) for row in rows])
        ),
    }


def fit_cross_context(
    values: np.ndarray,
    *,
    mode: Literal["identity", "pca", "fastica"],
    max_components: int = 8,
    max_samples: int = 32768,
    seed: int = 0,
) -> CrossContextFit:
    """Fit a bounded global transform to a sampled innovation bank."""
    matrix = _matrix(values, "values")
    if len(matrix) > int(max_samples):
        raise ValueError("cross-context sample cap exceeded")
    components = min(int(max_components), matrix.shape[1])
    if components < 1:
        raise ValueError("at least one component is required")
    center = matrix.mean(axis=0)
    centered = matrix - center
    explained: np.ndarray | None = None
    diagnostics: dict[str, Any] = {}
    if mode == "identity":
        components = matrix.shape[1]
        transform = np.eye(components)
        inverse = np.eye(components)
        converged = True
    elif mode == "pca":
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        transform = vt[:components]
        inverse = transform.T
        variance = singular * singular / max(len(matrix) - 1, 1)
        explained = variance[:components] / max(float(np.sum(variance)), 1e-12)
        converged = True
    elif mode == "fastica":
        from sklearn.decomposition import FastICA

        model = FastICA(
            n_components=components,
            whiten="unit-variance",
            random_state=int(seed),
            max_iter=1000,
            tol=1e-5,
        )
        model.fit(matrix)
        center = np.asarray(model.mean_, dtype=np.float64)
        transform = np.asarray(model.components_, dtype=np.float64)
        inverse = np.asarray(model.mixing_, dtype=np.float64)
        converged = int(model.n_iter_) < int(model.max_iter)
        diagnostics["iterations"] = int(model.n_iter_)
    else:
        raise ValueError("mode must be identity, pca, or fastica")
    return CrossContextFit(
        mode=mode,
        input_count=matrix.shape[1],
        component_count=components,
        center=np.asarray(center, dtype=np.float64),
        transform=np.asarray(transform, dtype=np.float64),
        inverse_transform=np.asarray(inverse, dtype=np.float64),
        explained_variance=explained,
        converged=converged,
        seed=int(seed),
        sample_count=len(matrix),
        diagnostics=diagnostics,
    )


def predeclared_group_energy(
    standardized_components: np.ndarray,
    groups: dict[str, list[int] | tuple[int, ...]],
) -> dict[str, np.ndarray]:
    matrix = _matrix(standardized_components, "standardized_components")
    result: dict[str, np.ndarray] = {}
    for group_id, raw_indices in groups.items():
        indices = np.asarray(tuple(raw_indices), dtype=np.int64)
        if (
            not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= matrix.shape[1])
        ):
            raise ValueError(f"invalid component indices for group {group_id}")
        result[group_id] = group_energy(matrix[:, indices], axis=1)
    return result
