"""Pure-array foundations for dependent multiscale demixing.

Movies use ``[T,Y,X]`` orientation. Local spatial factors use ``[P,R]`` and
temporal factors use ``[R,T]``. This module performs no filesystem access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter


@dataclass(frozen=True)
class ScaleViewSpec:
    view_id: str
    support_px: int
    operator_kind: str
    normalization_kind: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class LocalFactorization:
    patch_id: str
    view_id: str
    origin_yx: tuple[int, int]
    shape_yx: tuple[int, int]
    rank: int
    spatial_factors: np.ndarray
    temporal_factors: np.ndarray
    component_energy: np.ndarray
    reconstruction_nmse: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class JointQuietNoiseModel:
    view_ids: tuple[str, ...]
    center: np.ndarray
    covariance: np.ndarray
    inverse_sqrt: np.ndarray
    eigenvalues: np.ndarray
    kernel_bandwidth: float
    reference_samples: np.ndarray | None
    model_kind: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DependencyEdge:
    left_group: str
    right_group: str
    relation: str
    conditioning_variables: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class DependencyGraph:
    groups: tuple[str, ...]
    edges: tuple[DependencyEdge, ...]


@dataclass(frozen=True)
class DependentMultiscaleFit:
    patch_id: str
    view_ids: tuple[str, ...]
    neural_latent: np.ndarray
    background_latent: np.ndarray
    artifact_latent: np.ndarray | None
    private_residuals: tuple[np.ndarray, ...]
    neural_loadings: tuple[np.ndarray, ...]
    background_loadings: tuple[np.ndarray, ...]
    artifact_loadings: tuple[np.ndarray, ...] | None
    dependency_graph: DependencyGraph
    objective_history: tuple[float, ...]
    converged: bool
    iterations: int
    seed: int
    assignment_status: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PatchDecomposition:
    patch_id: str
    background: np.ndarray
    structured_signal: np.ndarray
    structured_artifact: np.ndarray
    noise_candidate: np.ndarray
    closure_residual: np.ndarray
    posterior_uncertainty: np.ndarray | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DependentMultiscaleRunSummary:
    experiment_id: str
    status: str
    scale_views: tuple[str, ...]
    patch_count: int
    resolved_patch_count: int
    fallback_patch_count: int
    failed_patch_count: int
    closure_metrics: dict[str, float]
    attribution_metrics: dict[str, Any]
    residual_metrics: dict[str, Any]
    stability_metrics: dict[str, Any]
    resource_metrics: dict[str, Any]


def default_dependency_graph() -> DependencyGraph:
    """Return the explicit frozen v1 dependence contract."""
    nuisance = ("global_intensity", "translation_motion", "slow_drift")
    return DependencyGraph(
        groups=("neural", "background", "artifact", "private_residual"),
        edges=(
            DependencyEdge("neural", "neural", "preserve", (), 1.0),
            DependencyEdge("background", "background", "preserve", (), 1.0),
            DependencyEdge("artifact", "artifact", "preserve", (), 1.0),
            DependencyEdge("neural", "background", "penalize", nuisance, 0.05),
            DependencyEdge("neural", "artifact", "penalize", nuisance, 0.05),
            DependencyEdge("background", "artifact", "explain", nuisance, 0.0),
            DependencyEdge("private_residual", "private_residual", "preserve", (), 1.0),
        ),
    )


def _movie(values: np.ndarray, name: str = "observation") -> np.ndarray:
    movie = np.asarray(values, dtype=np.float64)
    if movie.ndim != 3 or not movie.size or not np.isfinite(movie).all():
        raise ValueError(f"{name} must be a non-empty finite [T,Y,X] array")
    return movie


def validate_nested_scale_specs(specs: Sequence[ScaleViewSpec]) -> None:
    if not specs:
        raise ValueError("at least one scale view is required")
    ids = [spec.view_id for spec in specs]
    supports = [int(spec.support_px) for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("scale view IDs must be unique")
    if any(value < 3 or value % 2 != 1 for value in supports):
        raise ValueError("support sizes must be odd integers of at least three")
    claims_nested = any(bool(spec.parameters.get("nested", False)) for spec in specs)
    if claims_nested and supports != sorted(set(supports)):
        raise ValueError("operators claiming nesting require strictly increasing supports")


def build_scale_views(
    observation: np.ndarray,
    specs: Sequence[ScaleViewSpec],
    quiet_count: int,
) -> dict[str, np.ndarray]:
    """Build deterministic, aligned local-support views from quiet-only scaling."""
    # Full review movies remain float32 here. Local factorization uses float64,
    # but promoting a complete real movie for spatial views adds substantial
    # memory without changing the float32 view artifact contract.
    movie = np.asarray(observation, dtype=np.float32)
    if movie.ndim != 3 or not movie.size or not np.isfinite(movie).all():
        raise ValueError("observation must be a non-empty finite [T,Y,X] array")
    validate_nested_scale_specs(specs)
    quiet = int(quiet_count)
    if not 2 <= quiet < len(movie):
        raise ValueError("quiet_count must retain at least one non-quiet frame")
    output: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.operator_kind not in {
            "normalized_box_support", "quiet_normalized_local_support"
        }:
            raise ValueError(f"unsupported operator_kind: {spec.operator_kind}")
        if spec.normalization_kind == "quiet_robust":
            center = np.median(movie[:quiet], axis=0)
            mad = 1.4826 * np.median(np.abs(movie[:quiet] - center), axis=0)
            positive = mad[mad > np.finfo(np.float32).eps]
            floor = float(np.median(positive)) * 1e-3 if positive.size else 1.0
            normalized = (movie - center) / np.maximum(mad, floor)
        elif spec.normalization_kind == "none":
            normalized = movie
        else:
            raise ValueError(f"unsupported normalization_kind: {spec.normalization_kind}")
        support = int(spec.support_px)
        view = uniform_filter(
            normalized, size=(1, support, support), mode="reflect"
        )
        output[spec.view_id] = view.astype(np.float32)
    return output


def fit_local_pca(
    patch: np.ndarray,
    *,
    patch_id: str,
    view_id: str,
    origin_yx: tuple[int, int],
    rank: int,
) -> LocalFactorization:
    """Fit an explicitly centered local PCA to one ``[T,Y,X]`` patch."""
    movie = _movie(patch, "patch")
    pixels_by_time = movie.reshape(len(movie), -1).T
    pixel_center = pixels_by_time.mean(axis=1, keepdims=True)
    centered = pixels_by_time - pixel_center
    selected_rank = int(rank)
    if not 1 <= selected_rank <= min(centered.shape):
        raise ValueError("rank is outside local factor dimensions")
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    spatial = u[:, :selected_rank] * singular[:selected_rank]
    temporal = vt[:selected_rank]
    reconstruction = spatial @ temporal
    denominator = max(float(np.sum(centered**2)), np.finfo(float).eps)
    nmse = float(np.sum((centered - reconstruction) ** 2) / denominator)
    return LocalFactorization(
        patch_id=str(patch_id),
        view_id=str(view_id),
        origin_yx=(int(origin_yx[0]), int(origin_yx[1])),
        shape_yx=(int(movie.shape[1]), int(movie.shape[2])),
        rank=selected_rank,
        spatial_factors=spatial.astype(np.float32),
        temporal_factors=temporal.astype(np.float32),
        component_energy=np.square(singular[:selected_rank]).astype(np.float64),
        reconstruction_nmse=nmse,
        diagnostics={
            "orientation": "spatial_factors[P,R], temporal_factors[R,T]",
            "dtype": "float32",
            "centering": "per_pixel_temporal_mean",
            "pixel_center": pixel_center[:, 0].astype(np.float32),
            "labels_used": False,
        },
    )


def reconstruct_local_factorization(fit: LocalFactorization) -> np.ndarray:
    restored = np.asarray(fit.spatial_factors) @ np.asarray(fit.temporal_factors)
    center = np.asarray(fit.diagnostics["pixel_center"], dtype=np.float64)
    restored = restored + center[:, None]
    y, x = fit.shape_yx
    return restored.T.reshape(-1, y, x).astype(np.float32)


def residualize_nuisance(values: np.ndarray, nuisance: np.ndarray | None) -> np.ndarray:
    """Project declared nuisance columns from sample-by-feature coordinates."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("values must be finite [samples,features]")
    if nuisance is None:
        return data.copy()
    design = np.asarray(nuisance, dtype=np.float64)
    if design.ndim != 2 or len(design) != len(data) or not np.isfinite(design).all():
        raise ValueError("nuisance must be finite and sample-aligned")
    design = np.column_stack((np.ones(len(design)), design))
    return data - design @ np.linalg.lstsq(design, data, rcond=None)[0]


def orthogonal_shared_private(
    factor_coordinates: Mapping[str, np.ndarray], *, shared_rank: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Return a deterministic JIVE-like shared/private orthogonal baseline.

    Inputs are view-specific ``[samples,features]`` matrices with the same
    sample count. Shared temporal scores come from the concatenated feature
    matrix; each view is projected onto and away from that shared subspace.
    """
    if not factor_coordinates:
        raise ValueError("factor coordinates cannot be empty")
    ids = tuple(sorted(factor_coordinates))
    arrays = {key: np.asarray(factor_coordinates[key], dtype=np.float64) for key in ids}
    sample_count = len(next(iter(arrays.values())))
    if any(
        value.ndim != 2 or len(value) != sample_count or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("factor coordinates must be finite and sample-aligned")
    centered = {key: value - value.mean(axis=0, keepdims=True) for key, value in arrays.items()}
    joined = np.concatenate([centered[key] for key in ids], axis=1)
    rank = int(shared_rank)
    if not 1 <= rank <= min(joined.shape):
        raise ValueError("shared_rank is outside joined factor dimensions")
    left, _, _ = np.linalg.svd(joined, full_matrices=False)
    basis = left[:, :rank]
    shared = {key: basis @ (basis.T @ centered[key]) for key in ids}
    private = {key: centered[key] - shared[key] for key in ids}
    maximum_cross = max(
        float(np.max(np.abs(shared[key].T @ private[key]))) for key in ids
    )
    return shared, private, {
        "view_ids": ids,
        "shared_rank": rank,
        "maximum_shared_private_cross_product": maximum_cross,
        "interpretation": "orthogonality is numerical, not physical independence",
    }


def decompose_patch_baseline(
    observation_patch: np.ndarray,
    scale_views: Mapping[str, np.ndarray],
    *,
    patch_id: str,
) -> PatchDecomposition:
    """Create the reversible W3 reference decomposition in original coordinates.

    The baseline assigns the broad shared projection to
    background, compact cross-scale agreement to structured signal, coherent
    compact disagreement to artifact, and closes exactly with noise_candidate.
    It is a structural baseline, not a qualified physical separation.
    """
    observation = _movie(observation_patch, "observation_patch")
    required = ("scale_5", "scale_7", "scale_15")
    if tuple(sorted(scale_views)) != tuple(sorted(required)):
        raise ValueError("baseline requires exactly scale_5, scale_7, scale_15")
    views = {key: _movie(scale_views[key], key) for key in required}
    if any(value.shape != observation.shape for value in views.values()):
        raise ValueError("all patch views must align with the observation")
    coordinates = {key: value.reshape(len(value), -1) for key, value in views.items()}
    shared, private, jive_diagnostics = orthogonal_shared_private(
        coordinates, shared_rank=min(4, len(observation) - 1)
    )
    means = {
        key: coordinates[key].mean(axis=0, keepdims=True) for key in required
    }
    shared_original = {
        key: (shared[key] + means[key]).reshape(observation.shape)
        for key in required
    }
    background = shared_original["scale_15"]
    compact = 0.5 * (shared_original["scale_5"] + shared_original["scale_7"])
    structured_signal = compact - background
    structured_artifact = (
        0.5 * (private["scale_5"] - private["scale_7"])
    ).reshape(observation.shape)
    noise_candidate = observation - background - structured_signal - structured_artifact
    closure = observation - background - structured_signal - structured_artifact - noise_candidate
    scale = max(float(np.max(np.abs(observation))), np.finfo(float).eps)
    return PatchDecomposition(
        patch_id=str(patch_id),
        background=background.astype(np.float32),
        structured_signal=structured_signal.astype(np.float32),
        structured_artifact=structured_artifact.astype(np.float32),
        noise_candidate=noise_candidate.astype(np.float32),
        closure_residual=closure.astype(np.float32),
        posterior_uncertainty=None,
        diagnostics={
            "method_id": "orthogonal_shared_private",
            "shared_private": jive_diagnostics,
            "artifact_status": "structural_compact_disagreement",
            "noise_status": "noise_candidate",
            "closure_max_normalized": float(np.max(np.abs(closure)) / scale),
            "labels_used": False,
        },
    )


def overlap_add(
    patches: Sequence[tuple[tuple[int, int], np.ndarray]],
    output_shape: tuple[int, int, int],
    *,
    floor: float = 0.1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blend ``[T,Yp,Xp]`` patches with a separable floored Hann window."""
    if not patches:
        raise ValueError("at least one patch is required")
    t, y_size, x_size = (int(value) for value in output_shape)
    total = np.zeros((t, y_size, x_size), dtype=np.float64)
    weights = np.zeros((y_size, x_size), dtype=np.float64)
    for (y0, x0), raw in patches:
        patch = _movie(raw, "patch")
        if len(patch) != t or y0 < 0 or x0 < 0 or y0 + patch.shape[1] > y_size or x0 + patch.shape[2] > x_size:
            raise ValueError("patch lies outside overlap-add output")
        wy = np.maximum(np.hanning(patch.shape[1]), float(floor))
        wx = np.maximum(np.hanning(patch.shape[2]), float(floor))
        window = wy[:, None] * wx[None, :]
        total[:, y0:y0 + patch.shape[1], x0:x0 + patch.shape[2]] += patch * window
        weights[y0:y0 + patch.shape[1], x0:x0 + patch.shape[2]] += window
    if np.any(weights <= 0):
        raise ValueError("overlap-add coverage has a zero denominator")
    restored = total / weights[None]
    return restored.astype(np.float32), {
        "denominator_minimum": float(weights.min()),
        "denominator_maximum": float(weights.max()),
    }
