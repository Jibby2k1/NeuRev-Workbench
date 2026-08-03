"""Canonical validation contract for imported provider local-PCA factors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from neurobench.algorithms.dependent_multiscale import (
    LocalFactorization, reconstruct_local_factorization,
)


@dataclass(frozen=True)
class ProviderLocalPCARecord:
    source_movie_checksum: str
    frame_range_zero_half_open: tuple[int, int]
    coordinate_convention: str
    patch_size_yx: tuple[int, int]
    stride_yx: tuple[int, int]
    centering_rule: str
    normalization_rule: str
    rank_selection_rule: str
    overlap_window: str
    blending_rule: str
    software_provenance: str
    factors: tuple[LocalFactorization, ...]
    diagnostics: dict[str, Any]


def validate_provider_local_pca(
    record: ProviderLocalPCARecord,
    *,
    movie_shape: tuple[int, int, int],
    reconstructed_patches: Sequence[np.ndarray] | None = None,
    closure_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Validate metadata and shapes without guessing a missing convention."""
    errors: list[str] = []
    t, y_size, x_size = (int(value) for value in movie_shape)
    start, stop = record.frame_range_zero_half_open
    if not record.source_movie_checksum.strip():
        errors.append("source movie checksum is missing")
    if not (0 <= start < stop and stop - start == t):
        errors.append("frame range does not match movie time dimension")
    if record.coordinate_convention != "x=column,y=row;zero_based":
        errors.append("coordinate convention is not canonical")
    if any(value <= 0 for value in (*record.patch_size_yx, *record.stride_yx)):
        errors.append("patch and stride dimensions must be positive")
    required_text = (
        record.centering_rule,
        record.normalization_rule,
        record.rank_selection_rule,
        record.overlap_window,
        record.blending_rule,
        record.software_provenance,
    )
    if any(not value.strip() for value in required_text):
        errors.append("provider metadata contains an unspecified rule")
    for fit in record.factors:
        y0, x0 = fit.origin_yx
        py, px = fit.shape_yx
        if y0 < 0 or x0 < 0 or y0 + py > y_size or x0 + px > x_size:
            errors.append(f"{fit.patch_id}: patch is outside movie bounds")
        if fit.spatial_factors.shape != (py * px, fit.rank):
            errors.append(f"{fit.patch_id}: spatial factor shape is incompatible")
        if fit.temporal_factors.shape != (fit.rank, t):
            errors.append(f"{fit.patch_id}: temporal factor shape is incompatible")
        if fit.component_energy.shape != (fit.rank,):
            errors.append(f"{fit.patch_id}: component energies do not match rank")
        arrays = (fit.spatial_factors, fit.temporal_factors, fit.component_energy)
        if any(not np.isfinite(value).all() for value in arrays):
            errors.append(f"{fit.patch_id}: factors contain non-finite values")
    closure_checked = reconstructed_patches is not None
    closure_max = None
    if reconstructed_patches is not None:
        if len(reconstructed_patches) != len(record.factors):
            errors.append("reconstruction patch count does not match factor count")
        else:
            closure_values = []
            for fit, original in zip(record.factors, reconstructed_patches):
                source = np.asarray(original, dtype=np.float64)
                restored = reconstruct_local_factorization(fit).astype(np.float64)
                if source.shape != restored.shape or not np.isfinite(source).all():
                    errors.append(f"{fit.patch_id}: original patch is incompatible")
                    continue
                denominator = max(float(np.sum(source**2)), np.finfo(float).eps)
                closure_values.append(float(np.sum((source - restored) ** 2) / denominator))
            closure_max = max(closure_values, default=float("inf"))
            if closure_max > float(closure_tolerance):
                errors.append("provider reconstruction closure exceeds tolerance")
    valid = not errors and closure_checked
    return {
        "status": "valid_initializer" if valid else "external_baseline_only",
        "valid_initializer": valid,
        "errors": errors,
        "factor_count": len(record.factors),
        "closure_checked": closure_checked,
        "maximum_reconstruction_nmse": closure_max,
        "silent_inference_performed": False,
    }
