"""CPU reference for quiet-fitted global and tiled covariance whitening.

All covariance fitting is float64. Scientific arrays and application outputs are
float32. The module never assigns biological meaning to whitened coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from neurobench.metrics.whiteness import effective_rank


Estimator = Literal["oas", "ledoit_wolf", "fixed_ridge", "diagonal"]


@dataclass(frozen=True)
class WhiteningFeatureBank:
    feature_ids: tuple[str, ...]
    values: np.ndarray
    valid_frames: np.ndarray
    source_contexts: tuple[dict[str, object], ...]
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        valid = np.asarray(self.valid_frames)
        if (
            values.ndim != 4
            or values.dtype != np.float32
            or not values.size
            or not np.isfinite(values).all()
            or values.shape[-1] != len(self.feature_ids)
            or len(set(self.feature_ids)) != len(self.feature_ids)
            or valid.dtype != np.bool_
            or valid.shape != (len(values),)
            or len(self.source_contexts) != len(self.feature_ids)
        ):
            raise ValueError("invalid WhiteningFeatureBank")
        for context in self.source_contexts:
            if context.get("display_clipped", False) or context.get("squared", False):
                raise ValueError("display-clipped or squared arrays cannot enter the feature bank")
            if context.get("scientific_array", "signed_msln") != "signed_msln":
                raise ValueError("feature bank requires signed scientific MSLN arrays")


@dataclass(frozen=True)
class CovarianceEstimatorConfig:
    method: Estimator = "oas"
    ridge_ratio: float = 0.05
    eigenvalue_floor_ratio: float = 1e-5
    maximum_condition_number: float = 1e4
    center: Literal["mean", "median"] = "mean"

    def __post_init__(self) -> None:
        if self.method not in {"oas", "ledoit_wolf", "fixed_ridge", "diagonal"}:
            raise ValueError("unsupported covariance estimator")
        if not 0 <= self.ridge_ratio <= 1:
            raise ValueError("ridge_ratio must lie in [0,1]")
        if not 0 < self.eigenvalue_floor_ratio < 1:
            raise ValueError("eigenvalue_floor_ratio must lie in (0,1)")
        if self.maximum_condition_number <= 1 or self.center not in {"mean", "median"}:
            raise ValueError("invalid conditioning cap or center")


@dataclass(frozen=True)
class SpatialTileConfig:
    tile_height: int = 64
    tile_width: int = 64
    stride_y: int = 32
    stride_x: int = 32
    blend: Literal["hann", "triangular", "uniform"] = "hann"
    boundary_mode: Literal["crop", "reflect"] = "crop"
    minimum_raw_samples: int = 64
    maximum_fit_samples: int = 65536
    spatial_subsample: int = 2
    temporal_subsample: int = 1

    def __post_init__(self) -> None:
        integers = (
            self.tile_height, self.tile_width, self.stride_y, self.stride_x,
            self.minimum_raw_samples, self.maximum_fit_samples,
            self.spatial_subsample, self.temporal_subsample,
        )
        if any(int(item) != item or item < 1 for item in integers):
            raise ValueError("tile counts, sizes, strides, and subsampling must be positive integers")
        if self.stride_y > self.tile_height or self.stride_x > self.tile_width:
            raise ValueError("tile stride cannot exceed tile size")
        if self.minimum_raw_samples > self.maximum_fit_samples:
            raise ValueError("minimum_raw_samples cannot exceed maximum_fit_samples")
        if self.blend not in {"hann", "triangular", "uniform"}:
            raise ValueError("unsupported tile blend")
        if self.boundary_mode not in {"crop", "reflect"}:
            raise ValueError("unsupported boundary mode")


@dataclass(frozen=True)
class LocalCovarianceFit:
    fit_id: str
    feature_ids: tuple[str, ...]
    tile_bounds_yx: tuple[int, int, int, int] | None
    mean: np.ndarray
    sample_covariance: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    eigenvalues_raw: np.ndarray
    eigenvalues_regularized: np.ndarray
    shrinkage: float
    condition_number: float
    effective_rank: float
    sample_count: int
    block_count: int
    resolved: bool
    unresolved_reason: str | None
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        dimension = len(self.feature_ids)
        matrices = (self.sample_covariance, self.covariance, self.whitening)
        if (
            not self.fit_id
            or dimension < 1
            or np.asarray(self.mean).shape != (dimension,)
            or any(np.asarray(item).shape != (dimension, dimension) for item in matrices)
            or np.asarray(self.eigenvalues_raw).shape != (dimension,)
            or np.asarray(self.eigenvalues_regularized).shape != (dimension,)
            or not all(np.isfinite(item).all() for item in (self.mean, *matrices, self.eigenvalues_raw, self.eigenvalues_regularized))
            or self.sample_count < 0
            or self.block_count < 0
            or (self.resolved and self.unresolved_reason is not None)
            or (not self.resolved and not self.unresolved_reason)
        ):
            raise ValueError("invalid LocalCovarianceFit")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "mean", "sample_covariance", "covariance", "whitening",
            "eigenvalues_raw", "eigenvalues_regularized",
        ):
            payload[key] = np.asarray(payload[key]).tolist()
        payload["feature_ids"] = list(self.feature_ids)
        payload["tile_bounds_yx"] = None if self.tile_bounds_yx is None else list(self.tile_bounds_yx)
        return payload


@dataclass(frozen=True)
class TiledWhiteningFits:
    feature_ids: tuple[str, ...]
    tile_bounds_yx: tuple[tuple[int, int, int, int], ...]
    primary_fits: tuple[LocalCovarianceFit, ...]
    local_diagonal_fits: tuple[LocalCovarianceFit, ...]
    global_full_fit: LocalCovarianceFit
    tile_config: SpatialTileConfig
    fallback_policy: tuple[str, ...] = (
        "matching_global_full_fit", "local_diagonal", "recorded_identity"
    )

    def __post_init__(self) -> None:
        count = len(self.tile_bounds_yx)
        if count == 0 or len(self.primary_fits) != count or len(self.local_diagonal_fits) != count:
            raise ValueError("tiled fits must align with tile geometry")


@dataclass(frozen=True)
class EmpiricalQuietCalibration:
    sorted_energy: np.ndarray
    probability_floor: float
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.sorted_energy)
        if (
            values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or np.any(np.diff(values) < 0)
            or not 0 < self.probability_floor <= 1
        ):
            raise ValueError("invalid empirical quiet calibration")

    def survival_probability(self, values: np.ndarray) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        if not np.isfinite(query).all():
            raise ValueError("energy query must be finite")
        first_ge = np.searchsorted(self.sorted_energy, query, side="left")
        probability = (len(self.sorted_energy) - first_ge + 1.0) / (len(self.sorted_energy) + 1.0)
        return np.maximum(probability, self.probability_floor)

    def surprise(self, values: np.ndarray) -> np.ndarray:
        return (-np.log10(self.survival_probability(values))).astype(np.float32)


@dataclass(frozen=True)
class LocalWhiteningResult:
    zca_features: np.ndarray
    mahalanobis_energy: np.ndarray
    quiet_surprise: np.ndarray
    blend_weight: np.ndarray
    unresolved_tile_mask: np.ndarray
    valid_frames: np.ndarray
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        zca = np.asarray(self.zca_features)
        energy = np.asarray(self.mahalanobis_energy)
        surprise = np.asarray(self.quiet_surprise)
        if (
            zca.ndim != 4 or zca.dtype != np.float32 or not np.isfinite(zca).all()
            or energy.shape != zca.shape[:3] or energy.dtype != np.float32
            or surprise.shape != energy.shape or surprise.dtype != np.float32
            or np.asarray(self.blend_weight).shape != zca.shape[1:3]
            or np.asarray(self.unresolved_tile_mask).shape != zca.shape[1:3]
            or np.asarray(self.valid_frames).shape != (len(zca),)
            or np.any(np.asarray(self.blend_weight) <= 0)
        ):
            raise ValueError("invalid LocalWhiteningResult")


def build_whitening_feature_bank(
    feature_ids: Sequence[str],
    msln_results: Sequence[Any],
) -> WhiteningFeatureBank:
    """Stack aligned signed ``MSLNResult`` objects into frozen ``T,H,W,D`` order."""
    ids = tuple(str(item) for item in feature_ids)
    if len(ids) != len(msln_results) or not ids:
        raise ValueError("feature IDs and MSLN results must align")
    arrays = [np.asarray(item.values) for item in msln_results]
    if any(array.ndim != 3 or array.dtype != np.float32 for array in arrays):
        raise ValueError("MSLN arrays must be float32 TYX")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("MSLN arrays must share TYX shape")
    valid = np.logical_and.reduce([np.asarray(item.valid_frames, dtype=bool) for item in msln_results])
    contexts = tuple({
        "context_id": feature_id,
        "scientific_array": "signed_msln",
        "display_clipped": False,
        "squared": False,
        "scale_floor": float(item.scale_floor),
        "support": dict(item.diagnostics),
    } for feature_id, item in zip(ids, msln_results))
    return WhiteningFeatureBank(
        feature_ids=ids,
        values=np.stack(arrays, axis=-1).astype(np.float32, copy=False),
        valid_frames=valid,
        source_contexts=contexts,
        diagnostics={"feature_order_frozen": True, "scientific_array": "signed_msln"},
    )


def contiguous_quiet_partitions(
    quiet_mask: np.ndarray,
    fractions: Sequence[float] = (0.5, 0.25, 0.25),
) -> dict[str, np.ndarray]:
    """Split consecutive selected quiet frames into fit/calibration/holdout masks."""
    mask = np.asarray(quiet_mask, dtype=bool)
    weights = np.asarray(fractions, dtype=np.float64)
    if mask.ndim != 1 or weights.shape != (3,) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("quiet mask and three positive fractions summing to one are required")
    indices = np.flatnonzero(mask)
    if len(indices) < 6 or np.any(np.diff(indices) != 1):
        raise ValueError("quiet support must be one contiguous block with at least six frames")
    first = max(1, int(np.floor(len(indices) * weights[0])))
    second = max(first + 1, int(np.floor(len(indices) * (weights[0] + weights[1]))))
    second = min(second, len(indices) - 1)
    pieces = (indices[:first], indices[first:second], indices[second:])
    if any(len(piece) == 0 for piece in pieces):
        raise ValueError("every quiet partition must contain at least one frame")
    result = {}
    for name, piece in zip(("fit", "calibration", "holdout"), pieces):
        selected = np.zeros_like(mask); selected[piece] = True; result[name] = selected
    return result


def enumerate_spatial_tiles(
    height: int, width: int, config: SpatialTileConfig
) -> tuple[tuple[int, int, int, int], ...]:
    """Return deterministic crop-mode tiles, forcing final starts to the field edge."""
    if config.tile_height > height or config.tile_width > width:
        raise ValueError("tile size cannot exceed the field in V1")
    if config.boundary_mode == "reflect":
        raise ValueError("reflect boundary fitting is not implemented in the V1 CPU reference; preflight must reject it")
    def starts(length: int, size: int, stride: int) -> list[int]:
        result = list(range(0, length - size + 1, stride))
        final = length - size
        if result[-1] != final:
            result.append(final)
        return result
    tiles = tuple(
        (y, y + config.tile_height, x, x + config.tile_width)
        for y in starts(height, config.tile_height, config.stride_y)
        for x in starts(width, config.tile_width, config.stride_x)
    )
    coverage = np.zeros((height, width), dtype=np.int32)
    for y0, y1, x0, x1 in tiles:
        coverage[y0:y1, x0:x1] += 1
    if np.any(coverage == 0):
        raise ValueError("declared tile geometry does not cover the field")
    return tiles


def tile_blend_weight(height: int, width: int, blend: str) -> np.ndarray:
    """Return a strictly positive deterministic 2-D overlap weight."""
    if blend == "uniform":
        return np.ones((height, width), dtype=np.float32)
    if blend == "hann":
        y = np.hanning(height + 2)[1:-1]
        x = np.hanning(width + 2)[1:-1]
    elif blend == "triangular":
        y = 1.0 - np.abs((np.arange(height) + 0.5) / height * 2.0 - 1.0)
        x = 1.0 - np.abs((np.arange(width) + 0.5) / width * 2.0 - 1.0)
    else:
        raise ValueError("unsupported blend")
    return np.maximum(np.outer(y, x), np.finfo(np.float32).eps).astype(np.float32)


def gather_covariance_samples(
    feature_bank: WhiteningFeatureBank,
    quiet_fit_mask: np.ndarray,
    *,
    tile_bounds_yx: tuple[int, int, int, int] | None = None,
    maximum_samples: int = 65536,
    spatial_subsample: int = 1,
    temporal_subsample: int = 1,
) -> tuple[np.ndarray, dict[str, int]]:
    """Gather deterministic lexicographic ``[N,D]`` quiet samples with a hard cap."""
    quiet = np.asarray(quiet_fit_mask, dtype=bool)
    if quiet.shape != feature_bank.valid_frames.shape:
        raise ValueError("quiet mask must align with feature-bank frames")
    frames = np.flatnonzero(quiet & feature_bank.valid_frames)[::int(temporal_subsample)]
    if tile_bounds_yx is None:
        y0, y1, x0, x1 = 0, feature_bank.values.shape[1], 0, feature_bank.values.shape[2]
    else:
        y0, y1, x0, x1 = map(int, tile_bounds_yx)
    selected = feature_bank.values[
        frames,
        y0:y1:int(spatial_subsample),
        x0:x1:int(spatial_subsample),
        :,
    ].reshape(-1, len(feature_bank.feature_ids))
    if not np.isfinite(selected).all():
        raise ValueError("covariance samples must be finite")
    raw_count = len(selected)
    if raw_count > int(maximum_samples):
        indices = np.linspace(0, raw_count - 1, int(maximum_samples), dtype=np.int64)
        selected = selected[indices]
    return selected.astype(np.float64), {
        "raw_sample_count": int(raw_count),
        "selected_sample_count": int(len(selected)),
        "frame_count": int(len(frames)),
        "blocked_effective_sample_count": int(len(frames)),
    }


def _unresolved_fit(
    dimension: int,
    feature_ids: tuple[str, ...],
    fit_id: str,
    bounds: tuple[int, int, int, int] | None,
    sample_count: int,
    block_count: int,
    reason: str,
) -> LocalCovarianceFit:
    identity = np.eye(dimension, dtype=np.float64)
    return LocalCovarianceFit(
        fit_id, feature_ids, bounds, np.zeros(dimension), identity, identity,
        identity, np.ones(dimension), np.ones(dimension), 1.0, 1.0,
        float(dimension), sample_count, block_count, False, reason,
        {"fallback_required": True, "identity_is_placeholder_not_success": True},
    )


def fit_covariance(
    samples: np.ndarray,
    config: CovarianceEstimatorConfig,
    *,
    fit_id: str = "global_covariance_fit",
    feature_ids: Sequence[str] | None = None,
    tile_bounds_yx: tuple[int, int, int, int] | None = None,
    block_count: int = 1,
) -> LocalCovarianceFit:
    """Fit a regularized covariance and explicit ZCA transform in float64."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("samples must be a finite N,D matrix")
    ids = tuple(feature_ids or (f"feature_{index:02d}" for index in range(values.shape[1])))
    if len(ids) != values.shape[1]:
        raise ValueError("feature IDs must match sample columns")
    if len(values) < max(2, values.shape[1] + 1):
        return _unresolved_fit(values.shape[1], ids, fit_id, tile_bounds_yx, len(values), block_count, "insufficient_samples")
    center = np.mean(values, axis=0) if config.center == "mean" else np.median(values, axis=0)
    centered = values - center
    sample_covariance = (centered.T @ centered) / len(centered)
    sample_covariance = 0.5 * (sample_covariance + sample_covariance.T)
    target_scale = float(np.trace(sample_covariance) / values.shape[1])
    if not np.isfinite(target_scale) or target_scale <= np.finfo(np.float64).eps:
        return _unresolved_fit(values.shape[1], ids, fit_id, tile_bounds_yx, len(values), block_count, "degenerate_covariance")
    if config.method == "diagonal":
        estimated = np.diag(np.diag(sample_covariance)); shrinkage = 1.0
    elif config.method == "fixed_ridge":
        shrinkage = float(config.ridge_ratio)
        estimated = (1.0 - shrinkage) * sample_covariance + shrinkage * target_scale * np.eye(values.shape[1])
    elif config.method in {"oas", "ledoit_wolf"}:
        try:
            from sklearn.covariance import ledoit_wolf, oas
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for OAS/Ledoit-Wolf covariance") from exc
        estimator = oas if config.method == "oas" else ledoit_wolf
        estimated, shrinkage = estimator(centered, assume_centered=True)
    else:  # pragma: no cover - dataclass validation owns this boundary.
        raise ValueError("unsupported covariance estimator")
    estimated = 0.5 * (estimated + estimated.T)
    raw_eigenvalues = np.linalg.eigvalsh(sample_covariance)[::-1]
    eigenvalues, vectors = np.linalg.eigh(estimated)
    maximum = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).eps)
    floor = max(maximum * config.eigenvalue_floor_ratio, np.finfo(np.float64).eps)
    floored = np.maximum(eigenvalues, floor)
    condition = float(np.max(floored) / np.min(floored))
    regularized_covariance = vectors @ np.diag(floored) @ vectors.T
    whitening = vectors @ np.diag(1.0 / np.sqrt(floored)) @ vectors.T
    identity_error = float(np.linalg.norm(whitening @ regularized_covariance @ whitening.T - np.eye(values.shape[1]), ord="fro"))
    resolved = bool(condition <= config.maximum_condition_number and identity_error <= 1e-8)
    reason = None if resolved else ("condition_number_exceeds_cap" if condition > config.maximum_condition_number else "zca_identity_check_failed")
    return LocalCovarianceFit(
        fit_id=fit_id, feature_ids=ids, tile_bounds_yx=tile_bounds_yx,
        mean=np.asarray(center, dtype=np.float64), sample_covariance=sample_covariance,
        covariance=regularized_covariance, whitening=whitening,
        eigenvalues_raw=raw_eigenvalues,
        eigenvalues_regularized=np.sort(floored)[::-1], shrinkage=float(shrinkage),
        condition_number=condition, effective_rank=effective_rank(np.maximum(raw_eigenvalues, 0)),
        sample_count=len(values), block_count=int(block_count), resolved=resolved,
        unresolved_reason=reason,
        diagnostics={
            "method": config.method, "center": config.center,
            "eigenvalue_floor": float(floor), "zca_identity_error": identity_error,
            "raw_condition_number": float(np.max(np.maximum(raw_eigenvalues, 0)) / max(float(np.min(np.maximum(raw_eigenvalues, 0))), np.finfo(np.float64).eps)),
            "fallback_required": not resolved,
        },
    )


def fit_tiled_feature_whiteners(
    feature_bank: WhiteningFeatureBank,
    quiet_fit_mask: np.ndarray,
    tile_config: SpatialTileConfig,
    estimator_config: CovarianceEstimatorConfig,
) -> TiledWhiteningFits:
    """Fit all local primary/diagonal transforms plus a global full fallback."""
    tiles = enumerate_spatial_tiles(feature_bank.values.shape[1], feature_bank.values.shape[2], tile_config)
    global_samples, global_diagnostics = gather_covariance_samples(
        feature_bank, quiet_fit_mask, maximum_samples=tile_config.maximum_fit_samples,
        spatial_subsample=tile_config.spatial_subsample,
        temporal_subsample=tile_config.temporal_subsample,
    )
    global_config = CovarianceEstimatorConfig(
        method="oas" if estimator_config.method == "diagonal" else estimator_config.method,
        ridge_ratio=estimator_config.ridge_ratio,
        eigenvalue_floor_ratio=estimator_config.eigenvalue_floor_ratio,
        maximum_condition_number=estimator_config.maximum_condition_number,
        center=estimator_config.center,
    )
    global_fit = fit_covariance(
        global_samples, global_config, fit_id="global_covariance_fit",
        feature_ids=feature_bank.feature_ids,
        block_count=global_diagnostics["blocked_effective_sample_count"],
    )
    primary: list[LocalCovarianceFit] = []
    diagonal: list[LocalCovarianceFit] = []
    for index, bounds in enumerate(tiles):
        samples, diagnostics = gather_covariance_samples(
            feature_bank, quiet_fit_mask, tile_bounds_yx=bounds,
            maximum_samples=tile_config.maximum_fit_samples,
            spatial_subsample=tile_config.spatial_subsample,
            temporal_subsample=tile_config.temporal_subsample,
        )
        if diagnostics["raw_sample_count"] < tile_config.minimum_raw_samples:
            primary_fit = _unresolved_fit(len(feature_bank.feature_ids), feature_bank.feature_ids, f"tile_{index:04d}_{estimator_config.method}", bounds, len(samples), diagnostics["blocked_effective_sample_count"], "minimum_raw_samples_not_met")
            diagonal_fit = _unresolved_fit(len(feature_bank.feature_ids), feature_bank.feature_ids, f"tile_{index:04d}_diagonal", bounds, len(samples), diagnostics["blocked_effective_sample_count"], "minimum_raw_samples_not_met")
        else:
            primary_fit = fit_covariance(
                samples, estimator_config, fit_id=f"tile_{index:04d}_{estimator_config.method}",
                feature_ids=feature_bank.feature_ids, tile_bounds_yx=bounds,
                block_count=diagnostics["blocked_effective_sample_count"],
            )
            diagonal_fit = fit_covariance(
                samples,
                CovarianceEstimatorConfig(
                    method="diagonal", ridge_ratio=estimator_config.ridge_ratio,
                    eigenvalue_floor_ratio=estimator_config.eigenvalue_floor_ratio,
                    maximum_condition_number=estimator_config.maximum_condition_number,
                    center=estimator_config.center,
                ),
                fit_id=f"tile_{index:04d}_diagonal", feature_ids=feature_bank.feature_ids,
                tile_bounds_yx=bounds, block_count=diagnostics["blocked_effective_sample_count"],
            )
        primary.append(primary_fit); diagonal.append(diagonal_fit)
    return TiledWhiteningFits(
        feature_ids=feature_bank.feature_ids, tile_bounds_yx=tiles,
        primary_fits=tuple(primary), local_diagonal_fits=tuple(diagonal),
        global_full_fit=global_fit, tile_config=tile_config,
    )


def fit_empirical_quiet_calibration(
    energy: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    probability_floor: float = 1e-6,
) -> EmpiricalQuietCalibration:
    """Fit deterministic add-one empirical survival on a calibration-only block."""
    values = np.asarray(energy, dtype=np.float64)
    mask = np.asarray(calibration_mask, dtype=bool)
    if mask.shape == (len(values),):
        selected = values[mask]
    else:
        try:
            selected = values[np.broadcast_to(mask, values.shape)]
        except ValueError as exc:
            raise ValueError("calibration mask cannot align with energy") from exc
    selected = selected.ravel()
    if not selected.size or not np.isfinite(selected).all() or not 0 < probability_floor <= 1:
        raise ValueError("finite calibration energy and a valid probability floor are required")
    return EmpiricalQuietCalibration(
        sorted_energy=np.sort(selected), probability_floor=float(probability_floor),
        diagnostics={"sample_count": int(len(selected)), "tie_rule": "left_inclusive_add_one", "probability_floor": float(probability_floor)},
    )


def _selected_fit(
    primary: LocalCovarianceFit,
    global_full: LocalCovarianceFit,
    diagonal: LocalCovarianceFit,
) -> tuple[LocalCovarianceFit | None, str]:
    if primary.resolved:
        return primary, "local_covariance_fit"
    if global_full.resolved:
        return global_full, "global_covariance_fit"
    if diagonal.resolved:
        return diagonal, "local_diagonal"
    return None, "recorded_identity"


def apply_tiled_feature_whiteners(
    feature_bank: WhiteningFeatureBank,
    fits: TiledWhiteningFits,
    tile_config: SpatialTileConfig,
    quiet_calibration: np.ndarray | EmpiricalQuietCalibration,
    *,
    frame_chunk: int = 8,
    probability_floor: float = 1e-6,
) -> LocalWhiteningResult:
    """Apply frozen tiled transforms, blend channels, then derive energy/surprise."""
    if tile_config != fits.tile_config or feature_bank.feature_ids != fits.feature_ids:
        raise ValueError("feature order or tile configuration differs from fitted state")
    if frame_chunk < 1:
        raise ValueError("frame_chunk must be positive")
    frames, height, width, dimension = feature_bank.values.shape
    accumulated = np.zeros((frames, height, width, dimension), dtype=np.float32)
    blend_sum = np.zeros((height, width), dtype=np.float32)
    unresolved = np.zeros((height, width), dtype=bool)
    fallback_counts: dict[str, int] = {}
    weight = tile_blend_weight(tile_config.tile_height, tile_config.tile_width, tile_config.blend)
    for bounds, primary, diagonal in zip(fits.tile_bounds_yx, fits.primary_fits, fits.local_diagonal_fits):
        y0, y1, x0, x1 = bounds
        selected, provenance = _selected_fit(primary, fits.global_full_fit, diagonal)
        fallback_counts[provenance] = fallback_counts.get(provenance, 0) + 1
        if not primary.resolved:
            unresolved[y0:y1, x0:x1] = True
        for start in range(0, frames, int(frame_chunk)):
            stop = min(frames, start + int(frame_chunk))
            block = np.asarray(feature_bank.values[start:stop, y0:y1, x0:x1], dtype=np.float64)
            if selected is None:
                transformed = block
            else:
                transformed = np.einsum("...d,ed->...e", block - selected.mean, selected.whitening, optimize=True)
            accumulated[start:stop, y0:y1, x0:x1] += (transformed * weight[None, :, :, None]).astype(np.float32)
        blend_sum[y0:y1, x0:x1] += weight
    if np.any(blend_sum <= 0):
        raise RuntimeError("tile blending left uncovered pixels")
    zca = accumulated / blend_sum[None, :, :, None]
    zca[~feature_bank.valid_frames] = 0
    energy = np.sum(np.square(zca, dtype=np.float32), axis=-1, dtype=np.float32)
    calibration = (
        quiet_calibration
        if isinstance(quiet_calibration, EmpiricalQuietCalibration)
        else fit_empirical_quiet_calibration(energy, quiet_calibration, probability_floor=probability_floor)
    )
    surprise = calibration.surprise(energy)
    surprise[~feature_bank.valid_frames] = 0
    return LocalWhiteningResult(
        zca_features=zca.astype(np.float32, copy=False),
        mahalanobis_energy=energy, quiet_surprise=surprise,
        blend_weight=blend_sum, unresolved_tile_mask=unresolved,
        valid_frames=feature_bank.valid_frames.copy(),
        diagnostics={
            "feature_ids": list(feature_bank.feature_ids),
            "energy_derivation": "sum_of_squares_after_feature_blending",
            "fallback_policy": list(fits.fallback_policy),
            "fallback_counts": fallback_counts,
            "unresolved_pixel_fraction": float(np.mean(unresolved)),
            "calibration": calibration.diagnostics,
            "float64_fit_float32_apply": True,
            "application_backend": "numpy_cpu",
        },
    )
