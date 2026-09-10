"""Exact sparse application of single-stage movie-level whitening operators."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from neurobench.algorithms.local_whitening import (
    center_response_kernel, deterministic_spatial_samples,
    fit_fractional_whitening,
)

from .operators import (
    _regions, _temporal_samples, deterministic_joint_samples,
)


@dataclass(frozen=True)
class SparseRegionKernel:
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    kernel: np.ndarray
    offset: float
    condition_number: float
    effective_rank_fraction: float
    resolved: bool


@dataclass(frozen=True)
class SparseWhiteningOperator:
    geometry: str
    causality: str
    spatial_width: int
    temporal_width: int
    raw_preserving_blend: float
    regions: tuple[SparseRegionKernel, ...]


@dataclass(frozen=True)
class SparseSequentialOperator:
    geometry: str
    first: SparseWhiteningOperator
    second: SparseWhiteningOperator
    raw_preserving_blend: float


def fit_sparse_single_stage(
    movie: np.ndarray, quiet_frames: int, specification: dict[str, Any],
    *, maximum_samples: int,
) -> SparseWhiteningOperator:
    values = np.asarray(movie, dtype=np.float32)
    geometry = str(specification["whitening_geometry"])
    if geometry not in {"spatial", "temporal", "joint_spatiotemporal"}:
        raise ValueError("sparse single-stage fitter requires spatial, temporal, or joint whitening")
    sw = int(specification.get("spatial_width_px") or 1)
    tw = int(specification.get("temporal_width_frames") or 1)
    prefix = "joint" if geometry == "joint_spatiotemporal" else geometry
    result = []
    for region_index, (ys, xs) in enumerate(_regions(values.shape[1:], specification["covariance_scope"])):
        quiet = values[:quiet_frames, ys, xs]
        seed = int(specification["seed"]) + region_index * 1009
        if geometry == "spatial":
            samples = deterministic_spatial_samples(
                quiet, sw, maximum_samples=maximum_samples, seed=seed
            )
        elif geometry == "temporal":
            samples = _temporal_samples(
                quiet, tw, causality=specification["causality"],
                maximum_samples=maximum_samples, seed=seed,
            )
        else:
            samples = deterministic_joint_samples(
                quiet, sw, tw, causality=specification["causality"],
                maximum_samples=maximum_samples, seed=seed,
            )
        fit = fit_fractional_whitening(
            samples, exponent=float(specification[f"{prefix}_exponent"]),
            shrinkage=float(specification[f"{prefix}_shrinkage"]),
            eigen_floor_ratio=float(specification[f"{prefix}_eigen_floor_ratio"]),
        )
        kernel, offset = center_response_kernel(fit)
        if geometry == "spatial": kernel = kernel.reshape(1, sw, sw)
        elif geometry == "temporal": kernel = kernel.reshape(tw, 1, 1)
        else: kernel = kernel.reshape(tw, sw, sw)
        result.append(SparseRegionKernel(
            y_start=int(ys.start or 0), y_stop=int(ys.stop or values.shape[1]),
            x_start=int(xs.start or 0), x_stop=int(xs.stop or values.shape[2]),
            kernel=np.asarray(kernel, dtype=np.float64), offset=float(offset),
            condition_number=fit.condition_number,
            effective_rank_fraction=fit.effective_rank_fraction,
            resolved=fit.resolved,
        ))
    return SparseWhiteningOperator(
        geometry=geometry, causality=str(specification["causality"]),
        spatial_width=sw, temporal_width=tw,
        raw_preserving_blend=float(specification["raw_preserving_blend"]),
        regions=tuple(result),
    )


def apply_sparse(
    movie: np.ndarray, times: np.ndarray, rows: np.ndarray, columns: np.ndarray,
    operator: SparseWhiteningOperator, *, chunk_size: int = 65536,
) -> np.ndarray:
    values = np.asarray(movie, dtype=np.float32)
    t = np.asarray(times, dtype=np.int64); y = np.asarray(rows, dtype=np.int64); x = np.asarray(columns, dtype=np.int64)
    if not (t.shape == y.shape == x.shape) or t.ndim != 1:
        raise ValueError("sparse coordinates must be aligned vectors")
    if np.any(t < 0) or np.any(t >= len(values)) or np.any(y < 0) or np.any(y >= values.shape[1]) or np.any(x < 0) or np.any(x >= values.shape[2]):
        raise ValueError("sparse coordinate outside movie")
    output = np.empty(len(t), dtype=np.float64)
    assigned = np.zeros(len(t), dtype=bool)
    for region in operator.regions:
        selected = np.flatnonzero(
            (y >= region.y_start) & (y < region.y_stop)
            & (x >= region.x_start) & (x < region.x_stop)
        )
        if not len(selected): continue
        regional = values[:, region.y_start:region.y_stop, region.x_start:region.x_stop]
        kernel = region.kernel
        # scipy.ndimage.convolve reverses spatial weights; the dense temporal
        # helper applies temporal weights directly, so reverse Y/X only.
        if kernel.shape[1] > 1 or kernel.shape[2] > 1:
            kernel = kernel[:, ::-1, ::-1]
        kt, ky, kx = kernel.shape; sr_y, sr_x = ky // 2, kx // 2
        if kt > 1 and operator.causality == "causal": before, after = kt - 1, 0
        else: before = after = kt // 2
        # Match the dense reference exactly: temporal helpers use NumPy's
        # non-edge-repeating ``reflect`` padding, whereas scipy.ndimage's
        # spatial ``mode='reflect'`` repeats the edge sample (NumPy
        # ``symmetric`` semantics).
        padded = np.pad(regional, ((before, after), (0, 0), (0, 0)), mode="reflect")
        padded = np.pad(padded, ((0, 0), (sr_y, sr_y), (sr_x, sr_x)), mode="symmetric")
        offsets_t = np.arange(-before, after + 1)
        offsets_y = np.arange(-sr_y, sr_y + 1)
        offsets_x = np.arange(-sr_x, sr_x + 1)
        for start in range(0, len(selected), chunk_size):
            take = selected[start:start + chunk_size]
            local_y = y[take] - region.y_start; local_x = x[take] - region.x_start
            tt = t[take, None, None, None] + before + offsets_t[None, :, None, None]
            yy = local_y[:, None, None, None] + sr_y + offsets_y[None, None, :, None]
            xx = local_x[:, None, None, None] + sr_x + offsets_x[None, None, None, :]
            neighborhoods = padded[tt, yy, xx]
            transformed = np.einsum("ntyx,tyx->n", neighborhoods, kernel, optimize=True) - region.offset
            beta = operator.raw_preserving_blend
            output[take] = (1.0 - beta) * values[t[take], y[take], x[take]] + beta * transformed
        assigned[selected] = True
    if not assigned.all(): raise RuntimeError("not all sparse coordinates were assigned to a covariance region")
    return output


def _reflect_indices(index: np.ndarray, length: int, *, repeat_edge: bool) -> np.ndarray:
    """Map the small supported offsets used here to padded-array coordinates."""
    if repeat_edge:
        return np.where(index < 0, -index - 1,
                        np.where(index >= length, 2 * length - index - 1, index))
    return np.where(index < 0, -index,
                    np.where(index >= length, 2 * length - index - 2, index))


def _second_stage_samples(
    movie: np.ndarray, quiet_frames: int, specification: dict[str, Any],
    geometry: str, first: SparseWhiteningOperator, *, maximum_samples: int,
) -> tuple[SparseRegionKernel, ...]:
    """Fit stage two from exact sparse evaluations of stage one."""
    values = np.asarray(movie, dtype=np.float32)
    sw = int(specification.get("spatial_width_px") or 1)
    tw = int(specification.get("temporal_width_frames") or 1)
    prefix = geometry
    result = []
    for region_index, (ys, xs) in enumerate(_regions(values.shape[1:], specification["covariance_scope"])):
        height, width = int(ys.stop - ys.start), int(xs.stop - xs.start)
        total = quiet_frames * height * width
        seed = int(specification["seed"]) + region_index * 1009
        rng = np.random.default_rng(seed)
        flat = np.sort(rng.choice(total, size=min(maximum_samples, total), replace=False))
        t, local_y, local_x = np.unravel_index(flat, (quiet_frames, height, width))
        if geometry == "spatial":
            before = after = 0; radius = sw // 2
        else:
            radius = 0
            if specification["causality"] == "centered": before = after = tw // 2
            else: before, after = tw - 1, 0
        toff = np.arange(-before, after + 1)
        soff = np.arange(-radius, radius + 1)
        tt = _reflect_indices(t[:, None, None, None] + toff[None, :, None, None], quiet_frames, repeat_edge=False)
        yy = _reflect_indices(local_y[:, None, None, None] + soff[None, None, :, None], height, repeat_edge=False) + int(ys.start)
        xx = _reflect_indices(local_x[:, None, None, None] + soff[None, None, None, :], width, repeat_edge=False) + int(xs.start)
        shape = (len(flat), len(toff), len(soff), len(soff))
        tt = np.broadcast_to(tt, shape).reshape(-1)
        yy = np.broadcast_to(yy, shape).reshape(-1)
        xx = np.broadcast_to(xx, shape).reshape(-1)
        samples = apply_sparse(values, tt, yy, xx, first).reshape(len(flat), -1)
        fit = fit_fractional_whitening(
            samples, exponent=float(specification[f"{prefix}_exponent"]),
            shrinkage=float(specification[f"{prefix}_shrinkage"]),
            eigen_floor_ratio=float(specification[f"{prefix}_eigen_floor_ratio"]),
        )
        kernel, offset = center_response_kernel(fit)
        kernel = kernel.reshape((1, sw, sw) if geometry == "spatial" else (tw, 1, 1))
        result.append(SparseRegionKernel(
            y_start=int(ys.start), y_stop=int(ys.stop), x_start=int(xs.start), x_stop=int(xs.stop),
            kernel=np.asarray(kernel, dtype=np.float64), offset=float(offset),
            condition_number=fit.condition_number,
            effective_rank_fraction=fit.effective_rank_fraction, resolved=fit.resolved,
        ))
    return tuple(result)


def fit_sparse_sequential(
    movie: np.ndarray, quiet_frames: int, specification: dict[str, Any],
    *, maximum_samples: int,
) -> SparseSequentialOperator:
    geometry = str(specification["whitening_geometry"])
    orders = {
        "spatial_then_temporal": ("spatial", "temporal"),
        "temporal_then_spatial": ("temporal", "spatial"),
    }
    if geometry not in orders:
        raise ValueError("sequential sparse fitter requires a separable geometry")
    first_geometry, second_geometry = orders[geometry]
    first_spec = {**specification, "whitening_geometry": first_geometry,
                  "raw_preserving_blend": 1.0}
    first = fit_sparse_single_stage(
        movie, quiet_frames, first_spec, maximum_samples=maximum_samples
    )
    second_regions = _second_stage_samples(
        movie, quiet_frames, specification, second_geometry, first,
        maximum_samples=maximum_samples,
    )
    second = SparseWhiteningOperator(
        geometry=second_geometry, causality=str(specification["causality"]),
        spatial_width=int(specification.get("spatial_width_px") or 1),
        temporal_width=int(specification.get("temporal_width_frames") or 1),
        raw_preserving_blend=1.0, regions=second_regions,
    )
    return SparseSequentialOperator(
        geometry=geometry, first=replace(first, raw_preserving_blend=1.0),
        second=second, raw_preserving_blend=float(specification["raw_preserving_blend"]),
    )


def apply_sparse_sequential(
    movie: np.ndarray, times: np.ndarray, rows: np.ndarray, columns: np.ndarray,
    operator: SparseSequentialOperator, *, chunk_size: int = 32768,
) -> np.ndarray:
    values = np.asarray(movie, dtype=np.float32)
    t = np.asarray(times, dtype=np.int64); y = np.asarray(rows, dtype=np.int64); x = np.asarray(columns, dtype=np.int64)
    output = np.empty(len(t), dtype=np.float64); assigned = np.zeros(len(t), dtype=bool)
    for region in operator.second.regions:
        selected = np.flatnonzero((y >= region.y_start) & (y < region.y_stop)
                                  & (x >= region.x_start) & (x < region.x_stop))
        if not len(selected): continue
        kernel = region.kernel
        if kernel.shape[1] > 1 or kernel.shape[2] > 1: kernel = kernel[:, ::-1, ::-1]
        kt, ky, kx = kernel.shape; ry, rx = ky // 2, kx // 2
        if kt > 1 and operator.second.causality == "causal": before, after = kt - 1, 0
        else: before = after = kt // 2
        toff = np.arange(-before, after + 1); yoff = np.arange(-ry, ry + 1); xoff = np.arange(-rx, rx + 1)
        height, width = region.y_stop - region.y_start, region.x_stop - region.x_start
        for start in range(0, len(selected), chunk_size):
            take = selected[start:start + chunk_size]; n = len(take)
            tt = _reflect_indices(t[take, None, None, None] + toff[None, :, None, None], len(values), repeat_edge=False)
            yy = _reflect_indices((y[take] - region.y_start)[:, None, None, None] + yoff[None, None, :, None], height, repeat_edge=True) + region.y_start
            xx = _reflect_indices((x[take] - region.x_start)[:, None, None, None] + xoff[None, None, None, :], width, repeat_edge=True) + region.x_start
            shape = (n, len(toff), len(yoff), len(xoff))
            stage_one = apply_sparse(
                values, np.broadcast_to(tt, shape).reshape(-1),
                np.broadcast_to(yy, shape).reshape(-1),
                np.broadcast_to(xx, shape).reshape(-1), operator.first,
            ).reshape(shape)
            transformed = np.einsum("ntyx,tyx->n", stage_one, kernel, optimize=True) - region.offset
            beta = operator.raw_preserving_blend
            output[take] = (1.0 - beta) * values[t[take], y[take], x[take]] + beta * transformed
        assigned[selected] = True
    if not assigned.all(): raise RuntimeError("not all sequential coordinates were assigned")
    return output


def whitened_patch_observations(
    movie: np.ndarray, times: np.ndarray, rows: np.ndarray, columns: np.ndarray,
    specification: dict[str, Any], operator: SparseWhiteningOperator | SparseSequentialOperator,
    *, center_chunk_size: int = 4096,
) -> np.ndarray:
    family = specification["family"]
    sw = int(specification.get("spatial_width_px") or 1)
    tw = int(specification.get("temporal_width_frames") or 1)
    sr = sw // 2 if family in {"spatial", "joint_spatiotemporal"} else 0
    if family in {"temporal", "joint_spatiotemporal"}:
        before, after = ((tw // 2, tw // 2) if specification["causality"] == "centered" else (tw - 1, 0))
    else: before = after = 0
    toff = np.arange(-before, after + 1)
    yoff = np.arange(-sr, sr + 1)
    xoff = np.arange(-sr, sr + 1)
    feature_count = len(toff) * len(yoff) * len(xoff)
    result = np.empty((feature_count, len(times)), dtype=np.float64)
    # Feature coordinates obey the same reflected movie boundary as patch extraction.
    for start in range(0, len(times), center_chunk_size):
        stop = min(start + center_chunk_size, len(times)); n = stop - start
        base_t = np.asarray(times[start:stop])[:, None, None, None] + toff[None, :, None, None]
        base_y = np.asarray(rows[start:stop])[:, None, None, None] + yoff[None, None, :, None]
        base_x = np.asarray(columns[start:stop])[:, None, None, None] + xoff[None, None, None, :]
        # np.pad(..., mode=reflect) index mapping for coordinates at most two samples outside.
        def reflect(index: np.ndarray, length: int) -> np.ndarray:
            return np.where(index < 0, -index, np.where(index >= length, 2 * length - index - 2, index))
        tt = np.broadcast_to(reflect(base_t, movie.shape[0]), (n, len(toff), len(yoff), len(xoff))).reshape(-1)
        yy = np.broadcast_to(reflect(base_y, movie.shape[1]), (n, len(toff), len(yoff), len(xoff))).reshape(-1)
        xx = np.broadcast_to(reflect(base_x, movie.shape[2]), (n, len(toff), len(yoff), len(xoff))).reshape(-1)
        # Adjacent event frames and overlapping spatial patches contain many
        # identical TYX coordinates.  Apply the linear operator once per unique
        # coordinate, then restore the exact feature/center ordering.
        flat = np.ravel_multi_index((tt, yy, xx), movie.shape)
        unique, inverse = np.unique(flat, return_inverse=True)
        unique_t, unique_y, unique_x = np.unravel_index(unique, movie.shape)
        unique_values = (
            apply_sparse_sequential(movie, unique_t, unique_y, unique_x, operator)
            if isinstance(operator, SparseSequentialOperator)
            else apply_sparse(movie, unique_t, unique_y, unique_x, operator)
        )
        transformed = unique_values[inverse]
        result[:, start:stop] = transformed.reshape(n, feature_count).T
    return result


def sparse_diagnostics(operator: SparseWhiteningOperator | SparseSequentialOperator) -> dict[str, Any]:
    if isinstance(operator, SparseSequentialOperator):
        stages = [sparse_diagnostics(operator.first), sparse_diagnostics(operator.second)]
        return {
            "geometry": operator.geometry,
            "resolved": all(stage["resolved"] for stage in stages),
            "maximum_condition_number": max(stage["maximum_condition_number"] for stage in stages),
            "minimum_effective_rank_fraction": min(stage["minimum_effective_rank_fraction"] for stage in stages),
            "region_count": sum(stage["region_count"] for stage in stages), "stages": stages,
        }
    return {
        "geometry": operator.geometry,
        "resolved": all(region.resolved for region in operator.regions),
        "maximum_condition_number": max(region.condition_number for region in operator.regions),
        "minimum_effective_rank_fraction": min(region.effective_rank_fraction for region in operator.regions),
        "region_count": len(operator.regions),
    }
