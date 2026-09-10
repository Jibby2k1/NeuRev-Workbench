"""Movie-level fractional whitening operators with explicit support semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import convolve

from neurobench.algorithms.local_whitening import (
    FractionalWhiteningFit,
    center_response_kernel,
    deterministic_spatial_samples,
    deterministic_temporal_samples,
    fit_fractional_whitening,
    raw_preserving_blend,
)


@dataclass(frozen=True)
class WhiteningApplication:
    output: np.ndarray
    diagnostics: dict[str, Any]
    kernels: tuple[np.ndarray, ...]


def _finite_movie(movie: np.ndarray) -> np.ndarray:
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("movie must be finite TYX")
    return values


def deterministic_joint_samples(
    quiet: np.ndarray,
    spatial_width: int,
    temporal_width: int,
    *,
    causality: str,
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample aligned TYX neighborhoods without allocating a dense patch view."""
    values = _finite_movie(quiet)
    if spatial_width < 3 or spatial_width % 2 == 0:
        raise ValueError("spatial_width must be odd and >=3")
    if temporal_width < 3 or temporal_width % 2 == 0:
        raise ValueError("temporal_width must be odd and >=3")
    if causality not in {"centered", "causal"}:
        raise ValueError("causality must be centered or causal")
    spatial_radius = spatial_width // 2
    if causality == "centered":
        before = after = temporal_width // 2
    else:
        before, after = temporal_width - 1, 0
    padded = np.pad(
        values,
        ((before, after), (spatial_radius, spatial_radius), (spatial_radius, spatial_radius)),
        mode="reflect",
    )
    total = int(np.prod(values.shape))
    rng = np.random.default_rng(int(seed))
    flat = np.sort(rng.choice(total, size=min(maximum_samples, total), replace=False))
    t, y, x = np.unravel_index(flat, values.shape)
    temporal_offsets = np.arange(-before, after + 1)
    spatial_offsets = np.arange(-spatial_radius, spatial_radius + 1)
    tt = t[:, None, None, None] + before + temporal_offsets[None, :, None, None]
    yy = y[:, None, None, None] + spatial_radius + spatial_offsets[None, None, :, None]
    xx = x[:, None, None, None] + spatial_radius + spatial_offsets[None, None, None, :]
    return padded[tt, yy, xx].reshape(len(flat), -1).astype(np.float64)


def _temporal_samples(
    quiet: np.ndarray, width: int, *, causality: str, maximum_samples: int, seed: int
) -> np.ndarray:
    if causality == "centered":
        return deterministic_temporal_samples(
            quiet, width, maximum_samples=maximum_samples, seed=seed
        )
    values = _finite_movie(quiet)
    padded = np.pad(values, ((width - 1, 0), (0, 0), (0, 0)), mode="reflect")
    total = int(np.prod(values.shape))
    rng = np.random.default_rng(int(seed))
    flat = np.sort(rng.choice(total, size=min(maximum_samples, total), replace=False))
    t, y, x = np.unravel_index(flat, values.shape)
    offsets = np.arange(-(width - 1), 1)
    return padded[t[:, None] + width - 1 + offsets[None, :], y[:, None], x[:, None]].astype(np.float64)


def _apply_temporal_kernel(
    movie: np.ndarray, kernel: np.ndarray, offset: float, *, causality: str
) -> np.ndarray:
    values = _finite_movie(movie)
    weights = np.asarray(kernel, dtype=np.float64).ravel()
    width = len(weights)
    if causality == "centered":
        before = after = width // 2
    else:
        before, after = width - 1, 0
    padded = np.pad(values, ((before, after), (0, 0), (0, 0)), mode="reflect")
    output = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(weights):
        output += float(weight) * padded[index:index + len(values)]
    return (output - float(offset)).astype(np.float32)


def _apply_spatial_kernel(movie: np.ndarray, kernel: np.ndarray, offset: float) -> np.ndarray:
    values = _finite_movie(movie)
    width = int(round(np.sqrt(np.asarray(kernel).size)))
    if width * width != np.asarray(kernel).size:
        raise ValueError("spatial kernel is not square")
    return (
        convolve(values, np.asarray(kernel).reshape(1, width, width), mode="reflect")
        - float(offset)
    ).astype(np.float32)


def _apply_joint_kernel(
    movie: np.ndarray,
    kernel: np.ndarray,
    offset: float,
    *,
    spatial_width: int,
    temporal_width: int,
    causality: str,
) -> np.ndarray:
    values = _finite_movie(movie)
    weights = np.asarray(kernel, dtype=np.float64).reshape(
        temporal_width, spatial_width, spatial_width
    )
    if causality == "centered":
        before = after = temporal_width // 2
    else:
        before, after = temporal_width - 1, 0
    padded = np.pad(values, ((before, after), (0, 0), (0, 0)), mode="reflect")
    output = np.zeros_like(values, dtype=np.float64)
    for lag, spatial_kernel in enumerate(weights):
        spatial = convolve(
            padded[lag:lag + len(values)], spatial_kernel[None], mode="reflect"
        )
        output += spatial
    return (output - float(offset)).astype(np.float32)


def _fit_summary(fit: FractionalWhiteningFit) -> dict[str, Any]:
    return {
        "condition_number": fit.condition_number,
        "effective_rank_fraction": fit.effective_rank_fraction,
        "sample_count": fit.sample_count,
        "resolved": fit.resolved,
        "stop_reason": fit.stop_reason,
        "eigenvalues_raw": fit.eigenvalues_raw.tolist(),
        "eigenvalues_regularized": fit.eigenvalues_regularized.tolist(),
        "shrinkage": fit.shrinkage,
        "exponent": fit.exponent,
        "eigen_floor_ratio": fit.eigen_floor_ratio,
    }


def _regions(shape: tuple[int, int], scope: str) -> tuple[tuple[slice, slice], ...]:
    height, width = shape
    if scope == "global_quiet":
        return ((slice(0, height), slice(0, width)),)
    if scope != "regional_quiet":
        raise ValueError(f"unsupported covariance scope: {scope}")
    middle_y, middle_x = height // 2, width // 2
    return (
        (slice(0, middle_y), slice(0, middle_x)),
        (slice(0, middle_y), slice(middle_x, width)),
        (slice(middle_y, height), slice(0, middle_x)),
        (slice(middle_y, height), slice(middle_x, width)),
    )


def _one_geometry(
    movie: np.ndarray,
    quiet_frames: int,
    *,
    geometry: str,
    covariance_scope: str,
    causality: str,
    spatial_width: int | None,
    temporal_width: int | None,
    exponent: float,
    shrinkage: float,
    eigen_floor_ratio: float,
    maximum_samples: int,
    seed: int,
) -> WhiteningApplication:
    values = _finite_movie(movie)
    if not 1 <= quiet_frames <= len(values):
        raise ValueError("quiet_frames outside movie")
    output = np.empty_like(values)
    summaries: list[dict[str, Any]] = []
    kernels: list[np.ndarray] = []
    for region_index, (ys, xs) in enumerate(_regions(values.shape[1:], covariance_scope)):
        regional = values[:, ys, xs]
        quiet = regional[:quiet_frames]
        regional_seed = int(seed) + region_index * 1009
        if geometry == "spatial":
            assert spatial_width is not None
            samples = deterministic_spatial_samples(
                quiet, spatial_width, maximum_samples=maximum_samples, seed=regional_seed
            )
        elif geometry == "temporal":
            assert temporal_width is not None
            samples = _temporal_samples(
                quiet, temporal_width, causality=causality,
                maximum_samples=maximum_samples, seed=regional_seed,
            )
        elif geometry == "joint_spatiotemporal":
            assert spatial_width is not None and temporal_width is not None
            samples = deterministic_joint_samples(
                quiet, spatial_width, temporal_width, causality=causality,
                maximum_samples=maximum_samples, seed=regional_seed,
            )
        else:
            raise ValueError(f"unsupported one-stage geometry: {geometry}")
        fit = fit_fractional_whitening(
            samples, shrinkage=shrinkage, exponent=exponent,
            eigen_floor_ratio=eigen_floor_ratio,
        )
        kernel, offset = center_response_kernel(fit)
        if geometry == "spatial":
            transformed = _apply_spatial_kernel(regional, kernel, offset)
        elif geometry == "temporal":
            transformed = _apply_temporal_kernel(
                regional, kernel, offset, causality=causality
            )
        else:
            transformed = _apply_joint_kernel(
                regional, kernel, offset, spatial_width=spatial_width,
                temporal_width=temporal_width, causality=causality,
            )
        output[:, ys, xs] = transformed
        summaries.append({"region": region_index, **_fit_summary(fit)})
        kernels.append(np.asarray(kernel, dtype=np.float64))
    return WhiteningApplication(
        output=output,
        diagnostics={
            "geometry": geometry,
            "covariance_scope": covariance_scope,
            "causality": causality,
            "region_fits": summaries,
            "resolved": all(item["resolved"] for item in summaries),
            "maximum_condition_number": max(item["condition_number"] for item in summaries),
            "minimum_effective_rank_fraction": min(
                item["effective_rank_fraction"] for item in summaries
            ),
            "finite_output_fraction": float(np.mean(np.isfinite(output))),
        },
        kernels=tuple(kernels),
    )


def apply_whitening(
    movie: np.ndarray,
    quiet_frames: int,
    specification: dict[str, Any],
    *,
    maximum_samples: int,
) -> WhiteningApplication:
    """Apply the specified movie-level whitener and Raw-preserving skip."""
    values = _finite_movie(movie)
    geometry = str(specification["whitening_geometry"])
    blend = float(specification["raw_preserving_blend"])
    if geometry == "none":
        return WhiteningApplication(
            output=values.copy(),
            diagnostics={
                "geometry": "none", "resolved": True,
                "maximum_condition_number": 1.0,
                "minimum_effective_rank_fraction": 1.0,
                "finite_output_fraction": 1.0,
                "raw_preserving_blend": blend,
            },
            kernels=(),
        )
    common = {
        "movie": values,
        "quiet_frames": quiet_frames,
        "covariance_scope": specification["covariance_scope"],
        "causality": specification["causality"],
        "spatial_width": specification.get("spatial_width_px"),
        "temporal_width": specification.get("temporal_width_frames"),
        "maximum_samples": maximum_samples,
        "seed": int(specification["seed"]),
    }
    if geometry in {"spatial", "temporal", "joint_spatiotemporal"}:
        prefix = "joint" if geometry == "joint_spatiotemporal" else geometry
        result = _one_geometry(
            geometry=geometry,
            exponent=float(specification[f"{prefix}_exponent"]),
            shrinkage=float(specification[f"{prefix}_shrinkage"]),
            eigen_floor_ratio=float(specification[f"{prefix}_eigen_floor_ratio"]),
            **common,
        )
    else:
        order = (
            ("spatial", "temporal")
            if geometry == "spatial_then_temporal"
            else ("temporal", "spatial")
        )
        first = _one_geometry(
            geometry=order[0],
            exponent=float(specification[f"{order[0]}_exponent"]),
            shrinkage=float(specification[f"{order[0]}_shrinkage"]),
            eigen_floor_ratio=float(specification[f"{order[0]}_eigen_floor_ratio"]),
            **common,
        )
        second_common = {**common, "movie": first.output}
        second = _one_geometry(
            geometry=order[1],
            exponent=float(specification[f"{order[1]}_exponent"]),
            shrinkage=float(specification[f"{order[1]}_shrinkage"]),
            eigen_floor_ratio=float(specification[f"{order[1]}_eigen_floor_ratio"]),
            **second_common,
        )
        result = WhiteningApplication(
            output=second.output,
            diagnostics={
                "geometry": geometry,
                "covariance_scope": specification["covariance_scope"],
                "causality": specification["causality"],
                "stages": [first.diagnostics, second.diagnostics],
                "resolved": first.diagnostics["resolved"] and second.diagnostics["resolved"],
                "maximum_condition_number": max(
                    first.diagnostics["maximum_condition_number"],
                    second.diagnostics["maximum_condition_number"],
                ),
                "minimum_effective_rank_fraction": min(
                    first.diagnostics["minimum_effective_rank_fraction"],
                    second.diagnostics["minimum_effective_rank_fraction"],
                ),
                "finite_output_fraction": float(np.mean(np.isfinite(second.output))),
            },
            kernels=first.kernels + second.kernels,
        )
    mixed = raw_preserving_blend(values, result.output, blend)
    return WhiteningApplication(
        output=mixed,
        diagnostics={**result.diagnostics, "raw_preserving_blend": blend},
        kernels=result.kernels,
    )
