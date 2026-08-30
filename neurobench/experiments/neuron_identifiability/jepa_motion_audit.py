"""Deterministic acquisition-confound screen for the compact video JEPA pilot.

The screen is deliberately label-free and read-only.  It samples the whole
field at deterministic frame locations, quantifies raw-intensity drift,
sensor-rail occupancy, adjacent-frame change, and translation-like motion, and
returns a portable JSON-ready record.  It does *not* motion-correct a movie or
establish that any measured change is biological, artifactual, or causal.

All temporal quantities remain in frames and all spatial quantities remain in
native pixels unless the frozen recording descriptor resolves the corresponding
physical metadata.  Missing cadence or pixel scale is never inferred.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from .contracts import stable_hash
from .discovery import sha256_file
from .jepa_data import (
    RecordingDescriptor,
    RecordingInventory,
    assert_spon_excluded,
    deterministic_uniform_frame_indices,
    open_recording_memmap,
    validate_060126_inventory,
)


MOTION_AUDIT_SCHEMA_VERSION = "neurobench.jepa_motion_audit.v1"
MOTION_AUDIT_ALGORITHM = "robust_windowed_phase_correlation_v1"


@dataclass(frozen=True)
class MotionAuditThresholds:
    """Predeclared review triggers, not validated biological cut points."""

    absolute_relative_intensity_drift: float = 0.08
    high_sensor_rail_fraction: float = 0.001
    normalized_frame_difference_p95: float = 0.75
    global_translation_p95_px: float = 1.5
    tile_median_disagreement_p95_px: float = 1.5
    minimum_valid_global_pair_fraction: float = 0.80
    minimum_median_valid_tile_fraction: float = 0.50

    def __post_init__(self) -> None:
        positive = (
            self.absolute_relative_intensity_drift,
            self.high_sensor_rail_fraction,
            self.normalized_frame_difference_p95,
            self.global_translation_p95_px,
            self.tile_median_disagreement_p95_px,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("motion-audit magnitude thresholds must be finite and positive")
        fractions = (
            self.minimum_valid_global_pair_fraction,
            self.minimum_median_valid_tile_fraction,
        )
        if any(not np.isfinite(value) or not 0 < value <= 1 for value in fractions):
            raise ValueError("motion-audit reliability thresholds must be in (0, 1]")

    def to_manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "policy": "heuristic_screening_thresholds_require_review_not_scientific_pass_fail",
        }


@dataclass(frozen=True)
class MotionAuditConfig:
    """Bounded deterministic sampling and registration configuration."""

    frames_per_recording: int = 24
    intensity_spatial_stride: int = 4
    registration_spatial_stride: int = 2
    tile_grid_yx: tuple[int, int] = (3, 4)
    minimum_registration_extent: int = 16
    minimum_robust_scale: float = 1e-6
    registration_clip_z: float = 8.0
    minimum_peak_to_median_ratio: float = 3.0
    maximum_translation_search_px: float = 8.0
    require_frozen_060126_inventory: bool = True
    verify_hashes: bool = False
    thresholds: MotionAuditThresholds = field(default_factory=MotionAuditThresholds)

    def __post_init__(self) -> None:
        if self.frames_per_recording < 2:
            raise ValueError("motion audit requires at least two sampled frames per recording")
        if min(self.intensity_spatial_stride, self.registration_spatial_stride) < 1:
            raise ValueError("spatial strides must be positive")
        if len(self.tile_grid_yx) != 2 or min(self.tile_grid_yx) < 1:
            raise ValueError("tile_grid_yx must contain two positive counts")
        if self.minimum_registration_extent < 4:
            raise ValueError("minimum_registration_extent must be at least four sampled pixels")
        if not np.isfinite(self.minimum_robust_scale) or self.minimum_robust_scale <= 0:
            raise ValueError("minimum_robust_scale must be finite and positive")
        if not np.isfinite(self.registration_clip_z) or self.registration_clip_z <= 0:
            raise ValueError("registration_clip_z must be finite and positive")
        if not np.isfinite(self.minimum_peak_to_median_ratio) or self.minimum_peak_to_median_ratio <= 1:
            raise ValueError("minimum_peak_to_median_ratio must exceed one")
        if not np.isfinite(self.maximum_translation_search_px) or self.maximum_translation_search_px <= 0:
            raise ValueError("maximum_translation_search_px must be finite and positive")

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tile_grid_yx"] = list(self.tile_grid_yx)
        payload["thresholds"] = self.thresholds.to_manifest()
        return payload


def _quantile(values: Sequence[float] | np.ndarray, probability: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.quantile(array, probability))


def _distribution_summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _normal_consistent_mad(values: np.ndarray, center: float | None = None) -> float:
    array = np.asarray(values, dtype=np.float64)
    if center is None:
        center = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - center)))


def _quadratic_peak_offset(previous: float, center: float, following: float) -> float:
    denominator = previous - 2.0 * center + following
    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(np.float64).eps:
        return 0.0
    offset = 0.5 * (previous - following) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def _registration_image(
    image: np.ndarray,
    *,
    spatial_stride: int,
    minimum_extent: int,
    minimum_scale: float,
    clip_z: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    sampled = np.asarray(image)[::spatial_stride, ::spatial_stride].astype(np.float64, copy=False)
    if sampled.ndim != 2:
        raise ValueError("phase translation requires a two-dimensional frame")
    if min(sampled.shape) < minimum_extent:
        return None, {
            "valid": False,
            "reason": "sampled_extent_below_minimum",
            "sampled_shape_yx": list(sampled.shape),
        }
    center = float(np.median(sampled))
    scale = _normal_consistent_mad(sampled, center)
    if not np.isfinite(scale) or scale < minimum_scale:
        return None, {
            "valid": False,
            "reason": "insufficient_robust_texture",
            "sampled_shape_yx": list(sampled.shape),
            "robust_scale_intensity_units": scale,
        }
    normalized = np.clip((sampled - center) / scale, -clip_z, clip_z)
    window = np.outer(np.hanning(sampled.shape[0]), np.hanning(sampled.shape[1]))
    normalized = normalized * window
    normalized -= float(np.mean(normalized))
    energy = float(np.sqrt(np.sum(np.square(normalized))))
    if not np.isfinite(energy) or energy <= minimum_scale:
        return None, {
            "valid": False,
            "reason": "insufficient_windowed_energy",
            "sampled_shape_yx": list(sampled.shape),
            "robust_scale_intensity_units": scale,
        }
    return normalized, {
        "valid": True,
        "sampled_shape_yx": list(sampled.shape),
        "robust_scale_intensity_units": scale,
        "windowed_energy": energy,
    }


def estimate_phase_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    spatial_stride: int = 1,
    minimum_extent: int = 16,
    minimum_scale: float = 1e-6,
    clip_z: float = 8.0,
    minimum_peak_to_median_ratio: float = 3.0,
    maximum_translation_search_px: float = 8.0,
) -> dict[str, Any]:
    """Estimate the native-pixel shift that registers ``moving`` to ``reference``.

    The returned direction is a *registration correction*.  If ``moving`` is a
    positive-y roll of ``reference``, the reported y shift is negative.  The
    magnitude is used only as a translation-like nuisance diagnostic.
    """
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be positive")
    if not np.isfinite(maximum_translation_search_px) or maximum_translation_search_px <= 0:
        raise ValueError("maximum_translation_search_px must be finite and positive")
    if np.shape(reference) != np.shape(moving):
        raise ValueError("reference and moving frames must share one shape")

    reference_image, reference_quality = _registration_image(
        reference,
        spatial_stride=spatial_stride,
        minimum_extent=minimum_extent,
        minimum_scale=minimum_scale,
        clip_z=clip_z,
    )
    moving_image, moving_quality = _registration_image(
        moving,
        spatial_stride=spatial_stride,
        minimum_extent=minimum_extent,
        minimum_scale=minimum_scale,
        clip_z=clip_z,
    )
    if reference_image is None or moving_image is None:
        return {
            "valid": False,
            "reason": "reference_" + reference_quality.get("reason", "invalid")
            if reference_image is None
            else "moving_" + moving_quality.get("reason", "invalid"),
            "registration_shift_yx_px": None,
            "magnitude_px": None,
            "peak_to_median_ratio": None,
            "spatial_stride": spatial_stride,
            "reference_quality": reference_quality,
            "moving_quality": moving_quality,
        }

    reference_fft = np.fft.fft2(reference_image)
    moving_fft = np.fft.fft2(moving_image)
    cross_power = reference_fft * np.conjugate(moving_fft)
    magnitude = np.abs(cross_power)
    nonzero = magnitude > np.finfo(np.float64).eps
    if not np.any(nonzero):
        return {
            "valid": False,
            "reason": "empty_cross_power_spectrum",
            "registration_shift_yx_px": None,
            "magnitude_px": None,
            "peak_to_median_ratio": None,
            "spatial_stride": spatial_stride,
            "reference_quality": reference_quality,
            "moving_quality": moving_quality,
        }
    normalized_cross_power = np.zeros_like(cross_power)
    normalized_cross_power[nonzero] = cross_power[nonzero] / magnitude[nonzero]
    correlation = np.abs(np.fft.ifft2(normalized_cross_power))
    unconstrained_y, unconstrained_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    signed_y_axis = np.arange(correlation.shape[0], dtype=np.float64)
    signed_x_axis = np.arange(correlation.shape[1], dtype=np.float64)
    signed_y_axis[signed_y_axis > correlation.shape[0] // 2] -= correlation.shape[0]
    signed_x_axis[signed_x_axis > correlation.shape[1] // 2] -= correlation.shape[1]
    maximum_sampled_shift = maximum_translation_search_px / spatial_stride
    search_mask = (
        np.abs(signed_y_axis[:, None]) <= maximum_sampled_shift
    ) & (
        np.abs(signed_x_axis[None, :]) <= maximum_sampled_shift
    )
    if not np.any(search_mask):
        raise ValueError("maximum_translation_search_px selects no correlation samples")
    constrained = np.where(search_mask, correlation, -np.inf)
    peak_y, peak_x = np.unravel_index(int(np.argmax(constrained)), correlation.shape)
    peak_value = float(correlation[peak_y, peak_x])
    unconstrained_peak_value = float(correlation[unconstrained_y, unconstrained_x])
    median_value = float(np.median(correlation))
    peak_ratio = peak_value / max(median_value, np.finfo(np.float64).eps)
    local_to_global_peak_ratio = peak_value / max(unconstrained_peak_value, np.finfo(np.float64).eps)
    unconstrained_signed_y = float(
        unconstrained_y
        if unconstrained_y <= correlation.shape[0] // 2
        else unconstrained_y - correlation.shape[0]
    )
    unconstrained_signed_x = float(
        unconstrained_x
        if unconstrained_x <= correlation.shape[1] // 2
        else unconstrained_x - correlation.shape[1]
    )
    unconstrained_peak_within_search = bool(
        abs(unconstrained_signed_y) <= maximum_sampled_shift
        and abs(unconstrained_signed_x) <= maximum_sampled_shift
    )

    offset_y = _quadratic_peak_offset(
        float(correlation[(peak_y - 1) % correlation.shape[0], peak_x]),
        peak_value,
        float(correlation[(peak_y + 1) % correlation.shape[0], peak_x]),
    )
    offset_x = _quadratic_peak_offset(
        float(correlation[peak_y, (peak_x - 1) % correlation.shape[1]]),
        peak_value,
        float(correlation[peak_y, (peak_x + 1) % correlation.shape[1]]),
    )
    signed_y = float(peak_y if peak_y <= correlation.shape[0] // 2 else peak_y - correlation.shape[0])
    signed_x = float(peak_x if peak_x <= correlation.shape[1] // 2 else peak_x - correlation.shape[1])
    shift_y = (signed_y + offset_y) * spatial_stride
    shift_x = (signed_x + offset_x) * spatial_stride
    shift_magnitude = float(math.hypot(shift_y, shift_x))
    valid = bool(
        np.isfinite(peak_ratio)
        and peak_ratio >= minimum_peak_to_median_ratio
        and unconstrained_peak_within_search
    )
    return {
        "valid": valid,
        "reason": (
            None
            if valid
            else (
                "dominant_peak_outside_plausible_adjacent_frame_search"
                if not unconstrained_peak_within_search
                else "ambiguous_phase_correlation_peak"
            )
        ),
        "registration_shift_yx_px": [float(shift_y), float(shift_x)],
        "magnitude_px": shift_magnitude,
        "peak_to_median_ratio": float(peak_ratio),
        "local_to_global_peak_ratio": float(local_to_global_peak_ratio),
        "unconstrained_peak_shift_yx_px": [
            unconstrained_signed_y * spatial_stride,
            unconstrained_signed_x * spatial_stride,
        ],
        "unconstrained_peak_within_search": unconstrained_peak_within_search,
        "maximum_translation_search_px": maximum_translation_search_px,
        "spatial_stride": spatial_stride,
        "sampled_shape_yx": list(correlation.shape),
        "reference_quality": reference_quality,
        "moving_quality": moving_quality,
    }


def _tile_slices(shape_yx: tuple[int, int], grid_yx: tuple[int, int]) -> tuple[tuple[slice, slice], ...]:
    y_edges = np.linspace(0, shape_yx[0], grid_yx[0] + 1, dtype=np.int64)
    x_edges = np.linspace(0, shape_yx[1], grid_yx[1] + 1, dtype=np.int64)
    return tuple(
        (slice(int(y_edges[row]), int(y_edges[row + 1])), slice(int(x_edges[column]), int(x_edges[column + 1])))
        for row in range(grid_yx[0])
        for column in range(grid_yx[1])
    )


def _pair_translation_diagnostics(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    config: MotionAuditConfig,
) -> dict[str, Any]:
    arguments = {
        "spatial_stride": config.registration_spatial_stride,
        "minimum_extent": config.minimum_registration_extent,
        "minimum_scale": config.minimum_robust_scale,
        "clip_z": config.registration_clip_z,
        "minimum_peak_to_median_ratio": config.minimum_peak_to_median_ratio,
        "maximum_translation_search_px": config.maximum_translation_search_px,
    }
    global_result = estimate_phase_translation(reference, moving, **arguments)
    tiles: list[dict[str, Any]] = []
    for tile_index, (y_slice, x_slice) in enumerate(_tile_slices(tuple(reference.shape), config.tile_grid_yx)):
        result = estimate_phase_translation(reference[y_slice, x_slice], moving[y_slice, x_slice], **arguments)
        result["tile_index"] = tile_index
        result["bounds_yx_zero_half_open"] = [
            int(y_slice.start),
            int(y_slice.stop),
            int(x_slice.start),
            int(x_slice.stop),
        ]
        tiles.append(result)

    valid_shifts = np.asarray(
        [item["registration_shift_yx_px"] for item in tiles if item["valid"]],
        dtype=np.float64,
    )
    if valid_shifts.size:
        median_shift = np.median(valid_shifts, axis=0)
        deviations = np.linalg.norm(valid_shifts - median_shift[None, :], axis=1)
        tile_summary = {
            "count": len(tiles),
            "valid_count": int(valid_shifts.shape[0]),
            "valid_fraction": float(valid_shifts.shape[0] / len(tiles)),
            "median_registration_shift_yx_px": [float(value) for value in median_shift],
            "median_shift_magnitude_px": float(np.linalg.norm(median_shift)),
            "disagreement_px": _distribution_summary(deviations),
        }
    else:
        tile_summary = {
            "count": len(tiles),
            "valid_count": 0,
            "valid_fraction": 0.0,
            "median_registration_shift_yx_px": None,
            "median_shift_magnitude_px": None,
            "disagreement_px": _distribution_summary([]),
        }
    return {"global": global_result, "tile_summary": tile_summary, "tiles": tiles}


def _theil_sen_drift(frame_indices: Sequence[int], frame_medians: Sequence[float], minimum_scale: float) -> dict[str, Any]:
    x = np.asarray(frame_indices, dtype=np.float64)
    y = np.asarray(frame_medians, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        raise ValueError("Theil-Sen drift requires two aligned samples")
    slopes = []
    for left in range(len(x) - 1):
        delta_x = x[left + 1 :] - x[left]
        slopes.extend(((y[left + 1 :] - y[left]) / delta_x).tolist())
    slope = float(np.median(np.asarray(slopes, dtype=np.float64)))
    intercept = float(np.median(y - slope * x))
    fitted_start = intercept + slope * float(x[0])
    fitted_end = intercept + slope * float(x[-1])
    baseline = max(abs(float(np.median(y))), minimum_scale)
    relative_change = float((fitted_end - fitted_start) / baseline)
    return {
        "estimator": "theil_sen_on_sampled_full_field_frame_medians",
        "slope_intensity_units_per_frame": slope,
        "fitted_relative_change_over_sampled_span": relative_change,
        "sampled_span_frames": int(x[-1] - x[0]),
        "sign_convention": "negative_relative_change_is_dimming",
    }


def _rail_counts(values: np.ndarray, dtype: np.dtype[Any]) -> tuple[int, int, bool]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return int(np.count_nonzero(values == info.min)), int(np.count_nonzero(values == info.max)), True
    return 0, 0, False


def _evaluate_recording_gate(recording: dict[str, Any], thresholds: MotionAuditThresholds) -> dict[str, Any]:
    intensity = recording["intensity"]
    difference = recording["adjacent_frame_difference"]
    translation = recording["translation"]
    observed = {
        "absolute_relative_intensity_drift": abs(intensity["robust_linear_drift"]["fitted_relative_change_over_sampled_span"]),
        "high_sensor_rail_fraction": intensity["sensor_rail_occupancy"]["high_fraction"],
        "normalized_frame_difference_p95": difference["normalized_p95_abs_difference"]["p95"],
        "global_translation_p95_px": translation["global_magnitude_px"]["p95"],
        "tile_median_disagreement_p95_px": translation["tile_median_disagreement_px"]["p95"],
        "valid_global_pair_fraction": translation["valid_global_pair_fraction"],
        "median_valid_tile_fraction": translation["valid_tile_fraction"]["median"],
    }
    threshold_map = asdict(thresholds)
    comparisons = {
        "absolute_relative_intensity_drift": "greater_than",
        "high_sensor_rail_fraction": "greater_than",
        "normalized_frame_difference_p95": "greater_than",
        "global_translation_p95_px": "greater_than",
        "tile_median_disagreement_p95_px": "greater_than",
        "valid_global_pair_fraction": "less_than",
        "median_valid_tile_fraction": "less_than",
    }
    threshold_key = {
        "valid_global_pair_fraction": "minimum_valid_global_pair_fraction",
        "median_valid_tile_fraction": "minimum_median_valid_tile_fraction",
    }
    checks: dict[str, Any] = {}
    for name, value in observed.items():
        limit = float(threshold_map[threshold_key.get(name, name)])
        missing = value is None
        if missing:
            triggered = True
        elif comparisons[name] == "greater_than":
            triggered = bool(float(value) > limit)
        else:
            triggered = bool(float(value) < limit)
        checks[name] = {
            "observed": value,
            "review_threshold": limit,
            "comparison": comparisons[name],
            "triggered": triggered,
            "missing_is_review_trigger": True,
        }
    reasons = sorted(name for name, check in checks.items() if check["triggered"])
    extreme = 0
    for name in (
        "absolute_relative_intensity_drift",
        "high_sensor_rail_fraction",
        "normalized_frame_difference_p95",
        "global_translation_p95_px",
        "tile_median_disagreement_p95_px",
    ):
        value = observed[name]
        limit = checks[name]["review_threshold"]
        if value is not None and float(value) > 2.0 * limit:
            extreme += 1
    if not reasons:
        severity = "none_detected_at_screening_thresholds"
    elif extreme or len(reasons) >= 2:
        severity = "high_review_priority"
    else:
        severity = "moderate_review_priority"
    return {
        "decision": "review_required" if reasons else "screen_clear",
        "screening_severity": severity,
        "triggered_reasons": reasons,
        "checks": checks,
        "interpretation": "A trigger identifies a nuisance sensitivity to inspect; it is not a biological or causal classification.",
    }


def audit_recording_motion(
    descriptor: RecordingDescriptor,
    *,
    config: MotionAuditConfig = MotionAuditConfig(),
) -> dict[str, Any]:
    """Run the bounded deterministic screen for one raw recording."""
    assert_spon_excluded((descriptor,))
    movie = open_recording_memmap(descriptor)
    if descriptor.frame_count < 2:
        raise ValueError("motion audit requires at least two source frames")
    intensity_indices = deterministic_uniform_frame_indices(descriptor.frame_count, config.frames_per_recording)
    pair_starts = deterministic_uniform_frame_indices(descriptor.frame_count - 1, config.frames_per_recording)

    intensity_samples: list[np.ndarray] = []
    frame_medians: list[float] = []
    low_rail_count = 0
    high_rail_count = 0
    rail_resolved = True
    sampled_pixel_count = 0
    for frame_index in intensity_indices:
        raw = np.asarray(
            movie[frame_index, :: config.intensity_spatial_stride, :: config.intensity_spatial_stride]
        )
        low, high, resolved = _rail_counts(raw, movie.dtype)
        low_rail_count += low
        high_rail_count += high
        rail_resolved = rail_resolved and resolved
        sampled_pixel_count += int(raw.size)
        sample = raw.astype(np.float32, copy=False).reshape(-1)
        intensity_samples.append(sample)
        frame_medians.append(float(np.median(sample)))
    pooled = np.concatenate(intensity_samples).astype(np.float64, copy=False)
    pooled_center = float(np.median(pooled))
    pooled_scale = _normal_consistent_mad(pooled, pooled_center)
    scale_floor_applied = pooled_scale < config.minimum_robust_scale
    normalization_scale = max(pooled_scale, config.minimum_robust_scale)
    drift = _theil_sen_drift(intensity_indices, frame_medians, config.minimum_robust_scale)

    pair_rows: list[dict[str, Any]] = []
    normalized_medians: list[float] = []
    normalized_p95s: list[float] = []
    global_magnitudes: list[float] = []
    valid_tile_fractions: list[float] = []
    tile_disagreement_medians: list[float] = []
    tile_disagreement_p95: list[float] = []
    for start in pair_starts:
        reference = np.asarray(movie[start])
        moving = np.asarray(movie[start + 1])
        reference_sample = reference[:: config.intensity_spatial_stride, :: config.intensity_spatial_stride].astype(
            np.float32,
            copy=False,
        )
        moving_sample = moving[:: config.intensity_spatial_stride, :: config.intensity_spatial_stride].astype(
            np.float32,
            copy=False,
        )
        absolute_difference = np.abs(moving_sample - reference_sample)
        median_difference = float(np.median(absolute_difference))
        p95_difference = float(np.quantile(absolute_difference, 0.95))
        normalized_median = median_difference / normalization_scale
        normalized_p95 = p95_difference / normalization_scale
        normalized_medians.append(normalized_median)
        normalized_p95s.append(normalized_p95)

        registration = _pair_translation_diagnostics(reference, moving, config=config)
        if registration["global"]["valid"]:
            global_magnitudes.append(float(registration["global"]["magnitude_px"]))
        tile_fraction = float(registration["tile_summary"]["valid_fraction"])
        valid_tile_fractions.append(tile_fraction)
        disagreement_median = registration["tile_summary"]["disagreement_px"]["median"]
        if disagreement_median is not None:
            tile_disagreement_medians.append(float(disagreement_median))
        disagreement_p95 = registration["tile_summary"]["disagreement_px"]["p95"]
        if disagreement_p95 is not None:
            tile_disagreement_p95.append(float(disagreement_p95))
        pair_rows.append(
            {
                "reference_frame_zero": int(start),
                "moving_frame_zero": int(start + 1),
                "frame_gap": 1,
                "absolute_difference_intensity_units": {
                    "median": median_difference,
                    "p95": p95_difference,
                },
                "absolute_difference_over_recording_robust_scale": {
                    "median": float(normalized_median),
                    "p95": float(normalized_p95),
                },
                "registration": registration,
            }
        )

    valid_global_fraction = float(len(global_magnitudes) / len(pair_rows))
    recording = {
        "recording": descriptor.to_manifest(),
        "sampling": {
            "intensity_frame_indices_zero": list(intensity_indices),
            "adjacent_pair_start_indices_zero": list(pair_starts),
            "frame_count_per_lane": len(intensity_indices),
            "intensity_spatial_stride": config.intensity_spatial_stride,
            "registration_spatial_stride": config.registration_spatial_stride,
            "full_field_sampling": True,
        },
        "units": {
            "temporal": "frames",
            "spatial": "native_pixels",
            "intensity": descriptor.dtype + "_raw_units",
            "seconds_available": descriptor.frame_interval_s is not None,
            "micrometers_available": descriptor.pixel_size_um is not None,
            "conversion_policy": "do_not_infer_unresolved_cadence_or_pixel_scale",
        },
        "intensity": {
            "sampled_pixel_count": sampled_pixel_count,
            "sampled_raw_intensity": {
                "p01": _quantile(pooled, 0.01),
                "median": pooled_center,
                "p99": _quantile(pooled, 0.99),
                "normal_consistent_mad": pooled_scale,
            },
            "frame_median_intensity": {
                **_distribution_summary(frame_medians),
                "robust_cv": _normal_consistent_mad(np.asarray(frame_medians))
                / max(abs(float(np.median(frame_medians))), config.minimum_robust_scale),
            },
            "robust_linear_drift": drift,
            "sensor_rail_occupancy": {
                "status": "resolved_for_integer_dtype" if rail_resolved else "unresolved_for_noninteger_dtype",
                "low_count": low_rail_count if rail_resolved else None,
                "high_count": high_rail_count if rail_resolved else None,
                "low_fraction": float(low_rail_count / sampled_pixel_count) if rail_resolved else None,
                "high_fraction": float(high_rail_count / sampled_pixel_count) if rail_resolved else None,
                "denominator": sampled_pixel_count if rail_resolved else None,
            },
            "frame_difference_normalization": {
                "scale": normalization_scale,
                "estimator": "normal_consistent_mad_of_uniform_full_field_intensity_samples",
                "scale_floor_applied": scale_floor_applied,
            },
        },
        "adjacent_frame_difference": {
            "pair_count": len(pair_rows),
            "frame_gap": 1,
            "normalized_median_abs_difference": _distribution_summary(normalized_medians),
            "normalized_p95_abs_difference": _distribution_summary(normalized_p95s),
            "interpretation": "Nonspecific temporal change; activity, gain, noise, and motion can all contribute.",
        },
        "translation": {
            "algorithm": MOTION_AUDIT_ALGORITHM,
            "direction": "shift_applied_to_moving_frame_to_register_it_to_reference",
            "global_magnitude_px": _distribution_summary(global_magnitudes),
            "valid_global_pair_count": len(global_magnitudes),
            "valid_global_pair_fraction": valid_global_fraction,
            "valid_tile_fraction": _distribution_summary(valid_tile_fractions),
            "tile_median_disagreement_px": _distribution_summary(tile_disagreement_medians),
            "tile_tail_disagreement_p95_px": _distribution_summary(tile_disagreement_p95),
            "pair_diagnostics": pair_rows,
            "interpretation": "Translation-like screen only; no rotation, deformation, correction, or causal attribution.",
        },
    }
    recording["gate"] = _evaluate_recording_gate(recording, config.thresholds)
    portable_fingerprint = {
        "recording": descriptor.to_manifest(),
        "config": config.to_manifest(),
        "intensity_frame_indices_zero": list(intensity_indices),
        "adjacent_pair_start_indices_zero": list(pair_starts),
    }
    recording["sampling_manifest_sha256"] = stable_hash(portable_fingerprint)
    return recording


def _worst_observed(recordings: Sequence[dict[str, Any]], check_name: str) -> float | None:
    values = [item["gate"]["checks"][check_name]["observed"] for item in recordings]
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    if check_name in {"valid_global_pair_fraction", "median_valid_tile_fraction"}:
        return min(finite)
    return max(finite)


def _aggregate_gate(recordings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(item["gate"]["decision"] for item in recordings)
    reason_counts = Counter(
        reason
        for item in recordings
        for reason in item["gate"]["triggered_reasons"]
    )
    review_ids = [
        item["recording"]["recording_id"]
        for item in recordings
        if item["gate"]["decision"] == "review_required"
    ]
    check_names = tuple(recordings[0]["gate"]["checks"]) if recordings else ()
    return {
        "decision": "review_required" if review_ids else "screen_clear",
        "recording_count": len(recordings),
        "decision_counts": dict(sorted(decisions.items())),
        "review_recording_ids": review_ids,
        "triggered_reason_counts": dict(sorted(reason_counts.items())),
        "worst_observed_by_check": {
            name: _worst_observed(recordings, name)
            for name in check_names
        },
        "next_step_if_clear": "Proceed only to matched representation baselines with explicit nuisance probes.",
        "next_step_if_review_required": "Inspect flagged recordings and run nuisance-stratified sensitivity analysis before promoting JEPA evidence.",
    }


def _verify_noncanonical_live_inventory(inventory: RecordingInventory, *, verify_hashes: bool) -> dict[str, Any]:
    rows = []
    for descriptor in inventory.recordings:
        movie = open_recording_memmap(descriptor)
        hash_matches = None
        if verify_hashes:
            hash_matches = sha256_file(descriptor.resolved_path) == descriptor.sha256
            if not hash_matches:
                raise ValueError(f"recording hash mismatch: {descriptor.recording_id}")
        rows.append(
            {
                "recording_id": descriptor.recording_id,
                "shape_matches": tuple(movie.shape) == descriptor.shape,
                "dtype_matches": movie.dtype.name == descriptor.dtype,
                "hash_matches": hash_matches,
            }
        )
    return {
        "status": "passed_noncanonical_fixture_or_subset",
        "checks": rows,
        "hashes_verified": verify_hashes,
    }


def run_jepa_motion_audit(
    inventory: RecordingInventory,
    *,
    config: MotionAuditConfig = MotionAuditConfig(),
) -> dict[str, Any]:
    """Audit an inventory and return one deterministic, portable JSON object.

    The default fails closed unless the inventory is the frozen 11-recording
    060126 corpus.  Tests and deliberately bounded fixtures may opt out of that
    *shape/count* contract, but Spon exclusion and live read-only memmaps remain
    mandatory.
    """
    assert_spon_excluded(inventory.recordings)
    if config.require_frozen_060126_inventory:
        inventory_validation = validate_060126_inventory(
            inventory,
            verify_live=True,
            verify_hashes=config.verify_hashes,
        )
    else:
        inventory_validation = _verify_noncanonical_live_inventory(
            inventory,
            verify_hashes=config.verify_hashes,
        )

    recording_results = [
        audit_recording_motion(descriptor, config=config)
        for descriptor in sorted(inventory.recordings, key=lambda item: item.recording_number)
    ]
    manifest = {
        "schema_version": MOTION_AUDIT_SCHEMA_VERSION,
        "algorithm": MOTION_AUDIT_ALGORITHM,
        "dataset_id": inventory.dataset_id,
        "inventory": {
            "descriptor_uri": inventory.descriptor_uri,
            "descriptor_sha256": inventory.descriptor_sha256,
            "independence": inventory.independence,
            "recordings": [item.to_manifest() for item in inventory.recordings],
        },
        "config": config.to_manifest(),
        "sample_indices": {
            item["recording"]["recording_id"]: item["sampling"]
            for item in recording_results
        },
        "coordinate_convention": "x_is_column_y_is_row_zero_based_half_open_bounds",
        "unit_contract": "native_pixels_and_frames_only_when_physical_metadata_is_unresolved",
        "source_policy": "raw_060126_only_spon_excluded",
        "write_policy": "pure_return_value_no_raw_copy_no_output_write",
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    result = {
        "schema_version": MOTION_AUDIT_SCHEMA_VERSION,
        "status": "screen_complete",
        "manifest": manifest,
        "inventory_validation": inventory_validation,
        "aggregate_gate": _aggregate_gate(recording_results),
        "recordings": recording_results,
        "limitations": [
            "This is a label-free acquisition-confound screen, not a neuron or artifact classifier.",
            "Sparse deterministic samples can miss short transients between sampled adjacent-frame pairs.",
            "Phase correlation measures translation-like change and does not resolve rotation or deformation.",
            "Frame difference is nonspecific and can include neural activity, measurement noise, gain change, or motion.",
            "No motion correction is applied and no motion-corrected, biological, or causal conclusion is supported.",
            "Cadence and physical pixel scale remain unresolved wherever absent from the frozen descriptor.",
        ],
    }
    result["result_sha256"] = stable_hash(result)
    return result


def assert_json_ready(value: dict[str, Any]) -> None:
    """Fail if an audit payload contains non-portable paths or non-finite data."""
    import json

    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    if '"resolved_path"' in encoded:
        raise ValueError("motion-audit payload leaked a runtime resolved_path")


__all__ = [
    "MOTION_AUDIT_ALGORITHM",
    "MOTION_AUDIT_SCHEMA_VERSION",
    "MotionAuditConfig",
    "MotionAuditThresholds",
    "assert_json_ready",
    "audit_recording_motion",
    "estimate_phase_translation",
    "run_jepa_motion_audit",
]
