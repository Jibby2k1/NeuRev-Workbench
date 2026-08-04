"""Signed multi-scale local normalization with auditable reference supports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Estimator = Literal["mean_std", "median_mad"]


@dataclass(frozen=True)
class SpatialMSLNContext:
    context_id: str
    outer_width_px: int
    guard_width_px: int
    estimator: Estimator = "mean_std"
    scale_floor_percentile: float = 10.0

    def __post_init__(self) -> None:
        _validate_id(self.context_id)
        if (
            self.outer_width_px < 3
            or self.outer_width_px > 31
            or self.outer_width_px % 2 != 1
            or self.guard_width_px < 1
            or self.guard_width_px % 2 != 1
            or self.guard_width_px >= self.outer_width_px
        ):
            raise ValueError("spatial widths must be odd with 1 <= guard < outer <= 31")
        _validate_estimator(self.estimator)
        _validate_percentile(self.scale_floor_percentile)


@dataclass(frozen=True)
class TemporalMSLNContext:
    context_id: str
    window_frames: int
    guard_frames: int = 1
    estimator: Estimator = "mean_std"
    causal: bool = True
    scale_floor_percentile: float = 10.0

    def __post_init__(self) -> None:
        _validate_id(self.context_id)
        if self.window_frames < 2 or not 0 <= self.guard_frames < self.window_frames:
            raise ValueError("temporal window must exceed its nonnegative guard")
        if not self.causal:
            raise ValueError("v1 temporal MSLN is causal")
        _validate_estimator(self.estimator)
        _validate_percentile(self.scale_floor_percentile)


@dataclass(frozen=True)
class SequentialSTContext:
    context_id: str
    spatial_context_id: str
    temporal_context_id: str
    order: Literal["temporal_then_spatial", "spatial_then_temporal"]

    def __post_init__(self) -> None:
        for value in (
            self.context_id,
            self.spatial_context_id,
            self.temporal_context_id,
        ):
            _validate_id(value)
        if self.order not in {"temporal_then_spatial", "spatial_then_temporal"}:
            raise ValueError("invalid sequential context order")


@dataclass(frozen=True)
class JointSTContext:
    """Causal joint space-time reference volume."""

    context_id: str
    spatial_outer_width_px: int
    spatial_guard_width_px: int
    temporal_window_frames: int
    temporal_guard_frames: int = 1
    estimator: Estimator = "mean_std"
    scale_floor_percentile: float = 10.0

    def __post_init__(self) -> None:
        _validate_id(self.context_id)
        SpatialMSLNContext(
            self.context_id,
            self.spatial_outer_width_px,
            self.spatial_guard_width_px,
            self.estimator,
            self.scale_floor_percentile,
        )
        TemporalMSLNContext(
            self.context_id,
            self.temporal_window_frames,
            self.temporal_guard_frames,
            self.estimator,
            True,
            self.scale_floor_percentile,
        )


@dataclass(frozen=True)
class MSLNResult:
    values: np.ndarray
    valid_frames: np.ndarray
    scale_floor: float
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 3
            or self.valid_frames.shape != (len(self.values),)
            or self.values.dtype != np.float32
            or not np.isfinite(self.values).all()
            or not np.isfinite(self.scale_floor)
            or self.scale_floor <= 0
        ):
            raise ValueError("invalid MSLN result")


def _validate_id(value: str) -> None:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value):
        raise ValueError("context IDs must use lowercase ASCII letters, digits, and underscores")


def _validate_estimator(value: str) -> None:
    if value not in {"mean_std", "median_mad"}:
        raise ValueError("estimator must be mean_std or median_mad")


def _validate_percentile(value: float) -> None:
    if not np.isfinite(value) or not 0 <= float(value) <= 100:
        raise ValueError("scale-floor percentile must lie in [0,100]")


def _video(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError("values must be a finite non-empty TYX array")
    return result.astype(np.float64, copy=False)


def robust_center_scale(
    values: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact median and Gaussian-consistent MAD."""
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("values must be finite and non-empty")
    center_keepdims = np.median(array, axis=axis, keepdims=True)
    scale = 1.4826 * np.median(
        np.abs(array - center_keepdims), axis=axis, keepdims=True
    )
    if axis is None:
        return np.asarray(center_keepdims).reshape(()), np.asarray(scale).reshape(())
    axes = tuple(
        sorted(
            {item % array.ndim for item in ((axis,) if isinstance(axis, int) else axis)}
        )
    )
    return np.squeeze(center_keepdims, axis=axes), np.squeeze(scale, axis=axes)


def fit_scale_floor(
    scales: np.ndarray,
    percentile: float,
    *,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Fit a positive scalar floor from training scales only."""
    values = np.asarray(scales, dtype=np.float64)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        try:
            mask = np.broadcast_to(mask, values.shape)
        except ValueError as exc:
            raise ValueError("valid_mask cannot broadcast to scales") from exc
        values = values[mask]
    positive = values[np.isfinite(values) & (values > 0)]
    floor = float(np.percentile(positive, float(percentile))) if positive.size else 1.0
    return max(floor, np.finfo(np.float32).eps)


def _valid_box_sum(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-padded box sums plus exact valid-cell counts."""
    from scipy.ndimage import uniform_filter

    size = (1, int(width), int(width))
    area = float(int(width) ** 2)
    sums = uniform_filter(
        values, size=size, mode="constant", cval=0.0
    ).astype(np.float64, copy=False) * area
    counts = uniform_filter(
        np.ones_like(values, dtype=np.float64),
        size=size,
        mode="constant",
        cval=0.0,
    ) * area
    return sums, counts


def _spatial_mean_std(values: np.ndarray, outer: int, guard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer_sum, outer_count = _valid_box_sum(values, outer)
    guard_sum, guard_count = _valid_box_sum(values, guard)
    outer_sq_sum, _ = _valid_box_sum(values * values, outer)
    guard_sq_sum, _ = _valid_box_sum(values * values, guard)
    count = outer_count - guard_count
    if np.any(count <= 0):
        raise ValueError("spatial context has no reference cells")
    total = outer_sum - guard_sum
    total_sq = outer_sq_sum - guard_sq_sum
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    return mean, np.sqrt(variance), count


def _quiet_mask_for(values: np.ndarray, quiet_mask: np.ndarray | None) -> np.ndarray | None:
    if quiet_mask is None:
        return None
    mask = np.asarray(quiet_mask, dtype=bool)
    if mask.shape == (len(values),):
        mask = mask[:, None, None]
    try:
        return np.broadcast_to(mask, values.shape)
    except ValueError as exc:
        raise ValueError("quiet_mask must be frame-level or align with values") from exc


def spatial_msln(
    values: np.ndarray,
    context: SpatialMSLNContext,
    *,
    scale_floor: float | None = None,
    quiet_mask: np.ndarray | None = None,
) -> MSLNResult:
    """Compute signed square-annulus local normalization."""
    video = _video(values)
    if context.estimator != "mean_std":
        raise NotImplementedError(
            "full-field rolling spatial median/MAD is intentionally outside v1"
        )
    mean, scale, count = _spatial_mean_std(
        video, context.outer_width_px, context.guard_width_px
    )
    fitted_floor = (
        fit_scale_floor(
            scale,
            context.scale_floor_percentile,
            valid_mask=_quiet_mask_for(video, quiet_mask),
        )
        if scale_floor is None
        else float(scale_floor)
    )
    if not np.isfinite(fitted_floor) or fitted_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    z = ((video - mean) / np.maximum(scale, fitted_floor)).astype(np.float32)
    return MSLNResult(
        values=z,
        valid_frames=np.ones(len(video), dtype=bool),
        scale_floor=fitted_floor,
        diagnostics={
            "context_id": context.context_id,
            "kind": "spatial",
            "estimator": context.estimator,
            "outer_width_px": context.outer_width_px,
            "guard_width_px": context.guard_width_px,
            "reference_count_min": int(np.min(count)),
            "reference_count_max": int(np.max(count)),
            "boundary_corrected": True,
        },
    )


def _temporal_mean_std(
    video: np.ndarray,
    window_frames: int,
    guard_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = len(video)
    cumulative = np.concatenate(
        [np.zeros((1, *video.shape[1:]), dtype=np.float64), np.cumsum(video, axis=0)],
        axis=0,
    )
    cumulative_sq = np.concatenate(
        [
            np.zeros((1, *video.shape[1:]), dtype=np.float64),
            np.cumsum(video * video, axis=0),
        ],
        axis=0,
    )
    mean = np.zeros_like(video)
    scale = np.zeros_like(video)
    count = np.zeros(frames, dtype=np.int32)
    for index in range(window_frames, frames):
        start = index - window_frames
        stop = index - guard_frames
        reference_count = stop - start
        total = cumulative[stop] - cumulative[start]
        total_sq = cumulative_sq[stop] - cumulative_sq[start]
        local_mean = total / reference_count
        variance = np.maximum(total_sq / reference_count - local_mean * local_mean, 0.0)
        mean[index] = local_mean
        scale[index] = np.sqrt(variance)
        count[index] = reference_count
    return mean, scale, count


def temporal_msln(
    values: np.ndarray,
    context: TemporalMSLNContext,
    *,
    scale_floor: float | None = None,
    quiet_mask: np.ndarray | None = None,
) -> MSLNResult:
    """Compute causal temporal MSLN without a sliding-window tensor."""
    video = _video(values)
    if context.window_frames >= len(video):
        raise ValueError("temporal window must be shorter than the video")
    if context.estimator != "mean_std":
        raise NotImplementedError(
            "full-field rolling temporal median/MAD is an authorized ablation only"
        )
    mean, scale, count = _temporal_mean_std(
        video, context.window_frames, context.guard_frames
    )
    valid = count > 0
    calibration_mask = valid[:, None, None]
    supplied = _quiet_mask_for(video, quiet_mask)
    if supplied is not None:
        calibration_mask = calibration_mask & supplied
    fitted_floor = (
        fit_scale_floor(
            scale,
            context.scale_floor_percentile,
            valid_mask=calibration_mask,
        )
        if scale_floor is None
        else float(scale_floor)
    )
    if not np.isfinite(fitted_floor) or fitted_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    z = np.zeros_like(video, dtype=np.float32)
    z[valid] = (
        (video[valid] - mean[valid]) / np.maximum(scale[valid], fitted_floor)
    ).astype(np.float32)
    return MSLNResult(
        values=z,
        valid_frames=valid,
        scale_floor=fitted_floor,
        diagnostics={
            "context_id": context.context_id,
            "kind": "temporal",
            "estimator": context.estimator,
            "window_frames": context.window_frames,
            "guard_frames": context.guard_frames,
            "causal": True,
            "invalid_prefix_frames": int(np.argmax(valid)) if np.any(valid) else len(video),
            "reference_count": int(np.max(count)),
        },
    )


def sequential_msln(
    values: np.ndarray,
    context: SequentialSTContext,
    *,
    spatial_context: SpatialMSLNContext,
    temporal_context: TemporalMSLNContext,
    quiet_mask: np.ndarray | None = None,
) -> MSLNResult:
    """Compose declared spatial and temporal MSLN transforms exactly."""
    if context.spatial_context_id != spatial_context.context_id:
        raise ValueError("sequential spatial context ID mismatch")
    if context.temporal_context_id != temporal_context.context_id:
        raise ValueError("sequential temporal context ID mismatch")
    if context.order == "temporal_then_spatial":
        first = temporal_msln(values, temporal_context, quiet_mask=quiet_mask)
        second = spatial_msln(
            first.values,
            spatial_context,
            quiet_mask=(
                first.valid_frames
                if quiet_mask is None
                else first.valid_frames & np.asarray(quiet_mask, dtype=bool)
            ),
        )
    else:
        first = spatial_msln(values, spatial_context, quiet_mask=quiet_mask)
        second = temporal_msln(
            first.values, temporal_context, quiet_mask=quiet_mask
        )
    result = second.values.copy()
    result[~first.valid_frames] = 0
    valid = first.valid_frames & second.valid_frames
    result[~valid] = 0
    return MSLNResult(
        values=result,
        valid_frames=valid,
        scale_floor=second.scale_floor,
        diagnostics={
            "context_id": context.context_id,
            "kind": "sequential_spatiotemporal",
            "order": context.order,
            "first": first.diagnostics,
            "second": second.diagnostics,
        },
    )


def causal_joint_msln(
    values: np.ndarray,
    context: JointSTContext,
    *,
    scale_floor: float | None = None,
    quiet_mask: np.ndarray | None = None,
) -> MSLNResult:
    """Normalize against one causal 3-D annulus-by-time reference."""
    source = np.asarray(values)
    if source.ndim != 3 or not source.size:
        raise ValueError("values must be a finite non-empty TYX array")
    if context.temporal_window_frames >= len(source):
        raise ValueError("temporal window must be shorter than the video")
    if context.estimator != "mean_std":
        raise NotImplementedError("joint full-field median/MAD is not implemented")

    frames, height, width = source.shape
    window = int(context.temporal_window_frames)
    temporal_guard = int(context.temporal_guard_frames)
    reference_frames = window - temporal_guard
    ring_length = window + temporal_guard + 1
    ring_sum = np.zeros((ring_length, height, width), dtype=np.float32)
    ring_sum_sq = np.zeros_like(ring_sum)
    ring_tags = np.full(ring_length, -1, dtype=np.int64)
    running_sum = np.zeros((height, width), dtype=np.float64)
    running_sum_sq = np.zeros_like(running_sum)
    numerator = np.zeros((frames, height, width), dtype=np.float32)
    local_scale = np.zeros_like(numerator)
    valid = np.zeros(frames, dtype=bool)

    sample_step = max(1, frames // 16)
    offset = float(
        np.median(np.asarray(source[::sample_step, ::4, ::4], dtype=np.float64))
    )
    spatial_count: np.ndarray | None = None
    for frame in range(frames):
        add_index = frame - temporal_guard - 1
        if add_index >= 0:
            slot = add_index % ring_length
            if ring_tags[slot] != add_index:
                raise RuntimeError("joint MSLN rolling-buffer add invariant failed")
            running_sum += ring_sum[slot]
            running_sum_sq += ring_sum_sq[slot]
        remove_index = frame - window - 1
        if remove_index >= 0:
            slot = remove_index % ring_length
            if ring_tags[slot] != remove_index:
                raise RuntimeError("joint MSLN rolling-buffer remove invariant failed")
            running_sum -= ring_sum[slot]
            running_sum_sq -= ring_sum_sq[slot]

        centered = np.asarray(source[frame], dtype=np.float64) - offset
        centered_sq = np.square(centered, dtype=np.float64)
        outer_sum, outer_count = _valid_box_sum(
            centered[None], context.spatial_outer_width_px
        )
        guard_sum, guard_count = _valid_box_sum(
            centered[None], context.spatial_guard_width_px
        )
        outer_sq, _ = _valid_box_sum(
            centered_sq[None], context.spatial_outer_width_px
        )
        guard_sq, _ = _valid_box_sum(
            centered_sq[None], context.spatial_guard_width_px
        )
        annulus_sum = outer_sum[0] - guard_sum[0]
        annulus_sum_sq = outer_sq[0] - guard_sq[0]
        if spatial_count is None:
            spatial_count = outer_count[0] - guard_count[0]
            if np.any(spatial_count <= 0):
                raise ValueError("joint context has no spatial reference cells")
        slot = frame % ring_length
        ring_sum[slot] = annulus_sum
        ring_sum_sq[slot] = annulus_sum_sq
        ring_tags[slot] = frame

        if frame >= window:
            count = spatial_count * reference_frames
            mean = running_sum / count
            variance = np.maximum(running_sum_sq / count - mean * mean, 0.0)
            numerator[frame] = (centered - mean).astype(np.float32)
            local_scale[frame] = np.sqrt(variance).astype(np.float32)
            valid[frame] = True

    calibration_mask = valid[:, None, None]
    supplied = _quiet_mask_for(source, quiet_mask)
    if supplied is not None:
        calibration_mask = calibration_mask & supplied
    fitted_floor = (
        fit_scale_floor(
            local_scale,
            context.scale_floor_percentile,
            valid_mask=calibration_mask,
        )
        if scale_floor is None
        else float(scale_floor)
    )
    if not np.isfinite(fitted_floor) or fitted_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    for start in range(0, frames, 16):
        stop = min(start + 16, frames)
        numerator[start:stop] /= np.maximum(
            local_scale[start:stop], fitted_floor
        )
    numerator[~valid] = 0
    if not np.isfinite(numerator).all():
        raise ValueError("joint MSLN produced non-finite values")
    return MSLNResult(
        values=numerator,
        valid_frames=valid,
        scale_floor=fitted_floor,
        diagnostics={
            "context_id": context.context_id,
            "kind": "causal_joint_spatiotemporal",
            "estimator": context.estimator,
            "spatial_outer_width_px": context.spatial_outer_width_px,
            "spatial_guard_width_px": context.spatial_guard_width_px,
            "temporal_window_frames": window,
            "temporal_guard_frames": temporal_guard,
            "reference_frame_count": reference_frames,
            "reference_count_min": int(np.min(spatial_count) * reference_frames),
            "reference_count_max": int(np.max(spatial_count) * reference_frames),
            "causal": True,
            "current_frame_excluded": True,
            "boundary_corrected": True,
            "centering_offset": offset,
            "invalid_prefix_frames": window,
        },
    )
