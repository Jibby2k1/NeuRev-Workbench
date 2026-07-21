"""Streamed CPU detector for dark-soma anatomy and positive excitation.

Dark cores are anatomical anchors; CFAR remains a positive-excursion detector.
Only two-dimensional count maps and small traces survive chunk processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from neurobench.algorithms.cfar import robust_local_cfar
from neurobench.data.video import VideoChunk, VideoStore, iter_video_chunks

from .config import CFARConfig
from .zones import DarkSomaZoneConfig, DarkSomaZones, detect_dark_soma_zones

FrameSource = str | Path | VideoStore


@dataclass(frozen=True)
class StreamedDetectorResult:
    """JSON-ready summary and bounded arrays from one detector run."""

    summary: dict[str, Any]
    baseline_raw: np.ndarray
    baseline_normalized: np.ndarray
    dark_contrast: np.ndarray
    dark_robust_z: np.ndarray
    core_mask: np.ndarray
    ring_mask: np.ndarray
    count_maps: dict[str, np.ndarray]
    frame_indices: np.ndarray
    ui_frames: np.ndarray
    is_score_frame: np.ndarray
    traces: dict[str, np.ndarray]
    zone_ring_traces: dict[str, np.ndarray]

    def array_payload(self) -> dict[str, np.ndarray]:
        """Return flat, stable names suitable for ``numpy.savez_compressed``."""
        payload = {
            "baseline_raw": self.baseline_raw,
            "baseline_normalized": self.baseline_normalized,
            "dark_contrast": self.dark_contrast,
            "dark_robust_z": self.dark_robust_z,
            "core_mask": self.core_mask,
            "ring_mask": self.ring_mask,
            "frame_indices": self.frame_indices,
            "ui_frames": self.ui_frames,
            "is_score_frame": self.is_score_frame,
        }
        payload.update(
            {f"count_{key}": value for key, value in self.count_maps.items()}
        )
        payload.update({f"trace_{key}": value for key, value in self.traces.items()})
        payload.update(
            {f"zone_trace_{key}": value for key, value in self.zone_ring_traces.items()}
        )
        return payload


@dataclass(frozen=True)
class _Calibration:
    baseline: np.ndarray
    lower: float
    upper: float
    scale: float
    method: str
    constant: bool
    shape: tuple[int, int]


class _LaneAccumulator:
    def __init__(
        self,
        shape: tuple[int, int],
        frame_count: int,
        zone_rings: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.control_counts = np.zeros(shape, dtype=np.uint32)
        self.score_counts = np.zeros(shape, dtype=np.uint32)
        self.global_fraction = np.zeros(frame_count, dtype=np.float32)
        self.core_fraction = np.zeros(frame_count, dtype=np.float32)
        self.ring_fraction = np.zeros(frame_count, dtype=np.float32)
        self.zone_ring_fraction = np.zeros(
            (frame_count, len(zone_rings)), dtype=np.float32
        )
        self._zone_rings = zone_rings

    def consume(
        self, mask: np.ndarray, *, trace_start: int, phase: str,
        core_mask: np.ndarray, ring_mask: np.ndarray,
    ) -> None:
        frame_count = int(mask.shape[0])
        trace_slice = slice(trace_start, trace_start + frame_count)
        counts = mask.sum(axis=0, dtype=np.uint32)
        if phase == "control":
            self.control_counts += counts
        else:
            self.score_counts += counts
        self.global_fraction[trace_slice] = mask.mean(axis=(1, 2), dtype=np.float64)
        self.core_fraction[trace_slice] = _masked_fraction(mask, core_mask)
        self.ring_fraction[trace_slice] = _masked_fraction(mask, ring_mask)
        for zone_index, (yy, xx) in enumerate(self._zone_rings):
            if yy.size:
                self.zone_ring_fraction[trace_slice, zone_index] = mask[:, yy, xx].mean(
                    axis=1, dtype=np.float64
                )


class _SignalAccumulator:
    """Retain half-wave residual traces, never temporal image masks."""

    def __init__(self, frame_count: int,
        zone_rings: list[tuple[np.ndarray, np.ndarray]],) -> None:
        self.global_mean = np.zeros(frame_count, dtype=np.float32)
        self.core_mean = np.zeros(frame_count, dtype=np.float32)
        self.ring_mean = np.zeros(frame_count, dtype=np.float32)
        self.zone_ring_mean = np.zeros(
            (frame_count, len(zone_rings)), dtype=np.float32)
        self._zone_rings = zone_rings
    def consume(
        self, positive_residual: np.ndarray, *, trace_start: int,
        core_mask: np.ndarray, ring_mask: np.ndarray,
    ) -> None:
        frame_count = int(positive_residual.shape[0])
        trace_slice = slice(trace_start, trace_start + frame_count)
        self.global_mean[trace_slice] = positive_residual.mean(
            axis=(1, 2), dtype=np.float64)
        self.core_mean[trace_slice] = _masked_mean(positive_residual, core_mask)
        self.ring_mean[trace_slice] = _masked_mean(positive_residual, ring_mask)
        for zone_index, (yy, xx) in enumerate(self._zone_rings):
            if yy.size:
                values = positive_residual[:, yy, xx]
                self.zone_ring_mean[trace_slice, zone_index] = values.mean(
                    axis=1, dtype=np.float64)


def run_streamed_detector(
    frame_source: FrameSource, *, control_start: int, control_stop: int,
    score_start: int, score_stop: int, chunk_frames: int = 4,
    cfar: CFARConfig | None = None, zone_config: DarkSomaZoneConfig | None = None,
    zone_threshold_method: str = "p99",
    normalization_sample_limit: int = 1_000_000,
) -> StreamedDetectorResult:
    """Calibrate on control frames and score positive excitation in bounded chunks.

    Ranges are zero-based and half-open; outputs also expose one-based UI frames.
    """
    _validate_inputs(control_start, control_stop, score_start, score_stop,
        chunk_frames, normalization_sample_limit, zone_threshold_method,)
    cfar_config = cfar or CFARConfig()
    cfar_config.validate()
    dark_config = zone_config or DarkSomaZoneConfig()
    dark_config.validate()

    calibration = _calibrate_control(frame_source, control_start, control_stop,
        chunk_frames, normalization_sample_limit,)
    baseline_normalized = _normalize(
        calibration.baseline, calibration.lower, calibration.scale)
    anatomy = detect_dark_soma_zones(calibration.baseline, dark_config)
    zone_rings = _zone_ring_indices(calibration.shape, anatomy, dark_config)

    control_frames = control_stop - control_start
    score_frames = score_stop - score_start
    total_frames = control_frames + score_frames
    raw_lane = _LaneAccumulator(calibration.shape, total_frames, zone_rings)
    residual_lane = _LaneAccumulator(calibration.shape, total_frames, zone_rings)
    signal_lane = _SignalAccumulator(total_frames, zone_rings)

    for phase, start, stop, trace_base in (
        ("control", control_start, control_stop, 0),
        ("score", score_start, score_stop, control_frames),
    ):
        expected = start
        for chunk in _iter_source_chunks(
            frame_source, chunk_frames=chunk_frames, start=start, stop=stop):
            frames = _validated_chunk(chunk, expected, calibration.shape)
            expected = chunk.end_frame
            trace_start = trace_base + chunk.start_frame - start
            normalized = _normalize(frames, calibration.lower, calibration.scale)

            raw_mask = _cfar_mask(normalized, cfar_config)
            raw_lane.consume(raw_mask, trace_start=trace_start, phase=phase,
                core_mask=anatomy.core_mask, ring_mask=anatomy.ring_mask,)
            del raw_mask

            residual = normalized - baseline_normalized[None, :, :]
            np.maximum(residual, 0.0, out=residual)
            signal_lane.consume(residual, trace_start=trace_start,
                core_mask=anatomy.core_mask, ring_mask=anatomy.ring_mask,)
            residual_mask = _cfar_mask(residual, cfar_config)
            residual_lane.consume(residual_mask, trace_start=trace_start, phase=phase,
                core_mask=anatomy.core_mask, ring_mask=anatomy.ring_mask,)
            del residual_mask, residual, normalized, frames
        if expected != stop:
            raise RuntimeError(f"{phase} iterator stopped at frame {expected}, expected {stop}")

    frame_indices = np.concatenate(
        (
            np.arange(control_start, control_stop, dtype=np.int64),
            np.arange(score_start, score_stop, dtype=np.int64),
        )
    )
    is_score = np.concatenate(
        (
            np.zeros(control_frames, dtype=bool),
            np.ones(score_frames, dtype=bool),
        )
    )
    raw_metrics, raw_activation = _lane_summary(
        raw_lane, control_frames, score_start, zone_threshold_method)
    residual_metrics, residual_activation = _lane_summary(
        residual_lane, control_frames, score_start, zone_threshold_method)
    signal_metrics, signal_activation = _trace_summary(
        signal_lane.global_mean, signal_lane.core_mean, signal_lane.ring_mean,
        signal_lane.zone_ring_mean, control_frames, score_start,
        zone_threshold_method, measure="mean",
    )

    summary = {
        "schema_version": 1,
        "source": _source_label(frame_source),
        "image_shape": list(calibration.shape),
        "frame_ranges": {
            "control": _range_summary(control_start, control_stop),
            "score": _range_summary(score_start, score_stop),
        },
        "chunk_frames": chunk_frames,
        "normalization": {
            "method": calibration.method,
            "calibration_frames": control_frames,
            "p1": calibration.lower, "p99": calibration.upper,
            "scale": calibration.scale,
            "constant_control": calibration.constant,
            "source": "control_only",
        },
        "cfar": {
            **asdict(cfar_config),
            "device": "cpu",
            "raw_evidence": "positive normalized intensity; dark soma cores are background",
            "residual_evidence": "positive deviation from frozen control baseline",
        },
        "dark_zones": {
            "semantics": "provisional dark-core anatomy, not CFAR events",
            "config": asdict(dark_config),
            "count": len(anatomy.zones),
            "zones": anatomy.metadata(),
        },
        "metrics": {
            "raw": raw_metrics, "residual": residual_metrics,
            "positive_residual_signal": signal_metrics,
        },
        "zone_activation": {
            "raw": raw_activation, "residual": residual_activation,
            "positive_residual_signal": signal_activation,
        },
        "memory_contract": {
            "temporal_masks_retained": False,
            "retained_temporal_data": "per-frame and per-zone traces only",
            "count_map_dtype": "uint32",
        },
    }
    return StreamedDetectorResult(
        summary=summary,
        baseline_raw=calibration.baseline,
        baseline_normalized=baseline_normalized,
        dark_contrast=anatomy.contrast,
        dark_robust_z=anatomy.robust_z,
        core_mask=anatomy.core_mask,
        ring_mask=anatomy.ring_mask,
        count_maps={
            "raw_control": raw_lane.control_counts,
            "raw_score": raw_lane.score_counts,
            "residual_control": residual_lane.control_counts,
            "residual_score": residual_lane.score_counts,
        },
        frame_indices=frame_indices,
        ui_frames=frame_indices + 1,
        is_score_frame=is_score,
        traces={
            "raw_global_fraction": raw_lane.global_fraction,
            "raw_core_fraction": raw_lane.core_fraction,
            "raw_ring_fraction": raw_lane.ring_fraction,
            "residual_global_fraction": residual_lane.global_fraction,
            "residual_core_fraction": residual_lane.core_fraction,
            "residual_ring_fraction": residual_lane.ring_fraction,
            "positive_residual_global_mean": signal_lane.global_mean,
            "positive_residual_core_mean": signal_lane.core_mean,
            "positive_residual_ring_mean": signal_lane.ring_mean,
        },
        zone_ring_traces={
            "raw_ring_fraction": raw_lane.zone_ring_fraction,
            "residual_ring_fraction": residual_lane.zone_ring_fraction,
            "positive_residual_ring_mean": signal_lane.zone_ring_mean,
        },
    )


def _calibrate_control(
    source: FrameSource, start: int, stop: int,
    chunk_frames: int, sample_limit: int,
) -> _Calibration:
    baseline_sum: np.ndarray | None = None
    histogram: np.ndarray | None = None
    samples: list[np.ndarray] = []
    sample_stride = 1
    shape: tuple[int, int] | None = None
    expected = start
    method = ""
    for chunk in _iter_source_chunks(source, chunk_frames=chunk_frames, start=start, stop=stop):
        frames = _validated_chunk(chunk, expected, shape)
        expected = chunk.end_frame
        if shape is None:
            shape = tuple(int(value) for value in frames.shape[1:])
            baseline_sum = np.zeros(shape, dtype=np.float64)
            use_histogram = frames.dtype.kind == "u" and frames.dtype.itemsize <= 2
            histogram = np.zeros(65_536, dtype=np.uint64) if use_histogram else None
            sample_stride = max(
                1, int(np.ceil(((stop - start) * np.prod(shape)) / sample_limit)))
            method = "exact_uint_histogram" if use_histogram else "deterministic_spatial_sample"
        assert baseline_sum is not None
        baseline_sum += frames.sum(axis=0, dtype=np.float64)
        if histogram is not None:
            histogram += np.bincount(frames.reshape(-1), minlength=65_536).astype(
                np.uint64, copy=False)
        else:
            flat = frames.reshape(frames.shape[0], -1)
            parts = []
            for local_index, frame in enumerate(flat):
                global_start = (chunk.start_frame + local_index - start) * flat.shape[1]
                first = (-global_start) % sample_stride
                if first < flat.shape[1]:
                    parts.append(frame[first::sample_stride])
            if parts:
                samples.append(np.concatenate(parts).astype(np.float32, copy=True))
    if expected != stop or baseline_sum is None or shape is None:
        raise RuntimeError(f"control iterator stopped at frame {expected}, expected {stop}")
    baseline = (baseline_sum / float(stop - start)).astype(np.float32)
    if histogram is not None:
        lower, upper = _histogram_percentiles(histogram, (1.0, 99.0))
    else:
        sample = np.concatenate(samples)
        lower, upper = (float(value) for value in np.percentile(sample, (1.0, 99.0)))
    dynamic_range = upper - lower
    constant = bool(not np.isfinite(dynamic_range) or dynamic_range <= np.finfo(np.float32).eps)
    scale = 1.0 if constant else float(dynamic_range)
    return _Calibration(baseline, lower, upper, scale, method, constant, shape)


def _iter_source_chunks(
    source: FrameSource, *, chunk_frames: int, start: int, stop: int
) -> Iterator[VideoChunk]:
    if isinstance(source, VideoStore):
        yield from source.iter_chunks(chunk_frames, start_frame=start, end_frame=stop)
    else:
        yield from iter_video_chunks(source, chunk_size=chunk_frames,
            start_frame=start, end_frame=stop,)


def _validated_chunk(
    chunk: VideoChunk, expected_start: int, shape: tuple[int, int] | None
) -> np.ndarray:
    if chunk.start_frame != expected_start or chunk.end_frame <= chunk.start_frame:
        raise RuntimeError("frame iterator returned a gap, overlap, or empty chunk")
    frames = np.asarray(chunk.data)
    if frames.ndim != 3 or frames.shape[0] != chunk.frame_count:
        raise ValueError(f"expected [T,H,W] chunk, got {frames.shape}")
    if shape is not None and tuple(frames.shape[1:]) != shape:
        raise ValueError(f"frame shape changed from {shape} to {frames.shape[1:]}")
    if not np.issubdtype(frames.dtype, np.number) or not np.all(np.isfinite(frames)):
        raise ValueError("video chunks must contain finite numeric values")
    return frames


def _normalize(values: np.ndarray, lower: float, scale: float) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float32).copy()
    normalized -= lower
    normalized /= scale
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized


def _cfar_mask(evidence: np.ndarray, config: CFARConfig) -> np.ndarray:
    result = robust_local_cfar(evidence, pfa=config.pfa,
        guard_px=config.small_radius_px, training_radius_px=config.large_radius_px,
        epsilon=config.epsilon, device="cpu",)
    mask = np.asarray(result["mask"], dtype=bool)
    for key in ("score", "local_mean", "local_std"):
        result.pop(key, None)
    del result
    return mask


def _masked_fraction(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    if not np.any(region):
        return np.zeros(mask.shape[0], dtype=np.float32)
    return mask[:, region].mean(axis=1, dtype=np.float64).astype(np.float32)

def _masked_mean(values: np.ndarray, region: np.ndarray) -> np.ndarray:
    if not np.any(region):
        return np.zeros(values.shape[0], dtype=np.float32)
    return values[:, region].mean(axis=1, dtype=np.float64).astype(np.float32)

def _zone_ring_indices(
    shape: tuple[int, int], zones: DarkSomaZones, config: DarkSomaZoneConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices: list[tuple[np.ndarray, np.ndarray]] = []
    radius = int(np.ceil(config.ring_outer_radius))
    for zone in zones.zones:
        y0, y1 = max(0, zone.y - radius), min(shape[0], zone.y + radius + 1)
        x0, x1 = max(0, zone.x - radius), min(shape[1], zone.x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = (yy - zone.y) ** 2 + (xx - zone.x) ** 2
        local_y, local_x = np.nonzero((distance > config.ring_inner_radius**2)
            & (distance <= config.ring_outer_radius**2))
        indices.append((local_y + y0, local_x + x0))
    return indices


def _lane_summary(
    lane: _LaneAccumulator, control_frames: int, score_start: int,
    threshold_method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _trace_summary(
        lane.global_fraction, lane.core_fraction, lane.ring_fraction,
        lane.zone_ring_fraction, control_frames, score_start,
        threshold_method, measure="fraction",
    )

def _trace_summary(
    global_trace: np.ndarray, core_trace: np.ndarray, ring_trace: np.ndarray,
    zone_traces: np.ndarray, control_frames: int, score_start: int,
    threshold_method: str, *, measure: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    traces = {"global": global_trace, "core": core_trace, "ring": ring_trace}
    sections = {name: _pre_post(trace, control_frames, measure)
        for name, trace in traces.items()}
    pre_key, post_key = f"pre_{measure}", f"post_{measure}"
    pre_enrichment = _safe_ratio(
        sections["ring"][pre_key], sections["global"][pre_key])
    post_enrichment = _safe_ratio(
        sections["ring"][post_key], sections["global"][post_key])
    sections["ring_enrichment"] = {
        "pre_ratio_to_global": pre_enrichment,
        "post_ratio_to_global": post_enrichment,
        "difference": _optional_difference(post_enrichment, pre_enrichment),
        "pre_excess": sections["ring"][pre_key] - sections["global"][pre_key],
        "post_excess": sections["ring"][post_key] - sections["global"][post_key],
    }
    return sections, _zone_activation(
        zone_traces, control_frames, score_start, threshold_method)

def _zone_activation(
    traces: np.ndarray, control_frames: int, score_start: int,
    threshold_method: str,
) -> dict[str, Any]:
    rows, activated = [], 0
    for zone_id in range(traces.shape[1]):
        trace = traces[:, zone_id]
        control = trace[:control_frames]
        threshold = (float(np.quantile(control, 0.99))
            if threshold_method == "p99"
            else float(control.mean() + 3.0 * control.std()))
        post = trace[control_frames:]
        hits = np.flatnonzero((post[:-1] > threshold) & (post[1:] > threshold))
        onset_index = None if hits.size == 0 else score_start + int(hits[0])
        activated += onset_index is not None
        rows.append({
            "zone_id": zone_id, "control_threshold": threshold,
            "onset_source_index": onset_index,
            "onset_ui_frame": None if onset_index is None else onset_index + 1,
        })
    return {
        "threshold_method": threshold_method, "required_consecutive_frames": 2,
        "activated_zone_count": activated, "zones": rows,
    }

def _pre_post(
    trace: np.ndarray, control_frames: int, measure: str,
) -> dict[str, float | None]:
    pre = float(trace[:control_frames].mean())
    post = float(trace[control_frames:].mean())
    return {f"pre_{measure}": pre, f"post_{measure}": post,
        "difference": post - pre, "ratio": _safe_ratio(post, pre)}

def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0.0 else float(numerator / denominator)


def _optional_difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _histogram_percentiles(
    histogram: np.ndarray, quantiles: tuple[float, ...]
) -> tuple[float, ...]:
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    count = int(cumulative[-1])
    values = []
    for quantile in quantiles:
        rank = (quantile / 100.0) * (count - 1)
        lower_rank, upper_rank = int(np.floor(rank)), int(np.ceil(rank))
        lower = int(np.searchsorted(cumulative, lower_rank, side="right"))
        upper = int(np.searchsorted(cumulative, upper_rank, side="right"))
        values.append(float(lower + (upper - lower) * (rank - lower_rank)))
    return tuple(values)


def _range_summary(start: int, stop: int) -> dict[str, int]:
    return {
        "source_start_index": start, "source_stop_index_exclusive": stop,
        "ui_start_frame": start + 1, "ui_end_frame_inclusive": stop,
        "frame_count": stop - start,
    }


def _source_label(source: FrameSource) -> str:
    if isinstance(source, VideoStore):
        return source.source_path or "<in-memory VideoStore>"
    return str(Path(source))


def _validate_inputs(
    control_start: int, control_stop: int, score_start: int, score_stop: int,
    chunk_frames: int, sample_limit: int, threshold_method: str,
) -> None:
    integers = {
        "control_start": control_start, "control_stop": control_stop,
        "score_start": score_start, "score_stop": score_stop,
        "chunk_frames": chunk_frames, "normalization_sample_limit": sample_limit,
    }
    for name, value in integers.items():
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if control_start < 0 or control_stop <= control_start:
        raise ValueError("control range must be non-empty and non-negative")
    if score_start < control_stop or score_stop <= score_start:
        raise ValueError("score range must be non-empty and start after control")
    if not 1 <= chunk_frames <= 128:
        raise ValueError("chunk_frames must be between 1 and 128")
    if sample_limit <= 0:
        raise ValueError("normalization_sample_limit must be positive")
    if threshold_method not in {"p99", "mean_plus_3sd"}:
        raise ValueError("zone_threshold_method must be 'p99' or 'mean_plus_3sd'")
