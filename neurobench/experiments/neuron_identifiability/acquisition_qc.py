"""Bounded descriptive acquisition-physics diagnostics."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from neurobench.algorithms.scientific_feature_audit import fit_poisson_gaussian_noise

from .contracts import ObservationRecord


def quiet_indices(frame_count: int, records: Iterable[ObservationRecord], margin: int = 30) -> np.ndarray:
    allowed = np.ones(frame_count, dtype=bool)
    for record in records:
        allowed[max(0, record.start_zero-margin):min(frame_count, record.stop_zero_exclusive+margin)] = False
    return np.flatnonzero(allowed)


def _serializable_noise_fit(values: np.ndarray) -> dict[str, Any]:
    result = fit_poisson_gaussian_noise(values, intensity_bins=12, saturation_value=4095)
    return {key: value for key, value in result.items() if key not in {"quiet_mean_map", "pair_difference_variance_map"}}


def acquisition_diagnostics(video: np.ndarray, records: list[ObservationRecord], *, spatial_stride: int = 4, maximum_quiet_frames: int = 240) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    quiet = quiet_indices(len(video), records)
    if len(quiet) < 40:
        raise ValueError("insufficient quiet frames for acquisition QC")
    chosen = quiet[np.linspace(0, len(quiet)-1, min(maximum_quiet_frames, len(quiet)), dtype=int)]
    sample = np.asarray(video[chosen, ::spatial_stride, ::spatial_stride], dtype=np.float32)
    width = sample.shape[2]; boundary_sample = 286 // spatial_stride
    x_values = np.asarray([r.x_px for r in records]); y_values = np.asarray([r.y_px for r in records])
    ax0, ax1 = max(0, int(y_values.min())//spatial_stride-16), min(sample.shape[1], int(y_values.max())//spatial_stride+17)
    ay0, ay1 = max(0, int(x_values.min())//spatial_stride-16), min(width, int(x_values.max())//spatial_stride+17)
    fits = {
        "full": _serializable_noise_fit(sample),
        "left_field": _serializable_noise_fit(sample[:, :, :max(boundary_sample, 2)]),
        "right_field": _serializable_noise_fit(sample[:, :, min(boundary_sample, width-2):]),
        "annotated_region": _serializable_noise_fit(sample[:, ax0:ax1, ay0:ay1]),
    }
    global_trace = np.asarray(video[:, ::8, ::8], dtype=np.float64).mean(axis=(1,2))
    centered = global_trace - np.mean(global_trace)
    autocorrelation = np.correlate(centered, centered, mode="full")[len(centered)-1:len(centered)+20]
    autocorrelation /= max(autocorrelation[0], np.finfo(float).eps)
    frequencies = np.fft.rfftfreq(len(centered), d=.020)
    psd = np.abs(np.fft.rfft(centered))**2 / len(centered)
    mean_map = np.asarray(video[chosen[::max(1,len(chosen)//60)]], dtype=np.float64).mean(axis=0)
    variance_map = np.asarray(video[chosen[::max(1,len(chosen)//60)]], dtype=np.float64).var(axis=0)
    column_jump = np.mean(np.abs(np.diff(mean_map, axis=1)), axis=0)
    strongest = np.argsort(column_jump)[-10:][::-1] + 1
    row_jump = np.mean(np.abs(np.diff(mean_map, axis=0)), axis=1)
    frame_difference = np.mean(np.abs(np.diff(sample, axis=0)), axis=(1,2))
    median_fd = np.median(frame_difference); mad_fd = np.median(np.abs(frame_difference-median_fd))*1.4826
    change_points = chosen[1:][frame_difference > median_fd + 6*max(mad_fd,np.finfo(float).eps)]
    metrics = {
        "quiet_frame_count": int(len(quiet)), "noise_models": fits,
        "global_trace": {"mean": float(np.mean(global_trace)), "drift_native_per_frame": float(np.polyfit(np.arange(len(global_trace)),global_trace,1)[0]), "autocorrelation_lags_0_20": autocorrelation.tolist()},
        "psd": {"dominant_nonzero_hz": float(frequencies[1+np.argmax(psd[1:])]), "low_frequency_fraction_below_0_1hz": float(np.sum(psd[frequencies<.1])/np.sum(psd))},
        "spatial": {"declared_boundary_x": 286, "declared_boundary_jump": float(column_jump[285]), "strongest_column_boundaries": strongest.tolist(), "strongest_row_boundaries": (np.argsort(row_jump)[-10:][::-1]+1).tolist()},
        "temporal": {"frame_difference_median": float(median_fd), "frame_difference_mad": float(mad_fd), "robust_change_point_frames_zero": change_points.tolist()},
        "saturation_fraction": float(np.mean(sample >= 4095)),
        "sampling": {"spatial_stride": spatial_stride, "quiet_frames_used": int(len(chosen))},
    }
    arrays = {"global_trace":global_trace, "psd_frequency_hz":frequencies, "psd_power":psd, "mean_map":mean_map, "variance_map":variance_map, "column_jump":column_jump, "row_jump":row_jump}
    return metrics, arrays
