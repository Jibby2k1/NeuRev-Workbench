"""Truth-aware metrics for generated and semi-synthetic source separation."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    left_norm = np.linalg.norm(left_centered, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right_centered, axis=1, keepdims=True)
    return (left_centered @ right_centered.T) / np.maximum(
        left_norm * right_norm.T, np.finfo(float).eps
    )


def aligned_source_metrics(
    true_sources: np.ndarray,
    estimated_sources: np.ndarray,
) -> dict[str, Any]:
    """Permutation/sign/scale-align sources and report recovery and cross-talk."""
    truth = _matrix(true_sources, "true_sources")
    estimate = _matrix(estimated_sources, "estimated_sources")
    if truth.shape[1] != estimate.shape[1]:
        raise ValueError("true and estimated sources must share sample count")
    correlations = _correlation_matrix(truth, estimate)
    truth_indices, estimate_indices = linear_sum_assignment(-np.abs(correlations))
    matched_rows = []
    aligned = np.zeros((len(truth_indices), truth.shape[1]), dtype=np.float64)
    for output_index, (truth_index, estimate_index) in enumerate(
        zip(truth_indices, estimate_indices)
    ):
        target = truth[truth_index]
        candidate = estimate[estimate_index]
        centered_candidate = candidate - candidate.mean()
        centered_target = target - target.mean()
        denominator = max(float(centered_candidate @ centered_candidate), np.finfo(float).eps)
        scale = float((centered_candidate @ centered_target) / denominator)
        fitted = scale * centered_candidate + float(target.mean())
        aligned[output_index] = fitted
        nmse = float(
            np.sum((target - fitted) ** 2)
            / max(float(np.sum(centered_target**2)), np.finfo(float).eps)
        )
        matched_rows.append({
            "true_source": int(truth_index),
            "estimated_source": int(estimate_index),
            "absolute_correlation": float(abs(correlations[truth_index, estimate_index])),
            "signed_correlation": float(correlations[truth_index, estimate_index]),
            "alignment_scale": scale,
            "nmse": nmse,
        })
    matched_estimates = {int(index) for index in estimate_indices}
    unmatched_estimates = sorted(set(range(estimate.shape[0])) - matched_estimates)
    off_diagonal = []
    for truth_index, estimate_index in zip(truth_indices, estimate_indices):
        off_diagonal.extend(
            abs(float(correlations[other_truth, estimate_index]))
            for other_truth in range(truth.shape[0])
            if other_truth != truth_index
        )
    return {
        "true_source_count": int(truth.shape[0]),
        "estimated_source_count": int(estimate.shape[0]),
        "matched_source_count": int(len(matched_rows)),
        "mean_absolute_correlation": float(np.mean([
            row["absolute_correlation"] for row in matched_rows
        ])),
        "worst_absolute_correlation": float(np.min([
            row["absolute_correlation"] for row in matched_rows
        ])),
        "mean_aligned_nmse": float(np.mean([row["nmse"] for row in matched_rows])),
        "worst_aligned_nmse": float(np.max([row["nmse"] for row in matched_rows])),
        "mean_absolute_crosstalk": float(np.mean(off_diagonal)) if off_diagonal else 0.0,
        "worst_absolute_crosstalk": float(np.max(off_diagonal)) if off_diagonal else 0.0,
        "unmatched_estimated_sources": unmatched_estimates,
        "matches": matched_rows,
        "correlation_matrix": correlations.tolist(),
    }


def trace_fidelity_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    threshold_fraction: float = 0.1,
) -> dict[str, Any]:
    """Measure amplitude, area, onset, peak timing, and waveform fidelity."""
    truth = np.asarray(reference, dtype=np.float64).ravel()
    fitted = np.asarray(estimate, dtype=np.float64).ravel()
    if (
        truth.shape != fitted.shape
        or len(truth) < 3
        or not np.isfinite(truth).all()
        or not np.isfinite(fitted).all()
        or not 0 < threshold_fraction < 1
    ):
        raise ValueError("trace inputs or threshold fraction are invalid")
    truth_positive = np.maximum(truth, 0)
    fitted_positive = np.maximum(fitted, 0)
    truth_peak = float(np.max(truth_positive))
    fitted_peak = float(np.max(fitted_positive))
    truth_area = float(np.sum(truth_positive))
    fitted_area = float(np.sum(fitted_positive))

    def onset(values: np.ndarray, peak: float) -> int | None:
        indices = np.flatnonzero(values >= threshold_fraction * peak)
        return int(indices[0]) if peak > 0 and len(indices) else None

    truth_onset = onset(truth_positive, truth_peak)
    fitted_onset = onset(fitted_positive, fitted_peak)
    truth_peak_frame = int(np.argmax(truth_positive))
    fitted_peak_frame = int(np.argmax(fitted_positive))
    if np.std(truth) == 0 or np.std(fitted) == 0:
        correlation = 1.0 if np.allclose(truth, fitted) else 0.0
    else:
        correlation = float(np.corrcoef(truth, fitted)[0, 1])
    return {
        "peak_retention": float(fitted_peak / max(truth_peak, np.finfo(float).eps)),
        "area_retention": float(fitted_area / max(truth_area, np.finfo(float).eps)),
        "waveform_correlation": correlation,
        "reference_onset_frame": truth_onset,
        "estimated_onset_frame": fitted_onset,
        "onset_error_frames": (
            None if truth_onset is None or fitted_onset is None
            else int(fitted_onset - truth_onset)
        ),
        "reference_peak_frame": truth_peak_frame,
        "estimated_peak_frame": fitted_peak_frame,
        "peak_error_frames": int(fitted_peak_frame - truth_peak_frame),
    }


def footprint_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    threshold_fraction: float = 0.25,
) -> dict[str, float]:
    """Measure thresholded footprint IoU and intensity-weighted centroid error."""
    truth = np.asarray(reference, dtype=np.float64)
    fitted = np.asarray(estimate, dtype=np.float64)
    if (
        truth.ndim != 2
        or truth.shape != fitted.shape
        or not truth.size
        or not np.isfinite(truth).all()
        or not np.isfinite(fitted).all()
        or not 0 < threshold_fraction < 1
    ):
        raise ValueError("footprints must be aligned finite 2D arrays")
    truth_positive = np.maximum(truth, 0)
    fitted_positive = np.maximum(fitted, 0)
    truth_mask = truth_positive >= threshold_fraction * max(float(truth_positive.max()), np.finfo(float).eps)
    fitted_mask = fitted_positive >= threshold_fraction * max(float(fitted_positive.max()), np.finfo(float).eps)
    union = int(np.count_nonzero(truth_mask | fitted_mask))
    intersection = int(np.count_nonzero(truth_mask & fitted_mask))

    def centroid(values: np.ndarray) -> np.ndarray:
        rows, columns = np.indices(values.shape)
        weight = max(float(values.sum()), np.finfo(float).eps)
        return np.asarray([
            float((columns * values).sum() / weight),
            float((rows * values).sum() / weight),
        ])

    return {
        "footprint_iou": float(intersection / union) if union else 1.0,
        "centroid_error_px": float(np.linalg.norm(
            centroid(truth_positive) - centroid(fitted_positive)
        )),
    }
