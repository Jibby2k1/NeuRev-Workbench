"""Localized exact-truth metrics for denoisers on the four-shape fixture."""
from __future__ import annotations

import numpy as np


def _finite_tyx(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite non-empty TYX array")
    return result


def localized_synthetic_denoising_metrics(
    estimate: np.ndarray,
    observed: np.ndarray,
    truth: np.ndarray,
) -> dict[str, object]:
    """Evaluate each injected morphology in its own spatial quadrant.

    Full-frame summed traces are dominated by real quiet residual noise and can
    report false timing shifts even for purely framewise denoisers.  Matched
    spatial projections provide a meaningful signal trace for each injected
    source while the global error still measures residual noise.
    """
    output = _finite_tyx(estimate, "estimate")
    noisy = _finite_tyx(observed, "observed")
    signal = _finite_tyx(truth, "truth")
    if output.shape != noisy.shape or output.shape != signal.shape:
        raise ValueError("estimate, observed, and truth must align")
    height, width = output.shape[1:]
    if height % 2 or width % 2:
        raise ValueError("the four-shape fixture must have even spatial dimensions")
    quadrants = (
        (slice(0, height // 2), slice(0, width // 2)),
        (slice(0, height // 2), slice(width // 2, width)),
        (slice(height // 2, height), slice(0, width // 2)),
        (slice(height // 2, height), slice(width // 2, width)),
    )
    rows: list[dict[str, float | int]] = []
    for morphology_index, (ys, xs) in enumerate(quadrants):
        local_truth = signal[:, ys, xs]
        template = np.max(local_truth, axis=0)
        norm = max(float(np.sum(template * template)), np.finfo(float).tiny)
        template = template / norm
        truth_trace = np.einsum("tyx,yx->t", local_truth, template)
        estimate_trace = np.einsum("tyx,yx->t", output[:, ys, xs], template)
        observed_trace = np.einsum("tyx,yx->t", noisy[:, ys, xs], template)
        active = np.flatnonzero(truth_trace > np.finfo(float).eps)
        if not len(active):
            raise ValueError("every fixture quadrant must contain an injected source")
        onset = int(active[0])
        baseline_stop = max(8, onset)
        estimate_trace -= np.median(estimate_trace[:baseline_stop])
        observed_trace -= np.median(observed_trace[:baseline_stop])
        correlation = (
            float(np.corrcoef(truth_trace, estimate_trace)[0, 1])
            if np.std(estimate_trace) > 0
            else 0.0
        )
        input_correlation = (
            float(np.corrcoef(truth_trace, observed_trace)[0, 1])
            if np.std(observed_trace) > 0
            else 0.0
        )
        truth_peak = max(float(np.max(truth_trace)), np.finfo(float).tiny)
        truth_area = max(
            float(np.sum(np.maximum(truth_trace, 0))), np.finfo(float).tiny
        )
        rows.append({
            "morphology_index": morphology_index,
            "onset_frame": onset,
            "correlation": correlation,
            "input_correlation": input_correlation,
            "peak_frame_error": abs(
                int(np.argmax(estimate_trace)) - int(np.argmax(truth_trace))
            ),
            "peak_amplitude_ratio": float(np.max(estimate_trace) / truth_peak),
            "area_ratio": float(
                np.sum(np.maximum(estimate_trace, 0)) / truth_area
            ),
        })
    denominator = max(float(np.mean(signal * signal)), np.finfo(float).tiny)
    output_error = float(np.mean((output - signal) ** 2))
    input_error = float(np.mean((noisy - signal) ** 2))
    correlations = [float(row["correlation"]) for row in rows]
    input_correlations = [float(row["input_correlation"]) for row in rows]
    return {
        "synthetic_nmse": output_error / denominator,
        "synthetic_input_nmse": input_error / denominator,
        "synthetic_noise_attenuation_db": float(
            10.0 * np.log10(max(input_error, np.finfo(float).tiny)
                            / max(output_error, np.finfo(float).tiny))
        ),
        "synthetic_correlation": float(np.median(correlations)),
        "synthetic_input_correlation": float(np.median(input_correlations)),
        "synthetic_correlation_gain": float(
            np.median(correlations) - np.median(input_correlations)
        ),
        "synthetic_peak_frame_error": float(
            np.median([float(row["peak_frame_error"]) for row in rows])
        ),
        "synthetic_peak_amplitude_ratio": float(
            np.median([float(row["peak_amplitude_ratio"]) for row in rows])
        ),
        "synthetic_area_ratio": float(
            np.median([float(row["area_ratio"]) for row in rows])
        ),
        "synthetic_morphology_rows": rows,
    }
