"""Claim-bounded integrity, SNR, coherence, response, and stability diagnostics."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import coherence


def reconstruction_integrity(
    observations: np.ndarray, sources: np.ndarray, mixing: np.ndarray,
    observation_mean: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(observations, dtype=np.float64)
    reconstructed = np.asarray(mixing) @ np.asarray(sources) + np.asarray(observation_mean)[:, None]
    residual = values - reconstructed
    centered = values - np.asarray(observation_mean)[:, None]
    denominator = max(float(np.sum(centered**2)), np.finfo(float).eps)
    flat_values, flat_reconstructed = values.ravel(), reconstructed.ravel()
    correlation = float(np.corrcoef(flat_values, flat_reconstructed)[0, 1])
    return {
        "reconstruction_nmse_centered": float(np.sum(residual**2) / denominator),
        "raw_reconstruction_correlation": correlation,
        "absolute_amplitude_retention": float(
            np.median(np.abs(reconstructed))
            / max(float(np.median(np.abs(values))), np.finfo(float).eps)
        ),
        "absolute_area_retention": float(
            np.sum(np.abs(reconstructed))
            / max(float(np.sum(np.abs(values))), np.finfo(float).eps)
        ),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "finite_fraction": float(np.mean(np.isfinite(reconstructed))),
    }


def approximate_source_snr(
    sources: np.ndarray, sample_times: np.ndarray, quiet_count: int,
) -> dict[str, Any]:
    """Robust event/quiet contrast; explicitly not truth-known SNR."""
    values = np.asarray(sources, dtype=np.float64)
    times = np.asarray(sample_times)
    quiet = values[:, times < quiet_count]
    event = values[:, times >= quiet_count]
    if quiet.shape[1] < 32 or event.shape[1] < 32:
        raise ValueError("approximate SNR requires at least 32 quiet and event samples")
    center = np.median(quiet, axis=1)
    quiet_scale = np.maximum(
        1.4826 * np.median(np.abs(quiet - center[:, None]), axis=1),
        np.finfo(float).eps,
    )
    event_scale = np.median(np.abs(event - center[:, None]), axis=1)
    ratio = event_scale / quiet_scale
    rows = [{
        "component": int(index), "event_to_quiet_robust_ratio": float(value),
        "event_to_quiet_robust_db": float(20.0 * np.log10(max(value, np.finfo(float).eps))),
    } for index, value in enumerate(ratio)]
    return {
        "semantics": "approximate_event_to_quiet_robust_contrast_not_true_snr",
        "components": rows,
        "median_ratio": float(np.median(ratio)),
        "median_db": float(np.median([row["event_to_quiet_robust_db"] for row in rows])),
    }


def ordered_trace_coherence(
    raw_trace: np.ndarray, output_trace: np.ndarray, *, sample_period_seconds: float,
    maximum_lag: int = 12,
) -> dict[str, float]:
    raw = np.asarray(raw_trace, dtype=np.float64).ravel()
    output = np.asarray(output_trace, dtype=np.float64).ravel()
    if raw.shape != output.shape or len(raw) < 16 or not np.isfinite(raw).all() or not np.isfinite(output).all():
        raise ValueError("coherence traces must be aligned, finite, and length >=16")
    raw = raw - raw.mean(); output = output - output.mean()
    norm = max(float(np.linalg.norm(raw) * np.linalg.norm(output)), np.finfo(float).eps)
    zero = float(np.dot(raw, output) / norm)
    lags = np.arange(-maximum_lag, maximum_lag + 1)
    lag_values = []
    for lag in lags:
        if lag < 0: a, b = raw[-lag:], output[:lag]
        elif lag > 0: a, b = raw[:-lag], output[lag:]
        else: a, b = raw, output
        denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), np.finfo(float).eps)
        lag_values.append(float(np.dot(a, b) / denominator))
    best_index = int(np.argmax(np.abs(lag_values)))
    fs = 1.0 / sample_period_seconds
    frequencies, spectral = coherence(
        raw, output, fs=fs, nperseg=min(64, len(raw)), detrend="linear"
    )
    nyquist = fs / 2.0
    def band(low: float, high: float) -> float:
        selected = (frequencies >= low * nyquist) & (frequencies < high * nyquist)
        return float(np.mean(spectral[selected])) if np.any(selected) else float("nan")
    return {
        "zero_lag_correlation": zero,
        "maximum_absolute_lagged_correlation": float(lag_values[best_index]),
        "lag_at_maximum_frames": int(lags[best_index]),
        "mean_magnitude_squared_coherence": float(np.mean(spectral)),
        "low_band_coherence": band(0.0, 1 / 3),
        "mid_band_coherence": band(1 / 3, 2 / 3),
        "high_band_coherence": band(2 / 3, 1.000001),
    }


def _row_normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(float).eps)


def seed_stability(demixings: Iterable[np.ndarray]) -> dict[str, Any]:
    matrices = [np.asarray(item, dtype=np.float64) for item in demixings]
    if len(matrices) < 2:
        raise ValueError("stability requires at least two demixing fits")
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("stability demixings must have matching shapes")
    rows = []
    for (left_index, left), (right_index, right) in combinations(enumerate(matrices), 2):
        left_n, right_n = _row_normalized(left), _row_normalized(right)
        similarity = np.abs(left_n @ right_n.T)
        li, ri = linear_sum_assignment(-similarity)
        aligned = similarity[li, ri]
        ql = np.linalg.qr(left.T)[0]; qr = np.linalg.qr(right.T)[0]
        singular = np.linalg.svd(ql.T @ qr, compute_uv=False)
        rows.append({
            "left_fit_index": left_index, "right_fit_index": right_index,
            "mean_aligned_component_cosine": float(np.mean(aligned)),
            "worst_aligned_component_cosine": float(np.min(aligned)),
            "mean_principal_angle_degrees": float(np.mean(np.degrees(np.arccos(np.clip(singular, -1, 1))))),
            "worst_principal_angle_degrees": float(np.max(np.degrees(np.arccos(np.clip(singular, -1, 1))))),
        })
    return {
        "pair_count": len(rows), "pairs": rows,
        "mean_aligned_component_cosine": float(np.mean([row["mean_aligned_component_cosine"] for row in rows])),
        "worst_aligned_component_cosine": float(np.min([row["worst_aligned_component_cosine"] for row in rows])),
        "mean_principal_angle_degrees": float(np.mean([row["mean_principal_angle_degrees"] for row in rows])),
        "individual_component_interpretation_allowed": bool(
            min(row["worst_aligned_component_cosine"] for row in rows) >= 0.8
        ),
    }
