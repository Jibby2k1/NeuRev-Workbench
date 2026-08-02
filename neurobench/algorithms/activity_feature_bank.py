"""Causal and diagnostic feature channels for calcium-activity decisions."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


def _video(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError("values must be a finite non-empty TYX array")
    return result


def quiet_robust_z(
    values: np.ndarray,
    quiet_count: int,
    *,
    floor_percentile: float = 10.0,
) -> np.ndarray:
    """Per-pixel robust standardization frozen on a leading quiet interval."""
    video = _video(values)
    quiet = int(quiet_count)
    if not 3 <= quiet <= len(video):
        raise ValueError("quiet_count must define a leading interval")
    center = np.median(video[:quiet], axis=0).astype(np.float32)
    mad = (
        1.4826 * np.median(np.abs(video[:quiet] - center[None]), axis=0)
    ).astype(np.float32)
    positive = mad[mad > 0]
    floor = (
        float(np.percentile(positive, float(floor_percentile)))
        if positive.size
        else 1.0
    )
    return ((video - center[None]) / np.maximum(mad, max(floor, 1e-6))).astype(
        np.float32
    )


def unit_positive(values: np.ndarray, clip_z: float) -> np.ndarray:
    """Map positive standardized evidence to a bounded unit feature."""
    if float(clip_z) <= 0:
        raise ValueError("clip_z must be positive")
    return np.clip(np.maximum(values, 0) / float(clip_z), 0, 1).astype(
        np.float32
    )


def bounded_square(values: np.ndarray, tau: float) -> np.ndarray:
    """Bounded even energy z²/(z²+tau²)."""
    array = np.asarray(values, dtype=np.float32)
    scale = float(tau)
    if scale <= 0:
        raise ValueError("tau must be positive")
    energy = np.square(array, dtype=np.float32)
    return (energy / (energy + scale * scale)).astype(np.float32)


def derivative_feature_iterator(
    standardized_carrier: np.ndarray,
    *,
    quiet_count: int,
    spatial_sigma_px: float,
    lags: Sequence[int],
    clip_z: float,
    power: float,
    energy_tau_z: float,
    huber_delta_z: float,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any]]]:
    """Yield derivative polarity and comparable even nonlinearities."""
    from scipy.ndimage import gaussian_filter

    carrier = _video(standardized_carrier)
    smooth = gaussian_filter(
        carrier,
        sigma=(0, float(spatial_sigma_px), float(spatial_sigma_px)),
        mode="reflect",
    ).astype(np.float32)
    selected_lags = tuple(int(value) for value in lags)
    if not selected_lags or any(value < 1 or value >= quiet_count for value in selected_lags):
        raise ValueError("derivative lags must be positive and inside quiet")
    for lag in selected_lags:
        difference = np.zeros_like(smooth)
        difference[lag:] = smooth[lag:] - smooth[:-lag]
        standardized = quiet_robust_z(
            difference[lag:],
            quiet_count - lag,
        )
        derivative_z = np.zeros_like(difference)
        derivative_z[lag:] = standardized
        del standardized, difference
        common = {
            "lag_frames": lag,
            "spatial_sigma_px": float(spatial_sigma_px),
            "quiet_standardized": True,
        }
        if lag == selected_lags[0]:
            yield (
                f"derivative_positive_lag{lag}",
                unit_positive(derivative_z, clip_z),
                {**common, "transform": "positive_part"},
            )
            yield (
                f"derivative_negative_lag{lag}",
                unit_positive(-derivative_z, clip_z),
                {**common, "transform": "negative_part"},
            )
            magnitude = np.abs(derivative_z)
            yield (
                f"derivative_absolute_lag{lag}",
                np.clip(magnitude / float(clip_z), 0, 1).astype(np.float32),
                {**common, "transform": "absolute"},
            )
            yield (
                f"derivative_power{str(power).replace('.', 'p')}_lag{lag}",
                np.power(
                    np.clip(magnitude / float(clip_z), 0, 1),
                    float(power),
                ).astype(np.float32),
                {**common, "transform": "absolute_power", "power": float(power)},
            )
            yield (
                f"derivative_log_square_lag{lag}",
                (
                    np.log1p(np.square(derivative_z / float(energy_tau_z)))
                    / np.log1p((float(clip_z) / float(energy_tau_z)) ** 2)
                ).clip(0, 1).astype(np.float32),
                {
                    **common,
                    "transform": "log_square",
                    "tau_z": float(energy_tau_z),
                },
            )
            delta = float(huber_delta_z)
            if delta <= 0:
                raise ValueError("huber_delta_z must be positive")
            huber = np.where(
                magnitude <= delta,
                0.5 * magnitude * magnitude,
                delta * (magnitude - 0.5 * delta),
            )
            maximum = delta * (float(clip_z) - 0.5 * delta)
            yield (
                f"derivative_huber_lag{lag}",
                np.clip(huber / max(maximum, 1e-6), 0, 1).astype(np.float32),
                {**common, "transform": "huber_energy", "delta_z": delta},
            )
        yield (
            f"derivative_square_lag{lag}",
            bounded_square(derivative_z, energy_tau_z),
            {**common, "transform": "bounded_square", "tau_z": float(energy_tau_z)},
        )
        del derivative_z


def cross_scale_consensus_score(
    values: np.ndarray,
    *,
    spatial_scales_px: Sequence[float],
    agreement_power: float,
    evidence_threshold_z: float,
) -> np.ndarray:
    """Return the unit consensus gain before it modulates a carrier."""
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit

    video = _video(values)
    scales = tuple(float(value) for value in spatial_scales_px)
    if len(scales) < 2 or any(value <= 0 for value in scales):
        raise ValueError("at least two positive spatial scales are required")
    estimates = np.stack(
        [
            gaussian_filter(video, sigma=(0, scale, scale), mode="reflect")
            for scale in scales
        ]
    )
    agreement = np.power(
        np.abs(np.mean(np.sign(estimates), axis=0)),
        float(agreement_power),
    )
    evidence = np.median(np.abs(estimates), axis=0)
    strength = expit(
        (evidence - float(evidence_threshold_z))
        / max(0.25 * float(evidence_threshold_z), 0.1)
    )
    return np.clip(agreement * strength, 0, 1).astype(np.float32)


def persistence_features(
    standardized_carrier: np.ndarray,
    *,
    frame_period_ms: float,
    persistence_half_life_seconds: float,
    dynamic_half_life_seconds: float,
    energy_tau_z: float,
) -> dict[str, np.ndarray]:
    """Causal persistence, dynamic activity gate, and persistent-artifact score."""
    carrier = _video(standardized_carrier)
    period = float(frame_period_ms) / 1000.0
    persistence_decay = 0.5 ** (
        period / float(persistence_half_life_seconds)
    )
    dynamic_decay = 0.5 ** (period / float(dynamic_half_life_seconds))
    persistent = np.empty_like(carrier)
    dynamic = np.empty_like(carrier)
    persistent[0] = np.abs(carrier[0])
    dynamic[0] = 0
    for index in range(1, len(carrier)):
        persistent[index] = (
            persistence_decay * persistent[index - 1]
            + (1.0 - persistence_decay) * np.abs(carrier[index])
        )
        derivative_energy = bounded_square(
            carrier[index] - carrier[index - 1], energy_tau_z
        )
        dynamic[index] = (
            dynamic_decay * dynamic[index - 1]
            + (1.0 - dynamic_decay) * derivative_energy
        )
    persistent_unit = persistent / (persistent + float(energy_tau_z))
    activity_gate = dynamic / np.maximum(dynamic + persistent_unit, 1e-6)
    artifact = persistent_unit * (1.0 - dynamic)
    return {
        "persistence_activity_gate": np.clip(activity_gate, 0, 1).astype(
            np.float32
        ),
        "persistent_artifact_score": np.clip(artifact, 0, 1).astype(np.float32),
    }


def morphology_feature_iterator(
    standardized_carrier: np.ndarray,
    *,
    quiet_count: int,
    center_sigma_px: float,
    ring_sigma_px: float,
    crowd_sigma_px: float,
    clip_z: float,
) -> Iterator[tuple[str, np.ndarray, dict[str, Any]]]:
    """Yield four distinct center/annulus by isolated/crowded expert scores."""
    from scipy.ndimage import gaussian_filter

    carrier = _video(standardized_carrier)
    positive = np.maximum(carrier, 0)
    center_inner = gaussian_filter(
        positive,
        sigma=(0, float(center_sigma_px), float(center_sigma_px)),
        mode="reflect",
    )
    center_outer = gaussian_filter(
        positive,
        sigma=(0, float(ring_sigma_px), float(ring_sigma_px)),
        mode="reflect",
    )
    crowd_raw = gaussian_filter(
        positive,
        sigma=(0, float(crowd_sigma_px), float(crowd_sigma_px)),
        mode="reflect",
    )
    center_raw = np.maximum(center_inner - center_outer, 0)
    membrane_raw = np.maximum(center_outer - 0.55 * center_inner, 0)
    center = unit_positive(quiet_robust_z(center_raw, quiet_count), clip_z)
    membrane = unit_positive(quiet_robust_z(membrane_raw, quiet_count), clip_z)
    crowd = unit_positive(quiet_robust_z(crowd_raw, quiet_count), clip_z)
    definitions = (
        ("morphology_center_isolated", center * (1.0 - crowd), "center", "isolated"),
        (
            "morphology_membrane_isolated",
            membrane * (1.0 - crowd),
            "membrane",
            "isolated",
        ),
        ("morphology_center_crowded", center * crowd, "center", "crowded"),
        (
            "morphology_membrane_crowded",
            membrane * crowd,
            "membrane",
            "crowded",
        ),
    )
    for feature_id, values, geometry, context in definitions:
        yield feature_id, values.astype(np.float32), {
            "geometry": geometry,
            "context": context,
            "center_sigma_px": float(center_sigma_px),
            "ring_sigma_px": float(ring_sigma_px),
            "crowd_sigma_px": float(crowd_sigma_px),
        }


def localized_feature_trace_metrics(
    feature: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    """Compare a nonnegative feature with each injected morphology trace."""
    values = _video(feature)
    signal = _video(truth)
    if values.shape != signal.shape:
        raise ValueError("feature and truth must align")
    height, width = values.shape[1:]
    quadrants = (
        (slice(0, height // 2), slice(0, width // 2)),
        (slice(0, height // 2), slice(width // 2, width)),
        (slice(height // 2, height), slice(0, width // 2)),
        (slice(height // 2, height), slice(width // 2, width)),
    )
    rows = []
    for index, (ys, xs) in enumerate(quadrants):
        local_truth = signal[:, ys, xs]
        template = np.max(local_truth, axis=0)
        template /= max(float(np.sum(template * template)), 1e-12)
        truth_trace = np.einsum("tyx,yx->t", local_truth, template)
        feature_trace = np.einsum("tyx,yx->t", values[:, ys, xs], template)
        onset = int(np.flatnonzero(truth_trace > 1e-12)[0])
        feature_trace -= np.median(feature_trace[: max(8, onset)])
        correlation = (
            float(np.corrcoef(truth_trace, feature_trace)[0, 1])
            if np.std(feature_trace) > 0
            else 0.0
        )
        rows.append(
            {
                "morphology_index": index,
                "correlation": correlation,
                "peak_frame_error": abs(
                    int(np.argmax(feature_trace)) - int(np.argmax(truth_trace))
                ),
            }
        )
    return {
        "synthetic_feature_correlation": float(
            np.median([row["correlation"] for row in rows])
        ),
        "synthetic_feature_peak_frame_error": float(
            np.median([row["peak_frame_error"] for row in rows])
        ),
        "synthetic_morphology_rows": rows,
    }
