"""Label-free component evidence and explicit unresolved decisions."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kurtosis


def qualify_temporal_components(
    spatial_maps: np.ndarray,
    temporal_sources: np.ndarray,
    *,
    spatial_shape: tuple[int, int],
    minimum_score: float = 0.55,
    minimum_margin: float = 0.05,
) -> dict[str, Any]:
    """Rank neural-like components without labels and allow `unresolved`.

    This is an intentionally transparent evidence gate, not a learned neuron
    classifier. It combines compact spatial support, temporal persistence,
    non-Gaussianity, and robust burst evidence. Component signs are oriented
    toward positive skew only for reporting.
    """
    maps = np.asarray(spatial_maps, dtype=np.float64)
    traces = np.asarray(temporal_sources, dtype=np.float64)
    pixel_count = int(np.prod(spatial_shape))
    if (
        maps.ndim != 2
        or traces.ndim != 2
        or maps.shape != (pixel_count, traces.shape[0])
        or traces.shape[1] < 16
        or not np.isfinite(maps).all()
        or not np.isfinite(traces).all()
    ):
        raise ValueError("spatial maps and temporal sources are incompatible")
    if not 0 <= minimum_margin <= 1 or not 0 <= minimum_score <= 1:
        raise ValueError("qualification thresholds must be in [0,1]")
    rows = []
    for component in range(traces.shape[0]):
        spatial = maps[:, component]
        temporal = traces[component]
        sign = 1.0
        centered = temporal - np.median(temporal)
        positive_tail = float(np.quantile(centered, 0.99))
        negative_tail = abs(float(np.quantile(centered, 0.01)))
        if negative_tail > positive_tail:
            sign = -1.0
            spatial = -spatial
            temporal = -temporal
            centered = -centered
        energy = spatial * spatial
        effective_pixels = float(
            energy.sum() ** 2 / max(float(np.sum(energy * energy)), np.finfo(float).eps)
        )
        effective_fraction = effective_pixels / pixel_count
        compact_score = float(np.clip(1.0 - effective_fraction / 0.35, 0.0, 1.0))
        left = temporal[:-1] - temporal[:-1].mean()
        right = temporal[1:] - temporal[1:].mean()
        autocorrelation = float(
            (left @ right)
            / max(float(np.linalg.norm(left) * np.linalg.norm(right)), np.finfo(float).eps)
        )
        persistence_score = float(np.clip((autocorrelation - 0.05) / 0.90, 0.0, 1.0))
        excess_kurtosis = float(kurtosis(temporal, fisher=True, bias=False))
        non_gaussian_score = float(np.clip(abs(excess_kurtosis) / 4.0, 0.0, 1.0))
        median = float(np.median(temporal))
        mad = 1.4826 * float(np.median(np.abs(temporal - median)))
        robust_peak_z = float((np.max(temporal) - median) / max(mad, 1e-8))
        burst_score = float(np.clip((robust_peak_z - 2.0) / 6.0, 0.0, 1.0))
        score = float(
            0.35 * compact_score
            + 0.20 * persistence_score
            + 0.20 * non_gaussian_score
            + 0.25 * burst_score
        )
        rows.append({
            "component": component,
            "orientation_sign": int(sign),
            "effective_spatial_support_fraction": effective_fraction,
            "compact_score": compact_score,
            "lag1_autocorrelation": autocorrelation,
            "persistence_score": persistence_score,
            "excess_kurtosis": excess_kurtosis,
            "non_gaussian_score": non_gaussian_score,
            "robust_peak_z": robust_peak_z,
            "burst_score": burst_score,
            "neural_evidence_score": score,
        })
    ordered = sorted(rows, key=lambda row: row["neural_evidence_score"], reverse=True)
    top_score = float(ordered[0]["neural_evidence_score"])
    second_score = float(ordered[1]["neural_evidence_score"]) if len(ordered) > 1 else 0.0
    margin = top_score - second_score
    resolved = bool(top_score >= minimum_score and margin >= minimum_margin)
    return {
        "status": "resolved" if resolved else "unresolved",
        "selected_component": int(ordered[0]["component"]) if resolved else None,
        "top_score": top_score,
        "second_score": second_score,
        "score_margin": margin,
        "minimum_score": float(minimum_score),
        "minimum_margin": float(minimum_margin),
        "selection_uses_labels": False,
        "components": rows,
    }
