"""Deterministic metrics for sparse-positive spatial detection experiments.

Unmatched candidates are deliberately represented as unknown.  These helpers
never infer exhaustive negatives or call known-label candidate yield precision.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


Peak = tuple[float, int, int]


def temporal_pool(frames: np.ndarray, mode: str = "lme0.25") -> np.ndarray:
    """Pool a non-empty ``T,Y,X`` stack into one score map."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("frames must be a non-empty finite T,Y,X array")
    if mode == "mean":
        return values.mean(axis=0, dtype=np.float64).astype(np.float32)
    if mode == "max":
        return values.max(axis=0).astype(np.float32)
    if mode == "occupancy":
        return values.mean(axis=0, dtype=np.float64).astype(np.float32)
    if not mode.startswith("lme"):
        raise ValueError(f"Unsupported temporal pool: {mode}")
    from scipy.special import logsumexp

    tau = float(mode.removeprefix("lme"))
    if tau <= 0:
        raise ValueError("LME temperature must be positive")
    return (tau * (logsumexp(values / tau, axis=0) - math.log(len(values)))).astype(np.float32)


def extract_local_maxima(
    score: np.ndarray,
    distance: int,
    threshold: float = -np.inf,
    limit: int = 10_000,
    *,
    tie_breaker: np.ndarray | None = None,
) -> list[Peak]:
    """Return spatial maxima sorted by score, optional tie score, then y/x.

    When no tie-breaker is supplied, the legacy NumPy ordering is retained so
    existing Raw Direct candidate order remains byte-for-byte compatible.
    """
    from scipy.ndimage import maximum_filter

    values = np.asarray(score)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("score must be a finite 2D array")
    distance = int(distance)
    limit = int(limit)
    if distance < 1 or limit < 1:
        raise ValueError("distance and limit must be positive")
    keep = (values == maximum_filter(values, size=2 * distance + 1, mode="nearest")) & (values >= threshold)
    keep[:distance] = False
    keep[-distance:] = False
    keep[:, :distance] = False
    keep[:, -distance:] = False
    y, x = np.nonzero(keep)
    scores = values[y, x]
    if tie_breaker is None:
        order = np.argsort(scores)[::-1][:limit]
    else:
        secondary = np.asarray(tie_breaker)
        if secondary.shape != values.shape or not np.isfinite(secondary).all():
            raise ValueError("tie_breaker must be finite and match score")
        order = np.lexsort((x, y, -secondary[y, x], -scores))[:limit]
    return [(float(scores[i]), int(x[i]), int(y[i])) for i in order]


def quiet_calibrated_threshold(
    maps: Sequence[np.ndarray], distance: int, peaks_per_map: float, *, limit: int = 2000
) -> float:
    """Calibrate a threshold using only quiet-map local maxima."""
    if not maps or peaks_per_map <= 0:
        raise ValueError("quiet maps and a positive rate are required")
    ranked = sorted(
        (peak[0] for score in maps for peak in extract_local_maxima(score, distance, limit=limit)),
        reverse=True,
    )
    allowed = max(1, int(round(peaks_per_map * len(maps))))
    if len(ranked) <= allowed:
        raise RuntimeError("Too few quiet peaks for threshold calibration")
    return float(np.nextafter(ranked[allowed], np.inf))


def match_peaks_one_to_one(
    peaks: Sequence[Peak], labels: Sequence[Mapping[str, Any]], radius: float
) -> tuple[list[tuple[int, float, int, int, float]], set[int]]:
    """Greedily match score-ranked peaks to the nearest remaining label."""
    remaining = set(range(len(labels)))
    matches: list[tuple[int, float, int, int, float]] = []
    peak_indices: set[int] = set()
    for peak_index, (score, x, y) in enumerate(peaks):
        choices = [
            (math.hypot(x - float(labels[i]["x_px"]), y - float(labels[i]["y_px"])), i)
            for i in remaining
        ]
        if not choices:
            break
        distance, label_index = min(choices)
        if distance <= radius:
            remaining.remove(label_index)
            matches.append((label_index, float(score), int(x), int(y), float(distance)))
            peak_indices.add(peak_index)
    return matches, peak_indices


def capacity_select(peaks: Sequence[Peak], capacity: int) -> list[Peak]:
    """Return the first score-ranked candidates under a fixed capacity."""
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    return list(peaks[:capacity])


def known_label_recall_summary(
    peaks: Sequence[Peak], labels: Sequence[Mapping[str, Any]], radius: float
) -> dict[str, Any]:
    matches, matched_peak_indices = match_peaks_one_to_one(peaks, labels, radius)
    return {
        "matched": len(matches),
        "labels": len(labels),
        "candidates": len(peaks),
        "recall": len(matches) / len(labels) if labels else 0.0,
        "known_label_candidate_fraction_lower_bound": len(matches) / len(peaks) if peaks else 0.0,
        "matched_peak_indices": sorted(matched_peak_indices),
    }


def candidate_records(
    lane: str,
    frame_or_burst_id: int,
    peaks: Sequence[Peak],
    labels: Sequence[Mapping[str, Any]],
    radius: float,
) -> list[dict[str, Any]]:
    """Create explicit known-match/unknown-candidate records."""
    _, matched = match_peaks_one_to_one(peaks, labels, radius)
    records = []
    for index, (score, x, y) in enumerate(peaks):
        nearest = min(
            (math.hypot(x - float(row["x_px"]), y - float(row["y_px"])) for row in labels),
            default=float("inf"),
        )
        records.append({
            "lane": lane,
            "frame_or_burst_id": int(frame_or_burst_id),
            "score": float(score),
            "x_px": int(x),
            "y_px": int(y),
            "matched_known_label": index in matched,
            "nearest_known_label_px": float(nearest),
            "interpretation": "known_match" if index in matched else "unknown_candidate",
        })
    return records
