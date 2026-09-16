"""Controlled local-max prefilter width with unchanged6px separation/border."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from .two_stencil_evaluation import CALIBRATION_BURDEN_UNIT


def extract_candidates(score: Any, *, source_frame_ui: int, window: int = 13,
                       cell_id: str = "") -> list[dict[str, Any]]:
    """Complete strict-positive prefix; only the3x3/13x13 prefilter changes.

    Both selectors preserve the6px image border, raw-score/y/x ordering and
    greedy Euclidean separation strictly greater than6px. A spatial hash makes
    that exact greedy check efficient without changing retained coordinates.
    Regional thresholds are applied by the caller AFTER this global selector.
    """
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 12 or not np.isfinite(values).all():
        raise ValueError("score must be a finite YX frame with nonempty6px interior")
    if isinstance(window, (bool, np.bool_)) or window not in (3, 13):
        raise ValueError("window must be3 or13")
    try:
        frame_number = float(source_frame_ui)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_frame_ui must be a positive integer") from exc
    if isinstance(source_frame_ui, (bool, np.bool_)) or not math.isfinite(frame_number) or frame_number < 1 or frame_number != math.floor(frame_number):
        raise ValueError("source_frame_ui must be a positive integer")
    frame = int(frame_number)
    keep = (values == maximum_filter(values, size=int(window), mode="nearest")) & (values > 0)
    keep[:6] = False; keep[-6:] = False; keep[:, :6] = False; keep[:, -6:] = False
    yy, xx = np.nonzero(keep)
    order = np.lexsort((xx, yy, -values[yy, xx]))
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    peaks = []
    for index in order:
        x, y = int(xx[index]), int(yy[index])
        bx, by = x // 6, y // 6
        nearby = (point for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  for point in buckets.get((bx + dx, by + dy), ()))
        if any((x - old_x) ** 2 + (y - old_y) ** 2 <= 36 for old_x, old_y in nearby):
            continue
        peaks.append((float(values[y, x]), x, y))
        buckets.setdefault((bx, by), []).append((x, y))
    return [dict(
        proposal_id=f"{cell_id}__w{window}__ui{frame:06d}__r{rank:05d}", cell_id=str(cell_id),
        target_proposals_per_frame=None, calibration_burden_unit=CALIBRATION_BURDEN_UNIT,
        threshold_z=0.0, source_frame_ui=frame, source_time_s=(frame - 1) * 20.0 / 1000.0,
        source_time_basis="recording_relative_declared_frame_interval", frame_interval_ms=20.0,
        candidate_rank_within_frame=rank, score=value, x_px=x, y_px=y,
        biological_status="unknown_unreviewed_proposal", temporal_linking_applied=False,
    ) for rank, (value, x, y) in enumerate(peaks, 1)]
