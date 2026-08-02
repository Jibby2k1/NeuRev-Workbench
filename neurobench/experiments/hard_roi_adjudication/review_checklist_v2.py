"""Conservative contiguous-event timing suggestions for review-checklist v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import HardRoiAdjudicationConfig
from . import review_checklist as base


def timing_suggestion_v2(
    frames_ui: np.ndarray,
    trace: np.ndarray,
    *,
    original_start_ui: int,
    original_end_ui: int,
) -> dict[str, Any]:
    frames = np.asarray(frames_ui, dtype=np.int64)
    raw = np.asarray(trace, dtype=np.float64)
    baseline_mask = frames < int(original_start_ui)
    if baseline_mask.sum() < 4:
        baseline_mask = np.arange(len(frames)) < min(5, len(frames))
    baseline_values = base._smooth(raw[baseline_mask])
    baseline = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - baseline))) * 1.4826
    search_mask = (frames >= int(original_start_ui) - 12) & (
        frames <= int(original_end_ui) + 12
    )
    search = np.flatnonzero(search_mask)
    if not search.size:
        raise ValueError("timing search interval is empty")
    # Smooth only inside the permitted search interval. This prevents a strong
    # earlier event from leaking into the first eligible frame through the
    # three-frame kernel.
    search_values = base._smooth(raw[search])
    peak_local = int(np.argmax(search_values))
    peak_index = int(search[peak_local])
    peak = float(search_values[peak_local])
    threshold = baseline + max(3.0 * mad, 0.25 * max(peak - baseline, 0.0), 1.0)
    onset_local = peak_local
    while onset_local > 0 and search_values[onset_local - 1] >= threshold:
        onset_local -= 1
    end_local = peak_local
    while end_local < len(search_values) - 1 and search_values[end_local + 1] >= threshold:
        end_local += 1
    onset_index = int(search[onset_local])
    end_index = int(search[end_local])
    signal_to_noise = (peak - baseline) / max(mad, 1e-6)
    return {
        "suggested_onset_ui": int(frames[onset_index]),
        "suggested_peak_ui": int(frames[peak_index]),
        "suggested_end_ui": int(frames[end_index]),
        "baseline_local_contrast": baseline,
        "baseline_robust_sigma": mad,
        "suggestion_threshold": threshold,
        "peak_local_contrast": peak,
        "peak_above_baseline": peak - baseline,
        "peak_precedes_original_window": bool(frames[peak_index] < int(original_start_ui)),
        "suggestion_signal_to_noise": signal_to_noise,
        "suggestion_strength": (
            "strong" if signal_to_noise >= 6 else
            "moderate" if signal_to_noise >= 3 else "weak"
        ),
        "method": "v2_contiguous_three_frame_center_minus_annulus",
        "interpretation": "reviewer_aid_not_adjudication",
    }


def generate_review_checklist_v2(
    config: HardRoiAdjudicationConfig,
    *,
    adjudication_tsv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    original = base.timing_suggestion
    base.timing_suggestion = timing_suggestion_v2
    try:
        payload = base.generate_review_checklist(
            config,
            adjudication_tsv=adjudication_tsv,
            output_dir=output_dir,
        )
    finally:
        base.timing_suggestion = original
    payload["timing_suggestion_version"] = 2
    return payload
