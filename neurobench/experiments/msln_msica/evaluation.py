"""Synthetic morphology and nuisance audit used by the guarded stage gates."""
from __future__ import annotations

import numpy as np

from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import (
    candidate_records,
    extract_local_maxima,
    known_label_recall_summary,
    temporal_pool,
)

from .config import MSLNMSICAConfig
from .context_bank import evaluate_context, ordered_contexts


def sparse_positive_evaluation(
    raw: np.ndarray,
    evidence: np.ndarray,
    config: MSLNMSICAConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate fixed candidate budgets while keeping unmatched candidates unknown."""
    labels = label_core.load_labels(config.source.labels_path)
    review_start, _ = config.source.review_interval_ui
    maximum_budget = max(config.evaluation.candidate_budgets)
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for burst_id, interval in sorted(config.source.burst_intervals_ui.items()):
        start = interval[0] - review_start
        stop = interval[1] - review_start + 1
        if start < 0 or stop > len(raw):
            continue
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst_id]
        lane_maps = {
            "raw_amplitude_carrier": temporal_pool(raw[start:stop], "max"),
            "fixed_unsupervised_activity_evidence": temporal_pool(
                evidence[start:stop], "max"
            ),
        }
        for lane, score_map in lane_maps.items():
            peaks = extract_local_maxima(
                score_map,
                config.evaluation.nms_distance_px,
                limit=maximum_budget,
            )
            candidate_rows.extend(
                candidate_records(
                    lane,
                    burst_id,
                    peaks,
                    burst_labels,
                    config.evaluation.match_radius_px,
                )
            )
            for budget in config.evaluation.candidate_budgets:
                summary = known_label_recall_summary(
                    peaks[:budget], burst_labels, config.evaluation.match_radius_px
                )
                rows.append(
                    {
                        "burst_id": burst_id,
                        "lane": lane,
                        "budget": budget,
                        **summary,
                        "unmatched_candidates_are": "unknown",
                    }
                )
    return (
        {
            "protocol": "fixed_unsupervised",
            "proposal_mode": "native",
            "rows": rows,
            "precision_estimated": False,
            "labels_are_sparse_known_positives": True,
        },
        candidate_rows,
    )


def synthetic_fixture_audit(
    config: MSLNMSICAConfig,
    seed: int = 20260803,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    frames, height, width = 72, 31, 31
    yy, xx = np.mgrid[:height, :width]
    fixtures = {
        "compact_transient": np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 4),
        "broad_neural": np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 64),
        "ring": np.exp(-((np.sqrt((xx - 15) ** 2 + (yy - 15) ** 2) - 6) ** 2) / 2),
        "correlated_neighbors": np.exp(-((xx - 11) ** 2 + (yy - 15) ** 2) / 4) + np.exp(-((xx - 19) ** 2 + (yy - 15) ** 2) / 4),
        "broad_drift": np.ones((height, width)),
        "moving_edge": (xx > 15).astype(float),
        "heteroscedastic_noise": np.zeros((height, width)),
        "static_bright": np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 4),
        "saturation": np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 4),
        "quiet_null": np.zeros((height, width)),
    }
    rows = []
    for index, (name, spatial) in enumerate(fixtures.items()):
        video = rng.normal(0, 0.2, size=(frames, height, width))
        if name == "heteroscedastic_noise":
            video *= np.linspace(0.2, 2.0, width)[None, None, :]
        elif name == "static_bright":
            video += 5 * spatial
        elif name == "broad_drift":
            video += np.linspace(0, 2, frames)[:, None, None]
        elif name == "moving_edge":
            for t in range(frames):
                video[t] += (xx > (5 + t // 3)).astype(float)
        elif name != "quiet_null":
            amplitude = 8 if name == "saturation" else 3
            video[44:47] += amplitude * spatial
            if name == "saturation":
                video = np.clip(video, -2, 4)
        center_trace = video[:, 15, 15]
        peak_frame = int(np.argmax(center_trace))
        quiet = np.arange(frames) < 32
        truth_mask = (
            spatial >= 0.5 * np.max(spatial)
            if np.max(spatial) > 0
            else np.zeros_like(spatial, dtype=bool)
        )
        if not np.any(truth_mask) or np.all(truth_mask):
            truth_mask[:] = False
            truth_mask[15, 15] = True
        background = ~truth_mask
        carrier_centered = video - np.median(video[quiet], axis=0, keepdims=True)
        carrier_event = np.max(np.abs(carrier_centered[44:47]), axis=0)
        carrier_contrast = float(
            np.mean(carrier_event[truth_mask])
            / max(float(np.percentile(carrier_event[background], 95)), 1e-6)
        )
        context_metrics = []
        for definition in ordered_contexts(config):
            result = evaluate_context(video.astype(np.float32), definition, quiet_mask=quiet)
            event_map = np.max(np.abs(result.values[44:47]), axis=0)
            context_metrics.append(
                {
                    "context_id": definition.context_id,
                    "contrast": float(
                        np.mean(event_map[truth_mask])
                        / max(float(np.percentile(event_map[background], 95)), 1e-6)
                    ),
                    "half_peak_area": int(
                        np.sum(event_map >= 0.5 * np.max(event_map))
                    ),
                    "finite": bool(np.isfinite(result.values).all()),
                }
            )
        rows.append({"fixture": name, "peak_frame": peak_frame, "peak": float(np.max(center_trace)), "area_above_half_peak": int(np.sum(spatial >= 0.5 * np.max(spatial))) if np.max(spatial) else 0, "carrier_contrast": carrier_contrast, "best_context_contrast": max(row["contrast"] for row in context_metrics), "best_context_id": max(context_metrics, key=lambda row: row["contrast"])["context_id"], "contexts": context_metrics, "finite": bool(np.isfinite(video).all()) and all(row["finite"] for row in context_metrics), "truth_complete": True})
    return {"fixtures": rows, "fixture_count": len(rows), "all_finite": all(row["finite"] for row in rows), "context_count": len(ordered_contexts(config)), "scientific_interpretation": "generated complete-truth morphology/nuisance traversal of configured MSLN; not real-data efficacy"}
