"""Frozen-panel re-evaluation under versioned adjudicated label views."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.patch_information_program import (
    _pool_values,
)
from neurobench.metrics.sparse_detection import extract_local_maxima

from .adjudication import label_view, load_tsv
from .config import HardRoiAdjudicationConfig


REVIEW_START_UI = 1800
PRIMARY_FAILURE_BUDGET = 58


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty failure table")
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _decode_tiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        description = json.loads(tif.pages[0].description or "{}")
        stored = tif.asarray()
    display = description.get("global_display", {})
    black = float(display.get("black", 0.0))
    white = float(display.get("white", 1.0))
    values = black + np.asarray(stored, dtype=np.float32) * (
        (white - black) / np.iinfo(stored.dtype).max
    )
    metadata = {
        "storage": "display_tiff_uint16",
        "quantized": True,
        "display_clipped": True,
        "description": description,
        "interpretation": (
            "Monotone display decoding preserves most spatial ordering but is not "
            "the original floating-point feature tensor."
        ),
    }
    return values, metadata


def _load_feature(row: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(str(row["path"])).resolve()
    if row["storage"] == "npy_float":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return values, {
            "storage": "npy_float", "quantized": False,
            "display_clipped": False,
            "interpretation": "Quantitative frozen feature tensor.",
        }
    return _decode_tiff(path)


def _event_bounds(rows: Sequence[Mapping[str, Any]], burst: int) -> tuple[int, int]:
    selected = [row for row in rows if int(row["burst_id"]) == int(burst)]
    if not selected:
        raise ValueError(f"label view has no observations for burst {burst}")
    return (
        min(int(row["start_frame_ui"]) for row in selected),
        max(int(row["end_frame_ui"]) for row in selected),
    )


def _frames(values: np.ndarray, start_ui: int, end_ui: int) -> np.ndarray:
    start = int(start_ui) - REVIEW_START_UI
    stop = int(end_ui) - REVIEW_START_UI + 1
    if not (0 <= start < stop <= len(values)):
        raise ValueError(
            f"event interval {start_ui}--{end_ui} is outside frozen frames "
            f"{REVIEW_START_UI}--{REVIEW_START_UI + len(values) - 1}"
        )
    return np.asarray(values[start:stop])


def _candidate_records(
    values: np.ndarray,
    score_map: np.ndarray,
    start_ui: int,
    end_ui: int,
    *,
    distance: int,
    limit: int,
) -> list[dict[str, Any]]:
    event = _frames(values, start_ui, end_ui)
    peaks = extract_local_maxima(score_map, int(distance), limit=int(limit))
    records = []
    for rank, (score, x, y) in enumerate(peaks, start=1):
        peak_ui = start_ui + int(np.argmax(event[:, int(y), int(x)]))
        records.append({
            "rank": rank, "score": float(score), "x_px": int(x),
            "y_px": int(y), "peak_frame_ui": int(peak_ui),
        })
    return records


def _spatial_distance(candidate: Mapping[str, Any], label: Mapping[str, Any]) -> float:
    return math.hypot(
        float(candidate["x_px"]) - float(label["x_px"]),
        float(candidate["y_px"]) - float(label["y_px"]),
    )


def _time_valid(candidate: Mapping[str, Any], label: Mapping[str, Any]) -> bool:
    return (
        int(label["start_frame_ui"])
        <= int(candidate["peak_frame_ui"])
        <= int(label["end_frame_ui"])
    )


def _match_spatiotemporal(
    candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    radius: float,
) -> tuple[dict[int, int], set[int]]:
    remaining = set(range(len(labels)))
    matches: dict[int, int] = {}
    used: set[int] = set()
    for candidate_index, candidate in enumerate(candidates):
        choices = [
            (_spatial_distance(candidate, labels[index]), index)
            for index in remaining
            if _time_valid(candidate, labels[index])
        ]
        if not choices:
            continue
        distance, label_index = min(choices)
        if distance <= float(radius):
            remaining.remove(label_index)
            matches[label_index] = candidate_index
            used.add(candidate_index)
    return matches, used


def _failure_reason(
    label_index: int,
    labels: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    raw_candidates: Sequence[Mapping[str, Any]],
    matches: Mapping[int, int],
    *,
    budget: int,
    radius: float,
    relaxed_radius: float,
) -> tuple[str, int | None, float | None, int | None]:
    if label_index in matches:
        candidate = candidates[matches[label_index]]
        return "matched", int(candidate["rank"]), _spatial_distance(candidate, labels[label_index]), int(candidate["peak_frame_ui"])
    label = labels[label_index]
    within_budget = [
        candidate for candidate in candidates[:budget]
        if _spatial_distance(candidate, label) <= radius and _time_valid(candidate, label)
    ]
    if within_budget:
        candidate = min(within_budget, key=lambda item: _spatial_distance(item, label))
        return "identity_conflict", int(candidate["rank"]), _spatial_distance(candidate, label), int(candidate["peak_frame_ui"])
    correctly_timed = [
        candidate for candidate in candidates
        if _spatial_distance(candidate, label) <= radius and _time_valid(candidate, label)
    ]
    if correctly_timed:
        candidate = min(correctly_timed, key=lambda item: int(item["rank"]))
        return "ranking_miss", int(candidate["rank"]), _spatial_distance(candidate, label), int(candidate["peak_frame_ui"])
    spatial_only = [
        candidate for candidate in candidates
        if _spatial_distance(candidate, label) <= radius
    ]
    if spatial_only:
        candidate = min(spatial_only, key=lambda item: _spatial_distance(item, label))
        return "temporal_miss", int(candidate["rank"]), _spatial_distance(candidate, label), int(candidate["peak_frame_ui"])
    relaxed = [
        candidate for candidate in candidates
        if _spatial_distance(candidate, label) <= relaxed_radius and _time_valid(candidate, label)
    ]
    if relaxed:
        candidate = min(relaxed, key=lambda item: _spatial_distance(item, label))
        return "localization_miss", int(candidate["rank"]), _spatial_distance(candidate, label), int(candidate["peak_frame_ui"])
    suppressed = [
        candidate for candidate in raw_candidates
        if _spatial_distance(candidate, label) <= radius and _time_valid(candidate, label)
    ]
    if suppressed:
        candidate = min(suppressed, key=lambda item: _spatial_distance(item, label))
        return "nms_suppressed", int(candidate["rank"]), _spatial_distance(candidate, label), int(candidate["peak_frame_ui"])
    return "proposal_miss", None, None, None


def _evaluate_map(
    feature_id: str,
    values: np.ndarray,
    labels: Sequence[Mapping[str, Any]],
    config: HardRoiAdjudicationConfig,
    *,
    label_view_id: str,
    timing_view_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = {}
    event_bounds = {}
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        start_ui, end_ui = _event_bounds(labels, burst)
        event_bounds[burst] = [start_ui, end_ui]
        events[burst] = _frames(values, start_ui, end_ui)
    maps = _pool_values(
        np.asarray(values[:100]), events,
        float(config.evaluation["temporal_pool_temperature"]),
    )
    folds = []
    failures: list[dict[str, Any]] = []
    budgets = [int(value) for value in config.evaluation["budgets"]]
    for burst in sorted(event_bounds):
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
        start_ui, end_ui = event_bounds[burst]
        candidates = _candidate_records(
            values, maps["events"][burst], start_ui, end_ui,
            distance=int(config.evaluation["nms_distance_px"]), limit=500,
        )
        raw_candidates = _candidate_records(
            values, maps["events"][burst], start_ui, end_ui,
            distance=1, limit=5000,
        )
        budget_results = {}
        primary_matches: dict[int, int] = {}
        for budget in budgets:
            matches, _ = _match_spatiotemporal(
                candidates[:budget], burst_labels,
                float(config.evaluation["match_radius_px"]),
            )
            budget_results[str(budget)] = {
                "labels": len(burst_labels), "matched": len(matches),
                "candidates": min(budget, len(candidates)),
                "recall": len(matches) / len(burst_labels),
            }
            if budget == PRIMARY_FAILURE_BUDGET:
                primary_matches = matches
        reasons: dict[str, int] = {}
        for label_index, label in enumerate(burst_labels):
            reason, rank, distance, peak_ui = _failure_reason(
                label_index, burst_labels, candidates, raw_candidates,
                primary_matches, budget=PRIMARY_FAILURE_BUDGET,
                radius=float(config.evaluation["match_radius_px"]),
                relaxed_radius=float(config.evaluation["relaxed_localization_radius_px"]),
            )
            reasons[reason] = reasons.get(reason, 0) + 1
            failures.append({
                "feature_id": feature_id, "label_view": label_view_id,
                "timing_view": timing_view_id, "burst_id": burst,
                "observation_id": label["observation_id"],
                "original_roi_id": label["original_roi_id"],
                "canonical_roi_id": label["roi_identity"],
                "disposition": label["disposition"],
                "start_frame_ui": label["start_frame_ui"],
                "end_frame_ui": label["end_frame_ui"],
                "failure_class_at_budget_58": reason,
                "nearest_candidate_rank": "" if rank is None else rank,
                "nearest_candidate_distance_px": "" if distance is None else f"{distance:.6f}",
                "candidate_peak_frame_ui": "" if peak_ui is None else peak_ui,
                "unmatched_candidates_interpretation": "unknown_not_negative",
            })
        folds.append({
            "burst_id": burst, "event_frames_ui_inclusive": [start_ui, end_ui],
            "budgets": budget_results, "failure_counts_at_budget_58": reasons,
            "nms_candidate_count": len(candidates), "raw_local_maximum_count": len(raw_candidates),
        })
    summary = {
        str(budget): float(np.mean([fold["budgets"][str(budget)]["recall"] for fold in folds]))
        for budget in budgets
    }
    pooled = {
        str(budget): {
            "matched": sum(fold["budgets"][str(budget)]["matched"] for fold in folds),
            "labels": sum(fold["budgets"][str(budget)]["labels"] for fold in folds),
        }
        for budget in budgets
    }
    return {
        "feature_id": feature_id, "label_view": label_view_id,
        "timing_view": timing_view_id, "macro_recall": summary,
        "pooled_counts": pooled, "folds": folds,
    }, failures


def reevaluate(
    config: HardRoiAdjudicationConfig,
    *,
    adjudication_tsv: Path,
    output_dir: Path,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    partial = Path(str(output_dir) + ".partial")
    if output_dir.exists() or partial.exists():
        raise FileExistsError("completed or partial re-evaluation output already exists")
    required = set(map(str, config.review["target_roi_ids"]))
    rows = load_tsv(
        adjudication_tsv.resolve(),
        require_adjudicated_targets=None if allow_provisional else required,
    )
    partial.mkdir(parents=True, exist_ok=False)
    all_results: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    feature_provenance = {}
    for panel_row in config.frozen_panel:
        feature_id = str(panel_row["feature_id"])
        values, provenance = _load_feature(panel_row)
        if values.shape != (560, 340, 573):
            raise ValueError(f"frozen feature geometry differs for {feature_id}: {values.shape}")
        feature_provenance[feature_id] = {**dict(panel_row), **provenance}
        for view in ("original", "confirmed", "inclusive"):
            for timing in ("original", "adjudicated"):
                labels = label_view(rows, view, timing)
                result, failures = _evaluate_map(
                    feature_id, values, labels, config,
                    label_view_id=view, timing_view_id=timing,
                )
                all_results.append(result)
                all_failures.extend(failures)
        del values
    payload = {
        "schema_version": 1,
        "status": "provisional_preview" if allow_provisional else "completed",
        "experiment_id": config.experiment_id,
        "adjudication_tsv": str(adjudication_tsv.resolve()),
        "adjudication_is_final": not allow_provisional,
        "model_tuning_performed": False,
        "original_labels_overwritten": False,
        "unmatched_candidates_are_negatives": False,
        "review_start_frame_ui": REVIEW_START_UI,
        "primary_failure_budget": PRIMARY_FAILURE_BUDGET,
        "feature_provenance": feature_provenance,
        "results": all_results,
        "failure_class_counts": {
            reason: sum(row["failure_class_at_budget_58"] == reason for row in all_failures)
            for reason in sorted({row["failure_class_at_budget_58"] for row in all_failures})
        },
        "interpretation": (
            "Original/confirmed/inclusive label views and original/adjudicated timing "
            "are separate estimands. Display-TIFF lanes are monotone decoded, quantized, "
            "and clipped; they are frozen diagnostic reconstructions rather than exact "
            "floating-point reruns. Precision remains unidentified outside an exhaustive field."
        ),
    }
    _atomic_json(partial / "metrics.json", payload)
    _atomic_tsv(partial / "observation_failure_audit.tsv", all_failures)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output_dir)
    return payload
