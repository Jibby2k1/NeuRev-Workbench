"""Generate detector-independent timing suggestions for human adjudication."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench.experiments.learnable_contrast import core as label_core

from .adjudication import load_tsv
from .config import HardRoiAdjudicationConfig


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _disk_masks(
    shape: tuple[int, int], x: float, y: float
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    distance = np.sqrt((xx - float(x)) ** 2 + (yy - float(y)) ** 2)
    return distance <= 2.5, (distance >= 4.0) & (distance <= 7.0)


def _smooth(values: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(values, dtype=np.float64), (1, 1), mode="edge")
    return np.convolve(padded, np.ones(3) / 3.0, mode="valid")


def timing_suggestion(
    frames_ui: np.ndarray,
    trace: np.ndarray,
    *,
    original_start_ui: int,
    original_end_ui: int,
) -> dict[str, Any]:
    """Return a conservative raw-trace timing suggestion for reviewer inspection."""
    frames = np.asarray(frames_ui, dtype=np.int64)
    values = _smooth(np.asarray(trace, dtype=np.float64))
    baseline_mask = frames < int(original_start_ui)
    if baseline_mask.sum() < 4:
        baseline_mask = np.arange(len(frames)) < min(5, len(frames))
    baseline_values = values[baseline_mask]
    baseline = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - baseline))) * 1.4826
    event_search = (frames >= int(original_start_ui) - 12) & (
        frames <= int(original_end_ui) + 12
    )
    search_indices = np.flatnonzero(event_search)
    peak_index = int(search_indices[np.argmax(values[event_search])])
    peak = float(values[peak_index])
    threshold = baseline + max(3.0 * mad, 0.25 * max(peak - baseline, 0.0), 1.0)
    before = np.flatnonzero((np.arange(len(frames)) <= peak_index) & (values >= threshold))
    after = np.flatnonzero((np.arange(len(frames)) >= peak_index) & (values < threshold))
    onset_index = int(before[0]) if before.size else peak_index
    end_index = int(after[0] - 1) if after.size and after[0] > peak_index else len(frames) - 1
    end_index = max(peak_index, end_index)
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
        "method": "three_frame_smoothed_center_mean_minus_annulus_median",
        "interpretation": "reviewer_aid_not_adjudication",
    }


def _plot_trace(
    path: Path,
    frames: np.ndarray,
    trace: np.ndarray,
    row: Mapping[str, Any],
    suggestion: Mapping[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 3.5), dpi=130)
    axis.plot(frames, trace, color="#375a7f", linewidth=1.0, alpha=0.55, label="local contrast")
    axis.plot(frames, _smooth(trace), color="#00a6a6", linewidth=1.8, label="3-frame smooth")
    axis.axvspan(
        int(row["original_start_frame_ui"]), int(row["original_end_frame_ui"]),
        color="#e2b714", alpha=0.14, label="original window",
    )
    colors = {
        "suggested_onset_ui": "#3cb44b",
        "suggested_peak_ui": "#e6194b",
        "suggested_end_ui": "#4363d8",
    }
    labels = {
        "suggested_onset_ui": "suggested onset",
        "suggested_peak_ui": "suggested peak",
        "suggested_end_ui": "suggested end",
    }
    for key in colors:
        axis.axvline(int(suggestion[key]), color=colors[key], linestyle="--", linewidth=1.2, label=labels[key])
    axis.axhline(float(suggestion["suggestion_threshold"]), color="#777777", linestyle=":", linewidth=1.0, label="heuristic threshold")
    axis.set_title(
        f"{row['observation_id']} | raw local trace | detector outcomes hidden"
    )
    axis.set_xlabel("UI frame (one-based)")
    axis.set_ylabel("center mean - annulus median")
    axis.legend(loc="upper right", fontsize=7, ncol=3)
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _nearest_identity_distance(
    row: Mapping[str, Any], labels: list[Mapping[str, Any]]
) -> tuple[str, float]:
    candidates = []
    for other in labels:
        if str(other["roi_identity"]) == str(row["original_roi_id"]):
            continue
        distance = math.hypot(
            float(other["x_px"]) - float(row["x_px"]),
            float(other["y_px"]) - float(row["y_px"]),
        )
        candidates.append((distance, str(other["roi_identity"])))
    distance, identity = min(candidates)
    return identity, float(distance)


def generate_review_checklist(
    config: HardRoiAdjudicationConfig,
    *,
    adjudication_tsv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    destination = output_dir.resolve()
    partial = Path(str(destination) + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("completed or partial review checklist already exists")
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    original = label_core.load_labels(config.original_labels_tsv)
    rows = load_tsv(adjudication_tsv.resolve())
    targets = set(map(str, config.review["target_roi_ids"]))
    target_rows = [row for row in rows if row["original_roi_id"] in targets]
    partial.mkdir(parents=True, exist_ok=False)
    plots = partial / "trace_plots"
    plots.mkdir()
    checklist = []
    pad_before = int(config.review["pad_before_frames"])
    pad_after = int(config.review["pad_after_frames"])
    unique_labels = []
    seen = set()
    for label in original:
        if label["roi_identity"] not in seen:
            seen.add(label["roi_identity"])
            unique_labels.append(label)
    for row in target_rows:
        start_ui = max(1, int(row["original_start_frame_ui"]) - pad_before)
        end_ui = min(len(source), int(row["original_end_frame_ui"]) + pad_after)
        frames = np.arange(start_ui, end_ui + 1)
        movie = np.asarray(source[start_ui - 1 : end_ui], dtype=np.float64)
        center, annulus = _disk_masks(source.shape[1:], row["x_px"], row["y_px"])
        trace = movie[:, center].mean(axis=1) - np.median(movie[:, annulus], axis=1)
        suggestion = timing_suggestion(
            frames, trace,
            original_start_ui=int(row["original_start_frame_ui"]),
            original_end_ui=int(row["original_end_frame_ui"]),
        )
        plot_name = f"{row['observation_id']}__raw_trace.png"
        _plot_trace(plots / plot_name, frames, trace, row, suggestion)
        nearest_id, nearest_distance = _nearest_identity_distance(row, unique_labels)
        checklist.append({
            "observation_id": row["observation_id"],
            "burst_id": row["burst_id"],
            "original_roi_id": row["original_roi_id"],
            "current_canonical_roi_id": row["canonical_roi_id"],
            "current_review_status": row["review_status"],
            "current_disposition": row["disposition"],
            "current_neuron_confidence": row["neuron_confidence"],
            "current_activity_confidence": row["activity_confidence"],
            "current_morphology": row["morphology"],
            "current_context": row["context"],
            "original_start_frame_ui": row["original_start_frame_ui"],
            "original_end_frame_ui": row["original_end_frame_ui"],
            **suggestion,
            "nearest_other_roi_id": nearest_id,
            "nearest_other_roi_distance_px": f"{nearest_distance:.6f}",
            "trace_plot": str((plots / plot_name).relative_to(partial)),
            "source_note": row["source_note"],
            "reviewer_final_disposition": "",
            "reviewer_final_canonical_roi_id": "",
            "reviewer_event_onset_ui": "",
            "reviewer_event_peak_ui": "",
            "reviewer_event_end_ui": "",
            "reviewer_neuron_confidence": "",
            "reviewer_activity_confidence": "",
            "reviewer_morphology": "",
            "reviewer_context": "",
            "reviewer_id": "",
            "reviewed_at": "",
            "reviewer_note": "",
        })
    fields = list(checklist[0])
    with (partial / "pending_review_checklist.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(checklist)
    payload = {
        "schema_version": 1,
        "status": "review_aid_ready",
        "source_adjudication_tsv": str(adjudication_tsv.resolve()),
        "target_observations": len(checklist),
        "pending_observations": sum(row["current_review_status"] == "pending" for row in checklist),
        "provisional_observations": sum(row["current_review_status"] == "provisional_expert_note" for row in checklist),
        "trace_plot_count": len(list(plots.glob("*.png"))),
        "detector_outcomes_used": False,
        "automatic_adjudication_performed": False,
        "timing_suggestions_are_final": False,
        "interpretation": (
            "Timing suggestions come only from a raw local center-minus-annulus trace. "
            "They are reviewer aids and must not populate final labels without human acceptance."
        ),
    }
    _atomic_json(partial / "manifest.json", payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(destination)
    return payload
