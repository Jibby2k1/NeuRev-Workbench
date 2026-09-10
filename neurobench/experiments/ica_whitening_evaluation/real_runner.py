"""Resumable label-blind common-universe ICA screen on the real recording."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from scipy.special import logsumexp

from neurobench.experiments.frame_difference import _atomic_json
from neurobench.experiments.learned_operator_selection.data import load_and_validate_labels
from neurobench.metrics.sparse_detection import match_peaks_one_to_one

from .config import ICAWhiteningConfig
from .model import (
    PatchObservations, activity_priority_order, extract_patch_observations, fit_ica,
)
from .operators import apply_whitening
from .real_config import RealDataConfig
from .responses import component_response_summary
from .sparse_whitening import (
    fit_sparse_sequential, fit_sparse_single_stage, sparse_diagnostics,
    whitened_patch_observations,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed_review(source: np.ndarray, parent: ICAWhiteningConfig) -> np.ndarray:
    start, stop = parent.frames.review_start_ui - 1, parent.frames.review_end_ui
    values = np.asarray(source[start:stop], dtype=np.float32)
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    center = float(np.median(values[:quiet]))
    low, high = np.percentile(values[:quiet, ::4, ::4], [1, 99.9])
    return ((values - center) / max(float(high - low), 1.0)).astype(np.float32)


def _load_specs(stage_root: Path, *, whitening_geometry: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(stage_root.glob("S1_NUMERICAL_TRUTH_KNOWN_SHARD_*_OF_04/fit_results.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["all_numerically_resolved"] and row["whitening_geometry"] == whitening_geometry:
                    rows.append({key: value for key, value in row.items() if key != "cases"})
    rows.sort(key=lambda row: row["fit_id"])
    return rows


def _load_proposals(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {**row, "burst_id": int(row["burst_id"]), "x_px": int(row["x_px"]),
             "y_px": int(row["y_px"]),
             "peak_frame_review_zero": int(row["peak_frame_review_zero"]),
             "origin_score": float(row["origin_score"])}
            for row in csv.DictReader(stream, delimiter="\t")
        ]


def _candidate_observations(
    movie: np.ndarray, proposals: list[dict[str, Any]], specification: dict[str, Any]
) -> np.ndarray:
    family = specification["family"]
    spatial_width = specification.get("spatial_width_px")
    temporal_width = specification.get("temporal_width_frames")
    causality = specification["causality"]
    sw = int(spatial_width or 1); tw = int(temporal_width or 1)
    sr = sw // 2 if family in {"spatial", "joint_spatiotemporal"} else 0
    if family in {"temporal", "joint_spatiotemporal"}:
        before, after = ((tw // 2, tw // 2) if causality == "centered" else (tw - 1, 0))
    else:
        before = after = 0
    padded = np.pad(movie, ((before, after), (sr, sr), (sr, sr)), mode="reflect")
    toff = np.arange(-before, after + 1) if family in {"temporal", "joint_spatiotemporal"} else np.asarray([0])
    soff = np.arange(-sr, sr + 1) if family in {"spatial", "joint_spatiotemporal"} else np.asarray([0])
    t = np.asarray([row["peak_frame_review_zero"] for row in proposals])
    y = np.asarray([row["y_px"] for row in proposals])
    x = np.asarray([row["x_px"] for row in proposals])
    tt = t[:, None, None, None] + before + toff[None, :, None, None]
    yy = y[:, None, None, None] + sr + soff[None, None, :, None]
    xx = x[:, None, None, None] + sr + soff[None, None, None, :]
    return padded[tt, yy, xx].reshape(len(proposals), -1).T.astype(np.float64)


def _observations_at_coordinates(
    movie: np.ndarray, times: np.ndarray, rows: np.ndarray, columns: np.ndarray,
    specification: dict[str, Any],
) -> np.ndarray:
    family = specification["family"]
    sw = int(specification.get("spatial_width_px") or 1)
    tw = int(specification.get("temporal_width_frames") or 1)
    sr = sw // 2 if family in {"spatial", "joint_spatiotemporal"} else 0
    if family in {"temporal", "joint_spatiotemporal"}:
        before, after = ((tw // 2, tw // 2) if specification["causality"] == "centered" else (tw - 1, 0))
    else:
        before = after = 0
    padded = np.pad(movie, ((before, after), (sr, sr), (sr, sr)), mode="reflect")
    temporal_offsets = np.arange(-before, after + 1) if family in {"temporal", "joint_spatiotemporal"} else np.asarray([0])
    spatial_offsets = np.arange(-sr, sr + 1) if family in {"spatial", "joint_spatiotemporal"} else np.asarray([0])
    tt = times[:, None, None, None] + before + temporal_offsets[None, :, None, None]
    yy = rows[:, None, None, None] + sr + spatial_offsets[None, None, :, None]
    xx = columns[:, None, None, None] + sr + spatial_offsets[None, None, None, :]
    return padded[tt, yy, xx].reshape(len(times), -1).T.astype(np.float64)


def _candidate_event_observations(
    movie: np.ndarray, proposals: list[dict[str, Any]], specification: dict[str, Any],
    intervals: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    times, rows, columns, groups = _candidate_event_layout(proposals, intervals)
    values = _observations_at_coordinates(
        movie, times, rows, columns, specification,
    )
    return values, groups


def _candidate_event_layout(
    proposals: list[dict[str, Any]], intervals: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build the label-free candidate-major event coordinate layout."""
    time_parts = []; row_parts = []; column_parts = []; groups = []; offset = 0
    for burst in sorted(intervals):
        indices = np.asarray([index for index, row in enumerate(proposals) if row["burst_id"] == burst], dtype=np.int32)
        start, stop = intervals[burst]; duration = stop - start
        time_parts.append(np.tile(np.arange(start, stop, dtype=np.int32), len(indices)))
        row_parts.append(np.repeat([proposals[index]["y_px"] for index in indices], duration))
        column_parts.append(np.repeat([proposals[index]["x_px"] for index in indices], duration))
        count = len(indices) * duration
        groups.append({"burst_id": burst, "candidate_indices": indices,
                       "column_start": offset, "column_stop": offset + count,
                       "duration": duration})
        offset += count
    return (np.concatenate(time_parts), np.concatenate(row_parts),
            np.concatenate(column_parts), groups)


def _label_metrics(
    proposals: list[dict[str, Any]], scores: np.ndarray,
    labels: list[dict[str, Any]], budgets: tuple[int, ...], radius: int,
) -> dict[str, Any]:
    folds = []
    reciprocal_ranks = []
    for burst in (1, 2, 3, 4):
        indices = [index for index, row in enumerate(proposals) if row["burst_id"] == burst]
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), proposals[index]["candidate_id"]))
        truth = [row for row in labels if int(row["burst_id"]) == burst]
        budget_rows = []
        for budget in budgets:
            peaks = [
                (float(scores[index]), proposals[index]["x_px"], proposals[index]["y_px"])
                for index in ordered[:budget]
            ]
            matches, _ = match_peaks_one_to_one(peaks, truth, radius)
            budget_rows.append({"budget": budget, "matched": len(matches), "recall": len(matches) / len(truth)})
        for label in truth:
            rank = next((rank for rank, index in enumerate(ordered, start=1)
                         if np.hypot(proposals[index]["x_px"] - label["x_px"],
                                     proposals[index]["y_px"] - label["y_px"]) <= radius), None)
            reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        folds.append({"burst_id": burst, "known_positive_count": len(truth), "budgets": budget_rows})
    primary = budgets[budgets.index(58)] if 58 in budgets else budgets[-1]
    primary_rows = [next(row for row in fold["budgets"] if row["budget"] == primary) for fold in folds]
    return {
        "folds": folds, "primary_budget": primary,
        "macro_known_positive_recall": float(np.mean([row["recall"] for row in primary_rows])),
        "pooled_known_positive_recall": sum(row["matched"] for row in primary_rows) / len(labels),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "unmatched_candidates": "unknown_not_negative", "precision_identified": False,
    }


def evaluate_no_whitening_spec(
    specification: dict[str, Any], movie: np.ndarray,
    candidate_values: np.ndarray, proposals: list[dict[str, Any]],
    labels: list[dict[str, Any]], parent: ICAWhiteningConfig, real: RealDataConfig,
    *, fit_values: np.ndarray | None = None,
) -> dict[str, Any]:
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    if fit_values is None:
        fit_values = extract_patch_observations(
            movie, family=specification["family"],
            spatial_width=specification.get("spatial_width_px"),
            temporal_width=specification.get("temporal_width_frames"),
            causality=specification["causality"], maximum_samples=real.fitting.maximum_fit_samples,
            seed=int(specification["seed"]), quiet_frames=quiet,
            activity_fraction=real.fitting.activity_fraction,
        ).values
    model = fit_ica(fit_values, specification, maximum_fit_samples=real.fitting.maximum_fit_samples)
    centered = candidate_values - model.whitening.mean[:, None]
    component_values = model.demixing @ centered
    scores = np.max(np.abs(component_values), axis=0)
    score_digest = hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()
    metrics = _label_metrics(
        proposals, scores, labels, real.proposals.evaluation_budgets,
        real.proposals.match_radius_px,
    )
    response = component_response_summary(
        model.demixing, family=specification["family"],
        spatial_width=specification.get("spatial_width_px"),
        temporal_width=specification.get("temporal_width_frames"),
        frame_period_ms=parent.frames.frame_period_ms,
    )
    return {
        **{key: value for key, value in specification.items() if key not in {
            "all_numerically_resolved", "converged_fraction", "mean_truth_source_correlation",
            "worst_truth_source_correlation", "mean_truth_crosstalk",
            "mean_trace_preservation", "unresolved_accuracy", "case_count",
        }},
        "real_data_fit_converged": model.converged,
        "real_data_iterations": model.iterations,
        "real_data_objective": model.objective,
        "real_data_condition_number": model.whitening.condition_number,
        "real_data_explained_fraction": model.whitening.explained_fraction,
        "candidate_score_sha256_before_label_metrics": score_digest,
        "label_metrics": metrics,
        "component_response": response["permutation_invariant_mean"],
        "synthetic_warning_non_blocking": True,
    }


def evaluate_event_standardized_spec(
    specification: dict[str, Any], fit_observations: Any,
    candidate_values: np.ndarray, candidate_groups: list[dict[str, Any]],
    proposals: list[dict[str, Any]], labels: list[dict[str, Any]],
    parent: ICAWhiteningConfig, real: RealDataConfig,
) -> dict[str, Any]:
    model = fit_ica(
        fit_observations.values, specification,
        maximum_fit_samples=real.fitting.maximum_fit_samples,
    )
    quiet_count = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    scores, calibration = _event_standardized_scores(
        model.sources, fit_observations.times, model.demixing,
        model.whitening.mean, candidate_values, candidate_groups,
        len(proposals), quiet_count,
    )
    score_digest = hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()
    metrics = _label_metrics(
        proposals, scores, labels, real.proposals.evaluation_budgets,
        real.proposals.match_radius_px,
    )
    response = component_response_summary(
        model.demixing, family=specification["family"],
        spatial_width=specification.get("spatial_width_px"),
        temporal_width=specification.get("temporal_width_frames"),
        frame_period_ms=parent.frames.frame_period_ms,
    )
    return {
        **{key: value for key, value in specification.items() if key not in {
            "all_numerically_resolved", "converged_fraction", "mean_truth_source_correlation",
            "worst_truth_source_correlation", "mean_truth_crosstalk",
            "mean_trace_preservation", "unresolved_accuracy", "case_count",
        }},
        "real_data_fit_converged": model.converged,
        "real_data_iterations": model.iterations,
        "real_data_objective": model.objective,
        "real_data_condition_number": model.whitening.condition_number,
        "real_data_explained_fraction": model.whitening.explained_fraction,
        "candidate_score_sha256_before_label_metrics": score_digest,
        "label_metrics": metrics,
        "component_response": response["permutation_invariant_mean"],
        "quiet_calibration": calibration,
        "score_semantics": "quiet_mad_standardized_max_abs_component_lme0.25_over_event",
        "synthetic_warning_non_blocking": True,
    }


def _event_standardized_scores(
    fit_sources: np.ndarray, fit_times: np.ndarray, demixing: np.ndarray,
    observation_mean: np.ndarray, candidate_values: np.ndarray,
    candidate_groups: list[dict[str, Any]], candidate_count: int,
    quiet_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score candidates without labels; return robust quiet calibration."""
    quiet_sources = fit_sources[:, np.asarray(fit_times) < quiet_count]
    if quiet_sources.shape[1] < 32:
        raise RuntimeError("too few quiet fit samples for component calibration")
    center = np.median(quiet_sources, axis=1)
    scale = 1.4826 * np.median(np.abs(quiet_sources - center[:, None]), axis=1)
    scale = np.maximum(scale, np.finfo(float).eps)
    sources = demixing @ (candidate_values - observation_mean[:, None])
    scores = np.empty(candidate_count, dtype=np.float64)
    for group in candidate_groups:
        selected = sources[:, group["column_start"]:group["column_stop"]]
        selected = selected.reshape(fit_sources.shape[0], len(group["candidate_indices"]), group["duration"])
        standardized = np.abs((selected - center[:, None, None]) / scale[:, None, None])
        frame_evidence = np.max(standardized, axis=0)
        tau = 0.25
        pooled = tau * (logsumexp(frame_evidence / tau, axis=1) - np.log(group["duration"]))
        scores[group["candidate_indices"]] = pooled
    return scores, {
        "quiet_sample_count": int(quiet_sources.shape[1]),
        "component_center": center.tolist(), "component_mad_scale": scale.tolist(),
    }


def run_no_whitening_shard(
    real: RealDataConfig, *, preflight_dir: str | Path, shard_index: int,
) -> dict[str, Any]:
    preflight_root = Path(preflight_dir).expanduser().resolve()
    preflight = json.loads((preflight_root / "preflight.json").read_text(encoding="utf-8"))
    if not preflight.get("ready") or preflight["proposal_universe"]["digest"] != _sha256_json_rows(
        _load_proposals(preflight_root / "common_proposal_universe.tsv")
    ):
        raise RuntimeError("matching ready real-data preflight is required")
    if not 0 <= shard_index < real.fitting.shard_count:
        raise ValueError("invalid shard index")
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    proposals = _load_proposals(preflight_root / "common_proposal_universe.tsv")
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    stage_root = real.synthetic_registry.parent.parent
    specifications = _load_specs(stage_root, whitening_geometry="none")
    specifications = [row for index, row in enumerate(specifications)
                      if index % real.fitting.shard_count == shard_index]
    specifications.sort(key=lambda row: (
        row["family"], row.get("spatial_width_px") or 0,
        row.get("temporal_width_frames") or 0, row["causality"],
        int(row["seed"]), row["fit_id"],
    ))
    root = real.output_dir / "stages" / f"S2A_NO_WHITENING_SHARD_{shard_index:02d}_OF_{real.fitting.shard_count:02d}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "stage": "S2A_NO_WHITENING_COMMON_UNIVERSE",
        "shard_index": shard_index, "shard_count": real.fitting.shard_count,
        "expected_fit_count": len(specifications),
        "proposal_digest": preflight["proposal_universe"]["digest"],
        "label_coordinates_used_for_fitting_or_candidate_generation": False,
        "label_metric_function_invoked_after_score_hash": True,
        "synthetic_warning_non_blocking": True,
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing shard manifest differs")
    if not manifest_path.exists(): _atomic_json(manifest_path, manifest)
    checkpoint = root / "fit_results.jsonl"
    completed = set()
    if checkpoint.is_file():
        with checkpoint.open(encoding="utf-8") as stream:
            completed = {json.loads(line)["fit_id"] for line in stream if line.strip()}
    started = time.monotonic()
    candidate_cache: dict[tuple[Any, ...], np.ndarray] = {}
    fit_cache: dict[tuple[Any, ...], np.ndarray] = {}
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    for specification in specifications:
        if specification["fit_id"] in completed: continue
        key = (specification["family"], specification.get("spatial_width_px"),
               specification.get("temporal_width_frames"), specification["causality"])
        if key not in candidate_cache:
            candidate_cache[key] = _candidate_observations(movie, proposals, specification)
        fit_key = (*key, int(specification["seed"]))
        if fit_key not in fit_cache:
            fit_cache[fit_key] = extract_patch_observations(
                movie, family=specification["family"],
                spatial_width=specification.get("spatial_width_px"),
                temporal_width=specification.get("temporal_width_frames"),
                causality=specification["causality"],
                maximum_samples=real.fitting.maximum_fit_samples,
                seed=int(specification["seed"]), quiet_frames=quiet,
                activity_fraction=real.fitting.activity_fraction,
            ).values
        result = evaluate_no_whitening_spec(
            specification, movie, candidate_cache[key], proposals, labels, parent, real,
            fit_values=fit_cache[fit_key],
        )
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        completed.add(specification["fit_id"])
        _atomic_json(root / "progress.json", {
            "status": "running", "completed": len(completed), "expected": len(specifications),
            "last_fit_id": specification["fit_id"], "elapsed_seconds": time.monotonic() - started,
        })
    summary = {
        "status": "complete", "fit_count": len(completed), "expected_fit_count": len(specifications),
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", {
        "status": "pass" if len(completed) == len(specifications) else "fail",
        "observed": len(completed), "expected": len(specifications),
    })
    _atomic_json(root / "progress.json", {"status": "complete", "completed": len(completed), "expected": len(specifications)})
    return summary


def run_event_standardized_shard(
    real: RealDataConfig, *, preflight_dir: str | Path, shard_index: int,
) -> dict[str, Any]:
    preflight_root = Path(preflight_dir).expanduser().resolve()
    preflight = json.loads((preflight_root / "preflight.json").read_text(encoding="utf-8"))
    proposals = _load_proposals(preflight_root / "common_proposal_universe.tsv")
    if not preflight.get("ready") or preflight["proposal_universe"]["digest"] != _sha256_json_rows(proposals):
        raise RuntimeError("matching ready real-data preflight is required")
    if not 0 <= shard_index < real.fitting.shard_count:
        raise ValueError("invalid shard index")
    interval_payload = preflight["folds"].get("event_intervals_review_zero_half_open")
    if interval_payload is None:
        interval_contract = json.loads(
            (preflight_root / "event_interval_contract.json").read_text(encoding="utf-8")
        )
        interval_payload = interval_contract["intervals"]
    intervals = {int(key): (int(value[0]), int(value[1])) for key, value in interval_payload.items()}
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    specifications = _load_specs(real.synthetic_registry.parent.parent, whitening_geometry="none")
    specifications = [row for index, row in enumerate(specifications)
                      if index % real.fitting.shard_count == shard_index]
    specifications.sort(key=lambda row: (
        row["family"], row.get("spatial_width_px") or 0,
        row.get("temporal_width_frames") or 0, row["causality"],
        int(row["seed"]), row["fit_id"],
    ))
    root = real.output_dir / "stages" / f"S2B_EVENT_STANDARDIZED_SHARD_{shard_index:02d}_OF_{real.fitting.shard_count:02d}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "stage": "S2B_EVENT_STANDARDIZED_COMMON_UNIVERSE",
        "whitening_geometry": "none", "shard_index": shard_index,
        "shard_count": real.fitting.shard_count, "expected_fit_count": len(specifications),
        "proposal_digest": preflight["proposal_universe"]["digest"],
        "score_semantics": "quiet_mad_standardized_max_abs_component_lme0.25_over_event",
        "label_coordinates_used_for_fitting_or_candidate_generation": False,
        "label_metric_function_invoked_after_score_hash": True,
        "synthetic_warning_non_blocking": True,
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing standardized shard manifest differs")
    if not manifest_path.exists(): _atomic_json(manifest_path, manifest)
    checkpoint = root / "fit_results.jsonl"; completed = set()
    if checkpoint.is_file():
        with checkpoint.open(encoding="utf-8") as stream:
            completed = {json.loads(line)["fit_id"] for line in stream if line.strip()}
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    started = time.monotonic(); current_key = None
    candidate_values = None; candidate_groups = None; fit_cache: dict[int, Any] = {}
    for specification in specifications:
        if specification["fit_id"] in completed: continue
        key = (specification["family"], specification.get("spatial_width_px"),
               specification.get("temporal_width_frames"), specification["causality"])
        if key != current_key:
            candidate_values, candidate_groups = _candidate_event_observations(
                movie, proposals, specification, intervals
            )
            fit_cache = {}; current_key = key
        seed = int(specification["seed"])
        if seed not in fit_cache:
            fit_cache[seed] = extract_patch_observations(
                movie, family=specification["family"],
                spatial_width=specification.get("spatial_width_px"),
                temporal_width=specification.get("temporal_width_frames"),
                causality=specification["causality"],
                maximum_samples=real.fitting.maximum_fit_samples, seed=seed,
                quiet_frames=quiet, activity_fraction=real.fitting.activity_fraction,
            )
        assert candidate_values is not None and candidate_groups is not None
        result = evaluate_event_standardized_spec(
            specification, fit_cache[seed], candidate_values, candidate_groups,
            proposals, labels, parent, real,
        )
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        completed.add(specification["fit_id"])
        _atomic_json(root / "progress.json", {
            "status": "running", "completed": len(completed), "expected": len(specifications),
            "last_fit_id": specification["fit_id"], "elapsed_seconds": time.monotonic() - started,
        })
    summary = {
        "status": "complete", "fit_count": len(completed),
        "expected_fit_count": len(specifications), "runtime_seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "score_semantics": manifest["score_semantics"],
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", {
        "status": "pass" if len(completed) == len(specifications) else "fail",
        "observed": len(completed), "expected": len(specifications),
    })
    _atomic_json(root / "progress.json", {
        "status": "complete", "completed": len(completed), "expected": len(specifications)
    })
    return summary


def run_single_stage_whitening_shard(
    real: RealDataConfig, *, preflight_dir: str | Path, shard_index: int,
    whitening_geometry: str,
) -> dict[str, Any]:
    """Run one exact sparse-whitening geometry with the frozen event scorer."""
    supported = {"spatial", "temporal", "joint_spatiotemporal",
                 "spatial_then_temporal", "temporal_then_spatial"}
    if whitening_geometry not in supported:
        raise ValueError("whitening geometry is invalid")
    preflight_root = Path(preflight_dir).expanduser().resolve()
    preflight = json.loads((preflight_root / "preflight.json").read_text(encoding="utf-8"))
    proposals = _load_proposals(preflight_root / "common_proposal_universe.tsv")
    if not preflight.get("ready") or preflight["proposal_universe"]["digest"] != _sha256_json_rows(proposals):
        raise RuntimeError("matching ready real-data preflight is required")
    if not 0 <= shard_index < real.fitting.shard_count:
        raise ValueError("invalid shard index")
    interval_payload = preflight["folds"].get("event_intervals_review_zero_half_open")
    if interval_payload is None:
        interval_payload = json.loads((preflight_root / "event_interval_contract.json").read_text())["intervals"]
    intervals = {int(k): (int(v[0]), int(v[1])) for k, v in interval_payload.items()}
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    specifications = _load_specs(real.synthetic_registry.parent.parent, whitening_geometry=whitening_geometry)
    specifications = [row for index, row in enumerate(specifications)
                      if index % real.fitting.shard_count == shard_index]
    specifications.sort(key=lambda row: row["fit_id"])
    tag = whitening_geometry.upper()
    root = real.output_dir / "stages" / f"S2C_{tag}_WHITENING_SHARD_{shard_index:02d}_OF_{real.fitting.shard_count:02d}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "stage": "S2C_SPARSE_WHITENING",
        "whitening_geometry": whitening_geometry, "shard_index": shard_index,
        "shard_count": real.fitting.shard_count, "expected_fit_count": len(specifications),
        "proposal_digest": preflight["proposal_universe"]["digest"],
        "score_semantics": "quiet_mad_standardized_max_abs_component_lme0.25_over_event",
        "sparse_operator_equivalence_tested": True,
        "label_coordinates_used_for_fitting_or_candidate_generation": False,
        "label_metric_function_invoked_after_score_hash": True,
        "synthetic_warning_non_blocking": True,
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        compatible = (
            existing_manifest.get("stage") in {
                "S2C_SINGLE_STAGE_WHITENING", "S2C_SPARSE_WHITENING",
            }
            and {**existing_manifest, "stage": manifest["stage"]} == manifest
        )
        if not compatible:
            raise RuntimeError("existing whitening shard manifest differs")
    if not manifest_path.exists(): _atomic_json(manifest_path, manifest)
    checkpoint = root / "fit_results.jsonl"; completed = set()
    if checkpoint.is_file():
        with checkpoint.open(encoding="utf-8") as stream:
            completed = {json.loads(line)["fit_id"] for line in stream if line.strip()}
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    shared_activity_order = activity_priority_order(movie, quiet)
    started = time.monotonic()
    raw_fit_cache: dict[tuple[Any, ...], PatchObservations] = {}
    for specification in specifications:
        if specification["fit_id"] in completed: continue
        raw_fit_key = (
            specification["family"], specification.get("spatial_width_px"),
            specification.get("temporal_width_frames"), specification["causality"],
            int(specification["seed"]),
        )
        if raw_fit_key not in raw_fit_cache:
            raw_fit_cache[raw_fit_key] = extract_patch_observations(
                movie, family=specification["family"],
                spatial_width=specification.get("spatial_width_px"),
                temporal_width=specification.get("temporal_width_frames"),
                causality=specification["causality"], maximum_samples=real.fitting.maximum_fit_samples,
                seed=int(specification["seed"]), quiet_frames=quiet,
                activity_fraction=real.fitting.activity_fraction,
                activity_order=shared_activity_order,
            )
        raw_fit = raw_fit_cache[raw_fit_key]
        # The dense compiled-convolution reference is exact and substantially
        # faster than sparse application when joint model patches cover much of
        # the event field. Sparse/dense equivalence is guarded by focused tests.
        whitening = apply_whitening(
            movie, quiet, specification,
            maximum_samples=real.fitting.maximum_fit_samples,
        )
        fit_values = _observations_at_coordinates(
            whitening.output, raw_fit.times, raw_fit.rows, raw_fit.columns,
            specification,
        )
        fit_observations = PatchObservations(
            fit_values, raw_fit.times, raw_fit.rows, raw_fit.columns
        )
        candidate_values, groups = _candidate_event_observations(
            whitening.output, proposals, specification, intervals
        )
        result = evaluate_event_standardized_spec(
            specification, fit_observations, candidate_values, groups,
            proposals, labels, parent, real,
        )
        result["external_whitening_diagnostics"] = whitening.diagnostics
        result["external_whitening_application"] = "dense_compiled_reference"
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        completed.add(specification["fit_id"])
        _atomic_json(root / "progress.json", {
            "status": "running", "completed": len(completed), "expected": len(specifications),
            "last_fit_id": specification["fit_id"], "elapsed_seconds": time.monotonic() - started,
        })
    summary = {"status": "complete", "whitening_geometry": whitening_geometry,
               "fit_count": len(completed), "expected_fit_count": len(specifications),
               "runtime_seconds": time.monotonic() - started,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", {
        "status": "pass" if len(completed) == len(specifications) else "fail",
        "observed": len(completed), "expected": len(specifications),
    })
    _atomic_json(root / "progress.json", {"status": "complete", "completed": len(completed), "expected": len(specifications)})
    return summary


def _sha256_json_rows(rows: list[dict[str, Any]]) -> str:
    normalized = [{key: row[key] for key in row} for row in rows]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
