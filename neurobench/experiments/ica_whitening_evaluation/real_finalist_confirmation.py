"""Strict held-out-burst finalist refits with model and diagnostic exports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from neurobench.experiments.frame_difference import _atomic_json

from .finalist_diagnostics import (
    approximate_source_snr, ordered_trace_coherence, reconstruction_integrity,
)
from .model import (
    PatchObservations, activity_priority_order, extract_patch_observations,
    fit_ica, model_summary,
)
from .operators import apply_whitening
from .real_config import RealDataConfig
from .real_controls import evaluate_rank_matched_controls
from .real_runner import (
    _candidate_event_layout, _candidate_event_observations,
    _event_standardized_scores, _label_metrics, _observations_at_coordinates,
)
from .responses import component_response_summary
from .finalist_diagnostics import seed_stability


def confirmation_seed_set(original_seed: int) -> tuple[int, ...]:
    return (int(original_seed), int(original_seed) + 104729,
            int(original_seed) + 209759)


def _held_out_fit_filter(
    observations: PatchObservations, interval: tuple[int, int],
) -> PatchObservations:
    keep = ~((observations.times >= interval[0]) & (observations.times < interval[1]))
    if int(np.sum(keep)) < 128:
        raise RuntimeError("held-out burst exclusion leaves too few fit observations")
    return PatchObservations(
        observations.values[:, keep], observations.times[keep],
        observations.rows[keep], observations.columns[keep],
    )


def _trace_diagnostics(
    movie: np.ndarray, proposals: list[dict[str, Any]],
    groups: list[dict[str, Any]], candidate_sources: np.ndarray,
    calibration: dict[str, Any], *, sample_period_seconds: float,
) -> dict[str, Any]:
    center = np.asarray(calibration["component_center"])[:, None]
    scale = np.asarray(calibration["component_mad_scale"])[:, None]
    rows = []
    for group in groups:
        selected = candidate_sources[:, group["column_start"]:group["column_stop"]]
        selected = selected.reshape(candidate_sources.shape[0],
                                    len(group["candidate_indices"]), group["duration"])
        for local_index, candidate_index in enumerate(group["candidate_indices"]):
            standardized = (selected[:, local_index] - center) / scale
            component = int(np.argmax(np.max(np.abs(standardized), axis=1)))
            proposal = proposals[int(candidate_index)]
            start = int(proposal["peak_frame_review_zero"])  # metadata only; interval comes from group columns
            del start
            # Recover the exact event times from group order and duration.
            column = group["column_start"] + local_index * group["duration"]
            # Raw center traces are aligned to the same candidate-major columns.
            source_trace = standardized[component]
            rows.append({
                "candidate_id": proposal["candidate_id"], "burst_id": group["burst_id"],
                "selected_component": component, "source_trace": source_trace,
                "column_start": column,
            })
    # Callers add raw traces using the event layout; keep only compact aggregate outputs.
    return {"rows": rows, "sample_period_seconds": sample_period_seconds}


def refit_finalist_fold(
    movie: np.ndarray, specification: dict[str, Any],
    proposals: list[dict[str, Any]], labels: list[dict[str, Any]],
    intervals: dict[int, tuple[int, int]], parent: Any, real: RealDataConfig,
    *, confirmation_seed: int, held_out_burst: int,
    activity_order: np.ndarray,
) -> dict[str, Any]:
    """Refit without held-out event samples; consume labels only after score hash."""
    if held_out_burst not in intervals:
        raise ValueError("held_out_burst is invalid")
    spec = {**specification, "seed": int(confirmation_seed)}
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    raw_fit = extract_patch_observations(
        movie, family=spec["family"], spatial_width=spec.get("spatial_width_px"),
        temporal_width=spec.get("temporal_width_frames"), causality=spec["causality"],
        maximum_samples=real.fitting.maximum_fit_samples, seed=int(confirmation_seed),
        quiet_frames=quiet, activity_fraction=real.fitting.activity_fraction,
        activity_order=activity_order,
    )
    raw_fit = _held_out_fit_filter(raw_fit, intervals[held_out_burst])
    whitening = apply_whitening(
        movie, quiet, spec, maximum_samples=real.fitting.maximum_fit_samples
    )
    fit_values = _observations_at_coordinates(
        whitening.output, raw_fit.times, raw_fit.rows, raw_fit.columns, spec
    )
    fit_observations = PatchObservations(
        fit_values, raw_fit.times, raw_fit.rows, raw_fit.columns
    )
    model = fit_ica(fit_values, spec,
                    maximum_fit_samples=real.fitting.maximum_fit_samples)
    candidate_values, groups = _candidate_event_observations(
        whitening.output, proposals, spec, intervals
    )
    scores, calibration = _event_standardized_scores(
        model.sources, fit_observations.times, model.demixing,
        model.whitening.mean, candidate_values, groups, len(proposals), quiet,
    )
    score_digest = hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()
    # Labels are accessed only after the complete candidate-score vector is frozen.
    all_metrics = _label_metrics(
        proposals, scores, labels, real.proposals.evaluation_budgets,
        real.proposals.match_radius_px,
    )
    held_out_metrics = next(item for item in all_metrics["folds"]
                            if int(item["burst_id"]) == held_out_burst)
    controls = evaluate_rank_matched_controls(
        fit_observations, candidate_values, groups, proposals, labels, spec,
        quiet_count=quiet, budgets=real.proposals.evaluation_budgets,
        match_radius_px=real.proposals.match_radius_px,
    )
    candidate_sources = model.demixing @ (candidate_values - model.whitening.mean[:, None])
    trace_payload = _trace_diagnostics(
        movie, proposals, groups, candidate_sources, calibration,
        sample_period_seconds=parent.frames.frame_period_ms / 1000.0,
    )
    event_times, event_rows, event_columns, _ = _candidate_event_layout(proposals, intervals)
    compact_coherence = []
    for item in trace_payload["rows"]:
        start = int(item.pop("column_start")); duration = next(
            group["duration"] for group in groups if group["burst_id"] == item["burst_id"]
        )
        raw_trace = movie[event_times[start:start + duration],
                          event_rows[start:start + duration], event_columns[start:start + duration]]
        source_trace = item.pop("source_trace")
        compact_coherence.append({**item, **ordered_trace_coherence(
            raw_trace, source_trace,
            sample_period_seconds=parent.frames.frame_period_ms / 1000.0,
            maximum_lag=min(12, max(1, duration // 3)),
        )})
    ranking = {
        str(burst): [proposals[index]["candidate_id"] for index in sorted(
            [i for i, row in enumerate(proposals) if int(row["burst_id"]) == burst],
            key=lambda i: (-float(scores[i]), proposals[i]["candidate_id"]),
        )[:200]] for burst in (1, 2, 3, 4)
    }
    response = component_response_summary(
        model.demixing, family=spec["family"],
        spatial_width=spec.get("spatial_width_px"),
        temporal_width=spec.get("temporal_width_frames"),
        frame_period_ms=parent.frames.frame_period_ms,
    )
    return {
        "source_fit_id": specification["fit_id"], "confirmation_seed": confirmation_seed,
        "held_out_burst": held_out_burst,
        "held_out_interval_review_zero_half_open": list(intervals[held_out_burst]),
        "held_out_samples_excluded_from_model_fit": True,
        "candidate_score_sha256_before_label_metrics": score_digest,
        "held_out_label_metrics": held_out_metrics,
        "all_fold_metrics_descriptive": all_metrics,
        "model": model_summary(model),
        "external_whitening": {"diagnostics": whitening.diagnostics,
                               "kernels": [kernel.tolist() for kernel in whitening.kernels]},
        "component_response": response,
        "reconstruction_integrity": reconstruction_integrity(
            fit_values, model.sources, model.mixing, model.whitening.mean
        ),
        "approximate_snr": approximate_source_snr(
            model.sources, fit_observations.times, quiet
        ),
        "candidate_trace_coherence": compact_coherence,
        "rank_matched_controls": controls,
        "top_200_candidate_ids_by_burst": ranking,
        "unmatched_candidates": "unknown_not_negative",
        "claim_scope": "strict_held_out_burst_within_recording_confirmation",
    }


def select_confirmation_fit_ids(factorial_summary: dict[str, Any], *, maximum: int = 6) -> list[str]:
    """Frozen compact rule: repeated LOO selections on Pareto, then deterministic fill."""
    loo = factorial_summary["protected_leave_one_burst_out"]
    frequency = {str(key): int(value) for key, value in loo["selection_frequency"].items()}
    pareto = set(map(str, factorial_summary["preliminary_pareto_fit_ids"]))
    primary = sorted(
        [fit_id for fit_id, count in frequency.items() if count >= 2 and fit_id in pareto],
        key=lambda fit_id: (-frequency[fit_id], fit_id),
    )
    fill = sorted(
        (set(frequency) | pareto) - set(primary),
        key=lambda fit_id: (-frequency.get(fit_id, 0), fit_id not in pareto, fit_id),
    )
    return (primary + fill)[:maximum]


def confirmation_fit_ids_by_held_out_burst(
    factorial_summary: dict[str, Any], *, maximum_per_fold: int = 3,
) -> dict[int, list[str]]:
    """Keep each held-out burst paired only with choices made without its labels."""
    if maximum_per_fold < 1:
        raise ValueError("maximum_per_fold must be positive")
    result = {}
    for fold in factorial_summary["protected_leave_one_burst_out"]["folds"]:
        burst = int(fold["held_out_burst"])
        selected = list(map(str, fold["selected_fit_ids"]))[:maximum_per_fold]
        if not selected:
            raise RuntimeError(f"no protected finalists for held-out burst {burst}")
        result[burst] = selected
    if set(result) != {1, 2, 3, 4}:
        raise RuntimeError("protected selection must cover exactly four held-out bursts")
    return result


def summarize_protected_refits(
    payloads: list[dict[str, Any]], *, budget: int = 58,
    bootstrap_replicates: int = 4096, seed: int = 20260831,
) -> dict[str, Any]:
    """Aggregate at the held-out-burst level; seeds/finalists are robustness runs."""
    if not payloads:
        raise ValueError("protected refit summary requires payloads")

    def fold_recall(metrics: dict[str, Any]) -> float:
        return float(next(row["recall"] for row in metrics["budgets"]
                          if int(row["budget"]) == budget))

    task_rows = []
    for payload in payloads:
        controls = {}
        for control in payload["rank_matched_controls"]["controls"]:
            fold = next(row for row in control["label_metrics"]["folds"]
                        if int(row["burst_id"]) == int(payload["held_out_burst"]))
            controls[str(control["control_id"])] = fold_recall(fold)
        task_rows.append({
            "source_fit_id": str(payload["source_fit_id"]),
            "confirmation_seed": int(payload["confirmation_seed"]),
            "held_out_burst": int(payload["held_out_burst"]),
            "known_positive_recall_at_58": fold_recall(payload["held_out_label_metrics"]),
            "rank_matched_control_recall_at_58": controls,
        })
    burst_rows = []
    for burst in (1, 2, 3, 4):
        selected = [row for row in task_rows if row["held_out_burst"] == burst]
        if not selected:
            raise RuntimeError(f"protected refits missing held-out burst {burst}")
        control_ids = sorted(set.intersection(*(
            set(row["rank_matched_control_recall_at_58"]) for row in selected
        )))
        burst_rows.append({
            "held_out_burst": burst, "robustness_run_count": len(selected),
            "mean_selected_panel_recall_at_58": float(np.mean([
                row["known_positive_recall_at_58"] for row in selected
            ])),
            "median_selected_panel_recall_at_58": float(np.median([
                row["known_positive_recall_at_58"] for row in selected
            ])),
            "range_selected_panel_recall_at_58": [
                float(min(row["known_positive_recall_at_58"] for row in selected)),
                float(max(row["known_positive_recall_at_58"] for row in selected)),
            ],
            "mean_rank_matched_control_recall_at_58": {
                control_id: float(np.mean([
                    row["rank_matched_control_recall_at_58"][control_id]
                    for row in selected
                ])) for control_id in control_ids
            },
        })
    burst_values = np.asarray([
        row["mean_selected_panel_recall_at_58"] for row in burst_rows
    ])
    rng = np.random.default_rng(seed)
    bootstrap = burst_values[
        rng.integers(0, len(burst_values), size=(bootstrap_replicates, len(burst_values)))
    ].mean(axis=1)
    return {
        "budget": budget, "task_rows": task_rows, "burst_rows": burst_rows,
        "macro_held_out_burst_recall_at_58": float(np.mean(burst_values)),
        "burst_bootstrap_95_interval": [
            float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975)),
        ],
        "grouping_unit": "held_out_burst",
        "robustness_run_semantics": (
            "finalist and seed runs within a burst are descriptive robustness checks, "
            "not independent biological replicates"
        ),
        "unmatched_candidates": "unknown_not_negative",
    }


def _completed_factorial_rows(real: RealDataConfig) -> dict[str, dict[str, Any]]:
    geometries = ("none", "spatial", "temporal", "joint_spatiotemporal",
                  "spatial_then_temporal", "temporal_then_spatial")
    result = {}
    for geometry in geometries:
        tag = geometry.upper()
        for shard in range(real.fitting.shard_count):
            root = (real.output_dir / "stages"
                    / (f"S2B_EVENT_STANDARDIZED_SHARD_{shard:02d}_OF_{real.fitting.shard_count:02d}"
                       if geometry == "none" else
                       f"S2C_{tag}_WHITENING_SHARD_{shard:02d}_OF_{real.fitting.shard_count:02d}"))
            validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
            if validation.get("status") != "pass":
                raise RuntimeError(f"unvalidated factorial source: {root}")
            with (root / "fit_results.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line); result[str(row["fit_id"])] = row
    return result


def run_finalist_confirmation(
    real: RealDataConfig, *, preflight_dir: str | Path,
    maximum_finalists: int = 3,
) -> dict[str, Any]:
    """Run or resume every frozen finalist x seed x held-out-burst refit."""
    preflight_root = Path(preflight_dir).expanduser().resolve()
    s3 = real.output_dir / "stages" / "S3_COMPLETE_FACTORIAL_ANALYSIS"
    validation = json.loads((s3 / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "pass":
        raise RuntimeError("complete factorial validation is required")
    factorial = json.loads((s3 / "summary.json").read_text(encoding="utf-8"))
    selected_by_burst = confirmation_fit_ids_by_held_out_burst(
        factorial, maximum_per_fold=maximum_finalists,
    )
    selected_ids = sorted({fit_id for values in selected_by_burst.values() for fit_id in values})
    source_rows = _completed_factorial_rows(real)
    missing = sorted(set(selected_ids) - set(source_rows))
    if missing: raise RuntimeError(f"selected finalist rows missing: {missing}")
    interval_payload = json.loads(
        (preflight_root / "event_interval_contract.json").read_text(encoding="utf-8")
    )["intervals"]
    intervals = {int(key): (int(value[0]), int(value[1]))
                 for key, value in interval_payload.items()}
    from .config import ICAWhiteningConfig
    from .real_runner import _load_proposals, _signed_review
    from neurobench.experiments.learned_operator_selection.data import load_and_validate_labels
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    proposals = _load_proposals(preflight_root / "common_proposal_universe.tsv")
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    activity_order = activity_priority_order(movie, quiet)
    root = real.output_dir / "stages" / "S4_FINALIST_CONFIRMATION"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "stage": "S4_FINALIST_CONFIRMATION",
        "source_factorial_validation": str((s3 / "validation.json").relative_to(real.output_dir)),
        "selection_rule": "fold_specific_selection_without_held_out_burst_labels",
        "maximum_finalists_per_held_out_fold": maximum_finalists,
        "selected_fit_ids_union_for_interpretability": selected_ids,
        "selected_fit_ids_by_held_out_burst": {
            str(key): value for key, value in selected_by_burst.items()
        },
        "confirmation_seed_offsets": [0, 104729, 209759],
        "held_out_bursts": [1, 2, 3, 4],
        "held_out_event_samples_excluded_from_fit": True,
        "labels_consumed_only_after_candidate_score_hash": True,
        "unmatched_candidates": "unknown_not_negative",
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing finalist confirmation manifest differs")
    if not manifest_path.exists(): _atomic_json(manifest_path, manifest)
    expected = []
    for burst in (1, 2, 3, 4):
        for fit_id in selected_by_burst[burst]:
            original = int(source_rows[fit_id]["seed"])
            for seed in confirmation_seed_set(original):
                expected.append((fit_id, seed, burst))
    started = time.monotonic(); completed = 0
    for fit_id, seed, burst in expected:
        name = f"{fit_id}__seed_{seed}__heldout_{burst}.json"
        destination = root / "refits" / name
        if destination.is_file():
            completed += 1; continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = refit_finalist_fold(
            movie, source_rows[fit_id], proposals, labels, intervals, parent, real,
            confirmation_seed=seed, held_out_burst=burst,
            activity_order=activity_order,
        )
        _atomic_json(destination, result); completed += 1
        _atomic_json(root / "progress.json", {
            "status": "running", "completed": completed, "expected": len(expected),
            "last_refit": name, "elapsed_seconds": time.monotonic() - started,
        })
    stability = []
    for burst in (1, 2, 3, 4):
        for fit_id in selected_by_burst[burst]:
            original = int(source_rows[fit_id]["seed"])
            payloads = [json.loads((root / "refits"
                        / f"{fit_id}__seed_{seed}__heldout_{burst}.json").read_text())
                        for seed in confirmation_seed_set(original)]
            stability.append({
                "source_fit_id": fit_id, "held_out_burst": burst,
                "seed_stability": seed_stability([
                    np.asarray(payload["model"]["demixing"]) for payload in payloads
                ]),
            })
    protected_payloads = [json.loads((root / "refits"
                          / f"{fit_id}__seed_{seed}__heldout_{burst}.json").read_text())
                          for fit_id, seed, burst in expected]
    protected_performance = summarize_protected_refits(protected_payloads)
    summary = {
        "schema_version": 1, "status": "complete", "refit_count": completed,
        "expected_refit_count": len(expected), "selected_fit_ids": selected_ids,
        "selected_fit_ids_by_held_out_burst": {
            str(key): value for key, value in selected_by_burst.items()
        },
        "held_out_bursts_by_fit_id": {
            fit_id: [burst for burst, values in selected_by_burst.items() if fit_id in values]
            for fit_id in selected_ids
        },
        "stability": stability,
        "protected_held_out_performance": protected_performance,
        "individual_component_interpretation_allowed_for_all": all(
            row["seed_stability"]["individual_component_interpretation_allowed"]
            for row in stability
        ),
        "claim_scope": "fold_specific_strict_held_out_burst_within_recording_confirmation",
        "promotion_status": "pending_scientific_audit",
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", {
        "status": "pass" if completed == len(expected) else "fail",
        "expected_refit_count": len(expected), "observed_refit_count": completed,
    })
    _atomic_json(root / "artifact_index.json", {"artifacts": [
        {"path": "run_manifest.json", "role": "frozen_selection_and_refit_contract"},
        {"path": "refits/*.json", "role": "model_control_metric_and_trace_evidence"},
        {"path": "summary.json", "role": "seed_and_fold_stability_summary"},
        {"path": "validation.json", "role": "exact_refit_coverage"},
    ]})
    _atomic_json(root / "progress.json", {"status": "complete", "completed": completed,
                                           "expected": len(expected)})
    return summary
