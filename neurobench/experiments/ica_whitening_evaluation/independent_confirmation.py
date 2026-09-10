"""Label-blind candidate contract for independent-recording confirmation."""
from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from neurobench.experiments.frame_difference import _atomic_json
from neurobench.metrics.sparse_detection import (
    extract_separated_local_maxima, temporal_pool,
)

from .real_preflight import _lane_stacks
from .finalist_diagnostics import approximate_source_snr, reconstruction_integrity
from .finalist_diagnostics import seed_stability
from .model import activity_priority_order, extract_patch_observations, fit_ica, model_summary
from .operators import apply_whitening
from .real_runner import _observations_at_coordinates
from .responses import component_response_summary
from .sparse_whitening import _reflect_indices


INDEPENDENT_CANDIDATE_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "reference_frame_count": 100,
    "reference_semantics": "predeclared_initial_frames_not_asserted_event_free",
    "block_frames": 50,
    "block_semantics": "fixed_nonoverlapping_one_second_blocks_at_50_hz",
    "minimum_final_block_frames": 25,
    "lanes": ["raw_activity", "signed_temporal_difference", "spatial_highpass"],
    "temporal_pool": "lme0.25",
    "per_lane_per_block": 256,
    "proposal_nms_distance_px": 3,
    "union_separation_px": 3,
    "score_window_frames": 21,
    "score_window_semantics": "centered_on_label_blind_lane_peak",
    "evaluation_budgets_per_block": [10, 20, 58, 100, 200],
    "spatial_match_radius_px": 6,
    "temporal_match": "candidate_peak_frame_inside_manual_positive_interval",
    "unmatched_candidates": "unknown_not_negative",
    "label_access_order": [
        "construct_candidate_universe", "fit_unsupervised_finalist_on_recording",
        "freeze_complete_candidate_score_hash", "join_sparse_positive_labels",
    ],
}


def candidate_contract_digest() -> str:
    payload = json.dumps(
        INDEPENDENT_CANDIDATE_CONTRACT, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_independent_movie(movie: np.ndarray) -> np.ndarray:
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("independent movie must be finite TYX")
    count = int(INDEPENDENT_CANDIDATE_CONTRACT["reference_frame_count"])
    if len(values) < count:
        raise ValueError("independent movie is shorter than the reference contract")
    reference = values[:count]
    baseline = np.median(reference, axis=0)
    low, high = np.percentile(reference[:, ::4, ::4], [1, 99.9])
    scale = max(float(high - low), 1.0)
    return ((values - baseline[None]) / scale).astype(np.float32)


def build_independent_candidate_universe(movie: np.ndarray) -> tuple[list[dict[str, Any]], str]:
    """Construct fixed-block proposals without accepting labels or label-derived windows."""
    values = normalize_independent_movie(movie)
    lanes = _lane_stacks(values)
    block_frames = int(INDEPENDENT_CANDIDATE_CONTRACT["block_frames"])
    minimum = int(INDEPENDENT_CANDIDATE_CONTRACT["minimum_final_block_frames"])
    lane_names = list(INDEPENDENT_CANDIDATE_CONTRACT["lanes"])
    nms = int(INDEPENDENT_CANDIDATE_CONTRACT["proposal_nms_distance_px"])
    separation2 = int(INDEPENDENT_CANDIDATE_CONTRACT["union_separation_px"]) ** 2
    limit = int(INDEPENDENT_CANDIDATE_CONTRACT["per_lane_per_block"])
    rows: list[dict[str, Any]] = []
    block_id = 0
    for start in range(0, len(values), block_frames):
        stop = min(start + block_frames, len(values))
        if stop - start < minimum:
            break
        block_id += 1
        peaks = {}
        for lane in lane_names:
            pooled = temporal_pool(
                lanes[lane][start:stop],
                str(INDEPENDENT_CANDIDATE_CONTRACT["temporal_pool"]),
            )
            peaks[lane] = extract_separated_local_maxima(pooled, nms, limit=limit)
        selected: list[tuple[int, int]] = []
        depth = 0
        while True:
            added = False
            for lane in lane_names:
                if depth >= len(peaks[lane]):
                    continue
                added = True
                score, x, y = peaks[lane][depth]
                if any((x - old_x) ** 2 + (y - old_y) ** 2 <= separation2
                       for old_x, old_y in selected):
                    continue
                selected.append((x, y))
                peak = start + int(np.argmax(lanes[lane][start:stop, y, x]))
                rows.append({
                    "candidate_id": f"ind_b{block_id:03d}_{len(selected):04d}",
                    "block_id": block_id, "block_start_frame": start,
                    "block_stop_frame_exclusive": stop, "peak_frame": peak,
                    "x_px": int(x), "y_px": int(y), "origin_lane": lane,
                    "origin_score": float(score),
                })
            if not added:
                break
            depth += 1
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def independent_score_digest(scores: np.ndarray) -> str:
    values = np.asarray(scores, dtype="<f8")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("independent candidate scores must be a finite vector")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _independent_positives(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    block_frames = int(INDEPENDENT_CANDIDATE_CONTRACT["block_frames"])
    rows = []
    for annotation in annotations:
        for index, interval in enumerate(annotation["spike_intervals"]):
            start, stop = int(interval["start_frame"]), int(interval["end_frame"])
            center = (start + stop) // 2
            rows.append({
                "positive_id": f"{annotation['annotation_id']}:interval_{index + 1:02d}",
                "annotation_id": str(annotation["annotation_id"]),
                "x_px": float(annotation["crop_x"]), "y_px": float(annotation["crop_y"]),
                "start_frame": start, "end_frame": stop,
                "assigned_block_id": center // block_frames + 1,
            })
    return rows


def independent_sparse_positive_metrics(
    candidates: list[dict[str, Any]], scores: np.ndarray,
    annotations: list[dict[str, Any]], *, score_sha256_before_label_join: str,
) -> dict[str, Any]:
    """Join labels only after caller proves the complete score vector is frozen."""
    values = np.asarray(scores, dtype=np.float64)
    digest = independent_score_digest(values)
    if score_sha256_before_label_join != digest:
        raise RuntimeError("candidate score hash does not match the frozen score vector")
    if len(candidates) != len(values):
        raise ValueError("candidate and score counts differ")
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("independent candidate IDs are not unique")
    positives = _independent_positives(annotations)
    budgets = list(INDEPENDENT_CANDIDATE_CONTRACT["evaluation_budgets_per_block"])
    radius2 = float(INDEPENDENT_CANDIDATE_CONTRACT["spatial_match_radius_px"]) ** 2
    block_ids = sorted({int(row["assigned_block_id"]) for row in positives})
    folds = []
    reciprocal_ranks = []
    for block_id in block_ids:
        candidate_indices = [index for index, row in enumerate(candidates)
                             if int(row["block_id"]) == block_id]
        ranking = sorted(candidate_indices, key=lambda index: (-float(values[index]), identifiers[index]))
        block_positives = [row for row in positives if int(row["assigned_block_id"]) == block_id]

        def eligible(candidate: dict[str, Any], positive: dict[str, Any]) -> bool:
            spatial = ((float(candidate["x_px"]) - positive["x_px"]) ** 2
                       + (float(candidate["y_px"]) - positive["y_px"]) ** 2 <= radius2)
            temporal = (positive["start_frame"] <= int(candidate["peak_frame"])
                        <= positive["end_frame"])
            return bool(spatial and temporal)

        for positive in block_positives:
            ranks = [rank for rank, index in enumerate(ranking, 1)
                     if eligible(candidates[index], positive)]
            reciprocal_ranks.append(0.0 if not ranks else 1.0 / min(ranks))
        budget_rows = []
        for budget in budgets:
            unmatched = set(range(len(block_positives)))
            matches = []
            for rank, index in enumerate(ranking[:budget], 1):
                possible = [positive_index for positive_index in unmatched
                            if eligible(candidates[index], block_positives[positive_index])]
                if not possible:
                    continue
                chosen = min(possible, key=lambda positive_index: (
                    (float(candidates[index]["x_px"]) - block_positives[positive_index]["x_px"]) ** 2
                    + (float(candidates[index]["y_px"]) - block_positives[positive_index]["y_px"]) ** 2,
                    block_positives[positive_index]["positive_id"],
                ))
                unmatched.remove(chosen)
                matches.append({
                    "candidate_id": identifiers[index],
                    "positive_id": block_positives[chosen]["positive_id"], "rank": rank,
                })
            budget_rows.append({
                "budget": budget, "matched_known_positives": len(matches),
                "known_positive_count": len(block_positives),
                "recall": len(matches) / len(block_positives) if block_positives else None,
                "matches": matches,
            })
        folds.append({
            "block_id": block_id, "candidate_count": len(candidate_indices),
            "known_positive_count": len(block_positives), "budgets": budget_rows,
        })
    pooled = []
    for budget in budgets:
        selected = [next(item for item in fold["budgets"] if item["budget"] == budget)
                    for fold in folds]
        matched = sum(item["matched_known_positives"] for item in selected)
        total = sum(item["known_positive_count"] for item in selected)
        pooled.append({
            "budget_per_block": budget, "matched_known_positives": matched,
            "known_positive_count": total, "pooled_known_positive_recall": matched / total,
            "macro_block_known_positive_recall": float(np.mean([item["recall"] for item in selected])),
        })
    return {
        "candidate_score_sha256_before_label_join": digest,
        "known_positive_count": len(positives), "positive_block_count": len(folds),
        "folds": folds, "pooled_budgets": pooled,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "unmatched_candidates": "unknown_not_negative",
        "precision_specificity_and_false_positive_rate": "not_identified",
        "claim_scope": "single_independent_recording_sparse_positive_retrieval",
    }


def fit_and_score_independent_finalist(
    movie: np.ndarray, candidates: list[dict[str, Any]], specification: dict[str, Any],
    *, maximum_fit_samples: int = 4096, activity_fraction: float = .5,
    frame_period_ms: float = 20.0, score_chunk_candidates: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit a frozen hyperparameter specification without labels, then score all candidates."""
    values = normalize_independent_movie(movie)
    reference = int(INDEPENDENT_CANDIDATE_CONTRACT["reference_frame_count"])
    activity_order = activity_priority_order(values, reference)
    raw_fit = extract_patch_observations(
        values, family=str(specification["family"]),
        spatial_width=specification.get("spatial_width_px"),
        temporal_width=specification.get("temporal_width_frames"),
        causality=str(specification["causality"]), maximum_samples=maximum_fit_samples,
        seed=int(specification["seed"]), quiet_frames=reference,
        activity_fraction=activity_fraction, activity_order=activity_order,
    )
    whitening = apply_whitening(
        values, reference, specification, maximum_samples=maximum_fit_samples,
    )
    fit_values = _observations_at_coordinates(
        whitening.output, raw_fit.times, raw_fit.rows, raw_fit.columns, specification,
    )
    model = fit_ica(
        fit_values, specification, maximum_fit_samples=maximum_fit_samples,
    )
    reference_sources = model.sources[:, raw_fit.times < reference]
    if reference_sources.shape[1] < 32:
        raise RuntimeError("too few predeclared reference samples for independent calibration")
    center = np.median(reference_sources, axis=1)
    scale = 1.4826 * np.median(
        np.abs(reference_sources - center[:, None]), axis=1
    )
    scale = np.maximum(scale, np.finfo(float).eps)
    window = int(INDEPENDENT_CANDIDATE_CONTRACT["score_window_frames"])
    before = window // 2
    offsets = np.arange(-before, before + 1, dtype=np.int64)
    scores = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), score_chunk_candidates):
        selected = candidates[start:start + score_chunk_candidates]
        peaks = np.asarray([int(row["peak_frame"]) for row in selected], dtype=np.int64)
        times = _reflect_indices(
            peaks[:, None] + offsets[None], len(values), repeat_edge=False,
        ).reshape(-1)
        rows = np.repeat([int(row["y_px"]) for row in selected], window)
        columns = np.repeat([int(row["x_px"]) for row in selected], window)
        observations = _observations_at_coordinates(
            whitening.output, times, rows, columns, specification,
        )
        sources = model.demixing @ (observations - model.whitening.mean[:, None])
        sources = sources.reshape(model.sources.shape[0], len(selected), window)
        standardized = np.abs(
            (sources - center[:, None, None]) / scale[:, None, None]
        )
        evidence = np.max(standardized, axis=0)
        tau = .25
        scores[start:start + len(selected)] = tau * (
            logsumexp(evidence / tau, axis=1) - np.log(window)
        )
    digest = independent_score_digest(scores)
    response = component_response_summary(
        model.demixing, family=str(specification["family"]),
        spatial_width=specification.get("spatial_width_px"),
        temporal_width=specification.get("temporal_width_frames"),
        frame_period_ms=frame_period_ms,
    )
    summary = {
        "source_fit_id": specification.get("fit_id"),
        "candidate_count": len(candidates),
        "candidate_score_sha256_before_label_join": digest,
        "labels_accessed": False,
        "model": model_summary(model),
        "external_whitening": {
            "diagnostics": whitening.diagnostics,
            "kernels": [kernel.tolist() for kernel in whitening.kernels],
        },
        "calibration": {
            "reference_frame_count": reference,
            "reference_semantics": INDEPENDENT_CANDIDATE_CONTRACT["reference_semantics"],
            "sample_count": int(reference_sources.shape[1]),
            "component_center": center.tolist(), "component_mad_scale": scale.tolist(),
        },
        "component_response": response,
        "reconstruction_integrity": reconstruction_integrity(
            fit_values, model.sources, model.mixing, model.whitening.mean,
        ),
        "approximate_snr": approximate_source_snr(
            model.sources, raw_fit.times, reference,
        ),
        "score_semantics": "reference_mad_standardized_max_abs_component_lme0.25_over_21_frames",
        "component_identity_transfer": "prohibited",
        "unmatched_candidates": "unknown_not_negative",
    }
    return scores, summary


def freeze_independent_confirmation_contract(
    *, independent_preflight: str | Path, destination: str | Path,
) -> dict[str, Any]:
    preflight = Path(independent_preflight).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_payload.get("status") != "ready_for_frozen_finalist":
        raise RuntimeError("eligible independent-recording preflight is required")
    payload = {
        "schema_version": 1, "status": "frozen_awaiting_within_recording_finalists",
        "candidate_contract": INDEPENDENT_CANDIDATE_CONTRACT,
        "candidate_contract_sha256": candidate_contract_digest(),
        "independent_preflight_path": str(preflight),
        "eligible_recordings": preflight_payload["eligible_recordings"],
        "finalist_policy": (
            "reuse S4-selected hyperparameters; refit whitening and ICA label-blind on the "
            "independent recording; do not transfer component identity"
        ),
        "claim_limit": "single independent recording; sparse-positive retrieval only",
    }
    target.mkdir(parents=True, exist_ok=False)
    _atomic_json(target / "contract.json", payload)
    return payload


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_independent_confirmation(real: Any, *, contract_dir: str | Path) -> dict[str, Any]:
    """Run the frozen finalist x three-seed independent confirmation package."""
    from tifffile import memmap
    from .real_finalist_confirmation import (
        _completed_factorial_rows, confirmation_seed_set,
    )

    contract_root = Path(contract_dir).expanduser().resolve()
    contract = json.loads((contract_root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_awaiting_within_recording_finalists":
        raise RuntimeError("frozen independent confirmation contract is required")
    s4 = real.output_dir / "stages" / "S4_FINALIST_CONFIRMATION"
    if json.loads((s4 / "validation.json").read_text()).get("status") != "pass":
        raise RuntimeError("validated within-recording finalists are required")
    s4_summary = json.loads((s4 / "summary.json").read_text(encoding="utf-8"))
    selected_ids = list(map(str, s4_summary["selected_fit_ids"]))
    source_rows = _completed_factorial_rows(real)
    preflight_path = Path(contract["independent_preflight_path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    eligible = list(preflight["eligible_recordings"])
    if eligible != ["15 right"]:
        raise RuntimeError(f"independent eligibility changed: {eligible}")
    recording = next(row for row in preflight["recordings"] if row["video_id"] == eligible[0])
    video_path = Path(recording["cropped_video"]["path"])
    if _hash_file(video_path) != recording["cropped_video"]["sha256"]:
        raise RuntimeError("independent video hash changed after preflight")
    movie = memmap(video_path, mode="r")
    if list(movie.shape) != recording["shape_tyx"]:
        raise RuntimeError("independent video shape changed after preflight")
    root = real.output_dir / "stages" / "S6_INDEPENDENT_CONFIRMATION"
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / "candidate_universe.tsv"
    candidate_manifest_path = root / "candidate_manifest.json"
    if not candidate_path.is_file():
        candidates, candidate_digest = build_independent_candidate_universe(movie)
        with candidate_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidates[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(candidates)
        _atomic_json(candidate_manifest_path, {
            "status": "frozen_before_model_scoring_and_label_join",
            "candidate_count": len(candidates), "candidate_digest": candidate_digest,
            "contract_sha256": contract["candidate_contract_sha256"],
            "labels_accessed": False,
        })
    else:
        with candidate_path.open(encoding="utf-8", newline="") as stream:
            candidates = [{
                **row, "block_id": int(row["block_id"]),
                "block_start_frame": int(row["block_start_frame"]),
                "block_stop_frame_exclusive": int(row["block_stop_frame_exclusive"]),
                "peak_frame": int(row["peak_frame"]), "x_px": int(row["x_px"]),
                "y_px": int(row["y_px"]), "origin_score": float(row["origin_score"]),
            } for row in csv.DictReader(stream, delimiter="\t")]
        manifest = json.loads(candidate_manifest_path.read_text())
        payload = json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != manifest["candidate_digest"]:
            raise RuntimeError("independent candidate universe changed")
    expected = []
    for fit_id in selected_ids:
        original_seed = int(source_rows[fit_id]["seed"])
        expected.extend((fit_id, seed) for seed in confirmation_seed_set(original_seed))
    score_root = root / "label_blind_scores"; score_root.mkdir(exist_ok=True)
    for fit_id, seed in expected:
        stem = f"{fit_id}__seed_{seed}"
        score_path, model_path = score_root / f"{stem}.npy", score_root / f"{stem}.json"
        if score_path.is_file() and model_path.is_file():
            scores = np.load(score_path, allow_pickle=False)
            model_payload = json.loads(model_path.read_text())
            if independent_score_digest(scores) != model_payload[
                "candidate_score_sha256_before_label_join"
            ]:
                raise RuntimeError(f"independent score hash mismatch: {stem}")
            continue
        specification = {**source_rows[fit_id], "seed": int(seed)}
        scores, model_payload = fit_and_score_independent_finalist(
            movie, candidates, specification,
            maximum_fit_samples=real.fitting.maximum_fit_samples,
            activity_fraction=real.fitting.activity_fraction,
            frame_period_ms=1000.0 / float(recording["frame_rate_hz"]),
        )
        np.save(score_path, scores, allow_pickle=False)
        _atomic_json(model_path, model_payload)
        _atomic_json(root / "progress.json", {
            "status": "label_blind_scoring", "completed_score_files": sum(
                (score_root / f"{candidate_fit}__seed_{candidate_seed}.npy").is_file()
                for candidate_fit, candidate_seed in expected
            ), "expected_score_files": len(expected), "labels_accessed": False,
        })
    # This is the first annotation-manifest read in this function, after every score is frozen.
    annotation_path = Path(preflight["source_contracts"]["annotation_manifest"]["path"])
    if _hash_file(annotation_path) != preflight["source_contracts"]["annotation_manifest"]["sha256"]:
        raise RuntimeError("independent annotations changed after preflight")
    annotations_all = json.loads(annotation_path.read_text(encoding="utf-8"))["annotations"]
    annotations = [row for row in annotations_all if row["video_id"] == eligible[0]]
    metric_root = root / "sparse_positive_metrics"; metric_root.mkdir(exist_ok=True)
    for fit_id, seed in expected:
        stem = f"{fit_id}__seed_{seed}"
        destination = metric_root / f"{stem}.json"
        if destination.is_file():
            continue
        scores = np.load(score_root / f"{stem}.npy", allow_pickle=False)
        digest = independent_score_digest(scores)
        metrics = independent_sparse_positive_metrics(
            candidates, scores, annotations,
            score_sha256_before_label_join=digest,
        )
        _atomic_json(destination, metrics)
    finalists = []
    for fit_id in selected_ids:
        seeds = confirmation_seed_set(int(source_rows[fit_id]["seed"]))
        metrics = [json.loads((metric_root / f"{fit_id}__seed_{seed}.json").read_text())
                   for seed in seeds]
        models = [json.loads((score_root / f"{fit_id}__seed_{seed}.json").read_text())
                  for seed in seeds]
        primary = []
        for item in metrics:
            primary.append(next(row for row in item["pooled_budgets"]
                                if row["budget_per_block"] == 58))
        finalists.append({
            "source_fit_id": fit_id, "seeds": list(seeds),
            "known_positive_count": metrics[0]["known_positive_count"],
            "mean_pooled_known_positive_recall_at_58": float(np.mean([
                row["pooled_known_positive_recall"] for row in primary
            ])),
            "seed_range_pooled_known_positive_recall_at_58": [
                float(min(row["pooled_known_positive_recall"] for row in primary)),
                float(max(row["pooled_known_positive_recall"] for row in primary)),
            ],
            "mean_reciprocal_rank": float(np.mean([
                item["mean_reciprocal_rank"] for item in metrics
            ])),
            "seed_stability": seed_stability([
                np.asarray(item["model"]["demixing"]) for item in models
            ]),
        })
    expected_stems = {f"{fit_id}__seed_{seed}" for fit_id, seed in expected}
    observed_score_stems = {path.stem for path in score_root.glob("*.npy")}
    observed_model_stems = {path.stem for path in score_root.glob("*.json")}
    observed_metric_stems = {path.stem for path in metric_root.glob("*.json")}
    exact_coverage = (
        observed_score_stems == expected_stems
        and observed_model_stems == expected_stems
        and observed_metric_stems == expected_stems
    )
    summary = {
        "schema_version": 1, "status": "complete",
        "recording_id": eligible[0], "finalist_count": len(selected_ids),
        "fit_seed_count": len(expected), "finalists": finalists,
        "exact_artifact_coverage": exact_coverage,
        "candidate_count": len(candidates), "labels_loaded_after_all_score_files": True,
        "unmatched_candidates": "unknown_not_negative",
        "precision_specificity_and_false_positive_rate": "not_identified",
        "claim_scope": "single_independent_recording_sparse_positive_retrieval",
    }
    _atomic_json(root / "summary.json", summary)
    validation = {
        "status": "pass" if exact_coverage else "fail",
        "expected_stems": sorted(expected_stems),
        "missing_score_stems": sorted(expected_stems - observed_score_stems),
        "extra_score_stems": sorted(observed_score_stems - expected_stems),
        "missing_model_stems": sorted(expected_stems - observed_model_stems),
        "extra_model_stems": sorted(observed_model_stems - expected_stems),
        "missing_metric_stems": sorted(expected_stems - observed_metric_stems),
        "extra_metric_stems": sorted(observed_metric_stems - expected_stems),
    }
    _atomic_json(root / "validation.json", validation)
    _atomic_json(root / "artifact_index.json", {"artifacts": [
        {"path": "candidate_universe.tsv", "role": "frozen_label_blind_candidates"},
        {"path": "candidate_manifest.json", "role": "candidate_hash_and_contract"},
        {"path": "label_blind_scores/*.npy", "role": "complete_pre_label_score_vectors"},
        {"path": "label_blind_scores/*.json", "role": "independent_refit_models_and_diagnostics"},
        {"path": "sparse_positive_metrics/*.json", "role": "post_hash_label_metrics"},
        {"path": "summary.json", "role": "seed_stability_and_independent_findings"},
        {"path": "validation.json", "role": "exact_artifact_coverage"},
    ]})
    _atomic_json(root / "progress.json", {
        "status": "complete" if exact_coverage else "failed_exact_coverage",
        "completed": len(observed_metric_stems), "expected": len(expected),
    })
    if not exact_coverage:
        raise RuntimeError(f"independent confirmation artifact coverage failed: {validation}")
    return summary
