"""Guarded, checkpointed event-balanced CS-Parzen ICA sweep runner."""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from neurobench.algorithms.pairwise_separation import (
    fit_cs_parzen_ica,
    quiet_difference_stats,
    standardized_positive_mask,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import evaluate_lane
from neurobench.experiments.pairwise_separation.sampling import (
    causal_preprocess,
    uniform_anatomy_mask,
)

from .artifacts import (
    SCIENTIFIC_STATUS,
    atomic_json,
    evaluate_gate_c,
    write_figures,
    write_fit_tables,
    write_results_note,
)
from .config import EventWeightedCSParzenConfig
from .fitting import (
    CanonicalFit,
    apply_whitening,
    canonicalize_fit,
    fit_weighted_batch,
    fit_weighted_whitening_2d,
    natural_objective,
    whitening_diagnostics,
)
from .sample_weights import (
    build_weighted_pair_batch,
    repeat_equivalent,
)
from .sampling import (
    build_fold_sample_pools,
    extract_pair_samples,
    sample_natural_indices,
)
from .videos import render_selected_videos


def _heartbeat(path: Path, stage: str, **payload: Any) -> None:
    from datetime import datetime, timezone

    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _copy_preflight(
    preflight_dir: Path,
    destination: Path,
    config: EventWeightedCSParzenConfig,
) -> dict[str, Any]:
    payload = json.loads((preflight_dir / "preflight.json").read_text())
    resolved = json.loads((preflight_dir / "config.resolved.json").read_text())
    expected = json.loads(json.dumps(config.to_dict()))
    if not payload.get("ready") or resolved != expected:
        raise ValueError("preflight does not match this ready configuration")
    shutil.copy2(preflight_dir / "preflight.json", destination / "preflight.json")
    shutil.copy2(
        preflight_dir / "label_projection_overlay.png",
        destination / "label_projection_overlay.png",
    )
    return payload


def _circular_shift_degrees(angle: float, baseline: float) -> float:
    return float((angle - baseline + 90.0) % 180.0 - 90.0)


def _apply_innovation(
    filtered: np.ndarray,
    canonical: CanonicalFit,
    *,
    frame_chunk: int = 16,
) -> np.ndarray:
    output = np.zeros_like(filtered, dtype=np.float32)
    for start in range(1, len(filtered), frame_chunk):
        stop = min(len(filtered), start + frame_chunk)
        samples = np.column_stack(
            (
                filtered[start - 1 : stop - 1].reshape(-1),
                filtered[start:stop].reshape(-1),
            )
        )
        whitened = apply_whitening(samples, canonical.whitening)
        separated = canonical.fit.demixing @ whitened
        output[start:stop] = (
            separated[canonical.innovation_component]
            * canonical.innovation_sign
        ).reshape(stop - start, *filtered.shape[1:])
    return output


def _evaluation_shim(config: EventWeightedCSParzenConfig) -> SimpleNamespace:
    return SimpleNamespace(
        frames=SimpleNamespace(
            review_start_ui=config.source.review_interval_ui[0]
        ),
        evaluation=SimpleNamespace(
            nms_distance_px=config.evaluation.nms_distance_px,
            primary_match_radius_px=config.evaluation.primary_match_radius_px,
            quiet_false_peaks_per_map=config.evaluation.quiet_false_peaks_per_map,
        ),
    )


def _heldout_detection(
    filtered: np.ndarray,
    canonical: CanonicalFit,
    labels: list[dict[str, Any]],
    config: EventWeightedCSParzenConfig,
    heldout_event_id: int,
) -> tuple[float, int]:
    activity = _apply_innovation(filtered, canonical)
    quiet_count = (
        config.source.quiet_interval_ui[1]
        - config.source.quiet_interval_ui[0]
        + 1
    )
    stats = quiet_difference_stats(
        activity[1:quiet_count],
        floor_percentile=config.evaluation.quiet_mad_floor_percentile,
    )
    z, mask = standardized_positive_mask(
        activity,
        stats,
        config.evaluation.primary_z_threshold,
        undefined_leading_frames=1,
    )
    metrics, _, _ = evaluate_lane(
        "event_weighted_cs_parzen",
        mask,
        labels,
        _evaluation_shim(config),
        binary=True,
        tie_values=np.maximum(z, 0),
    )
    fold = next(
        row
        for row in metrics["outer_folds"]
        if int(row["burst_id"]) == heldout_event_id
    )
    return float(fold["recall"]), int(fold["candidates"])


def _baseline_parity(
    filtered: np.ndarray,
    anatomy_mask: np.ndarray,
    config: EventWeightedCSParzenConfig,
) -> dict[str, Any]:
    start, stop = config.source.review_interval_ui
    evidence = config.source.baseline_evidence_dir
    if evidence is None:
        confirmation_samples = config.sampling.confirmation_samples
        screen_samples = config.sampling.screen_samples
        block_rows = config.parzen.kernel_block_rows
        coarse_step = config.angle_search.coarse_step_degrees
        refine_half_width = config.angle_search.refine_half_width_degrees
        refine_step = config.angle_search.refine_step_degrees
    else:
        confirmation_samples = 4096
        screen_samples = 1024
        block_rows = 256
        coarse_step = 3.0
        refine_half_width = 3.0
        refine_step = 0.25
    indices = sample_natural_indices(
        range(start + 1, stop + 1),
        anatomy_mask,
        sample_count=confirmation_samples,
        seed=config.sampling.seed,
    )
    samples = extract_pair_samples(filtered, indices, review_start_ui=start)
    whitened, whitening = fit_weighted_whitening_2d(
        samples,
        np.ones(len(samples), dtype=np.float64),
        eigenvalue_floor_ratio=config.whitening.eigenvalue_floor_ratio,
    )
    screen = whitened[:, :screen_samples]
    fit = fit_cs_parzen_ica(
        screen,
        whitened,
        bandwidth=config.parzen.bandwidth,
        block_rows=block_rows,
        screen_step_degrees=coarse_step,
        refine_half_width_degrees=refine_half_width,
        refine_step_degrees=refine_step,
    )
    canonical = canonicalize_fit(fit, whitening)
    result = {
        "status": "measured_without_external_evidence",
        "angle_degrees": fit.diagnostics["selected_angle_degrees"],
        "objective": fit.objective,
        "objective_evaluations": fit.iterations,
        "cosine_to_derivative": canonical.cosine_to_derivative,
        "expected": None,
        "tolerances": {
            "objective_absolute": 1e-8,
            "angle_degrees": 0.25,
            "cosine_to_derivative": 1e-6,
        },
    }
    if evidence is None:
        return result
    fit_path = evidence / "source_evidence" / "fit.json"
    metric_path = evidence / "preliminary_metrics.json"
    if not fit_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError("baseline evidence packet is incomplete")
    expected_fit = json.loads(fit_path.read_text())
    expected_metrics = json.loads(metric_path.read_text())
    expected = {
        "angle_degrees": expected_fit["diagnostics"]["selected_angle_degrees"],
        "objective": expected_fit["objective_value"],
        "objective_evaluations": expected_fit["iterations"],
        "cosine_to_derivative": expected_metrics["direction"][
            "absolute_cosine_to_derivative"
        ],
    }
    checks = {
        "angle": abs(result["angle_degrees"] - expected["angle_degrees"]) <= 0.25,
        "objective": abs(result["objective"] - expected["objective"]) <= 1e-8,
        "evaluations": result["objective_evaluations"]
        == expected["objective_evaluations"],
        "cosine": abs(
            result["cosine_to_derivative"]
            - expected["cosine_to_derivative"]
        )
        <= 1e-6,
    }
    result.update(
        {
            "status": "passed" if all(checks.values()) else "failed",
            "expected": expected,
            "checks": checks,
        }
    )
    return result


def _fit_path(root: Path, fold: int, lane: str, alpha: float) -> Path:
    alpha_slug = f"{alpha:.6f}".replace(".", "p")
    return root / "fits" / f"fold_{fold}" / lane / f"alpha_{alpha_slug}.json"


def _fit_one(
    *,
    lane: str,
    weight_mode: str,
    whitening_mode: str,
    alpha: float,
    fold: int,
    pools,
    filtered: np.ndarray,
    anatomy_mask: np.ndarray,
    labels: list[dict[str, Any]],
    config: EventWeightedCSParzenConfig,
) -> tuple[dict[str, Any], CanonicalFit]:
    review_start = config.source.review_interval_ui[0]
    natural_screen = extract_pair_samples(
        filtered, pools.natural_screen_indices, review_start_ui=review_start
    )
    natural_confirm = extract_pair_samples(
        filtered, pools.natural_confirm_indices, review_start_ui=review_start
    )
    event_screen = extract_pair_samples(
        filtered, pools.event_screen_indices, review_start_ui=review_start
    )
    event_confirm = extract_pair_samples(
        filtered, pools.event_confirm_indices, review_start_ui=review_start
    )
    screen = build_weighted_pair_batch(
        natural_screen,
        pools.natural_screen_indices,
        event_screen,
        pools.event_screen_indices,
        alpha=alpha,
    )
    confirmation = build_weighted_pair_batch(
        natural_confirm,
        pools.natural_confirm_indices,
        event_confirm,
        pools.event_confirm_indices,
        alpha=alpha,
    )
    started = time.perf_counter()
    canonical = fit_weighted_batch(
        screen,
        confirmation,
        natural_confirm,
        whitening_mode=whitening_mode,
        bandwidth=config.parzen.bandwidth,
        block_rows=config.parzen.kernel_block_rows,
        coarse_step_degrees=config.angle_search.coarse_step_degrees,
        refine_half_width_degrees=config.angle_search.refine_half_width_degrees,
        refine_step_degrees=config.angle_search.refine_step_degrees,
        eigenvalue_floor_ratio=config.whitening.eigenvalue_floor_ratio,
        kernel_dtype=np.dtype(config.parzen.kernel_dtype),
    )
    interval = config.source.burst_intervals_ui[fold]
    holdout_indices = sample_natural_indices(
        range(interval[0], interval[1] + 1),
        anatomy_mask,
        sample_count=config.sampling.confirmation_samples,
        seed=config.sampling.seed + 50021 + fold,
    )
    holdout_samples = extract_pair_samples(
        filtered, holdout_indices, review_start_ui=review_start
    )
    holdout_objective = natural_objective(
        canonical,
        holdout_samples,
        bandwidth=config.parzen.bandwidth,
        block_rows=config.parzen.kernel_block_rows,
        kernel_dtype=np.dtype(config.parzen.kernel_dtype),
    )
    holdout_z = apply_whitening(holdout_samples, canonical.whitening)
    holdout_output = (
        canonical.fit.demixing @ holdout_z
    )[canonical.innovation_component] * canonical.innovation_sign
    derivative = holdout_samples[:, 1] - holdout_samples[:, 0]
    correlation = (
        float(np.corrcoef(holdout_output, derivative)[0, 1])
        if np.std(holdout_output) and np.std(derivative)
        else None
    )
    recall, candidates = _heldout_detection(
        filtered, canonical, labels, config, fold
    )
    common = np.asarray([1.0, 1.0]) / np.sqrt(2)
    row = {
        "fold_id": fold,
        "heldout_event_id": fold,
        "sample_seed": config.sampling.seed,
        "event_screen_seed": config.sampling.seed + 7919,
        "lane": lane,
        "weight_mode": weight_mode,
        "whitening_mode": whitening_mode,
        "alpha": alpha,
        "repeat_equivalent": (
            None
            if alpha >= 1
            else repeat_equivalent(
                alpha,
                len(pools.natural_confirm_indices),
                len(pools.event_confirm_indices),
            )
        ),
        "unique_sample_count": len(confirmation.samples),
        "natural_evaluation_sample_count": len(holdout_samples),
        "weight_sum": confirmation.weight_sum,
        "weight_ess": confirmation.weight_ess,
        "weight_ess_fraction": confirmation.weight_ess / len(confirmation.samples),
        "per_event_mass": confirmation.per_event_mass,
        "angle_degrees": canonical.angle_degrees,
        "angle_shift_from_alpha0_degrees": None,
        "objective_weighted_train": canonical.fit.objective,
        "objective_natural_holdout": holdout_objective,
        "cosine_to_derivative": canonical.cosine_to_derivative,
        "cosine_to_common": float(
            abs(canonical.effective_innovation_direction @ common)
        ),
        "correlation_to_fixed_derivative": correlation,
        "known_label_recall": recall,
        "candidate_count": candidates,
        "precision_identified": False,
        "converged": canonical.fit.converged,
        "objective_evaluations": canonical.fit.iterations,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_vram_mb": None,
        "weight_concentration_warning": (
            confirmation.weight_ess / len(confirmation.samples) < 0.2
        ),
        "whitening": whitening_diagnostics(canonical.whitening),
        "effective_common_direction": canonical.effective_common_direction.tolist(),
        "effective_innovation_direction": canonical.effective_innovation_direction.tolist(),
        "excluded_interval_ui": list(pools.excluded_interval_ui),
        "train_event_ids": list(pools.train_event_ids),
        "screen_sample_identities": [
            list(identity) for identity in screen.indices.identities()
        ],
        "confirmation_sample_identities": [
            list(identity) for identity in confirmation.indices.identities()
        ],
        "objective_by_angle": canonical.fit.diagnostics["objective_by_angle"],
    }
    return row, canonical


def run(
    config: EventWeightedCSParzenConfig,
    *,
    preflight_dir: str | Path,
    authorize_full_spon: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.compute.max_worker_processes)
    root = config.outputs.root_dir
    if root.exists() and not resume:
        raise FileExistsError(f"Output exists: {root}")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        audit = _copy_preflight(Path(preflight_dir).resolve(), root, config)
        atomic_json(root / "config.resolved.json", config.to_dict())
        atomic_json(root / "run_state.json", {"status": "running", "phase": "load"})
    else:
        audit = json.loads((root / "preflight.json").read_text())
        expected = json.loads(json.dumps(config.to_dict()))
        if json.loads((root / "config.resolved.json").read_text()) != expected:
            raise ValueError("resume configuration does not match output")
    progress = root / "progress.jsonl"
    movie = np.load(config.source.movie_path, mmap_mode="r", allow_pickle=False)
    is_full_spon = tuple(movie.shape) == (2359, 340, 573)
    if is_full_spon and not authorize_full_spon:
        atomic_json(
            root / "run_state.json",
            {
                "status": "authorization_required",
                "phase": "not_started",
                "message": "Full Spon run requires --authorize-full-spon.",
            },
        )
        raise PermissionError("Full Spon run requires explicit authorization")
    labels = label_core.load_labels(config.source.labels_path)
    start, stop = config.source.review_interval_ui
    raw = np.asarray(movie[start - 1 : stop], dtype=np.float32)
    _heartbeat(progress, "preprocess", status="started")
    ema_span = 2 / config.preprocessing.ema_alpha - 1
    filtered = causal_preprocess(
        raw, config.preprocessing.gaussian_sigma_px, ema_span
    )
    quiet_count = (
        config.source.quiet_interval_ui[1]
        - config.source.quiet_interval_ui[0]
        + 1
    )
    anatomy, anatomy_summary = uniform_anatomy_mask(filtered[:quiet_count])
    baseline = _baseline_parity(filtered, anatomy, config)
    atomic_json(root / "baseline_parity.json", baseline)
    if baseline["status"] == "failed":
        atomic_json(
            root / "run_state.json",
            {"status": "failed", "phase": "gate_a", "baseline": baseline},
        )
        raise RuntimeError("Gate A failed: alpha=0 baseline did not reproduce")

    rows: list[dict[str, Any]] = []
    existing = sorted((root / "fits").rglob("alpha_*.json")) if (root / "fits").exists() else []
    for path in existing:
        rows.append(json.loads(path.read_text()))
    completed = {
        (row["fold_id"], row["lane"], float(row["alpha"])) for row in rows
    }
    angle_zero = {
        (row["fold_id"], row["lane"]): row["angle_degrees"]
        for row in rows
        if float(row["alpha"]) == 0
    }
    latest_fits: dict[tuple[int, str, float], CanonicalFit] = {}
    for fold in config.fold_ids:
        for mode in config.weighting.modes:
            pools = build_fold_sample_pools(
                heldout_event_id=fold,
                event_intervals_ui=config.source.burst_intervals_ui,
                review_interval_ui=config.source.review_interval_ui,
                anatomy_mask=anatomy,
                labels=labels,
                mode=mode,
                heldout_guard_frames=config.sampling.heldout_guard_frames,
                screen_samples=config.sampling.screen_samples,
                confirmation_samples=config.sampling.confirmation_samples,
                event_screen_max_samples_per_event=config.sampling.event_screen_max_samples_per_event,
                event_confirmation_max_samples_per_event=config.sampling.event_confirmation_max_samples_per_event,
                event_roi_radius_px=config.sampling.event_roi_radius_px,
                seed=config.sampling.seed,
                bad_frames_ui=config.sampling.bad_frames_ui,
            )
            lanes = [(mode, "natural_fixed", alpha) for alpha in config.weighting.alpha_grid]
            if mode == "roi_balanced" and config.whitening.run_weighted_ablation:
                lanes += [
                    ("roi_balanced_weighted_whitening", "weighted", alpha)
                    for alpha in config.whitening.weighted_ablation_alphas
                ]
            if mode == "roi_balanced":
                lanes.insert(0, ("natural", "natural_fixed", 0.0))
            for lane, whitening_mode, alpha in lanes:
                key = (fold, lane, float(alpha))
                if key in completed:
                    continue
                row, canonical = _fit_one(
                    lane=lane,
                    weight_mode="none" if lane == "natural" else mode,
                    whitening_mode=whitening_mode,
                    alpha=float(alpha),
                    fold=fold,
                    pools=pools,
                    filtered=filtered,
                    anatomy_mask=anatomy,
                    labels=labels,
                    config=config,
                )
                zero_key = (fold, lane)
                if alpha == 0:
                    angle_zero[zero_key] = row["angle_degrees"]
                row["angle_shift_from_alpha0_degrees"] = _circular_shift_degrees(
                    row["angle_degrees"], angle_zero[zero_key]
                )
                path = _fit_path(root, fold, lane, float(alpha))
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(path, row)
                rows.append(row)
                latest_fits[key] = canonical
                _heartbeat(
                    progress,
                    "fit",
                    status="complete",
                    fold=fold,
                    lane=lane,
                    alpha=alpha,
                )
                atomic_json(
                    root / "run_state.json",
                    {
                        "status": "running",
                        "phase": "fits",
                        "completed_fits": len(rows),
                    },
                )

    rows.sort(key=lambda row: (row["fold_id"], row["lane"], row["alpha"]))
    write_fit_tables(root, rows)
    gate = evaluate_gate_c(rows)
    atomic_json(root / "stage_gate.json", gate)
    write_figures(root, rows)
    write_results_note(root, rows, gate, baseline)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    manifest = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "scientific_status": SCIENTIFIC_STATUS,
        "source_hashes": {
            row["role"]: row["sha256"] for row in audit["inputs"]
        },
        "git_commit": git_commit,
        "configuration_hash": audit["config_sha256"],
        "random_seeds": {
            "natural_and_event_confirmation": config.sampling.seed,
            "event_screen": config.sampling.seed + 7919,
            "holdout_offset": 50021,
        },
        "split_definitions": audit["splits"],
        "resource_measurements": {
            "peak_rss_mb": max(row["peak_rss_mb"] for row in rows),
            "peak_vram_mb": None,
            "compute_device": "cpu",
        },
        "fit_count": len(rows),
        "stage_gate": gate,
    }
    atomic_json(root / "manifest.json", manifest)
    render_selected_videos(
        root,
        filtered,
        labels,
        rows,
        config.outputs.selected_video_alphas,
        review_start_ui=start,
    )
    summary = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "scientific_status": SCIENTIFIC_STATUS,
        "fit_count": len(rows),
        "baseline_parity": baseline["status"],
        "gate_c_passed": gate["passed"],
        "spatial_extension_launched": False,
        "anatomy": anatomy_summary,
    }
    atomic_json(root / "experiment_summary.json", summary)
    atomic_json(root / "run_state.json", {"status": "complete", "phase": "complete"})
    _heartbeat(progress, "complete", status="complete")
    return summary
