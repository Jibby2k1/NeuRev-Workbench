"""Resumable metrics-only execution for the center/whiten/ICA screen."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from neurobench.algorithms.local_covariance_whitening import (
    CovarianceEstimatorConfig,
    SpatialTileConfig,
    WhiteningFeatureBank,
    contiguous_quiet_partitions,
    fit_covariance,
    fit_tiled_feature_whiteners,
    gather_covariance_samples,
)
from neurobench.algorithms.local_covariance_whitening_cuda import (
    apply_tiled_feature_whiteners_cuda,
)
from neurobench.algorithms.multiscale_local_normalization import (
    JointSTContext,
    causal_joint_centered_residual,
    causal_joint_msln,
)
from neurobench.experiments.msln_msica.artifacts import atomic_json
from neurobench.metrics.whiteness import covariance_identity_error

from .center_whiten_ica_benchmark import (
    CONTEXT_BANKS,
    SCREEN_AUDIT_OPTOUT,
    WHITENING_MODES,
    apply_ica_rotation,
    component_ambiguity,
    deterministic_sample_rows,
    fit_ica_rotation,
    load_screen_config,
    preflight_screen,
    stage_a_lanes,
    stage_b_lanes,
    stage_c_lanes,
)


def _context(context_id: str) -> JointSTContext:
    parts = context_id.split("_")
    return JointSTContext(
        context_id, int(parts[1][1:]), int(parts[2][1:]),
        int(parts[3][1:]), int(parts[4][1:]),
    )


def _quiet_and_event_masks(config: dict[str, Any], frames: int) -> tuple[np.ndarray, np.ndarray]:
    review_start = int(config["source"]["review_interval_ui"][0])
    quiet = np.zeros(frames, dtype=bool)
    q0, q1 = map(int, config["source"]["quiet_interval_ui"])
    quiet[q0 - review_start:q1 - review_start + 1] = True
    event = np.zeros(frames, dtype=bool)
    for interval in config["source"]["burst_intervals_ui"].values():
        start, stop = map(int, interval)
        event[start - review_start:stop - review_start + 1] = True
    if not quiet.any() or not event.any() or np.any(quiet & event):
        raise ValueError("quiet and event masks must be nonempty and disjoint")
    return quiet, event


def build_center_bank(
    movie: np.ndarray,
    config: dict[str, Any],
    center_family: str,
    bank_id: str,
) -> tuple[WhiteningFeatureBank, np.ndarray, np.ndarray]:
    """Materialize exactly one aligned three-channel bank."""
    context_ids = CONTEXT_BANKS[bank_id]
    contexts = tuple(_context(item) for item in context_ids)
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    pre_roll = max(item.temporal_window_frames for item in contexts)
    source_start = review_start - 1 - pre_roll
    if source_start < 0 or review_stop > len(movie):
        raise ValueError("review interval lacks causal pre-roll or exceeds movie")
    source = np.asarray(movie[source_start:review_stop], dtype=np.float32)
    source_quiet = np.zeros(len(source), dtype=bool)
    q0, q1 = map(int, config["source"]["quiet_interval_ui"])
    source_quiet[q0 - 1 - source_start:q1 - source_start] = True
    results = []
    for context in contexts:
        if center_family == "joint_msln":
            result = causal_joint_msln(source, context, quiet_mask=source_quiet)
            scientific_array = "signed_msln"
            scale_floor = float(result.scale_floor)
        elif center_family == "joint_residual":
            result = causal_joint_centered_residual(source, context)
            scientific_array = "centered_residual_numerator"
            scale_floor = None
        else:
            raise ValueError(f"unknown center family: {center_family}")
        results.append((result, scientific_array, scale_floor))
    arrays = [item[0].values[pre_roll:] for item in results]
    valid = np.logical_and.reduce([item[0].valid_frames[pre_roll:] for item in results])
    contexts_meta = tuple({
        "context_id": context_id,
        "scientific_array": scientific_array,
        "display_clipped": False,
        "squared": False,
        "scale_floor": scale_floor,
        "support": dict(result.diagnostics),
    } for context_id, (result, scientific_array, scale_floor) in zip(context_ids, results))
    bank = WhiteningFeatureBank(
        context_ids, np.stack(arrays, axis=-1).astype(np.float32, copy=False),
        valid.astype(bool), contexts_meta,
        {"center_family": center_family, "context_bank": bank_id, "feature_order_frozen": True},
    )
    quiet, event = _quiet_and_event_masks(config, len(bank.values))
    return bank, quiet, event


def _subsample_energy(values: np.ndarray, mask: np.ndarray, limit: int = 200_000) -> np.ndarray:
    selected = np.sum(np.square(values[np.asarray(mask, dtype=bool)], dtype=np.float32), axis=-1).ravel()
    if len(selected) > limit:
        selected = selected[np.linspace(0, len(selected) - 1, limit, dtype=np.int64)]
    return np.asarray(selected, dtype=np.float64)


def representation_metrics(
    values: np.ndarray,
    valid: np.ndarray,
    quiet_holdout: np.ndarray,
    event: np.ndarray,
    *,
    quiet_calibration: np.ndarray | None = None,
    runtime_seconds: float,
    unresolved_fraction: float = 0.0,
) -> dict[str, float]:
    active_quiet = np.asarray(quiet_holdout, dtype=bool) & valid
    active_event = np.asarray(event, dtype=bool) & valid
    calibration_mask = active_quiet if quiet_calibration is None else (
        np.asarray(quiet_calibration, dtype=bool) & valid
    )
    quiet_energy = _subsample_energy(values, active_quiet)
    calibration_energy = _subsample_energy(values, calibration_mask)
    event_energy = _subsample_energy(values, active_event)
    if not len(quiet_energy) or not len(calibration_energy) or not len(event_energy):
        raise ValueError("metrics require valid quiet-holdout and event samples")
    quiet_rows, _ = deterministic_sample_rows(values, active_quiet, maximum_samples=65536, seed=29)
    covariance = np.cov(quiet_rows.T, bias=True)
    quiet_median = float(np.median(quiet_energy))
    event_median = float(np.median(event_energy))
    q99 = float(np.quantile(calibration_energy, 0.99))
    event_tail_fraction = float(np.mean(event_energy > q99))
    quiet_tail_error = float(abs(np.mean(quiet_energy > q99) - 0.01))
    # Measure cohesion and recurrence on a bounded 2x spatial subsample.
    energy_small = np.sum(
        np.square(values[:, ::2, ::2], dtype=np.float32), axis=-1, dtype=np.float32,
    )
    active = energy_small > q99
    event_active = active[active_event]
    if event_active.any():
        from scipy.ndimage import uniform_filter
        neighbor_fraction = uniform_filter(event_active.astype(np.float32), size=(1, 3, 3), mode="constant")
        spatial_cohesion = float(np.mean(neighbor_fraction[event_active] > (1.5 / 9.0)))
    else:
        spatial_cohesion = 0.0
    event_indices = np.flatnonzero(active_event)
    recurrence_values = []
    for left, right in zip(event_indices[:-1], event_indices[1:]):
        if right != left + 1:
            continue
        union = np.count_nonzero(active[left] | active[right])
        if union:
            recurrence_values.append(np.count_nonzero(active[left] & active[right]) / union)
    temporal_recurrence = float(np.mean(recurrence_values)) if recurrence_values else 0.0
    return {
        "heldout_covariance_identity_error": covariance_identity_error(covariance),
        "event_quiet_log_contrast": float(np.log1p(event_median) - np.log1p(quiet_median)),
        "event_above_quiet_q99_fraction": event_tail_fraction,
        "quiet_q99_calibration_error": quiet_tail_error,
        "event_spatial_cohesion": spatial_cohesion,
        "event_temporal_recurrence": temporal_recurrence,
        "unresolved_fraction": float(unresolved_fraction),
        "runtime_seconds": float(runtime_seconds),
    }


def _rank_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministic equal-weight rank aggregation with explicit directions."""
    terms = (
        ("heldout_covariance_identity_error", False),
        ("event_quiet_log_contrast", True),
        ("event_above_quiet_q99_fraction", True),
        ("quiet_q99_calibration_error", False),
        ("event_spatial_cohesion", True),
        ("event_temporal_recurrence", True),
        ("unresolved_fraction", False),
        ("runtime_seconds", False),
    )
    scores = np.zeros(len(rows), dtype=np.float64)
    for key, higher_is_better in terms:
        order = sorted(range(len(rows)), key=lambda index: (float(rows[index][key]), str(rows[index].get("lane_id", ""))))
        ranks = np.empty(len(rows), dtype=np.float64)
        for rank, index in enumerate(order):
            ranks[index] = rank
        if higher_is_better:
            ranks = len(rows) - 1 - ranks
        scores += ranks
    ranked = []
    for index, row in enumerate(rows):
        ranked.append({**row, "selection_rank_sum": float(scores[index])})
    ranked.sort(key=lambda row: (row["selection_rank_sum"], row.get("lane_id", "")))
    return ranked[:limit]


def _global_transform(
    bank: WhiteningFeatureBank,
    fit_mask: np.ndarray,
    method: str,
    ridge: float = 0.05,
) -> tuple[np.ndarray, float]:
    samples, diagnostics = gather_covariance_samples(
        bank, fit_mask, maximum_samples=65536, spatial_subsample=2,
    )
    fit = fit_covariance(
        samples,
        CovarianceEstimatorConfig(method=method, ridge_ratio=ridge),
        feature_ids=bank.feature_ids, block_count=diagnostics["blocked_effective_sample_count"],
    )
    if not fit.resolved:
        return bank.values, 1.0
    transformed = np.einsum(
        "...d,ed->...e", bank.values.astype(np.float64) - fit.mean,
        fit.whitening, optimize=True,
    ).astype(np.float32)
    transformed[~bank.valid_frames] = 0
    return transformed, 0.0


def whiten_bank(
    bank: WhiteningFeatureBank,
    partitions: dict[str, np.ndarray],
    mode: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if mode == "none" or mode == "ica_native_global":
        return bank.values, 0.0, {"mode": mode, "deferred_to_ica": mode == "ica_native_global"}
    if mode == "global_diagonal_quiet":
        values, unresolved = _global_transform(bank, partitions["fit"], "diagonal")
        return values, unresolved, {"mode": mode}
    if mode == "global_full_oas_quiet":
        values, unresolved = _global_transform(bank, partitions["fit"], "oas")
        return values, unresolved, {"mode": mode}
    if mode == "global_full_fixed_ridge_0p05_quiet":
        values, unresolved = _global_transform(bank, partitions["fit"], "fixed_ridge", 0.05)
        return values, unresolved, {"mode": mode}
    if mode not in {"local_diagonal_quiet_t64", "local_full_oas_quiet_t64"}:
        raise ValueError(f"unknown whitening mode: {mode}")
    tile = SpatialTileConfig(64, 64, 32, 32, "hann", "crop", 64, 65536, 2, 1)
    estimator = "diagonal" if mode == "local_diagonal_quiet_t64" else "oas"
    fits = fit_tiled_feature_whiteners(
        bank, partitions["fit"], tile, CovarianceEstimatorConfig(method=estimator),
    )
    result = apply_tiled_feature_whiteners_cuda(
        bank, fits, tile, partitions["calibration"],
        frame_chunk=int(config["compute"]["frame_chunk"]),
        max_vram_bytes=int(config["compute"]["maximum_peak_vram_gb"] * 2**30),
    )
    unresolved = float(np.mean(result.unresolved_tile_mask))
    return result.zca_features, unresolved, dict(result.diagnostics)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("rows", []))


def _checkpoint(path: Path, rows: list[dict[str, Any]], expected: int) -> None:
    atomic_json(path, {"complete": len(rows) == expected, "expected": expected, "rows": rows})


def _release_cuda() -> None:
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass
    gc.collect()


def run_metrics_screen(config_path: str | Path) -> dict[str, Any]:
    """Run/resume Stages A-C and freeze at most four label-free finalists."""
    config = load_screen_config(config_path)
    root = Path(config["outputs"]["root_dir"])
    preflight_path = root / "preflight.json"
    if not preflight_path.is_file():
        preflight_screen(config_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("labels_loaded") is not False or preflight.get("audit_opt_out_reason") != SCREEN_AUDIT_OPTOUT:
        raise RuntimeError("screen preflight does not preserve the label-free audit contract")
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    atomic_json(root / "status.json", {"status": "running", "stage": "A", "labels_loaded": False})

    stage_a_path = root / "stage_a" / "checkpoint.json"
    stage_a_rows = _load_rows(stage_a_path)
    completed_a = {(row["center_family"], row["context_bank"]) for row in stage_a_rows}
    for center, bank_id in stage_a_lanes():
        if (center, bank_id) in completed_a:
            continue
        started = time.monotonic()
        bank, quiet, event = build_center_bank(movie, config, center, bank_id)
        partitions = contiguous_quiet_partitions(quiet & bank.valid_frames)
        metrics = representation_metrics(bank.values, bank.valid_frames, partitions["holdout"], event, quiet_calibration=partitions["calibration"], runtime_seconds=time.monotonic() - started)
        stage_a_rows.append({"lane_id": f"{center}__{bank_id}", "center_family": center, "context_bank": bank_id, **metrics})
        _checkpoint(stage_a_path, stage_a_rows, len(stage_a_lanes()))
        del bank; _release_cuda()
    retained_a_rows = _rank_rows(stage_a_rows, int(config["screen"]["maximum_stage_a"]))
    retained_a = [(row["center_family"], row["context_bank"]) for row in retained_a_rows]
    atomic_json(root / "stage_a" / "retained.json", {"rows": retained_a_rows, "labels_loaded": False})

    stage_b_defs = stage_b_lanes(retained_a)
    stage_b_path = root / "stage_b" / "checkpoint.json"
    stage_b_rows = _load_rows(stage_b_path)
    completed_b = {(row["center_family"], row["context_bank"], row["whitening_mode"]) for row in stage_b_rows}
    atomic_json(root / "status.json", {"status": "running", "stage": "B", "labels_loaded": False})
    for center, bank_id in retained_a:
        bank, quiet, event = build_center_bank(movie, config, center, bank_id)
        partitions = contiguous_quiet_partitions(quiet & bank.valid_frames)
        for whitening in WHITENING_MODES:
            if (center, bank_id, whitening) in completed_b:
                continue
            started = time.monotonic()
            values, unresolved, diagnostics = whiten_bank(bank, partitions, whitening, config)
            metrics = representation_metrics(values, bank.valid_frames, partitions["holdout"], event, quiet_calibration=partitions["calibration"], runtime_seconds=time.monotonic() - started, unresolved_fraction=unresolved)
            stage_b_rows.append({"lane_id": f"{center}__{bank_id}__{whitening}", "center_family": center, "context_bank": bank_id, "whitening_mode": whitening, **metrics, "fallback_counts": diagnostics.get("fallback_counts", {})})
            _checkpoint(stage_b_path, stage_b_rows, len(stage_b_defs))
            if values is not bank.values:
                del values
            _release_cuda()
        del bank; _release_cuda()
    retained_b_rows = _rank_rows(stage_b_rows, int(config["screen"]["maximum_stage_b"]))
    retained_b = [(row["center_family"], row["context_bank"], row["whitening_mode"]) for row in retained_b_rows]
    atomic_json(root / "stage_b" / "retained.json", {"rows": retained_b_rows, "labels_loaded": False})

    stage_c_defs = stage_c_lanes(retained_b)
    stage_c_path = root / "stage_c" / "checkpoint.json"
    stage_c_rows = _load_rows(stage_c_path)
    completed_c = {row["lane_id"] for row in stage_c_rows}
    atomic_json(root / "status.json", {"status": "running", "stage": "C", "labels_loaded": False})
    for center, bank_id, whitening in retained_b:
        bank, quiet, event = build_center_bank(movie, config, center, bank_id)
        partitions = contiguous_quiet_partitions(quiet & bank.valid_frames)
        whitened, unresolved, diagnostics = whiten_bank(bank, partitions, whitening, config)
        samples, sample_ids = deterministic_sample_rows(
            whitened, bank.valid_frames,
            maximum_samples=int(config["screen"]["maximum_ica_samples"]),
            seed=int(config["screen"]["sample_seed"]),
        )
        for lane in (item for item in stage_c_defs if item.center_family == center and item.context_bank == bank_id and item.whitening_mode == whitening):
            if lane.lane_id in completed_c:
                continue
            started = time.monotonic()
            fit = fit_ica_rotation(samples, lane.ica, native_global_whitening=whitening == "ica_native_global")
            transformed = apply_ica_rotation(whitened, fit)
            metrics = representation_metrics(transformed, bank.valid_frames, partitions["holdout"], event, quiet_calibration=partitions["calibration"], runtime_seconds=time.monotonic() - started, unresolved_fraction=unresolved)
            stage_c_rows.append({**lane.to_dict(), **metrics, "converged": fit.converged, "iterations": fit.iterations, "internal_whitening": fit.internal_whitening, "component_ambiguity": component_ambiguity(fit.rotation), "sample_id_sha256": __import__("hashlib").sha256(sample_ids.tobytes()).hexdigest(), "fallback_counts": diagnostics.get("fallback_counts", {})})
            _checkpoint(stage_c_path, stage_c_rows, len(stage_c_defs))
            del transformed; _release_cuda()
        if whitened is not bank.values:
            del whitened
        del bank; _release_cuda()
    eligible = [row for row in stage_c_rows if bool(row["converged"]) and float(row["unresolved_fraction"]) < 1.0]
    finalists = _rank_rows(eligible, min(int(config["screen"]["maximum_finalists"]), len(eligible)))
    freeze = {
        "status": "frozen",
        "labels_loaded": False,
        "selection_is_label_free": True,
        "audit_opt_out_reason": SCREEN_AUDIT_OPTOUT,
        "finalists": finalists,
        "all_stage_c_lanes": len(stage_c_rows),
        "protected_evaluation_status": "not_run",
        "scientific_audit_status": "mandatory_for_finalists_not_run",
    }
    atomic_json(root / "freeze_decision.json", freeze)
    _atomic_csv(root / "stage_a" / "metrics.csv", stage_a_rows)
    _atomic_csv(root / "stage_b" / "metrics.csv", stage_b_rows)
    _atomic_csv(root / "stage_c" / "metrics.csv", stage_c_rows)
    atomic_json(root / "status.json", {"status": "screen_complete", "stage": "C", "labels_loaded": False, "finalist_count": len(finalists), "scientific_status": "protected_evaluation_not_run"})
    return freeze
