"""Bounded state application and artifact writing for latent dynamics."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from neurobench.algorithms.latent_dynamics import (
    dynamic_drive,
    estimate_quiet_noise,
    fit_shared_ar1_grid,
    kalman_filter_ar1,
    rts_smoother_ar1,
    state_difference,
    standardized_filter_innovation,
)
from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib, _sha256
from neurobench.metrics.latent_signal import latent_reconstruction_metrics

from .config import LatentDynamicsConfig
from .synthetic import synthetic_suite


def _tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _progress(path: Path, stage: str, **details: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_unix": time.time(), "stage": stage, **details}, sort_keys=True) + "\n")
        stream.flush()


def _atomic_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _create_memmap(path: Path, shape: tuple[int, ...]) -> tuple[np.memmap, Path]:
    temporary = path.with_name(path.name + ".partial")
    mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    return mmap, temporary


def _complete_memmap(mmap: np.memmap, temporary: Path, destination: Path) -> None:
    mmap.flush()
    del mmap
    loaded = np.load(temporary, mmap_mode="r", allow_pickle=False)
    if loaded.dtype != np.float32 or not np.isfinite(loaded).all():
        raise FloatingPointError(f"Invalid dense artifact: {temporary}")
    del loaded
    temporary.replace(destination)


def raw_direct_pool(residual: np.ndarray) -> np.ndarray:
    """Frozen Raw Direct temporal anchor used by this feature-only benchmark."""
    from scipy.special import logsumexp
    values = np.asarray(residual)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("residual must be finite with time as the first axis")
    return (0.25 * (logsumexp(values / 0.25, axis=0) - math.log(len(values)))).astype(np.float32)


def _legacy_asymmetric_ema(residual: np.ndarray) -> np.ndarray:
    """Array form of the historical baseline's default asymmetric gains."""
    values = np.asarray(residual, dtype=np.float32)
    baseline = np.median(values[:min(50, len(values))], axis=0).astype(np.float32)
    positive = np.empty_like(values)
    for index, frame in enumerate(values):
        innovation = frame - baseline
        positive[index] = np.maximum(innovation, 0.0)
        gain = np.where(innovation > 0, 0.012, 0.09).astype(np.float32)
        baseline += gain * innovation
    return positive


def _lane_is_causal(name: str) -> bool:
    return not name.startswith("smoother")


def _evaluate_segment_maps(
    segment_maps: dict[str, dict[str, np.ndarray]],
    labels: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    nms_distance_px: int = 6,
    candidate_cap: int = 500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen quiet calibration, NMS, cap, and one-to-one matching."""
    from neurobench.experiments.learnable_contrast import core as v1

    summaries: list[dict[str, Any]] = []
    known_matches: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    folds_by_lane: dict[str, list[dict[str, Any]]] = {}
    for lane, segments in segment_maps.items():
        quiet_peaks = []
        for index in range(4):
            quiet_peaks.extend(v1._peaks(
                segments[f"quiet_{index}"].reshape(height, width), nms_distance_px, limit=2000
            ))
        ranked = sorted((peak[0] for peak in quiet_peaks), reverse=True)
        if not ranked:
            raise RuntimeError(f"No quiet calibration peaks for lane {lane}")
        reference = ranked[min(4, len(ranked) - 1)]
        threshold = float(np.nextafter(reference, np.inf))
        folds = []
        for burst in range(1, 5):
            score_map = segments[f"burst_{burst}"].reshape(height, width)
            peaks = v1._peaks(score_map, nms_distance_px, threshold, limit=candidate_cap)
            rows = [row for row in labels if row["burst_id"] == burst]
            matches = v1._match(peaks, rows, nms_distance_px)
            matched_keys = {(float(score), int(x), int(y)) for _, score, x, y, _ in matches}
            for label_index, score, x, y, distance in matches:
                row = rows[label_index]
                known_matches.append({
                    "lane": lane, "burst_id": burst,
                    "label_id": row.get("roi_identity", f"burst{burst}_label{label_index}"),
                    "label_x_px": row["x_px"], "label_y_px": row["y_px"],
                    "candidate_x_px": x, "candidate_y_px": y,
                    "score": score, "distance_px": distance, "status": "known_match",
                })
            for score, x, y in peaks:
                if (float(score), int(x), int(y)) not in matched_keys:
                    unmatched.append({
                        "lane": lane, "burst_id": burst, "x_px": x, "y_px": y,
                        "score": score, "status": "unknown_unmatched_candidate",
                    })
            folds.append({
                "lane": lane, "heldout_burst": burst, "recall": len(matches) / len(rows),
                "matched": len(matches), "labels": len(rows), "event_peaks": len(peaks),
                "threshold": threshold, "candidate_cap": candidate_cap,
            })
        folds_by_lane[lane] = folds
        summaries.append({
            "lane": lane, "causal": _lane_is_causal(lane),
            "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
            "matched": sum(fold["matched"] for fold in folds),
            "labels": sum(fold["labels"] for fold in folds),
            "event_peaks": sum(fold["event_peaks"] for fold in folds),
            "quiet_threshold": threshold, "candidate_cap_per_burst": candidate_cap,
            "status": "evaluated",
        })
    raw = next(row for row in summaries if row["lane"] == "raw_direct")
    raw_folds = {row["heldout_burst"]: row for row in folds_by_lane["raw_direct"]}
    for row in summaries:
        folds = folds_by_lane[row["lane"]]
        row["delta_mean_recall_vs_raw_direct"] = row["mean_recall"] - raw["mean_recall"]
        row["fold_wins_vs_raw_direct"] = sum(
            fold["recall"] > raw_folds[fold["heldout_burst"]]["recall"] for fold in folds
        )
    metrics = {
        "raw_direct": raw,
        "best_lane": max(summaries, key=lambda row: (row["mean_recall"], -row["event_peaks"])),
        "lanes": summaries,
        "outer_folds": [fold for lane in segment_maps for fold in folds_by_lane[lane]],
        "evaluation_contract": {
            "quiet_calibration": "fifth-highest pooled quiet NMS peak",
            "nms_distance_px": nms_distance_px, "match_radius_px": nms_distance_px,
            "candidate_cap_per_burst": candidate_cap,
            "pooling": "temperature_0.25_log_mean_exp",
            "unmatched_semantics": "unknown_not_negative",
        },
    }
    return summaries, known_matches, unmatched, metrics


def _matching_preflight(config: LatentDynamicsConfig, directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((directory / "config.resolved.json").read_text(encoding="utf-8"))
    if not payload.get("ready") or resolved != config.to_dict():
        raise RuntimeError("Run requires a ready preflight generated from the identical resolved config")
    return payload


def run(config: LatentDynamicsConfig, *, preflight_dir: str | Path) -> dict[str, Any]:
    """Run a reviewed CPU job; callers remain responsible for explicit dataset selection."""
    reviewed = Path(preflight_dir).expanduser().resolve()
    preflight_payload = _matching_preflight(config, reviewed)
    root = config.output_dir
    if root.exists():
        raise FileExistsError(f"Output root exists: {root}")
    for relative in ("fit", "noise", "states", "features", "diagnostics", "evaluation"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    if config.features.write_selected_tiffs:
        (root / "features" / "selected_review_tiffs").mkdir()
    _atomic_json(root / "config.resolved.json", config.to_dict())
    _atomic_json(root / "preflight.json", preflight_payload)
    _atomic_json(root / "run_state.json", {"status": "running", "experiment_id": config.experiment_id})
    progress = root / "progress.jsonl"
    _progress(progress, "started")
    started = time.monotonic()
    try:
        source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        f = config.frames
        review = source[f.review_start_ui - 1:f.review_end_ui]
        quiet = source[f.quiet_start_ui - 1:f.quiet_end_ui]
        frames, height, width = review.shape
        pixels = height * width
        quiet_flat = np.asarray(quiet, dtype=np.float32).reshape(len(quiet), pixels)
        direct_low, direct_high = np.percentile(np.asarray(quiet[:, ::4, ::4], dtype=np.float32), [1.0, 99.9])
        direct_scale = max(float(direct_high - direct_low), 1.0)
        noise = estimate_quiet_noise(
            quiet_flat, floor_percentile=config.preprocessing.quiet_scale_floor_percentile
        )
        _atomic_array(root / "noise" / "quiet_center.npy", noise.center.reshape(height, width))
        _atomic_array(root / "noise" / "quiet_scale.npy", noise.scale.reshape(height, width))
        _atomic_json(root / "noise" / "quiet_noise_summary.json", {
            **noise.diagnostics, "scale_floor": noise.scale_floor,
            "axes": "YX", "units": "source_intensity", "causal": None,
        })
        del quiet_flat
        rng = np.random.default_rng(config.fit.sample_seed)
        sample = np.sort(rng.choice(pixels, size=config.fit.sample_pixels, replace=False))
        sampled_raw = np.asarray(review.reshape(frames, pixels)[:, sample], dtype=np.float64)
        sampled = (sampled_raw - noise.center[sample]) / noise.scale[sample]
        quiet_variance = float(np.median(noise.difference_variance / np.maximum(noise.scale, 1e-12) ** 2))
        observation_variance = max(quiet_variance, 1e-6)
        model, candidates = fit_shared_ar1_grid(
            sampled, frame_period_ms=f.frame_period_ms,
            decay_time_ms_grid=config.fit.decay_time_ms_grid,
            process_to_observation_grid=config.fit.process_to_observation_grid,
            observation_variance=observation_variance,
            stability_epsilon=config.fit.stability_epsilon,
        )
        _atomic_json(root / "fit" / "sample_manifest.json", {
            "seed": config.fit.sample_seed, "pixel_indices_flat_yx": sample.tolist(),
            "sample_count": len(sample), "labels_available_to_fit": False,
            "temporal_validation_blocks": config.fit.temporal_validation_blocks,
        })
        candidate_fields = list(candidates[0])
        _tsv(root / "fit" / "candidate_models.tsv", candidates, candidate_fields)
        _tsv(root / "fit" / "predictive_likelihood.tsv", candidates, candidate_fields)
        _tsv(root / "fit" / "parameter_history.tsv", [{"step": 0, "mode": "bounded_grid", "gamma": model.gamma}], ["step", "mode", "gamma"])
        selected_payload = {**asdict(model), "model_id": "stable_ar1_shared_v1", "labels_available_to_fit": False}
        _atomic_json(root / "fit" / "selected_model.json", selected_payload)
        _atomic_json(root / "fit" / "stability.json", {
            "passed": model.stability_margin >= config.fit.stability_epsilon,
            "gamma": model.gamma, "stability_margin": model.stability_margin,
            "required_margin": config.fit.stability_epsilon,
        })
        _progress(progress, "fit_complete", candidate_models=len(candidates), gamma=model.gamma)

        # Labels enter only after model selection; fitting above is label-blind.
        from neurobench.experiments.learnable_contrast import core as v1
        labels = v1.load_labels(config.labels_tsv)
        quiet_offset = f.quiet_start_ui - f.review_start_ui
        segments: dict[str, slice] = {
            "quiet_0": slice(quiet_offset, quiet_offset + 24),
            "quiet_1": slice(quiet_offset + 24, quiet_offset + 48),
            "quiet_2": slice(quiet_offset + 48, quiet_offset + 76),
            "quiet_3": slice(quiet_offset + 53, quiet_offset + 100),
        }
        for burst in range(1, 5):
            rows = [row for row in labels if row["burst_id"] == burst]
            segments[f"burst_{burst}"] = slice(
                rows[0]["start_frame_ui"] - f.review_start_ui,
                rows[0]["end_frame_ui"] - f.review_start_ui + 1,
            )
        if any(segment.start < 0 or segment.stop > frames or segment.start >= segment.stop for segment in segments.values()):
            raise ValueError(f"Evaluation segment falls outside the review interval: {segments}")

        shape = (frames, height, width)
        state_outputs: dict[str, tuple[np.memmap, Path, Path]] = {}
        if config.application.write_filter_mean:
            destination = root / "states" / "filter_mean.npy"
            mmap, partial = _create_memmap(destination, shape)
            state_outputs["filter"] = (mmap, partial, destination)
        if config.application.write_smoother_mean:
            destination = root / "states" / "smoother_mean.npy"
            mmap, partial = _create_memmap(destination, shape)
            state_outputs["smoother"] = (mmap, partial, destination)
        maps: dict[str, np.ndarray] = {
            "raw_direct": np.full(pixels, -np.inf, dtype=np.float32),
            "legacy_asymmetric_ema": np.full(pixels, -np.inf, dtype=np.float32),
            "filter_innovation_z": np.full(pixels, -np.inf, dtype=np.float32),
        }
        for lag in config.features.lags:
            maps[f"raw_state_difference_lag_{lag}"] = np.full(pixels, -np.inf, dtype=np.float32)
        for lane in state_outputs:
            maps[f"{lane}_amplitude"] = np.full(pixels, -np.inf, dtype=np.float32)
            maps[f"{lane}_positive_dynamic_drive"] = np.full(pixels, -np.inf, dtype=np.float32)
            for lag in config.features.lags:
                maps[f"{lane}_state_difference_lag_{lag}"] = np.full(pixels, -np.inf, dtype=np.float32)
        segment_maps = {
            lane: {segment: np.full(pixels, -np.inf, dtype=np.float32) for segment in segments}
            for lane in maps
        }
        block_pixels = config.application.tile_height * config.application.tile_width
        filter_variance = smoother_variance = None
        flat_review = review.reshape(frames, pixels)
        for start in range(0, pixels, block_pixels):
            stop = min(pixels, start + block_pixels)
            raw_block = np.asarray(flat_review[:, start:stop], dtype=np.float32)
            physical_residual = raw_block - noise.center[start:stop]
            direct_signed = physical_residual / direct_scale
            residual = physical_residual / noise.scale[start:stop]
            lane_values: dict[str, tuple[np.ndarray, int]] = {
                "raw_direct": (np.maximum(direct_signed, 0), 0),
                "legacy_asymmetric_ema": (_legacy_asymmetric_ema(direct_signed), 0),
            }
            for lag in config.features.lags:
                lane_values[f"raw_state_difference_lag_{lag}"] = (state_difference(direct_signed, lag), lag)
            filtered = kalman_filter_ar1(residual, model)
            lane_values["filter_innovation_z"] = (standardized_filter_innovation(filtered), 0)
            filter_variance = filtered.filter_variance
            scale = noise.scale[start:stop]
            if "filter" in state_outputs:
                state_outputs["filter"][0].reshape(frames, pixels)[:, start:stop] = filtered.filter_mean * scale
                filter_physical = filtered.filter_mean * scale / direct_scale
                lane_values["filter_amplitude"] = (np.maximum(filter_physical, 0), 0)
                lane_values["filter_positive_dynamic_drive"] = (np.maximum(dynamic_drive(filter_physical, model.gamma), 0), 1)
                for lag in config.features.lags:
                    lane_values[f"filter_state_difference_lag_{lag}"] = (state_difference(filter_physical, lag), lag)
            if "smoother" in state_outputs:
                smoothed = rts_smoother_ar1(filtered, model)
                smoother_variance = smoothed.variance
                state_outputs["smoother"][0].reshape(frames, pixels)[:, start:stop] = smoothed.mean * scale
                smoother_physical = smoothed.mean * scale / direct_scale
                lane_values["smoother_amplitude"] = (np.maximum(smoother_physical, 0), 0)
                lane_values["smoother_positive_dynamic_drive"] = (np.maximum(dynamic_drive(smoother_physical, model.gamma), 0), 1)
                for lag in config.features.lags:
                    lane_values[f"smoother_state_difference_lag_{lag}"] = (state_difference(smoother_physical, lag), lag)
            if set(lane_values) != set(maps):
                raise RuntimeError(f"Feature lane mismatch: {sorted(set(maps) ^ set(lane_values))}")
            for lane, (values, undefined_leading) in lane_values.items():
                maps[lane][start:stop] = raw_direct_pool(values[undefined_leading:])
                for segment_name, segment in segments.items():
                    segment_start = max(segment.start, undefined_leading)
                    if segment_start >= segment.stop:
                        raise ValueError(f"No defined frames for {lane} in {segment_name}")
                    segment_maps[lane][segment_name][start:stop] = raw_direct_pool(values[segment_start:segment.stop])
            _progress(progress, "application_chunk", start_pixel=start, stop_pixel=stop, total_pixels=pixels)
        for mmap, partial, destination in state_outputs.values():
            _complete_memmap(mmap, partial, destination)
        review_tiffs = []
        if config.features.write_selected_tiffs:
            import tifffile
            for lane, (_, _, state_path) in state_outputs.items():
                destination = root / "features" / "selected_review_tiffs" / f"{lane}_mean.tif"
                temporary = destination.with_name(destination.name + ".partial")
                tifffile.imwrite(
                    temporary, np.load(state_path, mmap_mode="r", allow_pickle=False),
                    bigtiff=True, photometric="minisblack", metadata={"axes": "TYX"},
                )
                with tifffile.TiffFile(temporary) as handle:
                    if len(handle.pages) != frames:
                        raise RuntimeError(f"Incomplete selected review TIFF: {temporary}")
                temporary.replace(destination)
                review_tiffs.append({"lane": lane, "path": str(destination), "causal": lane == "filter"})
        if filter_variance is None:
            filter_variance = kalman_filter_ar1(sampled[:, :1], model).filter_variance
        _atomic_array(root / "states" / "filter_variance_by_time.npy", np.asarray(filter_variance, dtype=np.float32))
        if smoother_variance is None:
            smoother_variance = rts_smoother_ar1(kalman_filter_ar1(sampled[:, :1], model), model).variance
        _atomic_array(root / "states" / "smoother_variance_by_time.npy", np.asarray(smoother_variance, dtype=np.float32))
        map_arrays = {name: values.reshape(height, width) for name, values in maps.items()}
        map_arrays.update({
            f"{lane}__{segment}": values.reshape(height, width)
            for lane, lane_segments in segment_maps.items()
            for segment, values in lane_segments.items()
        })
        feature_path = root / "features" / "pooled_candidate_maps.npz"
        feature_partial = feature_path.with_name(feature_path.name + ".partial")
        with feature_partial.open("wb") as stream:
            np.savez_compressed(stream, **map_arrays)
        with np.load(feature_partial, allow_pickle=False) as archive:
            if set(archive.files) != set(map_arrays) or any(not np.isfinite(archive[name]).all() for name in archive.files):
                raise FloatingPointError("Invalid pooled feature archive")
        feature_partial.replace(feature_path)
        source_checksum = _sha256(config.source_video)
        _atomic_json(root / "features" / "feature_manifest.json", {
            "shape": [height, width], "axes": "YX", "dtype": "float32",
            "frame_alignment": "review interval; lag-leading frames undefined",
            "units": "quiet-normalized residual", "normalization": "quiet median/MAD",
            "model_id": "stable_ar1_shared_v1", "source_sha256": source_checksum,
            "lanes": {name: {"causal": _lane_is_causal(name)} for name in maps},
            "evaluation_segments_zero_half_open_relative_to_review": {
                name: [segment.start, segment.stop] for name, segment in segments.items()
            },
            "selected_review_tiffs": review_tiffs,
        })
        _atomic_json(root / "states" / "state_manifest.json", {
            "shape": list(shape), "axes": "TYX", "dtype": "float32",
            "frame_alignment": {"ui_inclusive": [f.review_start_ui, f.review_end_ui]},
            "units": "source_intensity residual", "normalization": "restored after quiet median/MAD fitting",
            "model_id": "stable_ar1_shared_v1", "source_sha256": source_checksum,
            "filter_mean": {"written": config.application.write_filter_mean, "causal": True},
            "smoother_mean": {"written": config.application.write_smoother_mean, "causal": False},
        })
        from neurobench.experiments.learnable_contrast.direct_tuning import _direct_map
        direct_sample = np.maximum((sampled_raw - noise.center[sample]) / direct_scale, 0).astype(np.float32)
        raw_reference = _direct_map(direct_sample)
        raw_candidate = raw_direct_pool(direct_sample)
        anchor_error = float(np.max(np.abs(raw_reference - raw_candidate)))
        lanes, known_matches, unmatched_candidates, benchmark = _evaluate_segment_maps(
            segment_maps, labels, height=height, width=width,
            nms_distance_px=int(round(config.evaluation.primary_match_radius_px)),
            candidate_cap=500,
        )
        metrics = {**benchmark,
            "raw_direct_anchor": {"max_abs_error": anchor_error, "passed": anchor_error == 0.0},
            "sparse_positive_semantics": "unmatched_candidates_are_unknown_not_negative",
            "scientific_status": "labeled feature benchmark completed; advancement gates require review",
        }
        _atomic_json(root / "evaluation" / "metrics.json", metrics)
        _tsv(root / "evaluation" / "lane_summary.tsv", lanes, list(lanes[0]))
        known_fields = ["lane", "burst_id", "label_id", "label_x_px", "label_y_px",
                        "candidate_x_px", "candidate_y_px", "score", "distance_px", "status"]
        unmatched_fields = ["lane", "burst_id", "x_px", "y_px", "score", "status"]
        _tsv(root / "evaluation" / "known_matches.tsv", known_matches, known_fields)
        _tsv(root / "evaluation" / "unmatched_candidates.tsv", unmatched_candidates, unmatched_fields)
        residual_summary = {"quiet_scale_floor": noise.scale_floor, "finite": True}
        _atomic_json(root / "diagnostics" / "residual_summary.json", residual_summary)
        _atomic_json(root / "diagnostics" / "innovation_summary.json", {"finite": True, "definition": "observation minus one-step prediction"})
        for name, fields in (("quiet_autocorrelation.tsv", ["lane", "lag", "correlation"]),
                             ("event_preservation.tsv", ["lane", "status"]),
                             ("perturbation_stability.tsv", ["perturbation", "status"])):
            _tsv(root / "diagnostics" / name, [], fields)
        elapsed = time.monotonic() - started
        _atomic_json(root / "resource_summary.json", {
            "elapsed_seconds": elapsed, "cpu_threads": config.resources.cpu_threads,
            "ram_available_mib_at_completion": _available_ram_mib(), "gpu_used": False,
        })
        report = (
            "# Latent-dynamics run report\n\n"
            f"Model: stable shared AR(1), gamma={model.gamma:.8f}.\n\n"
            f"Raw Direct anchor max absolute error: {anchor_error:.3g}.\n\n"
            f"Raw Direct mean held-out recall: {metrics['raw_direct']['mean_recall']:.4f} "
            f"({metrics['raw_direct']['matched']}/{metrics['raw_direct']['labels']} matches; "
            f"{metrics['raw_direct']['event_peaks']} candidates).\n\n"
            f"Best single lane: {metrics['best_lane']['lane']} at "
            f"{metrics['best_lane']['mean_recall']:.4f} mean recall "
            f"({metrics['best_lane']['matched']}/{metrics['best_lane']['labels']} matches; "
            f"{metrics['best_lane']['event_peaks']} candidates).\n\n"
            "Unmatched candidates are unknown, not negatives. Completion is not scientific success.\n"
        )
        (root / "report.md").write_text(report, encoding="utf-8")
        final = {"status": "complete", "experiment_id": config.experiment_id,
                 "output_dir": str(root), "raw_direct_anchor_passed": anchor_error == 0.0,
                 "raw_direct_mean_recall": metrics["raw_direct"]["mean_recall"],
                 "best_lane": metrics["best_lane"],
                 "full_spon_run_was_implicitly_authorized": False}
        _atomic_json(root / "run_state.json", final)
        _progress(progress, "complete", elapsed_seconds=elapsed)
        return final
    except BaseException as exc:
        for partial in root.rglob("*.partial"):
            partial.unlink(missing_ok=True)
        _atomic_json(root / "run_state.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        _progress(progress, "failed", error_type=type(exc).__name__, error=str(exc))
        raise


def run_synthetic(output_dir: str | Path, *, seeds: tuple[int, ...] = (7, 13, 19, 29, 37)) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Synthetic output exists: {destination}")
    destination.mkdir(parents=True)
    rows = []
    for case in synthetic_suite(seeds):
        model, _ = fit_shared_ar1_grid(
            case.observation, frame_period_ms=20.0,
            decay_time_ms_grid=(40.0, 80.0, 160.0, 320.0),
            process_to_observation_grid=(0.03, 0.1, 0.3, 1.0),
            observation_variance=max(case.parameters["observation_std"] ** 2, 1e-6),
        )
        filtered = kalman_filter_ar1(case.observation, model)
        metrics = latent_reconstruction_metrics(case.latent, filtered.filter_mean)
        rows.append({"case_id": case.case_id, "seed": case.parameters["seed"],
                     "gamma": model.gamma, **metrics})
    _tsv(destination / "synthetic_metrics.tsv", rows, list(rows[0]))
    payload = {"status": "complete", "cases": len(rows), "seeds": list(seeds),
               "scientific_status": "falsification evidence; not real-data authorization"}
    _atomic_json(destination / "summary.json", payload)
    return payload


def feature_benchmark(run_dir: str | Path) -> dict[str, Any]:
    """Read an already-produced benchmark; it performs no new dataset work."""
    root = Path(run_dir).expanduser().resolve()
    return json.loads((root / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
