"""Generate and cross-fit a bounded activity-feature bank on Spon Ca Burst."""
from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Sequence

import numpy as np
import tifffile

from neurobench.algorithms.activity_feature_bank import (
    cross_scale_consensus_score,
    derivative_feature_iterator,
    localized_feature_trace_metrics,
    morphology_feature_iterator,
    persistence_features,
    quiet_robust_z,
    unit_positive,
)
from neurobench.algorithms.innovative_denoising import local_noise_psd_wiener
from neurobench.algorithms.cfar import robust_local_cfar
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import (
    QUIET_DURATIONS,
    QUIET_STARTS,
    event_intervals,
)
from neurobench.experiments.pairwise_separation.fusion import fit_bounded_lambda
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
    quiet_calibrated_threshold,
    temporal_pool,
)

from .denoise_audit import _synthetic_fixture
from .feature_utility_config import FEATURE_IDS, FeatureUtilityConfig
from .innovation_denoising_program import _cached_dense, _fit_context
from .innovation_grid import (
    _atomic_json,
    _available_ram_mib,
    _progress,
    _sha256,
    _snapshots,
)
from .signal_noise_split import (
    _coefficients,
    _innovation_residual,
    _quiet_standardization,
)


CONTEXT_ONLY_FEATURES = {
    "persistent_artifact_score",
    "cfar_background",
    "cfar_noise",
}
OFFLINE_FEATURES = {"local_psd_signal", "local_psd_correction"}
OFFLINE_FIT_FEATURES = {"asymmetric_state", "asymmetric_innovation"}


def _atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resource_checkpoint(
    config: FeatureUtilityConfig, progress: Path, stage: str
) -> float:
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    _progress(
        progress,
        "resource_checkpoint",
        checkpoint=stage,
        max_rss_mib=rss_mib,
    )
    if rss_mib > float(config.resources["max_ram_mib"]):
        raise MemoryError(
            f"{stage} exceeded RAM cap: {rss_mib:.1f} MiB > "
            f"{config.resources['max_ram_mib']} MiB"
        )
    return rss_mib


def preflight(
    config: FeatureUtilityConfig, *, write_artifacts: bool = True
) -> dict[str, Any]:
    inputs = (config.source_video, config.labels_tsv, config.architecture_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    shape = None
    dtype = None
    bounds = finite = labels_valid = fit_valid = False
    labels: list[dict[str, Any]] = []
    gpu: dict[str, Any] = {"available": False}
    if not missing:
        video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(video.shape), str(video.dtype)
        start = int(config.frames["review_start_ui"]) - 1
        stop = int(config.frames["review_end_ui"])
        bounds = (
            video.ndim == 3
            and 0 <= start < int(config.frames["quiet_end_ui"]) <= stop <= len(video)
        )
        finite = bounds and bool(
            np.isfinite(video[start:stop:20, ::16, ::16]).all()
        )
        labels = label_core.load_labels(config.labels_tsv)
        labels_valid = bool(
            len(labels) == 79
            and len({row["roi_identity"] for row in labels}) == 27
            and all(
                0 <= row["x_px"] < video.shape[2]
                and 0 <= row["y_px"] < video.shape[1]
                for row in labels
            )
        )
        architecture = json.loads(
            config.architecture_manifest.read_text(encoding="utf-8")
        )
        fit = architecture.get("raw_stochastic_fit", {})
        fit_valid = bool(
            architecture.get("source_video") == str(config.source_video)
            and fit.get("classification_status") == "resolved"
            and fit.get("optimizer_converged") is True
            and fit.get("safety", {}).get("status") == "accepted"
        )
    try:
        import torch

        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            free, total = torch.cuda.mem_get_info()
            gpu.update(
                name=torch.cuda.get_device_name(0),
                free_mib=free / 2**20,
                total_mib=total / 2**20,
            )
    except ImportError:
        pass
    frames = (
        int(config.frames["review_end_ui"])
        - int(config.frames["review_start_ui"])
        + 1
    )
    pixels = 0 if shape is None else int(shape[1]) * int(shape[2])
    dense_mib = frames * pixels * 4 / 2**20
    stored_feature_mib = (config.feature_count + 1) * dense_mib / 2
    tiff_mib = (
        len(config.feature_bank["tiff_feature_ids"])
        * frames
        * pixels
        * 2
        / 2**20
    )
    estimated_output = stored_feature_mib + tiff_mib + 256
    estimated_ram = 18.0 * dense_mib + 1024
    patch = int(config.shared_ica["patch_size"])
    batch = int(config.resources["frame_batch_size"])
    estimated_gpu = (
        3
        * batch
        * pixels
        * (patch**2 + 2 * int(config.shared_ica["rank"]))
        * 4
        / 2**20
        + 768
    )
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    cuda_requested = config.resources["device"] == "cuda"
    gates = {
        "inputs_exist": not missing,
        "source_is_npy": config.source_video.suffix == ".npy",
        "frame_bounds_valid": bounds,
        "finite_sample": finite,
        "labels_valid": labels_valid,
        "accepted_fit_matches_source": fit_valid,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(
            str(config.output_dir) + ".partial"
        ).exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram
        <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk
        >= int(config.resources["min_free_disk_mib"]) + estimated_output,
        "output_cap_sufficient": estimated_output
        <= int(config.resources["max_output_mib"]),
        "requested_device_available": (not cuda_requested) or gpu["available"],
        "gpu_memory_cap_sufficient": (
            (not cuda_requested)
            or estimated_gpu <= int(config.resources["max_gpu_memory_mib"])
        ),
        "live_gpu_memory_sufficient": (
            (not cuda_requested) or estimated_gpu <= gpu.get("free_mib", 0)
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_spon_ca_feature_utility_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "source_shape": shape,
        "source_dtype": dtype,
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "design": {
            "feature_count": config.feature_count,
            "feature_ids": list(FEATURE_IDS),
            "fixed_lane_count": config.fixed_lane_count,
            "learned_scalar_fit_count": config.learned_scalar_fit_count,
            "multifeature_fit_count": config.multifeature_fit_count,
            "tiff_count": len(config.feature_bank["tiff_feature_ids"]),
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "estimated_peak_gpu_memory_mib": estimated_gpu,
            "estimated_output_mib": estimated_output,
            "free_disk_mib": free_disk,
            "gpu": gpu,
            **config.resources,
        },
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in inputs
            if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "scientific_contract": (
            "Features are auxiliary evidence. Even nonlinearities never replace "
            "the signed carrier. Learned weights are nonnegative, carrier-"
            "initialized, and evaluated by held-out burst. Event candidates "
            "unmatched to sparse labels remain unknown."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(
            config.preflight_dir / "config.resolved.json", config.to_dict()
        )
        if not missing and bounds:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"feature utility preflight failed: {payload}")
    return payload


def _matching_preflight(config: FeatureUtilityConfig) -> dict[str, Any]:
    audit = json.loads(
        (config.preflight_dir / "preflight.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (config.preflight_dir / "config.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(
        str(config.output_dir) + ".partial"
    ).exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _write_feature(
    directory: Path,
    feature_id: str,
    values: np.ndarray,
    metadata: dict[str, Any],
    *,
    chunk_frames: int = 16,
) -> dict[str, Any]:
    path = directory / f"{feature_id}.npy"
    temporary = path.with_name(path.name + ".partial")
    target = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=values.shape,
    )
    for start in range(0, len(values), int(chunk_frames)):
        stop = min(len(values), start + int(chunk_frames))
        target[start:stop] = values[start:stop]
    target.flush()
    del target
    temporary.replace(path)
    entry = {
        "feature_id": feature_id,
        "path": str(path.relative_to(directory.parent)),
        "shape": list(values.shape),
        "dtype": "float16",
        "bytes": path.stat().st_size,
        "causal_status": (
            "offline_diagnostic"
            if feature_id in OFFLINE_FEATURES
            else (
                "context_only"
                if feature_id in CONTEXT_ONLY_FEATURES
                else (
                    "causal_after_offline_fit"
                    if feature_id in OFFLINE_FIT_FEATURES
                    else "causal_after_quiet_calibration"
                )
            )
        ),
        "fusion_eligible": feature_id
        not in CONTEXT_ONLY_FEATURES | OFFLINE_FEATURES,
        **metadata,
    }
    _atomic_json(directory / f"{feature_id}.json", entry)
    return entry


def _write_cfar_features(
    directory: Path,
    carrier: np.ndarray,
    config: FeatureUtilityConfig,
) -> list[dict[str, Any]]:
    settings = config.feature_bank["cfar"]
    feature_ids = (
        "cfar_score",
        "cfar_background",
        "cfar_noise",
        "spatial_coherence",
    )
    temporaries = {
        feature_id: directory / f"{feature_id}.npy.partial"
        for feature_id in feature_ids
    }
    stores = {
        feature_id: np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=carrier.shape
        )
        for feature_id, path in temporaries.items()
    }
    chunk = int(settings["chunk_frames"])
    for start in range(0, len(carrier), chunk):
        stop = min(len(carrier), start + chunk)
        result = robust_local_cfar(
            carrier[start:stop],
            pfa=float(settings["pfa"]),
            guard_px=int(settings["guard_px"]),
            training_radius_px=int(settings["training_radius_px"]),
            epsilon=float(settings["epsilon"]),
            device=str(config.resources["device"]),
        )
        stores["cfar_score"][start:stop] = np.clip(
            result["score"] / float(settings["score_clip_z"]), 0, 1
        )
        stores["cfar_background"][start:stop] = result["local_mean"]
        stores["cfar_noise"][start:stop] = result["local_std"]
        stores["spatial_coherence"][start:stop] = np.maximum(
            np.maximum(carrier[start:stop], 0) - result["local_mean"], 0
        )
    entries = []
    for feature_id, store in stores.items():
        store.flush()
        del store
        final = directory / f"{feature_id}.npy"
        temporaries[feature_id].replace(final)
        entry = {
            "feature_id": feature_id,
            "path": str(final.relative_to(directory.parent)),
            "shape": list(carrier.shape),
            "dtype": "float16",
            "bytes": final.stat().st_size,
            "causal_status": (
                "context_only"
                if feature_id in CONTEXT_ONLY_FEATURES
                else "causal_after_quiet_calibration"
            ),
            "fusion_eligible": feature_id not in CONTEXT_ONLY_FEATURES,
            "family": "cfar_continuous_statistics",
            "parameters": dict(settings),
        }
        _atomic_json(directory / f"{feature_id}.json", entry)
        entries.append(entry)
    return entries


def _normalization(
    values: np.ndarray, quiet_count: int
) -> tuple[np.ndarray, float]:
    quiet = np.asarray(values[:quiet_count], dtype=np.float32)
    baseline = np.median(quiet, axis=0).astype(np.float32)
    low, high = np.percentile(quiet[:, ::4, ::4], [1.0, 99.9])
    scale = max(float(high - low), 1e-6)
    return baseline, scale


def _pool(
    values: np.ndarray,
    start: int,
    stop: int,
    baseline: np.ndarray,
    scale: float,
    tau: float,
) -> np.ndarray:
    frames = np.maximum(
        (np.asarray(values[start:stop], dtype=np.float32) - baseline[None])
        / float(scale),
        0,
    )
    return temporal_pool(frames, f"lme{float(tau)}")


def _pooled_maps(
    values: np.ndarray,
    quiet_count: int,
    labels: list[dict[str, Any]],
    config: FeatureUtilityConfig,
) -> dict[str, Any]:
    baseline, scale = _normalization(values, quiet_count)
    tau = float(config.evaluation["temporal_pool_tau"])
    quiet_maps = [
        _pool(values, start, start + duration, baseline, scale, tau)
        for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS)
    ]
    events = {
        int(burst): _pool(values, start, stop, baseline, scale, tau)
        for burst, (start, stop) in event_intervals(
            labels, int(config.frames["review_start_ui"])
        ).items()
    }
    return {
        "quiet": quiet_maps,
        "events": events,
        "normalization_scale": scale,
    }


def _evaluate_maps(
    lane: str,
    maps: dict[str, Any],
    labels: list[dict[str, Any]],
    config: FeatureUtilityConfig,
) -> dict[str, Any]:
    threshold = quiet_calibrated_threshold(
        maps["quiet"],
        int(config.evaluation["nms_distance_px"]),
        float(config.evaluation["quiet_false_peaks_per_map"]),
        limit=3000,
    )
    folds = []
    for burst, score in maps["events"].items():
        ranked = extract_local_maxima(
            score,
            int(config.evaluation["nms_distance_px"]),
            limit=500,
        )
        selected = [peak for peak in ranked if peak[0] >= threshold]
        fixed = ranked[: int(config.evaluation["fixed_candidates_per_burst"])]
        rows = [row for row in labels if int(row["burst_id"]) == int(burst)]
        matched = match_peaks_one_to_one(
            selected, rows, float(config.evaluation["match_radius_px"])
        )[0]
        fixed_matched = match_peaks_one_to_one(
            fixed, rows, float(config.evaluation["match_radius_px"])
        )[0]
        folds.append(
            {
                "burst_id": int(burst),
                "labels": len(rows),
                "matched": len(matched),
                "recall": len(matched) / len(rows),
                "candidates": len(selected),
                "fixed_matched": len(fixed_matched),
                "fixed_recall": len(fixed_matched) / len(rows),
            }
        )
    return {
        "lane": lane,
        "mean_recall": float(np.mean([row["recall"] for row in folds])),
        "pooled_recall": sum(row["matched"] for row in folds)
        / sum(row["labels"] for row in folds),
        "event_candidates": sum(row["candidates"] for row in folds),
        "fixed_budget_mean_recall": float(
            np.mean([row["fixed_recall"] for row in folds])
        ),
        "threshold": threshold,
        "folds": folds,
    }


def _combine_maps(
    carrier: dict[str, Any],
    feature: dict[str, Any],
    *,
    kind: str,
    value: float,
) -> dict[str, Any]:
    def combine(raw: np.ndarray, auxiliary: np.ndarray) -> np.ndarray:
        unit = np.clip(auxiliary, 0, 1)
        if kind == "boost":
            return (raw * (1.0 + float(value) * unit)).astype(np.float32)
        if kind == "gate":
            return (
                raw * (float(value) + (1.0 - float(value)) * unit)
            ).astype(np.float32)
        raise ValueError("unknown map fusion kind")

    return {
        "quiet": [
            combine(raw, auxiliary)
            for raw, auxiliary in zip(carrier["quiet"], feature["quiet"])
        ],
        "events": {
            burst: combine(carrier["events"][burst], feature["events"][burst])
            for burst in carrier["events"]
        },
    }


def _samples(
    carrier: dict[str, Any],
    contribution: dict[str, np.ndarray],
    labels: list[dict[str, Any]],
    held_out: int,
) -> tuple[np.ndarray, ...]:
    raw_positive = []
    feature_positive = []
    for burst, score in carrier["events"].items():
        if int(burst) == int(held_out):
            continue
        feature_map = contribution[int(burst)]
        for row in labels:
            if int(row["burst_id"]) != int(burst):
                continue
            x = int(round(row["x_px"]))
            y = int(round(row["y_px"]))
            ys = slice(max(0, y - 2), min(score.shape[0], y + 3))
            xs = slice(max(0, x - 2), min(score.shape[1], x + 3))
            raw_positive.append(float(np.max(score[ys, xs])))
            feature_positive.append(float(np.max(feature_map[ys, xs])))
    raw_negative = []
    feature_negative = []
    for raw_map, feature_map in zip(carrier["quiet"], contribution["quiet"]):
        count = min(256, raw_map.size)
        indices = np.argpartition(raw_map.ravel(), -count)[-count:]
        raw_negative.extend(raw_map.ravel()[indices].tolist())
        feature_negative.extend(feature_map.ravel()[indices].tolist())
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            raw_positive,
            feature_positive,
            raw_negative,
            feature_negative,
        )
    )


def _evaluate_fold(
    quiet_maps: Sequence[np.ndarray],
    event_map: np.ndarray,
    burst: int,
    labels: list[dict[str, Any]],
    config: FeatureUtilityConfig,
) -> dict[str, Any]:
    threshold = quiet_calibrated_threshold(
        quiet_maps,
        int(config.evaluation["nms_distance_px"]),
        float(config.evaluation["quiet_false_peaks_per_map"]),
        limit=3000,
    )
    ranked = extract_local_maxima(
        event_map,
        int(config.evaluation["nms_distance_px"]),
        limit=500,
    )
    selected = [peak for peak in ranked if peak[0] >= threshold]
    fixed = ranked[: int(config.evaluation["fixed_candidates_per_burst"])]
    rows = [row for row in labels if int(row["burst_id"]) == int(burst)]
    matched = match_peaks_one_to_one(
        selected, rows, float(config.evaluation["match_radius_px"])
    )[0]
    fixed_matched = match_peaks_one_to_one(
        fixed, rows, float(config.evaluation["match_radius_px"])
    )[0]
    return {
        "burst_id": int(burst),
        "labels": len(rows),
        "matched": len(matched),
        "recall": len(matched) / len(rows),
        "candidates": len(selected),
        "fixed_matched": len(fixed_matched),
        "fixed_recall": len(fixed_matched) / len(rows),
    }


def _fit_weights(
    raw_positive: np.ndarray,
    feature_positive: np.ndarray,
    raw_negative: np.ndarray,
    feature_negative: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    maximum_total: float,
) -> tuple[np.ndarray, list[float]]:
    if feature_positive.ndim != 2 or feature_negative.ndim != 2:
        raise ValueError("multi-feature samples must be matrices")
    count = max(len(raw_positive), len(raw_negative))
    pi = np.arange(count) % len(raw_positive)
    ni = np.arange(count) % len(raw_negative)
    base_delta = raw_positive[pi] - raw_negative[ni]
    feature_delta = feature_positive[pi] - feature_negative[ni]
    weights = np.zeros(feature_positive.shape[1], dtype=np.float64)
    history = []
    for _ in range(int(epochs)):
        margin = np.clip(base_delta + feature_delta @ weights, -40, 40)
        probability = 1.0 / (1.0 + np.exp(margin))
        gradient = np.mean(-feature_delta * probability[:, None], axis=0)
        gradient += 2.0 * float(l2) * weights
        weights = np.maximum(
            weights - float(learning_rate) * gradient,
            0,
        )
        total = float(weights.sum())
        if total > float(maximum_total):
            weights *= float(maximum_total) / total
        history.append(
            float(np.mean(np.logaddexp(0, -margin)) + float(l2) * np.sum(weights**2))
        )
    return weights, history


def _write_tiff(
    path: Path,
    values: np.ndarray,
    metadata: dict[str, Any],
    config: FeatureUtilityConfig,
) -> dict[str, Any]:
    visualization = config.visualization
    sampled = np.asarray(values[
        :: int(visualization["sample_frame_stride"]),
        :: int(visualization["sample_row_stride"]),
        :: int(visualization["sample_column_stride"]),
    ], dtype=np.float32)
    maximum = max(
        float(np.percentile(np.maximum(sampled, 0), visualization["upper_percentile"])),
        1e-6,
    )
    temporary = path.with_name(path.name + ".partial")
    with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
        for index in range(len(values)):
            frame = np.asarray(values[index], dtype=np.float32)
            page = np.rint(
                np.clip(np.maximum(frame, 0) / maximum, 0, 1) * 65535
            ).astype(np.uint16)
            writer.write(
                page,
                photometric="minisblack",
                compression=visualization["compression"],
                metadata=None,
                description=(
                    json.dumps(
                        {
                            **metadata,
                            "display_source_limits": [0, maximum],
                            "axes": "TYX",
                        },
                        sort_keys=True,
                    )
                    if index == 0
                    else None
                ),
            )
    temporary.replace(path)
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != len(values) or tiff.pages[0].shape != values.shape[1:]:
            raise RuntimeError(f"TIFF verification failed: {path}")
    return {
        "path": f"feature_tiffs/{path.name}",
        "pages": len(values),
        "shape": list(values.shape[1:]),
        "bytes": path.stat().st_size,
        "display_maximum": maximum,
    }


def _small_features(
    synthetic_z: np.ndarray,
    quiet_count: int,
    context: dict[str, Any],
    config: FeatureUtilityConfig,
) -> dict[str, np.ndarray]:
    bank = config.feature_bank
    features: dict[str, np.ndarray] = {}
    for feature_id, values, _ in derivative_feature_iterator(
        synthetic_z,
        quiet_count=quiet_count,
        spatial_sigma_px=float(bank["derivative"]["spatial_sigma_px"]),
        lags=bank["derivative"]["lags"],
        clip_z=float(bank["clip_z"]),
        power=float(bank["derivative"]["power"]),
        energy_tau_z=float(bank["derivative"]["energy_tau_z"]),
        huber_delta_z=float(bank["derivative"]["huber_delta_z"]),
    ):
        features[feature_id] = values
    psd = local_noise_psd_wiener(
        synthetic_z,
        quiet_count=quiet_count,
        **bank["local_psd"],
    )
    features["local_psd_signal"] = psd
    features["local_psd_correction"] = np.abs(psd - synthetic_z)
    for name, settings in bank["cross_scale"].items():
        features[f"cross_scale_{name}"] = cross_scale_consensus_score(
            synthetic_z, **settings
        )
    asym = bank["asymmetric"]
    state = _cached_dense(
        synthetic_z,
        "feature_utility_synthetic",
        context,
        config,
        mode="asymmetric",
        cache_suffix=tuple(asym.values()),
        asymmetric_rise_gain=float(asym["rise_gain"]),
        asymmetric_decay_gain=float(asym["decay_gain"]),
        asymmetric_innovation_threshold_z=float(
            asym["innovation_threshold_z"]
        ),
        asymmetric_innovation_temperature_z=float(
            asym["innovation_temperature_z"]
        ),
    )
    features["asymmetric_state"] = state
    features["asymmetric_innovation"] = np.abs(synthetic_z - state)
    features.update(
        persistence_features(
            synthetic_z,
            frame_period_ms=float(config.frames["frame_period_ms"]),
            **bank["persistence"],
        )
    )
    for feature_id, values, _ in morphology_feature_iterator(
        synthetic_z,
        quiet_count=quiet_count,
        clip_z=float(bank["clip_z"]),
        **bank["morphology"],
    ):
        features[feature_id] = values
    cfar = robust_local_cfar(
        synthetic_z,
        pfa=float(bank["cfar"]["pfa"]),
        guard_px=int(bank["cfar"]["guard_px"]),
        training_radius_px=int(bank["cfar"]["training_radius_px"]),
        epsilon=float(bank["cfar"]["epsilon"]),
        device=str(config.resources["device"]),
    )
    features["cfar_score"] = np.clip(
        cfar["score"] / float(bank["cfar"]["score_clip_z"]), 0, 1
    )
    features["cfar_background"] = cfar["local_mean"]
    features["cfar_noise"] = cfar["local_std"]
    features["spatial_coherence"] = np.maximum(
        np.maximum(synthetic_z, 0) - cfar["local_mean"], 0
    )
    if set(features) != set(FEATURE_IDS):
        raise RuntimeError(
            f"small feature set mismatch: {sorted(set(FEATURE_IDS)-set(features))}"
        )
    return features


def _report(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["carrier_baseline"]
    lines = [
        f"# {payload['experiment_id']}",
        "",
        "## Outcome",
        "",
        payload["conclusion"],
        "",
        f"Carrier recall: {baseline['mean_recall']:.3f}; fixed-budget recall: "
        f"{baseline['fixed_budget_mean_recall']:.3f}; candidates: "
        f"{baseline['event_candidates']}.",
        "",
        "## Best feature utilities",
        "",
        "| Feature | Standalone recall | Standalone fixed | Best fixed lane | Fixed recall | Candidates | Synthetic r |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in payload["feature_summaries"]:
        lines.append(
            f"| `{row['feature_id']}` | "
            f"{row['standalone_mean_recall']:.3f} | "
            f"{row['standalone_fixed_recall']:.3f} | "
            f"`{row['best_fixed_lane']}` | "
            f"{row['best_fixed_recall']:.3f} | "
            f"{row['best_fixed_candidates']} | "
            f"{row['synthetic_feature_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Real labels are sparse positives. Candidate count remains a "
            "selectivity-pressure proxy, not measured false positives. Learned "
            "weights use only known positives from three bursts and quiet hard "
            "negatives; the fourth burst is held out.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: FeatureUtilityConfig) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    feature_dir = partial / "features"
    tiff_dir = partial / "feature_tiffs"
    evaluation_dir = partial / "evaluation"
    for directory in (feature_dir, tiff_dir, evaluation_dir):
        directory.mkdir()
    _atomic_json(partial / "preflight.json", audit)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    progress = partial / "progress.jsonl"
    started = time.time()
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[start:stop], dtype=np.float32)
    labels = label_core.load_labels(config.labels_tsv)
    quiet_count = (
        int(config.frames["quiet_end_ui"])
        - int(config.frames["quiet_start_ui"])
        + 1
    )
    residual, innovation = _innovation_residual(
        raw, quiet_count, _coefficients(config), config
    )
    structure = np.median(raw[:quiet_count], axis=0)
    low, high = np.percentile(structure[::4, ::4], [1, 99.9])
    structure_unit = np.clip((structure - low) / max(float(high - low), 1e-6), 0, 1)
    np.save(partial / "structure_unit.npy", structure_unit.astype(np.float32))
    del raw, structure, structure_unit
    center, scale, standardization = _quiet_standardization(
        residual, quiet_count, 10.0
    )
    centered = residual - center[None]
    standardized = centered / scale[None]
    entries: list[dict[str, Any]] = []
    carrier_entry = _write_feature(
        feature_dir,
        "carrier_signed",
        standardized,
        {
            "family": "carrier",
            "normalization": "quiet_per_pixel_median_mad",
            "causal_status": "causal_after_quiet_calibration",
            "fusion_eligible": False,
        },
    )
    _progress(progress, "carrier_written")
    bank = config.feature_bank
    for feature_id, values, metadata in derivative_feature_iterator(
        standardized,
        quiet_count=quiet_count,
        spatial_sigma_px=float(bank["derivative"]["spatial_sigma_px"]),
        lags=bank["derivative"]["lags"],
        clip_z=float(bank["clip_z"]),
        power=float(bank["derivative"]["power"]),
        energy_tau_z=float(bank["derivative"]["energy_tau_z"]),
        huber_delta_z=float(bank["derivative"]["huber_delta_z"]),
    ):
        entries.append(
            _write_feature(
                feature_dir,
                feature_id,
                values,
                {"family": "derivative", "parameters": metadata},
            )
        )
        _progress(progress, "feature_written", feature_id=feature_id)
        del values
        gc.collect()
    psd = local_noise_psd_wiener(
        standardized,
        quiet_count=quiet_count,
        **bank["local_psd"],
    )
    entries.append(
        _write_feature(
            feature_dir,
            "local_psd_signal",
            psd,
            {"family": "local_psd", "parameters": bank["local_psd"]},
        )
    )
    entries.append(
        _write_feature(
            feature_dir,
            "local_psd_correction",
            np.abs(psd - standardized),
            {
                "family": "local_psd",
                "transform": "absolute_correction",
                "parameters": bank["local_psd"],
            },
        )
    )
    del psd
    for name, settings in bank["cross_scale"].items():
        feature_id = f"cross_scale_{name}"
        values = cross_scale_consensus_score(standardized, **settings)
        entries.append(
            _write_feature(
                feature_dir,
                feature_id,
                values,
                {"family": "cross_scale_consensus", "parameters": settings},
            )
        )
        del values
    _resource_checkpoint(config, progress, "spatial_features")
    context = _fit_context(standardized, quiet_count, config)
    asym = bank["asymmetric"]
    state = _cached_dense(
        standardized,
        "feature_utility_real",
        context,
        config,
        mode="asymmetric",
        cache_suffix=tuple(asym.values()),
        asymmetric_rise_gain=float(asym["rise_gain"]),
        asymmetric_decay_gain=float(asym["decay_gain"]),
        asymmetric_innovation_threshold_z=float(
            asym["innovation_threshold_z"]
        ),
        asymmetric_innovation_temperature_z=float(
            asym["innovation_temperature_z"]
        ),
    )
    entries.append(
        _write_feature(
            feature_dir,
            "asymmetric_state",
            state,
            {"family": "asymmetric_component_dynamics", "parameters": asym},
        )
    )
    entries.append(
        _write_feature(
            feature_dir,
            "asymmetric_innovation",
            np.abs(standardized - state),
            {
                "family": "asymmetric_component_dynamics",
                "transform": "absolute_innovation",
                "parameters": asym,
            },
        )
    )
    del state
    context["cache"].clear()
    persistent = persistence_features(
        standardized,
        frame_period_ms=float(config.frames["frame_period_ms"]),
        **bank["persistence"],
    )
    for feature_id, values in persistent.items():
        entries.append(
            _write_feature(
                feature_dir,
                feature_id,
                values,
                {"family": "causal_persistence", "parameters": bank["persistence"]},
            )
        )
        del values
    del persistent
    for feature_id, values, metadata in morphology_feature_iterator(
        standardized,
        quiet_count=quiet_count,
        clip_z=float(bank["clip_z"]),
        **bank["morphology"],
    ):
        entries.append(
            _write_feature(
                feature_dir,
                feature_id,
                values,
                {"family": "morphology_expert", "parameters": metadata},
            )
        )
        del values
        gc.collect()
    entries.extend(_write_cfar_features(feature_dir, standardized, config))
    if {entry["feature_id"] for entry in entries} != set(FEATURE_IDS):
        raise RuntimeError("generated feature ids differ from frozen contract")
    _resource_checkpoint(config, progress, "feature_generation_complete")
    _atomic_json(
        partial / "feature_manifest.json",
        {
            "schema_version": 1,
            "carrier": carrier_entry,
            "structure": {
                "path": "structure_unit.npy",
                "shape": list(center.shape),
                "dtype": "float32",
                "semantics": "quiet structural context; not a detector feature",
            },
            "features": entries,
            "feature_count": len(entries),
            "normalization_contract": (
                "Signed carrier uses quiet per-pixel median/MAD. Even and score "
                "features are bounded or re-standardized from quiet only."
            ),
        },
    )

    synthetic_observed, synthetic_truth, synthetic_scale = _synthetic_fixture(
        centered, scale, config
    )
    synthetic_z = synthetic_observed / synthetic_scale[None]
    synthetic_quiet = min(quiet_count, max(8, len(synthetic_z) // 4))
    synthetic_features = _small_features(
        synthetic_z, synthetic_quiet, context, config
    )
    synthetic_metrics = {}
    for feature_id, values in synthetic_features.items():
        if feature_id in {"local_psd_signal", "asymmetric_state"}:
            evaluated = unit_positive(values, float(bank["clip_z"]))
        elif feature_id in {"cfar_background", "cfar_noise", "spatial_coherence"}:
            evaluated = unit_positive(
                quiet_robust_z(values, synthetic_quiet),
                float(bank["clip_z"]),
            )
        else:
            evaluated = np.maximum(values, 0)
        synthetic_metrics[feature_id] = localized_feature_trace_metrics(
            evaluated, synthetic_truth
        )
    del synthetic_features
    _atomic_json(evaluation_dir / "synthetic_feature_metrics.json", synthetic_metrics)

    carrier_maps = _pooled_maps(
        centered, quiet_count, labels, config
    )
    carrier_result = _evaluate_maps(
        "carrier", carrier_maps, labels, config
    )
    feature_maps: dict[str, dict[str, Any]] = {}
    fixed_rows: list[dict[str, Any]] = [carrier_result]
    standalone_rows: list[dict[str, Any]] = []
    for index, feature_id in enumerate(FEATURE_IDS, start=1):
        _progress(
            progress,
            "feature_evaluation_start",
            feature_id=feature_id,
            feature_index=index,
            feature_total=len(FEATURE_IDS),
        )
        values = np.load(
            feature_dir / f"{feature_id}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        maps = _pooled_maps(values, quiet_count, labels, config)
        feature_maps[feature_id] = maps
        standalone = _evaluate_maps(
            f"standalone__{feature_id}", maps, labels, config
        )
        standalone.update(
            feature_id=feature_id,
            kind="standalone",
            **synthetic_metrics[feature_id],
        )
        standalone_rows.append(standalone)
        fixed_rows.append(standalone)
        for value in config.fusion["boost_values"]:
            lane = f"boost__{feature_id}__{float(value):g}"
            result = _evaluate_maps(
                lane,
                _combine_maps(
                    carrier_maps, maps, kind="boost", value=float(value)
                ),
                labels,
                config,
            )
            result.update(
                feature_id=feature_id, kind="boost", value=float(value)
            )
            fixed_rows.append(result)
        for value in config.fusion["gate_floors"]:
            lane = f"gate__{feature_id}__{float(value):g}"
            result = _evaluate_maps(
                lane,
                _combine_maps(
                    carrier_maps, maps, kind="gate", value=float(value)
                ),
                labels,
                config,
            )
            result.update(
                feature_id=feature_id, kind="gate", value=float(value)
            )
            fixed_rows.append(result)
        del values
    _atomic_json(
        evaluation_dir / "fixed_lanes.json",
        {"lane_count": len(fixed_rows), "rows": fixed_rows},
    )

    nested_fixed = []
    candidates = [row for row in fixed_rows if row["lane"] != "carrier"]
    for held_out in (1, 2, 3, 4):
        def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
            training = [
                fold for fold in row["folds"]
                if int(fold["burst_id"]) != held_out
            ]
            return (
                float(np.mean([fold["fixed_recall"] for fold in training])),
                float(np.mean([fold["recall"] for fold in training])),
                -sum(fold["candidates"] for fold in training),
                row["lane"],
            )

        selected = max(candidates, key=selection_key)
        fold = next(
            fold for fold in selected["folds"]
            if int(fold["burst_id"]) == held_out
        )
        nested_fixed.append(
            {
                "held_out_burst": held_out,
                "selected_lane": selected["lane"],
                **fold,
            }
        )

    learned_scalar = []
    scalar_summary = []
    for feature_id in FEATURE_IDS:
        maps = feature_maps[feature_id]
        contribution = {
            "quiet": [
                raw * np.clip(feature, 0, 1)
                for raw, feature in zip(carrier_maps["quiet"], maps["quiet"])
            ],
            **{
                burst: carrier_maps["events"][burst]
                * np.clip(maps["events"][burst], 0, 1)
                for burst in carrier_maps["events"]
            },
        }
        folds = []
        for held_out in (1, 2, 3, 4):
            samples = _samples(
                carrier_maps, contribution, labels, held_out
            )
            weight, history = fit_bounded_lambda(
                *samples,
                learning_rate=float(config.fusion["learning_rate"]),
                epochs=int(config.fusion["epochs"]),
                l2=float(config.fusion["l2_to_carrier"]),
                maximum=float(config.fusion["maximum_total_weight"]),
            )
            quiet = [
                raw + weight * auxiliary
                for raw, auxiliary in zip(
                    carrier_maps["quiet"], contribution["quiet"]
                )
            ]
            event = (
                carrier_maps["events"][held_out]
                + weight * contribution[held_out]
            )
            fold = _evaluate_fold(
                quiet, event, held_out, labels, config
            )
            fold.update(
                feature_id=feature_id,
                held_out_burst=int(held_out),
                weight=weight,
                loss_initial=history[0],
                loss_final=history[-1],
            )
            learned_scalar.append(fold)
            folds.append(fold)
        scalar_summary.append(
            {
                "feature_id": feature_id,
                "mean_weight": float(np.mean([fold["weight"] for fold in folds])),
                "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
                "fixed_budget_mean_recall": float(
                    np.mean([fold["fixed_recall"] for fold in folds])
                ),
                "event_candidates": sum(fold["candidates"] for fold in folds),
                "burst_wins_fixed": sum(
                    fold["fixed_recall"]
                    > next(
                        base["fixed_recall"]
                        for base in carrier_result["folds"]
                        if base["burst_id"] == fold["burst_id"]
                    )
                    for fold in folds
                ),
            }
        )
    _atomic_json(
        evaluation_dir / "learned_scalar.json",
        {"fit_count": len(learned_scalar), "folds": learned_scalar, "summary": scalar_summary},
    )

    eligible = [
        feature_id
        for feature_id in FEATURE_IDS
        if feature_id not in CONTEXT_ONLY_FEATURES | OFFLINE_FEATURES
    ]
    multifeature_folds = []
    for held_out in (1, 2, 3, 4):
        rank_rows = [
            row for row in learned_scalar
            if row["held_out_burst"] == held_out and row["feature_id"] in eligible
        ]
        selected_ids = [
            row["feature_id"]
            for row in sorted(
                rank_rows,
                key=lambda row: (
                    row["loss_initial"] - row["loss_final"],
                    row["weight"],
                    row["feature_id"],
                ),
                reverse=True,
            )[: int(config.fusion["top_features_per_fold"])]
        ]
        contributions = {}
        for feature_id in selected_ids:
            maps = feature_maps[feature_id]
            contributions[feature_id] = {
                "quiet": [
                    raw * np.clip(feature, 0, 1)
                    for raw, feature in zip(carrier_maps["quiet"], maps["quiet"])
                ],
                **{
                    burst: carrier_maps["events"][burst]
                    * np.clip(maps["events"][burst], 0, 1)
                    for burst in carrier_maps["events"]
                },
            }
        sample_sets = [
            _samples(carrier_maps, contributions[feature_id], labels, held_out)
            for feature_id in selected_ids
        ]
        raw_positive = sample_sets[0][0]
        raw_negative = sample_sets[0][2]
        feature_positive = np.stack([samples[1] for samples in sample_sets], axis=1)
        feature_negative = np.stack([samples[3] for samples in sample_sets], axis=1)
        weights, history = _fit_weights(
            raw_positive,
            feature_positive,
            raw_negative,
            feature_negative,
            learning_rate=float(config.fusion["learning_rate"]),
            epochs=int(config.fusion["epochs"]),
            l2=float(config.fusion["l2_to_carrier"]),
            maximum_total=float(config.fusion["maximum_total_weight"]),
        )
        quiet = []
        for quiet_index, raw_map in enumerate(carrier_maps["quiet"]):
            score = raw_map.copy()
            for feature_id, weight in zip(selected_ids, weights):
                score += float(weight) * contributions[feature_id]["quiet"][quiet_index]
            quiet.append(score)
        event = carrier_maps["events"][held_out].copy()
        for feature_id, weight in zip(selected_ids, weights):
            event += float(weight) * contributions[feature_id][held_out]
        fold = _evaluate_fold(quiet, event, held_out, labels, config)
        fold.update(
            selected_feature_ids=selected_ids,
            weights={key: float(value) for key, value in zip(selected_ids, weights)},
            total_weight=float(weights.sum()),
            loss_initial=history[0],
            loss_final=history[-1],
        )
        multifeature_folds.append(fold)
    multifeature = {
        "folds": multifeature_folds,
        "mean_recall": float(
            np.mean([fold["recall"] for fold in multifeature_folds])
        ),
        "fixed_budget_mean_recall": float(
            np.mean([fold["fixed_recall"] for fold in multifeature_folds])
        ),
        "event_candidates": sum(
            fold["candidates"] for fold in multifeature_folds
        ),
    }
    _atomic_json(evaluation_dir / "learned_multifeature.json", multifeature)

    matrix = []
    for feature_id in FEATURE_IDS:
        maps = feature_maps[feature_id]
        vector = np.concatenate(
            [maps["events"][burst][::4, ::4].ravel() for burst in sorted(maps["events"])]
        )
        matrix.append(vector)
    correlations = np.corrcoef(np.asarray(matrix, dtype=np.float64))
    redundancy_rows = [
        {
            "feature_a": FEATURE_IDS[left],
            "feature_b": FEATURE_IDS[right],
            "correlation": float(correlations[left, right]),
        }
        for left in range(len(FEATURE_IDS))
        for right in range(left + 1, len(FEATURE_IDS))
    ]
    _atomic_tsv(evaluation_dir / "feature_redundancy.tsv", redundancy_rows)

    tiff_rows = []
    for feature_id in bank["tiff_feature_ids"]:
        values = np.load(
            feature_dir / f"{feature_id}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        tiff_rows.append(
            {
                "feature_id": feature_id,
                **_write_tiff(
                    tiff_dir / f"{feature_id}.tif",
                    values,
                    {
                        "feature_id": feature_id,
                        "source": "quiet_calibrated_activity_feature_bank",
                    },
                    config,
                ),
            }
        )
        del values
    _resource_checkpoint(config, progress, "evaluation_and_tiffs_complete")

    feature_summaries = []
    for standalone in standalone_rows:
        feature_id = standalone["feature_id"]
        fixed = [
            row for row in fixed_rows
            if row.get("feature_id") == feature_id and row["kind"] != "standalone"
        ]
        best = max(
            fixed,
            key=lambda row: (
                row["fixed_budget_mean_recall"],
                row["mean_recall"],
                -row["event_candidates"],
            ),
        )
        scalar = next(
            row for row in scalar_summary if row["feature_id"] == feature_id
        )
        feature_summaries.append(
            {
                "feature_id": feature_id,
                "standalone_mean_recall": standalone["mean_recall"],
                "standalone_fixed_recall": standalone[
                    "fixed_budget_mean_recall"
                ],
                "standalone_candidates": standalone["event_candidates"],
                "best_fixed_lane": best["lane"],
                "best_fixed_recall": best["fixed_budget_mean_recall"],
                "best_fixed_mean_recall": best["mean_recall"],
                "best_fixed_candidates": best["event_candidates"],
                "learned_mean_weight": scalar["mean_weight"],
                "learned_fixed_recall": scalar[
                    "fixed_budget_mean_recall"
                ],
                "learned_mean_recall": scalar["mean_recall"],
                "learned_candidates": scalar["event_candidates"],
                "learned_burst_wins_fixed": scalar["burst_wins_fixed"],
                "synthetic_feature_correlation": standalone[
                    "synthetic_feature_correlation"
                ],
                "synthetic_feature_peak_frame_error": standalone[
                    "synthetic_feature_peak_frame_error"
                ],
            }
        )
    best_feature = max(
        feature_summaries,
        key=lambda row: (
            row["learned_fixed_recall"],
            row["learned_mean_recall"],
            -row["learned_candidates"],
        ),
    )
    nested_fixed_summary = {
        "folds": nested_fixed,
        "mean_recall": float(np.mean([row["recall"] for row in nested_fixed])),
        "fixed_budget_mean_recall": float(
            np.mean([row["fixed_recall"] for row in nested_fixed])
        ),
        "event_candidates": sum(row["candidates"] for row in nested_fixed),
    }
    baseline_fixed = float(carrier_result["fixed_budget_mean_recall"])
    conclusion = (
        f"The strongest cross-fitted single feature was "
        f"{best_feature['feature_id']} with fixed-budget recall "
        f"{best_feature['learned_fixed_recall']:.3f} versus carrier "
        f"{baseline_fixed:.3f}. The constrained multifeature model reached "
        f"{multifeature['fixed_budget_mean_recall']:.3f}. "
        "These are feature-utility results, not exhaustive precision estimates."
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "completed",
        "carrier_baseline": carrier_result,
        "feature_summaries": feature_summaries,
        "nested_fixed_selection": nested_fixed_summary,
        "learned_multifeature": multifeature,
        "best_crossfitted_single_feature": best_feature["feature_id"],
        "tiffs": tiff_rows,
        "conclusion": conclusion,
        "innovation_calibration": innovation,
        "quiet_standardization": standardization,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "precision_contract": (
            "Sparse unmatched event candidates are unknown. Candidate burden "
            "and quiet hard negatives do not establish full real-data precision."
        ),
    }
    _atomic_json(partial / "metrics.json", metrics)
    _atomic_tsv(partial / "feature_summary.tsv", feature_summaries)
    _report(partial / "REPORT.md", metrics)
    _atomic_json(
        partial / "run_state.json",
        {
            "status": "completed",
            "elapsed_seconds": metrics["elapsed_seconds"],
            "max_rss_mib": metrics["max_rss_mib"],
            "feature_count": len(entries),
            "fixed_lane_count": len(fixed_rows),
            "learned_scalar_fit_count": len(learned_scalar),
            "multifeature_fit_count": len(multifeature_folds),
            "tiff_count": len(tiff_rows),
        },
    )
    partial.replace(config.output_dir)
    return metrics
