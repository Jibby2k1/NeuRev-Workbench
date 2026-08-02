"""Sequential, independently auditable denoising methods on Parzen Innovation."""
from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Callable

import numpy as np
import tifffile

from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import (
    QUIET_DURATIONS,
    QUIET_STARTS,
    event_intervals,
)
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
    quiet_calibrated_threshold,
    temporal_pool,
)

from .denoise_audit_config import DenoiseAuditConfig
from .denoise_methods import (
    causal_kalman,
    frame_gamma,
    local_low_rank,
    robust_gamma,
    savgol_signal,
    spatial_evidence_gate,
    temporal_evidence_gate,
    undecimated_haar_like,
    quiet_wiener,
)
from .innovation_grid import (
    _aggregate_observations,
    _atomic_json,
    _available_ram_mib,
    _progress,
    _roi_mask_and_matrix,
    _roi_observations,
    _sha256,
    _snapshots,
)
from .signal_noise_split import (
    _coefficients,
    _innovation_residual,
    _quiet_standardization,
)


VARIANT_IDS = (
    "01_pointwise_frame_gamma",
    "01_pointwise_robust_gamma",
    "01_pointwise_quiet_wiener",
    "02_spatial_gate",
    "03_temporal_gate",
    "04_temporal_savgol",
    "04_temporal_haar",
    "04_temporal_kalman",
    "05_local_pca",
    "06_noise_normalized_pca",
    "07_component_parzen_ica",
)


def _atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def preflight(config: DenoiseAuditConfig, *, write_artifacts: bool = True) -> dict[str, Any]:
    inputs = (config.source_video, config.labels_tsv, config.architecture_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    shape = None
    dtype = None
    bounds = labels_valid = fit_valid = finite = False
    labels: list[dict[str, Any]] = []
    gpu = {"available": False}
    if not missing:
        video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(video.shape), str(video.dtype)
        start = int(config.frames["review_start_ui"]) - 1
        stop = int(config.frames["review_end_ui"])
        bounds = (
            video.ndim == 3 and 0 <= start
            < int(config.frames["quiet_end_ui"]) <= stop <= len(video)
        )
        finite = bounds and bool(np.isfinite(video[start:stop:20, ::16, ::16]).all())
        labels = label_core.load_labels(config.labels_tsv)
        labels_valid = bool(
            len(labels) == 79
            and len({row["roi_identity"] for row in labels}) == 27
            and all(
                0 <= row["x_px"] < video.shape[2]
                and 0 <= row["y_px"] < video.shape[1] for row in labels
            )
        )
        architecture = json.loads(config.architecture_manifest.read_text(encoding="utf-8"))
        fit_valid = bool(
            architecture.get("source_video") == str(config.source_video)
            and architecture.get("raw_stochastic_fit", {}).get("safety", {}).get("status")
            == "accepted"
        )
    try:
        import torch

        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            free, total = torch.cuda.mem_get_info()
            gpu.update(
                name=torch.cuda.get_device_name(0),
                free_mib=free / 2**20, total_mib=total / 2**20,
            )
    except ImportError:
        pass
    frames = int(config.frames["review_end_ui"]) - int(config.frames["review_start_ui"]) + 1
    pixels = 0 if shape is None else int(shape[1]) * int(shape[2])
    dense_mib = frames * pixels * 4 / 2**20
    estimated_ram = 4.5 * dense_mib + 1024
    uncompressed_output = config.variant_count * 2 * frames * pixels * 2 / 2**20
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    cuda_requested = config.resources["device"] == "cuda"
    gates = {
        "inputs_exist": not missing, "source_is_npy": config.source_video.suffix == ".npy",
        "frame_bounds_valid": bounds, "finite_sample": finite,
        "labels_valid": labels_valid, "accepted_fit_matches_source": fit_valid,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk >= int(config.resources["min_free_disk_mib"]),
        "output_cap_sufficient": uncompressed_output <= int(config.resources["max_output_mib"]),
        "requested_device_available": (not cuda_requested) or gpu["available"],
        "gpu_memory_sufficient": (
            (not cuda_requested)
            or gpu.get("free_mib", 0) >= int(config.resources["max_gpu_memory_mib"])
        ),
    }
    payload = {
        "schema_version": 1, "kind": "read_only_sequential_denoise_audit_preflight",
        "experiment_id": config.experiment_id, "ready": all(gates.values()),
        "gates": gates, "variant_ids": list(VARIANT_IDS),
        "variant_count": len(VARIANT_IDS), "source_shape": shape, "source_dtype": dtype,
        "label_rows": len(labels), "roi_identities": len({row["roi_identity"] for row in labels}),
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "uncompressed_output_mib": uncompressed_output,
            "free_disk_mib": free_disk, "gpu": gpu,
            **config.resources,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs if path.is_file()
        ],
        "system_snapshot": _snapshots(),
        "interpretation_contract": (
            "Sparse real labels evaluate preservation and known-label recall only. "
            "Semi-synthetic injections provide exact signal/noise truth."
        ),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
        if not missing and bounds:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels, config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"sequential denoise preflight failed: {payload}")
    return payload


def _matching_preflight(config: DenoiseAuditConfig) -> dict[str, Any]:
    audit = json.loads((config.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8"))
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _roi_matrix(
    shape: tuple[int, int], labels: list[dict[str, Any]], radius: int
):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    mask = np.zeros(shape, dtype=bool)
    for row in labels:
        mask |= (
            (xx - int(round(row["x_px"]))) ** 2
            + (yy - int(round(row["y_px"]))) ** 2 <= radius**2
        )
    selected = np.flatnonzero(mask)
    _, matrix = _roi_mask_and_matrix(shape, labels, radius, selected)
    return selected, matrix


def _timing_metrics(
    reference: np.ndarray,
    signal: np.ndarray,
    labels: list[dict[str, Any]],
    intervals: dict[int, tuple[int, int]],
    quiet_count: int,
) -> dict[str, float]:
    peak_errors = []
    onset_errors = []
    duration_errors = []
    derivative_correlations = []
    for index, row in enumerate(labels):
        start, stop = intervals[int(row["burst_id"])]
        ref = np.maximum(
            reference[start:stop, index] - np.median(reference[:quiet_count, index]), 0
        )
        out = np.maximum(
            signal[start:stop, index] - np.median(signal[:quiet_count, index]), 0
        )
        if max(float(ref.max()), float(out.max())) <= 1e-6:
            continue
        peak_errors.append(abs(int(np.argmax(out)) - int(np.argmax(ref))))
        ref_threshold = 0.2 * float(ref.max())
        out_threshold = 0.2 * float(out.max())
        ref_on = int(np.argmax(ref >= ref_threshold))
        out_on = int(np.argmax(out >= out_threshold))
        onset_errors.append(abs(out_on - ref_on))
        ref_duration = int(np.count_nonzero(ref >= 0.5 * float(ref.max())))
        out_duration = int(np.count_nonzero(out >= 0.5 * float(out.max())))
        duration_errors.append(abs(out_duration - ref_duration))
        if len(ref) > 2 and np.std(np.diff(ref)) > 1e-9 and np.std(np.diff(out)) > 1e-9:
            derivative_correlations.append(float(np.corrcoef(np.diff(ref), np.diff(out))[0, 1]))
    return {
        "median_peak_frame_error": float(np.median(peak_errors)),
        "p95_peak_frame_error": float(np.percentile(peak_errors, 95)),
        "median_onset_frame_error": float(np.median(onset_errors)),
        "median_fwhm_frame_error": float(np.median(duration_errors)),
        "median_derivative_correlation": float(np.median(derivative_correlations)),
    }


def _detection_metrics(
    signal: np.ndarray,
    labels: list[dict[str, Any]],
    quiet_count: int,
    config: DenoiseAuditConfig,
) -> dict[str, Any]:
    values = np.asarray(signal, dtype=np.float32).copy()
    baseline = np.median(values[:quiet_count], axis=0)
    low, high = np.percentile(values[:quiet_count, ::4, ::4], [1, 99.9])
    values -= baseline
    values /= max(float(high - low), 1e-6)
    np.maximum(values, 0, out=values)
    tau = float(config.evaluation["temporal_pool_tau"])
    quiet_maps = [
        temporal_pool(values[start:start + duration], f"lme{tau}")
        for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS)
    ]
    threshold = quiet_calibrated_threshold(
        quiet_maps, int(config.evaluation["nms_distance_px"]),
        float(config.evaluation["quiet_false_peaks_per_map"]), limit=3000,
    )
    folds = []
    for burst, (start, stop) in event_intervals(
        labels, int(config.frames["review_start_ui"])
    ).items():
        score = temporal_pool(values[start:stop], f"lme{tau}")
        ranked = extract_local_maxima(
            score, int(config.evaluation["nms_distance_px"]), limit=500
        )
        selected = [peak for peak in ranked if peak[0] >= threshold]
        fixed = ranked[: int(config.evaluation["fixed_candidates_per_burst"])]
        rows = [row for row in labels if int(row["burst_id"]) == burst]
        matches = match_peaks_one_to_one(
            selected, rows, float(config.evaluation["match_radius_px"])
        )[0]
        fixed_matches = match_peaks_one_to_one(
            fixed, rows, float(config.evaluation["match_radius_px"])
        )[0]
        folds.append({
            "burst_id": burst, "labels": len(rows), "matched": len(matches),
            "recall": len(matches) / len(rows), "candidates": len(selected),
            "fixed_matched": len(fixed_matches),
            "fixed_recall": len(fixed_matches) / len(rows),
        })
    del values
    return {
        "mean_recall": float(np.mean([row["recall"] for row in folds])),
        "pooled_recall": sum(row["matched"] for row in folds) / sum(row["labels"] for row in folds),
        "event_candidates": sum(row["candidates"] for row in folds),
        "fixed_budget_mean_recall": float(np.mean([row["fixed_recall"] for row in folds])),
        "folds": folds,
    }


def _synthetic_fixture(
    centered_residual: np.ndarray,
    scale: np.ndarray,
    config: DenoiseAuditConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = config.evaluation
    frames, size = int(e["synthetic_frames"]), int(e["synthetic_size"])
    rng = np.random.default_rng(int(e["synthetic_seed"]))
    max_y = centered_residual.shape[1] - size
    max_x = centered_residual.shape[2] - size
    y = int(rng.integers(0, max_y + 1))
    x = int(rng.integers(0, max_x + 1))
    noise_source = centered_residual[:100, y:y + size, x:x + size]
    noise = np.stack([noise_source[index % len(noise_source)] for index in range(frames)])
    yy, xx = np.ogrid[:size, :size]
    shapes = []
    shapes.append(np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / (2 * 2.5**2)))
    radius = np.sqrt((xx - 48) ** 2 + (yy - 16) ** 2)
    shapes.append(np.exp(-((radius - 4) ** 2) / (2 * 1.1**2)))
    crowded = np.exp(-((xx - 14) ** 2 + (yy - 48) ** 2) / (2 * 2.2**2))
    crowded += 0.7 * np.exp(-((xx - 20) ** 2 + (yy - 48) ** 2) / (2 * 2.2**2))
    shapes.append(crowded)
    radius = np.sqrt((xx - 48) ** 2 + (yy - 48) ** 2)
    ring = np.exp(-((radius - 4) ** 2) / (2 * 1.0**2))
    ring += 0.6 * np.exp(-((xx - 42) ** 2 + (yy - 48) ** 2) / (2 * 2.0**2))
    shapes.append(ring)
    signal = np.zeros_like(noise, dtype=np.float32)
    amplitudes = list(float(value) for value in e["synthetic_snr_multipliers"])
    base_amplitude = float(np.median(scale[y:y + size, x:x + size]))
    for index, (shape, amplitude) in enumerate(zip(shapes, amplitudes)):
        onset = 16 + index * 24
        trace = np.zeros(frames, dtype=np.float32)
        length = frames - onset
        trace[onset:] = (
            1.0 - np.exp(-np.arange(length) / 2.0)
        ) * np.exp(-np.arange(length) / (12.0 + 4 * index))
        signal += (
            amplitude * base_amplitude * trace[:, None, None]
            * np.asarray(shape, dtype=np.float32)[None]
        )
    return (noise + signal).astype(np.float32), signal, scale[y:y + size, x:x + size]


def _synthetic_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    denominator = max(float(np.mean(np.asarray(truth, dtype=np.float64) ** 2)), 1e-12)
    error = np.asarray(estimate, dtype=np.float64) - truth
    correlation = float(np.corrcoef(truth.ravel(), estimate.ravel())[0, 1])
    truth_trace = truth.reshape(len(truth), -1).sum(axis=1)
    estimate_trace = estimate.reshape(len(estimate), -1).sum(axis=1)
    return {
        "synthetic_nmse": float(np.mean(error**2) / denominator),
        "synthetic_correlation": correlation,
        "synthetic_peak_frame_error": abs(
            int(np.argmax(estimate_trace)) - int(np.argmax(truth_trace))
        ),
        "synthetic_peak_amplitude_ratio": float(
            np.max(estimate_trace) / max(float(np.max(truth_trace)), 1e-6)
        ),
        "synthetic_area_ratio": float(
            np.sum(np.maximum(estimate_trace, 0))
            / max(float(np.sum(np.maximum(truth_trace, 0))), 1e-6)
        ),
    }


def _display_scale(
    values: np.ndarray, percentile: float, config: DenoiseAuditConfig
) -> float:
    v = config.visualization
    sampled = values[
        :: int(v["sample_frame_stride"]),
        :: int(v["sample_row_stride"]),
        :: int(v["sample_column_stride"]),
    ]
    return max(float(np.percentile(np.abs(sampled), percentile)), 1e-6)


def _write_tiffs(
    destination: Path,
    signal: np.ndarray,
    residual: np.ndarray,
    common_positive_maximum: float,
    variant_id: str,
    parameters: dict[str, Any],
    config: DenoiseAuditConfig,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    remainder = residual - signal
    remainder_magnitude = _display_scale(
        remainder, float(config.visualization["remainder_absolute_percentile"]), config
    )
    paths = {
        "signal": destination / "signal_positive.tif",
        "remainder": destination / "remainder_detail.tif",
    }
    temporary = {key: path.with_name(path.name + ".partial") for key, path in paths.items()}
    closure_max = 0.0
    description = {
        "variant_id": variant_id, "parameters": parameters,
        "input_equals_signal_plus_remainder": True,
        "signal_display_source_limits": [0, common_positive_maximum],
        "remainder_display_source_limits": [-remainder_magnitude, remainder_magnitude],
        "remainder_scale_is_variant_specific_detail": True,
    }
    with tifffile.TiffWriter(temporary["signal"], bigtiff=True) as signal_writer, \
         tifffile.TiffWriter(temporary["remainder"], bigtiff=True) as remainder_writer:
        for index in range(len(signal)):
            closure_max = max(
                closure_max,
                float(np.max(np.abs(residual[index] - signal[index] - remainder[index]))),
            )
            positive = np.rint(
                np.clip(signal[index] / common_positive_maximum, 0, 1) * 65535
            ).astype(np.uint16)
            signed = np.rint(
                (
                    np.clip(remainder[index] / remainder_magnitude, -1, 1) + 1
                ) * 32767.5
            ).astype(np.uint16)
            first = json.dumps(description, sort_keys=True) if index == 0 else None
            signal_writer.write(
                positive, photometric="minisblack",
                compression=config.visualization["compression"], metadata=None,
                description=first,
            )
            remainder_writer.write(
                signed, photometric="minisblack",
                compression=config.visualization["compression"], metadata=None,
                description=first,
            )
    for key, path in paths.items():
        temporary[key].replace(path)
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) != len(signal) or tiff.pages[0].shape != signal.shape[1:]:
                raise RuntimeError(f"TIFF verification failed: {path}")
    return {
        "signal_tiff": str(paths["signal"].relative_to(destination.parent.parent)),
        "remainder_tiff": str(paths["remainder"].relative_to(destination.parent.parent)),
        "remainder_detail_magnitude": remainder_magnitude,
        "closure_max_absolute": closure_max,
    }


def _variant_metrics(
    variant_id: str,
    signal: np.ndarray,
    residual: np.ndarray,
    labels: list[dict[str, Any]],
    roi_selected: np.ndarray,
    roi_matrix: Any,
    quiet_count: int,
    config: DenoiseAuditConfig,
    runtime_seconds: float,
    synthetic_estimate: np.ndarray,
    synthetic_truth: np.ndarray,
) -> dict[str, Any]:
    reference_traces = np.asarray(
        roi_matrix @ residual.reshape(len(residual), -1)[:, roi_selected].T
    ).T
    signal_traces = np.asarray(
        roi_matrix @ signal.reshape(len(signal), -1)[:, roi_selected].T
    ).T
    intervals = event_intervals(labels, int(config.frames["review_start_ui"]))
    observations = _roi_observations(
        variant_id, signal_traces, reference_traces, labels, intervals, quiet_count
    )
    summary = _aggregate_observations(observations)
    summary.update(_timing_metrics(
        reference_traces, signal_traces, labels, intervals, quiet_count
    ))
    remainder = residual - signal
    quiet_input_rms = max(float(np.sqrt(np.mean(residual[:quiet_count] ** 2))), 1e-6)
    quiet_signal_rms = float(np.sqrt(np.mean(signal[:quiet_count] ** 2)))
    post_input_energy = max(
        float(np.mean(np.asarray(residual[quiet_count:], dtype=np.float64) ** 2)), 1e-12
    )
    post_remainder_energy = float(
        np.mean(np.asarray(remainder[quiet_count:], dtype=np.float64) ** 2)
    )
    summary.update({
        "variant_id": variant_id, "runtime_seconds_compute": runtime_seconds,
        "quiet_signal_rms_ratio": quiet_signal_rms / quiet_input_rms,
        "post_remainder_energy_fraction": post_remainder_energy / post_input_energy,
        **_synthetic_metrics(synthetic_estimate, synthetic_truth),
    })
    detection = _detection_metrics(signal, labels, quiet_count, config)
    summary.update({
        "mean_recall": detection["mean_recall"],
        "pooled_recall": detection["pooled_recall"],
        "event_candidates": detection["event_candidates"],
        "fixed_budget_mean_recall": detection["fixed_budget_mean_recall"],
        "detection_folds": detection["folds"],
    })
    summary["audit_pass"] = bool(
        summary["median_peak_retention"] >= 0.85
        and summary["median_area_retention"] >= 0.85
        and summary["median_waveform_correlation"] >= 0.98
        and summary["median_peak_frame_error"] <= 1
        and summary["quiet_signal_rms_ratio"] <= 0.8
        and summary["synthetic_correlation"] >= 0.9
    )
    return summary


def _method_functions(
    quiet_count: int,
    config: DenoiseAuditConfig,
) -> list[tuple[str, dict[str, Any], Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, Any]]]]]:
    m = config.methods
    frame_period = float(config.frames["frame_period_ms"])

    return [
        (
            "01_pointwise_frame_gamma", m["pointwise"]["frame_gamma"],
            lambda values, local_scale: (
                frame_gamma(values, **m["pointwise"]["frame_gamma"]), {}
            ),
        ),
        (
            "01_pointwise_robust_gamma", m["pointwise"]["robust_gamma"],
            lambda values, local_scale: (
                robust_gamma(values, **m["pointwise"]["robust_gamma"]), {}
            ),
        ),
        (
            "01_pointwise_quiet_wiener", m["pointwise"]["quiet_wiener"],
            lambda values, local_scale: (
                quiet_wiener(values, local_scale, **m["pointwise"]["quiet_wiener"]), {}
            ),
        ),
        (
            "02_spatial_gate", m["spatial_gate"],
            lambda values, local_scale: (
                spatial_evidence_gate(values, local_scale, **m["spatial_gate"]), {}
            ),
        ),
        (
            "03_temporal_gate", m["temporal_gate"],
            lambda values, local_scale: (
                temporal_evidence_gate(
                    values, local_scale, frame_period_ms=frame_period,
                    **m["temporal_gate"],
                ), {}
            ),
        ),
        (
            "04_temporal_savgol",
            {
                "window": m["temporal_filters"]["savgol_window_frames"],
                "polyorder": m["temporal_filters"]["savgol_polyorder"],
            },
            lambda values, local_scale: (
                savgol_signal(
                    values,
                    m["temporal_filters"]["savgol_window_frames"],
                    m["temporal_filters"]["savgol_polyorder"],
                ), {}
            ),
        ),
        (
            "04_temporal_haar",
            {
                "levels": m["temporal_filters"]["haar_levels"],
                "threshold_z": m["temporal_filters"]["haar_threshold_z"],
            },
            lambda values, local_scale: (
                undecimated_haar_like(
                    values, local_scale,
                    levels=m["temporal_filters"]["haar_levels"],
                    threshold_z=m["temporal_filters"]["haar_threshold_z"],
                ), {}
            ),
        ),
        (
            "04_temporal_kalman",
            {
                "decay_ms": m["temporal_filters"]["kalman_decay_ms"],
                "process_variance": m["temporal_filters"]["kalman_process_variance"],
                "observation_variance": m["temporal_filters"]["kalman_observation_variance"],
            },
            lambda values, local_scale: (
                causal_kalman(
                    values, local_scale, frame_period_ms=frame_period,
                    decay_ms=m["temporal_filters"]["kalman_decay_ms"],
                    process_variance=m["temporal_filters"]["kalman_process_variance"],
                    observation_variance=m["temporal_filters"]["kalman_observation_variance"],
                ), {}
            ),
        ),
        (
            "05_local_pca", m["local_pca"],
            lambda values, local_scale: local_low_rank(
                values, local_scale, **m["local_pca"],
                batch_size=int(config.resources["patch_batch_size"]),
                device=str(config.resources["device"]), noise_normalized=False,
                quiet_count=quiet_count,
            ),
        ),
        (
            "06_noise_normalized_pca", m["noise_normalized_pca"],
            lambda values, local_scale: local_low_rank(
                values, local_scale, **m["noise_normalized_pca"],
                batch_size=int(config.resources["patch_batch_size"]),
                device=str(config.resources["device"]), noise_normalized=True,
                quiet_count=quiet_count,
            ),
        ),
        (
            "07_component_parzen_ica", m["component_parzen"],
            lambda values, local_scale: local_low_rank(
                values, local_scale,
                patch_size=m["component_parzen"]["patch_size"],
                stride=m["component_parzen"]["stride"],
                rank=m["component_parzen"]["rank"],
                oversample=m["component_parzen"]["oversample"],
                batch_size=int(config.resources["patch_batch_size"]),
                device=str(config.resources["device"]), noise_normalized=True,
                component_parzen=m["component_parzen"], quiet_count=quiet_count,
            ),
        ),
    ]


def _write_final_report(path: Path, metrics: dict[str, Any]) -> None:
    rows = metrics["variants"]
    lines = [
        f"# {metrics['experiment_id']}", "",
        "## Essential result", "",
        metrics["conclusion"], "",
        "## Comparable results", "",
        "| Method | Peak | Area | Waveform r | Peak error | Quiet RMS | Synthetic r | Synthetic NMSE | Recall | Fixed-budget recall | Candidates | Compute s | Audit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant_id']}` | {row['median_peak_retention']:.3f} | "
            f"{row['median_area_retention']:.3f} | "
            f"{row['median_waveform_correlation']:.3f} | "
            f"{row['median_peak_frame_error']:.1f} | "
            f"{row['quiet_signal_rms_ratio']:.3f} | "
            f"{row['synthetic_correlation']:.3f} | {row['synthetic_nmse']:.3f} | "
            f"{row['mean_recall']:.3f} | {row['fixed_budget_mean_recall']:.3f} | "
            f"{row['event_candidates']} | {row['runtime_seconds_compute']:.2f} | "
            f"{'pass' if row['audit_pass'] else 'flag'} |"
        )
    lines.extend([
        "", "## How to audit", "",
        "Each method directory contains `signal_positive.tif` and "
        "`remainder_detail.tif`. Signal videos share one fixed scale. Remainder "
        "videos use a method-specific symmetric detail scale recorded in TIFF "
        "metadata and `metrics.json`.", "",
        "Visible neurons, propagating activity, or stable anatomy in a remainder "
        "indicate signal leakage. Unmatched event candidates remain unknown because "
        "the real labels are sparse; candidate burden is not precision.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: DenoiseAuditConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    _atomic_json(partial / "preflight.json", audit)
    progress = partial / "progress.jsonl"
    started = time.time()
    _progress(progress, "freeze_common_contract")
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[start:stop], dtype=np.float32)
    labels = label_core.load_labels(config.labels_tsv)
    quiet_count = int(config.frames["quiet_end_ui"]) - int(config.frames["quiet_start_ui"]) + 1
    residual, innovation = _innovation_residual(
        raw, quiet_count, _coefficients(config), config
    )
    del raw
    center, scale, standardization = _quiet_standardization(
        residual, quiet_count, 10.0
    )
    centered = residual - center[None]
    positive_maximum = _display_scale(
        np.maximum(centered, 0),
        float(config.visualization["positive_upper_percentile"]), config,
    )
    roi_selected, roi_matrix = _roi_matrix(
        residual.shape[1:], labels, int(config.evaluation["roi_radius_px"])
    )
    synthetic_observed, synthetic_truth, synthetic_scale = _synthetic_fixture(
        centered, scale, config
    )
    _atomic_json(partial / "common_contract.json", {
        "input_lane": config.input_lane, "innovation_calibration": innovation,
        "quiet_standardization": standardization,
        "signal_positive_shared_maximum": positive_maximum,
        "frame_count": len(residual), "shape": list(residual.shape[1:]),
        "variant_ids": list(VARIANT_IDS),
        "synthetic": {
            "frames": len(synthetic_truth), "shape": list(synthetic_truth.shape[1:]),
            "types": ["centered_blob", "annular", "crowded_center", "crowded_annular"],
            "snr_multipliers": config.evaluation["synthetic_snr_multipliers"],
        },
    })
    rows = []
    for variant_index, (variant_id, parameters, function) in enumerate(
        _method_functions(quiet_count, config), start=1
    ):
        _progress(
            progress, "method_start", variant_id=variant_id,
            variant_index=variant_index, variant_total=len(VARIANT_IDS),
        )
        method_started = time.perf_counter()
        signal, diagnostics = function(centered, scale)
        runtime = time.perf_counter() - method_started
        if signal.shape != centered.shape or not np.isfinite(signal).all():
            raise RuntimeError(f"invalid signal output: {variant_id}")
        synthetic_estimate, _ = function(synthetic_observed, synthetic_scale)
        summary = _variant_metrics(
            variant_id, signal, residual, labels, roi_selected, roi_matrix,
            quiet_count, config, runtime, synthetic_estimate, synthetic_truth,
        )
        summary["parameters"] = parameters
        summary["method_diagnostics"] = diagnostics
        summary["videos"] = _write_tiffs(
            partial / "methods" / variant_id, signal, residual,
            positive_maximum, variant_id, parameters, config,
        )
        rows.append(summary)
        _atomic_json(partial / "checkpoint.json", {
            "phase": "sequential_methods", "completed": variant_index,
            "total": len(VARIANT_IDS), "last_variant": variant_id,
        })
        del signal, synthetic_estimate
        gc.collect()
    accepted = [row for row in rows if row["audit_pass"]]
    best_detection = max(
        rows,
        key=lambda row: (
            row["fixed_budget_mean_recall"],
            row["mean_recall"],
            -row["event_candidates"],
        ),
    )
    best_synthetic = min(rows, key=lambda row: row["synthetic_nmse"])
    conclusion = (
        f"{len(accepted)} of {len(rows)} methods passed the preregistered automatic "
        f"audit flags. The strongest fixed-budget detection candidate was "
        f"{bt}{best_detection['variant_id']}{bt} and the lowest synthetic NMSE was "
        f"{bt}{best_synthetic['variant_id']}{bt}. "
        "Visual remainder review remains required before scientific acceptance."
    )
    metrics = {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "status": "completed", "variant_count": len(rows), "variants": rows,
        "accepted_variant_ids": [row["variant_id"] for row in accepted],
        "best_quantitative_variant": best_detection["variant_id"],
        "best_detection_variant": best_detection["variant_id"],
        "best_synthetic_nmse_variant": best_synthetic["variant_id"],
        "conclusion": conclusion,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "scientific_contract": (
            "Automatic flags and sparse-label metrics do not replace visual review. "
            "Semi-synthetic truth is exact only for the injected fixtures."
        ),
    }
    _atomic_json(partial / "metrics.json", metrics)
    flat_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    _atomic_tsv(partial / "comparison.tsv", flat_rows)
    _write_final_report(partial / "REPORT.md", metrics)
    _atomic_json(partial / "run_state.json", {
        "status": "completed", "completed_unix": time.time(),
        "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"],
        "variant_count": len(rows), "tiff_count": 2 * len(rows),
    })
    partial.replace(config.output_dir)
    return metrics
