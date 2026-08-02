"""Real-data noisy-Parzen posterior split of Parzen Innovation into signal/noise.

This isolates the posterior-denoising concept required by Stage 2.  It is not
the complete patchwise ICA/subspace/qualification stage and reports that
limitation in every durable artifact.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np
import tifffile

from neurobench.algorithms.hierarchical_parzen_ica import noisy_parzen_posterior_mean
from neurobench.experiments.hierarchical_parzen_ica.architecture_lanes import (
    AffineICAReconstruction,
    calibrate_reference_parzen_innovation,
    quiet_median_background,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.experiments.pairwise_separation.evaluation import event_intervals

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
from .signal_noise_config import SignalNoiseConfig


SCIENTIFIC_STATUS = (
    "scalar_noisy_parzen_posterior_ablation_not_complete_patchwise_noisy_ica"
)


def _atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _coefficients(config: SignalNoiseConfig) -> AffineICAReconstruction:
    payload = json.loads(config.architecture_manifest.read_text(encoding="utf-8"))
    return AffineICAReconstruction.from_feedback(
        payload["raw_stochastic_fit"]["safety"]["feedback"]
    )


def preflight(config: SignalNoiseConfig, *, write_artifacts: bool = True) -> dict[str, Any]:
    inputs = (config.source_video, config.labels_tsv, config.architecture_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    shape = None
    dtype = None
    bounds_valid = labels_valid = fit_valid = finite = False
    labels: list[dict[str, Any]] = []
    if not missing:
        video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        shape, dtype = list(video.shape), str(video.dtype)
        start = int(config.frames["review_start_ui"]) - 1
        stop = int(config.frames["review_end_ui"])
        quiet_stop = int(config.frames["quiet_end_ui"])
        bounds_valid = video.ndim == 3 and 0 <= start < quiet_stop <= stop <= len(video)
        finite = bounds_valid and bool(np.isfinite(video[start:stop:20, ::16, ::16]).all())
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
        architecture = json.loads(config.architecture_manifest.read_text(encoding="utf-8"))
        fit = architecture.get("raw_stochastic_fit", {})
        fit_valid = bool(
            architecture.get("source_video") == str(config.source_video)
            and fit.get("classification_status") == "resolved"
            and fit.get("optimizer_converged") is True
            and fit.get("safety", {}).get("status") == "accepted"
        )
    frame_count = int(config.frames["review_end_ui"]) - int(config.frames["review_start_ui"]) + 1
    pixels = 0 if shape is None else int(shape[1]) * int(shape[2])
    dense_mib = frame_count * pixels * 4 / 2**20
    estimated_ram = 3.5 * dense_mib + 768
    uncompressed_output = 3 * frame_count * pixels * 2 / 2**20
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    gates = {
        "inputs_exist": not missing,
        "source_is_npy": config.source_video.suffix.lower() == ".npy",
        "frame_bounds_valid": bounds_valid,
        "finite_sample": finite,
        "labels_79_rows_27_identities_and_in_bounds": labels_valid,
        "accepted_fit_matches_source": fit_valid,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_ram <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk >= int(config.resources["min_free_disk_mib"]),
        "output_cap_sufficient": uncompressed_output <= int(config.resources["max_output_mib"]),
        "cpu_only": config.resources["device"] == "cpu",
    }
    payload = {
        "schema_version": 1, "kind": "read_only_noisy_parzen_signal_split_preflight",
        "experiment_id": config.experiment_id, "ready": all(gates.values()),
        "gates": gates, "missing": missing, "source_shape": shape, "source_dtype": dtype,
        "label_rows": len(labels), "roi_identities": len({row["roi_identity"] for row in labels}),
        "posterior_grid_combinations": (
            len(config.posterior["bandwidths"])
            * len(config.posterior["noise_variance_multipliers"])
        ),
        "scientific_status": SCIENTIFIC_STATUS,
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "available_ram_mib": _available_ram_mib(),
            "max_ram_mib": int(config.resources["max_ram_mib"]),
            "uncompressed_output_mib": uncompressed_output,
            "free_disk_mib": free_disk,
            "max_output_mib": int(config.resources["max_output_mib"]),
            "cpu_threads": int(config.resources["cpu_threads"]),
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs if path.is_file()
        ],
        "system_snapshot": _snapshots(),
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
        if not missing and bounds_valid:
            label_core._write_overlay(
                np.load(config.source_video, mmap_mode="r", allow_pickle=False),
                labels,
                config.preflight_dir / "label_projection_overlay.png",
            )
    if not payload["ready"]:
        raise RuntimeError(f"signal/noise preflight failed: {payload}")
    return payload


def _matching_preflight(config: SignalNoiseConfig) -> dict[str, Any]:
    audit = json.loads((config.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((config.preflight_dir / "config.resolved.json").read_text(encoding="utf-8"))
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a matching ready preflight")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return audit


def _innovation_residual(
    raw: np.ndarray,
    quiet_count: int,
    coefficients: AffineICAReconstruction,
    config: SignalNoiseConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    lane = config.input_lane
    calibration = calibrate_reference_parzen_innovation(
        raw, quiet_count, coefficients,
        frame_period_ms=float(config.frames["frame_period_ms"]),
        reference_half_life_seconds=float(lane["reference_half_life_seconds"]),
        correction_fraction=float(lane["correction_fraction"]),
        correction_clip_mad=float(lane["correction_clip_mad"]),
    )
    base = calibration.quiet_background.astype(np.float64)
    residual = np.empty_like(raw, dtype=np.float32)
    residual[0] = raw[0] - base
    state = base.copy()
    for index in range(1, len(raw)):
        state = (
            (1.0 - calibration.reference_refresh) * state
            + calibration.reference_refresh * raw[index]
        )
        correction = (
            coefficients.teacher_forced(raw[index - 1], raw[index])
            - state - calibration.correction_bias
        )
        background = state + calibration.correction_fraction * np.clip(
            correction, -calibration.correction_limit, calibration.correction_limit
        )
        residual[index] = raw[index] - background
    return residual, {
        "reference_refresh": calibration.reference_refresh,
        "quiet_correction_mad": calibration.quiet_correction_mad,
        "correction_limit": calibration.correction_limit,
    }


def _quiet_standardization(
    residual: np.ndarray, quiet_count: int, floor_percentile: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    quiet = residual[:quiet_count]
    center = np.median(quiet, axis=0).astype(np.float32)
    mad = (1.4826 * np.median(np.abs(quiet - center), axis=0)).astype(np.float32)
    positive = mad[mad > 0]
    floor = float(np.percentile(positive, floor_percentile)) if positive.size else 1.0
    scale = np.maximum(mad, max(floor, 1e-6)).astype(np.float32)
    return center, scale, {
        "quiet_scale_floor": floor,
        "quiet_scale_median": float(np.median(scale)),
        "quiet_scale_p95": float(np.percentile(scale, 95)),
    }


def _dictionary(
    residual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    quiet_count: int,
    config: SignalNoiseConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    p = config.posterior
    rng = np.random.default_rng(int(p["sample_seed"]))
    count = min(int(p["dictionary_sample_pixels"]), center.size)
    pixels = np.sort(rng.choice(center.size, size=count, replace=False))
    values = (
        residual[quiet_count:].reshape(len(residual) - quiet_count, -1)[:, pixels]
        - center.ravel()[pixels]
    ) / scale.ravel()[pixels]
    values = values.ravel()
    active = values[np.abs(values) >= float(p["dictionary_activation_abs_z"])]
    maximum = int(p["dictionary_centers"])
    zero_count = max(1, min(maximum - 2, int(round(
        maximum * float(p["dictionary_zero_mass_fraction"])
    ))))
    slab_count = maximum - zero_count
    if len(active) < slab_count:
        raise RuntimeError("too few label-free active samples for the Parzen slab")
    slab = np.quantile(active, np.linspace(0, 1, slab_count)).astype(np.float64)
    centers = np.concatenate([np.zeros(zero_count, dtype=np.float64), slab])
    return centers, {
        "center_count": len(centers), "zero_centers": zero_count,
        "slab_centers": slab_count, "active_sample_count": len(active),
        "fit_uses_labels": False, "fit_interval": "post_quiet_review_frames",
        "slab_min": float(np.min(slab)), "slab_max": float(np.max(slab)),
    }


def _lookup(
    centers: np.ndarray,
    bandwidth: float,
    noise_multiplier: float,
    config: SignalNoiseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    p = config.posterior
    grid = np.linspace(
        -float(p["lookup_abs_z"]), float(p["lookup_abs_z"]),
        int(p["lookup_points"]), dtype=np.float64,
    )
    posterior = noisy_parzen_posterior_mean(
        grid, centers, float(bandwidth), float(noise_multiplier)
    )
    return grid, np.asarray(posterior, dtype=np.float64)


def _apply_lookup(values: np.ndarray, grid: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, grid[0], grid[-1])
    return np.interp(clipped, grid, posterior).astype(np.float32)


def _selected_pixels_and_matrix(
    shape: tuple[int, int],
    labels: list[dict[str, Any]],
    config: SignalNoiseConfig,
):
    height, width = shape
    radius = int(config.selection["roi_radius_px"])
    yy, xx = np.ogrid[:height, :width]
    roi_mask = np.zeros(shape, dtype=bool)
    for row in labels:
        roi_mask |= (
            (xx - int(round(row["x_px"]))) ** 2
            + (yy - int(round(row["y_px"]))) ** 2 <= radius**2
        )
    rng = np.random.default_rng(int(config.posterior["sample_seed"]))
    count = min(int(config.posterior["dictionary_sample_pixels"]), height * width)
    uniform = rng.choice(height * width, size=count, replace=False)
    selected = np.unique(np.concatenate([np.flatnonzero(roi_mask), uniform]))
    _, matrix = _roi_mask_and_matrix(shape, labels, radius, selected)
    return selected, matrix


def _screen(
    residual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    centers: np.ndarray,
    labels: list[dict[str, Any]],
    quiet_count: int,
    config: SignalNoiseConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, matrix = _selected_pixels_and_matrix(residual.shape[1:], labels, config)
    observed = residual.reshape(len(residual), -1)[:, selected]
    z = (observed - center.ravel()[selected]) / scale.ravel()[selected]
    reference_traces = np.asarray(matrix @ observed.T).T
    intervals = event_intervals(labels, int(config.frames["review_start_ui"]))
    quiet_reference_rms = max(float(np.sqrt(np.mean(z[:quiet_count] ** 2))), 1e-6)
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    lookups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selection = config.selection
    for bandwidth in config.posterior["bandwidths"]:
        for noise in config.posterior["noise_variance_multipliers"]:
            lane_id = f"posterior_h{float(bandwidth):g}_n{float(noise):g}"
            grid, posterior = _lookup(centers, float(bandwidth), float(noise), config)
            signal_z = _apply_lookup(z, grid, posterior)
            signal = signal_z * scale.ravel()[selected]
            traces = np.asarray(matrix @ signal.T).T
            lane_observations = _roi_observations(
                lane_id, traces, reference_traces, labels, intervals, quiet_count
            )
            observations.extend(lane_observations)
            summary = _aggregate_observations(lane_observations)
            quiet_ratio = float(
                np.sqrt(np.mean(signal_z[:quiet_count] ** 2)) / quiet_reference_rms
            )
            noise_z = z - signal_z
            lag_pairs = noise_z[:quiet_count]
            numerator = float(np.mean(lag_pairs[1:] * lag_pairs[:-1]))
            denominator = max(float(np.mean(lag_pairs[:-1] ** 2)), 1e-6)
            summary.update({
                "lane_id": lane_id, "bandwidth": float(bandwidth),
                "noise_variance_multiplier": float(noise),
                "quiet_signal_rms_ratio": quiet_ratio,
                "quiet_noise_lag1_correlation": numerator / denominator,
                "posterior_mean_absolute_shrinkage": float(np.mean(np.abs(z - signal_z))),
                "lookup_monotonic": bool(np.all(np.diff(posterior) >= -1e-10)),
            })
            passed = bool(
                summary["median_peak_retention"] >= float(selection["minimum_peak_retention"])
                and summary["median_area_retention"] >= float(selection["minimum_area_retention"])
                and summary["median_late_retention"] >= float(selection["minimum_late_retention"])
                and summary["median_waveform_correlation"] >= float(
                    selection["minimum_waveform_correlation"]
                )
                and quiet_ratio <= float(selection["maximum_quiet_signal_rms_ratio"])
            )
            summary["selection_gate_pass"] = passed
            summary["selection_score"] = float(
                0.25 * min(summary["median_peak_retention"], 1.25)
                + 0.25 * min(summary["median_area_retention"], 1.25)
                + 0.20 * min(summary["median_late_retention"], 1.25)
                + 0.15 * np.clip(summary["median_waveform_correlation"], -1, 1)
                + 0.15 * (1.0 - min(quiet_ratio, 1.5) / 1.5)
            )
            rows.append(summary)
            lookups[lane_id] = (grid, posterior)
    winner = max(
        rows,
        key=lambda row: (
            row["selection_gate_pass"], row["selection_score"],
            -row["quiet_signal_rms_ratio"], row["lane_id"],
        ),
    )
    grid, posterior = lookups[str(winner["lane_id"])]
    return rows, observations, {
        "winner": winner, "lookup_grid": grid, "lookup_posterior": posterior,
        "selection_uses_labels": True,
        "selection_semantics": "exploratory real-data visualization, not unbiased performance",
    }


def _display_scales(
    residual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    grid: np.ndarray,
    posterior: np.ndarray,
    config: SignalNoiseConfig,
) -> dict[str, Any]:
    v = config.visualization
    sampled = residual[
        :: int(v["sample_frame_stride"]),
        :: int(v["sample_row_stride"]),
        :: int(v["sample_column_stride"]),
    ]
    sampled_center = center[
        :: int(v["sample_row_stride"]), :: int(v["sample_column_stride"])
    ]
    sampled_scale = scale[
        :: int(v["sample_row_stride"]), :: int(v["sample_column_stride"])
    ]
    z = (sampled - sampled_center) / sampled_scale
    signal = _apply_lookup(z, grid, posterior) * sampled_scale
    noise = sampled - signal
    magnitude = max(float(np.percentile(
        np.abs(np.concatenate([signal.ravel(), noise.ravel()])),
        float(v["signed_absolute_percentile"]),
    )), 1e-6)
    positive = max(float(np.percentile(
        np.maximum(signal, 0), float(v["positive_upper_percentile"])
    )), 1e-6)
    return {
        "signed_shared_source_limits": [-magnitude, magnitude],
        "signed_display_limits": [0, 65535], "signed_display_zero": 32768,
        "positive_signal_source_limits": [0, positive],
        "positive_display_limits": [0, 65535],
        "fixed_across_frames": True, "signal_noise_share_signed_scale": True,
        "sampling": {
            "frame_stride": int(v["sample_frame_stride"]),
            "row_stride": int(v["sample_row_stride"]),
            "column_stride": int(v["sample_column_stride"]),
        },
        "display_only": True,
    }


def _signed(frame: np.ndarray, magnitude: float) -> np.ndarray:
    return np.rint(
        (np.clip(frame / max(magnitude, 1e-6), -1, 1) + 1) * 32767.5
    ).astype(np.uint16)


def _positive(frame: np.ndarray, maximum: float) -> np.ndarray:
    return np.rint(
        np.clip(frame / max(maximum, 1e-6), 0, 1) * 65535
    ).astype(np.uint16)


def _write_videos(
    root: Path,
    residual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    grid: np.ndarray,
    posterior: np.ndarray,
    scales: dict[str, Any],
    winner: dict[str, Any],
    config: SignalNoiseConfig,
) -> dict[str, Any]:
    paths = {
        "signal_signed": root / "parzen_signal.tif",
        "noise_signed": root / "parzen_noise.tif",
        "signal_positive": root / "parzen_signal_positive.tif",
    }
    temporary = {name: path.with_name(path.name + ".partial") for name, path in paths.items()}
    description = {
        "experiment_id": config.experiment_id,
        "selected_lane": winner["lane_id"],
        "scientific_status": SCIENTIFIC_STATUS,
        "closure": "Parzen Innovation residual = posterior signal + residual noise",
        "display_scales": scales,
    }
    magnitude = float(scales["signed_shared_source_limits"][1])
    positive_maximum = float(scales["positive_signal_source_limits"][1])
    closure_max = 0.0
    signal_energy = noise_energy = 0.0
    with tifffile.TiffWriter(temporary["signal_signed"], bigtiff=True) as signal_writer, \
         tifffile.TiffWriter(temporary["noise_signed"], bigtiff=True) as noise_writer, \
         tifffile.TiffWriter(temporary["signal_positive"], bigtiff=True) as positive_writer:
        for index, frame in enumerate(residual):
            z = (frame - center) / scale
            signal = _apply_lookup(z, grid, posterior) * scale
            noise = frame - signal
            closure_max = max(closure_max, float(np.max(np.abs(frame - signal - noise))))
            signal_energy += float(np.sum(np.asarray(signal, dtype=np.float64) ** 2))
            noise_energy += float(np.sum(np.asarray(noise, dtype=np.float64) ** 2))
            first_description = json.dumps(description, sort_keys=True) if index == 0 else None
            signal_writer.write(
                _signed(signal, magnitude), photometric="minisblack",
                compression=config.visualization["compression"], metadata=None,
                description=first_description,
            )
            noise_writer.write(
                _signed(noise, magnitude), photometric="minisblack",
                compression=config.visualization["compression"], metadata=None,
                description=first_description,
            )
            positive_writer.write(
                _positive(np.maximum(signal, 0), positive_maximum),
                photometric="minisblack", compression=config.visualization["compression"],
                metadata=None, description=first_description,
            )
    for name, path in paths.items():
        temporary[name].replace(path)
        with tifffile.TiffFile(path) as tiff:
            if (
                len(tiff.pages) != len(residual)
                or tiff.pages[0].shape != residual.shape[1:]
                or tiff.pages[0].dtype != np.dtype(np.uint16)
            ):
                raise RuntimeError(f"TIFF verification failed: {path}")
    total = signal_energy + noise_energy
    return {
        "files": {name: path.name for name, path in paths.items()},
        "frame_count": len(residual), "shape": list(residual.shape[1:]),
        "closure_max_absolute": closure_max,
        "signal_energy_fraction_nonorthogonal": signal_energy / max(total, 1e-12),
        "noise_energy_fraction_nonorthogonal": noise_energy / max(total, 1e-12),
    }


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    winner = metrics["selection"]["winner"]
    lines = [
        f"# {metrics['experiment_id']}", "",
        f"Status: `{metrics['status']}`.", "",
        "## Result", "",
        (
            f"The selected noisy-Parzen posterior used bandwidth `{winner['bandwidth']}` "
            f"and standardized noise variance multiplier "
            f"`{winner['noise_variance_multiplier']}`."
        ), "",
        (
            f"Median labeled-ROI peak, area, and late retention were "
            f"`{winner['median_peak_retention']:.4f}`, "
            f"`{winner['median_area_retention']:.4f}`, and "
            f"`{winner['median_late_retention']:.4f}`. Quiet signal RMS ratio was "
            f"`{winner['quiet_signal_rms_ratio']:.4f}`."
        ), "",
        "## Videos", "",
        "- `parzen_signal.tif`: signed posterior signal, mid-gray zero.",
        "- `parzen_noise.tif`: signed residual remainder, mid-gray zero.",
        "- `parzen_signal_positive.tif`: positive posterior signal, black zero.",
        "",
        "Signal and noise use the same fixed signed scale. Every TIFF contains the "
        "complete 560-frame review interval. The split has exact arithmetic closure.",
        "",
        "## Scientific limitation", "",
        (
            "This isolates scalar noise-convolved Parzen posterior shrinkage on the "
            "Parzen Innovation residual. It does not implement the complete patchwise "
            "noise-corrected subspace, ICA demixing, overlap-add, or neural/artifact "
            "qualification stage. Visual quality is diagnostic, not proof that the "
            "noise remainder is pure measurement noise."
        ), "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: SignalNoiseConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    _atomic_json(partial / "preflight.json", audit)
    progress = partial / "progress.jsonl"
    started = time.time()
    _progress(progress, "load_input")
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[start:stop], dtype=np.float32)
    labels = label_core.load_labels(config.labels_tsv)
    quiet_count = int(config.frames["quiet_end_ui"]) - int(config.frames["quiet_start_ui"]) + 1
    _progress(progress, "build_parzen_innovation")
    residual, innovation = _innovation_residual(
        raw, quiet_count, _coefficients(config), config
    )
    del raw
    gc.collect()
    center, scale, standardization = _quiet_standardization(
        residual, quiet_count,
        float(config.posterior["quiet_scale_floor_percentile"]),
    )
    centers, dictionary = _dictionary(
        residual, center, scale, quiet_count, config
    )
    _atomic_json(partial / "dictionary.json", {
        **dictionary, "centers": centers.tolist(),
    })
    _progress(
        progress, "screen_posterior_grid",
        combinations=(
            len(config.posterior["bandwidths"])
            * len(config.posterior["noise_variance_multipliers"])
        ),
    )
    screen, observations, selection = _screen(
        residual, center, scale, centers, labels, quiet_count, config
    )
    _atomic_tsv(partial / "posterior_grid.tsv", screen)
    _atomic_tsv(partial / "roi_observation_metrics.tsv", observations)
    _atomic_json(partial / "selection.json", {
        key: value for key, value in selection.items()
        if not isinstance(value, np.ndarray)
    })
    scales = _display_scales(
        residual, center, scale, selection["lookup_grid"],
        selection["lookup_posterior"], config,
    )
    _atomic_json(partial / "display_scales.json", scales)
    _progress(progress, "write_tiffs", selected_lane=selection["winner"]["lane_id"])
    videos = _write_videos(
        partial, residual, center, scale, selection["lookup_grid"],
        selection["lookup_posterior"], scales, selection["winner"], config,
    )
    metrics = {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "status": "completed", "scientific_status": SCIENTIFIC_STATUS,
        "input_lane": config.input_lane, "innovation_calibration": innovation,
        "quiet_standardization": standardization, "dictionary": dictionary,
        "posterior_grid_combinations": len(screen), "selection": {
            "winner": selection["winner"],
            "selection_uses_labels": selection["selection_uses_labels"],
            "selection_semantics": selection["selection_semantics"],
        },
        "videos": videos, "display_scales": scales,
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "interpretation": (
            "Visual posterior-denoising diagnostic only. The noise remainder may contain "
            "structured neural or artifact signal until patchwise ICA and qualification pass."
        ),
    }
    _atomic_json(partial / "metrics.json", metrics)
    _write_report(partial / "REPORT.md", metrics)
    _atomic_json(partial / "run_state.json", {
        "status": "completed", "completed_unix": time.time(),
        "elapsed_seconds": metrics["elapsed_seconds"],
        "max_rss_mib": metrics["max_rss_mib"],
        "tiff_count": 3, "frame_count_per_tiff": len(residual),
    })
    partial.replace(config.output_dir)
    return metrics
