"""Broad-to-deep real-data architecture grid for stochastic Parzen separation.

The screen is deliberately cheap: every architecture sees all labeled ROI
disks plus bounded proxy strata, but no dense video is materialized.  Only the
union of leave-one-burst-out finalists receives full-field detection.  This
keeps 193 architecture hypotheses feasible while preserving the frozen sparse-
positive detection contract.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import tifffile

from neurobench.experiments.hierarchical_parzen_ica.architecture_lanes import (
    AffineICAReconstruction,
    calibrate_reference_parzen_innovation,
    quiet_median_background,
)
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

from .innovation_grid_config import InnovationGridConfig


RAW_DIRECT_EXPECTED = 0.6056159420289855
CONTROL_LANES = (
    "raw_direct",
    "teacher_forced_stochastic",
    "raw_stochastic_recurrence",
    "current_fixed_point_recurrence",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_ram_mib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def _snapshots() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"processes": [], "gpu": []}
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,comm,%cpu,%mem,rss", "--sort=-rss"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        result["processes"] = completed.stdout.splitlines()[:11]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=True, capture_output=True, text=True, timeout=5,
        )
        result["gpu"] = completed.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _grid_counts(config: InnovationGridConfig) -> dict[str, int]:
    return {
        "innovation_unique": len(config.innovation_specs),
        "fixed_point": len(config.fixed_specs),
        "screen_total": len(config.innovation_specs) + len(config.fixed_specs),
        "always_evaluated_controls": len(CONTROL_LANES),
    }


def preflight(config: InnovationGridConfig, *, write_artifacts: bool = True) -> dict[str, Any]:
    """Validate exact inputs, collisions, resource headroom, labels, and fit."""
    inputs = (config.source_video, config.labels_tsv, config.architecture_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    source_shape: list[int] | None = None
    source_dtype: str | None = None
    bounds_valid = labels_valid = fit_valid = finite_sample = False
    labels: list[dict[str, Any]] = []
    if not missing:
        video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        source_shape = list(video.shape)
        source_dtype = str(video.dtype)
        f = config.frames
        start, stop = int(f["review_start_ui"]) - 1, int(f["review_end_ui"])
        quiet_stop = int(f["quiet_end_ui"])
        bounds_valid = video.ndim == 3 and 0 <= start < quiet_stop <= stop <= len(video)
        if bounds_valid:
            finite_sample = bool(np.isfinite(video[start:stop:20, ::16, ::16]).all())
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
            and fit.get("safety", {}).get("reference_anchoring", {}).get(
                "accepted_learned_fraction"
            ) == 1.0
        )
    frames = int(config.frames["review_end_ui"]) - int(config.frames["review_start_ui"]) + 1
    pixels = 0 if not source_shape else source_shape[1] * source_shape[2]
    dense_stack_mib = frames * pixels * 4 / 2**20
    estimated_peak_ram_mib = 4.25 * dense_stack_mib + 1024
    visual_count = int(config.screening["global_visual_finalists"])
    uncompressed_visual_mib = visual_count * 2 * frames * pixels * 2 / 2**20
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk_mib = shutil.disk_usage(probe).free / 2**20
    gates = {
        "inputs_exist": not missing,
        "source_is_npy": config.source_video.suffix.lower() == ".npy",
        "frame_bounds_valid": bounds_valid,
        "finite_sample": finite_sample,
        "labels_79_rows_27_identities_and_in_bounds": labels_valid,
        "accepted_full_fraction_fit_matches_source": fit_valid,
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "preflight_separate_from_output": config.preflight_dir != config.output_dir,
        "ram_cap_sufficient": estimated_peak_ram_mib <= int(config.resources["max_ram_mib"]),
        "available_ram_sufficient": estimated_peak_ram_mib <= _available_ram_mib(),
        "disk_headroom_sufficient": free_disk_mib >= int(config.resources["min_free_disk_mib"]),
        "output_cap_sufficient": uncompressed_visual_mib <= int(config.resources["max_output_mib"]),
        "cpu_only": config.resources["device"] == "cpu",
    }
    payload = {
        "schema_version": 1,
        "kind": "read_only_spon_stochastic_architecture_grid_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "missing": missing,
        "source_shape": source_shape,
        "source_dtype": source_dtype,
        "review_interval_ui_inclusive": [
            int(config.frames["review_start_ui"]), int(config.frames["review_end_ui"])
        ],
        "quiet_interval_ui_inclusive": [
            int(config.frames["quiet_start_ui"]), int(config.frames["quiet_end_ui"])
        ],
        "label_rows": len(labels),
        "roi_identities": len({row["roi_identity"] for row in labels}),
        "grid": _grid_counts(config),
        "selection_contract": (
            "Per-fold finalists are selected using the other three labeled bursts; "
            "held-out detection is not used for promotion."
        ),
        "sparse_label_contract": (
            "Known positives identify recall. Unmatched event candidates remain unknown, "
            "so candidate burden and lower-bound known-label yield are not precision."
        ),
        "resources": {
            "cpu_threads": int(config.resources["cpu_threads"]),
            "estimated_peak_ram_mib": estimated_peak_ram_mib,
            "max_ram_mib": int(config.resources["max_ram_mib"]),
            "available_ram_mib": _available_ram_mib(),
            "free_disk_mib": free_disk_mib,
            "uncompressed_visual_mib": uncompressed_visual_mib,
            "max_output_mib": int(config.resources["max_output_mib"]),
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
        raise RuntimeError(f"architecture-grid preflight failed: {payload}")
    return payload


def _matching_preflight(config: InnovationGridConfig) -> dict[str, Any]:
    preflight_path = config.preflight_dir / "preflight.json"
    resolved_path = config.preflight_dir / "config.resolved.json"
    if not preflight_path.is_file() or not resolved_path.is_file():
        raise RuntimeError("run requires a completed matching preflight")
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not payload.get("ready") or resolved != config.to_dict():
        raise RuntimeError("preflight does not match the resolved run configuration")
    if config.output_dir.exists() or Path(str(config.output_dir) + ".partial").exists():
        raise FileExistsError("completed or partial output already exists")
    return payload


def _progress(path: Path, stage: str, **details: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"time_unix": time.time(), "stage": stage, **details}, sort_keys=True)
            + "\n"
        )
        stream.flush()


def _coefficients(config: InnovationGridConfig) -> AffineICAReconstruction:
    payload = json.loads(config.architecture_manifest.read_text(encoding="utf-8"))
    return AffineICAReconstruction.from_feedback(
        payload["raw_stochastic_fit"]["safety"]["feedback"]
    )


def _roi_mask_and_matrix(
    shape: tuple[int, int],
    labels: list[dict[str, Any]],
    radius: int,
    selected_flat: np.ndarray,
):
    from scipy.sparse import csr_matrix

    height, width = shape
    flat_to_selected = {int(value): index for index, value in enumerate(selected_flat)}
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    roi_mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    for row_index, label in enumerate(labels):
        disk = (
            (xx - int(round(label["x_px"]))) ** 2
            + (yy - int(round(label["y_px"]))) ** 2
            <= radius**2
        )
        roi_mask |= disk
        flats = np.flatnonzero(disk)
        kept = [flat_to_selected[int(value)] for value in flats if int(value) in flat_to_selected]
        if not kept:
            raise RuntimeError("ROI selection lost a labeled disk")
        weight = 1.0 / len(kept)
        rows.extend([row_index] * len(kept))
        columns.extend(kept)
        weights.extend([weight] * len(kept))
    return roi_mask, csr_matrix(
        (weights, (rows, columns)),
        shape=(len(labels), len(selected_flat)),
        dtype=np.float32,
    )


def _bounded_sample(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    values = np.flatnonzero(mask)
    if len(values) <= count:
        return values
    return np.sort(rng.choice(values, size=count, replace=False))


def _proxy_selection(
    frames: np.ndarray,
    labels: list[dict[str, Any]],
    config: InnovationGridConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any], Any]:
    from scipy.ndimage import sobel

    quiet_count = int(config.frames["quiet_end_ui"]) - int(config.frames["quiet_start_ui"]) + 1
    quiet = np.asarray(frames[:quiet_count], dtype=np.float32)
    median = np.median(quiet, axis=0)
    mad = 1.4826 * np.median(np.abs(quiet - median), axis=0)
    gradient = np.hypot(sobel(median, axis=0), sobel(median, axis=1))
    radius = int(config.screening["roi_radius_px"])
    yy, xx = np.ogrid[: frames.shape[1], : frames.shape[2]]
    roi_mask = np.zeros(frames.shape[1:], dtype=bool)
    for row in labels:
        roi_mask |= (
            (xx - int(round(row["x_px"]))) ** 2
            + (yy - int(round(row["y_px"]))) ** 2
            <= radius**2
        )
    bright = np.percentile(median, 99.8)
    stable = mad <= np.percentile(mad, 55)
    saturated_fraction = np.mean(quiet >= np.iinfo(frames.dtype).max, axis=0)
    artifact = ((median >= bright) & stable) | (saturated_fraction >= 0.8)
    artifact &= ~roi_mask
    anatomy = (
        (median >= np.percentile(median, 45))
        & (median <= np.percentile(median, 98))
        & (gradient >= np.percentile(gradient, 55))
        & (mad <= np.percentile(mad, 70))
        & ~artifact
        & ~roi_mask
    )
    background = (
        (median <= np.percentile(median, 35))
        & (mad <= np.percentile(mad, 55))
        & ~roi_mask
    )
    post = np.asarray(frames[quiet_count:], dtype=np.float32)
    post_drive = np.mean(np.abs(np.diff(post, axis=0)), axis=0)
    active = (
        (post_drive >= np.percentile(post_drive, 97))
        & ~artifact
        & ~roi_mask
    )
    count = int(config.screening["proxy_pixels_per_stratum"])
    rng = np.random.default_rng(int(config.screening["random_seed"]))
    flat_by_stratum = {
        "artifact": _bounded_sample(artifact, count, rng),
        "anatomy": _bounded_sample(anatomy, count, rng),
        "background": _bounded_sample(background, count, rng),
        "active_unlabeled": _bounded_sample(active, count, rng),
    }
    flat_by_stratum["uniform"] = np.sort(
        rng.choice(median.size, size=min(count, median.size), replace=False)
    )
    roi_flats = np.flatnonzero(roi_mask)
    selected = np.unique(np.concatenate([roi_flats, *flat_by_stratum.values()]))
    _, roi_matrix = _roi_mask_and_matrix(
        frames.shape[1:], labels, radius, selected
    )
    selected_lookup = {int(value): index for index, value in enumerate(selected)}
    positions = {
        name: np.asarray([selected_lookup[int(value)] for value in flats], dtype=np.int64)
        for name, flats in flat_by_stratum.items()
    }
    summary = {
        "definitions": {
            "artifact": "quiet p99.8 brightness and stable, or saturated in >=80% quiet frames; labeled disks excluded",
            "anatomy": "stable mid/high-intensity spatial-gradient pixels; artifact and labeled disks excluded",
            "background": "stable lower-intensity pixels; labeled disks excluded",
            "active_unlabeled": "top 3% post-quiet absolute first-difference pixels; artifact and labels excluded",
            "uniform": "bounded uniform sample used for global calibration",
        },
        "full_mask_pixels": {
            "artifact": int(artifact.sum()), "anatomy": int(anatomy.sum()),
            "background": int(background.sum()), "active_unlabeled": int(active.sum()),
            "labeled_roi_union": int(roi_mask.sum()),
        },
        "sample_pixels": {name: len(values) for name, values in positions.items()},
        "selected_union_pixels": len(selected),
        "proxy_semantics": "algorithmic strata for screening, not manually verified ground truth",
    }
    return selected, positions, summary, roi_matrix


def _roi_observations(
    lane_id: str,
    traces: np.ndarray,
    reference: np.ndarray,
    labels: list[dict[str, Any]],
    intervals: dict[int, tuple[int, int]],
    quiet_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        start, stop = intervals[int(label["burst_id"])]
        ref_center = float(np.median(reference[:quiet_count, index]))
        out_center = float(np.median(traces[:quiet_count, index]))
        ref_quiet_mad = max(
            float(1.4826 * np.median(np.abs(reference[:quiet_count, index] - ref_center))),
            1e-6,
        )
        ref = np.maximum(reference[start:stop, index] - ref_center, 0)
        out = np.maximum(traces[start:stop, index] - out_center, 0)
        ref_peak = float(np.max(ref))
        eligible = ref_peak > 3.0 * ref_quiet_mad
        late_start = start + max(1, (stop - start) * 2 // 3)
        ref_late = np.maximum(reference[late_start:stop, index] - ref_center, 0)
        out_late = np.maximum(traces[late_start:stop, index] - out_center, 0)
        if np.std(ref) > 1e-9 and np.std(out) > 1e-9:
            correlation = float(np.corrcoef(ref, out)[0, 1])
        else:
            correlation = 0.0
        rows.append({
            "lane_id": lane_id,
            "observation_index": index,
            "roi_identity": label["roi_identity"],
            "burst_id": int(label["burst_id"]),
            "eligible": eligible,
            "reference_peak": ref_peak,
            "reference_quiet_mad": ref_quiet_mad,
            "peak_retention": float(np.max(out) / max(ref_peak, 1e-6)),
            "area_retention": float(np.sum(out) / max(float(np.sum(ref)), 1e-6)),
            "late_retention": float(np.sum(out_late) / max(float(np.sum(ref_late)), 1e-6)),
            "waveform_correlation": correlation,
        })
    return rows


def _aggregate_observations(
    observations: list[dict[str, Any]], excluded_burst: int | None = None
) -> dict[str, float | int]:
    used = [
        row for row in observations
        if row["eligible"] and (excluded_burst is None or row["burst_id"] != excluded_burst)
    ]
    if not used:
        return {
            "eligible_observations": 0, "median_peak_retention": 0.0,
            "median_area_retention": 0.0, "median_late_retention": 0.0,
            "median_waveform_correlation": 0.0,
        }
    return {
        "eligible_observations": len(used),
        "median_peak_retention": float(np.median([row["peak_retention"] for row in used])),
        "median_area_retention": float(np.median([row["area_retention"] for row in used])),
        "median_late_retention": float(np.median([row["late_retention"] for row in used])),
        "median_waveform_correlation": float(
            np.median([row["waveform_correlation"] for row in used])
        ),
    }


def _score(metrics: dict[str, Any], gates: dict[str, Any]) -> tuple[float, bool]:
    peak = float(metrics["median_peak_retention"])
    area = float(metrics["median_area_retention"])
    late = float(metrics["median_late_retention"])
    corr = float(metrics["median_waveform_correlation"])
    quiet = float(metrics["quiet_rms_ratio"])
    artifact = float(metrics["artifact_dynamics_ratio"])
    active = float(metrics["active_unlabeled_dynamics_ratio"])
    score = (
        0.25 * min(peak, 1.5)
        + 0.25 * min(area, 1.5)
        + 0.25 * min(late, 1.5)
        + 0.10 * np.clip(corr, -1, 1)
        + 0.10 * min(active, 1.5)
        + 0.05 * max(0.0, 1.0 - min(artifact, 2.0) / 2.0)
        - 0.10 * max(0.0, quiet - 1.0)
    )
    passed = bool(
        peak >= float(gates["minimum_peak_retention"])
        and area >= float(gates["minimum_area_retention"])
        and late >= float(gates["minimum_late_retention"])
        and corr >= float(gates["minimum_waveform_correlation"])
        and quiet <= float(gates["maximum_quiet_rms_ratio"])
        and artifact <= float(gates["maximum_artifact_dynamics_ratio"])
    )
    return float(score), passed


def _proxy_accumulators(count: int, strata: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"{name}_{phase}": np.zeros(count, dtype=np.float64)
        for name in strata
        for phase in ("quiet_sq", "post_sq")
    }


def _accumulate_proxy(
    accumulators: dict[str, np.ndarray],
    dynamics: np.ndarray,
    strata: dict[str, np.ndarray],
    *,
    quiet: bool,
) -> None:
    phase = "quiet_sq" if quiet else "post_sq"
    for name, positions in strata.items():
        if len(positions):
            accumulators[f"{name}_{phase}"] += np.mean(
                np.asarray(dynamics[:, positions], dtype=np.float64) ** 2,
                axis=1,
            )


def _proxy_metrics(
    accumulators: dict[str, np.ndarray],
    input_reference: dict[str, float],
    quiet_count: int,
    post_count: int,
) -> list[dict[str, float]]:
    result = []
    for index in range(len(accumulators["uniform_quiet_sq"])):
        def ratio(name: str, phase: str, count: int) -> float:
            rms = math.sqrt(accumulators[f"{name}_{phase}_sq"][index] / max(count, 1))
            return rms / max(input_reference[f"{name}_{phase}_rms"], 1e-6)

        result.append({
            "quiet_rms_ratio": ratio("uniform", "quiet", quiet_count),
            "artifact_dynamics_ratio": ratio("artifact", "post", post_count),
            "anatomy_dynamics_ratio": ratio("anatomy", "post", post_count),
            "background_dynamics_ratio": ratio("background", "post", post_count),
            "active_unlabeled_dynamics_ratio": ratio("active_unlabeled", "post", post_count),
        })
    return result


def _input_proxy_reference(
    residual: np.ndarray,
    strata: dict[str, np.ndarray],
    quiet_count: int,
) -> dict[str, float]:
    result = {}
    for name, positions in strata.items():
        for phase, values in (
            ("quiet", residual[:quiet_count, positions]),
            ("post", residual[quiet_count:, positions]),
        ):
            result[f"{name}_{phase}_rms"] = float(
                np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2))
            )
    return result


def _screen_innovation_half_life(
    frames: np.ndarray,
    specs: list[dict[str, Any]],
    coefficients: AffineICAReconstruction,
    base: np.ndarray,
    roi_matrix: Any,
    strata: dict[str, np.ndarray],
    input_traces: np.ndarray,
    input_proxy: dict[str, float],
    labels: list[dict[str, Any]],
    intervals: dict[int, tuple[int, int]],
    quiet_count: int,
    config: InnovationGridConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    half_life = float(specs[0]["half_life_seconds"])
    refresh = 1.0 - 0.5 ** (
        float(config.frames["frame_period_ms"]) / (1000.0 * half_life)
    )
    state = base.copy()
    corrections = []
    for frame_index in range(1, quiet_count):
        current = frames[frame_index]
        previous = frames[frame_index - 1]
        state = (1.0 - refresh) * state + refresh * current
        corrections.append(
            coefficients.teacher_forced(previous, current) - state
        )
    correction_values = np.asarray(corrections, dtype=np.float32)
    bias = np.median(correction_values, axis=0)
    centered = correction_values - bias
    mad = max(float(1.4826 * np.median(np.abs(centered))), 1e-6)
    del correction_values, centered, corrections
    count = len(specs)
    fractions = np.asarray([spec["correction_fraction"] for spec in specs], dtype=np.float32)
    limits = np.asarray([spec["correction_clip_mad"] * mad for spec in specs], dtype=np.float32)
    traces = np.empty((count, len(frames), len(labels)), dtype=np.float32)
    accumulators = _proxy_accumulators(count, strata)
    state = base.copy()
    first_dynamics = frames[0] - base
    traces[:, 0] = np.asarray(roi_matrix @ first_dynamics).reshape(1, -1)
    _accumulate_proxy(
        accumulators, np.broadcast_to(first_dynamics, (count, len(base))),
        strata, quiet=True,
    )
    last_background = np.broadcast_to(base, (count, len(base))).copy()
    for frame_index in range(1, len(frames)):
        current = frames[frame_index]
        previous = frames[frame_index - 1]
        state = (1.0 - refresh) * state + refresh * current
        correction = coefficients.teacher_forced(previous, current) - state - bias
        clipped = np.clip(
            correction[None], -limits[:, None], limits[:, None]
        )
        background = state[None] + fractions[:, None] * clipped
        dynamics = current[None] - background
        traces[:, frame_index] = np.asarray(roi_matrix @ dynamics.T).T
        _accumulate_proxy(
            accumulators, dynamics, strata, quiet=frame_index < quiet_count
        )
        last_background = background
    proxy_rows = _proxy_metrics(
        accumulators, input_proxy, quiet_count, len(frames) - quiet_count
    )
    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    gates = config.screening["hard_gates"]
    for index, spec in enumerate(specs):
        lane_observations = _roi_observations(
            str(spec["lane_id"]), traces[index], input_traces, labels,
            intervals, quiet_count,
        )
        observations.extend(lane_observations)
        summary = {
            **spec, **_aggregate_observations(lane_observations), **proxy_rows[index],
            "quiet_correction_mad": mad,
            "final_background_median_absolute_drift": float(
                np.median(np.abs(last_background[index] - base))
            ),
        }
        summary["screen_score"], summary["hard_gate_pass"] = _score(summary, gates)
        summaries.append(summary)
    return summaries, observations


def _screen_fixed(
    frames: np.ndarray,
    specs: list[dict[str, Any]],
    base: np.ndarray,
    roi_matrix: Any,
    strata: dict[str, np.ndarray],
    input_traces: np.ndarray,
    input_proxy: dict[str, float],
    labels: list[dict[str, Any]],
    intervals: dict[int, tuple[int, int]],
    quiet_count: int,
    config: InnovationGridConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = len(specs)
    memory = np.asarray([spec["memory_coefficient"] for spec in specs], dtype=np.float32)
    current_gain = np.asarray([spec["current_coefficient"] for spec in specs], dtype=np.float32)
    state = np.broadcast_to(base, (count, len(base))).copy()
    traces = np.empty((count, len(frames), len(labels)), dtype=np.float32)
    accumulators = _proxy_accumulators(count, strata)
    dynamics = np.broadcast_to(frames[0] - base, state.shape)
    traces[:, 0] = np.asarray(roi_matrix @ dynamics.T).T
    _accumulate_proxy(accumulators, dynamics, strata, quiet=True)
    for frame_index in range(1, len(frames)):
        current_delta = frames[frame_index] - base
        state = (
            base[None]
            + memory[:, None] * (state - base[None])
            + current_gain[:, None] * current_delta[None]
        )
        dynamics = frames[frame_index][None] - state
        traces[:, frame_index] = np.asarray(roi_matrix @ dynamics.T).T
        _accumulate_proxy(
            accumulators, dynamics, strata, quiet=frame_index < quiet_count
        )
    proxy_rows = _proxy_metrics(
        accumulators, input_proxy, quiet_count, len(frames) - quiet_count
    )
    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    gates = config.screening["hard_gates"]
    for index, spec in enumerate(specs):
        lane_observations = _roi_observations(
            str(spec["lane_id"]), traces[index], input_traces, labels,
            intervals, quiet_count,
        )
        observations.extend(lane_observations)
        summary = {
            **spec, **_aggregate_observations(lane_observations), **proxy_rows[index],
            "final_background_median_absolute_drift": float(
                np.median(np.abs(state[index] - base))
            ),
        }
        summary["screen_score"], summary["hard_gate_pass"] = _score(summary, gates)
        summaries.append(summary)
    return summaries, observations


def _select_finalists(
    screen: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    config: InnovationGridConfig,
) -> dict[str, Any]:
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        by_lane.setdefault(str(row["lane_id"]), []).append(row)
    base_by_lane = {str(row["lane_id"]): row for row in screen}
    per_family = int(config.screening["finalists_per_family_per_fold"])
    gates = config.screening["hard_gates"]
    folds = []
    union: set[str] = set()
    for heldout in range(1, 5):
        ranked = []
        for lane_id, base_row in base_by_lane.items():
            metrics = {
                **base_row,
                **_aggregate_observations(by_lane[lane_id], excluded_burst=heldout),
            }
            score, passed = _score(metrics, gates)
            ranked.append({
                "heldout_burst": heldout, "lane_id": lane_id,
                "family": base_row["family"], "training_screen_score": score,
                "training_hard_gate_pass": passed,
            })
        promoted = []
        for family in ("innovation", "fixed_point"):
            family_rows = sorted(
                (row for row in ranked if row["family"] == family),
                key=lambda row: (
                    row["training_hard_gate_pass"], row["training_screen_score"],
                    row["lane_id"],
                ),
                reverse=True,
            )[:per_family]
            promoted.extend(family_rows)
            union.update(row["lane_id"] for row in family_rows)
        primary = max(
            promoted,
            key=lambda row: (
                row["training_hard_gate_pass"], row["training_screen_score"],
                row["lane_id"],
            ),
        )
        folds.append({
            "heldout_burst": heldout,
            "promoted": promoted,
            "primary_lane": primary["lane_id"],
            "selection_uses_heldout_labels": False,
        })
    global_ranked = sorted(
        screen,
        key=lambda row: (row["hard_gate_pass"], row["screen_score"], row["lane_id"]),
        reverse=True,
    )
    visual = [
        row["lane_id"]
        for row in global_ranked[: int(config.screening["global_visual_finalists"])]
    ]
    current = "innovation_h10_e0.1_c4"
    if current in base_by_lane and current not in visual:
        visual[-1] = current
    union.update(visual)
    return {
        "folds": folds,
        "full_field_lane_ids": sorted(union),
        "global_visual_lane_ids": visual,
        "global_visual_selection_uses_all_labels": True,
        "global_visual_semantics": "interpretive only; not an unbiased detection estimate",
    }


def _full_innovation_calibration(
    raw: np.ndarray,
    quiet_count: int,
    coefficients: AffineICAReconstruction,
    spec: dict[str, Any],
    config: InnovationGridConfig,
):
    return calibrate_reference_parzen_innovation(
        raw, quiet_count, coefficients,
        frame_period_ms=float(config.frames["frame_period_ms"]),
        reference_half_life_seconds=float(spec["half_life_seconds"]),
        correction_fraction=float(spec["correction_fraction"]),
        correction_clip_mad=max(float(spec["correction_clip_mad"]), 1.0),
    )


def _full_stack(
    lane_id: str,
    raw: np.ndarray,
    quiet_count: int,
    coefficients: AffineICAReconstruction,
    specs: dict[str, dict[str, Any]],
    config: InnovationGridConfig,
    calibration_cache: dict[float, Any],
) -> np.ndarray:
    base = quiet_median_background(raw, quiet_count).astype(np.float32)
    output = np.empty_like(raw, dtype=np.float32)
    output[0] = raw[0] - base
    if lane_id == "raw_direct":
        output[:] = raw
        return output
    if lane_id == "teacher_forced_stochastic":
        for index in range(1, len(raw)):
            background = coefficients.teacher_forced(raw[index - 1], raw[index])
            output[index] = raw[index] - background
        return output
    if lane_id == "raw_stochastic_recurrence":
        state = np.asarray(raw[0], dtype=np.float64)
        for index in range(1, len(raw)):
            state = coefficients.teacher_forced(state, raw[index])
            output[index] = raw[index] - state
        return output
    if lane_id == "current_fixed_point_recurrence":
        state = np.asarray(base, dtype=np.float64)
        for index in range(1, len(raw)):
            state = (
                base
                + coefficients.previous_coefficient * (state - base)
                + coefficients.current_coefficient * (raw[index] - base)
            )
            output[index] = raw[index] - state
        return output
    spec = specs[lane_id]
    if spec["family"] == "fixed_point":
        state = np.asarray(base, dtype=np.float32)
        memory = float(spec["memory_coefficient"])
        current = float(spec["current_coefficient"])
        for index in range(1, len(raw)):
            state = base + memory * (state - base) + current * (raw[index] - base)
            output[index] = raw[index] - state
        return output
    half_life = float(spec["half_life_seconds"])
    if half_life not in calibration_cache:
        calibration_cache[half_life] = _full_innovation_calibration(
            raw, quiet_count, coefficients, spec, config
        )
    calibration = calibration_cache[half_life]
    state = np.asarray(base, dtype=np.float64)
    fraction = float(spec["correction_fraction"])
    limit = float(spec["correction_clip_mad"]) * float(calibration.quiet_correction_mad)
    for index in range(1, len(raw)):
        state = (
            (1.0 - calibration.reference_refresh) * state
            + calibration.reference_refresh * raw[index]
        )
        correction = (
            coefficients.teacher_forced(raw[index - 1], raw[index])
            - state - calibration.correction_bias
        )
        background = state + fraction * np.clip(correction, -limit, limit)
        output[index] = raw[index] - background
    return output


def _normalize_positive_in_place(stack: np.ndarray, quiet_count: int) -> dict[str, float]:
    baseline = np.median(stack[:quiet_count], axis=0)
    low, high = np.percentile(stack[:quiet_count, ::4, ::4], [1.0, 99.9])
    scale = max(float(high - low), 1e-6)
    stack -= baseline
    stack /= scale
    np.maximum(stack, 0, out=stack)
    return {"quiet_baseline": "per_pixel_median", "global_scale": scale}


def _evaluate_detection(
    lane_id: str,
    stack: np.ndarray,
    labels: list[dict[str, Any]],
    config: InnovationGridConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[tuple[int, str]], dict[int, dict[str, Any]]]:
    d = config.detection
    tau = float(d["temporal_pool_tau"])
    quiet_maps = [
        temporal_pool(stack[start : start + duration], f"lme{tau}")
        for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS)
    ]
    thresholds = {
        str(float(rate)): quiet_calibrated_threshold(
            quiet_maps, int(d["nms_distance_px"]), float(rate), limit=3000
        )
        for rate in d["froc_quiet_peaks_per_map"]
    }
    primary = thresholds[str(float(d["quiet_false_peaks_per_map"]))]
    intervals = event_intervals(labels, int(config.frames["review_start_ui"]))
    folds = []
    candidates: list[dict[str, Any]] = []
    matched_keys: set[tuple[int, str]] = set()
    overlay: dict[int, dict[str, Any]] = {}
    radius_rows = {str(value): [] for value in d["match_radii_px"]}
    for burst, (start, stop) in intervals.items():
        score_map = temporal_pool(stack[start:stop], f"lme{tau}")
        ranked = extract_local_maxima(
            score_map, int(d["nms_distance_px"]),
            limit=int(d["candidate_cap_per_burst"]),
        )
        peaks = [peak for peak in ranked if peak[0] >= primary]
        rows = [row for row in labels if int(row["burst_id"]) == burst]
        matches, matched_peak_indices = match_peaks_one_to_one(
            peaks, rows, float(d["primary_match_radius_px"])
        )
        matched_label_indices = {item[0] for item in matches}
        for row_index in matched_label_indices:
            matched_keys.add((burst, str(rows[row_index]["roi_identity"])))
        fixed = ranked[: int(d["fixed_candidates_per_burst"])]
        fixed_matches = match_peaks_one_to_one(
            fixed, rows, float(d["primary_match_radius_px"])
        )[0]
        folds.append({
            "burst_id": burst, "labels": len(rows), "matched": len(matches),
            "recall": len(matches) / len(rows), "event_candidates": len(peaks),
            "fixed_budget_matched": len(fixed_matches),
            "fixed_budget_recall": len(fixed_matches) / len(rows),
        })
        for index, (value, x, y) in enumerate(peaks):
            nearest = min(math.hypot(x - row["x_px"], y - row["y_px"]) for row in rows)
            candidates.append({
                "lane_id": lane_id, "burst_id": burst, "score": value,
                "x_px": x, "y_px": y,
                "matched_known_label": index in matched_peak_indices,
                "nearest_known_label_px": nearest,
                "interpretation": (
                    "known_match" if index in matched_peak_indices
                    else "unmatched_candidate_truth_unknown"
                ),
            })
        for radius in d["match_radii_px"]:
            radius_rows[str(radius)].append(
                len(match_peaks_one_to_one(peaks, rows, float(radius))[0]) / len(rows)
            )
        overlay[burst] = {
            "score_map": score_map, "peaks": peaks, "labels": rows,
            "matched_peak_indices": matched_peak_indices,
            "matched_label_indices": matched_label_indices,
        }
    froc = []
    for rate, threshold in thresholds.items():
        recalls = []
        counts = []
        for burst, (start, stop) in intervals.items():
            score_map = overlay[burst]["score_map"]
            peaks = [
                peak for peak in extract_local_maxima(
                    score_map, int(d["nms_distance_px"]),
                    limit=int(d["candidate_cap_per_burst"]),
                )
                if peak[0] >= threshold
            ]
            rows = overlay[burst]["labels"]
            recalls.append(
                len(match_peaks_one_to_one(
                    peaks, rows, float(d["primary_match_radius_px"])
                )[0]) / len(rows)
            )
            counts.append(len(peaks))
        froc.append({
            "quiet_peaks_per_map_target": float(rate),
            "mean_recall": float(np.mean(recalls)),
            "mean_event_candidates": float(np.mean(counts)),
        })
    result = {
        "lane_id": lane_id, "outer_folds": folds,
        "mean_recall": float(np.mean([row["recall"] for row in folds])),
        "pooled_recall": sum(row["matched"] for row in folds) / sum(row["labels"] for row in folds),
        "total_matched": sum(row["matched"] for row in folds),
        "total_labels": sum(row["labels"] for row in folds),
        "total_event_candidates": sum(row["event_candidates"] for row in folds),
        "known_label_candidate_fraction_lower_bound": (
            sum(row["matched"] for row in folds)
            / max(1, sum(row["event_candidates"] for row in folds))
        ),
        "fixed_budget_mean_recall": float(
            np.mean([row["fixed_budget_recall"] for row in folds])
        ),
        "mean_recall_by_match_radius": {
            radius: float(np.mean(values)) for radius, values in radius_rows.items()
        },
        "primary_threshold": primary,
        "froc": froc,
        "precision_identified": False,
    }
    return result, candidates, matched_keys, overlay


def _circle(image: np.ndarray, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    height, width = image.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    for angle in np.linspace(0, 2 * math.pi, max(24, radius * 8), endpoint=False):
        px = int(round(cx + radius * math.cos(angle)))
        py = int(round(cy + radius * math.sin(angle)))
        if 0 <= px < width and 0 <= py < height:
            image[max(0, py - 1):min(height, py + 2), max(0, px - 1):min(width, px + 2)] = color


def _write_detection_maps(path: Path, lane_id: str, overlay: dict[int, dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with tifffile.TiffWriter(temporary) as writer:
        for page_index, burst in enumerate(sorted(overlay)):
            item = overlay[burst]
            score = item["score_map"]
            low, high = np.percentile(score, [1, 99.8])
            gray = np.rint(
                np.clip((score - low) / max(float(high - low), 1e-6), 0, 1) * 255
            ).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            for index, (_, x, y) in enumerate(item["peaks"]):
                _circle(
                    rgb, x, y,
                    (0, 255, 0) if index in item["matched_peak_indices"] else (255, 191, 0),
                    4,
                )
            for index, row in enumerate(item["labels"]):
                if index not in item["matched_label_indices"]:
                    _circle(rgb, row["x_px"], row["y_px"], (255, 0, 255), 6)
            description = {
                "lane_id": lane_id, "burst_id": burst,
                "green": "candidate matched to a known positive",
                "magenta": "known positive not matched",
                "amber": "unmatched candidate; truth unknown, not false positive",
                "temporal_semantics": "one temporally pooled score map for the complete burst window",
            }
            writer.write(
                rgb, photometric="rgb", compression="zlib", metadata=None,
                description=json.dumps(description, sort_keys=True) if page_index == 0 else None,
            )
    temporary.replace(path)


def _encode_linear(frame: np.ndarray, limits: list[float]) -> np.ndarray:
    low, high = map(float, limits)
    return np.rint(
        np.clip((np.asarray(frame) - low) / max(high - low, 1e-6), 0, 1) * 65535
    ).astype(np.uint16)


def _encode_signed(frame: np.ndarray, magnitude: float) -> np.ndarray:
    return np.rint(
        (np.clip(np.asarray(frame) / max(float(magnitude), 1e-6), -1, 1) + 1) * 32767.5
    ).astype(np.uint16)


def _write_visual_pair(
    root: Path,
    lane_id: str,
    raw: np.ndarray,
    dynamics: np.ndarray,
    config: InnovationGridConfig,
) -> None:
    manifest = json.loads(config.architecture_manifest.read_text(encoding="utf-8"))
    scales = manifest["display_normalization"]
    destination = root / "visuals" / lane_id
    destination.mkdir(parents=True, exist_ok=False)
    description = {
        "lane_id": lane_id,
        "review_interval_ui_inclusive": [
            int(config.frames["review_start_ui"]), int(config.frames["review_end_ui"])
        ],
        "shared_display_normalization_source": str(config.architecture_manifest),
        "display_only": True,
    }
    jobs = []
    if config.visualization["write_background"]:
        jobs.append((
            destination / "background.tif",
            (_encode_linear(raw[index] - dynamics[index], scales["background"]["source_limits"])
             for index in range(len(raw))),
            "minisblack",
        ))
    if config.visualization["write_dynamics"]:
        magnitude = scales["dynamics_noise"]["source_limits"][1]
        jobs.append((
            destination / "dynamics_noise.tif",
            (_encode_signed(dynamics[index], magnitude) for index in range(len(raw))),
            "minisblack",
        ))
    for path, pages, photometric in jobs:
        temporary = path.with_name(path.name + ".partial")
        with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
            for index, page in enumerate(pages):
                writer.write(
                    page, photometric=photometric,
                    compression=config.visualization["compression"], metadata=None,
                    description=json.dumps(description, sort_keys=True) if index == 0 else None,
                )
        temporary.replace(path)


def _bootstrap_crossvalidated(
    selected_keys: set[tuple[int, str]],
    raw_keys: set[tuple[int, str]],
    labels: list[dict[str, Any]],
    config: InnovationGridConfig,
) -> dict[str, Any]:
    identities = sorted({str(row["roi_identity"]) for row in labels})
    keys_by_identity = {
        identity: [
            (int(row["burst_id"]), str(row["roi_identity"]))
            for row in labels if str(row["roi_identity"]) == identity
        ]
        for identity in identities
    }
    rng = np.random.default_rng(int(config.screening["random_seed"]))
    samples = int(config.screening["bootstrap_samples"])
    differences = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        selected_hits = raw_hits = total = 0
        for identity in sampled:
            keys = keys_by_identity[str(identity)]
            total += len(keys)
            selected_hits += sum(key in selected_keys for key in keys)
            raw_hits += sum(key in raw_keys for key in keys)
        differences[index] = (selected_hits - raw_hits) / total
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "cluster_unit": "roi_identity", "identity_count": len(identities),
        "samples": samples, "seed": int(config.screening["random_seed"]),
        "point_pooled_recall_difference": (
            len(selected_keys) - len(raw_keys)
        ) / len(labels),
        "percentile_95_ci": [float(low), float(high)],
        "probability_difference_gt_zero": float(np.mean(differences > 0)),
        "discordant_gains": len(selected_keys - raw_keys),
        "discordant_losses": len(raw_keys - selected_keys),
    }


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        f"# {metrics['experiment_id']}", "",
        f"Status: `{metrics['status']}`.", "",
        "## What was tested", "",
        (
            f"The bounded screen evaluated **{metrics['grid']['screen_total']} unique "
            f"architectures**: {metrics['grid']['innovation_unique']} reference/Parzen "
            f"innovation variants and {metrics['grid']['fixed_point']} stable fixed-point "
            "variants. Zero-correction reference-only lanes were canonicalized, so clip "
            "settings were not redundantly repeated when epsilon was zero."
        ), "",
        "## Unbiased next-burst result", "",
        (
            f"Leave-one-burst-out promotion produced mean recall "
            f"`{metrics['crossvalidated_selection']['mean_recall']:.4f}` and pooled recall "
            f"`{metrics['crossvalidated_selection']['pooled_recall']:.4f}`. Raw Direct was "
            f"`{metrics['raw_direct']['mean_recall']:.4f}` mean recall."
        ), "",
        (
            "Each held-out burst was scored by a lane selected using only the other three "
            "bursts plus fixed proxy metrics. The ROI-identity bootstrap comparison is "
            f"`{metrics['paired_roi_bootstrap']['point_pooled_recall_difference']:+.4f}` "
            f"with 95% percentile interval "
            f"`{metrics['paired_roi_bootstrap']['percentile_95_ci']}`."
        ), "",
        "## Full-field finalists", "",
        "| Lane | Mean recall | Fixed-budget recall | Matches | Candidates | Known-label yield (lower bound) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        metrics["full_field_lanes"],
        key=lambda item: (item["mean_recall"], -item["total_event_candidates"]),
        reverse=True,
    ):
        lines.append(
            f"| `{row['lane_id']}` | {row['mean_recall']:.4f} | "
            f"{row['fixed_budget_mean_recall']:.4f} | {row['total_matched']} | "
            f"{row['total_event_candidates']} | "
            f"{row['known_label_candidate_fraction_lower_bound']:.4f} |"
        )
    lines.extend([
        "",
        "Unmatched event candidates remain unknown because annotations are sparse. "
        "The final column is a lower bound on known-label yield, not precision.",
        "",
        "## Visual audit", "",
        "Each selected visual lane contains `background.tif`, `dynamics_noise.tif`, "
        "and a four-page `detection_burst_maps.tif`. Green is a known match, magenta "
        "is a missed known label, and amber is an unmatched candidate with unknown truth.",
        "",
        "## Interpretation", "",
        metrics["decision"]["interpretation"], "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: InnovationGridConfig) -> dict[str, Any]:
    """Execute the bounded screen, cross-fitted promotion, and dense finalists."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources["cpu_threads"])
    audit = _matching_preflight(config)
    partial = Path(str(config.output_dir) + ".partial")
    partial.mkdir(parents=True)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    _atomic_json(partial / "preflight.json", audit)
    progress_path = partial / "progress.jsonl"
    started = time.time()
    _progress(progress_path, "load_inputs")
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["review_end_ui"])
    raw = np.asarray(source[start:stop], dtype=np.float32)
    labels = label_core.load_labels(config.labels_tsv)
    coefficients = _coefficients(config)
    quiet_count = int(config.frames["quiet_end_ui"]) - int(config.frames["quiet_start_ui"]) + 1
    intervals = event_intervals(labels, int(config.frames["review_start_ui"]))

    _progress(progress_path, "build_proxy_strata")
    selected, strata, proxy_summary, roi_matrix = _proxy_selection(
        source[start:stop], labels, config
    )
    small = raw.reshape(len(raw), -1)[:, selected]
    base = np.median(small[:quiet_count], axis=0).astype(np.float32)
    input_residual = small - base
    input_traces = np.asarray(roi_matrix @ input_residual.T).T
    input_proxy = _input_proxy_reference(input_residual, strata, quiet_count)
    _atomic_json(partial / "proxy_strata.json", proxy_summary)

    screen_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    innovation_specs = list(config.innovation_specs)
    for half_life in config.grid["innovation_half_life_seconds"]:
        group = [
            dict(spec) for spec in innovation_specs
            if float(spec["half_life_seconds"]) == float(half_life)
        ]
        _progress(
            progress_path, "screen_innovation_half_life",
            half_life_seconds=float(half_life), combinations=len(group),
        )
        summaries, observations = _screen_innovation_half_life(
            small, group, coefficients, base, roi_matrix, strata,
            input_traces, input_proxy, labels, intervals, quiet_count, config,
        )
        screen_rows.extend(summaries)
        observation_rows.extend(observations)
        _atomic_json(
            partial / "checkpoint.json",
            {"phase": "screen", "completed": len(screen_rows), "total": _grid_counts(config)["screen_total"]},
        )
    _progress(progress_path, "screen_fixed_point", combinations=len(config.fixed_specs))
    summaries, observations = _screen_fixed(
        small, [dict(spec) for spec in config.fixed_specs], base, roi_matrix,
        strata, input_traces, input_proxy, labels, intervals, quiet_count, config,
    )
    screen_rows.extend(summaries)
    observation_rows.extend(observations)
    del small, input_residual, input_traces, roi_matrix
    gc.collect()

    selection = _select_finalists(screen_rows, observation_rows, config)
    _atomic_json(partial / "selection.json", selection)
    screen_fields = sorted({key for row in screen_rows for key in row})
    observation_fields = list(observation_rows[0])
    _atomic_tsv(partial / "screen_metrics.tsv", screen_rows, screen_fields)
    _atomic_tsv(partial / "roi_observation_metrics.tsv", observation_rows, observation_fields)
    _progress(
        progress_path, "screen_complete",
        combinations=len(screen_rows),
        full_field_grid_finalists=len(selection["full_field_lane_ids"]),
    )

    specs = {
        str(spec["lane_id"]): dict(spec)
        for spec in (*config.innovation_specs, *config.fixed_specs)
    }
    dense_lane_ids = list(CONTROL_LANES) + [
        lane for lane in selection["full_field_lane_ids"] if lane not in CONTROL_LANES
    ]
    dense_results: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    match_keys: dict[str, set[tuple[int, str]]] = {}
    calibration_cache: dict[float, Any] = {}
    visual_ids = set(selection["global_visual_lane_ids"])
    for lane_index, lane_id in enumerate(dense_lane_ids, start=1):
        _progress(
            progress_path, "full_field_lane_start",
            lane_id=lane_id, lane_index=lane_index, lane_total=len(dense_lane_ids),
        )
        stack = _full_stack(
            lane_id, raw, quiet_count, coefficients, specs, config,
            calibration_cache,
        )
        if lane_id in visual_ids:
            _write_visual_pair(partial, lane_id, raw, stack, config)
        normalization = _normalize_positive_in_place(stack, quiet_count)
        result, candidates, keys, overlay = _evaluate_detection(
            lane_id, stack, labels, config
        )
        result["normalization"] = normalization
        dense_results.append(result)
        candidate_rows.extend(candidates)
        match_keys[lane_id] = keys
        if lane_id in visual_ids and config.visualization["write_detection_maps"]:
            _write_detection_maps(
                partial / "visuals" / lane_id / "detection_burst_maps.tif",
                lane_id, overlay,
            )
        del stack, overlay
        gc.collect()
        _atomic_json(
            partial / "checkpoint.json",
            {
                "phase": "full_field", "completed": lane_index,
                "total": len(dense_lane_ids), "last_lane": lane_id,
            },
        )

    result_by_lane = {row["lane_id"]: row for row in dense_results}
    raw_result = result_by_lane["raw_direct"]
    if abs(raw_result["mean_recall"] - RAW_DIRECT_EXPECTED) > 1e-12:
        raise RuntimeError(
            f"Raw Direct anchor failed: {raw_result['mean_recall']} != {RAW_DIRECT_EXPECTED}"
        )
    selected_folds = []
    selected_keys: set[tuple[int, str]] = set()
    for fold in selection["folds"]:
        burst = int(fold["heldout_burst"])
        lane_id = str(fold["primary_lane"])
        lane_fold = next(
            row for row in result_by_lane[lane_id]["outer_folds"]
            if int(row["burst_id"]) == burst
        )
        selected_folds.append({"lane_id": lane_id, **lane_fold})
        selected_keys.update(
            key for key in match_keys[lane_id] if key[0] == burst
        )
    crossvalidated = {
        "outer_folds": selected_folds,
        "mean_recall": float(np.mean([row["recall"] for row in selected_folds])),
        "pooled_recall": (
            sum(row["matched"] for row in selected_folds)
            / sum(row["labels"] for row in selected_folds)
        ),
        "total_matched": sum(row["matched"] for row in selected_folds),
        "total_labels": sum(row["labels"] for row in selected_folds),
        "selection_uses_heldout_labels": False,
    }
    bootstrap = _bootstrap_crossvalidated(
        selected_keys, match_keys["raw_direct"], labels, config
    )
    best = max(
        dense_results,
        key=lambda row: (
            row["mean_recall"], row["fixed_budget_mean_recall"],
            -row["total_event_candidates"],
        ),
    )
    improved = crossvalidated["mean_recall"] > raw_result["mean_recall"]
    metrics = {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "status": "completed", "grid": _grid_counts(config),
        "screen_rows": len(screen_rows),
        "full_field_lane_count": len(dense_results),
        "full_field_lanes": dense_results,
        "raw_direct": raw_result,
        "best_exploratory_full_field_lane": best,
        "crossvalidated_selection": crossvalidated,
        "paired_roi_bootstrap": bootstrap,
        "selection": selection,
        "decision": {
            "advance": improved and bootstrap["probability_difference_gt_zero"] >= 0.95,
            "interpretation": (
                "The cross-fitted architecture selection improved Raw Direct and the "
                "paired identity bootstrap supports a positive difference. Confirm the "
                "visual proxy masks and manually review unmatched candidates before "
                "claiming precision."
                if improved
                else
                "The broader architecture grid did not improve the cross-fitted Raw "
                "Direct recall anchor. Treat the strongest visual lane as a separation "
                "feature, not a replacement detector; next review proxy-mask validity "
                "and unmatched candidates rather than widening this grid again."
            ),
        },
        "scientific_contract": (
            "Sparse positives support known-label recall, FROC, candidate burden, and "
            "lower-bound known-label yield. They do not identify TN, FP, or precision."
        ),
        "elapsed_seconds": time.time() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(partial / "metrics.json", metrics)
    candidate_fields = list(candidate_rows[0])
    _atomic_tsv(partial / "candidates.tsv", candidate_rows, candidate_fields)
    _write_report(partial / "REPORT.md", metrics)
    _atomic_json(
        partial / "run_state.json",
        {
            "status": "completed", "completed_unix": time.time(),
            "elapsed_seconds": metrics["elapsed_seconds"],
            "screen_combinations": len(screen_rows),
            "full_field_lanes": len(dense_results),
            "visual_lanes": len(visual_ids),
            "max_rss_mib": metrics["max_rss_mib"],
        },
    )
    partial.replace(config.output_dir)
    return metrics
