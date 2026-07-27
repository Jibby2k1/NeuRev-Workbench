"""Checkpointed breadth/depth program for causal Spon Ca Burst proposals."""
from __future__ import annotations

import csv
import gc
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .activity_gated_video import _artifact_score
from .frame_difference import _atomic_json, _available_ram_mib, _sha256
from .learnable_contrast import core as v1


QUIET_STARTS = (0, 24, 48, 53)
DURATIONS = (24, 24, 28, 47)
POLICIES = (
    "quiet_threshold",
    "cap_40", "cap_50", "cap_58", "cap_75", "cap_100",
    "consensus_r4", "consensus_r6", "consensus_r8",
)
RAW_EXPECTED = 0.6056159420289855
CAUSAL_EXPECTED = 0.7341873706004141


class PlannedStop(RuntimeError):
    pass


@dataclass(frozen=True)
class CausalProposalProgramConfig:
    experiment_id: str
    source_video: Path
    source_tiff: Path
    labels_tsv: Path
    design_document: Path
    cfar_config: Path
    output_dir: Path
    preflight_dir: Path
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    spatial_sigmas: tuple[float, ...]
    temporal_ema_spans: tuple[float, ...]
    artifact_attenuations: tuple[float, ...]
    intensity_transforms: tuple[str, ...]
    baseline_modes: tuple[str, ...]
    pool_modes: tuple[str, ...]
    fractional_count: int
    fractional_seed: int
    fusion_finalists: int
    fusion_variants_per_finalist: int
    robustness_finalists: int
    calibration_conditions: int
    perturbation_types: tuple[str, ...]
    perturbation_severities: tuple[float, ...]
    horizon_frames: tuple[int, ...]
    nms_distance_px: int
    match_radius_px: int
    quiet_peaks_per_map: float
    primary_cap: int
    cpu_threads: int
    max_ram_mib: int
    max_gpu_memory_mib: int
    min_free_disk_mib: int
    max_output_mib: int
    wall_clock_hours: float
    heartbeat_seconds: int

    @classmethod
    def load(cls, path: str | Path) -> "CausalProposalProgramConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        frames, design, detector, resources = (
            raw["frames"], raw["design"], raw["detector"], raw["resources"]
        )
        p = lambda key: (root / raw[key]).resolve()
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=p("source_video"), source_tiff=p("source_tiff"),
            labels_tsv=p("labels_tsv"), design_document=p("design_document"),
            cfar_config=p("cfar_config"), output_dir=p("output_dir"),
            preflight_dir=p("preflight_dir"),
            review_start_ui=int(frames["review_start_ui"]),
            review_end_ui=int(frames["review_end_ui"]),
            quiet_start_ui=int(frames["quiet_start_ui"]),
            quiet_end_ui=int(frames["quiet_end_ui"]),
            spatial_sigmas=tuple(float(x) for x in design["spatial_sigmas"]),
            temporal_ema_spans=tuple(float(x) for x in design["temporal_ema_spans"]),
            artifact_attenuations=tuple(float(x) for x in design["artifact_attenuations"]),
            intensity_transforms=tuple(str(x) for x in design["intensity_transforms"]),
            baseline_modes=tuple(str(x) for x in design["baseline_modes"]),
            pool_modes=tuple(str(x) for x in design["pool_modes"]),
            fractional_count=int(design["fractional_count"]),
            fractional_seed=int(design["fractional_seed"]),
            fusion_finalists=int(design["fusion_finalists"]),
            fusion_variants_per_finalist=int(design["fusion_variants_per_finalist"]),
            robustness_finalists=int(design["robustness_finalists"]),
            calibration_conditions=int(design["calibration_conditions"]),
            perturbation_types=tuple(str(x) for x in design["perturbation_types"]),
            perturbation_severities=tuple(float(x) for x in design["perturbation_severities"]),
            horizon_frames=tuple(int(x) for x in design["horizon_frames"]),
            nms_distance_px=int(detector["nms_distance_px"]),
            match_radius_px=int(detector["match_radius_px"]),
            quiet_peaks_per_map=float(detector["quiet_peaks_per_map"]),
            primary_cap=int(detector["primary_cap"]),
            cpu_threads=int(resources["cpu_threads"]),
            max_ram_mib=int(resources["max_ram_mib"]),
            max_gpu_memory_mib=int(resources["max_gpu_memory_mib"]),
            min_free_disk_mib=int(resources["min_free_disk_mib"]),
            max_output_mib=int(resources["max_output_mib"]),
            wall_clock_hours=float(resources["wall_clock_hours"]),
            heartbeat_seconds=int(resources["heartbeat_seconds"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.output_dir.exists():
            raise FileExistsError(f"Output exists: {self.output_dir}")
        if self.review_start_ui != self.quiet_start_ui or self.quiet_end_ui - self.quiet_start_ui + 1 != 100:
            raise ValueError("The program requires the frozen 100-frame quiet interval at review start")
        expected = ((0.0, 0.5, 1.0, 1.5, 2.0), (1.0, 2.0, 4.0, 8.0, 16.0))
        if (self.spatial_sigmas, self.temporal_ema_spans) != expected:
            raise ValueError("Spatial and temporal breadth levels differ from the preregistration")
        if len(self.perturbation_types) != 9 or len(self.perturbation_severities) != 2:
            raise ValueError("Robustness requires nine perturbations at two severities")
        if self.horizon_frames != (12, 24, 28, 47, 64):
            raise ValueError("Early-detection horizons differ from the preregistration")
        if not 1 <= self.cpu_threads <= 16 or self.max_ram_mib > 49152:
            raise ValueError("Resource envelope exceeds the approved local planning bound")


def _method_id(prefix: str, factors: dict[str, Any]) -> str:
    import hashlib

    digest = hashlib.sha1(json.dumps(factors, sort_keys=True).encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _reference_factors() -> dict[str, Any]:
    return {
        "spatial_sigma": 1.0, "ema_span": 4.0, "artifact_attenuation": 0.7,
        "intensity_transform": "asinh5", "baseline_mode": "frozen_median",
        "pool_mode": "lme0.25",
    }


def build_method_specs(config: CausalProposalProgramConfig) -> list[dict[str, Any]]:
    reference = _reference_factors()
    anchors = [
        {"method_id": "raw_direct", "kind": "raw_direct", "factors": {
            **reference, "spatial_sigma": 0.0, "ema_span": 1.0,
            "artifact_attenuation": 0.0, "intensity_transform": "linear",
        }},
        {"method_id": "raw_direct_static_artifact", "kind": "causal", "factors": {
            **reference, "spatial_sigma": 0.0, "ema_span": 1.0,
            "intensity_transform": "linear",
        }},
        {"method_id": "fixed_cfar_center_r8", "kind": "cfar", "factors": {
            "expert_id": "center_r8_sector_censored_causal_coherence",
        }},
        {"method_id": "causal_reference", "kind": "causal", "factors": reference},
    ]
    levels = {
        "spatial_sigma": config.spatial_sigmas,
        "ema_span": config.temporal_ema_spans,
        "artifact_attenuation": config.artifact_attenuations,
        "intensity_transform": config.intensity_transforms,
        "baseline_mode": config.baseline_modes,
        "pool_mode": config.pool_modes,
    }
    ofat = []
    for key, values in levels.items():
        for value in values:
            if value == reference[key]:
                continue
            factors = reference | {key: value}
            ofat.append({"method_id": _method_id(f"ofat_{key}", factors), "kind": "causal", "factors": factors})
    if len(ofat) != 20:
        raise RuntimeError(f"Expected 20 one-factor methods, got {len(ofat)}")
    used = {tuple(item["factors"].get(key) for key in levels) for item in anchors if item["kind"] == "causal"}
    used.update(tuple(item["factors"][key] for key in levels) for item in ofat)
    grid = [combo for combo in itertools.product(*levels.values()) if combo not in used]
    rng = np.random.default_rng(config.fractional_seed)
    rng.shuffle(grid)
    chosen: list[tuple[Any, ...]] = []
    while len(chosen) < config.fractional_count:
        sample = grid if len(grid) <= 512 else [grid[i] for i in rng.choice(len(grid), 512, replace=False)]
        candidate = max(sample, key=lambda row: min((sum(a != b for a, b in zip(row, prior)) for prior in chosen), default=6))
        chosen.append(candidate)
        grid.remove(candidate)
    fractional = []
    keys = tuple(levels)
    for combo in chosen:
        factors = dict(zip(keys, combo))
        fractional.append({"method_id": _method_id("fractional", factors), "kind": "causal", "factors": factors})
    methods = anchors + ofat + fractional
    if len(methods) != 72 or len({row["method_id"] for row in methods}) != 72:
        raise RuntimeError(f"Breadth design must contain 72 unique methods, got {len(methods)}")
    return methods


def planned_counts(config: CausalProposalProgramConfig) -> dict[str, int]:
    methods = len(build_method_specs(config))
    breadth = methods * len(POLICIES)
    fusion_methods = config.fusion_finalists * config.fusion_variants_per_finalist
    fusion = fusion_methods * len(POLICIES)
    conditions = config.calibration_conditions + len(config.perturbation_types) * len(config.perturbation_severities) + len(config.horizon_frames)
    robustness = config.robustness_finalists * conditions
    return {
        "breadth_methods": methods, "policies": len(POLICIES),
        "breadth_evaluations": breadth, "maximum_fusion_methods": fusion_methods,
        "maximum_fusion_evaluations": fusion, "robustness_conditions": conditions,
        "maximum_robustness_evaluations": robustness,
        "maximum_logical_evaluations": breadth + fusion + robustness,
        "maximum_fold_condition_scores": (breadth + fusion + robustness) * 4,
    }


def _gpu_snapshot() -> dict[str, Any]:
    try:
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.free,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip().split(", ")
        return {"name": text[0], "free_mib": int(text[1]), "total_mib": int(text[2]),
                "temperature_c": int(text[3]), "power_w": float(text[4])}
    except Exception as exc:
        return {"error": repr(exc), "free_mib": 0}


def preflight(config: CausalProposalProgramConfig, write_artifacts: bool = True) -> dict[str, Any]:
    inputs = (config.source_video, config.source_tiff, config.labels_tsv,
              config.design_document, config.cfar_config)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = v1.load_labels(config.labels_tsv)
    if video.ndim != 3 or config.review_end_ui > len(video):
        raise ValueError(f"Invalid source or review interval: {video.shape}")
    if any(not (0 <= row["x_px"] < video.shape[2] and 0 <= row["y_px"] < video.shape[1]) for row in labels):
        raise ValueError("At least one normalized label lies outside the field")
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    disk_free = shutil.disk_usage(probe).free // 2**20
    gpu = _gpu_snapshot()
    counts = planned_counts(config)
    ready = bool(
        counts["maximum_logical_evaluations"] == 1884
        and _available_ram_mib() >= config.max_ram_mib
        and disk_free >= config.min_free_disk_mib
        and gpu.get("free_mib", 0) >= config.max_gpu_memory_mib
        and gpu.get("temperature_c", 100) < 80
    )
    payload = {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "ready": ready, "source_shape": list(video.shape), "label_rows": len(labels),
        "unique_rois": len({row["roi_identity"] for row in labels}),
        "counts": counts, "policies": list(POLICIES),
        "resources": {
            "cpu_threads": config.cpu_threads, "ram_available_mib": _available_ram_mib(),
            "ram_cap_mib": config.max_ram_mib, "gpu": gpu,
            "gpu_cap_mib": config.max_gpu_memory_mib, "disk_free_mib": disk_free,
            "minimum_free_disk_mib": config.min_free_disk_mib,
            "output_cap_mib": config.max_output_mib, "wall_clock_hours": config.wall_clock_hours,
        },
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in inputs],
        "scientific_contract": "Sparse positives support recall, ranking, FROC, and candidate burden; unmatched event peaks remain unknown, not false positives.",
        "launch_contract": "This program is Spon-only and must not start grid128 Stage A or Stage B.",
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        overlay = config.preflight_dir / "label_projection_overlay.png"
        v1._write_overlay(video, labels, overlay)
        payload["label_projection_overlay"] = str(overlay)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
        _atomic_json(config.preflight_dir / "resolved_config.json", _jsonable(asdict(config)))
    if not ready:
        raise RuntimeError(f"Overnight program preflight failed: {payload}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        stream.flush()


def _causal_filter(raw: np.ndarray, sigma: float, span: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    if sigma > 0:
        spatial = gaussian_filter(raw, sigma=(0, sigma, sigma), mode="reflect", truncate=4).astype(np.float32)
    else:
        spatial = np.asarray(raw, dtype=np.float32).copy()
    if span <= 1:
        return spatial
    alpha = 2 / (span + 1)
    filtered = np.empty_like(spatial)
    filtered[0] = spatial[0]
    for index in range(1, len(spatial)):
        filtered[index] = alpha * spatial[index] + (1 - alpha) * filtered[index - 1]
    return filtered


def _compress(filtered: np.ndarray, transform: str) -> np.ndarray:
    quiet = filtered[:100]
    low, high = np.percentile(quiet[:, ::4, ::4], [1, 99.8])
    unit = np.clip((filtered - low) / max(float(high - low), 1e-6), 0, 1)
    if transform == "linear":
        return unit.astype(np.float32)
    if not transform.startswith("asinh"):
        raise ValueError(transform)
    gain = float(transform.removeprefix("asinh"))
    return (np.arcsinh(gain * unit) / np.arcsinh(gain)).astype(np.float32)


def _residual_stack(transformed: np.ndarray, mode: str) -> np.ndarray:
    quiet = transformed[:100]
    baseline = np.median(quiet, axis=0).astype(np.float32)
    low, high = np.percentile(quiet[:, ::4, ::4], [1, 99.9])
    scale = max(float(high - low), 1e-6)
    residual = np.maximum((transformed - baseline[None]) / scale, 0).astype(np.float32)
    if mode == "frozen_median":
        return residual
    mad = np.median(np.abs(quiet - baseline), axis=0).astype(np.float32) * 1.4826
    floor = max(float(np.percentile(mad[mad > 0], 10)) if np.any(mad > 0) else 1e-3, 1e-3)
    local = np.maximum(mad, floor)
    state = baseline.copy()
    alpha, upper = (0.004, 3.0) if mode == "slow_clipped_ema" else (0.01, 1.0)
    for index in range(100, len(transformed)):
        innovation = transformed[index] - state
        residual[index] = np.maximum(innovation / scale, 0)
        state += alpha * np.clip(innovation, -3 * local, upper * local)
    return residual


def _pool(frames: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return frames.mean(0, dtype=np.float64).astype(np.float32)
    if mode == "max":
        return frames.max(0).astype(np.float32)
    from scipy.special import logsumexp

    tau = float(mode.removeprefix("lme"))
    return (tau * (logsumexp(frames / tau, axis=0) - math.log(len(frames)))).astype(np.float32)


def _maps_from_residual(
    residual: np.ndarray,
    labels: list[dict[str, Any]],
    config: CausalProposalProgramConfig,
    pool_mode: str,
    horizon: int | None = None,
) -> dict[str, Any]:
    quiet_maps = []
    for start, duration in zip(QUIET_STARTS, DURATIONS):
        count = min(duration, horizon) if horizon else duration
        quiet_maps.append(_pool(residual[start:start + count], pool_mode))
    review_start_zero = config.review_start_ui - 1
    events = {}
    for burst_id in range(1, 5):
        rows = [row for row in labels if row["burst_id"] == burst_id]
        start = rows[0]["start_frame_zero"] - review_start_zero
        stop = rows[0]["stop_frame_zero_exclusive"] - review_start_zero
        if horizon:
            stop = min(stop, start + horizon)
        events[burst_id] = _pool(residual[start:stop], pool_mode)
    return {"quiet": quiet_maps, "events": events}


def _score_causal(
    raw: np.ndarray,
    spec: dict[str, Any],
    labels: list[dict[str, Any]],
    config: CausalProposalProgramConfig,
    horizon: int | None = None,
) -> dict[str, Any]:
    factors = spec["factors"]
    if spec["kind"] == "raw_direct":
        filtered = np.asarray(raw, dtype=np.float32)
        transformed = filtered
    else:
        filtered = _causal_filter(raw, float(factors["spatial_sigma"]), float(factors["ema_span"]))
        artifact, _ = _artifact_score(filtered[:100])
        compressed = _compress(filtered, str(factors["intensity_transform"]))
        transformed = compressed * (1 - float(factors["artifact_attenuation"]) * artifact[None])
    residual = _residual_stack(transformed, str(factors["baseline_mode"]))
    maps = _maps_from_residual(residual, labels, config, str(factors["pool_mode"]), horizon=horizon)
    del filtered, transformed, residual
    gc.collect()
    return maps


def _score_cfar(
    raw: np.ndarray,
    labels: list[dict[str, Any]],
    config: CausalProposalProgramConfig,
) -> dict[str, Any]:
    import torch
    from .learnable_contrast import multihypothesis as multi

    cfar = multi.MultiCFARConfig.load(config.cfar_config)
    specs = [spec for spec in multi.expert_matrix(cfar.radii_px)
             if spec.expert_id == "center_r8_sector_censored_causal_coherence"]
    if len(specs) != 1:
        raise RuntimeError("Exact CFAR anchor was not resolved uniquely")
    bank = multi.build_kernel_bank(specs, cfar.support_px)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, config.max_gpu_memory_mib * 2**20 / total))
    raw_spec = {"kind": "raw_direct", "factors": {
        **_reference_factors(), "spatial_sigma": 0.0, "ema_span": 1.0,
        "artifact_attenuation": 0.0, "intensity_transform": "linear",
    }}
    filtered = np.asarray(raw, dtype=np.float32)
    residual = _residual_stack(filtered, "frozen_median")
    quiet = [residual[start:start + duration] for start, duration in zip(QUIET_STARTS, DURATIONS)]
    review_start_zero = config.review_start_ui - 1
    events = {}
    for burst_id in range(1, 5):
        rows = [row for row in labels if row["burst_id"] == burst_id]
        start = rows[0]["start_frame_zero"] - review_start_zero
        stop = rows[0]["stop_frame_zero_exclusive"] - review_start_zero
        events[burst_id] = residual[start:stop]
    maps = {
        "quiet": [multi._score_maps(frames, specs, bank, cfar)[0] for frames in quiet],
        "events": {burst: multi._score_maps(frames, specs, bank, cfar)[0] for burst, frames in events.items()},
    }
    del raw_spec, residual
    torch.cuda.empty_cache()
    return maps


def _cache_maps(path: Path, maps: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = {f"quiet_{i}": value for i, value in enumerate(maps["quiet"])}
    payload.update({f"event_{burst}": value for burst, value in maps["events"].items()})
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def _load_maps(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "quiet": [data[f"quiet_{i}"].astype(np.float32) for i in range(4)],
            "events": {burst: data[f"event_{burst}"].astype(np.float32) for burst in range(1, 5)},
        }


def _threshold(quiet_maps: list[np.ndarray], distance: int, rate: float = 1.0) -> float:
    values = []
    for score in quiet_maps:
        values.extend(value for value, _, _ in v1._peaks(score, distance, limit=3000))
    ranked = sorted(values, reverse=True)
    allowed = max(1, int(round(rate * len(quiet_maps))))
    if len(ranked) <= allowed:
        raise RuntimeError("Too few quiet peaks")
    return float(np.nextafter(ranked[allowed], np.inf))


def _primary_peaks(maps: dict[str, Any], config: CausalProposalProgramConfig) -> tuple[float, dict[int, list[tuple]]]:
    threshold = _threshold(maps["quiet"], config.nms_distance_px, config.quiet_peaks_per_map)
    peaks = {
        burst: [p for p in v1._peaks(score, config.nms_distance_px, limit=3000) if p[0] >= threshold]
        for burst, score in maps["events"].items()
    }
    return threshold, peaks


def _policy_metrics(
    peaks: dict[int, list[tuple]],
    labels: list[dict[str, Any]],
    policy: str,
    raw_peaks: dict[int, list[tuple]] | None,
    match_radius: int,
) -> dict[str, Any]:
    folds = []
    for burst in range(1, 5):
        selected = list(peaks[burst])
        if policy.startswith("cap_"):
            selected = selected[:int(policy.split("_")[1])]
        elif policy.startswith("consensus_r"):
            radius = int(policy.removeprefix("consensus_r"))
            reference = raw_peaks[burst] if raw_peaks is not None else selected
            selected = [p for p in selected if any(math.hypot(p[1] - q[1], p[2] - q[2]) <= radius for q in reference)]
        rows = [row for row in labels if row["burst_id"] == burst]
        matches = v1._match(selected, rows, match_radius)
        folds.append({"burst_id": burst, "matched": len(matches), "labels": len(rows),
                      "recall": len(matches) / len(rows), "candidates": len(selected)})
    return {
        "policy": policy, "folds": folds,
        "mean_recall": float(np.mean([row["recall"] for row in folds])),
        "pooled_recall": sum(row["matched"] for row in folds) / sum(row["labels"] for row in folds),
        "matched": sum(row["matched"] for row in folds),
        "labels": sum(row["labels"] for row in folds),
        "candidates": sum(row["candidates"] for row in folds),
        "known_label_fraction_lower_bound": sum(row["matched"] for row in folds) / max(1, sum(row["candidates"] for row in folds)),
    }


def _quiet_crossfit(maps: dict[str, Any], config: CausalProposalProgramConfig) -> float:
    counts = []
    for heldout in range(4):
        training = [score for index, score in enumerate(maps["quiet"]) if index != heldout]
        threshold = _threshold(training, config.nms_distance_px, config.quiet_peaks_per_map)
        peaks = [p for p in v1._peaks(maps["quiet"][heldout], config.nms_distance_px, limit=3000) if p[0] >= threshold]
        counts.append(len(peaks))
    return float(np.mean(counts))


def _summarize_method(
    spec: dict[str, Any], maps: dict[str, Any], labels: list[dict[str, Any]],
    config: CausalProposalProgramConfig, raw_peaks: dict[int, list[tuple]] | None,
) -> tuple[dict[str, Any], dict[int, list[tuple]]]:
    threshold, peaks = _primary_peaks(maps, config)
    policies = {policy: _policy_metrics(peaks, labels, policy, raw_peaks, config.match_radius_px) for policy in POLICIES}
    summary = {
        "method_id": spec["method_id"], "kind": spec["kind"], "factors": spec["factors"],
        "threshold": threshold, "quiet_crossfit_peaks_per_map": _quiet_crossfit(maps, config),
        "policies": policies,
    }
    return summary, peaks


def _nested_selection(summaries: list[dict[str, Any]], policy: str, raw: dict[str, Any]) -> dict[str, Any]:
    folds = []
    for heldout in range(4):
        eligible = [row for row in summaries if row["kind"] != "cfar"]
        selected = max(eligible, key=lambda row: (
            np.mean([fold["recall"] for index, fold in enumerate(row["policies"][policy]["folds"]) if index != heldout]),
            -row["quiet_crossfit_peaks_per_map"], -len(json.dumps(row["factors"])),
        ))
        test = selected["policies"][policy]["folds"][heldout]
        baseline = raw["policies"][policy]["folds"][heldout]
        folds.append({"heldout_burst": heldout + 1, "selected_method": selected["method_id"],
                      "recall": test["recall"], "raw_recall": baseline["recall"],
                      "win": test["recall"] > baseline["recall"]})
    return {"folds": folds, "mean_recall": float(np.mean([x["recall"] for x in folds])),
            "raw_mean_recall": float(np.mean([x["raw_recall"] for x in folds])),
            "wins": sum(x["win"] for x in folds)}


def _margin_maps(maps: dict[str, Any], config: CausalProposalProgramConfig) -> dict[str, Any]:
    threshold = _threshold(maps["quiet"], config.nms_distance_px, 1.0)
    values = []
    for score in maps["quiet"]:
        values.extend(value for value, _, _ in v1._peaks(score, config.nms_distance_px, limit=3000))
    median = float(np.median(values)); mad = float(np.median(np.abs(np.asarray(values) - median)) * 1.4826)
    scale = max(mad, float(np.std(values)), 1e-4)
    return {"quiet": [(score - threshold) / scale for score in maps["quiet"]],
            "events": {burst: (score - threshold) / scale for burst, score in maps["events"].items()}}


def _combine_maps(parts: list[tuple[float, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "quiet": [sum(weight * maps["quiet"][i] for weight, maps in parts).astype(np.float32) for i in range(4)],
        "events": {burst: sum(weight * maps["events"][burst] for weight, maps in parts).astype(np.float32) for burst in range(1, 5)},
    }


def _derivative_aux(raw: np.ndarray, labels: list[dict[str, Any]], config: CausalProposalProgramConfig) -> dict[str, Any]:
    filtered = _causal_filter(raw, 1.0, 4.0)
    derivative = np.empty_like(filtered); derivative[0] = 0; derivative[1:] = filtered[1:] - filtered[:-1]
    quiet = derivative[:100]
    center = np.median(quiet, axis=0); mad = np.median(np.abs(quiet - center), axis=0) * 1.4826
    floor = max(float(np.percentile(mad[mad > 0], 10)) if np.any(mad > 0) else 1.0, 1e-6)
    z = (derivative - center[None]) / np.maximum(mad, floor)[None]
    energy = np.empty_like(z); energy[0] = z[0] ** 2; alpha = 0.4
    for index in range(1, len(z)):
        energy[index] = alpha * z[index] ** 2 + (1 - alpha) * energy[index - 1]
    maps = _maps_from_residual(np.log1p(energy).astype(np.float32), labels, config, "mean")
    del filtered, derivative, z, energy
    gc.collect()
    return maps


def _perturb(raw: np.ndarray, kind: str, severity: float, seed: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, maximum_filter

    value = np.asarray(raw, dtype=np.float32).copy()
    if kind == "gain":
        value *= 1 + severity
    elif kind == "offset":
        value += severity * 400
    elif kind == "noise":
        value += np.random.default_rng(seed).normal(0, severity * 30, value.shape).astype(np.float32)
    elif kind == "translation":
        shift = 1 if severity < 0.15 else 2
        value = np.roll(value, shift=(0, shift, shift), axis=(0, 1, 2)); value[:, :shift] = 0; value[:, :, :shift] = 0
    elif kind == "stripe":
        value[:, :, value.shape[2] // 3:value.shape[2] // 3 + (1 if severity < 0.15 else 3)] += severity * 1000
    elif kind == "saturation_bloom":
        bright = value > np.percentile(value[:100], 99.8)
        bloom = maximum_filter(bright.astype(np.uint8), size=(1, 3 if severity < 0.15 else 5, 3 if severity < 0.15 else 5))
        value = np.where(bloom, np.maximum(value, 4095 * severity), value)
    elif kind == "frame_drop":
        every = 20 if severity < 0.15 else 10
        for index in range(every, len(value), every): value[index] = value[index - 1]
    elif kind == "quiet_contamination":
        yy, xx = np.ogrid[:value.shape[1], :value.shape[2]]; mask = (xx - 420) ** 2 + (yy - 215) ** 2 <= (6 if severity < 0.15 else 10) ** 2
        value[20:45, mask] += severity * 800
    elif kind == "photobleach":
        factor = np.linspace(1, 1 - severity, len(value), dtype=np.float32)
        value *= factor[:, None, None]
    else:
        raise ValueError(kind)
    return np.clip(value, 0, 4095).astype(np.float32)


def _resource_guard(config: CausalProposalProgramConfig, started: float) -> None:
    if time.monotonic() - started > config.wall_clock_hours * 3600:
        raise PlannedStop("wall_clock_limit")
    rss = v1.rss_mib()
    if rss > config.max_ram_mib:
        raise PlannedStop(f"rss_limit:{rss:.1f}")
    if _available_ram_mib() < 8192:
        raise PlannedStop("system_available_ram_below_8192_mib")
    if shutil.disk_usage(config.output_dir).free // 2**20 < config.min_free_disk_mib:
        raise PlannedStop("disk_headroom")
    gpu = _gpu_snapshot()
    if gpu.get("temperature_c", 0) >= 84:
        raise PlannedStop(f"gpu_temperature:{gpu['temperature_c']}")


def _review_batch(
    output: Path, best_maps: dict[str, Any], raw_maps: dict[str, Any], cfar_maps: dict[str, Any],
    labels: list[dict[str, Any]], config: CausalProposalProgramConfig,
) -> dict[str, Any]:
    _, best = _primary_peaks(best_maps, config); _, raw = _primary_peaks(raw_maps, config); _, cfar = _primary_peaks(cfar_maps, config)
    rng = np.random.default_rng(20260727); rows = []
    def add(category: str, candidates: list[tuple[int, tuple]], limit: int = 40) -> None:
        selected = []
        for burst, peak in sorted(candidates, key=lambda item: item[1][0], reverse=True):
            if any(burst == b and math.hypot(peak[1] - p[1], peak[2] - p[2]) <= 6 for b, p in selected): continue
            selected.append((burst, peak))
            if len(selected) == limit: break
        for burst, (score, x, y) in selected:
            label_rows = [row for row in labels if row["burst_id"] == burst]
            if burst == 0:
                start_frame_ui = config.quiet_start_ui
                end_frame_ui = config.quiet_end_ui
            else:
                start_frame_ui = label_rows[0]["start_frame_ui"]
                end_frame_ui = label_rows[0]["end_frame_ui"]
            rows.append({"category": category, "burst_id": burst, "x_px": x, "y_px": y,
                         "score": score, "start_frame_ui": start_frame_ui,
                         "end_frame_ui": end_frame_ui, "state": "unlabeled",
                         "coverage_mode": "candidate_review"})
    best_all = [(b, p) for b, ps in best.items() for p in ps]
    raw_all = [(b, p) for b, ps in raw.items() for p in ps]
    cfar_all = [(b, p) for b, ps in cfar.items() for p in ps]
    add("causal_only", [(b, p) for b, p in best_all if not any(b == rb and math.hypot(p[1]-q[1], p[2]-q[2]) <= 8 for rb, q in raw_all)])
    add("raw_only", [(b, p) for b, p in raw_all if not any(b == cb and math.hypot(p[1]-q[1], p[2]-q[2]) <= 8 for cb, q in best_all)])
    add("raw_causal_consensus", [(b, p) for b, p in best_all if any(b == rb and math.hypot(p[1]-q[1], p[2]-q[2]) <= 8 for rb, q in raw_all)])
    add("cfar_only", [(b, p) for b, p in cfar_all if not any(b == cb and math.hypot(p[1]-q[1], p[2]-q[2]) <= 8 for cb, q in best_all)])
    quiet_candidates = []
    for window, score in enumerate(best_maps["quiet"], 1):
        quiet_candidates.extend((0, p) for p in v1._peaks(score, config.nms_distance_px, limit=100))
    add("quiet_hard_negative", quiet_candidates)
    for _ in range(40):
        burst = int(rng.integers(1, 5)); label_rows = [row for row in labels if row["burst_id"] == burst]
        rows.append({"category": "detector_independent_random", "burst_id": burst,
                     "x_px": int(rng.integers(8, 565)), "y_px": int(rng.integers(8, 332)), "score": "",
                     "start_frame_ui": label_rows[0]["start_frame_ui"], "end_frame_ui": label_rows[0]["end_frame_ui"],
                     "state": "unlabeled", "coverage_mode": "candidate_review"})
    path = output / "review_batch.tsv"
    fields = ["category", "burst_id", "x_px", "y_px", "score", "start_frame_ui", "end_frame_ui", "state", "coverage_mode"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    _atomic_json(output / "workbench_review_queue.json", {
        "schema_version": 1, "coverage_mode": "candidate_review",
        "precision_identified": False, "records": rows,
    })
    return {"rows": len(rows), "by_category": {key: sum(row["category"] == key for row in rows) for key in sorted({r["category"] for r in rows})}}


def _write_morning_report(output: Path, payload: dict[str, Any]) -> None:
    gate = payload["gates"]
    lines = [f"# {payload['experiment_id']}", "", f"Status: `{payload['status']}`.", "",
             "## Gates", "", f"- C0 baseline reproduction: `{gate['C0']['status']}`",
             f"- C1 breadth: `{gate['C1']['status']}`",
             f"- C2 bounded fusion: `{gate['C2']['status']}`",
             f"- C3 robustness: `{gate['C3']['status']}`", "",
             "## Morning review", "", f"Best method: `{payload.get('best_method_id')}`.",
             f"Review queue rows: `{payload.get('review_batch',{}).get('rows',0)}`.", "",
             "Sparse labels do not identify ordinary precision. Unmatched event candidates remain unknown.", "",
             "Use `summary.json`, `stage_a_summary.json`, `stage_b_summary.json`, `stage_c_summary.json`, and `resource_observations.json` for evidence.", ""]
    (output / "morning_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(config: CausalProposalProgramConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.cpu_threads)
    audit = preflight(config, write_artifacts=True)
    started = time.monotonic(); started_at = v1.utc_now()
    config.output_dir.mkdir(parents=True, exist_ok=False)
    cache = config.output_dir / "map_cache"; cache.mkdir()
    _atomic_json(config.output_dir / "resolved_config.json", _jsonable(asdict(config)))
    _atomic_json(config.output_dir / "preflight.json", audit)
    _atomic_json(config.output_dir / "program_run.json", {
        "schema_version": 1, "experiment_id": config.experiment_id, "status": "running",
        "pid": os.getpid(), "started_at": started_at,
    })
    progress = config.output_dir / "progress.jsonl"
    resources = config.output_dir / "resource_observations.jsonl"
    def heartbeat(stage: str, **data: Any) -> None:
        _append_jsonl(progress, {"at": v1.utc_now(), "stage": stage, **data})
        _append_jsonl(resources, {"at": v1.utc_now(), "stage": stage, "rss_mib": v1.rss_mib(),
                                  "available_ram_mib": _available_ram_mib(), "gpu": _gpu_snapshot()})
    labels = v1.load_labels(config.labels_tsv)
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    v1._write_overlay(video, labels, config.output_dir / "label_projection_overlay.png")
    raw = np.asarray(video[config.review_start_ui - 1:config.review_end_ui], dtype=np.float32)
    methods = build_method_specs(config)
    stage_a_rows, maps_by_id, peaks_by_id = [], {}, {}
    try:
        heartbeat("C0", status="started", validation_methods=2)
        raw_peaks = None
        ordered_methods = [
            next(item for item in methods if item["method_id"] == "raw_direct"),
            next(item for item in methods if item["method_id"] == "causal_reference"),
            *(item for item in methods if item["method_id"] not in {"raw_direct", "causal_reference"}),
        ]
        for index, spec in enumerate(ordered_methods, 1):
            _resource_guard(config, started)
            map_path = cache / f"{spec['method_id']}.npz"
            maps = _score_cfar(raw, labels, config) if spec["kind"] == "cfar" else _score_causal(raw, spec, labels, config)
            _cache_maps(map_path, maps); maps_by_id[spec["method_id"]] = maps
            summary, peaks = _summarize_method(spec, maps, labels, config, raw_peaks)
            if spec["method_id"] == "raw_direct": raw_peaks = peaks
            elif raw_peaks is not None:
                summary, peaks = _summarize_method(spec, maps, labels, config, raw_peaks)
            stage_a_rows.append(summary); peaks_by_id[spec["method_id"]] = peaks
            _append_jsonl(config.output_dir / "stage_a_results.jsonl", summary)
            if index == 2:
                raw_row = next(row for row in stage_a_rows if row["method_id"] == "raw_direct")
                causal_row = next(row for row in stage_a_rows if row["method_id"] == "causal_reference")
                raw_mean = raw_row["policies"]["quiet_threshold"]["mean_recall"]
                causal_mean = causal_row["policies"]["quiet_threshold"]["mean_recall"]
                c0_pass = abs(raw_mean - RAW_EXPECTED) < 1e-12 and abs(causal_mean - CAUSAL_EXPECTED) < 1e-12
                heartbeat("C0", status="pass" if c0_pass else "failed",
                          raw_mean_recall=raw_mean, causal_mean_recall=causal_mean)
                if not c0_pass:
                    raise RuntimeError(f"Baseline reproduction failed: raw={raw_mean}, causal={causal_mean}")
            heartbeat("C1_breadth", status="method_complete", completed=index, total=len(methods), method_id=spec["method_id"])
        nested = _nested_selection(stage_a_rows, "cap_58", raw_row)
        c1_pass = nested["mean_recall"] >= nested["raw_mean_recall"] + 0.03 and nested["wins"] >= 3
        stage_a_summary = {"status": "complete", "method_count": len(stage_a_rows), "nested_cap58": nested,
                           "gate_pass": c1_pass, "top_methods": []}
        ranked_a = sorted([row for row in stage_a_rows if row["kind"] != "cfar"], key=lambda row: (
            row["policies"]["cap_58"]["pooled_recall"], -row["quiet_crossfit_peaks_per_map"],
        ), reverse=True)
        stage_a_summary["top_methods"] = [row["method_id"] for row in ranked_a[:config.fusion_finalists]]
        _atomic_json(config.output_dir / "stage_a_summary.json", stage_a_summary)
        heartbeat("C1_breadth", status="complete", gate_pass=c1_pass, nested=nested)

        stage_b_rows: list[dict[str, Any]] = []
        stage_b_maps: dict[str, dict[str, Any]] = {}
        c2_pass = False
        if c1_pass:
            raw_margin = _margin_maps(maps_by_id["raw_direct"], config)
            cfar_margin = _margin_maps(maps_by_id["fixed_cfar_center_r8"], config)
            derivative_margin = _margin_maps(_derivative_aux(raw, labels, config), config)
            fusion_defs = [
                ("raw025", ((0.75, "base"), (0.25, "raw"))),
                ("raw050", ((0.50, "base"), (0.50, "raw"))),
                ("raw075", ((0.25, "base"), (0.75, "raw"))),
                ("deriv010", ((1.0, "base"), (0.10, "deriv"))),
                ("deriv025", ((1.0, "base"), (0.25, "deriv"))),
                ("deriv050", ((1.0, "base"), (0.50, "deriv"))),
                ("cfar010", ((1.0, "base"), (0.10, "cfar"))),
                ("cfar025", ((1.0, "base"), (0.25, "cfar"))),
                ("cfar050", ((1.0, "base"), (0.50, "cfar"))),
                ("combo_light", ((1.0, "base"), (0.25, "raw"), (0.10, "deriv"), (0.10, "cfar"))),
                ("combo_medium", ((1.0, "base"), (0.50, "raw"), (0.25, "deriv"), (0.25, "cfar"))),
                ("combo_selective", ((1.0, "base"), (0.25, "raw"), (-0.10, "deriv"), (0.25, "cfar"))),
            ]
            for finalist in ranked_a[:config.fusion_finalists]:
                base_margin = _margin_maps(maps_by_id[finalist["method_id"]], config)
                sources = {"base": base_margin, "raw": raw_margin, "deriv": derivative_margin, "cfar": cfar_margin}
                for token, definition in fusion_defs:
                    method_id = f"fusion__{finalist['method_id']}__{token}"
                    maps = _combine_maps([(weight, sources[name]) for weight, name in definition])
                    spec = {"method_id": method_id, "kind": "fusion", "factors": {"parent": finalist["method_id"], "definition": definition}}
                    summary, peaks = _summarize_method(spec, maps, labels, config, raw_peaks)
                    stage_b_rows.append(summary); stage_b_maps[method_id] = maps; peaks_by_id[method_id] = peaks
                    _append_jsonl(config.output_dir / "stage_b_results.jsonl", summary)
                    heartbeat("C2_fusion", status="method_complete", completed=len(stage_b_rows), total=config.fusion_finalists * 12, method_id=method_id)
            ranked_b = sorted(stage_b_rows, key=lambda row: row["policies"]["cap_58"]["pooled_recall"], reverse=True)
            best_b = ranked_b[0]
            best_a = ranked_a[0]
            gain = best_b["policies"]["cap_58"]["mean_recall"] - best_a["policies"]["cap_58"]["mean_recall"]
            reduction = 1 - best_b["policies"]["quiet_threshold"]["candidates"] / max(1, best_a["policies"]["quiet_threshold"]["candidates"])
            same_or_better = best_b["policies"]["cap_58"]["mean_recall"] >= best_a["policies"]["cap_58"]["mean_recall"]
            c2_pass = gain >= 0.02 or (reduction >= 0.20 and same_or_better)
            stage_b_summary = {"status": "complete", "method_count": len(stage_b_rows), "gate_pass": c2_pass,
                               "best_method": best_b["method_id"], "gain_vs_best_breadth": gain,
                               "candidate_reduction_vs_best_breadth": reduction}
        else:
            ranked_b = []
            stage_b_summary = {"status": "not_run_C1_gate", "method_count": 0, "gate_pass": False}
        _atomic_json(config.output_dir / "stage_b_summary.json", stage_b_summary)
        heartbeat("C2_fusion", status=stage_b_summary["status"], gate_pass=c2_pass)

        base_finalists = ranked_a[:config.robustness_finalists]
        stage_c_rows = []
        for finalist_index, finalist in enumerate(base_finalists, 1):
            spec = next(item for item in methods if item["method_id"] == finalist["method_id"])
            base_maps = maps_by_id[finalist["method_id"]]
            for calibration_seed in range(config.calibration_conditions):
                rng = np.random.default_rng(9100 + calibration_seed)
                indices = rng.integers(0, 4, size=4)
                calibrated = {"quiet": [base_maps["quiet"][int(i)] for i in indices], "events": base_maps["events"]}
                _, peaks = _primary_peaks(calibrated, config)
                metric = _policy_metrics(peaks, labels, "cap_58", raw_peaks, config.match_radius_px)
                row = {"method_id": finalist["method_id"], "condition_family": "quiet_calibration_bootstrap",
                       "condition": calibration_seed, "metrics": metric}
                stage_c_rows.append(row); _append_jsonl(config.output_dir / "stage_c_results.jsonl", row)
                heartbeat("C3_robustness", completed=len(stage_c_rows), total=config.robustness_finalists * 31)
            for kind_index, kind in enumerate(config.perturbation_types):
                for severity in config.perturbation_severities:
                    _resource_guard(config, started)
                    perturbed = _perturb(raw, kind, severity, 20000 + finalist_index * 100 + kind_index)
                    maps = _score_causal(perturbed, spec, labels, config)
                    _, peaks = _primary_peaks(maps, config)
                    metric = _policy_metrics(peaks, labels, "cap_58", raw_peaks, config.match_radius_px)
                    row = {"method_id": finalist["method_id"], "condition_family": "raw_perturbation",
                           "condition": kind, "severity": severity, "metrics": metric}
                    stage_c_rows.append(row); _append_jsonl(config.output_dir / "stage_c_results.jsonl", row)
                    del perturbed, maps; gc.collect()
                    heartbeat("C3_robustness", completed=len(stage_c_rows), total=config.robustness_finalists * 31)
            for horizon in config.horizon_frames:
                maps = _score_causal(raw, spec, labels, config, horizon=horizon)
                _, peaks = _primary_peaks(maps, config)
                metric = _policy_metrics(peaks, labels, "cap_58", raw_peaks, config.match_radius_px)
                row = {"method_id": finalist["method_id"], "condition_family": "early_detection_horizon",
                       "condition": horizon, "metrics": metric}
                stage_c_rows.append(row); _append_jsonl(config.output_dir / "stage_c_results.jsonl", row)
                del maps; gc.collect()
                heartbeat("C3_robustness", completed=len(stage_c_rows), total=config.robustness_finalists * 31)
        raw_cap = raw_row["policies"]["cap_58"]["mean_recall"]
        robustness_by_method = {}
        for finalist in base_finalists:
            values = [row["metrics"]["mean_recall"] for row in stage_c_rows if row["method_id"] == finalist["method_id"]]
            robustness_by_method[finalist["method_id"]] = {"median": float(np.median(values)),
                                                            "q25": float(np.percentile(values, 25)), "minimum": float(min(values))}
        best_robust = max(robustness_by_method, key=lambda key: (robustness_by_method[key]["median"], robustness_by_method[key]["q25"]))
        c3_pass = robustness_by_method[best_robust]["median"] > raw_cap and robustness_by_method[best_robust]["q25"] >= raw_cap - 0.05
        stage_c_summary = {"status": "complete", "evaluation_count": len(stage_c_rows), "gate_pass": c3_pass,
                           "best_robust_method": best_robust, "by_method": robustness_by_method}
        _atomic_json(config.output_dir / "stage_c_summary.json", stage_c_summary)
        heartbeat("C3_robustness", status="complete", gate_pass=c3_pass)

        candidates = ranked_a + ranked_b
        best = max(candidates, key=lambda row: (row["policies"]["cap_58"]["pooled_recall"],
                                                 -row["quiet_crossfit_peaks_per_map"]))
        best_maps = stage_b_maps.get(best["method_id"], maps_by_id.get(best["method_id"]))
        review = _review_batch(config.output_dir, best_maps, maps_by_id["raw_direct"], maps_by_id["fixed_cfar_center_r8"], labels, config)
        payload = {
            "schema_version": 1, "experiment_id": config.experiment_id, "status": "complete",
            "completed_at": v1.utc_now(), "elapsed_seconds": time.monotonic() - started,
            "counts": audit["counts"], "best_method_id": best["method_id"],
            "best_cap58": best["policies"]["cap_58"], "review_batch": review,
            "gates": {
                "C0": {"status": "pass", "raw_mean_recall": raw_mean, "causal_mean_recall": causal_mean},
                "C1": {"status": "pass" if c1_pass else "stop", "nested": nested},
                "C2": {"status": "pass" if c2_pass else stage_b_summary["status"]},
                "C3": {"status": "pass" if c3_pass else "do_not_advance"},
            },
            "precision_contract": "Unmatched event candidates remain unknown; review_batch is candidate_review, not exhaustive truth.",
        }
        _atomic_json(config.output_dir / "summary.json", payload)
        _write_morning_report(config.output_dir, payload)
        _atomic_json(config.output_dir / "program_run.json", {
            "schema_version": 1, "experiment_id": config.experiment_id, "status": "completed",
            "pid": os.getpid(), "started_at": started_at, "completed_at": v1.utc_now(),
            "elapsed_seconds": payload["elapsed_seconds"],
        })
        return payload
    except PlannedStop as exc:
        _atomic_json(config.output_dir / "program_run.json", {
            "schema_version": 1, "experiment_id": config.experiment_id, "status": "stopped",
            "pid": os.getpid(), "started_at": started_at, "stopped_at": v1.utc_now(), "reason": str(exc),
        })
        raise
    except Exception as exc:
        _atomic_json(config.output_dir / "program_run.json", {
            "schema_version": 1, "experiment_id": config.experiment_id, "status": "failed",
            "pid": os.getpid(), "started_at": started_at, "failed_at": v1.utc_now(), "error": repr(exc),
        })
        raise
