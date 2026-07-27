"""Paired detection benchmark for offline and causal activity-gated inputs."""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .activity_gated_video import _artifact_score, _smooth_review
from .frame_difference import _atomic_json, _available_ram_mib, _sha256
from .learnable_contrast import core as v1


LANES = (
    "raw_direct",
    "raw_direct_static_artifact",
    "offline_artifact_gate",
    "causal_artifact_only",
    "causal_artifact_gate_floor_0p2",
    "causal_artifact_gate_floor_0p4",
)


@dataclass(frozen=True)
class ActivityGateBenchmarkConfig:
    experiment_id: str
    source_video: Path
    source_tiff: Path
    labels_tsv: Path
    design_document: Path
    output_dir: Path
    preflight_dir: Path
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    spatial_sigma_px: float
    offline_temporal_window_frames: int
    offline_temporal_polyorder: int
    causal_temporal_ema_span_frames: float
    derivative_lag_frames: int
    energy_ema_span_frames: float
    gate_tau_z: float
    causal_structural_floors: tuple[float, ...]
    artifact_attenuation: float
    intensity_asinh_gain: float
    quiet_mad_floor_percentile: float
    temporal_pool_tau: float
    nms_distance_px: int
    primary_match_radius_px: int
    match_radii_px: tuple[int, ...]
    quiet_false_peaks_per_map: float
    cpu_threads: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int

    @classmethod
    def load(cls, path: str | Path) -> "ActivityGateBenchmarkConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        frames, filtering, detector, resources = (
            raw["frames"], raw["filtering"], raw["detector"], raw["resources"]
        )
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            source_tiff=(root / raw["source_tiff"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            design_document=(root / raw["design_document"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            review_start_ui=int(frames["review_start_ui"]),
            review_end_ui=int(frames["review_end_ui"]),
            quiet_start_ui=int(frames["quiet_start_ui"]),
            quiet_end_ui=int(frames["quiet_end_ui"]),
            spatial_sigma_px=float(filtering["spatial_sigma_px"]),
            offline_temporal_window_frames=int(filtering["offline_temporal_window_frames"]),
            offline_temporal_polyorder=int(filtering["offline_temporal_polyorder"]),
            causal_temporal_ema_span_frames=float(filtering["causal_temporal_ema_span_frames"]),
            derivative_lag_frames=int(filtering["derivative_lag_frames"]),
            energy_ema_span_frames=float(filtering["energy_ema_span_frames"]),
            gate_tau_z=float(filtering["gate_tau_z"]),
            causal_structural_floors=tuple(float(x) for x in filtering["causal_structural_floors"]),
            artifact_attenuation=float(filtering["artifact_attenuation"]),
            intensity_asinh_gain=float(filtering["intensity_asinh_gain"]),
            quiet_mad_floor_percentile=float(filtering["quiet_mad_floor_percentile"]),
            temporal_pool_tau=float(detector["temporal_pool_tau"]),
            nms_distance_px=int(detector["nms_distance_px"]),
            primary_match_radius_px=int(detector["primary_match_radius_px"]),
            match_radii_px=tuple(int(x) for x in detector["match_radii_px"]),
            quiet_false_peaks_per_map=float(detector["quiet_false_peaks_per_map"]),
            cpu_threads=int(resources["cpu_threads"]),
            max_ram_mib=int(resources["max_ram_mib"]),
            min_free_disk_mib=int(resources["min_free_disk_mib"]),
            max_output_mib=int(resources["max_output_mib"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.output_dir.exists():
            raise FileExistsError(f"Output exists: {self.output_dir}")
        if self.review_start_ui != self.quiet_start_ui or self.quiet_end_ui - self.quiet_start_ui + 1 != 100:
            raise ValueError("Benchmark requires the frozen 100-frame quiet interval at review start")
        if self.review_end_ui < self.quiet_end_ui:
            raise ValueError("Review interval does not contain the quiet interval")
        if self.offline_temporal_window_frames < 5 or self.offline_temporal_window_frames % 2 != 1:
            raise ValueError("Offline smoothing window must be odd and >=5")
        if self.derivative_lag_frames != 1:
            raise ValueError("Benchmark holds derivative lag at 1")
        if self.causal_structural_floors != (0.2, 0.4):
            raise ValueError("The preregistered causal floors are 0.2 and 0.4")
        if self.primary_match_radius_px not in self.match_radii_px:
            raise ValueError("Primary radius must appear in the sensitivity radii")
        if not 1 <= self.cpu_threads <= 8 or self.max_ram_mib < 1024:
            raise ValueError("Invalid resource bounds")


def _resolved(config: ActivityGateBenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def preflight(config: ActivityGateBenchmarkConfig, write_artifacts: bool = True) -> dict[str, Any]:
    inputs = (config.source_video, config.source_tiff, config.labels_tsv, config.design_document)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = v1.load_labels(config.labels_tsv)
    if video.ndim != 3 or config.review_end_ui > len(video):
        raise ValueError(f"Invalid source shape or review interval: {video.shape}")
    for row in labels:
        if not (0 <= row["x_px"] < video.shape[2] and 0 <= row["y_px"] < video.shape[1]):
            raise ValueError(f"Label outside source field: {row}")
    review_frames = config.review_end_ui - config.review_start_ui + 1
    float_stack_mib = math.ceil(review_frames * video.shape[1] * video.shape[2] * 4 / 2**20)
    estimated_peak_ram_mib = 7 * float_stack_mib + 512
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    disk_free_mib = shutil.disk_usage(probe).free // 2**20
    ram_available_mib = _available_ram_mib()
    ready = bool(
        estimated_peak_ram_mib <= config.max_ram_mib
        and ram_available_mib >= config.max_ram_mib
        and disk_free_mib >= config.min_free_disk_mib
    )
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "ready": ready,
        "source_shape": list(video.shape),
        "label_rows": len(labels),
        "unique_label_coordinates": len({(row["x_px"], row["y_px"]) for row in labels}),
        "review_ui_inclusive": [config.review_start_ui, config.review_end_ui],
        "lanes": list(LANES),
        "lane_count": len(LANES),
        "real_time_contract": {
            "offline_artifact_gate": "not real-time: centered filter has 3-frame look-ahead",
            "causal_lanes": "streamable after fixed 100-frame calibration; state is one temporal EMA frame and one energy EMA frame",
        },
        "resources": {
            "cpu_threads": config.cpu_threads,
            "estimated_peak_ram_mib": estimated_peak_ram_mib,
            "ram_cap_mib": config.max_ram_mib,
            "ram_available_mib": ram_available_mib,
            "disk_free_mib": disk_free_mib,
            "output_cap_mib": config.max_output_mib,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs
        ],
        "interpretation_contract": "Unmatched event candidates are unknown, not false positives. Known-label candidate fraction is only a lower bound on precision.",
    }
    if write_artifacts:
        config.preflight_dir.mkdir(parents=True, exist_ok=True)
        overlay = config.preflight_dir / "label_projection_overlay.png"
        v1._write_overlay(video, labels, overlay)
        payload["label_projection_overlay"] = str(overlay)
        _atomic_json(config.preflight_dir / "preflight.json", payload)
    if not ready:
        raise RuntimeError(f"Activity-gate benchmark preflight failed: {payload}")
    return payload


def _direct_map(frames: np.ndarray, tau: float) -> np.ndarray:
    from scipy.special import logsumexp

    return (tau * (logsumexp(frames / tau, axis=0) - math.log(len(frames)))).astype(np.float32)


def _quiet_windows(quiet: np.ndarray) -> list[np.ndarray]:
    return [quiet[start : start + duration] for start, duration in zip((0, 24, 48, 53), (24, 24, 28, 47))]


def _threshold(maps: list[np.ndarray], nms_distance: int, peaks_per_map: float) -> float:
    values = []
    for score in maps:
        values.extend(value for value, _, _ in v1._peaks(score, nms_distance, limit=2000))
    ranked = sorted(values, reverse=True)
    allowed = max(1, int(round(peaks_per_map * len(maps))))
    if len(ranked) <= allowed:
        raise RuntimeError("Too few quiet peaks for threshold calibration")
    return float(np.nextafter(ranked[allowed], np.inf))


def _match_with_peak_indices(peaks, rows, radius: int) -> tuple[list[tuple], set[int]]:
    remaining = set(range(len(rows)))
    matches, peak_indices = [], set()
    for peak_index, (score, x, y) in enumerate(peaks):
        choices = [(math.hypot(x - rows[i]["x_px"], y - rows[i]["y_px"]), i) for i in remaining]
        if not choices:
            break
        distance, row_index = min(choices)
        if distance <= radius:
            remaining.remove(row_index)
            matches.append((row_index, score, x, y, distance))
            peak_indices.add(peak_index)
    return matches, peak_indices


def _crossfit_quiet(quiet: np.ndarray, config: ActivityGateBenchmarkConfig) -> dict[str, Any]:
    starts, durations = (0, 24, 0, 3), (24, 24, 28, 47)
    halves = (quiet[:50], quiet[50:])
    directions = []
    for calibration_index, evaluation_index in ((0, 1), (1, 0)):
        calibration_maps = [_direct_map(halves[calibration_index][s : s + d], config.temporal_pool_tau) for s, d in zip(starts, durations)]
        threshold = _threshold(calibration_maps, config.nms_distance_px, config.quiet_false_peaks_per_map)
        evaluation_maps = [_direct_map(halves[evaluation_index][s : s + d], config.temporal_pool_tau) for s, d in zip(starts, durations)]
        counts = [len([p for p in v1._peaks(score, config.nms_distance_px, limit=2000) if p[0] >= threshold]) for score in evaluation_maps]
        directions.append({
            "calibration_half": calibration_index + 1,
            "evaluation_half": evaluation_index + 1,
            "threshold": threshold,
            "evaluation_peak_counts": counts,
            "peaks_per_map": float(np.mean(counts)),
        })
    return {
        "directions": directions,
        "mean_heldout_quiet_peaks_per_map": float(np.mean([row["peaks_per_map"] for row in directions])),
    }


def _evaluate_lane(
    name: str,
    quiet: np.ndarray,
    bursts: dict[int, np.ndarray],
    labels: list[dict[str, Any]],
    config: ActivityGateBenchmarkConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    quiet_maps = [_direct_map(window, config.temporal_pool_tau) for window in _quiet_windows(quiet)]
    thresholds = {
        str(rate): _threshold(quiet_maps, config.nms_distance_px, rate)
        for rate in (0.25, 0.5, 1.0, 2.0, 5.0)
    }
    primary = thresholds[str(float(config.quiet_false_peaks_per_map))]
    fold_rows, candidates = [], []
    radius_summaries = {str(radius): [] for radius in config.match_radii_px}
    for burst_id in range(1, 5):
        score = _direct_map(bursts[burst_id], config.temporal_pool_tau)
        peaks = [peak for peak in v1._peaks(score, config.nms_distance_px, limit=2000) if peak[0] >= primary]
        rows = [row for row in labels if row["burst_id"] == burst_id]
        primary_matches, matched_peak_indices = _match_with_peak_indices(peaks, rows, config.primary_match_radius_px)
        fold_rows.append({
            "heldout_burst": burst_id,
            "matched": len(primary_matches),
            "labels": len(rows),
            "recall": len(primary_matches) / len(rows),
            "event_peaks": len(peaks),
            "known_label_candidate_fraction_lower_bound": len(primary_matches) / len(peaks) if peaks else 0.0,
        })
        for index, (value, x, y) in enumerate(peaks):
            nearest = min(math.hypot(x - row["x_px"], y - row["y_px"]) for row in rows)
            candidates.append({
                "lane": name, "burst_id": burst_id, "score": value, "x_px": x, "y_px": y,
                "matched_known_label": index in matched_peak_indices,
                "nearest_known_label_px": nearest,
                "interpretation": "known match" if index in matched_peak_indices else "unmatched candidate; truth unknown",
            })
        for radius in config.match_radii_px:
            matches = v1._match(peaks, rows, radius)
            radius_summaries[str(radius)].append(len(matches) / len(rows))
    froc = []
    for rate, threshold in thresholds.items():
        recalls, peak_counts = [], []
        for burst_id in range(1, 5):
            score = _direct_map(bursts[burst_id], config.temporal_pool_tau)
            peaks = [peak for peak in v1._peaks(score, config.nms_distance_px, limit=2000) if peak[0] >= threshold]
            rows = [row for row in labels if row["burst_id"] == burst_id]
            recalls.append(len(v1._match(peaks, rows, config.primary_match_radius_px)) / len(rows))
            peak_counts.append(len(peaks))
        froc.append({"quiet_peaks_per_map_target": float(rate), "threshold": threshold,
                     "mean_recall": float(np.mean(recalls)), "mean_event_peaks": float(np.mean(peak_counts))})
    result = {
        "lane": name,
        "primary_threshold": primary,
        "outer_folds": fold_rows,
        "mean_recall": float(np.mean([row["recall"] for row in fold_rows])),
        "pooled_recall": sum(row["matched"] for row in fold_rows) / sum(row["labels"] for row in fold_rows),
        "total_matched": sum(row["matched"] for row in fold_rows),
        "total_labels": sum(row["labels"] for row in fold_rows),
        "total_event_peaks": sum(row["event_peaks"] for row in fold_rows),
        "known_label_candidate_fraction_lower_bound": sum(row["matched"] for row in fold_rows) / max(1, sum(row["event_peaks"] for row in fold_rows)),
        "mean_recall_by_match_radius": {radius: float(np.mean(values)) for radius, values in radius_summaries.items()},
        "froc": froc,
        "crossfit_quiet": _crossfit_quiet(quiet, config),
    }
    return result, candidates


def _residual_segments(
    transformed: np.ndarray,
    labels: list[dict[str, Any]],
    config: ActivityGateBenchmarkConfig,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    quiet_count = config.quiet_end_ui - config.quiet_start_ui + 1
    quiet_raw = transformed[:quiet_count]
    baseline = np.median(quiet_raw, axis=0)
    low, high = np.percentile(quiet_raw[:, ::4, ::4], [1.0, 99.9])
    scale = max(float(high - low), 1e-6)
    quiet = np.maximum((quiet_raw - baseline) / scale, 0).astype(np.float32)
    review_start_zero = config.review_start_ui - 1
    bursts = {}
    for burst_id in range(1, 5):
        rows = [row for row in labels if row["burst_id"] == burst_id]
        start = rows[0]["start_frame_zero"] - review_start_zero
        stop = rows[0]["stop_frame_zero_exclusive"] - review_start_zero
        bursts[burst_id] = np.maximum((transformed[start:stop] - baseline) / scale, 0).astype(np.float32)
    return quiet, bursts


def _gate_from_filtered(
    filtered: np.ndarray,
    config: ActivityGateBenchmarkConfig,
    previous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    quiet_count = config.quiet_end_ui - config.quiet_start_ui + 1
    if previous is not None:
        if previous.shape != filtered.shape:
            raise ValueError("Previous-frame stack must align with filtered frames")
        derivative = filtered - previous
    else:
        derivative = np.empty_like(filtered, dtype=np.float32)
        derivative[0] = 0
        derivative[1:] = filtered[1:] - filtered[:-1]
    quiet_derivative = derivative[:quiet_count]
    center = np.median(quiet_derivative, axis=0)
    mad = np.median(np.abs(quiet_derivative - center), axis=0) * 1.4826
    positive = mad[mad > 0]
    floor = float(np.percentile(positive, config.quiet_mad_floor_percentile)) if positive.size else 1.0
    derivative -= center[None]
    derivative /= np.maximum(mad, max(floor, 1e-6))[None]
    alpha = 2 / (config.energy_ema_span_frames + 1)
    energy = np.empty_like(derivative, dtype=np.float32)
    energy[0] = derivative[0] ** 2
    for index in range(1, len(derivative)):
        energy[index] = alpha * derivative[index] ** 2 + (1 - alpha) * energy[index - 1]
    gate = (1 - np.exp(-energy / (2 * config.gate_tau_z**2))).astype(np.float32)
    artifact, summary = _artifact_score(filtered[:quiet_count])
    summary["derivative_scale_floor"] = floor
    return gate, artifact, summary


def _compress_intensity(frames: np.ndarray, calibration: np.ndarray, gain: float) -> np.ndarray:
    low, high = np.percentile(calibration[:, ::4, ::4], [1.0, 99.8])
    unit = np.clip((frames - low) / max(float(high - low), 1e-6), 0, 1)
    return (np.arcsinh(gain * unit) / np.arcsinh(gain)).astype(np.float32)


def _causal_filter(raw: np.ndarray, config: ActivityGateBenchmarkConfig) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    spatial = gaussian_filter(raw, sigma=(0, config.spatial_sigma_px, config.spatial_sigma_px), mode="reflect", truncate=4).astype(np.float32)
    alpha = 2 / (config.causal_temporal_ema_span_frames + 1)
    causal = np.empty_like(spatial, dtype=np.float32)
    causal[0] = spatial[0]
    for index in range(1, len(spatial)):
        causal[index] = alpha * spatial[index] + (1 - alpha) * causal[index - 1]
    return causal


def _write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["lane", "burst_id", "score", "x_px", "y_px", "matched_known_label", "nearest_known_label_px", "interpretation"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _matched_at_reference_budget(
    lane: str,
    candidates: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    reference_folds: list[dict[str, Any]],
    radius: int,
) -> tuple[dict[str, Any], set[tuple[int, str]]]:
    folds, matched_keys = [], set()
    for burst_id in range(1, 5):
        budget = int(reference_folds[burst_id - 1]["event_peaks"])
        lane_rows = sorted(
            (row for row in candidates if row["lane"] == lane and row["burst_id"] == burst_id),
            key=lambda row: row["score"],
            reverse=True,
        )[:budget]
        peaks = [(row["score"], row["x_px"], row["y_px"]) for row in lane_rows]
        burst_labels = [row for row in labels if row["burst_id"] == burst_id]
        matches = v1._match(peaks, burst_labels, radius)
        for row_index, *_ in matches:
            matched_keys.add((burst_id, burst_labels[row_index]["roi_identity"]))
        folds.append({
            "heldout_burst": burst_id,
            "candidate_budget": budget,
            "available_candidates": len(lane_rows),
            "matched": len(matches),
            "labels": len(burst_labels),
            "recall": len(matches) / len(burst_labels),
        })
    return {
        "outer_folds": folds,
        "mean_recall": float(np.mean([row["recall"] for row in folds])),
        "pooled_recall": sum(row["matched"] for row in folds) / sum(row["labels"] for row in folds),
        "matched": sum(row["matched"] for row in folds),
        "labels": sum(row["labels"] for row in folds),
    }, matched_keys


def _cluster_bootstrap_difference(
    lane_keys: set[tuple[int, str]],
    baseline_keys: set[tuple[int, str]],
    labels: list[dict[str, Any]],
    seed: int = 20260726,
    samples: int = 10000,
) -> dict[str, Any]:
    identities = sorted({row["roi_identity"] for row in labels})
    by_identity = {
        identity: [(row["burst_id"], row["roi_identity"]) for row in labels if row["roi_identity"] == identity]
        for identity in identities
    }
    gains = len(lane_keys - baseline_keys)
    losses = len(baseline_keys - lane_keys)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        lane_hits = baseline_hits = total = 0
        for identity in sampled:
            keys = by_identity[str(identity)]
            total += len(keys)
            lane_hits += sum(key in lane_keys for key in keys)
            baseline_hits += sum(key in baseline_keys for key in keys)
        differences[index] = (lane_hits - baseline_hits) / total
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "cluster_unit": "roi_identity",
        "bootstrap_samples": samples,
        "seed": seed,
        "point_difference_pooled_recall": (len(lane_keys) - len(baseline_keys)) / len(labels),
        "percentile_95_ci": [float(low), float(high)],
        "probability_difference_gt_zero": float(np.mean(differences > 0)),
        "discordant_gains": gains,
        "discordant_losses": losses,
    }


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        f"# {metrics['experiment_id']}", "", f"Status: `{metrics['status']}`.", "",
        "## Real-time finding", "",
        "The offline artifact gate is not real-time because its centered seven-frame Savitzky-Golay filter uses three future frames (60 ms at 50 Hz). The causal lanes replace it with a one-state temporal EMA and can stream after a fixed 100-frame calibration period.", "",
        "## Primary comparison", "",
        "| Lane | Mean recall | Pooled recall | Recall at Raw Direct candidate budget | Event peaks | Known-label fraction (lower bound only) | Held-out quiet peaks/map |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["lanes"]:
        lines.append(f"| `{row['lane']}` | {row['mean_recall']:.4f} | {row['pooled_recall']:.4f} | {row['matched_reference_budget']['pooled_recall']:.4f} | {row['total_event_peaks']} | {row['known_label_candidate_fraction_lower_bound']:.4f} | {row['crossfit_quiet']['mean_heldout_quiet_peaks_per_map']:.3f} |")
    lines.extend(["", "Unmatched event peaks remain unknown because the event labels are not exhaustive. The known-label fraction is not an estimate of true precision.", "", "## Decision", "", metrics["decision"]["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: ActivityGateBenchmarkConfig) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.cpu_threads)
    audit = preflight(config, write_artifacts=True)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(config.output_dir / "resolved_config.json", _resolved(config))
    _atomic_json(config.output_dir / "preflight.json", audit)
    _atomic_json(config.output_dir / "run_state.json", {"status": "running", "phase": "raw_direct"})
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = v1.load_labels(config.labels_tsv)
    v1._write_overlay(video, labels, config.output_dir / "label_projection_overlay.png")
    start, stop = config.review_start_ui - 1, config.review_end_ui
    raw = np.asarray(video[start:stop], dtype=np.float32)
    lane_results, candidate_rows, preprocessing = [], [], {}

    quiet, bursts = _residual_segments(raw, labels, config)
    result, candidates = _evaluate_lane("raw_direct", quiet, bursts, labels, config)
    lane_results.append(result); candidate_rows.extend(candidates)
    raw_artifact, raw_artifact_summary = _artifact_score(raw[:100])
    quiet, bursts = _residual_segments(raw * (1 - config.artifact_attenuation * raw_artifact[None]), labels, config)
    result, candidates = _evaluate_lane("raw_direct_static_artifact", quiet, bursts, labels, config)
    lane_results.append(result); candidate_rows.extend(candidates)
    preprocessing["raw_static_artifact"] = raw_artifact_summary
    del quiet, bursts, raw_artifact
    gc.collect()

    smoothing_config = type("SmoothingConfig", (), {
        "review_start_ui": config.review_start_ui,
        "review_end_ui": config.review_end_ui,
        "derivative_lag_frames": config.derivative_lag_frames,
        "temporal_window_frames": config.offline_temporal_window_frames,
        "temporal_polyorder": config.offline_temporal_polyorder,
        "spatial_sigma_px": config.spatial_sigma_px,
    })()
    smoothed, load_start = _smooth_review(video, smoothing_config)
    offline = smoothed[start - load_start : stop - load_start]
    offline_previous = smoothed[start - 1 - load_start : stop - 1 - load_start]
    offline_gate, offline_artifact, offline_summary = _gate_from_filtered(
        offline, config, previous=offline_previous
    )
    offline_compressed = _compress_intensity(offline, offline, config.intensity_asinh_gain)
    offline_transformed = offline_compressed * (0.2 + 0.8 * offline_gate) * (1 - config.artifact_attenuation * offline_artifact[None])
    quiet, bursts = _residual_segments(offline_transformed, labels, config)
    result, candidates = _evaluate_lane("offline_artifact_gate", quiet, bursts, labels, config)
    lane_results.append(result); candidate_rows.extend(candidates)
    preprocessing["offline"] = offline_summary
    del smoothed, offline, offline_previous, offline_gate, offline_artifact, offline_compressed, offline_transformed, quiet, bursts
    gc.collect()

    _atomic_json(config.output_dir / "run_state.json", {"status": "running", "phase": "causal_lanes"})
    causal = _causal_filter(raw, config)
    causal_gate, causal_artifact, causal_summary = _gate_from_filtered(causal, config)
    causal_compressed = _compress_intensity(causal, causal[:100], config.intensity_asinh_gain)
    artifact_factor = 1 - config.artifact_attenuation * causal_artifact[None]
    causal_only = causal_compressed * artifact_factor
    quiet, bursts = _residual_segments(causal_only, labels, config)
    result, candidates = _evaluate_lane("causal_artifact_only", quiet, bursts, labels, config)
    lane_results.append(result); candidate_rows.extend(candidates)
    del causal_only, quiet, bursts
    for floor in config.causal_structural_floors:
        lane_name = f"causal_artifact_gate_floor_{str(floor).replace('.', 'p')}"
        transformed = causal_compressed * (floor + (1 - floor) * causal_gate) * artifact_factor
        quiet, bursts = _residual_segments(transformed, labels, config)
        result, candidates = _evaluate_lane(lane_name, quiet, bursts, labels, config)
        lane_results.append(result); candidate_rows.extend(candidates)
        del transformed, quiet, bursts
    preprocessing["causal"] = causal_summary
    del causal, causal_gate, causal_artifact, causal_compressed, artifact_factor, raw
    gc.collect()

    baseline = next(row for row in lane_results if row["lane"] == "raw_direct")
    matched_key_sets = {}
    for row in lane_results:
        row["fold_wins_vs_raw_direct"] = sum(
            fold["recall"] > baseline["outer_folds"][index]["recall"]
            for index, fold in enumerate(row["outer_folds"])
        )
        row["mean_recall_delta_vs_raw_direct"] = row["mean_recall"] - baseline["mean_recall"]
        matched_budget, matched_keys = _matched_at_reference_budget(
            row["lane"], candidate_rows, labels, baseline["outer_folds"],
            config.primary_match_radius_px,
        )
        row["matched_reference_budget"] = matched_budget
        row["event_candidate_count_ratio_vs_raw_direct"] = row["total_event_peaks"] / baseline["total_event_peaks"]
        matched_key_sets[row["lane"]] = matched_keys
    causal_candidates = [row for row in lane_results if row["lane"].startswith("causal_")]
    best_causal = max(causal_candidates, key=lambda row: (
        row["matched_reference_budget"]["pooled_recall"], row["mean_recall"],
        -row["crossfit_quiet"]["mean_heldout_quiet_peaks_per_map"],
    ))
    bootstrap = _cluster_bootstrap_difference(
        matched_key_sets[best_causal["lane"]], matched_key_sets[baseline["lane"]], labels
    )
    advance = bool(
        best_causal["mean_recall"] > baseline["mean_recall"]
        and best_causal["fold_wins_vs_raw_direct"] >= 3
        and best_causal["matched_reference_budget"]["pooled_recall"] > baseline["pooled_recall"]
        and best_causal["crossfit_quiet"]["mean_heldout_quiet_peaks_per_map"]
        <= baseline["crossfit_quiet"]["mean_heldout_quiet_peaks_per_map"]
    )
    decision = {
        "status": "promising_causal_ranking_not_precision" if advance else "do_not_replace_raw_direct",
        "best_causal_lane": best_causal["lane"],
        "paired_roi_bootstrap": bootstrap,
        "interpretation": (
            f"{best_causal['lane']} improved recall and ranking at the Raw Direct candidate budget without worse held-out quiet behavior. Its event candidate count is still higher, so this is not a demonstrated precision improvement; confirm on reviewed hard negatives before replacement."
            if advance else
            f"No causal lane exceeded Raw Direct in at least three of four bursts. Keep Raw Direct primary; use the artifact gate as a visualization or auxiliary feature until hard-negative labels exist."
        ),
    }
    metrics = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "source_video_sha256": _sha256(config.source_video),
        "lane_count": len(lane_results),
        "lanes": lane_results,
        "preprocessing": preprocessing,
        "decision": decision,
        "precision_contract": "Known-label candidate fraction is a lower bound only. Unmatched event candidates are unknown, not false positives.",
    }
    _atomic_json(config.output_dir / "metrics.json", metrics)
    _write_candidates(config.output_dir / "candidate_peaks.tsv", candidate_rows)
    _write_report(config.output_dir / "report.md", metrics)
    summary = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "raw_direct_mean_recall": baseline["mean_recall"],
        "best_causal_lane": best_causal["lane"],
        "best_causal_mean_recall": best_causal["mean_recall"],
        "decision": decision,
        "metrics": "metrics.json",
        "report": "report.md",
    }
    _atomic_json(config.output_dir / "experiment_summary.json", summary)
    _atomic_json(config.output_dir / "run_state.json", {"status": "complete", "phase": "complete"})
    return summary
