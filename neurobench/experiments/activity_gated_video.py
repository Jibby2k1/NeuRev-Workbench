"""Derivative-energy gating and persistent-artifact attenuation review stacks."""
from __future__ import annotations

import csv
import gc
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .frame_difference import _atomic_json, _available_ram_mib, _sha256


@dataclass(frozen=True)
class ActivityGateConfig:
    experiment_id: str
    source_video: Path
    source_tiff: Path
    labels_tsv: Path
    output_dir: Path
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    spatial_sigma_px: float
    temporal_window_frames: int
    temporal_polyorder: int
    derivative_lag_frames: int
    energy_ema_span_frames: float
    gate_tau_z: float
    structural_floor: float
    artifact_attenuation: float
    intensity_asinh_gain: float
    quiet_mad_floor_percentile: float
    frame_chunk: int
    cpu_threads: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int

    @classmethod
    def load(cls, path: str | Path) -> "ActivityGateConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        frames, smoothing, gate, resources = (
            raw["frames"], raw["smoothing"], raw["gate"], raw["resources"]
        )
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            source_tiff=(root / raw["source_tiff"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            review_start_ui=int(frames["review_start_ui"]),
            review_end_ui=int(frames["review_end_ui"]),
            quiet_start_ui=int(frames["quiet_start_ui"]),
            quiet_end_ui=int(frames["quiet_end_ui"]),
            spatial_sigma_px=float(smoothing.get("spatial_sigma_px", 1)),
            temporal_window_frames=int(smoothing.get("temporal_window_frames", 7)),
            temporal_polyorder=int(smoothing.get("temporal_polyorder", 2)),
            derivative_lag_frames=int(gate.get("derivative_lag_frames", 1)),
            energy_ema_span_frames=float(gate.get("energy_ema_span_frames", 4)),
            gate_tau_z=float(gate.get("gate_tau_z", 2.5)),
            structural_floor=float(gate.get("structural_floor", 0.2)),
            artifact_attenuation=float(gate.get("artifact_attenuation", 0.7)),
            intensity_asinh_gain=float(gate.get("intensity_asinh_gain", 5)),
            quiet_mad_floor_percentile=float(gate.get("quiet_mad_floor_percentile", 10)),
            frame_chunk=int(resources.get("frame_chunk", 64)),
            cpu_threads=int(resources.get("cpu_threads", 6)),
            max_ram_mib=int(resources.get("max_ram_mib", 4096)),
            min_free_disk_mib=int(resources.get("min_free_disk_mib", 4096)),
            max_output_mib=int(resources.get("max_output_mib", 1024)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.review_end_ui < self.review_start_ui:
            raise ValueError("Review interval is empty")
        if not (self.review_start_ui <= self.quiet_start_ui <= self.quiet_end_ui <= self.review_end_ui):
            raise ValueError("Quiet interval must lie inside the review interval")
        if self.temporal_window_frames < 5 or self.temporal_window_frames % 2 != 1:
            raise ValueError("Temporal window must be odd and >=5")
        if not 1 <= self.temporal_polyorder < self.temporal_window_frames:
            raise ValueError("Invalid temporal polynomial order")
        if self.derivative_lag_frames != 1:
            raise ValueError("This comparison holds derivative lag fixed at 1")
        if self.energy_ema_span_frames < 1 or self.gate_tau_z <= 0:
            raise ValueError("EMA span and gate tau must be positive")
        if not 0 <= self.structural_floor <= 1 or not 0 <= self.artifact_attenuation <= 1:
            raise ValueError("Floor and attenuation must be in [0,1]")
        if not 1 <= self.frame_chunk <= 256 or not 1 <= self.cpu_threads <= 24:
            raise ValueError("Invalid resource chunk/thread setting")


def _disk_free_mib(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free // 2**20


def preflight(config: ActivityGateConfig) -> dict[str, Any]:
    inputs = (config.source_video, config.source_tiff, config.labels_tsv)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if config.output_dir.exists():
        raise FileExistsError(f"Output exists: {config.output_dir}")
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if video.ndim != 3 or config.review_end_ui > len(video):
        raise ValueError(f"Invalid video or frame interval: {video.shape}")
    review_frames = config.review_end_ui - config.review_start_ui + 1
    final_bytes = review_frames * video.shape[1] * video.shape[2] * 2 * 4
    final_mib = math.ceil(final_bytes / 2**20)
    float_stack_mib = math.ceil(
        review_frames * video.shape[1] * video.shape[2] * 4 / 2**20
    )
    # Six concurrent float stacks, one encoded uint16 stack, and scientific
    # runtime/TIFF headroom bound the sequential writer's expected peak.
    estimated_peak_ram_mib = 6 * float_stack_mib + math.ceil(float_stack_mib / 2) + 512
    disk_free = _disk_free_mib(config.output_dir.parent)
    ram_free = _available_ram_mib()
    ready = (
        final_mib <= config.max_output_mib
        and disk_free >= final_mib + config.min_free_disk_mib
        and ram_free >= config.max_ram_mib
        and estimated_peak_ram_mib <= config.max_ram_mib
    )
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "ready": ready,
        "source_shape": list(video.shape),
        "review_ui_inclusive": [config.review_start_ui, config.review_end_ui],
        "review_frames": review_frames,
        "planned_outputs": ["strict_gate", "floored_gate", "artifact_gate", "baseline_residual"],
        "expected_output_mib": final_mib,
        "processing": {
            "motion_correction": False,
            "spatial_sigma_px": config.spatial_sigma_px,
            "temporal_savgol_window": config.temporal_window_frames,
            "temporal_savgol_polyorder": config.temporal_polyorder,
            "derivative_lag_frames": config.derivative_lag_frames,
            "energy_ema_span_frames": config.energy_ema_span_frames,
            "gate_tau_z": config.gate_tau_z,
            "structural_floor": config.structural_floor,
            "artifact_attenuation": config.artifact_attenuation,
        },
        "resources": {
            "frame_chunk": config.frame_chunk,
            "cpu_threads": config.cpu_threads,
            "ram_available_mib": ram_free,
            "ram_cap_mib": config.max_ram_mib,
            "estimated_peak_ram_mib": estimated_peak_ram_mib,
            "disk_free_mib": disk_free,
            "output_cap_mib": config.max_output_mib,
        },
    }
    if not ready:
        raise RuntimeError(f"Activity-gate preflight failed: {payload}")
    return payload


def _load_labels(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [{
        **row,
        "burst_id": int(row["burst_id"]),
        "start_frame_zero": int(row["start_frame_zero"]),
        "stop_frame_zero_exclusive": int(row["stop_frame_zero_exclusive"]),
        "x_px": float(row["x_px"]),
        "y_px": float(row["y_px"]),
    } for row in rows]


def _smooth_review(video: np.ndarray, config: ActivityGateConfig) -> tuple[np.ndarray, int]:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import savgol_filter

    start = config.review_start_ui - 1
    stop = config.review_end_ui
    half = config.temporal_window_frames // 2
    load_start = max(0, start - config.derivative_lag_frames - half)
    load_stop = min(len(video), stop + half)
    block = np.asarray(video[load_start:load_stop], dtype=np.float32)
    spatial = gaussian_filter(
        block,
        sigma=(0, config.spatial_sigma_px, config.spatial_sigma_px),
        mode="reflect",
        truncate=4,
    )
    smoothed = savgol_filter(
        spatial,
        config.temporal_window_frames,
        config.temporal_polyorder,
        axis=0,
        mode="interp",
    ).astype(np.float32)
    return smoothed, load_start


def _artifact_score(quiet: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    baseline = np.median(quiet, axis=0)
    mad = np.median(np.abs(quiet - baseline), axis=0) * 1.4826
    q98, q999 = np.percentile(baseline, [98, 99.9])
    brightness = np.clip((baseline - q98) / max(q999 - q98, 1e-6), 0, 1)
    coefficient = mad / np.maximum(baseline, 1)
    c10, c50 = np.percentile(coefficient[baseline >= q98], [10, 50]) if np.any(baseline >= q98) else (0, 1)
    stability = 1 - np.clip((coefficient - c10) / max(c50 - c10, 1e-6), 0, 1)
    saturation = np.mean(quiet >= 4094, axis=0).astype(np.float32)
    score = np.maximum(brightness * stability, saturation).astype(np.float32)
    return score, {
        "baseline_q98": float(q98),
        "baseline_q99_9": float(q999),
        "bright_cv_q10": float(c10),
        "bright_cv_q50": float(c50),
        "artifact_area_ge_0_5": float((score >= 0.5).mean()),
        "artifact_area_ge_0_8": float((score >= 0.8).mean()),
    }


def _encode_unit(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.clip(value, 0, 1) * 65535), 0, 65535).astype(np.uint16)


def _roi_event_contrast(stack: np.ndarray, labels: list[dict[str, Any]], review_start_zero: int, quiet_slice: slice) -> dict[str, float]:
    quiet = stack[quiet_slice]
    contrasts = []
    for row in labels:
        x, y = int(round(row["x_px"])), int(round(row["y_px"]))
        start = row["start_frame_zero"] - review_start_zero
        stop = row["stop_frame_zero_exclusive"] - review_start_zero
        if start < 0 or stop > len(stack):
            continue
        event_value = float(np.max(stack[start:stop, y, x]))
        quiet_value = float(np.median(quiet[:, y, x]))
        contrasts.append(event_value - quiet_value)
    return {
        "label_rows": len(contrasts),
        "median_event_minus_quiet": float(np.median(contrasts)) if contrasts else 0.0,
        "p25_event_minus_quiet": float(np.percentile(contrasts, 25)) if contrasts else 0.0,
        "positive_contrast_fraction": float(np.mean(np.asarray(contrasts) > 0)) if contrasts else 0.0,
    }


def run(config: ActivityGateConfig) -> dict[str, Any]:
    audit = preflight(config)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    resolved = asdict(config)
    for key, value in tuple(resolved.items()):
        if isinstance(value, Path):
            resolved[key] = str(value)
    _atomic_json(config.output_dir / "config.resolved.json", resolved)
    _atomic_json(config.output_dir / "run_state.json", {
        "status": "running", "phase": "smoothing", "completed_variants": [],
    })
    source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = _load_labels(config.labels_tsv)
    smoothed, load_start = _smooth_review(source, config)
    review_start = config.review_start_ui - 1
    review_stop = config.review_end_ui
    current = smoothed[review_start - load_start : review_stop - load_start]
    previous = smoothed[review_start - config.derivative_lag_frames - load_start : review_stop - config.derivative_lag_frames - load_start]
    derivative = current - previous
    q0 = config.quiet_start_ui - config.review_start_ui
    q1 = config.quiet_end_ui - config.review_start_ui + 1
    quiet_derivative = derivative[q0:q1]
    derivative_center = np.median(quiet_derivative, axis=0)
    derivative_mad = np.median(np.abs(quiet_derivative - derivative_center), axis=0) * 1.4826
    positive = derivative_mad[derivative_mad > 0]
    derivative_floor = float(np.percentile(positive, config.quiet_mad_floor_percentile)) if positive.size else 1.0
    derivative_scale = np.maximum(derivative_mad, max(derivative_floor, 1e-6)).astype(np.float32)
    derivative -= derivative_center[None]
    derivative /= derivative_scale[None]
    alpha = 2 / (config.energy_ema_span_frames + 1)
    energy = np.empty_like(derivative, dtype=np.float32)
    energy[0] = derivative[0] ** 2
    for index in range(1, len(derivative)):
        energy[index] = alpha * derivative[index] ** 2 + (1 - alpha) * energy[index - 1]
    gate = (1 - np.exp(-energy / (2 * config.gate_tau_z**2))).astype(np.float32)
    del energy, derivative, quiet_derivative
    gc.collect()
    quiet = current[q0:q1]
    baseline = np.median(quiet, axis=0).astype(np.float32)
    artifact, artifact_summary = _artifact_score(quiet)
    intensity_lo, intensity_hi = np.percentile(current[:, ::4, ::4], [1, 99.8])
    intensity_unit = np.clip((current - intensity_lo) / max(intensity_hi - intensity_lo, 1e-6), 0, 1)
    compressed = np.arcsinh(config.intensity_asinh_gain * intensity_unit) / np.arcsinh(config.intensity_asinh_gain)
    source_sha256 = _sha256(config.source_video)
    records = []
    artifact_mask = artifact >= 0.5
    anatomy_mask = (baseline >= np.percentile(baseline, 50)) & (baseline < np.percentile(baseline, 95)) & ~artifact_mask
    def write_variant(name: str, value: np.ndarray) -> None:
        path = config.output_dir / f"spon_ca_burst_{name}.tif"
        description = json.dumps({
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "variant": name,
            "review_start_ui": config.review_start_ui,
            "review_end_ui": config.review_end_ui,
            "source_frame_offset_zero": review_start,
            "zero": "black",
            "motion_correction": False,
            "source_video_sha256": source_sha256,
        }, sort_keys=True)
        encoded = _encode_unit(value)
        temporary = path.with_name(path.name + ".partial")
        tifffile.imwrite(
            temporary,
            encoded,
            bigtiff=True,
            photometric="minisblack",
            metadata={"axes": "TYX"},
            description=description,
        )
        temporary.replace(path)
        with tifffile.TiffFile(path) as tif:
            if tuple(tif.series[0].shape) != tuple(encoded.shape):
                raise RuntimeError(f"TIFF validation failed: {path}")
        quiet_median = np.median(value[q0:q1], axis=0)
        records.append({
            "variant": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "shape": list(encoded.shape),
            "artifact_quiet_median": float(np.median(quiet_median[artifact_mask])) if np.any(artifact_mask) else None,
            "anatomy_quiet_median": float(np.median(quiet_median[anatomy_mask])) if np.any(anatomy_mask) else None,
            "nonzero_fraction": float((encoded > 0).mean()),
            "label_event_contrast": _roi_event_contrast(value, labels, review_start, slice(q0, q1)),
        })
        _atomic_json(config.output_dir / "run_state.json", {
            "status": "running",
            "phase": "writing_variants",
            "completed_variants": [record["variant"] for record in records],
        })

    strict = compressed * gate
    write_variant("strict_gate", strict)
    del strict
    floored = compressed * (config.structural_floor + (1 - config.structural_floor) * gate)
    write_variant("floored_gate", floored)
    artifact_gate = floored * (1 - config.artifact_attenuation * artifact[None])
    del floored
    write_variant("artifact_gate", artifact_gate)
    del artifact_gate, gate, intensity_unit
    gc.collect()
    residual = np.maximum(current - baseline[None], 0)
    residual_scale = max(float(np.percentile(residual[:, ::4, ::4], 99.5)), 1e-6)
    baseline_unit = np.clip((baseline - intensity_lo) / max(intensity_hi - intensity_lo, 1e-6), 0, 1)
    baseline_compressed = np.arcsinh(config.intensity_asinh_gain * baseline_unit) / np.arcsinh(config.intensity_asinh_gain)
    baseline_residual = np.clip(0.15 * baseline_compressed[None] + residual / residual_scale, 0, 1)
    write_variant("baseline_residual", baseline_residual)
    del baseline_residual, residual, compressed, current, smoothed
    gc.collect()
    artifact_path = config.output_dir / "persistent_artifact_score.tif"
    tifffile.imwrite(artifact_path, _encode_unit(artifact), photometric="minisblack", description=json.dumps(artifact_summary, sort_keys=True))
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "preflight": audit,
        "source_video": str(config.source_video),
        "source_video_sha256": source_sha256,
        "artifact_score_map": str(artifact_path),
        "artifact_summary": artifact_summary,
        "normalization": {
            "intensity_lo": float(intensity_lo),
            "intensity_hi": float(intensity_hi),
            "residual_scale": residual_scale,
            "derivative_scale_floor": derivative_floor,
            "fixed_across_frames": True,
        },
        "outputs": records,
    }
    _atomic_json(config.output_dir / "manifest.json", payload)
    _atomic_json(config.output_dir / "run_state.json", {
        "status": "complete",
        "phase": "complete",
        "completed_variants": [record["variant"] for record in records],
    })
    return payload
