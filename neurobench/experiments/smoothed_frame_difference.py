"""Smoothed signed derivatives with global and quiet-MAD display encodings."""
from __future__ import annotations

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
class SmoothedDifferenceConfig:
    experiment_id: str
    source_video: Path
    source_tiff: Path
    output_dir: Path
    lags: tuple[int, ...]
    frame_period_ms: float
    quiet_start_ui: int
    quiet_end_ui: int
    spatial_sigma_px: float
    temporal_window_frames: int
    temporal_polyorder: int
    global_absolute_percentile: float
    quiet_mad_floor_percentile: float
    quiet_clip_z: float
    quiet_deadband_z: float
    sample_spatial_stride: int
    frame_chunk: int
    cpu_threads: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int

    @classmethod
    def load(cls, path: str | Path) -> "SmoothedDifferenceConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        smoothing = raw["smoothing"]
        normalization = raw["normalization"]
        resources = raw["resources"]
        quiet = raw["quiet_frames"]
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            source_tiff=(root / raw["source_tiff"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            lags=tuple(int(value) for value in raw.get("lags", (1, 4))),
            frame_period_ms=float(raw.get("frame_period_ms", 20)),
            quiet_start_ui=int(quiet["start_ui"]),
            quiet_end_ui=int(quiet["end_ui"]),
            spatial_sigma_px=float(smoothing.get("spatial_sigma_px", 1.0)),
            temporal_window_frames=int(smoothing.get("temporal_window_frames", 7)),
            temporal_polyorder=int(smoothing.get("temporal_polyorder", 2)),
            global_absolute_percentile=float(normalization.get("global_absolute_percentile", 99.5)),
            quiet_mad_floor_percentile=float(normalization.get("quiet_mad_floor_percentile", 10)),
            quiet_clip_z=float(normalization.get("quiet_clip_z", 5)),
            quiet_deadband_z=float(normalization.get("quiet_deadband_z", 2.5)),
            sample_spatial_stride=int(normalization.get("sample_spatial_stride", 4)),
            frame_chunk=int(resources.get("frame_chunk", 64)),
            cpu_threads=int(resources.get("cpu_threads", 6)),
            max_ram_mib=int(resources.get("max_ram_mib", 6144)),
            min_free_disk_mib=int(resources.get("min_free_disk_mib", 8192)),
            max_output_mib=int(resources.get("max_output_mib", 4096)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.lags != (1, 4):
            raise ValueError("This diagnostic contract requires lags [1, 4]")
        if self.spatial_sigma_px <= 0 or self.spatial_sigma_px > 3:
            raise ValueError("spatial_sigma_px must be in (0, 3]")
        if self.temporal_window_frames < 5 or self.temporal_window_frames % 2 != 1:
            raise ValueError("temporal_window_frames must be odd and >=5")
        if not 1 <= self.temporal_polyorder < self.temporal_window_frames:
            raise ValueError("temporal_polyorder must be smaller than the window")
        if not 90 <= self.global_absolute_percentile < 100:
            raise ValueError("global_absolute_percentile must be in [90,100)")
        if not 0 <= self.quiet_deadband_z < self.quiet_clip_z:
            raise ValueError("quiet_deadband_z must be in [0, quiet_clip_z)")
        if not 1 <= self.sample_spatial_stride <= 16:
            raise ValueError("sample_spatial_stride must be in [1,16]")
        if not 1 <= self.frame_chunk <= 256:
            raise ValueError("frame_chunk must be in [1,256]")
        if not 1 <= self.cpu_threads <= 24:
            raise ValueError("cpu_threads must be in [1,24]")


def _disk_free_mib(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free // 2**20


def preflight(config: SmoothedDifferenceConfig) -> dict[str, Any]:
    missing = [str(path) for path in (config.source_video, config.source_tiff) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if config.output_dir.exists():
        raise FileExistsError(f"Output exists: {config.output_dir}")
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if video.ndim != 3:
        raise ValueError(f"Expected T,Y,X source, got {video.shape}")
    if not (1 <= config.quiet_start_ui <= config.quiet_end_ui <= len(video)):
        raise ValueError("Quiet interval is outside the video")
    pixels = int(np.prod(video.shape, dtype=np.int64))
    output_bytes = pixels * 2 * 4
    cache_bytes = pixels * 4
    output_mib = math.ceil(output_bytes / 2**20)
    peak_disk_mib = math.ceil((output_bytes + cache_bytes) / 2**20)
    disk_free = _disk_free_mib(config.output_dir.parent)
    ram_free = _available_ram_mib()
    ready = (
        output_mib <= config.max_output_mib
        and disk_free >= peak_disk_mib + config.min_free_disk_mib
        and ram_free >= config.max_ram_mib
    )
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "ready": ready,
        "video_shape": list(video.shape),
        "video_dtype": str(video.dtype),
        "planned_outputs": 4,
        "expected_final_output_mib": output_mib,
        "expected_peak_disk_mib": peak_disk_mib,
        "smoothing": {
            "motion_correction": False,
            "spatial": f"Gaussian sigma={config.spatial_sigma_px}px",
            "temporal": f"centered Savitzky-Golay window={config.temporal_window_frames}, polyorder={config.temporal_polyorder}",
            "order": "smooth original, then signed difference",
        },
        "normalization": {
            "global": f"per-lag abs percentile {config.global_absolute_percentile}",
            "quiet_mad": "per-pixel 1.4826*MAD of quiet smoothed differences",
            "quiet_clip_z": config.quiet_clip_z,
            "quiet_deadband_z": config.quiet_deadband_z,
            "zero_code": 32768,
            "fixed_across_frames": True,
        },
        "resources": {
            "frame_chunk": config.frame_chunk,
            "cpu_threads": config.cpu_threads,
            "ram_available_mib": ram_free,
            "ram_cap_mib": config.max_ram_mib,
            "disk_free_mib": disk_free,
            "output_cap_mib": config.max_output_mib,
        },
    }
    if not ready:
        raise RuntimeError(f"Smoothed derivative preflight failed: {payload}")
    return payload


def _smooth_to_cache(video: np.ndarray, cache: np.ndarray, config: SmoothedDifferenceConfig, heartbeat) -> None:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import savgol_filter

    half = config.temporal_window_frames // 2
    for start in range(0, len(video), config.frame_chunk):
        stop = min(len(video), start + config.frame_chunk)
        load_start = max(0, start - half)
        load_stop = min(len(video), stop + half)
        block = np.asarray(video[load_start:load_stop], dtype=np.float32)
        spatial = gaussian_filter(
            block,
            sigma=(0, config.spatial_sigma_px, config.spatial_sigma_px),
            mode="reflect",
            truncate=4,
        )
        temporal = savgol_filter(
            spatial,
            window_length=config.temporal_window_frames,
            polyorder=config.temporal_polyorder,
            axis=0,
            mode="interp",
        ).astype(np.float32)
        cache[start:stop] = temporal[start - load_start : stop - load_start]
        heartbeat({"stage": "smooth", "completed_frames": stop, "total_frames": len(video)})
    cache.flush()


def _global_scales(cache: np.ndarray, config: SmoothedDifferenceConfig) -> tuple[dict[int, float], dict[int, int]]:
    samples: dict[int, list[np.ndarray]] = {lag: [] for lag in config.lags}
    counts = {lag: 0 for lag in config.lags}
    stride = config.sample_spatial_stride
    for start in range(0, len(cache), config.frame_chunk):
        stop = min(len(cache), start + config.frame_chunk)
        for lag in config.lags:
            valid_start = max(start, lag)
            if valid_start >= stop:
                continue
            difference = np.asarray(cache[valid_start:stop]) - np.asarray(cache[valid_start - lag : stop - lag])
            sample = np.abs(difference[:, ::stride, ::stride]).astype(np.float32).ravel()
            samples[lag].append(sample)
            counts[lag] += sample.size
    scales = {
        lag: max(float(np.percentile(np.concatenate(samples[lag]), config.global_absolute_percentile)), 1e-6)
        for lag in config.lags
    }
    return scales, counts


def _quiet_scales(cache: np.ndarray, config: SmoothedDifferenceConfig) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    q0, q1 = config.quiet_start_ui - 1, config.quiet_end_ui
    scales = {}
    summaries = {}
    for lag in config.lags:
        start = max(q0, lag)
        difference = np.asarray(cache[start:q1]) - np.asarray(cache[start - lag : q1 - lag])
        center = np.median(difference, axis=0)
        mad = np.median(np.abs(difference - center), axis=0) * 1.4826
        positive = mad[mad > 0]
        floor = float(np.percentile(positive, config.quiet_mad_floor_percentile)) if positive.size else 1.0
        scale = np.maximum(mad, max(floor, 1e-6)).astype(np.float32)
        scales[lag] = scale
        summaries[lag] = {
            "quiet_difference_frames": len(difference),
            "scale_floor": floor,
            "scale_median": float(np.median(scale)),
            "scale_p95": float(np.percentile(scale, 95)),
        }
    return scales, summaries


def _description(config: SmoothedDifferenceConfig, *, lag: int, normalization: str, details: dict[str, Any], source_sha256: str) -> str:
    return json.dumps({
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "transform": f"smooth(I), then D[t]=smooth(I[t])-smooth(I[t-{lag}])",
        "lag_frames": lag,
        "lag_ms": lag * config.frame_period_ms,
        "spatial_smoothing": {"method": "Gaussian", "sigma_px": config.spatial_sigma_px, "mode": "reflect"},
        "temporal_smoothing": {"method": "centered Savitzky-Golay", "window_frames": config.temporal_window_frames, "polyorder": config.temporal_polyorder},
        "motion_correction": False,
        "normalization": normalization,
        "normalization_details": details,
        "zero_code": 32768,
        "negative_direction": "below 32768",
        "positive_direction": "above 32768",
        "undefined_leading_frames": lag,
        "axes": "TYX",
        "source_video_sha256": source_sha256,
    }, sort_keys=True)


def run(config: SmoothedDifferenceConfig) -> dict[str, Any]:
    audit = preflight(config)
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    source_sha256 = _sha256(config.source_video)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    resolved = asdict(config)
    for key, value in tuple(resolved.items()):
        if isinstance(value, Path):
            resolved[key] = str(value)
    _atomic_json(config.output_dir / "config.resolved.json", resolved)
    progress = config.output_dir / "progress.jsonl"

    def heartbeat(payload: dict[str, Any]) -> None:
        from datetime import datetime, timezone
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": datetime.now(timezone.utc).isoformat()} | payload, sort_keys=True) + "\n")

    cache_path = config.output_dir / "smoothed_source.partial.npy"
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=video.shape)
    heartbeat({"stage": "smooth", "status": "started"})
    _smooth_to_cache(video, cache, config, heartbeat)
    global_scales, global_sample_counts = _global_scales(cache, config)
    quiet_scales, quiet_summaries = _quiet_scales(cache, config)
    output_maps = {}
    output_paths = {}
    for lag in config.lags:
        variants = {
            "global": {
                "global_absolute_percentile": config.global_absolute_percentile,
                "global_scale": global_scales[lag],
                "sample_spatial_stride": config.sample_spatial_stride,
                "sample_count": global_sample_counts[lag],
                "deadband": None,
            },
            "quiet_mad": {
                **quiet_summaries[lag],
                "clip_z": config.quiet_clip_z,
                "deadband_z": config.quiet_deadband_z,
                "deadband_is_visualization_only": True,
            },
        }
        for variant, details in variants.items():
            name = f"spon_ca_burst_smoothed_derivative_lag{lag}_{variant}.tif"
            temporary = config.output_dir / (name + ".partial")
            final = config.output_dir / name
            output_maps[(lag, variant)] = tifffile.memmap(
                temporary,
                shape=video.shape,
                dtype=np.uint16,
                bigtiff=True,
                photometric="minisblack",
                metadata={"axes": "TYX"},
                description=_description(config, lag=lag, normalization=variant, details=details, source_sha256=source_sha256),
            )
            output_paths[(lag, variant)] = (temporary, final)
    counters = {(lag, variant): {"below": 0, "above": 0, "neutral": lag * video.shape[1] * video.shape[2], "clipped_low": 0, "clipped_high": 0} for lag in config.lags for variant in ("global", "quiet_mad")}
    heartbeat({"stage": "encode", "status": "started"})
    for start in range(0, len(cache), config.frame_chunk):
        stop = min(len(cache), start + config.frame_chunk)
        for lag in config.lags:
            valid_start = max(start, lag)
            local = valid_start - start
            difference = None
            if valid_start < stop:
                difference = np.asarray(cache[valid_start:stop]) - np.asarray(cache[valid_start - lag : stop - lag])
            for variant in ("global", "quiet_mad"):
                encoded = np.full((stop - start, *video.shape[1:]), 32768, dtype=np.uint16)
                if difference is not None:
                    if variant == "global":
                        value = difference / global_scales[lag]
                    else:
                        value = difference / quiet_scales[lag][None]
                        value[np.abs(value) < config.quiet_deadband_z] = 0
                        value = value / config.quiet_clip_z
                    normalized = np.rint(value * 32767)
                    encoded_valid = np.clip(32768 + normalized, 1, 65535).astype(np.uint16)
                    encoded[local:] = encoded_valid
                    state = counters[(lag, variant)]
                    state["below"] += int((encoded_valid < 32768).sum())
                    state["above"] += int((encoded_valid > 32768).sum())
                    state["neutral"] += int((encoded_valid == 32768).sum())
                    state["clipped_low"] += int((encoded_valid == 1).sum())
                    state["clipped_high"] += int((encoded_valid == 65535).sum())
                output_maps[(lag, variant)][start:stop] = encoded
        heartbeat({"stage": "encode", "completed_frames": stop, "total_frames": len(video)})
    records = []
    total_pixels = int(np.prod(video.shape, dtype=np.int64))
    for key, mmap in tuple(output_maps.items()):
        lag, variant = key
        mmap.flush()
        del output_maps[key]
        temporary, final = output_paths[key]
        temporary.replace(final)
        with tifffile.TiffFile(final) as tif:
            series = tif.series[0]
            if tuple(series.shape) != tuple(video.shape) or series.dtype != np.dtype("uint16"):
                raise RuntimeError(f"TIFF validation failed: {final}")
        state = counters[key]
        records.append({
            "lag_frames": lag,
            "lag_ms": lag * config.frame_period_ms,
            "normalization": variant,
            "path": str(final),
            "bytes": final.stat().st_size,
            "shape": list(video.shape),
            "dtype": "uint16",
            "below_midpoint_fraction": state["below"] / total_pixels,
            "above_midpoint_fraction": state["above"] / total_pixels,
            "neutral_fraction": state["neutral"] / total_pixels,
            "clipped_low_fraction": state["clipped_low"] / total_pixels,
            "clipped_high_fraction": state["clipped_high"] / total_pixels,
            "global_scale": global_scales[lag] if variant == "global" else None,
            "quiet_scale_summary": quiet_summaries[lag] if variant == "quiet_mad" else None,
        })
    del cache
    cache_path.unlink()
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "preflight": audit,
        "source_video": str(config.source_video),
        "source_tiff": str(config.source_tiff),
        "source_video_sha256": source_sha256,
        "frame_alignment": "Output t equals source t; first lag frames are neutral.",
        "outputs": sorted(records, key=lambda item: (item["lag_frames"], item["normalization"])),
    }
    _atomic_json(config.output_dir / "manifest.json", payload)
    heartbeat({"stage": "complete", "status": "complete", "outputs": len(records)})
    return payload
