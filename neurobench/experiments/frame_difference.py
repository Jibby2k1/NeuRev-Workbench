"""Chunked signed frame-difference TIFF generation with fixed normalization."""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


@dataclass(frozen=True)
class FrameDifferenceConfig:
    experiment_id: str
    source_video: Path
    source_tiff: Path
    output_dir: Path
    lags: tuple[int, ...]
    frame_period_ms: float
    absolute_percentile: float
    zero_code: int
    frame_chunk: int
    cpu_threads: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int

    @classmethod
    def load(cls, path: str | Path) -> "FrameDifferenceConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        resources = raw["resources"]
        normalization = raw["normalization"]
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            source_tiff=(root / raw["source_tiff"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            lags=tuple(int(value) for value in raw.get("lags", (1, 4))),
            frame_period_ms=float(raw.get("frame_period_ms", 20.0)),
            absolute_percentile=float(normalization.get("absolute_percentile", 99.5)),
            zero_code=int(normalization.get("zero_code", 32768)),
            frame_chunk=int(resources.get("frame_chunk", 64)),
            cpu_threads=int(resources.get("cpu_threads", 6)),
            max_ram_mib=int(resources.get("max_ram_mib", 4096)),
            min_free_disk_mib=int(resources.get("min_free_disk_mib", 4096)),
            max_output_mib=int(resources.get("max_output_mib", 2048)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.lags or len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be a non-empty unique list")
        if min(self.lags) < 1 or max(self.lags) > 64:
            raise ValueError("lags must be between 1 and 64 frames")
        if not 90 <= self.absolute_percentile < 100:
            raise ValueError("absolute_percentile must be in [90, 100)")
        if self.zero_code != 32768:
            raise ValueError("zero_code must be 32768 for the signed uint16 display contract")
        if not 1 <= self.frame_chunk <= 256:
            raise ValueError("frame_chunk must be in [1, 256]")
        if not 1 <= self.cpu_threads <= 24:
            raise ValueError("cpu_threads must be in [1, 24]")
        if self.frame_period_ms <= 0:
            raise ValueError("frame_period_ms must be positive")


def _available_ram_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(config: FrameDifferenceConfig) -> dict[str, Any]:
    missing = [str(path) for path in (config.source_video, config.source_tiff) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    if config.output_dir.exists():
        raise FileExistsError(f"Output exists: {config.output_dir}")
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if video.ndim != 3:
        raise ValueError(f"Expected T,Y,X video, got {video.shape}")
    if max(config.lags) >= len(video):
        raise ValueError("A requested lag is not shorter than the video")
    expected_bytes = int(np.prod(video.shape, dtype=np.int64)) * 2 * len(config.lags)
    expected_mib = math.ceil(expected_bytes / 2**20)
    disk_probe = config.output_dir.parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    disk_free_mib = shutil.disk_usage(disk_probe).free // 2**20
    ram_available_mib = _available_ram_mib()
    ready = (
        expected_mib <= config.max_output_mib
        and disk_free_mib >= max(config.min_free_disk_mib, expected_mib * 2)
        and ram_available_mib >= config.max_ram_mib
    )
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "ready": ready,
        "video_shape": list(video.shape),
        "video_dtype": str(video.dtype),
        "lags": list(config.lags),
        "lag_ms": {str(lag): lag * config.frame_period_ms for lag in config.lags},
        "expected_output_bytes": expected_bytes,
        "expected_output_mib": expected_mib,
        "normalization": {
            "encoding": "uint16 signed display",
            "zero_code": config.zero_code,
            "absolute_percentile": config.absolute_percentile,
            "fixed_across_frames": True,
            "per_lag_scale": True,
        },
        "resources": {
            "frame_chunk": config.frame_chunk,
            "cpu_threads": config.cpu_threads,
            "ram_available_mib": ram_available_mib,
            "ram_cap_mib": config.max_ram_mib,
            "disk_free_mib": disk_free_mib,
            "output_cap_mib": config.max_output_mib,
        },
        "output_collision": False,
    }
    if not ready:
        raise RuntimeError(f"Frame-difference preflight failed: {payload}")
    return payload


def _histograms(
    video: np.ndarray,
    lags: tuple[int, ...],
    chunk: int,
) -> dict[int, dict[str, np.ndarray | int]]:
    bins = int(np.iinfo(video.dtype).max) + 1 if np.issubdtype(video.dtype, np.integer) else 65536
    bins = max(65536, bins)
    histograms = {
        lag: {
            "absolute": np.zeros(bins, dtype=np.int64),
            "positive": np.zeros(bins, dtype=np.int64),
            "negative": np.zeros(bins, dtype=np.int64),
            "count": 0,
        }
        for lag in lags
    }
    for start in range(0, len(video), chunk):
        stop = min(len(video), start + chunk)
        for lag in lags:
            valid_start = max(start, lag)
            if valid_start >= stop:
                continue
            current = np.asarray(video[valid_start:stop], dtype=np.int32)
            previous = np.asarray(video[valid_start - lag : stop - lag], dtype=np.int32)
            difference = current - previous
            absolute = np.abs(difference)
            state = histograms[lag]
            state["absolute"] += np.bincount(absolute.ravel(), minlength=bins)[:bins]
            positive = difference[difference > 0]
            negative = -difference[difference < 0]
            if positive.size:
                state["positive"] += np.bincount(positive, minlength=bins)[:bins]
            if negative.size:
                state["negative"] += np.bincount(negative, minlength=bins)[:bins]
            state["count"] = int(state["count"]) + difference.size
    return histograms


def _scale_from_histogram(histogram: np.ndarray, percentile: float) -> int:
    total = int(histogram.sum())
    if not total:
        raise ValueError("Difference histogram is empty")
    target = math.ceil(percentile / 100 * total)
    scale = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
    return max(scale, 1)


def _description(
    config: FrameDifferenceConfig,
    *,
    lag: int,
    scale: int,
    positive_clip_fraction: float,
    negative_clip_fraction: float,
    source_sha256: str,
) -> str:
    return json.dumps({
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "transform": f"D[t] = I[t] - I[t-{lag}]",
        "lag_frames": lag,
        "lag_ms": lag * config.frame_period_ms,
        "undefined_leading_frames": lag,
        "undefined_leading_frame_value": config.zero_code,
        "dtype": "uint16",
        "zero_code": config.zero_code,
        "negative_direction": "values below zero_code are negative changes",
        "positive_direction": "values above zero_code are positive changes",
        "normalization": f"clip(round({config.zero_code} + D/{scale}*32767), 1, 65535)",
        "absolute_percentile": config.absolute_percentile,
        "scale_raw_intensity_units": scale,
        "positive_clip_fraction": positive_clip_fraction,
        "negative_clip_fraction": negative_clip_fraction,
        "fixed_scale_across_frames": True,
        "source_video": str(config.source_video),
        "source_tiff": str(config.source_tiff),
        "source_video_sha256": source_sha256,
        "axes": "TYX",
    }, sort_keys=True)


def run(config: FrameDifferenceConfig) -> dict[str, Any]:
    audit = preflight(config)
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    histograms = _histograms(video, config.lags, config.frame_chunk)
    source_sha256 = _sha256(config.source_video)
    scales: dict[int, int] = {}
    statistics: dict[int, dict[str, float | int]] = {}
    for lag in config.lags:
        state = histograms[lag]
        absolute = state["absolute"]
        assert isinstance(absolute, np.ndarray)
        scale = _scale_from_histogram(absolute, config.absolute_percentile)
        scales[lag] = scale
        positive = state["positive"]
        negative = state["negative"]
        assert isinstance(positive, np.ndarray) and isinstance(negative, np.ndarray)
        count = int(state["count"])
        statistics[lag] = {
            "scale_raw_intensity_units": scale,
            "sample_count": count,
            "zero_fraction": float(absolute[0] / count),
            "positive_fraction": float(positive.sum() / count),
            "negative_fraction": float(negative.sum() / count),
            "positive_clip_fraction": float(positive[scale + 1 :].sum() / count),
            "negative_clip_fraction": float(negative[scale + 1 :].sum() / count),
        }

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

    outputs = {}
    maps = {}
    for lag in config.lags:
        stats = statistics[lag]
        name = f"spon_ca_burst_derivative_lag{lag}.tif"
        final = config.output_dir / name
        temporary = config.output_dir / (name + ".partial")
        description = _description(
            config,
            lag=lag,
            scale=scales[lag],
            positive_clip_fraction=float(stats["positive_clip_fraction"]),
            negative_clip_fraction=float(stats["negative_clip_fraction"]),
            source_sha256=source_sha256,
        )
        maps[lag] = tifffile.memmap(
            temporary,
            shape=video.shape,
            dtype=np.uint16,
            bigtiff=True,
            photometric="minisblack",
            metadata={"axes": "TYX"},
            description=description,
        )
        outputs[lag] = (temporary, final)
    heartbeat({"stage": "write", "status": "started", "frames": len(video)})
    for start in range(0, len(video), config.frame_chunk):
        stop = min(len(video), start + config.frame_chunk)
        for lag in config.lags:
            encoded = np.full((stop - start, *video.shape[1:]), config.zero_code, dtype=np.uint16)
            valid_start = max(start, lag)
            if valid_start < stop:
                current = np.asarray(video[valid_start:stop], dtype=np.int32)
                previous = np.asarray(video[valid_start - lag : stop - lag], dtype=np.int32)
                difference = current - previous
                normalized = np.rint(difference.astype(np.float32) / scales[lag] * 32767)
                encoded[valid_start - start :] = np.clip(config.zero_code + normalized, 1, 65535).astype(np.uint16)
            maps[lag][start:stop] = encoded
        heartbeat({"stage": "write", "completed_frames": stop, "total_frames": len(video)})
    records = []
    for lag in config.lags:
        maps[lag].flush()
        del maps[lag]
        temporary, final = outputs[lag]
        temporary.replace(final)
        with tifffile.TiffFile(final) as tif:
            series = tif.series[0]
            if tuple(series.shape) != tuple(video.shape) or series.dtype != np.dtype("uint16"):
                raise RuntimeError(f"TIFF validation failed for {final}: {series.shape} {series.dtype}")
        records.append({
            "lag_frames": lag,
            "lag_ms": lag * config.frame_period_ms,
            "path": str(final),
            "bytes": final.stat().st_size,
            "shape": list(video.shape),
            "dtype": "uint16",
            "statistics": statistics[lag],
        })
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "preflight": audit,
        "source_video": str(config.source_video),
        "source_tiff": str(config.source_tiff),
        "source_video_sha256": source_sha256,
        "frame_alignment": "Output t equals source t; first lag frames are neutral because the difference is undefined.",
        "outputs": records,
    }
    _atomic_json(config.output_dir / "manifest.json", payload)
    heartbeat({"stage": "write", "status": "complete", "outputs": len(records)})
    return payload
