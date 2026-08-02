"""Stream complete framewise patch-information maps to NPY and TIFF."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np

from neurobench.algorithms.patch_information import (
    information_fields_tensor,
    local_histogram_tensor,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
    InnovationRankerConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.patch_information_config import (
    PatchInformationConfig,
)


_FAMILY_FIELDS = {
    "renyi2_ip": "renyi2_information_potential",
    "cs_quiet": "cs_quiet_divergence",
    "correntropy": "local_correntropy",
}


def parse_feature_id(feature_id: str) -> tuple[str, int, float]:
    """Parse a frozen ID such as ``cs_quiet__p7__bw0p5``."""
    parts = str(feature_id).split("__")
    if len(parts) != 3 or parts[0] not in _FAMILY_FIELDS:
        raise ValueError(f"unsupported patch-information feature id: {feature_id}")
    try:
        patch = int(parts[1].removeprefix("p"))
        bandwidth = float(parts[2].removeprefix("bw").replace("p", "."))
    except ValueError as exc:
        raise ValueError(
            f"unsupported patch-information feature id: {feature_id}"
        ) from exc
    if parts[1] != f"p{patch}" or parts[2] != f"bw{bandwidth:g}".replace(".", "p"):
        raise ValueError(f"non-canonical patch-information feature id: {feature_id}")
    return _FAMILY_FIELDS[parts[0]], patch, bandwidth


def display_limits(
    values: np.ndarray,
    *,
    quiet_count: int,
    upper_percentile: float = 99.8,
    stride: int = 4,
) -> tuple[float, float]:
    """Return deterministic global black/white points for a stable video."""
    array = np.asarray(values)
    if array.ndim != 3 or not 1 <= int(quiet_count) <= len(array):
        raise ValueError("values must be TYX with a valid quiet_count")
    if not 50.0 < float(upper_percentile) < 100.0 or int(stride) < 1:
        raise ValueError("invalid display percentile or stride")
    quiet = np.asarray(
        array[:quiet_count:stride, ::stride, ::stride], dtype=np.float32
    )
    sampled = np.asarray(array[::stride, ::stride, ::stride], dtype=np.float32)
    black = float(np.percentile(quiet, 50.0))
    white = float(np.percentile(sampled, float(upper_percentile)))
    if not np.isfinite(black) or not np.isfinite(white) or white <= black:
        raise ValueError("feature video has degenerate display limits")
    return black, white


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_display_tiff(
    source: np.ndarray,
    path: Path,
    *,
    black: float,
    white: float,
    compression: str,
    description: dict[str, Any],
) -> None:
    import tifffile

    scale = max(float(white) - float(black), 1e-8)
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for index in range(len(source)):
            frame = np.asarray(source[index], dtype=np.float32)
            display = np.clip((frame - black) / scale, 0.0, 1.0)
            writer.write(
                np.rint(display * 65535.0).astype(np.uint16),
                compression=compression,
                photometric="minisblack",
                metadata=None,
                description=(json.dumps(description, sort_keys=True) if index == 0 else None),
            )


def _benchmark_frame(
    carrier: np.ndarray,
    quiet_histogram,
    *,
    centers: tuple[float, ...],
    family: str,
    patch: int,
    bandwidth: float,
    device,
    repeats: int = 30,
) -> dict[str, float]:
    import torch

    index = min(len(carrier) - 1, 100)

    def evaluate() -> None:
        frame = torch.as_tensor(
            np.asarray(carrier[index:index + 1], dtype=np.float32), device=device
        )
        histogram = local_histogram_tensor(
            frame, centers=centers, patch_size_px=patch
        )
        information_fields_tensor(
            histogram,
            frame,
            quiet_histogram,
            centers=centers,
            bandwidth=bandwidth,
        )[family]

    with torch.inference_mode():
        for _ in range(5):
            evaluate()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations = []
        for _ in range(int(repeats)):
            started = time.perf_counter()
            evaluate()
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "single_frame_median_ms": float(np.median(durations)),
        "single_frame_p95_ms": float(np.percentile(durations, 95.0)),
        "frame_period_ms": 20.0,
        "meets_20ms_compute_only_p95": bool(np.percentile(durations, 95.0) <= 20.0),
    }


def generate(
    config: PatchInformationConfig,
    *,
    feature_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate every available review-frame map for one frozen ITL feature."""
    import tifffile
    import torch

    family, patch, bandwidth = parse_feature_id(feature_id)
    if patch not in [int(value) for value in config.itl["patch_sizes_px"]]:
        raise ValueError(f"patch size {patch} is outside the frozen experiment")
    if bandwidth not in [float(value) for value in config.itl["kernel_bandwidths_z"]]:
        raise ValueError(f"bandwidth {bandwidth} is outside the frozen experiment")
    destination = Path(output_dir).resolve()
    partial = Path(str(destination) + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("completed or partial feature-video output already exists")
    metrics = json.loads(
        (config.source_ranker_root / "metrics.json").read_text(encoding="utf-8")
    )
    if metrics.get("status") != "completed":
        raise RuntimeError("source ranker is not completed")
    ranker = InnovationRankerConfig.load(config.source_ranker_config)
    carrier_path = ranker.feature_root / "features" / "carrier_signed.npy"
    carrier = np.load(carrier_path, mmap_mode="r", allow_pickle=False)
    expected = (
        int(ranker.frames["review_end_ui"])
        - int(ranker.frames["review_start_ui"]) + 1
    )
    if carrier.ndim != 3 or len(carrier) != expected:
        raise ValueError("carrier does not match the frozen review interval")
    device = torch.device(str(config.resources["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    raw_bytes = int(np.prod(carrier.shape)) * np.dtype(np.float16).itemsize
    estimated_bytes = raw_bytes * 3
    probe = destination.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < estimated_bytes + int(config.resources["min_free_disk_mib"]) * 2**20:
        raise RuntimeError("insufficient disk headroom for feature video")
    if estimated_bytes > int(config.resources["max_output_mib"]) * 2**20:
        raise RuntimeError("estimated feature video exceeds configured output cap")

    partial.mkdir(parents=True)
    started = time.time()
    centers = tuple(float(value) for value in config.itl["bin_centers_z"])
    quiet_count = int(ranker.frames["quiet_count"])
    batch_size = int(config.itl["frame_batch_size"])
    height, width = carrier.shape[1:]
    quiet_histogram = torch.zeros(
        (len(centers), height, width), dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        for start in range(0, quiet_count, batch_size):
            stop = min(quiet_count, start + batch_size)
            frames = torch.as_tensor(
                np.asarray(carrier[start:stop], dtype=np.float32), device=device
            )
            quiet_histogram += local_histogram_tensor(
                frames, centers=centers, patch_size_px=patch
            ).sum(dim=0)
        quiet_histogram /= float(quiet_count)

    benchmark = _benchmark_frame(
        carrier,
        quiet_histogram,
        centers=centers,
        family=family,
        patch=patch,
        bandwidth=bandwidth,
        device=device,
    )
    start_ui = int(ranker.frames["review_start_ui"])
    end_ui = int(ranker.frames["review_end_ui"])
    frame_label = f"frames{start_ui}-{end_ui}"
    raw_path = partial / f"{feature_id}__{frame_label}.npy"
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.float16, shape=carrier.shape
    )
    compute_seconds = 0.0
    with torch.inference_mode():
        for start in range(0, len(carrier), batch_size):
            stop = min(len(carrier), start + batch_size)
            batch_started = time.perf_counter()
            frames = torch.as_tensor(
                np.asarray(carrier[start:stop], dtype=np.float32), device=device
            )
            histogram = local_histogram_tensor(
                frames, centers=centers, patch_size_px=patch
            )
            values = information_fields_tensor(
                histogram,
                frames,
                quiet_histogram,
                centers=centers,
                bandwidth=bandwidth,
            )[family]
            raw[start:stop] = values.to("cpu").numpy().astype(np.float16)
            compute_seconds += time.perf_counter() - batch_started
    raw.flush()
    if not np.isfinite(np.asarray(raw[::16, ::8, ::8], dtype=np.float32)).all():
        raise RuntimeError("non-finite values found in feature video")
    black, white = display_limits(raw, quiet_count=quiet_count)
    tiff_path = partial / f"{feature_id}__{frame_label}__global.tif"
    description = {
        "feature_id": feature_id,
        "source": str(carrier_path),
        "frame_count": len(carrier),
        "source_ui_frames_inclusive": [
            start_ui,
            end_ui,
        ],
        "quiet_ui_frames_inclusive": [
            start_ui,
            start_ui + quiet_count - 1,
        ],
        "online_causal_start_ui": start_ui + quiet_count,
        "display": {
            "kind": "global_linear_uint16",
            "black": black,
            "white": white,
            "upper_percentile": 99.8,
        },
    }
    _write_display_tiff(
        raw,
        tiff_path,
        black=black,
        white=white,
        compression=str(config.visualization["compression"]),
        description=description,
    )
    with tifffile.TiffFile(tiff_path) as tiff:
        if len(tiff.pages) != len(carrier) or tiff.pages[0].shape != carrier.shape[1:]:
            raise RuntimeError("written TIFF failed frame-count or shape validation")
    elapsed = time.time() - started
    manifest = {
        "schema_version": 1,
        "status": "completed",
        **description,
        "raw_npy": raw_path.name,
        "display_tiff": tiff_path.name,
        "raw_dtype": "float16",
        "shape": list(carrier.shape),
        "causal_status": "causal_after_frozen_quiet_calibration",
        "benchmark": benchmark,
        "batch_size": batch_size,
        "compute_seconds": compute_seconds,
        "compute_frames_per_second": len(carrier) / compute_seconds,
        "elapsed_seconds": elapsed,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    _atomic_json(partial / "manifest.json", manifest)
    _atomic_json(
        partial / "run_state.json",
        {
            "status": "completed",
            "feature_id": feature_id,
            "frame_count": len(carrier),
            "elapsed_seconds": elapsed,
        },
    )
    del raw, quiet_histogram
    if device.type == "cuda":
        torch.cuda.empty_cache()
    partial.replace(destination)
    manifest["output_dir"] = str(destination)
    return manifest
