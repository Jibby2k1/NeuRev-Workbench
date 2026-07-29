"""Read-only validation and collision-safe artifacts for representation runs."""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib, _sha256
from neurobench.experiments.learnable_contrast import core as v1

from .config import RepresentationBenchmarkConfig


def _overlay(quiet: np.ndarray, labels: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(quiet, dtype=np.float32).mean(axis=0)
    low, high = np.percentile(image, [1, 99.5])
    figure, axis = plt.subplots(figsize=(10, 6), dpi=140)
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    axis.scatter(
        [row["x_px"] for row in labels], [row["y_px"] for row in labels],
        s=34, facecolors="none", edgecolors="cyan", linewidths=1,
    )
    axis.set(title="Representation benchmark label projection", xlabel="x = column", ylabel="y = row")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _gpu_status(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {"requested": device, "available": True, "name": "cpu"}
    try:
        import torch
        available = torch.cuda.is_available()
        if not available:
            return {"requested": device, "available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "requested": device, "available": True,
            "name": torch.cuda.get_device_name(0),
            "free_mib": int(free // 2**20), "total_mib": int(total // 2**20),
        }
    except Exception as exc:
        return {"requested": device, "available": False, "error": repr(exc)}


def preflight(config: RepresentationBenchmarkConfig, *, artifact_dir: str | Path) -> dict[str, Any]:
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Preflight artifact directory exists: {destination}")
    missing = [str(path) for path in (config.source_video, config.labels_tsv) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if config.output_dir.exists():
        raise FileExistsError(f"Output root exists: {config.output_dir}")
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = v1.load_labels(config.labels_tsv)
    f, r = config.frames, config.resources
    if video.ndim != 3 or f.review_end_ui > len(video):
        raise ValueError(f"Review interval outside TYX source {video.shape}")
    if any(not (0 <= row["x_px"] < video.shape[2] and 0 <= row["y_px"] < video.shape[1]) for row in labels):
        raise ValueError("Label coordinate outside source")
    review = video[f.review_start_ui - 1:f.review_end_ui]
    quiet = video[f.quiet_start_ui - 1:f.quiet_end_ui]
    frames, height, width = review.shape
    pixels = height * width
    if config.ica.fit_sample_pixels > pixels:
        raise ValueError("ICA fit sample exceeds available pixels")
    a = config.autoencoder
    if a.train_pixels + a.validation_pixels > pixels:
        raise ValueError("Autoencoder train and validation samples exceed available pixels")
    pca_count = len(config.pca.inputs) * len(config.pca.ranks)
    ica_count = len(config.ica.inputs) * len(config.ica.ranks) * len(config.ica.seeds)
    auto_count = (
        len(a.inputs) * len(a.kinds) * len(a.ranks) * len(a.seeds) if a.enabled else 0
    )
    combination_count = pca_count + ica_count + auto_count
    dense_stack_bytes = frames * pixels * 4
    component_bytes = pixels * max(config.pca.ranks) * 4
    tiff_count = 5 if config.evaluation.write_representative_tiffs else 0
    estimated_output_bytes = component_bytes * 2 + dense_stack_bytes // 2 + tiff_count * frames * pixels * 2
    estimated_output_mib = math.ceil(estimated_output_bytes / 2**20)
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    disk_free_mib = shutil.disk_usage(probe).free // 2**20
    ram_available_mib = _available_ram_mib()
    gpu = _gpu_status(r.device)
    gpu_ready = bool(
        gpu.get("available")
        and (r.device != "cuda" or int(gpu.get("free_mib", 0)) >= r.gpu_reserve_mib + 2048)
    )
    ready = bool(
        gpu_ready
        and ram_available_mib >= r.max_ram_mib
        and disk_free_mib >= r.min_free_disk_mib + estimated_output_mib
        and estimated_output_mib <= r.max_output_mib
    )
    payload = {
        "schema_version": 1, "experiment_id": config.experiment_id, "ready": ready,
        "source": {"shape": list(video.shape), "dtype": str(video.dtype), "axes": "TYX"},
        "review": {
            "ui_inclusive": [f.review_start_ui, f.review_end_ui],
            "zero_half_open": [f.review_start_ui - 1, f.review_end_ui],
            "shape": list(review.shape), "pixels": pixels,
        },
        "quiet": {"ui_inclusive": [f.quiet_start_ui, f.quiet_end_ui], "frames": len(quiet)},
        "labels": {"rows": len(labels), "coordinates": "x=column,y=row"},
        "combinations": {
            "pca": pca_count, "spatial_fastica": ica_count,
            "autoencoder": auto_count, "total_fits": combination_count,
            "umap_optional_postfit": config.umap.enabled_if_available,
        },
        "resources": {
            "device": r.device, "gpu": gpu, "cpu_threads": r.cpu_threads,
            "projection_chunk_pixels": r.projection_chunk_pixels,
            "ram_available_mib": ram_available_mib, "ram_guard_mib": r.max_ram_mib,
            "disk_free_mib": disk_free_mib, "minimum_free_disk_mib": r.min_free_disk_mib,
            "estimated_output_mib": estimated_output_mib, "output_cap_mib": r.max_output_mib,
        },
        "output_collision": False,
        "scientific_contract": {
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "pca_centering": "explicit preprocessing only; factorization is uncentered SVD",
            "ica_orientation": "spatial ICA: pixels are observations and frames are features",
            "umap_role": "optional visualization, never source-separation evidence",
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (config.source_video, config.labels_tsv)
        ],
    }
    destination.mkdir(parents=True, exist_ok=False)
    _overlay(quiet, labels, destination / "label_projection_overlay.png")
    _atomic_json(destination / "preflight.json", payload)
    _atomic_json(destination / "config.resolved.json", config.to_dict())
    if not ready:
        raise RuntimeError(f"Representation preflight did not authorize the run: {payload}")
    return payload
