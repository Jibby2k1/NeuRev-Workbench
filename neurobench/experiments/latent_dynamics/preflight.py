"""Read-only validation and explicit artifacts for latent-dynamics runs."""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib, _sha256
from neurobench.experiments.learnable_contrast import core as v1

from .config import LatentDynamicsConfig


def _write_projection_overlay(video: np.ndarray, labels: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image = np.asarray(video, dtype=np.float32).mean(axis=0)
    low, high = np.percentile(image, [1, 99.5])
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=140)
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    for row in labels:
        axis.scatter(row["x_px"], row["y_px"], s=35, facecolors="none", edgecolors="cyan", linewidths=1)
    axis.set(title="Labels projected on review mean", xlabel="x = column", ylabel="y = row")
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def preflight(config: LatentDynamicsConfig, *, artifact_dir: str | Path) -> dict[str, Any]:
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Preflight artifact directory exists: {destination}")
    missing = [str(path) for path in (config.source_video, config.labels_tsv) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    collision = config.output_dir.exists()
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if video.ndim != 3:
        raise ValueError(f"source_video must have TYX axes, got {video.shape}")
    f = config.frames
    if f.review_end_ui > len(video):
        raise ValueError("Review interval extends past source video")
    labels = v1.load_labels(config.labels_tsv)
    if any(not (0 <= row["x_px"] < video.shape[2] and 0 <= row["y_px"] < video.shape[1]) for row in labels):
        raise ValueError("Label coordinate outside source")
    review = video[f.review_start_ui - 1:f.review_end_ui]
    quiet_frames = f.quiet_end_ui - f.quiet_start_ui + 1
    pixel_count = video.shape[1] * video.shape[2]
    if config.fit.sample_pixels > pixel_count:
        raise ValueError("fit.sample_pixels exceeds the image pixel count")
    state_count = int(config.application.write_filter_mean) + int(config.application.write_smoother_mean)
    state_bytes_each = review.size * np.dtype(np.float32).itemsize
    center_scale_bytes = 2 * pixel_count * np.dtype(np.float32).itemsize
    variance_bytes = 2 * len(review) * np.dtype(np.float32).itemsize
    selected_tiff_bytes = state_bytes_each * state_count if config.features.write_selected_tiffs else 0
    pooled_map_count = 3 + len(config.features.lags) + state_count * (2 + len(config.features.lags))
    pooled_map_variants = 9  # full review plus four quiet and four burst score maps
    pooled_map_bytes = pooled_map_count * pooled_map_variants * pixel_count * np.dtype(np.float32).itemsize
    dense_artifacts = {
        "state_arrays_mib": math.ceil(state_bytes_each * state_count / 2**20),
        "selected_review_tiffs_mib": math.ceil(selected_tiff_bytes / 2**20),
        "quiet_center_scale_mib": math.ceil(center_scale_bytes / 2**20),
        "variance_by_time_mib": math.ceil(variance_bytes / 2**20),
        "pooled_candidate_maps_uncompressed_mib": math.ceil(pooled_map_bytes / 2**20),
    }
    output_bytes = state_bytes_each * state_count + selected_tiff_bytes + center_scale_bytes + variance_bytes + pooled_map_bytes
    output_mib = math.ceil(output_bytes / 2**20)
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    disk_free_mib = shutil.disk_usage(probe).free // 2**20
    ram_available_mib = _available_ram_mib()
    res = config.resources
    ready = bool(not collision and output_mib <= res.max_output_mib
                 and disk_free_mib >= res.min_free_disk_mib + output_mib
                 and ram_available_mib >= res.max_ram_mib)
    payload: dict[str, Any] = {
        "schema_version": 1, "experiment_id": config.experiment_id, "ready": ready,
        "source_shape": list(video.shape), "source_dtype": str(video.dtype), "axes": "TYX",
        "frames": {"review_ui_inclusive": [f.review_start_ui, f.review_end_ui],
                   "review_zero_half_open": [f.review_start_ui - 1, f.review_end_ui],
                   "quiet_ui_inclusive": [f.quiet_start_ui, f.quiet_end_ui],
                   "quiet_frame_count": quiet_frames},
        "labels": {"rows": len(labels), "coordinates": "x=column,y=row"},
        "fit": {"sample_pixels": config.fit.sample_pixels,
                "candidate_models": len(config.fit.decay_time_ms_grid) * len(config.fit.process_to_observation_grid),
                "labels_available_to_fit": False},
        "resources": {"cpu_threads": res.cpu_threads, "estimated_output_mib": output_mib,
                      "output_cap_mib": res.max_output_mib, "ram_available_mib": ram_available_mib,
                      "ram_cap_mib": res.max_ram_mib, "disk_free_mib": disk_free_mib,
                      "minimum_free_disk_mib": res.min_free_disk_mib,
                      "dense_artifacts": dense_artifacts},
        "output_collision": collision,
        "permissions": {"synthetic_smoke_ready": True, "cpu_run_ready": ready,
                        "full_spon_run_requires_explicit_user_selection": True},
        "interpretation": "Unlabeled event pixels remain unknown, never assumed negative.",
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                   for path in (config.source_video, config.labels_tsv)],
    }
    destination.mkdir(parents=True, exist_ok=False)
    overlay = destination / "label_projection_overlay.png"
    _write_projection_overlay(review, labels, overlay)
    payload["label_projection_overlay"] = str(overlay)
    _atomic_json(destination / "preflight.json", payload)
    _atomic_json(destination / "config.resolved.json", config.to_dict())
    if not ready:
        raise RuntimeError(f"Latent-dynamics preflight did not authorize a run: {payload}")
    return payload
