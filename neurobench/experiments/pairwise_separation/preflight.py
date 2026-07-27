"""Read-only validation plus explicitly directed preflight artifacts."""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.algorithms.motion import estimate_integer_shift
from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib, _sha256
from neurobench.experiments.learnable_contrast import core as v1

from .config import MethodConfig, PairwiseSeparationConfig
from .sampling import uniform_anatomy_mask


def _write_projection_overlay(video: np.ndarray, labels: list[dict[str, Any]], quiet: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(quiet, dtype=np.float32).mean(axis=0)
    low, high = np.percentile(image, [1, 99.5])
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=140)
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    for row in labels:
        axis.scatter(row["x_px"], row["y_px"], s=35, facecolors="none", edgecolors="cyan", linewidths=1)
    axis.set_title("Label projection on configured quiet mean")
    axis.set_xlabel("x = column"); axis.set_ylabel("y = row")
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def preflight(config: PairwiseSeparationConfig, *, artifact_dir: str | Path) -> dict[str, Any]:
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Preflight artifact directory exists: {destination}")
    inputs = (config.source_video, config.source_tiff, config.labels_tsv, config.design_document)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    labels = v1.load_labels(config.labels_tsv)
    f, p, s, t, r = config.frames, config.preprocessing, config.sampling, config.thresholding, config.resources
    if video.ndim != 3 or f.review_end_ui > len(video):
        raise ValueError(f"Review interval outside source {video.shape}")
    if any(not (0 <= row["x_px"] < video.shape[2] and 0 <= row["y_px"] < video.shape[1]) for row in labels):
        raise ValueError("Label coordinate outside source")
    if any(not (f.review_start_ui <= frame <= f.review_end_ui) for frame in s.pairwise_diagnostic_frames_ui):
        raise ValueError("Diagnostic frame outside review interval")
    review = video[f.review_start_ui - 1:f.review_end_ui]
    quiet = review[f.quiet_start_ui - f.review_start_ui:f.quiet_end_ui - f.review_start_ui + 1]
    anatomy, anatomy_summary = uniform_anatomy_mask(quiet)
    population = (len(review) - p.lag_frames) * int(anatomy.sum())
    if s.confirm_samples > population:
        raise ValueError("confirm_samples exceeds valid sample population")
    enabled = [name for name in MethodConfig.__dataclass_fields__ if getattr(config.methods, name)["enabled"]]
    pixels = int(np.prod(review.shape, dtype=np.int64))
    continuous_methods = len(enabled)
    binary_methods = len(enabled)
    review_tiff_bytes = continuous_methods * 4 if t.write_binary_tiff else 0
    output_bytes = pixels * (
        continuous_methods * 4
        + binary_methods
        + (binary_methods if t.write_binary_tiff else 0)
        + review_tiff_bytes
    )
    output_mib = math.ceil(output_bytes / 2**20)
    probe = config.output_dir.parent
    while not probe.exists(): probe = probe.parent
    disk_free = shutil.disk_usage(probe).free // 2**20
    ram_available = _available_ram_mib()
    coarse_angles = math.ceil(90 / s.screen_angle_step_degrees)
    refine_angles = math.floor(2 * s.refine_half_width_degrees / s.refine_angle_step_degrees) + 1
    kernel_work = (coarse_angles + refine_angles) * s.screen_samples**2 + 3 * s.confirm_samples**2
    reference = np.median(np.asarray(quiet, dtype=np.float32), axis=0)
    motion_rows = []
    for frame_ui in s.pairwise_diagnostic_frames_ui:
        frame = np.asarray(video[frame_ui - 1], dtype=np.float32)
        shift = estimate_integer_shift(reference[::4, ::4], frame[::4, ::4], max_shift_px=p.max_shift_px)
        motion_rows.append({"frame_ui": frame_ui, "dy_downsampled_px": shift["dy"], "dx_downsampled_px": shift["dx"], "score": shift["score"]})
    ready = bool(not config.output_dir.exists() and output_mib <= r.max_output_mib and
                 disk_free >= r.min_free_disk_mib + output_mib and ram_available >= r.max_ram_mib)
    payload = {
        "schema_version": 1, "experiment_id": config.experiment_id, "ready": ready,
        "source_shape": list(video.shape), "source_dtype": str(video.dtype), "axes": "TYX",
        "frames": {"review_ui_inclusive": [f.review_start_ui, f.review_end_ui],
                   "review_zero_half_open": [f.review_start_ui - 1, f.review_end_ui],
                   "quiet_ui_inclusive": [f.quiet_start_ui, f.quiet_end_ui],
                   "undefined_leading_derivative_frames": p.lag_frames},
        "labels": {"rows": len(labels), "coordinates": "x=column,y=row"},
        "sampling": {**anatomy_summary, "valid_pair_pixel_population": population,
                     "screen_samples": s.screen_samples, "confirm_samples": s.confirm_samples},
        "methods": enabled, "threshold_count": len(t.z_thresholds),
        "kernel_work": {"coarse_angles": coarse_angles, "refine_angles": refine_angles,
                        "estimated_pairwise_kernel_elements": kernel_work,
                        "block_rows": r.kernel_block_rows,
                        "peak_kernel_block_bytes": r.kernel_block_rows * s.confirm_samples * 8 * 2},
        "motion_diagnostic": {"motion_correction": False, "sampled_integer_shifts": motion_rows},
        "resources": {"cpu_threads": r.cpu_threads, "estimated_output_bytes": output_bytes,
                      "estimated_output_mib": output_mib, "output_cap_mib": r.max_output_mib,
                      "review_tiffs_included": bool(t.write_binary_tiff),
                      "ram_available_mib": ram_available, "ram_cap_mib": r.max_ram_mib,
                      "disk_free_mib": disk_free, "minimum_free_disk_mib": r.min_free_disk_mib},
        "output_collision": config.output_dir.exists(),
        "permissions": {"synthetic_smoke_ready": True, "cpu_run_ready": ready,
                        "full_spon_run_requires_explicit_user_selection": True},
        "interpretation": "Unmatched candidates remain unknown; known-label candidate fraction is a lower bound only, not precision.",
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in inputs],
    }
    destination.mkdir(parents=True, exist_ok=False)
    overlay = destination / "label_projection_overlay.png"
    _write_projection_overlay(video, labels, quiet, overlay)
    payload["label_projection_overlay"] = str(overlay)
    _atomic_json(destination / "preflight.json", payload)
    _atomic_json(destination / "config.resolved.json", config.to_dict())
    if not ready:
        raise RuntimeError(f"Pairwise preflight did not authorize a CPU run: {payload}")
    return payload
