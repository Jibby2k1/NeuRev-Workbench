"""Collision-safe, read-only-source MSLN/MS-ICA preflight."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.algorithms.multiscale_subspace import fit_per_context_ica
from neurobench.experiments.learnable_contrast import core as label_core

from .artifacts import atomic_json, sha256_file, sha256_payload
from .config import MSLNMSICAConfig
from .context_bank import evaluate_context, ordered_contexts


def _gpu() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=8,
        )
        rows = []
        for line in completed.stdout.strip().splitlines():
            name, total, free = [part.strip() for part in line.split(",")]
            rows.append({"name": name, "total_mib": int(total), "free_mib": int(free)})
        return {"available": bool(rows), "devices": rows}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "reason": type(exc).__name__}


def _fingerprint_directory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path), "exists": True, "file_count": len(files),
        "sha256": sha256_payload([(str(item.relative_to(path)), sha256_file(item)) for item in files]),
    }


def _overlay(movie: np.ndarray, labels: list[dict[str, Any]], config: MSLNMSICAConfig, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    start, stop = config.source.review_interval_ui
    indices = np.unique(np.linspace(start - 1, stop - 1, min(16, stop - start + 1)).astype(int))
    projection = np.max(np.asarray(movie[indices], dtype=np.float32), axis=0)
    low, high = np.percentile(projection, [1, 99.8])
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.imshow(projection, cmap="gray", vmin=low, vmax=high)
    for row in labels:
        axis.scatter(row["x_px"], row["y_px"], s=18, facecolors="none", edgecolors="tab:red")
    axis.set(title="Sparse known-positive label projection", xlabel="x = column", ylabel="y = row")
    fig.tight_layout()
    temp = path.with_suffix(".partial.png")
    fig.savefig(temp, dpi=120)
    plt.close(fig)
    temp.replace(path)


def _tiny_smoke(config: MSLNMSICAConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.sampling.seed)
    frames = max(40, max(config.contexts.temporal.windows_frames, default=3) + 4)
    video = rng.normal(0, 0.4, size=(frames, 19, 19)).astype(np.float32)
    video[frames // 2, 8:11, 8:11] += 5
    quiet = np.arange(frames) < frames // 3
    rows = []
    first_values = None
    for definition in ordered_contexts(config):
        result = evaluate_context(video, definition, quiet_mask=quiet)
        rows.append({"context_id": definition.context_id, "finite": bool(np.isfinite(result.values).all()), "valid_frames": int(result.valid_frames.sum())})
        if first_values is None:
            first_values = result.values
    flat = first_values.reshape(frames, -1)  # type: ignore[union-attr]
    pairs = np.column_stack([flat[:-1].ravel(), flat[1:].ravel()])
    limit = min(len(pairs), 192)
    fit = fit_per_context_ica(ordered_contexts(config)[0].context_id, pairs[:limit], pairs[-limit:], parzen_bandwidth=config.per_context_ica.parzen_bandwidth, kernel_block_rows=min(config.per_context_ica.kernel_block_rows, 64), coarse_step_degrees=config.per_context_ica.coarse_step_degrees, refine_half_width_degrees=config.per_context_ica.refine_half_width_degrees, refine_step_degrees=config.per_context_ica.refine_step_degrees)
    return {"contexts": rows, "ica_finite": bool(np.isfinite(fit.demixing).all()), "ica_objective": fit.objective_value}


def preflight(config: MSLNMSICAConfig) -> dict[str, Any]:
    root = config.outputs.root_dir
    if root.exists():
        raise FileExistsError(f"Output root already exists: {root}")
    for path in (config.source.movie_path, config.source.labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.source.baseline_evidence_dir is not None and not config.source.baseline_evidence_dir.is_dir():
        raise FileNotFoundError(config.source.baseline_evidence_dir)
    movie = np.load(config.source.movie_path, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3 or not np.issubdtype(movie.dtype, np.number):
        raise ValueError("movie must be a numeric TYX .npy array")
    start, stop = config.source.review_interval_ui
    if not 1 <= start <= stop <= len(movie):
        raise ValueError("review interval lies outside the movie")
    quiet_start, quiet_stop = config.source.quiet_interval_ui
    if not start <= quiet_start <= quiet_stop <= stop:
        raise ValueError("quiet interval must lie inside the review interval")
    labels = label_core.load_labels(config.source.labels_path)
    height, width = movie.shape[1:]
    for row in labels:
        if not (0 <= int(row["x_px"]) < width and 0 <= int(row["y_px"]) < height):
            raise ValueError("label coordinate outside movie")
    for frame_start in range(start - 1, stop, max(1, config.compute.frame_chunk)):
        frame_stop = min(frame_start + config.compute.frame_chunk, stop)
        if not np.isfinite(np.asarray(movie[frame_start:frame_stop], dtype=np.float32)).all():
            raise ValueError("source review interval contains non-finite values")
    context_count = len(ordered_contexts(config))
    review_frames = stop - start + 1
    map_bytes = review_frames * height * width * np.dtype(np.float32).itemsize
    scalar_route_count = len(set(config.routing.modes) - {"none"})
    persisted_map_equivalents = (
        context_count * 6
        + (context_count if config.outputs.save_all_latent_maps else 0)
        + len(config.cross_context.groups)
        + scalar_route_count
        + 3
    )
    selected_bytes = map_bytes * persisted_map_equivalents
    peak_bytes = int(map_bytes * 16 + config.compute.frame_chunk * height * width * 8)
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory = {"total_gib": vm.total / 2**30, "available_gib": vm.available / 2**30, "active_python_processes": sum("python" in (p.info["name"] or "").lower() for p in psutil.process_iter(["name"]))}
    except (ImportError, OSError):
        memory = {"available": False}
    ancestor = root.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    disk = shutil.disk_usage(ancestor)
    resolved = config.to_dict()
    input_fingerprints = {
        "movie": {"path": str(config.source.movie_path), "sha256": sha256_file(config.source.movie_path), "shape": list(movie.shape), "dtype": str(movie.dtype)},
        "labels": {"path": str(config.source.labels_path), "sha256": sha256_file(config.source.labels_path), "rows": len(labels)},
        "baseline": None if config.source.baseline_evidence_dir is None else _fingerprint_directory(config.source.baseline_evidence_dir),
    }
    fingerprint = sha256_payload({"config": resolved, "inputs": input_fingerprints})
    resource_plan = {
        "context_count": context_count, "bytes_per_context_map": map_bytes,
        "persisted_map_equivalents": persisted_map_equivalents,
        "selected_output_bytes_estimate": selected_bytes, "peak_ram_bytes_estimate": peak_bytes,
        "ram_cap_bytes": int(config.compute.max_peak_ram_gb * 2**30), "vram_cap_bytes": int(config.compute.max_peak_vram_gb * 2**30),
        "disk_free_bytes": disk.free, "memory": memory, "gpu": _gpu(),
        "one_context_at_a_time": True, "context_batch": config.compute.context_batch,
    }
    if peak_bytes > resource_plan["ram_cap_bytes"] or selected_bytes > disk.free:
        raise RuntimeError("resource estimate exceeds configured RAM or available disk")
    smoke = _tiny_smoke(config)
    payload = {
        "schema_version": 1, "experiment_id": config.experiment_id, "ready": True,
        "preflight_fingerprint": fingerprint, "source_read_only": True,
        "labels_used_for_fitting": False, "input_fingerprints": input_fingerprints,
        "tiny_smoke": smoke, "authorization": {"full_spon_run_selected": False, "preflight_is_not_run_authorization": True},
    }
    root.mkdir(parents=True, exist_ok=False)
    import yaml
    atomic_json(root / "config.input.json", yaml.safe_load(config.config_path.read_text(encoding="utf-8")))
    atomic_json(root / "config.resolved.json", resolved)
    atomic_json(root / "resource_plan.json", resource_plan)
    atomic_json(root / "preflight.json", payload)
    atomic_json(root / "run_manifest.partial.json", {"status": "preflight_ready", "preflight_fingerprint": fingerprint})
    atomic_json(root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    _overlay(movie, labels, config, root / "label_projection_overlay.png")
    return payload


def matching_preflight(config: MSLNMSICAConfig) -> dict[str, Any]:
    path = config.outputs.root_dir / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError("matching preflight.json is required")
    stored = json.loads(path.read_text(encoding="utf-8"))
    resolved = config.to_dict()
    fingerprints = stored.get("input_fingerprints")
    expected = sha256_payload({"config": resolved, "inputs": fingerprints})
    if not stored.get("ready") or stored.get("preflight_fingerprint") != expected:
        raise RuntimeError("preflight does not exactly match this resolved config")
    # Recheck source hashes so preflight cannot authorize changed inputs.
    if fingerprints["movie"]["sha256"] != sha256_file(config.source.movie_path) or fingerprints["labels"]["sha256"] != sha256_file(config.source.labels_path):
        raise RuntimeError("source changed after preflight")
    if config.source.baseline_evidence_dir is not None and fingerprints["baseline"] != _fingerprint_directory(config.source.baseline_evidence_dir):
        raise RuntimeError("baseline evidence changed after preflight")
    return stored
