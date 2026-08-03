"""Collision-safe, read-only-source preflight for event-weighted CS-Parzen."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.experiments.learnable_contrast import core as label_core

from .artifacts import SCIENTIFIC_STATUS, atomic_json
from .config import EventWeightedCSParzenConfig


def _sha256(path: Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _gpu_telemetry() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": type(exc).__name__}
    rows = []
    for line in completed.stdout.strip().splitlines():
        name, total, used, free = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
            }
        )
    return {"available": bool(rows), "devices": rows}


def _projection_overlay(
    movie: np.ndarray,
    labels: list[dict[str, Any]],
    config: EventWeightedCSParzenConfig,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, stop = config.source.review_interval_ui
    frame_indices = np.linspace(start - 1, stop - 1, min(24, stop - start + 1))
    frame_indices = np.unique(frame_indices.astype(int))
    projection = np.max(np.asarray(movie[frame_indices], dtype=np.float32), axis=0)
    low, high = np.percentile(projection, [1, 99.8])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.imshow(projection, cmap="gray", vmin=low, vmax=high)
    for row in labels:
        axis.scatter(
            [row["x_px"]],
            [row["y_px"]],
            s=18,
            facecolors="none",
            edgecolors=f"C{(int(row['burst_id']) - 1) % 10}",
            linewidths=0.7,
        )
    axis.set(
        title="Preflight label projection (rings are sparse known positives)",
        xlabel="x = column",
        ylabel="y = row",
    )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    figure.tight_layout()
    figure.savefig(temporary, format="png", dpi=130)
    plt.close(figure)
    temporary.replace(destination)


def preflight(
    config: EventWeightedCSParzenConfig,
    *,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Preflight destination exists: {destination}")
    if config.outputs.root_dir.exists():
        raise FileExistsError(f"Output destination exists: {config.outputs.root_dir}")
    for path in (config.source.movie_path, config.source.labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    movie = np.load(config.source.movie_path, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3:
        raise ValueError("movie must have TYX shape")
    review_start, review_stop = config.source.review_interval_ui
    if not (1 <= review_start <= review_stop <= len(movie)):
        raise ValueError("review interval is outside movie")
    labels = label_core.load_labels(config.source.labels_path)
    height, width = movie.shape[1:]
    for row in labels:
        if not (0 <= row["x_px"] < width and 0 <= row["y_px"] < height):
            raise ValueError("label coordinate lies outside movie")
        event_id = int(row["burst_id"])
        if event_id not in config.source.burst_intervals_ui:
            raise ValueError(f"label uses undeclared event {event_id}")
        interval = config.source.burst_intervals_ui[event_id]
        if (
            int(row["start_frame_ui"]),
            int(row["end_frame_ui"]),
        ) != interval:
            raise ValueError(f"event {event_id} label interval disagrees with config")

    max_unique = (
        config.sampling.confirmation_samples
        + (len(config.source.burst_intervals_ui) - 1)
        * config.sampling.event_confirmation_max_samples_per_event
    )
    block_bytes = (
        config.parzen.kernel_block_rows * max_unique * np.dtype(np.float32).itemsize * 2
    )
    try:
        import psutil

        memory = psutil.virtual_memory()
        active_python = sum(
            1
            for process in psutil.process_iter(["name"])
            if "python" in (process.info["name"] or "").lower()
        )
        memory_payload = {
            "total_gib": memory.total / 2**30,
            "available_gib": memory.available / 2**30,
            "active_python_processes": active_python,
        }
    except (ImportError, OSError):
        memory_payload = {"available": False}
    disk = shutil.disk_usage(_existing_ancestor(config.outputs.root_dir))
    config_payload = config.to_dict()
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    inputs = [
        {
            "role": "movie",
            "path": str(config.source.movie_path),
            "sha256": _sha256(config.source.movie_path),
            "shape": list(movie.shape),
            "dtype": str(movie.dtype),
        },
        {
            "role": "labels",
            "path": str(config.source.labels_path),
            "sha256": _sha256(config.source.labels_path),
            "rows": len(labels),
        },
    ]
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "ready": True,
        "scientific_status": SCIENTIFIC_STATUS,
        "source_read_only": True,
        "output_collision": False,
        "config_sha256": config_hash,
        "inputs": inputs,
        "splits": {
            str(event_id): {
                "heldout_interval_ui": list(interval),
                "guard_frames": config.sampling.heldout_guard_frames,
                "train_event_ids": sorted(
                    set(config.source.burst_intervals_ui) - {event_id}
                ),
            }
            for event_id, interval in config.source.burst_intervals_ui.items()
            if event_id in config.fold_ids
        },
        "resources": {
            "device": config.compute.device,
            "max_unique_confirmation_samples": max_unique,
            "peak_kernel_block_bytes": block_bytes,
            "resident_full_kernel_matrix": False,
            "memory": memory_payload,
            "disk_free_gib": disk.free / 2**30,
            "gpu": _gpu_telemetry(),
        },
        "authorization": {
            "full_spon_run_selected": False,
            "message": (
                "Preflight validates readiness only. Running the full Spon sweep "
                "still requires explicit user selection."
            ),
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "config.resolved.json", config_payload)
    atomic_json(destination / "preflight.json", payload)
    _projection_overlay(
        movie, labels, config, destination / "label_projection_overlay.png"
    )
    return payload
