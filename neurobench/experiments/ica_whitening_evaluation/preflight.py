"""Collision-safe, read-only preflight for the factorial program."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib
from neurobench.experiments.learned_operator_selection.data import (
    load_and_validate_labels,
)
from neurobench.portable_paths import data_root, portable_path

from .config import ICAWhiteningConfig
from .design import build_design, design_digest, enumerate_cells


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_status() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "free_mib": int(free // 2**20),
            "total_mib": int(total // 2**20),
        }
    except Exception as error:  # pragma: no cover - host dependent
        return {"available": False, "error": repr(error)}


def _active_processes() -> list[dict[str, Any]]:
    try:
        import psutil
        rows = []
        current = os.getpid()
        for process in psutil.process_iter(("pid", "name", "cmdline")):
            if process.info["pid"] == current:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if any(token in command.lower() for token in ("neurobench", "python", "cuda")):
                rows.append({
                    "pid": process.info["pid"],
                    "name": process.info.get("name") or "",
                    "command": command[:240],
                })
        return rows[:32]
    except Exception as error:  # pragma: no cover - host dependent
        return [{"inspection_error": repr(error)}]


def _write_design(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _projection_overlay(
    quiet: np.ndarray, labels: list[dict[str, Any]], destination: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(quiet, dtype=np.float32).mean(axis=0)
    low, high = np.percentile(image, [1, 99.5])
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=140)
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    axis.scatter(
        [row["x_px"] for row in labels], [row["y_px"] for row in labels],
        s=32, facecolors="none", edgecolors="cyan", linewidths=1,
    )
    axis.set(
        title="ICA/whitening evaluation label projection",
        xlabel="x = column", ylabel="y = row",
    )
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)


def preflight(
    config: ICAWhiteningConfig, *, artifact_dir: str | Path
) -> dict[str, Any]:
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"preflight directory exists: {destination}")
    missing = [
        str(path) for path in (config.source_video, config.labels_tsv, config.label_summary)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"required inputs are missing: {missing}")
    if config.output_dir.exists():
        raise FileExistsError(f"program output root exists: {config.output_dir}")

    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if video.ndim != 3 or not np.issubdtype(video.dtype, np.number):
        raise ValueError(f"source must be numeric TYX, got {video.shape} {video.dtype}")
    frames = config.frames
    review_start, review_stop = frames.review_start_ui - 1, frames.review_end_ui
    quiet_start, quiet_stop = frames.quiet_start_ui - 1, frames.quiet_end_ui
    if review_stop > len(video) or quiet_stop > review_stop:
        raise ValueError("configured frame interval exceeds the source movie")
    labels = load_and_validate_labels(
        config.labels_tsv, config.label_summary, tuple(video.shape[1:])
    )
    rows = build_design(config)
    cells = enumerate_cells(config)
    review_mib = (
        (review_stop - review_start) * video.shape[1] * video.shape[2]
        * np.dtype(np.float32).itemsize / 2**20
    )
    estimated_peak_ram_mib = int(np.ceil(review_mib * 3.0))
    estimated_output_mib = max(16, int(np.ceil(len(rows) * 0.02)))
    disk_probe = config.output_dir.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    disk_free_mib = shutil.disk_usage(disk_probe).free // 2**20
    ram_available_mib = _available_ram_mib()
    ready = bool(
        ram_available_mib >= config.resources.max_ram_mib
        and estimated_peak_ram_mib <= config.resources.max_ram_mib
        and disk_free_mib >= config.resources.min_free_disk_mib + estimated_output_mib
        and estimated_output_mib <= config.resources.max_output_mib
    )
    repository = config.manifest_path.parent.parent
    authority = data_root(repository)
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "ready": ready,
        "source": {
            "path": portable_path(
                config.source_video, repository=repository, data=authority
            ),
            "shape": list(video.shape), "dtype": str(video.dtype), "axes": "TYX",
            "sha256": _sha256(config.source_video),
        },
        "labels": {
            "path": portable_path(config.labels_tsv, repository=repository, data=authority),
            "rows": len(labels),
            "unique_identities": len({row["roi_identity"] for row in labels}),
            "sha256": _sha256(config.labels_tsv),
            "summary_sha256": _sha256(config.label_summary),
            "coordinates": "x=column,y=row",
            "interpretation": "sparse known positives; unmatched candidates unknown",
        },
        "frames": {
            "review_ui_inclusive": [frames.review_start_ui, frames.review_end_ui],
            "review_zero_half_open": [review_start, review_stop],
            "quiet_ui_inclusive": [frames.quiet_start_ui, frames.quiet_end_ui],
            "quiet_zero_half_open": [quiet_start, quiet_stop],
            "frame_period_ms": frames.frame_period_ms,
        },
        "design": {
            "cell_count": len(cells), "fit_count": len(rows),
            "sobol_points_per_cell": config.design.sobol_points_per_cell,
            "screen_seeds": list(config.design.screen_seeds),
            "digest": design_digest(rows),
        },
        "resources": {
            "estimated_peak_ram_mib": estimated_peak_ram_mib,
            "ram_available_mib": ram_available_mib,
            "ram_cap_mib": config.resources.max_ram_mib,
            "estimated_output_mib": estimated_output_mib,
            "output_cap_mib": config.resources.max_output_mib,
            "disk_free_mib": disk_free_mib,
            "minimum_free_disk_mib": config.resources.min_free_disk_mib,
            "gpu": _gpu_status(), "active_processes": _active_processes(),
        },
        "collisions": {"artifact_dir": False, "program_dir": False},
        "scientific_audit": {
            "enabled": config.scientific_audit.enabled,
            "opt_out_reason": config.scientific_audit.opt_out_reason,
        },
        "current_video_limitation": (
            "within-recording evidence only; no biological-source identity, "
            "precision, or cross-recording generalization"
        ),
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_design(destination / "frozen_design.tsv", rows)
    _projection_overlay(video[quiet_start:quiet_stop], labels, destination / "label_projection_overlay.png")
    _atomic_json(destination / "preflight.json", payload)
    _atomic_json(destination / "resolved_config.json", config.to_dict())
    if not ready:
        raise RuntimeError(f"preflight is not ready: {payload['resources']}")
    return payload


def matching_preflight(
    config: ICAWhiteningConfig, directory: str | Path
) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    payload = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    rows = build_design(config)
    if not payload.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires a ready preflight for the identical config")
    if payload["design"]["digest"] != design_digest(rows):
        raise RuntimeError("frozen design digest changed")
    for expected, path in (
        (payload["source"]["sha256"], config.source_video),
        (payload["labels"]["sha256"], config.labels_tsv),
        (payload["labels"]["summary_sha256"], config.label_summary),
    ):
        if expected != _sha256(path):
            raise RuntimeError(f"input fingerprint changed: {path}")
    return payload
