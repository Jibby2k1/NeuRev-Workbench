"""Collision-safe preparation and orchestration for the conclusive batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from neurobench.experiments.learnable_contrast import core as label_core

from .cnmf_adapter import audit_caiman_backend
from .conclusive_config import ConclusiveBatchConfig
from .screen_runner import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_audit(device: str) -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "error": "torch reports CUDA unavailable"}
        index = torch.device(device).index or 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "--id", str(index)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        values = query.stdout.strip().split(",") if query.returncode == 0 else []
        return {
            "available": True, "device": device, "name": properties.name,
            "total_mib": total_bytes / 2**20, "free_mib": free_bytes / 2**20,
            "temperature_c": float(values[0]) if len(values) == 3 else None,
            "utilization_percent": float(values[1]) if len(values) == 3 else None,
            "used_mib_nvidia_smi": float(values[2]) if len(values) == 3 else None,
        }
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def _memory_audit() -> dict[str, float]:
    values: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key + "_mib"] = float(raw.strip().split()[0]) / 1024
    return values


def _active_python_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,rss=,comm=,args="], capture_output=True,
        text=True, timeout=15, check=False,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if "python" not in line.lower() or "conclusive_batch" in line:
            continue
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            rows.append({"pid": int(parts[0]), "rss_mib": int(parts[1]) / 1024,
                         "command": parts[2], "args": parts[3][:500]})
    return rows


def audit(
    config: ConclusiveBatchConfig,
    *,
    run_authorized_by_user: bool = False,
) -> dict[str, Any]:
    """Perform a read-only input, collision, backend, and resource audit."""
    output = config.output_root
    partial = Path(str(output) + ".partial")
    paths = {
        "scientific_config": config.scientific_config_path,
        "source_video": config.source_video, "source_tiff": config.source_tiff,
        "labels_tsv": config.labels_tsv, "caiman_python": config.caiman_python,
    }
    existing = {key: path.is_file() for key, path in paths.items()}
    movie_shape = None
    movie_dtype = None
    frame_gate = False
    if config.source_video.is_file():
        movie = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        movie_shape = list(movie.shape)
        movie_dtype = str(movie.dtype)
        frame_gate = bool(
            movie.ndim == 3
            and int(config.frames["review_end_ui"]) <= movie.shape[0]
        )
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    gpu = _gpu_audit(str(config.resources["gpu_device"]))
    caiman = audit_caiman_backend(
        "1.13.1", python_executable=config.caiman_python
    )
    memory = _memory_audit()
    processes = _active_python_processes()
    enabled = config.enabled_configuration_count()
    counts = {
        "enabled_configurations": enabled,
        "development_base_fit_ceiling": int(config.design["development_fixture_count"]) * enabled,
        "confidence_numerical_fit_ceiling": int(config.design["development_fixture_count"]) * enabled * (1 + int(config.design["confidence_perturbations"])),
        "confirmation_fit_ceiling": int(config.design["confirmation_fixture_count"]) * 12,
        "semi_synthetic_fit_ceiling": int(config.design["semi_synthetic_fixture_count"]) * 8,
    }
    counts["total_numerical_fit_ceiling"] = sum(
        counts[key] for key in (
            "development_base_fit_ceiling", "confidence_numerical_fit_ceiling",
            "confirmation_fit_ceiling", "semi_synthetic_fit_ceiling",
        )
    )
    gates = {
        "all_inputs_exist": all(existing.values()),
        "output_absent": not output.exists(),
        "partial_output_absent": not partial.exists(),
        "frame_contract_inside_movie": frame_gate,
        "memory_headroom_sufficient": memory.get("MemAvailable_mib", 0) >= int(config.resources["rss_hard_stop_mib"]),
        "disk_headroom_sufficient": disk.free / 2**20 >= int(config.resources["minimum_free_disk_mib"]),
        "cuda_available": bool(gpu.get("available")),
        "gpu_headroom_sufficient": float(gpu.get("free_mib", 0)) >= int(config.resources["minimum_free_gpu_mib"]),
        "gpu_temperature_safe": gpu.get("temperature_c") is None or float(gpu["temperature_c"]) < int(config.resources["gpu_stop_c"]),
        "caiman_exact_version_ready": bool(caiman["fit_authorized"]),
        "no_other_neurobench_python_job": not any("neurobench" in row["args"] for row in processes),
    }
    return {
        "schema_version": 1,
        "kind": "information_source_separation_conclusive_batch_read_only_preflight",
        "experiment_id": config.experiment_id,
        "ready": bool(all(gates.values())),
        "run_authorized_by_user": bool(run_authorized_by_user),
        "output_root": str(output), "partial_root": str(partial),
        "gates": gates, "inputs": {key: str(path) for key, path in paths.items()},
        "input_exists": existing, "movie_shape": movie_shape,
        "movie_dtype": movie_dtype, "counts": counts,
        "memory": memory,
        "disk": {"free_mib": disk.free / 2**20, "total_mib": disk.total / 2**20},
        "gpu": gpu, "caiman": caiman, "other_python_processes": processes,
        "resources": config.resources,
    }


def _projection_overlay(config: ConclusiveBatchConfig, destination: Path) -> dict[str, Any]:
    movie = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    start = int(config.frames["review_start_ui"]) - 1
    stop = int(config.frames["quiet_end_ui"])
    projection = np.max(np.asarray(movie[start:stop], dtype=np.float32), axis=0)
    lo, hi = np.percentile(projection, [1, 99.8])
    scaled = np.asarray(np.clip((projection-lo)/max(float(hi-lo), 1e-8), 0, 1)*255, dtype=np.uint8)
    image = Image.fromarray(scaled).convert("RGB")
    draw = ImageDraw.Draw(image)
    labels = label_core.load_labels(config.labels_tsv)
    for row in labels:
        x = int(round(float(row["x_px"])))
        y = int(round(float(row["y_px"])))
        draw.ellipse((x-5, y-5, x+5, y+5), outline=(0, 255, 255), width=2)
        draw.text((x+6, y-6), str(row["roi_identity"]).replace("roi_", ""), fill=(255, 255, 0))
    image.save(destination)
    return {"path": str(destination), "label_row_count": len(labels),
            "projection_frames_ui_inclusive": [int(config.frames["review_start_ui"]), int(config.frames["quiet_end_ui"])],
            "coordinates": "x=column,y=row"}


def prepare(
    config: ConclusiveBatchConfig,
    *,
    run_authorized_by_user: bool = False,
) -> dict[str, Any]:
    """Create the immutable partial root after a passing read-only audit."""
    if not run_authorized_by_user:
        raise RuntimeError(
            "preparing a conclusive batch requires explicit run authorization"
        )
    report = audit(config, run_authorized_by_user=True)
    if not report["ready"]:
        failed = [key for key, value in report["gates"].items() if not value]
        raise RuntimeError(f"conclusive batch preflight failed: {failed}")
    partial = Path(str(config.output_root) + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    for name in ("stages", "fits", "videos", "review", "logs"):
        (partial / name).mkdir()
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    input_manifest = {
        "scientific_config": {"path": str(config.scientific_config_path), "sha256": _sha256(config.scientific_config_path)},
        "labels_tsv": {"path": str(config.labels_tsv), "sha256": _sha256(config.labels_tsv)},
        "source_video": {"path": str(config.source_video), "bytes": config.source_video.stat().st_size,
                         "mtime_ns": config.source_video.stat().st_mtime_ns},
        "source_tiff": {"path": str(config.source_tiff), "bytes": config.source_tiff.stat().st_size,
                        "mtime_ns": config.source_tiff.stat().st_mtime_ns},
        "caiman_python": str(config.caiman_python),
    }
    _atomic_json(partial / "input_manifest.json", input_manifest)
    projection = _projection_overlay(config, partial / "label_projection_overlay.png")
    prepared = {
        "schema_version": 1, "status": "prepared", "completed_stages": [],
        "current_stage": None, "terminal_disposition": None,
        "started_unix": time.time(), "updated_unix": time.time(),
        "projection": projection, "preflight": report,
    }
    _atomic_json(partial / "run_state.json", prepared)
    _atomic_json(partial / "preflight.json", report)
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare"))
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--authorize-run",
        action="store_true",
        help="Explicitly authorize creation of a runnable partial output root.",
    )
    args = parser.parse_args(argv)
    config = ConclusiveBatchConfig.load(args.config)
    payload = (
        audit(config)
        if args.action == "audit"
        else prepare(config, run_authorized_by_user=args.authorize_run)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
