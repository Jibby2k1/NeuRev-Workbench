"""Read-only, resource-aware preflight for soma-excitation experiments."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from neurobench.data.video import video_metadata

from .config import SomaExcitationConfig


MIB = 1024 * 1024
AVAILABLE_RAM_FRACTION = 0.5
DISK_SAFETY_MULTIPLIER = 1.25
# Importing the scientific runtime (Torch, SciPy, OpenMP/MKL) and executing a
# frozen model retained roughly 640 MiB beyond the explicitly counted arrays in
# the July 2026 pilot. Keep that empirical overhead separate and visible rather
# than allowing the array-only estimate to imply a much smaller process RSS.
SCIENTIFIC_RUNTIME_OVERHEAD_BYTES = 640 * MIB
RUNTIME_GUARD_HEADROOM_BYTES = 32 * MIB


class PreflightError(RuntimeError):
    """Raised when an experiment cannot safely start."""


class ResourceBudgetError(PreflightError):
    """Raised when RAM, output, or available-disk limits cannot be met."""


@dataclass(frozen=True)
class _MemoryEstimate:
    fixed_bytes: int
    per_chunk_frame_bytes: int
    resolved_chunk_frames: int
    peak_bytes: int
    effective_budget_bytes: int


def build_soma_excitation_preflight(
    config: SomaExcitationConfig | Mapping[str, Any] | str | Path,
    *,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    """Return a side-effect-free plan; reject existing outputs unless explicitly allowed."""
    resolved_config = _coerce_config(config)
    resolved_config.validate()

    source = Path(resolved_config.source_video)
    if not source.is_file():
        raise FileNotFoundError(f"Soma-excitation source video does not exist or is not a file: {source}")

    output = Path(resolved_config.output_dir)
    if output.exists() and not allow_existing_output:
        raise FileExistsError(
            f"Refusing output collision at {output}. Pass allow_existing_output=True only for an intentional resume."
        )
    if output.exists() and not output.is_dir():
        raise PreflightError(f"Configured output_dir exists but is not a directory: {output}")

    checkpoint_rows = _checkpoint_rows(resolved_config)
    metadata = video_metadata(source)
    frame_count = _positive_metadata_int(metadata, "frames")
    height = _positive_metadata_int(metadata, "height")
    width = _positive_metadata_int(metadata, "width")
    dtype_itemsize = _dtype_itemsize(str(metadata.get("dtype") or ""))

    onset_zero = resolved_config.onset_frame_zero
    control_start_zero = resolved_config.control_start_frame_zero
    score_stop_zero = frame_count if resolved_config.end_frame_ui is None else resolved_config.end_frame_ui
    if onset_zero >= frame_count:
        raise PreflightError(
            f"onset_frame_ui={resolved_config.onset_frame_ui} maps to zero-based index {onset_zero}, "
            f"outside a {frame_count}-frame source."
        )
    if score_stop_zero > frame_count:
        raise PreflightError(
            f"end_frame_ui={resolved_config.end_frame_ui} exceeds the source's last one-based frame {frame_count}."
        )
    if score_stop_zero <= onset_zero:
        raise PreflightError("The selected score interval contains no frames.")
    if onset_zero - control_start_zero != resolved_config.control_preroll_frames:
        raise PreflightError("The control pre-roll could not be represented exactly.")

    analysis_frames = score_stop_zero - control_start_zero
    score_frames = score_stop_zero - onset_zero
    available_ram, ram_source = available_ram_bytes()
    largest_checkpoint = max((row["size_bytes"] for row in checkpoint_rows), default=0)
    memory = _resolve_memory_estimate(
        height=height,
        width=width,
        source_itemsize=dtype_itemsize,
        requested_chunk_frames=resolved_config.resources.chunk_frames,
        analysis_frames=analysis_frames,
        max_ram_bytes=resolved_config.resources.max_ram_mib * MIB,
        available_ram=available_ram,
        largest_checkpoint_bytes=largest_checkpoint,
        has_checkpoints=bool(checkpoint_rows),
    )

    output_estimate = _estimate_output_bytes(
        height=height,
        width=width,
        score_frames=score_frames,
        checkpoint_count=len(checkpoint_rows),
    )
    output_limit = resolved_config.resources.max_output_mib * MIB
    if output_estimate > output_limit:
        raise ResourceBudgetError(
            f"Estimated output {_format_mib(output_estimate)} exceeds max_output_mib="
            f"{resolved_config.resources.max_output_mib} MiB. Reduce the frame interval or explicitly raise the cap."
        )

    disk_path = _nearest_existing_parent(output)
    disk = shutil.disk_usage(disk_path)
    required_disk = int(math.ceil(output_estimate * DISK_SAFETY_MULTIPLIER))
    if disk.free < required_disk:
        raise ResourceBudgetError(
            f"Available disk at {disk_path} is {_format_mib(disk.free)}, below the conservative requirement "
            f"of {_format_mib(required_disk)}."
        )

    warnings: list[str] = []
    if memory.resolved_chunk_frames < resolved_config.resources.chunk_frames:
        warnings.append(
            "RAM preflight reduced chunk_frames from "
            f"{resolved_config.resources.chunk_frames} to {memory.resolved_chunk_frames}."
        )
    if output.exists():
        warnings.append("Existing output directory was explicitly allowed; the runner must use resume-safe writes.")
    if not checkpoint_rows:
        warnings.append("No dynamics checkpoints were supplied; only detector/baseline lanes can run.")
    elif any(row["horizon_frames"] is None for row in checkpoint_rows):
        warnings.append("At least one dynamics checkpoint lacks explicit horizon_frames; model evaluation may be refused.")

    frame_bounds = {
        "ui_index_base": 1,
        "array_index_base": 0,
        "onset_frame_ui": int(resolved_config.onset_frame_ui),
        "onset_frame_zero": int(onset_zero),
        "control_start_frame_ui": int(control_start_zero + 1),
        "control_start_frame_zero": int(control_start_zero),
        "control_stop_frame_zero_exclusive": int(onset_zero),
        "control_frame_count": int(resolved_config.control_preroll_frames),
        "score_start_frame_ui": int(resolved_config.onset_frame_ui),
        "score_start_frame_zero": int(onset_zero),
        "score_stop_frame_zero_exclusive": int(score_stop_zero),
        "score_last_frame_ui": int(score_stop_zero),
        "score_frame_count": int(score_frames),
        "analysis_start_frame_zero": int(control_start_zero),
        "analysis_stop_frame_zero_exclusive": int(score_stop_zero),
        "analysis_frame_count": int(analysis_frames),
    }
    resources = {
        "device": "cpu",
        "worker_count": 1,
        "cpu_threads": int(resolved_config.resources.cpu_threads),
        "requested_chunk_frames": int(resolved_config.resources.chunk_frames),
        "resolved_chunk_frames": int(memory.resolved_chunk_frames),
        "max_ram_mib": int(resolved_config.resources.max_ram_mib),
        "available_ram_bytes": available_ram,
        "available_ram_source": ram_source,
        "available_ram_safety_fraction": AVAILABLE_RAM_FRACTION,
        "effective_ram_budget_bytes": int(memory.effective_budget_bytes),
        "estimated_fixed_workspace_bytes": int(memory.fixed_bytes),
        "scientific_runtime_overhead_bytes": (
            SCIENTIFIC_RUNTIME_OVERHEAD_BYTES if checkpoint_rows else 0
        ),
        "runtime_guard_headroom_bytes": RUNTIME_GUARD_HEADROOM_BYTES if checkpoint_rows else 0,
        "estimated_per_chunk_frame_bytes": int(memory.per_chunk_frame_bytes),
        "estimated_peak_ram_bytes": int(memory.peak_bytes),
        "max_output_mib": int(resolved_config.resources.max_output_mib),
        "estimated_output_bytes": int(output_estimate),
        "disk": {
            "path": str(disk_path),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
            "required_with_safety_bytes": int(required_disk),
            "safety_multiplier": DISK_SAFETY_MULTIPLIER,
        },
    }
    checks = [
        {"name": "source_exists", "status": "pass", "detail": str(source)},
        {
            "name": "checkpoint_paths_exist",
            "status": "pass",
            "detail": f"{len(checkpoint_rows)} checkpoint(s) validated",
        },
        {
            "name": "frame_bounds_valid",
            "status": "pass",
            "detail": f"control pre-roll resolved to {resolved_config.control_preroll_frames} frame(s)",
        },
        {
            "name": "ram_budget",
            "status": "pass",
            "detail": f"resolved chunk={memory.resolved_chunk_frames}, peak={memory.peak_bytes} bytes",
        },
        {
            "name": "output_budget",
            "status": "pass",
            "detail": f"estimate={output_estimate} bytes, cap={output_limit} bytes",
        },
        {
            "name": "disk_space",
            "status": "pass",
            "detail": f"free={disk.free} bytes, required={required_disk} bytes",
        },
        {
            "name": "output_collision",
            "status": "pass",
            "detail": "absent" if not output.exists() else "existing path explicitly allowed",
        },
    ]
    return {
        "schema_version": 1,
        "experiment_type": "soma_excitation_transfer",
        "experiment_id": resolved_config.experiment_id,
        "status": "ready",
        "ready": True,
        "config": resolved_config.to_dict(),
        "source": {
            "path": str(source),
            "metadata": _json_ready_metadata(metadata),
        },
        "output_dir": str(output),
        "allow_existing_output": bool(allow_existing_output),
        "frame_bounds": frame_bounds,
        "cfar": {
            "small_radius_px": resolved_config.cfar.small_radius_px,
            "large_radius_px": resolved_config.cfar.large_radius_px,
            "pfa": resolved_config.cfar.pfa,
            "epsilon": resolved_config.cfar.epsilon,
            "signal_polarity": "positive_excitation",
        },
        "dark_zones": {
            **resolved_config.to_dict()["dark_zones"],
            "zone_api_parameters": resolved_config.dark_zones.as_zone_kwargs(),
            "interpretation": "dark soma core is background; excitation is evaluated in the surrounding annulus",
        },
        "dynamics_checkpoints": checkpoint_rows,
        "resources": resources,
        "processing_contract": {
            "source_access": "memory_mapped_or_chunked",
            "checkpoint_order": "sequential",
            "model_batch_size": 1,
            "dense_full_video_outputs": False,
            "control_frames_are_scored": False,
        },
        "checks": checks,
        "warnings": warnings,
    }


preflight_soma_excitation = build_soma_excitation_preflight
run_preflight = build_soma_excitation_preflight


def available_ram_bytes() -> tuple[int | None, str]:
    """Return available RAM without importing optional process packages."""
    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024, "/proc/meminfo:MemAvailable"
    except (OSError, ValueError):
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size, "os.sysconf:SC_AVPHYS_PAGES"
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None, "unavailable"


def _coerce_config(config: SomaExcitationConfig | Mapping[str, Any] | str | Path) -> SomaExcitationConfig:
    if isinstance(config, SomaExcitationConfig):
        return config
    if isinstance(config, Mapping):
        return SomaExcitationConfig.from_dict(config)
    return SomaExcitationConfig.load_json(config)


def _checkpoint_rows(config: SomaExcitationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in config.dynamics_checkpoints:
        path = Path(checkpoint.path)
        if not path.is_file():
            raise FileNotFoundError(f"Dynamics checkpoint does not exist or is not a file: {path}")
        rows.append(
            {
                "model_id": checkpoint.model_id or path.stem,
                "horizon_frames": checkpoint.horizon_frames,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "evaluation_order": len(rows),
            }
        )
    return rows


def _resolve_memory_estimate(
    *,
    height: int,
    width: int,
    source_itemsize: int,
    requested_chunk_frames: int,
    analysis_frames: int,
    max_ram_bytes: int,
    available_ram: int | None,
    largest_checkpoint_bytes: int,
    has_checkpoints: bool,
) -> _MemoryEstimate:
    pixels = height * width
    fixed_bytes = 16 * MIB + pixels * (8 * 4 + 4) + largest_checkpoint_bytes * 4
    if has_checkpoints:
        fixed_bytes += SCIENTIFIC_RUNTIME_OVERHEAD_BYTES + RUNTIME_GUARD_HEADROOM_BYTES
    per_frame_bytes = pixels * (max(source_itemsize, 4) + 12 * 4 + 4)
    available_limit = (
        int(available_ram * AVAILABLE_RAM_FRACTION) if available_ram is not None else max_ram_bytes
    )
    effective_budget = min(max_ram_bytes, available_limit)
    if effective_budget <= fixed_bytes:
        raise ResourceBudgetError(
            f"RAM budget {_format_mib(effective_budget)} cannot hold the conservative fixed workspace "
            f"of {_format_mib(fixed_bytes)}. Increase max_ram_mib or remove oversized checkpoints."
        )
    max_chunk = (effective_budget - fixed_bytes) // per_frame_bytes
    resolved = min(requested_chunk_frames, analysis_frames, int(max_chunk))
    if resolved < 1:
        raise ResourceBudgetError(
            f"RAM budget {_format_mib(effective_budget)} cannot hold one frame of the estimated chunk workspace."
        )
    peak = fixed_bytes + resolved * per_frame_bytes
    return _MemoryEstimate(
        fixed_bytes=int(fixed_bytes),
        per_chunk_frame_bytes=int(per_frame_bytes),
        resolved_chunk_frames=int(resolved),
        peak_bytes=int(peak),
        effective_budget_bytes=int(effective_budget),
    )


def _estimate_output_bytes(*, height: int, width: int, score_frames: int, checkpoint_count: int) -> int:
    pixels = height * width
    base_metadata = 8 * MIB
    zone_maps_and_projections = pixels * (6 * 4 + 4)
    sparse_event_ceiling = int(math.ceil(score_frames * pixels * 0.05)) * 32
    selected_preview_frames = min(score_frames, 16) * pixels * 2
    scalar_metrics = score_frames * (512 + checkpoint_count * 256)
    checkpoint_reports = checkpoint_count * 256 * 1024
    return int(
        base_metadata
        + zone_maps_and_projections
        + sparse_event_ceiling
        + selected_preview_frames
        + scalar_metrics
        + checkpoint_reports
    )


def _dtype_itemsize(dtype_name: str) -> int:
    try:
        import numpy as np

        return int(np.dtype(dtype_name).itemsize)
    except (ImportError, TypeError) as exc:
        raise PreflightError(f"Could not interpret source video dtype {dtype_name!r}.") from exc


def _positive_metadata_int(metadata: Mapping[str, Any], field: str) -> int:
    try:
        value = int(metadata[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError(f"video_metadata did not provide a valid '{field}' value.") from exc
    if value <= 0:
        raise PreflightError(f"video_metadata reported non-positive {field}={value}.")
    return value


def _nearest_existing_parent(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _json_ready_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, Path):
            result[str(key)] = str(value)
        elif isinstance(value, tuple):
            result[str(key)] = [int(item) if isinstance(item, int) else item for item in value]
        elif hasattr(value, "item") and callable(value.item):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _format_mib(value: int) -> str:
    return f"{value / MIB:.1f} MiB"
