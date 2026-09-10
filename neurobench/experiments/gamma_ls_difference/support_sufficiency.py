"""Fail-closed Gamma-LS radial-support sufficiency screen and diagnostics.

This is an upstream, burst-window-supervised screen.  It never opens sparse
coordinates or identities and cannot by itself establish support sufficiency.
It freezes fold-local candidates and larger-support comparators for a later
protected recall sensitivity, while measuring repeated Gamma-stage latency and
executing named legacy/square controls outside the primary selector.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)

from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .grid import GammaContext, SCREEN_REPRESENTATIONS, TRAINING_FOLDS
from .screen import (
    GAMMA_EPSILON,
    QUIET_SWAPS,
    TAIL_QUANTILE_METHOD,
    _artifact_index,
    _atomic_json,
    _atomic_tsv,
    _elapsed,
    _evaluate_context_arm,
    _interval_mask,
    _positive_scale_floor,
    _representation,
    _sha256,
    _source_contract,
    _stream_common_history_to_device,
    _tail_metrics,
    _verify_ready_preflight,
    build_fold_contracts,
)
from .support_config import GammaSupportConfig
from .support_grid import (
    CONTROLS,
    PRIMARY_MAX_HALF_WIDTH,
    STAGE_A_ENDPOINT,
    STAGE_B_ENDPOINT,
    best_context_by_fold_and_width,
    design_counts,
    enumerate_stage_contexts,
    stage_b_required,
    support_boundary_status,
)


class GammaSupportUnavailable(RuntimeError):
    """Raised before destination mutation when the guarded run is unavailable."""


def _canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _implementation_hashes(config: GammaSupportConfig) -> dict[str, str]:
    names = (
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
        "neurobench/experiments/gamma_ls_difference/screen.py",
        "neurobench/experiments/gamma_ls_difference/support_config.py",
        "neurobench/experiments/gamma_ls_difference/support_grid.py",
        "neurobench/experiments/gamma_ls_difference/support_sufficiency.py",
    )
    return {name: _sha256(config.repository / name) for name in names}


def _verify_index(root: Path) -> dict[str, Any]:
    index_path = root / "artifact_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not isinstance(index.get("artifacts"), list):
        raise ValueError("base screen artifact index is invalid")
    for item in index["artifacts"]:
        path = root / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"base screen indexed artifact is missing: {path}")
        if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
            raise ValueError(f"base screen indexed artifact changed: {path.name}")
    return {
        "artifact_index_sha256": _sha256(index_path),
        "verified_artifact_count": len(index["artifacts"]),
    }


_OLD_CONTEXT = re.compile(
    r"^gamma_h(?P<h>[0-9]+)_g(?P<g>[0-9]+)_n(?P<n>[0-9]+(?:p[0-9]+)?)_m(?P<m>[0-9]+(?:p[0-9]+)?)$"
)


def _untoken(value: str) -> float:
    return float(value.replace("p", "."))


def _old_context_dict(context_id: str) -> dict[str, Any]:
    match = _OLD_CONTEXT.fullmatch(context_id)
    if match is None:
        raise ValueError(f"cannot decode original screen context {context_id!r}")
    half = int(match.group("h"))
    guard = int(match.group("g"))
    shape = _untoken(match.group("n"))
    mode = _untoken(match.group("m"))
    return {
        "context_id": context_id,
        "stage": "base_g2",
        "half_width_px": half,
        "guard_radius_px": guard,
        "shape": shape,
        "mode_fraction_of_half_width": mode,
        "mode_radius_px": mode * half,
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
        "eligible_primary": True,
    }


def _verify_base_screen(config: GammaSupportConfig) -> dict[str, Any]:
    root = config.base_screen_path
    if not root.is_dir():
        raise FileNotFoundError(f"base GPU screen is missing: {root}")
    indexed = _verify_index(root)
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    boundary = json.loads((root / "claim_boundary.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed_screen_artifact_contract":
        raise ValueError("base GPU screen did not pass its artifact contract")
    if summary.get("status") != "complete_screen_only":
        raise ValueError("base GPU screen is not the completed screen-only artifact")
    if boundary.get("positive_coordinates_used") is not False or boundary.get(
        "positive_identities_used"
    ) is not False:
        raise ValueError("base GPU screen is contaminated by protected fields")
    if selection.get("selection_scope") != "outer_training_fold_only" or selection.get(
        "across_fold_pooling_used_for_protected_selection"
    ) is not False:
        raise ValueError("base screen selection scope changed")
    folds = selection.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ValueError("base screen must expose four fold-local contexts")
    originals = {}
    for item in folds:
        fold = int(item["training_fold"])
        if fold in originals:
            raise ValueError("duplicate base-screen training fold")
        originals[fold] = _old_context_dict(str(item["common_g2_finalist_context_id"]))
    if set(originals) != set(TRAINING_FOLDS):
        raise ValueError("base screen fold set changed")
    return {
        **indexed,
        "root": str(root),
        "summary_sha256": _sha256(root / "summary.json"),
        "selection_sha256": _sha256(root / "selection.json"),
        "original_fold_contexts": originals,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }


def _planned_context_rows() -> list[dict[str, Any]]:
    rows = []
    for stage in ("support_a", "support_b"):
        conditional = stage == "support_b"
        for context in enumerate_stage_contexts(stage):
            rows.append({**context.as_dict(), "conditional": conditional})
    for control in CONTROLS:
        rows.append(
            {
                "context_id": control.control_id,
                "stage": "diagnostic_control",
                "half_width_px": "",
                "guard_radius_px": "",
                "shape": "",
                "mode_fraction_of_half_width": "",
                "mode_radius_px": "",
                "support": control.family,
                "padding": control.semantics,
                "eligible_primary": False,
                "conditional": False,
            }
        )
    return rows


def run_support_preflight(
    config: GammaSupportConfig,
    *,
    base_preflight_dir: str | Path,
    artifact_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Validate all sealed inputs and write a non-colliding support preflight."""

    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"support preflight exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"support preflight parent is missing: {destination.parent}")
    base_preflight = _verify_ready_preflight(config.base_config, base_preflight_dir)
    base_screen = _verify_base_screen(config)
    requested = str(device or config.payload["resources"]["device"])
    try:
        runtime = require_cuda_device(requested)
    except CudaRuntimeUnavailable as error:
        raise GammaSupportUnavailable(str(error)) from error
    disk = shutil.disk_usage(destination.parent)
    minimum = int(float(config.payload["resources"]["minimum_free_disk_gib"]) * 2**30)
    if disk.free < minimum:
        raise GammaSupportUnavailable(
            f"free disk {disk.free} is below frozen minimum {minimum} bytes"
        )
    if int(runtime["free_vram_bytes_before"]) < int(
        float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30
    ):
        raise GammaSupportUnavailable("free VRAM is below the frozen 8-GiB run cap")
    payload = {
        "schema_version": 1,
        "experiment_id": config.payload["experiment_id"],
        "status": "ready_support_screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_ready": True,
        "gpu_run_ready": True,
        "base_preflight": base_preflight,
        "base_screen": base_screen,
        "runtime": runtime,
        "design_counts_stage_a_only": design_counts(stage_b_run=False),
        "design_counts_stage_a_plus_b_maximum": design_counts(stage_b_run=True),
        "adaptive_boundary_rule": config.payload["stopping_rule"],
        "sparse_positive_tables_opened": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
        "scientific_audit_enabled": True,
        "scientific_audit_status": "pending_downstream_frozen_candidate_evaluation",
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "implementation": _implementation_hashes(config),
    }
    partial = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        partial.mkdir()
        _atomic_json(partial / "preflight.json", payload)
        _atomic_json(partial / "config.portable.json", config.portable_dict())
        _atomic_tsv(partial / "planned_contexts_and_controls.tsv", _planned_context_rows())
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "preflight.json",
                "grain": "radial context or ineligible diagnostic control",
                "stage_b_trigger": config.payload["stopping_rule"]["stage_b_trigger"],
                "protected_fields": "not opened",
            },
        )
        _atomic_json(partial / "validation.json", {"status": "passed_support_preflight"})
        _atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        if destination.exists():
            raise FileExistsError("support preflight destination appeared during commit")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return payload


def _verify_support_preflight(
    config: GammaSupportConfig,
    *,
    base_preflight_dir: str | Path,
    support_preflight_dir: str | Path,
) -> dict[str, Any]:
    root = Path(support_preflight_dir).expanduser().resolve()
    payload = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    portable = json.loads((root / "config.portable.json").read_text(encoding="utf-8"))
    if portable != config.portable_dict():
        raise ValueError("support preflight config does not match")
    if payload.get("status") != "ready_support_screen" or not payload.get("gpu_run_ready"):
        raise ValueError("support preflight is not GPU-ready")
    if payload.get("implementation") != _implementation_hashes(config):
        raise ValueError("support implementation changed after preflight")
    current_base = _verify_ready_preflight(config.base_config, base_preflight_dir)
    if payload.get("base_preflight", {}).get("preflight_sha256") != current_base.get(
        "preflight_sha256"
    ):
        raise ValueError("base preflight changed after support preflight")
    current_screen = _verify_base_screen(config)
    if payload.get("base_screen", {}).get("artifact_index_sha256") != current_screen.get(
        "artifact_index_sha256"
    ):
        raise ValueError("base screen changed after support preflight")
    return {
        "root": str(root),
        "preflight_sha256": _sha256(root / "preflight.json"),
        "base_preflight_sha256": current_base["preflight_sha256"],
        "base_screen_artifact_index_sha256": current_screen["artifact_index_sha256"],
        "original_fold_contexts": current_screen["original_fold_contexts"],
    }


def _square_annulus_moments(
    values: Any,
    *,
    outer: int = 11,
    guard: int = 3,
    chunk_frames: int | None = None,
) -> tuple[Any, Any]:
    """Signed square-annulus moments with valid-reference renormalization."""

    import torch
    import torch.nn.functional as functional

    if values.ndim != 3 or not 0 <= guard < outer:
        raise ValueError("square annulus requires TYX values and 0 <= guard < outer")
    frames_per_chunk = len(values) if chunk_frames is None else int(chunk_frames)
    if frames_per_chunk < 1:
        raise ValueError("chunk_frames must be positive")
    width = 2 * outer + 1
    kernel = torch.ones((width, width), dtype=values.dtype, device=values.device)
    center = outer
    kernel[
        center - guard : center + guard + 1,
        center - guard : center + guard + 1,
    ] = 0.0
    weight = kernel.reshape(1, 1, width, width)
    valid = functional.conv2d(
        torch.ones((1, 1, values.shape[-2], values.shape[-1]), dtype=values.dtype, device=values.device),
        weight,
        padding=outer,
    )
    if bool((valid <= 0).any()):
        raise ValueError("square annulus has zero border reference count")
    means = []
    stds = []
    for start in range(0, len(values), frames_per_chunk):
        frames = values[start : start + frames_per_chunk].unsqueeze(1)
        mean = (functional.conv2d(frames, weight, padding=outer) / valid).squeeze(1)
        second = (
            functional.conv2d(frames.square(), weight, padding=outer) / valid
        ).squeeze(1)
        means.append(mean)
        stds.append(torch.sqrt((second - mean.square()).clamp_min(0.0)))
    return torch.cat(means), torch.cat(stds)


def _maintained_positive_box_score(
    values: Any,
    *,
    outer: int = 11,
    guard: int = 3,
    chunk_frames: int | None = None,
) -> Any:
    """GPU-native parity with the maintained robust_local_cfar Torch semantics."""

    import torch
    import torch.nn.functional as functional

    frames_per_chunk = len(values) if chunk_frames is None else int(chunk_frames)
    if frames_per_chunk < 1:
        raise ValueError("chunk_frames must be positive")

    def box_mean(source: Any, radius: int) -> Any:
        frames = source.unsqueeze(1)
        padded = functional.pad(frames, (radius, radius, radius, radius), mode="replicate")
        return functional.avg_pool2d(
            padded, kernel_size=2 * radius + 1, stride=1
        ).squeeze(1)

    outer_area = float((2 * outer + 1) ** 2)
    guard_area = float((2 * guard + 1) ** 2)
    training_area = outer_area - guard_area
    scores = []
    for start in range(0, len(values), frames_per_chunk):
        evidence = values[start : start + frames_per_chunk].clamp_min(0.0)
        outer_mean = box_mean(evidence, outer)
        outer_second = box_mean(evidence.square(), outer)
        guard_mean = box_mean(evidence, guard)
        guard_second = box_mean(evidence.square(), guard)
        mean = (outer_mean * outer_area - guard_mean * guard_area) / training_area
        second = (outer_second * outer_area - guard_second * guard_area) / training_area
        std = torch.sqrt((second - mean.square()).clamp_min(0.0) + GAMMA_EPSILON)
        scores.append(((evidence - mean) / (std + GAMMA_EPSILON)).clamp_min(0.0))
    return torch.cat(scores)


def _evaluate_moments_control(
    representation: Any,
    *,
    representation_name: str,
    control_id: str,
    folds: Sequence[Any],
    frame_ui: Any,
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    tail_quantile: float,
    chunk_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    (mean, std), runtime_ms = _elapsed(
        representation.device,
        lambda: _square_annulus_moments(
            representation, chunk_frames=chunk_frames
        ),
    )
    rows: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    for fold in folds:
        fold_swaps = []
        for swap_name, floor_interval, tail_interval in (
            (QUIET_SWAPS[0], fold.quiet_half_a_ui, fold.quiet_half_b_ui),
            (QUIET_SWAPS[1], fold.quiet_half_b_ui, fold.quiet_half_a_ui),
        ):
            floor_mask = _interval_mask(frame_ui, floor_interval) & ~_interval_mask(
                frame_ui, fold.heldout_guard_ui
            )

            def measure() -> tuple[Any, Any, Any, dict[str, Any]]:
                floor = _positive_scale_floor(std, floor_mask, scale_floor_percentile)
                score = (representation - mean) / (torch.maximum(std, floor) + GAMMA_EPSILON)
                event, quiet, counts = _tail_metrics(
                    score,
                    frame_ui=frame_ui,
                    fold=fold,
                    bursts=bursts,
                    quiet_tail_interval=tail_interval,
                    tail_quantile=tail_quantile,
                )
                return floor, event, quiet, counts

            (floor, event, quiet, counts), post_ms = _elapsed(representation.device, measure)
            item = {
                "stage": "diagnostic_control",
                "context_id": control_id,
                "representation": representation_name,
                "training_fold": fold.training_fold,
                "heldout_burst": fold.heldout_burst,
                "quiet_swap": swap_name,
                "scale_floor": float(floor.item()),
                "event_positive_tail": float(event.item()),
                "quiet_positive_tail": float(quiet.item()),
                "positive_tail_contrast": float((event - quiet).item()),
                "runtime_ms_per_frame": runtime_ms / len(representation),
                "postprocess_runtime_ms": post_ms,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
                "eligible_primary": False,
                "training_event_frame_counts": json.dumps(counts["training_event_frame_counts"], sort_keys=True),
                "quiet_tail_frames": counts["quiet_tail_frames"],
            }
            swaps.append(item)
            fold_swaps.append(item)
        rows.append(
            {
                "stage": "diagnostic_control",
                "context_id": control_id,
                "representation": representation_name,
                "training_fold": fold.training_fold,
                "event_positive_tail": sum(row["event_positive_tail"] for row in fold_swaps) / 2,
                "quiet_positive_tail": sum(row["quiet_positive_tail"] for row in fold_swaps) / 2,
                "runtime_ms_per_frame": runtime_ms / len(representation),
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
                "eligible_primary": False,
            }
        )
    del mean, std
    return rows, swaps, {"context_id": control_id, "runtime_ms": runtime_ms}


def _evaluate_fixed_score_control(
    representation: Any,
    *,
    representation_name: str,
    control_id: str,
    operation: Callable[[], Any],
    folds: Sequence[Any],
    frame_ui: Any,
    bursts: Mapping[str, Sequence[int]],
    tail_quantile: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    score, runtime_ms = _elapsed(representation.device, operation)
    rows: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    for fold in folds:
        fold_swaps = []
        for swap_name, _, tail_interval in (
            (QUIET_SWAPS[0], fold.quiet_half_a_ui, fold.quiet_half_b_ui),
            (QUIET_SWAPS[1], fold.quiet_half_b_ui, fold.quiet_half_a_ui),
        ):
            event, quiet, counts = _tail_metrics(
                score,
                frame_ui=frame_ui,
                fold=fold,
                bursts=bursts,
                quiet_tail_interval=tail_interval,
                tail_quantile=tail_quantile,
            )
            item = {
                "stage": "diagnostic_control",
                "context_id": control_id,
                "representation": representation_name,
                "training_fold": fold.training_fold,
                "heldout_burst": fold.heldout_burst,
                "quiet_swap": swap_name,
                "scale_floor": "not_applicable_fixed_control_semantics",
                "event_positive_tail": float(event.item()),
                "quiet_positive_tail": float(quiet.item()),
                "positive_tail_contrast": float((event - quiet).item()),
                "runtime_ms_per_frame": runtime_ms / len(representation),
                "postprocess_runtime_ms": 0.0,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
                "eligible_primary": False,
                "training_event_frame_counts": json.dumps(counts["training_event_frame_counts"], sort_keys=True),
                "quiet_tail_frames": counts["quiet_tail_frames"],
            }
            swaps.append(item)
            fold_swaps.append(item)
        rows.append(
            {
                "stage": "diagnostic_control",
                "context_id": control_id,
                "representation": representation_name,
                "training_fold": fold.training_fold,
                "event_positive_tail": sum(row["event_positive_tail"] for row in fold_swaps) / 2,
                "quiet_positive_tail": sum(row["quiet_positive_tail"] for row in fold_swaps) / 2,
                "runtime_ms_per_frame": runtime_ms / len(representation),
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
                "eligible_primary": False,
            }
        )
    del score
    return rows, swaps, {"context_id": control_id, "runtime_ms": runtime_ms}


def _run_contexts(
    common: Any,
    *,
    contexts: Sequence[GammaContext],
    folds: Sequence[Any],
    frame_ui: Any,
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    tail_quantile: float,
    chunk_frames: int,
    max_vram_bytes: int,
    heartbeat: Callable[[Mapping[str, Any]], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    import torch

    rows: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    peak_by_width: dict[int, int] = {}
    complete = 0
    total = len(contexts) * len(SCREEN_REPRESENTATIONS)
    for representation_name in SCREEN_REPRESENTATIONS:
        representation = _representation(common, representation_name)
        for context in contexts:
            torch.cuda.reset_peak_memory_stats(representation.device)
            aggregate, quiet_rows, timing = _evaluate_context_arm(
                representation,
                representation_name=representation_name,
                context=context,
                stage=context.stage,
                folds=folds,
                frame_ui=frame_ui,
                bursts=bursts,
                scale_floor_percentile=scale_floor_percentile,
                tail_quantile=tail_quantile,
                chunk_frames=chunk_frames,
            )
            peak = int(torch.cuda.max_memory_allocated(representation.device))
            if peak > max_vram_bytes:
                raise RuntimeError(
                    f"context {context.context_id} exceeded VRAM cap: {peak} > {max_vram_bytes}"
                )
            peak_by_width[context.half_width_px] = max(
                peak_by_width.get(context.half_width_px, 0), peak
            )
            rows.extend(aggregate)
            swaps.extend(quiet_rows)
            timings.append({**timing, "max_memory_allocated_bytes": peak})
            complete += 1
            heartbeat(
                {
                    "stage": context.stage,
                    "completed_context_arm_maps": complete,
                    "total_context_arm_maps": total,
                }
            )
        del representation
    return rows, swaps, timings, peak_by_width


def _run_controls(
    common: Any,
    *,
    folds: Sequence[Any],
    frame_ui: Any,
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    tail_quantile: float,
    chunk_frames: int,
    max_vram_bytes: int,
    heartbeat: Callable[[Mapping[str, Any]], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    rows: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    complete = 0
    for representation_name in SCREEN_REPRESENTATIONS:
        representation = _representation(common, representation_name)
        legacy_spec = GammaReferenceSpec.legacy_exact(epsilon=64.0)
        torch.cuda.reset_peak_memory_stats(representation.device)
        result = _evaluate_fixed_score_control(
            representation,
            representation_name=representation_name,
            control_id=CONTROLS[0].control_id,
            operation=lambda r=representation, s=legacy_spec: gamma_local_standardization(
                r, s, chunk_frames=chunk_frames, return_statistics=False
            ).values,
            folds=folds,
            frame_ui=frame_ui,
            bursts=bursts,
            tail_quantile=tail_quantile,
        )
        peak = int(torch.cuda.max_memory_allocated(representation.device))
        if peak > max_vram_bytes:
            raise RuntimeError(f"legacy control exceeded VRAM cap: {peak} > {max_vram_bytes}")
        rows.extend(result[0]); swaps.extend(result[1]); timings.append({**result[2], "max_memory_allocated_bytes": peak})
        complete += 1; heartbeat({"stage": "controls", "completed_control_arm_maps": complete, "total_control_arm_maps": 9})
        torch.cuda.reset_peak_memory_stats(representation.device)
        result = _evaluate_moments_control(
            representation,
            representation_name=representation_name,
            control_id=CONTROLS[1].control_id,
            folds=folds,
            frame_ui=frame_ui,
            bursts=bursts,
            scale_floor_percentile=scale_floor_percentile,
            tail_quantile=tail_quantile,
            chunk_frames=chunk_frames,
        )
        peak = int(torch.cuda.max_memory_allocated(representation.device))
        if peak > max_vram_bytes:
            raise RuntimeError(f"signed square control exceeded VRAM cap: {peak} > {max_vram_bytes}")
        rows.extend(result[0]); swaps.extend(result[1]); timings.append({**result[2], "max_memory_allocated_bytes": peak})
        complete += 1; heartbeat({"stage": "controls", "completed_control_arm_maps": complete, "total_control_arm_maps": 9})
        torch.cuda.reset_peak_memory_stats(representation.device)
        result = _evaluate_fixed_score_control(
            representation,
            representation_name=representation_name,
            control_id=CONTROLS[2].control_id,
            operation=lambda r=representation: _maintained_positive_box_score(
                r, chunk_frames=chunk_frames
            ),
            folds=folds,
            frame_ui=frame_ui,
            bursts=bursts,
            tail_quantile=tail_quantile,
        )
        peak = int(torch.cuda.max_memory_allocated(representation.device))
        if peak > max_vram_bytes:
            raise RuntimeError(f"maintained box control exceeded VRAM cap: {peak} > {max_vram_bytes}")
        rows.extend(result[0]); swaps.extend(result[1]); timings.append({**result[2], "max_memory_allocated_bytes": peak})
        complete += 1; heartbeat({"stage": "controls", "completed_control_arm_maps": complete, "total_control_arm_maps": 9})
        del representation
    return rows, swaps, timings


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(np.max(array)),
        "mean_ms": float(np.mean(array)),
    }


def _benchmark_operation(
    device: Any,
    operation: Callable[[], Any],
    *,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, float], int]:
    import torch

    for _ in range(warmups):
        value = operation()
        del value
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        value, elapsed = _elapsed(device, operation)
        timings.append(elapsed)
        del value
    torch.cuda.synchronize(device)
    return _percentiles(timings), int(torch.cuda.max_memory_allocated(device))


def _repeated_latency(
    common: Any,
    *,
    contexts: Sequence[GammaContext],
    config: GammaSupportConfig,
) -> list[dict[str, Any]]:
    raw = _representation(common, "raw")[:1]
    warmups = int(config.payload["latency"]["warmup_iterations"])
    iterations = int(config.payload["latency"]["timed_iterations"])
    rows = []
    seen = set()
    for context in sorted(contexts, key=lambda row: (row.half_width_px, row.context_id)):
        if context.half_width_px in seen:
            continue
        seen.add(context.half_width_px)
        spec = GammaReferenceSpec.from_mode(
            context.context_id,
            support_width_px=2 * context.half_width_px + 1,
            shape_n=context.shape,
            mode_radius_px=context.mode_radius_px,
            guard_radius_px=context.guard_radius_px,
            support_geometry="disk",
            boundary_mode="valid_renormalized_zero",
            epsilon=GAMMA_EPSILON,
        )
        stats, peak = _benchmark_operation(
            raw.device,
            lambda r=raw, s=spec: gamma_local_standardization(
                r, s, chunk_frames=1, return_statistics=False
            ).values,
            warmups=warmups,
            iterations=iterations,
        )
        rows.append(
            {
                "family": "radial_gamma_ls",
                "context_id": context.context_id,
                "half_width_px": context.half_width_px,
                "support_width_px": 2 * context.half_width_px + 1,
                "warmup_iterations": warmups,
                "timed_iterations": iterations,
                **stats,
                "max_memory_allocated_bytes": peak,
            }
        )
    control_operations = (
        (
            CONTROLS[0],
            lambda: gamma_local_standardization(
                raw,
                GammaReferenceSpec.legacy_exact(epsilon=64.0),
                chunk_frames=1,
                return_statistics=False,
            ).values,
        ),
        (CONTROLS[1], lambda: _square_annulus_moments(raw)),
        (CONTROLS[2], lambda: _maintained_positive_box_score(raw)),
    )
    for control, operation in control_operations:
        stats, peak = _benchmark_operation(
            raw.device, operation, warmups=warmups, iterations=iterations
        )
        rows.append(
            {
                "family": control.family,
                "context_id": control.control_id,
                "half_width_px": 11,
                "support_width_px": 23,
                "warmup_iterations": warmups,
                "timed_iterations": iterations,
                **stats,
                "max_memory_allocated_bytes": peak,
            }
        )
    del raw
    radial = {int(row["half_width_px"]): row for row in rows if row["family"] == "radial_gamma_ls"}
    baseline = float(radial[11]["p50_ms"])
    latency = config.payload["latency"]
    # Keep a single strict TSV schema across primary radial rows and ineligible
    # controls.  Controls receive the same descriptive h11-relative timings,
    # but ``eligible_primary`` prevents those values from entering selection.
    for row in rows:
        increase = float(row["p50_ms"]) - baseline
        ratio = float(row["p50_ms"]) / baseline if baseline > 0 else math.inf
        row["p50_ratio_vs_h11"] = ratio
        row["p50_increase_vs_h11_ms"] = increase
        row["latency_gate_pass"] = (
            float(row["p99_ms"]) <= float(latency["single_frame_deadline_ms"])
            and (
                ratio <= float(latency["relative_p50_cap_vs_h11"])
                or increase <= float(latency["absolute_p50_increase_cap_ms"])
            )
        )
        row["eligible_primary"] = row["family"] == "radial_gamma_ls"
    return rows


def _merge_best(
    stage_a: Mapping[int, Mapping[int, Mapping[str, Any]]],
    stage_b: Mapping[int, Mapping[int, Mapping[str, Any]]] | None,
) -> dict[int, dict[int, dict[str, Any]]]:
    return {
        fold: {**dict(stage_a[fold]), **(dict(stage_b[fold]) if stage_b else {})}
        for fold in TRAINING_FOLDS
    }


def _context_record(context: GammaContext) -> dict[str, Any]:
    return context.as_dict()


def _fold_context_contract(
    *,
    best: Mapping[int, Mapping[int, Mapping[str, Any]]],
    original: Mapping[int, Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
    config: GammaSupportConfig,
) -> dict[str, Any]:
    stopping = config.payload["stopping_rule"]
    absolute = float(stopping["training_near_optimal_absolute_contrast"])
    relative = float(stopping["training_near_optimal_relative_fraction"])
    minimum_floor = float(stopping["minimum_representation_contrast_floor"])
    latency_by_width = {
        int(row["half_width_px"]): row
        for row in latency_rows
        if row["family"] == "radial_gamma_ls"
    }
    folds = []
    for fold in TRAINING_FOLDS:
        by_width = best[fold]
        training_best_width, training_best = sorted(
            by_width.items(),
            key=lambda item: (
                -float(item[1]["mean_positive_tail_contrast"]),
                -float(item[1]["minimum_representation_positive_tail_contrast"]),
                float(item[1]["mean_runtime_ms_per_frame"]),
                item[1]["context"].context_id,
            ),
        )[0]
        tolerance = max(
            absolute, abs(float(training_best["mean_positive_tail_contrast"])) * relative
        )
        primary_pool = [
            (width, row)
            for width, row in by_width.items()
            if width <= PRIMARY_MAX_HALF_WIDTH
            and float(row["mean_positive_tail_contrast"])
            >= float(training_best["mean_positive_tail_contrast"]) - tolerance
            and float(row["minimum_representation_positive_tail_contrast"]) >= minimum_floor
        ]
        if primary_pool:
            candidate_width, candidate = sorted(primary_pool, key=lambda item: item[0])[0]
            candidate_status = "smallest_training_near_optimal_under_primary_size_cap"
        else:
            candidate_width, candidate = sorted(
                (
                    (width, row)
                    for width, row in by_width.items()
                    if width <= PRIMARY_MAX_HALF_WIDTH
                ),
                key=lambda item: (
                    -float(item[1]["mean_positive_tail_contrast"]),
                    -float(item[1]["minimum_representation_positive_tail_contrast"]),
                    item[0],
                ),
            )[0]
            candidate_status = "fallback_best_h_le_23_not_training_near_optimal"
        larger_width, larger = sorted(
            ((width, row) for width, row in by_width.items() if width > candidate_width),
            key=lambda item: (
                -float(item[1]["mean_positive_tail_contrast"]),
                -float(item[1]["minimum_representation_positive_tail_contrast"]),
                float(item[1]["mean_runtime_ms_per_frame"]),
                item[1]["context"].context_id,
            ),
        )[0]
        endpoint_width = max(by_width)
        endpoint = by_width[endpoint_width]

        def selected_payload(width: int, row: Mapping[str, Any], basis: str) -> dict[str, Any]:
            return {
                **_context_record(row["context"]),
                "selection_basis": basis,
                "mean_training_positive_tail_contrast": row["mean_positive_tail_contrast"],
                "minimum_representation_positive_tail_contrast": row[
                    "minimum_representation_positive_tail_contrast"
                ],
                "batch_gamma_runtime_ms_per_frame": row["mean_runtime_ms_per_frame"],
                "repeated_latency": dict(latency_by_width[width]),
            }

        folds.append(
            {
                "training_fold": fold,
                "heldout_burst": str(fold),
                "original_screen_context": dict(original[fold]),
                "support_candidate_context": selected_payload(
                    candidate_width, candidate, candidate_status
                ),
                "larger_support_comparator": selected_payload(
                    larger_width, larger, "best_training_contrast_above_support_candidate"
                ),
                "training_best_context": selected_payload(
                    training_best_width, training_best, "best_training_contrast_all_tested_widths"
                ),
                "max_support_endpoint": selected_payload(
                    endpoint_width, endpoint, "predeclared_largest_tested_support"
                ),
                "training_near_optimal_tolerance": tolerance,
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
            }
        )
    return {
        "schema_version": 1,
        "selection_scope": "outer_training_fold_only",
        "selection_uses_positive_coordinates": False,
        "selection_uses_positive_identities": False,
        "burst_windows_used": True,
        "folds": folds,
    }


def run_support_screen(
    config: GammaSupportConfig,
    *,
    base_preflight_dir: str | Path,
    support_preflight_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the adaptive support screen and atomically commit its evidence."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"support screen output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"support screen parent is missing: {destination.parent}")
    preflight = _verify_support_preflight(
        config,
        base_preflight_dir=base_preflight_dir,
        support_preflight_dir=support_preflight_dir,
    )
    requested = str(device or config.payload["resources"]["device"])
    try:
        runtime = require_cuda_device(requested)
    except CudaRuntimeUnavailable as error:
        raise GammaSupportUnavailable(str(error)) from error
    source = _source_contract(config.base_config)
    partial = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    if partial.exists():
        raise FileExistsError(f"support screen partial collision: {partial}")
    try:
        partial.mkdir()

        def heartbeat(payload: Mapping[str, Any]) -> None:
            _atomic_json(
                partial / "heartbeat.json",
                {"updated_at_utc": datetime.now(timezone.utc).isoformat(), **payload},
            )

        heartbeat({"status": "running", "stage": "causal_preprocessing"})
        import torch

        torch_device = torch.device(str(runtime["resolved_device"]))
        base = config.base_config
        start_ui, stop_ui = map(int, base.payload["frames"]["review_interval_ui"])
        common, history_timing = _stream_common_history_to_device(
            base.source_paths["movie"],
            review_start_ui=start_ui,
            review_stop_ui=stop_ui,
            chunk_frames=int(config.payload["resources"]["chunk_frames"]),
            device=torch_device,
            heartbeat=heartbeat,
        )
        frame_ui = torch.arange(start_ui, stop_ui + 1, device=torch_device, dtype=torch.int64)
        folds = build_fold_contracts(base)
        bursts = base.payload["frames"]["burst_intervals_ui"]
        scale_percentile = float(base.payload["gamma_ls_grid"]["screen_scale_floor_percentile"])
        tail_quantile = float(base.payload["screen"]["positive_tail_quantile"])
        chunk_frames = int(config.payload["resources"]["chunk_frames"])
        max_vram = int(float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30)

        stage_a_contexts = enumerate_stage_contexts("support_a")
        a_rows, a_swaps, a_timings, a_peak = _run_contexts(
            common,
            contexts=stage_a_contexts,
            folds=folds,
            frame_ui=frame_ui,
            bursts=bursts,
            scale_floor_percentile=scale_percentile,
            tail_quantile=tail_quantile,
            chunk_frames=chunk_frames,
            max_vram_bytes=max_vram,
            heartbeat=heartbeat,
        )
        stage_a_best = best_context_by_fold_and_width(a_rows, stage_a_contexts)
        stopping = config.payload["stopping_rule"]
        expand, trigger_rows = stage_b_required(
            stage_a_best,
            absolute_tolerance=float(stopping["training_near_optimal_absolute_contrast"]),
            relative_tolerance=float(stopping["training_near_optimal_relative_fraction"]),
        )
        b_rows: list[dict[str, Any]] = []
        b_swaps: list[dict[str, Any]] = []
        b_timings: list[dict[str, Any]] = []
        b_peak: dict[int, int] = {}
        stage_b_best = None
        stage_b_contexts: tuple[GammaContext, ...] = ()
        if expand:
            stage_b_contexts = enumerate_stage_contexts("support_b")
            b_rows, b_swaps, b_timings, b_peak = _run_contexts(
                common,
                contexts=stage_b_contexts,
                folds=folds,
                frame_ui=frame_ui,
                bursts=bursts,
                scale_floor_percentile=scale_percentile,
                tail_quantile=tail_quantile,
                chunk_frames=chunk_frames,
                max_vram_bytes=max_vram,
                heartbeat=heartbeat,
            )
            stage_b_best = best_context_by_fold_and_width(b_rows, stage_b_contexts)

        control_rows, control_swaps, control_timings = _run_controls(
            common,
            folds=folds,
            frame_ui=frame_ui,
            bursts=bursts,
            scale_floor_percentile=scale_percentile,
            tail_quantile=tail_quantile,
            chunk_frames=chunk_frames,
            max_vram_bytes=max_vram,
            heartbeat=heartbeat,
        )
        all_contexts = stage_a_contexts + stage_b_contexts
        latency_rows = _repeated_latency(common, contexts=all_contexts, config=config)
        all_best = _merge_best(stage_a_best, stage_b_best)
        boundary = support_boundary_status(
            all_best,
            stage_b_run=expand,
            absolute_tolerance=float(stopping["training_near_optimal_absolute_contrast"]),
            relative_tolerance=float(stopping["training_near_optimal_relative_fraction"]),
        )
        fold_contexts = _fold_context_contract(
            best=all_best,
            original=preflight["original_fold_contexts"],
            latency_rows=latency_rows,
            config=config,
        )
        expected = design_counts(stage_b_run=expand)
        all_rows = a_rows + b_rows
        all_swaps = a_swaps + b_swaps
        if len(all_rows) != expected["eligible_fold_metric_cells"] or len(all_swaps) != expected[
            "eligible_quiet_swap_rows"
        ]:
            raise AssertionError("eligible support screen cardinality changed")
        if len(control_rows) != expected["diagnostic_control_fold_metric_cells"] or len(
            control_swaps
        ) != expected["diagnostic_control_quiet_swap_rows"]:
            raise AssertionError("diagnostic control cardinality changed")

        _atomic_tsv(partial / "eligible_metric_rows.tsv", all_rows)
        _atomic_tsv(partial / "eligible_quiet_swap_rows.tsv", all_swaps)
        _atomic_tsv(partial / "control_metric_rows.tsv", control_rows)
        _atomic_tsv(partial / "control_quiet_swap_rows.tsv", control_swaps)
        _atomic_tsv(partial / "repeated_single_frame_latency.tsv", latency_rows)
        _atomic_json(partial / "fold_contexts.json", fold_contexts)
        _atomic_json(
            partial / "boundary_decision.json",
            {
                "stage_b_trigger_rule": stopping["stage_b_trigger"],
                "stage_b_triggered": expand,
                "stage_a_trigger_diagnostics": trigger_rows,
                "support_boundary": boundary,
                "final_sufficiency_status": "requires_protected_recall_sensitivity",
                "final_sufficiency_rule": {
                    "training": stopping,
                    "protected": {
                        "b58_total_match_loss_at_most": stopping[
                            "protected_b58_total_match_tolerance"
                        ],
                        "per_burst_match_loss_at_most": stopping[
                            "protected_per_burst_match_tolerance"
                        ],
                        "budget_curve_auc_loss_at_most": stopping[
                            "protected_budget_curve_auc_tolerance"
                        ],
                    },
                    "latency": config.payload["latency"],
                },
            },
        )
        timings = {
            **history_timing,
            "eligible_context_arm_maps": a_timings + b_timings,
            "diagnostic_control_arm_maps": control_timings,
            "peak_memory_allocated_by_half_width_bytes": {
                str(key): value for key, value in sorted({**a_peak, **b_peak}.items())
            },
            "repeated_single_frame_latency": latency_rows,
        }
        _atomic_json(partial / "timings.json", timings)
        claim_boundary = {
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "burst_windows_used": True,
            "fully_label_free_claimed": False,
            "protected_recall_computed": False,
            "support_sufficiency_claimed": False,
            "controls_eligible_for_primary_selection": False,
            "scientific_audit_complete": False,
            "interpretation": (
                "Training-window contrast and repeated Gamma-stage latency are complete. "
                "A support-sufficiency claim remains prohibited until the frozen fold contexts "
                "receive protected recall sensitivity and the downstream audit is complete."
            ),
        }
        _atomic_json(partial / "claim_boundary.json", claim_boundary)
        summary = {
            "schema_version": 1,
            "experiment_id": config.payload["experiment_id"],
            "status": "complete_support_screen_only",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "runtime": runtime,
            "design_counts": expected,
            "stage_b_triggered": expand,
            "largest_tested_half_width_px": STAGE_B_ENDPOINT if expand else STAGE_A_ENDPOINT,
            "support_boundary": boundary,
            "fold_contexts": "fold_contexts.json",
            "named_controls_executed": [control.as_dict() for control in CONTROLS],
            "claim_boundary": claim_boundary,
        }
        _atomic_json(partial / "summary.json", summary)
        _atomic_json(
            partial / "provenance_hashes.json",
            {
                "portable_config_sha256": _canonical_sha256(config.portable_dict()),
                "support_preflight": preflight,
                "implementation": _implementation_hashes(config),
                "movie_sha256": source.get("movie_sha256"),
            },
        )
        _atomic_json(
            partial / "validation.json",
            {
                "status": "passed_support_screen_artifact_contract_scientific_audit_pending",
                "checks": {
                    "cuda_required_and_used": True,
                    "eligible_cell_count_exact": len(all_rows) == expected[
                        "eligible_fold_metric_cells"
                    ],
                    "control_cell_count_exact": len(control_rows) == expected[
                        "diagnostic_control_fold_metric_cells"
                    ],
                    "three_named_controls_executed": len(control_timings) == 9,
                    "controls_ineligible_for_primary_selection": True,
                    "fold_local_contexts_exposed": len(fold_contexts["folds"]) == 4,
                    "adaptive_stage_b_rule_applied": True,
                    "positive_coordinates_used": False,
                    "positive_identities_used": False,
                    "protected_recall_pending": True,
                    "scientific_audit_pending": True,
                },
            },
        )
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": "radial context x fixed representation x outer training fold",
                "fold_context_contract": "fold_contexts.json",
                "boundary_decision": "boundary_decision.json",
                "metrics": ["eligible_metric_rows.tsv", "control_metric_rows.tsv"],
                "latency": "repeated_single_frame_latency.tsv",
                "labels": "burst windows used; sparse-positive coordinates and identities not opened",
                "scientific_interpretation": "support sufficiency pending protected recall",
            },
        )
        report = (
            "# Gamma-LS radial-support sufficiency screen\n\n"
            f"Status: `{summary['status']}`.\n\n"
            f"Stage B (h39/h47) triggered: `{str(expand).lower()}`.  Largest tested half-width: "
            f"`{summary['largest_tested_half_width_px']}` px.\n\n"
            "The run used human-declared burst windows but no sparse-positive coordinates or "
            "identities. The legacy exact, signed square-annulus, and maintained positive box-CFAR "
            "controls were executed and were ineligible for radial Gamma selection.\n\n"
            "No support-sufficiency claim is made here. Read `fold_contexts.json`, apply its frozen "
            "contexts to protected recall sensitivity, and combine that result with "
            "`repeated_single_frame_latency.tsv` under `boundary_decision.json`.\n"
        )
        (partial / "REPORT.md").write_text(report, encoding="utf-8")
        heartbeat({"status": "complete_support_screen_only", "stage": "complete"})
        _atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        if destination.exists():
            raise FileExistsError("support output appeared during execution")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gamma-LS radial-support sufficiency extension")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--base-preflight-dir", required=True)
    preflight.add_argument("--artifact-dir", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--base-preflight-dir", required=True)
    run.add_argument("--support-preflight-dir", required=True)
    run.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = GammaSupportConfig.load(args.config)
    try:
        if args.command == "preflight":
            payload = run_support_preflight(
                config,
                base_preflight_dir=args.base_preflight_dir,
                artifact_dir=args.artifact_dir,
            )
        else:
            payload = run_support_screen(
                config,
                base_preflight_dir=args.base_preflight_dir,
                support_preflight_dir=args.support_preflight_dir,
                output_dir=args.output_dir,
            )
    except (RuntimeError, ValueError, OSError, GammaSupportUnavailable) as error:
        print(
            json.dumps(
                {
                    "status": "blocked_support_preflight_or_runtime",
                    "output_mutated": False,
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GammaSupportUnavailable",
    "run_support_preflight",
    "run_support_screen",
]
