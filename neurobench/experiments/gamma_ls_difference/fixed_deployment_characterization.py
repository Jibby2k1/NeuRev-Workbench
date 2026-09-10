"""Hash-bound fixed-deployment replay for protected within-recording audit media.

This module is deliberately *not* a selection runner.  It replays the already
frozen signed-adjacent-difference plus global radial Gamma-LS h15 context over
the four configured burst windows.  Both predeclared quiet cross-fit swaps,
all five empirical quiet burdens, deterministic NMS6, and the existing
candidate budgets are retained unchanged.

Dense conditioned, difference, Gamma-score, occupancy, and candidate-peak
evidence is sealed before the protected-v1 label table is parsed.  The joined
result is post-selection, within-recording characterization; it is neither a
protected selection estimate nor independent confirmation.
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
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)

from .config import GammaLSDifferenceConfig
from .cuda_runtime import require_cuda_device
from .evaluation import (
    CANDIDATE_BUDGETS_PER_BURST,
    MATCH_RADIUS_PX,
    NMS_DISTANCE_PX,
    QUIET_NMS_PEAK_BURDENS,
    calibrate_training_quiet_thresholds,
    duration_matched_quiet_windows,
    extract_burst_candidates,
)
from .full_recording import (
    _artifact_index,
    _atomic_json,
    _atomic_text,
    _atomic_tsv,
    _canonical_sha256,
    _sha256,
    _verify_support_context,
    verify_indexed_artifact,
)
from .preflight import verify_matching_preflight
from .protected import (
    QUIET_SWAPS,
    _read_sparse_positives,
    aggregate_match_rows,
    observation_match_rows,
)
from .screen import (
    GAMMA_EPSILON,
    _elapsed,
    _positive_scale_floor,
    _stream_common_history_to_device,
    build_fold_contracts,
)
from . import gpu_representations


EXPERIMENT_ID = "spon_ca_burst_gamma_ls_fixed_deployment_characterization_v1"
CONTEXT_ID = "support_support_a_h15_g7_n9_m0p5"
REPRESENTATION = "difference_signed"
CONTEXT_ROLE = "frozen_global_deployment_context"
COHORT = "protected_v1_post_selection_within_recording_characterization"
EXPECTED_REVIEW_INTERVAL_UI = (1800, 2359)
EXPECTED_MOVIE_SHAPE = (2359, 340, 573)
EXPECTED_PROTECTED_V1_ROWS = 79
SCALE_FLOOR_PERCENTILE = 10.0
DEFAULT_CHUNK_FRAMES = 64
DEFAULT_AUDIT_SWAP = "a_train_b_test"
DEFAULT_AUDIT_BURDEN = 1.0
PRIMARY_AUDIT_BUDGET = 58
STAGE_ARRAY_FILES = {
    "conditioned_current_frame": "stage_arrays/conditioned_current_frame.npy",
    "difference_signed": "stage_arrays/difference_signed.npy",
    "gamma_a_train_b_test": "stage_arrays/gamma_a_train_b_test.npy",
    "gamma_b_train_a_test": "stage_arrays/gamma_b_train_a_test.npy",
    "occupancy_maps": "stage_arrays/occupancy_maps.npy",
}


class FixedDeploymentCharacterizationUnavailable(RuntimeError):
    """Raised when a frozen prerequisite or isolation invariant is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedDeploymentCharacterizationUnavailable(
            f"JSON object required: {path}"
        )
    return value


def _implementation_files(repository: Path) -> dict[str, dict[str, Any]]:
    relatives = (
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/metrics/sparse_detection.py",
        "neurobench/experiments/gamma_ls_difference/config.py",
        "neurobench/experiments/gamma_ls_difference/cuda_runtime.py",
        "neurobench/experiments/gamma_ls_difference/evaluation.py",
        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
        "neurobench/experiments/gamma_ls_difference/preflight.py",
        "neurobench/experiments/gamma_ls_difference/protected.py",
        "neurobench/experiments/gamma_ls_difference/screen.py",
        "neurobench/experiments/gamma_ls_difference/full_recording.py",
        "neurobench/experiments/gamma_ls_difference/scientific_audit.py",
        "neurobench/experiments/gamma_ls_difference/fixed_deployment_characterization.py",
        "tests/test_gamma_ls_fixed_deployment_characterization.py",
    )
    output: dict[str, dict[str, Any]] = {}
    for relative in relatives:
        path = repository / relative
        if not path.is_file():
            raise FixedDeploymentCharacterizationUnavailable(
                f"implementation input is missing: repo://{relative}"
            )
        output[f"repo://{relative}"] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return output


def _science_contract(
    *, audit_swap: str = DEFAULT_AUDIT_SWAP, audit_burden: float = DEFAULT_AUDIT_BURDEN
) -> dict[str, Any]:
    if audit_swap not in QUIET_SWAPS:
        raise ValueError(f"audit_swap must be one of {QUIET_SWAPS}")
    burden = float(audit_burden)
    if burden not in QUIET_NMS_PEAK_BURDENS:
        raise ValueError(
            f"audit_burden must be one of {QUIET_NMS_PEAK_BURDENS}"
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "evidence_role": "post_selection_within_recording_characterization",
        "protected_selection_estimate": False,
        "independent_confirmation": False,
        "representation": REPRESENTATION,
        "context_role": CONTEXT_ROLE,
        "context": {
            "context_id": CONTEXT_ID,
            "support_width_px": 31,
            "half_width_px": 15,
            "guard_radius_px": 7,
            "shape_n": 9.0,
            "mode_radius_px": 7.5,
            "support_geometry": "disk",
            "boundary_mode": "valid_renormalized_zero",
            "epsilon": GAMMA_EPSILON,
        },
        "review_interval_ui": list(EXPECTED_REVIEW_INTERVAL_UI),
        "quiet_cross_fit_swaps": list(QUIET_SWAPS),
        "quiet_nms_peaks_per_pseudo_burst": list(QUIET_NMS_PEAK_BURDENS),
        "scale_floor_percentile": SCALE_FLOOR_PERCENTILE,
        "nms_distance_px": NMS_DISTANCE_PX,
        "candidate_budgets_per_burst": list(CANDIDATE_BUDGETS_PER_BURST),
        "primary_candidate_budget_per_burst": PRIMARY_AUDIT_BUDGET,
        "audit_representative_state": {
            "quiet_swap": audit_swap,
            "target_nms_peaks_per_pseudo_burst": burden,
            "candidate_budget_per_burst": PRIMARY_AUDIT_BUDGET,
            "selection_basis": "pre_label_fixed_display_state_only_not_metric_selection",
        },
        "candidate_generation_uses_positive_coordinates": False,
        "candidate_generation_uses_positive_identities": False,
        "temporal_aggregation": "threshold_occupancy_within_each_declared_burst",
        "temporal_max_pooling_used": False,
        "full_record_q1_proposal_stream_covered": False,
    }


def _verify_exact_truth_gate(root: str | Path) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    indexed = verify_indexed_artifact(directory)
    validation = _read_json(directory / "validation.json")
    summary = _read_json(directory / "summary.json")
    checks = validation.get("checks", {})
    required = (
        "exact_truth_rows_complete",
        "no_temporal_pooling",
        "calibration_source_count_zero",
        "thresholds_not_keyed_by_source_count",
    )
    if not validation.get("all_checks_pass") or not all(
        checks.get(key) is True for key in required
    ):
        raise FixedDeploymentCharacterizationUnavailable(
            "exact-truth/NMS validation gate is not complete"
        )
    pipelines = summary.get("design", {}).get("pipelines", [])
    expected = "difference_signed__radial_gamma_h15_g7_n9_m0p5"
    if expected not in pipelines:
        raise FixedDeploymentCharacterizationUnavailable(
            "exact-truth artifact lacks the frozen signed h15 pipeline"
        )
    return {
        **indexed,
        "validation_sha256": _sha256(directory / "validation.json"),
        "summary_sha256": _sha256(directory / "summary.json"),
        "required_checks": list(required),
        "scientific_audit_pending_in_source": bool(
            checks.get("scientific_audit_pending")
        ),
        "role": "exact_NMS_and_synthetic_truth_gate_not_selection_input",
    }


def _expected_resource_footprint() -> dict[str, Any]:
    frames = EXPECTED_REVIEW_INTERVAL_UI[1] - EXPECTED_REVIEW_INTERVAL_UI[0] + 1
    frame_bytes = frames * EXPECTED_MOVIE_SHAPE[1] * EXPECTED_MOVIE_SHAPE[2] * 4
    occupancy_bytes = (
        len(QUIET_SWAPS)
        * len(QUIET_NMS_PEAK_BURDENS)
        * 4
        * EXPECTED_MOVIE_SHAPE[1]
        * EXPECTED_MOVIE_SHAPE[2]
        * 4
    )
    return {
        "stage_float32_array_count": 4,
        "bytes_per_560_frame_float32_array": frame_bytes,
        "conditioned_difference_and_two_gamma_bytes": 4 * frame_bytes,
        "occupancy_maps_bytes": occupancy_bytes,
        "minimum_stage_array_bytes": 4 * frame_bytes + occupancy_bytes,
        "minimum_stage_array_gib": (4 * frame_bytes + occupancy_bytes) / 2**30,
        "recommended_free_disk_gib_before_run": 4.0,
        "configured_peak_vram_cap_gib": 8.0,
        "configured_peak_ram_cap_gib": 24.0,
        "media_not_included_in_estimate": True,
    }


def write_replay_preflight(
    config_path: str | Path,
    *,
    base_preflight_dir: str | Path,
    support_dir: str | Path,
    exact_truth_dir: str | Path,
    output_dir: str | Path,
    audit_swap: str = DEFAULT_AUDIT_SWAP,
    audit_burden: float = DEFAULT_AUDIT_BURDEN,
) -> dict[str, Any]:
    """Freeze all replay inputs without parsing either annotation table."""

    config = GammaLSDifferenceConfig.load(config_path)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    contract = _science_contract(audit_swap=audit_swap, audit_burden=audit_burden)
    base_root = Path(base_preflight_dir).expanduser().resolve()
    base_indexed = verify_indexed_artifact(base_root)
    base = verify_matching_preflight(config, base_root, require_gpu_ready=True)
    reference, support = _verify_support_context(support_dir, CONTEXT_ID)
    exact_truth = _verify_exact_truth_gate(exact_truth_dir)
    expected_context = contract["context"]
    observed_context = {
        "context_id": reference.context_id,
        "support_width_px": reference.support_width_px,
        "guard_radius_px": int(reference.guard_radius_px),
        "shape_n": float(reference.shape_n),
        "mode_radius_px": float(reference.nominal_mode_radius_px or math.nan),
        "support_geometry": reference.support_geometry,
        "boundary_mode": reference.boundary_mode,
        "epsilon": float(reference.epsilon),
    }
    for key in (
        "context_id", "support_width_px", "guard_radius_px", "shape_n",
        "mode_radius_px", "support_geometry", "boundary_mode", "epsilon",
    ):
        if observed_context[key] != expected_context[key]:
            raise FixedDeploymentCharacterizationUnavailable(
                f"frozen context geometry changed at {key}"
            )
    source_shape = tuple(base["source"]["movie"]["shape"])
    if source_shape != EXPECTED_MOVIE_SHAPE:
        raise FixedDeploymentCharacterizationUnavailable(
            f"movie shape changed: {source_shape}"
        )

    work = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        payload = {
            "schema_version": 1,
            "status": "ready_hash_bound_gpu_run_not_started",
            "created_at_utc": _utc_now(),
            "science_contract": contract,
            "science_contract_sha256": _canonical_sha256(contract),
            "inputs": {
                "base_preflight": {
                    **base_indexed,
                    "preflight_sha256": _sha256(base_root / "preflight.json"),
                    "source_movie_sha256": base["source"]["movie"]["sha256"],
                    "protected_v1_sha256": base["source"]["protected_labels_v1"]["sha256"],
                    "annotation_content_parsed": False,
                },
                "support_screen": support,
                "exact_truth_nms_gate": exact_truth,
            },
            "implementation": _implementation_files(config.repository),
            "resources": _expected_resource_footprint(),
            "execution_gate": {
                "gpu_authorization_required_at_run": True,
                "no_gpu_work_performed_by_preflight": True,
                "resumable_work_directory": f".{destination.name}.fixed-deployment-work",
            },
            "label_isolation": {
                "protected_v1_path_known_from_portable_manifest": True,
                "protected_v1_bytes_hashed_by_upstream_preflight": True,
                "protected_v1_rows_parsed": False,
                "candidate_and_stage_seal_required_before_parse": True,
            },
            "full_record_audit_gap": {
                "headline": "371 proposals across 2259 application frames at the full-record q=1 state",
                "covered_by_this_replay": False,
                "reason": "this replay is restricted to UI 1800--2359 and burst-level quiet cross-fit",
                "required_separate_action": "model-only full-record stage replay/audit or explicit readiness-gap artifact",
            },
        }
        _atomic_json(work / "config.portable.json", config.portable_dict())
        _atomic_json(work / "preflight.json", payload)
        _atomic_json(
            work / "validation.json",
            {
                "schema_version": 1,
                "status": "ready_hash_bound_gpu_run_not_started",
                "all_checks_pass": True,
                "annotation_rows_parsed": False,
                "gpu_work_performed": False,
                "exact_truth_nms_gate_passed": True,
                "context_exact": True,
                "science_contract_frozen": True,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
    except BaseException:
        raise
    return {
        "output": str(destination),
        "status": payload["status"],
        "science_contract_sha256": payload["science_contract_sha256"],
        "minimum_stage_array_gib": payload["resources"]["minimum_stage_array_gib"],
    }


def _verify_replay_preflight(
    config: GammaLSDifferenceConfig, preflight_dir: str | Path
) -> dict[str, Any]:
    root = Path(preflight_dir).expanduser().resolve()
    indexed = verify_indexed_artifact(root)
    payload = _read_json(root / "preflight.json")
    validation = _read_json(root / "validation.json")
    portable = _read_json(root / "config.portable.json")
    if portable != config.portable_dict():
        raise FixedDeploymentCharacterizationUnavailable(
            "replay preflight portable config changed"
        )
    if not validation.get("all_checks_pass") or payload.get("status") != (
        "ready_hash_bound_gpu_run_not_started"
    ):
        raise FixedDeploymentCharacterizationUnavailable(
            "replay preflight is not run-ready"
        )
    expected_contract = _science_contract(
        audit_swap=payload["science_contract"]["audit_representative_state"]["quiet_swap"],
        audit_burden=float(
            payload["science_contract"]["audit_representative_state"][
                "target_nms_peaks_per_pseudo_burst"
            ]
        ),
    )
    if payload.get("science_contract") != expected_contract or payload.get(
        "science_contract_sha256"
    ) != _canonical_sha256(expected_contract):
        raise FixedDeploymentCharacterizationUnavailable(
            "replay science contract changed"
        )
    if payload.get("implementation") != _implementation_files(config.repository):
        raise FixedDeploymentCharacterizationUnavailable(
            "implementation changed after replay preflight"
        )
    base = payload["inputs"]["base_preflight"]
    base_root = Path(base["root"])
    if verify_indexed_artifact(base_root)["artifact_index_sha256"] != base[
        "artifact_index_sha256"
    ]:
        raise FixedDeploymentCharacterizationUnavailable("base preflight changed")
    verify_matching_preflight(config, base_root, require_gpu_ready=True)
    support = payload["inputs"]["support_screen"]
    _, observed_support = _verify_support_context(support["root"], CONTEXT_ID)
    if observed_support["artifact_index_sha256"] != support["artifact_index_sha256"]:
        raise FixedDeploymentCharacterizationUnavailable("support artifact changed")
    exact = payload["inputs"]["exact_truth_nms_gate"]
    observed_exact = _verify_exact_truth_gate(exact["root"])
    if observed_exact["artifact_index_sha256"] != exact["artifact_index_sha256"]:
        raise FixedDeploymentCharacterizationUnavailable("exact-truth gate changed")
    return {**payload, "replay_preflight": indexed}


def _reference() -> GammaReferenceSpec:
    return GammaReferenceSpec.from_mode(
        CONTEXT_ID,
        support_width_px=31,
        shape_n=9.0,
        mode_radius_px=7.5,
        guard_radius_px=7.0,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=GAMMA_EPSILON,
        scale_floor=0.0,
    )


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npy")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _new_npy_memmap(
    path: Path, *, shape: Sequence[int], dtype: np.dtype[Any] | str
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npy")
    if temporary.exists():
        raise FixedDeploymentCharacterizationUnavailable(
            f"unresolved partial array exists: {temporary}"
        )
    return np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
    )


def _seal_memmap(path: Path, array: np.memmap) -> None:
    array.flush()
    temporary = Path(str(array.filename)).resolve()
    del array
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _array_description(path: Path) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.as_posix(),
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "finite": bool(np.isfinite(values).all()),
    }


def _verify_stage_manifest(work: Path) -> dict[str, Any]:
    manifest_path = work / "stage_array_manifest.json"
    manifest = _read_json(manifest_path)
    expected_shape = [560, 340, 573]
    for key in (
        "conditioned_current_frame",
        "difference_signed",
        "gamma_a_train_b_test",
        "gamma_b_train_a_test",
    ):
        row = manifest.get("arrays", {}).get(key)
        if not isinstance(row, Mapping):
            raise FixedDeploymentCharacterizationUnavailable(
                f"stage manifest missing {key}"
            )
        path = work / STAGE_ARRAY_FILES[key]
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != expected_shape or str(values.dtype) != "float32":
            raise FixedDeploymentCharacterizationUnavailable(
                f"stage array contract changed: {key}"
            )
        if path.stat().st_size != int(row["size_bytes"]) or _sha256(path) != row[
            "sha256"
        ]:
            raise FixedDeploymentCharacterizationUnavailable(
                f"stage array changed after checkpoint: {key}"
            )
    return manifest


def _interval_mask(frame_ui: np.ndarray, interval: Sequence[int]) -> np.ndarray:
    start, stop = map(int, interval)
    return (frame_ui >= start) & (frame_ui <= stop)


def _ui_window(
    interval: Sequence[int], *, review_start_ui: int
) -> tuple[int, int]:
    start, stop = map(int, interval)
    return start - review_start_ui, stop - review_start_ui + 1


def _duration_map(bursts: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {
        str(key): int(bounds[1]) - int(bounds[0]) + 1
        for key, bounds in sorted(bursts.items())
    }


def _heartbeat(work: Path, stage: str, **details: Any) -> None:
    _atomic_json(
        work / "heartbeat.json",
        {"updated_at_utc": _utc_now(), "stage": stage, **details},
    )


def _write_stage_arrays(
    config: GammaLSDifferenceConfig,
    work: Path,
    *,
    device_name: str,
    chunk_frames: int,
) -> dict[str, Any]:
    """Run the one authorized CUDA replay and checkpoint the four dense arrays."""

    import torch

    runtime = require_cuda_device(device_name)
    device = torch.device(runtime["resolved_device"])
    torch.set_num_threads(int(config.payload["resources"]["cpu_threads"]))
    if chunk_frames not in tuple(map(int, config.payload["efficiency"]["frame_chunks"])):
        raise ValueError("chunk_frames must be one of the frozen efficiency chunks")
    review_start, review_stop = map(
        int, config.payload["frames"]["review_interval_ui"]
    )
    if (review_start, review_stop) != EXPECTED_REVIEW_INTERVAL_UI:
        raise FixedDeploymentCharacterizationUnavailable("review interval changed")
    _heartbeat(work, "cuda_causal_preprocessing", device=runtime["resolved_device"])
    torch.cuda.reset_peak_memory_stats(device)
    common, preprocessing = _stream_common_history_to_device(
        config.source_paths["movie"],
        review_start_ui=review_start,
        review_stop_ui=review_stop,
        chunk_frames=chunk_frames,
        device=device,
        heartbeat=lambda row: _heartbeat(work, **dict(row)),
    )
    conditioned = common[1:]
    representation = gpu_representations.signed_difference_representation(common).values
    expected_shape = (560, 340, 573)
    if tuple(conditioned.shape) != expected_shape or tuple(representation.shape) != expected_shape:
        raise FixedDeploymentCharacterizationUnavailable(
            "conditioned or signed-difference stage shape changed"
        )
    _heartbeat(work, "persist_conditioned_and_difference")
    _atomic_npy(
        work / STAGE_ARRAY_FILES["conditioned_current_frame"],
        conditioned.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    _atomic_npy(
        work / STAGE_ARRAY_FILES["difference_signed"],
        representation.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    _heartbeat(work, "cuda_gamma_moments")
    moments, gamma_ms = _elapsed(
        device,
        lambda: gamma_local_standardization(
            representation,
            _reference(),
            chunk_frames=chunk_frames,
            return_statistics=True,
        ),
    )
    if moments.local_mean is None or moments.local_std is None:
        raise AssertionError("Gamma-LS moments are required")
    frame_ui = np.arange(review_start, review_stop + 1, dtype=np.int64)
    frame_ui_device = torch.as_tensor(frame_ui, device=device)
    folds = build_fold_contracts(config)
    quiet_a, quiet_b = folds[0].quiet_half_a_ui, folds[0].quiet_half_b_ui
    floors: dict[str, float] = {}
    score_timings: dict[str, float] = {}
    for swap, floor_interval in (
        (QUIET_SWAPS[0], quiet_a),
        (QUIET_SWAPS[1], quiet_b),
    ):
        floor_mask = (
            (frame_ui_device >= int(floor_interval[0]))
            & (frame_ui_device <= int(floor_interval[1]))
        )
        floor = _positive_scale_floor(
            moments.local_std, floor_mask, SCALE_FLOOR_PERCENTILE
        )

        def make_score() -> Any:
            return (representation - moments.local_mean) / (
                torch.maximum(moments.local_std, floor) + GAMMA_EPSILON
            )

        score, elapsed = _elapsed(device, make_score)
        key = f"gamma_{swap}"
        _heartbeat(work, "persist_gamma_score", quiet_swap=swap)
        _atomic_npy(
            work / STAGE_ARRAY_FILES[key],
            score.detach().cpu().numpy().astype(np.float32, copy=False),
        )
        floors[swap] = float(floor.item())
        score_timings[swap] = float(elapsed)
        del score
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_after, total = torch.cuda.mem_get_info(device)
    del moments, representation, conditioned, common
    torch.cuda.empty_cache()
    arrays = {}
    for key in (
        "conditioned_current_frame",
        "difference_signed",
        "gamma_a_train_b_test",
        "gamma_b_train_a_test",
    ):
        path = work / STAGE_ARRAY_FILES[key]
        row = _array_description(path)
        row["path"] = path.relative_to(work).as_posix()
        arrays[key] = row
    manifest = {
        "schema_version": 1,
        "status": "complete_pre_label_stage_checkpoint",
        "created_at_utc": _utc_now(),
        "source_frame_ui": [review_start, review_stop],
        "frame_alignment": {
            "conditioned_current_frame": "common causal state at each source UI frame",
            "difference_signed": "conditioned[t]-conditioned[t-1] aligned to current UI frame",
            "gamma_scores": "signed Gamma-LS aligned one-to-one to difference_signed",
        },
        "arrays": arrays,
        "quiet_scale_floors": floors,
        "runtime": {
            "cuda": runtime,
            "preprocessing": preprocessing,
            "gamma_moments_ms": float(gamma_ms),
            "gamma_score_ms_by_swap": score_timings,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "free_vram_bytes_after": int(free_after),
            "total_vram_bytes": int(total),
        },
        "labels_parsed": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }
    _atomic_json(work / "stage_array_manifest.json", manifest)
    return manifest


CANDIDATE_FIELDS = (
    "candidate_id",
    "training_scope",
    "burst_id",
    "context_role",
    "context_id",
    "representation",
    "quiet_swap",
    "nms_distance_px",
    "nms_role",
    "target_nms_peaks_per_pseudo_burst",
    "threshold_z",
    "scale_floor_percentile",
    "scale_floor",
    "candidate_rank",
    "occupancy_score",
    "x_px",
    "y_px",
    "peak_frame_review_zero",
    "peak_source_frame_ui",
    "peak_gamma_score_z",
    "threshold_exceedance_frame_count",
    "interpretation_before_label_join",
)

CALIBRATION_FIELDS = (
    "training_scope",
    "context_role",
    "context_id",
    "representation",
    "quiet_swap",
    "nms_distance_px",
    "nms_role",
    "floor_interval_ui",
    "test_interval_ui",
    "scale_floor_percentile",
    "scale_floor",
    "target_nms_peaks_per_pseudo_burst",
    "threshold_z",
    "calibration_achieved_peaks_per_pseudo_burst",
    "heldout_quiet_peaks_per_pseudo_burst",
    "all_four_burst_candidate_count",
    "probability_of_false_alarm_claimed",
    "positive_coordinates_used",
    "positive_identities_used",
)

OCCUPANCY_INDEX_FIELDS = (
    "occupancy_swap_index",
    "occupancy_burden_index",
    "occupancy_burst_index",
    "quiet_swap",
    "target_nms_peaks_per_pseudo_burst",
    "burst_id",
    "source_start_ui",
    "source_stop_ui",
    "threshold_z",
    "map_semantics",
)


def _candidate_id(swap: str, burden: float, burst: int, rank: int) -> str:
    token = str(float(burden)).replace(".", "p")
    return f"gfd_{swap}_q{token}_b{burst}_r{rank:04d}"


def build_label_free_candidate_evidence(
    score_maps: Mapping[str, np.ndarray],
    *,
    frame_ui: np.ndarray,
    burst_intervals_ui: Mapping[str, Sequence[int]],
    quiet_half_a_ui: Sequence[int],
    quiet_half_b_ui: Sequence[int],
    scale_floors: Mapping[str, float],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    list[dict[str, Any]],
]:
    """Create every frozen curve point without reading label coordinates."""

    frames = np.asarray(frame_ui, dtype=np.int64)
    if frames.ndim != 1 or not np.array_equal(
        frames, np.arange(frames[0], frames[0] + len(frames), dtype=np.int64)
    ):
        raise ValueError("frame_ui must be a contiguous one-dimensional sequence")
    if tuple(score_maps) != QUIET_SWAPS:
        raise ValueError(f"score maps must be ordered exactly as {QUIET_SWAPS}")
    shapes = {tuple(np.asarray(score_maps[swap]).shape) for swap in QUIET_SWAPS}
    if len(shapes) != 1:
        raise ValueError("quiet-swap score maps must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[0] != len(frames):
        raise ValueError("score maps must be aligned TYX arrays")
    if not all(np.isfinite(np.asarray(score_maps[swap])).all() for swap in QUIET_SWAPS):
        raise ValueError("score maps contain non-finite values")
    bursts = {
        str(key): tuple(map(int, value))
        for key, value in sorted(burst_intervals_ui.items())
    }
    if tuple(bursts) != ("1", "2", "3", "4"):
        raise ValueError("exactly four frozen burst windows are required")
    burst_windows = {
        key: _ui_window(bounds, review_start_ui=int(frames[0]))
        for key, bounds in bursts.items()
    }
    durations = _duration_map(bursts)
    occupancy = np.empty(
        (
            len(QUIET_SWAPS),
            len(QUIET_NMS_PEAK_BURDENS),
            4,
            shape[1],
            shape[2],
        ),
        dtype=np.float32,
    )
    candidates: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    occupancy_index: list[dict[str, Any]] = []
    for swap_index, (swap, floor_interval, test_interval) in enumerate(
        (
            (QUIET_SWAPS[0], quiet_half_a_ui, quiet_half_b_ui),
            (QUIET_SWAPS[1], quiet_half_b_ui, quiet_half_a_ui),
        )
    ):
        score = np.asarray(score_maps[swap], dtype=np.float32)
        floor_mask = _interval_mask(frames, floor_interval)
        test_mask = _interval_mask(frames, test_interval)
        if int(floor_mask.sum()) != 50 or int(test_mask.sum()) != 50:
            raise ValueError("each frozen quiet half must contain exactly 50 frames")
        calibration = calibrate_training_quiet_thresholds(
            score,
            floor_mask,
            durations,
            target_peak_burdens=QUIET_NMS_PEAK_BURDENS,
            nms_distance_px=NMS_DISTANCE_PX,
        )
        test_windows = duration_matched_quiet_windows(test_mask, durations)
        by_burden = {
            float(row["target_nms_peaks_per_pseudo_burst"]): row
            for row in calibration.operating_points
        }
        if tuple(sorted(by_burden)) != QUIET_NMS_PEAK_BURDENS:
            raise AssertionError("quiet calibration omitted a frozen burden")
        for burden_index, burden in enumerate(QUIET_NMS_PEAK_BURDENS):
            operating = by_burden[burden]
            threshold = float(operating["threshold_z"])
            detected = extract_burst_candidates(
                score,
                burst_windows,
                threshold_z=threshold,
                nms_distance_px=NMS_DISTANCE_PX,
            )
            quiet_test = extract_burst_candidates(
                score,
                test_windows,
                threshold_z=threshold,
                nms_distance_px=NMS_DISTANCE_PX,
            )
            heldout_burden = sum(
                len(peaks) for peaks in quiet_test.peaks.values()
            ) / len(quiet_test.peaks)
            all_count = int(sum(len(peaks) for peaks in detected.peaks.values()))
            calibrations.append(
                {
                    "training_scope": "global_context_paired_quiet_cross_fit",
                    "context_role": CONTEXT_ROLE,
                    "context_id": CONTEXT_ID,
                    "representation": REPRESENTATION,
                    "quiet_swap": swap,
                    "nms_distance_px": NMS_DISTANCE_PX,
                    "nms_role": "primary",
                    "floor_interval_ui": json.dumps(list(map(int, floor_interval))),
                    "test_interval_ui": json.dumps(list(map(int, test_interval))),
                    "scale_floor_percentile": SCALE_FLOOR_PERCENTILE,
                    "scale_floor": float(scale_floors[swap]),
                    "target_nms_peaks_per_pseudo_burst": float(burden),
                    "threshold_z": threshold,
                    "calibration_achieved_peaks_per_pseudo_burst": float(
                        operating["achieved_nms_peaks_per_pseudo_burst"]
                    ),
                    "heldout_quiet_peaks_per_pseudo_burst": float(heldout_burden),
                    "all_four_burst_candidate_count": all_count,
                    "probability_of_false_alarm_claimed": False,
                    "positive_coordinates_used": False,
                    "positive_identities_used": False,
                }
            )
            for burst_index, burst_id in enumerate(("1", "2", "3", "4")):
                occupancy[swap_index, burden_index, burst_index] = np.asarray(
                    detected.occupancy_maps[burst_id], dtype=np.float32
                )
                start, stop = burst_windows[burst_id]
                occupancy_index.append(
                    {
                        "occupancy_swap_index": swap_index,
                        "occupancy_burden_index": burden_index,
                        "occupancy_burst_index": burst_index,
                        "quiet_swap": swap,
                        "target_nms_peaks_per_pseudo_burst": float(burden),
                        "burst_id": int(burst_id),
                        "source_start_ui": bursts[burst_id][0],
                        "source_stop_ui": bursts[burst_id][1],
                        "threshold_z": threshold,
                        "map_semantics": "count_of_frames_gamma_score_strictly_above_threshold",
                    }
                )
                for rank, (occupancy_score, x_px, y_px) in enumerate(
                    detected.peaks[burst_id], start=1
                ):
                    trace = score[start:stop, int(y_px), int(x_px)]
                    relative_peak = int(np.argmax(trace))
                    review_peak = start + relative_peak
                    candidates.append(
                        {
                            "candidate_id": _candidate_id(
                                swap, burden, int(burst_id), rank
                            ),
                            "training_scope": "global_context_paired_quiet_cross_fit",
                            "burst_id": int(burst_id),
                            "context_role": CONTEXT_ROLE,
                            "context_id": CONTEXT_ID,
                            "representation": REPRESENTATION,
                            "quiet_swap": swap,
                            "nms_distance_px": NMS_DISTANCE_PX,
                            "nms_role": "primary",
                            "target_nms_peaks_per_pseudo_burst": float(burden),
                            "threshold_z": threshold,
                            "scale_floor_percentile": SCALE_FLOOR_PERCENTILE,
                            "scale_floor": float(scale_floors[swap]),
                            "candidate_rank": rank,
                            "occupancy_score": float(occupancy_score),
                            "x_px": int(x_px),
                            "y_px": int(y_px),
                            "peak_frame_review_zero": review_peak,
                            "peak_source_frame_ui": int(frames[review_peak]),
                            "peak_gamma_score_z": float(trace[relative_peak]),
                            "threshold_exceedance_frame_count": int(
                                np.count_nonzero(trace > threshold)
                            ),
                            "interpretation_before_label_join": "unknown_candidate",
                        }
                    )
    return candidates, calibrations, occupancy, occupancy_index


def _validate_candidate_evidence(
    candidates: Sequence[Mapping[str, Any]],
    calibrations: Sequence[Mapping[str, Any]],
    occupancy: np.ndarray,
    occupancy_index: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    expected_calibrations = len(QUIET_SWAPS) * len(QUIET_NMS_PEAK_BURDENS)
    keys = {
        (
            row["quiet_swap"],
            float(row["target_nms_peaks_per_pseudo_burst"]),
        )
        for row in calibrations
    }
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    groups: dict[tuple[str, float, int], list[int]] = {}
    for row in candidates:
        key = (
            str(row["quiet_swap"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
            int(row["burst_id"]),
        )
        groups.setdefault(key, []).append(int(row["candidate_rank"]))
    ranks_contiguous = all(
        sorted(ranks) == list(range(1, len(ranks) + 1))
        for ranks in groups.values()
    )
    checks = {
        "ten_operating_points": len(calibrations) == expected_calibrations
        and len(keys) == expected_calibrations,
        "all_burdens_and_swaps": keys
        == {
            (swap, burden)
            for swap in QUIET_SWAPS
            for burden in QUIET_NMS_PEAK_BURDENS
        },
        "occupancy_shape": tuple(occupancy.shape) == (2, 5, 4, 340, 573),
        "occupancy_index_complete": len(occupancy_index) == 40,
        "candidate_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "candidate_ranks_contiguous": ranks_contiguous,
        "all_four_bursts_covered_by_operating_contract": {
            int(row["burst_id"]) for row in occupancy_index
        }
        == {1, 2, 3, 4},
        "no_labels_used": all(
            row["positive_coordinates_used"] is False
            and row["positive_identities_used"] is False
            for row in calibrations
        )
        and all(
            row["interpretation_before_label_join"] == "unknown_candidate"
            for row in candidates
        ),
        "finite_candidate_values": all(
            math.isfinite(float(row["occupancy_score"]))
            and math.isfinite(float(row["peak_gamma_score_z"]))
            for row in candidates
        ),
    }
    return checks


def _write_candidate_checkpoint(
    config: GammaLSDifferenceConfig, work: Path, stage_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    frame_ui = np.arange(
        EXPECTED_REVIEW_INTERVAL_UI[0],
        EXPECTED_REVIEW_INTERVAL_UI[1] + 1,
        dtype=np.int64,
    )
    folds = build_fold_contracts(config)
    scores = {
        swap: np.load(
            work / STAGE_ARRAY_FILES[f"gamma_{swap}"],
            mmap_mode="r",
            allow_pickle=False,
        )
        for swap in QUIET_SWAPS
    }
    candidates, calibrations, occupancy, occupancy_index = (
        build_label_free_candidate_evidence(
            scores,
            frame_ui=frame_ui,
            burst_intervals_ui=config.payload["frames"]["burst_intervals_ui"],
            quiet_half_a_ui=folds[0].quiet_half_a_ui,
            quiet_half_b_ui=folds[0].quiet_half_b_ui,
            scale_floors=stage_manifest["quiet_scale_floors"],
        )
    )
    checks = _validate_candidate_evidence(
        candidates, calibrations, occupancy, occupancy_index
    )
    if not all(checks.values()):
        raise FixedDeploymentCharacterizationUnavailable(
            f"candidate checkpoint validation failed: {checks}"
        )
    _atomic_tsv(
        work / "candidates_label_sealed.tsv",
        candidates,
        fieldnames=CANDIDATE_FIELDS,
    )
    _atomic_tsv(
        work / "threshold_calibration.tsv",
        calibrations,
        fieldnames=CALIBRATION_FIELDS,
    )
    _atomic_tsv(
        work / "occupancy_map_index.tsv",
        occupancy_index,
        fieldnames=OCCUPANCY_INDEX_FIELDS,
    )
    _atomic_npy(work / STAGE_ARRAY_FILES["occupancy_maps"], occupancy)
    frame_rows = [
        {
            "review_frame_zero": index,
            "source_frame_ui": int(source),
            "source_frame_zero": int(source) - 1,
        }
        for index, source in enumerate(frame_ui)
    ]
    _atomic_tsv(
        work / "frame_index.tsv",
        frame_rows,
        fieldnames=("review_frame_zero", "source_frame_ui", "source_frame_zero"),
    )
    sealed_files = [
        "stage_array_manifest.json",
        "frame_index.tsv",
        "threshold_calibration.tsv",
        "candidates_label_sealed.tsv",
        "occupancy_map_index.tsv",
        *STAGE_ARRAY_FILES.values(),
    ]
    seal = {
        "schema_version": 1,
        "status": "sealed_before_protected_v1_label_parse",
        "sealed_at_utc": _utc_now(),
        "candidate_rows": len(candidates),
        "calibration_rows": len(calibrations),
        "occupancy_maps": len(occupancy_index),
        "checks": checks,
        "files": {
            relative: {
                "size_bytes": (work / relative).stat().st_size,
                "sha256": _sha256(work / relative),
            }
            for relative in sealed_files
        },
        "protected_v1_rows_parsed": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "candidate_membership_frozen": True,
    }
    _atomic_json(work / "candidate_score_seal.json", seal)
    return seal


def _verify_candidate_seal(work: Path) -> dict[str, Any]:
    seal = _read_json(work / "candidate_score_seal.json")
    if seal.get("status") != "sealed_before_protected_v1_label_parse" or not all(
        seal.get("checks", {}).values()
    ):
        raise FixedDeploymentCharacterizationUnavailable("candidate seal is invalid")
    for relative, row in seal.get("files", {}).items():
        path = work / relative
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise FixedDeploymentCharacterizationUnavailable(
                f"sealed file is absent or resized: {relative}"
            )
        if _sha256(path) != row["sha256"]:
            raise FixedDeploymentCharacterizationUnavailable(
                f"sealed file changed: {relative}"
            )
    return seal


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _crossfit_summary(
    observation_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for burden in QUIET_NMS_PEAK_BURDENS:
        for budget in CANDIDATE_BUDGETS_PER_BURST:
            selected = [
                row
                for row in observation_rows
                if float(row["target_nms_peaks_per_pseudo_burst"]) == burden
                and int(row["candidate_budget"]) == budget
            ]
            by_swap = {
                swap: [row for row in selected if row["quiet_swap"] == swap]
                for swap in QUIET_SWAPS
            }
            if any(len(rows) != EXPECTED_PROTECTED_V1_ROWS for rows in by_swap.values()):
                raise FixedDeploymentCharacterizationUnavailable(
                    "joined observation count changed within a quiet swap"
                )
            matched_by_swap = {
                swap: sum(bool(row["matched"]) for row in rows)
                for swap, rows in by_swap.items()
            }
            candidate_count_by_swap = {
                swap: sum(
                    min(
                        len(
                            [
                                row
                                for row in candidate_rows
                                if row["quiet_swap"] == swap
                                and float(
                                    row["target_nms_peaks_per_pseudo_burst"]
                                )
                                == burden
                                and int(row["burst_id"]) == burst
                            ]
                        ),
                        budget,
                    )
                    for burst in (1, 2, 3, 4)
                )
                for swap in QUIET_SWAPS
            }
            total_matched = sum(matched_by_swap.values())
            total_known = sum(len(rows) for rows in by_swap.values())
            output.append(
                {
                    "target_nms_peaks_per_pseudo_burst": burden,
                    "candidate_budget_per_burst": budget,
                    "a_train_b_test_matched_known_positive_count": matched_by_swap[
                        QUIET_SWAPS[0]
                    ],
                    "a_train_b_test_known_positive_recall": matched_by_swap[
                        QUIET_SWAPS[0]
                    ]
                    / EXPECTED_PROTECTED_V1_ROWS,
                    "a_train_b_test_effective_candidate_count_all_bursts": candidate_count_by_swap[
                        QUIET_SWAPS[0]
                    ],
                    "b_train_a_test_matched_known_positive_count": matched_by_swap[
                        QUIET_SWAPS[1]
                    ],
                    "b_train_a_test_known_positive_recall": matched_by_swap[
                        QUIET_SWAPS[1]
                    ]
                    / EXPECTED_PROTECTED_V1_ROWS,
                    "b_train_a_test_effective_candidate_count_all_bursts": candidate_count_by_swap[
                        QUIET_SWAPS[1]
                    ],
                    "paired_quiet_crossfit_mean_known_positive_recall": total_matched
                    / total_known,
                    "known_positive_occurrences_per_swap": EXPECTED_PROTECTED_V1_ROWS,
                    "precision_identified": False,
                    "unmatched_candidates": "unknown_not_negative",
                    "evidence_role": "post_selection_within_recording_characterization",
                }
            )
    return output


def _write_join_and_audit_inputs(
    config: GammaLSDifferenceConfig,
    work: Path,
    replay_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    seal = _verify_candidate_seal(work)
    label_path = config.source_paths["protected_labels_v1"]
    expected_label_hash = replay_preflight["inputs"]["base_preflight"][
        "protected_v1_sha256"
    ]
    if _sha256(label_path) != expected_label_hash:
        raise FixedDeploymentCharacterizationUnavailable(
            "protected-v1 table changed after preflight"
        )
    # This is the first parse of annotation rows in this runner.  All dense
    # arrays, thresholds, candidates, peak frames, and occupancy maps above are
    # already content-hash sealed.
    positives = _read_sparse_positives(
        label_path,
        selector="include_inclusive",
        expected_rows=EXPECTED_PROTECTED_V1_ROWS,
        movie_shape_yx=EXPECTED_MOVIE_SHAPE[1:],
    )
    candidates = _read_tsv(work / "candidates_label_sealed.tsv")
    calibrations = _read_tsv(work / "threshold_calibration.tsv")
    observations = observation_match_rows(
        candidates,
        positives,
        cohort=COHORT,
        operating_rows=calibrations,
    )
    aggregates = aggregate_match_rows(observations)
    crossfit = _crossfit_summary(observations, candidates)
    _atomic_tsv(
        work / "protected_v1_observation_matches.tsv",
        observations,
        fieldnames=tuple(observations[0]),
    )
    _atomic_tsv(
        work / "protected_v1_burst_metrics.tsv",
        aggregates,
        fieldnames=tuple(aggregates[0]),
    )
    _atomic_tsv(
        work / "protected_v1_crossfit_summary.tsv",
        crossfit,
        fieldnames=tuple(crossfit[0]),
    )
    audit_state = replay_preflight["science_contract"]["audit_representative_state"]
    audit_swap = str(audit_state["quiet_swap"])
    audit_burden = float(audit_state["target_nms_peaks_per_pseudo_burst"])
    audit_budget = int(audit_state["candidate_budget_per_burst"])
    audit_candidates = [
        row
        for row in candidates
        if row["quiet_swap"] == audit_swap
        and float(row["target_nms_peaks_per_pseudo_burst"]) == audit_burden
        and int(row["candidate_rank"]) <= audit_budget
    ]
    audit_matches = [
        row
        for row in observations
        if row["quiet_swap"] == audit_swap
        and float(row["target_nms_peaks_per_pseudo_burst"]) == audit_burden
        and int(row["candidate_budget"]) == audit_budget
    ]
    bursts = config.payload["frames"]["burst_intervals_ui"]
    audit_experts = [
        {
            **row,
            "source_start_ui": int(bursts[str(row["burst_id"])][0]),
            "source_stop_ui": int(bursts[str(row["burst_id"])][1]),
            "temporal_extent_semantics": "configured_burst_window_not_per_roi_onset",
        }
        for row in positives
    ]
    # ``_atomic_tsv`` intentionally assumes its parent already exists.  The
    # metric tables live at the work root, whereas these post-seal renderer
    # inputs live in their own directory, so establish that directory before
    # the first atomic write.  This is a post-score packaging step only.
    (work / "audit_inputs").mkdir(parents=True, exist_ok=True)
    _atomic_tsv(
        work / "audit_inputs" / "expert_occurrences.tsv",
        audit_experts,
        fieldnames=tuple(audit_experts[0]),
    )
    _atomic_tsv(
        work / "audit_inputs" / "model_occurrences.tsv",
        audit_candidates,
        fieldnames=CANDIDATE_FIELDS,
    )
    _atomic_tsv(
        work / "audit_inputs" / "one_to_one_matches.tsv",
        audit_matches,
        fieldnames=tuple(audit_matches[0]),
    )
    _atomic_json(
        work / "audit_inputs" / "stage_sources.json",
        {
            "schema_version": 1,
            "representative_state": audit_state,
            "conditioned_current_frame": STAGE_ARRAY_FILES[
                "conditioned_current_frame"
            ],
            "difference_signed": STAGE_ARRAY_FILES["difference_signed"],
            "gamma_score": STAGE_ARRAY_FILES[f"gamma_{audit_swap}"],
            "occupancy_maps": STAGE_ARRAY_FILES["occupancy_maps"],
            "occupancy_map_index": "occupancy_map_index.tsv",
            "frame_index": "frame_index.tsv",
            "candidate_score_seal_sha256": _sha256(
                work / "candidate_score_seal.json"
            ),
            "source_movie": config.payload["sources"]["movie"],
            "source_movie_sha256": replay_preflight["inputs"]["base_preflight"][
                "source_movie_sha256"
            ],
            "annotation_separation_required": "strict_three_section",
            "comparison_spatial_panel_count": 2,
        },
    )
    return {
        "candidate_seal": seal,
        "protected_v1_label_sha256": expected_label_hash,
        "positive_rows": len(positives),
        "unique_positive_roi_ids": len(
            {str(row["canonical_roi_id"]) for row in positives}
        ),
        "observation_match_rows": len(observations),
        "burst_metric_rows": len(aggregates),
        "crossfit_summary_rows": len(crossfit),
        "audit_model_occurrences": len(audit_candidates),
        "audit_match_rows": len(audit_matches),
        "audit_representative_state": audit_state,
        "annotation_rows_first_parsed_after_candidate_score_seal": True,
    }


def _rebind_sealed_resume_contract(
    work: Path,
    *,
    existing_contract: Mapping[str, Any],
    requested_contract: Mapping[str, Any],
    requested_preflight: Mapping[str, Any],
) -> None:
    """Rebind a sealed checkpoint after a post-score packaging-only hotfix.

    The method fails closed unless every scientific/execution dimension is
    unchanged, the old and new replay preflights are both indexed, only this
    packaging module and its regression test changed, and all dense/candidate
    checkpoint hashes still verify.  It never calls a scorer or CUDA.
    """

    invariant_keys = (
        "schema_version",
        "experiment_id",
        "destination",
        "science_contract_sha256",
        "device",
        "chunk_frames",
    )
    if any(existing_contract.get(key) != requested_contract.get(key) for key in invariant_keys):
        raise FixedDeploymentCharacterizationUnavailable(
            "sealed resume science or execution contract changed"
        )
    _verify_stage_manifest(work)
    candidate_seal = _verify_candidate_seal(work)
    old_root = Path(str(existing_contract["replay_preflight_root"])).resolve()
    old_indexed = verify_indexed_artifact(old_root)
    if old_indexed["artifact_index_sha256"] != existing_contract[
        "replay_preflight_artifact_index_sha256"
    ]:
        raise FixedDeploymentCharacterizationUnavailable(
            "checkpoint's original replay preflight changed"
        )
    old_preflight = _read_json(old_root / "preflight.json")
    if old_preflight.get("science_contract") != requested_preflight.get(
        "science_contract"
    ):
        raise FixedDeploymentCharacterizationUnavailable(
            "old and new replay science contracts differ"
        )
    old_implementation = old_preflight.get("implementation", {})
    new_implementation = requested_preflight.get("implementation", {})
    keys = set(old_implementation) | set(new_implementation)
    changed = {
        key
        for key in keys
        if old_implementation.get(key) != new_implementation.get(key)
    }
    allowed = {
        "repo://neurobench/experiments/gamma_ls_difference/fixed_deployment_characterization.py",
        "repo://tests/test_gamma_ls_fixed_deployment_characterization.py",
    }
    if not changed or not changed.issubset(allowed):
        raise FixedDeploymentCharacterizationUnavailable(
            f"resume preflight changes are not packaging-only: {sorted(changed)}"
        )
    prior_contract_path = work / "work_contract.pre_packaging_hotfix.json"
    if prior_contract_path.exists():
        if _read_json(prior_contract_path) != dict(existing_contract):
            raise FixedDeploymentCharacterizationUnavailable(
                "preserved pre-hotfix work contract is inconsistent"
            )
    else:
        _atomic_json(prior_contract_path, dict(existing_contract))
    rebind = {
        "schema_version": 1,
        "rebound_at_utc": _utc_now(),
        "reason": "post_score_audit_inputs_parent_directory_packaging_hotfix",
        "cuda_or_scoring_reexecuted": False,
        "candidate_membership_changed": False,
        "metric_selection_changed": False,
        "labels_were_first_parsed_only_after_candidate_seal": True,
        "old_replay_preflight_artifact_index_sha256": old_indexed[
            "artifact_index_sha256"
        ],
        "new_replay_preflight_artifact_index_sha256": requested_contract[
            "replay_preflight_artifact_index_sha256"
        ],
        "science_contract_sha256": requested_contract["science_contract_sha256"],
        "candidate_score_seal_sha256": _sha256(
            work / "candidate_score_seal.json"
        ),
        "stage_array_manifest_sha256": _sha256(
            work / "stage_array_manifest.json"
        ),
        "sealed_candidate_rows": int(candidate_seal["candidate_rows"]),
        "changed_implementation_files": {
            key: {
                "old": old_implementation.get(key),
                "new": new_implementation.get(key),
            }
            for key in sorted(changed)
        },
    }
    _atomic_json(work / "preflight_rebind.json", rebind)
    _atomic_json(work / "work_contract.json", dict(requested_contract))
    _heartbeat(work, "sealed_checkpoint_rebound_after_packaging_hotfix")


def _validate_completed_artifact(
    work: Path,
    join: Mapping[str, Any],
    replay_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = _read_tsv(work / "candidates_label_sealed.tsv")
    calibrations = _read_tsv(work / "threshold_calibration.tsv")
    observations = _read_tsv(work / "protected_v1_observation_matches.tsv")
    crossfit = _read_tsv(work / "protected_v1_crossfit_summary.tsv")
    expected_observations = (
        len(QUIET_SWAPS)
        * len(QUIET_NMS_PEAK_BURDENS)
        * len(CANDIDATE_BUDGETS_PER_BURST)
        * EXPECTED_PROTECTED_V1_ROWS
    )
    audit_state = replay_preflight["science_contract"]["audit_representative_state"]
    audit_matches = _read_tsv(work / "audit_inputs" / "one_to_one_matches.tsv")
    checks = {
        "stage_manifest_verified": bool(_verify_stage_manifest(work)),
        "candidate_seal_verified": bool(_verify_candidate_seal(work)),
        "all_ten_operating_points": len(calibrations) == 10,
        "all_four_bursts_candidate_contract": {
            int(row["burst_id"]) for row in candidates
        }.issubset({1, 2, 3, 4})
        and {int(row["burst_id"]) for row in _read_tsv(work / "occupancy_map_index.tsv")}
        == {1, 2, 3, 4},
        "observation_rows_exact": len(observations) == expected_observations,
        "crossfit_curve_complete": len(crossfit)
        == len(QUIET_NMS_PEAK_BURDENS) * len(CANDIDATE_BUDGETS_PER_BURST),
        "audit_state_one_row_per_positive": len(audit_matches)
        == EXPECTED_PROTECTED_V1_ROWS,
        "audit_state_predeclared": join["audit_representative_state"] == audit_state,
        "label_join_after_seal": bool(
            join["annotation_rows_first_parsed_after_candidate_score_seal"]
        ),
        "claim_scope_post_selection_only": replay_preflight["science_contract"][
            "evidence_role"
        ]
        == "post_selection_within_recording_characterization",
        "not_protected_selection_estimate": replay_preflight["science_contract"][
            "protected_selection_estimate"
        ]
        is False,
        "not_independent_confirmation": replay_preflight["science_contract"][
            "independent_confirmation"
        ]
        is False,
        "full_record_q1_not_silently_covered": replay_preflight["science_contract"][
            "full_record_q1_proposal_stream_covered"
        ]
        is False,
        "precision_not_identified": all(
            row["precision_identified"] in (False, "False", "false")
            for row in crossfit
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "passed_metric_and_audit_input_contract_scientific_audit_pending"
            if all(checks.values())
            else "failed"
        ),
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "row_counts": {
            "candidates": len(candidates),
            "calibrations": len(calibrations),
            "observation_matches": len(observations),
            "crossfit_summary": len(crossfit),
            "audit_matches": len(audit_matches),
        },
        "scientific_audit_complete": False,
    }


def run_fixed_deployment_characterization(
    config_path: str | Path,
    *,
    replay_preflight_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    gpu_authorized: bool = False,
) -> dict[str, Any]:
    """Execute or resume the frozen replay, then join protected-v1 labels."""

    config = GammaLSDifferenceConfig.load(config_path)
    replay_preflight = _verify_replay_preflight(config, replay_preflight_dir)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    work = destination.parent / f".{destination.name}.fixed-deployment-work"
    work_contract = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "destination": str(destination),
        "replay_preflight_root": str(Path(replay_preflight_dir).expanduser().resolve()),
        "replay_preflight_artifact_index_sha256": replay_preflight[
            "replay_preflight"
        ]["artifact_index_sha256"],
        "science_contract_sha256": replay_preflight["science_contract_sha256"],
        "device": device,
        "chunk_frames": int(chunk_frames),
    }
    if work.exists():
        existing_contract = _read_json(work / "work_contract.json")
        if existing_contract != work_contract:
            _rebind_sealed_resume_contract(
                work,
                existing_contract=existing_contract,
                requested_contract=work_contract,
                requested_preflight=replay_preflight,
            )
    else:
        if not gpu_authorized:
            raise FixedDeploymentCharacterizationUnavailable(
                "GPU replay requires explicit --gpu-authorized acknowledgement"
            )
        if shutil.disk_usage(destination.parent).free < 4 * 2**30:
            raise FixedDeploymentCharacterizationUnavailable(
                "at least 4 GiB free disk is required before stage replay"
            )
        work.mkdir()
        _atomic_json(work / "work_contract.json", work_contract)
    try:
        if not (work / "stage_array_manifest.json").is_file():
            if not gpu_authorized:
                raise FixedDeploymentCharacterizationUnavailable(
                    "resuming before the CUDA stage requires --gpu-authorized"
                )
            stage = _write_stage_arrays(
                config,
                work,
                device_name=device,
                chunk_frames=int(chunk_frames),
            )
        else:
            stage = _verify_stage_manifest(work)
        _heartbeat(work, "cpu_exact_candidate_replay")
        if not (work / "candidate_score_seal.json").is_file():
            seal = _write_candidate_checkpoint(config, work, stage)
        else:
            seal = _verify_candidate_seal(work)
        _heartbeat(work, "protected_v1_join_after_seal")
        join = _write_join_and_audit_inputs(config, work, replay_preflight)
        summary = {
            "schema_version": 1,
            "status": "complete_metric_and_audit_inputs_scientific_audit_pending",
            "completed_at_utc": _utc_now(),
            "science_contract": replay_preflight["science_contract"],
            "science_contract_sha256": replay_preflight["science_contract_sha256"],
            "replay_preflight_artifact_index_sha256": replay_preflight[
                "replay_preflight"
            ]["artifact_index_sha256"],
            "stage_array_manifest_sha256": _sha256(
                work / "stage_array_manifest.json"
            ),
            "candidate_score_seal_sha256": _sha256(
                work / "candidate_score_seal.json"
            ),
            "candidate_rows": int(seal["candidate_rows"]),
            "join": join,
            "claim_boundary": {
                "post_selection_within_recording_characterization": True,
                "protected_selection_estimate": False,
                "independent_confirmation": False,
                "precision_specificity_false_positive_rate_identified": False,
                "unmatched_candidates": "unknown_not_negative",
                "single_recording": True,
                "full_record_q1_371_over_2259_audit_covered": False,
                "scientific_audit_complete": False,
            },
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "schema_version": 1,
                "entrypoint": "summary.json",
                "stage_order": [
                    "causally conditioned acquired frame",
                    "signed adjacent temporal difference",
                    "signed radial Gamma-LS h15/g7/n9/mode7.5",
                    "burst threshold-occupancy map",
                    "deterministic NMS6 candidates",
                    "protected-v1 join after candidate_score_seal.json",
                ],
                "primary_tables": [
                    "protected_v1_crossfit_summary.tsv",
                    "protected_v1_burst_metrics.tsv",
                    "protected_v1_observation_matches.tsv",
                    "threshold_calibration.tsv",
                    "candidates_label_sealed.tsv",
                ],
                "scientific_audit_inputs": "audit_inputs/",
                "claim_boundary": summary["claim_boundary"],
                "full_record_model_only_audit_gap": (
                    "The separate q=1 371/2259 full-record proposal ledger is not covered."
                ),
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Fixed global-h15 protected within-recording characterization\n\n"
            "This hash-bound replay applies the already frozen `difference_signed` "
            "representation and global `support_support_a_h15_g7_n9_m0p5` context "
            "to all four configured burst windows. Both predeclared quiet cross-fit "
            "swaps, all five empirical quiet burdens, NMS6, and the five candidate "
            "budgets are retained without retuning. Conditioned, signed-difference, "
            "Gamma-score, threshold-occupancy, and peak-frame evidence was sealed "
            "before protected-v1 labels were parsed. The result is post-selection "
            "within-recording characterization, not a protected selection estimate "
            "or independent confirmation. Unmatched candidates are unknown. The "
            "separate full-record q=1 proposal ledger behind the 371/2259 headline "
            "still requires its own model-only scientific media audit.\n",
        )
        validation = _validate_completed_artifact(
            work, join, replay_preflight
        )
        if not validation["all_checks_pass"]:
            raise FixedDeploymentCharacterizationUnavailable(
                f"completed artifact validation failed: {validation['checks']}"
            )
        _atomic_json(work / "validation.json", validation)
        _atomic_json(
            work / "status.json",
            {
                "status": summary["status"],
                "scientific_audit_complete": False,
                "candidate_score_seal_sha256": summary[
                    "candidate_score_seal_sha256"
                ],
                "resumable_work_completed": True,
            },
        )
        _heartbeat(work, "complete_metric_and_audit_inputs")
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
        return {
            "output": str(destination),
            "status": summary["status"],
            "validation": validation,
        }
    except BaseException:
        # The fixed, non-colliding work directory is retained for exact resume.
        raise


def write_full_record_model_audit_gap(
    full_record_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Record why the 371/2259 operational ledger lacks model-only media."""

    source = Path(full_record_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    indexed = verify_indexed_artifact(source)
    validation = _read_json(source / "validation.json")
    if not validation.get("all_checks_pass"):
        raise FixedDeploymentCharacterizationUnavailable(
            "full-record proposal ledger is not validated"
        )
    rows = _read_tsv(source / "proposal_summary.tsv")
    headline = [
        row
        for row in rows
        if row["variant_id"] == "initial_100_annotation_file_and_location_free"
        and float(row["target_nms_peaks_per_calibration_unit"]) == 1.0
    ]
    if len(headline) != 1:
        raise FixedDeploymentCharacterizationUnavailable(
            "full-record q=1 headline row is not unique"
        )
    row = headline[0]
    if int(row["proposal_row_count"]) != 371 or int(row["eligible_frame_count"]) != 2259:
        raise FixedDeploymentCharacterizationUnavailable(
            "expected 371/2259 full-record headline changed"
        )
    expected_media_sources = (
        "conditioned_current_frame.npy",
        "difference_signed.npy",
        "signed_radial_gamma_ls.npy",
        "framewise_detection_maps.npy",
        "candidate_peak_traces.tsv",
    )
    present = [name for name in expected_media_sources if (source / name).is_file()]
    work = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        summary = {
            "schema_version": 1,
            "status": "blocked_fail_closed_model_only_media_gap",
            "scientific_audit_complete": False,
            "source": {
                **indexed,
                "validation_sha256": _sha256(source / "validation.json"),
                "candidate_seal_sha256": _sha256(source / "candidate_seal.json"),
            },
            "headline": {
                "variant_id": row["variant_id"],
                "target_nms_peaks_per_calibration_unit": 1.0,
                "eligible_frame_count": 2259,
                "frame_level_proposal_row_count": 371,
                "frames_with_at_least_one_proposal": int(
                    row["frames_with_at_least_one_proposal"]
                ),
                "unique_biological_event_count": "not_identified",
            },
            "required_media_sources": list(expected_media_sources),
            "present_media_sources": present,
            "blockers": [
                "the ledger persists coordinates and counts but not synchronized conditioned/difference/Gamma frame arrays",
                "the ledger contains frame-level proposals but no exact-pixel full-duration model traces",
                "a new exact frozen replay is required for model-only media; burst-only protected replay cannot substitute",
            ],
            "claim_boundary": (
                "This gap record does not alter or invalidate the 371 frame-level "
                "proposal rows. It establishes only that their model-only scientific "
                "media audit is not complete."
            ),
        }
        _atomic_json(work / "summary.json", summary)
        _atomic_json(
            work / "llm_context.json",
            {
                "schema_version": 1,
                "entrypoint": "summary.json",
                "headline_semantics": "automated_frame_level_proposals_not_unique_neurons_or_events",
                "full_record_q1_model_only_audit_covered": False,
                "protected_burst_replay_is_not_a_substitute": True,
                "blockers": summary["blockers"],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Full-record q=1 model-only scientific-audit gap\n\n"
            "The validated operational ledger reports 371 automated frame-level "
            "proposals across 2,259 eligible application frames for the annotation-"
            "file-and-location-free initial-100 q=1 state. These are not 371 unique "
            "neurons or biological events. The current ledger lacks synchronized "
            "conditioned, signed-difference, Gamma-score, framewise detection-map, "
            "and exact-pixel trace sources, so its model-only scientific media audit "
            "fails closed. The 560-frame protected burst replay must not be used as "
            "a substitute for this full-record audit.\n",
        )
        _atomic_json(
            work / "validation.json",
            {
                "schema_version": 1,
                "status": "failed_expected_missing_model_media_sources",
                "all_checks_pass": False,
                "source_metric_artifact_verified": True,
                "headline_371_over_2259_verified": True,
                "scientific_audit_complete": False,
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        work.replace(destination)
    except BaseException:
        raise
    return {"output": str(destination), "summary": summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="freeze replay inputs without parsing annotations"
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--base-preflight-dir", required=True)
    preflight.add_argument("--support-dir", required=True)
    preflight.add_argument("--exact-truth-dir", required=True)
    preflight.add_argument("--output-dir", required=True)
    preflight.add_argument("--audit-swap", choices=QUIET_SWAPS, default=DEFAULT_AUDIT_SWAP)
    preflight.add_argument(
        "--audit-burden",
        type=float,
        choices=QUIET_NMS_PEAK_BURDENS,
        default=DEFAULT_AUDIT_BURDEN,
    )
    run = subparsers.add_parser("run", help="run or resume the frozen CUDA replay")
    run.add_argument("--config", required=True)
    run.add_argument("--replay-preflight-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    run.add_argument("--gpu-authorized", action="store_true")
    gap = subparsers.add_parser(
        "full-record-audit-gap", help="write the fail-closed 371/2259 media-gap packet"
    )
    gap.add_argument("--full-record-root", required=True)
    gap.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = write_replay_preflight(
            args.config,
            base_preflight_dir=args.base_preflight_dir,
            support_dir=args.support_dir,
            exact_truth_dir=args.exact_truth_dir,
            output_dir=args.output_dir,
            audit_swap=args.audit_swap,
            audit_burden=args.audit_burden,
        )
    elif args.command == "run":
        result = run_fixed_deployment_characterization(
            args.config,
            replay_preflight_dir=args.replay_preflight_dir,
            output_dir=args.output_dir,
            device=args.device,
            chunk_frames=args.chunk_frames,
            gpu_authorized=args.gpu_authorized,
        )
    else:
        result = write_full_record_model_audit_gap(
            args.full_record_root, args.output_dir
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
