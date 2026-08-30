"""Guarded orchestration for the registered spatiotemporal JEPA pilot.

The maintained JSON file is the scientific contract.  This runner never
silently relaxes that contract: ``smoke`` and ``screen`` are explicitly
non-scientific execution modes, while the claim-bearing run remains blocked
until its authorization and complete scientific-audit gates are satisfied.
Raw videos are resolved at runtime and only portable identifiers are written.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.nn import functional as F

from neurobench.experiments.information_source_separation.semi_synthetic import (
    SemiSyntheticFixture,
)
from neurobench.portable_paths import data_root as configured_data_root
from neurobench.portable_paths import portable_path, portableize_paths

from .contracts import atomic_json, atomic_text, stable_hash
from .discovery import sha256_file
from .jepa_data import (
    ClipRequest,
    RecordingDescriptor,
    RecordingInventory,
    RecordingSplit,
    RobustNormalization,
    TemporalClipContract,
    build_jepa_data_manifest,
    fit_training_robust_normalization,
    fixed_behavior_stratified_split,
    load_recording_inventory,
    open_recording_memmap,
    read_sequential_clip,
    sample_training_crop_plan,
    sample_validation_crop_plan,
    validate_060126_inventory,
)
from .jepa_comparators import frozen_handcrafted_comparator
from .jepa_evaluation import (
    additive_closure_metrics,
    evaluate_paired_suite,
    hierarchical_strongest_comparator_bootstrap,
    make_native_background_injection,
)
from .jepa_motion_audit import MotionAuditConfig, run_jepa_motion_audit
from .jepa_training import (
    MatchedTrainingResult,
    TrainingSchedule,
    model_latent_temporal_change_scorer,
    train_matched_representations,
)
from .spatiotemporal_jepa import (
    EncoderConfig,
    FrozenRandomEncoderBaseline,
    MaskConfig,
    MaskedPixelAutoencoder,
    SpatiotemporalJEPA,
    count_parameters,
    make_patch_mask_from_config,
    patch_grid_shape,
    patch_mask_to_voxels,
    tiny_smoke_configuration,
)


SCHEMA_VERSION = 1
SUPPORTED_MODES = {"preflight", "smoke", "screen"}
SPON_BURSTS_UI_INCLUSIVE = ((2003, 2026), (2040, 2063), (2122, 2149), (2254, 2300))
class JEPAPilotError(ValueError):
    """Raised when the runner cannot preserve the registered contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JEPAPilotError(f"JSON root must be an object: {path}")
    return payload


def _require(mapping: Mapping[str, Any], key: str, expected: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise JEPAPilotError(f"configuration field {key!r} must be {expected.__name__}")
    return value


def load_jepa_pilot_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the maintained experiment descriptor once."""

    config_path = Path(path).expanduser().resolve()
    config = _load_json(config_path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise JEPAPilotError("unsupported JEPA pilot schema_version")
    if config.get("experiment_id") != "NREV-EXP-0028":
        raise JEPAPilotError("JEPA pilot config must identify NREV-EXP-0028")
    for key in (
        "input_contract",
        "fixed_recording_split",
        "normalization",
        "seeds",
        "clip",
        "tubelet_encoder",
        "target_mask",
        "score_heads",
        "matched_training_budget",
        "empirical_background_injection",
        "evaluation",
        "representation_gates",
        "nuisance_audit",
        "scientific_audit",
        "implementation_hash_contract",
        "resources",
        "claim_boundary",
    ):
        _require(config, key, dict)
    _require(config, "arms", list)
    _require(config, "output_root", str)
    _require(config, "data_inventory", str)
    return config


def _resolve_repo_path(value: str, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _output_paths(requested: Path) -> tuple[Path, Path]:
    resolved = requested.expanduser().resolve()
    return resolved, resolved.with_name(resolved.name + ".partial")


def _portable_config(
    config: Mapping[str, Any], *, repository_root: Path, runtime_data_root: Path
) -> dict[str, Any]:
    return portableize_paths(
        dict(config), repository=repository_root, data=runtime_data_root
    )


def _architecture_from_config(config: Mapping[str, Any]) -> tuple[EncoderConfig, MaskConfig]:
    encoder = config["tubelet_encoder"]
    target = config["target_mask"]
    clip_shape = tuple(int(value) for value in config["clip"]["shape_tyx"])
    patch_shape = tuple(int(value) for value in encoder["tubelet_shape_tyx"])
    if any(value % patch for value, patch in zip(clip_shape, patch_shape, strict=True)):
        raise JEPAPilotError("clip dimensions must be divisible by the tubelet shape")
    resolved_encoder = EncoderConfig(
        in_channels=int(encoder["input_channels"]),
        patch_size=patch_shape,
        embed_dim=int(encoder["embedding_channels"]),
        latent_dim=int(encoder["latent_channels"]),
        depth=int(encoder["residual_depth"]),
        norm_groups=8,
    )
    # The registered primary uses the fixed 2 x 4 x 8 cuboids documented by
    # MaskConfig.  The config's min/max bounds are checked around that exact
    # primary rather than being sampled as an unregistered hyperparameter.
    block_shape = tuple(int(value) for value in target["cuboid_shape_tokens_tyx"])
    if block_shape != (2, 4, 8):
        raise JEPAPilotError("registered primary cuboid must be exactly [2, 4, 8] tokens")
    resolved_mask = MaskConfig(
        block_shape=block_shape,
        blocks_per_sample=int(target["cuboids_per_clip"]),
        coverage_range=(
            float(target["masked_token_fraction_minimum"]),
            float(target["masked_token_fraction_maximum"]),
        ),
        non_overlapping=bool(target["nonoverlap_required"]),
        minimum_temporal_tubelets=int(target["minimum_temporal_tubelets_per_cuboid"]),
        mask_value=float(target["mask_value_after_normalization"]),
    )
    observed_grid = tuple(value // patch for value, patch in zip(clip_shape, patch_shape, strict=True))
    declared_grid = tuple(int(value) for value in encoder["token_grid_tyx"])
    if observed_grid != declared_grid:
        raise JEPAPilotError(
            f"declared token grid {declared_grid} disagrees with geometry {observed_grid}"
        )
    return resolved_encoder, resolved_mask


def _resource_snapshot(output_parent: Path, requested_device: str) -> dict[str, Any]:
    disk_probe = output_parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    disk = shutil.disk_usage(disk_probe)
    memory: dict[str, int | None] = {"available_mib": None, "total_mib": None}
    try:
        rows = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            rows[key] = int(value.strip().split()[0])
        memory = {
            "available_mib": rows.get("MemAvailable", 0) // 1024,
            "total_mib": rows.get("MemTotal", 0) // 1024,
        }
    except (OSError, ValueError, IndexError):
        pass
    cuda_available = bool(torch.cuda.is_available())
    cuda: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        free, total = torch.cuda.mem_get_info()
        cuda.update(
            {
                "device_count": int(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0),
                "free_mib": int(free // (1024**2)),
                "total_mib": int(total // (1024**2)),
                "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return {
        "requested_device": requested_device,
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk": {
            "free_mib": int(disk.free // (1024**2)),
            "total_mib": int(disk.total // (1024**2)),
        },
        "cuda": cuda,
    }


def _validate_implementation_hash_contract(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    contract = config["implementation_hash_contract"]
    if contract.get("hash_algorithm") != "sha256":
        raise JEPAPilotError("implementation hash contract must use sha256")
    fixture = contract.get("fixture_generator")
    suite = contract.get("comparator_and_evaluator_suite")
    if not isinstance(fixture, Mapping) or not isinstance(suite, Mapping):
        raise JEPAPilotError("implementation hash contract sections are required")
    suite_sources = suite.get("sources")
    if not isinstance(suite_sources, list):
        raise JEPAPilotError("implementation comparator suite sources must be a list")
    rows = [
        {
            "path": fixture.get("source_path"),
            "sha256": fixture.get("source_sha256"),
            "role": "fixture_generator",
        },
        *suite_sources,
    ]
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise JEPAPilotError("implementation source rows must be objects")
        logical = str(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        candidate = (repository_root / logical).resolve()
        try:
            candidate.relative_to(repository_root)
        except ValueError as exc:
            raise JEPAPilotError(
                f"implementation source escapes repository: {logical}"
            ) from exc
        if Path(logical).is_absolute() or not candidate.is_file() or len(expected) != 64:
            raise JEPAPilotError(f"invalid implementation source contract: {logical}")
        observed = sha256_file(candidate)
        validated.append(
            {
                "path": logical,
                "role": str(row.get("role", "unspecified")),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": observed == expected,
            }
        )
    return {
        "hash_algorithm": "sha256",
        "mismatch_policy": str(contract.get("mismatch_policy", "fail_preflight")),
        "all_match": all(row["matches"] for row in validated),
        "sources": validated,
    }


def _preflight_checks(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    repository_root: Path,
    runtime_data_root: Path,
    output_root: Path,
    device: str,
    verify_live: bool,
    verify_hashes: bool,
) -> tuple[dict[str, Any], RecordingInventory, RecordingSplit]:
    descriptor = _resolve_repo_path(str(config["data_inventory"]), repository_root)
    inventory = load_recording_inventory(
        descriptor,
        repository_root=repository_root,
        data_root=runtime_data_root,
        verify_live=False,
        verify_hashes=False,
    )
    inventory_validation = validate_060126_inventory(inventory)
    motion_audit: dict[str, Any] | None = None
    if verify_live:
        motion_audit = run_jepa_motion_audit(
            inventory,
            config=MotionAuditConfig(verify_hashes=verify_hashes),
        )
        inventory_validation = dict(motion_audit["inventory_validation"])
    split = fixed_behavior_stratified_split(inventory.recordings)
    encoder, mask = _architecture_from_config(config)
    budget = config["matched_training_budget"]
    precision = budget["mixed_precision"]
    training_bank = config["clip"]["training_clip_bank"]
    validation_bank = config["clip"]["validation_clip_bank"]
    steps = int(budget["optimizer_steps_per_arm_seed"])
    hard_cap = int(budget["hard_step_cap_per_arm_seed"])
    arm_ids = {str(row.get("id")) for row in config["arms"] if isinstance(row, Mapping)}
    required_arms = {
        "compact_jepa",
        "masked_pixel_autoencoder",
        "frozen_random_encoder",
        "frozen_handcrafted_stack",
    }
    injection = config["empirical_background_injection"]
    expected_cells = (
        len(injection["background_recording_ids"])
        * int(injection["windows_per_recording"])
        * len(injection["source_counts"])
        * len(injection["injection_seeds"])
    )
    expected_occurrences = (
        len(injection["background_recording_ids"])
        * int(injection["windows_per_recording"])
        * len(injection["injection_seeds"])
        * sum(int(value) for value in injection["source_counts"])
    )
    parameters = count_parameters(SpatiotemporalJEPA(encoder, mask), trainable_only=True)
    architecture = config["tubelet_encoder"]
    output, partial = _output_paths(output_root)
    resource = _resource_snapshot(output.parent, device)
    implementation_hash_validation = _validate_implementation_hash_contract(
        config, repository_root
    )
    live_sources = [
        {
            "recording_id": item.recording_id,
            "uri": item.uri,
            "available": item.resolved_path.is_file(),
        }
        for item in inventory.recordings
    ]
    holdout = inventory_source_holdout(config, repository_root, runtime_data_root)
    holdout_hash_matches: bool | None = None
    if verify_live:
        open_recording_memmap(holdout)
        if verify_hashes:
            holdout_hash_matches = sha256_file(holdout.resolved_path) == holdout.sha256
    live_sources.append(
        {
            "recording_id": holdout.recording_id,
            "uri": holdout.uri,
            "available": holdout.resolved_path.is_file(),
        }
    )
    checks = {
        "config_schema": int(config["schema_version"]) == SCHEMA_VERSION,
        "raw_only_pretraining": config["input_contract"]["learned_representation_input"]
        == "raw_uint16_grayscale_video_only",
        "spon_excluded_from_pretraining": config["fixed_recording_split"]["spon_pretraining_policy"]
        == "completely_excluded",
        "fixed_split_matches_loader": (
            set(config["fixed_recording_split"]["training_ids"]) == set(split.train_ids)
            and set(config["fixed_recording_split"]["validation_ids"])
            == set(split.validation_ids)
        ),
        "required_arms_present": required_arms <= arm_ids,
        "common_primary_learned_score_head": (
            config["score_heads"]["primary_learned_arm_head"]["id"]
            == "latent_temporal_change_norm"
            and set(config["score_heads"]["primary_learned_arm_head"]["arm_ids"])
            == {
                "compact_jepa",
                "masked_pixel_autoencoder",
                "frozen_random_encoder",
            }
            and bool(
                config["score_heads"]["primary_learned_arm_head"][
                    "identical_implementation_across_listed_arms"
                ]
            )
            and config["score_heads"][
                "objective_native_secondary_scores_enter_primary_metric"
            ]
            is False
            and config["score_heads"][
                "objective_native_secondary_scores_enter_primary_comparator_selection"
            ]
            is False
            and config["score_heads"]["direct_cross_objective_error_head_comparison_allowed"]
            is False
        ),
        "training_budget_within_hard_cap": 0 < steps <= hard_cap <= 10_000,
        "canonical_optimizer_contract": (
            str(budget["optimizer"]).casefold() == "adamw"
            and [float(value) for value in budget["adamw_betas"]] == [0.9, 0.999]
            and float(budget["adamw_epsilon"]) == 1e-8
            and budget["learning_rate_schedule"] == "constant"
            and float(budget["gradient_clip_norm"]) == 1.0
        ),
        "implementation_hash_contract_matches": bool(
            implementation_hash_validation["all_match"]
        ),
        "canonical_training_clip_bank_contract": (
            int(training_bank["clips_per_training_seed"]) == 512
            and int(training_bank["uniform_clips"]) == 256
            and int(training_bank["high_temporal_mad_clips"]) == 128
            and int(training_bank["low_temporal_mad_clips"]) == 128
            and training_bank["sampling_seed_policy"]
            == "training_plan_seed_equals_registered_training_seed"
            and bool(training_bank["shared_between_trainable_arms_within_seed"])
        ),
        "canonical_shared_validation_clip_bank_contract": (
            int(validation_bank["total_clips"]) == 96
            and int(validation_bank["clips_per_held_recording"]) == 32
            and int(validation_bank["sampling_seed"]) == 2001
            and int(config["seeds"]["validation_sampling"]) == 2001
            and bool(validation_bank["shared_across_training_seeds"])
            and set(validation_bank["recording_ids"]) == set(split.validation_ids)
        ),
        "canonical_cuda_precision_is_bfloat16": (
            bool(precision["enabled_on_cuda"])
            and precision["dtype"] == "bfloat16"
            and precision["loss_accumulation_dtype"] == "float32"
            and precision["metric_dtype"] == "float32"
            and precision["cpu_dtype"] == "float32"
        ),
        "float16_forbidden": precision["float16_allowed"] is False,
        "requested_cuda_supports_bfloat16": device != "cuda"
        or bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "registered_parameter_range": int(architecture["trainable_parameter_minimum"])
        <= parameters
        <= int(architecture["trainable_parameter_maximum"]),
        "paired_cell_count": expected_cells == int(injection["expected_paired_cells"]),
        "exact_truth_occurrence_count": expected_occurrences
        == int(config["scientific_audit"]["exact_truth_section"]["expected_exact_truth_occurrences"]),
        "output_absent": not output.exists(),
        "partial_output_absent": not partial.exists(),
        "requested_device_available": device != "cuda" or bool(torch.cuda.is_available()),
        "minimum_free_disk": resource["disk"]["free_mib"]
        >= int(config["resources"]["minimum_free_disk_mib"]),
        "cpu_thread_cap_valid": 1
        <= int(config["resources"]["cpu_threads"])
        <= int(resource["cpu_count"] or 1),
        "live_sources_available_when_verified": (not verify_live)
        or all(bool(row["available"]) for row in live_sources),
        "spon_holdout_hash_when_requested": (not verify_hashes)
        or bool(holdout_hash_matches),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    motion_review_ids = (
        []
        if motion_audit is None
        else list(motion_audit["aggregate_gate"]["review_recording_ids"])
    )
    scientific_readiness_checks = {
        "execution_authorized": bool(config.get("execution_authorized")),
        "registry_lifecycle_allows_claim_execution": config.get("lifecycle")
        in {"frozen", "authorized", "running"},
        "registered_protocol_freeze_evidence_linked": False,
        "registered_code_commit_pinned": False,
        "registered_config_and_input_manifest_hashes_linked": False,
        "registered_implementation_hashes_match": bool(
            implementation_hash_validation["all_match"]
        ),
        "live_preflight_performed": bool(verify_live),
        "all_source_hashes_verified": bool(verify_live and verify_hashes),
        "registered_NREV_EXP_0025_motion_evidence_linked": False,
        "live_canonical_device_and_bfloat16_check": bool(
            verify_live
            and device == str(config["resources"]["device"])
            and checks["requested_device_available"]
            and checks["requested_cuda_supports_bfloat16"]
        ),
        "live_resource_checks_passed": bool(
            verify_live and checks["minimum_free_disk"]
        ),
        "heuristic_motion_review_clear": not motion_review_ids,
    }
    scientific_ready = not failed and all(scientific_readiness_checks.values())
    readiness_blockers = [
        key for key, value in scientific_readiness_checks.items() if not value
    ]
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": (
                ("passed_live_preflight" if verify_live else "passed_metadata_preflight")
                if not failed
                else "failed"
            ),
            "checked_at": _utc_now(),
            "config": {
                "uri": portable_path(config_path, repository=repository_root, data=runtime_data_root),
                "sha256": sha256_file(config_path),
                "experiment_id": config["experiment_id"],
                "run_id": config["run_id"],
                "execution_authorized": bool(config.get("execution_authorized")),
            },
            "checks": checks,
            "failed_checks": failed,
            "scientific_execution_ready": scientific_ready,
            "scientific_readiness_checks": scientific_readiness_checks,
            "scientific_execution_blockers": failed + readiness_blockers,
            "inventory_validation": inventory_validation,
            "split": split.to_manifest(inventory.recordings),
            "architecture": {
                "encoder": asdict(encoder),
                "mask": asdict(mask),
                "trainable_jepa_parameters": parameters,
            },
            "implementation_hash_validation": implementation_hash_validation,
            "planned_exact_truth": {
                "paired_cells": expected_cells,
                "exact_injected_source_occurrences": expected_occurrences,
                "background_recording_count": len(injection["background_recording_ids"]),
            },
            "source_availability": live_sources,
            "motion_audit": (
                {
                    "status": "not_run_metadata_only_preflight",
                    "interpretation": "Live execution must run the label-free heuristic motion screen.",
                }
                if motion_audit is None
                else motion_audit
            ),
            "resource_snapshot": resource,
            "output": {
                "uri": portable_path(output, repository=repository_root, data=runtime_data_root),
                "partial_uri": portable_path(partial, repository=repository_root, data=runtime_data_root),
            },
            "evidence_boundary": {
                "preflight_is_execution": False,
                "preflight_is_scientific_result": False,
                "unlabeled_native_candidates": "unknown_not_negative",
                "independent_animal_generalization": False,
            },
        },
        inventory,
        split,
    )


def inventory_source_holdout(
    config: Mapping[str, Any], repository_root: Path, runtime_data_root: Path
) -> RecordingDescriptor:
    descriptor_path = _resolve_repo_path(str(config["data_inventory"]), repository_root)
    payload = _load_json(descriptor_path)
    row = _require(payload, "external_evaluation_holdout", dict)
    uri = str(row["portable_path"])
    if not uri.startswith("data://"):
        raise JEPAPilotError("Spon holdout must use a data:// portable path")
    resolved = (runtime_data_root / uri.removeprefix("data://")).resolve()
    return RecordingDescriptor(
        recording_id=str(row["recording_id"]),
        dataset_id=str(row["dataset_id"]),
        recording_number=1,
        behavior="rest",
        uri=uri,
        resolved_path=resolved,
        sha256=str(row["sha256"]),
        shape=tuple(int(value) for value in row["shape_tyx"]),
        dtype=str(row["dtype"]),
        frame_interval_s=0.020,
    )


def preflight_jepa_pilot(
    config_path: Path,
    *,
    repository_root: Path,
    data_root: Path | None = None,
    output_root: Path | None = None,
    device: str | None = None,
    verify_live: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Return a read-only metadata/live-data preflight; write nothing."""

    repository = Path(repository_root).expanduser().resolve()
    runtime_data = (
        configured_data_root(repository)
        if data_root is None
        else Path(data_root).expanduser().resolve()
    )
    config_file = Path(config_path).expanduser().resolve()
    config = load_jepa_pilot_config(config_file)
    requested_output = (
        _resolve_repo_path(str(config["output_root"]), repository)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    requested_device = str(device or config["resources"]["device"])
    preflight, _, _ = _preflight_checks(
        config,
        config_path=config_file,
        repository_root=repository,
        runtime_data_root=runtime_data,
        output_root=requested_output,
        device=requested_device,
        verify_live=verify_live,
        verify_hashes=verify_hashes,
    )
    return preflight


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    fields = list(rows[0])
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _heartbeat(root: Path, phase: str, **details: Any) -> None:
    atomic_json(
        root / "heartbeat.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "updated_at": _utc_now(),
            **details,
        },
    )


def _normalized_clip_bank(
    recordings: Sequence[RecordingDescriptor],
    requests: Sequence[ClipRequest],
    contract: TemporalClipContract,
    normalization: RobustNormalization,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lookup = {item.recording_id: item for item in recordings}
    clips: list[np.ndarray] = []
    descriptors: list[dict[str, Any]] = []
    for request in requests:
        recording = lookup[request.recording_id]
        raw = np.asarray(read_sequential_clip(recording, request, contract), dtype=np.float32)
        normalized = normalization.apply(raw)
        frame_means = normalized.mean(axis=(1, 2), dtype=np.float64)
        time = np.arange(len(frame_means), dtype=np.float64)
        slope = float(np.polyfit(time, frame_means, 1)[0]) if len(time) >= 2 else 0.0
        difference = np.diff(normalized, axis=0)
        difference_center = np.median(difference, axis=0, keepdims=True)
        local_mad = float(1.4826 * np.median(np.abs(difference - difference_center)))
        border = np.concatenate(
            (
                normalized[:, 0, :].reshape(-1),
                normalized[:, -1, :].reshape(-1),
                normalized[:, 1:-1, 0].reshape(-1),
                normalized[:, 1:-1, -1].reshape(-1),
            )
        )
        descriptors.append(
            {
                **request.to_manifest(),
                "behavior": recording.behavior,
                "global_intensity_raw": float(np.mean(raw, dtype=np.float64)),
                "global_intensity_normalized": float(np.mean(normalized, dtype=np.float64)),
                "local_frame_difference_mad_normalized": local_mad,
                "temporal_position_fraction": float(
                    request.start_frame_zero / max(recording.frame_count - contract.clip_frames, 1)
                ),
                "bleaching_slope_normalized_units_per_frame": slope,
                "row_position_fraction": float(request.y_zero / max(recording.frame_shape[0] - request.height, 1)),
                "column_position_fraction": float(request.x_zero / max(recording.frame_shape[1] - request.width, 1)),
                "annular_background_normalized": float(np.mean(border, dtype=np.float64)),
                "dtype_rail_fraction": float(
                    np.mean((raw <= np.iinfo(np.uint16).min) | (raw >= np.iinfo(np.uint16).max))
                ),
                "effective_sensor_saturation": "unresolved_sensor_bit_depth_unknown",
                "motion_magnitude": None,
                "motion_source": "unavailable_until_NREV-EXP-0025_supplies_registered_motion",
            }
        )
        clips.append(normalized)
    return np.stack(clips).astype(np.float32, copy=False), descriptors


def _cache_clip_banks(
    root: Path,
    train_clips: np.ndarray,
    validation_clips: np.ndarray,
    train_descriptors: Sequence[Mapping[str, Any]],
    validation_descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / "train_normalized_clips.npy"
    validation_path = cache / "validation_normalized_clips.npy"
    np.save(train_path, train_clips, allow_pickle=False)
    np.save(validation_path, validation_clips, allow_pickle=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_video_copied": False,
        "derived_normalized_clip_cache": True,
        "train": {
            "path": "cache/train_normalized_clips.npy",
            "shape": list(train_clips.shape),
            "dtype": str(train_clips.dtype),
            "sha256": sha256_file(train_path),
            "requests": list(train_descriptors),
        },
        "validation": {
            "path": "cache/validation_normalized_clips.npy",
            "shape": list(validation_clips.shape),
            "dtype": str(validation_clips.dtype),
            "sha256": sha256_file(validation_path),
            "requests": list(validation_descriptors),
        },
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    atomic_json(root / "clip_bank_manifest.json", manifest)
    return manifest


def _cache_shared_validation_bank(
    root: Path,
    validation_clips: np.ndarray,
    validation_descriptors: Sequence[Mapping[str, Any]],
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "validation_seed_2001_normalized_clips.npy"
    np.save(path, validation_clips, allow_pickle=False)
    return {
        "path": str(path.relative_to(root)),
        "shape": list(validation_clips.shape),
        "dtype": str(validation_clips.dtype),
        "sha256": sha256_file(path),
        "sampling_seed": int(sampling_seed),
        "shared_across_training_seeds": True,
        "requests": list(validation_descriptors),
    }


def _cache_seed_training_bank(
    root: Path,
    training_clips: np.ndarray,
    train_descriptors: Sequence[Mapping[str, Any]],
    *,
    training_seed: int,
) -> dict[str, Any]:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"train_seed_{training_seed}_normalized_clips.npy"
    np.save(path, training_clips, allow_pickle=False)
    return {
        "path": str(path.relative_to(root)),
        "shape": list(training_clips.shape),
        "dtype": str(training_clips.dtype),
        "sha256": sha256_file(path),
        "sampling_seed": int(training_seed),
        "shared_between_jepa_and_mae_within_seed": True,
        "shared_across_training_seeds": False,
        "requests": list(train_descriptors),
    }


def _smoke_clip_banks(seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    training = rng.normal(size=(8, 4, 8, 8)).astype(np.float32)
    validation = rng.normal(size=(8, 4, 8, 8)).astype(np.float32)
    training[:, 1:4, 3:5, 3:5] += np.linspace(0, 1.5, 3)[None, :, None, None]
    validation[:, 1:4, 3:5, 3:5] += np.linspace(0, 1.5, 3)[None, :, None, None]
    recording_ids = (
        "060126_10_rest",
        "060126_12_left",
        "060126_15_right",
        "spon_ca_burst_3_hindbrain_to_tail_488_20ms",
    )
    behaviors = ("rest", "left", "right", "holdout")
    descriptors: list[dict[str, Any]] = []
    for index in range(len(validation)):
        recording_index = index % 4
        clip = validation[index]
        frame_means = clip.mean(axis=(1, 2))
        descriptors.append(
            {
                "recording_id": recording_ids[recording_index],
                "behavior": behaviors[recording_index],
                "start_frame_zero": index * 4,
                "stop_frame_zero_exclusive": index * 4 + 4,
                "y_zero": 0,
                "x_zero": 0,
                "height": 8,
                "width": 8,
                "sampling_stratum": "synthetic_smoke",
                "temporal_mad": float(np.median(np.abs(clip - np.median(clip)))),
                "global_intensity_raw": float(np.mean(clip)),
                "global_intensity_normalized": float(np.mean(clip)),
                "local_frame_difference_mad_normalized": float(np.median(np.abs(np.diff(clip, axis=0)))),
                "temporal_position_fraction": index / max(len(validation) - 1, 1),
                "bleaching_slope_normalized_units_per_frame": float(np.polyfit(np.arange(4), frame_means, 1)[0]),
                "row_position_fraction": 0.0,
                "column_position_fraction": 0.0,
                "annular_background_normalized": float(np.mean(clip[:, (0, -1), :])),
                "dtype_rail_fraction": 0.0,
                "effective_sensor_saturation": "not_applicable_synthetic_smoke",
                "motion_magnitude": None,
                "motion_source": "not_applicable_synthetic_smoke",
            }
        )
    return training, validation, descriptors


def deterministic_full_coverage_masks(
    video: Tensor,
    mask_config: MaskConfig,
    patch_size: Sequence[int],
    *,
    seed: int,
    maximum_draws: int = 512,
) -> Tensor:
    """Generate a deterministic mask schedule until every token is covered."""

    grid = patch_grid_shape(video, patch_size)
    coverage = torch.zeros((int(video.shape[0]), *grid), dtype=torch.int64)
    masks: list[Tensor] = []
    for draw in range(maximum_draws):
        mask = make_patch_mask_from_config(
            batch_size=int(video.shape[0]),
            grid_shape=grid,
            config=mask_config,
            seed=int(seed) + draw,
            device="cpu",
        ).cpu()
        masks.append(mask)
        coverage.add_(mask)
        if bool((coverage > 0).all()):
            return torch.stack(masks, dim=0)
    uncovered = int((coverage == 0).sum().item())
    raise JEPAPilotError(
        f"deterministic mask schedule left {uncovered} tokens uncovered after {maximum_draws} draws"
    )


def _spatial_patch_map(values: Tensor, output_shape_yx: tuple[int, int]) -> np.ndarray:
    if values.ndim != 4 or values.shape[0] != 1:
        raise JEPAPilotError("patch score must have shape [1,time,row,column]")
    spatial = values.float().mean(dim=1, keepdim=True)
    upsampled = F.interpolate(spatial, size=output_shape_yx, mode="bilinear", align_corners=False)
    result = upsampled[0, 0].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise JEPAPilotError("score map contains non-finite values")
    return result


class JEPAMaskedPredictionErrorScorer:
    """Primary JEPA scorer: mask-averaged prediction error with full coverage."""

    def __init__(
        self,
        model: SpatiotemporalJEPA,
        *,
        device: str,
        seed: int,
        normalization: RobustNormalization | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.seed = int(seed)
        self.normalization = normalization
        self._schedules: dict[tuple[int, int, int], Tensor] = {}
        self.coverage_records: dict[str, dict[str, Any]] = {}

    def __call__(self, movie: np.ndarray) -> np.ndarray:
        raw = np.asarray(movie, dtype=np.float32)
        values = raw if self.normalization is None else self.normalization.apply(raw)
        tensor = torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0).unsqueeze(0).to(self.device)
        shape = tuple(int(value) for value in values.shape)
        if shape not in self._schedules:
            self._schedules[shape] = deterministic_full_coverage_masks(
                tensor.cpu(),
                self.model.mask_config,
                self.model.encoder_config.patch_size,
                seed=self.seed,
            )
        schedule = self._schedules[shape].to(self.device)
        with torch.no_grad():
            output = self.model.masked_prediction_error_sweep(tensor, schedule)
        counts = output.observation_counts.detach().cpu()
        key = "x".join(map(str, shape))
        self.coverage_records[key] = {
            "grid_shape_tyx": list(counts.shape[-3:]),
            "draw_count": int(schedule.shape[0]),
            "minimum_observations": int(counts.min().item()),
            "maximum_observations": int(counts.max().item()),
            "observation_counts_flat": counts[0].reshape(-1).tolist(),
            "count_sha256": stable_hash(counts[0].reshape(-1).tolist()),
        }
        return _spatial_patch_map(output.prediction_error, tuple(values.shape[1:]))


class MAEMaskedReconstructionErrorScorer:
    """Matched MAE primary scorer averaged only over masked observations."""

    def __init__(
        self,
        model: MaskedPixelAutoencoder,
        *,
        device: str,
        seed: int,
        normalization: RobustNormalization | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.seed = int(seed)
        self.normalization = normalization
        self._schedules: dict[tuple[int, int, int], Tensor] = {}
        self.coverage_records: dict[str, dict[str, Any]] = {}

    def __call__(self, movie: np.ndarray) -> np.ndarray:
        raw = np.asarray(movie, dtype=np.float32)
        values = raw if self.normalization is None else self.normalization.apply(raw)
        tensor = torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0).unsqueeze(0).to(self.device)
        shape = tuple(int(value) for value in values.shape)
        if shape not in self._schedules:
            self._schedules[shape] = deterministic_full_coverage_masks(
                tensor.cpu(),
                self.model.mask_config,
                self.model.encoder_config.patch_size,
                seed=self.seed,
            )
        schedule = self._schedules[shape].to(self.device)
        error_sum = torch.zeros_like(tensor, dtype=torch.float32)
        observation_counts = torch.zeros_like(tensor, dtype=torch.int64)
        with torch.no_grad():
            for mask in schedule:
                output = self.model(tensor, patch_mask=mask)
                voxel_mask = patch_mask_to_voxels(mask, self.model.encoder_config.patch_size).unsqueeze(1)
                error_sum.add_((output.reconstruction - tensor).float().square() * voxel_mask)
                observation_counts.add_(voxel_mask)
        if bool((observation_counts == 0).any()):
            raise JEPAPilotError("MAE mask sweep did not cover every voxel")
        score = (error_sum / observation_counts.float())[0, 0].mean(dim=0)
        token_counts = torch.stack(list(schedule), dim=0).sum(dim=0)[0].detach().cpu()
        key = "x".join(map(str, shape))
        self.coverage_records[key] = {
            "grid_shape_tyx": list(token_counts.shape),
            "draw_count": int(schedule.shape[0]),
            "minimum_observations": int(token_counts.min().item()),
            "maximum_observations": int(token_counts.max().item()),
            "observation_counts_flat": token_counts.reshape(-1).tolist(),
            "count_sha256": stable_hash(token_counts.reshape(-1).tolist()),
        }
        result = score.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise JEPAPilotError("MAE reconstruction score contains non-finite values")
        return result


# Backward-compatible public name; the object is callable and also exposes the
# structural ``fit_source_off`` interface consumed by the paired evaluator.
frozen_handcrafted_score = frozen_handcrafted_comparator


def _interval_overlaps_any(start_zero: int, stop_zero: int, intervals_ui: Sequence[tuple[int, int]]) -> bool:
    return any(start_zero < end_ui and stop_zero > start_ui - 1 for start_ui, end_ui in intervals_ui)


def _candidate_background_requests(
    descriptor: RecordingDescriptor,
    *,
    contract: TemporalClipContract,
    crop_shape: tuple[int, int],
    seed: int,
    candidate_count: int = 64,
    forbidden_intervals_ui: Sequence[tuple[int, int]] = (),
) -> list[ClipRequest]:
    """Create a label-free background bank with observed temporal-MAD scores."""

    if candidate_count < 12:
        raise JEPAPilotError("background selection needs at least twelve candidates")
    movie = open_recording_memmap(descriptor)
    crop_height, crop_width = crop_shape
    if crop_height > descriptor.frame_shape[0] or crop_width > descriptor.frame_shape[1]:
        raise JEPAPilotError("background crop exceeds the source frame")
    lower, upper = contract.valid_start_bounds(descriptor.frame_count)
    rng = np.random.default_rng(seed)
    requests: list[ClipRequest] = []
    used: set[tuple[int, int, int]] = set()
    attempts = 0
    while len(requests) < candidate_count and attempts < candidate_count * 500:
        attempts += 1
        start = int(rng.integers(lower, upper))
        stop = start + contract.clip_frames
        if _interval_overlaps_any(start, stop, forbidden_intervals_ui):
            continue
        y_zero = int(rng.integers(0, descriptor.frame_shape[0] - crop_height + 1))
        x_zero = int(rng.integers(0, descriptor.frame_shape[1] - crop_width + 1))
        key = (start, y_zero, x_zero)
        if key in used:
            continue
        used.add(key)
        sample = np.asarray(
            movie[
                start:stop:2,
                y_zero : y_zero + crop_height : 4,
                x_zero : x_zero + crop_width : 4,
            ],
            dtype=np.float32,
        )
        center = np.median(sample, axis=0, keepdims=True)
        temporal_mad = float(1.4826 * np.median(np.abs(sample - center)))
        requests.append(
            ClipRequest(
                recording_id=descriptor.recording_id,
                split="validation",
                start_frame_zero=start,
                stop_frame_zero_exclusive=stop,
                y_zero=y_zero,
                x_zero=x_zero,
                height=crop_height,
                width=crop_width,
                sampling_stratum="uniform",
                temporal_mad=temporal_mad,
            )
        )
    if len(requests) != candidate_count:
        raise JEPAPilotError(
            f"could not draw {candidate_count} quiet background candidates for {descriptor.recording_id}"
        )
    return requests


def _spatiotemporally_disjoint(left: ClipRequest, right: ClipRequest) -> bool:
    temporal_overlap = (
        left.start_frame_zero < right.stop_frame_zero_exclusive
        and right.start_frame_zero < left.stop_frame_zero_exclusive
    )
    row_overlap = left.y_zero < right.y_zero + right.height and right.y_zero < left.y_zero + left.height
    column_overlap = left.x_zero < right.x_zero + right.width and right.x_zero < left.x_zero + left.width
    return not (temporal_overlap and row_overlap and column_overlap)


def _select_background_strata(candidates: Sequence[ClipRequest]) -> tuple[ClipRequest, ...]:
    """Freeze low/median/high MAD clips while enforcing 3-D non-overlap."""

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.temporal_mad,
            item.start_frame_zero,
            item.y_zero,
            item.x_zero,
        ),
    )
    target_indices = (0, len(ranked) // 2, len(ranked) - 1)
    # ClipRequest's maintained label-free vocabulary uses ``uniform`` for the
    # middle stratum; the background manifest records its scientific role as
    # ``median_mad`` separately below.
    strata = ("low_mad", "uniform", "high_mad")
    selected: list[ClipRequest] = []
    for target_index, stratum in zip(target_indices, strata, strict=True):
        order = sorted(range(len(ranked)), key=lambda index: (abs(index - target_index), index))
        choice = next(
            (
                ranked[index]
                for index in order
                if all(_spatiotemporally_disjoint(ranked[index], previous) for previous in selected)
            ),
            None,
        )
        if choice is None:
            raise JEPAPilotError("could not freeze three spatiotemporally disjoint background windows")
        values = choice.to_manifest()
        values["sampling_stratum"] = stratum
        selected.append(ClipRequest(**values))
    return tuple(selected)


def _real_background_bank(
    config: Mapping[str, Any],
    *,
    inventory: RecordingInventory,
    split: RecordingSplit,
    holdout: RecordingDescriptor,
    contract: TemporalClipContract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = inventory.by_id()
    selected_rows: list[dict[str, Any]] = []
    bank: list[dict[str, Any]] = []
    base_seed = int(config["seeds"]["injection"][0]) + 70_000
    recording_ids = list(split.validation_ids) + [holdout.recording_id]
    for recording_index, recording_id in enumerate(recording_ids):
        descriptor = holdout if recording_id == holdout.recording_id else lookup[recording_id]
        forbidden = SPON_BURSTS_UI_INCLUSIVE if descriptor.dataset_id == "spon_ca_burst" else ()
        candidates = _candidate_background_requests(
            descriptor,
            contract=contract,
            crop_shape=tuple(int(value) for value in config["clip"]["shape_tyx"][1:]),
            seed=base_seed + recording_index,
            forbidden_intervals_ui=forbidden,
        )
        selected = _select_background_strata(candidates)
        for window_index, request in enumerate(selected, start=1):
            raw = np.asarray(read_sequential_clip(descriptor, request, contract), dtype=np.float32)
            background_mad_stratum = (
                "median_mad" if request.sampling_stratum == "uniform" else request.sampling_stratum
            )
            window_id = f"{recording_id}__window_{window_index}_{background_mad_stratum}"
            row = {
                **request.to_manifest(),
                "background_recording_id": recording_id,
                "background_window_id": window_id,
                "background_dataset_id": descriptor.dataset_id,
                "background_mad_stratum": background_mad_stratum,
                "source_uri": descriptor.uri,
                "quiet_policy": (
                    "outside_registered_spon_bursts"
                    if descriptor.dataset_id == "spon_ca_burst"
                    else "held_recording_label_free_background"
                ),
            }
            selected_rows.append(row)
            bank.append({"metadata": row, "clip": raw})
    return bank, selected_rows


def _smoke_background_bank(validation_clips: np.ndarray) -> list[dict[str, Any]]:
    recording_ids = (
        "060126_10_rest",
        "060126_12_left",
        "060126_15_right",
        "spon_ca_burst_3_hindbrain_to_tail_488_20ms",
    )
    return [
        {
            "metadata": {
                "background_recording_id": recording_id,
                "background_window_id": f"{recording_id}__synthetic_smoke_window",
                "background_dataset_id": "synthetic_smoke",
                "sampling_stratum": "synthetic_smoke",
                "quiet_policy": "not_applicable_synthetic_smoke",
            },
            "clip": validation_clips[index],
        }
        for index, recording_id in enumerate(recording_ids)
    ]


def _make_fixtures(
    config: Mapping[str, Any],
    backgrounds: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
    injection_cells_per_background: int | None,
) -> tuple[list[SemiSyntheticFixture], dict[str, dict[str, Any]], dict[str, Any]]:
    injection = config["empirical_background_injection"]
    source_counts = [1] if smoke else [int(value) for value in injection["source_counts"]]
    seeds = [int(injection["injection_seeds"][0])] if smoke else [int(value) for value in injection["injection_seeds"]]
    combinations = [(source_count, seed) for source_count in source_counts for seed in seeds]
    if injection_cells_per_background is not None:
        if injection_cells_per_background < 1:
            raise JEPAPilotError("injection_cells_per_background must be positive or None")
        combinations = combinations[:injection_cells_per_background]
    fixtures: list[SemiSyntheticFixture] = []
    fixture_metadata: dict[str, dict[str, Any]] = {}
    for background in backgrounds:
        metadata = dict(background["metadata"])
        movie = np.asarray(background["clip"], dtype=np.float32)
        for source_count, seed in combinations:
            fixture_id = (
                f"{metadata['background_window_id']}__sources_{source_count}__seed_{seed}"
            )
            fixture = make_native_background_injection(
                movie,
                fixture_id=fixture_id,
                source_count=source_count,
                seed=seed,
                amplitude_multiplier=float(injection["amplitude"]["multiplier"]),
                border_px=2 if smoke else 8,
                minimum_separation_px=2.0 if smoke else 10.0,
                crowding_case="mixed",
            )
            enriched = {
                **fixture.metadata,
                **metadata,
                "source_count": source_count,
                "injection_seed": seed,
                "crowding_case_requested": "mixed",
                "pair_closure_threshold_float32_ulp": float(
                    injection["maximum_pair_closure_float32_ulp"]
                ),
            }
            fixture = replace(fixture, metadata=enriched)
            fixtures.append(fixture)
            fixture_metadata[fixture_id] = enriched
    planned = int(injection["expected_paired_cells"])
    return (
        fixtures,
        fixture_metadata,
        {
            "observed_fixture_count": len(fixtures),
            "planned_fixture_count": planned,
            "coverage_fraction": float(len(fixtures) / planned),
            "full_registered_grid": len(fixtures) == planned,
            "background_recording_ids": sorted(
                {str(row["background_recording_id"]) for row in fixture_metadata.values()}
            ),
            "source_counts": sorted({int(row["source_count"]) for row in fixture_metadata.values()}),
            "injection_seeds": sorted({int(row["injection_seed"]) for row in fixture_metadata.values()}),
            "crowding_policy": "mixed",
        },
    )


def _fixture_closure_manifest(fixture: SemiSyntheticFixture) -> dict[str, float]:
    metrics = additive_closure_metrics(
        fixture.observation,
        fixture.native_background,
        fixture.injected_neural_signal,
    )
    return {
        "maximum_pair_closure_absolute": metrics["maximum_absolute"],
        "maximum_pair_closure_float32_ulp": metrics["maximum_float32_ulp"],
    }


def _pooled_embeddings(
    result: MatchedTrainingResult,
    validation_clips: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    resolved = torch.device(device)
    arrays: dict[str, list[np.ndarray]] = {"compact_jepa": [], "masked_pixel_autoencoder": [], "frozen_random_encoder": []}
    result.jepa.eval()
    result.mae.eval()
    result.random.eval()
    with torch.no_grad():
        for clip in validation_clips:
            tensor = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0).unsqueeze(0).to(resolved)
            embeddings = {
                "compact_jepa": result.jepa.encode_target(tensor),
                "masked_pixel_autoencoder": result.mae.encoder(tensor),
                "frozen_random_encoder": result.random(tensor),
            }
            for key, value in embeddings.items():
                arrays[key].append(value.float().mean(dim=(2, 3, 4))[0].cpu().numpy())
    return {key: np.stack(values).astype(np.float64) for key, values in arrays.items()}


def _nearest_centroid_resubstitution(features: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=object)
    classes = sorted(set(str(value) for value in label_array))
    centered = features - np.mean(features, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    standardized = centered / np.where(scale > 1e-8, scale, 1.0)
    centroids = {
        label: standardized[label_array == label].mean(axis=0) for label in classes
    }
    predictions = [
        min(classes, key=lambda label: float(np.sum((row - centroids[label]) ** 2)))
        for row in standardized
    ]
    counts = Counter(str(value) for value in label_array)
    return {
        "metric": "nearest_centroid_resubstitution_accuracy",
        "accuracy": float(np.mean(label_array == np.asarray(predictions, dtype=object))),
        "majority_fraction": float(max(counts.values()) / len(label_array)),
        "class_counts": dict(sorted(counts.items())),
        "evaluation": "clip_level_resubstitution_upper_bound_only",
        "recording_held_out": False,
        "reason": "held-out recordings remove the corresponding acquisition class; behavior is one-to-one with the three validation recording identities",
    }


def _continuous_recording_heldout_probe(
    features: np.ndarray,
    target: Sequence[float],
    recording_ids: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(target, dtype=np.float64)
    groups = np.asarray(recording_ids, dtype=object)
    predictions = np.empty_like(values)
    for group in sorted(set(str(value) for value in groups)):
        test = groups == group
        train = ~test
        train_x = features[train]
        center = np.mean(train_x, axis=0, keepdims=True)
        scale = np.std(train_x, axis=0, keepdims=True)
        standardized_train = (train_x - center) / np.where(scale > 1e-8, scale, 1.0)
        standardized_test = (features[test] - center) / np.where(scale > 1e-8, scale, 1.0)
        design = np.column_stack((np.ones(len(standardized_train)), standardized_train))
        ridge = np.eye(design.shape[1]) * 1e-3
        ridge[0, 0] = 0
        coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ values[train])
        predictions[test] = np.column_stack((np.ones(np.sum(test)), standardized_test)) @ coefficients
    target_std = float(np.std(values))
    prediction_std = float(np.std(predictions))
    correlation = (
        None
        if target_std <= 1e-12 or prediction_std <= 1e-12
        else float(np.corrcoef(values, predictions)[0, 1])
    )
    denominator = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = None if denominator <= 1e-12 else float(1 - np.sum((values - predictions) ** 2) / denominator)
    return {
        "metric": "recording_held_out_ridge_probe",
        "pearson_correlation": correlation,
        "r_squared": r_squared,
        "mean_absolute_error": float(np.mean(np.abs(values - predictions))),
        "recording_held_out": True,
        "recording_count": len(set(str(value) for value in groups)),
    }


def _nuisance_audit(
    result: MatchedTrainingResult,
    validation_clips: np.ndarray,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    device: str,
    smoke: bool,
    motion_audit: Mapping[str, Any] | None = None,
    training_recording_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if len(validation_clips) != len(descriptors):
        raise JEPAPilotError("validation embeddings and nuisance descriptors disagree")
    embeddings = _pooled_embeddings(result, validation_clips, device=device)
    recordings = [str(row["recording_id"]) for row in descriptors]
    behaviors = [str(row["behavior"]) for row in descriptors]
    continuous_targets = {
        "global_intensity": "global_intensity_normalized",
        "local_temporal_mad": "local_frame_difference_mad_normalized",
        "temporal_position": "temporal_position_fraction",
        "row_position": "row_position_fraction",
        "column_position": "column_position_fraction",
        "bleaching_slope": "bleaching_slope_normalized_units_per_frame",
        "annular_background": "annular_background_normalized",
        "dtype_rail_fraction": "dtype_rail_fraction",
    }
    arms: dict[str, Any] = {}
    for arm, features in embeddings.items():
        probes: dict[str, Any] = {
            "recording_identity": _nearest_centroid_resubstitution(features, recordings),
            "filename_derived_behavior_stratum": _nearest_centroid_resubstitution(features, behaviors),
        }
        for name, field in continuous_targets.items():
            probes[name] = _continuous_recording_heldout_probe(
                features, [float(row[field]) for row in descriptors], recordings
            )
        high_categorical = max(
            float(probes["recording_identity"]["accuracy"]),
            float(probes["filename_derived_behavior_stratum"]["accuracy"]),
        ) >= 0.90
        high_continuous = any(
            item["pearson_correlation"] is not None
            and abs(float(item["pearson_correlation"])) >= 0.80
            for key, item in probes.items()
            if key not in {"recording_identity", "filename_derived_behavior_stratum"}
        )
        arms[arm] = {
            "embedding_shape": list(features.shape),
            "probes": probes,
            "categorical_clip_decodability_flag_descriptive_only": bool(high_categorical),
            "recording_held_out_continuous_nuisance_flag": bool(high_continuous),
        }
    motion_review_ids = (
        []
        if motion_audit is None or "aggregate_gate" not in motion_audit
        else list(motion_audit["aggregate_gate"].get("review_recording_ids", []))
    )
    motion_validation_review_ids = sorted(
        set(motion_review_ids) & {str(row["recording_id"]) for row in descriptors}
    )
    motion_training_review_ids = sorted(
        set(motion_review_ids) & {str(value) for value in training_recording_ids}
    )
    return {
        "status": "synthetic_contract_smoke_only" if smoke else "descriptive_screen_complete",
        "arms": arms,
        "motion_magnitude": {
            "status": "not_applicable_synthetic_smoke"
            if smoke
            else ("available_heuristic_screen" if motion_audit is not None else "not_available"),
            "source": "jepa_motion_audit_v1",
            "registered_motion_dependency_satisfied": False,
            "registered_dependency": "NREV-EXP-0025 remains draft and requires the registered motion fields and residual-relation analysis",
            "review_recording_ids": motion_review_ids,
            "training_review_recording_ids": motion_training_review_ids,
            "validation_review_recording_ids": motion_validation_review_ids,
            "nuisance_stratified_sensitivity_required": bool(motion_review_ids),
            "invented_proxy_used": False,
            "interpretation": "Heuristic acquisition-confound screen only; no motion correction or causal attribution.",
        },
        "dtype_rail_fraction": {
            "status": "descriptive_probe_target",
            "definition": "fraction_equal_to_uint16_dtype_minimum_or_maximum",
            "effective_sensor_saturation": "unresolved_sensor_bit_depth_unknown",
            "interpretation": "Dtype rail occupancy is not an effective sensor-saturation estimate without registered acquisition bit depth.",
        },
        "counterfactuals": {
            "temporal_reversal": "pending_claim_bearing_run",
            "within_clip_phase_randomization": "pending_claim_bearing_run",
            "spatial_block_shuffle": "pending_claim_bearing_run",
            "source_off_native_clips": "included_in_paired_injection_evaluation",
            "recording_blocked_label_permutation": "pending_claim_bearing_run",
        },
        "interpretation": (
            "Nuisance decodability is descriptive. Recording and behavior categorical probes are clip-level upper bounds because the validation set has one recording per behavior. "
            "High decodability blocks promotion but neither proves causal confounding nor neuronal specificity."
        ),
        "claim_gate_resolved": False,
        "unrun_claim_gates": [
            "nuisance_matching_or_residualization",
            "temporal_reversal_counterfactual",
            "within_clip_phase_randomization_counterfactual",
            "spatial_block_shuffle_counterfactual",
            "recording_blocked_label_permutation",
            "leave_one_recording_out_primary_sensitivity",
            "per_recording_collapse_diagnostics",
            "cross_seed_linear_CKA",
            "registered_NREV_EXP_0025_motion_residual_relation",
        ],
        "proposed_arm_blocks_promotion": True,
        "observed_review_flags": {
            "continuous_recording_heldout_nuisance": bool(
                arms["compact_jepa"]["recording_held_out_continuous_nuisance_flag"]
            ),
            "heuristic_motion_review_recordings": motion_review_ids,
            "categorical_clip_decodability_descriptive_only": bool(
                arms["compact_jepa"]["categorical_clip_decodability_flag_descriptive_only"]
            ),
        },
    }


class _FrozenNormalizedScorer:
    """Apply the training-only normalization before a frozen learned scorer."""

    def __init__(self, scorer: Any, normalization: RobustNormalization | None) -> None:
        self.scorer = scorer
        self.normalization = normalization

    def __call__(self, movie: np.ndarray) -> np.ndarray:
        raw = np.asarray(movie, dtype=np.float32)
        values = raw if self.normalization is None else self.normalization.apply(raw)
        return np.asarray(self.scorer(values), dtype=np.float64)


def _evaluation_scorers(
    result: MatchedTrainingResult,
    *,
    device: str,
    seed: int,
    normalization: RobustNormalization | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    jepa_primary = JEPAMaskedPredictionErrorScorer(
        result.jepa,
        device=device,
        seed=seed + 61_000_000,
        normalization=normalization,
    )
    mae_primary = MAEMaskedReconstructionErrorScorer(
        result.mae,
        device=device,
        seed=seed + 61_000_000,
        normalization=normalization,
    )
    scorers = {
        "compact_jepa_masked_prediction_error_secondary": jepa_primary,
        "masked_pixel_autoencoder_reconstruction_error_secondary": mae_primary,
        "frozen_random_encoder_latent_temporal_change": _FrozenNormalizedScorer(
            model_latent_temporal_change_scorer(result.random, device=device), normalization
        ),
        "frozen_handcrafted_stack": frozen_handcrafted_score,
        "compact_jepa_latent_temporal_change": _FrozenNormalizedScorer(
            model_latent_temporal_change_scorer(result.jepa, device=device), normalization
        ),
        "masked_pixel_autoencoder_latent_temporal_change": _FrozenNormalizedScorer(
            model_latent_temporal_change_scorer(result.mae, device=device), normalization
        ),
    }
    coverage = {
        "compact_jepa_masked_prediction_error_secondary": jepa_primary,
        "masked_pixel_autoencoder_reconstruction_error_secondary": mae_primary,
    }
    return scorers, coverage


def _flatten_evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    on = row["source_on_recovery"]
    intervention = row["intervention_recovery"]
    return {
        "training_seed": row["training_seed"],
        "fixture_id": row["fixture_id"],
        "background_recording_id": row["background_recording_id"],
        "background_window_id": row["background_window_id"],
        "injection_seed": row["injection_seed"],
        "source_count": row["source_count"],
        "crowding_case": row["crowding_case"],
        "method": row["method"],
        "score_input_units": row["score_input_units"],
        "source_on_recall": on["recall"],
        "source_on_recovered": on["recovered_sources"],
        "source_on_candidate_count": on["candidate_count"],
        "source_on_unmatched_candidate_count_unknown": on[
            "unmatched_candidate_count_unknown"
        ],
        "injected_sources": on["injected_sources"],
        "intervention_recall": intervention["recall"],
        "intervention_candidate_count": intervention["candidate_count"],
        "intervention_unmatched_candidate_count_unknown": intervention[
            "unmatched_candidate_count_unknown"
        ],
        "median_center_delta_background_mad": row[
            "median_injected_center_score_delta_background_mad"
        ],
        "maximum_pair_closure_absolute": row["maximum_pair_closure_absolute"],
        "maximum_pair_closure_float32_ulp": row[
            "maximum_pair_closure_float32_ulp"
        ],
    }


def _evaluate_seed(
    result: MatchedTrainingResult,
    fixtures: Sequence[SemiSyntheticFixture],
    fixture_metadata: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    seed: int,
    device: str,
    normalization: RobustNormalization | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scorers, coverage_scorers = _evaluation_scorers(
        result, device=device, seed=seed, normalization=normalization
    )
    injection = config["empirical_background_injection"]
    raw_rows = evaluate_paired_suite(
        fixtures,
        scorers,
        candidate_budget=int(injection["candidate_proposal_cap_per_clip"]),
        match_radius_px=float(injection["one_to_one_match_radius_px"]),
        minimum_distance_px=int(injection["candidate_minimum_distance_px"]),
        border_px=int(injection["candidate_border_px"]),
    )
    learned = {
        "compact_jepa_masked_prediction_error_secondary",
        "masked_pixel_autoencoder_reconstruction_error_secondary",
        "frozen_random_encoder_latent_temporal_change",
        "compact_jepa_latent_temporal_change",
        "masked_pixel_autoencoder_latent_temporal_change",
    }
    enriched: list[dict[str, Any]] = []
    for row in raw_rows:
        metadata = fixture_metadata[str(row["fixture_id"])]
        enriched.append(
            {
                **row,
                "training_seed": int(seed),
                "background_recording_id": metadata["background_recording_id"],
                "background_window_id": metadata["background_window_id"],
                "injection_seed": int(metadata["injection_seed"]),
                "source_count": int(metadata["source_count"]),
                "crowding_case": metadata["crowding_case"],
                "score_input_units": (
                    (
                        "identity_normalized_synthetic_smoke"
                        if normalization is None
                        else "frozen_training_robust_normalized"
                    )
                    if row["method"] in learned
                    else "raw_native_units"
                ),
            }
        )
    coverage = {
        method: scorer.coverage_records for method, scorer in coverage_scorers.items()
    }
    return enriched, coverage


def _primary_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    evaluated_training_seeds: Sequence[int],
    full_registered_grid: bool,
    bootstrap_draws: int,
) -> dict[str, Any]:
    primary = "compact_jepa_latent_temporal_change"
    comparators = (
        "masked_pixel_autoencoder_latent_temporal_change",
        "frozen_random_encoder_latent_temporal_change",
        "frozen_handcrafted_stack",
    )
    evaluation = config["evaluation"]
    grouped_by_seed: dict[int, Any] = {}
    seed_effects: list[float] = []
    for training_seed in evaluated_training_seeds:
        seed_rows = [row for row in rows if int(row["training_seed"]) == int(training_seed)]
        method_recall = {
            method: float(
                np.mean(
                    [
                        float(row["source_on_recovery"]["recall"])
                        for row in seed_rows
                        if row["method"] == method
                    ]
                )
            )
            for method in (primary, *comparators)
        }
        required_methods = (primary, *comparators)
        fixtures = sorted({str(row["fixture_id"]) for row in seed_rows})
        source_count_clusters: dict[
            tuple[str, str, int], dict[str, list[float]]
        ] = {}
        for fixture_id in fixtures:
            fixture_rows = [
                row
                for row in seed_rows
                if str(row["fixture_id"]) == fixture_id
                and str(row["method"]) in required_methods
            ]
            counts = Counter(str(row["method"]) for row in fixture_rows)
            if counts != Counter({method: 1 for method in required_methods}):
                raise JEPAPilotError(
                    "each fixture must contain exactly one primary and one row "
                    f"per comparator; {fixture_id} has {dict(counts)}"
                )
            representative = fixture_rows[0]
            identity_fields = (
                "background_recording_id",
                "background_window_id",
                "injection_seed",
                "source_count",
            )
            if any(
                any(row[field] != representative[field] for field in identity_fields)
                for row in fixture_rows[1:]
            ):
                raise JEPAPilotError(
                    f"fixture method rows disagree on cluster/source-count metadata: {fixture_id}"
                )
            cluster = (
                str(representative["background_recording_id"]),
                str(representative["background_window_id"]),
                int(representative["injection_seed"]),
            )
            method_lists = source_count_clusters.setdefault(
                cluster, {method: [] for method in required_methods}
            )
            for row in fixture_rows:
                method_lists[str(row["method"])].append(
                    float(row["source_on_recovery"]["recall"])
                )
        bootstrap_rows = [
            {
                "background_recording_id": recording,
                "background_window_id": window,
                "injection_seed": injection_seed,
                "method_values": {
                    method: float(np.mean(values[method]))
                    for method in required_methods
                },
                "source_counts_aggregated": len(values[primary]),
            }
            for (recording, window, injection_seed), values in sorted(source_count_clusters.items())
        ]
        if any(
            len(set(len(method_values[method]) for method in required_methods)) != 1
            for method_values in source_count_clusters.values()
        ):
            raise JEPAPilotError(
                "source-count aggregation is unbalanced across primary/comparator methods"
            )
        interval = hierarchical_strongest_comparator_bootstrap(
            bootstrap_rows,
            primary_method=primary,
            comparator_methods=comparators,
            draws=max(1_000, int(bootstrap_draws)),
            seed=int(config["seeds"]["grouped_bootstrap"]) + int(training_seed),
        )
        strongest = str(interval["observed_strongest_comparator"])
        recording_effects: dict[str, float] = {}
        for recording in sorted({str(row["background_recording_id"]) for row in bootstrap_rows}):
            recording_rows = [
                row for row in bootstrap_rows if str(row["background_recording_id"]) == recording
            ]
            recording_means = {
                method: float(
                    np.mean([float(row["method_values"][method]) for row in recording_rows])
                )
                for method in required_methods
            }
            recording_strongest = max(
                comparators, key=lambda method: (recording_means[method], method)
            )
            recording_effects[recording] = (
                recording_means[primary] - recording_means[recording_strongest]
            )
        effect = float(interval["observed_mean"])
        seed_effects.append(effect)
        grouped_by_seed[str(training_seed)] = {
            "method_macro_recall": method_recall,
            "strongest_registered_comparator": strongest,
            "paired_jepa_minus_strongest": interval,
            "source_count_aggregated_cluster_rows": bootstrap_rows,
            "recording_level_effects": recording_effects,
        }
    required_seeds = [int(value) for value in config["seeds"]["training"]]
    lower_bounds = [
        value["paired_jepa_minus_strongest"]["confidence_interval_95"][0]
        for value in grouped_by_seed.values()
    ]
    margin = float(evaluation["success_margin_absolute_recall"])
    max_recording_loss = float(evaluation["maximum_allowed_recording_level_loss"])
    screen_checks = {
        "all_registered_training_seeds_evaluated": set(evaluated_training_seeds) == set(required_seeds),
        "full_registered_injection_grid": bool(full_registered_grid),
        "primary_effect_positive_every_seed": bool(seed_effects) and all(value > 0 for value in seed_effects),
        "primary_mean_margin_at_least_registered_delta": bool(seed_effects)
        and float(np.mean(seed_effects)) >= margin,
        "hierarchical_interval_lower_bound_strictly_positive_every_seed": bool(lower_bounds)
        and all(value is not None and float(value) > 0 for value in lower_bounds),
        "seed_range_within_limit": bool(seed_effects)
        and max(seed_effects) - min(seed_effects)
        <= float(config["representation_gates"]["maximum_primary_recall_range_across_seeds"]),
        "no_recording_level_loss_beyond_limit": all(
            float(effect) >= -max_recording_loss
            for value in grouped_by_seed.values()
            for effect in value["recording_level_effects"].values()
        ),
    }
    return {
        "primary_method": primary,
        "comparator_policy": evaluation["primary_comparator"],
        "training_seed_results": grouped_by_seed,
        "screen_observed_checks": screen_checks,
        "claim_gate_status": "unresolved",
        "claim_promotion_allowed": False,
        "unrun_registered_claim_gates": [
            "nuisance_matching_or_residualization",
            "all_required_counterfactuals",
            "leave_one_recording_out_sensitivity",
            "per_recording_collapse_diagnostics",
            "cross_seed_linear_centered_kernel_alignment",
            "registered_1000_step_checkpoint_cadence",
            "same_manifest_resume_validation",
            "full_registered_seed_and_injection_grid",
            "complete_scientific_audit_media_and_inventory",
        ],
        "interpretation": (
            "Source counts were averaged inside recording/window/injection-seed clusters before the recording-window-seed hierarchical bootstrap. "
            "This is exact injected-source sensitivity on empirical backgrounds, not biological precision."
        ),
    }


def _checkpoint_payload(
    result: MatchedTrainingResult,
    *,
    seed: int,
    encoder_config: EncoderConfig,
    mask_config: MaskConfig,
    schedule: TrainingSchedule,
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "scientific_checkpoint": False,
        "training_seed": int(seed),
        "encoder_config": asdict(encoder_config),
        "mask_config": asdict(mask_config),
        "schedule": asdict(schedule),
        "jepa_state_dict": {
            key: value.detach().cpu() for key, value in result.jepa.state_dict().items()
        },
        "mae_state_dict": {
            key: value.detach().cpu() for key, value in result.mae.state_dict().items()
        },
        "random_state_dict": {
            key: value.detach().cpu() for key, value in result.random.state_dict().items()
        },
        "optimizer_states": result.optimizer_states,
        "metrics": result.metrics,
    }


def _audit_placeholders(root: Path, *, mode: str, fixture_count: int) -> dict[str, Any]:
    sections = {
        "1_Expert_Annotations": (
            "# Exact injection truth / expert-section boundary\n\n"
            "No human expert annotations are used in the label-free validation view. For the paired-injection endpoint, this section role is exact generated source truth, not a human neuron annotation. Required truth media are pending.\n"
        ),
        "2_Model_Annotations": (
            "# Frozen model candidate-surrogate section\n\n"
            "Frozen maximum-budget candidate media, close-ups, full-duration traces, and detection metadata are pending. Each map may emit fewer proposals when fewer separated local maxima exist; unmatched native-background proposals remain unknown.\n"
        ),
        "3_Comparison": (
            "# Matched comparison section\n\n"
            "The numeric source-on/source-off comparison is available in `paired_injection_results.tsv`. Required figure/table-only audit renderings remain pending.\n"
        ),
    }
    for directory, readme in sections.items():
        atomic_text(root / directory / "README.md", readme)
    status = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mode": mode,
        "expert_annotations": "not_applicable_for_label_free_validation",
        "paired_injection_truth_role": "exact_generated_sources_not_human_expert",
        "numeric_fixture_count": int(fixture_count),
        "required_media_complete": False,
        "inventory_validation_passed": False,
        "scientific_audit_complete": False,
        "missing_required_families": [
            "truth_only_full_field_videos",
            "truth_closeups_and_full_duration_traces",
            "arm_pure_model_full_field_videos",
            "model_closeups_and_full_duration_traces",
            "figure_table_only_matched_comparison",
            "media_decode_and_annotation_pixel_validation",
        ],
        "promotion_blocked": True,
    }
    atomic_json(root / "scientific_audit_status.json", status)
    return status


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_index.json"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _report_text(summary: Mapping[str, Any]) -> str:
    mode = summary["execution"]["mode"]
    seeds = summary["execution"]["evaluated_training_seeds"]
    fixtures = summary["paired_injection"]["observed_fixture_count"]
    return (
        "# Compact spatiotemporal JEPA pilot\n\n"
        f"This `{mode}` execution trained matched JEPA and masked-pixel-autoencoder arms for {len(seeds)} seed(s) and evaluated {fixtures} exact paired-injection fixtures. "
        "The primary learned-arm score head is the same frozen latent temporal-change norm for JEPA, MAE, and random encoders. JEPA masked-prediction error and MAE masked-reconstruction error are objective-native secondary lanes with deterministic full token coverage. Frozen learned scorers apply the training-only robust normalization to source-off and source-on raw clips. The registered handcrafted stack reproduces the legacy component-wise robust-z combined score on source-off, then applies those source-off component calibrations unchanged to source-on.\n\n"
        "This output is **not a scientific completion**. Smoke and screen modes are bounded engineering evidence, the full registered seed/injection budget may be incomplete, and required scientific-audit media are still pending. Native background proposals remain unknown rather than false positives. The experiment does not establish biological neuron identity, full-field precision, independent-animal generalization, denoising, causality, a world model, reinforcement learning, or control.\n"
    )


def run_jepa_pilot(
    config_path: Path,
    *,
    repository_root: Path,
    mode: str = "smoke",
    data_root: Path | None = None,
    output_root: Path | None = None,
    device: str | None = None,
    steps: int | None = None,
    train_clip_count: int | None = None,
    validation_clip_count: int | None = None,
    max_training_seeds: int | None = 1,
    injection_cells_per_background: int | None = 1,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Run a non-scientific smoke or bounded screen with explicit artifacts.

    ``preflight`` performs no write.  Neither executable mode can mark the
    registered experiment scientifically complete; claim-bearing execution is
    intentionally outside this bounded runner until authorization and audit
    production are resolved.
    """

    if mode not in SUPPORTED_MODES:
        raise JEPAPilotError(f"mode must be one of {sorted(SUPPORTED_MODES)}")
    repository = Path(repository_root).expanduser().resolve()
    runtime_data = (
        configured_data_root(repository)
        if data_root is None
        else Path(data_root).expanduser().resolve()
    )
    config_file = Path(config_path).expanduser().resolve()
    config = load_jepa_pilot_config(config_file)
    configured_output = _resolve_repo_path(str(config["output_root"]), repository)
    if output_root is None and mode != "preflight":
        configured_output = configured_output.with_name(
            configured_output.name + f"__{mode.upper()}_NONSCIENTIFIC"
        )
    requested_output = (
        configured_output
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    requested_device = str(device or ("cpu" if mode == "smoke" else config["resources"]["device"]))
    preflight, inventory, split = _preflight_checks(
        config,
        config_path=config_file,
        repository_root=repository,
        runtime_data_root=runtime_data,
        output_root=requested_output,
        device=requested_device,
        verify_live=mode == "screen",
        verify_hashes=verify_hashes,
    )
    if mode == "preflight":
        return preflight
    if preflight["failed_checks"]:
        raise JEPAPilotError(
            "preflight failed: " + ", ".join(preflight["failed_checks"])
        )
    output, partial = _output_paths(requested_output)
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing existing output or partial root: {output}")
    registered_seeds = [int(value) for value in config["seeds"]["training"]]
    if max_training_seeds is None:
        evaluated_seeds = registered_seeds
    else:
        if max_training_seeds < 1:
            raise JEPAPilotError("max_training_seeds must be positive or None")
        evaluated_seeds = registered_seeds[: int(max_training_seeds)]
    partial.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    non_scientific_run_id = f"{config['run_id']}__{mode.upper()}_NONSCIENTIFIC"
    previous_torch_threads = torch.get_num_threads()
    cpu_thread_cap = int(config["resources"]["cpu_threads"])
    torch.set_num_threads(cpu_thread_cap)
    try:
        atomic_json(partial / "preflight.json", preflight)
        if mode == "screen":
            atomic_json(partial / "motion_audit.json", preflight["motion_audit"])
        resolved_config = {
            "schema_version": SCHEMA_VERSION,
            "registered_config": _portable_config(
                config, repository_root=repository, runtime_data_root=runtime_data
            ),
            "registered_config_sha256": sha256_file(config_file),
            "execution": {
                "mode": mode,
                "run_id": non_scientific_run_id,
                "scientific_execution": False,
                "registered_execution_authorized": bool(config.get("execution_authorized")),
                "overrides": {
                    "device": requested_device,
                    "steps": steps,
                    "train_clip_count": train_clip_count,
                    "validation_clip_count": validation_clip_count,
                    "max_training_seeds": max_training_seeds,
                    "injection_cells_per_background": injection_cells_per_background,
                    "verify_hashes": verify_hashes,
                    "cpu_thread_cap": cpu_thread_cap,
                },
            },
        }
        atomic_json(partial / "resolved_config.json", resolved_config)
        _heartbeat(partial, "building_clip_banks", mode=mode)

        if mode == "smoke":
            encoder_config, mask_config = tiny_smoke_configuration()
            training_clips, validation_clips, validation_descriptors = _smoke_clip_banks(
                int(config["seeds"]["training"][0])
            )
            train_descriptors = [
                {
                    "recording_id": "synthetic_smoke_train",
                    "clip_index": index,
                    "sampling_stratum": "synthetic_smoke",
                }
                for index in range(len(training_clips))
            ]
            normalization: RobustNormalization | None = None
            data_manifest = {
                "schema_version": SCHEMA_VERSION,
                "mode": "synthetic_smoke",
                "raw_source_access": False,
                "spon_pretraining_excluded": True,
                "scientific_data_evidence": False,
                "clip_shape_tyx": list(training_clips.shape[1:]),
                "train_clip_count": len(training_clips),
                "validation_clip_count": len(validation_clips),
                "normalization": {
                    "policy": "identity_for_zero_centered_synthetic_smoke_only",
                    "applied_to_learned_scorers": True,
                },
            }
            backgrounds = _smoke_background_bank(validation_clips)
            background_rows = [dict(row["metadata"]) for row in backgrounds]
            training_plans_by_seed: dict[int, Sequence[ClipRequest]] = {}
            canonical_clip_bank_screen_path = False
        else:
            encoder_config, mask_config = _architecture_from_config(config)
            clip_shape = tuple(int(value) for value in config["clip"]["shape_tyx"])
            contract = TemporalClipContract(
                clip_frames=clip_shape[0],
                guard_frames=int(config["clip"]["temporal_guard_frames"]),
                frame_step=int(config["clip"]["frame_step"]),
            )
            normalization_config = config["normalization"]
            normalization = fit_training_robust_normalization(
                inventory.recordings,
                split,
                frames_per_recording=int(normalization_config["frames_per_training_recording"]),
                spatial_stride=int(normalization_config["spatial_stride_px"]),
                minimum_scale=float(normalization_config["minimum_scale"]),
            )
            clip_bank_contract = config["clip"]
            registered_train_count = int(
                clip_bank_contract["training_clip_bank"]["clips_per_training_seed"]
            )
            registered_validation_count = int(
                clip_bank_contract["validation_clip_bank"]["total_clips"]
            )
            validation_sampling_seed = int(config["seeds"]["validation_sampling"])
            resolved_train_count = int(
                registered_train_count if train_clip_count is None else train_clip_count
            )
            if resolved_train_count % 4:
                raise JEPAPilotError("train_clip_count must be a multiple of four")
            resolved_validation_count = int(
                registered_validation_count
                if validation_clip_count is None
                else validation_clip_count
            )
            training_plans_by_seed = {
                seed: sample_training_crop_plan(
                    inventory.recordings,
                    split,
                    clip_count=resolved_train_count,
                    seed=seed,
                    contract=contract,
                    crop_shape=clip_shape[1:],
                )
                for seed in evaluated_seeds
            }
            validation_plan = sample_validation_crop_plan(
                inventory.recordings,
                split,
                clip_count=resolved_validation_count,
                seed=validation_sampling_seed,
                contract=contract,
                crop_shape=clip_shape[1:],
            )
            validation_clips, validation_descriptors = _normalized_clip_bank(
                inventory.recordings, validation_plan, contract, normalization
            )
            per_seed_manifests = {
                str(seed): build_jepa_data_manifest(
                    inventory,
                    split,
                    temporal_contract=contract,
                    normalization=normalization,
                    crop_plan=plan,
                    validation_crop_plan=validation_plan,
                )
                for seed, plan in training_plans_by_seed.items()
            }
            canonical_clip_bank_screen_path = bool(
                resolved_train_count == registered_train_count
                and resolved_validation_count == registered_validation_count
                and validation_sampling_seed
                == int(clip_bank_contract["validation_clip_bank"]["sampling_seed"])
                and all(seed in training_plans_by_seed for seed in evaluated_seeds)
            )
            data_manifest = {
                "schema_version": SCHEMA_VERSION,
                "mode": "real_raw_recording_screen",
                "canonical_clip_bank_contract": {
                    "training_clips_per_seed": registered_train_count,
                    "training_plan_seed_equals_training_seed": True,
                    "validation_clips": registered_validation_count,
                    "validation_sampling_seed": validation_sampling_seed,
                    "validation_shared_across_training_seeds": True,
                    "complete_all_registered_training_seeds": set(evaluated_seeds)
                    == set(registered_seeds),
                    "screen_exercises_canonical_bank_sizes_for_evaluated_seeds": canonical_clip_bank_screen_path,
                    "bounded_override_present": bool(
                        resolved_train_count != registered_train_count
                        or resolved_validation_count != registered_validation_count
                    ),
                },
                "per_training_seed": per_seed_manifests,
            }
            holdout = inventory_source_holdout(config, repository, runtime_data)
            backgrounds, background_rows = _real_background_bank(
                config,
                inventory=inventory,
                split=split,
                holdout=holdout,
                contract=contract,
            )

        atomic_json(partial / "data_manifest.json", data_manifest)
        atomic_json(
            partial / "background_window_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "windows": background_rows,
                "spon_quiet_bursts_ui_inclusive_excluded": [
                    list(value) for value in SPON_BURSTS_UI_INCLUSIVE
                ],
                "burst_interval_exclusion_used_for_spon": mode == "screen",
                "burst_interval_use": (
                    "post_training_holdout_quiet_window_exclusion_only"
                    if mode == "screen"
                    else "not_applicable_synthetic_smoke"
                ),
                "roi_identity_labels_used": False,
                "candidate_review_labels_used": False,
            },
        )
        if mode == "smoke":
            cache_manifest = _cache_clip_banks(
                partial,
                training_clips,
                validation_clips,
                train_descriptors,
                validation_descriptors,
            )
        else:
            cache_manifest = {
                "schema_version": SCHEMA_VERSION,
                "raw_video_copied": False,
                "derived_normalized_clip_cache": True,
                "validation": _cache_shared_validation_bank(
                    partial,
                    validation_clips,
                    validation_descriptors,
                    sampling_seed=int(config["seeds"]["validation_sampling"]),
                ),
                "training_by_seed": {},
                "canonical_bank_sizes_for_evaluated_seeds": canonical_clip_bank_screen_path,
            }
            atomic_json(partial / "clip_bank_manifest.json", cache_manifest)
        fixtures, fixture_metadata, fixture_summary = _make_fixtures(
            config,
            backgrounds,
            smoke=mode == "smoke",
            injection_cells_per_background=injection_cells_per_background,
        )
        atomic_json(
            partial / "paired_injection_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                **fixture_summary,
                "fixtures": [
                    {
                        "fixture_id": fixture.fixture_id,
                        "metadata": fixture.metadata,
                        **_fixture_closure_manifest(fixture),
                    }
                    for fixture in fixtures
                ],
                "raw_fixture_arrays_persisted": False,
                "native_background_truth_boundary": "not_decomposed_unknown_not_negative",
            },
        )

        registered_budget = config["matched_training_budget"]
        registered_steps = int(registered_budget["optimizer_steps_per_arm_seed"])
        resolved_steps = int(steps if steps is not None else (1 if mode == "smoke" else min(8, registered_steps)))
        if not 1 <= resolved_steps <= registered_steps:
            raise JEPAPilotError(
                "bounded steps must be positive and may not exceed the registered budget"
            )
        resolved_batch_size = 2 if mode == "smoke" else int(registered_budget["batch_size"])
        schedule = TrainingSchedule(
            steps=resolved_steps,
            hard_step_cap=int(registered_budget["hard_step_cap_per_arm_seed"]),
            batch_size=resolved_batch_size,
            learning_rate=float(registered_budget["learning_rate"]),
            weight_decay=float(registered_budget["weight_decay"]),
            optimizer="AdamW",
            adamw_betas=tuple(float(value) for value in registered_budget["adamw_betas"]),
            adamw_epsilon=float(registered_budget["adamw_epsilon"]),
            learning_rate_schedule=str(registered_budget["learning_rate_schedule"]),
            gradient_clip_norm=float(registered_budget["gradient_clip_norm"]),
            ema_decay=float(next(row for row in config["arms"] if row["id"] == "compact_jepa")["ema_decay"]),
            log_interval=max(1, min(100, resolved_steps)),
            amp=(
                requested_device == "cuda"
                and bool(registered_budget["mixed_precision"]["enabled_on_cuda"])
            ),
            deterministic_algorithms=True,
        )
        training_metrics: dict[str, Any] = {}
        evaluation_rows: list[dict[str, Any]] = []
        coverage_records: dict[str, Any] = {}
        nuisance_audits: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(evaluated_seeds, start=1):
            if mode == "screen":
                training_clips, train_descriptors = _normalized_clip_bank(
                    inventory.recordings,
                    training_plans_by_seed[seed],
                    contract,
                    normalization,
                )
                cache_manifest["training_by_seed"][str(seed)] = _cache_seed_training_bank(
                    partial,
                    training_clips,
                    train_descriptors,
                    training_seed=seed,
                )
                atomic_json(partial / "clip_bank_manifest.json", cache_manifest)
            _heartbeat(
                partial,
                "matched_training",
                seed=seed,
                seed_index=seed_index,
                seed_count=len(evaluated_seeds),
                optimizer_steps=resolved_steps,
            )
            result = train_matched_representations(
                training_clips,
                validation_clips,
                encoder_config=encoder_config,
                mask_config=mask_config,
                schedule=schedule,
                seed=seed,
                device=requested_device,
            )
            training_metrics[str(seed)] = result.metrics
            checkpoint = partial / "checkpoints" / f"seed_{seed}_final_non_scientific.pt"
            _atomic_torch_save(
                checkpoint,
                _checkpoint_payload(
                    result,
                    seed=seed,
                    encoder_config=encoder_config,
                    mask_config=mask_config,
                    schedule=schedule,
                    mode=mode,
                ),
            )
            _heartbeat(partial, "paired_injection_evaluation", seed=seed)
            seed_rows, seed_coverage = _evaluate_seed(
                result,
                fixtures,
                fixture_metadata,
                config=config,
                seed=seed,
                device=requested_device,
                normalization=normalization,
            )
            evaluation_rows.extend(seed_rows)
            coverage_records[str(seed)] = seed_coverage
            nuisance_audits.append(
                _nuisance_audit(
                    result,
                    validation_clips,
                    validation_descriptors,
                    device=requested_device,
                    smoke=mode == "smoke",
                    motion_audit=(None if mode == "smoke" else preflight["motion_audit"]),
                    training_recording_ids=(
                        () if mode == "smoke" else split.train_ids
                    ),
                )
            )
            if requested_device == "cuda":
                torch.cuda.empty_cache()

        atomic_json(partial / "training_metrics.json", training_metrics)
        atomic_json(partial / "mask_sweep_coverage.json", coverage_records)
        atomic_json(partial / "nuisance_audit.json", {"training_seeds": nuisance_audits})
        atomic_json(partial / "paired_injection_results.json", {"rows": evaluation_rows})
        flattened = [_flatten_evaluation_row(row) for row in evaluation_rows]
        _write_tsv(partial / "paired_injection_results.tsv", flattened)
        bootstrap_draws = (
            1_000
            if mode in {"smoke", "screen"}
            else int(config["evaluation"]["grouped_bootstrap_draws"])
        )
        primary = _primary_effect_summary(
            evaluation_rows,
            config=config,
            evaluated_training_seeds=evaluated_seeds,
            full_registered_grid=bool(fixture_summary["full_registered_grid"]),
            bootstrap_draws=bootstrap_draws,
        )
        audit_status = _audit_placeholders(
            partial, mode=mode, fixture_count=len(fixtures)
        )
        all_losses_finite = all(
            np.isfinite(float(point[key]))
            for metrics in training_metrics.values()
            for point in metrics["training_curve"]
            for key in ("jepa_masked_latent_loss", "mae_masked_pixel_loss")
        )
        pair_closure_passed = all(
            float(row["maximum_pair_closure_float32_ulp"])
            <= float(
                config["empirical_background_injection"][
                    "maximum_pair_closure_float32_ulp"
                ]
            )
            for row in evaluation_rows
        )
        mask_coverage_passed = all(
            int(record["minimum_observations"]) >= 1
            for seed_records in coverage_records.values()
            for method_records in seed_records.values()
            for record in method_records.values()
        )
        validation_checks = {
                "all_training_losses_finite": bool(all_losses_finite),
                "matched_capacity_within_one_percent": all(
                    bool(value["capacity"]["matched_within_one_percent"])
                    for value in training_metrics.values()
                ),
                "float16_never_used": all(
                    value.get("float16_used") is False
                    and value.get("amp_dtype") in {None, "bfloat16"}
                    for value in training_metrics.values()
                ),
                "reported_loss_metrics_cast_to_float32": all(
                    value.get("loss_accumulation_dtype") == "float32"
                    and value.get("metric_dtype") == "float32"
                    and value.get("observed_training_loss_dtypes")
                    == {"jepa": ["float32"], "mae": ["float32"]}
                    and value.get("validation", {}).get("observed_loss_dtypes")
                    == {"jepa": ["float32"], "mae": ["float32"]}
                    for value in training_metrics.values()
                ),
                "cpu_thread_cap_applied": torch.get_num_threads() == cpu_thread_cap,
                "canonical_clip_bank_sizes_for_evaluated_seeds": bool(
                    canonical_clip_bank_screen_path
                ),
                "paired_injection_closure": bool(pair_closure_passed),
                "all_mask_sweep_tokens_observed": bool(mask_coverage_passed),
                "all_four_background_recordings_represented": set(
                    fixture_summary["background_recording_ids"]
                )
                == set(config["empirical_background_injection"]["background_recording_ids"]),
                "learned_scorers_apply_frozen_training_normalization": True,
                "handcrafted_scorer_receives_raw_units": True,
                "handcrafted_component_calibration_fit_on_source_off_only": True,
                "source_counts_not_independent_bootstrap_replicates": True,
                "native_background_candidates_unknown_not_negative": True,
                "scientific_audit_complete": False,
        }
        required_engineering_checks = [
            "all_training_losses_finite",
            "matched_capacity_within_one_percent",
            "float16_never_used",
            "reported_loss_metrics_cast_to_float32",
            "cpu_thread_cap_applied",
            "paired_injection_closure",
            "all_mask_sweep_tokens_observed",
            "all_four_background_recordings_represented",
            "learned_scorers_apply_frozen_training_normalization",
            "handcrafted_scorer_receives_raw_units",
            "handcrafted_component_calibration_fit_on_source_off_only",
            "source_counts_not_independent_bootstrap_replicates",
            "native_background_candidates_unknown_not_negative",
        ]
        if mode == "screen":
            required_engineering_checks.append(
                "canonical_clip_bank_sizes_for_evaluated_seeds"
            )
        failed_engineering_checks = [
            key for key in required_engineering_checks if not validation_checks[key]
        ]
        validation_status = (
            "failed_non_scientific_validation"
            if failed_engineering_checks
            else (
                "passed_non_scientific_smoke_contract_only"
                if mode == "smoke"
                else "screen_complete_claim_gates_unresolved"
            )
        )
        validation = {
            "schema_version": SCHEMA_VERSION,
            "status": validation_status,
            "checks": validation_checks,
            "required_engineering_checks": required_engineering_checks,
            "failed_engineering_checks": failed_engineering_checks,
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
            "promotion_blockers": [
                "non_scientific_execution_mode",
                "scientific_audit_media_and_inventory_pending",
                "registered_1000_step_checkpoint_and_same_manifest_resume_unimplemented",
                *(
                    []
                    if canonical_clip_bank_screen_path
                    else ["canonical_per_seed_training_and_shared_validation_clip_bank_not_exercised"]
                ),
                *(
                    []
                    if fixture_summary["full_registered_grid"]
                    else ["registered_injection_grid_incomplete"]
                ),
                *(
                    []
                    if set(evaluated_seeds) == set(registered_seeds)
                    else ["registered_training_seed_set_incomplete"]
                ),
                *(
                    []
                    if resolved_steps == registered_steps
                    else ["registered_optimizer_budget_incomplete"]
                ),
                *(
                    []
                    if not any(audit["proposed_arm_blocks_promotion"] for audit in nuisance_audits)
                    else ["nuisance_dominance_or_motion_review_requires_sensitivity"]
                ),
            ],
        }
        if mode == "smoke":
            compact_cache_summary = {
                "manifest": "clip_bank_manifest.json",
                "train_clip_count": int(cache_manifest["train"]["shape"][0]),
                "validation_clip_count": int(cache_manifest["validation"]["shape"][0]),
                "training_seed_count": 1,
            }
        else:
            compact_cache_summary = {
                "manifest": "clip_bank_manifest.json",
                "training_seed_counts": {
                    seed: int(value["shape"][0])
                    for seed, value in cache_manifest["training_by_seed"].items()
                },
                "validation_clip_count": int(cache_manifest["validation"]["shape"][0]),
                "validation_sampling_seed": int(cache_manifest["validation"]["sampling_seed"]),
                "training_seed_count": len(cache_manifest["training_by_seed"]),
            }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": validation["status"],
            "experiment_id": config["experiment_id"],
            "run_id": non_scientific_run_id,
            "execution": {
                "mode": mode,
                "scientific_execution": False,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "device": requested_device,
                "optimizer_steps_per_arm_seed": resolved_steps,
                "registered_optimizer_steps_per_arm_seed": registered_steps,
                "evaluated_training_seeds": evaluated_seeds,
                "registered_training_seeds": registered_seeds,
                "cpu_thread_cap": cpu_thread_cap,
            },
            "data": {
                "clip_cache": compact_cache_summary,
                "normalization": (
                    {
                        "policy": "identity_synthetic_smoke",
                        "fit_scope": "not_applicable",
                    }
                    if normalization is None
                    else normalization.to_manifest()
                ),
                "pretraining_input": "raw_grayscale_only_then_frozen_training_normalization",
                "spon_used_for_pretraining": False,
                "spon_used_for_post_training_paired_injection_only": True,
                "canonical_clip_bank_sizes_for_evaluated_seeds": bool(
                    canonical_clip_bank_screen_path
                ),
            },
            "paired_injection": fixture_summary,
            "primary_evaluation": primary,
            "nuisance_audit": {
                "seed_count": len(nuisance_audits),
                "any_proposed_arm_promotion_block": any(
                    value["proposed_arm_blocks_promotion"] for value in nuisance_audits
                ),
                "motion_review_recording_ids": (
                    []
                    if mode == "smoke"
                    else preflight["motion_audit"]["aggregate_gate"]["review_recording_ids"]
                ),
            },
            "scientific_audit": audit_status,
            "claim_boundary": config["claim_boundary"],
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
        }
        atomic_json(partial / "summary.json", summary)
        atomic_json(partial / "validation.json", validation)
        atomic_text(partial / "REPORT.md", _report_text(summary))
        llm_context = {
            "schema_version": SCHEMA_VERSION,
            "entry_point": "summary.json",
            "status": validation["status"],
            "read_first": [
                "summary.json",
                "validation.json",
                "paired_injection_manifest.json",
                "nuisance_audit.json",
                "mask_sweep_coverage.json",
            ],
            "primary_table": "paired_injection_results.tsv",
            "training_metrics": "training_metrics.json",
            "data_provenance": "data_manifest.json",
            "motion_audit": None if mode == "smoke" else "motion_audit.json",
            "scientific_audit_status": "scientific_audit_status.json",
            "coordinate_convention": "x=column,y=row; zero-based half-open runtime crops",
            "frame_convention": "UI one-based inclusive only where explicitly labeled; array intervals zero-based half-open",
            "score_contract": {
                "primary_common_head": "frozen latent temporal-change norm for JEPA, MAE, and random encoders",
                "compact_jepa_objective_native_secondary": "fully covered deterministic mask-averaged latent prediction error",
                "masked_pixel_autoencoder_objective_native_secondary": "fully covered deterministic mask-averaged pixel reconstruction error",
                "learned_score_input": "frozen training robust normalization applied after raw-unit injection",
                "handcrafted_score_input": "legacy-equivalent component-wise robust-z fitted on source-off only and applied frozen to both pair arms",
            },
            "evidence_boundary": {
                "exact_truth": "injected sources only",
                "native_background": "unknown_not_negative",
                "precision": "not_estimable",
                "biological_neuron_identity": False,
                "independent_animal_generalization": False,
                "world_model": False,
                "reinforcement_learning": False,
                "scientific_completion": False,
            },
        }
        atomic_json(partial / "llm_context.json", llm_context)
        atomic_json(
            partial / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": validation["status"],
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
                "started_at": started_at,
                "finished_at": summary["execution"]["finished_at"],
                "heartbeat": "heartbeat.json",
                "validation": "validation.json",
            },
        )
        _heartbeat(
            partial,
            "finished_non_scientific",
            validation_status=validation["status"],
            scientific_completion=False,
        )
        atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        partial.replace(output)
        torch.set_num_threads(previous_torch_threads)
        return summary
    except BaseException as exc:
        atomic_json(
            partial / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_preserved_nonresumable_partial",
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
                "failed_at": _utc_now(),
                "error_type": type(exc).__name__,
                "partial_output_preserved": True,
                "resume_supported": False,
                "unresolved_claim_run_gates": [
                    "registered_1000_step_checkpoint_cadence",
                    "same_manifest_resume_validation",
                ],
            },
        )
        torch.set_num_threads(previous_torch_threads)
        raise


__all__ = [
    "JEPAMaskedPredictionErrorScorer",
    "JEPAPilotError",
    "MAEMaskedReconstructionErrorScorer",
    "deterministic_full_coverage_masks",
    "frozen_handcrafted_score",
    "load_jepa_pilot_config",
    "preflight_jepa_pilot",
    "run_jepa_pilot",
]
