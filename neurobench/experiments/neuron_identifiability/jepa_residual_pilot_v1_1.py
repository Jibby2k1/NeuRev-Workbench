"""Bounded conditional-JEPA pixel-residual screen v1.1 for ``NREV-EXP-0029``.

Version 1.1 is the fail-closed Run-B amendment to the original runner.  Its
only scientific change is to evaluate the raw handcrafted-comparator arm in
the same native raw units used by the immutable EXP-0028 Screen-B parent.
JEPA and random-encoder residual arms remain in frozen-training-normalized
pixel units.  This restores a byte-exact cross-run raw-HC anchor without
changing the decoder, residual definition, fixtures, budgets, or guards.

This runner is deliberately downstream of the immutable EXP-0028 Screen-B
artifacts.  It verifies every indexed parent artifact before loading the
checkpoint, reuses the already-normalized clip caches without applying a
second normalization, and reconstructs the exact paired-injection fixture
grid from the frozen raw-window manifest.  All executable modes remain
non-scientific engineering screens; no result produced here can promote a
claim or complete the scientific audit.

The learned operation is pixel-space conditional background prediction.  The
signed residual is ``normalized_video - predicted_background``.  A residual
is not assumed to be a neuron probability: exact injected-signal retention,
predictor absorption, background suppression, seams, and localization are
reported separately.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from neurobench.portable_paths import data_root as configured_data_root
from neurobench.portable_paths import portable_path, portableize_paths

from .contracts import atomic_json, atomic_text, stable_hash
from .discovery import sha256_file
from .jepa_comparators import frozen_handcrafted_comparator
from .jepa_data import (
    ClipRequest,
    RobustNormalization,
    TemporalClipContract,
    load_recording_inventory,
    read_sequential_clip,
)
from .jepa_evaluation import evaluate_paired_fixture, hierarchical_strongest_comparator_bootstrap
from .jepa_pilot import (
    _fixture_closure_manifest,
    _make_fixtures,
    inventory_source_holdout,
)


SCHEMA_VERSION = 2
RUNNER_VERSION = "1.1"
EXPERIMENT_ID = "NREV-EXP-0029"
STRICT_RUN_ID = "NREV-RUN-EXP-0029-SCREEN-20260830-B"
STRICT_OUTPUT_ROOT = (
    "Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/"
    "NREV-RUN-EXP-0029-SCREEN-20260830-B"
)
STRICT_PREDICTION_SHAPE_BCTHW = (1, 1, 32, 64, 64)
SUPPORTED_MODES = {"preflight", "smoke", "screen"}
RAW_METHOD = "raw_frozen_handcrafted_stack"
PARENT_RAW_METHOD = "frozen_handcrafted_stack"
JEPA_RESIDUAL_METHOD = (
    "jepa_conditional_pixel_residual_frozen_handcrafted_stack"
)
RANDOM_RESIDUAL_METHOD = (
    "random_encoder_conditional_pixel_residual_frozen_handcrafted_stack"
)
STRICT_PARENT_SEED = 1001
STRICT_PARENT_STEPS = 500
STRICT_SCREEN_TRAIN_CLIPS = 512
STRICT_SCREEN_VALIDATION_CLIPS = 96
STRICT_SCREEN_BACKGROUNDS = 12
STRICT_SCREEN_FIXTURES = 108
STRICT_SCREEN_SOURCES = 252


class JEPAResidualPilotError(ValueError):
    """Raised when the residual screen cannot preserve its frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JEPAResidualPilotError(f"could not read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise JEPAResidualPilotError(f"JSON root must be an object: {path}")
    return payload


def _require(mapping: Mapping[str, Any], key: str, expected: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise JEPAResidualPilotError(
            f"configuration field {key!r} must be {expected.__name__}"
        )
    return value


def _resolve_repo_path(value: str, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _relative_child(root: Path, value: str, *, field: str) -> Path:
    if Path(value).is_absolute():
        raise JEPAResidualPilotError(f"{field} must be relative to its declared root")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise JEPAResidualPilotError(f"{field} escapes its declared root") from exc
    return candidate


def _valid_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise JEPAResidualPilotError(f"{field} must be a lowercase SHA-256 digest")
    return text


def load_jepa_residual_config(path: Path) -> dict[str, Any]:
    """Load the maintained EXP-0029 descriptor and reject implicit budgets."""

    config_path = Path(path).expanduser().resolve()
    config = _json_object(config_path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise JEPAResidualPilotError("unsupported residual-pilot schema_version")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise JEPAResidualPilotError(f"config must identify {EXPERIMENT_ID}")
    if config.get("runner_version") != RUNNER_VERSION:
        raise JEPAResidualPilotError(
            f"v1.1 runner requires runner_version={RUNNER_VERSION!r}"
        )
    if config.get("run_id") != STRICT_RUN_ID:
        raise JEPAResidualPilotError(
            f"v1.1 runner requires the Run-B amendment ID {STRICT_RUN_ID}"
        )
    if config.get("output_root") != STRICT_OUTPUT_ROOT:
        raise JEPAResidualPilotError(
            "v1.1 runner requires the canonical Run-B output_root "
            f"{STRICT_OUTPUT_ROOT}"
        )
    for key in (
        "upstream_screen",
        "frozen_data_contract",
        "pixel_decoder",
        "blind_tube_geometry",
        "frozen_jepa",
        "paired_evaluation",
        "dependency_readiness",
        "descriptive_screen_advancement_thresholds",
        "training_plan",
        "resource_guards",
        "scientific_audit",
        "seeds",
        "implementation_hash_contract",
        "execution_modes",
    ):
        _require(config, key, dict)
    for key in ("run_id", "output_root", "data_inventory"):
        _require(config, key, str)
    for key in (
        "engineering_screen_execution_authorized",
        "execution_authorized",
        "scientific_execution_authorized",
        "claim_bearing_execution_authorized",
        "scientific_completion_allowed",
        "claim_bearing_execution",
    ):
        _require(config, key, bool)
    modes = config["execution_modes"]
    if set(modes) != SUPPORTED_MODES:
        raise JEPAResidualPilotError(
            "execution_modes must explicitly define preflight, smoke, and screen"
        )
    for mode in SUPPORTED_MODES:
        _require(modes, mode, dict)
    screen = modes["screen"]
    strict = {
        "train_clip_count": STRICT_SCREEN_TRAIN_CLIPS,
        "validation_clip_count": STRICT_SCREEN_VALIDATION_CLIPS,
        "decoder_steps": STRICT_PARENT_STEPS,
        "background_window_count": STRICT_SCREEN_BACKGROUNDS,
        "fixture_cells_per_background": STRICT_SCREEN_FIXTURES
        // STRICT_SCREEN_BACKGROUNDS,
    }
    mismatches = {
        key: (screen.get(key), expected)
        for key, expected in strict.items()
        if int(screen.get(key, -1)) != expected
    }
    if mismatches:
        raise JEPAResidualPilotError(
            f"screen mode must retain the bounded frozen budget: {mismatches}"
        )
    upstream = config["upstream_screen"]
    if upstream.get("experiment_id") != "NREV-EXP-0028":
        raise JEPAResidualPilotError("upstream_screen must identify NREV-EXP-0028")
    if upstream.get("run_id") != "NREV-RUN-EXP-0028-SCREEN-20260829-B":
        raise JEPAResidualPilotError("upstream_screen must identify the frozen Screen-B run")
    checkpoint = _require(upstream, "checkpoint", dict)
    if int(checkpoint.get("training_seed", STRICT_PARENT_SEED)) != STRICT_PARENT_SEED:
        raise JEPAResidualPilotError("upstream checkpoint seed must be 1001")
    frozen_data = config["frozen_data_contract"]
    normalization = _require(frozen_data, "normalization", dict)
    pixel_decoder = config["pixel_decoder"]
    training_plan = config["training_plan"]
    paired = config["paired_evaluation"]
    geometry = config["blind_tube_geometry"]
    resources = config["resource_guards"]
    if resources.get("conflicting_process_check") != "operator_observed_external_preflight":
        raise JEPAResidualPilotError(
            "screen conflicting-process guard must be an operator observation"
        )
    if resources.get("screen_require_cuda") is not True or resources.get(
        "screen_require_bfloat16"
    ) is not True:
        raise JEPAResidualPilotError("screen CUDA/BF16 guards must remain enabled")
    _require(config, "engineering_integrity_thresholds", dict)
    required_outputs = _require(config, "required_outputs", list)
    if not required_outputs or any(not isinstance(value, str) for value in required_outputs):
        raise JEPAResidualPilotError("required_outputs must be a nonempty string list")
    # Internal canonical sections keep execution code compact while the
    # persisted resolved configuration remains the authored descriptor above.
    config["_normalization"] = normalization
    config["_decoder"] = {
        "seed": int(training_plan["decoder_seed"]),
        "training_seed": int(training_plan["tile_schedule_seed"]),
        "trainable_parameters": int(pixel_decoder["trainable_parameters"]),
        "batch_size": int(pixel_decoder["batch_size"]),
        "learning_rate": float(pixel_decoder["learning_rate"]),
        "weight_decay": float(pixel_decoder["weight_decay"]),
        "adamw_betas": list(pixel_decoder["betas"]),
        "adamw_epsilon": float(pixel_decoder["epsilon"]),
        "gradient_clip_norm": float(pixel_decoder["gradient_clip_norm"]),
        "loss_beta": float(pixel_decoder["smooth_l1_beta"]),
        "hard_step_cap": int(pixel_decoder["hard_step_cap"]),
        "log_interval": int(pixel_decoder["log_interval_steps"]),
        "amp": str(pixel_decoder["autocast"])
        == "cuda_bfloat16_if_supported",
        "target_batch_size": int(pixel_decoder.get("inference_target_batch_size", 8)),
    }
    config["_evaluation"] = {
        "proposal_cap": int(paired["proposal_cap"]),
        "match_radius_px": float(paired["match_radius_px"]),
        "minimum_distance_px": int(paired["minimum_distance_px"]),
        "candidate_border_px": int(paired["candidate_border_px"]),
        "grouped_bootstrap_draws": int(paired["bootstrap_draws"]),
        "grouped_bootstrap_seed": int(paired["bootstrap_seed"]),
    }
    config["_blind_tube"] = {
        "token_shape_tyx": list(config["frozen_jepa"]["tubelet_shape_t_y_x"]),
        "target_pixels_yx": list(geometry["written_target_pixels_t_y_x"][-2:]),
    }
    config["_resources"] = {
        "minimum_free_disk_mib": int(float(resources["minimum_free_disk_gib"]) * 1024),
        "minimum_available_ram_mib": int(float(resources["minimum_available_ram_gib"]) * 1024),
        "minimum_free_cuda_mib": int(resources["minimum_free_cuda_mib"]),
        "cpu_threads": int(resources["cpu_threads"]),
    }
    return config


def _output_paths(requested: Path) -> tuple[Path, Path]:
    output = requested.expanduser().resolve()
    return output, output.with_name(output.name + ".partial")


def _guard_output_override(mode: str, output_root: Path | None) -> None:
    """Forbid redirecting the registered screen while allowing test modes."""

    if mode == "screen" and output_root is not None:
        raise JEPAResidualPilotError(
            "screen mode forbids output_root overrides; use the canonical "
            f"registered root {STRICT_OUTPUT_ROOT}"
        )


def guard_output_collision(requested: Path) -> tuple[Path, Path]:
    """Reject both committed and interrupted output roots before any write."""

    output, partial = _output_paths(requested)
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing existing output or partial root: {output}")
    return output, partial


def _artifact_index_rows(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = index.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise JEPAResidualPilotError("parent artifact index must contain artifacts")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise JEPAResidualPilotError("parent artifact row must be an object")
        logical = str(raw.get("path", ""))
        if not logical or logical in seen:
            raise JEPAResidualPilotError("parent artifact paths must be unique and nonempty")
        seen.add(logical)
        parsed.append(
            {
                "path": logical,
                "bytes": int(raw.get("bytes", -1)),
                "sha256": _valid_sha256(raw.get("sha256"), field=f"artifact {logical}"),
            }
        )
    return parsed


def verify_parent_run_integrity(
    parent_root: Path,
    *,
    expected_artifact_index_sha256: str,
) -> dict[str, Any]:
    """Hash every indexed parent file; this must run before ``torch.load``."""

    root = Path(parent_root).expanduser().resolve()
    index_path = root / "artifact_index.json"
    expected_index = _valid_sha256(
        expected_artifact_index_sha256, field="upstream artifact_index_sha256"
    )
    if not index_path.is_file():
        raise JEPAResidualPilotError("upstream artifact_index.json is missing")
    observed_index = sha256_file(index_path)
    if observed_index != expected_index:
        raise JEPAResidualPilotError("upstream artifact index hash mismatch")
    index = _json_object(index_path)
    rows = _artifact_index_rows(index)
    verified: list[dict[str, Any]] = []
    for row in rows:
        candidate = _relative_child(root, row["path"], field="parent artifact path")
        if not candidate.is_file() or candidate.is_symlink():
            raise JEPAResidualPilotError(
                f"upstream artifact is absent or is a symlink: {row['path']}"
            )
        size = int(candidate.stat().st_size)
        observed = sha256_file(candidate)
        if size != row["bytes"] or observed != row["sha256"]:
            raise JEPAResidualPilotError(
                f"upstream artifact hash/size mismatch: {row['path']}"
            )
        verified.append(
            {
                "path": row["path"],
                "bytes": size,
                "sha256": observed,
                "matches": True,
            }
        )
    snapshot = {
        "artifact_index_sha256": observed_index,
        "artifacts": verified,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "all_parent_artifacts_verified": True,
        "verified_before_checkpoint_load": True,
        "artifact_count": len(verified),
        **snapshot,
        "snapshot_sha256": stable_hash(snapshot),
    }


def assert_parent_unchanged(
    before: Mapping[str, Any],
    parent_root: Path,
    *,
    expected_artifact_index_sha256: str,
) -> dict[str, Any]:
    """Re-hash the parent after execution and prove byte identity."""

    after = verify_parent_run_integrity(
        parent_root,
        expected_artifact_index_sha256=expected_artifact_index_sha256,
    )
    unchanged = (
        str(before.get("snapshot_sha256")) == str(after.get("snapshot_sha256"))
    )
    if not unchanged:
        raise JEPAResidualPilotError("upstream Screen-B root changed during execution")
    return {
        "parent_root_byte_identical_pre_post": True,
        "before_snapshot_sha256": before["snapshot_sha256"],
        "after_snapshot_sha256": after["snapshot_sha256"],
        "artifact_count": after["artifact_count"],
    }


def _pinned_parent_paths(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[Path, dict[str, Path]]:
    upstream = config["upstream_screen"]
    root = _resolve_repo_path(str(upstream["output_root"]), repository_root)
    keys = {
        "resolved_config": "resolved_config.json",
        "data_manifest": "data_manifest.json",
        "clip_bank_manifest": "clip_bank_manifest.json",
        "background_window_manifest": "background_window_manifest.json",
        "paired_injection_manifest": "paired_injection_manifest.json",
        "paired_injection_results": "paired_injection_results.json",
    }
    paths = {key: _relative_child(root, name, field=key) for key, name in keys.items()}
    paths["checkpoint"] = _relative_child(
        root, str(upstream["checkpoint"]["path"]), field="checkpoint.path"
    )
    paths["train_cache"] = _relative_child(
        root, str(upstream["train_cache"]["path"]), field="train_cache.path"
    )
    paths["validation_cache"] = _relative_child(
        root,
        str(upstream["validation_cache"]["path"]),
        field="validation_cache.path",
    )
    return root, paths


def _verify_pinned_parent_files(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    upstream = config["upstream_screen"]
    expected = {
        "resolved_config": upstream["resolved_config_sha256"],
        "data_manifest": upstream["data_manifest_sha256"],
        "clip_bank_manifest": upstream["clip_bank_manifest_sha256"],
        "background_window_manifest": upstream["background_window_manifest_sha256"],
        "paired_injection_manifest": upstream["paired_injection_manifest_sha256"],
        "paired_injection_results": upstream["paired_injection_results_sha256"],
        "checkpoint": upstream["checkpoint"]["sha256"],
        "train_cache": upstream["train_cache"]["sha256"],
        "validation_cache": upstream["validation_cache"]["sha256"],
    }
    rows = []
    for key, digest in expected.items():
        expected_hash = _valid_sha256(digest, field=f"upstream {key} hash")
        observed = sha256_file(paths[key])
        if observed != expected_hash:
            raise JEPAResidualPilotError(f"pinned upstream file hash mismatch: {key}")
        rows.append({"role": key, "sha256": observed, "matches": True})
    return {"all_pinned_parent_files_match": True, "files": rows}


def load_normalized_parent_cache(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: Sequence[int],
    expected_dtype: str,
) -> tuple[np.memmap, dict[str, Any]]:
    """Memory-map the derived normalized cache without transforming values."""

    observed_hash = sha256_file(path)
    if observed_hash != _valid_sha256(expected_sha256, field="cache sha256"):
        raise JEPAResidualPilotError(f"normalized cache hash mismatch: {path.name}")
    array = np.load(path, mmap_mode="c", allow_pickle=False)
    shape = tuple(int(value) for value in expected_shape)
    if tuple(array.shape) != shape or array.dtype.name != str(expected_dtype):
        raise JEPAResidualPilotError(
            f"normalized cache shape/dtype mismatch: {array.shape}/{array.dtype}"
        )
    if not np.isfinite(array).all():
        raise JEPAResidualPilotError("normalized cache contains non-finite values")
    return array, {
        "path_role": "upstream_derived_normalized_cache",
        "shape": list(shape),
        "dtype": array.dtype.name,
        "sha256": observed_hash,
        "normalization_applied_by_this_runner": False,
        "double_normalization_forbidden": True,
        "mmap_mode": "copy_on_write",
    }


def normalization_from_config(config: Mapping[str, Any]) -> RobustNormalization:
    values = config["_normalization"]
    return RobustNormalization(
        center=float(values["center"]),
        scale=float(values["scale"]),
        unscaled_mad=float(values.get("unscaled_mad", float(values["scale"]) / 1.4826)),
        scale_was_floored=bool(values.get("scale_was_floored", False)),
        fit_recording_ids=tuple(str(value) for value in values.get("fit_recording_ids", ())),
        uniform_frame_indices=tuple(),
        spatial_stride=int(values.get("spatial_stride", 8)),
    )


def normalize_raw_movie_once(movie: np.ndarray, normalization: RobustNormalization) -> np.ndarray:
    """Raw-to-model-units boundary used only by the learned residual arms."""

    raw = np.asarray(movie, dtype=np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise JEPAResidualPilotError("raw evaluation movie must be finite TYX")
    return normalization.apply(raw)


def tensor_mapping_sha256(state: Mapping[str, Tensor]) -> str:
    """Stable content hash for a tensor state mapping, independent of serialization."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    return tensor_mapping_sha256(module.state_dict())


def _load_strict_checkpoint_payload(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load only after parent verification, using PyTorch's restricted loader."""

    if sha256_file(path) != _valid_sha256(expected_sha256, field="checkpoint sha256"):
        raise JEPAResidualPilotError("checkpoint hash mismatch before torch.load")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise JEPAResidualPilotError("checkpoint payload must be a mapping")
    schedule = payload.get("schedule")
    checks = {
        "schema_version_1": int(payload.get("schema_version", -1)) == 1,
        "screen_mode": payload.get("mode") == "screen",
        "non_scientific_checkpoint": payload.get("scientific_checkpoint") is False,
        "training_seed_1001": int(payload.get("training_seed", -1)) == STRICT_PARENT_SEED,
        "optimizer_steps_500": isinstance(schedule, Mapping)
        and int(schedule.get("steps", -1)) == STRICT_PARENT_STEPS,
        "jepa_state_present": isinstance(payload.get("jepa_state_dict"), Mapping),
        "random_state_present": isinstance(payload.get("random_state_dict"), Mapping),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise JEPAResidualPilotError(
            "upstream checkpoint contract failed: " + ", ".join(failed)
        )
    payload["_strict_checks"] = checks
    payload["_jepa_tensor_sha256"] = tensor_mapping_sha256(payload["jepa_state_dict"])
    payload["_random_tensor_sha256"] = tensor_mapping_sha256(payload["random_state_dict"])
    return payload


def _validate_implementation_hashes(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    contract = config["implementation_hash_contract"]
    if contract.get("hash_algorithm") != "sha256":
        raise JEPAResidualPilotError("implementation hash contract must use sha256")
    sources = [
        *contract.get("upstream_sources", []),
        *contract.get("new_sources", []),
    ]
    if not isinstance(sources, list) or not sources:
        raise JEPAResidualPilotError("implementation hash sources must be nonempty")
    rows = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise JEPAResidualPilotError("implementation source must be an object")
        logical = str(source.get("path", ""))
        path = _relative_child(repository_root, logical, field="implementation path")
        raw_expected = source.get("sha256")
        if raw_expected == "PENDING_STABLE_IMPLEMENTATION_HASH":
            rows.append(
                {
                    "path": logical,
                    "role": str(source.get("role", "unspecified")),
                    "expected_sha256": raw_expected,
                    "observed_sha256": sha256_file(path),
                    "matches": False,
                    "status": "pending_hash_must_be_frozen_before_execution",
                }
            )
            continue
        expected = _valid_sha256(raw_expected, field=f"implementation {logical}")
        observed = sha256_file(path)
        rows.append(
            {
                "path": logical,
                "role": str(source.get("role", "unspecified")),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": observed == expected,
            }
        )
    return {
        "hash_algorithm": "sha256",
        "sources": rows,
        "all_match": all(row["matches"] for row in rows),
    }


def _resource_snapshot(output_parent: Path, device: str) -> dict[str, Any]:
    probe = output_parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    cuda = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        cuda.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "free_mib": int(free // 2**20),
                "total_mib": int(total // 2**20),
                "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    available_ram_mib: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_ram_mib = int(line.split()[1]) // 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    return {
        "requested_device": device,
        "cpu_count": os.cpu_count(),
        "disk_free_mib": int(disk.free // 2**20),
        "available_ram_mib": available_ram_mib,
        "cuda": cuda,
    }


def _execution_provenance(repository_root: Path) -> dict[str, Any]:
    """Collect non-secret, path-free runtime and Git identity evidence."""

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD").decode("ascii")
        branch = git("branch", "--show-current").decode("utf-8") or "detached"
        dirty_bytes = git("status", "--porcelain=v1", "-z")
        dirty_entry_count = dirty_bytes.count(b"\0")
        git_status = "captured"
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        commit = "unavailable"
        branch = "unavailable"
        dirty_bytes = b""
        dirty_entry_count = -1
        git_status = "unavailable"
    cuda_runtime = torch.version.cuda
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "git": {
            "status": git_status,
            "commit": commit,
            "branch": branch,
            "dirty": dirty_entry_count > 0,
            "dirty_entry_count": dirty_entry_count,
            "dirty_status_sha256": hashlib.sha256(dirty_bytes).hexdigest(),
            "dirty_paths_persisted": False,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable_basename": Path(sys.executable).name,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": cuda_runtime,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": cuda_device,
        },
    }


def _authorization_state(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the one allowed engineering authorization and all forbidden ones."""

    forbidden_fields = (
        "execution_authorized",
        "scientific_execution_authorized",
        "claim_bearing_execution_authorized",
        "scientific_completion_allowed",
        "claim_bearing_execution",
    )
    forbidden = {field: config.get(field) for field in forbidden_fields}
    return {
        "engineering_screen_execution_authorized": config.get(
            "engineering_screen_execution_authorized"
        )
        is True,
        "engineering_screen_authorization_basis_present": bool(
            str(config.get("engineering_screen_authorization_basis", "")).strip()
        ),
        "forbidden_scientific_or_claim_authorizations": forbidden,
        "all_scientific_and_claim_bearing_authorizations_false": all(
            value is False for value in forbidden.values()
        ),
        "scope": "bounded_engineering_screen_only",
    }


def preflight_jepa_residual_pilot(
    config_path: Path,
    *,
    repository_root: Path,
    mode: str = "preflight",
    data_root: Path | None = None,
    output_root: Path | None = None,
    operator_process_check_acknowledged: bool = False,
) -> dict[str, Any]:
    """Perform a read-only integrity/resource preflight; never load weights."""

    if mode not in SUPPORTED_MODES:
        raise JEPAResidualPilotError(f"mode must be one of {sorted(SUPPORTED_MODES)}")
    _guard_output_override(mode, output_root)
    repository = Path(repository_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = load_jepa_residual_config(config_file)
    requested_output = (
        _resolve_repo_path(config["output_root"], repository)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    output, partial = _output_paths(requested_output)
    parent_root, paths = _pinned_parent_paths(config, repository)
    parent_integrity = verify_parent_run_integrity(
        parent_root,
        expected_artifact_index_sha256=config["upstream_screen"][
            "artifact_index_sha256"
        ],
    )
    pinned = _verify_pinned_parent_files(config, paths)
    implementation = _validate_implementation_hashes(config, repository)
    mode_config = config["execution_modes"][mode]
    device = str(mode_config.get("device", "cpu"))
    resource = _resource_snapshot(output.parent, device)
    provenance = _execution_provenance(repository)
    authorization = _authorization_state(config)
    checks = {
        "output_absent": not output.exists(),
        "partial_output_absent": not partial.exists(),
        "all_parent_artifacts_verified": parent_integrity[
            "all_parent_artifacts_verified"
        ],
        "all_pinned_parent_files_match": pinned["all_pinned_parent_files_match"],
        "implementation_hashes_match": bool(implementation["all_match"]),
        "engineering_screen_execution_authorized": authorization[
            "engineering_screen_execution_authorized"
        ],
        "engineering_screen_authorization_basis_present": authorization[
            "engineering_screen_authorization_basis_present"
        ],
        "all_scientific_and_claim_bearing_authorizations_false": authorization[
            "all_scientific_and_claim_bearing_authorizations_false"
        ],
        "screen_requests_cuda": mode != "screen" or device == "cuda",
        "screen_cuda_device_available": mode != "screen"
        or bool(torch.cuda.is_available()),
        "screen_cuda_bfloat16_available": mode != "screen"
        or bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "screen_minimum_free_cuda_memory": mode != "screen"
        or int(resource["cuda"].get("free_mib", -1))
        >= int(config["_resources"]["minimum_free_cuda_mib"]),
        "screen_operator_conflicting_process_check_acknowledged": mode != "screen"
        or bool(operator_process_check_acknowledged),
        "minimum_free_disk": resource["disk_free_mib"]
        >= int(config["_resources"]["minimum_free_disk_mib"]),
        "minimum_available_ram": resource["available_ram_mib"] is not None
        and int(resource["available_ram_mib"])
        >= int(config["_resources"]["minimum_available_ram_mib"]),
        "execution_is_non_scientific": config.get("scientific_completion_allowed")
        is False
        and config.get("scientific_execution_authorized") is False,
        "promotion_is_forbidden": config.get("claim_bearing_execution") is False
        and config.get("claim_bearing_execution_authorized") is False,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failed else "failed",
        "checked_at": _utc_now(),
        "mode": mode,
        "checkpoint_loaded": False,
        "checks": checks,
        "failed_checks": failed,
        "parent_integrity": parent_integrity,
        "pinned_parent_files": pinned,
        "implementation_hash_validation": implementation,
        "config": {
            "uri": portable_path(
                config_file, repository=repository, data=data_root
            ),
            "sha256": sha256_file(config_file),
            "experiment_id": config["experiment_id"],
            "run_id": config["run_id"],
        },
        "resource_snapshot": resource,
        "execution_provenance": provenance,
        "authorization": {
            **authorization,
            "operator_process_check_acknowledged": bool(
                operator_process_check_acknowledged
            ),
            "operator_process_check_acknowledged_at": (
                _utc_now() if operator_process_check_acknowledged else None
            ),
        },
        "scientific_execution_ready": False,
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
        "output": {
            "uri": portable_path(output, repository=repository, data=data_root),
            "partial_uri": portable_path(partial, repository=repository, data=data_root),
        },
    }


def _parent_registered_config(paths: Mapping[str, Path]) -> dict[str, Any]:
    resolved = _json_object(paths["resolved_config"])
    registered = resolved.get("registered_config")
    if not isinstance(registered, dict):
        raise JEPAResidualPilotError("parent resolved_config lacks registered_config")
    return registered


def _normalization_contract_check(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    data_manifest = _json_object(paths["data_manifest"])
    per_seed = data_manifest.get("per_training_seed")
    if not isinstance(per_seed, Mapping) or not isinstance(per_seed.get("1001"), Mapping):
        raise JEPAResidualPilotError("parent data manifest lacks seed-1001 normalization")
    parent = per_seed["1001"].get("normalization")
    if not isinstance(parent, Mapping):
        raise JEPAResidualPilotError("parent normalization manifest is absent")
    declared = config["_normalization"]
    checks = {
        "center_exact": float(declared["center"]) == float(parent["center"]),
        "scale_exact": float(declared["scale"]) == float(parent["scale"]),
        "normalization_sha256_exact": str(declared["sha256"])
        == str(parent["normalization_sha256"]),
    }
    if not all(checks.values()):
        raise JEPAResidualPilotError("EXP-0029 normalization differs from parent Screen-B")
    return {
        "checks": checks,
        "normalization_sha256": parent["normalization_sha256"],
        "raw_hc_fixture_normalization_application_count": 0,
        "residual_arm_fixture_normalization_application_count": 1,
        "normalized_cache_normalization_application_count": 0,
    }


def reconstruct_frozen_backgrounds(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    runtime_data_root: Path,
    paths: Mapping[str, Path],
    verify_source_hashes: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the exact 12 raw clips named by the frozen parent manifest."""

    parent_config = _parent_registered_config(paths)
    descriptor = _resolve_repo_path(str(config["data_inventory"]), repository_root)
    inventory = load_recording_inventory(
        descriptor,
        repository_root=repository_root,
        data_root=runtime_data_root,
        verify_live=True,
        verify_hashes=bool(verify_source_hashes),
    )
    holdout = inventory_source_holdout(parent_config, repository_root, runtime_data_root)
    if verify_source_hashes and sha256_file(holdout.resolved_path) != holdout.sha256:
        raise JEPAResidualPilotError("Spon holdout hash mismatch")
    lookup = inventory.by_id()
    lookup[holdout.recording_id] = holdout
    manifest = _json_object(paths["background_window_manifest"])
    rows = manifest.get("windows")
    if not isinstance(rows, list) or len(rows) != STRICT_SCREEN_BACKGROUNDS:
        raise JEPAResidualPilotError("parent background manifest must contain 12 windows")
    shape = tuple(int(value) for value in parent_config["clip"]["shape_tyx"])
    contract = TemporalClipContract(
        clip_frames=shape[0],
        guard_frames=int(parent_config["clip"]["temporal_guard_frames"]),
        frame_step=int(parent_config["clip"]["frame_step"]),
    )
    backgrounds: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        recording_id = str(row["recording_id"])
        if recording_id not in lookup:
            raise JEPAResidualPilotError(f"background recording is unavailable: {recording_id}")
        request = ClipRequest(
            recording_id=recording_id,
            split=str(row["split"]),
            start_frame_zero=int(row["start_frame_zero"]),
            stop_frame_zero_exclusive=int(row["stop_frame_zero_exclusive"]),
            y_zero=int(row["y_zero"]),
            x_zero=int(row["x_zero"]),
            height=int(row["height"]),
            width=int(row["width"]),
            sampling_stratum=str(row["sampling_stratum"]),
            temporal_mad=float(row["temporal_mad"]),
        )
        clip = np.asarray(
            read_sequential_clip(lookup[recording_id], request, contract),
            dtype=np.float32,
        )
        if tuple(clip.shape) != shape:
            raise JEPAResidualPilotError("reconstructed background shape drifted")
        backgrounds.append({"metadata": row, "clip": clip})
    return backgrounds, {
        "schema_version": SCHEMA_VERSION,
        "source": "upstream_frozen_background_window_manifest",
        "window_count": len(backgrounds),
        "windows": rows,
        "raw_video_copied": False,
        "source_hashes_verified": bool(verify_source_hashes),
    }


def assert_fixture_manifest_identity(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require byte-semantic identity of the regenerated fixture manifest."""

    if dict(observed) != dict(expected):
        raise JEPAResidualPilotError("regenerated paired fixture manifest differs from parent")
    fixtures = observed.get("fixtures")
    if not isinstance(fixtures, list):
        raise JEPAResidualPilotError("paired fixture manifest lacks fixtures")
    closure = max(
        float(row["maximum_pair_closure_float32_ulp"])
        for row in fixtures
        if isinstance(row, Mapping)
    )
    if closure > 0.51:
        raise JEPAResidualPilotError("paired injection closure exceeds 0.51 float32 ULP")
    return {
        "exact_parent_fixture_manifest_identity": True,
        "fixture_count": len(fixtures),
        "maximum_pair_closure_float32_ulp": closure,
        "maximum_allowed_float32_ulp": 0.51,
    }


def reconstruct_exact_fixtures(
    parent_config: Mapping[str, Any],
    backgrounds: Sequence[Mapping[str, Any]],
    expected_manifest: Mapping[str, Any],
) -> tuple[list[Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    fixtures, metadata, summary = _make_fixtures(
        parent_config,
        backgrounds,
        smoke=False,
        injection_cells_per_background=None,
    )
    manifest = {
        "schema_version": 1,
        **summary,
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
    }
    identity = assert_fixture_manifest_identity(manifest, expected_manifest)
    if len(fixtures) != STRICT_SCREEN_FIXTURES:
        raise JEPAResidualPilotError("exact fixture grid must contain 108 cells")
    source_total = sum(int(fixture.footprints.shape[0]) for fixture in fixtures)
    if source_total != STRICT_SCREEN_SOURCES:
        raise JEPAResidualPilotError("exact fixture grid must contain 252 source occurrences")
    return fixtures, metadata, summary, identity


def _execution_fixture_summary(
    fixtures: Sequence[Any], registered_summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Separate actually executed fixture coverage from the registered grid."""

    if not fixtures:
        raise JEPAResidualPilotError("execution fixture subset must be nonempty")
    planned = int(registered_summary["planned_fixture_count"])
    observed = len(fixtures)
    result = {
        "observed_fixture_count": observed,
        "planned_registered_fixture_count": planned,
        "registered_exact_fixture_count": STRICT_SCREEN_FIXTURES,
        "registered_exact_injected_source_occurrences": STRICT_SCREEN_SOURCES,
        "execution_coverage_fraction_of_registered_grid": float(observed / planned),
        "full_registered_grid": observed == planned,
        "background_recording_ids": sorted(
            {str(fixture.metadata["background_recording_id"]) for fixture in fixtures}
        ),
        "background_window_ids": sorted(
            {str(fixture.metadata["background_window_id"]) for fixture in fixtures}
        ),
        "source_counts": sorted(
            {int(fixture.metadata["source_count"]) for fixture in fixtures}
        ),
        "injection_seeds": sorted(
            {int(fixture.metadata["injection_seed"]) for fixture in fixtures}
        ),
        "exact_injected_source_occurrences": int(
            sum(int(fixture.footprints.shape[0]) for fixture in fixtures)
        ),
        "registered_grid": dict(registered_summary),
    }
    if result["execution_coverage_fraction_of_registered_grid"] > 1.0:
        raise JEPAResidualPilotError("execution fixture coverage exceeds registered grid")
    return result


def raw_hc_cross_run_regression(
    parent_payload: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    executed_fixture_ids: Sequence[str],
    parent_run_id: str,
    current_run_id: str,
) -> dict[str, Any]:
    """Prove exact raw-HC RecoveryResult continuity with EXP-0028 Screen B."""

    parent_rows = parent_payload.get("rows")
    if not isinstance(parent_rows, list):
        raise JEPAResidualPilotError("parent paired results lack rows")
    selected_parent = [
        row
        for row in parent_rows
        if isinstance(row, Mapping)
        and row.get("method") == PARENT_RAW_METHOD
        and int(row.get("training_seed", -1)) == STRICT_PARENT_SEED
    ]
    if len(selected_parent) != STRICT_SCREEN_FIXTURES:
        raise JEPAResidualPilotError(
            "parent raw-HC anchor must contain exactly 108 seed-1001 fixtures"
        )
    parent_by_fixture = {
        str(row["fixture_id"]): row for row in selected_parent
    }
    if len(parent_by_fixture) != len(selected_parent):
        raise JEPAResidualPilotError("parent raw-HC anchor fixture IDs are not unique")
    selected_current = [
        row for row in current_rows if str(row.get("method")) == RAW_METHOD
    ]
    current_by_fixture = {
        str(row["fixture_id"]): row for row in selected_current
    }
    expected_ids = sorted(str(value) for value in executed_fixture_ids)
    if len(expected_ids) != len(set(expected_ids)):
        raise JEPAResidualPilotError("execution fixture IDs are not unique")
    if sorted(current_by_fixture) != expected_ids:
        raise JEPAResidualPilotError(
            "current raw-HC rows do not exactly cover the execution fixture subset"
        )
    missing_parent = sorted(set(expected_ids) - set(parent_by_fixture))
    if missing_parent:
        raise JEPAResidualPilotError(
            "execution fixtures are missing from the parent raw-HC anchor"
        )
    compared_fields = ("source_on_recovery", "intervention_recovery")
    mismatches: list[dict[str, Any]] = []
    for fixture_id in expected_ids:
        parent_row = parent_by_fixture[fixture_id]
        current_row = current_by_fixture[fixture_id]
        for field in compared_fields:
            if parent_row.get(field) != current_row.get(field):
                mismatches.append(
                    {
                        "fixture_id": fixture_id,
                        "field": field,
                        "parent_sha256": stable_hash(parent_row.get(field)),
                        "current_sha256": stable_hash(current_row.get(field)),
                    }
                )
    mismatched_fixture_ids = sorted(
        {str(row["fixture_id"]) for row in mismatches}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_experiment_id": "NREV-EXP-0028",
        "parent_run_id": str(parent_run_id),
        "current_experiment_id": EXPERIMENT_ID,
        "current_run_id": str(current_run_id),
        "parent_method": PARENT_RAW_METHOD,
        "current_method": RAW_METHOD,
        "parent_registered_fixture_count": len(parent_by_fixture),
        "expected_execution_fixture_count": len(expected_ids),
        "observed_current_raw_row_count": len(selected_current),
        "execution_fixture_count": len(expected_ids),
        "matched_fixture_count": len(expected_ids) - len(mismatched_fixture_ids),
        "executed_fixture_ids_sha256": stable_hash(expected_ids),
        "compared_fields": list(compared_fields),
        "recall_compared_as_part_of_each_recovery_result": True,
        "numeric_tolerance": 0.0,
        "exact_recovery_result_and_recall_identity": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatched_fixture_ids": mismatched_fixture_ids,
        "mismatches": mismatches,
        "scientific_claim_consequence": "none_engineering_continuity_check_only",
    }


def paired_signal_algebra(
    source_off: np.ndarray,
    source_on: np.ndarray,
    predicted_off: np.ndarray,
    predicted_on: np.ndarray,
    injected_signal: np.ndarray,
) -> dict[str, float]:
    """Quantify retention/absorption and verify their additive algebra."""

    off, on, pred_off, pred_on, signal = (
        np.asarray(value, dtype=np.float64)
        for value in (source_off, source_on, predicted_off, predicted_on, injected_signal)
    )
    if not (off.shape == on.shape == pred_off.shape == pred_on.shape == signal.shape):
        raise JEPAResidualPilotError("paired residual algebra arrays must share shape")
    energy = float(np.sum(signal * signal))
    if not math.isfinite(energy) or energy <= np.finfo(float).eps:
        raise JEPAResidualPilotError("injected signal energy must be positive")
    prediction_delta = pred_on - pred_off
    residual_delta = (on - pred_on) - (off - pred_off)
    absorption = float(np.sum(prediction_delta * signal) / energy)
    retention = float(np.sum(residual_delta * signal) / energy)
    closure_array = residual_delta + prediction_delta - signal
    total_error = float(np.linalg.norm(residual_delta - signal) / math.sqrt(energy))
    orthogonal_distortion = float(
        np.linalg.norm(residual_delta - retention * signal) / math.sqrt(energy)
    )
    return {
        "retained_gain_projection": retention,
        "predictor_absorption_projection": absorption,
        "retention_plus_absorption_minus_one": retention + absorption - 1.0,
        "pair_closure_max_abs": float(np.max(np.abs(closure_array))),
        "total_signal_error_ratio": total_error,
        "orthogonal_distortion_ratio": orthogonal_distortion,
    }


def _decoder_diagnostic_agreement(
    algebra: Mapping[str, Any],
    decoder_reference: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Cross-check canonical runner ratios against the independent core helper."""

    signal = decoder_reference.get("signal")
    if not isinstance(signal, Mapping):
        raise JEPAResidualPilotError("decoder residual diagnostics lack signal metrics")
    fields: dict[str, Any] = {}
    for field in ("total_signal_error_ratio", "orthogonal_distortion_ratio"):
        runner_value = float(algebra[field])
        decoder_value = float(signal[field])
        difference = abs(runner_value - decoder_value)
        fields[field] = {
            "runner": runner_value,
            "decoder_reference": decoder_value,
            "absolute_difference": difference,
            "absolute_tolerance": float(absolute_tolerance),
            "matches": difference <= float(absolute_tolerance),
        }
    result = {"fields": fields, "all_match": all(row["matches"] for row in fields.values())}
    if not result["all_match"]:
        raise JEPAResidualPilotError(
            "runner and decoder residual distortion diagnostics disagree"
        )
    return result


def _production_invariance_targets(grid_yx: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Enumerate every original target tile, including every edge position."""

    if len(grid_yx) != 2:
        raise JEPAResidualPilotError("target tile grid must contain y and x extents")
    rows, columns = (int(value) for value in grid_yx)
    if rows <= 0 or columns <= 0:
        raise JEPAResidualPilotError("target tile grid extents must be positive")
    return tuple((row, column) for row in range(rows) for column in range(columns))


def _normal_consistent_mad(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(array - np.median(array))))


def background_suppression_diagnostics(
    source_off: np.ndarray, predicted_off: np.ndarray
) -> dict[str, float]:
    off = np.asarray(source_off, dtype=np.float64)
    predicted = np.asarray(predicted_off, dtype=np.float64)
    if off.shape != predicted.shape or off.ndim != 3:
        raise JEPAResidualPilotError("background diagnostics require matching TYX arrays")
    centered = off - np.median(off, axis=0, keepdims=True)
    residual = off - predicted
    centered_residual = residual - np.median(residual, axis=0, keepdims=True)
    input_rms = float(np.sqrt(np.mean(centered * centered)))
    residual_rms = float(np.sqrt(np.mean(centered_residual * centered_residual)))
    input_mad = _normal_consistent_mad(centered)
    residual_mad = _normal_consistent_mad(centered_residual)
    input_dynamic_mad = _normal_consistent_mad(np.diff(off, axis=0))
    residual_dynamic_mad = _normal_consistent_mad(np.diff(residual, axis=0))
    eps = np.finfo(float).eps
    return {
        "source_off_centered_input_rms": input_rms,
        "source_off_residual_rms": residual_rms,
        "background_rms_ratio": residual_rms / max(input_rms, eps),
        "background_suppression_fraction_rms": 1.0
        - residual_rms / max(input_rms, eps),
        "source_off_centered_input_mad": input_mad,
        "source_off_centered_residual_mad": residual_mad,
        "background_mad_ratio": residual_mad / max(input_mad, eps),
        "source_off_input_frame_difference_mad": input_dynamic_mad,
        "source_off_residual_frame_difference_mad": residual_dynamic_mad,
        "dynamic_mad_ratio": residual_dynamic_mad / max(input_dynamic_mad, eps),
    }


def seam_diagnostics(background: np.ndarray, *, patch_yx: Sequence[int]) -> dict[str, float]:
    values = np.asarray(background, dtype=np.float64)
    if values.ndim != 3 or len(patch_yx) != 2:
        raise JEPAResidualPilotError("seam diagnostics require TYX and patch YX")
    height_step, width_step = (int(value) for value in patch_yx)
    vertical = np.abs(np.diff(values, axis=1))
    horizontal = np.abs(np.diff(values, axis=2))
    vertical_indices = np.arange(1, values.shape[1]) % height_step == 0
    horizontal_indices = np.arange(1, values.shape[2]) % width_step == 0
    seam_values = np.concatenate(
        (vertical[:, vertical_indices, :].reshape(-1), horizontal[:, :, horizontal_indices].reshape(-1))
    )
    interior_values = np.concatenate(
        (vertical[:, ~vertical_indices, :].reshape(-1), horizontal[:, :, ~horizontal_indices].reshape(-1))
    )
    seam_mean = float(np.mean(seam_values))
    interior_mean = float(np.mean(interior_values))
    return {
        "mean_absolute_spatial_seam_jump": seam_mean,
        "mean_absolute_spatial_interior_jump": interior_mean,
        "seam_to_interior_jump_ratio": seam_mean / max(interior_mean, np.finfo(float).eps),
    }


def footprint_halo_leakage_diagnostics(
    fixtures: Sequence[Any], *, patch_yx: Sequence[int]
) -> dict[str, Any]:
    """Recompute exact registered footprint mass/energy outside each 3x3 halo."""

    patch_height, patch_width = (int(value) for value in patch_yx)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        for source_index, footprint in enumerate(np.asarray(fixture.footprints, dtype=np.float64)):
            peak_row, peak_column = np.unravel_index(int(np.argmax(footprint)), footprint.shape)
            token_y, token_x = peak_row // patch_height, peak_column // patch_width
            y0 = max(0, (token_y - 1) * patch_height)
            y1 = min(footprint.shape[0], (token_y + 2) * patch_height)
            x0 = max(0, (token_x - 1) * patch_width)
            x1 = min(footprint.shape[1], (token_x + 2) * patch_width)
            outside = np.ones_like(footprint, dtype=bool)
            outside[y0:y1, x0:x1] = False
            mass = float(np.sum(footprint))
            energy = float(np.sum(footprint * footprint))
            rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "source_index": int(source_index),
                    "peak_yx": [int(peak_row), int(peak_column)],
                    "target_token_yx": [int(token_y), int(token_x)],
                    "mass_fraction_outside_halo": float(np.sum(footprint[outside]) / mass),
                    "energy_fraction_outside_halo": float(
                        np.sum((footprint * footprint)[outside]) / energy
                    ),
                }
            )
    if not rows:
        raise JEPAResidualPilotError("footprint leakage diagnostics require sources")
    mass_values = np.asarray(
        [row["mass_fraction_outside_halo"] for row in rows], dtype=np.float64
    )
    energy_values = np.asarray(
        [row["energy_fraction_outside_halo"] for row in rows], dtype=np.float64
    )
    return {
        "source_count": len(rows),
        "halo_tokens_yx": [3, 3],
        "patch_pixels_yx": [patch_height, patch_width],
        "algorithm": "float64_fraction_outside_peak_token_centered_3x3_spatial_halo",
        "mass_fraction_outside_halo": {
            "median": float(np.median(mass_values)),
            "p95": float(np.quantile(mass_values, 0.95)),
            "maximum": float(np.max(mass_values)),
        },
        "energy_fraction_outside_halo": {
            "median": float(np.median(energy_values)),
            "p95": float(np.quantile(energy_values, 0.95)),
            "maximum": float(np.max(energy_values)),
        },
        "rows": rows,
    }


class PrecomputedResidualHandcraftedScorer:
    """Expose pair-safe source-off fitting over already-computed residual movies."""

    def __init__(
        self,
        source_off: np.ndarray,
        source_on: np.ndarray,
        residual_off: np.ndarray,
        residual_on: np.ndarray,
    ) -> None:
        self.source_off = np.asarray(source_off, dtype=np.float32)
        self.source_on = np.asarray(source_on, dtype=np.float32)
        self.residual_off = np.asarray(residual_off, dtype=np.float32)
        self.residual_on = np.asarray(residual_on, dtype=np.float32)
        self.fit_calls = 0
        self.apply_calls: list[str] = []

    def fit_source_off(self, source_off: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        if not np.array_equal(np.asarray(source_off, dtype=np.float32), self.source_off):
            raise JEPAResidualPilotError("residual calibration received anything but source-off")
        self.fit_calls += 1
        frozen = frozen_handcrafted_comparator.fit_source_off(self.residual_off)

        def score(movie: np.ndarray) -> np.ndarray:
            values = np.asarray(movie, dtype=np.float32)
            if np.array_equal(values, self.source_off):
                self.apply_calls.append("source_off")
                residual = self.residual_off
            elif np.array_equal(values, self.source_on):
                self.apply_calls.append("source_on")
                residual = self.residual_on
            else:
                raise JEPAResidualPilotError("paired evaluator supplied an unknown movie")
            return np.asarray(frozen(residual), dtype=np.float64)

        return score


class FrozenNormalizedHandcraftedComparator:
    """Normalize raw fixture movies once, fitting HC calibration on source-off."""

    def __init__(self, normalization: RobustNormalization) -> None:
        self.normalization = normalization
        self.fit_calls = 0
        self.apply_calls: list[str] = []
        self._source_off: np.ndarray | None = None

    def fit_source_off(self, source_off: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        self.fit_calls += 1
        self._source_off = np.asarray(source_off, dtype=np.float32)
        normalized_off = normalize_raw_movie_once(source_off, self.normalization)
        frozen = frozen_handcrafted_comparator.fit_source_off(normalized_off)

        def score(movie: np.ndarray) -> np.ndarray:
            values = np.asarray(movie, dtype=np.float32)
            self.apply_calls.append(
                "source_off"
                if self._source_off is not None
                and np.array_equal(values, self._source_off)
                else "source_on"
            )
            normalized = normalize_raw_movie_once(movie, self.normalization)
            return np.asarray(frozen(normalized), dtype=np.float64)

        return score


class FrozenNativeHandcraftedComparator:
    """Fit and apply the parent Screen-B HC comparator in native raw units.

    The wrapper is intentionally thin: it adds pair-calibration evidence while
    forwarding the original float32 fixture arrays without normalization or
    any other transform.  This is the only scientific amendment in v1.1.
    """

    def __init__(self) -> None:
        self.fit_calls = 0
        self.apply_calls: list[str] = []
        self._source_off: np.ndarray | None = None

    def fit_source_off(self, source_off: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        self.fit_calls += 1
        self._source_off = np.asarray(source_off, dtype=np.float32)
        frozen = frozen_handcrafted_comparator.fit_source_off(self._source_off)

        def score(movie: np.ndarray) -> np.ndarray:
            values = np.asarray(movie, dtype=np.float32)
            self.apply_calls.append(
                "source_off"
                if self._source_off is not None
                and np.array_equal(values, self._source_off)
                else "source_on"
            )
            return np.asarray(frozen(values), dtype=np.float64)

        return score


def _compute_native_raw_hc_rows(
    fixtures: Sequence[Any],
    fixture_metadata: Mapping[str, Mapping[str, Any]],
    *,
    evaluation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every raw-HC pair in parent-native units before training.

    The returned rows are immutable-by-convention inputs to the later joined
    three-arm table.  Running this first makes the 216 RecoveryResult objects
    (source-on and intervention for 108 fixtures) a hard prerequisite for the
    first decoder optimizer step.
    """

    rows: list[dict[str, Any]] = []
    for fixture in sorted(fixtures, key=lambda item: item.fixture_id):
        metadata = fixture_metadata[str(fixture.fixture_id)]
        scorer = FrozenNativeHandcraftedComparator()
        row = evaluate_paired_fixture(
            fixture,
            scorer,
            method=RAW_METHOD,
            candidate_budget=int(evaluation["proposal_cap"]),
            match_radius_px=float(evaluation["match_radius_px"]),
            minimum_distance_px=int(evaluation["minimum_distance_px"]),
            border_px=int(evaluation["candidate_border_px"]),
        )
        if scorer.fit_calls != 1 or scorer.apply_calls != [
            "source_off",
            "source_on",
        ]:
            raise JEPAResidualPilotError(
                "native raw-HC source-off calibration contract drifted"
            )
        rows.append(
            {
                **row,
                "background_recording_id": metadata["background_recording_id"],
                "background_window_id": metadata["background_window_id"],
                "injection_seed": int(metadata["injection_seed"]),
                "source_count": int(metadata["source_count"]),
                "crowding_case": metadata["crowding_case"],
                "score_input_units": "raw_native_units",
            }
        )
    return rows, {
        "raw_source_off_calibration_pair_count": len(rows),
        "raw_source_on_recovery_result_count": len(rows),
        "raw_intervention_recovery_result_count": len(rows),
        "raw_recovery_result_object_count": 2 * len(rows),
        "score_input_units": "raw_native_units",
        "normalization_application_count": 0,
        "computed_before_first_decoder_optimizer_step": True,
    }


def _raw_hc_pretraining_anchor(
    fixtures: Sequence[Any],
    fixture_metadata: Mapping[str, Mapping[str, Any]],
    *,
    evaluation: Mapping[str, Any],
    parent_payload: Mapping[str, Any],
    parent_run_id: str,
    current_run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Compute and require the complete 108-pair native raw-HC anchor."""

    if len(fixtures) != STRICT_SCREEN_FIXTURES:
        raise JEPAResidualPilotError(
            "native raw-HC pretraining anchor requires all 108 fixtures"
        )
    rows, evidence = _compute_native_raw_hc_rows(
        fixtures,
        fixture_metadata,
        evaluation=evaluation,
    )
    anchor = raw_hc_cross_run_regression(
        parent_payload,
        rows,
        executed_fixture_ids=[fixture.fixture_id for fixture in fixtures],
        parent_run_id=parent_run_id,
        current_run_id=current_run_id,
    )
    anchor.update(
        {
            "raw_score_input_units": "raw_native_units",
            "runner_version": RUNNER_VERSION,
            "anchor_execution_phase": "after_fixture_reconstruction_before_decoder_construction_or_training",
            "source_on_recovery_result_count": len(rows),
            "intervention_recovery_result_count": len(rows),
            "recovery_result_object_count": 2 * len(rows),
        }
    )
    if len(rows) != STRICT_SCREEN_FIXTURES or anchor["recovery_result_object_count"] != 216:
        raise JEPAResidualPilotError(
            "native raw-HC pretraining anchor did not cover 216 RecoveryResults"
        )
    return rows, anchor, evidence


def _flatten_row(row: Mapping[str, Any]) -> dict[str, Any]:
    on = row["source_on_recovery"]
    intervention = row["intervention_recovery"]
    return {
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
        "injected_sources": on["injected_sources"],
        "source_on_candidate_count": on["candidate_count"],
        "source_on_unmatched_candidate_count_unknown": on[
            "unmatched_candidate_count_unknown"
        ],
        "intervention_recall": intervention["recall"],
        "intervention_candidate_count": intervention["candidate_count"],
        "median_center_delta_background_mad": row[
            "median_injected_center_score_delta_background_mad"
        ],
        "maximum_pair_closure_float32_ulp": row[
            "maximum_pair_closure_float32_ulp"
        ],
    }


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


def _grouped_primary_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    evaluation: Mapping[str, Any],
    expected_source_counts_per_cluster: int = 3,
    expected_source_count_values: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    methods = (JEPA_RESIDUAL_METHOD, RAW_METHOD, RANDOM_RESIDUAL_METHOD)
    fixtures = sorted({str(row["fixture_id"]) for row in rows})
    clusters: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    cluster_source_counts: dict[tuple[str, str, int], list[int]] = {}
    for fixture_id in fixtures:
        selected = [
            row
            for row in rows
            if str(row["fixture_id"]) == fixture_id and str(row["method"]) in methods
        ]
        if Counter(str(row["method"]) for row in selected) != Counter(methods):
            raise JEPAResidualPilotError(
                "each fixture requires one raw, JEPA-residual, and random-residual row"
            )
        representative = selected[0]
        cluster = (
            str(representative["background_recording_id"]),
            str(representative["background_window_id"]),
            int(representative["injection_seed"]),
        )
        fixture_source_counts = {int(row["source_count"]) for row in selected}
        if len(fixture_source_counts) != 1:
            raise JEPAResidualPilotError(
                "methods disagree about the fixture source-count condition"
            )
        cluster_source_counts.setdefault(cluster, []).append(
            next(iter(fixture_source_counts))
        )
        values = clusters.setdefault(cluster, {method: [] for method in methods})
        for row in selected:
            values[str(row["method"])].append(
                float(row["source_on_recovery"]["recall"])
            )
    cluster_rows = [
        {
            "background_recording_id": recording,
            "background_window_id": window,
            "injection_seed": seed,
            "method_values": {
                method: float(np.mean(values[method])) for method in methods
            },
            "source_counts_aggregated": len(values[RAW_METHOD]),
            "source_count_values": sorted(set(cluster_source_counts[(recording, window, seed)])),
        }
        for (recording, window, seed), values in sorted(clusters.items())
    ]
    expected_values = sorted(int(value) for value in expected_source_count_values)
    if len(expected_values) != len(set(expected_values)):
        raise JEPAResidualPilotError("expected source-count conditions must be unique")
    if any(
        row["source_counts_aggregated"] != int(expected_source_counts_per_cluster)
        or row["source_count_values"] != expected_values
        for row in cluster_rows
    ):
        raise JEPAResidualPilotError(
            "source-count conditions were not exactly nested within each cluster"
        )
    recording_ids = sorted(
        {str(row["background_recording_id"]) for row in cluster_rows}
    )
    comparisons = {
        "jepa_residual_minus_raw": (JEPA_RESIDUAL_METHOD, RAW_METHOD),
        "random_residual_minus_raw": (RANDOM_RESIDUAL_METHOD, RAW_METHOD),
        "jepa_residual_minus_random_residual": (
            JEPA_RESIDUAL_METHOD,
            RANDOM_RESIDUAL_METHOD,
        ),
    }
    bootstraps: dict[str, Any] = {}
    recording_effects_by_comparison: dict[str, dict[str, float]] = {}
    recording_directions_by_comparison: dict[str, dict[str, str]] = {}
    for name, (primary, comparator) in comparisons.items():
        if len(recording_ids) >= 2:
            comparison_rows = [
                {
                    "background_recording_id": row["background_recording_id"],
                    "background_window_id": row["background_window_id"],
                    "injection_seed": row["injection_seed"],
                    "method_values": {
                        primary: row["method_values"][primary],
                        comparator: row["method_values"][comparator],
                    },
                }
                for row in cluster_rows
            ]
            bootstraps[name] = hierarchical_strongest_comparator_bootstrap(
                comparison_rows,
                primary_method=primary,
                comparator_methods=(comparator,),
                draws=int(evaluation["grouped_bootstrap_draws"]),
                seed=int(evaluation["grouped_bootstrap_seed"]),
            )
        else:
            observed = float(
                np.mean(
                    [
                        row["method_values"][primary]
                        - row["method_values"][comparator]
                        for row in cluster_rows
                    ]
                )
            )
            bootstraps[name] = {
                "status": "not_run_requires_at_least_two_recordings",
                "observed_mean": observed,
                "confidence_interval_95": [None, None],
                "draws": 0,
                "recording_count": len(recording_ids),
                "nested_not_resampled_as_independent": ["source_count"],
            }
        recording_effects: dict[str, float] = {}
        for recording in recording_ids:
            selected = [
                row
                for row in cluster_rows
                if row["background_recording_id"] == recording
            ]
            recording_effects[recording] = float(
                np.mean(
                    [
                        row["method_values"][primary]
                        - row["method_values"][comparator]
                        for row in selected
                    ]
                )
            )
        recording_effects_by_comparison[name] = recording_effects
        recording_directions_by_comparison[name] = {
            recording: (
                "positive"
                if value > 0.0
                else "negative"
                if value < 0.0
                else "zero"
            )
            for recording, value in recording_effects.items()
        }
    method_recall = {
        method: float(
            np.mean(
                [
                    float(row["source_on_recovery"]["recall"])
                    for row in rows
                    if row["method"] == method
                ]
            )
        )
        for method in sorted({str(row["method"]) for row in rows})
    }
    return {
        "primary_method": JEPA_RESIDUAL_METHOD,
        "frozen_comparator": RAW_METHOD,
        "method_macro_recall": method_recall,
        "paired_residual_minus_raw_hierarchical_bootstrap": bootstraps[
            "jepa_residual_minus_raw"
        ],
        "paired_random_minus_raw_hierarchical_bootstrap": bootstraps[
            "random_residual_minus_raw"
        ],
        "paired_jepa_minus_random_hierarchical_bootstrap": bootstraps[
            "jepa_residual_minus_random_residual"
        ],
        "recording_level_effects": recording_effects_by_comparison[
            "jepa_residual_minus_raw"
        ],
        "recording_level_effects_by_comparison": recording_effects_by_comparison,
        "recording_level_directions_by_comparison": recording_directions_by_comparison,
        "source_count_aggregated_cluster_rows": cluster_rows,
        "nested_not_resampled_as_independent": ["source_count"],
        "descriptive_screen_only": True,
        "formal_advancement_gate": False,
        "claim_promotion_allowed": False,
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_json_dict") and callable(value.to_json_dict):
        return _jsonable(value.to_json_dict())
    if hasattr(value, "summary") and callable(value.summary):
        return _jsonable(value.summary())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _decoder_training_config(decoder_module: Any, values: Mapping[str, Any], steps: int) -> Any:
    cls = decoder_module.DecoderTrainingConfig
    signature = inspect.signature(cls)
    aliases = {
        "steps": steps,
        "batch_size": int(values["batch_size"]),
        "learning_rate": float(values["learning_rate"]),
        "weight_decay": float(values["weight_decay"]),
        "adamw_betas": tuple(float(value) for value in values["adamw_betas"]),
        "betas": tuple(float(value) for value in values["adamw_betas"]),
        "adamw_epsilon": float(values["adamw_epsilon"]),
        "epsilon": float(values["adamw_epsilon"]),
        "gradient_clip_norm": float(values["gradient_clip_norm"]),
        "smooth_l1_beta": float(values.get("loss_beta", 1.0)),
        "loss_beta": float(values.get("loss_beta", 1.0)),
        "log_interval": int(values.get("log_interval", max(1, min(100, steps)))),
        "amp": bool(values.get("amp", True)),
    }
    kwargs = {name: aliases[name] for name in signature.parameters if name in aliases}
    return cls(**kwargs)


def _provider_module(provider: Any) -> nn.Module:
    for candidate in (provider, getattr(provider, "model", None), getattr(provider, "module", None)):
        if isinstance(candidate, nn.Module):
            return candidate
    raise JEPAResidualPilotError("loaded frozen provider does not expose an nn.Module")


def _decoder_module_only(model: nn.Module) -> nn.Module:
    decoder = getattr(model, "decoder", None)
    if not isinstance(decoder, nn.Module):
        decoder = getattr(model, "pixel_decoder", None)
    return decoder if isinstance(decoder, nn.Module) else model


def _require_complete_native_raw_anchor(
    raw_anchor_validation: Mapping[str, Any] | None,
) -> None:
    if (
        raw_anchor_validation is None
        or raw_anchor_validation.get(
            "exact_recovery_result_and_recall_identity"
        )
        is not True
        or int(raw_anchor_validation.get("recovery_result_object_count", -1))
        != 216
        or raw_anchor_validation.get("anchor_execution_phase")
        != "after_fixture_reconstruction_before_decoder_construction_or_training"
    ):
        raise JEPAResidualPilotError(
            "learned model construction and decoder training are blocked until "
            "the complete native raw-HC pretraining anchor passes"
        )


def _construct_decoder_models(
    decoder_module: Any,
    *,
    provider: Any,
    random_provider: Any,
    decoder_seed: int,
    device: str,
    raw_anchor_validation: Mapping[str, Any] | None,
) -> tuple[nn.Module, nn.Module]:
    """Construct learned arms only behind the complete native-anchor gate."""

    _require_complete_native_raw_anchor(raw_anchor_validation)
    model = decoder_module.FrozenLatentPixelBackgroundModel(
        provider.provider,
        decoder_initialization_seed=int(decoder_seed),
    ).to(device)
    random_model = decoder_module.FrozenLatentPixelBackgroundModel(
        random_provider.provider,
        decoder_initialization_seed=int(decoder_seed),
    ).to(device)
    return model, random_model


def _guarded_train_decoder_arms(
    decoder_module: Any,
    *,
    model: nn.Module,
    random_model: nn.Module,
    training_tensor: Tensor,
    training_contract: Any,
    seed: int,
    device: str,
    geometry_validation: Mapping[str, Any],
    raw_anchor_validation: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Make raw-anchor and geometry gates precede the first optimizer step."""

    if geometry_validation.get("passed_before_first_optimizer_step") is not True:
        raise JEPAResidualPilotError(
            "decoder training is blocked until all geometry checks pass"
        )
    _require_complete_native_raw_anchor(raw_anchor_validation)
    first = decoder_module.train_pixel_decoder(
        model,
        training_tensor,
        config=training_contract,
        seed=int(seed),
        device=device,
    )
    second = decoder_module.train_pixel_decoder(
        random_model,
        training_tensor,
        config=training_contract,
        seed=int(seed),
        device=device,
    )
    return first, second


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _decoder_checkpoint_identity(
    *,
    jepa_checkpoint_tensor_sha256: str,
    jepa_runtime_provider_sha256: str,
    random_checkpoint_tensor_sha256: str,
    random_runtime_provider_sha256: str,
    jepa_decoder_state_sha256: str,
    random_decoder_state_sha256: str,
) -> dict[str, Any]:
    """Describe exact digest algorithms, scopes, and both frozen providers."""

    return {
        "hash_contract": {
            "serialized_checkpoint_file": {
                "algorithm": "sha256",
                "scope": "complete_serialized_torch_checkpoint_file_bytes",
            },
            "parent_checkpoint_tensor_mapping": {
                "algorithm": "sha256",
                "scope": "sorted_state_dict_keys_then_name_utf8_nul_dtype_ascii_shape_numpy_int64_bytes_and_contiguous_tensor_c_order_bytes",
                "implementation": "jepa_residual_pilot_v1_1.tensor_mapping_sha256",
            },
            "runtime_module_state": {
                "algorithm": "sha256",
                "scope": "sorted_state_dict_items_then_name_utf8_dtype_ascii_shape_tuple_ascii_and_contiguous_uint8_tensor_bytes",
                "implementation": "jepa_background_residual.module_state_sha256",
            },
        },
        "frozen_provider_identity": {
            "jepa_checkpoint_tensor_mapping_sha256": jepa_checkpoint_tensor_sha256,
            "jepa_runtime_provider_module_state_sha256": jepa_runtime_provider_sha256,
            "random_checkpoint_tensor_mapping_sha256": random_checkpoint_tensor_sha256,
            "random_runtime_provider_module_state_sha256": random_runtime_provider_sha256,
        },
        "decoder_state_identity": {
            "jepa_arm_runtime_module_state_sha256": jepa_decoder_state_sha256,
            "random_arm_runtime_module_state_sha256": random_decoder_state_sha256,
        },
    }


def _prediction_numpy(output: Any) -> tuple[np.ndarray, dict[str, Any]]:
    background = getattr(output, "background", None)
    coverage = getattr(output, "coverage_counts", None)
    if not isinstance(background, Tensor) or not isinstance(coverage, Tensor):
        raise JEPAResidualPilotError("background sweep output lacks tensors")
    expected_shape = STRICT_PREDICTION_SHAPE_BCTHW
    background_shape = tuple(int(value) for value in background.shape)
    coverage_shape = tuple(int(value) for value in coverage.shape)
    if background_shape != expected_shape:
        raise JEPAResidualPilotError(
            "background sweep shape mismatch: "
            f"expected={expected_shape}, observed={background_shape}"
        )
    if coverage_shape != expected_shape:
        raise JEPAResidualPilotError(
            "coverage sweep shape mismatch: "
            f"expected={expected_shape}, observed={coverage_shape}"
        )
    if background_shape != coverage_shape:
        raise JEPAResidualPilotError(
            "background and coverage sweep shapes do not match exactly"
        )
    background_finite = bool(torch.isfinite(background).all())
    if not background_finite:
        raise JEPAResidualPilotError("background sweep contains nonfinite values")
    counts = coverage.detach().cpu()
    coverage_exactly_once = bool((counts == 1).all())
    if not coverage_exactly_once:
        raise JEPAResidualPilotError("blind-tube sweep must cover every target exactly once")
    manifest = _jsonable(getattr(output, "manifest", {}))
    return (
        background[0, 0].detach().cpu().numpy().astype(np.float32, copy=False),
        {
            "expected_background_shape_bcthw": list(expected_shape),
            "observed_background_shape_bcthw": list(background_shape),
            "expected_coverage_shape_bcthw": list(expected_shape),
            "observed_coverage_shape_bcthw": list(coverage_shape),
            "background_shape_exact": background_shape == expected_shape,
            "coverage_shape_exact": coverage_shape == expected_shape,
            "background_coverage_shapes_exact_match": background_shape
            == coverage_shape,
            "background_all_finite": background_finite,
            "coverage_exactly_once": coverage_exactly_once,
            "prediction_contract_passed": True,
            "minimum_coverage": int(counts.min().item()),
            "maximum_coverage": int(counts.max().item()),
            "coverage_sha256": tensor_mapping_sha256({"coverage": counts}),
            "manifest": manifest,
        },
    )


def _predict(
    decoder_module: Any,
    model: nn.Module,
    movie_normalized: np.ndarray,
    *,
    device: str,
    target_batch_size: int,
    amp: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = (
        torch.from_numpy(np.ascontiguousarray(movie_normalized))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output = decoder_module.predict_full_background(
            model,
            tensor,
            target_batch_size=int(target_batch_size),
            amp=bool(amp),
        )
    return _prediction_numpy(output)


def _audit_placeholders(root: Path, *, mode: str, fixture_count: int) -> dict[str, Any]:
    sections = {
        "1_Expert_Annotations": (
            "# Exact injection truth / expert-section boundary\n\n"
            "No human expert annotations are used. Exact generated source media are pending.\n"
        ),
        "2_Model_Annotations": (
            "# Conditional-background residual candidates\n\n"
            "Frozen candidate maps, close-ups, and full-duration traces are pending. Native unmatched candidates remain unknown.\n"
        ),
        "3_Comparison": (
            "# Raw versus conditional residual\n\n"
            "Numeric paired results are available; required figure/table-only audit renderings are pending.\n"
        ),
    }
    for directory, readme in sections.items():
        atomic_text(root / directory / "README.md", readme)
    status = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mode": mode,
        "numeric_fixture_count": int(fixture_count),
        "required_media_complete": False,
        "inventory_validation_passed": False,
        "scientific_audit_complete": False,
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
        "promotion_blocked": True,
        "missing_required_families": [
            "truth_only_full_field_videos",
            "truth_closeups_and_full_duration_traces",
            "conditional_residual_full_field_videos",
            "model_closeups_and_full_duration_traces",
            "figure_table_only_matched_comparison",
            "media_decode_and_annotation_pixel_validation",
        ],
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


def _validate_required_outputs(
    root: Path, required_outputs: Sequence[str]
) -> dict[str, Any]:
    """Require every configured output and its byte-identical index entry."""

    if not required_outputs:
        raise JEPAResidualPilotError("required output contract must be nonempty")
    logicals = [str(value) for value in required_outputs]
    if len(logicals) != len(set(logicals)):
        raise JEPAResidualPilotError("required output contract contains duplicates")
    missing: list[str] = []
    for logical in logicals:
        path = _relative_child(root, logical, field="required output")
        if not path.is_file() or path.is_symlink():
            missing.append(logical)
    if missing:
        raise JEPAResidualPilotError(
            "required output contract is incomplete: " + ", ".join(missing)
        )
    index_path = root / "artifact_index.json"
    persisted_index = _json_object(index_path)
    recomputed_index = _artifact_index(root)
    if persisted_index != recomputed_index:
        raise JEPAResidualPilotError(
            "persisted artifact index differs from the complete live output tree"
        )
    index_rows = _artifact_index_rows(persisted_index)
    indexed = {str(row["path"]): row for row in index_rows}
    actual_non_index = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_index.json"
    }
    unindexed_existing = sorted(actual_non_index - set(indexed))
    stale_index_entries = sorted(set(indexed) - actual_non_index)
    full_index_drift: list[str] = []
    for logical, row in indexed.items():
        path = _relative_child(root, logical, field="artifact index path")
        if (
            not path.is_file()
            or path.is_symlink()
            or int(row["bytes"]) != int(path.stat().st_size)
            or str(row["sha256"]) != sha256_file(path)
        ):
            full_index_drift.append(logical)
    if unindexed_existing or stale_index_entries or full_index_drift:
        details = []
        if unindexed_existing:
            details.append("unindexed_existing=" + ",".join(unindexed_existing))
        if stale_index_entries:
            details.append("stale_index=" + ",".join(stale_index_entries))
        if full_index_drift:
            details.append("full_index_drift=" + ",".join(full_index_drift))
        raise JEPAResidualPilotError(
            "full artifact-index coverage validation failed: " + "; ".join(details)
        )
    unindexed: list[str] = []
    index_drift: list[str] = []
    for logical in logicals:
        if logical == "artifact_index.json":
            continue
        row = indexed.get(logical)
        if row is None:
            unindexed.append(logical)
            continue
        path = _relative_child(root, logical, field="required output")
        if int(row["bytes"]) != int(path.stat().st_size) or str(row["sha256"]) != sha256_file(path):
            index_drift.append(logical)
    if unindexed or index_drift:
        details = []
        if unindexed:
            details.append("unindexed=" + ",".join(unindexed))
        if index_drift:
            details.append("index_drift=" + ",".join(index_drift))
        raise JEPAResidualPilotError(
            "required output artifact-index validation failed: " + "; ".join(details)
        )
    return {
        "required_output_count": len(logicals),
        "all_required_outputs_exist": True,
        "all_required_non_index_outputs_are_byte_identical_to_index": True,
        "all_existing_non_index_files_are_byte_identical_to_index": True,
    }


def _proposal_count_summary(
    rows: Sequence[Mapping[str, Any]], *, proposal_cap: int
) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if str(row["method"]) == method]
        source_on = [int(row["source_on_recovery"]["candidate_count"]) for row in selected]
        intervention = [
            int(row["intervention_recovery"]["candidate_count"]) for row in selected
        ]
        by_method[method] = {
            "fixture_count": len(selected),
            "source_on_candidate_count_total": sum(source_on),
            "source_on_candidate_count_minimum": min(source_on),
            "source_on_candidate_count_maximum": max(source_on),
            "intervention_candidate_count_total": sum(intervention),
            "intervention_candidate_count_minimum": min(intervention),
            "intervention_candidate_count_maximum": max(intervention),
            "all_counts_within_proposal_cap": max(source_on + intervention)
            <= int(proposal_cap),
        }
    return {
        "proposal_cap_per_fixture_arm": int(proposal_cap),
        "candidate_fill_to_cap": False,
        "by_method": by_method,
        "all_counts_within_proposal_cap": all(
            row["all_counts_within_proposal_cap"] for row in by_method.values()
        ),
    }


def _algebra_integrity_summary(
    diagnostics: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if not diagnostics:
        raise JEPAResidualPilotError("residual algebra integrity requires diagnostics")
    projection_limit = float(thresholds["maximum_projection_closure_absolute_error"])
    voxel_limit = float(
        thresholds["maximum_voxelwise_pair_closure_absolute_error"]
    )
    by_method: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in diagnostics}):
        selected = [row for row in diagnostics if str(row["method"]) == method]
        by_method[method] = {
            "maximum_projection_closure_absolute_error": float(
                max(
                    abs(
                        float(
                            row["algebra"][
                                "retention_plus_absorption_minus_one"
                            ]
                        )
                    )
                    for row in selected
                )
            ),
            "maximum_voxelwise_pair_closure_absolute_error": float(
                max(float(row["algebra"]["pair_closure_max_abs"]) for row in selected)
            ),
        }
    projection_max = max(
        row["maximum_projection_closure_absolute_error"] for row in by_method.values()
    )
    voxel_max = max(
        row["maximum_voxelwise_pair_closure_absolute_error"]
        for row in by_method.values()
    )
    return {
        "maximum_projection_closure_absolute_error": float(projection_max),
        "maximum_voxelwise_pair_closure_absolute_error": float(voxel_max),
        "maximum_projection_closure_absolute_error_allowed": projection_limit,
        "maximum_voxelwise_pair_closure_absolute_error_allowed": voxel_limit,
        "projection_closure_passed": projection_max <= projection_limit,
        "voxelwise_pair_closure_passed": voxel_max <= voxel_limit,
        "by_method": by_method,
        "distinct_from_fixture_injection_additive_closure_ulp_check": True,
    }


def _heartbeat(root: Path, phase: str, **details: Any) -> None:
    atomic_json(
        root / "heartbeat.json",
        {"schema_version": SCHEMA_VERSION, "phase": phase, "updated_at": _utc_now(), **details},
    )


def _report(summary: Mapping[str, Any]) -> str:
    primary = summary["primary_evaluation"]
    recall = primary["method_macro_recall"]
    effect = primary["paired_residual_minus_raw_hierarchical_bootstrap"]
    random_effect = primary["paired_random_minus_raw_hierarchical_bootstrap"]
    specificity_effect = primary["paired_jepa_minus_random_hierarchical_bootstrap"]
    paired = summary["paired_injection"]
    diagnostics = summary["diagnostics"]
    integrity = diagnostics["engineering_algebra_integrity"]
    proposals = summary["proposal_counts"]
    directions = primary["recording_level_directions_by_comparison"]
    hashes = summary["hashes"]
    implementation_hashes = hashes["implementation_sources"]
    provenance_hash_lines = "\n".join(
        f"- `{key}`: `{value}`"
        for key, value in hashes.items()
        if key != "implementation_sources"
    )
    implementation_hash_lines = "\n".join(
        f"- `{path}`: `{digest}`"
        for path, digest in sorted(implementation_hashes.items())
    )
    proposal_lines = "\n".join(
        f"- `{method}`: source-on `{values['source_on_candidate_count_total']}`, "
        f"intervention `{values['intervention_candidate_count_total']}` candidates "
        f"across `{values['fixture_count']}` fixtures."
        for method, values in proposals["by_method"].items()
    )
    return (
        "# Conditional JEPA pixel-residual screen\n\n"
        f"This `{summary['execution']['mode']}` run reused the hash-verified EXP-0028 Screen-B seed-1001 checkpoint and normalized clip bank, trained only a {summary['decoder']['trainable_parameters']}-parameter pixel decoder, and evaluated `{paired['observed_fixture_count']}` of `{paired['registered_exact_fixture_count']}` exact source-on/source-off fixtures containing `{paired['exact_injected_source_occurrences']}` of `{paired['registered_exact_injected_source_occurrences']}` registered source occurrences.\n\n"
        "## Paired recovery results\n\n"
        f"Source-on macro recall was `{recall.get(JEPA_RESIDUAL_METHOD)}` for the JEPA conditional residual, `{recall.get(RANDOM_RESIDUAL_METHOD)}` for the matched random-provider residual, and `{recall.get(RAW_METHOD)}` for the parent-matched handcrafted comparator evaluated in native raw units. The paired JEPA-residual-minus-raw effect was `{effect['observed_mean']}` with a hierarchical 95% interval of `{effect['confidence_interval_95']}`; random-residual-minus-raw was `{random_effect['observed_mean']}` with `{random_effect['confidence_interval_95']}`, and JEPA-residual-minus-random was `{specificity_effect['observed_mean']}` with `{specificity_effect['confidence_interval_95']}`. Source counts were averaged within recording/window/injection-seed clusters before resampling.\n\n"
        f"Per-recording directions for all three comparisons were `{json.dumps(directions, sort_keys=True)}`. Before any learned model was loaded or constructed, the native raw-HC anchor matched EXP-0028 exactly for all `{summary['raw_hc_cross_run_regression']['execution_fixture_count']}` registered anchor fixtures (216 source-on/intervention RecoveryResult objects); the joined residual evaluation used `{summary['paired_injection']['observed_fixture_count']}` fixtures in this execution mode.\n\n"
        f"The proposal cap was `{proposals['proposal_cap_per_fixture_arm']}` per fixture/arm; observed proposal totals were:\n\n{proposal_lines}\n\n"
        "## Residual integrity and signal behavior\n\n"
        f"For the JEPA arm, mean total signal error ratio was `{diagnostics['mean_total_signal_error_ratio']}` and mean orthogonal distortion ratio was `{diagnostics['mean_orthogonal_distortion_ratio']}`. Across both residual arms, maximum projection closure error was `{integrity['maximum_projection_closure_absolute_error']}` (limit `{integrity['maximum_projection_closure_absolute_error_allowed']}`), and maximum voxelwise pair closure error was `{integrity['maximum_voxelwise_pair_closure_absolute_error']}` (limit `{integrity['maximum_voxelwise_pair_closure_absolute_error_allowed']}`). These checks are distinct from the registered fixture-additivity ULP check.\n\n"
        "The residual is a signed diagnostic, not a neuron probability. Signal retention, predictor absorption, total and orthogonal distortion, background RMS/MAD, dynamic MAD, seams, and exact tiled coverage must be interpreted together.\n\n"
        "## Frozen seeds and hashes\n\n"
        f"Seeds: `{json.dumps(summary['seeds'], sort_keys=True)}`.\n\n"
        f"{provenance_hash_lines}\n\nImplementation source hashes:\n\n{implementation_hash_lines}\n\n"
        "## Evidence boundary\n\n"
        f"The scientific audit is complete: `{summary['scientific_audit']['scientific_audit_complete']}`. Motion dependency NREV-EXP-0025 is satisfied: `{summary['motion_dependency']['satisfied']}`. Therefore nuisance-robust interpretation, biological neuron identity, precision/specificity, independent-animal generalization, scientific denoising benefit, causality, reinforcement learning, scientific completion, and promotion remain unresolved or forbidden. This output is a bounded, non-claim-bearing engineering screen.\n"
    )


def _descriptive_advancement_panel(
    grouped: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen engineering heuristics without turning them into gates."""

    primary_diagnostics = [
        row for row in diagnostics if row.get("method") == JEPA_RESIDUAL_METHOD
    ]
    if not primary_diagnostics:
        raise JEPAResidualPilotError("descriptive panel lacks JEPA residual diagnostics")
    bootstrap = grouped["paired_residual_minus_raw_hierarchical_bootstrap"]
    random_bootstrap = grouped["paired_random_minus_raw_hierarchical_bootstrap"]
    specificity_bootstrap = grouped[
        "paired_jepa_minus_random_hierarchical_bootstrap"
    ]
    recall_effect = float(bootstrap["observed_mean"])
    raw_lower_value = bootstrap["confidence_interval_95"][0]
    specificity_lower_value = specificity_bootstrap["confidence_interval_95"][0]
    lower = (
        float(raw_lower_value)
        if raw_lower_value is not None and math.isfinite(float(raw_lower_value))
        else None
    )
    specificity_lower = (
        float(specificity_lower_value)
        if specificity_lower_value is not None
        and math.isfinite(float(specificity_lower_value))
        else None
    )
    median_retention = float(
        np.median([row["algebra"]["retained_gain_projection"] for row in primary_diagnostics])
    )
    median_absorption = float(
        np.median(
            [row["algebra"]["predictor_absorption_projection"] for row in primary_diagnostics]
        )
    )
    median_rms_ratio = float(
        np.median([row["background"]["background_rms_ratio"] for row in primary_diagnostics])
    )
    maximum_closure = float(
        max(abs(row["algebra"]["retention_plus_absorption_minus_one"]) for row in primary_diagnostics)
    )
    maximum_voxelwise_closure = float(
        max(row["algebra"]["pair_closure_max_abs"] for row in primary_diagnostics)
    )
    recording_nonnegative = all(
        float(value) >= 0.0 for value in grouped["recording_level_effects"].values()
    )
    specificity_recording_effects = grouped["recording_level_effects_by_comparison"][
        "jepa_residual_minus_random_residual"
    ]
    checks = {
        "recall_effect_at_least_descriptive_minimum": recall_effect
        >= float(thresholds["jepa_residual_minus_raw_macro_recall_min"]),
        "bootstrap_lower_bound_strictly_positive": lower is not None
        and lower > float(thresholds["grouped_bootstrap_lower_bound_min_exclusive"]),
        "median_retained_gain_at_least_descriptive_minimum": median_retention
        >= float(thresholds["minimum_median_aligned_retained_gain"]),
        "median_absorption_at_most_descriptive_maximum": median_absorption
        <= float(thresholds["maximum_median_predictor_absorption"]),
        "median_background_rms_ratio_at_most_descriptive_maximum": median_rms_ratio
        <= float(thresholds["maximum_median_background_rms_ratio"]),
        "recording_effect_nonnegative_each_recording": recording_nonnegative,
        "jepa_minus_random_observed_strictly_above_descriptive_minimum": float(
            specificity_bootstrap["observed_mean"]
        )
        > float(thresholds["jepa_residual_minus_random_observed_mean_min_exclusive"]),
        "jepa_minus_random_interval_lower_bound_strictly_above_preferred_minimum": specificity_lower
        is not None
        and specificity_lower
        > float(
            thresholds[
                "jepa_residual_minus_random_grouped_interval_lower_bound_preferred_exclusive"
            ]
        ),
    }
    return {
        "status": "descriptive_engineering_triage_only_not_formal_gate",
        "observed": {
            "jepa_residual_minus_raw_macro_recall": recall_effect,
            "grouped_bootstrap_lower_bound": lower,
            "grouped_bootstrap_lower_bound_status": (
                "available" if lower is not None else "not_evaluated_requires_two_recordings"
            ),
            "median_aligned_retained_gain": median_retention,
            "median_predictor_absorption": median_absorption,
            "median_background_rms_ratio": median_rms_ratio,
            "maximum_projection_closure_absolute_error": maximum_closure,
            "maximum_voxelwise_pair_closure_absolute_error": maximum_voxelwise_closure,
            "recording_effect_nonnegative_each_recording": recording_nonnegative,
        },
        "threshold_crossings": checks,
        "all_descriptive_thresholds_crossed": all(checks.values()),
        "formal_claim_consequence": "none",
        "scientific_promotion_allowed": False,
        "specificity_control": {
            "random_residual_minus_raw_observed": float(
                random_bootstrap["observed_mean"]
            ),
            "random_residual_minus_raw_confidence_interval_95": random_bootstrap[
                "confidence_interval_95"
            ],
            "jepa_residual_minus_random_residual_observed": float(
                specificity_bootstrap["observed_mean"]
            ),
            "jepa_residual_minus_random_residual_confidence_interval_95": specificity_bootstrap[
                "confidence_interval_95"
            ],
            "jepa_residual_minus_random_interval_lower_bound_status": (
                "available"
                if specificity_lower is not None
                else "not_evaluated_requires_two_recordings"
            ),
            "jepa_residual_minus_random_recording_effects": specificity_recording_effects,
            "jepa_outperforms_random_observed": float(
                specificity_bootstrap["observed_mean"]
            )
            > 0.0,
            "enters_primary_formal_gate": False,
        },
    }


def run_jepa_residual_pilot(
    config_path: Path,
    *,
    repository_root: Path,
    mode: str = "smoke",
    data_root: Path | None = None,
    output_root: Path | None = None,
    operator_process_check_acknowledged: bool = False,
) -> dict[str, Any]:
    """Execute a config-bounded non-scientific residual smoke or screen."""

    if mode not in SUPPORTED_MODES:
        raise JEPAResidualPilotError(f"mode must be one of {sorted(SUPPORTED_MODES)}")
    _guard_output_override(mode, output_root)
    if mode == "preflight":
        return preflight_jepa_residual_pilot(
            config_path,
            repository_root=repository_root,
            mode=mode,
            data_root=data_root,
            output_root=output_root,
            operator_process_check_acknowledged=operator_process_check_acknowledged,
        )
    repository = Path(repository_root).expanduser().resolve()
    runtime_data = (
        configured_data_root(repository)
        if data_root is None
        else Path(data_root).expanduser().resolve()
    )
    config_file = Path(config_path).expanduser().resolve()
    config = load_jepa_residual_config(config_file)
    requested_output = (
        _resolve_repo_path(config["output_root"], repository)
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    output, partial = guard_output_collision(requested_output)
    preflight = preflight_jepa_residual_pilot(
        config_file,
        repository_root=repository,
        mode=mode,
        data_root=runtime_data,
        output_root=None if mode == "screen" else output,
        operator_process_check_acknowledged=operator_process_check_acknowledged,
    )
    if preflight["failed_checks"]:
        raise JEPAResidualPilotError(
            "preflight failed: " + ", ".join(preflight["failed_checks"])
        )
    parent_root, paths = _pinned_parent_paths(config, repository)
    parent_before = preflight["parent_integrity"]
    mode_config = config["execution_modes"][mode]
    decoder_config = config["_decoder"]
    requested_device = str(mode_config["device"])
    started_at = _utc_now()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(config["_resources"]["cpu_threads"]))
    partial.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(partial / "preflight.json", preflight)
        atomic_json(
            partial / "resolved_config.json",
            {
                "schema_version": SCHEMA_VERSION,
                "config": portableize_paths(
                    {
                        key: value
                        for key, value in config.items()
                        if not key.startswith("_")
                    },
                    repository=repository,
                    data=runtime_data,
                ),
                "config_sha256": sha256_file(config_file),
                "execution": {
                    "mode": mode,
                    "device": requested_device,
                    "scientific_execution": False,
                    "scientific_completion": False,
                    "scientific_promotion_allowed": False,
                    "mode_budget": mode_config,
                    "authorization": preflight["authorization"],
                    "provenance": preflight["execution_provenance"],
                },
            },
        )
        _heartbeat(partial, "pretraining_native_raw_anchor")
        backgrounds, background_manifest = reconstruct_frozen_backgrounds(
            config,
            repository_root=repository,
            runtime_data_root=runtime_data,
            paths=paths,
            verify_source_hashes=bool(mode_config.get("verify_source_hashes", True)),
        )
        expected_fixture_manifest = _json_object(paths["paired_injection_manifest"])
        parent_config = _parent_registered_config(paths)
        (
            fixtures,
            fixture_metadata,
            registered_fixture_summary,
            fixture_identity,
        ) = reconstruct_exact_fixtures(parent_config, backgrounds, expected_fixture_manifest)
        # Fail closed on the complete parent-native raw-HC anchor before a
        # decoder is even constructed, and therefore before either optimizer
        # can take its first step.  Later evaluation reuses these exact rows.
        evaluation = config["_evaluation"]
        (
            all_native_raw_rows,
            raw_anchor,
            raw_pretraining_calibration_evidence,
        ) = _raw_hc_pretraining_anchor(
            fixtures,
            fixture_metadata,
            evaluation=evaluation,
            parent_payload=_json_object(paths["paired_injection_results"]),
            parent_run_id=str(config["upstream_screen"]["run_id"]),
            current_run_id=str(config["run_id"]),
        )
        raw_anchor["parent_paired_results_sha256"] = sha256_file(
            paths["paired_injection_results"]
        )
        atomic_json(partial / "raw_hc_cross_run_regression.json", raw_anchor)
        _require_complete_native_raw_anchor(raw_anchor)
        all_native_raw_by_fixture = {
            str(row["fixture_id"]): row for row in all_native_raw_rows
        }
        if len(all_native_raw_by_fixture) != STRICT_SCREEN_FIXTURES:
            raise JEPAResidualPilotError(
                "native raw-HC pretraining rows do not have 108 unique fixture IDs"
            )
        geometry_leakage = footprint_halo_leakage_diagnostics(
            fixtures,
            patch_yx=config["blind_tube_geometry"]["written_target_pixels_t_y_x"][-2:],
        )
        geometry_expected = config["blind_tube_geometry"]["geometry_preflight"]
        geometry_tolerance = float(geometry_expected["absolute_tolerance"])
        observed_mass = geometry_leakage["mass_fraction_outside_halo"]["maximum"]
        observed_energy = geometry_leakage["energy_fraction_outside_halo"]["maximum"]
        geometry_leakage["registered_exact_maxima"] = {
            "mass": float(
                geometry_expected[
                    "worst_registered_injected_footprint_mass_outside_halo_max"
                ]
            ),
            "energy": float(
                geometry_expected[
                    "worst_registered_injected_footprint_energy_outside_halo_max"
                ]
            ),
            "absolute_tolerance": geometry_tolerance,
        }
        geometry_leakage["matches_registered_exact_maxima"] = bool(
            abs(observed_mass - geometry_leakage["registered_exact_maxima"]["mass"])
            <= geometry_tolerance
            and abs(observed_energy - geometry_leakage["registered_exact_maxima"]["energy"])
            <= geometry_tolerance
            and geometry_leakage["source_count"] == STRICT_SCREEN_SOURCES
        )
        if not geometry_leakage["matches_registered_exact_maxima"]:
            raise JEPAResidualPilotError(
                "registered footprint blind-halo leakage recomputation drifted"
            )
        if mode == "smoke":
            background_limit = int(mode_config["background_window_count"])
            cell_limit = int(mode_config["fixture_cells_per_background"])
            allowed_windows = {
                str(item["metadata"]["background_window_id"])
                for item in backgrounds[:background_limit]
            }
            fixtures = [
                fixture
                for fixture in fixtures
                if str(fixture.metadata["background_window_id"]) in allowed_windows
            ][:cell_limit]
        fixture_summary = _execution_fixture_summary(fixtures, registered_fixture_summary)
        atomic_json(partial / "background_window_manifest.json", background_manifest)
        atomic_json(
            partial / "paired_injection_manifest.json",
            {
                **expected_fixture_manifest,
                "execution_fixture_ids": [fixture.fixture_id for fixture in fixtures],
                "execution_fixture_count": len(fixtures),
                "exact_parent_identity": fixture_identity,
            },
        )

        # Only after the complete 216-object native raw-HC anchor passes may
        # this amendment load or construct either learned provider/decoder.
        _heartbeat(partial, "loading_verified_parent_after_raw_anchor")
        checkpoint_payload = _load_strict_checkpoint_payload(
            paths["checkpoint"],
            expected_sha256=config["upstream_screen"]["checkpoint"]["sha256"],
        )
        from . import jepa_background_residual as decoder_module

        provider = decoder_module.load_frozen_jepa_checkpoint(
            paths["checkpoint"],
            expected_sha256=config["upstream_screen"]["checkpoint"]["sha256"],
            map_location="cpu",
        )
        provider_nn = _provider_module(provider)
        frozen_hash_before = decoder_module.module_state_sha256(provider_nn)
        expected_tensor_hash = checkpoint_payload["_jepa_tensor_sha256"]
        checkpoint_model = getattr(provider, "model", None)
        if not isinstance(checkpoint_model, nn.Module):
            raise JEPAResidualPilotError(
                "loaded provider lacks its strict checkpoint module"
            )
        if tensor_mapping_sha256(checkpoint_model.state_dict()) != expected_tensor_hash:
            raise JEPAResidualPilotError(
                "loaded JEPA weights differ from checkpoint state"
            )
        if any(parameter.requires_grad for parameter in provider_nn.parameters()):
            raise JEPAResidualPilotError("loaded JEPA provider has trainable weights")
        random_provider = decoder_module.load_frozen_random_checkpoint(
            paths["checkpoint"],
            expected_sha256=config["upstream_screen"]["checkpoint"]["sha256"],
            map_location="cpu",
        )
        random_provider_nn = _provider_module(random_provider)
        random_hash_before = decoder_module.module_state_sha256(random_provider_nn)
        random_checkpoint_model = getattr(random_provider, "model", None)
        if not isinstance(random_checkpoint_model, nn.Module) or tensor_mapping_sha256(
            random_checkpoint_model.state_dict()
        ) != checkpoint_payload["_random_tensor_sha256"]:
            raise JEPAResidualPilotError(
                "loaded random-control weights differ from checkpoint"
            )
        if any(parameter.requires_grad for parameter in random_provider_nn.parameters()):
            raise JEPAResidualPilotError(
                "loaded random provider has trainable weights"
            )

        normalization_check = _normalization_contract_check(config, paths)
        normalization = normalization_from_config(config)
        train_cache, train_cache_manifest = load_normalized_parent_cache(
            paths["train_cache"],
            expected_sha256=config["upstream_screen"]["train_cache"]["sha256"],
            expected_shape=config["upstream_screen"]["train_cache"]["shape_n_t_y_x"],
            expected_dtype=config["upstream_screen"]["train_cache"]["dtype"],
        )
        validation_cache, validation_cache_manifest = load_normalized_parent_cache(
            paths["validation_cache"],
            expected_sha256=config["upstream_screen"]["validation_cache"]["sha256"],
            expected_shape=config["upstream_screen"]["validation_cache"][
                "shape_n_t_y_x"
            ],
            expected_dtype=config["upstream_screen"]["validation_cache"]["dtype"],
        )
        train_count = int(mode_config["train_clip_count"])
        validation_count = int(mode_config["validation_clip_count"])
        training_tensor = torch.from_numpy(train_cache[:train_count]).unsqueeze(1)
        # Read-only validation evidence; the decoder trainer never receives it.
        validation_view = validation_cache[:validation_count]
        atomic_json(
            partial / "clip_reuse_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "train": train_cache_manifest,
                "validation": validation_cache_manifest,
                "train_clips_consumed": train_count,
                "validation_clips_reserved_not_trained_on": validation_count,
                "validation_first_clip_sha256": hashlib.sha256(
                    np.ascontiguousarray(validation_view[0]).tobytes()
                ).hexdigest(),
                "normalization": normalization_check,
            },
        )

        model, random_model = _construct_decoder_models(
            decoder_module,
            provider=provider,
            random_provider=random_provider,
            decoder_seed=int(decoder_config["seed"]),
            device=requested_device,
            raw_anchor_validation=raw_anchor,
        )
        initial_decoder_hash = decoder_module.module_state_sha256(
            _decoder_module_only(model)
        )
        random_initial_decoder_hash = decoder_module.module_state_sha256(
            _decoder_module_only(random_model)
        )
        if initial_decoder_hash != random_initial_decoder_hash:
            raise JEPAResidualPilotError(
                "provider arms did not start from an identical pixel decoder"
            )
        invariance_clip = (
            torch.from_numpy(np.ascontiguousarray(train_cache[0]))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(requested_device)
        )
        invariance_targets = _production_invariance_targets(
            config["blind_tube_geometry"]["target_tile_grid_y_x"]
        )
        expected_target_count = int(
            geometry_expected["target_tube_adversarial_targets_required"]
        )
        if (
            len(invariance_targets) != expected_target_count
            or len(invariance_targets)
            != int(config["blind_tube_geometry"]["tile_count"])
        ):
            raise JEPAResidualPilotError(
                "blind-tube invariance target count differs from frozen geometry"
            )
        arm_geometry: dict[str, Any] = {}
        for arm_name, arm_model in (
            ("jepa", model),
            ("random_control", random_model),
        ):
            production = decoder_module.production_contract_preflight(
                arm_model, invariance_clip
            )
            invariance = [
                decoder_module.target_tube_invariance_preflight(
                    arm_model, invariance_clip, target_yx=target
                )
                for target in invariance_targets
            ]
            arm_geometry[arm_name] = {
                "production_contract": _jsonable(production),
                "target_tube_invariance": _jsonable(invariance),
                "passed": bool(production.get("passed"))
                and all(bool(row.get("passed")) for row in invariance),
                "all_reflected_aliases_masked": all(
                    bool(row["masked_provider_inputs_exactly_identical"])
                    for row in invariance
                ),
                "reflected_alias_mask_value_mismatch_count": sum(
                    not bool(row["masked_provider_inputs_exactly_identical"])
                    for row in invariance
                ),
            }
        geometry_validation = {
            "schema_version": SCHEMA_VERSION,
            "execution_order_contract": "completed_before_first_optimizer_step",
            "passed_before_first_optimizer_step": all(
                bool(value["passed"]) for value in arm_geometry.values()
            )
            and bool(geometry_leakage["matches_registered_exact_maxima"]),
            "tested_target_classes": [
                "all_64_tiles_including_corners_edges_near_edges_and_interior",
            ],
            "tested_target_count_per_provider_arm": len(invariance_targets),
            "required_target_count_per_provider_arm": expected_target_count,
            "provider_arms": arm_geometry,
            "registered_fixture_footprint_halo_leakage": geometry_leakage,
        }
        if not geometry_validation["passed_before_first_optimizer_step"]:
            raise JEPAResidualPilotError(
                "blind-tube geometry/invariance pretraining gate failed"
            )
        atomic_json(
            partial / "blind_tube_geometry_validation.json", geometry_validation
        )

        decoder_runtime_config = {
            **decoder_config,
            "amp": bool(decoder_config["amp"] and requested_device.startswith("cuda")),
        }
        training_contract = _decoder_training_config(
            decoder_module, decoder_runtime_config, int(mode_config["decoder_steps"])
        )
        _heartbeat(partial, "training_decoder", steps=int(mode_config["decoder_steps"]))
        training_result, random_training_result = _guarded_train_decoder_arms(
            decoder_module,
            model=model,
            random_model=random_model,
            training_tensor=training_tensor,
            training_contract=training_contract,
            seed=int(decoder_config["training_seed"]),
            device=requested_device,
            geometry_validation=geometry_validation,
            raw_anchor_validation=raw_anchor,
        )
        frozen_hash_after_training = decoder_module.module_state_sha256(provider_nn)
        if frozen_hash_after_training != frozen_hash_before:
            raise JEPAResidualPilotError("frozen JEPA weights changed while training decoder")
        decoder_only = _decoder_module_only(model)
        decoder_parameter_count = sum(parameter.numel() for parameter in decoder_only.parameters())
        if decoder_parameter_count != int(decoder_config["trainable_parameters"]):
            raise JEPAResidualPilotError("decoder parameter count differs from frozen config")
        if any(parameter.requires_grad for parameter in provider_nn.parameters()):
            raise JEPAResidualPilotError("JEPA weights became trainable")
        random_hash_after_training = decoder_module.module_state_sha256(
            random_provider_nn
        )
        if random_hash_after_training != random_hash_before:
            raise JEPAResidualPilotError(
                "frozen random-provider weights changed while training decoder"
            )
        random_decoder_only = _decoder_module_only(random_model)
        random_decoder_parameter_count = sum(
            parameter.numel() for parameter in random_decoder_only.parameters()
        )
        if random_decoder_parameter_count != decoder_parameter_count:
            raise JEPAResidualPilotError(
                "random-control decoder capacity differs from JEPA decoder"
            )
        training_summary = _jsonable(training_result.summary())
        random_training_summary = _jsonable(random_training_result.summary())
        logged_target_schedule_identical = [
            row["target_token_yx"] for row in training_summary["training_curve"]
        ] == [
            row["target_token_yx"]
            for row in random_training_summary["training_curve"]
        ]
        if not logged_target_schedule_identical:
            raise JEPAResidualPilotError(
                "provider arms did not share the logged target-tile schedule"
            )
        sampling_schedule_sha256_equal = bool(
            training_summary["sampling_schedule_sha256"]
            == random_training_summary["sampling_schedule_sha256"]
        )
        if not sampling_schedule_sha256_equal:
            raise JEPAResidualPilotError(
                "provider arms did not share every target and clip-index draw"
            )
        atomic_json(
            partial / "decoder_training_metrics.json",
            {
                "schema_version": SCHEMA_VERSION,
                "jepa_result": training_summary,
                "random_control_result": random_training_summary,
                "training_config": _jsonable(training_contract),
                "train_clip_count": train_count,
                "normalized_cache_reused_without_transform": train_cache_manifest[
                    "normalization_applied_by_this_runner"
                ]
                is False,
                "decoder_seed": int(decoder_config["seed"]),
                "tile_schedule_seed": int(decoder_config["training_seed"]),
                "initial_decoder_state_sha256": initial_decoder_hash,
                "random_initial_decoder_state_sha256": random_initial_decoder_hash,
                "decoder_initialization_identical_between_arms": initial_decoder_hash
                == random_initial_decoder_hash,
                "identical_training_tensor_config_and_seed_passed_by_guarded_runner": True,
                "logged_target_schedule_identical_between_arms": logged_target_schedule_identical,
                "sampling_schedule_sha256_equal_between_provider_arms": sampling_schedule_sha256_equal,
                "sampling_schedule_sha256": training_summary[
                    "sampling_schedule_sha256"
                ],
                "sampling_schedule_hashed_fields": training_summary[
                    "sampling_schedule_hashed_fields"
                ],
                "scientific_training": False,
            },
        )
        decoder_checkpoint_path = (
            partial / "checkpoints" / "decoder_seed_6201_final_non_scientific.pt"
        )
        decoder_state_sha256 = decoder_module.module_state_sha256(decoder_only)
        random_decoder_state_sha256 = decoder_module.module_state_sha256(
            random_decoder_only
        )
        checkpoint_identity = _decoder_checkpoint_identity(
            jepa_checkpoint_tensor_sha256=expected_tensor_hash,
            jepa_runtime_provider_sha256=frozen_hash_before,
            random_checkpoint_tensor_sha256=checkpoint_payload[
                "_random_tensor_sha256"
            ],
            random_runtime_provider_sha256=random_hash_before,
            jepa_decoder_state_sha256=decoder_state_sha256,
            random_decoder_state_sha256=random_decoder_state_sha256,
        )
        _atomic_torch_save(
            decoder_checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "scientific_checkpoint": False,
                "decoder_seed": int(decoder_config["seed"]),
                "decoder_steps": int(mode_config["decoder_steps"]),
                "upstream_checkpoint_sha256": config["upstream_screen"]["checkpoint"]["sha256"],
                **checkpoint_identity,
                "decoder_state_dict": {
                    key: value.detach().cpu() for key, value in decoder_only.state_dict().items()
                },
                "random_control_decoder_state_dict": {
                    key: value.detach().cpu()
                    for key, value in random_decoder_only.state_dict().items()
                },
            },
        )
        decoder_checkpoint_sha256 = sha256_file(decoder_checkpoint_path)
        atomic_json(
            partial / "decoder_checkpoints_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                **checkpoint_identity,
                "checkpoints": [
                    {
                        "path": "checkpoints/decoder_seed_6201_final_non_scientific.pt",
                        "sha256": decoder_checkpoint_sha256,
                        "sha256_scope": "complete_serialized_torch_checkpoint_file_bytes",
                        "decoder_seed": int(decoder_config["seed"]),
                        "tile_schedule_seed": int(decoder_config["training_seed"]),
                        "steps": int(mode_config["decoder_steps"]),
                        "scientific_checkpoint": False,
                        "arms": [
                            JEPA_RESIDUAL_METHOD,
                            RANDOM_RESIDUAL_METHOD,
                        ],
                        "frozen_provider_identity": checkpoint_identity[
                            "frozen_provider_identity"
                        ],
                        "decoder_state_identity": checkpoint_identity[
                            "decoder_state_identity"
                        ],
                    }
                ],
                "selection": "final_step_only_no_validation_selection",
            },
        )

        target_batch_size = int(decoder_config.get("target_batch_size", 16))
        amp = bool(decoder_runtime_config["amp"])
        off_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        random_off_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        calibration_evidence = {
            **raw_pretraining_calibration_evidence,
            "raw_rows_reused_during_joined_evaluation": 0,
            "residual_source_off_calibration_pair_count": 0,
        }
        _heartbeat(partial, "evaluating_exact_pairs", fixture_count=len(fixtures))
        for fixture in sorted(fixtures, key=lambda item: item.fixture_id):
            metadata = fixture_metadata[fixture.fixture_id]
            window_id = str(metadata["background_window_id"])
            if window_id not in off_cache:
                off_normalized = normalize_raw_movie_once(
                    fixture.native_background, normalization
                )
                predicted_off, coverage_off = _predict(
                    decoder_module,
                    model,
                    off_normalized,
                    device=requested_device,
                    target_batch_size=target_batch_size,
                    amp=amp,
                )
                residual_off = off_normalized - predicted_off
                off_cache[window_id] = (predicted_off, residual_off, coverage_off)
                coverage_rows.append(
                    {"background_window_id": window_id, "arm": "source_off", **coverage_off}
                )
            predicted_off, residual_off, coverage_off = off_cache[window_id]
            if window_id not in random_off_cache:
                random_off_normalized = normalize_raw_movie_once(
                    fixture.native_background, normalization
                )
                random_predicted_off, random_coverage_off = _predict(
                    decoder_module,
                    random_model,
                    random_off_normalized,
                    device=requested_device,
                    target_batch_size=target_batch_size,
                    amp=amp,
                )
                random_residual_off = random_off_normalized - random_predicted_off
                random_off_cache[window_id] = (
                    random_predicted_off,
                    random_residual_off,
                    random_coverage_off,
                )
                coverage_rows.append(
                    {
                        "background_window_id": window_id,
                        "arm": "random_control_source_off",
                        **random_coverage_off,
                    }
                )
            random_predicted_off, random_residual_off, _ = random_off_cache[window_id]
            on_normalized = normalize_raw_movie_once(fixture.observation, normalization)
            predicted_on, coverage_on = _predict(
                decoder_module,
                model,
                on_normalized,
                device=requested_device,
                target_batch_size=target_batch_size,
                amp=amp,
            )
            residual_on = on_normalized - predicted_on
            coverage_rows.append(
                {"fixture_id": fixture.fixture_id, "arm": "source_on", **coverage_on}
            )
            random_predicted_on, random_coverage_on = _predict(
                decoder_module,
                random_model,
                on_normalized,
                device=requested_device,
                target_batch_size=target_batch_size,
                amp=amp,
            )
            random_residual_on = on_normalized - random_predicted_on
            coverage_rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "arm": "random_control_source_on",
                    **random_coverage_on,
                }
            )
            raw_row = all_native_raw_by_fixture[str(fixture.fixture_id)]
            residual_scorer = PrecomputedResidualHandcraftedScorer(
                fixture.native_background,
                fixture.observation,
                residual_off,
                residual_on,
            )
            residual_row = evaluate_paired_fixture(
                fixture,
                residual_scorer,
                method=JEPA_RESIDUAL_METHOD,
                candidate_budget=int(evaluation["proposal_cap"]),
                match_radius_px=float(evaluation["match_radius_px"]),
                minimum_distance_px=int(evaluation["minimum_distance_px"]),
                border_px=int(evaluation["candidate_border_px"]),
            )
            random_residual_scorer = PrecomputedResidualHandcraftedScorer(
                fixture.native_background,
                fixture.observation,
                random_residual_off,
                random_residual_on,
            )
            random_residual_row = evaluate_paired_fixture(
                fixture,
                random_residual_scorer,
                method=RANDOM_RESIDUAL_METHOD,
                candidate_budget=int(evaluation["proposal_cap"]),
                match_radius_px=float(evaluation["match_radius_px"]),
                minimum_distance_px=int(evaluation["minimum_distance_px"]),
                border_px=int(evaluation["candidate_border_px"]),
            )
            if residual_scorer.fit_calls != 1 or residual_scorer.apply_calls != [
                "source_off",
                "source_on",
            ]:
                raise JEPAResidualPilotError("residual source-off calibration contract drifted")
            if random_residual_scorer.fit_calls != 1 or random_residual_scorer.apply_calls != [
                "source_off",
                "source_on",
            ]:
                raise JEPAResidualPilotError(
                    "random residual source-off calibration contract drifted"
                )
            calibration_evidence["raw_rows_reused_during_joined_evaluation"] += 1
            calibration_evidence["residual_source_off_calibration_pair_count"] += 2
            enriched_rows = [raw_row]
            for row, units in (
                (residual_row, "signed_frozen_normalized_pixel_residual"),
                (
                    random_residual_row,
                    "signed_random_control_normalized_pixel_residual",
                ),
            ):
                enriched_rows.append(
                    {
                        **row,
                        "background_recording_id": metadata["background_recording_id"],
                        "background_window_id": window_id,
                        "injection_seed": int(metadata["injection_seed"]),
                        "source_count": int(metadata["source_count"]),
                        "crowding_case": metadata["crowding_case"],
                        "score_input_units": units,
                    }
                )
            rows.extend(enriched_rows)
            injected_normalized = (
                np.asarray(fixture.injected_neural_signal, dtype=np.float32)
                / np.float32(normalization.scale)
            )
            algebra = paired_signal_algebra(
                normalize_raw_movie_once(fixture.native_background, normalization),
                on_normalized,
                predicted_off,
                predicted_on,
                injected_normalized,
            )
            background_diag = background_suppression_diagnostics(
                normalize_raw_movie_once(fixture.native_background, normalization),
                predicted_off,
            )
            seam = seam_diagnostics(
                predicted_on,
                patch_yx=config["_blind_tube"]["target_pixels_yx"],
            )
            decoder_diag = _jsonable(decoder_module.paired_residual_diagnostics(
                torch.from_numpy(normalize_raw_movie_once(fixture.native_background, normalization)).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(on_normalized).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(predicted_off).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(predicted_on).unsqueeze(0).unsqueeze(0),
                injected_signal=torch.from_numpy(injected_normalized).unsqueeze(0).unsqueeze(0),
                patch_size=tuple(int(value) for value in config["_blind_tube"]["token_shape_tyx"]),
            ))
            decoder_agreement = _decoder_diagnostic_agreement(algebra, decoder_diag)
            diagnostics.append(
                {
                    "method": JEPA_RESIDUAL_METHOD,
                    "fixture_id": fixture.fixture_id,
                    "background_recording_id": metadata["background_recording_id"],
                    "background_window_id": window_id,
                    "injection_seed": int(metadata["injection_seed"]),
                    "source_count": int(metadata["source_count"]),
                    "algebra": algebra,
                    "background": background_diag,
                    "seams": seam,
                    "decoder_reference": decoder_diag,
                    "decoder_reference_agreement": decoder_agreement,
                }
            )
            random_algebra = paired_signal_algebra(
                normalize_raw_movie_once(fixture.native_background, normalization),
                on_normalized,
                random_predicted_off,
                random_predicted_on,
                injected_normalized,
            )
            random_background_diag = background_suppression_diagnostics(
                normalize_raw_movie_once(fixture.native_background, normalization),
                random_predicted_off,
            )
            random_seam = seam_diagnostics(
                random_predicted_on,
                patch_yx=config["_blind_tube"]["target_pixels_yx"],
            )
            random_decoder_diag = _jsonable(
                decoder_module.paired_residual_diagnostics(
                    torch.from_numpy(
                        normalize_raw_movie_once(
                            fixture.native_background, normalization
                        )
                    ).unsqueeze(0).unsqueeze(0),
                    torch.from_numpy(on_normalized).unsqueeze(0).unsqueeze(0),
                    torch.from_numpy(random_predicted_off).unsqueeze(0).unsqueeze(0),
                    torch.from_numpy(random_predicted_on).unsqueeze(0).unsqueeze(0),
                    injected_signal=torch.from_numpy(injected_normalized)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    patch_size=tuple(
                        int(value)
                        for value in config["_blind_tube"]["token_shape_tyx"]
                    ),
                )
            )
            random_decoder_agreement = _decoder_diagnostic_agreement(
                random_algebra, random_decoder_diag
            )
            diagnostics.append(
                {
                    "method": RANDOM_RESIDUAL_METHOD,
                    "fixture_id": fixture.fixture_id,
                    "background_recording_id": metadata["background_recording_id"],
                    "background_window_id": window_id,
                    "injection_seed": int(metadata["injection_seed"]),
                    "source_count": int(metadata["source_count"]),
                    "algebra": random_algebra,
                    "background": random_background_diag,
                    "seams": random_seam,
                    "decoder_reference": random_decoder_diag,
                    "decoder_reference_agreement": random_decoder_agreement,
                }
            )

        grouped = _grouped_primary_summary(
            rows,
            evaluation=evaluation,
            expected_source_counts_per_cluster=(3 if mode == "screen" else 1),
            expected_source_count_values=(
                (1, 2, 4)
                if mode == "screen"
                else (int(fixtures[0].metadata["source_count"]),)
            ),
        )
        advancement = (
            _descriptive_advancement_panel(
                grouped,
                diagnostics,
                config["descriptive_screen_advancement_thresholds"],
            )
            if mode == "screen"
            else {
                "status": "not_evaluated_in_implementation_smoke",
                "formal_claim_consequence": "none",
                "scientific_promotion_allowed": False,
            }
        )
        flattened = [_flatten_row(row) for row in rows]
        atomic_json(
            partial / "paired_injection_results.json",
            {
                "schema_version": SCHEMA_VERSION,
                "rows": rows,
                "native_background_candidates": "unknown_not_negative",
                "precision_available": False,
            },
        )
        _write_tsv(partial / "paired_injection_results.tsv", flattened)
        atomic_json(
            partial / "residual_diagnostics.json",
            {"schema_version": SCHEMA_VERSION, "rows": diagnostics},
        )
        diagnostic_flat = [
            {
                "fixture_id": row["fixture_id"],
                "background_recording_id": row["background_recording_id"],
                "background_window_id": row["background_window_id"],
                "injection_seed": row["injection_seed"],
                "source_count": row["source_count"],
                **row["algebra"],
                **row["background"],
                **row["seams"],
            }
            for row in diagnostics
        ]
        _write_tsv(partial / "residual_diagnostics.tsv", diagnostic_flat)
        atomic_json(
            partial / "prediction_coverage.json",
            {"schema_version": SCHEMA_VERSION, "rows": coverage_rows},
        )
        frozen_hash_final = decoder_module.module_state_sha256(provider_nn)
        if frozen_hash_final != frozen_hash_before:
            raise JEPAResidualPilotError("frozen JEPA weights changed during evaluation")
        random_hash_final = decoder_module.module_state_sha256(random_provider_nn)
        if random_hash_final != random_hash_before:
            raise JEPAResidualPilotError(
                "frozen random-provider weights changed during evaluation"
            )
        freeze_integrity = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_jepa_tensor_sha256": expected_tensor_hash,
            "loaded_before_training_sha256": frozen_hash_before,
            "after_decoder_training_sha256": frozen_hash_after_training,
            "after_evaluation_sha256": frozen_hash_final,
            "all_hash_identical": len(
                {frozen_hash_before, frozen_hash_after_training, frozen_hash_final}
            )
            == 1,
            "random_loaded_before_training_sha256": random_hash_before,
            "random_after_decoder_training_sha256": random_hash_after_training,
            "random_after_evaluation_sha256": random_hash_final,
            "random_all_hash_identical": len(
                {random_hash_before, random_hash_after_training, random_hash_final}
            )
            == 1,
            "all_jepa_parameters_frozen": not any(
                parameter.requires_grad for parameter in provider_nn.parameters()
            ),
            "all_random_provider_parameters_frozen": not any(
                parameter.requires_grad for parameter in random_provider_nn.parameters()
            ),
            "decoder_only_trainable": all(
                parameter.requires_grad for parameter in decoder_only.parameters()
            )
            and all(
                parameter.requires_grad for parameter in random_decoder_only.parameters()
            )
            and not any(parameter.requires_grad for parameter in provider_nn.parameters())
            and not any(
                parameter.requires_grad for parameter in random_provider_nn.parameters()
            ),
        }
        atomic_json(partial / "freeze_integrity.json", freeze_integrity)
        audit = _audit_placeholders(partial, mode=mode, fixture_count=len(fixtures))
        finished_at = _utc_now()
        primary_diagnostics = [
            row for row in diagnostics if row["method"] == JEPA_RESIDUAL_METHOD
        ]
        algebra_integrity = _algebra_integrity_summary(
            diagnostics, config["engineering_integrity_thresholds"]
        )
        proposal_counts = _proposal_count_summary(
            rows, proposal_cap=int(evaluation["proposal_cap"])
        )
        calibration_evidence.update(
            {
                "expected_raw_source_off_calibration_pair_count": STRICT_SCREEN_FIXTURES,
                "expected_raw_recovery_result_object_count": 2
                * STRICT_SCREEN_FIXTURES,
                "expected_raw_rows_reused_during_joined_evaluation": len(fixtures),
                "expected_residual_source_off_calibration_pair_count": 2
                * len(fixtures),
            }
        )
        calibration_evidence["all_source_off_calibrations_exact"] = bool(
            calibration_evidence["raw_source_off_calibration_pair_count"]
            == calibration_evidence[
                "expected_raw_source_off_calibration_pair_count"
            ]
            and calibration_evidence["raw_recovery_result_object_count"]
            == calibration_evidence["expected_raw_recovery_result_object_count"]
            and calibration_evidence["raw_rows_reused_during_joined_evaluation"]
            == calibration_evidence[
                "expected_raw_rows_reused_during_joined_evaluation"
            ]
            and calibration_evidence[
                "residual_source_off_calibration_pair_count"
            ]
            == calibration_evidence[
                "expected_residual_source_off_calibration_pair_count"
            ]
        )
        decoder_reference_all_match = all(
            bool(row["decoder_reference_agreement"]["all_match"])
            for row in diagnostics
        )
        expected_nested_source_count = 3 if mode == "screen" else 1
        nested_source_counts_valid = bool(
            grouped["source_count_aggregated_cluster_rows"]
        ) and all(
            int(row["source_counts_aggregated"])
            == expected_nested_source_count
            and row["source_count_values"]
            == ([1, 2, 4] if mode == "screen" else fixture_summary["source_counts"])
            for row in grouped["source_count_aggregated_cluster_rows"]
        )
        implementation_source_hashes = {
            str(row["path"]): str(row["observed_sha256"])
            for row in preflight["implementation_hash_validation"]["sources"]
        }
        hashes = {
            "resolved_input_config_sha256": sha256_file(config_file),
            "parent_artifact_index_sha256": preflight["parent_integrity"][
                "artifact_index_sha256"
            ],
            "parent_checkpoint_sha256": config["upstream_screen"]["checkpoint"][
                "sha256"
            ],
            "parent_train_cache_sha256": config["upstream_screen"]["train_cache"][
                "sha256"
            ],
            "parent_validation_cache_sha256": config["upstream_screen"][
                "validation_cache"
            ]["sha256"],
            "parent_paired_injection_manifest_sha256": config["upstream_screen"][
                "paired_injection_manifest_sha256"
            ],
            "parent_paired_injection_results_sha256": config["upstream_screen"][
                "paired_injection_results_sha256"
            ],
            "frozen_jepa_provider_pretraining_sha256": frozen_hash_before,
            "frozen_jepa_provider_after_training_sha256": frozen_hash_after_training,
            "frozen_jepa_provider_after_evaluation_sha256": frozen_hash_final,
            "frozen_random_provider_pretraining_sha256": random_hash_before,
            "frozen_random_provider_after_training_sha256": random_hash_after_training,
            "frozen_random_provider_after_evaluation_sha256": random_hash_final,
            "decoder_checkpoint_sha256": decoder_checkpoint_sha256,
            "implementation_sources": implementation_source_hashes,
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": config["run_id"],
            "execution": {
                "mode": mode,
                "started_at": started_at,
                "finished_at": finished_at,
                "device": requested_device,
                "decoder_steps": int(mode_config["decoder_steps"]),
                "scientific_execution": False,
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
                "authorization": preflight["authorization"],
                "provenance": preflight["execution_provenance"],
                "config_sha256": sha256_file(config_file),
            },
            "upstream": {
                "experiment_id": "NREV-EXP-0028",
                "run_id": config["upstream_screen"]["run_id"],
                "all_parent_artifacts_verified_before_checkpoint_load": bool(
                    preflight["parent_integrity"]["all_parent_artifacts_verified"]
                    and preflight["checkpoint_loaded"] is False
                ),
                "checkpoint_seed": STRICT_PARENT_SEED,
                "checkpoint_steps": STRICT_PARENT_STEPS,
                "checkpoint_scientific": False,
            },
            "decoder": {
                "trainable_parameters": decoder_parameter_count,
                "seed": int(decoder_config["seed"]),
                "tile_schedule_seed": int(decoder_config["training_seed"]),
                "frozen_jepa_hash_identical_pre_post": freeze_integrity[
                    "all_hash_identical"
                ],
                "frozen_random_hash_identical_pre_post": freeze_integrity[
                    "random_all_hash_identical"
                ],
            },
            "seeds": _jsonable(config["seeds"]),
            "hashes": hashes,
            "paired_injection": fixture_summary,
            "proposal_counts": proposal_counts,
            "raw_hc_cross_run_regression": raw_anchor,
            "primary_evaluation": grouped,
            "descriptive_advancement_panel": advancement,
            "random_encoder_control": {
                "status": "emitted_checkpoint_matched_negative_control",
                "primary_pair_affected": False,
                "method": RANDOM_RESIDUAL_METHOD,
                "same_decoder_initialization": initial_decoder_hash
                == random_initial_decoder_hash,
                "same_logged_target_schedule": logged_target_schedule_identical,
                "sampling_schedule_sha256_equal_between_provider_arms": sampling_schedule_sha256_equal,
                "sampling_schedule_sha256": training_summary[
                    "sampling_schedule_sha256"
                ],
                "same_training_tensor_config_and_seed_by_guarded_call_contract": True,
            },
            "diagnostics": {
                "fixture_count": len(primary_diagnostics),
                "mean_retained_gain_projection": float(
                    np.mean([row["algebra"]["retained_gain_projection"] for row in primary_diagnostics])
                ),
                "mean_predictor_absorption_projection": float(
                    np.mean([row["algebra"]["predictor_absorption_projection"] for row in primary_diagnostics])
                ),
                "mean_background_rms_ratio": float(
                    np.mean([row["background"]["background_rms_ratio"] for row in primary_diagnostics])
                ),
                "mean_dynamic_mad_ratio": float(
                    np.mean([row["background"]["dynamic_mad_ratio"] for row in primary_diagnostics])
                ),
                "mean_total_signal_error_ratio": float(
                    np.mean(
                        [
                            row["algebra"]["total_signal_error_ratio"]
                            for row in primary_diagnostics
                        ]
                    )
                ),
                "median_total_signal_error_ratio": float(
                    np.median(
                        [
                            row["algebra"]["total_signal_error_ratio"]
                            for row in primary_diagnostics
                        ]
                    )
                ),
                "mean_orthogonal_distortion_ratio": float(
                    np.mean(
                        [
                            row["algebra"]["orthogonal_distortion_ratio"]
                            for row in primary_diagnostics
                        ]
                    )
                ),
                "median_orthogonal_distortion_ratio": float(
                    np.median(
                        [
                            row["algebra"]["orthogonal_distortion_ratio"]
                            for row in primary_diagnostics
                        ]
                    )
                ),
                "engineering_algebra_integrity": algebra_integrity,
                "decoder_reference_ratios_agree": decoder_reference_all_match,
                "source_off_calibration": calibration_evidence,
            },
            "scientific_audit": audit,
            "motion_dependency": {
                "experiment_id": "NREV-EXP-0025",
                "satisfied": bool(
                    config["dependency_readiness"]["NREV-EXP-0025"]["satisfied"]
                ),
                "nuisance_robust_conclusion_allowed": False,
                "status": "unresolved_blocks_claim_bearing_interpretation",
            },
            "claim_boundary": {
                "supported": "bounded paired exact-injection residual-versus-raw sensitivity and signal-retention diagnostics only",
                "not_established": [
                    "biological neuron identity",
                    "precision or specificity",
                    "independent-animal generalization",
                    "scientific denoising benefit",
                    "causality",
                    "reinforcement learning",
                    "motion_or_registration_nuisance_robustness",
                ],
            },
        }
        atomic_json(partial / "summary.json", summary)
        validation_checks = {
            "all_parent_artifacts_verified_before_checkpoint_load": bool(
                preflight["parent_integrity"]["all_parent_artifacts_verified"]
                and preflight["checkpoint_loaded"] is False
            ),
            "parent_checkpoint_seed_1001_steps_500_non_scientific": all(
                bool(value) for value in checkpoint_payload["_strict_checks"].values()
            ),
            "normalized_cache_reused_without_double_normalization": bool(
                train_cache_manifest["normalization_applied_by_this_runner"] is False
                and validation_cache_manifest[
                    "normalization_applied_by_this_runner"
                ]
                is False
                and normalization_check[
                    "normalized_cache_normalization_application_count"
                ]
                == 0
            ),
            "exact_parent_fixture_manifest_identity": fixture_identity[
                "exact_parent_fixture_manifest_identity"
            ],
            "fixture_injection_additive_closure_at_most_0_51_ulp": fixture_identity[
                "maximum_pair_closure_float32_ulp"
            ]
            <= 0.51,
            "frozen_jepa_hash_identical_pre_post": freeze_integrity["all_hash_identical"],
            "frozen_random_provider_hash_identical_pre_post": freeze_integrity[
                "random_all_hash_identical"
            ],
            "decoder_is_only_trainable_component": freeze_integrity[
                "decoder_only_trainable"
            ],
            "random_control_decoder_initialization_and_full_sampling_schedule_match": bool(
                initial_decoder_hash == random_initial_decoder_hash
                and logged_target_schedule_identical
                and sampling_schedule_sha256_equal
            ),
            "source_off_only_handcrafted_calibration": calibration_evidence[
                "all_source_off_calibrations_exact"
            ],
            "blind_tube_coverage_exactly_once": all(
                row["minimum_coverage"] == row["maximum_coverage"] == 1
                for row in coverage_rows
            ),
            "prediction_shape_finite_and_coverage_contract": all(
                row["prediction_contract_passed"] is True
                and row["background_shape_exact"] is True
                and row["coverage_shape_exact"] is True
                and row["background_coverage_shapes_exact_match"] is True
                and row["background_all_finite"] is True
                and row["coverage_exactly_once"] is True
                for row in coverage_rows
            ),
            "all_64_target_tubes_invariant_before_training": bool(
                geometry_validation["passed_before_first_optimizer_step"]
                and geometry_validation["tested_target_count_per_provider_arm"]
                == geometry_validation["required_target_count_per_provider_arm"]
                == 64
            ),
            "registered_252_source_footprint_geometry_matches": geometry_leakage[
                "matches_registered_exact_maxima"
            ],
            "source_counts_nested_in_grouped_bootstrap": nested_source_counts_valid,
            "raw_hc_exact_cross_run_anchor_identity": raw_anchor[
                "exact_recovery_result_and_recall_identity"
            ],
            "projection_retention_absorption_closure_within_engineering_threshold": algebra_integrity[
                "projection_closure_passed"
            ],
            "voxelwise_residual_prediction_pair_closure_within_engineering_threshold": algebra_integrity[
                "voxelwise_pair_closure_passed"
            ],
            "runner_and_decoder_distortion_diagnostics_agree": decoder_reference_all_match,
            "all_candidate_counts_within_proposal_cap": proposal_counts[
                "all_counts_within_proposal_cap"
            ],
            "required_recording_count_represented_for_mode": len(
                {row["background_recording_id"] for row in rows}
            )
            == (4 if mode == "screen" else 1),
            "engineering_execution_authorized_and_claim_authorizations_false": bool(
                preflight["authorization"][
                    "engineering_screen_execution_authorized"
                ]
                and preflight["authorization"][
                    "all_scientific_and_claim_bearing_authorizations_false"
                ]
            ),
            "execution_provenance_captured": bool(
                preflight["execution_provenance"]["git"]["status"] == "captured"
                and preflight["execution_provenance"]["runtime"]["python_version"]
                and preflight["execution_provenance"]["runtime"]["torch_version"]
            ),
        }
        scientific_claim_checks = {
            "scientific_audit_complete": bool(audit["scientific_audit_complete"]),
            "motion_dependency_satisfied": bool(
                config["dependency_readiness"]["NREV-EXP-0025"]["satisfied"]
            ),
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
        }
        validation = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "screen_complete_claim_gates_unresolved"
                if mode == "screen"
                else "smoke_complete_non_scientific"
            ),
            "engineering_checks": validation_checks,
            "scientific_claim_checks": scientific_claim_checks,
            "checks": {**validation_checks, **scientific_claim_checks},
            "engineering_integrity_observed": algebra_integrity,
            "failed_engineering_checks": sorted(
                key
                for key, value in validation_checks.items()
                if not value
            ),
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
            "promotion_blockers": [
                "non_scientific_execution_mode",
                "scientific_audit_media_and_inventory_pending",
                "single_decoder_seed_only",
                "independent_animal_generalization_unresolved",
                "native_background_biological_truth_not_exhaustive",
            ],
        }
        if validation["failed_engineering_checks"]:
            raise JEPAResidualPilotError(
                "engineering validation failed: "
                + ", ".join(validation["failed_engineering_checks"])
            )
        atomic_json(partial / "validation.json", validation)
        atomic_json(
            partial / "llm_context.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "run_id": config["run_id"],
                "result_file": "summary.json",
                "paired_rows": "paired_injection_results.tsv",
                "diagnostics": "residual_diagnostics.tsv",
                "primary_method": JEPA_RESIDUAL_METHOD,
                "comparator": RAW_METHOD,
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
                "unmatched_native_candidates": "unknown_not_negative",
            },
        )
        atomic_text(partial / "REPORT.md", _report(summary))
        parent_unchanged = assert_parent_unchanged(
            parent_before,
            parent_root,
            expected_artifact_index_sha256=config["upstream_screen"][
                "artifact_index_sha256"
            ],
        )
        atomic_json(
            partial / "upstream_integrity.json",
            {**parent_before, "post_execution": parent_unchanged},
        )
        atomic_json(
            partial / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": (
                    "screen_complete_claim_gates_unresolved"
                    if mode == "screen"
                    else "smoke_complete_non_scientific"
                ),
                "mode": mode,
                "started_at": started_at,
                "finished_at": finished_at,
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
                "validation": "validation.json",
            },
        )
        _heartbeat(partial, "complete_non_scientific")
        atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        required_outputs = config.get("required_outputs")
        if not isinstance(required_outputs, list):
            raise JEPAResidualPilotError("config required_outputs must be a list")
        _validate_required_outputs(partial, required_outputs)
        partial.replace(output)
        return summary
    except BaseException as exc:
        if partial.exists():
            atomic_json(
                partial / "status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_nonresumable_partial",
                    "mode": mode,
                    "failed_at": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scientific_completion": False,
                    "scientific_promotion_allowed": False,
                },
            )
        raise
    finally:
        torch.set_num_threads(previous_threads)


__all__ = [
    "EXPERIMENT_ID",
    "RUNNER_VERSION",
    "JEPA_RESIDUAL_METHOD",
    "JEPAResidualPilotError",
    "PrecomputedResidualHandcraftedScorer",
    "FrozenNormalizedHandcraftedComparator",
    "FrozenNativeHandcraftedComparator",
    "RANDOM_RESIDUAL_METHOD",
    "RAW_METHOD",
    "assert_fixture_manifest_identity",
    "assert_parent_unchanged",
    "background_suppression_diagnostics",
    "guard_output_collision",
    "footprint_halo_leakage_diagnostics",
    "load_jepa_residual_config",
    "load_normalized_parent_cache",
    "module_state_sha256",
    "normalize_raw_movie_once",
    "paired_signal_algebra",
    "preflight_jepa_residual_pilot",
    "reconstruct_exact_fixtures",
    "reconstruct_frozen_backgrounds",
    "run_jepa_residual_pilot",
    "seam_diagnostics",
    "tensor_mapping_sha256",
    "verify_parent_run_integrity",
]
