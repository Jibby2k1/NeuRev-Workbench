"""Bounded motion, registration-residual, and acquisition-confound screen.

This module operationalizes the draft NREV-EXP-0025 question without changing
the frozen NREV-EXP-0029 evaluation.  It reuses the twelve immutable 32x64x64
background windows, reads the raw TIFFs through memmaps, and relates
translation-like diagnostics to temporal MAD and already-frozen raw-HC and
conditional-residual endpoints. Raw and registered difference MADs used in a
ratio are evaluated on the exact same shift-valid interior for each pair.

The output is an engineering audit, not motion correction, neuron truth, or a
causal attribution.  Phase correlation cannot distinguish specimen motion from
scan/acquisition effects, and integer TIFF rails do not identify the analog
detector or ADC rails when acquisition metadata are absent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker
from scipy import ndimage, stats
import tifffile
import yaml

from .contracts import atomic_json, atomic_text, stable_hash
from .discovery import sha256_file
from .jepa_motion_audit import estimate_phase_translation


SCHEMA_VERSION = "neurobench.motion_registration_confound_audit.v2"
ALGORITHM = "bidirectional_phase_translation_and_matched_support_registered_difference_v2"
EXPECTED_STRATA = ("low_mad", "median_mad", "high_mad")
EXPECTED_PARENT_METHODS = (
    "jepa_conditional_pixel_residual_frozen_handcrafted_stack",
    "random_encoder_conditional_pixel_residual_frozen_handcrafted_stack",
)
PRIMARY_ASSOCIATIONS = (
    ("manifest_temporal_mad", "global_translation_p95_px"),
    ("manifest_temporal_mad", "registered_over_raw_difference_mad_ratio"),
    ("global_translation_p95_px", "jepa_background_rms_ratio"),
    ("global_translation_p95_px", "jepa_dynamic_mad_ratio"),
    ("global_translation_p95_px", "jepa_seam_to_interior_jump_ratio"),
    ("global_translation_p95_px", "raw_hc_source_on_micro_recall"),
    ("global_translation_p95_px", "raw_hc_source_on_seed_sd_mean"),
    ("two_step_cycle_closure_p95_px", "jepa_background_rms_ratio"),
    ("tile_global_disagreement_p95_px", "jepa_background_rms_ratio"),
    ("registered_over_raw_difference_mad_ratio", "jepa_background_rms_ratio"),
    ("registered_over_raw_difference_mad_ratio", "jepa_dynamic_mad_ratio"),
    ("registered_over_raw_difference_mad_ratio", "raw_hc_source_on_micro_recall"),
    ("raw_difference_mad_median", "raw_hc_source_on_micro_recall"),
    ("raw_difference_mad_median", "raw_hc_center_delta_median"),
)
RUN_MODE = "bounded_non_claim_bearing_engineering_screen"
RUNNER_MODULE = "neurobench.experiments.neuron_identifiability.motion_registration_confound_audit"


@dataclass(frozen=True)
class MotionRegistrationConfoundConfig:
    expected_window_count: int = 12
    expected_recording_count: int = 4
    expected_windows_per_recording: int = 3
    expected_window_frames: int = 32
    expected_window_shape_yx: tuple[int, int] = (64, 64)
    tile_grid_yx: tuple[int, int] = (2, 2)
    minimum_registration_extent: int = 12
    minimum_robust_scale: float = 1e-6
    registration_clip_z: float = 8.0
    minimum_peak_to_median_ratio: float = 3.0
    maximum_translation_search_px: float = 8.0
    temporal_mad_time_stride: int = 2
    temporal_mad_spatial_stride: int = 4
    temporal_mad_absolute_tolerance: float = 1e-4
    minimum_forward_backward_valid_fraction: float = 0.50
    search_boundary_component_fraction_review: float = 0.10
    tile_global_disagreement_review_px: float = 2.0
    minimum_registration_difference_reduction_fraction: float = 0.10
    registration_residual_interpolation_order: int = 1
    verify_raw_hashes: bool = True
    scientific_audit_enabled: bool = True
    scientific_completion_allowed: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.expected_window_count,
            self.expected_recording_count,
            self.expected_windows_per_recording,
            self.expected_window_frames,
            *self.expected_window_shape_yx,
            *self.tile_grid_yx,
            self.minimum_registration_extent,
            self.temporal_mad_time_stride,
            self.temporal_mad_spatial_stride,
        )
        if any(int(value) < 1 for value in counts):
            raise ValueError("motion-confound integer configuration values must be positive")
        if self.expected_window_count != self.expected_recording_count * self.expected_windows_per_recording:
            raise ValueError("expected window count must equal recordings times windows per recording")
        positive = (
            self.minimum_robust_scale,
            self.registration_clip_z,
            self.minimum_peak_to_median_ratio,
            self.maximum_translation_search_px,
            self.temporal_mad_absolute_tolerance,
            self.tile_global_disagreement_review_px,
            self.minimum_registration_difference_reduction_fraction,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("motion-confound magnitude configuration values must be finite and positive")
        if not 0 < self.minimum_forward_backward_valid_fraction <= 1:
            raise ValueError("minimum_forward_backward_valid_fraction must be in (0, 1]")
        if not 0 <= self.search_boundary_component_fraction_review <= 1:
            raise ValueError("search_boundary_component_fraction_review must be in [0, 1]")
        if self.minimum_registration_difference_reduction_fraction > 1:
            raise ValueError("minimum_registration_difference_reduction_fraction must be at most one")
        if self.registration_residual_interpolation_order not in {0, 1, 3}:
            raise ValueError("registration residual interpolation order must be 0, 1, or 3")
        if self.scientific_completion_allowed:
            raise ValueError("this bounded runner never allows scientific completion")

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_window_shape_yx"] = list(self.expected_window_shape_yx)
        payload["tile_grid_yx"] = list(self.tile_grid_yx)
        payload["threshold_policy"] = "engineering_review_triggers_not_validated_biological_cut_points"
        return payload


def _normal_consistent_mad(values: np.ndarray, *, center: float | None = None) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    resolved_center = float(np.median(array)) if center is None else float(center)
    return float(1.4826 * np.median(np.abs(array - resolved_center)))


def _finite_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"count": 0, "median": None, "p95": None, "minimum": None, "maximum": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _nullable(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(_format_tsv_value(item) for item in value)
    return str(value)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path.name}")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError(f"TSV rows do not share one schema: {path.name}")
    lines = ["\t".join(columns)]
    lines.extend("\t".join(_format_tsv_value(row[column]) for column in columns) for row in rows)
    atomic_text(path, "\n".join(lines) + "\n")


def _portable_repository_path(path: Path, repository_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return "repo://" + resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {resolved}") from exc


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        )
        return completed.stdout

    repository = run("remote", "get-url", "origin").decode("utf-8").strip()
    if repository.startswith("git@") and ":" in repository:
        authority, path = repository.removeprefix("git@").split(":", 1)
        repository = f"https://{authority}/{path.removesuffix('.git')}"
    elif repository.startswith("ssh://git@"):
        authority_and_path = repository.removeprefix("ssh://git@")
        authority, path = authority_and_path.split("/", 1)
        repository = f"https://{authority}/{path.removesuffix('.git')}"
    elif repository.endswith(".git"):
        repository = repository[:-4]
    if not repository.startswith("https://") or "@" in repository.removeprefix("https://").split("/", 1)[0]:
        raise ValueError("Git origin must normalize to a credential-free HTTPS repository URI")

    status = run("status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty = bool(status)
    result = {
        "repository": repository,
        "commit": run("rev-parse", "HEAD").decode("ascii").strip().lower(),
        "branch": run("branch", "--show-current").decode("utf-8").strip() or "detached",
        "dirty": dirty,
        "dirty_entry_count": status.count(b"\0"),
        "dirty_status_sha256": hashlib.sha256(status).hexdigest(),
        "dirty_paths_persisted": False,
        "diff_algorithm": "NEUREV-DIRTY-STATE-V1",
    }
    if dirty:
        digest = hashlib.sha256()
        digest.update(b"NEUREV-DIRTY-STATE-V1\0")
        digest.update(status)
        digest.update(run("diff", "--binary", "HEAD", "--"))
        untracked = run("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        for encoded in sorted(item for item in untracked if item):
            relative = encoded.decode("utf-8", errors="surrogateescape")
            path = repository_root / relative
            digest.update(b"\0PATH\0")
            digest.update(encoded)
            if path.is_symlink():
                digest.update(b"\0SYMLINK\0")
                digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            elif path.is_file():
                digest.update(b"\0FILE\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        result["diff_sha256"] = digest.hexdigest()
    return result


def _registry_code_record(git_provenance: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "repository": git_provenance["repository"],
        "commit": git_provenance["commit"],
        "branch": git_provenance["branch"],
        "dirty": git_provenance["dirty"],
    }
    if git_provenance["dirty"]:
        record["diff_sha256"] = git_provenance["diff_sha256"]
    return record


def validate_registry_code_against_run_schema(
    code_record: Mapping[str, Any],
    run_schema: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate the exact code projection against run.schema.json."""
    properties = run_schema.get("properties")
    definitions = run_schema.get("$defs")
    if not isinstance(properties, Mapping) or not isinstance(properties.get("code"), Mapping):
        raise ValueError("run schema is missing properties.code")
    if not isinstance(definitions, Mapping):
        raise ValueError("run schema is missing $defs")
    code_schema = {
        "$schema": run_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        **dict(properties["code"]),
        "$defs": dict(definitions),
    }
    Draft202012Validator.check_schema(code_schema)
    validator = Draft202012Validator(code_schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(code_record)), key=lambda error: list(error.absolute_path))
    if errors:
        locations = [".".join(map(str, error.absolute_path)) or "<code>" for error in errors]
        raise ValueError(f"registry code provenance violates run.schema.json at {locations}")
    return {
        "run_schema_code_subschema_valid": True,
        "dirty_diff_conditional_satisfied": True,
        "additional_properties_absent": True,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("run timestamp must be timezone-aware UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must resolve to UTC")
    return parsed


def _require_hex_digest(value: Any, *, field: str, length: int = 64) -> str:
    digest = str(value)
    if len(digest) != length or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase {length}-character hexadecimal digest")
    return digest


def validate_run_provenance(
    payload: Mapping[str, Any],
    *,
    runner_path: Path | None = None,
) -> dict[str, bool]:
    """Fail closed unless a successful run has complete, ordered provenance."""
    required = {
        "schema_version",
        "record_type",
        "experiment_id",
        "run_id",
        "lifecycle",
        "planned_at",
        "started_at",
        "ended_at",
        "duration_seconds",
        "execution",
        "code",
        "configuration",
        "inputs",
        "environment",
        "scientific_boundary",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"run provenance missing required fields: {missing}")
    if payload["record_type"] != "run_provenance" or payload["lifecycle"] != "succeeded":
        raise ValueError("run provenance must describe one succeeded run")
    if payload["experiment_id"] != "NREV-EXP-0025" or not str(payload["run_id"]).startswith(
        "NREV-RUN-EXP-0025-"
    ):
        raise ValueError("run provenance experiment/run identity mismatch")
    planned = _parse_utc(payload["planned_at"], field="planned_at")
    started = _parse_utc(payload["started_at"], field="started_at")
    ended = _parse_utc(payload["ended_at"], field="ended_at")
    if not planned <= started <= ended:
        raise ValueError("run provenance timestamps must satisfy planned <= started <= ended")
    duration = float(payload["duration_seconds"])
    observed_duration = (ended - started).total_seconds()
    if not np.isfinite(duration) or duration < 0 or abs(duration - observed_duration) > 1e-6:
        raise ValueError("duration_seconds must exactly match ended_at minus started_at")

    execution = payload["execution"]
    if not isinstance(execution, Mapping):
        raise ValueError("run provenance execution must be an object")
    command = execution.get("command")
    if execution.get("mode") != RUN_MODE or not isinstance(command, list) or not command:
        raise ValueError("run provenance requires the bounded mode and a nonempty command list")
    if any(not isinstance(token, str) or not token for token in command):
        raise ValueError("run provenance command tokens must be nonempty strings")
    if any("/home/" in token for token in command):
        raise ValueError("run provenance command must not leak workstation paths")

    code = payload["code"]
    if not isinstance(code, Mapping):
        raise ValueError("run provenance code must be an object")
    if code.get("runner_module") != RUNNER_MODULE:
        raise ValueError("run provenance runner module mismatch")
    runner_sha256 = _require_hex_digest(code.get("runner_sha256"), field="runner_sha256")
    resolved_runner_path = Path(__file__) if runner_path is None else Path(runner_path)
    if not resolved_runner_path.is_file() or sha256_file(resolved_runner_path) != runner_sha256:
        raise ValueError("runner_sha256 does not match the live runner module")
    git = code.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("run provenance requires git state")
    _require_hex_digest(git.get("commit"), field="git.commit", length=40)
    if not str(git.get("repository", "")).startswith("https://") or not git.get("branch"):
        raise ValueError("run provenance requires a portable repository URI and nonempty branch")
    if not isinstance(git.get("dirty"), bool) or not isinstance(git.get("dirty_entry_count"), int):
        raise ValueError("run provenance requires typed git dirty state")
    _require_hex_digest(git.get("dirty_status_sha256"), field="git.dirty_status_sha256")
    if git.get("diff_algorithm") != "NEUREV-DIRTY-STATE-V1":
        raise ValueError("run provenance requires the canonical dirty-state algorithm")
    registry_code = {
        "repository": code.get("repository"),
        "commit": code.get("commit"),
        "branch": code.get("branch"),
        "dirty": code.get("dirty"),
    }
    expected_registry_code = {
        "repository": git.get("repository"),
        "commit": git.get("commit"),
        "branch": git.get("branch"),
        "dirty": git.get("dirty"),
    }
    if registry_code != expected_registry_code:
        raise ValueError("code fields do not mirror the captured Git state")
    if code.get("dirty") is True:
        diff_sha256 = _require_hex_digest(code.get("diff_sha256"), field="code.diff_sha256")
        if _require_hex_digest(git.get("diff_sha256"), field="git.diff_sha256") != diff_sha256:
            raise ValueError("code.diff_sha256 disagrees with the canonical Git-state digest")
    elif code.get("diff_sha256") is not None or git.get("diff_sha256") is not None:
        raise ValueError("clean code provenance must not carry a stale diff_sha256")
    registry_schema = code.get("registry_schema")
    if not isinstance(registry_schema, Mapping):
        raise ValueError("code provenance requires the registry schema reference")
    if registry_schema.get("uri") != "repo://research/schemas/run.schema.json":
        raise ValueError("code provenance registry schema URI is not canonical")
    _require_hex_digest(registry_schema.get("sha256"), field="code.registry_schema.sha256")
    if not isinstance(registry_schema.get("checks"), Mapping) or not all(
        registry_schema["checks"].values()
    ):
        raise ValueError("code provenance registry schema validation did not pass")

    configuration = payload["configuration"]
    if not isinstance(configuration, Mapping):
        raise ValueError("run provenance configuration must be an object")
    config_sha256 = _require_hex_digest(configuration.get("config_sha256"), field="config_sha256")
    if not isinstance(configuration.get("config"), Mapping):
        raise ValueError("run provenance requires the exact resolved scientific config")
    if stable_hash(configuration["config"]) != config_sha256:
        raise ValueError("config_sha256 does not match the persisted scientific config")
    _require_hex_digest(configuration.get("resolved_config_sha256"), field="resolved_config_sha256")
    if configuration.get("resolved_config_uri") != "run://resolved_config.json":
        raise ValueError("resolved config must use the run-local portable URI")

    inputs = payload["inputs"]
    records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list) or not records:
        raise ValueError("run provenance requires a nonempty input record list")
    ids = [str(row.get("id", "")) for row in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("run provenance input IDs must be nonempty and unique")
    if ids != sorted(ids):
        raise ValueError("run provenance input records must be sorted by ID")
    for row in records:
        if not isinstance(row, Mapping) or not str(row.get("role", "")):
            raise ValueError("run provenance input records require a nonempty role")
        if not str(row.get("uri", "")).startswith(("repo://", "data://")):
            raise ValueError("run provenance input URI must be portable repo:// or data://")
        _require_hex_digest(row.get("sha256"), field=f"input[{row.get('id')}].sha256")
        if not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool) or row["bytes"] < 0:
            raise ValueError("run provenance input bytes must be a nonnegative integer")
    schema_inputs = [row for row in records if row.get("id") == "registry_run_schema"]
    if len(schema_inputs) != 1 or schema_inputs[0].get("uri") != registry_schema["uri"]:
        raise ValueError("run provenance must bind the exact registry schema as an input")
    if schema_inputs[0].get("sha256") != registry_schema["sha256"]:
        raise ValueError("registry schema input hash disagrees with code provenance")
    _require_hex_digest(inputs.get("input_set_sha256"), field="input_set_sha256")
    if inputs["input_set_sha256"] != stable_hash(records):
        raise ValueError("input_set_sha256 does not match the exact ordered input records")

    runtime = payload["environment"].get("runtime")
    required_runtime = {
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "scipy_version",
        "tifffile_version",
        "pyyaml_version",
        "jsonschema_version",
    }
    if not isinstance(runtime, Mapping) or any(not runtime.get(key) for key in required_runtime):
        raise ValueError("run provenance runtime versions are incomplete")
    scientific = payload["scientific_boundary"]
    if any(
        scientific.get(key) is not False
        for key in ("scientific_completion", "scientific_promotion_allowed", "claim_bearing")
    ):
        raise ValueError("bounded run provenance must keep all scientific promotion flags false")
    return {
        "timestamps_complete_and_ordered": True,
        "duration_matches_timestamps": True,
        "git_state_complete": True,
        "registry_dirty_diff_schema_contract": True,
        "runtime_versions_complete": True,
        "command_and_mode_complete": True,
        "runner_hash_matches_live_module": True,
        "config_hash_matches_exact_config": True,
        "input_hash_set_complete": True,
        "scientific_boundary_closed": True,
    }


def validate_status_against_provenance(
    status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    provenance_sha256: str,
    runner_path: Path | None = None,
) -> dict[str, bool]:
    """Fail if status omits or disagrees with the authoritative run record."""
    _require_hex_digest(provenance_sha256, field="provenance_sha256")
    validate_run_provenance(provenance, runner_path=runner_path)
    if status.get("run_id") != provenance.get("run_id") or status.get("experiment_id") != provenance.get(
        "experiment_id"
    ):
        raise ValueError("status identity disagrees with run provenance")
    if status.get("lifecycle") != "succeeded" or status.get("status") != "succeeded_engineering_screen":
        raise ValueError("status must describe a succeeded engineering screen")
    for field in ("planned_at", "started_at", "ended_at", "duration_seconds"):
        if status.get(field) != provenance.get(field):
            raise ValueError(f"status {field} disagrees with run provenance")
    if status.get("execution_mode") != provenance["execution"]["mode"]:
        raise ValueError("status execution mode disagrees with run provenance")
    if status.get("command") != provenance["execution"]["command"]:
        raise ValueError("status command disagrees with run provenance")
    expected_hashes = {
        "runner_sha256": provenance["code"]["runner_sha256"],
        "config_sha256": provenance["configuration"]["config_sha256"],
        "resolved_config_sha256": provenance["configuration"]["resolved_config_sha256"],
        "input_set_sha256": provenance["inputs"]["input_set_sha256"],
    }
    if provenance["code"].get("diff_sha256") is not None:
        expected_hashes["diff_sha256"] = provenance["code"]["diff_sha256"]
    if status.get("hashes") != expected_hashes:
        raise ValueError("status hash anchors disagree with run provenance")
    if status.get("git") != provenance["code"]["git"] or status.get("runtime") != provenance["environment"][
        "runtime"
    ]:
        raise ValueError("status git/runtime mirror disagrees with run provenance")
    if status.get("provenance") != {
        "path": "run_provenance.json",
        "sha256": provenance_sha256,
    }:
        raise ValueError("status provenance pointer is absent or incorrect")
    if status.get("scientific_completion") is not False or status.get("scientific_promotion_allowed") is not False:
        raise ValueError("status must keep scientific completion and promotion false")
    return {
        "identity_matches": True,
        "timestamps_and_duration_match": True,
        "command_mode_and_hashes_match": True,
        "git_and_runtime_match": True,
        "provenance_pointer_matches": True,
        "scientific_boundary_closed": True,
    }


def validate_output_provenance_bundle(
    root: Path,
    *,
    runner_path: Path | None = None,
) -> dict[str, bool]:
    """Validate the three-file resolved-config/provenance/status bundle."""
    resolved_root = Path(root)
    resolved_config_path = resolved_root / "resolved_config.json"
    provenance_path = resolved_root / "run_provenance.json"
    status_path = resolved_root / "status.json"
    for path in (resolved_config_path, provenance_path, status_path):
        if not path.is_file():
            raise ValueError(f"required provenance bundle member is absent: {path.name}")
    resolved_config = _read_json(resolved_config_path)
    provenance = _read_json(provenance_path)
    status = _read_json(status_path)
    provenance_checks = validate_run_provenance(provenance, runner_path=runner_path)
    observed_resolved_sha = sha256_file(resolved_config_path)
    if observed_resolved_sha != provenance["configuration"]["resolved_config_sha256"]:
        raise ValueError("resolved_config_sha256 does not match resolved_config.json")
    if resolved_config.get("runner_module") != provenance["code"]["runner_module"]:
        raise ValueError("resolved config runner module disagrees with provenance")
    if resolved_config.get("runner_sha256") != provenance["code"]["runner_sha256"]:
        raise ValueError("resolved config runner hash disagrees with provenance")
    if resolved_config.get("config") != provenance["configuration"]["config"]:
        raise ValueError("resolved config scientific config disagrees with provenance")
    if resolved_config.get("config_sha256") != provenance["configuration"]["config_sha256"]:
        raise ValueError("resolved config config hash disagrees with provenance")
    if resolved_config.get("inputs") != provenance["inputs"]:
        raise ValueError("resolved config exact input inventory disagrees with provenance")
    execution = resolved_config.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("resolved config execution is absent")
    expected_execution = {
        "mode": provenance["execution"]["mode"],
        "command": provenance["execution"]["command"],
        "planned_at": provenance["planned_at"],
        "started_at": provenance["started_at"],
        "ended_at": provenance["ended_at"],
        "duration_seconds": provenance["duration_seconds"],
        "git": provenance["code"]["git"],
        "runtime": provenance["environment"]["runtime"],
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError("resolved config execution disagrees with provenance")
    status_checks = validate_status_against_provenance(
        status,
        provenance,
        provenance_sha256=sha256_file(provenance_path),
        runner_path=runner_path,
    )
    return {
        **{f"provenance_{key}": value for key, value in provenance_checks.items()},
        "resolved_config_hash_matches_file": True,
        "resolved_config_mirrors_provenance": True,
        **{f"status_{key}": value for key, value in status_checks.items()},
    }


def validate_artifact_index(root: Path) -> dict[str, bool]:
    """Fail closed unless the artifact index exactly covers every other file."""
    resolved_root = Path(root)
    index_path = resolved_root / "artifact_index.json"
    if not index_path.is_file():
        raise ValueError("artifact_index.json is absent")
    payload = _read_json(index_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact index requires an artifacts list")
    expected_paths = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in resolved_root.rglob("*")
        if path.is_file() and path.name != "artifact_index.json"
    )
    observed_paths = [str(row.get("path", "")) for row in artifacts]
    if observed_paths != expected_paths or len(observed_paths) != len(set(observed_paths)):
        raise ValueError("artifact index does not exactly cover the run files in sorted order")
    for row in artifacts:
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact index path must be portable and run-relative")
        path = resolved_root / relative
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise ValueError(f"artifact index hash/size mismatch: {relative.as_posix()}")
    if payload.get("artifact_count") != len(artifacts):
        raise ValueError("artifact index count mismatch")
    if payload.get("artifact_set_sha256") != stable_hash(artifacts):
        raise ValueError("artifact_set_sha256 mismatch")
    return {
        "exact_inventory": True,
        "all_sizes_and_hashes_match": True,
        "artifact_set_hash_matches": True,
        "paths_are_portable": True,
    }


def validate_frozen_window_manifest(
    payload: Mapping[str, Any],
    *,
    config: MotionRegistrationConfoundConfig,
) -> list[dict[str, Any]]:
    """Fail closed unless the exact expected grouped 12-window geometry exists."""
    rows = payload.get("windows")
    if not isinstance(rows, list):
        raise ValueError("background-window manifest requires a windows list")
    windows = [dict(row) for row in rows]
    if payload.get("window_count") != config.expected_window_count or len(windows) != config.expected_window_count:
        raise ValueError("background-window manifest must contain exactly twelve windows")
    ids = [str(row.get("background_window_id", "")) for row in windows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("background-window IDs must be nonempty and unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        required = {
            "background_dataset_id",
            "background_mad_stratum",
            "background_recording_id",
            "background_window_id",
            "height",
            "quiet_policy",
            "recording_id",
            "source_uri",
            "start_frame_zero",
            "stop_frame_zero_exclusive",
            "temporal_mad",
            "width",
            "x_zero",
            "y_zero",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"window {row.get('background_window_id')} missing fields: {missing}")
        if row["recording_id"] != row["background_recording_id"]:
            raise ValueError("window recording aliases disagree")
        if row["background_mad_stratum"] not in EXPECTED_STRATA:
            raise ValueError("unexpected temporal-MAD stratum")
        if int(row["stop_frame_zero_exclusive"]) - int(row["start_frame_zero"]) != config.expected_window_frames:
            raise ValueError("frozen window temporal extent changed")
        if (int(row["height"]), int(row["width"])) != config.expected_window_shape_yx:
            raise ValueError("frozen window spatial extent changed")
        if min(int(row["x_zero"]), int(row["y_zero"]), int(row["start_frame_zero"])) < 0:
            raise ValueError("frozen window coordinates must be nonnegative")
        if not np.isfinite(float(row["temporal_mad"])) or float(row["temporal_mad"]) < 0:
            raise ValueError("manifest temporal MAD must be finite and nonnegative")
        if not str(row["source_uri"]).startswith("data://"):
            raise ValueError("raw source URI must remain portable data://")
        grouped[str(row["recording_id"])].append(row)
    if len(grouped) != config.expected_recording_count:
        raise ValueError("frozen manifest must contain four recording groups")
    for recording_id, group in grouped.items():
        if len(group) != config.expected_windows_per_recording:
            raise ValueError(f"recording {recording_id} must contain three windows")
        if {str(row["background_mad_stratum"]) for row in group} != set(EXPECTED_STRATA):
            raise ValueError(f"recording {recording_id} is missing a frozen MAD stratum")
        for left, right in itertools.combinations(group, 2):
            temporal_overlap = max(int(left["start_frame_zero"]), int(right["start_frame_zero"])) < min(
                int(left["stop_frame_zero_exclusive"]), int(right["stop_frame_zero_exclusive"])
            )
            spatial_overlap = max(int(left["y_zero"]), int(right["y_zero"])) < min(
                int(left["y_zero"]) + int(left["height"]), int(right["y_zero"]) + int(right["height"])
            ) and max(int(left["x_zero"]), int(right["x_zero"])) < min(
                int(left["x_zero"]) + int(left["width"]), int(right["x_zero"]) + int(right["width"])
            )
            if temporal_overlap and spatial_overlap:
                raise ValueError(f"frozen windows overlap spatiotemporally within {recording_id}")
    return sorted(windows, key=lambda row: (str(row["recording_id"]), EXPECTED_STRATA.index(str(row["background_mad_stratum"]))))


def _descriptor_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("recordings")
    holdout = payload.get("external_evaluation_holdout")
    if not isinstance(rows, list) or not isinstance(holdout, Mapping):
        raise ValueError("data descriptor lacks recordings or external holdout")
    combined = [dict(row) for row in rows] + [dict(holdout)]
    lookup = {str(row["recording_id"]): row for row in combined}
    if len(lookup) != len(combined):
        raise ValueError("duplicate recording ID in data descriptor")
    return lookup


def _resolve_data_uri(uri: str, data_root: Path) -> Path:
    if not uri.startswith("data://"):
        raise ValueError(f"unsupported raw URI: {uri}")
    return (data_root / uri.removeprefix("data://")).resolve()


def _verify_parent_chain(
    *,
    repository_root: Path,
    parent_root: Path,
    parent_run_record_path: Path,
    parent_provenance_index_path: Path,
    referenced_parent_artifacts: Sequence[str],
    expected_run_id: str,
) -> dict[str, Any]:
    record = yaml.safe_load(parent_run_record_path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping) or record.get("id") != expected_run_id:
        raise ValueError("wrong frozen parent run record")
    expected_index_uri = str(record["artifacts"]["manifest_path"])
    expected_index_path = (repository_root / expected_index_uri).resolve()
    if expected_index_path != parent_provenance_index_path.resolve():
        raise ValueError("parent provenance index path disagrees with run record")
    expected_index_sha = str(record["artifacts"]["manifest_sha256"])
    provenance_index_sha = sha256_file(parent_provenance_index_path)
    live_index_path = parent_root / "artifact_index.json"
    live_index_sha = sha256_file(live_index_path)
    if provenance_index_sha != expected_index_sha or live_index_sha != expected_index_sha:
        raise ValueError("frozen parent artifact index hash mismatch")
    provenance_index = _read_json(parent_provenance_index_path)
    live_index = _read_json(live_index_path)
    if provenance_index != live_index:
        raise ValueError("live parent artifact index differs from committed provenance")
    indexed = {str(row["path"]): row for row in provenance_index.get("artifacts", [])}
    checks = []
    for relative in referenced_parent_artifacts:
        if relative not in indexed:
            raise ValueError(f"required parent artifact absent from frozen index: {relative}")
        path = parent_root / relative
        observed_sha = sha256_file(path)
        observed_bytes = path.stat().st_size
        expected = indexed[relative]
        if observed_sha != expected["sha256"] or observed_bytes != int(expected["bytes"]):
            raise ValueError(f"frozen parent artifact changed: {relative}")
        checks.append(
            {
                "path": relative,
                "sha256": observed_sha,
                "bytes": observed_bytes,
                "matches_committed_parent_index": True,
            }
        )
    return {
        "status": "passed",
        "parent_run_id": record["id"],
        "parent_run_record_uri": _portable_repository_path(parent_run_record_path, repository_root),
        "parent_run_record_sha256": sha256_file(parent_run_record_path),
        "parent_artifact_index_uri": _portable_repository_path(parent_provenance_index_path, repository_root),
        "parent_artifact_index_sha256": expected_index_sha,
        "live_parent_index_matches_committed_provenance": True,
        "referenced_artifacts": checks,
    }


def _verify_raw_sources(
    windows: Sequence[Mapping[str, Any]],
    *,
    descriptor_payload: Mapping[str, Any],
    data_root: Path,
    verify_hashes: bool,
) -> tuple[dict[str, np.memmap], dict[str, Any]]:
    lookup = _descriptor_rows(descriptor_payload)
    movies: dict[str, np.memmap] = {}
    checks = []
    for recording_id in sorted({str(row["recording_id"]) for row in windows}):
        if recording_id not in lookup:
            raise ValueError(f"frozen window recording is absent from descriptor: {recording_id}")
        descriptor = lookup[recording_id]
        uri = str(descriptor.get("portable_path", descriptor.get("uri", "")))
        manifest_uris = {str(row["source_uri"]) for row in windows if row["recording_id"] == recording_id}
        if manifest_uris != {uri}:
            raise ValueError(f"window source URI disagrees with descriptor for {recording_id}")
        path = _resolve_data_uri(uri, data_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha = sha256_file(path) if verify_hashes else None
        expected_sha = str(descriptor["sha256"])
        if verify_hashes and observed_sha != expected_sha:
            raise ValueError(f"raw recording hash mismatch: {recording_id}")
        movie = tifffile.memmap(path, mode="r")
        expected_shape = tuple(int(value) for value in descriptor["shape_tyx"])
        expected_dtype = np.dtype(str(descriptor["dtype"])).name
        if tuple(movie.shape) != expected_shape or movie.dtype.name != expected_dtype or movie.flags.writeable:
            raise ValueError(f"raw recording live contract mismatch: {recording_id}")
        for window in (row for row in windows if row["recording_id"] == recording_id):
            if int(window["stop_frame_zero_exclusive"]) > movie.shape[0]:
                raise ValueError(f"window exceeds raw time axis: {window['background_window_id']}")
            if int(window["y_zero"]) + int(window["height"]) > movie.shape[1]:
                raise ValueError(f"window exceeds raw y axis: {window['background_window_id']}")
            if int(window["x_zero"]) + int(window["width"]) > movie.shape[2]:
                raise ValueError(f"window exceeds raw x axis: {window['background_window_id']}")
        movies[recording_id] = movie
        checks.append(
            {
                "recording_id": recording_id,
                "source_uri": uri,
                "sha256": expected_sha,
                "bytes": path.stat().st_size,
                "hash_verified_live": verify_hashes,
                "shape_tyx": list(expected_shape),
                "dtype": expected_dtype,
                "memmap_read_only": True,
            }
        )
    return movies, {
        "status": "passed",
        "raw_video_copied": False,
        "raw_recording_count": len(checks),
        "hashes_verified": verify_hashes,
        "recordings": checks,
    }


def _tile_slices(shape_yx: tuple[int, int], grid_yx: tuple[int, int]) -> list[tuple[int, slice, slice]]:
    y_edges = np.linspace(0, shape_yx[0], grid_yx[0] + 1, dtype=np.int64)
    x_edges = np.linspace(0, shape_yx[1], grid_yx[1] + 1, dtype=np.int64)
    return [
        (
            row * grid_yx[1] + column,
            slice(int(y_edges[row]), int(y_edges[row + 1])),
            slice(int(x_edges[column]), int(x_edges[column + 1])),
        )
        for row in range(grid_yx[0])
        for column in range(grid_yx[1])
    ]


def _registration_arguments(config: MotionRegistrationConfoundConfig) -> dict[str, Any]:
    return {
        "spatial_stride": 1,
        "minimum_extent": config.minimum_registration_extent,
        "minimum_scale": config.minimum_robust_scale,
        "clip_z": config.registration_clip_z,
        "minimum_peak_to_median_ratio": config.minimum_peak_to_median_ratio,
        "maximum_translation_search_px": config.maximum_translation_search_px,
    }


def _matched_support_difference_diagnostics(
    reference: np.ndarray,
    moving: np.ndarray,
    shift_yx: Sequence[float],
    *,
    interpolation_order: int,
) -> dict[str, Any]:
    """Compare raw and registered change on one identical valid interior."""
    shift = np.asarray(shift_yx, dtype=np.float64)
    reference_float = np.asarray(reference, dtype=np.float64)
    moving_float = np.asarray(moving, dtype=np.float64)
    aligned = ndimage.shift(
        moving_float,
        shift=tuple(float(value) for value in shift),
        order=interpolation_order,
        mode="nearest",
        prefilter=interpolation_order > 1,
    )
    margin = int(math.ceil(float(np.max(np.abs(shift))))) + 2
    if 2 * margin >= min(reference.shape):
        raise ValueError("registration shift leaves no stable residual interior")
    bounds = [margin, int(reference.shape[0]) - margin, margin, int(reference.shape[1]) - margin]
    interior = (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]))
    reference_interior = reference_float[interior]
    raw_difference = moving_float[interior] - reference_interior
    registered_difference = aligned[interior] - reference_interior
    pixel_count = int(raw_difference.size)
    if pixel_count != int(registered_difference.size) or pixel_count < 1:
        raise ValueError("raw and registered residual supports must be identical and nonempty")
    return {
        "raw_difference_mad_matched_support": _normal_consistent_mad(raw_difference),
        "raw_difference_rms_matched_support": float(np.sqrt(np.mean(np.square(raw_difference)))),
        "registered_residual_mad": _normal_consistent_mad(registered_difference),
        "registered_residual_rms": float(np.sqrt(np.mean(np.square(registered_difference)))),
        "matched_support_margin_px": margin,
        "matched_support_bounds_yx_zero_half_open": bounds,
        "matched_support_pixel_count": pixel_count,
        "matched_support_pixel_fraction": float(pixel_count / reference_float.size),
        "registered_raw_mad_support_contract": "exact_same_shift_valid_interior",
    }


def compute_pair_diagnostics(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    config: MotionRegistrationConfoundConfig,
) -> dict[str, Any]:
    """Compute bidirectional global/tile translation and residual diagnostics."""
    if np.shape(reference) != np.shape(moving) or np.ndim(reference) != 2:
        raise ValueError("adjacent pair must contain two same-shaped 2-D frames")
    arguments = _registration_arguments(config)
    forward = estimate_phase_translation(reference, moving, **arguments)
    backward = estimate_phase_translation(moving, reference, **arguments)
    raw_difference = np.asarray(moving, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    raw_mad_full_frame = _normal_consistent_mad(raw_difference)
    raw_rms_full_frame = float(np.sqrt(np.mean(np.square(raw_difference))))
    bidirectional_valid = bool(forward["valid"] and backward["valid"])
    closure = None
    matched = None
    if bidirectional_valid:
        closure = float(
            np.linalg.norm(
                np.asarray(forward["registration_shift_yx_px"], dtype=np.float64)
                + np.asarray(backward["registration_shift_yx_px"], dtype=np.float64)
            )
        )
    if forward["valid"]:
        matched = _matched_support_difference_diagnostics(
            reference,
            moving,
            forward["registration_shift_yx_px"],
            interpolation_order=config.registration_residual_interpolation_order,
        )
    raw_mad_matched = None if matched is None else float(matched["raw_difference_mad_matched_support"])
    registered_mad = None if matched is None else float(matched["registered_residual_mad"])
    ratio = (
        None
        if registered_mad is None
        or raw_mad_matched is None
        or raw_mad_matched <= config.minimum_robust_scale
        else registered_mad / raw_mad_matched
    )
    near_search_boundary = bool(
        forward["valid"]
        and max(abs(float(value)) for value in forward["registration_shift_yx_px"])
        >= 0.875 * config.maximum_translation_search_px
    )

    tile_rows = []
    for tile_index, y_slice, x_slice in _tile_slices(tuple(reference.shape), config.tile_grid_yx):
        tile_forward = estimate_phase_translation(reference[y_slice, x_slice], moving[y_slice, x_slice], **arguments)
        tile_backward = estimate_phase_translation(moving[y_slice, x_slice], reference[y_slice, x_slice], **arguments)
        tile_closure = None
        global_disagreement = None
        if tile_forward["valid"] and tile_backward["valid"]:
            tile_closure = float(
                np.linalg.norm(
                    np.asarray(tile_forward["registration_shift_yx_px"], dtype=np.float64)
                    + np.asarray(tile_backward["registration_shift_yx_px"], dtype=np.float64)
                )
            )
        if tile_forward["valid"] and forward["valid"]:
            global_disagreement = float(
                np.linalg.norm(
                    np.asarray(tile_forward["registration_shift_yx_px"], dtype=np.float64)
                    - np.asarray(forward["registration_shift_yx_px"], dtype=np.float64)
                )
            )
        tile_rows.append(
            {
                "tile_index": tile_index,
                "bounds_yx_zero_half_open": [y_slice.start, y_slice.stop, x_slice.start, x_slice.stop],
                "forward_valid": bool(tile_forward["valid"]),
                "backward_valid": bool(tile_backward["valid"]),
                "forward_shift_y_px": (
                    None if not tile_forward["valid"] else float(tile_forward["registration_shift_yx_px"][0])
                ),
                "forward_shift_x_px": (
                    None if not tile_forward["valid"] else float(tile_forward["registration_shift_yx_px"][1])
                ),
                "forward_backward_closure_px": tile_closure,
                "forward_to_global_disagreement_px": global_disagreement,
            }
        )
    return {
        "raw_difference_mad": raw_mad_full_frame,
        "raw_difference_rms": raw_rms_full_frame,
        "raw_difference_support_contract": "full_frame_standalone_diagnostic_not_ratio_denominator",
        "raw_difference_mad_matched_support": raw_mad_matched,
        "raw_difference_rms_matched_support": (
            None if matched is None else float(matched["raw_difference_rms_matched_support"])
        ),
        "forward_valid": bool(forward["valid"]),
        "backward_valid": bool(backward["valid"]),
        "bidirectional_valid": bidirectional_valid,
        "forward_shift_y_px": None if not forward["valid"] else float(forward["registration_shift_yx_px"][0]),
        "forward_shift_x_px": None if not forward["valid"] else float(forward["registration_shift_yx_px"][1]),
        "forward_shift_magnitude_px": None if not forward["valid"] else float(forward["magnitude_px"]),
        "forward_near_search_boundary": near_search_boundary,
        "forward_peak_to_median_ratio": _nullable(forward.get("peak_to_median_ratio")),
        "backward_shift_y_px": None if not backward["valid"] else float(backward["registration_shift_yx_px"][0]),
        "backward_shift_x_px": None if not backward["valid"] else float(backward["registration_shift_yx_px"][1]),
        "forward_backward_closure_px": closure,
        "registered_residual_mad": registered_mad,
        "registered_residual_rms": None if matched is None else float(matched["registered_residual_rms"]),
        "registered_residual_margin_px": None if matched is None else int(matched["matched_support_margin_px"]),
        "matched_support_margin_px": None if matched is None else int(matched["matched_support_margin_px"]),
        "matched_support_bounds_yx_zero_half_open": (
            None if matched is None else list(matched["matched_support_bounds_yx_zero_half_open"])
        ),
        "matched_support_pixel_count": None if matched is None else int(matched["matched_support_pixel_count"]),
        "matched_support_pixel_fraction": (
            None if matched is None else float(matched["matched_support_pixel_fraction"])
        ),
        "registered_raw_mad_support_contract": (
            None if matched is None else str(matched["registered_raw_mad_support_contract"])
        ),
        "registered_over_raw_difference_mad_ratio_denominator": "raw_difference_mad_matched_support",
        "registered_over_raw_difference_mad_ratio": ratio,
        "registered_difference_reduction_fraction": None if ratio is None else 1.0 - ratio,
        "tile_count": len(tile_rows),
        "tile_forward_valid_fraction": float(np.mean([row["forward_valid"] for row in tile_rows])),
        "tile_bidirectional_valid_fraction": float(
            np.mean([row["forward_valid"] and row["backward_valid"] for row in tile_rows])
        ),
        "tile_forward_backward_closure": _finite_summary(
            row["forward_backward_closure_px"] for row in tile_rows
        ),
        "tile_global_disagreement": _finite_summary(
            row["forward_to_global_disagreement_px"] for row in tile_rows
        ),
        "tile_rows": tile_rows,
        "forward_backward_interpretation": (
            "reciprocal phase correlation is structurally conjugate and therefore checks implementation symmetry, "
            "not independent biological reliability"
        ),
    }


def _pair_matched_support_contract_valid(
    row: Mapping[str, Any],
    *,
    config: MotionRegistrationConfoundConfig,
) -> bool:
    matched_fields = (
        "raw_difference_mad_matched_support",
        "registered_residual_mad",
        "matched_support_margin_px",
        "matched_support_bounds_yx_zero_half_open",
        "matched_support_pixel_count",
        "matched_support_pixel_fraction",
        "registered_raw_mad_support_contract",
    )
    if not row.get("forward_valid"):
        return all(row.get(field) is None for field in matched_fields) and row.get(
            "registered_over_raw_difference_mad_ratio"
        ) is None
    if any(row.get(field) is None for field in matched_fields):
        return False
    if row.get("registered_raw_mad_support_contract") != "exact_same_shift_valid_interior":
        return False
    if row.get("registered_over_raw_difference_mad_ratio_denominator") != "raw_difference_mad_matched_support":
        return False
    if row.get("raw_difference_support_contract") != "full_frame_standalone_diagnostic_not_ratio_denominator":
        return False
    margin = int(row["matched_support_margin_px"])
    height, width = config.expected_window_shape_yx
    expected_bounds = [margin, height - margin, margin, width - margin]
    if list(row["matched_support_bounds_yx_zero_half_open"]) != expected_bounds:
        return False
    expected_pixels = (height - 2 * margin) * (width - 2 * margin)
    if int(row["matched_support_pixel_count"]) != expected_pixels:
        return False
    if not math.isclose(
        float(row["matched_support_pixel_fraction"]),
        expected_pixels / (height * width),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        return False
    raw_matched = float(row["raw_difference_mad_matched_support"])
    registered = float(row["registered_residual_mad"])
    ratio = row.get("registered_over_raw_difference_mad_ratio")
    if raw_matched <= config.minimum_robust_scale:
        return ratio is None
    return ratio is not None and math.isclose(
        float(ratio),
        registered / raw_matched,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _sensor_diagnostics(clip: np.ndarray) -> dict[str, Any]:
    dtype = clip.dtype
    values = np.asarray(clip)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low_count = int(np.count_nonzero(values == info.min))
        high_count = int(np.count_nonzero(values == info.max))
        digital = {
            "status": "resolved_for_stored_integer_dtype_only",
            "stored_dtype": dtype.name,
            "stored_low_code": int(info.min),
            "stored_high_code": int(info.max),
            "observed_minimum": int(np.min(values)),
            "observed_maximum": int(np.max(values)),
            "low_code_count": low_count,
            "high_code_count": high_count,
            "low_code_fraction": float(low_count / values.size),
            "high_code_fraction": float(high_count / values.size),
        }
    else:
        digital = {
            "status": "unresolved_noninteger_storage",
            "stored_dtype": dtype.name,
            "stored_low_code": None,
            "stored_high_code": None,
            "observed_minimum": float(np.min(values)),
            "observed_maximum": float(np.max(values)),
            "low_code_count": None,
            "high_code_count": None,
            "low_code_fraction": None,
            "high_code_fraction": None,
        }
    return {
        "stored_digital_codes": digital,
        "analog_sensor_rail": {
            "status": "unresolved",
            "reason": "ADC_bit_depth_detector_black_level_gain_and_clipping_metadata_absent",
            "interpretation": "No stored dtype-rail hit does not rule out analog or upstream digital saturation.",
        },
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3 or np.unique(left[finite]).size < 2 or np.unique(right[finite]).size < 2:
        return None
    return float(stats.spearmanr(left[finite], right[finite]).statistic)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3 or np.ptp(left[finite]) == 0 or np.ptp(right[finite]) == 0:
        return None
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def analyze_window(
    window: Mapping[str, Any],
    movie: np.ndarray,
    *,
    config: MotionRegistrationConfoundConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = int(window["start_frame_zero"])
    stop = int(window["stop_frame_zero_exclusive"])
    y_zero = int(window["y_zero"])
    x_zero = int(window["x_zero"])
    height = int(window["height"])
    width = int(window["width"])
    clip = np.asarray(movie[start:stop, y_zero : y_zero + height, x_zero : x_zero + width])
    expected_shape = (config.expected_window_frames, *config.expected_window_shape_yx)
    if tuple(clip.shape) != expected_shape:
        raise ValueError(f"live frozen-window shape mismatch: {window['background_window_id']}")
    sampled = clip[
        :: config.temporal_mad_time_stride,
        :: config.temporal_mad_spatial_stride,
        :: config.temporal_mad_spatial_stride,
    ].astype(np.float32, copy=False)
    sampled_center = np.median(sampled, axis=0, keepdims=True)
    reconstructed_mad = float(1.4826 * np.median(np.abs(sampled - sampled_center)))
    full = clip.astype(np.float32, copy=False)
    full_center = np.median(full, axis=0, keepdims=True)
    full_mad = float(1.4826 * np.median(np.abs(full - full_center)))

    pair_rows = []
    for local_index in range(clip.shape[0] - 1):
        pair = compute_pair_diagnostics(clip[local_index], clip[local_index + 1], config=config)
        tile_closure = pair.pop("tile_forward_backward_closure")
        tile_disagreement = pair.pop("tile_global_disagreement")
        pair.pop("tile_rows")
        pair_rows.append(
            {
                "background_window_id": str(window["background_window_id"]),
                "recording_id": str(window["recording_id"]),
                "background_mad_stratum": str(window["background_mad_stratum"]),
                "reference_frame_zero": start + local_index,
                "moving_frame_zero": start + local_index + 1,
                **pair,
                "tile_forward_backward_closure_median_px": tile_closure["median"],
                "tile_forward_backward_closure_p95_px": tile_closure["p95"],
                "tile_global_disagreement_median_px": tile_disagreement["median"],
                "tile_global_disagreement_p95_px": tile_disagreement["p95"],
            }
        )
    two_step_cycle_rows = []
    arguments = _registration_arguments(config)
    for local_index in range(clip.shape[0] - 2):
        first = pair_rows[local_index]
        second = pair_rows[local_index + 1]
        direct = estimate_phase_translation(clip[local_index], clip[local_index + 2], **arguments)
        valid = bool(first["forward_valid"] and second["forward_valid"] and direct["valid"])
        closure = None
        if valid:
            composed = np.asarray(
                [
                    float(first["forward_shift_y_px"]) + float(second["forward_shift_y_px"]),
                    float(first["forward_shift_x_px"]) + float(second["forward_shift_x_px"]),
                ],
                dtype=np.float64,
            )
            closure = float(
                np.linalg.norm(
                    np.asarray(direct["registration_shift_yx_px"], dtype=np.float64) - composed
                )
            )
        two_step_cycle_rows.append(
            {
                "start_frame_zero": start + local_index,
                "valid": valid,
                "closure_px": closure,
            }
        )
    sensor = _sensor_diagnostics(clip)
    global_valid = [row for row in pair_rows if row["forward_valid"]]
    bidirectional_valid = [row for row in pair_rows if row["bidirectional_valid"]]
    frame_medians = np.median(full, axis=(1, 2))
    frame_index = np.arange(len(frame_medians), dtype=np.float64)
    brightness_slope = float(stats.theilslopes(frame_medians, frame_index).slope)
    brightness_relative_change = brightness_slope * (len(frame_medians) - 1) / max(
        abs(float(np.median(frame_medians))), config.minimum_robust_scale
    )
    shift_change_spearman = _spearman(
        [float(row["forward_shift_magnitude_px"]) for row in global_valid],
        [float(row["raw_difference_mad"]) for row in global_valid],
    )
    two_step_summary = _finite_summary(row["closure_px"] for row in two_step_cycle_rows)
    boundary_fraction = float(np.mean([row["forward_near_search_boundary"] for row in pair_rows]))
    tile_disagreement_p95 = _finite_summary(
        row["tile_global_disagreement_p95_px"] for row in pair_rows
    )["p95"]
    registered_ratio = _finite_summary(
        row["registered_over_raw_difference_mad_ratio"] for row in pair_rows
    )["median"]
    reliability_reasons = []
    if len(global_valid) / len(pair_rows) < 0.80:
        reliability_reasons.append("low_global_pair_valid_fraction")
    if boundary_fraction > config.search_boundary_component_fraction_review:
        reliability_reasons.append("search_boundary_concentration")
    if tile_disagreement_p95 is None or float(tile_disagreement_p95) > config.tile_global_disagreement_review_px:
        reliability_reasons.append("large_tile_to_global_disagreement")
    if registered_ratio is None or 1.0 - float(registered_ratio) < config.minimum_registration_difference_reduction_fraction:
        reliability_reasons.append("small_registered_difference_reduction")
    summary = {
        "background_window_id": str(window["background_window_id"]),
        "background_dataset_id": str(window["background_dataset_id"]),
        "recording_id": str(window["recording_id"]),
        "background_mad_stratum": str(window["background_mad_stratum"]),
        "quiet_policy": str(window["quiet_policy"]),
        "start_frame_zero": start,
        "stop_frame_zero_exclusive": stop,
        "y_zero": y_zero,
        "x_zero": x_zero,
        "height": height,
        "width": width,
        "manifest_temporal_mad": float(window["temporal_mad"]),
        "recomputed_sampling_contract_temporal_mad": reconstructed_mad,
        "recomputed_full_resolution_temporal_mad": full_mad,
        "temporal_mad_absolute_error": abs(reconstructed_mad - float(window["temporal_mad"])),
        "pair_count": len(pair_rows),
        "global_forward_valid_fraction": float(len(global_valid) / len(pair_rows)),
        "forward_backward_valid_fraction": float(len(bidirectional_valid) / len(pair_rows)),
        "global_translation_median_px": _finite_summary(
            row["forward_shift_magnitude_px"] for row in pair_rows
        )["median"],
        "global_translation_p95_px": _finite_summary(
            row["forward_shift_magnitude_px"] for row in pair_rows
        )["p95"],
        "forward_backward_closure_median_px": _finite_summary(
            row["forward_backward_closure_px"] for row in pair_rows
        )["median"],
        "forward_backward_closure_p95_px": _finite_summary(
            row["forward_backward_closure_px"] for row in pair_rows
        )["p95"],
        "forward_backward_consistency_status": "structurally_symmetric_not_independent",
        "two_step_cycle_valid_fraction": float(
            np.mean([row["valid"] for row in two_step_cycle_rows])
        ),
        "two_step_cycle_closure_median_px": two_step_summary["median"],
        "two_step_cycle_closure_p95_px": two_step_summary["p95"],
        "search_boundary_pair_fraction": boundary_fraction,
        "tile_forward_valid_fraction_median": _finite_summary(
            row["tile_forward_valid_fraction"] for row in pair_rows
        )["median"],
        "tile_bidirectional_valid_fraction_median": _finite_summary(
            row["tile_bidirectional_valid_fraction"] for row in pair_rows
        )["median"],
        "tile_global_disagreement_median_px": _finite_summary(
            row["tile_global_disagreement_median_px"] for row in pair_rows
        )["median"],
        "tile_global_disagreement_p95_px": tile_disagreement_p95,
        "raw_difference_mad_median": _finite_summary(row["raw_difference_mad"] for row in pair_rows)["median"],
        "raw_difference_mad_support_contract": "full_frame_standalone_diagnostic_not_ratio_denominator",
        "raw_difference_mad_matched_support_median": _finite_summary(
            row["raw_difference_mad_matched_support"] for row in pair_rows
        )["median"],
        "registered_residual_mad_median": _finite_summary(
            row["registered_residual_mad"] for row in pair_rows
        )["median"],
        "registered_raw_mad_support_contract": "exact_same_shift_valid_interior_per_pair",
        "matched_support_pixel_fraction_median": _finite_summary(
            row["matched_support_pixel_fraction"] for row in pair_rows
        )["median"],
        "matched_support_pixel_fraction_minimum": _finite_summary(
            row["matched_support_pixel_fraction"] for row in pair_rows
        )["minimum"],
        "registered_over_raw_difference_mad_ratio": registered_ratio,
        "registered_difference_reduction_fraction_median": _finite_summary(
            row["registered_difference_reduction_fraction"] for row in pair_rows
        )["median"],
        "motion_raw_change_spearman_within_window": shift_change_spearman,
        "frame_median_relative_change_theil_sen": float(brightness_relative_change),
        "stored_dtype": sensor["stored_digital_codes"]["stored_dtype"],
        "stored_low_code_fraction": sensor["stored_digital_codes"]["low_code_fraction"],
        "stored_high_code_fraction": sensor["stored_digital_codes"]["high_code_fraction"],
        "observed_minimum_stored_code": sensor["stored_digital_codes"]["observed_minimum"],
        "observed_maximum_stored_code": sensor["stored_digital_codes"]["observed_maximum"],
        "analog_sensor_rail_status": sensor["analog_sensor_rail"]["status"],
        "analog_sensor_rail_reason": sensor["analog_sensor_rail"]["reason"],
        "local_translation_reliability_status": (
            "review_required" if reliability_reasons else "screen_clear"
        ),
        "local_translation_reliability_reasons": reliability_reasons,
        "forward_backward_review_trigger": bool(
            len(bidirectional_valid) / len(pair_rows) < config.minimum_forward_backward_valid_fraction
        ),
    }
    return summary, pair_rows


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate_parent_endpoints(
    raw_rows: Sequence[Mapping[str, str]],
    residual_rows: Sequence[Mapping[str, str]],
    *,
    window_ids: set[str],
) -> dict[str, dict[str, Any]]:
    def residual_value(row: Mapping[str, Any], name: str) -> float:
        if name in row:
            return float(row[name])
        if name in {"background_rms_ratio", "dynamic_mad_ratio"}:
            return float(row["background"][name])
        if name == "seam_to_interior_jump_ratio":
            return float(row["seams"][name])
        raise KeyError(name)

    raw = [row for row in raw_rows if row.get("method") == "raw_frozen_handcrafted_stack"]
    if len(raw) != 108:
        raise ValueError("frozen parent must contain exactly 108 raw-HC rows")
    residual = [row for row in residual_rows if row.get("method") in EXPECTED_PARENT_METHODS]
    if len(residual) != 216:
        raise ValueError("frozen parent must contain exactly 216 conditional-residual diagnostic rows")
    if {row["background_window_id"] for row in raw} != window_ids:
        raise ValueError("raw-HC parent window IDs differ from frozen manifest")
    if {row["background_window_id"] for row in residual} != window_ids:
        raise ValueError("residual parent window IDs differ from frozen manifest")
    result: dict[str, dict[str, Any]] = {}
    for window_id in sorted(window_ids):
        window_raw = [row for row in raw if row["background_window_id"] == window_id]
        if len(window_raw) != 9:
            raise ValueError(f"raw-HC window {window_id} does not contain nine frozen fixture cells")
        fixtures = {row["fixture_id"] for row in window_raw}
        if len(fixtures) != 9:
            raise ValueError(f"raw-HC window {window_id} contains duplicate fixture IDs")
        source_on_recovered = sum(int(row["source_on_recovered"]) for row in window_raw)
        injected = sum(int(row["injected_sources"]) for row in window_raw)
        intervention_recovered = sum(
            round(float(row["intervention_recall"]) * int(row["injected_sources"])) for row in window_raw
        )
        source_count_groups: dict[int, list[float]] = defaultdict(list)
        for row in window_raw:
            source_count_groups[int(row["source_count"])].append(float(row["source_on_recall"]))
        if set(source_count_groups) != {1, 2, 4} or any(len(values) != 3 for values in source_count_groups.values()):
            raise ValueError("raw-HC fixture grid must contain three seeds for source counts 1, 2, and 4")
        seed_sd_mean = _mean([float(np.std(values, ddof=0)) for values in source_count_groups.values()])
        seed_range_mean = _mean([max(values) - min(values) for values in source_count_groups.values()])
        joined: dict[str, Any] = {
            "raw_hc_source_on_micro_recall": float(source_on_recovered / injected),
            "raw_hc_intervention_micro_recall": float(intervention_recovered / injected),
            "raw_hc_source_on_macro_recall": _mean([float(row["source_on_recall"]) for row in window_raw]),
            "raw_hc_intervention_macro_recall": _mean([float(row["intervention_recall"]) for row in window_raw]),
            "raw_hc_source_on_seed_sd_mean": seed_sd_mean,
            "raw_hc_source_on_seed_range_mean": seed_range_mean,
            "raw_hc_center_delta_median": _median(
                [float(row["median_center_delta_background_mad"]) for row in window_raw]
            ),
            "raw_hc_fixture_count": len(window_raw),
            "raw_hc_injected_source_count": injected,
        }
        for method, prefix in (
            (EXPECTED_PARENT_METHODS[0], "jepa"),
            (EXPECTED_PARENT_METHODS[1], "random"),
        ):
            selected = [
                row for row in residual if row["background_window_id"] == window_id and row["method"] == method
            ]
            if len(selected) != 9 or {row["fixture_id"] for row in selected} != fixtures:
                raise ValueError(f"{prefix} residual grid differs from raw-HC fixtures for {window_id}")
            joined[f"{prefix}_background_rms_ratio"] = _median(
                [residual_value(row, "background_rms_ratio") for row in selected]
            )
            joined[f"{prefix}_dynamic_mad_ratio"] = _median(
                [residual_value(row, "dynamic_mad_ratio") for row in selected]
            )
            joined[f"{prefix}_seam_to_interior_jump_ratio"] = _median(
                [residual_value(row, "seam_to_interior_jump_ratio") for row in selected]
            )
        result[window_id] = joined
    return result


def _rank_within_groups(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    ranked = np.empty_like(values, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    for group in sorted(set(groups)):
        mask = group_array == group
        local = stats.rankdata(values[mask], method="average")
        ranked[mask] = local - float(np.mean(local))
    return ranked


def grouped_exact_association(
    rows: Sequence[Mapping[str, Any]],
    *,
    predictor: str,
    outcome: str,
    group_key: str = "recording_id",
) -> dict[str, Any]:
    """Descriptive association plus an exact within-recording permutation test."""
    if "registered_over_raw_difference_mad_ratio" in {predictor, outcome}:
        difference_mad_support_contract = "ratio_uses_exact_same_shift_valid_interior_per_pair"
    elif "raw_difference_mad_median" in {predictor, outcome}:
        difference_mad_support_contract = "full_frame_raw_difference_standalone_association"
    else:
        difference_mad_support_contract = "not_applicable"
    selected = [
        row
        for row in rows
        if row.get(predictor) is not None
        and row.get(outcome) is not None
        and np.isfinite(float(row[predictor]))
        and np.isfinite(float(row[outcome]))
    ]
    groups = [str(row[group_key]) for row in selected]
    x = np.asarray([float(row[predictor]) for row in selected], dtype=np.float64)
    y = np.asarray([float(row[outcome]) for row in selected], dtype=np.float64)
    unique_groups = sorted(set(groups))
    if len(selected) < 6 or len(unique_groups) < 2:
        return {
            "predictor": predictor,
            "outcome": outcome,
            "difference_mad_support_contract": difference_mad_support_contract,
            "n_windows": len(selected),
            "n_recordings": len(unique_groups),
            "spearman_all_windows": _spearman(x, y),
            "within_recording_rank_correlation": None,
            "grouped_exact_two_sided_p": None,
            "grouped_permutation_count": 0,
            "loro_spearman_min": None,
            "loro_spearman_max": None,
            "loro_sign_consistent": False,
            "status": "insufficient_grouped_support",
        }
    x_rank = _rank_within_groups(x, groups)
    y_rank = _rank_within_groups(y, groups)
    observed = _pearson(x_rank, y_rank)
    group_indices = [np.flatnonzero(np.asarray(groups, dtype=object) == group) for group in unique_groups]
    permutation_sets = [tuple(itertools.permutations(indices.tolist())) for indices in group_indices]
    permuted_values = []
    if observed is not None:
        for choices in itertools.product(*permutation_sets):
            permuted_y = y.copy()
            for destination, source in zip(group_indices, choices):
                permuted_y[destination] = y[np.asarray(source, dtype=np.int64)]
            permuted_rank = _rank_within_groups(permuted_y, groups)
            value = _pearson(x_rank, permuted_rank)
            if value is not None:
                permuted_values.append(float(value))
    exact_p = None
    if observed is not None and permuted_values:
        exact_p = float(np.mean(np.abs(permuted_values) >= abs(observed) - 1e-12))
    loro = []
    group_array = np.asarray(groups, dtype=object)
    for held_out in unique_groups:
        mask = group_array != held_out
        value = _spearman(x[mask], y[mask])
        if value is not None:
            loro.append(value)
    nonzero_signs = {int(np.sign(value)) for value in loro if abs(value) > 1e-12}
    return {
        "predictor": predictor,
        "outcome": outcome,
        "difference_mad_support_contract": difference_mad_support_contract,
        "n_windows": len(selected),
        "n_recordings": len(unique_groups),
        "spearman_all_windows": _spearman(x, y),
        "within_recording_rank_correlation": observed,
        "grouped_exact_two_sided_p": exact_p,
        "grouped_permutation_count": len(permuted_values),
        "loro_spearman_min": None if not loro else float(min(loro)),
        "loro_spearman_max": None if not loro else float(max(loro)),
        "loro_sign_consistent": bool(loro and len(nonzero_signs) <= 1),
        "status": "descriptive_grouped_screen_complete",
    }


def adjust_grouped_association_family(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add Benjamini-Hochberg q values across the frozen association family."""
    output = [dict(row) for row in rows]
    finite = [
        (index, float(row["grouped_exact_two_sided_p"]))
        for index, row in enumerate(output)
        if row.get("grouped_exact_two_sided_p") is not None
        and np.isfinite(float(row["grouped_exact_two_sided_p"]))
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 1.0
    family_size = len(ordered)
    for reverse_index in range(family_size - 1, -1, -1):
        original_index, probability = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, probability * family_size / rank)
        adjusted[original_index] = float(min(1.0, running))
    for index, row in enumerate(output):
        q_value = adjusted.get(index)
        row["grouped_exact_bh_q"] = q_value
        row["passes_bh_0_05"] = bool(q_value is not None and q_value <= 0.05)
        row["association_family_size"] = family_size
    return output


def _recording_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    numeric = (
        "manifest_temporal_mad",
        "global_translation_p95_px",
        "forward_backward_closure_p95_px",
        "two_step_cycle_closure_p95_px",
        "search_boundary_pair_fraction",
        "tile_global_disagreement_p95_px",
        "raw_difference_mad_median",
        "raw_difference_mad_matched_support_median",
        "registered_over_raw_difference_mad_ratio",
        "raw_hc_source_on_micro_recall",
        "raw_hc_source_on_seed_sd_mean",
        "jepa_background_rms_ratio",
        "jepa_dynamic_mad_ratio",
        "jepa_seam_to_interior_jump_ratio",
        "local_window_to_full_field_p95_ratio",
    )
    for recording_id in sorted({str(row["recording_id"]) for row in rows}):
        selected = [row for row in rows if row["recording_id"] == recording_id]
        result: dict[str, Any] = {
            "recording_id": recording_id,
            "background_dataset_id": selected[0]["background_dataset_id"],
            "window_count": len(selected),
            "mad_strata": ";".join(sorted(str(row["background_mad_stratum"]) for row in selected)),
            "forward_backward_review_window_count": sum(bool(row["forward_backward_review_trigger"]) for row in selected),
            "local_translation_reliability_review_window_count": sum(
                row["local_translation_reliability_status"] == "review_required" for row in selected
            ),
        }
        for field in numeric:
            values = [float(row[field]) for row in selected if row.get(field) is not None]
            result[field + "_median_across_windows"] = None if not values else _median(values)
        output.append(result)
    return output


def _feature_portability(
    *,
    repository_root: Path,
    windows: Sequence[Mapping[str, Any]],
    feature_panel_root: Path,
    trace_atlas_root: Path,
) -> dict[str, Any]:
    feature_files = [
        feature_panel_root / "summary.json",
        feature_panel_root / "tables" / "occurrence_feature_metrics.tsv",
        trace_atlas_root / "traces.npz",
        trace_atlas_root / "llm_context.json",
    ]
    for path in feature_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    spon_windows = [row for row in windows if row["background_dataset_id"] == "spon_ca_burst"]
    supported_start_zero = 1799
    supported_stop_zero_exclusive = 2359
    overlaps = [
        row["background_window_id"]
        for row in spon_windows
        if max(int(row["start_frame_zero"]), supported_start_zero)
        < min(int(row["stop_frame_zero_exclusive"]), supported_stop_zero_exclusive)
    ]
    if overlaps:
        raise ValueError("unexpected overlap requires a separately frozen ROI/feature alignment protocol")
    return {
        "status": "not_joined_no_compatible_recording_and_frame_support",
        "reason": (
            "Current validated carrier/coherence/recurrence trace arrays are finite only for Spon UI frames "
            "1800-2359; all three frozen Spon EXP-0029 windows end before that range, and 060126 lacks "
            "matched exported feature traces. Forcing a join would change the estimand."
        ),
        "portable_feature_lanes": ["carrier_signed", "coherence_w15", "propagation_lag2_w15"],
        "spon_supported_numpy_interval_zero_half_open": [supported_start_zero, supported_stop_zero_exclusive],
        "overlapping_frozen_window_ids": overlaps,
        "artifacts": [
            {
                "uri": _portable_repository_path(path, repository_root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in feature_files
        ],
        "interpretation": "Feature non-join is an explicit compatibility result, not missing-data imputation.",
    }


def _resource_preflight(output_root: Path) -> dict[str, Any]:
    disk_anchor = output_root.parent
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    disk = shutil.disk_usage(disk_anchor)
    memory_available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in meminfo.read_text(encoding="utf-8").splitlines()
            if ":" in line
        }
        if "MemAvailable" in fields:
            memory_available = int(fields["MemAvailable"].split()[0]) * 1024
    return {
        "output_root_absent": not output_root.exists(),
        "partial_root_absent": not output_root.with_name(output_root.name + ".partial").exists(),
        "free_disk_bytes": int(disk.free),
        "minimum_free_disk_bytes": 1_000_000_000,
        "free_disk_passed": disk.free >= 1_000_000_000,
        "available_memory_bytes": memory_available,
        "minimum_available_memory_bytes": 2_000_000_000,
        "available_memory_passed": memory_available is None or memory_available >= 2_000_000_000,
        "gpu_required": False,
        "raw_copy_forbidden": True,
        "bounded_workload": "12 windows x 31 adjacent pairs x 1 global and 4 tile bidirectional registrations",
    }


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_index.json"
    ]
    payload = {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts}
    payload["artifact_set_sha256"] = stable_hash(artifacts)
    return payload


def _report(summary: Mapping[str, Any]) -> str:
    key = summary["key_results"]
    return f"""# NREV-EXP-0025 bounded motion and registration-confound audit

## Outcome

The bounded engineering screen completed on the exact twelve frozen NREV-EXP-0029 background windows. It is label-free and non-claim-bearing. It does not motion-correct the recordings or identify biological motion.

- Windows: {summary['scope']['window_count']} across {summary['scope']['recording_count']} recording groups.
- Adjacent-frame pairs: {summary['scope']['adjacent_pair_count']}.
- Median global translation across windows: {key['median_window_global_translation_median_px']:.6g} native pixels.
- Maximum window p95 global translation: {key['maximum_window_global_translation_p95_px']:.6g} native pixels.
- Median registered/raw difference-MAD ratio on identical shift-valid pair interiors: {key['median_registered_over_raw_difference_mad_ratio']:.6g}.
- Median retained matched-support pixel fraction: {key['median_matched_support_pixel_fraction']:.6g}.
- Median matched/full-frame raw difference-MAD ratio: {key['median_matched_to_full_raw_difference_mad_ratio']:.6g}.
- Windows with forward/back review triggers: {key['forward_backward_review_window_count']} / {summary['scope']['window_count']}.
- Windows requiring local-translation reliability review: {key['local_translation_reliability_review_window_count']} / {summary['scope']['window_count']}.
- Median local-window/full-field p95 translation ratio on shared 060126 recordings: {key['median_shared_window_to_full_field_p95_ratio']:.6g}.
- Smallest grouped exact p / BH q: {key['minimum_grouped_exact_p']:.6g} / {key['minimum_grouped_bh_q']:.6g}; BH-passing associations: {key['association_count_passing_bh_0_05']}.
- Stored uint16 high-code occupancy: maximum {key['maximum_stored_high_code_fraction']:.6g}; analog rail status remains unresolved.

## Frozen endpoint linkage

Raw-HC and JEPA/random residual values are read from the hash-verified NREV-EXP-0029 Run-B tables without recomputation or tuning. Associations use twelve window-level observations, retain recording groups, enumerate all 6^4 = 1,296 within-recording permutations when complete, and report leave-one-recording-out ranges. They remain descriptive with only four recording groups.

Reciprocal forward/back phase estimates are conjugate by construction and therefore validate implementation symmetry rather than supply independent biological evidence. Independent two-step temporal cycle closure, search-boundary concentration, tile/global disagreement, and registered-difference reduction are used as the local reliability diagnostics. The local 64x64 estimates are contextualized against, but not equated with, the differently sampled prior full-field 11-recording screen.

For every valid pair, the raw and registered difference MADs entering the reduction ratio use the exact same conservative shift-valid interior. The separately reported full-frame raw difference MAD remains a standalone change diagnostic and is never used as that ratio's denominator.

Current validated carrier/coherence/recurrence traces were not joined: their finite Spon support is UI 1800-2359, while every frozen Spon background window ends earlier; 060126 has no matched exported feature traces. This prevents an estimand-changing or NaN-based comparison.

## Sensor and registration boundary

Stored integer dtype rails are exact digital-code diagnostics only. ADC bit depth, detector rail, black level, gain, and upstream clipping metadata are absent, so analog saturation cannot be ruled in or out. Phase correlation measures translation-like change and cannot distinguish specimen motion, deformation, scan effects, neural activity, gain drift, or structured noise. Registered residuals also include interpolation error and non-translational change.

## Scientific audit status

The small numeric package is validated, but the required full-field videos, candidate-surrogate close-ups/traces, matched figures, and media decoding checks were not produced. Therefore scientific audit completion and scientific promotion are false.
"""


def run_motion_registration_confound_audit(
    *,
    repository_root: Path,
    data_root: Path,
    parent_root: Path,
    output_root: Path,
    parent_run_record_path: Path,
    parent_provenance_index_path: Path,
    acquisition_coverage_root: Path,
    acquisition_run_record_path: Path,
    acquisition_provenance_index_path: Path,
    data_descriptor_path: Path,
    feature_panel_root: Path,
    trace_atlas_root: Path,
    run_id: str,
    config: MotionRegistrationConfoundConfig = MotionRegistrationConfoundConfig(),
) -> dict[str, Any]:
    planned_dt = _utc_now()
    repository_root = repository_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    parent_root = parent_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    parent_run_record_path = parent_run_record_path.expanduser().resolve()
    parent_provenance_index_path = parent_provenance_index_path.expanduser().resolve()
    acquisition_coverage_root = acquisition_coverage_root.expanduser().resolve()
    acquisition_run_record_path = acquisition_run_record_path.expanduser().resolve()
    acquisition_provenance_index_path = acquisition_provenance_index_path.expanduser().resolve()
    data_descriptor_path = data_descriptor_path.expanduser().resolve()
    feature_panel_root = feature_panel_root.expanduser().resolve()
    trace_atlas_root = trace_atlas_root.expanduser().resolve()
    if output_root.exists() or output_root.with_name(output_root.name + ".partial").exists():
        raise FileExistsError(f"refusing existing completed or partial output root: {output_root}")
    if output_root.name != run_id:
        raise ValueError("output-root basename must equal run_id")

    started_dt = _utc_now()
    planned_at = _format_utc(planned_dt)
    started_at = _format_utc(started_dt)
    runner_sha256 = sha256_file(Path(__file__))
    config_manifest = config.to_manifest()
    config_sha256 = stable_hash(config_manifest)
    git_provenance = _git_provenance(repository_root)
    registry_code_record = _registry_code_record(git_provenance)
    run_schema_path = repository_root / "research" / "schemas" / "run.schema.json"
    run_schema_payload = _read_json(run_schema_path)
    registry_schema_checks = validate_registry_code_against_run_schema(
        registry_code_record,
        run_schema_payload,
    )
    run_schema_sha256 = sha256_file(run_schema_path)
    runtime_provenance = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "tifffile_version": tifffile.__version__,
        "pyyaml_version": yaml.__version__,
        "jsonschema_version": package_version("jsonschema"),
        "process_id_persisted": False,
        "cpu_count": os.cpu_count(),
    }

    def replay_path(path: Path) -> str:
        return _portable_repository_path(path, repository_root).removeprefix("repo://")

    portable_command = [
        ".venv-neurobench/bin/python",
        "-m",
        RUNNER_MODULE,
        "--repository-root",
        ".",
        "--data-root",
        "$NEUROBENCH_DATA_ROOT",
        "--parent-root",
        replay_path(parent_root),
        "--output-root",
        replay_path(output_root),
        "--run-id",
        run_id,
        "--parent-run-record",
        replay_path(parent_run_record_path),
        "--parent-provenance-index",
        replay_path(parent_provenance_index_path),
        "--acquisition-coverage-root",
        replay_path(acquisition_coverage_root),
        "--acquisition-run-record",
        replay_path(acquisition_run_record_path),
        "--acquisition-provenance-index",
        replay_path(acquisition_provenance_index_path),
        "--data-descriptor",
        replay_path(data_descriptor_path),
        "--feature-panel-root",
        replay_path(feature_panel_root),
        "--trace-atlas-root",
        replay_path(trace_atlas_root),
    ]

    resources = _resource_preflight(output_root)
    if not all(
        resources[key]
        for key in ("output_root_absent", "partial_root_absent", "free_disk_passed", "available_memory_passed")
    ):
        raise RuntimeError("resource/output preflight failed")
    referenced = (
        "background_window_manifest.json",
        "paired_injection_results.tsv",
        "residual_diagnostics.tsv",
        "residual_diagnostics.json",
        "summary.json",
        "validation.json",
    )
    parent_integrity = _verify_parent_chain(
        repository_root=repository_root,
        parent_root=parent_root,
        parent_run_record_path=parent_run_record_path,
        parent_provenance_index_path=parent_provenance_index_path,
        referenced_parent_artifacts=referenced,
        expected_run_id="NREV-RUN-EXP-0029-SCREEN-20260830-B",
    )
    acquisition_integrity = _verify_parent_chain(
        repository_root=repository_root,
        parent_root=acquisition_coverage_root,
        parent_run_record_path=acquisition_run_record_path,
        parent_provenance_index_path=acquisition_provenance_index_path,
        referenced_parent_artifacts=("motion_audit.json",),
        expected_run_id="NREV-RUN-EXP-0028-SCREEN-20260829-B",
    )
    acquisition_coverage = _read_json(acquisition_coverage_root / "motion_audit.json")
    if (
        acquisition_coverage.get("schema_version") != "neurobench.jepa_motion_audit.v1"
        or acquisition_coverage.get("status") != "screen_complete"
        or len(acquisition_coverage.get("recordings", [])) != 11
        or acquisition_coverage.get("manifest", {}).get("source_policy") != "raw_060126_only_spon_excluded"
        or not acquisition_coverage.get("inventory_validation", {}).get("live_validation", {}).get("hashes_verified")
    ):
        raise ValueError("frozen 11-recording acquisition-coverage audit is incomplete or incompatible")
    descriptor_sha = sha256_file(data_descriptor_path)
    parent_run_record = yaml.safe_load(parent_run_record_path.read_text(encoding="utf-8"))
    expected_descriptor_hashes = {
        str(row["sha256"])
        for row in parent_run_record.get("inputs", [])
        if row.get("id") == "JEPA-RAW-VIDEO-SOURCES-V1"
    }
    if expected_descriptor_hashes != {descriptor_sha}:
        raise ValueError("data descriptor hash differs from frozen parent run record")
    descriptor = _read_json(data_descriptor_path)
    window_manifest_path = parent_root / "background_window_manifest.json"
    window_manifest = _read_json(window_manifest_path)
    windows = validate_frozen_window_manifest(window_manifest, config=config)
    movies, raw_integrity = _verify_raw_sources(
        windows,
        descriptor_payload=descriptor,
        data_root=data_root,
        verify_hashes=config.verify_raw_hashes,
    )
    feature_portability = _feature_portability(
        repository_root=repository_root,
        windows=windows,
        feature_panel_root=feature_panel_root,
        trace_atlas_root=trace_atlas_root,
    )

    def repository_input(input_id: str, role: str, path: Path) -> dict[str, Any]:
        return {
            "id": input_id,
            "role": role,
            "uri": _portable_repository_path(path, repository_root),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    input_records = [
        repository_input("acquisition_artifact_index_committed", "frozen_provenance_index", acquisition_provenance_index_path),
        repository_input(
            "acquisition_artifact_index_live",
            "frozen_live_artifact_index",
            acquisition_coverage_root / "artifact_index.json",
        ),
        repository_input(
            "acquisition_motion_audit",
            "full_11_recording_acquisition_coverage",
            acquisition_coverage_root / "motion_audit.json",
        ),
        repository_input("acquisition_run_record", "frozen_registry_run_record", acquisition_run_record_path),
        repository_input("data_descriptor", "raw_recording_descriptor", data_descriptor_path),
        repository_input("parent_artifact_index_committed", "frozen_provenance_index", parent_provenance_index_path),
        repository_input(
            "parent_artifact_index_live",
            "frozen_live_artifact_index",
            parent_root / "artifact_index.json",
        ),
        repository_input("parent_run_record", "frozen_registry_run_record", parent_run_record_path),
        repository_input("registry_run_schema", "registry_schema", run_schema_path),
    ]
    input_records.extend(
        repository_input(
            f"parent_artifact_{index:02d}",
            "frozen_exp_0029_endpoint_or_manifest",
            parent_root / str(row["path"]),
        )
        for index, row in enumerate(parent_integrity["referenced_artifacts"])
    )
    input_records.extend(
        {
            "id": f"feature_portability_artifact_{index:02d}",
            "role": "feature_support_compatibility_check",
            "uri": str(row["uri"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
        for index, row in enumerate(feature_portability["artifacts"])
    )
    input_records.extend(
        {
            "id": f"raw_recording_{row['recording_id']}",
            "role": "raw_frozen_window_source",
            "uri": str(row["source_uri"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
        for row in raw_integrity["recordings"]
    )
    input_records = sorted(input_records, key=lambda row: str(row["id"]))
    input_set_sha256 = stable_hash(input_records)
    parent_metrics = _aggregate_parent_endpoints(
        _read_tsv(parent_root / "paired_injection_results.tsv"),
        list(_read_json(parent_root / "residual_diagnostics.json")["rows"]),
        window_ids={str(row["background_window_id"]) for row in windows},
    )

    preflight_checks = {
        "new_noncolliding_output_root": resources["output_root_absent"] and resources["partial_root_absent"],
        "resource_headroom": resources["free_disk_passed"] and resources["available_memory_passed"],
        "frozen_parent_index_matches_committed_provenance": parent_integrity["status"] == "passed",
        "frozen_parent_referenced_artifacts_hash_verified": all(
            row["matches_committed_parent_index"] for row in parent_integrity["referenced_artifacts"]
        ),
        "frozen_full_11_recording_acquisition_coverage_verified": acquisition_integrity["status"] == "passed",
        "full_11_recording_raw_hashes_previously_verified": bool(
            acquisition_coverage["inventory_validation"]["live_validation"]["hashes_verified"]
        ),
        "data_descriptor_bound_to_parent": expected_descriptor_hashes == {descriptor_sha},
        "exact_twelve_window_group_contract": len(windows) == config.expected_window_count,
        "raw_sources_memmap_read_only_and_hash_verified": raw_integrity["status"] == "passed"
        and raw_integrity["hashes_verified"],
        "raw_video_not_copied": not raw_integrity["raw_video_copied"],
        "feature_join_compatibility_explicit": feature_portability["status"].startswith("not_joined_"),
        "exact_portable_input_inventory_hashed": bool(input_records)
        and input_set_sha256 == stable_hash(input_records),
        "runner_hash_matches_live_module": runner_sha256 == sha256_file(Path(__file__)),
        "content_aware_dirty_diff_captured_before_execution": not git_provenance["dirty"]
        or (
            isinstance(git_provenance.get("diff_sha256"), str)
            and len(git_provenance["diff_sha256"]) == 64
        ),
        "registry_run_schema_code_contract_passed": all(registry_schema_checks.values()),
        "scientific_completion_forbidden": not config.scientific_completion_allowed,
    }
    if not all(preflight_checks.values()):
        raise RuntimeError(f"fail-closed preflight checks failed: {[key for key, value in preflight_checks.items() if not value]}")
    partial = output_root.with_name(output_root.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_id,
        "planned_at": planned_at,
        "started_at": started_at,
        "execution_mode": RUN_MODE,
        "runner_sha256": runner_sha256,
        "config_sha256": config_sha256,
        "input_set_sha256": input_set_sha256,
        "git": git_provenance,
        "registry_code": registry_code_record,
        "registry_run_schema": {
            "uri": _portable_repository_path(run_schema_path, repository_root),
            "sha256": run_schema_sha256,
            "checks": registry_schema_checks,
        },
        "checks": preflight_checks,
        "resources": resources,
        "parent_integrity": parent_integrity,
        "acquisition_coverage_integrity": acquisition_integrity,
        "full_11_recording_acquisition_coverage": {
            "status": acquisition_coverage["status"],
            "recording_count": len(acquisition_coverage["recordings"]),
            "recording_ids": [row["recording"]["recording_id"] for row in acquisition_coverage["recordings"]],
            "sampled_adjacent_pair_count": sum(
                int(row["adjacent_frame_difference"]["pair_count"])
                for row in acquisition_coverage["recordings"]
            ),
            "aggregate_gate": acquisition_coverage["aggregate_gate"],
            "association_endpoint_rows_available": False,
            "reason": "Only the three held 060126 recordings plus Spon have frozen EXP-0029 window endpoints.",
        },
        "raw_input_integrity": raw_integrity,
        "feature_portability": feature_portability,
    }
    atomic_json(partial / "preflight.json", preflight)

    window_rows = []
    pair_rows = []
    for window in windows:
        summary, pairs = analyze_window(window, movies[str(window["recording_id"])], config=config)
        summary.update(parent_metrics[str(window["background_window_id"])])
        window_rows.append(summary)
        pair_rows.extend(pairs)

    full_field_p95 = {
        str(row["recording"]["recording_id"]): row["translation"]["global_magnitude_px"]["p95"]
        for row in acquisition_coverage["recordings"]
    }
    for row in window_rows:
        context = full_field_p95.get(str(row["recording_id"]))
        row["full_field_context_global_translation_p95_px"] = context
        row["local_window_to_full_field_p95_ratio"] = (
            None
            if context is None or float(context) <= config.minimum_robust_scale
            else float(row["global_translation_p95_px"]) / float(context)
        )

    associations = adjust_grouped_association_family(
        [
            grouped_exact_association(window_rows, predictor=predictor, outcome=outcome)
            for predictor, outcome in PRIMARY_ASSOCIATIONS
        ]
    )
    recording_rows = _recording_summaries(window_rows)
    _write_tsv(partial / "window_metrics.tsv", window_rows)
    _write_tsv(partial / "pair_metrics.tsv", pair_rows)
    _write_tsv(partial / "association_tests.tsv", associations)
    _write_tsv(partial / "recording_summary.tsv", recording_rows)

    mad_errors = [float(row["temporal_mad_absolute_error"]) for row in window_rows]
    key_results = {
        "median_window_global_translation_median_px": _median(
            [float(row["global_translation_median_px"]) for row in window_rows if row["global_translation_median_px"] is not None]
        ),
        "maximum_window_global_translation_p95_px": max(
            float(row["global_translation_p95_px"]) for row in window_rows if row["global_translation_p95_px"] is not None
        ),
        "median_registered_over_raw_difference_mad_ratio": _median(
            [
                float(row["registered_over_raw_difference_mad_ratio"])
                for row in window_rows
                if row["registered_over_raw_difference_mad_ratio"] is not None
            ]
        ),
        "median_matched_support_pixel_fraction": _median(
            [float(row["matched_support_pixel_fraction_median"]) for row in window_rows]
        ),
        "median_matched_to_full_raw_difference_mad_ratio": _median(
            [
                float(row["raw_difference_mad_matched_support_median"])
                / float(row["raw_difference_mad_median"])
                for row in window_rows
                if float(row["raw_difference_mad_median"]) > config.minimum_robust_scale
            ]
        ),
        "forward_backward_review_window_count": sum(
            bool(row["forward_backward_review_trigger"]) for row in window_rows
        ),
        "local_translation_reliability_review_window_count": sum(
            row["local_translation_reliability_status"] == "review_required" for row in window_rows
        ),
        "median_two_step_cycle_closure_p95_px": _median(
            [
                float(row["two_step_cycle_closure_p95_px"])
                for row in window_rows
                if row["two_step_cycle_closure_p95_px"] is not None
            ]
        ),
        "median_search_boundary_pair_fraction": _median(
            [float(row["search_boundary_pair_fraction"]) for row in window_rows]
        ),
        "median_shared_window_to_full_field_p95_ratio": _median(
            [
                float(row["local_window_to_full_field_p95_ratio"])
                for row in window_rows
                if row["local_window_to_full_field_p95_ratio"] is not None
            ]
        ),
        "minimum_grouped_exact_p": min(
            float(row["grouped_exact_two_sided_p"])
            for row in associations
            if row["grouped_exact_two_sided_p"] is not None
        ),
        "minimum_grouped_bh_q": min(
            float(row["grouped_exact_bh_q"])
            for row in associations
            if row["grouped_exact_bh_q"] is not None
        ),
        "association_count_passing_bh_0_05": sum(bool(row["passes_bh_0_05"]) for row in associations),
        "maximum_stored_high_code_fraction": max(float(row["stored_high_code_fraction"]) for row in window_rows),
        "analog_sensor_rail_status": "unresolved_for_all_windows",
    }
    summary = {
        "schema_version": 1,
        "experiment_id": "NREV-EXP-0025",
        "run_id": run_id,
        "status": "bounded_engineering_screen_complete_scientific_audit_incomplete",
        "question": "Are translation-like motion, registration residuals, or acquisition rails associated with temporal variability and frozen EXP-0029 endpoint stability?",
        "scope": {
            "window_count": len(window_rows),
            "recording_count": len({row["recording_id"] for row in window_rows}),
            "adjacent_pair_count": len(pair_rows),
            "window_shape_tyx": [config.expected_window_frames, *config.expected_window_shape_yx],
            "grouping_unit": "recording_with_three_temporal_MAD_strata_windows",
            "recording_groups_are_not_independent_animals": True,
            "full_060126_acquisition_coverage_recordings": len(acquisition_coverage["recordings"]),
            "full_060126_acquisition_coverage_sampled_adjacent_pairs": sum(
                int(row["adjacent_frame_difference"]["pair_count"])
                for row in acquisition_coverage["recordings"]
            ),
            "endpoint_association_coverage": "three held 060126 recordings plus Spon only",
        },
        "methods": {
            "translation": ALGORITHM,
            "translation_direction": "correction_applied_to_moving_frame_to_register_to_reference",
            "registration_residual": "normal_consistent_MAD_after_subpixel_translation_on_shift-valid_interior",
            "registered_raw_ratio_support": (
                "exact_same_shift_valid_interior_per_pair; full_frame_raw_MAD_is_standalone_only"
            ),
            "forward_backward_consistency": (
                "L2_norm_of_forward_plus_reverse_correction_vectors; structurally conjugate implementation check"
            ),
            "independent_temporal_consistency": (
                "two_step_direct_correction_minus_sum_of_two_adjacent_corrections"
            ),
            "association": "all-window_Spearman_plus_within-recording_rank_exact_permutation_and_leave-one-recording-out_range",
            "association_permutation_space_when_complete": 6**4,
        },
        "key_results": key_results,
        "full_11_recording_acquisition_coverage": {
            "source_run_id": "NREV-RUN-EXP-0028-SCREEN-20260829-B",
            "source_motion_audit_sha256": sha256_file(acquisition_coverage_root / "motion_audit.json"),
            "aggregate_gate": acquisition_coverage["aggregate_gate"],
            "raw_hashes_verified_in_source_audit": True,
            "endpoint_association_not_imputed_for_other_eight_recordings": True,
        },
        "association_rows": associations,
        "feature_portability": feature_portability,
        "sensor_boundary": {
            "stored_integer_rails": "resolved_exactly_for_stored_uint16_codes",
            "analog_sensor_rail": "unresolved_without_ADC_detector_gain_black_level_and_clipping_metadata",
        },
        "claim_boundary": {
            "label_free": True,
            "motion_correction_applied_to_scientific_endpoint": False,
            "causal_attribution": False,
            "biological_motion_identified": False,
            "neuron_identity_or_precision_identified": False,
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
        },
        "limitations": [
            "Only twelve selected 64x64 windows and four recording groups are analyzed.",
            "The recording groups may share animal, preparation, acquisition date, and hardware.",
            "Phase translation does not model rotation, deformation, raster/scan distortion, or depth motion.",
            "Reciprocal phase-correlation closure is structurally symmetric and not an independent reliability estimate.",
            "Small-window and full-field motion summaries use different frame samples and spatial support.",
            "Registered difference includes interpolation, fluorescence activity, gain drift, and noise.",
            "Matched pair support uses a conservative symmetric margin of ceil(max absolute shift) plus two pixels.",
            "Stored uint16 rails do not resolve the analog detector or ADC acquisition rails.",
            "Raw-HC and learned-residual endpoints are frozen injection-screen diagnostics, not biological precision.",
            "Current full-trace feature arrays have no compatible support on the twelve frozen windows and were not joined.",
            "Exact grouped permutation tests have only four recording groups and are descriptive.",
            "Required scientific-audit media were not generated.",
        ],
    }
    summary["summary_sha256"] = stable_hash(summary)
    scientific_audit = {
        "schema_version": 1,
        "enabled": config.scientific_audit_enabled,
        "numeric_window_and_association_tables_complete": True,
        "required_media_complete": False,
        "inventory_validation_passed": False,
        "scientific_audit_complete": False,
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
        "promotion_blocked": True,
        "expert_section": "not_applicable_label_free_background_window_screen",
        "missing_required_families": [
            "model_or_candidate_surrogate_full_field_videos",
            "candidate_surrogate_closeups_and_full_duration_traces",
            "figure_table_only_matched_comparison",
            "media_decode_and_annotation_pixel_validation",
        ],
    }
    validation_checks = {
        "preflight_passed": all(preflight_checks.values()),
        "full_11_recording_acquisition_coverage_preserved": len(acquisition_coverage["recordings"]) == 11,
        "exact_frozen_window_count": len(window_rows) == config.expected_window_count,
        "exact_recording_group_count": len(recording_rows) == config.expected_recording_count,
        "exact_adjacent_pair_count": len(pair_rows) == config.expected_window_count * (config.expected_window_frames - 1),
        "manifest_temporal_mad_reproduced": max(mad_errors) <= config.temporal_mad_absolute_tolerance,
        "all_windows_have_parent_raw_hc_grid": all(int(row["raw_hc_fixture_count"]) == 9 for row in window_rows),
        "parent_residual_methods_joined_without_refit": all(
            all(row.get(prefix + field) is not None for prefix in ("jepa_", "random_") for field in ("background_rms_ratio", "dynamic_mad_ratio", "seam_to_interior_jump_ratio"))
            for row in window_rows
        ),
        "grouped_association_rows_complete": len(associations) == len(PRIMARY_ASSOCIATIONS),
        "association_multiplicity_control_complete": all(
            row["grouped_exact_bh_q"] is not None and row["association_family_size"] == len(PRIMARY_ASSOCIATIONS)
            for row in associations
        ),
        "four_registered_ratio_associations_use_matched_support": sum(
            row["difference_mad_support_contract"]
            == "ratio_uses_exact_same_shift_valid_interior_per_pair"
            for row in associations
        )
        == 4,
        "grouped_permutation_space_exact_when_supported": all(
            row["grouped_permutation_count"] in {0, 6**4} for row in associations
        ),
        "stored_digital_rail_status_explicit": all(row["stored_dtype"] == "uint16" for row in window_rows),
        "forward_backward_symmetry_not_overinterpreted": all(
            row["forward_backward_consistency_status"] == "structurally_symmetric_not_independent"
            for row in window_rows
        ),
        "independent_two_step_cycle_reported": all(
            row["two_step_cycle_valid_fraction"] is not None for row in window_rows
        ),
        "local_translation_reliability_reported": all(
            row["local_translation_reliability_status"] in {"screen_clear", "review_required"}
            for row in window_rows
        ),
        "raw_registered_mad_exact_matched_support_per_pair": all(
            _pair_matched_support_contract_valid(row, config=config) for row in pair_rows
        ),
        "matched_support_window_semantics_persisted": all(
            row["registered_raw_mad_support_contract"] == "exact_same_shift_valid_interior_per_pair"
            and row["raw_difference_mad_support_contract"]
            == "full_frame_standalone_diagnostic_not_ratio_denominator"
            for row in window_rows
        ),
        "full_field_context_joined_only_where_recording_compatible": sum(
            row["full_field_context_global_translation_p95_px"] is not None for row in window_rows
        )
        == 9,
        "analog_sensor_rail_uncertainty_explicit": all(
            row["analog_sensor_rail_status"] == "unresolved" for row in window_rows
        ),
        "feature_nonjoin_is_explicit_and_estimand_preserving": feature_portability["status"].startswith("not_joined_"),
        "raw_video_not_copied": not raw_integrity["raw_video_copied"],
        "exact_input_inventory_hashed": input_set_sha256 == stable_hash(input_records),
        "runner_hash_matches_live_code": runner_sha256 == sha256_file(Path(__file__)),
        "config_hash_matches_exact_manifest": config_sha256 == stable_hash(config_manifest),
        "scientific_audit_complete": False,
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
    }
    engineering_keys = [
        key
        for key in validation_checks
        if key not in {"scientific_audit_complete", "scientific_completion", "scientific_promotion_allowed"}
    ]
    failed_engineering = [key for key in engineering_keys if not validation_checks[key]]
    if failed_engineering:
        atomic_json(
            partial / "validation.json",
            {
                "schema_version": 1,
                "status": "failed_engineering_validation",
                "checks": validation_checks,
                "failed_engineering_checks": failed_engineering,
                "scientific_completion": False,
                "scientific_promotion_allowed": False,
            },
        )
        raise RuntimeError(f"engineering validation failed: {failed_engineering}")

    atomic_json(partial / "summary.json", summary)
    atomic_json(partial / "scientific_audit_status.json", scientific_audit)
    atomic_json(
        partial / "llm_context.json",
        {
            "schema_version": 1,
            "experiment_id": "NREV-EXP-0025",
            "run_id": run_id,
            "entry_point": "summary.json",
            "run_provenance": "run_provenance.json",
            "status_record": "status.json",
            "primary_tables": ["window_metrics.tsv", "association_tests.tsv", "recording_summary.tsv"],
            "pair_diagnostics": "pair_metrics.tsv",
            "parent_run": "NREV-RUN-EXP-0029-SCREEN-20260830-B",
            "coordinate_contract": "x_is_column_y_is_row_zero_based_half_open",
            "temporal_contract": "frames_zero_based_half_open_native_cadence_unresolved",
            "sensor_contract": "stored_uint16_rails_exact_analog_sensor_rails_unresolved",
            "scientific_audit_complete": False,
            "scientific_promotion_allowed": False,
        },
    )
    atomic_text(partial / "REPORT.md", _report(summary))

    ended_dt = _utc_now()
    ended_at = _format_utc(ended_dt)
    duration_seconds = (ended_dt - started_dt).total_seconds()
    resolved_config = {
        "schema_version": 2,
        "experiment_id": "NREV-EXP-0025",
        "run_id": run_id,
        "runner_module": RUNNER_MODULE,
        "runner_schema_version": SCHEMA_VERSION,
        "runner_sha256": runner_sha256,
        "config": config_manifest,
        "config_sha256": config_sha256,
        "inputs": {
            "records": input_records,
            "input_set_sha256": input_set_sha256,
        },
        "execution": {
            "mode": RUN_MODE,
            "command": portable_command,
            "planned_at": planned_at,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "git": git_provenance,
            "runtime": runtime_provenance,
            "data_root_argument": "$NEUROBENCH_DATA_ROOT",
            "timestamp_semantics": "started_before_preflight_ended_after_numeric_and_report_assembly",
        },
        "scientific_audit": {
            "enabled": config.scientific_audit_enabled,
            "expected_status": "incomplete_numeric_screen_only",
            "completion_or_promotion_if_incomplete": False,
        },
    }
    atomic_json(partial / "resolved_config.json", resolved_config)
    resolved_config_sha256 = sha256_file(partial / "resolved_config.json")
    run_provenance = {
        "schema_version": 1,
        "record_type": "run_provenance",
        "experiment_id": "NREV-EXP-0025",
        "run_id": run_id,
        "lifecycle": "succeeded",
        "planned_at": planned_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "execution": {
            "mode": RUN_MODE,
            "command": portable_command,
            "working_directory": "repo://",
            "data_root_argument": "$NEUROBENCH_DATA_ROOT",
            "timestamp_semantics": "started_before_preflight_ended_after_numeric_and_report_assembly",
        },
        "code": {
            **registry_code_record,
            "runner_module": RUNNER_MODULE,
            "runner_sha256": runner_sha256,
            "git": git_provenance,
            "registry_schema": {
                "uri": _portable_repository_path(run_schema_path, repository_root),
                "sha256": run_schema_sha256,
                "checks": registry_schema_checks,
            },
        },
        "configuration": {
            "config": config_manifest,
            "config_sha256": config_sha256,
            "resolved_config_uri": "run://resolved_config.json",
            "resolved_config_sha256": resolved_config_sha256,
        },
        "inputs": {
            "records": input_records,
            "input_set_sha256": input_set_sha256,
        },
        "environment": {"runtime": runtime_provenance},
        "scientific_boundary": {
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
            "claim_bearing": False,
            "scientific_audit": "incomplete_required_default_media_not_produced",
        },
    }
    provenance_contract_checks = validate_run_provenance(run_provenance, runner_path=Path(__file__))
    atomic_json(partial / "run_provenance.json", run_provenance)
    run_provenance_sha256 = sha256_file(partial / "run_provenance.json")
    validation_status = "passed_engineering_screen_scientific_audit_incomplete"
    status = {
        "schema_version": 1,
        "experiment_id": "NREV-EXP-0025",
        "run_id": run_id,
        "lifecycle": "succeeded",
        "status": "succeeded_engineering_screen",
        "planned_at": planned_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "execution_mode": RUN_MODE,
        "command": portable_command,
        "hashes": {
            "runner_sha256": runner_sha256,
            "config_sha256": config_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "input_set_sha256": input_set_sha256,
            **(
                {"diff_sha256": git_provenance["diff_sha256"]}
                if git_provenance["dirty"]
                else {}
            ),
        },
        "git": git_provenance,
        "runtime": runtime_provenance,
        "provenance": {
            "path": "run_provenance.json",
            "sha256": run_provenance_sha256,
        },
        "validation": validation_status,
        "scientific_audit": "incomplete",
        "claim_bearing": False,
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
    }
    status_contract_checks = validate_status_against_provenance(
        status,
        run_provenance,
        provenance_sha256=run_provenance_sha256,
        runner_path=Path(__file__),
    )
    atomic_json(partial / "status.json", status)
    bundle_checks = validate_output_provenance_bundle(partial, runner_path=Path(__file__))
    validation_checks.update(
        {
            "planned_started_ended_utc_persisted": provenance_contract_checks[
                "timestamps_complete_and_ordered"
            ],
            "duration_matches_persisted_timestamps": provenance_contract_checks[
                "duration_matches_timestamps"
            ],
            "registry_dirty_diff_schema_contract_passed": provenance_contract_checks[
                "registry_dirty_diff_schema_contract"
            ],
            "run_provenance_contract_passed": all(provenance_contract_checks.values()),
            "status_provenance_mirror_passed": all(status_contract_checks.values()),
            "resolved_config_bundle_passed": all(bundle_checks.values()),
            "resolved_config_hash_matches_written_file": resolved_config_sha256
            == sha256_file(partial / "resolved_config.json"),
            "runner_hash_still_matches_live_code_at_completion": runner_sha256
            == sha256_file(Path(__file__)),
            "portable_command_has_no_workstation_path": all("/home/" not in token for token in portable_command),
        }
    )
    engineering_keys = [
        key
        for key in validation_checks
        if key not in {"scientific_audit_complete", "scientific_completion", "scientific_promotion_allowed"}
    ]
    failed_engineering = [key for key in engineering_keys if not validation_checks[key]]
    validation = {
        "schema_version": 1,
        "status": validation_status if not failed_engineering else "failed_engineering_validation",
        "checks": validation_checks,
        "failed_engineering_checks": failed_engineering,
        "maximum_temporal_mad_reproduction_absolute_error": max(mad_errors),
        "artifact_index_validation": "performed_fail_closed_after_artifact_index_write",
        "scientific_completion": False,
        "scientific_promotion_allowed": False,
    }
    atomic_json(partial / "validation.json", validation)
    if failed_engineering:
        raise RuntimeError(f"engineering validation failed: {failed_engineering}")

    index = _artifact_index(partial)
    atomic_json(partial / "artifact_index.json", index)
    validate_artifact_index(partial)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output_root)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--parent-run-record",
        type=Path,
        default=Path("research/registry/runs/NREV-RUN-EXP-0029-SCREEN-20260830-B.yaml"),
    )
    parser.add_argument(
        "--parent-provenance-index",
        type=Path,
        default=Path("research/run-provenance/NREV-RUN-EXP-0029-SCREEN-20260830-B/artifact_index.json"),
    )
    parser.add_argument(
        "--acquisition-coverage-root",
        type=Path,
        default=Path("Outputs/NeuronIdentifiability/NREV-EXP-0028/runs/NREV-RUN-EXP-0028-SCREEN-20260829-B"),
    )
    parser.add_argument(
        "--acquisition-run-record",
        type=Path,
        default=Path("research/registry/runs/NREV-RUN-EXP-0028-SCREEN-20260829-B.yaml"),
    )
    parser.add_argument(
        "--acquisition-provenance-index",
        type=Path,
        default=Path("research/run-provenance/NREV-RUN-EXP-0028-SCREEN-20260829-B/artifact_index.json"),
    )
    parser.add_argument(
        "--data-descriptor",
        type=Path,
        default=Path("research/data-registry/jepa_raw_video_sources_v1.json"),
    )
    parser.add_argument(
        "--feature-panel-root",
        type=Path,
        default=Path("Outputs/NeuronIdentifiability/spon_ca_burst_full_trace_feature_panel_v2"),
    )
    parser.add_argument(
        "--trace-atlas-root",
        type=Path,
        default=Path("Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v6/03_trace_atlas"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.expanduser().resolve()

    def resolve_repository_path(path: Path) -> Path:
        return path.expanduser().resolve() if path.is_absolute() else (repository_root / path).resolve()

    summary = run_motion_registration_confound_audit(
        repository_root=repository_root,
        data_root=args.data_root,
        parent_root=resolve_repository_path(args.parent_root),
        output_root=resolve_repository_path(args.output_root),
        parent_run_record_path=resolve_repository_path(args.parent_run_record),
        parent_provenance_index_path=resolve_repository_path(args.parent_provenance_index),
        acquisition_coverage_root=resolve_repository_path(args.acquisition_coverage_root),
        acquisition_run_record_path=resolve_repository_path(args.acquisition_run_record),
        acquisition_provenance_index_path=resolve_repository_path(args.acquisition_provenance_index),
        data_descriptor_path=resolve_repository_path(args.data_descriptor),
        feature_panel_root=resolve_repository_path(args.feature_panel_root),
        trace_atlas_root=resolve_repository_path(args.trace_atlas_root),
        run_id=args.run_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGORITHM",
    "MotionRegistrationConfoundConfig",
    "SCHEMA_VERSION",
    "adjust_grouped_association_family",
    "analyze_window",
    "compute_pair_diagnostics",
    "grouped_exact_association",
    "run_motion_registration_confound_audit",
    "validate_artifact_index",
    "validate_frozen_window_manifest",
    "validate_output_provenance_bundle",
    "validate_registry_code_against_run_schema",
    "validate_run_provenance",
    "validate_status_against_provenance",
]
