"""Protected continuation of the Gamma-LS conditioning sensitivity screen.

Only the fold/representation-specific historical anchor and Pareto-selected
setting frozen by :mod:`.conditioning_sensitivity` enter this executor.  All
candidate and calibration rows are written and sealed before either sparse-
positive table is hashed or parsed.  The v1 comparison uses a paired bootstrap
over 26 canonical identities; v7 remains descriptive sensitivity only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from neurobench.portable_paths import data_root

from . import conditioning_sensitivity as upstream
from .conditioning_sensitivity import (
    ANCHOR,
    ConditioningSensitivityConfig,
    ConditioningSetting,
    EMA_ALPHAS,
    GAUSSIAN_SIGMAS,
    REPRESENTATIONS,
    SCALE_FLOOR_PERCENTILES,
)
from .evaluation import (
    CANDIDATE_BUDGETS_PER_BURST,
    MATCH_RADIUS_PX,
    NMS_DISTANCE_PX,
    QUIET_NMS_PEAK_BURDENS,
)
from .protected import (
    ContextLane,
    _read_sparse_positives,
    _score_candidates_for_arm,
    _summary_by_arm,
    aggregate_match_rows,
    observation_match_rows,
)
from .screen import build_fold_contracts


EXPERIMENT_ID = "spon_ca_burst_gamma_ls_conditioning_protected_v1"
CONDITIONING_ROLES = ("historical_anchor", "training_pareto_selected")
NMS_DISTANCES_PX = (4, 6, 8)
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CLUSTER_COUNT = 26
MATERIAL_AUC_DELTA = 0.02
EXPECTED_OPERATING_ROWS = 720
EXPECTED_ROLE_CELLS = 24
EXPECTED_BOOTSTRAP_ROWS = 135

_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "sources",
    "design",
    "label_join",
    "bootstrap",
    "materiality",
    "resources",
    "scientific_audit",
}
_SOURCE_KEYS = {
    "conditioning_config",
    "conditioning_preflight_artifact_index_sha256",
    "conditioning_preflight_sha256",
    "conditioning_screen_artifact_index_sha256",
    "conditioning_selection_sha256",
}
_DESIGN_KEYS = {
    "conditioning_roles",
    "representations",
    "outer_folds",
    "quiet_crossfit",
    "quiet_nms_peaks_per_pseudo_burst",
    "nms_distance_px",
    "primary_nms_distance_px",
    "candidate_budgets_per_burst",
    "primary_candidate_budget_per_burst",
    "match_radius_px",
    "burst_aggregation",
    "unmatched_candidates",
}
_LABEL_KEYS = {
    "protected_v1_selector",
    "protected_v1_expected_occurrences",
    "protected_v1_expected_canonical_identities",
    "latest_v7_selector",
    "latest_v7_expected_occurrences",
    "latest_v7_inferential_claim",
    "candidate_seal_required_before_join",
}
_BOOTSTRAP_KEYS = {
    "cluster_field",
    "cluster_count",
    "seed",
    "replicates",
    "confidence_interval_percent",
}
_MATERIALITY_KEYS = {
    "primary_scope",
    "budget_curve_auc_absolute_delta",
    "requires_ci_excluding_zero",
    "representation_level_rule",
}
_RESOURCE_KEYS = {
    "device",
    "source_chunk_frames",
    "gamma_chunk_frames",
    "max_peak_vram_gib",
    "minimum_free_disk_gib",
}


class ConditioningProtectedUnavailable(RuntimeError):
    """Raised before final artifact commit when a frozen gate fails."""


class ConditioningProtectedConfigError(ValueError):
    """Raised when the strict protected manifest is not the v1 design."""


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], scope: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ConditioningProtectedConfigError(
            f"invalid {scope} fields: missing={sorted(expected - actual)}; "
            f"unknown={sorted(actual - expected)}"
        )


def _resolve_uri(value: str, *, repository: Path, authority: Path) -> Path:
    if value.startswith("repo://"):
        return (repository / value.removeprefix("repo://")).resolve()
    if value.startswith("data://"):
        return (authority / value.removeprefix("data://")).resolve()
    raise ConditioningProtectedConfigError(
        f"paths must use repo:// or data:// identifiers, got {value!r}"
    )


@dataclass(frozen=True)
class ConditioningProtectedConfig:
    """Validated protected-comparison manifest."""

    manifest_path: Path
    repository: Path
    authority: Path
    payload: dict[str, Any]
    conditioning_config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "ConditioningProtectedConfig":
        manifest = Path(path).expanduser().resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConditioningProtectedConfigError("manifest must be an object")
        _require_exact_keys(raw, _TOP_LEVEL_KEYS, "top-level")
        repository = Path(__file__).resolve().parents[3]
        authority = data_root(repository)
        sources = raw["sources"]
        if not isinstance(sources, dict):
            raise ConditioningProtectedConfigError("sources must be an object")
        _require_exact_keys(sources, _SOURCE_KEYS, "sources")
        for key in _SOURCE_KEYS - {"conditioning_config"}:
            if re.fullmatch(r"[0-9a-f]{64}", str(sources[key])) is None:
                raise ConditioningProtectedConfigError(
                    f"sources.{key} must be a lowercase SHA-256 digest"
                )
        config = cls(
            manifest_path=manifest,
            repository=repository,
            authority=authority,
            payload=deepcopy(raw),
            conditioning_config_path=_resolve_uri(
                str(sources["conditioning_config"]),
                repository=repository,
                authority=authority,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        raw = self.payload
        if raw["schema_version"] != 1 or raw["experiment_id"] != EXPERIMENT_ID:
            raise ConditioningProtectedConfigError("unexpected schema or experiment id")
        design = raw["design"]
        if not isinstance(design, dict):
            raise ConditioningProtectedConfigError("design must be an object")
        _require_exact_keys(design, _DESIGN_KEYS, "design")
        if tuple(design["conditioning_roles"]) != CONDITIONING_ROLES:
            raise ConditioningProtectedConfigError("conditioning roles changed")
        if tuple(design["representations"]) != REPRESENTATIONS:
            raise ConditioningProtectedConfigError("representations changed")
        if tuple(map(float, design["quiet_nms_peaks_per_pseudo_burst"])) != (
            QUIET_NMS_PEAK_BURDENS
        ):
            raise ConditioningProtectedConfigError("quiet burden grid changed")
        if tuple(map(int, design["nms_distance_px"])) != NMS_DISTANCES_PX:
            raise ConditioningProtectedConfigError("NMS sensitivity grid changed")
        if tuple(map(int, design["candidate_budgets_per_burst"])) != (
            CANDIDATE_BUDGETS_PER_BURST
        ):
            raise ConditioningProtectedConfigError("candidate budgets changed")
        exact_design = {
            "outer_folds": "leave_one_burst_out",
            "quiet_crossfit": "a_train_b_test_and_b_train_a_test",
            "primary_nms_distance_px": NMS_DISTANCE_PX,
            "primary_candidate_budget_per_burst": 58,
            "match_radius_px": MATCH_RADIUS_PX,
            "burst_aggregation": "temporal_threshold_occupancy",
            "unmatched_candidates": "unknown_not_negative",
        }
        if any(design.get(key) != value for key, value in exact_design.items()):
            raise ConditioningProtectedConfigError("protected metric contract changed")
        labels = raw["label_join"]
        if not isinstance(labels, dict):
            raise ConditioningProtectedConfigError("label_join must be an object")
        _require_exact_keys(labels, _LABEL_KEYS, "label_join")
        if labels != {
            "protected_v1_selector": "include_inclusive",
            "protected_v1_expected_occurrences": 79,
            "protected_v1_expected_canonical_identities": 26,
            "latest_v7_selector": "include_confirmed",
            "latest_v7_expected_occurrences": 106,
            "latest_v7_inferential_claim": False,
            "candidate_seal_required_before_join": True,
        }:
            raise ConditioningProtectedConfigError("label-join contract changed")
        bootstrap = raw["bootstrap"]
        if not isinstance(bootstrap, dict):
            raise ConditioningProtectedConfigError("bootstrap must be an object")
        _require_exact_keys(bootstrap, _BOOTSTRAP_KEYS, "bootstrap")
        if bootstrap != {
            "cluster_field": "canonical_roi_id",
            "cluster_count": BOOTSTRAP_CLUSTER_COUNT,
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_interval_percent": 95.0,
        }:
            raise ConditioningProtectedConfigError("paired bootstrap changed")
        materiality = raw["materiality"]
        if not isinstance(materiality, dict):
            raise ConditioningProtectedConfigError("materiality must be an object")
        _require_exact_keys(materiality, _MATERIALITY_KEYS, "materiality")
        if materiality != {
            "primary_scope": "crossfit_average_nms6",
            "budget_curve_auc_absolute_delta": MATERIAL_AUC_DELTA,
            "requires_ci_excluding_zero": True,
            "representation_level_rule": (
                "all_five_quiet_burdens_same_material_direction"
            ),
        }:
            raise ConditioningProtectedConfigError("materiality rule changed")
        resources = raw["resources"]
        if not isinstance(resources, dict):
            raise ConditioningProtectedConfigError("resources must be an object")
        _require_exact_keys(resources, _RESOURCE_KEYS, "resources")
        if not isinstance(resources["device"], str) or re.fullmatch(
            r"cuda(?::(?:0|[1-9][0-9]*))?", resources["device"]
        ) is None:
            raise ConditioningProtectedConfigError("protected stage is CUDA-only")
        if int(resources["source_chunk_frames"]) != 64 or int(
            resources["gamma_chunk_frames"]
        ) != 64:
            raise ConditioningProtectedConfigError("chunk contract changed")
        if float(resources["max_peak_vram_gib"]) != 8.0 or float(
            resources["minimum_free_disk_gib"]
        ) != 20.0:
            raise ConditioningProtectedConfigError("resource caps changed")
        if raw["scientific_audit"] != {"enabled": True}:
            raise ConditioningProtectedConfigError("scientific audit is mandatory")

    @property
    def conditioning_config(self) -> ConditioningSensitivityConfig:
        return ConditioningSensitivityConfig.load(self.conditioning_config_path)

    def portable_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


@dataclass(frozen=True)
class FrozenRole:
    training_fold: int
    heldout_burst: str
    representation: str
    support_context_id: str
    role: str
    setting: ConditioningSetting

    def as_dict(self) -> dict[str, Any]:
        return {
            "training_fold": self.training_fold,
            "heldout_burst": self.heldout_burst,
            "representation": self.representation,
            "support_context_id": self.support_context_id,
            "conditioning_role": self.role,
            **self.setting.as_dict(),
            "raw_lane_semantics": upstream.raw_lane_semantics(
                self.representation,
                self.setting.gaussian_sigma_px,
                self.setting.causal_ema_alpha,
            ),
        }


@dataclass(frozen=True)
class CandidateExecution:
    candidate_rows: tuple[Mapping[str, Any], ...]
    calibration_rows: tuple[Mapping[str, Any], ...]
    timing_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def _implementation_hashes(
    config: ConditioningProtectedConfig,
) -> dict[str, Mapping[str, Any]]:
    paths = (
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/metrics/sparse_detection.py",
        "neurobench/experiments/gamma_ls_difference/evaluation.py",
        "neurobench/experiments/gamma_ls_difference/protected.py",
        "neurobench/experiments/gamma_ls_difference/conditioning_sensitivity.py",
        "neurobench/experiments/gamma_ls_difference/conditioning_protected.py",
    )
    rows: dict[str, Mapping[str, Any]] = {}
    for relative in paths:
        path = config.repository / relative
        if not path.is_file():
            raise ConditioningProtectedUnavailable(
                f"protected implementation file is missing: {relative}"
            )
        rows[f"repo://{relative}"] = {
            "sha256": upstream._sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return rows


def _setting_from_frozen_role(
    payload: Mapping[str, Any], *, expected_role: str
) -> ConditioningSetting:
    if str(payload.get("role")) != expected_role:
        raise ConditioningProtectedUnavailable("frozen conditioning role changed")
    setting = ConditioningSetting(
        float(payload.get("gaussian_sigma_px")),
        float(payload.get("causal_ema_alpha")),
        float(payload.get("training_quiet_scale_floor_percentile")),
    )
    if (
        setting.gaussian_sigma_px not in GAUSSIAN_SIGMAS
        or setting.causal_ema_alpha not in EMA_ALPHAS
        or setting.scale_floor_percentile not in SCALE_FLOOR_PERCENTILES
        or setting.setting_id != str(payload.get("setting_id"))
        or setting.conditioning_id != str(payload.get("conditioning_id"))
    ):
        raise ConditioningProtectedUnavailable(
            "frozen role is outside the upstream conditioning grid"
        )
    if expected_role == "historical_anchor" and (
        setting.gaussian_sigma_px,
        setting.causal_ema_alpha,
        setting.scale_floor_percentile,
    ) != ANCHOR:
        raise ConditioningProtectedUnavailable("historical anchor changed")
    if expected_role == "training_pareto_selected" and int(
        payload.get("pareto_layer", -1)
    ) != 0:
        raise ConditioningProtectedUnavailable(
            "selected setting is not on the frozen Pareto front"
        )
    return setting


_STABLE_CUDA_RUNTIME_FIELDS = (
    "requested_device",
    "resolved_device",
    "torch_version",
    "torch_cuda_build",
    "device_name",
    "compute_capability",
    "visible_device_count",
    "total_vram_bytes",
    "cuda_available",
)


def _verify_completed_conditioning_preflight_snapshot(
    config: ConditioningProtectedConfig,
    conditioning_preflight_dir: str | Path,
) -> dict[str, Any]:
    """Verify an immutable completed preflight without rerunning its code gate.

    The sensitivity screen is already a completed, indexed artifact.  Its
    historical implementation fingerprints remain provenance; comparing them
    with the current sensitivity source would incorrectly make a completed
    screen stale after maintenance.  This verifier instead revalidates every
    indexed byte, the exact portable manifests, the movie, and the frozen
    support artifact.  It deliberately does not open or hash either sparse-
    positive source and it does not probe CUDA.
    """

    conditioning_config = config.conditioning_config
    root = Path(conditioning_preflight_dir).expanduser().resolve()
    indexed = upstream._verify_indexed_artifact(
        root, role="completed conditioning preflight"
    )
    sources = config.payload["sources"]
    if indexed["artifact_index_sha256"] != str(
        sources["conditioning_preflight_artifact_index_sha256"]
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight artifact index is not the manifest-frozen one"
        )
    preflight_path = root / "preflight.json"
    portable_path = root / "config.portable.json"
    preflight_sha256 = upstream._sha256(preflight_path)
    if preflight_sha256 != str(sources["conditioning_preflight_sha256"]):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight payload is not the manifest-frozen one"
        )
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    if portable != conditioning_config.portable_dict():
        raise ConditioningProtectedUnavailable(
            "conditioning preflight portable manifest is stale"
        )
    if payload.get("portable_config_sha256") != upstream._canonical_sha256(
        conditioning_config.portable_dict()
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight portable-manifest seal changed"
        )
    if (
        payload.get("status") != "ready_conditioning_sensitivity"
        or payload.get("data_ready") is not True
        or payload.get("gpu_run_ready") is not True
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight was not frozen as GPU-run ready"
        )
    for key in (
        "annotation_sources_reopened",
        "positive_coordinates_used",
        "positive_identities_used",
    ):
        if payload.get(key) is not False:
            raise ConditioningProtectedUnavailable(
                f"conditioning preflight leakage boundary changed: {key}"
            )

    frozen_base = payload.get("base_preflight")
    if not isinstance(frozen_base, Mapping) or not isinstance(
        frozen_base.get("root"), str
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight lacks its frozen base preflight"
        )
    base_root = Path(str(frozen_base["root"])).expanduser().resolve()
    base_indexed = upstream._verify_indexed_artifact(
        base_root, role="frozen base preflight"
    )
    if base_indexed["artifact_index_sha256"] != str(
        frozen_base.get("artifact_index_sha256", "")
    ):
        raise ConditioningProtectedUnavailable(
            "base preflight changed after conditioning preflight"
        )
    base_preflight_path = base_root / "preflight.json"
    base_portable_path = base_root / "config.portable.json"
    if (
        upstream._sha256(base_preflight_path)
        != str(frozen_base.get("preflight_sha256", ""))
        or upstream._sha256(base_portable_path)
        != str(frozen_base.get("portable_config_sha256", ""))
    ):
        raise ConditioningProtectedUnavailable(
            "base preflight payload or portable manifest changed"
        )
    base_payload = json.loads(base_preflight_path.read_text(encoding="utf-8"))
    base_portable = json.loads(base_portable_path.read_text(encoding="utf-8"))
    base_config = conditioning_config.base_config
    if base_portable != base_config.portable_dict():
        raise ConditioningProtectedUnavailable("base preflight manifest is stale")
    if (
        base_payload.get("data_ready") is not True
        or base_payload.get("gpu_run_ready") is not True
    ):
        raise ConditioningProtectedUnavailable("base preflight is not GPU-ready")
    frozen_movie = base_payload.get("source", {}).get("movie", {})
    movie_path = base_config.source_paths["movie"]
    movie_sha256 = str(frozen_movie.get("sha256", ""))
    if (
        not movie_sha256
        or movie_sha256 != str(frozen_base.get("movie_sha256", ""))
        or str(frozen_movie.get("path", ""))
        != str(base_config.payload["sources"]["movie"])
        or upstream._sha256(movie_path) != movie_sha256
    ):
        raise ConditioningProtectedUnavailable(
            "movie source changed after the frozen base preflight"
        )
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(movie, np.memmap)
        or tuple(map(int, movie.shape)) != upstream.EXPECTED_MOVIE_SHAPE
        or str(movie.dtype) != upstream.EXPECTED_MOVIE_DTYPE
    ):
        raise ConditioningProtectedUnavailable(
            "movie shape or dtype changed after conditioning preflight"
        )
    frozen_source = payload.get("source", {})
    if (
        frozen_source.get("source_id")
        != base_config.payload["sources"]["movie"]
        or tuple(map(int, frozen_source.get("movie_shape_tyx", ())))
        != upstream.EXPECTED_MOVIE_SHAPE
        or frozen_source.get("movie_dtype") != upstream.EXPECTED_MOVIE_DTYPE
        or frozen_source.get("labels_opened") is not False
        or frozen_source.get("positive_coordinates_used") is not False
        or frozen_source.get("positive_identities_used") is not False
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight source contract changed"
        )

    fold_contexts, support = upstream._verify_support_screen(conditioning_config)
    frozen_support = payload.get("support_screen")
    if not isinstance(frozen_support, Mapping) or (
        support["artifact_index_sha256"]
        != frozen_support.get("artifact_index_sha256")
        or support["fold_contexts_sha256"]
        != frozen_support.get("fold_contexts_sha256")
    ):
        raise ConditioningProtectedUnavailable(
            "support screen changed after conditioning preflight"
        )
    frozen_runtime = payload.get("runtime")
    if not isinstance(frozen_runtime, Mapping) or any(
        key not in frozen_runtime for key in _STABLE_CUDA_RUNTIME_FIELDS
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight lacks a complete frozen CUDA identity"
        )
    if (
        frozen_runtime.get("cuda_available") is not True
        or not str(frozen_runtime.get("resolved_device", "")).startswith("cuda:")
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning preflight CUDA identity is invalid"
        )
    return {
        **indexed,
        "preflight_sha256": preflight_sha256,
        "portable_config_sha256": upstream._sha256(portable_path),
        "base_preflight_artifact_index_sha256": base_indexed[
            "artifact_index_sha256"
        ],
        "support_artifact_index_sha256": support["artifact_index_sha256"],
        "support_fold_contexts_sha256": support["fold_contexts_sha256"],
        "runtime": dict(frozen_runtime),
        "fold_contexts": fold_contexts,
        "historical_conditioning_implementation": payload.get("implementation"),
        "annotation_sources_reopened": False,
    }


def _verify_current_cuda_runtime(
    config: ConditioningProtectedConfig,
    frozen_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Probe the execution runtime separately from immutable screen provenance."""

    requested = str(config.payload["resources"]["device"])
    if requested != str(frozen_runtime.get("requested_device", "")):
        raise ConditioningProtectedUnavailable(
            "protected device request differs from conditioning preflight"
        )
    current = upstream._require_cuda(requested)
    if any(
        current.get(key) != frozen_runtime.get(key)
        for key in _STABLE_CUDA_RUNTIME_FIELDS
    ):
        raise ConditioningProtectedUnavailable(
            "CUDA runtime identity changed after conditioning preflight"
        )
    required_vram = int(
        float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30
    )
    if int(current.get("free_vram_bytes_before", -1)) < required_vram:
        raise ConditioningProtectedUnavailable(
            "free CUDA memory is below the frozen 8-GiB execution cap"
        )
    return dict(current)


def _verify_conditioning_screen(
    config: ConditioningProtectedConfig,
    *,
    conditioning_preflight_dir: str | Path,
    conditioning_screen_dir: str | Path,
) -> tuple[tuple[FrozenRole, ...], dict[int, ContextLane], dict[str, Any]]:
    """Bind the exact upstream preflight, screen, contexts, and selection."""

    preflight = _verify_completed_conditioning_preflight_snapshot(
        config, conditioning_preflight_dir
    )
    screen_root = Path(conditioning_screen_dir).expanduser().resolve()
    indexed = upstream._verify_indexed_artifact(
        screen_root, role="conditioning screen"
    )
    if indexed["artifact_index_sha256"] != str(
        config.payload["sources"]["conditioning_screen_artifact_index_sha256"]
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning screen is not the manifest-frozen artifact"
        )
    summary_path = screen_root / "summary.json"
    validation_path = screen_root / "validation.json"
    contract_path = screen_root / "run_contract.json"
    selection_path = screen_root / "fold_selected_settings.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_conditioning_sensitivity_screen_only":
        raise ConditioningProtectedUnavailable("conditioning screen is not complete")
    if validation.get("status") != (
        "passed_conditioning_sensitivity_screen_artifact_contract_"
        "scientific_audit_pending"
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning screen artifact contract did not pass"
        )
    upstream._reject_coordinate_or_identity_fields(selection)
    selection_sha256 = upstream.verify_selection_seal(selection)
    if (
        selection_sha256
        != str(config.payload["sources"]["conditioning_selection_sha256"])
        or contract.get("schema_version") != 1
        or contract.get("experiment_id") != upstream.EXPERIMENT_ID
        or summary.get("selection_sha256") != selection_sha256
        or contract.get("selection_sha256") != selection_sha256
        or contract.get("selection_sealed_before_any_label_join") is not True
        or contract.get("label_join_api_present") is not False
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning selection seal or label boundary changed"
        )
    preflight_root = Path(conditioning_preflight_dir).expanduser().resolve()
    preflight_sha256 = upstream._sha256(preflight_root / "preflight.json")
    if contract.get("sensitivity_preflight_sha256") != preflight_sha256:
        raise ConditioningProtectedUnavailable(
            "conditioning screen was not produced from the supplied preflight"
        )
    required_contract = {
        "annotation_sources_reopened": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
        "protected_evaluation_in_this_run": False,
        "dense_map_device": "cuda",
        "outer_folds": 4,
        "fold_context_role": "support_candidate_context",
    }
    if any(contract.get(key) != value for key, value in required_contract.items()):
        raise ConditioningProtectedUnavailable(
            "conditioning screen execution or leakage contract changed"
        )
    if (
        contract.get("base_preflight_artifact_index_sha256")
        != preflight["base_preflight_artifact_index_sha256"]
        or contract.get("support_artifact_index_sha256")
        != preflight["support_artifact_index_sha256"]
        or contract.get("support_fold_contexts_sha256")
        != preflight["support_fold_contexts_sha256"]
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning screen support binding changed"
        )
    if (
        selection.get("selection_scope")
        != "independent_per_representation_within_outer_training_fold"
        or selection.get("downstream_roles") != list(CONDITIONING_ROLES)
        or selection.get("downstream_stage")
        != "protected_B_and_NMS_evaluation"
        or selection.get("positive_coordinates_used") is not False
        or selection.get("positive_identities_used") is not False
        or selection.get("burst_windows_used") is not True
    ):
        raise ConditioningProtectedUnavailable(
            "conditioning selection scope or leakage boundary changed"
        )
    rows = selection.get("fold_representation_selections")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ConditioningProtectedUnavailable(
            "conditioning selection must contain 12 fold/representation rows"
        )
    roles: list[FrozenRole] = []
    contexts: dict[int, ContextLane] = {}
    seen: set[tuple[int, str]] = set()
    fold_contexts = preflight["fold_contexts"]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ConditioningProtectedUnavailable("conditioning selection row is invalid")
        fold = int(row.get("training_fold", 0))
        representation = str(row.get("representation", ""))
        key = (fold, representation)
        if fold not in upstream.OUTER_FOLDS or representation not in REPRESENTATIONS:
            raise ConditioningProtectedUnavailable(
                "conditioning selection references invalid fold/representation"
            )
        if key in seen or str(row.get("heldout_burst")) != str(fold):
            raise ConditioningProtectedUnavailable(
                "conditioning selection duplicates or misaligns a fold"
            )
        seen.add(key)
        if (
            row.get("selection_scope") != "outer_training_fold_only"
            or row.get("selector_uses_burst_windows") is not True
            or row.get("selector_uses_positive_coordinates") is not False
            or row.get("selector_uses_positive_identities") is not False
        ):
            raise ConditioningProtectedUnavailable(
                "fold conditioning selection scope or leakage boundary changed"
            )
        reference, candidate = fold_contexts[fold]
        context_id = str(row.get("support_context_id", ""))
        if context_id != reference.context_id or context_id != str(
            candidate.get("context_id")
        ):
            raise ConditioningProtectedUnavailable(
                "conditioning selection changed its fold support context"
            )
        if fold not in contexts:
            contexts[fold] = ContextLane(
                role="frozen_support_candidate",
                context_id=context_id,
                half_width_px=int(candidate["half_width_px"]),
                guard_radius_px=int(candidate["guard_radius_px"]),
                shape=float(candidate["shape"]),
                mode_fraction_of_half_width=float(
                    candidate["mode_fraction_of_half_width"]
                ),
                selection_source="conditioning_screen_fold_selected_settings",
            )
            contexts[fold].spec()
        for role in CONDITIONING_ROLES:
            role_payload = row.get(role)
            if not isinstance(role_payload, Mapping):
                raise ConditioningProtectedUnavailable(
                    f"conditioning selection lacks role {role}"
                )
            setting = _setting_from_frozen_role(role_payload, expected_role=role)
            if role == "training_pareto_selected" and setting.setting_id not in {
                str(value) for value in row.get("pareto_front_setting_ids", [])
            }:
                raise ConditioningProtectedUnavailable(
                    "selected setting is absent from its frozen Pareto front"
                )
            roles.append(
                FrozenRole(
                    training_fold=fold,
                    heldout_burst=str(fold),
                    representation=representation,
                    support_context_id=context_id,
                    role=role,
                    setting=setting,
                )
            )
    expected = {
        (fold, representation)
        for fold in upstream.OUTER_FOLDS
        for representation in REPRESENTATIONS
    }
    if seen != expected or len(roles) != EXPECTED_ROLE_CELLS:
        raise ConditioningProtectedUnavailable(
            "conditioning selection does not cover the frozen comparison"
        )
    roles.sort(
        key=lambda item: (
            item.training_fold,
            item.representation,
            CONDITIONING_ROLES.index(item.role),
        )
    )
    return tuple(roles), contexts, {
        **indexed,
        "conditioning_screen_summary_sha256": upstream._sha256(summary_path),
        "conditioning_screen_validation_sha256": upstream._sha256(validation_path),
        "conditioning_screen_run_contract_sha256": upstream._sha256(contract_path),
        "conditioning_selection_file_sha256": upstream._sha256(selection_path),
        "conditioning_selection_sha256": selection_sha256,
        "conditioning_preflight_sha256": preflight_sha256,
        "conditioning_preflight_artifact_index_sha256": preflight[
            "artifact_index_sha256"
        ],
        "base_preflight_artifact_index_sha256": preflight[
            "base_preflight_artifact_index_sha256"
        ],
        "support_artifact_index_sha256": preflight[
            "support_artifact_index_sha256"
        ],
        "conditioning_preflight_runtime": preflight["runtime"],
        "historical_conditioning_implementation": preflight[
            "historical_conditioning_implementation"
        ],
        "candidate_selection_uses_positive_coordinates": False,
        "candidate_selection_uses_positive_identities": False,
    }


def _lane_for_role(context: ContextLane, role: str) -> ContextLane:
    return ContextLane(
        role=role,
        context_id=context.context_id,
        half_width_px=context.half_width_px,
        guard_radius_px=context.guard_radius_px,
        shape=context.shape,
        mode_fraction_of_half_width=context.mode_fraction_of_half_width,
        selection_source=context.selection_source,
    )


def _annotate_rows(
    rows: Sequence[Mapping[str, Any]], frozen: FrozenRole
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(
            {
                **dict(row),
                "context_role": frozen.role,
                "conditioning_role": frozen.role,
                "conditioning_setting_id": frozen.setting.setting_id,
                "conditioning_id": frozen.setting.conditioning_id,
                "gaussian_sigma_px": frozen.setting.gaussian_sigma_px,
                "causal_ema_alpha": frozen.setting.causal_ema_alpha,
                "scale_floor_percentile": frozen.setting.scale_floor_percentile,
                "raw_lane_semantics": upstream.raw_lane_semantics(
                    frozen.representation,
                    frozen.setting.gaussian_sigma_px,
                    frozen.setting.causal_ema_alpha,
                ),
            }
        )
    return result


def _execute_label_free_candidates(
    config: ConditioningProtectedConfig,
    *,
    roles: Sequence[FrozenRole],
    contexts: Mapping[int, ContextLane],
    runtime: Mapping[str, Any],
    heartbeat: Callable[[Mapping[str, Any]], None] | None,
) -> CandidateExecution:
    """Build the complete protected candidate stream without opening labels."""

    import torch

    device = torch.device(str(runtime["resolved_device"]))
    if device.type != "cuda":
        raise ConditioningProtectedUnavailable("protected execution is CUDA-only")
    conditioning_config = config.conditioning_config
    base = conditioning_config.base_config
    folds = {fold.training_fold: fold for fold in build_fold_contracts(base)}
    frame_ui = np.arange(
        upstream.REVIEW_INTERVAL_UI[0],
        upstream.REVIEW_INTERVAL_UI[1] + 1,
        dtype=np.int64,
    )
    bursts = {
        str(key): tuple(map(int, value))
        for key, value in base.payload["frames"]["burst_intervals_ui"].items()
    }
    resources = config.payload["resources"]
    tasks: dict[tuple[Any, ...], list[FrozenRole]] = {}
    for frozen in roles:
        key = (
            frozen.setting.gaussian_sigma_px,
            frozen.setting.causal_ema_alpha,
            frozen.representation,
            frozen.training_fold,
            frozen.setting.setting_id,
        )
        tasks.setdefault(key, []).append(frozen)
    candidates: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    conditioning_timings = []
    representation_timings = []
    completed_tasks = 0
    unique_task_count = len(tasks)
    required_sigmas = sorted({float(key[0]) for key in tasks})
    for sigma in required_sigmas:
        required_alphas = sorted(
            {float(key[1]) for key in tasks if float(key[0]) == sigma},
            reverse=True,
        )
        common_by_alpha, timing_by_alpha = (
            upstream._stream_sigma_all_ema_history_to_device(
                base.source_paths["movie"],
                sigma_px=sigma,
                ema_alphas=required_alphas,
                chunk_frames=int(resources["source_chunk_frames"]),
                device=device,
                heartbeat=heartbeat,
            )
        )
        for alpha in required_alphas:
            common = common_by_alpha.pop(alpha)
            conditioning_timings.append(
                {
                    "conditioning_id": ConditioningSetting(
                        sigma, alpha, 10.0
                    ).conditioning_id,
                    **timing_by_alpha[alpha],
                }
            )
            required_representations = [
                representation
                for representation in REPRESENTATIONS
                if any(
                    float(key[0]) == sigma
                    and float(key[1]) == alpha
                    and str(key[2]) == representation
                    for key in tasks
                )
            ]
            for representation_name in required_representations:
                representation, representation_ms = upstream._elapsed(
                    device,
                    lambda name=representation_name: upstream._form_representation(
                        common, name
                    ),
                )
                representation_timings.append(
                    {
                        "conditioning_id": ConditioningSetting(
                            sigma, alpha, 10.0
                        ).conditioning_id,
                        "representation": representation_name,
                        "representation_ms": representation_ms,
                        "representation_ms_per_frame": representation_ms
                        / len(representation),
                    }
                )
                matching_keys = sorted(
                    (
                        key
                        for key in tasks
                        if float(key[0]) == sigma
                        and float(key[1]) == alpha
                        and str(key[2]) == representation_name
                    ),
                    key=lambda key: (int(key[3]), str(key[4])),
                )
                for key in matching_keys:
                    semantic_roles = sorted(
                        tasks[key], key=lambda row: CONDITIONING_ROLES.index(row.role)
                    )
                    representative = semantic_roles[0]
                    context = contexts[representative.training_fold]
                    lane = _lane_for_role(context, representative.role)
                    candidate_rows, calibration_rows, timing = (
                        _score_candidates_for_arm(
                            representation,
                            arm=representation_name,
                            lane=lane,
                            fold=folds[representative.training_fold],
                            frame_ui=frame_ui,
                            bursts=bursts,
                            scale_floor_percentile=(
                                representative.setting.scale_floor_percentile
                            ),
                            chunk_frames=int(resources["gamma_chunk_frames"]),
                            nms_distances_px=NMS_DISTANCES_PX,
                        )
                    )
                    for frozen in semantic_roles:
                        candidates.extend(_annotate_rows(candidate_rows, frozen))
                        calibrations.extend(_annotate_rows(calibration_rows, frozen))
                        timings.extend(_annotate_rows([timing], frozen))
                    completed_tasks += 1
                    if heartbeat is not None:
                        heartbeat(
                            {
                                "stage": "label_free_candidate_generation",
                                "completed_unique_setting_tasks": completed_tasks,
                                "total_unique_setting_tasks": unique_task_count,
                                "training_fold": representative.training_fold,
                                "representation": representation_name,
                                "conditioning_setting_id": (
                                    representative.setting.setting_id
                                ),
                                "semantic_roles_reusing_result": [
                                    row.role for row in semantic_roles
                                ],
                            }
                        )
                del representation
            del common
    candidates.sort(
        key=lambda row: (
            int(row["training_fold"]),
            str(row["representation"]),
            CONDITIONING_ROLES.index(str(row["conditioning_role"])),
            str(row["quiet_swap"]),
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
            int(row["candidate_rank"]),
        )
    )
    calibrations.sort(
        key=lambda row: (
            int(row["training_fold"]),
            str(row["representation"]),
            CONDITIONING_ROLES.index(str(row["conditioning_role"])),
            str(row["quiet_swap"]),
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
        )
    )
    timings.sort(
        key=lambda row: (
            int(row["training_fold"]),
            str(row["representation"]),
            CONDITIONING_ROLES.index(str(row["conditioning_role"])),
        )
    )
    return CandidateExecution(
        candidate_rows=tuple(candidates),
        calibration_rows=tuple(calibrations),
        timing_rows=tuple(timings),
        summary={
            "semantic_role_cells": len(roles),
            "unique_dense_setting_tasks": unique_task_count,
            "dense_reuse_when_selected_equals_anchor": len(roles)
            - unique_task_count,
            "candidate_rows": len(candidates),
            "calibration_rows": len(calibrations),
            "timing_rows": len(timings),
            "conditioning_timings": conditioning_timings,
            "representation_timings": representation_timings,
            "dense_score_device": "cuda",
            "candidate_extraction_device": "cpu_exact_maintained_nms",
            "labels_opened": False,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
        },
    )


def _validate_candidate_execution(
    execution: CandidateExecution, roles: Sequence[FrozenRole]
) -> dict[str, Any]:
    if len(roles) != EXPECTED_ROLE_CELLS:
        raise ValueError("protected comparison requires exactly 24 semantic role cells")
    if len(execution.calibration_rows) != EXPECTED_OPERATING_ROWS:
        raise ValueError("protected comparison requires exactly 720 operating rows")
    if len(execution.timing_rows) != EXPECTED_ROLE_CELLS:
        raise ValueError("protected comparison requires 24 semantic timing rows")
    expected_dimensions = {
        (
            frozen.training_fold,
            frozen.role,
            frozen.representation,
            swap,
            nms,
            burden,
        )
        for frozen in roles
        for swap in ("a_train_b_test", "b_train_a_test")
        for nms in NMS_DISTANCES_PX
        for burden in QUIET_NMS_PEAK_BURDENS
    }
    actual_dimensions = {
        (
            int(row["training_fold"]),
            str(row["conditioning_role"]),
            str(row["representation"]),
            str(row["quiet_swap"]),
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
        )
        for row in execution.calibration_rows
    }
    if actual_dimensions != expected_dimensions:
        raise ValueError("calibration dimensions do not match the frozen comparison")
    calibration_lookup = {
        (
            int(row["training_fold"]),
            str(row["conditioning_role"]),
            str(row["representation"]),
            str(row["quiet_swap"]),
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
        ): row
        for row in execution.calibration_rows
    }
    role_lookup = {
        (frozen.training_fold, frozen.role, frozen.representation): frozen
        for frozen in roles
    }
    for row in (*execution.calibration_rows, *execution.candidate_rows):
        key = (
            int(row["training_fold"]),
            str(row["conditioning_role"]),
            str(row["representation"]),
        )
        frozen = role_lookup.get(key)
        if frozen is None:
            raise ValueError("candidate output contains an unfrozen role")
        if (
            str(row["conditioning_setting_id"]) != frozen.setting.setting_id
            or str(row["context_id"]) != frozen.support_context_id
        ):
            raise ValueError("candidate output changed a frozen setting/context")
    if any(
        row.get("interpretation_before_label_join") != "unknown_candidate"
        for row in execution.candidate_rows
    ):
        raise ValueError("every candidate must be unknown before the label join")
    candidate_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in execution.candidate_rows:
        dimension = (
            int(row["training_fold"]),
            str(row["conditioning_role"]),
            str(row["representation"]),
            str(row["quiet_swap"]),
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
        )
        operating = calibration_lookup.get(dimension)
        if operating is None:
            raise ValueError("candidate row lacks a frozen operating point")
        if (
            int(row["burst_id"]) != dimension[0]
            or not math.isclose(
                float(row["threshold_z"]),
                float(operating["threshold_z"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isfinite(float(row["occupancy_score"]))
            or not 0 <= int(row["x_px"]) < upstream.EXPECTED_MOVIE_SHAPE[2]
            or not 0 <= int(row["y_px"]) < upstream.EXPECTED_MOVIE_SHAPE[1]
        ):
            raise ValueError("candidate row violates its operating-point contract")
        candidate_groups.setdefault(dimension, []).append(row)
    for dimension, operating in calibration_lookup.items():
        rows = candidate_groups.get(dimension, [])
        rows.sort(key=lambda row: int(row["candidate_rank"]))
        if [int(row["candidate_rank"]) for row in rows] != list(
            range(1, len(rows) + 1)
        ):
            raise ValueError("candidate ranks are not contiguous")
        if len(rows) != int(operating["heldout_burst_candidate_count"]):
            raise ValueError("candidate count does not reconcile to calibration")
    if execution.summary.get("labels_opened") is not False:
        raise ValueError("candidate execution must attest labels were not opened")
    return {
        "semantic_role_cell_count_exact": True,
        "operating_row_count_exact": True,
        "quiet_crossfit_complete": True,
        "nms_4_6_8_complete": True,
        "all_five_quiet_burdens_complete": True,
        "candidate_settings_and_contexts_frozen": True,
        "candidate_rows_reconcile_to_calibration_counts": True,
        "all_candidates_unknown_before_join": True,
    }


def _verify_candidate_seal(root: Path) -> dict[str, Any]:
    seal_path = root / "candidate_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("status") != "sealed_before_label_join"
        or seal.get(
            "label_source_bytes_reopened_or_rehashed_by_this_executor_before_seal"
        )
        is not False
        or seal.get("label_fields_parsed_before_seal") is not False
        or seal.get("unmatched_candidates") != "unknown_not_negative"
    ):
        raise ConditioningProtectedUnavailable("candidate seal boundary is invalid")
    for key in (
        "conditioning_screen_artifact_index_sha256",
        "conditioning_preflight_artifact_index_sha256",
        "conditioning_preflight_sha256",
        "conditioning_selection_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(seal.get(key, ""))) is None:
            raise ConditioningProtectedUnavailable(
                f"candidate seal lacks a valid upstream digest: {key}"
            )
    for key in ("candidate_table", "calibration_table", "frozen_roles"):
        item = seal.get(key)
        if not isinstance(item, Mapping):
            raise ConditioningProtectedUnavailable(f"candidate seal lacks {key}")
        path = root / str(item.get("path", ""))
        if not path.is_file() or upstream._sha256(path) != str(item.get("sha256", "")):
            raise ConditioningProtectedUnavailable(
                f"sealed artifact changed before label join: {key}"
            )
    return {
        "candidate_seal_path": str(seal_path),
        "candidate_seal_file_sha256": upstream._sha256(seal_path),
        "candidate_table_sha256": seal["candidate_table"]["sha256"],
        "calibration_table_sha256": seal["calibration_table"]["sha256"],
        "frozen_roles_sha256": seal["frozen_roles"]["sha256"],
        "conditioning_screen_artifact_index_sha256": seal[
            "conditioning_screen_artifact_index_sha256"
        ],
        "conditioning_preflight_artifact_index_sha256": seal[
            "conditioning_preflight_artifact_index_sha256"
        ],
        "conditioning_preflight_sha256": seal[
            "conditioning_preflight_sha256"
        ],
        "conditioning_selection_sha256": seal["conditioning_selection_sha256"],
        "verified_before_label_join": True,
    }


def _read_labels_after_candidate_seal(
    config: ConditioningProtectedConfig,
    *,
    work_dir: Path,
    conditioning_preflight_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Verify the disk seal, then hash and parse v1/v7 for the first time."""

    seal = _verify_candidate_seal(work_dir)
    conditioning_preflight_root = Path(conditioning_preflight_dir).expanduser().resolve()
    current_conditioning_index = upstream._verify_indexed_artifact(
        conditioning_preflight_root,
        role="conditioning preflight at protected join",
    )
    if current_conditioning_index["artifact_index_sha256"] != seal[
        "conditioning_preflight_artifact_index_sha256"
    ]:
        raise ConditioningProtectedUnavailable(
            "conditioning preflight changed before the protected label join"
        )
    conditioning_preflight_path = conditioning_preflight_root / "preflight.json"
    if upstream._sha256(conditioning_preflight_path) != seal[
        "conditioning_preflight_sha256"
    ]:
        raise ConditioningProtectedUnavailable(
            "conditioning preflight payload changed before the protected label join"
        )
    sensitivity_preflight = json.loads(
        conditioning_preflight_path.read_text(encoding="utf-8")
    )
    base_root = Path(
        str(sensitivity_preflight.get("base_preflight", {}).get("root", ""))
    ).expanduser().resolve()
    if not base_root.is_dir():
        raise ConditioningProtectedUnavailable(
            "conditioning preflight does not resolve a base preflight"
        )
    current_base_index = upstream._verify_indexed_artifact(
        base_root, role="base preflight at protected join"
    )
    if current_base_index["artifact_index_sha256"] != sensitivity_preflight.get(
        "base_preflight", {}
    ).get("artifact_index_sha256"):
        raise ConditioningProtectedUnavailable(
            "base preflight changed before the protected label join"
        )
    base_preflight_path = base_root / "preflight.json"
    base_preflight = json.loads(base_preflight_path.read_text(encoding="utf-8"))
    base = config.conditioning_config.base_config
    labels = config.payload["label_join"]
    expected_sources = {
        "protected_labels_v1": base.source_paths["protected_labels_v1"],
        "latest_labels_v7": base.source_paths["latest_labels_v7"],
    }
    hashes = {}
    for source_name, path in expected_sources.items():
        frozen_source = base_preflight.get("source", {}).get(source_name, {})
        frozen = frozen_source.get("sha256")
        if (
            frozen_source.get("path") != base.payload["sources"][source_name]
            or not frozen
            or upstream._sha256(path) != str(frozen)
        ):
            raise ConditioningProtectedUnavailable(
                f"{source_name} changed after the frozen base preflight"
            )
        hashes[source_name] = str(frozen)
    movie_shape = (upstream.EXPECTED_MOVIE_SHAPE[1], upstream.EXPECTED_MOVIE_SHAPE[2])
    v1 = _read_sparse_positives(
        expected_sources["protected_labels_v1"],
        selector=str(labels["protected_v1_selector"]),
        expected_rows=int(labels["protected_v1_expected_occurrences"]),
        movie_shape_yx=movie_shape,
    )
    v7 = _read_sparse_positives(
        expected_sources["latest_labels_v7"],
        selector=str(labels["latest_v7_selector"]),
        expected_rows=int(labels["latest_v7_expected_occurrences"]),
        movie_shape_yx=movie_shape,
    )
    identity_count = len({str(row["canonical_roi_id"]) for row in v1})
    if identity_count != int(labels["protected_v1_expected_canonical_identities"]):
        raise ConditioningProtectedUnavailable(
            f"protected v1 identity count changed: {identity_count}"
        )
    return v1, v7, {
        **seal,
        "base_preflight_artifact_index_sha256": current_base_index[
            "artifact_index_sha256"
        ],
        "label_source_hashes": hashes,
        "label_source_bytes_first_rehashed_by_this_executor_after_candidate_seal": True,
        "label_fields_first_parsed_after_candidate_seal": True,
        "protected_v1_occurrences": len(v1),
        "protected_v1_canonical_identities": identity_count,
        "latest_v7_occurrences": len(v7),
        "latest_v7_inferential_claim": False,
    }


def _role_macro_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    cluster_weights: Mapping[str, int],
) -> tuple[float, float, dict[int, float]]:
    recall_by_budget: dict[int, float] = {}
    b58_by_burst: dict[int, float] = {}
    for budget in CANDIDATE_BUDGETS_PER_BURST:
        burst_values = []
        for burst in upstream.OUTER_FOLDS:
            selected = [
                row
                for row in rows
                if str(row["context_role"]) == role
                and int(row["candidate_budget"]) == budget
                and int(row["burst_id"]) == burst
            ]
            if not selected:
                raise ValueError("bootstrap metric is missing a role/budget/burst cell")
            numerator = sum(
                int(cluster_weights.get(str(row["canonical_roi_id"]), 0))
                * bool(row["matched"])
                for row in selected
            )
            denominator = sum(
                int(cluster_weights.get(str(row["canonical_roi_id"]), 0))
                for row in selected
            )
            value = numerator / denominator if denominator else float("nan")
            burst_values.append(value)
            if budget == 58:
                b58_by_burst[burst] = value
        recall_by_budget[budget] = float(np.nanmean(burst_values))
    x = np.asarray(CANDIDATE_BUDGETS_PER_BURST, dtype=np.float64)
    y = np.asarray([recall_by_budget[budget] for budget in x], dtype=np.float64)
    auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    return auc, recall_by_budget[58], b58_by_burst


def _paired_role_metric_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    identities: Sequence[str],
    bootstrap_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized AUC, B58, and observed burst-B58 for paired role weights."""

    role_index = {role: index for index, role in enumerate(CONDITIONING_ROLES)}
    budget_index = {
        budget: index for index, budget in enumerate(CANDIDATE_BUDGETS_PER_BURST)
    }
    identity_index = {identity: index for index, identity in enumerate(identities)}
    matched = np.zeros(
        (
            len(CONDITIONING_ROLES),
            len(CANDIDATE_BUDGETS_PER_BURST),
            len(upstream.OUTER_FOLDS),
            len(identities),
        ),
        dtype=np.float64,
    )
    denominator = np.zeros_like(matched)
    for row in rows:
        role = role_index[str(row["context_role"])]
        budget = budget_index[int(row["candidate_budget"])]
        burst = upstream.OUTER_FOLDS.index(int(row["burst_id"]))
        identity = identity_index[str(row["canonical_roi_id"])]
        denominator[role, budget, burst, identity] += 1.0
        matched[role, budget, burst, identity] += float(bool(row["matched"]))
    if np.any(denominator.sum(axis=-1) == 0.0):
        raise ValueError("paired bootstrap subset is incomplete")
    weights = np.asarray(bootstrap_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(identities):
        raise ValueError("bootstrap weight matrix has the wrong identity axis")
    numerators = np.einsum("qi,rbti->qrbt", weights, matched)
    denominators = np.einsum("qi,rbti->qrbt", weights, denominator)
    burst_recall = np.divide(
        numerators,
        denominators,
        out=np.full_like(numerators, np.nan),
        where=denominators > 0.0,
    )
    macro = np.nanmean(burst_recall, axis=-1)
    x = np.asarray(CANDIDATE_BUDGETS_PER_BURST, dtype=np.float64)
    auc = np.trapezoid(macro, x, axis=-1) / (x[-1] - x[0])
    b58_index = CANDIDATE_BUDGETS_PER_BURST.index(58)
    return auc, macro[:, :, b58_index], burst_recall[0, :, b58_index, :]


def conditioning_clustered_bootstrap_contrasts(
    v1_match_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[dict[str, Any]]:
    """Paired selected-minus-anchor curves over 26 identity clusters."""

    identities = sorted({str(row["canonical_roi_id"]) for row in v1_match_rows})
    if len(identities) != BOOTSTRAP_CLUSTER_COUNT:
        raise ValueError(
            f"paired bootstrap requires 26 identities, got {len(identities)}"
        )
    if int(seed) != BOOTSTRAP_SEED or int(replicates) < 2:
        raise ValueError("bootstrap seed changed or replicate count is invalid")
    present_roles = {str(row["context_role"]) for row in v1_match_rows}
    if present_roles != set(CONDITIONING_ROLES):
        raise ValueError("paired bootstrap requires selected and anchor roles")
    base_dimensions = sorted(
        {
            (
                str(row["representation"]),
                str(row["quiet_swap"]),
                int(row["nms_distance_px"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            )
            for row in v1_match_rows
        }
    )
    pooled_dimensions = sorted(
        {
            (
                str(row["representation"]),
                "crossfit_average",
                int(row["nms_distance_px"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            )
            for row in v1_match_rows
        }
    )
    dimensions = base_dimensions + pooled_dimensions
    rng = np.random.default_rng(int(seed))
    bootstrap_weights = np.zeros(
        (int(replicates), len(identities)), dtype=np.int16
    )
    for replicate in range(int(replicates)):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        for identity, count in zip(unique.tolist(), counts.tolist()):
            bootstrap_weights[replicate, identities.index(identity)] = int(count)
    all_weights = np.vstack(
        (np.ones((1, len(identities)), dtype=np.int16), bootstrap_weights)
    )
    output = []
    for representation, quiet_scope, nms_distance, burden in dimensions:
        subset = [
            row
            for row in v1_match_rows
            if str(row["representation"]) == representation
            and (
                quiet_scope == "crossfit_average"
                or str(row["quiet_swap"]) == quiet_scope
            )
            and int(row["nms_distance_px"]) == nms_distance
            and float(row["target_nms_peaks_per_pseudo_burst"]) == burden
        ]
        auc, b58, burst_b58 = _paired_role_metric_arrays(
            subset, identities=identities, bootstrap_weights=all_weights
        )
        anchor_index = CONDITIONING_ROLES.index("historical_anchor")
        selected_index = CONDITIONING_ROLES.index("training_pareto_selected")
        anchor_auc = float(auc[0, anchor_index])
        selected_auc = float(auc[0, selected_index])
        anchor_b58 = float(b58[0, anchor_index])
        selected_b58 = float(b58[0, selected_index])
        anchor_bursts = burst_b58[anchor_index]
        selected_bursts = burst_b58[selected_index]
        auc_deltas = auc[1:, selected_index] - auc[1:, anchor_index]
        b58_deltas = b58[1:, selected_index] - b58[1:, anchor_index]
        auc_delta = selected_auc - anchor_auc
        auc_low = float(np.nanpercentile(auc_deltas, 2.5))
        auc_high = float(np.nanpercentile(auc_deltas, 97.5))
        if auc_delta >= MATERIAL_AUC_DELTA and auc_low > 0.0:
            direction = "material_improvement"
        elif auc_delta <= -MATERIAL_AUC_DELTA and auc_high < 0.0:
            direction = "material_degradation"
        else:
            direction = "no_material_change_at_this_operating_point"
        output.append(
            {
                "representation": representation,
                "quiet_scope": quiet_scope,
                "nms_distance_px": nms_distance,
                "nms_role": (
                    "primary" if nms_distance == NMS_DISTANCE_PX else "descriptive_sensitivity"
                ),
                "target_nms_peaks_per_pseudo_burst": burden,
                "selected_budget_curve_auc": selected_auc,
                "anchor_budget_curve_auc": anchor_auc,
                "selected_minus_anchor_budget_curve_auc": auc_delta,
                "budget_curve_auc_delta_ci95_low": auc_low,
                "budget_curve_auc_delta_ci95_high": auc_high,
                "selected_b58_macro_recall": selected_b58,
                "anchor_b58_macro_recall": anchor_b58,
                "selected_minus_anchor_b58_macro_recall": selected_b58
                - anchor_b58,
                "b58_delta_ci95_low": float(np.nanpercentile(b58_deltas, 2.5)),
                "b58_delta_ci95_high": float(np.nanpercentile(b58_deltas, 97.5)),
                "selected_b58_burst_wins": sum(
                    selected_bursts[index] > anchor_bursts[index]
                    for index, _ in enumerate(upstream.OUTER_FOLDS)
                ),
                "selected_b58_burst_ties": sum(
                    selected_bursts[index] == anchor_bursts[index]
                    for index, _ in enumerate(upstream.OUTER_FOLDS)
                ),
                "materiality_threshold_absolute_auc_delta": MATERIAL_AUC_DELTA,
                "materiality_direction": direction,
                "cluster_field": "canonical_roi_id",
                "cluster_count": len(identities),
                "bootstrap_seed": int(seed),
                "bootstrap_replicates": int(replicates),
                "v7_inferential_claim": False,
            }
        )
    return output


def protected_curve_materiality(
    contrast_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen all-five-burdens primary-scope materiality rule."""

    decisions = []
    for representation in REPRESENTATIONS:
        primary = [
            row
            for row in contrast_rows
            if str(row["representation"]) == representation
            and str(row["quiet_scope"]) == "crossfit_average"
            and int(row["nms_distance_px"]) == NMS_DISTANCE_PX
        ]
        if {
            float(row["target_nms_peaks_per_pseudo_burst"]) for row in primary
        } != set(QUIET_NMS_PEAK_BURDENS) or len(primary) != len(
            QUIET_NMS_PEAK_BURDENS
        ):
            raise ValueError("materiality summary requires five primary burden rows")
        directions = [str(row["materiality_direction"]) for row in primary]
        if all(direction == "material_improvement" for direction in directions):
            conclusion = "material_improvement_across_curve_family"
        elif all(direction == "material_degradation" for direction in directions):
            conclusion = "material_degradation_across_curve_family"
        else:
            conclusion = "no_consistent_material_change_across_curve_family"
        decisions.append(
            {
                "representation": representation,
                "primary_scope": "crossfit_average_nms6",
                "quiet_burdens": list(QUIET_NMS_PEAK_BURDENS),
                "per_burden_directions": {
                    str(row["target_nms_peaks_per_pseudo_burst"]): row[
                        "materiality_direction"
                    ]
                    for row in sorted(
                        primary,
                        key=lambda item: float(
                            item["target_nms_peaks_per_pseudo_burst"]
                        ),
                    )
                },
                "conclusion": conclusion,
                "materially_changes_protected_curves": conclusion
                != "no_consistent_material_change_across_curve_family",
            }
        )
    return {
        "schema_version": 1,
        "primary_scope": "crossfit_average_nms6",
        "material_auc_delta": MATERIAL_AUC_DELTA,
        "requires_ci_excluding_zero": True,
        "representation_level_rule": (
            "all_five_quiet_burdens_same_material_direction"
        ),
        "decisions": decisions,
        "any_representation_materially_changes_protected_curves": any(
            row["materially_changes_protected_curves"] for row in decisions
        ),
    }


def _report_markdown(materiality: Mapping[str, Any]) -> str:
    lines = [
        "# Protected conditioning comparison v1",
        "",
        "## Outcome",
        "",
        "| Representation | Primary protected-curve conclusion |",
        "|---|---|",
    ]
    for row in materiality["decisions"]:
        lines.append(f"| {row['representation']} | {row['conclusion']} |")
    lines.extend(
        [
            "",
            "The primary materiality scope is the crossfit-average NMS-6 budget "
            "curve at each of five quiet burdens. NMS-4 and NMS-8 are descriptive "
            "sensitivity checks.",
            "",
            "## Claim boundary",
            "",
            "Candidates were sealed before either v1 or v7 was reopened, rehashed, "
            "or parsed by this executor. "
            "V1 supports known-positive sensitivity and paired identity-cluster "
            "contrasts. V7 is descriptive only. Sparse positives do not identify "
            "proposal precision; every unpaired proposal remains unknown.",
            "",
            "Scientific promotion remains pending the candidate-surrogate montage, "
            "expert review ledger, and failure taxonomy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_conditioning_protected(
    config: ConditioningProtectedConfig,
    *,
    conditioning_preflight_dir: str | Path,
    conditioning_screen_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate, seal, join, and report the frozen protected comparison."""

    if not isinstance(config, ConditioningProtectedConfig):
        raise TypeError("config must be ConditioningProtectedConfig")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"protected conditioning output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"output parent is missing: {destination.parent}")
    roles, contexts, upstream_provenance = _verify_conditioning_screen(
        config,
        conditioning_preflight_dir=conditioning_preflight_dir,
        conditioning_screen_dir=conditioning_screen_dir,
    )
    resources = config.payload["resources"]
    free_disk = shutil.disk_usage(destination.parent).free
    minimum_disk = int(float(resources["minimum_free_disk_gib"]) * 2**30)
    if free_disk < minimum_disk:
        raise ConditioningProtectedUnavailable(
            f"free disk {free_disk} is below the frozen minimum {minimum_disk} bytes"
        )
    runtime = _verify_current_cuda_runtime(
        config, upstream_provenance["conditioning_preflight_runtime"]
    )
    upstream_provenance = {
        **upstream_provenance,
        "execution_runtime": runtime,
        "runtime_identity_reverified_after_upstream_artifact_verification": True,
    }
    implementation = _implementation_hashes(config)
    partial = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        partial.mkdir()

        def heartbeat(payload: Mapping[str, Any]) -> None:
            upstream._atomic_json(
                partial / "heartbeat.json",
                {
                    "experiment_id": EXPERIMENT_ID,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    **dict(payload),
                },
            )

        frozen_roles_payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "conditioning_selection_sha256": upstream_provenance[
                "conditioning_selection_sha256"
            ],
            "roles": [role.as_dict() for role in roles],
            "positive_coordinates_used_for_selection": False,
            "positive_identities_used_for_selection": False,
        }
        upstream._atomic_json(partial / "frozen_roles.json", frozen_roles_payload)
        execution = _execute_label_free_candidates(
            config,
            roles=roles,
            contexts=contexts,
            runtime=runtime,
            heartbeat=heartbeat,
        )
        candidate_checks = _validate_candidate_execution(execution, roles)
        upstream._atomic_tsv(
            partial / "candidates_label_sealed.tsv", execution.candidate_rows
        )
        upstream._atomic_tsv(
            partial / "threshold_calibration.tsv", execution.calibration_rows
        )
        upstream._atomic_tsv(partial / "timings.tsv", execution.timing_rows)
        upstream._atomic_json(
            partial / "candidate_execution_summary.json", dict(execution.summary)
        )
        if implementation != _implementation_hashes(config):
            raise ConditioningProtectedUnavailable(
                "protected implementation changed during candidate generation"
            )
        reverified_roles, reverified_contexts, reverified_upstream = (
            _verify_conditioning_screen(
                config,
                conditioning_preflight_dir=conditioning_preflight_dir,
                conditioning_screen_dir=conditioning_screen_dir,
            )
        )
        immutable_upstream_keys = (
            "artifact_index_sha256",
            "conditioning_preflight_sha256",
            "conditioning_preflight_artifact_index_sha256",
            "base_preflight_artifact_index_sha256",
            "support_artifact_index_sha256",
            "conditioning_selection_sha256",
        )
        if (
            reverified_roles != roles
            or reverified_contexts != contexts
            or any(
                reverified_upstream.get(key) != upstream_provenance.get(key)
                for key in immutable_upstream_keys
            )
        ):
            raise ConditioningProtectedUnavailable(
                "frozen upstream artifacts changed during candidate generation"
            )
        candidate_seal = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "sealed_before_label_join",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "conditioning_screen_artifact_index_sha256": upstream_provenance[
                "artifact_index_sha256"
            ],
            "conditioning_preflight_sha256": upstream_provenance[
                "conditioning_preflight_sha256"
            ],
            "conditioning_preflight_artifact_index_sha256": upstream_provenance[
                "conditioning_preflight_artifact_index_sha256"
            ],
            "conditioning_selection_sha256": upstream_provenance[
                "conditioning_selection_sha256"
            ],
            "implementation": implementation,
            "candidate_table": {
                "path": "candidates_label_sealed.tsv",
                "sha256": upstream._sha256(
                    partial / "candidates_label_sealed.tsv"
                ),
                "rows": len(execution.candidate_rows),
            },
            "calibration_table": {
                "path": "threshold_calibration.tsv",
                "sha256": upstream._sha256(partial / "threshold_calibration.tsv"),
                "rows": len(execution.calibration_rows),
            },
            "frozen_roles": {
                "path": "frozen_roles.json",
                "sha256": upstream._sha256(partial / "frozen_roles.json"),
                "rows": len(roles),
            },
            "base_preflight_label_hash_metadata_available_before_seal": True,
            (
                "label_source_bytes_reopened_or_rehashed_by_this_executor_before_seal"
            ): False,
            "label_fields_parsed_before_seal": False,
            "all_candidates_unknown_before_seal": True,
            "unmatched_candidates": "unknown_not_negative",
        }
        upstream._atomic_json(partial / "candidate_seal.json", candidate_seal)
        heartbeat(
            {
                "stage": "candidate_sealed_before_label_join",
                "candidate_seal_file_sha256": upstream._sha256(
                    partial / "candidate_seal.json"
                ),
            }
        )
        v1, v7, label_join = _read_labels_after_candidate_seal(
            config,
            work_dir=partial,
            conditioning_preflight_dir=conditioning_preflight_dir,
        )
        v1_matches = observation_match_rows(
            execution.candidate_rows,
            v1,
            cohort="protected_v1",
            operating_rows=execution.calibration_rows,
        )
        v7_matches = observation_match_rows(
            execution.candidate_rows,
            v7,
            cohort="latest_v7_sensitivity",
            operating_rows=execution.calibration_rows,
        )
        v1_aggregate = aggregate_match_rows(v1_matches)
        v7_aggregate = aggregate_match_rows(v7_matches)
        v1_summary = _summary_by_arm(v1_matches)
        v7_summary = _summary_by_arm(v7_matches)
        contrasts = conditioning_clustered_bootstrap_contrasts(
            v1_matches,
            seed=int(config.payload["bootstrap"]["seed"]),
            replicates=int(config.payload["bootstrap"]["replicates"]),
        )
        if len(contrasts) != EXPECTED_BOOTSTRAP_ROWS:
            raise ConditioningProtectedUnavailable(
                "paired bootstrap did not return the frozen 135-row contrast family"
            )
        materiality = protected_curve_materiality(contrasts)
        upstream._atomic_tsv(
            partial / "protected_v1_observation_matches.tsv", v1_matches
        )
        upstream._atomic_tsv(partial / "protected_v1_recall.tsv", v1_aggregate)
        upstream._atomic_tsv(
            partial / "protected_v1_role_summary.tsv", v1_summary
        )
        upstream._atomic_tsv(
            partial / "protected_v1_paired_bootstrap.tsv", contrasts
        )
        upstream._atomic_tsv(
            partial / "latest_v7_observation_matches.tsv", v7_matches
        )
        upstream._atomic_tsv(partial / "latest_v7_sensitivity.tsv", v7_aggregate)
        upstream._atomic_tsv(
            partial / "latest_v7_role_summary.tsv", v7_summary
        )
        upstream._atomic_json(
            partial / "conditioning_curve_materiality.json", materiality
        )
        upstream._atomic_json(partial / "label_join_provenance.json", label_join)
        claim_boundary = {
            "protected_v1_known_positive_sensitivity_computed": True,
            "paired_26_identity_cluster_bootstrap_computed": True,
            "conditioning_curve_materiality_reported": True,
            "latest_v7_descriptive_sensitivity_computed": True,
            "latest_v7_inferential_claim": False,
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "proposal_identity_claimed": False,
            "scientific_audit_complete": False,
            "interpretation": (
                "Protected known-positive curve comparisons are complete, but "
                "unmatched proposals remain unknown and scientific promotion "
                "awaits candidate-surrogate expert audit."
            ),
        }
        upstream._atomic_json(partial / "claim_boundary.json", claim_boundary)
        audit = {
            "schema_version": 1,
            "status": "pending_candidate_surrogate_expert_audit",
            "candidate_surrogate_montage": "pending",
            "expert_review_ledger": "pending",
            "failure_taxonomy": "pending",
            "metric_artifact_complete": True,
            "scientific_promotion_allowed": False,
        }
        upstream._atomic_json(partial / "scientific_audit_status.json", audit)
        checks = {
            **candidate_checks,
            "conditioning_screen_and_preflight_hashes_bound": True,
            "candidate_seal_precedes_v1_v7_join": label_join[
                "label_fields_first_parsed_after_candidate_seal"
            ],
            "protected_v1_uses_79_inclusive": len(v1) == 79,
            "protected_v1_has_26_identity_clusters": len(
                {str(row["canonical_roi_id"]) for row in v1}
            )
            == 26,
            "latest_v7_uses_106_confirmed": len(v7) == 106,
            "paired_bootstrap_uses_2000_replicates": all(
                int(row["bootstrap_replicates"]) == BOOTSTRAP_REPLICATES
                for row in contrasts
            ),
            "unmatched_candidates_remain_unknown": True,
            "precision_not_claimed": True,
            "scientific_audit_pending": True,
        }
        if not all(checks.values()):
            raise ConditioningProtectedUnavailable(
                "protected artifact validation did not pass"
            )
        summary = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "complete_protected_conditioning_metrics_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "upstream": upstream_provenance,
            "candidate_execution": dict(execution.summary),
            "candidate_seal": candidate_seal,
            "label_join": label_join,
            "protected_v1": {
                "occurrences": len(v1),
                "canonical_identities": len(
                    {str(row["canonical_roi_id"]) for row in v1}
                ),
                "paired_bootstrap_rows": len(contrasts),
            },
            "latest_v7_sensitivity": {
                "occurrences": len(v7),
                "inferential_claim": False,
            },
            "materiality": materiality,
            "claim_boundary": claim_boundary,
        }
        upstream._atomic_json(partial / "summary.json", summary)
        upstream._atomic_json(
            partial / "validation.json",
            {
                "status": (
                    "passed_protected_conditioning_metric_artifact_contract_"
                    "scientific_audit_pending"
                ),
                "checks": checks,
                "all_checks_pass": True,
            },
        )
        upstream._atomic_json(
            partial / "provenance_hashes.json",
            {
                "protected_manifest_sha256": upstream._sha256(config.manifest_path),
                "conditioning_screen_artifact_index_sha256": upstream_provenance[
                    "artifact_index_sha256"
                ],
                "conditioning_preflight_sha256": upstream_provenance[
                    "conditioning_preflight_sha256"
                ],
                "conditioning_selection_sha256": upstream_provenance[
                    "conditioning_selection_sha256"
                ],
                "candidate_seal_file_sha256": upstream._sha256(
                    partial / "candidate_seal.json"
                ),
                "label_source_hashes": label_join["label_source_hashes"],
                "implementation": implementation,
            },
        )
        upstream._atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "primary_result": "conditioning_curve_materiality.json",
                "paired_inference": "protected_v1_paired_bootstrap.tsv",
                "candidate_seal": "candidate_seal.json",
                "v7_role": "descriptive_sensitivity_only",
                "unmatched_candidates": "unknown_not_negative",
                "scientific_audit": "pending",
            },
        )
        upstream._atomic_text(partial / "REPORT.md", _report_markdown(materiality))
        heartbeat(
            {
                "stage": "complete",
                "status": summary["status"],
                "candidate_rows": len(execution.candidate_rows),
                "paired_bootstrap_rows": len(contrasts),
            }
        )
        upstream._atomic_json(
            partial / "artifact_index.json", upstream._artifact_index(partial)
        )
        if destination.exists():
            raise FileExistsError("protected destination appeared during commit")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protected Gamma-LS conditioning comparison"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--conditioning-preflight", required=True)
    parser.add_argument("--conditioning-screen", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = ConditioningProtectedConfig.load(arguments.config)
    payload = run_conditioning_protected(
        config,
        conditioning_preflight_dir=arguments.conditioning_preflight,
        conditioning_screen_dir=arguments.conditioning_screen,
        output_dir=arguments.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CONDITIONING_ROLES",
    "CandidateExecution",
    "ConditioningProtectedConfig",
    "ConditioningProtectedConfigError",
    "ConditioningProtectedUnavailable",
    "FrozenRole",
    "NMS_DISTANCES_PX",
    "conditioning_clustered_bootstrap_contrasts",
    "main",
    "protected_curve_materiality",
    "run_conditioning_protected",
]


if __name__ == "__main__":
    raise SystemExit(main())
