"""Nested, coordinate-free conditioning sensitivity for radial Gamma-LS.

This upstream screen varies spatial Gaussian conditioning, causal EMA
conditioning, and the training-quiet local-scale floor.  It uses the human-
declared burst *time windows* but has no API for sparse-positive coordinates or
identities.  Every outer fold consumes its previously frozen radial
``support_candidate_context`` and selects one setting independently for each
nonlearned representation.  The historical conditioning/floor anchor and the
fold-local Pareto selection are sealed for a later protected B/NMS evaluation.

Only ``raw`` with sigma zero and EMA alpha one is acquisition raw (after the
lossless uint16-to-float32 cast).  The historically named raw lane is spatially
and temporally conditioned; output rows make that distinction explicit.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
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
from neurobench.portable_paths import data_root

from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .preflight import _implementation_status
from .screen import (
    ENERGY_EPSILON,
    GAMMA_EPSILON,
    QUIET_SWAPS,
    TAIL_QUANTILE_METHOD,
    _elapsed,
    _interval_mask,
    _positive_scale_floor,
    _tail_metrics,
    build_fold_contracts,
)


EXPERIMENT_ID = "spon_ca_burst_gamma_ls_conditioning_sensitivity_v1"
GAUSSIAN_SIGMAS = (0.0, 0.5, 1.0, 1.5)
EMA_ALPHAS = (1.0, 2.0 / 3.0, 0.4, 0.25)
SCALE_FLOOR_PERCENTILES = (5.0, 10.0, 20.0)
REPRESENTATIONS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
)
OUTER_FOLDS = (1, 2, 3, 4)
ANCHOR = (1.0, 0.4, 10.0)
GAUSSIAN_TRUNCATE = 4.0
EXPECTED_MOVIE_SHAPE = (2359, 340, 573)
EXPECTED_MOVIE_DTYPE = "uint16"
REVIEW_INTERVAL_UI = (1800, 2359)
RETAINED_INTERVAL_UI = (1799, 2359)
EXPECTED_FOLD_CELLS = 576
EXPECTED_SWAP_ROWS = 1152
EXPECTED_GRID_ROWS = 144

_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "sources",
    "design",
    "downstream",
    "resources",
    "scientific_audit",
}
_SOURCE_KEYS = {"base_config", "support_screen"}
_DESIGN_KEYS = {
    "gaussian_sigma_px",
    "causal_ema_alpha",
    "training_quiet_scale_floor_percentile",
    "representations",
    "outer_folds",
    "support_context_role",
    "quiet_calibration",
    "tail_quantile",
    "tail_population",
    "event_aggregation",
    "selection_scope",
    "selection_objectives",
    "pareto_tie_break",
    "anchor",
    "acquisition_raw_definition",
    "historical_raw_lane_definition",
}
_DOWNSTREAM_KEYS = {
    "freeze_roles",
    "next_stage",
    "protected_evaluation_in_this_run",
}
_RESOURCE_KEYS = {
    "device",
    "source_chunk_frames",
    "gamma_chunk_frames",
    "max_peak_vram_gib",
    "minimum_free_disk_gib",
}
_ANCHOR_KEYS = {
    "gaussian_sigma_px",
    "causal_ema_alpha",
    "training_quiet_scale_floor_percentile",
}


class ConditioningSensitivityUnavailable(RuntimeError):
    """Raised before destination mutation when a frozen gate fails."""


class ConditioningSensitivityConfigError(ValueError):
    """Raised when the strict conditioning manifest is not the v1 design."""


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], scope: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = ",".join(sorted(expected - actual))
        unknown = ",".join(sorted(actual - expected))
        raise ConditioningSensitivityConfigError(
            f"invalid {scope} fields: missing={missing or '-'}; unknown={unknown or '-'}"
        )


def _resolve_uri(value: str, *, repository: Path, authority: Path) -> Path:
    if value.startswith("repo://"):
        return (repository / value.removeprefix("repo://")).resolve()
    if value.startswith("data://"):
        return (authority / value.removeprefix("data://")).resolve()
    raise ConditioningSensitivityConfigError(
        f"paths must use repo:// or data:// identifiers, got {value!r}"
    )


@dataclass(frozen=True)
class ConditioningSensitivityConfig:
    """Validated manifest plus resolved source paths."""

    manifest_path: Path
    repository: Path
    authority: Path
    payload: dict[str, Any]
    base_config_path: Path
    support_screen_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "ConditioningSensitivityConfig":
        manifest = Path(path).expanduser().resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConditioningSensitivityConfigError("manifest must be an object")
        _require_exact_keys(raw, _TOP_LEVEL_KEYS, "top-level")
        repository = Path(__file__).resolve().parents[3]
        authority = data_root(repository)
        sources = raw["sources"]
        if not isinstance(sources, dict):
            raise ConditioningSensitivityConfigError("sources must be an object")
        _require_exact_keys(sources, _SOURCE_KEYS, "sources")
        config = cls(
            manifest_path=manifest,
            repository=repository,
            authority=authority,
            payload=deepcopy(raw),
            base_config_path=_resolve_uri(
                str(sources["base_config"]),
                repository=repository,
                authority=authority,
            ),
            support_screen_path=_resolve_uri(
                str(sources["support_screen"]),
                repository=repository,
                authority=authority,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        raw = self.payload
        if raw["schema_version"] != 1 or raw["experiment_id"] != EXPERIMENT_ID:
            raise ConditioningSensitivityConfigError("unexpected schema or experiment id")
        design = raw["design"]
        if not isinstance(design, dict):
            raise ConditioningSensitivityConfigError("design must be an object")
        _require_exact_keys(design, _DESIGN_KEYS, "design")
        if tuple(map(float, design["gaussian_sigma_px"])) != GAUSSIAN_SIGMAS:
            raise ConditioningSensitivityConfigError("Gaussian sigma grid changed")
        if tuple(map(float, design["causal_ema_alpha"])) != EMA_ALPHAS:
            raise ConditioningSensitivityConfigError("causal EMA alpha grid changed")
        if tuple(map(float, design["training_quiet_scale_floor_percentile"])) != (
            SCALE_FLOOR_PERCENTILES
        ):
            raise ConditioningSensitivityConfigError("scale-floor grid changed")
        if tuple(design["representations"]) != REPRESENTATIONS:
            raise ConditioningSensitivityConfigError("representation grid changed")
        expected_literals = {
            "outer_folds": "leave_one_burst_out",
            "support_context_role": "support_candidate_context",
            "quiet_calibration": "split_contiguous_halves_and_reverse_roles",
            "tail_population": "all_pixels",
            "event_aggregation": "equal_mean_of_three_training_burst_frame_tails",
            "selection_scope": "independent_per_representation_within_outer_training_fold",
            "acquisition_raw_definition": (
                "raw_representation_with_gaussian_sigma0_and_ema_alpha1"
            ),
            "historical_raw_lane_definition": (
                "raw_representation_with_gaussian_sigma1_and_ema_alpha0p4"
            ),
        }
        if any(design.get(key) != value for key, value in expected_literals.items()):
            raise ConditioningSensitivityConfigError("nested design semantics changed")
        if float(design["tail_quantile"]) != 0.999:
            raise ConditioningSensitivityConfigError("tail quantile changed")
        if design["selection_objectives"] != [
            "maximize_mean_event_minus_quiet_tail_contrast",
            "maximize_minimum_quiet_swap_tail_contrast",
            "minimize_online_inference_runtime_ms_per_frame",
        ]:
            raise ConditioningSensitivityConfigError("Pareto objectives changed")
        if design["pareto_tie_break"] != [
            "highest_mean_tail_contrast",
            "highest_minimum_swap_tail_contrast",
            "lowest_online_inference_runtime_ms_per_frame",
            "lexicographic_setting_id",
        ]:
            raise ConditioningSensitivityConfigError("Pareto tie-break changed")
        anchor = design["anchor"]
        if not isinstance(anchor, dict):
            raise ConditioningSensitivityConfigError("anchor must be an object")
        _require_exact_keys(anchor, _ANCHOR_KEYS, "design.anchor")
        frozen_anchor = (
            float(anchor["gaussian_sigma_px"]),
            float(anchor["causal_ema_alpha"]),
            float(anchor["training_quiet_scale_floor_percentile"]),
        )
        if frozen_anchor != ANCHOR:
            raise ConditioningSensitivityConfigError("historical anchor changed")
        downstream = raw["downstream"]
        if not isinstance(downstream, dict):
            raise ConditioningSensitivityConfigError("downstream must be an object")
        _require_exact_keys(downstream, _DOWNSTREAM_KEYS, "downstream")
        if downstream != {
            "freeze_roles": ["historical_anchor", "training_pareto_selected"],
            "next_stage": "protected_B_and_NMS_evaluation",
            "protected_evaluation_in_this_run": False,
        }:
            raise ConditioningSensitivityConfigError("downstream boundary changed")
        resources = raw["resources"]
        if not isinstance(resources, dict):
            raise ConditioningSensitivityConfigError("resources must be an object")
        _require_exact_keys(resources, _RESOURCE_KEYS, "resources")
        device = resources["device"]
        if not isinstance(device, str) or re.fullmatch(
            r"cuda(?::(?:0|[1-9][0-9]*))?", device
        ) is None:
            raise ConditioningSensitivityConfigError("v1 is CUDA-only")
        if int(resources["source_chunk_frames"]) != 64 or int(
            resources["gamma_chunk_frames"]
        ) != 64:
            raise ConditioningSensitivityConfigError("chunk contract changed")
        if float(resources["max_peak_vram_gib"]) != 8.0 or float(
            resources["minimum_free_disk_gib"]
        ) != 20.0:
            raise ConditioningSensitivityConfigError("resource caps changed")
        if raw["scientific_audit"] != {"enabled": True}:
            raise ConditioningSensitivityConfigError("scientific audit is mandatory")

    def portable_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)

    @property
    def base_config(self) -> GammaLSDifferenceConfig:
        return GammaLSDifferenceConfig.load(self.base_config_path)


@dataclass(frozen=True, order=True)
class ConditioningSetting:
    """One fixed conditioning and scale-floor setting."""

    gaussian_sigma_px: float
    causal_ema_alpha: float
    scale_floor_percentile: float

    @property
    def conditioning_id(self) -> str:
        return (
            f"sigma{_token(self.gaussian_sigma_px)}_"
            f"alpha{_alpha_token(self.causal_ema_alpha)}"
        )

    @property
    def setting_id(self) -> str:
        return f"{self.conditioning_id}_p{_token(self.scale_floor_percentile)}"

    @property
    def is_anchor(self) -> bool:
        return (
            self.gaussian_sigma_px,
            self.causal_ema_alpha,
            self.scale_floor_percentile,
        ) == ANCHOR

    def as_dict(self) -> dict[str, Any]:
        return {
            "setting_id": self.setting_id,
            "conditioning_id": self.conditioning_id,
            "gaussian_sigma_px": self.gaussian_sigma_px,
            "causal_ema_alpha": self.causal_ema_alpha,
            "training_quiet_scale_floor_percentile": self.scale_floor_percentile,
            "historical_anchor": self.is_anchor,
        }


@dataclass(frozen=True)
class SensitivityExecution:
    """Host-resident metric tables returned by the CUDA executor."""

    fold_rows: tuple[Mapping[str, Any], ...]
    swap_rows: tuple[Mapping[str, Any], ...]
    timing_rows: tuple[Mapping[str, Any], ...]
    execution_summary: Mapping[str, Any]


def _token(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g").replace(".", "p")


def _alpha_token(value: float) -> str:
    if math.isclose(float(value), 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-15):
        return "2over3"
    return _token(value)


def enumerate_settings() -> tuple[ConditioningSetting, ...]:
    """Return the frozen 48-setting grid in deterministic order."""

    return tuple(
        ConditioningSetting(sigma, alpha, percentile)
        for sigma in GAUSSIAN_SIGMAS
        for alpha in EMA_ALPHAS
        for percentile in SCALE_FLOOR_PERCENTILES
    )


def raw_lane_semantics(
    representation: str, gaussian_sigma_px: float, causal_ema_alpha: float
) -> str:
    """Disambiguate acquisition raw from the historically conditioned lane."""

    if representation != "raw":
        return "not_a_raw_representation_lane"
    sigma = float(gaussian_sigma_px)
    alpha = float(causal_ema_alpha)
    if sigma == 0.0 and alpha == 1.0:
        return "acquisition_raw_float32"
    if sigma == 1.0 and alpha == 0.4:
        return "historically_conditioned_raw_lane"
    return "conditioned_level_raw_representation_lane"


def conditioning_grid_rows() -> list[dict[str, Any]]:
    rows = []
    for representation in REPRESENTATIONS:
        for setting in enumerate_settings():
            rows.append(
                {
                    "representation": representation,
                    **setting.as_dict(),
                    "raw_lane_semantics": raw_lane_semantics(
                        representation,
                        setting.gaussian_sigma_px,
                        setting.causal_ema_alpha,
                    ),
                    "outer_fold_count": 4,
                    "quiet_role_swaps_per_fold": 2,
                    "protected_coordinate_or_identity_fields_used": False,
                    "burst_window_supervised": True,
                }
            )
    if len(rows) != EXPECTED_GRID_ROWS:
        raise AssertionError("conditioning grid row count changed")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    fields = list(rows[0])
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"invalid TSV schema for {path.name}")
    if any(list(row) != fields for row in rows):
        raise ValueError(f"table {path.name} has inconsistent fields")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ):
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": artifacts}


def _verify_indexed_artifact(root: str | Path, *, role: str) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    index_path = directory / "artifact_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("artifacts"), list
    ):
        raise ConditioningSensitivityUnavailable(f"{role} artifact index is invalid")
    seen: set[str] = set()
    for item in payload["artifacts"]:
        if not isinstance(item, Mapping):
            raise ConditioningSensitivityUnavailable(
                f"{role} artifact index row is invalid"
            )
        relative = str(item.get("path", ""))
        candidate = Path(relative)
        if not relative or relative in seen or candidate.is_absolute():
            raise ConditioningSensitivityUnavailable(
                f"{role} artifact index path is invalid"
            )
        seen.add(relative)
        path = (directory / candidate).resolve()
        if directory not in path.parents:
            raise ConditioningSensitivityUnavailable(
                f"{role} artifact path escapes its root"
            )
        if not path.is_file():
            raise ConditioningSensitivityUnavailable(
                f"{role} indexed artifact is missing: {relative}"
            )
        if path.stat().st_size != int(item.get("size_bytes", -1)) or _sha256(
            path
        ) != str(item.get("sha256", "")):
            raise ConditioningSensitivityUnavailable(
                f"{role} indexed artifact changed after freeze: {relative}"
            )
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "artifact_index.json"
        and not path.name.endswith(".partial")
    }
    if actual != seen:
        raise ConditioningSensitivityUnavailable(
            f"{role} contains files outside its frozen artifact index"
        )
    return {
        "root": str(directory),
        "artifact_index_sha256": _sha256(index_path),
        "verified_artifact_count": len(seen),
    }


_FALSE_LEAKAGE_ATTESTATIONS = {
    "selection_uses_positive_coordinates",
    "selection_uses_positive_identities",
    "selector_uses_positive_coordinates",
    "selector_uses_positive_identities",
    "positive_coordinates_used",
    "positive_identities_used",
}
_FORBIDDEN_CONTEXT_KEY_FRAGMENTS = (
    "x_px",
    "y_px",
    "coordinate",
    "identity",
    "canonical_roi",
    "known_positive",
    "sparse_positive",
    "protected_",
    "label",
    "recall",
    "precision",
    "neuron",
    "annotation_row",
)


def _reject_coordinate_or_identity_fields(payload: Any, *, path: str = "$.") -> None:
    """Reject protected content while allowing explicit false attestations."""

    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            lowered = key.lower()
            child = f"{path}{key}"
            if key in _FALSE_LEAKAGE_ATTESTATIONS:
                if value is not False:
                    raise ConditioningSensitivityUnavailable(
                        f"support leakage attestation must be false: {child}"
                    )
            elif any(fragment in lowered for fragment in _FORBIDDEN_CONTEXT_KEY_FRAGMENTS):
                raise ConditioningSensitivityUnavailable(
                    f"support context contains coordinate/identity field: {child}"
                )
            _reject_coordinate_or_identity_fields(value, path=child + ".")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_coordinate_or_identity_fields(value, path=f"{path}{index}.")


def _reference_from_support_candidate(
    payload: Mapping[str, Any],
) -> GammaReferenceSpec:
    required = {
        "context_id",
        "half_width_px",
        "guard_radius_px",
        "shape",
        "mode_fraction_of_half_width",
        "mode_radius_px",
        "support",
        "padding",
        "eligible_primary",
    }
    if not required.issubset(payload):
        raise ConditioningSensitivityUnavailable(
            "support candidate lacks explicit radial geometry"
        )
    half = int(payload["half_width_px"])
    guard = int(payload["guard_radius_px"])
    shape = float(payload["shape"])
    mode_fraction = float(payload["mode_fraction_of_half_width"])
    mode_radius = float(payload["mode_radius_px"])
    if (
        payload["support"] != "radial_disk"
        or payload["padding"] != "valid_renormalized_zero"
        or payload["eligible_primary"] is not True
        or not 0 <= guard < half
        or not math.isfinite(shape)
        or shape <= 0.0
        or not math.isclose(
            mode_radius,
            half * mode_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ConditioningSensitivityUnavailable(
            "support candidate is not an eligible modern radial Gamma-LS context"
        )
    return GammaReferenceSpec.from_mode(
        str(payload["context_id"]),
        support_width_px=2 * half + 1,
        shape_n=shape,
        mode_radius_px=mode_radius,
        guard_radius_px=guard,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=GAMMA_EPSILON,
        scale_floor=0.0,
    )


def _verify_support_screen(
    config: ConditioningSensitivityConfig,
) -> tuple[dict[int, tuple[GammaReferenceSpec, Mapping[str, Any]]], dict[str, Any]]:
    root = config.support_screen_path
    indexed = _verify_indexed_artifact(root, role="support screen")
    summary_path = root / "summary.json"
    validation_path = root / "validation.json"
    contexts_path = root / "fold_contexts.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_support_screen_only":
        raise ConditioningSensitivityUnavailable("support screen is not complete")
    if validation.get("status") != (
        "passed_support_screen_artifact_contract_scientific_audit_pending"
    ):
        raise ConditioningSensitivityUnavailable(
            "support screen artifact contract did not pass"
        )
    _reject_coordinate_or_identity_fields(contexts)
    if (
        contexts.get("selection_scope") != "outer_training_fold_only"
        or contexts.get("burst_windows_used") is not True
    ):
        raise ConditioningSensitivityUnavailable("support selection scope changed")
    folds = contexts.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ConditioningSensitivityUnavailable(
            "support screen must contain exactly four fold contexts"
        )
    resolved: dict[int, tuple[GammaReferenceSpec, Mapping[str, Any]]] = {}
    geometry_by_context_id: dict[str, tuple[Any, ...]] = {}
    for fold_payload in folds:
        if not isinstance(fold_payload, Mapping):
            raise ConditioningSensitivityUnavailable("support fold row is invalid")
        fold = int(fold_payload.get("training_fold", 0))
        if fold not in OUTER_FOLDS or fold in resolved:
            raise ConditioningSensitivityUnavailable(
                "support screen contains duplicate or invalid training fold"
            )
        if str(fold_payload.get("heldout_burst")) != str(fold):
            raise ConditioningSensitivityUnavailable(
                "support heldout-burst mapping changed"
            )
        candidate = fold_payload.get("support_candidate_context")
        if not isinstance(candidate, Mapping):
            raise ConditioningSensitivityUnavailable(
                "support fold lacks support_candidate_context"
            )
        reference = _reference_from_support_candidate(candidate)
        geometry = (
            int(candidate["half_width_px"]),
            int(candidate["guard_radius_px"]),
            float(candidate["shape"]),
            float(candidate["mode_radius_px"]),
            str(candidate["support"]),
            str(candidate["padding"]),
        )
        prior = geometry_by_context_id.setdefault(reference.context_id, geometry)
        if prior != geometry:
            raise ConditioningSensitivityUnavailable(
                "support context ID has conflicting fold-local geometry"
            )
        resolved[fold] = (reference, candidate)
    if set(resolved) != set(OUTER_FOLDS):
        raise ConditioningSensitivityUnavailable("support fold set changed")
    return resolved, {
        **indexed,
        "summary_sha256": _sha256(summary_path),
        "validation_sha256": _sha256(validation_path),
        "fold_contexts_sha256": _sha256(contexts_path),
        "fold_support_candidate_context_ids": {
            str(fold): str(candidate["context_id"])
            for fold, (_, candidate) in resolved.items()
        },
        "support_context_role": "support_candidate_context",
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
    }


def _verify_base_preflight_without_annotation_reads(
    config: ConditioningSensitivityConfig, base_preflight_dir: str | Path
) -> dict[str, Any]:
    """Verify the base contract while reopening only the movie source."""

    base = config.base_config
    root = Path(base_preflight_dir).expanduser().resolve()
    indexed = _verify_indexed_artifact(root, role="base preflight")
    preflight_path = root / "preflight.json"
    portable_path = root / "config.portable.json"
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    if portable != base.portable_dict():
        raise ConditioningSensitivityUnavailable("base preflight config is stale")
    if not payload.get("data_ready") or not payload.get("gpu_run_ready"):
        raise ConditioningSensitivityUnavailable("base preflight is not GPU-ready")
    current = _implementation_status(base)
    frozen_files = payload.get("implementation", {}).get("files")
    if not current.get("complete") or current.get("files") != frozen_files:
        raise ConditioningSensitivityUnavailable(
            "base implementation fingerprints changed after preflight"
        )
    movie = base.source_paths["movie"]
    frozen_movie_hash = payload.get("source", {}).get("movie", {}).get("sha256")
    if not frozen_movie_hash or _sha256(movie) != str(frozen_movie_hash):
        raise ConditioningSensitivityUnavailable(
            "movie fingerprint changed after base preflight"
        )
    return {
        **indexed,
        "preflight_sha256": _sha256(preflight_path),
        "portable_config_sha256": _sha256(portable_path),
        "movie_sha256": str(frozen_movie_hash),
        "annotation_sources_reopened": False,
    }


def _implementation_hashes(
    config: ConditioningSensitivityConfig,
) -> dict[str, Mapping[str, Any]]:
    relative_paths = (
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/experiments/gamma_ls_difference/config.py",
        "neurobench/experiments/gamma_ls_difference/cuda_runtime.py",
        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
        "neurobench/experiments/gamma_ls_difference/preflight.py",
        "neurobench/experiments/gamma_ls_difference/screen.py",
        "neurobench/experiments/gamma_ls_difference/conditioning_sensitivity.py",
    )
    rows: dict[str, Mapping[str, Any]] = {}
    for relative in relative_paths:
        path = config.repository / relative
        rows[f"repo://{relative}"] = (
            {
                "present": True,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if path.is_file()
            else {"present": False}
        )
    if not all(row.get("present") for row in rows.values()):
        raise ConditioningSensitivityUnavailable(
            "conditioning sensitivity implementation is incomplete"
        )
    return rows


_SELECTOR_REQUIRED_FIELDS = {
    "representation",
    "training_fold",
    "heldout_burst",
    "support_context_id",
    "setting_id",
    "conditioning_id",
    "gaussian_sigma_px",
    "causal_ema_alpha",
    "training_quiet_scale_floor_percentile",
    "mean_event_positive_tail",
    "mean_quiet_positive_tail",
    "mean_positive_tail_contrast",
    "minimum_swap_positive_tail_contrast",
    "online_inference_runtime_ms_per_frame",
}
_SELECTOR_ALLOWED_FIELDS = _SELECTOR_REQUIRED_FIELDS | {
    "training_bursts",
    "heldout_guard_ui",
    "historical_anchor",
    "raw_lane_semantics",
    "conditioning_runtime_ms_per_frame",
    "representation_runtime_ms_per_frame",
    "gamma_runtime_ms_per_frame",
    "mean_score_runtime_ms_per_frame",
    "mean_scale_floor_fit_runtime_ms",
    "mean_tail_metric_runtime_ms",
    "quiet_role_swap_count",
    "positive_coordinates_used",
    "positive_identities_used",
    "burst_windows_used",
}


def _selector_coordinates(row: Mapping[str, Any]) -> tuple[float, float, float]:
    values = (
        float(row["mean_positive_tail_contrast"]),
        float(row["minimum_swap_positive_tail_contrast"]),
        float(row["online_inference_runtime_ms_per_frame"]),
    )
    if not all(math.isfinite(value) for value in values) or values[2] < 0.0:
        raise ValueError("selector objectives must be finite and runtime nonnegative")
    return values


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_mean, left_minimum, left_runtime = _selector_coordinates(left)
    right_mean, right_minimum, right_runtime = _selector_coordinates(right)
    weak = (
        left_mean >= right_mean
        and left_minimum >= right_minimum
        and left_runtime <= right_runtime
    )
    strict = (
        left_mean > right_mean
        or left_minimum > right_minimum
        or left_runtime < right_runtime
    )
    return weak and strict


def pareto_layers(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Assign deterministic Pareto layers for two maxima and one minimum."""

    remaining = {str(row["setting_id"]): row for row in rows}
    if len(remaining) != len(rows):
        raise ValueError("Pareto rows contain duplicate setting IDs")
    layers: dict[str, int] = {}
    layer = 0
    while remaining:
        front = [
            setting_id
            for setting_id, row in remaining.items()
            if not any(
                other_id != setting_id and _dominates(other, row)
                for other_id, other in remaining.items()
            )
        ]
        if not front:
            raise AssertionError("Pareto decomposition produced an empty front")
        for setting_id in sorted(front):
            layers[setting_id] = layer
            del remaining[setting_id]
        layer += 1
    return layers


def select_training_setting(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, int]]:
    """Select one setting from a single representation and outer fold.

    The function has no parameter through which positive coordinates or
    identities can enter.  It also rejects protected-looking fields in metric
    rows before inspecting the three frozen selector objectives.
    """

    if len(rows) != len(enumerate_settings()):
        raise ValueError("selector requires the complete 48-setting grid")
    for row in rows:
        if not _SELECTOR_REQUIRED_FIELDS.issubset(row):
            raise ValueError("selector row is missing frozen objective fields")
        _reject_coordinate_or_identity_fields(row)
        unknown = set(row) - _SELECTOR_ALLOWED_FIELDS
        if unknown:
            raise ValueError(
                "selector row contains noncontract fields: "
                + ",".join(sorted(map(str, unknown)))
            )
    representations = {str(row["representation"]) for row in rows}
    folds = {int(row["training_fold"]) for row in rows}
    if len(representations) != 1 or not representations.issubset(REPRESENTATIONS):
        raise ValueError("selector rows must share one frozen representation")
    if len(folds) != 1 or not folds.issubset(OUTER_FOLDS):
        raise ValueError("selector rows must share one valid outer fold")
    expected_ids = {setting.setting_id for setting in enumerate_settings()}
    actual_ids = {str(row["setting_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise ValueError("selector setting IDs do not match the frozen grid")
    for row in rows:
        expected = ConditioningSetting(
            float(row["gaussian_sigma_px"]),
            float(row["causal_ema_alpha"]),
            float(row["training_quiet_scale_floor_percentile"]),
        )
        if expected.setting_id != str(row["setting_id"]):
            raise ValueError("selector setting parameters and ID disagree")
        _selector_coordinates(row)
    layers = pareto_layers(rows)
    front = [row for row in rows if layers[str(row["setting_id"])] == 0]
    selected = sorted(
        front,
        key=lambda row: (
            -float(row["mean_positive_tail_contrast"]),
            -float(row["minimum_swap_positive_tail_contrast"]),
            float(row["online_inference_runtime_ms_per_frame"]),
            str(row["setting_id"]),
        ),
    )[0]
    return selected, layers


def freeze_fold_selections(
    fold_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze anchor and selected settings for all 12 fold/arm groups."""

    if len(fold_rows) != EXPECTED_FOLD_CELLS:
        raise ValueError("fold metric table must contain exactly 576 cells")
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in fold_rows:
        _reject_coordinate_or_identity_fields(row)
        key = (int(row["training_fold"]), str(row["representation"]))
        groups.setdefault(key, []).append(row)
    expected_groups = {
        (fold, representation)
        for fold in OUTER_FOLDS
        for representation in REPRESENTATIONS
    }
    if set(groups) != expected_groups:
        raise ValueError("fold metric table has an incomplete group set")
    annotated: list[dict[str, Any]] = []
    frozen_rows = []
    for fold, representation in sorted(groups):
        selected, layers = select_training_setting(groups[(fold, representation)])
        anchor_rows = [
            row
            for row in groups[(fold, representation)]
            if str(row["setting_id"])
            == ConditioningSetting(*ANCHOR).setting_id
        ]
        if len(anchor_rows) != 1:
            raise ValueError("historical anchor is missing or duplicated")
        anchor = anchor_rows[0]
        front_ids = sorted(
            setting_id for setting_id, layer in layers.items() if layer == 0
        )

        def freeze_role(role: str, row: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "role": role,
                "setting_id": str(row["setting_id"]),
                "conditioning_id": str(row["conditioning_id"]),
                "gaussian_sigma_px": float(row["gaussian_sigma_px"]),
                "causal_ema_alpha": float(row["causal_ema_alpha"]),
                "training_quiet_scale_floor_percentile": float(
                    row["training_quiet_scale_floor_percentile"]
                ),
                "mean_positive_tail_contrast": float(
                    row["mean_positive_tail_contrast"]
                ),
                "minimum_swap_positive_tail_contrast": float(
                    row["minimum_swap_positive_tail_contrast"]
                ),
                "online_inference_runtime_ms_per_frame": float(
                    row["online_inference_runtime_ms_per_frame"]
                ),
                "pareto_layer": int(layers[str(row["setting_id"])]),
            }

        frozen_rows.append(
            {
                "training_fold": fold,
                "heldout_burst": str(fold),
                "representation": representation,
                "support_context_id": str(selected["support_context_id"]),
                "selection_scope": "outer_training_fold_only",
                "selector_uses_burst_windows": True,
                "selector_uses_positive_coordinates": False,
                "selector_uses_positive_identities": False,
                "pareto_front_setting_ids": front_ids,
                "historical_anchor": freeze_role("historical_anchor", anchor),
                "training_pareto_selected": freeze_role(
                    "training_pareto_selected", selected
                ),
            }
        )
        selected_id = str(selected["setting_id"])
        for row in groups[(fold, representation)]:
            setting_id = str(row["setting_id"])
            annotated.append(
                {
                    **dict(row),
                    "pareto_layer": layers[setting_id],
                    "pareto_nondominated": layers[setting_id] == 0,
                    "frozen_historical_anchor": setting_id
                    == ConditioningSetting(*ANCHOR).setting_id,
                    "frozen_training_pareto_selected": setting_id == selected_id,
                }
            )
    annotated.sort(
        key=lambda row: (
            int(row["training_fold"]),
            str(row["representation"]),
            str(row["setting_id"]),
        )
    )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "selection_scope": "independent_per_representation_within_outer_training_fold",
        "selection_method": (
            "three_objective_nondominated_front_then_frozen_deterministic_tie_break"
        ),
        "objectives": [
            "maximize_mean_event_minus_quiet_tail_contrast",
            "maximize_minimum_quiet_swap_tail_contrast",
            "minimize_online_inference_runtime_ms_per_frame",
        ],
        "downstream_roles": ["historical_anchor", "training_pareto_selected"],
        "downstream_stage": "protected_B_and_NMS_evaluation",
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
        "fold_representation_selections": frozen_rows,
        "selection_hash_scope": (
            "canonical_json_of_payload_excluding_selection_sha256"
        ),
    }
    payload["selection_sha256"] = _canonical_sha256(payload)
    return payload, annotated


def verify_selection_seal(payload: Mapping[str, Any]) -> str:
    """Verify and return the self-excluding canonical selection seal."""

    expected = str(payload.get("selection_sha256", ""))
    if len(expected) != 64:
        raise ValueError("selection seal is missing or invalid")
    unhashed = dict(payload)
    unhashed.pop("selection_sha256", None)
    actual = _canonical_sha256(unhashed)
    if actual != expected:
        raise ValueError("selection payload changed after sealing")
    return actual


def _require_cuda(device: str) -> dict[str, Any]:
    try:
        return require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise ConditioningSensitivityUnavailable(str(error)) from error


def _source_contract(base: GammaLSDifferenceConfig) -> dict[str, Any]:
    movie_path = base.source_paths["movie"]
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap) or movie.ndim != 3:
        raise ConditioningSensitivityUnavailable(
            "source movie must be a memory-mappable TYX .npy"
        )
    shape = tuple(int(value) for value in movie.shape)
    if shape != EXPECTED_MOVIE_SHAPE or str(movie.dtype) != EXPECTED_MOVIE_DTYPE:
        raise ConditioningSensitivityUnavailable(
            "source acquisition shape or dtype changed"
        )
    frames = base.payload["frames"]
    if tuple(map(int, frames["review_interval_ui"])) != REVIEW_INTERVAL_UI:
        raise ConditioningSensitivityUnavailable("base review interval changed")
    return {
        "source_id": base.payload["sources"]["movie"],
        "movie_shape_tyx": list(shape),
        "movie_dtype": str(movie.dtype),
        "causal_history_processed_ui": [1, REVIEW_INTERVAL_UI[1]],
        "retained_conditioned_interval_ui": list(RETAINED_INTERVAL_UI),
        "aligned_representation_interval_ui": list(REVIEW_INTERVAL_UI),
        "load_mode": "numpy_memmap_chunked_full_causal_history",
        "labels_opened": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
    }


def run_sensitivity_preflight(
    config: ConditioningSensitivityConfig,
    *,
    base_preflight_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Freeze all executable inputs in a noncolliding preflight artifact."""

    if not isinstance(config, ConditioningSensitivityConfig):
        raise TypeError("config must be ConditioningSensitivityConfig")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"sensitivity preflight exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"sensitivity preflight parent is missing: {destination.parent}"
        )
    base_preflight = _verify_base_preflight_without_annotation_reads(
        config, base_preflight_dir
    )
    _, support = _verify_support_screen(config)
    source = _source_contract(config.base_config)
    requested = str(device or config.payload["resources"]["device"])
    runtime = _require_cuda(requested)
    resources = config.payload["resources"]
    free_disk = shutil.disk_usage(destination.parent).free
    minimum_disk = int(float(resources["minimum_free_disk_gib"]) * 2**30)
    if free_disk < minimum_disk:
        raise ConditioningSensitivityUnavailable(
            f"free disk {free_disk} is below the frozen minimum {minimum_disk} bytes"
        )
    required_vram = int(float(resources["max_peak_vram_gib"]) * 2**30)
    if int(runtime["free_vram_bytes_before"]) < required_vram:
        raise ConditioningSensitivityUnavailable(
            "free CUDA memory is below the frozen 8-GiB execution cap"
        )
    implementation = _implementation_hashes(config)
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "ready_conditioning_sensitivity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_ready": True,
        "gpu_run_ready": True,
        "base_preflight": base_preflight,
        "support_screen": support,
        "source": source,
        "runtime": runtime,
        "resources": deepcopy(resources),
        "design_counts": {
            "settings_per_representation": 48,
            "representation_settings": EXPECTED_GRID_ROWS,
            "outer_fold_metric_cells": EXPECTED_FOLD_CELLS,
            "quiet_swap_rows": EXPECTED_SWAP_ROWS,
            "fold_representation_selections": 12,
        },
        "implementation": implementation,
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "annotation_sources_reopened": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
        "protected_evaluation_status": "not_started_by_design",
        "scientific_audit_status": (
            "pending_downstream_protected_B_and_NMS_evaluation"
        ),
    }
    partial = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        partial.mkdir()
        _atomic_json(partial / "preflight.json", payload)
        _atomic_json(partial / "config.portable.json", config.portable_dict())
        _atomic_tsv(partial / "planned_conditioning_grid.tsv", conditioning_grid_rows())
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "preflight.json",
                "grain": "representation by conditioning/floor setting",
                "selection_scope": config.payload["design"]["selection_scope"],
                "protected_fields": "not opened",
                "next_stage": "protected_B_and_NMS_evaluation",
            },
        )
        _atomic_json(
            partial / "validation.json",
            {"status": "passed_conditioning_sensitivity_preflight"},
        )
        _atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        if destination.exists():
            raise FileExistsError(
                "sensitivity preflight destination appeared during commit"
            )
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return payload


def _verify_sensitivity_preflight(
    config: ConditioningSensitivityConfig,
    sensitivity_preflight_dir: str | Path,
) -> dict[str, Any]:
    root = Path(sensitivity_preflight_dir).expanduser().resolve()
    indexed = _verify_indexed_artifact(root, role="sensitivity preflight")
    preflight_path = root / "preflight.json"
    portable_path = root / "config.portable.json"
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    if portable != config.portable_dict():
        raise ConditioningSensitivityUnavailable(
            "sensitivity preflight manifest is stale"
        )
    if (
        payload.get("status") != "ready_conditioning_sensitivity"
        or payload.get("gpu_run_ready") is not True
    ):
        raise ConditioningSensitivityUnavailable(
            "sensitivity preflight is not GPU-run ready"
        )
    if payload.get("implementation") != _implementation_hashes(config):
        raise ConditioningSensitivityUnavailable(
            "sensitivity implementation changed after preflight"
        )
    base_root = payload.get("base_preflight", {}).get("root")
    if not isinstance(base_root, str):
        raise ConditioningSensitivityUnavailable(
            "sensitivity preflight lacks its base-preflight root"
        )
    current_base = _verify_base_preflight_without_annotation_reads(config, base_root)
    if current_base["artifact_index_sha256"] != payload.get(
        "base_preflight", {}
    ).get("artifact_index_sha256"):
        raise ConditioningSensitivityUnavailable(
            "base preflight changed after sensitivity preflight"
        )
    contexts, current_support = _verify_support_screen(config)
    frozen_support = payload.get("support_screen", {})
    if (
        current_support["artifact_index_sha256"]
        != frozen_support.get("artifact_index_sha256")
        or current_support["fold_contexts_sha256"]
        != frozen_support.get("fold_contexts_sha256")
    ):
        raise ConditioningSensitivityUnavailable(
            "support screen changed after sensitivity preflight"
        )
    frozen_runtime = payload.get("runtime", {})
    runtime = _require_cuda(str(frozen_runtime.get("requested_device", "")))
    stable_runtime_fields = (
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
    if any(runtime.get(key) != frozen_runtime.get(key) for key in stable_runtime_fields):
        raise ConditioningSensitivityUnavailable(
            "CUDA runtime identity changed after sensitivity preflight"
        )
    required_vram = int(
        float(config.payload["resources"]["max_peak_vram_gib"]) * 2**30
    )
    if int(runtime["free_vram_bytes_before"]) < required_vram:
        raise ConditioningSensitivityUnavailable(
            "free CUDA memory fell below the frozen 8-GiB execution cap"
        )
    return {
        **indexed,
        "preflight_sha256": _sha256(preflight_path),
        "portable_config_sha256": _sha256(portable_path),
        "base_preflight_artifact_index_sha256": current_base[
            "artifact_index_sha256"
        ],
        "support_artifact_index_sha256": current_support[
            "artifact_index_sha256"
        ],
        "support_fold_contexts_sha256": current_support["fold_contexts_sha256"],
        "runtime": runtime,
        "fold_contexts": contexts,
        "annotation_sources_reopened": False,
    }


def _scipy_reflect_indices(length: int, radius: int, *, device: Any) -> Any:
    import torch

    coordinates = torch.arange(-radius, length + radius, device=device)
    period = 2 * length
    wrapped = torch.remainder(coordinates, period)
    return torch.where(wrapped < length, wrapped, period - 1 - wrapped).to(
        torch.int64
    )


def spatial_gaussian_reflect(values: Any, sigma_px: float) -> Any:
    """Apply the frozen SciPy-compatible Gaussian family to a TYX tensor."""

    import torch
    import torch.nn.functional as functional

    if not torch.is_tensor(values) or values.ndim != 3 or values.dtype != torch.float32:
        raise ValueError("Gaussian input must be a float32 TYX torch tensor")
    sigma = float(sigma_px)
    if sigma not in GAUSSIAN_SIGMAS:
        raise ValueError("sigma is outside the frozen conditioning grid")
    if sigma == 0.0:
        return values
    radius = int(GAUSSIAN_TRUNCATE * sigma + 0.5)
    coordinates = torch.arange(
        -radius, radius + 1, dtype=torch.float32, device=values.device
    )
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    frames = values.unsqueeze(1)
    y_indices = _scipy_reflect_indices(
        int(values.shape[-2]), radius, device=values.device
    )
    spatial = functional.conv2d(
        torch.index_select(frames, -2, y_indices),
        kernel.reshape(1, 1, -1, 1),
    )
    x_indices = _scipy_reflect_indices(
        int(values.shape[-1]), radius, device=values.device
    )
    return functional.conv2d(
        torch.index_select(spatial, -1, x_indices),
        kernel.reshape(1, 1, 1, -1),
    ).squeeze(1)


def causal_condition_dense(values: Any, *, sigma_px: float, ema_alpha: float) -> Any:
    """Reference dense conditioning used by focused parity tests."""

    import torch

    if not torch.is_tensor(values) or values.ndim != 3 or values.dtype != torch.float32:
        raise ValueError("conditioning input must be a float32 TYX torch tensor")
    alpha = float(ema_alpha)
    if alpha not in EMA_ALPHAS:
        raise ValueError("EMA alpha is outside the frozen conditioning grid")
    spatial = spatial_gaussian_reflect(values, sigma_px)
    if alpha == 1.0:
        return spatial.clone()
    filtered = [spatial[0]]
    for frame in range(1, int(spatial.shape[0])):
        filtered.append(alpha * spatial[frame] + (1.0 - alpha) * filtered[-1])
    return torch.stack(filtered)


def _stream_sigma_all_ema_history_to_device(
    movie_path: Path,
    *,
    sigma_px: float,
    ema_alphas: Sequence[float] = EMA_ALPHAS,
    chunk_frames: int,
    device: Any,
    heartbeat: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[dict[float, Any], dict[float, dict[str, Any]]]:
    """Reuse each transfer/Gaussian pass across requested causal EMA arms.

    The physical grid run performs four source passes, one per sigma. Runtime
    assigned to each candidate nevertheless includes the *full* transfer and
    spatial-filter cost, because a deployed single candidate cannot divide
    those costs by the number of alpha arms screened here.
    """

    import torch

    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if tuple(movie.shape) != EXPECTED_MOVIE_SHAPE or str(movie.dtype) != (
        EXPECTED_MOVIE_DTYPE
    ):
        raise ConditioningSensitivityUnavailable("source contract changed during run")
    sigma = float(sigma_px)
    if sigma not in GAUSSIAN_SIGMAS or int(chunk_frames) < 1:
        raise ValueError("sigma/chunk request is outside the frozen grid")
    alphas = tuple(float(value) for value in ema_alphas)
    if (
        not alphas
        or len(alphas) != len(set(alphas))
        or any(alpha not in EMA_ALPHAS for alpha in alphas)
    ):
        raise ValueError("EMA subset must contain unique members of the frozen grid")
    retain_start_zero = RETAINED_INTERVAL_UI[0] - 1
    stop_zero = REVIEW_INTERVAL_UI[1]
    retained: dict[float, list[Any]] = {alpha: [] for alpha in alphas}
    states: dict[float, Any | None] = {alpha: None for alpha in alphas}
    h2d_ms = 0.0
    spatial_ms = 0.0
    ema_ms = {alpha: 0.0 for alpha in alphas}
    transfer_count = 0
    total_chunks = int(math.ceil(stop_zero / int(chunk_frames)))
    for chunk_index, start in enumerate(
        range(0, stop_zero, int(chunk_frames)), start=1
    ):
        stop = min(start + int(chunk_frames), stop_zero)
        host_array = np.array(
            movie[start:stop], dtype=np.float32, order="C", copy=True
        )
        host_tensor = torch.from_numpy(host_array)
        device_chunk, transfer_ms = _elapsed(
            device,
            lambda source=host_tensor: source.to(
                device=device, dtype=torch.float32, non_blocking=False
            ),
        )
        h2d_ms += transfer_ms
        transfer_count += 1
        spatial, chunk_spatial_ms = _elapsed(
            device, lambda: spatial_gaussian_reflect(device_chunk, sigma)
        )
        spatial_ms += chunk_spatial_ms
        for alpha in alphas:

            def ema_chunk(alpha_value: float = alpha) -> tuple[Any, list[Any]]:
                state = states[alpha_value]
                kept = []
                for offset in range(int(spatial.shape[0])):
                    absolute = start + offset
                    if state is None or alpha_value == 1.0:
                        state = spatial[offset]
                    else:
                        state = (
                            alpha_value * spatial[offset]
                            + (1.0 - alpha_value) * state
                        )
                    if absolute >= retain_start_zero:
                        kept.append(state.clone())
                return state, kept

            (state, kept), elapsed_ms = _elapsed(device, ema_chunk)
            states[alpha] = state
            retained[alpha].extend(kept)
            ema_ms[alpha] += elapsed_ms
        del spatial, device_chunk, host_tensor, host_array
        if heartbeat is not None:
            heartbeat(
                {
                    "stage": "shared_sigma_conditioning",
                    "gaussian_sigma_px": sigma,
                    "completed_history_chunks": chunk_index,
                    "total_history_chunks": total_chunks,
                    "ema_arms_updated": len(alphas),
                }
            )
    expected = RETAINED_INTERVAL_UI[1] - RETAINED_INTERVAL_UI[0] + 1
    outputs: dict[float, Any] = {}
    timings: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        result = torch.stack(retained[alpha])
        if int(result.shape[0]) != expected or result.device != device:
            raise AssertionError("conditioned history alignment changed")
        if not bool(torch.isfinite(result).all()):
            raise ConditioningSensitivityUnavailable("conditioned history is non-finite")
        outputs[alpha] = result
        candidate_total_ms = h2d_ms + spatial_ms + ema_ms[alpha]
        timings[alpha] = {
            "gaussian_sigma_px": sigma,
            "causal_ema_alpha": alpha,
            "shared_grid_h2d_ms": h2d_ms,
            "shared_grid_spatial_ms": spatial_ms,
            "candidate_ema_ms": ema_ms[alpha],
            "candidate_online_conditioning_including_full_shared_cost_ms": (
                candidate_total_ms
            ),
            "online_conditioning_including_h2d_ms_per_frame": (
                candidate_total_ms / stop_zero
            ),
            "causal_history_frame_count": stop_zero,
            "source_h2d_transfer_count": transfer_count,
            "source_chunk_frames": int(chunk_frames),
            "ema_arms_in_shared_pass": list(alphas),
            "grid_execution_reuses_h2d_and_spatial_across_alphas": True,
            "candidate_runtime_charges_full_h2d_and_spatial_cost": True,
            "ema_state_carried_across_chunks": True,
            "first_frame_initialization": (
                "first_spatially_conditioned_acquisition_frame"
            ),
            "acquisition_raw_exact_after_float32_cast": sigma == 0.0
            and alpha == 1.0,
        }
    return outputs, timings


def _form_representation(common: Any, name: str) -> Any:
    import torch

    if name == "raw":
        return common[1:].clone()
    previous = common[:-1]
    current = common[1:]
    if name == "difference_signed":
        return current - previous
    if name == "difference_energy_normalized":
        return (current - previous) / torch.sqrt(
            previous.square() + current.square() + ENERGY_EPSILON
        )
    raise ValueError(f"unsupported conditioning representation: {name!r}")


def _evaluate_setting_for_fold(
    representation: Any,
    *,
    representation_name: str,
    setting: ConditioningSetting,
    fold: Any,
    reference: GammaReferenceSpec,
    frame_ui: Any,
    bursts: Mapping[str, Sequence[int]],
    moments: Any,
    conditioning_ms_per_frame: float,
    representation_ms_per_frame: float,
    gamma_ms_per_frame: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import torch

    if moments.local_mean is None or moments.local_std is None:
        raise AssertionError("Gamma-LS moments were not returned")
    mean = moments.local_mean
    std = moments.local_std
    swap_rows = []
    for swap_name, floor_interval, tail_interval in (
        (QUIET_SWAPS[0], fold.quiet_half_a_ui, fold.quiet_half_b_ui),
        (QUIET_SWAPS[1], fold.quiet_half_b_ui, fold.quiet_half_a_ui),
    ):
        guard_mask = _interval_mask(frame_ui, fold.heldout_guard_ui)
        floor_mask = _interval_mask(frame_ui, floor_interval) & ~guard_mask
        floor_count = int(torch.count_nonzero(floor_mask).item())
        if floor_count < 1:
            raise ValueError("quiet floor interval has no guard-safe frames")
        floor, floor_fit_ms = _elapsed(
            representation.device,
            lambda: _positive_scale_floor(
                std, floor_mask, setting.scale_floor_percentile
            ),
        )
        score, score_ms = _elapsed(
            representation.device,
            lambda: (representation - mean)
            / (torch.maximum(std, floor) + GAMMA_EPSILON),
        )
        (event_tail, quiet_tail, counts), tail_ms = _elapsed(
            representation.device,
            lambda: _tail_metrics(
                score,
                frame_ui=frame_ui,
                fold=fold,
                bursts=bursts,
                quiet_tail_interval=tail_interval,
                tail_quantile=0.999,
            ),
        )
        score_ms_per_frame = score_ms / int(representation.shape[0])
        online_runtime = (
            conditioning_ms_per_frame
            + representation_ms_per_frame
            + gamma_ms_per_frame
            + score_ms_per_frame
        )
        swap_rows.append(
            {
                "representation": representation_name,
                "training_fold": int(fold.training_fold),
                "heldout_burst": str(fold.heldout_burst),
                "training_bursts": json.dumps(list(fold.training_bursts)),
                "heldout_guard_ui": json.dumps(list(fold.heldout_guard_ui)),
                "support_context_id": reference.context_id,
                **setting.as_dict(),
                "raw_lane_semantics": raw_lane_semantics(
                    representation_name,
                    setting.gaussian_sigma_px,
                    setting.causal_ema_alpha,
                ),
                "quiet_swap": swap_name,
                "floor_interval_ui": json.dumps(list(floor_interval)),
                "tail_interval_ui": json.dumps(list(tail_interval)),
                "floor_eligible_aligned_frames": floor_count,
                "scale_floor": float(floor.item()),
                "positive_tail_quantile": 0.999,
                "positive_tail_method": TAIL_QUANTILE_METHOD,
                "event_positive_tail": float(event_tail.item()),
                "quiet_positive_tail": float(quiet_tail.item()),
                "positive_tail_contrast": float((event_tail - quiet_tail).item()),
                "conditioning_runtime_ms_per_frame": conditioning_ms_per_frame,
                "representation_runtime_ms_per_frame": representation_ms_per_frame,
                "gamma_runtime_ms_per_frame": gamma_ms_per_frame,
                "score_runtime_ms_per_frame": score_ms_per_frame,
                "online_inference_runtime_ms_per_frame": online_runtime,
                "scale_floor_fit_runtime_ms": floor_fit_ms,
                "tail_metric_runtime_ms": tail_ms,
                "training_event_frame_counts": json.dumps(
                    counts["training_event_frame_counts"], sort_keys=True
                ),
                "quiet_tail_frames": counts["quiet_tail_frames"],
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
            }
        )
        del score
    mean_event = sum(float(row["event_positive_tail"]) for row in swap_rows) / 2.0
    mean_quiet = sum(float(row["quiet_positive_tail"]) for row in swap_rows) / 2.0
    mean_contrast = sum(float(row["positive_tail_contrast"]) for row in swap_rows) / 2.0
    minimum_contrast = min(
        float(row["positive_tail_contrast"]) for row in swap_rows
    )
    mean_score_ms = sum(
        float(row["score_runtime_ms_per_frame"]) for row in swap_rows
    ) / 2.0
    online_runtime = (
        conditioning_ms_per_frame
        + representation_ms_per_frame
        + gamma_ms_per_frame
        + mean_score_ms
    )
    aggregate = {
        "representation": representation_name,
        "training_fold": int(fold.training_fold),
        "heldout_burst": str(fold.heldout_burst),
        "training_bursts": json.dumps(list(fold.training_bursts)),
        "heldout_guard_ui": json.dumps(list(fold.heldout_guard_ui)),
        "support_context_id": reference.context_id,
        **setting.as_dict(),
        "raw_lane_semantics": raw_lane_semantics(
            representation_name,
            setting.gaussian_sigma_px,
            setting.causal_ema_alpha,
        ),
        "mean_event_positive_tail": mean_event,
        "mean_quiet_positive_tail": mean_quiet,
        "mean_positive_tail_contrast": mean_contrast,
        "minimum_swap_positive_tail_contrast": minimum_contrast,
        "conditioning_runtime_ms_per_frame": conditioning_ms_per_frame,
        "representation_runtime_ms_per_frame": representation_ms_per_frame,
        "gamma_runtime_ms_per_frame": gamma_ms_per_frame,
        "mean_score_runtime_ms_per_frame": mean_score_ms,
        "online_inference_runtime_ms_per_frame": online_runtime,
        "mean_scale_floor_fit_runtime_ms": sum(
            float(row["scale_floor_fit_runtime_ms"]) for row in swap_rows
        )
        / 2.0,
        "mean_tail_metric_runtime_ms": sum(
            float(row["tail_metric_runtime_ms"]) for row in swap_rows
        )
        / 2.0,
        "quiet_role_swap_count": 2,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
    }
    timing = {
        "representation": representation_name,
        "training_fold": int(fold.training_fold),
        "support_context_id": reference.context_id,
        "setting_id": setting.setting_id,
        "conditioning_id": setting.conditioning_id,
        "gaussian_sigma_px": setting.gaussian_sigma_px,
        "causal_ema_alpha": setting.causal_ema_alpha,
        "training_quiet_scale_floor_percentile": setting.scale_floor_percentile,
        "conditioning_including_h2d_ms_per_frame": conditioning_ms_per_frame,
        "representation_ms_per_frame": representation_ms_per_frame,
        "gamma_ms_per_frame": gamma_ms_per_frame,
        "mean_score_ms_per_frame": mean_score_ms,
        "online_inference_runtime_ms_per_frame": online_runtime,
        "mean_calibration_floor_fit_ms": aggregate[
            "mean_scale_floor_fit_runtime_ms"
        ],
        "mean_evaluation_tail_metric_ms": aggregate["mean_tail_metric_runtime_ms"],
        "online_runtime_excludes_floor_fit_and_tail_metric": True,
        "cuda_event_synchronized": True,
    }
    return aggregate, swap_rows, timing


def _execute_device_sensitivity(
    movie_path: Path,
    *,
    base: GammaLSDifferenceConfig,
    fold_contexts: Mapping[
        int, tuple[GammaReferenceSpec, Mapping[str, Any]]
    ],
    runtime: Mapping[str, Any],
    source_chunk_frames: int,
    gamma_chunk_frames: int,
    heartbeat: Callable[[Mapping[str, Any]], None] | None,
) -> SensitivityExecution:
    """Execute the frozen design; all dense maps remain on CUDA."""

    import torch

    device = torch.device(str(runtime["resolved_device"]))
    if device.type != "cuda":
        raise ConditioningSensitivityUnavailable("sensitivity execution is CUDA-only")
    folds = build_fold_contracts(base)
    if tuple(int(fold.training_fold) for fold in folds) != OUTER_FOLDS:
        raise ConditioningSensitivityUnavailable("outer fold construction changed")
    bursts = {
        str(key): tuple(map(int, value))
        for key, value in base.payload["frames"]["burst_intervals_ui"].items()
    }
    frame_ui = torch.arange(
        REVIEW_INTERVAL_UI[0],
        REVIEW_INTERVAL_UI[1] + 1,
        dtype=torch.int64,
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    fold_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    conditioning_timings = []
    representation_timings = []
    gamma_map_count = 0
    for sigma in GAUSSIAN_SIGMAS:
        common_by_alpha, timing_by_alpha = _stream_sigma_all_ema_history_to_device(
            movie_path,
            sigma_px=sigma,
            chunk_frames=source_chunk_frames,
            device=device,
            heartbeat=heartbeat,
        )
        for alpha in EMA_ALPHAS:
            common = common_by_alpha.pop(alpha)
            conditioning_timing = timing_by_alpha[alpha]
            conditioning_id = ConditioningSetting(sigma, alpha, 10.0).conditioning_id
            conditioning_timings.append(
                {"conditioning_id": conditioning_id, **conditioning_timing}
            )
            conditioning_ms_per_frame = float(
                conditioning_timing[
                    "online_conditioning_including_h2d_ms_per_frame"
                ]
            )
            for representation_name in REPRESENTATIONS:
                representation, representation_ms = _elapsed(
                    device,
                    lambda name=representation_name: _form_representation(common, name),
                )
                if tuple(int(value) for value in representation.shape) != (
                    REVIEW_INTERVAL_UI[1] - REVIEW_INTERVAL_UI[0] + 1,
                    EXPECTED_MOVIE_SHAPE[1],
                    EXPECTED_MOVIE_SHAPE[2],
                ):
                    raise AssertionError("representation alignment changed")
                representation_ms_per_frame = representation_ms / int(
                    representation.shape[0]
                )
                representation_timings.append(
                    {
                        "conditioning_id": conditioning_id,
                        "representation": representation_name,
                        "raw_lane_semantics": raw_lane_semantics(
                            representation_name, sigma, alpha
                        ),
                        "representation_ms": representation_ms,
                        "representation_ms_per_frame": representation_ms_per_frame,
                    }
                )
                for fold in folds:
                    reference, _ = fold_contexts[int(fold.training_fold)]
                    moments, gamma_ms = _elapsed(
                        device,
                        lambda spec=reference: gamma_local_standardization(
                            representation,
                            spec,
                            chunk_frames=gamma_chunk_frames,
                            return_statistics=True,
                        ),
                    )
                    if (
                        moments.local_mean is None
                        or moments.local_std is None
                        or moments.values.device != device
                    ):
                        raise ConditioningSensitivityUnavailable(
                            "Gamma-LS did not return CUDA-resident local moments"
                        )
                    gamma_ms_per_frame = gamma_ms / int(representation.shape[0])
                    gamma_map_count += 1
                    for percentile in SCALE_FLOOR_PERCENTILES:
                        setting = ConditioningSetting(sigma, alpha, percentile)
                        aggregate, swaps, timing = _evaluate_setting_for_fold(
                            representation,
                            representation_name=representation_name,
                            setting=setting,
                            fold=fold,
                            reference=reference,
                            frame_ui=frame_ui,
                            bursts=bursts,
                            moments=moments,
                            conditioning_ms_per_frame=conditioning_ms_per_frame,
                            representation_ms_per_frame=representation_ms_per_frame,
                            gamma_ms_per_frame=gamma_ms_per_frame,
                        )
                        fold_rows.append(aggregate)
                        swap_rows.extend(swaps)
                        timing_rows.append(timing)
                    if heartbeat is not None:
                        heartbeat(
                            {
                                "stage": "fold_context_complete",
                                "conditioning_id": conditioning_id,
                                "representation": representation_name,
                                "training_fold": int(fold.training_fold),
                                "support_context_id": reference.context_id,
                                "completed_gamma_maps": gamma_map_count,
                                "total_gamma_maps": 16 * 3 * 4,
                            }
                        )
                    del moments
                del representation
            del common
    elapsed_seconds = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    return SensitivityExecution(
        fold_rows=tuple(fold_rows),
        swap_rows=tuple(swap_rows),
        timing_rows=tuple(timing_rows),
        execution_summary={
            "wall_time_seconds": elapsed_seconds,
            "resolved_device": str(device),
            "dense_scoring_device": "cuda",
            "source_history_passes": 4,
            "spatial_gaussian_history_passes": 4,
            "grid_execution_reuses_transfer_and_spatial_filter_across_ema_arms": True,
            "candidate_runtime_charges_full_transfer_and_spatial_filter_cost": True,
            "conditioning_combinations": 16,
            "representation_maps": 48,
            "fold_context_gamma_maps": gamma_map_count,
            "fold_metric_cells": len(fold_rows),
            "quiet_swap_rows": len(swap_rows),
            "peak_memory_allocated_bytes": peak,
            "conditioning_timings": conditioning_timings,
            "representation_timings": representation_timings,
            "cuda_timing": "event_stop_synchronized",
            "scale_floor_fit_charged_to_online_runtime": False,
            "tail_metric_charged_to_online_runtime": False,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "burst_windows_used": True,
        },
    )


def _validate_execution(execution: SensitivityExecution) -> dict[str, Any]:
    if len(execution.fold_rows) != EXPECTED_FOLD_CELLS:
        raise ValueError("executor did not return 576 fold cells")
    if len(execution.swap_rows) != EXPECTED_SWAP_ROWS:
        raise ValueError("executor did not return 1,152 quiet-swap rows")
    if len(execution.timing_rows) != EXPECTED_FOLD_CELLS:
        raise ValueError("executor did not return one timing row per fold cell")
    expected_setting_ids = {setting.setting_id for setting in enumerate_settings()}
    aggregate_groups: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in execution.fold_rows:
        _reject_coordinate_or_identity_fields(row)
        if not _SELECTOR_REQUIRED_FIELDS.issubset(row):
            raise ValueError("executor fold row lacks selector fields")
        key = (
            int(row["training_fold"]),
            str(row["representation"]),
            str(row["setting_id"]),
        )
        if key in aggregate_groups:
            raise ValueError("executor returned a duplicate fold cell")
        if key[0] not in OUTER_FOLDS or key[1] not in REPRESENTATIONS:
            raise ValueError("executor returned an invalid fold or representation")
        expected_semantics = raw_lane_semantics(
            key[1],
            float(row["gaussian_sigma_px"]),
            float(row["causal_ema_alpha"]),
        )
        if row.get("raw_lane_semantics") != expected_semantics:
            raise ValueError("raw-lane semantics are missing or incorrect")
        if str(row.get("heldout_burst")) != str(key[0]):
            raise ValueError("fold and heldout burst disagree")
        aggregate_groups[key] = row
    for fold in OUTER_FOLDS:
        for representation in REPRESENTATIONS:
            found = {
                setting_id
                for (row_fold, row_representation, setting_id) in aggregate_groups
                if row_fold == fold and row_representation == representation
            }
            if found != expected_setting_ids:
                raise ValueError("executor fold group is not the frozen 48-setting grid")
    swap_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for row in execution.swap_rows:
        _reject_coordinate_or_identity_fields(row)
        key = (
            int(row["training_fold"]),
            str(row["representation"]),
            str(row["setting_id"]),
        )
        if key not in aggregate_groups:
            raise ValueError("quiet-swap row has no aggregate cell")
        swap_groups.setdefault(key, []).append(row)
    if set(swap_groups) != set(aggregate_groups):
        raise ValueError("quiet-swap and aggregate group sets differ")
    for key, swaps in swap_groups.items():
        if len(swaps) != 2 or {str(row["quiet_swap"]) for row in swaps} != set(
            QUIET_SWAPS
        ):
            raise ValueError("every fold cell requires the two frozen quiet swaps")
        aggregate = aggregate_groups[key]
        contrast = sum(float(row["positive_tail_contrast"]) for row in swaps) / 2.0
        minimum = min(float(row["positive_tail_contrast"]) for row in swaps)
        if not math.isclose(
            contrast,
            float(aggregate["mean_positive_tail_contrast"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            minimum,
            float(aggregate["minimum_swap_positive_tail_contrast"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("quiet-swap metrics do not reconcile to aggregate")
    return {
        "fold_cell_count_exact": True,
        "quiet_swap_row_count_exact": True,
        "timing_row_count_exact": True,
        "quiet_swaps_reconcile": True,
        "fold_setting_grid_complete": True,
        "raw_lane_semantics_explicit": True,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
    }


def _selection_report(selection: Mapping[str, Any]) -> str:
    lines = [
        "# Gamma-LS conditioning sensitivity v1",
        "",
        "## Outcome",
        "",
        "The burst-window-supervised training screen completed. The historical "
        "anchor and one Pareto-selected setting are sealed independently for each "
        "outer-fold/representation pair. Protected B/NMS evaluation remains pending.",
        "",
        "## Frozen settings",
        "",
        "| Fold | Representation | Context | Anchor | Selected | Same |",
        "|---:|---|---|---|---|:---:|",
    ]
    for row in selection["fold_representation_selections"]:
        anchor = row["historical_anchor"]["setting_id"]
        selected = row["training_pareto_selected"]["setting_id"]
        lines.append(
            f"| {row['training_fold']} | {row['representation']} | "
            f"{row['support_context_id']} | {anchor} | {selected} | "
            f"{'yes' if anchor == selected else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This screen uses declared burst time windows, but no sparse-positive "
            "coordinate or identity table. It measures training-window tail "
            "contrast and CUDA runtime only. It does not establish protected recall, "
            "proposal precision, neuron identity, or biological discovery.",
            "",
            "The `raw` representation at sigma 0 / alpha 1 is acquisition raw "
            "float32. The historical `raw` lane at sigma 1 / alpha 0.4 is conditioned.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_conditioning_sensitivity(
    config: ConditioningSensitivityConfig,
    *,
    sensitivity_preflight_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute and atomically commit the coordinate-free nested screen."""

    if not isinstance(config, ConditioningSensitivityConfig):
        raise TypeError("config must be ConditioningSensitivityConfig")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"conditioning sensitivity output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"conditioning sensitivity output parent is missing: {destination.parent}"
        )
    # Every mutable-output operation occurs after these stale-input gates.
    preflight = _verify_sensitivity_preflight(config, sensitivity_preflight_dir)
    resources = config.payload["resources"]
    free_disk = shutil.disk_usage(destination.parent).free
    minimum_disk = int(float(resources["minimum_free_disk_gib"]) * 2**30)
    if free_disk < minimum_disk:
        raise ConditioningSensitivityUnavailable(
            f"free disk {free_disk} is below the frozen minimum {minimum_disk} bytes"
        )
    partial = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        partial.mkdir()

        def heartbeat(payload: Mapping[str, Any]) -> None:
            _atomic_json(
                partial / "heartbeat.json",
                {
                    "experiment_id": EXPERIMENT_ID,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    **dict(payload),
                },
            )

        execution = _execute_device_sensitivity(
            config.base_config.source_paths["movie"],
            base=config.base_config,
            fold_contexts=preflight["fold_contexts"],
            runtime=preflight["runtime"],
            source_chunk_frames=int(resources["source_chunk_frames"]),
            gamma_chunk_frames=int(resources["gamma_chunk_frames"]),
            heartbeat=heartbeat,
        )
        checks = _validate_execution(execution)
        selection, annotated_fold_rows = freeze_fold_selections(execution.fold_rows)
        verify_selection_seal(selection)
        # This is the seal. There is no label-join path in this executor.
        _atomic_json(partial / "fold_selected_settings.json", selection)
        _atomic_tsv(partial / "conditioning_grid.tsv", conditioning_grid_rows())
        _atomic_tsv(partial / "fold_metric_cells.tsv", annotated_fold_rows)
        _atomic_tsv(partial / "quiet_swap_rows.tsv", execution.swap_rows)
        _atomic_tsv(partial / "runtime_components.tsv", execution.timing_rows)
        _atomic_json(
            partial / "execution_summary.json", dict(execution.execution_summary)
        )
        run_contract = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "sensitivity_preflight_sha256": preflight["preflight_sha256"],
            "base_preflight_artifact_index_sha256": preflight[
                "base_preflight_artifact_index_sha256"
            ],
            "support_artifact_index_sha256": preflight[
                "support_artifact_index_sha256"
            ],
            "support_fold_contexts_sha256": preflight[
                "support_fold_contexts_sha256"
            ],
            "selection_sha256": selection["selection_sha256"],
            "selection_sealed_before_any_label_join": True,
            "label_join_api_present": False,
            "dense_map_device": "cuda",
            "outer_folds": 4,
            "fold_context_role": "support_candidate_context",
            "protected_evaluation_in_this_run": False,
            "annotation_sources_reopened": False,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "burst_windows_used": True,
        }
        _atomic_json(partial / "run_contract.json", run_contract)
        claim_boundary = {
            "training_window_tail_contrast_computed": True,
            "online_cuda_runtime_computed": True,
            "conditioning_sensitivity_screen_complete": True,
            "burst_windows_used": True,
            "fully_label_free_claimed": False,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "protected_B_and_NMS_evaluation_computed": False,
            "protected_recall_claimed": False,
            "proposal_precision_claimed": False,
            "neuron_identity_claimed": False,
            "scientific_audit_complete": False,
            "interpretation": (
                "Anchor and fold-local Pareto settings are training-screen "
                "candidates only until the separately frozen protected B/NMS "
                "evaluation and scientific audit complete."
            ),
        }
        _atomic_json(partial / "claim_boundary.json", claim_boundary)
        audit = {
            "schema_version": 1,
            "status": "pending_downstream_protected_B_and_NMS_evaluation",
            "candidate_surrogate_montage": "not_applicable_to_upstream_parameter_screen",
            "expert_review_ledger": "pending_after_downstream_candidate_generation",
            "failure_taxonomy": "pending_after_downstream_candidate_generation",
            "rationale": (
                "This artifact selects conditioning settings from aggregate "
                "training-window tails and produces no proposal coordinates."
            ),
            "positive_coordinates_used": False,
            "positive_identities_used": False,
        }
        _atomic_json(partial / "scientific_audit_status.json", audit)
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "complete_conditioning_sensitivity_screen_only",
            "completed_at_utc": completed_at,
            "design_counts": {
                "settings_per_representation": 48,
                "representation_settings": EXPECTED_GRID_ROWS,
                "fold_metric_cells": len(execution.fold_rows),
                "quiet_swap_rows": len(execution.swap_rows),
                "fold_representation_selections": len(
                    selection["fold_representation_selections"]
                ),
            },
            "selection": "fold_selected_settings.json",
            "selection_sha256": selection["selection_sha256"],
            "execution": dict(execution.execution_summary),
            "claim_boundary": claim_boundary,
        }
        _atomic_json(partial / "summary.json", summary)
        _atomic_json(
            partial / "validation.json",
            {
                "status": (
                    "passed_conditioning_sensitivity_screen_artifact_contract_"
                    "scientific_audit_pending"
                ),
                "checks": checks,
            },
        )
        _atomic_json(
            partial / "provenance_hashes.json",
            {
                "sensitivity_manifest_sha256": _sha256(config.manifest_path),
                "sensitivity_preflight_sha256": preflight["preflight_sha256"],
                "support_fold_contexts_sha256": preflight[
                    "support_fold_contexts_sha256"
                ],
                "selection_sha256": selection["selection_sha256"],
                "implementation": _implementation_hashes(config),
            },
        )
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "primary_selection": "fold_selected_settings.json",
                "metric_grain": "representation by setting by outer training fold",
                "raw_lane_warning": (
                    "raw is acquisition raw only for sigma0_alpha1; the historical "
                    "raw lane is sigma1_alpha0p4 conditioned level data"
                ),
                "protected_evaluation": "pending",
                "positive_coordinates_or_identities": "not opened",
            },
        )
        _atomic_text(partial / "REPORT.md", _selection_report(selection))
        heartbeat(
            {
                "stage": "complete",
                "fold_metric_cells": len(execution.fold_rows),
                "quiet_swap_rows": len(execution.swap_rows),
                "selection_sha256": selection["selection_sha256"],
            }
        )
        _atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        if destination.exists():
            raise FileExistsError("output destination appeared during commit")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coordinate-free Gamma-LS conditioning sensitivity"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--base-preflight", required=True)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--device")
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--sensitivity-preflight", required=True)
    run.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = ConditioningSensitivityConfig.load(arguments.config)
    if arguments.command == "preflight":
        payload = run_sensitivity_preflight(
            config,
            base_preflight_dir=arguments.base_preflight,
            output_dir=arguments.output,
            device=arguments.device,
        )
    else:
        payload = run_conditioning_sensitivity(
            config,
            sensitivity_preflight_dir=arguments.sensitivity_preflight,
            output_dir=arguments.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ANCHOR",
    "ConditioningSensitivityConfig",
    "ConditioningSensitivityConfigError",
    "ConditioningSensitivityUnavailable",
    "ConditioningSetting",
    "EMA_ALPHAS",
    "GAUSSIAN_SIGMAS",
    "REPRESENTATIONS",
    "SCALE_FLOOR_PERCENTILES",
    "SensitivityExecution",
    "causal_condition_dense",
    "conditioning_grid_rows",
    "enumerate_settings",
    "freeze_fold_selections",
    "main",
    "pareto_layers",
    "raw_lane_semantics",
    "run_conditioning_sensitivity",
    "run_sensitivity_preflight",
    "select_training_setting",
    "spatial_gaussian_reflect",
    "verify_selection_seal",
]


if __name__ == "__main__":
    raise SystemExit(main())
