"""Canonical, non-claim-bearing runner for NREV-EXP-0021.

This module deliberately separates three operations:

* a read-only, fail-closed preflight of identities, authorities, and hashes;
* a single frozen contiguous-x positive-unlabeled evaluation; and
* an atomic artifact publication step whose index covers the complete tree.

The estimand is within-recording held-out positive-versus-unlabeled ranking.
Unlabeled event proposals and quiet source-off proposals are never negatives,
and coordinates are used only to freeze folds and guard bands, never as model
inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from neurobench.experiments.neuron_identifiability import uncertainty_aware_census as census_core
from neurobench.experiments.neuron_identifiability import uncertainty_aware_data as data_core
from neurobench.experiments.neuron_identifiability import uncertainty_aware_models as model_core


SCHEMA_VERSION = 1
EXPERIMENT_ID = "NREV-EXP-0021"
RUN_ID = "NREV-RUN-EXP-0021-SCREEN-20260830-B"
DESCRIPTOR_ID = "NREV-DATA-UNCERTAINTY-AWARE-LEARNING-SOURCES-V1"
RUNNER_MODULE = "neurobench.experiments.neuron_identifiability.uncertainty_aware_learning"
CONFIG_RELATIVE_PATH = Path("examples/uncertainty_aware_feature_learning_v1_1.example.json")
PROTOCOL_RELATIVE_PATH = Path("docs/workflows/uncertainty_aware_feature_learning_v1_1.md")
DESCRIPTOR_RELATIVE_PATH = Path("research/data-registry/uncertainty_aware_learning_sources_v1.json")
RUNNER_RELATIVE_PATH = Path("neurobench/experiments/neuron_identifiability/uncertainty_aware_learning.py")
DEPENDENCY_DECLARATION_RELATIVE_PATH = Path("pyproject.toml")
OUTPUT_RELATIVE_PATH = Path(
    "Outputs/NeuronIdentifiability/NREV-EXP-0021/runs/"
    "NREV-RUN-EXP-0021-SCREEN-20260830-B"
)

EXPECTED_IMPLEMENTATIONS = {
    "neurobench/experiments/neuron_identifiability/uncertainty_aware_census.py":
        "frozen_broad_candidate_census",
    "neurobench/experiments/neuron_identifiability/uncertainty_aware_data.py":
        "label_harmonization_exclusion_and_fold_contract",
    "neurobench/experiments/neuron_identifiability/uncertainty_aware_models.py":
        "fold_local_baselines_linear_and_tiny_mlp_evaluation",
    "neurobench/algorithms/proposal_ranking.py":
        "upstream_candidate_nms_and_deduplication",
    "neurobench/experiments/hierarchical_parzen_ica/feature_utility_config.py":
        "upstream_feature_utility_configuration",
    "neurobench/experiments/hierarchical_parzen_ica/feature_utility_program.py":
        "upstream_feature_utility_implementation",
    "neurobench/experiments/hierarchical_parzen_ica/innovation_ranker_config.py":
        "upstream_map_and_proposal_configuration",
    "neurobench/experiments/hierarchical_parzen_ica/innovation_ranker_program.py":
        "upstream_map_and_proposal_implementation",
    "neurobench/experiments/learnable_contrast/core.py":
        "legacy_event_window_label_loader",
    "neurobench/experiments/pairwise_separation/evaluation.py":
        "legacy_event_interval_and_temporal_pool_contract",
    "neurobench/metrics/sparse_detection.py":
        "upstream_sparse_detection_primitives",
    RUNNER_RELATIVE_PATH.as_posix(): "canonical_runner_provenance_outputs_and_validation",
}
LOADED_UPSTREAM_MODULES = {
    "neurobench.algorithms.proposal_ranking":
        "neurobench/algorithms/proposal_ranking.py",
    "neurobench.experiments.hierarchical_parzen_ica.feature_utility_config":
        "neurobench/experiments/hierarchical_parzen_ica/feature_utility_config.py",
    "neurobench.experiments.hierarchical_parzen_ica.feature_utility_program":
        "neurobench/experiments/hierarchical_parzen_ica/feature_utility_program.py",
    "neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config":
        "neurobench/experiments/hierarchical_parzen_ica/innovation_ranker_config.py",
    "neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_program":
        "neurobench/experiments/hierarchical_parzen_ica/innovation_ranker_program.py",
    "neurobench.experiments.learnable_contrast.core":
        "neurobench/experiments/learnable_contrast/core.py",
    "neurobench.experiments.pairwise_separation.evaluation":
        "neurobench/experiments/pairwise_separation/evaluation.py",
    "neurobench.metrics.sparse_detection": "neurobench/metrics/sparse_detection.py",
}
EXPECTED_SOURCE_LOCATORS = {
    "innovation_ranker_v5_resolved_config": (
        "data://Outputs/HierarchicalParzenICA/"
        "spon_ca_burst_innovation_ranker_v5/config.resolved.json"
    ),
    "source_video": (
        "data://Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/"
        "spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
    ),
    "legacy_event_window_labels": "data://Inputs/Spon Ca Burst/labels/labels_normalized.tsv",
    "feature_manifest": (
        "data://Outputs/HierarchicalParzenICA/"
        "spon_ca_burst_feature_utility_v1/feature_manifest.json"
    ),
    "feature_utility_resolved_config": (
        "data://Outputs/HierarchicalParzenICA/"
        "spon_ca_burst_feature_utility_v1/config.resolved.json"
    ),
    "feature_utility_run_state": (
        "data://Outputs/HierarchicalParzenICA/"
        "spon_ca_burst_feature_utility_v1/run_state.json"
    ),
    **{
        f"feature_array_{feature}": (
            "data://Outputs/HierarchicalParzenICA/"
            f"spon_ca_burst_feature_utility_v1/features/{feature}.npy"
        )
        for feature in (
            "carrier_signed",
            "local_psd_signal",
            "asymmetric_state",
            "spatial_coherence",
            "cross_scale_rank",
            "cross_scale_recall",
            "cfar_score",
            "cfar_background",
            "cfar_noise",
            "derivative_positive_lag1",
            "derivative_negative_lag1",
            "persistence_activity_gate",
            "persistent_artifact_score",
        )
    },
    "feature_structure_unit": (
        "data://Outputs/HierarchicalParzenICA/"
        "spon_ca_burst_feature_utility_v1/structure_unit.npy"
    ),
    "canonical_v7_adjudication": (
        "data://Outputs/HardROIAdjudication/"
        "spon_ca_burst_hard_roi_adjudication_final_v7/adjudication_final.tsv"
    ),
    "new_candidate_single_reviewer_labels": (
        "repo-workspace://Outputs/NeuronIdentifiability/"
        "new_candidate_roi_review_batch_v1/user_review_v1.tsv"
    ),
    "detection_profile_taxonomy_v5_sites": (
        "repo-workspace://Outputs/NeuronIdentifiability/"
        "spon_ca_burst_identifiability_paper_v1_v8/"
        "detection_profile_taxonomy_v5/detection_site_profiles.tsv"
    ),
    "detection_profile_taxonomy_v5_occurrences": (
        "repo-workspace://Outputs/NeuronIdentifiability/"
        "spon_ca_burst_identifiability_paper_v1_v8/"
        "detection_profile_taxonomy_v5/detection_occurrence_profiles.tsv"
    ),
}
CONFIG_INPUT_TO_SOURCE = {
    "innovation_ranker_v5_config_relative_to_data_root": "innovation_ranker_v5_resolved_config",
    "canonical_v7_labels_relative_to_data_root": "canonical_v7_adjudication",
    "new_review_labels_relative_to_repository": "new_candidate_single_reviewer_labels",
    "legacy_taxonomy_sites_relative_to_repository": "detection_profile_taxonomy_v5_sites",
    "legacy_taxonomy_occurrences_relative_to_repository": "detection_profile_taxonomy_v5_occurrences",
}
PRIMARY_FEATURES = (
    "carrier_signed",
    "local_psd_signal",
    "asymmetric_state",
    "spatial_coherence",
    "cross_scale_rank",
    "cross_scale_recall",
    "cfar_score",
    "cfar_background",
    "cfar_noise",
    "persistent_artifact_score",
    "cut_center_sigma2p5",
    "cut_ring_r4p5_t1p25",
)
EXPERT_FEATURES = PRIMARY_FEATURES[:7]
EXPECTED_METHODS = (
    "carrier_signed",
    "cfar_score",
    "expert_separation_equal_weight",
    "positive_reference",
    "linear_spu",
    "elastic_linear_spu",
    "bagged_pu_linear",
    "tiny_mlp_spu",
)
MODEL_RESULT_NAMES = {
    "equal_weight_feature_separation": "expert_separation_equal_weight",
    "positive_reference_distance": "positive_reference",
    "logistic_l2": "linear_spu",
    "logistic_elastic": "elastic_linear_spu",
    "bagged_pu_logistic": "bagged_pu_linear",
    "tiny_mlp_4_tanh": "tiny_mlp_spu",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_END_TO_END_DEADLINE_SECONDS = 30 * 60


class LearningContractError(RuntimeError):
    """Raised when the frozen EXP-0021 execution contract is not exact."""


class LearningDeadlineExceeded(LearningContractError):
    """Raised by the frozen process alarm before an overlong run can promote."""


class _PosixProcessDeadline:
    """Bound the public run from entry through atomic promotion."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self._previous_handler: Any = None
        self._armed = False

    @staticmethod
    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise LearningDeadlineExceeded("end-to-end process deadline exceeded")

    def __enter__(self) -> "_PosixProcessDeadline":
        if os.name != "posix" or not hasattr(signal, "setitimer"):
            raise LearningContractError("POSIX process-alarm deadline is unavailable")
        if self.seconds <= 0:
            raise LearningContractError("process deadline must be positive")
        active_seconds, active_interval = signal.getitimer(signal.ITIMER_REAL)
        if active_seconds > 0 or active_interval > 0:
            raise LearningContractError("an existing process alarm would make the run ambiguous")
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        self._armed = True
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self._armed:
            try:
                self.disarm_for_atomic_commit()
            except Exception:
                # The run body has already failed or returned.  Never let
                # cleanup mask the scientifically relevant exception.
                pass

    def disarm_for_atomic_commit(self) -> None:
        """Synchronously disarm while the validated tree is still partial."""

        if not self._armed:
            raise LearningContractError("process deadline is not armed at commit")
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._previous_handler)
        self._armed = False


def _runner_path() -> Path:
    """Indirection retained so tests can validate root authority safely."""

    return Path(__file__).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LearningContractError("cannot capture authoritative Git provenance") from exc
    return completed.stdout


def _credential_free_remote(raw: str) -> str:
    value = raw.strip()
    if value.startswith("git@") and ":" in value:
        authority, path = value.removeprefix("git@").split(":", 1)
        value = f"https://{authority}/{path.removesuffix('.git')}"
    elif value.startswith("ssh://git@"):
        authority_and_path = value.removeprefix("ssh://git@")
        authority, path = authority_and_path.split("/", 1)
        value = f"https://{authority}/{path.removesuffix('.git')}"
    elif value.endswith(".git"):
        value = value[:-4]
    if not value.startswith("https://") or "@" in value.removeprefix("https://").split("/", 1)[0]:
        raise LearningContractError("Git origin must normalize to credential-free HTTPS")
    return value


def capture_git_provenance(repository: Path) -> dict[str, Any]:
    """Capture commit plus content-aware dirty state at execution start."""

    repository = Path(repository).resolve()
    worktree = Path(
        _git_bytes(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    if worktree != repository:
        raise LearningContractError("repository_root is not the substantive Git worktree")
    status = _git_bytes(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    commit = _git_bytes(repository, "rev-parse", "HEAD").decode("ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise LearningContractError("Git commit is not a full SHA")
    branch = _git_bytes(repository, "branch", "--show-current").decode("utf-8").strip() or "detached"
    result: dict[str, Any] = {
        "repository": _credential_free_remote(
            _git_bytes(repository, "remote", "get-url", "origin").decode("utf-8")
        ),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_entry_count": status.count(b"\0"),
        "dirty_status_sha256": hashlib.sha256(status).hexdigest(),
        "dirty_paths_persisted": False,
        "diff_algorithm": "NEUREV-DIRTY-STATE-V1",
    }
    if status:
        digest = hashlib.sha256()
        digest.update(b"NEUREV-DIRTY-STATE-V1\0")
        digest.update(status)
        digest.update(_git_bytes(repository, "diff", "--binary", "HEAD", "--"))
        untracked = _git_bytes(
            repository, "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0")
        for encoded in sorted(item for item in untracked if item):
            relative = encoded.decode("utf-8", errors="surrogateescape")
            path = repository / relative
            digest.update(b"\0PATH\0")
            digest.update(encoded)
            if path.is_symlink():
                digest.update(b"\0SYMLINK\0")
                digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            elif path.is_file():
                digest.update(b"\0FILE\0")
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        result["diff_sha256"] = digest.hexdigest()
    return result


def _validate_repository_authority(
    repository: Path,
    contract: Mapping[str, Any],
    *,
    captured_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the checkout matches the frozen Git authority."""

    observed = dict(
        captured_provenance
        if captured_provenance is not None
        else capture_git_provenance(repository)
    )
    if observed.get("repository") != contract.get("origin_https"):
        raise LearningContractError("Git origin differs from the frozen repository authority")
    if observed.get("branch") != contract.get("branch"):
        raise LearningContractError("Git branch differs from the frozen repository authority")
    base_commit = str(contract.get("base_commit_sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise LearningContractError("frozen repository base commit is invalid")
    # A successful, read-only ancestry query proves that the executed commit
    # includes the declared base without requiring HEAD to equal that base.
    _git_bytes(repository, "merge-base", "--is-ancestor", base_commit, "HEAD")
    if observed.get("dirty"):
        if not contract.get("dirty_execution_allowed_with_content_digest"):
            raise LearningContractError("dirty execution is not authorized")
        if not _SHA256_RE.fullmatch(str(observed.get("diff_sha256", ""))):
            raise LearningContractError("dirty checkout lacks a content-aware diff digest")
    return observed


def _runtime_provenance(repository: Path) -> dict[str, Any]:
    dependency_declaration = repository / DEPENDENCY_DECLARATION_RELATIVE_PATH
    if not dependency_declaration.is_file():
        raise LearningContractError("declared dependency file pyproject.toml is absent")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "tifffile_version": importlib.metadata.version("tifffile"),
        "psutil_version": psutil.__version__,
        "threadpoolctl_version": importlib.metadata.version("threadpoolctl"),
        "platform": f"{sys.platform}/{platform.machine() or 'unknown-machine'}",
        "dependency_declaration": {
            "path": "repo-workspace://" + DEPENDENCY_DECLARATION_RELATIVE_PATH.as_posix(),
            "sha256": _sha256_file(dependency_declaration),
            "semantics": "project_dependency_declaration_not_fully_resolved_environment_lock",
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearningContractError(f"cannot read valid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise LearningContractError(f"JSON root must be an object: {path.name}")
    return value


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise LearningContractError(f"TSV has no header: {path.name}")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise LearningContractError(f"cannot read TSV: {path.name}") from exc
    return tuple(reader.fieldnames), rows


def _inside(path: Path, root: Path, role: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise LearningContractError(f"{role} escapes its declared authority") from exc


def _resolve_inside(root: Path, relative: str | Path, role: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise LearningContractError(f"{role} must be a portable relative path")
    result = (root / raw).resolve()
    _inside(result, root, role)
    return result


def _portable_path(path: Path, *, repository: Path, data: Path) -> str:
    resolved = path.resolve()
    try:
        return "repo-workspace://" + resolved.relative_to(repository).as_posix()
    except ValueError:
        pass
    try:
        return "data://" + resolved.relative_to(data).as_posix()
    except ValueError as exc:
        raise LearningContractError("cannot serialize a path outside declared authorities") from exc


def _source_path(locator: str, *, repository: Path, data: Path) -> Path:
    if locator.startswith("data://"):
        return _resolve_inside(data, locator.removeprefix("data://"), "data source")
    if locator.startswith("repo-workspace://"):
        return _resolve_inside(
            repository, locator.removeprefix("repo-workspace://"), "repository source"
        )
    raise LearningContractError(f"unsupported source locator: {locator!r}")


def _exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise LearningContractError(f"{name} differs from frozen contract")


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact(config.get("schema_version"), 1, "config schema_version")
    _exact(config.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _exact(config.get("run_id"), RUN_ID, "run_id")
    _exact(config.get("runner_module"), RUNNER_MODULE, "runner_module")
    _exact(config.get("protocol_path"), PROTOCOL_RELATIVE_PATH.as_posix(), "protocol_path")
    _exact(config.get("output_root"), OUTPUT_RELATIVE_PATH.as_posix(), "output_root")
    _exact(
        config.get("execution_role"),
        "bounded_non_claim_bearing_positive_unlabeled_feature_fusion_screen",
        "execution_role",
    )
    _exact(config.get("engineering_execution_authorized"), True, "engineering authorization")
    for key in (
        "scientific_execution_authorized",
        "claim_bearing_execution_authorized",
        "scientific_completion_allowed",
        "claim_or_evidence_capsule_creation_allowed",
    ):
        _exact(config.get(key), False, key)
    _exact(config.get("input_descriptor"), DESCRIPTOR_RELATIVE_PATH.as_posix(), "input_descriptor")
    _exact(config.get("data_root_environment_variable"), "NEUROBENCH_DATA_ROOT", "data-root variable")
    repository_contract = config.get("repository_contract", {})
    _exact(
        repository_contract.get("origin_https"),
        "https://github.com/Jibby2k1/NeuRev-Workbench",
        "repository origin",
    )
    _exact(
        repository_contract.get("branch"),
        "codex/neuron-identifiability-paper-20260822",
        "repository branch",
    )
    _exact(
        repository_contract.get("base_commit_sha"),
        "0fe1b38aec6e1748fa5ec056aa883977d9999f05",
        "repository base commit",
    )
    _exact(
        repository_contract.get("dirty_execution_allowed_with_content_digest"),
        True,
        "dirty execution policy",
    )

    candidate = config.get("candidate_census", {})
    _exact(candidate.get("materialized_census_id"), "innovation_ranker_v5_broad_proposal_census", "census ID")
    _exact(candidate.get("harmonizer_contract_id"), "innovation_ranker_v5_broad_union", "harmonizer ID")
    _exact(candidate.get("proposal_source_count"), 22, "proposal source count")
    _exact(candidate.get("per_source_limit"), 100, "per-source limit")
    _exact(candidate.get("nms_distance_px"), 6, "NMS distance")
    _exact(float(candidate.get("dedupe_radius_px", -1)), 3.0, "dedupe radius")
    _exact(candidate.get("event_candidate_counts_by_burst"), {"1": 544, "2": 530, "3": 476, "4": 449}, "event counts")
    _exact(candidate.get("quiet_candidate_counts_by_map"), [683, 748, 759, 748], "quiet counts")
    _exact(candidate.get("feature_count"), 34, "census feature count")
    _exact(
        candidate.get("current_label_tables_joined_only_after_census_materialization"),
        True,
        "current-label join timing",
    )
    _exact(
        candidate.get("canonical_v7_or_new_review_labels_used_to_rank_or_select_candidates"),
        False,
        "current-label candidate selection policy",
    )
    _exact(candidate.get("legacy_label_derived_event_windows_used"), True, "legacy event windows")
    _exact(
        candidate.get("historically_label_informed_feature_program"),
        True,
        "historically label-informed feature program",
    )

    label = config.get("label_contract", {})
    for key, expected in (
        ("canonical_confirmed_match_radius_px", 6.0),
        ("canonical_inclusive_exclusion_radius_px", 12.0),
        ("new_review_reserve_radius_px", 6.0),
        ("new_review_unlabeled_exclusion_radius_px", 12.0),
    ):
        _exact(float(label.get(key, -1)), expected, key)
    _exact(label.get("unmatched_event_candidates"), "unlabeled_unknown", "unmatched candidate policy")
    _exact(label.get("quiet_candidates"), "source_off_null_control", "quiet policy")
    _exact(label.get("review_artifact_label"), "case_study_not_negative", "review artifact policy")
    _exact(label.get("review_neighborhoods_reserved_from_primary_fit_and_evaluation"), True, "review reserve")

    features = config.get("features", {})
    _exact(tuple(features.get("primary_compact_feature_ids", ())), PRIMARY_FEATURES, "primary features")
    _exact(tuple(features.get("expert_separation_feature_ids", ())), EXPERT_FEATURES, "expert features")
    _exact(tuple(features.get("scalar_baselines", ())), ("carrier_signed", "cfar_score"), "baselines")
    model_core.validate_raw_feature_names(PRIMARY_FEATURES)

    outer = config.get("outer_validation", {})
    _exact(
        outer.get("scheme"),
        "five_positive_anchor_median_x_blocks_with_whole_leakage_group_closure",
        "outer scheme",
    )
    _exact(outer.get("fold_count"), 5, "fold count")
    _exact(float(outer.get("boundary_guard_px", -1)), 12.0, "boundary guard")
    _exact(
        outer.get("positive_group_representative_x"),
        "median_unreserved_positive_anchor_x",
        "positive group representative",
    )
    _exact(
        outer.get("unlabeled_only_group_representative_x"),
        "median_all_eligible_group_rows_x",
        "unlabeled group representative",
    )
    _exact(
        outer.get("raw_candidate_extent_overlap_allowed_only_by_whole_group_closure"),
        True,
        "whole-group closure overlap policy",
    )
    _exact(
        outer.get("training_guard_rule"),
        "purge_whole_group_on_nominal_boundary_plus_or_minus_12px_interval_intersection_or_when_any_member_is_within_inclusive_euclidean_12px_of_any_held_test_candidate",
        "training guard rule",
    )
    _exact(outer.get("minimum_positive_anchors_per_fold"), 10, "minimum fold positives")
    _exact(outer.get("minimum_unlabeled_spatial_groups_per_fold"), 10, "minimum fold U groups")
    _exact(float(outer.get("spatial_group_cell_size_px", -1)), 12.0, "spatial group cell size")
    _exact(
        outer.get("expected_assignment_sha256"),
        "517fbb0ef37ea7b57917367f9e39efe931c01ea9f848a53acde097cc47b221ba",
        "expected fold assignment hash",
    )
    _exact(
        outer.get("expected_train_test_guard_membership_sha256"),
        "4425f9a12a8d367969edd50c2daf4192b450d4de27fc5776628881c5538e0a9c",
        "expected train/test/guard membership hash",
    )
    _exact(outer.get("identity_cross_fold_allowed"), False, "identity crossing")
    _exact(outer.get("feature_or_score_informed_boundaries_allowed"), False, "boundary information")
    _exact(outer.get("all_eligible_event_rows_receive_exactly_one_oof_score"), True, "OOF coverage")

    models = config.get("models", {})
    _exact(tuple(models.get("methods", ())), EXPECTED_METHODS, "method set/order")
    _exact(models.get("hyperparameter_selection"), "none_fixed_before_outer_scoring", "hyperparameter policy")
    _exact(models.get("output_semantics"), "positive_versus_unlabeled_ranking_score_not_probability", "score semantics")
    _exact(models.get("tiny_mlp_spu", {}).get("hidden_units"), 4, "MLP width")
    _exact(models.get("tiny_mlp_spu", {}).get("activation"), "tanh", "MLP activation")
    _exact(models.get("tiny_mlp_spu", {}).get("seeds"), list(range(3101, 3111)), "MLP seeds")
    linear = models.get("linear_spu", {})
    elastic = models.get("elastic_linear_spu", {})
    bagged = models.get("bagged_pu_linear", {})
    mlp = models.get("tiny_mlp_spu", {})
    for name, value, expected in (
        ("linear l2", linear.get("l2"), 0.01),
        ("linear iterations", linear.get("maximum_proximal_gradient_iterations"), 50_000),
        ("linear tolerance", linear.get("convergence_tolerance"), 1e-6),
        ("elastic penalty", elastic.get("penalty_strength"), 0.01),
        ("elastic l1 ratio", elastic.get("l1_ratio"), 0.5),
        ("elastic iterations", elastic.get("maximum_proximal_gradient_iterations"), 50_000),
        ("elastic tolerance", elastic.get("convergence_tolerance"), 1e-6),
        ("bagged bags", bagged.get("bags"), 10),
        ("bagged U per P", bagged.get("unlabeled_per_positive"), 3),
        ("bagged l2", bagged.get("base_l2"), 0.01),
        ("bagged iterations", bagged.get("base_maximum_proximal_gradient_iterations"), 50_000),
        ("bagged tolerance", bagged.get("base_convergence_tolerance"), 1e-6),
        ("bagged seed", bagged.get("seed"), 2101),
        ("MLP hidden units", mlp.get("hidden_units"), 4),
        ("MLP learning rate", mlp.get("learning_rate"), 0.003),
        ("MLP l2", mlp.get("l2"), 0.01),
        ("MLP epochs", mlp.get("maximum_epochs"), 500),
        ("MLP patience", mlp.get("training_only_early_stop_patience"), 35),
        ("MLP selection folds", mlp.get("training_only_epoch_selection_folds"), 2),
    ):
        _exact(value, expected, name)
    _exact(mlp.get("optimizer"), "deterministic_full_batch_Adam_style_with_explicit_L2", "MLP optimizer")
    _exact(mlp.get("batch_mode"), "deterministic_full_batch", "MLP batch mode")
    _exact(mlp.get("seed_aggregation"), "mean_score_per_candidate", "MLP seed aggregation")

    evaluation = config.get("evaluation", {})
    _exact(evaluation.get("global_candidate_budgets"), [20, 58, 100], "candidate budgets")
    _exact(evaluation.get("primary_budget"), 58, "primary budget")
    _exact(evaluation.get("primary_metric"), "spu_auc", "primary metric")
    _exact(evaluation.get("canonical_proposal_stage_misses_in_recall_denominator"), True, "miss denominator")
    _exact(evaluation.get("grouped_bootstrap_draws"), 2000, "bootstrap draws")
    _exact(evaluation.get("grouped_bootstrap_seed"), 4101, "bootstrap seed")
    _exact(
        evaluation.get("grouped_bootstrap_resampling_unit"),
        "whole_leakage_group_with_positive_and_unlabeled_members_joint",
        "bootstrap resampling unit",
    )
    _exact(
        evaluation.get("grouped_bootstrap_strata"),
        "positive_containing_vs_unlabeled_only_within_outer_fold",
        "bootstrap strata",
    )
    _exact(evaluation.get("representative_unit_score_permutations"), 2000, "permutation draws")
    _exact(evaluation.get("representative_unit_score_permutation_seed"), 4201, "permutation seed")
    _exact(
        evaluation.get("permutation_semantics"),
        "fold_stratified_one_pre_score_representative_per_whole_leakage_group_score_assignment_permutation_posthoc_not_refitted",
        "permutation semantics",
    )
    _exact(
        evaluation.get("cross_fold_score_contract", {}).get("global_fixed_budget_score"),
        "label_free_within_test_fold_midrank_percentile",
        "global score scale",
    )
    _exact(evaluation.get("ordinary_auc_precision_specificity_or_fpr_language_allowed"), False, "metric language")

    audits = config.get("locked_audits", {})
    _exact(audits.get("panel_values_may_tune_or_select_model"), False, "audit tuning policy")
    scientific = audits.get("scientific_audit", {})
    _exact(scientific.get("enabled"), True, "scientific audit enabled")
    _exact(scientific.get("full_three_section_media_generated_in_this_screen"), False, "media status")
    _exact(scientific.get("required_status"), "incomplete", "scientific audit status")
    resources = config.get("resources", {})
    _exact(resources.get("device"), "cpu", "device")
    _exact(resources.get("numpy_threads"), 1, "NumPy thread cap")
    _exact(
        resources.get("maximum_end_to_end_wall_minutes"),
        30,
        "maximum end-to-end wall minutes",
    )
    _exact(
        resources.get("deadline_enforcement"),
        "posix_process_alarm_from_run_start_through_pre_promotion_validation_then_disarm_before_single_atomic_rename",
        "deadline enforcement",
    )
    _exact(resources.get("atomic_partial_then_rename"), True, "atomic output")
    _exact(resources.get("resume_supported"), False, "resume policy")
    _exact(resources.get("refuse_existing_final_or_partial"), True, "collision policy")
    advance = config.get("advance_signals", {})
    for name, expected in (
        ("minimum_unreserved_positive_anchors", 75),
        ("tiny_mlp_minus_linear_spu_auc_minimum", 0.03),
        ("tiny_mlp_minus_linear_grouped_ci_lower_minimum", 0.0),
        ("tiny_mlp_minus_cfar_spu_auc_minimum", 0.03),
        ("tiny_mlp_minus_cfar_grouped_ci_lower_minimum", 0.0),
        ("tiny_mlp_minus_strongest_frozen_recall_at_58_minimum", 0.03),
        ("maximum_per_fold_recall_loss", 0.10),
        ("maximum_representative_unit_permutation_p", 0.05),
        ("minimum_mlp_seed_rank_correlation", 0.90),
        ("maximum_mlp_seed_recall_at_58_range", 0.05),
    ):
        _exact(advance.get(name), expected, f"advance threshold {name}")
    _exact(
        advance.get("interpretation"),
        "descriptive_engineering_advance_signal_not_scientific_gate",
        "advance signal interpretation",
    )


def _preflight_context(
    config_path: Path = CONFIG_RELATIVE_PATH,
    *,
    repository_root: Path | None = None,
    data_root: Path | None = None,
    git_provenance_at_start: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runner = _runner_path()
    repository = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else runner.parents[3]
    )
    if not repository.is_dir() or not (repository / "AGENTS.md").is_file():
        raise LearningContractError("repository_root is not the substantive NeuRev checkout")
    _inside(runner, repository, "loaded runner")
    if runner != (repository / RUNNER_RELATIVE_PATH).resolve():
        raise LearningContractError("loaded runner is not at the frozen repository path")
    for module, relative in (
        (census_core, Path("neurobench/experiments/neuron_identifiability/uncertainty_aware_census.py")),
        (data_core, Path("neurobench/experiments/neuron_identifiability/uncertainty_aware_data.py")),
        (model_core, Path("neurobench/experiments/neuron_identifiability/uncertainty_aware_models.py")),
    ):
        loaded = Path(str(module.__file__)).resolve()
        if loaded != (repository / relative).resolve():
            raise LearningContractError(f"loaded module authority differs: {relative.as_posix()}")
    for module_name, relative in LOADED_UPSTREAM_MODULES.items():
        module = sys.modules.get(module_name)
        loaded_path = getattr(module, "__file__", None)
        if loaded_path is None:
            raise LearningContractError(f"pinned upstream module is not loaded: {module_name}")
        if Path(str(loaded_path)).resolve() != (repository / relative).resolve():
            raise LearningContractError(f"loaded upstream authority differs: {relative}")
    raw_config_path = Path(config_path).expanduser()
    config_file = (
        raw_config_path.resolve()
        if raw_config_path.is_absolute()
        else _resolve_inside(repository, raw_config_path, "config")
    )
    _inside(config_file, repository, "config")
    if config_file != (repository / CONFIG_RELATIVE_PATH).resolve():
        raise LearningContractError("config_path is not the maintained EXP-0021 config")
    if not config_file.is_file():
        raise LearningContractError("maintained EXP-0021 config is absent")
    config = _read_json(config_file)
    _validate_config(config)
    repository_git = _validate_repository_authority(
        repository,
        config["repository_contract"],
        captured_provenance=git_provenance_at_start,
    )

    if data_root is None:
        raw_data = os.environ.get(str(config["data_root_environment_variable"]), "")
        if not raw_data:
            raise LearningContractError("NEUROBENCH_DATA_ROOT is required")
        data = Path(raw_data).expanduser().resolve()
    else:
        data = Path(data_root).expanduser().resolve()
    if not data.is_dir():
        raise LearningContractError("data_root does not exist")

    output = _resolve_inside(repository, config["output_root"], "output root")
    if output != (repository / OUTPUT_RELATIVE_PATH).resolve():
        raise LearningContractError("output root differs from the frozen run")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("completed or partial EXP-0021 output already exists")

    protocol = _resolve_inside(repository, config["protocol_path"], "protocol")
    descriptor_file = _resolve_inside(repository, config["input_descriptor"], "input descriptor")
    dependency_declaration = _resolve_inside(
        repository, DEPENDENCY_DECLARATION_RELATIVE_PATH, "dependency declaration"
    )
    if not dependency_declaration.is_file():
        raise LearningContractError("dependency declaration is absent")
    if _sha256_file(protocol) != config.get("protocol_sha256"):
        raise LearningContractError("protocol hash differs from config")
    if _sha256_file(descriptor_file) != config.get("input_descriptor_sha256"):
        raise LearningContractError("input descriptor hash differs from config")
    descriptor = _read_json(descriptor_file)
    _exact(descriptor.get("schema_version"), 1, "descriptor schema")
    _exact(descriptor.get("descriptor_id"), DESCRIPTOR_ID, "descriptor ID")
    _exact(descriptor.get("experiment_id"), EXPERIMENT_ID, "descriptor experiment ID")
    sources = descriptor.get("sources")
    if not isinstance(sources, list):
        raise LearningContractError("descriptor sources must be a list")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for record in sources:
        if not isinstance(record, dict) or not isinstance(record.get("source_id"), str):
            raise LearningContractError("descriptor source record is malformed")
        if record["source_id"] in source_by_id:
            raise LearningContractError("descriptor source IDs are not unique")
        source_by_id[record["source_id"]] = record
    _exact(set(source_by_id), set(EXPECTED_SOURCE_LOCATORS), "descriptor source IDs")

    source_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for source_id, locator in EXPECTED_SOURCE_LOCATORS.items():
        record = source_by_id[source_id]
        _exact(record.get("portable_path"), locator, f"{source_id} locator")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            raise LearningContractError(f"{source_id} descriptor hash is invalid")
        path = _source_path(locator, repository=repository, data=data)
        if not path.is_file():
            raise LearningContractError(f"required input is absent: {source_id}")
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            raise LearningContractError(f"input hash differs: {source_id}")
        source_paths[source_id] = path
        source_records[source_id] = {
            "path": locator,
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
        }
    for config_key, source_id in CONFIG_INPUT_TO_SOURCE.items():
        configured = str(config.get("inputs", {}).get(config_key, ""))
        locator = EXPECTED_SOURCE_LOCATORS[source_id]
        expected_relative = locator.split("://", 1)[1]
        _exact(configured, expected_relative, f"config input {config_key}")

    hash_contract = config.get("implementation_hash_contract", {})
    _exact(hash_contract.get("algorithm"), "sha256", "implementation hash algorithm")
    _exact(hash_contract.get("scope"), "complete_file_bytes", "implementation hash scope")
    implementation_records = hash_contract.get("sources")
    if not isinstance(implementation_records, list):
        raise LearningContractError("implementation hash sources must be a list")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in implementation_records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise LearningContractError("implementation hash record is malformed")
        if record["path"] in by_path:
            raise LearningContractError("implementation paths are not unique")
        by_path[record["path"]] = record
    _exact(set(by_path), set(EXPECTED_IMPLEMENTATIONS), "implementation source paths")
    code_records: dict[str, dict[str, Any]] = {}
    for relative, role in EXPECTED_IMPLEMENTATIONS.items():
        record = by_path[relative]
        _exact(record.get("role"), role, f"implementation role {relative}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            raise LearningContractError(f"implementation hash is not pinned: {relative}")
        path = _resolve_inside(repository, relative, "implementation source")
        if not path.is_file():
            raise LearningContractError(f"implementation source is absent: {relative}")
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            raise LearningContractError(f"implementation hash differs: {relative}")
        code_records[relative] = {
            "path": "repo-workspace://" + relative,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
        }

    census_output = partial / "census"
    census_progress = partial / "progress" / "census.jsonl"
    census_preflight = census_core.preflight_uncertainty_aware_census(
        repository_root=repository,
        data_root=data,
        config_path=source_paths["innovation_ranker_v5_resolved_config"],
        output_root=census_output,
        progress=census_progress,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "read_only_uncertainty_aware_learning_preflight",
        "ready": True,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "gates": {
            "repository_authority_exact": True,
            "repository_git_origin_branch_and_base_exact": True,
            "loaded_module_authority_exact_for_all_pinned_execution_modules": True,
            "dirty_state_content_digest_present_when_required": (
                not repository_git.get("dirty")
                or bool(_SHA256_RE.fullmatch(str(repository_git.get("diff_sha256", ""))))
            ),
            "execution_authorization_bounded": True,
            "output_absent": True,
            "partial_output_absent": True,
            "protocol_hash_exact": True,
            "descriptor_hash_exact": True,
            "descriptor_identity_exact": True,
            "dependency_declaration_present_and_hashed": True,
            "input_hashes_exact": True,
            "implementation_hashes_exact": True,
            "census_preflight_ready": bool(census_preflight.get("ready")),
        },
        "paths": {
            "config": _portable_path(config_file, repository=repository, data=data),
            "protocol": _portable_path(protocol, repository=repository, data=data),
            "descriptor": _portable_path(descriptor_file, repository=repository, data=data),
            "output": _portable_path(output, repository=repository, data=data),
        },
        "hashes": {
            "config_sha256": _sha256_file(config_file),
            "protocol_sha256": _sha256_file(protocol),
            "descriptor_sha256": _sha256_file(descriptor_file),
            "dependency_declaration_sha256": _sha256_file(dependency_declaration),
        },
        "inputs": source_records,
        "implementation": code_records,
        "repository_git": repository_git,
        "census_preflight": census_preflight,
        "claim_boundary": {
            "scientific_audit_complete": False,
            "scientific_completion": False,
            "claim_promotion_allowed": False,
        },
    }
    return {
        "payload": payload,
        "repository": repository,
        "data": data,
        "config_file": config_file,
        "config": config,
        "repository_git": repository_git,
        "protocol": protocol,
        "descriptor_file": descriptor_file,
        "descriptor": descriptor,
        "source_paths": source_paths,
        "output": output,
        "partial": partial,
        "census_output": census_output,
        "census_progress": census_progress,
    }


def preflight_uncertainty_aware_learning(
    config_path: Path = CONFIG_RELATIVE_PATH,
    *,
    repository_root: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the exact run authority and hashes without writing anything."""

    return _preflight_context(
        config_path, repository_root=repository_root, data_root=data_root
    )["payload"]


def _leakage_group_units(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Materialize whole leakage groups with deterministic x representatives.

    Positive groups use the median x of their unreserved canonical anchors;
    groups containing U only use the median x of all eligible members.  Row
    extents are retained for honest closure and guard-band diagnostics, but
    overlapping extents are never transitively merged (that percolates across
    the recording and destroys the preregistered anchor-balanced estimand).
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    canonical_id_to_groups: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_ids:
            raise LearningContractError(f"duplicate fold candidate ID: {candidate_id}")
        seen_ids.add(candidate_id)
        leakage = str(row["leakage_group_id"])
        grouped[leakage].append(row)
        if row.get("eligible_for_anchor_balance"):
            canonical_id = row.get("primary_anchor_canonical_roi_id")
            if canonical_id in (None, ""):
                raise LearningContractError("positive anchor lacks canonical identity")
            canonical_id_to_groups[str(canonical_id)].add(leakage)
    split = sorted(identity for identity, groups in canonical_id_to_groups.items() if len(groups) > 1)
    if split:
        raise LearningContractError("canonical identity spans leakage groups: " + ", ".join(split))

    units = []
    for leakage, members in grouped.items():
        xs = [float(row["x_px"]) for row in members]
        if any(not math.isfinite(value) for value in xs):
            raise LearningContractError("fold coordinates must be finite")
        positive_members = [row for row in members if row.get("eligible_for_anchor_balance")]
        positive_x = [float(row["x_px"]) for row in positive_members]
        representative = float(np.median(positive_x if positive_x else xs))
        units.append(
            {
                "leakage_group_id": leakage,
                "members": list(members),
                "minimum_x_px": min(xs),
                "maximum_x_px": max(xs),
                "representative_x_px": representative,
                "representative_semantics": (
                    "median_unreserved_positive_anchor_x"
                    if positive_members
                    else "median_all_eligible_member_x"
                ),
                "positive_anchor_count": len(positive_members),
                "positive_identity_ids": {
                    str(row["primary_anchor_canonical_roi_id"])
                    for row in positive_members
                },
                "unlabeled_spatial_groups": {
                    str(row["spatial_group_id"])
                    for row in members
                    if row["training_state"] == "unlabeled"
                },
            }
        )
    return sorted(units, key=lambda item: (item["representative_x_px"], item["leakage_group_id"]))


def _balanced_contiguous_partition(components: Sequence[Mapping[str, Any]], fold_count: int) -> tuple[int, ...]:
    if len(components) < fold_count:
        raise LearningContractError("fewer indivisible x components than outer folds")
    positive_counts = [int(item["positive_anchor_count"]) for item in components]
    if sum(value > 0 for value in positive_counts) < fold_count:
        raise LearningContractError("positive identities cannot support all contiguous folds")
    prefix = [0]
    for value in positive_counts:
        prefix.append(prefix[-1] + value)
    target = prefix[-1] / fold_count
    # DP state is (cost, cut tuple); tuple tie-breaking makes the result stable.
    previous: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    n = len(components)
    for fold in range(1, fold_count + 1):
        current: dict[int, tuple[float, tuple[int, ...]]] = {}
        minimum_stop = fold
        maximum_stop = n - (fold_count - fold)
        for stop in range(minimum_stop, maximum_stop + 1):
            best: tuple[float, tuple[int, ...]] | None = None
            for start, (cost, cuts) in previous.items():
                if start >= stop:
                    continue
                segment_positives = prefix[stop] - prefix[start]
                if segment_positives < 1:
                    continue
                candidate = (cost + (segment_positives - target) ** 2, cuts + (stop,))
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                current[stop] = best
        previous = current
    if n not in previous:
        raise LearningContractError("no positive-supported contiguous fold partition exists")
    return previous[n][1]


def build_contiguous_x_block_folds(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    fold_count: int = 5,
    boundary_guard_px: float = 12.0,
    min_positive_anchors_per_fold: int = 10,
    min_unlabeled_spatial_groups_per_fold: int = 10,
) -> dict[str, Any]:
    """Freeze label-balanced contiguous x blocks and whole-group train purges.

    Only geometry, group identity, and positive-vs-unlabeled role are consumed.
    No feature value or model score is accepted by this function.
    """

    if fold_count < 2 or boundary_guard_px <= 0:
        raise LearningContractError("invalid outer-fold geometry contract")
    eligible = [dict(row) for row in fold_rows if bool(row.get("eligible_for_primary_fit"))]
    if not eligible:
        raise LearningContractError("primary fold census is empty")
    unexpected = sorted(
        {str(row.get("training_state")) for row in eligible} - {"positive", "unlabeled"}
    )
    if unexpected:
        raise LearningContractError(f"eligible fold rows have invalid states: {unexpected}")
    units = _leakage_group_units(eligible)
    positive_units = [unit for unit in units if unit["positive_anchor_count"] > 0]
    # Equal representative coordinates are one indivisible anchor-x unit so a
    # midpoint boundary can never split a spatial tie.
    anchor_units: list[dict[str, Any]] = []
    for unit in positive_units:
        if anchor_units and unit["representative_x_px"] == anchor_units[-1]["representative_x_px"]:
            anchor_units[-1]["positive_anchor_count"] += unit["positive_anchor_count"]
            anchor_units[-1]["leakage_group_ids"].append(unit["leakage_group_id"])
        else:
            anchor_units.append(
                {
                    "representative_x_px": unit["representative_x_px"],
                    "positive_anchor_count": unit["positive_anchor_count"],
                    "leakage_group_ids": [unit["leakage_group_id"]],
                }
            )
    cuts = _balanced_contiguous_partition(anchor_units, fold_count)
    positive_group_fold: dict[str, int] = {}
    boundary_rows: list[dict[str, Any]] = []
    start = 0
    for fold_id, stop in enumerate(cuts, start=1):
        for anchor_unit in anchor_units[start:stop]:
            for group in anchor_unit["leakage_group_ids"]:
                positive_group_fold[group] = fold_id
        if fold_id < fold_count:
            left_x = float(anchor_units[stop - 1]["representative_x_px"])
            right_x = float(anchor_units[stop]["representative_x_px"])
            if not left_x < right_x:
                raise LearningContractError("anchor-derived fold boundary is not strictly ordered")
            boundary_rows.append(
                {
                    "boundary_id": fold_id,
                    "left_fold_id": fold_id,
                    "right_fold_id": fold_id + 1,
                    "left_anchor_representative_x_px": left_x,
                    "right_anchor_representative_x_px": right_x,
                    "boundary_x_px": (left_x + right_x) / 2.0,
                    "derivation": "midpoint_between_adjacent_anchor_group_representatives",
                }
            )
        start = stop
    boundaries = [float(row["boundary_x_px"]) for row in boundary_rows]
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise LearningContractError("anchor-derived midpoint boundaries are not strictly increasing")

    group_fold: dict[str, int] = {}
    for unit in units:
        expected_bin = int(np.searchsorted(boundaries, unit["representative_x_px"], side="right")) + 1
        assigned = positive_group_fold.get(unit["leakage_group_id"], expected_bin)
        if assigned != expected_bin:
            raise LearningContractError("positive group assignment disagrees with its anchor-derived boundary bin")
        group_fold[unit["leakage_group_id"]] = assigned

    assignments: dict[str, int] = {}
    folds: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for fold_id in range(1, fold_count + 1):
        selected_units = [unit for unit in units if group_fold[unit["leakage_group_id"]] == fold_id]
        members = [row for unit in selected_units for row in unit["members"]]
        if not members:
            raise LearningContractError(f"outer fold {fold_id} is empty")
        for row in members:
            candidate_id = str(row["candidate_id"])
            if candidate_id in assignments:
                raise LearningContractError("fold assignment duplicated a candidate")
            assignments[candidate_id] = fold_id
        xs = [float(row["x_px"]) for row in members]
        positives = [row for row in members if row["training_state"] == "positive"]
        unlabeled = [row for row in members if row["training_state"] == "unlabeled"]
        positive_identity_ids = {str(row["primary_anchor_canonical_roi_id"]) for row in positives}
        unlabeled_groups = {str(row["spatial_group_id"]) for row in unlabeled}
        if len(positives) < min_positive_anchors_per_fold:
            raise LearningContractError(
                f"outer fold {fold_id} has {len(positives)} positive anchors; "
                f"requires {min_positive_anchors_per_fold}"
            )
        if len(unlabeled_groups) < min_unlabeled_spatial_groups_per_fold:
            raise LearningContractError(
                f"outer fold {fold_id} has {len(unlabeled_groups)} U spatial groups; "
                f"requires {min_unlabeled_spatial_groups_per_fold}"
            )
        nominal_low = None if fold_id == 1 else boundaries[fold_id - 2]
        nominal_high = None if fold_id == fold_count else boundaries[fold_id - 1]
        for unit in selected_units:
            left_extension = (
                0.0 if nominal_low is None else max(0.0, nominal_low - float(unit["minimum_x_px"]))
            )
            right_extension = (
                0.0 if nominal_high is None else max(0.0, float(unit["maximum_x_px"]) - nominal_high)
            )
            if left_extension > 0 or right_extension > 0:
                closure_rows.append(
                    {
                        "fold_id": fold_id,
                        "leakage_group_id": unit["leakage_group_id"],
                        "representative_x_px": unit["representative_x_px"],
                        "minimum_x_px": unit["minimum_x_px"],
                        "maximum_x_px": unit["maximum_x_px"],
                        "left_extension_px": left_extension,
                        "right_extension_px": right_extension,
                        "reason": "whole_group_closure_extension",
                    }
                )
        folds.append(
            {
                "fold_id": fold_id,
                "nominal_minimum_x_px": nominal_low,
                "nominal_maximum_x_px": nominal_high,
                "observed_minimum_x_px": min(xs),
                "observed_maximum_x_px": max(xs),
                "test_candidate_ids": sorted(str(row["candidate_id"]) for row in members),
                "test_candidate_count": len(members),
                "test_positive_anchor_count": len(positives),
                "test_positive_identity_count": len(positive_identity_ids),
                "test_unlabeled_count": len(unlabeled),
                "test_unlabeled_spatial_group_count": len(unlabeled_groups),
            }
        )
    if set(assignments) != {str(row["candidate_id"]) for row in eligible}:
        raise LearningContractError("outer folds do not cover every eligible candidate exactly once")
    eligible_by_group = {str(unit["leakage_group_id"]): list(unit["members"]) for unit in units}
    unit_by_group = {str(unit["leakage_group_id"]): unit for unit in units}
    all_ids = set(assignments)
    guard_records: list[dict[str, Any]] = []
    retained_minimum_distances: list[float] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_ids = set(fold["test_candidate_ids"])
        test_groups = {
            str(row["leakage_group_id"])
            for row in eligible
            if str(row["candidate_id"]) in test_ids
        }
        purge_groups: set[str] = set()
        guard_detail: dict[str, dict[str, Any]] = {}
        held_boundaries = [
            float(value)
            for value in (fold["nominal_minimum_x_px"], fold["nominal_maximum_x_px"])
            if value is not None
        ]
        test_xy = np.asarray(
            [(float(row["x_px"]), float(row["y_px"])) for row in eligible if str(row["candidate_id"]) in test_ids],
            dtype=np.float64,
        )
        for group, members in eligible_by_group.items():
            if group in test_groups:
                continue
            unit = unit_by_group[group]
            boundary_hit = any(
                float(unit["maximum_x_px"]) >= boundary - boundary_guard_px
                and float(unit["minimum_x_px"]) <= boundary + boundary_guard_px
                for boundary in held_boundaries
            )
            member_xy = np.asarray(
                [(float(row["x_px"]), float(row["y_px"])) for row in members],
                dtype=np.float64,
            )
            minimum_distance = float(
                np.sqrt(
                    np.min(
                        np.sum(
                            (member_xy[:, None, :] - test_xy[None, :, :]) ** 2,
                            axis=2,
                        )
                    )
                )
            )
            candidate_proximity_hit = minimum_distance <= boundary_guard_px
            guard_detail[group] = {
                "boundary_interval_guard_intersection": boundary_hit,
                "held_candidate_euclidean_guard_intersection": candidate_proximity_hit,
                "minimum_distance_to_held_test_candidate_px": minimum_distance,
            }
            if boundary_hit or candidate_proximity_hit:
                purge_groups.add(group)
        purge_ids = {
            str(row["candidate_id"])
            for group in purge_groups
            for row in eligible_by_group[group]
        }
        train_ids = sorted(all_ids - test_ids - purge_ids)
        train_rows = [row for row in eligible if str(row["candidate_id"]) in set(train_ids)]
        if {row["training_state"] for row in train_rows} != {"positive", "unlabeled"}:
            raise LearningContractError(f"outer fold {fold_id} training purge loses P/U support")
        if len({row["leakage_group_id"] for row in train_rows} & test_groups):
            raise LearningContractError(f"outer fold {fold_id} leaks a whole group")
        train_xy = np.asarray(
            [(float(row["x_px"]), float(row["y_px"])) for row in train_rows],
            dtype=np.float64,
        )
        retained_minimum_distance = float(
            np.sqrt(
                np.min(
                    np.sum((train_xy[:, None, :] - test_xy[None, :, :]) ** 2, axis=2)
                )
            )
        )
        if retained_minimum_distance <= boundary_guard_px:
            raise LearningContractError(
                f"outer fold {fold_id} retains a train/test candidate pair within the inclusive guard"
            )
        retained_minimum_distances.append(retained_minimum_distance)
        fold["train_candidate_ids"] = train_ids
        fold["guard_purged_candidate_ids"] = sorted(purge_ids)
        fold["guard_purged_leakage_group_ids"] = sorted(purge_groups)
        fold["train_candidate_count"] = len(train_ids)
        fold["guard_purged_candidate_count"] = len(purge_ids)
        fold["minimum_retained_train_test_candidate_distance_px"] = retained_minimum_distance
        fold["retained_train_test_pairs_within_inclusive_guard"] = 0
        for group in sorted(purge_groups):
            detail = guard_detail[group]
            reasons = []
            if detail["boundary_interval_guard_intersection"]:
                reasons.append("held_boundary_interval_guard")
            if detail["held_candidate_euclidean_guard_intersection"]:
                reasons.append("held_candidate_euclidean_guard")
            for row in sorted(eligible_by_group[group], key=lambda item: str(item["candidate_id"])):
                guard_records.append(
                    {
                        "fold_id": fold_id,
                        "leakage_group_id": group,
                        "candidate_id": str(row["candidate_id"]),
                        "x_px": float(row["x_px"]),
                        "y_px": float(row["y_px"]),
                        **detail,
                        "reason": "+".join(reasons),
                    }
                )
    result = {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": "five_contiguous_x_blocks_v1",
        "fold_count": fold_count,
        "boundary_guard_px": float(boundary_guard_px),
        "boundaries": boundary_rows,
        "assignments": dict(sorted(assignments.items())),
        "folds": folds,
        "guard_purge_rows": guard_records,
        "whole_group_closure_extensions": closure_rows,
        "assignment_sha256": _stable_hash(dict(sorted(assignments.items()))),
        "validation": {
            "passed": True,
            "eligible_candidate_count": len(eligible),
            "leakage_group_count": len(eligible_by_group),
            "positive_anchor_group_count": len(positive_units),
            "positive_anchor_x_unit_count": len(anchor_units),
            "boundary_count": len(boundary_rows),
            "all_test_rows_exactly_once": True,
            "identity_groups_whole": True,
            "leakage_groups_whole": True,
            "representative_x_boundary_bins_exact": True,
            "nominal_anchor_x_blocks_contiguous": True,
            "row_extent_overlap_allowed_only_by_whole_group_closure": True,
            "whole_group_closure_extension_count": len(closure_rows),
            "maximum_whole_group_closure_extension_px": max(
                (max(row["left_extension_px"], row["right_extension_px"]) for row in closure_rows),
                default=0.0,
            ),
            "whole_group_guard_purge": True,
            "guard_combines_boundary_interval_and_held_candidate_euclidean_rules": True,
            "zero_retained_train_test_candidate_pairs_within_inclusive_guard": True,
            "minimum_retained_train_test_candidate_distance_px": min(
                retained_minimum_distances
            ),
            "features_or_scores_used_for_boundaries": False,
        },
    }
    membership_contract = {
        str(fold["fold_id"]): {
            "test": fold["test_candidate_ids"],
            "train": fold["train_candidate_ids"],
            "purged": fold["guard_purged_candidate_ids"],
        }
        for fold in folds
    }
    result["train_test_guard_membership_sha256"] = _stable_hash(membership_contract)
    result["plan_sha256"] = _stable_hash(result)
    return result


def materialize_frozen_outer_folds(
    model_rows: Sequence[Mapping[str, Any]], fold_plan: Mapping[str, Any]
) -> tuple[tuple[list[int], list[int]], ...]:
    """Translate candidate IDs to the evaluator's original input-row indices."""

    index_by_id: dict[str, int] = {}
    for index, row in enumerate(model_rows):
        row_id = str(row["row_id"])
        if row_id in index_by_id:
            raise LearningContractError(f"duplicate model row ID: {row_id}")
        index_by_id[row_id] = index
    if set(index_by_id) != set(fold_plan["assignments"]):
        raise LearningContractError("fold plan and model rows have different candidate IDs")
    frozen = []
    covered: list[int] = []
    for fold in fold_plan["folds"]:
        train = sorted(index_by_id[value] for value in fold["train_candidate_ids"])
        test = sorted(index_by_id[value] for value in fold["test_candidate_ids"])
        if set(train) & set(test):
            raise LearningContractError("materialized frozen fold overlaps train/test")
        covered.extend(test)
        frozen.append((train, test))
    if sorted(covered) != list(range(len(model_rows))):
        raise LearningContractError("materialized test folds do not cover model rows exactly once")
    return tuple(frozen)


def adjust_recall_for_proposal_misses(
    metrics: Mapping[str, Any], proposal_stage_miss_count: int
) -> dict[str, Any]:
    """Add only *unreserved* confirmed proposal misses to recall denominator."""

    if not isinstance(proposal_stage_miss_count, int) or proposal_stage_miss_count < 0:
        raise LearningContractError("proposal_stage_miss_count must be a non-negative integer")
    try:
        matched = int(metrics["positive_count"])
        recovered = int(metrics["positive_recovered_at_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LearningContractError("rank metrics lack a valid positive denominator") from exc
    if matched < 1 or recovered < 0 or recovered > matched:
        raise LearningContractError("rank metrics have an invalid positive recovery count")
    denominator = matched + proposal_stage_miss_count
    output = dict(metrics)
    output.update(
        {
            "matched_positive_anchor_count": matched,
            "unreserved_proposal_stage_miss_count": proposal_stage_miss_count,
            "known_positive_proposal_denominator": denominator,
            "known_positive_proposal_recall_at_budget": recovered / denominator,
            "denominator_policy": "matched_unreserved_anchors_plus_unreserved_confirmed_proposal_misses",
        }
    )
    return output


def grouped_paired_bootstrap_macro_fold_auc(
    positive_mask: Sequence[bool],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    fold_ids: Sequence[int],
    leakage_groups: Sequence[Any],
    *,
    draws: int,
    seed: int,
    score_a_name: str = "score_a",
    score_b_name: str = "score_b",
) -> dict[str, Any]:
    """Bootstrap whole dependency components for macro within-fold SPU-AUC.

    Leakage components are stratified by whether they contain any positive
    anchor.  Every sampled component contributes all of its P and U rows
    jointly, so no canonical identity, spatial cell, or mixed component is
    split during resampling.
    """

    positive = np.asarray(positive_mask, dtype=bool)
    left = np.asarray(scores_a, dtype=np.float64)
    right = np.asarray(scores_b, dtype=np.float64)
    folds = np.asarray(fold_ids, dtype=int)
    groups = np.asarray([str(value) for value in leakage_groups], dtype=object)
    shape = positive.shape
    if (
        positive.ndim != 1
        or left.shape != shape
        or right.shape != shape
        or folds.shape != shape
        or groups.shape != shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or draws < 1
    ):
        raise LearningContractError("macro-fold bootstrap inputs must be finite aligned vectors")
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if unique_folds != list(range(1, len(unique_folds) + 1)) or len(unique_folds) < 2:
        raise LearningContractError("macro-fold bootstrap fold IDs must be contiguous from one")
    observed_by_fold = []
    units: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_id in unique_folds:
        in_fold = folds == fold_id
        fold_groups = np.unique(groups[in_fold])
        positive_components = np.asarray(
            [group for group in fold_groups if np.any(positive[in_fold & (groups == group)])],
            dtype=object,
        )
        unlabeled_only_components = np.asarray(
            [group for group in fold_groups if not np.any(positive[in_fold & (groups == group)])],
            dtype=object,
        )
        if len(positive_components) < 2 or len(unlabeled_only_components) < 2:
            raise LearningContractError(f"fold {fold_id} lacks grouped bootstrap support")
        if any(len(np.unique(folds[groups == group])) != 1 for group in fold_groups):
            raise LearningContractError(f"fold {fold_id} contains a cross-fold leakage component")
        units[fold_id] = (positive_components, unlabeled_only_components)
        observed_by_fold.append(
            {
                "fold_id": fold_id,
                "positive_containing_leakage_component_count": len(positive_components),
                "unlabeled_only_leakage_component_count": len(unlabeled_only_components),
                "unlabeled_rows_inside_positive_containing_components": int(
                    np.sum(
                        in_fold
                        & ~positive
                        & np.isin(groups, positive_components)
                    )
                ),
                "unlabeled_rows_inside_unlabeled_only_components": int(
                    np.sum(
                        in_fold
                        & ~positive
                        & np.isin(groups, unlabeled_only_components)
                    )
                ),
                "score_a": model_core.positive_vs_unlabeled_rank_auc(positive[in_fold], left[in_fold]),
                "score_b": model_core.positive_vs_unlabeled_rank_auc(positive[in_fold], right[in_fold]),
            }
        )
    observed_a = float(np.mean([row["score_a"] for row in observed_by_fold]))
    observed_b = float(np.mean([row["score_b"] for row in observed_by_fold]))
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        fold_deltas = []
        for fold_id in unique_folds:
            positive_components, unlabeled_only_components = units[fold_id]
            sampled_positive = rng.choice(
                positive_components, size=len(positive_components), replace=True
            )
            sampled_unlabeled = rng.choice(
                unlabeled_only_components,
                size=len(unlabeled_only_components),
                replace=True,
            )
            indices: list[int] = []
            for group in (*sampled_positive, *sampled_unlabeled):
                group_indices = np.flatnonzero((folds == fold_id) & (groups == group))
                indices.extend(int(index) for index in group_indices)
            sampled_indices = np.asarray(indices, dtype=int)
            sampled_labels = positive[sampled_indices]
            fold_deltas.append(
                model_core.positive_vs_unlabeled_rank_auc(sampled_labels, left[sampled_indices])
                - model_core.positive_vs_unlabeled_rank_auc(sampled_labels, right[sampled_indices])
            )
        deltas[draw] = float(np.mean(fold_deltas))
    return {
        "estimand": "paired delta in macro mean of within-fold positive-vs-unlabeled rank AUC",
        "score_a_name": score_a_name,
        "score_b_name": score_b_name,
        "fold_count": len(unique_folds),
        "observed_score_a_macro_fold_spu_auc": observed_a,
        "observed_score_b_macro_fold_spu_auc": observed_b,
        "observed_delta_a_minus_b": observed_a - observed_b,
        "observed_by_fold": observed_by_fold,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "whole_leakage_group_with_all_positive_and_unlabeled_members_joint",
        "resampling_strata": "positive_containing_vs_unlabeled_only_within_outer_fold",
        "dependency_components_split": False,
        "bootstrap_delta_median": float(np.median(deltas)),
        "bootstrap_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "bootstrap_delta_ci95_high": float(np.quantile(deltas, 0.975)),
    }


def fold_stratified_representative_unit_score_permutation_null(
    positive_mask: Sequence[bool],
    scores: Sequence[float],
    sample_ids: Sequence[Any],
    fold_ids: Sequence[int],
    leakage_groups: Sequence[Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Posthoc score-assignment null with one row per dependency component.

    A positive-containing leakage component contributes its smallest-ID
    positive anchor.  A U-only leakage component contributes its smallest-ID
    candidate.  Fixed representative scores are reassigned only within the
    same outer fold, so neither component size nor a split dependency graph can
    generate an association.
    """

    positive = np.asarray(positive_mask, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray([str(value) for value in sample_ids], dtype=object)
    folds = np.asarray(fold_ids, dtype=int)
    groups = np.asarray([str(value) for value in leakage_groups], dtype=object)
    shape = positive.shape
    if (
        positive.ndim != 1
        or values.shape != shape
        or ids.shape != shape
        or folds.shape != shape
        or groups.shape != shape
        or len(set(ids.tolist())) != len(ids)
        or np.any(~np.isfinite(values))
        or draws < 1
    ):
        raise LearningContractError("representative-unit null inputs must be finite aligned vectors")
    units: list[dict[str, Any]] = []
    for group in sorted(np.unique(groups)):
        component_indices = np.flatnonzero(groups == group)
        group_folds = np.unique(folds[component_indices])
        if len(group_folds) != 1:
            raise LearningContractError(f"leakage component crosses outer folds: {group}")
        positive_indices = component_indices[positive[component_indices]]
        is_positive_component = len(positive_indices) > 0
        eligible_representatives = (
            positive_indices if is_positive_component else component_indices
        )
        representative = min(
            eligible_representatives, key=lambda index: str(ids[index])
        )
        unit_type = (
            "positive_containing_leakage_component"
            if is_positive_component
            else "unlabeled_only_leakage_component"
        )
        units.append(
            {
                "unit_id": f"{unit_type}::{group}",
                "unit_type": unit_type,
                "source_group_id": str(group),
                "outer_fold": int(group_folds[0]),
                "representative_candidate_id": str(ids[representative]),
                "representative_input_index": int(representative),
                "source_group_candidate_count": int(len(component_indices)),
                "source_group_positive_candidate_count": int(len(positive_indices)),
                "source_group_unlabeled_candidate_count": int(
                    len(component_indices) - len(positive_indices)
                ),
                "is_positive_unit": is_positive_component,
                "score": float(values[representative]),
            }
        )
    units.sort(key=lambda row: (row["outer_fold"], row["unit_type"], row["unit_id"]))
    unit_scores = np.asarray([row["score"] for row in units], dtype=np.float64)
    unit_labels = np.asarray([row["is_positive_unit"] for row in units], dtype=bool)
    unit_folds = np.asarray([row["outer_fold"] for row in units], dtype=int)
    unique_folds = sorted(int(value) for value in np.unique(unit_folds))
    observed_by_fold = []
    for fold_id in unique_folds:
        selected = unit_folds == fold_id
        if set(np.unique(unit_labels[selected])) != {False, True}:
            raise LearningContractError(f"representative null fold {fold_id} lacks P/U support")
        observed_by_fold.append(
            {
                "fold_id": fold_id,
                "positive_containing_leakage_component_count": int(
                    np.sum(unit_labels[selected])
                ),
                "unlabeled_only_leakage_component_count": int(
                    np.sum(~unit_labels[selected])
                ),
                "positive_vs_unlabeled_rank_auc": model_core.positive_vs_unlabeled_rank_auc(
                    unit_labels[selected], unit_scores[selected]
                ),
            }
        )
    observed = float(np.mean([row["positive_vs_unlabeled_rank_auc"] for row in observed_by_fold]))
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        fold_values = []
        for fold_id in unique_folds:
            selected = unit_folds == fold_id
            permuted_scores = rng.permutation(unit_scores[selected])
            fold_values.append(
                model_core.positive_vs_unlabeled_rank_auc(
                    unit_labels[selected], permuted_scores
                )
            )
        null[draw] = float(np.mean(fold_values))
    return {
        "null_semantics": "one_pre_score_representative_per_whole_leakage_group_then_fold_stratified_score_assignment_permutation_not_model_refit",
        "observed_macro_fold_representative_unit_spu_auc": observed,
        "observed_by_fold": observed_by_fold,
        "draws": int(draws),
        "seed": int(seed),
        "positive_unit": "positive_containing_leakage_group_min_candidate_id_positive_anchor",
        "unlabeled_unit": "unlabeled_only_leakage_group_min_candidate_id",
        "positive_containing_leakage_component_count": int(np.sum(unit_labels)),
        "unlabeled_only_leakage_component_count": int(np.sum(~unit_labels)),
        "greater_or_equal_p_value": float((1 + np.sum(null >= observed)) / (draws + 1)),
        "null_median": float(np.median(null)),
        "null_ci95_low": float(np.quantile(null, 0.025)),
        "null_ci95_high": float(np.quantile(null, 0.975)),
        "group_size_order_statistic_used": False,
        "dependency_components_split": False,
        "representatives": units,
    }


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def classify_proposal_stage_misses(
    canonical_rows: Sequence[Mapping[str, Any]],
    harmonized_rows: Sequence[Mapping[str, Any]],
    review_site_rows: Sequence[Mapping[str, Any]],
    *,
    review_reserve_radius_px: float = 6.0,
    candidate_match_radius_px: float = 6.0,
) -> dict[str, Any]:
    """Separate unmatched confirmed labels into unreserved and review-reserved.

    The reserve check is performed independently of proposal matching, so a
    confirmed occurrence with no candidate anchor cannot accidentally re-enter
    the primary denominator merely because no harmonized row carries its ID.
    """

    matched_ids = {
        str(row.get("canonical_v7", {}).get("observation_id"))
        for row in harmonized_rows
        if row.get("canonical_v7", {}).get("observation_id") not in (None, "")
    }
    review_sites = []
    for row in review_site_rows:
        review_sites.append(
            (str(row.get("detection_site_id", row.get("review_site_id", ""))), float(row["x_px"]), float(row["y_px"]))
        )
    misses = []
    for row in canonical_rows:
        if not _truth(row.get("include_confirmed")):
            continue
        observation_id = str(row["observation_id"])
        if observation_id in matched_ids:
            continue
        x = float(row["x_px"])
        y = float(row["y_px"])
        nearest = min(
            ((math.hypot(x - sx, y - sy), site_id) for site_id, sx, sy in review_sites),
            default=(math.inf, None),
        )
        nearest_candidate = min(
            (
                (
                    math.hypot(
                        x - float(candidate["x_px"]),
                        y - float(candidate["y_px"]),
                    ),
                    str(candidate["candidate_id"]),
                )
                for candidate in harmonized_rows
                if candidate.get("partition") == "event"
                and int(candidate.get("partition_id")) == int(row["burst_id"])
            ),
            default=(math.inf, None),
        )
        reserved = nearest[0] <= review_reserve_radius_px
        misses.append(
            {
                "observation_id": observation_id,
                "burst_id": int(row["burst_id"]),
                "canonical_roi_id": str(row["canonical_roi_id"]),
                "x_px": x,
                "y_px": y,
                "inside_locked_review_reserve": reserved,
                "nearest_review_site_id": nearest[1],
                "nearest_review_distance_px": None if not math.isfinite(nearest[0]) else nearest[0],
                "nearest_frozen_candidate_id": nearest_candidate[1],
                "nearest_frozen_candidate_distance_px": (
                    None if not math.isfinite(nearest_candidate[0]) else nearest_candidate[0]
                ),
                "no_candidate_within_match_radius": nearest_candidate[0] > candidate_match_radius_px,
                "candidate_match_radius_px": candidate_match_radius_px,
                "primary_denominator_role": "excluded_review_reserve" if reserved else "unrecovered_unreserved_positive",
            }
        )
    return {
        "rows": sorted(misses, key=lambda item: (item["burst_id"], item["observation_id"])),
        "total_unmatched_confirmed": len(misses),
        "reserved_unmatched_confirmed": sum(row["inside_locked_review_reserve"] for row in misses),
        "unreserved_unmatched_confirmed": sum(not row["inside_locked_review_reserve"] for row in misses),
    }


def _assign_miss_folds(miss_rows: Sequence[Mapping[str, Any]], fold_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for raw in miss_rows:
        row = dict(raw)
        x = float(row["x_px"])
        boundaries = [float(item["boundary_x_px"]) for item in fold_plan["boundaries"]]
        row["assigned_outer_fold"] = int(np.searchsorted(boundaries, x, side="right")) + 1
        output.append(row)
    return output


def _evaluation_config(config: Mapping[str, Any]) -> model_core.EvaluationConfig:
    models = config["models"]
    evaluation = config["evaluation"]
    return model_core.EvaluationConfig(
        outer_splits=int(config["outer_validation"]["fold_count"]),
        inner_splits=int(models["tiny_mlp_spu"]["training_only_epoch_selection_folds"]),
        candidate_budget=int(evaluation["primary_budget"]),
        seed=int(models["bagged_pu_linear"]["seed"]),
        hyperparameter_selection="fixed",
        logistic_penalties=(float(models["linear_spu"]["l2"]),),
        elastic_penalties=(float(models["elastic_linear_spu"]["penalty_strength"]),),
        pu_penalties=(float(models["bagged_pu_linear"]["base_l2"]),),
        mlp_l2_values=(float(models["tiny_mlp_spu"]["l2"]),),
        mlp_seeds=tuple(int(value) for value in models["tiny_mlp_spu"]["seeds"]),
        pu_bags=int(models["bagged_pu_linear"]["bags"]),
        pu_unlabeled_ratio=float(models["bagged_pu_linear"]["unlabeled_per_positive"]),
        max_logistic_iterations=int(models["linear_spu"]["maximum_proximal_gradient_iterations"]),
        logistic_tolerance=float(models["linear_spu"]["convergence_tolerance"]),
        mlp_max_epochs=int(models["tiny_mlp_spu"]["maximum_epochs"]),
        mlp_patience=int(models["tiny_mlp_spu"]["training_only_early_stop_patience"]),
        mlp_learning_rate=float(models["tiny_mlp_spu"]["learning_rate"]),
        # Zero explicitly disables the rejected group-maximum null in the
        # frozen core; only the dependency-safe runner null is computed.
        permutation_draws=0,
        preprocess_clip=10.0,
    )


def _matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    model_core.validate_raw_feature_names(feature_names)
    if any(name not in PRIMARY_FEATURES for name in feature_names):
        raise LearningContractError("model matrix contains a feature outside the frozen compact set")
    values = np.empty((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, feature in enumerate(feature_names):
            value = row.get(feature)
            values[row_index, column_index] = np.nan if value is None else float(value)
    return values


def _method_results(evaluation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    methods: dict[str, Mapping[str, Any]] = {}
    for name, payload in evaluation["baselines"].items():
        methods[str(name)] = payload
    for internal, public in MODEL_RESULT_NAMES.items():
        methods[public] = evaluation["models"][internal]
    if set(methods) != set(EXPECTED_METHODS):
        raise LearningContractError("evaluator result methods differ from the frozen method set")
    return methods


def _method_score_vector(
    payload: Mapping[str, Any],
    *,
    raw_fold_specific: bool = False,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return one evaluator score vector across learned and scalar schemas.

    Learned methods expose ``oof_scores`` while frozen external scalar
    baselines expose ``scores``.  Keeping that schema distinction in one
    fail-closed accessor prevents downstream consumers from silently assuming
    that every method was fitted by the cross-validation evaluator.
    """

    normalized_aliases = ("oof_scores", "scores")
    normalized_present = [key for key in normalized_aliases if key in payload]
    if len(normalized_present) != 1:
        raise LearningContractError(
            "method payload must expose exactly one canonical "
            "fold-percentile score alias"
        )
    normalized_alias = normalized_present[0]
    if raw_fold_specific:
        raw_aliases = ("oof_scores_raw_fold_specific", "scores_raw_fold_specific")
        raw_present = [key for key in raw_aliases if key in payload]
        expected_raw_alias = {
            "oof_scores": "oof_scores_raw_fold_specific",
            "scores": "scores_raw_fold_specific",
        }[normalized_alias]
        if raw_present != [expected_raw_alias]:
            raise LearningContractError(
                "method payload raw score alias differs from normalized score schema"
            )
        score_key = expected_raw_alias
    else:
        score_key = normalized_alias
    scores = np.asarray(payload[score_key], dtype=np.float64)
    if scores.ndim != 1:
        raise LearningContractError("method score vector must be one-dimensional")
    if expected_length is not None and len(scores) != int(expected_length):
        raise LearningContractError("method score vector length differs from candidate count")
    if np.any(~np.isfinite(scores)):
        raise LearningContractError("method score vector contains non-finite values")
    if not raw_fold_specific:
        canonical_key = "oof_scores_fold_percentile"
        if canonical_key not in payload:
            raise LearningContractError(
                "method payload is missing canonical fold-percentile scores"
            )
        canonical = np.asarray(payload[canonical_key], dtype=np.float64)
        if canonical.ndim != 1 or canonical.shape != scores.shape:
            raise LearningContractError(
                "canonical fold-percentile scores differ in shape from schema alias"
            )
        if np.any(~np.isfinite(canonical)):
            raise LearningContractError(
                "canonical fold-percentile scores contain non-finite values"
            )
        if not np.array_equal(canonical, scores):
            raise LearningContractError(
                "canonical fold-percentile scores differ from schema alias"
            )
        scores = canonical
    return scores


def _evaluate_budgets(
    methods: Mapping[str, Mapping[str, Any]],
    positive_mask: np.ndarray,
    sample_ids: Sequence[str],
    budgets: Sequence[int],
    unreserved_misses: Sequence[Mapping[str, Any]],
    fold_plan: Mapping[str, Any],
) -> dict[str, Any]:
    miss_rows = _assign_miss_folds(unreserved_misses, fold_plan)
    misses_by_fold = defaultdict(int)
    for row in miss_rows:
        misses_by_fold[int(row["assigned_outer_fold"])] += 1
    aggregate = {}
    per_fold = []
    fold_indices = {
        int(fold["fold_id"]): np.asarray(
            [index for index, candidate_id in enumerate(sample_ids) if candidate_id in set(fold["test_candidate_ids"])],
            dtype=int,
        )
        for fold in fold_plan["folds"]
    }
    for method_name, result in methods.items():
        scores = _method_score_vector(result, expected_length=len(sample_ids))
        aggregate[method_name] = {}
        for budget in budgets:
            metrics = model_core.rank_metrics(positive_mask, scores, budget=int(budget), tie_breakers=sample_ids)
            aggregate[method_name][str(budget)] = adjust_recall_for_proposal_misses(metrics, len(unreserved_misses))
        for fold_id, indices in fold_indices.items():
            fold_budget = 58
            metrics = model_core.rank_metrics(
                positive_mask[indices], scores[indices], budget=fold_budget,
                tie_breakers=[sample_ids[index] for index in indices],
            )
            per_fold.append(
                {
                    "method": method_name,
                    "fold_id": fold_id,
                    "per_fold_budget_semantics": "literal_K_equals_min_58_and_test_fold_candidate_count",
                    **adjust_recall_for_proposal_misses(metrics, misses_by_fold[fold_id]),
                }
            )
    return {"aggregate": aggregate, "per_fold": per_fold, "proposal_misses": miss_rows}


def _global_budget_metric_output(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Label the global fold-percentile AUC as sensitivity, never primary."""

    output = dict(metrics)
    output["global_fold_percentile_spu_auc_sensitivity"] = output.pop(
        "positive_vs_unlabeled_rank_auc"
    )
    output["metric_scope"] = (
        "global_fixed_budget_over_label_free_within_test_fold_percentile_scores"
    )
    output["primary_macro_fold_spu_auc_in_this_record"] = False
    return output


def _score_locked_audits(
    harmonized: data_core.HarmonizedInnovationCensus,
    model_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    evaluation_config: model_core.EvaluationConfig,
) -> dict[str, Any]:
    full_by_id = {str(row["candidate_id"]): row for row in harmonized.rows}
    audit_source = [
        row for row in harmonized.all_feature_rows()
        if str(row["training_state"]).startswith("heldout_review")
    ]
    quiet_source = list(harmonized.null_control_feature_rows())
    combined = list(model_rows) + audit_source + quiet_source
    raw = _matrix(combined, feature_names)
    train = np.arange(len(model_rows), dtype=int)
    test = np.arange(len(model_rows), len(combined), dtype=int)
    y = np.asarray([row["training_state"] == "positive" for row in combined], dtype=np.int8)
    groups = np.asarray([str(row["leakage_group_id"]) for row in combined], dtype=object)
    scores, seed_scores, epochs = model_core._fit_mlp_ensemble(  # noqa: SLF001 - frozen internal core
        raw, y, groups, train, test, feature_names,
        l2=float(evaluation_config.mlp_l2_values[0]),
        config=evaluation_config,
        seed_offset=77_000_021,
    )
    split = len(audit_source)
    review_rows = [
        {
            "candidate_id": row["row_id"],
            "partition_id": row["partition_id"],
            "training_state": row["training_state"],
            "review_site_id": full_by_id[str(row["row_id"])]["review_holdout"]["nearest_review_site_id"],
            "review_label": full_by_id[str(row["row_id"])]["review_holdout"]["review_label"],
            "canonical_anchor_status": full_by_id[str(row["row_id"])]["canonical_v7"]["anchor_status"],
            "inside_confirmed_exclusion_radius": full_by_id[str(row["row_id"])]["canonical_v7"]["inside_confirmed_exclusion_radius"],
            "tiny_mlp_final_fit_score": float(scores[index]),
            "audit_role": "locked_review_conflict_panel_descriptive_only",
        }
        for index, row in enumerate(audit_source)
    ]
    quiet_rows = [
        {
            "candidate_id": row["row_id"],
            "partition_id": row["partition_id"],
            "training_state": row["training_state"],
            "tiny_mlp_final_fit_score": float(scores[split + index]),
            "audit_role": "source_off_null_control_descriptive_only",
        }
        for index, row in enumerate(quiet_source)
    ]
    return {
        "review_rows": review_rows,
        "quiet_rows": quiet_rows,
        "final_fit": {
            "training_candidate_count": len(model_rows),
            "review_candidate_count": len(review_rows),
            "quiet_candidate_count": len(quiet_rows),
            "seed_count": len(seed_scores),
            "training_only_epoch_selection": epochs,
            "panel_values_used_for_selection_or_tuning": False,
        },
    }


def aggregate_review_site_audit(
    review_site_rows: Sequence[Mapping[str, Any]],
    review_occurrence_rows: Sequence[Mapping[str, Any]],
    new_review_rows: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
    candidate_score_rows: Sequence[Mapping[str, Any]],
    *,
    match_radius_px: float = 6.0,
) -> list[dict[str, Any]]:
    """Aggregate locked candidate scores and label overlap to reviewed sites."""

    review_by_site = {str(row["detection_site_id"]): row for row in new_review_rows}
    occurrences_by_site: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in review_occurrence_rows:
        occurrences_by_site[str(row["detection_site_id"])].append(row)
    scores_by_site: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_score_rows:
        site_id = row.get("review_site_id")
        if site_id not in (None, ""):
            scores_by_site[str(site_id)].append(row)
    output = []
    for site in sorted(review_site_rows, key=lambda row: str(row["detection_site_id"])):
        site_id = str(site["detection_site_id"])
        if site_id not in review_by_site:
            continue
        review = review_by_site[site_id]
        matched_observations: set[str] = set()
        matched_confirmed: set[str] = set()
        for occurrence in occurrences_by_site.get(site_id, ()):
            burst_id = int(occurrence["burst_id"])
            x, y = float(occurrence["x_px"]), float(occurrence["y_px"])
            for canonical in canonical_rows:
                if int(canonical["burst_id"]) != burst_id:
                    continue
                if math.hypot(x - float(canonical["x_px"]), y - float(canonical["y_px"])) <= match_radius_px:
                    observation_id = str(canonical["observation_id"])
                    matched_observations.add(observation_id)
                    if _truth(canonical.get("include_confirmed")):
                        matched_confirmed.add(observation_id)
        candidate_rows = scores_by_site.get(site_id, [])
        candidate_scores = [float(row["tiny_mlp_final_fit_score"]) for row in candidate_rows]
        normalized_label = str(review["normalized_label"])
        conflict = bool(matched_confirmed) and normalized_label in {"uncertain", "artifact_or_noise"}
        output.append(
            {
                "review_site_id": site_id,
                "normalized_label": normalized_label,
                "confidence_1_to_5": int(review["confidence_1_to_5"]),
                "x_px": float(site["x_px"]),
                "y_px": float(site["y_px"]),
                "review_occurrence_count": len(occurrences_by_site.get(site_id, ())),
                "canonical_v7_any_overlap": bool(matched_observations),
                "canonical_v7_confirmed_overlap": bool(matched_confirmed),
                "canonical_v7_matched_observation_count": len(matched_observations),
                "canonical_v7_confirmed_observation_count": len(matched_confirmed),
                "cross_source_disagreement": conflict,
                "reserved_candidate_count": len(candidate_rows),
                "reserved_candidate_score_max": max(candidate_scores) if candidate_scores else None,
                "reserved_candidate_score_median": float(np.median(candidate_scores)) if candidate_scores else None,
                "panel_role": "locked_18_site_review_conflict_audit_descriptive_only",
                "panel_used_for_model_selection_or_tuning": False,
            }
        )
    if len(output) != len(review_by_site):
        raise LearningContractError("review site/label join is incomplete")
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = tuple(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        raise LearningContractError(f"cannot write headerless TSV: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) if row.get(key) is not None else "" for key in columns})
    temporary.replace(path)


def _flatten_harmonized(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        flat = {
            "candidate_id": row["candidate_id"],
            "partition": row["partition"],
            "partition_id": row["partition_id"],
            "x_px": row["x_px"],
            "y_px": row["y_px"],
            "source_count": row["source_count"],
            "training_state": row["training_state"],
            "identity_group_id": row["identity_group_id"],
            "spatial_group_id": row["spatial_group_id"],
            "leakage_group_id": row["leakage_group_id"],
            "canonical_observation_id": row["canonical_v7"]["observation_id"],
            "canonical_roi_id": row["canonical_v7"]["canonical_roi_id"],
            "canonical_match_distance_px": row["canonical_v7"]["match_distance_px"],
            "review_reserved": row["review_holdout"]["reserved"],
            "nearest_review_site_id": row["review_holdout"]["nearest_review_site_id"],
            "known_negative": False,
        }
        flat.update({f"feature__{name}": row["features"][name] for name in feature_names})
        output.append(flat)
    return output


def _artifact_index(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_index.json"):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "index_scope": "complete_run_tree_excluding_artifact_index_itself",
        "artifact_count": len(records),
        "artifacts": records,
    }


def write_and_validate_artifact_index(root: Path) -> dict[str, Any]:
    """Write the final tree index and immediately verify every indexed byte."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise LearningContractError("artifact root does not exist")
    index = _artifact_index(root)
    _write_json(root / "artifact_index.json", index)
    loaded = _read_json(root / "artifact_index.json")
    observed = _artifact_index(root)
    if loaded != observed:
        raise LearningContractError("artifact index does not match the complete run tree")
    return loaded


def finalize_atomic_run_tree(
    partial: Path,
    output: Path,
    *,
    before_atomic_rename: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fully validate the partial tree, then make one failure-free rename."""

    partial = Path(partial).resolve()
    output = Path(output).resolve()
    if not partial.is_dir() or output.exists():
        raise FileExistsError("atomic finalization requires one partial tree and no final tree")
    if partial.parent != output.parent or partial.name != output.name + ".partial":
        raise LearningContractError("partial/final paths do not satisfy the atomic naming contract")
    # Every schema/file-set/hash check that can fail happens while the tree is
    # still recoverably partial.  The only operation after this returns is the
    # same-filesystem rename itself.
    index = write_and_validate_artifact_index(partial)
    if before_atomic_rename is not None:
        before_atomic_rename(index)
    partial.replace(output)
    return index


def _progress(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _report(summary: Mapping[str, Any]) -> str:
    auc = summary["primary_macro_fold_spu_auc_by_method"]
    budget = summary["primary_budget_58_metrics_by_method"]
    contrasts = summary["tiny_mlp_cluster_bootstrap_contrasts"]
    method_rows = "\n".join(
        f"| `{method}` | {float(auc[method]):.6f} | "
        f"{float(budget[method]['known_positive_proposal_recall_at_budget']):.6f} |"
        for method in EXPECTED_METHODS
    )
    contrast_rows = "\n".join(
        f"| tiny MLP - `{comparator}` | "
        f"{float(record['tiny_mlp_minus_comparator_observed_delta']):.6f} | "
        f"[{float(record['cluster_bootstrap_ci95_low']):.6f}, "
        f"{float(record['cluster_bootstrap_ci95_high']):.6f}] |"
        for comparator, record in contrasts.items()
    )
    return (
        "# NREV-EXP-0021 bounded feature-learning screen\n\n"
        f"Run `{summary['run_id']}` completed the engineering screen. The descriptive "
        f"advance verdict is **{summary['advance_verdict'].replace('_', ' ')}**; the "
        "scientific audit is still incomplete and claim promotion remains false.\n\n"
        "## Answer-first results\n\n"
        "The primary metric is the macro mean of within-held-fold positive-versus-unlabeled "
        "rank AUC. Recall is reported at the global top-58 operating point after label-free "
        "within-fold percentile normalization.\n\n"
        "| Method | Primary macro-fold SPU-AUC | Known-positive proposal recall @58 |\n"
        "|---|---:|---:|\n"
        f"{method_rows}\n\n"
        "Whole-leakage-component paired bootstrap contrasts:\n\n"
        "| Prespecified contrast | AUC delta | 95% cluster CI |\n"
        "|---|---:|---:|\n"
        f"{contrast_rows}\n\n"
        "Positive-containing leakage-component support by fold was 9/3/3/2/4. "
        "Because some folds contain only two or three positive components, these 95% "
        "cluster-bootstrap intervals are descriptive and non-claim-bearing.\n\n"
        f"The observed strongest frozen baseline at budget 58 was "
        f"`{summary['observed_strongest_frozen_baseline_at_budget_58']}`; the MLP recall "
        f"delta was {float(summary['tiny_mlp_minus_strongest_frozen_recall_at_58']):.6f}. "
        "That observed-baseline selection was not repeated inside bootstrap draws, so its "
        "corresponding prespecified baseline CI is descriptive. The MLP representative-"
        f"component null p-value was {float(summary['tiny_mlp_representative_component_null_p']):.6g}; "
        f"median seed rank correlation was {float(summary['tiny_mlp_seed_median_pairwise_spearman']):.6f}, "
        f"and the seed recall@58 range was {float(summary['tiny_mlp_seed_recall_at_58_range']):.6f}.\n\n"
        "## Cohort and denominator\n\n"
        f"Primary cohort: {summary['primary_positive_anchors']} matched unreserved positive "
        f"anchors and {summary['primary_unlabeled_candidates']} unknown event candidates. "
        f"Unreserved proposal-stage misses: {summary['unreserved_proposal_stage_misses']}; "
        f"matched review-reserved anchors: {summary['matched_review_reserved_positive_anchors']}; "
        f"review-reserved unmatched confirmed occurrences: {summary['reserved_proposal_stage_misses']}.\n\n"
        "## Applicability and conventions\n\n"
        f"Stage: `{summary['stage']}`. Applicability: `{summary['applicability']}`. This is "
        "model-only spatial holdout within one recording, conditional on legacy label-derived "
        "event windows and a historically label-informed feature program. U means unknown, "
        "not negative. N/A is used when a denominator or audit is outside this bounded screen. "
        "The CFAR arm reranks the same broad union and `cfar_score` is also a fusion input; "
        "this is not end-to-end detector replacement or independent-recording generalization.\n\n"
        "## Inspect next\n\n"
        "Start with `summary.json`, `tables/aggregate_metrics.tsv`, "
        "`tables/per_fold_metrics.tsv`, `bootstrap.json`, `advance_signals.json`, and "
        "`validation.json`. Reproducibility authority is in `execution_provenance.json`, "
        "`resolved_config.json`, and the complete-tree `artifact_index.json`. Locked review "
        "and source-off panels are descriptive only.\n"
    )


def compute_advance_signal_panel(
    config: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
    budget_results: Mapping[str, Any],
    bootstrap_comparisons: Mapping[str, Mapping[str, Any]],
    seed_stability: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    *,
    positive_anchor_count: int,
) -> dict[str, Any]:
    """Evaluate every preregistered engineering signal without promoting claims."""

    thresholds = config["advance_signals"]
    primary_budget = str(config["evaluation"]["primary_budget"])
    recall = {
        method: float(values[primary_budget]["known_positive_proposal_recall_at_budget"])
        for method, values in budget_results["aggregate"].items()
    }
    frozen_baselines = ("carrier_signed", "cfar_score", "expert_separation_equal_weight")
    strongest_baseline = max(frozen_baselines, key=lambda name: (recall[name], name))
    mlp_auc = float(
        methods["tiny_mlp_spu"]["metrics"]["macro_fold_positive_vs_unlabeled_rank_auc"]
    )
    linear_auc = float(
        methods["linear_spu"]["metrics"]["macro_fold_positive_vs_unlabeled_rank_auc"]
    )
    by_method_fold = {
        (str(row["method"]), int(row["fold_id"])): float(
            row["known_positive_proposal_recall_at_budget"]
        )
        for row in budget_results["per_fold"]
    }
    per_fold_losses = [
        {
            "fold_id": fold_id,
            "tiny_mlp_recall_at_58": by_method_fold[("tiny_mlp_spu", fold_id)],
            "strongest_frozen_baseline": strongest_baseline,
            "strongest_frozen_baseline_recall_at_58": by_method_fold[(strongest_baseline, fold_id)],
            "tiny_mlp_loss_relative_to_baseline": max(
                0.0,
                by_method_fold[(strongest_baseline, fold_id)]
                - by_method_fold[("tiny_mlp_spu", fold_id)],
            ),
        }
        for fold_id in range(1, int(config["outer_validation"]["fold_count"]) + 1)
    ]
    maximum_fold_loss = max(row["tiny_mlp_loss_relative_to_baseline"] for row in per_fold_losses)
    permutation_p = float(
        methods["tiny_mlp_spu"]["representative_unit_score_permutation_null"][
            "greater_or_equal_p_value"
        ]
    )
    checks = {
        "minimum_unreserved_positive_anchors": {
            "observed": positive_anchor_count,
            "threshold": int(thresholds["minimum_unreserved_positive_anchors"]),
            "passed": positive_anchor_count >= int(thresholds["minimum_unreserved_positive_anchors"]),
        },
        "minimum_fold_support": {
            "observed_minimum_positive_anchors": min(
                int(fold["test_positive_anchor_count"]) for fold in fold_plan["folds"]
            ),
            "observed_minimum_unlabeled_spatial_groups": min(
                int(fold["test_unlabeled_spatial_group_count"]) for fold in fold_plan["folds"]
            ),
            "threshold_each": 10,
            "passed": all(
                int(fold["test_positive_anchor_count"]) >= 10
                and int(fold["test_unlabeled_spatial_group_count"]) >= 10
                for fold in fold_plan["folds"]
            ),
        },
        "tiny_mlp_minus_linear_macro_fold_spu_auc": {
            "observed": mlp_auc - linear_auc,
            "threshold": float(thresholds["tiny_mlp_minus_linear_spu_auc_minimum"]),
            "passed": mlp_auc - linear_auc
            >= float(thresholds["tiny_mlp_minus_linear_spu_auc_minimum"]),
        },
        "tiny_mlp_minus_linear_grouped_ci_lower": {
            "observed": float(
                bootstrap_comparisons["linear_spu"]["bootstrap_delta_ci95_low"]
            ),
            "threshold": float(thresholds["tiny_mlp_minus_linear_grouped_ci_lower_minimum"]),
            "passed": float(
                bootstrap_comparisons["linear_spu"]["bootstrap_delta_ci95_low"]
            )
            > float(thresholds["tiny_mlp_minus_linear_grouped_ci_lower_minimum"]),
        },
        "tiny_mlp_minus_cfar_macro_fold_spu_auc": {
            "observed": mlp_auc
            - float(
                methods["cfar_score"]["metrics"][
                    "macro_fold_positive_vs_unlabeled_rank_auc"
                ]
            ),
            "threshold": float(thresholds["tiny_mlp_minus_cfar_spu_auc_minimum"]),
            "passed": mlp_auc
            - float(
                methods["cfar_score"]["metrics"][
                    "macro_fold_positive_vs_unlabeled_rank_auc"
                ]
            )
            >= float(thresholds["tiny_mlp_minus_cfar_spu_auc_minimum"]),
        },
        "tiny_mlp_minus_cfar_grouped_ci_lower": {
            "observed": float(
                bootstrap_comparisons["cfar_score"]["bootstrap_delta_ci95_low"]
            ),
            "threshold": float(thresholds["tiny_mlp_minus_cfar_grouped_ci_lower_minimum"]),
            "passed": float(
                bootstrap_comparisons["cfar_score"]["bootstrap_delta_ci95_low"]
            )
            > float(thresholds["tiny_mlp_minus_cfar_grouped_ci_lower_minimum"]),
        },
        "tiny_mlp_minus_strongest_frozen_recall_at_58": {
            "strongest_frozen_baseline": strongest_baseline,
            "observed": recall["tiny_mlp_spu"] - recall[strongest_baseline],
            "threshold": float(
                thresholds["tiny_mlp_minus_strongest_frozen_recall_at_58_minimum"]
            ),
            "passed": recall["tiny_mlp_spu"] - recall[strongest_baseline]
            >= float(thresholds["tiny_mlp_minus_strongest_frozen_recall_at_58_minimum"]),
        },
        "maximum_per_fold_recall_loss": {
            "observed": maximum_fold_loss,
            "threshold": float(thresholds["maximum_per_fold_recall_loss"]),
            "passed": maximum_fold_loss <= float(thresholds["maximum_per_fold_recall_loss"]),
            "per_fold": per_fold_losses,
        },
        "tiny_mlp_representative_unit_permutation_p": {
            "observed": permutation_p,
            "threshold": float(thresholds["maximum_representative_unit_permutation_p"]),
            "passed": permutation_p
            <= float(thresholds["maximum_representative_unit_permutation_p"]),
        },
        "minimum_mlp_seed_rank_correlation": {
            "observed": float(seed_stability["median_pairwise_spearman"]),
            "threshold": float(thresholds["minimum_mlp_seed_rank_correlation"]),
            "passed": float(seed_stability["median_pairwise_spearman"])
            >= float(thresholds["minimum_mlp_seed_rank_correlation"]),
        },
        "maximum_mlp_seed_recall_at_58_range": {
            "observed": float(seed_stability["known_positive_proposal_recall_at_58_range"]),
            "threshold": float(thresholds["maximum_mlp_seed_recall_at_58_range"]),
            "passed": float(seed_stability["known_positive_proposal_recall_at_58_range"])
            <= float(thresholds["maximum_mlp_seed_recall_at_58_range"]),
        },
        "single_burst_spatial_block_and_review_policy_explanation_excluded": {
            "observed": "not_evaluated_in_this_bounded_primary_screen",
            "threshold": "must_be_excluded_before_advancement",
            "passed": False,
            "reason": "leave-one-burst-out and alternate frozen review-policy sensitivities are not primary outputs",
        },
    }
    return {
        "schema_version": 1,
        "interpretation": "descriptive_engineering_advance_signal_not_scientific_gate",
        "checks": checks,
        "all_preregistered_advance_signals_passed": all(
            bool(record["passed"]) for record in checks.values()
        ),
        "scientific_claim_promotion_allowed": False,
    }


def run_uncertainty_aware_learning(
    config_path: Path = CONFIG_RELATIVE_PATH,
    *,
    repository_root: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Execute once under the frozen entry-to-promotion process deadline."""

    with _PosixProcessDeadline(_END_TO_END_DEADLINE_SECONDS) as deadline:
        return _run_uncertainty_aware_learning_once(
            config_path,
            repository_root=repository_root,
            data_root=data_root,
            promotion_deadline=deadline,
        )


def _run_uncertainty_aware_learning_once(
    config_path: Path = CONFIG_RELATIVE_PATH,
    *,
    repository_root: Path | None = None,
    data_root: Path | None = None,
    promotion_deadline: _PosixProcessDeadline,
) -> dict[str, Any]:
    """Execute the frozen screen once and atomically publish a complete tree."""

    started_dt = datetime.now(timezone.utc)
    start_repository = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else _runner_path().parents[3]
    )
    git_provenance = capture_git_provenance(start_repository)
    runtime_provenance = _runtime_provenance(start_repository)
    context = _preflight_context(
        config_path,
        repository_root=repository_root,
        data_root=data_root,
        git_provenance_at_start=git_provenance,
    )
    repository = context["repository"]
    data = context["data"]
    config = context["config"]
    partial = context["partial"]
    output = context["output"]
    progress = partial / "progress" / "runner.jsonl"
    os.environ.setdefault("OMP_NUM_THREADS", str(config["resources"]["numpy_threads"]))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(config["resources"]["numpy_threads"]))
    threadpools_before_limit = threadpool_info()
    thread_limiter = threadpool_limits(limits=int(config["resources"]["numpy_threads"]))
    thread_limit_restore_attempted = False
    threadpools_inside_limit = threadpool_info()
    disk = psutil.disk_usage(repository)
    memory = psutil.virtual_memory()
    resource_snapshot = {
        "captured_at": started_dt.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes": int(disk.free),
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
        "threadpools_before_limit": threadpools_before_limit,
        "threadpools_inside_limit": threadpools_inside_limit,
        "requested_numpy_thread_limit": int(config["resources"]["numpy_threads"]),
        "torch_threads_configuration_applicable": False,
        "thread_backend": "numpy_threadpoolctl_not_torch",
    }
    partial.mkdir(parents=True)
    try:
        _progress(progress, "preflight_passed")
        _write_json(partial / "preflight.json", context["payload"])
        _write_json(partial / "configuration.source.json", config)
        _write_json(
            partial / "status.json",
            {
                "schema_version": 1,
                "status": "running_engineering_screen",
                "scientific_audit_complete": False,
                "scientific_completion": False,
                "claim_promotion_allowed": False,
            },
        )
        census_core.build_uncertainty_aware_census(
            repository_root=repository,
            data_root=data,
            config_path=context["source_paths"]["innovation_ranker_v5_resolved_config"],
            output_root=context["census_output"],
            progress=context["census_progress"],
        )
        _progress(progress, "census_materialized")
        event_columns, event_rows = _read_tsv(context["census_output"] / "event_candidates.tsv")
        quiet_columns, quiet_rows = _read_tsv(context["census_output"] / "quiet_candidates.tsv")
        if event_columns != quiet_columns:
            raise LearningContractError("event and quiet census schemas differ")
        candidate_rows = event_rows + quiet_rows

        harmonized = data_core.harmonize_innovation_candidate_census(
            candidate_rows,
            context["source_paths"]["canonical_v7_adjudication"],
            data_core.INNOVATION_RANKER_V5_CENSUS,
            repository_root=repository,
            data_root=data,
            review_site_source=context["source_paths"]["detection_profile_taxonomy_v5_sites"],
            review_occurrence_source=context["source_paths"]["detection_profile_taxonomy_v5_occurrences"],
            new_review_source=context["source_paths"]["new_candidate_single_reviewer_labels"],
            readiness_requirements=data_core.ReadinessRequirements(),
        )
        harmonized.assert_model_ready()
        _progress(progress, "labels_harmonized")
        outer = config["outer_validation"]
        fold_plan = build_contiguous_x_block_folds(
            harmonized.fold_assignment_rows(),
            fold_count=int(outer["fold_count"]),
            boundary_guard_px=float(outer["boundary_guard_px"]),
            min_positive_anchors_per_fold=int(outer["minimum_positive_anchors_per_fold"]),
            min_unlabeled_spatial_groups_per_fold=int(outer["minimum_unlabeled_spatial_groups_per_fold"]),
        )
        if fold_plan["assignment_sha256"] != outer["expected_assignment_sha256"]:
            raise LearningContractError("live fold assignment differs from the frozen hash")
        if (
            fold_plan["train_test_guard_membership_sha256"]
            != outer["expected_train_test_guard_membership_sha256"]
        ):
            raise LearningContractError(
                "live train/test/guard membership differs from the frozen hash"
            )
        harmonizer_fold_validation = harmonized.validate_outer_fold_assignments(fold_plan["assignments"])
        model_rows = list(harmonized.model_feature_rows())
        frozen_folds = materialize_frozen_outer_folds(model_rows, fold_plan)
        feature_names = tuple(config["features"]["primary_compact_feature_ids"])
        X = _matrix(model_rows, feature_names)
        if X.shape[1] != 12:
            raise LearningContractError("learned model input width is not exactly 12 raw features")
        labels = [row["training_state"] for row in model_rows]
        groups = [row["leakage_group_id"] for row in model_rows]
        sample_ids = [str(row["row_id"]) for row in model_rows]
        evaluation_config = _evaluation_config(config)
        evaluation = model_core.nested_spatial_pu_evaluation(
            X,
            labels,
            groups,
            feature_names=feature_names,
            sample_ids=sample_ids,
            baseline_scores={name: X[:, feature_names.index(name)] for name in ("carrier_signed", "cfar_score")},
            equal_weight_feature_names=tuple(config["features"]["expert_separation_feature_ids"]),
            frozen_outer_folds=frozen_folds,
            config=evaluation_config,
        )
        evaluation["runner_metric_contract"] = {
            "primary_spu_auc": "macro_mean_of_within_test_fold_spu_auc",
            "global_fixed_budget_scores": "label_free_within_test_fold_midrank_percentiles",
            "primary_per_fold_recall_budget": "literal_K_equals_min_58_and_test_fold_candidate_count",
            "core_proportional_per_fold_budget_rows": "sensitivity_only_not_per_fold_recall_at_58",
        }
        solver_completion = {
            "schema_version": 1,
            "contract": {
                "common_penalty": float(evaluation_config.logistic_penalties[0]),
                "convergence_tolerance": float(evaluation_config.logistic_tolerance),
                "maximum_iterations": int(evaluation_config.max_logistic_iterations),
                "outer_folds": int(evaluation_config.outer_splits),
                "bagged_pu_base_fits_per_fold": int(evaluation_config.pu_bags),
            },
            "completed_without_nonconvergence_exception": True,
            "completed_l2_fits": int(evaluation_config.outer_splits),
            "completed_elastic_fits": int(evaluation_config.outer_splits),
            "completed_bagged_pu_base_fits": int(
                evaluation_config.outer_splits * evaluation_config.pu_bags
            ),
            "completed_reference_fit_total": int(
                evaluation_config.outer_splits * (2 + evaluation_config.pu_bags)
            ),
            "nonconvergence_exception_count": 0,
            "per_fit_iteration_telemetry_available": False,
            "interpretation": "completion evidence for the canonical evaluation only; pre-run convergence-probe telemetry is a separate artifact",
        }
        _progress(progress, "primary_oof_evaluation_complete")

        _, canonical_rows = _read_tsv(context["source_paths"]["canonical_v7_adjudication"])
        _, review_site_rows = _read_tsv(context["source_paths"]["detection_profile_taxonomy_v5_sites"])
        _, review_occurrence_rows = _read_tsv(
            context["source_paths"]["detection_profile_taxonomy_v5_occurrences"]
        )
        _, new_review_rows = _read_tsv(
            context["source_paths"]["new_candidate_single_reviewer_labels"]
        )
        misses = classify_proposal_stage_misses(
            canonical_rows,
            harmonized.rows,
            review_site_rows,
            review_reserve_radius_px=float(config["label_contract"]["new_review_reserve_radius_px"]),
            candidate_match_radius_px=float(
                config["label_contract"]["canonical_confirmed_match_radius_px"]
            ),
        )
        if misses["total_unmatched_confirmed"] != harmonized.validation["confirmed_labels_without_anchor"]:
            raise LearningContractError("proposal-stage miss derivation disagrees with harmonizer")
        misses["denominator_audit"] = {
            "confirmed_canonical_occurrences": harmonized.validation[
                "confirmed_canonical_labels"
            ],
            "matched_primary_unreserved_anchors": harmonized.validation[
                "primary_positive_anchors_after_review_reserve"
            ],
            "matched_review_reserved_anchors": harmonized.validation[
                "confirmed_anchors_reserved_for_review"
            ],
            "all_matched_anchors": harmonized.validation["confirmed_positive_anchors"],
            "unmatched_confirmed_total": misses["total_unmatched_confirmed"],
            "unmatched_confirmed_review_reserved": misses["reserved_unmatched_confirmed"],
            "unmatched_confirmed_unreserved": misses["unreserved_unmatched_confirmed"],
        }
        misses["denominator_audit"]["reconciles"] = (
            misses["denominator_audit"]["matched_primary_unreserved_anchors"]
            + misses["denominator_audit"]["matched_review_reserved_anchors"]
            + misses["denominator_audit"]["unmatched_confirmed_total"]
            == misses["denominator_audit"]["confirmed_canonical_occurrences"]
        )
        if not misses["denominator_audit"]["reconciles"]:
            raise LearningContractError("canonical denominator audit does not reconcile")
        unreserved_misses = [row for row in misses["rows"] if not row["inside_locked_review_reserve"]]
        methods = _method_results(evaluation)
        positive_mask = np.asarray([label == "positive" for label in labels], dtype=bool)
        budget_results = _evaluate_budgets(
            methods,
            positive_mask,
            sample_ids,
            config["evaluation"]["global_candidate_budgets"],
            unreserved_misses,
            fold_plan,
        )
        misses["rows"] = _assign_miss_folds(misses["rows"], fold_plan)

        mlp_scores = _method_score_vector(
            methods["tiny_mlp_spu"], expected_length=len(sample_ids)
        )
        fold_ids = [fold_plan["assignments"][candidate_id] for candidate_id in sample_ids]
        representative_unit_rows: dict[str, dict[str, Any]] = {}
        for method_name, payload in methods.items():
            # permutation_draws=0 disables the core's confounded group-maximum
            # null; this pop is defensive against an incompatible core result.
            payload.pop("spatial_group_score_association_null", None)
            method_scores = _method_score_vector(
                payload, expected_length=len(sample_ids)
            )
            null_result = fold_stratified_representative_unit_score_permutation_null(
                positive_mask,
                method_scores,
                sample_ids,
                fold_ids,
                groups,
                draws=int(config["evaluation"]["representative_unit_score_permutations"]),
                seed=int(config["evaluation"]["representative_unit_score_permutation_seed"]),
            )
            representatives = null_result.pop("representatives")
            payload["representative_unit_score_permutation_null"] = null_result
            for record in representatives:
                unit_id = str(record["unit_id"])
                row = representative_unit_rows.setdefault(
                    unit_id, {key: value for key, value in record.items() if key != "score"}
                )
                row[f"score__{method_name}"] = record["score"]
        bootstrap_comparators = (
            "linear_spu",
            "carrier_signed",
            "cfar_score",
            "expert_separation_equal_weight",
        )
        bootstrap_comparisons = {
            comparator: grouped_paired_bootstrap_macro_fold_auc(
                positive_mask,
                mlp_scores,
                _method_score_vector(
                    methods[comparator], expected_length=len(sample_ids)
                ),
                fold_ids,
                groups,
                draws=int(config["evaluation"]["grouped_bootstrap_draws"]),
                seed=int(config["evaluation"]["grouped_bootstrap_seed"]),
                score_a_name="tiny_mlp_spu_within_fold_oof",
                score_b_name=f"{comparator}_within_fold_oof",
            )
            for comparator in bootstrap_comparators
        }
        bootstrap = {
            "schema_version": 1,
            "estimand": "paired_delta_in_macro_mean_within_fold_positive_vs_unlabeled_rank_auc",
            "comparisons": bootstrap_comparisons,
            "observed_strongest_baseline_selection_repeated_inside_draws": False,
        }
        seed_metrics = {}
        for seed, scores in methods["tiny_mlp_spu"]["seed_oof_scores"].items():
            seed_metrics[str(seed)] = adjust_recall_for_proposal_misses(
                model_core.rank_metrics(positive_mask, scores, budget=58, tie_breakers=sample_ids),
                len(unreserved_misses),
            )
        seed_recalls = [row["known_positive_proposal_recall_at_budget"] for row in seed_metrics.values()]
        seed_stability = {
            **methods["tiny_mlp_spu"]["seed_rank_stability"],
            "per_seed_budget_58": seed_metrics,
            "known_positive_proposal_recall_at_58_range": max(seed_recalls) - min(seed_recalls),
        }
        advance_signals = compute_advance_signal_panel(
            config,
            methods,
            budget_results,
            bootstrap_comparisons,
            seed_stability,
            fold_plan,
            positive_anchor_count=int(np.sum(positive_mask)),
        )
        audits = _score_locked_audits(harmonized, model_rows, feature_names, evaluation_config)
        audits["site_rows"] = aggregate_review_site_audit(
            review_site_rows,
            review_occurrence_rows,
            new_review_rows,
            canonical_rows,
            audits["review_rows"],
            match_radius_px=float(config["label_contract"]["canonical_confirmed_match_radius_px"]),
        )
        if len(audits["site_rows"]) != 18:
            raise LearningContractError("locked review audit is not the exact 18-site panel")
        _progress(progress, "locked_descriptive_panels_scored")

        tables = partial / "tables"
        _write_tsv(tables / "harmonized_candidates.tsv", _flatten_harmonized(harmonized.rows, harmonized.feature_columns))
        assignment_rows = []
        for row in harmonized.fold_assignment_rows():
            if row["eligible_for_primary_fit"]:
                assignment_rows.append({**row, "fold_id": fold_plan["assignments"][row["candidate_id"]]})
        _write_tsv(tables / "fold_assignments.tsv", assignment_rows)
        _write_tsv(tables / "fold_boundaries.tsv", fold_plan["boundaries"])
        _write_tsv(
            tables / "fold_summary.tsv",
            [
                {
                    "fold_id": fold["fold_id"],
                    "test_candidate_count": fold["test_candidate_count"],
                    "test_positive_anchor_count": fold["test_positive_anchor_count"],
                    "test_positive_identity_count": fold["test_positive_identity_count"],
                    "test_unlabeled_count": fold["test_unlabeled_count"],
                    "test_unlabeled_spatial_group_count": fold[
                        "test_unlabeled_spatial_group_count"
                    ],
                    "train_candidate_count": fold["train_candidate_count"],
                    "guard_purged_candidate_count": fold[
                        "guard_purged_candidate_count"
                    ],
                    "guard_purged_leakage_group_count": len(
                        fold["guard_purged_leakage_group_ids"]
                    ),
                    "minimum_retained_train_test_candidate_distance_px": fold[
                        "minimum_retained_train_test_candidate_distance_px"
                    ],
                    "retained_train_test_pairs_within_inclusive_guard": fold[
                        "retained_train_test_pairs_within_inclusive_guard"
                    ],
                }
                for fold in fold_plan["folds"]
            ],
        )
        _write_tsv(
            tables / "fold_whole_group_closure_extensions.tsv",
            fold_plan["whole_group_closure_extensions"],
            columns=(
                "fold_id",
                "leakage_group_id",
                "representative_x_px",
                "minimum_x_px",
                "maximum_x_px",
                "left_extension_px",
                "right_extension_px",
                "reason",
            ),
        )
        _write_tsv(
            tables / "fold_guard_purges.tsv",
            fold_plan["guard_purge_rows"],
            columns=(
                "fold_id",
                "leakage_group_id",
                "candidate_id",
                "x_px",
                "y_px",
                "boundary_interval_guard_intersection",
                "held_candidate_euclidean_guard_intersection",
                "minimum_distance_to_held_test_candidate_px",
                "reason",
            ),
        )
        _write_tsv(tables / "proposal_stage_misses.tsv", misses["rows"], columns=("observation_id", "burst_id", "canonical_roi_id", "x_px", "y_px", "inside_locked_review_reserve", "nearest_review_site_id", "nearest_review_distance_px", "nearest_frozen_candidate_id", "nearest_frozen_candidate_distance_px", "no_candidate_within_match_radius", "candidate_match_radius_px", "primary_denominator_role", "assigned_outer_fold"))
        oof_rows = []
        for index, candidate_id in enumerate(sample_ids):
            row = {"candidate_id": candidate_id, "training_state": labels[index], "fold_id": fold_plan["assignments"][candidate_id]}
            for method_name, result in methods.items():
                scores = _method_score_vector(
                    result, expected_length=len(sample_ids)
                )
                raw_scores = _method_score_vector(
                    result,
                    raw_fold_specific=True,
                    expected_length=len(sample_ids),
                )
                row[f"score__{method_name}__fold_percentile"] = scores[index]
                row[f"score__{method_name}__raw_fold_specific"] = raw_scores[index]
            oof_rows.append(row)
        _write_tsv(tables / "oof_scores.tsv", oof_rows)
        aggregate_rows = [
            {
                "method": method,
                "budget": int(budget),
                **_global_budget_metric_output(metrics),
            }
            for method, budgets in budget_results["aggregate"].items()
            for budget, metrics in budgets.items()
        ]
        _write_tsv(tables / "aggregate_metrics.tsv", aggregate_rows)
        _write_tsv(
            tables / "per_fold_metrics.tsv",
            [
                {
                    **row,
                    "metric_scope": "within_held_test_fold_primary_auc_and_literal_budget_58_recall",
                }
                for row in budget_results["per_fold"]
            ],
        )
        permutation_rows = [
            {"method": name, **payload["representative_unit_score_permutation_null"]}
            for name, payload in methods.items()
        ]
        _write_tsv(tables / "permutation_controls.tsv", permutation_rows)
        _write_tsv(
            tables / "representative_unit_scores.tsv",
            [representative_unit_rows[key] for key in sorted(representative_unit_rows)],
        )
        _write_tsv(tables / "seed_stability.tsv", [{"seed": seed, **metrics} for seed, metrics in seed_metrics.items()])
        _write_tsv(
            tables / "advance_signals.tsv",
            [
                {
                    "signal": name,
                    "observed": record.get("observed"),
                    "threshold": record.get("threshold"),
                    "passed": record["passed"],
                    "reason": record.get("reason"),
                }
                for name, record in advance_signals["checks"].items()
            ],
            columns=("signal", "observed", "threshold", "passed", "reason"),
        )
        _write_tsv(tables / "review_conflict_panel.tsv", audits["site_rows"])
        _write_tsv(tables / "review_conflict_candidate_scores.tsv", audits["review_rows"])
        _write_tsv(tables / "quiet_null_panel.tsv", audits["quiet_rows"], columns=("candidate_id", "partition_id", "training_state", "tiny_mlp_final_fit_score", "audit_role"))

        primary_budget_58_metrics = {
            method: _global_budget_metric_output(
                values[str(config["evaluation"]["primary_budget"])]
            )
            for method, values in budget_results["aggregate"].items()
        }
        primary_macro_fold_spu_auc = {
            method: float(
                payload["metrics"]["macro_fold_positive_vs_unlabeled_rank_auc"]
            )
            for method, payload in methods.items()
        }
        cluster_bootstrap_contrasts = {
            comparator: {
                "tiny_mlp_minus_comparator_observed_delta": float(
                    record["observed_delta_a_minus_b"]
                ),
                "cluster_bootstrap_ci95_low": float(record["bootstrap_delta_ci95_low"]),
                "cluster_bootstrap_ci95_high": float(record["bootstrap_delta_ci95_high"]),
                "resampling_unit": record["resampling_unit"],
                "interval_interpretation": "descriptive_cluster_bootstrap_interval_non_claim_bearing",
            }
            for comparator, record in bootstrap_comparisons.items()
        }
        strongest_frozen_baseline = advance_signals["checks"][
            "tiny_mlp_minus_strongest_frozen_recall_at_58"
        ]["strongest_frozen_baseline"]
        summary = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "status": "complete_non_claim_bearing_engineering_screen",
            "stage": "bounded_model_only_spatial_positive_unlabeled_feature_fusion_screen",
            "applicability": "within_this_recording_and_frozen_broad_candidate_union_only",
            "estimand": evaluation["estimand"],
            "generalization_scope": "model_only_spatial_holdout_within_one_recording_conditional_on_frozen_label_derived_temporal_windows",
            "oracle_temporal_window_conditioning": True,
            "historical_same_recording_label_informed_feature_program": True,
            "cfar_comparison_semantics": "cfar_score_reranking_over_same_broad_union_and_also_one_feature_fusion_input",
            "end_to_end_cfar_detector_replacement_evaluated": False,
            "primary_positive_anchors": int(np.sum(positive_mask)),
            "primary_unlabeled_candidates": int(np.sum(~positive_mask)),
            "matched_review_reserved_positive_anchors": misses["denominator_audit"][
                "matched_review_reserved_anchors"
            ],
            "confirmed_canonical_occurrences": misses["denominator_audit"][
                "confirmed_canonical_occurrences"
            ],
            "unreserved_proposal_stage_misses": misses["unreserved_unmatched_confirmed"],
            "reserved_proposal_stage_misses": misses["reserved_unmatched_confirmed"],
            "primary_budget": int(config["evaluation"]["primary_budget"]),
            "operating_point": "global_top_58_by_label_free_within_test_fold_midrank_percentile",
            "primary_macro_fold_spu_auc_by_method": primary_macro_fold_spu_auc,
            "primary_budget_58_metrics_by_method": primary_budget_58_metrics,
            "primary_budget_58_auc_field_semantics": "global_fold_percentile_spu_auc_sensitivity_not_primary_macro_fold_auc",
            "tiny_mlp_cluster_bootstrap_contrasts": cluster_bootstrap_contrasts,
            "positive_containing_leakage_components_by_fold": [9, 3, 3, 2, 4],
            "bootstrap_interval_support_caveat": (
                "descriptive_only_due_to_2_or_3_positive_containing_leakage_components_in_some_folds"
            ),
            "tiny_mlp_minus_linear_cluster_bootstrap": cluster_bootstrap_contrasts[
                "linear_spu"
            ],
            "tiny_mlp_minus_cfar_cluster_bootstrap": cluster_bootstrap_contrasts[
                "cfar_score"
            ],
            "observed_strongest_frozen_baseline_at_budget_58": strongest_frozen_baseline,
            "tiny_mlp_minus_observed_strongest_frozen_cluster_bootstrap_descriptive": (
                cluster_bootstrap_contrasts[strongest_frozen_baseline]
            ),
            "strongest_baseline_selection_repeated_inside_bootstrap_draws": False,
            "tiny_mlp_minus_strongest_frozen_recall_at_58": float(
                primary_budget_58_metrics["tiny_mlp_spu"][
                    "known_positive_proposal_recall_at_budget"
                ]
                - primary_budget_58_metrics[strongest_frozen_baseline][
                    "known_positive_proposal_recall_at_budget"
                ]
            ),
            "tiny_mlp_representative_component_null_p": float(
                methods["tiny_mlp_spu"]["representative_unit_score_permutation_null"][
                    "greater_or_equal_p_value"
                ]
            ),
            "tiny_mlp_seed_median_pairwise_spearman": float(
                seed_stability["median_pairwise_spearman"]
            ),
            "tiny_mlp_seed_recall_at_58_range": float(
                seed_stability["known_positive_proposal_recall_at_58_range"]
            ),
            "all_preregistered_advance_signals_passed": advance_signals[
                "all_preregistered_advance_signals_passed"
            ],
            "advance_verdict": (
                "all_descriptive_engineering_signals_passed"
                if advance_signals["all_preregistered_advance_signals_passed"]
                else "one_or_more_descriptive_engineering_signals_did_not_pass"
            ),
            "scientific_audit_complete": False,
            "scientific_completion": False,
            "claim_promotion_allowed": False,
        }
        report_text = _report(summary)
        frozen_inputs_and_implementation_unchanged = all(
            _sha256_file(context["source_paths"][source_id]) == record["sha256"]
            for source_id, record in context["payload"]["inputs"].items()
        ) and all(
            _sha256_file(repository / relative) == record["sha256"]
            for relative, record in context["payload"]["implementation"].items()
        )
        post_run_authority_hashes = {
            "config_sha256": _sha256_file(context["config_file"]),
            "protocol_sha256": _sha256_file(context["protocol"]),
            "descriptor_sha256": _sha256_file(context["descriptor_file"]),
        }
        frozen_authority_documents_unchanged = all(
            observed == context["payload"]["hashes"][hash_key]
            for hash_key, observed in post_run_authority_hashes.items()
        )
        frozen_bytes_unchanged = (
            frozen_inputs_and_implementation_unchanged
            and frozen_authority_documents_unchanged
        )
        dependency_declaration_unchanged = (
            _sha256_file(repository / DEPENDENCY_DECLARATION_RELATIVE_PATH)
            == runtime_provenance["dependency_declaration"]["sha256"]
        )
        all_oof_exact = all(
            len(_method_score_vector(payload, expected_length=len(model_rows)))
            == len(model_rows)
            for payload in methods.values()
        )
        ended_dt = datetime.now(timezone.utc)
        started_at = started_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        ended_at = ended_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        duration_seconds = (ended_dt - started_dt).total_seconds()
        controlled_threadpools = [
            record
            for record in threadpools_inside_limit
            if record.get("user_api") in {"blas", "openmp"}
        ]
        thread_cap_enforced = bool(controlled_threadpools) and all(
            int(record.get("num_threads", 1)) <= int(config["resources"]["numpy_threads"])
            for record in controlled_threadpools
        )
        engineering_checks = {
            "preflight_all_gates_passed": all(context["payload"]["gates"].values()),
            "frozen_authority_cardinality_exact_24_inputs_12_implementations": (
                len(context["payload"]["inputs"]) == 24
                and len(context["payload"]["implementation"]) == 12
            ),
            "harmonizer_model_readiness_passed": bool(
                harmonized.validation["model_readiness"]["passed"]
            ),
            "harmonizer_fold_assignment_validation_passed": bool(
                harmonizer_fold_validation.get("passed")
            ),
            "fold_plan_passed": bool(fold_plan["validation"]["passed"]),
            "frozen_fold_assignment_hash_exact": (
                fold_plan["assignment_sha256"]
                == config["outer_validation"]["expected_assignment_sha256"]
            ),
            "frozen_train_test_guard_membership_hash_exact": (
                fold_plan["train_test_guard_membership_sha256"]
                == config["outer_validation"][
                    "expected_train_test_guard_membership_sha256"
                ]
            ),
            "model_input_feature_names_exact": feature_names == PRIMARY_FEATURES,
            "model_input_width_exact_12": X.shape[1] == 12,
            "coordinates_absent_from_model_inputs": not bool(
                {"x", "y", "x_px", "y_px", "coordinates"} & set(feature_names)
            ),
            "all_eligible_rows_have_one_finite_oof_score_per_method": all_oof_exact,
            "frozen_inputs_and_implementation_unchanged_pre_post": frozen_bytes_unchanged,
            "config_protocol_and_descriptor_unchanged_pre_post": (
                frozen_authority_documents_unchanged
            ),
            "dependency_declaration_unchanged_pre_post": (
                dependency_declaration_unchanged
            ),
            "zero_retained_train_test_candidate_pairs_within_inclusive_guard": (
                bool(
                    fold_plan["validation"][
                        "zero_retained_train_test_candidate_pairs_within_inclusive_guard"
                    ]
                )
                and all(
                    fold["retained_train_test_pairs_within_inclusive_guard"] == 0
                    and float(
                        fold["minimum_retained_train_test_candidate_distance_px"]
                    )
                    > float(config["outer_validation"]["boundary_guard_px"])
                    for fold in fold_plan["folds"]
                )
            ),
            "canonical_reference_solver_fits_completed_without_exception": (
                solver_completion["completed_reference_fit_total"] == 60
                and solver_completion["nonconvergence_exception_count"] == 0
                and solver_completion["contract"]
                == {
                    "common_penalty": 0.01,
                    "convergence_tolerance": 1e-6,
                    "maximum_iterations": 50_000,
                    "outer_folds": 5,
                    "bagged_pu_base_fits_per_fold": 10,
                }
            ),
            "numeric_thread_cap_observed": thread_cap_enforced,
            "process_deadline_alarm_active_before_atomic_promotion": (
                signal.getitimer(signal.ITIMER_REAL)[0] > 0
                and config["resources"]["deadline_enforcement"]
                == "posix_process_alarm_from_run_start_through_pre_promotion_validation_then_disarm_before_single_atomic_rename"
            ),
            "assembly_completed_inside_end_to_end_deadline": duration_seconds
            <= float(config["resources"]["maximum_end_to_end_wall_minutes"])
            * 60.0,
            "proposal_miss_count_reconciled": (
                misses["total_unmatched_confirmed"]
                == harmonized.validation["confirmed_labels_without_anchor"]
            ),
            "macro_fold_bootstrap_observed_scores_reconciled": (
                abs(
                    bootstrap["comparisons"]["linear_spu"][
                        "observed_score_a_macro_fold_spu_auc"
                    ]
                    - methods["tiny_mlp_spu"]["metrics"][
                        "macro_fold_positive_vs_unlabeled_rank_auc"
                    ]
                )
                <= 1e-12
                and abs(
                    bootstrap["comparisons"]["linear_spu"][
                        "observed_score_b_macro_fold_spu_auc"
                    ]
                    - methods["linear_spu"]["metrics"][
                        "macro_fold_positive_vs_unlabeled_rank_auc"
                    ]
                )
                <= 1e-12
            ),
            "all_prespecified_mlp_baseline_bootstrap_contrasts_reconciled": (
                set(bootstrap["comparisons"])
                == {
                    "linear_spu",
                    "carrier_signed",
                    "cfar_score",
                    "expert_separation_equal_weight",
                }
                and all(
                    abs(
                        record["observed_score_a_macro_fold_spu_auc"]
                        - methods["tiny_mlp_spu"]["metrics"][
                            "macro_fold_positive_vs_unlabeled_rank_auc"
                        ]
                    )
                    <= 1e-12
                    and abs(
                        record["observed_score_b_macro_fold_spu_auc"]
                        - methods[comparator]["metrics"][
                            "macro_fold_positive_vs_unlabeled_rank_auc"
                        ]
                    )
                    <= 1e-12
                    for comparator, record in bootstrap["comparisons"].items()
                )
            ),
            "macro_fold_bootstrap_dependency_component_support_exact": (
                [
                    (
                        row["positive_containing_leakage_component_count"],
                        row["unlabeled_only_leakage_component_count"],
                    )
                    for row in bootstrap["comparisons"]["linear_spu"]["observed_by_fold"]
                ]
                == [(9, 317), (3, 14), (3, 22), (2, 38), (4, 128)]
                and sum(
                    row["unlabeled_rows_inside_positive_containing_components"]
                    for row in bootstrap["comparisons"]["linear_spu"]["observed_by_fold"]
                )
                == 138
                and sum(
                    row["unlabeled_rows_inside_unlabeled_only_components"]
                    for row in bootstrap["comparisons"]["linear_spu"]["observed_by_fold"]
                )
                == 1403
            ),
            "locked_review_site_panel_exact_18": len(audits["site_rows"]) == 18,
            "locked_review_overlap_and_conflict_counts_reconciled": (
                sum(row["canonical_v7_confirmed_overlap"] for row in audits["site_rows"])
                == harmonized.validation["review_holdout"]["canonical_v7_overlap"][
                    "reviewed_sites_with_confirmed_canonical_v7_match"
                ]
                and sum(row["cross_source_disagreement"] for row in audits["site_rows"])
                == harmonized.validation["review_holdout"]["canonical_v7_overlap"][
                    "reviewed_sites_with_cross_source_disagreement"
                ]
            ),
            "representative_unit_null_live_support_exact": (
                methods["tiny_mlp_spu"]["representative_unit_score_permutation_null"][
                    "positive_containing_leakage_component_count"
                ]
                == 21
                and methods["tiny_mlp_spu"]["representative_unit_score_permutation_null"][
                    "unlabeled_only_leakage_component_count"
                ]
                == 519
                and [
                    (
                        row["positive_containing_leakage_component_count"],
                        row["unlabeled_only_leakage_component_count"],
                    )
                    for row in methods["tiny_mlp_spu"][
                        "representative_unit_score_permutation_null"
                    ]["observed_by_fold"]
                ]
                == [(9, 317), (3, 14), (3, 22), (2, 38), (4, 128)]
                and len(representative_unit_rows) == 540
            ),
        }
        failed_engineering_checks = sorted(
            name for name, passed in engineering_checks.items() if not bool(passed)
        )
        if failed_engineering_checks:
            _write_json(
                partial / "validation.failed.json",
                {
                    "schema_version": 1,
                    "passed_engineering_execution": False,
                    "failed_engineering_checks": failed_engineering_checks,
                    "engineering_checks": engineering_checks,
                    "scientific_audit_complete": False,
                    "scientific_completion": False,
                    "claim_promotion_allowed": False,
                },
            )
            raise LearningContractError(
                "engineering validation failed: " + ", ".join(failed_engineering_checks)
            )
        canonical_command = [
            "python",
            "-m",
            RUNNER_MODULE,
            "run",
            "--config",
            CONFIG_RELATIVE_PATH.as_posix(),
            "--repository-root",
            ".",
            "--data-root",
            "$NEUROBENCH_DATA_ROOT",
        ]
        canonical_shell_replay = (
            f"python -m {RUNNER_MODULE} run --config {CONFIG_RELATIVE_PATH.as_posix()} "
            "--repository-root . --data-root \"$NEUROBENCH_DATA_ROOT\""
        )
        validation = {
            "schema_version": 1,
            "passed_engineering_execution": all(engineering_checks.values()),
            "engineering_checks": engineering_checks,
            "preflight_gates": context["payload"]["gates"],
            "harmonization": harmonized.validation,
            "fold_plan": fold_plan["validation"],
            "harmonizer_fold_validation": harmonizer_fold_validation,
            "model_input_feature_names": list(feature_names),
            "model_input_width": len(feature_names),
            "coordinates_in_model_matrix": not engineering_checks["coordinates_absent_from_model_inputs"],
            "all_eligible_rows_have_one_oof_score": all_oof_exact,
            "proposal_miss_accounting": {key: value for key, value in misses.items() if key != "rows"},
            "scope_disclosures": {
                "oracle_temporal_window_conditioning": True,
                "historical_same_recording_label_informed_feature_program": True,
                "cfar_arm_is_same_union_reranking": True,
                "cfar_score_is_also_a_model_input": "cfar_score" in feature_names,
                "end_to_end_detector_generalization_evaluated": False,
            },
            "scientific_audit_complete": False,
            "scientific_completion": False,
            "claim_promotion_allowed": False,
        }
        provenance = {
            "schema_version": 1,
            "record_type": "run_provenance",
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "lifecycle": "succeeded",
            "planned_at": config["planned_at_utc"],
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "execution": {
                "mode": "bounded_non_claim_bearing_engineering_screen",
                "command": canonical_command,
                "shell_replay_command": canonical_shell_replay,
                "working_directory_argument": ".",
                "working_directory_portable": "repo-workspace://",
                "data_root_argument": "$NEUROBENCH_DATA_ROOT",
                "timestamp_semantics": "started_at_public_run_entry; ended_at_is_pre_promotion_artifact_assembly_time; POSIX_alarm_covers_all_precommit_validation_and_is_synchronously_disarmed_immediately_before_the_single_atomic_rename",
            },
            "git_at_start": git_provenance,
            "runtime": runtime_provenance,
            "resources": {
                **resource_snapshot,
                "observed_duration_seconds": duration_seconds,
                "maximum_end_to_end_wall_minutes": float(
                    config["resources"]["maximum_end_to_end_wall_minutes"]
                ),
                "deadline_enforcement": config["resources"]["deadline_enforcement"],
                "assembly_completed_inside_end_to_end_deadline": engineering_checks[
                    "assembly_completed_inside_end_to_end_deadline"
                ],
                "numeric_thread_cap_observed": thread_cap_enforced,
            },
            "paths": context["payload"]["paths"],
            "hashes": context["payload"]["hashes"],
            "post_run_authority_hashes": post_run_authority_hashes,
            "inputs": context["payload"]["inputs"],
            "implementation": context["payload"]["implementation"],
            "census_manifest": _read_json(context["census_output"] / "provenance.json"),
            "harmonization_manifest": harmonized.manifest,
            "fold_assignment_sha256": fold_plan["assignment_sha256"],
            "model_evaluation_config": asdict(evaluation_config),
            "model_score_scale": "label_free_within_test_fold_midrank_percentile",
            "scope_disclosures": validation["scope_disclosures"],
            "frozen_bytes_unchanged_pre_post": frozen_bytes_unchanged,
        }
        resolved_config = {
            "schema_version": 2,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "runner_module": RUNNER_MODULE,
            "runner_sha256": context["payload"]["implementation"][RUNNER_RELATIVE_PATH.as_posix()]["sha256"],
            "configuration": config,
            "configuration_sha256": context["payload"]["hashes"]["config_sha256"],
            "execution": {
                "planned_at": config["planned_at_utc"],
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": duration_seconds,
                "command": canonical_command,
                "shell_replay_command": canonical_shell_replay,
                "git_at_start": git_provenance,
                "runtime": runtime_provenance,
            },
        }
        status = {
            "schema_version": 1,
            "status": summary["status"],
            "engineering_execution_complete": all(engineering_checks.values()),
            "scientific_audit_complete": False,
            "scientific_completion": False,
            "claim_promotion_allowed": False,
        }
        llm_context = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "one_sentence_result": (
                "Bounded within-recording P/U feature-fusion screen completed; "
                f"advance verdict={summary['advance_verdict']}, scientific promotion=false."
            ),
            "stage": summary["stage"],
            "applicability": summary["applicability"],
            "operating_point": summary["operating_point"],
            "cohort": {
                "primary_positive_anchors": summary["primary_positive_anchors"],
                "primary_unlabeled_candidates": summary["primary_unlabeled_candidates"],
                "unreserved_proposal_stage_misses": summary[
                    "unreserved_proposal_stage_misses"
                ],
                "reserved_proposal_stage_misses": summary[
                    "reserved_proposal_stage_misses"
                ],
            },
            "primary_macro_fold_spu_auc_by_method": summary[
                "primary_macro_fold_spu_auc_by_method"
            ],
            "primary_budget_58_metrics_by_method": summary[
                "primary_budget_58_metrics_by_method"
            ],
            "tiny_mlp_cluster_bootstrap_contrasts": summary[
                "tiny_mlp_cluster_bootstrap_contrasts"
            ],
            "positive_containing_leakage_components_by_fold": summary[
                "positive_containing_leakage_components_by_fold"
            ],
            "bootstrap_interval_support_caveat": summary[
                "bootstrap_interval_support_caveat"
            ],
            "tiny_mlp_representative_component_null_p": summary[
                "tiny_mlp_representative_component_null_p"
            ],
            "tiny_mlp_seed_median_pairwise_spearman": summary[
                "tiny_mlp_seed_median_pairwise_spearman"
            ],
            "tiny_mlp_seed_recall_at_58_range": summary[
                "tiny_mlp_seed_recall_at_58_range"
            ],
            "advance_verdict": summary["advance_verdict"],
            "read_first": [
                "summary.json",
                "REPORT.md",
                "validation.json",
                "advance_signals.json",
                "artifact_index.json",
            ],
            "key_tables": [
                "tables/aggregate_metrics.tsv",
                "tables/per_fold_metrics.tsv",
                "tables/oof_scores.tsv",
                "tables/fold_summary.tsv",
                "tables/permutation_controls.tsv",
            ],
            "reproducibility": [
                "execution_provenance.json",
                "resolved_config.json",
                "artifact_index.json",
            ],
            "conventions": {
                "U": "unknown_unlabeled_not_verified_negative",
                "N/A": "not_applicable_or_not_evaluated_in_this_bounded_screen",
                "global_budget_auc": "fold_percentile_sensitivity_not_primary_macro_fold_auc",
            },
            "do_not_claim": ["verified negatives", "precision", "calibrated neuron probability", "independent-recording generalization", "scientific completion"],
        }
        _write_json(partial / "evaluation.json", evaluation)
        _write_json(partial / "solver_completion.json", solver_completion)
        _write_json(partial / "resolved_config.json", resolved_config)
        _write_json(partial / "fold_plan.json", fold_plan)
        _write_json(partial / "bootstrap.json", bootstrap)
        _write_json(partial / "advance_signals.json", advance_signals)
        _write_json(partial / "seed_stability.json", seed_stability)
        _write_json(
            partial / "locked_audits.json",
            {
                "final_fit": audits["final_fit"],
                "review_site_count": len(audits["site_rows"]),
                "review_sites_with_confirmed_canonical_overlap": sum(
                    row["canonical_v7_confirmed_overlap"] for row in audits["site_rows"]
                ),
                "review_sites_with_cross_source_disagreement": sum(
                    row["cross_source_disagreement"] for row in audits["site_rows"]
                ),
                "quiet_null_candidate_count": len(audits["quiet_rows"]),
                "panel_values_used_for_selection_or_tuning": False,
            },
        )
        _write_json(partial / "execution_provenance.json", provenance)
        _write_json(partial / "resource_usage.json", provenance["resources"])
        _write_json(partial / "validation.json", validation)
        _write_json(partial / "summary.json", summary)
        _write_json(partial / "llm_context.json", llm_context)
        (partial / "REPORT.md").write_text(report_text, encoding="utf-8")
        _write_json(partial / "status.json", status)
        # Cleanup that can fail happens before the tree is indexed or renamed.
        thread_limit_restore_attempted = True
        thread_limiter.restore_original_limits()
        _progress(progress, "artifacts_written")
        final_result: dict[str, Any] = {}

        def commit_validated_partial(index: Mapping[str, Any]) -> None:
            elapsed_to_commit = (
                datetime.now(timezone.utc) - started_dt
            ).total_seconds()
            if elapsed_to_commit > float(
                config["resources"]["maximum_end_to_end_wall_minutes"]
            ) * 60.0:
                raise LearningDeadlineExceeded(
                    "end-to-end process deadline exceeded before atomic commit"
                )
            final_result.update(
                {
                    **summary,
                    "artifact_count": int(index["artifact_count"]),
                    "output_root": OUTPUT_RELATIVE_PATH.as_posix(),
                }
            )
            promotion_deadline.disarm_for_atomic_commit()

        # No failure-prone operation follows the single same-filesystem rename.
        finalize_atomic_run_tree(
            partial,
            output,
            before_atomic_rename=commit_validated_partial,
        )
        return final_result
    except Exception as exc:
        if not thread_limit_restore_attempted:
            thread_limit_restore_attempted = True
            try:
                thread_limiter.restore_original_limits()
            except Exception:
                pass
        if partial.exists():
            try:
                _write_json(
                    partial / "status.json",
                    {
                        "schema_version": 1,
                        "status": "failed_incomplete_partial",
                        "error_type": type(exc).__name__,
                        "scientific_audit_complete": False,
                        "scientific_completion": False,
                        "claim_promotion_allowed": False,
                    },
                )
            except Exception:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=CONFIG_RELATIVE_PATH)
        child.add_argument("--repository-root", type=Path)
        child.add_argument("--data-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    function = preflight_uncertainty_aware_learning if args.command == "preflight" else run_uncertainty_aware_learning
    result = function(args.config, repository_root=args.repository_root, data_root=args.data_root)
    json.dump(_json_safe(result), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LearningContractError",
    "adjust_recall_for_proposal_misses",
    "build_contiguous_x_block_folds",
    "classify_proposal_stage_misses",
    "aggregate_review_site_audit",
    "compute_advance_signal_panel",
    "finalize_atomic_run_tree",
    "fold_stratified_representative_unit_score_permutation_null",
    "grouped_paired_bootstrap_macro_fold_auc",
    "main",
    "materialize_frozen_outer_folds",
    "preflight_uncertainty_aware_learning",
    "run_uncertainty_aware_learning",
    "write_and_validate_artifact_index",
]
