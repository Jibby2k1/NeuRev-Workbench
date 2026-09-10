"""Read-only-verified salvage finalizer for the protected adjacent-frame run.

The original executor completed every label-free fit and candidate cell, sealed
the candidate universe, and then failed inside the scalar clustered bootstrap.
This module does not refit, rescore, or mutate a sealed table.  It verifies the
stable partial artifact, reconstructs the v1/v7 joins, evaluates the identical
26-identity bootstrap with vectorized integer sufficient statistics, writes the
remaining derived artifacts, and atomically promotes the work directory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import GammaLSDifferenceConfig
from .evaluation import CANDIDATE_BUDGETS_PER_BURST, NMS_DISTANCE_PX
from . import protected as protected_module
from .protected import (
    ALL_ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FIXED_ARMS,
    QUIET_SWAPS,
    _artifact_index,
    _atomic_json,
    _atomic_tsv,
    _read_sparse_positives,
    _summary_by_arm,
    aggregate_match_rows,
    observation_match_rows,
)


EXPECTED_CONTEXT_ROLES = (
    "larger_support_comparator",
    "original_screen_context",
    "size_sufficient_candidate",
)
EXPECTED_NMS_DISTANCES = (4, 6, 8)
EXPECTED_QUIET_BURDENS = (0.25, 0.5, 1.0, 2.0, 5.0)
EXPECTED_FIT_MODEL_COUNT = 36
EXPECTED_FIT_GRID_ROWS = 108
EXPECTED_SELECTED_FIT_ROWS = 24
EXPECTED_CANDIDATE_ROWS = 7278
EXPECTED_THRESHOLD_ROWS = 2160
EXPECTED_TIMING_ROWS = 72
EXPECTED_EQUIVALENCE_ROWS = 36
EXPECTED_V1_ROWS = 79
EXPECTED_V1_IDENTITIES = 26
EXPECTED_V7_ROWS = 106
EXPECTED_V7_IDENTITIES = 44
ORIGINAL_FAILURE_FRAGMENT = (
    "unsupported operand type(s) for *: 'type' and 'bool'"
)


class ProtectedFinalizeUnavailable(RuntimeError):
    """Raised when the partial artifact cannot be safely finalized."""


@dataclass(frozen=True)
class VerifiedPartial:
    work: Path
    destination: Path
    contract: Mapping[str, Any]
    preflight: Mapping[str, Any]
    candidate_seal: Mapping[str, Any]
    prior_status: Mapping[str, Any]
    fit_rows: Sequence[Mapping[str, str]]
    selected_rows: Sequence[Mapping[str, str]]
    candidate_rows: Sequence[Mapping[str, str]]
    calibration_rows: Sequence[Mapping[str, str]]
    timing_rows: Sequence[Mapping[str, str]]
    equivalence_rows: Sequence[Mapping[str, str]]
    upstream_sha256: Mapping[str, str]
    sample_identity_audit: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ProtectedFinalizeUnavailable(f"sealed table is empty: {path.name}")
    return rows


def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ProtectedFinalizeUnavailable(
        f"field {field} is not a strict serialized boolean: {value!r}"
    )


def _original_scalar_failure(status: Mapping[str, Any]) -> str | None:
    """Recover the immutable upstream failure through a finalizer retry."""

    if status.get("status") == "interrupted_or_failed_resumable":
        candidate = str(status.get("error", ""))
    elif status.get("status") == "finalizer_failed_resumable":
        candidate = str(status.get("original_failure", ""))
    else:
        return None
    return candidate if ORIGINAL_FAILURE_FRAGMENT in candidate else None


def _exact_values(
    rows: Sequence[Mapping[str, Any]], field: str, expected: Sequence[Any]
) -> None:
    observed = {str(row[field]) for row in rows}
    wanted = {str(value) for value in expected}
    if observed != wanted:
        raise ProtectedFinalizeUnavailable(
            f"{field} changed: observed={sorted(observed)}, expected={sorted(wanted)}"
        )


def _bootstrap_weight_matrix(
    identities: Sequence[str], *, seed: int, replicates: int
) -> np.ndarray:
    """Reproduce the scalar executor's RNG calls exactly, then vectorize."""

    identity_values = tuple(str(value) for value in identities)
    if not identity_values or int(replicates) < 1:
        raise ValueError("bootstrap identities and replicates must be non-empty")
    index = {identity: position for position, identity in enumerate(identity_values)}
    if len(index) != len(identity_values):
        raise ValueError("bootstrap identities must be unique")
    rng = np.random.default_rng(int(seed))
    weights = np.zeros(
        (int(replicates), len(identity_values)), dtype=np.float64
    )
    # Keep the original one-rng-choice-per-replicate call pattern.  Each row is
    # still only 26 draws, so this loop is negligible; all expensive metric
    # evaluation below is vectorized over the resulting weight matrix.
    for replicate in range(int(replicates)):
        sampled = rng.choice(
            identity_values,
            size=len(identity_values),
            replace=True,
        )
        for identity in sampled:
            weights[replicate, index[str(identity)]] += 1.0
    if not np.all(weights.sum(axis=1) == len(identity_values)):
        raise AssertionError("cluster bootstrap weights do not sum to 26")
    return weights


def _dimension_metric_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    identities: Sequence[str],
    bootstrap_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return observed/bootstrap AUC, B58, and observed burst-B58 arrays."""

    arm_index = {str(arm): position for position, arm in enumerate(arms)}
    identity_index = {
        str(identity): position for position, identity in enumerate(identities)
    }
    budgets = tuple(int(value) for value in CANDIDATE_BUDGETS_PER_BURST)
    budget_index = {value: position for position, value in enumerate(budgets)}
    counts = np.zeros(
        (len(arms), len(budgets), 4, len(identities)), dtype=np.float64
    )
    matched = np.zeros_like(counts)
    for row in rows:
        arm = str(row["representation"])
        if arm not in arm_index:
            continue
        budget = int(row["candidate_budget"])
        burst = int(row["burst_id"])
        identity = str(row["canonical_roi_id"])
        if budget not in budget_index or burst not in (1, 2, 3, 4):
            raise ValueError("match row escaped the frozen budget/burst contract")
        if identity not in identity_index:
            raise ValueError("match row contains an unknown bootstrap identity")
        matched_value = row["matched"]
        if not isinstance(matched_value, (bool, np.bool_)):
            raise TypeError(
                "vectorized bootstrap requires in-memory boolean match rows"
            )
        key = (
            arm_index[arm],
            budget_index[budget],
            burst - 1,
            identity_index[identity],
        )
        counts[key] += 1.0
        matched[key] += float(bool(matched_value))

    if np.any(counts.sum(axis=-1) == 0.0):
        raise ValueError("bootstrap dimension is missing an arm/budget/burst cell")

    def evaluate(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count_flat = counts.transpose(3, 0, 1, 2).reshape(len(identities), -1)
        match_flat = matched.transpose(3, 0, 1, 2).reshape(len(identities), -1)
        denominators = (weights @ count_flat).reshape(
            len(weights), len(arms), len(budgets), 4
        )
        numerators = (weights @ match_flat).reshape(denominators.shape)
        recalls = np.full_like(numerators, np.nan)
        np.divide(
            numerators,
            denominators,
            out=recalls,
            where=denominators != 0.0,
        )
        with np.errstate(invalid="ignore"):
            macro = np.nanmean(recalls, axis=-1)
        x = np.asarray(budgets, dtype=np.float64)
        auc = np.trapezoid(macro, x=x, axis=-1) / (x[-1] - x[0])
        b58_index = budget_index[58]
        return auc, macro[:, :, b58_index], recalls[:, :, b58_index, :]

    unit = np.ones((1, len(identities)), dtype=np.float64)
    observed_auc, observed_b58, observed_bursts = evaluate(unit)
    bootstrap_auc, bootstrap_b58, _ = evaluate(bootstrap_weights)
    return (
        observed_auc[0],
        observed_b58[0],
        observed_bursts[0],
        bootstrap_auc,
        bootstrap_b58,
        counts,
    )


def vectorized_clustered_bootstrap_contrasts(
    v1_match_rows: Sequence[Mapping[str, Any]],
    *,
    learned_arm: str = "cs_parzen_two_frame",
    controls: Sequence[str] = FIXED_ARMS,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[dict[str, Any]]:
    """Exact sufficient-statistic equivalent of the protected scalar bootstrap."""

    controls_tuple = tuple(str(value) for value in controls)
    if str(learned_arm) in controls_tuple or len(set(controls_tuple)) != len(
        controls_tuple
    ):
        raise ValueError("bootstrap controls must be unique and exclude learned arm")
    identities = sorted(
        {str(row["canonical_roi_id"]) for row in v1_match_rows}
    )
    if len(identities) != EXPECTED_V1_IDENTITIES:
        raise ValueError(
            f"protected v1 bootstrap requires 26 identities, got {len(identities)}"
        )
    bootstrap_weights = _bootstrap_weight_matrix(
        identities, seed=int(seed), replicates=int(replicates)
    )
    base_groups: dict[tuple[str, str, int, float], list[Mapping[str, Any]]] = {}
    pooled_groups: dict[tuple[str, int, float], list[Mapping[str, Any]]] = {}
    for row in v1_match_rows:
        context = str(row["context_role"])
        swap = str(row["quiet_swap"])
        nms = int(row["nms_distance_px"])
        target = float(row["target_nms_peaks_per_pseudo_burst"])
        base_groups.setdefault((context, swap, nms, target), []).append(row)
        pooled_groups.setdefault((context, nms, target), []).append(row)
    base_dimensions = sorted(base_groups)
    pooled_dimensions = [
        (context, "crossfit_average", nms, target)
        for context, nms, target in sorted(pooled_groups)
    ]
    dimensions = base_dimensions + pooled_dimensions
    arms = (str(learned_arm), *controls_tuple)
    output: list[dict[str, Any]] = []
    for context, swap, nms, target in dimensions:
        subset = (
            pooled_groups[(context, nms, target)]
            if swap == "crossfit_average"
            else base_groups[(context, swap, nms, target)]
        )
        (
            observed_auc,
            observed_b58,
            observed_bursts,
            bootstrap_auc,
            bootstrap_b58,
            _,
        ) = _dimension_metric_arrays(
            subset,
            arms=arms,
            identities=identities,
            bootstrap_weights=bootstrap_weights,
        )
        for control_position, control in enumerate(controls_tuple, start=1):
            auc_deltas = bootstrap_auc[:, 0] - bootstrap_auc[:, control_position]
            b58_deltas = (
                bootstrap_b58[:, 0]
                - bootstrap_b58[:, control_position]
            )
            learned_bursts = observed_bursts[0]
            control_bursts = observed_bursts[control_position]
            output.append(
                {
                    "context_role": context,
                    "quiet_swap": swap,
                    "nms_distance_px": nms,
                    "nms_role": (
                        "primary"
                        if nms == NMS_DISTANCE_PX
                        else "descriptive_sensitivity"
                    ),
                    "target_nms_peaks_per_pseudo_burst": target,
                    "learned_representation": str(learned_arm),
                    "control_representation": control,
                    "learned_budget_auc": float(observed_auc[0]),
                    "control_budget_auc": float(
                        observed_auc[control_position]
                    ),
                    "budget_auc_delta": float(
                        observed_auc[0] - observed_auc[control_position]
                    ),
                    "budget_auc_delta_ci95_low": float(
                        np.nanpercentile(auc_deltas, 2.5)
                    ),
                    "budget_auc_delta_ci95_high": float(
                        np.nanpercentile(auc_deltas, 97.5)
                    ),
                    "learned_b58_macro_recall": float(observed_b58[0]),
                    "control_b58_macro_recall": float(
                        observed_b58[control_position]
                    ),
                    "b58_macro_recall_delta": float(
                        observed_b58[0] - observed_b58[control_position]
                    ),
                    "b58_delta_ci95_low": float(
                        np.nanpercentile(b58_deltas, 2.5)
                    ),
                    "b58_delta_ci95_high": float(
                        np.nanpercentile(b58_deltas, 97.5)
                    ),
                    "b58_burst_wins": int(
                        np.sum(learned_bursts > control_bursts)
                    ),
                    "b58_burst_ties": int(
                        np.sum(learned_bursts == control_bursts)
                    ),
                    "cluster_field": "canonical_roi_id",
                    "cluster_count": EXPECTED_V1_IDENTITIES,
                    "bootstrap_seed": int(seed),
                    "bootstrap_replicates": int(replicates),
                    "v7_inferential_claim": False,
                }
            )
    return output


def _upstream_hashes(work: Path, fit_files: Sequence[Path]) -> dict[str, str]:
    relative = (
        "run_contract.json",
        "candidate_seal.json",
        "candidates_label_sealed.tsv",
        "threshold_calibration.tsv",
        "fit_grid.tsv",
        "selected_fit_rows.tsv",
        "timings.tsv",
        "representation_equivalence.tsv",
    )
    result = {name: _sha256(work / name) for name in relative}
    for path in fit_files:
        result[path.relative_to(work).as_posix()] = _sha256(path)
    return dict(sorted(result.items()))


def _audit_fit_models(
    work: Path,
    contract: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, str]],
    selected_rows: Sequence[Mapping[str, str]],
) -> tuple[list[Path], dict[str, Any]]:
    fit_files = sorted((work / "fit_models").glob("*.json"))
    if len(fit_files) != EXPECTED_FIT_MODEL_COUNT:
        raise ProtectedFinalizeUnavailable(
            f"expected 36 fit checkpoints, found {len(fit_files)}"
        )
    bandwidths = tuple(float(value) for value in contract["fit_grid"]["bandwidths"])
    seeds = tuple(int(value) for value in contract["fit_grid"]["sample_seeds"])
    expected_keys = {
        (fold, seed, bandwidth)
        for fold in (1, 2, 3, 4)
        for seed in seeds
        for bandwidth in bandwidths
    }
    manifests: dict[tuple[int, int, float], Mapping[str, Any]] = {}
    hashes_by_fold_seed: dict[tuple[int, int], set[str]] = {}
    manifest_by_fold_seed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for path in fit_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            int(payload["training_fold"]),
            int(payload["sample_seed"]),
            float(payload["bandwidth"]),
        )
        if key not in expected_keys or key in manifests:
            raise ProtectedFinalizeUnavailable(
                f"fit checkpoint key is duplicate or unexpected: {key}"
            )
        manifest = payload["sample_manifest"]
        fit = payload["fit"]
        if (
            manifest.get("positive_coordinates_used") is not False
            or manifest.get("positive_identities_used") is not False
            or manifest.get("pair_guard_rule")
            != "neither_previous_nor_current_frame_touches_inclusive_guard"
            or int(manifest.get("sample_count", -1)) != 4096
            or fit.get("positive_coordinates_used") is not False
            or fit.get("positive_identities_used") is not False
            or fit.get("archived_fit_loaded") is not False
            or fit.get("fit_scope")
            != "outer_training_fold_excluding_heldout_burst_plus_guard"
            or fit.get("full_rank_no_components_discarded") is not True
            or int(fit.get("whitening_rank", -1)) != 2
        ):
            raise ProtectedFinalizeUnavailable(
                f"fit checkpoint violates leakage/rank contract: {path.name}"
            )
        sample_hash = str(manifest["sample_identity_sha256"])
        pair = key[:2]
        hashes_by_fold_seed.setdefault(pair, set()).add(sample_hash)
        manifest_by_fold_seed.setdefault(pair, manifest)
        manifests[key] = manifest
    if set(manifests) != expected_keys or any(
        len(values) != 1 for values in hashes_by_fold_seed.values()
    ):
        raise ProtectedFinalizeUnavailable(
            "bandwidth-specific fits did not reuse one sample per fold/seed"
        )

    for row in fit_rows:
        key = (
            int(row["training_fold"]),
            int(row["sample_seed"]),
            float(row["bandwidth"]),
        )
        if str(row["sample_identity_sha256"]) != str(
            manifests[key]["sample_identity_sha256"]
        ):
            raise ProtectedFinalizeUnavailable(
                "fit-grid sample hash does not match its model checkpoint"
            )
    for row in selected_rows:
        fold_seed = (int(row["training_fold"]), int(row["sample_seed"]))
        if str(row["sample_identity_sha256"]) not in hashes_by_fold_seed[fold_seed]:
            raise ProtectedFinalizeUnavailable(
                "selected-fit sample hash does not match its fold/seed checkpoint"
            )

    physical_ambiguities = []
    pairs = sorted(manifest_by_fold_seed)
    for left_index, left in enumerate(pairs):
        left_manifest = manifest_by_fold_seed[left]
        for right in pairs[left_index + 1 :]:
            right_manifest = manifest_by_fold_seed[right]
            if (
                str(left_manifest["sample_identity_sha256"])
                == str(right_manifest["sample_identity_sha256"])
                and list(left_manifest["heldout_guard_ui"])
                != list(right_manifest["heldout_guard_ui"])
            ):
                physical_ambiguities.append(
                    {
                        "left_fold_seed": list(left),
                        "right_fold_seed": list(right),
                        "shared_ordinal_sha256": str(
                            left_manifest["sample_identity_sha256"]
                        ),
                        "left_guard_ui": list(left_manifest["heldout_guard_ui"]),
                        "right_guard_ui": list(right_manifest["heldout_guard_ui"]),
                    }
                )
    unique_hashes = {
        str(manifest["sample_identity_sha256"])
        for manifest in manifest_by_fold_seed.values()
    }
    audit = {
        "fold_seed_samples": len(manifest_by_fold_seed),
        "unique_sample_identity_sha256_values": len(unique_hashes),
        "bandwidth_reuse_consistent": True,
        "hash_payload_semantics": (
            "sorted ordinal indices into each fold-specific eligible-frame by "
            "anatomy-pixel population"
        ),
        "hash_includes_resolved_frame_y_x_coordinates": False,
        "coordinate_complete_identity_provenance": False,
        "cross_fold_same_hash_different_guard_examples": physical_ambiguities,
        "interpretation": (
            "hash consistency is valid within a fold/seed, but equal hashes "
            "across fold-specific populations do not prove equal physical samples"
        ),
    }
    if len(manifest_by_fold_seed) != 12 or not physical_ambiguities:
        raise ProtectedFinalizeUnavailable(
            "expected sample-identity ordinal-hash ambiguity was not reproduced"
        )
    return fit_files, audit


def verify_sealed_partial(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    destination: str | Path,
    expected_candidate_seal_sha256: str,
) -> VerifiedPartial:
    """Verify every stable upstream byte before opening sparse-positive rows."""

    destination_path = Path(destination).expanduser().resolve()
    work = destination_path.parent / f".{destination_path.name}.protected-work"
    if destination_path.exists():
        raise FileExistsError(f"protected destination already exists: {destination_path}")
    if not work.is_dir():
        raise FileNotFoundError(f"protected partial is missing: {work}")
    partial_files = sorted(work.rglob("*.partial"))
    if partial_files:
        raise ProtectedFinalizeUnavailable(
            f"partial contains unfinished atomic files: {partial_files}"
        )
    required = (
        "run_contract.json",
        "candidate_seal.json",
        "candidates_label_sealed.tsv",
        "threshold_calibration.tsv",
        "fit_grid.tsv",
        "selected_fit_rows.tsv",
        "timings.tsv",
        "representation_equivalence.tsv",
        "heartbeat.json",
        "status.json",
    )
    missing = [name for name in required if not (work / name).is_file()]
    if missing:
        raise ProtectedFinalizeUnavailable(f"partial lacks required files: {missing}")

    prior_status = json.loads((work / "status.json").read_text(encoding="utf-8"))
    original_failure = _original_scalar_failure(prior_status)
    if original_failure is None:
        raise ProtectedFinalizeUnavailable(
            "partial is not stably stopped at the known scalar-bootstrap failure"
        )
    contract = json.loads((work / "run_contract.json").read_text(encoding="utf-8"))
    if (
        contract.get("run_type")
        != "protected_outer_fold_two_frame_representation"
        or int(contract.get("fit_grid", {}).get("fits", -1))
        != EXPECTED_FIT_MODEL_COUNT
        or int(contract.get("bootstrap", {}).get("replicates", -1))
        != BOOTSTRAP_REPLICATES
        or int(contract.get("bootstrap", {}).get("seed", -1))
        != BOOTSTRAP_SEED
        or contract.get("archived_fits_role") != "parity_only_not_loaded"
        or contract.get("portable_config_sha256")
        != _canonical_sha256(config.portable_dict())
    ):
        raise ProtectedFinalizeUnavailable("run contract does not match this finalizer")
    executor_path = Path(protected_module.__file__).resolve()
    if _sha256(executor_path) != str(contract["protected_executor_sha256"]):
        raise ProtectedFinalizeUnavailable(
            "protected.py differs from the executor that sealed the partial"
        )

    preflight_path = Path(preflight_dir).expanduser().resolve() / "preflight.json"
    if _sha256(preflight_path) != str(contract["preflight_sha256"]):
        raise ProtectedFinalizeUnavailable("preflight bytes differ from run contract")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "ready"
        or preflight.get("gpu_run_ready") is not True
        or preflight["source"]["movie"]["sha256"] != contract["movie_sha256"]
    ):
        raise ProtectedFinalizeUnavailable("frozen preflight is not ready or aligned")
    for source_key in ("protected_labels_v1", "latest_labels_v7"):
        if _sha256(config.source_paths[source_key]) != str(
            preflight["source"][source_key]["sha256"]
        ):
            raise ProtectedFinalizeUnavailable(
                f"label source changed since preflight: {source_key}"
            )
    context = contract["context_selection"]
    context_path = Path(context["root"]) / str(context["source_file"])
    if _sha256(context_path) != str(context["source_sha256"]):
        raise ProtectedFinalizeUnavailable("fold-context source changed")

    candidate_seal_path = work / "candidate_seal.json"
    candidate_seal_sha256 = _sha256(candidate_seal_path)
    if candidate_seal_sha256 != str(expected_candidate_seal_sha256):
        raise ProtectedFinalizeUnavailable(
            "candidate-seal JSON hash differs from the operator-supplied hash"
        )
    candidate_seal = json.loads(candidate_seal_path.read_text(encoding="utf-8"))
    candidate_path = work / str(candidate_seal["candidate_table"]["path"])
    if (
        candidate_path != work / "candidates_label_sealed.tsv"
        or _sha256(candidate_path) != candidate_seal["candidate_table"]["sha256"]
        or int(candidate_seal["candidate_table"]["rows"])
        != EXPECTED_CANDIDATE_ROWS
        or _sha256(work / "threshold_calibration.tsv")
        != candidate_seal["threshold_table_sha256"]
        or _sha256(work / "fit_grid.tsv")
        != candidate_seal["fit_grid_table_sha256"]
        or _sha256(work / "selected_fit_rows.tsv")
        != candidate_seal["selected_fit_table_sha256"]
        or candidate_seal.get("sparse_positive_fields_parsed_before_seal")
        is not False
        or candidate_seal.get("source_files_hashed_by_preflight_before_seal")
        is not True
        or candidate_seal.get("positive_coordinates_used") is not False
        or candidate_seal.get("positive_identities_used") is not False
        or candidate_seal.get("unmatched_candidates") != "unknown_not_negative"
    ):
        raise ProtectedFinalizeUnavailable("candidate seal or a sealed table changed")
    heartbeat = json.loads((work / "heartbeat.json").read_text(encoding="utf-8"))
    sealed_at = datetime.fromisoformat(str(candidate_seal["sealed_at_utc"]))
    label_join_at = datetime.fromisoformat(str(heartbeat["updated_at_utc"]))
    if heartbeat.get("stage") != "protected_label_join" or sealed_at > label_join_at:
        raise ProtectedFinalizeUnavailable(
            "candidate seal does not precede the recorded label-join heartbeat"
        )

    fit_rows = _read_tsv(work / "fit_grid.tsv")
    selected_rows = _read_tsv(work / "selected_fit_rows.tsv")
    candidate_rows = _read_tsv(work / "candidates_label_sealed.tsv")
    calibration_rows = _read_tsv(work / "threshold_calibration.tsv")
    timing_rows = _read_tsv(work / "timings.tsv")
    equivalence_rows = _read_tsv(work / "representation_equivalence.tsv")
    expected_counts = (
        (fit_rows, EXPECTED_FIT_GRID_ROWS, "fit-grid"),
        (selected_rows, EXPECTED_SELECTED_FIT_ROWS, "selected-fit"),
        (candidate_rows, EXPECTED_CANDIDATE_ROWS, "candidate"),
        (calibration_rows, EXPECTED_THRESHOLD_ROWS, "threshold"),
        (timing_rows, EXPECTED_TIMING_ROWS, "timing"),
        (equivalence_rows, EXPECTED_EQUIVALENCE_ROWS, "equivalence"),
    )
    for rows, expected, label in expected_counts:
        if len(rows) != expected:
            raise ProtectedFinalizeUnavailable(
                f"{label} row count changed: {len(rows)} != {expected}"
            )
    _exact_values(fit_rows, "context_role", EXPECTED_CONTEXT_ROLES)
    _exact_values(calibration_rows, "context_role", EXPECTED_CONTEXT_ROLES)
    _exact_values(calibration_rows, "representation", ALL_ARMS)
    _exact_values(calibration_rows, "quiet_swap", QUIET_SWAPS)
    _exact_values(calibration_rows, "nms_distance_px", EXPECTED_NMS_DISTANCES)
    _exact_values(
        calibration_rows,
        "target_nms_peaks_per_pseudo_burst",
        EXPECTED_QUIET_BURDENS,
    )
    if any(
        row["interpretation_before_label_join"] != "unknown_candidate"
        for row in candidate_rows
    ) or any(
        field in candidate_rows[0]
        for field in ("observation_id", "canonical_roi_id", "matched")
    ):
        raise ProtectedFinalizeUnavailable(
            "candidate table contains a label-derived field or interpretation"
        )
    for row in calibration_rows:
        if (
            _strict_bool(
                row["probability_of_false_alarm_claimed"],
                field="probability_of_false_alarm_claimed",
            )
            or _strict_bool(
                row["positive_coordinates_used"],
                field="positive_coordinates_used",
            )
            or _strict_bool(
                row["positive_identities_used"],
                field="positive_identities_used",
            )
        ):
            raise ProtectedFinalizeUnavailable(
                "threshold table violates its label/PFA boundary"
            )
    fit_files, sample_identity_audit = _audit_fit_models(
        work, contract, fit_rows, selected_rows
    )
    upstream = _upstream_hashes(work, fit_files)
    return VerifiedPartial(
        work=work,
        destination=destination_path,
        contract=contract,
        preflight=preflight,
        candidate_seal=candidate_seal,
        prior_status=prior_status,
        fit_rows=fit_rows,
        selected_rows=selected_rows,
        candidate_rows=candidate_rows,
        calibration_rows=calibration_rows,
        timing_rows=timing_rows,
        equivalence_rows=equivalence_rows,
        upstream_sha256=upstream,
        sample_identity_audit=sample_identity_audit,
    )


def _runtime_from_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    gpu = preflight["resources"]["gpu"]
    return {
        "source": "frozen_host_visible_preflight",
        "requested_device": gpu["requested_device"],
        "device_name": gpu["device_name"],
        "torch_version": gpu["torch_version"],
        "torch_cuda_build": gpu["torch_cuda_build"],
        "nvidia_smi": gpu["nvidia_smi"],
        "original_runtime_object_not_preserved_before_interruption": True,
    }


def finalize_protected_partial(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    output_dir: str | Path,
    expected_candidate_seal_sha256: str,
) -> dict[str, Any]:
    """Finalize the known stable partial and atomically promote it."""

    verified = verify_sealed_partial(
        config,
        preflight_dir=preflight_dir,
        destination=output_dir,
        expected_candidate_seal_sha256=expected_candidate_seal_sha256,
    )
    work = verified.work
    try:
        movie_shape = tuple(
            int(value) for value in verified.preflight["source"]["movie"]["shape"][1:]
        )
        # This is the finalizer's first sparse-positive parse, after all sealed
        # hashes and the original seal-before-join heartbeat were verified.
        v1 = _read_sparse_positives(
            config.source_paths["protected_labels_v1"],
            selector="include_inclusive",
            expected_rows=EXPECTED_V1_ROWS,
            movie_shape_yx=movie_shape,
        )
        v7 = _read_sparse_positives(
            config.source_paths["latest_labels_v7"],
            selector="include_confirmed",
            expected_rows=EXPECTED_V7_ROWS,
            movie_shape_yx=movie_shape,
        )
        v1_identities = {str(row["canonical_roi_id"]) for row in v1}
        v7_identities = {str(row["canonical_roi_id"]) for row in v7}
        if len(v1_identities) != EXPECTED_V1_IDENTITIES:
            raise ProtectedFinalizeUnavailable("v1 identity denominator changed")
        if len(v7_identities) != EXPECTED_V7_IDENTITIES:
            raise ProtectedFinalizeUnavailable("v7 identity denominator changed")

        v1_matches = observation_match_rows(
            verified.candidate_rows,
            v1,
            cohort="protected_v1",
            operating_rows=verified.calibration_rows,
        )
        v7_matches = observation_match_rows(
            verified.candidate_rows,
            v7,
            cohort="latest_v7_sensitivity",
            operating_rows=verified.calibration_rows,
        )
        v1_aggregate = aggregate_match_rows(v1_matches)
        v7_aggregate = aggregate_match_rows(v7_matches)
        controls = (
            *FIXED_ARMS,
            "pca_whitened_derivative",
            "pca_matched_to_selected_ica",
        )
        contrasts = vectorized_clustered_bootstrap_contrasts(
            v1_matches,
            controls=controls,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
        )
        summary_rows = _summary_by_arm(v1_matches)
        v7_summary_rows = _summary_by_arm(v7_matches)

        expected_v1_match_rows = (
            EXPECTED_V1_ROWS
            * len(EXPECTED_CONTEXT_ROLES)
            * len(ALL_ARMS)
            * len(QUIET_SWAPS)
            * len(EXPECTED_NMS_DISTANCES)
            * len(EXPECTED_QUIET_BURDENS)
            * len(CANDIDATE_BUDGETS_PER_BURST)
        )
        expected_v7_match_rows = expected_v1_match_rows // EXPECTED_V1_ROWS * len(v7)
        expected_aggregate_rows = (
            len(EXPECTED_CONTEXT_ROLES)
            * len(ALL_ARMS)
            * len(QUIET_SWAPS)
            * len(EXPECTED_NMS_DISTANCES)
            * len(EXPECTED_QUIET_BURDENS)
            * 4
            * len(CANDIDATE_BUDGETS_PER_BURST)
        )
        expected_summary_rows = (
            len(EXPECTED_CONTEXT_ROLES)
            * len(ALL_ARMS)
            * (len(QUIET_SWAPS) + 1)
            * len(EXPECTED_NMS_DISTANCES)
            * len(EXPECTED_QUIET_BURDENS)
            * len(CANDIDATE_BUDGETS_PER_BURST)
        )
        expected_contrast_rows = (
            len(EXPECTED_CONTEXT_ROLES)
            * (len(QUIET_SWAPS) + 1)
            * len(EXPECTED_NMS_DISTANCES)
            * len(EXPECTED_QUIET_BURDENS)
            * len(controls)
        )
        expected_derived_counts = {
            "v1_match_rows": expected_v1_match_rows,
            "v7_match_rows": expected_v7_match_rows,
            "v1_aggregate_rows": expected_aggregate_rows,
            "v7_aggregate_rows": expected_aggregate_rows,
            "v1_summary_rows": expected_summary_rows,
            "v7_summary_rows": expected_summary_rows,
            "bootstrap_contrast_rows": expected_contrast_rows,
        }
        observed_derived_counts = {
            "v1_match_rows": len(v1_matches),
            "v7_match_rows": len(v7_matches),
            "v1_aggregate_rows": len(v1_aggregate),
            "v7_aggregate_rows": len(v7_aggregate),
            "v1_summary_rows": len(summary_rows),
            "v7_summary_rows": len(v7_summary_rows),
            "bootstrap_contrast_rows": len(contrasts),
        }
        if observed_derived_counts != expected_derived_counts:
            raise ProtectedFinalizeUnavailable(
                "derived label-join grain changed: "
                f"{observed_derived_counts} != {expected_derived_counts}"
            )

        _atomic_tsv(work / "protected_v1_observation_matches.tsv", v1_matches)
        _atomic_tsv(work / "protected_v1_recall.tsv", v1_aggregate)
        _atomic_tsv(
            work / "protected_v1_clustered_bootstrap_contrasts.tsv",
            contrasts,
        )
        _atomic_tsv(work / "latest_v7_observation_matches.tsv", v7_matches)
        _atomic_tsv(work / "latest_v7_sensitivity.tsv", v7_aggregate)
        _atomic_tsv(work / "protected_v1_arm_summary.tsv", summary_rows)
        _atomic_tsv(work / "latest_v7_arm_summary.tsv", v7_summary_rows)

        claim_boundary = {
            "protected_v1_population": {
                "selector": "include_inclusive",
                "occurrences": len(v1),
                "canonical_identity_count": len(v1_identities),
            },
            "latest_v7_population": {
                "selector": "include_confirmed",
                "occurrences": len(v7),
                "canonical_identity_count": len(v7_identities),
                "role": "descriptive_sensitivity_not_independent_confirmation",
            },
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "archived_fit_used": False,
            "fold_local_models": True,
            "heldout_burst_and_guard_excluded_from_fit": True,
            "candidate_artifacts_sealed_before_label_join": True,
            "context_selection_used_sparse_positive_labels": False,
            "sample_identity_hash_is_coordinate_complete": False,
            "sample_identity_hash_role": "within_fold_seed_consistency_only",
            "scientific_audit_complete": False,
            "paper_promotion_ready": False,
        }
        _atomic_json(work / "claim_boundary.json", claim_boundary)

        eligible_pixels = {
            int(
                json.loads(path.read_text(encoding="utf-8"))["sample_manifest"]
                ["eligible_anatomy_pixel_count"]
            )
            for path in sorted((work / "fit_models").glob("*.json"))
        }
        if len(eligible_pixels) != 1:
            raise ProtectedFinalizeUnavailable(
                "fit checkpoints disagree on anatomy-mask population"
            )
        finalizer_sha256 = _sha256(Path(__file__).resolve())
        finalization = {
            "mode": "salvaged_after_scalar_bootstrap_type_failure",
            "original_failure": _original_scalar_failure(
                verified.prior_status
            ),
            "original_protected_executor_sha256": verified.contract[
                "protected_executor_sha256"
            ],
            "finalizer_sha256": finalizer_sha256,
            "candidate_seal_sha256": expected_candidate_seal_sha256,
            "sealed_upstream_sha256": verified.upstream_sha256,
            "vectorized_bootstrap": {
                "algorithm": "26_identity_integer_sufficient_statistics_v1",
                "rng_call_pattern": "one_default_rng_choice_call_per_replicate",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "scalar_parity_test": (
                    "tests/test_gamma_ls_difference_protected_finalize.py"
                ),
            },
            "derived_row_counts": observed_derived_counts,
            "sample_identity_audit": verified.sample_identity_audit,
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(work / "finalization_provenance.json", finalization)

        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "protected_outer_fold_two_frame_representation",
            "status": "complete_protected_metrics_scientific_audit_pending",
            "completed_at_utc": finalization["finalized_at_utc"],
            "runtime": _runtime_from_preflight(verified.preflight),
            "preprocessing_timing": {
                "status": "not_preserved_before_original_interruption",
                "candidate_timing_table_preserved": "timings.tsv",
                "paper_runtime_claim_from_this_field_allowed": False,
            },
            "anatomy_sampling": {
                "eligible_pixels": next(iter(eligible_pixels)),
                "source": "consistent_fit_checkpoint_manifests",
                "quiet_intensity_bounds_not_preserved": True,
            },
            "fit_count": EXPECTED_FIT_MODEL_COUNT,
            "fit_grid_row_count": len(verified.fit_rows),
            "selected_fit_row_count": len(verified.selected_rows),
            "context_roles": verified.contract["context_selection"]
            ["context_roles_by_fold"],
            "candidate_row_count": len(verified.candidate_rows),
            "candidate_seal": verified.candidate_seal,
            "protected_v1": {
                "occurrences": len(v1),
                "canonical_identities": len(v1_identities),
                "arm_summary": summary_rows,
                "clustered_bootstrap_contrasts": contrasts,
            },
            "latest_v7_sensitivity": {
                "occurrences": len(v7),
                "canonical_identities": len(v7_identities),
                "arm_summary": v7_summary_rows,
                "inferential_claim": False,
            },
            "finalization": finalization,
            "claim_boundary": claim_boundary,
        }
        _atomic_json(work / "summary.json", summary)
        checks = {
            "known_scalar_bootstrap_failure_captured": (
                _original_scalar_failure(verified.prior_status) is not None
            ),
            "sealed_upstream_hashes_unchanged": (
                _upstream_hashes(
                    work, sorted((work / "fit_models").glob("*.json"))
                )
                == verified.upstream_sha256
            ),
            "exact_36_cs_parzen_fits": len(
                list((work / "fit_models").glob("*.json"))
            )
            == EXPECTED_FIT_MODEL_COUNT,
            "exact_108_fit_grid_rows": len(verified.fit_rows)
            == EXPECTED_FIT_GRID_ROWS,
            "exact_24_selected_fit_rows": len(verified.selected_rows)
            == EXPECTED_SELECTED_FIT_ROWS,
            "exact_72_candidate_cells": len(verified.timing_rows)
            == EXPECTED_TIMING_ROWS,
            "exact_7278_sealed_candidate_rows": len(verified.candidate_rows)
            == EXPECTED_CANDIDATE_ROWS,
            "exact_2160_threshold_rows": len(verified.calibration_rows)
            == EXPECTED_THRESHOLD_ROWS,
            "exact_36_equivalence_rows": len(verified.equivalence_rows)
            == EXPECTED_EQUIVALENCE_ROWS,
            "candidate_seal_precedes_label_join": True,
            "protected_v1_uses_79_inclusive": len(v1) == EXPECTED_V1_ROWS,
            "protected_v1_has_26_identity_clusters": len(v1_identities)
            == EXPECTED_V1_IDENTITIES,
            "v7_uses_106_confirmed": len(v7) == EXPECTED_V7_ROWS,
            "v7_has_44_canonical_label_identities": len(v7_identities)
            == EXPECTED_V7_IDENTITIES,
            "derived_row_grain_exact": observed_derived_counts
            == expected_derived_counts,
            "vectorized_bootstrap_exact_675_rows": len(contrasts) == 675,
            "sample_identity_hash_ambiguity_explicit": (
                verified.sample_identity_audit[
                    "coordinate_complete_identity_provenance"
                ]
                is False
            ),
            "all_candidates_unknown_before_join": all(
                row["interpretation_before_label_join"] == "unknown_candidate"
                for row in verified.candidate_rows
            ),
            "precision_not_claimed": True,
            "scientific_audit_pending": True,
        }
        validation = {
            "status": (
                "passed_protected_metric_artifact_contract"
                if all(checks.values())
                else "failed_protected_metric_artifact_contract"
            ),
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "finalizer_sha256": finalizer_sha256,
        }
        _atomic_json(work / "validation.json", validation)
        if not validation["all_checks_pass"]:
            raise ProtectedFinalizeUnavailable(
                "salvage finalization validation failed closed"
            )
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": (
                    "context role x representation x quiet swap x quiet burden "
                    "x heldout burst x candidate budget"
                ),
                "candidate_seal": "candidate_seal.json",
                "protected_primary": "protected_v1_arm_summary.tsv",
                "paired_inference": (
                    "protected_v1_clustered_bootstrap_contrasts.tsv"
                ),
                "per_observation_recomputability": (
                    "protected_v1_observation_matches.tsv"
                ),
                "latest_sensitivity": "latest_v7_arm_summary.tsv",
                "context_selection": verified.contract["context_selection"],
                "finalization_provenance": "finalization_provenance.json",
                "limitations": [
                    "sparse positives do not identify precision",
                    "v7 is candidate-assisted descriptive sensitivity",
                    "scientific-audit media are still pending",
                    "six-lag fold-local ICA refitting is outside this stage",
                    (
                        "sample_identity_sha256 hashes fold-relative population "
                        "ordinals, not resolved frame/y/x coordinates"
                    ),
                    (
                        "original preprocessing timing was not preserved before "
                        "the scalar-bootstrap interruption"
                    ),
                ],
            },
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "complete_protected_metrics_scientific_audit_pending",
                "completed_at_utc": finalization["finalized_at_utc"],
                "validation_passed": True,
                "salvaged_from_error": _original_scalar_failure(
                    verified.prior_status
                ),
                "candidate_seal_sha256": expected_candidate_seal_sha256,
                "scientific_audit_complete": False,
                "paper_promotion_ready": False,
            },
        )
        _atomic_json(
            work / "heartbeat.json",
            {
                "updated_at_utc": finalization["finalized_at_utc"],
                "stage": "complete",
                "status": "complete_protected_metrics_scientific_audit_pending",
                "finalized_by": "protected_finalize.py",
            },
        )
        if (
            _upstream_hashes(
                work, sorted((work / "fit_models").glob("*.json"))
            )
            != verified.upstream_sha256
        ):
            raise ProtectedFinalizeUnavailable(
                "sealed upstream bytes changed during finalization"
            )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        index = json.loads(
            (work / "artifact_index.json").read_text(encoding="utf-8")
        )
        for artifact in index["artifacts"]:
            path = work / artifact["path"]
            if (
                path.stat().st_size != int(artifact["size_bytes"])
                or _sha256(path) != artifact["sha256"]
            ):
                raise ProtectedFinalizeUnavailable(
                    f"artifact index verification failed: {artifact['path']}"
                )
        if verified.destination.exists():
            raise FileExistsError(
                f"protected destination appeared during finalization: {verified.destination}"
            )
        work.replace(verified.destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "finalizer_failed_resumable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "original_failure": _original_scalar_failure(
                        verified.prior_status
                    ),
                    "sealed_tables_must_not_be_modified": True,
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-candidate-seal-sha256", required=True)
    arguments = parser.parse_args(argv)
    config = GammaLSDifferenceConfig.load(arguments.config)
    result = finalize_protected_partial(
        config,
        preflight_dir=arguments.preflight_dir,
        output_dir=arguments.output_dir,
        expected_candidate_seal_sha256=(
            arguments.expected_candidate_seal_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProtectedFinalizeUnavailable",
    "VerifiedPartial",
    "finalize_protected_partial",
    "main",
    "vectorized_clustered_bootstrap_contrasts",
    "verify_sealed_partial",
]
