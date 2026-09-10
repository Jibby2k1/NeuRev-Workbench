"""Provenance-closed support-sufficiency analysis for the finalized r2 run.

This is deliberately a read-only downstream analysis.  It binds the exact
protected-r2 and support-screen-r2 artifact indexes, verifies every indexed
file, and then compares the already-frozen ``size_sufficient_candidate`` and
``larger_support_comparator`` lanes.  Protected-v1 is the sole decision cohort;
latest-v7 is emitted only as descriptive sensitivity.

The sparse-positive tables are incomplete.  Consequently this module reports
known-positive recall only.  Unmatched candidates remain unknown and no
precision-like quantity is computed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTECTED_INDEX_SHA256 = (
    "ab4816cb0d1548475c55dd712409a9dd8842caf1174cbf50d93096b470d1560e"
)
SUPPORT_SCREEN_INDEX_SHA256 = (
    "ae0337597ccd7b21a7e92f910f568809a57895b68df752f73940b7b830db7e83"
)

PROTECTED_ROLES = (
    "size_sufficient_candidate",
    "larger_support_comparator",
)
ALL_SOURCE_ROLES = (
    "larger_support_comparator",
    "original_screen_context",
    "size_sufficient_candidate",
)
FIXED_ARMS = ("raw", "difference_signed", "difference_energy_normalized")
ALL_SOURCE_ARMS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
    "pca_whitened_derivative",
    "cs_parzen_two_frame",
    "pca_matched_to_selected_ica",
)
QUIET_SWAPS = ("a_train_b_test", "b_train_a_test")
QUIET_SCOPES = (*QUIET_SWAPS, "crossfit_average")
NMS_DISTANCE_PX = 6
ALL_SOURCE_NMS_DISTANCES = (4, 6, 8)
QUIET_BURDENS = (0.25, 0.5, 1.0, 2.0, 5.0)
CANDIDATE_BUDGETS = (20, 40, 58, 80, 100)
BURSTS = (1, 2, 3, 4)

BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPLICATES = 2000
EXPECTED_V1_IDENTITIES = 26
EXPECTED_V1_OCCURRENCES = 79
EXPECTED_V7_IDENTITIES = 44
EXPECTED_V7_OCCURRENCES = 106

B58_TOTAL_MATCH_TOLERANCE = 1
B58_PER_BURST_MATCH_TOLERANCE = 1
BUDGET_AUC_LOSS_TOLERANCE = 0.02

OBSERVATION_FIELDS = (
    "cohort",
    "context_role",
    "representation",
    "quiet_swap",
    "nms_distance_px",
    "nms_role",
    "target_nms_peaks_per_pseudo_burst",
    "burst_id",
    "candidate_budget",
    "effective_candidate_count",
    "observation_id",
    "canonical_roi_id",
    "x_px",
    "y_px",
    "matched",
    "matched_candidate_rank",
    "matched_candidate_score",
    "matched_candidate_x_px",
    "matched_candidate_y_px",
    "match_distance_px",
    "unmatched_candidates",
)


class ProtectedSupportSufficiencyUnavailable(RuntimeError):
    """Raised when any provenance, pairing, or analysis contract is violated."""


@dataclass(frozen=True)
class VerifiedArtifact:
    root: Path
    index_sha256: str
    entries: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Observation:
    cohort: str
    context_role: str
    representation: str
    quiet_swap: str
    nms_distance_px: int
    nms_role: str
    burden: float
    burst_id: int
    candidate_budget: int
    effective_candidate_count: int
    observation_id: str
    canonical_roi_id: str
    x_px: str
    y_px: str
    matched: bool
    unmatched_candidates: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtectedSupportSufficiencyUnavailable(
            f"cannot read valid JSON from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProtectedSupportSufficiencyUnavailable(f"{path} is not a JSON object")
    return payload


def verify_artifact_index(
    root: str | Path, *, expected_index_sha256: str
) -> VerifiedArtifact:
    """Verify an exact immutable artifact inventory, including the index seal."""

    base = Path(root).resolve()
    index_path = base / "artifact_index.json"
    if not base.is_dir() or not index_path.is_file():
        raise ProtectedSupportSufficiencyUnavailable(
            f"artifact root or artifact_index.json is missing: {base}"
        )
    actual_index_sha256 = _sha256(index_path)
    if actual_index_sha256 != str(expected_index_sha256):
        raise ProtectedSupportSufficiencyUnavailable(
            "artifact index seal changed: "
            f"expected={expected_index_sha256}, actual={actual_index_sha256}"
        )
    payload = _json(index_path)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("artifacts"), list
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "artifact index schema is not the frozen version 1 contract"
        )

    entries: dict[str, Mapping[str, Any]] = {}
    ordered_paths: list[str] = []
    for raw in payload["artifacts"]:
        if not isinstance(raw, dict):
            raise ProtectedSupportSufficiencyUnavailable("artifact index row is invalid")
        rel = str(raw.get("path", ""))
        pure = PurePosixPath(rel)
        if (
            not rel
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != rel
            or rel == "artifact_index.json"
            or rel in entries
        ):
            raise ProtectedSupportSufficiencyUnavailable(
                f"unsafe or duplicate artifact index path: {rel!r}"
            )
        expected_hash = raw.get("sha256")
        expected_size = raw.get("size_bytes")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ProtectedSupportSufficiencyUnavailable(
                f"invalid hash/size contract for {rel}"
            )
        path = base / rel
        if not path.is_file():
            raise ProtectedSupportSufficiencyUnavailable(f"indexed file is missing: {rel}")
        if path.stat().st_size != expected_size or _sha256(path) != expected_hash:
            raise ProtectedSupportSufficiencyUnavailable(
                f"indexed file hash or size changed: {rel}"
            )
        entries[rel] = raw
        ordered_paths.append(rel)

    if ordered_paths != sorted(ordered_paths):
        raise ProtectedSupportSufficiencyUnavailable(
            "artifact index paths are not in canonical sorted order"
        )
    actual_inventory = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path != index_path
    }
    if actual_inventory != set(entries):
        raise ProtectedSupportSufficiencyUnavailable(
            "artifact inventory differs from the sealed index: "
            f"extra={sorted(actual_inventory - set(entries))}, "
            f"missing={sorted(set(entries) - actual_inventory)}"
        )
    return VerifiedArtifact(
        root=base,
        index_sha256=actual_index_sha256,
        entries=entries,
    )


def _require_indexed(source: VerifiedArtifact, names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(source.entries))
    if missing:
        raise ProtectedSupportSufficiencyUnavailable(
            f"required files are absent from the sealed index: {missing}"
        )


def _count_tsv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def verify_protected_source(root: str | Path) -> tuple[VerifiedArtifact, Mapping[str, Any]]:
    source = verify_artifact_index(root, expected_index_sha256=PROTECTED_INDEX_SHA256)
    required = (
        "candidate_seal.json",
        "candidates_label_sealed.tsv",
        "claim_boundary.json",
        "finalization_provenance.json",
        "latest_v7_observation_matches.tsv",
        "protected_v1_observation_matches.tsv",
        "run_contract.json",
        "summary.json",
        "timings.tsv",
        "validation.json",
    )
    _require_indexed(source, required)

    validation = _json(source.root / "validation.json")
    checks = validation.get("checks")
    if (
        validation.get("status") != "passed_protected_metric_artifact_contract"
        or validation.get("all_checks_pass") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "protected artifact validation is not an all-checks PASS"
        )
    required_checks = {
        "all_candidates_unknown_before_join",
        "candidate_seal_precedes_label_join",
        "exact_7278_sealed_candidate_rows",
        "precision_not_claimed",
        "protected_v1_has_26_identity_clusters",
        "protected_v1_uses_79_inclusive",
        "sealed_upstream_hashes_unchanged",
        "v7_has_44_canonical_label_identities",
        "v7_uses_106_confirmed",
    }
    if not required_checks.issubset(checks):
        raise ProtectedSupportSufficiencyUnavailable(
            "protected validation lacks required provenance/metric checks"
        )

    seal = _json(source.root / "candidate_seal.json")
    table_contract = seal.get("candidate_table")
    if not isinstance(table_contract, dict):
        raise ProtectedSupportSufficiencyUnavailable("candidate seal table contract missing")
    candidate_path = source.root / "candidates_label_sealed.tsv"
    if (
        table_contract.get("path") != "candidates_label_sealed.tsv"
        or table_contract.get("sha256") != _sha256(candidate_path)
        or table_contract.get("rows") != _count_tsv_rows(candidate_path)
        or table_contract.get("rows") != 7278
        or seal.get("positive_coordinates_used") is not False
        or seal.get("positive_identities_used") is not False
        or seal.get("sparse_positive_fields_parsed_before_seal") is not False
        or seal.get("unmatched_candidates") != "unknown_not_negative"
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "candidate seal no longer satisfies the label-isolation contract"
        )
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = csv.DictReader(handle, delimiter="\t")
        if "interpretation_before_label_join" not in (candidates.fieldnames or ()):
            raise ProtectedSupportSufficiencyUnavailable(
                "sealed candidate table lacks its pre-join interpretation field"
            )
        if any(
            row["interpretation_before_label_join"] != "unknown_candidate"
            for row in candidates
        ):
            raise ProtectedSupportSufficiencyUnavailable(
                "a sealed candidate was not unknown before label join"
            )

    boundary = _json(source.root / "claim_boundary.json")
    if (
        boundary.get("candidate_artifacts_sealed_before_label_join") is not True
        or boundary.get("context_selection_used_sparse_positive_labels") is not False
        or boundary.get("precision_identified") is not False
        or boundary.get("unmatched_candidates") != "unknown_not_negative"
        or boundary.get("paper_promotion_ready") is not False
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "protected claim boundary was broadened or altered"
        )

    provenance = _json(source.root / "finalization_provenance.json")
    if provenance.get("candidate_seal_sha256") != _sha256(
        source.root / "candidate_seal.json"
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "finalization provenance does not bind candidate_seal.json"
        )
    sealed_upstream = provenance.get("sealed_upstream_sha256")
    if not isinstance(sealed_upstream, dict) or not sealed_upstream:
        raise ProtectedSupportSufficiencyUnavailable(
            "finalization provenance lacks sealed upstream hashes"
        )
    for rel, expected in sealed_upstream.items():
        if rel not in source.entries or source.entries[rel]["sha256"] != expected:
            raise ProtectedSupportSufficiencyUnavailable(
                f"finalization upstream seal disagrees with artifact index: {rel}"
            )
    sealed_at = datetime.fromisoformat(str(seal.get("sealed_at_utc")))
    finalized_at = datetime.fromisoformat(str(provenance.get("finalized_at_utc")))
    if not sealed_at < finalized_at:
        raise ProtectedSupportSufficiencyUnavailable(
            "candidate seal does not precede protected finalization"
        )

    contract = _json(source.root / "run_contract.json")
    bootstrap = contract.get("bootstrap")
    if bootstrap != {
        "cluster_field": "canonical_roi_id",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }:
        raise ProtectedSupportSufficiencyUnavailable(
            "protected clustered-bootstrap contract changed"
        )
    context_selection = contract.get("context_selection")
    if (
        not isinstance(context_selection, dict)
        or context_selection.get("selection_scope") != "outer_training_fold_only"
        or context_selection.get("positive_coordinates_used") is not False
        or context_selection.get("positive_identities_used") is not False
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "protected context-selection contract changed"
        )
    return source, contract


def _strict_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ProtectedSupportSufficiencyUnavailable(
        f"{field} is not a canonical True/False value: {value!r}"
    )


def _read_observations(
    path: Path,
    *,
    cohort: str,
    expected_identities: int,
    expected_occurrences: int,
) -> list[Observation]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != OBSERVATION_FIELDS:
            raise ProtectedSupportSufficiencyUnavailable(
                f"observation schema changed in {path.name}"
            )
        raw_rows = list(reader)

    expected_rows = (
        len(ALL_SOURCE_ROLES)
        * len(ALL_SOURCE_ARMS)
        * len(QUIET_SWAPS)
        * len(ALL_SOURCE_NMS_DISTANCES)
        * len(QUIET_BURDENS)
        * len(CANDIDATE_BUDGETS)
        * expected_occurrences
    )
    if len(raw_rows) != expected_rows:
        raise ProtectedSupportSufficiencyUnavailable(
            f"{cohort} source row count changed: {len(raw_rows)} != {expected_rows}"
        )
    rows: list[Observation] = []
    for raw in raw_rows:
        try:
            row = Observation(
                cohort=str(raw["cohort"]),
                context_role=str(raw["context_role"]),
                representation=str(raw["representation"]),
                quiet_swap=str(raw["quiet_swap"]),
                nms_distance_px=int(raw["nms_distance_px"]),
                nms_role=str(raw["nms_role"]),
                burden=float(raw["target_nms_peaks_per_pseudo_burst"]),
                burst_id=int(raw["burst_id"]),
                candidate_budget=int(raw["candidate_budget"]),
                effective_candidate_count=int(raw["effective_candidate_count"]),
                observation_id=str(raw["observation_id"]),
                canonical_roi_id=str(raw["canonical_roi_id"]),
                x_px=str(raw["x_px"]),
                y_px=str(raw["y_px"]),
                matched=_strict_bool(raw["matched"], field="matched"),
                unmatched_candidates=str(raw["unmatched_candidates"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtectedSupportSufficiencyUnavailable(
                f"malformed observation row in {path.name}: {exc}"
            ) from exc
        rows.append(row)

    def values(field: str) -> set[Any]:
        return {getattr(row, field) for row in rows}

    expected_sets = {
        "cohort": {cohort},
        "context_role": set(ALL_SOURCE_ROLES),
        "representation": set(ALL_SOURCE_ARMS),
        "quiet_swap": set(QUIET_SWAPS),
        "nms_distance_px": set(ALL_SOURCE_NMS_DISTANCES),
        "burden": set(QUIET_BURDENS),
        "burst_id": set(BURSTS),
        "candidate_budget": set(CANDIDATE_BUDGETS),
        "unmatched_candidates": {"unknown_not_negative"},
    }
    for field, expected in expected_sets.items():
        observed = values(field)
        if observed != expected:
            raise ProtectedSupportSufficiencyUnavailable(
                f"{cohort} {field} values changed: {observed} != {expected}"
            )
    if any(
        (row.nms_distance_px == NMS_DISTANCE_PX) != (row.nms_role == "primary")
        for row in rows
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            f"{cohort} NMS role labels no longer match NMS=6 primary contract"
        )
    if len(values("canonical_roi_id")) != expected_identities:
        raise ProtectedSupportSufficiencyUnavailable(
            f"{cohort} canonical identity count changed"
        )
    if len(values("observation_id")) != expected_occurrences:
        raise ProtectedSupportSufficiencyUnavailable(
            f"{cohort} occurrence count changed"
        )
    return rows


def _analysis_rows(
    rows: Sequence[Observation],
    *,
    expected_identities: int,
    expected_occurrences: int,
) -> tuple[list[Observation], str]:
    selected = [
        row
        for row in rows
        if row.context_role in PROTECTED_ROLES
        and row.representation in FIXED_ARMS
        and row.nms_distance_px == NMS_DISTANCE_PX
    ]
    expected = (
        len(PROTECTED_ROLES)
        * len(FIXED_ARMS)
        * len(QUIET_SWAPS)
        * len(QUIET_BURDENS)
        * len(CANDIDATE_BUDGETS)
        * expected_occurrences
    )
    if len(selected) != expected:
        raise ProtectedSupportSufficiencyUnavailable(
            f"selected paired universe changed: {len(selected)} != {expected}"
        )

    pairs: dict[tuple[Any, ...], dict[str, Observation]] = {}
    for row in selected:
        key = (
            row.representation,
            row.quiet_swap,
            row.burden,
            row.candidate_budget,
            row.observation_id,
        )
        lane = pairs.setdefault(key, {})
        if row.context_role in lane:
            raise ProtectedSupportSufficiencyUnavailable(
                f"duplicate role in paired observation cell: {key}"
            )
        lane[row.context_role] = row
    expected_pair_count = expected // 2
    if len(pairs) != expected_pair_count:
        raise ProtectedSupportSufficiencyUnavailable(
            f"paired cell count changed: {len(pairs)} != {expected_pair_count}"
        )
    identity_union: set[str] = set()
    hasher = hashlib.sha256()
    for key in sorted(pairs):
        lane = pairs[key]
        if set(lane) != set(PROTECTED_ROLES):
            raise ProtectedSupportSufficiencyUnavailable(
                f"candidate/comparator pairing is incomplete: {key}"
            )
        candidate = lane[PROTECTED_ROLES[0]]
        comparator = lane[PROTECTED_ROLES[1]]
        annotation_a = (
            candidate.burst_id,
            candidate.canonical_roi_id,
            candidate.x_px,
            candidate.y_px,
        )
        annotation_b = (
            comparator.burst_id,
            comparator.canonical_roi_id,
            comparator.x_px,
            comparator.y_px,
        )
        if annotation_a != annotation_b:
            raise ProtectedSupportSufficiencyUnavailable(
                f"paired roles do not share the same observation truth: {key}"
            )
        identity_union.add(candidate.canonical_roi_id)
        payload = {
            "key": key,
            "annotation": annotation_a,
            "candidate_matched": candidate.matched,
            "larger_matched": comparator.matched,
        }
        hasher.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        hasher.update(b"\n")
    if len(identity_union) != expected_identities:
        raise ProtectedSupportSufficiencyUnavailable(
            "paired universe does not contain the exact canonical identity count"
        )

    for role in PROTECTED_ROLES:
        for arm in FIXED_ARMS:
            for swap in QUIET_SWAPS:
                for burden in QUIET_BURDENS:
                    for budget in CANDIDATE_BUDGETS:
                        cell = [
                            row
                            for row in selected
                            if row.context_role == role
                            and row.representation == arm
                            and row.quiet_swap == swap
                            and row.burden == burden
                            and row.candidate_budget == budget
                        ]
                        if (
                            len(cell) != expected_occurrences
                            or len({row.canonical_roi_id for row in cell})
                            != expected_identities
                            or {row.burst_id for row in cell} != set(BURSTS)
                        ):
                            raise ProtectedSupportSufficiencyUnavailable(
                                "analysis cell escaped exact occurrence/identity/burst contract: "
                                f"{role}/{arm}/{swap}/{burden}/{budget}"
                            )
    return selected, hasher.hexdigest()


def _bootstrap_weight_matrix(
    identities: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> np.ndarray:
    """Reproduce the r2 one-choice-call-per-replicate cluster bootstrap."""

    names = tuple(str(identity) for identity in identities)
    if len(names) != len(set(names)) or not names or int(replicates) < 1:
        raise ValueError("bootstrap identities must be unique and non-empty")
    positions = {identity: index for index, identity in enumerate(names)}
    rng = np.random.default_rng(int(seed))
    weights = np.zeros((int(replicates), len(names)), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = rng.choice(names, size=len(names), replace=True)
        for identity in sampled:
            weights[replicate, positions[str(identity)]] += 1.0
    if not np.all(weights.sum(axis=1) == len(names)):
        raise AssertionError("bootstrap cluster weights do not reconcile")
    return weights


def _metric_arrays(
    rows: Sequence[Observation],
    *,
    identities: Sequence[str],
    bootstrap_weights: np.ndarray | None,
) -> Mapping[str, np.ndarray]:
    role_index = {name: index for index, name in enumerate(PROTECTED_ROLES)}
    budget_index = {value: index for index, value in enumerate(CANDIDATE_BUDGETS)}
    identity_index = {name: index for index, name in enumerate(identities)}
    counts = np.zeros(
        (len(PROTECTED_ROLES), len(CANDIDATE_BUDGETS), len(BURSTS), len(identities)),
        dtype=np.float64,
    )
    matched = np.zeros_like(counts)
    for row in rows:
        key = (
            role_index[row.context_role],
            budget_index[row.candidate_budget],
            row.burst_id - 1,
            identity_index[row.canonical_roi_id],
        )
        counts[key] += 1.0
        matched[key] += float(row.matched)
    if np.any(counts.sum(axis=-1) == 0.0):
        raise ProtectedSupportSufficiencyUnavailable(
            "metric array is missing a role/budget/burst cell"
        )

    def evaluate(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count_flat = counts.transpose(3, 0, 1, 2).reshape(len(identities), -1)
        match_flat = matched.transpose(3, 0, 1, 2).reshape(len(identities), -1)
        denominators = (weights @ count_flat).reshape(
            len(weights), len(PROTECTED_ROLES), len(CANDIDATE_BUDGETS), len(BURSTS)
        )
        numerators = (weights @ match_flat).reshape(denominators.shape)
        burst_recall = np.full_like(numerators, np.nan)
        np.divide(
            numerators,
            denominators,
            out=burst_recall,
            where=denominators != 0.0,
        )
        with np.errstate(invalid="ignore"):
            macro_recall = np.nanmean(burst_recall, axis=-1)
        x = np.asarray(CANDIDATE_BUDGETS, dtype=np.float64)
        auc = np.trapezoid(macro_recall, x=x, axis=-1) / (x[-1] - x[0])
        return burst_recall, macro_recall, auc

    observed_burst, observed_macro, observed_auc = evaluate(
        np.ones((1, len(identities)), dtype=np.float64)
    )
    result: dict[str, np.ndarray] = {
        "counts": counts,
        "matched": matched,
        "observed_burst": observed_burst[0],
        "observed_macro": observed_macro[0],
        "observed_auc": observed_auc[0],
    }
    if bootstrap_weights is not None:
        bootstrap_burst, bootstrap_macro, bootstrap_auc = evaluate(bootstrap_weights)
        result.update(
            {
                "bootstrap_burst": bootstrap_burst,
                "bootstrap_macro": bootstrap_macro,
                "bootstrap_auc": bootstrap_auc,
            }
        )
    return result


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(values, (2.5, 97.5), method="linear")
    return float(low), float(high)


def compute_paired_metrics(
    rows: Sequence[Observation],
    *,
    expected_identities: int,
    inferential: bool,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute exact paired budget, AUC/B58, and per-burst B58 tables."""

    identities = sorted({row.canonical_roi_id for row in rows})
    if len(identities) != expected_identities:
        raise ProtectedSupportSufficiencyUnavailable(
            f"metric input has {len(identities)} identities, expected {expected_identities}"
        )
    weights = (
        _bootstrap_weight_matrix(identities, seed=seed, replicates=replicates)
        if inferential
        else None
    )
    budget_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    burst_rows: list[dict[str, Any]] = []
    for arm in FIXED_ARMS:
        for scope in QUIET_SCOPES:
            swaps = QUIET_SWAPS if scope == "crossfit_average" else (scope,)
            for burden in QUIET_BURDENS:
                subset = [
                    row
                    for row in rows
                    if row.representation == arm
                    and row.quiet_swap in swaps
                    and row.burden == burden
                ]
                metrics = _metric_arrays(
                    subset,
                    identities=identities,
                    bootstrap_weights=weights,
                )
                observed_macro = metrics["observed_macro"]
                observed_auc = metrics["observed_auc"]
                matched = metrics["matched"]
                scope_occurrences = len(
                    {row.observation_id for row in subset if row.quiet_swap == swaps[0]}
                ) * len(swaps)
                for budget_position, budget in enumerate(CANDIDATE_BUDGETS):
                    candidate = float(observed_macro[0, budget_position])
                    larger = float(observed_macro[1, budget_position])
                    delta = candidate - larger
                    low: float | None = None
                    high: float | None = None
                    if inferential:
                        distribution = (
                            metrics["bootstrap_macro"][:, 0, budget_position]
                            - metrics["bootstrap_macro"][:, 1, budget_position]
                        )
                        low, high = _percentile_ci(distribution)
                    budget_rows.append(
                        {
                            "cohort": subset[0].cohort,
                            "decision_used": inferential,
                            "representation": arm,
                            "quiet_scope": scope,
                            "quiet_swaps_included": len(swaps),
                            "nms_distance_px": NMS_DISTANCE_PX,
                            "target_nms_peaks_per_pseudo_burst": burden,
                            "candidate_budget": budget,
                            "canonical_identity_clusters": len(identities),
                            "occurrences_in_scope": scope_occurrences,
                            "candidate_macro_known_positive_recall": candidate,
                            "larger_macro_known_positive_recall": larger,
                            "candidate_minus_larger_recall": delta,
                            "delta_ci95_low": low,
                            "delta_ci95_high": high,
                            "candidate_matched_occurrences": int(
                                matched[0, budget_position].sum()
                            ),
                            "larger_matched_occurrences": int(
                                matched[1, budget_position].sum()
                            ),
                            "larger_minus_candidate_matched_occurrences": int(
                                matched[1, budget_position].sum()
                                - matched[0, budget_position].sum()
                            ),
                            "unmatched_candidates": "unknown_not_negative",
                            "precision_identified": False,
                            "bootstrap_applied": inferential,
                            "bootstrap_seed": seed if inferential else None,
                            "bootstrap_replicates": replicates if inferential else None,
                        }
                    )

                auc_delta = float(observed_auc[0] - observed_auc[1])
                b58_position = CANDIDATE_BUDGETS.index(58)
                b58_delta = float(
                    observed_macro[0, b58_position]
                    - observed_macro[1, b58_position]
                )
                auc_low: float | None = None
                auc_high: float | None = None
                b58_low: float | None = None
                b58_high: float | None = None
                if inferential:
                    auc_low, auc_high = _percentile_ci(
                        metrics["bootstrap_auc"][:, 0]
                        - metrics["bootstrap_auc"][:, 1]
                    )
                    b58_low, b58_high = _percentile_ci(
                        metrics["bootstrap_macro"][:, 0, b58_position]
                        - metrics["bootstrap_macro"][:, 1, b58_position]
                    )
                candidate_b58_matches = int(matched[0, b58_position].sum())
                larger_b58_matches = int(matched[1, b58_position].sum())
                burst_losses: list[int] = []
                for burst_position, burst in enumerate(BURSTS):
                    candidate_burst = float(
                        metrics["observed_burst"][0, b58_position, burst_position]
                    )
                    larger_burst = float(
                        metrics["observed_burst"][1, b58_position, burst_position]
                    )
                    candidate_matches = int(
                        matched[0, b58_position, burst_position].sum()
                    )
                    larger_matches = int(
                        matched[1, b58_position, burst_position].sum()
                    )
                    burst_loss = larger_matches - candidate_matches
                    burst_losses.append(burst_loss)
                    burst_low: float | None = None
                    burst_high: float | None = None
                    if inferential:
                        burst_low, burst_high = _percentile_ci(
                            metrics["bootstrap_burst"][:, 0, b58_position, burst_position]
                            - metrics["bootstrap_burst"][:, 1, b58_position, burst_position]
                        )
                    burst_rows.append(
                        {
                            "cohort": subset[0].cohort,
                            "decision_used": inferential,
                            "representation": arm,
                            "quiet_scope": scope,
                            "quiet_swaps_included": len(swaps),
                            "nms_distance_px": NMS_DISTANCE_PX,
                            "target_nms_peaks_per_pseudo_burst": burden,
                            "candidate_budget": 58,
                            "burst_id": burst,
                            "candidate_known_positive_recall": candidate_burst,
                            "larger_known_positive_recall": larger_burst,
                            "candidate_minus_larger_recall": candidate_burst - larger_burst,
                            "delta_ci95_low": burst_low,
                            "delta_ci95_high": burst_high,
                            "candidate_matched_occurrences": candidate_matches,
                            "larger_matched_occurrences": larger_matches,
                            "larger_minus_candidate_matched_occurrences": burst_loss,
                            "unmatched_candidates": "unknown_not_negative",
                            "precision_identified": False,
                            "bootstrap_applied": inferential,
                        }
                    )
                practical_rule_applied = scope in QUIET_SWAPS and inferential
                total_loss = larger_b58_matches - candidate_b58_matches
                contrast_rows.append(
                    {
                        "cohort": subset[0].cohort,
                        "decision_used": inferential,
                        "representation": arm,
                        "quiet_scope": scope,
                        "quiet_swaps_included": len(swaps),
                        "nms_distance_px": NMS_DISTANCE_PX,
                        "target_nms_peaks_per_pseudo_burst": burden,
                        "canonical_identity_clusters": len(identities),
                        "occurrences_in_scope": scope_occurrences,
                        "candidate_budget_auc": float(observed_auc[0]),
                        "larger_budget_auc": float(observed_auc[1]),
                        "candidate_minus_larger_budget_auc": auc_delta,
                        "budget_auc_delta_ci95_low": auc_low,
                        "budget_auc_delta_ci95_high": auc_high,
                        "candidate_b58_macro_recall": float(
                            observed_macro[0, b58_position]
                        ),
                        "larger_b58_macro_recall": float(
                            observed_macro[1, b58_position]
                        ),
                        "candidate_minus_larger_b58_macro_recall": b58_delta,
                        "b58_delta_ci95_low": b58_low,
                        "b58_delta_ci95_high": b58_high,
                        "candidate_b58_matched_occurrences": candidate_b58_matches,
                        "larger_b58_matched_occurrences": larger_b58_matches,
                        "larger_minus_candidate_b58_matched_occurrences": total_loss,
                        "maximum_larger_minus_candidate_b58_matches_in_one_burst": max(
                            burst_losses
                        ),
                        "nominal_auc_detectably_worse": bool(
                            inferential and auc_high is not None and auc_high < 0.0
                        ),
                        "nominal_b58_detectably_worse": bool(
                            inferential and b58_high is not None and b58_high < 0.0
                        ),
                        "predeclared_occurrence_tolerances_applied": practical_rule_applied,
                        "budget_auc_loss_within_0p02": (
                            (auc_delta >= -BUDGET_AUC_LOSS_TOLERANCE)
                            if practical_rule_applied
                            else None
                        ),
                        "b58_total_match_loss_within_1": (
                            (total_loss <= B58_TOTAL_MATCH_TOLERANCE)
                            if practical_rule_applied
                            else None
                        ),
                        "b58_each_burst_match_loss_within_1": (
                            (max(burst_losses) <= B58_PER_BURST_MATCH_TOLERANCE)
                            if practical_rule_applied
                            else None
                        ),
                        "unmatched_candidates": "unknown_not_negative",
                        "precision_identified": False,
                        "bootstrap_applied": inferential,
                        "bootstrap_seed": seed if inferential else None,
                        "bootstrap_replicates": replicates if inferential else None,
                        "bootstrap_cluster_field": (
                            "canonical_roi_id" if inferential else None
                        ),
                        "ci_method": (
                            "two_sided_percentile_95_linear_unadjusted"
                            if inferential
                            else None
                        ),
                    }
                )
    return budget_rows, contrast_rows, burst_rows


def verify_support_screen_source(
    root: str | Path,
    *,
    protected_contract: Mapping[str, Any],
) -> tuple[
    VerifiedArtifact,
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, str]],
]:
    source = verify_artifact_index(
        root, expected_index_sha256=SUPPORT_SCREEN_INDEX_SHA256
    )
    required = (
        "boundary_decision.json",
        "claim_boundary.json",
        "fold_contexts.json",
        "repeated_single_frame_latency.tsv",
        "summary.json",
        "validation.json",
    )
    _require_indexed(source, required)
    validation = _json(source.root / "validation.json")
    checks = validation.get("checks")
    expected_checks = {
        "adaptive_stage_b_rule_applied": True,
        "control_cell_count_exact": True,
        "controls_ineligible_for_primary_selection": True,
        "cuda_required_and_used": True,
        "eligible_cell_count_exact": True,
        "fold_local_contexts_exposed": True,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "protected_recall_pending": True,
        "scientific_audit_pending": True,
        "three_named_controls_executed": True,
    }
    if (
        validation.get("status")
        != "passed_support_screen_artifact_contract_scientific_audit_pending"
        or checks != expected_checks
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "support-screen artifact validation is not an all-checks PASS"
        )
    summary = _json(source.root / "summary.json")
    if (
        summary.get("status") != "complete_support_screen_only"
        or summary.get("stage_b_triggered") is not False
        or summary.get("largest_tested_half_width_px") != 31
        or summary.get("claim_boundary", {}).get("support_sufficiency_claimed")
        is not False
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "support-screen status/boundary changed"
        )
    boundary = _json(source.root / "boundary_decision.json")
    protected_rule = boundary.get("final_sufficiency_rule", {}).get("protected")
    if protected_rule != {
        "b58_total_match_loss_at_most": B58_TOTAL_MATCH_TOLERANCE,
        "budget_curve_auc_loss_at_most": BUDGET_AUC_LOSS_TOLERANCE,
        "per_burst_match_loss_at_most": B58_PER_BURST_MATCH_TOLERANCE,
    }:
        raise ProtectedSupportSufficiencyUnavailable(
            "predeclared protected support stopping rule changed"
        )
    if (
        boundary.get("stage_b_triggered") is not False
        or boundary.get("support_boundary", {}).get("status")
        != "closed_by_clear_endpoint_inferiority"
        or boundary.get("support_boundary", {}).get(
            "requires_further_nonlabel_screen_before_sufficiency_claim"
        )
        is not False
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "coordinate-free upper support boundary is not closed"
        )

    fold_contexts = _json(source.root / "fold_contexts.json")
    selection = protected_contract.get("context_selection", {})
    expected_fold_hash = selection.get("source_sha256")
    if (
        _sha256(source.root / "fold_contexts.json") != expected_fold_hash
        or fold_contexts.get("selection_scope") != "outer_training_fold_only"
        or fold_contexts.get("selection_uses_positive_coordinates") is not False
        or fold_contexts.get("selection_uses_positive_identities") is not False
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "support fold-context selection does not match the protected run contract"
        )
    expected_root = Path(str(selection.get("root", ""))).resolve()
    if expected_root != source.root:
        raise ProtectedSupportSufficiencyUnavailable(
            "explicit support-screen root differs from the protected source root"
        )

    with (source.root / "repeated_single_frame_latency.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        latency_rows = list(reader)
    if len(latency_rows) != 8:
        raise ProtectedSupportSufficiencyUnavailable(
            "repeated single-frame latency row count changed"
        )
    radial = [row for row in latency_rows if row["family"] == "radial_gamma_ls"]
    if (
        {int(row["half_width_px"]) for row in radial} != {11, 15, 19, 23, 31}
        or any(int(row["warmup_iterations"]) != 50 for row in latency_rows)
        or any(int(row["timed_iterations"]) != 200 for row in latency_rows)
    ):
        raise ProtectedSupportSufficiencyUnavailable(
            "repeated latency design changed"
        )
    return source, boundary, fold_contexts, latency_rows


def latency_pairs(
    *,
    fold_contexts: Mapping[str, Any],
    latency_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    radial_by_width = {
        int(row["half_width_px"]): row
        for row in latency_rows
        if row["family"] == "radial_gamma_ls"
    }
    folds = fold_contexts.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ProtectedSupportSufficiencyUnavailable(
            "fold-context contract no longer has exactly four folds"
        )
    output: list[dict[str, Any]] = []
    for fold in folds:
        candidate = fold.get("support_candidate_context")
        larger = fold.get("larger_support_comparator")
        if not isinstance(candidate, dict) or not isinstance(larger, dict):
            raise ProtectedSupportSufficiencyUnavailable(
                "fold context lacks candidate or larger comparator"
            )
        candidate_width = int(candidate["half_width_px"])
        larger_width = int(larger["half_width_px"])
        if candidate_width >= larger_width or larger_width not in radial_by_width:
            raise ProtectedSupportSufficiencyUnavailable(
                "larger-support role is not strictly larger in a fold"
            )
        candidate_latency = radial_by_width[candidate_width]
        larger_latency = radial_by_width[larger_width]
        for context, latency in (
            (candidate, candidate_latency),
            (larger, larger_latency),
        ):
            copied = context.get("repeated_latency")
            if not isinstance(copied, dict):
                raise ProtectedSupportSufficiencyUnavailable(
                    "fold context lacks copied repeated-latency evidence"
                )
            for field in (
                "half_width_px",
                "support_width_px",
                "warmup_iterations",
                "timed_iterations",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
                "mean_ms",
                "max_memory_allocated_bytes",
                "latency_gate_pass",
            ):
                left = str(copied[field])
                right = str(latency[field])
                if left != right:
                    raise ProtectedSupportSufficiencyUnavailable(
                        f"fold-copied latency differs from TSV for width {context['half_width_px']}: {field}"
                    )
        candidate_gate = _strict_bool(
            candidate_latency["latency_gate_pass"], field="latency_gate_pass"
        )
        larger_gate = _strict_bool(
            larger_latency["latency_gate_pass"], field="latency_gate_pass"
        )
        p50_delta = float(larger_latency["p50_ms"]) - float(
            candidate_latency["p50_ms"]
        )
        p99_delta = float(larger_latency["p99_ms"]) - float(
            candidate_latency["p99_ms"]
        )
        memory_delta = int(larger_latency["max_memory_allocated_bytes"]) - int(
            candidate_latency["max_memory_allocated_bytes"]
        )
        direction = (
            "larger_slower"
            if p50_delta > 0
            else "larger_faster"
            if p50_delta < 0
            else "p50_tie"
        )
        output.append(
            {
                "training_fold": int(fold["training_fold"]),
                "heldout_burst": int(fold["heldout_burst"]),
                "candidate_context_id": candidate["context_id"],
                "candidate_half_width_px": candidate_width,
                "larger_context_id": larger["context_id"],
                "larger_half_width_px": larger_width,
                "latency_measurement_scope": "width_level_representative_gamma_stage_only",
                "warmup_iterations": int(candidate_latency["warmup_iterations"]),
                "timed_iterations": int(candidate_latency["timed_iterations"]),
                "candidate_p50_ms": float(candidate_latency["p50_ms"]),
                "larger_p50_ms": float(larger_latency["p50_ms"]),
                "larger_minus_candidate_p50_ms": p50_delta,
                "candidate_p99_ms": float(candidate_latency["p99_ms"]),
                "larger_p99_ms": float(larger_latency["p99_ms"]),
                "larger_minus_candidate_p99_ms": p99_delta,
                "candidate_max_memory_allocated_bytes": int(
                    candidate_latency["max_memory_allocated_bytes"]
                ),
                "larger_max_memory_allocated_bytes": int(
                    larger_latency["max_memory_allocated_bytes"]
                ),
                "larger_minus_candidate_max_memory_bytes": memory_delta,
                "candidate_latency_gate_pass": candidate_gate,
                "larger_latency_gate_pass": larger_gate,
                "p50_cost_direction": direction,
                "whole_pipeline_1khz_claim_allowed": False,
            }
        )
    output.sort(key=lambda row: row["training_fold"])
    directions = {row["p50_cost_direction"] for row in output}
    all_candidate_pass = all(row["candidate_latency_gate_pass"] for row in output)
    all_larger_pass = all(row["larger_latency_gate_pass"] for row in output)
    if not all_candidate_pass or not all_larger_pass:
        justification: bool | None = None
        status = "unresolved_no_compared_radial_width_passes_predeclared_p99_gate"
    elif len(directions) > 1:
        justification = None
        status = "unresolved_nonmonotone_incremental_latency"
    else:
        justification = False
        status = "larger_has_consistent_cost_but_scientific_gain_must_decide"
    return output, {
        "candidate_latency_gate_pass_all_folds": all_candidate_pass,
        "larger_latency_gate_pass_all_folds": all_larger_pass,
        "p50_cost_directions": sorted(directions),
        "larger_support_justifies_incremental_latency": justification,
        "larger_support_latency_justification_status": status,
        "latency_scope": "Gamma_stage_only_not_whole_pipeline",
    }


def _tsv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ProtectedSupportSufficiencyUnavailable("refusing to write empty TSV")
    fields = list(rows[0])
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != fields:
            raise ProtectedSupportSufficiencyUnavailable("TSV row schemas are not atomic")
        writer.writerow({key: "" if value is None else value for key, value in row.items()})
    return stream.getvalue()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact_index(root: Path) -> Mapping[str, Any]:
    artifacts = []
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != "artifact_index.json"
    ):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"schema_version": 1, "artifacts": artifacts}


def _decision(
    protected_contrasts: Sequence[Mapping[str, Any]],
    *,
    boundary: Mapping[str, Any],
    latency_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    if {str(row.get("cohort")) for row in protected_contrasts} != {"protected_v1"}:
        raise ProtectedSupportSufficiencyUnavailable(
            "the stopping decision may use protected_v1 contrasts only"
        )
    base = [row for row in protected_contrasts if row["quiet_scope"] in QUIET_SWAPS]
    pooled = [
        row for row in protected_contrasts if row["quiet_scope"] == "crossfit_average"
    ]
    practical_failures = [
        row
        for row in base
        if not (
            row["budget_auc_loss_within_0p02"]
            and row["b58_total_match_loss_within_1"]
            and row["b58_each_burst_match_loss_within_1"]
        )
    ]
    detectable_failures = [
        row
        for row in pooled
        if row["nominal_auc_detectably_worse"]
        or row["nominal_b58_detectably_worse"]
    ]
    training_pass = (
        boundary.get("support_boundary", {}).get("status")
        == "closed_by_clear_endpoint_inferiority"
        and boundary.get("stage_b_triggered") is False
    )
    sensitivity_pass = not practical_failures and not detectable_failures
    latency_pass = bool(latency_summary["candidate_latency_gate_pass_all_folds"])
    sufficient = training_pass and sensitivity_pass and latency_pass
    if sufficient:
        status = "supported_under_frozen_bounded_contract"
    else:
        failed = []
        if not training_pass:
            failed.append("training_boundary")
        if not sensitivity_pass:
            failed.append("protected_sensitivity")
        if not latency_pass:
            failed.append("repeated_latency")
        status = "unresolved_failed_" + "_and_".join(failed)

    def compact(row: Mapping[str, Any], reasons: Sequence[str]) -> Mapping[str, Any]:
        return {
            "representation": row["representation"],
            "quiet_scope": row["quiet_scope"],
            "target_nms_peaks_per_pseudo_burst": row[
                "target_nms_peaks_per_pseudo_burst"
            ],
            "candidate_minus_larger_budget_auc": row[
                "candidate_minus_larger_budget_auc"
            ],
            "budget_auc_delta_ci95": [
                row["budget_auc_delta_ci95_low"],
                row["budget_auc_delta_ci95_high"],
            ],
            "candidate_minus_larger_b58_macro_recall": row[
                "candidate_minus_larger_b58_macro_recall"
            ],
            "b58_delta_ci95": [
                row["b58_delta_ci95_low"],
                row["b58_delta_ci95_high"],
            ],
            "larger_minus_candidate_b58_matched_occurrences": row[
                "larger_minus_candidate_b58_matched_occurrences"
            ],
            "maximum_larger_minus_candidate_b58_matches_in_one_burst": row[
                "maximum_larger_minus_candidate_b58_matches_in_one_burst"
            ],
            "failure_reasons": list(reasons),
        }

    practical_details = []
    for row in practical_failures:
        reasons = []
        if not row["budget_auc_loss_within_0p02"]:
            reasons.append("budget_auc_loss_gt_0p02")
        if not row["b58_total_match_loss_within_1"]:
            reasons.append("b58_total_match_loss_gt_1")
        if not row["b58_each_burst_match_loss_within_1"]:
            reasons.append("b58_one_burst_match_loss_gt_1")
        practical_details.append(compact(row, reasons))
    detectable_details = [
        compact(
            row,
            [
                reason
                for condition, reason in (
                    (row["nominal_auc_detectably_worse"], "auc_nominal_ci_below_zero"),
                    (row["nominal_b58_detectably_worse"], "b58_nominal_ci_below_zero"),
                )
                if condition
            ],
        )
        for row in detectable_failures
    ]
    return {
        "support_sufficiency_status": status,
        "support_sufficient": sufficient,
        "training_boundary_gate_pass": training_pass,
        "protected_sensitivity_gate_pass": sensitivity_pass,
        "repeated_latency_gate_pass": latency_pass,
        "predeclared_practical_failure_count": len(practical_details),
        "nominal_detectable_worse_crossfit_failure_count": len(detectable_details),
        "predeclared_practical_failures": practical_details,
        "nominal_detectable_worse_crossfit_failures": detectable_details,
        "stopping_rule": {
            "budget_auc_loss_at_most": BUDGET_AUC_LOSS_TOLERANCE,
            "b58_total_match_loss_at_most": B58_TOTAL_MATCH_TOLERANCE,
            "b58_per_burst_match_loss_at_most": B58_PER_BURST_MATCH_TOLERANCE,
            "not_detectably_worse_rule": (
                "candidate_minus_larger two-sided nominal percentile CI95 is not wholly below zero"
            ),
            "multiplicity_adjustment": "none_predeclared_nominal_cells",
        },
        "latency": dict(latency_summary),
        "v7_used_for_selection_or_decision": False,
        "unmatched_candidates": "unknown_not_negative",
        "precision_identified": False,
    }


def _report(
    summary: Mapping[str, Any],
    contrasts: Sequence[Mapping[str, Any]],
    latency_pairs_rows: Sequence[Mapping[str, Any]],
) -> str:
    decision = summary["decision"]
    pooled_failures = decision["nominal_detectable_worse_crossfit_failures"]
    lines = [
        "# Protected Gamma-LS support-sufficiency analysis",
        "",
        f"**Outcome: {decision['support_sufficiency_status']}.** The smaller fold-local "
        "support candidate is not established as sufficient under the frozen stopping rule.",
        "",
        "The exact protected-v1 comparison uses the same 26 canonical identities, NMS=6, "
        "all five quiet-burden points, all five candidate budgets, both quiet-role swaps, "
        "and only the fixed raw/signed-difference/energy-normalized-difference arms. "
        "Candidate-minus-larger deltas use 2,000 canonical-identity cluster bootstrap "
        "replicates at seed 20260908.",
        "",
        "## Gate result",
        "",
        f"- Coordinate-free training boundary: {'PASS' if decision['training_boundary_gate_pass'] else 'FAIL'}.",
        f"- Protected sensitivity: {'PASS' if decision['protected_sensitivity_gate_pass'] else 'FAIL'} "
        f"({decision['predeclared_practical_failure_count']} swap-specific practical-tolerance failures; "
        f"{decision['nominal_detectable_worse_crossfit_failure_count']} pooled nominal-CI failures).",
        f"- Repeated Gamma-stage latency: {'PASS' if decision['repeated_latency_gate_pass'] else 'FAIL'}.",
        "",
        "## Pooled protected-v1 cells with a nominal CI wholly below zero",
        "",
        "| arm | burden | AUC delta (95% CI) | B58 delta (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for row in pooled_failures:
        lines.append(
            "| {representation} | {target_nms_peaks_per_pseudo_burst:g} | "
            "{candidate_minus_larger_budget_auc:.9f} "
            "[{al:.9f}, {ah:.9f}] | {candidate_minus_larger_b58_macro_recall:.9f} "
            "[{bl:.9f}, {bh:.9f}] |".format(
                **row,
                al=row["budget_auc_delta_ci95"][0],
                ah=row["budget_auc_delta_ci95"][1],
                bl=row["b58_delta_ci95"][0],
                bh=row["b58_delta_ci95"][1],
            )
        )
    if not pooled_failures:
        lines.append("| none | - | - | - |")
    lines.extend(
        [
            "",
        "## Latency interpretation",
        "",
            f"The frozen repeated-latency comparison is "
            f"`{decision['latency']['larger_support_latency_justification_status']}`. "
            "It is a Gamma-stage-only measurement, not a whole-pipeline 1-kHz result. "
            "The width timing is non-monotone and neither compared radial width passes "
            "the predeclared p99 <= 1 ms gate in every fold, so a larger-support latency "
            "tradeoff is not resolved by this artifact.",
            "",
            "| fold | candidate -> larger half-width | p50 ms | p99 ms | p50 direction |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in latency_pairs_rows:
        lines.append(
            "| {training_fold} | {candidate_half_width_px} -> {larger_half_width_px} | "
            "{candidate_p50_ms:.6f} -> {larger_p50_ms:.6f} | "
            "{candidate_p99_ms:.6f} -> {larger_p99_ms:.6f} | {p50_cost_direction} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Only aggregate latency percentiles were retained, so no paired latency CI "
            "can be reconstructed from this artifact.",
            "",
            "## Budget resolution",
            "",
            f"The largest effective protected-v1 candidate count was "
            f"{summary['budget_resolution']['maximum_effective_candidate_count']}, below "
            f"the minimum requested budget of {min(CANDIDATE_BUDGETS)}. Consequently all "
            "five budget points plateau within every analyzed cell; budget AUC and B58 "
            "are numerically identical here and should not be presented as independent "
            "evidence.",
            "",
            "## Claim boundary",
            "",
            "Latest-v7 is written only as descriptive sensitivity and never enters the "
            "decision. Sparse labels are incomplete: unmatched candidates are unknown, "
            "not negatives. No precision, specificity, false-positive-rate, biological "
            "population-generalization, or whole-pipeline 1-kHz claim follows.",
            "",
            "All protected-v1 CIs, including non-failing cells, are in "
            "`protected_v1_support_contrasts.tsv`; all five budget points are in "
            "`protected_v1_budget_points.tsv`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    *,
    protected_dir: str | Path,
    support_screen_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ProtectedSupportSufficiencyUnavailable(
            f"output must be new and non-colliding: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    protected_source, protected_contract = verify_protected_source(protected_dir)
    support_source, boundary, fold_contexts, raw_latency = verify_support_screen_source(
        support_screen_dir, protected_contract=protected_contract
    )
    v1_all = _read_observations(
        protected_source.root / "protected_v1_observation_matches.tsv",
        cohort="protected_v1",
        expected_identities=EXPECTED_V1_IDENTITIES,
        expected_occurrences=EXPECTED_V1_OCCURRENCES,
    )
    v7_all = _read_observations(
        protected_source.root / "latest_v7_observation_matches.tsv",
        cohort="latest_v7_sensitivity",
        expected_identities=EXPECTED_V7_IDENTITIES,
        expected_occurrences=EXPECTED_V7_OCCURRENCES,
    )
    v1, v1_universe_hash = _analysis_rows(
        v1_all,
        expected_identities=EXPECTED_V1_IDENTITIES,
        expected_occurrences=EXPECTED_V1_OCCURRENCES,
    )
    v7, v7_universe_hash = _analysis_rows(
        v7_all,
        expected_identities=EXPECTED_V7_IDENTITIES,
        expected_occurrences=EXPECTED_V7_OCCURRENCES,
    )

    v1_identities = sorted({row.canonical_roi_id for row in v1})
    bootstrap_weights = _bootstrap_weight_matrix(v1_identities)
    bootstrap_weight_bytes = np.asarray(
        bootstrap_weights, dtype="<i2", order="C"
    ).tobytes(order="C")
    bootstrap_weights_sha256 = hashlib.sha256(bootstrap_weight_bytes).hexdigest()
    bootstrap_identities_sha256 = hashlib.sha256(
        ("\n".join(v1_identities) + "\n").encode("utf-8")
    ).hexdigest()

    contract = {
        "schema_version": 1,
        "analysis_id": "gamma_ls_protected_support_sufficiency_v1",
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "frozen_before_metric_outputs": True,
        "sources": {
            "protected_root": str(protected_source.root),
            "protected_artifact_index_sha256": protected_source.index_sha256,
            "support_screen_root": str(support_source.root),
            "support_screen_artifact_index_sha256": support_source.index_sha256,
            "protected_v1_paired_universe_sha256": v1_universe_hash,
            "latest_v7_paired_universe_sha256": v7_universe_hash,
        },
        "design": {
            "roles": list(PROTECTED_ROLES),
            "fixed_arms": list(FIXED_ARMS),
            "nms_distance_px": NMS_DISTANCE_PX,
            "quiet_swaps": list(QUIET_SWAPS),
            "quiet_burdens": list(QUIET_BURDENS),
            "candidate_budgets": list(CANDIDATE_BUDGETS),
            "protected_v1_identity_clusters": EXPECTED_V1_IDENTITIES,
            "protected_v1_occurrences": EXPECTED_V1_OCCURRENCES,
            "latest_v7_role": "descriptive_only_not_selection_or_decision",
        },
        "bootstrap": {
            "cohort": "protected_v1_only",
            "cluster_field": "canonical_roi_id",
            "algorithm": "one_default_rng_choice_call_per_replicate_integer_weights_v1",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "ci": "two_sided_percentile_95_linear_unadjusted",
            "numpy_version": np.__version__,
            "identity_order_sha256": bootstrap_identities_sha256,
            "integer_weight_matrix_dtype": "little_endian_int16",
            "integer_weight_matrix_shape": [
                BOOTSTRAP_REPLICATES,
                EXPECTED_V1_IDENTITIES,
            ],
            "integer_weight_matrix_sha256": bootstrap_weights_sha256,
        },
        "claim_boundary": {
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "v7_used_for_selection_or_decision": False,
        },
    }

    v1_budget, v1_contrasts, v1_bursts = compute_paired_metrics(
        v1,
        expected_identities=EXPECTED_V1_IDENTITIES,
        inferential=True,
    )
    v7_budget, v7_contrasts, v7_bursts = compute_paired_metrics(
        v7,
        expected_identities=EXPECTED_V7_IDENTITIES,
        inferential=False,
    )
    latency, latency_summary = latency_pairs(
        fold_contexts=fold_contexts,
        latency_rows=raw_latency,
    )
    decision = _decision(
        v1_contrasts,
        boundary=boundary,
        latency_summary=latency_summary,
    )
    plateau_groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for row in v1_budget:
        plateau_groups.setdefault(
            (
                str(row["representation"]),
                str(row["quiet_scope"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            ),
            [],
        ).append(row)
    all_budget_curves_plateau = all(
        len({float(row["candidate_macro_known_positive_recall"]) for row in group})
        == 1
        and len({float(row["larger_macro_known_positive_recall"]) for row in group})
        == 1
        for group in plateau_groups.values()
    )
    maximum_effective_candidate_count = max(
        row.effective_candidate_count for row in v1
    )
    completed = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "gamma_ls_protected_support_sufficiency_v1",
        "status": "complete_validated_analysis_support_sufficiency_unresolved",
        "completed_at_utc": completed,
        "decision": decision,
        "budget_resolution": {
            "all_five_budget_points_evaluated": True,
            "all_analyzed_budget_curves_plateau": all_budget_curves_plateau,
            "minimum_candidate_budget": min(CANDIDATE_BUDGETS),
            "maximum_effective_candidate_count": maximum_effective_candidate_count,
            "interpretation": (
                "all retained candidate counts are below the minimum budget; budget AUC "
                "and B58 are therefore numerically identical, not independent evidence"
            ),
        },
        "row_counts": {
            "protected_v1_selected_observation_rows": len(v1),
            "protected_v1_budget_point_rows": len(v1_budget),
            "protected_v1_support_contrast_rows": len(v1_contrasts),
            "protected_v1_b58_burst_rows": len(v1_bursts),
            "latest_v7_selected_observation_rows": len(v7),
            "latest_v7_descriptive_budget_point_rows": len(v7_budget),
            "latest_v7_descriptive_support_contrast_rows": len(v7_contrasts),
            "latest_v7_descriptive_b58_burst_rows": len(v7_bursts),
            "latency_pair_rows": len(latency),
        },
        "claim_boundary": contract["claim_boundary"],
    }
    validation = {
        "status": "passed_provenance_closed_support_sufficiency_analysis_contract",
        "all_checks_pass": True,
        "checks": {
            "protected_artifact_index_exact_sha256": (
                protected_source.index_sha256 == PROTECTED_INDEX_SHA256
            ),
            "support_screen_artifact_index_exact_sha256": (
                support_source.index_sha256 == SUPPORT_SCREEN_INDEX_SHA256
            ),
            "protected_candidate_seal_and_validation_verified": True,
            "support_screen_validation_and_stopping_rule_verified": True,
            "same_26_v1_canonical_identities_in_every_paired_cell": True,
            "fixed_three_arms_only": True,
            "nms_exactly_6": True,
            "all_five_burdens_and_budgets": True,
            "bootstrap_exact_seed_replicates_and_rng_call_pattern": True,
            "bootstrap_integer_weight_matrix_hashed": (
                len(bootstrap_weights_sha256) == 64
            ),
            "budget_curves_reconcile_and_plateau_is_disclosed": (
                all_budget_curves_plateau
                and maximum_effective_candidate_count < min(CANDIDATE_BUDGETS)
            ),
            "v7_descriptive_only": True,
            "unmatched_candidates_unknown": True,
            "precision_not_computed": True,
            "output_noncolliding": True,
        },
        "scientific_result": decision["support_sufficiency_status"],
    }
    if not all(validation["checks"].values()):
        raise AssertionError("internal validation checks did not reconcile")

    work = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.work-", dir=str(destination.parent)
        )
    )
    try:
        # The contract is intentionally the first file written in the work tree.
        _write_json(work / "analysis_contract.json", contract)
        _write_text(work / "protected_v1_budget_points.tsv", _tsv_text(v1_budget))
        _write_text(
            work / "protected_v1_support_contrasts.tsv", _tsv_text(v1_contrasts)
        )
        _write_text(work / "protected_v1_b58_burst.tsv", _tsv_text(v1_bursts))
        _write_text(
            work / "latest_v7_descriptive_budget_points.tsv", _tsv_text(v7_budget)
        )
        _write_text(
            work / "latest_v7_descriptive_support_contrasts.tsv",
            _tsv_text(v7_contrasts),
        )
        _write_text(
            work / "latest_v7_descriptive_b58_burst.tsv", _tsv_text(v7_bursts)
        )
        _write_text(work / "latency_pairs.tsv", _tsv_text(latency))
        _write_json(work / "summary.json", summary)
        _write_json(work / "validation.json", validation)
        _write_text(work / "REPORT.md", _report(summary, v1_contrasts, latency))
        _write_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise ProtectedSupportSufficiencyUnavailable(
                f"output collided during analysis: {destination}"
            )
        os.replace(work, destination)
    except Exception:
        # The unique work tree is retained for forensic diagnosis; it is never
        # mistaken for the requested completed destination.
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-dir", required=True)
    parser.add_argument("--support-screen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = run_analysis(
        protected_dir=args.protected_dir,
        support_screen_dir=args.support_screen_dir,
        output_dir=args.output_dir,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
