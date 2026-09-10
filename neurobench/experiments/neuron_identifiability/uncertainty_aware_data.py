"""Deterministic data contracts for uncertainty-aware neuron-likeness learning.

This module owns only the harmonization boundary.  It joins a *frozen*,
label-free proposal census to canonical-v7 occurrence labels and the later
site-level review without fitting preprocessing, a classifier, or a decision
threshold.  The two label sources remain separate because they are neither
independent nor interchangeable.

Raw profile measurements are the only model-facing features.  Coordinates,
labels, reviewer annotations, proposal ranks, priority scores, frozen taxonomy
classes, and the taxonomy's full-census ``z_*`` columns are never returned by
``model_feature_rows``.  Missing measurements remain ``None`` and receive an
explicit missingness flag; imputation and scaling belong inside each training
fold.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from neurobench.portable_paths import portable_path
from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
    FEATURE_IDS as INNOVATION_FEATURE_IDS,
    FEATURE_SETS as INNOVATION_FEATURE_SETS,
    NEGATIVE_EVIDENCE_IDS as INNOVATION_NEGATIVE_EVIDENCE_IDS,
)

from .contracts import stable_hash


OCCURRENCE_COLUMNS_V1 = (
    "detection_occurrence_id",
    "burst_id",
    "x_px",
    "y_px",
    "start_ui",
    "end_ui",
    "peak_frame_ui",
    "lane_agreement",
    "mean_rank_fraction",
    "coherence_rank",
    "propagation_rank",
    "coherence_score",
    "propagation_score",
    "quiet_intensity",
    "raw_peak",
    "residual_peak",
    "robust_snr",
    "event_area",
    "time_to_peak_frames",
    "time_to_peak_fraction",
    "spatial_specificity",
    "annulus_correlation",
    "nearest_known_positive_distance_px",
    "known_positive_within_6px",
    "nearest_known_positive_id",
    "detection_site_id",
    "recurrence_fraction",
    "z_log_raw_peak",
    "z_log_residual_peak",
    "z_log_snr",
    "z_spatial_specificity",
    "z_annulus_correlation",
    "z_event_area_scaled",
    "z_time_to_peak_fraction",
    "z_lane_agreement",
    "z_mean_rank_fraction",
    "z_recurrence_fraction",
    "z_quiet_intensity_scaled",
    "class_id",
    "class_name",
)

SITE_COLUMNS_V1 = (
    "detection_site_id",
    "occurrences",
    "bursts",
    "x_px",
    "y_px",
    "dominant_class",
    "class_consistency",
    "known_positive_occurrences",
    "median_raw_peak",
    "median_spatial_specificity",
)

CANONICAL_V7_REQUIRED_COLUMNS = (
    "observation_id",
    "burst_id",
    "original_roi_id",
    "canonical_roi_id",
    "x_px",
    "y_px",
    "neuron_confidence",
    "morphology",
    "context",
    "disposition",
    "include_confirmed",
    "include_inclusive",
    "review_status",
    "reviewer_id",
    "source_note",
)

NEW_REVIEW_COLUMNS_V1 = (
    "blind_id",
    "detection_site_id",
    "normalized_label",
    "confidence_1_to_5",
    "size_flag",
    "morphology",
    "low_snr",
    "artifact_proximity",
    "overlap_or_adjacent_source",
    "verbatim_feedback",
    "normalization_note",
)

NEW_REVIEW_LABELS = frozenset(
    {"definite_neuron", "probable_neuron", "uncertain", "artifact_or_noise"}
)
LABEL_POLICIES = (
    "strict",
    "inclusive",
    "canonical_only",
    "new_review_holdout",
)


@dataclass(frozen=True)
class FeatureDefinition:
    """One raw, pre-preprocessing feature and its scientific interpretation."""

    name: str
    taxonomy_name: str
    family: str
    description: str
    recommended_fold_local_transform: str
    occurrence_source: str
    site_aggregation: str = "median_over_site_occurrences_ignoring_missing"
    missing_policy: str = "preserve_null_and_add_missing_indicator"
    primary_model_input: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        "raw_peak",
        "log_raw_peak",
        "amplitude_snr",
        "Peak disk-mean fluorescence above the pre-event median.",
        "log1p_nonnegative_then_fold_local_robust_scale",
        "raw_peak",
    ),
    FeatureDefinition(
        "residual_peak",
        "log_residual_peak",
        "spatial_specificity",
        "Peak center-minus-annulus residual above its pre-event median.",
        "log1p_nonnegative_then_fold_local_robust_scale",
        "residual_peak",
    ),
    FeatureDefinition(
        "robust_snr",
        "log_snr",
        "amplitude_snr",
        "Raw peak divided by the normal-consistent pre-event MAD.",
        "log1p_nonnegative_then_fold_local_robust_scale",
        "robust_snr",
    ),
    FeatureDefinition(
        "spatial_specificity",
        "spatial_specificity",
        "spatial_specificity",
        "Signed center-versus-annulus peak contrast normalized by total magnitude.",
        "fold_local_robust_scale",
        "spatial_specificity",
    ),
    FeatureDefinition(
        "annulus_correlation",
        "annulus_correlation",
        "spatial_specificity",
        "Center/annulus trace correlation over baseline plus event frames.",
        "fold_local_robust_scale",
        "annulus_correlation",
    ),
    FeatureDefinition(
        "event_area",
        "event_area_scaled",
        "kinetics",
        "Signed sum of the baseline-subtracted event trace.",
        "signed_log1p_then_fold_local_robust_scale",
        "event_area",
    ),
    FeatureDefinition(
        "time_to_peak_fraction",
        "time_to_peak_fraction",
        "kinetics",
        "Peak offset divided by the event interval length minus one.",
        "fold_local_robust_scale",
        "time_to_peak_fraction",
    ),
    FeatureDefinition(
        "lane_agreement",
        "lane_agreement",
        "cross_lane_agreement",
        "Fraction of the two frozen proposal lanes represented at the occurrence.",
        "fold_local_robust_scale",
        "lane_agreement",
    ),
    FeatureDefinition(
        "mean_rank_fraction",
        "mean_rank_fraction",
        "proposal_context",
        "Mean within-lane rank divided by the frozen per-burst candidate budget.",
        "fold_local_robust_scale",
        "mean_rank_fraction",
    ),
    FeatureDefinition(
        "recurrence_fraction",
        "recurrence_fraction",
        "recurrence",
        "Fraction of the four bursts containing the consolidated detection site.",
        "fold_local_robust_scale",
        "recurrence_fraction",
    ),
    FeatureDefinition(
        "quiet_intensity",
        "quiet_intensity_scaled",
        "acquisition_context",
        "Median pre-event disk intensity at the proposal center.",
        "log1p_nonnegative_then_fold_local_robust_scale",
        "quiet_intensity",
    ),
)

RAW_FEATURE_COLUMNS = tuple(item.name for item in FEATURE_DEFINITIONS)
TAXONOMY_FEATURE_NAMES = tuple(item.taxonomy_name for item in FEATURE_DEFINITIONS)

PROHIBITED_EXACT_COLUMNS = (
    "x_px",
    "y_px",
    "burst_id",
    "detection_occurrence_id",
    "detection_site_id",
    "nearest_known_positive_distance_px",
    "known_positive_within_6px",
    "nearest_known_positive_id",
    "class_id",
    "class_name",
    "dominant_class",
    "class_consistency",
    "known_positive_occurrences",
    "priority_rank",
    "evidence_score",
    "uncertainty_score",
    "crowding_score",
    "priority_score",
    "normalized_label",
    "confidence_1_to_5",
    "morphology",
    "low_snr",
    "artifact_proximity",
    "overlap_or_adjacent_source",
)


@dataclass(frozen=True)
class ProposalCensusContract:
    """The proposal universe that must be frozen before any label join."""

    census_id: str
    fixed_candidates_per_burst: int
    lanes: tuple[str, ...] = ("coherence_w15", "propagation_lag2_w15")
    nms_radius_px: float = 6.0
    union_rule: str = "strict_object_separated_cross_lane_union"
    freeze_status: str = "frozen_before_label_join"
    labels_used_to_build_census: bool = False
    allow_additional_source_columns: bool = False

    def __post_init__(self) -> None:
        if not self.census_id:
            raise ValueError("census_id is required")
        if self.fixed_candidates_per_burst < 1:
            raise ValueError("fixed_candidates_per_burst must be positive")
        if not self.lanes or len(self.lanes) != len(set(self.lanes)):
            raise ValueError("proposal lanes must be non-empty and unique")
        if self.nms_radius_px <= 0:
            raise ValueError("nms_radius_px must be positive")
        if self.freeze_status != "frozen_before_label_join":
            raise ValueError("proposal census must be frozen before label joining")
        if self.labels_used_to_build_census:
            raise ValueError("label-informed proposal censuses are prohibited")

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


B20_V5_CENSUS = ProposalCensusContract(
    census_id="detection_profile_taxonomy_v5_b20",
    fixed_candidates_per_burst=20,
)


@dataclass(frozen=True)
class HarmonizationInputs:
    occurrence_profiles: Path
    site_profiles: Path
    taxonomy_summary: Path
    taxonomy_validation: Path
    canonical_v7_labels: Path
    new_review_labels: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("occurrence_profiles", self.occurrence_profiles),
            ("site_profiles", self.site_profiles),
            ("taxonomy_summary", self.taxonomy_summary),
            ("taxonomy_validation", self.taxonomy_validation),
            ("canonical_v7_labels", self.canonical_v7_labels),
            ("new_review_labels", self.new_review_labels),
        )


@dataclass(frozen=True)
class ReadinessRequirements:
    outer_folds: int = 5
    minimum_positive_groups_per_fold: int = 2
    minimum_unlabeled_groups_per_fold: int = 2

    def __post_init__(self) -> None:
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least two")
        if min(self.minimum_positive_groups_per_fold, self.minimum_unlabeled_groups_per_fold) < 1:
            raise ValueError("per-fold group minima must be positive")

    @property
    def minimum_positive_groups(self) -> int:
        return self.outer_folds * self.minimum_positive_groups_per_fold

    @property
    def minimum_unlabeled_groups(self) -> int:
        return self.outer_folds * self.minimum_unlabeled_groups_per_fold


LabelPolicy = Literal["strict", "inclusive", "canonical_only", "new_review_holdout"]


@dataclass(frozen=True)
class HarmonizedFeatureCensus:
    """In-memory, provenance-frozen harmonized proposal census."""

    occurrence_rows: tuple[dict[str, Any], ...]
    site_rows: tuple[dict[str, Any], ...]
    feature_dictionary: dict[str, Any]
    manifest: dict[str, Any]
    validation: dict[str, Any]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return RAW_FEATURE_COLUMNS

    def model_feature_rows(self, grain: Literal["occurrence", "site"] = "site") -> tuple[dict[str, Any], ...]:
        """Return only raw features, missing flags, row IDs, and leakage groups."""
        if grain not in {"occurrence", "site"}:
            raise ValueError(f"unsupported feature grain: {grain!r}")
        source = self.occurrence_rows if grain == "occurrence" else self.site_rows
        key = "detection_occurrence_id" if grain == "occurrence" else "detection_site_id"
        rows = []
        for item in source:
            row = {
                "row_id": item[key],
                "leakage_group_id": item["leakage_group_id"],
            }
            for name in RAW_FEATURE_COLUMNS:
                value = item["features"][name]
                row[name] = value
                row[f"missing__{name}"] = value is None
            rows.append(row)
        return tuple(rows)

    def label_rows(self, policy: LabelPolicy = "strict") -> tuple[dict[str, Any], ...]:
        """Materialize an explicit site-level PU policy without inventing negatives."""
        if policy not in LABEL_POLICIES:
            raise ValueError(f"unsupported label policy: {policy!r}")
        return tuple(_label_policy_row(row, policy) for row in self.site_rows)

    def assert_model_ready(self, policy: LabelPolicy = "strict") -> None:
        """Fail closed unless the requested grouped P/U policy passed its gate."""
        result = self.validation["model_readiness"][policy]
        if not result["passed"]:
            raise ValueError(
                f"proposal census is not ready for {policy!r} grouped learning: "
                + "; ".join(result["failures"])
            )


def feature_dictionary() -> dict[str, Any]:
    """Return the frozen raw-feature and leakage contract."""
    payload = {
        "schema_version": 1,
        "feature_grains": {
            "occurrence": "one frozen proposal occurrence in one burst",
            "site": "median of raw occurrence measurements within one consolidated detection site",
        },
        "features": [item.to_dict() for item in FEATURE_DEFINITIONS],
        "feature_order": list(RAW_FEATURE_COLUMNS),
        "missingness": "nulls are preserved; each model row has a missing__<feature> flag",
        "preprocessing": "all transforms, imputation, scaling, selection, and calibration are fit inside each training fold",
        "prohibited_model_columns": {
            "exact": list(PROHIBITED_EXACT_COLUMNS),
            "patterns": ["z_*", "canonical_v7__*", "new_review__*", "reviewer_*"],
            "reason": "identity/geometry, label, reviewer, post-label selection, or full-census preprocessing leakage",
        },
        "taxonomy_precomputed_z_columns": {
            "allowed_for_reproducing_frozen_taxonomy": True,
            "allowed_for_grouped_learning": False,
        },
        "reviewer_observations": {
            "size, morphology, low_snr, artifact proximity, and adjacent-source notes": "label-side subgroup metadata only",
        },
    }
    payload["feature_dictionary_sha256"] = stable_hash(payload)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = tuple(reader.fieldnames or ())
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"invalid or duplicate TSV columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty TSV table: {path}")
    return columns, rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_columns(actual: Sequence[str], required: Sequence[str], name: str, *, allow_additional: bool) -> None:
    missing = sorted(set(required) - set(actual))
    additional = sorted(set(actual) - set(required))
    if missing or (additional and not allow_additional):
        raise ValueError(f"{name} schema mismatch; missing={missing}, additional={additional}")


def _finite_float(value: Any, field: str, row_id: str, *, optional: bool = False) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        if optional:
            return None
        raise ValueError(f"missing {field} for {row_id}")
    parsed = float(text)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {field} for {row_id}")
    return parsed


def _integer(value: Any, field: str, row_id: str, *, optional: bool = False) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        if optional:
            return None
        raise ValueError(f"missing {field} for {row_id}")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"non-integer {field} for {row_id}: {value!r}") from exc
    return parsed


def _truth(value: Any, field: str, row_id: str) -> bool:
    text = str(value).strip().casefold()
    if text not in {"true", "false"}:
        raise ValueError(f"expected true/false {field} for {row_id}, got {value!r}")
    return text == "true"


def _unique(rows: Iterable[Mapping[str, Any]], key: str, table: str) -> None:
    values = [str(row[key]) for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        duplicates = sorted(value for value, count in Counter(values).items() if not value or count > 1)
        raise ValueError(f"{table} has empty or duplicate {key}: {duplicates}")


def _median(values: Iterable[float | None]) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _parse_occurrences(raw: list[dict[str, str]]) -> list[dict[str, Any]]:
    _unique(raw, "detection_occurrence_id", "occurrence table")
    rows = []
    for source in raw:
        row_id = source["detection_occurrence_id"]
        burst = _integer(source["burst_id"], "burst_id", row_id)
        start = _integer(source["start_ui"], "start_ui", row_id)
        end = _integer(source["end_ui"], "end_ui", row_id)
        peak = _integer(source["peak_frame_ui"], "peak_frame_ui", row_id)
        assert burst is not None and start is not None and end is not None and peak is not None
        if burst < 1 or start < 1 or not start <= peak <= end:
            raise ValueError(f"invalid burst/frame contract for {row_id}")
        features = {
            name: _finite_float(source[name], name, row_id, optional=True)
            for name in RAW_FEATURE_COLUMNS
        }
        rows.append(
            {
                "detection_occurrence_id": row_id,
                "detection_site_id": str(source["detection_site_id"]),
                "burst_id": burst,
                "x_px": _finite_float(source["x_px"], "x_px", row_id),
                "y_px": _finite_float(source["y_px"], "y_px", row_id),
                "start_ui": start,
                "end_ui": end,
                "peak_frame_ui": peak,
                "features": features,
                "proposal_metadata": {
                    "coherence_rank": _integer(source["coherence_rank"], "coherence_rank", row_id, optional=True),
                    "propagation_rank": _integer(source["propagation_rank"], "propagation_rank", row_id, optional=True),
                    "coherence_score": _finite_float(source["coherence_score"], "coherence_score", row_id, optional=True),
                    "propagation_score": _finite_float(source["propagation_score"], "propagation_score", row_id, optional=True),
                },
                "taxonomy_metadata": {
                    "class_id": _integer(source["class_id"], "class_id", row_id),
                    "class_name": str(source["class_name"]),
                },
                "postfreeze_known_label_characterization": {
                    "nearest_distance_px": _finite_float(
                        source["nearest_known_positive_distance_px"],
                        "nearest_known_positive_distance_px",
                        row_id,
                    ),
                    "within_6px": _truth(source["known_positive_within_6px"], "known_positive_within_6px", row_id),
                    "nearest_observation_id": str(source["nearest_known_positive_id"]) or None,
                    "allowed_as_model_feature": False,
                },
            }
        )
    return sorted(rows, key=lambda row: (row["burst_id"], row["detection_occurrence_id"]))


def _parse_sites(raw: list[dict[str, str]]) -> list[dict[str, Any]]:
    _unique(raw, "detection_site_id", "site table")
    rows = []
    for source in raw:
        row_id = source["detection_site_id"]
        rows.append(
            {
                "detection_site_id": row_id,
                "occurrences": _integer(source["occurrences"], "occurrences", row_id),
                "bursts": _integer(source["bursts"], "bursts", row_id),
                "x_px": _finite_float(source["x_px"], "x_px", row_id),
                "y_px": _finite_float(source["y_px"], "y_px", row_id),
                "dominant_class": _integer(source["dominant_class"], "dominant_class", row_id),
                "class_consistency": _finite_float(source["class_consistency"], "class_consistency", row_id),
                "known_positive_occurrences": _integer(
                    source["known_positive_occurrences"], "known_positive_occurrences", row_id
                ),
                "median_raw_peak": _finite_float(source["median_raw_peak"], "median_raw_peak", row_id),
                "median_spatial_specificity": _finite_float(
                    source["median_spatial_specificity"], "median_spatial_specificity", row_id
                ),
            }
        )
    return sorted(rows, key=lambda row: row["detection_site_id"])


def _parse_canonical_labels(raw: list[dict[str, str]]) -> list[dict[str, Any]]:
    _unique(raw, "observation_id", "canonical-v7 label table")
    rows = []
    for source in raw:
        row_id = source["observation_id"]
        include_confirmed = _truth(source["include_confirmed"], "include_confirmed", row_id)
        include_inclusive = _truth(source["include_inclusive"], "include_inclusive", row_id)
        if include_confirmed and not include_inclusive:
            raise ValueError(f"confirmed canonical label is not inclusive: {row_id}")
        canonical = str(source["canonical_roi_id"]).strip()
        if not canonical:
            raise ValueError(f"missing canonical_roi_id for {row_id}")
        rows.append(
            {
                "observation_id": row_id,
                "burst_id": _integer(source["burst_id"], "burst_id", row_id),
                "original_roi_id": str(source["original_roi_id"]),
                "canonical_roi_id": canonical,
                "x_px": _finite_float(source["x_px"], "x_px", row_id),
                "y_px": _finite_float(source["y_px"], "y_px", row_id),
                "neuron_confidence": str(source["neuron_confidence"]),
                "morphology": str(source["morphology"]),
                "context": str(source["context"]),
                "disposition": str(source["disposition"]),
                "include_confirmed": include_confirmed,
                "include_inclusive": include_inclusive,
                "review_status": str(source["review_status"]),
                "reviewer_id": str(source["reviewer_id"]),
                "source_note": str(source["source_note"]),
            }
        )
    return sorted(rows, key=lambda row: (row["burst_id"], row["observation_id"]))


def _parse_new_reviews(raw: list[dict[str, str]]) -> list[dict[str, Any]]:
    _unique(raw, "blind_id", "new-review table")
    _unique(raw, "detection_site_id", "new-review table")
    rows = []
    for source in raw:
        blind_id = source["blind_id"]
        label = str(source["normalized_label"])
        if label not in NEW_REVIEW_LABELS:
            raise ValueError(f"unsupported new-review label for {blind_id}: {label!r}")
        confidence = _integer(source["confidence_1_to_5"], "confidence_1_to_5", blind_id)
        assert confidence is not None
        if not 1 <= confidence <= 5:
            raise ValueError(f"review confidence outside 1..5 for {blind_id}")
        rows.append(
            {
                "blind_id": blind_id,
                "detection_site_id": str(source["detection_site_id"]),
                "normalized_label": label,
                "confidence_1_to_5": confidence,
                "size_flag": str(source["size_flag"]) or None,
                "morphology": str(source["morphology"]) or None,
                "low_snr": _truth(source["low_snr"], "low_snr", blind_id),
                "artifact_proximity": _truth(source["artifact_proximity"], "artifact_proximity", blind_id),
                "overlap_or_adjacent_source": str(source["overlap_or_adjacent_source"]) or None,
                "verbatim_feedback": str(source["verbatim_feedback"]),
                "normalization_note": str(source["normalization_note"]),
                "role": "label_side_subgroup_metadata_not_model_features",
            }
        )
    return sorted(rows, key=lambda row: row["detection_site_id"])


def _validate_site_rollup(sites: list[dict[str, Any]], occurrences: list[dict[str, Any]]) -> None:
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        by_site[row["detection_site_id"]].append(row)
    expected_ids = {row["detection_site_id"] for row in sites}
    if set(by_site) != expected_ids:
        raise ValueError(
            f"occurrence/site key mismatch; orphan_occurrence_sites={sorted(set(by_site)-expected_ids)}, "
            f"empty_sites={sorted(expected_ids-set(by_site))}"
        )
    for site in sites:
        site_id = site["detection_site_id"]
        rows = by_site[site_id]
        classes = Counter(row["taxonomy_metadata"]["class_id"] for row in rows)
        maximum = max(classes.values())
        valid_dominant = {class_id for class_id, count in classes.items() if count == maximum}
        checks = {
            "occurrences": site["occurrences"] == len(rows),
            "bursts": site["bursts"] == len({row["burst_id"] for row in rows}),
            "x_px": math.isclose(site["x_px"], _median(row["x_px"] for row in rows), abs_tol=1e-9),
            "y_px": math.isclose(site["y_px"], _median(row["y_px"] for row in rows), abs_tol=1e-9),
            "dominant_class": site["dominant_class"] in valid_dominant,
            "class_consistency": math.isclose(site["class_consistency"], maximum / len(rows), abs_tol=1e-12),
            "known_positive_occurrences": site["known_positive_occurrences"]
            == sum(row["postfreeze_known_label_characterization"]["within_6px"] for row in rows),
            "median_raw_peak": math.isclose(
                site["median_raw_peak"], _median(row["features"]["raw_peak"] for row in rows), abs_tol=1e-9
            ),
            "median_spatial_specificity": math.isclose(
                site["median_spatial_specificity"],
                _median(row["features"]["spatial_specificity"] for row in rows),
                abs_tol=1e-9,
            ),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"site rollup mismatch for {site_id}: {failed}")


def _canonical_matches(
    occurrences: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    radius_px: float,
) -> tuple[dict[str, tuple[dict[str, Any], float]], dict[str, dict[str, Any]]]:
    """Return deterministic one-to-one matches plus all-radius ambiguity metadata."""
    pairs: list[tuple[float, str, str, int, int]] = []
    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for occurrence_index, occurrence in enumerate(occurrences):
        for label_index, label in enumerate(labels):
            if occurrence["burst_id"] != label["burst_id"]:
                continue
            distance = math.hypot(occurrence["x_px"] - label["x_px"], occurrence["y_px"] - label["y_px"])
            if distance <= radius_px:
                pairs.append(
                    (
                        distance,
                        occurrence["detection_occurrence_id"],
                        label["observation_id"],
                        occurrence_index,
                        label_index,
                    )
                )
                candidates[occurrence["detection_occurrence_id"]].append((distance, label))
    used_occurrences: set[int] = set()
    used_labels: set[int] = set()
    matches: dict[str, tuple[dict[str, Any], float]] = {}
    for distance, _, _, occurrence_index, label_index in sorted(pairs):
        if occurrence_index in used_occurrences or label_index in used_labels:
            continue
        used_occurrences.add(occurrence_index)
        used_labels.add(label_index)
        occurrence = occurrences[occurrence_index]
        matches[occurrence["detection_occurrence_id"]] = (labels[label_index], distance)
    ambiguity = {}
    for occurrence in occurrences:
        occurrence_id = occurrence["detection_occurrence_id"]
        values = sorted(candidates.get(occurrence_id, ()), key=lambda item: (item[0], item[1]["observation_id"]))
        ambiguity[occurrence_id] = {
            "candidate_count_within_radius": len(values),
            "candidate_observation_ids": tuple(item[1]["observation_id"] for item in values),
            "candidate_canonical_roi_ids": tuple(sorted({item[1]["canonical_roi_id"] for item in values})),
            "ambiguous_within_radius": len(values) > 1,
        }
    return matches, ambiguity


def _component_ids(site_ids: Sequence[str], edges: Iterable[tuple[str, str]], prefix: str) -> dict[str, str]:
    parent = {site_id: site_id for site_id in site_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for left, right in sorted((tuple(sorted(edge)) for edge in edges)):
        union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for site_id in sorted(site_ids):
        components[find(site_id)].append(site_id)
    ordered = sorted((tuple(values) for values in components.values()), key=lambda values: values)
    return {
        site_id: f"{prefix}_{index:03d}"
        for index, values in enumerate(ordered, 1)
        for site_id in values
    }


def _group_sites(
    sites: list[dict[str, Any]],
    *,
    spatial_neighbor_radius_px: float,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, tuple[str, ...]]]:
    site_ids = [row["detection_site_id"] for row in sites]
    spatial_edges: set[tuple[str, str]] = set()
    neighbors: dict[str, set[str]] = {site_id: set() for site_id in site_ids}
    for index, left in enumerate(sites):
        for right in sites[index + 1 :]:
            distance = math.hypot(left["x_px"] - right["x_px"], left["y_px"] - right["y_px"])
            if distance <= spatial_neighbor_radius_px:
                edge = tuple(sorted((left["detection_site_id"], right["detection_site_id"])))
                spatial_edges.add(edge)
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
    by_identity: dict[str, set[str]] = defaultdict(set)
    for site in sites:
        for identity in site["canonical_v7"]["canonical_roi_ids"]:
            by_identity[identity].add(site["detection_site_id"])
    identity_edges: set[tuple[str, str]] = set()
    for linked in by_identity.values():
        ordered = sorted(linked)
        identity_edges.update((ordered[0], item) for item in ordered[1:])
    identity = _component_ids(site_ids, identity_edges, "identity_group")
    spatial = _component_ids(site_ids, spatial_edges, "spatial_group")
    leakage = _component_ids(site_ids, identity_edges | spatial_edges, "leakage_group")
    return identity, spatial, leakage, {key: tuple(sorted(value)) for key, value in neighbors.items()}


def _canonical_site_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matches = [row["canonical_v7"] for row in rows if row["canonical_v7"]["match_status"] == "matched"]
    matched_identities = tuple(sorted({item["canonical_roi_id"] for item in matches}))
    candidate_identities = tuple(
        sorted(
            {
                identity
                for row in rows
                for identity in row["canonical_v7"]["candidate_canonical_roi_ids"]
            }
        )
    )
    return {
        "matched_occurrences": len(matches),
        "confirmed_occurrences": sum(item["include_confirmed"] for item in matches),
        "inclusive_occurrences": sum(item["include_inclusive"] for item in matches),
        "matched_canonical_roi_ids": matched_identities,
        "canonical_roi_ids": candidate_identities,
        "observation_ids": tuple(sorted(item["observation_id"] for item in matches)),
        "dispositions": tuple(sorted({item["disposition"] for item in matches})),
        "reviewer_ids": tuple(sorted({item["reviewer_id"] for item in matches})),
        "candidate_assisted": any(item["reviewer_id"] != "original_workbook" for item in matches),
        "selection_role": "post_census_label_only",
    }


def _cross_source_disagreement(canonical: Mapping[str, Any], review: Mapping[str, Any] | None) -> bool:
    if not review or canonical["confirmed_occurrences"] == 0:
        return False
    return review["normalized_label"] in {"uncertain", "artifact_or_noise"}


def _label_policy_row(site: Mapping[str, Any], policy: LabelPolicy) -> dict[str, Any]:
    canonical = site["canonical_v7"]
    review = site["new_review"]
    state = "unlabeled"
    reason = "no strict positive label; absence of a label is not a negative"
    positive_sources: list[str] = []

    if policy == "new_review_holdout" and review is not None:
        state = "excluded"
        reason = "all newly reviewed sites are held out by policy"
    elif policy == "canonical_only":
        if canonical["confirmed_occurrences"]:
            state = "positive"
            reason = "at least one geometrically matched canonical-v7 confirmed occurrence"
            positive_sources.append("canonical_v7_confirmed")
    elif review is not None:
        label = review["normalized_label"]
        if label == "definite_neuron" or (policy == "inclusive" and label == "probable_neuron"):
            state = "positive"
            reason = f"new-review {label} under {policy} policy"
            positive_sources.append(f"new_review_{label}")
        elif label == "artifact_or_noise":
            state = "excluded"
            reason = "single artifact-or-noise case study is not a representative verified-negative class"
        else:
            reason = f"new-review {label} remains unlabeled under {policy} policy"
    elif canonical["confirmed_occurrences"]:
        state = "positive"
        reason = "at least one geometrically matched canonical-v7 confirmed occurrence"
        positive_sources.append("canonical_v7_confirmed")

    return {
        "detection_site_id": site["detection_site_id"],
        "leakage_group_id": site["leakage_group_id"],
        "policy": policy,
        "state": state,
        "positive_sources": tuple(positive_sources),
        "reason": reason,
        "known_negative": False,
        "cross_source_overlap": review is not None and canonical["matched_occurrences"] > 0,
        "cross_source_disagreement": site["cross_source_disagreement"],
    }


def _readiness(
    site_rows: Sequence[dict[str, Any]],
    policy: LabelPolicy,
    requirements: ReadinessRequirements,
) -> dict[str, Any]:
    labels = [_label_policy_row(row, policy) for row in site_rows]
    states = Counter(row["state"] for row in labels)
    positive_groups = {row["leakage_group_id"] for row in labels if row["state"] == "positive"}
    unlabeled_groups = {row["leakage_group_id"] for row in labels if row["state"] == "unlabeled"}
    excluded_groups = {row["leakage_group_id"] for row in labels if row["state"] == "excluded"}
    failures = []
    if len(positive_groups) < requirements.minimum_positive_groups:
        failures.append(
            f"positive leakage groups {len(positive_groups)} < required {requirements.minimum_positive_groups}"
        )
    if len(unlabeled_groups) < requirements.minimum_unlabeled_groups:
        failures.append(
            f"unlabeled leakage groups {len(unlabeled_groups)} < required {requirements.minimum_unlabeled_groups}"
        )
    return {
        "passed": not failures,
        "policy": policy,
        "site_counts": dict(sorted(states.items())),
        "positive_leakage_groups": len(positive_groups),
        "unlabeled_leakage_groups": len(unlabeled_groups),
        "excluded_leakage_groups": len(excluded_groups),
        "requirements": {
            "outer_folds": requirements.outer_folds,
            "minimum_positive_groups": requirements.minimum_positive_groups,
            "minimum_unlabeled_groups": requirements.minimum_unlabeled_groups,
            "rationale": "at least two independent positive and unlabeled leakage groups per requested outer fold",
        },
        "failures": failures,
        "scope_if_passed": "within-recording generalization to held-out identity/spatial groups only",
    }


def _source_manifest(
    inputs: HarmonizationInputs,
    columns: Mapping[str, Sequence[str] | None],
    row_counts: Mapping[str, int | None],
    *,
    repository_root: Path,
    data_root: Path | None,
) -> dict[str, Any]:
    sources = {}
    for name, path in inputs.items():
        sources[name] = {
            "uri": portable_path(path, repository=repository_root, data=data_root),
            "sha256": _sha256_file(path),
            "columns": list(columns[name]) if columns[name] is not None else None,
            "row_count": row_counts[name],
        }
    return dict(sorted(sources.items()))


def harmonize_feature_census(
    inputs: HarmonizationInputs,
    contract: ProposalCensusContract,
    *,
    repository_root: Path,
    data_root: Path | None = None,
    canonical_match_radius_px: float = 6.0,
    spatial_neighbor_radius_px: float = 12.0,
    readiness_requirements: ReadinessRequirements = ReadinessRequirements(),
) -> HarmonizedFeatureCensus:
    """Join one frozen proposal census to labels and validate every grain.

    The function performs no writes.  A future B58 census can use this same API
    provided it materializes the same raw-profile schema and a matching frozen
    proposal contract before either label file is opened.
    """
    if canonical_match_radius_px <= 0 or spatial_neighbor_radius_px <= 0:
        raise ValueError("match and spatial-neighbor radii must be positive")
    repository_root = repository_root.resolve()
    data_root = data_root.resolve() if data_root is not None else None

    occurrence_columns, raw_occurrences = _read_tsv(inputs.occurrence_profiles)
    site_columns, raw_sites = _read_tsv(inputs.site_profiles)
    canonical_columns, raw_canonical = _read_tsv(inputs.canonical_v7_labels)
    review_columns, raw_reviews = _read_tsv(inputs.new_review_labels)
    taxonomy_summary = _read_json(inputs.taxonomy_summary)
    taxonomy_validation = _read_json(inputs.taxonomy_validation)

    _require_columns(
        occurrence_columns,
        OCCURRENCE_COLUMNS_V1,
        "occurrence profile",
        allow_additional=contract.allow_additional_source_columns,
    )
    _require_columns(
        site_columns,
        SITE_COLUMNS_V1,
        "site profile",
        allow_additional=contract.allow_additional_source_columns,
    )
    _require_columns(canonical_columns, CANONICAL_V7_REQUIRED_COLUMNS, "canonical-v7 labels", allow_additional=True)
    _require_columns(review_columns, NEW_REVIEW_COLUMNS_V1, "new-review labels", allow_additional=False)

    if tuple(taxonomy_summary.get("features", ())) != TAXONOMY_FEATURE_NAMES:
        raise ValueError("taxonomy feature definitions disagree with the raw-feature contract")
    if taxonomy_summary.get("fitting_labels_used") is not False:
        raise ValueError("taxonomy summary does not prove label-free fitting")
    expected_validation = {
        "budget_per_burst_per_lane": contract.fixed_candidates_per_burst,
        "strict_nms_radius_px": contract.nms_radius_px,
        "labels_excluded_from_fit": True,
    }
    for key, expected in expected_validation.items():
        if taxonomy_validation.get(key) != expected:
            raise ValueError(f"taxonomy validation {key!r} disagrees with proposal contract")

    occurrences = _parse_occurrences(raw_occurrences)
    sites = _parse_sites(raw_sites)
    canonical = _parse_canonical_labels(raw_canonical)
    reviews = _parse_new_reviews(raw_reviews)
    _validate_site_rollup(sites, occurrences)
    if taxonomy_summary.get("detection_occurrences") != len(occurrences):
        raise ValueError("taxonomy summary occurrence count mismatch")
    if taxonomy_summary.get("detection_sites") != len(sites):
        raise ValueError("taxonomy summary site count mismatch")

    site_ids = {row["detection_site_id"] for row in sites}
    review_site_ids = {row["detection_site_id"] for row in reviews}
    if not review_site_ids <= site_ids:
        raise ValueError(f"new-review rows reference sites outside the frozen census: {sorted(review_site_ids-site_ids)}")

    matches, ambiguities = _canonical_matches(occurrences, canonical, canonical_match_radius_px)
    review_by_site = {row["detection_site_id"]: row for row in reviews}
    occurrence_with_labels = []
    for occurrence in occurrences:
        occurrence_id = occurrence["detection_occurrence_id"]
        matched = matches.get(occurrence_id)
        if matched is None:
            canonical_payload = {
                "match_status": "unmatched_unknown",
                "observation_id": None,
                "canonical_roi_id": None,
                "match_distance_px": None,
                "include_confirmed": False,
                "include_inclusive": False,
                "disposition": None,
                "review_status": None,
                "reviewer_id": None,
                "candidate_assisted": None,
                **ambiguities[occurrence_id],
            }
        else:
            label, distance = matched
            canonical_payload = {
                "match_status": "matched",
                "observation_id": label["observation_id"],
                "canonical_roi_id": label["canonical_roi_id"],
                "match_distance_px": distance,
                "include_confirmed": label["include_confirmed"],
                "include_inclusive": label["include_inclusive"],
                "disposition": label["disposition"],
                "review_status": label["review_status"],
                "reviewer_id": label["reviewer_id"],
                "candidate_assisted": label["reviewer_id"] != "original_workbook",
                "label_side_metadata": {
                    "neuron_confidence": label["neuron_confidence"],
                    "morphology": label["morphology"],
                    "context": label["context"],
                    "source_note": label["source_note"],
                    "allowed_as_model_features": False,
                },
                **ambiguities[occurrence_id],
            }
        occurrence_with_labels.append(
            {
                **occurrence,
                "canonical_v7": canonical_payload,
                "new_review": review_by_site.get(occurrence["detection_site_id"]),
            }
        )

    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrence_with_labels:
        by_site[row["detection_site_id"]].append(row)
    harmonized_sites = []
    for source_site in sites:
        site_id = source_site["detection_site_id"]
        rows = sorted(by_site[site_id], key=lambda row: (row["burst_id"], row["detection_occurrence_id"]))
        canonical_summary = _canonical_site_summary(rows)
        review = review_by_site.get(site_id)
        harmonized_sites.append(
            {
                "detection_site_id": site_id,
                "x_px": source_site["x_px"],
                "y_px": source_site["y_px"],
                "occurrence_ids": tuple(row["detection_occurrence_id"] for row in rows),
                "burst_ids": tuple(sorted({row["burst_id"] for row in rows})),
                "occurrence_count": len(rows),
                "features": {name: _median(row["features"][name] for row in rows) for name in RAW_FEATURE_COLUMNS},
                "feature_missing_counts": {
                    name: sum(row["features"][name] is None for row in rows) for name in RAW_FEATURE_COLUMNS
                },
                "taxonomy_metadata": {
                    "dominant_class": source_site["dominant_class"],
                    "class_consistency": source_site["class_consistency"],
                    "allowed_as_model_features": False,
                },
                "canonical_v7": canonical_summary,
                "new_review": review,
                "cross_source_disagreement": _cross_source_disagreement(canonical_summary, review),
            }
        )

    identity_groups, spatial_groups, leakage_groups, neighbors = _group_sites(
        harmonized_sites,
        spatial_neighbor_radius_px=spatial_neighbor_radius_px,
    )
    grouped_sites = []
    for site in harmonized_sites:
        site_id = site["detection_site_id"]
        grouped_sites.append(
            {
                **site,
                "identity_group_id": identity_groups[site_id],
                "spatial_group_id": spatial_groups[site_id],
                "leakage_group_id": leakage_groups[site_id],
                "spatial_neighbor_site_ids": neighbors[site_id],
            }
        )
    site_group_lookup = {row["detection_site_id"]: row for row in grouped_sites}
    grouped_occurrences = tuple(
        {
            **row,
            "identity_group_id": site_group_lookup[row["detection_site_id"]]["identity_group_id"],
            "spatial_group_id": site_group_lookup[row["detection_site_id"]]["spatial_group_id"],
            "leakage_group_id": site_group_lookup[row["detection_site_id"]]["leakage_group_id"],
        }
        for row in occurrence_with_labels
    )
    grouped_sites_tuple = tuple(grouped_sites)

    dictionary = feature_dictionary()
    columns = {
        "occurrence_profiles": occurrence_columns,
        "site_profiles": site_columns,
        "taxonomy_summary": None,
        "taxonomy_validation": None,
        "canonical_v7_labels": canonical_columns,
        "new_review_labels": review_columns,
    }
    row_counts = {
        "occurrence_profiles": len(raw_occurrences),
        "site_profiles": len(raw_sites),
        "taxonomy_summary": None,
        "taxonomy_validation": None,
        "canonical_v7_labels": len(raw_canonical),
        "new_review_labels": len(raw_reviews),
    }
    source_files = _source_manifest(
        inputs,
        columns,
        row_counts,
        repository_root=repository_root,
        data_root=data_root,
    )
    reviewed_sites = [row for row in grouped_sites_tuple if row["new_review"] is not None]
    source_overlap = {
        "reviewed_sites": len(reviewed_sites),
        "reviewed_sites_with_any_canonical_v7_match": sum(
            row["canonical_v7"]["matched_occurrences"] > 0 for row in reviewed_sites
        ),
        "reviewed_sites_with_confirmed_canonical_v7_match": sum(
            row["canonical_v7"]["confirmed_occurrences"] > 0 for row in reviewed_sites
        ),
        "reviewed_sites_with_cross_source_disagreement": sum(
            row["cross_source_disagreement"] for row in reviewed_sites
        ),
        "interpretation": "label sources overlap and are not independent replicates; neither source is silently overwritten",
    }
    readiness = {
        policy: _readiness(grouped_sites_tuple, policy, readiness_requirements)
        for policy in LABEL_POLICIES
    }
    missingness = {
        name: {
            "occurrence_missing": sum(row["features"][name] is None for row in grouped_occurrences),
            "site_missing": sum(row["features"][name] is None for row in grouped_sites_tuple),
        }
        for name in RAW_FEATURE_COLUMNS
    }
    validation = {
        "schema_version": 1,
        "table_integrity": "passed",
        "proposal_frozen_before_labels": True,
        "labels_used_to_build_census": False,
        "occurrence_rows": len(grouped_occurrences),
        "site_rows": len(grouped_sites_tuple),
        "canonical_v7_matched_occurrences": len(matches),
        "canonical_v7_unmatched_occurrences_remain_unknown": len(grouped_occurrences) - len(matches),
        "canonical_match_ambiguity_occurrences": sum(
            row["canonical_v7"]["ambiguous_within_radius"] for row in grouped_occurrences
        ),
        "label_source_overlap": source_overlap,
        "missingness": missingness,
        "identity_group_count": len(set(identity_groups.values())),
        "spatial_group_count": len(set(spatial_groups.values())),
        "leakage_group_count": len(set(leakage_groups.values())),
        "model_readiness": readiness,
        "no_verified_negative_population": True,
        "unmatched_semantics": "unknown_not_negative",
    }
    manifest = {
        "schema_version": 1,
        "contract": contract.to_manifest(),
        "source_files": source_files,
        "join_contract": {
            "occurrence_to_site": "exact detection_site_id many-to-one with rollup reconciliation",
            "canonical_v7": "deterministic within-burst greedy one-to-one geometry match",
            "canonical_match_radius_px": canonical_match_radius_px,
            "new_review": "exact detection_site_id; copied to nested occurrence rows without becoming a feature",
            "spatial_neighbor_radius_px": spatial_neighbor_radius_px,
            "leakage_group": "connected component of shared canonical identity or spatial-neighbor edges",
        },
        "feature_dictionary_sha256": dictionary["feature_dictionary_sha256"],
        "occurrence_rows_sha256": stable_hash(grouped_occurrences),
        "site_rows_sha256": stable_hash(grouped_sites_tuple),
        "validation_sha256": stable_hash(validation),
        "interpretation_boundaries": [
            "single-recording within-recording grouped evaluation only",
            "candidate-selected yield is not full-field precision",
            "canonical-v7 candidate-assisted labels are not independent validation",
            "new-review unmatched or uncertain sites are unknown, not negative",
            "the artifact-or-noise review is a case study, not a representative negative class",
        ],
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    return HarmonizedFeatureCensus(
        occurrence_rows=grouped_occurrences,
        site_rows=grouped_sites_tuple,
        feature_dictionary=dictionary,
        manifest=manifest,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# Primary broad-census adapter
# ---------------------------------------------------------------------------

INNOVATION_CENSUS_METADATA_COLUMNS = (
    "candidate_id",
    "partition",
    "partition_id",
    "proposal_order",
    "burst_id",
    "quiet_map_id",
    "x_px",
    "y_px",
    "source_count",
)


@dataclass(frozen=True)
class InnovationCensusContract:
    """Frozen contract for the broad Innovation Ranker proposal union."""

    census_id: str
    expected_event_rows: int | None = 1_999
    expected_quiet_rows: int | None = 2_938
    expected_event_partition_counts: tuple[int, ...] | None = (544, 530, 476, 449)
    expected_quiet_partition_counts: tuple[int, ...] | None = (683, 748, 759, 748)
    expected_feature_names: tuple[str, ...] = tuple(INNOVATION_FEATURE_IDS)
    partition_ids: tuple[int, ...] = (1, 2, 3, 4)
    maximum_source_count: int = 22
    expected_reviewed_sites: int | None = 18
    positive_match_radius_px: float = 6.0
    unlabeled_exclusion_radius_px: float = 12.0
    spatial_cell_size_px: float = 12.0
    review_reserve_radius_px: float = 6.0
    proposal_frozen_before_labels: bool = True
    labels_used_to_build_census: bool = False

    def __post_init__(self) -> None:
        if not self.census_id:
            raise ValueError("census_id is required")
        if self.expected_event_rows is not None and self.expected_event_rows < 1:
            raise ValueError("expected_event_rows must be positive or None")
        if self.expected_quiet_rows is not None and self.expected_quiet_rows < 1:
            raise ValueError("expected_quiet_rows must be positive or None")
        if not self.expected_feature_names or len(self.expected_feature_names) != len(set(self.expected_feature_names)):
            raise ValueError("expected innovation feature names must be non-empty and unique")
        if not self.partition_ids or len(self.partition_ids) != len(set(self.partition_ids)):
            raise ValueError("partition IDs must be non-empty and unique")
        for name, counts, total in (
            ("event", self.expected_event_partition_counts, self.expected_event_rows),
            ("quiet", self.expected_quiet_partition_counts, self.expected_quiet_rows),
        ):
            if counts is None:
                continue
            if len(counts) != len(self.partition_ids) or min(counts) < 1:
                raise ValueError(f"expected {name} partition counts must align with partition_ids")
            if total is not None and sum(counts) != total:
                raise ValueError(f"expected {name} partition counts do not sum to the declared total")
        if self.maximum_source_count < 1:
            raise ValueError("maximum_source_count must be positive")
        if self.expected_reviewed_sites is not None and self.expected_reviewed_sites < 1:
            raise ValueError("expected_reviewed_sites must be positive or None")
        if min(
            self.positive_match_radius_px,
            self.unlabeled_exclusion_radius_px,
            self.spatial_cell_size_px,
            self.review_reserve_radius_px,
        ) <= 0:
            raise ValueError("all spatial radii must be positive")
        if self.unlabeled_exclusion_radius_px < self.positive_match_radius_px:
            raise ValueError("the unlabeled exclusion radius cannot be smaller than the positive match radius")
        if self.unlabeled_exclusion_radius_px < self.review_reserve_radius_px:
            raise ValueError("the unlabeled exclusion radius cannot be smaller than the review reserve radius")
        if not self.proposal_frozen_before_labels or self.labels_used_to_build_census:
            raise ValueError("innovation proposal census must be frozen without labels")

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


INNOVATION_RANKER_V5_CENSUS = InnovationCensusContract(
    census_id="innovation_ranker_v5_broad_union",
)


PRIMARY_OUTER_FOLD_CONTRACT = {
    "schema_version": 1,
    "strategy_id": "five_contiguous_group_representative_x_blocks_v1",
    "outer_folds": 5,
    "fold_ids": [1, 2, 3, 4, 5],
    "coordinate_axis": "x_px",
    "assignment_unit": "leakage_group_id",
    "group_representative": (
        "median x_px of unreserved positive-anchor rows for groups containing P; "
        "median x_px of all primary-eligible rows for U-only groups"
    ),
    "balance_target": "confirmed_canonical_primary_anchor_identity_groups",
    "boundary_construction": (
        "midpoints between adjacent assigned leakage-group representative x ranges; "
        "boundaries are frozen before model scoring"
    ),
    "row_extent_semantics": (
        "candidate rows may cross a representative-x boundary only as a persisted "
        "whole-group closure extension"
    ),
    "review_reservation_precedes_assignment": True,
    "boundary_purge_radius_px": 12.0,
    "boundary_selection": "one deterministic pre-model assignment; never tuned on performance",
    "fold_ids_assigned_by_harmonizer": False,
}


TabularSource = Path | Sequence[Mapping[str, Any]]


def _primary_group_geometry(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not members:
        raise ValueError("primary group geometry requires at least one eligible row")
    anchors = [row for row in members if bool(row.get("eligible_for_anchor_balance"))]
    representative_rows = anchors or list(members)
    xs = [float(row["x_px"]) for row in members]
    return {
        "representative_x_px": float(_median(row["x_px"] for row in representative_rows)),
        "representative_basis": (
            "unreserved_positive_anchor_rows" if anchors else "all_primary_eligible_rows"
        ),
        "positive_anchor_row_count": len(anchors),
        "minimum_x_px": min(xs),
        "maximum_x_px": max(xs),
        "eligible_row_count": len(members),
    }


@dataclass(frozen=True)
class HarmonizedInnovationCensus:
    """Broad event/quiet census with explicit training and audit roles."""

    rows: tuple[dict[str, Any], ...]
    feature_columns: tuple[str, ...]
    feature_dictionary: dict[str, Any]
    manifest: dict[str, Any]
    validation: dict[str, Any]

    def label_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "candidate_id": row["candidate_id"],
                "partition": row["partition"],
                "partition_id": row["partition_id"],
                "leakage_group_id": row["leakage_group_id"],
                "training_state": row["training_state"],
                "is_positive": row["training_state"] == "positive",
                "is_unlabeled": row["training_state"] == "unlabeled",
                "is_null_control": row["training_state"] == "null_control",
                "reserved_for_review_audit": row["training_state"].startswith("heldout_review"),
                "known_negative": False,
            }
            for row in self.rows
        )

    def all_feature_rows(self) -> tuple[dict[str, Any], ...]:
        """Return safe features for inspection without asserting model readiness."""
        output = []
        for item in self.rows:
            row = {
                "row_id": item["candidate_id"],
                "partition": item["partition"],
                "partition_id": item["partition_id"],
                "leakage_group_id": item["leakage_group_id"],
                "training_state": item["training_state"],
            }
            for name in self.feature_columns:
                value = item["features"][name]
                row[name] = value
                row[f"missing__{name}"] = value is None
            output.append(row)
        return tuple(output)

    def model_feature_rows(self, *, require_ready: bool = True) -> tuple[dict[str, Any], ...]:
        """Return event P/U rows only; fail closed by default on grouped support."""
        if require_ready:
            self.assert_model_ready()
        return tuple(
            row
            for row in self.all_feature_rows()
            if row["partition"] == "event" and row["training_state"] in {"positive", "unlabeled"}
        )

    def null_control_feature_rows(self) -> tuple[dict[str, Any], ...]:
        """Return source-off quiet rows separately from biological P/U rows."""
        return tuple(
            row
            for row in self.all_feature_rows()
            if row["partition"] == "quiet" and row["training_state"] == "null_control"
        )

    def fold_assignment_rows(self) -> tuple[dict[str, Any], ...]:
        """Return non-feature geometry/group metadata for fixed fold planning.

        This deliberately does not assign fold IDs or boundaries.  A runner can
        make the single preregistered contiguous-x assignment from these rows,
        keep each leakage component indivisible, and apply the declared 12-px
        boundary purge without exposing coordinates to the model.
        """
        rows = []
        for item in self.rows:
            if item["partition"] != "event":
                continue
            rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "partition_id": item["partition_id"],
                    "x_px": item["x_px"],
                    "y_px": item["y_px"],
                    "identity_group_id": item["identity_group_id"],
                    "spatial_group_id": item["spatial_group_id"],
                    "leakage_group_id": item["leakage_group_id"],
                    "training_state": item["training_state"],
                    "primary_anchor_canonical_roi_id": item["canonical_v7"]["canonical_roi_id"],
                    "eligible_for_primary_fit": item["training_state"] in {"positive", "unlabeled"},
                    "eligible_for_anchor_balance": item["training_state"] == "positive",
                    "reserved_for_review_audit": item["review_holdout"]["reserved"],
                }
            )
        eligible_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["eligible_for_primary_fit"]:
                eligible_by_group[row["leakage_group_id"]].append(row)
        group_geometry = {
            group_id: _primary_group_geometry(members)
            for group_id, members in eligible_by_group.items()
        }
        output = []
        for row in rows:
            geometry = group_geometry.get(row["leakage_group_id"])
            output.append(
                {
                    **row,
                    "primary_group_representative_x_px": (
                        None if geometry is None else geometry["representative_x_px"]
                    ),
                    "primary_group_representative_basis": (
                        None if geometry is None else geometry["representative_basis"]
                    ),
                    "primary_group_positive_anchor_row_count": (
                        0 if geometry is None else geometry["positive_anchor_row_count"]
                    ),
                    "primary_group_minimum_x_px": (
                        None if geometry is None else geometry["minimum_x_px"]
                    ),
                    "primary_group_maximum_x_px": (
                        None if geometry is None else geometry["maximum_x_px"]
                    ),
                    "primary_group_eligible_row_count": (
                        0 if geometry is None else geometry["eligible_row_count"]
                    ),
                }
            )
        return tuple(
            sorted(output, key=lambda row: (row["x_px"], row["y_px"], row["candidate_id"]))
        )

    @property
    def outer_fold_contract(self) -> dict[str, Any]:
        """Return the frozen assignment policy without materialized boundaries."""
        return dict(self.manifest["outer_fold_contract"])

    def validate_outer_fold_assignments(
        self, assignments: Mapping[str, int]
    ) -> dict[str, Any]:
        """Validate whole groups ordered into five representative-x blocks.

        Only primary-fit rows belong in ``assignments``.  Fold IDs must be
        1--5 and each leakage/identity component must stay intact.  Contiguity
        is evaluated on the unreserved positive-anchor median x for P groups
        and the all-eligible-row median x for U-only groups, not on every member
        row.  A whole group may extend across a midpoint boundary, but each such
        row is reported explicitly as a closure extension.  This validates a
        runner's preregistered assignment; it never chooses folds from scores.
        """
        rows = [row for row in self.fold_assignment_rows() if row["eligible_for_primary_fit"]]
        expected_ids = {row["candidate_id"] for row in rows}
        supplied_ids = set(assignments)
        if supplied_ids != expected_ids:
            raise ValueError(
                "outer-fold assignment row mismatch; "
                f"missing={sorted(expected_ids - supplied_ids)}, extra={sorted(supplied_ids - expected_ids)}"
            )
        fold_ids = tuple(self.outer_fold_contract["fold_ids"])
        invalid = sorted({int(value) for value in assignments.values()} - set(fold_ids))
        if invalid:
            raise ValueError(f"outer-fold assignment has invalid fold IDs: {invalid}")
        groups: dict[str, set[int]] = defaultdict(set)
        group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        identity_folds: dict[str, set[int]] = defaultdict(set)
        canonical_folds: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            fold_id = int(assignments[row["candidate_id"]])
            group_id = row["leakage_group_id"]
            groups[group_id].add(fold_id)
            group_rows[group_id].append(row)
            identity_folds[row["identity_group_id"]].add(fold_id)
            canonical_id = row["primary_anchor_canonical_roi_id"]
            if canonical_id is not None:
                canonical_folds[str(canonical_id)].add(fold_id)
        split = sorted(group for group, values in groups.items() if len(values) != 1)
        if split:
            raise ValueError(f"outer-fold assignment splits leakage groups: {split}")
        split_identities = sorted(
            identity for identity, values in identity_folds.items() if len(values) != 1
        )
        if split_identities:
            raise ValueError(f"outer-fold assignment splits identity groups: {split_identities}")
        split_canonical = sorted(
            identity for identity, values in canonical_folds.items() if len(values) != 1
        )
        if split_canonical:
            raise ValueError(f"outer-fold assignment splits canonical identities: {split_canonical}")

        group_records = []
        for group_id, members in sorted(group_rows.items()):
            fold_id = next(iter(groups[group_id]))
            geometry = _primary_group_geometry(members)
            group_records.append(
                {
                    "leakage_group_id": group_id,
                    "fold_id": fold_id,
                    **geometry,
                }
            )
        ordered_groups = sorted(
            group_records,
            key=lambda row: (row["representative_x_px"], row["leakage_group_id"]),
        )
        ordered_fold_ids = [int(row["fold_id"]) for row in ordered_groups]
        if ordered_fold_ids != sorted(ordered_fold_ids):
            raise ValueError("outer-fold assignment is not contiguous by group representative x")

        representative_bounds = {}
        boundaries = []
        for fold_id in fold_ids:
            values = [
                row["representative_x_px"]
                for row in ordered_groups
                if row["fold_id"] == fold_id
            ]
            if not values:
                raise ValueError(f"outer-fold assignment leaves fold {fold_id} empty")
            representative_bounds[str(fold_id)] = {
                "minimum_representative_x_px": min(values),
                "maximum_representative_x_px": max(values),
            }
        for left, right in zip(fold_ids, fold_ids[1:]):
            left_edge = representative_bounds[str(left)]["maximum_representative_x_px"]
            right_edge = representative_bounds[str(right)]["minimum_representative_x_px"]
            if left_edge > right_edge:
                raise ValueError("outer-fold representative x ranges interleave")
            boundaries.append((left_edge + right_edge) / 2.0)

        extension_rows = []
        extension_groups: set[str] = set()
        maximum_extension_px = 0.0
        for row in rows:
            assigned = int(assignments[row["candidate_id"]])
            coordinate_fold = bisect.bisect_right(boundaries, float(row["x_px"])) + 1
            if coordinate_fold == assigned:
                continue
            crossed = boundaries[min(assigned, coordinate_fold) - 1 : max(assigned, coordinate_fold) - 1]
            extension = max(
                (abs(float(row["x_px"]) - boundary) for boundary in crossed),
                default=0.0,
            )
            maximum_extension_px = max(maximum_extension_px, extension)
            extension_groups.add(row["leakage_group_id"])
            extension_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "leakage_group_id": row["leakage_group_id"],
                    "assigned_fold_id": assigned,
                    "coordinate_implied_fold_id": coordinate_fold,
                    "x_px": row["x_px"],
                    "maximum_boundary_extension_px": extension,
                }
            )
        raw_bounds = {
            str(fold_id): {
                "minimum_candidate_x_px": min(
                    row["x_px"]
                    for row in rows
                    if int(assignments[row["candidate_id"]]) == fold_id
                ),
                "maximum_candidate_x_px": max(
                    row["x_px"]
                    for row in rows
                    if int(assignments[row["candidate_id"]]) == fold_id
                ),
            }
            for fold_id in fold_ids
        }
        raw_extent_overlaps = [
            {
                "left_fold_id": left,
                "right_fold_id": right,
                "left_maximum_candidate_x_px": raw_bounds[str(left)]["maximum_candidate_x_px"],
                "right_minimum_candidate_x_px": raw_bounds[str(right)]["minimum_candidate_x_px"],
            }
            for left, right in zip(fold_ids, fold_ids[1:])
            if raw_bounds[str(left)]["maximum_candidate_x_px"]
            > raw_bounds[str(right)]["minimum_candidate_x_px"]
        ]
        result = {
            "passed": True,
            "row_count": len(rows),
            "leakage_group_count": len(groups),
            "identity_group_count": len(identity_folds),
            "canonical_anchor_identity_count": len(canonical_folds),
            "contiguity_unit": (
                "positive_anchor_median_x_for_P_groups_else_all_eligible_median_x"
            ),
            "representative_fold_bounds": representative_bounds,
            "representative_midpoint_boundaries_x_px": boundaries,
            "raw_candidate_fold_bounds": raw_bounds,
            "raw_candidate_extent_overlap_is_failure": False,
            "raw_candidate_extent_overlaps": raw_extent_overlaps,
            "whole_group_closure_extension": {
                "candidate_count": len(extension_rows),
                "leakage_group_count": len(extension_groups),
                "maximum_boundary_extension_px": maximum_extension_px,
                "rows": sorted(extension_rows, key=lambda row: row["candidate_id"]),
            },
            "assignment_sha256": stable_hash(dict(sorted(assignments.items()))),
            "group_representative_assignment_sha256": stable_hash(ordered_groups),
        }
        return result

    def assert_model_ready(self) -> None:
        readiness = self.validation["model_readiness"]
        if not readiness["passed"]:
            raise ValueError(
                "broad proposal census is not ready for grouped P/U learning: "
                + "; ".join(readiness["failures"])
            )


def innovation_feature_dictionary(feature_names: Sequence[str]) -> dict[str, Any]:
    """Describe the frozen Innovation Ranker features without re-transforming them."""
    memberships = {
        feature: tuple(sorted(name for name, values in INNOVATION_FEATURE_SETS.items() if feature in values))
        for feature in feature_names
    }
    features = []
    for feature in feature_names:
        features.append(
            {
                "name": feature,
                "source_column": f"feature__{feature}",
                "family_memberships": memberships.get(feature, ()),
                "directional_role": (
                    "negative_evidence" if feature in INNOVATION_NEGATIVE_EVIDENCE_IDS else "positive_or_context_evidence"
                ),
                "value_semantics": "frozen Innovation Ranker candidate-map sample",
                "harmonizer_transform": "none",
                "model_preprocessing": "fit imputation and scaling inside each training fold only",
                "missing_policy": "preserve_null_and_add_missing_indicator",
            }
        )
    payload = {
        "schema_version": 1,
        "feature_order": list(feature_names),
        "features": features,
        "excluded_metadata": {
            "proposal_order": "proposal-generation metadata; sensitivity only",
            "source_count": "proposal-source consensus metadata; sensitivity only",
            "partition and partition_id": "sampling strata and grouping only",
            "x_px and y_px": "geometry/group construction only",
        },
        "excluded_label_fields": [
            "canonical-v7 match fields",
            "new-review labels and reviewer attributes",
            "positive-radius and review-neighborhood flags",
        ],
        "quiet_partition_role": "source-off null control only; never biological negative or unlabeled event",
    }
    payload["feature_dictionary_sha256"] = stable_hash(payload)
    return payload


def _load_tabular_source(source: TabularSource, name: str) -> tuple[tuple[str, ...], list[dict[str, Any]], dict[str, Any]]:
    if isinstance(source, Path):
        columns, raw = _read_tsv(source)
        rows: list[dict[str, Any]] = [dict(row) for row in raw]
        provenance = {
            "kind": "tsv",
            "sha256": _sha256_file(source),
            "row_count": len(rows),
            "columns": list(columns),
        }
        return columns, rows, provenance
    rows = [dict(row) for row in source]
    if not rows:
        raise ValueError(f"empty in-memory table: {name}")
    columns = tuple(rows[0])
    if not columns or len(columns) != len(set(columns)) or any(tuple(row) != columns for row in rows):
        raise ValueError(f"in-memory {name} rows must have one stable ordered schema")
    provenance = {
        "kind": "in_memory_frozen_rows",
        "sha256": stable_hash(rows),
        "row_count": len(rows),
        "columns": list(columns),
    }
    return columns, rows, provenance


def _portable_tabular_provenance(
    source: TabularSource,
    provenance: dict[str, Any],
    name: str,
    *,
    repository_root: Path,
    data_root: Path | None,
) -> dict[str, Any]:
    result = dict(provenance)
    result["uri"] = (
        portable_path(source, repository=repository_root, data=data_root)
        if isinstance(source, Path)
        else f"memory://{name}"
    )
    return result


def _parse_innovation_rows(
    raw: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    contract: InnovationCensusContract,
) -> list[dict[str, Any]]:
    _unique(raw, "candidate_id", "innovation candidate census")
    rows = []
    for source in raw:
        candidate_id = str(source["candidate_id"])
        partition = str(source["partition"]).casefold()
        if partition not in {"event", "quiet"}:
            raise ValueError(f"invalid partition for {candidate_id}: {partition!r}")
        partition_id = _integer(source["partition_id"], "partition_id", candidate_id)
        proposal_order = _integer(source["proposal_order"], "proposal_order", candidate_id)
        source_count = _integer(source["source_count"], "source_count", candidate_id)
        assert partition_id is not None and proposal_order is not None and source_count is not None
        if (
            partition_id not in contract.partition_ids
            or proposal_order < 1
            or not 1 <= source_count <= contract.maximum_source_count
        ):
            raise ValueError(f"invalid proposal metadata for {candidate_id}")
        burst_id = _integer(source["burst_id"], "burst_id", candidate_id, optional=True)
        quiet_map_id = _integer(source["quiet_map_id"], "quiet_map_id", candidate_id, optional=True)
        if partition == "event" and (burst_id != partition_id or quiet_map_id is not None):
            raise ValueError(f"event partition metadata disagree for {candidate_id}")
        if partition == "quiet" and (quiet_map_id != partition_id or burst_id is not None):
            raise ValueError(f"quiet partition metadata disagree for {candidate_id}")
        x_px = _finite_float(source["x_px"], "x_px", candidate_id)
        y_px = _finite_float(source["y_px"], "y_px", candidate_id)
        assert x_px is not None and y_px is not None
        if x_px < 0 or y_px < 0 or not x_px.is_integer() or not y_px.is_integer():
            raise ValueError(f"innovation coordinates must be nonnegative integers for {candidate_id}")
        expected_candidate_id = (
            f"irv5_{partition}_p{partition_id:02d}_x{int(x_px):05d}_y{int(y_px):05d}"
        )
        if candidate_id != expected_candidate_id:
            raise ValueError(
                f"candidate_id is not coordinate-stable for {candidate_id}; expected {expected_candidate_id}"
            )
        features = {
            feature: _finite_float(source[f"feature__{feature}"], f"feature__{feature}", candidate_id)
            for feature in feature_names
        }
        rows.append(
            {
                "candidate_id": candidate_id,
                "partition": partition,
                "partition_id": partition_id,
                "proposal_order": proposal_order,
                "burst_id": burst_id,
                "quiet_map_id": quiet_map_id,
                "x_px": x_px,
                "y_px": y_px,
                "source_count": source_count,
                "features": features,
            }
        )
    for partition in ("event", "quiet"):
        for partition_id in contract.partition_ids:
            selected = [
                row
                for row in rows
                if row["partition"] == partition and row["partition_id"] == partition_id
            ]
            proposal_orders = sorted(row["proposal_order"] for row in selected)
            if proposal_orders != list(range(1, len(selected) + 1)):
                raise ValueError(
                    f"{partition} partition {partition_id} proposal_order is not contiguous and unique"
                )
            coordinates = {(row["x_px"], row["y_px"]) for row in selected}
            if len(coordinates) != len(selected):
                raise ValueError(f"{partition} partition {partition_id} has duplicate coordinates")
    order = {"event": 0, "quiet": 1}
    return sorted(rows, key=lambda row: (order[row["partition"]], row["partition_id"], row["proposal_order"], row["candidate_id"]))


def _greedy_candidate_label_matches(
    candidates: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    radius_px: float,
) -> tuple[dict[str, tuple[dict[str, Any], float]], dict[str, tuple[dict[str, Any], ...]]]:
    pairs: list[tuple[float, str, str, int, int]] = []
    within: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        if candidate["partition"] != "event":
            continue
        for label_index, label in enumerate(labels):
            if candidate["burst_id"] != label["burst_id"]:
                continue
            distance = math.hypot(candidate["x_px"] - label["x_px"], candidate["y_px"] - label["y_px"])
            if distance <= radius_px:
                pairs.append((distance, candidate["candidate_id"], str(label["observation_id"]), candidate_index, label_index))
                within[candidate["candidate_id"]].append((distance, label))
    used_candidates: set[int] = set()
    used_labels: set[int] = set()
    matches: dict[str, tuple[dict[str, Any], float]] = {}
    for distance, _, _, candidate_index, label_index in sorted(pairs):
        if candidate_index in used_candidates or label_index in used_labels:
            continue
        used_candidates.add(candidate_index)
        used_labels.add(label_index)
        matches[candidates[candidate_index]["candidate_id"]] = (labels[label_index], distance)
    ordered_within = {
        candidate_id: tuple(
            label for _, label in sorted(values, key=lambda item: (item[0], item[1]["observation_id"]))
        )
        for candidate_id, values in within.items()
    }
    return matches, ordered_within


def _review_targets(
    site_source: TabularSource | None,
    occurrence_source: TabularSource | None,
    review_source: TabularSource | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    values = (site_source, occurrence_source, review_source)
    if all(value is None for value in values):
        return [], [], {}, {"enabled": False, "reviewed_sites": 0, "review_occurrences": 0}
    if any(value is None for value in values):
        raise ValueError("review holdout requires site profiles, occurrence profiles, and new-review labels together")
    assert site_source is not None and occurrence_source is not None and review_source is not None
    site_columns, raw_sites, site_provenance = _load_tabular_source(site_source, "review_site_profiles")
    occurrence_columns, raw_occurrences, occurrence_provenance = _load_tabular_source(
        occurrence_source, "review_occurrence_profiles"
    )
    review_columns, raw_reviews, review_provenance = _load_tabular_source(review_source, "new_review_labels")
    _require_columns(site_columns, SITE_COLUMNS_V1, "review site profiles", allow_additional=False)
    _require_columns(occurrence_columns, OCCURRENCE_COLUMNS_V1, "review occurrence profiles", allow_additional=False)
    _require_columns(review_columns, NEW_REVIEW_COLUMNS_V1, "new-review labels", allow_additional=False)
    sites = _parse_sites(raw_sites)
    occurrences = _parse_occurrences(raw_occurrences)
    reviews = _parse_new_reviews(raw_reviews)
    _validate_site_rollup(sites, occurrences)
    site_by_id = {row["detection_site_id"]: row for row in sites}
    review_by_site = {row["detection_site_id"]: row for row in reviews}
    missing = sorted(set(review_by_site) - set(site_by_id))
    if missing:
        raise ValueError(f"review labels reference missing review sites: {missing}")
    reviewed_sites = [
        {
            "review_site_id": site_id,
            "x_px": site_by_id[site_id]["x_px"],
            "y_px": site_by_id[site_id]["y_px"],
            "review": review,
        }
        for site_id, review in sorted(review_by_site.items())
    ]
    reviewed_occurrences = []
    for occurrence in occurrences:
        site_id = occurrence["detection_site_id"]
        if site_id not in review_by_site:
            continue
        reviewed_occurrences.append(
            {
                "observation_id": f"review__{occurrence['detection_occurrence_id']}",
                "burst_id": occurrence["burst_id"],
                "x_px": occurrence["x_px"],
                "y_px": occurrence["y_px"],
                "review_site_id": site_id,
                "review": review_by_site[site_id],
            }
        )
    provenance = {
        "enabled": True,
        "reviewed_sites": len(reviewed_sites),
        "review_occurrences": len(reviewed_occurrences),
        "site_profiles": site_provenance,
        "occurrence_profiles": occurrence_provenance,
        "new_review_labels": review_provenance,
    }
    return reviewed_sites, reviewed_occurrences, review_by_site, provenance


def _nearest_review_site(
    candidate: Mapping[str, Any], reviewed_sites: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, float | None]:
    if not reviewed_sites:
        return None, None
    values = sorted(
        (
            math.hypot(float(candidate["x_px"]) - float(site["x_px"]), float(candidate["y_px"]) - float(site["y_px"])),
            str(site["review_site_id"]),
            site,
        )
        for site in reviewed_sites
    )
    return values[0][2], values[0][0]


def _review_canonical_overlap(
    reviewed_sites: Sequence[Mapping[str, Any]],
    reviewed_occurrences: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    radius_px: float,
) -> dict[str, Any]:
    any_overlap: set[str] = set()
    confirmed_overlap: set[str] = set()
    for occurrence in reviewed_occurrences:
        matches = [
            label
            for label in canonical
            if occurrence["burst_id"] == label["burst_id"]
            and math.hypot(
                occurrence["x_px"] - label["x_px"],
                occurrence["y_px"] - label["y_px"],
            )
            <= radius_px
        ]
        if matches:
            any_overlap.add(str(occurrence["review_site_id"]))
        if any(label["include_confirmed"] for label in matches):
            confirmed_overlap.add(str(occurrence["review_site_id"]))
    disagreements = {
        str(site["review_site_id"])
        for site in reviewed_sites
        if site["review_site_id"] in confirmed_overlap
        and site["review"]["normalized_label"] in {"uncertain", "artifact_or_noise"}
    }
    return {
        "reviewed_sites": len(reviewed_sites),
        "reviewed_sites_with_any_canonical_v7_match": len(any_overlap),
        "reviewed_sites_with_confirmed_canonical_v7_match": len(confirmed_overlap),
        "reviewed_sites_with_cross_source_disagreement": len(disagreements),
        "interpretation": (
            "label sources overlap and are not independent replicates; "
            "review reservations precede the primary label join"
        ),
    }


def _spatial_edges_grid(rows: Sequence[dict[str, Any]], radius_px: float) -> set[tuple[str, str]]:
    """Find all radius-neighbor edges with deterministic spatial hashing."""
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    edges: set[tuple[str, str]] = set()
    cell_size = float(radius_px)
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        cell = (math.floor(row["x_px"] / cell_size), math.floor(row["y_px"] / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    if math.hypot(row["x_px"] - other["x_px"], row["y_px"] - other["y_px"]) <= radius_px:
                        edges.add(tuple(sorted((row["candidate_id"], other["candidate_id"]))))
        cells[cell].append(row)
    return edges


def _innovation_groups(
    rows: list[dict[str, Any]],
    *,
    spatial_cell_size_px: float,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build bounded groups without transitive radius-chain percolation."""
    event_rows = [row for row in rows if row["partition"] == "event"]
    event_ids = [row["candidate_id"] for row in event_rows]
    by_cell: dict[tuple[int, int], list[str]] = defaultdict(list)
    by_identity: dict[str, list[str]] = defaultdict(list)
    for row in event_rows:
        cell = (
            math.floor(row["x_px"] / spatial_cell_size_px),
            math.floor(row["y_px"] / spatial_cell_size_px),
        )
        by_cell[cell].append(row["candidate_id"])
        for identity in row["canonical_v7"]["candidate_canonical_roi_ids"]:
            by_identity[f"canonical::{identity}"].append(row["candidate_id"])
        if (
            row["review_holdout"]["nearest_review_site_id"] is not None
            and row["review_holdout"]["excluded_from_unlabeled"]
        ):
            review_identity = f"review::{row['review_holdout']['nearest_review_site_id']}"
            by_identity[review_identity].append(row["candidate_id"])
    spatial_edges: set[tuple[str, str]] = set()
    spatial: dict[str, str] = {}
    for cell, values in sorted(by_cell.items()):
        ordered = sorted(set(values))
        spatial_edges.update((ordered[0], value) for value in ordered[1:])
        group_id = f"innovation_spatial_cell_x{cell[0]:04d}_y{cell[1]:04d}"
        spatial.update({value: group_id for value in ordered})
    identity_edges: set[tuple[str, str]] = set()
    for values in by_identity.values():
        ordered = sorted(set(values))
        identity_edges.update((ordered[0], value) for value in ordered[1:])
    identity = _component_ids(event_ids, identity_edges, "innovation_identity_group")
    leakage = _component_ids(event_ids, spatial_edges | identity_edges, "innovation_leakage_group")
    for row in rows:
        candidate_id = row["candidate_id"]
        if row["partition"] == "event":
            continue
        identity[candidate_id] = f"quiet_identity_p{row['partition_id']:02d}_{candidate_id}"
        spatial[candidate_id] = f"quiet_spatial_p{row['partition_id']:02d}_{candidate_id}"
        leakage[candidate_id] = f"quiet_null_control_p{row['partition_id']:02d}_{candidate_id}"
    return identity, spatial, leakage


def _group_diagnostics(rows: Sequence[dict[str, Any]], group_field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["partition"] == "event":
            grouped[row[group_field]].append(row)
    summaries = [
        {
            "group_id": group_id,
            "rows": len(values),
            "x_span_px": max(row["x_px"] for row in values) - min(row["x_px"] for row in values),
            "y_span_px": max(row["y_px"] for row in values) - min(row["y_px"] for row in values),
        }
        for group_id, values in sorted(grouped.items())
    ]
    by_size = max(summaries, key=lambda row: (row["rows"], row["group_id"]))
    by_x_span = max(summaries, key=lambda row: (row["x_span_px"], row["group_id"]))
    by_y_span = max(summaries, key=lambda row: (row["y_span_px"], row["group_id"]))
    return {
        "group_count": len(summaries),
        "maximum_group_size": by_size["rows"],
        "maximum_group_size_id": by_size["group_id"],
        "maximum_x_span_px": by_x_span["x_span_px"],
        "maximum_x_span_group_id": by_x_span["group_id"],
        "maximum_y_span_px": by_y_span["y_span_px"],
        "maximum_y_span_group_id": by_y_span["group_id"],
    }


def _innovation_readiness(
    rows: Sequence[dict[str, Any]], requirements: ReadinessRequirements
) -> dict[str, Any]:
    event = [row for row in rows if row["partition"] == "event"]
    states = Counter(row["training_state"] for row in rows)
    positive_groups = {row["leakage_group_id"] for row in event if row["training_state"] == "positive"}
    unlabeled_groups = {row["leakage_group_id"] for row in event if row["training_state"] == "unlabeled"}
    failures = []
    if len(positive_groups) < requirements.minimum_positive_groups:
        failures.append(f"positive leakage groups {len(positive_groups)} < required {requirements.minimum_positive_groups}")
    if len(unlabeled_groups) < requirements.minimum_unlabeled_groups:
        failures.append(f"unlabeled leakage groups {len(unlabeled_groups)} < required {requirements.minimum_unlabeled_groups}")
    if not any(row["training_state"] == "null_control" for row in rows):
        failures.append("source-off null-control partition is empty")
    return {
        "passed": not failures,
        "row_counts_by_state": dict(sorted(states.items())),
        "positive_leakage_groups": len(positive_groups),
        "unlabeled_leakage_groups": len(unlabeled_groups),
        "requirements": {
            "outer_folds": requirements.outer_folds,
            "minimum_positive_groups": requirements.minimum_positive_groups,
            "minimum_unlabeled_groups": requirements.minimum_unlabeled_groups,
        },
        "failures": failures,
        "scope_if_passed": "within-recording generalization to held-out identity/spatial groups only",
    }


def harmonize_innovation_candidate_census(
    candidate_source: TabularSource,
    canonical_v7_source: TabularSource,
    contract: InnovationCensusContract = INNOVATION_RANKER_V5_CENSUS,
    *,
    repository_root: Path,
    data_root: Path | None = None,
    review_site_source: TabularSource | None = None,
    review_occurrence_source: TabularSource | None = None,
    new_review_source: TabularSource | None = None,
    readiness_requirements: ReadinessRequirements = ReadinessRequirements(),
) -> HarmonizedInnovationCensus:
    """Harmonize the frozen broad union without turning quiet or ambiguity into U.

    ``candidate_source`` and label sources may be TSV paths or already-frozen
    row mappings.  Confirmed canonical-v7 occurrences receive one deterministic
    positive anchor.  Other event candidates inside a confirmed label radius
    are excluded rather than mislabeled U.  Canonical uncertain/artifact zones
    are also excluded.  When the B20 review sources are supplied, every event
    candidate in a reviewed-site neighborhood is reserved from fitting.
    """
    repository_root = repository_root.resolve()
    data_root = data_root.resolve() if data_root is not None else None
    if readiness_requirements.outer_folds != PRIMARY_OUTER_FOLD_CONTRACT["outer_folds"]:
        raise ValueError("the primary broad-census outer-fold count is frozen at five")
    candidate_columns, raw_candidates, candidate_provenance = _load_tabular_source(
        candidate_source, "innovation_candidate_census"
    )
    canonical_columns, raw_canonical, canonical_provenance = _load_tabular_source(
        canonical_v7_source, "canonical_v7_labels"
    )
    expected_feature_columns = tuple(f"feature__{name}" for name in contract.expected_feature_names)
    expected_columns = INNOVATION_CENSUS_METADATA_COLUMNS + expected_feature_columns
    _require_columns(candidate_columns, expected_columns, "innovation candidate census", allow_additional=False)
    _require_columns(canonical_columns, CANONICAL_V7_REQUIRED_COLUMNS, "canonical-v7 labels", allow_additional=True)
    feature_names = tuple(column.removeprefix("feature__") for column in candidate_columns if column.startswith("feature__"))
    if feature_names != contract.expected_feature_names:
        raise ValueError("innovation feature columns differ from the frozen contract")
    prohibited_names = [
        name
        for name in feature_names
        if name.startswith("z_")
        or name.endswith("_z")
        or any(token in name.casefold() for token in ("normalized", "standardized", "scaled"))
    ]
    if prohibited_names:
        raise ValueError(f"precomputed full-census feature names are prohibited: {prohibited_names}")

    candidates = _parse_innovation_rows(raw_candidates, feature_names, contract)
    canonical = _parse_canonical_labels(raw_canonical)
    event_count = sum(row["partition"] == "event" for row in candidates)
    quiet_count = sum(row["partition"] == "quiet" for row in candidates)
    if contract.expected_event_rows is not None and event_count != contract.expected_event_rows:
        raise ValueError(f"event row count {event_count} != frozen {contract.expected_event_rows}")
    if contract.expected_quiet_rows is not None and quiet_count != contract.expected_quiet_rows:
        raise ValueError(f"quiet row count {quiet_count} != frozen {contract.expected_quiet_rows}")
    observed_partition_counts: dict[str, tuple[int, ...]] = {}
    for partition in ("event", "quiet"):
        observed = {row["partition_id"] for row in candidates if row["partition"] == partition}
        if observed != set(contract.partition_ids):
            raise ValueError(f"{partition} partition IDs differ from frozen contract: {sorted(observed)}")
        observed_partition_counts[partition] = tuple(
            sum(
                row["partition"] == partition and row["partition_id"] == partition_id
                for row in candidates
            )
            for partition_id in contract.partition_ids
        )
    for partition, expected in (
        ("event", contract.expected_event_partition_counts),
        ("quiet", contract.expected_quiet_partition_counts),
    ):
        if expected is not None and observed_partition_counts[partition] != expected:
            raise ValueError(
                f"{partition} partition counts {observed_partition_counts[partition]} "
                f"!= frozen {expected}"
            )

    reviewed_sites, reviewed_occurrences, _, review_provenance = _review_targets(
        review_site_source, review_occurrence_source, new_review_source
    )
    if reviewed_sites and (
        contract.expected_reviewed_sites is not None
        and len(reviewed_sites) != contract.expected_reviewed_sites
    ):
        raise ValueError(
            f"reviewed site count {len(reviewed_sites)} != frozen {contract.expected_reviewed_sites}"
        )
    if review_provenance["enabled"]:
        assert review_site_source is not None
        assert review_occurrence_source is not None
        assert new_review_source is not None
        for key, source in (
            ("site_profiles", review_site_source),
            ("occurrence_profiles", review_occurrence_source),
            ("new_review_labels", new_review_source),
        ):
            review_provenance[key] = _portable_tabular_provenance(
                source,
                review_provenance[key],
                f"review_holdout_{key}",
                repository_root=repository_root,
                data_root=data_root,
            )
    confirmed_labels = [row for row in canonical if row["include_confirmed"]]
    nonconfirmed_labels = [row for row in canonical if not row["include_confirmed"]]
    confirmed_matches, confirmed_match_within = _greedy_candidate_label_matches(
        candidates, confirmed_labels, contract.positive_match_radius_px
    )
    _, confirmed_exclusion_within = _greedy_candidate_label_matches(
        candidates, confirmed_labels, contract.unlabeled_exclusion_radius_px
    )
    _, nonconfirmed_exclusion_within = _greedy_candidate_label_matches(
        candidates, nonconfirmed_labels, contract.unlabeled_exclusion_radius_px
    )
    review_matches, review_within = _greedy_candidate_label_matches(
        candidates, reviewed_occurrences, contract.positive_match_radius_px
    ) if reviewed_occurrences else ({}, {})

    labeled_rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        confirmed_match = confirmed_matches.get(candidate_id)
        confirmed_match_candidates = confirmed_match_within.get(candidate_id, ())
        confirmed_exclusion_candidates = confirmed_exclusion_within.get(candidate_id, ())
        nonconfirmed_exclusion_candidates = nonconfirmed_exclusion_within.get(candidate_id, ())
        nearest_review, review_distance = _nearest_review_site(candidate, reviewed_sites)
        review_reserved = bool(
            candidate["partition"] == "event"
            and nearest_review is not None
            and review_distance is not None
            and review_distance <= contract.review_reserve_radius_px
        )
        review_excluded_from_unlabeled = bool(
            candidate["partition"] == "event"
            and nearest_review is not None
            and review_distance is not None
            and review_distance <= contract.unlabeled_exclusion_radius_px
        )
        review_match = review_matches.get(candidate_id)
        if candidate["partition"] == "quiet":
            training_state = "null_control"
        elif review_reserved:
            training_state = "heldout_review_anchor" if review_match is not None else "heldout_review_neighborhood"
        elif confirmed_match is not None:
            training_state = "positive"
        elif confirmed_exclusion_candidates:
            training_state = "excluded_canonical_confirmed_radius"
        elif nonconfirmed_exclusion_candidates:
            training_state = "excluded_canonical_nonconfirmed_radius"
        elif review_excluded_from_unlabeled:
            training_state = "excluded_review_radius"
        else:
            training_state = "unlabeled"

        if confirmed_match is None:
            confirmed_payload = {
                "anchor_status": "not_anchor",
                "observation_id": None,
                "canonical_roi_id": None,
                "match_distance_px": None,
            }
        else:
            label, distance = confirmed_match
            confirmed_payload = {
                "anchor_status": "confirmed_positive_anchor",
                "observation_id": label["observation_id"],
                "canonical_roi_id": label["canonical_roi_id"],
                "match_distance_px": distance,
                "reviewer_id": label["reviewer_id"],
                "candidate_assisted": label["reviewer_id"] != "original_workbook",
            }
        review_payload = {
            "reserved": review_reserved,
            "excluded_from_unlabeled": review_excluded_from_unlabeled,
            "nearest_review_site_id": None if nearest_review is None else nearest_review["review_site_id"],
            "nearest_review_distance_px": review_distance,
            "anchor_status": "review_anchor" if review_match is not None else "not_anchor",
            "review_label": None if nearest_review is None else nearest_review["review"]["normalized_label"],
            "role": (
                "heldout_audit_only"
                if review_reserved
                else "primary_unlabeled_exclusion_buffer"
                if review_excluded_from_unlabeled
                else "label_side_context_only"
            ),
        }
        labeled_rows.append(
            {
                **candidate,
                "training_state": training_state,
                "canonical_v7": {
                    **confirmed_payload,
                    "inside_confirmed_match_radius": bool(confirmed_match_candidates),
                    "inside_confirmed_exclusion_radius": bool(confirmed_exclusion_candidates),
                    "inside_nonconfirmed_exclusion_radius": bool(nonconfirmed_exclusion_candidates),
                    "candidate_observation_ids": tuple(
                        sorted(
                            {
                                row["observation_id"]
                                for row in (
                                    *confirmed_exclusion_candidates,
                                    *nonconfirmed_exclusion_candidates,
                                )
                            }
                        )
                    ),
                    "candidate_canonical_roi_ids": tuple(
                        sorted(
                            {
                                row["canonical_roi_id"]
                                for row in (
                                    *confirmed_exclusion_candidates,
                                    *nonconfirmed_exclusion_candidates,
                                )
                            }
                        )
                    ),
                },
                "review_holdout": review_payload,
                "known_negative": False,
            }
        )

    identity, spatial, leakage = _innovation_groups(
        labeled_rows, spatial_cell_size_px=contract.spatial_cell_size_px
    )
    grouped_rows = tuple(
        {
            **row,
            "identity_group_id": identity[row["candidate_id"]],
            "spatial_group_id": spatial[row["candidate_id"]],
            "leakage_group_id": leakage[row["candidate_id"]],
        }
        for row in labeled_rows
    )
    readiness = _innovation_readiness(grouped_rows, readiness_requirements)
    feature_info = innovation_feature_dictionary(feature_names)
    review_overlap = _review_canonical_overlap(
        reviewed_sites,
        reviewed_occurrences,
        canonical,
        contract.positive_match_radius_px,
    )
    candidate_source_manifest = _portable_tabular_provenance(
        candidate_source,
        candidate_provenance,
        "innovation_candidate_census",
        repository_root=repository_root,
        data_root=data_root,
    )
    canonical_source_manifest = _portable_tabular_provenance(
        canonical_v7_source,
        canonical_provenance,
        "canonical_v7_labels",
        repository_root=repository_root,
        data_root=data_root,
    )
    validation = {
        "schema_version": 1,
        "table_integrity": "passed",
        "proposal_frozen_before_labels": True,
        "labels_used_to_build_census": False,
        "candidate_rows": len(grouped_rows),
        "event_rows": event_count,
        "quiet_rows": quiet_count,
        "partition_counts": {
            partition: {
                str(partition_id): count
                for partition_id, count in zip(
                    contract.partition_ids, observed_partition_counts[partition]
                )
            }
            for partition in ("event", "quiet")
        },
        "feature_count": len(feature_names),
        "confirmed_canonical_labels": len(confirmed_labels),
        "confirmed_positive_anchors": len(confirmed_matches),
        "primary_positive_anchors_after_review_reserve": sum(
            row["training_state"] == "positive" for row in grouped_rows
        ),
        "confirmed_anchors_reserved_for_review": sum(
            row["review_holdout"]["reserved"]
            and row["canonical_v7"]["anchor_status"] == "confirmed_positive_anchor"
            for row in grouped_rows
        ),
        "confirmed_labels_without_anchor": len(confirmed_labels) - len(confirmed_matches),
        "unchosen_candidates_inside_positive_match_radius": sum(
            row["canonical_v7"]["inside_confirmed_match_radius"]
            and row["canonical_v7"]["anchor_status"] != "confirmed_positive_anchor"
            for row in grouped_rows
        ),
        "canonical_confirmed_radius_exclusions": sum(
            row["training_state"] == "excluded_canonical_confirmed_radius" for row in grouped_rows
        ),
        "canonical_nonconfirmed_radius_exclusions": sum(
            row["training_state"] == "excluded_canonical_nonconfirmed_radius" for row in grouped_rows
        ),
        "review_holdout": {
            "enabled": review_provenance["enabled"],
            "reviewed_sites": review_provenance["reviewed_sites"],
            "review_occurrences": review_provenance["review_occurrences"],
            "reserved_event_candidates": sum(row["review_holdout"]["reserved"] for row in grouped_rows),
            "buffered_event_candidates": sum(
                row["review_holdout"]["excluded_from_unlabeled"] for row in grouped_rows
            ),
            "buffer_only_event_candidates": sum(
                row["review_holdout"]["excluded_from_unlabeled"]
                and not row["review_holdout"]["reserved"]
                for row in grouped_rows
            ),
            "review_anchors": len(review_matches),
            "all_reviewed_neighborhoods_excluded_from_fitting": all(
                not row["review_holdout"]["reserved"] or row["training_state"].startswith("heldout_review")
                for row in grouped_rows
            ),
            "canonical_v7_overlap": review_overlap,
        },
        "quiet_semantics": "source_off_null_control_not_negative_not_unlabeled",
        "unmatched_event_semantics": "unknown_not_negative",
        "model_readiness": readiness,
        "group_counts": {
            "identity": len({row["identity_group_id"] for row in grouped_rows if row["partition"] == "event"}),
            "spatial": len({row["spatial_group_id"] for row in grouped_rows if row["partition"] == "event"}),
            "leakage": len({row["leakage_group_id"] for row in grouped_rows if row["partition"] == "event"}),
        },
        "group_diagnostics": {
            "identity": _group_diagnostics(grouped_rows, "identity_group_id"),
            "spatial_cell": _group_diagnostics(grouped_rows, "spatial_group_id"),
            "leakage": _group_diagnostics(grouped_rows, "leakage_group_id"),
            "radius_chain_components_used": False,
        },
        "missingness": {
            feature: sum(row["features"][feature] is None for row in grouped_rows) for feature in feature_names
        },
    }
    manifest = {
        "schema_version": 1,
        "contract": contract.to_manifest(),
        "source_files": {
            "candidate_census": candidate_source_manifest,
            "canonical_v7_labels": canonical_source_manifest,
            "review_holdout_sources": review_provenance,
        },
        "outer_fold_contract": {
            **PRIMARY_OUTER_FOLD_CONTRACT,
            "boundary_purge_radius_px": contract.unlabeled_exclusion_radius_px,
            "outer_folds": readiness_requirements.outer_folds,
        },
        "feature_dictionary_sha256": feature_info["feature_dictionary_sha256"],
        "rows_sha256": stable_hash(grouped_rows),
        "validation_sha256": stable_hash(validation),
        "label_join_order": [
            "freeze label-free event and quiet proposal union",
            "select one confirmed canonical-v7 anchor per occurrence",
            "reserve candidates within 6 px of all reviewed B20 sites from fitting and evaluation",
            "retain unreserved confirmed anchors as P",
            "exclude candidates within 12 px of canonical or reviewed coordinates from U",
            "assign remaining event candidates U and quiet candidates null-control",
        ],
        "prohibited_interpretations": [
            "quiet proposals as biological negatives",
            "unmatched event candidates as negatives",
            "candidate-selected yield as full-field precision",
            "review holdout as independent review replication",
            "cross-recording generalization",
        ],
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    return HarmonizedInnovationCensus(
        rows=grouped_rows,
        feature_columns=feature_names,
        feature_dictionary=feature_info,
        manifest=manifest,
        validation=validation,
    )
