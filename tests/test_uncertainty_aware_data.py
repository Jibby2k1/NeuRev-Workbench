from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import median

import pytest

from neurobench.experiments.neuron_identifiability.uncertainty_aware_data import (
    B20_V5_CENSUS,
    CANONICAL_V7_REQUIRED_COLUMNS,
    FEATURE_DEFINITIONS,
    HarmonizationInputs,
    INNOVATION_CENSUS_METADATA_COLUMNS,
    InnovationCensusContract,
    NEW_REVIEW_COLUMNS_V1,
    OCCURRENCE_COLUMNS_V1,
    ProposalCensusContract,
    RAW_FEATURE_COLUMNS,
    SITE_COLUMNS_V1,
    TAXONOMY_FEATURE_NAMES,
    feature_dictionary,
    harmonize_feature_census,
    harmonize_innovation_candidate_census,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("NEUROBENCH_DATA_ROOT", ROOT)).resolve()
V5_ROOT = ROOT / "Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8/detection_profile_taxonomy_v5"
V7_LABELS = DATA_ROOT / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v7/adjudication_final.tsv"
NEW_REVIEW = ROOT / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/user_review_v1.tsv"
BROAD_CENSUS_ROOT = Path(
    os.environ.get("NEUROBENCH_UAL_CENSUS_ROOT", "/tmp/neurev-ual-census-integration-v1")
).resolve()


def _write_tsv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _occurrence(
    occurrence_id: str,
    site_id: str,
    burst: int,
    x_px: float,
    y_px: float,
    *,
    raw_peak: float,
    residual_peak: object = 4.0,
) -> dict[str, object]:
    values: dict[str, object] = {
        "detection_occurrence_id": occurrence_id,
        "burst_id": burst,
        "x_px": x_px,
        "y_px": y_px,
        "start_ui": 10,
        "end_ui": 20,
        "peak_frame_ui": 15,
        "lane_agreement": 1.0,
        "mean_rank_fraction": 0.25,
        "coherence_rank": 1,
        "propagation_rank": 2,
        "coherence_score": 3.0,
        "propagation_score": 2.0,
        "quiet_intensity": 100.0,
        "raw_peak": raw_peak,
        "residual_peak": residual_peak,
        "robust_snr": 3.0,
        "event_area": 25.0,
        "time_to_peak_frames": 5,
        "time_to_peak_fraction": 0.5,
        "spatial_specificity": 0.4,
        "annulus_correlation": 0.5,
        "nearest_known_positive_distance_px": 20.0,
        "known_positive_within_6px": "false",
        "nearest_known_positive_id": "",
        "detection_site_id": site_id,
        "recurrence_fraction": 0.25,
        "class_id": 1,
        "class_name": "Class 1",
    }
    for column in OCCURRENCE_COLUMNS_V1:
        if column.startswith("z_"):
            values[column] = 999.0
    return values


def _site_rows(occurrences: list[dict[str, object]]) -> list[dict[str, object]]:
    by_site: dict[str, list[dict[str, object]]] = {}
    for row in occurrences:
        by_site.setdefault(str(row["detection_site_id"]), []).append(row)
    output = []
    for site_id, rows in sorted(by_site.items()):
        output.append(
            {
                "detection_site_id": site_id,
                "occurrences": len(rows),
                "bursts": len({int(row["burst_id"]) for row in rows}),
                "x_px": median(float(row["x_px"]) for row in rows),
                "y_px": median(float(row["y_px"]) for row in rows),
                "dominant_class": 1,
                "class_consistency": 1.0,
                "known_positive_occurrences": 0,
                "median_raw_peak": median(float(row["raw_peak"]) for row in rows),
                "median_spatial_specificity": median(float(row["spatial_specificity"]) for row in rows),
            }
        )
    return output


def _canonical(
    observation_id: str,
    burst: int,
    canonical_id: str,
    x_px: float,
    y_px: float,
    *,
    confirmed: bool = True,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "burst_id": burst,
        "original_roi_id": observation_id.split("__", 1)[-1],
        "canonical_roi_id": canonical_id,
        "x_px": x_px,
        "y_px": y_px,
        "neuron_confidence": "confirmed" if confirmed else "uncertain",
        "morphology": "localized_center",
        "context": "isolated",
        "disposition": "confirmed_neuron" if confirmed else "activity_visible_identity_uncertain",
        "include_confirmed": str(confirmed).lower(),
        "include_inclusive": "true",
        "review_status": "adjudicated",
        "reviewer_id": "Yinong",
        "source_note": "synthetic test label",
    }


def _review(blind_id: str, site_id: str, label: str) -> dict[str, object]:
    return {
        "blind_id": blind_id,
        "detection_site_id": site_id,
        "normalized_label": label,
        "confidence_1_to_5": 3,
        "size_flag": "",
        "morphology": "",
        "low_snr": "false",
        "artifact_proximity": "false",
        "overlap_or_adjacent_source": "",
        "verbatim_feedback": "synthetic review",
        "normalization_note": "synthetic normalization",
    }


def _innovation_row(
    partition: str,
    partition_id: int,
    proposal_order: int,
    x_px: int,
    y_px: int,
    *,
    feature_names: tuple[str, ...] = ("signal", "context"),
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": (
            f"irv5_{partition}_p{partition_id:02d}_x{x_px:05d}_y{y_px:05d}"
        ),
        "partition": partition,
        "partition_id": partition_id,
        "proposal_order": proposal_order,
        "burst_id": partition_id if partition == "event" else "",
        "quiet_map_id": partition_id if partition == "quiet" else "",
        "x_px": x_px,
        "y_px": y_px,
        "source_count": 2,
    }
    row.update(
        {
            f"feature__{name}": float(partition_id * 10 + proposal_order + index)
            for index, name in enumerate(feature_names)
        }
    )
    return row


def _broad_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], InnovationCensusContract]:
    candidates = [
        _innovation_row("event", 1, 1, 10, 10),
        _innovation_row("event", 1, 2, 12, 10),
        _innovation_row("event", 1, 3, 40, 10),
        _innovation_row("event", 2, 1, 20, 10),
        _innovation_row("event", 3, 1, 80, 10),
        _innovation_row("event", 4, 1, 120, 10),
        *[
            _innovation_row("quiet", partition_id, 1, 200 + partition_id * 20, 50)
            for partition_id in range(1, 5)
        ],
    ]
    canonical = [
        _canonical("b01__roi_001", 1, "roi_shared", 10, 10),
        _canonical("b02__roi_001", 2, "roi_shared", 20, 10),
        _canonical("b03__roi_001", 3, "roi_uncertain", 80, 10, confirmed=False),
    ]
    contract = InnovationCensusContract(
        census_id="synthetic_innovation_union",
        expected_event_rows=6,
        expected_quiet_rows=4,
        expected_event_partition_counts=(3, 1, 1, 1),
        expected_quiet_partition_counts=(1, 1, 1, 1),
        expected_feature_names=("signal", "context"),
        expected_reviewed_sites=None,
    )
    return candidates, canonical, contract


def _representative_fold_fixture(
) -> tuple[list[dict[str, object]], list[dict[str, object]], InnovationCensusContract]:
    candidates = [
        _innovation_row("event", 1, 1, 10, 10),
        _innovation_row("event", 1, 2, 44, 10),
        _innovation_row("event", 1, 3, 84, 10),
        _innovation_row("event", 1, 4, 124, 10),
        _innovation_row("event", 1, 5, 164, 10),
        _innovation_row("event", 2, 1, 52, 10),
        _innovation_row("event", 2, 2, 72, 10),
        _innovation_row("event", 3, 1, 104, 10),
        _innovation_row("event", 4, 1, 144, 10),
        *[
            _innovation_row("quiet", partition_id, 1, 220 + partition_id * 20, 50)
            for partition_id in range(1, 5)
        ],
    ]
    canonical = [
        _canonical("b01__roi_shared", 1, "roi_shared", 10, 10),
        _canonical("b02__roi_shared", 2, "roi_shared", 52, 10),
        _canonical("b01__roi_044", 1, "roi_044", 44, 10),
        _canonical("b01__roi_084", 1, "roi_084", 84, 10),
        _canonical("b01__roi_124", 1, "roi_124", 124, 10),
        _canonical("b01__roi_164", 1, "roi_164", 164, 10),
    ]
    contract = InnovationCensusContract(
        census_id="synthetic_representative_x_folds",
        expected_event_rows=9,
        expected_quiet_rows=4,
        expected_event_partition_counts=(5, 2, 1, 1),
        expected_quiet_partition_counts=(1, 1, 1, 1),
        expected_feature_names=("signal", "context"),
        expected_reviewed_sites=None,
    )
    return candidates, canonical, contract


def _fixture(
    tmp_path: Path,
    *,
    budget: int = 58,
) -> tuple[HarmonizationInputs, ProposalCensusContract]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    occurrences = [
        _occurrence("b01__det_001", "dsite_001", 1, 10, 10, raw_peak=1),
        _occurrence("b02__det_001", "dsite_002", 2, 18, 10, raw_peak=2),
        _occurrence("b03__det_001", "dsite_003", 3, 50, 10, raw_peak=3, residual_peak=""),
        _occurrence("b04__det_001", "dsite_004", 4, 90, 10, raw_peak=4),
    ]
    sites = _site_rows(occurrences)
    canonical = [
        _canonical("b01__roi_001", 1, "roi_shared", 10, 10),
        _canonical("b02__roi_001", 2, "roi_shared", 18, 10),
    ]
    reviews = [
        _review("NC001", "dsite_002", "uncertain"),
        _review("NC002", "dsite_003", "probable_neuron"),
        _review("NC003", "dsite_004", "artifact_or_noise"),
    ]

    occurrence_path = tmp_path / "occurrences.tsv"
    site_path = tmp_path / "sites.tsv"
    canonical_path = tmp_path / "canonical.tsv"
    review_path = tmp_path / "review.tsv"
    summary_path = tmp_path / "summary.json"
    validation_path = tmp_path / "validation.json"
    _write_tsv(occurrence_path, OCCURRENCE_COLUMNS_V1, occurrences)
    _write_tsv(site_path, SITE_COLUMNS_V1, sites)
    _write_tsv(canonical_path, CANONICAL_V7_REQUIRED_COLUMNS, canonical)
    _write_tsv(review_path, NEW_REVIEW_COLUMNS_V1, reviews)
    summary_path.write_text(
        json.dumps(
            {
                "features": list(TAXONOMY_FEATURE_NAMES),
                "fitting_labels_used": False,
                "detection_occurrences": len(occurrences),
                "detection_sites": len(sites),
            }
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(
            {
                "budget_per_burst_per_lane": budget,
                "strict_nms_radius_px": 6.0,
                "labels_excluded_from_fit": True,
            }
        ),
        encoding="utf-8",
    )
    inputs = HarmonizationInputs(
        occurrence_profiles=occurrence_path,
        site_profiles=site_path,
        taxonomy_summary=summary_path,
        taxonomy_validation=validation_path,
        canonical_v7_labels=canonical_path,
        new_review_labels=review_path,
    )
    contract = ProposalCensusContract(census_id=f"synthetic_b{budget}", fixed_candidates_per_burst=budget)
    return inputs, contract


def test_generic_b58_harmonization_preserves_raw_features_and_source_boundaries(tmp_path: Path) -> None:
    inputs, contract = _fixture(tmp_path)
    first = harmonize_feature_census(inputs, contract, repository_root=ROOT)
    second = harmonize_feature_census(inputs, contract, repository_root=ROOT)

    assert first.manifest == second.manifest
    assert first.manifest["contract"]["fixed_candidates_per_burst"] == 58
    assert first.feature_columns == RAW_FEATURE_COLUMNS
    assert first.validation["canonical_v7_unmatched_occurrences_remain_unknown"] == 2

    site = {row["detection_site_id"]: row for row in first.site_rows}
    assert site["dsite_001"]["identity_group_id"] == site["dsite_002"]["identity_group_id"]
    assert site["dsite_001"]["spatial_group_id"] == site["dsite_002"]["spatial_group_id"]
    assert site["dsite_001"]["leakage_group_id"] == site["dsite_002"]["leakage_group_id"]
    assert site["dsite_002"]["cross_source_disagreement"] is True
    assert site["dsite_003"]["features"]["residual_peak"] is None

    model_rows = first.model_feature_rows("site")
    assert model_rows[2]["residual_peak"] is None
    assert model_rows[2]["missing__residual_peak"] is True
    assert not any(key.startswith("z_") for row in model_rows for key in row)
    assert not any(key in {"x_px", "y_px", "normalized_label", "class_id"} for row in model_rows for key in row)


def test_label_policies_keep_uncertain_unlabeled_and_artifact_as_case_study(tmp_path: Path) -> None:
    inputs, contract = _fixture(tmp_path)
    census = harmonize_feature_census(inputs, contract, repository_root=ROOT)

    strict = {row["detection_site_id"]: row for row in census.label_rows("strict")}
    inclusive = {row["detection_site_id"]: row for row in census.label_rows("inclusive")}
    assert strict["dsite_001"]["state"] == "positive"
    assert strict["dsite_002"]["state"] == "unlabeled"
    assert strict["dsite_002"]["known_negative"] is False
    assert strict["dsite_003"]["state"] == "unlabeled"
    assert inclusive["dsite_003"]["state"] == "positive"
    assert strict["dsite_004"]["state"] == "excluded"
    assert strict["dsite_004"]["known_negative"] is False


def test_readiness_fails_closed_when_grouped_positive_unlabeled_support_is_too_small(tmp_path: Path) -> None:
    inputs, contract = _fixture(tmp_path)
    census = harmonize_feature_census(inputs, contract, repository_root=ROOT)
    assert census.validation["model_readiness"]["strict"]["passed"] is False
    with pytest.raises(ValueError, match="not ready"):
        census.assert_model_ready("strict")


def test_rollup_and_proposal_contract_tampering_are_rejected(tmp_path: Path) -> None:
    inputs, contract = _fixture(tmp_path)
    columns, rows = _load_tsv(inputs.site_profiles)
    rows[0]["occurrences"] = "99"
    _write_tsv(inputs.site_profiles, columns, rows)
    with pytest.raises(ValueError, match="site rollup mismatch"):
        harmonize_feature_census(inputs, contract, repository_root=ROOT)

    inputs, contract = _fixture(tmp_path / "second")
    validation = json.loads(inputs.taxonomy_validation.read_text(encoding="utf-8"))
    validation["budget_per_burst_per_lane"] = 20
    inputs.taxonomy_validation.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(ValueError, match="proposal contract"):
        harmonize_feature_census(inputs, contract, repository_root=ROOT)


def _load_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return tuple(reader.fieldnames or ()), list(reader)


def test_feature_dictionary_is_stable_and_prohibits_full_census_transforms() -> None:
    dictionary = feature_dictionary()
    assert tuple(item["name"] for item in dictionary["features"]) == RAW_FEATURE_COLUMNS
    assert tuple(item.taxonomy_name for item in FEATURE_DEFINITIONS) == TAXONOMY_FEATURE_NAMES
    assert "z_*" in dictionary["prohibited_model_columns"]["patterns"]
    assert dictionary["taxonomy_precomputed_z_columns"]["allowed_for_grouped_learning"] is False
    assert len(dictionary["feature_dictionary_sha256"]) == 64


def test_broad_census_preserves_pu_null_and_positive_radius_semantics() -> None:
    candidates, canonical, contract = _broad_fixture()
    census = harmonize_innovation_candidate_census(
        candidates,
        canonical,
        contract,
        repository_root=ROOT,
    )
    repeated = harmonize_innovation_candidate_census(
        candidates,
        canonical,
        contract,
        repository_root=ROOT,
    )

    assert census.manifest == repeated.manifest
    assert census.feature_columns == ("signal", "context")
    assert census.validation["partition_counts"] == {
        "event": {"1": 3, "2": 1, "3": 1, "4": 1},
        "quiet": {"1": 1, "2": 1, "3": 1, "4": 1},
    }
    states = {row["candidate_id"]: row["training_state"] for row in census.rows}
    assert states["irv5_event_p01_x00010_y00010"] == "positive"
    assert states["irv5_event_p01_x00012_y00010"] == "excluded_canonical_confirmed_radius"
    assert states["irv5_event_p01_x00040_y00010"] == "unlabeled"
    assert states["irv5_event_p02_x00020_y00010"] == "positive"
    assert states["irv5_event_p03_x00080_y00010"] == "excluded_canonical_nonconfirmed_radius"
    assert states["irv5_event_p04_x00120_y00010"] == "unlabeled"
    assert all(
        state == "null_control"
        for candidate_id, state in states.items()
        if candidate_id.startswith("irv5_quiet")
    )

    model_rows = census.model_feature_rows(require_ready=False)
    null_rows = census.null_control_feature_rows()
    assert len(model_rows) == 4
    assert len(null_rows) == 4
    assert all(row["training_state"] in {"positive", "unlabeled"} for row in model_rows)
    assert all(row["training_state"] == "null_control" for row in null_rows)
    assert all(row["known_negative"] is False for row in census.label_rows())
    assert not any(
        name in census.feature_columns
        for name in (*INNOVATION_CENSUS_METADATA_COLUMNS, "proposal_order", "source_count")
    )


def test_broad_fold_metadata_keeps_identity_groups_and_never_assigns_boundaries() -> None:
    candidates, canonical, contract = _broad_fixture()
    census = harmonize_innovation_candidate_census(
        candidates,
        canonical,
        contract,
        repository_root=ROOT,
    )
    fold_rows = census.fold_assignment_rows()
    by_id = {row["candidate_id"]: row for row in fold_rows}
    left = by_id["irv5_event_p01_x00010_y00010"]
    recurrence = by_id["irv5_event_p02_x00020_y00010"]

    assert left["leakage_group_id"] == recurrence["leakage_group_id"]
    assert left["primary_anchor_canonical_roi_id"] == "roi_shared"
    assert recurrence["primary_anchor_canonical_roi_id"] == "roi_shared"
    assert all("fold_id" not in row for row in fold_rows)
    assert [row["x_px"] for row in fold_rows] == sorted(row["x_px"] for row in fold_rows)
    assert census.outer_fold_contract == {
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
    assert census.validation["group_diagnostics"]["radius_chain_components_used"] is False
    assignments = {
        row["candidate_id"]: (1 if index == 0 else 2)
        for index, row in enumerate(
            item for item in fold_rows if item["eligible_for_primary_fit"]
        )
    }
    with pytest.raises(ValueError, match="splits leakage groups"):
        census.validate_outer_fold_assignments(assignments)


def test_representative_x_validator_allows_and_reports_whole_group_row_extensions() -> None:
    candidates, canonical, contract = _representative_fold_fixture()
    census = harmonize_innovation_candidate_census(
        candidates,
        canonical,
        contract,
        repository_root=ROOT,
    )
    rows = {row["candidate_id"]: row for row in census.fold_assignment_rows()}
    shared_ids = {
        "irv5_event_p01_x00010_y00010",
        "irv5_event_p02_x00052_y00010",
    }
    assert {rows[candidate_id]["primary_group_representative_x_px"] for candidate_id in shared_ids} == {
        31.0
    }
    assert {
        rows[candidate_id]["primary_group_representative_basis"] for candidate_id in shared_ids
    } == {"unreserved_positive_anchor_rows"}
    assert {rows[candidate_id]["primary_group_minimum_x_px"] for candidate_id in shared_ids} == {
        10.0
    }
    assert {rows[candidate_id]["primary_group_maximum_x_px"] for candidate_id in shared_ids} == {
        52.0
    }

    assignments = {
        "irv5_event_p01_x00010_y00010": 1,
        "irv5_event_p02_x00052_y00010": 1,
        "irv5_event_p01_x00044_y00010": 2,
        "irv5_event_p02_x00072_y00010": 2,
        "irv5_event_p01_x00084_y00010": 3,
        "irv5_event_p03_x00104_y00010": 3,
        "irv5_event_p01_x00124_y00010": 4,
        "irv5_event_p04_x00144_y00010": 4,
        "irv5_event_p01_x00164_y00010": 5,
    }
    validation = census.validate_outer_fold_assignments(assignments)

    assert validation["passed"] is True
    assert validation["contiguity_unit"] == (
        "positive_anchor_median_x_for_P_groups_else_all_eligible_median_x"
    )
    assert validation["representative_midpoint_boundaries_x_px"] == [37.5, 78.0, 114.0, 154.0]
    assert validation["raw_candidate_extent_overlap_is_failure"] is False
    assert validation["raw_candidate_extent_overlaps"] == [
        {
            "left_fold_id": 1,
            "right_fold_id": 2,
            "left_maximum_candidate_x_px": 52.0,
            "right_minimum_candidate_x_px": 44.0,
        }
    ]
    extension = validation["whole_group_closure_extension"]
    assert extension["candidate_count"] == 1
    assert extension["leakage_group_count"] == 1
    assert extension["maximum_boundary_extension_px"] == 14.5
    assert extension["rows"][0]["candidate_id"] == "irv5_event_p02_x00052_y00010"


def test_broad_review_neighborhoods_are_reserved_before_primary_labels(tmp_path: Path) -> None:
    review_inputs, _ = _fixture(tmp_path)
    candidates = [
        _innovation_row("event", 1, 1, 10, 10),
        _innovation_row("event", 2, 1, 18, 10),
        _innovation_row("event", 3, 1, 50, 10),
        _innovation_row("event", 4, 1, 90, 10),
        *[
            _innovation_row("quiet", partition_id, 1, 200 + partition_id * 20, 50)
            for partition_id in range(1, 5)
        ],
    ]
    contract = InnovationCensusContract(
        census_id="synthetic_review_reserve",
        expected_event_rows=4,
        expected_quiet_rows=4,
        expected_event_partition_counts=(1, 1, 1, 1),
        expected_quiet_partition_counts=(1, 1, 1, 1),
        expected_feature_names=("signal", "context"),
        expected_reviewed_sites=3,
    )
    census = harmonize_innovation_candidate_census(
        candidates,
        review_inputs.canonical_v7_labels,
        contract,
        repository_root=ROOT,
        data_root=tmp_path,
        review_site_source=review_inputs.site_profiles,
        review_occurrence_source=review_inputs.occurrence_profiles,
        new_review_source=review_inputs.new_review_labels,
    )

    event = {row["candidate_id"]: row for row in census.rows if row["partition"] == "event"}
    assert event["irv5_event_p01_x00010_y00010"]["training_state"] == "positive"
    assert event["irv5_event_p01_x00010_y00010"]["review_holdout"] == {
        "reserved": False,
        "excluded_from_unlabeled": True,
        "nearest_review_site_id": "dsite_002",
        "nearest_review_distance_px": 8.0,
        "anchor_status": "not_anchor",
        "review_label": "uncertain",
        "role": "primary_unlabeled_exclusion_buffer",
    }
    assert event["irv5_event_p02_x00018_y00010"]["training_state"] == "heldout_review_anchor"
    assert sum(row["review_holdout"]["reserved"] for row in event.values()) == 3
    assert census.validation["review_holdout"]["all_reviewed_neighborhoods_excluded_from_fitting"] is True
    assert census.manifest["source_files"]["review_holdout_sources"]["site_profiles"]["uri"].startswith(
        "data://"
    )


def test_broad_census_rejects_identity_schema_count_and_precomputed_feature_drift() -> None:
    candidates, canonical, contract = _broad_fixture()
    candidates[0]["candidate_id"] = "not_coordinate_stable"
    with pytest.raises(ValueError, match="coordinate-stable"):
        harmonize_innovation_candidate_census(candidates, canonical, contract, repository_root=ROOT)

    candidates, canonical, contract = _broad_fixture()
    candidates[1]["proposal_order"] = 4
    with pytest.raises(ValueError, match="proposal_order"):
        harmonize_innovation_candidate_census(candidates, canonical, contract, repository_root=ROOT)

    candidates, canonical, contract = _broad_fixture()
    count_contract = InnovationCensusContract(
        **{
            **contract.to_manifest(),
            "expected_event_partition_counts": (2, 2, 1, 1),
        }
    )
    with pytest.raises(ValueError, match="partition counts"):
        harmonize_innovation_candidate_census(candidates, canonical, count_contract, repository_root=ROOT)

    feature_names = ("signal_scaled",)
    transformed = [
        _innovation_row(
            str(row["partition"]),
            int(row["partition_id"]),
            int(row["proposal_order"]),
            int(row["x_px"]),
            int(row["y_px"]),
            feature_names=feature_names,
        )
        for row in candidates
    ]
    leakage_contract = InnovationCensusContract(
        census_id="synthetic_precomputed_leakage",
        expected_event_rows=6,
        expected_quiet_rows=4,
        expected_event_partition_counts=(3, 1, 1, 1),
        expected_quiet_partition_counts=(1, 1, 1, 1),
        expected_feature_names=feature_names,
        expected_reviewed_sites=None,
    )
    with pytest.raises(ValueError, match="precomputed full-census"):
        harmonize_innovation_candidate_census(
            transformed,
            canonical,
            leakage_contract,
            repository_root=ROOT,
        )


@pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            BROAD_CENSUS_ROOT / "event_candidates.tsv",
            BROAD_CENSUS_ROOT / "quiet_candidates.tsv",
            V7_LABELS,
            V5_ROOT / "detection_occurrence_profiles.tsv",
            NEW_REVIEW,
        )
    ),
    reason="exact broad-census and label authorities are not configured",
)
def test_exact_broad_census_live_join_pins_primary_readiness_boundary() -> None:
    candidate_rows = [
        *(_load_tsv(BROAD_CENSUS_ROOT / "event_candidates.tsv")[1]),
        *(_load_tsv(BROAD_CENSUS_ROOT / "quiet_candidates.tsv")[1]),
    ]
    census = harmonize_innovation_candidate_census(
        candidate_rows,
        V7_LABELS,
        repository_root=ROOT,
        data_root=DATA_ROOT,
        review_site_source=V5_ROOT / "detection_site_profiles.tsv",
        review_occurrence_source=V5_ROOT / "detection_occurrence_profiles.tsv",
        new_review_source=NEW_REVIEW,
    )

    assert census.validation["partition_counts"] == {
        "event": {"1": 544, "2": 530, "3": 476, "4": 449},
        "quiet": {"1": 683, "2": 748, "3": 759, "4": 748},
    }
    assert census.validation["feature_count"] == 34
    assert census.validation["confirmed_canonical_labels"] == 106
    assert census.validation["confirmed_positive_anchors"] == 105
    assert census.validation["confirmed_labels_without_anchor"] == 1
    assert census.validation["confirmed_anchors_reserved_for_review"] == 27
    assert census.validation["primary_positive_anchors_after_review_reserve"] == 78
    assert census.validation["primary_positive_anchors_after_review_reserve"] >= 75
    assert census.validation["review_holdout"] == {
        "enabled": True,
        "reviewed_sites": 18,
        "review_occurrences": 29,
        "reserved_event_candidates": 132,
        "buffered_event_candidates": 235,
        "buffer_only_event_candidates": 103,
        "review_anchors": 29,
        "all_reviewed_neighborhoods_excluded_from_fitting": True,
        "canonical_v7_overlap": {
            "reviewed_sites": 18,
            "reviewed_sites_with_any_canonical_v7_match": 18,
            "reviewed_sites_with_confirmed_canonical_v7_match": 10,
            "reviewed_sites_with_cross_source_disagreement": 2,
            "interpretation": (
                "label sources overlap and are not independent replicates; "
                "review reservations precede the primary label join"
            ),
        },
    }
    assert census.validation["model_readiness"] == {
        "passed": True,
        "row_counts_by_state": {
            "excluded_canonical_confirmed_radius": 138,
            "excluded_canonical_nonconfirmed_radius": 62,
            "excluded_review_radius": 48,
            "heldout_review_anchor": 29,
            "heldout_review_neighborhood": 103,
            "null_control": 2938,
            "positive": 78,
            "unlabeled": 1541,
        },
        "positive_leakage_groups": 21,
        "unlabeled_leakage_groups": 537,
        "requirements": {
            "outer_folds": 5,
            "minimum_positive_groups": 10,
            "minimum_unlabeled_groups": 10,
        },
        "failures": [],
        "scope_if_passed": "within-recording generalization to held-out identity/spatial groups only",
    }
    assert census.validation["group_diagnostics"]["radius_chain_components_used"] is False
    assert census.validation["group_diagnostics"]["spatial_cell"]["maximum_group_size"] == 15
    assert census.validation["group_diagnostics"]["spatial_cell"]["maximum_x_span_px"] == 11.0
    assert census.validation["group_diagnostics"]["leakage"]["maximum_group_size"] == 54
    assert census.validation["group_diagnostics"]["leakage"]["maximum_x_span_px"] == 42.0
    assert census.manifest["source_files"]["candidate_census"]["sha256"] == (
        "50176cdaedc16b30f4c9f6e561a02a247fbc30b586d4fbd57b28bc7a64a1fb16"
    )
    assert census.manifest["rows_sha256"] == (
        "820a8d6626e9008b6ed5d055293ac33b57444ab463e8e82228cb9fc10d133359"
    )


@pytest.mark.skipif(
    not all(path.is_file() for path in (V5_ROOT / "detection_occurrence_profiles.tsv", V7_LABELS, NEW_REVIEW)),
    reason="exact local v5/v7/new-review authorities are not configured",
)
def test_exact_v5_snapshot_reconciles_and_is_correctly_blocked_for_grouped_learning() -> None:
    inputs = HarmonizationInputs(
        occurrence_profiles=V5_ROOT / "detection_occurrence_profiles.tsv",
        site_profiles=V5_ROOT / "detection_site_profiles.tsv",
        taxonomy_summary=V5_ROOT / "summary.json",
        taxonomy_validation=V5_ROOT / "validation.json",
        canonical_v7_labels=V7_LABELS,
        new_review_labels=NEW_REVIEW,
    )
    census = harmonize_feature_census(
        inputs,
        B20_V5_CENSUS,
        repository_root=ROOT,
        data_root=DATA_ROOT,
    )

    assert len(census.occurrence_rows) == 86
    assert len(census.site_rows) == 36
    assert census.validation["table_integrity"] == "passed"
    assert census.validation["label_source_overlap"] == {
        "reviewed_sites": 18,
        "reviewed_sites_with_any_canonical_v7_match": 18,
        "reviewed_sites_with_confirmed_canonical_v7_match": 10,
        "reviewed_sites_with_cross_source_disagreement": 2,
        "interpretation": "label sources overlap and are not independent replicates; neither source is silently overwritten",
    }
    assert census.validation["model_readiness"]["strict"]["site_counts"] == {
        "excluded": 1,
        "positive": 27,
        "unlabeled": 8,
    }
    assert census.validation["model_readiness"]["inclusive"]["site_counts"] == {
        "excluded": 1,
        "positive": 31,
        "unlabeled": 4,
    }
    assert all(not result["passed"] for result in census.validation["model_readiness"].values())
    assert census.manifest["source_files"]["occurrence_profiles"]["sha256"] == (
        "9b01e2286ffbb203d97afbda04191d0e797390611008f8d6941259738c7b54ab"
    )
    assert census.manifest["source_files"]["site_profiles"]["sha256"] == (
        "4db074a06afb9b5610bd47a99256dd841e033c4a78b340defe44bc1f98d67a93"
    )
    assert census.manifest["manifest_sha256"] == (
        "bc6fb5bc998582cd8509e9ec33d39472119386219c673e92693c12fd8dd89659"
    )
