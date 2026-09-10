#!/usr/bin/env python3
"""Generate evidence-bound facts shared by both venue manuscripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
OUTPUT = PAPER / "macros" / "venue_results.tex"
MANIFEST = PAPER / "macros" / "venue_results_manifest.json"

SOURCES = {
    "automated_validation": PAPER / "generated_analysis/automated_feature_validation_v1/summary.json",
    "identity_offsets": PAPER / "generated_analysis/identity_aware_feature_inspection_v8/offset_stability.tsv",
    "identity_summary": PAPER / "generated_analysis/identity_aware_feature_inspection_v8/summary.json",
    "identity_contamination": REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_validation_suite_v2/identity_contamination.tsv",
    "candidate_review": REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/review_summary_v1.json",
    "movie_benchmark": REPO / "Outputs/NeuronIdentifiability/realistic_movie_detector_benchmark_v3/summary.json",
    "family_holdout": REPO / "Outputs/NeuronIdentifiability/realistic_movie_generator_family_holdout_v4/summary.json",
    "challenge_suite": REPO / "Outputs/NeuronIdentifiability/automated_challenge_suite_v7/summary.json",
    "long_delay_audit": PAPER / "figures/final/fig06_ica_model_internals.json",
    "local_standardization_audit": PAPER / "figures/final/fig07_ls_denominator_diagnostics.json",
    "trace_taxonomy": PAPER / "figures/final/fig04_trace_taxonomy_experiment.json",
    "two_frame_operator": REPO / "Outputs/UnsupervisedICAEval/two_frame_stage_a_v1/summary.json",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latex_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _latex_scientific(value: float, digits_after_decimal: int = 2) -> str:
    coefficient, exponent = f"{value:.{digits_after_decimal}e}".split("e")
    return f"{coefficient}\\times10^{{{int(exponent)}}}"


def _plain_scientific(value: float, digits_after_decimal: int = 2) -> str:
    """Return compact text-mode scientific notation for small table cells."""
    coefficient, exponent = f"{value:.{digits_after_decimal}e}".split("e")
    return f"{coefficient}e{int(exponent)}"


def _build() -> tuple[str, dict]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing venue fact source(s):\n" + "\n".join(missing))

    automated = _json(SOURCES["automated_validation"])
    identity = _json(SOURCES["identity_summary"])
    offsets = _tsv(SOURCES["identity_offsets"])
    contamination = _tsv(SOURCES["identity_contamination"])
    review = _json(SOURCES["candidate_review"])
    movie = _json(SOURCES["movie_benchmark"])
    holdout = _json(SOURCES["family_holdout"])
    challenge = _json(SOURCES["challenge_suite"])
    long_delay = _json(SOURCES["long_delay_audit"])
    local_standardization = _json(SOURCES["local_standardization_audit"])
    trace_taxonomy = _json(SOURCES["trace_taxonomy"])
    two_frame = _json(SOURCES["two_frame_operator"])

    population = automated["population"]
    recovery = automated["recovery_join"]
    canonical = next(
        row
        for row in automated["headline"]["recovery_models"]
        if row["population"] == "102_canonical_collapsed_sensitivity"
    )
    raw = automated["headline"]["best_temporal_feature"]
    null_rows = {
        row["feature_id"]: row for row in automated["headline"]["null_calibration"]
    }

    unique_status = {
        row["observation_id"]: row["identity_status"]
        for row in offsets
        if row["v8_outcome"] == "identity_collision"
    }
    same_collision = sum(
        value == "same_canonical_identity_collision" for value in unique_status.values()
    )
    other_collision = sum(value == "other_identity_collision" for value in unique_status.values())
    collision_r2 = sorted(
        float(row["multineighbor_r2"])
        for row in contamination
        if row["outcome"] == "collision"
    )
    clear_r2 = sorted(
        float(row["multineighbor_r2"])
        for row in contamination
        if row["outcome"] == "clear"
    )
    median = lambda values: (values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2)

    strict_recovered = int(recovery["recovered_any"])
    strict_total = int(population["occurrences"])
    canonical_recovered = int(canonical["recovered"])
    canonical_total = int(recovery["canonical_sensitivity_occurrences"])
    counts = review["counts"]

    facts: list[tuple[str, str]] = [
        ("VenueFactSchemaVersion", "1"),
        ("VenueFactSourceCount", str(len(SOURCES))),
        ("VenueCohortBursts", str(population["bursts"])),
        ("VenueCohortOccurrences", str(strict_total)),
        ("VenueCohortSites", str(population["sites"])),
        ("VenuePerLaneBudget", "58"),
        ("VenueCounterfactualBudget", "100"),
        ("VenueStrictRecovered", str(strict_recovered)),
        ("VenueStrictMissed", str(int(recovery["missed_any"]))),
        ("VenueStrictSensitivity", f"{100 * strict_recovered / strict_total:.1f}\\%"),
        ("VenueCanonicalRecovered", str(canonical_recovered)),
        ("VenueCanonicalOccurrences", str(canonical_total)),
        ("VenueCanonicalSensitivity", f"{100 * canonical_recovered / canonical_total:.1f}\\%"),
        ("VenueReviewedCandidates", str(sum(counts.values()))),
        ("VenueReviewDefinite", str(counts["definite_neuron"])),
        ("VenueReviewProbable", str(counts["probable_neuron"])),
        ("VenueReviewUncertain", str(counts["uncertain"])),
        ("VenueReviewArtifact", str(counts["artifact_or_noise"])),
        ("VenueReviewLikely", str(review["likely_neuron_count"])),
        ("VenueReviewPriorityAUC", _latex_number(review["priority_score_auc_likely_vs_unresolved_or_unlikely"])),
        ("VenueIdentityCollision", str(identity["population"]["identity_collision_misses"])),
        ("VenueIdentityClear", str(identity["population"]["identity_clear_misses"])),
        ("VenueSameIdentityCollision", str(same_collision)),
        ("VenueOtherIdentityCollision", str(other_collision)),
        ("VenueCollisionDirectionCosine", _latex_number(identity["headline"]["offset_direction_cosine_median_collision"])),
        ("VenueClearDirectionCosine", _latex_number(identity["headline"]["offset_direction_cosine_median_clear"])),
        ("VenueCollisionNeighborRtwo", _latex_number(median(collision_r2))),
        ("VenueClearNeighborRtwo", _latex_number(median(clear_r2))),
        ("VenueRawAUC", _latex_number(raw["mean_frame_auc"])),
        ("VenueRawAUCLow", _latex_number(raw["frame_auc_ci95_low"])),
        ("VenueRawAUCHigh", _latex_number(raw["frame_auc_ci95_high"])),
        ("VenueCarrierAUC", _latex_number(null_rows["carrier_signed"]["observed_site_mean_auc"])),
        ("VenueCoherenceAUC", _latex_number(null_rows["coherence_w15"]["observed_site_mean_auc"])),
        ("VenueLagAUC", _latex_number(null_rows["propagation_lag2_w15"]["observed_site_mean_auc"])),
        ("VenueLongDelayInversionRMS", _latex_scientific(long_delay["embedding_reconstruction_rms_max"])),
        ("VenueLSRMSE", _latex_scientific(local_standardization["ls_center_rmse"])),
        ("VenueLSMaxError", _latex_scientific(local_standardization["ls_center_max_abs_error"])),
        ("VenueLongDelayInversionRMSText", _plain_scientific(long_delay["embedding_reconstruction_rms_max"])),
        ("VenueLSRMSEText", _plain_scientific(local_standardization["ls_center_rmse"])),
        ("VenueLSMaxErrorText", _plain_scientific(local_standardization["ls_center_max_abs_error"])),
        ("VenueAuditFrames", str(local_standardization["frame_count"])),
        ("VenueTraceClassOneSites", str(trace_taxonomy["joint_class_sizes"]["1"])),
        ("VenueTraceClassTwoSites", str(trace_taxonomy["joint_class_sizes"]["2"])),
        (
            "VenueTwoFrameSampleCorrelation",
            f"{two_frame['analytic_comparisons']['difference_signed']['pearson']:.4f}",
        ),
        (
            "VenueTwoFrameSampleRtwo",
            f"{two_frame['analytic_comparisons']['difference_signed']['r_squared']:.4f}",
        ),
        ("VenueMovieBenchmarkCount", str(movie["design"]["movies"])),
        ("VenueMovieCombinedFone", _latex_number(movie["operating_points"]["combined"]["4"]["f1"])),
        ("VenueMovieSpatialFone", _latex_number(movie["operating_points"]["spatial_context"]["4"]["f1"])),
        ("VenueFamilyHoldoutCount", str(holdout["design"]["movies"])),
        ("VenueFamilySpatialFone", _latex_number(sum(holdout["family_operating_points"][family]["spatial_context"]["f1"] for family in holdout["design"]["families"]) / len(holdout["design"]["families"]))),
        ("VenueFamilyCombinedFone", _latex_number(sum(holdout["family_operating_points"][family]["combined"]["f1"] for family in holdout["design"]["families"]) / len(holdout["design"]["families"]))),
        ("VenueCrescentAUC", _latex_number(challenge["summaries"]["morphology_interventions"]["crescent"])),
    ]

    if same_collision + other_collision != int(identity["population"]["identity_collision_misses"]):
        raise ValueError("identity collision subtype counts do not reconcile")
    if strict_recovered + int(recovery["missed_any"]) != strict_total:
        raise ValueError("strict recovery counts do not reconcile")
    if not (long_delay["site_count"] == local_standardization["site_count"] == population["sites"]):
        raise ValueError("site counts do not reconcile")
    if sum(int(value) for value in trace_taxonomy["joint_class_sizes"].values()) != population["sites"]:
        raise ValueError("trace taxonomy counts do not reconcile with immutable sites")

    lines = [
        "% Generated by scripts/build_venue_result_facts.py; do not edit by hand.",
        "% Feature Atlas is registered descriptively but excluded pending media audit and claim promotion.",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in facts)
    text = "\n".join(lines) + "\n"
    manifest = {
        "schema_version": 2,
        "purpose": "evidence-bound repeated facts for both venue manuscripts",
        "excluded": {
            "feature_atlas_v1": "registered descriptively; model-annotation media audit and claim promotion incomplete"
        },
        "sources": [
            {
                "key": key,
                "path": str(path.relative_to(REPO)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "availability": (
                    "local_metadata_only"
                    if path.relative_to(REPO).parts[0] == "Outputs"
                    else "repository"
                ),
            }
            for key, path in SOURCES.items()
        ],
        "facts": {name: value for name, value in facts},
    }
    return text, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, manifest = _build()
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.check:
        stale = []
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            stale.append(str(OUTPUT))
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != manifest_text:
            stale.append(str(MANIFEST))
        if stale:
            raise SystemExit("stale venue fact output(s):\n" + "\n".join(stale))
        print(f"venue result facts current: {len(manifest['facts'])} facts, {len(manifest['sources'])} sources")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    MANIFEST.write_text(manifest_text, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
