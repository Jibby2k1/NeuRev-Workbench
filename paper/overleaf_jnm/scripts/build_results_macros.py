#!/usr/bin/env python3
"""Build compile-safe LaTeX result macros from NeuRev result artifacts.

This script supports the uploaded pre-repair repository layout so the manuscript
skeleton is immediately reproducible. For a protected rerun, prefer a generated
`manuscript_metrics.json` containing the same macro names and source metadata;
the script will consume that file when supplied through `--metrics-json`.

The output is written atomically. Missing values become visible LaTeX placeholders
unless `--strict` is used, in which case the command fails before replacing the
existing macro file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from neurobench.portable_paths import portable_path


@dataclass(frozen=True)
class Metric:
    macro: str
    value: Any
    source: str
    locator: str
    formatter: Callable[[Any], str]
    provisional: bool = True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def dotted(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(path)
        cur = cur[key]
    return cur


def fmt_int(value: Any) -> str:
    return str(int(value))


def fmt4(value: Any) -> str:
    return f"{float(value):.4f}"


def fmt3(value: Any) -> str:
    return f"{float(value):.3f}"


def fmt2(value: Any) -> str:
    return f"{float(value):.2f}"


def fmt_percent1(value: Any) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def parse_occurrence_table(path: Path) -> dict[str, int]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    coords = {(r["x"], r["y"]) for r in rows}
    canonical = {r["roi_id"] for r in rows}
    bursts = {r["burst_id"] for r in rows}
    observations = {r["observation_id"] for r in rows}
    if len(observations) != len(rows):
        raise ValueError("Duplicate observation_id in occurrence table")
    return {
        "occurrences": len(rows),
        "original_sites": len(coords),
        "canonical_neurons": len(canonical),
        "bursts": len(bursts),
    }


def regex_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise KeyError(pattern)
    return float(match.group(1))


def collect_from_current_repo(repo: Path) -> list[Metric]:
    base = repo / "Outputs" / "UnsupervisedICAEval"
    paths = {
        "stage_a": base / "two_frame_stage_a_v1" / "summary.json",
        "external": base / "two_frame_external_assay_v1" / "summary.json",
        "raw_sig": base / "raw_signature_null_generalization_v1" / "summary.json",
        "residual": base / "roi_minus_annulus_signature_v1" / "summary.json",
        "occurrences": base / "two_frame_external_assay_v1" / "tables" / "occurrence_metrics.csv",
        "feature_doc": repo / "docs" / "research" / "SPON_CA_BURST_SCIENTIFIC_FEATURE_AUDIT_V1_RESULTS.md",
        "spatial_morphology": repo / "Outputs" / "NeuronIdentifiability" / "spon_ca_burst_identifiability_paper_v1_v8" / "17_spatial_ica_objective_morphology_v2" / "summary.json",
        "spatial_filter_stability": repo / "Outputs" / "NeuronIdentifiability" / "spon_ca_burst_identifiability_paper_v1_v8" / "18_spatial_ica_filter_stability_v1" / "summary.json",
        "spatial_reconstruction_stability": repo / "Outputs" / "NeuronIdentifiability" / "spon_ca_burst_identifiability_paper_v1_v8" / "19_spatial_ica_reconstruction_stability_v2" / "summary.json",
        "spatial_class_certainty": repo / "Outputs" / "NeuronIdentifiability" / "spon_ca_burst_identifiability_paper_v1_v8" / "20_spatial_ica_class_certainty_alignment_v3" / "summary.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required current-layout sources:\n" + "\n".join(missing))

    stage_a = load_json(paths["stage_a"])
    external = load_json(paths["external"])
    raw_sig = load_json(paths["raw_sig"])
    residual = load_json(paths["residual"])
    spatial_morphology = load_json(paths["spatial_morphology"])
    spatial_reconstruction = load_json(paths["spatial_reconstruction_stability"])
    spatial_class_certainty = load_json(paths["spatial_class_certainty"])
    counts = parse_occurrence_table(paths["occurrences"])
    feature_text = paths["feature_doc"].read_text(encoding="utf-8")

    metrics: list[Metric] = []

    def add(macro: str, value: Any, key: str, locator: str, formatter: Callable[[Any], str], *, provisional: bool = True) -> None:
        metrics.append(Metric(macro, value, str(paths[key].relative_to(repo)), locator, formatter, provisional))

    add("NCurrentRecordings", 1, "external", "study scope", fmt_int)
    add("NCurrentBursts", counts["bursts"], "occurrences", "unique burst_id", fmt_int)
    add("NCurrentOccurrences", counts["occurrences"], "occurrences", "row count", fmt_int)
    add("NCurrentOriginalSites", counts["original_sites"], "occurrences", "unique (x,y)", fmt_int)
    add("NCurrentCanonicalNeurons", counts["canonical_neurons"], "occurrences", "unique roi_id", fmt_int)

    add("ICADiffRSquared", dotted(stage_a, "analytic_comparisons.difference_signed.r_squared"), "stage_a", "analytic_comparisons.difference_signed.r_squared", fmt4)
    add("ICADiffScoreCorrelation", dotted(external, "ica_vs_difference_event_score_correlation"), "external", "ica_vs_difference_event_score_correlation", fmt4)
    add("ICADiffShiftP", dotted(external, "ica.label_shift_p_upper"), "external", "ica.label_shift_p_upper", fmt3)

    add("RawHeldoutCorrelation", dotted(raw_sig, "held_out_event_correlation.mean"), "raw_sig", "held_out_event_correlation.mean", fmt4)
    add("RawFirstPCVariance", dotted(external, "candidate_signature.first_temporal_pca_variance_explained"), "external", "candidate_signature.first_temporal_pca_variance_explained", fmt_percent1)
    add("RawAnnulusDelta", dotted(raw_sig, "delta_vs_matched_annulus.mean"), "raw_sig", "delta_vs_matched_annulus.mean", fmt4)
    add("RawAnnulusCILow", dotted(raw_sig, "delta_vs_matched_annulus.ci95_low"), "raw_sig", "delta_vs_matched_annulus.ci95_low", fmt4)
    add("RawAnnulusCIHigh", dotted(raw_sig, "delta_vs_matched_annulus.ci95_high"), "raw_sig", "delta_vs_matched_annulus.ci95_high", fmt4)
    add("RawWithinSiteMedianCorrelation", dotted(raw_sig, "within_roi_pairwise_correlation.median"), "raw_sig", "within_roi_pairwise_correlation.median", fmt4)

    add("ResidualHeldoutCorrelation", dotted(residual, "held_out_event_correlation.mean"), "residual", "held_out_event_correlation.mean", fmt4)
    add("ResidualSpatialDelta", dotted(residual, "delta_vs_spatial_control.mean"), "residual", "delta_vs_spatial_control.mean", fmt4)
    add("ResidualSpatialCILow", dotted(residual, "delta_vs_spatial_control.ci95_low"), "residual", "delta_vs_spatial_control.ci95_low", fmt4)
    add("ResidualSpatialCIHigh", dotted(residual, "delta_vs_spatial_control.ci95_high"), "residual", "delta_vs_spatial_control.ci95_high", fmt4)
    add("ResidualPeakAmplitudeDelta", dotted(residual, "event_peak_amplitude_delta.mean"), "residual", "event_peak_amplitude_delta.mean", fmt2)
    add("ResidualPeakAmplitudeCILow", dotted(residual, "event_peak_amplitude_delta.ci95_low"), "residual", "event_peak_amplitude_delta.ci95_low", fmt2)
    add("ResidualPeakAmplitudeCIHigh", dotted(residual, "event_peak_amplitude_delta.ci95_high"), "residual", "event_peak_amplitude_delta.ci95_high", fmt2)
    add("ResidualWithinSiteMedianCorrelation", dotted(residual, "within_roi_pairwise_correlation.median"), "residual", "within_roi_pairwise_correlation.median", fmt4)

    add("CarrierRecallBtwenty", regex_float(feature_text, r"Frozen carrier\s*\|\s*([0-9.]+)"), "feature_doc", "Frozen carrier table, B20", fmt4)
    add("CoherenceRecallBtwenty", regex_float(feature_text, r"Local coherence\s*\|\s*\*\*([0-9.]+)"), "feature_doc", "Local coherence table, B20", fmt4)
    add("LagRecallBtwenty", regex_float(feature_text, r"budgets 20 and 58 \(`([0-9.]+)`"), "feature_doc", "standalone lagged lane, B20", fmt4)
    add("QuietVarianceSlope", regex_float(feature_text, r"\| Full \| ([0-9.]+) \|"), "feature_doc", "quiet variance table, full slope", fmt2)
    metrics.append(Metric("QuietVarianceWeightedRtwo", "0.955\\text{--}0.959", str(paths["feature_doc"].relative_to(repo)), "quiet variance table, weighted R2 range", str))
    metrics.append(Metric("ProvisionalFieldBoundaryX", "x=286", str(paths["feature_doc"].relative_to(repo)), "field-boundary paragraph", str))
    add("LeftRightGlobalCorrelation", regex_float(feature_text, r"only `([0-9.]+)`"), "feature_doc", "left/right global correlation", fmt3)

    morphology_tests = {
        (row["representation"], row["metric"]): row
        for row in spatial_morphology["tests"]
    }
    add("SpatialICAContrastDelta", morphology_tests[("Spatial ICA", "center_annulus_contrast")]["mean_observed_minus_control"], "spatial_morphology", "tests[Spatial ICA,center_annulus_contrast].mean_observed_minus_control", fmt3, provisional=False)
    add("SpatialICAContrastQ", morphology_tests[("Spatial ICA", "center_annulus_contrast")]["bh_q_across_12_tests"], "spatial_morphology", "tests[Spatial ICA,center_annulus_contrast].bh_q_across_12_tests", fmt4, provisional=False)
    add("SpatialICARadiusDelta", morphology_tests[("Spatial ICA", "effective_radius_px")]["mean_observed_minus_control"], "spatial_morphology", "tests[Spatial ICA,effective_radius_px].mean_observed_minus_control", fmt3, provisional=False)
    add("SpatialICARadiusQ", morphology_tests[("Spatial ICA", "effective_radius_px")]["bh_q_across_12_tests"], "spatial_morphology", "tests[Spatial ICA,effective_radius_px].bh_q_across_12_tests", fmt4, provisional=False)
    add("SpatialICAPeakOffsetDelta", morphology_tests[("Spatial ICA", "peak_offset_px")]["mean_observed_minus_control"], "spatial_morphology", "tests[Spatial ICA,peak_offset_px].mean_observed_minus_control", fmt3, provisional=False)
    add("SpatialICAPeakOffsetQ", morphology_tests[("Spatial ICA", "peak_offset_px")]["bh_q_across_12_tests"], "spatial_morphology", "tests[Spatial ICA,peak_offset_px].bh_q_across_12_tests", fmt4, provisional=False)
    add("SpatialICAReferenceRecall", spatial_reconstruction["reference_fixed_budget_mean_recall"], "spatial_reconstruction_stability", "reference_fixed_budget_mean_recall", fmt4, provisional=False)

    reconstruction_rows_path = paths["spatial_reconstruction_stability"].parent / "reconstruction_stability.tsv"
    with reconstruction_rows_path.open("r", encoding="utf-8", newline="") as f:
        reconstruction_rows = list(csv.DictReader(f, delimiter="\t"))
    add("SpatialICAMinPixelCorrelation", min(float(r["pixel_correlation"]) for r in reconstruction_rows), "spatial_reconstruction_stability", "reconstruction_stability.tsv:min(pixel_correlation)", fmt4, provisional=False)
    add("SpatialICAMinEventMapCorrelation", min(float(r["median_event_map_correlation"]) for r in reconstruction_rows), "spatial_reconstruction_stability", "reconstruction_stability.tsv:min(median_event_map_correlation)", fmt4, provisional=False)
    add("SpatialICAMinMorphologySpearman", min(float(r["median_morphology_spearman"]) for r in reconstruction_rows), "spatial_reconstruction_stability", "reconstruction_stability.tsv:min(median_morphology_spearman)", fmt4, provisional=False)

    v7 = spatial_class_certainty["canonical_v7_class_sensitivity"]
    boundary = next(row for row in v7["class_tests"] if row["representation"] == "Spatial ICA" and row["metric"] == "boundary_sharpness")
    certainty = spatial_class_certainty["canonical_v7_certainty"]["certainty_class_association"]
    add("SpatialICAVsevenBoundaryEpsilon", boundary["epsilon_squared"], "spatial_class_certainty", "canonical_v7_class_sensitivity.class_tests[Spatial ICA,boundary_sharpness].epsilon_squared", fmt3, provisional=False)
    add("SpatialICAVsevenBoundaryQ", boundary["bh_q_across_14_class_tests"], "spatial_class_certainty", "canonical_v7_class_sensitivity.class_tests[Spatial ICA,boundary_sharpness].bh_q_across_14_class_tests", fmt4, provisional=False)
    add("CertaintyDominantClassCramersV", certainty["cramers_v"], "spatial_class_certainty", "canonical_v7_certainty.certainty_class_association.cramers_v", fmt3, provisional=False)
    add("CertaintyDominantClassPermutationP", certainty["canonical_label_permutation_p"], "spatial_class_certainty", "canonical_v7_certainty.certainty_class_association.canonical_label_permutation_p", fmt3, provisional=False)
    return metrics


def collect_from_metrics_json(path: Path) -> list[Metric]:
    data = load_json(path)
    raw_metrics = data.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError("metrics JSON must contain an object named 'metrics'")
    out: list[Metric] = []
    for macro, record in raw_metrics.items():
        if not isinstance(record, dict) or "value" not in record:
            raise ValueError(f"Invalid metric record for {macro}")
        fmt_name = str(record.get("format", "str"))
        formatter = {
            "int": fmt_int,
            "2f": fmt2,
            "3f": fmt3,
            "4f": fmt4,
            "percent1": fmt_percent1,
            "str": str,
        }.get(fmt_name)
        if formatter is None:
            raise ValueError(f"Unknown format '{fmt_name}' for {macro}")
        out.append(Metric(
            macro=str(macro),
            value=record["value"],
            source=str(record.get("source", path.name)),
            locator=str(record.get("locator", "metrics." + str(macro))),
            formatter=formatter,
            provisional=bool(record.get("provisional", False)),
        ))
    return out


def render_macro(metric: Metric) -> str:
    rendered = metric.formatter(metric.value)
    if metric.macro in {"NCurrentRecordings", "NCurrentBursts", "NCurrentOccurrences", "NCurrentOriginalSites", "NCurrentCanonicalNeurons"}:
        body = rendered
    else:
        body = rf"\ensuremath{{{rendered}}}"
    if metric.provisional:
        body = rf"\Provisional{{{body}}}"
    return rf"\newcommand{{\{metric.macro}}}{{{body}}}"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, help="NeuRev repository root for current-layout extraction")
    parser.add_argument("--metrics-json", type=Path, help="Protected-run manuscript_metrics.json")
    parser.add_argument("--output", type=Path, default=Path("macros/results_macros.tex"))
    parser.add_argument("--source-map", type=Path, default=Path("result_source_map.json"))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--final", action="store_true", help="Reject provisional metrics")
    args = parser.parse_args()

    if bool(args.repo) == bool(args.metrics_json):
        parser.error("Provide exactly one of --repo or --metrics-json")

    errors: list[str] = []
    try:
        if args.metrics_json:
            metrics = collect_from_metrics_json(args.metrics_json.resolve())
            root = args.metrics_json.resolve().parent
        else:
            root = args.repo.resolve()
            metrics = collect_from_current_repo(root)
    except Exception as exc:  # precise error is emitted; output remains untouched
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    names = [m.macro for m in metrics]
    if len(names) != len(set(names)):
        errors.append("Duplicate macro names detected")
    if args.final and any(m.provisional for m in metrics):
        errors.append("--final requested but one or more metrics are provisional")

    if errors and (args.strict or args.final):
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        return 3

    source_map: dict[str, Any] = {
        "schema_version": 1,
        "mode": "protected" if args.metrics_json else "current_pre_repair",
        "root": portable_path(root, repository=args.repo.resolve() if args.repo else Path.cwd().resolve()),
        "macros": {},
        "warnings": errors,
    }
    lines = [
        "% Generated by scripts/build_results_macros.py; do not edit by hand.",
        rf"\newcommand{{\ResultsGeneratedOn}}{{{tex_escape(__import__('datetime').date.today().isoformat())}}}",
        rf"\newcommand{{\ResultsSourceStatus}}{{{'protected rerun' if args.metrics_json else 'pre-repair repository audit'}}}",
        "",
    ]
    for metric in sorted(metrics, key=lambda m: m.macro):
        lines.append(render_macro(metric))
        source_path = root / metric.source
        source_map["macros"][metric.macro] = {
            "value": metric.value,
            "rendered": metric.formatter(metric.value),
            "provisional": metric.provisional,
            "source": metric.source,
            "locator": metric.locator,
            "source_sha256": sha256(source_path) if source_path.exists() and source_path.is_file() else None,
        }

    atomic_write(args.output.resolve(), "\n".join(lines) + "\n")
    atomic_write(args.source_map.resolve(), json.dumps(source_map, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(metrics)} macros to {args.output}")
    print(f"Wrote source map to {args.source_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
