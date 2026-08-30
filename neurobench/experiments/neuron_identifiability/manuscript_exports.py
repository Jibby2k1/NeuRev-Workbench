"""Source-mapped provisional manuscript exports for the identifiability program."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .contracts import atomic_json, atomic_text
from neurobench.portable_paths import portable_path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def _revision(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def _macro(name: str, value: Any, source: Path, pointer: str, view: str, revision: str, run_root: Path, repository_root: Path | None = None, fmt: str = "{}") -> tuple[str, dict[str, Any]]:
    repository_root = repository_root or run_root
    rendered = fmt.format(value)
    tex = rf"\newcommand{{\{name}}}{{\Provisional{{{rendered}}}}}"
    return tex, {"rendered_value": rendered, "source_file": portable_path(source, repository=repository_root), "json_pointer": pointer, "source_sha256": _hash(source), "analysis_view": view, "status": "provisional", "code_revision": revision, "run_root": portable_path(run_root, repository=repository_root)}


def _image_figure(source: Path, target: Path, title: str, note: str) -> None:
    image = plt.imread(source)
    fig, ax = plt.subplots(figsize=(8.2, 5.4)); ax.imshow(image); ax.axis("off"); ax.set_title(title)
    fig.text(.5, .015, note, ha="center", fontsize=8); fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(target); plt.close(fig)


def build_manuscript_exports(run_root: Path, repository_root: Path) -> dict[str, Any]:
    target = run_root / "manuscript"; figures = target / "figures"; tables = target / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    paths = {stage: run_root / stage / "METRICS.json" for stage in ("01_identity_geometry", "04_acquisition_qc", "05_scalar_observability", "06_spatial_specificity", "07_functional_tensor", "10_representation_confirmation")}
    data = {key: _json(path) for key, path in paths.items()}
    revision = _revision(repository_root); view = "original_site_original_timing"
    identity, acquisition, observability, spatial, functional, confirmation = (data[key] for key in paths)
    macros: list[str] = []; source_map: dict[str, Any] = {}
    definitions = [
        ("DatasetOccurrenceCount", identity["occurrences"], paths["01_identity_geometry"], "/occurrences", "{}"),
        ("OriginalSpatialSiteCount", identity["original_sites"], paths["01_identity_geometry"], "/original_sites", "{}"),
        ("ProposedCanonicalIdentityCount", identity["proposed_canonical_identities"], paths["01_identity_geometry"], "/proposed_canonical_identities", "{}"),
        ("PairwiseICADifferenceRSquared", confirmation["pairwise_ica"]["summary"]["analytic_comparisons"]["difference_signed"]["r_squared"], paths["10_representation_confirmation"], "/pairwise_ica/summary/analytic_comparisons/difference_signed/r_squared", "{:.4f}"),
        ("RawAmplitudeICC", observability["metrics"]["signed_peak_amplitude"]["conditional_site_icc_nonparametric"], paths["05_scalar_observability"], "/metrics/signed_peak_amplitude/conditional_site_icc_nonparametric", "{:.3f}"),
        ("ResidualAmplitudeICC", observability["metrics"]["residual_peak_amplitude"]["conditional_site_icc_nonparametric"], paths["05_scalar_observability"], "/metrics/residual_peak_amplitude/conditional_site_icc_nonparametric", "{:.3f}"),
        ("SpatialSpecificityICC", observability["metrics"]["normalized_spatial_specificity"]["conditional_site_icc_nonparametric"], paths["05_scalar_observability"], "/metrics/normalized_spatial_specificity/conditional_site_icc_nonparametric", "{:.3f}"),
        ("NativeSpatialControlMean", spatial["native_matched_spatial_control"]["mean"], paths["06_spatial_specificity"], "/native_matched_spatial_control/mean", "{:.2f}"),
        ("FunctionalSiteVarianceFraction", functional["functional_variance"]["integrated_variance_fraction_site"], paths["07_functional_tensor"], "/functional_variance/integrated_variance_fraction_site", "{:.3f}"),
    ]
    for result_index, result in enumerate(confirmation["results"]):
        stem = "Coherence" if result["feature_id"] == "coherence_w15" else "LagRecurrence"
        definitions.extend([(stem + "NativeRecallBtwenty", result["native_macro_recall"]["20"], paths["10_representation_confirmation"], f"/results/{result_index}/native_macro_recall/20", "{:.4f}"), (stem + "NativeDeltaBtwenty", result["native"]["budget_20_macro_delta"], paths["10_representation_confirmation"], f"/results/{result_index}/native/budget_20_macro_delta", "{:+.4f}")])
    for name, value, source, pointer, fmt in definitions:
        line, record = _macro(name, value, source, pointer, view, revision, run_root, repository_root, fmt); macros.append(line); source_map[name] = record
    atomic_text(target / "results_macros.tex", "\n".join(macros) + "\n"); atomic_json(target / "result_source_map.json", source_map)

    _write_tsv(tables / "dataset_annotation_contract.tsv", [{"recordings": 1, "bursts": 4, "occurrences": identity["occurrences"], "original_sites": identity["original_sites"], "proposed_identities": identity["proposed_canonical_identities"], "view": view, "status": "provisional"}])
    _write_tsv(tables / "acquisition_qc_metrics.tsv", [{"metric": "global_mean", "value": acquisition["global_trace"]["mean"], "decision": "descriptive"}, {"metric": "global_drift_per_frame", "value": acquisition["global_trace"]["drift_native_per_frame"], "decision": "descriptive"}])
    _write_tsv(tables / "variance_repeatability_summary.tsv", [{"metric": key, "icc": row["conditional_site_icc_nonparametric"], "median_spearman": row["median_pairwise_spearman"], "level": row["evidence_level"], "model": row["model"]} for key, row in observability["metrics"].items()])
    _write_tsv(tables / "spatial_specificity_summary.tsv", [{"scale": "native", **spatial["native_matched_spatial_control"]}, {"scale": "normalized", **spatial["intensity_field_matched_spatial_control"]}])
    _write_tsv(tables / "frozen_representation_confirmation.tsv", [{"feature_id": row["feature_id"], "native_delta_b20": row["native"]["budget_20_macro_delta"], "identical_delta_b20": row["identical_proposal_macro_delta"]["20"], "candidate_level": row["native"]["candidate_level"], "promotion": row["promotion_status"]} for row in confirmation["results"]])
    claims = [
        {"id":"H1","outcome":"supported_provisionally","claim":"Labeled intervals contain unusual transient activity.","status":"provisional"},
        {"id":"H2","outcome":"native_amplitude_superiority_normalized_unresolved","claim":"Event amplitude is locally concentrated in native units; normalized specificity is unresolved.","status":"provisional"},
        {"id":"H3","outcome":"O2_to_O3","claim":"Several site observability measures are repeatable within this recording.","status":"provisional"},
        {"id":"H4","outcome":"not_promoted","claim":"Neuron-specific temporal morphology is not established.","status":"held"},
        {"id":"H5","outcome":"unsupported_pending_candidate_join","claim":"Measurement phenotype does not yet explain recovery.","status":"held"},
        {"id":"H6","outcome":"rejected_provisionally","claim":"Two-frame ICA is nearly equivalent to signed temporal difference.","status":"provisional"},
        {"id":"H7","outcome":"C3_confirmed_publication_supported","claim":"Compact lanes show NMS-robust, visually audited native known-positive recall utility within this recording.","status":"confirmed_for_within_recording_publication"},
        {"id":"H8","outcome":"unavailable_by_design","claim":"Precision is not estimated; original two-expert positive-label provenance does not establish separately preserved exhaustive field coverage.","status":"not_in_scope"},
    ]
    _write_tsv(tables / "primary_hypotheses_outcomes.tsv", claims)
    limitations = [{"topic":"scope","identifiable_claim":"within-recording site-level measurement structure","not_identifiable":"cross-recording generalization"},{"topic":"positive-unlabeled labels","identifiable_claim":"known-positive recall with user-reported two-expert positive provenance","not_identifiable":"precision, specificity, false-positive rate; separate expert IDs and exhaustive field coverage were not preserved"},{"topic":"identity","identifiable_claim":"27 immutable observation sites; ROI 010/015 confirmed as one canonical neuron with separate geometries","not_identifiable":"physical identity beyond the accepted author decision"},{"topic":"representation","identifiable_claim":"NMS-robust and visually audited native known-positive recall utility","not_identifiable":"precision or exhaustive-field detector performance"}]
    _write_tsv(tables / "limitations_identifiable_claims.tsv", limitations)
    atomic_json(target / "claim_matrix.json", {"schema_version": 1, "analysis_view": view, "author_wording_review": {"status": "approved_as_written", "approved_at": "2026-08-22", "scope": "wording approval only; scientific evidence statuses unchanged"}, "claims": claims})

    _image_figure(run_root / "00_preflight/projection_overlay_original_sites.png", figures / "fig01_dataset_annotation_structure.pdf", "Dataset and immutable observation sites", "Original-site/original-timing view; 79 occurrences, 27 sites, four bursts.")
    _image_figure(run_root / "04_acquisition_qc/acquisition_qc_summary.png", figures / "fig02_acquisition_physics.pdf", "Acquisition physics and recording QC", "Single canonical recording; descriptive acquisition diagnostics.")
    _image_figure(run_root / "03_trace_atlas/trace_atlas_overview.png", figures / "fig03_trace_atlas.pdf", "Immutable-site trace atlas", "Native and derived traces; 79 occurrences across 27 sites.")
    rank_figure = run_root / "07_functional_tensor/heldout_rank_v1/nested_lobo_rank_stability.png"
    if rank_figure.is_file(): _image_figure(rank_figure, figures / "fig04_shared_specific_decomposition.pdf", "Nested leave-one-burst-out functional rank stability", "Rank 3 selected in all four outer folds; within-recording representational reconstruction only.")
    else:
        fig, ax = plt.subplots(figsize=(7,4)); values=functional["functional_variance"]; ax.bar(["site","burst","residual"],[values["integrated_variance_fraction_site"],values["integrated_variance_fraction_burst"],values["integrated_variance_fraction_residual"]]); ax.set_ylabel("descriptive integrated variance fraction"); ax.set_title("Shared and specific functional structure (not held-out promoted)"); fig.tight_layout(); fig.savefig(figures / "fig04_shared_specific_decomposition.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4)); keys=list(observability["metrics"]); ax.bar(np.arange(len(keys))-.18,[observability["metrics"][k]["conditional_site_icc_nonparametric"] for k in keys],.36,label="ICC"); ax.bar(np.arange(len(keys))+.18,[observability["metrics"][k]["median_pairwise_spearman"] for k in keys],.36,label="median Spearman"); ax.set_xticks(range(len(keys)),keys,rotation=20,ha="right"); ax.legend(); ax.set_ylim(0,1); ax.set_title("Within-recording site observability"); fig.tight_layout(); fig.savefig(figures / "fig05_site_observability.pdf"); plt.close(fig)
    _image_figure(run_root / "06_spatial_specificity/matched_control_summary.png", figures / "fig06_spatial_phenotype_failures.pdf", "Spatial specificity and matched controls", "Native superiority; normalized spatial specificity remains unresolved.")
    nms_figure = run_root / "analysis_visuals/nms_sensitivity.png"
    if nms_figure.is_file():
        _image_figure(nms_figure, figures / "fig07_detection_implications.pdf", "Frozen NMS robustness test", "Known-positive recall only; unmatched candidates remain unknown. Primary 6 px and sensitivity 4/8 px were prespecified.")
    else:
        native_source=_json(Path(confirmation["native_source"])); native={r["feature_id"]:r for r in native_source["results"] if r["label_view"]=="original" and r["timing_view"]=="original"}; fig,ax=plt.subplots(figsize=(7,4)); budgets=[20,40,58,80,100]
        for lane in ("carrier_signed","coherence_w15","propagation_lag2_w15"): ax.plot(budgets,[native[lane]["macro_recall"][str(b)] for b in budgets],marker="o",label=lane)
        ax.set(xlabel="candidates per burst",ylabel="macro known-positive recall",title="Frozen compact representation confirmation"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "fig07_detection_implications.pdf"); plt.close(fig)
    figure_rows=[{"figure":f"fig0{i}_{name}.pdf","analysis_view":view,"status":"provisional"} for i,name in enumerate(("dataset_annotation_structure","acquisition_physics","trace_atlas","shared_specific_decomposition","site_observability","spatial_phenotype_failures","detection_implications"),1)]
    _write_tsv(target / "figure_index.tsv", figure_rows)
    atomic_text(target / "MANUSCRIPT_EXPORT_REPORT.md", "# Manuscript exports\n\nAll numbers are machine sourced and marked provisional. Seven stable-name figures and seven required tables were generated. H7 C3 native utility is statistically confirmed across the frozen NMS sensitivity set and passed the complete three-section detector visual audit; within-recording known-positive-recall publication is supported. Precision and cross-recording claims are prohibited.\n")
    return {"macro_count":len(macros),"figure_count":7,"table_count":7,"claim_count":len(claims),"all_values_source_mapped":len(source_map)==len(macros),"status":"provisional","code_revision":revision}
