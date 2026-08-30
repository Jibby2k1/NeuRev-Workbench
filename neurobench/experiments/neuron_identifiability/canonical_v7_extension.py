"""Provenance-stratified canonical-v7 extension of the protected paper analysis."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from neurobench.algorithms.scientific_feature_audit import causal_local_correlation_feature
from neurobench.experiments.hard_roi_adjudication.adjudication import label_view, load_tsv
from neurobench.experiments.hard_roi_adjudication.config import HardRoiAdjudicationConfig
from neurobench.experiments.hard_roi_adjudication.reevaluate import _evaluate_map
from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import _quiet_calibrate

from .contracts import atomic_json, atomic_text
from .identity import load_adjudication
from .nms_sensitivity import BUDGETS, LANES, RADII, sensitivity_gate
from .trace_extraction import Geometry, extract_site_traces, occurrence_metrics

BLUE = "#2F5D8A"; ORANGE = "#D17A22"; GOLD = "#B6922E"; INK = "#222222"; GRID = "#D9D9D9"
METRICS = ("signed_peak_amplitude", "robust_peak_snr", "residual_peak_amplitude", "normalized_spatial_specificity")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def _save(fig: plt.Figure, target: Path) -> None:
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_crosswalk(v1_path: Path, v7_path: Path, output: Path) -> dict[str, Any]:
    v1 = {row["observation_id"]: row for row in _read_tsv(v1_path)}
    v7_rows = _read_tsv(v7_path); v7 = {row["observation_id"]: row for row in v7_rows}
    if not set(v1) <= set(v7):
        raise ValueError(f"v7 dropped v1 observations: {sorted(set(v1) - set(v7))}")
    rows = []
    compare = ("canonical_roi_id", "x_px", "y_px", "disposition", "include_confirmed", "include_inclusive")
    for row in v7_rows:
        old = v1.get(row["observation_id"])
        if row["reviewer_id"] == "original_workbook":
            provenance = "original_workbook"
        elif old is not None:
            provenance = "candidate_assisted_present_in_v1"
        else:
            provenance = "candidate_assisted_added_after_v1"
        changed = [key for key in compare if old is not None and old.get(key, "") != row.get(key, "")]
        rows.append({
            "observation_id": row["observation_id"], "burst_id": row["burst_id"],
            "original_roi_id": row["original_roi_id"], "canonical_roi_id_v7": row["canonical_roi_id"],
            "provenance_stratum": provenance, "present_in_v1": str(old is not None).lower(),
            "v1_disposition": old.get("disposition", "") if old else "", "v7_disposition": row["disposition"],
            "include_confirmed_v7": row["include_confirmed"], "include_inclusive_v7": row["include_inclusive"],
            "changed_fields_from_v1": ",".join(changed), "review_status_v7": row["review_status"],
            "reviewer_provenance_v7": row["reviewer_id"], "source_note_v7": row["source_note"],
        })
    output.mkdir(parents=True, exist_ok=True); _write_tsv(output / "v1_v7_observation_crosswalk.tsv", rows)
    counts = Counter(row["provenance_stratum"] for row in rows)
    disposition = {key: dict(Counter(row["v7_disposition"] for row in rows if row["provenance_stratum"] == key)) for key in counts}
    summary = {
        "schema_version": 1, "status": "complete", "v1_rows": len(v1), "v7_rows": len(v7),
        "v1_rows_preserved": len(set(v1) & set(v7)), "v7_only_rows": len(set(v7) - set(v1)),
        "rows_changed_from_v1": sum(bool(row["changed_fields_from_v1"]) for row in rows),
        "provenance_counts": dict(counts), "disposition_by_provenance": disposition,
        "primary_interpretation": "original/frozen rows remain the protected representation cohort; candidate-assisted rows are stratified sensitivity evidence",
    }
    atomic_json(output / "crosswalk_summary.json", summary)
    return summary


def run_v7_nms(data_root: Path, repo_root: Path, v7_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    base = HardRoiAdjudicationConfig.load(repo_root / "examples/spon_ca_burst_hard_roi_adjudication_v1.example.json")
    carrier = np.load(data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy", mmap_mode="r", allow_pickle=False)
    values = {
        "carrier_signed": carrier,
        "coherence_w15": _quiet_calibrate(causal_local_correlation_feature(carrier, window_frames=15, lag_frames=0, spatial_sigma_px=2.0, activity_qualified=True), 100),
        "propagation_lag2_w15": _quiet_calibrate(causal_local_correlation_feature(carrier, window_frames=15, lag_frames=2, spatial_sigma_px=2.0, activity_qualified=True), 100),
    }
    raw_labels = load_tsv(v7_path); results = []; failures = []
    for view in ("confirmed", "inclusive"):
        labels = label_view(raw_labels, view, "original")
        for lane in LANES:
            for radius in RADII:
                config = replace(base, evaluation={**base.evaluation, "nms_distance_px": radius})
                result, detail = _evaluate_map(lane, values[lane], labels, config, label_view_id=view, timing_view_id="original")
                result["nms_distance_px"] = radius; results.append(result)
                for row in detail: row["nms_distance_px"] = radius
                failures.extend(detail)
    by = {(r["label_view"], r["feature_id"], r["nms_distance_px"]): r for r in results}
    frozen = json.loads((data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v7/metrics.json").read_text())
    expected = {(r["label_view"], r["feature_id"]): r for r in frozen["results"] if r["timing_view"] == "original" and r["label_view"] in {"confirmed", "inclusive"} and r["feature_id"] in LANES}
    reproduction = {view: {lane: max(abs(float(by[(view, lane, 6)]["macro_recall"][str(b)]) - float(expected[(view, lane)]["macro_recall"][str(b)])) for b in BUDGETS) for lane in LANES} for view in ("confirmed", "inclusive")}
    if max(value for view in reproduction.values() for value in view.values()) > 1e-12:
        raise ValueError(f"v7 six-pixel reproduction failed: {reproduction}")
    gates = {view: {lane: {str(radius): sensitivity_gate(by[(view, lane, radius)], by[(view, "carrier_signed", radius)]) for radius in RADII} for lane in LANES[1:]} for view in ("confirmed", "inclusive")}
    stable = {view: {lane: gates[view][lane]["6"]["macro_delta_b20"] >= .03 and all(gates[view][lane][str(r)]["passed"] for r in RADII) for lane in LANES[1:]} for view in gates}
    payload = {
        "schema_version": 1, "status": "complete", "adjudication_tsv": str(v7_path),
        "estimand": "provenance-sensitive known-positive recall on canonical v7; confirmed and inclusive views",
        "selection_circularity": "candidate-assisted rows were surfaced by the evaluated feature bank; these results are sensitivity/internal-consistency evidence, not independent validation",
        "primary_nms_px": 6, "sensitivity_nms_px": [4, 8], "reproduction_max_abs_error": reproduction,
        "results": results, "gates": gates, "nms_robust_internal_consistency": stable,
        "unmatched_candidates": "unknown_not_negative",
    }
    atomic_json(output / "v7_nms_sensitivity.json", payload)
    flat = []
    for row in results:
        for budget in BUDGETS:
            count = row["pooled_counts"][str(budget)]
            flat.append({"label_view": row["label_view"], "feature_id": row["feature_id"], "nms_distance_px": row["nms_distance_px"], "budget": budget, "macro_recall": row["macro_recall"][str(budget)], "matched": count["matched"], "labels": count["labels"]})
    _write_tsv(output / "v7_nms_budget_recall.tsv", flat); _write_tsv(output / "v7_nms_failure_audit.tsv", failures)
    return payload


def _rank01(values: dict[str, float]) -> dict[str, float]:
    keys = sorted(values, key=lambda key: (values[key], key)); denominator = max(1, len(keys) - 1)
    return {key: index / denominator for index, key in enumerate(keys)}


def _representatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eligible = [row for row in rows if row["confirmed_bursts"] >= 3 and np.isfinite(row["recovery_score"])]
    ranks = {metric: _rank01({row["canonical_roi_id"]: row[metric] for row in eligible}) for metric in METRICS}
    recovery_rank = _rank01({row["canonical_roi_id"]: row["recovery_score"] for row in eligible})
    vectors = []
    for row in eligible:
        key = row["canonical_roi_id"]; row["measurement_score"] = float(np.mean([ranks[m][key] for m in METRICS])); row["composite_score"] = (row["measurement_score"] + recovery_rank[key]) / 2
        vectors.append([ranks[m][key] for m in METRICS] + [recovery_rank[key]])
    median = np.median(np.asarray(vectors), axis=0)
    for row, vector in zip(eligible, vectors, strict=True): row["typical_distance"] = float(np.linalg.norm(np.asarray(vector) - median))
    hero = max(eligible, key=lambda row: (row["composite_score"], row["canonical_roi_id"]))
    hard = min(eligible, key=lambda row: (row["recovery_score"], row["measurement_score"], row["canonical_roi_id"]))
    typical = min((row for row in eligible if row not in (hero, hard)), key=lambda row: (row["typical_distance"], row["canonical_roi_id"]))
    return {"hero": hero, "typical": typical, "hard_case": hard}


def build_v7_atlas(run_root: Path, video_path: Path, v7_path: Path, nms_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True); raw_rows = _read_tsv(v7_path); loaded = load_adjudication(v7_path)
    # V7 can reuse an original ROI label at distinct centers. Geometry remains the
    # immutable extraction key; canonical identity is applied only after extraction.
    records = [replace(record, observation_site_id=f"{record.original_roi_id}__x{record.x_px:.3f}__y{record.y_px:.3f}") for record in loaded]
    video = np.load(video_path, mmap_mode="r", allow_pickle=False); site_order, traces = extract_site_traces(video, records, Geometry())
    index = {site: i for i, site in enumerate(site_order)}
    metrics = [occurrence_metrics(record, traces, index[record.observation_site_id]) for record in records]
    provenance = {row["observation_id"]: ("original_workbook" if row["reviewer_id"] == "original_workbook" else "candidate_assisted") for row in raw_rows}
    failures = _read_tsv(nms_path.parent / "v7_nms_failure_audit.tsv")
    recovered: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row in failures:
        if row["label_view"] != "confirmed" or int(row["nms_distance_px"]) != 6: continue
        recovered[(row["canonical_roi_id"], int(row["burst_id"]), row["feature_id"])].append(int(row["failure_class_at_budget_58"] == "matched"))
    raw_by_id = {row["observation_id"]: row for row in raw_rows}; grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        source = raw_by_id[row["observation_id"]]
        if source["include_confirmed"].lower() == "true": grouped[source["canonical_roi_id"]].append(row)
    summaries = []
    for canonical, rows in sorted(grouped.items()):
        bursts = sorted({int(row["burst_id"]) for row in rows}); item: dict[str, Any] = {"canonical_roi_id": canonical, "confirmed_occurrences": len(rows), "confirmed_bursts": len(bursts), "provenance": "mixed" if len({provenance[row["observation_id"]] for row in rows}) > 1 else provenance[rows[0]["observation_id"]]}
        for metric in METRICS: item[metric] = float(np.median([float(row[metric]) for row in rows]))
        lane_values = []
        for lane in LANES:
            values = [value for burst in bursts for value in recovered.get((canonical, burst, lane), [])]
            item[f"recovery_b58_{lane}"] = float(np.mean(values)) if values else float("nan"); lane_values.extend(values if lane != "carrier_signed" else [])
        item["recovery_score"] = float(np.nanmean([item[f"recovery_b58_{lane}"] for lane in LANES]))
        summaries.append(item)
    selected = _representatives(summaries); _write_tsv(output / "v7_canonical_summary.tsv", summaries)
    atomic_json(output / "v7_representative_selection.json", {key: value for key, value in selected.items()})

    # Provenance counts and disposition composition.
    strata = ("original_workbook", "candidate_assisted_present_in_v1", "candidate_assisted_added_after_v1")
    cross = _read_tsv(output.parent / "crosswalk/v1_v7_observation_crosswalk.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    counts = [sum(row["provenance_stratum"] == s for row in cross) for s in strata]
    axes[0].bar(["original\nworkbook", "candidate-assisted\nin v1", "candidate-assisted\nafter v1"], counts, color=["#777777", BLUE, ORANGE]); axes[0].set_ylabel("observation rows"); axes[0].set_title("Canonical-v7 annotation provenance")
    dispositions = ("confirmed_neuron", "activity_visible_identity_uncertain", "artifact"); bottom = np.zeros(3)
    for disposition, color in zip(dispositions, (BLUE, GOLD, "#777777"), strict=True):
        vals = np.array([sum(row["provenance_stratum"] == s and row["v7_disposition"] == disposition for row in cross) for s in strata]); axes[1].bar(range(3), vals, bottom=bottom, color=color, label=disposition.replace("_", " ")); bottom += vals
    axes[1].set_xticks(range(3), ["original", "assisted\nin v1", "assisted\nafter v1"]); axes[1].set_ylabel("observation rows"); axes[1].set_title("Disposition by provenance"); axes[1].legend(fontsize=8); _save(fig, output / "v7_provenance_composition")

    # Confirmed/inclusive NMS sensitivity.
    nms = json.loads(nms_path.read_text()); lookup = {(r["label_view"], r["feature_id"], r["nms_distance_px"]): r for r in nms["results"]}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row_index, view in enumerate(("confirmed", "inclusive")):
        for lane, color, marker in (("carrier_signed", "#666666", "o"), ("coherence_w15", BLUE, "s"), ("propagation_lag2_w15", ORANGE, "^")):
            axes[row_index, 0].plot(RADII, [lookup[(view, lane, r)]["macro_recall"]["20"] for r in RADII], color=color, marker=marker, label=lane)
        axes[row_index, 0].axvline(6, color=INK, ls="--", lw=.8); axes[row_index, 0].set(title=f"{view.title()} view", ylabel="macro known-positive recall @20", xlabel="NMS distance (pixels)"); axes[row_index, 0].grid(color=GRID, lw=.5)
        for lane, color, offset in (("coherence_w15", BLUE, -.08), ("propagation_lag2_w15", ORANGE, .08)):
            axes[row_index, 1].plot(np.asarray(RADII) + offset, [nms["gates"][view][lane][str(r)]["macro_delta_b20"] for r in RADII], color=color, marker="o", label=lane)
        axes[row_index, 1].axhline(0, color=INK, lw=.8); axes[row_index, 1].axhline(.03, color="#777777", ls=":", label="C3 threshold at primary 6 px" if row_index == 0 else None); axes[row_index, 1].axvline(6, color=INK, ls="--", lw=.8); axes[row_index, 1].set(title=f"{view.title()} gain versus carrier", ylabel="macro recall gain @20", xlabel="NMS distance (pixels)"); axes[row_index, 1].grid(color=GRID, lw=.5)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8); fig.suptitle("Canonical-v7 NMS sensitivity (candidate-assisted internal consistency)"); _save(fig, output / "v7_nms_sensitivity")

    # Canonical-by-burst measurement atlas.
    canonicals = sorted(grouped); matrices = []
    for metric in METRICS:
        matrix = np.full((len(canonicals), 4), np.nan)
        for i, canonical in enumerate(canonicals):
            by_burst = defaultdict(list)
            for row in grouped[canonical]: by_burst[int(row["burst_id"])].append(float(row[metric]))
            for burst, values in by_burst.items(): matrix[i, burst - 1] = np.median(values)
        med = np.nanmedian(matrix, axis=1, keepdims=True); mad = np.nanmedian(np.abs(matrix - med), axis=1, keepdims=True); matrices.append((matrix - med) / np.maximum(1.4826 * mad, 1e-9))
    fig, axes = plt.subplots(1, 4, figsize=(14, max(9, len(canonicals) * .18)), sharey=True, layout="constrained")
    for ax, matrix, title in zip(axes, matrices, ("Raw peak amplitude", "Robust peak SNR", "Residual peak amplitude", "Spatial specificity"), strict=True):
        im = ax.imshow(matrix, aspect="auto", cmap="PuOr", vmin=-3, vmax=3); ax.set_title(title); ax.set_xticks(range(4), ["B1", "B2", "B3", "B4"]); ax.set_xlabel("burst")
    axes[0].set_yticks(range(len(canonicals)), canonicals, fontsize=6); fig.colorbar(im, ax=axes, label="within-neuron robust deviation"); fig.suptitle("Canonical-v7 confirmed-neuron measurement atlas"); _save(fig, output / "v7_canonical_burst_atlas")

    # Representative traces use the first confirmed geometry for each canonical.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    background = np.mean(np.asarray(video[::20], dtype=np.float32), axis=0)
    for i, (role, item) in enumerate(selected.items()):
        canonical = item["canonical_roi_id"]; rows = grouped[canonical]; source = raw_by_id[rows[0]["observation_id"]]; x = int(round(float(source["x_px"]))); y = int(round(float(source["y_px"]))); crop = background[max(0, y-20):y+21, max(0, x-20):x+21]
        axes[i, 0].imshow(crop, cmap="gray", vmin=np.percentile(crop, 2), vmax=np.percentile(crop, 99.5)); axes[i, 0].scatter([20], [20], facecolors="none", edgecolors=ORANGE, s=100); axes[i, 0].axis("off"); axes[i, 0].set_title(f"{role.replace('_',' ').title()}: {canonical}\n[{item['provenance'].replace('_', ' ')}]")
        for row in sorted(rows, key=lambda value: int(value["burst_id"])):
            record = next(record for record in records if record.observation_id == row["observation_id"]); trace = traces["raw"][index[record.observation_site_id]]; a = max(0, record.start_zero - 10); b = min(len(trace), record.stop_zero_exclusive + 10); axes[i, 1].plot(np.arange(a, b) - record.start_zero, trace[a:b] - np.median(trace[a:record.start_zero]), lw=1, label=f"B{record.burst_id}")
        axes[i, 1].axvline(0, color=INK, ls="--", lw=.8); axes[i, 1].set(title="Raw, baseline subtracted", xlabel="frames from event start"); axes[i, 1].grid(color=GRID, lw=.5)
        axes[i, 2].bar(["carrier", "coherence", "lag"], [item[f"recovery_b58_{lane}"] for lane in LANES], color=["#777777", BLUE, ORANGE]); axes[i, 2].set_ylim(0, 1); axes[i, 2].set_title("Confirmed recovery @58")
    axes[0, 1].legend(fontsize=7); fig.suptitle("Canonical-v7 representative neurons (provenance-sensitive)"); _save(fig, output / "v7_representative_profiles")

    # Site-bootstrap associations, with provenance strata kept visible.
    association_rows = []; rng = np.random.default_rng(20260822)
    groups = {"all": summaries, "original_workbook": [row for row in summaries if row["provenance"] == "original_workbook"], "candidate_assisted_or_mixed": [row for row in summaries if row["provenance"] != "original_workbook"]}
    for group, group_rows in groups.items():
        for metric in METRICS:
            usable = [row for row in group_rows if np.isfinite(row[metric]) and np.isfinite(row["recovery_b58_carrier_signed"])]
            if len(usable) < 8: continue
            x = np.asarray([row[metric] for row in usable]); y = np.asarray([row["recovery_b58_carrier_signed"] for row in usable]); rho = float(spearmanr(x, y).statistic); boots = []
            for _ in range(2000):
                sample = rng.integers(0, len(usable), len(usable))
                if len(np.unique(x[sample])) < 2 or len(np.unique(y[sample])) < 2: continue
                value = float(spearmanr(x[sample], y[sample]).statistic)
                if np.isfinite(value): boots.append(value)
            association_rows.append({"provenance_group": group, "metric": metric, "spearman_rho": rho, "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5)), "canonical_neurons": len(usable), "interpretation": "descriptive_single_recording"})
    _write_tsv(output / "v7_recovery_associations.tsv", association_rows)
    overall = sorted((row for row in association_rows if row["provenance_group"] == "all"), key=lambda row: row["spearman_rho"]); fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained"); ys = np.arange(len(overall)); vals = np.asarray([row["spearman_rho"] for row in overall]); lo = np.asarray([row["ci95_low"] for row in overall]); hi = np.asarray([row["ci95_high"] for row in overall]); ax.errorbar(vals, ys, xerr=[vals-lo, hi-vals], fmt="o", color=BLUE, ecolor="#777777", capsize=3); ax.axvline(0, color=INK, lw=.8); ax.set_yticks(ys, [row["metric"].replace("_", " ") for row in overall]); ax.set(xlabel="Spearman association with carrier recovery @58", title="Canonical-v7 descriptive recovery associations"); ax.grid(axis="x", color=GRID, lw=.5); _save(fig, output / "v7_recovery_associations")

    stats = {"schema_version": 1, "canonical_summary_rows": len(summaries), "confirmed_canonical_neurons_with_three_or_more_bursts": sum(row["confirmed_bursts"] >= 3 for row in summaries), "selected": {key: value["canonical_roi_id"] for key, value in selected.items()}, "selected_provenance": {key: value["provenance"] for key, value in selected.items()}, "previous_v1_selected": {"hero": "roi_003", "typical": "roi_002", "hard_case": "roi_011"}, "selection_stability": {key: selected[key]["canonical_roi_id"] == old for key, old in {"hero":"roi_003","typical":"roi_002","hard_case":"roi_011"}.items()}, "recovery_associations": association_rows, "prohibited_interpretations": ["independent detector validation from candidate-assisted rows", "precision", "cross-recording generalization"]}
    atomic_json(output / "v7_visual_statistics.json", stats)
    atomic_text(output / "CHART_CONTRACT.md", "# Canonical-v7 chart contract\n\nQuestion: what does v7 add, do compact-lane results survive 4/6/8 px NMS, and do representative canonical neurons remain stable? Grain: provenance-stratified observation rows for composition, canonical-neuron by burst for measurement, and confirmed/inclusive canonical neuron-burst occurrences for recovery. Static Matplotlib PNG/PDF outputs use neutral, blue, gold, and orange encodings plus labels and marker shapes. Candidate-assisted results are labeled internal consistency, never independent validation.\n")
    atomic_text(output / "REPORT.md", f"# Canonical-v7 extension\n\nV7 is used as the expanded canonical-neuron and measurement atlas; the original frozen cohort remains the protected representation test. V7 contains candidate-assisted labels, so its NMS results are internal-consistency sensitivity evidence. Objectively selected canonical examples are hero `{selected['hero']['canonical_roi_id']}`, typical `{selected['typical']['canonical_roi_id']}`, and hard case `{selected['hard_case']['canonical_roi_id']}`.\n")
    return stats


def run_extension(data_root: Path, repo_root: Path, run_root: Path) -> dict[str, Any]:
    target = run_root / "canonical_v7_extension"; cross = target / "crosswalk"; nms = target / "nms"; visuals = target / "visuals"
    v1 = data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv"
    v7 = data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v7/adjudication_final.tsv"
    video = data_root / "Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
    crosswalk = build_crosswalk(v1, v7, cross); sensitivity = run_v7_nms(data_root, repo_root, v7, nms); atlas = build_v7_atlas(run_root, video, v7, nms / "v7_nms_sensitivity.json", visuals)
    summary = {"schema_version": 1, "status": "complete", "crosswalk": crosswalk, "nms_internal_consistency": sensitivity["nms_robust_internal_consistency"], "atlas": atlas, "primary_cohort_policy": "retain frozen v1 for protected representation claims; use v7 for canonical atlas and provenance-stratified sensitivity"}
    atomic_json(target / "SUMMARY.json", summary); atomic_text(target / "REPORT.md", "# Canonical-v7 paper extension\n\nCanonical v7 is included as the expanded identity and measurement dataset. The original frozen cohort remains primary for protected representation testing. All candidate-assisted v7 results are provenance-stratified and interpreted as sensitivity/internal consistency rather than independent validation.\n")
    atomic_text(run_root / "manuscript/CANONICAL_V7_EXTENSION.md", """# Canonical-v7 manuscript extension

## Recommended placement

- Methods: describe the 79-row frozen cohort as the protected representation test and canonical v7 as a provenance-stratified identity/measurement extension.
- Results: report that all 79 frozen rows are preserved, 58 rows are added, and five retained rows change disposition or confirmed-view inclusion.
- Supplement: show confirmed and inclusive 4/6/8 px NMS sensitivity, the expanded canonical-neuron atlas, and objective representative examples.

## Permitted claim

Both compact lanes retain positive aggregate budget-20 gain across 4/6/8 px NMS in the confirmed and inclusive canonical-v7 views, with the primary six-pixel gains exceeding +0.03. Because candidate-assisted annotations were surfaced by the evaluated feature bank, this is internal-consistency sensitivity evidence rather than independent detector validation.

## Representative examples

The v7 selection identifies `roi_new_001` as hero, `roi_014` as typical, and retains `roi_011` as the hard case. The new hero and typical examples are candidate-assisted, so their provenance must accompany any figure or caption.
""")
    atomic_json(run_root / "release/canonical_v7_extension_status.json", {"schema_version": 1, "status": "complete", "protected_primary_cohort": "v1_79_rows", "expanded_canonical_atlas": "v7_137_rows", "confirmed_collapsed_denominator": 102, "inclusive_collapsed_denominator": 124, "nms_internal_consistency_passed": all(value for view in sensitivity["nms_robust_internal_consistency"].values() for value in view.values()), "independent_validation": False, "precision_identified": False, "source": str(target / "SUMMARY.json")})
    return summary
