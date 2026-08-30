"""Advanced post-freeze profiling of detection-event classes."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

from .contracts import atomic_json, atomic_text
from .detection_class_extensions import CLASS_COLORS, INK, GRID, cramers_v, read_tsv, write_tsv
from .detection_profile_taxonomy import FEATURES
from neurobench.portable_paths import data_root

FIELD_BOUNDARY_X = 286.0


def membership_margins(detections: list[dict[str, str]], taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    centers = taxonomy["classification"]["standardized_centroids"]
    center_ids = sorted(int(value) for value in centers)
    center_matrix = np.asarray([[float(centers[str(class_id)][feature]) for feature in FEATURES] for class_id in center_ids])
    rows = []
    for detection in detections:
        vector = np.asarray([float(detection[f"z_{feature}"]) for feature in FEATURES])
        distances = np.sqrt(np.sum((center_matrix - vector) ** 2, axis=1))
        order = np.argsort(distances)
        nearest, second = int(order[0]), int(order[1])
        assigned = center_ids[nearest]
        if assigned != int(detection["class_id"]):
            raise ValueError(f"class/centroid mismatch: {detection['detection_occurrence_id']}")
        relative = float((distances[second] - distances[nearest]) / max(distances[second], np.finfo(float).eps))
        rows.append({
            "detection_occurrence_id": detection["detection_occurrence_id"], "detection_site_id": detection["detection_site_id"],
            "burst_id": int(detection["burst_id"]), "class_id": assigned,
            "nearest_centroid_distance": float(distances[nearest]), "second_centroid_id": center_ids[second],
            "second_centroid_distance": float(distances[second]), "absolute_distance_margin": float(distances[second] - distances[nearest]),
            "relative_distance_margin": relative,
            **{f"distance_to_class_{class_id}": float(distances[index]) for index, class_id in enumerate(center_ids)},
        })
    threshold = float(np.quantile([row["relative_distance_margin"] for row in rows], .25))
    for row in rows:
        row["ambiguous_lowest_margin_quartile"] = row["relative_distance_margin"] <= threshold
        row["ambiguity_threshold"] = threshold
    return rows


def _collapsed_morphology(value: str) -> str:
    if value in {"localized_center", "cell_center", "localized_signal", "spatially_cohesive"}:
        return "localized boundary"
    return "ambiguous/non-neuronal boundary"


def _collapsed_context(value: str) -> str:
    if value == "isolated": return "isolated"
    if value == "overlapping": return "overlapping"
    return "challenging/structured context"


def enrich_matches(matches: list[dict[str, str]], labels: list[dict[str, str]], margins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_by_id = {row["observation_id"]: row for row in labels}
    margin_by_id = {row["detection_occurrence_id"]: row for row in margins}
    rows = []
    for match in matches:
        if match["cohort"] != "canonical_v7_adjudicated": continue
        label = label_by_id[match["observation_id"]]; margin = margin_by_id[match["detection_occurrence_id"]]
        rows.append({
            **match,
            "neuron_confidence": label["neuron_confidence"], "disposition": label["disposition"],
            "morphology": label["morphology"], "morphology_group": _collapsed_morphology(label["morphology"]),
            "context": label["context"], "context_group": _collapsed_context(label["context"]),
            "relative_distance_margin": margin["relative_distance_margin"],
            "ambiguous_lowest_margin_quartile": margin["ambiguous_lowest_margin_quartile"],
        })
    return rows


def categorical_association(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    categories = sorted({str(row[field]) for row in rows}); classes = [1, 2, 3]
    table = np.asarray([[sum(str(row[field]) == category and int(row["class_id"]) == class_id for row in rows) for class_id in classes] for category in categories])
    active_rows = table.sum(axis=1) > 0; active_columns = table.sum(axis=0) > 0; active = table[active_rows][:, active_columns]
    return {"categories": categories, "classes": classes, "table": table.tolist(), "cramers_v": cramers_v(active), "n": int(table.sum())}


def spatial_and_burst(detections: list[dict[str, str]], sites: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for class_id in (1, 2, 3):
        subset = [row for row in sites if int(row["dominant_class"]) == class_id]
        rows.append({"class_id": class_id, "detection_sites": len(subset), "left_field_sites": sum(float(row["x_px"]) < FIELD_BOUNDARY_X for row in subset), "right_field_sites": sum(float(row["x_px"]) >= FIELD_BOUNDARY_X for row in subset), "median_x_px": float(np.median([float(row["x_px"]) for row in subset])), "median_y_px": float(np.median([float(row["y_px"]) for row in subset]))})
    field_table = np.asarray([[row["left_field_sites"], row["right_field_sites"]] for row in rows])
    burst_table = np.asarray([[sum(int(row["burst_id"]) == burst and int(row["class_id"]) == class_id for row in detections) for class_id in (1, 2, 3)] for burst in (1, 2, 3, 4)])
    x_groups = [[float(row["x_px"]) for row in sites if int(row["dominant_class"]) == class_id] for class_id in (1, 2, 3)]
    y_groups = [[float(row["y_px"]) for row in sites if int(row["dominant_class"]) == class_id] for class_id in (1, 2, 3)]
    rng = np.random.default_rng(20260824)
    site_labels = np.asarray([int(row["dominant_class"]) for row in sites])
    sides = np.asarray([float(row["x_px"]) >= FIELD_BOUNDARY_X for row in sites])
    occurrence_labels = np.asarray([int(row["class_id"]) for row in detections])
    bursts = np.asarray([int(row["burst_id"]) for row in detections])
    observed_field = cramers_v(field_table)
    observed_burst = cramers_v(burst_table)
    field_exceed = 0; burst_exceed = 0; repeats = 50_000
    for _ in range(repeats):
        shuffled = rng.permutation(site_labels)
        perm_field = np.asarray([[np.sum((shuffled == class_id) & (~sides)), np.sum((shuffled == class_id) & sides)] for class_id in (1, 2, 3)])
        shuffled_occurrences = rng.permutation(occurrence_labels)
        perm_burst = np.asarray([[np.sum((bursts == burst) & (shuffled_occurrences == class_id)) for class_id in (1, 2, 3)] for burst in (1, 2, 3, 4)])
        field_exceed += cramers_v(perm_field) >= observed_field - 1e-12
        burst_exceed += cramers_v(perm_burst) >= observed_burst - 1e-12
    return rows, {
        "spatial_grain": "36 consolidated detection sites using dominant class", "field_boundary_x_px": FIELD_BOUNDARY_X, "field_table_class_by_side": field_table.tolist(),
        "field_cramers_v": observed_field, "field_chi_square_p_not_primary_sparse_expected_counts": float(chi2_contingency(field_table, correction=False)[1]), "field_label_permutation_p": (field_exceed + 1) / (repeats + 1),
        "x_kruskal_p": float(kruskal(*x_groups).pvalue), "y_kruskal_p": float(kruskal(*y_groups).pvalue),
        "burst_table": burst_table.tolist(), "burst_cramers_v": observed_burst, "burst_chi_square_p_not_primary_sparse_expected_counts": float(chi2_contingency(burst_table, correction=False)[1]), "burst_label_permutation_p": (burst_exceed + 1) / (repeats + 1), "permutation_repeats": repeats,
    }


def canonical_trajectories(enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        if row["certainty_group"] == "artifact": continue
        by_canonical[row["canonical_roi_id"]].append(row)
    output = []
    for canonical, rows in sorted(by_canonical.items()):
        best_by_burst = {}
        for row in sorted(rows, key=lambda value: (int(value["burst_id"]), float(value["match_distance_px"]), value["detection_occurrence_id"])):
            best_by_burst.setdefault(int(row["burst_id"]), row)
        ordered = [best_by_burst[burst] for burst in sorted(best_by_burst)]
        counts = Counter(int(row["class_id"]) for row in ordered); probabilities = np.asarray(list(counts.values()), float) / len(ordered)
        output.append({
            "canonical_roi_id": canonical, "matched_bursts": len(ordered),
            "certainty_groups": ",".join(sorted({row["certainty_group"] for row in ordered})),
            "class_sequence": ">".join(str(row["class_id"]) for row in ordered),
            "burst_sequence": ">".join(str(row["burst_id"]) for row in ordered),
            "dominant_class_id": min((-count, class_id) for class_id, count in counts.items())[1],
            "dominant_class_fraction": max(counts.values()) / len(ordered),
            "class_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))) if len(probabilities) > 1 else 0.0,
            **{f"burst_{burst}_class": best_by_burst.get(burst, {}).get("class_id", "") for burst in (1, 2, 3, 4)},
        })
    return output


def _plot(output: Path, detections: list[dict[str, str]], sites: list[dict[str, str]], margins: list[dict[str, Any]], enriched: list[dict[str, Any]], trajectories: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    values = [[row["relative_distance_margin"] for row in margins if row["class_id"] == class_id] for class_id in (1, 2, 3)]
    box = axes[0, 0].boxplot(values, tick_labels=["Class 1", "Class 2", "Class 3"], patch_artist=True, showfliers=True)
    for patch, class_id in zip(box["boxes"], (1, 2, 3), strict=True): patch.set_facecolor(CLASS_COLORS[class_id]); patch.set_alpha(.75)
    axes[0, 0].set(ylabel="relative nearest-centroid margin", title="Hard-assignment separation"); axes[0, 0].grid(axis="y", color=GRID, lw=.5)

    morphology_groups = ["localized boundary", "ambiguous/non-neuronal boundary"]
    x = np.arange(3); width = .36
    for offset, group, color in ((-width / 2, morphology_groups[0], "#4C78A8"), (width / 2, morphology_groups[1], "#E17C24")):
        counts = [sum(row["morphology_group"] == group and int(row["class_id"]) == class_id for row in enriched) for class_id in (1, 2, 3)]
        axes[0, 1].bar(x + offset, counts, width, color=color, edgecolor=INK, label=group)
        for index, count in enumerate(counts): axes[0, 1].text(index + offset, count + .2, str(count), ha="center", fontsize=8)
    axes[0, 1].set(xticks=x, xticklabels=["Class 1", "Class 2", "Class 3"], ylabel="matched canonical-v7 observations", title="Reviewer morphology by frozen class"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(axis="y", color=GRID, lw=.5)

    for class_id in (1, 2, 3):
        subset = [row for row in sites if int(row["dominant_class"]) == class_id]
        axes[1, 0].scatter([float(row["x_px"]) for row in subset], [float(row["y_px"]) for row in subset], color=CLASS_COLORS[class_id], label=f"Class {class_id}", s=32, alpha=.75, edgecolor=INK, linewidth=.3)
    axes[1, 0].axvline(FIELD_BOUNDARY_X, color=INK, ls="--", lw=1, label="provisional field boundary")
    axes[1, 0].invert_yaxis(); axes[1, 0].set(xlabel="x (px)", ylabel="y (px)", title="Consolidated-site positions and field boundary"); axes[1, 0].legend(fontsize=8); axes[1, 0].set_aspect("equal")

    eligible = [row for row in trajectories if row["matched_bursts"] >= 2]
    eligible.sort(key=lambda row: (-row["matched_bursts"], row["canonical_roi_id"]))
    matrix = np.full((len(eligible), 4), np.nan)
    for i, row in enumerate(eligible):
        for burst in range(1, 5):
            value = row[f"burst_{burst}_class"]
            if value != "": matrix[i, burst - 1] = int(value)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap([CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]]); cmap.set_bad("#EEEEEE")
    axes[1, 1].imshow(matrix, aspect="auto", cmap=cmap, norm=BoundaryNorm([.5, 1.5, 2.5, 3.5], cmap.N))
    axes[1, 1].set(xticks=range(4), xticklabels=["B1", "B2", "B3", "B4"], yticks=range(len(eligible)), yticklabels=[row["canonical_roi_id"] for row in eligible], xlabel="burst", title="Canonical-v7 matched class trajectories")
    axes[1, 1].tick_params(axis="y", labelsize=7)
    fig.suptitle("Advanced post-freeze class profiling", fontsize=16)
    fig.savefig(output / "detection_class_advanced_extensions.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "detection_class_advanced_extensions.pdf", bbox_inches="tight")
    plt.close(fig)


def run(run_root: Path, v7_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_root = run_root / "detection_profile_taxonomy_v5"
    detections = read_tsv(taxonomy_root / "detection_occurrence_profiles.tsv")
    sites = read_tsv(taxonomy_root / "detection_site_profiles.tsv")
    taxonomy = json.loads((taxonomy_root / "summary.json").read_text())
    margins = membership_margins(detections, taxonomy)
    matches = read_tsv(run_root / "detection_class_extensions_v1/certainty_matches.tsv")
    v7_labels = read_tsv(v7_path)
    enriched = enrich_matches(matches, v7_labels, margins)
    trajectories = canonical_trajectories(enriched)
    spatial_rows, spatial_summary = spatial_and_burst(detections, sites)
    morphology = categorical_association(enriched, "morphology_group")
    context = categorical_association(enriched, "context_group")
    confirmed_margins = [float(row["relative_distance_margin"]) for row in enriched if row["certainty_group"] == "confirmed"]
    uncertain_margins = [float(row["relative_distance_margin"]) for row in enriched if row["certainty_group"] == "identity uncertain"]
    margin_test = mannwhitneyu(confirmed_margins, uncertain_margins, alternative="two-sided")
    ambiguous = [row for row in margins if row["ambiguous_lowest_margin_quartile"]]
    write_tsv(output / "membership_margins.tsv", margins)
    write_tsv(output / "AMBIGUOUS_CLASS_REVIEW.tsv", sorted(ambiguous, key=lambda row: (row["relative_distance_margin"], row["detection_occurrence_id"])))
    write_tsv(output / "matched_morphology_context.tsv", enriched)
    write_tsv(output / "spatial_class_summary.tsv", spatial_rows)
    write_tsv(output / "canonical_class_trajectories.tsv", trajectories)
    _plot(output, detections, sites, margins, enriched, trajectories)
    summary = {
        "schema_version": 1, "status": "passed", "class_fit_modified": False,
        "soft_membership": {"measure": "relative nearest-versus-second-centroid distance margin; not a calibrated probability", "ambiguous_occurrences": len(ambiguous), "ambiguity_fraction": len(ambiguous) / len(margins), "ambiguity_threshold": margins[0]["ambiguity_threshold"], "median_margin_by_class": {str(class_id): float(np.median([row["relative_distance_margin"] for row in margins if row["class_id"] == class_id])) for class_id in (1, 2, 3)}, "confirmed_vs_uncertain_mann_whitney_p": float(margin_test.pvalue), "confirmed_median": float(np.median(confirmed_margins)), "uncertain_median": float(np.median(uncertain_margins))},
        "morphology_association": morphology, "context_association": context,
        "spatial_and_burst": spatial_summary,
        "canonical_trajectories": {"canonical_ids": len(trajectories), "with_two_or_more_matched_bursts": sum(row["matched_bursts"] >= 2 for row in trajectories), "median_dominant_class_fraction_multi_burst": float(np.median([row["dominant_class_fraction"] for row in trajectories if row["matched_bursts"] >= 2]))},
        "limitations": ["centroid margin is relative geometry, not a calibrated posterior probability", "morphology/context and canonical trajectories use candidate-assisted v7 review", "field boundary is provisional", "all tests are descriptive within one recording and multiple secondary comparisons are not confirmatory"],
    }
    atomic_json(output / "summary.json", summary)
    atomic_text(output / "REPORT.md", "# Advanced detection-class extensions\n\nThis post-freeze package reports centroid-assignment margins, reviewer morphology/context, spatial and burst composition, and canonical-v7 matched trajectories. It does not refit the taxonomy or infer biological neuron types.\n")
    return summary


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[3]
    run_root = repository / "Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8"
    v7 = data_root(repository) / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v7/adjudication_final.tsv"
    print(json.dumps(run(run_root, v7, run_root / "detection_class_advanced_extensions_v1"), indent=2))
