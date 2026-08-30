"""Post-freeze associations and longitudinal summaries for detection classes.

The frozen detection classes are never refit here.  Reviewer certainty is joined
only after spatial matching, so this module describes association rather than
classification performance or biological cell type.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from .contracts import atomic_json, atomic_text
from neurobench.portable_paths import data_root as configured_data_root

BLUE = "#3F77B5"
ORANGE = "#E17C24"
OLIVE = "#7E9636"
INK = "#222222"
GRID = "#D9D9D9"
CLASS_COLORS = {1: BLUE, 2: ORANGE, 3: OLIVE}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def greedy_spatial_match(
    detections: Iterable[dict[str, Any]], labels: Iterable[dict[str, Any]], radius_px: float
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """Deterministic one-to-one, within-burst nearest-pair matching."""
    detections = list(detections)
    labels = list(labels)
    pairs: list[tuple[float, str, str, int, int]] = []
    for i, det in enumerate(detections):
        for j, label in enumerate(labels):
            if int(det["burst_id"]) != int(label["burst_id"]):
                continue
            distance = float(np.hypot(float(det["x_px"]) - float(label["x_px"]), float(det["y_px"]) - float(label["y_px"])))
            if distance <= radius_px:
                pairs.append((distance, str(det["detection_occurrence_id"]), str(label["observation_id"]), i, j))
    used_detections: set[int] = set()
    used_labels: set[int] = set()
    matches = []
    for distance, _, _, i, j in sorted(pairs):
        if i in used_detections or j in used_labels:
            continue
        used_detections.add(i)
        used_labels.add(j)
        matches.append((detections[i], labels[j], distance))
    return matches


def cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.sum() == 0 or min(table.shape) < 2:
        return float("nan")
    chi2 = float(chi2_contingency(table, correction=False)[0])
    return float(np.sqrt((chi2 / table.sum()) / min(table.shape[0] - 1, table.shape[1] - 1)))


def stratified_permutation_p(rows: list[dict[str, Any]], *, repeats: int = 20_000, seed: int = 20260824) -> tuple[float, float]:
    """Shuffle certainty within burst; return observed Cramer's V and Monte Carlo p."""
    classes = sorted({int(row["class_id"]) for row in rows})
    groups = sorted({str(row["certainty_group"]) for row in rows})

    def statistic(values: list[str]) -> float:
        table = np.zeros((len(groups), len(classes)), dtype=int)
        group_index = {value: i for i, value in enumerate(groups)}
        class_index = {value: i for i, value in enumerate(classes)}
        for row, value in zip(rows, values, strict=True):
            table[group_index[value], class_index[int(row["class_id"])]] += 1
        return cramers_v(table)

    observed_values = [str(row["certainty_group"]) for row in rows]
    observed = statistic(observed_values)
    rng = np.random.default_rng(seed)
    burst_indices: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        burst_indices[int(row["burst_id"])].append(index)
    exceed = 0
    for _ in range(repeats):
        shuffled = observed_values.copy()
        for indices in burst_indices.values():
            values = [shuffled[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values, strict=True):
                shuffled[index] = value
        exceed += statistic(shuffled) >= observed - 1e-12
    return observed, float((exceed + 1) / (repeats + 1))


def _certainty_v1(row: dict[str, str]) -> str | None:
    if row["review_status"] != "adjudicated":
        return None
    return {"confirmed": "confirmed", "uncertain": "identity uncertain", "probable": "probable"}.get(row["neuron_confidence"])


def _certainty_v7(row: dict[str, str]) -> str | None:
    if row["review_status"] != "adjudicated":
        return None
    return {
        "confirmed_neuron": "confirmed",
        "activity_visible_identity_uncertain": "identity uncertain",
        "artifact": "artifact",
    }.get(row["disposition"])


def _matched_rows(
    detections: list[dict[str, str]], labels: list[dict[str, str]], cohort: str, radius_px: float
) -> list[dict[str, Any]]:
    certainty = _certainty_v1 if cohort == "frozen_v1_adjudicated" else _certainty_v7
    eligible = [row for row in labels if certainty(row) is not None]
    provenance = "v1 frozen adjudicated" if cohort == "frozen_v1_adjudicated" else "v7 candidate-assisted adjudicated"
    rows = []
    for detection, label, distance in greedy_spatial_match(detections, eligible, radius_px):
        rows.append({
            "cohort": cohort,
            "detection_occurrence_id": detection["detection_occurrence_id"],
            "detection_site_id": detection["detection_site_id"],
            "burst_id": int(detection["burst_id"]),
            "class_id": int(detection["class_id"]),
            "certainty_group": certainty(label),
            "observation_id": label["observation_id"],
            "canonical_roi_id": label["canonical_roi_id"],
            "reviewer_provenance": label["reviewer_id"],
            "match_distance_px": distance,
            "association_status": "post_freeze_descriptive",
            "selection_note": provenance,
        })
    return rows


def _contingency(rows: list[dict[str, Any]], cohort: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = sorted({row["certainty_group"] for row in rows}, key=lambda x: (x != "confirmed", x))
    classes = [1, 2, 3]
    output = []
    table = np.zeros((len(groups), len(classes)), dtype=int)
    for i, group in enumerate(groups):
        denominator = sum(row["certainty_group"] == group for row in rows)
        for j, class_id in enumerate(classes):
            count = sum(row["certainty_group"] == group and row["class_id"] == class_id for row in rows)
            table[i, j] = count
            output.append({"cohort": cohort, "certainty_group": group, "class_id": class_id, "matched_occurrences": count, "certainty_group_total": denominator, "fraction_within_certainty_group": count / denominator if denominator else float("nan")})
    comparison_rows = [row for row in rows if row["certainty_group"] in {"confirmed", "identity uncertain"}]
    comparison_groups = {row["certainty_group"] for row in comparison_rows}
    effect, p_value = stratified_permutation_p(comparison_rows) if len(comparison_groups) == 2 else (None, None)
    return output, {"groups": groups, "classes": classes, "table": table.tolist(), "matched_occurrences": int(table.sum()), "confirmed_vs_uncertain_cramers_v": effect, "confirmed_vs_uncertain_burst_stratified_permutation_p": p_value, "permutation_repeats": 20_000 if effect is not None else 0}


def _label_matching_summary(labels: list[dict[str, str]], matches: list[dict[str, Any]], cohort: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    certainty = _certainty_v1 if cohort == "frozen_v1_adjudicated" else _certainty_v7
    eligible = [row for row in labels if certainty(row) is not None]
    matched_ids = {row["observation_id"] for row in matches}
    groups = sorted({certainty(row) for row in eligible}, key=lambda x: (x != "confirmed", x))
    rows = []
    for group in groups:
        group_rows = [row for row in eligible if certainty(row) == group]
        matched = sum(row["observation_id"] in matched_ids for row in group_rows)
        rows.append({"cohort": cohort, "certainty_group": group, "eligible_labels": len(group_rows), "matched_labels": matched, "unmatched_labels": len(group_rows) - matched, "matched_fraction": matched / len(group_rows)})
    lookup = {row["certainty_group"]: row for row in rows}
    fisher = None
    if {"confirmed", "identity uncertain"} <= set(lookup):
        confirmed = lookup["confirmed"]; uncertain = lookup["identity uncertain"]
        odds, p_value = fisher_exact([[confirmed["matched_labels"], confirmed["unmatched_labels"]], [uncertain["matched_labels"], uncertain["unmatched_labels"]]])
        fisher = {"odds_ratio": None if not np.isfinite(odds) else float(odds), "two_sided_p": float(p_value), "table": [[confirmed["matched_labels"], confirmed["unmatched_labels"]], [uncertain["matched_labels"], uncertain["unmatched_labels"]]]}
    return rows, {"confirmed_vs_uncertain_fisher_exact": fisher}


def _transition_summary(detections: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detections:
        by_site[row["detection_site_id"]].append(row)
    table = np.zeros((3, 3), dtype=int)
    site_rows = []
    for site, rows in sorted(by_site.items()):
        rows.sort(key=lambda row: int(row["burst_id"]))
        classes = [int(row["class_id"]) for row in rows]
        for left, right in zip(classes, classes[1:]):
            table[left - 1, right - 1] += 1
        counts = Counter(classes)
        probabilities = np.asarray(list(counts.values()), dtype=float) / len(classes)
        entropy = float(-np.sum(probabilities * np.log2(probabilities))) if len(classes) > 1 else 0.0
        site_rows.append({"detection_site_id": site, "detected_bursts": len(rows), "dominant_class_id": min((-count, class_id) for class_id, count in counts.items())[1], "dominant_class_fraction": max(counts.values()) / len(classes), "class_entropy_bits": entropy, "class_sequence": ">".join(map(str, classes))})
    transitions = []
    for source in range(1, 4):
        denominator = int(table[source - 1].sum())
        for target in range(1, 4):
            transitions.append({"source_class_id": source, "target_class_id": target, "transitions": int(table[source - 1, target - 1]), "fraction_within_source": table[source - 1, target - 1] / denominator if denominator else float("nan")})
    total = int(table.sum())
    same = int(np.trace(table))
    return transitions, {"transition_pairs": total, "same_class_transitions": same, "same_class_fraction": same / total if total else float("nan"), "sites_with_multiple_detected_bursts": sum(row["detected_bursts"] > 1 for row in site_rows), "table": table.tolist()}, site_rows


def _class_metrics(detections: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for class_id in (1, 2, 3):
        subset = [row for row in detections if int(row["class_id"]) == class_id]
        rows.append({
            "class_id": class_id,
            "occurrences": len(subset),
            "sites": len({row["detection_site_id"] for row in subset}),
            "median_recurrence_fraction": float(np.median([float(row["recurrence_fraction"]) for row in subset])),
            "median_lane_agreement": float(np.median([float(row["lane_agreement"]) for row in subset])),
            "median_rank_fraction": float(np.median([float(row["mean_rank_fraction"]) for row in subset])),
            "median_spatial_specificity": float(np.median([float(row["spatial_specificity"]) for row in subset])),
            "median_annulus_correlation": float(np.median([float(row["annulus_correlation"]) for row in subset])),
        })
    return rows


def _plot(output: Path, matching_rows: list[dict[str, Any]], contingency_rows: list[dict[str, Any]], transitions: list[dict[str, Any]], class_metrics: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    groups = ["confirmed", "identity uncertain"]
    x = np.arange(2); width = .34
    for offset, cohort, color, label in ((-width / 2, "frozen_v1_adjudicated", "#666666", "Frozen v1"), (width / 2, "canonical_v7_adjudicated", BLUE, "Canonical v7")):
        values = [next(row["matched_fraction"] for row in matching_rows if row["cohort"] == cohort and row["certainty_group"] == group) for group in groups]
        totals = [next(row["eligible_labels"] for row in matching_rows if row["cohort"] == cohort and row["certainty_group"] == group) for group in groups]
        bars = axes[0, 0].bar(x + offset, values, width, color=color, edgecolor=INK, label=label)
        for bar, value, total in zip(bars, values, totals, strict=True):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value + .025, f"{round(value * total)}/{total}", ha="center", fontsize=8)
    axes[0, 0].set(xticks=x, xticklabels=["Confirmed", "Identity uncertain"], ylim=(0, 1.05), ylabel="fraction matched by strict B20 union", title="Detection coverage by adjudicated certainty")
    axes[0, 0].grid(axis="y", color=GRID, lw=.5); axes[0, 0].legend(fontsize=8)

    subset = [row for row in contingency_rows if row["cohort"] == "canonical_v7_adjudicated" and row["certainty_group"] in groups]
    bottom = np.zeros(2)
    for class_id in (1, 2, 3):
        values = np.asarray([next(row["fraction_within_certainty_group"] for row in subset if row["certainty_group"] == group and row["class_id"] == class_id) for group in groups])
        axes[0, 1].bar(groups, values, bottom=bottom, color=CLASS_COLORS[class_id], edgecolor=INK, linewidth=.6, label=f"Class {class_id}")
        bottom += values
    totals = [next(row["certainty_group_total"] for row in subset if row["certainty_group"] == group) for group in groups]
    for index, total in enumerate(totals):
        axes[0, 1].text(index, 1.02, f"matched n={total}", ha="center", va="bottom", fontsize=9)
    axes[0, 1].set_ylim(0, 1.12); axes[0, 1].set_ylabel("class composition among matched detections"); axes[0, 1].set_title("Canonical-v7 class composition"); axes[0, 1].tick_params(axis="x", rotation=15); axes[0, 1].grid(axis="y", color=GRID, lw=.5); axes[0, 1].legend(ncol=3, fontsize=8, loc="upper center")

    matrix = np.asarray([[next(row["transitions"] for row in transitions if row["source_class_id"] == i and row["target_class_id"] == j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    image = axes[1, 0].imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[1, 0].text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else INK)
    axes[1, 0].set(xticks=range(3), yticks=range(3), xticklabels=["1", "2", "3"], yticklabels=["1", "2", "3"], xlabel="later detected class", ylabel="earlier detected class", title="Within-site class transitions")
    fig.colorbar(image, ax=axes[1, 0], label="transition pairs")

    x = np.arange(3); width = .34
    recurrence = [row["median_recurrence_fraction"] for row in class_metrics]
    lane = [row["median_lane_agreement"] for row in class_metrics]
    axes[1, 1].bar(x - width / 2, recurrence, width, color="#666666", edgecolor=INK, label="recurrence")
    axes[1, 1].bar(x + width / 2, lane, width, color=[CLASS_COLORS[i] for i in (1, 2, 3)], edgecolor=INK, label="lane agreement")
    axes[1, 1].set(xticks=x, xticklabels=["Class 1", "Class 2", "Class 3"], ylim=(0, 1.05), ylabel="median fraction", title="Class recurrence and detector-lane agreement")
    axes[1, 1].grid(axis="y", color=GRID, lw=.5); axes[1, 1].legend(fontsize=8)
    fig.suptitle("Post-freeze detection-class associations and stability", fontsize=16)
    fig.savefig(output / "detection_class_extensions.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "detection_class_extensions.pdf", bbox_inches="tight")
    plt.close(fig)


def run(run_root: Path, v1_path: Path, v7_path: Path, output: Path, *, radius_px: float = 6.0) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    detections = read_tsv(run_root / "detection_profile_taxonomy_v5/detection_occurrence_profiles.tsv")
    v1_labels = read_tsv(v1_path); v7_labels = read_tsv(v7_path)
    v1_matches = _matched_rows(detections, v1_labels, "frozen_v1_adjudicated", radius_px)
    v7_matches = _matched_rows(detections, v7_labels, "canonical_v7_adjudicated", radius_px)
    contingency_rows: list[dict[str, Any]] = []
    associations = {}
    for cohort, rows in (("frozen_v1_adjudicated", v1_matches), ("canonical_v7_adjudicated", v7_matches)):
        table_rows, summary = _contingency(rows, cohort)
        contingency_rows.extend(table_rows); associations[cohort] = summary
    matching_rows = []
    for cohort, labels, matches in (("frozen_v1_adjudicated", v1_labels, v1_matches), ("canonical_v7_adjudicated", v7_labels, v7_matches)):
        cohort_rows, matching_summary = _label_matching_summary(labels, matches, cohort)
        matching_rows.extend(cohort_rows); associations[cohort].update(matching_summary)
    transitions, transition_summary, site_rows = _transition_summary(detections)
    class_metrics = _class_metrics(detections)
    write_tsv(output / "certainty_matches.tsv", v1_matches + v7_matches)
    write_tsv(output / "label_matching_by_certainty.tsv", matching_rows)
    write_tsv(output / "class_by_certainty.tsv", contingency_rows)
    write_tsv(output / "class_transitions.tsv", transitions)
    write_tsv(output / "site_class_stability.tsv", site_rows)
    write_tsv(output / "class_extension_metrics.tsv", class_metrics)
    _plot(output, matching_rows, contingency_rows, transitions, class_metrics)
    summary = {
        "schema_version": 1,
        "status": "passed",
        "class_fit_modified": False,
        "matching_radius_px": radius_px,
        "associations": associations,
        "transition_summary": transition_summary,
        "interpretation": "post-freeze descriptive association of measurement-event classes with reviewer identity certainty",
        "limitations": [
            "legacy labels without adjudicated certainty are excluded from the frozen certainty comparison",
            "canonical-v7 labels are candidate-assisted and are not independent validation",
            "classes are detection-event measurement archetypes, not neuron types",
            "permutation p-values are descriptive because repeated sites and candidate selection limit exchangeability",
        ],
    }
    atomic_json(output / "summary.json", summary)
    atomic_text(output / "REPORT.md", "# Detection-class extensions\n\nThis post-freeze analysis compares the frozen detection-event classes with adjudicated identity certainty and summarizes within-site class transitions. Expert certainty was not used to fit or rename classes. See `summary.json` for effect sizes and limitations, and `class_by_certainty.tsv` for exact denominators.\n")
    return summary


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[3]
    data_root = configured_data_root(repository)
    run_root = repository / "Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8"
    print(json.dumps(run(
        run_root,
        data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv",
        data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v7/adjudication_final.tsv",
        run_root / "detection_class_extensions_v1",
    ), indent=2))
