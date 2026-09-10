#!/usr/bin/env python3
"""Build the six-figure venue-paper set from frozen, compact evidence.

The figures are presentation-only derivatives. They do not refit a model,
change a cohort, or create a new scientific estimand. Every quantitative panel
is reconstructed from a compact source table or JSON artifact already cited by
the manuscript, and the emitted manifest binds sources to outputs by SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from PIL import Image


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
OUT = PAPER / "figures" / "venue"

BLUE = "#2F6DB2"
BLUE_DARK = "#173F6D"
BLUE_LIGHT = "#DCEAF7"
GOLD = "#D89B24"
GOLD_LIGHT = "#F7E8BF"
ORANGE = "#C95D3E"
ORANGE_LIGHT = "#F4D9CF"
INK = "#17212B"
MID = "#586574"
GRID = "#D9DEE5"
PALE = "#F5F7FA"
WHITE = "#FFFFFF"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": MID,
        "axes.labelcolor": INK,
        "xtick.color": MID,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def quiet_axes(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_blossom(fig: plt.Figure) -> None:
    center_x, center_y = 0.976, 0.965
    radius = 0.007
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        fig.add_artist(
            Circle(
                (center_x + 0.010 * np.cos(angle), center_y + 0.010 * np.sin(angle)),
                radius,
                transform=fig.transFigure,
                facecolor=BLUE,
                edgecolor=BLUE_DARK,
                linewidth=0.35,
                clip_on=False,
            )
        )


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    add_blossom(fig)
    path = OUT / name
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return path


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    face: str = PALE,
    edge: str = BLUE,
    fontsize: float = 9,
    weight: str = "normal",
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=MID,
            shrinkA=4,
            shrinkB=4,
        )
    )


def build_jnm_design() -> Path:
    cohort = json.loads(
        (PAPER / "generated_analysis/automated_feature_validation_v1/summary.json").read_text(encoding="utf-8")
    )["population"]
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "Identity-safe evaluation under sparse-positive annotation",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    ax.text(0.15, 0.89, "Frozen evidence", ha="center", fontweight="bold", fontsize=11)
    box(ax, (0.03, 0.67), (0.24, 0.15), f"One fluorescence recording\n{cohort['bursts']} burst intervals", face=BLUE_LIGHT)
    box(ax, (0.03, 0.45), (0.24, 0.15), f"{cohort['occurrences']} confirmed occurrences\n{cohort['sites']} immutable observation sites", face=BLUE_LIGHT)
    box(ax, (0.03, 0.23), (0.24, 0.15), "Unmatched candidates remain U\nnot verified negatives", face=GOLD_LIGHT, edge=GOLD)

    ax.text(0.50, 0.89, "Identity contract", ha="center", fontweight="bold", fontsize=11)
    box(ax, (0.37, 0.68), (0.26, 0.12), "Observation site\nimmutable identifier", face=PALE)
    box(ax, (0.37, 0.50), (0.26, 0.12), "Geometry + trace provenance\nseparately hashed", face=PALE)
    box(ax, (0.37, 0.32), (0.26, 0.12), "Canonical identity\nexplicit, provisional relation", face=PALE)
    box(ax, (0.37, 0.14), (0.26, 0.12), "Candidate state\nP / U / review outcome", face=PALE)
    arrow(ax, (0.50, 0.68), (0.50, 0.62))
    arrow(ax, (0.50, 0.50), (0.50, 0.44))
    arrow(ax, (0.50, 0.32), (0.50, 0.26))

    ax.text(0.83, 0.89, "Failure-stage decomposition", ha="center", fontweight="bold", fontsize=11)
    stages = ["Proposal", "Ranking", "NMS", "Identity", "Review"]
    y = 0.75
    for index, stage in enumerate(stages):
        face = ORANGE_LIGHT if stage == "Identity" else PALE
        edge = ORANGE if stage == "Identity" else BLUE
        box(ax, (0.72, y - index * 0.12), (0.22, 0.075), stage, face=face, edge=edge, weight="bold")
        if index < len(stages) - 1:
            arrow(ax, (0.83, y - index * 0.12), (0.83, y - index * 0.12 - 0.045))

    arrow(ax, (0.27, 0.525), (0.37, 0.56))
    arrow(ax, (0.63, 0.50), (0.72, 0.51))
    ax.text(
        0.50,
        0.035,
        "Claim boundary: known-positive recovery is estimable; full-field precision and identity-safe recentering require additional truth.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MID,
    )
    return save(fig, "fig01_jnm_identity_contract.png")


def build_neuroinformatics_design() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "NeuRev Workbench evidence and provenance architecture",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    columns = [0.02, 0.275, 0.535, 0.795]
    widths = [0.19, 0.20, 0.20, 0.18]
    headers = ["Frozen inputs", "Identity-safe schema", "Bounded workflows", "Evidence outputs"]
    items = [
        ["Recording + hashes", "Sparse-positive labels", "Frozen event windows"],
        ["Observation site", "Geometry + trace key", "Canonical identity", "Candidate state P/U"],
        ["Grouped P/U evaluation", "Blinded review media", "Failure-stage audit", "Artifact validation"],
        ["Versioned run roots", "Evidence registry", "Figures + tables", "Venue manuscripts"],
    ]
    for col, (x, w, header, labels) in enumerate(zip(columns, widths, headers, items)):
        ax.text(x + w / 2, 0.89, header, ha="center", fontweight="bold", fontsize=10.5)
        start_y = 0.73
        spacing = 0.14 if len(labels) == 4 else 0.17
        for idx, label in enumerate(labels):
            face = BLUE_LIGHT if col in (0, 3) else PALE
            box(ax, (x, start_y - idx * spacing), (w, 0.095), label, face=face)
        if col < 3:
            arrow(ax, (x + w, 0.50), (columns[col + 1], 0.50))

    box(
        ax,
        (0.34, 0.075),
        (0.32, 0.105),
        "Human scientific-promotion gate\ncompletion is not claim authorization",
        face=GOLD_LIGHT,
        edge=GOLD,
        weight="bold",
    )
    arrow(ax, (0.70, 0.31), (0.60, 0.18))
    arrow(ax, (0.66, 0.13), (0.80, 0.31))
    ax.text(
        0.50,
        0.02,
        "Machine-readable facts and artifacts are synchronized; scientific interpretation remains author-reviewed.",
        ha="center",
        fontsize=9,
        color=MID,
    )
    return save(fig, "fig01_neuroinformatics_architecture.png")


def build_measurement_figure() -> Path:
    source_image = PAPER / "figures/final/fig03a_cs_parzen_hero_roi003_b03.png"
    trace_source = PAPER / "figures/final/fig04_population_trace_summary_source.tsv"
    image = Image.open(source_image).convert("RGB")
    crops = [
        image.crop((0, 112, 485, 288)),
        image.crop((0, 363, 485, 560)),
        image.crop((0, 615, 485, 808)),
    ]
    crop_titles = ["Raw fluorescence", "CS-Parzen temporal representation", "Local standardization"]

    rows = read_tsv(trace_source)
    sums: dict[tuple[str, float], list[float]] = defaultdict(list)
    site_time_values: dict[tuple[str, int, str, float], list[float]] = defaultdict(list)
    site_class: dict[str, int] = {}
    for row in rows:
        stage = row["stage"]
        time = float(row["relative_seconds"])
        value = float(row["shape_normalized"])
        class_id = int(row["trace_class_id"])
        site_id = row["site_id"]
        sums[(stage, time)].append(value)
        site_time_values[(stage, class_id, site_id, time)].append(value)
        if site_id in site_class and site_class[site_id] != class_id:
            raise ValueError(f"trace class changed within site: {site_id}")
        site_class[site_id] = class_id
    class_time_values: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (stage, class_id, _site_id, time), values in site_time_values.items():
        class_time_values[(stage, class_id, time)].append(float(np.mean(values)))
    class_counts = {
        class_id: sum(value == class_id for value in site_class.values())
        for class_id in (1, 2)
    }
    if class_counts != {1: 38, 2: 12}:
        raise ValueError(f"unexpected trace taxonomy counts: {class_counts}")
    stage_styles = {
        "Raw": (BLUE_DARK, "Raw"),
        "CS-Parzen ICA": (BLUE, "CS-Parzen temporal"),
        "Local standardization": (GOLD, "Local standardization"),
    }

    fig = plt.figure(figsize=(14.5, 9.0))
    outer = fig.add_gridspec(1, 2, width_ratios=(0.41, 0.59), wspace=0.20)
    image_grid = outer[0, 0].subgridspec(3, 1, hspace=0.22)
    trace_grid = outer[0, 1].subgridspec(2, 1, height_ratios=(0.61, 0.39), hspace=0.36)
    for index, (crop, title) in enumerate(zip(crops, crop_titles)):
        ax = fig.add_subplot(image_grid[index, 0])
        ax.imshow(crop)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")
        if index == 0:
            panel_label(ax, "A", x=-0.06, y=1.04)

    ax_trace = fig.add_subplot(trace_grid[0, 0])
    for stage, (color, label) in stage_styles.items():
        points = sorted((time, np.mean(values)) for (name, time), values in sums.items() if name == stage)
        ax_trace.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            linewidth=2.1,
            label=label,
        )
    ax_trace.axvline(0, color=MID, linestyle="--", linewidth=1.0)
    ax_trace.axhline(0, color=GRID, linewidth=0.8)
    ax_trace.set_title("Onset-aligned population trace morphology", loc="left", fontweight="bold")
    ax_trace.set_xlabel("Time from annotated onset (s)")
    ax_trace.set_ylabel("Mean shape-normalized signal")
    ax_trace.legend(frameon=False, loc="upper right")
    quiet_axes(ax_trace, axis="both")
    panel_label(ax_trace, "B", x=-0.07, y=1.035)
    ax_trace.text(
        0.02,
        0.02,
        "106 confirmed occurrences at 50 immutable sites\nShape comparison only; native amplitudes are not commensurate across representations.",
        transform=ax_trace.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        color=MID,
        bbox={"facecolor": WHITE, "edgecolor": GRID, "boxstyle": "round,pad=0.35", "alpha": 0.94},
    )

    class_grid = trace_grid[1, 0].subgridspec(1, 3, wspace=0.32)
    for index, (stage, (_color, title)) in enumerate(stage_styles.items()):
        ax_class = fig.add_subplot(class_grid[0, index])
        for class_id, class_color in ((1, BLUE), (2, GOLD)):
            points = sorted(
                (time, float(np.mean(values)))
                for (row_stage, row_class, time), values in class_time_values.items()
                if row_stage == stage and row_class == class_id
            )
            ax_class.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=class_color,
                linewidth=1.7,
                label=f"T{class_id} (n={class_counts[class_id]} sites)",
            )
        ax_class.axvline(0, color=MID, linestyle="--", linewidth=0.9)
        ax_class.axhline(0, color=GRID, linewidth=0.7)
        ax_class.set_title(title, fontsize=9, fontweight="bold")
        ax_class.set_xlabel("Time from onset (s)", fontsize=8)
        ax_class.tick_params(labelsize=7.5)
        if index == 0:
            ax_class.set_ylabel("Site-weighted\nmean shape", fontsize=8)
            ax_class.legend(frameon=False, fontsize=7.1, loc="upper right")
            panel_label(ax_class, "C", x=-0.26, y=1.07)
        quiet_axes(ax_class, axis="both")
    fig.suptitle(
        "Representative measurement views and shared event-aligned structure",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.70,
        0.005,
        "T1/T2 are exploratory within-recording measurement phenotypes, not neuronal cell types.",
        ha="center",
        fontsize=8.2,
        color=MID,
    )
    return save(fig, "fig02_measurement_representations.png")


def build_known_positive_figure() -> Path:
    temporal_path = PAPER / "generated_analysis/automated_feature_validation_v1/temporal_retrieval_summary.tsv"
    spatial_path = PAPER / "generated_analysis/automated_feature_validation_v1/spatial_displacement_summary.tsv"
    temporal = {row["feature_id"]: row for row in read_tsv(temporal_path)}
    spatial = {row["feature_id"]: row for row in read_tsv(spatial_path)}
    summary = json.loads(
        (PAPER / "generated_analysis/automated_feature_validation_v1/summary.json").read_text(encoding="utf-8")
    )
    recovery = summary["recovery_join"]

    fig = plt.figure(figsize=(15.0, 7.4))
    grid = fig.add_gridspec(1, 3, width_ratios=(0.72, 1.18, 1.45), wspace=0.47)

    ax_a = fig.add_subplot(grid[0, 0])
    strict_recovered = int(recovery["recovered_any"])
    strict_total = int(summary["population"]["occurrences"])
    canonical_total = int(recovery["canonical_sensitivity_occurrences"])
    canonical_row = next(
        row
        for row in summary["headline"]["recovery_models"]
        if row["population"] == "102_canonical_collapsed_sensitivity"
    )
    canonical_recovered = int(canonical_row["recovered"])
    labels = [
        f"Occurrence level\n{strict_recovered} / {strict_total}",
        f"Canonical-collapsed\n{canonical_recovered} / {canonical_total}",
    ]
    values = [strict_recovered / strict_total * 100, canonical_recovered / canonical_total * 100]
    bars = ax_a.barh(labels, values, color=[BLUE, GOLD], edgecolor=[BLUE_DARK, "#9B6B0D"], linewidth=0.9)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Known-positive sensitivity (%)")
    ax_a.set_title("Any-lane sensitivity at per-lane B58", loc="left", fontweight="bold")
    for bar, value in zip(bars, values):
        ax_a.text(value - 2.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", ha="right", va="center", color=WHITE, fontweight="bold")
    quiet_axes(ax_a, axis="x")
    panel_label(ax_a, "A", x=-0.18)

    ax_b = fig.add_subplot(grid[0, 1])
    selected_temporal = ["raw_center", "carrier_signed", "coherence_w15", "propagation_lag2_w15"]
    temporal_labels = ["Raw", "Carrier", "Coherence", "Lag-2 recurrence"]
    y = np.arange(len(selected_temporal))[::-1]
    for yy, feature, label in zip(y, selected_temporal, temporal_labels):
        row = temporal[feature]
        mean = float(row["mean_frame_auc"])
        low = float(row["frame_auc_ci95_low"])
        high = float(row["frame_auc_ci95_high"])
        ax_b.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt="o", color=BLUE, ecolor="#86AED6", capsize=3, markersize=6)
    ax_b.set_yticks(y, temporal_labels)
    ax_b.set_xlim(0.86, 0.97)
    ax_b.set_xlabel("Site-weighted frame ROC AUC")
    ax_b.set_title("Event versus guarded quiet frames", loc="left", fontweight="bold")
    ax_b.text(0.01, -0.15, "Focused x-axis; intervals use site bootstrap.", transform=ax_b.transAxes, fontsize=8, color=MID)
    quiet_axes(ax_b, axis="x")
    panel_label(ax_b, "B", x=-0.17)

    ax_c = fig.add_subplot(grid[0, 2])
    selected_spatial = [
        "raw_center",
        "carrier_signed",
        "coherence_w15",
        "propagation_lag2_w15",
        "representation_consensus",
        "multiscale_persistence",
    ]
    spatial_labels = ["Raw", "Carrier", "Coherence", "Lag-2 recurrence", "Representation consensus", "Multiscale persistence"]
    y2 = np.arange(len(selected_spatial))[::-1]
    for yy, feature in zip(y2, selected_spatial):
        row = spatial[feature]
        mean = float(row["mean_target_minus_displaced_auc"])
        low = float(row["site_bootstrap_ci95_low"])
        high = float(row["site_bootstrap_ci95_high"])
        ax_c.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt="o", color=GOLD, ecolor="#E7C571", capsize=3, markersize=6)
    ax_c.axvline(0, color=MID, linestyle="--", linewidth=1.0)
    ax_c.set_yticks(y2, spatial_labels)
    ax_c.set_xlim(-0.01, 0.29)
    ax_c.set_xlabel("Target minus displaced frame-AUC")
    ax_c.set_title("Same-field matched displacement", loc="left", fontweight="bold")
    quiet_axes(ax_c, axis="x")
    panel_label(ax_c, "C", x=-0.14)

    fig.suptitle("Known-positive recovery and localized measurement structure", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Quiet frames, displaced tissue, and unmatched proposals remain biologically unknown; none of these panels estimates full-field precision.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    return save(fig, "fig03_known_positive_evaluation.png")


def build_candidate_review_figure() -> Path:
    review_root = REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1"
    summary = json.loads((review_root / "review_summary_v1.json").read_text(encoding="utf-8"))
    counts = summary["counts"]
    examples = [
        ("NC003", "Definite neuron", BLUE, "NC003__new_candidate__still.png"),
        ("NC006", "Probable neuron; crescent / possible multiple", GOLD, "NC006__new_candidate__still.png"),
        ("NC013", "Uncertain; adjacent stronger source", ORANGE, "NC013__new_candidate__still.png"),
        ("NC008", "Unlikely / artifact-or-noise", MID, "NC008__new_candidate__still.png"),
    ]

    fig = plt.figure(figsize=(14.5, 9.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(0.72, 1.0, 1.0), hspace=0.34, wspace=0.12)
    ax_count = fig.add_subplot(grid[0, :])
    count_labels = ["Definite", "Probable", "Uncertain", "Unlikely / artifact-or-noise"]
    count_values = [counts["definite_neuron"], counts["probable_neuron"], counts["uncertain"], counts["artifact_or_noise"]]
    colors = [BLUE, GOLD, ORANGE, MID]
    bars = ax_count.bar(count_labels, count_values, color=colors, edgecolor=INK, linewidth=0.5)
    ax_count.set_ylim(0, 10.5)
    ax_count.set_ylabel("Sites")
    ax_count.set_title("Single-reviewer outcomes for all 18 selected unmatched sites", loc="left", fontweight="bold")
    for bar, value in zip(bars, count_values):
        ax_count.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", fontweight="bold")
    quiet_axes(ax_count, axis="y")
    panel_label(ax_count, "A", x=-0.04)

    for idx, (blind_id, label, color, filename) in enumerate(examples):
        ax = fig.add_subplot(grid[1 + idx // 2, idx % 2])
        image = Image.open(review_root / filename).convert("RGB")
        crop = image.crop((0, 42, image.width, 310))
        ax.imshow(crop)
        ax.axis("off")
        ax.set_title(f"{blind_id}: {label}", loc="left", fontsize=9.5, fontweight="bold", color=color, pad=4)
        panel_label(ax, chr(ord("B") + idx), x=-0.04, y=1.04)

    fig.suptitle("Blinded review broadens the provisional appearance envelope", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Flow: freeze detector-selected sites -> blind identifiers -> inspect six-panel media -> store normalized calls and attributes. "
        "Priority-score AUC = 0.308 (review value, not neuron probability).\n"
        "Definite + probable = 13/18: provisional single-reviewer yield, not detector precision or 13 validated new identities.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    return save(fig, "fig04_candidate_review.png")


def build_identity_miss_figure() -> Path:
    summary_path = PAPER / "generated_analysis/identity_aware_feature_inspection_v8/summary.json"
    offset_path = PAPER / "generated_analysis/identity_aware_feature_inspection_v8/offset_stability.tsv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    offsets = read_tsv(offset_path)
    contamination_path = REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_validation_suite_v2/identity_contamination.tsv"
    contamination = read_tsv(contamination_path)

    per_observation: dict[str, tuple[str, float]] = {}
    for row in offsets:
        per_observation[row["observation_id"]] = (row["v8_outcome"], float(row["mean_pairwise_direction_cosine"]))
    collision_cos = [value for outcome, value in per_observation.values() if outcome == "identity_collision"]
    clear_cos = [value for outcome, value in per_observation.values() if outcome == "identity_clear_miss"]
    identity_status = {
        row["observation_id"]: row["identity_status"]
        for row in offsets
        if row["v8_outcome"] == "identity_collision"
    }
    same_identity_count = sum(value == "same_canonical_identity_collision" for value in identity_status.values())
    other_identity_count = sum(value == "other_identity_collision" for value in identity_status.values())
    recovered_count = int(summary["population"]["recovered"])
    occurrence_count = int(summary["population"]["occurrences"])
    clear_count = int(summary["population"]["identity_clear_misses"])
    miss_count = occurrence_count - recovered_count

    fig = plt.figure(figsize=(14.5, 7.8))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)

    ax_a = fig.add_subplot(grid[0, :])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    box(ax_a, (0.01, 0.32), (0.18, 0.34), f"{occurrence_count} confirmed\noccurrences", face=BLUE_LIGHT, weight="bold", fontsize=11)
    box(ax_a, (0.29, 0.55), (0.18, 0.27), f"Recovered\n{recovered_count}", face=BLUE_LIGHT, weight="bold", fontsize=11)
    box(ax_a, (0.29, 0.12), (0.18, 0.27), f"All-lane per-burst\nB58 misses: {miss_count}", face=GOLD_LIGHT, edge=GOLD, weight="bold", fontsize=11)
    arrow(ax_a, (0.19, 0.53), (0.29, 0.68))
    arrow(ax_a, (0.19, 0.45), (0.29, 0.25))
    outcomes = [
        (f"Same canonical identity\ncollision: {same_identity_count}", 0.54, BLUE_LIGHT, BLUE),
        (f"Other labeled identity\ncollision: {other_identity_count}", 0.70, ORANGE_LIGHT, ORANGE),
        (f"Identity-clear\nhypothesis: {clear_count}", 0.86, GOLD_LIGHT, GOLD),
    ]
    for text_value, x, face, edge in outcomes:
        box(ax_a, (x, 0.12), (0.13, 0.27), text_value, face=face, edge=edge, fontsize=9, weight="bold")
        arrow(ax_a, (0.47, 0.255), (x, 0.255))
    ax_a.set_title("Identity guard changes the interpretation of local response gains", loc="left", fontweight="bold")
    panel_label(ax_a, "A", x=-0.025, y=1.02)

    ax_b = fig.add_subplot(grid[1, 0])
    rng = np.random.default_rng(20260907)
    groups = [("Collision", collision_cos, ORANGE), ("Identity-clear", clear_cos, GOLD)]
    for index, (_, values, color) in enumerate(groups):
        jitter = rng.uniform(-0.07, 0.07, len(values))
        ax_b.scatter(np.full(len(values), index) + jitter, values, s=42, color=color, edgecolor=INK, linewidth=0.35, zorder=3)
        median = float(np.median(values))
        ax_b.plot([index - 0.18, index + 0.18], [median, median], color=INK, linewidth=2.0)
        ax_b.text(index, median + 0.07, f"median {median:.3f}", ha="center", fontsize=8)
    ax_b.set_xticks([0, 1], ["Collision\n(n=8)", "Identity-clear\n(n=4)"])
    ax_b.set_ylabel("Cross-stage direction cosine")
    ax_b.set_ylim(-0.5, 1.18)
    ax_b.set_title("Response-maximizing shift agreement", loc="left", fontweight="bold")
    quiet_axes(ax_b, axis="y")
    panel_label(ax_b, "B", x=-0.10)

    ax_c = fig.add_subplot(grid[1, 1])
    labels = ["Neighbor partial r", "Multi-neighbor variance explained"]
    collision_r2 = float(np.median([float(row["multineighbor_r2"]) for row in contamination if row["outcome"] == "collision"]))
    clear_r2 = float(np.median([float(row["multineighbor_r2"]) for row in contamination if row["outcome"] == "clear"]))
    collision = [summary["headline"]["ls_neighbor_partial_r_median_collision"], collision_r2]
    clear = [summary["headline"]["ls_neighbor_partial_r_median_clear"], clear_r2]
    y = np.arange(len(labels))[::-1]
    ax_c.scatter(collision, y + 0.10, s=70, color=ORANGE, edgecolor=INK, linewidth=0.4, label="Collision")
    ax_c.scatter(clear, y - 0.10, s=70, facecolor=GOLD_LIGHT, edgecolor=GOLD, linewidth=1.4, label="Identity-clear")
    for yy, left, right in zip(y, clear, collision):
        ax_c.plot([left, right], [yy, yy], color=GRID, linewidth=2, zorder=0)
    ax_c.set_yticks(y, labels)
    ax_c.set_xlim(0, 0.62)
    ax_c.set_xlabel("Median association or fraction")
    ax_c.set_title("Neighbor coupling is stronger for collisions", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, loc="lower right")
    quiet_axes(ax_c, axis="x")
    panel_label(ax_c, "C", x=-0.10)
    ax_c.text(
        0.98,
        0.97,
        "0/4 identity-clear cases had a frozen\ncandidate within 8 px through per-burst B100",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": WHITE, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    fig.suptitle("Identity-aware decomposition of the 12 all-lane misses", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        "The eight-versus-four comparisons are descriptive; response-selected shifts are not prospective recovery evidence.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    return save(fig, "fig05_identity_aware_misses.png")


def build_feature_atlas_figure() -> Path:
    atlas = REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e"
    metrics = {row["model_id"]: row for row in read_tsv(atlas / "tables/model_metrics.tsv")}
    lobo = read_tsv(atlas / "tables/leave_one_burst_out.tsv")
    families = json.loads((atlas / "tables/family_paired_group_bootstraps.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(15.0, 9.2))
    grid = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.40)

    ax_a = fig.add_subplot(grid[0, 0])
    selected = ["carrier_signed", "retrained_existing_linear", "augmented_linear", "existing_plus_nuisance_competition"]
    names = ["Carrier only", "Existing 12-feature", "Existing + all Atlas", "Existing + nuisance/competition"]
    y = np.arange(len(selected))[::-1]
    auc_values = [float(metrics[key]["macro_fold_spu_auc"]) for key in selected]
    ax_a.scatter(auc_values, y, s=70, color=[MID, BLUE, BLUE, GOLD], edgecolor=INK, linewidth=0.4)
    for yy, value in zip(y, auc_values):
        ax_a.text(value + 0.0012, yy, f"{value:.4f}", va="center", fontsize=8)
    ax_a.set_yticks(y, names)
    ax_a.set_xlim(0.92, 0.995)
    ax_a.set_xlabel("Macro held-fold SPU-AUC")
    ax_a.set_title("Model comparison on the same 1,619 candidates", loc="left", fontweight="bold")
    ax_a.text(0.01, -0.16, "Focused x-axis; 78 P anchors and 1,541 U candidates.", transform=ax_a.transAxes, fontsize=8, color=MID)
    quiet_axes(ax_a, axis="x")
    panel_label(ax_a, "A", x=-0.10)

    ax_b = fig.add_subplot(grid[0, 1])
    family_map = {
        "augmented_linear": "All Atlas",
        "existing_plus_temporal_envelope": "Temporal envelope",
        "existing_plus_soma_morphology": "Soma morphology",
        "existing_plus_map_source_consistency": "Map/source consistency",
        "existing_plus_nuisance_competition": "Nuisance/competition",
    }
    family_rows = [row for row in families if row["score_a_name"] in family_map]
    family_rows.sort(key=lambda row: list(family_map).index(row["score_a_name"]))
    y2 = np.arange(len(family_rows))[::-1]
    for yy, row in zip(y2, family_rows):
        mean = float(row["observed_delta_a_minus_b"])
        low = float(row["bootstrap_delta_ci95_low"])
        high = float(row["bootstrap_delta_ci95_high"])
        color = GOLD if row["score_a_name"] == "existing_plus_nuisance_competition" else BLUE
        ax_b.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt="o", color=color, ecolor=color, alpha=0.9, capsize=3, markersize=6)
    ax_b.axvline(0, color=MID, linestyle="--", linewidth=1.0)
    ax_b.set_yticks(y2, [family_map[row["score_a_name"]] for row in family_rows])
    ax_b.set_xlabel("Global SPU-AUC delta vs existing model")
    ax_b.set_title("Grouped paired intervals", loc="left", fontweight="bold")
    quiet_axes(ax_b, axis="x")
    panel_label(ax_b, "B", x=-0.10)

    ax_c = fig.add_subplot(grid[1, 0])
    by_key = {(row["model_id"], int(row["heldout_burst"])): float(row["spu_auc"]) for row in lobo}
    bursts = np.array([1, 2, 3, 4])
    deltas = np.array([by_key[("augmented_linear", burst)] - by_key[("retrained_existing_linear", burst)] for burst in bursts])
    ax_c.bar(bursts, deltas, color=BLUE, edgecolor=BLUE_DARK, linewidth=0.7)
    for burst, value in zip(bursts, deltas):
        ax_c.text(burst, value + 0.00045, f"{value:+.4f}", ha="center", fontsize=8)
    ax_c.axhline(0, color=MID, linewidth=0.8)
    ax_c.set_xticks(bursts)
    ax_c.set_xlabel("Held-out burst")
    ax_c.set_ylabel("Augmented minus existing SPU-AUC")
    ax_c.set_ylim(0, max(deltas) * 1.28)
    ax_c.set_title("Leave-one-burst-out direction", loc="left", fontweight="bold")
    quiet_axes(ax_c, axis="y")
    panel_label(ax_c, "C", x=-0.10)

    ax_d = fig.add_subplot(grid[1, 1])
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    rows = [
        ("Temporal envelope", "No additive gain", MID),
        ("Soma morphology", "Useful alone; redundant", MID),
        ("Map/source consistency", "Quality control / abstention", BLUE),
        ("Local competition", "Confirm with identity-safe resolver", GOLD),
    ]
    ax_d.text(0.00, 0.94, "Feature-role decision", fontweight="bold", fontsize=11)
    for index, (family, decision, color) in enumerate(rows):
        yrow = 0.76 - index * 0.19
        ax_d.add_patch(FancyBboxPatch((0.00, yrow), 0.98, 0.13, boxstyle="round,pad=0.012", facecolor=PALE, edgecolor=GRID, linewidth=0.8))
        ax_d.text(0.03, yrow + 0.065, family, va="center", fontweight="bold", fontsize=8.5)
        ax_d.text(0.97, yrow + 0.065, decision, va="center", ha="right", color=color, fontsize=8.5)
    ax_d.text(0.02, 0.02, "Promotion gate: NOT PASSED\nAll primary grouped intervals include zero.", fontsize=9, color=ORANGE, fontweight="bold", va="bottom")
    panel_label(ax_d, "D", x=-0.10)

    fig.suptitle("Presubmission audit only: Feature Atlas v1 is not claim-bearing", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Candidate reranking within one recording; U is unlabeled, not negative. The result does not establish precision or end-to-end detector replacement.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    return save(fig, "figS1_feature_atlas_presubmission_audit.png")


def build_exact_truth_figure() -> Path:
    movie_path = REPO / "Outputs/NeuronIdentifiability/realistic_movie_detector_benchmark_v3/summary.json"
    holdout_path = REPO / "Outputs/NeuronIdentifiability/realistic_movie_generator_family_holdout_v4/summary.json"
    challenge_path = REPO / "Outputs/NeuronIdentifiability/automated_challenge_suite_v7/summary.json"
    movie = json.loads(movie_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(15.0, 8.9))
    grid = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.34)

    model_ids = ["carrier", "spatial_context", "kinetic", "combined"]
    model_labels = ["Carrier", "Spatial context", "Kinetic", "Combined"]
    colors = [MID, BLUE, GOLD, BLUE_DARK]

    ax_a = fig.add_subplot(grid[0, 0])
    values = [float(movie["operating_points"][model]["4"]["f1"]) for model in model_ids]
    bars = ax_a.bar(model_labels, values, color=colors, edgecolor=INK, linewidth=0.5)
    for bar, value in zip(bars, values):
        ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    ax_a.set_ylim(0, 0.56)
    ax_a.set_ylabel("Identity-matched F1")
    ax_a.set_title("432 exact-truth movies, budget 4", loc="left", fontweight="bold")
    quiet_axes(ax_a, axis="y")
    panel_label(ax_a, "A", x=-0.10)

    ax_b = fig.add_subplot(grid[0, 1])
    families = ["baseline", "dense_neuropil", "bleaching", "nonrigid_motion", "empirical_noise", "combined_shift"]
    family_labels = ["Baseline", "Dense\nneuropil", "Bleaching", "Nonrigid\nmotion", "Empirical\nnoise", "Compound\nshift"]
    x = np.arange(len(families))
    width = 0.36
    spatial = [float(holdout["family_operating_points"][family]["spatial_context"]["f1"]) for family in families]
    combined = [float(holdout["family_operating_points"][family]["combined"]["f1"]) for family in families]
    ax_b.bar(x - width / 2, spatial, width, color=BLUE, label="Spatial context", edgecolor=INK, linewidth=0.4)
    ax_b.bar(x + width / 2, combined, width, color=BLUE_DARK, label="Combined", edgecolor=INK, linewidth=0.4)
    ax_b.set_xticks(x, family_labels)
    ax_b.set_ylim(0, 0.72)
    ax_b.set_ylabel("Identity-matched F1")
    ax_b.set_title("108 held-generator-family movies", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=8, loc="upper right")
    quiet_axes(ax_b, axis="y")
    panel_label(ax_b, "B", x=-0.10)

    ax_c = fig.add_subplot(grid[1, 0])
    morphology = challenge["summaries"]["morphology_interventions"]
    morph_ids = ["ellipse", "crescent", "ring", "fragmented"]
    morph_labels = ["Ellipse", "Crescent", "Ring", "Fragmented"]
    morph_values = [float(morphology[name]) for name in morph_ids]
    morph_colors = [BLUE, ORANGE, GOLD, BLUE_LIGHT]
    bars = ax_c.bar(morph_labels, morph_values, color=morph_colors, edgecolor=INK, linewidth=0.5)
    for bar, value in zip(bars, morph_values):
        ax_c.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    ax_c.set_ylim(0, 1.08)
    ax_c.set_ylabel("Spatial-context pixel AUC")
    ax_c.set_title("Prespecified morphology intervention", loc="left", fontweight="bold")
    quiet_axes(ax_c, axis="y")
    panel_label(ax_c, "C", x=-0.10)

    ax_d = fig.add_subplot(grid[1, 1])
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    ax_d.text(0.00, 0.94, "What exact truth resolved", fontweight="bold", fontsize=11)
    rows = [
        ("Compact-budget recovery", "Combined led in the 432-movie benchmark", BLUE_DARK),
        ("Mechanism-shift robustness", "Spatial context led all six held families", BLUE),
        ("Close-source identity", "Weak neighbors remained a failure boundary", ORANGE),
        ("Deployable stopping", "Source-count-free stopping remained inadequate", ORANGE),
    ]
    for index, (topic, decision, color) in enumerate(rows):
        yrow = 0.76 - index * 0.19
        ax_d.add_patch(FancyBboxPatch((0.00, yrow), 0.98, 0.13, boxstyle="round,pad=0.012", facecolor=PALE, edgecolor=GRID, linewidth=0.8))
        ax_d.text(0.03, yrow + 0.083, topic, va="center", fontweight="bold", fontsize=8.5)
        ax_d.text(0.03, yrow + 0.040, decision, va="center", color=color, fontsize=8.2)
    ax_d.text(0.02, 0.02, "Evidence tier: computational simulation\nNot biological transfer or full-field precision.", fontsize=9, color=ORANGE, fontweight="bold", va="bottom")
    panel_label(ax_d, "D", x=-0.10)

    fig.suptitle("Exact-truth simulation supports roles and exposes failure boundaries", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Simulation protocols are independent of the one-recording observational estimands and are reported as mechanistic support only.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    return save(fig, "fig06_exact_truth_validation.png")


def main() -> int:
    sources = [
        PAPER / "figures/final/fig03a_cs_parzen_hero_roi003_b03.png",
        PAPER / "figures/final/fig04_population_trace_summary_source.tsv",
        PAPER / "figures/final/fig04_trace_taxonomy_experiment.json",
        REPO / "docs/research/SPON_CA_BURST_MULTILAG_MSICA_V5_RESULTS.md",
        REPO / "research/registry/index.yaml",
        REPO / "research/generated/canonical.json",
        REPO / "neurobench/research/registry.py",
        PAPER / "generated_analysis/automated_feature_validation_v1/temporal_retrieval_summary.tsv",
        PAPER / "generated_analysis/automated_feature_validation_v1/spatial_displacement_summary.tsv",
        PAPER / "generated_analysis/automated_feature_validation_v1/summary.json",
        REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/review_summary_v1.json",
        REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/user_review_v1.tsv",
        REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/NC003__new_candidate__still.png",
        REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/NC006__new_candidate__still.png",
        REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/NC013__new_candidate__still.png",
        REPO / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/NC008__new_candidate__still.png",
        PAPER / "generated_analysis/identity_aware_feature_inspection_v8/summary.json",
        PAPER / "generated_analysis/identity_aware_feature_inspection_v8/offset_stability.tsv",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_validation_suite_v2/identity_contamination.tsv",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_validation_suite_v2/budget_nms_counterfactual.tsv",
        REPO / "Outputs/NeuronIdentifiability/realistic_movie_detector_benchmark_v3/summary.json",
        REPO / "Outputs/NeuronIdentifiability/realistic_movie_generator_family_holdout_v4/summary.json",
        REPO / "Outputs/NeuronIdentifiability/automated_challenge_suite_v7/summary.json",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/tables/model_metrics.tsv",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/tables/leave_one_burst_out.tsv",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/tables/family_paired_group_bootstraps.json",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/tables/univariate_feature_metrics.tsv",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/tables/new_feature_spearman.tsv",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/summary.json",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/validation.json",
        REPO / "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e/status.json",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen figure sources:\n" + "\n".join(missing))

    outputs = [
        build_jnm_design(),
        build_neuroinformatics_design(),
        build_measurement_figure(),
        build_known_positive_figure(),
        build_candidate_review_figure(),
        build_identity_miss_figure(),
        build_exact_truth_figure(),
        build_feature_atlas_figure(),
    ]
    manifest = {
        "schema_version": 2,
        "purpose": "venue manuscript figure presentation derivatives",
        "scientific_boundary": "No model refit, cohort change, new label, or new estimand.",
        "sources": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "availability": (
                    "local_metadata_only"
                    if path.relative_to(REPO).parts[0] == "Outputs"
                    else "repository"
                ),
            }
            for path in sources
        ],
        "outputs": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256(path)} for path in outputs
        ],
    }
    manifest_path = OUT / "venue_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for output in outputs:
        print(f"{sha256(output)}  {output}")
    print(f"{sha256(manifest_path)}  {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
