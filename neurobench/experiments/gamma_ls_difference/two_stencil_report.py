"""Descriptive reporting of the complete, predeclared two-stencil campaign.

Heavy plotting/array dependencies are imported only inside the report functions.
The report does not select cells, retune thresholds, or change the frozen labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TARGETS = (.25, .5, 1., 2., 5.)
STAGES = ("Raw", "X", "A", "M", "sigma", "contrast", "Z")
INPUTS = ("conditioned_current_frame", "difference_signed")
METRICS = (
    "known_positive_occurrence_window_recall",
    "matched_known_positive_count",
    "emitted_frame_proposal_count",
    "application_proposals_per_frame",
    "spatial_representative_count_in_declared_windows",
)
DISPLAY_QUANTILE = .995
MAX_SAMPLES_PER_SNAPSHOT = 8192


def _true(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("a complete comparison table must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def normalize_results(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reject incomplete/duplicated grids and reconcile sparse-positive counts."""
    cells = {str(cell["cell_id"]): cell for cell in protocol["cells"]}
    if len(cells) != 25 or len(protocol["cells"]) != 25:
        raise ValueError("report requires the full 25-cell protocol")
    if tuple(float(q) for q in protocol["target_proposals_per_frame"]) != TARGETS:
        raise ValueError("unexpected target proposals/frame grid")
    expected = set()
    for cell_id, cell in cells.items():
        spec = cell["spec"]
        controls = _true(cell["is_deployed_anchor"]) or (
            spec["design"] in {"direct", "serial"}
            and spec["reference_family"] == "gamma" and float(spec["guard_radius_px"]) == 0
        )
        for readout in (("Z", "A", "contrast") if controls else ("Z",)):
            expected.update((cell_id, readout, q) for q in TARGETS)
    if len(expected) != 175:
        raise ValueError("protocol does not declare 125 Z rows and 50 fixed readout-control rows")
    seen = set()
    normalized = []
    for original in rows:
        row = dict(original)
        key = (str(row["operator_cell_id"]), str(row["readout"]),
               _number(row["target_proposals_per_frame"], "target"))
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicate cell/readout/target row")
        seen.add(key)
        cell = cells[key[0]]
        for field in ("design", "reference_family", "support_geometry"):
            if str(row[field]) != str(cell["spec"][field]):
                raise ValueError(f"result disagrees with protocol: {field}")
        if str(row["input_representation"]) != str(cell["input_representation"]):
            raise ValueError("result input disagrees with protocol")
        if float(row["guard_radius_px"]) != float(cell["spec"]["guard_radius_px"]):
            raise ValueError("result guard disagrees with protocol")
        if _true(row["is_deployed_anchor"]) != _true(cell["is_deployed_anchor"]):
            raise ValueError("result anchor role disagrees with protocol")
        for field in METRICS:
            row[field] = _number(row[field], field)
        for field in ("known_positive_count", "application_score_frame_count"):
            row[field] = _number(row[field], field)
        if row["known_positive_count"] != 79 or row["application_score_frame_count"] != 2259:
            raise ValueError("report requires 79 sparse positives and 2259 application score frames")
        matched = row["matched_known_positive_count"]
        proposals = row["emitted_frame_proposal_count"]
        sites = row["spatial_representative_count_in_declared_windows"]
        if any(value < 0 or value != math.floor(value) for value in (matched, proposals, sites)):
            raise ValueError("reported counts must be nonnegative integers")
        if matched > min(79, sites) or sites > proposals:
            raise ValueError("inconsistent proposal/site/match counts")
        if not math.isclose(row[METRICS[0]], matched / 79, rel_tol=0, abs_tol=1e-12):
            raise ValueError("known-positive recall does not equal matched/79")
        if not math.isclose(row["application_proposals_per_frame"], proposals / 2259, rel_tol=0, abs_tol=1e-12):
            raise ValueError("application rate does not reconcile with emitted proposals")
        row.update(operator_cell_id=key[0], readout=key[1], target_proposals_per_frame=key[2],
                   guard_radius_px=float(row["guard_radius_px"]),
                   is_deployed_anchor=_true(row["is_deployed_anchor"]))
        normalized.append(row)
    if seen != expected:
        raise ValueError(f"complete report requires all175 rows; found {len(seen)}")
    order = {cell_id: index for index, cell_id in enumerate(cells)}
    return sorted(normalized, key=lambda row: (
        order[row["operator_cell_id"]], ("Z", "A", "contrast").index(row["readout"]),
        row["target_proposals_per_frame"],
    ))


def _contrast(left: Mapping[str, Any], right: Mapping[str, Any], comparison: str) -> dict[str, Any]:
    row = {
        "comparison": comparison,
        "left_cell_id": left["operator_cell_id"], "right_cell_id": right["operator_cell_id"],
        "left_readout": left["readout"], "right_readout": right["readout"],
        "input_representation": left["input_representation"],
        "target_proposals_per_frame": left["target_proposals_per_frame"],
        "difference_direction": "left_minus_right",
        "scope": "paired_descriptive_same_recording_no_inferential_interval",
    }
    for metric in METRICS:
        row[f"left_{metric}"] = float(left[metric])
        row[f"right_{metric}"] = float(right[metric])
        row[f"delta_{metric}"] = float(left[metric]) - float(right[metric])
    return row


def factorial_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """All200 prespecified Z contrasts, excluding the separate disk anchor."""
    index = {}
    for row in rows:
        if row["readout"] == "Z" and not _true(row["is_deployed_anchor"]):
            key = (row["input_representation"], row["design"], row["reference_family"],
                   float(row["guard_radius_px"]), float(row["target_proposals_per_frame"]))
            if key in index:
                raise ValueError("duplicate factorial dimension")
            index[key] = row
    result = []
    for representation in INPUTS:
        for q in TARGETS:
            for family in ("uniform", "gamma"):
                for guard in (0., 7.):
                    for left, right in (("direct", "point"), ("serial", "direct")):
                        result.append(_contrast(index[(representation, left, family, guard, q)],
                                                index[(representation, right, family, guard, q)],
                                                f"{left}_minus_{right}"))
            for design in ("point", "direct", "serial"):
                for guard in (0., 7.):
                    result.append(_contrast(index[(representation, design, "gamma", guard, q)],
                                            index[(representation, design, "uniform", guard, q)],
                                            "gamma_minus_uniform"))
                for family in ("uniform", "gamma"):
                    result.append(_contrast(index[(representation, design, family, 7., q)],
                                            index[(representation, design, family, 0., q)],
                                            "guard7_minus_guard0"))
    return result


def readout_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """All75 paired contrasts for the five predeclared A/contrast/Z controls."""
    index = {(row["operator_cell_id"], row["readout"], float(row["target_proposals_per_frame"])): row for row in rows}
    controls = sorted({row["operator_cell_id"] for row in rows if row["readout"] == "A"})
    result = []
    for cell in controls:
        for q in TARGETS:
            for left, right in (("contrast", "A"), ("Z", "contrast"), ("Z", "A")):
                result.append(_contrast(index[(cell, left, q)], index[(cell, right, q)],
                                        f"{left}_minus_{right}"))
    return result


def signed_display_stage(representation: str, stage: str) -> bool:
    if representation not in INPUTS or stage not in STAGES:
        raise ValueError("undeclared representation or stage")
    return stage in {"contrast", "Z"} or (
        representation == "difference_signed" and stage in {"X", "A", "M"}
    )


def fixed_center_crop(height: int, width: int, side: int = 128) -> tuple[int, int, int, int]:
    """Return x0,x1,y0,y1, using a fixed central crop independent of annotations."""
    if min(height, width) < side or side < 1:
        raise ValueError("source image is too small for the declared central crop")
    x0, y0 = (width - side) // 2, (height - side) // 2
    return x0, x0 + side, y0, y0 + side


def _snapshot_sample(values: Any) -> Any:
    import numpy as np

    flat = np.asarray(values).reshape(-1)
    if not len(flat) or not np.isfinite(flat).all():
        raise ValueError("display snapshots must be nonempty and finite")
    indices = np.linspace(0, len(flat) - 1, min(MAX_SAMPLES_PER_SNAPSHOT, len(flat)), dtype=np.int64)
    return np.asarray(flat[indices], dtype=np.float32)


def display_limits(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Pool label-independent predeclared frames; outcomes do not select displays."""
    import numpy as np

    frames = tuple(int(value) for value in protocol["snapshot_source_ui"])
    positions = np.asarray(frames) - 1
    samples: dict[tuple[str, str], list[Any]] = defaultdict(list)
    source = np.load(protocol["source_movie"]["path"], mmap_mode="r")
    for representation in INPUTS:
        input_video = np.load(protocol["inputs"][representation]["path"], mmap_mode="r")
        for index in positions:
            samples[(representation, "Raw")].append(_snapshot_sample(source[index]))
            samples[(representation, "X")].append(_snapshot_sample(input_video[index]))
    cell_counts = defaultdict(int)
    for cell in protocol["cells"]:
        representation = cell["input_representation"]
        cell_counts[representation] += 1
        with np.load(root / "cells" / cell["cell_id"] / "stage_snapshots.npz") as snapshot:
            if tuple(int(x) for x in snapshot["source_frames_ui"]) != frames:
                raise ValueError("cell snapshots differ from the predeclared source frames")
            for stage in STAGES[2:]:
                values = snapshot[stage]
                if len(values) != len(frames) or tuple(values.shape[1:]) != tuple(source.shape[1:]):
                    raise ValueError("stage snapshot shape does not match source frames/image")
                for frame in values:
                    samples[(representation, stage)].append(_snapshot_sample(frame))
    limits = {}
    for representation in INPUTS:
        limits[representation] = {}
        for stage in STAGES:
            pooled = np.concatenate(samples[(representation, stage)])
            signed = signed_display_stage(representation, stage)
            if signed:
                maximum = max(float(np.quantile(np.abs(pooled), DISPLAY_QUANTILE)), 1e-12)
                minimum = -maximum
            else:
                minimum, maximum = np.quantile(pooled, [1.0 - DISPLAY_QUANTILE, DISPLAY_QUANTILE])
                minimum = max(0.0, float(minimum))
                maximum = max(float(maximum), minimum + 1e-12)
            limits[representation][stage] = {
                "vmin": float(minimum), "vmax": float(maximum), "signed": signed,
                "colormap": "gray", "shared_across_all_cells_of_input": True,
                "cell_count": cell_counts[representation] if stage not in {"Raw", "X"} else 1,
                "pooled_sample_count": len(pooled),
            }
    return {
        "source_frames_ui": list(frames), "limits": limits,
        "method": "deterministic_equal_pixel_ordinal_samples_per_predeclared_snapshot",
        "quantile": DISPLAY_QUANTILE, "max_samples_per_snapshot": MAX_SAMPLES_PER_SNAPSHOT,
        "signed_rule": "symmetric_abs_quantile_with_zero_at_midgray",
        "nonnegative_rule": "pooled_lower_upper_quantiles_clamped_to_nonnegative",
        "annotation_coordinates_used_for_display_selection": False,
        "fullfield_and_crop_share_identical_limits": True,
    }


def _plot_modules():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    return plt, np


def render_stage_comparisons(root: Path, destination: Path, protocol: Mapping[str, Any], limits: Mapping[str, Any]) -> list[dict[str, Any]]:
    """All fixed frames, full field and central128 crop, for six core arms + anchor."""
    plt, np = _plot_modules()
    destination.mkdir(parents=True, exist_ok=True)
    raw = np.load(protocol["source_movie"]["path"], mmap_mode="r")
    x0, x1, y0, y1 = fixed_center_crop(*raw.shape[1:])
    frames = tuple(protocol["snapshot_source_ui"])
    groups = []
    for representation in INPUTS:
        cells = [cell for design in ("point", "direct", "serial") for cell in protocol["cells"]
                 if cell["input_representation"] == representation
                 and not cell["is_deployed_anchor"] and cell["spec"]["design"] == design
                 and cell["spec"]["reference_family"] == "gamma"
                 and float(cell["spec"]["guard_radius_px"]) == 0]
        if len(cells) != 3:
            raise ValueError("missing predeclared point/direct/serial Gamma guard0 core arms")
        groups.append((representation + "__core_gamma_g0", representation, cells))
    anchors = [cell for cell in protocol["cells"] if cell["is_deployed_anchor"]]
    if len(anchors) != 1:
        raise ValueError("expected exactly one separate deployed anchor")
    groups.append(("deployed_anchor_separate", anchors[0]["input_representation"], anchors))
    index = []
    for group_id, representation, cells in groups:
        source = np.load(protocol["inputs"][representation]["path"], mmap_mode="r")
        arrays = {}
        for cell in cells:
            with np.load(root / "cells" / cell["cell_id"] / "stage_snapshots.npz") as snapshot:
                arrays[cell["cell_id"]] = {stage: snapshot[stage] for stage in STAGES[2:]}
        for position, frame_ui in enumerate(frames):
            for view in ("fullfield", "center128"):
                crop = (slice(None), slice(None)) if view == "fullfield" else (slice(y0, y1), slice(x0, x1))
                fig, axes = plt.subplots(len(cells), len(STAGES), figsize=(18.2, 2.8 * len(cells)), squeeze=False)
                for row_index, cell in enumerate(cells):
                    for column, stage in enumerate(STAGES):
                        values = raw[frame_ui - 1] if stage == "Raw" else (
                            source[frame_ui - 1] if stage == "X" else arrays[cell["cell_id"]][stage][position])
                        settings = limits["limits"][representation][stage]
                        ax = axes[row_index, column]
                        ax.imshow(values[crop], cmap="gray", vmin=settings["vmin"], vmax=settings["vmax"], interpolation="nearest")
                        if row_index == 0:
                            ax.set_title(f"{stage}\n[{settings['vmin']:.3g}, {settings['vmax']:.3g}]", fontsize=9)
                        if column == 0:
                            ax.set_ylabel("anchor: point Gamma g7 disk" if cell["is_deployed_anchor"] else cell["spec"]["design"], fontsize=9)
                        ax.set_xticks([])
                        ax.set_yticks([])
                scope = "full field" if view == "fullfield" else f"fixed crop x=[{x0},{x1}), y=[{y0},{y1})"
                fig.suptitle(f"{representation} | source UI {frame_ui} | {scope}\nShared stage scales from all declared cells and fixed snapshots; no annotation-selected crop", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, .91 if len(cells) == 1 else .94))
                path = destination / f"{group_id}__ui{frame_ui:04d}__{view}.png"
                fig.savefig(path, dpi=140)
                plt.close(fig)
                index.append({
                    "path": str(path.relative_to(root)), "group": group_id,
                    "source_frame_ui": frame_ui, "view": view,
                    "input_representation": representation,
                    "operator_cell_ids": [cell["cell_id"] for cell in cells],
                    "crop_xyxy": [0, raw.shape[2], 0, raw.shape[1]] if view == "fullfield" else [x0, x1, y0, y1],
                    "stages": list(STAGES), "shared_display_limits": "display_limits.json",
                    "annotation_overlays": False,
                })
    return index


def render_result_matrices(rows: Sequence[Mapping[str, Any]], destination: Path, protocol: Mapping[str, Any]) -> list[str]:
    plt, np = _plot_modules()
    destination.mkdir(parents=True, exist_ok=True)
    cells = protocol["cells"]
    lookup = {(row["operator_cell_id"], row["readout"], row["target_proposals_per_frame"]): row for row in rows}
    labels = [f"{'current' if cell['input_representation'] == INPUTS[0] else 'difference'} | {cell['spec']['design']} | {cell['spec']['reference_family']} | g{cell['spec']['guard_radius_px']:g} | {cell['spec']['support_geometry']}" for cell in cells]
    files = []
    for metric, label, filename in (
        (METRICS[0], "Known-positive occurrence-window recall", "all25_Z_recall.png"),
        ("emitted_frame_proposal_count", "Emitted frame-level proposals", "all25_Z_proposals.png"),
    ):
        values = np.asarray([[float(lookup[(cell["cell_id"], "Z", q)][metric]) for q in TARGETS] for cell in cells])
        fig, ax = plt.subplots(figsize=(13, 11))
        maximum = 1.0 if metric == METRICS[0] else max(float(values.max()), 1.)
        image = ax.imshow(values, cmap="Greys", vmin=0, vmax=maximum, aspect="auto")
        for y in range(len(cells)):
            for x in range(len(TARGETS)):
                text = f"{round(values[y,x]*79):d}/79" if metric == METRICS[0] else f"{int(values[y,x]):,}"
                ax.text(x, y, text, ha="center", va="center", fontsize=8, color="white" if values[y,x] > maximum*.55 else "black")
        ax.set_yticks(range(len(cells)), labels, fontsize=8)
        ax.set_xticks(range(len(TARGETS)), [str(q) for q in TARGETS])
        ax.set_xlabel("Target NMS proposals per calibration score frame")
        ax.set_title(label + " | Z readout | all24 factorial cells + separate disk anchor")
        fig.colorbar(image, ax=ax, shrink=.55, label=label)
        fig.tight_layout()
        path = destination / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files.append(str(path))
    controls = [cell for cell in cells if (cell["cell_id"], "A", TARGETS[0]) in lookup]
    fig, axes = plt.subplots(len(controls), 2, figsize=(12, 3*len(controls)), squeeze=False)
    max_rate = max(float(row["application_proposals_per_frame"]) for row in rows if row["operator_cell_id"] in {cell["cell_id"] for cell in controls})
    for index, cell in enumerate(controls):
        for readout, gray, style in (("A", ".6", "-"), ("contrast", ".3", "--"), ("Z", "black", ":")):
            paired = [lookup[(cell["cell_id"], readout, q)] for q in TARGETS]
            for column, metric in enumerate((METRICS[0], "application_proposals_per_frame")):
                axes[index,column].plot(TARGETS, [row[metric] for row in paired], marker="o", color=gray, linestyle=style, label=readout)
        for column in (0, 1):
            axes[index,column].set_title(cell["cell_id"], fontsize=8)
            axes[index,column].set_xlabel("Target proposals/calibration score frame")
            axes[index,column].grid(alpha=.2)
            axes[index,column].legend(fontsize=8)
        axes[index,0].set_ylim(0, 1)
        axes[index,0].set_ylabel("Known-positive occurrence-window recall")
        axes[index,1].set_ylim(0, max(max_rate*1.05, .01))
        axes[index,1].set_ylabel("Emitted proposals/application frame")
    fig.suptitle("Predeclared A, contrast, and Z controls | thresholds calibrated separately on the same99 frames", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .98))
    path = destination / "predeclared_readout_controls.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(str(path))
    # Compare realized application burden directly, without retuning on it.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for index, representation in enumerate(INPUTS):
        selected_cells = [cell for cell in cells if cell["input_representation"] == representation
            and (cell["is_deployed_anchor"] or (cell["spec"]["reference_family"] == "gamma" and cell["spec"]["guard_radius_px"] == 0))]
        q1_annotations = []
        for cell in selected_cells:
            paired = [lookup[(cell["cell_id"], "Z", q)] for q in TARGETS]
            name = "disk anchor" if cell["is_deployed_anchor"] else cell["spec"]["design"]
            line, = axes[index].plot([row["application_proposals_per_frame"] for row in paired],
                [row[METRICS[0]] for row in paired], marker="o", label=name)
            q1 = next(row for row in paired if row["target_proposals_per_frame"] == 1.0)
            color = line.get_color()
            axes[index].scatter([q1["application_proposals_per_frame"]], [q1[METRICS[0]]],
                s=85, facecolors="none", edgecolors=color, linewidths=1.3, zorder=4)
            q1_annotations.append((name, q1, color))
        # Repeated labels at every q obscured the current-frame low-burden
        # cluster. Keep every original point and identify the fixed audit q=1
        # with separated callouts; the complete q sequence is stated once.
        q1_annotations.sort(key=lambda item: (item[1][METRICS[0]], item[1]["application_proposals_per_frame"], item[0]))
        for rank, (name, row, color) in enumerate(q1_annotations):
            label_position = (.24, .16 + .07*rank) if index == 0 else (.44, .12 + .07*rank)
            axes[index].annotate(f"{name}: q=1",
                (row["application_proposals_per_frame"], row[METRICS[0]]),
                xytext=label_position, textcoords="axes fraction", fontsize=7,
                color=color, ha="left", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.5},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": .7, "shrinkA": 2, "shrinkB": 6})
        axes[index].set_title("Conditioned current frame" if index == 0 else "Signed adjacent difference")
        axes[index].set_xlabel("Realized proposals per application frame")
        axes[index].set_ylim(0, 1)
        axes[index].grid(alpha=.2)
        axes[index].legend(fontsize=8)
    axes[0].set_ylabel("Known-positive occurrence-window recall")
    fig.suptitle("Fixed core Gamma controls: coverage versus realized application burden", fontsize=11)
    fig.text(.5, .015,
        "Each curve follows q = 0.25, 0.5, 1, 2, 5; larger open circles and callouts mark the frozen audit point q=1.",
        ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, 1))
    path = destination / "core_Z_recall_vs_realized_burden.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))
    return files


def _report_text(rows: Sequence[Mapping[str, Any]], factorial: Sequence[Mapping[str, Any]], readouts: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Two-stencil factorial and LS-output comparison", "",
        "The complete fixed design contains 24 factorial cells and one separate deployed disk anchor. This report preserves all175 cell/readout/target results, all200 paired Z-factorial contrasts, and all75 paired readout-control contrasts. No winning cell is selected from these labels.", "",
        "Calibration uses the99 defined adjacent-difference score frames at source UI2–100. Application covers UI101–2359 (2,259 frames). The target q means NMS proposals per calibration score frame. It is a different unit from the older target per one-second initialization block; the old371-proposal ledger is not a same-calibration baseline. The first100 source frames are not assumed event-free, and calibration does not control the application proposal rate.", "",
        "The biological endpoint is descriptive occurrence-window spatial-site recall against the same79 sparse positive occurrences (26 recurring identities). Within each declared burst, proposals are grouped around fixed score-ranked spatial representatives before one-to-one matching. Emitted frame proposals, spatial representatives, and matched known occurrences are separate counts. This does not identify precision, the number of unique events, event onset latency, or external generalization.", "",
        "Point uses X as the test response; direct and serial use the added target-smoothed response A. Direct estimates reference moments from X, whereas serial estimates them from A. Serial therefore changes the reference variability and effective source support as well as smoothing the test response. Guard0 means no explicit guard exclusion: the n=9 Gamma profile still has zero center weight, while the uniform reference includes the center. These are the declared operator differences, not interchangeable estimates of a common null variance.", "",
        "## Paired factorial differences", "",
        "Every difference changes one specified factor while holding input, readout Z, target q, and the remaining factorial settings fixed. The disk anchor is excluded from these factorial differences. Means and ranges below summarize design contrasts on the same data; they are not independent replicates or inferential confidence intervals.", "",
        "| Difference | Input | Paired settings | Mean recall difference | Recall difference range | Mean emitted-proposal difference |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    grouped = defaultdict(list)
    for row in factorial:
        grouped[(row["comparison"], row["input_representation"])].append(row)
    for (comparison, representation), group in sorted(grouped.items()):
        recall = [row["delta_" + METRICS[0]] for row in group]
        count = [row["delta_emitted_frame_proposal_count"] for row in group]
        lines.append(f"| {comparison} | {representation} | {len(group)} | {sum(recall)/len(recall):+.5f} | [{min(recall):+.5f}, {max(recall):+.5f}] | {sum(count)/len(count):+.1f} |")
    lines += [
        "", "[All paired factorial rows](factorial_contrasts.tsv) retain every operating point and both sides of each difference.", "",
        "## LS-output readout controls", "",
        "A is the target response and contrast is A−M; both remain in native amplitude units with separately calibrated cutoffs. Positive global affine normalization of a completed readout would preserve NMS ordering and quantile-based threshold decisions apart from numerical rounding. Z divides contrast by the floored local reference standard deviation: it is dimensionless, with no standard-normal-distribution claim. A and contrast are scored only for the five predeclared control cells: direct Gamma guard0 and serial Gamma guard0 under each input, plus the deployed anchor. Each readout has its own calibration-only threshold at the same per-frame target. [All75 paired readout contrasts](readout_contrasts.tsv) retain contrast−A, Z−contrast, and Z−A; their comparison therefore concerns complete calibrated readouts, not a shared numeric threshold.", "",
        "![Predeclared readout curves](figures/predeclared_readout_controls.png)", "",
        "## Complete result matrix", "",
        "![All25 Z recall rows](figures/all25_Z_recall.png)", "",
        "![All25 Z proposal-count rows](figures/all25_Z_proposals.png)", "",
        "[All175 normalized results](results_all175.tsv) preserve the complete matrix, including each readout control. Readout-control cells are selected by protocol rather than observed recall.", "",
        "## Fixed stage comparisons", "",
        "[The stage figure index](stage_figure_index.json) lists every predeclared snapshot, both full-field and fixed central128-pixel views, for the six core point/direct/serial Gamma guard0 arms and the separate disk anchor. No crop or source frame is chosen from annotation locations or observed success. Raw, X, A, M, sigma, contrast, and Z are shown in grayscale. [Display limits](display_limits.json) pool fixed snapshots across all25 cells, with one shared scale per stage and input; full-field and crop views use the same limits. Signed stages use symmetric limits with zero at midgray. Reference sigma is local variability and is not a sampling-standard-error or calibrated null-distribution claim.", "",
        "## Validation boundary", "",
        "This module verifies matrix cardinality and count/recall reconciliation and produces comparison figures and tables. It does not replace the required complete Expert, Model, and Comparison scientific-audit media. Full scientific-audit completion remains pending until the separate renderer and inventory/media validation finish. No bootstrap, parameter refit, model selection, or new labels are introduced by this report.", "",
    ]
    return "\n".join(lines)


def generate_report(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    protocol_path, results_path = root / "protocol.json", root / "results.tsv"
    protocol = json.loads(protocol_path.read_text())
    for cell in protocol["cells"]:
        folder = root / "cells" / cell["cell_id"]
        complete = json.loads((folder / "numeric_complete.json").read_text())
        if complete.get("stage_snapshots_sha256") and _hash(folder / "stage_snapshots.npz") != complete["stage_snapshots_sha256"]:
            raise ValueError("stage snapshots changed after numerical completion")
    with results_path.open(newline="") as handle:
        rows = normalize_results(list(csv.DictReader(handle, delimiter="\t")), protocol)
    factorial, readouts = factorial_contrasts(rows), readout_contrasts(rows)
    if len(factorial) != 200 or len(readouts) != 75:
        raise ValueError("paired comparison cardinality mismatch")
    destination = root / "report_artifacts"
    destination.mkdir(exist_ok=True)
    _write_tsv(destination / "results_all175.tsv", rows)
    _write_tsv(destination / "factorial_contrasts.tsv", factorial)
    _write_tsv(destination / "readout_contrasts.tsv", readouts)
    limits = display_limits(root, protocol)
    _write_json(destination / "display_limits.json", limits)
    stage_index = render_stage_comparisons(root, destination / "stage_comparisons", protocol, limits)
    _write_json(destination / "stage_figure_index.json", stage_index)
    matrix_paths = render_result_matrices(rows, destination / "figures", protocol)
    report_text = _report_text(rows, factorial, readouts).replace("quantile-based threshold decisions", "rank-based threshold decisions")
    if protocol.get("calibration_amendment"):
        report_text += ("\n## Calibration correction\n\n"
            "This is the uniformly corrected development replay. The initial sampled-pixel threshold grid missed an attainable initialization cutoff: "
            "the current-frame point/Gamma guard0 arm emitted0 proposals at its selected q1 cutoff, while an exact NMS order-statistic cutoff emitted99 on the same99 calibration frames. "
            "The correction uses initialization data only and applies to all25 operators and every readout, with inputs, kernels, stages, and scale floors unchanged. "
            "The parent results were already computed; this replay is therefore development evidence, not untouched confirmation. "
            "Strict-threshold ties and any unattainable remainder are recorded in each calibration manifest.\n\n"
            "[Coverage versus realized application burden](figures/core_Z_recall_vs_realized_burden.png) shows the fixed core arms without retuning application thresholds.\n")
    (destination / "REPORT.md").write_text(report_text)
    summary = {
        "status": "COMPARISON_REPORT_COMPLETE_FULL_SCIENTIFIC_AUDIT_PENDING",
        "results_count": len(rows), "factorial_contrast_count": len(factorial),
        "readout_contrast_count": len(readouts), "stage_figure_count": len(stage_index),
        "result_figure_count": len(matrix_paths), "full_scientific_audit_complete": False,
        "winner_selected": False, "parameter_refits_performed": False,
        "source_protocol_sha256": _hash(protocol_path), "source_results_sha256": _hash(results_path),
        "report": str((destination / "REPORT.md").relative_to(root)),
    }
    _write_json(destination / "summary.json", summary)
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != "artifact_index.json")
    _write_json(destination / "artifact_index.json", {"artifacts": [
        {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _hash(path)} for path in files
    ]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="completed campaign root")
    arguments = parser.parse_args()
    print(json.dumps(generate_report(arguments.output), sort_keys=True))


if __name__ == "__main__":
    main()
