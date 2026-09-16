"""Redraw saved crowding diagnostic tables with compact, readable axis labels.

This presentation-only companion does not read movies or score arrays, rerun
NMS, fit cutoffs, reassign truth, or alter the original diagnostic capsule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "Outputs/GammaLSFollowup/followup_20260914_r1"
GATES = ("center_above_tau", "center_max_eligible_w3", "center_max_eligible_w13",
         "prefix_near2", "selected_near2", "weak_assigned_r2", "weak_assigned_r6")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Input changed while hashing: {path}")
    return dict(path=str(path), size_bytes=after.st_size, sha256=digest.hexdigest())


def verify(record):
    actual = binding(record["path"])
    require(all(actual[key] == record[key] for key in ("sha256", "size_bytes")),
            f"Changed input: {actual['path']}")
    return actual


def table(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def group_key(row, with_seed=True):
    base = (row["readout"], None if row["separation_px"] == "" else int(row["separation_px"]),
            int(row["selector_window_px"]))
    return base + (int(row["seed"]),) if with_seed else base


def geometry(fig, axes, artists):
    """Check text containment and between-row ylabel separation after layout."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    records = []
    for name, artist in artists:
        box = artist.get_window_extent(renderer)
        require(box.x0 >= bounds.x0 and box.y0 >= bounds.y0 and
                box.x1 <= bounds.x1 and box.y1 <= bounds.y1,
                f"Text outside figure canvas: {name}")
        records.append(dict(name=name, bounds_display_px=list(map(float, box.extents))))
    label_boxes = [ax.yaxis.label.get_window_extent(renderer) for ax in axes
                   if ax.yaxis.label.get_text()]
    for i, first in enumerate(label_boxes):
        for second in label_boxes[i + 1:]:
            require(not first.overlaps(second), "Y-axis labels overlap between panels")
    return dict(text_containment="PASS", ylabel_separation="PASS", text_boxes=records,
                axes=[dict(xlim=list(map(float, ax.get_xlim())),
                           ylim=list(map(float, ax.get_ylim()))) for ax in axes])


def render(output, plan, rows, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    figures = output / "figures"
    figures.mkdir()
    colors = ("#315b89", "#777777", "#985c8d")
    labels = dict(A="Target (a.u.)", C="Contrast (a.u.)", Z="Z (unitless)")
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    expected_keys = {(score, distance, window, seed) for score in plan["readouts"]
                     for distance in plan["separations_px"] for window in plan["selectors"]
                     for seed in plan["seeds"]}
    require(set(groups) == expected_keys and len(groups) == 72, "Incomplete trace groups")
    for trace in groups.values():
        require(len(trace) == 246, "Wrong active-frame count")
        require([int(row["source_frame_ui"]) for row in trace] == list(range(185, 431)),
                "Trace source-frame order changed")
        require([int(row["ms_since_sampled_fluorescence_onset"]) for row in trace] == list(range(0, 4920, 20)),
                "Trace time coordinates changed")
        require(len({row["threshold_q1"] for row in trace}) == 1, "Cutoff is not fixed")
    checks = {}
    for score in plan["readouts"]:
        # Identical plotted values, figure size, plotting order, shared axes,
        # default margins, and subplot bounds preserve the original axis scales.
        fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
        artists = []
        for row_index, distance in enumerate(plan["separations_px"]):
            for col, window in enumerate(plan["selectors"]):
                ax = axes[row_index, col]
                for color, seed in zip(colors, plan["seeds"]):
                    trace = groups[(score, distance, window, seed)]
                    time = np.asarray([int(r["ms_since_sampled_fluorescence_onset"]) for r in trace]) / 1000
                    ax.plot(time, [float(r["center_score"]) for r in trace], color=color, lw=1.0)
                    ax.axhline(float(trace[0]["threshold_q1"]), color=color, lw=.8, ls="--")
                label = "Weak alone" if distance is None else f"Neighbor {distance}px left"
                ax.set_title(f"{label} | {window}x{window} prefilter", fontsize=10)
                ax.grid(alpha=.2)
                artists.append((f"title_{row_index}_{col}", ax.title))
                if col == 0:
                    ax.set_ylabel(labels[score], fontsize=9)
                    artists.append((f"ylabel_{row_index}", ax.yaxis.label))
                if row_index == 3:
                    ax.set_xlabel("Seconds since first active fluorescence sample", fontsize=9)
                    artists.append((f"xlabel_{col}", ax.xaxis.label))
        handles = [Line2D([0], [0], color=color, label=str(seed))
                   for color, seed in zip(colors, plan["seeds"])]
        handles += [Line2D([0], [0], color="black", ls="--", label="Setup-frozen q1 cutoff")]
        legend = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                            fontsize=9, bbox_to_anchor=(.5, .035))
        title = fig.suptitle(f"Fixed weak-center {score}: all three paired seeds and all separations", fontsize=13)
        caption = fig.text(.5, .015, "Same saved score fields for both selectors; separately calibrated cutoffs. Active window only. No NMS rerun or causal attribution.", ha="center", fontsize=8)
        fig.subplots_adjust(left=.105, right=.98, top=.925, bottom=.11, hspace=.4, wspace=.15)
        name = f"weak_center_{score}_all_seeds.png"
        checks[name] = geometry(fig, list(axes.flat), artists + [("legend", legend), ("title", title), ("caption", caption)])
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
    expected_pooled = [(score, distance, window) for score in plan["readouts"]
                       for distance in plan["separations_px"] for window in plan["selectors"]]
    require([group_key(row, False) for row in pooled] == expected_pooled, "Pooled row order changed")
    require(all(int(row["weak_active_frame_count"]) == 738 and int(row["paired_seed_count"]) == 3
                for row in pooled), "Pooled denominators changed")
    counts = np.asarray([[int(row[f"{gate}_frame_count"]) for gate in GATES] for row in pooled])
    require(np.all((counts >= 0) & (counts <= 738)), "Invalid gate counts")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(counts / 738, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GATES)), ["Center\n> cutoff", "Center max\n3x3", "Center max\n13x13", "Prefix\nwithin2px", "Selected\nwithin2px", "Weak assigned\n2px", "Weak assigned\n6px"], fontsize=9)
    labels = [f"{r['readout']} | {'alone' if r['separation_px'] == '' else r['separation_px']+'px'} | w{r['selector_window_px']}" for r in pooled]
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for y in range(len(pooled)):
        for x in range(len(GATES)):
            ax.text(x, y, f"{counts[y, x]}/738", ha="center", va="center", fontsize=8,
                    color="white" if counts[y, x] > 400 else "black")
    ax.set_title("Distinct diagnostic counts across all three paired seeds\n738 active weak-source frames per row; columns are not a nested loss funnel", fontsize=12)
    fig.subplots_adjust(left=.22, right=.98, top=.91, bottom=.085)
    caption = fig.text(.5, .025, "Full source tables retain each seed. Shade represents count /738; frames are not independent replicates. Saved q1 assignments only.", ha="center", fontsize=8)
    name = "weak_gate_counts_pooled_seeds.png"
    checks[name] = geometry(fig, [ax], [("title", ax.title), ("caption", caption)] +
                            [(f"xtick_{i}", label) for i, label in enumerate(ax.get_xticklabels())] +
                            [(f"ytick_{i}", label) for i, label in enumerate(ax.get_yticklabels())])
    checks[name]["color_limits"] = [0.0, 1.0]
    fig.savefig(figures / name, dpi=150)
    plt.close(fig)
    return checks


def run(root=ROOT):
    root = Path(root).resolve()
    source = root / "crowding_diagnostics"
    output = root / "crowding_diagnostics_layout_v2"
    require(not output.exists(), f"Refusing to overwrite existing layout: {output}")
    index_binding = binding(source / "artifact_index.json")
    original = json.loads(Path(index_binding["path"]).read_text())["artifacts"]
    require(len(original) == 10, "Unexpected original artifact count")
    for record in original:
        require(Path(record["path"]).resolve().is_relative_to(source), "Artifact outside original capsule")
        verify(record)
    plan = json.loads((source / "plan.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    provenance = json.loads((source / "source_manifest.json").read_text())
    require(summary["status"] == "PASS" and summary["weak_active_frame_rows"] == 17712,
            "Original numerical diagnostics are incomplete")
    require(provenance["frozen_parameters"] == plan and provenance["status"] == "VERIFIED_SAVED_SOURCES",
            "Original plan/provenance disagree")
    old_code = REPO / "neurobench/experiments/gamma_ls_difference/followup_diagnostics.py"
    old_binding = next(record for record in provenance["sources"] if Path(record["path"]) == old_code)
    verify(old_binding)
    rows = table(source / "weak_active_frame_observations.tsv")
    pooled = table(source / "pooled_seed_gate_counts.tsv")
    require(len(rows) == 17712 and len(pooled) == 24, "Incomplete source tables")
    output.mkdir()
    checks = render(output, plan, rows, pooled)
    for record in original + [index_binding, old_binding]:
        verify(record)
    manifest = dict(status="RENDER_PASS_PENDING_SEPARATE_VISUAL_QA",
        created_utc=datetime.now(timezone.utc).isoformat(), output=str(output),
        generator=binding(__file__), original_generator=old_binding,
        original_artifact_index=index_binding, original_artifacts=original,
        frozen_parameters=plan, frame_rows=len(rows), pooled_rows=len(pooled),
        figure_count=4, figures=[binding(path) for path in sorted((output / "figures").glob("*.png"))],
        layout_checks=checks,
        changes="Shorter y-axis labels only: Target (a.u.), Contrast (a.u.), Z (unitless). All original numerical plot calls, coordinates, scales, seed colors, cutoffs, figure sizes and subplot bounds retained.",
        provenance_scope="Original small artifacts and diagnostic source code freshly hash-verified before and after rendering. Scientific source bindings are inherited through the unchanged original source_manifest; movies, dense score arrays and proposal ledgers were not reopened.",
        original_artifacts_preserved=True, rescoring_performed=False, nms_rerun=False,
        thresholds_refitted=False, assignments_recomputed=False, scientific_promotion=False,
        visual_qa="Actual inspection must be recorded separately in visual_qa.json after viewing all four new images.")
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(status=manifest["status"], output=str(output), figures=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), sort_keys=True))
