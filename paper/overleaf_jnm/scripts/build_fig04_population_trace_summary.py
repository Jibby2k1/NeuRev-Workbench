#!/usr/bin/env python3
"""Build the canonical-v7 population trace summary for Raw/CS-Parzen/LS.

The inferential and clustering unit is an immutable coordinate-defined original
site. Repeated burst occurrences are averaged within site before summaries.
Occurrence heatmaps retain all confirmed observations but use one shared order.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-mpl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from neurobench.portable_paths import data_root, media_root, portable_path
SEED = 20260826
FPS = 10.0
ALIGNMENT_START_UI = 1800
PRE_FRAMES = 15
STAGES = ("Raw", "CS-Parzen ICA", "Local standardization")
COLORS = ("#2463A8", "#D28A19", "#B14E63", "#777777")
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA_ROOT = data_root(REPO)
MEDIA_ROOT = media_root(REPO)
FIG_DIR = ROOT / "figures" / "final"
MEDIA_MANIFEST = MEDIA_ROOT / "v7_priority_neuron_media_cs_parzen/manifest.json"
TRACE_TAXONOMY_ASSIGNMENTS = FIG_DIR / "fig04_trace_taxonomy_site_assignments.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def robust_shape(trace: np.ndarray) -> tuple[np.ndarray, float, float]:
    baseline = float(np.nanmedian(trace[:PRE_FRAMES]))
    centered = trace - baseline
    scale = float(np.nanmax(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return centered / scale, baseline, scale


def site_key(item: dict) -> str:
    return f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"


def main() -> int:
    manifest = json.loads(MEDIA_MANIFEST.read_text())
    items = manifest["items"]
    input_paths = list(manifest["inputs"])
    raw_path = Path(next(p for p in input_paths if p.endswith("20ms.npy")))
    ica_path = Path(next(p for p in input_paths if p.endswith("recovery_msica.npy")))
    ls_path = Path(next(p for p in input_paths if p.endswith("recovery_msln.npy")))
    raw_all = np.load(raw_path, mmap_mode="r")
    ica = np.load(ica_path, mmap_mode="r")
    ls = np.load(ls_path, mmap_mode="r")
    raw = raw_all[ALIGNMENT_START_UI - 1 : ALIGNMENT_START_UI - 1 + len(ica)]
    arrays = (raw, ica, ls)
    if not (raw.shape == ica.shape == ls.shape):
        raise RuntimeError(f"stage alignment mismatch: {raw.shape}, {ica.shape}, {ls.shape}")

    max_post = max(i["clip_end_ui"] - i["event_start_ui"] for i in items)
    rel_frames = np.arange(-PRE_FRAMES, max_post + 1)
    n_time = len(rel_frames)
    shape = np.full((len(items), 3, n_time), np.nan, dtype=np.float64)
    native_peak = np.full((len(items), 3), np.nan, dtype=np.float64)
    peak_time = np.full((len(items), 3), np.nan, dtype=np.float64)
    site_ids = []

    for oi, item in enumerate(items):
        site_ids.append(site_key(item))
        x, y = item["x_int"], item["y_int"]
        start_ui = item["event_start_ui"] - PRE_FRAMES
        stop_ui = item["event_start_ui"] + max_post
        for si, arr in enumerate(arrays):
            a = max(start_ui, ALIGNMENT_START_UI)
            b = min(stop_ui, ALIGNMENT_START_UI + len(arr) - 1)
            trace = np.asarray(arr[a - ALIGNMENT_START_UI : b - ALIGNMENT_START_UI + 1, y, x], dtype=float)
            target_start = a - start_ui
            normalized, baseline, _ = robust_shape(trace)
            shape[oi, si, target_start : target_start + len(trace)] = normalized
            event_start = max(0, item["event_start_ui"] - a)
            event_stop = min(len(trace), item["event_end_ui"] - a + 1)
            event = trace[event_start:event_stop] - baseline
            p = int(np.nanargmax(event))
            native_peak[oi, si] = float(event[p])
            peak_time[oi, si] = p / FPS

    grouped = defaultdict(list)
    for i, key in enumerate(site_ids):
        grouped[key].append(i)
    sites = sorted(grouped)
    site_shape = np.stack([np.nanmean(shape[grouped[s]], axis=0) for s in sites])

    taxonomy_rows = list(csv.DictReader(TRACE_TAXONOMY_ASSIGNMENTS.open(newline="", encoding="utf-8"), delimiter="\t"))
    site_to_class = {row["site_id"]: int(row["joint_class_id"]) for row in taxonomy_rows}
    site_class = np.array([site_to_class[s] for s in sites], dtype=int)
    occurrence_class = np.array([site_to_class[key] for key in site_ids], dtype=int)
    display_groups = tuple(sorted(set(site_class)))
    class_names = tuple(f"Trace Class T{class_id}" for class_id in display_groups)
    rng = np.random.default_rng(SEED)
    occurrence_order = sorted(
        range(len(items)),
        key=lambda i: (display_groups.index(occurrence_class[i]), site_ids[i], items[i]["burst_id"]),
    )
    class_boundaries = []
    previous_class = occurrence_class[occurrence_order[0]]
    for position, oi in enumerate(occurrence_order[1:], 1):
        current_class = occurrence_class[oi]
        if current_class != previous_class:
            class_boundaries.append(position)
        previous_class = current_class

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "fig04_population_trace_summary.png"
    source_path = FIG_DIR / "fig04_population_trace_summary_source.tsv"
    provenance_path = FIG_DIR / "fig04_population_trace_summary_provenance.json"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 10})
    fig = plt.figure(figsize=(10.8, 12.6), constrained_layout=False)
    gs = fig.add_gridspec(4, 3, height_ratios=(1.05, 1.45, 2.25, 1.15), hspace=0.52, wspace=0.28)
    time_s = rel_frames / FPS
    event_end_s = max(i["event_end_ui"] - i["event_start_ui"] for i in items) / FPS

    for si, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, si])
        curves = site_shape[:, si, :]
        mean, sd = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0, ddof=1)
        ax.fill_between(time_s, mean - sd, mean + sd, color="#AFC9E5", alpha=0.55, linewidth=0)
        ax.plot(time_s, mean, color="#183A5A", lw=1.8)
        ax.axvspan(0, event_end_s, color="#D9D9D9", alpha=0.22, zorder=0)
        ax.axvline(0, color="#555555", lw=0.8, ls="--")
        ax.set_title(stage)
        ax.set_ylim(-1.05, 1.05)
        if si == 0:
            ax.set_ylabel("Mean normalized shape\n(shading: ±1 site SD)")
        ax.set_xlabel("Time from annotated onset (s)")

        ax = fig.add_subplot(gs[1, si])
        for ci, class_id in enumerate(display_groups):
            curves = site_shape[site_class == class_id, si, :]
            mean, sd = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else (np.zeros(n_time))
            ax.fill_between(time_s, mean - sd, mean + sd, color=COLORS[ci], alpha=0.11, linewidth=0)
            ax.plot(time_s, mean, color=COLORS[ci], lw=1.55 if class_id else 1.1,
                    label=f"{class_names[ci]} (n={len(curves)} sites)")
        ax.axvline(0, color="#555555", lw=0.8, ls="--")
        ax.set_ylim(-1.05, 1.05)
        if si == 0:
            ax.set_ylabel("Trace-class mean normalized shape\n(shading: ±1 site SD)")
            ax.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2)
        ax.set_xlabel("Time from annotated onset (s)")

        ax = fig.add_subplot(gs[2, si])
        mat = shape[occurrence_order, si, :]
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=(time_s[0], time_s[-1], len(items), 0))
        ax.axvline(0, color="black", lw=0.8, ls="--")
        for boundary in class_boundaries:
            ax.axhline(boundary, color="black", lw=1.1)
        if si == 0:
            ax.set_ylabel(f"All {len(items)} occurrences\n(joint trace-class order)")
        ax.set_xlabel("Time from annotated onset (s)")
        if si == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label("Per-occurrence normalized shape")

        ax = fig.add_subplot(gs[3, si])
        values = [np.array([np.nanmedian(native_peak[grouped[s], si]) for s in sites if site_to_class[s] == class_id])
                  for class_id in display_groups]
        parts = ax.violinplot(values, positions=np.arange(1, len(display_groups) + 1), showmeans=False,
                              showmedians=True, widths=0.8)
        for c, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLORS[c]); body.set_edgecolor(COLORS[c]); body.set_alpha(0.28)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[key].set_color("#333333"); parts[key].set_linewidth(0.8)
        for c, vals in enumerate(values):
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(np.full(len(vals), c + 1) + jitter, vals, s=9, color=COLORS[c], alpha=0.72, linewidths=0)
        ax.set_xticks(range(1, len(display_groups) + 1), [f"T{i}" for i in display_groups])
        ax.set_yscale("symlog", linthresh=0.5)
        if si == 0:
            ax.set_ylabel("Native event peak above\npre-event baseline (site median)")
        ax.set_xlabel("Joint trace-shape class")

    fig.suptitle("Reassessed canonical-v7 trace classes across the CS–Parzen pipeline", x=0.5, y=0.992,
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.958,
             f"{len(items)} confirmed occurrences, {len(sites)} immutable coordinate-defined sites; "
             + ", ".join(f"T{c}={np.sum(site_class == c)} sites" for c in display_groups),
             ha="center", fontsize=8.5, color="#444444")
    fig.text(0.01, 0.012,
             "Shape panels are baseline-centered and scaled independently per occurrence; amplitude violins remain in stage-native units. "
             "Classes are reassessed joint trace-shape profiles over all immutable sites; native amplitude was excluded from fitting.", fontsize=7.4, color="#444444")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with source_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["observation_id", "site_id", "burst_id", "trace_class_id", "stage", "relative_frame",
                  "relative_seconds", "shape_normalized", "native_event_peak", "time_to_peak_seconds"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for oi, item in enumerate(items):
            key = site_ids[oi]
            for si, stage in enumerate(STAGES):
                for ti, frame in enumerate(rel_frames):
                    writer.writerow({
                        "observation_id": item["observation_id"], "site_id": key,
                        "burst_id": item["burst_id"], "trace_class_id": int(occurrence_class[oi]),
                        "stage": stage, "relative_frame": int(frame), "relative_seconds": frame / FPS,
                        "shape_normalized": shape[oi, si, ti], "native_event_peak": native_peak[oi, si],
                        "time_to_peak_seconds": peak_time[oi, si],
                    })

    provenance = {
        "status": "descriptive_exploratory",
        "population": {"confirmed_occurrences": len(items), "immutable_sites": len(sites)},
        "site_definition": "original_roi_id plus rounded source x/y coordinate",
        "alignment": {"anchor": "annotated event onset", "pre_frames": PRE_FRAMES, "fps": FPS},
        "shape_normalization": "subtract pre-event median, divide by max absolute centered value per occurrence and stage",
        "aggregation": "mean shapes and median scalar metrics across bursts within immutable site",
        "trace_taxonomy": {
            "source": portable_path(TRACE_TAXONOMY_ASSIGNMENTS, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT),
            "source_sha256": sha256(TRACE_TAXONOMY_ASSIGNMENTS),
            "class_counts": {str(c): int(np.sum(site_class == c)) for c in display_groups},
            "class_interpretation": "reassessed joint trace-shape measurement profile, not biological cell type",
        },
        "source_manifest": portable_path(MEDIA_MANIFEST, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT),
        "source_manifest_sha256": sha256(MEDIA_MANIFEST),
        "source_arrays": {portable_path(p, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT): sha256(p) for p in (raw_path, ica_path, ls_path)},
        "outputs": {"figure": portable_path(fig_path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT), "source_tsv": portable_path(source_path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT)},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    provenance["outputs"]["figure_sha256"] = sha256(fig_path)
    provenance["outputs"]["source_tsv_sha256"] = sha256(source_path)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figure": str(fig_path), "sites": len(sites), "occurrences": len(items),
                      "trace_class_counts": {str(c): int(np.sum(site_class == c)) for c in display_groups}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
