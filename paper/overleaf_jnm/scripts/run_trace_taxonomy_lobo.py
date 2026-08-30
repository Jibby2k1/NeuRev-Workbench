#!/usr/bin/env python3
"""Leave-one-burst-out persistence analysis for reassessed trace classes."""
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
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neurobench.portable_paths import portable_path

SEED = 20260826
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FINAL = ROOT / "figures" / "final"
TRACE_SOURCE = FINAL / "fig04_population_trace_summary_source.tsv"
FULL_ASSIGNMENTS = FINAL / "fig04_trace_taxonomy_site_assignments.tsv"
OUT_TSV = FINAL / "fig04c_trace_taxonomy_lobo_assignments.tsv"
OUT_JSON = FINAL / "fig04c_trace_taxonomy_lobo_summary.json"
OUT_FIG = FINAL / "fig04c_trace_taxonomy_lobo_persistence.png"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def vectorize(stage_traces: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for trace in stage_traces:
        x = np.where(np.isfinite(trace), trace, 0.0)
        norm = np.linalg.norm(x)
        blocks.append(x / (norm if norm > 0 else 1.0))
    return np.concatenate(blocks)


def align(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    k = 2
    table = np.zeros((k, k), dtype=int)
    for a, b in zip(reference, target, strict=True):
        table[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(c): int(r) for r, c in zip(rows, cols, strict=True)}
    return np.array([mapping[int(x)] for x in target]), mapping


def bootstrap_site_mean(rows: list[dict], include_burst2: bool, rng: np.random.Generator) -> tuple[float, float, float]:
    selected = [r for r in rows if include_burst2 or r["burst_id"] != 2]
    by_site = defaultdict(list)
    for r in selected:
        by_site[r["site_id"]].append(float(r["agreement_with_training_class"]))
    values = np.array([np.mean(v) for v in by_site.values()])
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    draws = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(5000)])
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> int:
    full_rows = list(csv.DictReader(FULL_ASSIGNMENTS.open(newline="", encoding="utf-8"), delimiter="\t"))
    full_class = {r["site_id"]: int(r["joint_class_id"]) - 1 for r in full_rows}

    # Reconstruct one shape vector per observation from the long-form figure source.
    grouped = defaultdict(lambda: defaultdict(dict))
    meta = {}
    with TRACE_SOURCE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["observation_id"], row["site_id"], int(row["burst_id"]))
            meta[key] = key
            grouped[key][row["stage"]][int(row["relative_frame"])] = float(row["shape_normalized"])
    stage_order = ("Raw", "CS-Parzen ICA", "Local standardization")
    observations = []
    for (observation_id, site, burst), stages in grouped.items():
        traces = []
        for stage in stage_order:
            frames = stages[stage]
            traces.append(np.array([frames[k] for k in sorted(frames)], dtype=float))
        observations.append({"observation_id": observation_id, "site_id": site, "burst_id": burst,
                             "vector": vectorize(traces)})

    rows = []
    rng = np.random.default_rng(SEED)
    for held_burst in (1, 2, 3, 4):
        train = [o for o in observations if o["burst_id"] != held_burst]
        held = [o for o in observations if o["burst_id"] == held_burst]
        train_by_site = defaultdict(list)
        for o in train:
            train_by_site[o["site_id"]].append(o["vector"])
        train_sites = sorted(train_by_site)
        x_train = np.stack([np.mean(train_by_site[s], axis=0) for s in train_sites])
        scaler = StandardScaler(with_mean=True, with_std=False).fit(x_train)
        centered = scaler.transform(x_train)
        full_pca = PCA(svd_solver="full").fit(centered)
        n_pcs = int(np.searchsorted(np.cumsum(full_pca.explained_variance_ratio_), 0.90) + 1)
        n_pcs = max(2, min(n_pcs, len(train_sites) - 1, centered.shape[1]))
        pca = PCA(n_components=n_pcs, svd_solver="full").fit(centered)
        z_train = pca.transform(centered)
        raw_labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(z_train)
        ref = np.array([full_class[s] for s in train_sites])
        train_labels, mapping = align(ref, raw_labels)
        centroids = np.stack([np.mean(z_train[train_labels == c], axis=0) for c in (0, 1)])
        train_label_map = dict(zip(train_sites, train_labels, strict=True))

        for o in held:
            if o["site_id"] not in train_label_map:
                continue
            z = pca.transform(scaler.transform(o["vector"][None, :]))[0]
            distances = np.linalg.norm(centroids - z, axis=1)
            order = np.argsort(distances)
            predicted = int(order[0])
            margin = float((distances[order[1]] - distances[order[0]]) / (distances.sum() + 1e-12))
            training_class = int(train_label_map[o["site_id"]])
            rows.append({
                "observation_id": o["observation_id"], "site_id": o["site_id"], "burst_id": held_burst,
                "training_class_id": training_class + 1, "heldout_class_id": predicted + 1,
                "full_joint_class_id": full_class[o["site_id"]] + 1,
                "agreement_with_training_class": int(predicted == training_class),
                "agreement_with_full_joint_class": int(predicted == full_class[o["site_id"]]),
                "centroid_margin": margin, "nearest_distance": float(distances[order[0]]),
                "second_distance": float(distances[order[1]]), "training_sites": len(train_sites),
                "pca_components": n_pcs,
            })

    fields = list(rows[0])
    with OUT_TSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerows(rows)

    fold_summary = []
    for burst in (1, 2, 3, 4):
        q = [r for r in rows if r["burst_id"] == burst]
        fold_summary.append({
            "burst_id": burst, "eligible_recurrent_sites": len(q),
            "agreement_with_training_class": float(np.mean([r["agreement_with_training_class"] for r in q])) if q else None,
            "median_margin": float(np.median([r["centroid_margin"] for r in q])) if q else None,
            "training_T1_retained": int(sum(r["training_class_id"] == 1 and r["heldout_class_id"] == 1 for r in q)),
            "training_T1_total": int(sum(r["training_class_id"] == 1 for r in q)),
            "training_T2_retained": int(sum(r["training_class_id"] == 2 and r["heldout_class_id"] == 2 for r in q)),
            "training_T2_total": int(sum(r["training_class_id"] == 2 for r in q)),
        })
    overall = bootstrap_site_mean(rows, True, rng)
    no_b2 = bootstrap_site_mean(rows, False, rng)
    # Matrix: recurrent sites only, ordered by full class and then agreement.
    by_site_rows = defaultdict(list)
    for r in rows:
        by_site_rows[r["site_id"]].append(r)
    all_fold_agreement_sites = int(sum(all(r["agreement_with_training_class"] for r in q) for q in by_site_rows.values()))
    constant_heldout_label_sites = int(sum(len({r["heldout_class_id"] for r in q}) == 1 for q in by_site_rows.values()))
    all_margins = np.array([r["centroid_margin"] for r in rows], dtype=float)
    sites = sorted(by_site_rows, key=lambda s: (full_class[s], -np.mean([r["agreement_with_training_class"] for r in by_site_rows[s]]), s))
    label_matrix = np.full((len(sites), 4), np.nan)
    margin_matrix = np.full((len(sites), 4), np.nan)
    agreement_matrix = np.full((len(sites), 4), np.nan)
    for i, site in enumerate(sites):
        for r in by_site_rows[site]:
            j = r["burst_id"] - 1
            label_matrix[i, j] = r["heldout_class_id"]
            margin_matrix[i, j] = r["centroid_margin"]
            agreement_matrix[i, j] = r["agreement_with_training_class"]

    fig = plt.figure(figsize=(10.2, max(5.2, 0.29 * len(sites) + 2.3)))
    gs = fig.add_gridspec(1, 2, width_ratios=(3.2, 1.25), wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    cmap = ListedColormap(["#2463A8", "#D28A19"])
    masked = np.ma.masked_invalid(label_matrix - 1)
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for i in range(len(sites)):
        for j in range(4):
            if np.isfinite(label_matrix[i, j]):
                margin = margin_matrix[i, j]
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                           alpha=0.72 * (1 - margin), edgecolor="none"))
                ax.text(j, i, f"{margin:.2f}", ha="center", va="center", fontsize=6,
                        color="#111111", fontweight="bold" if agreement_matrix[i, j] == 0 else "normal")
    boundary = sum(full_class[s] == 0 for s in sites)
    ax.axhline(boundary - 0.5, color="black", lw=1.3)
    ax.set(xticks=range(4), xticklabels=["Burst 1", "Burst 2", "Burst 3", "Burst 4"],
           yticks=range(len(sites)), yticklabels=[s.split("@")[0] for s in sites],
           xlabel="Held-out burst", ylabel="Immutable recurrent site",
           title="Held-out class and centroid margin")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.text(3.55, max(0, boundary / 2), "full T1", va="center", fontsize=8, color="#2463A8")
    ax.text(3.55, boundary + max(0, (len(sites) - boundary) / 2), "full T2", va="center", fontsize=8, color="#D28A19")

    ax2 = fig.add_subplot(gs[0, 1])
    burst_agree = [x["agreement_with_training_class"] for x in fold_summary]
    burst_margin = [x["median_margin"] for x in fold_summary]
    y = np.arange(4)
    ax2.barh(y - 0.16, burst_agree, height=0.3, color="#2463A8", label="agreement")
    ax2.barh(y + 0.16, burst_margin, height=0.3, color="#D28A19", label="median margin")
    ax2.set(yticks=y, yticklabels=[f"Burst {i}" for i in range(1, 5)], xlim=(0, 1),
            xlabel="Fraction / margin", title="Fold summary")
    ax2.invert_yaxis(); ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.axvline(0.5, color="#666666", ls="--", lw=0.8)
    fig.suptitle("Leave-one-burst-out persistence of reassessed trace classes", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Cell color is held-out T1/T2; number is relative centroid margin. Pale cells are ambiguous; bold numbers disagree with the class learned from the other bursts. Blank means unavailable.", fontsize=7.5)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

    summary = {
        "status": "exploratory_single_recording",
        "contract": "leave one burst out; refit two-class joint shape taxonomy on remaining bursts; assign held-out recurrent sites by nearest training centroid",
        "eligible_rows": len(rows), "eligible_recurrent_sites": len(sites), "folds": fold_summary,
        "site_persistence": {"all_fold_agreement_sites": all_fold_agreement_sites,
                             "constant_heldout_label_sites": constant_heldout_label_sites,
                             "median_centroid_margin": float(np.median(all_margins)),
                             "fraction_margin_below_0p10": float(np.mean(all_margins < 0.10))},
        "site_blocked_agreement": {"all_bursts": {"mean": overall[0], "ci95": [overall[1], overall[2]]},
                                   "excluding_burst_2": {"mean": no_b2[0], "ci95": [no_b2[1], no_b2[2]]}},
        "margin_definition": "(second-nearest distance - nearest distance) / sum of both distances",
        "limits": ["only recurrent sites test persistence", "single recording", "class alignment uses full joint labels for naming only",
                   "centroid assignment is descriptive, not calibrated probability"],
        "seed": SEED,
        "inputs": {portable_path(path, repository=REPO): sha256(path) for path in (TRACE_SOURCE, FULL_ASSIGNMENTS)},
        "outputs": {"assignments": portable_path(OUT_TSV, repository=REPO), "figure": portable_path(OUT_FIG, repository=REPO)},
    }
    summary["outputs"]["assignments_sha256"] = sha256(OUT_TSV)
    summary["outputs"]["figure_sha256"] = sha256(OUT_FIG)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"eligible_rows": len(rows), "recurrent_sites": len(sites), "folds": fold_summary,
                      "site_blocked_agreement": summary["site_blocked_agreement"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
