#!/usr/bin/env python3
"""Reassess canonical-v7 trace classes and stage-wise label preservation."""
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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from neurobench.portable_paths import data_root, media_root, portable_path

SEED = 20260826
FPS = 10.0
ALIGNMENT_START_UI = 1800
PRE_FRAMES = 15
STAGES = ("Raw", "CS-Parzen ICA", "Local standardization")
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA_ROOT = data_root(REPO)
MEDIA_ROOT = media_root(REPO)
OUT = ROOT / "figures" / "final"
MANIFEST = MEDIA_ROOT / "v7_priority_neuron_media_cs_parzen/manifest.json"
COLORS = ("#2463A8", "#D28A19", "#B14E63", "#65733C", "#6E5A9A", "#96705B")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def site_key(item: dict) -> str:
    return f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"


def normalize(trace: np.ndarray) -> tuple[np.ndarray, float]:
    baseline = float(np.median(trace[:PRE_FRAMES]))
    centered = trace - baseline
    amplitude = float(np.max(centered))
    scale = float(np.max(np.abs(centered)))
    return centered / (scale if scale > 1e-12 else 1.0), amplitude


def pca_features(x: np.ndarray, max_components: int = 49) -> tuple[np.ndarray, int, float]:
    x = np.where(np.isfinite(x), x, 0.0)
    x = StandardScaler(with_mean=True, with_std=False).fit_transform(x)
    full = PCA(svd_solver="full").fit(x)
    n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.90) + 1)
    n = max(2, min(n, max_components, x.shape[0] - 1, x.shape[1]))
    model = PCA(n_components=n, svd_solver="full").fit(x)
    return model.transform(x), n, float(np.sum(model.explained_variance_ratio_))


def stability(features: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator, repeats: int = 500) -> list[float]:
    take = max(k * 3, int(np.ceil(0.8 * len(features))))
    values = []
    for _ in range(repeats):
        idx = np.sort(rng.choice(len(features), size=take, replace=False))
        refit = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(features[idx])
        values.append(float(adjusted_rand_score(labels[idx], refit)))
    return values


def align(reference: np.ndarray, target: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(reference, target, strict=True):
        matrix[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {int(c): int(r) for r, c in zip(rows, cols, strict=True)}
    aligned = np.array([mapping[int(x)] for x in target], dtype=int)
    aligned_matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(reference, aligned, strict=True):
        aligned_matrix[int(a), int(b)] += 1
    return aligned, mapping, aligned_matrix


def relabel_by_peak(labels: np.ndarray, peak: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int]]:
    order = sorted(range(k), key=lambda c: -float(np.median(peak[labels == c, 2])))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[int(x)] for x in labels]), mapping


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"]
    paths = list(manifest["inputs"])
    raw_path = Path(next(p for p in paths if p.endswith("20ms.npy")))
    ica_path = Path(next(p for p in paths if p.endswith("recovery_msica.npy")))
    ls_path = Path(next(p for p in paths if p.endswith("recovery_msln.npy")))
    raw_all = np.load(raw_path, mmap_mode="r")
    ica = np.load(ica_path, mmap_mode="r")
    ls = np.load(ls_path, mmap_mode="r")
    raw = raw_all[ALIGNMENT_START_UI - 1 : ALIGNMENT_START_UI - 1 + len(ica)]
    arrays = (raw, ica, ls)
    max_post = max(i["clip_end_ui"] - i["event_start_ui"] for i in items)
    rel = np.arange(-PRE_FRAMES, max_post + 1)
    traces = np.full((len(items), 3, len(rel)), np.nan)
    peaks = np.full((len(items), 3), np.nan)
    keys = []
    for oi, item in enumerate(items):
        keys.append(site_key(item))
        start_ui = item["event_start_ui"] - PRE_FRAMES
        stop_ui = item["event_start_ui"] + max_post
        for si, arr in enumerate(arrays):
            a, b = max(start_ui, ALIGNMENT_START_UI), min(stop_ui, ALIGNMENT_START_UI + len(arr) - 1)
            values = np.asarray(arr[a - ALIGNMENT_START_UI : b - ALIGNMENT_START_UI + 1, item["y_int"], item["x_int"]], dtype=float)
            norm, amp = normalize(values)
            offset = a - start_ui
            traces[oi, si, offset : offset + len(values)] = norm
            peaks[oi, si] = amp
    grouped = defaultdict(list)
    for i, key in enumerate(keys):
        grouped[key].append(i)
    sites = sorted(grouped)
    site_traces = np.stack([np.nanmean(traces[grouped[s]], axis=0) for s in sites])
    site_peaks = np.stack([np.nanmedian(peaks[grouped[s]], axis=0) for s in sites])

    # Equalize stage contributions before joint PCA.
    blocks = []
    for si in range(3):
        block = np.where(np.isfinite(site_traces[:, si]), site_traces[:, si], 0.0)
        norm = np.linalg.norm(block, axis=1, keepdims=True)
        blocks.append(block / np.where(norm > 0, norm, 1.0))
    joint_input = np.concatenate(blocks, axis=1)
    joint_features, joint_pcs, joint_var = pca_features(joint_input)
    rng = np.random.default_rng(SEED)
    candidates = {}
    for k in range(2, 7):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(joint_features)
        stab = stability(joint_features, labels, k, rng)
        sizes = np.bincount(labels, minlength=k)
        sil = float(silhouette_score(joint_features, labels))
        med = float(np.median(stab))
        viable = bool(np.min(sizes) >= 5)
        candidates[k] = {"labels": labels, "silhouette": sil, "median_subsample_ari": med,
                         "ari_q10": float(np.quantile(stab, 0.10)), "minimum_class_size": int(np.min(sizes)),
                         "viable": viable, "objective": sil * med if viable else -1.0}
    selected_k = max(candidates, key=lambda k: (candidates[k]["objective"], -k))
    joint_labels, _ = relabel_by_peak(candidates[selected_k]["labels"], site_peaks, selected_k)

    stage_labels = []
    stage_metrics = []
    for si, stage in enumerate(STAGES):
        features, pcs, var = pca_features(blocks[si])
        labels = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(features)
        aligned, mapping, table = align(joint_labels, labels, selected_k)
        stage_labels.append(aligned)
        stage_metrics.append({"stage": stage, "pca_components": pcs, "pca_variance": var,
                              "ari_vs_joint": float(adjusted_rand_score(joint_labels, aligned)),
                              "nmi_vs_joint": float(normalized_mutual_info_score(joint_labels, aligned)),
                              "aligned_confusion": table.tolist(), "label_mapping": mapping})
    stage_labels = np.stack(stage_labels, axis=1)

    # Representation sensitivities: PCA cap and amplitude-block weight.
    pca_cap_sensitivity = {}
    for cap in (8, 12, 20):
        cap_features, cap_pcs, cap_var = pca_features(joint_input, max_components=cap)
        cap_labels = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(cap_features)
        cap_labels, _, _ = align(joint_labels, cap_labels, selected_k)
        pca_cap_sensitivity[str(cap)] = {"components": cap_pcs, "variance": cap_var,
                                        "ari_vs_primary": float(adjusted_rand_score(joint_labels, cap_labels))}
    amp = StandardScaler().fit_transform(np.log1p(np.clip(site_peaks, 0, None)))
    amplitude_sensitivity = {}
    amp_labels = None
    for weight in (0.1, 0.25, 0.5, 1.0):
        augmented, amp_pcs, amp_var = pca_features(np.concatenate([joint_input, weight * amp], axis=1))
        labels = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(augmented)
        labels, _, table = align(joint_labels, labels, selected_k)
        amplitude_sensitivity[str(weight)] = {"pca_components": amp_pcs, "pca_variance": amp_var,
                                              "ari_vs_shape_only_joint": float(adjusted_rand_score(joint_labels, labels)),
                                              "aligned_confusion": table.tolist()}
        if weight == 0.5:
            amp_labels = labels
    assert amp_labels is not None

    assignment_path = OUT / "fig04_trace_taxonomy_site_assignments.tsv"
    summary_path = OUT / "fig04_trace_taxonomy_experiment.json"
    comparison_path = OUT / "fig04b_trace_taxonomy_stage_agreement.png"
    OUT.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["site_id", "occurrence_count", "joint_class_id", "raw_class_id", "ica_class_id", "ls_class_id",
                  "amplitude_sensitivity_class_id", "all_stages_agree"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader()
        for i, site in enumerate(sites):
            row = {"site_id": site, "occurrence_count": len(grouped[site]), "joint_class_id": int(joint_labels[i]) + 1,
                   "raw_class_id": int(stage_labels[i, 0]) + 1, "ica_class_id": int(stage_labels[i, 1]) + 1,
                   "ls_class_id": int(stage_labels[i, 2]) + 1, "amplitude_sensitivity_class_id": int(amp_labels[i]) + 1,
                   "all_stages_agree": bool(np.all(stage_labels[i] == stage_labels[i, 0]))}
            w.writerow(row)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), constrained_layout=True)
    for si, (ax, metric) in enumerate(zip(axes, stage_metrics, strict=True)):
        table = np.asarray(metric["aligned_confusion"])
        im = ax.imshow(table, cmap="Blues", vmin=0, vmax=max(1, table.max()))
        for r in range(selected_k):
            for c in range(selected_k):
                ax.text(c, r, str(table[r, c]), ha="center", va="center",
                        color="white" if table[r, c] > table.max() / 2 else "#222222")
        ax.set(xticks=range(selected_k), yticks=range(selected_k),
               xticklabels=[f"T{i+1}" for i in range(selected_k)], yticklabels=[f"T{i+1}" for i in range(selected_k)],
               xlabel=f"{STAGES[si]}-only class", ylabel="Joint class" if si == 0 else "",
               title=f"{STAGES[si]}\nARI={metric['ari_vs_joint']:.3f}")
    fig.suptitle("Stage-wise trace-class agreement after optimal label alignment", fontsize=12, fontweight="bold")
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

    summary = {
        "status": "exploratory_single_recording",
        "population": {"sites": len(sites), "occurrences": len(items)},
        "primary_contract": "shape-only, onset-aligned, per-occurrence amplitude removed, burst-averaged within immutable site",
        "joint_pca": {"components": joint_pcs, "variance_explained": joint_var},
        "candidate_k": {str(k): {key: value for key, value in row.items() if key != "labels"} for k, row in candidates.items()},
        "selected_k": selected_k,
        "joint_class_sizes": {str(c + 1): int(np.sum(joint_labels == c)) for c in range(selected_k)},
        "stage_comparison": stage_metrics,
        "all_stage_agreement_sites": int(np.sum(np.all(stage_labels == stage_labels[:, :1], axis=1))),
        "pca_cap_sensitivity": pca_cap_sensitivity,
        "amplitude_sensitivity": amplitude_sensitivity,
        "interpretation_limits": ["single recording", "classes are measurement trace profiles, not neuron types",
                                  "stage labels are aligned for comparison only", "no peak alignment"],
        "seed": SEED,
        "inputs": {portable_path(path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT): sha256(path) for path in (MANIFEST, raw_path, ica_path, ls_path)},
        "outputs": {"assignments": portable_path(assignment_path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT), "comparison_figure": portable_path(comparison_path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT)},
    }
    summary["outputs"]["assignments_sha256"] = sha256(assignment_path)
    summary["outputs"]["comparison_figure_sha256"] = sha256(comparison_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"selected_k": selected_k, "class_sizes": summary["joint_class_sizes"],
                      "stage_ari": {m["stage"]: m["ari_vs_joint"] for m in stage_metrics},
                      "all_stage_agreement_sites": summary["all_stage_agreement_sites"],
                      "pca_cap_sensitivity": pca_cap_sensitivity,
                      "amplitude_sensitivity_ari": {w: v["ari_vs_shape_only_joint"] for w, v in amplitude_sensitivity.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
