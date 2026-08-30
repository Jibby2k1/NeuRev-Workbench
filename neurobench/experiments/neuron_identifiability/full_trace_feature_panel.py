"""Exploratory full-duration feature panel for canonical-v7 ROI traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-full-trace-panel-mpl")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from neurobench.algorithms.scientific_feature_audit import (
    causal_local_correlation_feature,
    generalized_anscombe,
)
from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import (
    _quiet_calibrate,
)
from neurobench.portable_paths import data_root, media_root, portable_path


ALIGNMENT_START_UI = 1800
FPS = 10.0
QUIET_GUARD = 15
SEED = 20260827
BOOTSTRAPS = 5000
FEATURES = (
    "raw_center",
    "ica_center",
    "ls_center",
    "raw_center_annulus",
    "vst_center_annulus",
    "matched_filter_ls",
    "carrier_signed",
    "coherence_w15",
    "propagation_lag2_w15",
    "representation_consensus",
    "multiscale_persistence",
)
FEATURE_LABELS = {
    "raw_center": "Raw center",
    "ica_center": "CS-Parzen ICA center",
    "ls_center": "Local standardization center",
    "raw_center_annulus": "Raw center-annulus",
    "vst_center_annulus": "VST center-annulus",
    "matched_filter_ls": "LS exponential matched filter",
    "carrier_signed": "Frozen amplitude carrier",
    "coherence_w15": "Local coherence (15 frames)",
    "propagation_lag2_w15": "Lag-2 recurrence (15 frames)",
    "representation_consensus": "Raw/ICA/LS consensus",
    "multiscale_persistence": "LS multiscale persistence",
}
INNOVATIONS = {"representation_consensus", "multiscale_persistence"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.nanmedian(values))
    scale = float(1.4826 * np.nanmedian(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(values))
    return scale if np.isfinite(scale) and scale > 1e-12 else 1.0


def quiet_standardize(trace: np.ndarray, quiet_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    quiet = values[quiet_mask]
    return (values - float(np.nanmedian(quiet))) / robust_scale(quiet)


def empirical_quiet_cdf(trace: np.ndarray, quiet_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    reference = np.sort(values[quiet_mask & np.isfinite(values)])
    if not len(reference):
        raise ValueError("empty quiet reference")
    return np.searchsorted(reference, values, side="right") / len(reference)


def causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    total = np.cumsum(np.insert(x, 0, 0.0))
    output = np.empty_like(x)
    for index in range(len(x)):
        start = max(0, index - int(window) + 1)
        output[index] = (total[index + 1] - total[start]) / (index - start + 1)
    return output


def exponential_matched_filter(values: np.ndarray, tau_frames: float = 10.0, length: int = 31) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    kernel = np.exp(-np.arange(int(length), dtype=np.float64) / float(tau_frames))
    kernel /= np.linalg.norm(kernel)
    # Causal correlation: current output uses current and preceding samples only.
    return np.convolve(x, kernel, mode="full")[: len(x)]


def annulus_trace(video: np.ndarray, x: int, y: int, inner: float = 3.0, outer: float = 6.0) -> np.ndarray:
    height, width = video.shape[1:]
    radius = int(np.ceil(outer))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = (distance >= inner) & (distance <= outer)
    return np.mean(np.asarray(video[:, y0:y1, x0:x1], dtype=np.float64)[:, mask], axis=1)


def make_event_and_quiet_masks(frame_count: int, intervals_ui: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    event = np.zeros(frame_count, dtype=bool)
    guarded = np.zeros(frame_count, dtype=bool)
    for start_ui, end_ui in intervals_ui:
        start = max(0, int(start_ui) - ALIGNMENT_START_UI)
        stop = min(frame_count, int(end_ui) - ALIGNMENT_START_UI + 1)
        event[start:stop] = True
        guarded[max(0, start - QUIET_GUARD) : min(frame_count, stop + QUIET_GUARD)] = True
    return event, ~guarded


def quiet_window_maxima(feature: np.ndarray, length: int, quiet_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(feature, dtype=np.float64)
    if length <= 0 or length > len(values):
        raise ValueError("invalid window length")
    maxima = [
        float(np.nanmax(values[start : start + length]))
        for start in range(len(values) - length + 1)
        if bool(np.all(quiet_mask[start : start + length]))
    ]
    if not maxima:
        raise RuntimeError("no same-duration quiet windows remain")
    return np.asarray(maxima, dtype=np.float64)


def occurrence_score(
    feature: np.ndarray,
    start_ui: int,
    end_ui: int,
    quiet_mask: np.ndarray,
    event_union: np.ndarray,
) -> dict[str, float]:
    start = int(start_ui) - ALIGNMENT_START_UI
    stop = int(end_ui) - ALIGNMENT_START_UI + 1
    event_peak = float(np.nanmax(feature[start:stop]))
    quiet_peaks = quiet_window_maxima(feature, stop - start, quiet_mask)
    percentile = float((np.sum(quiet_peaks < event_peak) + 0.5 * np.sum(quiet_peaks == event_peak)) / len(quiet_peaks))
    effect = (event_peak - float(np.nanmedian(quiet_peaks))) / robust_scale(quiet_peaks)
    positive = np.clip(np.asarray(feature, dtype=np.float64), 0.0, None)
    energy_fraction = float(np.sum(positive[event_union]) / max(float(np.sum(positive)), 1e-12))
    return {
        "event_peak": event_peak,
        "quiet_window_count": int(len(quiet_peaks)),
        "event_localization_percentile": percentile,
        "event_quiet_effect": float(effect),
        "event_energy_fraction": energy_fraction,
    }


def bootstrap_site_mean(rows: list[dict[str, Any]], field: str, rng: np.random.Generator) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_id"])].append(float(row[field]))
    sites = sorted(grouped)
    values = np.asarray([np.mean(grouped[site]) for site in sites], dtype=np.float64)
    draws = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(BOOTSTRAPS)])
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_paired_site_delta(
    rows: list[dict[str, Any]], feature: str, baseline: str, rng: np.random.Generator
) -> tuple[float, float, float, np.ndarray]:
    by_observation = defaultdict(dict)
    site_for_observation = {}
    for row in rows:
        by_observation[row["observation_id"]][row["feature_id"]] = float(row["event_localization_percentile"])
        site_for_observation[row["observation_id"]] = row["site_id"]
    by_site: dict[str, list[float]] = defaultdict(list)
    for observation, values in by_observation.items():
        if feature in values and baseline in values:
            by_site[site_for_observation[observation]].append(values[feature] - values[baseline])
    site_values = np.asarray([np.mean(v) for _, v in sorted(by_site.items())], dtype=np.float64)
    draws = np.asarray([np.mean(rng.choice(site_values, size=len(site_values), replace=True)) for _ in range(BOOTSTRAPS)])
    return float(np.mean(site_values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def add_blossom(fig: plt.Figure) -> None:
    center = np.array([0.972, 0.976])
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        offset = 0.009 * np.array([np.cos(angle), np.sin(angle)])
        fig.add_artist(plt.Circle(center + offset, 0.0045, transform=fig.transFigure, color="#2463A8", alpha=0.85))


def feature_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    summaries = []
    burst_rows = []
    bootstrap_rows = []
    for feature in FEATURES:
        selected = [row for row in rows if row["feature_id"] == feature]
        mean, low, high = bootstrap_site_mean(selected, "event_localization_percentile", rng)
        delta, delta_low, delta_high, draws = bootstrap_paired_site_delta(rows, feature, "carrier_signed", rng)
        burst_medians = {}
        for burst in range(1, 5):
            values = [float(row["event_localization_percentile"]) for row in selected if int(row["burst_id"]) == burst]
            median = float(np.median(values))
            burst_medians[str(burst)] = median
            burst_rows.append({"feature_id": feature, "burst_id": burst, "occurrences": len(values), "median_event_localization_percentile": median})
        summaries.append({
            "feature_id": feature,
            "feature_label": FEATURE_LABELS[feature],
            "family": "prespecified_extension" if feature in INNOVATIONS else "established_or_frozen",
            "occurrences": len(selected),
            "sites": len({row["site_id"] for row in selected}),
            "mean_event_localization_percentile": mean,
            "ci95_low": low,
            "ci95_high": high,
            "median_event_quiet_effect": float(np.median([float(row["event_quiet_effect"]) for row in selected])),
            "median_event_energy_fraction": float(np.median([float(row["event_energy_fraction"]) for row in selected])),
            "paired_delta_vs_carrier": delta,
            "paired_delta_ci95_low": delta_low,
            "paired_delta_ci95_high": delta_high,
            "supported_full_trace_localization": bool(low > 0.5 and min(burst_medians.values()) >= 0.75),
            "incremental_over_carrier": bool(delta_low > 0),
        })
        for index, value in enumerate(draws):
            bootstrap_rows.append({"feature_id": feature, "draw": index, "paired_delta_vs_carrier": float(value)})
    return summaries, burst_rows, bootstrap_rows


def repeatability(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    output = []
    medians: dict[str, float | None] = {}
    for feature in FEATURES:
        selected = [row for row in rows if row["feature_id"] == feature]
        by_burst = defaultdict(dict)
        for row in selected:
            by_burst[int(row["burst_id"])][row["site_id"]] = float(row["event_localization_percentile"])
        correlations = []
        for first in range(1, 5):
            for second in range(first + 1, 5):
                shared = sorted(set(by_burst[first]) & set(by_burst[second]))
                if len(shared) < 5:
                    rho = np.nan
                else:
                    values_a = np.asarray([by_burst[first][s] for s in shared], dtype=np.float64)
                    values_b = np.asarray([by_burst[second][s] for s in shared], dtype=np.float64)
                    rho = np.nan if np.ptp(values_a) == 0 or np.ptp(values_b) == 0 else float(spearmanr(values_a, values_b).statistic)
                output.append({"feature_id": feature, "burst_a": first, "burst_b": second, "shared_sites": len(shared), "spearman_rho": rho})
                if np.isfinite(rho):
                    correlations.append(rho)
        medians[feature] = float(np.median(correlations)) if correlations else None
    return output, medians


def render_figure(
    path: Path,
    summaries: list[dict[str, Any]],
    burst_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
) -> None:
    summary_by = {row["feature_id"]: row for row in summaries}
    labels = [FEATURE_LABELS[feature] for feature in FEATURES]
    y = np.arange(len(FEATURES))
    colors = ["#D28A19" if feature in INNOVATIONS else "#2463A8" if feature != "carrier_signed" else "#666666" for feature in FEATURES]
    fig = plt.figure(figsize=(12.2, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), height_ratios=(1.15, 1), hspace=0.34, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    means = np.array([summary_by[f]["mean_event_localization_percentile"] for f in FEATURES])
    low = np.array([summary_by[f]["ci95_low"] for f in FEATURES])
    high = np.array([summary_by[f]["ci95_high"] for f in FEATURES])
    ax.errorbar(means, y, xerr=np.vstack([means - low, high - means]), fmt="none", ecolor="#27313B", lw=1.1, capsize=2)
    ax.scatter(means, y, c=colors, s=46, edgecolor="#27313B", linewidth=0.5, zorder=3)
    ax.axvline(0.5, color="#777777", ls="--", lw=0.9)
    ax.set(xlim=(0, 1), yticks=y, yticklabels=labels, xlabel="Mean event-localization percentile", title="Full-trace localization with site-bootstrap 95% intervals")
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[0, 1])
    matrix = np.array([[next(row["median_event_localization_percentile"] for row in burst_rows if row["feature_id"] == f and row["burst_id"] == b) for b in range(1, 5)] for f in FEATURES])
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.5, vmax=1.0)
    for row in range(len(FEATURES)):
        for column in range(4):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.82 else "#17202A")
    ax.set(xticks=range(4), xticklabels=["B1", "B2", "B3", "B4"], yticks=y, yticklabels=[], title="Median localization by burst")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Percentile")

    ax = fig.add_subplot(gs[1, 0])
    deltas = np.array([summary_by[f]["paired_delta_vs_carrier"] for f in FEATURES])
    dlo = np.array([summary_by[f]["paired_delta_ci95_low"] for f in FEATURES])
    dhi = np.array([summary_by[f]["paired_delta_ci95_high"] for f in FEATURES])
    ax.errorbar(deltas, y, xerr=np.vstack([deltas - dlo, dhi - deltas]), fmt="none", ecolor="#27313B", lw=1.1, capsize=2)
    ax.scatter(deltas, y, c=colors, s=46, edgecolor="#27313B", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set(yticks=y, yticklabels=labels, xlabel="Paired percentile difference from frozen carrier", title="Incremental full-trace localization")
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[1, 1])
    by_feature = defaultdict(list)
    for row in repeat_rows:
        if np.isfinite(float(row["spearman_rho"])):
            by_feature[row["feature_id"]].append(float(row["spearman_rho"]))
    medians = [float(np.median(by_feature[f])) if by_feature[f] else np.nan for f in FEATURES]
    for index, feature in enumerate(FEATURES):
        values = by_feature[feature]
        ax.scatter(values, np.full(len(values), index), color="#9FB9D5", s=18, zorder=2)
        ax.scatter(medians[index], index, color=colors[index], edgecolor="#27313B", s=44, zorder=3)
    ax.axvline(0.3, color="#777777", ls="--", lw=0.9)
    ax.set(xlim=(-1, 1), yticks=y, yticklabels=[], xlabel="Burst-pair site-rank Spearman correlation", title="Cross-burst scalar repeatability")
    ax.invert_yaxis()

    fig.suptitle("Canonical-v7 full-trace feature panel", fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.01, 0.008, "106 confirmed occurrences at 50 immutable geometries; equal-duration quiet windows exclude all burst intervals plus 15-frame guards. Gold marks prespecified extensions. Off-window activity is not a verified biological negative.", fontsize=7.7)
    add_blossom(fig)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def default_paths(repo_root: Path) -> dict[str, Path]:
    data = data_root(repo_root)
    media = media_root(repo_root)
    return {
        "manifest": media / "v7_priority_neuron_media_cs_parzen/manifest.json",
        "raw": data / "Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
        "ica": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/recovery_msica.npy",
        "ls": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/recovery_msln.npy",
        "carrier": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy",
        "old_audit": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1/metrics.json",
        "v7_rescore": data / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v7/metrics.json",
        "upstream_media": media / "v7_priority_neuron_full_traces_cs_parzen/manifest.json",
        "workflow": repo_root / "docs/workflows/spon_ca_burst_full_trace_feature_panel.md",
        "analysis_script": Path(__file__).resolve(),
    }


def run(repo_root: Path, output_root: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    paths = default_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    if output_root.exists() or Path(str(output_root) + ".partial").exists():
        raise FileExistsError(f"refusing existing completed or partial output: {output_root}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    items = manifest["items"]
    intervals = sorted({(int(item["event_start_ui"]), int(item["event_end_ui"])) for item in items})
    runtime_hashes = {str(path): sha256(path) for path in paths.values()}
    input_hashes = {
        portable_path(path, repository=repo_root, data=data_root(repo_root), media=media_root(repo_root)): digest
        for path, digest in ((path, runtime_hashes[str(path)]) for path in paths.values())
    }
    source_hashes = manifest["inputs"]
    for key in ("raw", "ica", "ls"):
        if source_hashes.get(str(paths[key])) != runtime_hashes[str(paths[key])]:
            raise RuntimeError(f"source hash differs from validated manifest for {key}")
    preflight = {
        "status": "passed",
        "population": {"occurrences": len(items), "sites": len({f"{i['original_roi_id']}@x{i['x_int']}_y{i['y_int']}" for i in items}), "bursts": len(intervals)},
        "frame_contract": {"first_ui": ALIGNMENT_START_UI, "frame_count": 560, "fps": FPS, "intervals_ui_inclusive": intervals},
        "feature_ids": list(FEATURES),
        "input_hashes": input_hashes,
        "output_collision": False,
    }
    if preflight_only:
        return preflight

    partial = Path(str(output_root) + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    atomic_json(partial / "preflight.json", preflight)
    raw_all = np.load(paths["raw"], mmap_mode="r")
    ica = np.load(paths["ica"], mmap_mode="r")
    ls = np.load(paths["ls"], mmap_mode="r")
    carrier = np.load(paths["carrier"], mmap_mode="r")
    raw = raw_all[ALIGNMENT_START_UI - 1 : ALIGNMENT_START_UI - 1 + len(ica)]
    if raw.shape != ica.shape or raw.shape != ls.shape or raw.shape != carrier.shape or len(raw) != 560:
        raise RuntimeError(f"shape mismatch raw={raw.shape}, ica={ica.shape}, ls={ls.shape}, carrier={carrier.shape}")
    event_union, quiet_mask = make_event_and_quiet_masks(len(raw), intervals)

    unique_items = {}
    for item in items:
        site = f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"
        unique_items.setdefault(site, item)
    site_features: dict[str, dict[str, np.ndarray]] = {}
    base_traces = {}
    for site, item in sorted(unique_items.items()):
        x, y = int(item["x_int"]), int(item["y_int"])
        raw_center = np.asarray(raw[:, y, x], dtype=np.float64)
        ica_center = np.asarray(ica[:, y, x], dtype=np.float64)
        ls_center = np.asarray(ls[:, y, x], dtype=np.float64)
        raw_annulus = annulus_trace(raw, x, y)
        raw_z = quiet_standardize(raw_center, quiet_mask)
        ica_z = quiet_standardize(ica_center, quiet_mask)
        ls_z = quiet_standardize(ls_center, quiet_mask)
        center_annulus = quiet_standardize(raw_center - raw_annulus, quiet_mask)
        # Apply VST pixelwise before annular averaging.
        vst_center = generalized_anscombe(raw_center, variance_intercept=0.0, variance_slope=3.6322)
        height, width = raw.shape[1:]
        radius = 6
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (((xx - x) ** 2 + (yy - y) ** 2) >= 3**2) & (((xx - x) ** 2 + (yy - y) ** 2) <= 6**2)
        vst_patch = generalized_anscombe(np.asarray(raw[:, y0:y1, x0:x1], dtype=np.float32), variance_intercept=0.0, variance_slope=3.6322)
        vst_annulus = np.mean(vst_patch[:, mask], axis=1)
        vst_contrast = quiet_standardize(vst_center - vst_annulus, quiet_mask)
        consensus = np.minimum.reduce([
            empirical_quiet_cdf(raw_z, quiet_mask),
            empirical_quiet_cdf(ica_z, quiet_mask),
            empirical_quiet_cdf(ls_z, quiet_mask),
        ])
        positive_ls = np.clip(ls_z, 0.0, None)
        persistence = np.cbrt(np.maximum(causal_mean(positive_ls, 3), 0) * np.maximum(causal_mean(positive_ls, 7), 0) * np.maximum(causal_mean(positive_ls, 15), 0))
        site_features[site] = {
            "raw_center": np.clip(raw_z, 0.0, None),
            "ica_center": np.clip(ica_z, 0.0, None),
            "ls_center": positive_ls,
            "raw_center_annulus": np.clip(center_annulus, 0.0, None),
            "vst_center_annulus": np.clip(vst_contrast, 0.0, None),
            "matched_filter_ls": exponential_matched_filter(ls_z),
            "carrier_signed": np.asarray(carrier[:, y, x], dtype=np.float64),
            "representation_consensus": consensus,
            "multiscale_persistence": persistence,
        }
        base_traces[site] = (x, y)

    for feature_id, lag in (("coherence_w15", 0), ("propagation_lag2_w15", 2)):
        reconstructed = causal_local_correlation_feature(
            carrier,
            window_frames=15,
            lag_frames=lag,
            spatial_sigma_px=2.0,
            activity_qualified=True,
        )
        reconstructed = _quiet_calibrate(reconstructed, 100)
        for site, (x, y) in base_traces.items():
            site_features[site][feature_id] = np.asarray(reconstructed[:, y, x], dtype=np.float64)
        del reconstructed

    occurrence_rows = []
    for item in items:
        site = f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"
        for feature_id in FEATURES:
            metrics = occurrence_score(site_features[site][feature_id], int(item["event_start_ui"]), int(item["event_end_ui"]), quiet_mask, event_union)
            occurrence_rows.append({
                "observation_id": item["observation_id"],
                "site_id": site,
                "canonical_roi_id": item["canonical_roi_id"],
                "burst_id": int(item["burst_id"]),
                "feature_id": feature_id,
                "feature_family": "prespecified_extension" if feature_id in INNOVATIONS else "established_or_frozen",
                "event_start_ui": int(item["event_start_ui"]),
                "event_end_ui": int(item["event_end_ui"]),
                **metrics,
            })
    summaries, burst_rows, bootstrap_rows = feature_summary(occurrence_rows)
    repeat_rows, repeat_medians = repeatability(occurrence_rows)
    for row in summaries:
        value = repeat_medians[row["feature_id"]]
        row["median_burst_pair_spearman"] = value
        row["repeatable_site_ordering"] = bool(value is not None and value >= 0.30)

    tables = partial / "tables"
    figures = partial / "figures"
    tables.mkdir()
    figures.mkdir()
    write_tsv(tables / "occurrence_feature_metrics.tsv", occurrence_rows)
    write_tsv(tables / "feature_summary.tsv", summaries)
    write_tsv(tables / "burst_feature_summary.tsv", burst_rows)
    write_tsv(tables / "burst_pair_repeatability.tsv", repeat_rows)
    write_tsv(tables / "paired_bootstrap_draws.tsv", bootstrap_rows)
    figure_path = figures / "full_trace_feature_panel.png"
    render_figure(figure_path, summaries, burst_rows, repeat_rows)

    old_audit = json.loads(paths["old_audit"].read_text(encoding="utf-8"))
    v7_rescore = json.loads(paths["v7_rescore"].read_text(encoding="utf-8"))
    headline = sorted(summaries, key=lambda row: row["mean_event_localization_percentile"], reverse=True)
    summary = {
        "schema_version": 1,
        "status": "complete_exploratory",
        "analysis_date": "2026-08-27",
        "population": preflight["population"],
        "estimand": "confirmed-event maximum relative to all same-duration guarded quiet-window maxima at the same immutable geometry",
        "quiet_frames": int(np.sum(quiet_mask)),
        "event_union_frames": int(np.sum(event_union)),
        "feature_panel": list(FEATURES),
        "feature_ranking": [row["feature_id"] for row in headline],
        "feature_summaries": summaries,
        "old_audit_anchor": {
            "feature_count": old_audit["feature_count"],
            "evaluated_lane_count": old_audit["evaluated_lane_count"],
            "conclusion": old_audit["conclusion"],
        },
        "v7_rescore_status": v7_rescore["status"],
        "interpretation_limits": [
            "exploratory reuse of one recording and four previously analyzed bursts",
            "quiet windows are temporal reference windows, not verified biological negatives",
            "event localization is not full-field precision or specificity",
            "lagged recurrence is not evidence of causal biological propagation",
            "prespecified extensions require independent-recording confirmation",
        ],
        "input_hashes": input_hashes,
    }
    atomic_json(partial / "summary.json", summary)
    llm_context = {
        "schema_version": 1,
        "status": summary["status"],
        "question": "Which compact interpretable features localize confirmed events over the complete trace, and which remain repeatable across bursts?",
        "grain": "106 confirmed occurrences nested in 50 immutable coordinate-defined sites",
        "primary_metric": "site-bootstrap mean event_localization_percentile",
        "unmatched_semantics": "unknown_not_negative",
        "primary_tables": ["tables/feature_summary.tsv", "tables/burst_feature_summary.tsv", "tables/burst_pair_repeatability.tsv"],
        "primary_figure": "figures/full_trace_feature_panel.png",
        "upstream_expert_media": portable_path(paths["upstream_media"], repository=repo_root, data=data_root(repo_root), media=media_root(repo_root)),
        "model_annotation_section": "not_applicable_no_new_candidates",
    }
    atomic_json(partial / "llm_context.json", llm_context)
    report_lines = [
        "# Canonical-v7 full-trace feature panel",
        "",
        "This exploratory analysis compares confirmed event windows with equal-duration guarded quiet windows at the same immutable geometry. It does not estimate precision or specificity.",
        "",
        "## Ranked full-trace localization",
        "",
    ]
    for rank, row in enumerate(headline, 1):
        report_lines.append(f"{rank}. `{row['feature_id']}`: mean percentile {row['mean_event_localization_percentile']:.3f} (site-bootstrap 95% interval {row['ci95_low']:.3f}--{row['ci95_high']:.3f}); delta from carrier {row['paired_delta_vs_carrier']:+.3f}.")
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "The panel is a within-recording feature-justification study. Agreement with the earlier protected feature audit strengthens mechanistic prioritization but is not independent replication. Off-window activity is not a verified negative class.",
        "",
    ]
    report_partial = partial / "REPORT.md.partial"
    report_partial.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    report_partial.replace(partial / "REPORT.md")
    validation = {
        "schema_version": 1,
        "status": "passed",
        "checks": {
            "occurrence_rows": len(occurrence_rows) == len(items) * len(FEATURES),
            "feature_summary_rows": len(summaries) == len(FEATURES),
            "finite_primary_metrics": all(np.isfinite(float(row["event_localization_percentile"])) for row in occurrence_rows),
            "percentiles_in_range": all(0 <= float(row["event_localization_percentile"]) <= 1 for row in occurrence_rows),
            "site_count": len(unique_items) == 50,
            "source_hashes_match_manifest": True,
            "figure_exists": figure_path.is_file() and figure_path.stat().st_size > 0,
        },
        "scientific_audit": {
            "mode": "analysis_only_reuses_validated_upstream_expert_media",
            "expert_annotations": portable_path(paths["upstream_media"], repository=repo_root, data=data_root(repo_root), media=media_root(repo_root)),
            "model_annotations": "not_applicable_no_new_detections",
            "comparison": "figures/full_trace_feature_panel.png and tables",
        },
    }
    if not all(validation["checks"].values()):
        raise RuntimeError(f"validation failed: {validation}")
    atomic_json(partial / "validation.json", validation)
    atomic_json(partial / "status.json", {"schema_version": 1, "status": "complete_exploratory", "validation": "passed"})
    artifacts = []
    for path in sorted(partial.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            artifacts.append({"path": str(path.relative_to(partial)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(partial / "artifact_index.json", {"schema_version": 1, "artifacts": artifacts})
    partial.replace(output_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.repo_root.resolve(), args.output_root.resolve(), preflight_only=args.preflight_only)
    print(json.dumps(result if args.preflight_only else {
        "status": result["status"],
        "population": result["population"],
        "feature_ranking": result["feature_ranking"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
