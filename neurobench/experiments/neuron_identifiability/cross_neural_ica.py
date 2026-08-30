"""Representation-matched cross-neural coactivity analysis.

The analysis is descriptive within one recording.  It compares immutable-site
event traces across Raw, ROI-minus-annulus residual, leave-one-site-out global
adjustment, and the hash-verified frozen two-frame ICA operator.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import ObservationRecord, atomic_json, atomic_text


REPRESENTATIONS = ("raw", "residual", "global_adjusted_raw", "frozen_two_frame_ica")
LAGS = tuple(range(-3, 4))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 4 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return 0.0
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _lag_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag < 0:
        return _corr(x[-lag:], y[:lag])
    if lag > 0:
        return _corr(x[:-lag], y[lag:])
    return _corr(x, y)


def _bh_adjust(values: Iterable[float]) -> list[float]:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        original = order[rank_index]
        rank = rank_index + 1
        running = min(running, float(p[original]) * len(p) / rank)
        adjusted[original] = running
    return adjusted.tolist()


def _global_adjust(cube: np.ndarray) -> np.ndarray:
    """Regress each site on the leave-one-site-out mean within each burst."""
    adjusted = np.empty_like(cube, dtype=float)
    for burst in range(cube.shape[1]):
        total = np.sum(cube[:, burst], axis=0)
        for site in range(cube.shape[0]):
            x = cube[site, burst]
            nuisance = (total - x) / max(cube.shape[0] - 1, 1)
            design = np.column_stack((np.ones(len(nuisance)), nuisance))
            beta = np.linalg.lstsq(design, x, rcond=None)[0]
            adjusted[site, burst] = x - design @ beta
    return adjusted


def event_cubes(
    records: list[ObservationRecord], site_ids: np.ndarray, traces: dict[str, np.ndarray]
) -> tuple[list[str], dict[str, np.ndarray], dict[str, tuple[float, float]], int]:
    counts = {site: sum(r.observation_site_id == site for r in records) for site in set(site_ids.tolist())}
    complete = sorted(site for site, count in counts.items() if count == 4)
    if not complete:
        raise ValueError("no sites observed in all four bursts")
    by_site = {site: idx for idx, site in enumerate(site_ids.tolist())}
    lengths = [r.stop_zero_exclusive - r.start_zero for r in records if r.observation_site_id in complete]
    length = min(lengths)
    record_index = {(r.observation_site_id, r.burst_id): r for r in records}
    cubes: dict[str, np.ndarray] = {}
    for representation in ("raw", "residual", "frozen_two_frame_ica"):
        cube = np.empty((len(complete), 4, length), dtype=float)
        for si, site in enumerate(complete):
            for burst in range(1, 5):
                record = record_index[(site, burst)]
                values = traces[representation][by_site[site]]
                event = values[record.start_zero:record.start_zero + length].astype(float)
                baseline = values[max(0, record.start_zero - 20):record.start_zero]
                baseline = baseline[np.isfinite(baseline)]
                cube[si, burst - 1] = event - (float(np.median(baseline)) if len(baseline) else 0.0)
        cubes[representation] = cube
    cubes["global_adjusted_raw"] = _global_adjust(cubes["raw"])
    coordinates = {}
    for site in complete:
        record = next(r for r in records if r.observation_site_id == site)
        coordinates[site] = (record.x_px, record.y_px)
    return complete, cubes, coordinates, length


def analyze_pairs(
    sites: list[str], cubes: dict[str, np.ndarray], coordinates: dict[str, tuple[float, float]],
    *, permutations: int = 2000, seed: int = 20260824,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        cube = cubes[representation]
        representation_rows = []
        observed_abs = []
        null_exceed = []
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                correlations = [_corr(cube[i, b], cube[j, b]) for b in range(4)]
                lag_rows = []
                for b in range(4):
                    candidates = [(lag, _lag_corr(cube[i, b], cube[j, b], lag)) for lag in LAGS]
                    lag_rows.append(max(candidates, key=lambda item: (abs(item[1]), -abs(item[0]))))
                median = float(np.median(correlations))
                observed = abs(median)
                exceed = 0
                for _ in range(permutations):
                    shifted = []
                    for b in range(4):
                        shift_i = int(rng.integers(2, cube.shape[2] - 1))
                        shift_j = int(rng.integers(2, cube.shape[2] - 1))
                        shifted.append(_corr(np.roll(cube[i, b], shift_i), np.roll(cube[j, b], shift_j)))
                    exceed += abs(float(np.median(shifted))) >= observed
                p_value = (exceed + 1) / (permutations + 1)
                x1, y1 = coordinates[sites[i]]; x2, y2 = coordinates[sites[j]]
                row = {
                    "representation": representation, "site_a": sites[i], "site_b": sites[j],
                    "distance_px": float(np.hypot(x1 - x2, y1 - y2)),
                    **{f"burst_{b + 1}_correlation": correlations[b] for b in range(4)},
                    "median_correlation": median,
                    "sign_agreement_fraction": float(max(sum(v > 0 for v in correlations), sum(v < 0 for v in correlations)) / 4),
                    "median_best_lag_frames": float(np.median([v[0] for v in lag_rows])),
                    "median_best_lag_correlation": float(np.median([v[1] for v in lag_rows])),
                    "circular_shift_p_two_sided": p_value,
                }
                representation_rows.append(row); observed_abs.append(observed); null_exceed.append(p_value)
        adjusted = _bh_adjust(null_exceed)
        for row, q in zip(representation_rows, adjusted):
            row["circular_shift_q_bh"] = q
            row["stable_edge"] = bool(q <= 0.05 and row["sign_agreement_fraction"] >= 0.75)
        rows.extend(representation_rows)
        summaries[representation] = {
            "pairs": len(representation_rows),
            "median_absolute_pair_correlation": float(np.median(observed_abs)),
            "stable_edges_q_le_0_05_sign_agreement_ge_0_75": int(sum(r["stable_edge"] for r in representation_rows)),
            "median_distance_px": float(np.median([r["distance_px"] for r in representation_rows])),
        }
    return rows, summaries


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def _figures(output: Path, sites: list[str], rows: list[dict[str, Any]], coordinates: dict[str, tuple[float, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"raw": "#3568a8", "residual": "#d08328", "global_adjusted_raw": "#6d7f32", "frozen_two_frame_ica": "#ad4d75"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, representation in zip(axes.flat, REPRESENTATIONS):
        matrix = np.eye(len(sites))
        for row in rows:
            if row["representation"] != representation: continue
            i, j = sites.index(row["site_a"]), sites.index(row["site_b"])
            matrix[i, j] = matrix[j, i] = row["median_correlation"]
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(title=representation.replace("_", " "), xticks=range(len(sites)), yticks=range(len(sites)), xticklabels=sites, yticklabels=sites)
        ax.tick_params(axis="x", rotation=90, labelsize=7); ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axes, shrink=.7, label="median event-window correlation across four bursts")
    fig.suptitle("Cross-neural correlation matrices by representation\n14 sites observed in all four bursts; 24-frame common event window", fontsize=14)
    fig.savefig(output / "cross_neural_representation_matrices.png", dpi=180); plt.close(fig)

    raw = {(r["site_a"], r["site_b"]): r["median_correlation"] for r in rows if r["representation"] == "raw"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, representation in zip(axes, REPRESENTATIONS[1:]):
        subset = [r for r in rows if r["representation"] == representation]
        x = [raw[(r["site_a"], r["site_b"])] for r in subset]; y = [r["median_correlation"] for r in subset]
        ax.scatter(x, y, s=24, color=palette[representation], alpha=.75, edgecolor="white", linewidth=.3)
        ax.axhline(0, color="#888888", lw=.8); ax.axvline(0, color="#888888", lw=.8); ax.plot([-1, 1], [-1, 1], "--", color="#555555", lw=.8)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Raw median correlation", ylabel=f"{representation.replace('_', ' ')} median correlation", title=representation.replace("_", " "))
        ax.grid(alpha=.15)
    fig.suptitle("Pairwise coactivity changes under nuisance removal and frozen ICA\nEach point is one immutable-site pair (n=91)", fontsize=14)
    fig.savefig(output / "cross_neural_representation_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, representation in zip(axes.flat, REPRESENTATIONS):
        subset = [r for r in rows if r["representation"] == representation and r["stable_edge"]]
        for row in subset:
            xa, ya = coordinates[row["site_a"]]; xb, yb = coordinates[row["site_b"]]
            ax.plot([xa, xb], [ya, yb], color=palette[representation], alpha=.45, lw=.7 + 1.8 * abs(row["median_correlation"]))
        for site in sites:
            x, y = coordinates[site]; ax.scatter(x, y, s=35, facecolor="white", edgecolor="#222222", zorder=3); ax.text(x + 3, y, site.replace("roi_", ""), fontsize=7)
        ax.invert_yaxis(); ax.set_aspect("equal", adjustable="datalim"); ax.set(title=f"{representation.replace('_', ' ')}: {len(subset)} stable edges", xlabel="x (pixels)", ylabel="y (pixels)")
    fig.suptitle("Spatial layout of exploratory stable coactivity edges\nBH q≤0.05 and same correlation sign in at least 3/4 bursts", fontsize=14)
    fig.savefig(output / "cross_neural_spatial_networks.png", dpi=180); plt.close(fig)


def run_cross_neural_ica(
    records: list[ObservationRecord], trace_npz: Path, output: Path, *, frozen_ica_sha256: str,
    permutations: int = 2000,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    archive = np.load(trace_npz, allow_pickle=False)
    site_ids = archive["site_ids"]
    traces = {key: archive[key] for key in ("raw", "residual", "frozen_two_frame_ica")}
    sites, cubes, coordinates, length = event_cubes(records, site_ids, traces)
    rows, summaries = analyze_pairs(sites, cubes, coordinates, permutations=permutations)
    _write_tsv(partial / "pairwise_coactivity.tsv", rows)
    _write_tsv(partial / "site_coordinates.tsv", [{"site_id": s, "x_px": coordinates[s][0], "y_px": coordinates[s][1]} for s in sites])
    _figures(partial, sites, rows, coordinates)
    raw_by_pair = {(r["site_a"], r["site_b"]): r["median_correlation"] for r in rows if r["representation"] == "raw"}
    for representation in REPRESENTATIONS[1:]:
        values = [r for r in rows if r["representation"] == representation]
        summaries[representation]["pair_pattern_correlation_with_raw"] = _corr(
            np.asarray([raw_by_pair[(r["site_a"], r["site_b"])] for r in values]),
            np.asarray([r["median_correlation"] for r in values]),
        )
    edge_sets = {
        representation: {
            (r["site_a"], r["site_b"])
            for r in rows if r["representation"] == representation and r["stable_edge"]
        }
        for representation in REPRESENTATIONS
    }
    for representation in REPRESENTATIONS:
        values = [r for r in rows if r["representation"] == representation]
        correlations = np.asarray([r["median_correlation"] for r in values])
        distances = np.asarray([r["distance_px"] for r in values])
        summaries[representation].update({
            "positive_median_correlations": int(np.sum(correlations > 0)),
            "negative_median_correlations": int(np.sum(correlations < 0)),
            "distance_vs_absolute_correlation_pearson_r": _corr(distances, np.abs(correlations)),
        })
        if representation != "raw":
            intersection = edge_sets["raw"] & edge_sets[representation]
            union = edge_sets["raw"] | edge_sets[representation]
            summaries[representation].update({
                "stable_edge_overlap_with_raw": len(intersection),
                "stable_edge_jaccard_with_raw": len(intersection) / len(union) if union else 1.0,
            })
    summary = {
        "schema_version": 1, "status": "complete_exploratory", "recordings": 1,
        "sites": len(sites), "bursts": 4, "event_window_frames": length, "pairs_per_representation": len(sites) * (len(sites) - 1) // 2,
        "representations": summaries,
        "frozen_ica_sha256": frozen_ica_sha256,
        "null": {"kind": "independent within-window circular shifts by site and burst", "permutations_per_pair": permutations, "seed": 20260824, "multiple_testing": "Benjamini-Hochberg within representation"},
        "stable_edge_rule": "BH q <= 0.05 and same correlation sign in at least 3 of 4 bursts",
        "interpretation": "descriptive within-recording functional coactivity; not anatomical wiring, synaptic connectivity, or causality",
        "limitations": ["one recording", "four bursts", "calcium kinetics limit lag interpretation", "within-window circular-shift null is exploratory", "complete-site cohort only", "ICA is a frozen two-frame change operator nearly equivalent to signed difference"],
    }
    atomic_json(partial / "summary.json", summary)
    atomic_json(partial / "validation.json", {"status": "passed_with_caveats", "all_representations_same_sites_windows_pairs": True, "ica_hash_recorded": True, "finite_pair_metrics": bool(all(np.isfinite(float(r["median_correlation"])) for r in rows)), "scientific_audit": "reuses validated trace-atlas source media; derived network figures generated here", "publication_claim_ready": False})
    atomic_json(partial / "llm_context.json", {"entrypoint": "summary.json", "primary_table": "pairwise_coactivity.tsv", "primary_figures": ["cross_neural_representation_matrices.png", "cross_neural_representation_comparison.png", "cross_neural_spatial_networks.png"], "source_trace_archive": str(trace_npz.resolve()), "evidence_boundary": summary["interpretation"]})
    atomic_text(partial / "REPORT.md", "# Cross-neural coactivity with frozen ICA\n\nThis representation-matched analysis compares Raw, ROI-minus-annulus residual, leave-one-site-out global-adjusted Raw, and the hash-verified frozen two-frame ICA output for the same 14 immutable sites, four bursts, 24-frame event windows, and 91 site pairs. Stable edges are exploratory and require BH q <= 0.05 plus sign agreement in at least three bursts. They describe functional coactivity, not anatomical wiring, synapses, direction, or causality. See `summary.json` for results and limitations.\n")
    atomic_json(partial / "artifact_index.json", {"artifacts": sorted([p.name for p in partial.iterdir() if p.is_file()] + ["artifact_index.json"])})
    partial.replace(output)
    return summary
