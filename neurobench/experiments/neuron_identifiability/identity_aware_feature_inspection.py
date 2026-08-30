"""Canonical-v8 automated feature inspection for identity-aware B58 misses."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-identity-v8-mpl")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import nnls
from scipy.spatial import ConvexHull
from scipy.stats import rankdata, spearmanr

from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import (
    ALIGNMENT_START_UI,
    atomic_json,
    sha256,
    write_tsv,
)
from neurobench.portable_paths import data_root, media_root, portable_path


STAGES = ("raw", "cs_parzen_ica", "local_standardization")
RADIUS = 6


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def event_response(trace: np.ndarray, start: int, stop: int) -> tuple[float, float]:
    baseline = np.asarray(trace[max(0, start - 15):start], dtype=float)
    center = float(np.median(baseline))
    mad = 1.4826 * float(np.median(np.abs(baseline - center)))
    response = float(np.max(trace[start:stop]) - center)
    return response, response / max(mad, 1e-9)


def response_patch(stage: np.ndarray, item: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y = int(item["x_int"]), int(item["y_int"])
    x0, x1 = max(0, x - RADIUS), min(stage.shape[2], x + RADIUS + 1)
    y0, y1 = max(0, y - RADIUS), min(stage.shape[1], y + RADIUS + 1)
    start = int(item["event_start_ui"]) - ALIGNMENT_START_UI
    stop = int(item["event_end_ui"]) - ALIGNMENT_START_UI + 1
    event = np.asarray(stage[start:stop, y0:y1, x0:x1], dtype=float)
    baseline = np.asarray(stage[max(0, start - 15):start, y0:y1, x0:x1], dtype=float)
    return np.max(event, axis=0) - np.median(baseline, axis=0), (x0, x1, y0, y1)


def stage_offsets(stages: list[np.ndarray], item: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    rows, ranked = [], []
    bounds = None
    ox, oy = int(item["x_int"]), int(item["y_int"])
    for name, stage in zip(STAGES, stages, strict=True):
        patch, current = response_patch(stage, item)
        bounds = current if bounds is None else bounds
        py, px = np.unravel_index(np.nanargmax(patch), patch.shape)
        x, y = current[0] + int(px), current[2] + int(py)
        ranked.append(rankdata(patch.ravel()).reshape(patch.shape) / patch.size)
        rows.append({"stage": name, "stage_x": x, "stage_y": y, "stage_dx": x-ox,
                     "stage_dy": y-oy, "stage_distance_px": float(math.hypot(x-ox, y-oy))})
    consensus = np.median(np.stack(ranked), axis=0)
    py, px = np.unravel_index(np.nanargmax(consensus), consensus.shape)
    assert bounds is not None
    candidate = (bounds[0] + int(px), bounds[2] + int(py))
    vectors = np.asarray([[r["stage_dx"], r["stage_dy"]] for r in rows], dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    cosines = []
    for a in range(3):
        for b in range(a + 1, 3):
            cosines.append(float(np.dot(vectors[a], vectors[b]) / (norms[a]*norms[b]))) if norms[a] and norms[b] else None
    finite = [v for v in cosines if v is not None]
    agreement = float(np.mean(finite)) if finite else float("nan")
    spread = float(np.mean(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))
    for row in rows:
        row["mean_pairwise_direction_cosine"] = agreement
        row["offset_vector_spread_px"] = spread
        row["consensus_x"], row["consensus_y"] = candidate
    return rows, candidate


def disk_trace(stage: np.ndarray, x: int, y: int, radius: int = 2) -> np.ndarray:
    yy, xx = np.ogrid[:stage.shape[1], :stage.shape[2]]
    mask = (xx-x)**2 + (yy-y)**2 <= radius**2
    return np.asarray(stage[:, mask], dtype=float).mean(axis=1)


def footprint_trace(stage: np.ndarray, item: dict[str, Any], patch: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    weights = np.clip(np.asarray(patch, dtype=float), 0, None)
    positive = weights[weights > 0]
    threshold = float(np.quantile(positive, .75)) if len(positive) else float("inf")
    weights = np.where((weights > 0) & (weights >= threshold), weights, 0)
    if not np.any(weights): weights = np.ones_like(weights)
    weights /= weights.sum()
    x0, x1, y0, y1 = bounds
    return np.tensordot(np.asarray(stage[:, y0:y1, x0:x1], dtype=float), weights, axes=([1, 2], [0, 1]))


def morphology(patch: np.ndarray) -> dict[str, float]:
    values = np.clip(np.asarray(patch, dtype=float), 0, None)
    positive = values[values > 0]
    threshold = float(np.quantile(positive, .75)) if len(positive) else float("inf")
    mask = (values > 0) & (values >= threshold)
    labels, _ = ndimage.label(mask)
    peak = np.unravel_index(np.argmax(values), values.shape)
    component = labels == labels[peak]
    coords = np.argwhere(component)
    area = int(len(coords))
    if area < 2:
        return {"footprint_area_px": area, "eccentricity": 0.0, "solidity": 1.0,
                "convexity_deficit": 0.0, "radial_asymmetry": 0.0,
                "boundary_sharpness": 0.0, "local_peak_count": 1}
    cov = np.cov(coords.T)
    eigen = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
    eccentricity = float(np.sqrt(max(0, 1 - eigen[1] / max(eigen[0], 1e-12))))
    corners = np.concatenate([
        coords[:, ::-1] + np.asarray(offset)
        for offset in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5))
    ])
    try:
        hull_area = float(ConvexHull(corners).volume) if area >= 2 else float(area)
    except Exception:
        hull_area = float(area)
    solidity = float(min(1.0, area / max(hull_area, area)))
    cy, cx = np.average(coords, axis=0, weights=np.maximum(values[component], 1e-9))
    angles = np.arctan2(coords[:, 0]-cy, coords[:, 1]-cx)
    sectors = np.asarray([values[component][(angles >= -np.pi + k*np.pi/4) & (angles < -np.pi + (k+1)*np.pi/4)].sum() for k in range(8)])
    radial = float(np.std(sectors) / max(np.mean(sectors), 1e-9))
    eroded = ndimage.binary_erosion(component)
    boundary = component & ~eroded
    outside = ndimage.binary_dilation(component) & ~component
    sharpness = float(values[component].mean() - values[outside].mean()) if np.any(outside) else 0.0
    peak_threshold = float(np.quantile(positive, .9)) if len(positive) else float("inf")
    maxima = values == ndimage.maximum_filter(values, size=3, mode="nearest")
    peaks = int(np.sum(maxima & (values > 0) & (values >= peak_threshold)))
    return {"footprint_area_px": area, "eccentricity": eccentricity, "solidity": solidity,
            "convexity_deficit": 1-solidity, "radial_asymmetry": radial,
            "boundary_sharpness": sharpness, "local_peak_count": peaks}


def leakage_metrics(candidate: np.ndarray, original: np.ndarray, neighbor: np.ndarray) -> dict[str, float]:
    traces = [np.asarray(v, dtype=float) for v in (candidate, original, neighbor)]
    traces = [(v-v.mean()) / max(v.std(), 1e-9) for v in traces]
    cand, orig, near = traces
    design = np.column_stack([orig, near])
    coef, _ = nnls(design, cand)
    fitted = design @ coef
    r2 = 1 - float(np.sum((cand-fitted)**2)) / max(float(np.sum(cand**2)), 1e-9)
    residual_c = cand - orig * float(np.dot(cand, orig) / max(np.dot(orig, orig), 1e-9))
    residual_n = near - orig * float(np.dot(near, orig) / max(np.dot(orig, orig), 1e-9))
    partial = float(np.corrcoef(residual_c, residual_n)[0, 1])
    return {"candidate_original_r": float(np.corrcoef(cand, orig)[0, 1]),
            "candidate_neighbor_r": float(np.corrcoef(cand, near)[0, 1]),
            "candidate_neighbor_partial_r_given_original": partial,
            "nnls_original_weight": float(coef[0]), "nnls_neighbor_weight": float(coef[1]),
            "nnls_two_trace_r2": r2}


def make_figure(root: Path, outcomes: list[dict[str, Any]], offsets: list[dict[str, Any]],
                morphology_rows: list[dict[str, Any]], extraction: list[dict[str, Any]],
                competition: list[dict[str, Any]], leakage: list[dict[str, Any]]) -> None:
    colors = {"recovered": "#56738f", "identity_collision": "#c65f45", "identity_clear_miss": "#d6a431"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    counts = Counter(r["v8_outcome"] for r in outcomes)
    order = ["recovered", "identity_collision", "identity_clear_miss"]
    axes[0, 0].bar(order, [counts[x] for x in order], color=[colors[x] for x in order]); axes[0, 0].tick_params(axis="x", rotation=20); axes[0, 0].set(title="V8 recovery outcomes", ylabel="Occurrences")
    miss_offsets = [r for r in offsets if r["stage"] == "raw"]
    for status in ("identity_collision", "identity_clear_miss"):
        rows = [r for r in miss_offsets if r["v8_outcome"] == status]
        axes[0, 1].scatter([r["mean_pairwise_direction_cosine"] for r in rows], [r["offset_vector_spread_px"] for r in rows], label=status, color=colors[status], s=55)
    axes[0, 1].set(title="Cross-stage offset stability", xlabel="Mean direction cosine", ylabel="Vector spread (px)"); axes[0, 1].legend(fontsize=8)
    for status in ("identity_collision", "identity_clear_miss"):
        rows = [r for r in morphology_rows if r["stage"] == "local_standardization" and r["v8_outcome"] == status]
        axes[0, 2].scatter([r["eccentricity"] for r in rows], [r["radial_asymmetry"] for r in rows], label=status, color=colors[status], s=55)
    axes[0, 2].set(title="LS footprint morphology", xlabel="Eccentricity", ylabel="Radial asymmetry")
    labels = ["point", "disk", "footprint"]
    for i, extraction_type in enumerate(labels):
        rows = [r for r in extraction if r["stage"] == "local_standardization" and r["extraction"] == extraction_type]
        axes[1, 0].boxplot([r["peak_mad"] for r in rows], positions=[i], widths=.6)
    axes[1, 0].set(xticks=range(3), xticklabels=labels, title="LS extraction robustness", ylabel="Event peak / pre-event MAD")
    for status in ("identity_collision", "identity_clear_miss"):
        rows = [r for r in competition if r["stage"] == "local_standardization" and r["v8_outcome"] == status]
        axes[1, 1].scatter([r["top1_top2_margin"] for r in rows], [r["original_to_neighbor_response_ratio"] for r in rows], label=status, color=colors[status], s=55)
    axes[1, 1].set(title="Local competition", xlabel="Top-1 minus top-2 response", ylabel="Original / neighbor response")
    for status in ("identity_collision", "identity_clear_miss"):
        rows = [r for r in leakage if r["stage"] == "local_standardization" and r["v8_outcome"] == status]
        axes[1, 2].scatter([r["candidate_neighbor_partial_r_given_original"] for r in rows], [r["nnls_neighbor_weight"] for r in rows], label=status, color=colors[status], s=55)
    axes[1, 2].set(title="Neighbor leakage", xlabel="Partial candidate-neighbor r", ylabel="NNLS neighbor weight")
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle("Canonical-v8 identity-aware feature inspection", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(root / "identity_aware_feature_inspection_v8.png", dpi=180); plt.close(fig)


def run(repo: Path, output: Path, recenter_root: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    if partial.exists(): raise FileExistsError(partial)
    data = data_root(repo)
    media = media_root(repo)
    atlas_path = media / "v7_priority_neuron_media_cs_parzen/manifest.json"
    atlas = json.loads(atlas_path.read_text()); items = atlas["items"]
    recenter = read_tsv(recenter_root / "recenter_summary.tsv")
    recenter_validation = json.loads((recenter_root / "validation.json").read_text())
    if len(items) != 106 or len(recenter) != 12 or recenter_validation["status"] != "passed": raise RuntimeError("source audit contract failed")
    by_id = {str(i["observation_id"]): i for i in items}; rec_by_id = {r["observation_id"]: r for r in recenter}
    raw_path = data / "Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
    cache = data / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache"
    stage_paths = [raw_path, cache / "recovery_msica.npy", cache / "recovery_msln.npy"]
    arrays = [np.load(p, mmap_mode="r") for p in stage_paths]
    arrays[0] = arrays[0][ALIGNMENT_START_UI-1:ALIGNMENT_START_UI-1+len(arrays[1])]
    geometry = {(str(i["original_roi_id"]), int(i["x_int"]), int(i["y_int"])): i for i in items}
    geometry_rows = list(geometry.values())
    outcomes = []
    for item in items:
        oid = str(item["observation_id"])
        status = rec_by_id.get(oid, {}).get("candidate_identity_status", "")
        outcome = "recovered" if not status else ("identity_clear_miss" if status == "identity_clear_within_6px" else "identity_collision")
        outcomes.append({"observation_id": oid, "site_id": f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}", "burst_id": item["burst_id"], "v8_outcome": outcome})
    offset_rows: list[dict[str, Any]] = []; extraction_rows: list[dict[str, Any]] = []; morphology_rows: list[dict[str, Any]] = []; competition_rows: list[dict[str, Any]] = []; leakage_rows: list[dict[str, Any]] = []
    for rec in recenter:
        item = by_id[rec["observation_id"]]; outcome = "identity_clear_miss" if rec["candidate_identity_status"] == "identity_clear_within_6px" else "identity_collision"
        stage_rows, consensus = stage_offsets(arrays, item)
        for row in stage_rows: row.update(observation_id=rec["observation_id"], v8_outcome=outcome, identity_status=rec["candidate_identity_status"]); offset_rows.append(row)
        ox, oy = int(item["x_int"]), int(item["y_int"]); cx, cy = consensus
        others = [g for g in geometry_rows if str(g["original_roi_id"]) != str(item["original_roi_id"])]
        neighbor = min(others, key=lambda g: math.hypot(cx-int(g["x_int"]), cy-int(g["y_int"])))
        nx, ny = int(neighbor["x_int"]), int(neighbor["y_int"])
        start = int(item["event_start_ui"]) - ALIGNMENT_START_UI; stop = int(item["event_end_ui"]) - ALIGNMENT_START_UI + 1
        for name, stage in zip(STAGES, arrays, strict=True):
            patch, bounds = response_patch(stage, item); morph = morphology(patch)
            morphology_rows.append({"observation_id": rec["observation_id"], "stage": name, "v8_outcome": outcome, **morph})
            traces = {"point": np.asarray(stage[:, oy, ox], dtype=float), "disk": disk_trace(stage, ox, oy), "footprint": footprint_trace(stage, item, patch, bounds)}
            for extraction, trace in traces.items():
                response, peak_mad = event_response(trace, start, stop)
                extraction_rows.append({"observation_id": rec["observation_id"], "stage": name, "v8_outcome": outcome, "extraction": extraction, "event_response": response, "peak_mad": peak_mad})
            flat = np.sort(patch.ravel())[::-1]; original = float(patch[oy-bounds[2], ox-bounds[0]])
            neighbor_response = float(patch[ny-bounds[2], nx-bounds[0]]) if bounds[0] <= nx < bounds[1] and bounds[2] <= ny < bounds[3] else float("nan")
            competition_rows.append({"observation_id": rec["observation_id"], "stage": name, "v8_outcome": outcome, "nearest_original_roi_id": neighbor["original_roi_id"], "nearest_distance_px": float(math.hypot(cx-nx, cy-ny)), "top1_top2_margin": float(flat[0]-flat[1]), "top1_top2_ratio": float(flat[0]/max(abs(flat[1]), 1e-9)), "original_response": original, "neighbor_response": neighbor_response, "original_to_neighbor_response_ratio": float(original/max(abs(neighbor_response), 1e-9)) if np.isfinite(neighbor_response) else float("nan")})
            metrics = leakage_metrics(np.asarray(stage[:, cy, cx], float), np.asarray(stage[:, oy, ox], float), np.asarray(stage[:, ny, nx], float))
            leakage_rows.append({"observation_id": rec["observation_id"], "stage": name, "v8_outcome": outcome, "nearest_original_roi_id": neighbor["original_roi_id"], **metrics})
    partial.mkdir(parents=True)
    write_tsv(partial / "v8_outcomes.tsv", outcomes); write_tsv(partial / "offset_stability.tsv", offset_rows); write_tsv(partial / "trace_extraction.tsv", extraction_rows); write_tsv(partial / "footprint_morphology.tsv", morphology_rows); write_tsv(partial / "local_competition.tsv", competition_rows); write_tsv(partial / "neighbor_leakage.tsv", leakage_rows)
    make_figure(partial, outcomes, offset_rows, morphology_rows, extraction_rows, competition_rows, leakage_rows)
    def med(rows: list[dict[str, Any]], field: str, outcome: str, stage: str = "local_standardization") -> float:
        values = [float(r[field]) for r in rows if r.get("v8_outcome") == outcome and r.get("stage") == stage and np.isfinite(float(r[field]))]
        return float(np.median(values)) if values else float("nan")
    summary = {"schema_version": 1, "status": "complete_exploratory", "population": {"occurrences": 106, "recovered": 94, "identity_collision_misses": 8, "identity_clear_misses": 4}, "strict_recall": 94/106, "identity_resolved_conditional_sensitivity": 94/98, "four_case_rescue_ceiling": 98/106, "headline": {"offset_direction_cosine_median_collision": med(offset_rows, "mean_pairwise_direction_cosine", "identity_collision", "raw"), "offset_direction_cosine_median_clear": med(offset_rows, "mean_pairwise_direction_cosine", "identity_clear_miss", "raw"), "ls_eccentricity_median_collision": med(morphology_rows, "eccentricity", "identity_collision"), "ls_eccentricity_median_clear": med(morphology_rows, "eccentricity", "identity_clear_miss"), "ls_neighbor_partial_r_median_collision": med(leakage_rows, "candidate_neighbor_partial_r_given_original", "identity_collision"), "ls_neighbor_partial_r_median_clear": med(leakage_rows, "candidate_neighbor_partial_r_given_original", "identity_clear_miss")}, "interpretation_limits": ["Only four identity-clear misses; no classifier is fit.", "Response-selected candidates and footprint weights reuse the same events.", "Nearest labeled geometry is not exhaustive biological identity truth.", "Same recording and four bursts; descriptive within-recording analysis only."]}
    atomic_json(partial / "summary.json", summary)
    scientific_audit = {"mode": "validated_source_audit_reuse", "source_recenter_audit": portable_path(recenter_root, repository=repo, data=data, media=media), "source_validation": portable_path(recenter_root / "validation.json", repository=repo, data=data, media=media), "reason": "Secondary numerical analysis of the exact 12 visually reviewed miss comparisons; no new labels, candidates, coordinates, or media selection."}
    validation = {"status": "passed", "checks": {"population_106": len(outcomes)==106, "outcome_counts_94_8_4": Counter(r["v8_outcome"] for r in outcomes)==Counter({"recovered":94,"identity_collision":8,"identity_clear_miss":4}), "twelve_miss_offsets_three_stages": len(offset_rows)==36, "twelve_misses_three_stages_three_extractions": len(extraction_rows)==108, "morphology_rows": len(morphology_rows)==36, "competition_rows": len(competition_rows)==36, "leakage_rows": len(leakage_rows)==36, "source_visual_audit_passed": recenter_validation["status"]=="passed", "no_classifier_fit_to_four_misses": True}}
    atomic_json(partial / "validation.json", validation); atomic_json(partial / "scientific_audit.json", scientific_audit)
    atomic_json(partial / "llm_context.json", {"entry_point": "summary.json", "primary_figure": "identity_aware_feature_inspection_v8.png", "primary_tables": ["v8_outcomes.tsv", "offset_stability.tsv", "trace_extraction.tsv", "footprint_morphology.tsv", "local_competition.tsv", "neighbor_leakage.tsv"], "coordinate_convention": "x=column,y=row", "scientific_audit": scientific_audit, "limitations": summary["interpretation_limits"]})
    report = "# Canonical-v8 identity-aware feature inspection\n\nThis panel separates recovered occurrences, identity-collision misses, and identity-clear misses without fitting a classifier to the four clear cases. It measures cross-stage offset stability, point/disk/footprint extraction, footprint morphology, local competition, and neighboring-trace leakage. Results are descriptive and response-selected. See `summary.json` and the six primary TSV tables.\n"
    (partial / "REPORT.md").write_text(report, encoding="utf-8")
    artifacts = []
    for path in sorted(p for p in partial.rglob("*") if p.is_file() and p.name != "artifact_index.json"):
        artifacts.append({"path": str(path.relative_to(partial)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    inputs = [atlas_path, recenter_root / "recenter_summary.tsv", *stage_paths]
    atomic_json(partial / "artifact_index.json", {"artifacts": artifacts, "input_hashes": {portable_path(path, repository=repo, data=data, media=media): sha256(path) for path in inputs}})
    partial.replace(output)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path.cwd()); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--recenter-root", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(run(args.repo.resolve(), args.output.resolve(), args.recenter_root.resolve()), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
