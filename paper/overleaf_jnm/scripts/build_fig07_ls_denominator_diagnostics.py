#!/usr/bin/env python3
"""Reproduce the frozen LS denominator at canonical-v7 sites exactly."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-mpl")
import matplotlib.pyplot as plt
import numpy as np

from neurobench.algorithms.multilag_msica import project_temporal_fit_at_sites
from neurobench.experiments.msln_msica.multilag_program import _fit_from_dict, _source
from neurobench.portable_paths import data_root, media_root, portable_path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA_ROOT = data_root(REPO)
MEDIA_ROOT = media_root(REPO)
FINAL = ROOT / "figures" / "final"
MANIFEST = MEDIA_ROOT / "v7_priority_neuron_media_cs_parzen/manifest.json"
RUN = DATA_ROOT / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5"
DIAGNOSTIC = DATA_ROOT / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics"
CONFIG_ID = "delay_embedding__cs_parzen__long__bandwidth-0p25"
LANE_ID = CONFIG_ID + "::residual_group::joint_s15_g5_t31_g1"
OUTER, GUARD, WINDOW, TEMPORAL_GUARD = 15, 5, 31, 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def residual_group_at_sites(values: np.ndarray, fit, sites: np.ndarray, chunk: int = 1000) -> np.ndarray:
    result = np.empty((len(values) - max(fit.lags), len(sites)), dtype=np.float32)
    residual = np.asarray(fit.residual_indices, dtype=np.int64)
    for start in range(0, len(sites), chunk):
        stop = min(start + chunk, len(sites))
        audit = project_temporal_fit_at_sites(values, fit, sites[start:stop])
        result[:, start:stop] = np.sqrt(np.sum(np.square(audit.components[residual]), axis=0)).astype(np.float32)
    return result


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    config_path = RUN / "config.resolved.json"
    surface_path = RUN / "stage_a" / "surface.json"
    shard_path = RUN / "stage_b" / "shards" / f"{CONFIG_ID}.json"
    config = json.loads(config_path.read_text())
    surface = json.loads(surface_path.read_text())
    matches = [row for row in surface["expansion_rows"] if row["config_id"] == CONFIG_ID]
    if len(matches) != 1:
        raise RuntimeError("frozen fit is not unique")
    fit = _fit_from_dict(matches[0]["fit"])
    lane_rows = [row for row in json.loads(shard_path.read_text())["rows"] if row["lane_id"] == LANE_ID]
    if len(lane_rows) != 1:
        raise RuntimeError("frozen pipeline lane is not unique")
    floor = float(lane_rows[0]["msln_scale_floor"])

    sites_by_id = {}
    for item in manifest["items"]:
        site_id = f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"
        sites_by_id.setdefault(site_id, (item["y_int"], item["x_int"]))
    site_ids = sorted(sites_by_id)
    centers = np.asarray([sites_by_id[key] for key in site_ids], dtype=np.int32)
    height, width = np.load(config["source"]["movie_path"], mmap_mode="r").shape[1:]
    support = sorted({
        (yy, xx) for y, x in centers
        for yy in range(max(0, y - OUTER // 2), min(height, y + OUTER // 2 + 1))
        for xx in range(max(0, x - OUTER // 2), min(width, x + OUTER // 2 + 1))
    })
    support_sites = np.asarray(support, dtype=np.int32)
    support_index = {tuple(site): index for index, site in enumerate(support_sites)}

    values, _, global_history, pre_roll = _source(config)
    projected = residual_group_at_sites(values, fit, support_sites)
    offset = global_history - max(fit.lags)
    branch = projected[offset:]
    if len(branch) != pre_roll + 560:
        raise RuntimeError("frozen branch alignment failed")

    numerator = np.empty((560, len(centers)), dtype=np.float32)
    scale = np.empty_like(numerator)
    for column, (y, x) in enumerate(centers):
        outer = [(yy, xx) for yy in range(max(0, y - 7), min(height, y + 8)) for xx in range(max(0, x - 7), min(width, x + 8))]
        guard = {(yy, xx) for yy in range(max(0, y - 2), min(height, y + 3)) for xx in range(max(0, x - 2), min(width, x + 3))}
        annulus = [support_index[pixel] for pixel in outer if pixel not in guard]
        spatial = np.asarray(branch[:, annulus], dtype=np.float64)
        center = np.asarray(branch[:, support_index[(int(y), int(x))]], dtype=np.float64)
        for review_t in range(560):
            t = pre_roll + review_t
            reference = spatial[t - WINDOW:t - TEMPORAL_GUARD]
            mean = float(np.mean(reference))
            numerator[review_t, column] = center[t] - mean
            scale[review_t, column] = np.std(reference, ddof=0)
    denominator = np.maximum(scale, floor)
    normalized = numerator / denominator
    floor_applied = scale < floor

    cached_ica = np.load(DIAGNOSTIC / "cache" / "recovery_msica.npy", mmap_mode="r")
    cached_ls = np.load(DIAGNOSTIC / "cache" / "recovery_msln.npy", mmap_mode="r")
    yy, xx = centers.T
    ica_error = normalized * 0  # shape-safe allocation
    ica_error[:] = branch[pre_roll:, [support_index[tuple(site)] for site in centers]] - cached_ica[:, yy, xx]
    ls_error = normalized - cached_ls[:, yy, xx]

    FINAL.mkdir(parents=True, exist_ok=True)
    npz_path = FINAL / "fig07_ls_denominator_diagnostics.npz"
    np.savez_compressed(npz_path, site_ids=np.asarray(site_ids), sites_yx=centers,
                        numerator=numerator, local_scale=scale, denominator=denominator,
                        floor_applied=floor_applied, normalized_reconstruction=normalized,
                        cached_ls_center=cached_ls[:, yy, xx], ica_center_error=ica_error,
                        ls_center_error=ls_error, scale_floor=np.asarray(floor))
    tsv_path = FINAL / "fig07_ls_denominator_by_site.tsv"
    with tsv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["site_id", "y_px", "x_px", "denominator_median", "denominator_p10", "denominator_p90", "floor_fraction", "ls_reconstruction_rmse"])
        for index, (site_id, (y, x)) in enumerate(zip(site_ids, centers)):
            writer.writerow([site_id, int(y), int(x), float(np.median(denominator[:, index])),
                             float(np.percentile(denominator[:, index], 10)), float(np.percentile(denominator[:, index], 90)),
                             float(np.mean(floor_applied[:, index])), float(np.sqrt(np.mean(np.square(ls_error[:, index]))))])

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    axes[0].boxplot(denominator, showfliers=False)
    axes[0].axhline(floor, color="#b8493f", linestyle="--", label=f"floor={floor:.3f}")
    axes[0].set(title="True LS denominator by site", xlabel="Canonical-v7 site", ylabel="Denominator")
    axes[0].set_xticks([]); axes[0].legend(frameon=False)
    floor_fraction = np.mean(floor_applied, axis=0)
    axes[1].hist(floor_fraction, bins=np.linspace(0, 1, 16), color="#8a6d3b")
    axes[1].set(title="Scale-floor usage", xlabel="Fraction of review frames", ylabel="Sites")
    axes[2].hist(np.sqrt(np.mean(np.square(ls_error), axis=0)), bins=14, color="#315f72")
    axes[2].set(title="Reproduction error vs saved LS", xlabel="RMS per site", ylabel="Sites")
    fig.suptitle("Frozen local-standardization denominator audit (50 sites)")
    png_path = FINAL / "fig07_ls_denominator_diagnostics.png"
    fig.savefig(png_path, dpi=220); plt.close(fig)

    summary = {
        "status": "reproduced_from_frozen_fit_and_scale_floor",
        "lane_id": LANE_ID, "site_count": len(site_ids), "support_pixel_count": len(support_sites),
        "frame_count": 560, "causal_pre_roll_frames": pre_roll, "scale_floor": floor,
        "median_site_denominator": float(np.median(denominator)),
        "median_site_floor_fraction": float(np.median(floor_fraction)),
        "ica_center_max_abs_error": float(np.max(np.abs(ica_error))),
        "ls_center_max_abs_error": float(np.max(np.abs(ls_error))),
        "ls_center_rmse": float(np.sqrt(np.mean(np.square(ls_error)))),
        "inputs": {portable_path(path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT): sha256(path) for path in (MANIFEST, config_path, surface_path, shard_path, DIAGNOSTIC / "cache" / "recovery_msica.npy", DIAGNOSTIC / "cache" / "recovery_msln.npy")},
        "outputs": [portable_path(path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT) for path in (png_path, npz_path, tsv_path)],
    }
    (FINAL / "fig07_ls_denominator_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
