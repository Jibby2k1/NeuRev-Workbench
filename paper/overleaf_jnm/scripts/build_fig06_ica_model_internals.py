#!/usr/bin/env python3
"""Export bounded ICA component diagnostics for all canonical-v7 sites."""
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
from neurobench.experiments.msln_msica.multilag_program import _fit_from_dict
from neurobench.portable_paths import data_root, media_root, portable_path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA_ROOT = data_root(REPO)
MEDIA_ROOT = media_root(REPO)
FINAL = ROOT / "figures" / "final"
MANIFEST = MEDIA_ROOT / "v7_priority_neuron_media_cs_parzen/manifest.json"
RUN = DATA_ROOT / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5"
CONFIG_ID = "delay_embedding__cs_parzen__long__bandwidth-0p25"
REVIEW_START_UI = 1800
REVIEW_STOP_UI = 2359


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    raw_path = Path(next(path for path in manifest["inputs"] if path.endswith("20ms.npy")))
    surface_path = RUN / "stage_a" / "surface.json"
    surface = json.loads(surface_path.read_text())
    rows = surface["calibration_rows"] + surface["expansion_rows"]
    matches = [row for row in rows if row.get("config_id") == CONFIG_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen fit for {CONFIG_ID}, found {len(matches)}")
    fit = _fit_from_dict(matches[0]["fit"])

    sites_by_id = {}
    for item in manifest["items"]:
        site_id = f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"
        sites_by_id.setdefault(site_id, (item["y_int"], item["x_int"]))
    site_ids = sorted(sites_by_id)
    sites = np.asarray([sites_by_id[site] for site in site_ids], dtype=np.int32)
    history = max(fit.lags)
    movie = np.load(raw_path, mmap_mode="r")
    start = REVIEW_START_UI - 1 - history
    stop = REVIEW_STOP_UI
    audit = project_temporal_fit_at_sites(movie[start:stop], fit, sites)
    if audit.components.shape[1] != REVIEW_STOP_UI - REVIEW_START_UI + 1:
        raise RuntimeError("review timeline alignment failed")

    FINAL.mkdir(parents=True, exist_ok=True)
    npz_path = FINAL / "fig06_ica_model_internals.npz"
    np.savez_compressed(
        npz_path,
        site_ids=np.asarray(site_ids), sites_yx=sites, components=audit.components,
        embedded_input=audit.embedded_input,
        reconstructed_embedding=audit.reconstructed_embedding,
        embedding_reconstruction_residual=audit.embedding_reconstruction_residual,
        component_energy_fraction=audit.residual_component_energy_fraction,
        mixing=audit.mixing, demixing=fit.demixing,
    )

    component_energy = np.mean(audit.residual_component_energy_fraction, axis=1)
    tsv_path = FINAL / "fig06_ica_component_energy_by_site.tsv"
    with tsv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["site_id", "y_px", "x_px", *[f"component_{i}_energy_fraction" for i in range(len(fit.lags))]])
        for site_id, (y, x), fractions in zip(site_ids, sites, component_energy.T):
            writer.writerow([site_id, int(y), int(x), *map(float, fractions)])

    residual_rms = np.sqrt(np.mean(np.square(audit.embedding_reconstruction_residual), axis=(0, 1)))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), constrained_layout=True)
    image = axes[0].imshow(fit.demixing, cmap="coolwarm", aspect="auto")
    axes[0].set(title="Frozen demixing matrix", xlabel="Lag input", ylabel="Component")
    axes[0].set_xticks(range(len(fit.lags)), labels=fit.lags)
    fig.colorbar(image, ax=axes[0], shrink=0.78)
    axes[1].boxplot(component_energy.T, showfliers=False)
    axes[1].set(title="Energy fraction across sites", xlabel="Component", ylabel="Mean fraction")
    axes[1].set_xticks(range(1, len(fit.lags) + 1), labels=range(len(fit.lags)))
    axes[2].hist(residual_rms, bins=14, color="#315f72")
    axes[2].set(title="Embedding inversion residual", xlabel="RMS per site", ylabel="Sites")
    fig.suptitle("Long-delay CS–Parzen ICA model internals (50 canonical-v7 sites)")
    png_path = FINAL / "fig06_ica_model_internals.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "observed_from_frozen_fit",
        "config_id": CONFIG_ID,
        "objective_family": fit.objective_family,
        "objective_parameter": fit.objective_parameter,
        "lags_frames": list(fit.lags),
        "site_count": len(site_ids),
        "frame_count": audit.components.shape[1],
        "component_count": audit.components.shape[0],
        "persistence_index": fit.persistence_index,
        "innovation_index": fit.innovation_index,
        "residual_indices": list(fit.residual_indices),
        "embedding_reconstruction_rms_max": float(np.max(residual_rms)),
        "interpretation_boundary": "Embedding inversion residual is a numerical transform audit, not a denoising or biological reconstruction error.",
        "inputs": {portable_path(path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT): sha256(path) for path in (MANIFEST, raw_path, surface_path)},
        "outputs": [portable_path(path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT) for path in (png_path, npz_path, tsv_path)],
    }
    (FINAL / "fig06_ica_model_internals.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
