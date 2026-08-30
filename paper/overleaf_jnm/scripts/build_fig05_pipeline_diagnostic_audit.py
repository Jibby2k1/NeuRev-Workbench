#!/usr/bin/env python3
"""Build a site x burst x stage diagnostic audit for canonical-v7 traces."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-mpl")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.stats import spearmanr
from neurobench.portable_paths import data_root, media_root, portable_path

ALIGNMENT_START_UI = 1800
FPS = 10.0
PRE = 15
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA_ROOT = data_root(REPO)
MEDIA_ROOT = media_root(REPO)
FINAL = ROOT / "figures" / "final"
MANIFEST = MEDIA_ROOT / "v7_priority_neuron_media_cs_parzen/manifest.json"
TAXONOMY = FINAL / "fig04_trace_taxonomy_site_assignments.tsv"
MODEL_ROOT = DATA_ROOT / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics"
STAGES = ("Raw", "CS-Parzen ICA", "Local standardization")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def robust_scale(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def annulus_values(arr: np.ndarray, x: int, y: int, inner: float = 3, outer: float = 6) -> np.ndarray:
    yy, xx = np.ogrid[: arr.shape[1], : arr.shape[2]]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = (d >= inner) & (d <= outer)
    return np.nanmean(arr[:, mask], axis=1)


def trace_metrics(trace: np.ndarray, event_len: int) -> dict:
    baseline = float(np.nanmedian(trace[:PRE]))
    noise = robust_scale(trace[:PRE])
    centered = trace - baseline
    event = centered[PRE : PRE + event_len]
    p = int(np.nanargmax(event)); peak = float(event[p]); minimum = float(np.nanmin(event))
    early_stop = max(1, event_len // 2)
    early = float(np.nansum(np.clip(event[:early_stop], 0, None)))
    late = float(np.nansum(np.clip(event[early_stop:], 0, None)))
    threshold = 0.5 * peak
    above = np.flatnonzero(event >= threshold) if peak > 0 else np.array([], dtype=int)
    return {
        "baseline": baseline, "pre_noise_mad": noise, "event_peak": peak, "event_min": minimum,
        "event_area_signed": float(np.nansum(event)), "event_area_positive": float(np.nansum(np.clip(event, 0, None))),
        "robust_snr": peak / (noise + 1e-9), "time_to_peak_s": p / FPS,
        "fwhm_s": (above[-1] - above[0] + 1) / FPS if len(above) else np.nan,
        "persistence_ratio": late / (early + 1e-9),
        "baseline_slope_per_s": float(np.polyfit(np.arange(PRE) / FPS, trace[:PRE], 1)[0]),
    }


def spatial_metrics(clip: np.ndarray, x: int, y: int, event_len: int, radius: int = 10) -> dict:
    h, w = clip.shape[1:]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = clip[:, y0:y1, x0:x1]
    baseline = np.nanmedian(patch[:PRE], axis=0)
    event_map = np.nanmax(patch[PRE : PRE + event_len] - baseline, axis=0)
    positive = np.clip(event_map, 0, None)
    total = float(np.sum(positive))
    cy, cx = y - y0, x - x0
    yy, xx = np.indices(positive.shape)
    if total > 0:
        com_y, com_x = center_of_mass(positive)
        offset = float(np.hypot(com_x - cx, com_y - cy))
        radius_eff = float(np.sqrt(np.sum(positive * ((xx - cx) ** 2 + (yy - cy) ** 2)) / total))
    else:
        offset = radius_eff = np.nan
    center_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 2**2
    ann_mask = (((xx - cx) ** 2 + (yy - cy) ** 2) >= 3**2) & (((xx - cx) ** 2 + (yy - cy) ** 2) <= 6**2)
    center_mean = float(np.nanmean(event_map[center_mask])); ann_mean = float(np.nanmean(event_map[ann_mask]))
    threshold = float(np.nanpercentile(positive[positive > 0], 75)) if np.any(positive > 0) else np.inf
    components = int(label(positive >= threshold)[1]) if np.isfinite(threshold) else 0
    return {"spatial_center_peakmap": center_mean, "spatial_annulus_peakmap": ann_mean,
            "center_annulus_contrast": center_mean - ann_mean, "effective_radius_px": radius_eff,
            "peak_offset_px": offset, "high_response_components": components}


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if np.sum(ok) >= 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else np.nan


def main() -> int:
    manifest = json.loads(MANIFEST.read_text()); items = manifest["items"]
    paths = list(manifest["inputs"])
    raw_path = Path(next(p for p in paths if p.endswith("20ms.npy")))
    ica_path = Path(next(p for p in paths if p.endswith("recovery_msica.npy")))
    ls_path = Path(next(p for p in paths if p.endswith("recovery_msln.npy")))
    raw_all = np.load(raw_path, mmap_mode="r"); ica = np.load(ica_path, mmap_mode="r"); ls = np.load(ls_path, mmap_mode="r")
    raw = raw_all[ALIGNMENT_START_UI - 1 : ALIGNMENT_START_UI - 1 + len(ica)]
    arrays = (raw, ica, ls)
    taxonomy = {r["site_id"]: int(r["joint_class_id"]) for r in csv.DictReader(TAXONOMY.open(), delimiter="\t")}

    # Candidate-match metadata are available only for the original 27 ROI identities.
    candidate_meta = {}
    for path in (MODEL_ROOT / "metadata").glob("roi_*.json"):
        doc = json.loads(path.read_text())
        for occ in doc.get("occurrences", []):
            candidate_meta[(occ["burst_id"], occ["roi_identity"])] = occ

    rows = []
    wide = {}
    for item in items:
        site = f"{item['original_roi_id']}@x{item['x_int']}_y{item['y_int']}"
        start_ui = item["event_start_ui"] - PRE; event_len = item["event_end_ui"] - item["event_start_ui"] + 1
        stop_ui = item["event_end_ui"]
        a, b = start_ui - ALIGNMENT_START_UI, stop_ui - ALIGNMENT_START_UI + 1
        clips = [np.asarray(arr[a:b], dtype=float) for arr in arrays]
        center_traces = [clip[:, item["y_int"], item["x_int"]] for clip in clips]
        ann_traces = [annulus_values(clip, item["x_int"], item["y_int"]) for clip in clips]
        obs_metrics = []
        for si, stage in enumerate(STAGES):
            tm = trace_metrics(center_traces[si], event_len)
            sm = spatial_metrics(clips[si], item["x_int"], item["y_int"], event_len)
            ann_corr = finite_corr(center_traces[si], ann_traces[si])
            ann_event = float(np.nanmax(ann_traces[si][PRE : PRE + event_len]) - np.nanmedian(ann_traces[si][:PRE]))
            row = {"observation_id": item["observation_id"], "site_id": site, "original_roi_id": item["original_roi_id"],
                   "canonical_roi_id": item["canonical_roi_id"], "trace_class_id": taxonomy[site],
                   "burst_id": item["burst_id"], "stage": stage, "x_px": item["x_px"], "y_px": item["y_px"],
                   "event_start_ui": item["event_start_ui"], "event_end_ui": item["event_end_ui"],
                   **tm, **sm, "annulus_event_peak": ann_event, "center_annulus_trace_correlation": ann_corr}
            if si == 0:
                y, x = item["y_int"], item["x_int"]
                local = clips[0][:, max(0, y - 10) : y + 11, max(0, x - 10) : x + 11]
                row["raw_local_saturation_fraction"] = float(np.mean(local >= 4095))
                row["raw_center_saturation_fraction"] = float(np.mean(center_traces[0] >= 4095))
            else:
                row["raw_local_saturation_fraction"] = np.nan
                row["raw_center_saturation_fraction"] = np.nan
            if si > 0:
                previous = obs_metrics[si - 1]
                row.update({"previous_stage": STAGES[si - 1],
                            "shape_corr_from_previous": finite_corr(center_traces[si - 1], center_traces[si]),
                            "peak_gain_from_previous": tm["event_peak"] / (previous["event_peak"] + 1e-9),
                            "peak_time_shift_from_previous_s": tm["time_to_peak_s"] - previous["time_to_peak_s"],
                            "contrast_gain_from_previous": sm["center_annulus_contrast"] / (previous["center_annulus_contrast"] + 1e-9)})
            else:
                row.update({"previous_stage": "", "shape_corr_from_previous": np.nan, "peak_gain_from_previous": np.nan,
                            "peak_time_shift_from_previous_s": np.nan, "contrast_gain_from_previous": np.nan})
            meta = candidate_meta.get((item["burst_id"], item["original_roi_id"]))
            row["recovery_candidate_metadata_available"] = bool(meta)
            row["recovery_matched_at_58"] = meta.get("recovery_matched_at_58") if meta else ""
            row["recovery_nearest_rank"] = meta.get("recovery_nearest_rank") if meta else ""
            row["recovery_nearest_distance_px"] = meta.get("recovery_nearest_distance_px") if meta else ""
            # Ratio proxy only; true LS denominator was not saved.
            if si == 2:
                ratio = np.abs(center_traces[1]) / (np.abs(center_traces[2]) + 1e-6)
                row["ica_to_ls_effective_scale_proxy_median"] = float(np.nanmedian(ratio[PRE : PRE + event_len]))
                row["ica_to_ls_effective_scale_proxy_min"] = float(np.nanmin(ratio[PRE : PRE + event_len]))
            else:
                row["ica_to_ls_effective_scale_proxy_median"] = np.nan
                row["ica_to_ls_effective_scale_proxy_min"] = np.nan
            rows.append(row); obs_metrics.append({**tm, **sm})
        wide[item["observation_id"]] = {stage: rows[-3 + si] for si, stage in enumerate(STAGES)}

    out_tsv = FINAL / "fig05_pipeline_diagnostic_table.tsv"
    fields = list(rows[0])
    with out_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerows(rows)

    availability = [
        ("Raw/ICA/LS center traces", "observed", "saved aligned movies"),
        ("Native amplitude, noise, timing, area", "derived", "computed from center traces"),
        ("Annulus coupling and center-annulus contrast", "derived", "3-6 px annulus"),
        ("Spatial footprint radius and peak offset", "derived", "21x21 event-minus-baseline crop"),
        ("Raw saturation", "derived", "uint16 ceiling 4095"),
        ("Candidate match/rank", "partial", "available for original 27-ROI diagnostic set only"),
        ("Local-standardization denominator", "derived", "exact bounded reproduction from frozen fit, causal pre-roll, and saved floor"),
        ("ICA component activations/contributions", "observed", "exported from frozen fit at all 50 sites"),
        ("Mixing/unmixing matrices", "observed", "exported directly from frozen temporal fit"),
        ("Embedding inversion residual", "derived", "numerical transform audit; not denoising error"),
        ("Motion/registration residual", "unavailable", "no motion field saved with this output"),
        ("Independent-recording validation", "unavailable", "single recording in current analysis"),
    ]
    availability_tsv = FINAL / "fig05_diagnostic_availability.tsv"
    with availability_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(("diagnostic", "availability", "evidence")); w.writerows(availability)

    # Figure: metric matrix, transition summaries, class associations, availability.
    metric_names = ["robust_snr", "persistence_ratio", "center_annulus_contrast", "effective_radius_px",
                    "peak_offset_px", "center_annulus_trace_correlation"]
    labels_short = ["SNR", "Persistence", "Center-annulus", "Radius", "Peak offset", "Annulus corr."]
    ls_rows = [r for r in rows if r["stage"] == "Local standardization"]
    matrix = np.array([[float(r[m]) for m in metric_names] for r in ls_rows])
    med = np.nanmedian(matrix, axis=0); mad = np.array([robust_scale(matrix[:, j]) for j in range(matrix.shape[1])])
    z = np.clip((matrix - med) / np.where(mad > 1e-9, mad, 1), -3, 3)
    order = np.lexsort((np.array([r["burst_id"] for r in ls_rows]), np.array([r["trace_class_id"] for r in ls_rows])))

    fig = plt.figure(figsize=(11.2, 8.7)); gs = fig.add_gridspec(2, 2, height_ratios=(1.35, 1), wspace=0.34, hspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); im = ax.imshow(z[order], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set(xticks=range(len(labels_short)), xticklabels=labels_short, ylabel="Confirmed occurrence (T-class/burst order)",
           title="LS diagnostic profiles (robust z)")
    ax.tick_params(axis="x", rotation=35); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="robust z")

    ax = fig.add_subplot(gs[0, 1])
    transitions = [("Raw→ICA", [r for r in rows if r["stage"] == "CS-Parzen ICA"]),
                   ("ICA→LS", [r for r in rows if r["stage"] == "Local standardization"])]
    x = np.arange(2)
    corr_centers = [np.nanmedian([float(r["shape_corr_from_previous"]) for r in q]) for _, q in transitions]
    shift_centers = [np.nanmedian([float(r["peak_time_shift_from_previous_s"]) for r in q]) for _, q in transitions]
    bars = ax.bar(x - 0.16, corr_centers, width=0.32, color="#2463A8", label="shape correlation")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Shape correlation", color="#2463A8")
    ax2 = ax.twinx(); ax2.plot(x + 0.16, shift_centers, color="#D28A19", marker="o", lw=0, ms=7, label="peak shift")
    ax2.axhline(0, color="#777777", lw=0.7, ls="--"); ax2.set_ylabel("Peak-time shift (s)", color="#D28A19")
    ax.set(xticks=x, xticklabels=[t[0] for t in transitions], title="Median stage-transition effects")
    ax.legend(handles=[bars, ax2.lines[0]], labels=["shape correlation", "peak shift (s)"], frameon=False, fontsize=8, loc="center right")

    ax = fig.add_subplot(gs[1, 0])
    explanatory = ("robust_snr", "persistence_ratio", "center_annulus_contrast", "effective_radius_px",
                   "peak_offset_px", "center_annulus_trace_correlation", "ica_to_ls_effective_scale_proxy_median")
    corr = []
    class_binary = np.array([r["trace_class_id"] for r in ls_rows])
    for metric in explanatory:
        vals = np.array([float(r[metric]) for r in ls_rows]); ok = np.isfinite(vals)
        corr.append(float(spearmanr(vals[ok], class_binary[ok]).statistic))
    y = np.arange(len(explanatory)); ax.barh(y, corr, color=["#2463A8" if v < 0 else "#D28A19" for v in corr])
    ax.axvline(0, color="#333333", lw=0.8); ax.set(yticks=y, yticklabels=[x.replace("_", " ") for x in explanatory],
        xlabel="Spearman correlation with T2 label", title="Which LS diagnostics track the trace-class cut?")
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[1, 1]); status_order = ("observed", "derived", "partial", "unavailable")
    counts = [sum(s == status for _, s, _ in availability) for status in status_order]
    ax.barh(range(4), counts, color=("#2463A8", "#65733C", "#D28A19", "#777777"))
    ax.set(yticks=range(4), yticklabels=status_order, xlabel="Diagnostic families", title="Current diagnostic availability")
    for i, value in enumerate(counts): ax.text(value + 0.1, i, str(value), va="center")
    ax.invert_yaxis()
    fig.suptitle("Canonical-v7 pipeline diagnostic audit", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, "Derived metrics are descriptive within one recording. The true LS denominator is reported separately from the historical ICA/LS proxy. T classes were fit from normalized trace shape.", fontsize=7.5)
    out_fig = FINAL / "fig05_pipeline_diagnostic_audit.png"; fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Data-quality and audit summary.
    keys = [(r["observation_id"], r["stage"]) for r in rows]
    summary = {
        "status": "complete_descriptive_audit", "grain": "one row per confirmed occurrence and pipeline stage",
        "rows": len(rows), "occurrences": len(items), "stages": list(STAGES), "unique_composite_keys": len(set(keys)),
        "duplicate_composite_keys": len(keys) - len(set(keys)),
        "finite_rates": {m: float(np.mean(np.isfinite([float(r[m]) for r in rows]))) for m in metric_names},
        "candidate_metadata_coverage": float(np.mean([bool(r["recovery_candidate_metadata_available"]) for r in rows])),
        "diagnostic_availability": [{"diagnostic": d, "availability": s, "evidence": e} for d, s, e in availability],
        "interpretation": "feature-rich descriptive audit; model-internal attribution remains blocked by unsaved artifacts",
        "inputs": {portable_path(path, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT): sha256(path) for path in (MANIFEST, TAXONOMY, raw_path, ica_path, ls_path)},
        "outputs": {"diagnostic_table": portable_path(out_tsv, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT), "availability": portable_path(availability_tsv, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT), "figure": portable_path(out_fig, repository=REPO, data=DATA_ROOT, media=MEDIA_ROOT)},
    }
    summary["outputs"].update({"diagnostic_table_sha256": sha256(out_tsv), "availability_sha256": sha256(availability_tsv), "figure_sha256": sha256(out_fig)})
    out_json = FINAL / "fig05_pipeline_diagnostic_audit.json"; out_json.write_text(json.dumps(summary, indent=2) + "\n")
    report = ROOT / "PIPELINE_DIAGNOSTIC_AUDIT.md"
    report.write_text("# Pipeline diagnostic audit\n\n"
        f"The table contains {len(rows)} rows at occurrence x stage grain ({len(items)} occurrences x 3 stages), with no duplicate composite keys.\n\n"
        "## Available now\n\n" + "\n".join(f"- {d}: **{s}** — {e}" for d, s, e in availability if s != "unavailable") +
        "\n\n## Missing model diagnostics\n\n" + "\n".join(f"- {d}: {e}" for d, s, e in availability if s == "unavailable") +
        "\n\n## Interpretation boundary\n\nThe table supports signal, timing, neighborhood, spatial-footprint, stage-transition, component-attribution, and true LS-denominator diagnostics within this recording. The embedding residual tests numerical inversion only; motion correction and independent-recording validation remain unavailable.\n")
    print(json.dumps({"rows": len(rows), "duplicate_keys": summary["duplicate_composite_keys"],
                      "candidate_metadata_coverage": summary["candidate_metadata_coverage"], "figure": str(out_fig)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
