"""Prespecified secondary deep dives for the canonical-v7 feature panel."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-deep-dives-mpl")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurobench.experiments.neuron_identifiability.automated_feature_validation import (
    SEED,
    _read_tsv,
    frame_auc_only,
)
from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import (
    ALIGNMENT_START_UI,
    add_blossom,
    atomic_json,
    make_event_and_quiet_masks,
    quiet_standardize,
    sha256,
    write_tsv,
)
from neurobench.reports.scientific_audit import require_three_section_scientific_audit
from neurobench.portable_paths import data_root, media_root, portable_path


BUDGETS = (20, 40, 58, 80, 100)
LANES = ("carrier_signed", "coherence_w15", "propagation_lag2_w15")
KINETIC_RISE = (1, 2, 4)
KINETIC_DECAY = (3, 5, 10, 20, 40)
REPEATS = 20
BOOTSTRAPS = 2000


def _paths(repo: Path) -> dict[str, Path]:
    data = data_root(repo)
    media = media_root(repo)
    previous = repo / "Outputs/NeuronIdentifiability/spon_ca_burst_automated_feature_validation_v1"
    return {
        "manifest": media / "v7_priority_neuron_media_cs_parzen/manifest.json",
        "previous": previous,
        "temporal": previous / "tables/temporal_retrieval_occurrence.tsv",
        "coordinate": previous / "tables/coordinate_perturbation.tsv",
        "recovery": previous / "tables/recovery_model_predictions.tsv",
        "detector": data / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v7/observation_failure_audit.tsv",
        "native": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1/evaluation/full_native_screen.json",
        "common": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1/evaluation/identical_proposal_screen.json",
        "ls": data / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/recovery_msln.npy",
        "source_audit": repo / "Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8/10_representation_confirmation/detector_visual_audit_v2",
        "trace_atlas_manifest": media / "v7_priority_neuron_full_traces_cs_parzen/manifest.json",
        "script": Path(__file__).resolve(),
    }


def _site(item: dict[str, Any]) -> str:
    return f"{item['original_roi_id']}@x{int(item['x_int'])}_y{int(item['y_int'])}"


def _mean_by_site(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_id"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _bootstrap_site_difference(rows: list[dict[str, Any]], field: str, group_field: str, rng: np.random.Generator) -> tuple[float, float, float]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["site_id"])][int(row[group_field])].append(float(row[field]))
    sites = sorted(grouped)
    def statistic(sample: list[str]) -> float:
        a, b = [], []
        for site in sample:
            for value in grouped[site].get(1, []): a.append(value)
            for value in grouped[site].get(0, []): b.append(value)
        return float(np.mean(a) - np.mean(b)) if a and b else float("nan")
    observed = statistic(sites)
    draws = np.asarray([statistic(list(rng.choice(sites, len(sites), replace=True))) for _ in range(BOOTSTRAPS)])
    draws = draws[np.isfinite(draws)]
    return observed, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def failure_taxonomy(items: list[dict[str, Any]], detector: list[dict[str, str]], temporal: list[dict[str, str]], rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids = {str(i["observation_id"]) for i in items}
    chosen = [r for r in detector if r["observation_id"] in ids and r["label_view"] == "original" and r["timing_view"] == "adjudicated" and r["feature_id"] in LANES]
    if len(chosen) != len(items) * len(LANES):
        raise RuntimeError(f"detector join {len(chosen)} != {len(items) * len(LANES)}")
    lane_by_obs: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in chosen: lane_by_obs[row["observation_id"]][row["feature_id"]] = row
    auc_by_obs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in temporal: auc_by_obs[row["observation_id"]][row["feature_id"]] = float(row["frame_auc"])
    out = []
    for item in items:
        oid = str(item["observation_id"]); lanes = lane_by_obs[oid]
        classes = [lanes[l]["failure_class_at_budget_58"] for l in LANES]
        failures = [x for x in classes if x != "matched"]
        if not failures: category = "recovered_all_lanes"
        elif len(failures) < len(LANES): category = "partial_lane_recovery"
        elif len(set(failures)) == 1: category = f"all_lane_{failures[0]}"
        else: category = "all_lane_mixed__" + "+".join(sorted(set(failures)))
        row: dict[str, Any] = {
            "observation_id": oid, "site_id": _site(item), "burst_id": int(item["burst_id"]),
            "failure_taxonomy": category, "recovered_any_b58": int(len(failures) < len(LANES)),
            "recovered_lane_count": len(LANES) - len(failures),
        }
        for lane in LANES:
            source = lanes[lane]
            row[f"{lane}__class"] = source["failure_class_at_budget_58"]
            row[f"{lane}__nearest_rank"] = source["nearest_candidate_rank"]
            row[f"{lane}__nearest_distance_px"] = source["nearest_candidate_distance_px"]
            row[f"{lane}__auc"] = auc_by_obs[oid][lane]
        out.append(row)
    summary = []
    for category, count in sorted(Counter(r["failure_taxonomy"] for r in out).items(), key=lambda x: (-x[1], x[0])):
        summary.append({"failure_taxonomy": category, "occurrences": count, "fraction": count / len(out), "estimate": "", "site_bootstrap_ci95_low": "", "site_bootstrap_ci95_high": ""})
    for lane in LANES:
        delta, low, high = _bootstrap_site_difference(out, f"{lane}__auc", "recovered_any_b58", rng)
        summary.append({"failure_taxonomy": f"recovered_minus_missed_auc__{lane}", "occurrences": len(out), "fraction": "", "estimate": delta, "site_bootstrap_ci95_low": low, "site_bootstrap_ci95_high": high})
    return out, summary


def proposal_ranking(native: dict[str, Any], common: dict[str, Any]) -> list[dict[str, Any]]:
    native_rows = {r["config_id"]: r for r in native["rows"]}
    common_rows = {r["config_id"]: r for r in common["rows"]}
    features = ("coherence_w15", "propagation_lag2_w15")
    out = []
    for feature in features:
        config = f"standalone__{feature}"
        for budget in BUDGETS:
            n = float(native_rows[config]["budget_mean_recall"][str(budget)])
            c = float(common_rows[config]["budget_mean_recall"][str(budget)])
            nb = float(native["carrier_baseline"]["budget_mean_recall"][str(budget)])
            cb = float(common["carrier_baseline"]["budget_mean_recall"][str(budget)])
            out.append({
                "feature_id": feature, "budget": budget,
                "native_recall": n, "native_carrier_recall": nb, "native_delta_vs_carrier": n - nb,
                "common_proposal_recall": c, "common_proposal_carrier_recall": cb,
                "common_proposal_ranking_delta_vs_carrier": c - cb,
                "native_minus_common_delta_contrast": (n - nb) - (c - cb),
                "interpretation": "descriptive_native_vs_common_proposal_contrast_not_additive_causal_decomposition",
            })
    return out


def spatial_crowding(items: list[dict[str, Any]], coordinate: list[dict[str, str]], failures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = {str(i["observation_id"]): (float(i["x_int"]), float(i["y_int"])) for i in items}
    by_obs_feature: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in coordinate: by_obs_feature[(row["observation_id"], row["feature_id"])][int(row["radius_px"])] = float(row["mean_frame_auc"])
    recovery = {r["observation_id"]: r["recovered_any_b58"] for r in failures}
    out = []
    for item in items:
        oid = str(item["observation_id"]); x, y = points[oid]
        distances = sorted(np.hypot(x - ox, y - oy) for other, (ox, oy) in points.items() if other != oid)
        # Duplicate burst occurrences at the same immutable site do not count as spatial neighbors.
        positive = [d for d in distances if d > 1e-9]
        nn = min(positive) if positive else float("nan")
        unique_xy = {(ox, oy) for other, (ox, oy) in points.items() if other != oid and np.hypot(x-ox, y-oy) > 1e-9}
        for feature in ("raw_center", "carrier_signed", "coherence_w15", "propagation_lag2_w15", "representation_consensus", "multiscale_persistence"):
            curve = by_obs_feature[(oid, feature)]
            center, edge = curve[0], curve[max(curve)]
            slope = (edge - center) / max(curve)
            target = .5 + .5 * (center - .5)
            half = ">6"
            for radius in sorted(r for r in curve if r > 0):
                if curve[radius] <= target: half = str(radius); break
            out.append({
                "observation_id": oid, "site_id": _site(item), "burst_id": int(item["burst_id"]), "feature_id": feature,
                "nearest_distinct_label_px": nn,
                "neighbors_within_6px": sum(np.hypot(x-ox, y-oy) <= 6 for ox, oy in unique_xy),
                "neighbors_within_12px": sum(np.hypot(x-ox, y-oy) <= 12 for ox, oy in unique_xy),
                "neighbors_within_24px": sum(np.hypot(x-ox, y-oy) <= 24 for ox, oy in unique_xy),
                "auc_center": center, "auc_radius6": edge, "auc_slope_per_px_0_to_6": slope,
                "half_excess_auc_radius_px_censored": half, "recovered_any_b58": recovery[oid],
            })
    associations = []
    for feature in sorted({r["feature_id"] for r in out}):
        rows = [r for r in out if r["feature_id"] == feature]
        for crowd in ("nearest_distinct_label_px", "neighbors_within_6px", "neighbors_within_12px", "neighbors_within_24px"):
            rho = float(spearmanr([r[crowd] for r in rows], [r["auc_slope_per_px_0_to_6"] for r in rows]).statistic)
            associations.append({"feature_id": feature, "crowding_metric": crowd, "outcome": "auc_slope_per_px_0_to_6", "spearman_rho": rho, "n_occurrences": len(rows)})
    return out, associations


def _kernel(rise: int, decay: int, length: int = 80) -> np.ndarray:
    t = np.arange(length, dtype=float)
    k = (1 - np.exp(-(t + 1) / rise)) * np.exp(-t / decay)
    return k / max(float(np.sum(k)), np.finfo(float).eps)


def kinetic_bank(items: list[dict[str, Any]], ls_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ls = np.load(ls_path, mmap_mode="r")
    intervals = sorted({(int(i["event_start_ui"]), int(i["event_end_ui"])) for i in items})
    _, quiet = make_event_and_quiet_masks(len(ls), intervals)
    traces: dict[str, np.ndarray] = {}
    for item in items:
        site = _site(item)
        if site not in traces:
            traces[site] = quiet_standardize(np.asarray(ls[:, int(item["y_int"]), int(item["x_int"])], dtype=float), quiet)
    rows = []
    for item in items:
        site = _site(item); base = traces[site]
        for rise in KINETIC_RISE:
            for decay in KINETIC_DECAY:
                filtered = np.convolve(base, _kernel(rise, decay), mode="full")[:len(base)]
                rows.append({"observation_id": item["observation_id"], "site_id": site, "burst_id": int(item["burst_id"]), "rise_frames": rise, "decay_frames": decay, "kernel_id": f"rise{rise}_decay{decay}", "frame_auc": frame_auc_only(filtered, int(item["event_start_ui"]), int(item["event_end_ui"]), quiet)})
    summary = []
    for kernel_id in sorted({r["kernel_id"] for r in rows}):
        selected = [r for r in rows if r["kernel_id"] == kernel_id]
        summary.append({"kernel_id": kernel_id, "rise_frames": selected[0]["rise_frames"], "decay_frames": selected[0]["decay_frames"], "site_weighted_mean_frame_auc": _mean_by_site(selected, "frame_auc")})
    selected_rows = []
    for held in range(1, 5):
        candidates = []
        for row in summary:
            training = [r for r in rows if r["kernel_id"] == row["kernel_id"] and int(r["burst_id"]) != held]
            candidates.append((_mean_by_site(training, "frame_auc"), row["kernel_id"]))
        _, chosen = max(candidates, key=lambda x: (x[0], x[1]))
        tests = [r for r in rows if r["kernel_id"] == chosen and int(r["burst_id"]) == held]
        selected_rows.append({"held_burst": held, "selected_kernel_id": chosen, "training_site_weighted_auc": max(candidates)[0], "held_burst_site_weighted_auc": _mean_by_site(tests, "frame_auc"), "held_occurrences": len(tests)})
    return rows, sorted(summary, key=lambda x: -x["site_weighted_mean_frame_auc"]), selected_rows


def _crossfit(rows: list[dict[str, str]], columns: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in columns] for r in rows]); y = np.asarray([int(r["recovered_any_b58"]) for r in rows]); groups = np.asarray([r["site_id"] for r in rows])
    predictions = np.full(len(rows), np.nan); coefficients = []
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, y, groups):
        if set(groups[train]) & set(groups[test]): raise RuntimeError("site leakage")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed))
        model.fit(x[train], y[train]); predictions[test] = model.predict_proba(x[test])[:, 1]
        coefficients.append(model[-1].coef_[0])
    return predictions, np.asarray(coefficients)


def complementarity(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common = ["x_scaled", "y_scaled", "burst_1", "burst_2", "burst_3", "burst_4"]
    models = {
        "carrier": common + ["auc__carrier_signed"],
        "carrier_plus_coherence": common + ["auc__carrier_signed", "auc__coherence_w15"],
        "carrier_plus_lag": common + ["auc__carrier_signed", "auc__propagation_lag2_w15"],
        "compact_three": common + ["auc__carrier_signed", "auc__coherence_w15", "auc__propagation_lag2_w15"],
        "role_specific_full": common + ["auc__carrier_signed", "auc__coherence_w15", "auc__propagation_lag2_w15", "auc__representation_consensus", "auc__multiscale_persistence", "auc__raw_center_annulus", "priority_scaled"],
    }
    y = np.asarray([int(r["recovered_any_b58"]) for r in rows]); metrics, signs = [], []
    for name, columns in models.items():
        repeat_metrics = []; sign_values = []
        for repeat in range(REPEATS):
            p, coef = _crossfit(rows, columns, SEED + repeat)
            repeat_metrics.append((log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p), roc_auc_score(y, p)))
            sign_values.extend(coef)
        arr = np.asarray(repeat_metrics)
        metrics.append({"model": name, "repeated_group_splits": REPEATS, "mean_log_loss": float(arr[:,0].mean()), "sd_log_loss": float(arr[:,0].std(ddof=1)), "mean_brier": float(arr[:,1].mean()), "mean_roc_auc": float(arr[:,2].mean()), "min_roc_auc": float(arr[:,2].min()), "max_roc_auc": float(arr[:,2].max())})
        sign_arr = np.asarray(sign_values)
        for index, column in enumerate(columns):
            signs.append({"model": name, "column": column, "positive_coefficient_fraction": float(np.mean(sign_arr[:, index] > 0)), "fold_fits": len(sign_arr)})
    baseline = next(r for r in metrics if r["model"] == "carrier")
    for row in metrics: row["mean_log_loss_improvement_vs_carrier"] = float(baseline["mean_log_loss"] - row["mean_log_loss"])
    return metrics, signs


def mechanism(native: dict[str, Any], temporal: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = ("coherence_w7", "coherence_w15", "coherence_w31", "propagation_lag1_w15", "propagation_lag2_w15", "propagation_lag4_w31")
    lookup = {r["config_id"]: r for r in native["rows"]}
    out = []
    for feature in wanted:
        row = lookup[f"standalone__{feature}"]
        for budget in BUDGETS:
            out.append({"analysis": "native_detector_variant", "feature_id": feature, "budget": budget, "value": float(row["budget_mean_recall"][str(budget)]), "partial_spearman_vs_carrier": "", "partial_spearman_vs_coherence": "", "interpretation": "window_or_lag_sensitivity_descriptive"})
    by_obs: dict[str, dict[str, float]] = defaultdict(dict)
    burst: dict[str, int] = {}
    for row in temporal:
        by_obs[row["observation_id"]][row["feature_id"]] = float(row["frame_auc"]); burst[row["observation_id"]] = int(row["burst_id"])
    ids = sorted(by_obs); carrier = np.asarray([by_obs[i]["carrier_signed"] for i in ids]); coherence = np.asarray([by_obs[i]["coherence_w15"] for i in ids]); lag = np.asarray([by_obs[i]["propagation_lag2_w15"] for i in ids])
    design = np.column_stack([np.ones(len(ids)), carrier, coherence] + [(np.asarray([burst[i] for i in ids]) == b).astype(float) for b in (2,3,4)])
    residual = lag - design @ np.linalg.lstsq(design, lag, rcond=None)[0]
    out.append({"analysis": "conditional_residual", "feature_id": "propagation_lag2_w15", "budget": "", "value": float(np.std(residual)), "partial_spearman_vs_carrier": float(spearmanr(residual, carrier).statistic), "partial_spearman_vs_coherence": float(spearmanr(residual, coherence).statistic), "interpretation": "residual_variation_after_carrier_coherence_and_burst_not_causal"})
    return out


def _figures(root: Path, proposal: list[dict[str, Any]], failure_summary: list[dict[str, Any]], kinetic_summary: list[dict[str, Any]], comp: list[dict[str, Any]], spatial: list[dict[str, Any]]) -> None:
    figures = root / "figures"; figures.mkdir()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cats = [r for r in failure_summary if not r["failure_taxonomy"].startswith("recovered_minus")]
    axes[0,0].barh([r["failure_taxonomy"].replace("_", " ") for r in reversed(cats)], [r["occurrences"] for r in reversed(cats)], color="#2463A8"); axes[0,0].set(title="B58 occurrence taxonomy", xlabel="Occurrences")
    for feature, color in (("coherence_w15", "#2463A8"), ("propagation_lag2_w15", "#D28A19")):
        rows = [r for r in proposal if r["feature_id"] == feature]
        axes[0,1].plot([r["budget"] for r in rows], [r["native_delta_vs_carrier"] for r in rows], marker="o", color=color, label=feature + " native")
        axes[0,1].plot([r["budget"] for r in rows], [r["common_proposal_ranking_delta_vs_carrier"] for r in rows], marker="s", ls="--", color=color, alpha=.75, label=feature + " common proposals")
    axes[0,1].axhline(0, color="#777", lw=1); axes[0,1].set(title="Proposal/ranking contrast", xlabel="Budget per burst", ylabel="Recall delta vs carrier"); axes[0,1].legend(fontsize=7)
    top = kinetic_summary[:8]
    axes[1,0].barh([r["kernel_id"] for r in reversed(top)], [r["site_weighted_mean_frame_auc"] for r in reversed(top)], color="#D28A19"); axes[1,0].axvline(.5, color="#777", ls="--"); axes[1,0].set(title="Kinetic bank (top fixed kernels)", xlabel="Site-weighted frame AUC")
    axes[1,1].bar([r["model"] for r in comp], [r["mean_log_loss"] for r in comp], color=["#777777"] + ["#2463A8"]*(len(comp)-1)); axes[1,1].set(title=f"Repeated site-grouped recovery models ({REPEATS} splits)", ylabel="Mean log loss"); axes[1,1].tick_params(axis="x", rotation=28, labelsize=7)
    for ax in axes.ravel(): ax.grid(alpha=.18)
    fig.suptitle("Canonical-v7 feature deep dives", fontweight="bold", fontsize=15); fig.text(.01,.008,"Sparse-positive, within-recording secondary analysis. Common-proposal contrasts are descriptive; unmatched candidates remain unknown.",fontsize=7.5); add_blossom(fig); fig.tight_layout(rect=(0,.025,1,.965)); fig.savefig(figures/"deep_dive_overview.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5,5.5))
    for feature, color in (("raw_center","#777777"),("carrier_signed","#2463A8"),("coherence_w15","#D28A19"),("propagation_lag2_w15","#7A6AA6")):
        rows=[r for r in spatial if r["feature_id"]==feature]; ax.scatter([r["nearest_distinct_label_px"] for r in rows],[r["auc_slope_per_px_0_to_6"] for r in rows],s=16,alpha=.55,label=feature,color=color)
    ax.axhline(0,color="#777",ls="--"); ax.set(title="Spatial decay versus local crowding",xlabel="Nearest distinct labeled center (px)",ylabel="AUC slope per pixel, radius 0 to 6"); ax.grid(alpha=.18); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figures/"spatial_decay_crowding.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)


def run(repo: Path, output: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    paths = _paths(repo); missing = [str(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError(missing)
    if output.exists() or Path(str(output)+".partial").exists(): raise FileExistsError(f"refusing existing output {output}")
    manifest = json.loads(paths["manifest"].read_text()); items = manifest["items"]
    source_audit_summary = json.loads((paths["source_audit"] / "summary.json").read_text())
    inventory = require_three_section_scientific_audit(paths["source_audit"], expected_expert_roi_count=int(source_audit_summary["expert_roi_identities"]), expected_model_roi_count=int(source_audit_summary["model_roi_identities"]), expected_expert_occurrence_count=int(source_audit_summary["expert_occurrences"]))
    trace_atlas = json.loads(paths["trace_atlas_manifest"].read_text())
    trace_atlas_root = paths["trace_atlas_manifest"].parent
    trace_images = [trace_atlas_root / item["trace_image"] for item in trace_atlas["items"]]
    trace_atlas_complete = len(trace_atlas["items"]) == len(items) and all(path.is_file() and path.stat().st_size > 0 for path in trace_images)
    if not trace_atlas_complete:
        raise RuntimeError("canonical-v7 106-occurrence full-trace atlas is incomplete")
    hashes = {name: sha256(path) for name, path in paths.items() if path.is_file()}
    preflight = {"status":"passed","population":{"occurrences":len(items),"original_geometries":len({_site(i) for i in items}),"bursts":len({int(i['burst_id']) for i in items})},"estimands":["failure taxonomy at frozen B58","native-vs-common-proposal recall contrast","spatial decay and label crowding","leave-one-burst-out kinetic-bank retrieval","repeated site-grouped recovery complementarity","coherence/lag variant and residual audit"],"scientific_audit":{"mode":"validated_source_audit_reuse","detector_audit_source":portable_path(paths["source_audit"],repository=repo,data=data_root(repo),media=media_root(repo)),"detector_inventory":inventory.to_dict(),"canonical_v7_trace_atlas":portable_path(paths["trace_atlas_manifest"],repository=repo,data=data_root(repo),media=media_root(repo)),"canonical_v7_trace_images":len(trace_images),"canonical_v7_trace_atlas_complete":trace_atlas_complete,"reason":"secondary analysis of the exact frozen detector lanes and immutable ROI centers; no new proposal images or model annotations"},"input_hashes":hashes,"seed":SEED}
    if preflight_only: return preflight
    partial = Path(str(output)+".partial"); partial.mkdir(parents=True); (partial/"tables").mkdir(); atomic_json(partial/"preflight.json",preflight)
    temporal = _read_tsv(paths["temporal"]); coordinate = _read_tsv(paths["coordinate"]); recovery = _read_tsv(paths["recovery"]); detector = _read_tsv(paths["detector"])
    native=json.loads(paths["native"].read_text()); common=json.loads(paths["common"].read_text()); rng=np.random.default_rng(SEED)
    failures, failure_summary = failure_taxonomy(items, detector, temporal, rng)
    proposal = proposal_ranking(native, common)
    spatial, associations = spatial_crowding(items, coordinate, failures)
    kinetics, kinetic_summary, kinetic_lobo = kinetic_bank(items, paths["ls"])
    comp, signs = complementarity(recovery)
    mechanisms = mechanism(native, temporal)
    burst_roles=[]
    for feature in sorted({r["feature_id"] for r in temporal}):
        for burst in range(1,5):
            rows=[r for r in temporal if r["feature_id"]==feature and int(r["burst_id"])==burst]
            burst_roles.append({"feature_id":feature,"burst_id":burst,"occurrences":len(rows),"site_weighted_mean_frame_auc":_mean_by_site(rows,"frame_auc")})
    tables={"failure_taxonomy_occurrence.tsv":failures,"failure_taxonomy_summary.tsv":failure_summary,"proposal_ranking_decomposition.tsv":proposal,"spatial_decay_crowding.tsv":spatial,"crowding_associations.tsv":associations,"kinetic_bank_occurrence.tsv":kinetics,"kinetic_bank_summary.tsv":kinetic_summary,"kinetic_bank_lobo_selection.tsv":kinetic_lobo,"recovery_complementarity.tsv":comp,"recovery_coefficient_stability.tsv":signs,"coherence_lag_mechanism_audit.tsv":mechanisms,"burst_conditioned_feature_roles.tsv":burst_roles}
    for name, rows in tables.items(): write_tsv(partial/"tables"/name,rows)
    _figures(partial,proposal,failure_summary,kinetic_summary,comp,spatial)
    any_recovered=sum(r["recovered_any_b58"] for r in failures); all_missed=len(failures)-any_recovered
    best_kernel=kinetic_summary[0]; compact=next(r for r in comp if r["model"]=="compact_three"); carrier=next(r for r in comp if r["model"]=="carrier")
    summary={"status":"complete","population":preflight["population"],"headline":{"recovered_any_b58":any_recovered,"all_lane_misses":all_missed,"best_fixed_kinetic_kernel":best_kernel,"lobo_kinetic_mean_auc":float(np.mean([r["held_burst_site_weighted_auc"] for r in kinetic_lobo])),"carrier_repeated_log_loss":carrier["mean_log_loss"],"compact_three_repeated_log_loss":compact["mean_log_loss"],"compact_log_loss_improvement":compact["mean_log_loss_improvement_vs_carrier"],"budget20_proposal_ranking":[r for r in proposal if r["budget"]==20]},"scientific_audit":preflight["scientific_audit"],"interpretation_limits":["Single recording and four bursts; no cross-fish generalization.","Known labels are sparse positives; unmatched candidates are unknown, not negatives.","Failure classes describe pipeline outcomes, not biological causes.","Native-vs-common-proposal contrasts are not an additive causal decomposition.","Kinetic-bank selection is leave-one-burst-out within the same recording.","Only twelve all-lane B58 misses make recovery modeling unstable and exploratory."]}
    checks={"occurrence_count":len(failures)==106,"all_lane_misses":all_missed==12,"detector_join_complete":all(len([k for k in r if k.endswith('__class')])==3 for r in failures),"proposal_rows":len(proposal)==10,"kinetic_rows":len(kinetics)==106*len(KINETIC_RISE)*len(KINETIC_DECAY),"kinetic_lobo_folds":len(kinetic_lobo)==4,"recovery_models":len(comp)==5,"source_audit_complete":inventory.complete,"canonical_v7_trace_atlas_complete":trace_atlas_complete,"figures":all((partial/"figures"/n).is_file() for n in ("deep_dive_overview.png","spatial_decay_crowding.png")),"finite_headlines":all(np.isfinite(float(x)) for x in (summary["headline"]["lobo_kinetic_mean_auc"],carrier["mean_log_loss"],compact["mean_log_loss"]))}
    validation={"status":"passed" if all(checks.values()) else "failed","checks":checks}; atomic_json(partial/"summary.json",summary); atomic_json(partial/"validation.json",validation)
    if validation["status"]!="passed": raise RuntimeError(checks)
    p20=summary["headline"]["budget20_proposal_ranking"]
    report=f"""# Canonical-v7 feature deep-dive suite\n\nThe suite completed on 106 occurrences at 50 immutable original geometries. At frozen B58, {any_recovered}/106 were recovered by at least one quantitative lane and {all_missed}/106 were missed by all three.\n\nThe strongest fixed kinetic kernel was `{best_kernel['kernel_id']}` (site-weighted frame AUC {best_kernel['site_weighted_mean_frame_auc']:.3f}); leave-one-burst-out selection achieved mean held-burst AUC {summary['headline']['lobo_kinetic_mean_auc']:.3f}. The compact three-feature recovery model changed repeated site-grouped log loss from {carrier['mean_log_loss']:.3f} to {compact['mean_log_loss']:.3f} (improvement {compact['mean_log_loss_improvement_vs_carrier']:.3f}).\n\nAt B20, coherence changed recall relative to carrier by {p20[0]['native_delta_vs_carrier']:+.3f} in native proposals and {p20[0]['common_proposal_ranking_delta_vs_carrier']:+.3f} on the common proposal universe; lag changed it by {p20[1]['native_delta_vs_carrier']:+.3f} and {p20[1]['common_proposal_ranking_delta_vs_carrier']:+.3f}. These contrasts are descriptive, not an additive causal decomposition.\n\nThis is within-recording secondary analysis. Sparse-positive labels do not identify precision or false-positive rate, and the small number of misses makes recovery-model comparisons exploratory. The source detector visual audit is reused by exact path and validated inventory because no new proposals or annotation overlays are introduced.\n"""
    (partial/"REPORT.md").write_text(report)
    atomic_json(partial/"llm_context.json",{"entrypoint":"summary.json","primary_tables":list(tables),"representative_figures":["figures/deep_dive_overview.png","figures/spatial_decay_crowding.png"],"coordinate_contract":"x=column,y=row","frame_contract":"UI one-based inclusive; NumPy zero-based half-open","annotation_semantics":"sparse positive; unknown is not negative","source_scientific_audit":preflight["scientific_audit"]})
    artifacts=[]
    for path in sorted(p for p in partial.rglob("*") if p.is_file()): artifacts.append({"path":str(path.relative_to(partial)),"sha256":sha256(path),"bytes":path.stat().st_size})
    atomic_json(partial/"artifact_index.json",{"artifacts":artifacts}); atomic_json(partial/"status.json",{"status":"complete","validation":"passed","scientific_audit":"validated_source_reuse"}); partial.replace(output)
    return summary


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[3]); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args()
    result=run(args.repo_root.resolve(),args.output_root.resolve(),preflight_only=args.preflight_only); print(json.dumps(result if args.preflight_only else {"status":result["status"],"headline":result["headline"]},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
