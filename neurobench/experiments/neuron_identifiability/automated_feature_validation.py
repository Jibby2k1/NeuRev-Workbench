"""No-human-in-the-loop validation of canonical-v7 full-trace features."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-automated-validation-mpl")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurobench.algorithms.scientific_feature_audit import causal_local_correlation_feature, generalized_anscombe
from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import _quiet_calibrate
from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import (
    ALIGNMENT_START_UI,
    FEATURES,
    FEATURE_LABELS,
    add_blossom,
    annulus_trace,
    atomic_json,
    causal_mean,
    empirical_quiet_cdf,
    exponential_matched_filter,
    make_event_and_quiet_masks,
    quiet_standardize,
    robust_scale,
    sha256,
    write_tsv,
)
from neurobench.portable_paths import data_root, media_root, portable_path

SEED = 20260827
BOOTSTRAPS = 2000
NULL_DRAWS = 500
SELECTED = (
    "raw_center",
    "carrier_signed",
    "coherence_w15",
    "propagation_lag2_w15",
    "representation_consensus",
    "multiscale_persistence",
)
CONTROL_OFFSETS = (
    (-32, 0), (32, 0), (0, -32), (0, 32), (-24, -24), (24, 24),
    (-24, 24), (24, -24), (-40, 16), (40, -16),
)
JITTER_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _site(item: dict[str, Any]) -> str:
    return f"{item['original_roi_id']}@x{int(item['x_int'])}_y{int(item['y_int'])}"


def frame_retrieval(trace: np.ndarray, start_ui: int, end_ui: int, quiet_mask: np.ndarray) -> dict[str, float]:
    """Event-vs-guarded-quiet retrieval metrics; quiet is a reference, not a negative class."""
    start = int(start_ui) - ALIGNMENT_START_UI
    stop = int(end_ui) - ALIGNMENT_START_UI + 1
    event_idx = np.arange(start, stop)
    quiet_idx = np.flatnonzero(quiet_mask)
    scores = np.concatenate([np.asarray(trace)[event_idx], np.asarray(trace)[quiet_idx]]).astype(float)
    labels = np.concatenate([np.ones(len(event_idx), dtype=int), np.zeros(len(quiet_idx), dtype=int)])
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranks = rankdata(-scores, method="average")
    event_ranks = ranks[labels == 1]
    output = {
        "frame_auc": float(roc_auc_score(labels, scores)),
        "reference_average_precision": float(average_precision_score(labels, scores)),
        "reciprocal_first_event_rank": float(1.0 / np.min(event_ranks)),
    }
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(fraction * len(ranked_labels))))
        output[f"top_{int(fraction * 100):02d}pct_event_recall"] = float(np.sum(ranked_labels[:count]) / max(np.sum(labels), 1))
    return output


def frame_auc_only(trace: np.ndarray, start_ui: int, end_ui: int, quiet_mask: np.ndarray) -> float:
    """Rank-sum ROC AUC used in resampling loops; equivalent to sklearn ROC AUC."""
    start = int(start_ui) - ALIGNMENT_START_UI
    stop = int(end_ui) - ALIGNMENT_START_UI + 1
    positive = np.asarray(trace[start:stop], dtype=float)
    negative = np.asarray(trace[quiet_mask], dtype=float)
    values = np.concatenate([positive, negative])
    finite = np.isfinite(values)
    labels = np.concatenate([np.ones(len(positive), dtype=bool), np.zeros(len(negative), dtype=bool)])[finite]
    ranks = rankdata(values[finite], method="average")
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    return float((np.sum(ranks[labels]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_site(rows: list[dict[str, Any]], field: str, rng: np.random.Generator, draws: int = BOOTSTRAPS) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_id"])].append(float(row[field]))
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    sampled = np.asarray([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(draws)])
    return float(np.mean(values)), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def nonparametric_site_fraction(matrix: np.ndarray) -> float:
    valid = np.sum(np.isfinite(matrix), axis=1) >= 2
    matrix = np.asarray(matrix, dtype=float)[valid]
    if len(matrix) < 2:
        return 0.0
    site_var = float(np.nanvar(np.nanmean(matrix, axis=1)))
    residual_var = float(np.nanmean(np.nanvar(matrix, axis=1)))
    return site_var / max(site_var + residual_var, np.finfo(float).eps)


def choose_control(
    x: int,
    y: int,
    raw: np.ndarray,
    quiet_mask: np.ndarray,
    labeled_xy: list[tuple[int, int]],
) -> tuple[int, int]:
    """Deterministic same-field, baseline-matched translated unknown control."""
    height, width = raw.shape[1:]
    target_field = x < 286
    target_baseline = float(np.median(raw[quiet_mask, y, x]))
    candidates: list[tuple[float, int, int]] = []
    for dx, dy in CONTROL_OFFSETS:
        cx, cy = x + dx, y + dy
        if not (0 <= cx < width and 0 <= cy < height) or (cx < 286) != target_field:
            continue
        if min(np.hypot(cx - sx, cy - sy) for sx, sy in labeled_xy) < 12.0:
            continue
        mismatch = abs(float(np.median(raw[quiet_mask, cy, cx])) - target_baseline)
        candidates.append((mismatch, cx, cy))
    if not candidates:
        raise RuntimeError(f"no valid translated control for x={x}, y={y}")
    _, cx, cy = min(candidates)
    return int(cx), int(cy)


def _point_base_features(
    raw: np.ndarray,
    ica: np.ndarray,
    ls: np.ndarray,
    carrier: np.ndarray,
    x: int,
    y: int,
    quiet_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    raw_z = quiet_standardize(np.asarray(raw[:, y, x], dtype=float), quiet_mask)
    ica_z = quiet_standardize(np.asarray(ica[:, y, x], dtype=float), quiet_mask)
    ls_z = quiet_standardize(np.asarray(ls[:, y, x], dtype=float), quiet_mask)
    positive_ls = np.clip(ls_z, 0.0, None)
    consensus = np.minimum.reduce([
        empirical_quiet_cdf(raw_z, quiet_mask),
        empirical_quiet_cdf(ica_z, quiet_mask),
        empirical_quiet_cdf(ls_z, quiet_mask),
    ])
    persistence = np.cbrt(
        np.maximum(causal_mean(positive_ls, 3), 0)
        * np.maximum(causal_mean(positive_ls, 7), 0)
        * np.maximum(causal_mean(positive_ls, 15), 0)
    )
    return {
        "raw_center": np.clip(raw_z, 0.0, None),
        "ica_center": np.clip(ica_z, 0.0, None),
        "ls_center": positive_ls,
        "matched_filter_ls": exponential_matched_filter(ls_z),
        "carrier_signed": np.asarray(carrier[:, y, x], dtype=float),
        "representation_consensus": consensus,
        "multiscale_persistence": persistence,
        "_ls_z": ls_z,
    }


def _center_only_features(base: dict[str, np.ndarray], raw: np.ndarray, x: int, y: int, quiet_mask: np.ndarray) -> None:
    raw_center = np.asarray(raw[:, y, x], dtype=float)
    raw_annulus = annulus_trace(raw, x, y)
    base["raw_center_annulus"] = np.clip(quiet_standardize(raw_center - raw_annulus, quiet_mask), 0.0, None)
    vst_center = generalized_anscombe(raw_center, variance_intercept=0.0, variance_slope=3.6322)
    height, width = raw.shape[1:]
    radius = 6
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (((xx - x) ** 2 + (yy - y) ** 2) >= 9) & (((xx - x) ** 2 + (yy - y) ** 2) <= 36)
    patch = generalized_anscombe(np.asarray(raw[:, y0:y1, x0:x1], dtype=np.float32), variance_intercept=0.0, variance_slope=3.6322)
    contrast = vst_center - np.mean(patch[:, mask], axis=1)
    base["vst_center_annulus"] = np.clip(quiet_standardize(contrast, quiet_mask), 0.0, None)


def _temporal_rows(items: list[dict[str, Any]], site_features: dict[str, dict[str, np.ndarray]], quiet_mask: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        site = _site(item)
        for feature in FEATURES:
            rows.append({
                "observation_id": item["observation_id"], "site_id": site,
                "original_roi_id": item["original_roi_id"], "canonical_roi_id": item["canonical_roi_id"],
                "burst_id": int(item["burst_id"]), "feature_id": feature,
                **frame_retrieval(site_features[site][feature], item["event_start_ui"], item["event_end_ui"], quiet_mask),
            })
    return rows


def _summarize_temporal(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    output = []
    for feature in FEATURES:
        selected = [r for r in rows if r["feature_id"] == feature]
        summary: dict[str, Any] = {"feature_id": feature, "feature_label": FEATURE_LABELS[feature], "occurrences": len(selected)}
        for field in ("frame_auc", "reference_average_precision", "top_01pct_event_recall", "top_05pct_event_recall", "top_10pct_event_recall", "reciprocal_first_event_rank"):
            mean, low, high = bootstrap_site(selected, field, rng)
            summary[f"mean_{field}"] = mean
            summary[f"{field}_ci95_low"] = low
            summary[f"{field}_ci95_high"] = high
        output.append(summary)
    return sorted(output, key=lambda r: float(r["mean_frame_auc"]), reverse=True)


def _matrix(rows: list[dict[str, Any]], feature: str, field: str = "frame_auc") -> tuple[np.ndarray, list[str], list[int]]:
    chosen = [r for r in rows if r["feature_id"] == feature]
    sites = sorted({str(r["site_id"]) for r in chosen})
    bursts = sorted({int(r["burst_id"]) for r in chosen})
    matrix = np.full((len(sites), len(bursts)), np.nan)
    si, bi = {s: i for i, s in enumerate(sites)}, {b: i for i, b in enumerate(bursts)}
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in chosen:
        grouped[(str(row["site_id"]), int(row["burst_id"]))].append(float(row[field]))
    for (site, burst), values in grouped.items():
        matrix[si[site], bi[burst]] = float(np.median(values))
    return matrix, sites, bursts


def _reliability(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    output = []
    for feature in FEATURES:
        matrix, _, bursts = _matrix(rows, feature)
        corrs = []
        for i in range(len(bursts)):
            for j in range(i + 1, len(bursts)):
                valid = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
                if valid.sum() >= 10:
                    corrs.append(float(spearmanr(matrix[valid, i], matrix[valid, j]).statistic))
        fraction = nonparametric_site_fraction(matrix)
        draws = np.asarray([nonparametric_site_fraction(matrix[rng.integers(0, len(matrix), len(matrix))]) for _ in range(BOOTSTRAPS)])
        output.append({
            "feature_id": feature, "recurrent_sites": int(np.sum(np.sum(np.isfinite(matrix), axis=1) >= 2)),
            "conditional_site_variance_fraction_nonparametric": fraction,
            "site_bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
            "site_bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
            "median_pairwise_spearman": float(np.median(corrs)) if corrs else float("nan"),
            "pair_count": len(corrs), "model": "nonparametric_variance_fraction_not_REML",
        })
    return output


def crossfit_recovery(rows: list[dict[str, Any]], columns: list[str], seed: int = SEED) -> np.ndarray:
    x = np.asarray([[float(row[c]) for c in columns] for row in rows], dtype=float)
    y = np.asarray([int(row["recovered_any_b58"]) for row in rows], dtype=int)
    groups = np.asarray([str(row["site_id"]) for row in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.full(len(rows), np.nan)
    for train, test in splitter.split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("site leakage in recovery folds")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed))
        model.fit(x[train], y[train])
        predictions[test] = model.predict_proba(x[test])[:, 1]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("incomplete cross-fitted recovery predictions")
    return predictions


def _prediction_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def _recovery_analysis(
    items: list[dict[str, Any]], temporal: list[dict[str, Any]], detector_rows: list[dict[str, str]], rng: np.random.Generator
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = {str(i["observation_id"]) for i in items}
    selected = [r for r in detector_rows if r["label_view"] == "original" and r["timing_view"] == "adjudicated" and r["feature_id"] in {"carrier_signed", "coherence_w15", "propagation_lag2_w15"} and r["observation_id"] in manifest_ids]
    if len(selected) != len(items) * 3:
        raise RuntimeError(f"detector join is {len(selected)}, expected {len(items) * 3}")
    outcomes: dict[str, dict[str, int]] = defaultdict(dict)
    for row in selected:
        outcomes[row["observation_id"]][row["feature_id"]] = int(row["failure_class_at_budget_58"] == "matched")
    by_obs: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in temporal:
        by_obs[str(row["observation_id"])][str(row["feature_id"])] = row
    modeling = []
    for item in items:
        oid, site = str(item["observation_id"]), _site(item)
        values: dict[str, Any] = {
            "observation_id": oid, "site_id": site, "burst_id": int(item["burst_id"]),
            "x_scaled": float(item["x_int"]) / 573.0, "y_scaled": float(item["y_int"]) / 340.0,
            "priority_scaled": float(item["priority_score_ls_3x3_max"]) / 30.0,
            "recovered_any_b58": int(any(outcomes[oid].values())),
            "recovered_all_b58": int(all(outcomes[oid].values())),
        }
        for burst in range(1, 5):
            values[f"burst_{burst}"] = float(int(item["burst_id"]) == burst)
        for feature in FEATURES:
            values[f"auc__{feature}"] = float(by_obs[oid][feature]["frame_auc"])
        modeling.append(values)
    if len({r["recovered_any_b58"] for r in modeling}) != 2:
        raise RuntimeError("recovery target does not contain both classes")
    common = ["x_scaled", "y_scaled", "burst_1", "burst_2", "burst_3", "burst_4"]
    baseline = common + ["auc__carrier_signed"]
    extensions = ["auc__coherence_w15", "auc__propagation_lag2_w15", "auc__representation_consensus", "auc__multiscale_persistence", "auc__raw_center_annulus", "priority_scaled"]
    extended = baseline + extensions
    p_base = crossfit_recovery(modeling, baseline)
    p_ext = crossfit_recovery(modeling, extended)
    y = np.asarray([r["recovered_any_b58"] for r in modeling], dtype=int)
    for row, pb, pe in zip(modeling, p_base, p_ext):
        row["prediction_carrier_model"] = float(pb)
        row["prediction_extended_model"] = float(pe)
    base_metrics, ext_metrics = _prediction_metrics(y, p_base), _prediction_metrics(y, p_ext)
    by_site: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(modeling):
        by_site[str(row["site_id"])].append(index)
    site_delta = []
    eps = 1e-12
    for indices in by_site.values():
        yy = y[indices]
        loss_base = -(yy * np.log(np.clip(p_base[indices], eps, 1 - eps)) + (1 - yy) * np.log(np.clip(1 - p_base[indices], eps, 1 - eps)))
        loss_ext = -(yy * np.log(np.clip(p_ext[indices], eps, 1 - eps)) + (1 - yy) * np.log(np.clip(1 - p_ext[indices], eps, 1 - eps)))
        site_delta.append(float(np.mean(loss_base - loss_ext)))
    site_delta_arr = np.asarray(site_delta)
    delta_draws = np.asarray([np.mean(rng.choice(site_delta_arr, len(site_delta_arr), replace=True)) for _ in range(BOOTSTRAPS)])
    blank_delta = {"log_loss_improvement_vs_carrier": float("nan"), "site_bootstrap_improvement_ci95_low": float("nan"), "site_bootstrap_improvement_ci95_high": float("nan")}
    summary = [{
        "population": "106_original_geometries", "model": "carrier_only", **base_metrics,
        "recovered": int(y.sum()), "missed": int((1 - y).sum()), **blank_delta,
    }, {
        "population": "106_original_geometries", "model": "extended_role_specific", **ext_metrics,
        "recovered": int(y.sum()), "missed": int((1 - y).sum()),
        "log_loss_improvement_vs_carrier": float(base_metrics["log_loss"] - ext_metrics["log_loss"]),
        "site_bootstrap_improvement_ci95_low": float(np.quantile(delta_draws, 0.025)),
        "site_bootstrap_improvement_ci95_high": float(np.quantile(delta_draws, 0.975)),
    }]

    # Sensitivity only: canonical label view collapses the four roi_015 geometries.
    confirmed = [r for r in detector_rows if r["label_view"] == "confirmed" and r["timing_view"] == "adjudicated" and r["feature_id"] in {"carrier_signed", "coherence_w15", "propagation_lag2_w15"}]
    confirmed_outcomes: dict[str, dict[str, int]] = defaultdict(dict)
    for row in confirmed:
        confirmed_outcomes[row["observation_id"]][row["feature_id"]] = int(row["failure_class_at_budget_58"] == "matched")
    sensitivity = [dict(row, recovered_any_b58=int(any(confirmed_outcomes[row["observation_id"]].values()))) for row in modeling if row["observation_id"] in confirmed_outcomes and len(confirmed_outcomes[row["observation_id"]]) == 3]
    if len(sensitivity) != 102:
        raise RuntimeError(f"canonical-collapsed sensitivity join is {len(sensitivity)}, expected 102")
    sens_y = np.asarray([r["recovered_any_b58"] for r in sensitivity], dtype=int)
    sens_base = crossfit_recovery(sensitivity, baseline)
    sens_ext = crossfit_recovery(sensitivity, extended)
    sens_base_metrics, sens_ext_metrics = _prediction_metrics(sens_y, sens_base), _prediction_metrics(sens_y, sens_ext)
    summary.extend([
        {"population": "102_canonical_collapsed_sensitivity", "model": "carrier_only", **sens_base_metrics, "recovered": int(sens_y.sum()), "missed": int((1-sens_y).sum()), **blank_delta},
        {"population": "102_canonical_collapsed_sensitivity", "model": "extended_role_specific", **sens_ext_metrics, "recovered": int(sens_y.sum()), "missed": int((1-sens_y).sum()), "log_loss_improvement_vs_carrier": float(sens_base_metrics["log_loss"]-sens_ext_metrics["log_loss"]), "site_bootstrap_improvement_ci95_low": float("nan"), "site_bootstrap_improvement_ci95_high": float("nan")},
    ])
    ablation = []
    for column in extensions:
        pred = crossfit_recovery(modeling, [c for c in extended if c != column])
        metrics = _prediction_metrics(y, pred)
        ablation.append({"removed_column": column, "ablated_log_loss": metrics["log_loss"], "log_loss_increase_when_removed": metrics["log_loss"] - ext_metrics["log_loss"], "ablated_roc_auc": metrics["roc_auc"]})
    audit = {"detector_rows": len(selected), "canonical_sensitivity_rows": len(confirmed), "canonical_sensitivity_occurrences": len(sensitivity), "recovered_any": int(y.sum()), "missed_any": int((1-y).sum()), "baseline_columns": baseline, "extended_columns": extended}
    return modeling, summary, ablation, audit


def _spatial_and_perturbation(
    items: list[dict[str, Any]], site_features: dict[str, dict[str, np.ndarray]], point_features: dict[tuple[int, int], dict[str, np.ndarray]],
    controls: dict[str, tuple[int, int]], jitters: dict[str, dict[int, list[tuple[int, int]]]], quiet_mask: np.ndarray, rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    spatial, coordinate, timing = [], [], []
    for item in items:
        site = _site(item)
        control = controls[site]
        for feature in SELECTED:
            target = frame_retrieval(site_features[site][feature], item["event_start_ui"], item["event_end_ui"], quiet_mask)["frame_auc"]
            displaced = frame_retrieval(point_features[control][feature], item["event_start_ui"], item["event_end_ui"], quiet_mask)["frame_auc"]
            spatial.append({"observation_id": item["observation_id"], "site_id": site, "burst_id": int(item["burst_id"]), "feature_id": feature, "target_frame_auc": target, "displaced_unknown_frame_auc": displaced, "target_minus_displaced_auc": target-displaced, "control_x": control[0], "control_y": control[1]})
            coordinate.append({"observation_id": item["observation_id"], "site_id": site, "feature_id": feature, "radius_px": 0, "mean_frame_auc": target, "delta_from_center": 0.0, "valid_offsets": 1})
            for radius in (1, 2, 4, 6):
                values = [frame_retrieval(point_features[xy][feature], item["event_start_ui"], item["event_end_ui"], quiet_mask)["frame_auc"] for xy in jitters[site][radius]]
                coordinate.append({"observation_id": item["observation_id"], "site_id": site, "feature_id": feature, "radius_px": radius, "mean_frame_auc": float(np.mean(values)), "delta_from_center": float(np.mean(values)-target), "valid_offsets": len(values)})
            for shift in (-10, -5, -3, -1, 0, 1, 3, 5, 10):
                shifted = frame_retrieval(site_features[site][feature], int(item["event_start_ui"])+shift, int(item["event_end_ui"])+shift, quiet_mask)["frame_auc"]
                timing.append({"observation_id": item["observation_id"], "site_id": site, "feature_id": feature, "shift_frames": shift, "frame_auc": shifted, "delta_from_unshifted": shifted-target})
    summary = []
    for feature in SELECTED:
        chosen = [r for r in spatial if r["feature_id"] == feature]
        mean, low, high = bootstrap_site(chosen, "target_minus_displaced_auc", rng)
        summary.append({"feature_id": feature, "mean_target_minus_displaced_auc": mean, "site_bootstrap_ci95_low": low, "site_bootstrap_ci95_high": high, "decision": "spatial_superiority" if low > 0 else "unresolved"})
    return spatial, summary, coordinate, timing


def _null_calibration(items: list[dict[str, Any]], site_features: dict[str, dict[str, np.ndarray]], quiet_mask: np.ndarray, rng: np.random.Generator) -> list[dict[str, Any]]:
    observed = {}
    for feature in SELECTED:
        values = [frame_auc_only(site_features[_site(i)][feature], i["event_start_ui"], i["event_end_ui"], quiet_mask) for i in items]
        by_site: dict[str, list[float]] = defaultdict(list)
        for item, value in zip(items, values): by_site[_site(item)].append(value)
        observed[feature] = float(np.mean([np.mean(v) for v in by_site.values()]))
    sites = sorted({_site(i) for i in items})
    null: dict[str, list[float]] = {f: [] for f in SELECTED}
    for _ in range(NULL_DRAWS):
        shifts = {site: int(rng.integers(30, 530)) for site in sites}
        for feature in SELECTED:
            by_site: dict[str, list[float]] = defaultdict(list)
            for item in items:
                site = _site(item)
                shifted = np.roll(site_features[site][feature], shifts[site])
                by_site[site].append(frame_auc_only(shifted, item["event_start_ui"], item["event_end_ui"], quiet_mask))
            null[feature].append(float(np.mean([np.mean(v) for v in by_site.values()])))
    return [{
        "feature_id": feature, "observed_site_mean_auc": observed[feature],
        "null_mean_auc": float(np.mean(null[feature])),
        "null_ci95_low": float(np.quantile(null[feature], .025)), "null_ci95_high": float(np.quantile(null[feature], .975)),
        "empirical_upper_tail_p": float((1 + np.sum(np.asarray(null[feature]) >= observed[feature])) / (NULL_DRAWS + 1)),
        "null_draws": NULL_DRAWS,
    } for feature in SELECTED]


def _synthetic(site_features: dict[str, dict[str, np.ndarray]], quiet_mask: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    quiet_idx = np.flatnonzero(quiet_mask)
    for site_index, site in enumerate(sorted(site_features)):
        base = np.asarray(site_features[site]["_ls_z"], dtype=float)
        valid_starts = [s for s in quiet_idx if s + 40 <= len(base) and np.all(quiet_mask[s:s+40])]
        if not valid_starts:
            continue
        start = valid_starts[(site_index * 17) % len(valid_starts)]
        for amplitude in (0.0, 0.5, 1.0, 2.0, 4.0):
            for tau in (5.0, 10.0, 20.0):
                t = np.arange(40, dtype=float)
                kernel = (1.0 - np.exp(-t / 2.0)) * np.exp(-t / tau)
                kernel /= max(float(np.max(kernel)), 1e-12)
                injected = base.copy(); injected[start:start+40] += amplitude * kernel
                positive = np.clip(injected, 0.0, None)
                candidates = {
                    "ls_center": positive,
                    "matched_filter_ls": exponential_matched_filter(injected),
                    "multiscale_persistence": np.cbrt(np.maximum(causal_mean(positive,3),0)*np.maximum(causal_mean(positive,7),0)*np.maximum(causal_mean(positive,15),0)),
                }
                synthetic_quiet = quiet_mask.copy(); synthetic_quiet[max(0,start-15):min(len(base),start+55)] = False
                for feature, trace in candidates.items():
                    metrics = frame_retrieval(trace, ALIGNMENT_START_UI+start, ALIGNMENT_START_UI+start+39, synthetic_quiet)
                    rows.append({"site_id": site, "feature_id": feature, "amplitude_quiet_mad": amplitude, "tau_frames": tau, **metrics})
    return rows


def _make_figures(root: Path, temporal_summary: list[dict[str, Any]], spatial_summary: list[dict[str, Any]], recovery_summary: list[dict[str, Any]], reliability: list[dict[str, Any]], coordinate: list[dict[str, Any]], timing: list[dict[str, Any]], synthetic: list[dict[str, Any]], null_rows: list[dict[str, Any]]) -> None:
    figures = root / "figures"; figures.mkdir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ordered = list(reversed(temporal_summary)); y = np.arange(len(ordered))
    means = np.asarray([r["mean_frame_auc"] for r in ordered]); low=np.asarray([r["frame_auc_ci95_low"] for r in ordered]); high=np.asarray([r["frame_auc_ci95_high"] for r in ordered])
    axes[0,0].errorbar(means,y,xerr=[means-low,high-means],fmt="o",color="#2463A8",ecolor="#8FB1D5",capsize=2); axes[0,0].set_yticks(y, [r["feature_label"] for r in ordered],fontsize=8); axes[0,0].axvline(.5,color="#777",ls="--"); axes[0,0].set(title="Temporal event-vs-quiet retrieval",xlabel="Site-weighted frame ROC AUC",xlim=(.45,1.01))
    sp=list(reversed(spatial_summary)); sy=np.arange(len(sp)); sm=np.asarray([r["mean_target_minus_displaced_auc"] for r in sp]); sl=np.asarray([r["site_bootstrap_ci95_low"] for r in sp]); sh=np.asarray([r["site_bootstrap_ci95_high"] for r in sp])
    axes[0,1].errorbar(sm,sy,xerr=[sm-sl,sh-sm],fmt="o",color="#D28A19",ecolor="#E7C27F",capsize=2); axes[0,1].set_yticks(sy,[FEATURE_LABELS[r["feature_id"]] for r in sp],fontsize=8); axes[0,1].axvline(0,color="#777",ls="--"); axes[0,1].set(title="Target vs baseline-matched displaced tissue",xlabel="Paired frame-AUC difference")
    primary_recovery=[r for r in recovery_summary if r["population"]=="106_original_geometries"]; names=[r["model"] for r in primary_recovery]; losses=[r["log_loss"] for r in primary_recovery]; axes[1,0].bar(names,losses,color=["#777777","#2463A8"]); axes[1,0].set(title="Site-grouped recovery model",ylabel="Cross-fitted log loss"); axes[1,0].tick_params(axis="x",rotation=12)
    rel=sorted(reliability,key=lambda r:r["conditional_site_variance_fraction_nonparametric"]); ry=np.arange(len(rel)); axes[1,1].scatter([r["conditional_site_variance_fraction_nonparametric"] for r in rel],ry,color="#2463A8"); axes[1,1].set_yticks(ry,[FEATURE_LABELS[r["feature_id"]] for r in rel],fontsize=8); axes[1,1].set(title="Across-burst site persistence",xlabel="Nonparametric site variance fraction",xlim=(0,1))
    for ax in axes.ravel(): ax.grid(alpha=.18)
    fig.suptitle("Canonical-v7 automated feature validation",fontweight="bold",fontsize=15); fig.text(.01,.008,"106 original geometries; site-blocked inference. Quiet and displaced references are unknown, not verified negatives. Recovery target is any quantitative-lane B58 match.",fontsize=7.5); add_blossom(fig); fig.tight_layout(rect=(0,.025,1,.965)); fig.savefig(figures/"automated_validation_overview.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(14,9))
    for feature in SELECTED:
        rows=[r for r in coordinate if r["feature_id"]==feature]; xs=sorted({int(r["radius_px"]) for r in rows}); ys=[np.mean([float(r["mean_frame_auc"]) for r in rows if int(r["radius_px"])==x]) for x in xs]; axes[0,0].plot(xs,ys,marker="o",label=FEATURE_LABELS[feature])
        rows=[r for r in timing if r["feature_id"]==feature]; xs2=sorted({int(r["shift_frames"]) for r in rows}); ys2=[np.mean([float(r["frame_auc"]) for r in rows if int(r["shift_frames"])==x]) for x in xs2]; axes[0,1].plot(xs2,ys2,marker="o",label=FEATURE_LABELS[feature])
    axes[0,0].set(title="Coordinate perturbation",xlabel="Offset radius (pixels)",ylabel="Mean frame AUC"); axes[0,1].set(title="Timing perturbation",xlabel="Boundary shift (frames)",ylabel="Mean frame AUC")
    amp=[.0,.5,1.,2.,4.]; synth_features=("ls_center","matched_filter_ls","multiscale_persistence")
    for feature in synth_features:
        axes[1,0].plot(amp,[np.mean([float(r["frame_auc"]) for r in synthetic if r["feature_id"]==feature and float(r["amplitude_quiet_mad"])==a]) for a in amp],marker="o",label=FEATURE_LABELS[feature])
    axes[1,0].set(title="Synthetic transient sensitivity",xlabel="Injected peak (quiet MAD units)",ylabel="Mean frame AUC")
    x=np.arange(len(null_rows)); obs=[r["observed_site_mean_auc"] for r in null_rows]; nm=[r["null_mean_auc"] for r in null_rows]; axes[1,1].scatter(x,obs,label="Observed",color="#2463A8"); axes[1,1].scatter(x,nm,label="Circular-shift null",color="#777777"); axes[1,1].set_xticks(x,[FEATURE_LABELS[r["feature_id"]] for r in null_rows],rotation=30,ha="right",fontsize=7); axes[1,1].set(title="Alignment falsification",ylabel="Site-weighted frame AUC")
    for ax in axes.ravel(): ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=6,ncol=2); axes[1,0].legend(fontsize=7); axes[1,1].legend(fontsize=7)
    fig.suptitle("Robustness and falsification",fontweight="bold",fontsize=15); fig.text(.01,.008,"Perturbations are automated and use fixed grids. Circular shifts preserve each feature trace's autocorrelation; injections test operators, not biological realism.",fontsize=7.5); add_blossom(fig); fig.tight_layout(rect=(0,.025,1,.965)); fig.savefig(figures/"robustness_falsification.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)


def default_paths(repo_root: Path) -> dict[str, Path]:
    data = data_root(repo_root)
    media = media_root(repo_root)
    return {
        "manifest": media/"v7_priority_neuron_media_cs_parzen/manifest.json",
        "raw": data/"Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
        "ica": data/"Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/recovery_msica.npy",
        "ls": data/"Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/recovery_msln.npy",
        "carrier": data/"Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy",
        "detector_audit": data/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v7/observation_failure_audit.tsv",
        "workflow": repo_root/"docs/workflows/spon_ca_burst_automated_feature_validation.md",
        "analysis_script": Path(__file__).resolve(),
    }


def run(repo_root: Path, output_root: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    paths=default_paths(repo_root); missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError(f"missing inputs: {missing}")
    if output_root.exists() or Path(str(output_root)+".partial").exists(): raise FileExistsError(f"refusing existing completed or partial output: {output_root}")
    manifest=json.loads(paths["manifest"].read_text()); items=manifest["items"]
    intervals=sorted({(int(i["event_start_ui"]),int(i["event_end_ui"])) for i in items})
    runtime_hashes={str(p):sha256(p) for p in paths.values()}; source=manifest["inputs"]
    hashes={portable_path(p,repository=repo_root,data=data_root(repo_root),media=media_root(repo_root)):runtime_hashes[str(p)] for p in paths.values()}
    for key in ("raw","ica","ls"):
        if source.get(str(paths[key])) != runtime_hashes[str(paths[key])]: raise RuntimeError(f"source hash differs from validated manifest for {key}")
    detector=_read_tsv(paths["detector_audit"])
    join=[r for r in detector if r["label_view"]=="original" and r["timing_view"]=="adjudicated" and r["feature_id"] in {"carrier_signed","coherence_w15","propagation_lag2_w15"} and r["observation_id"] in {i["observation_id"] for i in items}]
    preflight={"status":"passed","population":{"occurrences":len(items),"sites":len({_site(i) for i in items}),"bursts":len(intervals)},"frame_contract":{"first_ui":ALIGNMENT_START_UI,"frame_count":560,"intervals_ui_inclusive":intervals},"detector_join_rows":len(join),"expected_detector_join_rows":len(items)*3,"input_hashes":hashes,"random_seed":SEED,"bootstrap_draws":BOOTSTRAPS,"circular_shift_draws":NULL_DRAWS}
    if len(join)!=len(items)*3: raise RuntimeError(f"detector join is {len(join)} not {len(items)*3}")
    if preflight_only: return preflight
    partial=Path(str(output_root)+".partial"); partial.mkdir(parents=True); atomic_json(partial/"preflight.json",preflight); (partial/"tables").mkdir()
    raw_all=np.load(paths["raw"],mmap_mode="r"); ica=np.load(paths["ica"],mmap_mode="r"); ls=np.load(paths["ls"],mmap_mode="r"); carrier=np.load(paths["carrier"],mmap_mode="r"); raw=raw_all[ALIGNMENT_START_UI-1:ALIGNMENT_START_UI-1+len(ica)]
    if raw.shape!=ica.shape or raw.shape!=ls.shape or raw.shape!=carrier.shape or raw.shape!=(560,340,573): raise RuntimeError(f"shape mismatch: {raw.shape}, {ica.shape}, {ls.shape}, {carrier.shape}")
    _,quiet_mask=make_event_and_quiet_masks(len(raw),intervals)
    unique={}; [unique.setdefault(_site(i),i) for i in items]
    labeled=[(int(i["x_int"]),int(i["y_int"])) for i in unique.values()]
    controls={site:choose_control(int(i["x_int"]),int(i["y_int"]),raw,quiet_mask,labeled) for site,i in unique.items()}
    jitters: dict[str,dict[int,list[tuple[int,int]]]]={}; coordinates=set(controls.values())
    for site,item in unique.items():
        x,y=int(item["x_int"]),int(item["y_int"]); coordinates.add((x,y)); jitters[site]={}
        for radius in (1,2,4,6):
            values=[]
            for dx,dy in JITTER_DIRECTIONS:
                cx,cy=x+radius*dx,y+radius*dy
                if 0<=cx<573 and 0<=cy<340 and (cx<286)==(x<286): values.append((cx,cy)); coordinates.add((cx,cy))
            jitters[site][radius]=values
    point_features={xy:_point_base_features(raw,ica,ls,carrier,*xy,quiet_mask) for xy in sorted(coordinates)}
    site_features={site:point_features[(int(item["x_int"]),int(item["y_int"]))] for site,item in unique.items()}
    for site,item in unique.items(): _center_only_features(site_features[site],raw,int(item["x_int"]),int(item["y_int"]),quiet_mask)
    for feature,lag in (("coherence_w15",0),("propagation_lag2_w15",2)):
        feature_map=_quiet_calibrate(causal_local_correlation_feature(carrier,window_frames=15,lag_frames=lag,spatial_sigma_px=2.0,activity_qualified=True),100)
        for xy in coordinates: point_features[xy][feature]=np.asarray(feature_map[:,xy[1],xy[0]],dtype=float)
        del feature_map
    rng=np.random.default_rng(SEED)
    temporal=_temporal_rows(items,site_features,quiet_mask); temporal_summary=_summarize_temporal(temporal,rng); reliability=_reliability(temporal,rng)
    spatial,spatial_summary,coordinate,timing=_spatial_and_perturbation(items,site_features,point_features,controls,jitters,quiet_mask,rng)
    recovery,recovery_summary,ablation,recovery_audit=_recovery_analysis(items,temporal,detector,rng)
    null_rows=_null_calibration(items,site_features,quiet_mask,rng); synthetic=_synthetic(site_features,quiet_mask)
    tables={"temporal_retrieval_occurrence.tsv":temporal,"temporal_retrieval_summary.tsv":temporal_summary,"spatial_displacement_occurrence.tsv":spatial,"spatial_displacement_summary.tsv":spatial_summary,"coordinate_perturbation.tsv":coordinate,"timing_perturbation.tsv":timing,"recovery_model_predictions.tsv":recovery,"recovery_model_summary.tsv":recovery_summary,"recovery_model_ablation.tsv":ablation,"reliability_summary.tsv":reliability,"synthetic_injection.tsv":synthetic,"null_calibration.tsv":null_rows}
    for name,rows in tables.items(): write_tsv(partial/"tables"/name,rows)
    _make_figures(partial,temporal_summary,spatial_summary,recovery_summary,reliability,coordinate,timing,synthetic,null_rows)
    summary={"status":"complete","population":preflight["population"],"headline":{"best_temporal_feature":temporal_summary[0],"spatial_superiority_features":[r["feature_id"] for r in spatial_summary if r["decision"]=="spatial_superiority"],"recovery_models":recovery_summary,"null_calibration":null_rows},"recovery_join":recovery_audit,"interpretation_limits":["Same-recording validation is not independent replication.","Quiet frames and displaced tissue are unknown references, not verified negatives.","Recovery modeling has few misses and is exploratory.","Synthetic injections validate operators, not biological realism."]}
    checks={"temporal_rows":len(temporal)==len(items)*len(FEATURES),"spatial_rows":len(spatial)==len(items)*len(SELECTED),"recovery_rows":len(recovery)==len(items),"recovery_has_two_classes":len({r["recovered_any_b58"] for r in recovery})==2,"figures_exist":all((partial/"figures"/p).is_file() for p in ("automated_validation_overview.png","robustness_falsification.png")),"all_null_p_finite":all(np.isfinite(float(r["empirical_upper_tail_p"])) for r in null_rows)}
    validation={"status":"passed" if all(checks.values()) else "failed","checks":checks}; atomic_json(partial/"summary.json",summary); atomic_json(partial/"validation.json",validation)
    if validation["status"]!="passed": raise RuntimeError(f"validation failed: {checks}")
    report=["# Canonical-v7 automated feature validation","",f"The suite completed on {len(items)} occurrences at {len(unique)} immutable original geometries.","",f"The strongest frame-level temporal retrieval feature was `{temporal_summary[0]['feature_id']}` (site-weighted AUC {temporal_summary[0]['mean_frame_auc']:.3f}, 95% interval {temporal_summary[0]['frame_auc_ci95_low']:.3f}--{temporal_summary[0]['frame_auc_ci95_high']:.3f}).",f"At detector budget 58, {recovery_audit['recovered_any']}/{len(items)} original geometries were recovered by at least one quantitative lane. The carrier-only cross-fitted log loss was {recovery_summary[0]['log_loss']:.3f}; the extended model was {recovery_summary[1]['log_loss']:.3f}.","",f"Spatial superiority was supported for: {', '.join(summary['headline']['spatial_superiority_features']) or 'none'}.","","These are within-recording feature-engineering results. Guarded quiet frames and displaced tissue are reference sets with unknown biology; injection and circular-shift tests establish algorithm behavior, not neuron identity."]
    (partial/"REPORT.md").write_text("\n".join(report)+"\n")
    artifacts=[]
    for path in sorted(p for p in partial.rglob("*") if p.is_file()): artifacts.append({"path":str(path.relative_to(partial)),"sha256":sha256(path),"bytes":path.stat().st_size})
    atomic_json(partial/"artifact_index.json",{"artifacts":artifacts}); atomic_json(partial/"status.json",{"status":"complete","validation":"passed"}); partial.replace(output_root)
    return summary


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[3]); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args()
    result=run(args.repo_root.resolve(),args.output_root.resolve(),preflight_only=args.preflight_only); print(json.dumps(result if args.preflight_only else {"status":result["status"],"population":result["population"],"best_temporal_feature":result["headline"]["best_temporal_feature"]["feature_id"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
