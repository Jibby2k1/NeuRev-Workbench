"""Bounded feature-atlas evaluation on the frozen NREV-EXP-0021 candidate union.

The module adds analytically defined temporal, spatial, consistency, nuisance,
and local-competition features without changing candidate coordinates.  It
uses the frozen Run-B outer-fold assignments and reports positive-versus-
unlabeled (SPU) ranking metrics; unlabeled candidates are never interpreted as
verified negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-feature-atlas-v1-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_laplace
from scipy.spatial import cKDTree

from neurobench.experiments.neuron_identifiability.uncertainty_aware_models import (
    PenalizedLogisticRanker,
    RobustFoldPreprocessor,
    grouped_paired_bootstrap_rank_delta,
    summarize_folded_oof_ranking,
)


ALIGNMENT_START_UI = 1800
FRAME_COUNT = 560
CANDIDATE_BUDGET = 58
SEED = 20260905
BOOTSTRAP_DRAWS = 2000
MAX_LOGISTIC_ITERATIONS = 100000
BURSTS_UI_INCLUSIVE = {
    1: (2003, 2026),
    2: (2040, 2063),
    3: (2122, 2149),
    4: (2254, 2300),
}

EXISTING_FEATURES = (
    "carrier_signed",
    "local_psd_signal",
    "asymmetric_state",
    "spatial_coherence",
    "cross_scale_rank",
    "cross_scale_recall",
    "cfar_score",
    "cfar_background",
    "cfar_noise",
    "persistent_artifact_score",
    "cut_center_sigma2p5",
    "cut_ring_r4p5_t1p25",
)

FEATURE_FAMILIES = {
    "temporal_envelope": (
        "envelope_h3_peak",
        "envelope_h5_peak",
        "envelope_h5_area_ratio",
        "envelope_h5_core_fraction",
    ),
    "soma_morphology": (
        "log_soma_sigma1p5",
        "log_soma_sigma2p5",
        "log_soma_multiscale_min",
        "radial_center_ring_margin",
    ),
    "map_source_consistency": (
        "carrier_map_split_half_cosine",
        "map_trace_heldout_correlation",
        "ica_map_split_half_cosine",
    ),
    "nuisance_competition": (
        "global_nuisance_independence",
        "local_isolation_r12",
        "carrier_competition_margin_r12",
    ),
}
NEW_FEATURES = tuple(feature for family in FEATURE_FAMILIES.values() for feature in family)


@dataclass(frozen=True)
class AtlasPaths:
    candidates: Path
    oof_scores: Path
    carrier: Path
    ica: Path


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


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(partial, sep="\t", index=False)
    partial.replace(path)


def trailing_maximum(values: np.ndarray, window: int) -> np.ndarray:
    """Causal maximum including the current sample."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or window < 1:
        raise ValueError("trailing_maximum requires a vector and positive window")
    output = np.empty_like(array)
    for index in range(len(array)):
        output[index] = np.max(array[max(0, index - window + 1) : index + 1])
    return output


def envelope_features(trace: np.ndarray, start: int, stop: int) -> dict[str, float]:
    """Summarize H=A^2/U at two supports plus its non-peak morphology."""

    a = np.clip(np.asarray(trace, dtype=np.float64), 0.0, None)
    if not 0 <= start < stop <= len(a):
        raise ValueError("invalid event interval")
    result: dict[str, float] = {}
    h_by_window: dict[int, np.ndarray] = {}
    for window in (3, 5):
        upper = trailing_maximum(a, window)
        h = np.divide(a * a, np.maximum(upper, 1e-12), out=np.zeros_like(a), where=upper > 0)
        h_by_window[window] = h
        result[f"envelope_h{window}_peak"] = float(np.max(h[start:stop]))
    event_a = a[start:stop]
    event_h5 = h_by_window[5][start:stop]
    result["envelope_h5_area_ratio"] = float(np.sum(event_h5) / max(float(np.sum(event_a)), 1e-12))
    result["envelope_h5_core_fraction"] = float(np.max(event_h5) / max(float(np.sum(event_h5)), 1e-12))
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def split_half_map_cosine(event_patch: np.ndarray) -> float:
    """Cosine agreement between alternating-frame positive spatial maps."""

    patch = np.asarray(event_patch, dtype=np.float64)
    if patch.ndim != 3 or len(patch) < 4:
        raise ValueError("event_patch must contain at least four frames")
    first = np.mean(np.clip(patch[::2], 0.0, None), axis=0)
    second = np.mean(np.clip(patch[1::2], 0.0, None), axis=0)
    return _cosine(first, second)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return 0.0
    a = a[finite] - float(np.mean(a[finite]))
    b = b[finite] - float(np.mean(b[finite]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def heldout_map_trace_correlation(event_patch: np.ndarray) -> float:
    """Predict the center trace from a map learned on alternating frames.

    The center pixel is removed from the learned template, preventing the
    feature from becoming a tautological center-amplitude measurement.
    """

    patch = np.asarray(event_patch, dtype=np.float64)
    if patch.ndim != 3 or len(patch) < 6:
        raise ValueError("event_patch must contain at least six frames")
    height, width = patch.shape[1:]
    center = (height // 2, width // 2)
    scores: list[float] = []
    for train_slice, test_slice in ((slice(None, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(None, None, 2))):
        train = patch[train_slice].reshape(-1, height * width)
        test = patch[test_slice].reshape(-1, height * width)
        center_index = center[0] * width + center[1]
        train_center = train[:, center_index] - np.mean(train[:, center_index])
        train_center_norm = float(np.dot(train_center, train_center))
        if train_center_norm <= 1e-12:
            scores.append(0.0)
            continue
        train_centered = train - np.mean(train, axis=0, keepdims=True)
        template = train_centered.T @ train_center / train_center_norm
        template[center_index] = 0.0
        norm = float(np.linalg.norm(template))
        if norm <= 1e-12:
            scores.append(0.0)
            continue
        template /= norm
        projected = (test - np.mean(train, axis=0, keepdims=True)) @ template
        scores.append(_safe_correlation(projected, test[:, center_index]))
    return float(np.mean(scores))


def competition_features(
    coordinates: np.ndarray,
    carrier_scores: np.ndarray,
    partition_ids: np.ndarray,
    *,
    radius: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return negative local density and carrier margin to nearby proposals."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    scores = np.asarray(carrier_scores, dtype=np.float64)
    partitions = np.asarray(partition_ids)
    if coordinates.shape != (len(scores), 2) or partitions.shape != scores.shape:
        raise ValueError("competition inputs must be aligned")
    isolation = np.zeros(len(scores), dtype=np.float64)
    margin = np.zeros(len(scores), dtype=np.float64)
    for partition in np.unique(partitions):
        indices = np.flatnonzero(partitions == partition)
        tree = cKDTree(coordinates[indices])
        neighborhoods = tree.query_ball_point(coordinates[indices], r=float(radius))
        for local_index, neighbors in enumerate(neighborhoods):
            global_index = int(indices[local_index])
            others = [int(indices[item]) for item in neighbors if int(item) != local_index]
            isolation[global_index] = -float(len(others))
            margin[global_index] = float(scores[global_index] - np.max(scores[others])) if others else 0.0
    return isolation, margin


def _event_bounds(partition_id: int) -> tuple[int, int]:
    start_ui, end_ui = BURSTS_UI_INCLUSIVE[int(partition_id)]
    return start_ui - ALIGNMENT_START_UI, end_ui - ALIGNMENT_START_UI + 1


def _quiet_mask() -> np.ndarray:
    mask = np.ones(FRAME_COUNT, dtype=bool)
    for partition in BURSTS_UI_INCLUSIVE:
        start, stop = _event_bounds(partition)
        mask[max(0, start - 15) : min(FRAME_COUNT, stop + 15)] = False
    return mask


def _extract_patch(video: np.ndarray, frames: slice, x: int, y: int, radius: int = 4) -> np.ndarray:
    if x - radius < 0 or y - radius < 0 or x + radius >= video.shape[2] or y + radius >= video.shape[1]:
        raise ValueError(f"candidate ({x}, {y}) cannot support radius-{radius} patch")
    # ``np.asarray`` may preserve a read-only memmap view when the source is
    # already float32 (the ICA cache), while converting the float16 carrier
    # happens to allocate.  Both lanes need the same explicit writable copy
    # before quiet-baseline subtraction.
    return np.array(
        video[frames, y - radius : y + radius + 1, x - radius : x + radius + 1],
        dtype=np.float32,
        copy=True,
    )


def validate_inputs(paths: AtlasPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [str(path) for path in paths.__dict__.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing feature-atlas inputs: {missing}")
    candidates = pd.read_csv(paths.candidates, sep="\t")
    oof = pd.read_csv(paths.oof_scores, sep="\t")
    event = candidates[candidates["training_state"].isin(["positive", "unlabeled"])].copy()
    if len(event) != 1619 or int((event["training_state"] == "positive").sum()) != 78:
        raise RuntimeError("frozen event census differs from 1,619 candidates / 78 positives")
    if len(oof) != len(event) or oof["candidate_id"].duplicated().any():
        raise RuntimeError("OOF table does not uniquely cover the frozen event census")
    joined = oof[["candidate_id", "fold_id", "score__linear_spu__fold_percentile"]].merge(
        event, on="candidate_id", how="inner", validate="one_to_one"
    )
    if len(joined) != len(event) or sorted(joined["fold_id"].unique()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("candidate/OOF join or five-fold contract failed")
    carrier = np.load(paths.carrier, mmap_mode="r")
    ica = np.load(paths.ica, mmap_mode="r")
    if carrier.shape != (FRAME_COUNT, 340, 573) or ica.shape != carrier.shape:
        raise RuntimeError(f"source shape mismatch: carrier={carrier.shape}, ica={ica.shape}")
    source = {
        "candidate_sha256": sha256(paths.candidates),
        "oof_sha256": sha256(paths.oof_scores),
        "carrier": {"path": str(paths.carrier), "shape": list(carrier.shape), "dtype": str(carrier.dtype)},
        "ica": {"path": str(paths.ica), "shape": list(ica.shape), "dtype": str(ica.dtype)},
    }
    return joined.sort_values("candidate_id").reset_index(drop=True), source


def extract_features(candidates: pd.DataFrame, carrier: np.ndarray, ica: np.ndarray) -> pd.DataFrame:
    xs = candidates["x_px"].to_numpy(dtype=int)
    ys = candidates["y_px"].to_numpy(dtype=int)
    partitions = candidates["partition_id"].to_numpy(dtype=int)
    traces = np.asarray(carrier[:, ys, xs], dtype=np.float32)
    quiet = _quiet_mask()
    quiet_sample = np.flatnonzero(quiet)[::4]
    carrier_quiet_median = np.median(np.asarray(carrier[quiet_sample], dtype=np.float32), axis=0)
    ica_quiet_median = np.median(np.asarray(ica[quiet_sample], dtype=np.float32), axis=0)
    global_trace = np.median(np.asarray(carrier[:, ::8, ::8], dtype=np.float32), axis=(1, 2))

    rows: list[dict[str, float]] = []
    for index, (x, y, partition) in enumerate(zip(xs, ys, partitions, strict=True)):
        start, stop = _event_bounds(int(partition))
        row = envelope_features(traces[:, index], start, stop)
        carrier_patch = _extract_patch(carrier, slice(start, stop), int(x), int(y))
        carrier_patch -= carrier_quiet_median[y - 4 : y + 5, x - 4 : x + 5]
        ica_patch = _extract_patch(ica, slice(start, stop), int(x), int(y))
        ica_patch -= ica_quiet_median[y - 4 : y + 5, x - 4 : x + 5]
        row["carrier_map_split_half_cosine"] = split_half_map_cosine(carrier_patch)
        row["map_trace_heldout_correlation"] = heldout_map_trace_correlation(carrier_patch)
        row["ica_map_split_half_cosine"] = split_half_map_cosine(ica_patch)
        row["global_nuisance_independence"] = 1.0 - abs(
            _safe_correlation(traces[quiet, index], global_trace[quiet])
        )
        rows.append(row)

    output = pd.DataFrame(rows)
    for partition in sorted(BURSTS_UI_INCLUSIVE):
        indices = np.flatnonzero(partitions == partition)
        start, stop = _event_bounds(partition)
        pooled = np.max(np.clip(np.asarray(carrier[start:stop], dtype=np.float32), 0.0, None), axis=0)
        responses: dict[float, np.ndarray] = {}
        for sigma in (1.5, 2.5):
            responses[sigma] = -(sigma**2) * gaussian_laplace(pooled, sigma=sigma, mode="nearest")
            output.loc[indices, f"log_soma_sigma{str(sigma).replace('.', 'p')}"] = responses[sigma][ys[indices], xs[indices]]
        output.loc[indices, "log_soma_multiscale_min"] = np.minimum(
            responses[1.5][ys[indices], xs[indices]], responses[2.5][ys[indices], xs[indices]]
        )

    output["radial_center_ring_margin"] = (
        candidates["feature__cut_center_sigma2p5"].to_numpy(dtype=float)
        - candidates["feature__cut_ring_r4p5_t1p25"].to_numpy(dtype=float)
    )
    isolation, margin = competition_features(
        candidates[["x_px", "y_px"]].to_numpy(dtype=float),
        candidates["feature__carrier_signed"].to_numpy(dtype=float),
        partitions,
    )
    output["local_isolation_r12"] = isolation
    output["carrier_competition_margin_r12"] = margin
    if tuple(output.columns) != NEW_FEATURES:
        output = output.loc[:, NEW_FEATURES]
    if not np.all(np.isfinite(output.to_numpy(dtype=float))):
        raise RuntimeError("feature extraction produced non-finite values")
    return output


def _balanced_weights(target: np.ndarray) -> np.ndarray:
    positive = int(np.sum(target == 1))
    unlabeled = int(np.sum(target == 0))
    weights = np.where(target == 1, len(target) / (2 * positive), len(target) / (2 * unlabeled))
    return weights / np.mean(weights)


def crossfit_linear(
    values: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    feature_names: Iterable[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    names = tuple(feature_names)
    scores = np.full(len(labels), np.nan, dtype=float)
    diagnostics: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        test = fold_ids == fold
        train = ~test
        preprocessor = RobustFoldPreprocessor(clip=10.0, add_missing_indicators=True)
        train_values = preprocessor.fit_transform(values[train], labels[train] == 1, feature_names=names)
        test_values = preprocessor.transform(values[test])
        model = PenalizedLogisticRanker(
            penalty_strength=0.01,
            l1_ratio=0.0,
            max_iterations=MAX_LOGISTIC_ITERATIONS,
            tolerance=1e-6,
        )
        model.fit(train_values, labels[train], sample_weight=_balanced_weights(labels[train]))
        scores[test] = model.predict_score(test_values)
        diagnostics.append(
            {
                "fold_id": int(fold),
                "train_candidates": int(np.sum(train)),
                "test_candidates": int(np.sum(test)),
                "train_positives": int(np.sum(labels[train])),
                "test_positives": int(np.sum(labels[test])),
                "iterations": model.n_iterations_,
                "coefficients": {name: float(value) for name, value in zip(preprocessor.feature_names_out_, model.coef_, strict=True)},
            }
        )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("crossfit did not cover every candidate")
    return scores, diagnostics


def _test_folds(fold_ids: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(fold_ids == fold) for fold in sorted(np.unique(fold_ids))]


def evaluate_models(candidates: pd.DataFrame, new_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    labels = (candidates["training_state"].to_numpy() == "positive").astype(np.int8)
    folds = candidates["fold_id"].to_numpy(dtype=int)
    test_folds = _test_folds(folds)
    ties = candidates["candidate_id"].to_numpy(dtype=str)
    existing = candidates[[f"feature__{name}" for name in EXISTING_FEATURES]].to_numpy(dtype=float)
    new = new_features.loc[:, NEW_FEATURES].to_numpy(dtype=float)
    model_inputs: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {
        "retrained_existing_linear": (existing, EXISTING_FEATURES),
        "new_atlas_linear": (new, NEW_FEATURES),
        "augmented_linear": (np.column_stack([existing, new]), EXISTING_FEATURES + NEW_FEATURES),
    }
    for family, feature_names in FEATURE_FAMILIES.items():
        positions = [NEW_FEATURES.index(name) for name in feature_names]
        model_inputs[f"existing_plus_{family}"] = (
            np.column_stack([existing, new[:, positions]]), EXISTING_FEATURES + feature_names
        )
    scores: dict[str, np.ndarray] = {
        "carrier_signed": candidates["feature__carrier_signed"].to_numpy(dtype=float),
        "frozen_run_b_linear": candidates["score__linear_spu__fold_percentile"].to_numpy(dtype=float),
    }
    diagnostics: list[dict[str, Any]] = []
    for model_name, (matrix, names) in model_inputs.items():
        scores[model_name], fold_diagnostics = crossfit_linear(matrix, labels, folds, names)
        for row in fold_diagnostics:
            row["model_id"] = model_name
        diagnostics.extend(fold_diagnostics)

    metric_rows: list[dict[str, Any]] = []
    normalized: dict[str, np.ndarray] = {}
    for model_name, values in scores.items():
        summary = summarize_folded_oof_ranking(
            labels == 1, values, test_folds, budget=CANDIDATE_BUDGET, tie_breakers=ties
        )
        metrics = summary["metrics"]
        normalized[model_name] = np.asarray(summary["oof_scores_fold_percentile"], dtype=float)
        metric_rows.append(
            {
                "model_id": model_name,
                "macro_fold_spu_auc": metrics["macro_fold_positive_vs_unlabeled_rank_auc"],
                "global_spu_auc": metrics["positive_vs_unlabeled_rank_auc"],
                "known_positive_recovered_at_58": metrics["positive_recovered_at_budget"],
                "known_positive_recall_at_58": metrics["positive_recall_at_budget"],
            }
        )
    univariate_rows: list[dict[str, Any]] = []
    for feature_name in NEW_FEATURES:
        values = new_features[feature_name].to_numpy(dtype=float)
        summary = summarize_folded_oof_ranking(
            labels == 1, values, test_folds, budget=CANDIDATE_BUDGET, tie_breakers=ties
        )["metrics"]
        family = next(key for key, members in FEATURE_FAMILIES.items() if feature_name in members)
        univariate_rows.append(
            {
                "feature_id": feature_name,
                "feature_family": family,
                "macro_fold_spu_auc": summary["macro_fold_positive_vs_unlabeled_rank_auc"],
                "known_positive_recovered_at_58": summary["positive_recovered_at_budget"],
                "known_positive_recall_at_58": summary["positive_recall_at_budget"],
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(univariate_rows), normalized, diagnostics


def leave_one_burst_out(
    candidates: pd.DataFrame, new_features: pd.DataFrame
) -> pd.DataFrame:
    labels = (candidates["training_state"].to_numpy() == "positive").astype(np.int8)
    partitions = candidates["partition_id"].to_numpy(dtype=int)
    existing = candidates[[f"feature__{name}" for name in EXISTING_FEATURES]].to_numpy(dtype=float)
    new = new_features.loc[:, NEW_FEATURES].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for name, matrix, feature_names in (
        ("retrained_existing_linear", existing, EXISTING_FEATURES),
        ("augmented_linear", np.column_stack([existing, new]), EXISTING_FEATURES + NEW_FEATURES),
    ):
        for burst in sorted(BURSTS_UI_INCLUSIVE):
            test = partitions == burst
            train = ~test
            preprocessor = RobustFoldPreprocessor(clip=10.0, add_missing_indicators=True)
            train_values = preprocessor.fit_transform(matrix[train], labels[train] == 1, feature_names=feature_names)
            model = PenalizedLogisticRanker(
                penalty_strength=0.01,
                l1_ratio=0.0,
                max_iterations=MAX_LOGISTIC_ITERATIONS,
                tolerance=1e-6,
            )
            model.fit(train_values, labels[train], sample_weight=_balanced_weights(labels[train]))
            raw = model.predict_score(preprocessor.transform(matrix[test]))
            summary = summarize_folded_oof_ranking(
                labels[test] == 1,
                raw,
                [np.arange(int(np.sum(test)))],
                budget=max(1, round(CANDIDATE_BUDGET * np.sum(test) / len(labels))),
                tie_breakers=candidates.loc[test, "candidate_id"].to_numpy(dtype=str),
            )["metrics"]
            rows.append(
                {
                    "model_id": name,
                    "heldout_burst": burst,
                    "candidate_count": int(np.sum(test)),
                    "positive_count": int(np.sum(labels[test])),
                    "spu_auc": summary["positive_vs_unlabeled_rank_auc"],
                    "proportional_budget": summary["candidate_budget"],
                    "known_positive_recovered": summary["positive_recovered_at_budget"],
                }
            )
    return pd.DataFrame(rows)


def render_figures(output: Path, model_metrics: pd.DataFrame, univariate: pd.DataFrame, lobo: pd.DataFrame) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    order = model_metrics.sort_values("macro_fold_spu_auc").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].scatter(order["macro_fold_spu_auc"], np.arange(len(order)), color="#2457A6", s=55)
    axes[0].set_yticks(np.arange(len(order)), order["model_id"])
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel("Macro held-fold SPU-AUC")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].scatter(order["known_positive_recall_at_58"], np.arange(len(order)), color="#C26B28", s=55)
    axes[1].set_yticks(np.arange(len(order)), [])
    axes[1].set_xlim(0.0, 0.65)
    axes[1].set_xlabel("Known-positive recall at global budget 58")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Feature Atlas v1 | same 1,619-candidate union and frozen five folds")
    fig.savefig(figures / "model_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)

    feature_order = univariate.sort_values("macro_fold_spu_auc").reset_index(drop=True)
    colors = {
        "temporal_envelope": "#6B77B8",
        "soma_morphology": "#3A8C75",
        "map_source_consistency": "#B65F77",
        "nuisance_competition": "#B07A2A",
    }
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for family, group in feature_order.groupby("feature_family", sort=False):
        ax.scatter(group["macro_fold_spu_auc"], group.index, label=family.replace("_", " "), color=colors[family], s=50)
    ax.axvline(0.5, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(feature_order)), feature_order["feature_id"])
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("Macro held-fold SPU-AUC (standalone feature)")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Standalone features diagnose role; they are not calibrated neuron probabilities")
    fig.savefig(figures / "univariate_feature_utility.png", dpi=180, facecolor="white")
    plt.close(fig)

    pivot = lobo.pivot(index="heldout_burst", columns="model_id", values="spu_auc")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    burst_colors = {1: "#2457A6", 2: "#3A8C75", 3: "#B07A2A", 4: "#B65F77"}
    for burst, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row["retrained_existing_linear"], row["augmented_linear"]],
            marker="o",
            color=burst_colors[int(burst)],
            alpha=0.9,
            label=f"Burst {int(burst)}",
        )
    ax.set_xticks([0, 1], ["existing", "augmented"])
    ax.set_ylabel("Leave-one-burst-out SPU-AUC")
    ax.set_ylim(0.970, 1.001)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("Burst-held-out sensitivity | focused scale; exact values in table")
    fig.savefig(figures / "leave_one_burst_out.png", dpi=180, facecolor="white")
    plt.close(fig)


def default_paths(repo_root: Path, data_root: Path) -> AtlasPaths:
    run = repo_root / "Outputs/NeuronIdentifiability/NREV-EXP-0021/runs/NREV-RUN-EXP-0021-SCREEN-20260830-B/tables"
    return AtlasPaths(
        candidates=run / "harmonized_candidates.tsv",
        oof_scores=run / "oof_scores.tsv",
        carrier=data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy",
        ica=data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/recovery_msica.npy",
    )


def run(paths: AtlasPaths, output_root: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve()
    partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial.exists():
        raise FileExistsError(f"refusing existing output root: {output_root}")
    candidates, sources = validate_inputs(paths)
    preflight = {
        "status": "passed",
        "candidate_count": len(candidates),
        "positive_count": int((candidates["training_state"] == "positive").sum()),
        "unlabeled_count": int((candidates["training_state"] == "unlabeled").sum()),
        "fold_ids": sorted(int(value) for value in candidates["fold_id"].unique()),
        "new_feature_ids": list(NEW_FEATURES),
        "existing_feature_ids": list(EXISTING_FEATURES),
        "sources": sources,
        "interpretation": "positive-versus-unlabeled ranking; unlabeled is unknown, not negative",
    }
    if preflight_only:
        return preflight

    partial.mkdir(parents=True)
    atomic_json(partial / "preflight.json", preflight)
    carrier = np.load(paths.carrier, mmap_mode="r")
    ica = np.load(paths.ica, mmap_mode="r")
    new_features = extract_features(candidates, carrier, ica)
    feature_table = pd.concat(
        [candidates[["candidate_id", "partition_id", "training_state", "fold_id", "identity_group_id", "spatial_group_id", "leakage_group_id"]], new_features],
        axis=1,
    )
    model_metrics, univariate, normalized_scores, diagnostics = evaluate_models(candidates, new_features)
    lobo = leave_one_burst_out(candidates, new_features)
    labels = candidates["training_state"].to_numpy() == "positive"
    paired = grouped_paired_bootstrap_rank_delta(
        labels,
        normalized_scores["augmented_linear"],
        normalized_scores["retrained_existing_linear"],
        candidates["identity_group_id"].to_numpy(dtype=str),
        candidates["spatial_group_id"].to_numpy(dtype=str),
        draws=BOOTSTRAP_DRAWS,
        seed=SEED,
        score_a_name="augmented_linear",
        score_b_name="retrained_existing_linear",
    )
    family_bootstraps: list[dict[str, Any]] = [paired]
    for model_name in [f"existing_plus_{family}" for family in FEATURE_FAMILIES]:
        family_bootstraps.append(
            grouped_paired_bootstrap_rank_delta(
                labels,
                normalized_scores[model_name],
                normalized_scores["retrained_existing_linear"],
                candidates["identity_group_id"].to_numpy(dtype=str),
                candidates["spatial_group_id"].to_numpy(dtype=str),
                draws=BOOTSTRAP_DRAWS,
                seed=SEED,
                score_a_name=model_name,
                score_b_name="retrained_existing_linear",
            )
        )
    coefficient_rows: list[dict[str, Any]] = []
    for model_name in sorted({row["model_id"] for row in diagnostics}):
        model_diagnostics = [row for row in diagnostics if row["model_id"] == model_name]
        feature_names = sorted({name for row in model_diagnostics for name in row["coefficients"]})
        for feature_name in feature_names:
            values = np.asarray([row["coefficients"].get(feature_name, np.nan) for row in model_diagnostics], dtype=float)
            finite = values[np.isfinite(values)]
            coefficient_rows.append(
                {
                    "model_id": model_name,
                    "feature_id": feature_name,
                    "median_standardized_coefficient": float(np.median(finite)),
                    "minimum_standardized_coefficient": float(np.min(finite)),
                    "maximum_standardized_coefficient": float(np.max(finite)),
                    "same_sign_fraction": float(max(np.mean(finite > 0), np.mean(finite < 0))),
                }
            )
    coefficient_frame = pd.DataFrame(coefficient_rows)
    correlation = new_features.corr(method="spearman")
    correlation_rows = []
    for left_index, left in enumerate(NEW_FEATURES):
        for right in NEW_FEATURES[left_index + 1 :]:
            correlation_rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "spearman_rho": float(correlation.loc[left, right]),
                    "absolute_spearman_rho": float(abs(correlation.loc[left, right])),
                }
            )
    correlation_frame = pd.DataFrame(correlation_rows).sort_values("absolute_spearman_rho", ascending=False)
    score_frame = candidates[["candidate_id", "training_state", "fold_id"]].copy()
    for model_name, values in normalized_scores.items():
        score_frame[f"score__{model_name}__fold_percentile"] = values

    tables = partial / "tables"
    tables.mkdir()
    atomic_tsv(tables / "candidate_feature_atlas.tsv", feature_table)
    atomic_tsv(tables / "model_metrics.tsv", model_metrics)
    atomic_tsv(tables / "univariate_feature_metrics.tsv", univariate)
    atomic_tsv(tables / "leave_one_burst_out.tsv", lobo)
    atomic_tsv(tables / "oof_scores.tsv", score_frame)
    atomic_tsv(tables / "coefficient_summary.tsv", coefficient_frame)
    atomic_tsv(tables / "new_feature_spearman.tsv", correlation_frame)
    atomic_json(tables / "fold_diagnostics.json", diagnostics)
    atomic_json(tables / "paired_group_bootstrap.json", paired)
    atomic_json(tables / "family_paired_group_bootstraps.json", family_bootstraps)
    render_figures(partial, model_metrics, univariate, lobo)

    metric_index = model_metrics.set_index("model_id")
    existing = metric_index.loc["retrained_existing_linear"]
    augmented = metric_index.loc["augmented_linear"]
    family_rows = model_metrics[model_metrics["model_id"].str.startswith("existing_plus_")].copy()
    family_rows["delta_macro_auc_vs_existing"] = family_rows["macro_fold_spu_auc"] - float(existing["macro_fold_spu_auc"])
    best_family = family_rows.sort_values("delta_macro_auc_vs_existing", ascending=False).iloc[0]
    best_univariate = univariate.sort_values("macro_fold_spu_auc", ascending=False).iloc[0]
    summary = {
        "schema_version": 1,
        "status": "complete_exploratory",
        "population": {"candidates": len(candidates), "positives": int(np.sum(labels)), "unlabeled": int(np.sum(~labels))},
        "existing_linear_macro_fold_spu_auc": float(existing["macro_fold_spu_auc"]),
        "augmented_linear_macro_fold_spu_auc": float(augmented["macro_fold_spu_auc"]),
        "augmented_delta_macro_fold_spu_auc": float(augmented["macro_fold_spu_auc"] - existing["macro_fold_spu_auc"]),
        "existing_recovered_at_58": int(existing["known_positive_recovered_at_58"]),
        "augmented_recovered_at_58": int(augmented["known_positive_recovered_at_58"]),
        "best_incremental_family": str(best_family["model_id"]).removeprefix("existing_plus_"),
        "best_incremental_family_delta_macro_auc": float(best_family["delta_macro_auc_vs_existing"]),
        "best_standalone_new_feature": str(best_univariate["feature_id"]),
        "best_standalone_new_feature_macro_auc": float(best_univariate["macro_fold_spu_auc"]),
        "maximum_new_feature_absolute_spearman": float(correlation_frame.iloc[0]["absolute_spearman_rho"]),
        "maximum_new_feature_correlation_pair": [
            str(correlation_frame.iloc[0]["feature_a"]),
            str(correlation_frame.iloc[0]["feature_b"]),
        ],
        "paired_group_bootstrap": paired,
        "family_paired_group_bootstraps": family_bootstraps,
        "scientific_promotion": False,
        "scientific_audit": "incomplete_new_rankings_not_rendered_as_full_model-annotation audit",
        "limitations": [
            "single recording and historically label-informed event windows",
            "positive-versus-unlabeled metrics do not estimate precision or specificity",
            "features are evaluated on a frozen proposal union, not end-to-end proposal recall",
            "ICA split-half feature measures temporal map stability of a frozen ICA representation; ICA is not refit in each half",
        ],
    }
    atomic_json(partial / "summary.json", summary)
    validation = {
        "status": "passed",
        "checks": {
            "candidate_count": len(feature_table) == 1619,
            "positive_count": int(np.sum(labels)) == 78,
            "new_feature_count": len(NEW_FEATURES) == 14,
            "all_features_finite": bool(np.all(np.isfinite(new_features.to_numpy(dtype=float)))),
            "all_models_scored": bool(score_frame.filter(like="score__").notna().all().all()),
            "five_folds": sorted(candidates["fold_id"].unique().tolist()) == [1, 2, 3, 4, 5],
            "four_lobo_bursts_per_model": bool((lobo.groupby("model_id").size() == 4).all()),
            "figures_present": len(list((partial / "figures").glob("*.png"))) == 3,
        },
        "share_state": "share_with_caveats",
    }
    if not all(validation["checks"].values()):
        validation["status"] = "failed"
    atomic_json(partial / "validation.json", validation)
    atomic_json(
        partial / "llm_context.json",
        {
            "experiment": "Spon Ca Burst Feature Atlas v1",
            "grain": "1,619 frozen event candidates; 78 known-positive anchors and 1,541 unknown candidates",
            "primary_comparison": "augmented linear versus retrained existing 12-feature linear model on frozen five-fold assignments",
            "primary_tables": [
                "tables/model_metrics.tsv",
                "tables/univariate_feature_metrics.tsv",
                "tables/leave_one_burst_out.tsv",
                "tables/paired_group_bootstrap.json",
                "tables/family_paired_group_bootstraps.json",
                "tables/new_feature_spearman.tsv",
            ],
            "interpretation": "exploratory SPU ranking only; no precision, specificity, probability, or independent-recording claim",
        },
    )
    (partial / "1_Expert_Annotations").mkdir()
    (partial / "2_Model_Annotations").mkdir()
    (partial / "3_Comparison").mkdir()
    (partial / "1_Expert_Annotations/README.md").write_text(
        "# Expert annotations\n\nFrozen canonical-v7 identities are reused; no labels or coordinates were changed.\n",
        encoding="utf-8",
    )
    (partial / "2_Model_Annotations/README.md").write_text(
        "# Model annotations\n\nThe candidate universe is frozen, but new rankings have not yet been rendered into the complete scientific-audit media set.\n",
        encoding="utf-8",
    )
    (partial / "3_Comparison/README.md").write_text(
        "# Comparison\n\nPrimary numerical comparisons and figures are in ../tables and ../figures.\n",
        encoding="utf-8",
    )
    (partial / "REPORT.md").write_text(
        "# Spon Ca Burst Feature Atlas v1\n\n"
        "This bounded run evaluates analytically defined feature families on the frozen Run-B candidate union and folds. "
        "Read `summary.json` first. Unlabeled candidates are unknown, not negative; scientific promotion is false.\n",
        encoding="utf-8",
    )
    atomic_json(partial / "status.json", {"status": summary["status"], "validation": validation["status"], "scientific_promotion": False})
    files = sorted(path for path in partial.rglob("*") if path.is_file() and path.name != "artifact_index.json")
    atomic_json(
        partial / "artifact_index.json",
        {"artifacts": [{"path": str(path.relative_to(partial)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]},
    )
    partial.replace(output_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    paths = default_paths(args.repo_root.resolve(), args.data_root.resolve())
    print(json.dumps(run(paths, args.output_root, preflight_only=args.preflight_only), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
