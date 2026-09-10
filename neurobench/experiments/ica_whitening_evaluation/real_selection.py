"""Frozen exploratory Pareto and protected leave-one-burst-out selection."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _fold_recall(row: dict[str, Any], burst: int, budget: int) -> float:
    fold = next(item for item in row["label_metrics"]["folds"]
                if int(item["burst_id"]) == burst)
    item = next(item for item in fold["budgets"] if int(item["budget"]) == budget)
    return float(item["recall"])


def pareto_front(
    rows: list[dict[str, Any]], *, maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return deterministic nondominated rows for declared scalar fields."""
    if not rows or not (maximize or minimize):
        raise ValueError("Pareto selection requires rows and objectives")
    def vector(row: dict[str, Any]) -> np.ndarray:
        return np.asarray([float(row[key]) for key in maximize]
                          + [-float(row[key]) for key in minimize])
    values = [vector(row) for row in rows]
    selected = []
    for index, value in enumerate(values):
        dominated = any(
            other != index and np.all(values[other] >= value)
            and np.any(values[other] > value)
            for other in range(len(values))
        )
        if not dominated: selected.append(rows[index])
    return sorted(selected, key=lambda row: str(row["fit_id"]))


def nested_leave_one_burst_out(
    rows: list[dict[str, Any]], *, budget: int = 58,
    finalists_per_training_fold: int = 3,
) -> dict[str, Any]:
    """Select only on three bursts, then measure the untouched fourth burst."""
    if finalists_per_training_fold < 1:
        raise ValueError("finalists_per_training_fold must be positive")
    folds = []; selection_counts: Counter[str] = Counter()
    for held_out in (1, 2, 3, 4):
        training = tuple(burst for burst in (1, 2, 3, 4) if burst != held_out)
        training_rows = [{
            **row,
            "training_macro_recall": float(np.mean([
                _fold_recall(row, burst, budget) for burst in training
            ])),
            "selection_complexity_rank": int(row.get("rank", 1)),
        } for row in rows]
        front = pareto_front(
            training_rows, maximize=("training_macro_recall",),
            minimize=("selection_complexity_rank",),
        )
        ranked = sorted(front, key=lambda row: (
            -float(row["training_macro_recall"]),
            int(row["selection_complexity_rank"]), str(row["fit_id"]),
        ))
        selected = ranked[:min(finalists_per_training_fold, len(ranked))]
        for row in selected: selection_counts[str(row["fit_id"])] += 1
        folds.append({
            "held_out_burst": held_out, "training_bursts": list(training),
            "selected_fit_ids": [row["fit_id"] for row in selected],
            "training_pareto_candidate_count": len(front),
            "pareto_objectives": {
                "maximize": ["training_macro_known_positive_recall"],
                "minimize": ["ICA_rank_label_independent_complexity"],
            },
            "training_macro_recall": [float(row["training_macro_recall"])
                                      for row in selected],
            "held_out_recall": [_fold_recall(row, held_out, budget) for row in selected],
        })
    protected = sorted(selection_counts, key=lambda fit_id: (-selection_counts[fit_id], fit_id))
    held_out_values = [value for fold in folds for value in fold["held_out_recall"]]
    return {
        "selection_semantics": (
            "training_burst_only_recall_rank_complexity_pareto_then_evaluate_untouched_fourth"
        ),
        "budget": budget, "folds": folds,
        "selection_frequency": dict(sorted(selection_counts.items())),
        "candidate_finalist_fit_ids": protected,
        "mean_held_out_known_positive_recall": float(np.mean(held_out_values)),
        "labels_not_used_for_fitting_or_candidate_generation": True,
        "unmatched_candidates": "unknown_not_negative",
    }


def hyperparameter_signature(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    excluded = {
        "fit_id", "seed", "label_metrics", "candidate_score_sha256_before_label_metrics",
        "real_data_fit_converged", "real_data_iterations", "real_data_objective",
        "real_data_condition_number", "real_data_explained_fraction",
        "component_response", "quiet_calibration", "external_whitening_diagnostics",
    }
    return tuple(sorted((key, value) for key, value in row.items()
                        if key not in excluded and isinstance(value, (str, int, float, bool, type(None)))))


def paired_seed_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(hyperparameter_signature(row), []).append(row)
    return [sorted(group, key=lambda row: (int(row["seed"]), row["fit_id"]))
            for group in groups.values() if len({int(row["seed"]) for row in group}) >= 2]
