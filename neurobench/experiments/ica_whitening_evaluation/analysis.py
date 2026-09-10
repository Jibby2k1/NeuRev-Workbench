"""Post-gate descriptive analysis for the truth-known ICA factorial screen."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from neurobench.experiments.frame_difference import _atomic_json


OUTCOMES = (
    "mean_truth_source_correlation",
    "mean_truth_crosstalk",
    "mean_trace_preservation",
    "converged_fraction",
    "unresolved_accuracy",
)
DISCRETE_FACTORS = (
    "family", "objective", "whitening_geometry", "covariance_scope",
    "causality", "rank", "spatial_width_px", "temporal_width_frames",
    "point_kind",
)
CONTINUOUS_FACTORS = (
    "spatial_exponent", "temporal_exponent", "joint_exponent",
    "spatial_shrinkage", "temporal_shrinkage", "joint_shrinkage",
    "spatial_eigen_floor_ratio", "temporal_eigen_floor_ratio",
    "joint_eigen_floor_ratio", "raw_preserving_blend", "objective_scale",
)


def load_shard_rows(stage_root: str | Path, shard_count: int) -> list[dict[str, Any]]:
    root = Path(stage_root)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(shard_count):
        path = root / f"S1_NUMERICAL_TRUTH_KNOWN_SHARD_{index:02d}_OF_{shard_count:02d}" / "fit_results.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["fit_id"] in seen:
                    raise RuntimeError(f"duplicate fit id: {row['fit_id']}")
                seen.add(row["fit_id"])
                rows.append(row)
    return rows


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray([value for value in values if value is not None], dtype=float)
    return result[np.isfinite(result)]


def _describe(values: Iterable[Any]) -> dict[str, float | int | None]:
    data = _finite(values)
    if not len(data):
        return {"n": 0, "mean": None, "median": None, "q05": None, "q95": None}
    return {
        "n": int(len(data)), "mean": float(np.mean(data)),
        "median": float(np.median(data)), "q05": float(np.quantile(data, .05)),
        "q95": float(np.quantile(data, .95)),
    }


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    main_effects = []
    for factor in DISCRETE_FACTORS:
        levels: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            levels[str(row.get(factor))].append(row)
        for level, selected in sorted(levels.items()):
            main_effects.append({
                "factor": factor, "level": level, "fit_count": len(selected),
                **{metric: _describe(row.get(metric) for row in selected) for metric in OUTCOMES},
            })

    cell_seed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_seed[(row["cell_id"], row["point_id"])].append(row)
    stability = []
    for (cell_id, point_id), selected in sorted(cell_seed.items()):
        stability.append({
            "cell_id": cell_id, "point_id": point_id,
            "seed_count": len({row["seed"] for row in selected}),
            **{
                f"{metric}_seed_sd": float(np.std(_finite(row.get(metric) for row in selected), ddof=1))
                if len(_finite(row.get(metric) for row in selected)) > 1 else 0.0
                for metric in OUTCOMES
            },
        })

    associations = []
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("point_kind") == "sobol":
            by_cell[row["cell_id"]].append(row)
    for cell_id, selected in sorted(by_cell.items()):
        for parameter in CONTINUOUS_FACTORS:
            for metric in OUTCOMES:
                pairs = [
                    (float(row[parameter]), float(row[metric]))
                    for row in selected
                    if row.get(parameter) is not None and row.get(metric) is not None
                    and np.isfinite(float(row[parameter])) and np.isfinite(float(row[metric]))
                ]
                if (
                    len(pairs) < 8
                    or len({left for left, _ in pairs}) < 3
                    or len({right for _, right in pairs}) < 2
                ):
                    continue
                statistic = spearmanr([left for left, _ in pairs], [right for _, right in pairs])
                associations.append({
                    "cell_id": cell_id, "parameter": parameter, "outcome": metric,
                    "n": len(pairs), "spearman_rho": float(statistic.statistic),
                    "p_value_descriptive_only": float(statistic.pvalue),
                })

    response_by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    integrity_by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for case in row.get("cases", []):
            for key, value in case.get("component_response", {}).items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    response_by_family[row["family"]][key].append(float(value))
            for key, value in case.get("integrity", {}).items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    integrity_by_family[row["family"]][key].append(float(value))

    return {
        "schema_version": 1,
        "fit_count": len(rows),
        "main_effects": main_effects,
        "paired_seed_stability": stability,
        "within_cell_continuous_associations": associations,
        "response_summaries": {
            family: {key: _describe(values) for key, values in sorted(metrics.items())}
            for family, metrics in sorted(response_by_family.items())
        },
        "signal_integrity_summaries": {
            family: {key: _describe(values) for key, values in sorted(metrics.items())}
            for family, metrics in sorted(integrity_by_family.items())
        },
        "formal_sobol_indices": {
            "status": "not_identifiable_from_this_design",
            "reason": "Scrambled Sobol coverage was used without the paired A/B/AB matrices required for variance-decomposition estimators.",
            "substitute": "Within-compatible-cell Spearman associations and descriptive response surfaces.",
        },
        "inference_scope": "descriptive truth-known computational characterization; p-values are not multiplicity-adjusted claims",
    }


def _write_flat(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def write_analysis(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    result = analyze_rows(rows)
    _atomic_json(root / "analysis.json", result)
    _write_flat(root / "factor_main_effects.tsv", result["main_effects"])
    _write_flat(root / "paired_seed_stability.tsv", result["paired_seed_stability"])
    _write_flat(root / "continuous_associations.tsv", result["within_cell_continuous_associations"])
    return result
