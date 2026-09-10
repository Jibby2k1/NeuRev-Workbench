"""Deterministic complete-registry analysis for the real-data ICA factorial."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from neurobench.experiments.frame_difference import _atomic_json

from .real_config import RealDataConfig
from .real_runner import _load_specs
from .real_selection import nested_leave_one_burst_out, paired_seed_groups, pareto_front


DISCRETE_FACTORS = (
    "family", "whitening_geometry", "covariance_scope", "causality",
    "objective", "rank", "spatial_width_px", "temporal_width_frames",
)
CONTINUOUS_FACTORS = (
    "objective_scale", "raw_preserving_blend", "spatial_exponent",
    "spatial_shrinkage", "spatial_eigen_floor_ratio", "temporal_exponent",
    "temporal_shrinkage", "temporal_eigen_floor_ratio", "joint_exponent",
    "joint_shrinkage", "joint_eigen_floor_ratio",
)

RESPONSE_METRICS = (
    "temporal_frequency_centroid_hz",
    "temporal_frequency_bandwidth_hz",
    "spatial_frequency_centroid_cycles_per_px",
    "spatial_frequency_bandwidth_cycles_per_px",
    "separability_residual_fraction",
    "best_rank1_separable_energy_fraction",
    "kernel_l2",
    "kernel_sum",
)


def _utility(row: dict[str, Any]) -> float:
    return float(row["label_metrics"]["macro_known_positive_recall"])


def _bootstrap_burst_ci(row: dict[str, Any], *, budget: int, seed: int = 20260831,
                        replicates: int = 4096) -> dict[str, float]:
    recalls = []
    for fold in row["label_metrics"]["folds"]:
        recalls.append(float(next(item["recall"] for item in fold["budgets"]
                                  if int(item["budget"]) == budget)))
    values = np.asarray(recalls)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {"estimate": float(values.mean()), "lower_95": float(np.quantile(samples, .025)),
            "upper_95": float(np.quantile(samples, .975)), "grouping_unit": "burst"}


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)), "mean": float(np.mean(finite)),
        "median": float(np.median(finite)), "standard_deviation": float(np.std(finite)),
        "q05": float(np.quantile(finite, .05)), "q95": float(np.quantile(finite, .95)),
    }


def _response_groups(
    rows: list[dict[str, Any]], factors: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        response = row.get("component_response")
        if isinstance(response, dict):
            groups[tuple(str(row.get(factor)) for factor in factors)].append(response)
    result = []
    for levels, responses in sorted(groups.items()):
        metrics = {
            metric: _numeric_summary([
                float(response[metric]) for response in responses
                if response.get(metric) is not None
            ])
            for metric in RESPONSE_METRICS
        }
        result.append({
            **dict(zip(factors, levels)), "fit_count": len(responses), "metrics": metrics,
        })
    return result


def _conditional_factor_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe effects within family x whitening strata to avoid marginal confounding."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["family"]), str(row["whitening_geometry"]))].append(row)
    discrete = []
    continuous = []
    for (family, geometry), selected in sorted(strata.items()):
        for factor in DISCRETE_FACTORS:
            if factor in {"family", "whitening_geometry"}:
                continue
            levels: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                levels[str(row.get(factor))].append(_utility(row))
            if len(levels) < 2:
                continue
            discrete.append({
                "family": family, "whitening_geometry": geometry, "factor": factor,
                "levels": [{
                    "level": level, "fit_count": len(values),
                    "mean_macro_known_positive_recall_at_58": float(np.mean(values)),
                    "median_macro_known_positive_recall_at_58": float(np.median(values)),
                } for level, values in sorted(levels.items())],
            })
        for factor in CONTINUOUS_FACTORS:
            pairs = [(float(row[factor]), _utility(row)) for row in selected
                     if row.get(factor) is not None and np.isfinite(float(row[factor]))]
            if len(pairs) < 8 or len({value for value, _ in pairs}) < 3:
                continue
            statistic, pvalue = spearmanr(
                [value for value, _ in pairs], [utility for _, utility in pairs]
            )
            continuous.append({
                "family": family, "whitening_geometry": geometry, "factor": factor,
                "fit_count": len(pairs), "spearman_rank_correlation": float(statistic),
                "descriptive_two_sided_pvalue": float(pvalue),
                "semantics": "within_stratum_exploratory_association_not_causal_effect",
            })
    return {
        "stratification": ["family", "whitening_geometry"],
        "discrete": discrete, "continuous": continuous,
        "interpretation": (
            "Prefer these within-stratum summaries over marginal means; residual dependence "
            "among Sobol coordinates still precludes causal attribution."
        ),
    }


def _design_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, int] = defaultdict(int)
    point_kinds: dict[str, int] = defaultdict(int)
    for row in rows:
        cells[str(row.get("cell_id", "missing"))] += 1
        point_kinds[str(row.get("point_kind", "missing"))] += 1
    counts = list(cells.values())
    return {
        "categorical_ordinal_policy": "exact_cartesian_product_of_applicable_cells",
        "continuous_policy": "scrambled_sobol_points_within_each_applicable_cell",
        "cell_count": len(cells), "point_kind_counts": dict(sorted(point_kinds.items())),
        "fits_per_cell_minimum": min(counts), "fits_per_cell_maximum": max(counts),
        "formal_sobol_indices_identified": False,
        "reason": "point set is not a Saltelli A_B construction",
    }


def _learned_parameter_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    whitening_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        diagnostics = row.get("external_whitening_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        key = (
            str(row.get("whitening_geometry")), str(row.get("covariance_scope")),
            str(row.get("spatial_width_px")), str(row.get("temporal_width_frames")),
            str(row.get("raw_preserving_blend")),
        )
        whitening_groups[key].append(diagnostics)
    whitening = []
    for key, diagnostics in sorted(whitening_groups.items()):
        whitening.append({
            **dict(zip((
                "whitening_geometry", "covariance_scope", "spatial_width_px",
                "temporal_width_frames", "raw_preserving_blend",
            ), key)),
            "fit_count": len(diagnostics),
            "resolved_fraction": float(np.mean([
                bool(item.get("resolved", False)) for item in diagnostics
            ])),
            "maximum_condition_number": _numeric_summary([
                float(item["maximum_condition_number"]) for item in diagnostics
            ]),
            "minimum_effective_rank_fraction": _numeric_summary([
                float(item["minimum_effective_rank_fraction"]) for item in diagnostics
            ]),
        })
    return {
        "response_metric_semantics": {
            "temporal_frequency_units": "Hz using the frozen 20 ms frame period",
            "spatial_frequency_units": "cycles per pixel",
            "rank_and_support_interpretation": (
                "descriptive distributions of complete learned component-response summaries; "
                "not biological source identities"
            ),
        },
        "response_by_family_and_rank": _response_groups(rows, ("family", "rank")),
        "response_by_family_and_temporal_width": _response_groups(
            rows, ("family", "temporal_width_frames")
        ),
        "response_by_family_and_spatial_width": _response_groups(
            rows, ("family", "spatial_width_px")
        ),
        "response_by_family_and_whitening_geometry": _response_groups(
            rows, ("family", "whitening_geometry")
        ),
        "whitening_by_geometry_scope_support_and_blend": whitening,
    }


def _render_learned_parameter_figures(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Render all-fit response and whitening summaries without choosing components."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures" / "learned_parameters"
    figure_root.mkdir(parents=True, exist_ok=True)
    families = sorted({str(row["family"]) for row in rows})
    colors = {family: plt.get_cmap("tab10")(index) for index, family in enumerate(families)}
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for family in families:
        selected = [row for row in rows if str(row["family"]) == family]
        for axis, metric, ylabel in (
            (axes[0], "temporal_frequency_centroid_hz", "Temporal centroid (Hz)"),
            (axes[1], "spatial_frequency_centroid_cycles_per_px", "Spatial centroid (cycles/pixel)"),
        ):
            points = [(float(row["rank"]), float(row["component_response"][metric]))
                      for row in selected if row.get("component_response", {}).get(metric) is not None]
            if points:
                axis.scatter(*zip(*points), s=8, alpha=.16, color=colors[family], label=family)
            axis.set_xlabel("ICA rank (number of learned components)"); axis.set_ylabel(ylabel)
            axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles: axes[0].legend(handles, labels, fontsize=8)
    fig.suptitle("Learned response frequency versus number of ICA components")
    path = figure_root / "frequency_response_vs_rank.png"
    fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path.relative_to(root)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    support_specs = (
        (axes[0], "temporal_width_frames", "temporal_frequency_centroid_hz",
         "Temporal support (frames)", "Temporal centroid (Hz)"),
        (axes[1], "spatial_width_px", "spatial_frequency_centroid_cycles_per_px",
         "Spatial support (pixels)", "Spatial centroid (cycles/pixel)"),
    )
    for axis, support, metric, xlabel, ylabel in support_specs:
        for family in families:
            points = [(float(row[support]), float(row["component_response"][metric]))
                      for row in rows if str(row["family"]) == family
                      and row.get(support) is not None
                      and row.get("component_response", {}).get(metric) is not None]
            if points:
                axis.scatter(*zip(*points), s=8, alpha=.16, color=colors[family], label=family)
        axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles: axes[0].legend(handles, labels, fontsize=8)
    fig.suptitle("Learned response frequency versus temporal and spatial support")
    path = figure_root / "frequency_response_vs_support.png"
    fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path.relative_to(root)))

    whitening_rows = [row for row in rows
                      if isinstance(row.get("external_whitening_diagnostics"), dict)]
    geometries = sorted({str(row["whitening_geometry"]) for row in whitening_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for index, geometry in enumerate(geometries):
        selected = [row["external_whitening_diagnostics"] for row in whitening_rows
                    if str(row["whitening_geometry"]) == geometry]
        axes[0].scatter(np.full(len(selected), index), [item["maximum_condition_number"] for item in selected],
                        s=8, alpha=.16)
        axes[1].scatter(np.full(len(selected), index), [item["minimum_effective_rank_fraction"] for item in selected],
                        s=8, alpha=.16)
    for axis in axes:
        axis.set_xticks(range(len(geometries)), geometries, rotation=25, ha="right")
        axis.grid(alpha=.2)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Maximum whitening condition number")
    axes[1].set_ylabel("Minimum effective-rank fraction")
    fig.suptitle("Learned whitening diagnostics by geometry")
    path = figure_root / "whitening_diagnostics_by_geometry.png"
    fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path.relative_to(root)))
    return outputs


def summarize_factorial(rows: list[dict[str, Any]], *, budget: int = 58) -> dict[str, Any]:
    if not rows:
        raise ValueError("factorial summary requires rows")
    identifiers = [str(row["fit_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("factorial rows contain duplicate fit IDs")
    discrete = {}
    for factor in DISCRETE_FACTORS:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(factor))].append(_utility(row))
        discrete[factor] = [{
            "level": level, "fit_count": len(values),
            "mean_macro_known_positive_recall_at_58": float(np.mean(values)),
            "median_macro_known_positive_recall_at_58": float(np.median(values)),
        } for level, values in sorted(groups.items())]
    continuous = []
    for factor in CONTINUOUS_FACTORS:
        selected = [(float(row[factor]), _utility(row)) for row in rows
                    if row.get(factor) is not None and np.isfinite(float(row[factor]))]
        if len(selected) < 8 or len({item[0] for item in selected}) < 3:
            continue
        statistic, pvalue = spearmanr([item[0] for item in selected],
                                      [item[1] for item in selected])
        continuous.append({
            "factor": factor, "fit_count": len(selected),
            "spearman_rank_correlation": float(statistic),
            "descriptive_two_sided_pvalue": float(pvalue),
            "semantics": "exploratory_monotonic_sensitivity_proxy_not_formal_sobol_index",
        })
    eligible = [row for row in rows if bool(row.get("real_data_fit_converged", False))
                and bool(row.get("external_whitening_diagnostics", {}).get("resolved", True))]
    pareto_rows = []
    for row in eligible:
        pareto_rows.append({
            **row,
            "macro_recall": _utility(row),
            "mrr": float(row["label_metrics"]["mean_reciprocal_rank"]),
            "condition": float(row["real_data_condition_number"]),
        })
    front = pareto_front(pareto_rows, maximize=("macro_recall", "mrr"),
                         minimize=("condition",)) if pareto_rows else []
    loo = nested_leave_one_burst_out(eligible, budget=budget) if eligible else None
    top = sorted(eligible, key=lambda row: (-_utility(row), row["fit_id"]))[:20]
    return {
        "schema_version": 1, "fit_count": len(rows), "unique_fit_count": len(set(identifiers)),
        "real_data_converged_count": sum(bool(row.get("real_data_fit_converged")) for row in rows),
        "finalist_eligible_count": len(eligible),
        "discrete_factor_summaries": discrete,
        "continuous_sensitivity": continuous,
        "conditional_factor_summaries": _conditional_factor_summaries(rows),
        "design_audit": _design_audit(rows),
        "learned_parameter_summary": _learned_parameter_summary(rows),
        "formal_sobol_indices_identified": False,
        "formal_sobol_reason": "scrambled point design lacks Saltelli A_B matrices; report response-surface proxies",
        "preliminary_pareto_fit_ids": [row["fit_id"] for row in front],
        "protected_leave_one_burst_out": loo,
        "paired_seed_group_count": len(paired_seed_groups(eligible)),
        "top_fit_burst_bootstrap": [{
            "fit_id": row["fit_id"], "family": row["family"],
            "whitening_geometry": row["whitening_geometry"],
            "macro_recall_at_58": _bootstrap_burst_ci(row, budget=budget),
        } for row in top],
        "claim_scope": "exploratory_within_recording_sparse_positive_unmatched_unknown",
    }


def run_complete_factorial_analysis(real: RealDataConfig) -> dict[str, Any]:
    """Require every expected fit, then write the deterministic S3 package."""
    geometries = (
        "none", "spatial", "temporal", "joint_spatiotemporal",
        "spatial_then_temporal", "temporal_then_spatial",
    )
    expected_rows = [row for geometry in geometries
                     for row in _load_specs(real.synthetic_registry.parent.parent,
                                            whitening_geometry=geometry)]
    expected = {row["fit_id"] for row in expected_rows}
    rows = []; observed = set(); shard_manifest = []
    for geometry in geometries:
        tag = geometry.upper()
        for shard in range(real.fitting.shard_count):
            if geometry == "none":
                root = real.output_dir / "stages" / f"S2B_EVENT_STANDARDIZED_SHARD_{shard:02d}_OF_{real.fitting.shard_count:02d}"
            else:
                root = real.output_dir / "stages" / f"S2C_{tag}_WHITENING_SHARD_{shard:02d}_OF_{real.fitting.shard_count:02d}"
            validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
            if validation.get("status") != "pass":
                raise RuntimeError(f"unvalidated factorial shard: {root}")
            count = 0
            with (root / "fit_results.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip(): continue
                    row = json.loads(line); fit_id = str(row["fit_id"])
                    if fit_id in observed: raise RuntimeError(f"duplicate factorial fit: {fit_id}")
                    observed.add(fit_id); rows.append(row); count += 1
            shard_manifest.append({"geometry": geometry, "shard": shard,
                                   "fit_count": count, "path": str(root.relative_to(real.output_dir))})
    if observed != expected:
        raise RuntimeError(f"factorial coverage mismatch missing={len(expected-observed)} extra={len(observed-expected)}")
    root = real.output_dir / "stages" / "S3_COMPLETE_FACTORIAL_ANALYSIS"
    root.mkdir(parents=True, exist_ok=False)
    summary = summarize_factorial(rows)
    summary.update({"status": "complete", "exact_coverage": True,
                    "expected_fit_count": len(expected), "observed_fit_count": len(observed)})
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "validation.json", {
        "status": "pass", "expected_fit_count": len(expected),
        "observed_fit_count": len(observed), "unique_fit_count": len(observed),
        "missing_fit_count": 0, "extra_fit_count": 0,
    })
    _atomic_json(root / "shard_manifest.json", {"shards": shard_manifest})
    figure_paths = _render_learned_parameter_figures(root, rows)
    flat = [{
        "fit_id": row["fit_id"], "family": row["family"],
        "whitening_geometry": row["whitening_geometry"],
        "macro_known_positive_recall_at_58": _utility(row),
        "pooled_known_positive_recall_at_58": row["label_metrics"]["pooled_known_positive_recall"],
        "mean_reciprocal_rank": row["label_metrics"]["mean_reciprocal_rank"],
        "real_data_fit_converged": row["real_data_fit_converged"],
        "real_data_condition_number": row["real_data_condition_number"],
    } for row in sorted(rows, key=lambda item: item["fit_id"])]
    with (root / "fit_registry.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(flat)
    _atomic_json(root / "artifact_index.json", {"artifacts": [
        {"path": "summary.json", "role": "factorial_effects_pareto_and_protected_loo"},
        {"path": "validation.json", "role": "exact_coverage_gate"},
        {"path": "fit_registry.tsv", "role": "complete_compact_registry"},
        {"path": "shard_manifest.json", "role": "input_provenance"},
        *[{"path": path, "role": "all_fit_learned_response_or_whitening_summary"}
          for path in figure_paths],
    ]})
    return summary
