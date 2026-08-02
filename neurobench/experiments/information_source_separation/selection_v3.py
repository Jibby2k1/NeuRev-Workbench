"""Parsimonious posthoc selection with the manifest equivalence margin."""
from __future__ import annotations

from typing import Any, Iterable

from .selection_v2 import select_finalists_resolvable_only


def select_finalists_with_equivalence_margin(
    rows: Iterable[dict[str, Any]],
    *,
    expected_fixture_count: int,
    equivalence_margin: float,
    finalists_per_new_method: int = 2,
    minimum_converged_fraction: float = 0.8,
    require_unresolved_accuracy: bool = True,
) -> dict[str, Any]:
    """Prefer lower rank when recovery is within the frozen margin of best."""
    if not 0 <= equivalence_margin < 1:
        raise ValueError("equivalence_margin must be in [0,1)")
    base = select_finalists_resolvable_only(
        rows,
        expected_fixture_count=expected_fixture_count,
        finalists_per_new_method=finalists_per_new_method,
        minimum_converged_fraction=minimum_converged_fraction,
        require_unresolved_accuracy=require_unresolved_accuracy,
    )
    selected = []
    decision_rows = []
    summaries = base["configuration_summaries"]
    for method_id in sorted({row["method_id"] for row in summaries}):
        eligible = [
            row for row in summaries
            if row["method_id"] == method_id and row["passed"]
        ]
        if not eligible:
            continue
        best_recovery = max(row["mean_absolute_correlation"] for row in eligible)
        competitive = [
            row for row in eligible
            if best_recovery - row["mean_absolute_correlation"] <= equivalence_margin
        ]
        competitive.sort(key=lambda row: (
            row["complexity_rank"],
            -row["mean_absolute_correlation"],
            -row["worst_absolute_correlation"],
            row["mean_absolute_crosstalk"],
            row["configuration_json"],
        ))
        count = 1 if method_id == "pca_reference" else finalists_per_new_method
        chosen = competitive[:count]
        selected.extend(chosen)
        decision_rows.append({
            "method_id": method_id,
            "best_resolvable_mean_correlation": best_recovery,
            "equivalence_floor": best_recovery - equivalence_margin,
            "eligible_configuration_count": len(eligible),
            "competitive_configuration_count": len(competitive),
            "selection_rule": "lowest_rank_within_margin_then_recovery",
        })
    return {
        **base,
        "schema_version": 3,
        "equivalence_margin": float(equivalence_margin),
        "selection_rule": "resolvable_only_then_parsimony_within_equivalence_margin",
        "equivalence_decisions": decision_rows,
        "selected_finalists": selected,
        "status": "finalists_selected" if selected else "no_method_passed",
    }
