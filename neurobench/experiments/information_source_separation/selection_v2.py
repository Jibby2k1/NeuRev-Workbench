"""Corrected finalist selection that separates recovery from abstention controls."""
from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Iterable

import numpy as np


def select_finalists_resolvable_only(
    rows: Iterable[dict[str, Any]],
    *,
    expected_fixture_count: int,
    finalists_per_new_method: int = 2,
    minimum_converged_fraction: float = 0.8,
    require_unresolved_accuracy: bool = True,
) -> dict[str, Any]:
    """Rank on resolvable cases and use unresolved cases only for abstention."""
    if expected_fixture_count < 1 or not 0 <= minimum_converged_fraction <= 1:
        raise ValueError("invalid expected count or convergence fraction")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_id"]), str(row["configuration_json"]))].append(row)
    summaries = []
    for (method_id, configuration), values in grouped.items():
        recovery = [row for row in values if not bool(row["unresolved_expected"])]
        unresolved = [row for row in values if bool(row["unresolved_expected"])]
        complete = len(values) == expected_fixture_count
        finite = bool(recovery) and all(
            np.isfinite(float(row["mean_absolute_correlation"])) for row in recovery
        )
        converged_fraction = float(np.mean([bool(row["converged"]) for row in values]))
        unresolved_accuracy = (
            float(np.mean([bool(row["reported_unresolved"]) for row in unresolved]))
            if unresolved else 1.0
        )
        passed = bool(
            complete and finite and converged_fraction >= minimum_converged_fraction
            and (not require_unresolved_accuracy or unresolved_accuracy == 1.0)
        )
        parameters = json.loads(configuration)
        summaries.append({
            "method_id": method_id,
            "configuration_json": configuration,
            "fit_count": len(values),
            "recovery_fit_count": len(recovery),
            "unresolved_fit_count": len(unresolved),
            "unresolved_excluded_from_recovery_aggregation": True,
            "complete": complete,
            "finite": finite,
            "converged_fraction": converged_fraction,
            "unresolved_accuracy": unresolved_accuracy,
            "mean_absolute_correlation": float(np.mean([
                float(row["mean_absolute_correlation"]) for row in recovery
            ])),
            "worst_absolute_correlation": float(np.min([
                float(row["worst_absolute_correlation"]) for row in recovery
            ])),
            "mean_absolute_crosstalk": float(np.mean([
                float(row["mean_absolute_crosstalk"]) for row in recovery
            ])),
            "complexity_rank": int(parameters.get("rank", 10**6)),
            "passed": passed,
        })
    selected = []
    for method_id in sorted({row["method_id"] for row in summaries}):
        eligible = [
            row for row in summaries
            if row["method_id"] == method_id and row["passed"]
        ]
        eligible.sort(key=lambda row: (
            -row["mean_absolute_correlation"],
            -row["worst_absolute_correlation"],
            row["mean_absolute_crosstalk"],
            row["complexity_rank"],
            row["configuration_json"],
        ))
        count = 1 if method_id == "pca_reference" else finalists_per_new_method
        selected.extend(eligible[:count])
    return {
        "schema_version": 2,
        "status": "finalists_selected" if selected else "no_method_passed",
        "expected_fixture_count": int(expected_fixture_count),
        "minimum_converged_fraction": float(minimum_converged_fraction),
        "require_unresolved_accuracy": bool(require_unresolved_accuracy),
        "recovery_aggregation": "resolvable_fixtures_only",
        "configuration_summaries": summaries,
        "selected_finalists": selected,
    }
