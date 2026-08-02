"""Preregistered staged screen/confirmation design and finalist selection."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np

from .config import InformationSeparationConfig
from .preflight import method_configuration_counts


SCREEN_CASES = (
    "isolated", "overlap", "synchronous", "similar_persistence",
    "motion_edge", "saturation", "unresolved",
)


@dataclass(frozen=True)
class FixtureSpecification:
    case_id: str
    seed: int
    snr: float


@dataclass(frozen=True)
class MethodSpecification:
    method_id: str
    parameters: dict[str, Any]

    @property
    def key(self) -> str:
        return json.dumps(
            {"method_id": self.method_id, "parameters": self.parameters},
            sort_keys=True,
            separators=(",", ":"),
        )


def screen_fixtures(config: InformationSeparationConfig) -> tuple[FixtureSpecification, ...]:
    available = set(config.generated["case_ids"])
    cases = [case for case in SCREEN_CASES if case in available]
    if len(cases) != len(SCREEN_CASES):
        raise ValueError("manifest must contain every preregistered screen case")
    seeds = list(map(int, config.generated["seeds"][:2]))
    snr_values = list(map(float, config.generated["snr_levels"]))
    median_snr = sorted(snr_values)[len(snr_values) // 2]
    return tuple(
        FixtureSpecification(case, seed, median_snr)
        for case in cases
        for seed in seeds
    )


def confirmation_fixtures(config: InformationSeparationConfig) -> tuple[FixtureSpecification, ...]:
    return tuple(
        FixtureSpecification(str(case), int(seed), float(snr))
        for case in config.generated["case_ids"]
        for seed in config.generated["seeds"]
        for snr in config.generated["snr_levels"]
    )


def method_screen(config: InformationSeparationConfig) -> tuple[MethodSpecification, ...]:
    methods = config.methods
    result: list[MethodSpecification] = []
    if methods["pca_reference"]["enabled"]:
        result.extend(
            MethodSpecification("pca_reference", {"rank": int(rank)})
            for rank in methods["pca_reference"]["ranks"]
        )
    if methods["multilag_sobi"]["enabled"]:
        result.extend(
            MethodSpecification("multilag_sobi", {
                "rank": int(rank),
                "lags": list(map(int, lags)),
                "covariance_shrinkage": float(shrinkage),
            })
            for rank in methods["multilag_sobi"]["ranks"]
            for lags in methods["multilag_sobi"]["lag_sets"]
            for shrinkage in methods["multilag_sobi"]["covariance_shrinkages"]
        )
    if methods["kernel_hsic_pairwise_rotation"]["enabled"]:
        result.extend(
            MethodSpecification("kernel_hsic_pairwise_rotation", {
                "rank": int(rank), "bandwidth_scale": float(scale),
            })
            for rank in methods["kernel_hsic_pairwise_rotation"]["ranks"]
            for scale in methods["kernel_hsic_pairwise_rotation"]["bandwidth_scales"]
        )
    if methods["knn_mi_pairwise_rotation"]["enabled"]:
        result.extend(
            MethodSpecification("knn_mi_pairwise_rotation", {
                "rank": int(rank), "neighbors": int(neighbors),
            })
            for rank in methods["knn_mi_pairwise_rotation"]["ranks"]
            for neighbors in methods["knn_mi_pairwise_rotation"]["neighbors"]
        )
    expected = sum(method_configuration_counts(config).values())
    if len(result) != expected or len({item.key for item in result}) != len(result):
        raise RuntimeError("method screen count or uniqueness differs from preflight")
    return tuple(result)


def staged_fit_counts(config: InformationSeparationConfig, *, finalists_per_new_method: int = 2) -> dict[str, int]:
    if finalists_per_new_method < 1:
        raise ValueError("finalists_per_new_method must be positive")
    screen_methods = method_screen(config)
    screen_count = len(screen_fixtures(config)) * len(screen_methods)
    active_methods = {item.method_id for item in screen_methods}
    finalist_count = sum(
        1 if method_id == "pca_reference" else finalists_per_new_method
        for method_id in active_methods
    )
    confirm_count = len(confirmation_fixtures(config)) * finalist_count
    return {
        "screen_fixture_count": len(screen_fixtures(config)),
        "screen_configuration_count": len(screen_methods),
        "screen_fit_count": screen_count,
        "maximum_finalist_configuration_count": finalist_count,
        "confirmation_fixture_count": len(confirmation_fixtures(config)),
        "maximum_confirmation_fit_count": confirm_count,
        "maximum_staged_fit_count": screen_count + confirm_count,
        "full_cartesian_fit_count": (
            len(confirmation_fixtures(config)) * len(screen_methods)
        ),
    }


def select_finalists(
    rows: Iterable[dict[str, Any]],
    *,
    expected_fixture_count: int,
    finalists_per_new_method: int = 2,
    minimum_converged_fraction: float = 0.8,
    require_unresolved_accuracy: bool = True,
) -> dict[str, Any]:
    """Apply the preregistered numerical/unresolved/recovery selection rule."""
    if expected_fixture_count < 1 or not 0 <= minimum_converged_fraction <= 1:
        raise ValueError("invalid expected count or convergence fraction")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method_id = str(row["method_id"])
        configuration = str(row["configuration_json"])
        grouped[(method_id, configuration)].append(row)
    summaries = []
    for (method_id, configuration), values in grouped.items():
        complete = len(values) == expected_fixture_count
        finite = all(np.isfinite(float(row["mean_absolute_correlation"])) for row in values)
        converged_fraction = float(np.mean([bool(row["converged"]) for row in values]))
        unresolved_rows = [row for row in values if bool(row["unresolved_expected"])]
        unresolved_accuracy = (
            float(np.mean([bool(row["reported_unresolved"]) for row in unresolved_rows]))
            if unresolved_rows else 1.0
        )
        passed = bool(
            complete
            and finite
            and converged_fraction >= minimum_converged_fraction
            and (not require_unresolved_accuracy or unresolved_accuracy == 1.0)
        )
        parameters = json.loads(configuration)
        summaries.append({
            "method_id": method_id,
            "configuration_json": configuration,
            "fit_count": len(values),
            "complete": complete,
            "finite": finite,
            "converged_fraction": converged_fraction,
            "unresolved_accuracy": unresolved_accuracy,
            "mean_absolute_correlation": float(np.mean([
                float(row["mean_absolute_correlation"]) for row in values
            ])),
            "worst_absolute_correlation": float(np.min([
                float(row["worst_absolute_correlation"]) for row in values
            ])),
            "mean_absolute_crosstalk": float(np.mean([
                float(row["mean_absolute_crosstalk"]) for row in values
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
        "status": "finalists_selected" if selected else "no_method_passed",
        "expected_fixture_count": int(expected_fixture_count),
        "minimum_converged_fraction": float(minimum_converged_fraction),
        "require_unresolved_accuracy": bool(require_unresolved_accuracy),
        "configuration_summaries": summaries,
        "selected_finalists": selected,
    }
