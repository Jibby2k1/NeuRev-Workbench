"""Deterministic aggregate artifacts for event-balanced CS-Parzen sweeps."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCIENTIFIC_STATUS = (
    "diagnostic_event_weighting_study_not_validated_source_separation"
)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_fit_tables(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_json(root / "fit_metrics.json", list(rows))
    fields = sorted({key for row in rows for key in row if key != "per_event_mass"})
    temporary = root / "fit_metrics.csv.partial"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, "per_event_mass_json"])
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat["per_event_mass_json"] = json.dumps(
                row.get("per_event_mass", {}), sort_keys=True
            )
            writer.writerow(flat)
    temporary.replace(root / "fit_metrics.csv")


def evaluate_gate_c(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row
        for row in rows
        if row["weight_mode"] == "roi_balanced"
        and row["whitening_mode"] == "natural_fixed"
    ]
    baseline = {row["fold_id"]: row for row in primary if row["alpha"] == 0}
    candidates: list[dict[str, Any]] = []
    for alpha in sorted(
        {float(row["alpha"]) for row in primary if 0 < row["alpha"] <= 0.2}
    ):
        alpha_rows = [row for row in primary if row["alpha"] == alpha]
        shifts = np.asarray(
            [row["angle_shift_from_alpha0_degrees"] for row in alpha_rows],
            dtype=float,
        )
        median_shift = float(np.median(np.abs(shifts))) if len(shifts) else 0.0
        consistent_folds = (
            max(int(np.sum(shifts > 0)), int(np.sum(shifts < 0)))
            if len(shifts)
            else 0
        )
        baseline_rows = [
            baseline[row["fold_id"]]
            for row in alpha_rows
            if row["fold_id"] in baseline
        ]
        recall_ok = bool(
            baseline_rows
            and np.median([row["known_label_recall"] for row in alpha_rows])
            >= np.median([row["known_label_recall"] for row in baseline_rows]) - 0.05
        )
        candidate_limit = (
            max(
                np.median([row["candidate_count"] for row in baseline_rows]) * 1.5,
                np.median([row["candidate_count"] for row in baseline_rows]) + 10,
            )
            if baseline_rows
            else -1
        )
        candidates_ok = bool(
            baseline_rows
            and np.median([row["candidate_count"] for row in alpha_rows])
            <= candidate_limit
        )
        ess_ok = bool(
            alpha_rows
            and min(row["weight_ess_fraction"] for row in alpha_rows) >= 0.2
        )
        seeds = {
            int(row.get("sample_seed", -1))
            for row in alpha_rows
            if row.get("sample_seed") is not None
        }
        criteria = {
            "median_shift_at_least_one_degree": median_shift >= 1.0,
            "direction_consistent_three_of_four_folds": (
                len(alpha_rows) >= 4 and consistent_folds >= 3
            ),
            "not_exclusive_to_frame_weighting": bool(alpha_rows),
            "heldout_recall_not_materially_degraded": recall_ok,
            "heldout_candidates_not_materially_degraded": candidates_ok,
            "weight_ess_fraction_at_least_0p20": ess_ok,
            "independent_sample_seed_survived": len(seeds) >= 2,
        }
        candidates.append(
            {
                "alpha": alpha,
                "fold_count": len(alpha_rows),
                "median_absolute_shift_degrees": median_shift,
                "consistent_direction_folds": consistent_folds,
                "sample_seeds": sorted(seeds),
                "criteria": criteria,
                "passes_all": all(criteria.values()),
            }
        )
    passed_row = next((row for row in candidates if row["passes_all"]), None)
    reasons = []
    if passed_row is None:
        reasons.append(
            "No moderate ROI-balanced alpha satisfied every preregistered "
            "angle, fold, held-out, ESS, and independent-seed criterion."
        )
    if not any(
        len(row.get("sample_seeds", [])) >= 2 for row in candidates
    ):
        reasons.append(
            "The standard profile has one sample seed; an independent-seed "
            "confirmation is required before spatial extension eligibility."
        )
    return {
        "gate": "C",
        "passed": passed_row is not None,
        "eligible_alpha": None if passed_row is None else passed_row["alpha"],
        "engineering_thresholds": {
            "minimum_median_shift_degrees": 1.0,
            "minimum_consistent_folds": 3,
            "maximum_moderate_alpha": 0.2,
            "minimum_weight_ess_fraction": 0.2,
            "maximum_recall_drop": 0.05,
            "candidate_burden_multiplier": 1.5,
            "candidate_burden_additive": 10,
            "minimum_sample_seeds": 2,
        },
        "alpha_diagnostics": candidates,
        "reasons": reasons,
        "spatial_extension_authorized": False,
    }


def _save(figure, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    figure.tight_layout()
    figure.savefig(temporary, format="png", dpi=130)
    import matplotlib.pyplot as plt

    plt.close(figure)
    temporary.replace(destination)


def write_figures(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    records = list(rows)
    colors = {
        "natural": "black",
        "frame_balanced": "tab:blue",
        "roi_balanced": "tab:orange",
        "roi_balanced_weighted_whitening": "tab:green",
    }

    def lines(
        filename: str,
        y_key: str,
        ylabel: str,
        title: str,
        *,
        include_holdout: bool = False,
    ) -> None:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for lane in sorted({row["lane"] for row in records}):
            lane_rows = [row for row in records if row["lane"] == lane]
            for fold in sorted({row["fold_id"] for row in lane_rows}):
                fold_rows = sorted(
                    [row for row in lane_rows if row["fold_id"] == fold],
                    key=lambda row: row["alpha"],
                )
                axis.plot(
                    [row["alpha"] for row in fold_rows],
                    [row.get(y_key, np.nan) for row in fold_rows],
                    marker=".",
                    alpha=0.55,
                    color=colors.get(lane),
                    label=lane if fold == min(row["fold_id"] for row in lane_rows) else None,
                )
                if include_holdout:
                    axis.plot(
                        [row["alpha"] for row in fold_rows],
                        [row.get("objective_natural_holdout", np.nan) for row in fold_rows],
                        linestyle="--",
                        alpha=0.45,
                        color=colors.get(lane),
                    )
        axis.set(xlabel="Declared event mass alpha", ylabel=ylabel, title=title)
        if records:
            axis.legend(fontsize=7)
        _save(figure, destination / filename)

    lines(
        "angle_shift_vs_alpha.png",
        "angle_shift_from_alpha0_degrees",
        "Canonical angle shift (degrees)",
        "Event weighting angle path (four folds; descriptive)",
    )
    lines(
        "derivative_cosine_vs_alpha.png",
        "cosine_to_derivative",
        "Cosine to fixed derivative",
        "Innovation direction remains label-free",
    )
    lines(
        "train_and_holdout_objective_vs_alpha.png",
        "objective_weighted_train",
        "CS-Parzen objective",
        "Weighted train (solid) and natural holdout (dashed)",
        include_holdout=True,
    )
    lines(
        "weight_ess_vs_alpha.png",
        "weight_ess_fraction",
        "Weight ESS / unique samples",
        "Mixture-weight concentration",
    )

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for row in records:
        recall = row.get("known_label_recall")
        candidates = row.get("candidate_count")
        if recall is not None and candidates is not None:
            axis.scatter(
                candidates,
                recall,
                color=colors.get(row["lane"]),
                alpha=0.65,
            )
    axis.set(
        xlabel="Held-out candidates (not precision)",
        ylabel="Known-label recall",
        title="Natural-prevalence sparse-positive evaluation",
    )
    _save(figure, destination / "known_label_recall_vs_candidate_count.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for fold in sorted({row["fold_id"] for row in records}):
        fold_rows = [
            row
            for row in records
            if row["fold_id"] == fold
            and row["lane"] == "roi_balanced"
            and row["whitening_mode"] == "natural_fixed"
        ]
        axis.plot(
            [row["alpha"] for row in fold_rows],
            [row["angle_degrees"] for row in fold_rows],
            marker=".",
            label=f"fold {fold}",
        )
    axis.set(
        xlabel="Alpha",
        ylabel="Canonical innovation angle (degrees)",
        title="Fold angle stability (n=4; no confidence claim)",
    )
    if records:
        axis.legend(fontsize=7)
    _save(figure, destination / "fold_angle_stability.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for lane in ("frame_balanced", "roi_balanced"):
        lane_rows = [
            row
            for row in records
            if row["lane"] == lane and row["whitening_mode"] == "natural_fixed"
        ]
        for alpha in sorted({row["alpha"] for row in lane_rows}):
            values = [
                row["angle_shift_from_alpha0_degrees"]
                for row in lane_rows
                if row["alpha"] == alpha
            ]
            if values:
                axis.scatter(
                    [alpha] * len(values),
                    values,
                    color=colors[lane],
                    alpha=0.55,
                )
                axis.plot(
                    alpha,
                    float(np.median(values)),
                    marker="_",
                    markersize=14,
                    color=colors[lane],
                    label=lane if alpha == min(row["alpha"] for row in lane_rows) else None,
                )
    axis.set(
        xlabel="Alpha",
        ylabel="Angle shift (degrees)",
        title="Frame versus ROI event mass",
    )
    if records:
        axis.legend(fontsize=7)
    _save(figure, destination / "frame_vs_roi_weighting_comparison.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.axhline(0, color="0.7", linewidth=1)
    for row in records:
        if row["lane"] in {"frame_balanced", "roi_balanced"}:
            axis.scatter(
                row["alpha"],
                row["correlation_to_fixed_derivative"],
                color=colors[row["lane"]],
                alpha=0.55,
            )
    axis.set(
        xlabel="Alpha",
        ylabel="Natural-sample correlation to derivative",
        title="Representative weighted outputs at declared evaluation support",
    )
    _save(figure, destination / "representative_weighted_outputs.png")


def write_results_note(
    root: Path,
    rows: Sequence[dict[str, Any]],
    gate: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    measured = (
        f"{len(rows)} fits completed. Alpha=0 parity status: "
        f"{baseline.get('status', 'unavailable')}. Gate C passed: {gate['passed']}."
    )
    note = [
        "# Event-balanced CS-Parzen ICA results",
        "",
        f"Scientific status: {SCIENTIFIC_STATUS}.",
        "",
        "## Measured facts",
        "",
        measured,
        "",
        "## Interpretation",
        "",
        "Angle movement is a diagnostic response to declared training mass. Frame-only movement is compatible with global event-time structure and is not evidence of neuronal separation.",
        "",
        "## Unsupported claims",
        "",
        "Sparse labels do not identify precision, unmatched candidates are unknown, and this study does not validate physical neural/background source separation.",
        "",
        "## Stage gate",
        "",
        f"Gate C: {'passed' if gate['passed'] else 'not met'}. No spatial ICA was launched or authorized.",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(note), encoding="utf-8")
