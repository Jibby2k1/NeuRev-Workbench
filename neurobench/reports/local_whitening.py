"""Compact machine-readable and visual reports for local whitening stages."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from neurobench.experiments.msln_msica.artifacts import atomic_json


CURRENT_TRUTH = [
    "Current causal joint MSLN is a scalar studentized residual using a prior-frame spatial annulus, protected spatial core, temporal guard, and quiet-fitted scale floor.",
    "Maintained MSLN does not construct a multivariate local vector or force multivariate covariance to identity.",
    "The v2 joint_s15_g3_t31_g1 persistence lane retained 58/79 sparse known positives at 58 candidates per burst; the 49/79 Raw Direct anchor used a non-identical protocol.",
    "Broad ICA directions were bootstrap-unstable and do not establish biological source identity.",
    "Raw -> MSICA -> MSLN did not improve the broad v3 control, and the cascade stopped at its stability gate.",
    "The v4 five-seed energy ensemble was more stable than individual directions, but its 52/79 label-free result remains provisional.",
    "Sparse annotations define known positives only; unmatched candidates are unknown.",
    "This recording is development data, not an independent confirmation.",
    "Computational completion is not scientific success.",
]


METRIC_DEFINITIONS = {
    "covariance_identity_error": "Frobenius norm of C-I divided by sqrt(feature count).",
    "normalized_off_diagonal_energy": "Off-diagonal Frobenius norm divided by total covariance Frobenius norm.",
    "maximum_absolute_correlation": "Largest absolute off-diagonal covariance-normalized correlation.",
    "effective_rank": "Exponential Shannon entropy of normalized nonnegative eigenvalues.",
    "fit_holdout_covariance_error": "Identity error after whitening held-out covariance with a fit covariance.",
    "tile_boundary_discontinuity": "Mean transformed-value jump at internal tile starts relative to all adjacent-pixel jumps.",
}


def atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_stage_indices(
    root: Path,
    *,
    stage: str,
    status: str,
    summary: dict[str, Any],
    artifacts: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    """Write the small stable indices before any optional large media."""
    atomic_json(root / "summary.json", {"stage": stage, "status": status, **summary})
    atomic_json(root / "llm_context.json", {
        "experiment": "spon_ca_burst_local_whitening_v1",
        "stage": stage, "status": status,
        "scientific_sequence": ["signed_msln_feature_bank", "zca_whitened_features", "mahalanobis_energy", "empirical_quiet_surprise"],
        "metric_definitions": METRIC_DEFINITIONS,
        "current_repository_truth": CURRENT_TRUTH,
        "limitations": ["whiteness does not imply independence", "whitened coordinates are not biological sources", "unmatched candidates remain unknown"],
    })
    atomic_json(root / "artifact_index.json", {"stage": stage, "artifacts": artifacts})
    atomic_json(root / "validation.json", validation)


def render_synthetic_comparison(
    path: Path,
    carrier: np.ndarray,
    lane_maps: dict[str, np.ndarray],
) -> None:
    """Render a fixed-scale carrier and lane-energy comparison."""
    import matplotlib.pyplot as plt

    names = list(lane_maps)
    maximum = max(float(np.max(item)) for item in lane_maps.values())
    figure, axes = plt.subplots(1, len(names) + 1, figsize=(3 * (len(names) + 1), 3), constrained_layout=True)
    axes[0].imshow(carrier, cmap="gray"); axes[0].set_title("Signed carrier")
    for axis, name in zip(axes[1:], names):
        axis.imshow(lane_maps[name], cmap="gray", vmin=0, vmax=maximum)
        axis.set_title(name)
    for axis in axes: axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    figure.savefig(temporary, dpi=120); plt.close(figure); temporary.replace(path)


def _save_figure_atomic(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    figure.savefig(temporary, dpi=140)
    temporary.replace(path)


def render_covariance_audit_diagnostics(
    output_dir: Path,
    rows: list[dict[str, Any]],
    lane_fits: dict[str, Any],
    tail_samples: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, str]]:
    """Render fixed-scale covariance, conditioning, factorial, and null-tail evidence."""
    import matplotlib.pyplot as plt

    artifacts: list[dict[str, str]] = []
    modes = list(lane_fits)
    full_fit = lane_fits["global_full_zca"].global_full_fit
    matrices = [
        ("Sample covariance", full_fit.sample_covariance),
        ("Shrinkage covariance", full_fit.covariance),
        ("Post-ZCA covariance", full_fit.whitening @ full_fit.covariance @ full_fit.whitening.T),
    ]
    scale = max(float(np.max(np.abs(matrix))) for _, matrix in matrices)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for axis, (title, matrix) in zip(axes, matrices):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-scale, vmax=scale)
        axis.set_title(title); axis.set_xlabel("feature"); axis.set_ylabel("feature")
    figure.colorbar(image, ax=axes, shrink=.8)
    path = output_dir / "covariance_correlation_matrices.png"; _save_figure_atomic(figure, path); plt.close(figure)
    artifacts.append({"id": "covariance_correlation_matrices", "path": f"covariance_audit/{path.name}"})

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for mode in modes:
        fit = lane_fits[mode].global_full_fit
        axes[0].plot(np.arange(len(fit.eigenvalues_raw)), fit.eigenvalues_raw, marker="o", label=mode)
        axes[1].scatter(fit.shrinkage, fit.condition_number, label=mode)
    axes[0].set_yscale("log"); axes[0].set_title("Global covariance eigenvalues"); axes[0].set_xlabel("ordered eigenvalue")
    axes[1].set_yscale("log"); axes[1].set_title("Shrinkage and conditioning"); axes[1].set_xlabel("shrinkage"); axes[1].set_ylabel("condition number")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    path = output_dir / "eigenvalue_shrinkage_conditioning.png"; _save_figure_atomic(figure, path); plt.close(figure)
    artifacts.append({"id": "eigenvalue_shrinkage_conditioning", "path": f"covariance_audit/{path.name}"})

    local = lane_fits["local_full_zca"].primary_fits
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    values = [
        ("condition number", [fit.condition_number for fit in local], "viridis"),
        ("effective rank", [fit.effective_rank for fit in local], "magma"),
        ("unresolved", [not fit.resolved for fit in local], "gray_r"),
    ]
    centers_x = [(fit.tile_bounds_yx[2] + fit.tile_bounds_yx[3]) / 2 for fit in local]
    centers_y = [(fit.tile_bounds_yx[0] + fit.tile_bounds_yx[1]) / 2 for fit in local]
    for axis, (title, color_values, cmap) in zip(axes, values):
        scatter = axis.scatter(centers_x, centers_y, c=color_values, cmap=cmap, marker="s", s=24)
        axis.invert_yaxis(); axis.set_aspect("equal"); axis.set_title(title); figure.colorbar(scatter, ax=axis, shrink=.7)
    path = output_dir / "local_fit_spatial_diagnostics.png"; _save_figure_atomic(figure, path); plt.close(figure)
    artifacts.append({"id": "local_fit_spatial_diagnostics", "path": f"covariance_audit/{path.name}"})

    blocks = sorted({str(row["quiet_block"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    width = .8 / len(modes); positions = np.arange(len(blocks))
    for index, mode in enumerate(modes):
        lane_rows = [row for row in rows if row["lane"] == mode]
        axes[0].bar(positions + index * width, [row["heldout_identity_error"] for row in lane_rows], width, label=mode)
        axes[1].plot(blocks, [row["tile_boundary_ratio"] for row in lane_rows], marker="o", label=mode)
    axes[0].set_xticks(positions + width * (len(modes) - 1) / 2, blocks); axes[0].set_title("Held-out whiteness factorial"); axes[0].set_ylabel("identity error")
    axes[1].axhline(1, color="black", linewidth=.8); axes[1].set_title("Tile-boundary seam diagnostic"); axes[1].set_ylabel("boundary/all jump ratio")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    path = output_dir / "heldout_factorial_and_seams.png"; _save_figure_atomic(figure, path); plt.close(figure)
    artifacts.append({"id": "heldout_factorial_and_seams", "path": f"covariance_audit/{path.name}"})

    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for mode in modes:
        calibration = np.sort(tail_samples[mode]["calibration"])
        holdout = np.sort(tail_samples[mode]["holdout"])
        grid = np.unique(np.quantile(calibration, np.linspace(.5, .999, 120)))
        survival = np.asarray([np.mean(holdout >= threshold) for threshold in grid])
        axis.plot(grid, np.maximum(survival, 1 / (len(holdout) + 1)), label=mode)
    axis.set_yscale("log"); axis.set_xlabel("Mahalanobis energy"); axis.set_ylabel("held-out empirical survival"); axis.set_title("Quiet null-tail stability"); axis.legend(fontsize=7)
    path = output_dir / "empirical_null_tail_survival.png"; _save_figure_atomic(figure, path); plt.close(figure)
    artifacts.append({"id": "empirical_null_tail_survival", "path": f"covariance_audit/{path.name}"})
    return artifacts


def write_report(root: Path, stage: str, decision: str, findings: list[str]) -> None:
    lines = [
        "# Local covariance whitening V1", "", f"Stage: `{stage}`", f"Decision: `{decision}`", "",
        "## Current repository truth", "",
        *[f"- {item}" for item in CURRENT_TRUTH], "", "## Findings", "",
        *[f"- {item}" for item in findings], "", "## Interpretation boundary", "",
        "Whitened coordinates are covariance-calibrated statistical features, not independent biological sources or a cleaned movie. A completed generated-data stage does not authorize a Spon or GPU run.", "",
    ]
    temporary = root / "REPORT.md.partial"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(root / "REPORT.md")
