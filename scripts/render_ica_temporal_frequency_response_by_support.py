#!/usr/bin/env python3
"""Render paired learned temporal frequency responses for 3- and 5-frame support.

The factorial screen retained response summaries rather than every demixing
matrix.  This script therefore refits the protected temporal finalist while
varying only temporal support, using the same protected folds and seed set.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurobench.experiments.ica_whitening_evaluation.config import ICAWhiteningConfig
from neurobench.experiments.ica_whitening_evaluation.model import (
    PatchObservations,
    activity_priority_order,
    extract_patch_observations,
    fit_ica,
)
from neurobench.experiments.ica_whitening_evaluation.operators import apply_whitening
from neurobench.experiments.ica_whitening_evaluation.real_config import RealDataConfig
from neurobench.experiments.ica_whitening_evaluation.real_finalist_confirmation import (
    _completed_factorial_rows,
    _held_out_fit_filter,
    confirmation_seed_set,
)
from neurobench.experiments.ica_whitening_evaluation.real_runner import (
    _observations_at_coordinates,
    _signed_review,
)


LINE_SUPPORTS = (2, 3, 5, 8, 12, 16, 20)


def _magnitude_response(kernel: np.ndarray, frame_period_ms: float) -> tuple[np.ndarray, np.ndarray]:
    n_fft = 2048
    response = np.abs(np.fft.rfft(np.asarray(kernel, dtype=np.float64), n=n_fft))
    response /= max(float(np.max(response)), np.finfo(float).eps)
    frequency = np.fft.rfftfreq(n_fft, d=frame_period_ms / 1000.0)
    return frequency, response


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--fit-id", default="icaw_bf39743bcf6a7d4f")
    parser.add_argument("--supports", type=int, nargs="+", default=list(range(2, 21)))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    real = RealDataConfig.from_json(args.config)
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    source_rows = _completed_factorial_rows(real)
    if args.fit_id not in source_rows:
        raise RuntimeError(f"unknown factorial fit: {args.fit_id}")
    base = source_rows[args.fit_id]
    if base["family"] != "temporal":
        raise RuntimeError("paired support plot currently requires a temporal finalist")

    interval_payload = json.loads(
        (Path(args.preflight) / "event_interval_contract.json").read_text(encoding="utf-8")
    )["intervals"]
    intervals = {int(key): (int(value[0]), int(value[1]))
                 for key, value in interval_payload.items()}
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    activity_order = activity_priority_order(movie, quiet)
    seeds = confirmation_seed_set(int(base["seed"]))

    curves: dict[tuple[int, str], list[np.ndarray]] = {}
    curve_rows: list[dict[str, object]] = []
    for support in sorted(set(args.supports)):
        if base["causality"] != "causal":
            raise ValueError(
                "2-to-20 support sweep requires causal semantics; centered even supports "
                "need an explicit half-sample alignment convention"
            )
        if support < 2:
            raise ValueError("causal temporal supports must be at least 2")
        for held_out_burst in sorted(intervals):
            for seed in seeds:
                specification = {
                    **base,
                    "fit_id": f"{args.fit_id}__paired_support_{support}",
                    "temporal_width_frames": support,
                    "seed": seed,
                }
                raw_fit = extract_patch_observations(
                    movie,
                    family="temporal",
                    spatial_width=None,
                    temporal_width=support,
                    causality=specification["causality"],
                    maximum_samples=real.fitting.maximum_fit_samples,
                    seed=seed,
                    quiet_frames=quiet,
                    activity_fraction=real.fitting.activity_fraction,
                    activity_order=activity_order,
                )
                raw_fit = _held_out_fit_filter(raw_fit, intervals[held_out_burst])
                whitening = apply_whitening(
                    movie, quiet, specification,
                    maximum_samples=real.fitting.maximum_fit_samples,
                )
                fit_values = _observations_at_coordinates(
                    whitening.output, raw_fit.times, raw_fit.rows, raw_fit.columns,
                    specification,
                )
                observations = PatchObservations(
                    fit_values, raw_fit.times, raw_fit.rows, raw_fit.columns,
                )
                model = fit_ica(
                    observations.values, specification,
                    maximum_fit_samples=real.fitting.maximum_fit_samples,
                )
                # A permutation-invariant role: the component with larger
                # normalized DC response is "DC-dominant"; the other is
                # "difference-dominant".  This avoids component-index claims.
                prepared = []
                for kernel in model.demixing:
                    frequency, magnitude = _magnitude_response(kernel, parent.frames.frame_period_ms)
                    prepared.append((float(magnitude[0]), frequency, magnitude))
                prepared.sort(key=lambda item: item[0], reverse=True)
                for role, (_, frequency, magnitude) in zip(
                    ("DC-dominant", "difference-dominant"), prepared, strict=True
                ):
                    curves.setdefault((support, role), []).append(magnitude)
                    for hz, value in zip(frequency, magnitude, strict=True):
                        curve_rows.append({
                            "source_fit_id": args.fit_id,
                            "temporal_support_frames": support,
                            "held_out_burst": held_out_burst,
                            "seed": seed,
                            "component_role": role,
                            "frequency_hz": float(hz),
                            "normalized_magnitude": float(value),
                            "converged": bool(model.converged),
                        })

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "temporal_frequency_response_by_support_curves.csv", curve_rows)

    roles = ("DC-dominant", "difference-dominant")
    supports = sorted(set(args.supports))
    selected_frequency = frequency <= 25
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True,
                             constrained_layout=True)
    for column, role in enumerate(roles):
        medians = []
        interval_widths = []
        for support in supports:
            values = np.asarray(curves[(support, role)])[:, selected_frequency]
            medians.append(np.median(values, axis=0))
            lower, upper = np.quantile(values, [.1, .9], axis=0)
            interval_widths.append(upper - lower)
        extent = [float(frequency[selected_frequency][0]),
                  float(frequency[selected_frequency][-1]),
                  supports[0] - .5, supports[-1] + .5]
        magnitude_image = axes[0, column].imshow(
            np.asarray(medians), origin="lower", aspect="auto", extent=extent,
            cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
        )
        uncertainty_image = axes[1, column].imshow(
            np.asarray(interval_widths), origin="lower", aspect="auto", extent=extent,
            cmap="Oranges", vmin=0, vmax=1, interpolation="nearest",
        )
        axes[0, column].set_title(role)
        axes[1, column].set_xlabel("Frequency (Hz)")
        for axis in axes[:, column]:
            axis.set_yticks(supports)
            axis.set_xlim(0, 25)
    axes[0, 0].set_ylabel("Tap count\nMedian normalized |H(f)|")
    axes[1, 0].set_ylabel("Tap count\n90th minus 10th percentile")
    fig.colorbar(magnitude_image, ax=axes[0, :], label="Median normalized magnitude",
                 shrink=.9, pad=.015)
    fig.colorbar(uncertainty_image, ax=axes[1, :], label="10th–90th percentile width",
                 shrink=.9, pad=.015)
    fig.suptitle("Learned causal temporal ICA response across 2–20 taps", fontsize=16)
    fig.text(
        .5, -.012,
        "Protected-finalist settings held fixed; each cell summarizes 4 held-out folds × 3 seeds. "
        "Components ordered within each refit by normalized DC response.",
        ha="center", fontsize=9, color="#3E4650",
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"temporal_frequency_response_by_support.{suffix}",
                    dpi=180 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    available_lines = [support for support in LINE_SUPPORTS if support in supports]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharex=True, sharey=True,
                             constrained_layout=True)
    palette = plt.get_cmap("Blues")
    for axis, role in zip(axes, roles, strict=True):
        for index, support in enumerate(available_lines):
            values = np.asarray(curves[(support, role)])
            median = np.median(values, axis=0)
            color = palette(.35 + .6 * index / max(1, len(available_lines) - 1))
            axis.plot(frequency, median, color=color, linewidth=1.8,
                      label=f"{support} taps")
        axis.set_title(role)
        axis.set_xlabel("Frequency (Hz)")
        axis.grid(color="#D9DEE5", linewidth=.7, alpha=.8)
        axis.set_xlim(0, 25)
        axis.set_ylim(0, 1.04)
    axes[0].set_ylabel("Median normalized |H(f)|")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle("Selected learned causal temporal ICA responses", fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"temporal_frequency_response_selected_supports.{suffix}",
                    dpi=180 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "source_fit_id": args.fit_id,
        "supports_frames": sorted(set(args.supports)),
        "held_out_bursts": sorted(intervals),
        "seeds": list(seeds),
        "curves_per_support_and_role": len(intervals) * len(seeds),
        "normalization": "each component divided by its own maximum magnitude",
        "aggregation": "median with 10th_to_90th_percentile band",
        "primary_figure": "median response heatmap plus 10th_to_90th_percentile width heatmap",
        "selected_line_supports": available_lines,
        "component_role": "within-refit ordering by normalized DC magnitude",
        "vary_nothing_else": True,
        "claim_scope": "paired_temporal_support_sensitivity_of_protected_finalist",
    }
    (output / "temporal_frequency_response_by_support_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
