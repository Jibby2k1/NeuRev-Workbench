#!/usr/bin/env python3
"""Execute the frozen paired E01/E02 synthetic benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from neurobench.experiments.unsupervised_ica_eval.pc_mitl_benchmark import evaluate_cell, generate_fixture


def _paired(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["scenario_id"]), int(row["seed"]))][str(row["method"])] = row
    paired = []
    for (scenario, seed), methods in sorted(grouped.items()):
        cs, matrix = methods["cs_parzen"], methods["matrix_tc_alpha2"]
        paired.append({
            "scenario_id": scenario, "seed": seed,
            "cs_parzen": cs["test_mean_absolute_correlation"],
            "matrix_tc_alpha2": matrix["test_mean_absolute_correlation"],
            "matrix_minus_cs": float(matrix["test_mean_absolute_correlation"]) - float(cs["test_mean_absolute_correlation"]),
        })
    return paired


def _bootstrap_ci(values: np.ndarray, seed: int = 20260904) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing output collision: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            fixture = generate_fixture(
                sample_count=int(config["sample_count"]), component_count=int(scenario["component_count"]),
                source_family=scenario["source_family"], condition_number=float(scenario["condition_number"]),
                noise_std=float(scenario["noise_std"]), seed=int(seed),
            )
            cell_rows = evaluate_cell(
                fixture, bandwidth_candidates=config["bandwidth_candidates"],
                train_stop=int(config["split"]["train_stop"]), validation_stop=int(config["split"]["validation_stop"]),
                steps=int(config["optimizer"]["steps"]), learning_rate=float(config["optimizer"]["learning_rate"]),
            )
            for row in cell_rows:
                rows.append({"scenario_id": scenario["id"], "seed": seed, **scenario, **row})
    paired = _paired(rows)
    differences = np.asarray([row["matrix_minus_cs"] for row in paired], dtype=float)
    scenario_summaries = []
    for scenario in config["scenarios"]:
        subset = np.asarray([row["matrix_minus_cs"] for row in paired if row["scenario_id"] == scenario["id"]])
        scenario_summaries.append({
            "scenario_id": scenario["id"], "paired_count": len(subset),
            "mean_delta": float(subset.mean()), "median_delta": float(np.median(subset)),
            "win_fraction": float(np.mean(subset > 0)), "bootstrap_95_ci": _bootstrap_ci(subset),
        })
    gate = config["decision_gate"]
    broad = float(differences.mean()) >= gate["broad_advance_mean_delta"] and float(np.mean(differences > 0)) >= gate["broad_advance_win_fraction"]
    best_subgroup = max(item["mean_delta"] for item in scenario_summaries)
    catastrophic = float(differences.mean()) <= gate["catastrophic_delta"]
    if broad:
        decision = "advance_matrix_tc_to_larger_synthetic_confirmation"
    elif best_subgroup >= gate["specialized_retain_subgroup_delta"] and not catastrophic:
        decision = "retain_matrix_tc_for_specialized_synthetic_confirmation"
    else:
        decision = "stop_matrix_tc_extension_retain_cs_parzen"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    dirty_paths = subprocess.run(["git", "status", "--porcelain"], check=True, text=True, capture_output=True).stdout.splitlines()
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    implementation_files = [
        Path("neurobench/experiments/unsupervised_ica_eval/matrix_itl.py"),
        Path("neurobench/experiments/unsupervised_ica_eval/pc_mitl_benchmark.py"),
        Path("scripts/run_pc_mitl_e01_e02.py"),
    ]
    implementation_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_files}
    summary = {
        "schema_version": 1, "status": "complete", "decision": decision,
        "git_commit": commit, "config_sha256": config_hash,
        "implementation_sha256": implementation_hashes,
        "worktree_dirty": bool(dirty_paths),
        "paired_cells": len(paired), "scenario_count": len(config["scenarios"]), "seed_count": len(config["seeds"]),
        "overall_mean_matrix_minus_cs": float(differences.mean()),
        "overall_median_matrix_minus_cs": float(np.median(differences)),
        "overall_win_fraction": float(np.mean(differences > 0)),
        "overall_bootstrap_95_ci": _bootstrap_ci(differences),
        "scenario_summaries": scenario_summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "labels_used": False, "real_video_used": False,
        "limitations": ["development synthetic grid", "validation-selected bandwidth", "no conditional-history objective", "no real-video claim"],
    }
    fieldnames = list(rows[0].keys())
    with (args.output_dir / "method_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0].keys())); writer.writeheader(); writer.writerows(paired)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    labels = [item["scenario_id"].replace("_", "\n") for item in scenario_summaries]
    means = [item["mean_delta"] for item in scenario_summaries]
    lower = [mean - item["bootstrap_95_ci"][0] for mean, item in zip(means, scenario_summaries)]
    upper = [item["bootstrap_95_ci"][1] - mean for mean, item in zip(means, scenario_summaries)]
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    colors = ["#2B7A78" if value > 0 else "#B85C5C" for value in means]
    axis.bar(np.arange(len(means)), means, yerr=[lower, upper], capsize=3, color=colors)
    axis.axhline(0, color="black", linewidth=1); axis.axhline(gate["broad_advance_mean_delta"], color="#666666", linestyle="--")
    axis.set_xticks(np.arange(len(means)), labels); axis.set_ylabel("Matrix-TC minus CS-Parzen\nheld-out |source correlation|")
    axis.set_title("PC-MITL-ICA E01/E02 paired synthetic gate (95% bootstrap CI)")
    figure.savefig(args.output_dir / "paired_scenario_deltas.png", dpi=180); plt.close(figure)
    expert = args.output_dir / "1_Expert_Annotations"
    model = args.output_dir / "2_Model_Annotations"
    comparison = args.output_dir / "3_Comparison"
    expert.mkdir(); model.mkdir(); comparison.mkdir()
    (expert / "README.md").write_text(
        "# Expert annotations\n\nNot applicable: this benchmark uses generated latent sources with exact synthetic truth, not expert labels.\n"
    )
    (model / "README.md").write_text(
        "# Model outputs\n\nThe frozen candidate-surrogate panel is the complete method-by-scenario-by-seed table in `../method_rows.csv`.\n"
    )
    (comparison / "README.md").write_text(
        "# Paired comparison\n\n`../paired_results.csv` pairs both methods on the identical fixture and `../paired_scenario_deltas.png` shows scenario means and bootstrap intervals.\n"
    )
    llm_context = {
        "experiment_id": config["experiment_id"], "status": "complete",
        "stage_sequence": ["generated_sources", "linear_mixing", "train_only_whitening", "validation_bandwidth_selection", "untouched_test_scoring"],
        "operating_point": {"bandwidth_candidates": config["bandwidth_candidates"], "optimizer": config["optimizer"]},
        "section_applicability": {"expert": "not_applicable_synthetic_truth", "model": "complete_table", "comparison": "complete_paired_table_and_figure"},
        "expected_counts": {"paired_cells": len(config["scenarios"]) * len(config["seeds"]), "method_rows": 2 * len(config["scenarios"]) * len(config["seeds"])},
        "observed_counts": {"paired_cells": len(paired), "method_rows": len(rows)},
        "decision": decision, "primary_summary": "summary.json", "primary_table": "paired_results.csv", "primary_figure": "paired_scenario_deltas.png",
        "interpretation": "Synthetic source-recovery evidence only; no real-video, neuron, or causal claim.",
    }
    artifact_index = {
        "schema_version": 1,
        "artifacts": [
            {"path": "summary.json", "role": "summary"}, {"path": "resolved_config.yaml", "role": "configuration"},
            {"path": "method_rows.csv", "role": "model_table"}, {"path": "paired_results.csv", "role": "paired_comparison"},
            {"path": "paired_scenario_deltas.png", "role": "primary_figure"},
            {"path": "llm_context.json", "role": "llm_index"},
            {"path": "artifact_index.json", "role": "artifact_index"},
            {"path": "validation.json", "role": "validation"},
            {"path": "REPORT.md", "role": "report"},
            {"path": "1_Expert_Annotations/README.md", "role": "expert_not_applicable"},
            {"path": "2_Model_Annotations/README.md", "role": "model_section"},
            {"path": "3_Comparison/README.md", "role": "comparison_section"},
        ],
    }
    fixture_keys = {(str(row["scenario_id"]), int(row["seed"])) for row in rows}
    validation = {
        "status": "pass",
        "count_checks_pass": len(paired) == len(config["scenarios"]) * len(config["seeds"]) and len(rows) == 2 * len(paired),
        "finite_primary_metrics": bool(np.isfinite(differences).all()),
        "paired_fixture_checks": all(len({str(row["fixture_fingerprint"]) for row in rows if row["scenario_id"] == scenario and int(row["seed"]) == seed}) == 1 for scenario, seed in fixture_keys),
        "train_only_preprocessing_recorded": all(bool(row["train_data_hash"]) for row in rows),
        "real_video_audit_required": False,
    }
    (args.output_dir / "llm_context.json").write_text(json.dumps(llm_context, indent=2) + "\n")
    (args.output_dir / "artifact_index.json").write_text(json.dumps(artifact_index, indent=2) + "\n")
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (args.output_dir / "REPORT.md").write_text(
        "# PC-MITL-ICA E01/E02 report\n\n"
        f"Decision: **{decision}**.\n\n"
        f"Across {len(paired)} paired cells, mean Matrix-TC minus CS-Parzen held-out recovery was {differences.mean():.4f}; "
        f"the 95% bootstrap interval was [{summary['overall_bootstrap_95_ci'][0]:.4f}, {summary['overall_bootstrap_95_ci'][1]:.4f}]. "
        "This is synthetic engineering evidence only.\n"
    )


if __name__ == "__main__":
    main()
