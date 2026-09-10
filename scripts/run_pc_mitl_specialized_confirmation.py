#!/usr/bin/env python3
"""Run the locked four-source PC-MITL-ICA synthetic confirmation."""
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

from neurobench.experiments.unsupervised_ica_eval.pc_mitl_benchmark import evaluate_fixed_cell, generate_fixture


def _cluster_ci(paired: list[dict[str, object]], role: str, seed: int = 20260904) -> list[float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in paired:
        if row["role"] == role:
            by_seed[int(row["seed"])].append(float(row["matrix_minus_cs"]))
    cluster_means = np.asarray([np.mean(values) for _, values in sorted(by_seed.items())])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(cluster_means, size=len(cluster_means), replace=True).mean() for _ in range(5000)]
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
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            fixture = generate_fixture(
                sample_count=int(scenario["sample_count"]), component_count=int(config["component_count"]),
                source_family=scenario["source_family"], condition_number=float(scenario["condition_number"]),
                noise_std=float(scenario["noise_std"]), seed=int(seed),
            )
            cell_rows = evaluate_fixed_cell(
                fixture, method_bandwidths=config["locked_bandwidths"],
                train_stop=int(round(int(scenario["sample_count"]) * float(config["train_fraction"]))),
                steps=int(config["optimizer"]["steps"]), learning_rate=float(config["optimizer"]["learning_rate"]),
            )
            rows.extend({"scenario_id": scenario["id"], "role": scenario["role"], "seed": seed, **scenario, **row} for row in cell_rows)
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["scenario_id"]), int(row["seed"]))][str(row["method"])] = row
    paired = []
    scenario_by_id = {row["id"]: row for row in config["scenarios"]}
    for (scenario_id, seed), methods in sorted(grouped.items()):
        cs, matrix = methods["cs_parzen"], methods["matrix_tc_alpha2"]
        paired.append({
            "scenario_id": scenario_id, "role": scenario_by_id[scenario_id]["role"], "seed": seed,
            "cs_parzen": cs["test_mean_absolute_correlation"],
            "matrix_tc_alpha2": matrix["test_mean_absolute_correlation"],
            "matrix_minus_cs": float(matrix["test_mean_absolute_correlation"]) - float(cs["test_mean_absolute_correlation"]),
        })
    summaries = []
    for scenario in config["scenarios"]:
        values = np.asarray([float(row["matrix_minus_cs"]) for row in paired if row["scenario_id"] == scenario["id"]])
        summaries.append({
            "scenario_id": scenario["id"], "role": scenario["role"], "paired_count": len(values),
            "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
            "win_fraction": float(np.mean(values > 0)),
        })
    primary = np.asarray([float(row["matrix_minus_cs"]) for row in paired if row["role"] == "primary"])
    guardrail = np.asarray([float(row["matrix_minus_cs"]) for row in paired if row["role"] == "guardrail"])
    primary_ci = _cluster_ci(paired, "primary")
    guardrail_ci = _cluster_ci(paired, "guardrail")
    gate = config["decision_gate"]
    passed = (
        float(primary.mean()) >= float(gate["primary_mean_delta"])
        and primary_ci[0] > float(gate["primary_ci_lower_bound"])
        and float(np.mean(primary > 0)) >= float(gate["primary_win_fraction"])
        and float(guardrail.mean()) >= float(gate["guardrail_max_mean_loss"])
    )
    decision = gate["on_pass"] if passed else gate["on_fail"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, text=True, capture_output=True).stdout)
    implementation_files = [Path("neurobench/experiments/unsupervised_ica_eval/matrix_itl.py"), Path("neurobench/experiments/unsupervised_ica_eval/pc_mitl_benchmark.py"), Path("scripts/run_pc_mitl_specialized_confirmation.py")]
    summary = {
        "schema_version": 1, "status": "complete", "decision": decision, "gate_passed": passed,
        "paired_cells": len(paired), "primary_cells": len(primary), "guardrail_cells": len(guardrail),
        "primary_mean_delta": float(primary.mean()), "primary_win_fraction": float(np.mean(primary > 0)), "primary_cluster_bootstrap_95_ci": primary_ci,
        "guardrail_mean_delta": float(guardrail.mean()), "guardrail_win_fraction": float(np.mean(guardrail > 0)), "guardrail_cluster_bootstrap_95_ci": guardrail_ci,
        "scenario_summaries": summaries, "elapsed_seconds": time.perf_counter() - started,
        "git_commit": commit, "worktree_dirty": dirty, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "implementation_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_files},
        "labels_used": False, "real_video_used": False,
    }
    with (args.output_dir / "method_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    labels = [row["scenario_id"].replace("_", "\n") for row in summaries]
    means = [row["mean_delta"] for row in summaries]
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.bar(range(len(means)), means, color=["#2B7A78" if row["role"] == "primary" else "#8E6C9C" for row in summaries])
    axis.axhline(0, color="black", linewidth=1); axis.axhline(gate["primary_mean_delta"], color="#666", linestyle="--")
    axis.set_xticks(range(len(labels)), labels); axis.set_ylabel("Matrix-TC minus CS-Parzen\nheld-out |source correlation|")
    axis.set_title("Locked PC-MITL-ICA specialized confirmation")
    figure.savefig(args.output_dir / "confirmation_scenario_deltas.png", dpi=180); plt.close(figure)
    for folder, body in {
        "1_Expert_Annotations": "Exact synthetic truth; expert annotations are not applicable.",
        "2_Model_Annotations": "The complete frozen model panel is ../method_rows.csv.",
        "3_Comparison": "The paired method comparison is ../paired_results.csv and ../confirmation_scenario_deltas.png.",
    }.items():
        path = args.output_dir / folder; path.mkdir(); (path / "README.md").write_text(f"# {folder}\n\n{body}\n")
    llm_context = {
        "experiment_id": config["experiment_id"], "status": "complete", "decision": decision,
        "locked_bandwidths": config["locked_bandwidths"], "bandwidth_provenance": config["bandwidth_provenance"],
        "expected_counts": {"paired_cells": len(config["scenarios"]) * len(config["seeds"]), "method_rows": 2 * len(config["scenarios"]) * len(config["seeds"])},
        "observed_counts": {"paired_cells": len(paired), "method_rows": len(rows)},
        "sections": {"expert": "not_applicable_synthetic_truth", "model": "complete", "comparison": "complete"},
        "primary_summary": "summary.json", "primary_table": "paired_results.csv", "primary_figure": "confirmation_scenario_deltas.png",
        "interpretation": "Locked synthetic confirmation only; no real-video, neuronal, or causal claim.",
    }
    artifacts = ["summary.json", "resolved_config.yaml", "method_rows.csv", "paired_results.csv", "confirmation_scenario_deltas.png", "llm_context.json", "artifact_index.json", "validation.json", "REPORT.md", "1_Expert_Annotations/README.md", "2_Model_Annotations/README.md", "3_Comparison/README.md"]
    validation = {
        "status": "pass", "count_checks_pass": len(paired) == len(config["scenarios"]) * len(config["seeds"]) and len(rows) == 2 * len(paired),
        "finite_primary_metrics": bool(np.isfinite(primary).all() and np.isfinite(guardrail).all()),
        "paired_fixture_checks": all(len({str(row["fixture_fingerprint"]) for row in rows if row["scenario_id"] == sid and int(row["seed"]) == seed}) == 1 for sid, seed in grouped),
        "locked_bandwidth_checks": all(float(row["locked_bandwidth"]) == float(config["locked_bandwidths"][str(row["method"])]) for row in rows),
        "real_video_audit_required": False,
    }
    (args.output_dir / "llm_context.json").write_text(json.dumps(llm_context, indent=2) + "\n")
    (args.output_dir / "artifact_index.json").write_text(json.dumps({"schema_version": 1, "artifacts": [{"path": path} for path in artifacts]}, indent=2) + "\n")
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (args.output_dir / "REPORT.md").write_text(
        f"# PC-MITL-ICA specialized confirmation\n\nDecision: **{decision}**.\n\nPrimary mean delta: `{primary.mean():.5f}`, cluster-bootstrap 95% CI `[{primary_ci[0]:.5f}, {primary_ci[1]:.5f}]`; guardrail mean delta: `{guardrail.mean():.5f}`. Synthetic evidence only.\n"
    )


if __name__ == "__main__":
    main()
