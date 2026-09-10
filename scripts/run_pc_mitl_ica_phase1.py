#!/usr/bin/env python3
"""Run the bounded synthetic PC-MITL-ICA phase-1 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from neurobench.algorithms.pairwise_separation import fit_cs_parzen_ica
from neurobench.experiments.unsupervised_ica_eval.matrix_itl import fit_matrix_tc_ica


def _fingerprint(config_path: Path, data_id: str, commit: str) -> str:
    payload = {"config": yaml.safe_load(config_path.read_text()), "data_id": data_id, "git_commit": commit}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _score(reference: np.ndarray, recovered: np.ndarray) -> tuple[float, list[float]]:
    correlations = np.corrcoef(reference.T, recovered.T)[: reference.shape[1], reference.shape[1] :]
    rows, columns = linear_sum_assignment(-np.abs(correlations))
    matched = [float(abs(correlations[row, column])) for row, column in zip(rows, columns)]
    return float(np.mean(matched)), matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing output collision: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["data"]["seed"])
    n = int(config["data"]["sample_count"])
    rng = np.random.default_rng(seed)
    sources = np.column_stack((rng.laplace(size=n), rng.uniform(-np.sqrt(3), np.sqrt(3), size=n)))
    mixing = np.asarray([[1.0, 0.65], [0.25, 1.15]])
    observed = sources @ mixing.T
    train_stop = int(config["data"]["split"]["train"][1])
    mean = observed[:train_stop].mean(axis=0)
    covariance = np.cov((observed[:train_stop] - mean).T, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    whitener = vectors @ np.diag(1 / np.sqrt(np.maximum(values, values.max() * 1e-6))) @ vectors.T
    whitened = (observed - mean) @ whitener
    train = whitened[:train_stop]
    cs = fit_cs_parzen_ica(
        train.T, bandwidth=float(config["cs_parzen"]["bandwidth"]),
        screen_step_degrees=1.0, refine_half_width_degrees=1.0, refine_step_degrees=0.1,
    )
    cs_recovered = whitened @ cs.demixing.T
    matrix_config = config["matrix_tc"]
    train_tensor = torch.as_tensor(train, dtype=torch.float64)
    matrix_first = fit_matrix_tc_ica(
        train_tensor, matrix_config["bandwidths"], seed=seed,
        steps=int(matrix_config["steps"]), learning_rate=float(matrix_config["learning_rate"]),
    )
    matrix_second = fit_matrix_tc_ica(
        train_tensor, matrix_config["bandwidths"], seed=seed,
        steps=int(matrix_config["steps"]), learning_rate=float(matrix_config["learning_rate"]),
    )
    distinct_seed_fits = [
        fit_matrix_tc_ica(
            train_tensor, matrix_config["bandwidths"], seed=seed + offset,
            steps=int(matrix_config["steps"]), learning_rate=float(matrix_config["learning_rate"]),
        )
        for offset in range(3)
    ]
    matrix_recovered = torch.as_tensor(whitened, dtype=torch.float64) @ matrix_first.rotation.T
    test_start = int(config["data"]["split"]["test"][0])
    cs_mean, cs_components = _score(sources[test_start:], cs_recovered[test_start:])
    matrix_mean, matrix_components = _score(sources[test_start:], matrix_recovered[test_start:].numpy())
    same_seed_difference = float(torch.max(torch.abs(matrix_first.rotation - matrix_second.rotation)))
    seed_scores = []
    for fit in distinct_seed_fits:
        recovered = torch.as_tensor(whitened, dtype=torch.float64) @ fit.rotation.T
        score, _ = _score(sources[test_start:], recovered[test_start:].numpy())
        seed_scores.append(score)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    fingerprint = _fingerprint(args.config, config["data"]["identifier"], commit)
    summary = {
        "schema_version": 1, "status": "phase1_synthetic_smoke_complete",
        "baseline_fingerprint": fingerprint, "git_commit": commit,
        "split": config["data"]["split"], "labels_used": False, "real_video_used": False,
        "same_seed_max_abs_rotation_difference": same_seed_difference,
        "same_seed_gate_pass": same_seed_difference <= 1e-6,
        "distinct_seed_matrix_tc_test_correlations": seed_scores,
        "distinct_seed_matrix_tc_mean": float(np.mean(seed_scores)),
        "distinct_seed_matrix_tc_std": float(np.std(seed_scores)),
        "cs_parzen": {"test_mean_absolute_source_correlation": cs_mean, "components": cs_components, "objective": cs.objective},
        "matrix_tc_alpha2": {"test_mean_absolute_source_correlation": matrix_mean, "components": matrix_components, "objective": matrix_first.objective, "orthogonality_error": matrix_first.orthogonality_error},
        "interpretation": "Numerical and synthetic engineering evidence only; no real-video or neuronal-performance claim.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    labels = ["CS-Parzen", "Matrix-TC α=2"]
    axes[0].bar(labels, [cs_mean, matrix_mean], color=["#526D82", "#D28C45"])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Held-out |source correlation|")
    axes[0].set_title("Matched synthetic recovery")
    axes[1].scatter(sources[test_start:, 0], matrix_recovered[test_start:, 0], s=12, alpha=0.65, color="#D28C45")
    axes[1].set_xlabel("Synthetic source 1"); axes[1].set_ylabel("Matrix-TC component 1")
    axes[1].set_title("Diagnostic, sign/order unresolved")
    figure.savefig(args.output_dir / "phase1_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
