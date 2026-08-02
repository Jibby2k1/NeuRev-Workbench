"""Selective-risk calibration on disjoint continuous-identifiability splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from neurobench.metrics.source_separation import aligned_source_metrics

from .config import InformationSeparationConfig
from .confidence import decomposition_confidence_features
from .conclusive_config import ConclusiveBatchConfig
from .conclusive_methods import execute_common_input
from .continuum import make_continuum_fixture, space_filling_continuum
from .screen_runner import _atomic_json


BASE_FEATURES = (
    "qualification_top_score", "qualification_margin",
    "temporal_stability_mean", "temporal_stability_worst",
    "spatial_stability_mean", "spatial_stability_worst",
    "maximum_source_pair_correlation", "maximum_mixing_pair_cosine",
    "relative_observation_residual", "observation_spectral_entropy",
    "observation_leading_fraction", "observation_rank4_condition",
)


def _structural_features(movie: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(movie, dtype=np.float64).reshape(len(movie), -1)
    matrix -= matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energy = singular*singular
    fractions = energy/max(float(energy.sum()), np.finfo(float).eps)
    positive = fractions[fractions > 0]
    entropy = -float(np.sum(positive*np.log(positive)))/max(np.log(len(fractions)), 1.0)
    index = min(3, len(singular)-1)
    return {
        "observation_spectral_entropy": entropy,
        "observation_leading_fraction": float(fractions[0]),
        "observation_rank4_condition": float(singular[0]/max(singular[index], np.finfo(float).eps)),
    }


def _selected_methods(stage1: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for method_id in sorted({row["method_id"] for row in stage1["summaries"]}):
        rows = [row for row in stage1["summaries"] if row["method_id"] == method_id]
        rows.sort(key=lambda row: (
            row["converged_fraction"] < 0.95,
            -row["mean_absolute_correlation"],
            row["mean_absolute_crosstalk"],
            json.dumps(row["parameters"], sort_keys=True),
        ))
        selected.append({"method_id": method_id, "parameters": rows[0]["parameters"]})
    return selected


def _fit_id(split: str, fixture_id: str, method: dict[str, Any]) -> str:
    encoded = json.dumps({"split": split, "fixture_id": fixture_id,
                          "method": method}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def _threshold_from_negative_calibration(probabilities: np.ndarray, labels: np.ndarray) -> float:
    negatives = probabilities[labels == 0]
    if not len(negatives):
        raise RuntimeError("confidence calibration requires negative controls")
    return float(np.nextafter(float(np.max(negatives)), 1.0))


def _candidate_models(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    candidates: list[dict[str, Any]] = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c_value, solver="liblinear", max_iter=2000,
                                   random_state=20260801, class_weight="balanced")
        model.fit(scaled, y)
        candidates.append({"family": "regularized_logistic", "complexity": c_value,
                           "model": model, "scaler": scaler})
    for depth in (1, 2, 3):
        for leaf in (2, 4, 8):
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                           random_state=20260801,
                                           class_weight="balanced")
            model.fit(x, y)
            candidates.append({"family": "bounded_tree", "complexity": [depth, leaf],
                               "model": model, "scaler": None})
    return candidates


def _probability(candidate: dict[str, Any], x: np.ndarray) -> np.ndarray:
    values = candidate["scaler"].transform(x) if candidate["scaler"] is not None else x
    return candidate["model"].predict_proba(values)[:, 1]


def _serialize_model(candidate: dict[str, Any]) -> dict[str, Any]:
    model = candidate["model"]
    payload = {"family": candidate["family"], "complexity": candidate["complexity"]}
    if candidate["family"] == "regularized_logistic":
        payload.update({"scaler_mean": candidate["scaler"].mean_.tolist(),
                        "scaler_scale": candidate["scaler"].scale_.tolist(),
                        "coefficient": model.coef_[0].tolist(),
                        "intercept": float(model.intercept_[0])})
    else:
        payload.update({"feature": model.tree_.feature.tolist(),
                        "thresholds": model.tree_.threshold.tolist(),
                        "children_left": model.tree_.children_left.tolist(),
                        "children_right": model.tree_.children_right.tolist(),
                        "values": model.tree_.value.tolist()})
    return payload


def run(config: ConclusiveBatchConfig, *, maximum_rows: int | None = None) -> dict[str, Any]:
    partial = Path(str(config.output_root)+".partial")
    stage1_path = partial/"stages"/"01_continuous_identifiability"/"metrics.json"
    if not stage1_path.is_file():
        raise RuntimeError("Stage 1 must complete before selective-risk calibration")
    stage1 = json.loads(stage1_path.read_text())
    methods = _selected_methods(stage1)
    scientific = InformationSeparationConfig.load(config.scientific_config_path)
    split_contract = {"train": (48, 20260821), "calibration": (48, 20260831),
                      "evaluation": (96, 20260841)}
    stage = partial/"stages"/"02_selective_risk"
    rows_root = stage/"rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    expected = sum(count for count, _ in split_contract.values())*len(methods)
    completed = len(list(rows_root.glob("*.json")))
    created = 0
    for split, (count, seed) in split_contract.items():
        for specification in space_filling_continuum(count, seed=seed, split=split):
            fixture = make_continuum_fixture(specification)
            structural = _structural_features(fixture.observation)
            for method in methods:
                key = _fit_id(split, specification.fixture_id, method)
                destination = rows_root/f"{key}.json"
                if destination.exists():
                    continue
                if maximum_rows is not None and created >= maximum_rows:
                    return {"status": "stage2_partial", "completed": completed,
                            "expected": expected, "new_rows": created}
                started = perf_counter()
                def fit(values: np.ndarray, fit_seed: int) -> dict[str, Any]:
                    return execute_common_input(
                        values, method_id=method["method_id"], parameters=method["parameters"],
                        scientific_config=scientific, seed=fit_seed,
                        device=str(config.resources["gpu_device"]),
                    )
                features, details = decomposition_confidence_features(
                    fixture.observation, fit=fit, spatial_shape=fixture.observation.shape[1:],
                    seed=specification.seed,
                    perturbations=int(config.design["confidence_perturbations"]),
                    perturbation_scale=0.004,
                )
                features.update(structural)
                base = details["base"]
                recovery = aligned_source_metrics(fixture.traces, base["sources"]) if fixture.identifiable else None
                row = {"fit_id": key, "split": split, "fixture": specification.to_dict(),
                       "fixture_metadata": fixture.metadata, "method_id": method["method_id"],
                       "parameters": method["parameters"], "features": features,
                       "converged": bool(base["converged"]), "recovery": recovery,
                       "runtime_seconds": perf_counter()-started}
                _atomic_json(destination, row)
                completed += 1
                created += 1
                with (stage/"progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"completed": completed, "expected": expected,
                                             "fit_id": key, "split": split,
                                             "method_id": method["method_id"],
                                             "unix": time()}, sort_keys=True)+"\n")
                _atomic_json(stage/"heartbeat.json", {"completed": completed,
                             "expected": expected, "updated_unix": time()})
    rows = [json.loads(path.read_text()) for path in sorted(rows_root.glob("*.json"))]
    results = []
    for method in methods:
        method_rows = [row for row in rows if row["method_id"] == method["method_id"]]
        by_split = {name: [row for row in method_rows if row["split"] == name]
                    for name in split_contract}
        def arrays(name: str):
            values = by_split[name]
            return (np.asarray([[row["features"][key] for key in BASE_FEATURES] for row in values]),
                    np.asarray([int(row["fixture"]["identifiable"]) for row in values]))
        x_train, y_train = arrays("train")
        x_cal, y_cal = arrays("calibration")
        x_eval, y_eval = arrays("evaluation")
        candidates = []
        for candidate in _candidate_models(x_train, y_train):
            cal_probability = _probability(candidate, x_cal)
            threshold = _threshold_from_negative_calibration(cal_probability, y_cal)
            cal_resolved = cal_probability >= threshold
            candidates.append({"candidate": candidate, "threshold": threshold,
                               "calibration_false_resolutions": int(np.sum(cal_resolved & (y_cal == 0))),
                               "calibration_coverage": float(np.mean(cal_resolved[y_cal == 1])),
                               "calibration_brier": float(brier_score_loss(y_cal, cal_probability))})
        candidates.sort(key=lambda row: (row["calibration_false_resolutions"],
                                         -row["calibration_coverage"], row["calibration_brier"],
                                         row["candidate"]["family"], str(row["candidate"]["complexity"])))
        selected = candidates[0]
        probability = _probability(selected["candidate"], x_eval)
        resolved = probability >= selected["threshold"]
        false_resolution = int(np.sum(resolved & (y_eval == 0)))
        coverage = float(np.mean(resolved[y_eval == 1]))
        convergence = float(np.mean([row["converged"] for row in by_split["evaluation"]]))
        artifact_coverage = {}
        for family in sorted({row["fixture"]["artifact_family"] for row in by_split["evaluation"] if row["fixture"]["identifiable"]}):
            indices = [index for index, row in enumerate(by_split["evaluation"])
                       if row["fixture"]["identifiable"] and row["fixture"]["artifact_family"] == family]
            artifact_coverage[family] = float(np.mean(resolved[indices]))
        catastrophic = any(value < 0.5 for value in artifact_coverage.values())
        passed = bool(false_resolution <= int(config.gates["maximum_false_resolution_count"])
                      and coverage >= float(config.gates["minimum_identifiable_coverage"])
                      and convergence >= float(config.gates["minimum_converged_fraction"])
                      and not catastrophic)
        results.append({"method_id": method["method_id"], "parameters": method["parameters"],
                        "feature_names": list(BASE_FEATURES),
                        "selected_model": _serialize_model(selected["candidate"]),
                        "threshold": selected["threshold"],
                        "calibration": {key: value for key, value in selected.items() if key != "candidate"},
                        "evaluation": {"false_resolution_count": false_resolution,
                                       "unidentifiable_count": int(np.sum(y_eval == 0)),
                                       "identifiable_coverage": coverage,
                                       "identifiable_count": int(np.sum(y_eval == 1)),
                                       "converged_fraction": convergence,
                                       "artifact_family_coverage": artifact_coverage,
                                       "catastrophic_family": catastrophic,
                                       "brier_score": float(brier_score_loss(y_eval, probability)),
                                       "gate_passed": passed},
                        "evaluation_predictions": probability.tolist(),
                        "evaluation_resolved": resolved.tolist()})
    payload = {"schema_version": 1, "status": "selective_risk_complete",
               "row_count": len(rows), "numerical_fit_count": len(rows)*(1+int(config.design["confidence_perturbations"])),
               "split_contract": {key: {"fixture_count": value[0], "seed": value[1]} for key, value in split_contract.items()},
               "method_results": results,
               "passing_methods": [row["method_id"] for row in results if row["evaluation"]["gate_passed"]],
               "interpretation": "Models were trained, thresholded, and evaluated on disjoint fixture seeds. Evaluation labels were not used to retune thresholds."}
    _atomic_json(stage/"metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args(argv)
    payload = run(ConclusiveBatchConfig.load(args.config), maximum_rows=args.maximum_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
