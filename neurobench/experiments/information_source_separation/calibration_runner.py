"""Resumable disjoint identifiability calibration and evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from neurobench.metrics.source_separation import aligned_source_metrics

from .calibration_config import CalibrationConfig
from .confidence import decomposition_confidence_features
from .gpu_screen import _execute_cuda_method
from .identifiability import IDENTIFIABLE_CASES, assert_distinct_case_contract, make_identifiability_fixture
from .screen_runner import _atomic_json


FEATURE_NAMES = (
    "qualification_top_score", "qualification_margin",
    "temporal_stability_mean", "temporal_stability_worst",
    "spatial_stability_mean", "spatial_stability_worst",
    "maximum_source_pair_correlation", "maximum_mixing_pair_cosine",
    "relative_observation_residual",
)


def audit_calibration(config: CalibrationConfig, *, output_dir: Path) -> dict[str, Any]:
    """Read-only collision, fixture, input, disk, and CUDA audit."""
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    fixture_audit = assert_distinct_case_contract(seed=101)
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        if cuda:
            index = torch.device(str(config.resources["device"])).index or 0
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            free_gpu_mib = free_bytes / 2**20
            gpu_name = torch.cuda.get_device_properties(index).name
        else:
            free_gpu_mib, total_bytes, gpu_name = 0.0, 0, None
    except (ImportError, RuntimeError, ValueError):
        cuda, free_gpu_mib, total_bytes, gpu_name = False, 0.0, 0, None
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    free_disk_mib = shutil.disk_usage(probe).free / 2**20
    base_fits = (config.split_count("calibration") + config.split_count("evaluation")) * len(config.methods)
    numerical_fits = base_fits * (1 + int(config.confidence["perturbations"]))
    gates = {
        "source_exists": config.scientific_config.source_video.is_file(),
        "output_absent": not output.exists(), "partial_output_absent": not partial.exists(),
        "fixture_labels_numerically_distinct": not fixture_audit["collisions"],
        "case_families_disjoint": not set(config.calibration["case_ids"]) & set(config.evaluation["case_ids"]),
        "seeds_disjoint": not set(config.calibration["seeds"]) & set(config.evaluation["seeds"]),
        "cuda_available": cuda,
        "gpu_headroom_sufficient": free_gpu_mib >= int(config.resources["minimum_free_gpu_mib"]),
        "disk_headroom_sufficient": free_disk_mib >= int(config.resources["minimum_free_disk_mib"]),
        "cpu_threads_bounded": 1 <= int(config.resources["cpu_threads"]) <= 8,
    }
    return {
        "schema_version": 1, "kind": "identifiability_calibration_read_only_preflight",
        "experiment_id": config.experiment_id, "ready": bool(all(gates.values())),
        "run_authorized": False, "output_dir": str(output), "gates": gates,
        "fixture_audit": fixture_audit,
        "counts": {"calibration_rows": config.split_count("calibration") * len(config.methods),
                   "evaluation_rows": config.split_count("evaluation") * len(config.methods),
                   "base_fits": base_fits, "numerical_fits_including_perturbations": numerical_fits},
        "resources": {"gpu_name": gpu_name, "free_gpu_mib": free_gpu_mib,
                      "total_gpu_mib": total_bytes / 2**20, "free_disk_mib": free_disk_mib, **config.resources},
    }


def _fit_key(split: str, case: str, seed: int, snr: float, method: dict[str, Any]) -> str:
    encoded = json.dumps({"split": split, "case": case, "seed": seed, "snr": snr, "method": method}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def _specifications(config: CalibrationConfig, split: str):
    values = config.calibration if split == "calibration" else config.evaluation
    for case in values["case_ids"]:
        for seed in values["seeds"]:
            for snr in values["snr_levels"]:
                yield str(case), int(seed), float(snr)


def _confidence_model(rows: list[dict[str, Any]], c_grid: list[float]) -> dict[str, Any]:
    x = np.asarray([[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows])
    y = np.asarray([int(row["identifiable"]) for row in rows])
    groups = np.asarray([str(row["case_id"]) for row in rows])
    logo = LeaveOneGroupOut()
    candidates = []
    for c_value in c_grid:
        probabilities = np.zeros(len(rows), dtype=np.float64)
        for train, held in logo.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(C=float(c_value), solver="liblinear", random_state=20260801, max_iter=2000)
            model.fit(scaler.transform(x[train]), y[train])
            probabilities[held] = model.predict_proba(scaler.transform(x[held]))[:, 1]
        negative_max = float(np.max(probabilities[y == 0]))
        threshold = float(np.nextafter(negative_max, 1.0))
        predicted = probabilities >= threshold
        false_resolution = int(np.sum(predicted & (y == 0)))
        true_resolution = float(np.mean(predicted[y == 1]))
        abstention = float(np.mean(~predicted[y == 0]))
        balanced = 0.5 * (true_resolution + abstention)
        candidates.append({
            "c": float(c_value), "threshold": threshold,
            "false_resolution_count": false_resolution,
            "identifiable_resolution_rate": true_resolution,
            "unidentifiable_abstention_rate": abstention,
            "balanced_accuracy": balanced,
            "oof_probabilities": probabilities.tolist(),
        })
    candidates.sort(key=lambda row: (
        row["false_resolution_count"], -row["identifiable_resolution_rate"],
        -row["balanced_accuracy"], row["c"],
    ))
    selected = candidates[0]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=selected["c"], solver="liblinear", random_state=20260801, max_iter=2000)
    model.fit(scaler.transform(x), y)
    return {
        "feature_names": list(FEATURE_NAMES), "selected": selected,
        "candidates": candidates,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
    }


def _predict(model: dict[str, Any], features: dict[str, float]) -> float:
    values = np.asarray([features[name] for name in model["feature_names"]], dtype=np.float64)
    scaled = (values - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
    logit = float(np.asarray(model["coefficient"]) @ scaled + float(model["intercept"]))
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*np.sqrt((p*(1-p)+z*z/(4*total))/total)/denominator
    return [float(max(0, center-half)), float(min(1, center+half))]


def run_calibration(config: CalibrationConfig, *, output_dir: Path, resume: bool = False) -> dict[str, Any]:
    """Execute feature fits, freeze calibration models, and evaluate held families."""
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    if output.exists():
        raise FileExistsError(f"completed calibration exists: {output}")
    if partial.exists() and not resume:
        raise FileExistsError("partial calibration exists; audit then pass resume=True")
    if not partial.exists():
        partial.mkdir(parents=True, exist_ok=False)
        (partial / "rows").mkdir()
        _atomic_json(partial / "config.resolved.json", config.to_dict())
    elif json.loads((partial / "config.resolved.json").read_text()) != config.to_dict():
        raise RuntimeError("resume configuration differs")
    all_rows: list[dict[str, Any]] = []
    expected = (config.split_count("calibration") + config.split_count("evaluation")) * len(config.methods)
    progress = partial / "progress.jsonl"
    for split in ("calibration", "evaluation"):
        for case, seed, snr in _specifications(config, split):
            fixture = make_identifiability_fixture(case, seed=seed, snr=snr)
            for method in config.methods:
                key = _fit_key(split, case, seed, snr, method)
                path = partial / "rows" / f"{key}.json"
                if path.exists():
                    all_rows.append(json.loads(path.read_text()))
                    continue
                started = perf_counter()
                method_id = str(method["method_id"])
                parameters = dict(method["parameters"])
                def fit(movie: np.ndarray, fit_seed: int) -> dict[str, Any]:
                    return _execute_cuda_method(
                        movie, method_id, parameters, config.scientific_config,
                        fit_seed, str(config.resources["device"]),
                    )
                features, details = decomposition_confidence_features(
                    fixture.observation, fit=fit, spatial_shape=fixture.observation.shape[1:],
                    seed=seed, perturbations=int(config.confidence["perturbations"]),
                    perturbation_scale=float(config.confidence["perturbation_scale"]),
                )
                base = details["base"]
                recovery = (
                    aligned_source_metrics(fixture.traces, base["sources"])
                    if fixture.identifiable else None
                )
                row = {
                    "fit_id": key, "split": split, "case_id": case, "seed": seed, "snr": snr,
                    "identifiable": bool(fixture.identifiable), "method_id": method_id,
                    "configuration_json": json.dumps(parameters, sort_keys=True),
                    "execution_backend": base["execution_backend"],
                    "converged": bool(base["converged"]), "iterations": int(base["iterations"]),
                    "runtime_seconds": float(perf_counter()-started), "features": features,
                    "qualification_status": details["qualification"]["status"],
                    "recovery": recovery, "fixture_metadata": fixture.metadata,
                }
                _atomic_json(path, row)
                all_rows.append(row)
                with progress.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"completed": len(all_rows), "total": expected, "fit_id": key, "split": split, "case_id": case, "method_id": method_id}, sort_keys=True)+"\n")
    if len(all_rows) != expected:
        raise RuntimeError(f"row count differs: {len(all_rows)} != {expected}")
    method_results = []
    for method in config.methods:
        method_id = str(method["method_id"])
        calibration_rows = [row for row in all_rows if row["method_id"] == method_id and row["split"] == "calibration"]
        evaluation_rows = [row for row in all_rows if row["method_id"] == method_id and row["split"] == "evaluation"]
        model = _confidence_model(calibration_rows, [float(value) for value in config.confidence["logistic_c_grid"]])
        threshold = float(model["selected"]["threshold"])
        predictions = []
        for row in evaluation_rows:
            probability = _predict(model, row["features"])
            predictions.append({"fit_id": row["fit_id"], "case_id": row["case_id"], "seed": row["seed"], "snr": row["snr"], "identifiable": row["identifiable"], "probability_identifiable": probability, "reported_resolved": probability >= threshold, "mean_absolute_correlation": None if row["recovery"] is None else row["recovery"]["mean_absolute_correlation"], "mean_absolute_crosstalk": None if row["recovery"] is None else row["recovery"]["mean_absolute_crosstalk"]})
        negatives = [row for row in predictions if not row["identifiable"]]
        positives = [row for row in predictions if row["identifiable"]]
        false_resolution = sum(bool(row["reported_resolved"]) for row in negatives)
        true_resolution = sum(bool(row["reported_resolved"]) for row in positives)
        converged_fraction = float(np.mean([row["converged"] for row in evaluation_rows]))
        gate_passed = bool(
            false_resolution <= int(config.gate["maximum_false_resolution_count"])
            and true_resolution / len(positives) >= float(config.gate["minimum_identifiable_resolution_rate"])
            and converged_fraction >= float(config.gate["minimum_converged_fraction"])
        )
        resolved_positive = [row for row in predictions if row["identifiable"] and row["reported_resolved"]]
        method_results.append({
            "method_id": method_id, "configuration_json": json.dumps(method["parameters"], sort_keys=True),
            "model": model, "evaluation_predictions": predictions,
            "evaluation": {
                "identifiable_count": len(positives), "unidentifiable_count": len(negatives),
                "true_resolution_count": true_resolution, "false_abstention_count": len(positives)-true_resolution,
                "false_resolution_count": false_resolution, "true_abstention_count": len(negatives)-false_resolution,
                "identifiable_resolution_rate": true_resolution/len(positives),
                "identifiable_resolution_rate_wilson95": _wilson(true_resolution, len(positives)),
                "false_resolution_rate": false_resolution/len(negatives),
                "false_resolution_rate_wilson95": _wilson(false_resolution, len(negatives)),
                "converged_fraction": converged_fraction,
                "selective_mean_correlation": float(np.mean([row["mean_absolute_correlation"] for row in resolved_positive])) if resolved_positive else None,
                "selective_mean_crosstalk": float(np.mean([row["mean_absolute_crosstalk"] for row in resolved_positive])) if resolved_positive else None,
                "gate_passed": gate_passed,
            },
        })
    payload = {
        "schema_version": 1, "status": "identifiability_calibration_complete",
        "experiment_id": config.experiment_id, "row_count": len(all_rows),
        "numerical_fit_count": len(all_rows)*(1+int(config.confidence["perturbations"])),
        "calibration_case_families": config.calibration["case_ids"],
        "evaluation_case_families": config.evaluation["case_ids"],
        "method_results": method_results,
        "passing_methods": [row["method_id"] for row in method_results if row["evaluation"]["gate_passed"]],
        "confirmation_authorized": False,
        "interpretation": "Confidence was calibrated on disjoint case families/seeds and evaluated once. Passing this gate makes confirmation selectable; it does not run it.",
    }
    _atomic_json(partial / "metrics.json", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output)
    return payload
