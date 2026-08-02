"""Frozen generated confirmation for selective-risk survivors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, time
from typing import Any

import numpy as np

from neurobench.metrics.component_reconstruction import component_product_metrics
from neurobench.metrics.source_separation import aligned_source_metrics

from .config import InformationSeparationConfig
from .confidence import decomposition_confidence_features
from .conclusive_config import ConclusiveBatchConfig
from .conclusive_methods import execute_common_input
from .conclusive_stage2 import BASE_FEATURES, _structural_features
from .continuum import make_continuum_fixture, space_filling_continuum
from .screen_runner import _atomic_json


def _predict(model: dict[str, Any], features: dict[str, float]) -> float:
    values = np.asarray([features[name] for name in BASE_FEATURES], dtype=np.float64)
    if model["family"] == "regularized_logistic":
        scaled = (values-np.asarray(model["scaler_mean"]))/np.asarray(model["scaler_scale"])
        logit = float(np.asarray(model["coefficient"])@scaled+float(model["intercept"]))
        return float(1/(1+np.exp(-np.clip(logit, -50, 50))))
    node = 0
    while int(model["children_left"][node]) != int(model["children_right"][node]):
        feature = int(model["feature"][node])
        node = int(model["children_left"][node] if values[feature] <= float(model["thresholds"][node]) else model["children_right"][node])
    counts = np.asarray(model["values"][node], dtype=np.float64).reshape(-1)
    return float(counts[-1]/max(float(counts.sum()), np.finfo(float).eps))


def _fit_id(fixture_id: str, method: dict[str, Any]) -> str:
    encoded = json.dumps({"fixture_id": fixture_id, "method": method}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def run(config: ConclusiveBatchConfig, *, maximum_rows: int | None = None) -> dict[str, Any]:
    partial = Path(str(config.output_root)+".partial")
    stage2_path = partial/"stages"/"02_selective_risk"/"metrics.json"
    if not stage2_path.is_file():
        raise RuntimeError("Stage 2 must complete before generated confirmation")
    stage2 = json.loads(stage2_path.read_text())
    methods = [row for row in stage2["method_results"] if row["evaluation"]["gate_passed"]]
    stage = partial/"stages"/"03_generated_confirmation"
    stage.mkdir(parents=True, exist_ok=True)
    if not methods:
        payload = {"schema_version": 1, "status": "no_candidate_survived_selective_risk",
                   "fit_count": 0, "passing_methods": [],
                   "interpretation": "Generated confirmation stopped because no confidence-calibrated separator passed Stage 2."}
        _atomic_json(stage/"metrics.json", payload)
        return payload
    scientific = InformationSeparationConfig.load(config.scientific_config_path)
    specs = space_filling_continuum(int(config.design["confirmation_fixture_count"]),
                                    seed=20260901, split="confirmation")
    rows_root = stage/"rows"
    rows_root.mkdir(exist_ok=True)
    expected = len(specs)*len(methods)
    completed = len(list(rows_root.glob("*.json")))
    created = 0
    for specification in specs:
        fixture = make_continuum_fixture(specification)
        structural = _structural_features(fixture.observation)
        for method in methods:
            key = _fit_id(specification.fixture_id, method)
            destination = rows_root/f"{key}.json"
            if destination.exists():
                continue
            if maximum_rows is not None and created >= maximum_rows:
                return {"status": "stage3_partial", "completed": completed,
                        "expected": expected, "new_rows": created}
            started = perf_counter()
            def fit(values: np.ndarray, fit_seed: int) -> dict[str, Any]:
                return execute_common_input(values, method_id=method["method_id"],
                    parameters=method["parameters"], scientific_config=scientific,
                    seed=fit_seed, device=str(config.resources["gpu_device"]))
            features, details = decomposition_confidence_features(
                fixture.observation, fit=fit, spatial_shape=fixture.observation.shape[1:],
                seed=specification.seed,
                perturbations=int(config.design["confidence_perturbations"]),
                perturbation_scale=0.004)
            features.update(structural)
            probability = _predict(method["selected_model"], features)
            resolved = probability >= float(method["threshold"])
            base = details["base"]
            recovery = aligned_source_metrics(fixture.traces, base["sources"]) if fixture.identifiable else None
            fidelity = component_product_metrics(
                fixture.traces, fixture.footprints, base["sources"], base["spatial_maps"],
                background=fixture.background, structured_artifact=fixture.structured_artifact,
            ) if fixture.identifiable else None
            _atomic_json(destination, {"fit_id": key, "fixture": specification.to_dict(),
                "method_id": method["method_id"], "parameters": method["parameters"],
                "probability_identifiable": probability, "reported_resolved": resolved,
                "converged": bool(base["converged"]), "recovery": recovery,
                "fidelity": fidelity, "runtime_seconds": perf_counter()-started})
            completed += 1; created += 1
            with (stage/"progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"completed": completed, "expected": expected,
                    "method_id": method["method_id"], "fit_id": key, "unix": time()}, sort_keys=True)+"\n")
            _atomic_json(stage/"heartbeat.json", {"completed": completed, "expected": expected, "updated_unix": time()})
    rows = [json.loads(path.read_text()) for path in sorted(rows_root.glob("*.json"))]
    results = []
    for method in methods:
        group = [row for row in rows if row["method_id"] == method["method_id"]]
        positives = [row for row in group if row["fixture"]["identifiable"]]
        negatives = [row for row in group if not row["fixture"]["identifiable"]]
        resolved_positive = [row for row in positives if row["reported_resolved"]]
        false_resolution = sum(row["reported_resolved"] for row in negatives)
        coverage = sum(row["reported_resolved"] for row in positives)/len(positives)
        fidelity_pass = bool(resolved_positive and
            np.mean([row["fidelity"]["mean_peak_retention"] >= float(config.gates["minimum_peak_retention"])
                     and row["fidelity"]["mean_area_retention"] >= float(config.gates["minimum_area_retention"])
                     and row["fidelity"]["mean_waveform_correlation"] >= float(config.gates["minimum_waveform_correlation"])
                     and row["fidelity"]["maximum_absolute_peak_error_frames"] <= int(config.gates["maximum_timing_error_frames"])
                     for row in resolved_positive]) >= 0.8)
        passed = bool(false_resolution == 0 and coverage >= float(config.gates["minimum_identifiable_coverage"])
                      and np.mean([row["converged"] for row in group]) >= float(config.gates["minimum_converged_fraction"])
                      and fidelity_pass)
        results.append({"method_id": method["method_id"], "parameters": method["parameters"],
                        "false_resolution_count": false_resolution,
                        "identifiable_coverage": coverage,
                        "converged_fraction": float(np.mean([row["converged"] for row in group])),
                        "fidelity_gate_passed": fidelity_pass, "gate_passed": passed})
    payload = {"schema_version": 1, "status": "generated_confirmation_complete",
               "fit_count": len(rows), "method_results": results,
               "passing_methods": [row["method_id"] for row in results if row["gate_passed"]]}
    _atomic_json(stage/"metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(ConclusiveBatchConfig.load(args.config), maximum_rows=args.maximum_rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
