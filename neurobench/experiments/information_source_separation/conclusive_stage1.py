"""Resumable continuous-identifiability development screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, time
from typing import Any

from neurobench.metrics.source_separation import aligned_source_metrics

from .config import InformationSeparationConfig
from .conclusive_config import ConclusiveBatchConfig
from .conclusive_methods import GENERATED_COMMON_INPUT_METHODS, execute_common_input
from .continuum import make_continuum_fixture, space_filling_continuum
from .qualification import qualify_temporal_components
from .screen_runner import _atomic_json


def _key(specification: dict[str, Any], method_id: str, parameters: dict[str, Any]) -> str:
    value = json.dumps({"fixture": specification, "method_id": method_id,
                        "parameters": parameters}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _methods(config: ConclusiveBatchConfig) -> list[dict[str, Any]]:
    rows = []
    for method in config.methods:
        if not method["enabled"] or method["method_id"] not in GENERATED_COMMON_INPUT_METHODS:
            continue
        for parameters in method["configurations"]:
            rows.append({"method_id": method["method_id"], "track": method["track"],
                         "parameters": dict(parameters)})
    return rows


def run(config: ConclusiveBatchConfig, *, maximum_fits: int | None = None) -> dict[str, Any]:
    """Execute or resume Stage 1 inside the prepared partial batch root."""
    partial = Path(str(config.output_root)+".partial")
    if not partial.is_dir() or not (partial/"run_state.json").is_file():
        raise RuntimeError("batch must be prepared before Stage 1")
    scientific = InformationSeparationConfig.load(config.scientific_config_path)
    specifications = space_filling_continuum(
        int(config.design["development_fixture_count"]), seed=20260811,
        split="development",
    )
    methods = _methods(config)
    stage = partial/"stages"/"01_continuous_identifiability"
    fits = stage/"fits"
    fits.mkdir(parents=True, exist_ok=True)
    expected = len(specifications)*len(methods)
    completed = 0
    new_fits = 0
    for specification in specifications:
        fixture = make_continuum_fixture(specification)
        spec_dict = specification.to_dict()
        for method in methods:
            key = _key(spec_dict, method["method_id"], method["parameters"])
            path = fits/f"{key}.json"
            if path.exists():
                completed += 1
                continue
            if maximum_fits is not None and new_fits >= maximum_fits:
                return {"status": "stage1_partial", "completed": completed,
                        "expected": expected, "new_fits": new_fits}
            started = perf_counter()
            execution = execute_common_input(
                fixture.observation, method_id=method["method_id"],
                parameters=method["parameters"], scientific_config=scientific,
                seed=specification.seed, device=str(config.resources["gpu_device"]),
            )
            qualification = qualify_temporal_components(
                execution["spatial_maps"], execution["sources"],
                spatial_shape=fixture.observation.shape[1:],
            )
            recovery = aligned_source_metrics(fixture.traces, execution["sources"]) if fixture.identifiable else None
            row = {
                "fit_id": key, "fixture": spec_dict, "fixture_metadata": fixture.metadata,
                "method_id": method["method_id"], "track": method["track"],
                "parameters": method["parameters"],
                "converged": bool(execution["converged"]),
                "iterations": int(execution["iterations"]),
                "execution_backend": execution["execution_backend"],
                "relative_observation_residual": execution["relative_observation_residual"],
                "qualification": qualification,
                "reported_resolved": qualification["status"] == "resolved",
                "recovery": recovery, "runtime_seconds": perf_counter()-started,
            }
            _atomic_json(path, row)
            completed += 1
            new_fits += 1
            with (stage/"progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"completed": completed, "expected": expected,
                                         "fit_id": key, "method_id": method["method_id"],
                                         "fixture_id": specification.fixture_id,
                                         "unix": time()}, sort_keys=True)+"\n")
            _atomic_json(stage/"heartbeat.json", {"completed": completed,
                         "expected": expected, "updated_unix": time()})
    rows = [json.loads(path.read_text()) for path in sorted(fits.glob("*.json"))]
    summaries = []
    for method_id in sorted({row["method_id"] for row in rows}):
        configs = sorted({json.dumps(row["parameters"], sort_keys=True) for row in rows if row["method_id"] == method_id})
        for parameters in configs:
            group = [row for row in rows if row["method_id"] == method_id and json.dumps(row["parameters"], sort_keys=True) == parameters]
            positives = [row for row in group if row["fixture"]["identifiable"]]
            negatives = [row for row in group if not row["fixture"]["identifiable"]]
            recovered = [row["recovery"] for row in positives]
            summaries.append({
                "method_id": method_id, "parameters": json.loads(parameters),
                "fit_count": len(group),
                "converged_fraction": sum(row["converged"] for row in group)/len(group),
                "naive_identifiable_coverage": sum(row["reported_resolved"] for row in positives)/len(positives),
                "naive_false_resolution_count": sum(row["reported_resolved"] for row in negatives),
                "mean_absolute_correlation": sum(item["mean_absolute_correlation"] for item in recovered)/len(recovered),
                "worst_absolute_correlation": min(item["worst_absolute_correlation"] for item in recovered),
                "mean_absolute_crosstalk": sum(item["mean_absolute_crosstalk"] for item in recovered)/len(recovered),
            })
    payload = {
        "schema_version": 1, "status": "continuous_identifiability_screen_complete",
        "fit_count": len(rows), "fixture_count": len(specifications),
        "configuration_count": len(methods), "summaries": summaries,
        "stage_inapplicable_methods": sorted({item["method_id"] for item in config.methods if item["enabled"]}-GENERATED_COMMON_INPUT_METHODS),
        "interpretation": "Naive qualification is diagnostic only. Stage 2 must calibrate selective risk before advancement.",
    }
    _atomic_json(stage/"metrics.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-fits", type=int)
    args = parser.parse_args(argv)
    payload = run(ConclusiveBatchConfig.load(args.config), maximum_fits=args.maximum_fits)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
