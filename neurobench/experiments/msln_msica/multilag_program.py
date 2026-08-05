"""Staged, label-sealed multi-lag MSICA and Raw->MSICA->MSLN program."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from neurobench.algorithms.multilag_msica import (
    TemporalMSICAFit,
    fit_delay_embedding,
    fit_multilag_2d,
    gather_delay_embedding,
    gather_multilag_pairs,
    lag_weights,
    project_temporal_fit_chunked,
    sample_anchor_indices,
)
from neurobench.algorithms.multiscale_local_normalization import JointSTContext, causal_joint_msln
from neurobench.algorithms.msln_msica_cuda import causal_joint_msln_cuda, cuda_device_summary
from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one
from neurobench.experiments.learnable_contrast import core as label_core

from .artifacts import atomic_json, sha256_file, sha256_payload
from .joint_sweep import _label_overlay, _visual_stats


FORMULATIONS = ("multilag_2d", "delay_embedding")
BRANCHES = {
    "multilag_2d": ("persistence", "innovation"),
    "delay_embedding": ("persistence", "innovation", "residual_group"),
}


def _load(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_id", "source", "design", "fitting", "evaluation", "compute", "outputs"}
    if set(payload) != required or int(payload["schema_version"]) != 5:
        raise ValueError("multi-lag program requires the exact schema-v5 contract")
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path", "cascade_v4_root"):
        payload["source"][key] = str((config_path.parent / payload["source"][key]).resolve())
    payload["outputs"]["root_dir"] = str((config_path.parent / payload["outputs"]["root_dir"]).resolve())
    payload["_config_path"] = str(config_path)
    _validate(payload)
    return payload


def _validate(config: dict[str, Any]) -> None:
    source, design, fitting, evaluation, compute = (
        config["source"], config["design"], config["fitting"], config["evaluation"], config["compute"]
    )
    if source["axes"] != "TYX" or not source["ui_one_based"]:
        raise ValueError("source must be one-based UI TYX")
    if evaluation["selection_labels_used"] or evaluation["unlabeled_candidates"] != "unknown":
        raise ValueError("labels must remain sealed and unmatched candidates unknown")
    if evaluation["candidate_budgets"] != [20, 40, 58, 80, 100]:
        raise ValueError("candidate budgets must match prior experiments")
    if compute["device"] != "cuda" or int(compute["workers_per_gpu"]) != 1:
        raise ValueError("exactly one CUDA worker is required")
    if not 0 < float(compute["max_peak_vram_gb"]) <= 8 or int(compute["cpu_threads"]) > 4:
        raise ValueError("resource caps exceed the guarded limits")
    if design["spatial_outer_guard_pairs"] != [[5, 1], [7, 1], [7, 3], [11, 3], [15, 3], [15, 5]]:
        raise ValueError("the six prior spatial contexts are frozen")
    if design["temporal_windows_frames"] != [5, 9, 15, 23, 31]:
        raise ValueError("the five prior temporal contexts are frozen")
    if not design["full_mSLN_context_grid"]:
        raise ValueError("all 30 MSLN contexts are required")
    known = {"cs_parzen", "ksg_mi", "normalized_hsic", "matrix_renyi_mi"}
    if set(design["objective_grid"]) != known or set(fitting["estimator_sample_caps"]) != known:
        raise ValueError("the four higher-order objective families are required")
    for profiles in (design["multilag_profiles"], design["embedding_profiles"]):
        for values in profiles.values():
            if values[0] != 0 or values != sorted(set(values)):
                raise ValueError("lag profiles must be unique, increasing, and start at zero")


def _resolved(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _root(config: dict[str, Any]) -> Path:
    return Path(config["outputs"]["root_dir"])


def _adapt(config: dict[str, Any]) -> dict[str, Any]:
    return {"source": config["source"], "evaluation": config["evaluation"], "sweep": config["design"]}


def _contexts(config: dict[str, Any]) -> list[JointSTContext]:
    design = config["design"]
    return [
        JointSTContext(
            f"joint_s{outer}_g{guard}_t{window}_g1",
            int(outer), int(guard), int(window), int(design["temporal_guard_frames"]),
            "mean_std", float(design["scale_floor_percentile"]),
        )
        for outer, guard in design["spatial_outer_guard_pairs"]
        for window in design["temporal_windows_frames"]
    ]


def _parameter_grid(spec: dict[str, list[float | int]]) -> list[dict[str, float | int]]:
    keys = tuple(spec)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(spec[key] for key in keys))]


def _parameter_key(parameter: dict[str, float | int]) -> str:
    return "_".join(f"{key}-{str(value).replace('.', 'p')}" for key, value in sorted(parameter.items()))


def _config_id(formulation: str, objective: str, profile: str, parameter: dict[str, float | int], weight: str | None = None) -> str:
    pieces = [formulation, objective, profile]
    if weight is not None:
        pieces.append(weight)
    pieces.append(_parameter_key(parameter))
    return "__".join(pieces)


def _heartbeat(root: Path, stage: str, **extra: Any) -> None:
    path = root / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **extra}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _source(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int]:
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    maximum_pair_history = max(max(v) + 1 for v in config["design"]["multilag_profiles"].values())
    maximum_embedding_history = max(max(v) for v in config["design"]["embedding_profiles"].values())
    global_history = max(maximum_pair_history, maximum_embedding_history)
    msln_pre_roll = max(map(int, config["design"]["temporal_windows_frames"]))
    source_start = review_start - 1 - global_history - msln_pre_roll
    values = movie[source_start:review_stop]
    review_count = review_stop - review_start + 1
    quiet = np.zeros(review_count, dtype=bool)
    quiet_start, quiet_stop = map(int, config["source"]["quiet_interval_ui"])
    quiet[quiet_start - review_start:quiet_stop - review_start + 1] = True
    expected = global_history + msln_pre_roll + review_count
    if len(values) != expected:
        raise RuntimeError("source alignment invariant failed")
    return values, quiet, global_history, msln_pre_roll


def _fit_from_dict(payload: dict[str, Any]) -> TemporalMSICAFit:
    return TemporalMSICAFit(
        formulation=payload["formulation"],
        objective_family=payload["objective_family"],
        objective_parameter=dict(payload["objective_parameter"]),
        lags=tuple(payload["lags"]),
        lag_weights=tuple(payload["lag_weights"]),
        center=np.asarray(payload["center"], dtype=np.float64),
        whitening=np.asarray(payload["whitening"], dtype=np.float64),
        rotation=np.asarray(payload["rotation"], dtype=np.float64),
        demixing=np.asarray(payload["demixing"], dtype=np.float64),
        objective=float(payload["objective"]),
        baseline_objective=float(payload["baseline_objective"]),
        persistence_index=int(payload["persistence_index"]),
        innovation_index=int(payload["innovation_index"]),
        residual_indices=tuple(payload["residual_indices"]),
        component_signs=tuple(payload["component_signs"]),
        converged=bool(payload["converged"]),
        diagnostics=dict(payload["diagnostics"]),
    )


def _objective_gain(fit: TemporalMSICAFit) -> float:
    return float((fit.baseline_objective - fit.objective) / max(abs(fit.baseline_objective), 1e-12))


def _sample_fit(
    values: np.ndarray,
    config: dict[str, Any],
    *,
    formulation: str,
    lags: tuple[int, ...],
    objective: str,
    parameter: dict[str, float | int],
    weight_decay: float = 0.0,
    seed_offset: int = 0,
) -> TemporalMSICAFit:
    fitting = config["fitting"]
    cap = int(fitting["estimator_sample_caps"][objective])
    screen_count = min(int(fitting["surface_screen_samples"]), cap)
    confirmation_count = min(int(fitting["surface_confirmation_samples"]), cap)
    history = max(lags) + (1 if formulation == "multilag_2d" else 0)
    screen_anchors = sample_anchor_indices(values.shape, history=history, count=screen_count, seed=int(config["design"]["seed"]) + seed_offset)
    confirmation_anchors = sample_anchor_indices(values.shape, history=history, count=confirmation_count, seed=int(config["design"]["seed"]) + seed_offset + 1)
    if formulation == "multilag_2d":
        return fit_multilag_2d(
            gather_multilag_pairs(values, screen_anchors, lags),
            gather_multilag_pairs(values, confirmation_anchors, lags),
            lags=lags,
            weights=lag_weights(lags, weight_decay),
            objective_family=objective,
            objective_parameter=parameter,
            coarse_step_degrees=float(fitting["coarse_angle_step_degrees"]),
            refine_half_width_degrees=float(fitting["refine_half_width_degrees"]),
            refine_step_degrees=float(fitting["refine_step_degrees"]),
            sharpness_delta_degrees=float(fitting["objective_sharpness_delta_degrees"]),
            eigenvalue_floor_ratio=float(fitting["eigenvalue_floor_ratio"]),
        )
    return fit_delay_embedding(
        gather_delay_embedding(values, screen_anchors, lags),
        gather_delay_embedding(values, confirmation_anchors, lags),
        lags=lags,
        objective_family=objective,
        objective_parameter=parameter,
        angle_step_degrees=float(fitting["embedding_angle_step_degrees"]),
        max_sweeps=int(fitting["embedding_max_sweeps"]),
        eigenvalue_floor_ratio=float(fitting["eigenvalue_floor_ratio"]),
    )


def _aligned_outputs(
    values: np.ndarray,
    fit: TemporalMSICAFit,
    config: dict[str, Any],
    global_history: int,
    msln_pre_roll: int,
) -> dict[str, np.ndarray]:
    projected = project_temporal_fit_chunked(
        values, fit, backend="cuda",
        frame_chunk=int(config["compute"]["projection_frame_chunk"]),
        output_cpu=True,
    )
    projection_history = max(fit.lags) if fit.formulation == "delay_embedding" else 1
    offset = global_history - projection_history
    expected = msln_pre_roll + (
        int(config["source"]["review_interval_ui"][1]) - int(config["source"]["review_interval_ui"][0]) + 1
    )
    result = {name: np.asarray(array[offset:], dtype=np.float32) for name, array in projected.items()}
    if any(len(array) != expected for array in result.values()):
        raise RuntimeError("projected timeline alignment failed")
    del projected
    return result


def _label_free_artifacts(values: Any, quiet: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    is_cuda = hasattr(values, "__cuda_array_interface__")
    if is_cuda:
        import cupy as cp
        evidence = cp.square(values, dtype=cp.float32)
    else:
        evidence = np.square(np.asarray(values, dtype=np.float32), dtype=np.float32)
    visual = _visual_stats(evidence, quiet, _adapt(config))
    proposals: dict[str, list[list[float | int]]] = {}
    review_start = int(config["source"]["review_interval_ui"][0])
    limit = max(map(int, config["evaluation"]["candidate_budgets"]))
    for burst, interval in sorted(config["source"]["burst_intervals_ui"].items(), key=lambda item: int(item[0])):
        start = int(interval[0]) - review_start
        stop = int(interval[1]) - review_start + 1
        pooled = evidence[start:stop].max(axis=0)
        if is_cuda:
            pooled = cp.asnumpy(pooled)
        peaks = extract_local_maxima(
            np.asarray(pooled), int(config["evaluation"]["nms_distance_px"]), limit=limit
        )
        proposals[str(burst)] = [[float(score), int(x), int(y)] for score, x, y in peaks]
    score = float(
        np.log1p(visual["event_quiet_ratio_p999"])
        + 0.5 * np.log1p(1000.0 * visual["event_fraction_above_quiet_p999"])
    )
    del evidence
    return {"visual_stats": visual, "selection_score": score, "proposals": proposals}


def _protected_from_proposals(proposals: dict[str, list[list[float | int]]], labels: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    budgets = [int(value) for value in config["evaluation"]["candidate_budgets"]]
    totals = {budget: 0 for budget in budgets}
    identities = {budget: [] for budget in budgets}
    rows = []
    total_labels = 0
    for burst_text, peaks_payload in sorted(proposals.items(), key=lambda item: int(item[0])):
        burst = int(burst_text)
        peaks = [(float(score), int(x), int(y)) for score, x, y in peaks_payload]
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
        total_labels += len(burst_labels)
        for budget in budgets:
            matches, _ = match_peaks_one_to_one(peaks[:budget], burst_labels, float(config["evaluation"]["match_radius_px"]))
            totals[budget] += len(matches)
            identities[budget].extend(str(burst_labels[index]["roi_identity"]) for index, *_ in matches)
            rows.append({"burst_id": burst, "budget": budget, "matched": len(matches), "labels": len(burst_labels), "candidates": min(budget, len(peaks))})
    return {
        "rows": rows,
        "total_labels": total_labels,
        "matched_by_budget": {str(key): value for key, value in totals.items()},
        "recall_by_budget": {str(key): value / total_labels if total_labels else 0.0 for key, value in totals.items()},
        "matched_roi_identities_by_budget": {str(key): value for key, value in identities.items()},
        "unmatched_candidates_are": "unknown",
    }


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path)
    root = _root(config)
    if root.exists():
        raise FileExistsError(f"output root exists: {root}")
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path"):
        if not Path(config["source"][key]).is_file():
            raise FileNotFoundError(config["source"][key])
    if not Path(config["source"]["cascade_v4_root"]).is_dir():
        raise FileNotFoundError(config["source"]["cascade_v4_root"])
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    values, quiet, global_history, msln_pre_roll = _source(config)
    labels = label_core.load_labels(Path(config["source"]["labels_path"]))
    if movie.ndim != 3 or movie.shape != (2359, 340, 573) or str(movie.dtype) != "uint16":
        raise ValueError("unexpected source movie contract")
    if not np.isfinite(np.asarray(values[::73, ::19, ::23], dtype=np.float32)).all():
        raise ValueError("sampled source contains non-finite values")
    calibration = 2 * sum(len(_parameter_grid(spec)) for spec in config["design"]["objective_grid"].values())
    promoted_parameters = int(config["design"]["staging"]["parameters_promoted_per_objective"])
    expanded_multi = len(config["design"]["objective_grid"]) * promoted_parameters * len(config["design"]["multilag_profiles"]) * len(config["design"]["lag_weight_profiles"])
    expanded_embedding = len(config["design"]["objective_grid"]) * promoted_parameters * len(config["design"]["embedding_profiles"])
    frozen_configs = len(config["design"]["objective_grid"]) * 2 * int(config["design"]["promoted_per_objective_per_formulation"])
    frozen_lanes = frozen_configs * 2 + (len(config["design"]["objective_grid"]) * int(config["design"]["promoted_per_objective_per_formulation"]))
    pipeline_lanes = frozen_lanes * len(_contexts(config))
    disk = shutil.disk_usage(root.parent)
    if disk.free < 20 * 2**30:
        raise RuntimeError("less than 20 GiB disk headroom")
    root.mkdir(parents=True, exist_ok=False)
    fingerprints = {
        "movie": {"sha256": sha256_file(Path(config["source"]["movie_path"])), "shape": list(movie.shape), "dtype": str(movie.dtype)},
        "labels": {"sha256": sha256_file(Path(config["source"]["labels_path"])), "rows": len(labels)},
        "cascade_v4_status": sha256_file(Path(config["source"]["cascade_v4_root"]) / "status.json"),
    }
    fingerprint = sha256_payload({"config": _resolved(config), "inputs": fingerprints})
    atomic_json(root / "config.resolved.json", _resolved(config))
    atomic_json(root / "resource_plan.json", {
        "parameter_calibration_fits": calibration,
        "expanded_multilag_fits": expanded_multi,
        "expanded_embedding_fits": expanded_embedding,
        "expanded_total_fits": expanded_multi + expanded_embedding,
        "frozen_configuration_count": frozen_configs,
        "frozen_raw_lane_count": frozen_lanes,
        "full_msln_context_count": len(_contexts(config)),
        "pipeline_lane_count": pipeline_lanes,
        "workers_per_gpu": 1,
        "cpu_threads": int(config["compute"]["cpu_threads"]),
        "projection_frame_chunk": int(config["compute"]["projection_frame_chunk"]),
        "full_maps_retained_for_finalists_only": True,
        "disk_free_bytes": disk.free,
    })
    atomic_json(root / "preflight.json", {
        "ready": True,
        "fingerprint": fingerprint,
        "input_fingerprints": fingerprints,
        "source_read_only": True,
        "global_msica_history_frames": global_history,
        "msln_pre_roll_frames": msln_pre_roll,
        "review_frames": len(quiet),
        "labels_used_for_selection": False,
        "labels_used_preflight_for": "coordinate bounds and projection overlay only",
        "label_scoring_stage": "finalize_only",
    })
    atomic_json(root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    _label_overlay(root, movie, labels, _adapt(config))
    return {"ready": True, "root": str(root), "fingerprint": fingerprint}


def gpu_preflight(config_path: str | Path) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path)
    root = _root(config)
    if not (root / "preflight.json").is_file():
        raise RuntimeError("read-only preflight is required")
    device = cuda_device_summary()
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    if int(device["free_bytes"]) < cap:
        raise RuntimeError("free GPU memory is below the frozen cap")
    rng = np.random.default_rng(31)
    tiny = rng.normal(size=(70, 13, 17)).astype(np.float32)
    anchors = sample_anchor_indices(tiny.shape, history=9, count=128, seed=2)
    embedded = gather_delay_embedding(tiny, anchors, (0, 1, 2, 4, 8))
    fit = fit_delay_embedding(
        embedded[:, :64], embedded[:, 64:], lags=(0, 1, 2, 4, 8),
        objective_family="ksg_mi", objective_parameter={"neighbors": 3},
        angle_step_degrees=30.0, max_sweeps=2,
    )
    cpu = project_temporal_fit_chunked(tiny, fit, backend="cpu", frame_chunk=7)
    gpu = project_temporal_fit_chunked(tiny, fit, backend="cuda", frame_chunk=7)
    parity = {name: float(np.max(np.abs(cpu[name] - gpu[name]))) for name in cpu}
    if max(parity.values()) > 2e-5:
        raise RuntimeError("chunked CUDA projection parity failed")
    context = JointSTContext("v5_gpu_smoke", 7, 3, 9, 1, "mean_std", 10.0)
    quiet = np.arange(len(tiny)) < 35
    cpu_msln = causal_joint_msln(tiny, context, quiet_mask=np.where(np.arange(len(tiny)) >= 9, quiet, False))
    gpu_msln = causal_joint_msln_cuda(tiny, context, quiet_mask=quiet, review_crop_frames=9, max_vram_bytes=min(cap, 2**30))
    msln_error = float(np.max(np.abs(cpu_msln.values[9:] - cp.asnumpy(gpu_msln.values))))
    if msln_error > 2e-5:
        raise RuntimeError("CUDA MSLN parity failed")
    payload = {"ready": True, "device": device, "projection_max_abs_error": parity, "msln_max_abs_error": msln_error, "max_vram_bytes": cap}
    atomic_json(root / "gpu_validation.json", payload)
    del gpu_msln
    cp.get_default_memory_pool().free_all_blocks()
    return payload


def _require_run(config: dict[str, Any], authorize_full_spon: bool) -> Path:
    if not authorize_full_spon:
        raise PermissionError("full Spon run requires --authorize-full-spon")
    root = _root(config)
    if not (root / "preflight.json").is_file() or not (root / "gpu_validation.json").is_file():
        raise RuntimeError("matching read-only and CUDA preflights are required")
    return root


def run_surface(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon)
    surface_path = root / "stage_a" / "surface.json"
    if surface_path.exists():
        if resume:
            return json.loads(surface_path.read_text(encoding="utf-8"))
        raise FileExistsError(surface_path)
    atomic_json(root / "status.json", {"status": "running", "stage": "objective_surface"})
    values, quiet, global_history, msln_pre_roll = _source(config)
    design = config["design"]
    staging = design["staging"]
    calibration_rows = []
    for objective_index, (objective, spec) in enumerate(design["objective_grid"].items()):
        for formulation_index, formulation in enumerate(FORMULATIONS):
            profile_name = staging[f"parameter_calibration_{'multilag' if formulation == 'multilag_2d' else 'embedding'}_profile"]
            profiles = design["multilag_profiles"] if formulation == "multilag_2d" else design["embedding_profiles"]
            lags = tuple(profiles[profile_name])
            decay = float(design["lag_weight_profiles"][staging["parameter_calibration_weight_profile"]])
            for parameter_index, parameter in enumerate(_parameter_grid(spec)):
                fit = _sample_fit(
                    values, config, formulation=formulation, lags=lags, objective=objective,
                    parameter=parameter, weight_decay=decay,
                    seed_offset=100000 * objective_index + 10000 * formulation_index + 10 * parameter_index,
                )
                calibration_rows.append({
                    "formulation": formulation, "objective_family": objective,
                    "profile": profile_name, "parameter": parameter,
                    "held_out_gain_fraction": _objective_gain(fit),
                    "objective": fit.objective, "baseline_objective": fit.baseline_objective,
                    "converged": fit.converged, "fit": fit.to_dict(),
                })
                _heartbeat(root, "parameter_calibration", completed=len(calibration_rows), total=2 * sum(len(_parameter_grid(item)) for item in design["objective_grid"].values()))
    promoted_parameters: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    count = int(staging["parameters_promoted_per_objective"])
    for objective in design["objective_grid"]:
        promoted_parameters[objective] = {}
        for formulation in FORMULATIONS:
            candidates = [row for row in calibration_rows if row["objective_family"] == objective and row["formulation"] == formulation]
            candidates.sort(key=lambda row: (-row["held_out_gain_fraction"], not row["converged"], _parameter_key(row["parameter"])))
            promoted_parameters[objective][formulation] = [row["parameter"] for row in candidates[:count]]
    expansion_rows = []
    total_expanded = (
        len(design["objective_grid"]) * count * len(design["multilag_profiles"]) * len(design["lag_weight_profiles"])
        + len(design["objective_grid"]) * count * len(design["embedding_profiles"])
    )
    for objective_index, objective in enumerate(design["objective_grid"]):
        for formulation_index, formulation in enumerate(FORMULATIONS):
            profiles = design["multilag_profiles"] if formulation == "multilag_2d" else design["embedding_profiles"]
            weights = design["lag_weight_profiles"] if formulation == "multilag_2d" else {"none": 0.0}
            for parameter_index, parameter in enumerate(promoted_parameters[objective][formulation]):
                for profile_index, (profile, lag_list) in enumerate(profiles.items()):
                    for weight_index, (weight_name, decay) in enumerate(weights.items()):
                        fit = _sample_fit(
                            values, config, formulation=formulation, lags=tuple(lag_list),
                            objective=objective, parameter=parameter, weight_decay=float(decay),
                            seed_offset=500000 + objective_index * 100000 + formulation_index * 30000 + parameter_index * 3000 + profile_index * 100 + weight_index,
                        )
                        config_id = _config_id(formulation, objective, profile, parameter, None if formulation == "delay_embedding" else weight_name)
                        outputs = _aligned_outputs(values, fit, config, global_history, msln_pre_roll)
                        branches = {}
                        for branch in BRANCHES[formulation]:
                            branches[branch] = _label_free_artifacts(outputs[branch][msln_pre_roll:], quiet, config)
                        expansion_rows.append({
                            "config_id": config_id, "formulation": formulation,
                            "objective_family": objective, "profile": profile,
                            "weight_profile": None if formulation == "delay_embedding" else weight_name,
                            "parameter": parameter, "fit": fit.to_dict(),
                            "held_out_gain_fraction": _objective_gain(fit), "branches": branches,
                        })
                        del outputs, fit
                        cp.get_default_memory_pool().free_all_blocks()
                        gc.collect()
                        _heartbeat(root, "expanded_surface", completed=len(expansion_rows), total=total_expanded, config_id=config_id)
                        print(f"SURFACE {len(expansion_rows)}/{total_expanded} {config_id}", flush=True)
    promoted_configs = []
    per_group = int(design["promoted_per_objective_per_formulation"])
    for objective in design["objective_grid"]:
        for formulation in FORMULATIONS:
            candidates = [row for row in expansion_rows if row["objective_family"] == objective and row["formulation"] == formulation]
            eligible = [row for row in candidates if row["held_out_gain_fraction"] > 0.0]
            candidates = eligible or candidates
            candidates.sort(key=lambda row: (-max(item["selection_score"] for item in row["branches"].values()), -row["held_out_gain_fraction"], row["config_id"]))
            promoted_configs.extend(row["config_id"] for row in candidates[:per_group])
    payload = {
        "complete": True,
        "selection_labels_used": False,
        "calibration_rows": calibration_rows,
        "promoted_parameters": promoted_parameters,
        "expansion_rows": expansion_rows,
        "promoted_config_ids": promoted_configs,
        "promotion_rule": "positive held-out objective gain required when available; then top label-free event/quiet score within each objective x formulation",
    }
    atomic_json(surface_path, payload)
    atomic_json(root / "stage_a" / "frozen_raw_panel.json", {
        "complete": True, "selection_labels_used": False,
        "config_ids": promoted_configs,
        "lane_ids": [f"{config_id}::{branch}" for config_id in promoted_configs for branch in BRANCHES[next(row["formulation"] for row in expansion_rows if row["config_id"] == config_id)]],
    })
    atomic_json(root / "status.json", {"status": "surface_complete", "scientific_status": "labels_sealed"})
    return payload


def _promoted_config_ids(root: Path, surface: dict[str, Any]) -> list[str]:
    amendment = root / "stage_a" / "promotion_amendment.json"
    if amendment.is_file():
        return list(json.loads(amendment.read_text(encoding="utf-8"))["promoted_config_ids"])
    return list(surface["promoted_config_ids"])


def run_pipeline(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon)
    surface = json.loads((root / "stage_a" / "surface.json").read_text(encoding="utf-8"))
    promoted = set(_promoted_config_ids(root, surface))
    selected_rows = [row for row in surface["expansion_rows"] if row["config_id"] in promoted]
    values, quiet, global_history, msln_pre_roll = _source(config)
    quiet_extended = np.concatenate((np.zeros(msln_pre_roll, dtype=bool), quiet))
    contexts = _contexts(config)
    shard_root = root / "stage_b" / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    total = sum(len(BRANCHES[row["formulation"]]) * len(contexts) for row in selected_rows)
    completed = 0
    for config_index, row in enumerate(selected_rows):
        shard_path = shard_root / f"{row['config_id']}.json"
        if resume and shard_path.is_file() and json.loads(shard_path.read_text()).get("complete"):
            completed += len(BRANCHES[row["formulation"]]) * len(contexts)
            continue
        fit = _fit_from_dict(row["fit"])
        outputs = _aligned_outputs(values, fit, config, global_history, msln_pre_roll)
        shard_rows = []
        for branch in BRANCHES[row["formulation"]]:
            branch_values = outputs[branch]
            for context in contexts:
                tick = time.monotonic()
                normalized = causal_joint_msln_cuda(
                    branch_values, context, quiet_mask=quiet_extended,
                    review_crop_frames=msln_pre_roll, max_vram_bytes=cap,
                )
                metrics = _label_free_artifacts(normalized.values, quiet, config)
                lane_id = f"{row['config_id']}::{branch}::{context.context_id}"
                shard_rows.append({
                    "lane_id": lane_id, "config_id": row["config_id"], "formulation": row["formulation"],
                    "objective_family": row["objective_family"], "branch": branch,
                    "context_id": context.context_id, "metrics": metrics,
                    "msln_scale_floor": normalized.scale_floor,
                    "msln_diagnostics": normalized.diagnostics,
                    "runtime_seconds": time.monotonic() - tick,
                })
                completed += 1
                _heartbeat(root, "pipeline_screen", completed=completed, total=total, lane_id=lane_id)
                del normalized
                cp.get_default_memory_pool().free_all_blocks()
            print(f"PIPELINE {config_index + 1}/{len(selected_rows)} {row['config_id']} {branch}", flush=True)
        atomic_json(shard_path, {"complete": True, "selection_labels_used": False, "rows": shard_rows})
        del outputs
        gc.collect()
    rows = []
    for path in sorted(shard_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise RuntimeError(f"incomplete shard: {path}")
        rows.extend(payload["rows"])
    if len(rows) != total:
        raise RuntimeError(f"expected {total} pipeline lanes, found {len(rows)}")
    rows.sort(key=lambda row: (-row["metrics"]["selection_score"], row["lane_id"]))
    finalists = [row["lane_id"] for row in rows[:int(config["design"]["pipeline_finalists"])]]
    family_champions = []
    for objective in config["design"]["objective_grid"]:
        for formulation in FORMULATIONS:
            candidates = [row for row in rows if row["objective_family"] == objective and row["formulation"] == formulation]
            family_champions.append(candidates[0]["lane_id"])
    atomic_json(root / "stage_b" / "all_pipeline_metrics.json", {
        "complete": True, "selection_labels_used": False, "rows": rows,
        "global_finalist_lane_ids": finalists,
        "family_champion_lane_ids": family_champions,
    })
    atomic_json(root / "status.json", {"status": "pipeline_complete", "scientific_status": "labels_sealed"})
    return {"status": "pipeline_complete", "lanes": len(rows), "global_finalists": finalists, "family_champions": family_champions}


def finalize(config_path: str | Path, *, authorize_label_open: bool) -> dict[str, Any]:
    config = _load(config_path)
    root = _root(config)
    if not authorize_label_open:
        raise PermissionError("protected evaluation requires --authorize-label-open")
    surface = json.loads((root / "stage_a" / "surface.json").read_text(encoding="utf-8"))
    pipeline = json.loads((root / "stage_b" / "all_pipeline_metrics.json").read_text(encoding="utf-8"))
    labels = label_core.load_labels(Path(config["source"]["labels_path"]))
    promoted = set(_promoted_config_ids(root, surface))
    raw_rows = []
    for row in surface["expansion_rows"]:
        if row["config_id"] not in promoted:
            continue
        for branch, artifacts in row["branches"].items():
            raw_rows.append({
                "lane_id": f"{row['config_id']}::{branch}", "config_id": row["config_id"],
                "formulation": row["formulation"], "objective_family": row["objective_family"],
                "branch": branch, "selection_score": artifacts["selection_score"],
                "protected": _protected_from_proposals(artifacts["proposals"], labels, config),
            })
    pipeline_rows = []
    for row in pipeline["rows"]:
        pipeline_rows.append({
            "lane_id": row["lane_id"], "config_id": row["config_id"],
            "formulation": row["formulation"], "objective_family": row["objective_family"],
            "branch": row["branch"], "context_id": row["context_id"],
            "selection_score": row["metrics"]["selection_score"],
            "protected": _protected_from_proposals(row["metrics"]["proposals"], labels, config),
        })
    guardrail = str(config["evaluation"]["guardrail_budget"])
    raw_best_label_free = max(raw_rows, key=lambda row: row["selection_score"])
    pipeline_best_label_free = max(pipeline_rows, key=lambda row: row["selection_score"])
    raw_best_protected = max(raw_rows, key=lambda row: row["protected"]["matched_by_budget"][guardrail])
    pipeline_best_protected = max(pipeline_rows, key=lambda row: row["protected"]["matched_by_budget"][guardrail])
    result = {
        "complete": True,
        "selection_labels_used": False,
        "protected_scoring_after_freeze": True,
        "raw_msica": raw_rows,
        "raw_msica_then_msln": pipeline_rows,
        "summary": {
            "guardrail_budget": int(guardrail),
            "raw_best_label_free_lane": raw_best_label_free["lane_id"],
            "raw_best_label_free_matches": raw_best_label_free["protected"]["matched_by_budget"][guardrail],
            "raw_protected_ceiling_lane": raw_best_protected["lane_id"],
            "raw_protected_ceiling_matches": raw_best_protected["protected"]["matched_by_budget"][guardrail],
            "pipeline_best_label_free_lane": pipeline_best_label_free["lane_id"],
            "pipeline_best_label_free_matches": pipeline_best_label_free["protected"]["matched_by_budget"][guardrail],
            "pipeline_protected_ceiling_lane": pipeline_best_protected["lane_id"],
            "pipeline_protected_ceiling_matches": pipeline_best_protected["protected"]["matched_by_budget"][guardrail],
            "raw_direct_reference_matches": 49,
            "raw_direct_reference_total_labels": 79,
            "interpretation": "Protected ceilings are descriptive and are not deployable selections.",
        },
    }
    atomic_json(root / "stage_c" / "protected_results.json", result)
    atomic_json(root / "status.json", {"status": "complete", "scientific_status": "protected_evaluation_complete"})
    return result


def _synthetic_fixture(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames, pixels = 320, 64
    common = np.sin(np.linspace(0, 7 * np.pi, frames))[:, None]
    gains = rng.uniform(0.7, 1.3, size=(1, pixels))
    local = np.zeros((frames, pixels), dtype=np.float64)
    noise = rng.normal(size=(frames, pixels))
    for frame in range(1, frames):
        local[frame] = 0.96 * local[frame - 1] + 0.25 * noise[frame]
    background = 1.8 * common * gains + local
    neural = np.zeros((frames, pixels), dtype=np.float64)
    for pixel in range(pixels):
        for frame in rng.choice(np.arange(30, frames - 20), size=3, replace=False):
            neural[frame:frame + 10, pixel] += np.exp(-np.arange(10) / 2.5) * rng.uniform(2.0, 4.0)
    movie = 1000.0 + 20.0 * background + 8.0 * neural + rng.normal(0, 2.0, size=(frames, pixels))
    return movie[:, None, :].astype(np.float32), background[:, None, :].astype(np.float32), neural[:, None, :].astype(np.float32)


def confirm_panel(config_path: str | Path, *, authorize_full_spon: bool) -> dict[str, Any]:
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon)
    surface = json.loads((root / "stage_a" / "surface.json").read_text(encoding="utf-8"))
    promoted = set(_promoted_config_ids(root, surface))
    selected = [row for row in surface["expansion_rows"] if row["config_id"] in promoted]
    values, _, _, _ = _source(config)
    seed_count = int(config["fitting"]["projection_seeds"])
    synthetic_count = int(config["fitting"]["synthetic_seeds"])
    rows = []
    for index, row in enumerate(selected):
        formulation = row["formulation"]
        profiles = config["design"]["multilag_profiles"] if formulation == "multilag_2d" else config["design"]["embedding_profiles"]
        lags = tuple(profiles[row["profile"]])
        decay = 0.0 if formulation == "delay_embedding" else float(config["design"]["lag_weight_profiles"][row["weight_profile"]])
        reference = _fit_from_dict(row["fit"])
        real_rows = []
        for seed_index in range(seed_count):
            fit = _sample_fit(
                values, config, formulation=formulation, lags=lags,
                objective=row["objective_family"], parameter=row["parameter"],
                weight_decay=decay, seed_offset=2_000_000 + index * 1000 + seed_index * 10,
            )
            ref_p = reference.demixing[reference.persistence_index]
            fit_p = fit.demixing[fit.persistence_index]
            ref_i = reference.demixing[reference.innovation_index]
            fit_i = fit.demixing[fit.innovation_index]
            real_rows.append({
                "seed_index": seed_index,
                "held_out_gain_fraction": _objective_gain(fit),
                "persistence_demixing_cosine": float(abs(np.dot(ref_p, fit_p)) / max(np.linalg.norm(ref_p) * np.linalg.norm(fit_p), 1e-12)),
                "innovation_demixing_cosine": float(abs(np.dot(ref_i, fit_i)) / max(np.linalg.norm(ref_i) * np.linalg.norm(fit_i), 1e-12)),
                "converged": fit.converged,
            })
        synthetic_rows = []
        for seed_index in range(synthetic_count):
            fixture, background, neural = _synthetic_fixture(int(config["design"]["seed"]) + 100 * index + seed_index)
            fit = _sample_fit(
                fixture, config, formulation=formulation, lags=lags,
                objective=row["objective_family"], parameter=row["parameter"],
                weight_decay=decay, seed_offset=3_000_000 + index * 1000 + seed_index * 10,
            )
            projected = project_temporal_fit_chunked(fixture, fit, backend="cpu", frame_chunk=16)
            history = max(lags) if formulation == "delay_embedding" else 1
            persistence_target = background[history:].ravel()
            innovation_target = np.diff(neural, axis=0)[history - 1:].ravel()
            persistence_values = projected["persistence"].ravel()
            innovation_values = projected["innovation"].ravel()
            synthetic_rows.append({
                "seed_index": seed_index,
                "held_out_gain_fraction": _objective_gain(fit),
                "persistence_background_abs_correlation": float(abs(np.corrcoef(persistence_values, persistence_target)[0, 1])),
                "innovation_neural_derivative_abs_correlation": float(abs(np.corrcoef(innovation_values, innovation_target)[0, 1])),
            })
        gains = [item["held_out_gain_fraction"] for item in real_rows]
        rows.append({
            "config_id": row["config_id"], "formulation": formulation,
            "objective_family": row["objective_family"],
            "real_seed_rows": real_rows, "synthetic_seed_rows": synthetic_rows,
            "real_median_held_out_gain": float(np.median(gains)),
            "real_positive_gain_fraction": float(np.mean(np.asarray(gains) > 0)),
            "median_persistence_demixing_cosine": float(np.median([item["persistence_demixing_cosine"] for item in real_rows])),
            "median_innovation_demixing_cosine": float(np.median([item["innovation_demixing_cosine"] for item in real_rows])),
            "synthetic_median_persistence_correlation": float(np.median([item["persistence_background_abs_correlation"] for item in synthetic_rows])),
            "synthetic_median_innovation_correlation": float(np.median([item["innovation_neural_derivative_abs_correlation"] for item in synthetic_rows])),
        })
        _heartbeat(root, "panel_confirmation", completed=len(rows), total=len(selected), config_id=row["config_id"])
        print(f"CONFIRM {len(rows)}/{len(selected)} {row['config_id']}", flush=True)
    payload = {
        "complete": True, "selection_labels_used": False,
        "real_projection_seeds": seed_count, "synthetic_seeds": synthetic_count,
        "rows": rows,
        "qualification_rule": "diagnostic only: report median held-out gain, positive-gain fraction, demixing consistency, and synthetic recovery; retain the frozen objective-diverse panel",
    }
    atomic_json(root / "stage_a" / "panel_confirmation.json", payload)
    return payload



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "gpu-preflight", "surface", "confirm", "pipeline", "finalize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-full-spon", action="store_true")
    parser.add_argument("--authorize-label-open", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "preflight":
        result = preflight(args.config)
    elif args.action == "gpu-preflight":
        result = gpu_preflight(args.config)
    elif args.action == "surface":
        result = run_surface(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    elif args.action == "confirm":
        result = confirm_panel(args.config, authorize_full_spon=args.authorize_full_spon)
    elif args.action == "pipeline":
        result = run_pipeline(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    else:
        result = finalize(args.config, authorize_label_open=args.authorize_label_open)
    print(json.dumps(result if args.action != "surface" else {"complete": result["complete"], "promoted_config_ids": result["promoted_config_ids"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
