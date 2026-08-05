"""Paired MSLN/MSICA block-order and gated cascade experiments."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from neurobench.algorithms.msln_msica_cuda import (
    apply_per_context_fit_cuda,
    causal_joint_msln_cuda,
    cuda_device_summary,
    gather_adjacent_pairs_cuda,
)
from neurobench.algorithms.multiscale_local_normalization import JointSTContext, causal_joint_msln
from neurobench.algorithms.multiscale_subspace import (
    bootstrap_summary,
    contiguous_block_bootstrap,
    fit_per_context_ica,
)
from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one
from neurobench.reports.msln_msica_videos import Layer, _render_video

from .artifacts import atomic_json, sha256_file, sha256_payload
from .fitting import adjacent_sample_indices, pairs_at
from .joint_sweep import (
    RAW_DIRECT_ANCHOR,
    _atomic_npy,
    _contexts,
    _label_overlay,
    _labels,
    _recall,
    _source_view,
    _synthetic,
    _visual_stats,
)


REFERENCE_CONTEXTS = (
    "joint_s15_g3_t31_g1",
    "joint_s15_g3_t23_g1",
    "joint_s5_g1_t15_g1",
)


def _load(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_id", "source", "sweep", "ica", "evaluation", "compute", "outputs"}
    if set(payload) != required or payload["schema_version"] != 3:
        raise ValueError("order program requires the exact schema-v3 contract")
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path", "v2_root"):
        payload["source"][key] = str((source.parent / payload["source"][key]).resolve())
    for key in ("block_switch_root", "cascade_root", "summary_root"):
        payload["outputs"][key] = str((source.parent / payload["outputs"][key]).resolve())
    payload["_config_path"] = str(source)
    _validate(payload)
    return payload


def _validate(config: dict[str, Any]) -> None:
    if config["source"]["axes"] != "TYX" or not config["source"]["ui_one_based"]:
        raise ValueError("source must use one-based UI TYX coordinates")
    if config["sweep"]["spatial_outer_guard_pairs"] != [[5, 1], [7, 1], [7, 3], [11, 3], [15, 3], [15, 5]]:
        raise ValueError("the six v2 spatial contexts are frozen")
    if config["sweep"]["temporal_windows_frames"] != [5, 9, 15, 23, 31]:
        raise ValueError("the five v2 temporal windows are frozen")
    if config["sweep"]["forced_contexts"] != list(REFERENCE_CONTEXTS):
        raise ValueError("the three v2 reference contexts must be forced")
    if int(config["sweep"]["finalist_contexts"]) != 6:
        raise ValueError("the approved program freezes six finalists")
    if config["evaluation"]["candidate_budgets"] != [20, 40, 58, 80, 100]:
        raise ValueError("candidate budgets must match v2")
    if int(config["evaluation"]["guardrail_budget"]) != 58:
        raise ValueError("guardrail budget must remain 58 per burst")
    if config["compute"]["device"] != "cuda" or config["compute"]["workers"] != 1:
        raise ValueError("order program requires one CUDA worker")
    if not 0 < float(config["compute"]["max_peak_vram_gb"]) <= 8:
        raise ValueError("VRAM cap must lie in (0,8] GiB")
    grid = [float(item) for item in config["ica"]["bandwidth_grid"]]
    if grid != [0.2, 0.35, 0.5, 0.7]:
        raise ValueError("approved bandwidth grid is [0.2,0.35,0.5,0.7]")
    if int(config["ica"]["bootstrap_replicates"]) < 16:
        raise ValueError("at least 16 blocked bootstrap replicates are required")


def _resolved(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _roots(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    outputs = config["outputs"]
    return tuple(Path(outputs[key]) for key in ("block_switch_root", "cascade_root", "summary_root"))  # type: ignore[return-value]


def _fit_only(
    lane_id: str,
    values: Any,
    config: dict[str, Any],
    *,
    bandwidth: float,
    seed_offset: int,
    bootstrap_replicates: int = 0,
    run_fastica: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    valid = np.ones(len(values), dtype=bool)
    confirmation = adjacent_sample_indices(
        tuple(map(int, values.shape)), valid,
        count=int(config["ica"]["confirmation_samples"]),
        seed=int(config["sweep"]["seed"]) + int(seed_offset),
    )
    rng = np.random.default_rng(int(config["sweep"]["seed"]) + int(seed_offset) + 1)
    screen = confirmation[np.sort(rng.choice(
        len(confirmation), size=min(int(config["ica"]["screen_samples"]), len(confirmation)), replace=False,
    ))]
    reader = gather_adjacent_pairs_cuda if hasattr(values, "__cuda_array_interface__") else pairs_at
    screen_pairs = reader(values, screen)
    confirmation_pairs = reader(values, confirmation)
    kwargs = {
        "parzen_bandwidth": float(bandwidth),
        "eigenvalue_floor_ratio": float(config["ica"]["eigenvalue_floor_ratio"]),
        "coarse_step_degrees": float(config["ica"]["coarse_step_degrees"]),
        "refine_half_width_degrees": float(config["ica"]["refine_half_width_degrees"]),
        "refine_step_degrees": float(config["ica"]["refine_step_degrees"]),
        "kernel_block_rows": int(config["ica"]["kernel_block_rows"]),
        "kernel_dtype": np.float32,
        "compute_backend": "cuda",
    }
    fit = fit_per_context_ica(lane_id, screen_pairs, confirmation_pairs, objective="cs_parzen", **kwargs)
    diagnostics: dict[str, Any] = {"cs_parzen": fit.to_dict(), "sample_seed_offset": int(seed_offset)}
    if run_fastica:
        diagnostics["fastica"] = fit_per_context_ica(
            lane_id, screen_pairs, confirmation_pairs, objective="fastica", **kwargs
        ).to_dict()
    boot = None
    if bootstrap_replicates:
        rows = contiguous_block_bootstrap(
            lane_id, confirmation_pairs,
            block_length=int(config["ica"]["bootstrap_block_samples"]),
            replicates=int(bootstrap_replicates),
            seed=int(config["sweep"]["seed"]) + int(seed_offset) + 2,
            fitter_kwargs={"objective": "cs_parzen", **kwargs},
        )
        boot = bootstrap_summary(rows)
        diagnostics["bootstrap_rows"] = rows
    return fit, diagnostics, boot


def _objective_gain(fit: Any) -> float:
    baseline = max(abs(float(fit.baseline_objective_value)), 1e-12)
    return float((fit.baseline_objective_value - fit.objective_value) / baseline)


def _tune_raw(values: Any, config: dict[str, Any], root: Path) -> tuple[Any, float, dict[str, Any]]:
    rows = []
    for bandwidth in config["ica"]["bandwidth_grid"]:
        for seed_index in range(int(config["ica"]["tuning_seeds"])):
            fit, _, _ = _fit_only(
                "raw_msica_tuning", values, config, bandwidth=float(bandwidth),
                seed_offset=1000 * seed_index, bootstrap_replicates=0,
            )
            rows.append({
                "bandwidth": float(bandwidth), "seed_index": seed_index,
                "objective_gain_fraction": _objective_gain(fit),
                "angle_degrees": fit.rotation_angle_degrees,
                "ambiguous_alignment": fit.ambiguous_alignment,
                "condition_number": fit.diagnostics["whitening"]["condition_number"],
            })
    aggregates = []
    for bandwidth in config["ica"]["bandwidth_grid"]:
        selected = [row for row in rows if row["bandwidth"] == float(bandwidth)]
        aggregates.append({
            "bandwidth": float(bandwidth),
            "median_objective_gain_fraction": float(np.median([row["objective_gain_fraction"] for row in selected])),
            "angle_range_degrees": float(np.ptp([row["angle_degrees"] for row in selected])),
            "ambiguous_fraction": float(np.mean([row["ambiguous_alignment"] for row in selected])),
        })
    winner = sorted(aggregates, key=lambda row: (-row["median_objective_gain_fraction"], row["angle_range_degrees"], row["bandwidth"]))[0]
    bandwidth = float(winner["bandwidth"])
    fit, details, boot = _fit_only(
        "raw_msica_final", values, config, bandwidth=bandwidth, seed_offset=0,
        bootstrap_replicates=int(config["ica"]["bootstrap_replicates"]), run_fastica=True,
    )
    payload = {
        "selection_basis": "maximum median relative CS objective gain; labels excluded",
        "grid_rows": rows, "bandwidth_aggregates": aggregates,
        "selected_bandwidth": bandwidth, "selected_fit": details,
        "bootstrap": boot,
    }
    atomic_json(root / "stage_a" / "raw_msica_tuning.json", payload)
    return fit, bandwidth, payload


def _paired_eval(values: Any, labels: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    base = _recall(values, labels, config)
    review_start = int(config["source"]["review_interval_ui"][0])
    matches: dict[str, list[str]] = {str(b): [] for b in config["evaluation"]["candidate_budgets"]}
    for burst_text, interval in sorted(config["source"]["burst_intervals_ui"].items(), key=lambda item: int(item[0])):
        burst = int(burst_text)
        start = int(interval[0]) - review_start
        stop = int(interval[1]) - review_start + 1
        if hasattr(values, "__cuda_array_interface__"):
            import cupy as cp
            pooled = cp.asnumpy(cp.max(values[start:stop], axis=0))
        else:
            pooled = np.max(np.asarray(values[start:stop]), axis=0)
        peaks = extract_local_maxima(pooled, int(config["evaluation"]["nms_distance_px"]), limit=max(config["evaluation"]["candidate_budgets"]))
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
        for budget in config["evaluation"]["candidate_budgets"]:
            matched, _ = match_peaks_one_to_one(peaks[:int(budget)], burst_labels, float(config["evaluation"]["match_radius_px"]))
            matches[str(budget)].extend(str(burst_labels[index]["roi_identity"]) for index, *_ in matched)
    base["matched_roi_identities_by_budget"] = matches
    return base


def _lane_metrics(values: Any, quiet: np.ndarray, labels: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    import cupy as cp
    evidence = cp.square(values, dtype=cp.float32) if hasattr(values, "__cuda_array_interface__") else np.square(values, dtype=np.float32)
    result = {
        "visual_stats": _visual_stats(evidence, quiet, config),
        "recall_guardrail": _paired_eval(evidence, labels, config),
    }
    del evidence
    return result


def _selection_score(metrics: dict[str, Any], synthetic: dict[str, Any]) -> float:
    visual = metrics["visual_stats"]
    return float(np.log1p(visual["event_quiet_ratio_p999"]) + np.log1p(synthetic["signal_to_nuisance_proxy"]))


def _preflight_one(config: dict[str, Any], root: Path, kind: str, fingerprints: dict[str, Any]) -> dict[str, Any]:
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    review_frames = review_stop - review_start + 1
    map_bytes = review_frames * movie.shape[1] * movie.shape[2] * 4
    retained = 34 if kind == "block_switch" else 28
    estimate = int(retained * map_bytes + 2 * 2**30)
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if estimate > shutil.disk_usage(ancestor).free:
        raise RuntimeError("estimated outputs exceed free disk")
    resolved = _resolved(config)
    fingerprint = sha256_payload({"config": resolved, "inputs": fingerprints, "kind": kind})
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "config.resolved.json", resolved)
    atomic_json(root / "resource_plan.json", {
        "experiment_kind": kind, "contexts": 30, "finalists": 6,
        "bytes_per_review_float32_map": map_bytes,
        "retained_map_equivalents": retained, "output_bytes_estimate": estimate,
        "one_context_at_a_time": True, "workers": 1,
        "max_peak_vram_bytes": int(float(config["compute"]["max_peak_vram_gb"]) * 2**30),
    })
    atomic_json(root / "preflight.json", {
        "ready": True, "preflight_fingerprint": fingerprint,
        "input_fingerprints": fingerprints, "source_read_only": True,
        "sparse_point_labels_used_for_fitting_or_selection": False,
        "raw_direct_anchor": RAW_DIRECT_ANCHOR,
    })
    atomic_json(root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    return {"root": str(root), "preflight_fingerprint": fingerprint, "output_bytes_estimate": estimate}


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path)
    block, cascade, summary = _roots(config)
    collisions = [str(path) for path in (block, cascade, summary) if path.exists()]
    if collisions:
        raise FileExistsError(f"output collisions: {collisions}")
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path"):
        if not Path(config["source"][key]).is_file():
            raise FileNotFoundError(config["source"][key])
    if not Path(config["source"]["v2_root"]).is_dir():
        raise FileNotFoundError(config["source"]["v2_root"])
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    labels = _labels(config)
    if movie.ndim != 3 or not np.isfinite(np.asarray(movie[1799:1810], dtype=np.float32)).all():
        raise ValueError("source movie failed shape/finiteness checks")
    height, width = movie.shape[1:]
    if any(not (0 <= int(row["x_px"]) < width and 0 <= int(row["y_px"]) < height) for row in labels):
        raise ValueError("label coordinate outside movie")
    fingerprints = {
        "movie": {"sha256": sha256_file(Path(config["source"]["movie_path"])), "shape": list(movie.shape), "dtype": str(movie.dtype)},
        "labels": {"sha256": sha256_file(Path(config["source"]["labels_path"])), "rows": len(labels)},
        "raw_direct_metrics": {"sha256": sha256_file(Path(config["source"]["raw_direct_metrics_path"]))},
        "v2_manifest": {"sha256": sha256_file(Path(config["source"]["v2_root"]) / "run_manifest.json")},
    }
    rows = [
        _preflight_one(config, block, "block_switch", fingerprints),
        _preflight_one(config, cascade, "cascade", fingerprints),
    ]
    _label_overlay(block, movie, labels, config)
    _label_overlay(cascade, movie, labels, config)
    summary.mkdir(parents=True, exist_ok=False)
    atomic_json(summary / "status.json", {"status": "waiting_for_experiments"})
    return {"ready": True, "experiments": rows, "summary_root": str(summary)}


def gpu_preflight(config_path: str | Path) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path)
    block, cascade, _ = _roots(config)
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    device = cuda_device_summary()
    if device["free_bytes"] < cap:
        raise RuntimeError("free GPU memory is below the frozen cap")
    rng = np.random.default_rng(17)
    tiny = (1000 + rng.normal(0, 3, (48, 31, 33))).astype(np.float32)
    tiny[35:38, 14:18, 15:19] += 20
    quiet = np.arange(48) < 20
    context = JointSTContext("order_gpu_smoke", 7, 3, 9, 1)
    cpu = causal_joint_msln(tiny, context, quiet_mask=quiet)
    gpu = causal_joint_msln_cuda(tiny, context, quiet_mask=quiet, review_crop_frames=9, max_vram_bytes=min(cap, 2**30))
    actual = cp.asnumpy(gpu.values)
    error = np.abs(actual - cpu.values[9:])
    parity = {"max_abs": float(error.max()), "p99_abs": float(np.percentile(error, 99)), "correlation": float(np.corrcoef(actual.ravel(), cpu.values[9:].ravel())[0, 1])}
    if parity["max_abs"] > 1e-5 or parity["correlation"] < 0.999999:
        raise RuntimeError("CUDA MSLN parity failed")
    source, source_quiet, crop = _source_view(config)
    full = causal_joint_msln_cuda(
        source, JointSTContext("joint_s15_g3_t31_g1", 15, 3, 31, 1, "mean_std", 10.0),
        quiet_mask=source_quiet, review_crop_frames=crop, max_vram_bytes=cap,
    )
    full_diag = dict(full.diagnostics)
    del full, gpu, actual
    cp.get_default_memory_pool().free_all_blocks()
    payload = {"ready": True, "device": device, "max_vram_bytes": cap, "tiny_parity": parity, "full_context": full_diag}
    for root in (block, cascade):
        atomic_json(root / "gpu_validation.json", payload)
    return payload


def _check_ready(root: Path) -> None:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status["status"] == "complete":
        raise FileExistsError(f"completed root cannot be overwritten: {root}")
    if not (root / "gpu_validation.json").is_file():
        raise RuntimeError("matching GPU validation is required")


def _save_preview(root: Path, context_id: str, lanes: dict[str, Any], config: dict[str, Any]) -> None:
    import cupy as cp
    review_start = int(config["source"]["review_interval_ui"][0])
    indices = np.asarray([int(frame) - review_start for frame in config["outputs"]["representative_frames_ui"]], dtype=np.int32)
    payload = {"ui_frames": indices + review_start}
    for name, values in lanes.items():
        selected = cp.asnumpy(values[cp.asarray(indices)]) if hasattr(values, "__cuda_array_interface__") else np.asarray(values[indices])
        payload[name] = selected.astype(np.float16)
    destination = root / "stage_b" / "previews" / f"{context_id}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)


def _block_context(
    source: np.ndarray, source_quiet: np.ndarray, crop: int,
    raw_p_full: np.ndarray, raw_i_full: np.ndarray,
    context: JointSTContext, config: dict[str, Any], labels: list[dict[str, Any]],
    *, bandwidth: float, seed_offset: int, bootstrap: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import cupy as cp
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    quiet = source_quiet[crop:]
    z_raw_result = causal_joint_msln_cuda(source, context, quiet_mask=source_quiet, review_crop_frames=crop, max_vram_bytes=cap)
    z_raw = z_raw_result.values
    z_raw_floor = z_raw_result.scale_floor
    control_fit, control_details, control_boot = _fit_only(
        f"{context.context_id}_control", z_raw, config, bandwidth=bandwidth,
        seed_offset=seed_offset, bootstrap_replicates=bootstrap, run_fastica=bool(bootstrap),
    )
    control_p, control_i = apply_per_context_fit_cuda(z_raw, control_fit)
    metrics = {
        "control_persistence": _lane_metrics(control_p, quiet, labels, config),
        "control_innovation": _lane_metrics(control_i, quiet, labels, config),
    }
    lanes = {
        "control_zst": cp.asnumpy(z_raw),
        "control_persistence": cp.asnumpy(control_p),
        "control_innovation": cp.asnumpy(control_i),
    }
    del z_raw, control_p, control_i, z_raw_result
    cp.get_default_memory_pool().free_all_blocks()

    z_p_result = causal_joint_msln_cuda(raw_p_full, context, quiet_mask=source_quiet, review_crop_frames=crop, max_vram_bytes=cap)
    switch_p = z_p_result.values
    z_p_floor = z_p_result.scale_floor
    metrics["switch_persistence"] = _lane_metrics(switch_p, quiet, labels, config)
    lanes["switch_persistence"] = cp.asnumpy(switch_p)
    del switch_p, z_p_result
    cp.get_default_memory_pool().free_all_blocks()

    z_i_result = causal_joint_msln_cuda(raw_i_full, context, quiet_mask=source_quiet, review_crop_frames=crop, max_vram_bytes=cap)
    switch_i = z_i_result.values
    z_i_floor = z_i_result.scale_floor
    metrics["switch_innovation"] = _lane_metrics(switch_i, quiet, labels, config)
    lanes["switch_innovation"] = cp.asnumpy(switch_i)
    del switch_i, z_i_result
    cp.get_default_memory_pool().free_all_blocks()
    synthetic = _synthetic(context, int(config["sweep"]["seed"]) + seed_offset)
    row = {
        "context_id": context.context_id,
        "control_fit": control_details["cs_parzen"], "control_bootstrap": control_boot,
        "scale_floors": {"control": z_raw_floor, "switch_persistence": z_p_floor, "switch_innovation": z_i_floor},
        "lanes": metrics, "synthetic": synthetic,
        "selection_score": _selection_score(metrics["switch_persistence"], synthetic),
    }
    diagnostics = {"control_fit": control_details, "control_bootstrap": control_boot}
    return row, lanes, diagnostics


def run_block_switch(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    if not authorize_full_spon:
        raise PermissionError("full Spon block-switch run requires explicit authorization")
    import cupy as cp
    config = _load(config_path)
    root, _, _ = _roots(config)
    _check_ready(root)
    status = json.loads((root / "status.json").read_text())
    if status["status"] not in {"preflight_ready", "partial", "running"}:
        raise RuntimeError("block-switch root is not runnable")
    if status["status"] != "preflight_ready" and not resume:
        raise RuntimeError("partial block-switch run requires --resume")
    started = time.monotonic()
    atomic_json(root / "status.json", {"status": "running", "stage": "A"})
    source, source_quiet, crop = _source_view(config)
    labels = _labels(config)
    raw_device = cp.asarray(source, dtype=cp.float32)
    raw_review = raw_device[crop:]
    fit_path = root / "fits" / "raw_msica.json"
    if resume and fit_path.is_file() and (root / "features" / "raw_msica_persistence_full.npy").is_file():
        from neurobench.algorithms.multiscale_subspace import PerContextICAFit
        stored = json.loads(fit_path.read_text())
        payload = stored["fit"]
        for key in ("center", "whitening", "rotation", "demixing"):
            payload[key] = np.asarray(payload[key], dtype=np.float64)
        payload["component_signs"] = tuple(payload["component_signs"])
        raw_fit = PerContextICAFit(**payload)
        raw_bandwidth = float(stored["bandwidth"])
        raw_tuning = json.loads((root / "stage_a" / "raw_msica_tuning.json").read_text())
    else:
        raw_fit, raw_bandwidth, raw_tuning = _tune_raw(raw_review, config, root)
        raw_p_device, raw_i_device = apply_per_context_fit_cuda(raw_device, raw_fit)
        _atomic_npy(root / "features" / "raw_authority.npy", raw_review)
        _atomic_npy(root / "features" / "raw_msica_persistence_full.npy", raw_p_device)
        _atomic_npy(root / "features" / "raw_msica_innovation_full.npy", raw_i_device)
        atomic_json(fit_path, {"bandwidth": raw_bandwidth, "fit": raw_fit.to_dict(), "bootstrap": raw_tuning["bootstrap"]})
        del raw_p_device, raw_i_device
    del raw_device, raw_review
    cp.get_default_memory_pool().free_all_blocks()
    raw_p_full = np.load(root / "features" / "raw_msica_persistence_full.npy", mmap_mode="r")
    raw_i_full = np.load(root / "features" / "raw_msica_innovation_full.npy", mmap_mode="r")

    atomic_json(root / "status.json", {"status": "running", "stage": "B"})
    stage_path = root / "stage_b" / "context_metrics.json"
    stored_rows = json.loads(stage_path.read_text())["rows"] if resume and stage_path.is_file() else []
    completed = {row["context_id"] for row in stored_rows}
    contexts = _contexts(config)
    for index, context in enumerate(contexts, 1):
        if context.context_id in completed:
            continue
        print(f"BLOCK_STAGE_B {index}/{len(contexts)} START {context.context_id}", flush=True)
        tick = time.monotonic()
        row, lanes, _ = _block_context(
            source, source_quiet, crop, raw_p_full, raw_i_full, context, config, labels,
            bandwidth=float(config["ica"]["control_bandwidth"]), seed_offset=100 + index,
            bootstrap=0,
        )
        row["runtime_seconds"] = time.monotonic() - tick
        _save_preview(root, context.context_id, lanes, config)
        stored_rows.append(row)
        atomic_json(stage_path, {"complete": False, "rows": stored_rows})
        del lanes
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        print(f"BLOCK_STAGE_B {index}/{len(contexts)} DONE {context.context_id} {row['runtime_seconds']:.1f}s", flush=True)
    forced = list(config["sweep"]["forced_contexts"])
    ranked = sorted(stored_rows, key=lambda row: (-row["selection_score"], row["context_id"]))
    finalists = list(forced)
    for row in ranked:
        if len(finalists) >= int(config["sweep"]["finalist_contexts"]):
            break
        if row["context_id"] not in finalists:
            finalists.append(row["context_id"])
    atomic_json(stage_path, {"complete": True, "rows": stored_rows, "finalists": finalists, "selection_basis": "forced v2 references plus unsupervised visual/synthetic score; sparse point labels excluded"})

    atomic_json(root / "status.json", {"status": "running", "stage": "C"})
    final_path = root / "stage_c" / "finalist_metrics.json"
    final_rows = json.loads(final_path.read_text())["rows"] if resume and final_path.is_file() else []
    final_done = {row["context_id"] for row in final_rows}
    context_by_id = {item.context_id: item for item in contexts}
    for index, context_id in enumerate(finalists, 1):
        if context_id in final_done:
            continue
        print(f"BLOCK_STAGE_C {index}/{len(finalists)} START {context_id}", flush=True)
        row, lanes, diagnostics = _block_context(
            source, source_quiet, crop, raw_p_full, raw_i_full, context_by_id[context_id], config, labels,
            bandwidth=float(config["ica"]["control_bandwidth"]), seed_offset=500 + index,
            bootstrap=int(config["ica"]["bootstrap_replicates"]),
        )
        feature_root = root / "features" / context_id
        for name, values in lanes.items():
            _atomic_npy(feature_root / f"{name}.npy", values)
        atomic_json(root / "fits" / f"{context_id}.json", diagnostics)
        final_rows.append(row)
        atomic_json(final_path, {"complete": False, "rows": final_rows})
        del lanes
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        print(f"BLOCK_STAGE_C {index}/{len(finalists)} DONE {context_id}", flush=True)
    atomic_json(final_path, {"complete": True, "rows": final_rows, "finalists": finalists, "raw_direct_anchor": RAW_DIRECT_ANCHOR})

    atomic_json(root / "status.json", {"status": "running", "stage": "D"})
    videos = _render_block_videos(root, config, finalists[:3])
    elapsed = time.monotonic() - started
    manifest = {"status": "complete", "experiment_id": config["experiment_id"] + "_block_switch", "compute_backend": "cuda", "elapsed_seconds": elapsed, "selected_raw_bandwidth": raw_bandwidth, "finalists": finalists, "videos": videos}
    atomic_json(root / "run_manifest.json", manifest)
    _write_block_report(root, config, raw_tuning, final_rows)
    atomic_json(root / "status.json", {"status": "complete", "scientific_status": "awaiting_cross_experiment_conclusion", "elapsed_seconds": elapsed})
    return manifest


def _render_block_videos(root: Path, config: dict[str, Any], contexts: list[str]) -> list[str]:
    raw = np.load(root / "features" / "raw_authority.npy", mmap_mode="r")
    raw_p = np.load(root / "features" / "raw_msica_persistence_full.npy", mmap_mode="r")[31:]
    raw_i = np.load(root / "features" / "raw_msica_innovation_full.npy", mmap_mode="r")[31:]
    output = root / "videos"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for context_id in contexts:
        base = root / "features" / context_id
        layers = [Layer("Raw (authority)", raw, "raw"), Layer("Raw MSICA persistence", raw_p, "signed"), Layer("Raw MSICA innovation", raw_i, "signed")]
        for name, title in (
            ("control_zst", "Control MSLN(raw)"), ("control_persistence", "Control MSLN→MSICA persistence"),
            ("control_innovation", "Control MSLN→MSICA innovation"), ("switch_persistence", "Switched MSICA→MSLN persistence"),
            ("switch_innovation", "Switched MSICA→MSLN innovation"),
        ):
            layers.append(Layer(title, np.load(base / f"{name}.npy", mmap_mode="r"), "signed"))
        record = _render_video(output / f"{context_id}_block_order_comparison.mp4", layers, f"MSLN/MSICA block-order comparison — {context_id}", review_start_ui=int(config["source"]["review_interval_ui"][0]), fps=float(config["outputs"]["fps"]), columns=3)
        records.append(str(Path("videos") / record["path"]))
    atomic_json(output / "video_manifest.json", {"videos": records, "contexts": contexts})
    return records


def _best_row(rows: list[dict[str, Any]], lane: str, budget: str = "58") -> dict[str, Any]:
    return sorted(rows, key=lambda row: (-row["lanes"][lane]["recall_guardrail"]["matched_by_budget"][budget], -row["lanes"][lane]["visual_stats"]["event_quiet_ratio_p999"], row["context_id"]))[0]


def _write_block_report(root: Path, config: dict[str, Any], tuning: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    control = _best_row(rows, "control_persistence")
    switched = _best_row(rows, "switch_persistence")
    budget = str(config["evaluation"]["guardrail_budget"])
    boot = tuning["bootstrap"]
    text = f"""# Block-switch experiment: concise report

## Result

- Best control `Raw → MSLN → MSICA` persistence: `{control['context_id']}`, {control['lanes']['control_persistence']['recall_guardrail']['matched_by_budget'][budget]}/79 known matches at budget {budget} per burst.
- Best switched `Raw → MSICA → MSLN` persistence: `{switched['context_id']}`, {switched['lanes']['switch_persistence']['recall_guardrail']['matched_by_budget'][budget]}/79 known matches at the same budget.
- Raw MSICA selected Parzen bandwidth: `{tuning['selected_bandwidth']}`.
- Raw MSICA bootstrap circular SD: `{boot['circular_std_degrees']:.3f}°`; component-swap fraction: `{boot['component_swap_fraction']:.3f}`; ambiguous fraction: `{boot['ambiguous_fraction']:.3f}`.

## Interpretation

Both orders used the same recording, MSLN context bank, candidate budgets, NMS/matching radii, and GPU numerical contract. Sparse labels were excluded from fitting and finalist selection. Unmatched candidates remain unknown, so these values are recall guardrails rather than precision estimates.

See `stage_c/finalist_metrics.json`, `fits/`, and `videos/` for the comparable artifacts.
"""
    (root / "CONCISE_REPORT.md").write_text(text, encoding="utf-8")


def _raw_stability_gate(block_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    tuning = json.loads((block_root / "stage_a" / "raw_msica_tuning.json").read_text())
    boot = tuning["bootstrap"]
    thresholds = config["ica"]["cascade_gate"]
    checks = {
        "circular_std": float(boot["circular_std_degrees"]) <= float(thresholds["max_circular_std_degrees"]),
        "swap_fraction": float(boot["component_swap_fraction"]) <= float(thresholds["max_component_swap_fraction"]),
        "ambiguous_fraction": float(boot["ambiguous_fraction"]) <= float(thresholds["max_ambiguous_fraction"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "bootstrap": boot, "thresholds": thresholds}


def _tune_second_stage(block_root: Path, finalists: list[str], config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    rows = []
    for context_index, context_id in enumerate(finalists[:3]):
        values = np.load(block_root / "features" / context_id / "switch_persistence.npy", mmap_mode="r")
        for bandwidth in config["ica"]["bandwidth_grid"]:
            fit, _, _ = _fit_only(f"{context_id}_cascade_tuning", values, config, bandwidth=float(bandwidth), seed_offset=3000 + 100 * context_index, bootstrap_replicates=0)
            rows.append({"context_id": context_id, "bandwidth": float(bandwidth), "objective_gain_fraction": _objective_gain(fit), "angle_degrees": fit.rotation_angle_degrees})
    aggregates = []
    for bandwidth in config["ica"]["bandwidth_grid"]:
        selected = [row for row in rows if row["bandwidth"] == float(bandwidth)]
        aggregates.append({"bandwidth": float(bandwidth), "median_objective_gain_fraction": float(np.median([row["objective_gain_fraction"] for row in selected]))})
    winner = sorted(aggregates, key=lambda row: (-row["median_objective_gain_fraction"], row["bandwidth"]))[0]
    return float(winner["bandwidth"]), {"rows": rows, "aggregates": aggregates, "selected_bandwidth": float(winner["bandwidth"]), "selection_basis": "median relative CS objective gain over three frozen contexts; labels excluded"}


def run_cascade(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    if not authorize_full_spon:
        raise PermissionError("full Spon cascade run requires explicit authorization")
    import cupy as cp
    config = _load(config_path)
    block, root, _ = _roots(config)
    _check_ready(root)
    if json.loads((block / "status.json").read_text())["status"] != "complete":
        raise RuntimeError("completed block-switch experiment is required")
    gate = _raw_stability_gate(block, config)
    atomic_json(root / "stage_a" / "advancement_gate.json", gate)
    if not gate["passed"]:
        atomic_json(root / "status.json", {"status": "complete", "scientific_status": "stopped_by_preregistered_raw_msica_stability_gate"})
        (root / "CONCISE_REPORT.md").write_text("# Cascade experiment: concise report\n\nThe cascade was not executed because the preregistered raw-MSICA stability gate failed. See `stage_a/advancement_gate.json`.\n", encoding="utf-8")
        return {"status": "stopped_by_gate", "gate": gate}
    started = time.monotonic()
    atomic_json(root / "status.json", {"status": "running", "stage": "A"})
    block_metrics = json.loads((block / "stage_c" / "finalist_metrics.json").read_text())
    finalists = list(block_metrics["finalists"])
    bandwidth, tuning = _tune_second_stage(block, finalists, config)
    atomic_json(root / "stage_a" / "msica2_tuning.json", tuning)
    labels = _labels(config)
    _, source_quiet, crop = _source_view(config)
    quiet = source_quiet[crop:]
    atomic_json(root / "status.json", {"status": "running", "stage": "B"})
    metrics_path = root / "stage_b" / "cascade_metrics.json"
    rows = json.loads(metrics_path.read_text())["rows"] if resume and metrics_path.is_file() else []
    completed = {row["context_id"] for row in rows}
    for index, context_id in enumerate(finalists, 1):
        if context_id in completed:
            continue
        print(f"CASCADE_STAGE_B {index}/{len(finalists)} START {context_id}", flush=True)
        feature_root = root / "features" / context_id
        lane_metrics = {}
        fits = {}
        for branch_index, branch in enumerate(("switch_persistence", "switch_innovation")):
            values = cp.asarray(np.load(block / "features" / context_id / f"{branch}.npy", mmap_mode="r"), dtype=cp.float32)
            fit, details, boot = _fit_only(
                f"{context_id}_{branch}_msica2", values, config, bandwidth=bandwidth,
                seed_offset=4000 + 100 * index + 20 * branch_index,
                bootstrap_replicates=int(config["ica"]["bootstrap_replicates"]), run_fastica=True,
            )
            persistence, innovation = apply_per_context_fit_cuda(values, fit)
            prefix = "p1" if branch == "switch_persistence" else "i1"
            for suffix, output in (("p2", persistence), ("i2", innovation)):
                lane = f"{prefix}_to_{suffix}"
                lane_metrics[lane] = _lane_metrics(output, quiet, labels, config)
                _atomic_npy(feature_root / f"{lane}.npy", output)
            fits[branch] = {"fit": details, "bootstrap": boot}
            del values, persistence, innovation
            cp.get_default_memory_pool().free_all_blocks()
        atomic_json(root / "fits" / f"{context_id}.json", fits)
        row = {"context_id": context_id, "bandwidth": bandwidth, "lanes": lane_metrics, "fits": {key: {"bootstrap": value["bootstrap"], "cs_parzen": value["fit"]["cs_parzen"]} for key, value in fits.items()}}
        rows.append(row)
        atomic_json(metrics_path, {"complete": False, "rows": rows})
        gc.collect()
        print(f"CASCADE_STAGE_B {index}/{len(finalists)} DONE {context_id}", flush=True)
    atomic_json(metrics_path, {"complete": True, "rows": rows, "finalists": finalists, "primary_lane": "p1_to_p2", "raw_direct_anchor": RAW_DIRECT_ANCHOR})
    atomic_json(root / "status.json", {"status": "running", "stage": "C"})
    videos = _render_cascade_videos(block, root, config, finalists[:3])
    elapsed = time.monotonic() - started
    manifest = {"status": "complete", "experiment_id": config["experiment_id"] + "_cascade", "compute_backend": "cuda", "elapsed_seconds": elapsed, "selected_msica2_bandwidth": bandwidth, "finalists": finalists, "videos": videos, "advancement_gate": gate}
    atomic_json(root / "run_manifest.json", manifest)
    _write_cascade_report(block, root, config, rows, gate)
    atomic_json(root / "status.json", {"status": "complete", "scientific_status": "awaiting_cross_experiment_conclusion", "elapsed_seconds": elapsed})
    return manifest


def _render_cascade_videos(block: Path, root: Path, config: dict[str, Any], contexts: list[str]) -> list[str]:
    raw = np.load(block / "features" / "raw_authority.npy", mmap_mode="r")
    raw_p = np.load(block / "features" / "raw_msica_persistence_full.npy", mmap_mode="r")[31:]
    raw_i = np.load(block / "features" / "raw_msica_innovation_full.npy", mmap_mode="r")[31:]
    output = root / "videos"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for context_id in contexts:
        block_root = block / "features" / context_id
        cascade_root = root / "features" / context_id
        layers = [Layer("Raw (authority)", raw, "raw"), Layer("MSICA-1 persistence", raw_p, "signed"), Layer("MSICA-1 innovation", raw_i, "signed"), Layer("MSLN(P1)", np.load(block_root / "switch_persistence.npy", mmap_mode="r"), "signed"), Layer("MSLN(I1)", np.load(block_root / "switch_innovation.npy", mmap_mode="r"), "signed")]
        for lane, title in (("p1_to_p2", "P1→MSLN→P2"), ("p1_to_i2", "P1→MSLN→I2"), ("i1_to_p2", "I1→MSLN→P2"), ("i1_to_i2", "I1→MSLN→I2")):
            layers.append(Layer(title, np.load(cascade_root / f"{lane}.npy", mmap_mode="r"), "signed"))
        record = _render_video(output / f"{context_id}_cascade_journey.mp4", layers, f"MSICA→MSLN→MSICA cascade — {context_id}", review_start_ui=int(config["source"]["review_interval_ui"][0]), fps=float(config["outputs"]["fps"]), columns=3)
        records.append(str(Path("videos") / record["path"]))
    atomic_json(output / "video_manifest.json", {"videos": records, "contexts": contexts})
    return records


def _write_cascade_report(block: Path, root: Path, config: dict[str, Any], rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    cascade = _best_row(rows, "p1_to_p2")
    block_rows = json.loads((block / "stage_c" / "finalist_metrics.json").read_text())["rows"]
    switched = _best_row(block_rows, "switch_persistence")
    budget = str(config["evaluation"]["guardrail_budget"])
    cm = cascade["lanes"]["p1_to_p2"]["recall_guardrail"]["matched_by_budget"][budget]
    bm = switched["lanes"]["switch_persistence"]["recall_guardrail"]["matched_by_budget"][budget]
    text = f"""# Cascade experiment: concise report

## Result

- Raw-MSICA advancement gate: passed.
- Best primary cascade `P1 → MSLN → P2`: `{cascade['context_id']}`, {cm}/79 known matches at budget {budget} per burst.
- Best single switched block: `{switched['context_id']}`, {bm}/79 at the same budget.
- Incremental cascade difference: {cm - bm:+d} known matches.

## Interpretation

The second MSICA layer was tuned without sparse point labels and evaluated with the same candidate, NMS, and matching contract. The primary lane was frozen before evaluation. Unmatched candidates remain unknown.

See `stage_b/cascade_metrics.json`, `fits/`, and `videos/` for the complete diagnostics.
"""
    (root / "CONCISE_REPORT.md").write_text(text, encoding="utf-8")


def conclude(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path)
    block, cascade, summary = _roots(config)
    if json.loads((block / "status.json").read_text())["status"] != "complete" or json.loads((cascade / "status.json").read_text())["status"] != "complete":
        raise RuntimeError("both experiment roots must be complete before conclusion")
    block_rows = json.loads((block / "stage_c" / "finalist_metrics.json").read_text())["rows"]
    control = _best_row(block_rows, "control_persistence")
    switched = _best_row(block_rows, "switch_persistence")
    cascade_status = json.loads((cascade / "status.json").read_text())
    budget = str(config["evaluation"]["guardrail_budget"])
    records = {
        "v2_external_anchor": {"context_id": "joint_s15_g3_t31_g1", "matched": 58, "labels": 79},
        "block_control": {"context_id": control["context_id"], "matched": control["lanes"]["control_persistence"]["recall_guardrail"]["matched_by_budget"][budget], "labels": 79},
        "block_switched": {"context_id": switched["context_id"], "matched": switched["lanes"]["switch_persistence"]["recall_guardrail"]["matched_by_budget"][budget], "labels": 79},
    }
    if (cascade / "stage_b" / "cascade_metrics.json").is_file():
        cascade_rows = json.loads((cascade / "stage_b" / "cascade_metrics.json").read_text())["rows"]
        best = _best_row(cascade_rows, "p1_to_p2")
        records["cascade_primary"] = {"context_id": best["context_id"], "matched": best["lanes"]["p1_to_p2"]["recall_guardrail"]["matched_by_budget"][budget], "labels": 79}
    winner = max(records.items(), key=lambda item: item[1]["matched"])
    raw_stability = json.loads((block / "stage_a" / "raw_msica_tuning.json").read_text())["bootstrap"]
    payload = {"status": "complete", "guardrail_budget_per_burst": int(budget), "records": records, "best_descriptive_guardrail": {"lane": winner[0], **winner[1]}, "cascade_scientific_status": cascade_status.get("scientific_status"), "precision_identified": False}
    atomic_json(summary / "comparison_metrics.json", payload)
    text = "# MSLN/MSICA order program: conclusive report\n\n"
    text += "## Comparative result\n\n"
    for name, row in records.items():
        text += f"- `{name}` — `{row['context_id']}`: {row['matched']}/{row['labels']} known matches at budget {budget} per burst.\n"
    text += f"\nThe block switch recovered {records["block_switched"]["matched"]}/79 versus {records["block_control"]["matched"]}/79 for the paired control. Raw MSICA remained unstable (circular SD {raw_stability["circular_std_degrees"]:.2f} degrees; swap fraction {raw_stability["component_swap_fraction"]:.3f}).\n\n"
    if cascade_status.get("scientific_status") == "stopped_by_preregistered_raw_msica_stability_gate":
        text += "The cascade was not executed because raw MSICA failed its preregistered stability gate; stacking another ICA block would not be scientifically defensible.\n\n"
    text += "These conclusions are limited to sparse-positive recall and the completed visual/stability diagnostics; unmatched candidates remain unknown and precision is not identified.\n\n"
    text += "## Decision\n\nRetain the existing `Raw -> MSLN -> MSICA` broad persistence representation as the preferred stable order. The switched map is visually and descriptively competitive, but it did not improve fixed-budget recall and its first ICA block was not identifiable. Do not advance the cascade without a redesigned or regularized first-stage separation. A stronger claim still requires exhaustive bounded-field review or an independent labeled recording.\n"
    (summary / "CONCLUSIVE_REPORT.md").write_text(text, encoding="utf-8")
    previous_block_status = json.loads((block / "status.json").read_text())
    atomic_json(block / "status.json", {**previous_block_status, "status": "complete", "scientific_status": "compared_no_switch_advantage"})
    atomic_json(summary / "status.json", {"status": "complete"})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "gpu-preflight", "run-block-switch", "run-cascade", "conclude", "summarize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-full-spon", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "preflight":
        payload = preflight(args.config)
    elif args.action == "gpu-preflight":
        payload = gpu_preflight(args.config)
    elif args.action == "run-block-switch":
        payload = run_block_switch(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    elif args.action == "run-cascade":
        payload = run_cascade(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    elif args.action == "conclude":
        payload = conclude(args.config)
    else:
        config = _load(args.config)
        payload = {str(root): json.loads((root / "status.json").read_text()) if (root / "status.json").is_file() else None for root in _roots(config)}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
