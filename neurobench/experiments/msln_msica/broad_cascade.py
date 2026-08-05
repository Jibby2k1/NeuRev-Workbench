"""Broad, sharded MSLN/MSICA cascade and parallel-control program."""
from __future__ import annotations

import argparse
import copy
import gc
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

from neurobench.algorithms.msln_msica_cuda import (
    apply_per_context_fit_cuda,
    causal_joint_msln_cuda,
    cuda_device_summary,
)
from neurobench.algorithms.multiscale_local_normalization import JointSTContext, causal_joint_msln
from neurobench.algorithms.multiscale_subspace import fit_per_context_ica
from neurobench.reports.msln_msica_videos import Layer, _render_video

from .artifacts import atomic_json, sha256_file, sha256_payload
from .joint_sweep import RAW_DIRECT_ANCHOR, _atomic_npy, _contexts, _label_overlay, _labels, _recall, _visual_stats
from .order_program import _fit_only, _objective_gain, _paired_eval


EXPERIMENTS = (
    "00_original_shallow",
    "01_original_deep",
    "02_switched_per_branch",
    "03_switched_seed_ensemble",
    "04_cross_branch",
    "05_parallel_fusion_control",
)


def _load(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_id", "source", "sweep", "ica", "evaluation", "compute", "outputs"}
    if set(payload) != required or payload["schema_version"] != 4:
        raise ValueError("broad cascade requires the exact schema-v4 contract")
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path", "v2_root", "v3_root"):
        payload["source"][key] = str((source.parent / payload["source"][key]).resolve())
    payload["outputs"]["root_dir"] = str((source.parent / payload["outputs"]["root_dir"]).resolve())
    payload["_config_path"] = str(source)
    _validate(payload)
    return payload


def _validate(config: dict[str, Any]) -> None:
    sweep = config["sweep"]
    if config["source"]["axes"] != "TYX" or not config["source"]["ui_one_based"]:
        raise ValueError("source must use one-based UI TYX coordinates")
    if sweep["spatial_outer_guard_pairs"] != [[5, 1], [7, 1], [7, 3], [11, 3], [15, 3], [15, 5]]:
        raise ValueError("all six v2 spatial context definitions are frozen")
    if sweep["temporal_windows_frames"] != [5, 9, 15, 23, 31]:
        raise ValueError("all five v2 temporal windows are frozen")
    if sweep["msica2_bandwidth_grid"] != [0.15, 0.25, 0.35, 0.5, 0.7]:
        raise ValueError("the five-value MSICA2 grid is frozen")
    if not sweep["full_context_pairs"] or sweep["branches"] != ["persistence", "innovation"]:
        raise ValueError("v4 requires all ordered context pairs and both branches")
    if int(sweep["ensemble_seeds"]) != 5 or int(sweep["finalists_per_experiment"]) != 3:
        raise ValueError("v4 freezes five ensemble seeds and three finalists")
    if config["evaluation"]["candidate_budgets"] != [20, 40, 58, 80, 100]:
        raise ValueError("evaluation budgets must match v2/v3")
    if config["compute"]["device"] != "cuda" or int(config["compute"]["workers_per_gpu"]) != 1:
        raise ValueError("one CUDA worker per GPU is mandatory")
    if not 0 < float(config["compute"]["max_peak_vram_gb"]) <= 8:
        raise ValueError("VRAM cap must lie in (0,8] GiB")


def _resolved(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _root(config: dict[str, Any]) -> Path:
    return Path(config["outputs"]["root_dir"])


def _experiment_root(root: Path, experiment: str) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    return root / experiment


def _heartbeat(root: Path, stage: str, **extra: Any) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **extra}
    path = root / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _extended_source(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int]:
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    first_crop = max(map(int, config["sweep"]["temporal_windows_frames"]))
    second_crop = first_crop + 1
    total_pre_roll = first_crop + second_crop
    source_start = review_start - 1 - total_pre_roll
    if source_start < 0:
        raise ValueError("recording lacks required two-stage causal pre-roll")
    source = movie[source_start:review_stop]
    quiet = np.zeros(len(source), dtype=bool)
    quiet_start, quiet_stop = map(int, config["source"]["quiet_interval_ui"])
    quiet[quiet_start - 1 - source_start : quiet_stop - source_start] = True
    if len(source) - total_pre_roll != review_stop - review_start + 1:
        raise RuntimeError("extended source alignment invariant failed")
    return source, quiet, first_crop, second_crop


def _screen_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["ica"]["screen_samples"] = int(config["ica"]["screen_samples"])
    result["ica"]["confirmation_samples"] = int(config["ica"]["confirmation_samples"])
    result["ica"]["bootstrap_replicates"] = int(config["ica"]["final_bootstrap_replicates"])
    result["ica"]["bootstrap_block_samples"] = int(config["ica"]["bootstrap_block_samples"])
    return result


def _final_config(config: dict[str, Any]) -> dict[str, Any]:
    result = _screen_config(config)
    result["ica"]["screen_samples"] = int(config["ica"]["final_screen_samples"])
    result["ica"]["confirmation_samples"] = int(config["ica"]["final_confirmation_samples"])
    return result


def _context_by_id(config: dict[str, Any]) -> dict[str, JointSTContext]:
    return {context.context_id: context for context in _contexts(config)}


def _compact_fit(fit: Any) -> dict[str, Any]:
    return {
        "rotation_angle_degrees": float(fit.rotation_angle_degrees),
        "objective_value": float(fit.objective_value),
        "baseline_objective_value": float(fit.baseline_objective_value),
        "objective_gain_fraction": _objective_gain(fit),
        "derivative_angle_distance_degrees": float(fit.derivative_angle_distance_degrees),
        "converged": bool(fit.converged),
        "ambiguous_alignment": bool(fit.ambiguous_alignment),
        "condition_number": float(fit.diagnostics["whitening"]["condition_number"]),
    }


def _label_free_metrics(values: Any, quiet: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    import cupy as cp
    evidence = cp.square(values, dtype=cp.float32) if hasattr(values, "__cuda_array_interface__") else np.square(values, dtype=np.float32)
    visual = _visual_stats(evidence, quiet, config)
    score = float(
        np.log1p(visual["event_quiet_ratio_p999"])
        + 0.5 * np.log1p(1000.0 * visual["event_fraction_above_quiet_p999"])
    )
    del evidence
    return {"visual_stats": visual, "selection_score": score}


def _protected_metrics(values: Any, quiet: np.ndarray, labels: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    import cupy as cp
    evidence = cp.square(values, dtype=cp.float32) if hasattr(values, "__cuda_array_interface__") else np.square(values, dtype=np.float32)
    payload = {
        "visual_stats": _visual_stats(evidence, quiet, config),
        "recall_guardrail": _paired_eval(evidence, labels, config),
    }
    del evidence
    return payload


def _preview(path: Path, lanes: dict[str, Any], config: dict[str, Any]) -> None:
    import cupy as cp
    review_start = int(config["source"]["review_interval_ui"][0])
    indices = np.asarray([int(frame) - review_start for frame in config["outputs"]["representative_frames_ui"]], dtype=np.int32)
    stride = int(config["outputs"]["preview_stride_px"])
    payload: dict[str, np.ndarray] = {"ui_frames": indices + review_start}
    for name, values in lanes.items():
        selected = cp.asnumpy(values[cp.asarray(indices), ::stride, ::stride]) if hasattr(values, "__cuda_array_interface__") else np.asarray(values[indices, ::stride, ::stride])
        payload[name] = selected.astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _tune_msica2(values: Any, config: dict[str, Any], *, lane_id: str, seed_offset: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    fits = []
    rows = []
    screen = _screen_config(config)
    for bandwidth in config["sweep"]["msica2_bandwidth_grid"]:
        fit, _, _ = _fit_only(
            lane_id, values, screen, bandwidth=float(bandwidth),
            seed_offset=int(seed_offset), bootstrap_replicates=0,
        )
        fits.append((float(bandwidth), fit))
        rows.append({"bandwidth": float(bandwidth), **_compact_fit(fit)})
    winner_bandwidth, winner = sorted(
        fits,
        key=lambda item: (-_objective_gain(item[1]), item[1].ambiguous_alignment, item[0]),
    )[0]
    persistence, innovation = apply_per_context_fit_cuda(values, winner)
    return winner, persistence, innovation, {
        "grid": rows,
        "selected_bandwidth": winner_bandwidth,
        "selection_basis": "maximum relative CS objective gain; labels excluded",
    }


def _first_stage(
    source: np.ndarray,
    source_quiet: np.ndarray,
    first_crop: int,
    second_crop: int,
    context: JointSTContext,
    config: dict[str, Any],
    *,
    bandwidth: float,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cupy as cp
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    result = causal_joint_msln_cuda(
        source, context, quiet_mask=source_quiet,
        review_crop_frames=first_crop, max_vram_bytes=cap,
    )
    z1 = result.values
    fit, details, _ = _fit_only(
        f"{context.context_id}_msica1", z1[second_crop:], _screen_config(config),
        bandwidth=float(bandwidth), seed_offset=int(seed_offset),
        bootstrap_replicates=0,
    )
    p1, i1 = apply_per_context_fit_cuda(z1, fit)
    p_cpu = cp.asnumpy(p1)
    i_cpu = cp.asnumpy(i1)
    diagnostics = {
        "context_id": context.context_id,
        "bandwidth": float(bandwidth),
        "scale_floor": result.scale_floor,
        "fit": _compact_fit(fit),
    }
    del result, z1, p1, i1
    cp.get_default_memory_pool().free_all_blocks()
    return p_cpu, i_cpu, diagnostics


def _second_msln(
    values: np.ndarray,
    first_quiet: np.ndarray,
    second_crop: int,
    context: JointSTContext,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    result = causal_joint_msln_cuda(
        values, context, quiet_mask=first_quiet,
        review_crop_frames=second_crop, max_vram_bytes=cap,
    )
    return result.values, {"scale_floor": result.scale_floor, "diagnostics": result.diagnostics}


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path)
    root = _root(config)
    if root.exists():
        raise FileExistsError(f"output root exists: {root}")
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path"):
        if not Path(config["source"][key]).is_file():
            raise FileNotFoundError(config["source"][key])
    for key in ("v2_root", "v3_root"):
        if not Path(config["source"][key]).is_dir():
            raise FileNotFoundError(config["source"][key])
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    labels = _labels(config)
    source, quiet, first_crop, second_crop = _extended_source(config)
    if movie.ndim != 3 or len(source) != 623 or first_crop != 31 or second_crop != 32:
        raise ValueError("source shape or two-stage alignment contract failed")
    if not np.isfinite(np.asarray(source[::64, ::17, ::19], dtype=np.float32)).all():
        raise ValueError("source contains non-finite values")
    height, width = movie.shape[1:]
    if any(not (0 <= int(row["x_px"]) < width and 0 <= int(row["y_px"]) < height) for row in labels):
        raise ValueError("label coordinate outside movie")
    contexts = _contexts(config)
    original_combinations = len(contexts) ** 2 * len(config["sweep"]["branches"])
    switched_combinations = len(contexts) * len(config["sweep"]["branches"])
    total_screen_fits = (
        original_combinations * len(config["sweep"]["msica2_bandwidth_grid"])
        + switched_combinations * len(config["sweep"]["msica2_bandwidth_grid"])
        + len(contexts) * len(config["sweep"]["msica2_bandwidth_grid"])
    )
    disk = shutil.disk_usage(root.parent)
    estimate = 48 * 2**30
    if disk.free < estimate * 2:
        raise RuntimeError("insufficient disk headroom for broad cascade")
    fingerprints = {
        "movie": {"sha256": sha256_file(Path(config["source"]["movie_path"])), "shape": list(movie.shape), "dtype": str(movie.dtype)},
        "labels": {"sha256": sha256_file(Path(config["source"]["labels_path"])), "rows": len(labels)},
        "v2_manifest": sha256_file(Path(config["source"]["v2_root"]) / "run_manifest.json"),
        "v3_manifest": sha256_file(Path(config["source"]["v3_root"]) / "run_manifest.json"),
    }
    fingerprint = sha256_payload({"config": _resolved(config), "inputs": fingerprints})
    root.mkdir(parents=True, exist_ok=False)
    for experiment in EXPERIMENTS:
        experiment_root = _experiment_root(root, experiment)
        experiment_root.mkdir(parents=True)
        atomic_json(experiment_root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    atomic_json(root / "config.resolved.json", _resolved(config))
    atomic_json(root / "resource_plan.json", {
        "contexts": len(contexts),
        "ordered_original_context_pairs": len(contexts) ** 2,
        "original_branch_combinations": original_combinations,
        "switched_branch_combinations": switched_combinations,
        "estimated_msica2_screen_fits_minimum": total_screen_fits,
        "workers_per_gpu": 1,
        "full_maps_retained_for_label_free_finalists_only": True,
        "output_bytes_estimate": estimate,
        "disk_free_bytes": disk.free,
    })
    atomic_json(root / "preflight.json", {
        "ready": True,
        "preflight_fingerprint": fingerprint,
        "input_fingerprints": fingerprints,
        "source_read_only": True,
        "two_stage_pre_roll_frames": first_crop + second_crop,
        "labels_used_for_screening_or_tuning": False,
        "raw_direct_anchor": RAW_DIRECT_ANCHOR,
    })
    atomic_json(root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    _label_overlay(root, movie, labels, config)
    return {"ready": True, "root": str(root), "preflight_fingerprint": fingerprint, "resource_plan": json.loads((root / "resource_plan.json").read_text())}


def gpu_preflight(config_path: str | Path) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path)
    root = _root(config)
    if not (root / "preflight.json").is_file():
        raise RuntimeError("matching read-only preflight is required")
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    device = cuda_device_summary()
    if device["free_bytes"] < cap:
        raise RuntimeError("free GPU memory is below the frozen cap")
    rng = np.random.default_rng(19)
    tiny = (1000 + rng.normal(0, 3, (80, 31, 33))).astype(np.float32)
    tiny[60:64, 14:18, 15:19] += 20
    quiet = np.arange(80) < 45
    context = JointSTContext("broad_cascade_gpu_smoke", 7, 3, 9, 1)
    cpu_quiet = quiet.copy()
    cpu_quiet[:31] = False
    cpu1 = causal_joint_msln(tiny, context, quiet_mask=cpu_quiet)
    gpu1 = causal_joint_msln_cuda(tiny, context, quiet_mask=quiet, review_crop_frames=31, max_vram_bytes=min(cap, 2**30))
    first = cp.asnumpy(gpu1.values)
    error1 = np.abs(first - cpu1.values[31:])
    quiet2 = quiet[31:]
    cpu_quiet2 = quiet2.copy()
    cpu_quiet2[:10] = False
    cpu2 = causal_joint_msln(first, context, quiet_mask=cpu_quiet2)
    gpu2 = causal_joint_msln_cuda(first, context, quiet_mask=quiet2, review_crop_frames=10, max_vram_bytes=min(cap, 2**30))
    second = cp.asnumpy(gpu2.values)
    error2 = np.abs(second - cpu2.values[10:])
    parity = {
        "stage1_max_abs": float(error1.max()), "stage1_correlation": float(np.corrcoef(first.ravel(), cpu1.values[31:].ravel())[0, 1]),
        "stage2_max_abs": float(error2.max()), "stage2_correlation": float(np.corrcoef(second.ravel(), cpu2.values[10:].ravel())[0, 1]),
    }
    if max(parity["stage1_max_abs"], parity["stage2_max_abs"]) > 1e-5 or min(parity["stage1_correlation"], parity["stage2_correlation"]) < 0.999999:
        raise RuntimeError("two-stage CUDA parity failed")
    source, source_quiet, first_crop, _ = _extended_source(config)
    full = causal_joint_msln_cuda(
        source, JointSTContext("joint_s15_g3_t31_g1", 15, 3, 31, 1, "mean_std", 10.0),
        quiet_mask=source_quiet, review_crop_frames=first_crop, max_vram_bytes=cap,
    )
    payload = {"ready": True, "device": device, "max_vram_bytes": cap, "two_stage_tiny_parity": parity, "full_first_stage": full.diagnostics}
    del full, gpu1, gpu2, first, second
    cp.get_default_memory_pool().free_all_blocks()
    atomic_json(root / "gpu_validation.json", payload)
    for experiment in EXPERIMENTS:
        atomic_json(_experiment_root(root, experiment) / "gpu_validation.json", payload)
    return payload


def _require_run(config: dict[str, Any], *, authorize_full_spon: bool) -> Path:
    if not authorize_full_spon:
        raise PermissionError("full Spon cascade program requires explicit authorization")
    root = _root(config)
    if not (root / "preflight.json").is_file() or not (root / "gpu_validation.json").is_file():
        raise RuntimeError("matching read-only and CUDA preflights are required")
    if json.loads((root / "status.json").read_text())["status"] == "complete":
        raise FileExistsError("completed broad-cascade root cannot be overwritten")
    return root


def _write_shard(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, {**payload, "complete": True})


def _merge_shards(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("complete"):
            raise RuntimeError(f"incomplete shard: {path}")
        rows.extend(payload["rows"])
    return rows


def run_original_screen(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    """Evaluate all 30x30 original-order context pairs and both branches."""
    import cupy as cp
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon=authorize_full_spon)
    shallow_root = _experiment_root(root, "00_original_shallow")
    deep_root = _experiment_root(root, "01_original_deep")
    if not resume and any(json.loads((p / "status.json").read_text())["status"] != "preflight_ready" for p in (shallow_root, deep_root)):
        raise RuntimeError("non-fresh original screen requires --resume")
    atomic_json(root / "status.json", {"status": "running", "stage": "original_screen"})
    atomic_json(shallow_root / "status.json", {"status": "running", "stage": "full_factorial_screen"})
    atomic_json(deep_root / "status.json", {"status": "running", "stage": "full_factorial_screen"})
    source, source_quiet, first_crop, second_crop = _extended_source(config)
    first_quiet = source_quiet[first_crop:]
    review_quiet = source_quiet[first_crop + second_crop:]
    contexts = _contexts(config)
    shard_dir = root / "shards" / "original"
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed_pairs = 0
    for first_index, first_context in enumerate(contexts):
        shard_path = shard_dir / f"{first_context.context_id}.json"
        if resume and shard_path.is_file() and json.loads(shard_path.read_text()).get("complete"):
            completed_pairs += len(contexts) * 2
            continue
        print(f"ORIGINAL_SHARD {first_index + 1}/{len(contexts)} START {first_context.context_id}", flush=True)
        p1, i1, first_diagnostics = _first_stage(
            source, source_quiet, first_crop, second_crop, first_context, config,
            bandwidth=float(config["sweep"]["original_msica1_bandwidth"]), seed_offset=10000 + 100 * first_index,
        )
        branch_values = {"persistence": p1, "innovation": i1}
        rows = []
        for second_index, second_context in enumerate(contexts):
            for branch_index, branch in enumerate(config["sweep"]["branches"]):
                combo_id = f"{first_context.context_id}__{second_context.context_id}__{branch}"
                tick = time.monotonic()
                z2, z2_diagnostics = _second_msln(branch_values[branch], first_quiet, second_crop, second_context, config)
                shallow = _label_free_metrics(z2, review_quiet, config)
                winner, deep_p, deep_i, tuning = _tune_msica2(
                    z2, config, lane_id=f"original_{combo_id}", seed_offset=200000 + first_index * 10000 + second_index * 100 + branch_index,
                )
                rows.append({
                    "combination_id": combo_id, "first_context_id": first_context.context_id,
                    "second_context_id": second_context.context_id, "branch": branch,
                    "msica1": first_diagnostics, "msln2": z2_diagnostics,
                    "shallow": shallow, "deep_persistence": _label_free_metrics(deep_p, review_quiet, config),
                    "deep_innovation": _label_free_metrics(deep_i, review_quiet, config),
                    "msica2_selected_fit": _compact_fit(winner), "msica2_tuning": tuning,
                    "runtime_seconds": time.monotonic() - tick,
                })
                _preview(shallow_root / "previews" / first_context.context_id / f"{second_context.context_id}_{branch}.npz", {"shallow": z2}, config)
                _preview(deep_root / "previews" / first_context.context_id / f"{second_context.context_id}_{branch}.npz", {"deep_persistence": deep_p, "deep_innovation": deep_i}, config)
                completed_pairs += 1
                _heartbeat(root, "original_screen", completed_combinations=completed_pairs, total_combinations=len(contexts) ** 2 * 2, combination_id=combo_id)
                del z2, deep_p, deep_i, winner
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
        _write_shard(shard_path, {"first_context_id": first_context.context_id, "rows": rows})
        del p1, i1, branch_values
        gc.collect()
        print(f"ORIGINAL_SHARD {first_index + 1}/{len(contexts)} DONE {first_context.context_id}", flush=True)
    all_rows = _merge_shards(shard_dir)
    expected = len(contexts) ** 2 * 2
    if len(all_rows) != expected:
        raise RuntimeError(f"original screen expected {expected} rows, found {len(all_rows)}")
    shallow_rows = [{"combination_id": r["combination_id"], "first_context_id": r["first_context_id"], "second_context_id": r["second_context_id"], "branch": r["branch"], "metrics": r["shallow"]} for r in all_rows]
    deep_rows = [{"combination_id": r["combination_id"], "first_context_id": r["first_context_id"], "second_context_id": r["second_context_id"], "branch": r["branch"], "persistence_metrics": r["deep_persistence"], "innovation_metrics": r["deep_innovation"], "msica2_selected_fit": r["msica2_selected_fit"], "msica2_tuning": r["msica2_tuning"]} for r in all_rows]
    atomic_json(shallow_root / "stage_a" / "all_context_pair_metrics.json", {"complete": True, "rows": shallow_rows, "combination_count": expected, "selection_labels_used": False})
    atomic_json(deep_root / "stage_a" / "all_context_pair_metrics.json", {"complete": True, "rows": deep_rows, "combination_count": expected, "msica2_bandwidths_tested_per_combination": config["sweep"]["msica2_bandwidth_grid"], "selection_labels_used": False})
    elapsed = time.monotonic() - started
    atomic_json(shallow_root / "status.json", {"status": "screen_complete", "scientific_status": "protected_evaluation_not_run", "elapsed_seconds": elapsed})
    atomic_json(deep_root / "status.json", {"status": "screen_complete", "scientific_status": "protected_evaluation_not_run", "elapsed_seconds": elapsed})
    return {"status": "screen_complete", "original_combinations": expected, "elapsed_seconds": elapsed}


def _raw_first_stage(source: np.ndarray, total_crop: int, config: dict[str, Any], *, seed_offset: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cupy as cp
    device = cp.asarray(source, dtype=cp.float32)
    fit, _, _ = _fit_only(
        "raw_msica1_broad", device[total_crop:], _screen_config(config),
        bandwidth=float(config["sweep"]["switched_msica1_bandwidth"]), seed_offset=seed_offset,
        bootstrap_replicates=0,
    )
    persistence, innovation = apply_per_context_fit_cuda(device, fit)
    result = cp.asnumpy(persistence), cp.asnumpy(innovation), {"fit": _compact_fit(fit), "seed_offset": seed_offset}
    del device, persistence, innovation
    cp.get_default_memory_pool().free_all_blocks()
    return result


def run_switched_screen(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon=authorize_full_spon)
    experiment_root = _experiment_root(root, "02_switched_per_branch")
    if json.loads((experiment_root / "status.json").read_text())["status"] != "preflight_ready" and not resume:
        raise RuntimeError("partial switched screen requires --resume")
    atomic_json(root / "status.json", {"status": "running", "stage": "switched_screen"})
    atomic_json(experiment_root / "status.json", {"status": "running", "stage": "full_context_screen"})
    source, source_quiet, first_crop, second_crop = _extended_source(config)
    total_crop = first_crop + second_crop
    review_quiet = source_quiet[total_crop:]
    raw_p, raw_i, raw_diag = _raw_first_stage(source, total_crop, config, seed_offset=500000)
    branches = {"persistence": raw_p, "innovation": raw_i}
    contexts = _contexts(config)
    shard_dir = root / "shards" / "switched"
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, context in enumerate(contexts):
        path = shard_dir / f"{context.context_id}.json"
        if resume and path.is_file() and json.loads(path.read_text()).get("complete"):
            continue
        rows = []
        for branch_index, branch in enumerate(config["sweep"]["branches"]):
            z, zdiag = _second_msln(branches[branch], source_quiet, total_crop, context, config)
            winner, deep_p, deep_i, tuning = _tune_msica2(z, config, lane_id=f"switched_{context.context_id}_{branch}", seed_offset=600000 + 100 * index + branch_index)
            rows.append({
                "combination_id": f"{context.context_id}__{branch}", "context_id": context.context_id, "branch": branch,
                "raw_msica1": raw_diag, "msln": zdiag, "shallow": _label_free_metrics(z, review_quiet, config),
                "deep_persistence": _label_free_metrics(deep_p, review_quiet, config), "deep_innovation": _label_free_metrics(deep_i, review_quiet, config),
                "msica2_selected_fit": _compact_fit(winner), "msica2_tuning": tuning,
            })
            _preview(experiment_root / "previews" / f"{context.context_id}_{branch}.npz", {"shallow": z, "deep_persistence": deep_p, "deep_innovation": deep_i}, config)
            del z, deep_p, deep_i, winner
            cp.get_default_memory_pool().free_all_blocks()
        _write_shard(path, {"context_id": context.context_id, "rows": rows})
        _heartbeat(root, "switched_screen", completed_contexts=index + 1, total_contexts=len(contexts))
        print(f"SWITCHED {index + 1}/{len(contexts)} DONE {context.context_id}", flush=True)
    rows = _merge_shards(shard_dir)
    atomic_json(experiment_root / "stage_a" / "all_context_metrics.json", {"complete": True, "rows": rows, "combination_count": len(rows), "selection_labels_used": False})
    elapsed = time.monotonic() - started
    atomic_json(experiment_root / "status.json", {"status": "screen_complete", "scientific_status": "protected_evaluation_not_run", "elapsed_seconds": elapsed})
    return {"status": "screen_complete", "combinations": len(rows), "elapsed_seconds": elapsed}


def run_ensemble_screen(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    """Run five raw-MSICA seeds for every context and aggregate output energy."""
    import cupy as cp
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon=authorize_full_spon)
    experiment_root = _experiment_root(root, "03_switched_seed_ensemble")
    if json.loads((experiment_root / "status.json").read_text())["status"] != "preflight_ready" and not resume:
        raise RuntimeError("partial ensemble screen requires --resume")
    atomic_json(experiment_root / "status.json", {"status": "running", "stage": "five_seed_full_context_screen"})
    source, source_quiet, first_crop, second_crop = _extended_source(config)
    total_crop = first_crop + second_crop
    review_quiet = source_quiet[total_crop:]
    contexts = _contexts(config)
    shard_dir = root / "shards" / "ensemble"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for context_index, context in enumerate(contexts):
        path = shard_dir / f"{context.context_id}.json"
        if resume and path.is_file() and json.loads(path.read_text()).get("complete"):
            continue
        shallow_energy, deep_energy, seed_rows = [], [], []
        for seed_index in range(int(config["sweep"]["ensemble_seeds"])):
            raw_p, _, raw_diag = _raw_first_stage(source, total_crop, config, seed_offset=700000 + 1000 * seed_index)
            z, _ = _second_msln(raw_p, source_quiet, total_crop, context, config)
            winner, deep_p, _, tuning = _tune_msica2(z, config, lane_id=f"ensemble_{context.context_id}_{seed_index}", seed_offset=710000 + context_index * 100 + seed_index)
            shallow_energy.append(cp.asnumpy(cp.square(z, dtype=cp.float32)))
            deep_energy.append(cp.asnumpy(cp.square(deep_p, dtype=cp.float32)))
            seed_rows.append({"seed_index": seed_index, "raw_msica1": raw_diag, "msica2": _compact_fit(winner), "selected_bandwidth": tuning["selected_bandwidth"]})
            del raw_p, z, deep_p, winner
            cp.get_default_memory_pool().free_all_blocks()
        shallow_median = np.median(np.stack(shallow_energy), axis=0).astype(np.float32)
        deep_median = np.median(np.stack(deep_energy), axis=0).astype(np.float32)
        shallow_amplitude = np.sqrt(shallow_median, dtype=np.float32)
        deep_amplitude = np.sqrt(deep_median, dtype=np.float32)
        row = {
            "context_id": context.context_id, "seed_count": len(seed_rows), "seeds": seed_rows,
            "shallow_ensemble": _label_free_metrics(shallow_amplitude, review_quiet, config),
            "deep_ensemble": _label_free_metrics(deep_amplitude, review_quiet, config),
            "mean_seed_map_correlation_shallow": _mean_pair_correlation(shallow_energy),
            "mean_seed_map_correlation_deep": _mean_pair_correlation(deep_energy),
        }
        _preview(experiment_root / "previews" / f"{context.context_id}.npz", {"shallow_ensemble": shallow_amplitude, "deep_ensemble": deep_amplitude}, config)
        _write_shard(path, {"context_id": context.context_id, "rows": [row]})
        del shallow_energy, deep_energy, shallow_median, deep_median, shallow_amplitude, deep_amplitude
        gc.collect()
        print(f"ENSEMBLE {context_index + 1}/{len(contexts)} DONE {context.context_id}", flush=True)
    rows = _merge_shards(shard_dir)
    atomic_json(experiment_root / "stage_a" / "all_context_metrics.json", {"complete": True, "rows": rows, "combination_count": len(rows), "selection_labels_used": False})
    atomic_json(experiment_root / "status.json", {"status": "screen_complete", "scientific_status": "protected_evaluation_not_run"})
    return {"status": "screen_complete", "contexts": len(rows)}


def _mean_pair_correlation(maps: list[np.ndarray]) -> float:
    sampled = [np.asarray(values[::11, ::7, ::7], dtype=np.float64).ravel() for values in maps]
    correlations = [float(np.corrcoef(sampled[i], sampled[j])[0, 1]) for i in range(len(sampled)) for j in range(i + 1, len(sampled))]
    return float(np.mean(correlations))


def _cross_branch_fit(zp: Any, zi: Any, config: dict[str, Any], *, seed_offset: int) -> tuple[Any, Any, dict[str, Any]]:
    import cupy as cp
    shape = tuple(map(int, zp.shape))
    total = int(np.prod(shape))
    rng = np.random.default_rng(int(config["sweep"]["seed"]) + seed_offset)
    count = min(int(config["ica"]["confirmation_samples"]), total)
    selected = np.sort(rng.choice(total, count, replace=False))
    t, rem = np.divmod(selected, shape[1] * shape[2])
    y, x = np.divmod(rem, shape[2])
    pairs = cp.asnumpy(cp.stack((zp[cp.asarray(t), cp.asarray(y), cp.asarray(x)], zi[cp.asarray(t), cp.asarray(y), cp.asarray(x)]), axis=1)).astype(np.float64)
    screen_count = min(int(config["ica"]["screen_samples"]), len(pairs))
    screen = pairs[np.sort(rng.choice(len(pairs), screen_count, replace=False))]
    fits, rows = [], []
    for bandwidth in config["sweep"]["msica2_bandwidth_grid"]:
        fit = fit_per_context_ica(
            "cross_branch", screen, pairs, objective="cs_parzen", parzen_bandwidth=float(bandwidth),
            eigenvalue_floor_ratio=float(config["ica"]["eigenvalue_floor_ratio"]), coarse_step_degrees=float(config["ica"]["coarse_step_degrees"]),
            refine_half_width_degrees=float(config["ica"]["refine_half_width_degrees"]), refine_step_degrees=float(config["ica"]["refine_step_degrees"]),
            kernel_block_rows=int(config["ica"]["kernel_block_rows"]), kernel_dtype=np.float32, compute_backend="cuda",
        )
        fits.append((float(bandwidth), fit)); rows.append({"bandwidth": float(bandwidth), **_compact_fit(fit)})
    bandwidth, fit = sorted(fits, key=lambda item: (-_objective_gain(item[1]), item[0]))[0]
    effective = cp.asarray(np.asarray(fit.demixing) @ np.asarray(fit.whitening), dtype=cp.float32)
    out0 = effective[0, 0] * (zp - np.float32(fit.center[0])) + effective[0, 1] * (zi - np.float32(fit.center[1]))
    out1 = effective[1, 0] * (zp - np.float32(fit.center[0])) + effective[1, 1] * (zi - np.float32(fit.center[1]))
    return out0.astype(cp.float32), out1.astype(cp.float32), {"selected_bandwidth": bandwidth, "fit": _compact_fit(fit), "grid": rows}


def run_cross_fusion_screens(config_path: str | Path, *, authorize_full_spon: bool, resume: bool = False) -> dict[str, Any]:
    """Run all-context cross-branch ICA and parallel fusion controls."""
    import cupy as cp
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon=authorize_full_spon)
    cross_root = _experiment_root(root, "04_cross_branch")
    fusion_root = _experiment_root(root, "05_parallel_fusion_control")
    source, source_quiet, first_crop, second_crop = _extended_source(config)
    total_crop = first_crop + second_crop
    review_quiet = source_quiet[total_crop:]
    raw_p, raw_i, _ = _raw_first_stage(source, total_crop, config, seed_offset=800000)
    contexts = _contexts(config)
    cross_rows, fusion_rows = [], []
    for index, context in enumerate(contexts):
        zp, _ = _second_msln(raw_p, source_quiet, total_crop, context, config)
        zi, _ = _second_msln(raw_i, source_quiet, total_crop, context, config)
        out0, out1, cross_fit = _cross_branch_fit(zp, zi, config, seed_offset=810000 + index)
        group_amplitude = cp.sqrt(cp.square(out0, dtype=cp.float32) + cp.square(out1, dtype=cp.float32))
        cross_rows.append({"context_id": context.context_id, "component_0": _label_free_metrics(out0, review_quiet, config), "component_1": _label_free_metrics(out1, review_quiet, config), "group_energy": _label_free_metrics(group_amplitude, review_quiet, config), "fit": cross_fit})
        _preview(cross_root / "previews" / f"{context.context_id}.npz", {"component_0": out0, "component_1": out1, "group_energy": group_amplitude}, config)
        # Release the completed cross phase before allocating the independent
        # control workspace; retaining both exceeds the bounded VRAM cap.
        del zp, zi, out0, out1, group_amplitude
        cp.get_default_memory_pool().free_all_blocks(); gc.collect()
        control_z = causal_joint_msln_cuda(source, context, quiet_mask=source_quiet, review_crop_frames=total_crop, max_vram_bytes=int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)).values
        control_fit, _, _ = _fit_only(f"fusion_control_{context.context_id}", control_z, _screen_config(config), bandwidth=float(config["sweep"]["original_msica1_bandwidth"]), seed_offset=820000 + index, bootstrap_replicates=0)
        control_p, control_i = apply_per_context_fit_cuda(control_z, control_fit)
        ec = cp.square(control_p, dtype=cp.float32)
        qc = max(float(cp.asnumpy(cp.percentile(ec[cp.asarray(review_quiet)], 99.0))), 1e-8)
        nc_cpu = cp.asnumpy(ec / qc)
        del control_z, control_p, control_i, ec
        cp.get_default_memory_pool().free_all_blocks(); gc.collect()
        switched, _ = _second_msln(raw_p, source_quiet, total_crop, context, config)
        es = cp.square(switched, dtype=cp.float32)
        qs = max(float(cp.asnumpy(cp.percentile(es[cp.asarray(review_quiet)], 99.0))), 1e-8)
        nc, ns = cp.asarray(nc_cpu), es / qs
        rules = {"normalized_mean": cp.sqrt((nc + ns) / 2), "geometric_agreement": cp.power(nc * ns, 0.25), "normalized_max": cp.sqrt(cp.maximum(nc, ns))}
        fusion_rows.append({"context_id": context.context_id, "rules": {name: _label_free_metrics(values, review_quiet, config) for name, values in rules.items()}})
        _preview(fusion_root / "previews" / f"{context.context_id}.npz", rules, config)
        del switched, es, nc, ns, nc_cpu, rules
        cp.get_default_memory_pool().free_all_blocks(); gc.collect()
        print(f"CROSS_FUSION {index + 1}/{len(contexts)} DONE {context.context_id}", flush=True)
    atomic_json(cross_root / "stage_a" / "all_context_metrics.json", {"complete": True, "rows": cross_rows, "selection_labels_used": False})
    atomic_json(fusion_root / "stage_a" / "all_context_metrics.json", {"complete": True, "rows": fusion_rows, "selection_labels_used": False})
    atomic_json(cross_root / "status.json", {"status": "screen_complete", "scientific_status": "protected_evaluation_not_run"})
    atomic_json(fusion_root / "status.json", {"status": "screen_complete", "scientific_status": "protected_evaluation_not_run"})
    return {"status": "screen_complete", "cross_contexts": len(cross_rows), "fusion_contexts": len(fusion_rows)}


def summarize(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path)
    root = _root(config)
    payload = {"root": str(root), "status": json.loads((root / "status.json").read_text()) if (root / "status.json").is_file() else None, "experiments": {}}
    for experiment in EXPERIMENTS:
        path = _experiment_root(root, experiment) / "status.json"
        payload["experiments"][experiment] = json.loads(path.read_text()) if path.is_file() else None
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "gpu-preflight", "run-original", "run-switched", "run-ensemble", "run-cross-fusion", "summarize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-full-spon", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "preflight":
        payload = preflight(args.config)
    elif args.action == "gpu-preflight":
        payload = gpu_preflight(args.config)
    elif args.action == "run-original":
        payload = run_original_screen(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    elif args.action == "run-switched":
        payload = run_switched_screen(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    elif args.action == "run-ensemble":
        payload = run_ensemble_screen(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    elif args.action == "run-cross-fusion":
        payload = run_cross_fusion_screens(args.config, authorize_full_spon=args.authorize_full_spon, resume=args.resume)
    else:
        payload = summarize(args.config)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
