"""Resource-bounded causal joint-MSLN, residual-gate, and dual-ICA sweep."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# Numerical libraries read these limits during import, before run() can parse
# the manifest. The v2 manifest freezes four threads; a task-specific override
# exists only for focused testing.
_EARLY_THREAD_LIMIT = os.environ.get("NEUROBENCH_JOINT_THREADS", "4")
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = _EARLY_THREAD_LIMIT

import numpy as np

from neurobench.algorithms.multiscale_local_normalization import (
    JointSTContext,
    causal_joint_msln,
)
from neurobench.algorithms.msln_msica_cuda import (
    apply_per_context_fit_cuda,
    atomic_npy_from_cuda,
    bounded_residual_gate_cuda,
    causal_joint_msln_cuda,
    cuda_device_summary,
    gather_adjacent_pairs_cuda,
)
from neurobench.algorithms.multiscale_subspace import (
    bootstrap_summary,
    contiguous_block_bootstrap,
    fit_per_context_ica,
)
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    known_label_recall_summary,
    temporal_pool,
)
from neurobench.reports.msln_msica_videos import Layer, _render_video

from .artifacts import atomic_json, sha256_file, sha256_payload
from .fitting import adjacent_sample_indices, pairs_at
from .inference import apply_innovation
from .routing import bounded_residual_gate


RAW_DIRECT_ANCHOR = {
    "lane": "raw_direct",
    "mean_recall": 0.6056159420289855,
    "pooled_recall": 0.620253164556962,
    "total_matched": 49,
    "total_labels": 79,
    "total_event_candidates": 232,
    "source": "spon_ca_burst_basic_parzen_ica_diagnostic_v1",
}


def _load(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_id", "source", "sweep", "ica", "evaluation", "compute", "outputs"}
    if set(payload) != required or payload["schema_version"] != 2:
        raise ValueError("joint sweep requires the exact schema-v2 top-level contract")
    root = config_path.parent
    for key in ("movie_path", "labels_path", "raw_direct_metrics_path"):
        payload["source"][key] = str((root / payload["source"][key]).resolve())
    payload["outputs"]["root_dir"] = str((root / payload["outputs"]["root_dir"]).resolve())
    payload["_config_path"] = str(config_path)
    _validate(payload)
    return payload


def _validate(config: dict[str, Any]) -> None:
    source = config["source"]
    sweep = config["sweep"]
    compute = config["compute"]
    if source["axes"] != "TYX" or not source["ui_one_based"]:
        raise ValueError("source must be one-based UI TYX")
    spatial = [tuple(map(int, item)) for item in sweep["spatial_outer_guard_pairs"]]
    temporal = [int(item) for item in sweep["temporal_windows_frames"]]
    if spatial != [(5, 1), (7, 1), (7, 3), (11, 3), (15, 3), (15, 5)]:
        raise ValueError("v2 freezes the six declared spatial outer/guard pairs")
    if temporal != [5, 9, 15, 23, 31]:
        raise ValueError("v2 freezes temporal windows [5,9,15,23,31]")
    if sweep["temporal_guard_frames"] != 1 or sweep["causal"] is not True:
        raise ValueError("joint sweep must remain causal with a one-frame guard")
    if sweep["gate_beta"] != [0.0, 0.25, 0.5] or sweep["gate_kappa"] != [0.5, 1.0, 2.0, 4.0]:
        raise ValueError("v2 freezes the 3x4 bounded residual-gate grid")
    if not 1 <= sweep["shortlist_contexts"] <= 10 or not 1 <= sweep["ica_finalists"] <= sweep["shortlist_contexts"]:
        raise ValueError("invalid shortlist sizes")
    if compute["cpu_threads"] > 8 or compute["max_peak_ram_gb"] > 24 or compute["workers"] != 1:
        raise ValueError("resource caps cannot exceed repository limits")
    if config["ica"]["screen_samples"] > config["ica"]["confirmation_samples"]:
        raise ValueError("ICA screen samples cannot exceed confirmation samples")
    if config["evaluation"]["winner_basis"] != "visual_primary_recall_guardrail":
        raise ValueError("v2 winner basis is frozen to visual-primary")


def _contexts(config: dict[str, Any]) -> list[JointSTContext]:
    result = []
    for outer, guard in config["sweep"]["spatial_outer_guard_pairs"]:
        for window in config["sweep"]["temporal_windows_frames"]:
            result.append(
                JointSTContext(
                    f"joint_s{outer}_g{guard}_t{window}_g1",
                    int(outer),
                    int(guard),
                    int(window),
                    int(config["sweep"]["temporal_guard_frames"]),
                    "mean_std",
                    float(config["sweep"]["scale_floor_percentile"]),
                )
            )
    return result


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    if hasattr(values, "__cuda_array_interface__"):
        atomic_npy_from_cuda(path, values, frame_chunk=8)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    mapped = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=values.dtype, shape=values.shape
    )
    for start in range(0, len(values), 8):
        mapped[start : start + 8] = values[start : start + 8]
    mapped.flush()
    del mapped
    temporary.replace(path)


def _source_view(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    review_start, review_stop = config["source"]["review_interval_ui"]
    pre_roll = max(config["sweep"]["temporal_windows_frames"])
    source_start = review_start - 1 - pre_roll
    if source_start < 0:
        raise ValueError("review interval lacks the required causal pre-roll")
    values = movie[source_start:review_stop]
    quiet_start, quiet_stop = config["source"]["quiet_interval_ui"]
    quiet = np.zeros(len(values), dtype=bool)
    quiet[quiet_start - 1 - source_start : quiet_stop - source_start] = True
    return values, quiet, pre_roll


def _labels(config: dict[str, Any]) -> list[dict[str, Any]]:
    return label_core.load_labels(Path(config["source"]["labels_path"]))


def _recall(
    evidence: np.ndarray,
    labels: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    review_start = int(config["source"]["review_interval_ui"][0])
    budgets = [int(item) for item in config["evaluation"]["candidate_budgets"]]
    rows = []
    totals = {budget: 0 for budget in budgets}
    label_total = 0
    for burst_text, interval in sorted(
        config["source"]["burst_intervals_ui"].items(), key=lambda item: int(item[0])
    ):
        burst = int(burst_text)
        start = int(interval[0]) - review_start
        stop = int(interval[1]) - review_start + 1
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
        label_total += len(burst_labels)
        if hasattr(evidence, "__cuda_array_interface__"):
            import cupy as cp

            pooled = cp.asnumpy(cp.max(evidence[start:stop], axis=0))
        else:
            pooled = temporal_pool(evidence[start:stop], "max")
        peaks = extract_local_maxima(
            pooled,
            int(config["evaluation"]["nms_distance_px"]),
            limit=max(budgets),
        )
        for budget in budgets:
            summary = known_label_recall_summary(
                peaks[:budget],
                burst_labels,
                float(config["evaluation"]["match_radius_px"]),
            )
            totals[budget] += int(summary["matched"])
            rows.append({"burst_id": burst, "budget": budget, **summary})
    return {
        "rows": rows,
        "total_labels": label_total,
        "matched_by_budget": {str(key): value for key, value in totals.items()},
        "recall_by_budget": {
            str(key): value / label_total if label_total else 0.0
            for key, value in totals.items()
        },
        "unmatched_candidates_are": "unknown",
        "comparator_note": "Fixed per-burst budgets are a guardrail, not protocol-identical to Raw Direct quiet-threshold proposals.",
    }


def _visual_stats(
    values: np.ndarray,
    quiet: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    if hasattr(values, "__cuda_array_interface__"):
        import cupy as cp

        quiet_values = values[cp.asarray(quiet)]
        event_blocks = []
        review_start = int(config["source"]["review_interval_ui"][0])
        for interval in config["source"]["burst_intervals_ui"].values():
            start = int(interval[0]) - review_start
            stop = int(interval[1]) - review_start + 1
            event_blocks.append(values[start:stop])
        event = cp.concatenate(event_blocks, axis=0)
        quiet_p99, quiet_p999 = map(
            float, cp.asnumpy(cp.percentile(quiet_values, (99.0, 99.9)))
        )
        event_p99, event_p999 = map(
            float, cp.asnumpy(cp.percentile(event, (99.0, 99.9)))
        )
        quiet_p99 = max(quiet_p99, 1e-8)
        quiet_p999 = max(quiet_p999, 1e-8)
        fraction = float(cp.asnumpy(cp.mean(event > quiet_p999)))
        return {
            "quiet_p99": quiet_p99,
            "quiet_p999": quiet_p999,
            "event_p99": event_p99,
            "event_p999": event_p999,
            "event_quiet_ratio_p99": event_p99 / quiet_p99,
            "event_quiet_ratio_p999": event_p999 / quiet_p999,
            "event_fraction_above_quiet_p999": fraction,
        }
    quiet_values = np.asarray(values[quiet], dtype=np.float32)
    event_blocks = []
    review_start = int(config["source"]["review_interval_ui"][0])
    for interval in config["source"]["burst_intervals_ui"].values():
        start = int(interval[0]) - review_start
        stop = int(interval[1]) - review_start + 1
        event_blocks.append(np.asarray(values[start:stop], dtype=np.float32))
    event = np.concatenate(event_blocks, axis=0)
    quiet_p99 = max(float(np.percentile(quiet_values, 99.0)), 1e-8)
    quiet_p999 = max(float(np.percentile(quiet_values, 99.9)), 1e-8)
    event_p99 = float(np.percentile(event, 99.0))
    event_p999 = float(np.percentile(event, 99.9))
    return {
        "quiet_p99": quiet_p99,
        "quiet_p999": quiet_p999,
        "event_p99": event_p99,
        "event_p999": event_p999,
        "event_quiet_ratio_p99": event_p99 / quiet_p99,
        "event_quiet_ratio_p999": event_p999 / quiet_p999,
        "event_fraction_above_quiet_p999": float(np.mean(event > quiet_p999)),
    }


def _synthetic(ctx: JointSTContext, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frames, size = 72, 31
    yy, xx = np.mgrid[:size, :size]
    compact = np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 4)
    broad = np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 64)
    fixtures = {
        "compact_transient": (compact, "signal"),
        "broad_sustained": (broad, "signal"),
        "moving_edge": ((xx > 15).astype(float), "nuisance"),
        "heteroscedastic_noise": (np.zeros_like(compact), "nuisance"),
        "broad_drift": (np.ones_like(compact), "nuisance"),
        "quiet_null": (np.zeros_like(compact), "nuisance"),
    }
    rows = []
    quiet = np.arange(frames) < 32
    for index, (name, (shape, role)) in enumerate(fixtures.items()):
        video = rng.normal(0, 0.2, size=(frames, size, size))
        if name == "heteroscedastic_noise":
            video *= np.linspace(0.2, 2.0, size)[None, None, :]
        elif name == "moving_edge":
            for frame in range(frames):
                video[frame] += (xx > 5 + frame // 3).astype(float)
        elif name == "broad_drift":
            video += np.linspace(0, 2, frames)[:, None, None]
        elif name != "quiet_null":
            stop = 48 if name == "broad_sustained" else 47
            video[44:stop] += 3 * shape
        result = causal_joint_msln(video.astype(np.float32), ctx, quiet_mask=quiet)
        energy = np.square(result.values, dtype=np.float32)
        truth = shape >= 0.5 * np.max(shape) if np.max(shape) else np.zeros_like(shape, dtype=bool)
        if not np.any(truth) or np.all(truth):
            truth[:] = False
            truth[15, 15] = True
        background = ~truth
        event_map = np.max(energy[44:49], axis=0)
        contrast = float(
            np.mean(event_map[truth])
            / max(float(np.percentile(event_map[background], 95)), 1e-8)
        )
        rows.append({"fixture": name, "role": role, "contrast": contrast})
    signal = np.mean([row["contrast"] for row in rows if row["role"] == "signal"])
    nuisance = np.mean([row["contrast"] for row in rows if row["role"] == "nuisance"])
    return {
        "rows": rows,
        "mean_signal_contrast": float(signal),
        "mean_nuisance_contrast": float(nuisance),
        "signal_to_nuisance_proxy": float(signal / max(nuisance, 1e-8)),
    }


def _preview(
    root: Path,
    context_id: str,
    z: np.ndarray,
    config: dict[str, Any],
) -> None:
    review_start = int(config["source"]["review_interval_ui"][0])
    frames = np.asarray(
        [int(item) - review_start for item in config["outputs"]["representative_frames_ui"]],
        dtype=np.int32,
    )
    on_cuda = hasattr(z, "__cuda_array_interface__")
    if on_cuda:
        import cupy as cp

        selected = cp.asnumpy(z[cp.asarray(frames)]).astype(np.float16)
    else:
        selected = np.asarray(z[frames], dtype=np.float16)
    payload: dict[str, np.ndarray] = {
        "ui_frames": frames + review_start,
        "signed_frames": selected,
        "energy_frames": np.square(selected, dtype=np.float16),
    }
    for burst, interval in config["source"]["burst_intervals_ui"].items():
        start = int(interval[0]) - review_start
        stop = int(interval[1]) - review_start + 1
        if on_cuda:
            signed_map = cp.asnumpy(cp.max(cp.abs(z[start:stop]), axis=0))
            energy_map = cp.asnumpy(
                cp.max(cp.square(z[start:stop], dtype=cp.float32), axis=0)
            )
        else:
            signed_map = np.max(np.abs(z[start:stop]), axis=0)
            energy_map = np.max(np.square(z[start:stop]), axis=0)
        payload[f"burst_{burst}_signed_absmax"] = signed_map.astype(np.float16)
        payload[f"burst_{burst}_energy_max"] = energy_map.astype(np.float16)
    destination = root / "stage_a" / "previews" / f"{context_id}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)


def _fit_lane(
    lane_id: str,
    values: np.ndarray,
    config: dict[str, Any],
    seed_offset: int,
    compute_backend: str = "cpu",
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    valid = np.ones(len(values), dtype=bool)
    confirmation = adjacent_sample_indices(
        values.shape,
        valid,
        count=int(config["ica"]["confirmation_samples"]),
        seed=int(config["sweep"]["seed"]) + seed_offset,
    )
    rng = np.random.default_rng(int(config["sweep"]["seed"]) + seed_offset + 1)
    screen = confirmation[
        np.sort(
            rng.choice(
                len(confirmation),
                size=min(int(config["ica"]["screen_samples"]), len(confirmation)),
                replace=False,
            )
        )
    ]
    pair_reader = (
        gather_adjacent_pairs_cuda
        if hasattr(values, "__cuda_array_interface__")
        else pairs_at
    )
    screen_pairs = pair_reader(values, screen)
    confirmation_pairs = pair_reader(values, confirmation)
    kwargs = {
        "parzen_bandwidth": float(config["ica"]["parzen_bandwidth"]),
        "eigenvalue_floor_ratio": float(config["ica"]["eigenvalue_floor_ratio"]),
        "coarse_step_degrees": float(config["ica"]["coarse_step_degrees"]),
        "refine_half_width_degrees": float(config["ica"]["refine_half_width_degrees"]),
        "refine_step_degrees": float(config["ica"]["refine_step_degrees"]),
        "kernel_block_rows": int(config["ica"]["kernel_block_rows"]),
        "kernel_dtype": np.float32,
        "compute_backend": compute_backend,
    }
    cs = fit_per_context_ica(
        lane_id,
        screen_pairs,
        confirmation_pairs,
        objective="cs_parzen",
        **kwargs,
    )
    fast = fit_per_context_ica(
        lane_id,
        screen_pairs,
        confirmation_pairs,
        objective="fastica",
        **kwargs,
    )
    bootstrap = contiguous_block_bootstrap(
        lane_id,
        confirmation_pairs,
        block_length=int(config["ica"]["bootstrap_block_samples"]),
        replicates=int(config["ica"]["bootstrap_replicates"]),
        seed=int(config["sweep"]["seed"]) + seed_offset + 2,
        fitter_kwargs={"objective": "cs_parzen", **kwargs},
    )
    if compute_backend == "cuda":
        persistence, innovation = apply_per_context_fit_cuda(values, cs)
    else:
        persistence, innovation = apply_innovation(values, cs, valid)
    return persistence, innovation, {
        "cs_parzen": cs.to_dict(),
        "fastica": fast.to_dict(),
    }, bootstrap_summary(bootstrap)


def _rank(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row[score_key]), row["context_id"]))


def _render_stage_a(root: Path, config: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contexts = _contexts(config)
    frames = config["outputs"]["representative_frames_ui"]
    output = root / "diagnostics" / "stage_a"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame_index, ui_frame in enumerate(frames):
        fig, axes = plt.subplots(6, 5, figsize=(15, 17))
        for axis, ctx in zip(axes.ravel(), contexts):
            data = np.load(root / "stage_a" / "previews" / f"{ctx.context_id}.npz")
            values = np.asarray(data["signed_frames"][frame_index], dtype=np.float32)
            limit = max(float(np.percentile(np.abs(values), 99.5)), 1e-6)
            axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set_title(ctx.context_id.replace("joint_", ""), fontsize=8)
            axis.axis("off")
        fig.suptitle(f"Causal joint MSLN screen — UI frame {ui_frame}; per-panel robust signed scale")
        fig.tight_layout()
        path = output / f"joint_bank_ui_{ui_frame}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path.relative_to(root)))
    return paths


def _label_overlay(
    root: Path,
    movie: np.ndarray,
    labels: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, stop = config["source"]["review_interval_ui"]
    indices = np.unique(
        np.linspace(start - 1, stop - 1, min(16, stop - start + 1)).astype(int)
    )
    projection = np.max(np.asarray(movie[indices], dtype=np.float32), axis=0)
    low, high = np.percentile(projection, [1.0, 99.8])
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.imshow(projection, cmap="gray", vmin=low, vmax=high)
    axis.scatter(
        [row["x_px"] for row in labels],
        [row["y_px"] for row in labels],
        s=18,
        facecolors="none",
        edgecolors="tab:red",
    )
    axis.set(
        title="Sparse known-positive projection (preflight)",
        xlabel="x = column",
        ylabel="y = row",
    )
    fig.tight_layout()
    temporary = root / "label_projection_overlay.partial.png"
    fig.savefig(temporary, dpi=120)
    plt.close(fig)
    temporary.replace(root / "label_projection_overlay.png")


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path)
    root = Path(config["outputs"]["root_dir"])
    if root.exists():
        raise FileExistsError(f"Output root already exists: {root}")
    movie_path = Path(config["source"]["movie_path"])
    labels_path = Path(config["source"]["labels_path"])
    metrics_path = Path(config["source"]["raw_direct_metrics_path"])
    for path in (movie_path, labels_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3 or not np.issubdtype(movie.dtype, np.number):
        raise ValueError("movie must be numeric TYX")
    review_start, review_stop = config["source"]["review_interval_ui"]
    if not 1 <= review_start <= review_stop <= len(movie):
        raise ValueError("review interval outside movie")
    labels = _labels(config)
    height, width = movie.shape[1:]
    if any(not (0 <= int(row["x_px"]) < width and 0 <= int(row["y_px"]) < height) for row in labels):
        raise ValueError("label coordinate outside movie")
    for frame_start in range(review_start - 1, review_stop, 8):
        frame_stop = min(frame_start + 8, review_stop)
        if not np.isfinite(
            np.asarray(movie[frame_start:frame_stop], dtype=np.float32)
        ).all():
            raise ValueError("review interval contains non-finite values")
    contexts = _contexts(config)
    review_frames = review_stop - review_start + 1
    map_bytes = review_frames * height * width * 4
    retained_maps = 1 + int(config["sweep"]["ica_finalists"]) * 8
    output_estimate = int(map_bytes * retained_maps + 1.5 * 2**30)
    peak_estimate = int(map_bytes * 5 + max(config["sweep"]["temporal_windows_frames"]) * height * width * 8)
    try:
        import psutil
        vm = psutil.virtual_memory()
        active = [
            {"pid": p.pid, "name": p.info["name"]}
            for p in psutil.process_iter(["name"])
            if "python" in (p.info["name"] or "").lower()
        ]
        memory = {"total_gib": vm.total / 2**30, "available_gib": vm.available / 2**30, "python_processes": active}
    except (ImportError, OSError):
        memory = {"available": False}
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    disk = shutil.disk_usage(ancestor)
    try:
        gpu_text = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu_text = "unavailable"
    tiny = np.random.default_rng(7).normal(size=(40, 19, 19)).astype(np.float32)
    tiny[25:28, 8:11, 8:11] += 5
    tiny_ctx = JointSTContext("joint_smoke", 5, 1, 5, 1)
    smoke = causal_joint_msln(tiny, tiny_ctx, quiet_mask=np.arange(40) < 15)
    if peak_estimate > int(config["compute"]["max_peak_ram_gb"] * 2**30):
        raise RuntimeError("estimated peak RAM exceeds configured cap")
    if output_estimate > disk.free:
        raise RuntimeError("estimated output exceeds free disk")
    baseline = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_lane = next(row for row in baseline["lanes"] if row["lane"] == "raw_direct")
    if int(raw_lane["total_matched"]) != 49:
        raise RuntimeError("Raw Direct anchor no longer matches 49/79")
    fingerprints = {
        "movie": {"sha256": sha256_file(movie_path), "shape": list(movie.shape), "dtype": str(movie.dtype)},
        "labels": {"sha256": sha256_file(labels_path), "rows": len(labels)},
        "raw_direct_metrics": {"sha256": sha256_file(metrics_path)},
    }
    resolved = {key: value for key, value in config.items() if key != "_config_path"}
    fingerprint = sha256_payload({"config": resolved, "inputs": fingerprints})
    resource = {
        "contexts_stage_a": len(contexts),
        "gate_variants_per_shortlist": len(config["sweep"]["gate_beta"]) * len(config["sweep"]["gate_kappa"]),
        "ica_input_lanes_per_finalist": 2,
        "bytes_per_review_float32_map": map_bytes,
        "retained_map_equivalents": retained_maps,
        "output_bytes_estimate": output_estimate,
        "peak_ram_bytes_estimate": peak_estimate,
        "disk_free_bytes": disk.free,
        "memory": memory,
        "gpu": gpu_text,
        "cpu_only": True,
        "one_context_at_a_time": True,
    }
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "config.resolved.json", resolved)
    atomic_json(root / "resource_plan.json", resource)
    atomic_json(root / "preflight.json", {
        "ready": True,
        "preflight_fingerprint": fingerprint,
        "input_fingerprints": fingerprints,
        "source_read_only": True,
        "labels_used_for_fitting": False,
        "raw_direct_anchor": RAW_DIRECT_ANCHOR,
        "tiny_smoke": {
            "finite": bool(np.isfinite(smoke.values).all()),
            "valid_frames": int(smoke.valid_frames.sum()),
            "current_frame_excluded": smoke.diagnostics["current_frame_excluded"],
        },
    })
    atomic_json(root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    _label_overlay(root, movie, labels, config)
    return {"ready": True, "output_root": str(root), "preflight_fingerprint": fingerprint, "resource_plan": resource}


def _check_preflight(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["outputs"]["root_dir"])
    stored = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    if not stored.get("ready"):
        raise RuntimeError("matching ready preflight required")
    resolved = {key: value for key, value in config.items() if key != "_config_path"}
    expected = sha256_payload({"config": resolved, "inputs": stored["input_fingerprints"]})
    if expected != stored["preflight_fingerprint"]:
        raise RuntimeError("configuration does not match preflight")
    return stored


def gpu_preflight(
    config_path: str | Path, *, max_vram_gb: float = 8.0
) -> dict[str, Any]:
    """Validate CUDA parity and one full-context resource bound."""
    import cupy as cp

    config = _load(config_path)
    root = Path(config["outputs"]["root_dir"])
    stored = _check_preflight(config)
    if not 0 < float(max_vram_gb) <= 8:
        raise ValueError("CUDA VRAM cap must lie in (0,8] GiB")
    cap = int(float(max_vram_gb) * 2**30)
    device = cuda_device_summary()
    if device["free_bytes"] < cap:
        raise RuntimeError("free GPU memory is below the selected VRAM cap")

    rng = np.random.default_rng(int(config["sweep"]["seed"]))
    tiny = (1000 + rng.normal(0, 3, size=(48, 31, 33))).astype(np.float32)
    tiny[35:38, 14:18, 15:19] += 20
    tiny_quiet = np.arange(len(tiny)) < 20
    tiny_ctx = JointSTContext("joint_cuda_preflight", 7, 3, 9, 1)
    cpu = causal_joint_msln(tiny, tiny_ctx, quiet_mask=tiny_quiet)
    gpu = causal_joint_msln_cuda(
        tiny,
        tiny_ctx,
        quiet_mask=tiny_quiet,
        review_crop_frames=9,
        max_vram_bytes=min(cap, 2**30),
    )
    gpu_values = cp.asnumpy(gpu.values)
    difference = np.abs(gpu_values - cpu.values[9:])
    tiny_parity = {
        "max_abs": float(np.max(difference)),
        "p99_abs": float(np.percentile(difference, 99)),
        "correlation": float(
            np.corrcoef(gpu_values.ravel(), cpu.values[9:].ravel())[0, 1]
        ),
        "scale_floor_abs": abs(gpu.scale_floor - cpu.scale_floor),
    }
    del gpu, gpu_values
    cp.get_default_memory_pool().free_all_blocks()
    if tiny_parity["max_abs"] > 1e-5 or tiny_parity["correlation"] < 0.999999:
        raise RuntimeError("CUDA joint MSLN failed CPU parity")

    from neurobench.algorithms.pairwise_separation import cs_parzen_objective

    samples = rng.normal(size=(1024, 2))
    samples[:, 1] = 0.6 * samples[:, 0] + 0.8 * samples[:, 1]
    cpu_objective = cs_parzen_objective(
        samples, 0.35, block_rows=256, kernel_dtype=np.float32
    )
    gpu_objective = cs_parzen_objective(
        samples,
        0.35,
        block_rows=256,
        kernel_dtype=np.float32,
        backend="cuda",
    )
    objective_difference = abs(cpu_objective.objective - gpu_objective.objective)
    if objective_difference > 1e-6:
        raise RuntimeError("CUDA CS-Parzen failed CPU parity")

    source, quiet, crop = _source_view(config)
    full_ctx = next(
        item
        for item in _contexts(config)
        if item.context_id == "joint_s15_g3_t31_g1"
    )
    tick = time.monotonic()
    full = causal_joint_msln_cuda(
        source,
        full_ctx,
        quiet_mask=quiet,
        review_crop_frames=crop,
        max_vram_bytes=cap,
    )
    cp.cuda.Stream.null.synchronize()
    full_runtime = time.monotonic() - tick
    stage_a = json.loads(
        (root / "stage_a" / "metrics.json").read_text(encoding="utf-8")
    )
    cpu_row = next(
        row for row in stage_a["rows"] if row["context_id"] == full_ctx.context_id
    )
    floor_relative = abs(full.scale_floor - cpu_row["scale_floor"]) / max(
        abs(cpu_row["scale_floor"]), 1e-12
    )
    full_diagnostics = dict(full.diagnostics)
    del full
    cp.get_default_memory_pool().free_all_blocks()
    if floor_relative > 1e-3:
        raise RuntimeError("full-context CUDA scale floor failed CPU parity")
    payload = {
        "ready": True,
        "compute_backend": "cupy_cuda",
        "max_vram_bytes": cap,
        "device": device,
        "tiny_joint_parity": tiny_parity,
        "cs_parzen_objective_abs_difference": objective_difference,
        "full_context": {
            "context_id": full_ctx.context_id,
            "runtime_seconds": full_runtime,
            "cpu_checkpoint_runtime_seconds": cpu_row["runtime_seconds"],
            "speedup": cpu_row["runtime_seconds"] / full_runtime,
            "scale_floor_relative_difference": floor_relative,
            "diagnostics": full_diagnostics,
        },
        "preflight_fingerprint": stored["preflight_fingerprint"],
        "scientific_parameters_changed": False,
    }
    atomic_json(root / "gpu_validation.json", payload)
    return payload


def run(
    config_path: str | Path,
    *,
    authorize_full_spon: bool,
    resume: bool = False,
    compute_backend: str = "cpu",
    max_vram_gb: float = 8.0,
) -> dict[str, Any]:
    if not authorize_full_spon:
        raise PermissionError("full Spon run requires --authorize-full-spon")
    config = _load(config_path)
    root = Path(config["outputs"]["root_dir"])
    preflight_data = _check_preflight(config)
    if compute_backend not in {"cpu", "cuda"}:
        raise ValueError("compute_backend must be cpu or cuda")
    max_vram_bytes = int(float(max_vram_gb) * 2**30)
    if compute_backend == "cuda":
        if not 0 < float(max_vram_gb) <= 8:
            raise ValueError("CUDA VRAM cap must lie in (0,8] GiB")
        device = cuda_device_summary()
        if device["free_bytes"] < max_vram_bytes:
            raise RuntimeError("free GPU memory is below the selected VRAM cap")
        validation_path = root / "gpu_validation.json"
        if not validation_path.is_file():
            raise RuntimeError("matching gpu-preflight validation is required")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            not validation.get("ready")
            or validation.get("preflight_fingerprint")
            != preflight_data["preflight_fingerprint"]
            or int(validation.get("max_vram_bytes", -1)) != max_vram_bytes
        ):
            raise RuntimeError("GPU validation does not match this run")
        atomic_json(root / "execution_amendment_gpu.json", {
            "reason": "user explicitly selected GPU acceleration after CPU Stage-C proved slow",
            "scientific_parameters_changed": False,
            "compute_backend": "cupy_cuda",
            "max_vram_bytes": max_vram_bytes,
            "device": device,
            "original_preflight_fingerprint": preflight_data["preflight_fingerprint"],
        })
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] == "complete":
        raise FileExistsError("completed output root cannot be overwritten")
    if status["status"] not in {"preflight_ready", "partial", "running"}:
        raise RuntimeError("output root is not resumable")
    if status["status"] != "preflight_ready" and not resume:
        raise RuntimeError("partial run requires --resume")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(config["compute"]["cpu_threads"])
    started = time.monotonic()
    atomic_json(status_path, {"status": "running", "stage": "A", "started_unix": time.time()})
    source, quiet_extended, crop = _source_view(config)
    raw = np.asarray(source[crop:], dtype=np.float32)
    raw_work: Any = raw
    if compute_backend == "cuda":
        import cupy as cp

        raw_work = cp.asarray(raw, dtype=cp.float32)
    quiet = quiet_extended[crop:]
    labels = _labels(config)
    contexts = _contexts(config)
    context_by_id = {item.context_id: item for item in contexts}
    stage_a_path = root / "stage_a" / "metrics.json"
    stage_a_rows = json.loads(stage_a_path.read_text())["rows"] if resume and stage_a_path.is_file() else []
    completed = {row["context_id"] for row in stage_a_rows}
    for index, ctx in enumerate(contexts, 1):
        if ctx.context_id in completed:
            continue
        print(f"STAGE_A {index}/{len(contexts)} START {ctx.context_id}", flush=True)
        tick = time.monotonic()
        if compute_backend == "cuda":
            result = causal_joint_msln_cuda(
                source,
                ctx,
                quiet_mask=quiet_extended,
                review_crop_frames=crop,
                max_vram_bytes=max_vram_bytes,
            )
            z = result.values
            energy = cp.square(z, dtype=cp.float32)
        else:
            result = causal_joint_msln(source, ctx, quiet_mask=quiet_extended)
            z = result.values[crop:]
            energy = np.square(z, dtype=np.float32)
        row = {
            "context_id": ctx.context_id,
            "diagnostics": result.diagnostics,
            "scale_floor": result.scale_floor,
            "visual_stats": _visual_stats(energy, quiet, config),
            "recall_guardrail": _recall(energy, labels, config),
            "synthetic": _synthetic(ctx, int(config["sweep"]["seed"]) + index),
            "runtime_seconds": time.monotonic() - tick,
        }
        row["visual_proxy"] = (
            np.log1p(row["visual_stats"]["event_quiet_ratio_p999"])
            + np.log1p(row["synthetic"]["signal_to_nuisance_proxy"])
        )
        _preview(root, ctx.context_id, z, config)
        stage_a_rows.append(row)
        atomic_json(stage_a_path, {"rows": stage_a_rows, "complete": False})
        print(f"STAGE_A {index}/{len(contexts)} DONE {ctx.context_id} {row['runtime_seconds']:.1f}s", flush=True)
        del result, z, energy
        if compute_backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    ranked_a = _rank(stage_a_rows, "visual_proxy")
    shortlist_ids = []
    for window in config["sweep"]["temporal_windows_frames"]:
        candidates = [row for row in ranked_a if f"_t{window}_" in row["context_id"]]
        if candidates and candidates[0]["context_id"] not in shortlist_ids:
            shortlist_ids.append(candidates[0]["context_id"])
    for row in ranked_a:
        if len(shortlist_ids) >= int(config["sweep"]["shortlist_contexts"]):
            break
        if row["context_id"] not in shortlist_ids:
            shortlist_ids.append(row["context_id"])
    atomic_json(stage_a_path, {
        "rows": stage_a_rows,
        "complete": True,
        "ranking_basis": "visual/morphology proxy only; not a scientific winner",
        "ranked_context_ids": [row["context_id"] for row in ranked_a],
        "shortlist_context_ids": shortlist_ids,
        "raw_direct_anchor": RAW_DIRECT_ANCHOR,
    })
    stage_a_figures = _render_stage_a(root, config)

    atomic_json(status_path, {"status": "running", "stage": "B", "started_unix": time.time()})
    stage_b_path = root / "stage_b" / "gate_metrics.json"
    stored_b = (
        json.loads(stage_b_path.read_text(encoding="utf-8"))
        if resume and stage_b_path.is_file()
        else None
    )
    stage_b_rows = [] if stored_b is None else stored_b.get("rows", [])
    best_gates: dict[str, dict[str, float]] = (
        {} if stored_b is None else stored_b.get("best_gate_by_context", {})
    )
    skip_stage_b = bool(stored_b and stored_b.get("complete"))
    for index, context_id in enumerate(() if skip_stage_b else shortlist_ids, 1):
        print(f"STAGE_B {index}/{len(shortlist_ids)} START {context_id}", flush=True)
        ctx = context_by_id[context_id]
        if compute_backend == "cuda":
            result = causal_joint_msln_cuda(
                source, ctx, quiet_mask=quiet_extended,
                review_crop_frames=crop, max_vram_bytes=max_vram_bytes,
            )
            z = result.values
        else:
            result = causal_joint_msln(source, ctx, quiet_mask=quiet_extended)
            z = result.values[crop:]
        rows = []
        for beta in config["sweep"]["gate_beta"]:
            for kappa in config["sweep"]["gate_kappa"]:
                if compute_backend == "cuda":
                    gate = bounded_residual_gate_cuda(
                        z, beta=float(beta), kappa=float(kappa)
                    )
                    raw_gate = raw_work * gate
                else:
                    gate = bounded_residual_gate(z, beta=float(beta), kappa=float(kappa))
                    raw_gate = np.multiply(raw, gate, dtype=np.float32)
                recall = _recall(raw_gate, labels, config)
                visual = _visual_stats(raw_gate, quiet, config)
                budget = str(config["evaluation"]["guardrail_budget"])
                score = np.log1p(visual["event_quiet_ratio_p999"]) + 0.25 * recall["recall_by_budget"][budget]
                rows.append({
                    "context_id": context_id,
                    "beta": float(beta),
                    "kappa": float(kappa),
                    "visual_stats": visual,
                    "recall_guardrail": recall,
                    "selection_proxy": float(score),
                })
                del gate, raw_gate
        winner = sorted(rows, key=lambda item: (-item["selection_proxy"], item["beta"], item["kappa"]))[0]
        best_gates[context_id] = {"beta": winner["beta"], "kappa": winner["kappa"]}
        stage_b_rows.extend(rows)
        atomic_json(root / "stage_b" / "gate_metrics.json", {"rows": stage_b_rows, "complete": False})
        print(f"STAGE_B {index}/{len(shortlist_ids)} DONE {context_id} beta={winner['beta']} kappa={winner['kappa']}", flush=True)
        del result, z
        if compute_backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    ranked_b = sorted(
        (
            max((row for row in stage_b_rows if row["context_id"] == context_id), key=lambda item: item["selection_proxy"])
            for context_id in shortlist_ids
        ),
        key=lambda row: (-row["selection_proxy"], row["context_id"]),
    )
    finalists = (
        list(stored_b["ica_finalists"])
        if skip_stage_b
        else [row["context_id"] for row in ranked_b[: int(config["sweep"]["ica_finalists"])]]
    )
    if not skip_stage_b:
        atomic_json(root / "stage_b" / "gate_metrics.json", {
        "rows": stage_b_rows,
        "complete": True,
        "best_gate_by_context": best_gates,
        "ica_finalists": finalists,
        "selection_note": "Automated shortlist only; final interpretation is visual-primary.",
        })

    atomic_json(status_path, {"status": "running", "stage": "C", "started_unix": time.time()})
    _atomic_npy(root / "features" / "raw_authority.npy", np.asarray(raw, dtype=np.float32))
    stage_c_path = root / "stage_c" / "ica_metrics.json"
    stored_c = (
        json.loads(stage_c_path.read_text(encoding="utf-8"))
        if resume and stage_c_path.is_file()
        else None
    )
    stage_c_rows = [] if stored_c is None else list(stored_c.get("rows", []))
    completed_c = {row["context_id"] for row in stage_c_rows}
    feature_roots: dict[str, Path] = {
        context_id: root / "features" / context_id
        for context_id in finalists
    }
    for index, context_id in enumerate(finalists, 1):
        if context_id in completed_c:
            print(
                f"STAGE_C {index}/{len(finalists)} REUSE {context_id}",
                flush=True,
            )
            continue
        print(f"STAGE_C {index}/{len(finalists)} START {context_id}", flush=True)
        ctx = context_by_id[context_id]
        if compute_backend == "cuda":
            result = causal_joint_msln_cuda(
                source, ctx, quiet_mask=quiet_extended,
                review_crop_frames=crop, max_vram_bytes=max_vram_bytes,
            )
            z = result.values
        else:
            result = causal_joint_msln(source, ctx, quiet_mask=quiet_extended)
            z = np.asarray(result.values[crop:], dtype=np.float32)
        del result
        beta = best_gates[context_id]["beta"]
        kappa = best_gates[context_id]["kappa"]
        if compute_backend == "cuda":
            gate = bounded_residual_gate_cuda(z, beta=beta, kappa=kappa)
            raw_gate = raw_work * gate
            literal_raw_z = raw_work * z
        else:
            gate = bounded_residual_gate(z, beta=beta, kappa=kappa)
            raw_gate = np.multiply(raw, gate, dtype=np.float32)
            literal_raw_z = np.multiply(raw, z, dtype=np.float32)
        feature_root = feature_roots[context_id]
        for name, values in {
            "zst": z,
            "gate": gate,
            "raw_times_gate": raw_gate,
            "raw_times_zst_signed": literal_raw_z,
        }.items():
            _atomic_npy(feature_root / f"{name}.npy", values)
        del values
        del gate, literal_raw_z
        if compute_backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()

        z_p, z_i, z_fits, z_boot = _fit_lane(
            f"{context_id}_zst", z, config, 100 * index, compute_backend
        )
        lane_rows = {}
        for lane, values in {
            "zst_persistence": z_p,
            "zst_innovation": z_i,
        }.items():
            evidence = (
                cp.square(values, dtype=cp.float32)
                if compute_backend == "cuda"
                else np.square(values, dtype=np.float32)
            )
            lane_rows[lane] = {
                "visual_stats": _visual_stats(evidence, quiet, config),
                "recall_guardrail": _recall(evidence, labels, config),
            }
            _atomic_npy(feature_root / f"{lane}.npy", values)
            del evidence
        del values
        del z_p, z_i
        if compute_backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()

        rg_p, rg_i, rg_fits, rg_boot = _fit_lane(
            f"{context_id}_raw_gate", raw_gate, config, 100 * index + 50,
            compute_backend,
        )
        for lane, values in {
            "raw_gate_persistence": rg_p,
            "raw_gate_innovation": rg_i,
        }.items():
            evidence = (
                cp.square(values, dtype=cp.float32)
                if compute_backend == "cuda"
                else np.square(values, dtype=np.float32)
            )
            lane_rows[lane] = {
                "visual_stats": _visual_stats(evidence, quiet, config),
                "recall_guardrail": _recall(evidence, labels, config),
            }
            _atomic_npy(feature_root / f"{lane}.npy", values)
            del evidence
        del values
        del rg_p, rg_i
        if compute_backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()

        atomic_json(root / "fits" / f"{context_id}.json", {
            "zst": z_fits,
            "raw_gate": rg_fits,
            "bootstrap": {"zst": z_boot, "raw_gate": rg_boot},
        })
        row = {
            "context_id": context_id,
            "gate": {"beta": beta, "kappa": kappa},
            "lanes": lane_rows,
            "zst_fit": z_fits["cs_parzen"],
            "raw_gate_fit": rg_fits["cs_parzen"],
            "bootstrap": {"zst": z_boot, "raw_gate": rg_boot},
        }
        stage_c_rows.append(row)
        atomic_json(stage_c_path, {"rows": stage_c_rows, "complete": False})
        print(f"STAGE_C {index}/{len(finalists)} DONE {context_id}", flush=True)
        del z, raw_gate
        if compute_backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    atomic_json(stage_c_path, {
        "rows": stage_c_rows,
        "complete": True,
        "outputs_preserved_separately": ["persistence", "innovation"],
        "winner_basis": "visual assessment primary; sparse-positive recall is a quantitative guardrail",
        "raw_direct_anchor": RAW_DIRECT_ANCHOR,
    })
    atomic_json(status_path, {"status": "running", "stage": "D", "started_unix": time.time()})
    raw_map = np.load(root / "features" / "raw_authority.npy", mmap_mode="r")
    videos = []
    video_dir = root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    review_start = int(config["source"]["review_interval_ui"][0])
    fps = float(config["outputs"]["fps"])
    for context_id in finalists:
        feature_root = feature_roots[context_id]
        load_map = lambda name: np.load(feature_root / f"{name}.npy", mmap_mode="r")
        layers = [
            Layer("Raw (authority)", raw_map, "raw"),
            Layer("Joint Zst (signed)", load_map("zst"), "signed"),
            Layer("Bounded gate", load_map("gate"), "energy"),
            Layer("Raw x gate", load_map("raw_times_gate"), "interaction"),
            Layer("Raw x Zst (signed diagnostic)", load_map("raw_times_zst_signed"), "signed"),
            Layer("ICA(Zst) persistence", load_map("zst_persistence"), "signed"),
            Layer("ICA(Zst) innovation", load_map("zst_innovation"), "signed"),
            Layer("ICA(Raw x gate) persistence", load_map("raw_gate_persistence"), "signed"),
            Layer("ICA(Raw x gate) innovation", load_map("raw_gate_innovation"), "signed"),
        ]
        videos.append(
            _render_video(
                video_dir / f"{context_id}_layer_journey.mp4",
                layers,
                f"Causal joint residual journey — {context_id}",
                review_start_ui=review_start,
                fps=fps,
                columns=3,
            )
        )
    comparison_layers = [Layer("Raw (authority)", raw_map, "raw")]
    for context_id in finalists:
        feature_root = feature_roots[context_id]
        comparison_layers.extend([
            Layer(f"{context_id} persistence", np.load(feature_root / "raw_gate_persistence.npy", mmap_mode="r"), "signed"),
            Layer(f"{context_id} innovation", np.load(feature_root / "raw_gate_innovation.npy", mmap_mode="r"), "signed"),
        ])
    videos.append(
        _render_video(
            video_dir / "finalist_persistence_innovation_comparison.mp4",
            comparison_layers,
            "Finalist comparison — both sustained and onset coordinates",
            review_start_ui=review_start,
            fps=fps,
            columns=3,
        )
    )
    atomic_json(video_dir / "video_manifest.json", {"videos": videos, "finalists": finalists})
    elapsed = time.monotonic() - started
    manifest = {
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "preflight_fingerprint": preflight_data["preflight_fingerprint"],
        "elapsed_seconds": elapsed,
        "stage_a_contexts": len(contexts),
        "stage_b_gate_fits": len(stage_b_rows),
        "stage_c_ica_fits": len(finalists) * 2,
        "compute_backend": compute_backend,
        "finalists": finalists,
        "stage_a_figures": stage_a_figures,
        "videos": [str((video_dir / row["path"]).relative_to(root)) for row in videos],
        "interpretation": {
            "winner_not_auto_declared": True,
            "visual_assessment_primary": True,
            "recall_is_guardrail": True,
            "persistence_and_innovation_preserved": True,
            "unmatched_candidates_are": "unknown",
        },
    }
    atomic_json(root / "run_manifest.json", manifest)
    atomic_json(status_path, {"status": "complete", "scientific_status": "awaiting_visual_review", "elapsed_seconds": elapsed})
    partial = root / "run_manifest.partial.json"
    if partial.exists():
        partial.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preflight", "gpu-preflight", "run", "summarize")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-full-spon", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compute-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-vram-gb", type=float, default=8.0)
    args = parser.parse_args()
    if args.action == "preflight":
        payload = preflight(args.config)
    elif args.action == "gpu-preflight":
        payload = gpu_preflight(args.config, max_vram_gb=args.max_vram_gb)
    elif args.action == "run":
        payload = run(
            args.config,
            authorize_full_spon=args.authorize_full_spon,
            resume=args.resume,
            compute_backend=args.compute_backend,
            max_vram_gb=args.max_vram_gb,
        )
    else:
        config = _load(args.config)
        root = Path(config["outputs"]["root_dir"])
        payload = {
            "status": json.loads((root / "status.json").read_text()),
            "manifest": json.loads((root / "run_manifest.json").read_text()) if (root / "run_manifest.json").is_file() else None,
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
