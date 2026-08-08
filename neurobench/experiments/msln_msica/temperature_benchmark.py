"""GPU expert benchmark for zero-anchored exp2 MSLN temperature maps."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one
from neurobench.reports.zero_anchored_display import zero_anchored_exp2, zero_anchored_signed

from .artifacts import atomic_json, sha256_file, sha256_payload
from .joint_sweep import _label_overlay


ALPHAS = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
POOLING = ["max", "mean", "top3_mean", "top5_mean"]


def _load(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "experiment_id", "source", "design", "evaluation", "compute", "outputs"}:
        raise ValueError("temperature benchmark requires the exact schema")
    if int(payload["schema_version"]) != 1:
        raise ValueError("unsupported schema version")
    for key in ("raw_path", "msica_path", "msln_path", "labels_path", "raw_direct_metrics_path"):
        payload["source"][key] = str((config_path.parent / payload["source"][key]).resolve())
    payload["outputs"]["root_dir"] = str((config_path.parent / payload["outputs"]["root_dir"]).resolve())
    payload["_config_path"] = str(config_path)
    _validate(payload)
    return payload


def _validate(config: dict[str, Any]) -> None:
    if config["design"]["alphas"] != ALPHAS or config["design"]["pooling"] != POOLING:
        raise ValueError("alpha and pooling grids are frozen")
    if config["design"]["primary_pooling"] != "mean" or config["design"]["controls"] != ["positive_msln"]:
        raise ValueError("primary family and controls are frozen")
    if not config["design"]["zero_anchor"] or not config["design"]["global_normalization"]:
        raise ValueError("zero anchoring and global normalization are required")
    if config["evaluation"]["candidate_budgets"] != [20, 40, 58, 80, 100]:
        raise ValueError("candidate budgets must match prior experiments")
    if config["evaluation"]["protected_selection"] != "leave_one_burst_out":
        raise ValueError("protected selection must be leave-one-burst-out")
    if config["evaluation"]["unmatched_candidates"] != "unknown":
        raise ValueError("unmatched candidates must remain unknown")
    compute = config["compute"]
    if compute["device"] != "cuda" or int(compute["workers_per_gpu"]) != 1:
        raise ValueError("exactly one CUDA worker is required")
    if int(compute["cpu_threads"]) > 4 or not 0 < float(compute["max_peak_vram_gb"]) <= 8:
        raise ValueError("resource limits exceed the guarded contract")


def _resolved(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _root(config: dict[str, Any]) -> Path:
    return Path(config["outputs"]["root_dir"])


def _pool(values: Any, mode: str) -> Any:
    xp = __import__("cupy") if hasattr(values, "__cuda_array_interface__") else np
    if mode == "max":
        return xp.max(values, axis=0)
    if mode == "mean":
        return xp.mean(values, axis=0, dtype=xp.float32)
    if mode.startswith("top"):
        k = int(mode[3])
        partitioned = xp.partition(values, values.shape[0] - k, axis=0)
        return xp.mean(partitioned[-k:], axis=0, dtype=xp.float32)
    raise ValueError(mode)


def _gpu_transform(values: Any, alpha: float, positive_max: float) -> Any:
    import cupy as cp
    positive = cp.clip(values, cp.float32(0.0), cp.float32(positive_max))
    scale = cp.float32(alpha * np.log(2.0))
    denominator = cp.expm1(cp.float64(alpha) * cp.float64(positive_max) * cp.log(cp.float64(2.0)))
    return cp.clip(cp.expm1(scale * positive) / denominator.astype(cp.float32), 0.0, 1.0)


def _lane_id(alpha: float | None, pooling: str) -> str:
    return f"positive_msln__{pooling}" if alpha is None else f"exp2_alpha_{str(alpha).replace('.', 'p')}__{pooling}"


def _evaluate_lane(
    burst_maps: dict[int, np.ndarray], labels: list[dict[str, Any]], config: dict[str, Any],
    *, alpha: float | None, pooling: str, quiet_stats: dict[str, float]
) -> dict[str, Any]:
    budgets = list(map(int, config["evaluation"]["candidate_budgets"]))
    rows, proposals = [], {}
    for burst, score_map in burst_maps.items():
        peaks = extract_local_maxima(score_map, int(config["evaluation"]["nms_distance_px"]), limit=max(budgets))
        proposals[str(burst)] = [[float(score), int(x), int(y)] for score, x, y in peaks]
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
        for budget in budgets:
            matched, _ = match_peaks_one_to_one(peaks[:budget], burst_labels, float(config["evaluation"]["match_radius_px"]))
            rows.append({"burst_id": burst, "budget": budget, "matched": len(matched), "labels": len(burst_labels), "candidates": min(budget, len(peaks)), "recall": len(matched) / len(burst_labels)})
    return {
        "lane_id": _lane_id(alpha, pooling), "alpha": alpha, "pooling": pooling,
        "rows": rows, "proposals": proposals, "unsupervised": quiet_stats,
        "unmatched_candidates_are": "unknown",
    }


def _protected(lanes: list[dict[str, Any]], config: dict[str, Any], *, family: str | None) -> dict[str, Any]:
    budget = int(config["evaluation"]["guardrail_budget"])
    selected = [lane for lane in lanes if family is None or lane["pooling"] == family]
    folds, total_matched, total_labels = [], 0, 0
    for holdout in (1, 2, 3, 4):
        def training_key(lane: dict[str, Any]) -> tuple[float, float, float, str]:
            rows = [row for row in lane["rows"] if row["budget"] == budget and row["burst_id"] != holdout]
            recall = float(np.mean([row["recall"] for row in rows]))
            unsup = float(lane["unsupervised"]["event_quiet_ratio"])
            alpha_distance = abs(float(lane["alpha"] or 0.0) - 0.175)
            return (-recall, -unsup, alpha_distance, lane["lane_id"])
        winner = sorted(selected, key=training_key)[0]
        test = next(row for row in winner["rows"] if row["budget"] == budget and row["burst_id"] == holdout)
        total_matched += int(test["matched"]); total_labels += int(test["labels"])
        folds.append({"held_out_burst": holdout, "selected_lane": winner["lane_id"], "training_mean_recall": -training_key(winner)[0], "matched": test["matched"], "labels": test["labels"], "recall": test["recall"]})
    return {"family": family or "all", "budget": budget, "folds": folds, "matched": total_matched, "labels": total_labels, "pooled_recall": total_matched / total_labels}


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = _load(config_path); root = _root(config)
    if root.exists():
        raise FileExistsError(root)
    for key in ("raw_path", "msica_path", "msln_path", "labels_path", "raw_direct_metrics_path"):
        if not Path(config["source"][key]).is_file():
            raise FileNotFoundError(config["source"][key])
    raw = np.load(config["source"]["raw_path"], mmap_mode="r", allow_pickle=False)
    msica = np.load(config["source"]["msica_path"], mmap_mode="r", allow_pickle=False)
    msln = np.load(config["source"]["msln_path"], mmap_mode="r", allow_pickle=False)
    if raw.shape != (2359, 340, 573) or raw.dtype != np.uint16:
        raise ValueError("unexpected raw movie contract")
    if msica.shape != msln.shape or msln.shape != (560, 340, 573) or msln.dtype != np.float32:
        raise ValueError("unexpected frozen stage-array contract")
    labels = label_core.load_labels(Path(config["source"]["labels_path"]))
    if len(labels) != 79:
        raise ValueError("expected 79 sparse-positive occurrences")
    root.mkdir(parents=True, exist_ok=False)
    _label_overlay(root, raw, labels, {"source": config["source"]})
    fingerprints = {key: sha256_file(Path(config["source"][key])) for key in ("raw_path", "msica_path", "msln_path", "labels_path", "raw_direct_metrics_path")}
    payload = {
        "status": "ready", "authorized_run_required": True, "shape_tyx": list(msln.shape),
        "lane_count": (len(ALPHAS) + 1) * len(POOLING), "label_count": len(labels),
        "estimated_peak_vram_bytes": int(msln.nbytes * 2.25), "estimated_retained_bytes": int(4 * max(config["evaluation"]["candidate_budgets"]) * 3 * 8 * (len(ALPHAS) + 1) * len(POOLING)),
        "max_pool_invariance_control": "all monotone alpha arms must have identical rankings under max pooling",
        "fingerprints": fingerprints, "config_sha256": sha256_payload(_resolved(config)),
    }
    atomic_json(root / "config.resolved.json", _resolved(config)); atomic_json(root / "preflight.json", payload)
    atomic_json(root / "status.json", {"status": "preflight_ready"})
    return payload


def gpu_preflight(config_path: str | Path) -> dict[str, Any]:
    import cupy as cp
    config = _load(config_path); root = _root(config)
    pre = json.loads((root / "preflight.json").read_text())
    if pre["config_sha256"] != sha256_payload(_resolved(config)):
        raise RuntimeError("preflight/config mismatch")
    rng = np.random.default_rng(20260808); sample = rng.normal(size=(7, 19, 23)).astype(np.float32)
    positive_max = float(np.max(sample))
    rows = []
    for alpha in (0.025, 0.175, 1.0):
        cpu = zero_anchored_exp2(sample, alpha=alpha, positive_max=positive_max)
        gpu = cp.asnumpy(_gpu_transform(cp.asarray(sample), alpha, positive_max))
        rows.append({"alpha": alpha, "max_abs_error": float(np.max(np.abs(cpu - gpu)))})
    free, total = cp.cuda.runtime.memGetInfo()
    passed = max(row["max_abs_error"] for row in rows) <= 2e-6 and pre["estimated_peak_vram_bytes"] <= int(config["compute"]["max_peak_vram_gb"] * 2**30) and free > pre["estimated_peak_vram_bytes"]
    payload = {"status": "passed" if passed else "failed", "parity": rows, "cuda_free_bytes": int(free), "cuda_total_bytes": int(total), "estimated_peak_vram_bytes": pre["estimated_peak_vram_bytes"]}
    atomic_json(root / "gpu_preflight.json", payload)
    if not passed:
        raise RuntimeError(payload)
    return payload


def run(config_path: str | Path, *, authorize_full_spon: bool) -> dict[str, Any]:
    if not authorize_full_spon:
        raise PermissionError("full Spon run requires --authorize-full-spon")
    import cupy as cp
    config = _load(config_path); root = _root(config)
    if json.loads((root / "gpu_preflight.json").read_text())["status"] != "passed":
        raise RuntimeError("passed GPU preflight required")
    atomic_json(root / "status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    msln = np.load(config["source"]["msln_path"], mmap_mode="r", allow_pickle=False)
    labels = label_core.load_labels(Path(config["source"]["labels_path"]))
    positive_max = float(np.max(msln)); gpu_values = cp.asarray(msln, dtype=cp.float32)
    review_start = int(config["source"]["review_interval_ui"][0]); quiet_stop = int(config["source"]["quiet_interval_ui"][1]) - review_start + 1
    lanes = []; started = time.monotonic()
    for alpha in [None] + ALPHAS:
        transformed = cp.clip(gpu_values / cp.float32(positive_max), 0.0, 1.0) if alpha is None else _gpu_transform(gpu_values, alpha, positive_max)
        quiet_p99 = float(cp.percentile(transformed[:quiet_stop], 99.0).get())
        for pooling in POOLING:
            burst_maps = {}
            event_values = []
            for burst_text, interval in config["source"]["burst_intervals_ui"].items():
                start = int(interval[0]) - review_start; stop = int(interval[1]) - review_start + 1
                selected = transformed[start:stop]
                pooled = cp.asnumpy(_pool(selected, pooling)).astype(np.float32, copy=False)
                burst_maps[int(burst_text)] = pooled
                event_values.append(float(cp.mean(selected).get()))
            quiet_mean = float(cp.mean(transformed[:quiet_stop]).get())
            stats = {"quiet_p99": quiet_p99, "quiet_mean": quiet_mean, "event_mean": float(np.mean(event_values)), "event_quiet_ratio": float(np.mean(event_values)) / max(quiet_mean, 1e-12), "fraction_above_quiet_p99": float(cp.mean(transformed > quiet_p99).get())}
            lane = _evaluate_lane(burst_maps, labels, config, alpha=alpha, pooling=pooling, quiet_stats=stats)
            lanes.append(lane)
            atomic_json(root / "lanes" / f"{lane['lane_id']}.json", lane)
        atomic_json(root / "progress.json", {"status": "running", "completed_temperature_groups": len(lanes) // len(POOLING), "total_temperature_groups": len(ALPHAS) + 1, "elapsed_seconds": time.monotonic() - started})
    cp.get_default_memory_pool().free_all_blocks()
    protected = {"primary_mean": _protected(lanes, config, family="mean"), "all_pooling_secondary": _protected(lanes, config, family=None)}
    budget = int(config["evaluation"]["guardrail_budget"])
    for lane in lanes:
        rows = [row for row in lane["rows"] if row["budget"] == budget]
        lane["guardrail_matches"] = int(sum(row["matched"] for row in rows))
        lane["guardrail_macro_recall"] = float(np.mean([row["recall"] for row in rows]))
    best = sorted(lanes, key=lambda row: (-row["guardrail_matches"], -row["unsupervised"]["event_quiet_ratio"], row["lane_id"]))[0]
    max_lanes = [lane for lane in lanes if lane["pooling"] == "max"]
    reference = [[p[1:] for p in max_lanes[0]["proposals"][str(b)]] for b in (1, 2, 3, 4)]
    invariant = all([[p[1:] for p in lane["proposals"][str(b)]] for b in (1, 2, 3, 4)] == reference for lane in max_lanes[1:])
    payload = {"status": "complete", "lane_count": len(lanes), "positive_msln_global_max": positive_max, "protected": protected, "posthoc_best_lane": best["lane_id"], "posthoc_best_matches": best["guardrail_matches"], "max_pool_ranking_invariance_passed": invariant, "raw_direct_external_anchor_matches": 49, "labels": 79, "runtime_seconds": time.monotonic() - started}
    atomic_json(root / "results.json", {"summary": payload, "lanes": lanes}); atomic_json(root / "status.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("preflight", "gpu-preflight", "run")); parser.add_argument("--config", required=True); parser.add_argument("--authorize-full-spon", action="store_true")
    args = parser.parse_args()
    payload = preflight(args.config) if args.action == "preflight" else gpu_preflight(args.config) if args.action == "gpu-preflight" else run(args.config, authorize_full_spon=args.authorize_full_spon)
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
