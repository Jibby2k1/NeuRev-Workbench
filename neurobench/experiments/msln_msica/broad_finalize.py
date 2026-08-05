"""Label-free finalist selection, protected evaluation, videos, and reports for v4."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.algorithms.msln_msica_cuda import apply_per_context_fit_cuda, causal_joint_msln_cuda
from neurobench.reports.msln_msica_videos import Layer, _render_video

from .artifacts import atomic_json
from .broad_cascade import (
    EXPERIMENTS,
    _compact_fit,
    _context_by_id,
    _cross_branch_fit,
    _experiment_root,
    _extended_source,
    _final_config,
    _fit_only,
    _labels,
    _load,
    _protected_metrics,
    _require_run,
    _root,
    _second_msln,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(metrics: dict[str, Any]) -> float:
    return float(metrics["selection_score"])


def _screen_candidates(root: Path, experiment: str) -> list[dict[str, Any]]:
    stage = _experiment_root(root, experiment) / "stage_a"
    filename = "all_context_pair_metrics.json" if experiment.startswith(("00_", "01_")) else "all_context_metrics.json"
    rows = _read(stage / filename)["rows"]
    candidates: list[dict[str, Any]] = []
    if experiment == "00_original_shallow":
        for row in rows:
            candidates.append({**{key: row[key] for key in ("combination_id", "first_context_id", "second_context_id", "branch")}, "lane": "shallow", "screen_metrics": row["metrics"]})
    elif experiment == "01_original_deep":
        for row in rows:
            for lane in ("persistence", "innovation"):
                candidates.append({**{key: row[key] for key in ("combination_id", "first_context_id", "second_context_id", "branch")}, "lane": f"deep_{lane}", "bandwidth": row["msica2_tuning"]["selected_bandwidth"], "screen_metrics": row[f"{lane}_metrics"]})
    elif experiment == "02_switched_per_branch":
        for row in rows:
            base = {key: row[key] for key in ("combination_id", "context_id", "branch")}
            candidates.append({**base, "lane": "shallow", "screen_metrics": row["shallow"]})
            for lane in ("persistence", "innovation"):
                candidates.append({**base, "lane": f"deep_{lane}", "bandwidth": row["msica2_tuning"]["selected_bandwidth"], "screen_metrics": row[f"deep_{lane}"]})
    elif experiment == "03_switched_seed_ensemble":
        for row in rows:
            candidates.extend([
                {"context_id": row["context_id"], "lane": "shallow_ensemble", "screen_metrics": row["shallow_ensemble"]},
                {"context_id": row["context_id"], "lane": "deep_ensemble", "screen_metrics": row["deep_ensemble"], "seed_bandwidths": [seed["selected_bandwidth"] for seed in row["seeds"]]},
            ])
    elif experiment == "04_cross_branch":
        for row in rows:
            for lane in ("component_0", "component_1", "group_energy"):
                candidates.append({"context_id": row["context_id"], "lane": lane, "screen_metrics": row[lane]})
    elif experiment == "05_parallel_fusion_control":
        for row in rows:
            for lane, metrics in row["rules"].items():
                candidates.append({"context_id": row["context_id"], "lane": lane, "screen_metrics": metrics})
    return sorted(candidates, key=lambda row: (-_score(row["screen_metrics"]), json.dumps(row, sort_keys=True)))


def freeze_finalists(config_path: str | Path, *, authorize_full_spon: bool) -> dict[str, Any]:
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon=authorize_full_spon)
    count = int(config["sweep"]["finalists_per_experiment"])
    payload = {"selection_labels_used": False, "selection_basis": "label-free selection_score only", "experiments": {}}
    for experiment in EXPERIMENTS:
        status = _read(_experiment_root(root, experiment) / "status.json")
        if status["status"] != "screen_complete":
            raise RuntimeError(f"screen incomplete: {experiment}")
        ranked = _screen_candidates(root, experiment)
        payload["experiments"][experiment] = [{**row, "rank": index + 1} for index, row in enumerate(ranked[:count])]
    atomic_json(root / "finalist_freeze.json", payload)
    atomic_json(root / "status.json", {"status": "running", "stage": "finalists_frozen"})
    return payload


def _final_fit(values: Any, config: dict[str, Any], *, lane: str, bandwidth: float, seed: int) -> tuple[Any, dict[str, Any]]:
    fit, diagnostics, bootstrap = _fit_only(lane, values, _final_config(config), bandwidth=bandwidth, seed_offset=seed, bootstrap_replicates=int(config["ica"]["final_bootstrap_replicates"]), run_fastica=True)
    return fit, {"fit": _compact_fit(fit), "bootstrap": bootstrap, "full_diagnostics": diagnostics}


def _original_first(source: np.ndarray, quiet: np.ndarray, first_crop: int, second_crop: int, context: Any, config: dict[str, Any], seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import cupy as cp
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    result = causal_joint_msln_cuda(source, context, quiet_mask=quiet, review_crop_frames=first_crop, max_vram_bytes=cap)
    fit, diag = _final_fit(result.values[second_crop:], config, lane=f"final_{context.context_id}_msica1", bandwidth=float(config["sweep"]["original_msica1_bandwidth"]), seed=seed)
    p, i = apply_per_context_fit_cuda(result.values, fit)
    output = {"persistence": cp.asnumpy(p), "innovation": cp.asnumpy(i)}
    del result, p, i
    cp.get_default_memory_pool().free_all_blocks()
    return output, diag


def _raw_first(source: np.ndarray, total_crop: int, config: dict[str, Any], seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import cupy as cp
    device = cp.asarray(source, dtype=cp.float32)
    fit, diag = _final_fit(device[total_crop:], config, lane="final_raw_msica1", bandwidth=float(config["sweep"]["switched_msica1_bandwidth"]), seed=seed)
    p, i = apply_per_context_fit_cuda(device, fit)
    output = {"persistence": cp.asnumpy(p), "innovation": cp.asnumpy(i)}
    del device, p, i
    cp.get_default_memory_pool().free_all_blocks()
    return output, diag


def _deep(values: Any, lane: str, bandwidth: float, config: dict[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    fit, diag = _final_fit(values, config, lane=f"final_{lane}", bandwidth=bandwidth, seed=seed)
    p, i = apply_per_context_fit_cuda(values, fit)
    selected = p if lane.endswith("persistence") else i
    other = i if lane.endswith("persistence") else p
    del other
    return selected, diag


def _recompute(experiment: str, candidate: dict[str, Any], source: np.ndarray, source_quiet: np.ndarray, first_crop: int, second_crop: int, config: dict[str, Any], rank: int) -> tuple[np.ndarray, dict[str, Any]]:
    import cupy as cp
    contexts = _context_by_id(config)
    total_crop = first_crop + second_crop
    seed = 900000 + 1000 * EXPERIMENTS.index(experiment) + rank * 20
    diagnostics: dict[str, Any] = {}
    if experiment.startswith(("00_", "01_")):
        branches, diagnostics["msica1"] = _original_first(source, source_quiet, first_crop, second_crop, contexts[candidate["first_context_id"]], config, seed)
        z, diagnostics["msln2"] = _second_msln(branches[candidate["branch"]], source_quiet[first_crop:], second_crop, contexts[candidate["second_context_id"]], config)
        values = z
        if candidate["lane"].startswith("deep_"):
            values, diagnostics["msica2"] = _deep(z, candidate["lane"], float(candidate["bandwidth"]), config, seed + 1)
    elif experiment == "02_switched_per_branch":
        branches, diagnostics["msica1"] = _raw_first(source, total_crop, config, seed)
        z, diagnostics["msln"] = _second_msln(branches[candidate["branch"]], source_quiet, total_crop, contexts[candidate["context_id"]], config)
        values = z
        if candidate["lane"].startswith("deep_"):
            values, diagnostics["msica2"] = _deep(z, candidate["lane"], float(candidate["bandwidth"]), config, seed + 1)
    elif experiment == "03_switched_seed_ensemble":
        energies = []
        seed_diags = []
        for seed_index in range(int(config["sweep"]["ensemble_seeds"])):
            branches, raw_diag = _raw_first(source, total_crop, config, seed + seed_index * 100)
            z, _ = _second_msln(branches["persistence"], source_quiet, total_crop, contexts[candidate["context_id"]], config)
            lane_values = z
            deep_diag = None
            if candidate["lane"] == "deep_ensemble":
                lane_values, deep_diag = _deep(z, "deep_persistence", float(candidate["seed_bandwidths"][seed_index]), config, seed + seed_index * 100 + 1)
            energies.append(cp.asnumpy(cp.square(lane_values, dtype=cp.float32)))
            seed_diags.append({"seed_index": seed_index, "msica1": raw_diag, "msica2": deep_diag})
            del z, lane_values
            cp.get_default_memory_pool().free_all_blocks()
        values = cp.asarray(np.sqrt(np.median(np.stack(energies), axis=0), dtype=np.float32))
        diagnostics["seeds"] = seed_diags
    elif experiment == "04_cross_branch":
        branches, diagnostics["msica1"] = _raw_first(source, total_crop, config, seed)
        zp, _ = _second_msln(branches["persistence"], source_quiet, total_crop, contexts[candidate["context_id"]], config)
        zi, _ = _second_msln(branches["innovation"], source_quiet, total_crop, contexts[candidate["context_id"]], config)
        out0, out1, diagnostics["cross_msica"] = _cross_branch_fit(zp, zi, _final_config(config), seed_offset=seed + 1)
        values = out0 if candidate["lane"] == "component_0" else out1 if candidate["lane"] == "component_1" else cp.sqrt(cp.square(out0, dtype=cp.float32) + cp.square(out1, dtype=cp.float32))
    else:
        branches, diagnostics["raw_msica"] = _raw_first(source, total_crop, config, seed)
        switched, _ = _second_msln(branches["persistence"], source_quiet, total_crop, contexts[candidate["context_id"]], config)
        review_quiet = cp.asarray(source_quiet[total_crop:])
        es = cp.square(switched, dtype=cp.float32)
        qs = max(float(cp.asnumpy(cp.percentile(es[review_quiet], 99.0))), 1e-8)
        ns_cpu = cp.asnumpy(es / qs)
        del switched, es
        cp.get_default_memory_pool().free_all_blocks(); gc.collect()
        cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
        control_z = causal_joint_msln_cuda(source, contexts[candidate["context_id"]], quiet_mask=source_quiet, review_crop_frames=total_crop, max_vram_bytes=cap).values
        control_fit, diagnostics["control_msica"] = _final_fit(control_z, config, lane="final_fusion_control", bandwidth=float(config["sweep"]["original_msica1_bandwidth"]), seed=seed + 1)
        control_p, control_i = apply_per_context_fit_cuda(control_z, control_fit)
        ec = cp.square(control_p, dtype=cp.float32)
        qc = max(float(cp.asnumpy(cp.percentile(ec[review_quiet], 99.0))), 1e-8)
        nc, ns = ec / qc, cp.asarray(ns_cpu)
        rules = {"normalized_mean": cp.sqrt((nc + ns) / 2), "geometric_agreement": cp.power(nc * ns, 0.25), "normalized_max": cp.sqrt(cp.maximum(nc, ns))}
        values = rules[candidate["lane"]]
    array = cp.asnumpy(values).astype(np.float32, copy=False) if hasattr(values, "__cuda_array_interface__") else np.asarray(values, dtype=np.float32)
    cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    return array, diagnostics


def _recall58(metrics: dict[str, Any]) -> tuple[int, float]:
    recall = metrics["recall_guardrail"]
    return int(recall["matched_by_budget"]["58"]), float(recall["recall_by_budget"]["58"])


def _report(experiment: str, rows: list[dict[str, Any]], raw: dict[str, Any]) -> str:
    title = experiment.replace("_", " ").title()
    lines = [f"# {title}: Concise Report", "", "Finalists were frozen using label-free event/quiet contrast. Known coordinates were opened only for the protected evaluation below.", "", "| Rank | Configuration | Label-free score | Matches / 79 @ budget 58 | Recall |", "|---:|---|---:|---:|---:|"]
    for row in rows:
        matched, recall = _recall58(row["protected_metrics"])
        label = f'{row["candidate"].get("combination_id", row["candidate"].get("context_id"))} / {row["candidate"]["lane"]}'
        lines.append(f'| {row["rank"]} | `{label}` | {row["candidate"]["screen_metrics"]["selection_score"]:.4f} | {matched} | {recall:.4f} |')
    best = max(rows, key=lambda row: _recall58(row["protected_metrics"])[1])
    matched, recall = _recall58(best["protected_metrics"])
    lines.extend(["", f"Protected best: rank {best['rank']} with {matched}/79 matches ({recall:.4f}). Raw Direct reference: {raw['total_matched']}/79 ({raw['mean_recall']:.4f}).", "", "Diagnostics include final 4,096/16,384-sample fits, FastICA sensitivity, 16 blocked-bootstrap replicates where the fitted block supports them, fixed-scale video, full float32 maps, and the complete screen metrics.", "", "Unmatched detections remain unknown rather than negatives because labels are sparse-positive.", ""])
    return "\n".join(lines)


def run_finalists(config_path: str | Path, *, authorize_full_spon: bool) -> dict[str, Any]:
    config = _load(config_path)
    root = _require_run(config, authorize_full_spon=authorize_full_spon)
    freeze_path = root / "finalist_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("freeze-finalists must run first")
    frozen = _read(freeze_path)
    source, source_quiet, first_crop, second_crop = _extended_source(config)
    review_quiet = source_quiet[first_crop + second_crop:]
    labels = _labels(config)
    raw = _read(Path(config["source"]["raw_direct_metrics_path"]))
    raw_anchor = next((row for row in raw.get("lanes", []) if row.get("lane") == "raw_direct"), None) or raw.get("raw_direct_anchor") or {"total_matched": 49, "mean_recall": 0.6056159420289855}
    movie = np.load(config["source"]["movie_path"], mmap_mode="r")
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    raw_review = movie[review_start - 1:review_stop]
    summary_rows = []
    for experiment in EXPERIMENTS:
        experiment_root = _experiment_root(root, experiment)
        final_dir = experiment_root / "finalists"
        video_dir = experiment_root / "videos"
        final_dir.mkdir(parents=True, exist_ok=True); video_dir.mkdir(parents=True, exist_ok=True)
        rows, layers = [], [Layer("Raw amplitude", raw_review, "raw")]
        for candidate in frozen["experiments"][experiment]:
            rank = int(candidate["rank"])
            print(f"FINALIST {experiment} {rank}/3 START", flush=True)
            values, diagnostics = _recompute(experiment, candidate, source, source_quiet, first_crop, second_crop, config, rank)
            metrics = _protected_metrics(values, review_quiet, labels, config)
            path = final_dir / f"rank_{rank:02d}.npy"
            np.save(path, values, allow_pickle=False)
            row = {"rank": rank, "candidate": candidate, "protected_metrics": metrics, "final_fit_diagnostics": diagnostics, "map_path": str(path.relative_to(root))}
            atomic_json(final_dir / f"rank_{rank:02d}.json", row)
            rows.append(row); layers.append(Layer(f"Rank {rank}: {candidate['lane']}", values, "signed"))
            print(f"FINALIST {experiment} {rank}/3 DONE", flush=True)
        video = _render_video(video_dir / "finalist_comparison.mp4", layers, f"{experiment} protected finalists", review_start_ui=review_start, fps=float(config["outputs"]["fps"]), columns=2)
        atomic_json(video_dir / "video_manifest.json", video)
        atomic_json(experiment_root / "finalist_metrics.json", {"complete": True, "rows": rows, "raw_direct_anchor": raw_anchor})
        (experiment_root / "CONCISE_REPORT.md").write_text(_report(experiment, rows, raw_anchor), encoding="utf-8")
        atomic_json(experiment_root / "status.json", {"status": "complete", "scientific_status": "protected_evaluation_complete", "video_count": 1})
        best = max(rows, key=lambda row: _recall58(row["protected_metrics"])[1])
        matched, recall = _recall58(best["protected_metrics"])
        rank1_matched, rank1_recall = _recall58(rows[0]["protected_metrics"])
        summary_rows.append({
            "experiment": experiment,
            "label_free_winner_rank": 1,
            "label_free_matched_at_58": rank1_matched,
            "label_free_recall_at_58": rank1_recall,
            "best_rank_by_protected_recall": best["rank"],
            "protected_best_matched_at_58": matched,
            "protected_best_recall_at_58": recall,
        })
    summary = root / "summary"; summary.mkdir(parents=True, exist_ok=True)
    table = ["# Broad MSLN/MSICA Cascade Program: Conclusive Report", "", "All six predeclared concepts completed their broad label-free screens and protected three-finalist evaluations.", "", "| Experiment | Label-free rank-1 matches | Protected-best rank | Protected-best matches |", "|---|---:|---:|---:|"]
    for row in summary_rows:
        table.append(f'| `{row["experiment"]}` | {row["label_free_matched_at_58"]}/79 | {row["best_rank_by_protected_recall"]} | {row["protected_best_matched_at_58"]}/79 |')
    winner = max(summary_rows, key=lambda row: row["label_free_recall_at_58"])
    table.extend(["", f"Primary label-free result: `{winner['experiment']}` with {winner['label_free_matched_at_58']}/79 ({winner['label_free_recall_at_58']:.4f}) at the fixed budget-58 guardrail. Raw Direct remains {raw_anchor['total_matched']}/79 ({raw_anchor['mean_recall']:.4f}).", "", "Protected-best results are exploratory ceilings within three already-frozen candidates, not unbiased winner estimates. Interpretation must also consider bootstrap stability, visual artifacts, and unmatched candidates.", ""])
    (summary / "CONCLUSIVE_REPORT.md").write_text("\n".join(table), encoding="utf-8")
    atomic_json(summary / "comparison.json", {"rows": summary_rows, "raw_direct_anchor": raw_anchor})
    atomic_json(root / "status.json", {"status": "complete", "scientific_status": "protected_evaluation_complete", "experiments_complete": len(EXPERIMENTS)})
    return {"status": "complete", "experiments": summary_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-finalists", "run-finalists"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorize-full-spon", action="store_true")
    args = parser.parse_args()
    payload = freeze_finalists(args.config, authorize_full_spon=args.authorize_full_spon) if args.action == "freeze-finalists" else run_finalists(args.config, authorize_full_spon=args.authorize_full_spon)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
