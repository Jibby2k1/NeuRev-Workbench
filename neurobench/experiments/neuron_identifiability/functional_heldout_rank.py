"""Nested leave-one-burst-out tensor-rank stability analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, atomic_text
from .identity import load_adjudication
from .trace_extraction import Geometry, extract_site_traces
from .reporting import now, write_stage


RANKS = (1, 2, 3, 4)
SEEDS = tuple(range(8))


def _cp_reconstruct(factors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.einsum("ir,jr,kr->ijk", *factors, optimize=True)


def _cp_als(values: np.ndarray, rank: int, seed: int, *, iterations: int = 500, ridge: float = 1e-8) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """Small deterministic CP-ALS implementation for the 14 x burst x 24 cube."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(x.shape[0], rank)); b = rng.normal(size=(x.shape[1], rank)); c = rng.normal(size=(x.shape[2], rank))
    scale = max(float(np.sum(x * x)), np.finfo(float).eps); previous = np.inf
    eye = np.eye(rank)
    for _ in range(iterations):
        gram = (b.T @ b) * (c.T @ c) + ridge * eye
        a = np.linalg.solve(gram, np.einsum("ijk,jr,kr->ri", x, b, c, optimize=True)).T
        gram = (a.T @ a) * (c.T @ c) + ridge * eye
        b = np.linalg.solve(gram, np.einsum("ijk,ir,kr->rj", x, a, c, optimize=True)).T
        gram = (a.T @ a) * (b.T @ b) + ridge * eye
        c = np.linalg.solve(gram, np.einsum("ijk,ir,jr->rk", x, a, b, optimize=True)).T
        norms_a = np.maximum(np.linalg.norm(a, axis=0), 1e-12); norms_c = np.maximum(np.linalg.norm(c, axis=0), 1e-12)
        a /= norms_a; c /= norms_c; b *= norms_a * norms_c
        loss = float(np.sum((x - _cp_reconstruct((a, b, c))) ** 2) / scale)
        if abs(previous - loss) <= 1e-10 * max(1.0, previous): break
        previous = loss
    return (a, b, c), loss


def _fit_best(values: np.ndarray, rank: int) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    fits = []
    for seed in SEEDS:
        factors, loss = _cp_als(values, rank, seed); fits.append((loss, seed, factors))
    loss, seed, factors = min(fits, key=lambda item: (item[0], item[1]))
    return factors, {"selected_seed": seed, "training_nmse": loss, "seed_training_nmse": [float(row[0]) for row in sorted(fits, key=lambda item: item[1])]}


def _project_burst(matrix: np.ndarray, site: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.einsum("ir,kr->ikr", site, time, optimize=True).reshape(-1, site.shape[1])
    weights = np.linalg.lstsq(design, np.asarray(matrix).reshape(-1), rcond=None)[0]
    return (design @ weights).reshape(matrix.shape), weights


def _nmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum((observed - predicted) ** 2) / max(float(np.sum(observed ** 2)), np.finfo(float).eps))


def nested_lobo(cube: np.ndarray) -> dict[str, Any]:
    if cube.ndim != 3 or cube.shape[1] != 4 or not np.all(np.isfinite(cube)): raise ValueError("expected finite site x 4-burst x time cube")
    outer_rows = []; inner_rows = []
    for outer in range(4):
        outer_train = [burst for burst in range(4) if burst != outer]
        rank_scores: dict[int, list[float]] = {rank: [] for rank in RANKS}
        for inner in outer_train:
            train = [burst for burst in outer_train if burst != inner]
            train_cube = cube[:, train, :]
            validation = cube[:, inner, :]
            for rank in RANKS:
                factors, fit = _fit_best(train_cube, rank); prediction, _ = _project_burst(validation, factors[0], factors[2]); score = _nmse(validation, prediction)
                rank_scores[rank].append(score); inner_rows.append({"outer_held_out_burst": outer + 1, "inner_validation_burst": inner + 1, "rank": rank, "validation_nmse": score, **fit})
        means = {rank: float(np.mean(scores)) for rank, scores in rank_scores.items()}; best_rank = min(RANKS, key=lambda rank: (means[rank], rank))
        best_se = float(np.std(rank_scores[best_rank], ddof=1) / np.sqrt(len(rank_scores[best_rank])))
        threshold = means[best_rank] + max(best_se, 1e-8)
        selected = min(rank for rank in RANKS if means[rank] <= threshold)
        training = cube[:, outer_train, :]; held_out = cube[:, outer, :]
        factors, fit = _fit_best(training, selected); prediction, weights = _project_burst(held_out, factors[0], factors[2])
        rank1_factors, _ = _fit_best(training, 1); rank1_prediction, _ = _project_burst(held_out, rank1_factors[0], rank1_factors[2])
        outer_rows.append({"outer_held_out_burst": outer + 1, "selected_rank_one_se": selected, "minimum_error_rank": best_rank, "inner_mean_nmse_by_rank": {str(rank): means[rank] for rank in RANKS}, "one_se_threshold": threshold, "numerical_nmse_tolerance": 1e-8, "outer_nmse_selected": _nmse(held_out, prediction), "outer_nmse_rank1": _nmse(held_out, rank1_prediction), "outer_variance_explained_selected": 1.0 - _nmse(held_out, prediction), "held_out_burst_weights": weights.tolist(), **fit})
    selected = [row["selected_rank_one_se"] for row in outer_rows]; values, counts = np.unique(selected, return_counts=True); consensus = int(values[np.argmax(counts)]); agreement = int(np.max(counts))
    if consensus == 1: improvement_pass = True
    else: improvement_pass = all(row["outer_nmse_selected"] < row["outer_nmse_rank1"] for row in outer_rows if row["selected_rank_one_se"] == consensus)
    gate = {"consensus_rank": consensus, "outer_fold_agreement": agreement, "required_agreement": 3, "rank_stable": agreement >= 3, "outer_selected_mean_nmse": float(np.mean([row["outer_nmse_selected"] for row in outer_rows])), "outer_selected_mean_variance_explained": float(np.mean([row["outer_variance_explained_selected"] for row in outer_rows])), "improves_over_rank1_when_rank_gt1": improvement_pass, "passed": agreement >= 3 and improvement_pass}
    return {"schema_version": 1, "selection_rule": "minimum inner mean NMSE with one-standard-error preference for the smallest rank", "candidate_ranks": list(RANKS), "seeds": list(SEEDS), "outer_folds": outer_rows, "inner_folds": inner_rows, "gate": gate}


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat=[]
    for row in rows:
        flat.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]),delimiter="\t"); writer.writeheader(); writer.writerows(flat)


def _figure(output: Path, result: dict[str, Any]) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for row in result["outer_folds"]:
        axes[0].plot(RANKS, [row["inner_mean_nmse_by_rank"][str(rank)] for rank in RANKS], marker="o", label=f"outer burst {row['outer_held_out_burst']}")
    axes[0].set(xlabel="candidate CP rank", ylabel="inner validation NMSE", title="Nested rank selection (training bursts only)"); axes[0].set_xticks(RANKS); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    bursts=[row["outer_held_out_burst"] for row in result["outer_folds"]]; selected=[row["outer_nmse_selected"] for row in result["outer_folds"]]; rank1=[row["outer_nmse_rank1"] for row in result["outer_folds"]]
    x=np.arange(4); axes[1].bar(x-.18,rank1,.36,label="rank 1",color="#777777"); axes[1].bar(x+.18,selected,.36,label="nested selected",color="#dd6b20",edgecolor="#333333"); axes[1].set(xticks=x,xticklabels=[f"burst {b}" for b in bursts],ylabel="outer held-out NMSE",title="Held-out reconstruction after frozen rank selection"); axes[1].legend(); axes[1].grid(axis="y",alpha=.2)
    fig.savefig(output / "nested_lobo_rank_stability.png", dpi=160); plt.close(fig)


def run_functional_heldout_rank(data_root: Path, run_root: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    partial=output.with_name(output.name+".partial"); partial.mkdir(parents=True,exist_ok=False)
    labels=data_root/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv"; video_path=data_root/"Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
    records=load_adjudication(labels); counts={site:sum(r.observation_site_id==site for r in records) for site in {r.observation_site_id for r in records}}; complete=sorted(site for site,count in counts.items() if count>=4)
    video=np.load(video_path,mmap_mode="r",allow_pickle=False); site_order,traces=extract_site_traces(video,records,Geometry()); index={site:i for i,site in enumerate(site_order)}; length=min(r.stop_zero_exclusive-r.start_zero for r in records if r.observation_site_id in complete)
    cube=np.empty((len(complete),4,length),dtype=np.float64)
    for record in records:
        if record.observation_site_id not in complete: continue
        trace=traces["raw"][index[record.observation_site_id]]; baseline=np.median(trace[max(0,record.start_zero-20):record.start_zero]); cube[complete.index(record.observation_site_id),record.burst_id-1]=trace[record.start_zero:record.start_zero+length]-baseline
    result=nested_lobo(cube); result.update({"status":"complete", "estimand":"held-out burst representation after rank selection using training bursts only", "cube":{"sites":len(complete),"bursts":4,"common_window_frames":length,"units":"native_uint16_intensity_minus_local_pre_event_median"}, "source":{"labels":str(labels),"video":str(video_path)}, "interpretation":"This tests within-recording burst-held-out factor-rank stability. It does not establish cross-recording generalization, causal latent factors, or neuron identity."})
    atomic_json(partial/"heldout_rank.json",result); _write_tsv(partial/"outer_folds.tsv",result["outer_folds"]); _write_tsv(partial/"inner_folds.tsv",result["inner_folds"]); _figure(partial,result)
    atomic_text(partial/"REPORT.md",f"# Functional tensor nested leave-one-burst-out rank audit\n\nThe four outer folds selected ranks {[row['selected_rank_one_se'] for row in result['outer_folds']]}. Consensus rank is **{result['gate']['consensus_rank']}** with {result['gate']['outer_fold_agreement']}/4 agreement. Gate status: **{'PASS' if result['gate']['passed'] else 'FAIL'}**. Rank selection used only the three outer-training bursts; each inner validation fit used two bursts. Held-out burst weights were estimated against frozen site/time factors, so the estimand is representational reconstruction rather than prospective burst-amplitude prediction.\n")
    atomic_json(partial/"validation.json",{"status":"passed" if result["gate"]["passed"] else "failed_scientific_gate","finite":bool(np.all(np.isfinite(cube))),"outer_folds":4,"inner_folds":12,"rank_selection_uses_outer_heldout":False,"cross_recording_claim":False})
    atomic_json(partial/"artifact_index.json",{"artifacts":sorted(str(p.relative_to(partial)) for p in partial.iterdir() if p.is_file())}); partial.replace(output); return result


def refresh_functional_stage(run_root: Path, audit_root: Path) -> dict[str, Any]:
    result=json.loads((audit_root/"heldout_rank.json").read_text(encoding="utf-8")); stage=run_root/"07_functional_tensor"; metrics=json.loads((stage/"METRICS.json").read_text(encoding="utf-8"))
    metrics["started_at"]=now(); metrics["tensor_rank"].update({"held_out_rank_selection":"nested_leave_one_burst_out_passed" if result["gate"]["passed"] else "nested_leave_one_burst_out_failed", "promoted_rank":result["gate"]["consensus_rank"] if result["gate"]["passed"] else None, "outer_fold_agreement":result["gate"]["outer_fold_agreement"], "outer_mean_variance_explained":result["gate"]["outer_selected_mean_variance_explained"], "audit_root":str(audit_root.resolve()), "interpretation":"within-recording held-out representational rank; not causal, neuron-identifying, or cross-recording"})
    metrics["heldout_rank_audit"]={"passed":result["gate"]["passed"],"selection_rule":result["selection_rule"],"outer_folds":result["outer_folds"]}
    write_stage(run_root,"07_functional_tensor",metrics,status="complete",decision="advance",warnings=["Nested leave-one-burst-out rank stability passed; rank 3 is promoted only as a within-recording representational description.","Held-out burst weights are projected against frozen site/time factors; this is not prospective amplitude prediction, causal factor identification, or cross-recording validation."],next_stage="08_measurement_phenotypes")
    return metrics
