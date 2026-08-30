"""Matched nested-LOBO tensor rank analysis across frozen representations."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ObservationRecord, atomic_json, atomic_text
from .cross_neural_ica import REPRESENTATIONS, event_cubes
from .functional_heldout_rank import RANKS, nested_lobo


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t"); writer.writeheader(); writer.writerows(rows)


def run_multirepresentation_tensor(records: list[ObservationRecord], trace_npz: Path, output: Path, *, frozen_ica_sha256: str) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    partial=output.with_name(output.name+".partial"); partial.mkdir(parents=True,exist_ok=False)
    archive=np.load(trace_npz,allow_pickle=False)
    sites,cubes,_,length=event_cubes(records,archive["site_ids"],{key:archive[key] for key in ("raw","residual","frozen_two_frame_ica")})
    results={}; rows=[]
    for representation in REPRESENTATIONS:
        result=nested_lobo(cubes[representation]); results[representation]=result
        gate=result["gate"]
        rows.append({"representation":representation,"selected_ranks":"/".join(str(r["selected_rank_one_se"]) for r in result["outer_folds"]),"consensus_rank":gate["consensus_rank"],"outer_fold_agreement":gate["outer_fold_agreement"],"mean_heldout_variance_explained":gate["outer_selected_mean_variance_explained"],"mean_heldout_nmse":gate["outer_selected_mean_nmse"],"rank1_improvement_pass":gate["improves_over_rank1_when_rank_gt1"],"gate_passed":gate["passed"]})
        atomic_json(partial/f"{representation}_nested_lobo.json",result)
    _write(partial/"representation_rank_summary.tsv",rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    palette={"raw":"#3568a8","residual":"#d08328","global_adjusted_raw":"#6d7f32","frozen_two_frame_ica":"#ad4d75"}
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    x=np.arange(4); labels=[r.replace("_"," ") for r in REPRESENTATIONS]
    axes[0].bar(x,[results[r]["gate"]["outer_selected_mean_variance_explained"] for r in REPRESENTATIONS],color=[palette[r] for r in REPRESENTATIONS],edgecolor="#333333")
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel="mean held-out variance explained",title="Nested selected-rank reconstruction"); axes[0].tick_params(axis="x",rotation=20); axes[0].grid(axis="y",alpha=.2)
    for i,r in enumerate(REPRESENTATIONS):
        selected=[o["selected_rank_one_se"] for o in results[r]["outer_folds"]]
        label=f"rank {results[r]['gate']['consensus_rank']}" if results[r]["gate"]["passed"] else "unstable " + "/".join(map(str,selected))
        axes[0].text(i,results[r]["gate"]["outer_selected_mean_variance_explained"]+.025,label,ha="center",fontsize=9)
    offsets=np.linspace(-.27,.27,4)
    for offset,r in zip(offsets,REPRESENTATIONS):
        axes[1].bar(np.arange(4)+offset,[o["outer_variance_explained_selected"] for o in results[r]["outer_folds"]],.18,label=r.replace("_"," "),color=palette[r])
    axes[1].set(xticks=np.arange(4),xticklabels=["burst 1","burst 2","burst 3","burst 4"],ylim=(0,1),ylabel="held-out variance explained",title="Outer-fold reconstruction by representation"); axes[1].legend(fontsize=8); axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("Matched functional tensor rank across representations\n14 immutable sites × 4 bursts × 24 frames; rank selected without held-out burst")
    fig.savefig(partial/"multirepresentation_tensor_rank.png",dpi=180); plt.close(fig)
    summary={"schema_version":1,"status":"complete","sites":len(sites),"bursts":4,"window_frames":length,"representations":{r:{"selected_ranks":[o["selected_rank_one_se"] for o in results[r]["outer_folds"]],**results[r]["gate"]} for r in REPRESENTATIONS},"frozen_ica_sha256":frozen_ica_sha256,"matched_contract":{"same_sites":True,"same_windows":True,"same_candidate_ranks":list(RANKS),"same_seeds":list(range(8)),"same_nested_lobo_rule":True},"interpretation":"within-recording representational complexity and held-out reconstruction; factors are not causal sources, anatomical networks, or cross-recording evidence"}
    atomic_json(partial/"summary.json",summary); atomic_json(partial/"validation.json",{"status":"passed" if all(results[r]["gate"]["passed"] for r in REPRESENTATIONS) else "mixed_scientific_gates","all_cubes_finite":all(np.all(np.isfinite(cubes[r])) for r in REPRESENTATIONS),"outer_holdout_used_for_rank_selection":False,"publication_claim_ready":False})
    atomic_text(partial/"REPORT.md","# Matched multi-representation functional tensor audit\n\nRaw, residual, leave-one-site-out global-adjusted Raw, and hash-verified frozen two-frame ICA were evaluated with identical nested leave-one-burst-out rank selection. Results describe representation-specific within-recording complexity and do not identify causal components, anatomical subnetworks, or cross-recording generalization.\n")
    atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","artifact_index.json","frozen_two_frame_ica_nested_lobo.json","global_adjusted_raw_nested_lobo.json","multirepresentation_tensor_rank.png","raw_nested_lobo.json","representation_rank_summary.tsv","residual_nested_lobo.json","summary.json","validation.json"]})
    partial.replace(output); return summary
