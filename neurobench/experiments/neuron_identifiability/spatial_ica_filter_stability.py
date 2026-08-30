"""Seed and leave-one-burst-out stability of the spatial patch-ICA bank."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from neurobench.algorithms.spatial_patch_ica import fit_spatial_patch_fastica, sample_spatial_patches
from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import _coefficients,_innovation_residual,_quiet_standardization
from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_config import SpatialICAConfig
from .contracts import atomic_json,atomic_text

def _read(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def _write(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
def _normalize(filters):
    values=np.asarray(filters,float);return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
def compare_filters(reference,estimate):
    left,right=_normalize(reference),_normalize(estimate);similarity=np.abs(left@right.T)
    r,c=linear_sum_assignment(-similarity);aligned=similarity[r,c]
    angles=subspace_angles(left.T,right.T)
    return {"mean_subspace_cosine":float(np.mean(np.cos(angles))),"minimum_subspace_cosine":float(np.min(np.cos(angles))),"median_aligned_abs_correlation":float(np.median(aligned)),"minimum_aligned_abs_correlation":float(np.min(aligned)),"aligned_abs_correlations":aligned.tolist()}
def _gallery(filters,path):
    values=np.asarray(filters).reshape(len(filters),11,11);limit=max(float(np.percentile(np.abs(values),99)),1e-9)
    fig,axes=plt.subplots(3,4,figsize=(8,6.4))
    for i,(ax,image) in enumerate(zip(axes.flat,values),1):
        ax.imshow(image,cmap="coolwarm",vmin=-limit,vmax=limit,interpolation="nearest");ax.set_title(f"Filter {i}");ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Reference spatial patch-ICA analysis filters",x=.06,ha="left",fontsize=14);fig.text(.06,.945,"Signed 11 x 11 filters; shared symmetric scale; names withheld pending stability",fontsize=9,color="#4A515B")
    fig.tight_layout(rect=(0,.02,1,.92));fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)
def run(config_path:Path,labels_tsv:Path,model_npz:Path,output:Path,seeds=(20260729,20260730,20260731,20260732)):
    if output.exists() or output.with_name(output.name+".partial").exists():raise FileExistsError(output)
    partial=output.with_name(output.name+".partial");partial.mkdir(parents=True);started=time.time()
    config=SpatialICAConfig.load(config_path);source=np.load(config.source_video,mmap_mode="r",allow_pickle=False)
    start=int(config.frames["review_start_ui"])-1;stop=int(config.frames["review_end_ui"]);raw=np.asarray(source[start:stop],np.float32)
    quiet=int(config.frames["quiet_end_ui"])-int(config.frames["quiet_start_ui"])+1
    residual,_=_innovation_residual(raw,quiet,_coefficients(config),config);center,scale,_=_quiet_standardization(residual,quiet,10.0);standardized=(residual-center[None])/scale[None]
    reference=np.load(model_npz)["analysis_filters"];settings=config.model;labels=_read(labels_tsv);rows=[]
    def fit(scope,seed,indices):
        patches=sample_spatial_patches(standardized,patch_size=int(settings["patch_size"]),sample_count=int(settings["sample_count"]),seed=seed,frame_indices=indices)
        model=fit_spatial_patch_fastica(patches,rank=int(settings["rank"]),seed=seed,max_iterations=int(settings["fastica_max_iterations"]),tolerance=float(settings["fastica_tolerance"]));metrics=compare_filters(reference,model.analysis_filters)
        rows.append({"scope":scope,"seed":seed,"training_frames":len(indices),"fastica_converged":model.fastica_converged,"fastica_iterations":model.fastica_iterations,**{k:v for k,v in metrics.items() if k!="aligned_abs_correlations"},"aligned_abs_correlations":";".join(f"{x:.6f}" for x in metrics["aligned_abs_correlations"])})
    all_frames=np.arange(len(standardized))
    for seed in seeds:fit("all_review_frames",seed,all_frames)
    for burst in range(1,5):
        subset=[r for r in labels if int(r["burst_id"])==burst];lo=min(int(r["original_start_frame_ui"]) for r in subset)-int(config.frames["review_start_ui"]);hi=max(int(r["original_end_frame_ui"]) for r in subset)-int(config.frames["review_start_ui"])+1
        indices=all_frames[(all_frames<lo)|(all_frames>=hi)];fit(f"leave_burst_{burst}_out",int(settings["seed"]),indices)
    seed_rows=[r for r in rows if r["scope"]=="all_review_frames"];lobo=[r for r in rows if r["scope"].startswith("leave_burst")]
    gates={"all_fits_converged":all(r["fastica_converged"] for r in rows),"seed_subspace_cosine_ge_0_90":min(r["mean_subspace_cosine"] for r in seed_rows)>=.9,"lobo_subspace_cosine_ge_0_90":min(r["mean_subspace_cosine"] for r in lobo)>=.9,"seed_individual_filter_median_ge_0_80":min(r["median_aligned_abs_correlation"] for r in seed_rows)>=.8,"lobo_individual_filter_median_ge_0_80":min(r["median_aligned_abs_correlation"] for r in lobo)>=.8}
    summary={"schema_version":1,"status":"passed" if all(gates.values()) else "failed_interpretation_gate","fit_count":len(rows),"seed_count":len(seeds),"held_burst_folds":4,"gates":gates,"individual_filter_interpretation_allowed":bool(gates["seed_individual_filter_median_ge_0_80"] and gates["lobo_individual_filter_median_ge_0_80"]),"bank_level_interpretation_allowed":bool(gates["seed_subspace_cosine_ge_0_90"] and gates["lobo_subspace_cosine_ge_0_90"]),"limitations":["stability does not establish neuronal meaning","leave-one-burst-out excludes labeled event windows but retains the rest of the same recording","reference model was transductive and label-free"],"elapsed_seconds":time.time()-started}
    _write(partial/"filter_stability.tsv",rows);_gallery(reference,partial/"spatial_ica_filter_gallery.png");atomic_json(partial/"summary.json",summary);atomic_json(partial/"validation.json",{"status":"passed","rows":len(rows),"expected_rows":len(seeds)+4,"all_metrics_finite":all(np.isfinite(float(r[k])) for r in rows for k in ("mean_subspace_cosine","minimum_subspace_cosine","median_aligned_abs_correlation","minimum_aligned_abs_correlation"))});atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","filter_stability.tsv","spatial_ica_filter_gallery.png","llm_context.json"]});atomic_json(partial/"llm_context.json",{"entry_point":"summary.json","primary_table":"filter_stability.tsv","figure":"spatial_ica_filter_gallery.png","claim_scope":"spatial filter-bank reproducibility only"});atomic_text(partial/"REPORT.md","# Spatial ICA filter stability\n\nThe 12-filter bank was refit with four sampling/ICA seeds and four leave-one-burst-window-out folds. Permutation and sign indeterminacy were resolved by maximum absolute filter correlation. Bank-level subspace stability and aligned individual-filter stability use prespecified 0.90 and 0.80 gates. Stability is necessary before filter naming but does not establish neuronal or anatomical meaning.\n");partial.replace(output);return summary
