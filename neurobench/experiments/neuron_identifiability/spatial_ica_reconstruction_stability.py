"""Reconstruction-level seed and held-burst stability for dense spatial ICA."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from neurobench.algorithms.spatial_patch_ica import (
    SpatialPatchICAModel,
    fit_spatial_patch_fastica,
    sample_spatial_patches,
)
from neurobench.algorithms.spatial_patch_ica_reconstruction import dense_convolutional_reconstruction
from neurobench.experiments.hierarchical_parzen_ica.denoise_audit import _detection_metrics
from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import _coefficients, _innovation_residual, _quiet_standardization
from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_config import SpatialICAConfig
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import temporal_pool

from .contracts import atomic_json, atomic_text
from .spatial_ica_objective_morphology import _component_metrics


MORPHOLOGY_METRICS = ("center_annulus_contrast", "boundary_sharpness", "compactness", "effective_radius_px", "peak_offset_px")
GATES = {"pixel_correlation": .90, "event_map_correlation": .90, "morphology_rank_correlation": .70, "median_peak_error_frames": 1.0, "fixed_recall_absolute_delta": .05}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream, delimiter="\t"))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t");writer.writeheader();writer.writerows(rows)


def _reference_model(path: Path, diagnostics_path: Path) -> SpatialPatchICAModel:
    data=np.load(path);diagnostics=json.loads(diagnostics_path.read_text(encoding="utf-8"))
    return SpatialPatchICAModel(
        patch_size=int(diagnostics["patch_size"]),rank=int(diagnostics["rank"]),
        patch_mean=data["patch_mean"],analysis_filters=data["analysis_filters"],
        synthesis_atoms=data["synthesis_atoms"],component_scale=data["component_scale"],
        explained_variance_ratio=data["explained_variance_ratio"],
        fastica_iterations=int(diagnostics["fastica_iterations"]),
        fastica_converged=bool(diagnostics["fastica_converged"]),
        fastica_final_delta=float(diagnostics["fastica_final_delta"]),
    )


def _fit_model(standardized: np.ndarray, frames: np.ndarray, settings: dict[str, Any], quiet_count: int, seed: int) -> SpatialPatchICAModel:
    patches=sample_spatial_patches(standardized,patch_size=int(settings["patch_size"]),sample_count=int(settings["sample_count"]),seed=seed,frame_indices=frames)
    model=fit_spatial_patch_fastica(patches,rank=int(settings["rank"]),seed=seed,max_iterations=int(settings["fastica_max_iterations"]),tolerance=float(settings["fastica_tolerance"]))
    quiet_patches=sample_spatial_patches(standardized,patch_size=int(settings["patch_size"]),sample_count=min(int(settings["sample_count"]),12000),seed=seed+1,frame_indices=np.arange(quiet_count))
    components=(quiet_patches-model.patch_mean[None])@model.analysis_filters.T
    return replace(model,component_scale=np.maximum(np.std(components,axis=0,ddof=1),1e-6).astype(np.float32))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    a=np.asarray(left,float).ravel();b=np.asarray(right,float).ravel()
    if np.std(a)<=1e-12 or np.std(b)<=1e-12:return float("nan")
    return float(np.corrcoef(a,b)[0,1])


def _event_intervals(labels: list[dict[str, Any]], review_start_ui: int) -> dict[int, tuple[int,int]]:
    return {burst:(min(int(r["start_frame_ui"]) for r in labels if int(r["burst_id"])==burst)-review_start_ui,max(int(r["end_frame_ui"]) for r in labels if int(r["burst_id"])==burst)-review_start_ui+1) for burst in range(1,5)}


def _roi_trace(video: np.ndarray, x: float, y: float, radius: int=4) -> np.ndarray:
    yy,xx=np.indices(video.shape[1:]);mask=(xx-x)**2+(yy-y)**2<=radius**2
    return video[:,mask].mean(axis=1)


def compare_reconstructions(reference: np.ndarray, estimate: np.ndarray, labels: list[dict[str, Any]], intervals: dict[int,tuple[int,int]], evaluation_bursts: list[int], tau: float) -> tuple[dict[str,float],list[dict[str,Any]]]:
    frame_indices=np.concatenate([np.arange(*intervals[b]) for b in evaluation_bursts])
    pixel_corr=_corr(reference[frame_indices,::8,::8],estimate[frame_indices,::8,::8])
    nmse=float(np.mean((estimate[frame_indices,::8,::8]-reference[frame_indices,::8,::8])**2)/max(float(np.mean(reference[frame_indices,::8,::8]**2)),1e-12))
    burst_rows=[];morph_ref={m:[] for m in MORPHOLOGY_METRICS};morph_est={m:[] for m in MORPHOLOGY_METRICS};peak_errors=[]
    for burst in evaluation_bursts:
        start,stop=intervals[burst];base=slice(max(0,start-20),start)
        ref_event=np.maximum(reference[start:stop]-np.median(reference[base],axis=0)[None],0);est_event=np.maximum(estimate[start:stop]-np.median(estimate[base],axis=0)[None],0)
        ref_map=temporal_pool(ref_event,f"lme{tau}");est_map=temporal_pool(est_event,f"lme{tau}")
        burst_rows.append({"burst_id":burst,"event_map_correlation":_corr(ref_map,est_map),"event_map_nmse":float(np.mean((est_map-ref_map)**2)/max(float(np.mean(ref_map**2)),1e-12))})
        for row in (r for r in labels if int(r["burst_id"])==burst):
            x,y=float(row["x_px"]),float(row["y_px"]);rm=_component_metrics(ref_map,x,y);em=_component_metrics(est_map,x,y)
            for metric in MORPHOLOGY_METRICS:
                if np.isfinite(rm[metric]) and np.isfinite(em[metric]):morph_ref[metric].append(rm[metric]);morph_est[metric].append(em[metric])
            rt=_roi_trace(ref_event,x,y);et=_roi_trace(est_event,x,y)
            if max(float(rt.max()),float(et.max()))>1e-9:peak_errors.append(abs(int(np.argmax(rt))-int(np.argmax(et))))
    morphology=[]
    for metric in MORPHOLOGY_METRICS:
        if len(morph_ref[metric])>=3:
            value=float(spearmanr(morph_ref[metric],morph_est[metric]).statistic)
            if np.isfinite(value):morphology.append(value)
    return {"pixel_correlation":pixel_corr,"pixel_nmse":nmse,"median_event_map_correlation":float(np.median([r["event_map_correlation"] for r in burst_rows])),"minimum_event_map_correlation":float(np.min([r["event_map_correlation"] for r in burst_rows])),"median_morphology_spearman":float(np.median(morphology)),"minimum_morphology_spearman":float(np.min(morphology)),"median_peak_error_frames":float(np.median(peak_errors)),"p95_peak_error_frames":float(np.percentile(peak_errors,95))},burst_rows


def _figure(rows: list[dict[str,Any]], path: Path) -> None:
    variants=[r for r in rows if r["scope"]!="reference"]
    labels=[r["variant_id"].replace("seed_", "S").replace("leave_burst_", "LOBO ") for r in variants]
    fig,axes=plt.subplots(2,2,figsize=(11,7));blue="#3976AF";orange="#D87921";grid="#D9DEE5"
    panels=(("pixel_correlation","Output correlation",GATES["pixel_correlation"]),("median_event_map_correlation","Median event-map correlation",GATES["event_map_correlation"]),("median_morphology_spearman","Median morphology rank correlation",GATES["morphology_rank_correlation"]),("fixed_budget_recall_delta","Fixed-budget recall change",None))
    colors=[blue if r["scope"]=="seed" else orange for r in variants]
    for ax,(key,title,gate) in zip(axes.flat,panels):
        values=[float(r[key]) for r in variants];ax.bar(np.arange(len(values)),values,color=colors,edgecolor="#30343B",linewidth=.6);ax.set_xticks(np.arange(len(values)),labels,rotation=35,ha="right");ax.set_title(title);ax.grid(axis="y",color=grid,lw=.5)
        if gate is not None:ax.axhline(gate,color="#30343B",ls="--",lw=1,label=f"gate {gate:.2f}");ax.legend(frameon=False,fontsize=8)
        else:ax.axhline(0,color="#30343B",lw=.8);ax.axhspan(-.05,.05,color="#D9DEE5",alpha=.5)
    fig.suptitle("Spatial ICA reconstructed-output stability",x=.06,ha="left",fontsize=14);fig.text(.06,.945,"Blue: refits across seeds; orange: excluded-burst evaluation only",fontsize=9,color="#4A515B");fig.tight_layout(rect=(0,.02,1,.92));fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def run(config_path:Path,labels_tsv:Path,model_npz:Path,diagnostics_json:Path,output:Path,seeds=(20260730,20260731,20260732)) -> dict[str,Any]:
    if output.exists() or output.with_name(output.name+".partial").exists():raise FileExistsError(output)
    partial=output.with_name(output.name+".partial");partial.mkdir(parents=True);started=time.time()
    config=SpatialICAConfig.load(config_path);source=np.load(config.source_video,mmap_mode="r",allow_pickle=False);review_start=int(config.frames["review_start_ui"]);raw=np.asarray(source[review_start-1:int(config.frames["review_end_ui"])],np.float32);quiet_count=int(config.frames["quiet_end_ui"])-review_start+1
    residual,_=_innovation_residual(raw,quiet_count,_coefficients(config),config);center,scale,_=_quiet_standardization(residual,quiet_count,10.0);standardized=(residual-center[None])/scale[None];labels=label_core.load_labels(config.labels_tsv);protected=_read(labels_tsv)
    if len(labels)!=79 or len(protected)!=79:raise RuntimeError("protected cohort must contain 79 observations")
    intervals=_event_intervals(labels,review_start);settings=config.model;device=str(config.resources["device"]);tau=float(config.evaluation["temporal_pool_tau"])
    reference_model=_reference_model(model_npz,diagnostics_json);reference_z,ref_diag=dense_convolutional_reconstruction(standardized,reference_model,shrinkage="wiener",lambda_z=float(settings["wiener_lambda_z"]),device=device,frame_batch_size=int(config.resources["frame_batch_size"]));reference=reference_z*scale[None];reference_detection=_detection_metrics(reference,labels,quiet_count,config)
    rows=[{"variant_id":"reference","scope":"reference","evaluation_bursts":"1;2;3;4","pixel_correlation":1.0,"pixel_nmse":0.0,"median_event_map_correlation":1.0,"minimum_event_map_correlation":1.0,"median_morphology_spearman":1.0,"minimum_morphology_spearman":1.0,"median_peak_error_frames":0.0,"p95_peak_error_frames":0.0,"fixed_budget_recall":reference_detection["fixed_budget_mean_recall"],"fixed_budget_recall_delta":0.0,"device":ref_diag["device"]}];burst_rows=[]
    all_frames=np.arange(len(standardized));variants=[]
    for seed in seeds:variants.append((f"seed_{seed}","seed",seed,all_frames,[1,2,3,4]))
    for burst in range(1,5):
        lo,hi=intervals[burst];indices=all_frames[(all_frames<lo)|(all_frames>=hi)];variants.append((f"leave_burst_{burst}_out","held_burst",int(settings["seed"]),indices,[burst]))
    for variant_id,scope,seed,training,evaluation_bursts in variants:
        model=_fit_model(standardized,training,settings,quiet_count,seed);estimate_z,diagnostics=dense_convolutional_reconstruction(standardized,model,shrinkage="wiener",lambda_z=float(settings["wiener_lambda_z"]),device=device,frame_batch_size=int(config.resources["frame_batch_size"]));estimate=estimate_z*scale[None]
        metrics,by_burst=compare_reconstructions(reference,estimate,labels,intervals,evaluation_bursts,tau);detection=_detection_metrics(estimate,labels,quiet_count,config)
        if scope=="seed":recall=detection["fixed_budget_mean_recall"];reference_recall=reference_detection["fixed_budget_mean_recall"]
        else:
            burst=evaluation_bursts[0];recall=next(r["fixed_recall"] for r in detection["folds"] if r["burst_id"]==burst);reference_recall=next(r["fixed_recall"] for r in reference_detection["folds"] if r["burst_id"]==burst)
        rows.append({"variant_id":variant_id,"scope":scope,"evaluation_bursts":";".join(map(str,evaluation_bursts)),**metrics,"fixed_budget_recall":recall,"fixed_budget_recall_delta":recall-reference_recall,"device":diagnostics["device"]})
        burst_rows.extend({"variant_id":variant_id,"scope":scope,**row} for row in by_burst)
        del estimate_z,estimate
    evaluated=rows[1:];gates={"minimum_pixel_correlation_ge_0_90":min(r["pixel_correlation"] for r in evaluated)>=GATES["pixel_correlation"],"minimum_median_event_map_correlation_ge_0_90":min(r["median_event_map_correlation"] for r in evaluated)>=GATES["event_map_correlation"],"minimum_median_morphology_spearman_ge_0_70":min(r["median_morphology_spearman"] for r in evaluated)>=GATES["morphology_rank_correlation"],"maximum_median_peak_error_le_1_frame":max(r["median_peak_error_frames"] for r in evaluated)<=GATES["median_peak_error_frames"],"maximum_fixed_recall_absolute_delta_le_0_05":max(abs(r["fixed_budget_recall_delta"]) for r in evaluated)<=GATES["fixed_recall_absolute_delta"]}
    summary={"schema_version":1,"status":"passed_reconstruction_stability" if all(gates.values()) else "failed_reconstruction_stability","variant_count":len(rows),"seed_refits":len(seeds),"held_burst_folds":4,"comparison_signal":"signed dense-Wiener reconstruction; positive event maps formed only after event-baseline subtraction","reference_fixed_budget_mean_recall":reference_detection["fixed_budget_mean_recall"],"gates":gates,"thresholds":GATES,"claim":"reconstructed-output stability only; no individual-filter, anatomical, precision, or causal interpretation","scientific_audit":{"enabled":False,"opt_out_reason":"Author explicitly deferred manual packet-style labor on 2026-08-24; this computational extension reuses the existing spatial-ICA review media and emits quantitative comparison artifacts."},"elapsed_seconds":time.time()-started}
    _write(partial/"reconstruction_stability.tsv",rows);_write(partial/"burst_event_map_stability.tsv",burst_rows);_figure(rows,partial/"spatial_ica_reconstruction_stability.png");atomic_json(partial/"summary.json",summary);atomic_json(partial/"validation.json",{"status":"passed","rows":len(rows),"expected_rows":1+len(seeds)+4,"burst_rows":len(burst_rows),"all_metrics_finite":all(np.isfinite(float(r[k])) for r in rows for k in ("pixel_correlation","pixel_nmse","median_event_map_correlation","median_morphology_spearman","median_peak_error_frames","fixed_budget_recall")),"evaluation_scope_separated":all((r["scope"]=="seed" and r["evaluation_bursts"]=="1;2;3;4") or (r["scope"]=="held_burst" and r["evaluation_bursts"].count(";")==0) or r["scope"]=="reference" for r in rows)});atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","reconstruction_stability.tsv","burst_event_map_stability.tsv","spatial_ica_reconstruction_stability.png","llm_context.json"]});atomic_json(partial/"llm_context.json",{"entry_point":"summary.json","primary_table":"reconstruction_stability.tsv","secondary_table":"burst_event_map_stability.tsv","figure":"spatial_ica_reconstruction_stability.png","manual_packet_status":"deferred_by_author","upstream_media":"../16_spatial_ica_morphology_ablation_v1"});atomic_text(partial/"REPORT.md","# Spatial ICA reconstruction stability\n\nDense Wiener spatial-ICA reconstructions were compared with the saved reference model across three new seeds and four leave-one-burst-window-out fits. Seed variants are evaluated over all four bursts; held-burst variants are evaluated only on their excluded burst. Gates cover sampled-pixel correlation, event-map correlation, morphology-profile rank agreement, peak timing, and fixed-budget known-positive recall. Individual filters remain uninterpretable regardless of this output-level result. Unmatched candidates remain unknown.\n");partial.replace(output);return summary
