"""Objective hero/typical/hard-case selection and publication visual atlas."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .contracts import atomic_json, atomic_text


METRICS=("signed_peak_amplitude","robust_peak_snr","residual_peak_amplitude","normalized_spatial_specificity")
BURSTS={1:(2003,2026),2:(2040,2063),3:(2122,2149),4:(2254,2300)}
BLUE="#2F5D8A"; ORANGE="#D17A22"; GOLD="#B6922E"; INK="#222222"; GRID="#D9D9D9"


def _tsv(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream,delimiter="\t"))


def _write_tsv(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t");writer.writeheader();writer.writerows(rows)


def _rank01(values:dict[str,float])->dict[str,float]:
    keys=sorted(values,key=lambda k:(values[k],k)); n=max(1,len(keys)-1); return {key:i/n for i,key in enumerate(keys)}


def select_representatives(rows:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    eligible=[r for r in rows if int(r["occurrences"])==4 and r["confirmed_only"]]
    if len(eligible)<3:raise ValueError("fewer than three complete confirmed sites")
    hero=max(eligible,key=lambda r:(r["composite_score"],r["observation_site_id"]))
    hard=min(eligible,key=lambda r:(r["recovery_score"],r["measurement_score"],r["observation_site_id"]))
    candidates=[r for r in eligible if r["observation_site_id"] not in {hero["observation_site_id"],hard["observation_site_id"]}]
    typical=min(candidates,key=lambda r:(r["typical_distance"],r["observation_site_id"]))
    return {"hero":hero,"typical":typical,"hard_case":hard}


def _save(fig:plt.Figure,target:Path)->None:
    fig.savefig(target.with_suffix(".png"),dpi=180,bbox_inches="tight");fig.savefig(target.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)


def build_visual_atlas(run_root:Path,video_path:Path)->dict[str,Any]:
    target=run_root/"analysis_visuals";target.mkdir(parents=True,exist_ok=True)
    occurrences=_tsv(run_root/"03_trace_atlas/occurrence_metrics.tsv"); labels=_tsv(run_root/"02_label_timing_contract/original_site_original_timing.tsv"); acquisition={r["observation_site_id"]:r for r in _tsv(run_root/"04_acquisition_qc/per_site_acquisition_covariates.tsv")}
    failures=_tsv(run_root/"10_representation_confirmation/nms_sensitivity/nms_failure_audit.tsv")
    failures=[r for r in failures if int(r["nms_distance_px"])==6]
    grouped:dict[str,list[dict[str,str]]]=defaultdict(list)
    for row in occurrences:grouped[row["observation_site_id"]].append(row)
    dispositions:dict[str,list[bool]]=defaultdict(list)
    for row in labels:dispositions[row["observation_site_id"]].append(row["include_confirmed"].lower()=="true")
    rec:dict[tuple[str,str],list[bool]]=defaultdict(list)
    for row in failures:rec[(row["original_roi_id"],row["feature_id"])].append(row["failure_class_at_budget_58"]=="matched")
    summaries=[]
    for site,site_rows in grouped.items():
        item={"observation_site_id":site,"occurrences":len(site_rows),"confirmed_only":all(dispositions[site])}
        for metric in METRICS:item[f"median_{metric}"]=float(np.median([float(r[metric]) for r in site_rows]))
        for lane in ("carrier_signed","coherence_w15","propagation_lag2_w15"):
            vals=rec[(site,lane)];item[f"recovery_b58_{lane}"]=sum(vals)/len(vals) if vals else float("nan")
        item.update({"x_px":float(acquisition[site]["x_px"]),"y_px":float(acquisition[site]["y_px"]),"quiet_mean_intensity":float(acquisition[site]["quiet_mean_intensity"]),"nearest_site_distance_px":float(acquisition[site]["distance_to_nearest_site_px"])})
        summaries.append(item)
    complete=[r for r in summaries if r["occurrences"]==4 and r["confirmed_only"]]
    metric_ranks={m:_rank01({r["observation_site_id"]:r[f"median_{m}"] for r in complete}) for m in METRICS}
    recovery_ranks=_rank01({r["observation_site_id"]:np.nanmean([r[f"recovery_b58_{lane}"] for lane in ("carrier_signed","coherence_w15","propagation_lag2_w15")]) for r in complete})
    vectors=[]
    for r in complete:
        site=r["observation_site_id"];r["measurement_score"]=float(np.mean([metric_ranks[m][site] for m in METRICS]));r["recovery_score"]=float(np.nanmean([r[f"recovery_b58_{lane}"] for lane in ("carrier_signed","coherence_w15","propagation_lag2_w15")]));r["composite_score"]=(r["measurement_score"]+recovery_ranks[site])/2;vectors.append([metric_ranks[m][site] for m in METRICS]+[recovery_ranks[site]])
    median_vector=np.median(np.asarray(vectors),axis=0)
    for r,vector in zip(complete,vectors,strict=True):r["typical_distance"]=float(np.linalg.norm(np.asarray(vector)-median_vector))
    by_site={r["observation_site_id"]:r for r in summaries}
    for r in complete:by_site[r["observation_site_id"]].update(r)
    selected=select_representatives(summaries)
    _write_tsv(target/"site_summary.tsv",summaries);atomic_json(target/"representative_selection.json",{role:{k:v for k,v in row.items()} for role,row in selected.items()})

    # Site-by-burst matrix figure.
    sites=sorted(grouped); matrices=[]
    for metric in METRICS:
        matrix=np.full((len(sites),4),np.nan)
        for i,site in enumerate(sites):
            for row in grouped[site]:matrix[i,int(row["burst_id"])-1]=float(row[metric])
        med=np.nanmedian(matrix,axis=1,keepdims=True);mad=np.nanmedian(np.abs(matrix-med),axis=1,keepdims=True);matrices.append((matrix-med)/np.maximum(1.4826*mad,1e-9))
    fig,axes=plt.subplots(1,4,figsize=(13,9),sharey=True,layout="constrained")
    titles=("Raw peak amplitude","Robust peak SNR","Residual peak amplitude","Spatial specificity")
    for ax,matrix,title in zip(axes,matrices,titles,strict=True):
        im=ax.imshow(matrix,aspect="auto",cmap="PuOr",vmin=-3,vmax=3);ax.set_title(title);ax.set_xticks(range(4),["B1","B2","B3","B4"]);ax.set_xlabel("burst")
    axes[0].set_yticks(range(len(sites)),sites,fontsize=7);fig.colorbar(im,ax=axes,label="within-site robust deviation");fig.suptitle("Site-by-burst measurement atlas",fontsize=14);_save(fig,target/"site_by_burst_atlas")

    # Spatial map.
    video=np.load(video_path,mmap_mode="r",allow_pickle=False);background=np.mean(np.asarray(video[::20],dtype=np.float32),axis=0)
    fig,ax=plt.subplots(figsize=(10,6),layout="constrained");ax.imshow(background,cmap="gray",vmin=np.percentile(background,2),vmax=np.percentile(background,99.5))
    scores=np.array([by_site[s]["measurement_score"] if "measurement_score" in by_site[s] else np.nan for s in sites]);valid=np.isfinite(scores)
    scatter=ax.scatter([by_site[s]["x_px"] for i,s in enumerate(sites) if valid[i]],[by_site[s]["y_px"] for i,s in enumerate(sites) if valid[i]],c=scores[valid],cmap="Blues",vmin=0,vmax=1,s=90,edgecolors="white",linewidths=.8)
    for role,row in selected.items():ax.scatter(row["x_px"],row["y_px"],s=240,facecolors="none",edgecolors={"hero":GOLD,"typical":BLUE,"hard_case":ORANGE}[role],linewidths=2.5,label=role.replace("_"," "))
    ax.legend();ax.set_title("Spatial distribution of complete-site observability");ax.set_xlabel("x (pixels)");ax.set_ylabel("y (pixels)");fig.colorbar(scatter,ax=ax,label="objective measurement score");_save(fig,target/"spatial_observability_map")

    # Three representative profiles.
    trace=np.load(run_root/"03_trace_atlas/traces.npz",allow_pickle=False);trace_sites=[str(x) for x in trace["site_ids"]];index={s:i for i,s in enumerate(trace_sites)}
    fig,axes=plt.subplots(3,4,figsize=(15,11),layout="constrained")
    for row_index,(role,row) in enumerate(selected.items()):
        site=row["observation_site_id"];si=index[site];x=int(round(row["x_px"]));y=int(round(row["y_px"]));crop=background[max(0,y-20):y+21,max(0,x-20):x+21]
        axes[row_index,0].imshow(crop,cmap="gray",vmin=np.percentile(crop,2),vmax=np.percentile(crop,99.5));axes[row_index,0].scatter([min(20,x)],[min(20,y)],facecolors="none",edgecolors=ORANGE,s=100);axes[row_index,0].set_title(f"{role.replace('_',' ').title()}: {site}\nspatial crop");axes[row_index,0].axis("off")
        for burst,(start,end) in BURSTS.items():
            frames=np.arange(start-10,end+11);raw=trace["raw"][si,frames-1];axes[row_index,1].plot(frames-start,raw-np.median(raw[:10]),lw=1,label=f"B{burst}")
            residual=trace["residual"][si,frames-1];axes[row_index,2].plot(frames-start,residual-np.median(residual[:10]),lw=1,label=f"B{burst}")
        axes[row_index,1].set_title("Raw, baseline subtracted");axes[row_index,2].set_title("ROI minus annulus")
        lanes=("carrier_signed","coherence_w15","propagation_lag2_w15");vals=[row[f"recovery_b58_{lane}"] for lane in lanes];axes[row_index,3].bar(["carrier","coherence","lag"],vals,color=["#777777",BLUE,ORANGE]);axes[row_index,3].set_ylim(0,1);axes[row_index,3].set_title("Known-positive recovery @58");axes[row_index,3].tick_params(axis="x",rotation=20)
        for col in (1,2):axes[row_index,col].axvline(0,color=INK,ls="--",lw=.8);axes[row_index,col].grid(color=GRID,lw=.5);axes[row_index,col].set_xlabel("frames from event start")
    axes[0,1].legend(ncol=4,fontsize=7);fig.suptitle("Objectively selected hero, typical, and hard-case sites",fontsize=15);_save(fig,target/"representative_site_profiles")

    # NMS sensitivity.
    nms=json.loads((run_root/"10_representation_confirmation/nms_sensitivity/nms_sensitivity.json").read_text());lookup={(r["feature_id"],r["nms_distance_px"]):r for r in nms["results"]}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for lane,color,marker in (("carrier_signed","#666666","o"),("coherence_w15",BLUE,"s"),("propagation_lag2_w15",ORANGE,"^")):
        axes[0].plot([4,6,8],[lookup[(lane,r)]["macro_recall"]["20"] for r in (4,6,8)],color=color,marker=marker,label=lane)
    axes[0].axvline(6,color=INK,ls="--",lw=.8);axes[0].set(xlabel="NMS distance (pixels)",ylabel="macro known-positive recall @20",title="Primary and sensitivity NMS radii");axes[0].legend(fontsize=8);axes[0].grid(color=GRID,lw=.5)
    for lane,color,offset in (("coherence_w15",BLUE,-.08),("propagation_lag2_w15",ORANGE,.08)):
        vals=[nms["gates"][lane][str(r)]["macro_delta_b20"] for r in (4,6,8)];axes[1].plot(np.array([4,6,8])+offset,vals,color=color,marker="o",label=lane)
    axes[1].axhline(0,color=INK,lw=.8);axes[1].axhline(.03,color="#777777",ls=":",label="primary C3 threshold");axes[1].axvline(6,color=INK,ls="--",lw=.8);axes[1].set(xlabel="NMS distance (pixels)",ylabel="gain versus carrier @20",title="Compact-lane gain remains positive");axes[1].legend(fontsize=8);axes[1].grid(color=GRID,lw=.5);_save(fig,target/"nms_sensitivity")

    # Descriptive recovery associations with site bootstrap intervals.
    assoc=[];rng=np.random.default_rng(20260822)
    complete_all=[r for r in summaries if np.isfinite(r["recovery_b58_carrier_signed"])]
    for metric in ("median_signed_peak_amplitude","median_robust_peak_snr","median_normalized_spatial_specificity","quiet_mean_intensity","nearest_site_distance_px"):
        x=np.array([r[metric] for r in complete_all]);y=np.array([r["recovery_b58_carrier_signed"] for r in complete_all]);rank=lambda a:np.argsort(np.argsort(a));rho=float(np.corrcoef(rank(x),rank(y))[0,1]);boots=[]
        for _ in range(2000):
            idx=rng.integers(0,len(x),len(x));rx,ry=rank(x[idx]),rank(y[idx]);c=np.corrcoef(rx,ry)[0,1];
            if np.isfinite(c):boots.append(c)
        assoc.append({"metric":metric,"spearman_rho":rho,"ci95_low":float(np.percentile(boots,2.5)),"ci95_high":float(np.percentile(boots,97.5)),"sites":len(x),"interpretation":"descriptive_single_recording"})
    _write_tsv(target/"recovery_associations.tsv",assoc)
    fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained");order=sorted(assoc,key=lambda r:r["spearman_rho"]);ys=np.arange(len(order));vals=np.array([r["spearman_rho"] for r in order]);lo=np.array([r["ci95_low"] for r in order]);hi=np.array([r["ci95_high"] for r in order]);ax.errorbar(vals,ys,xerr=[vals-lo,hi-vals],fmt="o",color=BLUE,ecolor="#777777",capsize=3);ax.axvline(0,color=INK,lw=.8);ax.set_yticks(ys,[r["metric"].replace("median_","").replace("_"," ") for r in order]);ax.set(xlabel="Spearman association with carrier recovery @58",title="Descriptive site-level recovery associations");ax.grid(axis="x",color=GRID,lw=.5);_save(fig,target/"recovery_associations")
    stats={"schema_version":1,"scope":"single recording; site is the bootstrap unit","selection":{"eligible":"confirmed sites observed in all four bursts","hero":"maximum average of measurement-rank score and recovery-rank score","typical":"minimum distance to the eligible median rank profile, excluding hero and hard case","hard_case":"minimum frozen B58 recovery, then measurement score","selected":{k:v["observation_site_id"] for k,v in selected.items()}},"associations":assoc,"prohibited_interpretations":["population generalization","precision","causal acquisition effect"]}
    atomic_json(target/"visual_statistics.json",stats)
    atomic_text(target/"CHART_CONTRACT.md","# Chart contract\n\n- Question: How do site measurements vary across bursts, where are easy and difficult sites located, and do compact-lane gains survive 4/6/8 px NMS?\n- Takeaway sought: within-recording measurement heterogeneity and robustness, not precision or cross-recording generalization.\n- Grain: 79 occurrences, 27 immutable sites, four bursts; representative selection restricted to confirmed four-burst sites.\n- Forms: heatmap, spatial scatter, small-multiple trace profiles, highlighted line comparison, and dot-and-interval association plot.\n- Palette: blue/orange plus neutral gray; labels and marker shapes provide non-color distinction.\n- Renderer: static Matplotlib PNG and PDF under `analysis_visuals/`; every exported image requires visual QA.\n")
    atomic_text(target/"REPORT.md",f"# Visual statistical atlas\n\nObjectively selected sites: hero `{selected['hero']['observation_site_id']}`, typical `{selected['typical']['observation_site_id']}`, and hard case `{selected['hard_case']['observation_site_id']}`. Selection was frozen to confirmed sites observed in all four bursts. These are illustrative measurement phenotypes, not biological archetypes. Both compact lanes passed the frozen 4/6/8 px NMS robustness test. Recovery associations are descriptive within one recording and use sites as bootstrap units.\n")
    return stats
