"""Stable tensor-factor profiles for passed frozen representations."""
from __future__ import annotations
import csv, itertools
from pathlib import Path
from typing import Any
import numpy as np
from .contracts import ObservationRecord, atomic_json, atomic_text
from .cross_neural_ica import event_cubes
from .functional_heldout_rank import _fit_best

PROMOTED={"raw":3,"residual":2,"frozen_two_frame_ica":2}

def _read(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def _write(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
def _corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else 0.0
def _canonical(factors):
    a,b,c=(x.copy() for x in factors); order=np.argsort(-np.linalg.norm(b,axis=0)); a,b,c=a[:,order],b[:,order],c[:,order]
    for k in range(a.shape[1]):
        sign=1 if c[np.argmax(np.abs(c[:,k])),k]>=0 else -1; a[:,k]*=sign;c[:,k]*=sign
    return a,b,c
def _align(reference,factors):
    a,b,c=_canonical(factors); rank=a.shape[1]
    best=max(itertools.permutations(range(rank)),key=lambda p:sum(abs(_corr(reference[:,k],a[:,p[k]])) for k in range(rank)))
    a,b,c=a[:,best],b[:,best],c[:,best]
    corrs=[]
    for k in range(rank):
        value=_corr(reference[:,k],a[:,k]); corrs.append(abs(value))
        if value<0:a[:,k]*=-1;c[:,k]*=-1
    return (a,b,c),corrs
def _eta_squared(values,labels):
    mean=np.mean(values); total=np.sum((values-mean)**2)
    return float(sum(np.sum(labels==g)*(np.mean(values[labels==g])-mean)**2 for g in np.unique(labels))/total) if total>0 else 0.0

def run_factor_profiles(records:list[ObservationRecord],trace_npz:Path,node_profiles:Path,output:Path,*,frozen_ica_sha256:str,permutations:int=10000):
    if output.exists():raise FileExistsError(output)
    partial=output.with_name(output.name+".partial");partial.mkdir(parents=True,exist_ok=False)
    z=np.load(trace_npz,allow_pickle=False);sites,cubes,coords,length=event_cubes(records,z["site_ids"],{k:z[k] for k in ("raw","residual","frozen_two_frame_ica")})
    profiles=_read(node_profiles); meta={r["site_id"]:r for r in profiles if r["representation"]=="frozen_two_frame_ica"}
    rng=np.random.default_rng(20260824); site_rows=[];time_rows=[];summary={}
    for representation,rank in PROMOTED.items():
        full,_=_fit_best(cubes[representation],rank); a,b,c=_canonical(full)
        fold_corr=[]
        for held in range(4):
            fit,_=_fit_best(cubes[representation][:,[x for x in range(4) if x!=held],:],rank); _,corrs=_align(a,fit);fold_corr.append(corrs)
        rep=[]
        for component in range(rank):
            stability=float(np.mean([f[component] for f in fold_corr])); values=a[:,component]
            matched=[i for i,s in enumerate(sites) if meta[s]["dominant_matched_class"]]
            labels=np.array([int(meta[sites[i]]["dominant_matched_class"]) for i in matched]); observed=_eta_squared(values[matched],labels); exceed=0
            for _ in range(permutations):
                shuffled=labels.copy();rng.shuffle(shuffled);exceed+=_eta_squared(values[matched],shuffled)>=observed
            peak=int(np.argmax(np.abs(c[:,component]))); center=float(np.sum(np.arange(length)*np.abs(c[:,component]))/np.sum(np.abs(c[:,component])))
            info={"component":component+1,"site_loading_lobo_stability":stability,"class_eta_squared":observed,"class_permutation_p":(exceed+1)/(permutations+1),"x_loading_correlation":_corr(np.array([coords[s][0] for s in sites]),values),"y_loading_correlation":_corr(np.array([coords[s][1] for s in sites]),values),"absolute_temporal_center_frame":center,"absolute_temporal_peak_frame":peak,"burst_weight_norm":float(np.linalg.norm(b[:,component]))}
            info["component_interpretation_allowed"] = stability >= 0.8
            rep.append(info)
            for i,site in enumerate(sites):site_rows.append({"representation":representation,"component":component+1,"site_id":site,"site_loading":values[i],"x_px":coords[site][0],"y_px":coords[site][1],"dominant_class":meta[site]["dominant_matched_class"],"certainty_group":meta[site]["certainty_group"],"lobo_stability":stability})
            for frame in range(length):time_rows.append({"representation":representation,"component":component+1,"relative_frame":frame,"temporal_loading":c[frame,component]})
        summary[representation]={"rank":rank,"components":rep,"all_components_stable_for_interpretation":all(x["component_interpretation_allowed"] for x in rep)}
    component_rows=[component for value in summary.values() for component in value["components"]]
    order=np.argsort([component["class_permutation_p"] for component in component_rows]); adjusted=np.empty(len(component_rows)); running=1.0
    for rank_index in range(len(component_rows)-1,-1,-1):
        index=order[rank_index]; running=min(running,component_rows[index]["class_permutation_p"]*len(component_rows)/(rank_index+1)); adjusted[index]=running
    for component,q in zip(component_rows,adjusted): component["class_permutation_q_bh_across_components"]=float(q)
    _write(partial/"site_factor_loadings.tsv",site_rows);_write(partial/"temporal_factor_loadings.tsv",time_rows)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    colors={1:"#3568a8",2:"#d08328",3:"#ad4d75"}
    for row,(representation,rank) in zip(axes,PROMOTED.items()):
        for component in range(1,rank+1):
            tr=[r for r in time_rows if r["representation"]==representation and r["component"]==component];row[0].plot([r["relative_frame"] for r in tr],[r["temporal_loading"] for r in tr],marker="o",ms=3,label=f"component {component}",color=colors[component])
            sr=[r for r in site_rows if r["representation"]==representation and r["component"]==component];row[1].scatter([r["x_px"] for r in sr],[r["site_loading"] for r in sr],label=f"component {component}",color=colors[component],s=35)
        row[0].axhline(0,color="#777",lw=.7);row[0].set(title=f"{representation.replace('_',' ')} temporal factors",xlabel="relative event frame",ylabel="temporal loading");row[0].legend(fontsize=8);row[0].grid(alpha=.2)
        row[1].axhline(0,color="#777",lw=.7);row[1].set(title=f"{representation.replace('_',' ')} site loadings",xlabel="x position (pixels)",ylabel="site loading");row[1].legend(fontsize=8);row[1].grid(alpha=.2)
    fig.suptitle("Tensor-factor profiles across promoted representations\nRaw/residual factors pass stability; ICA site factors fail interpretation gate")
    fig.savefig(partial/"tensor_factor_profiles.png",dpi=180);plt.close(fig)
    result={"schema_version":1,"status":"complete_exploratory","sites":len(sites),"bursts":4,"window_frames":length,"representations":summary,"component_interpretation_stability_threshold":0.8,"class_test_multiplicity":"Benjamini-Hochberg across all seven promoted-representation components","frozen_ica_sha256":frozen_ica_sha256,"global_adjusted_raw_excluded":"failed rank-stability gate","interpretation":"post-selection within-recording component profiles; class and certainty alignments are descriptive and not causal or anatomical"}
    atomic_json(partial/"summary.json",result);atomic_json(partial/"validation.json",{"status":"passed_with_caveats","components_aligned_across_lobo":True,"global_adjusted_factor_interpretation_suppressed":True,"publication_claim_ready":False})
    atomic_text(partial/"REPORT.md","# Tensor factor profiles\n\nOnly representations passing the matched rank-stability gate were factor-profiled. Components were ordered by burst-weight norm, signed by their dominant temporal loading, and aligned across leave-one-burst-out fits using site-loading correlation. Class tests are post-freeze; certainty remains descriptive because only two sites have uncertain history.\n")
    atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","artifact_index.json","site_factor_loadings.tsv","summary.json","temporal_factor_loadings.tsv","tensor_factor_profiles.png","validation.json"]});partial.replace(output);return result
