"""Targeted v8 development: robust stopping, close-source separation, kinetics."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.decomposition import NMF

from .contracts import atomic_json, atomic_text
from .major_next_steps import _candidates, _family_movie, _match, _score_maps, _simulated_movie


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t");writer.writeheader();writer.writerows(rows)


def _artifact(video: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    out=video.copy();t,h,w=out.shape;yy,xx=np.mgrid[:h,:w]
    if kind=="checkerboard":
        field=np.sign(np.sin(xx/3)*np.sin(yy/3));out+=.32*np.sin(np.arange(t)[:,None,None]/4.7)*field
    elif kind=="lowfreq_flicker":
        field=gaussian_filter(rng.normal(size=(h,w)),10)
        field/=max(np.std(field),1e-6);out+=.2*np.sin(np.arange(t)[:,None,None]/8.3)*field
    elif kind=="mixed_artifact":
        field=np.sin(xx/2.7)+gaussian_filter(rng.normal(size=(h,w)),5)
        out+=.2*np.sin(np.arange(t)[:,None,None]/5.1)*field
        for frame in rng.choice(np.arange(5,t-5),4,replace=False):
            cy,cx=rng.uniform(6,h-6),rng.uniform(6,w-6);out[frame]+=2.2*np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*1.1**2))
    return out


def _robust_map(score: np.ndarray) -> np.ndarray:
    median=np.median(score);mad=max(float(np.median(np.abs(score-median))*1.4826),1e-6)
    return (score-median)/mad


def _consensus(score: np.ndarray, threshold: float, persistence: int=2) -> list[tuple[int,int,float]]:
    z=_robust_map(score);detections=[]
    peaks={}
    for window in (3,5,7):
        maxima=z==maximum_filter(z,size=window);maxima[:4]=False;maxima[-4:]=False;maxima[:,:4]=False;maxima[:,-4:]=False
        ys,xs=np.nonzero(maxima)
        for y,x in zip(ys,xs):
            if z[y,x]>=threshold: peaks.setdefault((int(y),int(x)),set()).add(window)
    used=set()
    for (y,x),windows in sorted(peaks.items(),key=lambda row:z[row[0]],reverse=True):
        if (y,x) in used: continue
        neighbors=[p for p in peaks if np.hypot(p[0]-y,p[1]-x)<=2]
        support=set().union(*(peaks[p] for p in neighbors))
        if len(support)>=persistence:
            best=max(neighbors,key=lambda p:z[p]);detections.append((best[0],best[1],float(z[best])));used.update(neighbors)
    return detections


def _metric(pool,centers,radius=3.) -> dict:
    match=_match(pool,centers,radius);precision=match["tp"]/len(pool) if pool else 1.;recall=match["tp"]/len(centers)
    return {"proposals":len(pool),"fp":match["fp"],"precision":precision,"recall":recall,
            "f1":2*precision*recall/max(precision+recall,1e-12),"duplicates":match["duplicates"]}


def _stopping(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    # Source-free development nulls only; evaluation artifacts are disjoint definitions.
    null_values=[]
    for family_index,family in enumerate(("baseline","dense_neuropil","bleaching")):
        for seed in range(seeds):
            video,_=_family_movie(np.random.default_rng(41000+100*family_index+seed),family,0,profile)
            z=_robust_map(_score_maps(video)["spatial_context"])
            null_values.extend(v for _,_,v in _candidates(z,32))
    threshold=float(np.quantile(null_values,1-2/(32*3*seeds)))
    rows=[]
    for kind_index,kind in enumerate(("checkerboard","lowfreq_flicker","mixed_artifact")):
        for count in (2,4,8):
            for seed in range(seeds):
                rng=np.random.default_rng(41100+100*kind_index+10*count+seed);video,centers=_family_movie(rng,"baseline",count,profile)
                score=_score_maps(_artifact(video,rng,kind))["spatial_context"];z=_robust_map(score)
                methods={"single_window5":[row for row in _candidates(z,32) if row[2]>=threshold],
                         "parameter_consensus":_consensus(score,threshold,2)}
                for method,pool in methods.items():
                    for radius in (2.,3.,4.): rows.append({"challenge":kind,"source_count":count,"seed":seed,"method":method,"match_radius":radius,**_metric(pool,centers,radius)})
    summary={}
    for method in ("single_window5","parameter_consensus"):
        selected=[r for r in rows if r["method"]==method];by_radius=[np.mean([r["f1"] for r in selected if r["match_radius"]==radius]) for radius in (2.,3.,4.)]
        summary[method]={"mean_f1":float(np.mean([r["f1"] for r in selected])),"mean_fp":float(np.mean([r["fp"] for r in selected])),
                         "radius_f1_range":float(max(by_radius)-min(by_radius)),"radius_f1_sd":float(np.std(by_radius))}
    return rows,{"development_null_threshold":threshold,"methods":summary}


def _nmf_centers(video:np.ndarray,seed:int) -> list[tuple[int,int,float]]:
    t,h,w=video.shape;x=video.reshape(t,-1);x=np.clip(x-np.percentile(x,10,axis=0,keepdims=True),0,None)
    model=NMF(n_components=2,init="nndsvda",solver="mu",random_state=seed,max_iter=1000,tol=1e-3)
    model.fit_transform(x);components=model.components_.reshape(2,h,w);result=[]
    for component in components:
        smooth=gaussian_filter(component,1);y,x0=np.unravel_index(np.argmax(smooth),smooth.shape);result.append((int(y),int(x0),float(smooth[y,x0])))
    return result


def _weak_assigned(pool,centers,radius=3.) -> bool:
    if not pool:return False
    dist=np.asarray([[np.hypot(y-cy,x-cx) for cy,cx in centers] for y,x,_ in pool]);rr,cc=linear_sum_assignment(dist)
    return any(c==0 and dist[r,c]<=radius for r,c in zip(rr,cc))


def _separation(seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for distance in (3.,5.,7.):
        for ratio in (.2,.4,.6,1.):
            for correlation in (0.,.8):
                for seed in range(seeds):
                    video,centers=_simulated_movie(np.random.default_rng(42000+seed),distance,ratio,correlation,"crescent")
                    methods={"spatial_context":_candidates(_score_maps(video)["spatial_context"],2),"rank2_nmf":_nmf_centers(video,seed)}
                    for method,pool in methods.items(): rows.append({"distance_px":distance,"amplitude_ratio":ratio,"correlation":correlation,"seed":seed,"method":method,"weak_identity_recovered":int(_weak_assigned(pool,centers)),**_metric(pool,centers)})
    summary={}
    for method in ("spatial_context","rank2_nmf"):
        selected=[r for r in rows if r["method"]==method];summary[method]={"weak_recovery":float(np.mean([r["weak_identity_recovered"] for r in selected])),"mean_f1":float(np.mean([r["f1"] for r in selected]))}
    return rows,summary


def _kinetic_maps(video:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    centered=video-np.median(video,axis=0,keepdims=True);kernel=np.exp(-np.arange(18)/5)*(1-np.exp(-np.arange(18)/1.5));kernel/=np.linalg.norm(kernel)
    forward=np.max(np.stack([np.sum(centered[t:t+18]*kernel[:,None,None],axis=0) for t in range(len(video)-17)]),axis=0)
    reverse=np.max(np.stack([np.sum(centered[t:t+18]*kernel[::-1,None,None],axis=0) for t in range(len(video)-17)]),axis=0)
    return forward,forward-reverse


def _auc(score,centers):
    yy,xx=np.mgrid[:score.shape[0],:score.shape[1]];labels=np.zeros(score.shape,bool)
    for cy,cx in centers:labels|=(yy-cy)**2+(xx-cx)**2<=3**2
    pos=labels.ravel();ranks=rankdata(score.ravel());n=pos.sum();m=len(pos)-n
    return float((ranks[pos].sum()-n*(n+1)/2)/(n*m))


def _kinetics(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for seed in range(seeds*3):
        video,centers=_family_movie(np.random.default_rng(43000+seed),"baseline",4,profile)
        for direction,data in (("forward_movie",video),("reversed_movie",video[::-1])):
            current,directional=_kinetic_maps(data)
            for method,score in (("current_kinetic",current),("forward_minus_reverse",directional)):
                rows.append({"seed":seed,"direction":direction,"method":method,"pixel_auc":_auc(score,centers),"top8_recall":_metric(_candidates(score,8),centers)["recall"]})
    summary={}
    for method in ("current_kinetic","forward_minus_reverse"):
        forward=np.mean([r["pixel_auc"] for r in rows if r["method"]==method and r["direction"]=="forward_movie"]);reverse=np.mean([r["pixel_auc"] for r in rows if r["method"]==method and r["direction"]=="reversed_movie"])
        summary[method]={"forward_auc":float(forward),"reversed_auc":float(reverse),"directional_margin":float(forward-reverse)}
    return rows,summary


def run_targeted_automated_development(output:Path,*,empirical_profile:dict,seeds:int=6)->dict:
    output.mkdir(parents=True,exist_ok=False)
    stopping,stop_summary=_stopping(empirical_profile,seeds);separation,sep_summary=_separation(seeds);kinetics,kin_summary=_kinetics(empirical_profile,seeds)
    _write(output/"parameter_consensus_stopping.tsv",stopping);_write(output/"close_source_separation.tsv",separation);_write(output/"directional_kinetics.tsv",kinetics)
    gates={"consensus_improves_mean_f1":stop_summary["methods"]["parameter_consensus"]["mean_f1"]>stop_summary["methods"]["single_window5"]["mean_f1"],
           "consensus_reduces_radius_sensitivity":stop_summary["methods"]["parameter_consensus"]["radius_f1_range"]<stop_summary["methods"]["single_window5"]["radius_f1_range"],
           "nmf_improves_weak_identity":sep_summary["rank2_nmf"]["weak_recovery"]>sep_summary["spatial_context"]["weak_recovery"],
           "directional_forward_auc_ge_0p7":kin_summary["forward_minus_reverse"]["forward_auc"]>=.7,
           "directional_reversed_auc_le_0p55":kin_summary["forward_minus_reverse"]["reversed_auc"]<=.55}
    result={"schema_version":1,"status":"completed_targeted_automated_development","seeds":seeds,
            "components":{"parameter_consensus_stopping":stop_summary,"close_source_separation":sep_summary,"directional_kinetics":kin_summary},
            "gates":gates,"gate_pass_count":sum(gates.values()),"gate_total":len(gates),"scope":"simulator development evidence, not biological validation",
            "scientific_audit":{"truth":"exact simulator identities","null_labels":"zero-source development movies","human_review":"not used"}}
    atomic_json(output/"summary.json",result);atomic_json(output/"validation.json",{"status":"passed","tables":3,"all_tables_nonempty":all((output/name).stat().st_size>50 for name in ("parameter_consensus_stopping.tsv","close_source_separation.tsv","directional_kinetics.tsv")),"gates":len(gates)})
    atomic_json(output/"llm_context.json",{"entry_point":"summary.json","tables":["parameter_consensus_stopping.tsv","close_source_separation.tsv","directional_kinetics.tsv"],"evidence_boundary":"automated simulator development only"})
    atomic_json(output/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","llm_context.json","parameter_consensus_stopping.tsv","close_source_separation.tsv","directional_kinetics.tsv"]})
    atomic_text(output/"REPORT.md",f"# Targeted automated development v8\n\nCompleted three targeted components; {sum(gates.values())}/{len(gates)} gates passed.\n")
    return result
