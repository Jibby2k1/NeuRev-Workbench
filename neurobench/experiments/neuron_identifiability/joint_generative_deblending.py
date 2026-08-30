"""Constrained joint spatial-temporal deblending with kinetic likelihoods."""
from __future__ import annotations

import csv
from dataclasses import asdict,dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter
from scipy.optimize import linear_sum_assignment,nnls
from sklearn.metrics import roc_auc_score

from .contracts import atomic_json,atomic_text
from .major_next_steps import _candidates,_family_movie,_match,_score_maps,_simulated_movie


@dataclass(frozen=True)
class DeblendConfig:
    components:int=2
    iterations:int=8
    footprint_sigma_px:float=2.0
    footprint_support_radius_px:float=7.0
    initialization_exclusion_px:float=2.0
    calcium_decay_frames:float=5.0
    calcium_rise_frames:float=1.5
    event_l2:float=0.02


FROZEN_CONFIG=DeblendConfig()


def _kernel(length:int,reverse:bool=False)->np.ndarray:
    k=np.exp(-np.arange(18)/FROZEN_CONFIG.calcium_decay_frames)*(1-np.exp(-np.arange(18)/FROZEN_CONFIG.calcium_rise_frames))
    k/=max(np.linalg.norm(k),1e-9)
    if reverse:k=k[::-1]
    matrix=np.zeros((length,length))
    for onset in range(length):
        end=min(length,onset+len(k));matrix[onset:end,onset]=k[:end-onset]
    return matrix


def _initial_centers(video:np.ndarray,count:int)->list[tuple[int,int]]:
    score=_score_maps(video)["spatial_context"];maxima=score==maximum_filter(score,size=3);ys,xs=np.nonzero(maxima)
    order=np.argsort(score[ys,xs])[::-1];centers=[]
    for index in order:
        y,x=int(ys[index]),int(xs[index])
        if y<4 or x<4 or y>=score.shape[0]-4 or x>=score.shape[1]-4:continue
        if all(np.hypot(y-cy,x-cx)>FROZEN_CONFIG.initialization_exclusion_px for cy,cx in centers):centers.append((y,x))
        if len(centers)==count:break
    return centers


def _kinetic_fit(trace:np.ndarray,reverse:bool=False)->tuple[np.ndarray,float]:
    matrix=_kernel(len(trace),reverse);scale=max(float(np.std(trace)),1e-6)
    augmented=np.vstack([matrix,np.sqrt(FROZEN_CONFIG.event_l2)*np.eye(len(trace))])
    target=np.r_[np.clip(trace-np.percentile(trace,10),0,None),np.zeros(len(trace))]
    events=nnls(augmented,target,maxiter=20*len(trace))[0];fit=matrix@events
    return fit,float(np.mean((target[:len(trace)]-fit)**2)/(scale**2))


def deblend(video:np.ndarray,config:DeblendConfig=FROZEN_CONFIG)->dict:
    if config!=FROZEN_CONFIG:raise ValueError("v9 evaluator accepts only the frozen configuration")
    t,h,w=video.shape;x=video.reshape(t,-1).astype(float);x=np.clip(x-np.percentile(x,10,axis=0,keepdims=True),0,None)
    centers=_initial_centers(video,config.components)
    if len(centers)<config.components:raise RuntimeError("insufficient initialization peaks")
    yy,xx=np.mgrid[:h,:w];a=[]
    for cy,cx in centers:a.append(np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*config.footprint_sigma_px**2)).ravel())
    a=np.asarray(a).T;c=np.zeros((t,config.components))
    for _ in range(config.iterations):
        for frame in range(t):c[frame]=nnls(a,x[frame],maxiter=20*config.components)[0]
        for component in range(config.components):c[:,component]=_kinetic_fit(c[:,component],False)[0]
        a=np.clip(np.linalg.lstsq(c,x,rcond=1e-5)[0].T,0,None)
        for component in range(config.components):
            image=gaussian_filter(a[:,component].reshape(h,w),.8);cy,cx=np.unravel_index(np.argmax(image),image.shape)
            image*=((yy-cy)**2+(xx-cx)**2<=config.footprint_support_radius_px**2)
            norm=max(float(np.linalg.norm(image)),1e-9);a[:,component]=image.ravel()/norm;c[:,component]*=norm
    # Directionality must be audited on unconstrained coefficients. Reusing the
    # forward-projected factors here would make the likelihood circular.
    audit_c=np.zeros_like(c)
    for frame in range(t):audit_c[frame]=nnls(a,x[frame],maxiter=20*config.components)[0]
    proposals=[];likelihoods=[]
    for component in range(config.components):
        image=a[:,component].reshape(h,w);cy,cx=np.unravel_index(np.argmax(image),image.shape)
        forward,forward_rss=_kinetic_fit(audit_c[:,component],False);reverse,reverse_rss=_kinetic_fit(audit_c[:,component],True)
        likelihood=(reverse_rss-forward_rss);proposals.append((int(cy),int(cx),float(likelihood)));likelihoods.append(float(likelihood))
    return {"proposals":proposals,"spatial":a.reshape(h,w,config.components),"temporal":c,
            "unconstrained_audit_temporal":audit_c,"kinetic_likelihoods":likelihoods}


def _weak_match(pool,centers,radius=3.)->bool:
    dist=np.asarray([[np.hypot(y-cy,x-cx) for cy,cx in centers] for y,x,_ in pool]);rr,cc=linear_sum_assignment(dist)
    return any(col==0 and dist[row,col]<=radius for row,col in zip(rr,cc))


def _metrics(pool,centers)->dict:
    match=_match(pool,centers,3);precision=match["tp"]/len(pool);recall=match["tp"]/len(centers)
    return {"weak_identity_recovered":int(_weak_match(pool,centers)),"precision":precision,"recall":recall,
            "f1":2*precision*recall/max(precision+recall,1e-12),"duplicates":match["duplicates"]}


def _run_grid(distances,ratios,correlations,shape,seeds,seed_base)->list[dict]:
    rows=[]
    for distance in distances:
        for ratio in ratios:
            for correlation in correlations:
                for seed in range(seeds):
                    video,centers=_simulated_movie(np.random.default_rng(seed_base+seed),distance,ratio,correlation,shape)
                    baseline=_candidates(_score_maps(video)["spatial_context"],2);model=deblend(video)["proposals"]
                    for method,pool in (("spatial_context",baseline),("constrained_generative",model)):
                        rows.append({"distance_px":distance,"amplitude_ratio":ratio,"correlation":correlation,"shape":shape,"seed":seed,"method":method,**_metrics(pool,centers)})
    return rows


def _write(path:Path,rows:list[dict])->None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t");writer.writeheader();writer.writerows(rows)


def run_joint_generative_program(output:Path,*,empirical_profile:dict,seeds:int=6)->dict:
    output.mkdir(parents=True,exist_ok=False)
    development=_run_grid((4.,6.,8.),(.3,.5,.8),(.2,.6),"ellipse",seeds,51000)
    # One-shot locked v8 grid; no configuration selection follows this call.
    locked=_run_grid((3.,5.,7.),(.2,.4,.6,1.),(0.,.8),"crescent",seeds,52000)
    kinetic=[];normal_scores=[];reversed_scores=[]
    for seed in range(seeds*3):
        video,centers=_family_movie(np.random.default_rng(53000+seed),"baseline",4,empirical_profile)
        for direction,data in (("forward_movie",video),("reversed_movie",video[::-1])):
            fit=deblend(data);scores=fit["kinetic_likelihoods"]
            (normal_scores if direction=="forward_movie" else reversed_scores).extend(scores)
            kinetic.extend({"seed":seed,"direction":direction,"component":i,"kinetic_log_likelihood_ratio":score,
                            "positive_direction":int(score>0)} for i,score in enumerate(scores))
    labels=[1]*len(normal_scores)+[0]*len(reversed_scores);all_scores=normal_scores+reversed_scores
    kinetic_summary={"forward_mean":float(np.mean(normal_scores)),"reversed_mean":float(np.mean(reversed_scores)),
                     "direction_auc":float(roc_auc_score(labels,all_scores)),"forward_positive_fraction":float(np.mean(np.asarray(normal_scores)>0)),
                     "reversed_positive_fraction":float(np.mean(np.asarray(reversed_scores)>0))}
    def summarize(rows):
        return {method:{"weak_recovery":float(np.mean([r["weak_identity_recovered"] for r in rows if r["method"]==method])),
                        "mean_f1":float(np.mean([r["f1"] for r in rows if r["method"]==method])),
                        "duplicates":float(np.mean([r["duplicates"] for r in rows if r["method"]==method]))} for method in ("spatial_context","constrained_generative")}
    dev_summary=summarize(development);locked_summary=summarize(locked)
    gates={"development_weak_recovery_improves":dev_summary["constrained_generative"]["weak_recovery"]>dev_summary["spatial_context"]["weak_recovery"],
           "locked_weak_recovery_improves":locked_summary["constrained_generative"]["weak_recovery"]>locked_summary["spatial_context"]["weak_recovery"],
           "locked_f1_noninferior":locked_summary["constrained_generative"]["mean_f1"]>=locked_summary["spatial_context"]["mean_f1"]-.02,
           "kinetic_direction_auc_ge_0p7":kinetic_summary["direction_auc"]>=.7,
           "kinetic_forward_positive_ge_0p7":kinetic_summary["forward_positive_fraction"]>=.7,
           "kinetic_reversed_positive_le_0p3":kinetic_summary["reversed_positive_fraction"]<=.3}
    result={"schema_version":1,"status":"completed_joint_generative_program","frozen_config":asdict(FROZEN_CONFIG),
            "development_contract":{"distances":[4,6,8],"ratios":[.3,.5,.8],"correlations":[.2,.6],"shape":"ellipse"},
            "locked_v8_contract":{"distances":[3,5,7],"ratios":[.2,.4,.6,1],"correlations":[0,.8],"shape":"crescent","used_for_tuning":False},
            "development":dev_summary,"locked_v8":locked_summary,"source_specific_kinetics":kinetic_summary,
            "gates":gates,"gate_pass_count":sum(gates.values()),"gate_total":len(gates),"scope":"exact-truth simulator evidence, not biological validation",
            "scientific_audit":{"truth":"exact simulator identities","model":"constrained alternating nonnegative spatial-temporal factors","human_review":"not used"}}
    _write(output/"development_deblending.tsv",development);_write(output/"locked_v8_deblending.tsv",locked);_write(output/"source_specific_kinetics.tsv",kinetic)
    atomic_json(output/"summary.json",result);atomic_json(output/"validation.json",{"status":"passed","development_rows":len(development),"locked_rows":len(locked),"kinetic_rows":len(kinetic),"locked_used_for_tuning":False,"tables":3})
    atomic_json(output/"llm_context.json",{"entry_point":"summary.json","tables":["development_deblending.tsv","locked_v8_deblending.tsv","source_specific_kinetics.tsv"],"evidence_boundary":"simulator only; locked v8 one-shot evaluation"})
    atomic_json(output/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","llm_context.json","development_deblending.tsv","locked_v8_deblending.tsv","source_specific_kinetics.tsv"]})
    atomic_text(output/"REPORT.md",f"# Joint generative deblending\n\nFrozen constrained model completed development and one-shot locked v8 evaluation; {sum(gates.values())}/{len(gates)} gates passed.\n")
    return result
