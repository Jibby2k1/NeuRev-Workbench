"""Twelve-family exact-truth automated challenge suite for proposal features."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, shift
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from .contracts import atomic_json, atomic_text
from .major_next_steps import (
    _candidates, _family_movie, _footprint, _match, _nms_scored, _score_maps,
)


FAMILIES = (
    "null_control", "identity_suppression", "source_count_stopping", "weak_neighbor_frontier",
    "empirical_noise_spectrum", "motion_disentanglement", "morphology_interventions",
    "severity_curves", "ood_detection", "cross_simulator", "parameter_stability", "negative_controls",
)


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t");writer.writeheader();writer.writerows(rows)


def _threshold(null_movies: list[list[float]], target: float=2.) -> float:
    pooled=np.sort(np.asarray([v for row in null_movies for v in row]))[::-1]
    return float(pooled[min(len(pooled)-1,max(0,int(target*len(null_movies))-1))])


def _metrics(pool, centers, threshold, radius=3.) -> dict:
    selected=[row for row in pool if row[2]>=threshold];match=_match(selected,centers,radius)
    precision=match["tp"]/len(selected) if selected else 1.;recall=match["tp"]/len(centers) if centers else float("nan")
    return {"proposals":len(selected),"fp":match["fp"],"precision":precision,"recall":recall,
            "f1":2*precision*recall/max(precision+recall,1e-12) if centers else float("nan"),
            "duplicates":match["duplicates"],"localization_px":float(np.mean(match["localization"])) if match["localization"] else float("nan")}


def _colored_challenge(video: np.ndarray, rng: np.random.Generator, kind: str, severity: float=1.) -> np.ndarray:
    out=video.copy();t,h,w=out.shape;yy,xx=np.mgrid[:h,:w]
    if kind=="colored_noise":
        state=np.zeros((h,w))
        for frame in range(t):
            state=.82*state+gaussian_filter(rng.normal(size=(h,w)),1.2)
            out[frame]+=severity*.28*state
    elif kind=="stripes":
        stripe=np.sin(xx/2.3)+.6*np.cos(yy/3.1)
        out += severity*.25*np.sin(np.arange(t)[:,None,None]/5)*stripe
    elif kind=="transients":
        for frame in rng.choice(np.arange(8,t-8),size=5,replace=False):
            cy,cx=rng.uniform(5,h-5),rng.uniform(5,w-5)
            out[frame]+=severity*2*np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*1.2**2))
    return out


def _second_simulator(rng: np.random.Generator, source_count: int) -> tuple[np.ndarray,list[tuple[float,float]]]:
    t,h,w=90,64,64;yy,xx=np.mgrid[:h,:w];centers=[]
    while len(centers)<source_count:
        candidate=(float(rng.uniform(8,h-8)),float(rng.uniform(8,w-8)))
        if all(np.hypot(candidate[0]-y,candidate[1]-x)>6 for y,x in centers): centers.append(candidate)
    video=np.zeros((t,h,w),float);background=gaussian_filter(rng.gamma(2,.2,size=(h,w)),5)
    video+=background
    for index,(cy,cx) in enumerate(centers):
        events=(rng.random(t)<.055).astype(float);kernel=(np.exp(-np.arange(28)/8)-np.exp(-np.arange(28)/1.2))
        trace=np.convolve(events,kernel,mode="full")[:t]*rng.uniform(.7,1.5)
        theta=rng.uniform(0,np.pi);xr=(xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta);yr=-(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta)
        footprint=np.exp(-(xr**2/(2*2.8**2)+yr**2/(2*1.2**2)))
        video+=trace[:,None,None]*footprint
    photons=np.maximum(video+2,0)*20
    video=rng.poisson(photons).astype(float)/20+rng.normal(scale=.12,size=video.shape)
    return video.astype(np.float32),centers


def _pixel_auc(score: np.ndarray, centers: list[tuple[float,float]], radius=3.) -> float:
    yy,xx=np.mgrid[:score.shape[0],:score.shape[1]];labels=np.zeros(score.shape,bool)
    for cy,cx in centers: labels|=(yy-cy)**2+(xx-cx)**2<=radius**2
    ranks=rankdata(score.ravel());pos=labels.ravel();n=pos.sum();m=len(pos)-n
    return float((ranks[pos].sum()-n*(n+1)/2)/(n*m))


def _null_control(profile: dict, seeds: int) -> tuple[list[dict],dict]:
    rngs=[np.random.default_rng(21000+s) for s in range(seeds)]
    raw_null=[];z_null=[]
    for rng in rngs:
        video,_=_family_movie(rng,"baseline",0,profile);score=_score_maps(video)["spatial_context"]
        values=[v for _,_,v in _candidates(score,32)];raw_null.append(values)
        med=np.median(score);mad=max(np.median(np.abs(score-med))*1.4826,1e-6);z_null.append([(v-med)/mad for v in values])
    raw_t=_threshold(raw_null);z_t=_threshold(z_null);rows=[]
    for kind in ("colored_noise","stripes","transients"):
        for seed in range(seeds):
            rng=np.random.default_rng(21100+100*("colored_noise","stripes","transients").index(kind)+seed)
            video,centers=_family_movie(rng,"baseline",4,profile);video=_colored_challenge(video,rng,kind)
            score=_score_maps(video)["spatial_context"];pool=_candidates(score,32)
            med=np.median(score);mad=max(np.median(np.abs(score-med))*1.4826,1e-6);zpool=[(y,x,(v-med)/mad) for y,x,v in pool]
            for method,p,t in (("raw",pool,raw_t),("movie_robust_z",zpool,z_t)):
                rows.append({"challenge":kind,"seed":seed,"method":method,**_metrics(p,centers,t)})
    aggregate={m:float(np.mean([r["f1"] for r in rows if r["method"]==m])) for m in ("raw","movie_robust_z")}
    return rows,{"thresholds":{"raw":raw_t,"movie_robust_z":z_t},"mean_f1":aggregate}


def _suppression(profile: dict,seeds: int) -> tuple[list[dict],dict]:
    rows=[]
    for distance in (3.,5.,8.):
        for corr in (0.,.8):
            for seed in range(seeds):
                rng=np.random.default_rng(22000+int(distance*100)+int(corr*10)+seed)
                # Existing two-source simulator is the closest-neighbor identity stressor.
                from .major_next_steps import _simulated_movie
                video,centers=_simulated_movie(rng,distance,.55,corr,"crescent");maps=_score_maps(video)
                score=maps["spatial_context"];maxima=score==maximum_filter(score,size=3);ys,xs=np.nonzero(maxima)
                order=np.argsort(score[ys,xs])[::-1][:32];raw=[(int(ys[i]),int(xs[i]),float(score[ys[i],xs[i]])) for i in order]
                methods={"point_nms_r2":_nms_scored(raw,2),"footprint_nms_r4":_nms_scored(raw,4)}
                trace_pool=[]
                for y,x,s in raw:
                    keep=True
                    for ky,kx,_ in trace_pool:
                        if np.hypot(y-ky,x-kx)<=4 and np.corrcoef(video[:,y,x],video[:,ky,kx])[0,1]>.8: keep=False;break
                    if keep: trace_pool.append((y,x,s))
                methods["trace_aware_r4"]=trace_pool
                for method,pool in methods.items(): rows.append({"distance_px":distance,"correlation":corr,"seed":seed,"method":method,**_metrics(pool[:4],centers,-np.inf)})
    agg={m:{k:float(np.nanmean([r[k] for r in rows if r["method"]==m])) for k in ("recall","duplicates","f1")} for m in {r["method"] for r in rows}}
    return rows,agg


def _stopping(profile: dict,seeds: int) -> tuple[list[dict],dict]:
    null=[]
    for seed in range(seeds*3):
        score=_score_maps(_family_movie(np.random.default_rng(23000+seed),"baseline",0,profile)[0])["spatial_context"]
        null.extend(v for _,_,v in _candidates(score,32))
    null=np.asarray(null);rows=[]
    for count in (2,4,8):
        for seed in range(seeds):
            video,centers=_family_movie(np.random.default_rng(23100+count*10+seed),"combined_shift",count,profile);pool=_candidates(_score_maps(video)["spatial_context"],32)
            pvals=np.asarray([(1+np.sum(null>=v))/(1+len(null)) for _,_,v in pool]);order=np.argsort(pvals);q=.1
            valid=np.where(pvals[order]<=q*np.arange(1,len(order)+1)/len(order))[0]
            cutoff=pvals[order[valid[-1]]] if len(valid) else -1
            selected=[row for row,p in zip(pool,pvals) if p<=cutoff]
            rows.append({"source_count":count,"seed":seed,"method":"bh_fdr_0p1",**_metrics(selected,centers,-np.inf)})
    return rows,{"mean_proposals_by_source_count":{str(c):float(np.mean([r["proposals"] for r in rows if r["source_count"]==c])) for c in (2,4,8)},"mean_f1":float(np.mean([r["f1"] for r in rows]))}


def _weak_frontier(seeds:int) -> tuple[list[dict],dict]:
    from .major_next_steps import _simulated_movie
    rows=[]
    for distance in (3.,5.,7.,10.):
        for ratio in (.15,.3,.5,.75,1.):
            for corr in (0.,.8):
                recovered=[]
                for seed in range(seeds):
                    video,centers=_simulated_movie(np.random.default_rng(24000+seed),distance,ratio,corr,"crescent")
                    pool=_candidates(_score_maps(video)["spatial_context"],4)
                    dist=np.asarray([[np.hypot(y-cy,x-cx) for cy,cx in centers] for y,x,_ in pool])
                    rr,cc=linear_sum_assignment(dist)
                    recovered.append(any(c==0 and dist[r,c]<=3 for r,c in zip(rr,cc)))
                rows.append({"distance_px":distance,"amplitude_ratio":ratio,"correlation":corr,"weak_recovery":float(np.mean(recovered))})
    boundary={str(d):min((r["amplitude_ratio"] for r in rows if r["distance_px"]==d and r["weak_recovery"]>=.8),default=None) for d in (3.,5.,7.,10.)}
    return rows,{"minimum_ratio_for_80pct_recovery":boundary}


def _noise_spectrum(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for model in ("scalar_white","ar1_spatial"):
        for seed in range(seeds):
            rng=np.random.default_rng(25000+seed);video,centers=_family_movie(rng,"empirical_noise",4,profile)
            if model=="ar1_spatial": video=_colored_challenge(video,rng,"colored_noise",1.5)
            score=_score_maps(video)["spatial_context"]
            rows.append({"noise_model":model,"seed":seed,"pixel_auc":_pixel_auc(score,centers),"top8_recall":_metrics(_candidates(score,8),centers,-np.inf)["recall"]})
    return rows,{m:{k:float(np.mean([r[k] for r in rows if r["noise_model"]==m])) for k in ("pixel_auc","top8_recall")} for m in ("scalar_white","ar1_spatial")}


def _motion(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for relation in ("none","asynchronous","burst_synchronized"):
        for seed in range(seeds):
            rng=np.random.default_rng(26000+seed);video,centers=_family_movie(rng,"baseline",4,profile)
            projection=np.mean(video,axis=0)
            if relation!="none":
                for t in range(len(video)):
                    amplitude=1.5*np.sin(t/4) if relation=="asynchronous" else (2. if np.std(video[t])>np.median(np.std(video,axis=(1,2))) else 0.)
                    video[t]+=shift(projection,(0,amplitude),order=1,mode="nearest")-projection
            score=_score_maps(video)["spatial_context"];rows.append({"motion_relation":relation,"seed":seed,"pixel_auc":_pixel_auc(score,centers),**_metrics(_candidates(score,8),centers,-np.inf)})
    return rows,{r:float(np.mean([x["f1"] for x in rows if x["motion_relation"]==r])) for r in ("none","asynchronous","burst_synchronized")}


def _morphology(seeds:int) -> tuple[list[dict],dict]:
    rows=[];yy,xx=np.mgrid[:48,:48];centers=[(24.,20.),(24.,29.)]
    for morphology in ("ellipse","crescent","ring","fragmented"):
        for temporal in ("calcium","impulse"):
            for seed in range(seeds):
                rng=np.random.default_rng(27000+seed);video=rng.normal(scale=.25,size=(72,48,48));kernel=np.exp(-np.arange(18)/5)*(1-np.exp(-np.arange(18)/1.5))
                for index,(cy,cx) in enumerate(centers):
                    events=(rng.normal(size=72)>1.2).astype(float);trace=np.convolve(events,kernel,mode="full")[:72] if temporal=="calcium" else events
                    if morphology in ("ellipse","crescent"): fp=_footprint(yy,xx,cy,cx,morphology)
                    elif morphology=="ring": fp=np.clip(np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*3**2))-np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*1.5**2)),0,None)
                    else: fp=_footprint(yy,xx,cy-1,cx-1,"ellipse")*.55+_footprint(yy,xx,cy+1.5,cx+1.5,"ellipse")*.55
                    video+=trace[:,None,None]*fp
                maps=_score_maps(video);rows.append({"morphology":morphology,"temporal":temporal,"seed":seed,
                    "carrier_auc":_pixel_auc(maps["carrier"],centers),"context_auc":_pixel_auc(maps["spatial_context"],centers),"kinetic_auc":_pixel_auc(maps["kinetic"],centers)})
    return rows,{m:float(np.mean([r["context_auc"] for r in rows if r["morphology"]==m])) for m in ("ellipse","crescent","ring","fragmented")}


def _severity(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for nuisance in ("colored_noise","stripes","transients"):
        for severity in (0.,.5,1.,2.,4.):
            for seed in range(seeds):
                rng=np.random.default_rng(28000+seed);video,centers=_family_movie(rng,"baseline",4,profile);video=_colored_challenge(video,rng,nuisance,severity)
                rows.append({"nuisance":nuisance,"severity":severity,"seed":seed,**_metrics(_candidates(_score_maps(video)["spatial_context"],8),centers,-np.inf)})
    boundaries={n:min((s for s in (0.,.5,1.,2.,4.) if np.mean([r["recall"] for r in rows if r["nuisance"]==n and r["severity"]==s])<.7),default=None) for n in ("colored_noise","stripes","transients")}
    return rows,{"first_severity_recall_below_0p7":boundaries}


def _ood(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    def signature(video):
        maps=_score_maps(video);s=maps["spatial_context"];c=[v for _,_,v in _candidates(s,24)]
        return [np.median(s),np.median(np.abs(s-np.median(s)))*1.4826,np.quantile(s,.99),np.mean(c),np.std(c)]
    train=[]
    for seed in range(seeds*3): train.append(signature(_family_movie(np.random.default_rng(29000+seed),"baseline",4,profile)[0]))
    fit=LedoitWolf().fit(train);rows=[]
    for label,kind in ((0,"baseline"),(1,"colored_noise"),(1,"stripes"),(1,"transients")):
        for seed in range(seeds):
            rng=np.random.default_rng(29100+100*label+seed);video,_=_family_movie(rng,"baseline",4,profile)
            if kind!="baseline": video=_colored_challenge(video,rng,kind,2.)
            score=float(fit.mahalanobis([signature(video)])[0]);rows.append({"kind":kind,"seed":seed,"ood_label":label,"ood_score":score})
    auc=float(roc_auc_score([r["ood_label"] for r in rows],[r["ood_score"] for r in rows]));return rows,{"ood_auc":auc}


def _cross_sim(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    # Threshold is frozen on first-simulator nulls.
    null=[]
    for seed in range(seeds*2): null.append([v for _,_,v in _candidates(_score_maps(_family_movie(np.random.default_rng(30000+seed),"baseline",0,profile)[0])["spatial_context"],32)])
    threshold=_threshold(null);rows=[]
    for simulator in ("primary","independent_second"):
        for count in (2,4,8):
            for seed in range(seeds):
                rng=np.random.default_rng(30100+count*10+seed)
                video,centers=(_family_movie(rng,"baseline",count,profile) if simulator=="primary" else _second_simulator(rng,count))
                rows.append({"simulator":simulator,"source_count":count,"seed":seed,**_metrics(_candidates(_score_maps(video)["spatial_context"],32),centers,threshold)})
    return rows,{s:float(np.mean([r["f1"] for r in rows if r["simulator"]==s])) for s in ("primary","independent_second")}


def _stability(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for nms in (3,5,7):
        for radius in (2.,3.,4.):
            for seed in range(seeds):
                video,centers=_family_movie(np.random.default_rng(31000+seed),"combined_shift",4,profile);score=_score_maps(video)["spatial_context"]
                maxima=score==maximum_filter(score,size=nms);ys,xs=np.nonzero(maxima);order=np.argsort(score[ys,xs])[::-1][:8];pool=[(int(ys[i]),int(xs[i]),float(score[ys[i],xs[i]])) for i in order]
                rows.append({"nms_window":nms,"match_radius":radius,"seed":seed,**_metrics(pool,centers,-np.inf,radius)})
    values=[r["f1"] for r in rows];return rows,{"f1_range":[float(min(values)),float(max(values))],"f1_sd":float(np.std(values))}


def _negative_controls(profile:dict,seeds:int) -> tuple[list[dict],dict]:
    rows=[]
    for seed in range(seeds):
        rng=np.random.default_rng(32000+seed);video,centers=_family_movie(rng,"baseline",4,profile);maps=_score_maps(video)
        controls={"spatial_context":maps["spatial_context"],"spatial_shuffle":rng.permutation(maps["spatial_context"].ravel()).reshape(maps["spatial_context"].shape),
                  "random_projection":rng.normal(size=maps["spatial_context"].shape),"time_reversal_kinetic":_score_maps(video[::-1])["kinetic"]}
        for name,score in controls.items(): rows.append({"control":name,"seed":seed,"pixel_auc":_pixel_auc(score,centers)})
    agg={name:float(np.mean([r["pixel_auc"] for r in rows if r["control"]==name])) for name in {r["control"] for r in rows}}
    return rows,agg


def run_automated_challenge_suite(output: Path, *, empirical_profile: dict, seeds: int=6) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    runners={
        "null_control":lambda:_null_control(empirical_profile,seeds),
        "identity_suppression":lambda:_suppression(empirical_profile,seeds),
        "source_count_stopping":lambda:_stopping(empirical_profile,seeds),
        "weak_neighbor_frontier":lambda:_weak_frontier(seeds),
        "empirical_noise_spectrum":lambda:_noise_spectrum(empirical_profile,seeds),
        "motion_disentanglement":lambda:_motion(empirical_profile,seeds),
        "morphology_interventions":lambda:_morphology(seeds),
        "severity_curves":lambda:_severity(empirical_profile,seeds),
        "ood_detection":lambda:_ood(empirical_profile,seeds),
        "cross_simulator":lambda:_cross_sim(empirical_profile,seeds),
        "parameter_stability":lambda:_stability(empirical_profile,seeds),
        "negative_controls":lambda:_negative_controls(empirical_profile,seeds),
    }
    summaries={};tables=[]
    for family in FAMILIES:
        rows,summary=runners[family]();filename=f"{family}.tsv";_write_tsv(output/filename,rows);tables.append(filename);summaries[family]=summary
    gates={
        "robust_null_improves_f1":summaries["null_control"]["mean_f1"]["movie_robust_z"]>summaries["null_control"]["mean_f1"]["raw"],
        "suppression_duplicate_gate":min(v["duplicates"] for v in summaries["identity_suppression"].values())<=.1,
        "ood_auc_ge_0p8":summaries["ood_detection"]["ood_auc"]>=.8,
        "negative_controls_near_chance":max(v for k,v in summaries["negative_controls"].items() if k!="spatial_context")<.65,
        "cross_simulator_f1_ge_0p4":summaries["cross_simulator"]["independent_second"]>=.4,
    }
    result={"schema_version":1,"status":"completed_automated_challenge_suite","families":list(FAMILIES),"seeds":seeds,
            "summaries":summaries,"gates":gates,"gate_pass_count":sum(gates.values()),"gate_total":len(gates),
            "empirical_noise_profile":empirical_profile,"scope":"exact-truth automated stress tests; not biological validation",
            "scientific_audit":{"truth":"exact simulator identities","model_annotations":"all tables","human_review":"not used"}}
    atomic_json(output/"summary.json",result)
    atomic_json(output/"validation.json",{"status":"passed","families":len(summaries),"expected_families":12,
        "tables":len(tables),"all_tables_nonempty":all((output/t).stat().st_size>40 for t in tables),"finite_gate_count":len(gates)})
    atomic_json(output/"llm_context.json",{"entry_point":"summary.json","tables":tables,"evidence_boundary":"automated simulator evidence only"})
    atomic_json(output/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","llm_context.json",*tables]})
    atomic_text(output/"REPORT.md",f"# Automated challenge suite\n\nCompleted all 12 prespecified families. {sum(gates.values())}/{len(gates)} headline gates passed. See `summary.json` and the family TSVs.\n")
    return result
