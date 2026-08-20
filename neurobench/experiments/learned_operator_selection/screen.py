"""Sequential, non-caching S1 fixed-operator response screen."""
from __future__ import annotations
import csv, json, resource, time
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import spearmanr
from neurobench.algorithms.local_whitening import apply_spatial_center_response,apply_temporal_center_response,deterministic_spatial_samples,deterministic_temporal_samples,diagnostics,fit_fractional_whitening,quiet_standardize,raw_preserving_blend
from neurobench.experiments.frame_difference import _atomic_json
from .data import load_and_validate_labels,raw_direct_stack
from .design import unique_prefix
from .evaluation import evaluate_stack

def _signed(source,config):
    f=config.frames; raw=np.asarray(source[f.review_start_ui-1:f.review_end_ui],dtype=np.float32); q=f.quiet_end_ui-f.review_start_ui+1
    center=np.median(raw[:q],axis=0); low,high=np.percentile(raw[:q,::4,::4],[1,99.9]); scale=max(float(high-low),1.0)
    return ((raw-center)/scale).astype(np.float32),q

def _trace(raw,evidence,labels,config):
    values=[]; q=config.frames.quiet_end_ui-config.frames.review_start_ui+1; yy,xx=np.mgrid[-2:3,-2:3]; disk=(xx*xx+yy*yy)<=4
    for row in labels:
        x,y=int(round(row["x_px"])),int(round(row["y_px"])); ys=np.clip(y+yy[disk],0,raw.shape[1]-1); xs=np.clip(x+xx[disk],0,raw.shape[2]-1)
        a=raw[:,ys,xs].mean(axis=1); b=evidence[:,ys,xs].mean(axis=1)
        def z(v):
            c=np.median(v[:q]); s=max(1.4826*np.median(np.abs(v[:q]-c)),1e-6); return (v-c)/s
        start=max(0,int(row["start_frame_ui"])-config.frames.review_start_ui-2); stop=min(len(a),int(row["end_frame_ui"])-config.frames.review_start_ui+3)
        rho=spearmanr(z(a)[start:stop],z(b)[start:stop]).statistic
        if np.isfinite(rho): values.append(float(rho))
    return float(np.median(values)) if values else float("nan")

def _operator(signed,q,row):
    family=row["family"]; kwargs=dict(shrinkage=row["shrinkage"],eigen_floor_ratio=row["eigen_floor_ratio"])
    if family=="spatial":
        w=row["spatial_width_px"]; fit=fit_fractional_whitening(deterministic_spatial_samples(signed[:q],w,maximum_samples=4096),exponent=row["spatial_exponent"],**kwargs); z=apply_spatial_center_response(signed,fit,w); health=diagnostics(fit)
    elif family=="temporal":
        w=row["temporal_width_frames"]; fit=fit_fractional_whitening(deterministic_temporal_samples(signed[:q],w,maximum_samples=16384),exponent=row["temporal_exponent"],**kwargs); z=apply_temporal_center_response(signed,fit,w); health=diagnostics(fit)
    else:
        sw,tw=row["spatial_width_px"],row["temporal_width_frames"]; sf=fit_fractional_whitening(deterministic_spatial_samples(signed[:q],sw,maximum_samples=4096),exponent=row["spatial_exponent"],**kwargs); spatial=apply_spatial_center_response(signed,sf,sw); tf=fit_fractional_whitening(deterministic_temporal_samples(spatial[:q],tw,maximum_samples=16384),exponent=row["temporal_exponent"],**kwargs); z=apply_temporal_center_response(spatial,tf,tw); health={"condition_number":max(sf.condition_number,tf.condition_number),"effective_rank_fraction":min(sf.effective_rank_fraction,tf.effective_rank_fraction),"finite_output_fraction":float(np.mean(np.isfinite(z))),"resolved":sf.resolved and tf.resolved}
    mixed=raw_preserving_blend(signed,z,row["raw_blend"]); standardized,scale=quiet_standardize(mixed,q); return np.maximum(standardized,0).astype(np.float32),health,scale

def _family_summary(family,rows,raw):
    selected=[]
    for heldout in range(1,5):
        best=max(rows,key=lambda r:np.mean([x["recall_at_58"] for x in r["outer_folds"] if x["burst_id"]!=heldout])); fold=next(x for x in best["outer_folds"] if x["burst_id"]==heldout); selected.append({"heldout_burst":heldout,"design_id":best["design_id"],**fold})
    macro=float(np.mean([x["recall_at_58"] for x in selected])); matches=sum(x["matches_at_58"] for x in selected)
    oracle=[max(next(x for x in r["outer_folds"] if x["burst_id"]==b)["recall_at_58"] for r in rows) for b in range(1,5)]; oracle_macro=float(np.mean(oracle))
    delta=macro-raw["macro_kpr_at_58"]; promising=delta>=.02 and matches-raw["matches_at_58"]>=2 and sum(x["recall_at_58"]>=raw["per_burst_recall_at_58"][i] for i,x in enumerate(selected))>=3
    complementary=(oracle_macro-raw["macro_kpr_at_58"]>=.04 or sum(max(next(x for x in r["outer_folds"] if x["burst_id"]==b)["matches_at_58"] for r in rows) for b in range(1,5))-raw["matches_at_58"]>=4) and len({x["design_id"] for x in selected})>1
    return {"family":family,"fixed_select_macro_kpr_at_58":macro,"fixed_select_matches":matches,"delta_vs_raw":delta,"fixed_select_folds":selected,"oracle_macro_kpr_at_58":oracle_macro,"classification":"promising" if promising else "complementary" if complementary else "weak_or_dead"}

def run_screen(config,*,prefix_size:int=8,diagnostic_rescue:bool=False)->dict[str,Any]:
    suffix="_calibration_rescue" if diagnostic_rescue else ""
    root=config.output_dir/"stages"/"S1_OPERATOR_SCREEN"/f"prefix_{prefix_size}{suffix}"
    if root.exists(): raise FileExistsError(f"screen root exists: {root}")
    root.mkdir(parents=True); source=np.load(config.source_video,mmap_mode="r",allow_pickle=False); labels=load_and_validate_labels(config.labels_tsv,config.label_summary,tuple(source.shape[1:])); signed,q=_signed(source,config); raw_stack,_=raw_direct_stack(source,config.frames.review_start_ui,config.frames.review_end_ui,config.frames.quiet_end_ui); raw=evaluate_stack(raw_stack,labels,config); rows=[]; started=time.monotonic()
    for family in ("spatial","temporal","separable_spatiotemporal"):
        for design in unique_prefix(family,prefix_size,seed=config.design.master_seed):
            lane,health,scale=_operator(signed,q,design); metrics=evaluate_stack(lane,labels,config); rows.append({**design,"design_id":f"{family}_{design['design_index']:03d}",**metrics,"trace_preserve":_trace(np.maximum(signed,0),lane,labels,config),"numerical_health":health,"quiet_standardization":scale})
    families=[_family_summary(f,[r for r in rows if r["family"]==f],raw) for f in ("spatial","temporal","separable_spatiotemporal")]; advance=any(r["classification"] in {"promising","complementary"} for r in families)
    summary={"stage":"S1_OPERATOR_SCREEN","prefix_size":prefix_size,"diagnostic_rescue":diagnostic_rescue,"raw_direct":raw,"families":families,"operators":rows,"runtime_seconds":time.monotonic()-started,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"interpretation":"within-video sparse-positive screen; unmatched candidates unknown"}; decision={"stage":"S1_OPERATOR_SCREEN","decision":"advance" if advance else "stop_branch","primary_metric":"Macro-KPR@58","comparator":"Raw Direct and leakage-safe Fixed-Select","reason_codes":[],"rescue_consumed":diagnostic_rescue,"next_allowed_stages":["S2_GLOBAL_MIXTURE"] if advance else ["S5_ICA_UTILITY"]}
    _atomic_json(root/"stage_summary.json",summary); _atomic_json(root/"decision.json",decision); return {"summary":summary,"decision":decision,"root":str(root)}
