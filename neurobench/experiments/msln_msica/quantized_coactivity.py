"""Guarded GPU benchmark of quantization and causal neighborhood coactivity."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name,"4")

import numpy as np
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one
from neurobench.reports.zero_anchored_display import zero_anchored_exp2
from .artifacts import atomic_json, sha256_file, sha256_payload
from .joint_sweep import _label_overlay


def _load(path:str|Path)->dict[str,Any]:
    p=Path(path).resolve(); c=json.loads(p.read_text()); required={"schema_version","experiment_id","source","design","evaluation","compute","outputs"}
    if set(c)!=required or c["schema_version"]!=1: raise ValueError("exact schema-v1 contract required")
    for k in ("raw_path","msica_path","msln_path","labels_path"): c["source"][k]=str((p.parent/c["source"][k]).resolve())
    c["outputs"]["root_dir"]=str((p.parent/c["outputs"]["root_dir"]).resolve()); c["_config_path"]=str(p); _validate(c); return c


def _validate(c):
    d=c["design"]; e=c["evaluation"]
    if d["quantization_levels"]!=[8,16,32,64] or d["quantization_mappings"]!=["uniform","sqrt_companded","quiet_cdf"] or d["quantization_dynamics"]!=["memoryless","causal_hysteresis"]: raise ValueError("quantizer grid is frozen")
    if d["quiet_threshold_quantiles"]!=[.99,.999,.9999] or d["spatial_radii_px"]!=[1,2,3] or d["causal_windows_frames"]!=[1,3,5] or d["required_support"]!=[.25,.5,.75]: raise ValueError("coactivity grid is frozen")
    if d["operators"]!=["hard_max_min","hard_q90_identity","soft_q90_identity","soft_q90_median"] or d["pooling"]!="top5_mean": raise ValueError("operator contract is frozen")
    if e["candidate_budgets"]!=[20,40,58,80,100] or e["unmatched_candidates"]!="unknown": raise ValueError("evaluation contract mismatch")
    if c["compute"]["device"]!="cuda" or c["compute"]["workers_per_gpu"]!=1 or c["compute"]["max_peak_vram_gb"]>8: raise ValueError("guarded CUDA contract required")


def _resolved(c): return {k:v for k,v in c.items() if k!="_config_path"}
def _root(c): return Path(c["outputs"]["root_dir"])


def _quantize_gpu(x,levels,mapping,dynamics,quiet_sorted):
    import cupy as cp
    if mapping=="uniform": u=cp.clip(x,0,1)
    elif mapping=="sqrt_companded": u=cp.sqrt(cp.clip(x,0,1))
    elif mapping=="quiet_cdf":
        u=cp.searchsorted(quiet_sorted,x,side="right").astype(cp.float32)/cp.float32(len(quiet_sorted)); u=cp.clip(u,0,1)
    else: raise ValueError(mapping)
    if dynamics=="memoryless": indices=cp.rint(u*cp.float32(levels-1)).astype(cp.int16)
    else:
        indices=cp.empty(u.shape,dtype=cp.int16); indices[0]=cp.rint(u[0]*cp.float32(levels-1)).astype(cp.int16)
        margin=cp.float32(.15)
        for t in range(1,len(u)):
            prior=indices[t-1]; up=(prior.astype(cp.float32)+cp.float32(.5)+margin)/cp.float32(levels-1); down=(prior.astype(cp.float32)-cp.float32(.5)-margin)/cp.float32(levels-1); candidate=cp.rint(u[t]*cp.float32(levels-1)).astype(cp.int16); indices[t]=cp.where((u[t]>up)|(u[t]<down),candidate,prior)
    q=indices.astype(cp.float32)/cp.float32(levels-1)
    return cp.square(q) if mapping=="sqrt_companded" else q


def _causal_support(binary,window):
    import cupy as cp
    if window==1: return binary.astype(cp.float32)
    cs=cp.cumsum(binary.astype(cp.float32),axis=0); out=cs.copy(); out[window:]=cs[window:]-cs[:-window]; denom=cp.minimum(cp.arange(1,len(binary)+1,dtype=cp.float32),cp.float32(window)); return out/denom[:,None,None]


def _neighborhood_gpu(x,threshold,radius,window,support_required,operator):
    import cupy as cp
    from cupyx.scipy import ndimage
    size=(1,2*radius+1,2*radius+1); binary=x>=cp.float32(threshold); spatial=ndimage.uniform_filter(binary.astype(cp.float32),size=size,mode="nearest"); support=_causal_support(spatial,window)
    if operator=="hard_max_min": return cp.where(binary,ndimage.maximum_filter(x,size=size,mode="nearest"),ndimage.minimum_filter(x,size=size,mode="nearest"))
    upper=ndimage.percentile_filter(x,90,size=size,mode="nearest")
    lower=x if operator in {"hard_q90_identity","soft_q90_identity"} else ndimage.median_filter(x,size=size,mode="nearest")
    if operator.startswith("hard_"): gate=support>=cp.float32(support_required)
    else: gate=cp.asarray(1)/(cp.asarray(1)+cp.exp(-(support-cp.float32(support_required))/cp.float32(.08)))
    return cp.where(gate,upper,lower) if gate.dtype==cp.bool_ else lower+gate*(upper-lower)


def _pool_proposals(x,c):
    import cupy as cp
    start_ui=c["source"]["review_interval_ui"][0]; limit=max(c["evaluation"]["candidate_budgets"]); result={}
    for b,interval in c["source"]["burst_intervals_ui"].items():
        a=interval[0]-start_ui; z=interval[1]-start_ui+1; block=x[a:z]; pooled=cp.mean(cp.partition(block,len(block)-5,axis=0)[-5:],axis=0); peaks=extract_local_maxima(cp.asnumpy(pooled),c["evaluation"]["nms_distance_px"],limit=limit); result[str(b)]=[[float(s),int(px),int(py)] for s,px,py in peaks]
    return result


def _metrics(x,baseline,c):
    import cupy as cp
    q=c["source"]["quiet_frames"]; starts=[]
    for interval in c["source"]["burst_intervals_ui"].values(): starts.extend(range(interval[0]-c["source"]["review_interval_ui"][0],interval[1]-c["source"]["review_interval_ui"][0]+1))
    event=x[cp.asarray(starts)]; quiet=x[:q]; sample=x[::7,::5,::5].ravel(); base=baseline[::7,::5,::5].ravel(); corr=float(cp.corrcoef(sample,base)[0,1].get()); qm=float(cp.mean(quiet).get()); em=float(cp.mean(event).get()); zero=float(cp.mean(x==0).get()); tie=float(1-cp.unique(sample).size/sample.size)
    return {"event_mean":em,"quiet_mean":qm,"event_quiet_ratio":em/max(qm,1e-12),"integrity_correlation":corr,"zero_fraction":zero,"sample_tie_fraction":tie,"score":float(np.log1p(em/max(qm,1e-12))-2*max(0,.95-corr))}


def _evaluate(proposals,labels,c):
    totals={str(b):0 for b in c["evaluation"]["candidate_budgets"]}; rows=[]
    for burst_text,peaks0 in proposals.items():
        burst=int(burst_text); peaks=[tuple(p) for p in peaks0]; lab=[r for r in labels if int(r["burst_id"])==burst]
        for budget in c["evaluation"]["candidate_budgets"]:
            matches,_=match_peaks_one_to_one(peaks[:budget],lab,c["evaluation"]["match_radius_px"]); totals[str(budget)]+=len(matches); rows.append({"burst_id":burst,"budget":budget,"matched":len(matches),"labels":len(lab),"recall":len(matches)/len(lab)})
    return {"matched_by_budget":totals,"rows":rows,"unmatched_candidates":"unknown"}


def materialize_lane_gpu(base, lane: dict[str, Any], c: dict[str, Any], quiet_sorted):
    """Reconstruct one completed lane from its explicit artifact configuration."""
    import cupy as cp
    family, cfg = lane["family"], lane["config"]
    if family == "control": return base.copy()
    if family == "quantization": return _quantize_gpu(base,cfg["levels"],cfg["mapping"],cfg["dynamics"],quiet_sorted)
    if family == "neighborhood": return _neighborhood_gpu(base,cfg["threshold"],cfg["radius"],cfg["window"],cfg["support"],cfg["operator"])
    if family != "combined": raise ValueError(f"unsupported lane family: {family}")
    nc,qc,order=cfg["neighborhood"],cfg["quantization"],cfg["order"]
    if order=="support_then_quantize": first=_neighborhood_gpu(base,nc["threshold"],nc["radius"],nc["window"],nc["support"],nc["operator"]); result=_quantize_gpu(first,qc["levels"],qc["mapping"],qc["dynamics"],quiet_sorted)
    else:
        first=_quantize_gpu(base,qc["levels"],qc["mapping"],qc["dynamics"],quiet_sorted); threshold_q=c["design"]["quiet_threshold_quantiles"][nc["threshold_index"]]; transformed_threshold=float(cp.quantile(first[:c["source"]["quiet_frames"]],threshold_q).get()); result=_neighborhood_gpu(first,transformed_threshold,nc["radius"],nc["window"],nc["support"],nc["operator"])
    del first; return result


def preflight(config_path):
    c=_load(config_path); root=_root(c)
    if root.exists(): raise FileExistsError(root)
    for k in ("raw_path","msica_path","msln_path","labels_path"):
        if not Path(c["source"][k]).is_file(): raise FileNotFoundError(c["source"][k])
    raw=np.load(c["source"]["raw_path"],mmap_mode="r"); msln=np.load(c["source"]["msln_path"],mmap_mode="r"); labels=label_core.load_labels(Path(c["source"]["labels_path"]))
    if raw.shape!=(2359,340,573) or msln.shape!=(560,340,573) or len(labels)!=79: raise ValueError("source contract mismatch")
    root.mkdir(parents=True); _label_overlay(root,raw,labels,{"source":c["source"]}); fingerprints={k:sha256_file(Path(c["source"][k])) for k in ("raw_path","msica_path","msln_path","labels_path")}; estimated=int(msln.nbytes*6.5)
    payload={"status":"ready","full_spon_authorization_required":True,"source_shape":list(msln.shape),"estimated_peak_vram_bytes":estimated,"stage_a_quantization_lanes":24,"stage_b_neighborhood_lanes":252,"stage_c_combined_lanes":48,"total_lanes_including_control":325,"labels_preflight_use":"bounds and projection only","fingerprints":fingerprints,"config_sha256":sha256_payload(_resolved(c))}; atomic_json(root/"config.resolved.json",_resolved(c)); atomic_json(root/"preflight.json",payload); atomic_json(root/"status.json",{"status":"preflight_ready"}); return payload


def gpu_preflight(config_path):
    import cupy as cp
    c=_load(config_path); root=_root(c); pre=json.loads((root/"preflight.json").read_text()); rng=np.random.default_rng(8); x=rng.random((7,13,17),dtype=np.float32); quiet=np.sort(x[:3].ravel())[::2]; gpu=cp.asarray(x); qs=cp.asarray(quiet); qcpu=np.rint(x*15)/15; qgpu=cp.asnumpy(_quantize_gpu(gpu,16,"uniform","memoryless",qs)); n=_neighborhood_gpu(gpu,.7,1,3,.5,"soft_q90_identity"); free,total=cp.cuda.runtime.memGetInfo(); payload={"status":"passed" if np.max(np.abs(qcpu-qgpu))<1e-6 and cp.isfinite(n).all() and free>pre["estimated_peak_vram_bytes"] else "failed","quantization_max_abs_error":float(np.max(np.abs(qcpu-qgpu))),"neighborhood_finite":bool(cp.isfinite(n).all()),"cuda_free_bytes":int(free),"cuda_total_bytes":int(total),"estimated_peak_vram_bytes":pre["estimated_peak_vram_bytes"]}; atomic_json(root/"gpu_preflight.json",payload); 
    if payload["status"]!="passed": raise RuntimeError(payload)
    return payload


def run(config_path,authorize_full_spon=False):
    if not authorize_full_spon: raise PermissionError("--authorize-full-spon required")
    import cupy as cp
    c=_load(config_path); root=_root(c)
    if json.loads((root/"gpu_preflight.json").read_text())["status"]!="passed": raise RuntimeError("GPU preflight required")
    atomic_json(root/"status.json",{"status":"running","started_at":datetime.now(timezone.utc).isoformat()}); signed=np.load(c["source"]["msln_path"],mmap_mode="r"); gn=zero_anchored_exp2(signed,alpha=c["design"]["gn_alpha"],positive_max=float(np.max(signed))); base=cp.asarray(gn,dtype=cp.float32); quiet_cpu=np.sort(np.asarray(gn[:c["source"]["quiet_frames"]:2,::4,::4]).ravel()); quiet_sorted=cp.asarray(quiet_cpu[::max(1,len(quiet_cpu)//200000)]); thresholds=[float(np.quantile(quiet_cpu,q)) for q in c["design"]["quiet_threshold_quantiles"]]; rows=[]; started=time.monotonic()
    base_row={"lane_id":"float_gn_control","family":"control","config":{},"metrics":_metrics(base,base,c),"proposals":_pool_proposals(base,c)}; rows.append(base_row)
    quant_rows=[]
    for levels in c["design"]["quantization_levels"]:
      for mapping in c["design"]["quantization_mappings"]:
       for dynamics in c["design"]["quantization_dynamics"]:
        out=_quantize_gpu(base,levels,mapping,dynamics,quiet_sorted); row={"lane_id":f"q_{levels}_{mapping}_{dynamics}","family":"quantization","config":{"levels":levels,"mapping":mapping,"dynamics":dynamics},"metrics":_metrics(out,base,c),"proposals":_pool_proposals(out,c)}; quant_rows.append(row); rows.append(row); del out
    neighborhood_rows=[]
    for ti,threshold in enumerate(thresholds):
     for radius in c["design"]["spatial_radii_px"]:
      for window in c["design"]["causal_windows_frames"]:
       for support in c["design"]["required_support"]:
        for operator in c["design"]["operators"]:
         if operator=="hard_max_min" and (window!=1 or support!=.5): continue
         out=_neighborhood_gpu(base,threshold,radius,window,support,operator); cfg={"threshold_index":ti,"threshold":threshold,"radius":radius,"window":window,"support":support,"operator":operator}; row={"lane_id":f"n_t{ti}_r{radius}_w{window}_s{str(support).replace('.','p')}_{operator}","family":"neighborhood","config":cfg,"metrics":_metrics(out,base,c),"proposals":_pool_proposals(out,c)}; neighborhood_rows.append(row); rows.append(row); del out
    quant_final=sorted(quant_rows,key=lambda r:(-r["metrics"]["score"],r["lane_id"]))[:c["design"]["quantization_finalists"]]; neighbor_final=sorted(neighborhood_rows,key=lambda r:(-r["metrics"]["score"],r["lane_id"]))[:c["design"]["neighborhood_finalists"]]; combined=[]
    for nr in neighbor_final:
      cfg=nr["config"]
      for qr in quant_final:
       qc=qr["config"]
       for order in c["design"]["combined_orders"]:
        if order=="quantize_then_support": first=_quantize_gpu(base,qc["levels"],qc["mapping"],qc["dynamics"],quiet_sorted); transformed_threshold=float(cp.quantile(first[:c["source"]["quiet_frames"]],c["design"]["quiet_threshold_quantiles"][cfg["threshold_index"]]).get()); out=_neighborhood_gpu(first,transformed_threshold,cfg["radius"],cfg["window"],cfg["support"],cfg["operator"]); del first
        else: first=_neighborhood_gpu(base,cfg["threshold"],cfg["radius"],cfg["window"],cfg["support"],cfg["operator"]); out=_quantize_gpu(first,qc["levels"],qc["mapping"],qc["dynamics"],quiet_sorted); del first
        row={"lane_id":f"c_{order}__{nr['lane_id']}__{qr['lane_id']}","family":"combined","config":{"order":order,"neighborhood":cfg,"quantization":qc},"metrics":_metrics(out,base,c),"proposals":_pool_proposals(out,c)}; combined.append(row); rows.append(row); del out
    frozen=sorted(rows,key=lambda r:(-r["metrics"]["score"],r["lane_id"]))[:12]; labels=label_core.load_labels(Path(c["source"]["labels_path"])); evaluated=[]
    for row in rows: row["expert"]=_evaluate(row["proposals"],labels,c); evaluated.append(row)
    budget=str(c["evaluation"]["guardrail_budget"]); best=max(evaluated,key=lambda r:(r["expert"]["matched_by_budget"][budget],r["metrics"]["score"])); frozen_ids={r["lane_id"] for r in frozen}; frozen_best=max(frozen,key=lambda r:(r["expert"]["matched_by_budget"][budget],r["metrics"]["score"])); summary={"status":"computation_complete_audit_pending","lane_count":len(rows),"thresholds":thresholds,"label_free_frozen_lanes":[r["lane_id"] for r in frozen],"frozen_best_lane":frozen_best["lane_id"],"frozen_best_matches":frozen_best["expert"]["matched_by_budget"][budget],"posthoc_best_lane":best["lane_id"],"posthoc_best_matches":best["expert"]["matched_by_budget"][budget],"float_control_matches":base_row["expert"]["matched_by_budget"][budget],"runtime_seconds":time.monotonic()-started,"unmatched_candidates":"unknown"}; atomic_json(root/"results.json",{"summary":summary,"lanes":evaluated}); atomic_json(root/"status.json",summary); cp.get_default_memory_pool().free_all_blocks(); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("preflight","gpu-preflight","run")); p.add_argument("--config",required=True); p.add_argument("--authorize-full-spon",action="store_true"); a=p.parse_args(); result=preflight(a.config) if a.action=="preflight" else gpu_preflight(a.config) if a.action=="gpu-preflight" else run(a.config,a.authorize_full_spon); print(json.dumps(result,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
