"""Factorial spatiotemporal diagnostic for the Spon Ca Burst contrast detector."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import core as v1


@dataclass(frozen=True)
class DiagnosticConfig:
    experiment_id: str
    source_video: Path
    kalman_residual: Path
    kalman_summary: Path
    labels_tsv: Path
    label_summary: Path
    source_workbook: Path
    design_document: Path
    output_dir: Path
    quiet_start_ui: int
    quiet_end_ui: int
    spatial_sigma_px: float
    temporal_sigma_frames: float
    clip_z: float
    epochs: int
    fixed_seed: int
    jitter_seeds: tuple[int, ...]
    init_jitter_std: float
    frame_batch: int
    cpu_threads: int
    max_ram_mib: int
    max_gpu_memory_mib: int
    support_px: int = 21
    tolerance_px: int = 4
    nms_distance_px: int = 6

    @classmethod
    def load(cls, path: str | Path) -> "DiagnosticConfig":
        source=Path(path).resolve(); raw=json.loads(source.read_text()); root=source.parent
        p=lambda name:(root/raw[name]).resolve()
        pre=raw["preprocessing"]; tr=raw["training"]; res=raw["resources"]; fr=raw["frames"]
        c=cls(
            experiment_id=raw["experiment_id"],source_video=p("source_video"),kalman_residual=p("kalman_residual"),
            kalman_summary=p("kalman_summary"),labels_tsv=p("labels_tsv"),label_summary=p("label_summary"),
            source_workbook=p("source_workbook"),design_document=p("design_document"),output_dir=p("output_dir"),
            quiet_start_ui=int(fr["quiet_start_ui"]),quiet_end_ui=int(fr["quiet_end_ui"]),
            spatial_sigma_px=float(pre["spatial_sigma_px"]),temporal_sigma_frames=float(pre["temporal_sigma_frames"]),
            clip_z=float(pre["clip_z"]),epochs=int(tr["epochs"]),fixed_seed=int(tr["fixed_seed"]),
            jitter_seeds=tuple(map(int,tr["jitter_seeds"])),init_jitter_std=float(tr["init_jitter_std"]),
            frame_batch=int(res["frame_batch"]),cpu_threads=int(res["cpu_threads"]),
            max_ram_mib=int(res["max_ram_mib"]),max_gpu_memory_mib=int(res["max_gpu_memory_mib"]),
            support_px=int(raw.get("support_px",21)),tolerance_px=int(raw.get("tolerance_px",4)),nms_distance_px=int(raw.get("nms_distance_px",6)))
        if c.output_dir.exists(): raise FileExistsError(f"Output exists: {c.output_dir}")
        if len(c.jitter_seeds)!=3: raise ValueError("Diagnostic requires exactly three jitter seeds")
        return c


def factor_matrix() -> list[dict[str,Any]]:
    return [{"input":inp,"objective":obj,"initialization":init,
             "combination_id":f"{inp}__{obj}__{init}"}
            for inp in ("raw_quiet_residual","kalman_spatiotemporal")
            for obj in ("legacy_raw_score","stabilized_log_score")
            for init in ("fixed_guarded","jittered_guarded")]


def preflight(c:DiagnosticConfig,artifact_dir:Path|None=None) -> dict[str,Any]:
    import torch
    files=(c.source_video,c.kalman_residual,c.kalman_summary,c.labels_tsv,c.label_summary,c.source_workbook,c.design_document)
    missing=[str(p) for p in files if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    raw=np.load(c.source_video,mmap_mode="r"); kal=np.load(c.kalman_residual,mmap_mode="r")
    if raw.shape!=kal.shape or raw.shape!=(2359,340,573): raise ValueError(f"Input shape mismatch: {raw.shape}, {kal.shape}")
    labels=v1.load_labels(c.labels_tsv); free,total=torch.cuda.mem_get_info() if torch.cuda.is_available() else (0,0)
    ready=torch.cuda.is_available() and free//2**20>=c.max_gpu_memory_mib and v1.available_ram_mib()>=c.max_ram_mib
    payload={"schema_version":1,"ready":bool(ready),"generated_at":v1.utc_now(),"experiment_id":c.experiment_id,
             "matrix":factor_matrix(),"planned_combinations":8,"planned_outer_folds":32,
             "planned_learned_fits":16+48,
             "inputs":[{"path":str(p),"bytes":p.stat().st_size,"sha256":v1.sha256(p)} for p in files],
             "video_shape":list(raw.shape),"label_rows":len(labels),"unique_rois":len({r['roi_identity'] for r in labels}),
             "resources":{"cpu_threads":c.cpu_threads,"ram_available_mib":v1.available_ram_mib(),"ram_cap_mib":c.max_ram_mib,
                          "gpu_free_mib":free//2**20,"gpu_total_mib":total//2**20,"gpu_cap_mib":c.max_gpu_memory_mib,"frame_batch":c.frame_batch},
             "leakage_contract":"Kalman residual is causal; smoothing is unsupervised; whitening statistics use quiet frames only."}
    if artifact_dir:
        artifact_dir.mkdir(parents=True,exist_ok=True); v1.atomic_json(artifact_dir/"preflight.json",payload)
    if not ready: raise RuntimeError("Diagnostic preflight failed")
    return payload


def _raw_lane(c,labels):
    legacy=v1.Config(
        experiment_id=c.experiment_id,source_video=c.source_video,source_workbook=c.source_workbook,labels_tsv=c.labels_tsv,
        label_summary=c.label_summary,design_document=c.design_document,output_dir=c.output_dir,
        quiet_start_ui=c.quiet_start_ui,quiet_end_ui=c.quiet_end_ui,scored_start_ui=1900,scored_end_ui=2359,
        support_px=c.support_px,tolerance_px=c.tolerance_px,nms_distance_px=c.nms_distance_px,epochs=c.epochs,
        masked_seeds=tuple(range(10)),final_seeds=tuple(range(5)),device="cuda",cpu_threads=c.cpu_threads,
        frame_batch=c.frame_batch,max_ram_mib=c.max_ram_mib,max_gpu_memory_mib=c.max_gpu_memory_mib,
        min_free_disk_mib=1,max_output_mib=2048)
    quiet,bursts,_=v1._prepare_arrays(legacy,labels)
    return {"quiet_contrast":quiet,"quiet_amplitude":quiet,"bursts_contrast":bursts,"bursts_amplitude":bursts,
            "summary":{"mode":"raw quiet-median positive residual","quiet_only_normalization":True}}


def _robust_whiten(stack:np.ndarray,quiet_count:int,clip_z:float) -> tuple[np.ndarray,dict[str,float]]:
    q=stack[:quiet_count]; center=np.median(q,axis=0); mad=np.median(np.abs(q-center),axis=0)*1.4826
    positive=mad[mad>0]; floor=float(np.percentile(positive,10)) if positive.size else 1.0
    scale=np.maximum(mad,max(floor,1e-4)); z=np.clip((stack-center)/scale,0,clip_z).astype(np.float32)
    return z,{"scale_floor":floor,"quiet_z_mean":float(z[:quiet_count].mean()),"quiet_z_p99":float(np.percentile(z[:quiet_count],99))}


def _kalman_lane(c,labels):
    from scipy.ndimage import gaussian_filter
    arr=np.load(c.kalman_residual,mmap_mode="r")
    start=c.quiet_start_ui-1; stop=max(r["stop_frame_zero_exclusive"] for r in labels)
    continuous=np.asarray(arr[start:stop],dtype=np.float32)
    smooth=gaussian_filter(continuous,sigma=(c.temporal_sigma_frames,c.spatial_sigma_px,c.spatial_sigma_px),mode="nearest")
    qn=c.quiet_end_ui-c.quiet_start_ui+1
    contrast,cs=_robust_whiten(smooth,qn,c.clip_z); amplitude,as_=_robust_whiten(continuous,qn,c.clip_z)
    bursts_c={}; bursts_a={}
    for b in range(1,5):
        r=next(x for x in labels if x["burst_id"]==b); s=r["start_frame_zero"]-start; e=r["stop_frame_zero_exclusive"]-start
        bursts_c[b]=contrast[s:e]; bursts_a[b]=amplitude[s:e]
    return {"quiet_contrast":contrast[:qn],"quiet_amplitude":amplitude[:qn],"bursts_contrast":bursts_c,"bursts_amplitude":bursts_a,
            "summary":{"mode":"causal Kalman + separable Gaussian + quiet per-pixel MAD whitening",
                       "spatial_sigma_px":c.spatial_sigma_px,"temporal_sigma_frames":c.temporal_sigma_frames,
                       "contrast_whitening":cs,"amplitude_whitening":as_,"quiet_only_normalization":True}}


def prepare_lanes(c,labels): return {"raw_quiet_residual":_raw_lane(c,labels),"kalman_spatiotemporal":_kalman_lane(c,labels)}


def _paired_bags(lane,labels,radius):
    def build(quiet,bursts): return v1._bag_tensors(quiet,bursts,labels,radius)
    pc,nc,m=build(lane["quiet_contrast"],lane["bursts_contrast"]); pa,na,_=build(lane["quiet_amplitude"],lane["bursts_amplitude"])
    return pc,nc,pa,na,m


def _model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Model(nn.Module):
        def __init__(self,support,objective,jitter_std):
            super().__init__(); self.support=support; self.objective=objective
            r=support//2; yy,xx=torch.meshgrid(torch.arange(-r,r+1),torch.arange(-r,r+1),indexing="ij"); rr=(xx.float()**2+yy.float()**2).sqrt()
            kt=torch.exp(-.5*(rr/2)**2); kr=((rr>=4)&(rr<=10)).float()+.02
            rt=torch.log(torch.expm1(kt/kt.max()*2+.02)); rrw=torch.log(torch.expm1(kr/kr.max()*2+.02))
            if jitter_std: rt=rt+torch.randn_like(rt)*jitter_std; rrw=rrw+torch.randn_like(rrw)*jitter_std
            self.raw_t=nn.Parameter(rt); self.raw_r=nn.Parameter(rrw); self.raw_lambda=nn.Parameter(torch.tensor(-.25)); self.register_buffer("rr2",rr**2)
        def kernels(self):
            kt=F.softplus(self.raw_t); kr=F.softplus(self.raw_r); return kt/kt.sum(),kr/kr.sum()
        def forward(self,xc,xa):
            kt,kr=self.kernels(); p=self.support//2
            cp=F.pad(xc,(p,p,p,p),mode="reflect"); ap=F.pad(xa,(p,p,p,p),mode="reflect")
            t=F.conv2d(cp,kt[None,None]); mu=F.conv2d(cp,kr[None,None]); v=(F.conv2d(cp*cp,kr[None,None])-mu*mu).clamp_min(0)
            b=(1-(kr*kr).sum()).clamp_min(1e-4); q=((kt-kr)**2).sum().clamp_min(1e-6); contrast=F.relu((t-mu)/torch.sqrt(q*v/b+1e-6)).square()
            amp=F.conv2d(ap,kt[None,None]).clamp_min(0); lam=F.softplus(self.raw_lambda)
            if self.objective=="stabilized_log_score": return torch.log1p(contrast)+lam*torch.log1p(amp)
            return contrast+lam*torch.log1p(amp)
        def penalty(self):
            kt,kr=self.kernels(); overlap=(kt*kr).sum(); ess=1/(kr.square().sum()+1e-8); et=(kt*self.rr2).sum(); er=(kr*self.rr2).sum()
            return 2*overlap+.02*torch.relu(et+8-er).square()+.02*torch.relu(40-ess).square()
    return Model


def _bag_score(model,xc,xa,mask,tol):
    import torch
    b,t,_,h,w=xc.shape; s=model(xc.reshape(b*t,1,h,w),xa.reshape(b*t,1,h,w)).reshape(b,t,h,w)
    yy,xx=torch.meshgrid(torch.arange(h,device=s.device),torch.arange(w,device=s.device),indexing="ij"); disk=(yy-h//2)**2+(xx-w//2)**2<=tol**2
    valid=mask[:,:,None,None]&disk[None,None]; tau=.25; z=(s/tau).masked_fill(~valid,-torch.inf); count=valid.sum((1,2,3)).clamp_min(1)
    return tau*(torch.logsumexp(z.reshape(b,-1),1)-torch.log(count.float()))


def _fit(c,bags,indices,objective,jitter,seed):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    Model=_model_class(); model=Model(c.support_px,objective,c.init_jitter_std if jitter else 0).cuda(); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-5)
    tensors=[torch.from_numpy(x).cuda() for x in bags]; pc,nc,pa,na,m=tensors; ix=torch.tensor(indices,device="cuda")
    history=[]
    for epoch in range(1,c.epochs+1):
        opt.zero_grad(set_to_none=True); ps=_bag_score(model,pc[ix],pa[ix],m[ix],c.tolerance_px); qs=_bag_score(model,nc[ix],na[ix],m[ix],c.tolerance_px)
        rank=F.softplus(1-ps+qs).mean(); loss=rank+model.penalty(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        if epoch in (1,c.epochs): history.append({"epoch":epoch,"rank_loss":float(rank.item()),"total_loss":float(loss.item()),"positive_score":float(ps.mean().item()),"quiet_score":float(qs.mean().item())})
    del tensors; torch.cuda.empty_cache(); return model,history


def _score_map(model,fc,fa,batch):
    import torch
    acc=None; n=0; i=0; resolved=batch; model.eval()
    with torch.inference_mode():
        while i<len(fc):
            try:
                xc=torch.from_numpy(np.ascontiguousarray(fc[i:i+resolved,None])).cuda(); xa=torch.from_numpy(np.ascontiguousarray(fa[i:i+resolved,None])).cuda()
                s=model(xc,xa); part=torch.logsumexp(s/.25,0).squeeze(0); acc=part if acc is None else torch.logaddexp(acc,part); n+=len(xc); i+=len(xc)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); resolved//=2
                if resolved<1: raise
    return (.25*(acc-math.log(n))).cpu().numpy(),resolved


def _calibrate(model,lane,c):
    vals=[]; batch=c.frame_batch
    qc,qa=lane["quiet_contrast"],lane["quiet_amplitude"]
    for d,s in zip((24,24,28,47),(0,24,48,53)):
        score,batch=_score_map(model,qc[s:s+d],qa[s:s+d],batch); vals.extend(v1._peaks(score,c.nms_distance_px,limit=2000))
    ranked=sorted((x[0] for x in vals),reverse=True); return float(np.nextafter(ranked[4],np.inf)),batch


def _direct_map(frames):
    from scipy.special import logsumexp
    return (.25*(logsumexp(frames/.25,axis=0)-math.log(len(frames)))).astype(np.float32)


def _direct_baseline(lane,labels,c):
    vals=[]; qa=lane["quiet_amplitude"]
    for d,s in zip((24,24,28,47),(0,24,48,53)): vals.extend(v1._peaks(_direct_map(qa[s:s+d]),c.nms_distance_px,limit=2000))
    threshold=float(np.nextafter(sorted((x[0] for x in vals),reverse=True)[4],np.inf)); folds=[]
    for b in range(1,5):
        score=_direct_map(lane["bursts_amplitude"][b]); peaks=v1._peaks(score,c.nms_distance_px,threshold,limit=500); rows=[r for r in labels if r["burst_id"]==b]; matches=v1._match(peaks,rows,c.nms_distance_px)
        folds.append({"heldout_burst":b,"recall":len(matches)/len(rows),"matched":len(matches),"labels":len(rows),"event_peaks":len(peaks),"threshold":threshold})
    return {"mean_recall":float(np.mean([x["recall"] for x in folds])),"outer_folds":folds}


def run(c:DiagnosticConfig) -> dict[str,Any]:
    import torch
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[name]=str(c.cpu_threads)
    torch.backends.cudnn.allow_tf32=True; torch.backends.cuda.matmul.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(min(.95,c.max_gpu_memory_mib/(torch.cuda.get_device_properties(0).total_memory/2**20)))
    pre=preflight(c); c.output_dir.mkdir(parents=True,exist_ok=False); v1.atomic_json(c.output_dir/"preflight.json",pre)
    v1.atomic_json(c.output_dir/"resolved_config.json",{k:(str(v) if isinstance(v,Path) else list(v) if isinstance(v,tuple) else v) for k,v in c.__dict__.items()})
    state={"status":"running","started_at":v1.utc_now(),"pid":os.getpid()}; v1.atomic_json(c.output_dir/"run_state.json",state)
    progress=c.output_dir/"progress.jsonl"
    def log(stage,**kw):
        with progress.open("a") as f:f.write(json.dumps({"time":v1.utc_now(),"stage":stage,**kw})+"\n")
    labels=v1.load_labels(c.labels_tsv); lanes=prepare_lanes(c,labels); log("preprocessing_complete",summaries={k:v["summary"] for k,v in lanes.items()},rss_mib=v1.rss_mib())
    baselines={k:_direct_baseline(v,labels,c) for k,v in lanes.items()}; results=[]; radius=c.support_px//2+c.tolerance_px
    for combo in factor_matrix():
        lane=lanes[combo["input"]]; bags=_paired_bags(lane,labels,radius); seeds=[c.fixed_seed] if combo["initialization"]=="fixed_guarded" else list(c.jitter_seeds)
        for held in range(1,5):
            indices=[i for i,r in enumerate(labels) if r["burst_id"]!=held]
            for seed in seeds:
                model,history=_fit(c,bags,indices,combo["objective"],combo["initialization"]=="jittered_guarded",seed+held*100)
                threshold,batch=_calibrate(model,lane,c); score,batch=_score_map(model,lane["bursts_contrast"][held],lane["bursts_amplitude"][held],batch)
                peaks=v1._peaks(score,c.nms_distance_px,threshold,limit=500); rows=[r for r in labels if r["burst_id"]==held]; matches=v1._match(peaks,rows,c.nms_distance_px)
                row={**combo,"heldout_burst":held,"seed":seed,"recall":len(matches)/len(rows),"matched":len(matches),"labels":len(rows),"event_peaks":len(peaks),"threshold":threshold,"history":history,"resolved_batch":batch}
                results.append(row); log("fit_complete",**{k:v for k,v in row.items() if k!="history"}); del model; torch.cuda.empty_cache()
    summaries=[]
    for combo in factor_matrix():
        rows=[r for r in results if r["combination_id"]==combo["combination_id"]]; fold_means={b:float(np.mean([r["recall"] for r in rows if r["heldout_burst"]==b])) for b in range(1,5)}
        baseline=baselines[combo["input"]]; wins=sum(fold_means[b]>baseline["outer_folds"][b-1]["recall"] for b in range(1,5))
        summaries.append({**combo,"mean_recall":float(np.mean(list(fold_means.values()))),"fold_mean_recall":fold_means,"fold_wins_vs_direct":wins,"direct_baseline_mean_recall":baseline["mean_recall"],"fit_count":len(rows)})
    winner=max(summaries,key=lambda x:(x["mean_recall"],x["fold_wins_vs_direct"])); advance=winner["mean_recall"]>winner["direct_baseline_mean_recall"] and winner["fold_wins_vs_direct"]>=3
    metrics={"schema_version":1,"completed_at":v1.utc_now(),"planned_combinations":8,"learned_fit_count":len(results),"baselines":baselines,"combination_summaries":summaries,"fit_results":results,"winner":winner,
             "gate":{"decision":"advance_to_masked_and_final" if advance else "do_not_advance","reason":"requires mean recall above matched direct baseline and wins in at least 3/4 bursts"},
             "masked_stage":{"status":"not_run","reason":"held-out gate must pass first"},"final_stage":{"status":"not_run","reason":"held-out and masked gates must pass first"}}
    v1.atomic_json(c.output_dir/"metrics.json",metrics); _write_tables(c.output_dir,results,summaries); _write_report(c.output_dir/"report.md",c,metrics,lanes)
    state={"status":"completed","completed_at":v1.utc_now(),"pid":os.getpid(),"gate":metrics["gate"]["decision"],"learned_fit_count":len(results)}; v1.atomic_json(c.output_dir/"run_state.json",state); log("completed",**state)
    return metrics


def _write_tables(out,results,summaries):
    with (out/"combination_summary.tsv").open("w",newline="") as f:
        fields=["combination_id","input","objective","initialization","mean_recall","fold_wins_vs_direct","direct_baseline_mean_recall","fit_count"]
        w=csv.DictWriter(f,fields,delimiter="\t",extrasaction="ignore");w.writeheader();w.writerows(summaries)
    with (out/"fit_results.tsv").open("w",newline="") as f:
        fields=["combination_id","input","objective","initialization","heldout_burst","seed","recall","matched","labels","event_peaks","threshold","resolved_batch"]
        w=csv.DictWriter(f,fields,delimiter="\t",extrasaction="ignore");w.writeheader();w.writerows(results)


def _write_report(path,c,m,lanes):
    w=m["winner"]; lines=[f"# {c.experiment_id}","",f"Status: completed. Gate: `{m['gate']['decision']}`.","","## Matrix","",f"Eight factor combinations, four held-out bursts, `{m['learned_fit_count']}` learned fits. Jittered cells use three actual initialization seeds.","","## Winner","",f"- Combination: `{w['combination_id']}`",f"- Mean held-out recall: `{w['mean_recall']:.4f}`",f"- Matched direct baseline: `{w['direct_baseline_mean_recall']:.4f}`",f"- Fold wins: `{w['fold_wins_vs_direct']}/4`","","Detailed eight-cell results: `combination_summary.tsv`. Full fit histories: `metrics.json`.","","## Preprocessing","",json.dumps({k:v['summary'] for k,v in lanes.items()},indent=2),"","The masked and final stages are intentionally gated. Unlabeled event pixels remain unknown, not negatives."]
    path.write_text("\n".join(lines)+"\n")
