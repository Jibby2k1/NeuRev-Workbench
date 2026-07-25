"""Guarded weakly-supervised learnable-contrast experiment for Spon Ca Burst."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Config:
    experiment_id: str
    source_video: Path
    source_workbook: Path
    labels_tsv: Path
    label_summary: Path
    design_document: Path
    output_dir: Path
    quiet_start_ui: int
    quiet_end_ui: int
    scored_start_ui: int
    scored_end_ui: int
    support_px: int
    tolerance_px: int
    nms_distance_px: int
    epochs: int
    masked_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    device: str
    cpu_threads: int
    frame_batch: int
    max_ram_mib: int
    max_gpu_memory_mib: int
    min_free_disk_mib: int
    max_output_mib: int

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        def p(name: str) -> Path:
            return (root / raw[name]).resolve()
        resources = raw["resources"]
        training = raw["training"]
        frames = raw["frames"]
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=p("source_video"), source_workbook=p("source_workbook"),
            labels_tsv=p("labels_tsv"), label_summary=p("label_summary"),
            design_document=p("design_document"), output_dir=p("output_dir"),
            quiet_start_ui=int(frames["quiet_start_ui"]), quiet_end_ui=int(frames["quiet_end_ui"]),
            scored_start_ui=int(frames["scored_start_ui"]), scored_end_ui=int(frames["scored_end_ui"]),
            support_px=int(raw.get("support_px", 21)), tolerance_px=int(raw.get("tolerance_px", 4)),
            nms_distance_px=int(raw.get("nms_distance_px", 6)), epochs=int(training["epochs"]),
            masked_seeds=tuple(int(x) for x in training["masked_seeds"]),
            final_seeds=tuple(int(x) for x in training["final_seeds"]),
            device=str(resources["device"]), cpu_threads=int(resources["cpu_threads"]),
            frame_batch=int(resources["frame_batch"]), max_ram_mib=int(resources["max_ram_mib"]),
            max_gpu_memory_mib=int(resources["max_gpu_memory_mib"]),
            min_free_disk_mib=int(resources["min_free_disk_mib"]),
            max_output_mib=int(resources["max_output_mib"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device != "cuda": raise ValueError("This manifest intentionally requires CUDA.")
        if self.support_px < 3 or self.support_px % 2 != 1: raise ValueError("support_px must be odd and >=3")
        if not 1 <= self.cpu_threads <= 24: raise ValueError("cpu_threads must be in [1,24]")
        if not 1 <= self.frame_batch <= 128: raise ValueError("frame_batch must be in [1,128]")
        if len(self.masked_seeds) < 10: raise ValueError("At least 10 masked-discovery seeds are required")
        if len(self.final_seeds) < 5: raise ValueError("At least 5 final ensemble seeds are required")
        if self.output_dir.exists(): raise FileExistsError(f"Output directory already exists: {self.output_dir}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def available_ram_mib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20)

def rss_mib() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_labels(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for row in rows:
        out.append({
            **row, "burst_id": int(row["burst_id"]), "start_frame_ui": int(row["start_frame_ui"]),
            "end_frame_ui": int(row["end_frame_ui"]), "start_frame_zero": int(row["start_frame_zero"]),
            "stop_frame_zero_exclusive": int(row["stop_frame_zero_exclusive"]),
            "x_px": float(row["x_px"]), "y_px": float(row["y_px"]),
            "recurrence_count": int(row["recurrence_count"]),
        })
    return out


def preflight(config: Config, *, artifact_dir: Path | None = None) -> dict[str, Any]:
    import torch
    files = [config.source_video, config.source_workbook, config.labels_tsv, config.label_summary, config.design_document]
    missing = [str(p) for p in files if not p.is_file()]
    if missing: raise FileNotFoundError("Missing experiment inputs: " + ", ".join(missing))
    video = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
    if video.ndim != 3: raise ValueError(f"Expected grayscale frame stack, got {video.shape}")
    labels = load_labels(config.labels_tsv)
    summary = json.loads(config.label_summary.read_text(encoding="utf-8"))
    burst_counts = {b: sum(r["burst_id"] == b for r in labels) for b in range(1, 5)}
    if len(labels) != int(summary["total_point_window_labels"]): raise ValueError("TSV/summary label count mismatch")
    if len({r["roi_identity"] for r in labels}) != int(summary["unique_roi_coordinates"]): raise ValueError("TSV/summary ROI count mismatch")
    h, w = map(int, video.shape[1:])
    bad = [r for r in labels if not (0 <= r["x_px"] < w and 0 <= r["y_px"] < h)]
    if bad: raise ValueError(f"{len(bad)} label coordinates fall outside the video")
    if config.scored_end_ui > len(video) or config.quiet_start_ui < 1: raise ValueError("Frame contract exceeds source")
    ram_available = available_ram_mib(); disk_probe = config.output_dir.parent
    while not disk_probe.exists(): disk_probe = disk_probe.parent
    disk = shutil.disk_usage(disk_probe)
    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        prop = torch.cuda.get_device_properties(0)
        gpu.update(name=prop.name, total_mib=total // 2**20, free_mib=free // 2**20,
                   compute_capability=f"{prop.major}.{prop.minor}")
    ready = bool(torch.cuda.is_available() and ram_available >= config.max_ram_mib
                 and disk.free // 2**20 >= config.min_free_disk_mib
                 and gpu.get("free_mib", 0) >= config.max_gpu_memory_mib)
    payload = {
        "schema_version": 1, "ready": ready, "generated_at": utc_now(),
        "experiment_id": config.experiment_id,
        "input_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
        "video": {"shape": list(video.shape), "dtype": str(video.dtype), "memory_mapped": True},
        "labels": {"rows": len(labels), "unique_rois": len({r['roi_identity'] for r in labels}),
                   "burst_counts": burst_counts, "coordinate_contract": "x=column,y=row,native pixels",
                   "all_coordinates_in_bounds": not bad},
        "resources": {"cpu_logical": os.cpu_count(), "cpu_threads": config.cpu_threads,
                      "ram_available_mib": ram_available, "ram_cap_mib": config.max_ram_mib,
                      "disk_free_mib": disk.free // 2**20, "gpu": gpu,
                      "gpu_cap_mib": config.max_gpu_memory_mib, "frame_batch": config.frame_batch},
    }
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_overlay(video, labels, artifact_dir / "label_projection_overlay.png")
        payload["label_projection_overlay"] = str((artifact_dir / "label_projection_overlay.png").resolve())
        atomic_json(artifact_dir / "preflight.json", payload)
    if not ready: raise RuntimeError("Preflight resource or CUDA readiness check failed")
    return payload


def _write_overlay(video: np.ndarray, labels: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = np.asarray(video[1799:1899], dtype=np.float32).mean(axis=0)
    lo, hi = np.percentile(q, [1, 99.8])
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.imshow(q, cmap="gray", vmin=lo, vmax=hi)
    seen = {}
    for r in labels: seen[r["roi_identity"]] = (r["x_px"], r["y_px"], r["recurrence_count"])
    xs=[v[0] for v in seen.values()]; ys=[v[1] for v in seen.values()]; cs=[v[2] for v in seen.values()]
    sc=ax.scatter(xs,ys,c=cs,cmap="viridis",vmin=1,vmax=4,s=45,facecolors="none",linewidths=1.4)
    fig.colorbar(sc,ax=ax,label="burst recurrence")
    ax.set(title="All 27 normalized ROI centers on quiet projection (x=column, y=row)", xlabel="x / column", ylabel="y / row")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _configure(config: Config) -> None:
    value = str(config.cpu_threads)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = value


def _prepare_arrays(config: Config, labels: list[dict[str, Any]]) -> tuple[np.ndarray, dict[int,np.ndarray], np.ndarray]:
    video=np.load(config.source_video,mmap_mode="r",allow_pickle=False)
    q0,q1=config.quiet_start_ui-1,config.quiet_end_ui
    quiet_raw=np.asarray(video[q0:q1],dtype=np.float32)
    baseline=np.median(quiet_raw,axis=0)
    lo,hi=np.percentile(quiet_raw[:,::4,::4],[1.0,99.9]); scale=max(float(hi-lo),1.0)
    quiet=np.maximum((quiet_raw-baseline)/scale,0).astype(np.float32)
    bursts={}
    for b in range(1,5):
        rows=[r for r in labels if r["burst_id"]==b]; s=rows[0]["start_frame_zero"]; e=rows[0]["stop_frame_zero_exclusive"]
        bursts[b]=np.maximum((np.asarray(video[s:e],dtype=np.float32)-baseline)/scale,0).astype(np.float32)
    return quiet,bursts,baseline


def _patches(frames: np.ndarray, rows: list[dict[str,Any]], radius: int) -> tuple[np.ndarray,np.ndarray]:
    size=2*radius+1; max_t=max(len(frames) for _ in rows)
    data=np.zeros((len(rows),max_t,1,size,size),dtype=np.float32); mask=np.zeros((len(rows),max_t),dtype=bool)
    padded=np.pad(frames,((0,0),(radius,radius),(radius,radius)))
    for i,r in enumerate(rows):
        x=int(round(r["x_px"])); y=int(round(r["y_px"])); n=len(frames)
        data[i,:n,0]=padded[:,y:y+size,x:x+size]; mask[i,:n]=True
    return data,mask


def _bag_tensors(quiet: np.ndarray, bursts: dict[int,np.ndarray], labels: list[dict[str,Any]], radius: int):
    pos=[]; neg=[]; masks=[]
    max_t=max(len(v) for v in bursts.values()); size=2*radius+1
    for r in labels:
        b=r["burst_id"]; event=bursts[b]; duration=len(event)
        pd,pm=_patches(event,[r],radius)
        start=((b-1)*23) % (len(quiet)-duration+1)
        qd,qm=_patches(quiet[start:start+duration],[r],radius)
        pp=np.zeros((max_t,1,size,size),np.float32); qq=pp.copy(); mm=np.zeros(max_t,bool)
        pp[:duration]=pd[0]; qq[:duration]=qd[0]; mm[:duration]=True
        pos.append(pp); neg.append(qq); masks.append(mm)
    return np.stack(pos),np.stack(neg),np.stack(masks)


def _model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Model(nn.Module):
        def __init__(self,support:int,amplitude:bool):
            super().__init__(); self.support=support; self.amplitude=amplitude
            r=support//2; yy,xx=torch.meshgrid(torch.arange(-r,r+1),torch.arange(-r,r+1),indexing="ij")
            rr=torch.sqrt(xx.float()**2+yy.float()**2)
            kt=torch.exp(-0.5*(rr/2.0)**2); kr=((rr>=4)&(rr<=10)).float()+0.02
            self.raw_t=nn.Parameter(torch.log(torch.expm1(kt/kt.max()*2+0.02)))
            self.raw_r=nn.Parameter(torch.log(torch.expm1(kr/kr.max()*2+0.02)))
            self.raw_lambda=nn.Parameter(torch.tensor(-1.5))
            self.register_buffer("rr2",rr**2)
        def kernels(self):
            kt=F.softplus(self.raw_t); kr=F.softplus(self.raw_r); return kt/kt.sum(),kr/kr.sum()
        def forward(self,x):
            kt,kr=self.kernels(); pad=self.support//2
            xp=F.pad(x,(pad,pad,pad,pad),mode="reflect")
            t=F.conv2d(xp,kt[None,None]); mu=F.conv2d(xp,kr[None,None])
            v=(F.conv2d(xp*xp,kr[None,None])-mu*mu).clamp_min(0)
            b=(1-(kr*kr).sum()).clamp_min(1e-4); q=((kt-kr)**2).sum().clamp_min(1e-6)
            z=(t-mu)/torch.sqrt(q*v/b+1e-6); score=F.relu(z).square()
            if self.amplitude: score=score+F.softplus(self.raw_lambda)*torch.log1p(t.clamp_min(0))
            return score
        def penalty(self):
            kt,kr=self.kernels(); eps=1e-8
            yy,xx=torch.meshgrid(torch.arange(self.support,device=kt.device),torch.arange(self.support,device=kt.device),indexing="ij")
            ct=torch.stack(((kt*yy).sum(),(kt*xx).sum())); cr=torch.stack(((kr*yy).sum(),(kr*xx).sum()))
            et=(kt*self.rr2).sum(); er=(kr*self.rr2).sum(); ess=1/(kr.square().sum()+eps)
            tv=(kt[:,1:]-kt[:,:-1]).abs().mean()+(kt[1:]-kt[:-1]).abs().mean()+(kr[:,1:]-kr[:,:-1]).abs().mean()+(kr[1:]-kr[:-1]).abs().mean()
            return 2*(kt*kr).sum()+0.05*(ct-cr).square().sum()+0.02*F.relu(et+8-er).square()+0.02*F.relu(40-ess).square()+0.002*tv
    return Model


def _bag_scores(model,x,mask,tolerance):
    import torch
    b,t,_,h,w=x.shape; s=model(x.reshape(b*t,1,h,w)).reshape(b,t,h,w)
    c=h//2; disk=torch.zeros((h,w),dtype=torch.bool,device=x.device)
    yy,xx=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing="ij")
    disk=((yy-c)**2+(xx-c)**2)<=tolerance**2
    valid=mask[:,:,None,None]&disk[None,None]
    tau=0.25; z=(s/tau).masked_fill(~valid,-torch.inf); count=valid.sum((1,2,3)).clamp_min(1)
    return tau*(torch.logsumexp(z.reshape(b,-1),dim=1)-torch.log(count.float()))


def _fit(config:Config,pos,neg,masks,row_indices:list[int],amplitude:bool,seed:int,epochs:int,val_indices:list[int]|None=None):
    import torch
    import torch.nn.functional as F
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    Model=_model_class(); model=Model(config.support_px,amplitude).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-5)
    p=torch.from_numpy(pos).cuda(); q=torch.from_numpy(neg).cuda(); m=torch.from_numpy(masks).cuda()
    train=torch.tensor(row_indices,device="cuda"); val=None if not val_indices else torch.tensor(val_indices,device="cuda")
    best=(float("inf"),0,None); patience=0
    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        ps=_bag_scores(model,p[train],m[train],config.tolerance_px); qs=_bag_scores(model,q[train],m[train],config.tolerance_px)
        loss=F.softplus(1.0-ps+qs).mean()+model.penalty(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        if epoch%5==0 or epoch==epochs:
            model.eval()
            with torch.no_grad():
                ix=train if val is None else val
                metric=F.softplus(1.0-_bag_scores(model,p[ix],m[ix],config.tolerance_px)+_bag_scores(model,q[ix],m[ix],config.tolerance_px)).mean().item()+model.penalty().item()
            if metric<best[0]-1e-5: best=(metric,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); patience=0
            else: patience+=5
            if val is not None and patience>=30: break
    model.load_state_dict(best[2]); del p,q,m; torch.cuda.empty_cache()
    return model,best[1],best[0]


def _score_map(model,frames:np.ndarray,batch:int) -> tuple[np.ndarray,int]:
    import torch
    model.eval(); accumulator=None; count=0; i=0; resolved=batch
    with torch.inference_mode():
        while i<len(frames):
            try:
                x=torch.from_numpy(np.ascontiguousarray(frames[i:i+resolved,None])).cuda(non_blocking=True)
                s=model(x); chunk=torch.logsumexp(s/0.25,dim=0).squeeze(0)
                accumulator=chunk if accumulator is None else torch.logaddexp(accumulator,chunk)
                count+=len(x); i+=len(x); del x,s,chunk
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); resolved//=2
                if resolved<1: raise
    result=(0.25*(accumulator-math.log(count))).float().cpu().numpy(); return result,resolved


def _peaks(score:np.ndarray,distance:int,threshold:float=-np.inf,limit:int=10000):
    from scipy.ndimage import maximum_filter
    size=2*distance+1; keep=(score==maximum_filter(score,size=size,mode="nearest"))&(score>=threshold)
    keep[:distance]=False; keep[-distance:]=False; keep[:,:distance]=False; keep[:,-distance:]=False
    y,x=np.nonzero(keep); vals=score[y,x]; order=np.argsort(vals)[::-1][:limit]
    return [(float(vals[i]),int(x[i]),int(y[i])) for i in order]


def _calibrate(model,quiet,config):
    durations=[24,24,28,47]; starts=[0,24,48,53]; allp=[]; batch=config.frame_batch
    for d,s in zip(durations,starts):
        score,batch=_score_map(model,quiet[s:s+d],batch); allp.extend(_peaks(score,config.nms_distance_px,limit=2000))
    values=sorted((p[0] for p in allp),reverse=True); threshold=values[4] if len(values)>4 else values[-1]
    return float(np.nextafter(threshold,np.inf)),batch


def _match(peaks,rows,radius):
    remaining=set(range(len(rows))); matches=[]
    for score,x,y in peaks:
        choices=[(math.hypot(x-rows[i]["x_px"],y-rows[i]["y_px"]),i) for i in remaining]
        if not choices: break
        d,i=min(choices)
        if d<=radius: remaining.remove(i); matches.append((i,score,x,y,d))
    return matches


def _kernel_payload(model):
    import torch
    with torch.no_grad(): kt,kr=model.kernels()
    return {"test_kernel":kt.cpu().tolist(),"reference_kernel":kr.cpu().tolist(),
            "amplitude_lambda":float(torch.nn.functional.softplus(model.raw_lambda).item()) if model.amplitude else 0.0}


def _save_kernel_image(model,path,title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=_kernel_payload(model); fig,axs=plt.subplots(1,2,figsize=(7,3),dpi=130)
    for ax,key,name in zip(axs,("test_kernel","reference_kernel"),("Test kernel","Reference kernel")):
        im=ax.imshow(p[key],cmap="magma"); ax.set_title(name); fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _rows_for_bursts(labels,bursts:set[int]): return [i for i,r in enumerate(labels) if r["burst_id"] in bursts]


def run(config:Config) -> dict[str,Any]:
    _configure(config)
    import torch
    torch.backends.cudnn.allow_tf32=True; torch.backends.cuda.matmul.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(min(0.95,config.max_gpu_memory_mib/(torch.cuda.get_device_properties(0).total_memory/2**20)))
    config.output_dir.mkdir(parents=True,exist_ok=False)
    pre=preflight_existing(config,config.output_dir/"preflight")
    atomic_json(config.output_dir/"resolved_config.json",_config_payload(config))
    provenance=config.output_dir/"provenance"; provenance.mkdir()
    for src in (config.source_workbook,config.labels_tsv,config.label_summary,config.design_document): shutil.copy2(src,provenance/src.name)
    progress=config.output_dir/"progress.jsonl"
    def log(stage,**extra):
        with progress.open("a",encoding="utf-8") as f: f.write(json.dumps({"time":utc_now(),"stage":stage,**extra})+"\n")
    atomic_json(config.output_dir/"run_state.json",{"status":"running","started_at":utc_now(),"pid":os.getpid()}); log("start")
    labels=load_labels(config.labels_tsv); quiet,bursts,baseline=_prepare_arrays(config,labels); log("arrays_ready",ram_rss_mib=rss_mib())
    radius=config.support_px//2+config.tolerance_px
    pos,neg,masks=_bag_tensors(quiet,bursts,labels,radius)
    fold_rows=[]; fold_models={}
    for arm in ("contrast","contrast_amplitude"):
        amplitude=arm.endswith("amplitude")
        for held in range(1,5):
            train_b={1,2,3,4}-{held}; inner=[]
            for val_b in sorted(train_b):
                model,ep,metric=_fit(config,pos,neg,masks,_rows_for_bursts(labels,train_b-{val_b}),amplitude,1000+held*10+val_b,config.epochs,_rows_for_bursts(labels,{val_b}))
                inner.append({"validation_burst":val_b,"selected_epoch":ep,"ranking_loss":metric}); del model
            selected=max(10,int(np.median([x["selected_epoch"] for x in inner])))
            model,_,_=_fit(config,pos,neg,masks,_rows_for_bursts(labels,train_b),amplitude,2000+held,selected,None)
            threshold,resolved=_calibrate(model,quiet,config); score,resolved=_score_map(model,bursts[held],resolved)
            ps=_peaks(score,config.nms_distance_px,threshold); held_rows=[r for r in labels if r["burst_id"]==held]
            matches=_match(ps,held_rows,config.nms_distance_px)
            row={"arm":arm,"heldout_burst":held,"recall":len(matches)/len(held_rows),"matched":len(matches),"labels":len(held_rows),
                 "quiet_threshold":threshold,"event_peaks":len(ps),"selected_epoch":selected,"inner_folds":inner,"resolved_frame_batch":resolved}
            fold_rows.append(row); fold_models[f"{arm}_heldout_b{held}"]=_kernel_payload(model)
            _save_kernel_image(model,config.output_dir/f"kernel_{arm}_heldout_b{held}.png",f"{arm}, held-out burst {held}")
            log("outer_fold",**row); del model; torch.cuda.empty_cache()
    arm_means={arm:float(np.mean([r["recall"] for r in fold_rows if r["arm"]==arm])) for arm in ("contrast","contrast_amplitude")}
    best_arm=max(arm_means,key=arm_means.get); amplitude=best_arm.endswith("amplitude")
    masked=[]; identities=sorted({r["roi_identity"] for r in labels})
    for seed in config.masked_seeds:
        rng=random.Random(seed); strat={n:[x for x in identities if next(r for r in labels if r["roi_identity"]==x)["recurrence_count"]==n] for n in range(1,5)}
        hidden=set()
        for group in strat.values():
            rng.shuffle(group); hidden.update(group[:max(1,round(len(group)*.25))])
        train_ix=[i for i,r in enumerate(labels) if r["roi_identity"] not in hidden]
        model,_,_=_fit(config,pos,neg,masks,train_ix,amplitude,3000+seed,max(20,int(config.epochs*.75)),None)
        candidates=[]
        for b in range(1,5):
            score,_=_score_map(model,bursts[b],config.frame_batch); candidates.extend([(s,x,y,b) for s,x,y in _peaks(score,config.nms_distance_px,limit=50)])
        candidates.sort(reverse=True); hidden_rows=[]
        for identity in hidden:
            base=next(r for r in labels if r["roi_identity"]==identity); hidden_rows.append(base)
        vals={}
        for k in (10,20,50): vals[f"recall_at_{k}"]=len(_match([(s,x,y) for s,x,y,_ in candidates[:k]],hidden_rows,config.nms_distance_px))/len(hidden_rows)
        item={"seed":seed,"hidden_identities":sorted(hidden),**vals}; masked.append(item); log("masked_seed",**item); del model; torch.cuda.empty_cache()
    final_models=[]; ensemble_maps={b:np.zeros_like(bursts[b][0],dtype=np.float64) for b in range(1,5)}; thresholds=[]
    for seed in config.final_seeds:
        model,ep,_=_fit(config,pos,neg,masks,list(range(len(labels))),amplitude,4000+seed,config.epochs,None)
        threshold,_=_calibrate(model,quiet,config); thresholds.append(threshold)
        for b in range(1,5):
            score,_=_score_map(model,bursts[b],config.frame_batch); ensemble_maps[b]+=score/len(config.final_seeds)
        payload={"seed":seed,"epoch":ep,"threshold":threshold,"kernels":_kernel_payload(model)}; final_models.append(payload)
        torch.save({"state_dict":model.state_dict(),"config":_config_payload(config),"seed":seed},config.output_dir/f"final_model_seed_{seed}.pt")
        _save_kernel_image(model,config.output_dir/f"final_kernel_seed_{seed}.png",f"Final {best_arm}, seed {seed}")
        log("final_seed",seed=seed,threshold=threshold); del model; torch.cuda.empty_cache()
    threshold=float(np.mean(thresholds)); known=[]; novel=[]
    for b,score in ensemble_maps.items():
        rows=[r for r in labels if r["burst_id"]==b]; ps=_peaks(score,config.nms_distance_px,threshold,limit=500); matched=_match(ps,rows,config.nms_distance_px); mi={j for j,*_ in matched}
        matched_peak={(x,y) for _,_,x,y,_ in matched}
        for j,s,x,y,d in matched: known.append({"burst_id":b,"roi_identity":rows[j]["roi_identity"],"x_px":x,"y_px":y,"score":s,"distance_px":d})
        for s,x,y in ps:
            if (x,y) not in matched_peak: novel.append({"burst_id":b,"x_px":x,"y_px":y,"score":s})
    _write_tsv(config.output_dir/"known_recovery.tsv",known)
    _write_tsv(config.output_dir/"novel_candidate_rois.tsv",sorted(novel,key=lambda r:r["score"],reverse=True))
    with (config.output_dir/"froc.tsv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(fold_rows[0]),delimiter="\t",extrasaction="ignore"); w.writeheader(); w.writerows(fold_rows)
    baselines=evaluate_matched_baselines(config,quiet,bursts,labels,masked)
    learned_folds=[r for r in fold_rows if r["arm"]==best_arm]
    best_baseline_name=max(baselines,key=lambda k: baselines[k]["mean_recall"])
    best_baseline=baselines[best_baseline_name]
    fold_wins=sum(r["recall"]>best_baseline["outer_folds"][r["heldout_burst"]-1]["recall"] for r in learned_folds)
    go_no_go={"decision":"advance" if arm_means[best_arm]>best_baseline["mean_recall"] and fold_wins>=3 and float(np.mean([x["recall_at_20"] for x in masked]))>best_baseline["masked_mean_recall_at_20"] else "do_not_advance", "best_baseline":best_baseline_name,"learned_mean_recall":arm_means[best_arm],"baseline_mean_recall":best_baseline["mean_recall"],"fold_wins":fold_wins,"learned_masked_recall_at_20":float(np.mean([x["recall_at_20"] for x in masked])),"baseline_masked_recall_at_20":best_baseline["masked_mean_recall_at_20"]}
    metrics={"schema_version":1,"completed_at":utc_now(),"outer_folds":fold_rows,"mean_recall_by_arm":arm_means,"baselines":baselines,"go_no_go":go_no_go,"selected_arm":best_arm,
             "masked_discovery":masked,"masked_mean_recall_at_20":float(np.mean([x["recall_at_20"] for x in masked])),
             "final_ensemble":{"seeds":list(config.final_seeds),"threshold":threshold,"known_matches":len(known),"novel_peaks":len(novel)},
             "interpretation":"Unmatched peaks are candidates, not false positives; manual review is required."}
    atomic_json(config.output_dir/"metrics.json",metrics); write_auxiliary_artifacts(config,labels,masked,novel,metrics); atomic_json(config.output_dir/"model_details.json",{"outer_fold_kernels":fold_models,"final_models":final_models})
    _write_report(config.output_dir/"report.md",config,metrics,pre)
    size=sum(p.stat().st_size for p in config.output_dir.rglob("*") if p.is_file())
    if size>config.max_output_mib*2**20: raise RuntimeError("Output cap exceeded")
    state={"status":"completed","completed_at":utc_now(),"output_bytes":size,"selected_arm":best_arm,"pid":os.getpid()}; atomic_json(config.output_dir/"run_state.json",state); log("completed",**state)
    return metrics


def preflight_existing(config:Config,artifact_dir:Path):
    # Config normally forbids existing main output; the runner creates it immediately before this call.
    object.__setattr__(config,"output_dir",config.output_dir.parent/(config.output_dir.name+".__preflight_nonexistent__"))
    try: return preflight(config,artifact_dir=artifact_dir)
    finally: object.__setattr__(config,"output_dir",artifact_dir.parent)


def _config_payload(c:Config):
    return {k:(str(v) if isinstance(v,Path) else list(v) if isinstance(v,tuple) else v) for k,v in c.__dict__.items()}


def _write_tsv(path:Path,rows:list[dict[str,Any]]):
    fields=list(rows[0]) if rows else ["burst_id","x_px","y_px","score"]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(rows)


def _write_report(path:Path,c:Config,m:dict,p:dict):
    lines=[f"# {c.experiment_id}","",f"Status: completed at {m['completed_at']}.","","## Result",
           "",f"Selected arm: `{m['selected_arm']}`.",f"Mean LOBO recall: `{m['mean_recall_by_arm'][m['selected_arm']]:.3f}`.",
           f"Masked-ROI mean Recall@20: `{m['masked_mean_recall_at_20']:.3f}`.",
           f"Final known matches: `{m['final_ensemble']['known_matches']}`; unmatched candidate peaks: `{m['final_ensemble']['novel_peaks']}`.",f"Go/no-go: `{m.get('go_no_go',{}).get('decision','not_evaluated')}`.",
           "","Unmatched candidates are not false positives or scientific positives. They require workbench review.","",
           "## Resource contract","",f"CUDA GPU: {p['resources']['gpu'].get('name')}; frame batch {c.frame_batch}; GPU cap {c.max_gpu_memory_mib} MiB; RAM cap {c.max_ram_mib} MiB; CPU threads {c.cpu_threads}.","",
           "## Provenance","","All four supplied artifacts were hashed in `preflight/preflight.json` and copied under `provenance/`. The label overlay verifies x=column and y=row."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def evaluate_matched_baselines(config:Config,quiet:np.ndarray,bursts:dict[int,np.ndarray],labels:list[dict[str,Any]],masked:list[dict[str,Any]]) -> dict[str,Any]:
    """Evaluate direct residual and fixed guarded initialization under identical NMS/calibration."""
    from scipy.special import logsumexp
    import torch
    def direct_map(frames):
        return (0.25*(logsumexp(frames/0.25,axis=0)-math.log(len(frames)))).astype(np.float32)
    def direct_threshold():
        vals=[]
        for d,s in zip((24,24,28,47),(0,24,48,53)):
            vals.extend(_peaks(direct_map(quiet[s:s+d]),config.nms_distance_px,limit=2000))
        ranked=sorted((p[0] for p in vals),reverse=True)
        return float(np.nextafter(ranked[4],np.inf))
    Model=_model_class(); fixed=Model(config.support_px,False).cuda().eval()
    fixed_threshold,_=_calibrate(fixed,quiet,config); direct_thr=direct_threshold()
    result={}; candidate_cache={}
    for name,threshold in (("fixed_guarded_contrast",fixed_threshold),("direct_positive_residual",direct_thr)):
        folds=[]; candidates=[]
        for b in range(1,5):
            score=_score_map(fixed,bursts[b],config.frame_batch)[0] if name.startswith("fixed") else direct_map(bursts[b])
            peaks=_peaks(score,config.nms_distance_px,threshold,limit=500)
            rows=[r for r in labels if r["burst_id"]==b]; matches=_match(peaks,rows,config.nms_distance_px)
            folds.append({"heldout_burst":b,"recall":len(matches)/len(rows),"matched":len(matches),"labels":len(rows),"event_peaks":len(peaks),"threshold":threshold})
            candidates.extend([(s,x,y,b) for s,x,y in _peaks(score,config.nms_distance_px,limit=50)])
        candidates.sort(reverse=True); candidate_cache[name]=candidates
        masked_scores=[]
        for item in masked:
            rows=[next(r for r in labels if r["roi_identity"]==identity) for identity in item["hidden_identities"]]
            recall=len(_match([(s,x,y) for s,x,y,_ in candidates[:20]],rows,config.nms_distance_px))/len(rows)
            masked_scores.append(recall)
        result[name]={"outer_folds":folds,"mean_recall":float(np.mean([x["recall"] for x in folds])),
                      "masked_mean_recall_at_20":float(np.mean(masked_scores)),"masked_recall_at_20_by_seed":masked_scores}
    del fixed; torch.cuda.empty_cache(); return result


def write_auxiliary_artifacts(config:Config,labels:list[dict[str,Any]],masked:list[dict[str,Any]],novel:list[dict[str,Any]],metrics:dict[str,Any]) -> None:
    bursts={b:{"start_frame_ui":next(r["start_frame_ui"] for r in labels if r["burst_id"]==b),
               "end_frame_ui":next(r["end_frame_ui"] for r in labels if r["burst_id"]==b)} for b in range(1,5)}
    quiet_blocks=[]
    for i,(start,stop) in enumerate(((1800,1824),(1825,1849),(1850,1874),(1875,1899)),1):
        quiet_blocks.append({"block_id":i,"start_frame_ui":start,"end_frame_ui":stop,"semantics":"cross-fit calibration/null block"})
    splits={"schema_version":1,"frame_contract":"UI inclusive one-based; NumPy half-open zero-based",
            "outer_leave_one_burst_out":[{"heldout_burst":b,"training_bursts":[x for x in range(1,5) if x!=b],"inner_validation_rotation":[x for x in range(1,5) if x!=b]} for b in range(1,5)],
            "bursts":bursts,"quiet_blocks":quiet_blocks,
            "masked_roi_seeds":[{"seed":x["seed"],"hidden_identities":x["hidden_identities"]} for x in masked]}
    atomic_json(config.output_dir/"splits.json",splits)
    suggestions=[]
    threshold=metrics["final_ensemble"]["threshold"]
    for rank,row in enumerate(sorted(novel,key=lambda x:x["score"],reverse=True),1):
        suggestions.append({"id":f"LC{rank:03d}","centroidX":float(row["x_px"]),"centroidY":float(row["y_px"]),
                            "discoveryScore":float(row["score"]-threshold),"priorityScore":1.0/rank,
                            "burstId":int(row["burst_id"]),"rawScore":float(row["score"]),"threshold":threshold,
                            "provenance":config.experiment_id,"state":"unlabeled","interpretation":"unmatched candidate; manual review required"})
    review={"schema_version":1,"dataset":{"dataset_id":"spon_ca_burst_3_hindbrain_to_tail_488_20ms","modality":"calcium"},
            "video":{"name":"3 hindbrain to tail 488 20ms.tif","width":573,"height":340,"frames":2359,
                     "framePattern":"Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif"},
            "parameters":{"experiment_id":config.experiment_id,"operating_point":"<=1 quiet peak per pseudo-burst map"},
            "qc":{"go_no_go":metrics.get("go_no_go",{})},"discovery":{"evidenceMaps":[],"suggestions":suggestions},
            "rois":[],"assetBasePath":"."}
    atomic_json(config.output_dir/"workbench_discovery_review_data.json",review)
    atomic_json(config.output_dir/"experiment_summary.json",{"schema_version":1,"experiment_id":config.experiment_id,
                "status":"completed","selected_arm":metrics["selected_arm"],"mean_recall_by_arm":metrics["mean_recall_by_arm"],
                "masked_mean_recall_at_20":metrics["masked_mean_recall_at_20"],"go_no_go":metrics.get("go_no_go"),
                "known_matches":metrics["final_ensemble"]["known_matches"],"novel_candidates":len(suggestions),
                "report":"report.md","metrics":"metrics.json","review_data":"workbench_discovery_review_data.json"})
