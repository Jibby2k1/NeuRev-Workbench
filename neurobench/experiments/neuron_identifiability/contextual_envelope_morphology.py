"""Frozen visual-first audit of temporal contextual-envelope morphology."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurev-context-envelope-morphology-mpl")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from neurobench.experiments.neuron_identifiability.contextual_envelope_retrieval import (
    START_UI, FRAMES, calibrate, load_paths, load_source, trailing_max,
)
from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import make_event_and_quiet_masks

FEATURES = ("A", "A2", "U", "C", "P", "H")
SOURCES = ("RAW", "Z_CURRENT", "T_ENVELOPE")
LABELS = {"A":"instantaneous evidence", "A2":"amplitude squared", "U":"upper envelope",
          "C":"envelope agreement", "P":"envelope-gated evidence", "H":"agreement attenuation"}
COLORS = {"A":"#2457A6", "A2":"#A35D00", "U":"#6C6C6C", "C":"#8A4F9E", "P":"#C14A68", "H":"#587A2E"}
PRE, POST, BOOTSTRAPS, SEED = 20, 40, 2000, 20260831

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, payload: Any) -> None:
    q=path.with_suffix(path.suffix+".partial"); q.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); q.replace(path)

def write_tsv(path: Path, rows: list[dict[str,Any]]) -> None:
    q=path.with_suffix(path.suffix+".partial")
    with q.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t"); w.writeheader(); w.writerows(rows)
    q.replace(path)

def operators(a: np.ndarray, u: np.ndarray | None=None) -> dict[str,np.ndarray]:
    a=np.asarray(a,dtype=np.float32); u=trailing_max(a,5) if u is None else np.asarray(u,dtype=np.float32)
    c=np.divide(a,np.maximum(u,1e-6),out=np.zeros_like(a),where=u>0)
    return {"A":a,"A2":a*a,"U":u,"C":c,"P":a*u,"H":a*c}

def aligned(trace: np.ndarray, anchor: int) -> np.ndarray:
    out=np.full(PRE+POST+1,np.nan); lo=max(0,anchor-PRE); hi=min(len(trace),anchor+POST+1)
    out[lo-(anchor-PRE):hi-(anchor-PRE)]=trace[lo:hi]; return out

def halfmax_width(x: np.ndarray, anchor: int) -> int:
    peak=float(x[anchor]);
    if not np.isfinite(peak) or peak<=0: return 0
    keep=np.asarray(x>=peak/2); lo=anchor; hi=anchor
    while lo>0 and keep[lo-1]: lo-=1
    while hi+1<len(x) and keep[hi+1]: hi+=1
    return hi-lo+1

def valid_quiet_starts(mask: np.ndarray, n: int) -> np.ndarray:
    if n>len(mask): return np.array([],dtype=int)
    return np.flatnonzero(np.convolve(mask.astype(int),np.ones(n,dtype=int),mode="valid")==n)

def metrics(trace: np.ndarray, start: int, stop: int, anchor: int, quiet: np.ndarray) -> dict[str,float]:
    event=np.clip(trace[start:stop],0,None); n=stop-start; starts=valid_quiet_starts(quiet,n)
    qa=np.asarray([np.sum(np.clip(trace[s:s+n],0,None)) for s in starts]); qe=np.asarray([np.sum(np.clip(trace[s:s+n],0,None)**2) for s in starts])
    area=float(event.sum()); energy=float(np.sum(event**2)); qarea=float(np.median(qa)); qenergy=float(np.median(qe))
    z=aligned(np.clip(trace,0,None),anchor); denom=float(np.nansum(z[PRE-15:PRE+21])); peak_a=float(trace[anchor])
    return {"event_area":area,"event_energy":energy,"event_area_fraction":area/max(area+qarea,1e-9),
            "event_energy_fraction":energy/max(energy+qenergy,1e-9),
            "core_fraction":float(np.nansum(z[PRE-2:PRE+3])/max(denom,1e-9)),
            "pre_shoulder_fraction":float(np.nansum(z[PRE-15:PRE-2])/max(denom,1e-9)),
            "post_shoulder_fraction":float(np.nansum(z[PRE+3:PRE+21])/max(denom,1e-9)),
            "halfmax_width_frames":float(halfmax_width(np.clip(trace,0,None),anchor)),"anchor_value":peak_a}

def paired_summary(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    endpoints=("event_area_fraction","event_energy_fraction","core_fraction","pre_shoulder_fraction","post_shoulder_fraction","halfmax_width_frames","peak_preservation")
    rng=np.random.default_rng(SEED); out=[]
    for src in SOURCES:
        sr=[r for r in rows if r["source"]==src]; by=defaultdict(dict); sites={}
        for r in sr: by[r["observation_id"]][r["feature_id"]]=r; sites[r["observation_id"]]=r["site_id"]
        for feat in FEATURES:
            for ep in endpoints:
                g=defaultdict(list)
                for oid,v in by.items(): g[sites[oid]].append(float(v[feat][ep])-float(v["A"][ep]))
                x=np.asarray([np.mean(g[s]) for s in sorted(g)]); draws=np.asarray([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(BOOTSTRAPS)])
                vals=[float(v[feat][ep]) for v in by.values()]
                out.append({"source":src,"feature_id":feat,"endpoint":ep,"occurrences":len(vals),"sites":len(g),"median_value":float(np.median(vals)),"paired_site_mean_delta_vs_A":float(x.mean()),"delta_ci95_low":float(np.quantile(draws,.025)),"delta_ci95_high":float(np.quantile(draws,.975))})
    return out

def bootstrap_atlas(rows: list[dict[str,Any]], trace_map: dict[tuple[str,str,str],np.ndarray]) -> list[dict[str,Any]]:
    rng=np.random.default_rng(SEED); out=[]
    for src in SOURCES:
      for feat in FEATURES:
        g=defaultdict(list)
        for r in rows:
          if r["source"]==src and r["feature_id"]==feat:
            z=trace_map[(r["observation_id"],src,feat)]; g[r["site_id"]].append(z/max(np.nanmax(z),1e-9))
        site=np.asarray([np.nanmean(g[s],axis=0) for s in sorted(g)]); point=np.nanmean(site,axis=0)
        draws=np.asarray([np.nanmean(site[rng.integers(0,len(site),len(site))],axis=0) for _ in range(BOOTSTRAPS)])
        for offset,(m,lo,hi) in enumerate(zip(point,np.nanquantile(draws,.025,axis=0),np.nanquantile(draws,.975,axis=0)),-PRE): out.append({"source":src,"feature_id":feat,"offset_frames":offset,"site_mean":float(m),"ci95_low":float(lo),"ci95_high":float(hi)})
    return out

def atlas_figure(path: Path, atlas: list[dict[str,Any]]) -> None:
    fig,axes=plt.subplots(3,2,figsize=(15,12),sharex=True,sharey=True,constrained_layout=True)
    groups=(("A","U","H"),("A","A2","P","C"))
    for i,src in enumerate(SOURCES):
      for j,feats in enumerate(groups):
        ax=axes[i,j]
        for feat in feats:
          q=[r for r in atlas if r["source"]==src and r["feature_id"]==feat]; x=np.asarray([r["offset_frames"] for r in q]); m=np.asarray([r["site_mean"] for r in q]); lo=np.asarray([r["ci95_low"] for r in q]); hi=np.asarray([r["ci95_high"] for r in q]); ax.plot(x,m,label=feat,color=COLORS[feat],lw=2);ax.fill_between(x,lo,hi,color=COLORS[feat],alpha=.12)
        ax.axvline(0,color="#222",ls="--",lw=1);ax.grid(alpha=.18);ax.set_title(f"{src}: {'envelope morphology' if j==0 else 'all operator shapes'}");ax.legend(ncol=len(feats),fontsize=8)
        if j==0: ax.set_ylabel("own-window-peak normalized evidence")
    axes[-1,0].set_xlabel("frames from A event peak");axes[-1,1].set_xlabel("frames from A event peak");fig.suptitle("Temporal contextual-envelope morphology | 106 occurrences, 50 sites | pointwise site-bootstrap bands");fig.savefig(path,dpi=160);plt.close(fig)

def metric_figure(path: Path, summary: list[dict[str,Any]]) -> None:
    eps=("event_area_fraction","event_energy_fraction","core_fraction","pre_shoulder_fraction","post_shoulder_fraction","halfmax_width_frames")
    fig,axes=plt.subplots(3,2,figsize=(15,12),constrained_layout=True); feats=FEATURES[1:]
    for ax,ep in zip(axes.flat,eps,strict=True):
      x=np.arange(len(feats)); width=.24
      for k,src in enumerate(SOURCES):
        q=[next(r for r in summary if r["source"]==src and r["feature_id"]==f and r["endpoint"]==ep) for f in feats]; d=np.asarray([r["paired_site_mean_delta_vs_A"] for r in q]);lo=np.asarray([r["delta_ci95_low"] for r in q]);hi=np.asarray([r["delta_ci95_high"] for r in q]);ax.errorbar(x+(k-1)*width,d,yerr=[d-lo,hi-d],fmt="o",capsize=3,label=src)
      ax.axhline(0,color="#333",lw=1);ax.set_xticks(x,feats);ax.set_title(ep.replace("_"," "));ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle("Paired site-mean morphology changes versus A | 95% site-bootstrap intervals");fig.savefig(path,dpi=160);plt.close(fig)

def occurrence_figure(path: Path, item: dict[str,Any], traces: dict[str,dict[str,np.ndarray]], anchors: dict[str,int]) -> None:
    fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True,constrained_layout=True); x=np.arange(-PRE,POST+1)
    for i,src in enumerate(SOURCES):
      for j,norm in enumerate((False,True)):
        ax=axes[i,j]
        for feat in FEATURES:
          z=aligned(traces[src][feat],anchors[src]); z=z/max(np.nanmax(z),1e-9) if norm else z;ax.plot(x,z,color=COLORS[feat],lw=1.2,label=feat)
        ax.axvline(0,color="#222",ls="--");ax.grid(alpha=.18);ax.set_title(f"{src} | {'own-peak normalized' if norm else 'absolute calibrated operator'}")
      axes[i,0].set_ylabel(src)
    axes[0,0].legend(ncol=3,fontsize=8);axes[-1,0].set_xlabel("frames from A event peak");axes[-1,1].set_xlabel("frames from A event peak");fig.suptitle(f"{item['observation_id']} | immutable center ({int(item['x_int'])},{int(item['y_int'])})");fig.savefig(path,dpi=120);plt.close(fig)

def render_video(path: Path, source: str, f: dict[str,np.ndarray], fps: int=20) -> None:
    h,w=f["A"].shape[1:]; cols,rows=2,3; title_h=24; out_w=w*cols; out_h=(h+title_h)*rows
    scales={k:max(float(np.percentile(v[::5,::4,::4],99.5)),1e-6) for k,v in f.items()}
    cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{out_w}x{out_h}","-r",str(fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",str(path)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
      for t in range(FRAMES):
        canvas=Image.new("RGB",(out_w,out_h),"white");draw=ImageDraw.Draw(canvas)
        for n,k in enumerate(FEATURES):
          arr=np.clip(f[k][t]/scales[k],0,1); im=Image.fromarray(np.uint8(arr*255),mode="L").convert("RGB");x=(n%cols)*w;y=(n//cols)*(h+title_h);canvas.paste(im,(x,y+title_h));draw.text((x+6,y+5),f"{source} | {k}: {LABELS[k]} | frame {START_UI+t}",fill="black")
        assert p.stdin is not None;p.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
    finally:
      if p.stdin: p.stdin.close()
      rc=p.wait()
    if rc: raise RuntimeError(f"ffmpeg failed: {rc}")

def run(args: argparse.Namespace) -> dict[str,Any]:
    root=args.output_root.resolve();partial=root.with_name(root.name+".partial")
    if root.exists() or partial.exists(): raise FileExistsError(root)
    partial.mkdir(parents=True);manifest=Path(args.manifest).resolve();order=Path(args.order_root).resolve();protocol=Path(args.protocol).resolve();paths=load_paths(order)
    pre={"status":"ready","protocol":{"path":str(protocol),"sha256":sha256(protocol)},"manifest":{"path":str(manifest),"sha256":sha256(manifest)},"inputs":{k:{"path":str(v),"sha256":sha256(v)} for k,v in paths.items()},"code":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())},"features":list(FEATURES),"bootstraps":BOOTSTRAPS,"seed":SEED,"scientific_claim_promoted":False};atomic_json(partial/"preflight.json",pre);atomic_json(partial/"status.json",{"status":"preflight_ready"})
    if args.preflight_only:return pre
    items=json.loads(manifest.read_text())["items"]; intervals=sorted({(int(i["event_start_ui"]),int(i["event_end_ui"])) for i in items});event_union,quiet=make_event_and_quiet_masks(FRAMES,intervals)
    sites={f"{i['original_roi_id']}@x{int(i['x_int'])}_y{int(i['y_int'])}":(int(i["x_int"]),int(i["y_int"])) for i in items};all_traces=defaultdict(dict);cal={};video_dir=partial/"4_Videos";video_dir.mkdir()
    for src,path in paths.items():
      x=load_source(path,src);a,cal[src]=calibrate(x);u=trailing_max(a,5);f=operators(a,u)
      for site,(xx,yy) in sites.items(): all_traces[(site,src)]={k:np.asarray(v[:,yy,xx],dtype=np.float64) for k,v in f.items()}
      render_video(video_dir/f"{src.lower()}_six_operator_morphology.mp4",src,f);del x,a,u,f;print(f"SOURCE {src} complete",flush=True)
    rows=[];trace_map={};anchors_by_obs={}
    for item in items:
      oid=item["observation_id"];site=f"{item['original_roi_id']}@x{int(item['x_int'])}_y{int(item['y_int'])}";start=int(item["event_start_ui"])-START_UI;stop=int(item["event_end_ui"])-START_UI+1;anchors_by_obs[oid]={}
      for src in SOURCES:
        ts=all_traces[(site,src)];anchor=start+int(np.argmax(ts["A"][start:stop]));anchors_by_obs[oid][src]=anchor
        for feat in FEATURES:
          m=metrics(ts[feat],start,stop,anchor,quiet);m["peak_preservation"]=float(ts[feat][anchor]/max(ts["A"][anchor],1e-9));rows.append({"observation_id":oid,"site_id":site,"canonical_roi_id":item["canonical_roi_id"],"burst_id":int(item["burst_id"]),"source":src,"feature_id":feat,"anchor_ui":START_UI+anchor,**m});trace_map[(oid,src,feat)]=aligned(ts[feat],anchor)
    summary=paired_summary(rows);atlas=bootstrap_atlas(rows,trace_map);one=partial/"1_Expert_Annotations";two=partial/"2_Model_Annotations";three=partial/"3_Comparison";(three/"occurrence_panels").mkdir(parents=True);one.mkdir();two.mkdir();(one/"README.md").write_text("# Expert annotations\n\nReuses validated canonical-v7 upstream media; no identity or coordinate changes.\n");(two/"README.md").write_text("# Model annotations\n\nNot applicable: no detector, candidates, or model coordinates.\n")
    write_tsv(three/"occurrence_morphology_metrics.tsv",rows);write_tsv(three/"paired_morphology_summary.tsv",summary);write_tsv(three/"aligned_trace_atlas.tsv",atlas);atlas_figure(three/"aligned_trace_atlas.png",atlas);metric_figure(three/"paired_morphology_changes.png",summary)
    for n,item in enumerate(items,1):
      site=f"{item['original_roi_id']}@x{int(item['x_int'])}_y{int(item['y_int'])}";occurrence_figure(three/f"occurrence_panels/{item['observation_id']}.png",item,{s:all_traces[(site,s)] for s in SOURCES},anchors_by_obs[item["observation_id"]]);
      if n%20==0:print(f"PANEL {n}/{len(items)}",flush=True)
    nonzero=[r for r in summary if r["feature_id"]!="A" and (r["delta_ci95_low"]>0 or r["delta_ci95_high"]<0)];result={"status":"complete_exploratory","population":{"occurrences":len(items),"sites":len(sites)},"morphology_intervals_excluding_zero":len(nonzero),"scientific_claim_promoted":False,"calibration":cal};atomic_json(partial/"summary.json",result);atomic_json(partial/"llm_context.json",{"experiment":"NREV-EXP-0033 temporal contextual-envelope morphology audit","grain":"106 occurrences nested in 50 immutable sites","primary_figures":["3_Comparison/aligned_trace_atlas.png","3_Comparison/paired_morphology_changes.png"],"videos":3,"occurrence_panels":106,"limitations":["one previously analyzed recording","pointwise descriptive intervals","quiet references are not verified negatives","no detection precision specificity identity or transfer claim"]})
    checks={"population":len(items)==106 and len(sites)==50,"metric_rows":len(rows)==106*3*6,"summary_rows":len(summary)==3*6*7,"atlas_rows":len(atlas)==3*6*(PRE+POST+1),"panels":len(list((three/"occurrence_panels").glob("*.png")))==106,"videos":len(list(video_dir.glob("*.mp4")))==3,"finite":all(np.isfinite(float(r["event_area_fraction"])) for r in rows)};atomic_json(partial/"validation.json",{"status":"passed" if all(checks.values()) else "failed","checks":checks,"scientific_audit":{"mode":"analysis_only_reuses_validated_upstream_expert_media","expert":"upstream canonical-v7 media","model":"not_applicable_no_new_detections","comparison_panel_count":106}});(partial/"REPORT.md").write_text("# Temporal contextual-envelope morphology audit v1\n\nVisual-first within-recording operator audit. Scientific promotion is false.\n");atomic_json(partial/"status.json",{"status":"complete_exploratory","validation":"passed" if all(checks.values()) else "failed","scientific_claim_promoted":False});files=sorted(p for p in partial.rglob("*") if p.is_file() and p.name!="artifact_index.json");atomic_json(partial/"artifact_index.json",{"artifacts":[{"path":str(p.relative_to(partial)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files]});partial.replace(root);return result

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--order-root",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--preflight-only",action="store_true");print(json.dumps(run(p.parse_args()),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
