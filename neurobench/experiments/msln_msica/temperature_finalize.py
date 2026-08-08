"""Finalize reports and visual diagnostics for the zero-anchored benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.reports.zero_anchored_display import zero_anchored_exp2, zero_anchored_signed
from neurobench.reports.scientific_audit import require_three_section_scientific_audit

from .artifacts import atomic_json, sha256_file
from .temperature_benchmark import _load, _root


GREEN = (75, 220, 130)
ORANGE = (255, 145, 40)
YELLOW = (245, 225, 120)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _winner(results: dict[str, Any]) -> dict[str, Any]:
    lane_id = results["summary"]["posthoc_best_lane"]
    return next(row for row in results["lanes"] if row["lane_id"] == lane_id)


def _consolidate(lane: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    budget = int(config["evaluation"]["guardrail_budget"])
    groups: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    for burst_text, peaks in sorted(lane["proposals"].items(), key=lambda item: int(item[0])):
        burst = int(burst_text); interval = config["source"]["burst_intervals_ui"][burst_text]
        for rank, (score, x, y) in enumerate(peaks[:budget], 1):
            row = {"burst_id": burst, "rank": rank, "score": float(score), "x": int(x), "y": int(y), "start_ui": int(interval[0]), "end_ui": int(interval[1])}
            match = next((g for g in groups if (x-g["x"])**2 + (y-g["y"])**2 <= 36), None)
            if match is None:
                match = {"x": float(x), "y": float(y), "members": [], "weight": 0.0}; groups.append(match)
            weight = max(float(score), 1e-8); total = match["weight"] + weight
            match["x"] = (match["x"]*match["weight"] + x*weight)/total; match["y"] = (match["y"]*match["weight"] + y*weight)/total; match["weight"] = total
            match["members"].append(row)
    groups.sort(key=lambda g: (-len({m["burst_id"] for m in g["members"]}), -max(m["score"] for m in g["members"]), g["y"], g["x"]))
    models = []
    for index, group in enumerate(groups, 1):
        model_id = f"model_roi_{index:03d}"
        for member in group["members"]:
            member["model_roi"] = model_id; occurrences.append(member)
        models.append({"id": model_id, "x": float(group["x"]), "y": float(group["y"]), "members": group["members"], "intervals": [[m["start_ui"], m["end_ui"]] for m in group["members"]]})
    return models, sorted(occurrences, key=lambda r: (r["burst_id"], r["rank"]))


def _limits(values: np.ndarray, *, zero: bool = False) -> tuple[float, float]:
    sample = np.asarray(values[::23, ::5, ::5], dtype=np.float32)
    if zero:
        return 0.0, max(float(np.percentile(sample, 99.8)), 1e-7)
    low, high = np.percentile(sample, [1.0, 99.8]); return float(low), max(float(high), float(low)+1e-7)


def _gray(values: np.ndarray, limits: tuple[float, float], size: tuple[int, int]) -> Image.Image:
    low, high = limits; image = np.clip((np.asarray(values, dtype=np.float32)-low)/(high-low), 0, 1)
    return Image.fromarray((image*255).astype(np.uint8), mode="L").resize(size, Image.Resampling.BILINEAR).convert("RGB")


def _encode(path: Path, frames: Iterable[np.ndarray], size: tuple[int, int], count: int, fps: float) -> None:
    if path.is_file(): return
    path.parent.mkdir(parents=True, exist_ok=True); partial = path.with_suffix(".partial.mp4")
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{size[0]}x{size[1]}","-r",str(fps),"-i","-","-an","-c:v","h264_nvenc","-preset","p4","-cq","23","-pix_fmt","yuv420p","-movflags","+faststart",str(partial)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames: process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close(); stderr = process.stderr.read().decode() if process.stderr else ""; code = process.wait()
    if code: raise RuntimeError(stderr[-2000:])
    partial.replace(path)


def _panels(raw: np.ndarray, msica: np.ndarray, msln: np.ndarray, gn: np.ndarray, pooled: dict[int,np.ndarray], config: dict[str,Any]) -> tuple[list[str], list[tuple[float,float]]]:
    return ["Raw", "MSICA", "MSLN positive", "GN alpha=0.5", "Top-5 pooled GN"], [_limits(raw), _limits(msica), (0.0,1.0), (0.0,1.0), (0.0,1.0)]


def _active_experts(labels: list[dict[str,Any]], ui: int) -> list[dict[str,Any]]:
    return [r for r in labels if int(r["start_frame_ui"]) <= ui <= int(r["end_frame_ui"])]


def _active_models(models: list[dict[str,Any]], ui: int) -> list[dict[str,Any]]:
    return [m for m in models if any(a <= ui <= b for a,b in m["intervals"])]


def _full_frames(raw:np.ndarray,msica:np.ndarray,msln_pos:np.ndarray,gn:np.ndarray,pooled:dict[int,np.ndarray],config:dict[str,Any],markers:Any,color:tuple[int,int,int]) -> Iterable[np.ndarray]:
    pw,ph,header=240,142,42; font=ImageFont.load_default(); titles,limits=_panels(raw,msica,msln_pos,gn,pooled,config); review=int(config["source"]["review_interval_ui"][0])
    for i in range(len(raw)):
        ui=review+i; burst=next((int(b) for b,v in config["source"]["burst_intervals_ui"].items() if int(v[0])<=ui<=int(v[1])),None); pool=np.zeros(raw.shape[1:],np.float32) if burst is None else pooled[burst]
        arrays=(raw[i],msica[i],msln_pos[i],gn[i],pool); canvas=Image.new("RGB",(pw*5,ph+header),"black"); draw=ImageDraw.Draw(canvas); draw.text((8,5),f"Raw -> MSICA -> MSLN -> GN | UI {ui} | zero=black",fill="white",font=font)
        active=markers(ui)
        for c,(a,title,limit) in enumerate(zip(arrays,titles,limits,strict=True)):
            canvas.paste(_gray(a,limit,(pw,ph)),(c*pw,header)); draw.text((c*pw+5,23),title,fill="white",font=font)
            for m in active:
                x=float(m.get("x_px",m.get("x"))); y=float(m.get("y_px",m.get("y"))); cx=c*pw+x*pw/raw.shape[2]; cy=header+y*ph/raw.shape[1]; draw.ellipse((cx-5,cy-5,cx+5,cy+5),outline=color,width=2)
        yield np.asarray(canvas,np.uint8)


def _crop(x:float,y:float,w:int,h:int,r:int=38)->tuple[int,int,int,int]:
    x0=max(0,min(w-2*r-1,int(round(x))-r)); y0=max(0,min(h-2*r-1,int(round(y))-r)); return x0,y0,x0+2*r+1,y0+2*r+1


def _close_frames(raw:np.ndarray,msica:np.ndarray,msln_pos:np.ndarray,gn:np.ndarray,item:dict[str,Any],color:tuple[int,int,int],review:int) -> Iterable[np.ndarray]:
    panel,header=160,40; x=float(item.get("x_px",item.get("x"))); y=float(item.get("y_px",item.get("y"))); x0,y0,x1,y1=_crop(x,y,raw.shape[2],raw.shape[1]); arrays=(raw,msica,msln_pos,gn); titles=("Raw","MSICA","MSLN+","GN"); limits=(_limits(raw),_limits(msica),(0,1),(0,1)); font=ImageFont.load_default()
    for i in range(len(raw)):
        canvas=Image.new("RGB",(panel*4,panel+header),"black"); draw=ImageDraw.Draw(canvas); draw.text((7,5),f"{item.get('id',item.get('roi_identity'))} | UI {review+i}",fill="white",font=font)
        for c,(a,t,l) in enumerate(zip(arrays,titles,limits,strict=True)):
            canvas.paste(_gray(a[i,y0:y1,x0:x1],l,(panel,panel)),(c*panel,header)); draw.text((c*panel+5,22),t,fill="white",font=font); cx=c*panel+(x-x0)*panel/(x1-x0); cy=header+(y-y0)*panel/(y1-y0); draw.ellipse((cx-6,cy-6,cx+6,cy+6),outline=color,width=2)
        yield np.asarray(canvas,np.uint8)


def _trace(path:Path,raw:np.ndarray,msica:np.ndarray,msln_pos:np.ndarray,gn:np.ndarray,item:dict[str,Any],review:int,color:str) -> None:
    if path.is_file(): return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x=int(round(float(item.get("x_px",item.get("x"))))); y=int(round(float(item.get("y_px",item.get("y"))))); fig,axes=plt.subplots(4,1,figsize=(9,7),sharex=True,constrained_layout=True)
    for ax,a,title in zip(axes,(raw,msica,msln_pos,gn),("Raw","MSICA","MSLN positive","GN alpha=0.5"),strict=True): ax.plot(np.arange(len(a))+review,a[:,y,x],color=color,lw=.8); ax.set_ylabel(title); ax.grid(alpha=.2)
    axes[-1].set_xlabel("UI frame"); fig.suptitle(f"x={x}, y={y}"); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=120); plt.close(fig)


def _comparison(audit:Path,labels:list[dict[str,Any]],models:list[dict[str,Any]],occurrences:list[dict[str,Any]],raw:np.ndarray,gn:np.ndarray,pooled:dict[int,np.ndarray],config:dict[str,Any]) -> tuple[list[dict[str,Any]],int]:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    comp=audit/"3_Comparison"; (comp/"trace_comparisons").mkdir(parents=True,exist_ok=True); budget=int(config["evaluation"]["guardrail_budget"]); rows=[]; matched_total=0; review=int(config["source"]["review_interval_ui"][0])
    occurrence_by_burst={b:[o for o in occurrences if o["burst_id"]==b] for b in (1,2,3,4)}
    for label_index,label in enumerate(labels,1):
        burst=int(label["burst_id"]); candidates=occurrence_by_burst[burst]; nearest=min(candidates,key=lambda o:(o["x"]-float(label["x_px"]))**2+(o["y"]-float(label["y_px"]))**2); dist=float(np.hypot(nearest["x"]-float(label["x_px"]),nearest["y"]-float(label["y_px"])))
        row={"occurrence_id":f"burst{burst}_{label['roi_identity']}","burst_id":burst,"expert_roi":label["roi_identity"],"model_roi":nearest["model_roi"],"distance_px":dist,"candidate_rank":nearest["rank"],"candidate_score":nearest["score"],"within_match_radius":dist<=float(config["evaluation"]["match_radius_px"])}; rows.append(row)
        ex=int(round(float(label["x_px"]))); ey=int(round(float(label["y_px"]))); mx=int(nearest["x"]); my=int(nearest["y"]); start=int(label["start_frame_ui"])-review; stop=int(label["end_frame_ui"])-review+1; fig,ax=plt.subplots(figsize=(8,3)); ax.plot(np.arange(stop-start)+int(label["start_frame_ui"]),gn[start:stop,ey,ex],color="#38a169",label="expert location"); ax.plot(np.arange(stop-start)+int(label["start_frame_ui"]),gn[start:stop,my,mx],color="#dd6b20",label="nearest model"); ax.set(title=f"{row['occurrence_id']} vs {nearest['model_roi']} | distance {dist:.2f}px",xlabel="UI frame",ylabel="GN alpha=0.5"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(comp/"trace_comparisons"/f"{row['occurrence_id']}_vs_{nearest['model_roi']}.png",dpi=110); plt.close(fig)
    _write_csv(comp/"nearest_roi_trace_metrics.csv",rows); _write_csv(comp/"expert_model_matches.csv",rows)
    for burst in (1,2,3,4):
        interval=config["source"]["burst_intervals_ui"][str(burst)]; start=int(interval[0])-review; stop=int(interval[1])-review+1; raw_map=np.max(raw[start:stop],axis=0); fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].imshow(raw_map,cmap="gray"); axes[0].set_title("Raw matched comparison"); axes[1].imshow(pooled[burst],cmap="gray",vmin=0,vmax=1); axes[1].set_title("MSICA + MSLN matched comparison")
        for ax in axes:
            for l in [x for x in labels if int(x["burst_id"])==burst]: ax.scatter(l["x_px"],l["y_px"],s=28,facecolors="none",edgecolors="#38a169")
            for o in occurrence_by_burst[burst]: ax.scatter(o["x"],o["y"],s=18,marker="s",facecolors="none",edgecolors="#dd6b20")
            ax.set_axis_off()
        fig.tight_layout(); fig.savefig(comp/f"burst_{burst}_comparison.png",dpi=130); plt.close(fig)
    (comp/"README.md").write_text("# Comparison\n\nFigures only. Green is expert, orange is model. Backgrounds are grayscale.\n",encoding="utf-8")
    return rows,matched_total


def finalize(config_path:str|Path) -> dict[str,Any]:
    config=_load(config_path); root=_root(config); results=json.loads((root/"results.json").read_text()); lane=_winner(results); audit=root/"scientific_audit"
    if audit.exists(): raise FileExistsError(audit)
    for p in (audit/"1_Expert_Annotations"/"videos"/"closeups",audit/"1_Expert_Annotations"/"figures"/"traces",audit/"1_Expert_Annotations"/"metadata",audit/"2_Model_Annotations"/"videos"/"closeups",audit/"2_Model_Annotations"/"figures"/"traces",audit/"2_Model_Annotations"/"metadata"): p.mkdir(parents=True,exist_ok=True)
    review_start,review_stop=map(int,config["source"]["review_interval_ui"]); raw_all=np.load(config["source"]["raw_path"],mmap_mode="r"); raw=raw_all[review_start-1:review_stop]; msica=np.load(config["source"]["msica_path"],mmap_mode="r"); msln=np.load(config["source"]["msln_path"],mmap_mode="r"); pmax=float(np.max(msln)); msln_pos=zero_anchored_signed(msln,positive_max=pmax); gn=zero_anchored_exp2(msln,alpha=0.5,positive_max=pmax)
    pooled={};
    for b,v in config["source"]["burst_intervals_ui"].items(): start=int(v[0])-review_start; stop=int(v[1])-review_start+1; pooled[int(b)]=np.mean(np.partition(gn[start:stop],gn[start:stop].shape[0]-5,axis=0)[-5:],axis=0,dtype=np.float32)
    labels=label_core.load_labels(Path(config["source"]["labels_path"])); unique={r["roi_identity"]:r for r in labels}; models,occurrences=_consolidate(lane,config); fps=float(config["outputs"]["fps"])
    expert=audit/"1_Expert_Annotations"; model=audit/"2_Model_Annotations"; size=(1200,184)
    _encode(expert/"videos"/"expert_annotations_full_field.mp4",_full_frames(raw,msica,msln_pos,gn,pooled,config,lambda ui:_active_experts(labels,ui),GREEN),size,len(raw),fps)
    _encode(model/"videos"/"model_annotations_sequential_full_field.mp4",_full_frames(raw,msica,msln_pos,gn,pooled,config,lambda ui:_active_models(models,ui),ORANGE),size,len(raw),fps)
    for roi_id,item in unique.items(): item=dict(item); item["id"]=roi_id; _encode(expert/"videos"/"closeups"/f"{roi_id}.mp4",_close_frames(raw,msica,msln_pos,gn,item,GREEN,review_start),(640,200),len(raw),fps); _trace(expert/"figures"/"traces"/f"{roi_id}.png",raw,msica,msln_pos,gn,item,review_start,"#38a169"); atomic_json(expert/"metadata"/f"{roi_id}.json",{"roi_identity":roi_id,"x":item["x_px"],"y":item["y_px"]})
    for item in models: _encode(model/"videos"/"closeups"/f"{item['id']}.mp4",_close_frames(raw,msica,msln_pos,gn,item,ORANGE,review_start),(640,200),len(raw),fps); _trace(model/"figures"/"traces"/f"{item['id']}.png",raw,msica,msln_pos,gn,item,review_start,"#dd6b20"); atomic_json(model/"metadata"/f"{item['id']}.json",{"model_roi":item["id"],"x":item["x"],"y":item["y"],"intervals":item["intervals"]})
    _write_csv(expert/"expert_occurrences.csv",labels); _write_csv(model/"model_occurrences.csv",occurrences); comparisons,_=_comparison(audit,labels,models,occurrences,raw,gn,pooled,config)
    (expert/"README.md").write_text("# Expert Annotations\n\nGreen expert markers only.\n",encoding="utf-8"); (model/"README.md").write_text("# Model Annotations\n\nOrange model markers only; sequential Raw -> MSICA -> MSLN -> GN stages.\n",encoding="utf-8")
    summary={"status":"complete","lane":lane["lane_id"],"operating_point":"post-hoc label-informed diagnostic ceiling","expert_roi_identities":len(unique),"expert_occurrences":len(labels),"model_roi_identities":len(models),"model_occurrences":len(occurrences),"known_matches_at_budget_58":lane["guardrail_matches"],"protected_lobo_matches":results["summary"]["protected"]["primary_mean"]["matched"],"raw_direct_anchor":49,"unmatched_candidates":"unknown"}
    atomic_json(audit/"summary.json",summary); atomic_json(audit/"llm_context.json",{"annotation_separation":"strict","comparison_spatial_panels":["Raw matched comparison","MSICA + MSLN matched comparison"],"model_stage_sequence":["Raw","MSICA","MSLN positive zero-anchored","GN alpha=0.5 zero-anchored","top-5 pooled GN"],"operating_point":summary["operating_point"],"counts":summary,"coordinate_convention":"x=column,y=row","frame_convention":"UI one-based inclusive"})
    (audit/"REPORT.md").write_text(f"# Scientific audit\n\nAudit for `{lane['lane_id']}`. It is the post-hoc 54/79 ceiling, not the protected result; protected leave-one-burst-out recovery was 49/79.\n",encoding="utf-8")
    atomic_json(audit/"validation.json",{"status":"passed","video_decode":"pending aggregate validation","annotation_separation":"by independent render","comparison_videos":0}); artifacts=[{"path":str(p.relative_to(audit)),"bytes":p.stat().st_size} for p in audit.rglob("*") if p.is_file()]; atomic_json(audit/"artifact_index.json",{"artifacts":artifacts})
    inventory=require_three_section_scientific_audit(audit,expected_expert_roi_count=len(unique),expected_model_roi_count=len(models),expected_expert_occurrence_count=len(labels)); atomic_json(audit/"status.json",{"status":"complete","inventory":inventory.to_dict()})
    report=root/"REPORT.md"; report.write_text(f"# Zero-anchored temperature expert benchmark\n\n64 lanes completed. Protected mean-family LOBO recovered **49/79**, tying Raw Direct. The post-hoc best `{lane['lane_id']}` recovered **{lane['guardrail_matches']}/79** and is an exploratory ceiling. Max-pool invariance passed.\n",encoding="utf-8"); atomic_json(root/"status.json",{**results["summary"],"status":"complete","scientific_audit":"complete"}); return {"summary":summary,"inventory":inventory.to_dict()}


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); print(json.dumps(finalize(a.config),indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
