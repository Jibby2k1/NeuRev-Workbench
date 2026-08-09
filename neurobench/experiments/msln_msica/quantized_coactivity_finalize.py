"""Finalize reports and standard audits for quantized causal coactivity v1."""
from __future__ import annotations

import argparse,csv,json,subprocess
from pathlib import Path
from typing import Any,Iterable
import cupy as cp
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from neurobench.experiments.learnable_contrast import core as label_core
from neurobench.metrics.sparse_detection import match_peaks_one_to_one
from neurobench.reports.scientific_audit import require_three_section_scientific_audit
from neurobench.reports.zero_anchored_display import zero_anchored_exp2,zero_anchored_signed
from .artifacts import atomic_json
from .quantized_coactivity import _load,_root,materialize_lane_gpu
from .temperature_finalize import _consolidate,_encode,_gray,_limits,_crop,_write_csv

GREEN=(75,220,130); ORANGE=(255,145,40); START=1800

def _lane(rows,lane_id): return next(r for r in rows if r["lane_id"]==lane_id)
def _best(rows,predicate): return max((r for r in rows if predicate(r)),key=lambda r:(r["expert"]["matched_by_budget"]["58"],r["metrics"]["integrity_correlation"],r["metrics"]["score"]))

def _protected(rows,family=None,integrity=None):
    panel=[r for r in rows if (family is None or r["family"]==family) and (integrity is None or r["metrics"]["integrity_correlation"]>=integrity)]; folds=[]; total=0
    for hold in (1,2,3,4):
        def key(r):
            train=[x for x in r["expert"]["rows"] if x["budget"]==58 and x["burst_id"]!=hold]; return (-np.mean([x["recall"] for x in train]),-r["metrics"]["integrity_correlation"],-r["metrics"]["score"],r["lane_id"])
        winner=sorted(panel,key=key)[0]; test=next(x for x in winner["expert"]["rows"] if x["budget"]==58 and x["burst_id"]==hold); total+=test["matched"]; folds.append({"held_out_burst":hold,"selected_lane":winner["lane_id"],"matched":test["matched"],"labels":test["labels"],"recall":test["recall"]})
    return {"family":family or "all","integrity_floor":integrity,"matched":total,"labels":79,"folds":folds}

def _markers_expert(labels,ui): return [r for r in labels if r["start_frame_ui"]<=ui<=r["end_frame_ui"]]
def _markers_model(models,ui): return [m for m in models if any(a<=ui<=b for a,b in m["intervals"])]

def _full_frames(stages,titles,markers,color):
    raw=stages[0]; pw,ph,header,label=240,142,42,26; bounds=[_limits(stages[0]),_limits(stages[1])]+[(0,1)]*(len(stages)-2); font=ImageFont.load_default()
    for i in range(len(raw)):
        canvas=Image.new("RGB",(pw*len(stages),header+label+ph),"black"); draw=ImageDraw.Draw(canvas); ui=START+i; draw.text((8,5),f"Raw -> MSICA -> MSLN -> GN -> coactivity | UI {ui}",fill="white",font=font)
        for c,(array,title,bound) in enumerate(zip(stages,titles,bounds,strict=True)):
            x=c*pw; draw.text((x+5,23),title,fill="white",font=font); canvas.paste(_gray(array[i],bound,(pw,ph)),(x,header+label))
            for m in markers(ui):
                px=float(m.get("x_px",m.get("x"))); py=float(m.get("y_px",m.get("y"))); cx=x+px*pw/raw.shape[2]; cy=header+label+py*ph/raw.shape[1]; draw.ellipse((cx-5,cy-5,cx+5,cy+5),outline=color,width=2)
        yield np.asarray(canvas,np.uint8)

def _close_frames(stages,titles,item,color):
    raw=stages[0]; panel,header=150,40; x=float(item.get("x_px",item.get("x"))); y=float(item.get("y_px",item.get("y"))); x0,y0,x1,y1=_crop(x,y,raw.shape[2],raw.shape[1]); bounds=[_limits(stages[0]),_limits(stages[1])]+[(0,1)]*(len(stages)-2); font=ImageFont.load_default()
    for i in range(len(raw)):
        canvas=Image.new("RGB",(panel*len(stages),header+panel),"black"); draw=ImageDraw.Draw(canvas); draw.text((7,5),f"{item.get('id',item.get('roi_identity'))} | UI {START+i}",fill="white",font=font)
        for c,(array,title,bound) in enumerate(zip(stages,titles,bounds,strict=True)):
            canvas.paste(_gray(array[i,y0:y1,x0:x1],bound,(panel,panel)),(c*panel,header)); draw.text((c*panel+4,22),title,fill="white",font=font); cx=c*panel+(x-x0)*panel/(x1-x0); cy=header+(y-y0)*panel/(y1-y0); draw.ellipse((cx-5,cy-5,cx+5,cy+5),outline=color,width=2)
        yield np.asarray(canvas,np.uint8)

def _trace(path,stages,titles,item,color):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x=int(round(float(item.get("x_px",item.get("x"))))); y=int(round(float(item.get("y_px",item.get("y"))))); fig,axes=plt.subplots(len(stages),1,figsize=(9,8),sharex=True,constrained_layout=True)
    for ax,array,title in zip(axes,stages,titles,strict=True): ax.plot(np.arange(len(array))+START,array[:,y,x],color=color,lw=.7); ax.set_ylabel(title); ax.grid(alpha=.2)
    axes[-1].set_xlabel("UI frame"); fig.suptitle(f"x={x}, y={y}"); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=110); plt.close(fig)

def _pooled(array,c):
    result={}
    for b,v in c["source"]["burst_intervals_ui"].items(): a=v[0]-START; z=v[1]-START+1; block=array[a:z]; result[int(b)]=np.mean(np.partition(block,len(block)-5,axis=0)[-5:],axis=0,dtype=np.float32)
    return result

def _comparison(audit,labels,models,occurrences,raw,final,pools,c):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    comp=audit/"3_Comparison"; (comp/"trace_comparisons").mkdir(parents=True); byburst={b:[o for o in occurrences if o["burst_id"]==b] for b in (1,2,3,4)}; matched=set()
    for b,items in byburst.items():
        peaks=[(o["score"],o["x"],o["y"]) for o in items]; labs=[l for l in labels if l["burst_id"]==b]; _,peak_indices=match_peaks_one_to_one(peaks,labs,c["evaluation"]["match_radius_px"]); matched.update((b,pi) for pi in peak_indices)
    rows=[]
    for label in labels:
        b=label["burst_id"]; candidates=byburst[b]; nearest=min(enumerate(candidates),key=lambda z:(z[1]["x"]-label["x_px"])**2+(z[1]["y"]-label["y_px"])**2); idx,o=nearest; dist=float(np.hypot(o["x"]-label["x_px"],o["y"]-label["y_px"])); row={"occurrence_id":f"burst{b}_{label['roi_identity']}","burst_id":b,"expert_roi":label["roi_identity"],"model_roi":o["model_roi"],"distance_px":dist,"candidate_rank":o["rank"],"candidate_score":o["score"],"one_to_one_candidate_selected":(b,idx) in matched}; rows.append(row)
        ex,ey=int(round(label["x_px"])),int(round(label["y_px"])); mx,my=o["x"],o["y"]; a=label["start_frame_ui"]-START; z=label["end_frame_ui"]-START+1; fig,ax=plt.subplots(figsize=(8,3)); ax.plot(np.arange(a,z)+START,final[a:z,ey,ex],color="#38a169",label="expert location"); ax.plot(np.arange(a,z)+START,final[a:z,my,mx],color="#dd6b20",label="nearest model"); ax.set(title=f"{row['occurrence_id']} vs {o['model_roi']} | {dist:.2f}px",xlabel="UI frame",ylabel="final evidence"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(comp/"trace_comparisons"/f"{row['occurrence_id']}_vs_{o['model_roi']}.png",dpi=110); plt.close(fig)
    _write_csv(comp/"nearest_roi_trace_metrics.csv",rows); _write_csv(comp/"expert_model_matches.csv",rows)
    for b,v in c["source"]["burst_intervals_ui"].items():
        b=int(b); a=v[0]-START; z=v[1]-START+1; fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].imshow(np.max(raw[a:z],axis=0),cmap="gray"); axes[0].set_title("Raw matched comparison"); axes[1].imshow(pools[b],cmap="gray",vmin=0,vmax=1); axes[1].set_title("MSICA + MSLN matched comparison")
        for ax in axes:
            for l in [x for x in labels if x["burst_id"]==b]: ax.scatter(l["x_px"],l["y_px"],s=28,facecolors="none",edgecolors="#38a169")
            for o in byburst[b]: ax.scatter(o["x"],o["y"],s=18,marker="s",facecolors="none",edgecolors="#dd6b20")
            ax.set_axis_off()
        fig.tight_layout(); fig.savefig(comp/f"burst_{b}_comparison.png",dpi=120); plt.close(fig)
    images=[Image.open(comp/f"burst_{b}_comparison.png").convert("RGB") for b in (1,2,3,4)]; width=max(x.width for x in images); height=max(x.height for x in images); overview=Image.new("RGB",(2*width,2*height),"white")
    for index,image in enumerate(images): overview.paste(image,((index%2)*width,(index//2)*height))
    overview.save(comp/"spatial_overview.png"); (comp/"README.md").write_text("# Comparison\n\nFigures only; green expert, orange model, grayscale evidence.\n"); return rows

def finalize(config_path):
    c=_load(config_path); root=_root(c); payload=json.loads((root/"results.json").read_text()); rows=payload["lanes"]; summary=payload["summary"]; primary=_lane(rows,summary["frozen_best_lane"]); posthoc=_lane(rows,summary["posthoc_best_lane"]); quant=_best(rows,lambda r:r["family"]=="quantization"); integrity=_best(rows,lambda r:r["metrics"]["integrity_correlation"]>=.95 and r["family"]=="neighborhood"); protected={"all":_protected(rows),"quantization":_protected(rows,"quantization"),"neighborhood":_protected(rows,"neighborhood"),"integrity_ge_0p95":_protected(rows,integrity=.95)}
    signed=np.load(c["source"]["msln_path"],mmap_mode="r"); base_np=zero_anchored_exp2(signed,alpha=.5,positive_max=float(np.max(signed))); base=cp.asarray(base_np); quiet=np.sort(np.asarray(base_np[:100:2,::4,::4]).ravel()); quiet_gpu=cp.asarray(quiet[::max(1,len(quiet)//200000)]); chosen={"primary":primary,"quantization":quant,"integrity":integrity,"posthoc":posthoc}; arrays={}
    for name,lane in chosen.items(): arrays[name]=cp.asnumpy(materialize_lane_gpu(base,lane,c,quiet_gpu)).astype(np.float32,copy=False)
    cp.get_default_memory_pool().free_all_blocks(); raw_all=np.load(c["source"]["raw_path"],mmap_mode="r"); raw=raw_all[START-1:2359]; msica=np.load(c["source"]["msica_path"],mmap_mode="r"); msln_pos=zero_anchored_signed(signed,positive_max=float(np.max(signed))); stages=[raw,msica,msln_pos,base_np,arrays["primary"]]; titles=["Raw","MSICA","MSLN+","GN float","Frozen hard+Q8"]
    models,occurrences=_consolidate(primary,c); labels=label_core.load_labels(Path(c["source"]["labels_path"])); unique={r["roi_identity"]:r for r in labels}; audit=root/"scientific_audit"
    for p in (audit/"1_Expert_Annotations"/"videos"/"closeups",audit/"1_Expert_Annotations"/"figures"/"traces",audit/"1_Expert_Annotations"/"metadata",audit/"2_Model_Annotations"/"videos"/"closeups",audit/"2_Model_Annotations"/"figures"/"traces",audit/"2_Model_Annotations"/"metadata",audit/"4_Vis_Diagnostics"): p.mkdir(parents=True,exist_ok=True)
    expert=audit/"1_Expert_Annotations"; model=audit/"2_Model_Annotations"; fps=c["outputs"]["fps"]; size=(240*5,42+26+142)
    _encode(expert/"videos"/"expert_annotations_full_field.mp4",_full_frames(stages,titles,lambda ui:_markers_expert(labels,ui),GREEN),size,len(raw),fps); _encode(model/"videos"/"model_annotations_sequential_full_field.mp4",_full_frames(stages,titles,lambda ui:_markers_model(models,ui),ORANGE),size,len(raw),fps)
    for rid,item0 in unique.items(): item=dict(item0); item["id"]=rid; _encode(expert/"videos"/"closeups"/f"{rid}.mp4",_close_frames(stages,titles,item,GREEN),(150*5,190),len(raw),fps); _trace(expert/"figures"/"traces"/f"{rid}.png",stages,titles,item,"#38a169"); atomic_json(expert/"metadata"/f"{rid}.json",{"roi_identity":rid,"x":item["x_px"],"y":item["y_px"]})
    for item in models: _encode(model/"videos"/"closeups"/f"{item['id']}.mp4",_close_frames(stages,titles,item,ORANGE),(150*5,190),len(raw),fps); _trace(model/"figures"/"traces"/f"{item['id']}.png",stages,titles,item,"#dd6b20"); atomic_json(model/"metadata"/f"{item['id']}.json",{"model_roi":item["id"],"x":item["x"],"y":item["y"],"intervals":item["intervals"]})
    _write_csv(expert/"expert_occurrences.csv",labels); _write_csv(model/"model_occurrences.csv",occurrences); pools=_pooled(arrays["primary"],c); comparisons=_comparison(audit,labels,models,occurrences,raw,arrays["primary"],pools,c); (expert/"README.md").write_text("# Expert Annotations\n\nExpert markers only.\n"); (model/"README.md").write_text("# Model Annotations\n\nFrozen label-free model markers only.\n")
    vis_stages=[raw,base_np,arrays["primary"],arrays["quantization"],arrays["integrity"],arrays["posthoc"]]; vis_titles=["Raw","Float GN 54/79","Frozen hard+Q8 24/79",f"Quant-only {quant['expert']['matched_by_budget']['58']}/79",f"High-integrity {integrity['expert']['matched_by_budget']['58']}/79",f"Post-hoc ceiling {posthoc['expert']['matched_by_budget']['58']}/79"]; _encode(audit/"4_Vis_Diagnostics"/"01_control_and_finalists.mp4",_full_frames(vis_stages,vis_titles,lambda ui:[],ORANGE),(240*6,42+26+142),len(raw),fps); (audit/"4_Vis_Diagnostics"/"README.md").write_text("# Visualization Diagnostics\n\nAnnotation-free comparison of float GN, frozen hard extrema, quantization-only, high-integrity soft support, and post-hoc ceiling.\n")
    final_summary={**summary,"protected":protected,"audit_primary":"label-free frozen lane","quantization_best":{"lane":quant["lane_id"],"matches":quant["expert"]["matched_by_budget"]["58"]},"high_integrity_best":{"lane":integrity["lane_id"],"matches":integrity["expert"]["matched_by_budget"]["58"],"correlation":integrity["metrics"]["integrity_correlation"]},"posthoc_best":{"lane":posthoc["lane_id"],"matches":posthoc["expert"]["matched_by_budget"]["58"],"correlation":posthoc["metrics"]["integrity_correlation"]},"model_roi_identities":len(models)}; atomic_json(root/"summary.json",final_summary)
    report=f"# Quantized causal coactivity pooling v1\n\nThe frozen label-free selector failed: `{primary['lane_id']}` recovered **24/79** versus float GN **54/79**. Protected all-lane LOBO tied float GN at **{protected['all']['matched']}/79**; quantization-only reached **{protected['quantization']['matched']}/79** and the integrity>=0.95 panel reached **{protected['integrity_ge_0p95']['matched']}/79**. The post-hoc soft q90/median ceiling reached **{posthoc['expert']['matched_by_budget']['58']}/79** but correlation to float GN was only {posthoc['metrics']['integrity_correlation']:.3f}. The high-integrity soft q90/identity alternative reached **{integrity['expert']['matched_by_budget']['58']}/79** at correlation {integrity['metrics']['integrity_correlation']:.3f}. Hard max/min is rejected.\n"; (root/"REPORT.md").write_text(report); (root/"CONCLUSIVE_REPORT.md").write_text(report+"\n## Decision\n\nDo not replace float GN. Retain soft causal q90/identity as the only plausible confirmation family; treat quantization as auxiliary and require independent prospective validation.\n")
    llm={"annotation_separation":"strict","comparison_spatial_panels":["Raw matched comparison","MSICA + MSLN matched comparison"],"model_stage_sequence":titles,"operating_point":"label-free frozen hard-extrema failure control","summary":final_summary,"visual_diagnostics":"4_Vis_Diagnostics/01_control_and_finalists.mp4"}; atomic_json(audit/"llm_context.json",llm); (audit/"REPORT.md").write_text(report); atomic_json(audit/"summary.json",final_summary); atomic_json(audit/"validation.json",{"status":"inventory_ready_media_decode_pending","comparison_videos":0}); atomic_json(audit/"artifact_index.json",{"artifacts":[{"path":str(p.relative_to(audit)),"bytes":p.stat().st_size} for p in audit.rglob("*") if p.is_file()]}); inventory=require_three_section_scientific_audit(audit,expected_expert_roi_count=len(unique),expected_model_roi_count=len(models),expected_expert_occurrence_count=len(labels)); atomic_json(audit/"status.json",{"status":"complete","inventory":inventory.to_dict()}); atomic_json(root/"status.json",{**final_summary,"status":"complete","scientific_audit":"inventory_complete_media_decode_pending"}); return {"summary":final_summary,"inventory":inventory.to_dict()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); print(json.dumps(finalize(a.config),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
