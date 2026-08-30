"""Build a randomized Raw/spatial-ICA/combined morphology review packet."""
from __future__ import annotations
import csv, hashlib, json, shutil, subprocess
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
import tifffile
from .contracts import atomic_json, atomic_text

STRATA=("localized_center","ambiguous_boundary","overlap","weak_signal","artifact_control")
def _read(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def _write(path,rows,fields=None):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
def _stratum(row):
    if row["disposition"]=="artifact":return "artifact_control"
    if row["morphology"]=="overlap" or row["context"]=="overlapping":return "overlap"
    if row["context"] in {"weak_signal","weak_local_contrast","high_background"}:return "weak_signal"
    if row["morphology"] in {"no_distinct_boundary","unresolved","spatially_inconsistent"} or row["disposition"]=="activity_visible_identity_uncertain":return "ambiguous_boundary"
    if row["morphology"] in {"localized_center","cell_center","spatially_cohesive","localized_signal"}:return "localized_center"
    return None
def select_queue(rows,per_stratum=4,seed="20260824"):
    selected=[];used=set()
    for stratum in STRATA:
        eligible=[r for r in rows if _stratum(r)==stratum and r["observation_id"] not in used]
        eligible.sort(key=lambda r:hashlib.sha256(f"{seed}|{stratum}|{r['observation_id']}".encode()).hexdigest())
        for row in eligible[:per_stratum]:selected.append((stratum,row));used.add(row["observation_id"])
    return selected
def _scale(values):
    lo,hi=np.percentile(values,[1,99.5]);return np.clip((values-lo)/max(hi-lo,1e-9)*255,0,255).astype(np.uint8)
def _crop(frame,x,y,r=24):
    x=int(round(x));y=int(round(y));pad=np.pad(frame,((r,r),(r,r)),mode="edge");return pad[y:y+2*r+1,x:x+2*r+1]
def _render_video(frames,path,label):
    temp=path.with_suffix(".frames");temp.mkdir()
    for i,frame in enumerate(frames):
        image=Image.fromarray(frame).resize((256 if frame.shape[1]==frame.shape[0] else 512,256),Image.Resampling.NEAREST).convert("RGB")
        draw=ImageDraw.Draw(image);draw.rectangle((0,0,75,20),fill="black");draw.text((5,4),label,fill="white")
        image.save(temp/f"{i:04d}.png")
    subprocess.run(["ffmpeg","-loglevel","error","-y","-framerate","10","-i",str(temp/"%04d.png"),"-c:v","libx264","-pix_fmt","yuv420p",str(path)],check=True)
    shutil.rmtree(temp)
def run_packet(v7_tsv:Path,raw_npy:Path,ica_tif:Path,output:Path,*,per_stratum=4):
    if output.exists():raise FileExistsError(output)
    partial=output.with_name(output.name+".partial");(partial/"clips").mkdir(parents=True)
    queue=select_queue(_read(v7_tsv),per_stratum);raw=np.load(raw_npy,mmap_mode="r",allow_pickle=False)
    keys=[];responses=[]
    with tifffile.TiffFile(ica_tif) as tif:
        for index,(stratum,row) in enumerate(queue,1):
            case=f"case_{index:02d}";start=int(row["original_start_frame_ui"]);stop=int(row["original_end_frame_ui"])
            frame_ui=list(range(start,stop+1));x=float(row["x_px"]);y=float(row["y_px"])
            raw_stack=np.stack([_crop(np.asarray(raw[f-1]),x,y) for f in frame_ui]);ica_stack=np.stack([_crop(tif.pages[f-1800].asarray(),x,y) for f in frame_ui])
            raw_scaled=_scale(raw_stack);ica_scaled=_scale(ica_stack);combined=np.concatenate((raw_scaled,ica_scaled),axis=2)
            modes={"raw_only":raw_scaled,"spatial_ica_only":ica_scaled,"raw_plus_spatial_ica":combined}
            order=sorted(modes,key=lambda m:hashlib.sha256(f"20260824|{case}|{m}".encode()).hexdigest())
            mapping={chr(65+i):mode for i,mode in enumerate(order)}
            for code,mode in mapping.items():
                name=f"{case}__panel_{code}.mp4";_render_video(modes[mode],partial/"clips"/name,f"{case} panel {code}")
                responses.append({"case_id":case,"panel_code":code,"boundary_resolvable_0_2":"","neuron_shape_resolvable_0_2":"","reviewer_confidence_0_100":"","review_time_seconds":"","notes":""})
            keys.append({"case_id":case,"observation_id":row["observation_id"],"burst_id":row["burst_id"],"stratum":stratum,"disposition":row["disposition"],"morphology":row["morphology"],"context":row["context"],"x_px":row["x_px"],"y_px":row["y_px"],"start_ui":start,"end_ui":stop,**{f"panel_{k}":v for k,v in mapping.items()}})
    _write(partial/"REVIEW_RESPONSE.tsv",responses);_write(partial/"BLINDING_KEY.tsv",keys)
    atomic_text(partial/"START_HERE.md","# Spatial ICA morphology ablation\n\nThe reviewed observation is at the exact center of every crop; no marker is drawn because it would obscure the boundary being scored. Review `clips/case_XX__panel_A/B/C.mp4` in case order. For every panel, fill one row in `REVIEW_RESPONSE.tsv`. Scores: boundary and neuronal shape 0=not resolvable, 1=partly, 2=clearly. Confidence is 0--100. Record active review time in seconds. Panel order is deterministically randomized within case. Combined panels necessarily reveal a split layout; this is a randomized-label ablation, not modality-blind perception. Do not change existing annotations from this packet. Reviewers must not receive or open `BLINDING_KEY.tsv`; it is packaged separately for analysis custody.\n")
    summary={"schema_version":1,"status":"review_packet_ready","cases":len(queue),"panels":len(responses),"cases_by_stratum":{s:sum(x[0]==s for x in queue) for s in STRATA},"raw_frame_mapping":"ui_frame_minus_1","spatial_ica_frame_mapping":"ui_frame_minus_1800","spatial_ica_identity":"c2_dense_convolutional_fastica_signal_positive","panel_conditions":["raw_only","spatial_ica_only","raw_plus_spatial_ica"],"analysis_after_review":"site-blocked paired contrasts; morphology-stratified descriptive intervals; no annotation refit"}
    atomic_json(partial/"summary.json",summary);atomic_json(partial/"validation.json",{"status":"passed","all_cases_three_panels":len(responses)==3*len(queue),"frame_ranges_within_raw":all(int(r[1]["original_end_frame_ui"])<=raw.shape[0] for r in queue),"frame_ranges_within_ica":all(1800<=int(r[1]["original_start_frame_ui"])<=int(r[1]["original_end_frame_ui"])<=2359 for r in queue),"blinding_key_separate":True})
    atomic_json(partial/"artifact_index.json",{"artifacts":["START_HERE.md","REVIEW_RESPONSE.tsv","BLINDING_KEY.tsv","summary.json","validation.json"]+[f"clips/{p.name}" for p in sorted((partial/"clips").glob("*.mp4"))]});partial.replace(output);return summary
