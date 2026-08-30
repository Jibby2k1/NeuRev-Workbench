"""Blind full-recording six-panel media for previously unmatched candidate sites."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .contracts import atomic_json,atomic_text

ALIGNMENT_START_UI=1800
BURSTS=((2003,2026),(2040,2063),(2122,2149),(2254,2300))
STAGES=("Raw fluorescence","CS-Parzen ICA","Local standardization")
WIDTH,HEIGHT,HEADER,ROW_H,LEFT_W=960,812,62,250,480


def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))


def _font(size,bold=False):
    name="DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf";path=Path("/usr/share/fonts/truetype/dejavu")/name
    return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default()


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def select_new_sites(site_path:Path,occurrence_path:Path,margin_path:Path)->list[dict]:
    sites=_read(site_path);occurrences=_read(occurrence_path);margins=_read(margin_path)
    by_occ=defaultdict(list);by_margin=defaultdict(list)
    for row in occurrences:by_occ[row["detection_site_id"]].append(row)
    for row in margins:by_margin[row["detection_site_id"]].append(row)
    selected=[]
    for site in sites:
        if int(site["known_positive_occurrences"])!=0:continue
        oid=site["detection_site_id"];rows=by_occ[oid];mr=by_margin[oid]
        selected.append({**site,"robust_snr":float(np.median([float(r["robust_snr"]) for r in rows])),
            "lane_agreement":float(np.mean([float(r["lane_agreement"]) for r in rows])),
            "recurrence":float(np.mean([float(r["recurrence_fraction"]) for r in rows])),
            "nearest_known_distance_px":float(min(float(r["nearest_known_positive_distance_px"]) for r in rows)),
            "class_margin":float(np.median([float(r["relative_distance_margin"]) for r in mr])) if mr else .5,
            "occurrence_ids":";".join(r["detection_occurrence_id"] for r in rows),
            "burst_ids":";".join(r["burst_id"] for r in rows)})
    for key in ("median_raw_peak","median_spatial_specificity","robust_snr","recurrence","lane_agreement"):
        values=np.asarray([float(r[key]) for r in selected]);center=np.median(values);scale=np.median(np.abs(values-center))*1.4826
        if scale<=0:scale=max(float(np.std(values)),1.)
        for row,value in zip(selected,values):row[f"z_{key}"]=float((value-center)/scale)
    for row in selected:
        evidence=.30*row["z_median_raw_peak"]+.25*row["z_median_spatial_specificity"]+.20*row["z_robust_snr"]+.15*row["z_recurrence"]+.10*row["z_lane_agreement"]
        uncertainty=1-min(abs(evidence)/2,1);crowding=math.exp(-row["nearest_known_distance_px"]/6)
        row.update(evidence_score=evidence,uncertainty_score=uncertainty,crowding_score=crowding,
                   priority_score=.50*evidence+.30*uncertainty+.20*crowding)
    return sorted(selected,key=lambda row:-row["priority_score"])


def _limits(values):
    finite=np.asarray(values,float);finite=finite[np.isfinite(finite)];lo,hi=np.percentile(finite,[1,99.5])
    return float(lo),max(float(hi),float(lo)+1e-9)


def _gray(frame,lo,hi,size):
    scaled=np.clip((np.asarray(frame,np.float32)-lo)/(hi-lo),0,1)
    return Image.fromarray((scaled*255).astype(np.uint8),"L").resize(size,Image.Resampling.NEAREST).convert("RGB")


def _frames(stages:list[np.ndarray],site:dict):
    x,y=round(float(site["x_px"])),round(float(site["y_px"]));radius=32
    x0,x1=max(0,x-radius),min(stages[0].shape[2],x+radius+1);y0,y1=max(0,y-radius),min(stages[0].shape[1],y+radius+1)
    clips=[np.asarray(stage[:,y0:y1,x0:x1]) for stage in stages];limits=[_limits(clip) for clip in clips]
    traces=[np.asarray(stage[:,y,x],float) for stage in stages];trace_limits=[_limits(trace) for trace in traces]
    first_ui=ALIGNMENT_START_UI;last_ui=ALIGNMENT_START_UI+len(stages[0])-1
    for frame_index,ui in enumerate(range(first_ui,last_ui+1)):
        canvas=Image.new("RGB",(WIDTH,HEIGHT),(5,8,12));draw=ImageDraw.Draw(canvas)
        draw.text((10,6),f"Candidate {site['blind_id']} | review target ({x},{y}) | UI frame {ui}",fill="white",font=_font(18,True))
        draw.text((10,32),"New unmatched candidate only | yellow cross = review center | traces span complete aligned recording",fill=(180,190,202),font=_font(12))
        for row,(name,clip,(lo,hi),trace,(tlo,thi)) in enumerate(zip(STAGES,clips,limits,traces,trace_limits)):
            top=HEADER+row*ROW_H;image=_gray(clip[frame_index],lo,hi,(LEFT_W,ROW_H-28));canvas.paste(image,(0,top+28))
            draw.text((7,top+5),name,fill="white",font=_font(16));sx=LEFT_W/(x1-x0);sy=(ROW_H-28)/(y1-y0)
            px=(x-x0+.5)*sx;py=top+28+(y-y0+.5)*sy;draw.line((px-8,py,px+8,py),fill=(250,205,70),width=3);draw.line((px,py-8,px,py+8),fill=(250,205,70),width=3)
            gx0,gx1=LEFT_W+38,WIDTH-14;gy0,gy1=top+48,top+ROW_H-22;draw.rectangle((gx0,gy0,gx1,gy1),outline=(70,80,92),fill=(10,14,19))
            draw.text((LEFT_W+8,top+5),"Exact-center full-recording trace",fill="white",font=_font(15))
            for start,end in BURSTS:
                bx0=gx0+(start-first_ui)/(last_ui-first_ui)*(gx1-gx0);bx1=gx0+(end-first_ui)/(last_ui-first_ui)*(gx1-gx0)
                draw.rectangle((bx0,gy0,bx1,gy1),fill=(35,43,52))
            xs=np.linspace(gx0,gx1,len(trace));ys=gy1-np.clip((trace-tlo)/(thi-tlo),0,1)*(gy1-gy0)
            draw.line([(float(a),float(b)) for a,b in zip(xs,ys)],fill=(95,190,245),width=2)
            cursor=gx0+frame_index/max(len(trace)-1,1)*(gx1-gx0);draw.line((cursor,gy0,cursor,gy1),fill=(250,165,55),width=2)
        yield np.asarray(canvas,np.uint8)


def _encode(path:Path,frames)->None:
    partial=path.with_suffix(".partial.mp4");command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r","10","-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","24","-pix_fmt","yuv420p","-movflags","+faststart",str(partial)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);assert process.stdin is not None
    try:
        for frame in frames:process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close();error=process.stderr.read().decode() if process.stderr else "";code=process.wait()
    except BaseException:process.kill();raise
    if code:raise RuntimeError(error[-2000:])
    partial.replace(path)


def build_new_candidate_media(repo:Path,source_root:Path,output:Path)->dict:
    if output.exists():raise FileExistsError(output)
    partial=output.with_name(output.name+".partial")
    run=repo/"Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8"
    paths={"sites":run/"detection_profile_taxonomy_v5/detection_site_profiles.tsv",
           "occurrences":run/"detection_profile_taxonomy_v5/detection_occurrence_profiles.tsv",
           "margins":run/"detection_class_advanced_extensions_v1/membership_margins.tsv"}
    selected=select_new_sites(paths["sites"],paths["occurrences"],paths["margins"])
    if len(selected)!=18 or any(int(row["known_positive_occurrences"]) for row in selected):raise RuntimeError("new-candidate-only population contract failed")
    order=np.random.default_rng(20260828).permutation(len(selected));blind=[selected[i] for i in order]
    for index,row in enumerate(blind,1):row["blind_id"]=f"NC{index:03d}"
    raw_path=source_root/"Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
    cache=source_root/"Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache";stage_paths=[raw_path,cache/"recovery_msica.npy",cache/"recovery_msln.npy"]
    raw_all=np.load(stage_paths[0],mmap_mode="r");ica=np.load(stage_paths[1],mmap_mode="r");ls=np.load(stage_paths[2],mmap_mode="r");raw=raw_all[ALIGNMENT_START_UI-1:ALIGNMENT_START_UI-1+len(ica)];stages=[raw,ica,ls]
    if not (len(raw)==len(ica)==len(ls)==560):raise RuntimeError("stage alignment contract failed")
    partial.mkdir(parents=True,exist_ok=True);items=[]
    for index,site in enumerate(blind,1):
        stem=f"{site['blind_id']}__new_candidate";video=partial/f"{stem}__six_panel_full_trace.mp4";still=partial/f"{stem}__still.png"
        if not (video.is_file() and still.is_file()):
            _encode(video,_frames(stages,site));subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss","25","-i",str(video),"-frames:v","1",str(still)],check=True)
        items.append({"blind_id":site["blind_id"],"detection_site_id":site["detection_site_id"],"x_px":float(site["x_px"]),"y_px":float(site["y_px"]),"video":video.name,"still":still.name,"occurrence_ids":site["occurrence_ids"],"burst_ids":site["burst_ids"]})
        print(f"rendered {index}/18 {site['blind_id']}",flush=True)
    score_rows=[]
    for priority,row in enumerate(selected,1):score_rows.append({"priority_rank":priority,"detection_site_id":row["detection_site_id"],"x_px":row["x_px"],"y_px":row["y_px"],"occurrences":row["occurrences"],"dominant_class":row["dominant_class"],"evidence_score":row["evidence_score"],"uncertainty_score":row["uncertainty_score"],"crowding_score":row["crowding_score"],"priority_score":row["priority_score"]})
    with (partial/"selection_scores.tsv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(score_rows[0]),delimiter="\t");writer.writeheader();writer.writerows(score_rows)
    with (partial/"reviewer_template.tsv").open("w",newline="",encoding="utf-8") as handle:
        columns=["blind_id","site_label","confidence_1_to_5","corrected_x_px","corrected_y_px","duplicate_or_overlap_identity","morphology","primary_limitation","notes"]
        writer=csv.DictWriter(handle,fieldnames=columns,delimiter="\t");writer.writeheader();writer.writerows({"blind_id":row["blind_id"]} for row in blind)
    manifest={"schema_version":1,"status":"generated_visual_validation_pending","scope":"new unmatched candidates only",
        "population":{"candidate_sites":18,"known_positive_target_sites":0,"existing_decisions_reopened":False},
        "selection":{"filter":"known_positive_occurrences == 0","all_eligible_sites_included":True,"review_order":"deterministic blinded permutation","priority_formula":"0.50 evidence + 0.30 uncertainty + 0.20 crowding; evidence = 0.30 raw peak + 0.25 spatial specificity + 0.20 robust SNR + 0.15 recurrence + 0.10 lane agreement after robust cohort standardization"},
        "media":{"frames":560,"fps":10,"layout":"three pipeline-stage close-ups left; three exact-center complete traces right","marker":"yellow target cross only","burst_windows":"shaded on full traces"},
        "input_hashes":{str(path):_sha(path) for path in [*paths.values(),*stage_paths]},"items":items}
    atomic_json(partial/"manifest.json",manifest);atomic_text(partial/"README.md","# New-candidate-only ROI review\n\nAll 18 targets have zero known-positive occurrences. Existing decisions are not review targets. Review the blinded six-panel videos and fill `reviewer_template.tsv`. Allowed site labels: `definite_neuron`, `probable_neuron`, `uncertain`, `artifact_or_noise`, `duplicate_existing_identity`, or `distinct_but_overlapping_neuron`.\n")
    partial.replace(output);return manifest


def validate_media(output:Path,*,visual_inspected:bool=False)->dict:
    manifest=json.loads((output/"manifest.json").read_text());failures=[]
    for item in manifest["items"]:
        video=output/item["video"];still=output/item["still"]
        probe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames","-of","json",str(video)],capture_output=True,text=True)
        streams=json.loads(probe.stdout).get("streams",[]) if probe.returncode==0 else []
        if not streams or int(streams[0].get("width",0))!=WIDTH or int(streams[0].get("height",0))!=HEIGHT or int(streams[0].get("nb_frames",0))!=560:failures.append(item["blind_id"])
        try:
            with Image.open(still) as image:image.verify()
        except Exception:failures.append(item["blind_id"]+"_still")
    checks={"exactly_18_new_sites":len(manifest["items"])==18,"zero_known_positive_targets":manifest["population"]["known_positive_target_sites"]==0,
            "existing_decisions_not_reopened":manifest["population"]["existing_decisions_reopened"] is False,"videos_decode_560_frames":not failures,
            "review_template_18_rows":len(_read(output/"reviewer_template.tsv"))==18}
    status="passed" if all(checks.values()) and visual_inspected else ("pending_visual_inspection" if all(checks.values()) else "failed")
    result={"status":status,"checks":checks,"failures":failures,"visual_inspection":{"performed":visual_inspected,"passed":visual_inspected}}
    atomic_json(output/"validation.json",result);return result
