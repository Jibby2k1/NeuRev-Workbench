"""Detector-blind raw media for exhaustive bounded-field annotation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import atomic_json, atomic_text


BURSTS = {1: (2003, 2026), 2: (2040, 2063), 3: (2122, 2149), 4: (2254, 2300)}


def scale_uint8(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    if not high > low:
        raise ValueError("display limits must be ordered")
    return np.clip(np.rint((frame.astype(np.float32)-low)*(255.0/(high-low))),0,255).astype(np.uint8)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def _video(frames: np.ndarray, output: Path, *, burst: int, first_ui: int, low: float, high: float, fps: int=10) -> None:
    scale=3; width=frames.shape[2]*scale; image_height=frames.shape[1]*scale; header=54; height=image_height+header
    temporary=output.with_suffix(".partial.mp4")
    command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(temporary)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    assert process.stdin is not None
    font=ImageFont.load_default(size=16)
    try:
        for offset, frame in enumerate(frames):
            gray=Image.fromarray(scale_uint8(frame,low,high),mode="L").resize((width,image_height),Image.Resampling.NEAREST).convert("RGB")
            canvas=Image.new("RGB",(width,height),"black"); canvas.paste(gray,(0,header)); draw=ImageDraw.Draw(canvas)
            draw.text((10,5),f"RAW ONLY | Burst {burst} | UI frame {first_ui+offset}",fill="white",font=font)
            draw.text((10,29),"Global crop: x=318-509, y=111-302",fill="white",font=font)
            process.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
    finally: process.stdin.close()
    stderr=process.stderr.read().decode("utf-8",errors="replace"); code=process.wait()
    if code: raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")
    temporary.replace(output)


def build_bounded_field_media(run_root: Path, video_path: Path) -> Path:
    target=run_root/"review_packet/bounded_field_media"; target.mkdir(parents=True,exist_ok=True)
    video=np.load(video_path,mmap_mode="r",allow_pickle=False); x0,y0,width,height=318,111,192,192; pad=15
    spans={burst:(max(1,start-pad),min(video.shape[0],end+pad)) for burst,(start,end) in BURSTS.items()}
    sample=np.concatenate([np.asarray(video[first-1:last,y0:y0+height,x0:x0+width]).reshape(-1) for first,last in spans.values()])
    low,high=map(float,np.percentile(sample,[1.0,99.8]))
    media=[]
    for burst,(first,last) in spans.items():
        frames=np.asarray(video[first-1:last,y0:y0+height,x0:x0+width]); path=target/f"burst_{burst}_raw_review.mp4"
        _video(frames,path,burst=burst,first_ui=first,low=low,high=high)
        event_start,event_end=BURSTS[burst]; event=np.asarray(video[event_start-1:event_end,y0:y0+height,x0:x0+width],dtype=np.float32)
        for kind,array in (("mean",event.mean(axis=0)),("max",event.max(axis=0))):
            image=Image.fromarray(scale_uint8(array,low,high),mode="L"); image.save(target/f"burst_{burst}_{kind}_projection.png")
        media.append({"burst_id":burst,"path":path.name,"first_frame_ui":first,"last_frame_ui":last,"frame_count":len(frames),"fps":10,"sha256":_sha(path)})
    columns="reviewer_id\tmark_id\tx_px_global\ty_px_global\tburst_id\tclass\tvisibility_confidence\tonset_ui\tpeak_ui\tend_ui\tnotes\ttimestamp\n"
    atomic_text(target/"author_qualitative_review.tsv",columns)
    atomic_text(target/"reviewer_A.tsv","legacy_template_not_required\n"); atomic_text(target/"reviewer_B.tsv","legacy_template_not_required\n")
    manifest={"schema_version":1,"status":"ready_for_optional_single_author_qualitative_review","annotation_blinding":"raw_only_no_detector_candidates_scores_or_ranks","source_video":str(video_path),"source_sha256":_sha(video_path),"label_provenance":{"user_report":"original labels came from two experts","durable_record":"original_workbook","separate_expert_ids_preserved":False,"exhaustive_field_coverage_established":False},"region":{"x0":x0,"y0":y0,"width":width,"height":height,"scope":"local_enriched_not_full_field"},"coordinate_contract":"x=column, y=row, global image coordinates; UI frames one-based inclusive","display":{"shared_across_all_clips":True,"lower_percentile":1.0,"upper_percentile":99.8,"lower_value":low,"upper_value":high,"nearest_neighbor_scale":3},"media":media,"reviewers_required_for_qualitative_audit":1,"precision_estimation_enabled":False,"precision_scope":"not estimated; unmatched candidates remain unknown"}
    atomic_json(target/"media_manifest.json",manifest)
    atomic_text(target/"README.md","""# Bounded-field review media

The four MP4 files are detector-blind Raw-only clips of the frozen 192 x 192 local field. They are available for an optional single-author qualitative audit using `author_qualitative_review.tsv`. Coordinates entered in the TSV are global image coordinates (`x=column`, `y=row`); add 318 to crop-local x and 111 to crop-local y. Frames are one-based and inclusive.

If used, mark visible neuron-like events as `neuron`, `artifact_noise`, or `uncertain`. Do not interpret this optional review as precision: the original two-expert labels are positive-label provenance, while separate exhaustive coverage of all field locations was not preserved. Unmatched candidates remain unknown.
""")
    return target
