"""Verified reference and dedicated difficult-ROI diagnostic video package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.missed_neuron_video import _Mp4Writer, _probe_video
from neurobench.experiments.learnable_contrast import core as label_core

from .conclusive_config import ConclusiveBatchConfig
from .diagnostic_videos import _spon_frame
from .screen_runner import _atomic_json


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified_reference(source: Path, destination: Path) -> list[dict[str, Any]]:
    manifest=json.loads((source/"manifest.json").read_text())
    expected={row["path"]:row for row in manifest["generated_videos"]+manifest["spon_videos"]}
    rows=[]
    destination.mkdir()
    for name,row in sorted(expected.items()):
        source_path=source/name
        if _sha256(source_path) != row["sha256"]:
            raise RuntimeError(f"reference video hash differs: {source_path}")
        target=destination/name
        shutil.copy2(source_path,target)
        rows.append({"kind":"verified_prior_reference_copy","source":str(source_path.resolve()),
                     "path":str(target.relative_to(destination.parent)),
                     "sha256":_sha256(target),"bytes":target.stat().st_size,
                     "probe":_probe_video(target)})
    return rows


def _difficult_clips(config: ConclusiveBatchConfig, destination: Path) -> list[dict[str, Any]]:
    source=np.load(config.source_video,mmap_mode="r",allow_pickle=False)
    labels=label_core.load_labels(config.labels_tsv)
    groups={"roi_007":["roi_007"],"roi_008":["roi_008"],
            "roi_014":["roi_014"],"roi_017":["roi_017"],
            "roi_019":["roi_019"],"roi_020":["roi_020"],
            "roi_010_015_overlap":["roi_010","roi_015"]}
    start_ui,end_ui=1988,2068
    clip=np.asarray(source[start_ui-1:end_ui],dtype=np.float32)
    baseline=np.median(np.asarray(source[start_ui-1:2002],dtype=np.float32),axis=0)
    records=[]
    for name,identities in groups.items():
        selected=[]
        for identity in identities:
            selected.append(next(row for row in labels if row["roi_identity"]==identity))
        xs=[float(row["x_px"]) for row in selected]; ys=[float(row["y_px"]) for row in selected]
        crop=(max(0,int(min(xs))-22),max(0,int(min(ys))-22),
              min(source.shape[2],int(max(xs))+23),min(source.shape[1],int(max(ys))+23))
        sample=clip[:,crop[1]:crop[3],crop[0]:crop[2]]
        raw_limits=tuple(map(float,np.percentile(sample,[1,99.8])))
        change_hi=max(float(np.percentile(np.maximum(sample-baseline[crop[1]:crop[3],crop[0]:crop[2]],0),99.7)),1.0)
        derivative_hi=max(float(np.percentile(np.maximum(np.diff(sample,axis=0),0),99.7)),1.0)
        path=destination/f"difficult_{name}_ui1988-2068.mp4"
        first=_spon_frame(clip[0],baseline,np.asarray(source[start_ui-2],dtype=np.float32),
            rows=selected,crop=crop,frame_ui=start_ui,burst="1-2 review",raw_limits=raw_limits,
            change_hi=change_hi,derivative_hi=derivative_hi)
        writer=_Mp4Writer(path,first.shape[:2],8.0)
        try:
            writer.write(first)
            for offset in range(1,len(clip)):
                writer.write(_spon_frame(clip[offset],baseline,clip[offset-1],rows=selected,
                    crop=crop,frame_ui=start_ui+offset,burst="1-2 review",raw_limits=raw_limits,
                    change_hi=change_hi,derivative_hi=derivative_hi))
            writer.close()
        except Exception:
            writer.abort(); raise
        records.append({"kind":"dedicated_difficult_roi_review","roi_ids":identities,
                        "source_frames_ui_inclusive":[start_ui,end_ui],"crop_xyxy":list(crop),
                        "path":path.name,"sha256":_sha256(path),"bytes":path.stat().st_size,
                        "probe":_probe_video(path),
                        "interpretation":"Sparse-positive ROI overlay; no detector outcome. The interval includes pre-window activity."})
    return records


def generate(config: ConclusiveBatchConfig) -> dict[str, Any]:
    root=Path(str(config.output_root)+".partial")/"videos"/"reference_and_difficult_roi_v1"
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=False)
    copied=_copy_verified_reference(Path("Outputs/InformationSourceSeparation/diagnostic_videos_v1").resolve(),root/"verified_reference")
    difficult=_difficult_clips(config,root)
    payload={"schema_version":1,"status":"reference_and_difficult_roi_videos_complete",
             "verified_reference_videos":copied,"difficult_roi_videos":difficult,
             "video_count":len(copied)+len(difficult),
             "scientific_contract":"Prior videos are hash-verified copies. Dedicated clips show raw, pseudo-color, baseline change, and derivative with sparse-positive ROI overlays; unmatched pixels remain unknown."}
    _atomic_json(root/"manifest.json",payload)
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(generate(ConclusiveBatchConfig.load(args.config)),indent=2,sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
