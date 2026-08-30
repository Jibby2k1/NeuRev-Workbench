"""External sparse-positive assay for a hash-frozen two-frame representation."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

from .traces import candidate_signature_summary, paired_trace_metrics
from neurobench.experiments.neuron_identifiability.contracts import disk_geometry_hash


def _json(path: Path, value: object) -> None:
    tmp=path.with_suffix(path.suffix+".partial"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); tmp.replace(path)


def _read_rows(path: Path) -> list[dict[str,Any]]:
    with path.open(newline="",encoding="utf-8") as stream: raw=list(csv.DictReader(stream,delimiter="\t"))
    rows=[]
    for item in raw:
        if item.get("include_inclusive","").lower() != "true": continue
        original=item["original_roi_id"]
        rows.append({"observation_id":item["observation_id"],"roi_id":item["canonical_roi_id"],
                     "original_roi_id":original,"observation_site_id":original,
                     "canonical_neuron_id":item["canonical_roi_id"],
                     "burst_id":int(item["burst_id"]),"x":float(item["x_px"]),"y":float(item["y_px"]),
                     "geometry_hash":disk_geometry_hash(float(item["x_px"]),float(item["y_px"]),2),
                     "analysis_view":"original_site_adjudicated_timing",
                     "start":int(item["event_onset_ui"] or item["original_start_frame_ui"])-1,
                     "stop":int(item["event_end_ui"] or item["original_end_frame_ui"])})
    return rows


def _roi_trace(video: np.ndarray,x:float,y:float,radius:int) -> np.ndarray:
    xi,yi=int(round(x)),int(round(y)); y0=max(0,yi-radius); y1=min(video.shape[1],yi+radius+1); x0=max(0,xi-radius); x1=min(video.shape[2],xi+radius+1)
    return np.asarray(video[:,y0:y1,x0:x1],dtype=np.float64).mean(axis=(1,2))


def _operator(raw: np.ndarray, fit: dict[str,Any], selected:int) -> np.ndarray:
    direction=np.asarray(fit["effective_directions"],dtype=float)[selected]; mean=np.asarray(fit["mean"],dtype=float)
    derivative=np.asarray([-1.,1.])/np.sqrt(2); direction*=np.sign(direction@derivative) or 1
    return (np.column_stack((raw[:-1],raw[1:]))-mean)@direction


def _event_score(trace:np.ndarray,start:int,stop:int)->float:
    a=max(0,start); b=min(len(trace),stop-1); return float(np.max(trace[a:b])) if b>a else float("nan")


def run(video_path:Path, labels_path:Path, frozen_path:Path, output:Path, *, radius:int=2, pre:int=20, post:int=20, shifts:int=199, seed:int=20260820)->dict[str,Any]:
    if output.exists(): raise FileExistsError(f"refusing existing output: {output}")
    frozen=json.loads(frozen_path.read_text()); canonical=json.dumps({"fit":frozen["fit"],"selection":frozen["selection"]},sort_keys=True,separators=(",",":")).encode()
    if hashlib.sha256(canonical).hexdigest()!=frozen["sha256"]: raise ValueError("frozen representation fingerprint mismatch")
    rows=_read_rows(labels_path); video=np.load(video_path,mmap_mode="r",allow_pickle=False)
    if len(rows)!=79: raise ValueError(f"expected 79 inclusive occurrences, found {len(rows)}")
    selected=int(frozen["selection"]["selected_component"]); rng=np.random.default_rng(seed)
    records=[]; windows=[]; roi_ids=[]; window_length=pre+max(r["stop"]-r["start"] for r in rows)+post
    unique={(r["observation_site_id"],r["x"],r["y"]) for r in rows}; traces={key:_roi_trace(video,key[1],key[2],radius) for key in unique}
    for row in rows:
        raw=traces[(row["observation_site_id"],row["x"],row["y"])]; ica=_operator(raw,frozen["fit"],selected); diff=np.diff(raw)
        start,stop=row["start"],row["stop"]; a=max(0,start-pre); b=min(len(raw),stop+post)
        raw_window=raw[a:b]; ica_window=ica[a:min(b-1,len(ica))]; diff_window=diff[a:min(b-1,len(diff))]
        n=min(len(raw_window)-1,len(ica_window)); metrics=paired_trace_metrics(raw_window[:n],ica_window[:n],pre_event=min(pre,n-1))
        records.append({**row,"raw_baseline":float(np.median(raw[max(0,start-pre):start])),
                        "ica_event_score":_event_score(ica,start,stop),"difference_event_score":_event_score(diff,start,stop),**metrics})
        padded=np.full(window_length,np.nan); segment=raw[max(0,start-pre):min(len(raw),start-pre+window_length)]; padded[:len(segment)]=segment
        windows.append(padded); roi_ids.append(row["observation_site_id"])
    ica_true=np.asarray([r["ica_event_score"] for r in records]); diff_true=np.asarray([r["difference_event_score"] for r in records])
    valid_offsets=np.arange(80,len(video)-80); valid_offsets=valid_offsets[(valid_offsets<1500)|(valid_offsets>2300)]
    null_ica=[]; null_diff=[]
    for _ in range(shifts):
        offset=int(rng.choice(valid_offsets)); si=[]; sd=[]
        for row in rows:
            raw=traces[(row["observation_site_id"],row["x"],row["y"])]; ica=_operator(raw,frozen["fit"],selected); diff=np.diff(raw); duration=row["stop"]-row["start"]
            si.append(_event_score(ica,offset,offset+duration)); sd.append(_event_score(diff,offset,offset+duration))
        null_ica.append(float(np.mean(si))); null_diff.append(float(np.mean(sd)))
    null_ica=np.asarray(null_ica); null_diff=np.asarray(null_diff)
    signature=candidate_signature_summary(np.asarray(windows),np.asarray(roi_ids))
    summary={"schema_version":1,"status":"external_sparse_positive_assay_complete_audit_incomplete",
             "frozen_representation_sha256":frozen["sha256"],"occurrences":len(rows),"original_sites":len(set(roi_ids)),
             "proposed_canonical_identities":len({r["canonical_neuron_id"] for r in rows}),
             "labels_role":"external_evaluation_only","precision":"not_applicable_sparse_positive_labels",
             "ica":{"mean_event_score":float(ica_true.mean()),"label_shift_mean":float(null_ica.mean()),"label_shift_p_upper":float((1+np.sum(null_ica>=ica_true.mean()))/(1+len(null_ica)))},
             "difference":{"mean_event_score":float(diff_true.mean()),"label_shift_mean":float(null_diff.mean()),"label_shift_p_upper":float((1+np.sum(null_diff>=diff_true.mean()))/(1+len(null_diff)))},
             "ica_vs_difference_event_score_correlation":float(np.corrcoef(ica_true,diff_true)[0,1]),"candidate_signature":signature,
             "limitations":["Sparse positives do not identify precision or false-positive rate.","This ROI-centered assay does not yet produce a frozen full-field candidate panel.","Full scientific-audit videos and candidate close-ups remain incomplete."]}
    output.mkdir(parents=True); tables=output/"tables"; figures=output/"figures"; tables.mkdir(); figures.mkdir()
    with (tables/"occurrence_metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].scatter(diff_true,ica_true,s=18,alpha=.7); ax[0].set(xlabel="Signed difference event score",ylabel="Frozen ICA event score",title="Occurrence scores"); ax[0].grid(alpha=.2)
    ax[1].hist(null_ica,bins=30,alpha=.7,label="shifted-label means"); ax[1].axvline(ica_true.mean(),color="black",label="true-label mean"); ax[1].set(title="ICA label-shift assay",xlabel="mean event score"); ax[1].legend(); fig.tight_layout(); fig.savefig(figures/"external_assay_summary.png",dpi=180); plt.close(fig)
    centered=np.asarray(windows)-np.nanmedian(np.asarray(windows)[:,:pre],axis=1,keepdims=True); scale=np.nanmax(np.abs(centered),axis=1,keepdims=True); heat=centered/np.maximum(scale,1e-12)
    fig,ax=plt.subplots(figsize=(9,6)); im=ax.imshow(heat,aspect="auto",vmin=-1,vmax=1,cmap="gray",interpolation="nearest"); ax.axvline(pre,color="#38a169"); ax.set(xlabel="Samples relative to labeled interval onset",ylabel="Occurrences",title="Amplitude-normalized Raw event windows"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figures/"raw_signature_heatmap.png",dpi=180); plt.close(fig)
    _json(output/"summary.json",summary); _json(output/"llm_context.json",{"status":summary["status"],"frozen_representation_sha256":frozen["sha256"],"primary_metrics":"summary.json","occurrence_table":"tables/occurrence_metrics.csv","primary_figures":["figures/external_assay_summary.png","figures/raw_signature_heatmap.png"]})
    (output/"REPORT.md").write_text(f"# Frozen two-frame ICA external assay\n\nThe frozen representation `{frozen['sha256']}` was evaluated on {len(rows)} canonical sparse-positive occurrences without refitting. ICA label-shift p={summary['ica']['label_shift_p_upper']:.4g}; signed difference p={summary['difference']['label_shift_p_upper']:.4g}; their event-score correlation is {summary['ica_vs_difference_event_score_correlation']:.4f}. Precision is not available. The full scientific audit remains incomplete.\n")
    return summary


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--video-npy",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--frozen",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--roi-radius",type=int,default=2); p.add_argument("--shifts",type=int,default=199); a=p.parse_args()
    result=run(a.video_npy,a.labels,a.frozen,a.output_dir,radius=a.roi_radius,shifts=a.shifts); print(json.dumps({k:result[k] for k in ("status","frozen_representation_sha256","occurrences","original_sites","proposed_canonical_identities")},indent=2))
if __name__=="__main__": main()
