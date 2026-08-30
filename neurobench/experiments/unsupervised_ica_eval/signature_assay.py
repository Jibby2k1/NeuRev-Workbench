"""ROI-held-out Raw activation-signature null and generalization assay."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

from .external_assay import _json, _read_rows, _roi_trace


def _annulus_trace(video:np.ndarray,x:float,y:float,inner:int,outer:int)->np.ndarray:
    xi,yi=int(round(x)),int(round(y)); yy,xx=np.ogrid[-outer:outer+1,-outer:outer+1]
    mask=(xx*xx+yy*yy>=inner*inner)&(xx*xx+yy*yy<=outer*outer)
    y0,y1=yi-outer,yi+outer+1; x0,x1=xi-outer,xi+outer+1
    if y0<0 or x0<0 or y1>video.shape[1] or x1>video.shape[2]: raise ValueError("annulus exceeds video bounds")
    return np.asarray(video[:,y0:y1,x0:x1],dtype=np.float64)[:,mask].mean(axis=1)


def _window(trace:np.ndarray,start:int,length:int,pre:int)->np.ndarray:
    a=start-pre
    if a<0 or a+length>len(trace): raise ValueError("window exceeds trace")
    values=np.asarray(trace[a:a+length],dtype=float); baseline=values[:pre]
    center=float(np.median(baseline)); scale=max(float(np.median(np.abs(baseline-center)))*1.4826,np.finfo(float).eps)
    return (values-center)/scale


def _corr(a:np.ndarray,b:np.ndarray)->float:
    if np.std(a)<=np.finfo(float).eps or np.std(b)<=np.finfo(float).eps: return 0.0
    return float(np.corrcoef(a,b)[0,1])


def _fold(roi_id:str,folds:int)->int:
    return int(hashlib.sha256(roi_id.encode()).hexdigest()[:8],16)%folds


def _intervals(rows:list[dict[str,Any]])->list[tuple[int,int]]:
    return sorted({(r["start"],r["stop"]) for r in rows})


def _valid_shift_starts(n_frames:int,length:int,excluded:list[tuple[int,int]],margin:int=30)->np.ndarray:
    starts=[]
    for start in range(20,n_frames-length-20):
        window=(start,start+length)
        if all(window[1]<a-margin or window[0]>b+margin for a,b in excluded): starts.append(start)
    return np.asarray(starts,dtype=int)


def _bootstrap(values:np.ndarray,draws:int,rng:np.random.Generator)->dict[str,float]:
    means=np.mean(rng.choice(values,(draws,len(values)),replace=True),axis=1)
    return {"mean":float(np.mean(values)),"ci95_low":float(np.quantile(means,.025)),"ci95_high":float(np.quantile(means,.975))}


def _sign_flip_p(values:np.ndarray,draws:int,rng:np.random.Generator)->float:
    observed=float(np.mean(values)); flipped=np.mean(values*rng.choice((-1.,1.),(draws,len(values))),axis=1)
    return float((1+np.sum(flipped>=observed))/(draws+1))


def run(video_path:Path,labels_path:Path,output:Path,*,radius:int=2,inner:int=3,outer:int=6,pre:int=20,post:int=20,folds:int=5,shifts:int=199,bootstraps:int=5000,seed:int=20260820)->dict[str,Any]:
    if output.exists(): raise FileExistsError(f"refusing existing output: {output}")
    rows=_read_rows(labels_path); video=np.load(video_path,mmap_mode="r",allow_pickle=False)
    if len(rows)!=79 or len({r["observation_site_id"] for r in rows})!=27: raise ValueError("79-occurrence/27-site contract failed")
    duration=max(r["stop"]-r["start"] for r in rows); length=pre+duration+post; rng=np.random.default_rng(seed)
    geometry={r["observation_site_id"]:(r["x"],r["y"]) for r in rows}; roi_traces={rid:_roi_trace(video,*xy,radius) for rid,xy in geometry.items()}; annulus={rid:_annulus_trace(video,*xy,inner,outer) for rid,xy in geometry.items()}
    by_roi={rid:[r for r in rows if r["observation_site_id"]==rid] for rid in sorted(geometry)}
    event_windows={rid:np.stack([_window(roi_traces[rid],r["start"],length,pre) for r in by_roi[rid]]) for rid in by_roi}
    annulus_windows={rid:np.stack([_window(annulus[rid],r["start"],length,pre) for r in by_roi[rid]]) for rid in by_roi}
    valid=_valid_shift_starts(len(video),length,_intervals(rows)); shift_starts=rng.choice(valid,shifts,replace=False)
    records=[]
    for fold in range(folds):
        train=[rid for rid in by_roi if _fold(rid,folds)!=fold]; test=[rid for rid in by_roi if _fold(rid,folds)==fold]
        train_roi=np.stack([np.median(event_windows[rid],axis=0) for rid in train]); template=np.median(train_roi,axis=0)
        for rid in test:
            event_template=np.median(event_windows[rid],axis=0); annulus_template=np.median(annulus_windows[rid],axis=0)
            shifted=np.asarray([_corr(_window(roi_traces[rid],int(start),length,pre),template) for start in shift_starts])
            event_corr=_corr(event_template,template); annulus_corr=_corr(annulus_template,template)
            pair=[]
            ew=event_windows[rid]
            for i in range(len(ew)):
                for j in range(i+1,len(ew)): pair.append(_corr(ew[i],ew[j]))
            records.append({"roi_id":rid,"fold":fold,"event_count":len(ew),"held_out_event_correlation":event_corr,
                            "same_roi_shift_mean_correlation":float(shifted.mean()),"same_roi_shift_q95_correlation":float(np.quantile(shifted,.95)),
                            "matched_annulus_event_correlation":annulus_corr,"delta_vs_shift_mean":event_corr-float(shifted.mean()),
                            "delta_vs_annulus":event_corr-annulus_corr,"within_roi_pairwise_correlation":float(np.mean(pair)) if pair else float("nan")})
    delta_shift=np.asarray([r["delta_vs_shift_mean"] for r in records]); delta_annulus=np.asarray([r["delta_vs_annulus"] for r in records]); event=np.asarray([r["held_out_event_correlation"] for r in records]); within=np.asarray([r["within_roi_pairwise_correlation"] for r in records]); within=within[np.isfinite(within)]
    shift_stats=_bootstrap(delta_shift,bootstraps,rng); annulus_stats=_bootstrap(delta_annulus,bootstraps,rng)
    shift_stats["sign_flip_p_upper"]=_sign_flip_p(delta_shift,9999,rng); annulus_stats["sign_flip_p_upper"]=_sign_flip_p(delta_annulus,9999,rng)
    s1=bool(len(within)>0 and float(np.median(within))>0.5); s2=bool(shift_stats["ci95_low"]>0 and annulus_stats["ci95_low"]>0); s3=bool(s2 and shift_stats["sign_flip_p_upper"]<=.01 and annulus_stats["sign_flip_p_upper"]<=.01)
    level="S3_held_out_signature_within_recording" if s3 else "S2_across_roi_signature" if s2 else "S1_within_roi_repeatability" if s1 else "S0_no_reproducible_signature"
    summary={"schema_version":1,"status":"signature_null_generalization_complete_audit_incomplete","occurrences":len(rows),"original_sites":len(records),"proposed_canonical_identities":len({r["canonical_neuron_id"] for r in rows}),"analysis_view":"original_site_adjudicated_timing","folds":folds,"window":{"pre":pre,"event_duration":duration,"post":post,"total":length},
             "normalization":"subtract pre-event median and divide by pre-event MAD x 1.4826; no peak alignment or peak scaling",
             "held_out_event_correlation":{"mean":float(event.mean()),"median":float(np.median(event))},"within_roi_pairwise_correlation":{"mean":float(within.mean()),"median":float(np.median(within))},
             "delta_vs_same_roi_shift":shift_stats,"delta_vs_matched_annulus":annulus_stats,"signature_evidence_level":level,
             "scope":"label-anchored Raw activation morphology within one recording",
             "limitations":["All folds use one recording and four shared burst intervals.","Labels define assay windows; this is external validation, not label-free template discovery.","Sparse positives do not identify precision.","Full scientific-audit media and a frozen full-field candidate panel remain incomplete."]}
    output.mkdir(parents=True); (output/"tables").mkdir(); (output/"figures").mkdir()
    with (output/"tables/roi_generalization_metrics.csv").open("w",newline="") as stream: writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].scatter([r["same_roi_shift_mean_correlation"] for r in records],event,label="same-ROI shift",alpha=.75); ax[0].scatter([r["matched_annulus_event_correlation"] for r in records],event,label="matched annulus",alpha=.75); lo=min(ax[0].get_xlim()[0],ax[0].get_ylim()[0]); hi=max(ax[0].get_xlim()[1],ax[0].get_ylim()[1]); ax[0].plot([lo,hi],[lo,hi],"k--"); ax[0].set(xlabel="Control template correlation",ylabel="Held-out event correlation",title="ROI-held-out generalization"); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].boxplot([delta_shift,delta_annulus],tick_labels=["event - shift","event - annulus"]); ax[1].axhline(0,color="black",ls="--"); ax[1].set(ylabel="Per-ROI correlation delta",title=f"{level.replace('_',' ')}"); ax[1].grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(output/"figures/signature_generalization.png",dpi=180); plt.close(fig)
    _json(output/"summary.json",summary); _json(output/"llm_context.json",{"status":summary["status"],"signature_evidence_level":level,"primary_metrics":"summary.json","roi_table":"tables/roi_generalization_metrics.csv","primary_figure":"figures/signature_generalization.png"})
    (output/"REPORT.md").write_text(f"# Raw signature null/generalization assay\n\nFive-fold ROI-held-out evaluation assigned **{level}**. Mean held-out event correlation was {event.mean():.4f}. The event-minus-same-ROI-shift delta was {shift_stats['mean']:.4f} (95% ROI-bootstrap CI {shift_stats['ci95_low']:.4f}, {shift_stats['ci95_high']:.4f}; p={shift_stats['sign_flip_p_upper']:.4g}). The event-minus-annulus delta was {annulus_stats['mean']:.4f} (95% CI {annulus_stats['ci95_low']:.4f}, {annulus_stats['ci95_high']:.4f}; p={annulus_stats['sign_flip_p_upper']:.4g}). This is within-recording, label-anchored morphology evidence; it is not a precision or cross-recording claim.\n")
    return summary


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--video-npy",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--shifts",type=int,default=199); p.add_argument("--bootstraps",type=int,default=5000); a=p.parse_args()
    s=run(a.video_npy,a.labels,a.output_dir,shifts=a.shifts,bootstraps=a.bootstraps); print(json.dumps({k:s[k] for k in ("status","original_sites","proposed_canonical_identities","signature_evidence_level")},indent=2))
if __name__=="__main__": main()
