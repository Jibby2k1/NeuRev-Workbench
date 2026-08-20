"""Final guarded ROI-minus-annulus activation-signature diagnostic."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
import numpy as np

from .external_assay import _json, _read_rows, _roi_trace
from .signature_assay import (_annulus_trace, _bootstrap, _corr, _fold,
                              _intervals, _sign_flip_p, _valid_shift_starts, _window)


def _peak_amplitude(trace:np.ndarray,start:int,duration:int,pre:int)->float:
    baseline=trace[start-pre:start]; center=float(np.median(baseline))
    return float(np.max(np.abs(trace[start:start+duration]-center)))


def run(video_path:Path,labels_path:Path,output:Path,*,radius:int=2,annulus_inner:int=3,annulus_outer:int=6,outer_inner:int=7,outer_outer:int=10,pre:int=20,post:int=20,folds:int=5,shifts:int=199,bootstraps:int=5000,seed:int=20260820)->dict[str,Any]:
    if output.exists(): raise FileExistsError(f"refusing existing output: {output}")
    rows=_read_rows(labels_path); video=np.load(video_path,mmap_mode="r",allow_pickle=False)
    if len(rows)!=79 or len({r["roi_id"] for r in rows})!=26: raise ValueError("canonical 79-occurrence/26-ROI contract failed")
    duration=max(r["stop"]-r["start"] for r in rows); length=pre+duration+post; rng=np.random.default_rng(seed)
    geometry={r["roi_id"]:(r["x"],r["y"]) for r in rows}; by_roi={rid:[r for r in rows if r["roi_id"]==rid] for rid in sorted(geometry)}
    roi={rid:_roi_trace(video,*xy,radius) for rid,xy in geometry.items()}
    local={rid:_annulus_trace(video,*xy,annulus_inner,annulus_outer) for rid,xy in geometry.items()}
    outer={rid:_annulus_trace(video,*xy,outer_inner,outer_outer) for rid,xy in geometry.items()}
    residual={rid:roi[rid]-local[rid] for rid in roi}; spatial_control={rid:local[rid]-outer[rid] for rid in roi}
    event={rid:np.stack([_window(residual[rid],r["start"],length,pre) for r in by_roi[rid]]) for rid in by_roi}
    control={rid:np.stack([_window(spatial_control[rid],r["start"],length,pre) for r in by_roi[rid]]) for rid in by_roi}
    valid=_valid_shift_starts(len(video),length,_intervals(rows)); shift_starts=rng.choice(valid,shifts,replace=False)
    records=[]
    for fold in range(folds):
        train=[rid for rid in by_roi if _fold(rid,folds)!=fold]; test=[rid for rid in by_roi if _fold(rid,folds)==fold]
        template=np.median(np.stack([np.median(event[rid],axis=0) for rid in train]),axis=0)
        for rid in test:
            event_template=np.median(event[rid],axis=0); control_template=np.median(control[rid],axis=0)
            shifted_corr=np.asarray([_corr(_window(residual[rid],int(s),length,pre),template) for s in shift_starts])
            event_corr=_corr(event_template,template); spatial_corr=_corr(control_template,template)
            pair=[_corr(event[rid][i],event[rid][j]) for i in range(len(event[rid])) for j in range(i+1,len(event[rid]))]
            true_peaks=np.asarray([_peak_amplitude(residual[rid],r["start"],r["stop"]-r["start"],pre) for r in by_roi[rid]])
            shift_peaks=np.asarray([_peak_amplitude(residual[rid],int(s),duration,pre) for s in shift_starts])
            records.append({"roi_id":rid,"fold":fold,"event_count":len(event[rid]),"held_out_residual_event_correlation":event_corr,
                            "same_roi_shift_mean_correlation":float(shifted_corr.mean()),"spatial_control_event_correlation":spatial_corr,
                            "delta_vs_shift_mean":event_corr-float(shifted_corr.mean()),"delta_vs_spatial_control":event_corr-spatial_corr,
                            "within_roi_pairwise_correlation":float(np.mean(pair)) if pair else float("nan"),
                            "median_event_peak_residual_intensity":float(np.median(true_peaks)),"median_shift_peak_residual_intensity":float(np.median(shift_peaks)),
                            "peak_amplitude_delta":float(np.median(true_peaks)-np.median(shift_peaks))})
    dshift=np.asarray([r["delta_vs_shift_mean"] for r in records]); dspatial=np.asarray([r["delta_vs_spatial_control"] for r in records]); damp=np.asarray([r["peak_amplitude_delta"] for r in records]); held=np.asarray([r["held_out_residual_event_correlation"] for r in records]); within=np.asarray([r["within_roi_pairwise_correlation"] for r in records]); within=within[np.isfinite(within)]
    stats={"delta_vs_same_roi_shift":_bootstrap(dshift,bootstraps,rng),"delta_vs_spatial_control":_bootstrap(dspatial,bootstraps,rng),"event_peak_amplitude_delta":_bootstrap(damp,bootstraps,rng)}
    for key,values in (("delta_vs_same_roi_shift",dshift),("delta_vs_spatial_control",dspatial),("event_peak_amplitude_delta",damp)): stats[key]["sign_flip_p_upper"]=_sign_flip_p(values,9999,rng)
    s1=bool(len(within)>0 and float(np.median(within))>0.5); s2=bool(stats["delta_vs_same_roi_shift"]["ci95_low"]>0 and stats["delta_vs_spatial_control"]["ci95_low"]>0); s3=bool(s2 and stats["delta_vs_same_roi_shift"]["sign_flip_p_upper"]<=.01 and stats["delta_vs_spatial_control"]["sign_flip_p_upper"]<=.01)
    level="S3_held_out_residual_signature_within_recording" if s3 else "S2_across_roi_residual_signature" if s2 else "S1_within_roi_residual_repeatability" if s1 else "S0_no_reproducible_residual_signature"
    decision="advance_to_bounded_multitime_temporal_ica" if s3 else "stop_signature_branch"
    summary={"schema_version":1,"status":"roi_minus_annulus_signature_complete_audit_incomplete","occurrences":len(rows),"canonical_rois":len(records),"folds":folds,
             "representation":"amplitude-preserving ROI mean minus 3-6 px annulus mean","spatial_control":"3-6 px annulus mean minus 7-10 px outer-ring mean",
             "shape_normalization":"pre-event median and MAD only; residual amplitude retained separately; no peak alignment or scaling",
             "held_out_event_correlation":{"mean":float(held.mean()),"median":float(np.median(held))},"within_roi_pairwise_correlation":{"mean":float(within.mean()),"median":float(np.median(within))},
             **stats,"signature_evidence_level":level,"decision":decision,
             "limitations":["Single recording and four shared burst intervals.","Labels define evaluation windows.","Patch-minus-annulus can suppress spatially broad neural signals as well as nuisance signals.","Sparse positives do not identify precision.","Scientific-audit media remain incomplete."]}
    output.mkdir(parents=True); (output/"tables").mkdir(); (output/"figures").mkdir()
    with (output/"tables/roi_residual_generalization_metrics.csv").open("w",newline="") as stream: writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4)); ax[0].scatter([r["same_roi_shift_mean_correlation"] for r in records],held,label="same-ROI shift",alpha=.75); ax[0].scatter([r["spatial_control_event_correlation"] for r in records],held,label="annulus - outer ring",alpha=.75); lo=min(ax[0].get_xlim()[0],ax[0].get_ylim()[0]); hi=max(ax[0].get_xlim()[1],ax[0].get_ylim()[1]); ax[0].plot([lo,hi],[lo,hi],"k--"); ax[0].set(xlabel="Control correlation",ylabel="Held-out ROI-minus-annulus correlation",title="Residual signature generalization"); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].boxplot([dshift,dspatial],tick_labels=["event - shift","event - spatial"]); ax[1].axhline(0,color="black",ls="--"); ax[1].set(title="Correlation gates",ylabel="Per-ROI correlation delta"); ax[1].grid(axis="y",alpha=.2)
    ax[2].boxplot([damp],tick_labels=["event - shift"]); ax[2].axhline(0,color="black",ls="--"); ax[2].set(title="Residual peak amplitude",ylabel="Per-ROI intensity delta"); ax[2].grid(axis="y",alpha=.2); fig.suptitle(level.replace('_',' ')); fig.tight_layout(); fig.savefig(output/"figures/residual_signature_generalization.png",dpi=180); plt.close(fig)
    _json(output/"summary.json",summary); _json(output/"llm_context.json",{"status":summary["status"],"signature_evidence_level":level,"decision":decision,"primary_metrics":"summary.json","roi_table":"tables/roi_residual_generalization_metrics.csv","primary_figure":"figures/residual_signature_generalization.png"})
    (output/"REPORT.md").write_text(f"# ROI-minus-annulus signature diagnostic\n\nThe final guarded signature diagnostic assigned **{level}** and decision **{decision}**. Mean held-out correlation was {held.mean():.4f}. Correlation delta versus same-ROI shifts was {stats['delta_vs_same_roi_shift']['mean']:.4f} (95% CI {stats['delta_vs_same_roi_shift']['ci95_low']:.4f}, {stats['delta_vs_same_roi_shift']['ci95_high']:.4f}); versus the matched spatial residual it was {stats['delta_vs_spatial_control']['mean']:.4f} (95% CI {stats['delta_vs_spatial_control']['ci95_low']:.4f}, {stats['delta_vs_spatial_control']['ci95_high']:.4f}). Median event peak residual amplitude exceeded shifted windows by {stats['event_peak_amplitude_delta']['mean']:.4f} intensity units per ROI. This is a one-recording analytic diagnostic, not a precision claim.\n")
    return summary


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--video-npy",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--shifts",type=int,default=199); p.add_argument("--bootstraps",type=int,default=5000); a=p.parse_args()
    s=run(a.video_npy,a.labels,a.output_dir,shifts=a.shifts,bootstraps=a.bootstraps); print(json.dumps({k:s[k] for k in ("status","signature_evidence_level","decision")},indent=2))
if __name__=="__main__": main()
