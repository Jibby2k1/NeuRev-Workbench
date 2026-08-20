"""Matched-support Raw/pipeline trace and candidate-signature measurements."""
from __future__ import annotations
import numpy as np


def paired_trace_metrics(raw: np.ndarray, pipeline: np.ndarray, *, pre_event: int) -> dict[str, float]:
    raw = np.asarray(raw, dtype=float); pipeline = np.asarray(pipeline, dtype=float)
    if raw.ndim != 1 or raw.shape != pipeline.shape or not 2 <= pre_event < len(raw):
        raise ValueError("traces must align and pre_event must define a nonempty baseline")
    def one(x: np.ndarray) -> tuple[float, float, float, int]:
        baseline = float(np.median(x[:pre_event])); mad = max(float(np.median(np.abs(x[:pre_event]-baseline)))*1.4826, np.finfo(float).eps)
        centered=x-baseline; peak=int(np.argmax(np.abs(centered[pre_event:])))+pre_event
        return baseline, float(centered[peak]), mad, peak
    rb,rp,rn,ri=one(raw); pb,pp,pn,pi=one(pipeline)
    return {"raw_baseline_median":rb,"pipeline_baseline_median":pb,
            "raw_signed_peak":rp,"pipeline_signed_peak":pp,
            "raw_peak_snr":abs(rp)/rn,"pipeline_peak_snr":abs(pp)/pn,
            "snr_gain":(abs(pp)/pn)/max(abs(rp)/rn,np.finfo(float).eps),
            "peak_time_offset_samples":float(pi-ri),
            "trace_correlation":float(np.corrcoef(raw,pipeline)[0,1])}


def candidate_signature_summary(traces: np.ndarray, roi_ids: np.ndarray) -> dict[str, object]:
    """Equal-ROI weighted template and leave-one-ROI-out consistency."""
    x=np.asarray(traces,dtype=float); ids=np.asarray(roi_ids)
    if x.ndim != 2 or len(x) != len(ids) or len(np.unique(ids)) < 2:
        raise ValueError("need [events,time] traces from at least two ROIs")
    centered=x-np.median(x[:,:max(1,x.shape[1]//3)],axis=1,keepdims=True)
    scale=np.max(np.abs(centered),axis=1,keepdims=True); normalized=centered/np.maximum(scale,np.finfo(float).eps)
    per_roi={str(r):np.median(normalized[ids==r],axis=0) for r in np.unique(ids)}
    roi_templates=np.stack(list(per_roi.values())); population=np.median(roi_templates,axis=0)
    loo=[]
    for index,key in enumerate(per_roi):
        other=np.median(np.delete(roi_templates,index,axis=0),axis=0)
        loo.append({"roi_id":key,"correlation":float(np.corrcoef(roi_templates[index],other)[0,1])})
    singular=np.linalg.svd(roi_templates-roi_templates.mean(0),compute_uv=False)
    explained=float(singular[0]**2/max(np.sum(singular**2),np.finfo(float).eps))
    return {"event_count":len(x),"roi_count":len(per_roi),"population_template":population.tolist(),
            "leave_one_roi_out":loo,"mean_leave_one_roi_out_correlation":float(np.mean([v["correlation"] for v in loo])),
            "first_temporal_pca_variance_explained":explained}
