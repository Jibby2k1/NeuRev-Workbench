"""Collision-safe orchestration for the pairwise source-separation benchmark."""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.experiments.activity_gate_benchmark import _residual_segments
from neurobench.experiments.frame_difference import _atomic_json, _sha256
from neurobench.experiments.learnable_contrast import core as v1
from neurobench.metrics.sparse_detection import temporal_pool

from .artifacts import atomic_npz, write_candidates, write_figures, write_lane
from .config import PairwiseSeparationConfig
from .evaluation import evaluate_lane
from .fitting import fit_lanes
from .sampling import causal_preprocess


RAW_EXPECTED=0.6056159420289855


def _heartbeat(path:Path,stage:str,**payload) -> None:
    from datetime import datetime,timezone
    with path.open("a",encoding="utf-8") as stream: stream.write(json.dumps({"at":datetime.now(timezone.utc).isoformat(),"stage":stage,**payload},sort_keys=True)+"\n")


def _copy_preflight(source:Path,target:Path,config:PairwiseSeparationConfig) -> dict[str,Any]:
    payload=json.loads((source/"preflight.json").read_text())
    resolved=json.loads((source/"config.resolved.json").read_text())
    expected=json.loads(json.dumps(config.to_dict()))
    if payload.get("experiment_id")!=config.experiment_id or resolved!=expected or not payload.get("ready"):
        raise ValueError("Preflight directory does not match this ready configuration")
    shutil.copy2(source/"preflight.json",target/"preflight.json"); shutil.copy2(source/"label_projection_overlay.png",target/"label_projection_overlay.png")
    return payload


def run(config:PairwiseSeparationConfig, *, preflight_dir:str|Path) -> dict[str,Any]:
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[name]=str(config.resources.cpu_threads)
    if config.output_dir.exists(): raise FileExistsError(f"Output exists: {config.output_dir}")
    started=time.perf_counter(); config.output_dir.mkdir(parents=True,exist_ok=False)
    progress=config.output_dir/"progress.jsonl"; _atomic_json(config.output_dir/"run_state.json",{"status":"running","phase":"preflight"})
    try:
        audit=_copy_preflight(Path(preflight_dir).resolve(),config.output_dir,config)
        _atomic_json(config.output_dir/"config.resolved.json",config.to_dict())
        _atomic_json(config.output_dir/"input_manifest.json",{"source_video_sha256":_sha256(config.source_video),"source_tiff_sha256":_sha256(config.source_tiff),"labels_sha256":_sha256(config.labels_tsv)})
        video=np.load(config.source_video,mmap_mode="r",allow_pickle=False); labels=v1.load_labels(config.labels_tsv)
        f=config.frames; raw=np.asarray(video[f.review_start_ui-1:f.review_end_ui],dtype=np.float32)
        quiet_count=f.quiet_end_ui-f.quiet_start_ui+1
        _heartbeat(progress,"preprocess",status="started")
        filtered=causal_preprocess(raw,config.preprocessing.spatial_sigma_px,config.preprocessing.temporal_ema_span_frames)
        lanes,sample_manifest=fit_lanes(filtered,quiet_count,config)
        sampling_dir=config.output_dir/"sampling"; sampling_dir.mkdir()
        arrays={k:v for k,v in sample_manifest.items() if isinstance(v,np.ndarray)}; atomic_npz(sampling_dir/"primary_samples.npz",**arrays)
        _atomic_json(sampling_dir/"sample_manifest.json",{k:v for k,v in sample_manifest.items() if not isinstance(v,np.ndarray)})
        metrics=[]; all_candidates=[]
        for method_id,lane in lanes.items():
            lane["fit"].update({
                "schema_version": 1,
                "experiment_id": config.experiment_id,
                "method_id": method_id,
                "source_video_sha256": audit["inputs"][0]["sha256"],
                "sample_seed": config.sampling.seed,
                "axes": "TYX",
                "undefined_leading_frames": config.preprocessing.lag_frames,
            })
            objective=lane.get("fit",{}).get("diagnostics",{}).get("objective_by_angle")
            write_lane(config.output_dir,method_id,lane,write_tiff=config.thresholding.write_binary_tiff,objective_rows=objective)
            if lane.get("binary_mask") is None:
                metrics.append({"lane":method_id,"status":lane["fit"]["status"],"binary_mask_written":False}); continue
            result,candidates,maps=evaluate_lane(method_id,lane["binary_mask"],labels,config,binary=True,tie_values=lane["positive_z"])
            metrics.append(result); all_candidates.extend(candidates)
            atomic_npz(config.output_dir/"methods"/method_id/"candidate_maps.npz",**maps)
            _heartbeat(progress,"methods",method_id=method_id,status="complete")
        # Exact external Raw Direct anchor; this is evaluated, never redefined as a pairwise method.
        baseline=np.median(raw[:quiet_count],axis=0); low,high=np.percentile(raw[:quiet_count,::4,::4],[1,99.9]); scale=max(float(high-low),1e-6)
        residual=np.maximum((raw-baseline)/scale,0).astype(np.float32)
        raw_result,raw_candidates,_=evaluate_lane("raw_direct",residual,labels,config,binary=False)
        metrics.insert(0,raw_result); all_candidates.extend(raw_candidates)
        is_frozen_spon=len(labels)==79 and tuple(video.shape)==(2359,340,573)
        valid=not is_frozen_spon or abs(raw_result["mean_recall"]-RAW_EXPECTED)<1e-12
        if not valid: decision="invalid_baseline"
        else:
            nmf=next((lane for lane in lanes.values() if lane.get("fit",{}).get("equivalent_to_adaptive_residual")),None)
            decision="nmf_equivalent_to_adaptive_residual" if nmf else "implementation_only"
        candidates_dir=config.output_dir/"candidates"; candidates_dir.mkdir()
        selected=sorted(all_candidates,key=lambda row:(row["source_stratum"],row["lane"],-row["score"],row["y_px"],row["x_px"]))[:config.evaluation.candidate_review_rows]
        write_candidates(candidates_dir/"candidate_peaks.tsv",all_candidates); write_candidates(candidates_dir/"candidate_review_queue.tsv",selected)
        metrics_payload={"schema_version":1,"experiment_id":config.experiment_id,"status":"complete","lanes":metrics,
            "raw_direct_anchor_expected":RAW_EXPECTED,"raw_direct_anchor_valid":valid,
            "precision_contract":"Known-label candidate fraction is a lower bound only. Unmatched candidates are unknown, not false positives."}
        _atomic_json(config.output_dir/"metrics.json",metrics_payload)
        write_figures(config.output_dir, lanes, metrics)
        summary={"schema_version":1,"experiment_id":config.experiment_id,"status":decision,"implementation_status":"complete",
            "full_spon_run":is_frozen_spon,"raw_direct_mean_recall":raw_result["mean_recall"],"methods":list(lanes),
            "review_queue_rows":len(selected),"elapsed_seconds":time.perf_counter()-started}
        _atomic_json(config.output_dir/"experiment_summary.json",summary)
        report=[f"# {config.experiment_id}","",f"Status: `{decision}`.","","## Validity","",f"Raw Direct mean known-label recall: `{raw_result['mean_recall']:.9f}`; frozen-anchor validity: `{valid}`.","","## Theoretical finding","","InfoMax and bounded CS-Parzen lanes preserve explicit fit and component-selection diagnostics. Unresolved ICA fits omit binary masks by contract.","","## Practical finding","","Fixed and adaptive lanes use spatial Gaussian smoothing, causal EMA, lagged positive difference, fixed quiet MAD calibration, and one-sided binary thresholds.","","## Detection comparison","","Unmatched candidates remain unknown; candidate yield is not precision.","","## NMF finding","",f"Decision status: `{decision}`.","","## Motion sensitivity","","Primary lane is unregistered (`motion_correction: false`); sampled integer-shift diagnostics are in preflight.json.","","## Unknowns","","Sparse labels do not identify exhaustive negatives or guarantee that an activity-like ICA component is neuronal.","","## Decision","",f"`{decision}`. No existing detector is automatically replaced.","","## Next valid experiment","","Review the deterministic candidate queue before any model replacement or expanded parameter search.",""]
        (config.output_dir/"report.md").write_text("\n".join(report),encoding="utf-8")
        _atomic_json(config.output_dir/"run_state.json",{"status":"complete","phase":"complete","elapsed_seconds":summary["elapsed_seconds"]})
        _heartbeat(progress,"complete",status="complete")
        return summary
    except Exception as exc:
        _atomic_json(config.output_dir/"run_state.json",{"status":"failed","phase":"failed","error":repr(exc)})
        raise
