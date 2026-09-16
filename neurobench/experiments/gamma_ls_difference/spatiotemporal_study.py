"""Frozen, resumable 21-setting Gamma CFAR sensitivity experiment.

This runner owns source preparation, label-free calibration and candidate seals.
Synthetic labels and sparse real occurrence labels join only sealed candidates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np

from neurobench.algorithms.gamma_spatiotemporal import build_kernels, describe, spec_grid, iter_chunks
from .control_study import conditioned_frames
from .two_stencil_campaign import sha256, write_json, write_tsv, read_tsv
from .two_stencil_evaluation import extract_frame_candidates, evaluate_occurrence_windows

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "Outputs/GammaLSST/sensitivity_20260914_r2"
PRIOR = REPO / "Outputs/GammaLSControl/control_study_20260912_r1/protocol.json"
REFERENCE_AREA_PX = 340 * 573


def emit(root, **row):
    row = dict(updated_unix=time.time(), **row)
    write_json(root / "progress.json", row)
    print(json.dumps(row), flush=True)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path))


def verify(record):
    if binding(record["path"]) != record:
        raise RuntimeError(f"Source binding changed: {record['path']}")


def preflight(root):
    if root.exists():
        raise FileExistsError(f"Use a new root or the resume commands: {root}")
    prior = json.loads(PRIOR.read_text())
    for key in ("source", "labels"):
        verify(prior[key])
    source = np.load(prior["source"]["path"], mmap_mode="r")
    if source.shape != (2359, 340, 573):
        raise ValueError("Unexpected source geometry")
    kernels = [build_kernels(s) for s in spec_grid()]
    halo = max(k.metadata["spatial_half_width_px"] for k in kernels) + 4
    warmup = max(64, max(k.metadata["maximum_lag_frames"] for k in kernels))
    if shutil.disk_usage(root.parent.parent).free < 150 * 1024**3:
        raise RuntimeError("Require 150 GiB free headroom")
    root.mkdir(parents=True)
    code = [Path(__file__), Path(__file__).parents[2] / "algorithms/gamma_spatiotemporal.py"]
    code += [Path(__file__).with_name(name) for name in ("spatiotemporal_fixtures.py","spatiotemporal_metrics.py","control_study.py","two_stencil_evaluation.py","evaluation.py")]
    code += [REPO/"neurobench/algorithms/gamma_two_stencil.py",REPO/"neurobench/metrics/sparse_detection.py"]
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    protocol = dict(schema_version=1, experiment="gamma_spatiotemporal_sensitivity_20260914",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        initial_git_status=status, source=prior["source"], labels=prior["labels"],
        code_bindings=[binding(p) for p in dict.fromkeys(code)],
        specs=[dict(spec_id=s.spec_id, **asdict(s)) for s in spec_grid()],
        kernels=[describe(k) for k in kernels], common_spatial_halo_px=halo,
        common_synthetic_warmup_frames=warmup, synthetic_setup_frames=100,
        synthetic_application_frames=300, synthetic_scene_count=16,
        real_source_ui=[1400,2359], real_setup_ui=[1600,1799], real_application_ui=[1800,2359],
        real_descriptive_calibration_ui=[1,1799], frame_rate_hz=50, pixel_size_um=.5,
        indicator_bandwidth_reported_hz=30, indicator_identity="unknown",
        bandwidth_interpretation="reported, not measurable above 25 Hz Nyquist; no inverse-bandwidth decay assumption",
        conditioning=dict(gaussian_sigma_px=1., gaussian_truncate=4., ema_alpha=.4, signed_adjacent_difference=True),
        local_standardization="(A-M)/max(reference_std, setup_positive_std_percentile_10, 1e-6)",
        reference_construction="direct, no explicit guard; same input for target and reference moments",
        calibration=dict(nominal_proposals_per_reference_area_frame=1., reference_area_px=REFERENCE_AREA_PX,
            area="common interior minus maintained 6-pixel NMS border", floor_percentile=10., minimum_threshold=0.,
            cutoff="strict setup NMS order statistic, floor(area/194820 * setup_frames) budget; no label use"),
        evaluation=dict(synthetic="exhaustive simulated active source/frame truth; one-to-one 6px assignment",
            real="sparse occurrence-window coverage only; unmatched unknown; no real precision or FPR",
            temporal_grain="framewise active location plus separate event-window recovery and first-detection delay",
            common_crop="full kernel and Gaussian source halo available for every scored pixel",
            freeze="no outcome-driven kernel or threshold selection; within-recording development, no independent test"),
        scientific_audit=dict(enabled=True, scope="every dataset and all21 configurations at setup-frozen area-scaled operating point",
            numeric_scope="all threshold curves including explicit no-output endpoint", all_expert_and_model_rois=True),
        resource_preflight=dict(free_disk_bytes=shutil.disk_usage(root).free,
            cpu_policy="nice19 ionice3, one numerical/codec thread, CPUs25/28/31 only; at most3 audit workers",
            gpu_snapshot="RTX4070SUPER12GiB, used3135MiB, utilization17%; no active compute jobs observed",
            compute="bounded chunks; GPU only after parity/timing validation", preflight_time_unix=time.time()))
    write_json(root / "protocol.json", protocol)
    write_json(root / "preflight.json", dict(status="PASS", protocol_sha256=sha256(root/"protocol.json")))
    emit(root, status="PREFLIGHT_PASS", specs=21, synthetic_scenes=16, halo_px=halo)


def load_protocol(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if sha256(root/"protocol.json") != json.loads((root/"preflight.json").read_text())["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for record in protocol["code_bindings"]:verify(record)
    return protocol


def mmap(path, shape):
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def prepare_case(root, case_id, raw, meta, experts, active):
    folder = root / "datasets" / case_id
    if (folder/"prepared.json").exists():
        prepared=json.loads((folder/"prepared.json").read_text())
        for key in ("source","input","metadata","experts","active"):verify(prepared[key])
        return
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/"raw_source.npy", raw)
    fields = mmap(folder/"input_source.npy", raw.shape)
    for i, (_, delta) in enumerate(conditioned_frames(raw, 1., .4)):
        fields[i] = delta
    fields.flush()
    y0, x0, y1, x1 = meta["evaluation_box_yxyx"]
    np.save(folder/"Raw.npy", np.asarray(raw[:,y0:y1,x0:x1], np.float32))
    np.save(folder/"Input.npy", np.asarray(fields[:,y0:y1,x0:x1]))
    shifted = lambda rows: [dict(r, x_px=float(r["x_px"])-x0, y_px=float(r["y_px"])-y0) for r in rows]
    write_json(folder/"experts.json", shifted(experts))
    write_json(folder/"active.json", shifted(active))
    write_json(folder/"metadata.json", dict(meta, case_id=case_id,
        coordinate_transform="audit/score coordinates = original source minus [x0,y0]",
        original_source_offset_xy=[x0,y0]))
    write_json(folder/"prepared.json", dict(status="PASS", source=binding(folder/"raw_source.npy"),
        input=binding(folder/"input_source.npy"), metadata=binding(folder/"metadata.json"),
        experts=binding(folder/"experts.json"),active=binding(folder/"active.json")))
    emit(root,status="DATASET_PREPARED",case_id=case_id,shape=list(raw.shape))


def prepare(root):
    from .spatiotemporal_fixtures import scene_specs, generate_scene
    p = load_protocol(root)
    for spec in scene_specs():
        scene = generate_scene(spec, setup_frames=100, warmup_frames=p["common_synthetic_warmup_frames"],
            application_frames=300, shape=(128,128), spatial_halo_px=p["common_spatial_halo_px"])
        meta = dict(scene.metadata)
        meta.update(truth_mode="fully_synthetic", source_frames_ui=list(range(1,len(scene.raw)+1)))
        prepare_case(root, spec.scene_id, scene.raw, meta, scene.event_rows, scene.active_rows)
    halo = p["common_spatial_halo_px"]
    full = np.load(p["source"]["path"],mmap_mode="r")
    experts,excluded = [],[]
    for row in read_tsv(p["labels"]["path"]):
        if halo+6 <= float(row["x_px"]) < full.shape[2]-halo-6 and halo+6 <= float(row["y_px"]) < full.shape[1]-halo-6:
            experts.append(row)
        else: excluded.append(row)
    meta=dict(truth_mode="sparse_real", source_frames_ui=list(range(1400,2360)),
        setup_source_frames_ui=list(range(1600,1800)), application_source_frames_ui=list(range(1800,2360)),
        evaluation_box_yxyx=[halo,halo,340-halo,573-halo],
        known_occurrences_before_crop=79, known_occurrences_evaluable=len(experts), excluded_geometry_occurrences=excluded)
    prepare_case(root,"real",full[1399:2359],meta,experts,[])
    write_json(root/"datasets_complete.json",dict(status="PASS",case_count=17))


def frame_sets(meta):
    frames = list(map(int,meta["source_frames_ui"]))
    setup = meta.get("setup_source_frames_ui")
    application = meta.get("application_source_frames_ui")
    if setup is None or application is None:
        # Fixtures freeze warmup first, setup second, application last.
        warmup=int(meta["warmup_frames"]); nsetup=int(meta["setup_frames"])
        setup=frames[warmup:warmup+nsetup]; application=frames[warmup+nsetup:]
    return frames,list(map(int,setup)),list(map(int,application))


def calibrate(scores, indices, frames, eligible_area):
    peaks=[]
    for index,frame in zip(indices,frames):
        peaks.extend(extract_frame_candidates(scores[index],source_frame_ui=frame,threshold_z=0.))
    ranked=sorted((r["score"] for r in peaks),reverse=True)
    budget=math.floor(eligible_area/REFERENCE_AREA_PX*len(indices))
    tau=float(ranked[budget]) if len(ranked)>budget else 0.
    return dict(threshold=tau, threshold_z=tau, setup_proposal_budget=budget,
        setup_proposal_count=sum(v>tau for v in ranked), setup_frame_count=len(indices),
        target_proposals_per_frame=eligible_area/REFERENCE_AREA_PX,
        reference_area_px=REFERENCE_AREA_PX, eligible_area_px=eligible_area,
        nominal_proposals_per_reference_area_frame=1., threshold_frozen_from_calibration_only=True)


def score_case(root, folder, spec, *, chunk_frames=16, device="cpu"):
    from .spatiotemporal_metrics import extract_nms_prefix, seal_candidates
    out=root/"cells"/folder.name/spec.spec_id
    if (out/"sealed.json").exists():
        sealed=json.loads((out/"sealed.json").read_text())
        for record in [*sealed["stages"].values(),sealed["prefix"],sealed["audit_candidates"],sealed["calibration"]]:verify(record)
        return
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/"metadata.json").read_text())
    frames,setup,application=frame_sets(meta); frame_to_i={f:i for i,f in enumerate(frames)}
    source=np.load(folder/"input_source.npy",mmap_mode="r")
    y0,x0,y1,x1=meta["evaluation_box_yxyx"]
    shape=(len(source),y1-y0,x1-x0)
    stages={k:mmap(out/f"{k}.npy",shape) for k in ("A","M","Spread","Score")}
    kernel=build_kernels(spec)
    kwargs=dict(chunk_frames=chunk_frames)
    if device!="cpu":kwargs["device"]=device
    started=time.perf_counter();chunks=0
    for start,stop,values in iter_chunks(source,kernel,**kwargs):
        stages["A"][start:stop]=values["A"][:,y0:y1,x0:x1]
        stages["M"][start:stop]=values["M"][:,y0:y1,x0:x1]
        stages["Spread"][start:stop]=np.sqrt(values["variance"][:,y0:y1,x0:x1])
        chunks+=1
    convolution_seconds=time.perf_counter()-started
    setup_indices=[frame_to_i[f] for f in setup]
    population=np.asarray(stages["Spread"][setup_indices,6:-6,6:-6]).ravel()
    positive=population[population>0]
    floor=max(1e-6,float(np.percentile(positive,10.)) if len(positive) else 0.)
    del population,positive
    for start in range(0,len(source),chunk_frames):
        end=min(len(source),start+chunk_frames)
        stages["Score"][start:end]=(stages["A"][start:end]-stages["M"][start:end])/np.maximum(stages["Spread"][start:end],floor)
    for a in stages.values():a.flush()
    eligible_area=(shape[1]-12)*(shape[2]-12)
    op=calibrate(stages["Score"],setup_indices,setup,eligible_area)
    op.update(scale_floor=floor,application_source_start_ui=application[0],application_source_stop_ui=application[-1],
        case_id=folder.name,spec_id=spec.spec_id,truth_mode=meta["truth_mode"],
        setup_source_frames_ui=setup,application_frame_count=len(application))
    prefix=[]; audit=[]
    for frame in application:
        rows=extract_nms_prefix(stages["Score"][frame_to_i[frame]],source_frame_ui=frame,cell_id=spec.spec_id)
        prefix.extend(rows)
        for row in rows:
            if row["score"]>op["threshold"]:
                audit.append(dict(row,threshold_z=op["threshold"],target_proposals_per_frame=op["target_proposals_per_frame"]))
    seal_candidates(prefix,source_frames_ui=application)
    write_json(out/"prefix.json",prefix);write_json(out/"audit_candidates.json",audit)
    write_tsv(out/"audit_candidates.tsv",audit)
    write_json(out/"calibration.json",op)
    app_indices=[frame_to_i[f] for f in application]
    performance=dict(convolution_seconds=convolution_seconds,convolution_ms_per_frame=1000*convolution_seconds/len(source),
        batch_throughput_only=True,chunk_frames=chunk_frames,device=device,chunks=chunks,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        minimum_input_ring_bytes=int((kernel.metadata["maximum_lag_frames"]+1)*source.shape[1]*source.shape[2]*source.dtype.itemsize),
        setup_floor_fraction=float(np.mean(stages["Spread"][setup_indices,6:-6,6:-6]<floor)),
        application_floor_fraction=float(np.mean(stages["Spread"][app_indices,6:-6,6:-6]<floor)))
    write_json(out/"performance.json",performance)
    records={"Raw":binding(folder/"Raw.npy"),"Input":binding(folder/"Input.npy")}
    records.update({k:binding(out/f"{k}.npy") for k in stages})
    write_json(out/"sealed.json",dict(status="SEALED_BEFORE_LABEL_JOIN",stages=records,
        prefix=binding(out/"prefix.json"),audit_candidates=binding(out/"audit_candidates.json"),
        calibration=binding(out/"calibration.json"),kernel=describe(kernel)))
    emit(root,status="CELL_SEALED",case_id=folder.name,spec_id=spec.spec_id,
        seconds=round(convolution_seconds,2),audit_candidates=len(audit),threshold=op["threshold"])


def run(root,device="cpu",chunk_frames=16,limit=None):
    protocol=load_protocol(root)
    for record in protocol["code_bindings"]:verify(record)
    count=0
    for folder in sorted((root/"datasets").iterdir()):
        prepared=json.loads((folder/"prepared.json").read_text())
        for key in ("input","metadata"):verify(prepared[key])
        for spec in spec_grid():
            score_case(root,folder,spec,chunk_frames=chunk_frames,device=device)
            count+=1
            if limit and count>=limit:return
    write_json(root/"computation_complete.json",dict(status="PASS",cell_count=count))


def evaluate(root):
    from .spatiotemporal_metrics import seal_candidates,evaluate_threshold_sweep,evaluate_framewise
    protocol=load_protocol(root)
    for record in protocol["code_bindings"]:verify(record)
    folders=sorted((root/"datasets").iterdir())
    expected=[root/"cells"/folder.name/spec.spec_id/"sealed.json" for folder in folders for spec in spec_grid()]
    if len(folders)!=17 or len(expected)!=357 or not all(path.exists() for path in expected):
        raise RuntimeError("Seal all357 matrix cells before any evaluation truth join")
    all_curves=[]; operating=[]
    for folder in folders:
        prepared=json.loads((folder/"prepared.json").read_text())
        for key in ("experts","active","metadata"):verify(prepared[key])
        meta=json.loads((folder/"metadata.json").read_text());_,_,application=frame_sets(meta)
        experts=json.loads((folder/"experts.json").read_text());active=json.loads((folder/"active.json").read_text())
        for spec in spec_grid():
            out=root/"cells"/folder.name/spec.spec_id
            if not (out/"sealed.json").exists():continue
            if not (out/"evaluated.json").exists():
                seal=json.loads((out/"sealed.json").read_text());verify(seal["prefix"]);verify(seal["calibration"])
                rows=json.loads((out/"prefix.json").read_text());op=json.loads((out/"calibration.json").read_text())
                stream=seal_candidates(rows,source_frames_ui=application)
                if meta["truth_mode"]=="fully_synthetic":
                    result=evaluate_threshold_sweep(stream,active,experts)
                    fixed=evaluate_framewise(stream,active,experts,threshold_z=op["threshold"])
                    write_json(out/"operating_metrics.json",fixed)
                    write_json(out/"curves.json",result)
                else:
                    from .spatiotemporal_metrics import threshold_grid
                    curves=[];event_rows=[]
                    intervals={int(r["burst_id"]):(int(r["source_start_ui"]),int(r["source_stop_ui"])) for r in experts}
                    thresholds=threshold_grid()
                    for threshold_index,tau in enumerate((*thresholds,op["threshold"])):
                        selected=[dict(r,target_proposals_per_frame=None) for r in rows if r["score"]>tau]
                        result=evaluate_occurrence_windows(selected,experts,burst_intervals_ui=intervals)
                        row=dict(threshold_z=tau if math.isfinite(tau) else None,threshold_label=str(tau) if math.isfinite(tau) else "no_output",
                            precision=None,framewise_sensitivity=None,false_positive_count=None,proposal_count=len(selected),
                            proposals_per_frame=len(selected)/len(application),truth_mode="sparse_real",
                            occurrence_count=len(experts),**result["summary"])
                        if threshold_index==len(thresholds):
                            write_json(out/"operating_metrics.json",dict(summary=row,occurrence_rows=result["occurrence_rows"],site_rows=result["site_rows"]))
                        else:
                            curves.append(row)
                            event_rows.extend(dict(r,threshold_z=row["threshold_z"],threshold_label=row["threshold_label"]) for r in result["occurrence_rows"])
                    write_json(out/"curves.json",dict(curve_rows=curves,event_rows=event_rows))
                write_json(out/"evaluated.json",dict(status="PASS",candidate_seal_sha256=sha256(out/"sealed.json")))
            result=json.loads((out/"curves.json").read_text())
            op=json.loads((out/"calibration.json").read_text())
            for row in result["curve_rows"]:
                row=dict(row,case_id=folder.name,scene_id=folder.name,template_id=meta.get("template_id","real"),seed=meta.get("seed",0),
                    spec_id=spec.spec_id,**asdict(spec),eligible_area_px=op["eligible_area_px"])
                row["false_proposals_per_10000_um2_s"]=(row["false_proposals_per_second"]/(op["eligible_area_px"]*.25)*10000
                    if row.get("false_proposals_per_second") is not None else None)
                all_curves.append(row)
            fixed=json.loads((out/"operating_metrics.json").read_text())
            operating.append({**asdict(spec),**op,"case_id":folder.name,"spec_id":spec.spec_id,"metrics":fixed,
                "performance":json.loads((out/"performance.json").read_text())})
            emit(root,status="CELL_EVALUATED",case_id=folder.name,spec_id=spec.spec_id)
    write_json(root/"all_curves.json",all_curves);write_tsv(root/"all_curves.tsv",all_curves)
    write_json(root/"operating_points.json",operating)
    write_json(root/"evaluation_complete.json",dict(status="PASS",evaluated_cells=len(operating)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("preflight","prepare","run","evaluate"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--chunk-frames",type=int,default=16)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=="run":run(root,device=args.device,chunk_frames=args.chunk_frames,limit=args.limit)
    else:globals()[args.command](root)


if __name__=="__main__":main()
