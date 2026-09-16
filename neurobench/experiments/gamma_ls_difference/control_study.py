"""Chronological development bridge and component study for fish observations.

Calibration and scores never use evaluation labels. Known-window ranking and
configured expert-centre traces are explicitly retrospective diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.ndimage import gaussian_filter, shift

from neurobench.algorithms.gamma_two_stencil import TwoStencilSpec, two_stencil_local_standardization
from neurobench.metrics.sparse_detection import extract_local_maxima, match_peaks_one_to_one
from .evaluation import strict_separated_nms
from .two_stencil_campaign import REPO, SOURCE_ROOT, LABEL_PATH, sha256, read_tsv, write_tsv, write_json, progress
from .two_stencil_evaluation import extract_frame_candidates, evaluate_occurrence_windows, NMS_SEMANTICS
from .two_stencil_exact_calibration import calibrate_tau_exact_nms

START, STOP, CALIBRATION_COUNT = 1800, 2359, 100
TARGETS = (0.25, 0.5, 1.0, 2.0, 5.0)


def study_arms():
    arms = []
    for sigma in (0, 1):
        for alpha in (1.0, 0.4):
            for representation in ("current", "difference"):
                source = f"s{sigma}_a{alpha:g}_{representation}"
                readouts = ("A", "contrast", "Z") if sigma == 1 and alpha == .4 else ("A", "Z")
                for readout in readouts:
                    arms.append(dict(arm_id=f"{source}_{readout}", source=source, sigma=sigma, alpha=alpha,
                                     representation=representation, correction="none", design="point",
                                     reference="gamma", guard=0, geometry="square", readout=readout))
    for name in ("legacy_residual", "legacy_carrier"):
        arms.append(dict(arm_id=name, source=name, readout="native", historical=True))
    base = dict(source="s1_a0.4_difference", sigma=1, alpha=.4, representation="difference",
                correction="none", reference="gamma", guard=0, geometry="square")
    for design, readouts in (("direct", ("A", "contrast", "Z")), ("serial", ("Z",))):
        for readout in readouts:
            arms.append(dict(base, arm_id=f"centered_{design}_{readout}", design=design, readout=readout))
    arms.append(dict(base, arm_id="uniform_reference_Z", design="point", readout="Z", reference="uniform"))
    arms.append(dict(base, arm_id="guard7_disk_Z", design="point", readout="Z", guard=7, geometry="disk"))
    for correction in ("motion", "global_offset"):
        arms.append(dict(base, arm_id=f"{correction}_Z", source=f"{correction}_difference",
                         design="point", readout="Z", correction=correction))
    assert len(arms) == len({x["arm_id"] for x in arms}) == 28
    return arms


def conditioned_frames(raw, sigma, alpha):
    """Causal float32 recurrence; Gaussian is spatial-only SciPy reflect."""
    previous = None
    for frame in raw:
        spatial = np.asarray(frame, dtype=np.float32)
        if sigma:
            spatial = gaussian_filter(spatial, sigma=float(sigma), mode="reflect", truncate=4.0)
        current = spatial.copy() if previous is None else np.float32(alpha) * spatial + np.float32(1-alpha) * previous
        difference = np.zeros_like(current) if previous is None else current - previous
        yield current, difference
        previous = current


def rigid_shift(frame, template, maximum_shift=8):
    """Bounded integer phase-correlation translation against a fixed template."""
    a = np.asarray(template, dtype=np.float64)
    b = np.asarray(frame, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    if not a.any() or not b.any():
        return (0, 0)
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    offsets = range(-maximum_shift, maximum_shift + 1)
    return max(((dy, dx) for dy in offsets for dx in offsets),
               key=lambda d: (correlation[d[0] % a.shape[0], d[1] % a.shape[1]],
                              -abs(d[0])-abs(d[1]), -d[0], -d[1]))


def corrected_frames(raw, template, correction, transformations):
    for index, frame in enumerate(raw):
        values = np.asarray(frame, dtype=np.float32)
        if correction == "motion":
            dy, dx = rigid_shift(values, template)
            values = shift(values, (dy, dx), order=1, mode="nearest", prefilter=False)
            transformations.append(dict(source_frame_ui=START+index, dy_px=dy, dx_px=dx))
        elif correction == "global_offset":
            offset = float(np.median(values-template))
            values = values-offset
            transformations.append(dict(source_frame_ui=START+index, offset=offset))
        elif correction != "none":
            raise ValueError("unknown correction")
        yield values


def source_coordinate_frames(values, transformations):
    """Undo fitted rigid translations so scores and frozen labels share coordinates.

    Newly uncovered border pixels are zero, and never extrapolated as signal.
    Integer transforms preserve all overlapping source values exactly.
    """
    if len(values) != len(transformations):raise ValueError("Motion transform length mismatch")
    for frame,record in zip(values,transformations):
        yield shift(frame,(-int(record["dy_px"]),-int(record["dx_px"])),order=0,
                    mode="constant",cval=0,prefilter=False)


def binding(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=sha256(path), size_bytes=path.stat().st_size)


def verify(record):
    path = Path(record["path"])
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Changed sealed artifact: {path}")


def load_protocol(root):
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["arms"] != study_arms():
        raise ValueError("Current matrix differs from frozen protocol")
    for record in protocol["code_bindings"]:
        verify(record)
    return protocol


def preflight(root):
    if root.exists():
        raise FileExistsError(root)
    if shutil.disk_usage(REPO).free < 200 * 1024**3:
        raise RuntimeError("Less than200GiB free for stages and complete audits")
    old = json.loads((SOURCE_ROOT/"stage_manifest.json").read_text())
    raw_path = Path(old["source_movie"]["path"])
    if sha256(raw_path) != old["source_movie"]["file_sha256"]:
        raise ValueError("Source movie hash mismatch")
    raw = np.load(raw_path, mmap_mode="r")
    if list(raw.shape) != [2359,340,573] or str(raw.dtype) != "uint16":
        raise ValueError("Unexpected source shape/dtype")
    labels = read_tsv(LABEL_PATH)
    if len(labels) != 79 or len({x["canonical_roi_id"] for x in labels}) != 26:
        raise ValueError("Unexpected frozen cohort")
    windows = {int(x["burst_id"]): [int(x["source_start_ui"]),int(x["source_stop_ui"])] for x in labels}
    for row in labels:
        if not (0 <= float(row["x_px"]) < raw.shape[2] and 0 <= float(row["y_px"]) < raw.shape[1]):
            raise ValueError("Expert coordinate outside source image")
        if not (1900 <= int(row["source_start_ui"]) <= int(row["source_stop_ui"]) <= STOP):
            raise ValueError("Expert interval outside application")
        if windows[int(row["burst_id"])] != [int(row["source_start_ui"]), int(row["source_stop_ui"])]:
            raise ValueError("Inconsistent shared burst interval")
    root.mkdir(parents=True)
    shutil.copy2(LABEL_PATH, root/"expert_occurrences.tsv")
    files = [Path(__file__), Path(__file__).with_name("control_history.py"),
             Path(__file__).with_name("control_audit.py"), Path(__file__).with_name("two_stencil_audit.py"),
             Path(__file__).with_name("two_stencil_exact_calibration.py"), Path(__file__).with_name("two_stencil_evaluation.py"),
             Path(__file__).with_name("evaluation.py"), REPO/"neurobench/metrics/sparse_detection.py",
             REPO/"neurobench/algorithms/gamma_two_stencil.py"]
    from .control_history import SOURCE_DEPENDENCIES
    files = list(dict.fromkeys(files + list(SOURCE_DEPENDENCIES) + [
        Path(__file__).with_name("two_stencil_campaign.py"),
        REPO/"neurobench/experiments/hierarchical_parzen_ica/patch_information_program.py",
        REPO/"docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md",
        REPO/"docs/workflows/gamma_ls_control_study.md",
    ]))
    protocol = dict(schema_version=1, experiment="gamma_control_development_20260912", arms=study_arms(),
        source=binding(raw_path), code_bindings=[binding(p) for p in files],
        labels=binding(root/"expert_occurrences.tsv"), source_frames_ui=[START,STOP],
        calibration_input_ui=[1800,1899], calibration_score_ui=[1801,1899], application_ui=[1900,2359],
        burst_windows=windows, target_proposals_per_frame=list(TARGETS), frame_interval_ms=20,
        common_initialization="All recomputed frontends initialize atUI1800; calibration input prefix100frames",
        fitting_scope="calibration prefix only; evaluation coordinates used only in subsequent joins",
        readouts={"online":"current maintained spatialNMS with calibration-only exact thresholds",
          "retrospective":"quiet-normalized positiveLME0.25 over complete knownburst; top58; legacy and separatedNMS",
          "configured_rois":"retrospective known-centre trace diagnostic, not automatic ROI setup or onset truth"},
        scientific_audit=dict(enabled=True, operating_point=1, arms="all28", all_experts_and_model_sites=True),
        claim_boundary="within-recording development; no precision, neural onset, independent confirmation, or controller validation",
        source_cohort_reused=True, nms=NMS_SEMANTICS,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip())
    write_json(root/"protocol.json", protocol)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image=np.max(raw[START-1:STOP:5],axis=0)
    fig,ax=plt.subplots(figsize=(9,6)); ax.imshow(image,cmap="gray",vmin=np.percentile(image,1),vmax=np.percentile(image,99.5))
    ax.scatter([float(x["x_px"]) for x in labels],[float(x["y_px"]) for x in labels],facecolors="none",edgecolors="#46dc7d",s=25)
    ax.set_title("Geometry check: frozen79occurrences /26identities; no detector outcome")
    fig.savefig(root/"preflight_geometry.png",dpi=130);plt.close(fig)
    write_json(root/"preflight.json",dict(status="PASS", source_shape=list(raw.shape), code_count=len(files),
          free_disk_gib=shutil.disk_usage(root).free/1024**3, cpu_affinity=sorted(os.sched_getaffinity(0)), labels_geometry_only=True))
    progress(root,status="PREFLIGHT_PASS", arms=28)


def array(path, shape):
    path.parent.mkdir(parents=True,exist_ok=True)
    return np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=shape)


def generate_inputs(root, protocol):
    done=root/"inputs_complete.json"
    if done.exists():
        records=json.loads(done.read_text())
        for record in records["files"].values(): verify(record)
        verify(records["calibration_template"])
        for record in records["auxiliary"]:verify(record)
        return records
    verify(protocol["source"])
    source=np.load(protocol["source"]["path"],mmap_mode="r")[START-1:STOP]
    raw=array(root/"inputs/Raw.npy",source.shape)
    raw[:]=source;raw.flush()
    template=np.median(np.asarray(raw[:100]),axis=0).astype(np.float32)
    np.save(root/"inputs/calibration_template.npy",template)
    records={"Raw":binding(root/"inputs/Raw.npy")}
    for sigma in (0,1):
        for alpha in (1.0,.4):
            names=[f"s{sigma}_a{alpha:g}_{x}" for x in ("current","difference")]
            outputs=[array(root/f"inputs/{x}.npy",raw.shape) for x in names]
            for index,pair in enumerate(conditioned_frames(raw,sigma,alpha)):
                for out,value in zip(outputs,pair):out[index]=value
            for name,out in zip(names,outputs):out.flush();records[name]=binding(root/f"inputs/{name}.npy")
            progress(root,status="GENERATING_INPUTS",source=names[0])
    for correction in ("motion","global_offset"):
        transformations=[]
        out=array(root/f"inputs/{correction}_difference.npy",raw.shape)
        stream=corrected_frames(raw,template,correction,transformations)
        for index,(_,difference) in enumerate(conditioned_frames(stream,1,.4)):out[index]=difference
        out.flush();records[f"{correction}_difference"]=binding(root/f"inputs/{correction}_difference.npy")
        write_tsv(root/f"inputs/{correction}_transforms.tsv",transformations)
    from .control_history import build_historical_stages
    history_path=root/"history/metadata.json"
    if history_path.exists():
        history=json.loads(history_path.read_text())
        for item in history["files"].values():
            if sha256(root/"history"/item["path"]) != item["sha256"]:
                raise ValueError("Changed historical stage during resume")
        for name,digest in history["source_sha256"].items():
            if sha256(REPO/name) != digest:raise ValueError("Changed historical source during resume")
        # Exact quiet-input bytes also protect resume against a different setup.
        from .control_history import _array_sha256
        if _array_sha256(raw[:100]) != history["input"]["quiet_logical_c_order_bytes_sha256"]:
            raise ValueError("Changed historical calibration input")
        if _array_sha256(raw) != history["input"]["logical_c_order_bytes_sha256"]:
            raise ValueError("Changed historical full input")
    else:
        history=build_historical_stages(raw,root/"history",frame_interval_s=.02)
    write_json(root/"history_metadata.json",history)
    # Adapter's public metadata is reconciled explicitly by stage filename.
    for name,filename in (("legacy_residual","raw_residual.npy"),("legacy_carrier","carrier_signed.npy")):
        path=root/"history"/filename
        if not path.is_file():raise FileNotFoundError(path)
        records[name]=binding(path)
    result=dict(files=records,calibration_template=binding(root/"inputs/calibration_template.npy"),
                auxiliary=[binding(root/p) for p in ("inputs/motion_transforms.tsv", "inputs/global_offset_transforms.tsv",
                    "history/metadata.json", "history/calibration.npz", "history_metadata.json")])
    write_json(done,result)
    return result


def spatial_group(arm):
    return "__".join(str(arm[k]) for k in ("source","design","reference","guard","geometry"))


def generate_spatial(root, arm, input_record, device, chunk_frames):
    folder=root/"operators"/spatial_group(arm); done=folder/"complete.json"
    if done.exists():
        result=json.loads(done.read_text())
        if result["source"] != input_record:raise ValueError("Changed spatial input binding")
        for rec in result["stages"].values():verify(rec)
        verify(result["displayed_input"])
        return result
    folder.mkdir(parents=True,exist_ok=True)
    values=np.load(input_record["path"],mmap_mode="r")
    outputs={key:array(folder/f"{key}.npy",values.shape) for key in ("A","M","sigma","contrast","Z")}
    spec=TwoStencilSpec(design=arm["design"],reference_family=arm["reference"],guard_radius_px=arm["guard"],support_geometry=arm["geometry"])
    names={"A":"target_response","M":"reference_mean","sigma":"reference_std","contrast":"contrast"}
    elapsed=[]
    transforms=read_tsv(root/"inputs/motion_transforms.tsv") if arm.get("correction")=="motion" else None
    import torch
    if device=="cuda":torch.cuda.synchronize()
    for first in range(0,len(values),chunk_frames):
        started=time.perf_counter()
        result=two_stencil_local_standardization(np.array(values[first:first+chunk_frames]),spec,device=device,return_stages=True)
        for name,attribute in names.items():
            stage=getattr(result,attribute).cpu().numpy()
            if transforms is not None:stage=np.stack(list(source_coordinate_frames(stage,transforms[first:first+len(stage)])))
            outputs[name][first:first+chunk_frames]=stage
        if device=="cuda":torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-started)
        del result
    sigma=np.asarray(outputs["sigma"][1:100]);positive=sigma[sigma>0]
    floor=float(np.percentile(positive,10)) if len(positive) else 0.0
    for first in range(0,len(values),chunk_frames):
        outputs["Z"][first:first+chunk_frames]=outputs["contrast"][first:first+chunk_frames]/(np.maximum(outputs["sigma"][first:first+chunk_frames],floor)+spec.epsilon)
    for output in outputs.values():output.flush()
    displayed_input=input_record
    if transforms is not None:
        source_input=array(folder/"Input_source_coordinates.npy",values.shape)
        for index,frame in enumerate(source_coordinate_frames(values,transforms)):source_input[index]=frame
        source_input.flush();displayed_input=binding(folder/"Input_source_coordinates.npy")
    result=dict(spec=asdict(spec),scale_floor=floor,stages={k:binding(folder/f"{k}.npy") for k in outputs},
       timing_scope="offline development batch scoring inclD2H/stagewrites; not paced latency",device=device,
       chunk_frames=chunk_frames,elapsed_s=sum(elapsed),source=input_record,displayed_input=displayed_input,
       coordinate_system="original source frame; motion stages inverse-translated with zero uncovered border")
    write_json(done,result)
    return result


def score_arm(root, arm, inputs, device, chunk_frames):
    folder=root/"arms"/arm["arm_id"];done=folder/"sealed.json"
    if done.exists():
        seal=json.loads(done.read_text())
        for rec in seal["files"]:verify(rec)
        for rec in seal["stages"].values():verify(rec)
        return seal
    folder.mkdir(parents=True,exist_ok=True)
    input_record=inputs["files"][arm["source"]]
    if arm.get("historical"):
        stages={"Input":inputs["files"]["legacy_residual"],"Score":input_record}
        floor=None
    else:
        spatial=generate_spatial(root,arm,input_record,device,chunk_frames)
        stages={"Input":spatial["displayed_input"],"Mean":spatial["stages"]["M"],"Score":spatial["stages"][arm["readout"]]}
        floor=spatial["scale_floor"]
    stages={"Raw":inputs["files"]["Raw"],**stages}
    scores=np.load(stages["Score"]["path"],mmap_mode="r")
    calibration=calibrate_tau_exact_nms(scores[1:100],source_frames_ui=list(range(1801,1900)))
    write_json(folder/"calibration.json",asdict(calibration))
    thresholds={q:calibration.threshold_for(q) for q in TARGETS}
    proposals={q:[] for q in TARGETS}
    for index in range(100,len(scores)):
        prefix=extract_frame_candidates(scores[index],source_frame_ui=START+index,threshold_z=min(thresholds.values()),
              cell_id=arm["arm_id"],target_proposals_per_frame=5.0,frame_interval_ms=20.0)
        for q,tau in thresholds.items():
            for row in prefix:
                if row["score"]>tau:
                    proposals[q].append(dict(row,proposal_id=row["proposal_id"].replace("__q5__",f"__q{q:g}__"),
                                           threshold_z=tau,target_proposals_per_frame=q))
    files=[binding(folder/"calibration.json")]
    for q,rows in proposals.items():
        path=folder/f"q{q:g}/candidates.tsv";path.parent.mkdir(exist_ok=True)
        # Empty tables still carry their schema.
        if rows:write_tsv(path,rows)
        else:path.write_text("proposal_id\tcell_id\ttarget_proposals_per_frame\tthreshold_z\tsource_frame_ui\tx_px\ty_px\tscore\n")
        files.append(binding(path))
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_program import _pool_values
    protocol=json.loads((root/"protocol.json").read_text())
    events={int(b):scores[int(bounds[0])-START:int(bounds[1])-START+1] for b,bounds in protocol["burst_windows"].items()}
    pooled=_pool_values(scores[:100],events,.25)
    bridge=[]
    for burst,score in pooled["events"].items():
        np.save(folder/f"burst_{burst}_retrospective.npy",score)
        files.append(binding(folder/f"burst_{burst}_retrospective.npy"))
        for rule in ("legacy_maxfilter","separated_nms"):
            peaks=extract_local_maxima(score,6,limit=58) if rule=="legacy_maxfilter" else strict_separated_nms(score,threshold=-np.finfo(np.float64).max,limit=58)
            bridge.extend(dict(arm_id=arm["arm_id"],rule=rule,burst_id=burst,rank=rank,score=value,x_px=x,y_px=y)
                          for rank,(value,x,y) in enumerate(peaks,1))
    write_tsv(folder/"retrospective_candidates.tsv",bridge);files.append(binding(folder/"retrospective_candidates.tsv"))
    seal=dict(arm=arm,stages=stages,scale_floor=floor,files=files,
         application_ui=[1900,2359],calibration_score_ui=[1801,1899],labels_used_in_scoring=False,
         retrospective_pool_uses_known_windows=True,retrospective_global_scale=pooled["normalization_scale"],
         sealed_unix=time.time())
    write_json(done,seal)
    progress(root,status="ARM_SEALED",arm=arm["arm_id"],proposals_q1=len(proposals[1.0]))
    return seal


def run(root,device="cpu",chunk_frames=8):
    protocol=load_protocol(root)
    import torch
    torch.set_num_threads(1)
    if device=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("Requested CUDA unavailable")
        free,total=torch.cuda.mem_get_info()
        if free<4*1024**3:raise RuntimeError("Less than4GiB free GPUmemory")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=generate_inputs(root,protocol)
    for arm in protocol["arms"]:score_arm(root,arm,inputs,device,chunk_frames)
    write_json(root/"candidate_seal.json",dict(status="ALL_ARMS_SEALED_BEFORE_OUTCOME_JOIN",arms=28,
           seals=[binding(root/"arms"/a["arm_id"]/"sealed.json") for a in protocol["arms"]]))
    progress(root,status="SCORING_COMPLETE",arms=28)


def evaluate(root):
    protocol=load_protocol(root)
    allseal=json.loads((root/"candidate_seal.json").read_text())
    for rec in allseal["seals"]:verify(rec)
    labels=read_tsv(root/"expert_occurrences.tsv");verify(protocol["labels"])
    windows={int(k):v for k,v in protocol["burst_windows"].items()}
    summaries=[];bridges=[];diagnostics=[];tracking=[]
    for arm in protocol["arms"]:
        folder=root/"arms"/arm["arm_id"];seal=json.loads((folder/"sealed.json").read_text())
        for rec in seal["files"]:verify(rec)
        for rec in seal["stages"].values():verify(rec)
        scores=np.load(seal["stages"]["Score"]["path"],mmap_mode="r")
        calibration=json.loads((folder/"calibration.json").read_text())
        cutoffs={float(x["target_proposals_per_frame"]):float(x["threshold_z"]) for x in calibration["operating_points"]}
        for q in TARGETS:
            candidates=read_tsv(folder/f"q{q:g}/candidates.tsv")
            result=evaluate_occurrence_windows(candidates,labels,burst_intervals_ui=windows)
            for key in ("occurrence_rows","site_rows","membership_rows","burst_summaries"):
                write_tsv(folder/f"q{q:g}/{key}.tsv",result[key])
            write_json(folder/f"q{q:g}/summary.json",result["summary"])
            summaries.append(dict(arm_id=arm["arm_id"],q=q,**result["summary"]))
            if q==1:
                tau=cutoffs[q]
                outcomes={x["observation_id"]:x for x in result["occurrence_rows"]}
                yy,xx=np.indices(scores.shape[1:])
                for label in labels:
                    x,y=float(label["x_px"]),float(label["y_px"])
                    disk=(xx-x)**2+(yy-y)**2<=36
                    lo,hi=windows[int(label["burst_id"])]
                    peak=float(np.max(scores[lo-START:hi-START+1,disk]))
                    matched=outcomes[label["observation_id"]]["matched"]
                    category="matched" if matched else ("no_local_exceedance" if peak<=tau else "exceedance_unmatched")
                    diagnostics.append(dict(arm_id=arm["arm_id"],observation_id=label["observation_id"],category=category,
                                            local_peak=peak,threshold=tau,margin=peak-tau))
        retrospective=read_tsv(folder/"retrospective_candidates.tsv")
        for rule in ("legacy_maxfilter","separated_nms"):
            total=0
            for burst in windows:
                positives=sorted([x for x in labels if int(x["burst_id"])==burst],key=lambda x:x["observation_id"])
                peaks=[(float(x["score"]),int(x["x_px"]),int(x["y_px"])) for x in retrospective if x["rule"]==rule and int(x["burst_id"])==burst]
                matches,_=match_peaks_one_to_one(peaks,positives,6);total+=len(matches)
                bridges.append(dict(arm_id=arm["arm_id"],rule=rule,burst_id=burst,matched=len(matches),known=len(positives),proposals=len(peaks)))
            bridges.append(dict(arm_id=arm["arm_id"],rule=rule,burst_id="all",matched=total,known=79,proposals=sum(1 for x in retrospective if x["rule"]==rule)))
        # Configured-region branch: each unique coordinate is frozen from geometry,
        # without using subsequent activity outcomes to fit its score threshold.
        unique={}
        for label in labels:unique.setdefault(label["canonical_roi_id"],label)
        trace_rows=[]
        trace_cache={}
        for identity,label in unique.items():
            x,y=int(np.floor(float(label["x_px"])+.5)),int(np.floor(float(label["y_px"])+.5))
            values=np.asarray(scores[:,y,x],dtype=np.float64)
            tau=float(np.quantile(values[1:100],.99))
            trace_cache[identity]=(values,tau,x,y)
            trace_rows.extend(dict(canonical_roi_id=identity,source_frame_ui=START+i,x_px=x,y_px=y,score=float(v),
                                   configured_roi_threshold=tau,above_threshold=bool(v>tau),application=i>=100) for i,v in enumerate(values))
        write_tsv(folder/"configured_roi_traces.tsv",trace_rows)
        for label in labels:
            values,tau,x,y=trace_cache[label["canonical_roi_id"]]
            lo,hi=windows[int(label["burst_id"])]
            segment=values[lo-START:hi-START+1]
            hits=np.flatnonzero(segment>tau)
            tracking.append(dict(arm_id=arm["arm_id"],observation_id=label["observation_id"],configured_x_px=x,configured_y_px=y,
                 threshold=tau,known_interval_has_exceedance=bool(len(hits)),fraction_frames_above=float(np.mean(segment>tau)),
                 first_exceedance_source_ui=int(lo+hits[0]) if len(hits) else "",onset_latency_identified=False,
                 geometry_role="retrospectively_configured_known_center_not_automatic_discovery"))
        progress(root,status="EVALUATED",arm=arm["arm_id"])
    write_tsv(root/"online_summary.tsv",summaries);write_tsv(root/"retrospective_bridge.tsv",bridges)
    write_tsv(root/"q1_stage_diagnostics.tsv",diagnostics);write_tsv(root/"configured_roi_summary.tsv",tracking)
    write_json(root/"evaluation_complete.json",dict(status="NUMERICAL_PASS",arm_count=28,online_operating_points=len(summaries),
       diagnostic_occurrences=len(diagnostics),tracking_occurrences=len(tracking),precision=False,onset=False,independent_confirmation=False))


def display_group(arm, stage):
    if stage=="Raw":return "raw_fluorescence",False
    if arm.get("historical"):
        return ("historical_quiet_standardized",True) if stage=="Score" and arm["arm_id"]=="legacy_carrier" else ("historical_residual",True)
    representation=arm["representation"]
    if stage=="Score" and arm["readout"]=="Z":return "spatial_standardized",True
    if stage=="Score" and arm["readout"]=="contrast":return f"local_contrast_{representation}",True
    return f"native_{representation}",representation=="difference"


def audit_display_contract(root, arms):
    """One stable intensity range per unit/representation across all arms.

    Each unique array supplies a deterministic unlabeled sample. The shared
    range encloses all array-wise percentile ranges, avoiding run-time scales.
    """
    groups={}; seen=set(); arm_groups={}
    for arm in arms:
        seal=json.loads((root/"arms"/arm["arm_id"]/"sealed.json").read_text())
        arm_groups[arm["arm_id"]]={}
        for stage,record in seal["stages"].items():
            group,signed=display_group(arm,stage)
            arm_groups[arm["arm_id"]][stage]=group
            token=(group,record["sha256"])
            if token in seen:continue
            seen.add(token);verify(record)
            values=np.load(record["path"],mmap_mode="r")
            sample=np.asarray(values[::10,::4,::4],dtype=np.float32)
            if signed:
                bound=max(float(np.quantile(np.abs(sample),.995)),1e-6);lo,hi=-bound,bound
            else:
                lo,hi=map(float,np.percentile(sample,[1,99.5]));hi=max(hi,lo+1e-6)
            previous=groups.get(group,dict(limits=[lo,hi],signed=signed))
            groups[group]=dict(limits=[min(previous["limits"][0],lo),max(previous["limits"][1],hi)],signed=signed)
    contract=dict(groups=groups,arm_groups=arm_groups,labels_used=False,
                  rule="envelope of deterministic per-array percentiles; fixed across like-unit/representation panels")
    path=root/"audit_display_contract.json"
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError("Changed audit display contract")
    write_json(path,contract)
    return contract


def ordered_stage_paths(stages):
    """Restore semantic panel order after canonical JSON sorted mapping keys."""
    order=["Raw","Input",*(k for k in stages if k not in {"Raw","Input","Score"}),"Score"]
    return {name:stages[name]["path"] for name in order}


def audit(root):
    from .control_audit import run_control_audit
    protocol=load_protocol(root)
    if not (root/"evaluation_complete.json").exists():raise RuntimeError("Evaluate sealed candidates first")
    contract=audit_display_contract(root,protocol["arms"])
    summaries=[]
    for arm in protocol["arms"]:
        folder=root/"arms"/arm["arm_id"];seal=json.loads((folder/"sealed.json").read_text())
        display={};signed=[]
        for name,group in contract["arm_groups"][arm["arm_id"]].items():
            setting=contract["groups"][group];display[name]=setting["limits"]
            if setting["signed"]:signed.append(name)
        cal=json.loads((folder/"calibration.json").read_text());op=next(x for x in cal["operating_points"] if x["target_proposals_per_frame"]==1)
        point=dict(op,arm_id=arm["arm_id"],readout_name=arm["readout"],score_semantics=str(arm),
                   application_source_start_ui=1900,application_source_stop_ui=2359)
        sources=dict(stage_sha256={k:v["sha256"] for k,v in seal["stages"].items()},candidates_sha256=sha256(folder/"q1/candidates.tsv"),
                     expert_occurrences_sha256=sha256(root/"expert_occurrences.tsv"),candidate_seal_sha256=sha256(folder/"sealed.json"),
                     display_contract_sha256=sha256(root/"audit_display_contract.json"))
        result=run_control_audit(root/"audits"/arm["arm_id"],stage_paths=ordered_stage_paths(seal["stages"]),
                 display_limits=display,signed_stage_keys=signed,source_frames_ui=list(range(START,STOP+1)),
                 candidates_path=folder/"q1/candidates.tsv",expert_occurrences=root/"expert_occurrences.tsv",
                 operating_point=point,source_binding=sources,fullnumeric_q_paths={str(q):folder/f"q{q:g}" for q in TARGETS})
        summaries.append(dict(arm_id=arm["arm_id"],summary=result))
        progress(root,status="AUDIT_ARM_COMPLETE",completed=len(summaries),total=28)
    write_json(root/"audit_complete.json",dict(status="PASS",arm_count=28,arms=summaries))


def report(root):
    protocol=load_protocol(root)
    online=read_tsv(root/"online_summary.tsv");bridge=read_tsv(root/"retrospective_bridge.tsv")
    lines=["# Causal observation development study", "", "Common setup: UI1800–1899; application UI1900–2359. All rows use the same79known occurrences.",
      "", "These are within-recording development results. Known-window ranking and configured expert-center traces are retrospective diagnostics. They do not measure onset, precision, independent generalization, or fish control.",
      "", "| Score | Framewise q1 matches /79 | Emitted rows /460frames | Whole-burst top58 legacy | Whole-burst top58 separated |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in protocol["arms"]:
        name=arm["arm_id"];row=next(x for x in online if x["arm_id"]==name and float(x["q"])==1)
        bs={x["rule"]:x["matched"] for x in bridge if x["arm_id"]==name and x["burst_id"]=="all"}
        lines.append(f"| {name} | {row['matched_known_positive_count']} | {row['emitted_frame_proposal_count']} | {bs['legacy_maxfilter']} | {bs['separated_nms']} |")
    lines.extend(["", "## Interpretation and remaining measurements", "",
        "All calibration uses the common prefix; equal calibration targets do not guarantee equal application output counts. Compare the full curves in online_summary.tsv. The legacy carrier fit is restricted to setup; the retrospective top58 decision still consumes the entire declared activity interval.",
        "", "Configured-region diagnostics use known centers supplied retrospectively. A threshold crossing anywhere within a broad known interval is activity coverage, not event onset recovery. New source discovery requires future geometry adjudication before changing a controller's tracked region set.",
        "", "Motion correction is a bounded integer template-registration control. Global-offset subtraction can remove shared neural activity as well as illumination changes. Neither is accepted as a deployment default from this development comparison.",
        "", "The current data have no exhaustive negative annotations or per-neuron onset truth. Those measurements require a reviewed panel and synchronized acquisition/behavior timestamps. Batch scoring times are not streaming latency tests.", "",
        "Scientific audit: "+("PASS across all28q1 states." if (root/"audit_complete.json").exists() else "PENDING. Numerical completion is not media-audit completion."), ""])
    (root/"REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["preflight","run","evaluate","audit","report"])
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    parser.add_argument("--chunk-frames",type=int,default=8)
    args=parser.parse_args();root=args.output.resolve()
    if args.chunk_frames<1:raise ValueError("chunk size must be positive")
    if args.command=="run":run(root,args.device,args.chunk_frames)
    else:globals()[args.command](root)


if __name__=="__main__":main()
