"""Eight-arm, setup-calibrated component study; completed pilot is read-only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec, build_kernels, describe, iter_chunks
from .control_study import conditioned_frames
from .spatiotemporal_study import REPO, binding, verify, frame_sets, mmap, REFERENCE_AREA_PX
from .spatiotemporal_metrics import seal_candidates, evaluate_threshold_sweep, evaluate_framewise
from .two_stencil_campaign import write_json, write_tsv, sha256
from .two_stencil_evaluation import extract_frame_candidates
from .spatiotemporal_sparse_fast import evaluate_occurrence_windows

PRIOR = REPO / "Outputs/GammaLSST/sensitivity_20260914_r2"
DEFAULT_ROOT = REPO / "Outputs/GammaLSNecessity/necessity_20260914_r1"
BUDGETS = (0., .25, .5, 1., 2., 4., 8., 16.)
DEADLINES = (0, 20, 40, 60, 100, 200, 500)
SPEC = GammaSTSpec()


def read(path):
    return json.loads(Path(path).read_text())


def arms():
    return [dict(arm_id=f"{mode}_{score}", input_mode=mode, readout=score,
                 formula={"X":"X", "A":"K_target * X", "C":"A - M", "Z":"(A - M) / max(Spread, setup_floor)"}[score])
            for mode in ("level", "difference") for score in ("X", "A", "C", "Z")]


def emit(root, **row):
    row = dict(updated_unix=time.time(), **row)
    write_json(root / "progress.json", row)
    print(json.dumps(row), flush=True)


def cutoff(ranked_positive, budget):
    """Strict score cutoff; ties can reduce, never exceed, the setup budget."""
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    scores = np.sort(np.asarray(ranked_positive, dtype=float))[::-1]
    if not np.isfinite(scores).all() or np.any(scores <= 0):
        raise ValueError("expected finite positive setup NMS scores")
    tau = float(scores[budget]) if len(scores) > budget else 0.
    return tau, int(np.count_nonzero(scores > tau))


def threshold_plan(setup_scores, area, frame_count):
    rows = []
    for q in BUDGETS:
        budget = math.floor(q * area / REFERENCE_AREA_PX * frame_count)
        tau, observed = cutoff(setup_scores, budget)
        rows.append(dict(threshold_id=f"q{q:g}", setup_budget_per_reference_area_frame=q,
                         threshold=tau, setup_proposal_budget=budget, setup_proposal_count=observed))
    rows += [dict(threshold_id="all_positive", setup_budget_per_reference_area_frame=None,
                  threshold=0., setup_proposal_budget=None, setup_proposal_count=len(setup_scores)),
             dict(threshold_id="no_output", setup_budget_per_reference_area_frame=None,
                  threshold=None, setup_proposal_budget=0, setup_proposal_count=0)]
    return rows


def preflight(root):
    if root.exists():
        raise FileExistsError("A new experiment root is required")
    prior = read(PRIOR / "protocol.json")
    completion = read(PRIOR / "completion_manifest.json")
    if completion.get("status") != "PASS" or completion["counts"]["evaluated_cells"] != 357:
        raise RuntimeError("Prior pilot is not complete")
    for record in prior["code_bindings"]:
        verify(record)
    cases = sorted(p.name for p in (PRIOR / "datasets").iterdir())
    if len(cases) != 17:
        raise RuntimeError("Expected exactly the previous 17 datasets")
    if shutil.disk_usage(REPO).free < 100 * 1024**3:
        raise RuntimeError("Require 100 GiB free disk headroom")
    acceptance = read(PRIOR / "real_review/annotation_acceptance.json")
    if acceptance.get("accepted"):
        raise RuntimeError("New accepted real labels require an explicit versioned evaluation contract")
    root.mkdir(parents=True)
    code = [Path(__file__), Path(__file__).with_name("spatiotemporal_sparse_fast.py")]
    code += [Path(row["path"]) for row in prior["code_bindings"]]
    code += [Path(__file__).with_name("spatiotemporal_audit.py"), Path(__file__).with_name("two_stencil_audit.py"), REPO/"neurobench/reports/scientific_audit.py"]
    protocol = dict(experiment="gamma_ls_necessity", schema_version=1,
        prior_completion=binding(PRIOR/"completion_manifest.json"), prior_protocol=binding(PRIOR/"protocol.json"),
        real_annotation_acceptance=binding(PRIOR/"real_review/annotation_acceptance.json"),
        code_bindings=[binding(p) for p in dict.fromkeys(code)], cases=cases, arms=arms(), expected_cells=136,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        initial_git_status=subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True),
        kernel=describe(build_kernels(SPEC)), frame_rate_hz=50, pixel_size_um=.5,
        inputs="Gaussian sigma1 truncate4 reflect; EMA alpha.4 initialized to first spatial frame. Level=current EMA; difference=current EMA minus previous EMA. No rectification.",
        domain="Exactly prior halo49 and setup/application frames; full source support retained before cropping",
        standardization="Direct target/reference on the same input; no guard mask; Spread=sqrt(max(Kref*X^2-M^2,0)); setup positive Spread 10th percentile floor >=1e-6",
        calibration=dict(budgets=list(BUDGETS),reference_area_px=REFERENCE_AREA_PX,operating_q=1.,
            rule="Strict positive NMS setup order statistic. budget=floor(q*eligible_area/194820*setup_frame_count). Equal target setup burden, not equal application false alarm probability.",
            thresholds="Eight setup-derived cutoffs plus all-positive and no-output endpoints. Duplicate numerical cutoffs retain their budget identities."),
        evaluation=dict(discovery="Same one-to-one inclusive6px frame assignment as prior pilot; complete positive NMS prefixes sealed for all136cells before activity truth join",
            deadlines_ms=list(DEADLINES),deadline_denominator="All simulated events, including misses; sampled fluorescence onset only",
            monitoring="Separate synthetic oracle-configured exact center pixels. Per-ROI strict setup cutoff with floor(.01*setup_frames) positive exceedance budget. No NMS. Known source coordinates are geometry-only input before scoring; no activity times enter calibration.",
            real="Sparse broad-window coverage and unknown proposal burden only; no real precision, exhaustive sensitivity or onset delay",
            inference="Within-pilot development comparison; two paired seeds per template. No independent test, causal control, or universal necessity claim."),
        scientific_audit=dict(enabled=True,cells=136,operating_point="q1",scope="Every expert and consolidated model ROI; all fullfield/closeup videos, exact traces, comparison figures/tables, source hashes and full lossless decode. Numeric curves do not imply media for every cutoff."),
        resources=dict(free_disk_bytes=shutil.disk_usage(root).free,compute="CPU float64 FFT, chunks8, one numerical thread, low priority; prior difference stages reused by hash",media="at most3 disjoint workers, one numerical/codec thread each; CPU6/7 excluded"))
    write_json(root/"protocol.json",protocol)
    write_json(root/"preflight.json",dict(status="PASS",protocol_sha256=sha256(root/"protocol.json")))
    emit(root,status="PREFLIGHT_PASS",cells=136)


def load_protocol(root):
    p=read(root/"protocol.json")
    if sha256(root/"protocol.json") != read(root/"preflight.json")["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for record in p["code_bindings"]:
        verify(record)
    return p


def link(source, destination):
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != Path(source).resolve():
            raise RuntimeError(f"Unexpected existing link: {destination}")
    else:
        destination.symlink_to(Path(source).resolve())


def prepare(root):
    p=load_protocol(root)
    for case in p["cases"]:
        old=PRIOR/"datasets"/case; folder=root/"datasets"/case
        if (folder/"prepared.json").exists():
            for b in read(folder/"prepared.json")["bindings"]:verify(b)
            continue
        prior=read(old/"prepared.json")
        for b in prior.values():
            if isinstance(b,dict) and "sha256" in b:verify(b)
        folder.mkdir(parents=True,exist_ok=True)
        meta=read(old/"metadata.json"); y0,x0,y1,x1=meta["evaluation_box_yxyx"]
        for name in ("metadata.json","experts.json","active.json","Raw.npy"):
            link(old/name,folder/name)
        raw=np.load(old/"raw_source.npy",mmap_mode="r")
        saved_diff=np.load(old/"input_source.npy",mmap_mode="r")
        current=mmap(folder/"level_source.npy",raw.shape)
        for i,(level,delta) in enumerate(conditioned_frames(raw,1.,.4)):
            if not np.array_equal(delta,saved_diff[i]):
                raise RuntimeError(f"Conditioning disagrees with frozen difference at {case}:{i}")
            current[i]=level
        current.flush()
        np.save(folder/"level.npy",np.asarray(current[:,y0:y1,x0:x1]))
        link(old/"Input.npy",folder/"difference.npy")
        link(old/"input_source.npy",folder/"difference_source.npy")
        # Only positions/identities enter configured monitoring; no activity times.
        geometry={}
        if meta["truth_mode"]=="fully_synthetic":
            for event in read(old/"experts.json"):
                roi=str(event["canonical_roi_id"]);xy=(float(event["x_px"]),float(event["y_px"]))
                if roi in geometry and geometry[roi] != xy:raise ValueError("Configured source moved")
                geometry[roi]=xy
        write_json(folder/"configured_geometry.json",[dict(canonical_roi_id=k,x_px=v[0],y_px=v[1]) for k,v in sorted(geometry.items())])
        names=("metadata.json","experts.json","active.json","Raw.npy","level_source.npy","level.npy","difference.npy","difference_source.npy","configured_geometry.json")
        write_json(folder/"prepared.json",dict(status="PASS",conditioning_difference_byte_equal=True,
                    bindings=[binding(folder/n) for n in names],prior_prepared=binding(old/"prepared.json")))
        emit(root,status="DATASET_PREPARED",case=case)
    write_json(root/"datasets_complete.json",dict(status="PASS",datasets=17))


def prepare_mode(root,case,mode):
    folder=root/"datasets"/case;out=root/"stages"/case/mode
    if (out/"complete.json").exists():return read(out/"complete.json")
    out.mkdir(parents=True,exist_ok=True)
    meta=read(folder/"metadata.json");frames,setup,application=frame_sets(meta)
    lookup={f:i for i,f in enumerate(frames)};si=[lookup[f] for f in setup]
    inp=np.load(folder/f"{mode}.npy",mmap_mode="r")
    if mode=="difference":
        old=PRIOR/"cells"/case/SPEC.spec_id;seal=read(old/"sealed.json")
        for k in ("A","M","Spread","Score"):verify(seal["stages"][k])
        for k in ("A","M","Spread"):link(old/f"{k}.npy",out/f"{k}.npy")
        link(old/"Score.npy",out/"Z.npy")
        floor=read(old/"calibration.json")["scale_floor"]
    else:
        arrays={k:mmap(out/f"{k}.npy",inp.shape) for k in ("A","M","Spread")}
        source=np.load(folder/"level_source.npy",mmap_mode="r");y0,x0,y1,x1=meta["evaluation_box_yxyx"]
        for start,stop,values in iter_chunks(source,build_kernels(SPEC),chunk_frames=8):
            arrays["A"][start:stop]=values["A"][:,y0:y1,x0:x1]
            arrays["M"][start:stop]=values["M"][:,y0:y1,x0:x1]
            arrays["Spread"][start:stop]=np.sqrt(values["variance"][:,y0:y1,x0:x1])
        for value in arrays.values():value.flush()
        population=np.asarray(arrays["Spread"][si,6:-6,6:-6]).ravel()
        positive=population[population>0]
        floor=max(1e-6,float(np.percentile(positive,10)) if len(positive) else 0.)
        del population,positive,arrays
    a=np.load(out/"A.npy",mmap_mode="r");m=np.load(out/"M.npy",mmap_mode="r");spread=np.load(out/"Spread.npy",mmap_mode="r")
    contrast=mmap(out/"C.npy",inp.shape)
    z=mmap(out/"Z.npy",inp.shape) if mode=="level" else None
    for start in range(0,len(inp),8):
        stop=min(len(inp),start+8);contrast[start:stop]=a[start:stop]-m[start:stop]
        if z is not None:z[start:stop]=contrast[start:stop]/np.maximum(spread[start:stop],floor)
    contrast.flush()
    if z is not None:z.flush()
    records={k:binding(out/f"{k}.npy") for k in ("A","M","Spread","C","Z")}
    records["X"]=binding(folder/f"{mode}.npy");records["Raw"]=binding(folder/"Raw.npy")
    result=dict(status="PASS",stages=records,scale_floor=floor,input_mode=mode,
                difference_stages_reused=(mode=="difference"))
    write_json(out/"complete.json",result)
    return result


def configured_calibration(score,geometry,indices):
    rows=[]
    for roi in geometry:
        x,y=int(round(roi["x_px"])),int(round(roi["y_px"]))
        values=np.asarray(score[indices,y,x],float)
        tau,count=cutoff(values[values>0],math.floor(.01*len(indices)))
        rows.append(dict(roi,x_index=x,y_index=y,threshold=tau,setup_exceedance_count=count,
                         setup_exceedance_budget=math.floor(.01*len(indices))))
    return rows


def run(root):
    p=load_protocol(root)
    for case in p["cases"]:
        folder=root/"datasets"/case;meta=read(folder/"metadata.json")
        for b in read(folder/"prepared.json")["bindings"]:verify(b)
        frames,setup,application=frame_sets(meta);lookup={f:i for i,f in enumerate(frames)}
        si=[lookup[f] for f in setup]
        for mode in ("level","difference"):
            state=prepare_mode(root,case,mode)
            for arm in [a for a in arms() if a["input_mode"]==mode]:
                out=root/"cells"/case/arm["arm_id"]
                if (out/"sealed.json").exists():
                    seal=read(out/"sealed.json")
                    for b in [seal["prefix"],seal["audit_candidates"],seal["calibration"]]:verify(b)
                    continue
                out.mkdir(parents=True,exist_ok=True)
                score=np.load(state["stages"][arm["readout"]]["path"],mmap_mode="r")
                area=(score.shape[1]-12)*(score.shape[2]-12)
                peaks=[]
                for f in setup:
                    peaks.extend(r["score"] for r in extract_frame_candidates(score[lookup[f]],source_frame_ui=f,threshold_z=0.))
                plan=threshold_plan(peaks,area,len(setup));fixed=next(r for r in plan if r["threshold_id"]=="q1")
                op=dict(fixed,threshold_z=fixed["threshold"],threshold_frozen_from_calibration_only=True,
                    scale_floor=state["scale_floor"],case_id=case,spec_id=arm["arm_id"],arm=arm,
                    truth_mode=meta["truth_mode"],eligible_area_px=area,reference_area_px=REFERENCE_AREA_PX,
                    target_proposals_per_frame=area/REFERENCE_AREA_PX,nominal_proposals_per_reference_area_frame=1.,
                    setup_source_frames_ui=setup,application_source_start_ui=application[0],application_source_stop_ui=application[-1],
                    application_frame_count=len(application),score_units="dimensionless local standard score" if arm["readout"]=="Z" else "fluorescence intensity units")
                prefix=[]
                for f in application:
                    prefix.extend(extract_frame_candidates(score[lookup[f]],source_frame_ui=f,threshold_z=0.,cell_id=arm["arm_id"]))
                audit=[dict(r,threshold_z=op["threshold"],target_proposals_per_frame=op["target_proposals_per_frame"]) for r in prefix if r["score"]>op["threshold"]]
                seal_candidates(prefix,source_frames_ui=application)
                write_json(out/"prefix.json",prefix);write_json(out/"audit_candidates.json",audit);write_tsv(out/"audit_candidates.tsv",audit)
                write_json(out/"calibration.json",op);write_json(out/"threshold_plan.json",plan)
                write_json(out/"configured_calibration.json",configured_calibration(score,read(folder/"configured_geometry.json"),si))
                records=dict(state["stages"]);records["Input"]=records["X"];records["Score"]=records[arm["readout"]]
                write_json(out/"sealed.json",dict(status="SEALED_BEFORE_ACTIVITY_TRUTH_JOIN",stages=records,
                    prefix=binding(out/"prefix.json"),audit_candidates=binding(out/"audit_candidates.json"),calibration=binding(out/"calibration.json"),
                    threshold_plan=binding(out/"threshold_plan.json"),configured_calibration=binding(out/"configured_calibration.json")))
                emit(root,status="CELL_SEALED",case=case,arm=arm["arm_id"],prefix_count=len(prefix),audit_proposals=len(audit))
                del prefix,audit
    write_json(root/"computation_complete.json",dict(status="PASS",cells=136))


def deadline_rows(events):
    return [dict(deadline_ms=d,event_count=len(events),recovered_by_deadline=sum(e["first_delay_ms"] is not None and e["first_delay_ms"]<=d+1e-9 for e in events)) for d in DEADLINES]


def monitor(score,calibration,frames,application,active,events):
    lookup={f:i for i,f in enumerate(frames)}
    truth={(str(r["canonical_roi_id"]),int(r["source_frame_ui"])) for r in active}
    positives=set();count=0
    for roi in calibration:
        for frame in application:
            count+=1
            if float(score[lookup[frame],roi["y_index"],roi["x_index"]])>roi["threshold"]:
                positives.add((str(roi["canonical_roi_id"]),frame))
    tp=len(positives & truth);fp=len(positives-truth);fn=len(truth-positives);tn=count-tp-fp-fn
    event_output=[]
    for event in events:
        first=min((f for roi,f in positives if roi==str(event["canonical_roi_id"]) and int(event["source_start_ui"])<=f<=int(event["source_stop_ui"])),default=None)
        event_output.append(dict(event_id=event.get("event_id",event["observation_id"]),first_delay_ms=None if first is None else (first-int(event["source_start_ui"]))*20))
    return dict(truth_mode="fully_synthetic",monitoring_mode="oracle_configured_centers_no_NMS",roi_frame_count=count,
        true_positive_count=tp,false_positive_count=fp,false_negative_count=fn,true_negative_count=tn,
        precision=tp/(tp+fp) if tp+fp else None,sensitivity=tp/(tp+fn) if tp+fn else None,
        inactive_roi_frame_exceedance_rate=fp/(fp+tn) if fp+tn else None,event_rows=event_output,deadline_rows=deadline_rows(event_output))


def evaluate(root):
    p=load_protocol(root)
    if not all((root/"cells"/case/arm["arm_id"]/"sealed.json").exists() for case in p["cases"] for arm in arms()):
        raise RuntimeError("Seal all136cells before joining activity truth")
    all_rows=[];all_ops=[];all_monitors=[]
    for case in p["cases"]:
        folder=root/"datasets"/case;meta=read(folder/"metadata.json");frames,_,application=frame_sets(meta)
        experts=read(folder/"experts.json");active=read(folder/"active.json")
        for arm in arms():
            out=root/"cells"/case/arm["arm_id"];seal=read(out/"sealed.json")
            if not (out/"evaluated.json").exists():
                for k in ("prefix","calibration","threshold_plan","configured_calibration"):verify(seal[k])
                rows=read(out/"prefix.json");op=read(out/"calibration.json");plan=read(out/"threshold_plan.json")
                thresholds=list(dict.fromkeys(math.inf if r["threshold"] is None else r["threshold"] for r in plan))
                if meta["truth_mode"]=="fully_synthetic":
                    stream=seal_candidates(rows,source_frames_ui=application)
                    result=evaluate_threshold_sweep(stream,active,experts,thresholds=thresholds)
                    fixed=evaluate_framewise(stream,active,experts,threshold_z=op["threshold"])
                    fixed["deadline_rows"]=deadline_rows(fixed["event_rows"])
                    score=np.load(seal["stages"]["Score"]["path"],mmap_mode="r")
                    monitoring=monitor(score,read(out/"configured_calibration.json"),frames,application,active,experts)
                    write_json(out/"monitoring.json",monitoring)
                else:
                    curves=[];event_rows=[];fixed=None
                    intervals={int(r["burst_id"]):(int(r["source_start_ui"]),int(r["source_stop_ui"])) for r in experts}
                    for tau in thresholds:
                        selected=[r for r in rows if r["score"]>tau]
                        res=evaluate_occurrence_windows(selected,experts,burst_intervals_ui=intervals)
                        row=dict(threshold_z=None if math.isinf(tau) else tau,truth_mode="sparse_real",precision=None,
                            framewise_sensitivity=None,false_positive_count=None,event_count=None,proposal_count=len(selected),
                            application_frame_count=len(application),exposure_seconds=len(application)*.02,**res["summary"])
                        curves.append(row);event_rows.extend(dict(r,threshold_z=row["threshold_z"]) for r in res["occurrence_rows"])
                        if tau==op["threshold"]:fixed=dict(summary=row,occurrence_rows=res["occurrence_rows"],site_rows=res["site_rows"])
                    result=dict(curve_rows=curves,event_rows=event_rows)
                    write_json(out/"monitoring.json",dict(truth_mode="sparse_real",status="NOT_IDENTIFIABLE_FROM_SPARSE_WINDOWS",precision=None,sensitivity=None))
                by_tau={r["threshold_z"]:r for r in result["curve_rows"]}
                expanded=[]
                for setting in plan:
                    row=dict(by_tau[setting["threshold"]],**setting,case_id=case,**arm,eligible_area_px=op["eligible_area_px"])
                    row["proposal_rate_per_reference_area_frame"]=row["proposal_count"]/len(application)*REFERENCE_AREA_PX/op["eligible_area_px"]
                    row["false_proposals_per_10000um2_second"]=(row["false_positive_count"]/(len(application)*.02)/(op["eligible_area_px"]*.25)*10000 if row.get("false_positive_count") is not None else None)
                    if meta["truth_mode"]=="fully_synthetic":
                        ev=[e for e in result["event_rows"] if e["threshold_z"]==setting["threshold"]]
                        row["deadline_rows"]=deadline_rows(ev)
                    expanded.append(row)
                write_json(out/"curves.json",dict(curve_rows=expanded,event_rows=result["event_rows"]))
                write_json(out/"operating_metrics.json",fixed)
                write_json(out/"evaluated.json",dict(status="PASS",seal=binding(out/"sealed.json"),
                    outputs=[binding(out/n) for n in ("curves.json","operating_metrics.json","monitoring.json")]))
                emit(root,status="CELL_EVALUATED",case=case,arm=arm["arm_id"])
                del rows
            all_rows.extend(read(out/"curves.json")["curve_rows"])
            all_ops.append(dict(case_id=case,**arm,calibration=read(out/"calibration.json"),metrics=read(out/"operating_metrics.json")))
            all_monitors.append(dict(case_id=case,**arm,metrics=read(out/"monitoring.json")))
    write_json(root/"all_curves.json",all_rows);write_tsv(root/"all_curves.tsv",all_rows)
    write_json(root/"operating_points.json",all_ops);write_json(root/"monitoring.json",all_monitors)
    write_json(root/"evaluation_complete.json",dict(status="PASS",cells=len(all_ops),curve_rows=len(all_rows)))


def display(root):
    if (root/"display_contract.json").exists():return read(root/"display_contract.json")
    contract={}
    for case in load_protocol(root)["cases"]:
        raw=np.load(root/"datasets"/case/"Raw.npy",mmap_mode="r")
        raw_limits=list(map(float,np.percentile(raw[::5],[1,99.8])))
        if raw_limits[1]<=raw_limits[0]:raw_limits[1]=raw_limits[0]+1
        limits={"Raw":raw_limits}
        for mode in ("level","difference"):
            state=read(root/"stages"/case/mode/"complete.json")
            values={k:np.load(state["stages"][k]["path"],mmap_mode="r")[::5] for k in ("X","A","C")}
            if mode=="level":
                high=max(1.,*(float(np.percentile(v,99.8)) for v in (values["X"],values["A"])))
                common=[0.,high]
            else:
                high=max(1e-6,*(float(np.percentile(np.abs(v),99.8)) for v in (values["X"],values["A"])))
                common=[-high,high]
            c=max(1e-6,float(np.percentile(np.abs(values["C"]),99.8)))
            limits[mode]=dict(X=common,A=common,C=[-c,c],Z=[-8.,8.])
        contract[case]=limits
    result=dict(cases=contract,rule="Fixed per case/input/unit family across all four readouts. Input/A share scale. Contrast symmetric in fluorescence units; Z symmetric[-8,8]. Level and signed-change scales differ explicitly; display values do not enter scores or calibration.")
    write_json(root/"display_contract.json",result)
    return result


def media(root,worker=0,workers=1):
    from .spatiotemporal_audit import run_spatiotemporal_audit
    p=load_protocol(root)
    if read(root/"evaluation_complete.json").get("cells")!=136:raise RuntimeError("Evaluate all136cells first")
    limits=display(root)["cases"];count=0
    for ordinal,(case,arm) in enumerate((c,a) for c in p["cases"] for a in arms()):
        if ordinal%workers!=worker:continue
        folder=root/"datasets"/case;cell=root/"cells"/case/arm["arm_id"];seal=read(cell/"sealed.json")
        frames,_,_=frame_sets(read(folder/"metadata.json"));mode=arm["input_mode"]
        names=["Raw","Input"]+(["A"] if arm["readout"] in ("C","Z") else [])+["Score"]
        stages={k:seal["stages"][k]["path"] for k in names}
        dl={"Raw":limits[case]["Raw"],"Input":limits[case][mode]["X"],"Score":limits[case][mode][arm["readout"]]}
        if "A" in names:dl["A"]=limits[case][mode]["A"]
        signed=[k for k in names if dl[k][0]<0]
        source=dict(stage_sha256={k:seal["stages"][k]["sha256"] for k in names},
            candidates_sha256=seal["audit_candidates"]["sha256"],expert_occurrences_sha256=sha256(folder/"experts.json"),
            candidate_seal_sha256=sha256(cell/"sealed.json"),display_contract_sha256=sha256(root/"display_contract.json"),
            numeric_stage_bindings={k:seal["stages"][k] for k in ("M","Spread","C","Z") if seal["stages"][k]["path"] not in stages.values()},
            original_source_offset_xy=read(folder/"metadata.json")["original_source_offset_xy"],truth_mode=read(folder/"metadata.json")["truth_mode"],
            arm=arm,stage_semantics="Raw original fluorescence; Input chosen level or signed change; optional A centered target; Score exactly the selected readout. Linked stages are diagnostic branches, not hidden score dependencies.")
        summary=run_spatiotemporal_audit(root/"audits"/case/arm["arm_id"],stage_paths=stages,signed_stage_keys=signed,
            display_limits=dl,source_frames_ui=frames,source_binding=source,operating_point=read(cell/"calibration.json"),
            numeric_threshold_paths={"curves":str(cell/"curves.json"),"monitoring":str(cell/"monitoring.json")},
            candidates_path=cell/"audit_candidates.json",expert_occurrences=folder/"experts.json",fps=50.)
        count+=1
        print(json.dumps(dict(status="AUDIT_CELL_COMPLETE",case=case,arm=arm["arm_id"],worker=worker,count=count,videos=summary["video_count"])),flush=True)
    write_json(root/f"media_worker_{worker}.json",dict(status="PASS",worker=worker,workers=workers,cells=count))


def validate(root):
    from .two_stencil_audit import verify_completed_audit
    p=load_protocol(root);rows=[]
    for case in p["cases"]:
        for arm in arms():
            folder=root/"audits"/case/arm["arm_id"];verify_completed_audit(folder)
            summary=read(folder/"summary.json");v=read(folder/"validation.json")
            if not summary.get("scientific_audit_complete") or v.get("status")!="passed":raise RuntimeError(folder)
            rows.append(dict(case_id=case,arm_id=arm["arm_id"],summary=summary,metadata_bindings={n:binding(folder/n) for n in ("summary.json","status.json","validation.json","inventory.json","artifact_index.json","source_manifest.json","run_contract.json","llm_context.json")}))
    if len(rows)!=136:raise RuntimeError("Expected136audits")
    write_json(root/"audit_complete.json",dict(status="PASS",cells=136,audits=rows))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("preflight","prepare","run","evaluate","display","media","validate"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--worker",type=int,default=0);parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=="media":media(root,args.worker,args.workers)
    else:globals()[args.command](root)


if __name__=="__main__":main()
