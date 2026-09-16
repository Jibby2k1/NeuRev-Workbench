"""Regional setup calibration and paired crowded-source mechanism experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .necessity_study import (read, binding, verify, write_json, write_tsv, sha256,
                             threshold_plan, deadline_rows, prepare_mode, emit, link)
from .spatiotemporal_study import REPO, frame_sets, mmap, REFERENCE_AREA_PX, prepare_case
from .control_study import conditioned_frames
from .two_stencil_evaluation import extract_frame_candidates
from .spatiotemporal_metrics import seal_candidates, evaluate_framewise, evaluate_threshold_sweep
from .spatiotemporal_sparse_fast import evaluate_occurrence_windows
from .followup_calibration import regions, regional_plan, select

BASE = REPO / "Outputs/GammaLSNecessity/necessity_20260914_r1"
ROOT = REPO / "Outputs/GammaLSFollowup/followup_20260914_r1"
PAPER = REPO.parent / "Neural_Event_Extraction_Gamma_LS_Clarity_Revision_2026-09-12/editorial"
SEEDS = (20260916, 20260917, 20260918)
READOUTS = ("level_X", "level_A", "level_C", "level_Z", "difference_Z")


def cells(p):
    return [dict(case_id=c, arm_id=f"{a}__{method}", base_arm=a, study="regional",
                 calibration_method=method, input_mode=a.split("_")[0], readout=a[-1], window=13)
            for c in p["regional_cases"] for a in READOUTS for method in ("global", "regional")] + [
                dict(case_id=c["case_id"], arm_id=f"level_{a}__w{window}", base_arm=f"level_{a}",
                     study="crowding", calibration_method="global", input_mode="level", readout=a, window=window)
                for c in p["crowding_cases"] for a in ("A", "C", "Z") for window in (13, 3)]


def preflight(root):
    from .followup_fixtures import case_id
    if root.exists():
        raise FileExistsError("A new noncolliding study root is required")
    completed = read(BASE / "completion_manifest.json")
    if completed.get("status") != "PASS":
        raise RuntimeError("Completed necessity baseline required")
    old = read(BASE / "protocol.json")
    verify(old['prior_completion'])
    prior_completed=read(old['prior_completion']['path'])
    authority={str(Path(b['path']).resolve()):b for b in prior_completed['evidence_bindings']}
    for case in old['cases']:
        for name in ('metadata.json','experts.json','active.json'):
            path=(BASE/'datasets'/case/name).resolve()
            if str(path) not in authority:raise RuntimeError(f'Baseline file not bound by completed evidence: {path}')
            verify(authority[str(path)])
    for b in old["code_bindings"]:
        verify(b)
    if shutil.disk_usage(REPO).free < 100 * 1024**3:
        raise RuntimeError("Need 100GiB disk headroom")
    acceptance = Path(old["real_annotation_acceptance"]["path"])
    if read(acceptance).get("accepted"):
        raise RuntimeError("Accepted real labels require a separately frozen evaluation contract")
    code = [Path(__file__).with_name(n) for n in ("followup_study.py", "followup_calibration.py",
            "followup_fixtures.py", "followup_selection.py", "followup_audit.py", "followup_media.py")]
    code += [Path(b["path"]) for b in old["code_bindings"]]
    code += [REPO/"neurobench/experiments/gamma_ls_difference/necessity_study.py"]
    p = dict(schema_version=1, experiment="gamma_ls_regional_and_crowding_followup",
             baseline_completion=binding(BASE/"completion_manifest.json"),
             baseline_protocol=binding(BASE/"protocol.json"), real_annotation_acceptance=binding(acceptance),
             baseline_dataset_completion=old['prior_completion'],
             regional_cases=old["cases"], crowding_cases=[dict(case_id=case_id(s, d), seed=s, separation_px=d)
                 for s in SEEDS for d in (None, 8, 12, 16)],
             expected_cells=242, new_audits=157, reused_global_audits=85,
             code_bindings=[binding(f) for f in dict.fromkeys(code)],
             git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             initial_git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
             conditioning=old["inputs"], kernel=old["kernel"], frame_rate_hz=50, pixel_size_um=.5,
             regional=dict(grid=[2,3], domain="Same eligible6px interior and unchanged full-context Gamma scores. No masks.",
                 budget="Same total integer budget, Hamilton area allocation; row-major ties; setup order statistics; strict score>cutoff.",
                 selection="Unchanged positive NMS prefix; filter by proposal region without reranking; reassignment after each regional filter.",
                 scores=list(READOUTS), thresholds=old["calibration"], operating_point="q1"),
             crowding=dict(seeds=list(SEEDS), evaluation_shape=[128,128], halo=49, warmup=64, setup=100,
                 application=300, weak_center_xy=[68,64], weak_sigma_px=1, weak_amplitude=18,
                 neighbor_sigma_px=2, neighbor_amplitude=24, rise_ms=100, decay_ms=1000, onset_application_index=20,
                 temporal_profile="Same finite 1%-cutoff profile as baseline; technical stress settings, not measured kinetics.",
                 scenes="Weak alone or strong neighbor left at8/12/16px; same noise realization and identical setup within each seed.",
                 selection="13x13 vs3x3 local maximum; same score/y/x ranking,6px border, greedy strictly>6px Euclidean separation; calibrate each independently.",
                 match_radii_px=[6,2], localization="2px inclusive disks are disjoint at all declared separations.",
                 deadlines_ms=[0,20,40,60,100,200,500]),
             scope="Development mechanism tests. Real known-window coverage and unknown burden only. No controller, neural timing or independent-recording claims.",
             deferred="Configured-source pooled/per-center calibration and nuisance-at-source study remain separate next experiments; no human annotation acceptance inferred.",
             scientific_audit=dict(enabled=True, operating_point="q1", expected_cells=242,
                 policy="Every applicable expert/model fullfield and ROI closeup/trace plus comparisons;85identical global audits reused by hash,157new complete audits. All other cutoffs numerical only."),
             resources=dict(cpu="one numerical thread, chunks8, lowpriority, CPUs6/7excluded", media="at most3workers, onecodec thread each", free_disk_bytes=shutil.disk_usage(REPO).free))
    root.mkdir(parents=True)
    write_json(root/"protocol.json", p)
    write_json(root/"preflight.json", dict(status="PASS", protocol_sha256=sha256(root/"protocol.json")))
    emit(root, status="PREFLIGHT_PASS", cells=242)


def load(root):
    p = read(root/"protocol.json")
    if sha256(root/"protocol.json") != read(root/"preflight.json")["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for b in p["code_bindings"]:
        verify(b)
    verify(p["baseline_completion"])
    return p


def dataset_bindings(folder):
    return [binding(folder/n) for n in ("metadata.json","experts.json","active.json","prepared.json")]


def verify_seal(seal):
    for key in ("prefix","setup_prefix","audit_candidates","threshold_plan","calibration"):
        verify(seal[key])
    for b in seal["dataset_bindings"]:
        verify(b)


def prepare(root):
    from .followup_fixtures import generate_crowding
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = load(root)
    (root/"datasets").mkdir(exist_ok=True)
    (root/"stages").mkdir(exist_ok=True)
    for case in p["regional_cases"]:
        link(BASE/"datasets"/case, root/"datasets"/case)
        link(BASE/"stages"/case, root/"stages"/case)
    pairs = {}
    for c in p["crowding_cases"]:
        folder = root/"datasets"/c["case_id"]
        if not (folder/"level_prepared.json").exists():
            scene = generate_crowding(c["seed"], c["separation_px"])
            prepare_case(root, c["case_id"], scene.raw, scene.metadata, scene.event_rows, scene.active_rows)
            current = mmap(folder/"level_source.npy", scene.raw.shape)
            for i, (level, _) in enumerate(conditioned_frames(scene.raw,1.,.4)):
                current[i] = level
            current.flush()
            np.save(folder/"level.npy", current[:,49:-49,49:-49])
            del current, scene
            write_json(folder/"level_prepared.json", dict(status="PASS", bindings=[binding(folder/n) for n in ("level_source.npy", "level.npy")]))
        for b in read(folder/"level_prepared.json")["bindings"]:
            verify(b)
        raw = np.load(folder/"raw_source.npy",mmap_mode="r")
        if c["seed"] in pairs and not np.array_equal(raw[:164], np.load(pairs[c["seed"]],mmap_mode="r")[:164]):
            raise RuntimeError("Paired setup differs")
        pairs[c["seed"]] = folder/"raw_source.npy"
    # Geometry-only projection preflight: setup mean, source coordinates, fixed crop.
    out = root/"projection_preflight"; out.mkdir(exist_ok=True)
    records=[]; datasets={}
    for case in p["regional_cases"] + [c["case_id"] for c in p["crowding_cases"]]:
        folder=root/"datasets"/case; meta=read(folder/"metadata.json");frames,setup,_=frame_sets(meta)
        prepared=read(folder/'prepared.json')
        bound=prepared.get('bindings',[v for v in prepared.values() if isinstance(v,dict) and 'sha256' in v])
        for b in bound:verify(b)
        raw=np.load(folder/"Raw.npy",mmap_mode="r");lookup={f:i for i,f in enumerate(frames)}
        xy=sorted({(float(r["x_px"]),float(r["y_px"])) for r in read(folder/"experts.json")})
        if any(not(6<=x<raw.shape[2]-6 and 6<=y<raw.shape[1]-6) for x,y in xy):
            raise RuntimeError("Expert coordinate outside declared domain")
        path=out/f"{case}.png"
        if not path.exists():
            mean=np.mean(raw[[lookup[f] for f in setup]],axis=0)
            fig,ax=plt.subplots(figsize=(7,5));ax.imshow(mean,cmap="gray",origin="upper")
            if xy:ax.scatter(*zip(*xy),facecolors="none",edgecolors="#46dc7d",s=60)
            ax.set_title(f"{case}\nSetup mean; geometry only; score coordinates x=column,y=row",fontsize=10)
            fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)
        records.append(dict(case_id=case, unique_centers=len(xy), figure=binding(path), coordinates_within_domain=True))
        datasets[case]=dataset_bindings(folder)
    write_json(root/"datasets_complete.json",dict(status="PASS",datasets=29,paired_setup_byte_equal=True,projection_records=records,dataset_bindings=datasets))


def run(root):
    from .followup_selection import extract_candidates
    p=load(root)
    if read(root/"datasets_complete.json")["datasets"] != 29:
        raise RuntimeError("Preparation incomplete")
    datasets=read(root/"datasets_complete.json")["dataset_bindings"]
    for bs in datasets.values():
        for b in bs:verify(b)
    checked_stages=set()
    for cell in cells(p):
        case=cell["case_id"];out=root/"cells"/case/cell["arm_id"]
        if (out/"sealed.json").exists():
            verify_seal(read(out/"sealed.json"))
            continue
        folder=root/"datasets"/case;meta=read(folder/"metadata.json");frames,setup,application=frame_sets(meta)
        lookup={f:i for i,f in enumerate(frames)}
        state=prepare_mode(root,case,cell["input_mode"])
        if (case,cell["input_mode"]) not in checked_stages:
            for b in state["stages"].values():verify(b)
            checked_stages.add((case,cell["input_mode"]))
        records=dict(state["stages"]);records["Input"]=records["X"];records["Score"]=records[cell["readout"]]
        score=np.load(records["Score"]["path"],mmap_mode="r")
        area=(score.shape[1]-12)*(score.shape[2]-12)
        out.mkdir(parents=True,exist_ok=True)
        cache=root/"prefix_cache"/case/f'{cell["base_arm"]}__w{cell["window"]}'
        if not (cache/"complete.json").exists():
            cache.mkdir(parents=True,exist_ok=True)
            fn=(lambda f:extract_frame_candidates(score[lookup[f]],source_frame_ui=f,threshold_z=0.,cell_id=cell["base_arm"])) if cell["study"]=="regional" else (lambda f:extract_candidates(score[lookup[f]],source_frame_ui=f,window=cell["window"],cell_id=cell["arm_id"]))
            setup_rows=[r for f in setup for r in fn(f)]
            if cell["study"]=="regional":
                old=BASE/"cells"/case/cell["base_arm"];b=read(old/"sealed.json")["prefix"];verify(b)
                link(old/"prefix.json",cache/"application.json")
            else:
                prefix=[r for f in application for r in fn(f)]
                seal_candidates(prefix,source_frames_ui=application)
                write_json(cache/"application.json",prefix)
                del prefix
            write_json(cache/"setup.json",setup_rows)
            write_json(cache/"complete.json",dict(status="PASS",bindings=[binding(cache/n) for n in ("setup.json","application.json")]))
        for b in read(cache/"complete.json")["bindings"]:verify(b)
        setup_rows=read(cache/"setup.json")
        if cell["calibration_method"]=="regional":
            plan=regional_plan(setup_rows,regions(score.shape[1:]),len(setup))
        else:
            plan=threshold_plan([r["score"] for r in setup_rows],area,len(setup))
        if cell["study"]=="regional" and cell["calibration_method"]=="global":
            previous=read(BASE/"cells"/case/cell["base_arm"]/"threshold_plan.json")
            if plan != previous:raise RuntimeError("Global calibration replication failed")
        fixed=next(r for r in plan if r["threshold_id"]=="q1")
        op=dict(fixed,threshold_z=fixed["threshold"],threshold_frozen_from_calibration_only=True,
                eligible_area_px=area,target_proposals_per_frame=area/REFERENCE_AREA_PX,
                setup_source_frames_ui=setup,application_source_start_ui=application[0],application_source_stop_ui=application[-1],
                application_frame_count=len(application),scale_floor=state["scale_floor"],truth_mode=meta["truth_mode"],**cell)
        prefix=read(cache/"application.json");audit=select(prefix,fixed,area/REFERENCE_AREA_PX)
        write_json(out/"audit_candidates.json",audit);write_json(out/"threshold_plan.json",plan);write_json(out/"calibration.json",op)
        write_json(out/"sealed.json",dict(status="SEALED_BEFORE_ACTIVITY_TRUTH_JOIN",stages=records,
            dataset_bindings=datasets[case],
            prefix=binding(cache/"application.json"),setup_prefix=binding(cache/"setup.json"),
            audit_candidates=binding(out/"audit_candidates.json"),threshold_plan=binding(out/"threshold_plan.json"),calibration=binding(out/"calibration.json")))
        emit(root,status="CELL_SEALED",**cell,audit_proposals=len(audit))
    write_json(root/"computation_complete.json",dict(status="PASS",cells=len(cells(p))))


def evaluate(root):
    p=load(root);allcells=cells(p);rows=[];replication=[]
    if not all((root/"cells"/c["case_id"]/c["arm_id"]/"sealed.json").exists() for c in allcells):
        raise RuntimeError("Seal all242cells before application truth join")
    for c in allcells:
        out=root/"cells"/c["case_id"]/c["arm_id"];folder=root/"datasets"/c["case_id"]
        seal=read(out/"sealed.json");verify_seal(seal)
        meta=read(folder/"metadata.json");_,_,application=frame_sets(meta)
        if not (out/"evaluated.json").exists():
            prefix=read(seal["prefix"]["path"]);plan=read(out/"threshold_plan.json");op=read(out/"calibration.json")
            experts=read(folder/"experts.json");active=read(folder/"active.json")
            curves=[];event_curves=[]
            for setting in plan:
                selected=select(prefix,setting)
                # A regional subset is NOT a global score prefix. Reassign after filtering.
                if meta["truth_mode"]=="fully_synthetic":
                    stream=seal_candidates(selected,source_frames_ui=application)
                    for radius in ([6.,2.] if c["study"]=="crowding" else [6.]):
                        res=evaluate_threshold_sweep(stream,active,experts,thresholds=[0.],match_radius_px=radius)
                        row=dict(res["curve_rows"][0],**setting,**c,eligible_area_px=op["eligible_area_px"])
                        row["deadline_rows"]=deadline_rows(res["event_rows"])
                        curves.append(row)
                        event_curves.extend(dict(r,threshold_id=setting["threshold_id"],match_radius_px=radius) for r in res["event_rows"])
                        if setting["threshold_id"]=="q1":
                            fixed=evaluate_framewise(stream,active,experts,threshold_z=0.,match_radius_px=radius)
                            write_json(out/f"operating_metrics_r{radius:g}.json",fixed)
                else:
                    intervals={int(r["burst_id"]):(int(r["source_start_ui"]),int(r["source_stop_ui"])) for r in experts}
                    res=evaluate_occurrence_windows(selected,experts,burst_intervals_ui=intervals)
                    curves.append(dict(res["summary"],**setting,**c,truth_mode="sparse_real",precision=None,
                        framewise_sensitivity=None,false_positive_count=None,proposal_count=len(selected),
                        application_frame_count=len(application),exposure_seconds=len(application)*.02,eligible_area_px=op["eligible_area_px"],match_radius_px=6.))
                    if setting["threshold_id"]=="q1":write_json(out/"operating_metrics_r6.json",res)
            write_json(out/"curves.json",dict(curve_rows=curves,event_rows=event_curves))
            write_json(out/"evaluated.json",dict(status="PASS",seal=binding(out/"sealed.json"),outputs=[binding(f) for f in sorted(out.glob("*metrics*.json"))]+[binding(out/"curves.json")]))
            emit(root,status="CELL_EVALUATED",**c)
        else:
            evaluated=read(out/"evaluated.json");verify(evaluated["seal"])
            for b in evaluated["outputs"]:verify(b)
        current=read(out/"curves.json")["curve_rows"];rows.extend(current)
        if c["study"]=="regional" and c["calibration_method"]=="global":
            prior=read(BASE/"cells"/c["case_id"]/c["base_arm"]/"curves.json")["curve_rows"]
            fields=("proposal_count","true_positive_count","false_positive_count","recovered_event_count","matched_known_positive_count")
            for x,y in zip(current,prior):
                if x["threshold_id"]!=y["threshold_id"] or any(x.get(k)!=y.get(k) for k in fields):raise RuntimeError("Global evaluation replication failed")
            replication.append(dict(case_id=c["case_id"],arm_id=c["arm_id"],all_thresholds_equal=True,metrics_equal=True))
    write_json(root/"all_curves.json",rows);write_tsv(root/"all_curves.tsv",rows)
    write_json(root/"baseline_replication.json",dict(status="PASS",cells=replication))
    write_json(root/"evaluation_complete.json",dict(status="PASS",cells=len(allcells),curve_rows=len(rows)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["preflight","prepare","run","evaluate"]);p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args();globals()[a.command](a.root.resolve())


if __name__=="__main__":main()
