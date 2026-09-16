"""Bounded, resumable full-media dispatch for the Gamma sensitivity study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurobench.algorithms.gamma_spatiotemporal import spec_grid
from .spatiotemporal_study import DEFAULT_ROOT, frame_sets, binding, load_protocol, emit
from .two_stencil_campaign import write_json, sha256


def display_contract(root):
    path=root/"display_contract.json"
    if path.exists():return json.loads(path.read_text())
    result={}
    for folder in sorted((root/"datasets").iterdir()):
        raw=np.load(folder/"Raw.npy",mmap_mode="r")
        field=np.load(folder/"Input.npy",mmap_mode="r")
        raw_bounds=list(map(float,np.percentile(raw[::5],[1,99.8])))
        if raw_bounds[1]<=raw_bounds[0]:raw_bounds[1]=raw_bounds[0]+1
        limit=max(1e-6,float(np.percentile(np.abs(field[::5]),99.8)))
        result[folder.name]=dict(Raw=raw_bounds,Input=[-limit,limit],A=[-limit,limit],Score=[-8.,8.])
    write_json(path,dict(cases=result,rule="fixed across all21 settings; Raw/Input scales from shared source visualization only; A shares Input scale; Score symmetric8; display clipping does not change metrics"))
    return json.loads(path.read_text())


def run(root,worker=0,workers=1,limit=None):
    from .spatiotemporal_audit import run_spatiotemporal_audit
    load_protocol(root)
    if not (root/"evaluation_complete.json").exists():raise RuntimeError("Evaluate sealed candidates before media")
    display=display_contract(root)["cases"]
    jobs=[(folder,spec) for folder in sorted((root/"datasets").iterdir()) for spec in spec_grid()]
    count=0
    for ordinal,(folder,spec) in enumerate(jobs):
        if ordinal%workers!=worker:continue
        cell=root/"cells"/folder.name/spec.spec_id
        seal=json.loads((cell/"sealed.json").read_text())
        meta=json.loads((folder/"metadata.json").read_text())
        frames,_,_=frame_sets(meta)
        stages={k:seal["stages"][k]["path"] for k in ("Raw","Input","A","Score")}
        source=dict(stage_sha256={k:seal["stages"][k]["sha256"] for k in stages},
            candidates_sha256=seal["audit_candidates"]["sha256"],expert_occurrences_sha256=sha256(folder/"experts.json"),
            candidate_seal_sha256=sha256(cell/"sealed.json"),display_contract_sha256=sha256(root/"display_contract.json"),
            numeric_stage_bindings={k:seal["stages"][k] for k in ("M","Spread")},
            original_source_offset_xy=meta["original_source_offset_xy"],truth_mode=meta["truth_mode"])
        op=json.loads((cell/"calibration.json").read_text())
        summary=run_spatiotemporal_audit(root/"audits"/folder.name/spec.spec_id,
            stage_paths=stages,signed_stage_keys=("Input","A","Score"),display_limits=display[folder.name],
            source_frames_ui=frames,source_binding=source,operating_point=op,
            numeric_threshold_paths={"all_thresholds":str(cell/"curves.json")},
            candidates_path=cell/"audit_candidates.json",expert_occurrences=folder/"experts.json",fps=50.)
        count+=1
        print(json.dumps(dict(status="AUDIT_CELL_COMPLETE",worker=worker,case=folder.name,spec=spec.spec_id,
            completed=count,summary=summary)),flush=True)
        if limit and count>=limit:return
    write_json(root/f"media_worker_{worker}.json",dict(status="PASS",worker=worker,workers=workers,cell_count=count))


def validate(root):
    from .two_stencil_audit import verify_completed_audit
    rows=[]
    for folder in sorted((root/"datasets").iterdir()):
        for spec in spec_grid():
            audit=root/"audits"/folder.name/spec.spec_id
            verify_completed_audit(audit)
            summary=json.loads((audit/"summary.json").read_text())
            status=json.loads((audit/"status.json").read_text())
            validation=json.loads((audit/"validation.json").read_text())
            if status.get("status")!="complete" or not summary.get("scientific_audit_complete") or validation.get("status")!="passed":
                raise RuntimeError(f"Audit has not passed completion gates: {audit}")
            rows.append(dict(case_id=folder.name,spec_id=spec.spec_id,summary=summary,
                metadata_bindings={name:binding(audit/name) for name in ("summary.json","status.json","run_contract.json",
                    "source_manifest.json","artifact_index.json","llm_context.json","validation.json","inventory.json")}))
    if len(rows)!=357:raise RuntimeError("Expected all357 dataset/configuration audits")
    write_json(root/"audit_complete.json",dict(status="PASS",cell_count=len(rows),audits=rows))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("display","run","validate"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--worker",type=int,default=0);parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.command=="display":display_contract(args.root)
    elif args.command=="validate":validate(args.root)
    else:run(args.root,args.worker,args.workers,args.limit)


if __name__=="__main__":main()
