"""Bounded, disjoint CPU workers for the frozen control-study media audit.

This schedules the existing renderer without changing scores or media settings.
Completed arm checkpoints are verified and resumed by the original adapter.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .control_study import (load_protocol, audit_display_contract, ordered_stage_paths,
                           read_tsv, sha256, verify, binding, write_json, START, STOP, TARGETS)
from .control_audit import run_control_audit


def render_partition(root_string, cpu, arms, contract):
    os.sched_setaffinity(0,{cpu})
    root=Path(root_string)
    results=[]
    for arm in arms:
        folder=root/"arms"/arm["arm_id"]
        seal=json.loads((folder/"sealed.json").read_text())
        for record in seal["files"]:verify(record)
        display={};signed=[]
        for name,group in contract["arm_groups"][arm["arm_id"]].items():
            setting=contract["groups"][group];display[name]=setting["limits"]
            if setting["signed"]:signed.append(name)
        cal=json.loads((folder/"calibration.json").read_text())
        op=next(x for x in cal["operating_points"] if x["target_proposals_per_frame"]==1)
        point=dict(op,arm_id=arm["arm_id"],readout_name=arm["readout"],score_semantics=str(arm),
                   application_source_start_ui=1900,application_source_stop_ui=2359)
        sources=dict(stage_sha256={k:v["sha256"] for k,v in seal["stages"].items()},
                     candidates_sha256=sha256(folder/"q1/candidates.tsv"),
                     expert_occurrences_sha256=sha256(root/"expert_occurrences.tsv"),
                     candidate_seal_sha256=sha256(folder/"sealed.json"),
                     display_contract_sha256=sha256(root/"audit_display_contract.json"))
        result=run_control_audit(root/"audits"/arm["arm_id"],stage_paths=ordered_stage_paths(seal["stages"]),
                 display_limits=display,signed_stage_keys=signed,source_frames_ui=list(range(START,STOP+1)),
                 candidates_path=folder/"q1/candidates.tsv",expert_occurrences=root/"expert_occurrences.tsv",
                 operating_point=point,source_binding=sources,
                 fullnumeric_q_paths={str(q):folder/f"q{q:g}" for q in TARGETS})
        results.append(dict(arm_id=arm["arm_id"],summary=result))
        write_json(root/f"render_worker_cpu{cpu}.json",dict(cpu=cpu,completed=results,
                   assigned_arms=[a["arm_id"] for a in arms],updated_unix=time.time()))
    return results


def run(root, cpus=(25,28,31)):
    root=Path(root).resolve();protocol=load_protocol(root)
    if len(set(cpus)) != len(cpus) or len(cpus)>3:raise ValueError("Use at most three distinct CPUs")
    verify(protocol["labels"])
    seals=json.loads((root/"candidate_seal.json").read_text())
    for record in seals["seals"]:verify(record)
    contract=audit_display_contract(root,protocol["arms"])
    partitions=[protocol["arms"][i::len(cpus)] for i in range(len(cpus))]
    names=[a["arm_id"] for part in partitions for a in part]
    if len(names)!=28 or len(set(names))!=28:raise ValueError("Incomplete or overlapping partition")
    write_json(root/"render_execution.json",dict(kind="disjoint_existing_media_renderer_partitions",
        worker_source=binding(__file__),cpus=list(cpus),numerical_threads_per_worker=1,
        native_threads_per_codec=1,priority="nice19_ionice3",partitions={str(c):[a["arm_id"] for a in p] for c,p in zip(cpus,partitions)},
        protocol=binding(root/"protocol.json"),new_detector_scoring=False,started_unix=time.time()))
    results=[];children=[];streams=[]
    try:
        for cpu in cpus:
            stream=(root/f"render_worker_cpu{cpu}.log").open("a")
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable,"-m",__spec__.name,"--output",str(root),
                                               "--worker-cpu",str(cpu)],stdout=stream,stderr=subprocess.STDOUT))
        failures=[]
        for cpu,child in zip(cpus,children):
            code=child.wait()
            if code:failures.append(dict(cpu=cpu,exit_code=code))
            else:results.extend(json.loads((root/f"render_worker_cpu{cpu}.json").read_text())["completed"])
        if failures:raise RuntimeError(f"Rendering workers failed: {failures}")
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for stream in streams:stream.close()
    results.sort(key=lambda x:names.index(x["arm_id"]))
    if len(results)!=28 or not all(x["summary"]["scientific_audit_complete"] for x in results):
        raise RuntimeError("Incomplete audit")
    write_json(root/"audit_complete.json",dict(status="PASS",arm_count=28,arms=results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker-cpu",type=int)
    args=parser.parse_args()
    if args.worker_cpu is None:run(args.output)
    else:
        root=args.output.resolve();protocol=load_protocol(root)
        execution=json.loads((root/"render_execution.json").read_text());verify(execution["worker_source"])
        assigned=set(execution["partitions"][str(args.worker_cpu)])
        selected=[a for a in protocol["arms"] if a["arm_id"] in assigned]
        contract=json.loads((root/"audit_display_contract.json").read_text())
        render_partition(str(root),args.worker_cpu,selected,contract)
