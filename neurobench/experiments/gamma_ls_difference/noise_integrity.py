"""Post-scoring equality checks without outcome reads for the noise factorial.

Whole-file hashes verify provenance; numerical comparisons inspect only UI1–264.
No activity, expert, curve or metric JSON is parsed, and no detector is rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .followup_validate import FileVerifier, require


SEEDS = (20260916, 20260917, 20260918)
ARMS = ("mean2of3_n3", "mean1_n9", "mean4of3_n9")
STAGES = ("Input", "A", "M", "Spread", "C", "Z")
EPOCHS = dict(setup=[65,164], early=[165,264], late=[265,464], application=[165,464])
PREFIX_STOP = 264


def protocol_matrix(protocol):
    require(protocol.get("experiment") == "gamma_noise_variance_spatial_temporal_factorial",
            "Expected frozen noise factorial")
    require(protocol.get("expected_cells") == 81 and protocol.get("expected_datasets") == 27,
            "Expected the full 81-cell, 27-dataset study")
    require(protocol.get("epochs") == EPOCHS and protocol.get("frame_rate_hz") == 50
            and protocol.get("pixel_size_um") == .5 and protocol.get("eligible_area_px") == 13456,
            "Production epochs or physical geometry differ")
    require([r["arm_id"] for r in protocol["references"]] == list(ARMS), "Fixed reference choices differ")
    cases = protocol["cases"]
    require(len(cases) == 27 and len({c["case_id"] for c in cases}) == 27, "Dataset matrix differs")
    factorial = [c for c in cases if c["kind"] == "factorial"]
    legacy = [c for c in cases if c["kind"] == "legacy"]
    expected = {(seed,v,s,t) for seed in SEEDS for v in (0,1) for s in (0,1) for t in (0,1)}
    require(len(factorial) == 24 and {(c["seed"],c["V"],c["S"],c["T"]) for c in factorial} == expected,
            "Incomplete paired factorial")
    require(len(legacy) == 3 and {c["seed"] for c in legacy} == set(SEEDS), "Legacy bridge matrix differs")
    for c in cases:
        if c["kind"] == "legacy":
            identity = f"reference_null_variance_correlation__seed{c['seed']}"
            require(all(c[k] is None for k in ("V","S","T")), "Legacy cannot be a factorial level")
            reuse = True
        else:
            v,s,t = (c[k] for k in ("V","S","T"))
            reuse = (v,s,t) == (0,0,0)
            identity = (f"reference_null_stationary__seed{c['seed']}" if reuse else
                        f"noise_v{v}_s{s}_t{t}__seed{c['seed']}")
        require(c["case_id"] == identity and c["reused_dataset"] is reuse,
                "Case identity or reuse semantics differ")
    return [(c,arm) for c in cases for arm in ARMS]


def compare_prefix_arrays(left, right, stop, *, chunk_frames=8):
    """Bitwise comparison of finite float32 TYX prefixes, without future reads."""
    require(type(stop) is int and stop > 0 and type(chunk_frames) is int and chunk_frames > 0,
            "Invalid prefix/chunk length")
    require(left.shape == right.shape and len(left.shape) == 3 and left.shape[0] >= stop,
            "Paired array shapes differ or omit required frames")
    require(left.dtype == right.dtype == np.dtype("float32"), "Expected paired float32 stages")
    checked = 0
    for start in range(0, stop, chunk_frames):
        end = min(stop, start+chunk_frames)
        a,b = left[start:end],right[start:end]
        require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite compared stage prefix")
        require(a.tobytes(order="C") == b.tobytes(order="C"),
                f"Paired stage prefix differs at NumPy chunk [{start},{end})")
        checked += a.nbytes
    return dict(frames_compared=stop, shape_tyx=list(left.shape), bytes_compared_per_side=checked,
                bitwise_equal=True, finite=True)


def compare_prefix_rows(left, right, *, first_ui, last_ui, allowed_stop_ui):
    """Exact ordered full-record comparison; no identifier/field normalization."""
    def selected(rows):
        require(isinstance(rows,list), "Candidate prefix must be a record list")
        result=[]
        for row in rows:
            frame=row.get("source_frame_ui")
            require(type(frame) is int and first_ui <= frame <= allowed_stop_ui,
                    "Candidate source frame falls outside its declared stream")
            if frame <= last_ui:
                result.append(row)
        return result
    a,b=selected(left),selected(right)
    require(a == b, "Paired candidate prefix records differ")
    canonical=json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return dict(source_interval_ui=[first_ui,last_ui],record_count=len(a),exact_ordered_records_equal=True,
                compared_records_sha256=hashlib.sha256(canonical).hexdigest(),fields_removed=[])


def _read_source_json(verifier, path):
    name=Path(path).name
    require(name not in ("experts.json","active.json","curves.json","all_curves.json")
            and "metrics" not in name and not name.startswith("evaluated"),
            "Truth and evaluation JSON parsing is forbidden in this integrity check")
    return verifier.read_json(path)


def _calibration_group(plans, floors):
    require(len(plans) == len(floors) == 9, "Expected all nine conditions in each calibration group")
    require(all(p == plans[0] for p in plans) and all(f == floors[0] for f in floors),
            "Paired setup calibration plans or floors differ")
    require(isinstance(floors[0],(int,float)) and math.isfinite(floors[0]) and floors[0] > 0,
            "Invalid scale floor")
    return floors[0]


def check(root):
    root=Path(root).resolve();output=root/"paired_variance_prefix_check.json"
    require(not (root/"completion_manifest.json").exists(), "Preserve completed study")
    require(not output.exists(), "Preserve existing integrity receipt")
    verifier=FileVerifier();records={};started=time.monotonic()

    def bind(path):
        value=verifier.binding(path);records[value["path"]]=value;return value

    def verify(value):
        result=verifier.verify(value);records[result["path"]]=result;return result

    def read(path):
        bind(path);return _read_source_json(verifier,path)

    protocol=read(root/"protocol.json");matrix=protocol_matrix(protocol)
    require(all((root/"cells"/c["case_id"]/arm/"sealed.json").is_file() for c,arm in matrix),
            "Seal all 81 scoring states before this check")
    evaluation_started = ((root/"evaluation_complete.json").exists()
        or any((root/"cells"/c["case_id"]/arm/"evaluated.json").exists() for c,arm in matrix))
    require(read(root/"computation_complete.json") == dict(status="PASS",cells=81),
            "Complete all scoring and paired setup checks first")
    preflight=read(root/"preflight.json")
    require(preflight["status"] == "PASS" and preflight["protocol_sha256"] == bind(root/"protocol.json")["sha256"],
            "Frozen protocol changed")
    for b in protocol["code_bindings"]+protocol["kernel_bindings"]+protocol["baseline_sources"]:
        verify(b)
    for key in ("baseline_completion","baseline_protocol","paper_protocol"):
        verify(protocol[key])
    code=[bind(Path(__file__)),bind(Path(__file__).with_name("followup_validate.py")),
          bind(Path(__file__).resolve().parents[3]/"tests/test_noise_integrity.py")]
    seals={};calibrations={};plans={};metadata={};raw_sources={}
    for c,arm in matrix:
        case=c["case_id"];folder=root/"cells"/case/arm
        seal=read(folder/"sealed.json")
        require(seal.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", "Invalid scoring seal")
        for key in ("prefix","setup_prefix","audit_candidates","threshold_plan","calibration"):
            verify(seal[key])
        require(set(STAGES)|{"Raw","Score"} <= set(seal["stages"]), "Missing sealed scientific stages")
        for b in list(seal["stages"].values())+seal["dataset_bindings"]:
            verify(b)
        require(seal["stages"]["Score"]["sha256"] == seal["stages"]["Z"]["sha256"], "Score is not native Z")
        op=read(seal["calibration"]["path"]);plan=read(seal["threshold_plan"]["path"])
        require(op["threshold_frozen_from_calibration_only"] is True
                and op["setup_source_frames_ui"] == list(range(65,165))
                and op["application_source_start_ui"] == 165 and op["application_source_stop_ui"] == 464
                and op["application_frame_count"] == 300 and op["eligible_area_px"] == 13456,
                "Calibration/application domain differs")
        seals[case,arm],calibrations[case,arm],plans[case,arm]=seal,op,plan
        if case not in metadata:
            dataset=root/"datasets"/case;authority={Path(b["path"]).resolve():b for b in seal["dataset_bindings"]}
            for name in ("metadata.json","raw_source.npy"):
                require((dataset/name).resolve() in authority, "Dataset source missing from seal")
            meta=read(dataset/"metadata.json")
            require(meta["source_frames_ui"] == list(range(1,465))
                    and meta["setup_source_frames_ui"] == list(range(65,165))
                    and meta["application_source_frames_ui"] == list(range(165,465))
                    and meta["shape_tyx"] == [464,226,226]
                    and meta["evaluation_box_yxyx"] == [49,49,177,177], "Dataset frame/coordinate binding differs")
            metadata[case]=meta;raw_sources[case]=authority[(dataset/"raw_source.npy").resolve()]
    fixed=[]
    for c in protocol["cases"]:
        entries=[seals[c["case_id"],arm]["stages"]["A"] for arm in ARMS]
        require(len({b["sha256"] for b in entries}) == 1, "Target A changed across references")
        fixed.append(dict(case_id=c["case_id"],references=list(ARMS),A_bindings=entries,exact_full_file_hash_equal=True))
    groups=[]
    for seed in SEEDS:
        cases=[c["case_id"] for c in protocol["cases"] if c["seed"] == seed]
        for arm in ARMS:
            floor=_calibration_group([plans[case,arm] for case in cases],
                                     [calibrations[case,arm]["scale_floor"] for case in cases])
            groups.append(dict(seed=seed,arm_id=arm,case_ids=cases,scale_floor=floor,
                exact_plans_equal=True,exact_floors_equal=True,
                calibration_bindings=[seals[case,arm]["calibration"] for case in cases],
                threshold_plan_bindings=[seals[case,arm]["threshold_plan"] for case in cases]))
    raw_pairs=[];stage_pairs=[]
    factorial={(c["seed"],c["V"],c["S"],c["T"]):c["case_id"]
               for c in protocol["cases"] if c["kind"] == "factorial"}

    def arrays(a,b,shape):
        left=np.load(a["path"],mmap_mode="r",allow_pickle=False)
        right=np.load(b["path"],mmap_mode="r",allow_pickle=False)
        require(left.shape == right.shape == shape, "Production stage shape differs")
        result=compare_prefix_arrays(left,right,PREFIX_STOP,chunk_frames=8)
        del left,right
        return dict(left_binding=a,right_binding=b,**result)

    for seed in SEEDS:
        for s in (0,1):
            for t in (0,1):
                low,high=factorial[seed,0,s,t],factorial[seed,1,s,t]
                common=dict(seed=seed,S=s,T=t,V0_case_id=low,V1_case_id=high,source_interval_ui=[1,PREFIX_STOP])
                raw_pairs.append(dict(common,**arrays(raw_sources[low],raw_sources[high],(464,226,226))))
                for arm in ARMS:
                    left,right=seals[low,arm],seals[high,arm]
                    stages={key:arrays(left["stages"][key],right["stages"][key],(464,128,128)) for key in STAGES}
                    prefixes={}
                    for key,first,last,stop in (("setup_prefix",65,164,164),("prefix",165,264,464),("audit_candidates",165,264,464)):
                        a,b=read(left[key]["path"]),read(right[key]["path"])
                        prefixes[key]=dict(left_binding=left[key],right_binding=right[key],
                            **compare_prefix_rows(a,b,first_ui=first,last_ui=last,allowed_stop_ui=stop))
                        del a,b
                    stage_pairs.append(dict(common,arm_id=arm,stages=stages,candidate_streams=prefixes))
                print(dict(status="PAIRED_PREFIX_VERIFIED",raw_pairs=len(raw_pairs),stage_pairs=len(stage_pairs),total_stage_pairs=36),flush=True)
    require((len(fixed),len(groups),len(raw_pairs),len(stage_pairs)) == (27,9,12,36), "Integrity inventory differs")
    verifier.assert_unchanged()
    result=dict(schema_version=1,status="PASS",checked_utc=datetime.now(timezone.utc).isoformat(),
        cells=81,datasets=27,fixed_target_checks=fixed,calibration_groups=groups,
        raw_pairs=raw_pairs,stage_pairs=stage_pairs,source_interval_ui=[1,264],numpy_interval=[0,264],
        compared_stage_names=list(STAGES),code_bindings=code,bindings=sorted(records.values(),key=lambda b:b["path"]),
        all_scoring_seals_preceded_check=True,activity_truth_or_metrics_parsed=False,
        evaluation_started_before_check=evaluation_started,
        evaluation_order_scope="Existence-only observation at check start; no outcome file is parsed and concurrent evaluation is permitted.",
        detector_rerun=False,empirical_refit=False,
        unique_files_hashed=verifier.unique_file_count,bytes_hashed=verifier.bytes_hashed,
        cached_binding_checks=verifier.cache_hits,runtime_seconds=time.monotonic()-started,
        comparison_scope="Every compared numeric prefix is finite float32 and byte-identical through source UI264. No numerical array slice after UI264 is inspected for equality; full-file SHA reads separately establish integrity. All full ordered candidate fields and IDs are preserved without normalization. Setup streams cover UI65–164, positive/application and q1 streams are compared on UI165–264. No detector, assignment, event or sensitivity result is computed.")
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();result=check(args.root)
    print(dict(status=result["status"],raw_pairs=len(result["raw_pairs"]),stage_pairs=len(result["stage_pairs"])),flush=True)


if __name__ == "__main__":
    main()
