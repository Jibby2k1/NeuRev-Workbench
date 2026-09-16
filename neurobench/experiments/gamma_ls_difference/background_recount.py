"""Independent candidate-count reconciliation for the background controls.

No reporting, scoring, selection, or metric-calculation helpers are imported.
The sole shared helper verifies files. This script reads sealed positive NMS
rows and native cutoffs, not image arrays or activity-truth files. It runs only
after every physical state is evaluated and the numerical report exists.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import time

from .followup_validate import FileVerifier

DEFAULT_ROOT = Path(__file__).resolve().parents[3]/"Outputs/GammaLSBackground/background_20260915_r1"
SEEDS = (20260916,20260917,20260918)
ARMS = ("mean2of3_n3","mean1_n9","mean4of3_n9")
BUDGETS = (0.,.25,.5,1.,2.,4.,8.,16.)
THRESHOLDS = tuple(f"q{q:g}" for q in BUDGETS)+("all_positive","no_output")
RATE = "false_proposals_per_10000_um2_s"
AREA_PX, PIXEL_UM, FPS, REFERENCE_AREA_PX = 13456,.5,50.,194820
AREA_UM2 = AREA_PX*PIXEL_UM**2
WINDOWS = dict(setup=(65,164),early=(165,264),late=(265,464),
               settled_early=(215,264),settled_late=(315,464))
THRESHOLD_WINDOWS = dict(application=(165,464),late=(265,464))
FACTOR_FIELDS = ("seed","background","normalization","V","S","T")


def require(value,message):
    if not value:
        raise ValueError(message)


def close(actual,expected,message):
    if expected is None or isinstance(expected,bool):
        require(actual is expected,message)
    elif isinstance(expected,str):
        require(actual==expected,message)
    elif isinstance(expected,int):
        require(type(actual) is int and actual==expected,message)
    else:
        require(isinstance(actual,(int,float)) and not isinstance(actual,bool)
                and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=2e-11,abs_tol=2e-12),message)


def physical_id(seed,background,normalization,v,s,t):
    normalization="raw" if s==t==0 else normalization
    if background=="sloped" and normalization=="raw":
        return (f"reference_null_stationary__seed{seed}" if not (v or s or t)
                else f"noise_v{v}_s{s}_t{t}__seed{seed}")
    return f"bg_{background}__{normalization}_v{v}_s{s}_t{t}__seed{seed}"


def logical_id(row):
    if row["normalization"]=="conditioned" and row["S"]==row["T"]==0:
        return (f"logical__bg_{row['background']}__conditioned_"
                f"v{row['V']}_s0_t0__seed{row['seed']}")
    return row["canonical_case_id"]


def validate_design(protocol):
    require(protocol["experiment"]=="gamma_background_conditioned_variance_factorial","Unexpected study")
    require((protocol["expected_cells"],protocol["expected_datasets"],protocol["expected_curve_rows"])
            ==(252,84,5040),"Incorrect physical matrix sizes")
    require(tuple(r["arm_id"] for r in protocol["references"])==ARMS,"Reference order/inventory differs")
    require(protocol["frame_rate_hz"]==FPS and protocol["pixel_size_um"]==PIXEL_UM
            and protocol["eligible_area_px"]==AREA_PX,"Physical calibration differs")
    require(protocol["epochs"]==dict(setup=[65,164],early=[165,264],late=[265,464],application=[165,464])
            and protocol["settled_epochs"]==dict(early=[215,264],late=[315,464]),"Epoch definitions differ")
    expected=set(itertools.product(SEEDS,("sloped","flat"),("raw","conditioned"),(0,1),(0,1),(0,1)))
    cases={r["case_id"]:r for r in protocol["cases"]}
    require(len(cases)==len(protocol["cases"])==84,"Physical case inventory differs")
    physical={tuple(r[k] for k in FACTOR_FIELDS) for r in cases.values()}
    require(physical=={k for k in expected if not(k[2]=="conditioned" and k[4:]==(0,0))},"Physical factor matrix differs")
    for row in cases.values():
        key=tuple(row[k] for k in FACTOR_FIELDS)
        require(row["case_id"]==physical_id(*key) and row["kind"]=="factorial","Physical identity differs")
        require(row["reused_dataset"] is (row["background"]=="sloped" and row["normalization"]=="raw"),"Reuse classification differs")
    logical=protocol["logical_cases"]
    require(len(logical)==96 and {tuple(r[k] for k in FACTOR_FIELDS) for r in logical}==expected,"Logical matrix differs")
    for row in logical:
        require(row["canonical_case_id"]==physical_id(*(row[k] for k in FACTOR_FIELDS)),"Logical alias targets the wrong source")
        require(row.get("alias") is (row["normalization"]=="conditioned" and row["S"]==row["T"]==0),"Alias flag differs")
    require(len({logical_id(r) for r in logical})==96,"Duplicate logical table identity")
    return cases,logical


def check_prefix(rows,first,last):
    """Validate the existing stream, without regenerating peaks or NMS."""
    require(isinstance(rows,list),"Prefix is not a row list")
    ids=set();previous_frame=first;last_score={};last_rank={}
    for row in rows:
        frame=row["source_frame_ui"];score=row["score"];rank=row["candidate_rank_within_frame"]
        require(type(frame) is int and first<=frame<=last and frame>=previous_frame,"Prefix frame/order differs")
        require(isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and score>0,"Prefix score is not finite positive")
        require(type(rank) is int and rank==last_rank.get(frame,0)+1,"Prefix rank sequence differs")
        require(score<=last_score.get(frame,math.inf),"Prefix native score order differs")
        require(row["proposal_id"] not in ids,"Duplicate proposal identity")
        require(all(type(row[k]) is int and 6<=row[k]<122 for k in ("x_px","y_px")),"Proposal outside eligible integer-pixel interior")
        ids.add(row["proposal_id"]);previous_frame=frame;last_score[frame]=score;last_rank[frame]=rank
    return sorted(row["score"] for row in rows)


def above(sorted_scores,tau):
    return 0 if tau is None else len(sorted_scores)-bisect_right(sorted_scores,tau)


def check_plan(rows,setup):
    require(isinstance(rows,list) and tuple(r["threshold_id"] for r in rows)==THRESHOLDS,"Ten-cutoff inventory/order differs")
    sorted_scores=check_prefix(setup,65,164)
    descending=list(reversed(sorted_scores))
    for q,row in zip(BUDGETS,rows):
        budget=math.floor(q*AREA_PX/REFERENCE_AREA_PX*100)
        tau=descending[budget] if len(descending)>budget else 0.
        expected=dict(setup_budget_per_reference_area_frame=q,threshold=tau,
                      setup_proposal_budget=budget,setup_proposal_count=above(sorted_scores,tau))
        require(row["threshold"]==tau,"Setup native order-statistic cutoff differs")
        for key,value in expected.items():close(row[key],value,f"Setup {row['threshold_id']} {key} differs")
        require(row["setup_proposal_count"]<=budget,"Setup allowance exceeded")
    require(rows[-2]["threshold"]==0. and rows[-1]["threshold"] is None,"Endpoint cutoffs differ")
    require(rows[-2]["setup_proposal_count"]==len(setup) and rows[-1]["setup_proposal_count"]==0,"Endpoint setup counts differ")
    require(rows[-2]["setup_proposal_budget"] is None and rows[-1]["setup_proposal_budget"]==0
            and rows[-2]["setup_budget_per_reference_area_frame"] is None
            and rows[-1]["setup_budget_per_reference_area_frame"] is None,"Endpoint allowance definitions differ")
    fixed=rows[3]
    require(fixed["threshold_id"]=="q1" and fixed["setup_proposal_budget"]==6
            and fixed["setup_proposal_count"]<=6,"Primary setup allowance differs")
    return fixed


def q1_counts(prefix,setup,audit,op):
    tau=op["threshold"];target=AREA_PX/REFERENCE_AREA_PX
    expected=[dict(r,threshold_z=tau,target_proposals_per_frame=target,calibration_region_id="global")
              for r in prefix if tau is not None and r["score"]>tau]
    require(audit==expected,"Q1 candidate identity, native score, ordering or cutoff differs")
    counts=Counter({f:0 for f in range(65,465)})
    for row in itertools.chain(setup,prefix):
        if tau is not None and row["score"]>tau:counts[row["source_frame_ui"]]+=1
    require(sum(counts[f] for f in range(65,165))==op["setup_proposal_count"]<=6,"Q1 setup recount differs")
    return counts


def exposure_row(count,lo,hi):
    seconds=(hi-lo+1)/FPS
    return dict(source_start_ui=lo,source_stop_ui=hi,frame_count=hi-lo+1,
                eligible_area_px=AREA_PX,exposure_seconds=seconds,proposal_count=count,
                **{RATE:count*10000/(AREA_UM2*seconds)})


def check_null_metric(row,count,tau,radius):
    require(row["threshold_z"]==tau,"Metric native cutoff differs")
    expected=dict(threshold_z=tau,truth_mode="fully_synthetic",match_radius_px=float(radius),
        application_frame_count=300,exposure_seconds=6.,proposal_count=count,
        proposals_per_frame=count/300,proposals_per_second=count/6.,false_proposals_per_second=count/6.,
        false_positive_count=count,true_positive_count=0,false_negative_count=0,active_region_frame_count=0,
        duplicate_near_active_region_count=0,event_count=0,recovered_event_count=0,first_delay_recovered_denominator=0,
        framewise_sensitivity=None,event_window_coverage=None,first_delay_ms_mean_among_recovered=None,
        first_delay_ms_median_among_recovered=None,unmatched_unknown_count=None,fpr=None,event_precision=None,
        precision=0. if count else None)
    for key,value in expected.items():close(row[key],value,f"Null metric {key} differs")


def check_curves(curves,plan,prefix,case,arm):
    require(curves["event_rows"]==[],"Unexpected null event rows")
    rows=curves["curve_rows"]
    keyed={(r["threshold_id"],float(r["match_radius_px"])):r for r in rows}
    require(len(rows)==len(keyed)==20 and set(keyed)==set(itertools.product(THRESHOLDS,(2.,6.))),"Curves must contain both radii and all ten cutoffs")
    scores=check_prefix(prefix,165,464);counts={}
    for setting in plan:
        count=above(scores,setting["threshold"]);counts[setting["threshold_id"]]=count
        for radius in (2.,6.):
            row=keyed[setting["threshold_id"],radius]
            require(row["case_id"]==case and row["arm_id"]==arm,"Curve identity differs")
            check_null_metric(row,count,setting["threshold"],radius)
            for key,value in setting.items():close(row[key],value,f"Curve cutoff field {key} differs")
            require(row["eligible_area_px"]==AREA_PX,"Curve area differs")
    return counts


class Reader:
    """Stat-guarded metadata/row reads with an explicit no-array/no-truth scope."""
    def __init__(self):
        self.verifier=FileVerifier();self.bindings={}

    def bind(self,path,expected=None):
        path=Path(path)
        require(path.suffix not in (".npy",".npz",".tif",".tiff",".mp4"),"Recount must not read arrays or media")
        require(path.name not in ("experts.json","active.json"),"Recount must not read activity truth")
        record=self.verifier.verify(expected) if expected is not None else self.verifier.binding(path)
        require(Path(record["path"]).resolve()==path.resolve(),"Bound file owner differs")
        self.bindings[record["path"]]=record
        return record

    def read(self,path,expected=None):
        self.bind(path,expected)
        return self.verifier.read_json(path)

    def table(self,out,manifest,name):
        artifacts={r["path"]:r for r in manifest["artifacts"]}
        require(len(artifacts)==len(manifest["artifacts"]),"Duplicate report artifact path")
        record=artifacts[name+".json"]
        rows=self.read(out/(name+".json"),dict(record,path=str(out/record["path"])))
        require(len(rows)==manifest["table_row_counts"][name],"Report row-count declaration differs")
        tsv=out/(name+".tsv");record=artifacts[name+".tsv"]
        self.bind(tsv,dict(record,path=str(out/record["path"])))
        # Check the TSV export against its bound JSON; floats use Python's text repr.
        with tsv.open(newline="") as stream:
            exported=list(csv.DictReader(stream,delimiter="\t"))
        require(len(exported)==len(rows),"Report JSON/TSV row counts differ")
        for left,right in zip(rows,exported):
            for key,value in left.items():
                if isinstance(value,(dict,list)):continue
                require(key in right and right[key]==("" if value is None else str(value)),f"Report JSON/TSV differs: {name}/{key}")
        self.verifier.assert_unchanged()
        return rows


def _unique(rows,key,expected_keys,label):
    indexed={key(row):row for row in rows}
    require(len(indexed)==len(rows) and set(indexed)==set(expected_keys),f"{label} identity inventory differs")
    return indexed


def _fields(row,expected,label):
    for key,value in expected.items():close(row[key],value,f"{label}: {key} differs")


def verify_report_tables(tables,cases,logical,states):
    """Compare independently counted physical windows, aliases and rate pairs."""
    physical={};thresholds={}
    for (case,arm),state in states.items():
        factors={k:cases[case][k] for k in FACTOR_FIELDS}
        for window,(lo,hi) in WINDOWS.items():
            count=sum(state["frames"][f] for f in range(lo,hi+1))
            physical[case,arm,window]=dict(case_id=case,arm_id=arm,case_kind="factorial",**factors,
                window=window,epoch=window,threshold=state["op"]["threshold"],scale_floor=state["op"]["scale_floor"],
                **exposure_row(count,lo,hi))
        for setting in state["plan"]:
            for window,(lo,hi) in THRESHOLD_WINDOWS.items():
                count=state["threshold_counts"][setting["threshold_id"],window]
                thresholds[case,arm,setting["threshold_id"],window]=dict(case_id=case,arm_id=arm,**factors,
                    threshold_id=setting["threshold_id"],threshold=setting["threshold"],window=window,
                    false_positive_count=count,**exposure_row(count,lo,hi))
    pkey=lambda r:(r["case_id"],r["arm_id"],r["window"])
    tkey=lambda r:(r["case_id"],r["arm_id"],r["threshold_id"],r["window"])
    for name,wanted,key in (("physical_epochs",physical,pkey),("physical_threshold_windows",thresholds,tkey)):
        actual=_unique(tables[name],key,wanted,name)
        for k,row in wanted.items():_fields(actual[k],row,name)
    lexpect={};texpect={};panel_values={}
    for request in logical:
        canonical=request["canonical_case_id"];lid=logical_id(request)
        update=dict(case_id=lid,canonical_case_id=canonical,normalization=request["normalization"],
                    is_alias=lid!=canonical,source_reused=cases[canonical]["reused_dataset"])
        for arm in ARMS:
            for window in WINDOWS:
                row=dict(physical[canonical,arm,window],**update)
                lexpect[lid,arm,window]=row
                panel_values[request["seed"],arm,window,request["V"],request["S"],request["T"],request["background"],request["normalization"]]=row[RATE]
            for threshold,window in itertools.product(THRESHOLDS,THRESHOLD_WINDOWS):
                texpect[lid,arm,threshold,window]=dict(thresholds[canonical,arm,threshold,window],**update)
    for name,wanted,key in (("logical_epochs",lexpect,pkey),("logical_threshold_windows",texpect,tkey)):
        actual=_unique(tables[name],key,wanted,name)
        for k,row in wanted.items():_fields(actual[k],row,name)
    contrasts={}
    for seed,arm,window,v,s,t in itertools.product(SEEDS,ARMS,WINDOWS,(0,1),(0,1),(0,1)):
        key=(seed,arm,window,v,s,t)
        values={(bg,n):panel_values[key+(bg,n)] for bg,n in itertools.product(("flat","sloped"),("raw","conditioned"))}
        for norm in ("raw","conditioned"):
            contrasts[key+("flat_minus_sloped",norm)]=values["flat",norm]-values["sloped",norm]
        for bg in ("flat","sloped"):
            contrasts[key+("conditioned_minus_raw",bg)]=values[bg,"conditioned"]-values[bg,"raw"]
        contrasts[key+("background_by_normalization","difference_of_differences")]=(
            (values["flat","conditioned"]-values["flat","raw"])
            -(values["sloped","conditioned"]-values["sloped","raw"]))
    rate_rows=[r for r in tables["paired_effects"] if r["metric"]==RATE]
    ckey=lambda r:tuple(r[k] for k in ("seed","arm_id","window","V","S","T","contrast","condition"))
    actual=_unique(rate_rows,ckey,contrasts,"Paired rate effects")
    for k,value in contrasts.items():
        _fields(actual[k],dict(difference=value,replicate_unit="paired seed",logical_aliases_are_not_replicates=True),"Paired rate effects")
    return dict(physical_epoch_rows=len(physical),logical_epoch_rows=len(lexpect),
        physical_threshold_window_rows=len(thresholds),logical_threshold_window_rows=len(texpect),
        paired_rate_contrasts=len(contrasts),rate_contrast_windows=list(WINDOWS),
        alias_logical_cells=sum(r["alias"] for r in logical)*len(ARMS),
        primary_late_counts=[physical[k] for k in physical if k[-1]=="late"])


def _recount(root,reader):
    pre=reader.read(root/"preflight.json");protocol=reader.read(root/"protocol.json")
    require(pre["status"]=="PASS" and pre["protocol_sha256"]==reader.bind(root/"protocol.json")["sha256"],"Frozen protocol changed")
    cases,logical=validate_design(protocol)
    require(reader.read(root/"computation_complete.json")==dict(status="PASS",cells=252),"Scoring is incomplete")
    require(reader.read(root/"evaluation_complete.json")==dict(status="PASS",cells=252,curve_rows=5040),"Evaluation is incomplete")
    inventory=reader.read(root/"all_scoring_seals.json")
    require(inventory["status"]=="PASS" and inventory["cells"]==252,"All-seal inventory incomplete")
    reader.bind(root/"protocol.json",inventory["protocol"])
    wanted={(root/"cells"/case/arm/"sealed.json").resolve() for case in cases for arm in ARMS}
    require(len(inventory["seals"])==252 and {Path(b["path"]).resolve() for b in inventory["seals"]}==wanted,"All-seal inventory differs")
    for b in inventory["seals"]:reader.bind(b["path"],b)
    for b in protocol["code_bindings"]:reader.bind(b["path"],b)
    states={};all_rows=[];q1_rows=0;prefix_rows=0
    for case in cases:
        for arm in ARMS:
            folder=root/"cells"/case/arm;seal=reader.read(folder/"sealed.json")
            require(seal["status"]=="SEALED_BEFORE_ACTIVITY_TRUTH_JOIN","Invalid scoring seal")
            metadata=[b for b in seal["dataset_bindings"] if Path(b["path"]).name=="metadata.json"]
            require(len(metadata)==1,"Ambiguous metadata source")
            meta=reader.read(metadata[0]["path"],metadata[0])
            require(meta["truth_mode"]=="fully_synthetic" and meta["event_count"]==0
                    and meta["source_frames_ui"]==list(range(1,465))
                    and meta["setup_source_frames_ui"]==list(range(65,165))
                    and meta["application_source_frames_ui"]==list(range(165,465))
                    and meta["evaluation_box_yxyx"]==[49,49,177,177],"Declared null/frame geometry differs")
            evaluated=reader.read(folder/"evaluated.json")
            require(evaluated["status"]=="PASS","Cell evaluation incomplete")
            reader.bind(folder/"sealed.json",evaluated["seal"])
            outputs={Path(b["path"]).resolve():b for b in evaluated["outputs"]}
            plan=reader.read(folder/"threshold_plan.json",seal["threshold_plan"])
            op=reader.read(folder/"calibration.json",seal["calibration"])
            setup=reader.read(seal["setup_prefix"]["path"],seal["setup_prefix"])
            fixed=check_plan(plan,setup)
            _fields(op,fixed,"Primary calibration")
            require(op["threshold"]==fixed["threshold"]==op["threshold_z"],"Q1 native cutoff differs from setup plan")
            _fields(op,dict(threshold_z=fixed["threshold"],threshold_frozen_from_calibration_only=True,
                eligible_area_px=AREA_PX,reference_area_px=REFERENCE_AREA_PX,target_proposals_per_frame=AREA_PX/REFERENCE_AREA_PX,
                application_source_start_ui=165,application_source_stop_ui=464,application_frame_count=300),"Primary calibration")
            require(op["setup_source_frames_ui"]==list(range(65,165)) and math.isfinite(op["scale_floor"]) and op["scale_floor"]>0,"Setup/floor differs")
            prefix=reader.read(seal["prefix"]["path"],seal["prefix"])
            audit=reader.read(folder/"audit_candidates.json",seal["audit_candidates"])
            curves=reader.read(folder/"curves.json",outputs[(folder/"curves.json").resolve()])
            check_curves(curves,plan,prefix,case,arm)
            counts=q1_counts(prefix,setup,audit,op)
            for radius in (2,6):
                path=folder/f"operating_metrics_r{radius}.json"
                metrics=reader.read(path,outputs[path.resolve()])
                check_null_metric(metrics["summary"],len(audit),fixed["threshold"],radius)
                require(metrics["event_rows"]==[],"Unexpected null events")
                frame_rows=_unique(metrics["frame_rows"],lambda r:r["source_frame_ui"],range(165,465),"Q1 frame metrics")
                for f,row in frame_rows.items():
                    _fields(row,dict(proposal_count=counts[f],false_positive_count=counts[f],true_positive_count=0,
                                    false_negative_count=0,active_region_count=0,unmatched_unknown_count=None),"Q1 frame metrics")
            threshold_counts={}
            for window,(lo,hi) in THRESHOLD_WINDOWS.items():
                scores=sorted(r["score"] for r in prefix if lo<=r["source_frame_ui"]<=hi)
                for setting in plan:threshold_counts[setting["threshold_id"],window]=above(scores,setting["threshold"])
            states[case,arm]=dict(op=op,plan=plan,frames=counts,threshold_counts=threshold_counts)
            all_rows.extend(curves["curve_rows"]);q1_rows+=len(audit);prefix_rows+=len(prefix)
            del setup,prefix,audit,curves
        print(dict(status="INDEPENDENT_RECOUNT_PROGRESS",cells=len(states),total_cells=252),flush=True)
    aggregate=reader.read(root/"all_curves.json")
    require(len(aggregate)==5040 and aggregate==all_rows,"All-curves aggregate differs from recounted cells")
    out=root/"report";manifest=reader.read(out/"manifest.json")
    require(manifest["numerical_complete"] is True and manifest["physical_cell_count"]==252
            and manifest["logical_cell_count"]==288,"Numerical report incomplete")
    for b in manifest["reporting_code"]:reader.bind(b["path"],b)
    names=("physical_epochs","logical_epochs","physical_threshold_windows","logical_threshold_windows","paired_effects")
    tables={name:reader.table(out,manifest,name) for name in names}
    checks=verify_report_tables(tables,cases,logical,states)
    report_curves=reader.table(out,manifest,"curves")
    require(report_curves==aggregate,"Report curve rows differ from recounted aggregate")
    # Later media completion may change report status/manifest but not these tables.
    observed=reader.bindings.pop(str((out/"manifest.json").resolve()))
    return dict(schema_version=1,status="PASS",cells=252,curve_rows=5040,curve_rows_recounted=5040,q1_candidate_tables=252,
        q1_candidate_rows=q1_rows,positive_prefix_rows=prefix_rows,setup_plan_rows=2520,
        setup_q1_maximum_allowed=6,operating_metric_objects=504,
        all_curves=reader.bind(root/"all_curves.json"),protocol=reader.bind(root/"protocol.json"),
        evaluation=reader.bind(root/"evaluation_complete.json"),
        report_checks=checks,table_names_verified=list(names)+["curves"],
        report_manifest_observed_sha256=observed["sha256"],
        report_manifest_status_may_change_without_numeric_changes=True,
        physical_area=dict(eligible_area_px=AREA_PX,pixel_size_um=PIXEL_UM,eligible_area_um2=AREA_UM2,frame_rate_hz=FPS),
        denominator_examples={name:dict(frames=hi-lo+1,seconds=(hi-lo+1)/FPS,
            per_seed_um2_seconds=AREA_UM2*(hi-lo+1)/FPS,three_seed_um2_seconds=3*AREA_UM2*(hi-lo+1)/FPS)
            for name,(lo,hi) in {**WINDOWS,**THRESHOLD_WINDOWS}.items()},
        semantic_boundaries=["All counts are existing strict-native-cutoff readouts of sealed positive NMS proposals; no NMS or detector is rerun.",
            "Only synthetic source-free counts are checked. No event/activity truth file or image array is parsed; the separate full source/media validator covers those artifacts.",
            "Every unmatched proposal is false under the declared null generator, not a biological false-positive adjudication. Sensitivity is undefined; no false-alarm probability or true-negative unit is inferred.",
            "Three paired seeds are the replicates. Thirty-six logical cells are unit-gain aliases, not additional source realizations.",
            "Rate/count contrasts are independently checked; reporter stage-moment and score-quantile statistics are outside this recount's scope.",
            "Full application uses six seconds per seed; late uses four, early two, settled late three, and settled early one. V=1 full application mixes pre/post amplitude-step epochs.",
            "Numerical recount PASS does not establish media completion or biological/control performance."])


def recount(root=DEFAULT_ROOT):
    root=Path(root).resolve();out=root/"validation"/"independent_numeric_check.json"
    require(not (root/"completion_manifest.json").exists(),"Completed study is immutable")
    require(not out.exists(),"Preserve the existing independent numerical receipt")
    reader=Reader();started=time.monotonic()
    code=[reader.bind(Path(__file__)),reader.bind(Path(__file__).with_name("followup_validate.py")),
          reader.bind(Path(__file__).resolve().parents[3]/"tests/test_background_recount.py")]
    try:
        result=_recount(root,reader)
        reader.verifier.assert_unchanged()
        result.update(checked_utc=datetime.now(timezone.utc).isoformat(),code_binding=code[0],validation_code=code,
            source_bindings=sorted(reader.bindings.values(),key=lambda b:b["path"]),
            unique_files_hashed=reader.verifier.unique_file_count,bytes_hashed=reader.verifier.bytes_hashed,
            runtime_seconds=time.monotonic()-started)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("x") as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        return result
    except Exception as error:
        out.parent.mkdir(parents=True,exist_ok=True)
        failure=out.with_name("independent_numeric_failure.json")
        failure.write_text(json.dumps(dict(status="FAIL",error_type=type(error).__name__,error=str(error),
            validation_code=code,runtime_seconds=time.monotonic()-started),indent=2,sort_keys=True)+"\n")
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args();result=recount(args.root)
    print(dict(status=result["status"],cells=result["cells"],curve_rows=result["curve_rows"],
               q1_candidate_rows=result["q1_candidate_rows"]),flush=True)
