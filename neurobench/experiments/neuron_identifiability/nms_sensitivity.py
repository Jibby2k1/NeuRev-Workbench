"""Frozen 4/6/8 px NMS sensitivity for the compact quantitative panel."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.algorithms.scientific_feature_audit import causal_local_correlation_feature
from neurobench.experiments.hard_roi_adjudication.adjudication import label_view, load_tsv
from neurobench.experiments.hard_roi_adjudication.config import HardRoiAdjudicationConfig
from neurobench.experiments.hard_roi_adjudication.reevaluate import _evaluate_map
from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import _quiet_calibrate

from .contracts import atomic_json, atomic_text


RADII=(4,6,8); BUDGETS=(20,40,58,80,100); LANES=("carrier_signed","coherence_w15","propagation_lag2_w15")


def sensitivity_gate(lane: dict[str,Any], carrier: dict[str,Any]) -> dict[str,Any]:
    delta=float(lane["macro_recall"]["20"])-float(carrier["macro_recall"]["20"])
    burst=[float(a["budgets"]["20"]["recall"])-float(b["budgets"]["20"]["recall"]) for a,b in zip(lane["folds"],carrier["folds"],strict=True)]
    return {"macro_delta_b20":delta,"burst_deltas_b20":burst,"positive_aggregate":delta>0,"nonnegative_bursts":sum(x>=0 for x in burst),"catastrophic_drop":min(burst)<-0.05,"passed":delta>0 and sum(x>=0 for x in burst)>=3 and min(burst)>=-0.05}


def _tsv(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t"); writer.writeheader(); writer.writerows(rows)


def run_nms_sensitivity(data_root:Path, repo_root:Path, output:Path)->dict[str,Any]:
    output.mkdir(parents=True,exist_ok=True)
    base=HardRoiAdjudicationConfig.load(repo_root/"examples/spon_ca_burst_hard_roi_adjudication_v1.example.json")
    carrier_path=data_root/"Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy"
    label_path=data_root/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv"
    labels=label_view(load_tsv(label_path),"original","original")
    carrier=np.load(carrier_path,mmap_mode="r",allow_pickle=False)
    values:dict[str,np.ndarray]={"carrier_signed":carrier}
    results=[]; failures=[]
    for lane,spec in (("carrier_signed",None),("coherence_w15",(15,0)),("propagation_lag2_w15",(15,2))):
        if spec is not None:
            values[lane]=_quiet_calibrate(causal_local_correlation_feature(carrier,window_frames=spec[0],lag_frames=spec[1],spatial_sigma_px=2.0,activity_qualified=True),100)
        for radius in RADII:
            config=replace(base,evaluation={**base.evaluation,"nms_distance_px":radius})
            result,detail=_evaluate_map(lane,values[lane],labels,config,label_view_id="original",timing_view_id="original")
            result["nms_distance_px"]=radius; results.append(result)
            for row in detail: row["nms_distance_px"]=radius
            failures.extend(detail)
        if spec is not None: del values[lane]
    by={(r["feature_id"],r["nms_distance_px"]):r for r in results}
    frozen_final=json.loads((data_root/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v1/metrics.json").read_text())
    if Path(frozen_final["adjudication_tsv"]).resolve()!=label_path.resolve(): raise ValueError("frozen final rescore uses another adjudication table")
    expected={r["feature_id"]:r for r in frozen_final["results"] if r["label_view"]=="original" and r["timing_view"]=="original" and r["feature_id"] in LANES}
    reproduction={lane:max(abs(float(by[(lane,6)]["macro_recall"][str(b)])-float(expected[lane]["macro_recall"][str(b)])) for b in BUDGETS) for lane in LANES}
    if max(reproduction.values())>1e-12: raise ValueError(f"6 px reproduction failed: {reproduction}")
    gates={lane:{str(radius):sensitivity_gate(by[(lane,radius)],by[("carrier_signed",radius)]) for radius in RADII} for lane in LANES[1:]}
    primary_c3={lane:gates[lane]["6"]["macro_delta_b20"]>=.03 and gates[lane]["6"]["nonnegative_bursts"]>=3 and not gates[lane]["6"]["catastrophic_drop"] for lane in LANES[1:]}
    stable={lane:primary_c3[lane] and all(gates[lane][str(r)]["passed"] for r in RADII) for lane in LANES[1:]}
    payload={"schema_version":1,"status":"complete","estimand":"known-positive recall; 79-row final_v1 immutable original sites and original timing","primary_nms_px":6,"sensitivity_nms_px":[4,8],"match_radius_px":6,"scientific_rationale":{"user_reported_neuron_diameter_micrometers":[3,7],"primary_radius_px":6,"physical_pixel_calibration_found_in_source":False,"interpretation":"Six pixels remains the prespecified biologically motivated primary rule. TIFF metadata contains no physical pixel scale, so no pixel-to-micrometer conversion is asserted."},"frozen_contract":{"adjudication_tsv":str(label_path),"reproduction_source":str(data_root/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v1/metrics.json"),"features":list(LANES),"budgets":list(BUDGETS),"no_parameter_selection":True,"unmatched_candidates":"unknown_not_negative"},"reproduction_max_abs_error":reproduction,"results":results,"gates":gates,"primary_c3_pattern":primary_c3,"nms_robust_c3":stable,"decision":{lane:("C3_confirmed_compact_utility" if stable[lane] else "C2_or_held_due_nms_sensitivity") for lane in LANES[1:]}}
    atomic_json(output/"nms_sensitivity.json",payload)
    flat=[]
    for row in results:
        for budget in BUDGETS: flat.append({"feature_id":row["feature_id"],"nms_distance_px":row["nms_distance_px"],"budget":budget,"macro_known_positive_recall":row["macro_recall"][str(budget)],"matched":row["pooled_counts"][str(budget)]["matched"],"labels":row["pooled_counts"][str(budget)]["labels"]})
    _tsv(output/"nms_budget_recall.tsv",flat); _tsv(output/"nms_failure_audit.tsv",failures)
    atomic_text(output/"NMS_CONTRACT.md","# Frozen NMS sensitivity contract\n\nSix pixels is primary, motivated by the user-reported 3-7 micrometer neuron-size literature range. Four and eight pixels are sensitivity checks only and cannot be selected after evaluation. A lane survives when its budget-20 aggregate gain is positive at every radius, at least three of four burst deltas are nonnegative at every radius, no burst delta is below -0.05, and the primary six-pixel result meets the +0.03 C3 threshold. The TIFF contains no physical pixel-size metadata, so the physical conversion remains unverified.\n")
    return payload
