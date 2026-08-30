"""Corrected object-separated NMS sensitivity for compact detector lanes."""
from __future__ import annotations

import csv
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
from .nms_sensitivity import BUDGETS, LANES, RADII, sensitivity_gate


def _tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t"); writer.writeheader(); writer.writerows(rows)


def run_strict_nms_audit(data_root: Path, repo_root: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    partial=output.with_name(output.name+".partial"); partial.mkdir(parents=True,exist_ok=False)
    base=HardRoiAdjudicationConfig.load(repo_root/"examples/spon_ca_burst_hard_roi_adjudication_v1.example.json")
    label_path=data_root/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv"; labels=label_view(load_tsv(label_path),"original","original")
    carrier=np.load(data_root/"Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy",mmap_mode="r",allow_pickle=False)
    values: dict[str,np.ndarray]={"carrier_signed":carrier}; results=[]; failures=[]
    for lane,spec in (("carrier_signed",None),("coherence_w15",(15,0)),("propagation_lag2_w15",(15,2))):
        if spec is not None: values[lane]=_quiet_calibrate(causal_local_correlation_feature(carrier,window_frames=spec[0],lag_frames=spec[1],spatial_sigma_px=2.0,activity_qualified=True),100)
        for radius in RADII:
            config=replace(base,evaluation={**base.evaluation,"nms_distance_px":radius})
            result,detail=_evaluate_map(lane,values[lane],labels,config,label_view_id="original",timing_view_id="original",separated_nms=True)
            result["nms_distance_px"]=radius; result["nms_contract"]="explicit_greedy_euclidean_separation_with_label_free_event_max_tie_breaker"; results.append(result)
            for row in detail: row["nms_distance_px"]=radius
            failures.extend(detail)
        if spec is not None: del values[lane]
    by={(row["feature_id"],row["nms_distance_px"]):row for row in results}; gates={lane:{str(radius):sensitivity_gate(by[(lane,radius)],by[("carrier_signed",radius)]) for radius in RADII} for lane in LANES[1:]}
    primary={lane:gates[lane]["6"]["macro_delta_b20"]>=.03 and gates[lane]["6"]["nonnegative_bursts"]>=3 and not gates[lane]["6"]["catastrophic_drop"] for lane in LANES[1:]}; stable={lane:primary[lane] and all(gates[lane][str(radius)]["passed"] for radius in RADII) for lane in LANES[1:]}
    payload={"schema_version":1,"status":"complete","reason_for_corrected_contract":"maximum-filter plateaus produced adjacent pixel duplicates under the legacy candidate extractor","estimand":"known-positive recall on 79 immutable original occurrences","candidate_unit":"distinct spatial object proposal rather than plateau pixel","label_free_tie_breaker":"within-event maximum of the frozen lane","primary_nms_px":6,"sensitivity_nms_px":[4,8],"budgets":list(BUDGETS),"results":results,"gates":gates,"primary_c3_pattern":primary,"strict_nms_robust_c3":stable,"unmatched_candidates":"unknown_not_negative","decision":{lane:("C3_confirmed_under_strict_object_nms" if stable[lane] else "held_after_strict_object_nms") for lane in LANES[1:]}}
    atomic_json(partial/"strict_nms_sensitivity.json",payload); flat=[]
    for row in results:
        for budget in BUDGETS: flat.append({"feature_id":row["feature_id"],"nms_distance_px":row["nms_distance_px"],"budget":budget,"macro_known_positive_recall":row["macro_recall"][str(budget)],"matched":row["pooled_counts"][str(budget)]["matched"],"labels":row["pooled_counts"][str(budget)]["labels"]})
    _tsv(partial/"strict_nms_budget_recall.tsv",flat); _tsv(partial/"strict_nms_failure_audit.tsv",failures)
    atomic_text(partial/"REPORT.md",f"# Strict object-separated NMS audit\n\nThe legacy maximum-filter extractor admitted adjacent pixels from flat score plateaus. This corrected audit greedily enforces Euclidean separation after deterministic label-free plateau ranking. Strict 4/6/8 px C3 status: `{stable}`. Precision remains unidentified.\n")
    atomic_json(partial/"validation.json",{"status":"passed" if all(stable.values()) else "failed_scientific_gate","radii":list(RADII),"lanes":list(LANES),"labels":79,"legacy_result_overwritten":False}); partial.replace(output); return payload
