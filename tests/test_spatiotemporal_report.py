"""Small postprocessing checks; no study data, convolution, or media audit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import spatiotemporal_report as report


def _curve(**updates):
    row=dict(truth_mode="fully_synthetic",template_id="fast",spec_id="s",threshold_z=1.,
        true_positive_count=2,false_positive_count=1,false_negative_count=2,
        proposal_count=3,application_frame_count=10,event_count=2,recovered_event_count=1,
        first_delay_recovered_denominator=1,first_delay_ms_mean_among_recovered=20.,
        eligible_area_px=100,exposure_seconds=.2)
    return {**row,**updates}


def test_count_pooling_exposure_and_conditional_delay():
    a=_curve()
    b=_curve(true_positive_count=4,false_positive_count=0,false_negative_count=0,
        proposal_count=4,first_delay_recovered_denominator=2,recovered_event_count=2,
        first_delay_ms_mean_among_recovered=50.)
    pooled=report.aggregate_synthetic([a,b])[0]
    assert pooled["precision"]==6/7
    assert pooled["precision"]!=(a["true_positive_count"]/a["proposal_count"]+1)/2
    assert pooled["framewise_sensitivity"]==.75
    assert pooled["first_delay_ms_mean_among_recovered"]==40
    assert pooled["first_delay_recovered_denominator"]==3
    assert pooled["false_proposals_per_10000_um2_s"]==1000
    assert pooled["event_window_coverage"]==.75


def test_no_output_precision_and_empty_truth_are_undefined():
    row=_curve(threshold_z=None,true_positive_count=0,false_positive_count=0,
        false_negative_count=0,proposal_count=0,event_count=0,recovered_event_count=0,
        first_delay_recovered_denominator=0,first_delay_ms_mean_among_recovered=None)
    value=report.aggregate_synthetic([row])[0]
    assert value["threshold_label"]=="no_output"
    assert value["precision"] is None
    assert value["framewise_sensitivity"] is None
    assert value["first_delay_ms_mean_among_recovered"] is None


def _specs():
    return [dict(spec_id=f"s{time}",radius_px=7.5,target_sigma_px=1,
        reference_n=9,dt_ms=20,construction="joint",time_scale_ms=time) for time in (0,60)]


def _item(spec, hits, *, real=False):
    summary=dict(proposal_count=len(hits),active_region_frame_count=4,known_positive_count=3)
    if real:
        rows=[dict(observation_id=obs,matched=obs in hits) for obs in ("a","b","c")]
        metrics=dict(summary=summary,occurrence_rows=rows)
    else:
        metrics=dict(summary=summary,proposal_rows=[dict(source_frame_ui=frame,
            matched_canonical_roi_id=roi,is_true_positive=True) for frame,roi in hits])
    return {**spec,"case_id":"same_scene","truth_mode":"sparse_real" if real else "fully_synthetic",
        "threshold":2 if spec["time_scale_ms"] else 3,"metrics":metrics}


def test_paired_active_truth_keys_include_frame_and_identity():
    specs=_specs()
    left=_item(specs[1],[(1,"a"),(1,"b"),(2,"b")])
    right=_item(specs[0],[(1,"a"),(2,"a")])
    row=report.paired_fixed_rows([left,right],specs)[0]
    assert [row[f"active_frames_{key}"] for key in ("both","gained","lost","neither")]==[1,2,1,0]
    assert row["threshold_3d"]==2 and row["threshold_2d"]==3
    assert row["proposals_3d"]==3 and row["proposals_2d"]==2
    assert report.paired_fixed_rows([left],specs)==[]
    right["metrics"]["summary"]["active_region_frame_count"]=5
    with pytest.raises(ValueError,match="denominators differ"):
        report.paired_fixed_rows([left,right],specs)


def test_paired_sparse_windows_are_separate_and_real_accuracy_is_null(tmp_path):
    specs=_specs()
    row=report.paired_fixed_rows([_item(specs[1],["a","b"],real=True),
        _item(specs[0],["b","c"],real=True)],specs)[0]
    assert [row[f"known_windows_{key}"] for key in ("both","gained","lost","neither")]==[1,1,1,0]
    assert row["delta_framewise_sensitivity"] is None
    real=dict(truth_mode="sparse_real",precision=None,framewise_sensitivity=None,false_positive_count=None)
    assert report.aggregate_synthetic([real])==[]
    assert report._real_curves(tmp_path,[real],[],[],None)==[]
    for forbidden in ("precision","framewise_sensitivity","false_positive_count"):
        with pytest.raises(ValueError,match="Sparse real annotations"):
            report._real_curves(tmp_path,[{**real,forbidden:0}],[],[],None)


def _status_fixture(tmp_path, *, complete=True):
    out=tmp_path/"report";out.mkdir()
    expected=[dict(case_id=f"case{i//21}",spec_id=f"s{i%21}") for i in range(357)]
    paragraph=report._completion_paragraph(357,complete,False,"pending")
    source=tmp_path/"original_aggregate.json";source.write_text("frozen numerical record\n")
    context=dict(numerical_cell_count=357,numeric_complete=complete,expected_cells=expected,
        completion_inputs=[report._binding(source)],completion_paragraph=paragraph)
    report._json(out/"completion_context.json",context)
    (out/"REPORT.md").write_text("Heading\n"+paragraph+"\nOther content unchanged\n")
    (out/"fixed.png").write_bytes(b"unchanged scientific pixels")
    artifacts=[dict(report._binding(out/name),path=name) for name in ("completion_context.json","REPORT.md","fixed.png")]
    manifest=dict(status="GENERATED",artifacts=artifacts,inputs=[],numeric_complete=complete,
        numerical_cell_count=357,scientific_audit_complete=False,figure_count=17,
        visual_qa_complete=True,visual_qa={"reviewed":["fixed.png"]},
        dense_stage_bindings=[{"preserve":"original full-stage digest"}])
    report._json(out/"manifest.json",manifest)
    audit=dict(status="PASS",cell_count=357,audits=[dict(**cell,
        summary=dict(scientific_audit_complete=True)) for cell in expected])
    for row in audit["audits"]:
        folder=tmp_path/"audits"/row["case_id"]/row["spec_id"]
        metadata={"summary.json":row["summary"],"status.json":dict(status="complete",scientific_audit_complete=True),
            "validation.json":dict(status="passed",scientific_audit_complete=True),
            "run_contract.json":{},"source_manifest.json":{},"artifact_index.json":{},
            "llm_context.json":{},"inventory.json":{}}
        for name,value in metadata.items():report._json(folder/name,value)
        row["metadata_bindings"]={name:report._binding(folder/name) for name in metadata}
    report._json(tmp_path/"audit_complete.json",audit)
    return out,manifest,audit


def test_refresh_preserves_figures_bindings_and_visual_qa(tmp_path):
    out,before,_=_status_fixture(tmp_path)
    fixed=(out/"fixed.png").read_bytes()
    updated=report.refresh_completion(tmp_path)
    assert updated["scientific_audit_complete"] is True
    for key in ("figure_count","dense_stage_bindings","visual_qa_complete","visual_qa"):
        assert updated[key]==before[key]
    assert (out/"fixed.png").read_bytes()==fixed
    first=(out/"REPORT.md").read_bytes()
    report.refresh_completion(tmp_path)
    assert (out/"REPORT.md").read_bytes()==first
    assert "Other content unchanged" in (out/"REPORT.md").read_text()


@pytest.mark.parametrize("target",["figure","numerical_input","report_text"])
def test_refresh_rejects_changed_bound_content(tmp_path,target):
    out,_,_=_status_fixture(tmp_path)
    path={"figure":out/"fixed.png","numerical_input":tmp_path/"original_aggregate.json",
        "report_text":out/"REPORT.md"}[target]
    path.write_text("changed after original report")
    with pytest.raises(ValueError,match="Changed"):
        report.refresh_completion(tmp_path)


@pytest.mark.parametrize("fault",["missing","duplicate","incomplete_summary","missing_cell"])
def test_refresh_fails_closed_for_missing_or_incomplete_audit(tmp_path,fault):
    _,_,audit=_status_fixture(tmp_path)
    assert report.refresh_completion(tmp_path)["scientific_audit_complete"]
    if fault=="missing":(tmp_path/"audit_complete.json").unlink()
    else:
        audit=deepcopy(audit)
        if fault=="duplicate":audit["audits"][-1]=audit["audits"][0]
        if fault=="incomplete_summary":audit["audits"][0]["summary"]["scientific_audit_complete"]=False
        if fault=="missing_cell":audit["audits"].pop()
        report._json(tmp_path/"audit_complete.json",audit)
    assert not report.refresh_completion(tmp_path)["scientific_audit_complete"]


def test_refresh_cannot_promote_partial_numerical_report(tmp_path):
    _status_fixture(tmp_path,complete=False)
    value=report.refresh_completion(tmp_path)
    assert not value["numeric_complete"] and not value["scientific_audit_complete"]


def test_audit_promotion_checks_actual_bound_metadata(tmp_path):
    _,_,audit=_status_fixture(tmp_path)
    audit["audits"][0]["metadata_bindings"].pop("inventory.json")
    report._json(tmp_path/"audit_complete.json",audit)
    assert not report.refresh_completion(tmp_path)["scientific_audit_complete"]
    # A current metadata file cannot be substituted beneath an old aggregate.
    source=tmp_path/"audits"/"case0"/"s1"/"status.json"
    source.write_text("changed")
    with pytest.raises(ValueError,match="Changed"):
        report._verify(audit["audits"][1]["metadata_bindings"]["status.json"])


def test_descriptive_counts_require_completed_manifest_and_summary_hash(tmp_path):
    folder=tmp_path/"real_review"/"descriptives"
    report._table(folder/"summary.tsv",[dict(calibration_excursion_count=3,rise_fit_count=0,
        decay_fit_count=0,unresolved_fast_rise_count=1,footprint_status="unidentified")])
    manifest=dict(status="CALIBRATION_DESCRIPTIVES_COMPLETE",location_count=1,
        artifacts=[report._binding(folder/"summary.tsv")])
    report._json(folder/"manifest.json",manifest)
    assert report._descriptive_calibration(tmp_path,[])["excursion_count"]==3
    manifest["status"]="PENDING";report._json(folder/"manifest.json",manifest)
    with pytest.raises(ValueError,match="completed manifest"):
        report._descriptive_calibration(tmp_path,[])
    manifest["status"]="CALIBRATION_DESCRIPTIVES_COMPLETE";report._json(folder/"manifest.json",manifest)
    (folder/"summary.tsv").write_text("modified")
    with pytest.raises(ValueError,match="Changed"):
        report._descriptive_calibration(tmp_path,[])


def test_latency_pass_means_parity_not_meeting_deadline(tmp_path):
    folder=tmp_path/"latency";contract=dict(study_root=str(tmp_path))
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    report._json(folder/"contract.json",contract)
    values=[]
    for time in (0,20,60,200):
        for rep in (1,2):
            spec=f"R7.5_T{time}_S1_n9_joint"
            value=dict(spec_id=spec,repetition=rep,status="PASS",contract_sha256=digest,
                operating_point=dict(spec_id=spec),stage_comparison=dict(pass_=True),
                candidate_comparison={"pass":True},hardware=dict(device_name="mock device"),
                kernel_information_support=dict(maximum_history_ms=time),
                timing=dict(period_ms=20.,measured_frame_count=96,
                    processing_ms=dict(p50=10.+time,p99=12.+time,max=13.+time),
                    processing_deadline_exceedance_count=96 if time>=60 else 0,
                    release_to_result_deadline_exceedance_count=96 if time>=60 else 0,
                    release_to_result_ms=dict(p99=14.+time),completion_lateness_ms=dict(max=time)))
            value["stage_comparison"]={"pass":True}
            values.append(value);report._json(folder/spec/f"rep{rep}"/"result.json",value)
    summary=dict(status="PASS",complete=True,contract_sha256=digest,repetition_count=8,
        all_stage_comparisons_pass=True,all_nms_comparisons_pass=True,results=values)
    report._json(folder/"summary.json",summary)
    report._json(folder/"status.json",dict(status="PASS",complete=True,contract_sha256=digest))
    report._json(folder/"artifact_index.json",dict(artifacts=[report._binding(path)
        for path in sorted(folder.rglob("*.json"))]))
    rows=report._latency_rows(tmp_path,[])
    assert len(rows)==8 and all(row["parity_pass"] for row in rows)
    assert sum(row["processing_deadline_exceedance_count"] for row in rows)==384
    assert all(row["full_control_validation"] is False for row in rows)
    (folder/"summary.json").write_text("modified")
    with pytest.raises(ValueError,match="Changed"):
        report._latency_rows(tmp_path,[])
