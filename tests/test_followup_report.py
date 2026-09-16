"""Small reporting tests: units, denominators, paired inventory, and write gates."""
import copy
import json

import pytest

from neurobench.experiments.gamma_ls_difference import followup_report as report


def clip(case, *, tp=1, proposals=2, active=10, events=1, recovered=1, seconds=2):
    return dict(case_id=case, true_positive_count=tp, false_positive_count=proposals-tp,
        proposal_count=proposals, active_region_frame_count=active, event_count=events,
        recovered_event_count=recovered, eligible_area_px=100, exposure_seconds=seconds,
        duplicate_near_active_region_count=0,
        deadline_rows=[dict(deadline_ms=d,event_count=events,recovered_by_deadline=recovered) for d in report.DEADLINES])


def test_pool_uses_counts_and_area_time_not_average_clip_rates():
    result=report.pool([clip("one",tp=1,proposals=2,active=10,seconds=2),
                        clip("two",tp=9,proposals=9,active=90,seconds=8)])
    assert result["precision"]==pytest.approx(10/11)
    assert result["active_frame_coverage"]==pytest.approx(10/100)
    assert result["false_proposals_per_10000um2_second"]==pytest.approx(1*10000/(100*.25*10))
    assert result["deadline_rows"][0]["event_count"]==2
    assert report.pool([clip("none",tp=0,proposals=0,events=0,recovered=0)])["precision"] is None


def test_weak_pool_preserves_misses_and_whole_clip_false_burden():
    events=[dict(case_id=f"c{i}",source_role="weak",first_delay_ms=delay,recovered=delay is not None,
                 active_frame_count=100,matched_active_frame_count=matched)
            for i,(delay,matched) in enumerate([(20,10),(220,20),(None,0)])]
    rows=[clip(f"c{i}",tp=1,proposals=3,active=200,events=2,recovered=1) for i in range(3)]
    result=report.weak_pool(events,rows)
    assert result["weak_prompt_200ms_coverage"]==pytest.approx(1/3)
    assert result["weak_event_coverage"]==pytest.approx(2/3)
    assert result["weak_active_frame_coverage"]==pytest.approx(30/300)
    assert result["false_positive_count"]==6
    assert result["weak_deadline_rows"][0]["event_count"]==3
    events[0]["source_role"]="neighbor"
    with pytest.raises(RuntimeError,match="Neighbor"):
        report.weak_pool(events,rows)


def test_crowding_inventory_requires_both_radii_and_exact_threshold_identity():
    cell=dict(case_id="case",arm_id="arm",study="crowding")
    rows=[dict(case_id="case",arm_id="arm",threshold_id=t,match_radius_px=r)
          for t in report.THRESHOLDS for r in (2.,6.)]
    report.validate_curve_inventory(rows,[cell])
    with pytest.raises(RuntimeError,match="Incomplete or duplicate"):
        report.validate_curve_inventory(rows[:-1],[cell])
    with pytest.raises(RuntimeError,match="Incomplete or duplicate"):
        report.validate_curve_inventory(rows[:-1]+[rows[0]],[cell])


def test_both_real_preparation_schemas_and_scored_dataset_seal():
    small={name:dict(path=f"/data/{name}.json",sha256="digest",size_bytes=1)
           for name in ("metadata","experts","active")}
    named=report.dataset_bindings(small)
    listed=report.dataset_bindings(dict(bindings=[*small.values(),dict(path="/data/Raw.npy",sha256="dense",size_bytes=10**9)]))
    assert listed==named and len(listed)==3
    scored=dict(dataset_bindings=[*small.values(),dict(path="/data/prepared.json",sha256="seal",size_bytes=99)])
    assert set(report.dataset_bindings(scored,cell_seal=True))=={"metadata.json","experts.json","active.json","prepared.json"}
    scored["dataset_bindings"][-1]=small["metadata"]
    with pytest.raises(RuntimeError,match="Missing or duplicate"):
        report.dataset_bindings(scored,cell_seal=True)


def test_missing_audit_aggregate_is_numerical_only(tmp_path):
    state=report._audit_status(tmp_path,{})
    assert state["scientific_audit_complete"] is False
    assert state["status"]=="NUMERICAL_ONLY_AUDITS_PENDING"


def test_audit_aggregate_requires_explicit_reuse_inventory(tmp_path):
    report.write_json(tmp_path/"audit_complete.json",dict(status="PASS",cells=242,new_cells=156,reused_cells=86,audits=[]))
    with pytest.raises(RuntimeError,match="reuse inventory"):
        report._audit_status(tmp_path,{})


def test_completed_root_refuses_generation_and_update(tmp_path):
    (tmp_path/"completion_manifest.json").write_text('{"status":"PASS"}')
    for action in (report.generate,report.update):
        with pytest.raises(RuntimeError,match="Completed roots are immutable"):
            action(tmp_path)
    assert not (tmp_path/"report").exists()


def report_fixture(root):
    out=root/"report";out.mkdir()
    (root/"protocol.json").write_text("{}\n")
    (out/"REPORT.md").write_text("Earlier completion wording\n")
    (out/"figure.png").write_bytes(b"stand-in bytes for immutable figure artifact")
    (out/"sample.json").write_text("[]\n")
    manifest=dict(reporter=report.binding(report.__file__),inputs=[report.binding(root/"protocol.json")],
        artifacts=[dict(report.binding(p),path=p.name) for p in out.iterdir()],
        table_row_counts={"sample":0},figures=["figure.png"],visual_qa_complete=False)
    report.write_json(out/"manifest.json",manifest)
    return out


def test_update_changes_completion_text_without_replotting(tmp_path,monkeypatch):
    out=report_fixture(tmp_path)
    before={n:(out/n).read_bytes() for n in ("figure.png","sample.json")}
    monkeypatch.setattr(report,"_audit_status",lambda root,p:dict(status="NUMERICAL_ONLY_AUDITS_PENDING",scientific_audit_complete=False))
    monkeypatch.setattr(report,"_report_text",lambda tables,figures,audit: audit["status"]+"\n")
    result=report.update(tmp_path)
    assert all((out/n).read_bytes()==value for n,value in before.items())
    assert (out/"REPORT.md").read_text()=="NUMERICAL_ONLY_AUDITS_PENDING\n"
    assert result["visual_qa_complete"] is False


def test_update_rejects_changed_report_artifacts(tmp_path):
    out=report_fixture(tmp_path)
    (out/"figure.png").write_bytes(b"changed")
    before=(out/"REPORT.md").read_bytes()
    with pytest.raises(RuntimeError,match="artifact changed"):
        report.update(tmp_path)
    assert (out/"REPORT.md").read_bytes()==before


def test_report_uses_native_regional_cutoffs_and_counts_not_filtered_zero():
    pooled=[];real=[];cutoffs=[];weak=[]
    for arm in report.READOUTS:
        for method in ("global","regional"):
            pooled.append(dict(base_arm=arm,calibration_method=method,threshold_id="q1",precision=.5,
                active_frame_coverage=.1,false_positive_count=2,deadline_rows=[dict(deadline_ms=200,event_count=4,recovered_by_deadline=1)]))
            real.append(dict(base_arm=arm,calibration_method=method,threshold_id="q1",matched_known_positive_count=2,proposal_count=9,threshold_z=0.))
        for i in range(6):
            cutoffs.append(dict(base_arm=arm,region_id=f"r{i//3}c{i%3}",score_units="dimensionless" if arm.endswith("Z") else "native_fluorescence_units",
                global_native_cutoff=777.,global_setup_proposal_count=101,global_setup_proposal_budget=109,
                native_cutoff=123.+i,setup_proposal_budget=18,setup_proposal_count=17))
    for separation in (None,8,12,16):
        for score in ("A","C","Z"):
            for window in (13,3):
                weak.append(dict(threshold_id="q1",match_radius_px=2,separation_px=separation,readout=score,window=window,
                    weak_matched_active_frame_count=3,weak_active_frame_count=30,false_positive_count=4,proposal_count=7,
                    weak_deadline_rows=[dict(deadline_ms=200,event_count=3,recovered_by_deadline=1)]))
    text=report._report_text(dict(regional_pooled_curves=pooled,regional_real_curves=real,
        real_regional_q1_cutoffs=cutoffs,crowding_pooled_weak_curves=weak),[],dict(status="NUMERICAL_ONLY_AUDITS_PENDING"))
    assert "Global cutoff 777; global setup count/budget 101/109" in text
    assert "| r0c0 | 123 | 18 | 17 |" in text
    assert "internal zero" in text
    assert "real precision" in text.lower()
