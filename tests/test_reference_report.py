"""Small generated test records only; never read the active study or its truth."""
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import reference_report as report


def curve(seed, tp, active, fp, *, scenario="alone", arm="mean1_n9", threshold="q1", area=100, exposure=2):
    return dict(case_id=f"{scenario}_{seed}",scenario=scenario,seed=seed,arm_id=arm,threshold_id=threshold,
        match_radius_px=2.,proposal_count=tp+fp,true_positive_count=tp,false_positive_count=fp,
        active_region_frame_count=active,event_count=int(active>0),recovered_event_count=int(tp>0),
        duplicate_near_active_region_count=0,application_frame_count=100,eligible_area_px=area,exposure_seconds=exposure,
        false_proposals_per_10000_um2_s=10000*fp/(area*.25*exposure))


def event(seed, hits, delay, *, scenario="alone", arm="mean1_n9", threshold="q1", role="weak"):
    return dict(case_id=f"{scenario}_{seed}",scenario=scenario,seed=seed,arm_id=arm,threshold_id=threshold,
        match_radius_px=2.,event_id=f"{scenario}_{seed}_{role}",source_role=role,active_frame_count=246,
        matched_active_frame_count=hits,first_delay_ms=delay)


def test_pooling_uses_counts_and_area_time_not_average_ratios():
    rows=[curve(1,1,1,1,area=100,exposure=2),curve(2,1,9,8,area=200,exposure=4)]
    pooled=report.pool_curves(rows)[0]
    assert pooled["precision"]==pytest.approx(2/11)
    assert pooled["framewise_sensitivity"]==pytest.approx(2/10)
    assert pooled["false_proposals_per_10000_um2_s"]==pytest.approx(10000*9/(50+200))
    assert pooled["seeds"]==[1,2]


def test_null_denominators_are_not_fabricated():
    pooled=report.pool_curves([curve(1,0,0,7,scenario="stationary")])[0]
    assert pooled["framewise_sensitivity"] is None
    assert pooled["event_window_coverage"] is None
    assert pooled["precision"]==0
    assert report.pool_events([], [pooled])==[]
    assert report.pool_curves([curve(1,0,0,0,scenario="stationary")])[0]["precision"] is None


def test_deadline_denominator_keeps_misses_and_uses_source_roles():
    rows=[curve(s,10,246,4) for s in (1,2,3)]
    events=[event(1,10,20),event(2,0,None),event(3,1,2840)]
    result=report.pool_events(events,report.pool_curves(rows))
    at100=next(r for r in result if r["deadline_ms"]==100)
    assert at100["event_count"]==3 and at100["recovered_by_deadline"]==1
    assert at100["deadline_recovery"]==pytest.approx(1/3)
    assert at100["active_frame_count"]==738 and at100["matched_active_frame_count"]==11
    assert all(r["recovered_by_deadline"]==1 for r in result if r["deadline_ms"]>=20)


def test_duplicate_seeds_or_events_fail_closed():
    row=curve(1,1,246,0)
    with pytest.raises(RuntimeError,match="Duplicate seed"):
        report.pool_curves([row,row])
    e=event(1,1,20)
    with pytest.raises(RuntimeError,match="Duplicate event"):
        report.pool_events([e,e],report.pool_curves([row]))


def test_event_and_full_field_counts_reconcile_including_null():
    row=curve(1,1,246,2);e=event(1,1,20)
    report.validate_event_counts([row],[e])
    report.validate_event_counts([curve(1,0,0,3,scenario="stationary")],[])
    with pytest.raises(RuntimeError,match="counts disagree"):
        report.validate_event_counts([row],[dict(e,matched_active_frame_count=2)])


def test_all_seals_gate_precedes_activity_file_read(tmp_path):
    protocol=dict(cases=[dict(case_id=f"c{i}") for i in range(21)],references=[dict(arm_id=a) for a in report.ARMS],
                  expected_cells=189,expected_curve_rows=3780)
    class NoRead:
        def read(self,path):raise AssertionError("No file should be read while a seal is absent")
    with pytest.raises(RuntimeError,match="All 189"):
        report._seal_gate(tmp_path,protocol,NoRead())


def test_sources_reject_binding_conflict_and_later_change(tmp_path):
    path=tmp_path/"source.json";path.write_text("{}")
    sources=report.Sources();b=sources.bind(path)
    with pytest.raises(RuntimeError,match="Changed evidence"):
        sources.check(dict(b,sha256="0"*64))
    path.write_text("{\"changed\":true}")
    with pytest.raises(RuntimeError,match="Source changed"):
        sources.finish()


def _toy_tables():
    curves=[];events=[];geometry=[];profiles=[];mass=[];traces=[]
    for m,factor in (("2of3",2/3),("1",1),("4of3",4/3)):
        for n in (3,9,15):
            arm=f"mean{m}_n{n}"
            geometry.append(dict(arm_id=arm,realized_mean_radius_px=9*factor,reference_n=n))
            for x in range(6):
                profiles.append(dict(arm_id=arm,radius_px=x,normalized_pixel_weight=x/(100*(n+1))))
                mass.append(dict(arm_id=arm,radial_bin_low_px=x,normalized_annular_mass=x/30))
            for scenario in report.SCENARIOS:
                for t in report.THRESHOLDS:
                    for seed in (1,2,3):
                        null=scenario in report.SCENARIOS[4:];silent=t=="no_output"
                        hits=0 if null or silent else 12+seed
                        multiplier=100 if scenario=="variance_correlation" else (30 if scenario=="shared_brightness_motion" else 1)
                        row=curve(seed,hits,0 if null else 246,0 if silent else seed*multiplier,scenario=scenario,arm=arm,threshold=t)
                        curves.append(row)
                        if not null:
                            events.append(event(seed,hits,None if silent else 20*seed,scenario=scenario,arm=arm,threshold=t))
                            if scenario!="alone":events.append(event(seed,hits,None if silent else 20*seed,scenario=scenario,arm=arm,threshold=t,role="neighbor"))
            for scenario in ("alone","sep8"):
                for frame in range(1,465):
                    traces.append(dict(arm_id=arm,scenario=scenario,source_frame_ui=frame,A=100+frame/1000,
                        M=100+frame/2000,Spread=.5+frame/10000,C=frame/2000,Z=frame/1000,threshold_q1=.3))
    pooled=report.pool_curves(curves)
    return dict(kernel_summary=geometry,kernel_profiles=profiles,kernel_radial_mass=mass,
        pooled_curves=pooled,pooled_deadlines=report.pool_events(events,pooled),mechanism_traces=traces,
        q1_event_replicates=[r for r in events if r["threshold_id"]=="q1"],
        q1_null_replicates=[r for r in curves if r["threshold_id"]=="q1" and r["scenario"] in report.SCENARIOS[4:]])


def test_eight_figure_png_pdf_smoke_uses_only_generated_test_records(tmp_path,monkeypatch):
    tables=_toy_tables();original_save=report._save;checked=[]
    def checked_save(fig,out,stem,caption,names):
        if stem=="null_q1_burden":
            fig.canvas.draw()
            limits=[ax.get_ylim() for ax in fig.axes]
            assert all(pair==limits[0] for pair in limits)
            assert limits[0][0]==0
            assert max(r["false_proposals_per_10000_um2_s"] for r in tables["q1_null_replicates"])<limits[0][1]
            for ax in fig.axes:
                assert ax.get_yscale()=="symlog"
                for line in ax.lines:
                    assert all(limits[0][0]<=v<=limits[0][1] for v in line.get_ydata())
                for collection in ax.collections:
                    assert all(limits[0][0]<=v<=limits[0][1] for v in collection.get_offsets()[:,1])
                    half_marker_px=(max(collection.get_sizes())**.5 + max(collection.get_linewidths()))*fig.dpi/144
                    highest=max(ax.transData.transform(point)[1] for point in collection.get_offsets())
                    assert highest+half_marker_px<ax.bbox.y1
            checked.append(stem)
        return original_save(fig,out,stem,caption,names)
    monkeypatch.setattr(report,"_save",checked_save)
    figures=report._plot(tmp_path,tables)
    assert len(figures)==8
    assert checked==["null_q1_burden"]
    assert len(list(tmp_path.glob("*.png")))==len(list(tmp_path.glob("*.pdf")))==8
    assert all((tmp_path/r[ext]).stat().st_size>1000 for r in figures for ext in ("png","pdf"))


def test_zero_null_burden_retains_a_nonnegative_visible_axis():
    assert report._null_ylim([dict(false_proposals_per_10000_um2_s=0)])==(0.,1.)


def test_update_preserves_figures_tables_and_accepts_only_bound_visual_qa(tmp_path,monkeypatch):
    out=tmp_path/"report";out.mkdir();tables=_toy_tables()
    # No rendering needed for this mutation test: bytes represent frozen artifacts.
    figures=[]
    for i in range(8):
        (out/f"figure{i}.png").write_bytes(f"frozen{i}".encode())
        figures.append(dict(id=f"figure{i}",png=f"figure{i}.png",pdf=f"figure{i}.pdf",caption="test"))
    for name,rows in tables.items():report._write(out/f"{name}.json",rows)
    (out/"REPORT.md").write_text("Pending\n");report._write(tmp_path/"protocol.json",{})
    artifacts=[dict(report._binding(p),path=p.name) for p in sorted(out.iterdir())]
    manifest=dict(figures=figures,figure_count=8,inputs=[],reporter=report._binding(Path(report.__file__)),artifacts=artifacts,
                  table_row_counts={k:len(v) for k,v in tables.items()},visual_qa_complete=False)
    report._write(out/"manifest.json",manifest)
    monkeypatch.setattr(report,"_audit_status",lambda *args:dict(scientific_audit_complete=True,status="PASS"))
    report._write(out/"visual_qa.json",dict(status="PASS",figures=[report._binding(out/r["png"]) for r in figures]))
    result=report.update(tmp_path)
    assert result["scientific_audit_complete"] and result["visual_qa_complete"] and result["figure_count"]==8
    before={b["path"]:b["sha256"] for b in artifacts if b["path"]!="REPORT.md"}
    after={b["path"]:b["sha256"] for b in result["artifacts"] if b["path"]!="REPORT.md"}
    assert before==after
    (out/"figure0.png").write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="Changed evidence"):
        report.update(tmp_path)


def test_incomplete_audit_or_completed_campaign_is_not_promoted(tmp_path):
    report._write(tmp_path/"audit_complete.json",dict(status="PASS",cells=188,new_cells=176,reused_cells=12))
    with pytest.raises(RuntimeError,match="not complete"):
        report._audit_status(tmp_path,{},report.Sources())
    report._write(tmp_path/"completion_manifest.json",dict(status="PASS"))
    with pytest.raises(RuntimeError,match="immutable"):
        report.update(tmp_path)
