"""Isolated generated records; never open the active campaign or real movies."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import noise_report as report


def protocol():
    cases=[]
    for seed in report.SEEDS:
        for v,s,t in report.FACTORS:
            cases.append(dict(case_id=f"v{v}s{s}t{t}_{seed}",seed=seed,kind="factorial",V=v,S=s,T=t,
                              reused_dataset=(v,s,t)==(0,0,0)))
        cases.append(dict(case_id=f"legacy_{seed}",seed=seed,kind="legacy",V=None,S=None,T=None,reused_dataset=True))
    return dict(cases=cases,references=[dict(arm_id=a) for a in report.ARMS],expected_cells=81,expected_curve_rows=1620,
                frame_rate_hz=50,pixel_size_um=.5,code_bindings=[],kernel_bindings=[])


def test_exact_matrix_and_before_seal_read_gate(tmp_path):
    p=protocol();assert len(report._matrix(p))==81
    class NoRead:
        def read(self,path):raise AssertionError("No outcomes may be read")
    with pytest.raises(RuntimeError,match="All 81"):
        report._seal_gate(tmp_path,p,NoRead())
    p["cases"][0]["V"]=1
    with pytest.raises(RuntimeError,match="factorial"):
        report._matrix(p)


def test_known_background_residual_and_causal_covariance():
    yy,xx=np.mgrid[:18,:18]
    bg=100+.03*(xx+49-113)+.02*(yy+49-113)
    checker=((xx+yy)%2)*2-1
    a=np.stack([bg+checker,bg+2*checker,bg-3*checker])
    r=report.residual_diagnostics(a)
    assert [x["residual_variance"] for x in r]==pytest.approx([1,4,9])
    assert r[0]["temporal_lag1_correlation"] is None
    assert r[1]["temporal_lag1_covariance"]==pytest.approx(2)
    assert r[2]["temporal_lag1_correlation"]==pytest.approx(-1)
    assert r[0]["spatial_lag1_correlation"]==pytest.approx(-1)
    changed=a.copy();changed[2]+=np.arange(18)[None,:]*10
    assert report.residual_diagnostics(changed)[:2]==r[:2]
    constant=report.residual_diagnostics(np.repeat(bg[None],3,axis=0))
    assert all(x["residual_variance"]==0 and x["spatial_lag1_correlation"] is None for x in constant)


def test_strict_score_floor_rules_and_no_sensitivity_fabrication():
    a=np.ones((2,16,16));a[0,6:10,6:10]=np.arange(16).reshape(4,4)
    z=report.score_diagnostics(a,stage="Z",floor=1,threshold=1)
    spread=report.score_diagnostics(a,stage="Spread",floor=1,threshold=1)
    assert z[0]["exceedance_fraction"]==14/16 and z[1]["exceedance_fraction"]==0
    assert spread[0]["floor_active_fraction"]==1/16 and spread[1]["floor_active_fraction"]==0
    assert not any("precision" in k or "sensitivity" in k for r in z+spread for k in r)


def test_proposal_count_uses_native_strict_cutoff_and_all_rows():
    rows=[dict(proposal_id=str(i),source_frame_ui=165+i//2,score=s,x_px=10,y_px=10) for i,s in enumerate([1.,2.,3.])]
    assert report.proposal_frames(rows,first=165,last=167,threshold=1)=={165:1,166:1,167:0}
    with pytest.raises(RuntimeError,match="does not exceed"):
        report.proposal_frames(rows,first=165,last=167,threshold=1,already_filtered=True)
    with pytest.raises(RuntimeError,match="duplicate"):
        report.proposal_frames(rows+[rows[0]],first=165,last=167,threshold=1)


def base_row(frame, **extra):
    return dict(case_id="c",arm_id=report.ARMS[0],seed=report.SEEDS[0],case_kind="factorial",V=0,S=0,T=0,
                threshold=2.,scale_floor=.3,source_frame_ui=frame,epoch=report.epoch(frame),proposal_count=1,
                Raw_residual_variance=4.,Input_residual_variance=.5,Spread_q50=.6,Z_q999=2.,Z_exceedance_fraction=.1,
                floor_active_fraction=.2,**extra)


def test_epoch_denominators_and_means_of_frame_statistics():
    rows=[base_row(f) for f in range(65,465)]
    result=report.epoch_tables(rows);by={r["epoch"]:r for r in result}
    assert by["early"]["proposal_count"]==100 and by["late"]["proposal_count"]==200
    assert by["early"][report.RATE]==by["late"][report.RATE]==pytest.approx(10000*100/(13456*.25*2))
    assert by["early"]["exposure_seconds"]==2 and by["late"]["exposure_seconds"]==4
    assert by["early"]["Z_q999"]==2
    with pytest.raises(RuntimeError,match="epoch samples"):
        report.epoch_tables(rows[1:])
    bins=report.time_bins(rows)
    assert len(bins)==40 and sum(r["proposal_count"] for r in bins)==400
    assert bins[0]["source_start_ui"]==65 and bins[-1]["source_stop_ui"]==464


def test_factorial_contrast_scales_and_legacy_exclusion():
    rows=[]
    for v,s,t in report.FACTORS:
        value=10+2*v+3*s+5*t+7*v*s+11*v*t+13*s*t+17*v*s*t
        rows.append(dict(seed=1,arm_id="a",epoch="late",case_kind="factorial",V=v,S=s,T=t,y=value))
    rows.append(dict(seed=1,arm_id="a",epoch="late",case_kind="legacy",V=None,S=None,T=None,y=1e12))
    effects={r["effect"]:r["difference"] for r in report.factorial_effects(rows,metrics=("y",))}
    assert effects==pytest.approx(dict(V=2+7/2+11/2+17/4,S=3+7/2+13/2+17/4,T=5+11/2+13/2+17/4,
                                      VS=7+17/2,VT=11+17/2,ST=13+17/2,VST=17))
    with pytest.raises(RuntimeError,match="Incomplete factorial"):
        report.factorial_effects(rows[1:],metrics=("y",))
    with pytest.raises(RuntimeError,match="Duplicate factorial"):
        report.factorial_effects(rows+[rows[0]],metrics=("y",))


def test_early_variance_pairs_fail_on_any_changed_measured_quantity():
    rows=[dict(base_row(200),V=v) for v in (0,1)]
    assert report.verify_early_variance_pairing(rows)["compared_frame_pairs"]==1
    rows[1]["Z_q999"]+=.001
    with pytest.raises(RuntimeError,match="before its"):
        report.verify_early_variance_pairing(rows)


@pytest.fixture
def sealed_root(tmp_path):
    root=tmp_path;p=protocol()
    dummy=root/"baseline.json";report._write(dummy,{"status":"PASS"})
    p["baseline_completion"]=p["baseline_protocol"]=report._binding(dummy)
    report._write(root/"protocol.json",p)
    report._write(root/"preflight.json",dict(status="PASS",protocol_sha256=report._binding(root/"protocol.json")["sha256"]))
    all_curves=[];reused=[]
    for case in p["cases"]:
        ds=root/"datasets"/case["case_id"];ds.mkdir(parents=True)
        report._write(ds/"metadata.json",dict(truth_mode="fully_synthetic",source_frames_ui=list(range(1,465)),application_source_frames_ui=list(range(165,465)),
                    shape_tyx=[464,226,226],evaluation_shape_yx=[128,128],original_source_offset_xy=[49,49],
                    background=dict(offset=100.,gradient_x_per_px=.03,gradient_y_per_px=.02)))
        for name in ("experts.json","active.json"):report._write(ds/name,[])
        for arm in report.ARMS:
            folder=root/"cells"/case["case_id"]/arm;folder.mkdir(parents=True)
            plan=[dict(threshold_id=t,threshold=float(i),setup_proposal_count=0) for i,t in enumerate(report.common.THRESHOLDS)]
            op=dict(threshold_id="q1",threshold_frozen_from_calibration_only=True,window=3,eligible_area_px=13456,
                    setup_proposal_budget=6,scale_floor=.3,threshold=3.,setup_proposal_count=0)
            report._write(folder/"threshold_plan.json",plan);report._write(folder/"calibration.json",op)
            seal=dict(status="SEALED_BEFORE_ACTIVITY_TRUTH_JOIN",dataset_bindings=[report._binding(ds/n) for n in ("metadata.json","experts.json","active.json")],
                      stages={k:report._binding(dummy) for k in ("Raw","Input","A")},prefix=report._binding(dummy),
                      calibration=report._binding(folder/"calibration.json"),threshold_plan=report._binding(folder/"threshold_plan.json"))
            report._write(folder/"sealed.json",seal)
            rows=[dict(case_id=case["case_id"],arm_id=arm,threshold_id=x["threshold_id"],threshold_z=x["threshold"],match_radius_px=radius,
                       event_count=0,active_region_frame_count=0,true_positive_count=0,false_positive_count=0,proposal_count=0,
                       framewise_sensitivity=None,event_window_coverage=None,eligible_area_px=13456,exposure_seconds=6.)
                  for x in plan for radius in (2.,6.)]
            report._write(folder/"curves.json",dict(curve_rows=rows,event_rows=[]))
            report._write(folder/"evaluated.json",dict(status="PASS",seal=report._binding(folder/"sealed.json"),outputs=[report._binding(folder/"curves.json")]))
            all_curves.extend(rows)
            if case["reused_dataset"]:reused.append(dict(case_id=case["case_id"],arm_id=arm,all_thresholds_equal=True,metrics_equal=True,stages_equal=True,candidates_equal=True))
    report._write(root/"all_curves.json",all_curves);(root/"all_curves.tsv").write_text("toy-source\n")
    report._write(root/"baseline_replication.json",dict(status="PASS",cells=reused))
    report._write(root/"computation_complete.json",dict(status="PASS",cells=81))
    report._write(root/"evaluation_complete.json",dict(status="PASS",cells=81,curve_rows=1620))
    report._write(root/"datasets_complete.json",dict(status="PASS",datasets=27,paired_setup_byte_equal=True,variance_pairs_early_byte_equal=True))
    report._write(root/"paired_setup_calibration_check.json",dict(status="PASS",groups=[dict(seed=s,arm_id=a,cases=9,thresholds_equal=True,floors_equal=True) for s in report.SEEDS for a in report.ARMS]))
    return root


def test_loader_verifies_complete_null_matrix_and_native_cutoffs(sealed_root):
    data=report._load(sealed_root)
    assert len(data["curves"])==1620 and len(data["seals"])==81
    assert all(r["framewise_sensitivity"] is None for r in data["curves"])
    rows=report._read(sealed_root/"all_curves.json");rows[0]["proposal_count"]=1;report._write(sealed_root/"all_curves.json",rows)
    with pytest.raises(RuntimeError,match="Aggregate differs"):
        report._load(sealed_root)


def test_loader_rejects_unsealed_candidate_metadata_change(sealed_root):
    path=next((sealed_root/"cells").glob("*/*/calibration.json"))
    op=report._read(path);op["threshold"]=999;report._write(path,op)
    with pytest.raises(RuntimeError,match="Changed evidence"):
        report._load(sealed_root)


def toy_tables():
    epochs=[];bins=[]
    for arm_idx,arm in enumerate(report.ARMS):
        for seed in report.SEEDS:
            for v,s,t in report.FACTORS:
                c=f"{v}{s}{t}_{seed}";g=dict(case_id=c,arm_id=arm,seed=seed,case_kind="factorial",V=v,S=s,T=t)
                for phase in report.EPOCHS:
                    effect=1 if phase=="setup" else 1+s*4+t*9+(v*50 if phase=="late" else 0)
                    epochs.append(dict(g,epoch=phase,**{report.RATE:float(effect*(arm_idx+1)),
                        "Raw_residual_variance":4.+(21*v if phase=="late" else 0),"Input_residual_variance":effect/3,
                        "Spread_q50":effect/4,"Z_q999":2.+effect/20,"Z_exceedance_fraction":effect/100,
                        "floor_active_fraction":.2/effect}))
                for i in range(40):bins.append(dict(g,time_mid_seconds=-1.91+i*.2,**{report.RATE:float(i+s*4+t*8+v*30)}))
    return dict(epochs=epochs,factorial_effects=report.factorial_effects(epochs),time_bins=bins,per_frame=[],legacy_bridge=[])


def test_five_figures_have_shared_complete_data_extents(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    saved=[]
    def inspect(fig,out,stem,caption,tables):
        fig.canvas.draw()
        for ax in fig.axes:
            lo,hi=ax.get_ylim()
            for line in ax.lines:
                y=np.asarray(line.get_ydata(),dtype=float)
                assert np.isfinite(y).all() and (y>=lo).all() and (y<=hi).all(),stem
            for coll in ax.collections:
                if len(coll.get_offsets()):
                    y=np.asarray(coll.get_offsets())[:,1]
                    assert (y>=lo).all() and (y<=hi).all(),stem
        assert len(fig.axes) in (6,9)
        fig.savefig(tmp_path/(stem+".png"),dpi=45);plt.close(fig);saved.append(stem)
        return dict(id=stem,png=stem+".png",pdf=stem+".pdf",caption=caption,source_tables=tables)
    monkeypatch.setattr(report.common,"_save",inspect)
    assert len(report._plot(tmp_path,toy_tables()))==5 and len(set(saved))==5


def test_refresh_preserves_figures_checks_dependencies_and_resets_qa(tmp_path,monkeypatch):
    root=tmp_path;out=root/"report";out.mkdir();report._write(root/"protocol.json",protocol())
    (out/"figure.png").write_bytes(b"scientific-output");(out/"REPORT.md").write_text("old")
    tables=toy_tables()
    for name,rows in tables.items():report._write(out/(name+".json"),rows)
    m=dict(reporter=report._binding(Path(report.__file__)),reporting_code=[report._binding(Path(report.common.__file__))],
           inputs=[],artifacts=[dict(report._binding(p),path=p.name) for p in out.iterdir()],figures=[],
           table_row_counts={k:len(v) for k,v in tables.items()},visual_qa_complete=True)
    report._write(out/"manifest.json",m)
    monkeypatch.setattr(report,"_audit_status",lambda *args:dict(scientific_audit_complete=False,status="PENDING"))
    updated=report.update(root)
    assert (out/"figure.png").read_bytes()==b"scientific-output" and not updated["visual_qa_complete"]
    (out/"figure.png").write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="Changed evidence"):
        report.update(root)


def test_present_failed_audit_never_promotes(tmp_path):
    report._write(tmp_path/"audit_complete.json",dict(status="FAIL",cells=81,new_cells=63,reused_cells=18))
    with pytest.raises(RuntimeError,match="not complete"):
        report._audit_status(tmp_path,protocol(),report.Sources())


def test_variance_theory_does_not_treat_legacy_as_orthogonal_or_fit_measurements():
    theory=dict(gaussian=dict(sigma_px=1,truncate=4,white_noise_energy=.1,spatial_on_off_variance_ratio=5),
                ema=dict(alpha=.4,raw_ar_rho=.8,white_noise_variance_factor=.25,temporal_on_off_variance_ratio=3))
    r=dict(case_id="c",arm_id="a",seed=1,case_kind="factorial",V=1,S=1,T=1,epoch="late",Raw_residual_variance=24.,Input_residual_variance=9.)
    out=report.variance_theory_rows([r,dict(r,case_kind="legacy",V=None,S=None,T=None)],theory)
    assert out[0]["raw_pointwise_variance_theory"]==25
    assert out[0]["conditioned_stationary_pointwise_variance_theory"]==pytest.approx(25*.1*.25*5*3)
    assert out[0]["Raw_measured_over_pointwise_theory"]==24/25
    assert out[1]["raw_pointwise_variance_theory"]==pytest.approx(40)
    assert out[1]["conditioned_stationary_pointwise_variance_theory"] is None
