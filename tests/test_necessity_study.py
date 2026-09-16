import math

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import necessity_study as n
from neurobench.experiments.gamma_ls_difference.spatiotemporal_metrics import seal_candidates, evaluate_threshold_sweep
from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import extract_frame_candidates


def test_eight_explicit_component_arms():
    assert len(n.arms()) == len({a['arm_id'] for a in n.arms()}) == 8
    assert n.SPEC.time_scale_ms == 0 and n.SPEC.radius_px == 7.5
    assert {a['readout'] for a in n.arms()} == {'X','A','C','Z'}


@pytest.mark.parametrize('budget', [0,1,2,3,5])
def test_strict_calibration_ties_never_exceed_budget(budget):
    tau,count=n.cutoff([9,9,8,3],budget)
    assert count <= budget
    assert count == sum(x>tau for x in [9,9,8,3])


@pytest.mark.parametrize('bad', [[math.nan],[math.inf],[-1],[0]])
def test_invalid_setup_scores_fail(bad):
    with pytest.raises(ValueError):n.cutoff(bad,1)


def test_common_setup_budgets_and_scale_invariance():
    vals=[1,2,3,4,5,6]
    a=n.threshold_plan(vals,n.REFERENCE_AREA_PX,2)
    b=n.threshold_plan([4*x for x in vals],n.REFERENCE_AREA_PX,2)
    assert len(a)==10
    for x,y in zip(a[:-1],b[:-1]):
        assert x['setup_proposal_budget']==y['setup_proposal_budget']
        assert x['setup_proposal_count']==y['setup_proposal_count']
        assert y['threshold']==4*x['threshold']
    assert a[-1]['threshold'] is None
    assert next(x for x in a if x['threshold_id']=='q1')['setup_proposal_budget']==2


def test_budget_quantization_is_explicit():
    p=n.threshold_plan([1,2,3],13456,100)
    assert [r['setup_proposal_budget'] for r in p[:4]]==[0,1,3,6]


def test_frontend_causality_and_signed_plateau_behavior():
    raw=np.zeros((12,9,9),np.float32);raw[3:]=10
    original=list(n.conditioned_frames(raw,1,.4))
    changed=raw.copy();changed[8:]=10000
    alternative=list(n.conditioned_frames(changed,1,.4))
    for i in range(8):
        for k in range(2):np.testing.assert_array_equal(original[i][k],alternative[i][k])
    assert original[-1][0][4,4]>original[3][0][4,4]
    assert original[-1][1][4,4]<original[3][1][4,4]
    np.testing.assert_array_equal(original[0][1],np.zeros((9,9)))


def test_configured_threshold_has_no_application_leakage():
    score=np.arange(120,dtype=np.float32)[:,None,None]
    geom=[dict(canonical_roi_id='r',x_px=0,y_px=0)]
    before=n.configured_calibration(score,geom,list(range(100)))
    score[100:]=1e9
    assert n.configured_calibration(score,geom,list(range(100)))==before
    assert before[0]['setup_exceedance_count']==1


def test_monitoring_keeps_misses_and_inactive_roi_frames():
    score=np.array([0,3,0,0,0,2],np.float32)[:,None,None]
    config=[dict(canonical_roi_id='r',x_index=0,y_index=0,threshold=1.)]
    events=[dict(observation_id='e1',canonical_roi_id='r',source_start_ui=1,source_stop_ui=2),
            dict(observation_id='e2',canonical_roi_id='r',source_start_ui=4,source_stop_ui=4)]
    active=[dict(canonical_roi_id='r',source_frame_ui=f) for f in (1,2,4)]
    out=n.monitor(score,config,list(range(1,7)),list(range(1,7)),active,events)
    assert [out[k] for k in ('true_positive_count','false_positive_count','false_negative_count','true_negative_count')]==[1,1,2,2]
    assert out['precision']==.5 and out['sensitivity']==pytest.approx(1/3)
    assert out['deadline_rows'][0]['recovered_by_deadline']==0
    assert out['deadline_rows'][1]==dict(deadline_ms=20,event_count=2,recovered_by_deadline=1)


def test_no_output_precision_undefined_and_event_miss_denominator():
    event=dict(observation_id='e',event_id='e',canonical_roi_id='r',source_start_ui=1,source_stop_ui=1,x_px=10,y_px=10)
    active=[dict(event,source_frame_ui=1)]
    stream=seal_candidates([],source_frames_ui=[1,2])
    out=evaluate_threshold_sweep(stream,active,[event],thresholds=[0,math.inf])
    assert all(r['precision'] is None and r['framewise_sensitivity']==0 for r in out['curve_rows'])
    assert all(r['event_count']==1 and r['recovered_by_deadline']==0 for r in n.deadline_rows(out['event_rows'][:1]))


def test_nms_prefix_remains_exact_under_higher_cutoffs():
    rng=np.random.default_rng(42);score=rng.normal(10,3,(50,60)).astype(np.float32)
    prefix=extract_frame_candidates(score,source_frame_ui=1,threshold_z=0.)
    for tau in (5.,10.,13.,18.):
        full=extract_frame_candidates(score,source_frame_ui=1,threshold_z=tau)
        cached=[r for r in prefix if r['score']>tau]
        assert [(r['x_px'],r['y_px'],r['score']) for r in full]==[(r['x_px'],r['y_px'],r['score']) for r in cached]


@pytest.mark.parametrize('truth_mode',['fully_synthetic','sparse_real'])
def test_eight_arm_seal_and_evaluate_integration(tmp_path,monkeypatch,truth_mode):
    folder=tmp_path/'datasets/toy';folder.mkdir(parents=True)
    meta=dict(truth_mode=truth_mode,source_frames_ui=list(range(1,9)),setup_source_frames_ui=[1,2,3,4],application_source_frames_ui=[5,6,7,8])
    event=dict(event_id='e',observation_id='e',canonical_roi_id='r',burst_id=1,source_start_ui=6,source_stop_ui=7,x_px=15,y_px=15)
    n.write_json(folder/'metadata.json',meta)
    n.write_json(folder/'experts.json',[event])
    n.write_json(folder/'active.json',[dict(event,source_frame_ui=f) for f in (6,7)] if truth_mode=='fully_synthetic' else [])
    n.write_json(folder/'configured_geometry.json',[dict(canonical_roi_id='r',x_px=15,y_px=15)] if truth_mode=='fully_synthetic' else [])
    n.write_json(folder/'prepared.json',dict(bindings=[]))
    values=np.ones((8,30,30),np.float32);values[5:7,15,15]=10
    path=folder/'values.npy';np.save(path,values)
    state=dict(stages={k:n.binding(path) for k in ('X','A','M','Spread','C','Z','Raw')},scale_floor=1.)
    monkeypatch.setattr(n,'load_protocol',lambda root:dict(cases=['toy']))
    monkeypatch.setattr(n,'prepare_mode',lambda root,case,mode:state)
    n.run(tmp_path)
    assert len(list((tmp_path/'cells/toy').glob('*/sealed.json')))==8
    n.evaluate(tmp_path)
    rows=n.read(tmp_path/'all_curves.json');assert len(rows)==80
    assert all(row['proposal_count']==0 for row in rows if row['threshold_id']=='no_output')
    if truth_mode=='sparse_real':
        assert all(row['precision'] is None and row['framewise_sensitivity'] is None for row in rows)
    else:
        assert all(row['event_count']==1 for row in rows)
        monitors=n.read(tmp_path/'monitoring.json')
        assert all(row['metrics']['true_positive_count']==2 for row in monitors)
