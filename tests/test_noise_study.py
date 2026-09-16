"""Bounded scientific gates for the noise-control runner."""
import numpy as np
import pytest
from neurobench.experiments.gamma_ls_difference import noise_study as study


def test_factorial_matrix_and_exact_reuse_identity():
    p=dict(cases=study.case_design(),references=[dict(arm_id=a) for a in study.ARMS])
    states=study.cells(p)
    assert len(states)==len({(c['case_id'],c['arm_id']) for c in states})==81
    assert sum(c['reused_audit'] for c in states)==18
    assert len(p['cases'])==27
    for seed in study.SEEDS:
        factorial=[c for c in p['cases'] if c['seed']==seed and c['kind']=='factorial']
        assert {(c['V'],c['S'],c['T']) for c in factorial}=={(v,s,t) for v in (0,1) for s in (0,1) for t in (0,1)}
    assert all(c['baseline_arm']==c['arm_id'] and c['baseline_case_id']==c['case_id'] for c in states if c['reused_audit'])
    assert all(c['baseline_arm'] is None for c in states if not c['reused_audit'])


def test_completed_output_is_not_reopened(tmp_path):
    (tmp_path/'completion_manifest.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Preserve completed'):study.guard_open(tmp_path)


def _p():
    return dict(cases=[dict(case_id='toy',seed=1,kind='factorial',V=0,S=1,T=0,reused_dataset=False,baseline_case_id=None)],
                references=[dict(arm_id='mean2of3_n3')])


def test_global_seal_gate_precedes_any_truth_read(tmp_path,monkeypatch):
    p=_p();p['references'].append(dict(arm_id='mean1_n9'))
    folder=tmp_path/'cells/toy/mean2of3_n3';folder.mkdir(parents=True);(folder/'sealed.json').write_text('{}')
    monkeypatch.setattr(study,'load',lambda root:p)
    def forbidden(path):raise AssertionError('Truth or cell read before global seal gate')
    monkeypatch.setattr(study,'read',forbidden)
    with pytest.raises(RuntimeError,match='Seal all81'):study.evaluate(tmp_path)


def _tiny(tmp_path,monkeypatch,non_null=False):
    folder=tmp_path/'datasets/toy';folder.mkdir(parents=True)
    study.write_json(folder/'metadata.json',dict(truth_mode='fully_synthetic',source_frames_ui=list(range(1,167)),
        setup_source_frames_ui=list(range(65,165)),application_source_frames_ui=[165,166]))
    event=dict(event_id='invalid',observation_id='invalid',canonical_roi_id='invalid',source_start_ui=165,source_stop_ui=165,x_px=15,y_px=15)
    study.write_json(folder/'experts.json',[event] if non_null else [])
    study.write_json(folder/'active.json',[dict(event,source_frame_ui=165)] if non_null else [])
    study.write_json(folder/'prepared.json',dict(status='PASS'))
    score=np.zeros((166,30,30),np.float32);score[:164,15,15]=3;score[164:,15,15]=10
    np.save(folder/'score.npy',score)
    state=dict(stages={k:study.binding(folder/'score.npy') for k in ('A','M','Spread','C','Z','X','Raw')},scale_floor=1.)
    monkeypatch.setattr(study,'load',lambda root:_p());monkeypatch.setattr(study,'score_reference',lambda *args:state)
    monkeypatch.setattr(study,'SEEDS',(1,));monkeypatch.setattr(study,'ARMS',('mean2of3_n3',))
    bs=[study.binding(folder/n) for n in ('metadata.json','experts.json','active.json','prepared.json')]
    study.write_json(tmp_path/'datasets_complete.json',dict(status='PASS',datasets=27,variance_pairs_early_byte_equal=True,dataset_bindings={'toy':bs}))
    study.write_json(tmp_path/'projection_visual_qa.json',dict(status='PASS'))
    return folder


def test_null_evaluation_preserves_native_threshold_and_undefined_sensitivity(tmp_path,monkeypatch):
    _tiny(tmp_path,monkeypatch);study.run(tmp_path);study.evaluate(tmp_path)
    out=tmp_path/'cells/toy/mean2of3_n3';op=study.read(out/'calibration.json')
    assert op['threshold']==3 and op['setup_source_frames_ui']==list(range(65,165))
    rows=study.read(out/'curves.json')['curve_rows'];assert len(rows)==20
    for r in rows:
        assert r['event_count']==r['active_region_frame_count']==r['true_positive_count']==0
        assert r['framewise_sensitivity'] is None and r['event_window_coverage'] is None
        assert r['proposal_count']==r['false_positive_count']
    assert {r['proposal_count'] for r in rows if r['threshold_id']=='q1'}=={2}
    assert {r['proposal_count'] for r in rows if r['threshold_id']=='no_output'}=={0}
    study.evaluate(tmp_path)
    values=study.read(out/'curves.json');values['curve_rows'][0]['proposal_count']+=1;study.write_json(out/'curves.json',values)
    with pytest.raises((RuntimeError,ValueError)):study.evaluate(tmp_path)


def test_non_null_truth_is_rejected(tmp_path,monkeypatch):
    _tiny(tmp_path,monkeypatch,True);study.run(tmp_path)
    with pytest.raises(RuntimeError,match='Null truth must be empty'):study.evaluate(tmp_path)


def test_projection_review_is_required_before_scoring(tmp_path,monkeypatch):
    _tiny(tmp_path,monkeypatch);study.write_json(tmp_path/'projection_visual_qa.json',dict(status='PENDING'))
    with pytest.raises(RuntimeError,match='Projection visual QA'):study.run(tmp_path)


def test_changed_stage_is_rejected_before_resume(tmp_path,monkeypatch):
    folder=_tiny(tmp_path,monkeypatch);study.run(tmp_path)
    score=np.load(folder/'score.npy');score[165,15,15]=999;np.save(folder/'score.npy',score)
    with pytest.raises((RuntimeError,ValueError)):study.run(tmp_path)


def test_protocol_and_baseline_sources_are_hash_checked(tmp_path):
    bs=[]
    for name in ('baseline_completion.json','baseline_protocol.json','source.json'):
        study.write_json(tmp_path/name,dict(status='PASS'));bs.append(study.binding(tmp_path/name))
    p=dict(code_bindings=[],kernel_bindings=[],baseline_sources=[bs[2]],baseline_completion=bs[0],baseline_protocol=bs[1])
    study.write_json(tmp_path/'protocol.json',p);study.write_json(tmp_path/'preflight.json',dict(protocol_sha256=study.sha256(tmp_path/'protocol.json')))
    assert study.load(tmp_path)==p
    (tmp_path/'source.json').write_text('{}')
    with pytest.raises((RuntimeError,ValueError)):study.load(tmp_path)
