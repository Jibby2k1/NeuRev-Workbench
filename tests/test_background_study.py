from copy import deepcopy
from pathlib import Path
import pytest
from neurobench.experiments.gamma_ls_difference import background_study as s


def test_physical_reuse_and_logical_aliases_preserve_pairing():
    logical=s.logical_design();physical=s.case_design();byid={c['case_id']:c for c in physical}
    assert len(logical)==96 and len(physical)==84
    assert sum(r['alias'] for r in logical)==12
    assert sum(c['reused_dataset'] for c in physical)==24
    for r in logical:
        c=byid[r['canonical_case_id']]
        assert all(r[k]==c[k] for k in ('seed','background','V','S','T'))
        assert r['alias']==(r['normalization']=='conditioned' and r['S']==r['T']==0)
        assert c['normalization']==('raw' if r['alias'] else r['normalization'])
        assert c['reused_dataset']==(c['background']=='sloped' and c['normalization']=='raw')
    states=s.cells({'cases':physical,'references':[{'arm_id':a} for a in s.ARMS]})
    assert len(states)==252 and len({(r['case_id'],r['arm_id']) for r in states})==252
    assert sum(r['reused_audit'] for r in states)==72


def test_new_and_old_ids_remain_distinct():
    from neurobench.experiments.gamma_ls_difference.noise_fixtures import case_id
    for c in s.case_design():
        old=case_id(c['seed'],c['V'],c['S'],c['T'])
        if c['reused_dataset']:assert c['case_id']==c['baseline_case_id']==old
        else:assert c['case_id']!=old and c['baseline_case_id'] is None


def test_completed_root_guard(tmp_path):
    (tmp_path/'completion_manifest.json').write_text('{}')
    with pytest.raises(RuntimeError):s.guard_open(tmp_path)


def test_seal_all_requires_every_cell_before_group_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'load',lambda _:dict(cases=s.case_design(),references=[{'arm_id':a} for a in s.ARMS]))
    with pytest.raises(RuntimeError,match='Missing physical'):s.seal_all(tmp_path)
    assert not (tmp_path/'computation_complete.json').exists()


def test_corrupt_protocol_rejected_before_inputs(tmp_path):
    s.write_json(tmp_path/'protocol.json',{'bad':True})
    s.write_json(tmp_path/'preflight.json',{'protocol_sha256':'wrong'})
    with pytest.raises(RuntimeError,match='Frozen protocol'):s.load(tmp_path)


def test_all_seals_verified_before_inventory_acceptance(tmp_path,monkeypatch):
    protocol=dict(cases=[dict(case_id='a',kind='factorial',seed=1,V=0,S=0,T=0,background='flat',normalization='raw',reused_dataset=False),dict(case_id='b',kind='factorial',seed=1,V=0,S=0,T=0,background='flat',normalization='raw',reused_dataset=False)],references=[{'arm_id':'arm'}])
    for case in ('a','b'):
        f=tmp_path/'cells'/case/'arm';f.mkdir(parents=True)
        s.write_json(f/'sealed.json',dict(status='SEALED_BEFORE_ACTIVITY_TRUTH_JOIN' if case=='a' else 'INVALID'))
    checked=[]
    monkeypatch.setattr(s,'check_seal',lambda x:checked.append(x))
    with pytest.raises(RuntimeError,match='Invalid scoring seal'):s.validate_all_seals(tmp_path,protocol)
    assert len(checked)==1
