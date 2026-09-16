"""Bounded algebraic and tiny-file tests; no production movie generation."""
import itertools
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import noise_fixtures as n


def tiny(seed=20260916,v=0,s=0,t=0,application_frames=5):
    return np.stack(list(n._iter_config(seed,v,s,t,shape_yx=(9,11),
        warmup_frames=2,setup_frames=3,application_frames=application_frames,variance_step=2)))


def test_specs_are_exact_24_and_stationary_identity_is_preserved():
    rows=n.specs()
    assert len(rows)==len({r['case_id'] for r in rows})==24
    assert {r['seed'] for r in rows}==set(n.SEEDS)
    for seed in n.SEEDS:
        assert n.case_id(seed,0,0,0)==f'reference_null_stationary__seed{seed}'
    m=n.noise_metadata(20260916,1,1,1)
    assert m['shape_tyx']==[464,226,226] and m['evaluation_box_yxyx']==[49,49,177,177]
    assert m['setup_source_frames_ui']==list(range(65,165))
    assert m['factors_start_source_frame_ui']==165 and m['variance_step_source_frame_ui']==265
    assert m['event_count']==m['active_region_frame_count']==0


@pytest.mark.parametrize('args',[(1,0,0,0),(True,0,0,0),(20260916,2,0,0),(20260916,0,-1,0),(20260916,0,0,'1')])
def test_invalid_seed_and_factors_rejected(args):
    with pytest.raises(ValueError):n.case_id(*args)


def test_000_is_exact_frozen_stationary_arithmetic_on_tiny_complete_movie():
    actual=tiny();shape=actual.shape[1:];yy,xx=np.indices(shape,dtype=np.float32)
    background=100+.03*(xx-shape[1]/2)+.02*(yy-shape[0]/2)
    rng=np.random.default_rng(20260916)
    expected=np.stack([background+2.0*rng.standard_normal(shape,dtype=np.float32) for _ in actual])
    assert np.array_equal(actual,expected)


def test_000_first_production_frame_matches_existing_generator():
    from neurobench.experiments.gamma_ls_difference.reference_fixtures import iter_null_frames
    assert np.array_equal(next(n.iter_noise_frames(20260916,0,0,0)),
                          next(iter_null_frames(20260916,'stationary')))


@pytest.mark.parametrize('seed',n.SEEDS)
def test_all_setup_pairs_and_variance_pair_before_step_are_byte_identical(seed):
    baseline=tiny(seed)
    args=dict(shape_yx=(9,11),warmup_frames=2,setup_frames=3,application_frames=103)
    for s,t in itertools.product((0,1),repeat=2):
        low=np.stack(list(n._iter_config(seed,0,s,t,**args)))
        step=np.stack(list(n._iter_config(seed,1,s,t,**args)))
        assert np.array_equal(low[:5],baseline[:5])
        assert np.array_equal(step[:105],low[:105])  # Through application index99.
        assert not np.array_equal(step[105:],low[105:])


def test_independent_initializer_and_first_application_use_exact_declared_streams():
    seed=20260916;shape=(9,11)
    initial_rng=np.random.default_rng(np.random.SeedSequence([seed,20260915,1]))
    eta=initial_rng.standard_normal(shape,dtype=np.float32)
    assert np.array_equal(n._initial_state(seed,0,shape),eta)
    assert np.array_equal(n._initial_state(seed,1,shape),n._spatial(eta,1))
    main=np.random.default_rng(seed)
    innovations=[main.standard_normal(shape,dtype=np.float32) for _ in range(6)]
    expected_state=.8*eta+.6*innovations[-1]
    yy,xx=np.indices(shape,dtype=np.float32);background=100+.03*(xx-shape[1]/2)+.02*(yy-shape[0]/2)
    assert np.array_equal(tiny(seed,0,0,1)[5],background+2.0*expected_state)


def test_prefix_does_not_depend_on_future_duration_or_variance_step():
    assert np.array_equal(tiny(t=1,application_frames=2),tiny(t=1,application_frames=8)[:7])
    args=dict(shape_yx=(9,11),warmup_frames=2,setup_frames=3,application_frames=6)
    early=np.stack(list(n._iter_config(20260916,1,1,1,variance_step=2,**args)))
    later=np.stack(list(n._iter_config(20260916,1,1,1,variance_step=4,**args)))
    assert np.array_equal(early[:7],later[:7])


@pytest.mark.parametrize('offset',[(0,0),(0,1),(1,1),(2,0),(3,0)])
def test_spatial_covariance_is_filter_autocorrelation_not_sum_to_one(offset):
    kernel=np.zeros((9,9));kernel[3:6,3:6]=1/3
    covariance=np.sum(kernel*np.roll(kernel,offset,axis=(0,1)))
    assert n.unit_covariance(1,0,displacement_yx=offset)==pytest.approx(covariance)


def test_stationary_initializer_gives_exact_temporal_covariance_from_first_frame():
    # Each column represents one independent eta/innovation; no Monte Carlo.
    count=6;coeff=np.zeros((count,count+1));state=np.zeros(count+1);state[0]=1
    for frame in range(count):
        innovation=np.zeros(count+1);innovation[frame+1]=1
        state=.8*state+.6*innovation;coeff[frame]=state
    covariance=coeff@coeff.T
    for a,b in itertools.product(range(count),repeat=2):
        assert covariance[a,b]==pytest.approx(n.unit_covariance(0,1,lag_frames=abs(a-b)),abs=3e-16)
    assert covariance[0,0]==pytest.approx(1)
    # Zero initialization would instead yield .36, exposing the initial-variance confound.
    assert .6**2!=pytest.approx(covariance[0,0])


def test_theory_uses_exact_finite_gaussian_composition_and_ema_covariance():
    d=n.theory();g=np.asarray(d['gaussian']['weights_1d']);G=g[:,None]*g[None,:]
    composed=np.zeros((11,11))
    for y,x in itertools.product(range(3),repeat=2):composed[y:y+9,x:x+9]+=G/3
    assert d['gaussian']['white_noise_energy']==pytest.approx(np.sum(G*G))
    assert d['gaussian']['spatial_on_composed_energy']==pytest.approx(np.sum(composed*composed))
    weights=.4*.6**np.arange(100)
    covariance=.8**np.abs(np.arange(100)[:,None]-np.arange(100)[None,:])
    direct=float(weights@covariance@weights)
    assert d['ema']['ar_noise_variance_factor']==pytest.approx(direct,rel=2e-15)
    assert d['combined_conditioned_variance_ratio']==pytest.approx(14.840114777310179)


def test_legacy_same_innovation_cross_term_is_retained():
    # app0 old noise = sigma*epsilon + 1.8*L(epsilon); same random field.
    kernel=np.full((3,3),1.8/3);kernel[1,1]+=2
    assert n.legacy_variance(0)==pytest.approx(np.sum(kernel*kernel))
    assert n.legacy_variance(99)==pytest.approx(15.4)
    assert n.legacy_variance(100)==pytest.approx(40)


@pytest.fixture
def small_geometry(monkeypatch):
    monkeypatch.setattr(n,'EVALUATION_SHAPE_YX',(8,8));monkeypatch.setattr(n,'SPATIAL_HALO_PX',5)
    monkeypatch.setattr(n,'WARMUP_FRAMES',1);monkeypatch.setattr(n,'SETUP_FRAMES',2)
    monkeypatch.setattr(n,'APPLICATION_FRAMES',4)


def test_preparation_has_conditioned_arrays_empty_truth_and_verified_resume(tmp_path,small_geometry):
    from neurobench.experiments.gamma_ls_difference.control_study import conditioned_frames
    result=n.build_noise_dataset(tmp_path,20260916,0,1,1);folder=tmp_path/'datasets'/result['case_id']
    raw=np.load(folder/'raw_source.npy');expected=list(conditioned_frames(raw,1.,.4))
    assert np.array_equal(np.load(folder/'level_source.npy'),np.stack([v[0] for v in expected]))
    assert np.array_equal(np.load(folder/'input_source.npy'),np.stack([v[1] for v in expected]))
    assert json.loads((folder/'experts.json').read_text())==[]
    assert json.loads((folder/'active.json').read_text())==[]
    assert n.build_noise_dataset(tmp_path,20260916,0,1,1)['resumed'] is True


def test_resume_rejects_changed_artifact_or_inventory(tmp_path,small_geometry):
    result=n.build_noise_dataset(tmp_path,20260916,1,0,0);folder=tmp_path/'datasets'/result['case_id']
    receipt_path=folder/'noise_prepared.json';receipt=json.loads(receipt_path.read_text())
    original=receipt_path.read_bytes();receipt['bindings']=receipt['bindings'][:-1]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='inventory'):n.build_noise_dataset(tmp_path,20260916,1,0,0)
    receipt_path.write_bytes(original)
    with (folder/'Raw.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(RuntimeError,match='artifact changed'):n.build_noise_dataset(tmp_path,20260916,1,0,0)


def test_recipe_or_completed_root_cannot_be_reused(tmp_path,small_geometry,monkeypatch):
    n.build_noise_dataset(tmp_path,20260916,0,0,1)
    monkeypatch.setattr(n,'_source_bindings',lambda:[])
    with pytest.raises(RuntimeError,match='recipe source changed'):n.build_noise_dataset(tmp_path,20260916,0,0,1)
    (tmp_path/'completion_manifest.json').write_text('{}')
    with pytest.raises(RuntimeError,match='immutable'):n.build_noise_dataset(tmp_path,20260916,0,0,1)


def test_failure_never_installs_partial_dataset(tmp_path,small_geometry,monkeypatch):
    def broken(*args):
        yield np.zeros((18,18),dtype=np.float32)
        raise RuntimeError('injected fixture failure')
    monkeypatch.setattr(n,'iter_noise_frames',broken)
    with pytest.raises(RuntimeError,match='injected'):n.build_noise_dataset(tmp_path,20260916,1,1,1)
    assert not (tmp_path/'datasets'/n.case_id(20260916,1,1,1)).exists()
