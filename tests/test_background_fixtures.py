"""Small paired fixtures and independent moment checks; no production run."""
import itertools
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import background_fixtures as b
from neurobench.experiments.gamma_ls_difference import noise_fixtures as old


def tiny(seed=20260916,v=0,s=0,t=0,background="sloped",normalization="raw",**changes):
    config=dict(shape_yx=(9,11),warmup_frames=2,setup_frames=3,
                application_frames=7,variance_step=3)
    config.update(changes)
    return np.stack(list(b._iter_config(seed,v,s,t,background,normalization,**config)))


def test_canonical_identity_matrix_and_explicit_metadata():
    ids={b.case_id(seed,v,s,t,background,norm)
         for seed,background,norm,v,s,t in itertools.product(
             b.SEEDS,b.BACKGROUNDS,b.NORMALIZATIONS,(0,1),(0,1),(0,1))}
    assert len(ids)==84  # 96 requested rows minus the twelve unit-gain aliases.
    assert b.case_id(20260916,1,0,0,"flat","conditioned")=="bg_flat__raw_v1_s0_t0__seed20260916"
    m=b.background_metadata(20260916,1,1,1,"flat","conditioned")
    assert m['shape_tyx']==[464,226,226] and m['evaluation_box_yxyx']==[49,49,177,177]
    assert m['setup_source_frames_ui']==list(range(65,165))
    assert m['factors_start_source_frame_ui']==165 and m['variance_step_source_frame_ui']==265
    assert m['source_frames_ui']==list(range(1,465))
    assert m['event_count']==m['active_region_frame_count']==0
    assert m['B']=='flat' and m['background']['gradient_x_per_px']==0
    assert m['paired_setup_identical_within_seed_and_background']
    assert m['paired_setup_innovations_identical_across_backgrounds']
    assert not m['paired_setup_identical_within_seed']
    assert not m['activity_truth_parsed'] and not m['scientific_promotion']
    assert not m['noise_parameters']['conditioning_transients_matched']


@pytest.mark.parametrize('args',[
    (1,0,0,0,"flat","raw"), (True,0,0,0,"flat","raw"),
    (20260916,2,0,0,"flat","raw"), (20260916,0,0,"1","flat","raw"),
    (20260916,0,0,0,"Flat","raw"), (20260916,0,0,0,"flat","matched"),
    (20260916,0,0,0,None,"raw"), (20260916,0,0,0,"flat",["raw"]),
])
def test_invalid_conventions_rejected(args):
    with pytest.raises(ValueError):b.case_id(*args)


@pytest.mark.parametrize('seed',b.SEEDS)
def test_raw_sloped_exactly_reproduces_every_frozen_factorial(seed):
    config=dict(shape_yx=(9,11),warmup_frames=2,setup_frames=3,
                application_frames=7,variance_step=3)
    for v,s,t in itertools.product((0,1),repeat=3):
        expected=np.stack(list(old._iter_config(seed,v,s,t,**config)))
        actual=tiny(seed,v,s,t,**config)
        assert actual.dtype==np.float32
        assert np.array_equal(actual,expected)


def test_public_iterator_first_frame_matches_frozen_production_geometry():
    assert np.array_equal(next(b.iter_background_frames(20260916,1,1,1,"sloped","raw")),
                          next(old.iter_noise_frames(20260916,1,1,1)))


@pytest.mark.parametrize('background',b.BACKGROUNDS)
def test_unit_gain_alias_is_byte_exact_for_both_variance_levels(background):
    for v in (0,1):
        assert np.array_equal(tiny(v=v,background=background),
                              tiny(v=v,background=background,normalization="conditioned"))
        assert b.background_metadata(20260916,v,0,0,background,"raw")==b.background_metadata(
            20260916,v,0,0,background,"conditioned")


@pytest.mark.parametrize('seed',b.SEEDS)
def test_setup_is_independent_of_factors_and_normalization_with_paired_innovations(seed):
    for background in b.BACKGROUNDS:
        baseline=tiny(seed,background=background)[:5]
        for v,s,t,norm in itertools.product((0,1),(0,1),(0,1),b.NORMALIZATIONS):
            assert np.array_equal(tiny(seed,v,s,t,background,norm)[:5],baseline)
    rng=np.random.default_rng(seed)
    innovations=np.stack([rng.standard_normal((9,11),dtype=np.float32) for _ in range(5)])
    for background in b.BACKGROUNDS:
        assert np.array_equal(tiny(seed,background=background)[:5],
                              b._background((9,11),background)+2.*innovations)


@pytest.mark.parametrize('background',b.BACKGROUNDS)
def test_application_gain_cannot_scale_fixed_background(background,monkeypatch):
    def zero_noise(innovation,state,**kwargs):
        return np.zeros_like(innovation),state
    monkeypatch.setattr(old,'_application_step',zero_noise)
    actual=tiny(s=1,t=1,background=background,normalization="conditioned")
    fixed=b._background((9,11),background)
    assert np.array_equal(actual[5:],np.broadcast_to(fixed,actual[5:].shape))
    assert b.matching_parameters(1,1,"conditioned")['theoretical_application_noise_gain']<1


def test_gain_acts_on_same_noise_before_addition_and_does_not_change_ar_state(monkeypatch):
    recorded=[]
    original=old._application_step
    def observe(*args,**kwargs):
        noise,state=original(*args,**kwargs)
        recorded.append((noise.copy(),state.copy()))
        return noise,state
    monkeypatch.setattr(old,'_application_step',observe)
    actual=tiny(v=1,s=1,t=1,background="sloped",normalization="conditioned")
    first=[(noise.copy(),state.copy()) for noise,state in recorded];recorded.clear()
    tiny(v=1,s=1,t=1,background="flat",normalization="raw")
    gain=np.float32(b.matching_parameters(1,1,"conditioned")['applied_float32_application_noise_gain'])
    for frame,((noise,state),(noise_raw,state_raw)) in enumerate(zip(first,recorded)):
        assert np.array_equal(noise,noise_raw) and np.array_equal(state,state_raw)
        assert np.array_equal(actual[5+frame],b._background((9,11),"sloped")+noise*gain)


def test_stationary_moment_matching_uses_covariance_and_finite_filter_not_sample_fit():
    # Independently contract the actual finite Gaussian with raw covariance.
    from scipy.ndimage import gaussian_filter
    delta=np.zeros((21,21),dtype=np.float64);delta[10,10]=1
    G=gaussian_filter(delta,1.,mode="constant",truncate=4.)
    white_spatial=float(np.sum(G*G))
    correlated_spatial=0.
    for dy,dx in itertools.product(range(-2,3),repeat=2):
        covariance=(3-abs(dy))*(3-abs(dx))/9
        correlated_spatial+=covariance*np.sum(G*np.roll(G,(dy,dx),axis=(0,1)))
    h=.4*.6**np.arange(120)
    ar_cov=.8**np.abs(np.arange(120)[:,None]-np.arange(120)[None,:])
    white_temporal=float(h@h);correlated_temporal=float(h@ar_cov@h)
    for s,t in itertools.product((0,1),repeat=2):
        m=b.matching_parameters(s,t,"conditioned")
        spatial=correlated_spatial if s else white_spatial
        temporal=correlated_temporal if t else white_temporal
        assert m['k_S']==pytest.approx(spatial/white_spatial,rel=2e-14)
        assert m['k_T']==pytest.approx(temporal/white_temporal,rel=2e-14)
        assert m['theoretical_application_noise_gain']**2*spatial*temporal==pytest.approx(
            white_spatial*white_temporal,rel=2e-14)
        assert m['float32_gain_conditioned_variance_ratio_to_white']==pytest.approx(1.,abs=2e-7)
    assert b.matching_parameters(1,1,"conditioned")['k_S']==pytest.approx(5.21409438121709)
    assert b.matching_parameters(1,1,"conditioned")['k_T']==pytest.approx(2.846153846153846)


def test_raw_variance_metadata_scales_only_application_and_distinguishes_level_from_difference():
    meta=b.background_metadata(20260916,1,1,1,"sloped","conditioned")
    m=meta['variance_matching'];n=meta['noise_parameters']
    assert n['setup_noise_std']==2.
    assert n['raw_pointwise_variance_before_step']==pytest.approx(4/(m['k_S']*m['k_T']))
    assert n['raw_pointwise_variance_after_step']==pytest.approx(25/(m['k_S']*m['k_T']))
    assert not m['background_scaled'] and not m['setup_scaled']
    assert 'not signed-difference variance' in m['matched_quantity']
    assert 'stationary' in m['matching_scope'].lower()


def test_v_pairs_have_equal_pre_step_source_and_conditioned_prefixes():
    from neurobench.experiments.gamma_ls_difference.control_study import conditioned_frames
    for background,norm,s,t in itertools.product(b.BACKGROUNDS,b.NORMALIZATIONS,(0,1),(0,1)):
        a=tiny(v=0,s=s,t=t,background=background,normalization=norm)
        z=tiny(v=1,s=s,t=t,background=background,normalization=norm)
        assert np.array_equal(a[:8],z[:8])
        assert not np.array_equal(a[8:],z[8:])
        before=list(conditioned_frames(a,1.,.4));after=list(conditioned_frames(z,1.,.4))
        assert all(np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1])
                   for x,y in zip(before[:8],after[:8]))


def test_future_duration_and_step_do_not_change_existing_prefix():
    for background,norm in itertools.product(b.BACKGROUNDS,b.NORMALIZATIONS):
        a=tiny(v=1,s=1,t=1,background=background,normalization=norm,application_frames=2)
        z=tiny(v=1,s=1,t=1,background=background,normalization=norm,application_frames=9)
        assert np.array_equal(a,z[:len(a)])
        future=tiny(v=1,s=1,t=1,background=background,normalization=norm,variance_step=5)
        assert np.array_equal(z[:8],future[:8])


@pytest.fixture
def small_geometry(monkeypatch):
    monkeypatch.setattr(b,'EVALUATION_SHAPE_YX',(8,8));monkeypatch.setattr(b,'SPATIAL_HALO_PX',5)
    monkeypatch.setattr(b,'WARMUP_FRAMES',1);monkeypatch.setattr(b,'SETUP_FRAMES',2)
    monkeypatch.setattr(b,'APPLICATION_FRAMES',4)


def test_atomic_preparation_full_stage_contract_and_verified_resume(tmp_path,small_geometry):
    from neurobench.experiments.gamma_ls_difference.control_study import conditioned_frames
    result=b.build_background_dataset(tmp_path,20260916,1,1,1,"flat","conditioned")
    folder=tmp_path/'datasets'/result['case_id']
    raw=np.load(folder/'raw_source.npy');meta=json.loads((folder/'metadata.json').read_text())
    assert raw.shape==(7,18,18) and meta['shape_tyx']==list(raw.shape)
    assert np.array_equal(np.load(folder/'Raw.npy'),raw[:,5:13,5:13])
    stages=list(conditioned_frames(raw,1.,.4))
    level=np.stack([v[0] for v in stages]);change=np.stack([v[1] for v in stages])
    for name,expected in [('level_source.npy',level),('input_source.npy',change),
                          ('level.npy',level[:,5:13,5:13]),('Input.npy',change[:,5:13,5:13])]:
        assert np.array_equal(np.load(folder/name),expected)
    for name in ['experts.json','active.json']:assert json.loads((folder/name).read_text())==[]
    receipt=json.loads((folder/'background_prepared.json').read_text())
    assert len(receipt['source_bindings'])==4 and not receipt['activity_truth_parsed']
    assert receipt['factors']==dict(V=1,S=1,T=1,B='flat',normalization='conditioned')
    assert b.build_background_dataset(tmp_path,20260916,1,1,1,"flat","conditioned")['resumed']


def test_unit_alias_resumes_identical_canonical_dataset(tmp_path,small_geometry):
    a=b.build_background_dataset(tmp_path,20260916,1,0,0,"sloped","raw")
    z=b.build_background_dataset(tmp_path,20260916,1,0,0,"sloped","conditioned")
    assert a['case_id']==z['case_id'] and z['resumed'] and a['receipt']==z['receipt']


def test_resume_rejects_artifact_inventory_identity_and_recipe_changes(tmp_path,small_geometry,monkeypatch):
    result=b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    folder=tmp_path/'datasets'/result['case_id'];path=folder/'background_prepared.json'
    saved=path.read_bytes();receipt=json.loads(saved)
    receipt['factors']['B']='sloped';path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='identity'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    receipt=json.loads(saved);receipt['bindings'].pop();path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='inventory'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    receipt=json.loads(saved);receipt['variance_matching']['k_S']=1.;path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='variance-matching'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    receipt=json.loads(saved);receipt['scientific_promotion']=True;path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='scope'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    path.write_bytes(saved)
    with monkeypatch.context() as patch:
        patch.setattr(b,'_source_bindings',lambda:[])
        with pytest.raises(RuntimeError,match='recipe source changed'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    with (folder/'Raw.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(RuntimeError,match='artifact changed'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")


def test_completed_root_and_incomplete_existing_dataset_are_preserved(tmp_path,small_geometry):
    folder=tmp_path/'datasets'/b.case_id(20260916,0,1,1,"flat","raw")
    folder.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='no completed'):b.build_background_dataset(tmp_path,20260916,0,1,1,"flat","raw")
    (tmp_path/'completion_manifest.json').write_text('{}')
    with pytest.raises(RuntimeError,match='immutable'):b.build_background_dataset(tmp_path,20260916,0,1,1,"sloped","raw")


def test_failure_never_installs_partial_dataset(tmp_path,small_geometry,monkeypatch):
    def broken(*args):
        yield np.zeros((18,18),dtype=np.float32)
        raise RuntimeError('injected fixture failure')
    monkeypatch.setattr(b,'iter_background_frames',broken)
    with pytest.raises(RuntimeError,match='injected'):b.build_background_dataset(tmp_path,20260916,1,1,1,"flat","conditioned")
    assert not (tmp_path/'datasets'/b.case_id(20260916,1,1,1,"flat","conditioned")).exists()
