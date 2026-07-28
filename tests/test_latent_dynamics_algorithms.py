import numpy as np

from neurobench.algorithms.latent_dynamics import (
    StableAR1, dynamic_drive, estimate_quiet_noise, fit_shared_ar1_grid,
    kalman_filter_ar1, rts_smoother_ar1, stable_ar1_from_decay,
    standardized_filter_innovation, state_difference, validate_stable_ar1,
)


def _model(gamma=.8,q=.1,r=.2,p0=.5):
    return StableAR1(gamma,q,r,p0,1-gamma,"resolved",{})


def test_stability_parameterization_has_margin_at_extreme_decay():
    model=stable_ar1_from_decay(20,1e12,.1,.2,stability_epsilon=.01)
    assert model.gamma==.99 and np.isclose(model.stability_margin,.01)
    validate_stable_ar1(model,required_margin=.01)


def test_scalar_filter_matches_hand_calculated_first_update():
    result=kalman_filter_ar1(np.array([1.0]),_model(),output_dtype=np.float64)
    np.testing.assert_allclose(result.filter_mean,[5/7],rtol=1e-12)
    np.testing.assert_allclose(result.filter_variance,[1/7],rtol=1e-12)
    np.testing.assert_allclose(result.innovation_variance,[.7],rtol=1e-12)


def test_filter_covariance_positive_and_vectorized_matches_scalar_loops():
    rng=np.random.default_rng(4); observations=rng.normal(size=(30,5)); model=_model()
    vector=kalman_filter_ar1(observations,model,output_dtype=np.float64)
    scalar=np.column_stack([kalman_filter_ar1(observations[:,i],model,output_dtype=np.float64).filter_mean for i in range(5)])
    np.testing.assert_allclose(vector.filter_mean,scalar,atol=1e-12)
    assert np.all(vector.filter_variance>0) and np.isfinite(vector.log_likelihood)


def test_rts_matches_dense_gaussian_conditioning_reference():
    model=_model(gamma=.7,q=.15,r=.3,p0=.4); y=np.array([.2,-.4,.8,.3])
    filtered=kalman_filter_ar1(y,model,output_dtype=np.float64); smooth=rts_smoother_ar1(filtered,model,output_dtype=np.float64)
    t=len(y); prior=np.empty(t); prior[0]=model.initial_variance
    for i in range(1,t): prior[i]=model.gamma**2*prior[i-1]+model.process_variance
    cov=np.empty((t,t))
    for i in range(t):
        for j in range(t): cov[i,j]=model.gamma**abs(i-j)*prior[min(i,j)]
    posterior_mean=cov@np.linalg.solve(cov+model.observation_variance*np.eye(t),y)
    posterior_cov=cov-cov@np.linalg.solve(cov+model.observation_variance*np.eye(t),cov)
    np.testing.assert_allclose(smooth.mean,posterior_mean,atol=1e-10)
    np.testing.assert_allclose(smooth.variance,np.diag(posterior_cov),atol=1e-10)
    assert smooth.diagnostics["causal"] is False


def test_float32_output_tracks_float64_reference():
    y=np.random.default_rng(3).normal(size=(100,4)); model=_model()
    a=kalman_filter_ar1(y,model,output_dtype=np.float32); b=kalman_filter_ar1(y,model,output_dtype=np.float64)
    np.testing.assert_allclose(a.filter_mean,b.filter_mean,rtol=1e-6,atol=1e-6)


def test_difference_drive_identities_and_undefined_leading_frames():
    state=np.array([1.,2.,4.,7.]); difference=state_difference(state,1); drive=dynamic_drive(state,1)
    assert np.isnan(difference[0]) and np.isnan(drive[0]); np.testing.assert_allclose(difference[1:],drive[1:])
    lag4=state_difference(np.arange(8.),4); assert np.isnan(lag4[:4]).all() and np.all(lag4[4:]==4)


def test_quiet_noise_floor_and_standardized_innovation_are_finite():
    quiet=np.zeros((10,3)); quiet[:,1]=np.arange(10); noise=estimate_quiet_noise(quiet)
    assert noise.scale_floor>0 and np.all(noise.scale>0) and np.all(noise.difference_variance>0)
    z=standardized_filter_innovation(kalman_filter_ar1(quiet,_model()))
    assert np.isfinite(z).all()


def test_grid_fit_is_deterministic_and_strictly_stable():
    y=np.random.default_rng(8).normal(size=(40,6))
    args=dict(frame_period_ms=20,decay_time_ms_grid=[40,80,160],process_to_observation_grid=[.03,.1],observation_variance=.5)
    a,ca=fit_shared_ar1_grid(y,**args); b,cb=fit_shared_ar1_grid(y,**args)
    assert a==b and ca==cb and a.gamma<1 and len(ca)==6
