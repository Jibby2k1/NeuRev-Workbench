import numpy as np

from neurobench.algorithms.latent_dynamics import kalman_filter_ar1, rts_smoother_ar1, stable_ar1_from_decay
from neurobench.experiments.latent_dynamics.synthetic import generate_synthetic_case, synthetic_suite
from neurobench.metrics.latent_signal import latent_reconstruction_metrics


def test_synthetic_suite_is_deterministic_and_covers_falsification_cases():
    a=synthetic_suite(seeds=(7,)); b=synthetic_suite(seeds=(7,))
    assert len(a)==11 and [x.case_id for x in a]==[x.case_id for x in b]
    for x,y in zip(a,b): np.testing.assert_array_equal(x.observation,y.observation)


def test_smoother_improves_transient_nmse_without_large_peak_shift():
    case=generate_synthetic_case("transient",seed=13,signals=1,observation_std=.4)
    model=stable_ar1_from_decay(20,-20/np.log(.92),.12**2,.4**2)
    smooth=rts_smoother_ar1(kalman_filter_ar1(case.observation,model),model)
    raw=latent_reconstruction_metrics(case.latent,case.observation); denoised=latent_reconstruction_metrics(case.latent,smooth.mean)
    assert denoised["nmse"]<raw["nmse"] and abs(denoised["peak_time_error_frames"])<=3


def test_noise_free_identity_and_slow_ramp_are_not_erased():
    identity=generate_synthetic_case("identity",signals=1); model=stable_ar1_from_decay(20,1e8,.1,1e-12,initial_variance=1e6)
    result=kalman_filter_ar1(identity.observation,model,initial_mean=1.0,output_dtype=np.float64)
    np.testing.assert_allclose(result.filter_mean,identity.latent,atol=2e-6)
    ramp=generate_synthetic_case("slow_ramp",signals=1); smooth=rts_smoother_ar1(kalman_filter_ar1(ramp.observation,stable_ar1_from_decay(20,320,.03,.12)),stable_ar1_from_decay(20,320,.03,.12))
    assert smooth.mean[-1,0]>1.0 and smooth.mean[ramp.event_interval[1]-1,0]>1.0


def test_pure_noise_does_not_create_persistent_positive_state():
    case=generate_synthetic_case("pure_noise",signals=3); model=stable_ar1_from_decay(20,160,.01,.35**2)
    state=kalman_filter_ar1(case.observation,model).filter_mean
    assert np.max(np.mean(state>1.0,axis=0))<.1


def test_reconstruction_peak_timing_uses_frame_axis_for_multiple_signals():
    truth=np.zeros((10,3)); estimate=np.zeros_like(truth)
    truth[4,2]=2; estimate[6,1]=2
    metrics=latent_reconstruction_metrics(truth,estimate)
    assert metrics["truth_peak_frame"]==4 and metrics["estimated_peak_frame"]==6
    assert metrics["peak_time_error_frames"]==2
