import numpy as np
from neurobench.algorithms.local_whitening import (apply_spatial_center_response,apply_temporal_center_response,deterministic_spatial_samples,deterministic_temporal_samples,fit_fractional_whitening,full_joint_allowed,raw_preserving_blend)
from neurobench.experiments.learned_operator_selection.operators import synthetic_validation

def test_fractional_whitening_synthetic_contract() -> None:
    assert synthetic_validation()["passed"] is True

def test_sampling_and_center_alignment_are_deterministic() -> None:
    rng=np.random.default_rng(4); values=rng.normal(size=(9,8,7)).astype(np.float32)
    np.testing.assert_array_equal(deterministic_spatial_samples(values,3,maximum_samples=40,seed=3),deterministic_spatial_samples(values,3,maximum_samples=40,seed=3))
    np.testing.assert_array_equal(deterministic_temporal_samples(values,3,maximum_samples=40,seed=3),deterministic_temporal_samples(values,3,maximum_samples=40,seed=3))
    sf=fit_fractional_whitening(deterministic_spatial_samples(values,3,maximum_samples=300),shrinkage=1,exponent=0,eigen_floor_ratio=1e-6)
    tf=fit_fractional_whitening(deterministic_temporal_samples(values,3,maximum_samples=300),shrinkage=1,exponent=0,eigen_floor_ratio=1e-6)
    np.testing.assert_allclose(apply_spatial_center_response(values,sf,3),values-sf.mean[4],atol=2e-6)
    np.testing.assert_allclose(apply_temporal_center_response(values,tf,3),values-tf.mean[1],atol=2e-6)

def test_raw_bypass_and_joint_gate() -> None:
    raw=np.arange(12,dtype=np.float32).reshape(3,4); assert np.array_equal(raw_preserving_blend(raw,raw*0,0),raw)
    assert full_joint_allowed(dimension=129,nominal_samples=10000,condition_number=1,effective_rank_fraction=1,spectra_stable=True,memory_ready=True,separable_promising=True)[0] is False
