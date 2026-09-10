import numpy as np
from neurobench.experiments.neuron_identifiability.contextual_envelope_retrieval import agreement_attenuated,feature_arrays,trailing_max

def test_trailing_max_is_causal():
 x=np.array([1,3,2,5,4],dtype=np.float32)[:,None,None]
 np.testing.assert_array_equal(trailing_max(x,3)[:,0,0],[1,3,3,5,5])

def test_attenuation_is_bounded_when_envelope_contains_a():
 a=np.array([0,.2,.7,1],dtype=np.float32);u=np.maximum(a,np.array([.5,.5,.9,1],dtype=np.float32));h=agreement_attenuated(a,u);assert np.all(h>=0);assert np.all(h<=a+1e-6)

def test_controls_and_identity_shapes():
 rng=np.random.default_rng(3);a=rng.random((20,5,6),dtype=np.float32);f=feature_arrays(a);assert set(f)=={"A","A2","H_t3","H_t5","H_st5k3","H_t5_shuffle137","H_st5k3_displace11x17"};assert all(v.shape==a.shape for v in f.values());np.testing.assert_allclose(f["A2"],a*a)
