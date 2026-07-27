import numpy as np

from neurobench.experiments.pairwise_separation.fusion import additive_fusion, fit_bounded_lambda, soft_gate


def test_soft_gate_preserves_floor_and_activity_support():
    original=np.full((2,3,4),100,np.float32); feature=np.zeros_like(original); feature[:,1,2]=1
    result=soft_gate(original,feature,0.7)
    assert np.all(result[:,:,0]==70) and np.all(result[:,1,2]==100)


def test_additive_zero_is_exact_raw_initialization():
    raw=np.arange(12,dtype=np.float32).reshape(1,3,4); feature=np.ones_like(raw)
    np.testing.assert_array_equal(additive_fusion(raw,feature,0,2),raw)


def test_bounded_lambda_moves_only_when_feature_improves_ranking():
    raw_p=np.array([0.5,0.6]); raw_n=np.array([0.55,0.65])
    weight,history=fit_bounded_lambda(raw_p,np.ones(2),raw_n,np.zeros(2),learning_rate=0.01,epochs=200,l2=0.01,maximum=0.4)
    assert 0 < weight <= 0.4 and history[-1] < history[0]
