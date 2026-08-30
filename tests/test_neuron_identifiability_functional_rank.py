import numpy as np

from neurobench.experiments.neuron_identifiability.functional_heldout_rank import _cp_als, _cp_reconstruct, nested_lobo


def test_cp_als_recovers_rank_one_tensor():
    x=np.einsum("i,j,k->ijk",np.arange(1,6),np.arange(1,5),np.arange(1,7)).astype(float)
    factors,loss=_cp_als(x,1,0)
    assert loss < 1e-10
    assert np.mean((_cp_reconstruct(factors)-x)**2) < 1e-8


def test_nested_lobo_selects_stable_rank_one_structure():
    rng=np.random.default_rng(3); site=rng.normal(size=8); time=rng.normal(size=10); burst=np.array([.8,1.0,1.2,1.4])
    cube=np.einsum("i,j,k->ijk",site,burst,time)
    result=nested_lobo(cube)
    assert result["gate"]["passed"]
    assert result["gate"]["consensus_rank"] == 1
    assert all(row["selected_rank_one_se"] == 1 for row in result["outer_folds"])
