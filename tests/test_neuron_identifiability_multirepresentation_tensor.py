import numpy as np

from neurobench.experiments.neuron_identifiability.cross_neural_ica import REPRESENTATIONS
from neurobench.experiments.neuron_identifiability.functional_heldout_rank import nested_lobo


def test_all_declared_representations_accept_matched_tensor_contract():
    rng=np.random.default_rng(8); site=rng.normal(size=8); time=rng.normal(size=10); burst=np.array([.8,1.,1.2,1.4])
    cube=np.einsum("i,j,k->ijk",site,burst,time)
    results={representation:nested_lobo(cube) for representation in REPRESENTATIONS}
    assert set(results)==set(REPRESENTATIONS)
    assert all(result["gate"]["passed"] for result in results.values())
