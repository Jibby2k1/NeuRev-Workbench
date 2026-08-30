import numpy as np
from neurobench.experiments.neuron_identifiability.spatial_ica_class_certainty_alignment import _permutation_kruskal

def test_permutation_kruskal_detects_separated_fixture():
    values=np.r_[np.zeros(8),np.ones(8)*4,np.ones(8)*8];classes=np.repeat([1,2,3],8)
    result=_permutation_kruskal(values,classes,repeats=1000,seed=4)
    assert result["epsilon_squared"]>.8
    assert result["site_label_permutation_p"]<.01
