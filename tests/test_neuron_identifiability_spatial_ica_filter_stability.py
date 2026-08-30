import numpy as np
from neurobench.experiments.neuron_identifiability.spatial_ica_filter_stability import compare_filters
def test_filter_comparison_handles_sign_and_permutation():
    rng=np.random.default_rng(4);reference=rng.normal(size=(4,9));estimate=reference[[2,0,3,1]]*np.array([-1,1,-1,1])[:,None]
    result=compare_filters(reference,estimate)
    assert result["median_aligned_abs_correlation"]>.999
    assert result["mean_subspace_cosine"]>.999
