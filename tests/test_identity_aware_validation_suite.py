from neurobench.experiments.neuron_identifiability.identity_aware_validation_suite import exact_p


def test_exact_permutation_is_bounded_and_symmetric():
    p1 = exact_p([3, 4, 5], [0, 1])
    p2 = exact_p([0, 1], [3, 4, 5])
    assert 0 < p1 <= 1
    assert p1 == p2
