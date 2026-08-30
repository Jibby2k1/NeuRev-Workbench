from neurobench.experiments.neuron_identifiability.strict_nms_audit import BUDGETS, LANES, RADII


def test_strict_nms_contract_is_frozen():
    assert RADII == (4,6,8)
    assert BUDGETS == (20,40,58,80,100)
    assert LANES == ("carrier_signed","coherence_w15","propagation_lag2_w15")
