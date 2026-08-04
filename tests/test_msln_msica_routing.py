import numpy as np

from neurobench.experiments.msln_msica.routing import (
    bounded_residual_gate,
    product_interaction,
    route_evidence,
)


def test_fixed_routing_and_product_preserve_contract() -> None:
    evidence = np.asarray([[[1, 4]], [[4, 1]], [[2, 2]]], dtype=np.float32)
    ids = ("spatial_5_meanstd", "spatial_7_meanstd", "spatial_15_meanstd")
    compact, dominant = route_evidence(evidence, ids, mode="compact_agreement")
    np.testing.assert_allclose(compact, [[2, 2]])
    np.testing.assert_array_equal(dominant, [[1, 0]])
    raw = np.asarray([[10, -10]], dtype=np.float32)
    visual = product_interaction(raw, compact, beta=.25, kappa=2)
    assert np.all(np.sign(visual) == np.sign(raw))
    assert np.all(np.abs(visual) <= np.abs(raw))


def test_bounded_residual_gate_has_floor_and_saturates() -> None:
    values = np.asarray([-100.0, 0.0, 2.0, 100.0], dtype=np.float32)
    gate = bounded_residual_gate(values, beta=0.25, kappa=2.0)
    assert gate[1] == 0.25
    assert gate[2] == 0.625
    assert np.all(gate >= 0.25)
    assert np.all(gate <= 1.0)
