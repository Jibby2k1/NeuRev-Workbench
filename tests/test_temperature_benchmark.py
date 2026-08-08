import numpy as np

from neurobench.experiments.msln_msica.temperature_benchmark import _pool
from neurobench.reports.zero_anchored_display import zero_anchored_exp2


def test_monotone_temperature_is_max_pool_ranking_invariant() -> None:
    values = np.asarray([[[0.0, 1.0], [2.0, -1.0]], [[1.5, 0.5], [1.0, 3.0]]], dtype=np.float32)
    rankings = []
    for alpha in (0.025, 0.1, 0.25, 1.0):
        transformed = zero_anchored_exp2(values, alpha=alpha, positive_max=3.0)
        rankings.append(np.argsort(_pool(transformed, "max").ravel()).tolist())
    assert all(row == rankings[0] for row in rankings)


def test_topk_pooling_contract() -> None:
    values = np.arange(5 * 2, dtype=np.float32).reshape(5, 1, 2)
    np.testing.assert_allclose(_pool(values, "top3_mean"), [[6.0, 7.0]])
