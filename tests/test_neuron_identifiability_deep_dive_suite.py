from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.deep_dive_suite import (
    _kernel,
    proposal_ranking,
)


def _screen(carrier: float, coherence: float, lag: float) -> dict:
    budgets = {str(value): carrier for value in (20, 40, 58, 80, 100)}
    return {
        "carrier_baseline": {"budget_mean_recall": budgets},
        "rows": [
            {"config_id": "standalone__coherence_w15", "budget_mean_recall": {key: coherence for key in budgets}},
            {"config_id": "standalone__propagation_lag2_w15", "budget_mean_recall": {key: lag for key in budgets}},
        ],
    }


def test_proposal_ranking_contrast_uses_within_universe_carrier() -> None:
    rows = proposal_ranking(_screen(.50, .60, .65), _screen(.40, .43, .42))
    coherence = next(row for row in rows if row["feature_id"] == "coherence_w15" and row["budget"] == 20)
    assert np.isclose(coherence["native_delta_vs_carrier"], .10)
    assert np.isclose(coherence["common_proposal_ranking_delta_vs_carrier"], .03)
    assert np.isclose(coherence["native_minus_common_delta_contrast"], .07)
    assert "not_additive_causal" in coherence["interpretation"]


def test_kinetic_kernel_is_positive_causal_and_normalized() -> None:
    kernel = _kernel(rise=2, decay=10)
    assert kernel.shape == (80,)
    assert np.all(kernel > 0)
    assert np.isclose(kernel.sum(), 1.0)
    assert int(np.argmax(kernel)) > 0
