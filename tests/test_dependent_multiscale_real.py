import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_real import (
    _linear_proxy,
)


def test_full_frame_failure_analysis_proxy_closes_and_is_explicitly_diagnostic():
    rng = np.random.default_rng(31)
    observation = rng.normal(size=(12, 15, 17)).astype(np.float32)
    views = {
        "scale_5": 0.8 * observation,
        "scale_7": 0.7 * observation,
        "scale_15": 0.3 * observation,
    }
    result = _linear_proxy(observation, views)
    restored = (
        result.background + result.structured_signal
        + result.structured_artifact + result.noise_candidate
        + result.closure_residual
    )
    np.testing.assert_allclose(restored, observation, atol=2e-6)
    assert result.diagnostics["scientific_status"] == "diagnostic_only_do_not_advance"
    assert not result.diagnostics["patchwise_W7_claim"]
