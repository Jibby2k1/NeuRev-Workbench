import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_synthetic import (
    FIXTURE_IDS,
    make_fixture,
)


def test_all_required_fixtures_have_exact_truth_and_determinism():
    assert len(FIXTURE_IDS) == 15
    for fixture_id in FIXTURE_IDS:
        first = make_fixture(fixture_id, seed=7)
        second = make_fixture(fixture_id, seed=7)
        restored = first.background + first.structured_signal + first.structured_artifact + first.noise
        np.testing.assert_allclose(restored, first.observation, atol=2e-7)
        np.testing.assert_array_equal(first.observation, second.observation)
        assert first.observation.shape == (24, 25, 25)
    broad = make_fixture("broad_legitimate_neural_source")
    assert np.sum(broad.structured_signal**2) > 0
    motion = make_fixture("motion_edge_without_neural_activity")
    assert np.sum(motion.structured_artifact**2) > 0
    assert np.sum(motion.structured_signal**2) == 0
    quiet = make_fixture("pure_quiet_noise")
    assert np.sum(quiet.structured_signal**2) == 0
