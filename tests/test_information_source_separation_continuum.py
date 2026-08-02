import numpy as np

from neurobench.experiments.information_source_separation.continuum import (
    make_continuum_fixture, space_filling_continuum,
)


def test_continuum_is_deterministic_distinct_and_closes():
    left = space_filling_continuum(12, seed=17, split="test")
    right = space_filling_continuum(12, seed=17, split="test")
    assert left == right
    assert sum(row.identifiable for row in left) == 9
    assert len({row.fixture_id for row in left}) == 12
    for specification in left:
        fixture = make_continuum_fixture(specification, frame_count=96, shape=(8, 8))
        closure = fixture.observation-fixture.background-fixture.neural_signal-fixture.structured_artifact-fixture.measurement_noise
        assert np.max(np.abs(closure)) < 2e-6
        assert fixture.identifiable == specification.identifiable
