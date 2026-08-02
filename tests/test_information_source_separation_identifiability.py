import numpy as np

from neurobench.experiments.information_source_separation.identifiability import (
    ALL_IDENTIFIABILITY_CASES,
    IDENTIFIABLE_CASES,
    UNIDENTIFIABLE_CASES,
    assert_distinct_case_contract,
    make_identifiability_fixture,
)


def test_identifiability_cases_are_numerically_distinct_across_labels() -> None:
    audit = assert_distinct_case_contract(seed=101)
    assert audit["collisions"] == []
    assert set(IDENTIFIABLE_CASES).isdisjoint(UNIDENTIFIABLE_CASES)


def test_rank_deficient_fixtures_close_and_have_truthful_metadata() -> None:
    for case in ALL_IDENTIFIABILITY_CASES:
        fixture = make_identifiability_fixture(
            case, seed=103, frame_count=128, shape=(10, 12)
        )
        closure = fixture.observation - fixture.background - fixture.neural_signal - fixture.structured_artifact - fixture.measurement_noise
        assert float(np.max(np.abs(closure))) < 1e-5
        assert fixture.identifiable == (case in IDENTIFIABLE_CASES)
    spatial = make_identifiability_fixture("spatial_rank_deficient", seed=103)
    temporal = make_identifiability_fixture("temporal_rank_deficient", seed=103)
    assert spatial.metadata["mixing_rank"] < 3
    assert temporal.metadata["temporal_rank"] < 3
