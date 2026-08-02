from pathlib import Path

from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
    FEATURE_IDS,
    FEATURE_SETS,
    PROPOSAL_SOURCE_IDS,
    InnovationRankerConfig,
)


def test_nested_ranker_manifest_has_exact_frozen_counts() -> None:
    config = InnovationRankerConfig.load(
        Path("examples/spon_ca_burst_innovation_ranker_v1.example.json")
    )
    assert len(FEATURE_IDS) == 34
    assert len(PROPOSAL_SOURCE_IDS) == 22
    assert len(FEATURE_SETS) == 5
    assert config.linear_config_count == 135
    assert config.mlp_config_count == 120
    assert config.inner_fit_count == 2250
    assert config.outer_refit_count == 16
