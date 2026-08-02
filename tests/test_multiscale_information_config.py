from neurobench.experiments.hierarchical_parzen_ica.multiscale_information_config import (
    MultiscaleInformationConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.multiscale_information_program import (
    _feature_inventory,
)


def test_multiscale_information_config_has_frozen_counts():
    config = MultiscaleInformationConfig.load(
        "examples/spon_ca_burst_multiscale_information_v1.example.json"
    )
    assert config.base_feature_count == 12
    assert config.fused_feature_count == 30
    assert config.feature_count == 42
    assert config.lane_count == 168
    feature_ids, families = _feature_inventory(config)
    assert len(feature_ids) == len(set(feature_ids)) == 42
    assert set(families.values()) == {
        "single_scale",
        "scale_max",
        "soft_scale_selection",
        "adjacent_scale_agreement",
        "compact_broad_contrast",
        "center_annulus",
    }
