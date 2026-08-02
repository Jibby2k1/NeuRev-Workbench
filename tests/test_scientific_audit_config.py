from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_config import (
    ScientificAuditConfig,
)


def test_scientific_audit_config_has_frozen_counts():
    config = ScientificAuditConfig.load(
        "examples/spon_ca_burst_scientific_feature_audit_v1.example.json"
    )
    assert config.feature_count == 16
    assert config.lane_count_per_field == 64
    assert config.evaluated_lane_count == 192
