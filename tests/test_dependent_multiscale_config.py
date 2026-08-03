import json

import pytest

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_config import (
    DependentMultiscaleConfig,
    DependentMultiscaleConfigError,
)


def test_example_config_freezes_v1_and_resolves_paths():
    config = DependentMultiscaleConfig.load(
        "examples/spon_ca_burst_dependent_multiscale_v1.example.json"
    )
    assert config.views["supports_px"] == [5, 7, 15]
    assert config.optimization["seeds"] == [7, 13, 19]
    assert config.input["labels_role"] == "evaluation_only"
    assert config.preflight_dir.is_absolute()


def test_config_rejects_unknown_fields_and_duplicate_normalization(tmp_path):
    source = json.loads(
        open("examples/spon_ca_burst_dependent_multiscale_v1.example.json", encoding="utf-8").read()
    )
    source["views"]["unknown"] = 1
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(DependentMultiscaleConfigError, match="unknown"):
        DependentMultiscaleConfig.load(path)
    del source["views"]["unknown"]
    source["input"]["input_normalization_state"] = "quiet_standardized"
    source["views"]["normalization_kind"] = "quiet_robust"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(DependentMultiscaleConfigError, match="twice"):
        DependentMultiscaleConfig.load(path)
