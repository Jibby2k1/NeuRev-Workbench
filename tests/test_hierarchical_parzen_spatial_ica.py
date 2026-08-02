from pathlib import Path

import pytest

from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_config import (
    SpatialICAConfig,
    SpatialICAConfigError,
)
from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_screen import (
    VARIANT_IDS,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_spatial_ica_screen.example.json"


def test_spatial_ica_example_is_valid_and_resolves_paths() -> None:
    config = SpatialICAConfig.load(EXAMPLE)
    assert config.variant_count == 3
    assert len(VARIANT_IDS) == config.variant_count
    assert config.source_video.is_absolute()
    assert config.model["patch_size"] == 11
    assert config.resources["frame_batch_size"] == 1


def test_spatial_ica_config_rejects_even_patch() -> None:
    config = SpatialICAConfig.load(EXAMPLE)
    bad = dict(config.model)
    bad["patch_size"] = 10
    with pytest.raises(SpatialICAConfigError):
        SpatialICAConfig(
            **{**config.__dict__, "model": bad}
        ).validate()
