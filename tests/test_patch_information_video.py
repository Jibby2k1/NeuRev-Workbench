import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.patch_information_video import (
    display_limits,
    parse_feature_id,
)


def test_parse_feature_id_recovers_frozen_parameters():
    assert parse_feature_id("cs_quiet__p7__bw0p5") == (
        "cs_quiet_divergence",
        7,
        0.5,
    )
    assert parse_feature_id("renyi2_ip__p11__bw2") == (
        "renyi2_information_potential",
        11,
        2.0,
    )


def test_parse_feature_id_rejects_noncanonical_or_unknown_ids():
    with pytest.raises(ValueError):
        parse_feature_id("cs_quiet__p07__bw0p5")
    with pytest.raises(ValueError):
        parse_feature_id("entropy__p7__bw0p5")


def test_display_limits_use_quiet_black_and_global_upper_tail():
    values = np.zeros((8, 4, 4), dtype=np.float32)
    values[:4] = 1.0
    values[4:] = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 2.0
    black, white = display_limits(
        values, quiet_count=4, upper_percentile=95.0, stride=1
    )
    assert black == pytest.approx(1.0)
    assert white > black
