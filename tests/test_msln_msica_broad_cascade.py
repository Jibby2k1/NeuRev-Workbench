import json
from pathlib import Path

import numpy as np

from neurobench.experiments.msln_msica.broad_cascade import (
    EXPERIMENTS,
    _mean_pair_correlation,
    _validate,
)


def test_v4_manifest_freezes_full_factorial():
    path = Path("examples/spon_ca_burst_msln_msica_broad_cascade_v4.example.json")
    config = json.loads(path.read_text())
    config["_config_path"] = str(path.resolve())
    _validate(config)
    assert config["sweep"]["full_context_pairs"] is True
    assert len(config["sweep"]["spatial_outer_guard_pairs"]) * len(config["sweep"]["temporal_windows_frames"]) == 30
    assert len(EXPERIMENTS) == 6


def test_mean_pair_correlation_identical_maps():
    values = np.arange(20 * 14 * 14, dtype=np.float32).reshape(20, 14, 14)
    assert np.isclose(_mean_pair_correlation([values, values.copy(), values.copy()]), 1.0)
