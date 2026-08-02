import json

import numpy as np

from neurobench.experiments.information_source_separation.config import InformationSeparationConfig
from neurobench.experiments.information_source_separation.consensus import fit_multistart_consensus
from neurobench.experiments.information_source_separation.synthetic import make_spatiotemporal_fixture


def test_multistart_consensus_is_deterministic_and_label_free():
    config = InformationSeparationConfig.load(
        "examples/spon_ca_burst_information_source_separation_gpu_v1.example.json"
    )
    movie = make_spatiotemporal_fixture(
        "isolated", seed=3, frame_count=96, shape=(8,8), snr=8
    ).observation
    kwargs = dict(base_method="multilag_sobi", rank=4, starts=3,
                  scientific_config=config, seed=9, device="cpu")
    left = fit_multistart_consensus(movie, **kwargs)
    right = fit_multistart_consensus(movie, **kwargs)
    assert np.array_equal(left["sources"], right["sources"])
    assert left["diagnostics"] == right["diagnostics"]
    assert json.dumps(left["diagnostics"])
