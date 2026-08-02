import numpy as np

from neurobench.experiments.information_source_separation.config import (
    InformationSeparationConfig,
)
from neurobench.experiments.information_source_separation.qualification import (
    qualify_temporal_components,
)
from neurobench.experiments.information_source_separation.references import (
    fit_amplitude_pca_reference,
)
from neurobench.experiments.information_source_separation.screen_runner import (
    _execute_method,
)
from neurobench.experiments.information_source_separation.synthetic import (
    make_spatiotemporal_fixture,
)


def test_one_screen_fit_uses_exact_reference_and_label_free_qualification() -> None:
    config = InformationSeparationConfig.load(
        "examples/spon_ca_burst_information_source_separation_v1.example.json"
    )
    fixture = make_spatiotemporal_fixture(
        "isolated", seed=7, frame_count=128, shape=(10, 10)
    )
    direct = fit_amplitude_pca_reference(fixture.observation, rank=4)
    execution = _execute_method(
        fixture.observation, "pca_reference", {"rank": 4}, config, 7
    )
    assert execution["reported_method_id"] == "amplitude_pca_reference"
    assert np.allclose(execution["sources"], direct.temporal_sources)
    qualification = qualify_temporal_components(
        execution["spatial_maps"], execution["sources"], spatial_shape=(10, 10)
    )
    assert qualification["selection_uses_labels"] is False
