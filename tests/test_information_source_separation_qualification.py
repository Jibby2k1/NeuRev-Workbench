import numpy as np

from neurobench.experiments.information_source_separation.qualification import (
    qualify_temporal_components,
)


def test_qualification_resolves_compact_bursty_source() -> None:
    rng = np.random.default_rng(7)
    shape = (12, 12)
    maps = rng.normal(scale=0.02, size=(144, 3))
    maps[52:55, 0] += 2.0
    traces = rng.normal(scale=0.1, size=(3, 160))
    for frame in (45, 100):
        traces[0, frame:frame + 20] += np.exp(-np.arange(20) / 7)
    result = qualify_temporal_components(maps, traces, spatial_shape=shape)
    assert result["status"] == "resolved"
    assert result["selected_component"] == 0
    assert not result["selection_uses_labels"]


def test_qualification_abstains_on_dense_gaussian_noise() -> None:
    rng = np.random.default_rng(13)
    maps = rng.normal(size=(100, 4))
    traces = rng.normal(size=(4, 192))
    result = qualify_temporal_components(
        maps, traces, spatial_shape=(10, 10)
    )
    assert result["status"] == "unresolved"
    assert result["selected_component"] is None
