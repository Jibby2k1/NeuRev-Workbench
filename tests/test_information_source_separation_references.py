import numpy as np

from neurobench.experiments.information_source_separation.references import (
    fit_amplitude_pca_reference,
    fit_dense_patch_fastica_wiener_reference,
    fit_spatial_fastica_reference,
)
from neurobench.experiments.information_source_separation.synthetic import (
    make_spatiotemporal_fixture,
)


def test_existing_temporal_references_reconstruct_declared_rank() -> None:
    fixture = make_spatiotemporal_fixture(
        "isolated", seed=7, frame_count=128, shape=(10, 10)
    )
    pca = fit_amplitude_pca_reference(fixture.observation, rank=4)
    ica = fit_spatial_fastica_reference(
        fixture.observation, rank=4, seed=7, max_iterations=200
    )
    for result in (pca, ica):
        assert result.reconstruction.shape == fixture.observation.shape
        assert result.spatial_maps.shape == (100, 4)
        assert result.temporal_sources.shape == (4, 128)
        assert np.isfinite(result.reconstruction).all()


def test_dense_patch_fastica_wiener_has_exact_explicit_remainder() -> None:
    fixture = make_spatiotemporal_fixture(
        "isolated", seed=11, frame_count=96, shape=(9, 9)
    )
    result = fit_dense_patch_fastica_wiener_reference(
        fixture.observation,
        quiet_frames=24,
        patch_size=5,
        rank=4,
        sample_count=100,
        seed=11,
    )
    assert result["signal"].shape == fixture.observation.shape
    assert np.allclose(
        result["signal"] + result["remainder"], fixture.observation,
        atol=1e-5,
    )
    assert result["scientific_trace_status"].startswith("auxiliary")
