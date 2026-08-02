import numpy as np

from neurobench.algorithms.spatial_patch_ica import (
    fit_parzen_shrinkage,
    fit_spatial_patch_fastica,
    sample_spatial_patches,
    shrink_components,
)
from neurobench.algorithms.spatial_patch_ica_reconstruction import (
    dense_convolutional_reconstruction,
    patch_lattice_reconstruction,
)


def _movie() -> np.ndarray:
    rng = np.random.default_rng(12)
    video = 0.15 * rng.normal(size=(48, 20, 22))
    yy, xx = np.ogrid[:20, :22]
    blob = np.exp(-((xx - 11) ** 2 + (yy - 9) ** 2) / 5.0)
    video[20:32] += np.linspace(0, 2, 12)[:, None, None] * blob
    return video.astype(np.float32)


def test_spatial_patch_fastica_has_compatible_analysis_and_synthesis() -> None:
    video = _movie()
    patches = sample_spatial_patches(
        video, patch_size=5, sample_count=800, seed=4
    )
    model = fit_spatial_patch_fastica(patches, rank=4, seed=5)
    assert model.analysis_filters.shape == (4, 25)
    assert model.synthesis_atoms.shape == (25, 4)
    assert np.allclose(
        model.analysis_filters @ model.synthesis_atoms,
        np.eye(4),
        atol=2e-5,
    )
    assert model.diagnostics()["explained_variance_sum"] <= 1.000001


def test_patch_and_dense_application_return_finite_movies() -> None:
    video = _movie()
    patches = sample_spatial_patches(
        video, patch_size=5, sample_count=800, seed=6
    )
    model = fit_spatial_patch_fastica(patches, rank=4, seed=7)
    patchwise = patch_lattice_reconstruction(
        video, model, stride=3, shrinkage="wiener", lambda_z=1
    )
    dense, diagnostics = dense_convolutional_reconstruction(
        video,
        model,
        shrinkage="wiener",
        lambda_z=1,
        device="cpu",
        frame_batch_size=4,
    )
    assert patchwise.shape == video.shape
    assert dense.shape == video.shape
    assert np.isfinite(patchwise).all()
    assert np.isfinite(dense).all()
    assert diagnostics["application_stride"] == 1


def test_parzen_component_shrinkage_is_bounded_and_shape_preserving() -> None:
    rng = np.random.default_rng(9)
    samples = np.concatenate(
        (rng.normal(0, 0.5, 2000), rng.normal(3, 0.7, 500))
    )
    posterior = fit_parzen_shrinkage(
        samples,
        maximum_centers=24,
        zero_fraction=0.5,
        active_threshold_z=1.2,
        bandwidth=0.5,
        noise_variance=1,
        lookup_points=1024,
        lookup_abs_z=12,
    )
    components = rng.normal(size=(30, 4)).astype(np.float32)
    clean = shrink_components(
        components,
        np.ones(4, dtype=np.float32),
        method="parzen",
        parzen=posterior,
    )
    assert clean.shape == components.shape
    assert np.isfinite(clean).all()
    assert np.max(np.abs(clean)) <= np.max(np.abs(posterior.posterior_mean)) + 1e-5
