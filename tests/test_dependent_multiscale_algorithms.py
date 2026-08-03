import numpy as np

from neurobench.algorithms.dependent_multiscale import (
    ScaleViewSpec,
    build_scale_views,
    decompose_patch_baseline,
    fit_local_pca,
    orthogonal_shared_private,
    overlap_add,
    reconstruct_local_factorization,
)


def _specs():
    return tuple(
        ScaleViewSpec(
            f"scale_{size}", size, "quiet_normalized_local_support",
            "quiet_robust", {"nested": True},
        )
        for size in (5, 7, 15)
    )


def test_scale_views_are_aligned_deterministic_and_quiet_only():
    rng = np.random.default_rng(4)
    movie = rng.normal(size=(12, 19, 21)).astype(np.float32)
    movie[8:] += 20
    original = movie.copy()
    first = build_scale_views(movie, _specs(), quiet_count=8)
    second = build_scale_views(movie, _specs(), quiet_count=8)
    assert tuple(first) == ("scale_5", "scale_7", "scale_15")
    for key in first:
        assert first[key].shape == movie.shape
        np.testing.assert_array_equal(first[key], second[key])
    # Event frames cannot alter quiet normalization.
    changed = movie.copy()
    changed[8:] += 1000
    quiet_again = build_scale_views(changed, _specs(), quiet_count=8)
    for key in first:
        np.testing.assert_array_equal(first[key][:8], quiet_again[key][:8])
    np.testing.assert_array_equal(movie, original)


def test_local_pca_roundtrip_and_orthogonal_shared_private():
    rng = np.random.default_rng(5)
    spatial = rng.normal(size=(25, 3))
    temporal = rng.normal(size=(3, 16))
    patch = (spatial @ temporal).T.reshape(16, 5, 5)
    fit = fit_local_pca(
        patch, patch_id="p0", view_id="scale_5", origin_yx=(0, 0), rank=3
    )
    restored = reconstruct_local_factorization(fit)
    assert np.linalg.norm(restored - patch) / np.linalg.norm(patch) < 1e-5
    coordinates = {
        "scale_5": temporal.T,
        "scale_7": temporal.T + 0.01 * rng.normal(size=temporal.T.shape),
        "scale_15": rng.normal(size=temporal.T.shape),
    }
    shared, private, diagnostics = orthogonal_shared_private(coordinates, shared_rank=2)
    for key in coordinates:
        centered = coordinates[key] - coordinates[key].mean(axis=0)
        np.testing.assert_allclose(shared[key] + private[key], centered, atol=1e-10)
        np.testing.assert_allclose(shared[key].T @ private[key], 0, atol=1e-10)
    assert diagnostics["interpretation"].startswith("orthogonality")


def test_baseline_and_overlap_add_close_exactly():
    rng = np.random.default_rng(6)
    movie = rng.normal(size=(12, 17, 17)).astype(np.float32)
    views = build_scale_views(movie, _specs(), quiet_count=6)
    result = decompose_patch_baseline(movie, views, patch_id="whole")
    restored = (
        result.background + result.structured_signal
        + result.structured_artifact + result.noise_candidate
        + result.closure_residual
    )
    np.testing.assert_allclose(restored, movie, atol=2e-6)
    combined, diagnostics = overlap_add(
        [((0, 0), movie[:, :, :12]), ((0, 5), movie[:, :, 5:])],
        movie.shape,
    )
    np.testing.assert_allclose(combined, movie, atol=2e-6)
    assert diagnostics["denominator_minimum"] > 0
