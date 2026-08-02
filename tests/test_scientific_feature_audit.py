import numpy as np
import pytest

from neurobench.algorithms.scientific_feature_audit import (
    causal_local_correlation_feature,
    fit_poisson_gaussian_noise,
    fit_zcut_templates_at_points,
    radial_zone_histograms_tensor,
    zcut_template_bank,
    zcut_response_maps,
)


def test_noise_fit_recovers_positive_mean_variance_slope():
    rng = np.random.default_rng(4)
    means = np.linspace(20, 400, 40 * 50).reshape(40, 50)
    video = np.stack([
        rng.normal(means, np.sqrt(3 + 0.4 * means)) for _ in range(80)
    ]).astype(np.float32)
    result = fit_poisson_gaussian_noise(video, intensity_bins=12)
    assert result["variance_slope_raw"] > 0.2
    assert result["variance_intercept_raw2"] >= 0
    assert len(result["bin_rows"]) >= 8


def test_radial_histograms_preserve_probability_at_boundaries():
    torch = pytest.importorskip("torch")
    frames = torch.linspace(-2, 2, 2 * 13 * 15).reshape(2, 13, 15)
    result = radial_zone_histograms_tensor(
        frames, centers=(-2, -1, 0, 1, 2), center_radius_px=2,
        shell_radius_px=4, outer_radius_px=6,
    )
    assert set(result) == {"center", "shell", "outer"}
    for values in result.values():
        assert values.shape == (2, 5, 13, 15)
        assert torch.allclose(values.sum(1), torch.ones_like(values[:, 0]), atol=1e-5)


def test_causal_correlation_does_not_use_future_frames():
    rng = np.random.default_rng(2)
    video = rng.normal(size=(20, 9, 11)).astype(np.float32)
    first = causal_local_correlation_feature(
        video, window_frames=7, lag_frames=2, spatial_sigma_px=1.2
    )
    changed = video.copy()
    changed[12:] += 20
    second = causal_local_correlation_feature(
        changed, window_frames=7, lag_frames=2, spatial_sigma_px=1.2
    )
    assert np.array_equal(first[:12], second[:12])


def test_zcut_bank_identifies_its_own_center_template():
    bank = zcut_template_bank(
        size_px=21, radii_px=(4,), z_offsets_fraction=(0, 0.8),
        membrane_thickness_px=1, psf_sigmas_px=(1,),
    )
    selected = next(row for row in bank if row["phenotype"] == "cytosol_center")
    image = np.zeros((41, 41), dtype=np.float32)
    image[10:31, 10:31] = selected["template"]
    result = fit_zcut_templates_at_points(image, [(20, 20)], bank)[0]
    assert result["best_phenotype"] == "cytosol_center"
    assert result["best_score"] > 0.99
    maps = zcut_response_maps(image, bank)
    assert maps["cytosol_center"][20, 20] > 0.99
