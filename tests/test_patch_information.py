import numpy as np
import pytest

from neurobench.algorithms.patch_information import (
    cauchy_schwarz_divergence_tensor,
    gaussian_information_kernel,
    information_fields_tensor,
    local_center_annulus_histograms_tensor,
    local_histogram_tensor,
    local_histogram_pyramid_tensor,
    quantization_boundaries,
)
from neurobench.experiments.hierarchical_parzen_ica.patch_information_config import (
    PatchInformationConfig,
)


CENTERS = (-2, -1, 0, 1, 2)


def test_information_kernel_is_symmetric_and_bandwidth_sensitive():
    narrow = gaussian_information_kernel(CENTERS, 0.5)
    wide = gaussian_information_kernel(CENTERS, 1.0)

    assert np.allclose(narrow, narrow.T)
    assert np.allclose(np.diag(narrow), 1)
    assert wide[0, 1] > narrow[0, 1]


def test_local_information_distinguishes_ordered_patch_from_quiet_density():
    torch = pytest.importorskip("torch")
    quiet_frames = torch.tensor(
        np.stack([np.tile(np.arange(-2, 3), (5, 1)) for _ in range(5)]),
        dtype=torch.float32,
    )
    quiet = local_histogram_tensor(
        quiet_frames, centers=CENTERS, patch_size_px=5
    ).mean(dim=0)
    ordered = torch.zeros((1, 5, 5), dtype=torch.float32)
    ordered[:, 2, 2] = 2
    histogram = local_histogram_tensor(
        ordered, centers=CENTERS, patch_size_px=5
    )
    fields = information_fields_tensor(
        histogram, ordered, quiet, centers=CENTERS, bandwidth=0.5
    )

    assert fields["renyi2_information_potential"].shape == (1, 5, 5)
    assert torch.isfinite(fields["cs_quiet_divergence"]).all()
    assert float(fields["cs_quiet_divergence"][0, 2, 2]) > 0
    assert torch.all((fields["local_correntropy"] >= 0) & (fields["local_correntropy"] <= 1))


def test_quantization_boundaries_are_midpoints():
    assert np.allclose(quantization_boundaries(CENTERS), [-1.5, -0.5, 0.5, 1.5])


def test_histogram_pyramid_and_center_annulus_are_normalized():
    torch = pytest.importorskip("torch")
    frames = torch.linspace(-2, 2, 2 * 9 * 11).reshape(2, 9, 11)
    pyramid = local_histogram_pyramid_tensor(
        frames, centers=CENTERS, patch_sizes_px=(3, 5, 7)
    )
    assert set(pyramid) == {3, 5, 7}
    for histogram in pyramid.values():
        assert torch.allclose(
            histogram.sum(dim=1), torch.ones_like(frames), atol=1e-6
        )
    center, annulus = local_center_annulus_histograms_tensor(
        frames, centers=CENTERS, center_patch_px=3, outer_patch_px=7
    )
    assert torch.allclose(center.sum(dim=1), torch.ones_like(frames), atol=1e-6)
    assert torch.allclose(annulus.sum(dim=1), torch.ones_like(frames), atol=1e-6)
    divergence = cauchy_schwarz_divergence_tensor(
        center, center, centers=CENTERS, bandwidth=1.0
    )
    assert torch.max(divergence).item() < 1e-5


def test_example_patch_information_config_has_frozen_counts():
    config = PatchInformationConfig.load(
        "examples/spon_ca_burst_patch_information_v1.example.json"
    )

    assert config.feature_count == 27
    assert config.fixed_lane_count == 216
    assert config.linear_config_count == 72
    assert config.inner_fit_count == 864
    assert config.outer_refit_count == 16
