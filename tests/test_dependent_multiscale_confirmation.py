import numpy as np

from neurobench.algorithms.dependent_multiscale import ScaleViewSpec, build_scale_views
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_confirmation import (
    apply_confirmation_authority,
    build_confirmation_maps,
)
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_population import (
    population_preserving_movie,
)
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_synthetic import make_fixture


def test_confirmation_maps_are_bounded_and_label_free():
    fixture = make_fixture("population_burst_plus_private_activity", seed=7)
    specs = tuple(
        ScaleViewSpec(f"scale_{size}", size, "normalized_box_support", "none", {"nested": True})
        for size in (5, 7, 15)
    )
    views = build_scale_views(fixture.observation, specs, quiet_count=8)
    maps = build_confirmation_maps(fixture.observation, views, quiet_count=8)
    for values in (maps.coherence, maps.carrier, maps.motion):
        assert values.shape == fixture.observation.shape
        assert np.isfinite(values).all()
        assert float(values.min()) >= 0 and float(values.max()) <= 1
    assert not maps.diagnostics["labels_used"]


def test_combined_confirmation_closes_and_retains_unresolved_state():
    fixture = make_fixture("motion_edge_crossing_a_neuron", seed=13)
    specs = tuple(
        ScaleViewSpec(f"scale_{size}", size, "normalized_box_support", "none", {"nested": True})
        for size in (5, 7, 15)
    )
    views = build_scale_views(fixture.observation, specs, quiet_count=8)
    baseline = population_preserving_movie(
        fixture.observation, views, patch_px=15, stride_px=10,
        population_gain=0, residual_recapture_authority=0,
    ).decomposition
    population = population_preserving_movie(
        fixture.observation, views, patch_px=15, stride_px=10,
    ).decomposition
    result = apply_confirmation_authority(
        baseline, population, build_confirmation_maps(fixture.observation, views, quiet_count=8),
        lane_id="coherence_carrier_combined",
    )
    restored = result.background + result.structured_signal + result.structured_artifact + result.noise_candidate
    np.testing.assert_allclose(restored, fixture.observation, atol=3e-6)
    assert result.diagnostics["unresolved_fraction"] > 0
    assert not result.diagnostics["labels_used"]
    assert result.diagnostics["carrier_role"].endswith("not_output_target")
