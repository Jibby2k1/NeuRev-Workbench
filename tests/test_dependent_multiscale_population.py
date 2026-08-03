import numpy as np

from neurobench.algorithms.dependent_multiscale import ScaleViewSpec, build_scale_views
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_population import (
    population_preserving_movie,
    population_preserving_patch,
)
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_synthetic import make_fixture


def _views(observation):
    specs = tuple(
        ScaleViewSpec(f"scale_{size}", size, "normalized_box_support", "none", {"nested": True})
        for size in (5, 7, 15)
    )
    return build_scale_views(observation, specs, quiet_count=8)


def test_population_patch_closes_and_preserves_declared_semantics():
    fixture = make_fixture("population_burst_plus_private_activity", seed=7)
    result = population_preserving_patch(fixture.observation, _views(fixture.observation), patch_id="p")
    d = result.decomposition
    restored = d.background + d.structured_signal + d.structured_artifact + d.noise_candidate + d.closure_residual
    np.testing.assert_allclose(restored, fixture.observation, atol=2e-6)
    assert d.diagnostics["population_drive_preserved"]
    assert not d.diagnostics["individual_neural_independence_forced"]
    assert d.diagnostics["noise_status"] == "noise_candidate"


def test_patchwise_overlap_add_has_full_coverage_and_exact_closure():
    fixture = make_fixture("patch_boundary_source", seed=13)
    result = population_preserving_movie(
        fixture.observation, _views(fixture.observation), patch_px=15, stride_px=10
    )
    d = result.decomposition
    restored = d.background + d.structured_signal + d.structured_artifact + d.noise_candidate
    np.testing.assert_allclose(restored, fixture.observation, atol=3e-6)
    assert result.diagnostics["patch_count"] == 4
    assert all(item["denominator_minimum"] > 0 for item in result.diagnostics["overlap_add"].values())
