import numpy as np

from neurobench.algorithms.dependent_multiscale import fit_local_pca
from neurobench.integrations.local_pca import (
    ProviderLocalPCARecord,
    validate_provider_local_pca,
)


def _record(factor):
    return ProviderLocalPCARecord(
        source_movie_checksum="abc123",
        frame_range_zero_half_open=(0, 12),
        coordinate_convention="x=column,y=row;zero_based",
        patch_size_yx=(5, 5),
        stride_yx=(3, 3),
        centering_rule="per_pixel_temporal_mean",
        normalization_rule="none",
        rank_selection_rule="fixed_small_rank",
        overlap_window="hann_floor_0p1",
        blending_rule="normalized_overlap_add",
        software_provenance="provider-test-1",
        factors=(factor,),
        diagnostics={},
    )


def test_provider_requires_metadata_and_verified_closure():
    rng = np.random.default_rng(10)
    patch = rng.normal(size=(12, 5, 5)).astype(np.float32)
    factor = fit_local_pca(
        patch, patch_id="p0", view_id="scale_5", origin_yx=(0, 0), rank=5
    )
    unchecked = validate_provider_local_pca(_record(factor), movie_shape=(12, 5, 5))
    assert unchecked["status"] == "external_baseline_only"
    assert not unchecked["silent_inference_performed"]
    checked = validate_provider_local_pca(
        _record(factor), movie_shape=(12, 5, 5),
        reconstructed_patches=(patch,), closure_tolerance=1.0,
    )
    assert checked["status"] == "valid_initializer"


def test_provider_rejects_coordinate_ambiguity():
    rng = np.random.default_rng(11)
    patch = rng.normal(size=(12, 5, 5)).astype(np.float32)
    factor = fit_local_pca(
        patch, patch_id="p0", view_id="scale_5", origin_yx=(0, 0), rank=2
    )
    record = _record(factor)
    ambiguous = ProviderLocalPCARecord(
        **{**record.__dict__, "coordinate_convention": "unspecified"}
    )
    result = validate_provider_local_pca(
        ambiguous, movie_shape=(12, 5, 5), reconstructed_patches=(patch,),
    )
    assert result["status"] == "external_baseline_only"
    assert any("coordinate" in item for item in result["errors"])
