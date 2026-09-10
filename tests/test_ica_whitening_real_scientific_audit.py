import numpy as np
import pytest
import json

from neurobench.experiments.ica_whitening_evaluation.real_runner import (
    _observations_at_coordinates,
)
from neurobench.experiments.ica_whitening_evaluation.real_scientific_audit import (
    _component_evidence_movie, validate_all_scientific_audit_media,
)


@pytest.mark.parametrize("family", ["temporal", "spatial", "joint_spatiotemporal"])
def test_full_field_component_evidence_matches_exact_patch_observations(family):
    rng = np.random.default_rng(21)
    movie = rng.normal(size=(11, 7, 8)).astype(np.float32)
    specification = {
        "family": family, "spatial_width_px": 3,
        "temporal_width_frames": 3, "causality": "centered",
    }
    feature_count = {"temporal": 3, "spatial": 9,
                     "joint_spatiotemporal": 27}[family]
    demixing = rng.normal(size=(2, feature_count))
    mean = rng.normal(size=feature_count)
    model = {"demixing": demixing.tolist(),
             "internal_whitening": {"mean": mean.tolist()}}
    observed = _component_evidence_movie(movie, specification, model, quiet_frames=5)
    t, y, x = np.indices(movie.shape)
    patches = _observations_at_coordinates(
        movie, t.ravel(), y.ravel(), x.ravel(), specification
    )
    components = (demixing @ (patches - mean[:, None])).reshape(2, *movie.shape)
    expected = np.zeros_like(movie)
    for values in components:
        center = np.median(values[:5])
        scale = max(1.4826 * np.median(np.abs(values[:5] - center)), np.finfo(float).eps)
        expected = np.maximum(expected, np.abs((values - center) / scale))
    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-5)


def test_all_finalist_visual_validation_requires_exact_coverage(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "selected_fit_ids": ["f1", "f2"],
    }))
    with pytest.raises(ValueError, match="exact frozen finalist set"):
        validate_all_scientific_audit_media(
            tmp_path, visual_results={"f1": (True, "inspected")},
        )
