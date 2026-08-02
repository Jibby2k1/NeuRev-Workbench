import numpy as np

from neurobench.experiments.information_source_separation.confidence import decomposition_confidence_features


def test_confidence_features_are_finite_and_label_free() -> None:
    rng = np.random.default_rng(5)
    movie = rng.normal(size=(64, 4, 4)).astype(np.float32)
    def fit(values: np.ndarray, seed: int):
        matrix = values.reshape(len(values), -1).T
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        return {"spatial_maps": u[:, :3], "sources": vt[:3], "relative_observation_residual": 0.1, "converged": True, "iterations": 0, "execution_backend": "test", "diagnostics": {}}
    features, details = decomposition_confidence_features(movie, fit=fit, spatial_shape=(4, 4), seed=9, perturbations=1)
    assert all(np.isfinite(value) for value in features.values())
    assert "base" in details and "qualification" in details
