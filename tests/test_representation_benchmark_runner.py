import numpy as np

from neurobench.experiments.representation_benchmark.runner import (
    _component_evidence,
    _pca_full,
    _reconstruct,
)


def test_cpu_pca_and_component_evidence_are_finite() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(1400, 20)).astype(np.float32)
    scores, basis, singular, explained = _pca_full(
        values, 4, device="cpu", chunk_pixels=256
    )
    restored = _reconstruct(scores, basis, device="cpu", chunk_pixels=256)
    evidence, contract = _component_evidence(
        scores, basis, quiet_frames=5, events=[(8, 12), (14, 18)],
        device="cpu", chunk_pixels=256,
    )
    assert scores.shape == (1400, 4)
    assert basis.shape == (4, 20)
    assert restored.shape == values.shape
    assert evidence.shape == values.shape
    assert np.isfinite(evidence).all()
    assert np.all(np.diff(singular) <= 0)
    assert explained.sum() <= 1.000001
    assert "positive_spatial_z" in contract["aggregation"]
