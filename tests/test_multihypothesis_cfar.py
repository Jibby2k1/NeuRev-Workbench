from __future__ import annotations

import numpy as np

from neurobench.experiments.learnable_contrast.multihypothesis import (
    ExpertSpec,
    _fuse,
    build_kernel_bank,
    expert_matrix,
)
from neurobench.experiments.learnable_contrast.bounded_residual import model_class
from neurobench.experiments.learnable_contrast.diagnostic_video import _render_frame, _slug


def _contrast(image: np.ndarray, test: np.ndarray, reference: np.ndarray) -> float:
    return float((image * test).sum() - (image * reference).sum())


def test_factor_matrix_covers_four_observation_cases_at_three_scales() -> None:
    specs = expert_matrix()
    assert len(specs) == 24
    assert {spec.morphology for spec in specs} == {"center", "membrane"}
    assert {spec.radius_px for spec in specs} == {4, 6, 8}
    assert {spec.reference for spec in specs} == {"classic", "sector_censored"}
    assert {spec.temporal for spec in specs} == {"lme", "causal_coherence"}


def test_center_and_membrane_kernels_prefer_their_matching_geometry() -> None:
    specs = [
        ExpertSpec("center", 6, "classic", "lme"),
        ExpertSpec("membrane", 6, "classic", "lme"),
    ]
    bank = build_kernel_bank(specs, 35)
    yy, xx = np.mgrid[-17:18, -17:18]
    rr = np.sqrt(xx * xx + yy * yy)
    center = np.exp(-0.5 * (rr / 2.5) ** 2).astype(np.float32)
    membrane = ((rr >= 3.2) & (rr <= 6.0)).astype(np.float32)
    center_scores = [
        _contrast(center, bank["test"][i], bank["reference"][i]) for i in range(2)
    ]
    membrane_scores = [
        _contrast(membrane, bank["test"][i], bank["reference"][i]) for i in range(2)
    ]
    assert center_scores[0] > center_scores[1]
    assert membrane_scores[1] > membrane_scores[0]


def test_sector_censoring_rejects_one_bright_neighbor_sector() -> None:
    spec = ExpertSpec("center", 6, "sector_censored", "lme")
    bank = build_kernel_bank([spec], 35)
    yy, xx = np.mgrid[-17:18, -17:18]
    rr = np.sqrt(xx * xx + yy * yy)
    image = np.exp(-0.5 * (rr / 2.5) ** 2).astype(np.float32)
    image[(xx > 6) & (xx < 12) & (np.abs(yy) < 4)] += 5.0
    sector_means = np.asarray([(image * kernel).sum() for kernel in bank["sectors"][0]])
    classic_reference = float((image * bank["reference"][0]).sum())
    censored_reference = float(np.sort(sector_means)[:2].mean())
    assert censored_reference < classic_reference


def test_predeclared_fusions_preserve_spatial_shape() -> None:
    margins = np.asarray([[[1.0, -1.0]], [[0.0, 2.0]]], dtype=np.float32)
    assert np.allclose(_fuse(margins, "max_margin"), [[1.0, 2.0]])
    assert _fuse(margins, "logmeanexp_margin").shape == (1, 2)


def test_bounded_kernel_residual_starts_exactly_fixed_and_preserves_support() -> None:
    import torch

    specs = [ExpertSpec("center", 6, "classic", "lme")]
    bank = build_kernel_bank(specs, 35)
    Model = model_class()
    model = Model(bank, max_log_gain=0.05)
    initial = model.kernels()
    for name, value in initial.items():
        assert np.allclose(value.detach().numpy(), bank[name], atol=1e-7)
    with torch.no_grad():
        model.raw_test.fill_(100)
        model.raw_reference.fill_(-100)
    tuned = model.kernels()
    assert torch.equal(tuned["test"] == 0, model.base_test == 0)
    assert torch.equal(tuned["reference"] == 0, model.base_reference == 0)
    assert torch.allclose(tuned["test"].sum((-2, -1)), torch.ones(1))
    assert torch.allclose(tuned["reference"].sum((-2, -1)), torch.ones(1))


def test_diagnostic_frame_is_standalone_rgb_with_encoder_safe_dimensions() -> None:
    raw = np.arange(35, dtype=np.float32).reshape(5, 7)
    heat = np.zeros((5, 7, 3), dtype=np.uint8)
    strength = np.zeros((5, 7), dtype=np.float32)
    frame = _render_frame(
        raw,
        heat,
        strength,
        display_lo=0,
        display_hi=34,
        labels=[],
        peaks=[],
        matches=[],
        title="test",
        frame_ui=1,
    )
    assert frame.shape == (56, 8, 3)
    assert _slug("center_r8_sector_censored_causal_coherence") == "center-r8-censored-coherence"
