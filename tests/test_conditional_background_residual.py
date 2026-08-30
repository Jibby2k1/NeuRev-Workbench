from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from neurobench.experiments.neuron_identifiability.conditional_background_residual import (
    CausalAnnularContextConfig,
    CausalAnnularConvPredictor,
    ConditionalBackgroundError,
    ConditionalResidualOutput,
    ConvBackgroundConfig,
    DeterministicLowRankBackgroundPredictor,
    LowRankRidgeConfig,
    RobustResidualScaleConfig,
    RobustResidualStandardizer,
    build_causal_annular_context,
    conditional_background_loss,
    conditional_residual_diagnostics,
)


def _geometry(*, past_frames: int = 2) -> CausalAnnularContextConfig:
    return CausalAnnularContextConfig(
        past_frames=past_frames,
        target_radius_px=1.0,
        annulus_inner_radius_px=2.0,
        annulus_outer_radius_px=4.0,
    )


def _background_video(frames: int = 36) -> torch.Tensor:
    rows, columns = np.mgrid[-4:5, -4:5]
    spatial = 1.0 + 0.025 * rows + 0.04 * columns
    time = np.arange(frames, dtype=np.float64)
    amplitude = 20.0 + 1.5 * np.sin(time / 4.0) + 0.025 * time
    rng = np.random.default_rng(8231)
    noise = rng.normal(scale=0.025, size=(frames, 9, 9))
    video = amplitude[:, None, None] * spatial[None, :, :] + noise
    return torch.tensor(video, dtype=torch.float32).reshape(1, 1, frames, 9, 9)


def _manual_output(
    target_values: torch.Tensor,
    predicted_values: torch.Tensor,
) -> ConditionalResidualOutput:
    if target_values.shape != predicted_values.shape or target_values.ndim != 2:
        raise AssertionError("manual test values must share [frame,pixel] shape")
    frames, pixels = map(int, target_values.shape)
    if pixels != 5:
        raise AssertionError("manual fixture uses a five-pixel cross")
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[2, 2] = True
    mask[1, 2] = True
    mask[3, 2] = True
    mask[2, 1] = True
    mask[2, 3] = True
    target = torch.zeros((1, frames, 1, 5, 5), dtype=torch.float64)
    prediction = torch.zeros_like(target)
    target[..., mask] = target_values.reshape(1, frames, 1, pixels)
    prediction[..., mask] = predicted_values.reshape(1, frames, 1, pixels)
    residual = target - prediction
    return ConditionalResidualOutput(
        target=target,
        background_estimate=prediction,
        raw_residual=residual,
        target_mask=mask,
        target_frame_indices=tuple(range(2, frames + 2)),
        context_contract={"current_center_leakage_structurally_blocked": True},
        model_contract={"model": "manual_test_fixture"},
    )


def test_context_masks_current_center_and_uses_strictly_past_frames() -> None:
    config = _geometry()
    video = torch.arange(1, 1 + 1 * 1 * 7 * 9 * 9, dtype=torch.float32).reshape(
        1, 1, 7, 9, 9
    )
    batch = build_causal_annular_context(video, config)

    current_channel = batch.context[:, :, 0]
    lag_one_channel = batch.context[:, :, 1]
    assert torch.count_nonzero(current_channel[..., batch.target_mask]) == 0
    assert torch.equal(
        current_channel[0, 0, batch.annulus_mask],
        video[0, 0, config.past_frames, batch.annulus_mask],
    )
    assert torch.equal(
        lag_one_channel[0, 0, batch.past_support_mask],
        video[0, 0, config.past_frames - 1, batch.past_support_mask],
    )
    assert batch.target_frame_indices == (2, 3, 4, 5, 6)
    assert batch.summary()["current_center_leakage_structurally_blocked"] is True

    future_changed = video.clone()
    future_changed[:, :, -1] += 5000.0
    changed_batch = build_causal_annular_context(future_changed, config)
    # The final source frame is current only for the final target and is never
    # visible to any earlier target as future context.
    torch.testing.assert_close(
        batch.context[:, :-1], changed_batch.context[:, :-1], rtol=0.0, atol=0.0
    )


def test_current_center_change_cannot_change_conv_background_prediction() -> None:
    config = _geometry()
    model = CausalAnnularConvPredictor(
        context_config=config,
        model_config=ConvBackgroundConfig(hidden_channels=8, depth=2, norm_groups=4),
        initialization_seed=19,
    )
    video = _background_video(frames=8)
    mask = build_causal_annular_context(video, config).target_mask
    changed = video.clone()
    changed[0, 0, -1, mask] += 1000.0

    original = model(video)
    modified = model(changed)

    torch.testing.assert_close(
        original.background_estimate[:, -1],
        modified.background_estimate[:, -1],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        modified.raw_residual[:, -1] - original.raw_residual[:, -1],
        modified.target[:, -1] - original.target[:, -1],
    )
    assert model.contract()["current_center_available_to_model"] is False
    assert model.contract()["future_frames_available_to_model"] is False
    assert model.contract()["parameter_count"] < 100_000


def test_conv_loss_is_target_only_finite_and_differentiable() -> None:
    model = CausalAnnularConvPredictor(
        context_config=_geometry(),
        model_config=ConvBackgroundConfig(hidden_channels=8, depth=1, norm_groups=4),
        initialization_seed=23,
    )
    output = model(_background_video(frames=8))
    loss = conditional_background_loss(output, loss="smooth_l1", beta=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert output.summary()["pair_closure_max_abs"] == 0.0


def test_low_rank_source_off_baseline_is_deterministic_and_predictive() -> None:
    video = _background_video()
    first = DeterministicLowRankBackgroundPredictor(
        _geometry(), LowRankRidgeConfig(rank=5, ridge=1e-4)
    )
    second = DeterministicLowRankBackgroundPredictor(
        _geometry(), LowRankRidgeConfig(rank=5, ridge=1e-4)
    )
    first_summary = first.fit_source_off(video, source_off_id="synthetic_off")
    second.fit_source_off(video, source_off_id="synthetic_off")
    first_output = first.predict(video)
    second_output = second.predict(video)

    torch.testing.assert_close(
        first_output.background_estimate,
        second_output.background_estimate,
        rtol=0.0,
        atol=0.0,
    )
    observed = first_output.target[..., first_output.target_mask]
    predicted = first_output.background_estimate[..., first_output.target_mask]
    intercept = observed.mean(dim=(0, 1), keepdim=True)
    assert torch.mean((observed - predicted).square()) < torch.mean(
        (observed - intercept).square()
    )
    assert first_summary["fit_scope"] == "source_off_only"
    assert first_summary["fitted_rank"] <= 5
    json.dumps(first.export_state_json())


def test_source_off_robust_standardization_is_frozen_and_json_ready() -> None:
    frames = 11
    grid = torch.arange(frames * 5, dtype=torch.float64).reshape(frames, 5)
    predicted = 10.0 + 0.01 * grid
    residual = 0.2 * torch.sin(grid * 0.7) + 0.015 * grid
    residual[-1, -1] += 25.0  # robust fit should not be controlled by this outlier
    output = _manual_output(predicted + residual, predicted)
    standardizer = RobustResidualStandardizer(
        RobustResidualScaleConfig(granularity="per_channel", minimum_scale=1e-8)
    )
    contract = standardizer.fit_source_off(output, source_off_id="quiet_window_001")
    standardized = standardizer.transform(output)
    values = standardized.standardized_residual[..., output.target_mask]

    assert contract["fit_scope"] == "source_off_only"
    assert contract["source_on_refit_permitted"] is False
    assert abs(float(torch.median(values))) < 1e-10
    assert float(values.max()) > 10.0
    json.dumps(standardizer.contract(include_parameters=True))
    json.dumps(standardized.summary())


def test_standardizer_fails_closed_on_degenerate_noise_scale() -> None:
    predicted = torch.full((6, 5), 10.0, dtype=torch.float64)
    constant_residual = torch.full((6, 5), 0.25, dtype=torch.float64)
    output = _manual_output(predicted + constant_residual, predicted)
    standardizer = RobustResidualStandardizer()

    with pytest.raises(ConditionalBackgroundError, match="noise scale is degenerate"):
        standardizer.fit_source_off(output, source_off_id="constant_off")


def test_diagnostics_measure_background_suppression_signal_retention_and_collapse() -> None:
    frames = 12
    axis = torch.arange(frames, dtype=torch.float64)[:, None]
    spatial = torch.linspace(0.8, 1.2, 5, dtype=torch.float64)[None, :]
    prediction_off = (10.0 + 0.5 * torch.sin(axis / 2.0)) * spatial
    noise = 0.08 * torch.sin(axis * 1.3 + torch.arange(5, dtype=torch.float64))
    source_off = _manual_output(prediction_off + noise, prediction_off)
    signal_values = torch.zeros((frames, 5), dtype=torch.float64)
    signal_values[4:8, 1:4] = torch.tensor([0.5, 1.0, 0.5])
    prediction_on = prediction_off + 0.25 * signal_values
    source_on = _manual_output(
        prediction_off + noise + signal_values,
        prediction_on,
    )
    injected = torch.zeros_like(source_off.target)
    injected[..., source_off.target_mask] = signal_values.reshape(1, frames, 1, 5)

    diagnostics = conditional_residual_diagnostics(
        source_off,
        source_on=source_on,
        injected_signal_target=injected,
    )

    assert diagnostics["background"]["background_suppression_fraction_rms"] > 0.9
    assert diagnostics["signal"]["retention_projection_on_injected_signal"] == pytest.approx(
        0.75
    )
    assert diagnostics["signal"]["prediction_absorption_projection"] == pytest.approx(
        0.25
    )
    assert diagnostics["signal"]["pair_closure_max_abs"] == 0.0
    assert isinstance(diagnostics["collapse"]["screen_flag"], bool)
    assert diagnostics["interpretation_boundary"]["neuron_probability"] is False
    json.dumps(diagnostics)


@pytest.mark.parametrize(
    "bad_video, message",
    [
        (torch.zeros(2, 3, 4), "shape"),
        (torch.zeros(1, 1, 2, 9, 9), "more frames"),
        (torch.full((1, 1, 5, 9, 9), float("nan")), "finite"),
    ],
)
def test_context_fails_closed_on_invalid_video(bad_video: torch.Tensor, message: str) -> None:
    with pytest.raises(ConditionalBackgroundError, match=message):
        build_causal_annular_context(bad_video, _geometry())


def test_context_rejects_clipped_annulus_and_outputs_are_json_ready() -> None:
    clipped = CausalAnnularContextConfig(
        past_frames=1,
        target_radius_px=1.0,
        annulus_inner_radius_px=2.0,
        annulus_outer_radius_px=5.0,
    )
    video = torch.zeros((1, 1, 5, 9, 9), dtype=torch.float32)
    with pytest.raises(ConditionalBackgroundError, match="fit completely"):
        build_causal_annular_context(video, clipped)

    model = CausalAnnularConvPredictor(
        context_config=_geometry(),
        model_config=ConvBackgroundConfig(hidden_channels=4, depth=1, norm_groups=4),
    )
    output = model(_background_video(frames=7))
    json.dumps(model.contract())
    json.dumps(output.summary())
