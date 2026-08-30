from __future__ import annotations

from dataclasses import asdict
import hashlib

import pytest

torch = pytest.importorskip("torch")

from neurobench.experiments.neuron_identifiability.jepa_background_residual import (  # noqa: E402
    DecoderTrainingConfig,
    FrozenJEPALatentProvider,
    FrozenLatentPixelBackgroundModel,
    FrozenRandomLatentProvider,
    JEPABackgroundResidualError,
    PRODUCTION_DECODER_PARAMETERS,
    background_suppression_diagnostics,
    blind_halo_manifest,
    load_frozen_jepa_checkpoint,
    load_frozen_random_checkpoint,
    module_state_sha256,
    paired_residual_diagnostics,
    patch_seam_diagnostics,
    predict_full_background,
    production_contract_preflight,
    signed_residual,
    target_tube_invariance_preflight,
    train_pixel_decoder,
)
from neurobench.experiments.neuron_identifiability.spatiotemporal_jepa import (  # noqa: E402
    EncoderConfig,
    FrozenRandomEncoderBaseline,
    MaskConfig,
    SpatiotemporalJEPA,
)


def _tiny_configs() -> tuple[EncoderConfig, MaskConfig]:
    return (
        EncoderConfig(
            patch_size=(2, 2, 2),
            embed_dim=8,
            latent_dim=4,
            depth=1,
            norm_groups=4,
        ),
        MaskConfig(
            block_shape=(1, 1, 1),
            blocks_per_sample=1,
            coverage_range=None,
            minimum_temporal_tubelets=1,
        ),
    )


def _tiny_video(batch_size: int = 1, spatial_tokens: int = 3) -> torch.Tensor:
    side = spatial_tokens * 2
    values = torch.arange(
        batch_size * 1 * 4 * side * side, dtype=torch.float32
    ).reshape(batch_size, 1, 4, side, side)
    return values / max(float(values.numel()), 1.0)


def _tiny_model(seed: int = 7) -> FrozenLatentPixelBackgroundModel:
    encoder, mask = _tiny_configs()
    jepa = SpatiotemporalJEPA(
        encoder,
        mask,
        initialization_seed=seed,
        normalize_latents=True,
    )
    return FrozenLatentPixelBackgroundModel(
        FrozenJEPALatentProvider(jepa),
        decoder_initialization_seed=seed + 1,
    )


def _checkpoint(tmp_path, *, missing_jepa_key: bool = False):
    encoder, mask = _tiny_configs()
    jepa = SpatiotemporalJEPA(
        encoder,
        mask,
        initialization_seed=11,
        normalize_latents=True,
    )
    random = FrozenRandomEncoderBaseline(encoder, initialization_seed=11)
    jepa_state = dict(jepa.state_dict())
    if missing_jepa_key:
        jepa_state.pop(next(iter(jepa_state)))
    payload = {
        "schema_version": 1,
        "mode": "screen",
        "scientific_checkpoint": False,
        "training_seed": 1001,
        "encoder_config": asdict(encoder),
        "mask_config": asdict(mask),
        "schedule": {"steps": 500, "ema_decay": 0.996},
        "jepa_state_dict": jepa_state,
        "random_state_dict": random.state_dict(),
    }
    path = tmp_path / ("bad.pt" if missing_jepa_key else "checkpoint.pt")
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, jepa, random


def test_checkpoint_load_is_hash_verified_strict_and_frozen(tmp_path) -> None:
    path, digest, jepa, random = _checkpoint(tmp_path)
    loaded_jepa = load_frozen_jepa_checkpoint(path, expected_sha256=digest)
    loaded_random = load_frozen_random_checkpoint(path, expected_sha256=digest)

    assert loaded_jepa.hash_verified is True
    assert loaded_jepa.training_seed == 1001
    assert loaded_jepa.mode == "screen"
    assert loaded_jepa.scientific_checkpoint is False
    assert loaded_jepa.training_schedule["steps"] == 500
    assert loaded_jepa.provider.model.normalize_latents is True
    assert module_state_sha256(loaded_jepa.provider.model) == module_state_sha256(jepa)
    assert module_state_sha256(loaded_random.provider.model) == module_state_sha256(random)
    assert all(not parameter.requires_grad for parameter in loaded_jepa.provider.parameters())
    assert all(not parameter.requires_grad for parameter in loaded_random.provider.parameters())
    assert loaded_jepa.manifest()["checkpoint_sha256"] == digest

    with pytest.raises(JEPABackgroundResidualError, match="SHA-256 mismatch"):
        load_frozen_jepa_checkpoint(path, expected_sha256="0" * 64)
    bad_path, bad_digest, _, _ = _checkpoint(tmp_path, missing_jepa_key=True)
    with pytest.raises(JEPABackgroundResidualError, match="strict JEPA"):
        load_frozen_jepa_checkpoint(bad_path, expected_sha256=bad_digest)


def test_decoder_parameter_count_matches_registered_production_contract() -> None:
    model = FrozenLatentPixelBackgroundModel(
        FrozenJEPALatentProvider(SpatiotemporalJEPA(initialization_seed=3)),
        decoder_initialization_seed=5,
    )
    assert model.decoder_parameter_count == PRODUCTION_DECODER_PARAMETERS == 16_385
    assert model.trainable_parameter_count == 16_385
    assert all(not parameter.requires_grad for parameter in model.provider.parameters())

    production_video = torch.zeros((1, 1, 32, 64, 64), dtype=torch.float32)
    preflight = production_contract_preflight(model, production_video)
    assert preflight["passed"] is True
    assert preflight["manifest"]["target_count"] == 64


def test_full_time_three_by_three_halo_manifest_and_exact_voxel_coverage() -> None:
    video = _tiny_video(spatial_tokens=3)
    model = _tiny_model()
    manifest = blind_halo_manifest(video, model.patch_size)

    assert manifest.original_grid_tyx == (2, 3, 3)
    assert manifest.padded_grid_tyx == (2, 5, 5)
    assert manifest.blind_halo_tokens_tyx == (2, 3, 3)
    assert manifest.target_count == 9
    assert {target.full_time_masked_tokens for target in manifest.targets} == {18}
    assert manifest.as_dict()["spatial_padding"]["mode"] == "reflect"

    output = predict_full_background(model, video, target_batch_size=4)
    assert output.background.shape == video.shape
    assert torch.isfinite(output.background).all()
    assert torch.equal(output.coverage_counts, torch.ones_like(output.coverage_counts))
    assert output.minimum_coverage == output.maximum_coverage == 1
    assert output.summary()["exactly_once_coverage"] is True


@pytest.mark.parametrize("target_yx", [(0, 0), (1, 1), (3, 3), (7, 7)])
def test_adversarial_full_halo_change_cannot_change_target_prediction(target_yx) -> None:
    model = _tiny_model()
    video = _tiny_video(spatial_tokens=8)
    result = target_tube_invariance_preflight(
        model,
        video,
        target_yx=target_yx,
        adversarial_delta=1234.0,
    )

    assert result["adversarial_scope"].startswith("full_time")
    assert result["masked_provider_inputs_exactly_identical"] is True
    assert result["predicted_target_max_abs_change"] == 0.0
    assert result["passed"] is True


def test_training_updates_only_decoder_and_keeps_float32_loss() -> None:
    model = _tiny_model(seed=17)
    clips = torch.cat([_tiny_video(spatial_tokens=3) + offset for offset in (0.0, 0.1, 0.2)])
    provider_before = module_state_sha256(model.provider)
    decoder_before = module_state_sha256(model.pixel_decoder)
    result = train_pixel_decoder(
        model,
        clips,
        config=DecoderTrainingConfig(
            steps=3,
            hard_step_cap=3,
            batch_size=2,
            log_interval=1,
            amp=False,
        ),
        seed=19,
        device="cpu",
    )

    assert module_state_sha256(model.provider) == provider_before
    assert module_state_sha256(model.pixel_decoder) != decoder_before
    assert result.metrics["provider_unchanged"] is True
    assert result.metrics["decoder_changed"] is True
    assert result.metrics["trainable_parameters"] == model.decoder_parameter_count
    assert result.metrics["loss"] == {
        "name": "SmoothL1Loss",
        "beta": 1.0,
        "reduction": "mean",
        "accumulation_dtype": "float32",
    }
    assert result.metrics["amp"]["observed_loss_dtypes"] == ["float32"]


def test_decoder_training_is_deterministic_on_cpu() -> None:
    clips = torch.cat([_tiny_video(spatial_tokens=3) + offset for offset in (0.0, 0.2)])
    config = DecoderTrainingConfig(
        steps=2,
        hard_step_cap=2,
        batch_size=2,
        log_interval=1,
        amp=False,
    )
    first = _tiny_model(seed=23)
    repeated = _tiny_model(seed=23)
    first_result = train_pixel_decoder(first, clips, config=config, seed=29)
    repeated_result = train_pixel_decoder(repeated, clips, config=config, seed=29)

    assert module_state_sha256(first.pixel_decoder) == module_state_sha256(
        repeated.pixel_decoder
    )
    assert first_result.metrics["training_curve"] == repeated_result.metrics[
        "training_curve"
    ]
    assert first_result.metrics["sampling_schedule_sha256"] == repeated_result.metrics[
        "sampling_schedule_sha256"
    ]
    assert first_result.metrics["sampling_schedule_hashed_fields"] == [
        "zero_based_step_index",
        "target_token_flat_index",
        "ordered_training_clip_indices",
    ]

    different_seed = _tiny_model(seed=23)
    different_result = train_pixel_decoder(
        different_seed, clips, config=config, seed=30
    )
    assert different_result.metrics["sampling_schedule_sha256"] != first_result.metrics[
        "sampling_schedule_sha256"
    ]


def test_cpu_bfloat16_autocast_keeps_loss_float32() -> None:
    result = train_pixel_decoder(
        _tiny_model(seed=41),
        _tiny_video(batch_size=2, spatial_tokens=3),
        DecoderTrainingConfig(
            steps=1,
            hard_step_cap=1,
            batch_size=1,
            amp=True,
        ),
        seed=43,
        device="cpu",
    )
    assert result.metrics["amp"]["dtype"] == "bfloat16"
    assert result.metrics["amp"]["observed_prediction_dtypes"] == ["bfloat16"]
    assert result.metrics["amp"]["observed_loss_dtypes"] == ["float32"]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA bfloat16 integration requires compatible live hardware",
)
def test_cuda_bfloat16_autocast_keeps_loss_float32() -> None:
    result = train_pixel_decoder(
        _tiny_model(seed=47),
        _tiny_video(batch_size=2, spatial_tokens=3),
        DecoderTrainingConfig(
            steps=1,
            hard_step_cap=1,
            batch_size=1,
            amp=True,
        ),
        seed=53,
        device="cuda",
    )
    assert result.metrics["amp"]["dtype"] == "bfloat16"
    assert result.metrics["amp"]["observed_prediction_dtypes"] == ["bfloat16"]
    assert result.metrics["amp"]["observed_loss_dtypes"] == ["float32"]


def test_paired_metrics_close_and_separate_retention_from_absorption() -> None:
    time = torch.linspace(-1.0, 1.0, 4).reshape(1, 1, 4, 1, 1)
    spatial = torch.arange(36, dtype=torch.float32).reshape(1, 1, 1, 6, 6) / 36.0
    source_off = time + spatial
    injected = torch.zeros_like(source_off)
    injected[:, :, 1:3, 2:4, 2:4] = 2.0
    source_on = source_off + injected
    predicted_off = 0.75 * source_off
    predicted_on = predicted_off + 0.25 * injected

    diagnostics = paired_residual_diagnostics(
        source_off,
        source_on,
        predicted_off,
        predicted_on,
        injected_signal=injected,
        patch_size=(2, 2, 2),
    )
    signal = diagnostics["signal"]
    assert signal["retained_projection_alpha"] == pytest.approx(0.75)
    assert signal["prediction_absorption_alpha"] == pytest.approx(0.25)
    assert signal["projection_sum"] == pytest.approx(1.0)
    assert signal["total_signal_error_ratio"] == pytest.approx(0.25)
    assert signal["orthogonal_distortion_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["closure"]["pair_closure_max_abs"] <= 2 * torch.finfo(
        torch.float32
    ).eps
    assert diagnostics["closure"][
        "observed_delta_minus_injected_signal_max_abs"
    ] <= 2 * torch.finfo(torch.float32).eps
    assert diagnostics["background"]["background_suppression_fraction_rms"] == pytest.approx(
        0.75
    )
    assert diagnostics["seams"]["predicted_background_off"][
        "boundary_difference_count"
    ] > 0

    residual = signed_residual(source_on, predicted_on)
    assert torch.equal(residual, source_on - predicted_on)
    assert background_suppression_diagnostics(source_off, predicted_off)[
        "background_suppression_fraction_rms"
    ] == pytest.approx(0.75)
    assert patch_seam_diagnostics(predicted_off, (2, 2, 2))[
        "boundary_to_within_jump_ratio"
    ] >= 0.0


def test_paired_metrics_separate_gain_error_from_orthogonal_distortion() -> None:
    source_off = torch.randn(
        (1, 1, 4, 4, 4), generator=torch.Generator().manual_seed(701)
    )
    signal = torch.zeros_like(source_off)
    signal[..., 0, 0] = 2.0
    orthogonal = torch.zeros_like(source_off)
    orthogonal[..., 1, 1] = 0.5
    residual_delta = 0.5 * signal + orthogonal
    source_on = source_off + signal
    predicted_off = 0.5 * source_off
    predicted_on = predicted_off + signal - residual_delta

    metrics = paired_residual_diagnostics(
        source_off,
        source_on,
        predicted_off,
        predicted_on,
        injected_signal=signal,
    )["signal"]

    assert metrics["retained_projection_alpha"] == pytest.approx(0.5)
    assert metrics["orthogonal_distortion_ratio"] == pytest.approx(0.25)
    assert metrics["total_signal_error_ratio"] == pytest.approx(5**0.5 / 4.0)


def test_random_provider_uses_identical_decoder_geometry() -> None:
    encoder, mask = _tiny_configs()
    random = FrozenRandomEncoderBaseline(encoder, initialization_seed=31)
    model = FrozenLatentPixelBackgroundModel(
        FrozenRandomLatentProvider(random, mask_value=mask.mask_value),
        decoder_initialization_seed=37,
    )
    output = predict_full_background(model, _tiny_video(spatial_tokens=3))

    expected = encoder.latent_dim * encoder.in_channels
    expected *= encoder.patch_size[0] * encoder.patch_size[1] * encoder.patch_size[2]
    expected += encoder.in_channels
    assert model.decoder_parameter_count == expected
    assert output.summary()["exactly_once_coverage"] is True
