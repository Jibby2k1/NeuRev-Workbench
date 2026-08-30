from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurobench.experiments.neuron_identifiability.spatiotemporal_jepa import (  # noqa: E402
    EncoderConfig,
    FrozenRandomEncoderBaseline,
    MaskConfig,
    MaskedPixelAutoencoder,
    SpatiotemporalJEPA,
    apply_patch_mask,
    count_parameters,
    make_contiguous_patch_mask,
    make_coverage_bounded_patch_mask,
    patch_mask_to_voxels,
    representation_diagnostics,
    tiny_smoke_configuration,
)


def _tiny_configs() -> tuple[EncoderConfig, MaskConfig]:
    return tiny_smoke_configuration()


def _tiny_video(batch_size: int = 1) -> torch.Tensor:
    values = torch.arange(1, batch_size * 4 * 4 * 4 + 1, dtype=torch.float32)
    return values.reshape(batch_size, 1, 4, 4, 4) / 64.0


def test_default_encoder_is_compact_and_mae_trainable_capacity_is_matched():
    jepa = SpatiotemporalJEPA(initialization_seed=17)
    mae = MaskedPixelAutoencoder(initialization_seed=17)

    encoder_parameters = count_parameters(jepa.context_encoder)
    jepa_trainable = count_parameters(jepa, trainable_only=True)
    mae_trainable = count_parameters(mae, trainable_only=True)

    assert 1_000_000 <= encoder_parameters <= 2_000_000
    assert abs(jepa_trainable - mae_trainable) / jepa_trainable < 0.01
    assert all(not parameter.requires_grad for parameter in jepa.target_encoder.parameters())


def test_contiguous_patch_mask_is_seeded_private_and_cuboidal():
    torch.manual_seed(991)
    state_before = torch.random.get_rng_state().clone()
    first = make_contiguous_patch_mask(
        batch_size=2,
        grid_shape=(4, 5, 6),
        block_shape=(2, 3, 2),
        seed=23,
    )
    state_after = torch.random.get_rng_state()
    repeated = make_contiguous_patch_mask(
        batch_size=2,
        grid_shape=(4, 5, 6),
        block_shape=(2, 3, 2),
        seed=23,
    )

    assert torch.equal(state_before, state_after)
    assert torch.equal(first, repeated)
    for sample_mask in first:
        locations = sample_mask.nonzero()
        extents = locations.max(dim=0).values - locations.min(dim=0).values + 1
        assert tuple(extents.tolist()) == (2, 3, 2)
        assert int(sample_mask.sum()) == 2 * 3 * 2


def test_primary_mask_is_four_nonoverlapping_cuboids_at_bounded_coverage():
    config = MaskConfig()
    first = make_coverage_bounded_patch_mask(
        batch_size=3,
        grid_shape=(8, 8, 8),
        block_shape=config.block_shape,
        blocks_per_sample=config.blocks_per_sample,
        coverage_range=config.coverage_range,
        minimum_temporal_tubelets=config.minimum_temporal_tubelets,
        seed=29,
    )
    repeated = make_coverage_bounded_patch_mask(
        batch_size=3,
        grid_shape=(8, 8, 8),
        block_shape=config.block_shape,
        blocks_per_sample=config.blocks_per_sample,
        coverage_range=config.coverage_range,
        minimum_temporal_tubelets=config.minimum_temporal_tubelets,
        seed=29,
    )
    block_volume = config.block_shape[0] * config.block_shape[1] * config.block_shape[2]
    expected_tokens = config.blocks_per_sample * block_volume
    counts = first.reshape(3, -1).sum(dim=1)
    coverage = counts.to(dtype=torch.float32) / (8 * 8 * 8)

    assert torch.equal(first, repeated)
    # Exact token count proves that the four written cuboids did not overlap.
    assert torch.equal(counts, torch.full_like(counts, expected_tokens))
    assert torch.all(coverage >= config.coverage_range[0])
    assert torch.all(coverage <= config.coverage_range[1])
    assert config.block_shape[0] >= 2


def test_primary_model_uses_coverage_bounded_mask_contract():
    primary_grid_tiny_encoder = EncoderConfig(
        patch_size=(4, 8, 8),
        embed_dim=8,
        latent_dim=4,
        depth=1,
        norm_groups=4,
    )
    model = SpatiotemporalJEPA(primary_grid_tiny_encoder, initialization_seed=37)
    video = torch.linspace(0.0, 1.0, 32 * 64 * 64).reshape(1, 1, 32, 64, 64)
    with torch.no_grad():
        output = model(video, mask_seed=41)

    assert output.patch_mask.shape == (1, 8, 8, 8)
    assert int(output.patch_mask.sum()) == 256
    assert float(output.patch_mask.to(dtype=torch.float32).mean()) == 0.5


def test_jepa_masks_before_context_encoder_and_stops_target_gradients():
    encoder_config, mask_config = _tiny_configs()
    model = SpatiotemporalJEPA(
        encoder_config,
        mask_config,
        initialization_seed=5,
    )
    video = _tiny_video()
    captured: dict[str, torch.Tensor] = {}

    def capture_context(_module, arguments):
        captured["context_input"] = arguments[0].detach().clone()

    def capture_target(_module, arguments):
        captured["target_input"] = arguments[0].detach().clone()

    context_hook = model.context_encoder.register_forward_pre_hook(capture_context)
    target_hook = model.target_encoder.register_forward_pre_hook(capture_target)
    output = model(video, mask_seed=31)
    context_hook.remove()
    target_hook.remove()

    expected_context = apply_patch_mask(
        video,
        output.patch_mask,
        patch_size=encoder_config.patch_size,
        mask_value=mask_config.mask_value,
    )
    assert torch.equal(captured["context_input"], expected_context)
    assert torch.equal(captured["target_input"], video)
    assert output.target.requires_grad is False
    assert torch.allclose(
        output.loss,
        output.prediction_error_map()[output.patch_mask].mean(),
    )

    output.loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.context_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_jepa_ema_update_and_unmasked_embedding_api_are_explicit():
    encoder_config, mask_config = _tiny_configs()
    model = SpatiotemporalJEPA(
        encoder_config,
        mask_config,
        initialization_seed=7,
        ema_decay=0.25,
    )
    context_parameter = next(model.context_encoder.parameters())
    target_parameter = next(model.target_encoder.parameters())
    initial_target = target_parameter.detach().clone()
    with torch.no_grad():
        context_parameter.add_(2.0)

    model.update_target_encoder()
    assert torch.allclose(target_parameter, initial_target + 1.5)

    context_embedding = model.encode_context(_tiny_video())
    target_embedding = model.encode_target(_tiny_video())
    assert context_embedding.shape == target_embedding.shape == (1, 4, 2, 2, 2)
    assert context_embedding.requires_grad is False
    assert target_embedding.requires_grad is False


def test_mask_sweep_scores_only_masked_tokens_and_requires_complete_coverage():
    encoder_config, mask_config = _tiny_configs()
    model = SpatiotemporalJEPA(
        encoder_config,
        mask_config,
        initialization_seed=11,
    )
    schedule = torch.eye(8, dtype=torch.bool).reshape(8, 1, 2, 2, 2)
    result = model.masked_prediction_error_sweep(_tiny_video(), schedule)

    assert result.prediction_error.shape == (1, 2, 2, 2)
    assert torch.isfinite(result.prediction_error).all()
    assert torch.equal(result.observation_counts, torch.ones_like(result.observation_counts))
    assert result.minimum_observations == 1
    with pytest.raises(ValueError, match="unobserved"):
        model.masked_prediction_error_sweep(_tiny_video(), schedule[:1])


def test_masked_pixel_baseline_reconstructs_only_declared_mask_and_matches_init():
    encoder_config, mask_config = _tiny_configs()
    jepa = SpatiotemporalJEPA(
        encoder_config,
        mask_config,
        initialization_seed=13,
    )
    mae = MaskedPixelAutoencoder(
        encoder_config,
        mask_config,
        initialization_seed=13,
    )
    for jepa_parameter, mae_parameter in zip(
        jepa.context_encoder.parameters(), mae.encoder.parameters()
    ):
        assert torch.equal(jepa_parameter, mae_parameter)
    for jepa_parameter, mae_parameter in zip(
        jepa.predictor.parameters(), mae.predictor.parameters()
    ):
        assert torch.equal(jepa_parameter, mae_parameter)

    video = _tiny_video()
    output = mae(video, mask_seed=43)
    assert output.reconstruction.shape == video.shape
    assert torch.isfinite(output.loss)
    voxel_mask = patch_mask_to_voxels(
        output.patch_mask, encoder_config.patch_size
    ).unsqueeze(1)
    expected_loss = (output.reconstruction - video).square().masked_select(voxel_mask).mean()
    assert torch.allclose(output.loss, expected_loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in mae.encoder.parameters())


def test_frozen_random_control_is_deterministic_and_cannot_be_unfrozen_by_train():
    encoder_config, _ = _tiny_configs()
    first = FrozenRandomEncoderBaseline(encoder_config, initialization_seed=19)
    repeated = FrozenRandomEncoderBaseline(encoder_config, initialization_seed=19)
    different = FrozenRandomEncoderBaseline(encoder_config, initialization_seed=20)
    video = _tiny_video()

    first.train(True)
    first_embedding = first(video)
    assert first.training is False
    assert all(not parameter.requires_grad for parameter in first.parameters())
    assert first_embedding.requires_grad is False
    assert torch.equal(first_embedding, repeated(video))
    assert not torch.equal(first_embedding, different(video))


def test_representation_diagnostics_distinguish_ranked_features_from_collapse():
    generator = torch.Generator(device="cpu").manual_seed(71)
    healthy = torch.randn((256, 8), generator=generator)
    healthy_diagnostics = representation_diagnostics(healthy)
    collapsed_diagnostics = representation_diagnostics(torch.zeros((256, 8)))
    dense_diagnostics = representation_diagnostics(
        healthy[:32].T.reshape(1, 8, 2, 4, 4)
    )

    assert healthy_diagnostics.collapsed is False
    assert healthy_diagnostics.effective_rank_fraction > 0.8
    assert 0.0 < healthy_diagnostics.first_principal_component_variance_fraction < 0.5
    assert collapsed_diagnostics.collapsed is True
    assert collapsed_diagnostics.effective_rank == 0.0
    assert dense_diagnostics.sample_count == 32
    assert dense_diagnostics.feature_dimension == 8
    assert healthy_diagnostics.as_dict()["finite_fraction"] == 1.0


def test_default_model_completes_a_cpu_tiny_training_smoke():
    _, tiny_mask_config = tiny_smoke_configuration()
    model = SpatiotemporalJEPA(
        mask_config=tiny_mask_config,
        initialization_seed=101,
    ).cpu()
    video = torch.linspace(0.0, 1.0, 8 * 16 * 16).reshape(1, 1, 8, 16, 16)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    target_before = next(model.target_encoder.parameters()).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    output = model(video, mask_seed=103)
    output.loss.backward()
    optimizer.step()
    model.update_target_encoder(decay=0.9)

    assert output.prediction.shape == output.target.shape == (1, 64, 2, 2, 2)
    assert torch.isfinite(output.loss)
    assert not torch.equal(next(model.target_encoder.parameters()), target_before)
