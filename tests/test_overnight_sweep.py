from neurobench.dynamics.models import LatentTransformerPredictor, PixelConvGRUResidual, PixelConvLSTMResidual, TemporalCNNResidual, UNetConvGRUResidual
from neurobench.dynamics.overnight_sweep import build_specs


def test_latent_transformer_predictor_forward_shape():
    import torch

    model = LatentTransformerPredictor(latent_dim=8, model_dim=16, num_heads=2, num_layers=1, max_window_frames=8)
    x = torch.zeros(3, 4, 8)

    out = model(x)

    assert tuple(out.shape) == (3, 8)


def test_overnight_sweep_grid_sizes_are_deterministic():
    assert len(build_specs(profile="smoke", seeds=(7,), epochs=1, batch_size=4)) == 3
    assert len(build_specs(profile="overnight", seeds=(7, 13), epochs=50, batch_size=64)) == 408



def test_pixel_convgru_residual_forward_shape_and_range():
    import torch

    model = PixelConvGRUResidual(input_channels=1, hidden_channels=4, residual_scale=0.1)
    x = torch.zeros(2, 3, 1, 16, 16)

    out = model(x)

    assert tuple(out.shape) == (2, 1, 16, 16)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_upgrade_profile_adds_architecture_and_baseline_specs():
    specs = build_specs(profile="upgrade", seeds=(7,), epochs=2, batch_size=8, dataset_keys=("demo",))
    kinds = {spec.kind for spec in specs}

    assert "array_baseline" in kinds
    assert "linear_latent" in kinds
    assert "convgru_pixel" in kinds
    assert "residual_pixel" in kinds
    assert any(spec.params.get("loss_mode") == "motion_weighted_huber" for spec in specs)
    assert len(specs) > len(build_specs(profile="overnight", seeds=(7,), epochs=2, batch_size=8, dataset_keys=("demo",)))



def test_advanced_pixel_models_forward_shape_and_range():
    import torch

    x = torch.zeros(2, 4, 1, 16, 16)
    models = [
        PixelConvLSTMResidual(input_channels=1, hidden_channels=4, residual_scale=0.1),
        TemporalCNNResidual(input_channels=1, window_frames=4, hidden_channels=4, residual_scale=0.1, num_blocks=3),
        UNetConvGRUResidual(input_channels=1, base_channels=4, hidden_channels=8, residual_scale=0.1),
    ]

    for model in models:
        out = model(x)
        assert tuple(out.shape) == (2, 1, 16, 16)
        assert float(out.detach().min()) >= 0.0
        assert float(out.detach().max()) <= 1.0


def test_advanced_profile_contains_three_model_families():
    specs = build_specs(profile="advanced", seeds=(7,), epochs=2, batch_size=8, dataset_keys=("demo",))
    kinds = {spec.kind for spec in specs}

    assert {"unet_convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}.issubset(kinds)
    assert any(spec.params.get("loss_mode") == "residual_mse" for spec in specs)
    assert any(spec.params.get("loss_mode") == "motion_weighted_huber" for spec in specs)
    assert len(specs) == 36

def test_advanced_big_profile_includes_wide_and_deep_configs():
    specs = build_specs(profile="advanced_big", seeds=(7,), epochs=2, batch_size=8, dataset_keys=("demo",))
    kinds = {spec.kind for spec in specs}

    assert {"convgru_pixel", "unet_convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}.issubset(kinds)
    assert any(spec.params.get("hidden_channels") == 96 for spec in specs)
    assert any(spec.kind == "convgru_pixel" and spec.params.get("num_layers") == 2 for spec in specs)
    assert any(spec.kind == "convlstm_pixel" and spec.params.get("num_layers") == 2 for spec in specs)
    assert any(spec.kind == "temporal_cnn_pixel" and spec.params.get("num_layers") == 4 for spec in specs)
    assert any(spec.params.get("residual_scale") == 0.05 for spec in specs)
    assert len({spec.experiment_id for spec in specs}) == len(specs)
    assert len(specs) == 432

def test_advanced_overnight_profile_is_bounded_and_front_loaded():
    specs = build_specs(profile="advanced_overnight", seeds=(7, 13), epochs=2, batch_size=8, dataset_keys=("demo",))
    kinds = {spec.kind for spec in specs}

    assert [spec.kind for spec in specs[:6]] == [
        "array_baseline",
        "array_baseline",
        "array_baseline",
        "array_baseline",
        "linear_latent",
        "linear_latent",
    ]
    assert {"array_baseline", "linear_latent", "convgru_pixel", "unet_convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}.issubset(kinds)
    assert any(spec.params.get("hidden_channels") == 96 for spec in specs)
    assert any(spec.kind == "convgru_pixel" and spec.params.get("num_layers") == 2 for spec in specs)
    assert any(spec.kind == "convlstm_pixel" and spec.params.get("num_layers") == 2 for spec in specs)
    assert any(spec.kind == "temporal_cnn_pixel" and spec.params.get("num_layers") == 4 for spec in specs)
    assert any(spec.seed == 13 for spec in specs if spec.kind.endswith("pixel"))
    assert len({spec.experiment_id for spec in specs}) == len(specs)
    assert len(specs) == 138

def test_cropped32_restricted_profile_is_small_and_complete():
    specs = build_specs(profile="cropped32_restricted", seeds=(7, 13), epochs=2, batch_size=8, dataset_keys=("demo",))
    kinds = {spec.kind for spec in specs}

    assert {
        "array_baseline",
        "linear_latent",
        "residual_pixel",
        "latent_gru",
        "latent_transformer",
        "convgru_pixel",
        "unet_convgru_pixel",
        "convlstm_pixel",
        "temporal_cnn_pixel",
    }.issubset(kinds)
    assert any(spec.kind == "convgru_pixel" and spec.params.get("hidden_channels") == 8 for spec in specs)
    assert any(spec.kind == "temporal_cnn_pixel" and spec.params.get("num_layers") == 4 for spec in specs)
    assert any(spec.kind == "latent_transformer" and spec.params.get("model_dim") == 32 for spec in specs)
    assert any(spec.seed == 13 for spec in specs if spec.kind in {"residual_pixel", "latent_gru", "latent_transformer", "convgru_pixel"})
    assert len({spec.experiment_id for spec in specs}) == len(specs)
    assert len(specs) == 59
    assert len(specs) < len(build_specs(profile="advanced_big", seeds=(7,), epochs=2, batch_size=8, dataset_keys=("demo",)))



def test_cropped32_large_profile_expands_search_space():
    specs = build_specs(profile="cropped32_large", seeds=(7, 13), epochs=2, batch_size=8, dataset_keys=("demo",))
    kinds = {spec.kind for spec in specs}

    assert {
        "array_baseline",
        "linear_latent",
        "residual_pixel",
        "latent_gru",
        "latent_transformer",
        "convgru_pixel",
        "unet_convgru_pixel",
        "convlstm_pixel",
        "temporal_cnn_pixel",
    }.issubset(kinds)
    assert any(spec.kind == "temporal_cnn_pixel" and spec.params.get("num_layers") == 6 for spec in specs)
    assert any(spec.kind == "temporal_cnn_pixel" and spec.params.get("hidden_channels") == 128 for spec in specs)
    assert any(spec.kind == "convgru_pixel" and spec.params.get("num_layers") == 2 and spec.params.get("hidden_channels") == 96 for spec in specs)
    assert any(spec.kind == "residual_pixel" and spec.params.get("residual_scale") == 0.025 for spec in specs)
    assert any(spec.kind == "latent_transformer" and spec.params.get("model_dim") == 128 for spec in specs)
    assert any(spec.seed == 13 for spec in specs if spec.kind in {"residual_pixel", "latent_gru", "latent_transformer", "convgru_pixel"})
    assert len({spec.experiment_id for spec in specs}) == len(specs)
    assert len(specs) == 404
    assert len(specs) > len(build_specs(profile="cropped32_restricted", seeds=(7, 13), epochs=2, batch_size=8, dataset_keys=("demo",)))



def test_highres_temporal_cnn_scalable_profile_is_exhaustive_stage_a():
    specs = build_specs(profile="highres_temporal_cnn_scalable", seeds=(7, 13), epochs=2, batch_size=8, dataset_keys=("demo",))

    assert len(specs) == 240
    assert {spec.kind for spec in specs} == {"scalable_temporal_cnn_pixel"}
    assert len({spec.experiment_id for spec in specs}) == len(specs)
    assert len({spec.params["architecture_id"] for spec in specs}) == 15
    assert {spec.params["loss_mode"] for spec in specs} == {"residual_mse", "motion_weighted_huber"}
    assert {spec.params["residual_scale"] for spec in specs} == {0.05, 0.10}
    assert {spec.params["learning_rate"] for spec in specs} == {1e-4, 3e-5}
    assert {spec.seed for spec in specs} == {7, 13}
    assert all("architecture_spec" in spec.params for spec in specs)
