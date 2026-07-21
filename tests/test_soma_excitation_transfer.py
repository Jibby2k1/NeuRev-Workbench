import json

import numpy as np
import pytest
import torch

from neurobench.dynamics.models import TemporalCNNResidual
from neurobench.experiments.soma_excitation.transfer import (
    adaptive_max_pool_frame,
    evaluate_temporal_cnn_transfer,
    fit_robust_normalization_bounds,
    fixed_max_pool_frame,
    load_temporal_cnn_checkpoint,
    normalize_frame,
)


class RecordingProvider:
    def __init__(self, frames):
        self.frames = frames
        self.reads = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.frames[index]


def _checkpoint(tmp_path, *, architecture="temporal_cnn_pixel", provenance=True):
    model = TemporalCNNResidual(
        input_channels=1,
        window_frames=8,
        hidden_channels=2,
        num_blocks=1,
        residual_scale=0.1,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    payload = {
        "model_state": model.state_dict(),
        "architecture": architecture,
        "input_channels": 1,
        "window_frames": 8,
        "hidden_channels": 2,
        "num_layers": 1,
        "residual_scale": 0.1,
        "prediction_horizon_frames": 2,
    }
    if provenance:
        payload.update(
            {
                "model_id": "tiny_tcnn",
                "dataset_id": "synthetic",
                "normalization_mode": "quiet_robust_percentile",
                "grid_size": 4,
                "grid_pooling": "max_intensity",
            }
        )
    path = tmp_path / "concept_checkpoint.pt"
    torch.save(payload, path)
    return path, model


def test_checkpoint_round_trip_is_cpu_frozen_and_json_ready(tmp_path):
    path, _model = _checkpoint(tmp_path)

    loaded = load_temporal_cnn_checkpoint(path, cpu_threads=1)

    assert loaded.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    assert loaded.contract["architecture"] == "temporal_cnn_pixel"
    assert loaded.contract["horizon_frames"] == 2
    assert loaded.contract["inference_batch_size"] == 1
    assert loaded.contract["warnings"] == []
    json.dumps(loaded.contract)
    output = loaded.predict_one(np.zeros((8, 1, 4, 4), dtype=np.float32))
    assert output.shape == (1, 4, 4)
    with pytest.raises(ValueError, match="window must have shape"):
        loaded.predict_one(np.zeros((2, 8, 1, 4, 4), dtype=np.float32))


def test_checkpoint_rejects_wrong_architecture_and_reports_missing_provenance(tmp_path):
    wrong_path, _model = _checkpoint(tmp_path, architecture="convgru_pixel")
    with pytest.raises(ValueError, match="temporal_cnn_pixel"):
        load_temporal_cnn_checkpoint(wrong_path)

    valid_path, _model = _checkpoint(tmp_path, provenance=False)
    loaded = load_temporal_cnn_checkpoint(valid_path)
    warnings = " ".join(loaded.contract["warnings"])
    assert "training-dataset provenance" in warnings
    assert "normalization provenance" in warnings
    assert "grid geometry/pooling provenance" in warnings


def test_transfer_uses_exact_window_horizon_indices_and_matches_eager_model(tmp_path):
    path, model = _checkpoint(tmp_path)
    loaded = load_temporal_cnn_checkpoint(path)
    frames = np.stack([np.full((4, 4), index / 20.0, dtype=np.float32) for index in range(20)])
    provider = RecordingProvider(frames)

    result = evaluate_temporal_cnn_transfer(loaded, provider, [10, 11])

    assert set(provider.reads) == set(range(1, 12))
    assert result["evaluation"]["first_input_index"] == 1
    assert result["evaluation"]["last_input_index_for_first_target"] == 8
    assert result["evaluation"]["provider_frame_read_count"] == len(set(provider.reads))
    eager_window = torch.from_numpy(frames[1:9, None]).unsqueeze(0)
    with torch.inference_mode():
        eager = model(eager_window)[0].numpy()
    assert np.allclose(loaded.predict_one(frames[1:9]), eager)
    assert result["metrics"]["prediction_mse"] == pytest.approx(0.01)
    assert result["metrics"]["persistence_mse"] == pytest.approx(0.01)
    decay = result["metrics"]["exponential_decay_sensitivity_control"]
    assert decay["label"] == "sensitivity_control_not_fluorescence_model"
    assert decay["factor"] == pytest.approx(np.exp(-2 * 10 / 50))


def test_normalization_bounds_are_frozen_without_event_leakage():
    frames = np.zeros((4, 4, 4), dtype=np.float32)
    frames[1] = 1.0
    frames[2:] = 100.0
    first = fit_robust_normalization_bounds(
        frames, [0, 1], lower_percentile=0.0, upper_percentile=100.0
    )
    frames[2:] = 10_000.0
    second = fit_robust_normalization_bounds(
        frames, [0, 1], lower_percentile=0.0, upper_percentile=100.0
    )

    assert first == second
    assert first.frozen is True
    assert np.array_equal(normalize_frame(np.array([[0.0, 0.5, 1.0, 2.0]]), first), [[0, 0.5, 1, 1]])


def test_two_pooling_arms_have_expected_shapes_and_fixed_footprint():
    frame = np.arange(340 * 573, dtype=np.float32).reshape(340, 573)

    adaptive = adaptive_max_pool_frame(frame)
    fixed = fixed_max_pool_frame(frame, pool_size=4)

    assert adaptive.shape == (128, 128)
    assert fixed.shape == (85, 143)
    assert fixed[0, 0] == frame[:4, :4].max()
    assert fixed[-1, -1] == frame[336:340, 568:572].max()


def test_transfer_reports_high_change_and_core_ring_mask_metrics(tmp_path):
    path, _model = _checkpoint(tmp_path)
    loaded = load_temporal_cnn_checkpoint(path)
    frames = np.zeros((12, 4, 4), dtype=np.float32)
    frames[10, 0, 0] = 0.4
    frames[10, 0, 1] = 0.2
    core = np.zeros((4, 4), dtype=bool)
    ring = np.zeros((4, 4), dtype=bool)
    core[0, 0] = True
    ring[0, 1] = True

    result = evaluate_temporal_cnn_transfer(
        loaded,
        frames,
        [10],
        high_change_threshold=0.1,
        core_mask=core,
        ring_mask=ring,
    )

    metrics = result["metrics"]
    assert metrics["high_change_cell_count"] == 2
    assert metrics["high_change"]["prediction_mse"] == pytest.approx((0.16 + 0.04) / 2)
    assert metrics["masked"]["core"]["cell_count"] == 1
    assert metrics["masked"]["core"]["prediction_mse"] == pytest.approx(0.16)
    assert metrics["masked"]["ring"]["prediction_mse"] == pytest.approx(0.04)
