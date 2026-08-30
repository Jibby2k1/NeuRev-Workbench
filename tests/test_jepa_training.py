from __future__ import annotations

import numpy as np
import pytest
import torch

from neurobench.experiments.neuron_identifiability.jepa_training import (
    JEPATrainingError,
    TrainingSchedule,
    latent_temporal_change_score,
    model_latent_temporal_change_scorer,
    train_matched_representations,
)
from neurobench.experiments.neuron_identifiability.spatiotemporal_jepa import (
    tiny_smoke_configuration,
)


def _clips(seed: int, count: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, 8, 8, 8)).astype(np.float32)
    values[:, 3:6, 3:5, 3:5] += np.linspace(0, 2, 3)[None, :, None, None]
    return values


def test_tiny_matched_training_is_finite_and_capacity_matched() -> None:
    encoder, mask = tiny_smoke_configuration()
    result = train_matched_representations(
        _clips(1),
        _clips(2, 4),
        encoder_config=encoder,
        mask_config=mask,
        schedule=TrainingSchedule(
            steps=2,
            hard_step_cap=2,
            batch_size=2,
            log_interval=1,
            amp=False,
        ),
        seed=7,
        device="cpu",
    )
    assert len(result.metrics["training_curve"]) == 2
    assert result.metrics["capacity"]["matched_within_one_percent"] is True
    assert result.metrics["float16_used"] is False
    assert result.metrics["loss_accumulation_dtype"] == "float32"
    assert result.metrics["metric_dtype"] == "float32"
    assert result.metrics["observed_training_loss_dtypes"] == {
        "jepa": ["float32"],
        "mae": ["float32"],
    }
    assert result.metrics["validation"]["observed_loss_dtypes"] == {
        "jepa": ["float32"],
        "mae": ["float32"],
    }
    assert result.metrics["schedule"]["optimizer"] == "AdamW"
    assert result.metrics["schedule"]["learning_rate_schedule"] == "constant"
    assert result.optimizer_states["jepa"]["param_groups"][0]["betas"] == (0.9, 0.999)
    assert result.optimizer_states["jepa"]["param_groups"][0]["eps"] == pytest.approx(1e-8)
    assert np.isfinite(result.metrics["validation"]["jepa_masked_latent_loss"])
    assert result.metrics["validation"]["representation_diagnostics"]["jepa"]["finite_fraction"] == 1


def test_frozen_scorers_return_pixel_maps() -> None:
    encoder, mask = tiny_smoke_configuration()
    result = train_matched_representations(
        _clips(1),
        _clips(2, 4),
        encoder_config=encoder,
        mask_config=mask,
        schedule=TrainingSchedule(steps=1, hard_step_cap=1, batch_size=2, amp=False),
        seed=9,
        device="cpu",
    )
    movie = _clips(4, 1)[0]
    for model in (result.jepa, result.mae, result.random):
        score = model_latent_temporal_change_scorer(model, device="cpu")(movie)
        assert score.shape == (8, 8)
        assert np.isfinite(score).all()


def test_latent_score_rejects_invalid_shape_and_schedule_cap() -> None:
    with pytest.raises(JEPATrainingError, match="embeddings"):
        latent_temporal_change_score(torch.zeros(2, 3), output_shape_yx=(4, 4))
    with pytest.raises(JEPATrainingError, match="10k"):
        TrainingSchedule(steps=10_001, hard_step_cap=10_001)
    with pytest.raises(JEPATrainingError, match="only AdamW"):
        TrainingSchedule(learning_rate_schedule="cosine")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA bfloat16 integration requires compatible live hardware",
)
def test_cuda_bfloat16_autocast_keeps_observed_losses_float32() -> None:
    encoder, mask = tiny_smoke_configuration()
    result = train_matched_representations(
        _clips(51, 4),
        _clips(52, 4),
        encoder_config=encoder,
        mask_config=mask,
        schedule=TrainingSchedule(
            steps=1, hard_step_cap=1, batch_size=2, amp=True
        ),
        seed=53,
        device="cuda",
    )
    assert result.metrics["amp_dtype"] == "bfloat16"
    assert result.metrics["float16_used"] is False
    assert result.metrics["observed_training_loss_dtypes"] == {
        "jepa": ["float32"],
        "mae": ["float32"],
    }
