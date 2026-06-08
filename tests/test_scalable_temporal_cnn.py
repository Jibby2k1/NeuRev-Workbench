import pytest

from neurobench.dynamics.models import ScalableTemporalCNNResidual
from neurobench.dynamics.scalable import architecture_catalog, architecture_summary


def test_scalable_temporal_cnn_stack_forward_shape_and_range():
    import torch

    spec = {
        "architecture_id": "test_stack",
        "topology": "stack",
        "stack_channels": [8, 8],
        "stack_blocks": [1, 1],
        "normalization": "group",
        "activation": "silu",
        "dilations": [1, 2],
    }
    model = ScalableTemporalCNNResidual(input_channels=1, window_frames=4, architecture_spec=spec, residual_scale=0.1)
    x = torch.zeros(2, 4, 1, 32, 32)

    out, residual = model(x, return_residual=True)

    assert tuple(out.shape) == (2, 1, 32, 32)
    assert tuple(residual.shape) == (2, 1, 32, 32)
    assert float(out.detach().min()) >= 0.0
    assert float(out.detach().max()) <= 1.0


def test_scalable_temporal_cnn_encoder_decoder_forward_shape_and_range():
    import torch

    spec = {
        "architecture_id": "test_ed",
        "topology": "encoder_decoder",
        "encoder_channels": [8, 16, 32],
        "encoder_blocks": [1, 1, 1],
        "decoder_channels": [16, 8],
        "decoder_blocks": [1, 1],
        "bottleneck_channels": 32,
        "bottleneck_blocks": 1,
        "skip_connections": True,
        "normalization": "group",
        "activation": "relu",
    }
    model = ScalableTemporalCNNResidual(input_channels=1, window_frames=4, architecture_spec=spec, residual_scale=0.1)
    x = torch.zeros(2, 4, 1, 32, 32)

    out = model(x)

    assert tuple(out.shape) == (2, 1, 32, 32)
    assert float(out.detach().min()) >= 0.0
    assert float(out.detach().max()) <= 1.0


def test_scalable_temporal_cnn_rejects_invalid_specs():
    with pytest.raises(ValueError):
        ScalableTemporalCNNResidual(architecture_spec={"topology": "unknown"})
    with pytest.raises(ValueError):
        ScalableTemporalCNNResidual(architecture_spec={"kernel_size": 4})


def test_architecture_catalog_has_stage_a_components():
    catalog = architecture_catalog()
    summaries = [architecture_summary(spec, input_channels=1, window_frames=8, grid_size=64) for spec in catalog]

    assert len(catalog) == 15
    assert {summary["topology"] for summary in summaries} == {"stack", "encoder_decoder"}
    assert any(summary["skip_connections"] is False for summary in summaries)
    assert all(summary["parameter_count"] > 0 for summary in summaries)
