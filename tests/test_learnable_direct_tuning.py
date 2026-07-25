from __future__ import annotations

import numpy as np

from neurobench.experiments.learnable_contrast.direct_tuning import _model_class, screen_matrix


def test_screen_matrix_has_nine_unique_cumulative_conditions():
    rows = screen_matrix()
    assert len(rows) == 9
    assert len({row["combination_id"] for row in rows}) == 9
    assert {row["variant"] for row in rows} == {"temporal_amplitude", "spatial", "guarded_auxiliary"}
    assert {row["learning_rate"] for row in rows} == {3e-5, 1e-4, 3e-4}


def test_temporal_and_spatial_initialization_exactly_match_direct_pooling():
    import torch

    Model = _model_class()
    x = torch.from_numpy(np.random.default_rng(4).random((2, 7, 1, 11, 13), dtype=np.float32))
    expected = 0.25 * (torch.logsumexp(x.squeeze(2) / 0.25, dim=1) - np.log(7))
    for variant in ("temporal_amplitude", "spatial"):
        model = Model(variant, 21, 5, 0.001)
        actual = model.pool(x)
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_bounded_parameterizations_stay_near_direct_detector():
    import torch

    Model = _model_class()
    model = Model("guarded_auxiliary", 21, 5, 0.001)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(100)
    gain, gamma, tau = model.amplitude_parameters()
    beta_c, beta_h = model.auxiliary_weights()
    assert 0.77 < gain < 1.29 and 0.77 < gamma < 1.29
    assert 0.15 < tau < 0.42
    assert beta_c <= 0.05 and beta_h <= 0.05
    assert abs(float(model.spatial_kernel().sum().detach()) - 1) < 1e-5


def test_main_cli_registers_direct_tuning():
    from neurobench.cli.main import build_parser

    args = build_parser(active_command="experiment").parse_args([
        "experiment", "learnable-contrast", "direct-tuning", "--config", "v3.json"])
    assert args.experiment_workflow == "learnable-contrast"
    assert args.experiment_action == "direct-tuning"
    assert callable(args.func)
