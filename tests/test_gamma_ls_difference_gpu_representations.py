from __future__ import annotations

import numpy as np
import pytest
import torch

from neurobench.experiments.gamma_ls_difference import gpu_representations as torch_repr
from neurobench.experiments.gamma_ls_difference import representations as numpy_repr


def _two_frame_fit() -> dict[str, object]:
    return {
        "mean": [0.5, -0.25],
        "whitening": [[1.0, 1.0], [-2.0, 2.0]],
        "demixing": [[0.25, 0.75], [-1.0, 0.5]],
        "activity_component": 0,
        "activity_sign": -1,
    }


def _v5_fit() -> dict[str, object]:
    demixing = np.eye(6, dtype=np.float64)
    demixing[2, 0] = 0.2
    demixing[3, 1] = -0.3
    whitening = np.diag([1.0, 0.5, 2.0, 1.5, 0.75, 1.25])
    whitening[0, 1] = 0.1
    return {
        "formulation": "delay_embedding",
        "lags": [0, 1, 2, 4, 8, 16],
        "center": [0.25, -0.5, 0.75, 1.0, -1.25, 1.5],
        "whitening": whitening.tolist(),
        "demixing": demixing.tolist(),
        "residual_indices": [2, 3, 4, 5],
    }


def _assert_map_close(
    actual: torch_repr.DeviceRepresentationMap,
    expected: numpy_repr.RepresentationMap,
    *,
    rtol: float = 3e-6,
    atol: float = 3e-6,
) -> None:
    assert actual.values.dtype == torch.float32
    assert actual.source_frame_indices.dtype == torch.int64
    assert actual.values.device == actual.source_frame_indices.device
    torch.testing.assert_close(
        actual.values.cpu(),
        torch.from_numpy(expected.values),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        actual.source_frame_indices.cpu(),
        torch.from_numpy(expected.source_frame_indices),
        rtol=0,
        atol=0,
    )


def test_common_causal_preprocessing_matches_numpy_including_borders() -> None:
    movie = np.random.default_rng(19).normal(size=(7, 5, 4)).astype(np.float32)
    source_indices = np.arange(40, 47, dtype=np.int64)
    expected = numpy_repr.causal_preprocess_common_input(
        movie, source_frame_indices=source_indices
    )
    actual = torch_repr.causal_preprocess_common_input(
        torch.from_numpy(movie),
        source_frame_indices=torch.from_numpy(source_indices),
    )

    _assert_map_close(actual, expected)
    # The comparison above covers every border. Keep corner assertions explicit
    # so an accidental switch to Torch's different reflect mode cannot hide.
    torch.testing.assert_close(
        actual.values[:, (0, -1), :].cpu(),
        torch.from_numpy(expected.values[:, (0, -1), :]),
        rtol=3e-6,
        atol=3e-6,
    )
    torch.testing.assert_close(
        actual.values[:, :, (0, -1)].cpu(),
        torch.from_numpy(expected.values[:, :, (0, -1)]),
        rtol=3e-6,
        atol=3e-6,
    )
    assert actual.diagnostics["preprocessing"]["spatial_sigma_px"] == 1.0
    assert actual.diagnostics["preprocessing"]["ema_alpha"] == 0.4
    assert actual.diagnostics["labels_used"] is False


def test_all_individual_torch_arms_match_the_numpy_reference() -> None:
    movie = np.random.default_rng(23).normal(size=(21, 3, 4)).astype(np.float32)
    tensor = torch.from_numpy(movie)
    source_indices = np.arange(100, 121, dtype=np.int64)
    tensor_indices = torch.from_numpy(source_indices)
    two_frame = _two_frame_fit()
    v5 = {"fit": _v5_fit()}

    pairs = [
        (
            torch_repr.raw_representation(
                tensor, source_frame_indices=tensor_indices
            ),
            numpy_repr.raw_representation(
                movie, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.signed_difference_representation(
                tensor, source_frame_indices=tensor_indices
            ),
            numpy_repr.signed_difference_representation(
                movie, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.energy_normalized_difference_representation(
                tensor, epsilon=1e-8, source_frame_indices=tensor_indices
            ),
            numpy_repr.energy_normalized_difference_representation(
                movie, epsilon=1e-8, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.multilag_energy_normalized_difference_representation(
                tensor, epsilon=1e-8, source_frame_indices=tensor_indices
            ),
            numpy_repr.multilag_energy_normalized_difference_representation(
                movie, epsilon=1e-8, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.pca_whitened_derivative_representation(
                tensor, two_frame, source_frame_indices=tensor_indices
            ),
            numpy_repr.pca_whitened_derivative_representation(
                movie, two_frame, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.frozen_two_frame_cs_parzen_representation(
                tensor, two_frame, source_frame_indices=tensor_indices
            ),
            numpy_repr.frozen_two_frame_cs_parzen_representation(
                movie, two_frame, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.pca_whitened_delay_total_energy_representation(
                tensor, v5, source_frame_indices=tensor_indices
            ),
            numpy_repr.pca_whitened_delay_total_energy_representation(
                movie, v5, source_frame_indices=source_indices
            ),
        ),
        (
            torch_repr.frozen_v5_residual_group_representation(
                tensor, v5, source_frame_indices=tensor_indices
            ),
            numpy_repr.frozen_v5_residual_group_representation(
                movie, v5, source_frame_indices=source_indices
            ),
        ),
    ]
    for actual, expected in pairs:
        _assert_map_close(actual, expected, rtol=8e-6, atol=8e-6)
        assert actual.diagnostics["labels_used"] is False


def test_matched_six_lag_controls_use_exact_frozen_formulas() -> None:
    movie = torch.arange(40, dtype=torch.float32).reshape(20, 1, 2) / 7.0
    source_indices = torch.arange(200, 220, dtype=torch.int64)
    epsilon = 1e-6
    multilag = torch_repr.multilag_energy_normalized_difference_representation(
        movie,
        epsilon=epsilon,
        source_frame_indices=source_indices,
    )
    current = movie[16:]
    normalized = []
    for lag in (1, 2, 4, 8, 16):
        previous = movie[16 - lag : movie.shape[0] - lag]
        normalized.append(
            (current - previous)
            / torch.sqrt(current.square() + previous.square() + epsilon)
        )
    expected_multilag = torch.sqrt(
        torch.sum(torch.stack(normalized).square(), dim=0)
    )
    torch.testing.assert_close(multilag.values, expected_multilag, rtol=0, atol=0)

    frozen = _v5_fit()
    pca = torch_repr.pca_whitened_delay_total_energy_representation(
        movie, frozen, source_frame_indices=source_indices
    )
    lags = tuple(frozen["lags"])
    stack = torch.stack(
        [movie[16 - lag : movie.shape[0] - lag] for lag in lags]
    )
    center = torch.tensor(frozen["center"], dtype=torch.float32)
    whitening = torch.tensor(frozen["whitening"], dtype=torch.float32)
    whitened = (whitening @ (
        stack - center[:, None, None, None]
    ).reshape(6, -1)).reshape(6, 4, 1, 2)
    expected_pca = torch.sqrt(torch.sum(whitened.square(), dim=0))
    torch.testing.assert_close(pca.values, expected_pca, rtol=0, atol=0)

    torch.testing.assert_close(
        multilag.source_frame_indices, source_indices[16:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pca.source_frame_indices, source_indices[16:], rtol=0, atol=0
    )
    assert multilag.diagnostics["lags"] == [1, 2, 4, 8, 16]
    assert pca.diagnostics["operator"] == "frozen_whitening_Q_before_rotation"


def test_cs_parzen_uses_frozen_effective_w_matmul_q() -> None:
    movie = torch.tensor(
        [[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, -1.0]]],
        dtype=torch.float32,
    )
    fit = _two_frame_fit()
    result = torch_repr.frozen_two_frame_cs_parzen_representation(movie, fit)

    pair_stack = torch.stack((movie[:-1], movie[1:]))
    mean = torch.tensor(fit["mean"], dtype=torch.float32)
    whitening = torch.tensor(fit["whitening"], dtype=torch.float32)
    demixing = torch.tensor(fit["demixing"], dtype=torch.float32)
    centered = (pair_stack - mean[:, None, None, None]).reshape(2, -1)
    expected_all = ((demixing @ whitening) @ centered).reshape(2, 2, 1, 2)
    torch.testing.assert_close(result.values, -expected_all[0], rtol=0, atol=0)
    assert result.diagnostics["operator"] == "effective_demixing_equals_W_matmul_Q"


def test_bundle_matches_numpy_arms_and_aligns_to_six_lag_history() -> None:
    movie = np.random.default_rng(29).normal(size=(24, 4, 3)).astype(np.float32)
    source_indices = np.arange(500, 524, dtype=np.int64)
    expected = numpy_repr.build_representation_bundle(
        movie,
        frozen_two_frame=_two_frame_fit(),
        frozen_v5={"fit": _v5_fit()},
        source_frame_indices=source_indices,
    )
    actual = torch_repr.build_representation_bundle(
        torch.from_numpy(movie),
        frozen_two_frame=_two_frame_fit(),
        frozen_v5={"fit": _v5_fit()},
        source_frame_indices=torch.from_numpy(source_indices),
    )

    expected_names = set(expected.maps)
    assert set(actual.maps) == expected_names
    assert actual.diagnostics["quiet_mad_included"] is False
    assert actual.diagnostics["labels_used"] is False
    torch.testing.assert_close(
        actual.source_frame_indices,
        torch.from_numpy(expected.source_frame_indices),
        rtol=0,
        atol=0,
    )
    for name in expected_names:
        _assert_map_close(
            actual.maps[name], expected.maps[name], rtol=1e-5, atol=1e-5
        )
    assert actual.source_frame_indices.tolist() == list(range(516, 524))


def test_torch_bundle_routes_six_lag_raw_and_two_frame_preprocessed() -> None:
    movie = torch.randn(
        (24, 4, 5),
        generator=torch.Generator().manual_seed(37),
        dtype=torch.float32,
    )
    source_indices = torch.arange(800, 824, dtype=torch.int64)
    frozen_v5 = {"fit": _v5_fit()}
    bundle = torch_repr.build_representation_bundle(
        movie,
        frozen_two_frame=_two_frame_fit(),
        frozen_v5=frozen_v5,
        source_frame_indices=source_indices,
    )
    preprocessed = torch_repr.causal_preprocess_common_input(
        movie, source_frame_indices=source_indices
    )
    raw_domain_six_lag = {
        "difference_multilag_energy_normalized": (
            torch_repr.multilag_energy_normalized_difference_representation(
                movie, source_frame_indices=source_indices
            )
        ),
        "pca_whitened_delay_total_energy": (
            torch_repr.pca_whitened_delay_total_energy_representation(
                movie, frozen_v5, source_frame_indices=source_indices
            )
        ),
        "cs_parzen_delay_residual": (
            torch_repr.frozen_v5_residual_group_representation(
                movie, frozen_v5, source_frame_indices=source_indices
            )
        ),
    }
    wrongly_preprocessed_six_lag = {
        "difference_multilag_energy_normalized": (
            torch_repr.multilag_energy_normalized_difference_representation(
                preprocessed.values, source_frame_indices=source_indices
            )
        ),
        "pca_whitened_delay_total_energy": (
            torch_repr.pca_whitened_delay_total_energy_representation(
                preprocessed.values, frozen_v5, source_frame_indices=source_indices
            )
        ),
        "cs_parzen_delay_residual": (
            torch_repr.frozen_v5_residual_group_representation(
                preprocessed.values, frozen_v5, source_frame_indices=source_indices
            )
        ),
    }
    for name, expected in raw_domain_six_lag.items():
        torch.testing.assert_close(
            bundle.maps[name].values, expected.values, rtol=0, atol=0
        )
        assert not torch.allclose(
            bundle.maps[name].values,
            wrongly_preprocessed_six_lag[name].values,
            rtol=1e-5,
            atol=1e-6,
        )

    torch.testing.assert_close(
        bundle.maps["raw"].values, preprocessed.values[16:], rtol=0, atol=0
    )
    expected_two_frame = torch_repr.frozen_two_frame_cs_parzen_representation(
        preprocessed.values,
        _two_frame_fit(),
        source_frame_indices=source_indices,
    )
    torch.testing.assert_close(
        bundle.maps["cs_parzen_two_frame"].values,
        expected_two_frame.values[15:],
        rtol=0,
        atol=0,
    )
    assert "acquisition-raw" in bundle.diagnostics[
        "matched_six_lag_input_definition"
    ]


def test_torch_contracts_fail_closed_without_host_or_dtype_coercion() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        torch_repr.causal_preprocess_common_input(
            np.zeros((17, 2, 2), dtype=np.float32)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="torch.float32"):
        torch_repr.causal_preprocess_common_input(
            torch.zeros((17, 2, 2), dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="strictly contiguous"):
        torch_repr.signed_difference_representation(
            torch.zeros((17, 2, 2), dtype=torch.float32),
            source_frame_indices=torch.tensor(
                [*range(16), 18], dtype=torch.int64
            ),
        )
    invalid_whitening = _v5_fit()
    invalid_whitening["whitening"] = np.zeros((6, 6)).tolist()
    with pytest.raises(ValueError, match="whitening must be full rank"):
        torch_repr.pca_whitened_delay_total_energy_representation(
            torch.zeros((17, 2, 2), dtype=torch.float32),
            invalid_whitening,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_and_all_outputs_remain_on_device() -> None:
    movie = torch.randn(
        (23, 7, 6),
        generator=torch.Generator().manual_seed(31),
        dtype=torch.float32,
    )
    source_indices = torch.arange(700, 723, dtype=torch.int64)
    cpu = torch_repr.build_representation_bundle(
        movie,
        frozen_two_frame=_two_frame_fit(),
        frozen_v5={"fit": _v5_fit()},
        source_frame_indices=source_indices,
    )
    gpu = torch_repr.build_representation_bundle(
        movie.cuda(),
        frozen_two_frame=_two_frame_fit(),
        frozen_v5={"fit": _v5_fit()},
        source_frame_indices=source_indices.cuda(),
    )

    assert gpu.source_frame_indices.is_cuda
    for name, gpu_map in gpu.maps.items():
        assert gpu_map.values.is_cuda
        assert gpu_map.source_frame_indices.is_cuda
        assert gpu_map.values.dtype == torch.float32
        torch.testing.assert_close(
            gpu_map.values.cpu(), cpu.maps[name].values, rtol=3e-5, atol=3e-5
        )
        torch.testing.assert_close(
            gpu_map.source_frame_indices.cpu(),
            cpu.maps[name].source_frame_indices,
            rtol=0,
            atol=0,
        )
