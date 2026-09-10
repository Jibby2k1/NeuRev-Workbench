from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.pairwise_separation.sampling import causal_preprocess
from neurobench.experiments.gamma_ls_difference.representations import (
    CausalPreprocessingConfig,
    build_representation_bundle,
    causal_preprocess_common_input,
    energy_normalized_difference_representation,
    frozen_two_frame_cs_parzen_representation,
    frozen_v5_residual_group_representation,
    multilag_energy_normalized_difference_representation,
    pca_whitened_delay_total_energy_representation,
    pca_whitened_derivative_representation,
    quiet_mad_standardized_difference_representation,
    raw_representation,
    signed_difference_representation,
)


def _two_frame_fit() -> dict[str, object]:
    return {
        "mean": [0.5, -0.25],
        "whitening": [[1.0, 1.0], [-2.0, 2.0]],
        "demixing": [[0.25, 0.75], [-1.0, 0.5]],
        "activity_component": 0,
        "activity_sign": -1,
    }


def _v5_fit() -> dict[str, object]:
    return {
        "formulation": "delay_embedding",
        "lags": [0, 1, 2, 4, 8, 16],
        "center": [0.0] * 6,
        "whitening": np.eye(6).tolist(),
        "demixing": np.eye(6).tolist(),
        "residual_indices": [2, 3, 4, 5],
    }


def test_default_common_preprocessing_matches_the_maintained_pairwise_input() -> None:
    movie = np.random.default_rng(19).normal(size=(6, 5, 4)).astype(np.float32)
    expected = causal_preprocess(movie, spatial_sigma_px=1.0, ema_span_frames=4.0)
    actual = causal_preprocess_common_input(movie)
    np.testing.assert_array_equal(actual.values, expected)
    assert actual.diagnostics["preprocessing"]["ema_alpha"] == 0.4
    assert actual.diagnostics["preprocessing"]["equivalent_ema_span_frames"] == 4.0


def test_common_preprocessing_and_exact_difference_formulas_preserve_alignment() -> None:
    movie = np.asarray(
        [
            [[0.0, 2.0]],
            [[2.0, 4.0]],
            [[6.0, 2.0]],
            [[4.0, 8.0]],
        ],
        dtype=np.float32,
    )
    source_indices = np.arange(40, 44, dtype=np.int64)
    preprocessing = causal_preprocess_common_input(
        movie,
        config=CausalPreprocessingConfig(spatial_sigma_px=0.0, ema_alpha=0.5),
        source_frame_indices=source_indices,
    )
    expected_preprocessed = np.empty_like(movie)
    expected_preprocessed[0] = movie[0]
    for frame in range(1, movie.shape[0]):
        expected_preprocessed[frame] = (
            0.5 * movie[frame] + 0.5 * expected_preprocessed[frame - 1]
        )
    np.testing.assert_allclose(preprocessing.values, expected_preprocessed)
    np.testing.assert_array_equal(preprocessing.source_frame_indices, source_indices)
    assert preprocessing.diagnostics["preprocessing"]["ema_alpha"] == 0.5

    raw = raw_representation(movie, source_frame_indices=source_indices)
    signed = signed_difference_representation(movie, source_frame_indices=source_indices)
    energy = energy_normalized_difference_representation(
        movie,
        epsilon=1e-8,
        source_frame_indices=source_indices,
    )
    expected_signed = movie[1:].astype(np.float64) - movie[:-1].astype(np.float64)
    expected_energy = expected_signed / np.sqrt(
        movie[:-1].astype(np.float64) ** 2
        + movie[1:].astype(np.float64) ** 2
        + 1e-8
    )
    np.testing.assert_array_equal(raw.values, movie)
    np.testing.assert_allclose(signed.values, expected_signed.astype(np.float32))
    np.testing.assert_allclose(energy.values, expected_energy.astype(np.float32))
    np.testing.assert_array_equal(signed.source_frame_indices, source_indices[1:])
    np.testing.assert_array_equal(energy.source_frame_indices, source_indices[1:])
    assert energy.diagnostics["epsilon"] == 1e-8


def test_quiet_mad_uses_current_frame_alignment_and_a_spatial_floor() -> None:
    desired_difference = np.asarray(
        [
            [[0.0, 1.0]],
            [[2.0, 1.0]],
            [[4.0, 1.0]],
            [[6.0, 1.0]],
            [[8.0, 1.0]],
        ],
        dtype=np.float32,
    )
    movie = np.concatenate(
        (np.zeros((1, 1, 2), dtype=np.float32), np.cumsum(desired_difference, axis=0)),
        axis=0,
    )
    quiet_frame_mask = np.asarray([False, True, True, True, False, False])
    source_indices = np.arange(100, 106, dtype=np.int64)
    result = quiet_mad_standardized_difference_representation(
        movie,
        quiet_frame_mask,
        source_frame_indices=source_indices,
    )

    quiet_differences = desired_difference[:3].astype(np.float64)
    center = np.median(quiet_differences, axis=0)
    mad = 1.4826 * np.median(np.abs(quiet_differences - center), axis=0)
    floor = float(mad[0, 0])
    expected = (desired_difference.astype(np.float64) - center) / np.maximum(mad, floor)
    np.testing.assert_allclose(result.values, expected.astype(np.float32))
    np.testing.assert_array_equal(result.source_frame_indices, source_indices[1:])
    assert result.diagnostics["quiet_aligned_output_frame_count"] == 3
    assert result.diagnostics["mad_floor"] == pytest.approx(floor)
    assert result.diagnostics["floored_pixel_fraction"] == 0.5


def test_frozen_pca_and_cs_parzen_apply_the_supplied_linear_maps_exactly() -> None:
    movie = np.asarray(
        [
            [[0.0, 1.0]],
            [[2.0, 3.0]],
            [[4.0, -1.0]],
            [[1.0, 5.0]],
        ],
        dtype=np.float32,
    )
    fit = _two_frame_fit()
    source_indices = np.arange(10, 14, dtype=np.int64)
    pca = pca_whitened_derivative_representation(
        movie,
        {key: fit[key] for key in ("mean", "whitening")},
        source_frame_indices=source_indices,
    )
    cs_parzen = frozen_two_frame_cs_parzen_representation(
        movie,
        fit,
        source_frame_indices=source_indices,
    )

    pair_stack = np.stack((movie[:-1], movie[1:]), axis=0).astype(np.float64)
    centered = pair_stack - np.asarray(fit["mean"], dtype=np.float64)[:, None, None, None]
    flattened = centered.reshape(2, -1)
    whitening = np.asarray(fit["whitening"], dtype=np.float64)
    effective_demixing = np.asarray(fit["demixing"], dtype=np.float64) @ whitening
    expected_pca = (whitening @ flattened)[1].reshape(movie.shape[0] - 1, 1, 2)
    expected_cs_parzen = -(effective_demixing @ flattened)[0].reshape(
        movie.shape[0] - 1,
        1,
        2,
    )
    np.testing.assert_allclose(pca.values, expected_pca.astype(np.float32))
    np.testing.assert_allclose(cs_parzen.values, expected_cs_parzen.astype(np.float32))
    np.testing.assert_array_equal(pca.source_frame_indices, source_indices[1:])
    np.testing.assert_array_equal(cs_parzen.source_frame_indices, source_indices[1:])
    assert pca.diagnostics["selected_component"] == 1
    assert pca.diagnostics["signed_derivative_cosine"] == pytest.approx(1.0)
    assert cs_parzen.diagnostics["fit_reused_without_refitting"] is True


def test_frozen_v5_residual_energy_uses_exact_six_lag_stack_and_alignment() -> None:
    movie = np.arange(20, dtype=np.float32).reshape(20, 1, 1)
    source_indices = np.arange(200, 220, dtype=np.int64)
    result = frozen_v5_residual_group_representation(
        movie,
        _v5_fit(),
        source_frame_indices=source_indices,
    )
    expected = []
    for frame in range(16, movie.shape[0]):
        residual_values = [
            movie[frame - 2, 0, 0],
            movie[frame - 4, 0, 0],
            movie[frame - 8, 0, 0],
            movie[frame - 16, 0, 0],
        ]
        expected.append(np.sqrt(np.sum(np.square(residual_values, dtype=np.float64))))
    np.testing.assert_allclose(result.values[:, 0, 0], np.asarray(expected, dtype=np.float32))
    np.testing.assert_array_equal(result.source_frame_indices, source_indices[16:])
    assert result.diagnostics["lags"] == [0, 1, 2, 4, 8, 16]
    assert result.diagnostics["residual_indices"] == [2, 3, 4, 5]


def test_matched_six_lag_controls_use_exact_formulas_and_history() -> None:
    movie = (np.arange(20, dtype=np.float32) + 1.0).reshape(20, 1, 1)
    source_indices = np.arange(300, 320, dtype=np.int64)
    epsilon = 1e-6
    difference_energy = multilag_energy_normalized_difference_representation(
        movie,
        epsilon=epsilon,
        source_frame_indices=source_indices,
    )
    current = movie[16:].astype(np.float64)
    expected_squared = np.zeros_like(current)
    for lag in (1, 2, 4, 8, 16):
        previous = movie[16 - lag : 20 - lag].astype(np.float64)
        normalized = (current - previous) / np.sqrt(
            current**2 + previous**2 + epsilon
        )
        expected_squared += normalized**2
    np.testing.assert_allclose(
        difference_energy.values,
        np.sqrt(expected_squared).astype(np.float32),
    )
    np.testing.assert_array_equal(
        difference_energy.source_frame_indices,
        source_indices[16:],
    )
    assert difference_energy.diagnostics["lags"] == [1, 2, 4, 8, 16]

    fit = _v5_fit()
    fit["center"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    fit["whitening"] = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).tolist()
    fit["demixing"] = (100.0 * np.eye(6)).tolist()
    pca_energy = pca_whitened_delay_total_energy_representation(
        movie,
        fit,
        source_frame_indices=source_indices,
    )
    lag_stack = np.stack(
        [movie[16 - lag : 20 - lag] for lag in (0, 1, 2, 4, 8, 16)],
        axis=0,
    ).astype(np.float64)
    centered = lag_stack - np.asarray(fit["center"])[:, None, None, None]
    whitened = np.asarray(fit["whitening"]) @ centered.reshape(6, -1)
    expected_pca = np.sqrt(np.sum(whitened**2, axis=0)).reshape(4, 1, 1)
    np.testing.assert_allclose(pca_energy.values, expected_pca.astype(np.float32))
    np.testing.assert_array_equal(pca_energy.source_frame_indices, source_indices[16:])
    assert pca_energy.diagnostics["coordinate_count"] == 6
    assert pca_energy.diagnostics["ica_rotation_applied"] is False
    diagnostic_text = str(pca_energy.diagnostics).lower()
    assert "residual" not in diagnostic_text
    assert "subspace" not in diagnostic_text


def test_bundle_builds_once_and_trims_all_arms_to_v5_history() -> None:
    rng = np.random.default_rng(7)
    movie = rng.normal(size=(24, 2, 2)).astype(np.float32)
    source_indices = np.arange(500, 524, dtype=np.int64)
    bundle = build_representation_bundle(
        movie,
        frozen_two_frame=_two_frame_fit(),
        frozen_v5={"fit": _v5_fit()},
        preprocessing=CausalPreprocessingConfig(spatial_sigma_px=0.0, ema_alpha=1.0),
        source_frame_indices=source_indices,
    )
    expected_names = {
        "raw",
        "difference_signed",
        "difference_energy_normalized",
        "difference_multilag_energy_normalized",
        "pca_whitened_derivative",
        "pca_whitened_delay_total_energy",
        "cs_parzen_two_frame",
        "cs_parzen_delay_residual",
    }
    assert set(bundle.maps) == expected_names
    np.testing.assert_array_equal(bundle.source_frame_indices, source_indices[16:])
    for representation in bundle.maps.values():
        np.testing.assert_array_equal(
            representation.source_frame_indices,
            source_indices[16:],
        )
        assert representation.values.shape == (8, 2, 2)
    np.testing.assert_array_equal(bundle.maps["raw"].values, movie[16:])
    np.testing.assert_allclose(
        bundle.maps["difference_signed"].values,
        movie[16:] - movie[15:-1],
    )
    assert bundle.diagnostics["preprocessing_applied_once"] is True
    assert bundle.diagnostics["arm_count"] == 8
    assert bundle.diagnostics["raw_arm_definition"] == "common causal preprocessed input"
    assert bundle.diagnostics["common_alignment"]["history_frames"] == 16
    assert bundle.maps["raw"].diagnostics["common_alignment"]["trimmed_prefix_frames"] == 16
    assert (
        bundle.maps["difference_signed"].diagnostics["common_alignment"]["trimmed_prefix_frames"]
        == 15
    )


def test_bundle_keeps_v5_level_raw_and_adjacent_level_preprocessed() -> None:
    movie = np.random.default_rng(31).normal(size=(24, 4, 5)).astype(np.float32)
    source_indices = np.arange(700, 724, dtype=np.int64)
    frozen_v5 = {"fit": _v5_fit()}
    bundle = build_representation_bundle(
        movie,
        frozen_two_frame=_two_frame_fit(),
        frozen_v5=frozen_v5,
        source_frame_indices=source_indices,
    )
    preprocessed = causal_preprocess_common_input(
        movie, source_frame_indices=source_indices
    )

    raw_domain_six_lag = {
        "difference_multilag_energy_normalized": (
            multilag_energy_normalized_difference_representation(
                movie, source_frame_indices=source_indices
            )
        ),
        "pca_whitened_delay_total_energy": (
            pca_whitened_delay_total_energy_representation(
                movie, frozen_v5, source_frame_indices=source_indices
            )
        ),
        "cs_parzen_delay_residual": frozen_v5_residual_group_representation(
            movie, frozen_v5, source_frame_indices=source_indices
        ),
    }
    wrongly_preprocessed_six_lag = {
        "difference_multilag_energy_normalized": (
            multilag_energy_normalized_difference_representation(
                preprocessed.values, source_frame_indices=source_indices
            )
        ),
        "pca_whitened_delay_total_energy": (
            pca_whitened_delay_total_energy_representation(
                preprocessed.values, frozen_v5, source_frame_indices=source_indices
            )
        ),
        "cs_parzen_delay_residual": frozen_v5_residual_group_representation(
            preprocessed.values, frozen_v5, source_frame_indices=source_indices
        ),
    }
    for name, expected in raw_domain_six_lag.items():
        np.testing.assert_array_equal(bundle.maps[name].values, expected.values)
        assert not np.allclose(
            bundle.maps[name].values,
            wrongly_preprocessed_six_lag[name].values,
            rtol=1e-5,
            atol=1e-6,
        )

    np.testing.assert_array_equal(
        bundle.maps["raw"].values, preprocessed.values[16:]
    )
    expected_two_frame = frozen_two_frame_cs_parzen_representation(
        preprocessed.values,
        _two_frame_fit(),
        source_frame_indices=source_indices,
    )
    np.testing.assert_array_equal(
        bundle.maps["cs_parzen_two_frame"].values,
        expected_two_frame.values[15:],
    )
    assert "acquisition-raw" in bundle.diagnostics[
        "matched_six_lag_input_definition"
    ]


def test_alignment_and_frozen_v5_contracts_fail_closed() -> None:
    movie = np.zeros((20, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly contiguous"):
        signed_difference_representation(
            movie,
            source_frame_indices=np.asarray([*range(19), 21], dtype=np.int64),
        )
    invalid_v5 = _v5_fit()
    invalid_v5["lags"] = [0, 1, 2, 4, 8, 15]
    with pytest.raises(ValueError, match="lags must be exactly"):
        frozen_v5_residual_group_representation(movie, invalid_v5)
    invalid_whitening = _v5_fit()
    invalid_whitening["whitening"] = np.zeros((6, 6)).tolist()
    with pytest.raises(ValueError, match="whitening must be full rank"):
        pca_whitened_delay_total_energy_representation(movie, invalid_whitening)
