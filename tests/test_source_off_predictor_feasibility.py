from __future__ import annotations

import json

import numpy as np
import pytest

from neurobench.experiments.neuron_identifiability import source_off_predictor_feasibility as feasibility
from neurobench.experiments.neuron_identifiability.source_off_predictor_feasibility import (
    COMMON_EVALUATION_START_FRAME_ZERO,
    GATE_THRESHOLDS,
    METHODS,
    SourceOffPredictorFeasibilityError,
    causal_ema_prediction,
    causal_temporal_median_prediction,
    causal_zero_order_hold_prediction,
    evaluate_feasibility_gate,
    fit_low_rank_ar1,
    fit_per_pixel_ar1,
    low_rank_ar1_prediction,
    no_subtraction_prediction,
    per_pixel_ar1_prediction,
    source_off_metrics,
)


def _movie(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(32, dtype=np.float32)[:, None, None]
    spatial = np.linspace(-0.5, 0.5, 64 * 64, dtype=np.float32).reshape(1, 64, 64)
    return (spatial + 0.04 * time + rng.normal(0, 0.01, (32, 64, 64))).astype(np.float32)


def test_no_subtraction_is_residual_identity_and_metric_ratios_are_one() -> None:
    movie = _movie()
    prediction = no_subtraction_prediction(movie)
    assert np.array_equal(prediction, np.zeros_like(movie))
    assert np.array_equal(movie - prediction, movie)
    metrics = source_off_metrics(movie, prediction)
    assert metrics["evaluation_start_frame_zero"] == COMMON_EVALUATION_START_FRAME_ZERO
    assert metrics["evaluation_frame_count"] == 24
    assert metrics["centered_rms_ratio"] == pytest.approx(1.0)
    assert metrics["centered_mad_ratio"] == pytest.approx(1.0)
    assert metrics["dynamic_mad_ratio"] == pytest.approx(1.0)
    assert metrics["prediction_best_lag_frames"] is None


@pytest.mark.parametrize(
    "predictor",
    [
        causal_zero_order_hold_prediction,
        causal_ema_prediction,
        causal_temporal_median_prediction,
    ],
)
def test_fixed_causal_predictors_cannot_see_future(predictor) -> None:
    movie = _movie()
    changed = movie.copy()
    changed[20:] += 1000.0
    first = predictor(movie)
    second = predictor(changed)
    # The prediction at frame 20 uses frames strictly before 20.
    assert np.array_equal(first[:21], second[:21])


def test_zero_order_hold_reports_one_frame_prediction_delay() -> None:
    movie = np.random.default_rng(3).normal(size=(32, 64, 64)).astype(np.float32)
    metrics = source_off_metrics(movie, causal_zero_order_hold_prediction(movie))
    assert metrics["prediction_best_lag_frames"] == 1
    assert metrics["prediction_best_lag_correlation"] > 0.99


def test_train_only_per_pixel_ar1_recovers_known_affine_process() -> None:
    rng = np.random.default_rng(7)
    clips = np.empty((5, 12, 64, 64), dtype=np.float32)
    clips[:, 0] = rng.normal(size=(5, 64, 64))
    for frame in range(1, clips.shape[1]):
        clips[:, frame] = 0.2 + 0.75 * clips[:, frame - 1]
    state = fit_per_pixel_ar1(clips, chunk_size=2)
    assert np.median(state.coefficient) == pytest.approx(0.75, abs=1e-5)
    assert np.median(state.intercept) == pytest.approx(0.2, abs=1e-5)
    movie = np.concatenate((clips[0], clips[0, -1:]), axis=0)
    prediction = per_pixel_ar1_prediction(movie[:32] if movie.shape[0] >= 32 else np.pad(movie, ((0, 32 - movie.shape[0]), (0, 0), (0, 0)), mode="edge"), state)
    assert prediction.shape == (32, 64, 64)
    assert np.isfinite(prediction).all()


def test_low_rank_fit_is_deterministic_and_prediction_is_finite() -> None:
    rng = np.random.default_rng(11)
    spatial_a = np.sin(np.linspace(0, np.pi, 64, dtype=np.float32))[:, None]
    spatial_b = np.cos(np.linspace(0, np.pi, 64, dtype=np.float32))[None, :]
    clips = np.empty((4, 10, 64, 64), dtype=np.float32)
    for clip in range(clips.shape[0]):
        for frame in range(clips.shape[1]):
            clips[clip, frame] = (
                (0.3 + 0.02 * clip + 0.03 * frame) * spatial_a
                + (0.5 - 0.01 * frame) * spatial_b
                + rng.normal(0, 1e-3, (64, 64))
            )
    first = fit_low_rank_ar1(clips, rank=2, sample_frames=24, random_seed=19, chunk_size=2)
    repeated = fit_low_rank_ar1(clips, rank=2, sample_frames=24, random_seed=19, chunk_size=2)
    assert np.array_equal(first.sampled_flat_frame_indices, repeated.sampled_flat_frame_indices)
    assert np.allclose(first.components, repeated.components)
    movie = np.pad(clips[0], ((0, 22), (0, 0), (0, 0)), mode="edge")
    prediction = low_rank_ar1_prediction(movie, first)
    assert prediction.shape == movie.shape
    assert np.isfinite(prediction).all()


def _aggregate_row(partition: str, method: str, *, passing: bool) -> dict[str, object]:
    return {
        "dataset_partition": partition,
        "method_id": method,
        "centered_rms_ratio_median": 0.8 if passing else 1.1,
        "dynamic_mad_ratio_median": 0.9 if passing else 1.1,
        "residual_seam_to_interior_jump_ratio_median": 1.0 if passing else 1.5,
        "temporal_spectral_flatness_median": 0.7 if passing else 0.4,
        "residual_absolute_lag1_autocorrelation_median": 0.1 if passing else 0.8,
    }


def test_source_off_gate_uses_both_partitions_and_every_recording() -> None:
    aggregate = []
    recordings = []
    for partition in ("held_validation", "registered_source_off_windows"):
        for method in METHODS:
            passing = method == "causal_zero_order_hold"
            baseline = method == "no_subtraction_identity"
            row = _aggregate_row(partition, method, passing=passing)
            if baseline:
                row.update(
                    centered_rms_ratio_median=1.0,
                    dynamic_mad_ratio_median=1.0,
                    residual_seam_to_interior_jump_ratio_median=1.0,
                    temporal_spectral_flatness_median=0.5,
                    residual_absolute_lag1_autocorrelation_median=0.5,
                )
            aggregate.append(row)
            recordings.append(
                {
                    "dataset_partition": partition,
                    "recording_id": f"{partition}_recording",
                    "method_id": method,
                    "centered_rms_ratio_median": 0.8 if passing else 1.1,
                }
            )
    result = evaluate_feasibility_gate(aggregate, recordings)
    assert result["eligible_predictor_methods"] == ["causal_zero_order_hold"]
    assert result["methods"]["causal_zero_order_hold"]["passed"] is True
    assert result["methods"]["frozen_jepa_decoder_run_b"]["passed"] is False
    assert result["truth_access"] == "none_no_injections_no_ROIs_no_detector_outcomes"
    assert GATE_THRESHOLDS["median_centered_rms_ratio_max"] == 0.9


def test_shape_and_nonfinite_guards_fail_closed() -> None:
    with pytest.raises(SourceOffPredictorFeasibilityError, match="shape"):
        no_subtraction_prediction(np.zeros((31, 64, 64), dtype=np.float32))
    movie = _movie()
    movie[0, 0, 0] = np.nan
    with pytest.raises(SourceOffPredictorFeasibilityError, match="nonfinite"):
        causal_ema_prediction(movie)


def test_canonical_experiment_run_and_output_bindings(tmp_path) -> None:
    assert feasibility.EXPERIMENT_ID == "NREV-EXP-0030"
    assert feasibility.RUN_ID == "NREV-RUN-EXP-0030-SCREEN-20260830-B"
    assert feasibility.DEFAULT_OUTPUT_ROOT == (
        "Outputs/NeuronIdentifiability/NREV-EXP-0030/runs/"
        "NREV-RUN-EXP-0030-SCREEN-20260830-B"
    )
    repository = tmp_path.resolve()
    expected = repository / feasibility.DEFAULT_OUTPUT_ROOT
    assert feasibility._canonical_output(repository, None) == expected
    assert feasibility._canonical_output(repository, expected) == expected
    with pytest.raises(SourceOffPredictorFeasibilityError, match="non-canonical"):
        feasibility._canonical_output(repository, tmp_path / "redirected")


def test_portable_command_provenance_passes_publication_boundary_path_scan() -> None:
    runtime = feasibility._runtime_provenance("cpu", 4, render_media=True)
    command = runtime["command_argv_portable"]
    assert command[:3] == ["python", "-m", feasibility.RUNNER_MODULE]
    assert command[command.index("--repository-root") + 1] == "."
    assert command[command.index("--data-root") + 1] == "<runtime-data-root>"
    assert feasibility._portable_command_passes_publication_boundary(command)
    serialized = json.dumps(runtime, sort_keys=True)
    assert "/home/" not in serialized
    assert "/Users/" not in serialized
    assert "\\\\Users\\\\" not in serialized

    leaked = [*command, "/" + "home" + "/example/private-data"]
    assert not feasibility._portable_command_passes_publication_boundary(leaked)
