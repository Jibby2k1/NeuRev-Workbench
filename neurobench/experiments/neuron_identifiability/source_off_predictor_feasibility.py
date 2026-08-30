"""Source-off-only feasibility benchmark for conditional background predictors.

This is a bounded, non-claim-bearing engineering diagnostic.  It reuses the
hash-frozen EXP-0028 clip banks and the frozen EXP-0029 Run-B JEPA/random
providers and pixel decoders, but it does not reuse injected sources, source
truth, ROI labels, detector scores, or recovery outcomes.  Simple predictors
are fitted only on the 512 normalized training clips.  All comparisons are
made on the 96 recording-held-out normalized clips and the 12 registered quiet
source-off windows in the same normalized pixel domain.

The identity arm means *no subtraction*: its predicted background is zero and
its residual is exactly the observed normalized movie.  This avoids the
ambiguous and scientifically unhelpful convention in which an identity
predictor would subtract the complete observation and return a zero residual.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .contracts import atomic_json, atomic_text, stable_hash
from .discovery import sha256_file
from .jepa_background_residual import (
    FrozenLatentPixelBackgroundModel,
    load_frozen_jepa_checkpoint,
    load_frozen_random_checkpoint,
    module_state_sha256,
    predict_full_background,
)
from .jepa_data import ClipRequest, TemporalClipContract, load_recording_inventory, read_sequential_clip
from .jepa_pilot import inventory_source_holdout


SCHEMA_VERSION = 1
EXPERIMENT_ID = "NREV-EXP-0030"
RUN_ID = "NREV-RUN-EXP-0030-SCREEN-20260830-B"
PLANNED_AT_UTC = "2026-08-30T14:08:04Z"
RUNNER_MODULE = (
    "neurobench.experiments.neuron_identifiability."
    "source_off_predictor_feasibility"
)
DEFAULT_OUTPUT_ROOT = (
    "Outputs/NeuronIdentifiability/NREV-EXP-0030/runs/"
    "NREV-RUN-EXP-0030-SCREEN-20260830-B"
)
DESCRIPTOR_PATH = "examples/source_off_predictor_feasibility_v1.example.json"
EXP0028_ROOT = (
    "Outputs/NeuronIdentifiability/NREV-EXP-0028/runs/"
    "NREV-RUN-EXP-0028-SCREEN-20260829-B"
)
EXP0029_RUN_B_ROOT = (
    "Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/"
    "NREV-RUN-EXP-0029-SCREEN-20260830-B"
)

EXP0028_ARTIFACT_INDEX_SHA256 = "6e6c7ef9d8c70c2c52fcb809ea1f4a65578987e558302a3a63b576cb9eb448bd"
EXP0029_RUN_B_ARTIFACT_INDEX_SHA256 = "34bd983ffdf55cb0941bbd1443adf939f4cb86ac4ffe51a736b2d6d9383cfc1c"
JEPA_CHECKPOINT_SHA256 = "1d9a1c261f5d49fc418e211cfa1f3250079ea08da716c72077feb34065c63a67"
RUN_B_DECODER_CHECKPOINT_SHA256 = "37e4ea6db34d0b111de531b785e470717cdc39795a7c27b1d05979a1ab926d64"
TRAIN_CACHE_SHA256 = "d49e927e29044a5d7105d62375f678159fed59f93e7887fc856a435c3504c3bc"
VALIDATION_CACHE_SHA256 = "f253a35601034aa57b6c7bd13cc6aa0d1886f9e534db422014bcbc49d3c3dc38"
CLIP_BANK_MANIFEST_SHA256 = "ca9386a89cc42683fb5962cb4b4839ea4b3fc0918ed8dbab6d3d3aefcde3ac35"
BACKGROUND_WINDOW_MANIFEST_SHA256 = "ea3cd2a77ae34df3c1718b46aeb3b071c13734b15b33fd9e284a92e809350d25"
DATA_DESCRIPTOR_SHA256 = "42f1dedaee1a1acc2b24c1933ac10859cc06c8e8e6b53d5f3525c99dd21910ba"
NORMALIZATION_SHA256 = "ec48d7f89bb0fbf3bce5305ad1ecebcff2b43ad15ef1af7910405df28ecdb4a4"
NORMALIZATION_CENTER = 489.0
NORMALIZATION_SCALE = 289.10699999999997

TRAIN_SHAPE = (512, 32, 64, 64)
VALIDATION_SHAPE = (96, 32, 64, 64)
WINDOW_SHAPE = (32, 64, 64)
COMMON_EVALUATION_START_FRAME_ZERO = 8
EMA_ALPHA = 0.25
CAUSAL_MEDIAN_WINDOW = 7
LOW_RANK_RANK = 8
LOW_RANK_SAMPLE_FRAMES = 4096
LOW_RANK_RANDOM_SEED = 7301
PATCH_YX = (8, 8)

METHODS = (
    "no_subtraction_identity",
    "causal_zero_order_hold",
    "causal_ema_alpha_0p25",
    "causal_temporal_median_w7",
    "per_pixel_ar1_train512",
    "low_rank_ar1_rank8_train512",
    "frozen_jepa_decoder_run_b",
    "frozen_random_decoder_run_b",
)

DISPLAY_NAMES = {
    "no_subtraction_identity": "No subtraction",
    "causal_zero_order_hold": "Last frame / ZOH",
    "causal_ema_alpha_0p25": "Causal EMA (a=0.25)",
    "causal_temporal_median_w7": "Causal median (w=7)",
    "per_pixel_ar1_train512": "Per-pixel AR(1)",
    "low_rank_ar1_rank8_train512": "Low-rank AR(1), r=8",
    "frozen_jepa_decoder_run_b": "Frozen JEPA decoder",
    "frozen_random_decoder_run_b": "Frozen random decoder",
}

GATE_THRESHOLDS = {
    "median_centered_rms_ratio_max": 0.90,
    "median_dynamic_mad_ratio_max": 1.00,
    "median_residual_seam_ratio_max": 1.25,
    "spectral_flatness_change_vs_no_subtraction_min": 0.0,
    "absolute_lag1_autocorrelation_change_vs_no_subtraction_max": 0.0,
    "every_recording_centered_rms_ratio_max": 1.00,
}


class SourceOffPredictorFeasibilityError(ValueError):
    """Raised when a frozen benchmark contract cannot be preserved."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceOffPredictorFeasibilityError(f"JSON root must be an object: {path}")
    return value


def _finite_tyx(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or tuple(array.shape) != WINDOW_SHAPE:
        raise SourceOffPredictorFeasibilityError(
            f"{name} must have shape {WINDOW_SHAPE}, observed {tuple(array.shape)}"
        )
    if not np.isfinite(array).all():
        raise SourceOffPredictorFeasibilityError(f"{name} contains nonfinite values")
    return array


def _finite_bank(values: np.ndarray, expected: Sequence[int], *, name: str) -> np.ndarray:
    array = np.asarray(values)
    shape = tuple(int(value) for value in expected)
    if tuple(array.shape) != shape or array.dtype != np.float32:
        raise SourceOffPredictorFeasibilityError(
            f"{name} must be float32 {shape}, observed {array.dtype} {array.shape}"
        )
    if not np.isfinite(array).all():
        raise SourceOffPredictorFeasibilityError(f"{name} contains nonfinite values")
    return array


def _finite_fit_bank(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != 4
        or array.shape[0] < 2
        or array.shape[1] < 3
        or tuple(array.shape[2:]) != (64, 64)
        or array.dtype != np.float32
    ):
        raise SourceOffPredictorFeasibilityError(
            f"{name} must be float32 [N>=2,T>=3,64,64], observed {array.dtype} {array.shape}"
        )
    if not np.isfinite(array).all():
        raise SourceOffPredictorFeasibilityError(f"{name} contains nonfinite values")
    return array


def _normal_consistent_mad(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = np.median(array)
    return float(1.4826 * np.median(np.abs(array - median)))


def no_subtraction_prediction(movie: np.ndarray) -> np.ndarray:
    """Return a zero background so residual == normalized observation."""

    values = _finite_tyx(movie, name="movie")
    return np.zeros_like(values, dtype=np.float32)


def causal_zero_order_hold_prediction(movie: np.ndarray) -> np.ndarray:
    values = _finite_tyx(movie, name="movie")
    prediction = np.empty_like(values)
    prediction[0] = values[0]
    prediction[1:] = values[:-1]
    return prediction


def causal_ema_prediction(movie: np.ndarray, *, alpha: float = EMA_ALPHA) -> np.ndarray:
    values = _finite_tyx(movie, name="movie")
    resolved = float(alpha)
    if not 0.0 < resolved <= 1.0:
        raise SourceOffPredictorFeasibilityError("EMA alpha must be in (0, 1]")
    prediction = np.empty_like(values)
    state = values[0].copy()
    prediction[0] = state
    for frame in range(1, values.shape[0]):
        prediction[frame] = state
        state = resolved * values[frame] + (1.0 - resolved) * state
    return prediction


def causal_temporal_median_prediction(
    movie: np.ndarray, *, window: int = CAUSAL_MEDIAN_WINDOW
) -> np.ndarray:
    values = _finite_tyx(movie, name="movie")
    width = int(window)
    if width < 1:
        raise SourceOffPredictorFeasibilityError("causal median window must be positive")
    prediction = np.empty_like(values)
    prediction[0] = values[0]
    for frame in range(1, values.shape[0]):
        prediction[frame] = np.median(values[max(0, frame - width) : frame], axis=0)
    return prediction


@dataclass(frozen=True)
class PerPixelAR1State:
    intercept: np.ndarray
    coefficient: np.ndarray
    observation_count_per_pixel: int

    def validate(self) -> None:
        if self.intercept.shape != (64, 64) or self.coefficient.shape != (64, 64):
            raise SourceOffPredictorFeasibilityError("AR(1) state must be 64x64")
        if not np.isfinite(self.intercept).all() or not np.isfinite(self.coefficient).all():
            raise SourceOffPredictorFeasibilityError("AR(1) state must be finite")
        if self.observation_count_per_pixel < 1:
            raise SourceOffPredictorFeasibilityError("AR(1) fit count must be positive")


def fit_per_pixel_ar1(training_clips: np.ndarray, *, chunk_size: int = 16) -> PerPixelAR1State:
    """Fit one affine AR(1) coefficient per relative crop pixel on train only."""

    clips = _finite_fit_bank(training_clips, name="training_clips")
    total_x = np.zeros((64, 64), dtype=np.float64)
    total_y = np.zeros((64, 64), dtype=np.float64)
    total_xx = np.zeros((64, 64), dtype=np.float64)
    total_xy = np.zeros((64, 64), dtype=np.float64)
    count = 0
    for start in range(0, clips.shape[0], int(chunk_size)):
        batch = np.asarray(clips[start : start + int(chunk_size)], dtype=np.float64)
        previous = batch[:, :-1]
        current = batch[:, 1:]
        axes = (0, 1)
        total_x += previous.sum(axis=axes)
        total_y += current.sum(axis=axes)
        total_xx += np.square(previous).sum(axis=axes)
        total_xy += (previous * current).sum(axis=axes)
        count += int(previous.shape[0] * previous.shape[1])
    mean_x = total_x / count
    mean_y = total_y / count
    denominator = total_xx - total_x * total_x / count
    numerator = total_xy - total_x * total_y / count
    coefficient = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )
    coefficient = np.clip(coefficient, -0.999, 0.999)
    intercept = mean_y - coefficient * mean_x
    state = PerPixelAR1State(
        intercept=intercept.astype(np.float32),
        coefficient=coefficient.astype(np.float32),
        observation_count_per_pixel=count,
    )
    state.validate()
    return state


def per_pixel_ar1_prediction(movie: np.ndarray, state: PerPixelAR1State) -> np.ndarray:
    values = _finite_tyx(movie, name="movie")
    state.validate()
    prediction = np.empty_like(values)
    prediction[0] = values[0]
    prediction[1:] = state.intercept[None] + state.coefficient[None] * values[:-1]
    return prediction


@dataclass(frozen=True)
class LowRankAR1State:
    mean_image: np.ndarray
    components: np.ndarray
    latent_intercept: np.ndarray
    latent_coefficient: np.ndarray
    rank: int
    sampled_flat_frame_indices: np.ndarray
    observation_count_per_latent: int

    def validate(self) -> None:
        if self.mean_image.shape != (64, 64):
            raise SourceOffPredictorFeasibilityError("low-rank mean must be 64x64")
        if self.components.shape != (self.rank, 64 * 64):
            raise SourceOffPredictorFeasibilityError("low-rank basis shape mismatch")
        if self.latent_intercept.shape != (self.rank,) or self.latent_coefficient.shape != (self.rank,):
            raise SourceOffPredictorFeasibilityError("low-rank latent AR state mismatch")
        arrays = (
            self.mean_image,
            self.components,
            self.latent_intercept,
            self.latent_coefficient,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise SourceOffPredictorFeasibilityError("low-rank state must be finite")


def fit_low_rank_ar1(
    training_clips: np.ndarray,
    *,
    rank: int = LOW_RANK_RANK,
    sample_frames: int = LOW_RANK_SAMPLE_FRAMES,
    random_seed: int = LOW_RANK_RANDOM_SEED,
    chunk_size: int = 16,
) -> LowRankAR1State:
    """Fit a train-only rank-r spatial background plus diagonal latent AR(1)."""

    clips = _finite_fit_bank(training_clips, name="training_clips")
    resolved_rank = int(rank)
    if not 1 <= resolved_rank <= 64:
        raise SourceOffPredictorFeasibilityError("low-rank rank must be in [1, 64]")
    total_frames = int(clips.shape[0] * clips.shape[1])
    sample_count = min(int(sample_frames), total_frames)
    if sample_count < resolved_rank + 2:
        raise SourceOffPredictorFeasibilityError("too few frames for low-rank fit")
    flat_indices = np.linspace(0, total_frames - 1, sample_count, dtype=np.int64)
    clip_indices, frame_indices = np.divmod(flat_indices, clips.shape[1])
    samples = np.asarray(clips[clip_indices, frame_indices], dtype=np.float32).reshape(
        sample_count, -1
    )
    mean_flat = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = samples - mean_flat[None]
    from sklearn.utils.extmath import randomized_svd

    _, _, components = randomized_svd(
        centered,
        n_components=resolved_rank,
        n_iter=4,
        random_state=int(random_seed),
        flip_sign=True,
    )
    components = np.asarray(components, dtype=np.float32)

    total_x = np.zeros(resolved_rank, dtype=np.float64)
    total_y = np.zeros(resolved_rank, dtype=np.float64)
    total_xx = np.zeros(resolved_rank, dtype=np.float64)
    total_xy = np.zeros(resolved_rank, dtype=np.float64)
    count = 0
    for start in range(0, clips.shape[0], int(chunk_size)):
        batch = np.asarray(clips[start : start + int(chunk_size)], dtype=np.float32)
        flattened = batch.reshape(-1, 64 * 64) - mean_flat[None]
        latent = (flattened @ components.T).reshape(batch.shape[0], batch.shape[1], resolved_rank)
        previous = latent[:, :-1].astype(np.float64, copy=False)
        current = latent[:, 1:].astype(np.float64, copy=False)
        total_x += previous.sum(axis=(0, 1))
        total_y += current.sum(axis=(0, 1))
        total_xx += np.square(previous).sum(axis=(0, 1))
        total_xy += (previous * current).sum(axis=(0, 1))
        count += int(previous.shape[0] * previous.shape[1])
    mean_x = total_x / count
    mean_y = total_y / count
    denominator = total_xx - total_x * total_x / count
    numerator = total_xy - total_x * total_y / count
    coefficient = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )
    coefficient = np.clip(coefficient, -0.999, 0.999)
    intercept = mean_y - coefficient * mean_x
    state = LowRankAR1State(
        mean_image=mean_flat.reshape(64, 64),
        components=components,
        latent_intercept=intercept.astype(np.float32),
        latent_coefficient=coefficient.astype(np.float32),
        rank=resolved_rank,
        sampled_flat_frame_indices=flat_indices,
        observation_count_per_latent=count,
    )
    state.validate()
    return state


def low_rank_ar1_prediction(movie: np.ndarray, state: LowRankAR1State) -> np.ndarray:
    values = _finite_tyx(movie, name="movie")
    state.validate()
    prediction = np.empty_like(values)
    prediction[0] = values[0]
    previous = values[:-1].reshape(values.shape[0] - 1, -1) - state.mean_image.reshape(1, -1)
    latent = previous @ state.components.T
    forecast = state.latent_intercept[None] + state.latent_coefficient[None] * latent
    prediction[1:] = (
        state.mean_image.reshape(1, -1) + forecast @ state.components
    ).reshape(values.shape[0] - 1, 64, 64)
    return prediction


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    if not 1 <= int(lag) < centered.shape[0]:
        raise SourceOffPredictorFeasibilityError("invalid autocorrelation lag")
    first = centered[int(lag) :].reshape(-1)
    second = centered[: -int(lag)].reshape(-1)
    denominator = math.sqrt(float(np.dot(first, first) * np.dot(second, second)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _temporal_spectrum(values: np.ndarray) -> dict[str, float]:
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(centered, axis=0)) ** 2, axis=(1, 2))[1:]
    epsilon = max(float(np.mean(power)) * 1e-12, np.finfo(float).tiny)
    total = float(np.sum(power))
    low_bins = max(1, len(power) // 4)
    flatness = float(np.exp(np.mean(np.log(power + epsilon))) / np.mean(power + epsilon))
    return {
        "temporal_spectral_flatness": flatness,
        "temporal_low_frequency_power_fraction": float(np.sum(power[:low_bins]) / max(total, epsilon)),
        "temporal_power_bin_count_excluding_dc": int(len(power)),
        "temporal_low_frequency_bin_count": int(low_bins),
    }


def _best_prediction_lag(observed: np.ndarray, predicted: np.ndarray, *, maximum_lag: int = 4) -> dict[str, Any]:
    observed_centered = np.asarray(observed, dtype=np.float64)
    predicted_centered = np.asarray(predicted, dtype=np.float64)
    observed_centered -= observed_centered.mean(axis=0, keepdims=True)
    predicted_centered -= predicted_centered.mean(axis=0, keepdims=True)
    if float(np.sum(predicted_centered * predicted_centered)) <= np.finfo(float).eps:
        return {
            "prediction_best_lag_frames": None,
            "prediction_best_lag_correlation": None,
            "prediction_lag_convention": "positive means prediction trails observation",
        }
    rows: list[tuple[int, float]] = []
    for lag in range(-int(maximum_lag), int(maximum_lag) + 1):
        if lag > 0:
            pred = predicted_centered[lag:]
            obs = observed_centered[:-lag]
        elif lag < 0:
            pred = predicted_centered[:lag]
            obs = observed_centered[-lag:]
        else:
            pred = predicted_centered
            obs = observed_centered
        first = pred.reshape(-1)
        second = obs.reshape(-1)
        denominator = math.sqrt(float(np.dot(first, first) * np.dot(second, second)))
        correlation = 0.0 if denominator <= np.finfo(float).eps else float(np.dot(first, second) / denominator)
        rows.append((lag, correlation))
    best = max(rows, key=lambda row: (row[1], -abs(row[0]), -row[0]))
    return {
        "prediction_best_lag_frames": int(best[0]),
        "prediction_best_lag_correlation": float(best[1]),
        "prediction_lag_convention": "positive means prediction trails observation",
    }


def _seam_metrics(values: np.ndarray, *, patch_yx: Sequence[int] = PATCH_YX) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or len(patch_yx) != 2:
        raise SourceOffPredictorFeasibilityError("seam metrics require TYX and patch YX")
    patch_y, patch_x = (int(value) for value in patch_yx)
    vertical = np.abs(np.diff(array, axis=1))
    horizontal = np.abs(np.diff(array, axis=2))
    vertical_boundary = np.arange(1, array.shape[1]) % patch_y == 0
    horizontal_boundary = np.arange(1, array.shape[2]) % patch_x == 0
    seam = np.concatenate(
        (
            vertical[:, vertical_boundary, :].reshape(-1),
            horizontal[:, :, horizontal_boundary].reshape(-1),
        )
    )
    interior = np.concatenate(
        (
            vertical[:, ~vertical_boundary, :].reshape(-1),
            horizontal[:, :, ~horizontal_boundary].reshape(-1),
        )
    )
    seam_mean = float(np.mean(seam))
    interior_mean = float(np.mean(interior))
    return {
        "mean_absolute_spatial_seam_jump": seam_mean,
        "mean_absolute_spatial_interior_jump": interior_mean,
        "seam_to_interior_jump_ratio": seam_mean / max(interior_mean, np.finfo(float).eps),
    }


def source_off_metrics(
    observed: np.ndarray,
    predicted_background: np.ndarray,
    *,
    evaluation_start_frame_zero: int = COMMON_EVALUATION_START_FRAME_ZERO,
) -> dict[str, Any]:
    """Measure residual scale, dynamics, spectrum, seams, and prediction lag."""

    observed_values = _finite_tyx(observed, name="observed")
    predicted_values = _finite_tyx(predicted_background, name="predicted_background")
    start = int(evaluation_start_frame_zero)
    if not 1 <= start <= observed_values.shape[0] - 8:
        raise SourceOffPredictorFeasibilityError("evaluation start leaves too few frames")
    observed_eval = observed_values[start:].astype(np.float64)
    predicted_eval = predicted_values[start:].astype(np.float64)
    residual = observed_eval - predicted_eval
    centered_input = observed_eval - np.median(observed_eval, axis=0, keepdims=True)
    centered_residual = residual - np.median(residual, axis=0, keepdims=True)
    input_rms = float(np.sqrt(np.mean(np.square(centered_input))))
    residual_rms = float(np.sqrt(np.mean(np.square(centered_residual))))
    input_mad = _normal_consistent_mad(centered_input)
    residual_mad = _normal_consistent_mad(centered_residual)
    input_dynamic_mad = _normal_consistent_mad(np.diff(observed_eval, axis=0))
    residual_dynamic_mad = _normal_consistent_mad(np.diff(residual, axis=0))
    epsilon = np.finfo(float).eps
    autocorrelations = [_autocorrelation(residual, lag) for lag in range(1, 5)]
    return {
        "evaluation_start_frame_zero": start,
        "evaluation_stop_frame_zero_exclusive": int(observed_values.shape[0]),
        "evaluation_frame_count": int(observed_values.shape[0] - start),
        "source_off_centering": "per_pixel_temporal_median_on_common_evaluation_support",
        "centered_input_rms": input_rms,
        "centered_residual_rms": residual_rms,
        "centered_rms_ratio": residual_rms / max(input_rms, epsilon),
        "centered_input_mad_normal_consistent": input_mad,
        "centered_residual_mad_normal_consistent": residual_mad,
        "centered_mad_ratio": residual_mad / max(input_mad, epsilon),
        "input_dynamic_mad_normal_consistent": input_dynamic_mad,
        "residual_dynamic_mad_normal_consistent": residual_dynamic_mad,
        "dynamic_mad_ratio": residual_dynamic_mad / max(input_dynamic_mad, epsilon),
        **_temporal_spectrum(residual),
        "residual_lag1_autocorrelation": autocorrelations[0],
        "residual_absolute_lag1_autocorrelation": abs(autocorrelations[0]),
        "residual_portmanteau_mean_squared_autocorrelation_lags_1_4": float(
            np.mean(np.square(autocorrelations))
        ),
        "residual_seam": _seam_metrics(residual),
        "predicted_background_seam": _seam_metrics(predicted_eval),
        **_best_prediction_lag(observed_eval, predicted_eval),
    }


def _flatten_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(row)
    residual_seam = values.pop("residual_seam")
    prediction_seam = values.pop("predicted_background_seam")
    for key, value in residual_seam.items():
        values[f"residual_{key}"] = value
    for key, value in prediction_seam.items():
        values[f"prediction_{key}"] = value
    return values


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect="excel-tab", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect="excel", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _verify_indexed_root(root: Path, *, expected_index_sha256: str) -> dict[str, Any]:
    index_path = root / "artifact_index.json"
    observed_index = sha256_file(index_path)
    if observed_index != expected_index_sha256:
        raise SourceOffPredictorFeasibilityError(
            f"artifact index mismatch for {root.name}: {observed_index}"
        )
    index = _json(index_path)
    rows = index.get("artifacts")
    if not isinstance(rows, list) or len(rows) != int(index.get("artifact_count", -1)):
        raise SourceOffPredictorFeasibilityError("parent artifact index count mismatch")
    checked: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise SourceOffPredictorFeasibilityError("parent artifact row must be an object")
        relative = Path(str(raw["path"]))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise SourceOffPredictorFeasibilityError("parent artifact path escapes root") from exc
        observed_sha = sha256_file(candidate)
        observed_bytes = candidate.stat().st_size
        if observed_sha != str(raw["sha256"]) or observed_bytes != int(raw["bytes"]):
            raise SourceOffPredictorFeasibilityError(f"parent artifact drift: {relative}")
        checked.append({"path": relative.as_posix(), "sha256": observed_sha, "bytes": observed_bytes})
    snapshot = stable_hash(checked)
    return {
        "root_role": root.name,
        "artifact_index_sha256": observed_index,
        "artifact_count": len(checked),
        "snapshot_sha256": snapshot,
        "all_indexed_artifacts_verified": True,
    }


def _verify_implementation_descriptor(repository_root: Path) -> dict[str, Any]:
    path = (repository_root / DESCRIPTOR_PATH).resolve()
    payload = _json(path)
    checks = {
        "experiment_id": payload.get("experiment_id") == EXPERIMENT_ID,
        "run_id": payload.get("run_id") == RUN_ID,
        "planned_at_utc": payload.get("planned_at_utc") == PLANNED_AT_UTC,
        "output_root": payload.get("output_root") == DEFAULT_OUTPUT_ROOT,
        "runner_module": payload.get("runner_module")
        == "neurobench.experiments.neuron_identifiability.source_off_predictor_feasibility",
    }
    contract = payload.get("implementation_hash_contract")
    if not isinstance(contract, Mapping) or contract.get("algorithm") != "sha256":
        raise SourceOffPredictorFeasibilityError("implementation hash contract is invalid")
    sources = contract.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise SourceOffPredictorFeasibilityError("implementation sources are incomplete")
    rows: list[dict[str, Any]] = []
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise SourceOffPredictorFeasibilityError("implementation source row must be an object")
        relative = Path(str(raw["path"]))
        candidate = (repository_root / relative).resolve()
        try:
            candidate.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise SourceOffPredictorFeasibilityError("implementation source escapes repository") from exc
        expected = str(raw["sha256"])
        observed = sha256_file(candidate)
        rows.append(
            {
                "path": relative.as_posix(),
                "role": str(raw.get("role", "unspecified")),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": observed == expected,
            }
        )
    checks["all_source_hashes_match"] = all(row["matches"] for row in rows)
    if not all(checks.values()):
        raise SourceOffPredictorFeasibilityError(
            "canonical descriptor or implementation hash mismatch: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    return {
        "path": DESCRIPTOR_PATH,
        "sha256": sha256_file(path),
        "checks": checks,
        "sources": rows,
    }


def _load_clip_banks(parent_root: Path) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    train_path = parent_root / "cache/train_seed_1001_normalized_clips.npy"
    validation_path = parent_root / "cache/validation_seed_2001_normalized_clips.npy"
    checks = {
        "train_cache_sha256": sha256_file(train_path),
        "validation_cache_sha256": sha256_file(validation_path),
        "clip_bank_manifest_sha256": sha256_file(parent_root / "clip_bank_manifest.json"),
    }
    expected = {
        "train_cache_sha256": TRAIN_CACHE_SHA256,
        "validation_cache_sha256": VALIDATION_CACHE_SHA256,
        "clip_bank_manifest_sha256": CLIP_BANK_MANIFEST_SHA256,
    }
    if checks != expected:
        raise SourceOffPredictorFeasibilityError(f"frozen clip-bank hashes drifted: {checks}")
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    _finite_bank(train, TRAIN_SHAPE, name="training_cache")
    _finite_bank(validation, VALIDATION_SHAPE, name="validation_cache")
    return train, validation, {**checks, "train_shape": list(train.shape), "validation_shape": list(validation.shape)}


def _validation_metadata(parent_root: Path) -> list[dict[str, Any]]:
    manifest = _json(parent_root / "clip_bank_manifest.json")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(validation.get("requests"), list):
        raise SourceOffPredictorFeasibilityError("clip-bank manifest lacks validation requests")
    rows = [dict(row) for row in validation["requests"]]
    if len(rows) != VALIDATION_SHAPE[0]:
        raise SourceOffPredictorFeasibilityError("validation request count drifted")
    return rows


def _load_source_off_windows(
    repository_root: Path,
    data_root: Path,
    parent_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = parent_root / "background_window_manifest.json"
    if sha256_file(manifest_path) != BACKGROUND_WINDOW_MANIFEST_SHA256:
        raise SourceOffPredictorFeasibilityError("background-window manifest hash drifted")
    manifest = _json(manifest_path)
    rows = manifest.get("windows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise SourceOffPredictorFeasibilityError("expected exactly 12 source-off windows")
    descriptor_path = repository_root / "research/data-registry/jepa_raw_video_sources_v1.json"
    if sha256_file(descriptor_path) != DATA_DESCRIPTOR_SHA256:
        raise SourceOffPredictorFeasibilityError("data descriptor hash drifted")
    inventory = load_recording_inventory(
        descriptor_path,
        repository_root=repository_root,
        data_root=data_root,
        verify_live=False,
        verify_hashes=False,
    )
    lookup = inventory.by_id()
    pilot_config = {"data_inventory": "research/data-registry/jepa_raw_video_sources_v1.json"}
    holdout = inventory_source_holdout(pilot_config, repository_root, data_root)
    lookup[holdout.recording_id] = holdout
    selected_ids = sorted({str(row["recording_id"]) for row in rows})
    source_hashes: list[dict[str, Any]] = []
    for recording_id in selected_ids:
        descriptor = lookup.get(recording_id)
        if descriptor is None:
            raise SourceOffPredictorFeasibilityError(f"source-off recording unavailable: {recording_id}")
        observed = sha256_file(descriptor.resolved_path)
        if observed != descriptor.sha256:
            raise SourceOffPredictorFeasibilityError(f"source hash mismatch: {recording_id}")
        source_hashes.append(
            {
                "recording_id": recording_id,
                "uri": descriptor.uri,
                "sha256": observed,
                "shape_tyx": list(descriptor.shape),
                "dtype": descriptor.dtype,
            }
        )
    contract = TemporalClipContract(clip_frames=32, guard_frames=16, frame_step=1)
    windows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        descriptor = lookup[str(row["recording_id"])]
        request = ClipRequest(
            recording_id=str(row["recording_id"]),
            split=str(row["split"]),
            start_frame_zero=int(row["start_frame_zero"]),
            stop_frame_zero_exclusive=int(row["stop_frame_zero_exclusive"]),
            y_zero=int(row["y_zero"]),
            x_zero=int(row["x_zero"]),
            height=int(row["height"]),
            width=int(row["width"]),
            sampling_stratum=str(row["sampling_stratum"]),
            temporal_mad=float(row["temporal_mad"]),
        )
        raw_movie = np.asarray(read_sequential_clip(descriptor, request, contract), dtype=np.float32)
        normalized = ((raw_movie - np.float32(NORMALIZATION_CENTER)) / np.float32(NORMALIZATION_SCALE)).astype(
            np.float32,
            copy=False,
        )
        _finite_tyx(normalized, name=str(row["background_window_id"]))
        windows.append({"metadata": row, "movie": normalized})
    return windows, {
        "background_window_manifest_sha256": BACKGROUND_WINDOW_MANIFEST_SHA256,
        "window_count": len(windows),
        "source_recording_count": len(source_hashes),
        "source_hashes": source_hashes,
        "normalization": {
            "center": NORMALIZATION_CENTER,
            "scale": NORMALIZATION_SCALE,
            "normalization_sha256": NORMALIZATION_SHA256,
            "application_count": 1,
        },
        "raw_video_copied": False,
    }


@dataclass
class FrozenRunBDecoders:
    jepa: FrozenLatentPixelBackgroundModel
    random: FrozenLatentPixelBackgroundModel
    manifest: dict[str, Any]

    @classmethod
    def load(cls, parent_root: Path, run_b_root: Path, *, device: str) -> "FrozenRunBDecoders":
        upstream = parent_root / "checkpoints/seed_1001_final_non_scientific.pt"
        decoder_path = run_b_root / "checkpoints/decoder_seed_6201_final_non_scientific.pt"
        if sha256_file(upstream) != JEPA_CHECKPOINT_SHA256:
            raise SourceOffPredictorFeasibilityError("EXP-0028 JEPA checkpoint hash drifted")
        if sha256_file(decoder_path) != RUN_B_DECODER_CHECKPOINT_SHA256:
            raise SourceOffPredictorFeasibilityError("Run-B decoder checkpoint hash drifted")
        jepa_provider = load_frozen_jepa_checkpoint(
            upstream, expected_sha256=JEPA_CHECKPOINT_SHA256, map_location="cpu"
        )
        random_provider = load_frozen_random_checkpoint(
            upstream, expected_sha256=JEPA_CHECKPOINT_SHA256, map_location="cpu"
        )
        checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("scientific_checkpoint") is not False:
            raise SourceOffPredictorFeasibilityError("Run-B decoder checkpoint contract failed")
        jepa = FrozenLatentPixelBackgroundModel(
            jepa_provider.provider, decoder_initialization_seed=6201
        )
        random_model = FrozenLatentPixelBackgroundModel(
            random_provider.provider, decoder_initialization_seed=6201
        )
        jepa.decoder.load_state_dict(dict(checkpoint["decoder_state_dict"]), strict=True)
        random_model.decoder.load_state_dict(
            dict(checkpoint["random_control_decoder_state_dict"]), strict=True
        )
        jepa.requires_grad_(False).eval().to(device)
        random_model.requires_grad_(False).eval().to(device)
        observed_jepa = module_state_sha256(jepa.decoder)
        observed_random = module_state_sha256(random_model.decoder)
        expected_identity = checkpoint.get("decoder_state_identity", {})
        if observed_jepa != expected_identity.get("jepa_arm_runtime_module_state_sha256"):
            raise SourceOffPredictorFeasibilityError("JEPA decoder tensor identity drifted")
        if observed_random != expected_identity.get("random_arm_runtime_module_state_sha256"):
            raise SourceOffPredictorFeasibilityError("random decoder tensor identity drifted")
        manifest = {
            "upstream_checkpoint_sha256": JEPA_CHECKPOINT_SHA256,
            "decoder_checkpoint_sha256": RUN_B_DECODER_CHECKPOINT_SHA256,
            "scientific_checkpoint": False,
            "decoder_steps": int(checkpoint["decoder_steps"]),
            "decoder_seed": int(checkpoint["decoder_seed"]),
            "jepa_decoder_state_sha256": observed_jepa,
            "random_decoder_state_sha256": observed_random,
            "jepa_provider_state_sha256": module_state_sha256(jepa.provider),
            "random_provider_state_sha256": module_state_sha256(random_model.provider),
            "all_parameters_frozen": all(
                not parameter.requires_grad
                for model in (jepa, random_model)
                for parameter in model.parameters()
            ),
            "device": str(device),
            "selection": "immutable_Run-B_final_step_no_refit_no_validation_selection",
        }
        return cls(jepa=jepa, random=random_model, manifest=manifest)

    def predict(self, method: str, movie: np.ndarray, *, device: str) -> tuple[np.ndarray, dict[str, Any]]:
        model = self.jepa if method == "frozen_jepa_decoder_run_b" else self.random
        writable = np.array(_finite_tyx(movie, name="movie"), dtype=np.float32, copy=True, order="C")
        tensor = torch.from_numpy(writable).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = predict_full_background(model, tensor, target_batch_size=64, amp=False)
        prediction = output.background[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        return prediction, {
            "coverage_minimum": output.minimum_coverage,
            "coverage_maximum": output.maximum_coverage,
            "coverage_exactly_once": output.minimum_coverage == output.maximum_coverage == 1,
            "finite": bool(np.isfinite(prediction).all()),
            "shape_tyx": list(prediction.shape),
        }


def _simple_prediction(
    method: str,
    movie: np.ndarray,
    *,
    ar1: PerPixelAR1State,
    low_rank: LowRankAR1State,
) -> np.ndarray:
    if method == "no_subtraction_identity":
        return no_subtraction_prediction(movie)
    if method == "causal_zero_order_hold":
        return causal_zero_order_hold_prediction(movie)
    if method == "causal_ema_alpha_0p25":
        return causal_ema_prediction(movie)
    if method == "causal_temporal_median_w7":
        return causal_temporal_median_prediction(movie)
    if method == "per_pixel_ar1_train512":
        return per_pixel_ar1_prediction(movie, ar1)
    if method == "low_rank_ar1_rank8_train512":
        return low_rank_ar1_prediction(movie, low_rank)
    raise SourceOffPredictorFeasibilityError(f"unknown simple method: {method}")


AGGREGATE_METRICS = (
    "centered_rms_ratio",
    "centered_mad_ratio",
    "dynamic_mad_ratio",
    "temporal_spectral_flatness",
    "temporal_low_frequency_power_fraction",
    "residual_absolute_lag1_autocorrelation",
    "residual_portmanteau_mean_squared_autocorrelation_lags_1_4",
    "residual_seam_to_interior_jump_ratio",
    "prediction_seam_to_interior_jump_ratio",
    "prediction_best_lag_frames",
    "inference_seconds",
)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        summary: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        summary["clip_count"] = len(members)
        for metric in AGGREGATE_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in members if row.get(metric) is not None],
                dtype=np.float64,
            )
            if values.size:
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_q1"] = float(np.quantile(values, 0.25))
                summary[f"{metric}_q3"] = float(np.quantile(values, 0.75))
                summary[f"{metric}_mean"] = float(np.mean(values))
            else:
                summary[f"{metric}_median"] = None
                summary[f"{metric}_q1"] = None
                summary[f"{metric}_q3"] = None
                summary[f"{metric}_mean"] = None
        output.append(summary)
    return output


def evaluate_feasibility_gate(
    aggregate_rows: Sequence[Mapping[str, Any]],
    recording_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply a predeclared source-off-only safety/feasibility gate."""

    aggregate_lookup = {
        (str(row["dataset_partition"]), str(row["method_id"])): row
        for row in aggregate_rows
    }
    baseline = {
        partition: aggregate_lookup[(partition, "no_subtraction_identity")]
        for partition in ("held_validation", "registered_source_off_windows")
    }
    methods: dict[str, Any] = {}
    for method in METHODS:
        checks: dict[str, bool] = {}
        for partition in ("held_validation", "registered_source_off_windows"):
            row = aggregate_lookup[(partition, method)]
            reference = baseline[partition]
            prefix = "held" if partition == "held_validation" else "window"
            checks[f"{prefix}_median_centered_rms_ratio"] = (
                float(row["centered_rms_ratio_median"])
                <= GATE_THRESHOLDS["median_centered_rms_ratio_max"]
            )
            checks[f"{prefix}_median_dynamic_mad_ratio"] = (
                float(row["dynamic_mad_ratio_median"])
                <= GATE_THRESHOLDS["median_dynamic_mad_ratio_max"]
            )
            checks[f"{prefix}_median_residual_seam_ratio"] = (
                float(row["residual_seam_to_interior_jump_ratio_median"])
                <= GATE_THRESHOLDS["median_residual_seam_ratio_max"]
            )
            checks[f"{prefix}_spectral_flatness_noninferior_to_no_subtraction"] = (
                float(row["temporal_spectral_flatness_median"])
                - float(reference["temporal_spectral_flatness_median"])
                >= GATE_THRESHOLDS["spectral_flatness_change_vs_no_subtraction_min"]
            )
            checks[f"{prefix}_absolute_lag1_nonworse_than_no_subtraction"] = (
                float(row["residual_absolute_lag1_autocorrelation_median"])
                - float(reference["residual_absolute_lag1_autocorrelation_median"])
                <= GATE_THRESHOLDS[
                    "absolute_lag1_autocorrelation_change_vs_no_subtraction_max"
                ]
            )
        per_recording = [
            row
            for row in recording_rows
            if str(row["method_id"]) == method
        ]
        checks["every_recording_centered_rms_nonamplifying"] = bool(per_recording) and all(
            float(row["centered_rms_ratio_median"])
            <= GATE_THRESHOLDS["every_recording_centered_rms_ratio_max"]
            for row in per_recording
        )
        methods[method] = {
            "checks": checks,
            "passed": all(checks.values()),
            "passed_check_count": sum(checks.values()),
            "required_check_count": len(checks),
        }
    eligible = [method for method in METHODS if method != "no_subtraction_identity" and methods[method]["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": "source_off_background_prediction_feasibility_v1",
        "truth_access": "none_no_injections_no_ROIs_no_detector_outcomes",
        "selection_data": "none_gate_thresholds_are_constants_in_the_versioned_runner",
        "thresholds": GATE_THRESHOLDS,
        "methods": methods,
        "eligible_predictor_methods": eligible,
        "any_predictor_passed": bool(eligible),
        "consequence": (
            "eligible_only_for_a_separately_versioned_residual_detector_safety_screen"
            if eligible
            else "do_not_launch_a_new_residual_detector_experiment_from_these_predictors"
        ),
        "scientific_claim_consequence": "none",
    }


def _predictor_state_manifest(ar1: PerPixelAR1State, low_rank: LowRankAR1State) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fit_scope": "512_frozen_normalized_training_clips_only",
        "labels_or_source_truth_used": False,
        "validation_used_for_fit_or_selection": False,
        "per_pixel_ar1": {
            "model": "affine_AR1_per_relative_64x64_crop_pixel",
            "coefficient_clip": [-0.999, 0.999],
            "observation_count_per_pixel": ar1.observation_count_per_pixel,
            "coefficient_summary": {
                "minimum": float(ar1.coefficient.min()),
                "median": float(np.median(ar1.coefficient)),
                "maximum": float(ar1.coefficient.max()),
            },
        },
        "low_rank_ar1": {
            "model": "rank8_train_spatial_basis_plus_diagonal_latent_AR1",
            "rank": low_rank.rank,
            "basis_fit_sample_frames": int(low_rank.sampled_flat_frame_indices.size),
            "basis_sampling": "deterministic_even_spacing_over_all_512x32_train_frames",
            "randomized_svd_seed": LOW_RANK_RANDOM_SEED,
            "randomized_svd_power_iterations": 4,
            "latent_observation_count": low_rank.observation_count_per_latent,
            "coefficient_clip": [-0.999, 0.999],
            "latent_coefficient_summary": {
                "minimum": float(low_rank.latent_coefficient.min()),
                "median": float(np.median(low_rank.latent_coefficient)),
                "maximum": float(low_rank.latent_coefficient.max()),
            },
        },
        "fixed_predictors": {
            "EMA_alpha": EMA_ALPHA,
            "causal_median_window_frames": CAUSAL_MEDIAN_WINDOW,
            "common_evaluation_start_frame_zero": COMMON_EVALUATION_START_FRAME_ZERO,
        },
    }


def _save_predictor_state(path: Path, ar1: PerPixelAR1State, low_rank: LowRankAR1State) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        ar1_intercept=ar1.intercept,
        ar1_coefficient=ar1.coefficient,
        low_rank_mean_image=low_rank.mean_image,
        low_rank_components=low_rank.components,
        low_rank_latent_intercept=low_rank.latent_intercept,
        low_rank_latent_coefficient=low_rank.latent_coefficient,
        low_rank_sampled_flat_frame_indices=low_rank.sampled_flat_frame_indices,
    )
    temporary.replace(path)
    return sha256_file(path)


def _surrogate_pixel(movie: np.ndarray) -> tuple[int, int, float]:
    dynamics = np.diff(_finite_tyx(movie, name="movie"), axis=0)
    score = np.median(np.abs(dynamics - np.median(dynamics, axis=0, keepdims=True)), axis=0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(y), int(x), float(score[y, x])


def _scale_frame(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((np.asarray(values) - low) / (high - low), 0.0, 1.0).astype(np.float32) * 255.0


def _panel_image(
    values: np.ndarray,
    *,
    title: str,
    low: float,
    high: float,
    marker_yx: tuple[int, int],
    scale: int,
) -> Any:
    from PIL import Image, ImageDraw

    grayscale = _scale_frame(values, low, high).astype(np.uint8)
    image = Image.fromarray(grayscale, mode="L").convert("RGB").resize(
        (grayscale.shape[1] * scale, grayscale.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    header = 24
    panel = Image.new("RGB", (image.width, image.height + header), (18, 22, 28))
    panel.paste(image, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(238, 242, 246))
    y, x = marker_yx
    cx = x * scale + scale // 2
    cy = header + y * scale + scale // 2
    radius = max(4, scale * 2)
    orange = (255, 145, 36)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=orange, width=max(2, scale // 2))
    return panel


def _write_video(path: Path, frames: Iterable[np.ndarray], *, fps: int, size: tuple[int, int]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    count = 0
    assert process.stdin is not None
    try:
        for frame in frames:
            array = np.asarray(frame, dtype=np.uint8)
            if array.shape != (height, width, 3):
                raise SourceOffPredictorFeasibilityError(
                    f"video frame shape mismatch: expected {(height, width, 3)}, observed {array.shape}"
                )
            process.stdin.write(np.ascontiguousarray(array).tobytes())
            count += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
    except Exception:
        process.kill()
        raise
    if returncode != 0:
        raise SourceOffPredictorFeasibilityError(f"ffmpeg failed for {path.name}: {stderr}")
    return {"path": path.as_posix(), "frame_count": count, "fps": fps, "width": width, "height": height}


def _video_probe(path: Path, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    stream = payload["streams"][0]
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    frames = int(stream["nb_read_frames"])
    result = {
        "path": path.as_posix(),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pixel_format": str(stream["pix_fmt"]),
        "decoded_frame_count": frames,
        "expected_frame_count": int(expected_frames),
        "frame_count_exact": frames == int(expected_frames),
        "full_decode_passed": decode.returncode == 0,
        "decode_stderr": decode.stderr.strip(),
    }
    if not result["frame_count_exact"] or not result["full_decode_passed"]:
        raise SourceOffPredictorFeasibilityError(f"video validation failed: {path}")
    return result


def _render_audit_media(
    root: Path,
    source_windows: Sequence[Mapping[str, Any]],
    source_predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    expert = root / "1_Expert_Annotations"
    model = root / "2_Model_Annotations"
    comparison = root / "3_Comparison"
    atomic_text(
        expert / "README.md",
        "# Expert annotations: not applicable\n\n"
        "This source-off benchmark does not use human ROI labels, injected-source truth, or detector outcomes. "
        "The Expert section is therefore explicitly not applicable.\n",
    )
    atomic_text(
        model / "README.md",
        "# Frozen label-free predictor audit\n\n"
        "Orange markers identify one method-blind candidate-surrogate pixel per registered source-off window, "
        "selected before predictor comparison as the maximum raw temporal-difference MAD pixel. Videos show "
        "the normalized observation and all eight signed residuals on fixed within-window scales.\n",
    )
    atomic_text(
        comparison / "README.md",
        "# Source-off predictor comparison\n\n"
        "This section contains figures and tables only. It compares residual scale, dynamics, whiteness, seams, "
        "lag, compute, and per-recording consistency without source truth or human labels.\n",
    )
    occurrence_rows: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, item in enumerate(source_windows, start=1):
        metadata = dict(item["metadata"])
        window_id = str(metadata["background_window_id"])
        observed = np.asarray(item["movie"], dtype=np.float32)
        predictions = source_predictions[window_id]
        residuals = {method: observed - predictions[method] for method in METHODS}
        y, x, score = _surrogate_pixel(observed)
        roi_id = f"source_off_{index:02d}"
        occurrence = {
            "roi_id": roi_id,
            "background_window_id": window_id,
            "recording_id": metadata["recording_id"],
            "x_px": x,
            "y_px": y,
            "selection_score_raw_temporal_difference_mad": score,
            "selection_policy": "method_blind_argmax_raw_temporal_difference_MAD_per_window",
            "expert_label": "not_applicable",
            "native_activity_identity": "unknown",
        }
        occurrence_rows.append(occurrence)
        scales: dict[str, tuple[float, float]] = {
            "observed": (
                float(np.quantile(observed, 0.01)),
                float(np.quantile(observed, 0.99)),
            )
        }
        for method, residual in residuals.items():
            prediction = predictions[method]
            scales[f"prediction::{method}"] = (
                float(np.quantile(prediction, 0.01)),
                float(np.quantile(prediction, 0.99)),
            )
            limit = max(float(np.quantile(np.abs(residual), 0.99)), 1e-6)
            scales[f"residual::{method}"] = (-limit, limit)

        panel_order = [("observed", "Observed normalized")]
        for method in METHODS:
            panel_order.extend(
                (
                    (f"prediction::{method}", f"Pred: {DISPLAY_NAMES[method]}"),
                    (f"residual::{method}", f"Resid: {DISPLAY_NAMES[method]}"),
                )
            )

        def panel_values(key: str, frame_index: int) -> np.ndarray:
            if key == "observed":
                return observed[frame_index]
            role, method = key.split("::", 1)
            return predictions[method][frame_index] if role == "prediction" else residuals[method][frame_index]

        def full_frames() -> Iterable[np.ndarray]:
            for frame_index in range(observed.shape[0]):
                panels = []
                for key, title in panel_order:
                    values = panel_values(key, frame_index)
                    low, high = scales[key]
                    panels.append(
                        _panel_image(values, title=title, low=low, high=high, marker_yx=(y, x), scale=3)
                    )
                canvas = Image.new("RGB", (960, 864), (10, 13, 18))
                for panel_index, panel in enumerate(panels):
                    canvas.paste(panel, ((panel_index % 5) * 192, (panel_index // 5) * 216))
                yield np.asarray(canvas)

        full_path = model / "videos" / f"{roi_id}__full_field_predictor_grid.mp4"
        _write_video(full_path, full_frames(), fps=10, size=(960, 864))
        video_paths.append(full_path)

        pad = 8
        padded_sources = {
            "observed": np.pad(observed, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        }
        for method in METHODS:
            padded_sources[f"prediction::{method}"] = np.pad(
                predictions[method], ((0, 0), (pad, pad), (pad, pad)), mode="reflect"
            )
            padded_sources[f"residual::{method}"] = np.pad(
                residuals[method], ((0, 0), (pad, pad), (pad, pad)), mode="reflect"
            )
        crop_y = y + pad
        crop_x = x + pad

        def closeup_frames() -> Iterable[np.ndarray]:
            for frame_index in range(observed.shape[0]):
                panels = []
                for key, title in panel_order:
                    source = padded_sources[key]
                    crop = source[frame_index, crop_y - 8 : crop_y + 9, crop_x - 8 : crop_x + 9]
                    low, high = scales[key]
                    panels.append(
                        _panel_image(crop, title=title, low=low, high=high, marker_yx=(8, 8), scale=8)
                    )
                canvas = Image.new("RGB", (680, 640), (10, 13, 18))
                for panel_index, panel in enumerate(panels):
                    canvas.paste(panel, ((panel_index % 5) * 136, (panel_index // 5) * 160))
                yield np.asarray(canvas)

        close_path = model / "videos" / "closeups" / f"model_roi_{roi_id}.mp4"
        _write_video(close_path, closeup_frames(), fps=10, size=(680, 640))
        video_paths.append(close_path)

        figure, axes = plt.subplots(9, 1, figsize=(11, 15), sharex=True, constrained_layout=True)
        axes[0].plot(observed[:, y, x], color="#202830", linewidth=1.5)
        axes[0].set_title(f"{window_id}: exact pixel x={x}, y={y}")
        axes[0].set_ylabel("Observed")
        for axis, method in zip(axes[1:], METHODS):
            axis.plot(observed[:, y, x], color="#79838f", linewidth=0.8, alpha=0.65, label="observed")
            axis.plot(predictions[method][:, y, x], color="#e89a2f", linewidth=1.0, linestyle="--", label="prediction")
            axis.plot(residuals[method][:, y, x], color="#2778b8", linewidth=1.2, label="residual")
            axis.set_ylabel(DISPLAY_NAMES[method], fontsize=8)
            axis.axvline(COMMON_EVALUATION_START_FRAME_ZERO, color="#c89b2c", linestyle="--", linewidth=0.8)
        axes[1].legend(loc="upper right", ncol=3, fontsize=7)
        axes[-1].set_xlabel("Zero-based frame within 32-frame source-off window")
        trace_path = model / "figures" / "traces" / f"model_roi_{roi_id}.png"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(trace_path, dpi=150)
        plt.close(figure)

        instant_frame = COMMON_EVALUATION_START_FRAME_ZERO
        instant_panels = []
        for key, title in panel_order:
            values = panel_values(key, instant_frame)
            low, high = scales[key]
            instant_panels.append(_panel_image(values, title=title, low=low, high=high, marker_yx=(y, x), scale=3))
        instant_canvas = Image.new("RGB", (960, 864), (10, 13, 18))
        for panel_index, panel in enumerate(instant_panels):
            instant_canvas.paste(panel, ((panel_index % 5) * 192, (panel_index // 5) * 216))
        instant_path = model / "figures" / "instants" / f"model_roi_{roi_id}.png"
        instant_path.parent.mkdir(parents=True, exist_ok=True)
        instant_canvas.save(instant_path)

        metadata_path = model / "metadata" / f"roi_{roi_id}.json"
        atomic_json(
            metadata_path,
            {
                **occurrence,
                "frame_interval": [0, 32],
                "frame_convention": "zero_based_half_open",
                "coordinate_convention": "x_column_y_row",
                "trace_path": trace_path.relative_to(root).as_posix(),
                "full_field_video": full_path.relative_to(root).as_posix(),
                "closeup_video": close_path.relative_to(root).as_posix(),
                "instant_path": instant_path.relative_to(root).as_posix(),
                "display_scales": {key: list(value) for key, value in scales.items()},
            },
        )
        metadata_rows.append({**occurrence, "metadata_path": metadata_path.relative_to(root).as_posix()})
    _write_csv(model / "model_occurrences.csv", occurrence_rows)
    probes = []
    for path in video_paths:
        probe = _video_probe(path, 32)
        probe["path"] = path.relative_to(root).as_posix()
        probes.append(probe)
    return {
        "expert_section": "not_applicable_no_labels_or_truth_used",
        "model_surrogate_count": len(metadata_rows),
        "full_field_video_count": len(source_windows),
        "closeup_video_count": len(source_windows),
        "trace_figure_count": len(source_windows),
        "instant_figure_count": len(source_windows),
        "video_validation": probes,
        "all_videos_decode": all(row["full_decode_passed"] for row in probes),
        "all_video_frame_counts_exact": all(row["frame_count_exact"] for row in probes),
    }


def _comparison_figures(root: Path, rows: Sequence[Mapping[str, Any]], recording_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = root / "3_Comparison"
    source_rows = [row for row in rows if row["dataset_partition"] == "registered_source_off_windows"]
    metrics = [
        ("centered_rms_ratio", "Centered RMS ratio"),
        ("dynamic_mad_ratio", "Dynamic MAD ratio"),
        ("residual_seam_to_interior_jump_ratio", "Residual seam ratio"),
        ("temporal_spectral_flatness", "Spectral flatness"),
        ("residual_absolute_lag1_autocorrelation", "|lag-1 autocorrelation|"),
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(18, 5), constrained_layout=True)
    labels = [DISPLAY_NAMES[method] for method in METHODS]
    for axis, (metric, title) in zip(axes, metrics):
        values = [[float(row[metric]) for row in source_rows if row["method_id"] == method] for method in METHODS]
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=75, labelsize=7)
        if "ratio" in metric:
            axis.axhline(1.0, color="#9a6f13", linestyle="--", linewidth=0.9)
    path = comparison / "source_off_predictor_metric_overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)

    selected = [
        row
        for row in recording_rows
        if row["dataset_partition"] in {"held_validation", "registered_source_off_windows"}
    ]
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    recording_ids = sorted({str(row["recording_id"]) for row in selected})
    width = 0.1
    base = np.arange(len(recording_ids))
    for method_index, method in enumerate(METHODS):
        lookup = {
            str(row["recording_id"]): float(row["centered_rms_ratio_median"])
            for row in selected
            if row["method_id"] == method
        }
        values = [lookup.get(recording_id, np.nan) for recording_id in recording_ids]
        axis.bar(base + (method_index - 3.5) * width, values, width=width, label=DISPLAY_NAMES[method])
    axis.axhline(1.0, color="#9a6f13", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Median centered RMS ratio")
    axis.set_xticks(base, recording_ids, rotation=35, ha="right")
    axis.legend(ncol=2, fontsize=8)
    figure.savefig(comparison / "per_recording_consistency.png", dpi=170)
    plt.close(figure)


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"schema_version": SCHEMA_VERSION, "artifact_count": len(artifacts), "artifacts": artifacts}


def _resource_snapshot(output_parent: Path, device: str) -> dict[str, Any]:
    memory_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                memory_kib = int(line.split()[1])
                break
    disk_probe = Path(output_parent).resolve()
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    usage = shutil.disk_usage(disk_probe)
    return {
        "requested_device": device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "available_ram_mib": None if memory_kib is None else memory_kib / 1024.0,
        "available_disk_gib": usage.free / 1024.0**3,
        "torch_threads": int(torch.get_num_threads()),
    }


def _active_python_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    proc = Path("/proc")
    for candidate in proc.iterdir() if proc.is_dir() else ():
        if not candidate.name.isdigit():
            continue
        command_path = candidate / "cmdline"
        try:
            raw = command_path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = [part.decode("utf-8", errors="replace") for part in raw if part]
        if not command or not any("python" in Path(part).name.casefold() for part in command[:1]):
            continue
        rows.append(
            {
                "pid": int(candidate.name),
                "executable_name": Path(command[0]).name,
                "argument_count": len(command),
            }
        )
    return sorted(rows, key=lambda row: row["pid"])


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    branch = _git_bytes(repository_root, "branch", "--show-current").decode().strip()
    commit = _git_bytes(repository_root, "rev-parse", "HEAD").decode().strip()
    status = _git_bytes(repository_root, "status", "--porcelain=v1", "-z")
    tracked_diff = _git_bytes(repository_root, "diff", "--binary", "HEAD")
    untracked_raw = _git_bytes(
        repository_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_paths = sorted(
        path.decode("utf-8", errors="surrogateescape")
        for path in untracked_raw.split(b"\0")
        if path
    )
    untracked_rows: list[dict[str, Any]] = []
    for logical in untracked_paths:
        candidate = (repository_root / logical).resolve()
        if candidate.is_file():
            untracked_rows.append(
                {
                    "path": logical,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
        else:
            untracked_rows.append({"path": logical, "kind": "non_file"})
    status_entries = [
        entry.decode("utf-8", errors="surrogateescape")
        for entry in status.split(b"\0")
        if entry
    ]
    content = {
        "commit": commit,
        "branch": branch,
        "status_porcelain_v1_z_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_binary_diff_HEAD_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_content_manifest": untracked_rows,
    }
    return {
        **content,
        "dirty": bool(status),
        "status_entry_count": len(status_entries),
        "status_entries": status_entries,
        "dirty_digest_sha256": stable_hash(content),
        "digest_scope": "HEAD_commit_branch_status_porcelain_tracked_binary_diff_and_untracked_file_content",
    }


def _portable_command_argv(
    device: str,
    torch_threads: int,
    *,
    preflight: bool = False,
    render_media: bool = True,
) -> list[str]:
    """Return a reconstructable command without workstation-specific paths."""

    command = [
        "python",
        "-m",
        RUNNER_MODULE,
        "--repository-root",
        ".",
        "--data-root",
        "<runtime-data-root>",
        "--device",
        str(device),
        "--torch-threads",
        str(int(torch_threads)),
    ]
    if not render_media:
        command.append("--skip-media")
    if preflight:
        command.append("--preflight")
    return command


def _portable_command_passes_publication_boundary(command: Sequence[str]) -> bool:
    """Fail closed on home-directory tokens in publication-facing provenance."""

    markers = ("/home/", "/Users/", "\\\\Users\\\\")
    return bool(command) and all(
        isinstance(token, str)
        and bool(token)
        and not any(marker in token for marker in markers)
        for token in command
    )


def _runtime_provenance(
    device: str,
    torch_threads: int,
    *,
    preflight: bool = False,
    render_media: bool = True,
) -> dict[str, Any]:
    portable_command = _portable_command_argv(
        device,
        torch_threads,
        preflight=preflight,
        render_media=render_media,
    )
    if not _portable_command_passes_publication_boundary(portable_command):
        raise SourceOffPredictorFeasibilityError(
            "portable command provenance failed the publication-boundary path scan"
        )
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None
    return {
        "mode": "read_only_preflight" if preflight else "screen",
        "command_argv_portable": portable_command,
        "command_argv_sanitized": portable_command,
        "command_path_policy": (
            "portable_module_invocation_repository_dot_and_redacted_runtime_data_root"
        ),
        "publication_boundary_workstation_path_scan": "passed",
        "python_executable_name": Path(sys.executable).name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn_version,
        "device": device,
        "torch_threads": int(torch_threads),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _canonical_output(repository_root: Path, output_root: Path | None) -> Path:
    canonical = (repository_root / DEFAULT_OUTPUT_ROOT).resolve()
    if output_root is not None and Path(output_root).expanduser().resolve() != canonical:
        raise SourceOffPredictorFeasibilityError(
            "screen execution and preflight forbid non-canonical output_root overrides"
        )
    return canonical


def preflight_source_off_predictor_feasibility(
    *,
    repository_root: Path,
    data_root: Path,
    output_root: Path | None = None,
    device: str = "cpu",
    torch_threads: int = 4,
) -> dict[str, Any]:
    """Run all live input/resource checks without writing any output."""

    repository = Path(repository_root).expanduser().resolve()
    runtime_data = Path(data_root).expanduser().resolve()
    requested = _canonical_output(repository, output_root)
    partial = requested.with_name(requested.name + ".partial")
    torch.set_num_threads(int(torch_threads))
    implementation = _verify_implementation_descriptor(repository)
    exp0028 = (repository / EXP0028_ROOT).resolve()
    run_b = (repository / EXP0029_RUN_B_ROOT).resolve()
    parent_0028 = _verify_indexed_root(
        exp0028, expected_index_sha256=EXP0028_ARTIFACT_INDEX_SHA256
    )
    parent_run_b = _verify_indexed_root(
        run_b, expected_index_sha256=EXP0029_RUN_B_ARTIFACT_INDEX_SHA256
    )
    train, validation, cache_manifest = _load_clip_banks(exp0028)
    validation_rows = _validation_metadata(exp0028)
    source_windows, source_manifest = _load_source_off_windows(
        repository, runtime_data, exp0028
    )
    decoders = FrozenRunBDecoders.load(exp0028, run_b, device=device)
    runner_path = Path(__file__).resolve()
    checks = {
        "canonical_experiment_id": EXPERIMENT_ID == "NREV-EXP-0030",
        "canonical_run_id": RUN_ID == "NREV-RUN-EXP-0030-SCREEN-20260830-B",
        "canonical_output_root": requested == (repository / DEFAULT_OUTPUT_ROOT).resolve(),
        "output_root_absent": not requested.exists(),
        "partial_root_absent": not partial.exists(),
        "EXP0028_artifact_index_and_25_artifacts_verified": parent_0028["artifact_count"] == 25,
        "EXP0029_Run_B_artifact_index_and_27_artifacts_verified": parent_run_b["artifact_count"] == 27,
        "train_cache_exact": tuple(train.shape) == TRAIN_SHAPE,
        "held_validation_cache_exact": tuple(validation.shape) == VALIDATION_SHAPE,
        "held_validation_request_count_exact": len(validation_rows) == VALIDATION_SHAPE[0],
        "registered_source_off_window_count_exact": len(source_windows) == 12,
        "selected_live_source_hashes_verified": source_manifest["source_recording_count"] == 4,
        "normalization_exact": source_manifest["normalization"]["normalization_sha256"]
        == NORMALIZATION_SHA256,
        "implementation_hashes_frozen_and_verified": bool(
            implementation["checks"]["all_source_hashes_match"]
        ),
        "frozen_decoder_parameters_verified": bool(decoders.manifest["all_parameters_frozen"]),
        "device_available": not device.startswith("cuda") or torch.cuda.is_available(),
        "data_root_available": runtime_data.is_dir(),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "output_root": DEFAULT_OUTPUT_ROOT,
        "planned_at_utc": PLANNED_AT_UTC,
        "timestamp_utc": _utc_now(),
        "read_only": True,
        "writes_performed": False,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "resources": _resource_snapshot(requested.parent, device),
        "active_python_processes": _active_python_processes(),
        "git": _git_provenance(repository),
        "runtime": _runtime_provenance(device, torch_threads, preflight=True),
        "runner": {
            "path": runner_path.relative_to(repository).as_posix(),
            "sha256": sha256_file(runner_path),
        },
        "implementation_descriptor": implementation,
        "parent_integrity": {
            "EXP0028": parent_0028,
            "EXP0029_Run_B": parent_run_b,
        },
        "clip_banks": cache_manifest,
        "source_off_windows": source_manifest,
        "frozen_Run_B_decoders": decoders.manifest,
        "coverage_plan": {
            "fit_training_clips": 512,
            "held_validation_clips": 96,
            "registered_source_off_windows": 12,
            "methods": 8,
            "expected_metric_rows": 864,
        },
        "authorization": {
            "engineering_execution_authorized": True,
            "claim_bearing_execution_authorized": False,
        },
        "scientific_boundary": {
            "labels_used": False,
            "injected_source_truth_used": False,
            "detector_outcomes_used": False,
            "scientific_claim_consequence": "none",
        },
    }
    if payload["status"] != "passed":
        raise SourceOffPredictorFeasibilityError(
            "read-only preflight failed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    return payload


def _report(
    aggregate_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    lookup = {
        (str(row["dataset_partition"]), str(row["method_id"])): row
        for row in aggregate_rows
    }
    lines = [
        "# Source-off conditional-background predictor feasibility v1",
        "",
        "## Outcome first",
        "",
    ]
    eligible = list(gate["eligible_predictor_methods"])
    if eligible:
        lines.append(
            "At least one predictor passed the predeclared source-off-only engineering gate: "
            + ", ".join(f"`{method}`" for method in eligible)
            + ". This permits only a separately versioned residual-detector safety screen; it is not scientific evidence."
        )
    else:
        lines.append(
            "No tested predictor passed the complete source-off-only feasibility gate. The appropriate next action is not a new residual-detector experiment; first resolve the failed scale, dynamics, whiteness, seam, or recording-consistency checks."
        )
    lines.extend(
        [
            "",
            "This benchmark used no source truth, injected sources, ROI labels, detector scores, or recovery outcomes. Simple models were fitted on the frozen 512-clip training bank only. Evaluation used all 96 recording-held-out clips plus the exact 12 registered source-off/quiet windows. Native activity inside unlabeled clips remains unknown.",
            "",
            "## Aggregate results",
            "",
            "### All 96 held-recording clips",
            "",
            "| Method | Centered RMS ratio | Dynamic MAD ratio | Seam ratio | Spectral flatness | |lag-1 ACF| |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        row = lookup[("held_validation", method)]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {float(row['centered_rms_ratio_median']):.4f} | "
            f"{float(row['dynamic_mad_ratio_median']):.4f} | "
            f"{float(row['residual_seam_to_interior_jump_ratio_median']):.4f} | "
            f"{float(row['temporal_spectral_flatness_median']):.4f} | "
            f"{float(row['residual_absolute_lag1_autocorrelation_median']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Exact 12 registered source-off windows",
            "",
            "| Method | Centered RMS ratio | Dynamic MAD ratio | Seam ratio | Spectral flatness | |lag-1 ACF| | Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for method in METHODS:
        row = lookup[("registered_source_off_windows", method)]
        method_gate = gate["methods"][method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {float(row['centered_rms_ratio_median']):.4f} | "
            f"{float(row['dynamic_mad_ratio_median']):.4f} | "
            f"{float(row['residual_seam_to_interior_jump_ratio_median']):.4f} | "
            f"{float(row['temporal_spectral_flatness_median']):.4f} | "
            f"{float(row['residual_absolute_lag1_autocorrelation_median']):.4f} | "
            f"{'pass' if method_gate['passed'] else 'fail'} ({method_gate['passed_check_count']}/{method_gate['required_check_count']}) |"
        )
    lines.extend(
        [
            "",
            "## Frozen design",
            "",
            "- Common metric support is zero-based frames 8-31 (24 frames) for every method.",
            "- No-subtraction uses a zero background prediction, so its signed residual is exactly the normalized observation.",
            f"- Causal EMA alpha is `{EMA_ALPHA}` and causal median history is `{CAUSAL_MEDIAN_WINDOW}` frames; neither was validation-selected.",
            "- Per-pixel AR(1) and rank-8 spatial-basis/latent-AR(1) parameters were fit only on the 512 training clips.",
            "- Frozen JEPA and random-provider decoders are byte-verified final-step Run-B heads; there was no refit or checkpoint selection.",
            "- Centered RMS/MAD use per-pixel temporal-median centering. Dynamic MAD uses first differences. Spectral flatness and autocorrelations are residual-whiteness diagnostics. Positive prediction lag means prediction trails observation.",
            "- Seam ratio compares absolute jumps at the frozen 8-pixel decoder lattice with all other adjacent-pixel jumps.",
            "",
            "## Compute",
            "",
        ]
    )
    for method in METHODS:
        row = runtime["methods"][method]
        lines.append(
            f"- `{method}`: fit `{float(row['fit_seconds']):.3f}` s; inference `{float(row['inference_seconds_total']):.3f}` s across `{int(row['clip_count'])}` clips."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A source-off feasibility pass would only establish that a predictor is numerically safe enough to enter a new detector screen. It would not establish neuron preservation, denoising, specificity, biological identity, independent-animal generalization, or improved detection. This run remains non-claim-bearing and creates no evidence capsule.",
            "",
        ]
    )
    return "\n".join(lines)


def run_source_off_predictor_feasibility(
    *,
    repository_root: Path,
    data_root: Path,
    output_root: Path | None = None,
    device: str = "cpu",
    render_media: bool = True,
    torch_threads: int = 4,
) -> dict[str, Any]:
    started_at_utc = _utc_now()
    wall_started = time.perf_counter()
    repository = Path(repository_root).expanduser().resolve()
    runtime_data = Path(data_root).expanduser().resolve()
    requested = _canonical_output(repository, output_root)
    partial = requested.with_name(requested.name + ".partial")
    if requested.exists() or partial.exists():
        raise SourceOffPredictorFeasibilityError(
            f"output collision: {requested} or {partial} already exists"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SourceOffPredictorFeasibilityError("CUDA requested but unavailable")
    torch.set_num_threads(int(torch_threads))
    git_at_start = _git_provenance(repository)
    runtime_at_start = _runtime_provenance(
        device,
        torch_threads,
        render_media=render_media,
    )

    preflight = preflight_source_off_predictor_feasibility(
        repository_root=repository,
        data_root=runtime_data,
        output_root=requested,
        device=device,
        torch_threads=torch_threads,
    )

    exp0028 = (repository / EXP0028_ROOT).resolve()
    run_b = (repository / EXP0029_RUN_B_ROOT).resolve()
    before_0028 = _verify_indexed_root(exp0028, expected_index_sha256=EXP0028_ARTIFACT_INDEX_SHA256)
    before_run_b = _verify_indexed_root(run_b, expected_index_sha256=EXP0029_RUN_B_ARTIFACT_INDEX_SHA256)
    requested.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True)
    atomic_json(partial / "preflight.json", preflight)
    atomic_json(partial / "heartbeat.json", {"stage": "loading_frozen_inputs", "timestamp_utc": _utc_now()})
    try:
        train, validation, cache_manifest = _load_clip_banks(exp0028)
        validation_metadata = _validation_metadata(exp0028)
        source_windows, source_manifest = _load_source_off_windows(
            repository, runtime_data, exp0028
        )
        fit_times: dict[str, float] = {method: 0.0 for method in METHODS}
        fit_start = time.perf_counter()
        ar1 = fit_per_pixel_ar1(train)
        fit_times["per_pixel_ar1_train512"] = time.perf_counter() - fit_start
        fit_start = time.perf_counter()
        low_rank = fit_low_rank_ar1(train)
        fit_times["low_rank_ar1_rank8_train512"] = time.perf_counter() - fit_start
        state_path = partial / "predictor_state.npz"
        state_sha = _save_predictor_state(state_path, ar1, low_rank)
        state_manifest = _predictor_state_manifest(ar1, low_rank)
        state_manifest["state_path"] = "predictor_state.npz"
        state_manifest["state_sha256"] = state_sha
        atomic_json(partial / "predictor_state_manifest.json", state_manifest)

        decoders = FrozenRunBDecoders.load(exp0028, run_b, device=device)
        runner_path = Path(__file__).resolve()
        runner_sha256 = sha256_file(runner_path)
        input_manifest = {
            "schema_version": SCHEMA_VERSION,
            "runner": {
                "path": runner_path.relative_to(repository).as_posix(),
                "sha256": runner_sha256,
                "hash_scope": "complete_file_bytes",
            },
            "EXP0028_before": before_0028,
            "EXP0029_Run_B_before": before_run_b,
            "clip_banks": cache_manifest,
            "source_off_windows": source_manifest,
            "frozen_Run_B_decoders": decoders.manifest,
            "normalization_sha256": NORMALIZATION_SHA256,
            "normalization_application_count": {
                "frozen_normalized_clip_banks": 0,
                "raw_source_off_windows": 1,
            },
            "copied_raw_payload": False,
        }
        atomic_json(partial / "input_manifest.json", input_manifest)
        resolved_config = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "runner_sha256": runner_sha256,
            "status": "bounded_non_claim_bearing_engineering_benchmark",
            "output_root": DEFAULT_OUTPUT_ROOT,
            "methods": list(METHODS),
            "display_names": DISPLAY_NAMES,
            "training_clip_count": TRAIN_SHAPE[0],
            "held_validation_clip_count": VALIDATION_SHAPE[0],
            "registered_source_off_window_count": len(source_windows),
            "common_evaluation_start_frame_zero": COMMON_EVALUATION_START_FRAME_ZERO,
            "common_evaluation_frame_count": 24,
            "normalization": {
                "center": NORMALIZATION_CENTER,
                "scale": NORMALIZATION_SCALE,
                "sha256": NORMALIZATION_SHA256,
            },
            "fixed_hyperparameters": {
                "EMA_alpha": EMA_ALPHA,
                "causal_median_window": CAUSAL_MEDIAN_WINDOW,
                "low_rank_rank": LOW_RANK_RANK,
                "low_rank_sample_frames": LOW_RANK_SAMPLE_FRAMES,
                "low_rank_random_seed": LOW_RANK_RANDOM_SEED,
                "decoder_target_batch_size": 64,
            },
            "gate_thresholds": GATE_THRESHOLDS,
            "fit_selection_boundary": "train512_only_no_labels_no_source_truth_no_validation_selection",
            "scientific_audit": {
                "enabled": True,
                "expert_section": "not_applicable_unlabeled_benchmark",
                "model_candidate_surrogate": "one_method_blind_raw_dynamic_MAD_argmax_pixel_per_source_off_window",
                "comparison_media": "figures_and_tables_only",
            },
            "scientific_completion_allowed": False,
            "claim_or_capsule_creation_allowed": False,
        }
        atomic_json(partial / "resolved_config.json", resolved_config)

        rows: list[dict[str, Any]] = []
        inference_times: dict[str, list[float]] = {method: [] for method in METHODS}
        coverage_rows: list[dict[str, Any]] = []
        source_predictions: dict[str, dict[str, np.ndarray]] = {}

        def evaluate_one(
            movie: np.ndarray,
            *,
            dataset_partition: str,
            clip_id: str,
            recording_id: str,
            extra: Mapping[str, Any],
            retain_predictions: bool,
        ) -> None:
            retained: dict[str, np.ndarray] = {}
            for method in METHODS:
                prediction_start = time.perf_counter()
                if method in {
                    "frozen_jepa_decoder_run_b",
                    "frozen_random_decoder_run_b",
                }:
                    prediction, coverage = decoders.predict(method, movie, device=device)
                else:
                    prediction = _simple_prediction(method, movie, ar1=ar1, low_rank=low_rank)
                    coverage = {
                        "coverage_minimum": 1,
                        "coverage_maximum": 1,
                        "coverage_exactly_once": True,
                        "finite": bool(np.isfinite(prediction).all()),
                        "shape_tyx": list(prediction.shape),
                        "coverage_semantics": "direct_nontiled_prediction",
                    }
                elapsed = time.perf_counter() - prediction_start
                inference_times[method].append(elapsed)
                metrics = _flatten_metric_row(source_off_metrics(movie, prediction))
                rows.append(
                    {
                        "dataset_partition": dataset_partition,
                        "clip_id": clip_id,
                        "recording_id": recording_id,
                        "method_id": method,
                        "method_display_name": DISPLAY_NAMES[method],
                        "inference_seconds": elapsed,
                        **dict(extra),
                        **metrics,
                    }
                )
                coverage_rows.append(
                    {
                        "dataset_partition": dataset_partition,
                        "clip_id": clip_id,
                        "method_id": method,
                        **coverage,
                    }
                )
                if retain_predictions:
                    retained[method] = prediction
            if retain_predictions:
                source_predictions[clip_id] = retained

        atomic_json(partial / "heartbeat.json", {"stage": "held_recording_validation", "timestamp_utc": _utc_now()})
        for index in range(validation.shape[0]):
            metadata = validation_metadata[index]
            evaluate_one(
                np.asarray(validation[index], dtype=np.float32),
                dataset_partition="held_validation",
                clip_id=f"validation_clip_{index:03d}",
                recording_id=str(metadata["recording_id"]),
                extra={
                    "sampling_stratum": metadata["sampling_stratum"],
                    "background_mad_stratum": "not_registered_for_validation_bank",
                    "start_frame_zero": int(metadata["start_frame_zero"]),
                    "x_zero": int(metadata["x_zero"]),
                    "y_zero": int(metadata["y_zero"]),
                },
                retain_predictions=False,
            )
        atomic_json(partial / "heartbeat.json", {"stage": "registered_source_off_windows", "timestamp_utc": _utc_now()})
        for item in source_windows:
            metadata = item["metadata"]
            evaluate_one(
                np.asarray(item["movie"], dtype=np.float32),
                dataset_partition="registered_source_off_windows",
                clip_id=str(metadata["background_window_id"]),
                recording_id=str(metadata["recording_id"]),
                extra={
                    "sampling_stratum": metadata["sampling_stratum"],
                    "background_mad_stratum": metadata["background_mad_stratum"],
                    "start_frame_zero": int(metadata["start_frame_zero"]),
                    "x_zero": int(metadata["x_zero"]),
                    "y_zero": int(metadata["y_zero"]),
                },
                retain_predictions=True,
            )

        expected_rows = (VALIDATION_SHAPE[0] + len(source_windows)) * len(METHODS)
        if len(rows) != expected_rows:
            raise SourceOffPredictorFeasibilityError("metric coverage count drifted")
        if not all(row["coverage_exactly_once"] and row["finite"] for row in coverage_rows):
            raise SourceOffPredictorFeasibilityError("prediction coverage or finiteness failed")
        aggregate = _aggregate_rows(rows, ("dataset_partition", "method_id"))
        per_recording = _aggregate_rows(rows, ("dataset_partition", "recording_id", "method_id"))
        gate = evaluate_feasibility_gate(aggregate, per_recording)
        runtime = {
            "schema_version": SCHEMA_VERSION,
            "device": device,
            "torch_threads": int(torch_threads),
            "methods": {
                method: {
                    "fit_seconds": fit_times[method],
                    "inference_seconds_total": float(sum(inference_times[method])),
                    "inference_seconds_median_per_clip": float(np.median(inference_times[method])),
                    "clip_count": len(inference_times[method]),
                }
                for method in METHODS
            },
        }
        atomic_json(partial / "per_clip_metrics.json", {"schema_version": SCHEMA_VERSION, "rows": rows})
        _write_tsv(partial / "per_clip_metrics.tsv", rows)
        atomic_json(partial / "aggregate_metrics.json", {"schema_version": SCHEMA_VERSION, "rows": aggregate})
        _write_tsv(partial / "aggregate_metrics.tsv", aggregate)
        atomic_json(partial / "per_recording_metrics.json", {"schema_version": SCHEMA_VERSION, "rows": per_recording})
        _write_tsv(partial / "per_recording_metrics.tsv", per_recording)
        atomic_json(partial / "prediction_coverage.json", {"schema_version": SCHEMA_VERSION, "rows": coverage_rows})
        atomic_json(partial / "feasibility_gate.json", gate)
        atomic_json(partial / "runtime_metrics.json", runtime)

        atomic_json(partial / "heartbeat.json", {"stage": "rendering_scientific_audit", "timestamp_utc": _utc_now()})
        audit = (
            _render_audit_media(partial, source_windows, source_predictions)
            if render_media
            else {
                "expert_section": "not_applicable_no_labels_or_truth_used",
                "rendering_skipped": True,
                "scientific_audit_complete": False,
            }
        )
        _comparison_figures(partial, rows, per_recording)
        audit_status = {
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "expert_section_applicability": "not_applicable_unlabeled_benchmark",
            "model_candidate_surrogate_policy": "method_blind_argmax_raw_temporal_difference_MAD_per_source_off_window",
            "comparison_figures_only": True,
            **audit,
            "applicable_audit_outputs_complete": bool(render_media),
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
        }
        atomic_json(partial / "scientific_audit_status.json", audit_status)

        after_0028 = _verify_indexed_root(exp0028, expected_index_sha256=EXP0028_ARTIFACT_INDEX_SHA256)
        after_run_b = _verify_indexed_root(run_b, expected_index_sha256=EXP0029_RUN_B_ARTIFACT_INDEX_SHA256)
        upstream_integrity = {
            "schema_version": SCHEMA_VERSION,
            "EXP0028": {
                "before": before_0028,
                "after": after_0028,
                "unchanged": before_0028["snapshot_sha256"] == after_0028["snapshot_sha256"],
            },
            "EXP0029_Run_B": {
                "before": before_run_b,
                "after": after_run_b,
                "unchanged": before_run_b["snapshot_sha256"] == after_run_b["snapshot_sha256"],
            },
        }
        if not upstream_integrity["EXP0028"]["unchanged"] or not upstream_integrity["EXP0029_Run_B"]["unchanged"]:
            raise SourceOffPredictorFeasibilityError("frozen upstream output changed during benchmark")
        atomic_json(partial / "upstream_integrity.json", upstream_integrity)

        total_seconds = time.perf_counter() - wall_started
        ended_at_utc = _utc_now()
        runtime["total_wall_seconds"] = total_seconds
        atomic_json(partial / "runtime_metrics.json", runtime)
        execution_provenance = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "mode": "screen",
            "planned_at_utc": PLANNED_AT_UTC,
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "duration_seconds": total_seconds,
            "timestamp_authority": "explicit_UTC_runtime_values_not_filesystem_mtimes",
            "git_at_start": git_at_start,
            "runtime_at_start": runtime_at_start,
            "implementation": preflight["implementation_descriptor"],
            "configuration_hashes": {
                "canonical_descriptor_path": DESCRIPTOR_PATH,
                "canonical_descriptor_sha256": preflight["implementation_descriptor"]["sha256"],
                "resolved_config_path": "resolved_config.json",
                "resolved_config_sha256": sha256_file(partial / "resolved_config.json"),
            },
            "input_hashes": {
                "input_manifest_path": "input_manifest.json",
                "input_manifest_sha256": sha256_file(partial / "input_manifest.json"),
                "EXP0028_artifact_index_sha256": EXP0028_ARTIFACT_INDEX_SHA256,
                "EXP0029_Run_B_artifact_index_sha256": EXP0029_RUN_B_ARTIFACT_INDEX_SHA256,
                "training_cache_sha256": TRAIN_CACHE_SHA256,
                "validation_cache_sha256": VALIDATION_CACHE_SHA256,
                "background_window_manifest_sha256": BACKGROUND_WINDOW_MANIFEST_SHA256,
                "JEPA_checkpoint_sha256": JEPA_CHECKPOINT_SHA256,
                "Run_B_decoder_checkpoint_sha256": RUN_B_DECODER_CHECKPOINT_SHA256,
                "normalization_sha256": NORMALIZATION_SHA256,
            },
            "output_root": DEFAULT_OUTPUT_ROOT,
            "scientific_claim_consequence": "none",
        }
        atomic_json(partial / "execution_provenance.json", execution_provenance)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "status": "succeeded_engineering_benchmark",
            "scientific_status": "non_claim_bearing",
            "planned_at_utc": PLANNED_AT_UTC,
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "duration_seconds": total_seconds,
            "training_clip_count": TRAIN_SHAPE[0],
            "held_validation_clip_count": VALIDATION_SHAPE[0],
            "registered_source_off_window_count": len(source_windows),
            "method_count": len(METHODS),
            "per_clip_metric_row_count": len(rows),
            "eligible_predictor_methods": gate["eligible_predictor_methods"],
            "any_predictor_passed": gate["any_predictor_passed"],
            "gate_consequence": gate["consequence"],
            "aggregate_metrics": aggregate,
            "limitations": [
                "No injected-source or biological-signal preservation endpoint is evaluated.",
                "Held recordings may share animal, preparation, date, and acquisition conditions.",
                "Registered quiet/source-off windows do not establish absence of all native neural activity.",
                "Physical pixel scale and frame cadence remain unresolved for 060126.",
                "Frozen JEPA and random decoders are one-seed non-scientific screen checkpoints.",
            ],
            "scientific_claim_consequence": "none",
        }
        atomic_json(partial / "summary.json", summary)
        atomic_text(partial / "REPORT.md", _report(aggregate, gate, runtime))
        llm_context = {
            "schema_version": SCHEMA_VERSION,
            "entry_point": "summary.json",
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "experiment_role": "source_off_predictor_feasibility_engineering_benchmark",
            "model_stage_sequence": ["normalized observation", "predicted background", "signed residual"],
            "expert_section": "not_applicable_no_labels_or_truth",
            "annotation_separation": "strict",
            "candidate_surrogate": "one method-blind raw-dynamic-MAD argmax pixel per source-off window",
            "primary_tables": [
                "aggregate_metrics.tsv",
                "per_recording_metrics.tsv",
                "per_clip_metrics.tsv",
            ],
            "primary_gate": "feasibility_gate.json",
            "execution_provenance": "execution_provenance.json",
            "primary_figures": [
                "3_Comparison/source_off_predictor_metric_overview.png",
                "3_Comparison/per_recording_consistency.png",
            ],
            "coordinate_convention": "x_column_y_row",
            "frame_convention": "zero_based_half_open",
            "common_metric_support": [COMMON_EVALUATION_START_FRAME_ZERO, 32],
            "key_finding": gate["consequence"],
            "scientific_claim_consequence": "none",
        }
        atomic_json(partial / "llm_context.json", llm_context)
        checks = {
            "per_clip_metric_rows_exact": len(rows) == 864,
            "aggregate_rows_exact": len(aggregate) == 16,
            "per_recording_rows_exact": len(per_recording) == 56,
            "all_methods_present": {row["method_id"] for row in rows} == set(METHODS),
            "held_validation_coverage_exact": sum(row["dataset_partition"] == "held_validation" for row in rows) == 768,
            "registered_source_off_coverage_exact": sum(row["dataset_partition"] == "registered_source_off_windows" for row in rows) == 96,
            "all_predictions_finite_and_covered": all(row["coverage_exactly_once"] and row["finite"] for row in coverage_rows),
            "no_source_truth_or_labels_used": True,
            "validation_not_used_for_fit_or_selection": True,
            "frozen_upstreams_unchanged": upstream_integrity["EXP0028"]["unchanged"] and upstream_integrity["EXP0029_Run_B"]["unchanged"],
            "execution_provenance_complete": all(
                execution_provenance.get(field)
                for field in (
                    "planned_at_utc",
                    "started_at_utc",
                    "ended_at_utc",
                    "duration_seconds",
                    "git_at_start",
                    "runtime_at_start",
                    "implementation",
                    "configuration_hashes",
                    "input_hashes",
                )
            ),
            "portable_command_provenance_passes_publication_boundary": (
                _portable_command_passes_publication_boundary(
                    runtime_at_start["command_argv_portable"]
                )
                and runtime_at_start["command_argv_portable"][:3]
                == ["python", "-m", RUNNER_MODULE]
                and runtime_at_start["command_argv_portable"][4] == "."
            ),
            "audit_applicable_outputs_complete": bool(audit_status["applicable_audit_outputs_complete"]),
            "scientific_completion_false": True,
        }
        validation_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "scientific_completion": False,
            "scientific_promotion_allowed": False,
        }
        atomic_json(partial / "validation.json", validation_payload)
        atomic_json(
            partial / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "lifecycle": "succeeded",
                "engineering_execution": "complete",
                "validation": validation_payload["status"],
                "scientific_completion": False,
                "claim_bearing": False,
                "planned_at_utc": PLANNED_AT_UTC,
                "started_at_utc": started_at_utc,
                "ended_at_utc": ended_at_utc,
                "duration_seconds": total_seconds,
            },
        )
        atomic_json(partial / "heartbeat.json", {"stage": "complete", "timestamp_utc": _utc_now()})
        atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        requested.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(requested)
        return summary
    except Exception as exc:
        failed_at_utc = _utc_now()
        atomic_json(
            partial / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "lifecycle": "failed_nonresumable_partial",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "planned_at_utc": PLANNED_AT_UTC,
                "started_at_utc": started_at_utc,
                "ended_at_utc": failed_at_utc,
                "duration_seconds": time.perf_counter() - wall_started,
                "git_at_start": git_at_start,
                "runtime_at_start": runtime_at_start,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--skip-media", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight:
        preflight = preflight_source_off_predictor_feasibility(
            repository_root=args.repository_root,
            data_root=args.data_root,
            output_root=args.output_root,
            device=str(args.device),
            torch_threads=int(args.torch_threads),
        )
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return 0
    summary = run_source_off_predictor_feasibility(
        repository_root=args.repository_root,
        data_root=args.data_root,
        output_root=args.output_root,
        device=str(args.device),
        render_media=not bool(args.skip_media),
        torch_threads=int(args.torch_threads),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAUSAL_MEDIAN_WINDOW",
    "COMMON_EVALUATION_START_FRAME_ZERO",
    "EMA_ALPHA",
    "GATE_THRESHOLDS",
    "LOW_RANK_RANK",
    "LowRankAR1State",
    "METHODS",
    "PerPixelAR1State",
    "SourceOffPredictorFeasibilityError",
    "causal_ema_prediction",
    "causal_temporal_median_prediction",
    "causal_zero_order_hold_prediction",
    "evaluate_feasibility_gate",
    "fit_low_rank_ar1",
    "fit_per_pixel_ar1",
    "low_rank_ar1_prediction",
    "no_subtraction_prediction",
    "per_pixel_ar1_prediction",
    "preflight_source_off_predictor_feasibility",
    "run_source_off_predictor_feasibility",
    "source_off_metrics",
]
