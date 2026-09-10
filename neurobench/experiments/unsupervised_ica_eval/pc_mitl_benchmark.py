"""Paired synthetic E01/E02 benchmark for the PC-MITL-ICA research gate."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .matrix_itl import fit_cs_parzen_orthogonal, fit_matrix_tc_ica


@dataclass(frozen=True)
class SyntheticFixture:
    sources: np.ndarray
    observed: np.ndarray
    mixing: np.ndarray
    event_mask: np.ndarray
    noise: np.ndarray
    seed: int
    fingerprint: str


@dataclass(frozen=True)
class PreprocessState:
    mean: np.ndarray
    whitener: np.ndarray
    dewhitener: np.ndarray
    eigenvalues: np.ndarray
    train_data_hash: str


def generate_fixture(
    *,
    sample_count: int,
    component_count: Literal[2, 4],
    source_family: Literal["laplace", "sparse_calcium"],
    condition_number: float,
    noise_std: float,
    seed: int,
) -> SyntheticFixture:
    """Generate one paired fixture with independent component RNG streams."""
    if sample_count < 96 or component_count not in (2, 4):
        raise ValueError("sample_count >= 96 and component_count in {2,4} required")
    if condition_number < 1 or noise_std < 0:
        raise ValueError("condition_number >= 1 and noise_std >= 0 required")
    root = np.random.SeedSequence(seed)
    source_sequences = root.spawn(component_count + 3)
    sources = np.zeros((sample_count, component_count), dtype=np.float64)
    event_mask = np.zeros_like(sources, dtype=bool)
    for component in range(component_count):
        rng = np.random.default_rng(source_sequences[component])
        if source_family == "laplace":
            sources[:, component] = rng.laplace(size=sample_count)
            event_mask[:, component] = np.abs(sources[:, component]) > 2.5
        elif source_family == "sparse_calcium":
            events = rng.random(sample_count) < 0.06
            amplitudes = events * rng.lognormal(mean=0.0, sigma=0.35, size=sample_count)
            trace = np.zeros(sample_count, dtype=np.float64)
            for index in range(1, sample_count):
                trace[index] = 0.88 * trace[index - 1] + amplitudes[index]
            sources[:, component] = trace
            event_mask[:, component] = events
        else:
            raise ValueError("unknown source_family")
        sources[:, component] -= sources[:, component].mean()
        sources[:, component] /= max(sources[:, component].std(), np.finfo(float).eps)
    left, _ = np.linalg.qr(np.random.default_rng(source_sequences[-3]).normal(size=(component_count, component_count)))
    right, _ = np.linalg.qr(np.random.default_rng(source_sequences[-2]).normal(size=(component_count, component_count)))
    singular = np.geomspace(1.0, 1.0 / condition_number, component_count)
    mixing = left @ np.diag(singular) @ right.T
    noise = np.random.default_rng(source_sequences[-1]).normal(scale=noise_std, size=sources.shape)
    observed = sources @ mixing.T + noise
    digest = hashlib.sha256()
    for array in (sources, mixing, noise):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(json.dumps({"seed": seed, "family": source_family}, sort_keys=True).encode())
    return SyntheticFixture(sources, observed, mixing, event_mask, noise, seed, digest.hexdigest())


def fit_train_whitener(observed_train: np.ndarray, *, floor_ratio: float = 1e-6) -> PreprocessState:
    values = np.asarray(observed_train, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("observed_train must be finite [N,q]")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * floor_ratio, np.finfo(float).eps)
    floored = np.maximum(eigenvalues, floor)
    whitener = vectors @ np.diag(1 / np.sqrt(floored)) @ vectors.T
    dewhitener = vectors @ np.diag(np.sqrt(floored)) @ vectors.T
    return PreprocessState(mean, whitener, dewhitener, eigenvalues, hashlib.sha256(values.tobytes()).hexdigest())


def apply_whitener(observed: np.ndarray, state: PreprocessState) -> np.ndarray:
    return (np.asarray(observed, dtype=np.float64) - state.mean) @ state.whitener


def matched_recovery(reference: np.ndarray, recovered: np.ndarray) -> dict[str, object]:
    if reference.shape != recovered.shape or reference.ndim != 2:
        raise ValueError("reference and recovered must share [N,q] shape")
    correlations = np.corrcoef(reference.T, recovered.T)[: reference.shape[1], reference.shape[1] :]
    rows, columns = linear_sum_assignment(-np.abs(correlations))
    matched = np.asarray([abs(correlations[row, column]) for row, column in zip(rows, columns)])
    return {
        "mean_absolute_correlation": float(matched.mean()),
        "minimum_absolute_correlation": float(matched.min()),
        "matched_absolute_correlations": matched.tolist(),
        "permutation": columns.tolist(),
    }


def evaluate_cell(
    fixture: SyntheticFixture,
    *,
    bandwidth_candidates: Sequence[float],
    train_stop: int,
    validation_stop: int,
    steps: int,
    learning_rate: float,
) -> list[dict[str, object]]:
    """Select bandwidth on validation and evaluate once on untouched test."""
    if not 0 < train_stop < validation_stop < len(fixture.sources):
        raise ValueError("split boundaries must define non-empty contiguous blocks")
    state = fit_train_whitener(fixture.observed[:train_stop])
    whitened = apply_whitener(fixture.observed, state)
    train = torch.as_tensor(whitened[:train_stop], dtype=torch.float64)
    rows: list[dict[str, object]] = []
    fitters = {"cs_parzen": fit_cs_parzen_orthogonal, "matrix_tc_alpha2": fit_matrix_tc_ica}
    for method, fitter in fitters.items():
        candidates = []
        for bandwidth in bandwidth_candidates:
            started = time.perf_counter()
            fit = fitter(
                train, [float(bandwidth)] * train.shape[1], seed=fixture.seed,
                steps=steps, learning_rate=learning_rate,
            )
            elapsed = time.perf_counter() - started
            recovered = torch.as_tensor(whitened, dtype=torch.float64) @ fit.rotation.T
            validation = matched_recovery(
                fixture.sources[train_stop:validation_stop], recovered[train_stop:validation_stop].numpy()
            )
            candidates.append((float(validation["mean_absolute_correlation"]), bandwidth, fit, recovered, elapsed))
        validation_score, bandwidth, fit, recovered, elapsed = max(candidates, key=lambda row: row[0])
        test = matched_recovery(fixture.sources[validation_stop:], recovered[validation_stop:].numpy())
        rows.append({
            "method": method, "selected_bandwidth": float(bandwidth),
            "validation_mean_absolute_correlation": validation_score,
            "test_mean_absolute_correlation": test["mean_absolute_correlation"],
            "test_minimum_absolute_correlation": test["minimum_absolute_correlation"],
            "objective": fit.objective, "orthogonality_error": fit.orthogonality_error,
            "selected_fit_seconds": elapsed, "train_data_hash": state.train_data_hash,
            "fixture_fingerprint": fixture.fingerprint,
        })
    return rows


def evaluate_fixed_cell(
    fixture: SyntheticFixture,
    *,
    method_bandwidths: dict[str, float],
    train_stop: int,
    steps: int,
    learning_rate: float,
) -> list[dict[str, object]]:
    """Fit locked method bandwidths on train and score the untouched test block."""
    if not 0 < train_stop < len(fixture.sources):
        raise ValueError("train_stop must define non-empty contiguous train and test blocks")
    expected = {"cs_parzen", "matrix_tc_alpha2"}
    if set(method_bandwidths) != expected or any(not np.isfinite(value) or value <= 0 for value in method_bandwidths.values()):
        raise ValueError(f"method_bandwidths must provide positive values for {sorted(expected)}")
    state = fit_train_whitener(fixture.observed[:train_stop])
    whitened = apply_whitener(fixture.observed, state)
    train = torch.as_tensor(whitened[:train_stop], dtype=torch.float64)
    fitters = {"cs_parzen": fit_cs_parzen_orthogonal, "matrix_tc_alpha2": fit_matrix_tc_ica}
    rows = []
    for method, fitter in fitters.items():
        bandwidth = float(method_bandwidths[method])
        started = time.perf_counter()
        fit = fitter(
            train, [bandwidth] * train.shape[1], seed=fixture.seed,
            steps=steps, learning_rate=learning_rate,
        )
        elapsed = time.perf_counter() - started
        recovered = torch.as_tensor(whitened, dtype=torch.float64) @ fit.rotation.T
        test = matched_recovery(fixture.sources[train_stop:], recovered[train_stop:].numpy())
        rows.append({
            "method": method, "locked_bandwidth": bandwidth,
            "test_mean_absolute_correlation": test["mean_absolute_correlation"],
            "test_minimum_absolute_correlation": test["minimum_absolute_correlation"],
            "objective": fit.objective, "orthogonality_error": fit.orthogonality_error,
            "fit_seconds": elapsed, "train_data_hash": state.train_data_hash,
            "fixture_fingerprint": fixture.fingerprint,
        })
    return rows
