"""Pure-array mathematics for hierarchical clean and noisy Parzen ICA.

This module deliberately has no filesystem, experiment, or Spon-specific
dependencies. Arrays use component-first conventions where a demixer is
involved and all public functions validate shape, finiteness, and numerical
bounds before computing.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WhiteningResult:
    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    effective_rank: int
    identifiable: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ParzenDictionaryConfig:
    maximum_centers: int
    minimum_center_separation: float
    bandwidth: float
    bandwidth_min: float
    bandwidth_max: float
    update_rate: float
    replacement_policy: str
    warmup_samples: int
    seed: int

    def validate(self) -> None:
        numeric = (
            self.minimum_center_separation,
            self.bandwidth,
            self.bandwidth_min,
            self.bandwidth_max,
            self.update_rate,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("dictionary parameters must be finite")
        if self.maximum_centers < 2 or self.warmup_samples < 2:
            raise ValueError("dictionary size and warmup must be at least two")
        if self.minimum_center_separation < 0:
            raise ValueError("minimum center separation must be nonnegative")
        if not 0 < self.bandwidth_min <= self.bandwidth <= self.bandwidth_max:
            raise ValueError("dictionary bandwidth must lie within positive bounds")
        if not 0 <= self.update_rate <= 1:
            raise ValueError("dictionary update_rate must be in [0,1]")
        if self.replacement_policy not in {"farthest_center", "deterministic_reservoir"}:
            raise ValueError("unsupported dictionary replacement policy")


@dataclass(frozen=True)
class ParzenDictionaryState:
    centers: np.ndarray
    ages: np.ndarray
    usage: np.ndarray
    bandwidth: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DemixingFit:
    method_id: str
    demixing: np.ndarray
    converged: bool
    iterations: int
    objective: float | None
    gradient_norm: float | None
    update_count: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Stage1Result:
    background: np.ndarray
    dynamic_residual: np.ndarray
    differential_component: np.ndarray
    closure_residual: np.ndarray
    background_component: int | None
    confidence: float
    classification_status: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class NoiseModel:
    covariance: np.ndarray
    model_kind: str
    intensity_bins: np.ndarray | None
    variance_by_bin: np.ndarray | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SignalSubspace:
    basis: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    projected_noise_covariance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class NoisyParzenPosterior:
    noisy_output: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray | None
    score: np.ndarray
    projected_noise_variance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Stage2PatchResult:
    structured_reconstruction: np.ndarray
    residual: np.ndarray
    components: np.ndarray
    component_maps: np.ndarray
    component_classes: tuple[str, ...]
    diagnostics: dict[str, Any]


def _finite_array(value: np.ndarray, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("every Parzen evaluation must retain at least one center")
    shifted = np.exp(values - maximum)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(shifted, axis=axis))


def center_and_whiten_2d(
    samples: np.ndarray,
    *,
    eigenvalue_floor_ratio: float = 1e-6,
    condition_number_max: float = 1e8,
    covariance_mode: str = "ordinary",
) -> tuple[np.ndarray, WhiteningResult]:
    """Center and stably whiten a two-observation sample matrix ``[2,N]``."""
    values = _finite_array(samples, "samples", 2)
    if values.shape[0] != 2 or values.shape[1] < 3:
        raise ValueError("samples must have shape [2,N>=3]")
    if not 0 < eigenvalue_floor_ratio < 1 or condition_number_max <= 1:
        raise ValueError("invalid whitening floors or condition bound")
    if covariance_mode not in {"ordinary", "robust"}:
        raise ValueError("covariance_mode must be ordinary or robust")
    if covariance_mode == "ordinary":
        mean = values.mean(axis=1, keepdims=True)
        centered = values - mean
        covariance = centered @ centered.T / values.shape[1]
    else:
        mean = np.median(values, axis=1, keepdims=True)
        centered = values - mean
        radius = np.linalg.norm(centered, axis=0)
        cutoff = max(float(np.quantile(radius, 0.9)), np.finfo(float).eps)
        weights = np.minimum(1.0, cutoff / np.maximum(radius, np.finfo(float).eps))
        weighted = centered * np.sqrt(weights)[None, :]
        covariance = weighted @ weighted.T / max(float(weights.sum()), 1.0)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    largest = max(float(eigenvalues[0]), np.finfo(float).eps)
    floor = largest * float(eigenvalue_floor_ratio)
    floored = np.maximum(eigenvalues, floor)
    raw_condition = largest / max(float(eigenvalues[-1]), np.finfo(float).eps)
    effective_rank = int(np.sum(eigenvalues > floor))
    whitening = np.diag(1.0 / np.sqrt(floored)) @ vectors.T
    dewhitening = vectors @ np.diag(np.sqrt(floored))
    identifiable = bool(effective_rank == 2 and raw_condition <= condition_number_max)
    whitened = whitening @ centered
    result = WhiteningResult(
        mean=mean[:, 0],
        covariance=covariance,
        whitening=whitening,
        dewhitening=dewhitening,
        eigenvalues=eigenvalues,
        condition_number=float(raw_condition),
        effective_rank=effective_rank,
        identifiable=identifiable,
        diagnostics={
            "covariance_mode": covariance_mode,
            "eigenvalue_floor": float(floor),
            "floored_eigenvalues": floored.astype(float).tolist(),
            "condition_number_max": float(condition_number_max),
        },
    )
    return whitened, result


def _parzen_statistics(
    values: np.ndarray,
    centers: np.ndarray,
    variances: np.ndarray,
    exclude_center_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = _finite_array(values, "values")
    dictionary = _finite_array(centers, "centers", 1)
    if len(dictionary) < 1:
        raise ValueError("at least one center is required")
    variance = np.asarray(variances, dtype=np.float64)
    try:
        variance = np.broadcast_to(variance, observed.shape)
    except ValueError as exc:
        raise ValueError("variance must be scalar or broadcast to values") from exc
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("all Parzen variances must be finite and positive")
    flat = observed.ravel()
    flat_variance = variance.ravel()
    logits = (
        -0.5 * np.square(flat[:, None] - dictionary[None, :]) / flat_variance[:, None]
        -0.5 * np.log(2.0 * math.pi * flat_variance[:, None])
    )
    normalizer_count = len(dictionary)
    if exclude_center_indices is not None:
        excluded = np.asarray(exclude_center_indices, dtype=np.int64)
        if excluded.shape != observed.shape or len(dictionary) < 2:
            raise ValueError("leave-one-out indices must match values and retain a center")
        excluded = excluded.ravel()
        if np.any((excluded < 0) | (excluded >= len(dictionary))):
            raise ValueError("leave-one-out center index is out of bounds")
        logits[np.arange(len(flat)), excluded] = -np.inf
        normalizer_count -= 1
    log_normalizer = _logsumexp(logits, axis=1)
    responsibilities = np.exp(logits - log_normalizer[:, None])
    log_density = log_normalizer - math.log(normalizer_count)
    return (
        log_density.reshape(observed.shape),
        responsibilities.reshape(observed.shape + (len(dictionary),)),
    )


def parzen_responsibilities(
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    *,
    exclude_center_indices: np.ndarray | None = None,
) -> np.ndarray:
    bandwidth = _positive_scalar(bandwidth, "bandwidth")
    return _parzen_statistics(
        values, centers, np.asarray(bandwidth**2), exclude_center_indices
    )[1]


def gaussian_parzen_log_density(
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    *,
    exclude_center_indices: np.ndarray | None = None,
) -> np.ndarray:
    bandwidth = _positive_scalar(bandwidth, "bandwidth")
    return _parzen_statistics(
        values, centers, np.asarray(bandwidth**2), exclude_center_indices
    )[0]


def gaussian_parzen_score(
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    *,
    exclude_center_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the negative density score ``-d log(p)/dy``."""
    observed = _finite_array(values, "values")
    dictionary = _finite_array(centers, "centers", 1)
    bandwidth = _positive_scalar(bandwidth, "bandwidth")
    responsibility = parzen_responsibilities(
        observed, dictionary, bandwidth,
        exclude_center_indices=exclude_center_indices,
    )
    center_mean = np.sum(responsibility * dictionary, axis=-1)
    return (observed - center_mean) / bandwidth**2


def noisy_parzen_log_density(
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    noise_variance: np.ndarray | float,
) -> np.ndarray:
    observed = _finite_array(values, "values")
    bandwidth = _positive_scalar(bandwidth, "bandwidth")
    noise = np.asarray(noise_variance, dtype=np.float64)
    try:
        noise = np.broadcast_to(noise, observed.shape)
    except ValueError as exc:
        raise ValueError("noise variance must be scalar or broadcast to values") from exc
    if not np.isfinite(noise).all() or np.any(noise < 0):
        raise ValueError("noise variance must be finite and nonnegative")
    return _parzen_statistics(observed, centers, bandwidth**2 + noise)[0]


def noisy_parzen_score(
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    noise_variance: np.ndarray | float,
) -> np.ndarray:
    """Return the negative score for a Gaussian-noise-convolved dictionary."""
    observed = _finite_array(values, "values")
    dictionary = _finite_array(centers, "centers", 1)
    bandwidth = _positive_scalar(bandwidth, "bandwidth")
    noise = np.asarray(noise_variance, dtype=np.float64)
    try:
        noise = np.broadcast_to(noise, observed.shape)
    except ValueError as exc:
        raise ValueError("noise variance must be scalar or broadcast to values") from exc
    if not np.isfinite(noise).all() or np.any(noise < 0):
        raise ValueError("noise variance must be finite and nonnegative")
    _, responsibility = _parzen_statistics(observed, dictionary, bandwidth**2 + noise)
    center_mean = np.sum(responsibility * dictionary, axis=-1)
    return (observed - center_mean) / (bandwidth**2 + noise)


def noisy_parzen_posterior_mean(
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
    noise_variance: np.ndarray | float,
    *,
    return_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the Gaussian-mixture posterior mean of the clean scalar source."""
    observed = _finite_array(values, "values")
    dictionary = _finite_array(centers, "centers", 1)
    bandwidth = _positive_scalar(bandwidth, "bandwidth")
    noise = np.asarray(noise_variance, dtype=np.float64)
    try:
        noise = np.broadcast_to(noise, observed.shape)
    except ValueError as exc:
        raise ValueError("noise variance must be scalar or broadcast to values") from exc
    if not np.isfinite(noise).all() or np.any(noise < 0):
        raise ValueError("noise variance must be finite and nonnegative")
    source_variance = bandwidth**2
    total_variance = source_variance + noise
    _, responsibility = _parzen_statistics(observed, dictionary, total_variance)
    conditional_mean = (
        noise[..., None] * dictionary + source_variance * observed[..., None]
    ) / total_variance[..., None]
    posterior_mean = np.sum(responsibility * conditional_mean, axis=-1)
    if not return_variance:
        return posterior_mean
    conditional_variance = source_variance * noise / total_variance
    second_moment = np.sum(
        responsibility * (conditional_mean**2 + conditional_variance[..., None]),
        axis=-1,
    )
    posterior_variance = np.maximum(second_moment - posterior_mean**2, 0.0)
    return posterior_mean, posterior_variance


def symmetric_decorrelate(matrix: np.ndarray, *, eigenvalue_floor: float = 1e-8) -> np.ndarray:
    """Project a square matrix onto the row-orthogonal manifold."""
    values = _finite_array(matrix, "matrix", 2)
    floor = _positive_scalar(eigenvalue_floor, "eigenvalue_floor")
    if values.shape[0] != values.shape[1]:
        raise ValueError("demixing matrix must be square")
    gram = values @ values.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor))) @ vectors.T
    result = inverse_root @ values
    if not np.isfinite(result).all():
        raise FloatingPointError("decorrelation produced non-finite weights")
    return result


def initialize_parzen_dictionary(
    samples: np.ndarray,
    config: ParzenDictionaryConfig,
) -> ParzenDictionaryState:
    """Create a deterministic bounded farthest-point scalar dictionary."""
    config.validate()
    values = _finite_array(samples, "samples").ravel()
    warmup = values[: min(len(values), config.warmup_samples)]
    rng = np.random.default_rng(config.seed)
    tie_order = rng.permutation(len(warmup))
    first = int(np.argmin(np.abs(warmup - np.median(warmup))))
    selected = [first]
    while len(selected) < min(config.maximum_centers, len(warmup)):
        distances = np.min(
            np.abs(warmup[:, None] - warmup[np.asarray(selected)][None, :]), axis=1
        )
        distances[np.asarray(selected)] = -np.inf
        maximum = float(np.max(distances))
        if maximum < config.minimum_center_separation:
            break
        tied = set(np.flatnonzero(np.isclose(distances, maximum, rtol=0, atol=1e-15)))
        chosen = next(int(index) for index in tie_order if int(index) in tied)
        selected.append(chosen)
    centers = warmup[np.asarray(selected)].astype(np.float64)
    order = np.argsort(centers, kind="stable")
    centers = centers[order]
    return ParzenDictionaryState(
        centers=centers,
        ages=np.zeros(len(centers), dtype=np.int64),
        usage=np.zeros(len(centers), dtype=np.int64),
        bandwidth=np.asarray([config.bandwidth], dtype=np.float64),
        diagnostics={
            "initialized_samples": int(len(warmup)),
            "maximum_centers": int(config.maximum_centers),
            "seed": int(config.seed),
            "replacement_policy": config.replacement_policy,
            "replacements": 0,
            "updates": 0,
        },
    )


def update_parzen_dictionary(
    state: ParzenDictionaryState,
    posterior_samples: np.ndarray,
    config: ParzenDictionaryConfig,
) -> ParzenDictionaryState:
    """Update a scalar dictionary using clean posterior samples only."""
    config.validate()
    values = _finite_array(posterior_samples, "posterior_samples").ravel()
    centers = _finite_array(state.centers, "state.centers", 1).copy()
    ages = np.asarray(state.ages, dtype=np.int64).copy()
    usage = np.asarray(state.usage, dtype=np.int64).copy()
    if ages.shape != centers.shape or usage.shape != centers.shape:
        raise ValueError("dictionary centers, ages, and usage must align")
    replacements = int(state.diagnostics.get("replacements", 0))
    updates = int(state.diagnostics.get("updates", 0))
    rng = np.random.default_rng(config.seed + updates)
    seen = int(np.sum(usage)) + len(centers)
    for sample in values:
        ages += 1
        distances = np.abs(centers - sample)
        nearest = int(np.argmin(distances))
        if distances[nearest] < config.minimum_center_separation:
            centers[nearest] = (
                (1.0 - config.update_rate) * centers[nearest]
                + config.update_rate * sample
            )
            usage[nearest] += 1
            ages[nearest] = 0
        elif len(centers) < config.maximum_centers:
            centers = np.append(centers, sample)
            ages = np.append(ages, 0)
            usage = np.append(usage, 1)
        elif config.replacement_policy == "farthest_center":
            victim = min(range(len(centers)), key=lambda i: (usage[i], -ages[i], i))
            centers[victim] = sample
            usage[victim] = 1
            ages[victim] = 0
            replacements += 1
        else:
            seen += 1
            draw = int(rng.integers(0, seen))
            if draw < len(centers):
                centers[draw] = sample
                usage[draw] = 1
                ages[draw] = 0
                replacements += 1
        updates += 1
    order = np.argsort(centers, kind="stable")
    return ParzenDictionaryState(
        centers=centers[order],
        ages=ages[order],
        usage=usage[order],
        bandwidth=np.asarray(state.bandwidth, dtype=np.float64).copy(),
        diagnostics={
            **state.diagnostics,
            "replacements": replacements,
            "updates": updates,
            "center_count": int(len(centers)),
        },
    )


def fit_stochastic_parzen_ica(
    whitened: np.ndarray,
    dictionary_config: ParzenDictionaryConfig,
    *,
    initial_demixing: np.ndarray | None = None,
    learning_rate: float = 1e-3,
    gradient_clip: float = 5.0,
    maximum_angle_update_degrees: float = 1.0,
    batch_size: int = 128,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
    decorrelation_floor: float = 1e-8,
) -> tuple[DemixingFit, tuple[ParzenDictionaryState, ...]]:
    """Fit a bounded natural-gradient Parzen demixer on whitened samples."""
    z = _finite_array(whitened, "whitened", 2)
    rank, sample_count = z.shape
    if rank < 2 or sample_count < rank + 1:
        raise ValueError("whitened must have shape [components>=2,samples]")
    if min(learning_rate, gradient_clip, maximum_angle_update_degrees, tolerance) <= 0:
        raise ValueError("optimization bounds must be positive")
    if batch_size < 2 or maximum_iterations < 1:
        raise ValueError("batch_size and maximum_iterations are invalid")
    dictionary_config.validate()
    if initial_demixing is None:
        demixing = np.eye(rank, dtype=np.float64)
        initialization = "identity"
    else:
        initial_weights = _finite_array(
            initial_demixing, "initial_demixing", 2
        )
        if initial_weights.shape != (rank, rank):
            raise ValueError("initial_demixing must match the whitened rank")
        demixing = symmetric_decorrelate(
            initial_weights, eigenvalue_floor=decorrelation_floor
        )
        initialization = "provided_symmetric_decorrelation"
    initial = demixing @ z
    dictionaries = tuple(
        initialize_parzen_dictionary(initial[index], dictionary_config)
        for index in range(rank)
    )
    update_count = 0
    accepted = 0
    angle_caps = 0
    final_gradient_norm = None
    objective = None
    converged = False
    for iteration in range(1, maximum_iterations + 1):
        previous = demixing.copy()
        objective_terms = []
        for start in range(0, sample_count, batch_size):
            stop = min(sample_count, start + batch_size)
            batch = z[:, start:stop]
            outputs = demixing @ batch
            scores = np.vstack([
                gaussian_parzen_score(
                    outputs[index], dictionaries[index].centers,
                    float(dictionaries[index].bandwidth[0]),
                )
                for index in range(rank)
            ])
            natural = np.eye(rank) - scores @ outputs.T / outputs.shape[1]
            gradient = natural @ demixing
            gradient_norm = float(np.linalg.norm(gradient))
            final_gradient_norm = gradient_norm
            if gradient_norm > gradient_clip:
                gradient *= gradient_clip / gradient_norm
            candidate = symmetric_decorrelate(
                demixing + learning_rate * gradient,
                eigenvalue_floor=decorrelation_floor,
            )
            row_cosines = np.clip(
                np.abs(np.sum(candidate * demixing, axis=1)), 0.0, 1.0
            )
            maximum_angle = float(np.degrees(np.max(np.arccos(row_cosines))))
            if maximum_angle > maximum_angle_update_degrees:
                ratio = maximum_angle_update_degrees / maximum_angle
                candidate = symmetric_decorrelate(
                    demixing + ratio * (candidate - demixing),
                    eigenvalue_floor=decorrelation_floor,
                )
                angle_caps += 1
            if not np.isfinite(candidate).all():
                continue
            demixing = candidate
            accepted += 1
            update_count += 1
            outputs = demixing @ batch
            updated = []
            for index in range(rank):
                updated.append(update_parzen_dictionary(
                    dictionaries[index], outputs[index], dictionary_config
                ))
                objective_terms.append(float(np.mean(gaussian_parzen_log_density(
                    outputs[index], updated[-1].centers,
                    float(updated[-1].bandwidth[0]),
                ))))
            dictionaries = tuple(updated)
        objective = float(np.sum(objective_terms) / max(1, len(objective_terms)))
        delta = float(np.max(np.abs(np.abs(np.diag(demixing @ previous.T)) - 1.0)))
        if delta <= tolerance:
            converged = True
            break
    orthogonality = float(np.linalg.norm(demixing @ demixing.T - np.eye(rank)))
    fit = DemixingFit(
        method_id="stochastic_parzen_score",
        demixing=demixing,
        converged=converged,
        iterations=iteration,
        objective=objective,
        gradient_norm=final_gradient_norm,
        update_count=update_count,
        diagnostics={
            "accepted_updates": accepted,
            "angle_cap_count": angle_caps,
            "orthogonality_error": orthogonality,
            "batch_size": int(batch_size),
            "sample_count": int(sample_count),
            "score_sign": "negative_log_density_derivative",
            "initialization": initialization,
        },
    )
    return fit, dictionaries


def track_demixing_components(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Align current demixing rows to previous rows by absolute cosine."""
    left = _finite_array(previous, "previous", 2)
    right = _finite_array(current, "current", 2)
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("previous/current demixers must be equal square matrices")
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    similarity = left_norm @ right_norm.T
    candidates = list(permutations(range(len(left))))
    assignment = max(
        candidates,
        key=lambda perm: (
            sum(abs(similarity[i, perm[i]]) for i in range(len(left))),
            tuple(-x for x in perm),
        ),
    )
    signs = tuple(1 if similarity[i, assignment[i]] >= 0 else -1 for i in range(len(left)))
    aligned = np.vstack([
        right[assignment[index]] * signs[index] for index in range(len(left))
    ])
    matched = [abs(float(similarity[i, assignment[i]])) for i in range(len(left))]
    return aligned, tuple(int(x) for x in assignment), signs, {
        "matched_absolute_cosines": matched,
        "minimum_absolute_cosine": min(matched),
        "permutation_changed": assignment != tuple(range(len(left))),
        "sign_flip_count": sum(sign < 0 for sign in signs),
    }


def component_derivative_energy(components: np.ndarray) -> dict[str, np.ndarray]:
    """Return robust normalized first/second difference energy per component."""
    values = _finite_array(components, "components", 2)
    if values.shape[1] < 3:
        raise ValueError("components require at least three temporal samples")
    variance = np.var(values, axis=1)
    denominator = np.maximum(variance, np.finfo(float).eps)
    first = np.median(np.diff(values, axis=1) ** 2, axis=1) / denominator
    second = np.median(np.diff(values, n=2, axis=1) ** 2, axis=1) / denominator
    return {"first_difference_energy": first, "second_difference_energy": second}


def projected_noise_variance(
    demixing: np.ndarray,
    projected_noise_covariance: np.ndarray,
    *,
    variance_floor: float = 1e-6,
    variance_ceiling: float = 100.0,
) -> np.ndarray:
    weights = _finite_array(demixing, "demixing", 2)
    covariance = _finite_array(
        projected_noise_covariance, "projected_noise_covariance", 2
    )
    if covariance.shape != (weights.shape[1], weights.shape[1]):
        raise ValueError("noise covariance must match demixing input rank")
    if not 0 < variance_floor <= variance_ceiling:
        raise ValueError("invalid projected-noise variance bounds")
    raw = np.einsum("ij,jk,ik->i", weights, covariance, weights)
    return np.clip(raw, variance_floor, variance_ceiling)


def decomposition_closure(
    observation: np.ndarray,
    background_like: np.ndarray,
    structured_neural_signal: np.ndarray,
    structured_artifact: np.ndarray,
    measurement_noise: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return explicit B/S/A/N closure residual and normalized diagnostics."""
    arrays = [
        _finite_array(observation, "observation"),
        _finite_array(background_like, "background_like"),
        _finite_array(structured_neural_signal, "structured_neural_signal"),
        _finite_array(structured_artifact, "structured_artifact"),
        _finite_array(measurement_noise, "measurement_noise"),
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all decomposition channels must have identical shape")
    residual = arrays[0] - sum(arrays[1:])
    denominator = max(float(np.sum(arrays[0] ** 2)), np.finfo(float).eps)
    return residual, {
        "normalized_squared_error": float(np.sum(residual**2) / denominator),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "p99_absolute_error": float(np.quantile(np.abs(residual), 0.99)),
    }


def fit_batch_cs_parzen_2d(
    screen_whitened: np.ndarray,
    confirm_whitened: np.ndarray | None = None,
    *,
    bandwidth: float = 0.35,
    block_rows: int = 256,
    screen_step_degrees: float = 3.0,
    refine_half_width_degrees: float = 3.0,
    refine_step_degrees: float = 0.25,
) -> DemixingFit:
    """Wrap the established bounded two-dimensional batch CS-Parzen reference."""
    screen = _finite_array(screen_whitened, "screen_whitened", 2)
    confirm = screen if confirm_whitened is None else _finite_array(
        confirm_whitened, "confirm_whitened", 2
    )
    if screen.shape[0] != 2 or confirm.shape[0] != 2:
        raise ValueError("batch CS-Parzen inputs must have shape [2,N]")
    from neurobench.algorithms.pairwise_separation import fit_cs_parzen_ica

    legacy = fit_cs_parzen_ica(
        screen,
        confirm,
        bandwidth=bandwidth,
        block_rows=block_rows,
        screen_step_degrees=screen_step_degrees,
        refine_half_width_degrees=refine_half_width_degrees,
        refine_step_degrees=refine_step_degrees,
    )
    return DemixingFit(
        method_id="batch_cs_parzen_pairwise",
        demixing=np.asarray(legacy.demixing, dtype=np.float64),
        converged=bool(legacy.converged),
        iterations=int(legacy.iterations),
        objective=None if legacy.objective is None else float(legacy.objective),
        gradient_norm=None,
        update_count=0,
        diagnostics={
            **legacy.diagnostics,
            "compatibility_source": "neurobench.algorithms.pairwise_separation",
            "bounded_kernel_blocks": True,
        },
    )


def component_staticness_score(
    component_series: np.ndarray,
    observation_mixing: np.ndarray,
    *,
    alpha_gain: float = 1.0,
    global_intensity: np.ndarray | None = None,
    first_difference_weight: float = 1.0,
    second_difference_weight: float = 0.5,
    common_direction_weight: float = 1.0,
    spatial_high_frequency_weight: float = 0.25,
    global_intensity_weight: float = 0.25,
    minimum_confidence_margin: float = 0.1,
) -> dict[str, Any]:
    """Select a background-like component without using labels.

    ``component_series`` has shape ``[2,T,...]`` and ``observation_mixing``
    has observation coordinates in rows and components in columns.
    """
    values = np.asarray(component_series, dtype=np.float64)
    mixing = _finite_array(observation_mixing, "observation_mixing", 2)
    if values.ndim < 2 or values.shape[0] != 2 or values.shape[1] < 3:
        raise ValueError("component_series must have shape [2,T>=3,...]")
    if not np.isfinite(values).all() or mixing.shape != (2, 2):
        raise ValueError("component series and two-component mixing must be finite")
    alpha = _positive_scalar(alpha_gain, "alpha_gain")
    weights = np.asarray([
        first_difference_weight,
        second_difference_weight,
        common_direction_weight,
        spatial_high_frequency_weight,
        global_intensity_weight,
        minimum_confidence_margin,
    ], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("staticness weights and margin must be finite and nonnegative")

    reduce_axes = tuple(range(1, values.ndim))
    variance = np.var(values, axis=reduce_axes)
    denominator = np.maximum(variance, np.finfo(float).eps)
    first = np.median(np.diff(values, axis=1) ** 2, axis=tuple(range(1, values.ndim)))
    second = np.median(
        np.diff(values, n=2, axis=1) ** 2,
        axis=tuple(range(1, values.ndim)),
    )
    first /= denominator
    second /= denominator

    common = np.asarray([1.0, alpha], dtype=np.float64)
    differential = np.asarray([-alpha, 1.0], dtype=np.float64)
    common /= np.linalg.norm(common)
    differential /= np.linalg.norm(differential)
    normalized_mixing = mixing / np.maximum(
        np.linalg.norm(mixing, axis=0, keepdims=True), 1e-12
    )
    common_cosine = np.abs(common @ normalized_mixing)
    differential_cosine = np.abs(differential @ normalized_mixing)

    if values.ndim >= 4:
        spatial = np.median(np.abs(values), axis=1)
        dy = np.mean(np.abs(np.diff(spatial, axis=-2)), axis=tuple(range(1, spatial.ndim)))
        dx = np.mean(np.abs(np.diff(spatial, axis=-1)), axis=tuple(range(1, spatial.ndim)))
        scale = np.maximum(
            np.mean(np.abs(spatial), axis=tuple(range(1, spatial.ndim))), 1e-12
        )
        high_frequency = (dy + dx) / scale
        flat = spatial.reshape(2, -1)
        center = np.median(flat, axis=1)
        mad = 1.4826 * np.median(np.abs(flat - center[:, None]), axis=1)
        support_fraction = np.mean(
            flat > center[:, None] + np.maximum(mad[:, None], 1e-12), axis=1
        )
    else:
        high_frequency = np.zeros(2, dtype=np.float64)
        support_fraction = np.ones(2, dtype=np.float64)

    if global_intensity is None:
        global_correlation = np.zeros(2, dtype=np.float64)
    else:
        global_trace = _finite_array(global_intensity, "global_intensity", 1)
        if len(global_trace) != values.shape[1]:
            raise ValueError("global_intensity must align with the temporal axis")
        component_trace = values.reshape(2, values.shape[1], -1).mean(axis=2)
        global_correlation = np.zeros(2, dtype=np.float64)
        for index in range(2):
            if np.std(component_trace[index]) > 0 and np.std(global_trace) > 0:
                global_correlation[index] = abs(float(np.corrcoef(
                    component_trace[index], global_trace
                )[0, 1]))

    scores = (
        -first_difference_weight * np.log1p(first)
        -second_difference_weight * np.log1p(second)
        +common_direction_weight * common_cosine
        -spatial_high_frequency_weight * np.log1p(high_frequency)
        +global_intensity_weight * global_correlation
    )
    order = np.argsort(scores)[::-1]
    margin = float(scores[order[0]] - scores[order[1]])
    resolved = margin >= minimum_confidence_margin
    selected = int(order[0]) if resolved else None
    terms = []
    for index in range(2):
        terms.append({
            "component": index,
            "first_difference_energy": float(first[index]),
            "second_difference_energy": float(second[index]),
            "common_direction_cosine": float(common_cosine[index]),
            "differential_direction_cosine": float(differential_cosine[index]),
            "spatial_high_frequency": float(high_frequency[index]),
            "spatial_support_fraction": float(support_fraction[index]),
            "global_intensity_correlation": float(global_correlation[index]),
            "staticness_score": float(scores[index]),
        })
    return {
        "background_component": selected,
        "background_confidence": margin,
        "classification_status": "resolved" if resolved else "unresolved",
        "classification_terms": terms,
        "minimum_confidence_margin": float(minimum_confidence_margin),
        "labels_used": False,
    }


def reconstruct_selected_component(
    samples: np.ndarray,
    whitening: WhiteningResult,
    demixing: np.ndarray,
    component: int,
) -> np.ndarray:
    """Reconstruct one component in the original two-observation coordinates."""
    observations = _finite_array(samples, "samples", 2)
    weights = _finite_array(demixing, "demixing", 2)
    if observations.shape[0] != 2 or weights.shape != (2, 2):
        raise ValueError("Stage-1 reconstruction requires two observations/components")
    if component not in {0, 1}:
        raise ValueError("component must be zero or one")
    if whitening.mean.shape != (2,) or whitening.whitening.shape != (2, 2):
        raise ValueError("whitening result is incompatible with two observations")
    whitened = whitening.whitening @ (observations - whitening.mean[:, None])
    outputs = weights @ whitened
    selected = np.zeros_like(outputs)
    selected[component] = outputs[component]
    inverse = np.linalg.pinv(weights)
    centered = whitening.dewhitening @ inverse @ selected
    reconstructed = whitening.mean[:, None] + centered
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError("selected-component reconstruction is non-finite")
    return reconstructed


def stage1_background_residual(
    samples: np.ndarray,
    whitening: WhiteningResult,
    demixing: np.ndarray,
    background_component: int | None,
    *,
    confidence: float,
    subtraction_mode: str = "exact",
    method_id: str = "stage1",
) -> Stage1Result:
    """Reconstruct current-frame background and an amplitude-preserving residual."""
    observations = _finite_array(samples, "samples", 2)
    weights = _finite_array(demixing, "demixing", 2)
    if observations.shape[0] != 2 or weights.shape != (2, 2):
        raise ValueError("Stage 1 requires aligned two-observation samples")
    if subtraction_mode not in {"exact", "confidence_weighted", "no_subtraction"}:
        raise ValueError("unsupported Stage-1 subtraction mode")
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or confidence_value < 0:
        raise ValueError("background confidence must be finite and nonnegative")
    whitened = whitening.whitening @ (observations - whitening.mean[:, None])
    outputs = weights @ whitened
    if background_component is None or subtraction_mode == "no_subtraction":
        background = np.zeros(observations.shape[1], dtype=np.float64)
        residual = observations[1].copy()
        differential = observations[1] - observations[0]
        status = "unresolved" if background_component is None else "resolved"
        applied_weight = 0.0
    else:
        reconstructed = reconstruct_selected_component(
            observations, whitening, weights, background_component
        )
        applied_weight = 1.0 if subtraction_mode == "exact" else min(confidence_value, 1.0)
        background = applied_weight * reconstructed[1]
        residual = observations[1] - background
        differential = outputs[1 - background_component]
        status = "resolved"
    closure = observations[1] - background - residual
    return Stage1Result(
        background=background.astype(np.float32),
        dynamic_residual=residual.astype(np.float32),
        differential_component=differential.astype(np.float32),
        closure_residual=closure.astype(np.float32),
        background_component=background_component,
        confidence=confidence_value,
        classification_status=status,
        diagnostics={
            "method_id": method_id,
            "subtraction_mode": subtraction_mode,
            "applied_background_weight": float(applied_weight),
            "closure_max_absolute": float(np.max(np.abs(closure))),
            "axes": "P",
            "labels_used": False,
        },
    )


def stage1_recursive_background_residual(
    frames: np.ndarray,
    lag_frames: int,
    whitening: WhiteningResult,
    demixing: np.ndarray,
    background_component: int | None,
    *,
    confidence: float,
    subtraction_mode: str = "exact",
    method_id: str = "stage1",
) -> Stage1Result:
    """Apply a frozen Stage-1 model against its prior background estimate.

    Feeding the previous raw observation back into a pairwise model makes a
    sustained event become background after one frame. This causal recurrence
    instead uses ``[estimated_background(t-k), observation(t)]`` and therefore
    retains the event amplitude in the current-frame residual.
    """
    observations = _finite_array(frames, "frames", 3)
    lag = int(lag_frames)
    if not 1 <= lag < len(observations) - 2:
        raise ValueError("lag_frames must leave at least three output frames")
    weights = _finite_array(demixing, "demixing", 2)
    if weights.shape != (2, 2):
        raise ValueError("Stage-1 recursive inference requires two components")
    if subtraction_mode not in {"exact", "confidence_weighted", "no_subtraction"}:
        raise ValueError("unsupported Stage-1 subtraction mode")
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or confidence_value < 0:
        raise ValueError("background confidence must be finite and nonnegative")

    output_shape = observations[lag:].shape
    if background_component is None or subtraction_mode == "no_subtraction":
        background = np.zeros(output_shape, dtype=np.float64)
        residual = observations[lag:].copy()
        differential = observations[lag:] - observations[:-lag]
        status = "unresolved" if background_component is None else "resolved"
        applied_weight = 0.0
    else:
        applied_weight = (
            1.0 if subtraction_mode == "exact" else min(confidence_value, 1.0)
        )
        background_state = np.empty_like(observations)
        background_state[:lag] = observations[:lag]
        background = np.empty(output_shape, dtype=np.float64)
        residual = np.empty(output_shape, dtype=np.float64)
        differential = np.empty(output_shape, dtype=np.float64)
        for output_index, frame_index in enumerate(range(lag, len(observations))):
            pair = np.stack(
                (
                    background_state[frame_index - lag].reshape(-1),
                    observations[frame_index].reshape(-1),
                ),
                axis=0,
            )
            reconstructed = reconstruct_selected_component(
                pair, whitening, weights, background_component
            )
            model_background = reconstructed[1].reshape(observations.shape[1:])
            current_background = applied_weight * model_background
            background_state[frame_index] = model_background
            background[output_index] = current_background
            residual[output_index] = observations[frame_index] - current_background
            whitened = whitening.whitening @ (
                pair - whitening.mean[:, None]
            )
            outputs = weights @ whitened
            differential[output_index] = outputs[
                1 - background_component
            ].reshape(observations.shape[1:])
        status = "resolved"
    closure = observations[lag:] - background - residual
    return Stage1Result(
        background=background.astype(np.float32),
        dynamic_residual=residual.astype(np.float32),
        differential_component=differential.astype(np.float32),
        closure_residual=closure.astype(np.float32),
        background_component=background_component,
        confidence=confidence_value,
        classification_status=status,
        diagnostics={
            "method_id": method_id,
            "subtraction_mode": subtraction_mode,
            "applied_background_weight": float(applied_weight),
            "closure_max_absolute": float(np.max(np.abs(closure))),
            "axes": "TYX",
            "labels_used": False,
            "inference_mode": "recursive_background_state",
            "initial_state": "first_lag_observations",
            "causal_status": "causal_frozen_model",
        },
    )
