"""Bounded multi-lag and delay-embedding MSICA objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from neurobench.algorithms.information_source_separation import (
    knn_mutual_information,
    normalized_hsic,
    pca_whiten,
)
from neurobench.algorithms.pairwise_separation import cs_parzen_objective


@dataclass(frozen=True)
class TemporalMSICAFit:
    formulation: str
    objective_family: str
    objective_parameter: dict[str, float | int]
    lags: tuple[int, ...]
    lag_weights: tuple[float, ...]
    center: np.ndarray
    whitening: np.ndarray
    rotation: np.ndarray
    demixing: np.ndarray
    objective: float
    baseline_objective: float
    persistence_index: int
    innovation_index: int
    residual_indices: tuple[int, ...]
    component_signs: tuple[int, ...]
    converged: bool
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "objective_family": self.objective_family,
            "objective_parameter": self.objective_parameter,
            "lags": list(self.lags),
            "lag_weights": list(self.lag_weights),
            "center": self.center.tolist(),
            "whitening": self.whitening.tolist(),
            "rotation": self.rotation.tolist(),
            "demixing": self.demixing.tolist(),
            "objective": self.objective,
            "baseline_objective": self.baseline_objective,
            "persistence_index": self.persistence_index,
            "innovation_index": self.innovation_index,
            "residual_indices": list(self.residual_indices),
            "component_signs": list(self.component_signs),
            "converged": self.converged,
            "diagnostics": self.diagnostics,
        }


def lag_weights(lags: Sequence[int], decay: float) -> np.ndarray:
    values = np.asarray(tuple(int(item) for item in lags), dtype=np.float64)
    if values.ndim != 1 or not len(values) or values[0] != 0 or np.any(np.diff(values) <= 0):
        raise ValueError("lags must be unique increasing integers beginning at zero")
    if not 0 <= float(decay) <= 1:
        raise ValueError("decay must lie in [0,1]")
    weights = np.ones(len(values), dtype=np.float64) if float(decay) == 0 else np.power(float(decay), values)
    return weights / weights.sum()


def sample_anchor_indices(shape: tuple[int, int, int], *, history: int, count: int, seed: int) -> np.ndarray:
    frames, height, width = map(int, shape)
    if not 1 <= int(history) < frames or count < 1:
        raise ValueError("invalid anchor sampling request")
    total = (frames - history) * height * width
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(total, size=min(int(count), total), replace=False))
    t0, pixel = np.divmod(selected, height * width)
    return np.column_stack((t0 + history, pixel // width, pixel % width)).astype(np.int32)


def gather_multilag_pairs(values: Any, anchors: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    t, y, x = np.asarray(anchors, dtype=np.int32).T
    result = []
    for lag in tuple(int(item) for item in lags):
        if lag < 0 or np.any(t - lag - 1 < 0):
            raise ValueError("anchor lacks multi-lag pair history")
        result.append(np.column_stack((np.asarray(values[t - lag - 1, y, x]), np.asarray(values[t - lag, y, x]))))
    array = np.asarray(result, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("multi-lag samples must be finite")
    return array


def gather_delay_embedding(values: Any, anchors: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    t, y, x = np.asarray(anchors, dtype=np.int32).T
    rows = []
    for lag in tuple(int(item) for item in lags):
        if lag < 0 or np.any(t - lag < 0):
            raise ValueError("anchor lacks embedding history")
        rows.append(np.asarray(values[t - lag, y, x]))
    array = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("embedding samples must be finite")
    return array


def _median_bandwidth(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if len(sorted_values) > 512:
        sorted_values = sorted_values[np.linspace(0, len(sorted_values) - 1, 512).astype(np.int32)]
    distances = np.abs(sorted_values[:, None] - sorted_values[None, :])
    positive = distances[distances > 0]
    return max(float(np.median(positive)) if positive.size else 1.0, 1e-6)


def matrix_renyi_entropy(values: np.ndarray, *, alpha: float, bandwidth_scale: float) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    if len(x) < 8 or not np.isfinite(x).all() or alpha <= 0 or abs(alpha - 1.0) < 1e-8 or bandwidth_scale <= 0:
        raise ValueError("invalid matrix-Renyi entropy inputs")
    bandwidth = _median_bandwidth(x) * float(bandwidth_scale)
    gram = np.exp(-0.5 * ((x[:, None] - x[None, :]) / bandwidth) ** 2)
    gram /= max(float(np.trace(gram)), np.finfo(float).eps)
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    eigenvalues /= max(float(eigenvalues.sum()), np.finfo(float).eps)
    if abs(float(alpha) - 1.0) < 0.05:
        positive = eigenvalues[eigenvalues > np.finfo(float).eps]
        return float(-np.sum(positive * np.log(positive)))
    return float(np.log(max(float(np.sum(eigenvalues ** float(alpha))), np.finfo(float).tiny)) / (1.0 - float(alpha)))


def matrix_renyi_mutual_information(left: np.ndarray, right: np.ndarray, *, alpha: float, bandwidth_scale: float) -> float:
    x = np.asarray(left, dtype=np.float64).ravel(); y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape or len(x) < 8:
        raise ValueError("matrix-Renyi MI inputs must align")
    def gram(values: np.ndarray) -> np.ndarray:
        bandwidth = _median_bandwidth(values) * float(bandwidth_scale)
        result = np.exp(-0.5 * ((values[:, None] - values[None, :]) / bandwidth) ** 2)
        return result / max(float(np.trace(result)), np.finfo(float).eps)
    def entropy(matrix: np.ndarray) -> float:
        eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
        eigenvalues /= max(float(eigenvalues.sum()), np.finfo(float).eps)
        if abs(float(alpha) - 1.0) < 0.05:
            positive = eigenvalues[eigenvalues > np.finfo(float).eps]
            return float(-np.sum(positive * np.log(positive)))
        return float(np.log(max(float(np.sum(eigenvalues ** float(alpha))), np.finfo(float).tiny)) / (1.0 - float(alpha)))
    gx, gy = gram(x), gram(y)
    joint = gx * gy
    joint /= max(float(np.trace(joint)), np.finfo(float).eps)
    return float(max(0.0, entropy(gx) + entropy(gy) - entropy(joint)))


def dependence_function(family: str, parameter: dict[str, float | int]) -> Callable[[np.ndarray, np.ndarray], float]:
    if family == "cs_parzen":
        bandwidth = float(parameter["bandwidth"])
        return lambda a, b: float(cs_parzen_objective(np.column_stack((a, b)), bandwidth, block_rows=256, kernel_dtype=np.float32).objective)
    if family == "ksg_mi":
        neighbors = int(parameter["neighbors"])
        return lambda a, b: knn_mutual_information(a, b, neighbors=neighbors)
    if family == "normalized_hsic":
        scale = float(parameter["bandwidth_scale"])
        return lambda a, b: normalized_hsic(a, b, bandwidth_scale=scale)
    if family == "matrix_renyi_mi":
        alpha, scale = float(parameter["alpha"]), float(parameter["bandwidth_scale"])
        return lambda a, b: matrix_renyi_mutual_information(a, b, alpha=alpha, bandwidth_scale=scale)
    raise ValueError(f"unknown dependence family: {family}")


def _rotation_2d(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_degrees)); cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)


def fit_multilag_2d(
    screen_pairs: np.ndarray,
    confirmation_pairs: np.ndarray,
    *,
    lags: Sequence[int],
    weights: Sequence[float],
    objective_family: str,
    objective_parameter: dict[str, float | int],
    coarse_step_degrees: float = 3.0,
    refine_half_width_degrees: float = 3.0,
    refine_step_degrees: float = 0.25,
    sharpness_delta_degrees: float = 2.0,
    eigenvalue_floor_ratio: float = 1e-6,
) -> TemporalMSICAFit:
    screen = np.asarray(screen_pairs, dtype=np.float64); confirm = np.asarray(confirmation_pairs, dtype=np.float64)
    lag_tuple = tuple(int(item) for item in lags); weight_array = np.asarray(weights, dtype=np.float64)
    if (
        screen.ndim != 3
        or screen.shape[0] != len(lag_tuple)
        or screen.shape[2] != 2
        or confirm.ndim != 3
        or confirm.shape[0] != len(lag_tuple)
        or confirm.shape[2] != 2
        or weight_array.shape != (len(lag_tuple),)
    ):
        raise ValueError("multi-lag pairs must be [lags,samples,2]")
    current = screen[0]
    mean = current.mean(axis=0)
    covariance = np.cov((current - mean).T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(covariance); order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    floor = max(float(eigvals[0]) * float(eigenvalue_floor_ratio), np.finfo(float).eps)
    whitening = np.diag(1.0 / np.sqrt(np.maximum(eigvals, floor))) @ eigvecs.T
    dep = dependence_function(objective_family, objective_parameter)
    def whiten(pairs: np.ndarray) -> np.ndarray:
        return np.asarray([whitening @ (pairs[index].T - mean[:, None]) for index in range(len(lag_tuple))])
    ws, wc = whiten(screen), whiten(confirm)
    def score(array: np.ndarray, angle: float) -> float:
        outputs = np.asarray([_rotation_2d(angle) @ array[index] for index in range(len(lag_tuple))])
        current_out = outputs[0]
        terms = []
        for index in range(len(lag_tuple)):
            value = 0.5 * (dep(current_out[0], outputs[index, 1]) + dep(current_out[1], outputs[index, 0]))
            terms.append(float(value))
        return float(np.dot(weight_array, terms))
    coarse = np.arange(0.0, 90.0, float(coarse_step_degrees))
    coarse_rows = [(float(angle), score(ws, float(angle))) for angle in coarse]
    best_coarse = min(coarse_rows, key=lambda row: row[1])[0]
    refine = sorted(set(float(item % 90.0) for item in np.arange(best_coarse - refine_half_width_degrees, best_coarse + refine_half_width_degrees + refine_step_degrees / 2, refine_step_degrees)))
    screen_rows = [(angle, score(ws, angle)) for angle in refine]
    best_screen = min(screen_rows, key=lambda row: row[1])[0]
    candidates = sorted(set(float((best_screen + delta) % 90.0) for delta in (-refine_step_degrees, 0.0, refine_step_degrees)))
    confirm_rows = [(angle, score(wc, angle)) for angle in candidates]
    angle, objective = min(confirm_rows, key=lambda row: row[1])
    baseline = score(wc, 0.0)
    sharp_left = score(wc, (angle - sharpness_delta_degrees) % 90.0)
    sharp_right = score(wc, (angle + sharpness_delta_degrees) % 90.0)
    rotation = _rotation_2d(angle)
    effective = rotation @ whitening
    common = np.asarray([1.0, 1.0]) / np.sqrt(2.0); derivative = np.asarray([-1.0, 1.0]) / np.sqrt(2.0)
    normalized = effective / np.maximum(np.linalg.norm(effective, axis=1, keepdims=True), 1e-12)
    common_score = np.abs(normalized @ common); persistence = int(np.argmax(common_score)); innovation = 1 - persistence
    signs = [1, 1]
    signs[persistence] = 1 if normalized[persistence] @ common >= 0 else -1
    signs[innovation] = 1 if normalized[innovation] @ derivative >= 0 else -1
    signed_rotation = np.diag(signs) @ rotation
    return TemporalMSICAFit(
        formulation="multilag_2d", objective_family=objective_family, objective_parameter=dict(objective_parameter),
        lags=lag_tuple, lag_weights=tuple(float(item) for item in weight_array), center=mean, whitening=whitening,
        rotation=signed_rotation, demixing=signed_rotation @ whitening, objective=float(objective), baseline_objective=float(baseline),
        persistence_index=persistence, innovation_index=innovation, residual_indices=(), component_signs=tuple(signs), converged=True,
        diagnostics={"selected_angle_degrees": angle, "coarse_rows": coarse_rows, "refine_rows": screen_rows, "confirmation_rows": confirm_rows,
                     "sharpness_delta_degrees": float(sharpness_delta_degrees), "sharpness_margin": float(min(sharp_left, sharp_right) - objective),
                     "condition_number": float(max(eigvals) / max(min(eigvals), floor))},
    )


def fit_delay_embedding(
    observations: np.ndarray,
    confirmation_observations: np.ndarray | None = None,
    *,
    lags: Sequence[int],
    objective_family: str,
    objective_parameter: dict[str, float | int],
    angle_step_degrees: float = 5.0,
    max_sweeps: int = 6,
    improvement_tolerance: float = 1e-4,
    eigenvalue_floor_ratio: float = 1e-6,
) -> TemporalMSICAFit:
    values = np.asarray(observations, dtype=np.float64)
    lag_tuple = tuple(int(item) for item in lags)
    if values.ndim != 2 or values.shape[0] != len(lag_tuple) or values.shape[1] < 32 or not np.isfinite(values).all():
        raise ValueError("delay observations must be finite [lags,samples]")
    whitened, model = pca_whiten(values, rank=len(lag_tuple), eigenvalue_floor_ratio=eigenvalue_floor_ratio)
    confirm_values = values if confirmation_observations is None else np.asarray(confirmation_observations, dtype=np.float64)
    if confirm_values.ndim != 2 or confirm_values.shape[0] != len(lag_tuple) or confirm_values.shape[1] < 32 or not np.isfinite(confirm_values).all():
        raise ValueError("confirmation observations must be finite [lags,samples]")
    confirm_whitened = model.whitening @ (confirm_values - model.mean[:, None])
    dep = dependence_function(objective_family, objective_parameter)
    dimension = len(lag_tuple); rotation = np.eye(dimension); current = whitened.copy()
    angles = np.deg2rad(np.arange(-45.0, 45.0 + angle_step_degrees / 2, angle_step_degrees))
    def total(array: np.ndarray) -> float:
        return float(sum(dep(array[left], array[right]) for left in range(dimension - 1) for right in range(left + 1, dimension)))
    history = [total(current)]; accepted = 0; converged = False
    for sweep in range(1, int(max_sweeps) + 1):
        start = history[-1]
        for left in range(dimension - 1):
            for right in range(left + 1, dimension):
                pair = current[[left, right]]; candidates = []
                for angle in angles:
                    c, s = np.cos(angle), np.sin(angle)
                    candidates.append(dep(c * pair[0] + s * pair[1], -s * pair[0] + c * pair[1]))
                best = int(np.argmin(candidates)); zero = int(np.argmin(np.abs(angles)))
                if candidates[zero] - candidates[best] <= improvement_tolerance:
                    continue
                jacobi = np.eye(dimension); c, s = np.cos(angles[best]), np.sin(angles[best])
                jacobi[left, left] = c; jacobi[left, right] = s; jacobi[right, left] = -s; jacobi[right, right] = c
                current = jacobi @ current; rotation = jacobi @ rotation; accepted += 1
        history.append(total(current))
        if start - history[-1] <= improvement_tolerance:
            converged = True; break
    confirmation_baseline = total(confirm_whitened)
    confirmation_objective = total(rotation @ confirm_whitened)
    effective = rotation @ model.whitening
    normalized = effective / np.maximum(np.linalg.norm(effective, axis=1, keepdims=True), 1e-12)
    common = np.ones(dimension, dtype=np.float64); common /= np.linalg.norm(common)
    derivative = np.zeros(dimension, dtype=np.float64); derivative[0], derivative[1] = 1.0, -1.0; derivative /= np.linalg.norm(derivative)
    persistence = int(np.argmax(np.abs(normalized @ common)))
    innovation_candidates = [index for index in range(dimension) if index != persistence]
    innovation = max(innovation_candidates, key=lambda index: abs(float(normalized[index] @ derivative)))
    signs = np.ones(dimension, dtype=np.int32)
    signs[persistence] = 1 if normalized[persistence] @ common >= 0 else -1
    signs[innovation] = 1 if normalized[innovation] @ derivative >= 0 else -1
    signed_rotation = np.diag(signs) @ rotation
    residual = tuple(index for index in range(dimension) if index not in (persistence, innovation))
    return TemporalMSICAFit(
        formulation="delay_embedding", objective_family=objective_family, objective_parameter=dict(objective_parameter),
        lags=lag_tuple, lag_weights=(), center=model.mean, whitening=model.whitening, rotation=signed_rotation,
        demixing=signed_rotation @ model.whitening, objective=float(confirmation_objective), baseline_objective=float(confirmation_baseline),
        persistence_index=persistence, innovation_index=innovation, residual_indices=residual, component_signs=tuple(int(item) for item in signs),
        converged=converged, diagnostics={"objective_history": history, "accepted_updates": accepted, "sweeps": sweep,
                                           "screen_objective": float(history[-1]), "screen_baseline_objective": float(history[0]),
                                           "confirmation_samples": int(confirm_values.shape[1]),
                                           "condition_number": model.condition_number, "explained_fraction": model.explained_fraction,
                                           "persistence_cosines": (normalized @ common).tolist(), "innovation_cosines": (normalized @ derivative).tolist()},
    )


def project_temporal_fit(values: Any, fit: TemporalMSICAFit, *, backend: str = "cpu") -> dict[str, Any]:
    xp = np
    if backend == "cuda":
        import cupy as cp
        xp = cp
    elif backend != "cpu":
        raise ValueError("backend must be cpu or cuda")
    source = xp.asarray(values, dtype=xp.float32)
    if fit.formulation == "delay_embedding":
        lags = fit.lags
        history = max(lags)
        length = int(source.shape[0]) - history
        stack = xp.stack([source[history - lag:history - lag + length] for lag in lags], axis=0)
    elif fit.formulation == "multilag_2d":
        # Training columns are [x(t-1), x(t)]; preserve that exact ordering.
        lags = (1, 0)
        history = 1
        length = int(source.shape[0]) - history
        stack = xp.stack((source[:-1], source[1:]), axis=0)
    else:
        raise ValueError(f"unknown formulation: {fit.formulation}")
    center = xp.asarray(fit.center, dtype=xp.float32).reshape((-1, 1, 1, 1))
    demixing = xp.asarray(fit.demixing, dtype=xp.float32)
    flat = (stack - center).reshape((len(lags), -1))
    outputs = (demixing @ flat).reshape((demixing.shape[0], length, source.shape[1], source.shape[2]))
    result = {"persistence": outputs[fit.persistence_index], "innovation": outputs[fit.innovation_index]}
    if fit.residual_indices:
        selected = outputs[xp.asarray(fit.residual_indices)]
        result["residual_group"] = xp.sqrt(xp.sum(xp.square(selected, dtype=xp.float32), axis=0))
    return result


def project_temporal_fit_chunked(
    values: Any,
    fit: TemporalMSICAFit,
    *,
    backend: str = "cpu",
    frame_chunk: int = 8,
    output_cpu: bool = True,
) -> dict[str, Any]:
    """Project a movie with bounded working memory on the selected backend."""
    if frame_chunk < 1:
        raise ValueError("frame_chunk must be positive")
    xp = np
    cp = None
    if backend == "cuda":
        import cupy as cp_module
        cp = cp_module
        xp = cp_module
    elif backend != "cpu":
        raise ValueError("backend must be cpu or cuda")
    source = xp.asarray(values, dtype=xp.float32)
    if fit.formulation == "delay_embedding":
        history = max(fit.lags)
        input_count = len(fit.lags)
    elif fit.formulation == "multilag_2d":
        history = 1
        input_count = 2
    else:
        raise ValueError(f"unknown formulation: {fit.formulation}")
    length = int(source.shape[0]) - history
    shape = (length, int(source.shape[1]), int(source.shape[2]))
    allocate = np.empty if output_cpu else xp.empty
    result = {
        "persistence": allocate(shape, dtype=np.float32),
        "innovation": allocate(shape, dtype=np.float32),
    }
    if fit.residual_indices:
        result["residual_group"] = allocate(shape, dtype=np.float32)
    center = xp.asarray(fit.center, dtype=xp.float32).reshape((-1, 1))
    demixing = xp.asarray(fit.demixing, dtype=xp.float32)
    for start in range(0, length, int(frame_chunk)):
        count = min(int(frame_chunk), length - start)
        absolute = history + start
        if fit.formulation == "delay_embedding":
            stack = xp.stack(
                [source[absolute - lag:absolute - lag + count] for lag in fit.lags],
                axis=0,
            )
        else:
            stack = xp.stack(
                (source[absolute - 1:absolute - 1 + count], source[absolute:absolute + count]),
                axis=0,
            )
        transformed = (demixing @ (stack.reshape((input_count, -1)) - center)).reshape(
            (demixing.shape[0], count, shape[1], shape[2])
        )
        chunks = {
            "persistence": transformed[fit.persistence_index],
            "innovation": transformed[fit.innovation_index],
        }
        if fit.residual_indices:
            selected = transformed[xp.asarray(fit.residual_indices)]
            chunks["residual_group"] = xp.sqrt(xp.sum(xp.square(selected), axis=0))
        for name, chunk in chunks.items():
            result[name][start:start + count] = cp.asnumpy(chunk) if output_cpu and cp is not None else chunk
        del stack, transformed
    return result
