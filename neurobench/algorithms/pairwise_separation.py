"""Pure-array algorithms for bounded adjacent-frame source separation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class QuietDifferenceStats:
    center: np.ndarray
    scale: np.ndarray
    scale_floor: float


@dataclass(frozen=True)
class Whitening2D:
    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    identifiable: bool


@dataclass(frozen=True)
class SeparationFit:
    method_id: str
    demixing: np.ndarray
    mixing: np.ndarray | None
    objective: float | None
    converged: bool
    iterations: int
    activity_component: int | None
    activity_sign: int | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CSObjectiveResult:
    """Weighted Cauchy--Schwarz Parzen information-potential result."""

    objective: float
    v_joint: float
    v_marginal_1: float
    v_marginal_2: float
    v_product: float
    v_cross: float
    numerical_clamps: int
    clamped_terms: tuple[str, ...]
    sample_count: int
    positive_weight_count: int
    weight_sum: float
    block_rows: int


@dataclass(frozen=True)
class SharedBackgroundNMF:
    background: np.ndarray
    activity: np.ndarray
    converged: bool
    iterations: int
    objectives: tuple[float, ...]
    diagnostics: dict[str, Any]


def _finite_array(value: np.ndarray, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def fixed_difference(frames: np.ndarray, lag_frames: int = 1) -> np.ndarray:
    values = _finite_array(frames, "frames")
    if values.ndim < 2 or not 1 <= lag_frames < len(values):
        raise ValueError("lag_frames must be positive and shorter than frames")
    result = np.zeros_like(values, dtype=np.float64)
    result[lag_frames:] = values[lag_frames:] - values[:-lag_frames]
    return result.astype(np.float32)


def adaptive_difference(frames: np.ndarray, alpha: float, lag_frames: int = 1) -> np.ndarray:
    values = _finite_array(frames, "frames")
    if not np.isfinite(alpha) or alpha <= 0 or not 1 <= lag_frames < len(values):
        raise ValueError("alpha must be positive and lag_frames valid")
    result = np.zeros_like(values, dtype=np.float64)
    result[lag_frames:] = values[lag_frames:] - alpha * values[:-lag_frames]
    return result.astype(np.float32)


def quiet_difference_stats(
    differences: np.ndarray, *, floor_percentile: float = 10.0
) -> QuietDifferenceStats:
    values = _finite_array(differences, "differences")
    if values.ndim < 2 or not 0 <= floor_percentile <= 100:
        raise ValueError("differences must be frame-first and percentile in [0,100]")
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    positive = mad[mad > 0]
    floor = float(np.percentile(positive, floor_percentile)) if positive.size else 1.0
    floor = max(floor, np.finfo(np.float32).eps)
    return QuietDifferenceStats(
        center=center.astype(np.float32),
        scale=np.maximum(mad, floor).astype(np.float32),
        scale_floor=floor,
    )


def standardized_positive_mask(
    differences: np.ndarray,
    stats: QuietDifferenceStats,
    threshold: float,
    *,
    undefined_leading_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_array(differences, "differences")
    if values.shape[1:] != stats.center.shape or threshold <= 0:
        raise ValueError("stats shape must match and threshold must be positive")
    z = ((values - stats.center) / stats.scale).astype(np.float32)
    mask = (z >= threshold).astype(np.uint8)
    if undefined_leading_frames:
        z[:undefined_leading_frames] = 0
        mask[:undefined_leading_frames] = 0
    return z, mask


def estimate_quiet_gain(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    alpha_min: float = 0.8,
    alpha_max: float = 1.2,
    trim_fraction: float = 0.1,
    refinement_iterations: int = 3,
) -> tuple[float, dict[str, Any]]:
    x0 = _finite_array(previous, "previous")
    x1 = _finite_array(current, "current")
    if x0.shape != x1.shape or x0.ndim < 2:
        raise ValueError("previous/current must be aligned pair-first arrays")
    if not 0 <= trim_fraction < 0.5 or not alpha_min < alpha_max or refinement_iterations < 0:
        raise ValueError("invalid gain-estimation bounds")
    slopes, rejected = [], 0
    for a, b in zip(x0, x1):
        u, v = a.ravel(), b.ravel()
        valid = np.isfinite(u) & np.isfinite(v) & (np.abs(u) > np.finfo(float).eps)
        u, v = u[valid], v[valid]
        denominator = float(u @ u)
        if len(u) < 2 or denominator <= np.finfo(float).eps:
            rejected += 1
            continue
        alpha = float(np.clip((u @ v) / denominator, alpha_min, alpha_max))
        for _ in range(refinement_iterations):
            residual = np.abs(v - alpha * u)
            keep_count = max(2, int(np.floor(len(u) * (1 - trim_fraction))))
            keep = np.argpartition(residual, keep_count - 1)[:keep_count]
            denom = float(u[keep] @ u[keep])
            if denom <= np.finfo(float).eps:
                break
            alpha = float(np.clip((u[keep] @ v[keep]) / denom, alpha_min, alpha_max))
        slopes.append(alpha)
    if not slopes:
        raise ValueError("No non-degenerate quiet pairs for gain estimation")
    alpha = float(np.median(slopes))
    return alpha, {
        "pair_slopes": [float(x) for x in slopes],
        "accepted_pairs": len(slopes),
        "rejected_pairs": rejected,
        "alpha_median": alpha,
        "alpha_min_observed": float(min(slopes)),
        "alpha_max_observed": float(max(slopes)),
    }


def center_and_whiten_2d(samples: np.ndarray, *, eigenvalue_floor_ratio: float = 1e-6) -> tuple[np.ndarray, Whitening2D]:
    x = _finite_array(samples, "samples", 2)
    if x.shape[0] != 2 or x.shape[1] < 3 or not 0 < eigenvalue_floor_ratio < 1:
        raise ValueError("samples must have shape [2,N>=3]")
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    covariance = centered @ centered.T / x.shape[1]
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    largest = max(float(eigenvalues[0]), np.finfo(float).eps)
    raw_condition = largest / max(float(eigenvalues[-1]), np.finfo(float).eps)
    floor = largest * eigenvalue_floor_ratio
    floored = np.maximum(eigenvalues, floor)
    whitening = np.diag(1 / np.sqrt(floored)) @ vectors.T
    dewhitening = vectors @ np.diag(np.sqrt(floored))
    identifiable = bool(eigenvalues[-1] > floor and raw_condition <= 1e8)
    whitened = whitening @ centered
    return whitened.astype(np.float64), Whitening2D(
        mean=mean[:, 0], covariance=covariance, whitening=whitening,
        dewhitening=dewhitening, eigenvalues=eigenvalues,
        condition_number=float(raw_condition), identifiable=identifiable,
    )


def _rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], dtype=np.float64)


def _symmetric_decorrelate(matrix: np.ndarray) -> np.ndarray:
    gram = matrix @ matrix.T
    values, vectors = np.linalg.eigh(gram)
    return (vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-12))) @ vectors.T) @ matrix


def fit_infomax_tanh_ica(
    whitened: np.ndarray,
    *,
    initial_angles_degrees: Sequence[float] = (0, 15, 30, 45, 60, 75),
    max_iterations: int = 500,
    learning_rate: float = 0.01,
    tolerance: float = 1e-7,
) -> SeparationFit:
    z = _finite_array(whitened, "whitened", 2)
    if z.shape[0] != 2 or max_iterations < 1 or learning_rate <= 0 or tolerance <= 0:
        raise ValueError("invalid InfoMax inputs")
    restarts = []
    for initial in initial_angles_degrees:
        w = _rotation(float(initial))
        previous_objective = None
        converged = False
        for iteration in range(1, max_iterations + 1):
            y = w @ z
            phi = np.tanh(y)
            update = (np.eye(2) - (phi @ y.T) / z.shape[1]) @ w
            next_w = _symmetric_decorrelate(w + learning_rate * update)
            next_y = next_w @ z
            objective = float(np.mean(np.log(np.cosh(np.clip(next_y, -20, 20))), axis=1).sum())
            matrix_delta = float(np.linalg.norm(np.abs(next_w @ w.T) - np.eye(2)))
            objective_delta = float("inf") if previous_objective is None else abs(objective - previous_objective)
            w = next_w
            if matrix_delta <= tolerance and objective_delta <= tolerance:
                converged = True
                break
            previous_objective = objective
        restarts.append({"initial_angle_degrees": float(initial), "demixing": w,
                         "objective": objective, "iterations": iteration, "converged": converged})
    eligible = [row for row in restarts if row["converged"]] or restarts
    best = max(eligible, key=lambda row: row["objective"])
    return SeparationFit(
        method_id="infomax_tanh_ica", demixing=best["demixing"], mixing=best["demixing"].T,
        objective=float(best["objective"]), converged=bool(best["converged"]),
        iterations=int(best["iterations"]), activity_component=None, activity_sign=None,
        diagnostics={"restarts": [{k: v for k, v in row.items() if k != "demixing"} for row in restarts],
                     "selected_initial_angle_degrees": best["initial_angle_degrees"]},
    )


def cs_parzen_objective(
    y: np.ndarray,
    bandwidth: float,
    *,
    weights: np.ndarray | None = None,
    block_rows: int = 256,
    accumulator_dtype: np.dtype = np.float64,
    kernel_dtype: np.dtype = np.float64,
) -> CSObjectiveResult:
    """Evaluate the exact weighted, blockwise two-output CS-Parzen objective.

    y follows the public [N, 2] convention. Weighted centering and scaling
    preserve the legacy marginal standardization and make zero-weight exclusion
    and integer-repetition semantics exact. Kernel blocks are the only
    quadratic-sized temporaries.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not values.size:
        raise ValueError("y must be a non-empty array with shape [N,2]")
    if not np.isfinite(values).all():
        raise ValueError("y must be finite")
    if not np.isfinite(bandwidth) or bandwidth <= 0 or block_rows < 1:
        raise ValueError("bandwidth and block_rows must be positive")
    accumulator = np.dtype(accumulator_dtype)
    kernel = np.dtype(kernel_dtype)
    if accumulator.kind != "f" or kernel.kind != "f":
        raise ValueError("accumulator_dtype and kernel_dtype must be floating point")
    if weights is None:
        sample_weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != (values.shape[0],):
            raise ValueError("weights must have shape [N]")
        if not np.isfinite(sample_weights).all():
            raise ValueError("weights must be finite")
        if np.any(sample_weights < 0):
            raise ValueError("weights must be nonnegative")
    weight_sum = float(sample_weights.sum(dtype=accumulator))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have a positive finite sum")

    normalized = sample_weights / weight_sum
    mean = normalized @ values
    centered = values - mean
    variance = normalized @ (centered * centered)
    standardized = centered / np.sqrt(np.maximum(variance, 1e-24))
    u, v = standardized.T
    joint_num = accumulator.type(0)
    marginal_1_num = accumulator.type(0)
    marginal_2_num = accumulator.type(0)
    cross_num = accumulator.type(0)
    for start in range(0, len(values), block_rows):
        stop = min(len(values), start + block_rows)
        block_weights = sample_weights[start:stop]
        kernel_1 = np.exp(
            -0.5 * ((u[start:stop, None] - u[None, :]) / bandwidth) ** 2
        ).astype(kernel, copy=False)
        kernel_2 = np.exp(
            -0.5 * ((v[start:stop, None] - v[None, :]) / bandwidth) ** 2
        ).astype(kernel, copy=False)
        kernel_1_weights = kernel_1 @ sample_weights
        kernel_2_weights = kernel_2 @ sample_weights
        joint_num += np.dot(block_weights, (kernel_1 * kernel_2) @ sample_weights)
        marginal_1_num += np.dot(block_weights, kernel_1_weights)
        marginal_2_num += np.dot(block_weights, kernel_2_weights)
        cross_num += np.dot(block_weights, kernel_1_weights * kernel_2_weights)

    squared_sum = weight_sum * weight_sum
    v_joint = float(joint_num / squared_sum)
    v_marginal_1 = float(marginal_1_num / squared_sum)
    v_marginal_2 = float(marginal_2_num / squared_sum)
    v_product = v_marginal_1 * v_marginal_2
    v_cross = float(cross_num / (squared_sum * weight_sum))
    terms = {"v_joint": v_joint, "v_product": v_product, "v_cross": v_cross}
    epsilon = np.finfo(accumulator).tiny
    clamped = tuple(name for name, value in terms.items() if value <= epsilon)
    safe = {name: max(value, epsilon) for name, value in terms.items()}
    objective = -np.log(
        safe["v_cross"] / np.sqrt(safe["v_joint"] * safe["v_product"])
    )
    return CSObjectiveResult(
        objective=float(objective),
        v_joint=v_joint,
        v_marginal_1=v_marginal_1,
        v_marginal_2=v_marginal_2,
        v_product=v_product,
        v_cross=v_cross,
        numerical_clamps=len(clamped),
        clamped_terms=clamped,
        sample_count=len(values),
        positive_weight_count=int(np.count_nonzero(sample_weights)),
        weight_sum=weight_sum,
        block_rows=block_rows,
    )


def cs_parzen_independence(
    outputs: np.ndarray,
    bandwidth: float,
    *,
    weights: np.ndarray | None = None,
    block_rows: int = 256,
    kernel_dtype: np.dtype = np.float64,
) -> tuple[float, dict[str, Any]]:
    """Compatibility wrapper retaining the legacy [2,N] API."""
    y = _finite_array(outputs, "outputs", 2)
    if y.shape[0] != 2:
        raise ValueError("outputs must have shape [2,N]")
    result = cs_parzen_objective(
        y.T, bandwidth, weights=weights, block_rows=block_rows,
        kernel_dtype=kernel_dtype,
    )
    diagnostics = {
        "v_joint": result.v_joint,
        "v_marginal_1": result.v_marginal_1,
        "v_marginal_2": result.v_marginal_2,
        "v_product": result.v_product,
        "v_cross": result.v_cross,
        "numerical_clamps": result.numerical_clamps,
        "clamped_terms": list(result.clamped_terms),
        "sample_count": result.sample_count,
        "positive_weight_count": result.positive_weight_count,
        "weight_sum": result.weight_sum,
        "block_rows": result.block_rows,
    }
    return result.objective, diagnostics


def fit_cs_parzen_ica(
    screen_whitened: np.ndarray,
    confirm_whitened: np.ndarray | None = None,
    *,
    bandwidth: float = 0.35,
    block_rows: int = 256,
    screen_step_degrees: float = 3.0,
    refine_half_width_degrees: float = 3.0,
    refine_step_degrees: float = 0.25,
    screen_weights: np.ndarray | None = None,
    confirm_weights: np.ndarray | None = None,
    kernel_dtype: np.dtype = np.float64,
) -> SeparationFit:
    screen = _finite_array(screen_whitened, "screen_whitened", 2)
    confirm = screen if confirm_whitened is None else _finite_array(confirm_whitened, "confirm_whitened", 2)
    if screen.shape[0] != 2 or confirm.shape[0] != 2:
        raise ValueError("whitened samples must be [2,N]")
    coarse = np.arange(0, 90, screen_step_degrees)
    rows = []
    for angle in coarse:
        objective, diagnostics = cs_parzen_independence(
            _rotation(angle) @ screen, bandwidth, weights=screen_weights,
            block_rows=block_rows, kernel_dtype=kernel_dtype,
        )
        rows.append({"scope": "screen", "angle_degrees": float(angle), "objective": objective, **diagnostics})
    best_coarse = min(rows, key=lambda row: row["objective"])["angle_degrees"]
    refine = np.arange(best_coarse - refine_half_width_degrees, best_coarse + refine_half_width_degrees + refine_step_degrees / 2, refine_step_degrees) % 90
    for angle in sorted(set(float(x) for x in refine)):
        objective, diagnostics = cs_parzen_independence(
            _rotation(angle) @ screen, bandwidth, weights=screen_weights,
            block_rows=block_rows, kernel_dtype=kernel_dtype,
        )
        rows.append({"scope": "refine", "angle_degrees": angle, "objective": objective, **diagnostics})
    best = min((row for row in rows if row["scope"] == "refine"), key=lambda row: row["objective"])
    confirm_angles = sorted({(best["angle_degrees"] + delta) % 90 for delta in (-refine_step_degrees, 0, refine_step_degrees)})
    confirmed = []
    for angle in confirm_angles:
        objective, diagnostics = cs_parzen_independence(
            _rotation(angle) @ confirm, bandwidth, weights=confirm_weights,
            block_rows=block_rows, kernel_dtype=kernel_dtype,
        )
        row = {"scope": "confirm", "angle_degrees": float(angle), "objective": objective, **diagnostics}
        rows.append(row); confirmed.append(row)
    winner = min(confirmed, key=lambda row: row["objective"])
    w = _rotation(winner["angle_degrees"])
    return SeparationFit(
        method_id="cs_parzen_ica", demixing=w, mixing=w.T, objective=float(winner["objective"]),
        converged=True, iterations=len(rows), activity_component=None, activity_sign=None,
        diagnostics={"selected_angle_degrees": winner["angle_degrees"], "objective_by_angle": rows,
                     "bandwidth": bandwidth, "kernel_block_rows": block_rows},
    )


def orient_and_select_activity_component(
    components: np.ndarray,
    signed_difference: np.ndarray,
    common_signal: np.ndarray,
) -> tuple[np.ndarray, int | None, tuple[int, int], dict[str, Any]]:
    y = _finite_array(components, "components", 2)
    derivative = _finite_array(signed_difference, "signed_difference").ravel()
    common = _finite_array(common_signal, "common_signal").ravel()
    if y.shape[0] != 2 or y.shape[1] != len(derivative) or len(common) != len(derivative):
        raise ValueError("components must be [2,N] and references length N")
    oriented = y.copy(); stats = []
    for index in range(2):
        corr = float(np.corrcoef(oriented[index], derivative)[0, 1]) if np.std(oriented[index]) and np.std(derivative) else 0.0
        sign = 1 if corr >= 0 else -1
        oriented[index] *= sign; corr = abs(corr)
        centered = oriented[index] - oriented[index].mean()
        scale = max(float(oriented[index].std()), 1e-12)
        skew = float(np.mean((centered / scale) ** 3))
        tail = float(np.mean(oriented[index] >= np.percentile(oriented[index], 95)))
        common_corr = abs(float(np.corrcoef(oriented[index], common)[0, 1])) if np.std(common) else 0.0
        stats.append({"component": index, "sign": sign, "derivative_correlation": corr,
                      "positive_skewness": skew, "upper_tail_fraction": tail,
                      "absolute_common_correlation": common_corr})
    qualified = [row for row in stats if row["derivative_correlation"] >= 0.1 and row["positive_skewness"] > 0]
    if not qualified:
        return oriented, None, (stats[0]["sign"], stats[1]["sign"]), {"status": "unresolved", "components": stats}
    best_corr = max(row["derivative_correlation"] for row in qualified)
    near = [row for row in qualified if best_corr - row["derivative_correlation"] <= 0.05]
    chosen = sorted(near, key=lambda row: (-row["positive_skewness"], row["absolute_common_correlation"], row["component"]))[0]
    return oriented, int(chosen["component"]), (stats[0]["sign"], stats[1]["sign"]), {"status": "resolved", "components": stats}


def apply_linear_separation(samples: np.ndarray, mean: np.ndarray, whitening: np.ndarray, demixing: np.ndarray) -> np.ndarray:
    x = _finite_array(samples, "samples", 2)
    center = _finite_array(mean, "mean").reshape(2, 1)
    w = _finite_array(whitening, "whitening", 2)
    d = _finite_array(demixing, "demixing", 2)
    if x.shape[0] != 2 or w.shape != (2, 2) or d.shape != (2, 2):
        raise ValueError("linear separation requires two-channel inputs and 2x2 matrices")
    return (d @ w @ (x - center)).astype(np.float32)


def fit_shared_background_nmf(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
    *,
    activity_l1: float = 0.05,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> SharedBackgroundNMF:
    i0 = _finite_array(previous, "previous")
    i1 = _finite_array(current, "current")
    if i0.shape != i1.shape or np.min(i0) < 0 or np.min(i1) < 0 or alpha <= 0 or activity_l1 < 0:
        raise ValueError("NMF inputs must be aligned, nonnegative, with valid alpha/penalty")
    background = np.maximum(i0, 0).copy()
    activity = np.maximum(i1 - alpha * background - activity_l1, 0)
    objectives = []
    converged = False
    violations = 0
    for iteration in range(1, max_iterations + 1):
        background = np.maximum((i0 + alpha * (i1 - activity)) / (1 + alpha**2), 0)
        activity = np.maximum(i1 - alpha * background - activity_l1, 0)
        objective = float(0.5 * np.sum((i0 - background) ** 2) +
                          0.5 * np.sum((i1 - alpha * background - activity) ** 2) +
                          activity_l1 * np.sum(activity))
        if objectives and objective > objectives[-1] + max(1e-10, abs(objectives[-1]) * 1e-10):
            violations += 1
        objectives.append(objective)
        if len(objectives) > 1 and abs(objectives[-2] - objective) <= tolerance * max(1.0, abs(objectives[-2])):
            converged = True
            break
    positive_residual = np.maximum(i1 - alpha * i0, 0)
    corr = float(np.corrcoef(activity.ravel(), positive_residual.ravel())[0, 1]) if np.std(activity) and np.std(positive_residual) else 1.0
    nmad = float(np.mean(np.abs(activity - positive_residual)) / max(np.mean(np.abs(positive_residual)), 1e-12))
    return SharedBackgroundNMF(
        background=background.astype(np.float32), activity=activity.astype(np.float32),
        converged=converged and violations == 0, iterations=iteration, objectives=tuple(objectives),
        diagnostics={"monotonicity_violations": violations, "activity_nonzero_fraction": float(np.mean(activity > 0)),
                     "positive_residual_correlation": corr, "normalized_mean_absolute_difference": nmad,
                     "equivalent_to_positive_adaptive_residual": bool(corr > 0.995 and nmad < 0.05)},
    )
