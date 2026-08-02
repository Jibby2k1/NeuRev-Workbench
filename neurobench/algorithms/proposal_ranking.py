"""Candidate-level ranking primitives for sparse calcium-activity detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateTable:
    """One NMS-deduplicated candidate set and its feature matrix."""

    positions: np.ndarray
    features: np.ndarray
    source_count: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != 2
            or self.features.ndim != 2
            or len(self.positions) != len(self.features)
            or self.source_count.shape != (len(self.positions),)
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("candidate table arrays do not align")


def annular_kernel(radius_px: float, thickness_px: float) -> np.ndarray:
    """Return a normalized radial annulus matched filter."""
    radius = float(radius_px)
    thickness = float(thickness_px)
    if radius <= 0 or thickness <= 0:
        raise ValueError("radius and thickness must be positive")
    extent = max(2, int(np.ceil(radius + 3.0 * thickness)))
    yy, xx = np.mgrid[-extent : extent + 1, -extent : extent + 1]
    distance = np.sqrt(xx * xx + yy * yy)
    kernel = np.exp(-0.5 * ((distance - radius) / thickness) ** 2)
    kernel /= max(float(kernel.sum()), 1e-12)
    return kernel.astype(np.float32)


def cut_morphology_basis(
    score_map: np.ndarray,
    *,
    center_sigmas_px: Sequence[float],
    ring_specs: Sequence[Sequence[float]],
    crowd_sigma_px: float,
) -> dict[str, np.ndarray]:
    """Generate center, annular, and crowd responses from one pooled map."""
    from scipy.ndimage import convolve, gaussian_filter

    score = np.asarray(score_map, dtype=np.float32)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("score_map must be a finite YX array")
    positive = np.maximum(score, 0)
    result: dict[str, np.ndarray] = {}
    for sigma in center_sigmas_px:
        value = float(sigma)
        if value <= 0:
            raise ValueError("center sigmas must be positive")
        feature_id = f"cut_center_sigma{str(value).replace('.', 'p')}"
        result[feature_id] = gaussian_filter(
            positive, sigma=value, mode="reflect"
        ).astype(np.float32)
    for raw_radius, raw_thickness in ring_specs:
        radius = float(raw_radius)
        thickness = float(raw_thickness)
        feature_id = (
            f"cut_ring_r{str(radius).replace('.', 'p')}"
            f"_t{str(thickness).replace('.', 'p')}"
        )
        result[feature_id] = convolve(
            positive,
            annular_kernel(radius, thickness),
            mode="reflect",
        ).astype(np.float32)
    result["cut_crowd_context"] = gaussian_filter(
        positive, sigma=float(crowd_sigma_px), mode="reflect"
    ).astype(np.float32)
    return result


def robust_map_normalizer(
    quiet_maps: Sequence[np.ndarray],
    *,
    lower_percentile: float = 50.0,
    upper_percentile: float = 99.9,
    sample_stride: int = 4,
) -> tuple[float, float]:
    """Fit a scalar positive-evidence normalization from quiet maps only."""
    if not quiet_maps:
        raise ValueError("quiet maps are required")
    sampled = np.concatenate(
        [
            np.asarray(score, dtype=np.float32)[
                :: int(sample_stride), :: int(sample_stride)
            ].ravel()
            for score in quiet_maps
        ]
    )
    low, high = np.percentile(
        sampled, [float(lower_percentile), float(upper_percentile)]
    )
    return float(low), max(float(high - low), 1e-6)


def normalize_map(
    score_map: np.ndarray,
    normalizer: tuple[float, float],
    *,
    clip: float,
) -> np.ndarray:
    """Apply monotone soft compression after frozen quiet normalization.

    ``clip`` is a reference scale rather than a hard ceiling. A hard clip
    creates plateaus before NMS and can change peak locations.
    """
    low, scale = normalizer
    reference = float(clip)
    if reference <= 0:
        raise ValueError("clip reference must be positive")
    positive = np.maximum(
        (np.asarray(score_map, dtype=np.float32) - float(low)) / float(scale),
        0,
    )
    return (np.log1p(positive) / np.log1p(reference)).astype(np.float32)


def merge_peak_proposals(
    source_maps: Mapping[str, np.ndarray],
    normalizers: Mapping[str, tuple[float, float]],
    *,
    nms_distance_px: int,
    per_source_limit: int,
    dedupe_radius_px: float,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-source local maxima into a score-ranked spatial union."""
    from neurobench.metrics.sparse_detection import extract_local_maxima

    rows: list[tuple[float, int, int, str]] = []
    for source_id, score_map in source_maps.items():
        normalized = normalize_map(
            score_map, normalizers[source_id], clip=float(clip)
        )
        for score, x, y in extract_local_maxima(
            normalized,
            int(nms_distance_px),
            limit=int(per_source_limit),
        ):
            if float(score) <= 0:
                continue
            rows.append((float(score), int(x), int(y), source_id))
    rows.sort(key=lambda row: (row[0], row[3], -row[2], -row[1]), reverse=True)
    selected: list[tuple[int, int]] = []
    sources: list[set[str]] = []
    radius_sq = float(dedupe_radius_px) ** 2
    for _, x, y, source_id in rows:
        match = next(
            (
                index
                for index, (sx, sy) in enumerate(selected)
                if (x - sx) ** 2 + (y - sy) ** 2 <= radius_sq
            ),
            None,
        )
        if match is None:
            selected.append((x, y))
            sources.append({source_id})
        else:
            sources[match].add(source_id)
    return (
        np.asarray(selected, dtype=np.int32).reshape(-1, 2),
        np.asarray([len(value) for value in sources], dtype=np.int16),
    )


def sample_candidate_features(
    positions: np.ndarray,
    feature_maps: Mapping[str, np.ndarray],
    feature_ids: Sequence[str],
    normalizers: Mapping[str, tuple[float, float]],
    *,
    clip: float,
) -> np.ndarray:
    """Sample normalized feature values at XY candidate coordinates."""
    points = np.asarray(positions, dtype=np.int32)
    matrix = np.empty((len(points), len(feature_ids)), dtype=np.float32)
    for column, feature_id in enumerate(feature_ids):
        normalized = normalize_map(
            feature_maps[feature_id],
            normalizers[feature_id],
            clip=float(clip),
        )
        matrix[:, column] = normalized[points[:, 1], points[:, 0]]
    return matrix


def project_nonnegative_l1(
    weights: np.ndarray, maximum_total: float
) -> np.ndarray:
    """Project nonnegative weights onto an L1 ball."""
    result = np.maximum(np.asarray(weights, dtype=np.float64), 0)
    maximum = float(maximum_total)
    if maximum <= 0:
        raise ValueError("maximum_total must be positive")
    total = float(result.sum())
    if total > maximum:
        result *= maximum / total
    return result


def fit_bounded_pairwise_linear(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    carrier_column: int,
    auxiliary_columns: Sequence[int],
    auxiliary_directions: Sequence[float],
    learning_rate: float,
    epochs: int,
    l2: float,
    maximum_total: float,
) -> dict[str, Any]:
    """Fine-tune a nonnegative residual ranker from an exact carrier skip."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    columns = np.asarray(auxiliary_columns, dtype=np.int64)
    directions = np.asarray(auxiliary_directions, dtype=np.float64)
    if (
        pos.ndim != 2
        or neg.ndim != 2
        or pos.shape[1] != neg.shape[1]
        or not len(pos)
        or not len(neg)
        or len(columns) != len(directions)
    ):
        raise ValueError("invalid pairwise training arrays")
    count = max(len(pos), len(neg))
    positive_index = np.arange(count) % len(pos)
    negative_index = np.arange(count) % len(neg)
    base_delta = (
        pos[positive_index, int(carrier_column)]
        - neg[negative_index, int(carrier_column)]
    )
    feature_delta = (
        pos[positive_index][:, columns] - neg[negative_index][:, columns]
    ) * directions[None]
    weights = np.zeros(len(columns), dtype=np.float64)
    history: list[float] = []
    for _ in range(int(epochs)):
        margin = np.clip(base_delta + feature_delta @ weights, -40, 40)
        probability = 1.0 / (1.0 + np.exp(margin))
        gradient = np.mean(-feature_delta * probability[:, None], axis=0)
        gradient += 2.0 * float(l2) * weights
        weights = project_nonnegative_l1(
            weights - float(learning_rate) * gradient,
            float(maximum_total),
        )
        history.append(
            float(
                np.mean(np.logaddexp(0, -margin))
                + float(l2) * np.sum(weights**2)
            )
        )
    return {
        "kind": "bounded_linear",
        "carrier_column": int(carrier_column),
        "auxiliary_columns": columns.tolist(),
        "auxiliary_directions": directions.tolist(),
        "weights": weights.tolist(),
        "loss_initial": history[0],
        "loss_final": history[-1],
    }


def score_bounded_pairwise_linear(
    features: np.ndarray, model: Mapping[str, Any]
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    columns = np.asarray(model["auxiliary_columns"], dtype=np.int64)
    directions = np.asarray(model["auxiliary_directions"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    return (
        values[:, int(model["carrier_column"])]
        + (values[:, columns] * directions[None]) @ weights
    )


def _adam_step(
    value: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    step: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = 0.9 * first + 0.1 * gradient
    second = 0.999 * second + 0.001 * gradient * gradient
    corrected_first = first / (1.0 - 0.9**step)
    corrected_second = second / (1.0 - 0.999**step)
    value = value - float(learning_rate) * corrected_first / (
        np.sqrt(corrected_second) + 1e-8
    )
    return value, first, second


def fit_residual_mlp_ranker(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    carrier_column: int,
    input_columns: Sequence[int],
    hidden_units: int,
    maximum_residual: float,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    """Fit a tiny carrier-initialized residual MLP with bounded authority."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    columns = np.asarray(input_columns, dtype=np.int64)
    hidden = int(hidden_units)
    if (
        pos.ndim != 2
        or neg.ndim != 2
        or pos.shape[1] != neg.shape[1]
        or not len(pos)
        or not len(neg)
        or not len(columns)
        or hidden < 1
        or float(maximum_residual) <= 0
    ):
        raise ValueError("invalid residual MLP training contract")
    rng = np.random.default_rng(int(seed))
    weight_1 = rng.normal(0, 0.05, size=(len(columns), hidden))
    bias_1 = np.zeros(hidden, dtype=np.float64)
    weight_2 = np.zeros(hidden, dtype=np.float64)
    bias_2 = np.zeros(1, dtype=np.float64)
    parameters = [weight_1, bias_1, weight_2, bias_2]
    first = [np.zeros_like(value) for value in parameters]
    second = [np.zeros_like(value) for value in parameters]
    count = max(len(pos), len(neg))
    positive_index = np.arange(count) % len(pos)
    negative_index = np.arange(count) % len(neg)
    xp = pos[positive_index][:, columns]
    xn = neg[negative_index][:, columns]
    base_delta = (
        pos[positive_index, int(carrier_column)]
        - neg[negative_index, int(carrier_column)]
    )
    history: list[float] = []
    authority = float(maximum_residual)
    decay = float(weight_decay)
    for step in range(1, int(epochs) + 1):
        zp = xp @ weight_1 + bias_1[None]
        zn = xn @ weight_1 + bias_1[None]
        hp = np.maximum(zp, 0)
        hn = np.maximum(zn, 0)
        raw_p = hp @ weight_2 + bias_2[0]
        raw_n = hn @ weight_2 + bias_2[0]
        tanh_p = np.tanh(raw_p)
        tanh_n = np.tanh(raw_n)
        margin = np.clip(
            base_delta + authority * (tanh_p - tanh_n), -40, 40
        )
        dmargin = -1.0 / (1.0 + np.exp(margin)) / count
        draw_p = dmargin * authority * (1.0 - tanh_p * tanh_p)
        draw_n = -dmargin * authority * (1.0 - tanh_n * tanh_n)
        gradient_weight_2 = hp.T @ draw_p + hn.T @ draw_n
        gradient_bias_2 = np.asarray([draw_p.sum() + draw_n.sum()])
        dhp = draw_p[:, None] * weight_2[None]
        dhn = draw_n[:, None] * weight_2[None]
        dzp = dhp * (zp > 0)
        dzn = dhn * (zn > 0)
        gradient_weight_1 = xp.T @ dzp + xn.T @ dzn
        gradient_bias_1 = dzp.sum(axis=0) + dzn.sum(axis=0)
        gradients = [
            gradient_weight_1 + 2.0 * decay * weight_1,
            gradient_bias_1,
            gradient_weight_2 + 2.0 * decay * weight_2,
            gradient_bias_2,
        ]
        for index in range(len(parameters)):
            parameters[index], first[index], second[index] = _adam_step(
                parameters[index],
                gradients[index],
                first[index],
                second[index],
                step=step,
                learning_rate=float(learning_rate),
            )
        weight_1, bias_1, weight_2, bias_2 = parameters
        history.append(
            float(
                np.mean(np.logaddexp(0, -margin))
                + decay * (np.sum(weight_1**2) + np.sum(weight_2**2))
            )
        )
    return {
        "kind": "residual_mlp",
        "carrier_column": int(carrier_column),
        "input_columns": columns.tolist(),
        "hidden_units": hidden,
        "maximum_residual": authority,
        "weight_1": weight_1.tolist(),
        "bias_1": bias_1.tolist(),
        "weight_2": weight_2.tolist(),
        "bias_2": float(bias_2[0]),
        "loss_initial": history[0],
        "loss_final": history[-1],
        "seed": int(seed),
    }


def score_residual_mlp_ranker(
    features: np.ndarray, model: Mapping[str, Any]
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    columns = np.asarray(model["input_columns"], dtype=np.int64)
    weight_1 = np.asarray(model["weight_1"], dtype=np.float64)
    bias_1 = np.asarray(model["bias_1"], dtype=np.float64)
    weight_2 = np.asarray(model["weight_2"], dtype=np.float64)
    hidden = np.maximum(values[:, columns] @ weight_1 + bias_1[None], 0)
    residual = float(model["maximum_residual"]) * np.tanh(
        hidden @ weight_2 + float(model["bias_2"])
    )
    return values[:, int(model["carrier_column"])] + residual
