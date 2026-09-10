"""Leakage-resistant positive-unlabeled feature-fusion models.

This module contains no repository or artifact I/O.  It operates on a frozen
candidate census supplied by a caller and returns serialisable evaluation
records.  Scores are *ranking scores*: sparse positive-reference labels do not
give them a calibrated binary interpretation, and unlabeled candidates remain
unknown.

The public evaluator deliberately fails closed when the spatial groups cannot
support every requested outer, inner, and MLP early-stopping split.  Robust
imputation/scaling and the positive-reference distance are always learned from
the relevant training partition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1


def _normalise_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "positive" if bool(value) else "unlabeled"
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return "positive"
        if int(value) == 0:
            return "unlabeled"
        if int(value) == -1:
            return "excluded"
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return _normalise_label(int(value))
    text = str(value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


@dataclass(frozen=True)
class LabelPolicy:
    """Map review states into positive, unlabeled, or excluded roles."""

    name: str
    positive_values: tuple[str, ...]
    unlabeled_values: tuple[str, ...]
    excluded_values: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = [
            {_normalise_label(value) for value in self.positive_values},
            {_normalise_label(value) for value in self.unlabeled_values},
            {_normalise_label(value) for value in self.excluded_values},
        ]
        if not self.name.strip():
            raise ValueError("label policy name must be non-empty")
        if not groups[0] or not groups[1]:
            raise ValueError("label policy requires positive and unlabeled values")
        if (groups[0] & groups[1]) or (groups[0] & groups[2]) or (groups[1] & groups[2]):
            raise ValueError("label policy roles must be disjoint")


STRICT_LABEL_POLICY = LabelPolicy(
    name="strict_definite_positive",
    positive_values=("positive", "definite", "definite_neuron", "neuron", "yes"),
    unlabeled_values=(
        "unlabeled",
        "unknown",
        "uncertain",
        "possible",
        "probable",
        "might_be_neuron",
        "maybe_neuron",
    ),
    excluded_values=("exclude", "excluded", "artifact", "noise", "not_neuron", "negative", "no"),
)

INCLUSIVE_LABEL_POLICY = LabelPolicy(
    name="inclusive_definite_or_probable_positive",
    positive_values=(
        "positive",
        "definite",
        "definite_neuron",
        "neuron",
        "yes",
        "possible",
        "probable",
        "might_be_neuron",
        "maybe_neuron",
    ),
    unlabeled_values=("unlabeled", "unknown", "uncertain"),
    excluded_values=("exclude", "excluded", "artifact", "noise", "not_neuron", "negative", "no"),
)


@dataclass(frozen=True)
class ResolvedLabels:
    positive: np.ndarray
    unlabeled: np.ndarray
    excluded: np.ndarray
    canonical_values: tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.positive), len(self.unlabeled), len(self.excluded), len(self.canonical_values)}
        if len(lengths) != 1:
            raise ValueError("resolved label arrays must have equal length")
        if np.any(self.positive & self.unlabeled) or np.any(self.positive & self.excluded) or np.any(self.unlabeled & self.excluded):
            raise ValueError("resolved label roles must be disjoint")
        if not np.all(self.positive | self.unlabeled | self.excluded):
            raise ValueError("every label must resolve to exactly one role")


@dataclass(frozen=True)
class EvaluationConfig:
    """Predeclared controls for nested spatial positive-unlabeled evaluation."""

    outer_splits: int = 5
    inner_splits: int = 3
    candidate_budget: int = 58
    seed: int = 20260830
    hyperparameter_selection: str = "fixed"
    logistic_penalties: tuple[float, ...] = (0.01, 0.1, 1.0)
    elastic_penalties: tuple[float, ...] = (0.01, 0.1, 1.0)
    pu_penalties: tuple[float, ...] = (0.01, 0.1, 1.0)
    mlp_l2_values: tuple[float, ...] = (0.001, 0.01, 0.1)
    mlp_seeds: tuple[int, ...] = tuple(range(10))
    pu_bags: int = 32
    pu_unlabeled_ratio: float = 2.0
    max_logistic_iterations: int = 5000
    logistic_tolerance: float = 1e-7
    mlp_max_epochs: int = 400
    mlp_patience: int = 35
    mlp_learning_rate: float = 0.02
    permutation_draws: int = 1000
    preprocess_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("outer_splits and inner_splits must each be at least two")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if self.hyperparameter_selection not in {"fixed", "nested"}:
            raise ValueError("hyperparameter_selection must be 'fixed' or 'nested'")
        for name in ("logistic_penalties", "elastic_penalties", "pu_penalties", "mlp_l2_values"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or any((not np.isfinite(value)) or value < 0 for value in values):
                raise ValueError(f"{name} must contain finite non-negative values")
        if not self.mlp_seeds or len(set(self.mlp_seeds)) != len(self.mlp_seeds):
            raise ValueError("mlp_seeds must be a non-empty unique sequence")
        if self.pu_bags < 2 or self.pu_unlabeled_ratio <= 0:
            raise ValueError("PU configuration requires at least two bags and a positive sampling ratio")
        if self.max_logistic_iterations < 10 or self.mlp_max_epochs < 10 or self.mlp_patience < 1:
            raise ValueError("iteration and early-stopping limits are too small")
        try:
            logistic_tolerance = float(self.logistic_tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("logistic_tolerance must be finite and positive") from exc
        if not np.isfinite(logistic_tolerance) or logistic_tolerance <= 0:
            raise ValueError("logistic_tolerance must be finite and positive")
        object.__setattr__(self, "logistic_tolerance", logistic_tolerance)
        if self.mlp_learning_rate <= 0 or self.permutation_draws < 0 or self.preprocess_clip <= 0:
            raise ValueError(
                "learning rate and preprocessing clip must be positive; null draws must be non-negative"
            )


_FORBIDDEN_EXACT_FEATURES = {
    "x",
    "y",
    "x_int",
    "y_int",
    "row",
    "column",
    "coordinates",
    "coordinate",
    "burst",
    "burst_id",
    "event_id",
    "candidate_rank",
    "current_rank",
    "priority",
    "priority_score",
    "review_label",
    "expert_label",
    "human_label",
    "canonical_roi_id",
    "original_roi_id",
    "observation_site_id",
    "geometry_hash",
}
_FORBIDDEN_FEATURE_COMPONENT = re.compile(r"(?:^|_)(?:coord|coordinate|row|column|candidate_rank|current_rank|priority|review_label|expert_label|human_label)(?:_|$)")


def validate_raw_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Reject identity/leakage fields and globally pre-standardised columns."""

    names = tuple(str(name).strip() for name in feature_names)
    if not names or any(not name for name in names):
        raise ValueError("feature_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    violations: list[str] = []
    for original in names:
        name = original.casefold().replace("-", "_").replace(" ", "_")
        if (
            name in _FORBIDDEN_EXACT_FEATURES
            or _FORBIDDEN_FEATURE_COMPONENT.search(name)
            or name.startswith("z_")
            or name.endswith("_z")
            or "cohort_z" in name
            or "global_z" in name
            or "dataset_z" in name
            or "precomputed_z" in name
        ):
            violations.append(original)
    if violations:
        raise ValueError(
            "feature columns must contain raw fold-local inputs only; forbidden/leaky columns: "
            + ", ".join(sorted(violations))
        )
    return names


def resolve_labels(labels: Sequence[Any], policy: LabelPolicy = STRICT_LABEL_POLICY) -> ResolvedLabels:
    """Resolve review labels without converting unknown candidates to negatives."""

    positive_values = {_normalise_label(value) for value in policy.positive_values}
    unlabeled_values = {_normalise_label(value) for value in policy.unlabeled_values}
    excluded_values = {_normalise_label(value) for value in policy.excluded_values}
    canonical: list[str] = []
    roles: list[str] = []
    unknown: list[str] = []
    for value in labels:
        label = _normalise_label(value)
        canonical.append(label)
        if label in positive_values:
            roles.append("positive")
        elif label in unlabeled_values:
            roles.append("unlabeled")
        elif label in excluded_values:
            roles.append("excluded")
        else:
            roles.append("unknown")
            unknown.append(repr(value))
    if unknown:
        raise ValueError("unrecognised labels under policy " + policy.name + ": " + ", ".join(sorted(set(unknown))))
    array = np.asarray(roles, dtype=object)
    return ResolvedLabels(
        positive=array == "positive",
        unlabeled=array == "unlabeled",
        excluded=array == "excluded",
        canonical_values=tuple(canonical),
    )


def _as_float_matrix(values: Any, *, name: str = "X") -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    return array


def _balanced_reference_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    positives = int(np.sum(y == 1))
    unlabeled = int(np.sum(y == 0))
    if positives < 1 or unlabeled < 1:
        raise ValueError("training partition requires positive and unlabeled samples")
    weights = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * unlabeled))
    return weights / float(np.mean(weights))


class RobustFoldPreprocessor:
    """Median/MAD preprocessing with a separately scored positive reference."""

    def __init__(
        self,
        *,
        clip: float = 10.0,
        add_missing_indicators: bool = True,
        append_positive_reference_distance: bool = False,
    ) -> None:
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.clip = float(clip)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.append_positive_reference_distance = bool(append_positive_reference_distance)
        self.fitted_ = False

    def fit(
        self,
        X: Any,
        positive_mask: Sequence[bool],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "RobustFoldPreprocessor":
        matrix = _as_float_matrix(X)
        positive = np.asarray(positive_mask, dtype=bool)
        if positive.shape != (len(matrix),):
            raise ValueError("positive_mask length must match X")
        if int(np.sum(positive)) < 2:
            raise ValueError("positive-reference distance requires at least two training positives")
        if feature_names is None:
            names = tuple(f"feature_{index}" for index in range(matrix.shape[1]))
        else:
            names = validate_raw_feature_names(feature_names)
            if len(names) != matrix.shape[1]:
                raise ValueError("feature_names length must match X columns")
        finite = np.isfinite(matrix)
        if np.any(np.sum(finite, axis=0) == 0):
            bad = [names[index] for index in np.flatnonzero(np.sum(finite, axis=0) == 0)]
            raise ValueError("training partition has all-missing features: " + ", ".join(bad))
        median = np.nanmedian(np.where(finite, matrix, np.nan), axis=0)
        imputed = np.where(finite, matrix, median)
        mad = 1.4826 * np.median(np.abs(imputed - median), axis=0)
        q25, q75 = np.quantile(imputed, (0.25, 0.75), axis=0)
        iqr_scale = (q75 - q25) / 1.349
        floor = np.maximum(np.abs(median) * 1e-12, 1e-12)
        scale = np.where(mad > floor, mad, np.where(iqr_scale > floor, iqr_scale, 1.0))
        base = np.clip((imputed - median) / scale, -self.clip, self.clip)
        positive_base = base[positive]
        positive_center = np.median(positive_base, axis=0)
        positive_mad = 1.4826 * np.median(np.abs(positive_base - positive_center), axis=0)
        pq25, pq75 = np.quantile(positive_base, (0.25, 0.75), axis=0)
        positive_iqr = (pq75 - pq25) / 1.349
        positive_scale = np.where(positive_mad > 1e-6, positive_mad, np.where(positive_iqr > 1e-6, positive_iqr, 1.0))

        self.feature_names_in_ = names
        self.location_ = median
        self.scale_ = scale
        self.positive_center_ = positive_center
        self.positive_scale_ = positive_scale
        self.missing_indicator_mask_ = np.any(~finite, axis=0) if self.add_missing_indicators else np.zeros(matrix.shape[1], dtype=bool)
        indicator_names = tuple(f"missing__{names[index]}" for index in np.flatnonzero(self.missing_indicator_mask_))
        distance_names = ("positive_reference_distance",) if self.append_positive_reference_distance else ()
        self.feature_names_out_ = names + indicator_names + distance_names
        self.fitted_ = True
        transformed = self.transform(matrix)
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("fold-local preprocessing produced non-finite training values")
        if np.all(np.std(transformed[:, : matrix.shape[1]], axis=0) <= 1e-12):
            raise ValueError("training partition has no varying raw feature")
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("preprocessor must be fitted before transform")
        matrix = _as_float_matrix(X)
        if matrix.shape[1] != len(self.feature_names_in_):
            raise ValueError("X column count differs from fitted preprocessor")
        finite = np.isfinite(matrix)
        imputed = np.where(finite, matrix, self.location_)
        base = np.clip((imputed - self.location_) / self.scale_, -self.clip, self.clip)
        pieces = [base]
        if np.any(self.missing_indicator_mask_):
            pieces.append((~finite[:, self.missing_indicator_mask_]).astype(np.float64))
        if self.append_positive_reference_distance:
            pieces.append(self.positive_reference_distance(matrix)[:, None])
        transformed = np.concatenate(pieces, axis=1)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("transform produced non-finite values")
        return transformed

    def positive_reference_distance(self, X: Any) -> np.ndarray:
        """Score distance using only the fitted training-positive reference."""

        if not self.fitted_:
            raise RuntimeError("preprocessor must be fitted before distance scoring")
        matrix = _as_float_matrix(X)
        if matrix.shape[1] != len(self.feature_names_in_):
            raise ValueError("X column count differs from fitted preprocessor")
        finite = np.isfinite(matrix)
        imputed = np.where(finite, matrix, self.location_)
        base = np.clip((imputed - self.location_) / self.scale_, -self.clip, self.clip)
        positive_z = np.clip((base - self.positive_center_) / self.positive_scale_, -self.clip, self.clip)
        distance = np.sqrt(np.mean(positive_z * positive_z, axis=1, dtype=np.float64))
        if not np.all(np.isfinite(distance)):
            raise ValueError("positive-reference distance produced non-finite values")
        return distance

    def fit_transform(
        self,
        X: Any,
        positive_mask: Sequence[bool],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        return self.fit(X, positive_mask, feature_names=feature_names).transform(X)


class PenalizedLogisticRanker:
    """Deterministic proximal-gradient logistic reference ranker."""

    def __init__(
        self,
        *,
        penalty_strength: float = 0.1,
        l1_ratio: float = 0.0,
        max_iterations: int = 5000,
        tolerance: float = 1e-7,
    ) -> None:
        if penalty_strength < 0 or not 0 <= l1_ratio <= 1:
            raise ValueError("penalty_strength must be non-negative and l1_ratio must be in [0, 1]")
        if max_iterations < 1 or tolerance <= 0:
            raise ValueError("max_iterations and tolerance must be positive")
        self.penalty_strength = float(penalty_strength)
        self.l1_ratio = float(l1_ratio)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.fitted_ = False

    def fit(self, X: Any, y: Sequence[int], *, sample_weight: Sequence[float] | None = None) -> "PenalizedLogisticRanker":
        matrix = _as_float_matrix(X)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("logistic input must be finite after preprocessing")
        target = np.asarray(y, dtype=np.int8)
        if target.shape != (len(matrix),) or not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("y must be a binary positive-reference indicator")
        if len(np.unique(target)) != 2:
            raise ValueError("logistic training requires positive and unlabeled samples")
        if sample_weight is None:
            weights = np.ones(len(matrix), dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (len(matrix),) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("sample_weight must contain finite positive values")
        weights = weights / float(np.mean(weights))
        weight_sum = float(np.sum(weights))
        augmented = np.concatenate([matrix, np.ones((len(matrix), 1), dtype=np.float64)], axis=1)
        weighted = augmented * np.sqrt(weights[:, None] / weight_sum)
        spectral = float(np.linalg.norm(weighted, ord=2))
        smooth_l2 = self.penalty_strength * (1.0 - self.l1_ratio)
        lipschitz = max(0.25 * spectral * spectral + smooth_l2, 1e-8)
        step = 1.0 / lipschitz
        coef = np.zeros(matrix.shape[1], dtype=np.float64)
        intercept = _safe_logit(float(np.sum(weights * target) / weight_sum))
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            linear = np.clip(matrix @ coef + intercept, -40.0, 40.0)
            score = _sigmoid(linear)
            residual = weights * (score - target) / weight_sum
            grad_coef = matrix.T @ residual + smooth_l2 * coef
            grad_intercept = float(np.sum(residual))
            candidate = coef - step * grad_coef
            threshold = step * self.penalty_strength * self.l1_ratio
            candidate = np.sign(candidate) * np.maximum(np.abs(candidate) - threshold, 0.0)
            candidate_intercept = intercept - step * grad_intercept
            delta = max(float(np.max(np.abs(candidate - coef), initial=0.0)), abs(candidate_intercept - intercept))
            scale = 1.0 + max(float(np.max(np.abs(coef), initial=0.0)), abs(intercept))
            coef, intercept = candidate, candidate_intercept
            if delta <= self.tolerance * scale:
                converged = True
                break
        if not converged:
            raise RuntimeError("penalized logistic solver did not converge")
        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.n_iterations_ = int(iteration)
        self.converged_ = True
        self.fitted_ = True
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ranker must be fitted before scoring")
        matrix = _as_float_matrix(X)
        if matrix.shape[1] != len(self.coef_) or not np.all(np.isfinite(matrix)):
            raise ValueError("scoring matrix is non-finite or has the wrong column count")
        return matrix @ self.coef_ + self.intercept_

    def predict_score(self, X: Any) -> np.ndarray:
        """Return a bounded positive-reference ranking score."""

        return _sigmoid(np.clip(self.decision_function(X), -40.0, 40.0))


class BaggedPURanker:
    """Bag positive-reference versus resampled unlabeled logistic rankers."""

    def __init__(
        self,
        *,
        bags: int = 32,
        unlabeled_ratio: float = 2.0,
        penalty_strength: float = 0.1,
        seed: int = 0,
        max_iterations: int = 5000,
        tolerance: float = 1e-7,
    ) -> None:
        if bags < 2 or unlabeled_ratio <= 0:
            raise ValueError("bags must be at least two and unlabeled_ratio must be positive")
        try:
            logistic_tolerance = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("tolerance must be finite and positive") from exc
        if not np.isfinite(logistic_tolerance) or logistic_tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.bags = int(bags)
        self.unlabeled_ratio = float(unlabeled_ratio)
        self.penalty_strength = float(penalty_strength)
        self.seed = int(seed)
        self.max_iterations = int(max_iterations)
        self.tolerance = logistic_tolerance
        self.fitted_ = False

    def fit(self, X: Any, y: Sequence[int]) -> "BaggedPURanker":
        matrix = _as_float_matrix(X)
        target = np.asarray(y, dtype=np.int8)
        if target.shape != (len(matrix),) or not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("y must be a binary positive-reference indicator")
        positives = np.flatnonzero(target == 1)
        unlabeled = np.flatnonzero(target == 0)
        if len(positives) < 2 or len(unlabeled) < 2:
            raise ValueError("bagged PU training requires at least two positives and two unlabeled candidates")
        sample_count = min(len(unlabeled), max(2, int(math.ceil(self.unlabeled_ratio * len(positives)))))
        rng = np.random.default_rng(self.seed)
        models: list[PenalizedLogisticRanker] = []
        sampled_unlabeled: list[tuple[int, ...]] = []
        for _ in range(self.bags):
            chosen = np.sort(rng.choice(unlabeled, size=sample_count, replace=False))
            indices = np.concatenate([positives, chosen])
            bag_target = np.concatenate([np.ones(len(positives), dtype=np.int8), np.zeros(len(chosen), dtype=np.int8)])
            model = PenalizedLogisticRanker(
                penalty_strength=self.penalty_strength,
                l1_ratio=0.0,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )
            model.fit(matrix[indices], bag_target, sample_weight=_balanced_reference_weights(bag_target))
            models.append(model)
            sampled_unlabeled.append(tuple(int(index) for index in chosen))
        self.models_ = tuple(models)
        self.sampled_unlabeled_indices_ = tuple(sampled_unlabeled)
        self.fitted_ = True
        return self

    def predict_score(self, X: Any) -> np.ndarray:
        """Return the bag-mean positive-reference ranking score."""

        if not self.fitted_:
            raise RuntimeError("ranker must be fitted before scoring")
        components = np.vstack([model.predict_score(X) for model in self.models_])
        return np.mean(components, axis=0)

    def component_scores(self, X: Any) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ranker must be fitted before scoring")
        return np.vstack([model.predict_score(X) for model in self.models_])


class TinyTanhMLPRanker:
    """A deterministic one-hidden-layer ranker with exactly four tanh units."""

    hidden_units = 4

    def __init__(
        self,
        *,
        seed: int = 0,
        l2: float = 0.01,
        learning_rate: float = 0.02,
        max_epochs: int = 400,
        patience: int = 35,
        min_delta: float = 1e-6,
    ) -> None:
        if l2 < 0 or learning_rate <= 0 or max_epochs < 1 or patience < 1 or min_delta < 0:
            raise ValueError("invalid MLP optimisation configuration")
        self.seed = int(seed)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.fitted_ = False

    def fit(
        self,
        X: Any,
        y: Sequence[int],
        *,
        sample_weight: Sequence[float] | None = None,
        validation: tuple[Any, Sequence[int], Sequence[float] | None] | None = None,
        fixed_epochs: int | None = None,
    ) -> "TinyTanhMLPRanker":
        matrix = _as_float_matrix(X)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("MLP input must be finite after preprocessing")
        target = np.asarray(y, dtype=np.float64)
        if target.shape != (len(matrix),) or not set(np.unique(target)).issubset({0.0, 1.0}) or len(np.unique(target)) != 2:
            raise ValueError("MLP training requires positive and unlabeled samples")
        weights = _coerce_weights(sample_weight, len(matrix))
        if fixed_epochs is not None and (fixed_epochs < 1 or validation is not None):
            raise ValueError("fixed_epochs must be positive and cannot be combined with validation")
        validation_values: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if validation is not None:
            X_validation = _as_float_matrix(validation[0], name="validation X")
            y_validation = np.asarray(validation[1], dtype=np.float64)
            if X_validation.shape[1] != matrix.shape[1] or y_validation.shape != (len(X_validation),):
                raise ValueError("validation shapes do not match training data")
            if not set(np.unique(y_validation)).issubset({0.0, 1.0}) or len(np.unique(y_validation)) != 2:
                raise ValueError("validation partition requires positive and unlabeled samples")
            validation_values = (X_validation, y_validation, _coerce_weights(validation[2], len(X_validation)))

        rng = np.random.default_rng(self.seed)
        limit = math.sqrt(6.0 / (matrix.shape[1] + self.hidden_units))
        parameters = [
            rng.uniform(-limit, limit, size=(matrix.shape[1], self.hidden_units)),
            np.zeros(self.hidden_units, dtype=np.float64),
            rng.uniform(-limit, limit, size=self.hidden_units),
            np.zeros(1, dtype=np.float64),
        ]
        first = [np.zeros_like(value) for value in parameters]
        second = [np.zeros_like(value) for value in parameters]
        best = [value.copy() for value in parameters]
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        history: list[dict[str, float | int]] = []
        epochs = int(fixed_epochs if fixed_epochs is not None else self.max_epochs)
        for epoch in range(1, epochs + 1):
            hidden, score = _mlp_forward(matrix, parameters)
            residual = weights * (score - target) / float(np.sum(weights))
            gradients = [
                matrix.T @ ((residual[:, None] * parameters[2][None, :]) * (1.0 - hidden * hidden)) + self.l2 * parameters[0],
                np.sum((residual[:, None] * parameters[2][None, :]) * (1.0 - hidden * hidden), axis=0),
                hidden.T @ residual + self.l2 * parameters[2],
                np.asarray([np.sum(residual)], dtype=np.float64),
            ]
            for index, gradient in enumerate(gradients):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * (gradient * gradient)
                first_hat = first[index] / (1.0 - 0.9**epoch)
                second_hat = second[index] / (1.0 - 0.999**epoch)
                parameters[index] -= self.learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
            if not all(np.all(np.isfinite(value)) for value in parameters):
                raise RuntimeError("MLP optimisation produced non-finite parameters")
            train_loss = _mlp_loss(matrix, target, weights, parameters, self.l2)
            if validation_values is None:
                monitored = train_loss
            else:
                monitored = _mlp_loss(*validation_values, parameters, self.l2)
            history.append({"epoch": epoch, "training_reference_loss": train_loss, "stopping_reference_loss": monitored})
            if monitored < best_loss - self.min_delta:
                best_loss = monitored
                best_epoch = epoch
                best = [value.copy() for value in parameters]
                stale = 0
            else:
                stale += 1
            if fixed_epochs is None and validation_values is not None and stale >= self.patience:
                break
        if best_epoch < 1 or not np.isfinite(best_loss):
            raise RuntimeError("MLP did not produce a finite checkpoint")
        self.W1_, self.b1_, self.W2_, self.b2_ = best
        self.best_epoch_ = int(best_epoch)
        self.n_epochs_ = int(len(history))
        self.best_reference_loss_ = float(best_loss)
        self.history_ = tuple(history)
        self.parameter_count_ = int(sum(value.size for value in best))
        self.fitted_ = True
        return self

    def predict_score(self, X: Any) -> np.ndarray:
        """Return a bounded positive-reference ranking score."""

        if not self.fitted_:
            raise RuntimeError("ranker must be fitted before scoring")
        matrix = _as_float_matrix(X)
        if matrix.shape[1] != self.W1_.shape[0] or not np.all(np.isfinite(matrix)):
            raise ValueError("scoring matrix is non-finite or has the wrong column count")
        return _mlp_forward(matrix, [self.W1_, self.b1_, self.W2_, self.b2_])[1]


def _coerce_weights(values: Sequence[float] | None, length: int) -> np.ndarray:
    if values is None:
        result = np.ones(length, dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (length,) or np.any(~np.isfinite(result)) or np.any(result <= 0):
            raise ValueError("weights must be finite and positive")
    return result / float(np.mean(result))


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _safe_logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(math.log(clipped / (1.0 - clipped)))


def _mlp_forward(matrix: np.ndarray, parameters: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(matrix @ parameters[0] + parameters[1])
    score = _sigmoid(np.clip(hidden @ parameters[2] + float(parameters[3][0]), -40.0, 40.0))
    return hidden, score


def _mlp_loss(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    parameters: Sequence[np.ndarray],
    l2: float,
) -> float:
    score = np.clip(_mlp_forward(matrix, parameters)[1], 1e-12, 1.0 - 1e-12)
    reference_loss = -float(np.sum(weights * (target * np.log(score) + (1.0 - target) * np.log1p(-score))) / np.sum(weights))
    penalty = 0.5 * l2 * (float(np.sum(parameters[0] ** 2)) + float(np.sum(parameters[2] ** 2)))
    return reference_loss + penalty


def grouped_spatial_folds(
    y: Sequence[int],
    spatial_groups: Sequence[Any],
    *,
    n_splits: int,
    seed: int,
    attempts: int = 64,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create deterministic class-supporting folds with whole spatial groups."""

    target = np.asarray(y, dtype=np.int8)
    groups = np.asarray([str(value) for value in spatial_groups], dtype=object)
    if target.ndim != 1 or groups.shape != target.shape or not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("y and spatial_groups must be aligned binary/reference vectors")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < n_splits:
        raise ValueError(f"{len(unique_groups)} spatial groups cannot support {n_splits} folds")
    positive_counts = np.bincount(inverse, weights=target, minlength=len(unique_groups)).astype(int)
    total_counts = np.bincount(inverse, minlength=len(unique_groups)).astype(int)
    unlabeled_counts = total_counts - positive_counts
    if int(np.sum(positive_counts > 0)) < n_splits or int(np.sum(unlabeled_counts > 0)) < n_splits:
        raise ValueError(
            "insufficient positive/unlabeled spatial-group support for requested folds: "
            f"positive_groups={int(np.sum(positive_counts > 0))}, "
            f"unlabeled_groups={int(np.sum(unlabeled_counts > 0))}, folds={n_splits}"
        )
    target_positive = float(np.sum(positive_counts)) / n_splits
    target_unlabeled = float(np.sum(unlabeled_counts)) / n_splits
    target_total = float(np.sum(total_counts)) / n_splits
    target_group_count = len(unique_groups) / n_splits
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_cost = float("inf")
    for attempt in range(attempts):
        jitter = rng.random(len(unique_groups))
        order = sorted(
            range(len(unique_groups)),
            key=lambda index: (
                -max(
                    positive_counts[index] / max(target_positive, 1.0),
                    unlabeled_counts[index] / max(target_unlabeled, 1.0),
                ),
                -total_counts[index],
                jitter[index],
            ),
        )
        fold_positive = np.zeros(n_splits, dtype=float)
        fold_unlabeled = np.zeros(n_splits, dtype=float)
        fold_total = np.zeros(n_splits, dtype=float)
        fold_groups = np.zeros(n_splits, dtype=int)
        assignment = np.full(len(unique_groups), -1, dtype=int)
        for position, group_index in enumerate(order):
            if position < n_splits:
                candidates = np.flatnonzero(fold_groups == 0)
            else:
                candidates = np.arange(n_splits)
            candidate_order = rng.permutation(candidates)
            costs: list[tuple[float, int]] = []
            for fold in candidate_order:
                candidate_positive = fold_positive.copy()
                candidate_unlabeled = fold_unlabeled.copy()
                candidate_total = fold_total.copy()
                candidate_groups = fold_groups.copy()
                candidate_positive[fold] += positive_counts[group_index]
                candidate_unlabeled[fold] += unlabeled_counts[group_index]
                candidate_total[fold] += total_counts[group_index]
                candidate_groups[fold] += 1
                cost = (
                    np.sum(((candidate_positive - target_positive) / max(target_positive, 1.0)) ** 2)
                    + np.sum(((candidate_unlabeled - target_unlabeled) / max(target_unlabeled, 1.0)) ** 2)
                    + 0.2 * np.sum(((candidate_total - target_total) / max(target_total, 1.0)) ** 2)
                    + 0.05 * np.sum((candidate_groups - target_group_count) ** 2)
                )
                costs.append((float(cost), int(fold)))
            _, selected = min(costs, key=lambda item: item[0])
            assignment[group_index] = selected
            fold_positive[selected] += positive_counts[group_index]
            fold_unlabeled[selected] += unlabeled_counts[group_index]
            fold_total[selected] += total_counts[group_index]
            fold_groups[selected] += 1
        valid = bool(np.all(fold_positive > 0) and np.all(fold_unlabeled > 0))
        valid = valid and bool(np.all(np.sum(positive_counts) - fold_positive > 0) and np.all(np.sum(unlabeled_counts) - fold_unlabeled > 0))
        if not valid:
            continue
        cost = float(
            np.sum(((fold_positive - target_positive) / max(target_positive, 1.0)) ** 2)
            + np.sum(((fold_unlabeled - target_unlabeled) / max(target_unlabeled, 1.0)) ** 2)
            + 0.2 * np.sum(((fold_total - target_total) / max(target_total, 1.0)) ** 2)
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise ValueError("unable to construct leakage-free folds with positive and unlabeled support")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(target))
    for fold in range(n_splits):
        test_groups = np.flatnonzero(best_assignment == fold)
        test = np.flatnonzero(np.isin(inverse, test_groups))
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("spatial group leakage detected")
        if len(np.unique(target[train])) != 2 or len(np.unique(target[test])) != 2:
            raise RuntimeError("constructed fold lacks positive or unlabeled support")
        folds.append((train, test))
    return tuple(folds)


def positive_vs_unlabeled_rank_auc(positive_mask: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the positive-versus-unlabeled pairwise ranking curve."""

    positive = np.asarray(positive_mask, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if positive.shape != values.shape or positive.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("positive_mask and finite scores must be aligned vectors")
    n_positive = int(np.sum(positive))
    n_unlabeled = int(np.sum(~positive))
    if n_positive < 1 or n_unlabeled < 1:
        raise ValueError("positive-vs-unlabeled rank AUC requires both reference roles")
    ranks = _average_ranks(values)
    rank_sum = float(np.sum(ranks[positive]))
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_unlabeled))


def spu_auc(positive_mask: Sequence[bool], scores: Sequence[float]) -> float:
    """Concise alias for :func:`positive_vs_unlabeled_rank_auc`."""

    return positive_vs_unlabeled_rank_auc(positive_mask, scores)


def rank_metrics(
    positive_mask: Sequence[bool],
    scores: Sequence[float],
    *,
    budget: int,
    tie_breakers: Sequence[Any] | None = None,
) -> dict[str, float | int]:
    """Compute sparse-positive ranking metrics at a fixed candidate budget."""

    positive = np.asarray(positive_mask, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if positive.shape != values.shape or positive.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("positive_mask and finite scores must be aligned vectors")
    n_positive = int(np.sum(positive))
    n_unlabeled = int(np.sum(~positive))
    if n_positive < 1 or n_unlabeled < 1:
        raise ValueError("rank metrics require positive and unlabeled samples")
    if budget < 1:
        raise ValueError("budget must be positive")
    if tie_breakers is None:
        order = np.argsort(-values, kind="mergesort")
    else:
        tie = np.asarray([str(value) for value in tie_breakers], dtype=object)
        if tie.shape != values.shape or len(set(tie.tolist())) != len(tie):
            raise ValueError("tie_breakers must be aligned and unique")
        order = np.lexsort((tie, -values))
    effective_budget = min(int(budget), len(values))
    recovered = int(np.sum(positive[order[:effective_budget]]))
    descending_ranks = _average_ranks(-values)
    positive_ranks = descending_ranks[positive]
    percentiles = 1.0 - (positive_ranks - 1.0) / max(len(values) - 1, 1)
    return {
        "positive_vs_unlabeled_rank_auc": positive_vs_unlabeled_rank_auc(positive, values),
        "positive_recall_at_budget": float(recovered / n_positive),
        "positive_recovered_at_budget": recovered,
        "candidate_budget": int(budget),
        "effective_candidate_budget": effective_budget,
        "positive_count": n_positive,
        "unlabeled_reference_count": n_unlabeled,
        "mean_positive_rank": float(np.mean(positive_ranks)),
        "median_positive_rank": float(np.median(positive_ranks)),
        "mean_positive_rank_percentile": float(np.mean(percentiles)),
        "reciprocal_best_positive_rank": float(1.0 / np.min(positive_ranks)),
    }


def fold_percentile_normalize(
    scores: Sequence[float],
    test_folds: Sequence[Sequence[int]],
) -> np.ndarray:
    """Map each held-out fold to label-free mid-rank percentiles.

    This removes arbitrary score-location/scale differences between separately
    fitted outer-fold models before a global fixed-budget ranking is formed.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    normalized = np.full(len(values), np.nan, dtype=np.float64)
    covered: list[int] = []
    for fold_index, raw_test in enumerate(test_folds, start=1):
        test = np.asarray(raw_test, dtype=int)
        if test.ndim != 1 or len(test) < 1 or len(np.unique(test)) != len(test):
            raise ValueError(f"test fold {fold_index} must contain unique indices")
        if np.any(test < 0) or np.any(test >= len(values)):
            raise ValueError(f"test fold {fold_index} contains an out-of-range index")
        ranks = _average_ranks(values[test])
        normalized[test] = (ranks - 0.5) / len(test)
        covered.extend(int(index) for index in test)
    if sorted(covered) != list(range(len(values))):
        raise ValueError("test folds must cover each OOF row exactly once")
    if np.any(~np.isfinite(normalized)):
        raise RuntimeError("fold percentile normalization left missing scores")
    return normalized


def summarize_folded_oof_ranking(
    positive_mask: Sequence[bool],
    raw_scores: Sequence[float],
    test_folds: Sequence[Sequence[int]],
    *,
    budget: int,
    tie_breakers: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Summarize per-fold ranking and fold-percentile global ranking."""

    positive = np.asarray(positive_mask, dtype=bool)
    values = np.asarray(raw_scores, dtype=np.float64)
    if positive.shape != values.shape:
        raise ValueError("positive_mask and raw_scores must be aligned")
    if tie_breakers is None:
        ties = np.asarray([f"row_{index:08d}" for index in range(len(values))], dtype=object)
    else:
        ties = np.asarray([str(value) for value in tie_breakers], dtype=object)
        if ties.shape != values.shape:
            raise ValueError("tie_breakers must align to scores")
    normalized = fold_percentile_normalize(values, test_folds)
    per_fold: list[dict[str, Any]] = []
    fold_areas: list[float] = []
    for fold_index, raw_test in enumerate(test_folds, start=1):
        test = np.asarray(raw_test, dtype=int)
        fold_budget = max(1, int(round(budget * len(test) / len(values))))
        metrics = rank_metrics(positive[test], values[test], budget=fold_budget, tie_breakers=ties[test])
        area = float(metrics["positive_vs_unlabeled_rank_auc"])
        fold_areas.append(area)
        per_fold.append(
            {
                "outer_fold": fold_index,
                "candidate_count": int(len(test)),
                "positive_count": int(np.sum(positive[test])),
                "unlabeled_count": int(np.sum(~positive[test])),
                "proportional_candidate_budget": int(fold_budget),
                "metrics": metrics,
            }
        )
    global_metrics = rank_metrics(positive, normalized, budget=budget, tie_breakers=ties)
    global_metrics["macro_fold_positive_vs_unlabeled_rank_auc"] = float(np.mean(fold_areas))
    return {
        "primary_metric_name": "macro_fold_positive_vs_unlabeled_rank_auc",
        "metrics": global_metrics,
        "per_fold_metrics": per_fold,
        "oof_scores_fold_percentile": [float(value) for value in normalized],
        "raw_score_global_sensitivity_metrics": rank_metrics(positive, values, budget=budget, tie_breakers=ties),
    }


def cross_seed_rank_stability(seed_scores: Mapping[int, Sequence[float]]) -> dict[str, Any]:
    """Summarise pairwise Spearman rank stability across deterministic seeds."""

    if len(seed_scores) < 2:
        raise ValueError("rank stability requires at least two seeds")
    ordered = sorted((int(seed), np.asarray(values, dtype=np.float64)) for seed, values in seed_scores.items())
    lengths = {len(values) for _, values in ordered}
    if len(lengths) != 1 or any(np.any(~np.isfinite(values)) for _, values in ordered):
        raise ValueError("all seed score vectors must be finite and equally sized")
    correlations: list[dict[str, float | int]] = []
    for left_index, (left_seed, left_values) in enumerate(ordered):
        left_ranks = _average_ranks(left_values)
        for right_seed, right_values in ordered[left_index + 1 :]:
            right_ranks = _average_ranks(right_values)
            correlation = _correlation(left_ranks, right_ranks)
            correlations.append({"seed_a": left_seed, "seed_b": right_seed, "spearman": correlation})
    values = np.asarray([float(row["spearman"]) for row in correlations], dtype=np.float64)
    return {
        "seed_count": len(ordered),
        "pair_count": len(correlations),
        "median_pairwise_spearman": float(np.median(values)),
        "minimum_pairwise_spearman": float(np.min(values)),
        "maximum_pairwise_spearman": float(np.max(values)),
        "pairs": correlations,
    }


def grouped_paired_bootstrap_rank_delta(
    positive_mask: Sequence[bool],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    positive_identity_groups: Sequence[Any],
    unlabeled_spatial_groups: Sequence[Any],
    *,
    draws: int,
    seed: int,
    score_a_name: str = "score_a",
    score_b_name: str = "score_b",
) -> dict[str, Any]:
    """Paired group bootstrap of an aligned OOF ranking-score contrast.

    Positive identities and unlabeled spatial groups are resampled separately
    with replacement.  All rows belonging to a sampled group move together;
    both score vectors use the exact same draw.
    """

    positive = np.asarray(positive_mask, dtype=bool)
    left = np.asarray(scores_a, dtype=np.float64)
    right = np.asarray(scores_b, dtype=np.float64)
    positive_groups = np.asarray([str(value) for value in positive_identity_groups], dtype=object)
    unlabeled_groups = np.asarray([str(value) for value in unlabeled_spatial_groups], dtype=object)
    if (
        positive.ndim != 1
        or left.shape != positive.shape
        or right.shape != positive.shape
        or positive_groups.shape != positive.shape
        or unlabeled_groups.shape != positive.shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("labels, aligned OOF scores, and group vectors must be finite aligned vectors")
    if draws < 1:
        raise ValueError("draws must be positive")
    unique_positive = np.unique(positive_groups[positive])
    unique_unlabeled = np.unique(unlabeled_groups[~positive])
    if len(unique_positive) < 2 or len(unique_unlabeled) < 2:
        raise ValueError("paired group bootstrap requires at least two positive identities and two unlabeled spatial groups")
    observed_left = positive_vs_unlabeled_rank_auc(positive, left)
    observed_right = positive_vs_unlabeled_rank_auc(positive, right)
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_positive = rng.choice(unique_positive, size=len(unique_positive), replace=True)
        sampled_unlabeled = rng.choice(unique_unlabeled, size=len(unique_unlabeled), replace=True)
        indices: list[int] = []
        labels: list[bool] = []
        for group in sampled_positive:
            group_indices = np.flatnonzero(positive & (positive_groups == group))
            indices.extend(int(index) for index in group_indices)
            labels.extend([True] * len(group_indices))
        for group in sampled_unlabeled:
            group_indices = np.flatnonzero((~positive) & (unlabeled_groups == group))
            indices.extend(int(index) for index in group_indices)
            labels.extend([False] * len(group_indices))
        sampled_indices = np.asarray(indices, dtype=int)
        sampled_labels = np.asarray(labels, dtype=bool)
        if len(sampled_indices) < 2 or len(np.unique(sampled_labels)) != 2:
            raise RuntimeError("paired group bootstrap draw lost a reference role")
        deltas[draw] = positive_vs_unlabeled_rank_auc(sampled_labels, left[sampled_indices]) - positive_vs_unlabeled_rank_auc(
            sampled_labels, right[sampled_indices]
        )
    return {
        "estimand": "paired delta in positive-vs-unlabeled rank AUC",
        "score_a_name": str(score_a_name),
        "score_b_name": str(score_b_name),
        "observed_score_a": float(observed_left),
        "observed_score_b": float(observed_right),
        "observed_delta_a_minus_b": float(observed_left - observed_right),
        "draws": int(draws),
        "seed": int(seed),
        "positive_resampling_unit": "positive_identity_group",
        "unlabeled_resampling_unit": "unlabeled_spatial_group",
        "positive_identity_group_count": int(len(unique_positive)),
        "unlabeled_spatial_group_count": int(len(unique_unlabeled)),
        "bootstrap_delta_median": float(np.median(deltas)),
        "bootstrap_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "bootstrap_delta_ci95_high": float(np.quantile(deltas, 0.975)),
    }


def spatial_group_score_permutation_null(
    positive_mask: Sequence[bool],
    scores: Sequence[float],
    spatial_groups: Sequence[Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Test score association after permuting whole spatial-group assignments.

    The candidate-level sparse labels are aggregated to ``any positive`` per
    spatial group and compared with that group's maximum score.  Permuting the
    complete group scores preserves within-group dependence and avoids a false
    row-independence assumption.
    """

    positive = np.asarray(positive_mask, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    groups = np.asarray([str(value) for value in spatial_groups], dtype=object)
    if positive.shape != values.shape or groups.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("positive labels, scores, and spatial groups must be aligned and finite")
    if draws < 1:
        raise ValueError("draws must be positive")
    unique = np.unique(groups)
    group_positive = np.asarray([np.any(positive[groups == group]) for group in unique], dtype=bool)
    group_score = np.asarray([np.max(values[groups == group]) for group in unique], dtype=np.float64)
    if int(np.sum(group_positive)) < 1 or int(np.sum(~group_positive)) < 1:
        raise ValueError("group permutation null requires positive and unlabeled-only spatial groups")
    observed = positive_vs_unlabeled_rank_auc(group_positive, group_score)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        null[index] = positive_vs_unlabeled_rank_auc(group_positive, group_score[rng.permutation(len(group_score))])
    return {
        "null_semantics": "whole-spatial-group score-assignment permutation; labels and models are not refitted",
        "unit": "spatial_group",
        "group_score_aggregation": "maximum_candidate_score",
        "group_label_aggregation": "any_strict_or_policy_positive",
        "group_count": len(unique),
        "positive_group_count": int(np.sum(group_positive)),
        "unlabeled_only_group_count": int(np.sum(~group_positive)),
        "observed_group_positive_vs_unlabeled_rank_auc": float(observed),
        "draws": int(draws),
        "greater_or_equal_p_value": float((1 + np.sum(null >= observed)) / (draws + 1)),
        "null_median": float(np.median(null)),
        "null_ci95_low": float(np.quantile(null, 0.025)),
        "null_ci95_high": float(np.quantile(null, 0.975)),
    }


def spatial_group_permutation_null(
    positive_mask: Sequence[bool],
    scores: Sequence[float],
    spatial_groups: Sequence[Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Backward-compatible alias for the score-assignment null."""

    return spatial_group_score_permutation_null(
        positive_mask,
        scores,
        spatial_groups,
        draws=draws,
        seed=seed,
    )


def validate_frozen_outer_folds(
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    included_input_indices: Sequence[int],
    positive_mask: Sequence[bool],
    spatial_groups: Sequence[Any],
    expected_folds: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Validate caller-frozen folds, including guard-band training omissions.

    ``folds`` use indices from the original input census.  Test sets must cover
    each included positive/unlabeled row exactly once.  A training set may omit
    complement rows, which is how a caller represents a predeclared spatial
    guard around a contiguous held-out block.
    """

    included = np.asarray(included_input_indices, dtype=int)
    positive = np.asarray(positive_mask, dtype=bool)
    groups = np.asarray([str(value) for value in spatial_groups], dtype=object)
    if included.ndim != 1 or len(np.unique(included)) != len(included):
        raise ValueError("included_input_indices must be a unique vector")
    if positive.shape != (len(included),) or groups.shape != (len(included),):
        raise ValueError("positive_mask and spatial_groups must align to included_input_indices")
    if len(folds) != expected_folds:
        raise ValueError(f"expected {expected_folds} frozen outer folds, received {len(folds)}")
    original_to_local = {int(original): local for local, original in enumerate(included)}
    covered: list[int] = []
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index, (raw_train, raw_test) in enumerate(folds, start=1):
        train_original = np.asarray(raw_train, dtype=int)
        test_original = np.asarray(raw_test, dtype=int)
        if train_original.ndim != 1 or test_original.ndim != 1 or len(train_original) == 0 or len(test_original) == 0:
            raise ValueError(f"frozen outer fold {fold_index} must have non-empty one-dimensional train/test indices")
        if len(np.unique(train_original)) != len(train_original) or len(np.unique(test_original)) != len(test_original):
            raise ValueError(f"frozen outer fold {fold_index} contains duplicate indices")
        if set(train_original.tolist()) & set(test_original.tolist()):
            raise ValueError(f"frozen outer fold {fold_index} has train/test row overlap")
        unknown = (set(train_original.tolist()) | set(test_original.tolist())) - set(original_to_local)
        if unknown:
            raise ValueError(f"frozen outer fold {fold_index} references excluded or unknown input rows")
        train = np.asarray([original_to_local[int(index)] for index in train_original], dtype=int)
        test = np.asarray([original_to_local[int(index)] for index in test_original], dtype=int)
        if set(groups[train]) & set(groups[test]):
            raise ValueError(f"frozen outer fold {fold_index} leaks spatial groups")
        if len(np.unique(positive[train])) != 2 or len(np.unique(positive[test])) != 2:
            raise ValueError(f"frozen outer fold {fold_index} lacks positive or unlabeled support")
        if int(np.sum(positive[train])) < 2:
            raise ValueError(f"frozen outer fold {fold_index} has fewer than two training positives")
        covered.extend(test.tolist())
        output.append((train, test))
    if sorted(covered) != list(range(len(included))):
        raise ValueError("frozen outer test folds must cover every included candidate exactly once")
    return tuple(output)


def nested_spatial_pu_evaluation(
    X: Any,
    labels: Sequence[Any],
    spatial_groups: Sequence[Any],
    *,
    feature_names: Sequence[str],
    sample_ids: Sequence[Any] | None = None,
    baseline_scores: Mapping[str, Sequence[float]] | None = None,
    equal_weight_feature_names: Sequence[str] | None = None,
    frozen_outer_folds: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None,
    label_policy: LabelPolicy = STRICT_LABEL_POLICY,
    config: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    """Run nested spatial evaluation of linear, PU, and tiny nonlinear rankers.

    The primary estimand is held-out known-positive ranking within the frozen
    positive-unlabeled candidate census.  It is not a fully labeled
    classification estimand or an independent-recording generalisation claim.
    """

    matrix = _as_float_matrix(X)
    names = validate_raw_feature_names(feature_names)
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length must match X columns")
    if len(labels) != len(matrix) or len(spatial_groups) != len(matrix):
        raise ValueError("X, labels, and spatial_groups must have equal row counts")
    resolved = resolve_labels(labels, label_policy)
    included = resolved.positive | resolved.unlabeled
    included_indices = np.flatnonzero(included)
    if len(included_indices) < 4:
        raise ValueError("too few included positive-unlabeled candidates")
    X_included = matrix[included]
    y = resolved.positive[included].astype(np.int8)
    groups = np.asarray([str(value) for value in spatial_groups], dtype=object)[included]
    if sample_ids is None:
        ids = np.asarray([f"candidate_{index:06d}" for index in included_indices], dtype=object)
    else:
        if len(sample_ids) != len(matrix):
            raise ValueError("sample_ids length must match X")
        ids = np.asarray([str(value) for value in sample_ids], dtype=object)[included]
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("included sample_ids must be unique")
    _validate_global_support(y, groups, config)
    if frozen_outer_folds is None:
        outer_folds = grouped_spatial_folds(y, groups, n_splits=config.outer_splits, seed=config.seed)
        outer_fold_source = "deterministic_group_balancing"
    else:
        outer_folds = validate_frozen_outer_folds(
            frozen_outer_folds,
            included_input_indices=included_indices,
            positive_mask=y == 1,
            spatial_groups=groups,
            expected_folds=config.outer_splits,
        )
        outer_fold_source = "caller_frozen_with_optional_training_guard_omissions"

    equal_weight_indices: tuple[int, ...] | None = None
    if equal_weight_feature_names is not None:
        requested = tuple(str(value) for value in equal_weight_feature_names)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("equal_weight_feature_names must be non-empty and unique")
        missing = sorted(set(requested) - set(names))
        if missing:
            raise ValueError("equal-weight features are absent from X: " + ", ".join(missing))
        equal_weight_indices = tuple(names.index(name) for name in requested)

    model_names = ["positive_reference_distance", "logistic_l2", "logistic_elastic", "bagged_pu_logistic", "tiny_mlp_4_tanh"]
    if equal_weight_indices is not None:
        model_names.insert(0, "equal_weight_feature_separation")
    oof = {name: np.full(len(y), np.nan, dtype=np.float64) for name in model_names}
    mlp_seed_oof = {int(seed): np.full(len(y), np.nan, dtype=np.float64) for seed in config.mlp_seeds}
    fold_records: list[dict[str, Any]] = []

    for outer_index, (outer_train, outer_test) in enumerate(outer_folds):
        selected: dict[str, float] = {}
        inner_summaries: dict[str, list[dict[str, float]]] = {}
        grids = {
            "logistic_l2": config.logistic_penalties,
            "logistic_elastic": config.elastic_penalties,
            "bagged_pu_logistic": config.pu_penalties,
            "tiny_mlp_4_tanh": config.mlp_l2_values,
        }
        if config.hyperparameter_selection == "fixed":
            for model_name, grid in grids.items():
                selected[model_name] = float(grid[0])
                inner_summaries[model_name] = [{"predeclared_fixed_penalty": float(grid[0])}]
        else:
            inner_folds = grouped_spatial_folds(
                y[outer_train],
                groups[outer_train],
                n_splits=config.inner_splits,
                seed=config.seed + 1009 * (outer_index + 1),
            )
            for model_name, grid in grids.items():
                candidates: list[dict[str, float]] = []
                for grid_index, penalty in enumerate(grid):
                    predictions = np.full(len(outer_train), np.nan, dtype=np.float64)
                    for inner_index, (inner_train_local, inner_test_local) in enumerate(inner_folds):
                        train = outer_train[inner_train_local]
                        test = outer_train[inner_test_local]
                        seed = config.seed + 100_003 * (outer_index + 1) + 1009 * (inner_index + 1) + grid_index
                        if model_name == "tiny_mlp_4_tanh":
                            prediction, _, _ = _fit_mlp_ensemble(
                                X_included,
                                y,
                                groups,
                                train,
                                test,
                                names,
                                l2=float(penalty),
                                config=config,
                                seed_offset=seed,
                            )
                        else:
                            preprocessor = RobustFoldPreprocessor(clip=config.preprocess_clip)
                            X_train = preprocessor.fit_transform(X_included[train], y[train] == 1, feature_names=names)
                            X_test = preprocessor.transform(X_included[test])
                            prediction = _fit_reference_model(
                                model_name,
                                X_train,
                                y[train],
                                X_test,
                                penalty=float(penalty),
                                config=config,
                                seed=seed,
                            )
                        predictions[inner_test_local] = prediction
                    if np.any(~np.isfinite(predictions)):
                        raise RuntimeError("inner cross-fitting left missing ranking scores")
                    candidates.append(
                        {
                            "penalty": float(penalty),
                            "inner_positive_vs_unlabeled_rank_auc": positive_vs_unlabeled_rank_auc(
                                y[outer_train] == 1, predictions
                            ),
                        }
                    )
                winner = max(
                    candidates,
                    key=lambda row: (float(row["inner_positive_vs_unlabeled_rank_auc"]), -float(row["penalty"])),
                )
                selected[model_name] = float(winner["penalty"])
                inner_summaries[model_name] = candidates

        preprocessor = RobustFoldPreprocessor(clip=config.preprocess_clip)
        X_train = preprocessor.fit_transform(X_included[outer_train], y[outer_train] == 1, feature_names=names)
        X_test = preprocessor.transform(X_included[outer_test])
        if equal_weight_indices is not None:
            oof["equal_weight_feature_separation"][outer_test] = np.mean(X_test[:, equal_weight_indices], axis=1)
        oof["positive_reference_distance"][outer_test] = -preprocessor.positive_reference_distance(
            X_included[outer_test]
        )
        for model_index, model_name in enumerate(("logistic_l2", "logistic_elastic", "bagged_pu_logistic")):
            oof[model_name][outer_test] = _fit_reference_model(
                model_name,
                X_train,
                y[outer_train],
                X_test,
                penalty=selected[model_name],
                config=config,
                seed=config.seed + 1_000_003 * (outer_index + 1) + model_index,
            )
        mlp_prediction, seed_predictions, epoch_records = _fit_mlp_ensemble(
            X_included,
            y,
            groups,
            outer_train,
            outer_test,
            names,
            l2=selected["tiny_mlp_4_tanh"],
            config=config,
            seed_offset=config.seed + 9_000_019 * (outer_index + 1),
        )
        oof["tiny_mlp_4_tanh"][outer_test] = mlp_prediction
        for mlp_seed, values in seed_predictions.items():
            mlp_seed_oof[mlp_seed][outer_test] = values
        fold_records.append(
            {
                "outer_fold": outer_index + 1,
                "train_candidate_count": int(len(outer_train)),
                "test_candidate_count": int(len(outer_test)),
                "train_positive_count": int(np.sum(y[outer_train] == 1)),
                "test_positive_count": int(np.sum(y[outer_test] == 1)),
                "train_spatial_group_count": int(len(np.unique(groups[outer_train]))),
                "test_spatial_group_count": int(len(np.unique(groups[outer_test]))),
                "spatial_groups_disjoint": not bool(set(groups[outer_train]) & set(groups[outer_test])),
                "training_guard_omitted_candidate_count": int(len(y) - len(outer_train) - len(outer_test)),
                "selected_penalties": selected,
                "inner_selection": inner_summaries,
                "mlp_fixed_epochs_from_training_only": epoch_records,
                "learned_model_input_width": int(X_train.shape[1]),
                "learned_model_feature_names": list(preprocessor.feature_names_out_),
                "tiny_mlp_parameter_count": int(X_train.shape[1] * 4 + 4 + 4 + 1),
            }
        )

    for model_name, values in oof.items():
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"outer cross-fitting left missing scores for {model_name}")
    for seed, values in mlp_seed_oof.items():
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"outer cross-fitting left missing MLP scores for seed {seed}")

    test_folds = [test for _, test in outer_folds]
    models: dict[str, Any] = {}
    for model_index, model_name in enumerate(model_names):
        ranking_summary = summarize_folded_oof_ranking(
            y == 1,
            oof[model_name],
            test_folds,
            budget=config.candidate_budget,
            tie_breakers=ids,
        )
        percentile_scores = np.asarray(ranking_summary["oof_scores_fold_percentile"], dtype=np.float64)
        model_payload = {
            "score_semantics": "global ranking uses label-free within-test-fold mid-rank percentiles",
            **ranking_summary,
            "oof_scores": [float(value) for value in percentile_scores],
            "oof_scores_raw_fold_specific": [float(value) for value in oof[model_name]],
        }
        if config.permutation_draws > 0:
            model_payload["spatial_group_score_association_null"] = spatial_group_score_permutation_null(
                y == 1,
                percentile_scores,
                groups,
                draws=config.permutation_draws,
                seed=config.seed + 17_171 * (model_index + 1),
            )
        models[model_name] = model_payload
    mlp_seed_percentile = {
        seed: fold_percentile_normalize(values, test_folds) for seed, values in mlp_seed_oof.items()
    }
    models["tiny_mlp_4_tanh"]["seed_rank_stability"] = cross_seed_rank_stability(mlp_seed_percentile)
    models["tiny_mlp_4_tanh"]["seed_oof_scores"] = {
        str(seed): [float(value) for value in values] for seed, values in sorted(mlp_seed_percentile.items())
    }
    models["tiny_mlp_4_tanh"]["seed_oof_scores_raw_fold_specific"] = {
        str(seed): [float(value) for value in values] for seed, values in sorted(mlp_seed_oof.items())
    }

    baselines: dict[str, Any] = {}
    for baseline_index, (name, raw_scores) in enumerate(sorted((baseline_scores or {}).items())):
        values = np.asarray(raw_scores, dtype=np.float64)
        if values.shape != (len(matrix),):
            raise ValueError(f"baseline {name!r} length must match X")
        values = values[included]
        if np.any(~np.isfinite(values)):
            raise ValueError(f"baseline {name!r} contains non-finite scores")
        ranking_summary = summarize_folded_oof_ranking(
            y == 1,
            values,
            test_folds,
            budget=config.candidate_budget,
            tie_breakers=ids,
        )
        percentile_scores = np.asarray(ranking_summary["oof_scores_fold_percentile"], dtype=np.float64)
        baseline_payload = {
            "score_semantics": "frozen external score; global ranking uses label-free within-test-fold mid-rank percentiles",
            **ranking_summary,
            "scores": [float(value) for value in percentile_scores],
            "scores_raw_fold_specific": [float(value) for value in values],
        }
        if config.permutation_draws > 0:
            baseline_payload["spatial_group_score_association_null"] = spatial_group_score_permutation_null(
                y == 1,
                percentile_scores,
                groups,
                draws=config.permutation_draws,
                seed=config.seed + 101_111 * (baseline_index + 1),
            )
        baselines[str(name)] = baseline_payload

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_engineering_evaluation",
        "estimand": "held-out known-positive ranking within a frozen positive-unlabeled candidate census",
        "interpretation_limits": [
            "unlabeled candidates remain unknown",
            "positive-versus-unlabeled rank AUC is not a fully labeled discrimination metric",
            "scores have no binary calibration interpretation",
            "spatially grouped within-census evaluation is not independent-recording generalisation",
            "candidate-selected yield does not measure full-field error rates",
        ],
        "label_policy": asdict(label_policy),
        "config": asdict(config),
        "feature_names": list(names),
        "raw_feature_contract": "all imputation, scaling, missingness selection, and positive-reference statistics are fold-local",
        "included_candidate_indices": [int(index) for index in included_indices],
        "included_sample_ids": [str(value) for value in ids],
        "candidate_count": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "unlabeled_count": int(np.sum(y == 0)),
        "excluded_count": int(np.sum(resolved.excluded)),
        "spatial_group_count": int(len(np.unique(groups))),
        "outer_fold_source": outer_fold_source,
        "equal_weight_feature_names": list(equal_weight_feature_names or ()),
        "outer_folds": fold_records,
        "models": models,
        "baselines": baselines,
    }


def _validate_global_support(y: np.ndarray, groups: np.ndarray, config: EvaluationConfig) -> None:
    positive_groups = len(np.unique(groups[y == 1]))
    unlabeled_groups = len(np.unique(groups[y == 0]))
    selection_requirement = config.inner_splits + 1 if config.hyperparameter_selection == "nested" else 3
    required = max(config.outer_splits, selection_requirement)
    if positive_groups < required or unlabeled_groups < required:
        raise ValueError(
            "candidate census cannot support nested spatial evaluation and MLP early stopping: "
            f"positive_groups={positive_groups}, unlabeled_groups={unlabeled_groups}, required_at_least={required}"
        )


def _fit_reference_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    penalty: float,
    config: EvaluationConfig,
    seed: int,
) -> np.ndarray:
    if model_name == "logistic_l2":
        model = PenalizedLogisticRanker(
            penalty_strength=penalty,
            l1_ratio=0.0,
            max_iterations=config.max_logistic_iterations,
            tolerance=config.logistic_tolerance,
        )
        model.fit(X_train, y_train, sample_weight=_balanced_reference_weights(y_train))
    elif model_name == "logistic_elastic":
        model = PenalizedLogisticRanker(
            penalty_strength=penalty,
            l1_ratio=0.5,
            max_iterations=config.max_logistic_iterations,
            tolerance=config.logistic_tolerance,
        )
        model.fit(X_train, y_train, sample_weight=_balanced_reference_weights(y_train))
    elif model_name == "bagged_pu_logistic":
        model = BaggedPURanker(
            bags=config.pu_bags,
            unlabeled_ratio=config.pu_unlabeled_ratio,
            penalty_strength=penalty,
            seed=seed,
            max_iterations=config.max_logistic_iterations,
            tolerance=config.logistic_tolerance,
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"unknown reference model: {model_name}")
    prediction = model.predict_score(X_test)
    if prediction.shape != (len(X_test),) or np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model_name} produced invalid ranking scores")
    return prediction


def _fit_mlp_ensemble(
    raw_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    feature_names: Sequence[str],
    *,
    l2: float,
    config: EvaluationConfig,
    seed_offset: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], list[dict[str, int]]]:
    early_folds = grouped_spatial_folds(
        y[train],
        groups[train],
        n_splits=2,
        seed=seed_offset + 313,
    )
    early_train_local, early_stop_local = early_folds[0]
    early_train = train[early_train_local]
    early_stop = train[early_stop_local]
    epoch_records: list[dict[str, int]] = []
    selected_epochs: dict[int, int] = {}
    for seed in config.mlp_seeds:
        stop_preprocessor = RobustFoldPreprocessor(clip=config.preprocess_clip)
        X_early_train = stop_preprocessor.fit_transform(
            raw_matrix[early_train],
            y[early_train] == 1,
            feature_names=feature_names,
        )
        X_early_stop = stop_preprocessor.transform(raw_matrix[early_stop])
        stop_model = TinyTanhMLPRanker(
            seed=int(seed) + seed_offset,
            l2=l2,
            learning_rate=config.mlp_learning_rate,
            max_epochs=config.mlp_max_epochs,
            patience=config.mlp_patience,
        )
        stop_model.fit(
            X_early_train,
            y[early_train],
            sample_weight=_balanced_reference_weights(y[early_train]),
            validation=(X_early_stop, y[early_stop], _balanced_reference_weights(y[early_stop])),
        )
        selected_epochs[int(seed)] = int(stop_model.best_epoch_)
        epoch_records.append(
            {
                "seed": int(seed),
                "deterministic_fit_seed": int(seed) + seed_offset,
                "selected_epoch": int(stop_model.best_epoch_),
                "stopping_epochs_run": int(stop_model.n_epochs_),
            }
        )

    full_preprocessor = RobustFoldPreprocessor(clip=config.preprocess_clip)
    X_train = full_preprocessor.fit_transform(raw_matrix[train], y[train] == 1, feature_names=feature_names)
    X_test = full_preprocessor.transform(raw_matrix[test])
    seed_predictions: dict[int, np.ndarray] = {}
    for seed in config.mlp_seeds:
        model = TinyTanhMLPRanker(
            seed=int(seed) + seed_offset,
            l2=l2,
            learning_rate=config.mlp_learning_rate,
            max_epochs=config.mlp_max_epochs,
            patience=config.mlp_patience,
        )
        model.fit(
            X_train,
            y[train],
            sample_weight=_balanced_reference_weights(y[train]),
            fixed_epochs=selected_epochs[int(seed)],
        )
        seed_predictions[int(seed)] = model.predict_score(X_test)
    stacked = np.vstack([seed_predictions[int(seed)] for seed in config.mlp_seeds])
    return np.mean(stacked, axis=0), seed_predictions, epoch_records


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


__all__ = [
    "BaggedPURanker",
    "EvaluationConfig",
    "INCLUSIVE_LABEL_POLICY",
    "LabelPolicy",
    "PenalizedLogisticRanker",
    "ResolvedLabels",
    "RobustFoldPreprocessor",
    "SCHEMA_VERSION",
    "STRICT_LABEL_POLICY",
    "TinyTanhMLPRanker",
    "cross_seed_rank_stability",
    "fold_percentile_normalize",
    "grouped_paired_bootstrap_rank_delta",
    "grouped_spatial_folds",
    "nested_spatial_pu_evaluation",
    "positive_vs_unlabeled_rank_auc",
    "rank_metrics",
    "resolve_labels",
    "spatial_group_permutation_null",
    "spatial_group_score_permutation_null",
    "spu_auc",
    "summarize_folded_oof_ranking",
    "validate_frozen_outer_folds",
    "validate_raw_feature_names",
]
