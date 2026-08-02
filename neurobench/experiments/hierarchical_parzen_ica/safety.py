"""Safety mathematics for recursive Stage-1 background reconstruction."""
from __future__ import annotations

from typing import Any

import numpy as np

from neurobench.algorithms.hierarchical_parzen_ica import (
    WhiteningResult,
    track_demixing_components,
)


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 2 by 2 matrix")
    return result


def stage1_feedback_diagnostics(
    whitening: WhiteningResult,
    demixing: np.ndarray,
    background_component: int,
    *,
    maximum_previous_background_coefficient: float = 1.2,
    maximum_current_observation_coefficient: float = 0.1,
    maximum_reconstruction_operator_norm: float = 2.0,
) -> dict[str, Any]:
    """Describe and gate the affine recursive Stage-1 background operator."""
    weights = _matrix(demixing, "demixing")
    if background_component not in {0, 1}:
        raise ValueError("background_component must be zero or one")
    bounds = np.asarray(
        [
            maximum_previous_background_coefficient,
            maximum_current_observation_coefficient,
            maximum_reconstruction_operator_norm,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(bounds).all() or np.any(bounds <= 0):
        raise ValueError("feedback safety bounds must be finite and positive")
    selector = np.zeros((2, 2), dtype=np.float64)
    selector[background_component, background_component] = 1.0
    operator = (
        whitening.dewhitening
        @ np.linalg.pinv(weights)
        @ selector
        @ weights
        @ whitening.whitening
    )
    offset = whitening.mean - operator @ whitening.mean
    previous = float(operator[1, 0])
    current = float(operator[1, 1])
    operator_norm = float(np.linalg.norm(operator, ord=2))
    reasons: list[str] = []
    if abs(previous) > maximum_previous_background_coefficient:
        reasons.append("previous_background_coefficient")
    if abs(current) > maximum_current_observation_coefficient:
        reasons.append("current_observation_coefficient")
    if operator_norm > maximum_reconstruction_operator_norm:
        reasons.append("reconstruction_operator_norm")
    return {
        "previous_background_coefficient": previous,
        "current_observation_coefficient": current,
        "offset": float(offset[1]),
        "reconstruction_operator_norm": operator_norm,
        "operator": operator.astype(float).tolist(),
        "safe": not reasons,
        "rejection_reasons": reasons,
        "bounds": {
            "maximum_previous_background_coefficient": float(
                maximum_previous_background_coefficient
            ),
            "maximum_current_observation_coefficient": float(
                maximum_current_observation_coefficient
            ),
            "maximum_reconstruction_operator_norm": float(
                maximum_reconstruction_operator_norm
            ),
        },
    }


def anchor_demixing_to_reference(
    reference_demixing: np.ndarray,
    learned_demixing: np.ndarray,
    whitening: WhiteningResult,
    *,
    background_component: int = 0,
    maximum_learned_fraction: float = 0.1,
    minimum_learned_fraction: float = 0.0015625,
    maximum_previous_background_coefficient: float = 1.2,
    maximum_current_observation_coefficient: float = 0.1,
    maximum_reconstruction_operator_norm: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the largest safe learned interpolation around a stable reference."""
    reference = _matrix(reference_demixing, "reference_demixing")
    learned = _matrix(learned_demixing, "learned_demixing")
    maximum_fraction = float(maximum_learned_fraction)
    minimum_fraction = float(minimum_learned_fraction)
    if not (
        np.isfinite([maximum_fraction, minimum_fraction]).all()
        and 0 < minimum_fraction <= maximum_fraction <= 1
    ):
        raise ValueError("invalid learned-fraction bounds")
    aligned, assignment, signs, tracking = track_demixing_components(
        reference, learned
    )
    safety_kwargs = {
        "maximum_previous_background_coefficient":
            maximum_previous_background_coefficient,
        "maximum_current_observation_coefficient":
            maximum_current_observation_coefficient,
        "maximum_reconstruction_operator_norm":
            maximum_reconstruction_operator_norm,
    }
    raw_feedback = stage1_feedback_diagnostics(
        whitening, aligned, background_component, **safety_kwargs
    )
    fractions = []
    fraction = maximum_fraction
    while fraction >= minimum_fraction * (1.0 - 1e-12):
        fractions.append(float(fraction))
        fraction *= 0.5
    fractions.append(0.0)
    attempts = []
    accepted = reference.copy()
    accepted_fraction = 0.0
    accepted_feedback = stage1_feedback_diagnostics(
        whitening, reference, background_component, **safety_kwargs
    )
    for candidate_fraction in fractions:
        candidate = (
            (1.0 - candidate_fraction) * reference
            + candidate_fraction * aligned
        )
        feedback = stage1_feedback_diagnostics(
            whitening, candidate, background_component, **safety_kwargs
        )
        attempts.append(
            {
                "learned_fraction": candidate_fraction,
                "safe": feedback["safe"],
                "previous_background_coefficient":
                    feedback["previous_background_coefficient"],
                "current_observation_coefficient":
                    feedback["current_observation_coefficient"],
                "reconstruction_operator_norm":
                    feedback["reconstruction_operator_norm"],
                "rejection_reasons": feedback["rejection_reasons"],
            }
        )
        if feedback["safe"]:
            accepted = candidate
            accepted_fraction = candidate_fraction
            accepted_feedback = feedback
            break
    return accepted, {
        "tracking": {
            **tracking,
            "assignment": list(assignment),
            "signs": list(signs),
        },
        "raw_feedback": raw_feedback,
        "accepted_feedback": accepted_feedback,
        "maximum_learned_fraction": maximum_fraction,
        "minimum_learned_fraction": minimum_fraction,
        "accepted_learned_fraction": accepted_fraction,
        "reference_fallback": accepted_fraction == 0.0,
        "attempts": attempts,
    }
