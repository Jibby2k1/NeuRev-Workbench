"""Rank-matched label-safe controls for frozen real-data ICA finalists."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from neurobench.algorithms.information_source_separation import pca_whiten

from .real_runner import _event_standardized_scores, _label_metrics


def _orthogonal_rotation(rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.standard_normal((rank, rank)))
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs[None, :]


def evaluate_rank_matched_controls(
    fit_observations: Any, candidate_values: np.ndarray,
    candidate_groups: list[dict[str, Any]], proposals: list[dict[str, Any]],
    labels: list[dict[str, Any]], specification: dict[str, Any],
    *, quiet_count: int, budgets: tuple[int, ...], match_radius_px: int,
) -> dict[str, Any]:
    """Evaluate PCA and a deterministic random rotation before label access."""
    rank = int(specification["rank"])
    whitened, whitening = pca_whiten(fit_observations.values, rank=rank)
    rotations = {
        "rank_matched_pca": np.eye(rank),
        "rank_matched_random_rotation": _orthogonal_rotation(
            rank, int(specification["seed"]) + 7919
        ),
    }
    rows = []
    for control_id, rotation in rotations.items():
        sources = rotation @ whitened
        demixing = rotation @ whitening.whitening
        scores, calibration = _event_standardized_scores(
            sources, fit_observations.times, demixing, whitening.mean,
            candidate_values, candidate_groups, len(proposals), quiet_count,
        )
        digest = hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()
        # Sparse labels are intentionally consumed only after the score digest.
        metrics = _label_metrics(proposals, scores, labels, budgets, match_radius_px)
        rows.append({
            "control_id": control_id, "rank": rank,
            "candidate_score_sha256_before_label_metrics": digest,
            "label_metrics": metrics, "quiet_calibration": calibration,
            "condition_number": whitening.condition_number,
            "explained_fraction": whitening.explained_fraction,
            "random_rotation_seed": (None if control_id == "rank_matched_pca"
                                     else int(specification["seed"]) + 7919),
            "unmatched_candidates": "unknown_not_negative",
        })
    return {
        "fit_id": specification["fit_id"], "family": specification["family"],
        "whitening_geometry": specification["whitening_geometry"],
        "controls": rows, "label_safe_score_before_metrics": True,
    }
