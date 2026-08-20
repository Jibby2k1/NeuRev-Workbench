"""Canonical two-frame ICA utilities.

These functions deliberately accept arrays rather than labels.  A caller must freeze the
returned model fingerprint before beginning external annotation evaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobench.algorithms.pairwise_separation import center_and_whiten_2d


@dataclass(frozen=True)
class TwoFrameFit:
    seed: int
    mean: np.ndarray
    whitening: np.ndarray
    rotation: np.ndarray
    effective_directions: np.ndarray
    activations: np.ndarray
    converged: bool

    def frozen_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "model": "two_frame_fastica_logcosh",
            "seed": self.seed, "mean": self.mean.tolist(),
            "whitening": self.whitening.tolist(), "rotation": self.rotation.tolist(),
            "effective_directions": self.effective_directions.tolist(),
            "converged": self.converged,
        }


def _samples(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("samples must be a finite [N,2] array with N >= 8")
    return x


def fit_two_frame_ica(values: np.ndarray, *, seed: int = 0, max_iter: int = 500,
                      tolerance: float = 1e-6) -> TwoFrameFit:
    """Fit ICA after the repository's maintained 2-D centering/whitening."""
    from sklearn.decomposition import FastICA

    x = _samples(values)
    z, white = center_and_whiten_2d(x.T)
    model = FastICA(whiten=False, fun="logcosh", random_state=seed,
                    max_iter=max_iter, tol=tolerance).fit(z.T)
    rotation = np.asarray(model.components_, dtype=np.float64)
    effective = rotation @ white.whitening
    activations = (rotation @ z).T
    return TwoFrameFit(seed, white.mean, white.whitening, rotation, effective,
                       activations, int(model.n_iter_) < max_iter)


def analytic_baselines(values: np.ndarray, *, epsilon: float = 1e-8,
                       random_seed: int = 0) -> dict[str, np.ndarray]:
    """Return B0--B5 responses on exactly the supplied samples."""
    x = _samples(values); a, b = x[:, 0], x[:, 1]
    difference = b - a
    median = float(np.median(difference))
    mad = max(float(np.median(np.abs(difference - median))) * 1.4826, epsilon)
    z, _ = center_and_whiten_2d(x.T)
    angle = float(np.random.default_rng(random_seed).uniform(0.0, 2.0 * np.pi))
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    return {
        "raw_t0": a, "raw_t1": b, "difference_signed": difference,
        "difference_absolute": np.abs(difference),
        "difference_standardized": (difference - median) / mad,
        "difference_energy_normalized": difference / np.sqrt(a*a + b*b + epsilon),
        "pca_whitened_0": z[0], "pca_whitened_1": z[1],
        "random_rotation_0": (rotation @ z)[0],
        "random_rotation_1": (rotation @ z)[1],
    }


def align_components(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Resolve sign/permutation ambiguity by maximum absolute activation correlation."""
    a = np.asarray(reference, dtype=np.float64); b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("reference and candidate must have the same [N,C] shape")
    corr = np.corrcoef(a.T, b.T)[:a.shape[1], a.shape[1]:]
    if a.shape[1] != 2:
        raise ValueError("canonical alignment currently requires two components")
    permutations = ((0, 1), (1, 0))
    order = max(permutations, key=lambda p: sum(abs(corr[i, p[i]]) for i in range(2)))
    aligned = b[:, order].copy()
    for index in range(2):
        if np.corrcoef(a[:, index], aligned[:, index])[0, 1] < 0:
            aligned[:, index] *= -1
    return aligned, order


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("distribution values must contain at least four finite samples")
    mean = float(x.mean()); centered = x - mean; std = float(x.std())
    standardized = centered / max(std, np.finfo(float).eps)
    q = np.quantile(x, [0.001, .01, .05, .5, .95, .99, .999])
    robust_scale = max(float(np.median(np.abs(x - np.median(x)))) * 1.4826,
                       np.finfo(float).eps)
    return {"count": len(x), "mean": mean, "std": std,
            "skewness": float(np.mean(standardized**3)),
            "excess_kurtosis": float(np.mean(standardized**4) - 3.0),
            "quantiles": dict(zip(("q001","q01","q05","q50","q95","q99","q999"), map(float, q))),
            "robust_tail_mass_5sigma": float(np.mean(np.abs(x-np.median(x)) > 5*robust_scale))}


def _top_jaccard(a: np.ndarray, b: np.ndarray, fraction: float) -> float:
    k = max(1, int(round(len(a) * fraction)))
    ia = set(np.argpartition(np.abs(a), -k)[-k:].tolist())
    ib = set(np.argpartition(np.abs(b), -k)[-k:].tolist())
    return len(ia & ib) / len(ia | ib)


def stability_summary(fits: list[TwoFrameFit], *, top_fraction: float = .01) -> dict[str, Any]:
    if len(fits) < 2:
        raise ValueError("at least two fits are required")
    reference = fits[0].activations; rows = []
    ref_dirs = fits[0].effective_directions
    for fit in fits[1:]:
        aligned, order = align_components(reference, fit.activations)
        directions = fit.effective_directions[list(order)].copy()
        for c in range(2):
            if np.corrcoef(reference[:, c], aligned[:, c])[0, 1] < 0:
                directions[c] *= -1
            rows.append({"seed": fit.seed, "component": c,
                         "activation_correlation": float(np.corrcoef(reference[:, c], aligned[:, c])[0, 1]),
                         "absolute_direction_cosine": float(abs(ref_dirs[c] @ directions[c]) /
                             (np.linalg.norm(ref_dirs[c])*np.linalg.norm(directions[c]))),
                         "top_jaccard": _top_jaccard(reference[:, c], aligned[:, c], top_fraction)})
    return {"reference_seed": fits[0].seed, "comparisons": rows,
            "mean_activation_correlation": float(np.mean([r["activation_correlation"] for r in rows])),
            "mean_absolute_direction_cosine": float(np.mean([r["absolute_direction_cosine"] for r in rows])),
            "mean_top_jaccard": float(np.mean([r["top_jaccard"] for r in rows]))}


def representation_fingerprint(fit: TwoFrameFit, selection: dict[str, Any]) -> str:
    """Hash the label-free model plus frozen selection contract."""
    payload = {"fit": fit.frozen_dict(), "selection": selection}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def temporal_block_shuffle(values: np.ndarray, *, block_size: int, seed: int) -> np.ndarray:
    """N1: pair first-frame blocks with independently permuted second-frame blocks."""
    x = _samples(values); n_blocks = len(x) // block_size
    if block_size < 1 or n_blocks < 2:
        raise ValueError("need at least two complete positive-sized blocks")
    used = x[:n_blocks*block_size].copy().reshape(n_blocks, block_size, 2)
    order = np.random.default_rng(seed).permutation(n_blocks)
    used[:, :, 1] = used[order, :, 1]
    return used.reshape(-1, 2)
