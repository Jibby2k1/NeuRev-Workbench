"""Fixed, label-free routing formulas for calibrated context evidence."""
from __future__ import annotations

import numpy as np


def route_evidence(evidence: np.ndarray, context_ids: tuple[str, ...], *, mode: str, temperature: float = 1.0, complexity_penalty: float = 0.0, compact_minus_broad_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(evidence, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] != len(context_ids) or not np.isfinite(values).all():
        raise ValueError("evidence must be finite with context on axis zero")
    dominant = np.argmax(values, axis=0).astype(np.uint8)
    if mode == "none":
        return values.copy(), dominant
    if mode == "max":
        return np.max(values, axis=0), dominant
    compact = [i for i, name in enumerate(context_ids) if name.startswith("spatial_5_") or name.startswith("spatial_7_")]
    broad = [i for i, name in enumerate(context_ids) if name.startswith("spatial_15_") or name.startswith("temporal_31_")]
    if mode == "compact_agreement":
        if len(compact) < 2:
            raise ValueError("compact_agreement needs spatial 5 and 7 contexts")
        routed = np.sqrt(np.maximum(values[compact[0]], 0) * np.maximum(values[compact[1]], 0))
    elif mode == "compact_minus_broad":
        if not compact or not broad:
            raise ValueError("compact_minus_broad needs compact and broad contexts")
        routed = np.maximum(np.max(values[compact], axis=0) - float(compact_minus_broad_weight) * np.max(values[broad], axis=0), 0)
    elif mode == "softmax":
        if temperature <= 0:
            raise ValueError("softmax temperature must be positive")
        logits = (values - float(complexity_penalty)) / float(temperature)
        logits -= np.max(logits, axis=0, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=0, keepdims=True)
        routed = np.sum(weights * values, axis=0)
    else:
        raise ValueError("unknown routing mode")
    return np.asarray(routed, dtype=np.float32), dominant


def product_interaction(raw: np.ndarray, activity: np.ndarray, *, beta: float, kappa: float) -> np.ndarray:
    if not 0 <= beta <= 1 or kappa <= 0:
        raise ValueError("beta must be in [0,1] and kappa positive")
    amplitude = np.asarray(raw, dtype=np.float32)
    evidence = np.asarray(activity, dtype=np.float32)
    gate = beta + (1.0 - beta) * evidence * evidence / (kappa * kappa + evidence * evidence)
    return np.asarray(amplitude * gate, dtype=np.float32)


def bounded_residual_gate(
    values: np.ndarray, *, beta: float, kappa: float
) -> np.ndarray:
    """Return beta + (1-beta) z^2/(kappa^2+z^2)."""
    array = np.asarray(values, dtype=np.float32)
    floor = float(beta)
    threshold = float(kappa)
    if (
        not np.isfinite(array).all()
        or not 0.0 <= floor <= 1.0
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError(
            "finite values, beta in [0,1], and positive kappa are required"
        )
    square = np.square(array, dtype=np.float32)
    return (
        floor
        + (1.0 - floor) * square / (threshold * threshold + square)
    ).astype(np.float32)
