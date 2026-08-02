"""Truth-aware, scale-invariant component-product reconstruction metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .source_separation import footprint_metrics, trace_fidelity_metrics


def component_product_metrics(
    true_traces: np.ndarray, true_footprints: np.ndarray,
    estimated_sources: np.ndarray, estimated_spatial_maps: np.ndarray,
    *, background: np.ndarray | None = None,
    structured_artifact: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate reconstructed component movies without choosing an ICA scale."""
    traces = np.asarray(true_traces, dtype=np.float64)
    footprints = np.asarray(true_footprints, dtype=np.float64)
    sources = np.asarray(estimated_sources, dtype=np.float64)
    maps = np.asarray(estimated_spatial_maps, dtype=np.float64)
    if traces.ndim != 2 or footprints.ndim != 3 or sources.ndim != 2 or maps.ndim != 2:
        raise ValueError("component product inputs have invalid dimensions")
    if traces.shape[0] != footprints.shape[0] or traces.shape[1] != sources.shape[1]:
        raise ValueError("truth/source axes do not align")
    if maps.shape != (footprints.shape[1]*footprints.shape[2], sources.shape[0]):
        raise ValueError("spatial maps do not align with footprint geometry")
    centered_truth = traces-traces.mean(axis=1, keepdims=True)
    centered_sources = sources-sources.mean(axis=1, keepdims=True)
    correlation = centered_truth@centered_sources.T/np.maximum(
        np.linalg.norm(centered_truth, axis=1)[:, None]*np.linalg.norm(centered_sources, axis=1)[None],
        np.finfo(float).eps,
    )
    truth_indices, estimate_indices = linear_sum_assignment(-np.abs(correlation))
    rows = []
    reconstructed = np.zeros((traces.shape[1],)+footprints.shape[1:], dtype=np.float64)
    for truth_index, estimate_index in zip(truth_indices, estimate_indices):
        component = sources[estimate_index, :, None, None]*maps[:, estimate_index].reshape(footprints.shape[1:])[None]
        reconstructed += component
        footprint = footprints[truth_index]
        estimated_trace = np.sum(component*footprint[None], axis=(1, 2))/max(float(np.sum(footprint*footprint)), np.finfo(float).eps)
        trace = traces[truth_index]
        estimated_footprint = np.sum(component*trace[:, None, None], axis=0)/max(float(np.sum(trace*trace)), np.finfo(float).eps)
        rows.append({"true_source": int(truth_index), "estimated_source": int(estimate_index),
                     "absolute_temporal_correlation": float(abs(correlation[truth_index, estimate_index])),
                     "trace": trace_fidelity_metrics(trace, estimated_trace),
                     "footprint": footprint_metrics(footprint, estimated_footprint)})
    truth_movie = np.einsum("st,shw->thw", traces, footprints)
    neural_nmse = float(np.sum((truth_movie-reconstructed)**2)/max(float(np.sum((truth_movie-truth_movie.mean())**2)), np.finfo(float).eps))
    def dependence(other: np.ndarray | None) -> float | None:
        if other is None:
            return None
        left = reconstructed.ravel()-reconstructed.mean()
        right = np.asarray(other, dtype=np.float64).ravel()
        right -= right.mean()
        return float(abs(left@right)/max(float(np.linalg.norm(left)*np.linalg.norm(right)), np.finfo(float).eps))
    return {"matches": rows, "neural_reconstruction_nmse": neural_nmse,
            "mean_peak_retention": float(np.mean([row["trace"]["peak_retention"] for row in rows])),
            "mean_area_retention": float(np.mean([row["trace"]["area_retention"] for row in rows])),
            "mean_waveform_correlation": float(np.mean([row["trace"]["waveform_correlation"] for row in rows])),
            "maximum_absolute_peak_error_frames": int(max(abs(row["trace"]["peak_error_frames"]) for row in rows)),
            "mean_footprint_iou": float(np.mean([row["footprint"]["footprint_iou"] for row in rows])),
            "mean_centroid_error_px": float(np.mean([row["footprint"]["centroid_error_px"] for row in rows])),
            "background_dependence": dependence(background),
            "structured_artifact_dependence": dependence(structured_artifact)}
