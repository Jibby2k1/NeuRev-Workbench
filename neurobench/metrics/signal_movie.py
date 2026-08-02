"""Truth-aware metrics for native methods returning a reconstructed signal movie."""
from __future__ import annotations

from typing import Any

import numpy as np

from .source_separation import footprint_metrics, trace_fidelity_metrics


def signal_movie_metrics(
    true_traces: np.ndarray, true_footprints: np.ndarray,
    estimated_signal: np.ndarray, *, native_background: np.ndarray | None = None,
) -> dict[str, Any]:
    traces = np.asarray(true_traces, dtype=np.float64)
    footprints = np.asarray(true_footprints, dtype=np.float64)
    signal = np.asarray(estimated_signal, dtype=np.float64)
    truth = np.einsum("st,shw->thw", traces, footprints)
    if signal.shape != truth.shape or not np.isfinite(signal).all():
        raise ValueError("estimated signal must align with truth movie")
    rows = []
    for index, (trace, footprint) in enumerate(zip(traces, footprints)):
        estimated_trace = np.sum(signal*footprint[None], axis=(1,2))/max(float(np.sum(footprint*footprint)), np.finfo(float).eps)
        estimated_footprint = np.sum(signal*trace[:,None,None], axis=0)/max(float(np.sum(trace*trace)), np.finfo(float).eps)
        rows.append({"source": index, "trace": trace_fidelity_metrics(trace, estimated_trace),
                     "footprint": footprint_metrics(footprint, estimated_footprint)})
    centered = truth-truth.mean()
    nmse = float(np.sum((truth-signal)**2)/max(float(np.sum(centered*centered)), np.finfo(float).eps))
    leakage = None
    if native_background is not None:
        native = np.asarray(native_background, dtype=np.float64)
        left = signal.ravel()-signal.mean(); right = native.ravel()-native.mean()
        leakage = float(abs(left@right)/max(float(np.linalg.norm(left)*np.linalg.norm(right)), np.finfo(float).eps))
    return {"neural_reconstruction_nmse": nmse, "native_background_dependence": leakage,
            "mean_peak_retention": float(np.mean([row["trace"]["peak_retention"] for row in rows])),
            "mean_area_retention": float(np.mean([row["trace"]["area_retention"] for row in rows])),
            "mean_waveform_correlation": float(np.mean([row["trace"]["waveform_correlation"] for row in rows])),
            "maximum_absolute_peak_error_frames": int(max(abs(row["trace"]["peak_error_frames"]) for row in rows)),
            "mean_footprint_iou": float(np.mean([row["footprint"]["footprint_iou"] for row in rows])),
            "mean_centroid_error_px": float(np.mean([row["footprint"]["centroid_error_px"] for row in rows])),
            "sources": rows}
