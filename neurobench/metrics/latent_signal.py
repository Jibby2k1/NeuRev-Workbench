"""Ground-truth and calibration metrics for latent fluorescence estimates."""
from __future__ import annotations

from typing import Any
import numpy as np


def latent_reconstruction_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str,Any]:
    y=np.asarray(truth,dtype=np.float64); yhat=np.asarray(estimate,dtype=np.float64)
    if y.shape!=yhat.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("Truth and estimate must be aligned finite arrays")
    mse=float(np.mean((yhat-y)**2)); denom=max(float(np.mean(y**2)),1e-12)
    corr=float(np.corrcoef(y.ravel(),yhat.ravel())[0,1]) if np.std(y) and np.std(yhat) else float(y.shape==yhat.shape and np.allclose(y,yhat))
    truth_profile = np.max(y.reshape(len(y), -1), axis=1)
    estimate_profile = np.max(yhat.reshape(len(yhat), -1), axis=1)
    if np.ptp(truth_profile) <= 1e-12:
        peak = estimated_peak = 0
    else:
        peak = int(np.argmax(truth_profile)); estimated_peak = int(np.argmax(estimate_profile))
    truth_amplitude = float(np.max(y))
    amplitude_scale = abs(truth_amplitude) if abs(truth_amplitude) > 1e-12 else 1.0
    return {"nmse":mse/denom,"correlation":corr,
            "amplitude_bias":float(np.max(yhat)-truth_amplitude)/amplitude_scale,
            "peak_time_error_frames":estimated_peak-peak,"truth_peak_frame":peak,
            "estimated_peak_frame":estimated_peak}


def innovation_calibration(z: np.ndarray) -> dict[str,float]:
    values=np.asarray(z,dtype=np.float64)
    if not values.size or not np.isfinite(values).all(): raise ValueError("Innovations must be finite")
    return {"mean":float(values.mean()),"variance":float(values.var()),
            "tail_rate_abs_gt_3":float(np.mean(np.abs(values)>3))}


def interval_coverage(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray, z: float=1.96) -> float:
    y=np.asarray(truth); m=np.asarray(mean); v=np.asarray(variance)
    if y.shape!=m.shape or np.any(v<0): raise ValueError("Truth/mean must align and variance be nonnegative")
    scale=np.sqrt(v) if v.shape==y.shape else np.sqrt(v).reshape((-1,)+(1,)*(y.ndim-1))
    return float(np.mean(np.abs(y-m)<=z*scale))


def event_to_quiet_separation(feature: np.ndarray, quiet_slice: slice, event_slice: slice) -> float:
    values=np.asarray(feature,dtype=np.float64); quiet=values[quiet_slice]; event=values[event_slice]
    if not quiet.size or not event.size: raise ValueError("Quiet and event slices must be nonempty")
    return float((np.mean(event)-np.mean(quiet))/max(np.std(quiet),1e-12))
