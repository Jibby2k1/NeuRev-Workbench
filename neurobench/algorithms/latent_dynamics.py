"""Stable scalar AR(1) state-space inference for frame-first signal arrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class QuietNoiseModel:
    center: np.ndarray
    scale: np.ndarray
    scale_floor: float
    difference_variance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StableAR1:
    gamma: float
    process_variance: float
    observation_variance: float
    initial_variance: float
    stability_margin: float
    fit_status: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StateSpaceResult:
    filter_mean: np.ndarray
    filter_variance: np.ndarray
    predicted_mean: np.ndarray
    predicted_variance: np.ndarray
    innovation: np.ndarray
    innovation_variance: np.ndarray
    log_likelihood: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SmootherResult:
    mean: np.ndarray
    variance: np.ndarray
    lag_covariance: np.ndarray | None
    diagnostics: dict[str, Any]


def _signals(values: np.ndarray, name: str = "observations") -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    squeezed = array.ndim == 1
    if squeezed:
        array = array[:, None]
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite [T,N] or [T] array")
    return array, squeezed


def estimate_quiet_noise(values: np.ndarray, *, floor_percentile: float = 10.0) -> QuietNoiseModel:
    observations, _ = _signals(values, "quiet observations")
    if len(observations) < 3 or not 0 <= floor_percentile <= 100:
        raise ValueError("Quiet noise estimation requires at least three frames and a valid percentile")
    center = np.median(observations, axis=0)
    mad = 1.4826 * np.median(np.abs(observations - center), axis=0)
    positive = mad[mad > 0]
    floor = max(float(np.percentile(positive, floor_percentile)) if positive.size else 1.0,
                np.finfo(np.float32).eps)
    scale = np.maximum(mad, floor)
    differences = np.diff(observations, axis=0)
    difference_center = np.median(differences, axis=0)
    difference_mad = 1.4826 * np.median(np.abs(differences - difference_center), axis=0)
    difference_variance = np.maximum(0.5 * difference_mad**2, np.finfo(np.float64).eps)
    return QuietNoiseModel(
        center=center.astype(np.float32), scale=scale.astype(np.float32), scale_floor=floor,
        difference_variance=difference_variance.astype(np.float32),
        diagnostics={"frames":len(observations),"signals":observations.shape[1],
                     "zero_raw_scale_fraction":float(np.mean(mad==0)),
                     "difference_variance_median":float(np.median(difference_variance))},
    )


def stable_ar1_from_decay(
    frame_period_ms: float,
    decay_time_ms: float,
    process_variance: float,
    observation_variance: float,
    *,
    initial_variance: float | None = None,
    stability_epsilon: float = 1e-3,
    diagnostics: dict[str, Any] | None = None,
) -> StableAR1:
    if frame_period_ms <= 0 or decay_time_ms <= 0 or not 0 < stability_epsilon < 1:
        raise ValueError("Frame period, decay time, and stability epsilon must be positive")
    gamma = float(np.exp(-frame_period_ms / decay_time_ms))
    if gamma > 1 - stability_epsilon:
        gamma = 1 - stability_epsilon
    model = StableAR1(
        gamma=gamma, process_variance=float(process_variance),
        observation_variance=float(observation_variance),
        initial_variance=float(observation_variance if initial_variance is None else initial_variance),
        stability_margin=1-gamma, fit_status="resolved",
        diagnostics={"frame_period_ms":float(frame_period_ms),"decay_time_ms":float(decay_time_ms),
                     "stability_epsilon":float(stability_epsilon), **(diagnostics or {})},
    )
    validate_stable_ar1(model, required_margin=stability_epsilon)
    return model


def validate_stable_ar1(model: StableAR1, *, required_margin: float | None = None) -> None:
    values=(model.gamma,model.process_variance,model.observation_variance,model.initial_variance,model.stability_margin)
    if not np.isfinite(values).all() or not 0 <= model.gamma < 1:
        raise ValueError("AR(1) pole must be finite and strictly stable")
    if min(model.process_variance,model.observation_variance,model.initial_variance) <= 0:
        raise ValueError("AR(1) variances must be strictly positive")
    if abs(model.stability_margin-(1-model.gamma)) > 1e-12:
        raise ValueError("Recorded stability margin does not match gamma")
    if required_margin is not None and model.stability_margin + 1e-15 < required_margin:
        raise ValueError("AR(1) stability margin is below the declared requirement")


def kalman_filter_ar1(
    observations: np.ndarray,
    model: StableAR1,
    *,
    initial_mean: np.ndarray | float = 0.0,
    variance_floor: float = 1e-12,
    output_dtype: np.dtype = np.float32,
) -> StateSpaceResult:
    x, squeezed = _signals(observations)
    validate_stable_ar1(model)
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    initial = np.asarray(initial_mean,dtype=np.float64)
    if initial.ndim == 0: initial=np.full(x.shape[1],float(initial))
    if initial.shape != (x.shape[1],) or not np.isfinite(initial).all():
        raise ValueError("initial_mean must be scalar or [N]")
    t,n=x.shape; fm=np.empty((t,n)); pm=np.empty((t,n)); innovation=np.empty((t,n))
    fv=np.empty(t); pv=np.empty(t); iv=np.empty(t); increments=np.empty(t)
    previous_mean=initial; previous_variance=model.initial_variance
    gain_min=float("inf"); gain_max=0.0
    for index in range(t):
        if index == 0:
            predicted_mean=previous_mean; predicted_variance=previous_variance
        else:
            predicted_mean=model.gamma*previous_mean
            predicted_variance=model.gamma**2*previous_variance+model.process_variance
        predicted_variance=max(float(predicted_variance),variance_floor)
        innovation_variance=max(predicted_variance+model.observation_variance,variance_floor)
        residual=x[index]-predicted_mean; gain=predicted_variance/innovation_variance
        filtered_mean=predicted_mean+gain*residual
        # Joseph form remains nonnegative under finite rounding.
        filtered_variance=(1-gain)**2*predicted_variance+gain**2*model.observation_variance
        if not np.isfinite(filtered_mean).all() or filtered_variance < -1e-13:
            raise FloatingPointError("Non-finite state or negative covariance")
        filtered_variance=max(float(filtered_variance),variance_floor)
        pm[index]=predicted_mean; pv[index]=predicted_variance; innovation[index]=residual; iv[index]=innovation_variance
        fm[index]=filtered_mean; fv[index]=filtered_variance
        increments[index]=-0.5*(n*np.log(2*np.pi*innovation_variance)+float(residual@residual)/innovation_variance)
        previous_mean=filtered_mean; previous_variance=filtered_variance
        gain_min=min(gain_min,gain); gain_max=max(gain_max,gain)
    def shaped(a):
        cast=a.astype(output_dtype)
        return cast[:,0] if squeezed and cast.ndim==2 else cast
    return StateSpaceResult(
        filter_mean=shaped(fm), filter_variance=fv.astype(output_dtype),
        predicted_mean=shaped(pm), predicted_variance=pv.astype(output_dtype),
        innovation=shaped(innovation), innovation_variance=iv.astype(output_dtype),
        log_likelihood=float(increments.sum()),
        diagnostics={"causal":True,"axes":"TN" if not squeezed else "T",
                     "gain_min":gain_min,"gain_max":gain_max,
                     "predictive_variance_min":float(pv.min()),"predictive_variance_max":float(pv.max()),
                     "posterior_variance_min":float(fv.min()),"posterior_variance_max":float(fv.max()),
                     "log_likelihood_increment_min":float(increments.min()),
                     "log_likelihood_increment_max":float(increments.max())},
    )


def rts_smoother_ar1(filtered: StateSpaceResult, model: StableAR1, *, output_dtype: np.dtype=np.float32) -> SmootherResult:
    fm,squeezed=_signals(filtered.filter_mean,"filter_mean"); pm,_=_signals(filtered.predicted_mean,"predicted_mean")
    fv=np.asarray(filtered.filter_variance,dtype=np.float64); pv=np.asarray(filtered.predicted_variance,dtype=np.float64)
    if len(fm)!=len(fv) or pv.shape!=fv.shape or np.any(fv<0) or np.any(pv<=0):
        raise ValueError("Filter means and positive scalar variances must align")
    sm=fm.copy(); sv=fv.copy(); lag=np.empty(max(0,len(fm)-1),dtype=np.float64); gains=[]
    for index in range(len(fm)-2,-1,-1):
        gain=fv[index]*model.gamma/max(pv[index+1],1e-12); gains.append(gain)
        sm[index]=fm[index]+gain*(sm[index+1]-pm[index+1])
        sv[index]=fv[index]+gain**2*(sv[index+1]-pv[index+1])
        if sv[index] < -1e-12 or not np.isfinite(sm[index]).all():
            raise FloatingPointError("RTS smoother produced invalid covariance/state")
        sv[index]=max(sv[index],0.0); lag[index]=gain*sv[index+1]
    result=sm.astype(output_dtype)
    if squeezed: result=result[:,0]
    return SmootherResult(result,sv.astype(output_dtype),lag.astype(output_dtype),
                          {"causal":False,"look_ahead":"full_sequence","backward_gain_min":float(min(gains,default=0)),
                           "backward_gain_max":float(max(gains,default=0))})


def fit_shared_ar1_grid(
    observations: np.ndarray,
    *,
    frame_period_ms: float,
    decay_time_ms_grid: Iterable[float],
    process_to_observation_grid: Iterable[float],
    observation_variance: float,
    stability_epsilon: float=1e-3,
) -> tuple[StableAR1,list[dict[str,Any]]]:
    x,_=_signals(observations)
    candidates=[]
    for tau in tuple(decay_time_ms_grid):
        for ratio in tuple(process_to_observation_grid):
            if tau<=0 or ratio<=0: raise ValueError("Grid values must be positive")
            model=stable_ar1_from_decay(frame_period_ms,tau,ratio*observation_variance,observation_variance,
                                        stability_epsilon=stability_epsilon)
            result=kalman_filter_ar1(x,model,output_dtype=np.float64)
            candidates.append({"decay_time_ms":float(tau),"process_to_observation":float(ratio),
                               "gamma":model.gamma,"process_variance":model.process_variance,
                               "observation_variance":model.observation_variance,"log_likelihood":result.log_likelihood})
    winner=max(candidates,key=lambda row:(row["log_likelihood"],-row["decay_time_ms"],-row["process_to_observation"]))
    selected=stable_ar1_from_decay(frame_period_ms,winner["decay_time_ms"],winner["process_variance"],
                                   winner["observation_variance"],stability_epsilon=stability_epsilon,
                                   diagnostics={"parameter_mode":"bounded_grid","candidate_count":len(candidates),
                                                "selected_log_likelihood":winner["log_likelihood"]})
    return selected,candidates


def state_difference(state: np.ndarray, lag: int=1) -> np.ndarray:
    values,squeezed=_signals(state,"state")
    if not 1<=lag<len(values): raise ValueError("lag must be positive and shorter than state")
    result=np.full_like(values,np.nan); result[lag:]=values[lag:]-values[:-lag]
    return result[:,0] if squeezed else result


def dynamic_drive(state: np.ndarray, gamma: float) -> np.ndarray:
    values,squeezed=_signals(state,"state")
    if not 0<=gamma<=1: raise ValueError("gamma must be in [0,1] for dynamic-drive features")
    result=np.full_like(values,np.nan); result[1:]=values[1:]-gamma*values[:-1]
    return result[:,0] if squeezed else result


def standardized_filter_innovation(result: StateSpaceResult) -> np.ndarray:
    innovation=np.asarray(result.innovation,dtype=np.float64); scale=np.sqrt(np.asarray(result.innovation_variance,dtype=np.float64))
    if innovation.ndim==1: return innovation/scale
    return innovation/scale[:,None]


def smoother_observation_residual(observations: np.ndarray, smoother: SmootherResult) -> np.ndarray:
    x=np.asarray(observations,dtype=np.float64); mean=np.asarray(smoother.mean,dtype=np.float64)
    if x.shape!=mean.shape or not np.isfinite(x).all() or not np.isfinite(mean).all():
        raise ValueError("Observations and smoother state must be aligned and finite")
    return (x-mean).astype(np.float32)
