"""Fractional quiet-fitted whitening operators for learned selection.

Covariance fitting is float64. Application is a scalar center-coordinate linear
readout, so spatial and temporal operators can be applied as bounded convolutions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class FractionalWhiteningFit:
    mean: np.ndarray
    sample_covariance: np.ndarray
    regularized_covariance: np.ndarray
    transform: np.ndarray
    eigenvalues_raw: np.ndarray
    eigenvalues_regularized: np.ndarray
    shrinkage: float
    exponent: float
    eigen_floor_ratio: float
    condition_number: float
    effective_rank_fraction: float
    sample_count: int
    resolved: bool
    stop_reason: str | None

def fit_fractional_whitening(samples: np.ndarray, *, shrinkage: float, exponent: float, eigen_floor_ratio: float, maximum_condition_number: float=1e6, minimum_effective_rank_fraction: float=.25) -> FractionalWhiteningFit:
    values=np.asarray(samples,dtype=np.float64)
    if values.ndim!=2 or len(values)<2 or not np.isfinite(values).all(): raise ValueError("samples must be finite [N,D] with N>=2")
    if not 0<=shrinkage<=1 or not 0<=exponent<=1 or not 0<eigen_floor_ratio<1: raise ValueError("invalid whitening parameter")
    mean=values.mean(axis=0); centered=values-mean; covariance=centered.T@centered/(len(values)-1); dimension=values.shape[1]
    target=float(np.trace(covariance))/dimension; regularized=(1-shrinkage)*covariance+shrinkage*target*np.eye(dimension)
    eigenvalues,vectors=np.linalg.eigh(regularized); floor=max(target*eigen_floor_ratio,np.finfo(float).eps); floored=np.maximum(eigenvalues,floor)
    condition=float(floored[-1]/floored[0]); probabilities=floored/floored.sum(); effective=float(np.exp(-np.sum(probabilities*np.log(np.maximum(probabilities,1e-300))))/dimension)
    transform=(vectors*(floored**(-exponent/2)))@vectors.T
    resolved=bool(np.isfinite(transform).all() and condition<=maximum_condition_number and effective>=minimum_effective_rank_fraction)
    reason=None if resolved else ("condition_number" if condition>maximum_condition_number else "effective_rank_fraction")
    return FractionalWhiteningFit(mean,covariance,regularized,transform,eigenvalues,floored,float(shrinkage),float(exponent),float(eigen_floor_ratio),condition,effective,len(values),resolved,reason)

def center_response_kernel(fit: FractionalWhiteningFit) -> tuple[np.ndarray,float]:
    center=fit.transform.shape[0]//2
    return fit.transform[center].copy(),float(fit.transform[center]@fit.mean)

def raw_preserving_blend(raw: np.ndarray, whitened: np.ndarray, beta: float) -> np.ndarray:
    if raw.shape!=whitened.shape or not 0<=beta<=1: raise ValueError("blend arrays must align and beta lie in [0,1]")
    if beta==0: return np.asarray(raw).copy()
    return ((1-beta)*np.asarray(raw,dtype=np.float32)+beta*np.asarray(whitened,dtype=np.float32)).astype(np.float32)

def apply_spatial_center_response(frames: np.ndarray, fit: FractionalWhiteningFit, width: int) -> np.ndarray:
    from scipy.ndimage import convolve
    values=np.asarray(frames,dtype=np.float32)
    if values.ndim!=3 or width%2!=1 or width<3 or fit.transform.shape!=(width*width,width*width): raise ValueError("spatial fit/width mismatch")
    kernel,offset=center_response_kernel(fit); return (convolve(values,kernel.reshape(1,width,width),mode="reflect")-offset).astype(np.float32)

def apply_temporal_center_response(frames: np.ndarray, fit: FractionalWhiteningFit, width: int) -> np.ndarray:
    from scipy.ndimage import convolve1d
    values=np.asarray(frames,dtype=np.float32)
    if values.ndim!=3 or width%2!=1 or width<3 or fit.transform.shape!=(width,width): raise ValueError("temporal fit/width mismatch")
    kernel,offset=center_response_kernel(fit); return (convolve1d(values,kernel,axis=0,mode="reflect")-offset).astype(np.float32)

def deterministic_spatial_samples(quiet: np.ndarray, width: int, *, maximum_samples: int=65536, seed: int=20260819) -> np.ndarray:
    values=np.asarray(quiet,dtype=np.float32)
    if values.ndim!=3 or width%2!=1 or width<3: raise ValueError("quiet must be TYX and width odd")
    radius=width//2; padded=np.pad(values,((0,0),(radius,radius),(radius,radius)),mode="reflect"); total=np.prod(values.shape)
    rng=np.random.default_rng(seed); flat=np.sort(rng.choice(total,size=min(maximum_samples,total),replace=False)); t,y,x=np.unravel_index(flat,values.shape)
    offsets=np.arange(-radius,radius+1); yy=y[:,None,None]+radius+offsets[None,:,None]; xx=x[:,None,None]+radius+offsets[None,None,:]
    return padded[t[:,None,None],yy,xx].reshape(len(flat),width*width).astype(np.float64)

def deterministic_temporal_samples(quiet: np.ndarray, width: int, *, maximum_samples: int=65536, seed: int=20260819) -> np.ndarray:
    values=np.asarray(quiet,dtype=np.float32)
    if values.ndim!=3 or width%2!=1 or width<3: raise ValueError("quiet must be TYX and width odd")
    radius=width//2; padded=np.pad(values,((radius,radius),(0,0),(0,0)),mode="reflect"); total=np.prod(values.shape)
    rng=np.random.default_rng(seed); flat=np.sort(rng.choice(total,size=min(maximum_samples,total),replace=False)); t,y,x=np.unravel_index(flat,values.shape)
    offsets=np.arange(-radius,radius+1); return padded[t[:,None]+radius+offsets[None,:],y[:,None],x[:,None]].astype(np.float64)

def quiet_standardize(values: np.ndarray, quiet_frames: int, epsilon: float=1e-6) -> tuple[np.ndarray,dict[str,float]]:
    array=np.asarray(values,dtype=np.float32); quiet=array[:quiet_frames]; center=float(np.median(quiet)); scale=max(float(1.4826*np.median(np.abs(quiet-center))),epsilon)
    return ((array-center)/scale).astype(np.float32),{"quiet_median":center,"quiet_mad_scale":scale}

def full_joint_allowed(*, dimension:int, nominal_samples:int, condition_number:float, effective_rank_fraction:float, spectra_stable:bool, memory_ready:bool, separable_promising:bool) -> tuple[bool,str]:
    checks={"dimension":dimension<=128,"samples":nominal_samples>=20*dimension,"condition":condition_number<=1e6,"rank":effective_rank_fraction>=.25,"spectra":spectra_stable,"memory":memory_ready,"separable":separable_promising}
    failed=[key for key,value in checks.items() if not value]
    return (not failed,"permitted" if not failed else "rejected_by_conditioning_gate:"+",".join(failed))

def diagnostics(fit: FractionalWhiteningFit) -> dict[str,Any]:
    return {"condition_number":fit.condition_number,"effective_rank_fraction":fit.effective_rank_fraction,"finite_output_fraction":float(np.mean(np.isfinite(fit.transform))),"resolved":fit.resolved,"stop_reason":fit.stop_reason,"sample_count":fit.sample_count}
