"""Operator construction and synthetic S1A validation."""
from __future__ import annotations
from typing import Any
import numpy as np
from neurobench.algorithms.local_whitening import fit_fractional_whitening, raw_preserving_blend

def synthetic_validation(seed:int=20260820)->dict[str,Any]:
    rng=np.random.default_rng(seed); covariance=np.asarray([[1,.8,.3],[.8,1.2,.25],[.3,.25,.7]])
    train=rng.multivariate_normal(np.zeros(3),covariance,size=12000); heldout=rng.multivariate_normal(np.zeros(3),covariance,size=12000)
    centered=fit_fractional_whitening(train,shrinkage=0,exponent=0,eigen_floor_ratio=1e-6)
    full=fit_fractional_whitening(train,shrinkage=.02,exponent=1,eigen_floor_ratio=1e-6)
    ridge=fit_fractional_whitening(train,shrinkage=.2,exponent=1,eigen_floor_ratio=1e-6)
    output=(heldout-full.mean)@full.transform.T; output_cov=np.cov(output.T)
    raw=rng.normal(size=(8,9)).astype(np.float32); other=rng.normal(size=raw.shape).astype(np.float32)
    singular=np.repeat(train[:,:1],3,axis=1); singular_fit=fit_fractional_whitening(singular,shrinkage=.1,exponent=1,eigen_floor_ratio=1e-4)
    checks={"gamma_zero_centering":bool(np.allclose(centered.transform,np.eye(3),atol=1e-12)),"gamma_one_whitens":bool(np.linalg.norm(output_cov-np.eye(3),ord="fro")<.08),"shrinkage_improves_condition":bool(ridge.condition_number<fit_fractional_whitening(train,shrinkage=0,exponent=1,eigen_floor_ratio=1e-6).condition_number),"fractional_finite":bool(all(np.isfinite(fit_fractional_whitening(train,shrinkage=.02,exponent=g,eigen_floor_ratio=1e-6).transform).all() for g in np.linspace(0,1,9))),"raw_blend_exact":bool(np.array_equal(raw_preserving_blend(raw,other,0),raw)),"rank_deficient_safe":bool(singular_fit.resolved and np.isfinite(singular_fit.transform).all())}
    return {"seed":seed,"checks":checks,"passed":all(checks.values()),"full_condition_number":full.condition_number,"full_effective_rank_fraction":full.effective_rank_fraction,"heldout_covariance_identity_error":float(np.linalg.norm(output_cov-np.eye(3),ord="fro"))}
