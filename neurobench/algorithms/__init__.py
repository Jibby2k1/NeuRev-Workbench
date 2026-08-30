"""Algorithmic building blocks for executable Neurobench pipelines."""

from neurobench.algorithms.cfar import gamma_cfar_mask, robust_local_cfar
from neurobench.algorithms.chunking import process_independent_frame_chunks
from neurobench.algorithms.motion import estimate_integer_shift, estimate_rigid_shifts, shift_frame_integer
from neurobench.algorithms.multilag_msica import (
    TemporalProjectionSiteDiagnostics,
    project_temporal_fit_at_sites,
)
from neurobench.algorithms.multiscale_local_normalization import causal_joint_msln
from neurobench.algorithms.msln_msica_cuda import causal_joint_msln_cuda

__all__ = [
    "estimate_integer_shift",
    "estimate_rigid_shifts",
    "gamma_cfar_mask",
    "process_independent_frame_chunks",
    "robust_local_cfar",
    "shift_frame_integer",
    "TemporalProjectionSiteDiagnostics",
    "project_temporal_fit_at_sites",
    "causal_joint_msln",
    "causal_joint_msln_cuda",
]
