"""Parity-oriented CUDA backend for bounded normalized-HSIC rotations."""
from __future__ import annotations

from typing import Any

import numpy as np

from .information_source_separation import LinearSeparationResult, _finalize, _matrix, _rotation, pca_whiten


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CUDA HSIC backend") from exc
    return torch


def _require_cuda(device: str):
    torch = _torch()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return torch


def _bandwidth_torch(values: Any, scale: float, torch: Any) -> Any:
    ordered = torch.sort(values.flatten()).values
    if ordered.numel() > 512:
        indices = torch.linspace(0, ordered.numel() - 1, 512, device=ordered.device).to(torch.long)
        ordered = ordered[indices]
    distances = torch.abs(ordered[:, None] - ordered[None, :])
    positive = distances[distances > 0]
    if positive.numel() == 0:
        return torch.as_tensor(scale, dtype=values.dtype, device=values.device)
    return torch.clamp(torch.quantile(positive, 0.5) * scale, min=1e-6)


def normalized_hsic_cuda(left: np.ndarray, right: np.ndarray, *, bandwidth_scale: float = 1.0, device: str = "cuda:0") -> float:
    """Evaluate the CPU-defined normalized RBF-HSIC statistic on CUDA."""
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape or len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("HSIC inputs must be aligned finite vectors of length >=8")
    if bandwidth_scale <= 0:
        raise ValueError("bandwidth_scale must be positive")
    torch = _require_cuda(device)
    tx = torch.as_tensor(x, dtype=torch.float64, device=device)
    ty = torch.as_tensor(y, dtype=torch.float64, device=device)
    hx = _bandwidth_torch(tx, float(bandwidth_scale), torch)
    hy = _bandwidth_torch(ty, float(bandwidth_scale), torch)
    kernel_x = torch.exp(-0.5 * ((tx[:, None] - tx[None, :]) / hx) ** 2)
    kernel_y = torch.exp(-0.5 * ((ty[:, None] - ty[None, :]) / hy) ** 2)
    kernel_x = kernel_x - kernel_x.mean(0, keepdim=True)
    kernel_x = kernel_x - kernel_x.mean(1, keepdim=True) + kernel_x.mean()
    kernel_y = kernel_y - kernel_y.mean(0, keepdim=True)
    kernel_y = kernel_y - kernel_y.mean(1, keepdim=True) + kernel_y.mean()
    numerator = torch.sum(kernel_x * kernel_y)
    denominator = torch.sqrt(torch.sum(kernel_x**2) * torch.sum(kernel_y**2))
    score = torch.clamp(numerator / torch.clamp(denominator, min=torch.finfo(torch.float64).eps), min=0)
    return float(score.item())


def fit_kernel_hsic_pairwise_rotation_cuda(
    observations: np.ndarray, *, rank: int | None = None, bandwidth_scale: float = 1.0,
    angle_step_degrees: float = 5.0, max_sweeps: int = 8, improvement_tolerance: float = 1e-4,
    max_fit_samples: int = 256, seed: int = 20260801, device: str = "cuda:0",
) -> LinearSeparationResult:
    """Run the bounded HSIC Jacobi search with dependence scores on CUDA."""
    values = _matrix(observations, "observations")
    if bandwidth_scale <= 0 or not 0 < angle_step_degrees <= 15 or max_sweeps < 1:
        raise ValueError("invalid HSIC CUDA settings")
    if improvement_tolerance < 0 or max_fit_samples < 32:
        raise ValueError("invalid tolerance or fit-sample bound")
    torch = _require_cuda(device)
    z, model = pca_whiten(values, rank=rank)
    rng = np.random.default_rng(int(seed))
    if z.shape[1] > max_fit_samples:
        indices = np.sort(rng.choice(z.shape[1], size=max_fit_samples, replace=False))
        fit_values = z[:, indices]
    else:
        indices = np.arange(z.shape[1])
        fit_values = z.copy()
    dimension = fit_values.shape[0]
    rotation = np.eye(dimension, dtype=np.float64)
    angles = np.deg2rad(np.arange(-45.0, 45.0 + angle_step_degrees * 0.5, angle_step_degrees))

    def dependence(a: np.ndarray, b: np.ndarray) -> float:
        return normalized_hsic_cuda(a, b, bandwidth_scale=float(bandwidth_scale), device=device)

    def total_pairwise(current: np.ndarray) -> float:
        return float(sum(dependence(current[left], current[right]) for left in range(dimension - 1) for right in range(left + 1, dimension)))

    history = [total_pairwise(fit_values)]
    converged = False
    accepted_updates = 0
    for sweep in range(1, max_sweeps + 1):
        start_objective = history[-1]
        for left in range(dimension - 1):
            for right in range(left + 1, dimension):
                pair = fit_values[[left, right]]
                candidates = []
                for angle in angles:
                    cosine, sine = np.cos(angle), np.sin(angle)
                    candidates.append(dependence(cosine * pair[0] + sine * pair[1], -sine * pair[0] + cosine * pair[1]))
                best_index = int(np.argmin(candidates))
                current_index = int(np.argmin(np.abs(angles)))
                if candidates[current_index] - candidates[best_index] <= improvement_tolerance:
                    continue
                jacobi = _rotation(dimension, left, right, float(angles[best_index]))
                fit_values = jacobi @ fit_values
                rotation = jacobi @ rotation
                accepted_updates += 1
        objective = total_pairwise(fit_values)
        history.append(objective)
        if start_objective - objective <= improvement_tolerance:
            converged = True
            break
    torch.cuda.synchronize(device)
    return _finalize(
        "kernel_hsic_pairwise_rotation", values, z, model, rotation, converged=converged,
        iterations=sweep, objective=history[-1], diagnostics={
            "bandwidth_scale": float(bandwidth_scale), "angle_step_degrees": float(angle_step_degrees),
            "max_fit_samples": int(max_fit_samples), "fit_sample_count": int(len(indices)), "seed": int(seed),
            "accepted_pair_updates": int(accepted_updates), "pairwise_dependence_history": history,
            "objective_direction": "lower_is_better", "qualification": "bounded_pairwise_rotation_reference",
            "execution_backend": "torch_cuda", "device": str(device), "dtype": "float64",
        },
    )
