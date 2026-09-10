"""Reference matrix information-theoretic ICA primitives.

This module is intentionally limited to the phase-1 PC-MITL-ICA slice.  It
does not implement trajectory or conditional-process objectives.  Public
arrays use the repository convention ``[N, q]`` (samples by components).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import Tensor, nn


@dataclass
class ObjectiveResult:
    """Scalar objective together with differentiable terms and diagnostics."""

    total: Tensor
    terms: dict[str, Tensor]
    diagnostics: dict[str, Any]


def _matrix(x: Tensor, name: str) -> Tensor:
    if not isinstance(x, Tensor) or x.ndim != 2 or not x.numel():
        raise ValueError(f"{name} must be a non-empty rank-2 tensor")
    if not torch.is_floating_point(x) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must be finite and floating point")
    return x


def rbf_gram(
    x: Tensor,
    bandwidth: float | Tensor,
    *,
    squared_distance_floor: float = 0.0,
) -> Tensor:
    """Return an RBF Gram matrix for ``x[N,d]`` without breaking autograd."""
    x = _matrix(x, "x")
    if squared_distance_floor < 0:
        raise ValueError("squared_distance_floor must be nonnegative")
    scale = torch.as_tensor(bandwidth, dtype=x.dtype, device=x.device)
    if scale.ndim > 1 or (scale.ndim == 1 and scale.numel() not in (1, x.shape[1])):
        raise ValueError("bandwidth must be scalar or have one value per dimension")
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("bandwidth must be finite and positive")
    scaled = x / scale
    squared = (
        (scaled * scaled).sum(dim=1, keepdim=True)
        + (scaled * scaled).sum(dim=1).unsqueeze(0)
        - 2.0 * scaled @ scaled.T
    ).clamp_min(squared_distance_floor)
    gram = torch.exp(-0.5 * squared)
    return 0.5 * (gram + gram.T)


def normalize_gram(kernel: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize a positive-kernel Gram matrix to unit trace."""
    kernel = _matrix(kernel, "kernel")
    if kernel.shape[0] != kernel.shape[1] or eps <= 0:
        raise ValueError("kernel must be square and eps must be positive")
    symmetric = 0.5 * (kernel + kernel.T)
    diagonal = torch.diagonal(symmetric)
    if bool((diagonal <= 0).any()):
        raise ValueError("kernel diagonal must be positive")
    normalized = symmetric / torch.sqrt(
        diagonal.clamp_min(eps)[:, None] * diagonal.clamp_min(eps)[None, :]
    )
    normalized = normalized / normalized.shape[0]
    return 0.5 * (normalized + normalized.T)


def _effective_rank(eigenvalues: Tensor, eps: float) -> float:
    probabilities = eigenvalues.detach().clamp_min(0)
    probabilities = probabilities / probabilities.sum().clamp_min(eps)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum()
    return float(torch.exp(entropy).cpu())


def matrix_renyi_entropy(
    gram: Tensor,
    alpha: float,
    *,
    eps: float = 1e-12,
    method: Literal["auto", "alpha2", "eigh"] = "auto",
) -> ObjectiveResult:
    """Compute matrix-based Renyi entropy for a unit-trace Gram matrix."""
    gram = _matrix(gram, "gram")
    if gram.shape[0] != gram.shape[1] or eps <= 0 or alpha <= 0 or alpha == 1:
        raise ValueError("gram must be square; alpha > 0 and alpha != 1")
    if method not in {"auto", "alpha2", "eigh"}:
        raise ValueError("method must be auto, alpha2, or eigh")
    if method == "alpha2" and alpha != 2:
        raise ValueError("alpha2 method requires alpha == 2")
    use_fast = alpha == 2 and method in {"auto", "alpha2"}
    symmetric = 0.5 * (gram + gram.T)
    trace_error = torch.abs(torch.trace(symmetric) - 1.0)
    if use_fast:
        power_sum = torch.sum(symmetric * symmetric).clamp_min(eps)
        total = -torch.log2(power_sum)
        diagnostics = {
            "method": "alpha2", "alpha": 2.0, "trace_error": float(trace_error.detach().cpu()),
            "clipped_eigenvalue_count": 0,
        }
    else:
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        raw_minimum = float(eigenvalues.detach().min().cpu())
        tolerance = 100 * torch.finfo(gram.dtype).eps * max(1, gram.shape[0])
        if raw_minimum < -tolerance:
            raise ValueError(f"Gram matrix is not PSD: minimum eigenvalue {raw_minimum:.3e}")
        clipped_count = int((eigenvalues < 0).sum().detach().cpu())
        eigenvalues = eigenvalues.clamp_min(0)
        eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(eps)
        power_sum = torch.sum(eigenvalues.pow(alpha)).clamp_min(eps)
        total = torch.log2(power_sum) / (1.0 - alpha)
        diagnostics = {
            "method": "eigh", "alpha": float(alpha), "trace_error": float(trace_error.detach().cpu()),
            "raw_minimum_eigenvalue": raw_minimum,
            "clipped_eigenvalue_count": clipped_count,
            "effective_rank": _effective_rank(eigenvalues, eps),
        }
    return ObjectiveResult(total=total, terms={"power_sum": power_sum}, diagnostics=diagnostics)


def matrix_total_correlation(
    components: Tensor,
    bandwidths: Sequence[float] | Tensor,
    alpha: float = 2.0,
    *,
    sample_weights: Tensor | None = None,
    eps: float = 1e-12,
) -> ObjectiveResult:
    """Return matrix total correlation of ``components[N,q]``."""
    components = _matrix(components, "components")
    n, q = components.shape
    scales = torch.as_tensor(bandwidths, dtype=components.dtype, device=components.device).flatten()
    if scales.numel() != q or bool((scales <= 0).any()) or not bool(torch.isfinite(scales).all()):
        raise ValueError("bandwidths must contain q finite positive values")
    if sample_weights is not None:
        weights = torch.as_tensor(sample_weights, dtype=components.dtype, device=components.device)
        if weights.shape != (n,) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
            raise ValueError("sample_weights must be nonnegative [N] with positive mass")
        weights = weights / weights.sum()
        weight_root = torch.sqrt(weights)
    else:
        weight_root = None
    grams: list[Tensor] = []
    marginal_results: list[ObjectiveResult] = []
    for index in range(q):
        gram = rbf_gram(components[:, index:index + 1], scales[index])
        if weight_root is None:
            gram = normalize_gram(gram, eps)
        else:
            gram = weight_root[:, None] * gram * weight_root[None, :]
            gram = gram / torch.trace(gram).clamp_min(eps)
        grams.append(gram)
        marginal_results.append(matrix_renyi_entropy(gram, alpha, eps=eps))
    joint = grams[0]
    for gram in grams[1:]:
        joint = joint * gram
    joint = joint / torch.trace(joint).clamp_min(eps)
    joint_result = matrix_renyi_entropy(joint, alpha, eps=eps)
    marginal_sum = torch.stack([result.total for result in marginal_results]).sum()
    total = marginal_sum - joint_result.total
    terms = {f"marginal_entropy_{i}": result.total for i, result in enumerate(marginal_results)}
    terms.update({"marginal_entropy_sum": marginal_sum, "joint_entropy": joint_result.total})
    diagnostics = {
        "sample_count": n, "component_count": q, "alpha": float(alpha),
        "bandwidths": [float(value) for value in scales.detach().cpu()],
        "marginal_diagnostics": [result.diagnostics for result in marginal_results],
        "joint_diagnostics": joint_result.diagnostics,
    }
    return ObjectiveResult(total=total, terms=terms, diagnostics=diagnostics)


def cs_parzen_factorized(
    components: Tensor, bandwidths: Sequence[float] | Tensor, *, eps: float = 1e-12
) -> ObjectiveResult:
    """Small exact factorized CS-Parzen reference for ``[N,q]`` arrays."""
    components = _matrix(components, "components")
    scales = torch.as_tensor(bandwidths, dtype=components.dtype, device=components.device).flatten()
    if scales.numel() != components.shape[1] or bool((scales <= 0).any()):
        raise ValueError("bandwidths must contain q positive values")
    kernels = [rbf_gram(components[:, j:j + 1], scales[j]) for j in range(components.shape[1])]
    joint_kernel = torch.stack(kernels).prod(dim=0)
    a = joint_kernel.mean()
    b = torch.stack([kernel.mean() for kernel in kernels]).prod()
    row_product = torch.stack([kernel.sum(dim=1) for kernel in kernels]).prod(dim=0)
    c = row_product.sum() / (components.shape[0] ** (components.shape[1] + 1))
    total = torch.log(a.clamp_min(eps)) + torch.log(b.clamp_min(eps)) - 2 * torch.log(c.clamp_min(eps))
    return ObjectiveResult(total=total, terms={"A": a, "B": b, "C": c}, diagnostics={})


class MatrixExponentialOrthogonal(nn.Module):
    """Differentiable orthogonal rotation ``exp(theta - theta.T)``."""

    def __init__(self, component_count: int, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        if component_count < 2:
            raise ValueError("component_count must be at least two")
        self.theta = nn.Parameter(torch.zeros((component_count, component_count), dtype=dtype))

    def forward(self) -> Tensor:
        return torch.matrix_exp(self.theta - self.theta.T)


@dataclass(frozen=True)
class MatrixICAFit:
    rotation: Tensor
    recovered: Tensor
    objective: float
    iterations: int
    orthogonality_error: float


def fit_matrix_tc_ica(
    whitened: Tensor,
    bandwidths: Sequence[float],
    *,
    seed: int = 0,
    steps: int = 200,
    learning_rate: float = 0.03,
) -> MatrixICAFit:
    """Fit a small deterministic orthogonal matrix-TC ICA smoke model."""
    whitened = _matrix(whitened, "whitened")
    if whitened.shape[1] not in (2, 4) or steps < 1 or learning_rate <= 0:
        raise ValueError("whitened must be [N,2] or [N,4] and optimizer settings valid")
    torch.manual_seed(seed)
    model = MatrixExponentialOrthogonal(whitened.shape[1], dtype=whitened.dtype).to(whitened.device)
    with torch.no_grad():
        model.theta.normal_(mean=0.0, std=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    value = float("nan")
    for _ in range(steps):
        optimizer.zero_grad()
        rotation = model()
        result = matrix_total_correlation(whitened @ rotation.T, bandwidths, alpha=2.0)
        result.total.backward()
        optimizer.step()
        value = float(result.total.detach().cpu())
    with torch.no_grad():
        rotation = model()
        recovered = whitened @ rotation.T
        identity = torch.eye(rotation.shape[0], dtype=rotation.dtype, device=rotation.device)
        orthogonality_error = float((torch.linalg.norm(rotation @ rotation.T - identity) / rotation.shape[0]).cpu())
    return MatrixICAFit(rotation.detach(), recovered.detach(), value, steps, orthogonality_error)


def fit_cs_parzen_orthogonal(
    whitened: Tensor,
    bandwidths: Sequence[float],
    *,
    seed: int = 0,
    steps: int = 200,
    learning_rate: float = 0.03,
) -> MatrixICAFit:
    """Fit the factorized CS-Parzen objective for two or four components.

    This differentiable reference is used for paired q-dimensional synthetic
    comparisons.  The maintained two-component angle-search implementation
    remains the production baseline and is not replaced by this function.
    """
    whitened = _matrix(whitened, "whitened")
    if whitened.shape[1] not in (2, 4) or steps < 1 or learning_rate <= 0:
        raise ValueError("whitened must be [N,2] or [N,4] and optimizer settings valid")
    torch.manual_seed(seed)
    model = MatrixExponentialOrthogonal(whitened.shape[1], dtype=whitened.dtype).to(whitened.device)
    with torch.no_grad():
        model.theta.normal_(mean=0.0, std=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    value = float("nan")
    for _ in range(steps):
        optimizer.zero_grad()
        rotation = model()
        result = cs_parzen_factorized(whitened @ rotation.T, bandwidths)
        result.total.backward()
        optimizer.step()
        value = float(result.total.detach().cpu())
    with torch.no_grad():
        rotation = model()
        recovered = whitened @ rotation.T
        identity = torch.eye(rotation.shape[0], dtype=rotation.dtype, device=rotation.device)
        orthogonality_error = float((torch.linalg.norm(rotation @ rotation.T - identity) / rotation.shape[0]).cpu())
    return MatrixICAFit(rotation.detach(), recovered.detach(), value, steps, orthogonality_error)
