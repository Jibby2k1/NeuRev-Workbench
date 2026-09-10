import itertools

import numpy as np
import pytest
import torch

from neurobench.experiments.unsupervised_ica_eval.matrix_itl import (
    MatrixExponentialOrthogonal,
    cs_parzen_factorized,
    fit_matrix_tc_ica,
    matrix_renyi_entropy,
    matrix_total_correlation,
    normalize_gram,
    rbf_gram,
)


DTYPE = torch.float64


def test_t00_rbf_gram_contract_and_gradient():
    x = torch.tensor([[0.0, 1.0], [1.0, 3.0], [-2.0, 0.5]], dtype=DTYPE, requires_grad=True)
    gram = rbf_gram(x, torch.tensor([0.5, 2.0], dtype=DTYPE))
    assert torch.allclose(gram, gram.T)
    assert torch.allclose(torch.diagonal(gram), torch.ones(3, dtype=DTYPE))
    assert bool((gram >= 0).all())
    gram.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())


def test_t01_normalization_trace_psd_and_scale_invariance():
    x = torch.randn(12, 2, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    kernel = rbf_gram(x, 1.2)
    gram = normalize_gram(kernel)
    assert torch.allclose(torch.trace(gram), torch.tensor(1.0, dtype=DTYPE), atol=1e-12)
    assert float(torch.linalg.eigvalsh(gram).min()) >= -1e-12
    assert torch.allclose(gram, normalize_gram(7.3 * kernel), atol=1e-12)
    weights = torch.linspace(0.0, 1.0, 12, dtype=DTYPE)
    weighted = weights.sqrt()[:, None] * kernel * weights.sqrt()[None, :]
    weighted /= torch.trace(weighted)
    assert float(torch.linalg.eigvalsh(weighted).min()) >= -1e-12


def test_t02_alpha2_fast_path_matches_eigh_and_pairwise_identity():
    kernel = rbf_gram(torch.linspace(-2, 2, 9, dtype=DTYPE)[:, None], 0.7)
    gram = normalize_gram(kernel)
    fast = matrix_renyi_entropy(gram, 2.0, method="alpha2")
    spectral = matrix_renyi_entropy(gram, 2.0, method="eigh")
    explicit = -torch.log2(torch.sum(kernel * kernel) / kernel.shape[0] ** 2)
    assert torch.allclose(fast.total, spectral.total, rtol=1e-8, atol=1e-10)
    assert torch.allclose(fast.total, explicit, rtol=1e-8, atol=1e-10)


def test_t03_hadamard_product_is_anisotropic_joint_rbf():
    x = torch.randn(10, 3, generator=torch.Generator().manual_seed(2), dtype=DTYPE)
    bandwidths = torch.tensor([0.4, 1.1, 2.0], dtype=DTYPE)
    marginal_product = torch.stack(
        [rbf_gram(x[:, j:j + 1], bandwidths[j]) for j in range(3)]
    ).prod(dim=0)
    assert torch.allclose(marginal_product, rbf_gram(x, bandwidths), atol=1e-12)


def test_t04_matrix_tc_invariances_dependence_and_constant_handling():
    generator = torch.Generator().manual_seed(3)
    independent = torch.randn(500, 2, generator=generator, dtype=DTYPE)
    dependent = torch.column_stack((independent[:, 0], independent[:, 0]))
    base = matrix_total_correlation(independent, [1.0, 1.0]).total
    duplicate = matrix_total_correlation(dependent, [1.0, 1.0]).total
    sample_order = torch.randperm(len(independent), generator=generator)
    assert torch.allclose(base, matrix_total_correlation(independent[sample_order], [1.0, 1.0]).total, atol=1e-12)
    assert torch.allclose(base, matrix_total_correlation(independent[:, [1, 0]], [1.0, 1.0]).total, atol=1e-12)
    assert float(duplicate) > float(base)
    constant = matrix_total_correlation(torch.ones(8, 2, dtype=DTYPE), [1.0, 1.0]).total
    assert bool(torch.isfinite(constant))


def test_t05_cs_factorization_matches_direct_enumeration():
    y = torch.tensor([[-1.0, 0.2], [0.3, 1.1], [1.4, -0.7], [2.0, 0.4]], dtype=DTYPE)
    kernels = [rbf_gram(y[:, j:j + 1], 0.9 + j * 0.2) for j in range(2)]
    n, q = y.shape
    a = sum(torch.prod(torch.stack([kernels[j][r, s] for j in range(q)])) for r, s in itertools.product(range(n), repeat=2)) / n**2
    b = torch.prod(torch.stack([sum(kernels[j][r, s] for r, s in itertools.product(range(n), repeat=2)) / n**2 for j in range(q)]))
    c = sum(torch.prod(torch.stack([sum(kernels[j][r, s] for s in range(n)) for j in range(q)])) for r in range(n)) / n ** (q + 1)
    direct = torch.log(a) + torch.log(b) - 2 * torch.log(c)
    assert torch.allclose(cs_parzen_factorized(y, [0.9, 1.1]).total, direct, atol=1e-12)


def test_t06_matrix_tc_and_skew_parameter_gradients_match_finite_difference():
    y = torch.randn(9, 2, generator=torch.Generator().manual_seed(4), dtype=DTYPE, requires_grad=True)
    value = matrix_total_correlation(y, [0.8, 1.2]).total
    gradient = torch.autograd.grad(value, y)[0]
    delta = 1e-6
    with torch.no_grad():
        plus, minus = y.clone(), y.clone()
        plus[2, 1] += delta
        minus[2, 1] -= delta
    finite = (matrix_total_correlation(plus, [0.8, 1.2]).total - matrix_total_correlation(minus, [0.8, 1.2]).total) / (2 * delta)
    assert torch.allclose(gradient[2, 1], finite, rtol=2e-5, atol=2e-7)
    model = MatrixExponentialOrthogonal(2)
    rotated = y.detach() @ model().T
    matrix_total_correlation(rotated, [0.8, 1.2]).total.backward()
    assert model.theta.grad is not None and bool(torch.isfinite(model.theta.grad).all())


def test_t07_orthogonal_parameterization_and_smoke_fitter():
    model = MatrixExponentialOrthogonal(4)
    with torch.no_grad():
        model.theta.normal_(std=0.3)
    rotation = model()
    assert torch.allclose(rotation @ rotation.T, torch.eye(4, dtype=DTYPE), atol=1e-12)
    assert abs(abs(float(torch.linalg.det(rotation).detach())) - 1.0) < 1e-12
    generator = torch.Generator().manual_seed(5)
    sources = torch.column_stack((
        torch.randn(80, generator=generator, dtype=DTYPE),
        torch.rand(80, generator=generator, dtype=DTYPE) * 4 - 2,
    ))
    mix = torch.tensor([[1.0, 0.6], [0.25, 1.1]], dtype=DTYPE)
    observed = sources @ mix.T
    centered = observed - observed.mean(dim=0)
    values, vectors = torch.linalg.eigh(centered.T @ centered / len(centered))
    whitened = centered @ vectors @ torch.diag(values.rsqrt())
    first = fit_matrix_tc_ica(whitened, [1.0, 1.0], seed=9, steps=12)
    second = fit_matrix_tc_ica(whitened, [1.0, 1.0], seed=9, steps=12)
    assert first.orthogonality_error < 1e-12
    assert torch.equal(first.rotation, second.rotation)
    assert first.objective == second.objective
    distinct = [fit_matrix_tc_ica(whitened, [1.0, 1.0], seed=seed, steps=12) for seed in (9, 10, 11)]
    assert all(np.isfinite(fit.objective) and fit.orthogonality_error < 1e-12 for fit in distinct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_t00_cuda_matches_cpu():
    x = torch.randn(10, 2, generator=torch.Generator().manual_seed(7), dtype=DTYPE)
    assert torch.allclose(rbf_gram(x, 0.7), rbf_gram(x.cuda(), 0.7).cpu(), atol=1e-12)
