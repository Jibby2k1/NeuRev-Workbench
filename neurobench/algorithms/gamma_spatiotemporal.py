"""Literal causal x/y/past-time extension of the direct Gamma two-stencil score.

This module performs no conditioning, threshold calibration, NMS, or learning.
The returned variance is the reference-weighted variability of the input, not
the uncertainty of A-M. Positive temporal scales share the approved anisotropic
distance q=sqrt(r**2+(R*lag*dt/T)**2). The per-voxel Gamma exponent is unchanged
between dimensions; dimension-dependent shell mass is recorded explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class GammaSTSpec:
    radius_px: float = 7.5
    time_scale_ms: float = 0.0
    target_sigma_px: float = 1.0
    reference_n: float = 9.0
    dt_ms: float = 20.0
    construction: str = "joint"

    def __post_init__(self) -> None:
        for name in ("radius_px", "target_sigma_px", "dt_ms"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.time_scale_ms)) or self.time_scale_ms < 0:
            raise ValueError("time_scale_ms must be finite and nonnegative")
        if not math.isfinite(float(self.reference_n)) or self.reference_n <= 1:
            raise ValueError("reference_n must be finite and greater than one")
        if self.construction not in ("joint", "separable"):
            raise ValueError("construction must be joint or separable")
        if self.construction == "separable" and self.time_scale_ms <= 0:
            raise ValueError("The separable control requires positive temporal scale")

    @property
    def spec_id(self) -> str:
        return f"R{self.radius_px:g}_T{self.time_scale_ms:g}_S{self.target_sigma_px:g}_n{self.reference_n:g}_{self.construction}"


def spec_grid() -> tuple[GammaSTSpec, ...]:
    """Exactly the approved 12+1+8 sensitivity configurations, without outcomes."""
    main = tuple(GammaSTSpec(radius_px=radius, time_scale_ms=time)
                 for radius in (5.0, 7.5, 10.0) for time in (0.0, 20.0, 60.0, 200.0))
    separable = (GammaSTSpec(time_scale_ms=60.0, construction="separable"),)
    target = tuple(GammaSTSpec(time_scale_ms=time, target_sigma_px=sigma)
                   for sigma in (2.0, 4.0) for time in (0.0, 60.0))
    shape = tuple(GammaSTSpec(time_scale_ms=time, reference_n=n)
                  for n in (3.0, 15.0) for time in (0.0, 60.0))
    return main + separable + target + shape


@dataclass(frozen=True)
class GammaSTKernels:
    spec: GammaSTSpec
    target: Any
    reference: Any
    metadata: dict[str, Any]


@lru_cache(maxsize=16)
def _target_mu(sigma: float) -> tuple[float, int]:
    from .gamma_two_stencil import match_centered_gamma_mu
    # Sigma=1 reproduces the maintained 31-pixel matching convention exactly.
    # Wider targets need a larger calibration square; do not force sigma=4
    # through a cropped 31-pixel stencil. This is a data-independent 2D match.
    half = max(15, math.ceil(8 * sigma))
    width = 2 * half + 1
    return match_centered_gamma_mu(width, gaussian_sigma_px=sigma), width


def _shape_bounds(spec: GammaSTSpec, radius: float) -> tuple[int, int]:
    half = max(1, math.ceil(radius))
    lag = 0 if spec.time_scale_ms == 0 else math.ceil(radius * spec.time_scale_ms / (spec.radius_px * spec.dt_ms))
    return half, lag


def _weights(spec: GammaSTSpec, half: int, max_lag: int, target_mu: float, *, totals_only: bool = False):
    """Evaluate one lag plane at a time, including for large convergence boxes."""
    import numpy as np
    coordinate = np.arange(-half, half+1, dtype=np.float64)
    radius2 = coordinate[:, None]**2 + coordinate[None, :]**2
    shape = (max_lag+1, len(coordinate), len(coordinate))
    target = None if totals_only else np.empty(shape, dtype=np.float64)
    reference = None if totals_only else np.empty(shape, dtype=np.float64)
    target_sums, reference_sums = [], []
    reference_mu = (spec.reference_n-1) / spec.radius_px
    for lag in range(max_lag+1):
        lag_distance = 0.0 if spec.time_scale_ms == 0 else spec.radius_px * lag * spec.dt_ms / spec.time_scale_ms
        q = np.sqrt(radius2 + lag_distance**2)
        tw = np.exp(-target_mu*q)
        rw = np.zeros(q.shape, dtype=np.float64)
        positive = q > 0
        rw[positive] = np.exp((spec.reference_n-1)*np.log(q[positive]) - reference_mu*q[positive])
        target_sums.append(float(tw.sum(dtype=np.float64)))
        reference_sums.append(float(rw.sum(dtype=np.float64)))
        if not totals_only:
            target[lag] = tw; reference[lag] = rw
    totals = (math.fsum(target_sums), math.fsum(reference_sums))
    return totals if totals_only else (target, reference, totals)


def _quantile_radius(spec: GammaSTSpec, target_mu: float, probability: float) -> float:
    from scipy.special import gammaincinv
    dimension = 2 if spec.time_scale_ms == 0 else 3
    # Per-voxel r^(n-1) times the d-dimensional shell r^(d-1) gives
    # Gamma shape n+d-1. The order-one target has shell shape d.
    return max(float(gammaincinv(dimension, probability))/target_mu,
               float(gammaincinv(spec.reference_n+dimension-1, probability))/((spec.reference_n-1)/spec.radius_px))


def _moments(kernel, spec: GammaSTSpec) -> dict[str, Any]:
    import numpy as np
    lag = np.arange(kernel.shape[0], dtype=np.float64)
    y = np.arange(kernel.shape[1], dtype=np.float64) - kernel.shape[1]//2
    x = np.arange(kernel.shape[2], dtype=np.float64) - kernel.shape[2]//2
    temporal = kernel.sum(axis=(1, 2)); spatial = kernel.sum(axis=0)
    xm, ym = spatial.sum(axis=0), spatial.sum(axis=1)
    lag_mean = float(temporal @ lag)
    def lag_quantile(probability):
        return int(min(len(lag)-1, np.searchsorted(np.cumsum(temporal), probability)))
    radial_mass = []
    r = np.sqrt(y[:, None]**2 + x[None, :]**2)
    bins = np.floor(r).astype(np.int64)
    radial_mass = np.bincount(bins.ravel(), weights=spatial.ravel()).tolist()
    return dict(normalized_mass=float(kernel.sum(dtype=np.float64)),
        spatial_sigma_x_px=math.sqrt(float(xm @ (x*x))),
        spatial_sigma_y_px=math.sqrt(float(ym @ (y*y))),
        spatial_radial_rms_px=math.sqrt(float(xm @ (x*x) + ym @ (y*y))),
        effective_sample_count=1/float(np.sum(kernel*kernel)),
        mean_lag_frames=lag_mean, mean_lag_ms=lag_mean*spec.dt_ms,
        lag_std_ms=math.sqrt(float(temporal @ ((lag-lag_mean)**2)))*spec.dt_ms,
        lag_mode_frames=int(np.argmax(temporal)), lag_mode_ms=float(np.argmax(temporal))*spec.dt_ms,
        lag_p90_frames=lag_quantile(.9), lag_p90_ms=lag_quantile(.9)*spec.dt_ms,
        lag_p99_frames=lag_quantile(.99), lag_p99_ms=lag_quantile(.99)*spec.dt_ms,
        lag_frames=lag.astype(int).tolist(), lag_ms=(lag*spec.dt_ms).tolist(), lag_marginal=temporal.tolist(),
        spatial_x_px=x.astype(int).tolist(), spatial_x_marginal=xm.tolist(),
        spatial_y_px=y.astype(int).tolist(), spatial_y_marginal=ym.tolist(),
        spatial_radial_bin_edges_px=list(range(len(radial_mass)+1)), spatial_radial_bin_mass=radial_mass,
        spatial_marginal=spatial.tolist())


def build_kernels(spec: GammaSTSpec, *, retained_mass: float = .99) -> GammaSTKernels:
    """Build normalized (past lag,y,x) kernels on a deterministic finite box.

    The box contains the continuous half-ball with at least retained_mass of
    each ideal profile's mass. Discrete lattice retention is additionally
    estimated against two much larger, converged boxes and must meet the same
    target. The continuous bound is rigorous; the discrete ratio is explicitly
    a converged numerical estimate, not an analytic infinite-lattice identity.
    """
    import numpy as np
    from scipy.special import gammainc
    if not isinstance(spec, GammaSTSpec):
        raise TypeError("spec must be GammaSTSpec")
    if not math.isfinite(retained_mass) or not .99 <= retained_mass < 1-1e-8:
        raise ValueError("retained_mass must be at least .99 and below 1-1e-8")
    target_mu, matching_width = _target_mu(float(spec.target_sigma_px))
    dimension = 2 if spec.time_scale_ms == 0 else 3
    reference_mu = (spec.reference_n-1)/spec.radius_px
    # Keep a small numerical margin above .99 before inspecting the discrete
    # lattice. This also avoids equating a rounded continuous radius to .99.
    radius = _quantile_radius(spec, target_mu, min(1-2e-8, retained_mass+min(.001, (1-retained_mass)/10)))
    large_radius = _quantile_radius(spec, target_mu, 1-1e-10)
    larger_radius = _quantile_radius(spec, target_mu, 1-1e-12)
    large_bounds = _shape_bounds(spec, large_radius); larger_bounds = _shape_bounds(spec, larger_radius)
    large_totals = _weights(spec, *large_bounds, target_mu, totals_only=True)
    larger_totals = _weights(spec, *larger_bounds, target_mu, totals_only=True)
    convergence = [abs(a-b)/b for a, b in zip(large_totals, larger_totals)]
    if any(not math.isfinite(value) or value > 1e-7 for value in convergence):
        raise ValueError("Discrete support mass did not converge on the independent larger boxes")
    for expansion in range(32):
        half, max_lag = _shape_bounds(spec, radius)
        target, reference, totals = _weights(spec, half, max_lag, target_mu)
        ratios = [a/b for a, b in zip(totals, larger_totals)]
        if min(ratios) >= retained_mass:
            break
        radius *= 1.1
    else:
        raise ValueError("Could not attain the required discrete support retention")
    target /= totals[0]; reference /= totals[1]
    if spec.construction == "separable":
        # Keep both marginals of each own joint kernel exactly: this control is
        # temporal-by-2D-spatial, not independent x-by-y-by-time factorization.
        target = target.sum(axis=(1, 2))[:, None, None] * target.sum(axis=0)[None, :, :]
        reference = reference.sum(axis=(1, 2))[:, None, None] * reference.sum(axis=0)[None, :, :]
    metadata = dict(schema_version=1, spec_id=spec.spec_id, spec=asdict(spec),
        axis_order=["past_lag", "y", "x"], future_samples_used=False, explicit_guard=False,
        construction=spec.construction, reference_field="input and squared input (direct)",
        target_mu_per_px=target_mu, target_mu_2d_matching_square_width_px=matching_width,
        target_mu_rule="discrete 2D radial second moment matched to sigma Gaussian truncated at 4 sigma; inherited in 3D",
        reference_mu_per_px=reference_mu, per_voxel_reference_exponent=spec.reference_n-1,
        ideal_continuous_target_shell_shape=dimension,
        ideal_continuous_reference_shell_shape=spec.reference_n+dimension-1,
        temporal_scale_semantics="continuous reference profile mode along lag axis; not maximum history, mean lag, calcium decay, or imposed decision delay",
        support_shape=list(target.shape), spatial_half_width_px=half,
        maximum_lag_frames=max_lag, maximum_history_ms=max_lag*spec.dt_ms,
        support_geometry="rectangular spatial square by nonnegative lag interval",
        retained_mass_requirement=retained_mass, support_radius_for_continuous_bound_px=radius,
        ideal_continuous_mass_lower_bound=dict(target=float(gammainc(dimension, target_mu*radius)), reference=float(gammainc(spec.reference_n+dimension-1, reference_mu*radius))),
        discrete_retained_mass_estimate=dict(target=ratios[0], reference=ratios[1]),
        discrete_retention_semantics="parent joint profile finite-box mass divided by mass in larger convergence box; separable control is formed only afterward",
        convergence_box_half_width_and_max_lag=list(large_bounds),
        larger_convergence_box_half_width_and_max_lag=list(larger_bounds),
        relative_convergence_mass_gap=dict(target=convergence[0], reference=convergence[1]),
        deterministic_support_expansions=expansion,
        boundary_rule="omit out-of-image pixels and absent past; normalize target and reference valid mass separately; no reflection or future padding",
        target=_moments(target, spec), reference=_moments(reference, spec))
    for kernel in (target, reference):
        if not np.isfinite(kernel).all() or (kernel < 0).any() or not np.isclose(kernel.sum(), 1, rtol=0, atol=2e-14):
            raise ValueError("Invalid normalized kernel")
        kernel.setflags(write=False)
    return GammaSTKernels(spec, target, reference, metadata)


def describe(kernels: GammaSTKernels) -> dict[str, Any]:
    """Return independent JSON-serializable metadata, including both marginals."""
    import copy
    return copy.deepcopy(kernels.metadata)


def support_convergence(spec: GammaSTSpec, *, levels=(.99, .999, .9999)) -> list[dict]:
    """Small deterministic profile-only sensitivity check; no image processing."""
    rows = []
    for level in levels:
        kernels = build_kernels(spec, retained_mass=float(level))
        rows.append(dict(retained_mass=level, support_shape=kernels.metadata["support_shape"],
            target_sigma_x_px=kernels.metadata["target"]["spatial_sigma_x_px"],
            reference_sigma_x_px=kernels.metadata["reference"]["spatial_sigma_x_px"],
            target_mean_lag_ms=kernels.metadata["target"]["mean_lag_ms"],
            reference_mean_lag_ms=kernels.metadata["reference"]["mean_lag_ms"]))
    return rows


def _valid_mass(kernel, height: int, width: int, frame_count: int):
    """Exact rectangle sums for spatial boundaries, cumulatively over past lag."""
    import numpy as np
    hy, hx = kernel.shape[1]//2, kernel.shape[2]//2
    y = np.arange(height); x = np.arange(width)
    ylo, yhi = np.maximum(0, y+hy-height+1), np.minimum(2*hy+1, y+hy+1)
    xlo, xhi = np.maximum(0, x+hx-width+1), np.minimum(2*hx+1, x+hx+1)
    mass = np.empty((min(frame_count, len(kernel)), height, width), dtype=np.float64)
    previous = np.zeros((height, width), dtype=np.float64)
    for lag in range(len(mass)):
        prefix = np.pad(kernel[lag].cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        plane = prefix[yhi[:, None], xhi[None, :]] - prefix[ylo[:, None], xhi[None, :]] - prefix[yhi[:, None], xlo[None, :]] + prefix[ylo[:, None], xlo[None, :]]
        previous = previous + plane
        mass[lag] = previous
    if not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("A target/reference kernel has zero valid mass on the available source grid")
    return mass


def fft_working_memory_estimate(source_shape, kernel_shape, *, chunk_frames: int = 8) -> dict[str, Any]:
    """Conservative live FFT working-set estimate; excludes caller-owned output.

    Includes extra FFT workspace rather than merely counting the input tensor.
    This is an allocation guard, not a measured device-memory guarantee.
    """
    from scipy.fft import next_fast_len
    nt, height, width = map(int, source_shape)
    block_shape = (min(nt, chunk_frames+int(kernel_shape[0])-1), height, width)
    shape = tuple(next_fast_len(a+int(b)-1) for a, b in zip(block_shape, kernel_shape))
    real_bytes = math.prod(shape)*8
    complex_bytes = math.prod((*shape[:-1], shape[-1]//2+1))*16
    return dict(fft_shape=list(shape), input_block_shape=list(block_shape),
        conservative_fft_bytes=10*real_bytes+4*complex_bytes+3*math.prod(block_shape)*8,
        valid_mass_cache_cpu_bytes=2*min(nt, int(kernel_shape[0]))*height*width*8,
        arithmetic="float64 real / complex128 spectrum", fft_workers_cpu=1)


def iter_chunks(values, kernels: GammaSTKernels, *, chunk_frames: int = 8,
                device: str = "cpu", max_fft_bytes: int = 3*1024**3) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Bounded time-overlap FFT convolution; source is TYX or a NumPy memmap.

    Each block reads its current samples and at most max_lag earlier samples.
    Full linear FFT convolution is cropped at the causal output indices, never
    at a centered temporal index. Numerical arithmetic is float64, FFT workers=1.
    Working storage is bounded by spatial dimensions, kernel history and chunk
    size, independent of total movie duration; no full-movie output is allocated.
    device='cuda' optionally uses Torch float64 FFTs with the same linear crop
    and CPU valid-mass normalization. Outputs remain NumPy arrays. CPU/CUDA and
    different chunk sizes can differ by FFT floating-point roundoff.
    """
    import numpy as np
    from scipy.fft import irfftn, next_fast_len, rfftn
    if isinstance(chunk_frames, bool) or not isinstance(chunk_frames, int) or chunk_frames < 1:
        raise ValueError("chunk_frames must be a positive integer")
    source = np.asanyarray(values)
    if source.ndim != 3 or not all(source.shape) or source.dtype.kind not in "fui":
        raise ValueError("values must be a nonempty real numeric TYX array")
    target, reference = kernels.target, kernels.reference
    if target.shape != reference.shape or target.ndim != 3:
        raise ValueError("Target and reference require a common lag/y/x support")
    nframes, height, width = source.shape
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ValueError("device must be cpu or a CUDA device string")
    estimate = fft_working_memory_estimate(source.shape, target.shape, chunk_frames=chunk_frames)
    if not isinstance(max_fft_bytes, int) or max_fft_bytes <= 0 or estimate["conservative_fft_bytes"] > max_fft_bytes:
        raise MemoryError("Conservative FFT working set exceeds the declared byte budget; reduce chunk size or source dimensions")
    torch = None
    if device != "cpu":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA FFT requested but CUDA is unavailable")
        cuda_device = torch.device(device)
        def forward(field, shape):
            return torch.fft.rfftn(torch.tensor(field, dtype=torch.float64, device=cuda_device), s=shape)
        def inverse(spectrum, shape, selection):
            return torch.fft.irfftn(spectrum, s=shape)[selection].contiguous().cpu().numpy()
    else:
        def forward(field, shape):
            return rfftn(field, s=shape, workers=1)
        def inverse(spectrum, shape, selection):
            return irfftn(spectrum, s=shape, workers=1)[selection].copy()
    max_lag = len(target)-1; hy, hx = target.shape[1]//2, target.shape[2]//2
    masses = [_valid_mass(kernel, height, width, nframes) for kernel in (target, reference)]
    cached_shape = None; transforms = None
    for start in range(0, nframes, chunk_frames):
        stop = min(nframes, start+chunk_frames); prior = max(0, start-max_lag)
        block = np.asarray(source[prior:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite source sample in current/past convolution block")
        fft_shape = tuple(next_fast_len(a+b-1) for a, b in zip(block.shape, target.shape))
        if fft_shape != cached_shape:
            transforms = None
            transforms = [forward(kernel, fft_shape) for kernel in (target, reference)]
            cached_shape = fft_shape
        selection = (slice(start-prior, stop-prior), slice(hy, hy+height), slice(hx, hx+width))
        field_fft = forward(block, fft_shape)
        mass_indices = np.minimum(np.arange(start, stop), max_lag)
        a = inverse(field_fft*transforms[0], fft_shape, selection) / masses[0][mass_indices]
        mean = inverse(field_fft*transforms[1], fft_shape, selection) / masses[1][mass_indices]
        del field_fft
        squared_fft = forward(block*block, fft_shape)
        second = inverse(squared_fft*transforms[1], fft_shape, selection) / masses[1][mass_indices]
        del squared_fft
        variance = np.maximum(0, second-mean*mean)
        yield start, stop, {"A": a, "M": mean, "variance": variance}


def apply_chunks(values, kernels: GammaSTKernels, *, chunk_frames: int = 8,
                 output: Mapping[str, Any] | None = None, device: str = "cpu",
                 max_fft_bytes: int = 3*1024**3) -> dict[str, Any]:
    """Collect A/M/variance, or write bounded chunks to supplied output memmaps."""
    import numpy as np
    shape = np.shape(values)
    if output is None:
        result = {key: np.empty(shape, dtype=np.float64) for key in ("A", "M", "variance")}
    else:
        if set(output) != {"A", "M", "variance"}:
            raise ValueError("output must provide exactly A, M, and variance")
        result = dict(output)
        for array in result.values():
            if array.shape != shape or array.dtype.kind != "f" or not array.flags.writeable:
                raise ValueError("Output arrays must be writable floating arrays on the full source grid")
            if np.shares_memory(np.asanyarray(values), array):
                raise ValueError("Outputs must not overwrite source samples needed by later causal chunks")
        arrays = list(result.values())
        if any(np.shares_memory(arrays[i], arrays[j]) for i in range(3) for j in range(i)):
            raise ValueError("Output stage arrays must not overlap")
    for start, stop, stages in iter_chunks(values, kernels, chunk_frames=chunk_frames, device=device, max_fft_bytes=max_fft_bytes):
        for key in result:
            result[key][start:stop] = stages[key]
    return result
