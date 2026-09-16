"""Geometry-only, mean-matched spatial Gamma references for the next study.

Only the reference weights change. The runner must reuse its fixed target
response and conditioning, and fit its floors/cutoffs using setup data only.
No movies, labels, thresholds, or detector outcomes enter this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
from typing import Any

import numpy as np
from scipy.special import gammaincinv, logsumexp


MEAN_FACTORS = (("2of3", 2.0 / 3.0), ("1", 1.0), ("4of3", 4.0 / 3.0))
ORDERS = (3.0, 9.0, 15.0)
MINIMUM_RETAINED_MASS = 0.995
MAXIMUM_HALF_WIDTH_PX = 45
GAUSSIAN_HALF_WIDTH_PX = 4
AVAILABLE_SOURCE_HALO_PX = 49


@dataclass(frozen=True)
class ReferenceSpec:
    spec_id: str
    mean_factor: float
    requested_mean_radius_px: float
    reference_n: float
    reference_mu_per_px: float
    profile_mode_radius_px: float
    spatial_half_width_px: int
    preserved_anchor: bool = False


@dataclass(frozen=True)
class ReferenceKernel:
    weights: np.ndarray
    spec: ReferenceSpec
    metadata: dict[str, Any]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=128)
def _radii(half):
    coordinates = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.hypot(xx, yy)
    radius.setflags(write=False)
    return radius


def _logs(n, mu, half):
    radius = _radii(half)
    logs = np.full(radius.shape, -np.inf, dtype=np.float64)
    positive = radius > 0
    logs[positive] = (n - 1.0) * np.log(radius[positive]) - mu * radius[positive]
    return logs


def _profile(n, mu, half):
    logs = _logs(n, mu, half)
    # Subtract the same finite maximum over the fixed square for stability.
    weights = np.exp(logs - np.max(logs))
    weights /= weights.sum(dtype=np.float64)
    return weights


def _mean(weights):
    return float(np.sum(weights * _radii(weights.shape[0] // 2), dtype=np.float64))


def _solve_mu(n, requested_mean, half):
    """Bisection never changes the spatial domain within a solve."""
    maximum_mean = _mean(_profile(n, 0.0, half))
    _require(requested_mean < maximum_mean, "Requested mean is outside the fixed square's attainable range")
    lower, upper = 0.0, max(1.0, (n + 1.0) / requested_mean)
    while _mean(_profile(n, upper, half)) > requested_mean:
        upper *= 2.0
    for _ in range(96):
        midpoint = (lower + upper) / 2.0
        if _mean(_profile(n, midpoint, half)) > requested_mean:
            lower = midpoint
        else:
            upper = midpoint
    mu = (lower + upper) / 2.0
    return mu, _profile(n, mu, half)


def _retention(n, mu, half):
    """Finite square mass against two numerically converged larger squares."""
    h1 = max(half + 1, math.ceil(float(gammaincinv(n + 1.0, 1 - 1e-10)) / mu))
    h2 = max(h1 + 1, math.ceil(float(gammaincinv(n + 1.0, 1 - 1e-12)) / mu))
    log_current = float(logsumexp(_logs(n, mu, half)))
    log_first = float(logsumexp(_logs(n, mu, h1)))
    log_second = float(logsumexp(_logs(n, mu, h2)))
    gap = abs(math.expm1(log_first - log_second))
    _require(gap <= 1e-7, "Reference retention convergence failed")
    ratio = math.exp(log_current - log_second)
    _require(0 < ratio <= 1 + 1e-12, "Invalid retained reference mass estimate")
    return dict(discrete_retained_mass_estimate=min(1.0, ratio),
                convergence_half_width_px=h1, larger_convergence_half_width_px=h2,
                relative_convergence_mass_gap=gap,
                ideal_continuous_shell_shape=n + 1.0,
                interpretation="Unnormalized finite-square mass / converged larger-square mass; numerical estimate, not an infinite-lattice identity.")


def _conditioned_source(half, sigma_px, left_shift_px):
    """Unit-peak finite Gaussian source after the fixed sigma1 Gaussian input.

    Source footprint follows the fixture's circular four-sigma cutoff. Four
    extra pixels supply the exact source halo for the normalized 9x9 separable
    conditioning stencil. No image-edge reflection is needed in this geometry.
    EMA and amplitude are deliberately absent from this unit spatial overlap.
    """
    from scipy.ndimage import gaussian_filter
    padding = GAUSSIAN_HALF_WIDTH_PX
    coordinates = np.arange(-half - padding, half + padding + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius2 = (xx + left_shift_px) ** 2 + yy ** 2
    raw = np.exp(-radius2 / (2 * sigma_px ** 2)) * (radius2 <= (4 * sigma_px) ** 2)
    conditioned = gaussian_filter(raw, sigma=1.0, truncate=4.0, mode="constant", cval=0.0)
    return conditioned[padding:-padding, padding:-padding]


def _source_overlaps(weights):
    half = weights.shape[0] // 2
    coordinate = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    rows = []
    for label, distance, sigma in (("target", 0, 1.0), ("neighbor8", 8, 2.0),
                                   ("neighbor12", 12, 2.0), ("neighbor16", 16, 2.0)):
        disk = (xx + distance) ** 2 + yy ** 2 <= 4.0
        source = _conditioned_source(half, sigma, distance)
        mean = float(np.sum(weights * source))
        second = float(np.sum(weights * source * source))
        rows.append(dict(location=label, offset_x_px=-distance, offset_y_px=0,
                         inclusive_disk_radius_px=2.0, reference_mass_in_disk=float(weights[disk].sum()),
                         raw_source_sigma_px=sigma, raw_source_peak_amplitude=1.0,
                         conditioned_source_reference_mean_gain=mean,
                         conditioned_source_reference_second_moment_gain=second,
                         conditioned_source_reference_variance_gain=max(0.0, second - mean * mean)))
    return rows


def _describe(weights, spec, retention, solve_history):
    half = weights.shape[0] // 2
    radius = _radii(half)
    mean = _mean(weights)
    second = float(np.sum(weights * radius * radius))
    order = np.argsort(radius.ravel(), kind="stable")
    sorted_radius = radius.ravel()[order]
    cumulative = np.cumsum(weights.ravel()[order], dtype=np.float64)
    probabilities = (.1, .25, .5, .75, .9, .95, .99)
    quantiles = {f"p{round(100 * q)}": float(sorted_radius[min(len(order) - 1, np.searchsorted(cumulative, q))])
                 for q in probabilities}
    radial_bins = np.floor(radius).astype(np.int64)
    radial_mass = np.bincount(radial_bins.ravel(), weights=weights.ravel())
    return dict(schema_version=1, spec_id=spec.spec_id, spec=asdict(spec),
        axis_order=["y", "x"], weight_formula="r^(n-1) exp(-mu*r), normalized on the stated finite square",
        reference_field="conditioned input X and X squared; direct reference moments",
        explicit_guard=False, natural_zero_center=True,
        normalized_mass=float(weights.sum(dtype=np.float64)),
        realized_mean_radius_px=mean, mean_matching_error_px=mean - spec.requested_mean_radius_px,
        realized_radius_sd_px=math.sqrt(max(0.0, second - mean * mean)),
        radial_coefficient_of_variation=math.sqrt(max(0.0, second - mean * mean)) / mean,
        radial_rms_px=math.sqrt(second), radial_mass_quantiles_px=quantiles,
        radial_quantile_definition="Smallest attained Euclidean radius with cumulative normalized pixel mass >= probability; no radial-bin interpolation.",
        modal_per_pixel_weight_radii_px=sorted(set(map(float, radius[weights == weights.max()]))),
        radial_bin_edges_px=list(range(len(radial_mass) + 1)), radial_bin_mass=radial_mass.tolist(),
        modal_radial_mass_bin_px=int(np.argmax(radial_mass)),
        radial_mode_caution="Per-pixel weight mode and annular-mass mode are different; binned mass mode additionally depends on the stated 1px bins.",
        effective_sample_count=1.0 / float(np.sum(weights * weights)),
        effective_sample_count_caution="Geometric inverse concentration 1/sum(w^2), not an independent-sample count after Gaussian/EMA conditioning and not degrees of freedom for Z.",
        support_shape=list(weights.shape), spatial_half_width_px=half,
        gaussian_half_width_px=GAUSSIAN_HALF_WIDTH_PX, required_raw_halo_px=half + GAUSSIAN_HALF_WIDTH_PX,
        available_raw_halo_px=AVAILABLE_SOURCE_HALO_PX, minimum_retained_mass=MINIMUM_RETAINED_MASS,
        retention=retention, fixed_support_solve_history=solve_history,
        kernel_sha256_float64_c_order=hashlib.sha256(weights.tobytes(order="C")).hexdigest(),
        source_overlap=_source_overlaps(weights),
        source_overlap_definition="Inclusive 2px reference-weight mass at fixed target and left-neighbor locations; unit-peak source sigma1 at target and sigma2 at neighbors, four-sigma circular truncation, followed by fixed sigma1/truncate4 conditioning. Pure spatial geometry: no amplitude, EMA, event timing, detector, or measured outcome is implied.",
        target_response="Fixed target response is reused by the runner; this helper neither changes nor evaluates it.")


def mean_matched_reference(requested_mean_radius_px: float, reference_n: float, *,
                           spec_id: str = "custom", mean_factor: float = 1.0,
                           initial_half_width_px: int | None = None,
                           max_half_width_px: int = MAXIMUM_HALF_WIDTH_PX) -> ReferenceKernel:
    """Fit only mu, then inspect retention; an expansion starts a fresh solve."""
    mean, n = float(requested_mean_radius_px), float(reference_n)
    _require(math.isfinite(mean) and mean > 1, "Requested sampled mean radius must be finite and >1px")
    _require(math.isfinite(n) and n > 1, "Reference order must be finite and >1")
    _require(math.isfinite(mean_factor) and mean_factor > 0, "Mean factor must be finite and positive")
    _require(type(max_half_width_px) is int and 1 <= max_half_width_px <= MAXIMUM_HALF_WIDTH_PX,
             "Maximum halfwidth must be an integer in1..45; the raw halo is fixed")
    if initial_half_width_px is None:
        # Continuous 2D shell mean = (n+1)/mu provides an initial scale only.
        initial_mu = (n + 1.0) / mean
        half = max(1, math.ceil(float(gammaincinv(n + 1.0, MINIMUM_RETAINED_MASS)) / initial_mu))
    else:
        _require(type(initial_half_width_px) is int and initial_half_width_px >= 1,
                 "Initial halfwidth must be a positive integer")
        half = initial_half_width_px
    history = []
    while half <= max_half_width_px:
        if mean >= _mean(_profile(n, 0.0, half)):
            history.append(dict(half_width_px=half, status="mean_not_attainable"))
            half += 1
            continue
        mu, weights = _solve_mu(n, mean, half)
        retention = _retention(n, mu, half)
        history.append(dict(half_width_px=half, reference_mu_per_px=mu,
                            retained_mass=retention["discrete_retained_mass_estimate"],
                            status="accepted" if retention["discrete_retained_mass_estimate"] >= MINIMUM_RETAINED_MASS else "expand_for_retention"))
        if retention["discrete_retained_mass_estimate"] >= MINIMUM_RETAINED_MASS:
            break
        half += 1
    else:
        raise ValueError("Mean-matched reference exceeds the fixed45px halfwidth /49px raw halo")
    _require(abs(_mean(weights) - mean) <= 2e-11, "Discrete mean matching failed")
    spec = ReferenceSpec(spec_id, float(mean_factor), mean, n, mu, (n - 1) / mu, half)
    weights.setflags(write=False)
    return ReferenceKernel(weights, spec, _describe(weights, spec, retention, history))


def reference_grid(anchor_reference: np.ndarray | None = None) -> tuple[ReferenceKernel, ...]:
    """Return the fixed nine-cell grid, preserving the exact central reference.

    A supplied anchor must be the actual float64 2D central R7.5/n9 stencil
    from the existing ST implementation. It is copied byte for byte, without
    renormalization; the caller's array and writeability remain untouched.
    """
    from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec, build_kernels
    maintained = build_kernels(GammaSTSpec()).reference[0]
    if anchor_reference is None:
        anchor_reference = maintained
    anchor = np.asarray(anchor_reference)
    _require(anchor.dtype == np.float64 and anchor.shape == maintained.shape,
             "Anchor must be the maintained float64 2D central reference")
    _require(anchor.tobytes(order="C") == maintained.tobytes(order="C"),
             "Anchor bytes differ from the maintained R7.5/n9 reference")
    anchor_mean = _mean(anchor)
    result = []
    for label, factor in MEAN_FACTORS:
        requested_mean = factor * anchor_mean
        for n in ORDERS:
            spec_id = f"mean{label}_n{n:g}"
            if factor == 1.0 and n == 9.0:
                weights = np.array(anchor, dtype=np.float64, order="C", copy=True)
                half = weights.shape[0] // 2
                mu = 8.0 / 7.5
                retention = _retention(n, mu, half)
                _require(retention["discrete_retained_mass_estimate"] >= MINIMUM_RETAINED_MASS,
                         "Exact anchor does not meet the frozen retention requirement")
                spec = ReferenceSpec(spec_id, factor, requested_mean, n, mu, 7.5, half, True)
                weights.setflags(write=False)
                kernel = ReferenceKernel(weights, spec, _describe(weights, spec, retention,
                    [dict(half_width_px=half, status="exact_anchor_preserved_no_solve")]))
            else:
                kernel = mean_matched_reference(requested_mean, n, spec_id=spec_id, mean_factor=factor)
            kernel.metadata["anchor_mean_radius_px"] = anchor_mean
            kernel.metadata["anchor_sha256_float64_c_order"] = hashlib.sha256(anchor.tobytes(order="C")).hexdigest()
            result.append(kernel)
    _require(len(result) == 9 and all(k.spec.spatial_half_width_px + 4 <= 49 for k in result), "Incomplete or unsupported grid")
    return tuple(result)
