"""Physics-, morphology-, and propagation-aware calcium-imaging features."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _video(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("values must be a finite non-empty TYX video")
    return array


def fit_poisson_gaussian_noise(
    quiet: np.ndarray,
    *,
    intensity_bins: int = 24,
    saturation_value: float | None = None,
) -> dict[str, Any]:
    """Fit ``Var(noise | mean) = intercept + slope * mean`` from frame pairs.

    Consecutive-frame differences suppress static anatomy.  Half their
    variance is used as a conservative measurement-noise estimate.  The fit is
    descriptive because biological changes inside the quiet interval can only
    increase the estimate.
    """
    video = _video(quiet)
    bins = int(intensity_bins)
    if bins < 8:
        raise ValueError("intensity_bins must be at least eight")
    center = np.median(video, axis=0).astype(np.float32)
    differences = np.diff(video, axis=0)
    differences -= np.mean(differences, axis=0, keepdims=True)
    variance = (0.5 * np.mean(np.square(differences), axis=0)).astype(np.float32)
    finite = np.isfinite(center) & np.isfinite(variance)
    if saturation_value is not None:
        finite &= center < float(saturation_value)
    x = center[finite].astype(np.float64)
    y = variance[finite].astype(np.float64)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (x >= low) & (x <= high if high == edges[-1] else x < high)
        if selected.sum() < 32:
            continue
        rows.append({
            "count": int(selected.sum()),
            "mean_intensity": float(np.median(x[selected])),
            "noise_variance": float(np.median(y[selected])),
        })
    if len(rows) < 3:
        raise RuntimeError("insufficient populated intensity bins")
    bx = np.asarray([row["mean_intensity"] for row in rows], dtype=np.float64)
    by = np.asarray([row["noise_variance"] for row in rows], dtype=np.float64)
    weights = np.sqrt(np.asarray([row["count"] for row in rows], dtype=np.float64))
    design = np.column_stack([np.ones_like(bx), bx])
    coefficients = np.linalg.lstsq(design * weights[:, None], by * weights, rcond=None)[0]
    intercept = max(float(coefficients[0]), 0.0)
    slope = max(float(coefficients[1]), 0.0)
    prediction = intercept + slope * bx
    denominator = float(np.sum(weights * np.square(by - np.average(by, weights=weights))))
    r_squared = 1.0 - float(np.sum(weights * np.square(by - prediction))) / max(
        denominator, np.finfo(float).eps
    )
    median_level = float(np.median(bx))
    shot_fraction = slope * median_level / max(intercept + slope * median_level, 1e-12)
    return {
        "model": "descriptive_poisson_gaussian_pair_difference",
        "variance_intercept_raw2": intercept,
        "variance_slope_raw": slope,
        "weighted_r_squared": r_squared,
        "median_level_shot_variance_fraction": shot_fraction,
        "interpretation": (
            "shot_like_dominant" if shot_fraction >= 0.67 else
            "mixed" if shot_fraction >= 0.33 else "additive_like_dominant"
        ),
        "bin_rows": rows,
        "quiet_mean_map": center,
        "pair_difference_variance_map": variance,
    }


def generalized_anscombe(
    values: np.ndarray, *, variance_intercept: float, variance_slope: float
) -> np.ndarray:
    """Apply a finite generalized variance-stabilizing transform."""
    array = np.asarray(values, dtype=np.float32)
    intercept = max(float(variance_intercept), 0.0)
    slope = max(float(variance_slope), 1e-8)
    inside = slope * array + 0.375 * slope * slope + intercept
    return (2.0 / np.sqrt(slope) * np.sqrt(np.maximum(inside, 0.0))).astype(
        np.float32
    )


def radial_zone_histograms_tensor(
    frames,
    *,
    centers: Sequence[float],
    center_radius_px: float,
    shell_radius_px: float,
    outer_radius_px: float,
) -> dict[str, Any]:
    """Return spatially ordered center, shell, and exterior histograms.

    Unlike an unordered square-patch histogram, the three distributions retain
    the radial location of the observed intensities relative to every candidate
    center.  Boundary normalization uses only valid image samples.
    """
    import torch
    import torch.nn.functional as functional

    from neurobench.algorithms.patch_information import quantization_boundaries

    values = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
    radii = (
        float(center_radius_px), float(shell_radius_px), float(outer_radius_px)
    )
    if (
        values.ndim != 3
        or not values.is_floating_point()
        or not 0 < radii[0] < radii[1] < radii[2] <= 15
    ):
        raise ValueError("invalid frames or radial-zone radii")
    radius = int(np.ceil(radii[2]))
    coordinates = torch.arange(-radius, radius + 1, device=values.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    distance = torch.sqrt(xx.to(values.dtype) ** 2 + yy.to(values.dtype) ** 2)
    masks = {
        "center": distance <= radii[0],
        "shell": (distance > radii[0]) & (distance <= radii[1]),
        "outer": (distance > radii[1]) & (distance <= radii[2]),
    }
    boundaries = torch.as_tensor(
        quantization_boundaries(centers), dtype=values.dtype, device=values.device
    )
    indices = torch.bucketize(values.contiguous(), boundaries)
    bin_count = len(tuple(centers))
    one_hot = functional.one_hot(indices, num_classes=bin_count)
    one_hot = one_hot.permute(0, 3, 1, 2).to(values.dtype)
    result = {}
    for name, mask in masks.items():
        kernel = mask.to(values.dtype)
        weights = kernel[None, None].expand(bin_count, 1, -1, -1)
        counts = functional.conv2d(
            one_hot, weights, padding=radius, groups=bin_count
        )
        result[name] = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return result


def causal_local_correlation_feature(
    values: np.ndarray,
    *,
    window_frames: int,
    lag_frames: int,
    spatial_sigma_px: float,
    activity_qualified: bool = True,
) -> np.ndarray:
    """Return causal pixel-to-neighborhood correlation at a selected lag.

    At lag zero this is a local-coherence feature.  Positive lags ask whether a
    neighborhood's past predicts the current pixel, a bounded first test of
    propagation evidence.  No future frame contributes to any output.
    """
    from scipy.ndimage import gaussian_filter

    video = _video(values)
    window = int(window_frames)
    lag = int(lag_frames)
    if window < 4 or lag < 0 or lag >= window or float(spatial_sigma_px) <= 0:
        raise ValueError("invalid causal correlation parameters")
    neighborhood = gaussian_filter(
        video, sigma=(0, float(spatial_sigma_px), float(spatial_sigma_px)),
        mode="reflect",
    ).astype(np.float32)
    shape = video.shape[1:]
    sums = [np.zeros(shape, dtype=np.float64) for _ in range(5)]
    output = np.zeros_like(video, dtype=np.float32)
    count = 0
    for time_index in range(lag, len(video)):
        first = video[time_index].astype(np.float64, copy=False)
        second = neighborhood[time_index - lag].astype(np.float64, copy=False)
        sums[0] += first
        sums[1] += second
        sums[2] += first * first
        sums[3] += second * second
        sums[4] += first * second
        count += 1
        if count > window:
            old_first = video[time_index - window].astype(np.float64, copy=False)
            old_second = neighborhood[time_index - lag - window].astype(
                np.float64, copy=False
            )
            sums[0] -= old_first
            sums[1] -= old_second
            sums[2] -= old_first * old_first
            sums[3] -= old_second * old_second
            sums[4] -= old_first * old_second
            count -= 1
        if count < 4:
            continue
        covariance = sums[4] - sums[0] * sums[1] / count
        first_variance = np.maximum(sums[2] - sums[0] * sums[0] / count, 0.0)
        second_variance = np.maximum(sums[3] - sums[1] * sums[1] / count, 0.0)
        correlation = covariance / np.sqrt(
            np.maximum(first_variance * second_variance, 1e-12)
        )
        evidence = np.maximum(correlation, 0.0)
        if activity_qualified:
            evidence *= np.maximum(video[time_index], 0.0)
        output[time_index] = evidence.astype(np.float32)
    return output


def zcut_template_bank(
    *,
    size_px: int,
    radii_px: Sequence[float],
    z_offsets_fraction: Sequence[float],
    membrane_thickness_px: float,
    psf_sigmas_px: Sequence[float],
) -> list[dict[str, Any]]:
    """Generate normalized 2D cuts through fluorescent spheres and shells."""
    from scipy.ndimage import gaussian_filter

    size = int(size_px)
    if size < 9 or size % 2 != 1:
        raise ValueError("size_px must be an odd integer of at least nine")
    coordinates = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    distance = np.sqrt(xx * xx + yy * yy)
    bank = []
    for radius in [float(value) for value in radii_px]:
        if radius <= float(membrane_thickness_px):
            raise ValueError("radii must exceed membrane thickness")
        for fraction in [float(value) for value in z_offsets_fraction]:
            if not 0 <= fraction < 1:
                raise ValueError("z offsets must be in [0, 1)")
            z = fraction * radius
            outer = np.sqrt(max(radius * radius - z * z, 0.0))
            inner_radius = radius - float(membrane_thickness_px)
            inner = np.sqrt(max(inner_radius * inner_radius - z * z, 0.0))
            definitions = {
                "cytosol": (distance <= outer).astype(np.float32),
                "membrane": (
                    (distance <= outer) & ((distance >= inner) if inner > 0 else True)
                ).astype(np.float32),
            }
            for geometry, image in definitions.items():
                for sigma in [float(value) for value in psf_sigmas_px]:
                    template = gaussian_filter(image, sigma=sigma, mode="constant")
                    template -= float(template.mean())
                    norm = float(np.sqrt(np.sum(template * template)))
                    if norm <= 1e-8:
                        continue
                    template = (template / norm).astype(np.float32)
                    phenotype = (
                        "membrane_cap" if geometry == "membrane" and inner == 0
                        else "membrane_ring" if geometry == "membrane"
                        else "cytosol_center"
                    )
                    bank.append({
                        "template": template,
                        "phenotype": phenotype,
                        "geometry": geometry,
                        "radius_px": radius,
                        "z_offset_fraction": fraction,
                        "psf_sigma_px": sigma,
                    })
    if not bank:
        raise RuntimeError("empty z-cut template bank")
    return bank


def fit_zcut_templates_at_points(
    image: np.ndarray,
    points_xy: Sequence[tuple[float, float]],
    bank: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fit normalized template correlations at fixed global coordinates."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or not bank:
        raise ValueError("image and template bank are required")
    size = int(np.asarray(bank[0]["template"]).shape[0])
    if any(np.asarray(row["template"]).shape != (size, size) for row in bank):
        raise ValueError("template shapes differ")
    radius = size // 2
    padded = np.pad(values, radius, mode="reflect")
    rows = []
    for x_value, y_value in points_xy:
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        crop = padded[y:y + size, x:x + size].astype(np.float32)
        crop -= float(crop.mean())
        norm = float(np.sqrt(np.sum(crop * crop)))
        scored = []
        for definition in bank:
            score = (
                float(np.sum(crop * definition["template"])) / max(norm, 1e-8)
            )
            scored.append((score, definition))
        score, selected = max(scored, key=lambda item: item[0])
        by_phenotype = {
            phenotype: max(
                candidate_score for candidate_score, candidate in scored
                if candidate["phenotype"] == phenotype
            )
            for phenotype in sorted({row["phenotype"] for row in bank})
        }
        rows.append({
            "x_px": float(x_value), "y_px": float(y_value),
            "best_score": score,
            "best_phenotype": selected["phenotype"],
            "best_radius_px": selected["radius_px"],
            "best_z_offset_fraction": selected["z_offset_fraction"],
            "best_psf_sigma_px": selected["psf_sigma_px"],
            "phenotype_scores": by_phenotype,
        })
    return rows


def zcut_response_maps(
    image: np.ndarray, bank: Sequence[dict[str, Any]]
) -> dict[str, np.ndarray]:
    """Return the maximum normalized matched response for each phenotype."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import fftconvolve

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or not bank:
        raise ValueError("image and template bank are required")
    size = int(np.asarray(bank[0]["template"]).shape[0])
    if size % 2 != 1 or any(
        np.asarray(row["template"]).shape != (size, size) for row in bank
    ):
        raise ValueError("template shapes differ or are not odd")
    local_mean = uniform_filter(values, size=size, mode="reflect")
    local_second = uniform_filter(values * values, size=size, mode="reflect")
    local_norm = np.sqrt(
        np.maximum(local_second - local_mean * local_mean, 0.0) * size * size
    )
    phenotypes = sorted({str(row["phenotype"]) for row in bank})
    result = {
        phenotype: np.full(values.shape, -np.inf, dtype=np.float32)
        for phenotype in phenotypes
    }
    for definition in bank:
        template = np.asarray(definition["template"], dtype=np.float32)
        response = fftconvolve(values, template[::-1, ::-1], mode="same")
        response = response / np.maximum(local_norm, 1e-8)
        phenotype = str(definition["phenotype"])
        result[phenotype] = np.maximum(result[phenotype], response)
    margin = size // 2
    for phenotype in result:
        result[phenotype] = np.maximum(result[phenotype], 0.0)
        result[phenotype][:margin] = 0
        result[phenotype][-margin:] = 0
        result[phenotype][:, :margin] = 0
        result[phenotype][:, -margin:] = 0
    return result
