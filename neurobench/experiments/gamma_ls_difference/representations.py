"""Pure, alignment-aware representations for the Gamma-LS difference ablation.

Every temporal representation records the source-frame index of each output
frame. The adjacent-frame level uses one common causal preprocessing pass. The
matched six-lag level instead consumes acquisition-raw float32 frames because
the frozen v5 model was fit in that domain. The bundle trims both levels to the
exact source-frame intersection before downstream Gamma-LS or CFAR evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter


_V5_LAGS = (0, 1, 2, 4, 8, 16)
_V5_POSITIVE_LAGS = _V5_LAGS[1:]
_DERIVATIVE_DIRECTION = np.asarray((-1.0, 1.0), dtype=np.float64) / np.sqrt(2.0)


@dataclass(frozen=True)
class CausalPreprocessingConfig:
    """Configuration for the common causal input of adjacent-frame arms."""

    spatial_sigma_px: float = 1.0
    ema_alpha: float = 0.4
    gaussian_truncate: float = 4.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.spatial_sigma_px) or self.spatial_sigma_px < 0:
            raise ValueError("spatial_sigma_px must be finite and non-negative")
        if not np.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be finite and in (0,1]")
        if not np.isfinite(self.gaussian_truncate) or self.gaussian_truncate <= 0:
            raise ValueError("gaussian_truncate must be finite and positive")


@dataclass(frozen=True)
class RepresentationMap:
    """A TYX representation and the source movie frame attached to each row."""

    values: np.ndarray
    source_frame_indices: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        indices = np.asarray(self.source_frame_indices)
        if values.ndim != 3:
            raise ValueError("representation values must have shape [T,Y,X]")
        if values.shape[0] != indices.size or indices.ndim != 1:
            raise ValueError("one source-frame index is required per output frame")
        if values.shape[0] < 1:
            raise ValueError("a representation must contain at least one frame")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("source-frame indices must be integers")
        if indices.size > 1 and not np.all(np.diff(indices) == 1):
            raise ValueError("source-frame indices must be strictly contiguous")
        if not np.isfinite(values).all():
            raise ValueError("representation values must be finite")


@dataclass(frozen=True)
class RepresentationBundle:
    """Representations aligned to a shared set of source frames."""

    maps: Mapping[str, RepresentationMap]
    source_frame_indices: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.maps:
            raise ValueError("a representation bundle cannot be empty")
        expected = np.asarray(self.source_frame_indices)
        for name, representation in self.maps.items():
            if not np.array_equal(representation.source_frame_indices, expected):
                raise ValueError(f"representation {name!r} is not commonly aligned")


def _video(values: np.ndarray, *, minimum_frames: int = 1) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("values must have shape [T,Y,X]")
    if min(source.shape) < 1 or source.shape[0] < minimum_frames:
        raise ValueError(f"values must contain at least {minimum_frames} frame(s)")
    if not np.isfinite(source).all():
        raise ValueError("values must be finite")
    return source


def _indices(count: int, source_frame_indices: np.ndarray | None) -> np.ndarray:
    if source_frame_indices is None:
        return np.arange(count, dtype=np.int64)
    supplied = np.asarray(source_frame_indices)
    if supplied.ndim != 1 or supplied.shape[0] != count:
        raise ValueError("source_frame_indices must have one entry per input frame")
    if not np.issubdtype(supplied.dtype, np.integer):
        raise ValueError("source_frame_indices must have an integer dtype")
    result = supplied.astype(np.int64, copy=True)
    if result.size > 1 and not np.all(np.diff(result) == 1):
        raise ValueError("source_frame_indices must be strictly contiguous")
    return result


def _alignment_diagnostics(indices: np.ndarray, *, history_frames: int) -> dict[str, Any]:
    return {
        "output_frame_count": int(indices.size),
        "source_frame_first": int(indices[0]),
        "source_frame_last": int(indices[-1]),
        "history_frames": int(history_frames),
    }


def _frozen_payload(frozen: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(frozen, Mapping):
        raise TypeError("a frozen model dictionary is required")
    payload = frozen.get("fit", frozen)
    if not isinstance(payload, Mapping):
        raise ValueError("frozen['fit'] must be a dictionary when present")
    return payload


def _matrix(payload: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"frozen model is missing {key!r}")
    result = np.asarray(payload[key], dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"frozen {key!r} must be a finite array with shape {shape}")
    return result


def causal_preprocess_common_input(
    values: np.ndarray,
    *,
    config: CausalPreprocessingConfig = CausalPreprocessingConfig(),
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Apply spatial Gaussian filtering followed by a causal EMA exactly once."""

    source = _video(values)
    indices = _indices(source.shape[0], source_frame_indices)
    if config.spatial_sigma_px == 0:
        spatial = source.copy()
    else:
        spatial = gaussian_filter(
            source,
            sigma=(0.0, config.spatial_sigma_px, config.spatial_sigma_px),
            mode="reflect",
            truncate=config.gaussian_truncate,
        ).astype(np.float32, copy=False)
    alpha = config.ema_alpha
    filtered = np.empty_like(spatial, dtype=np.float32)
    filtered[0] = spatial[0]
    for frame in range(1, spatial.shape[0]):
        filtered[frame] = alpha * spatial[frame] + (1.0 - alpha) * filtered[frame - 1]
    diagnostics = {
        "representation": "common_causal_preprocessed_input",
        "preprocessing": {
            "spatial_filter": "gaussian_reflect",
            "spatial_sigma_px": float(config.spatial_sigma_px),
            "gaussian_truncate": float(config.gaussian_truncate),
            "temporal_filter": "causal_ema",
            "ema_alpha": float(alpha),
            "equivalent_ema_span_frames": float(2.0 / alpha - 1.0),
        },
        "alignment": _alignment_diagnostics(indices, history_frames=0),
    }
    return RepresentationMap(filtered, indices, diagnostics)


def raw_representation(
    common_input: np.ndarray,
    *,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Return the common causally preprocessed input, without another transform."""

    source = _video(common_input)
    indices = _indices(source.shape[0], source_frame_indices)
    diagnostics = {
        "representation": "raw_common_causal_input",
        "definition": "common causally preprocessed input (not unprocessed acquisition data)",
        "alignment": _alignment_diagnostics(indices, history_frames=0),
    }
    return RepresentationMap(source.copy(), indices, diagnostics)


def signed_difference_representation(
    common_input: np.ndarray,
    *,
    lag_frames: int = 1,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Return ``P[t] - P[t-lag]``, aligned to the current source frame ``t``."""

    source = _video(common_input, minimum_frames=2)
    indices = _indices(source.shape[0], source_frame_indices)
    if not isinstance(lag_frames, (int, np.integer)) or not 1 <= lag_frames < source.shape[0]:
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    difference = (
        source[lag_frames:].astype(np.float64)
        - source[:-lag_frames].astype(np.float64)
    ).astype(np.float32)
    output_indices = indices[lag_frames:]
    diagnostics = {
        "representation": "signed_temporal_difference",
        "formula": "P[t] - P[t-lag]",
        "lag_frames": int(lag_frames),
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag_frames),
    }
    return RepresentationMap(difference, output_indices, diagnostics)


def energy_normalized_difference_representation(
    common_input: np.ndarray,
    *,
    epsilon: float = 1e-8,
    lag_frames: int = 1,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Apply the repository's exact two-frame energy-normalized difference."""

    source = _video(common_input, minimum_frames=2)
    indices = _indices(source.shape[0], source_frame_indices)
    if not isinstance(lag_frames, (int, np.integer)) or not 1 <= lag_frames < source.shape[0]:
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    previous = source[:-lag_frames].astype(np.float64)
    current = source[lag_frames:].astype(np.float64)
    normalized = (current - previous) / np.sqrt(
        previous * previous + current * current + epsilon
    )
    output_indices = indices[lag_frames:]
    diagnostics = {
        "representation": "energy_normalized_temporal_difference",
        "formula": "(P[t]-P[t-lag]) / sqrt(P[t-lag]^2 + P[t]^2 + epsilon)",
        "epsilon": float(epsilon),
        "lag_frames": int(lag_frames),
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag_frames),
    }
    return RepresentationMap(normalized.astype(np.float32), output_indices, diagnostics)


def multilag_energy_normalized_difference_representation(
    acquisition_raw: np.ndarray,
    *,
    epsilon: float = 1e-8,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Combine fixed normalized differences at lags 1, 2, 4, 8, and 16."""

    history = max(_V5_POSITIVE_LAGS)
    source = _video(acquisition_raw, minimum_frames=history + 1)
    indices = _indices(source.shape[0], source_frame_indices)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    length = source.shape[0] - history
    current = source[history:].astype(np.float64)
    squared_energy = np.zeros_like(current, dtype=np.float64)
    for lag in _V5_POSITIVE_LAGS:
        previous = source[history - lag : history - lag + length].astype(np.float64)
        normalized = (current - previous) / np.sqrt(
            current * current + previous * previous + epsilon
        )
        squared_energy += normalized * normalized
    energy = np.sqrt(squared_energy)
    output_indices = indices[history:]
    diagnostics = {
        "representation": "multilag_energy_normalized_difference",
        "formula": (
            "sqrt(sum_lag(((I[t]-I[t-lag]) / "
            "sqrt(I[t]^2+I[t-lag]^2+epsilon))^2))"
        ),
        "input_domain": "acquisition_raw",
        "lags": list(_V5_POSITIVE_LAGS),
        "epsilon": float(epsilon),
        "reduction": "root_sum_of_squares_over_lags",
        "alignment": _alignment_diagnostics(output_indices, history_frames=history),
    }
    return RepresentationMap(energy.astype(np.float32), output_indices, diagnostics)


def quiet_mad_standardized_difference_representation(
    common_input: np.ndarray,
    quiet_frame_mask: np.ndarray,
    *,
    lag_frames: int = 1,
    mad_floor_percentile: float = 10.0,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Standardize signed differences with per-pixel statistics from quiet frames."""

    source = _video(common_input, minimum_frames=2)
    indices = _indices(source.shape[0], source_frame_indices)
    if not isinstance(lag_frames, (int, np.integer)) or not 1 <= lag_frames < source.shape[0]:
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    if not np.isfinite(mad_floor_percentile) or not 0 <= mad_floor_percentile <= 100:
        raise ValueError("mad_floor_percentile must be in [0,100]")
    quiet = np.asarray(quiet_frame_mask)
    if quiet.dtype != np.bool_ or quiet.shape != (source.shape[0],):
        raise ValueError("quiet_frame_mask must be a boolean vector with one entry per input frame")

    difference = source[lag_frames:].astype(np.float64) - source[:-lag_frames].astype(np.float64)
    quiet_for_outputs = quiet[lag_frames:]
    if np.count_nonzero(quiet_for_outputs) < 2:
        raise ValueError("at least two aligned quiet output frames are required")
    quiet_values = difference[quiet_for_outputs]
    center = np.median(quiet_values, axis=0)
    mad = 1.4826 * np.median(np.abs(quiet_values - center), axis=0)
    positive = mad[mad > 0]
    floor = float(np.percentile(positive, mad_floor_percentile)) if positive.size else 1.0
    scale = np.maximum(mad, floor)
    standardized = (difference - center) / scale
    output_indices = indices[lag_frames:]
    diagnostics = {
        "representation": "quiet_mad_standardized_temporal_difference",
        "formula": "(difference - quiet_pixel_median) / max(1.4826*quiet_pixel_MAD, floor)",
        "lag_frames": int(lag_frames),
        "quiet_input_frame_count": int(np.count_nonzero(quiet)),
        "quiet_aligned_output_frame_count": int(np.count_nonzero(quiet_for_outputs)),
        "mad_floor_percentile": float(mad_floor_percentile),
        "mad_floor": floor,
        "floored_pixel_fraction": float(np.mean(mad < floor)),
        "quiet_center_spatial_median": float(np.median(center)),
        "quiet_scale_spatial_median": float(np.median(scale)),
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag_frames),
    }
    return RepresentationMap(standardized.astype(np.float32), output_indices, diagnostics)


def _two_frame_stack(source: np.ndarray, lag_frames: int) -> np.ndarray:
    return np.stack((source[:-lag_frames], source[lag_frames:]), axis=0).astype(
        np.float64,
        copy=False,
    )


def _two_frame_whitening_arrays(
    frozen: Mapping[str, Any],
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    payload = _frozen_payload(frozen)
    mean = _matrix(payload, "mean", (2,))
    whitening = _matrix(payload, "whitening", (2, 2))
    if np.linalg.matrix_rank(whitening) < 2:
        raise ValueError("frozen two-frame whitening must be full rank")
    return payload, mean, whitening


def _two_frame_fit_arrays(
    frozen: Mapping[str, Any],
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload, mean, whitening = _two_frame_whitening_arrays(frozen)
    rotation_key = "demixing" if "demixing" in payload else "rotation"
    rotation = _matrix(payload, rotation_key, (2, 2))
    effective_demixing = rotation @ whitening
    if np.linalg.matrix_rank(effective_demixing) < 2:
        raise ValueError("frozen two-frame effective demixing must be full rank")
    return payload, mean, whitening, effective_demixing


def pca_whitened_derivative_representation(
    common_input: np.ndarray,
    frozen_two_frame: Mapping[str, Any],
    *,
    lag_frames: int = 1,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Project pairs onto the frozen whitened axis closest to ``[-1,+1]``."""

    source = _video(common_input, minimum_frames=2)
    indices = _indices(source.shape[0], source_frame_indices)
    if not isinstance(lag_frames, (int, np.integer)) or not 1 <= lag_frames < source.shape[0]:
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    _, mean, whitening = _two_frame_whitening_arrays(frozen_two_frame)
    norms = np.linalg.norm(whitening, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("frozen whitening rows must be non-zero")
    cosines = (whitening / norms[:, None]) @ _DERIVATIVE_DIRECTION
    component = int(np.argmax(np.abs(cosines)))
    sign = 1 if cosines[component] >= 0 else -1
    pair_stack = _two_frame_stack(source, lag_frames)
    flat = (pair_stack - mean[:, None, None, None]).reshape(2, -1)
    whitened = (whitening @ flat).reshape(
        2,
        source.shape[0] - lag_frames,
        source.shape[1],
        source.shape[2],
    )
    output_indices = indices[lag_frames:]
    diagnostics = {
        "representation": "frozen_pca_whitened_derivative",
        "selection_rule": "whitening row with maximum absolute cosine to [-1,+1]/sqrt(2)",
        "selected_component": component,
        "selected_sign": int(sign),
        "signed_derivative_cosine": float(sign * cosines[component]),
        "signed_effective_direction": (sign * whitening[component]).tolist(),
        "lag_frames": int(lag_frames),
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag_frames),
    }
    return RepresentationMap((sign * whitened[component]).astype(np.float32), output_indices, diagnostics)


def frozen_two_frame_cs_parzen_representation(
    common_input: np.ndarray,
    frozen_two_frame: Mapping[str, Any],
    *,
    lag_frames: int = 1,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Apply a frozen two-frame CS-Parzen activity component without refitting."""

    source = _video(common_input, minimum_frames=2)
    indices = _indices(source.shape[0], source_frame_indices)
    if not isinstance(lag_frames, (int, np.integer)) or not 1 <= lag_frames < source.shape[0]:
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    payload, mean, _, effective_demixing = _two_frame_fit_arrays(frozen_two_frame)
    if "activity_component" not in payload or "activity_sign" not in payload:
        raise ValueError("frozen two-frame model must freeze activity_component and activity_sign")
    component = int(payload["activity_component"])
    sign_value = float(payload["activity_sign"])
    if component not in (0, 1):
        raise ValueError("activity_component must be zero or one")
    if sign_value not in (-1.0, 1.0):
        raise ValueError("activity_sign must be -1 or +1")
    sign = int(sign_value)
    pair_stack = _two_frame_stack(source, lag_frames)
    flat = (pair_stack - mean[:, None, None, None]).reshape(2, -1)
    outputs = (effective_demixing @ flat).reshape(
        2,
        source.shape[0] - lag_frames,
        source.shape[1],
        source.shape[2],
    )
    effective_direction = sign * effective_demixing[component]
    norm = float(np.linalg.norm(effective_direction))
    derivative_cosine = float(effective_direction @ _DERIVATIVE_DIRECTION / norm)
    output_indices = indices[lag_frames:]
    diagnostics = {
        "representation": "frozen_two_frame_cs_parzen_activity",
        "fit_reused_without_refitting": True,
        "activity_component": component,
        "activity_sign": sign,
        "signed_effective_direction": effective_direction.tolist(),
        "signed_derivative_cosine": derivative_cosine,
        "lag_frames": int(lag_frames),
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag_frames),
    }
    return RepresentationMap((sign * outputs[component]).astype(np.float32), output_indices, diagnostics)


def _v5_delay_metadata(
    frozen_v5: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[int, ...], np.ndarray]:
    payload = _frozen_payload(frozen_v5)
    if payload.get("formulation") != "delay_embedding":
        raise ValueError("frozen v5 model must use formulation='delay_embedding'")
    lags = tuple(int(value) for value in payload.get("lags", ()))
    if lags != _V5_LAGS:
        raise ValueError(f"frozen v5 lags must be exactly {_V5_LAGS}")
    center = _matrix(payload, "center", (len(lags),))
    return payload, lags, center


def _v5_delay_stack(
    source: np.ndarray,
    lags: tuple[int, ...],
) -> tuple[np.ndarray, int, int]:
    history = max(lags)
    length = source.shape[0] - history
    stack = np.stack(
        [source[history - lag : history - lag + length] for lag in lags],
        axis=0,
    ).astype(np.float64, copy=False)
    return stack, history, length


def pca_whitened_delay_total_energy_representation(
    acquisition_raw: np.ndarray,
    frozen_v5: Mapping[str, Any],
    *,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Reduce all six frozen v5 whitening coordinates before ICA rotation."""

    source = _video(acquisition_raw, minimum_frames=max(_V5_LAGS) + 1)
    indices = _indices(source.shape[0], source_frame_indices)
    payload, lags, center = _v5_delay_metadata(frozen_v5)
    whitening = _matrix(payload, "whitening", (len(lags), len(lags)))
    if np.linalg.matrix_rank(whitening) != len(lags):
        raise ValueError("frozen v5 whitening must be full rank")
    stack, history, length = _v5_delay_stack(source, lags)
    flat = (stack - center[:, None, None, None]).reshape(len(lags), -1)
    coordinates = (whitening @ flat).reshape(
        len(lags),
        length,
        source.shape[1],
        source.shape[2],
    )
    total_energy = np.sqrt(np.sum(np.square(coordinates), axis=0))
    output_indices = indices[history:]
    diagnostics = {
        "representation": "frozen_v5_pca_whitened_delay_total_energy",
        "fit_reused_without_refitting": True,
        "lags": list(lags),
        "coordinate_count": len(lags),
        "full_rank_whitening": True,
        "ica_rotation_applied": False,
        "input_domain": "acquisition_raw",
        "reduction": "sqrt(sum(all_whitened_coordinates**2))",
        "alignment": _alignment_diagnostics(output_indices, history_frames=history),
    }
    return RepresentationMap(total_energy.astype(np.float32), output_indices, diagnostics)


def frozen_v5_residual_group_representation(
    acquisition_raw: np.ndarray,
    frozen_v5: Mapping[str, Any],
    *,
    source_frame_indices: np.ndarray | None = None,
) -> RepresentationMap:
    """Apply the frozen v5 six-lag demixing and residual-group energy map."""

    source = _video(acquisition_raw, minimum_frames=max(_V5_LAGS) + 1)
    indices = _indices(source.shape[0], source_frame_indices)
    payload, lags, center = _v5_delay_metadata(frozen_v5)
    demixing = _matrix(payload, "demixing", (len(lags), len(lags)))
    residual_indices = tuple(int(value) for value in payload.get("residual_indices", ()))
    if not residual_indices or len(set(residual_indices)) != len(residual_indices):
        raise ValueError("frozen v5 residual_indices must be non-empty and unique")
    if min(residual_indices) < 0 or max(residual_indices) >= demixing.shape[0]:
        raise ValueError("frozen v5 residual_indices are out of range")

    stack, history, length = _v5_delay_stack(source, lags)
    flat = (stack - center[:, None, None, None]).reshape(len(lags), -1)
    outputs = (demixing @ flat).reshape(
        demixing.shape[0],
        length,
        source.shape[1],
        source.shape[2],
    )
    residual = np.sqrt(np.sum(np.square(outputs[list(residual_indices)]), axis=0))
    output_indices = indices[history:]
    diagnostics = {
        "representation": "frozen_v5_delay_embedding_residual_group",
        "fit_reused_without_refitting": True,
        "lags": list(lags),
        "residual_indices": list(residual_indices),
        "residual_reduction": "sqrt(sum(component**2))",
        "input_domain": "acquisition_raw",
        "alignment": _alignment_diagnostics(output_indices, history_frames=history),
    }
    return RepresentationMap(residual.astype(np.float32), output_indices, diagnostics)


def align_representation_maps(
    maps: Mapping[str, RepresentationMap],
) -> tuple[dict[str, RepresentationMap], np.ndarray]:
    """Trim maps to their exact common source-frame intersection."""

    if not maps:
        raise ValueError("at least one representation is required")
    start = max(int(item.source_frame_indices[0]) for item in maps.values())
    stop = min(int(item.source_frame_indices[-1]) for item in maps.values())
    if start > stop:
        raise ValueError("representations have no shared source frames")
    common = np.arange(start, stop + 1, dtype=np.int64)
    aligned: dict[str, RepresentationMap] = {}
    for name, item in maps.items():
        first = int(np.searchsorted(item.source_frame_indices, start))
        last = first + common.size
        selected_indices = item.source_frame_indices[first:last]
        if not np.array_equal(selected_indices, common):
            raise ValueError(f"representation {name!r} does not cover the common alignment")
        diagnostics = dict(item.diagnostics)
        diagnostics["common_alignment"] = {
            **_alignment_diagnostics(common, history_frames=int(
                item.diagnostics.get("alignment", {}).get("history_frames", 0)
            )),
            "trimmed_prefix_frames": first,
            "trimmed_suffix_frames": int(item.values.shape[0] - last),
        }
        aligned[name] = RepresentationMap(
            item.values[first:last].copy(),
            common.copy(),
            diagnostics,
        )
    return aligned, common


def build_representation_bundle(
    values: np.ndarray,
    *,
    frozen_two_frame: Mapping[str, Any],
    frozen_v5: Mapping[str, Any],
    preprocessing: CausalPreprocessingConfig = CausalPreprocessingConfig(),
    source_frame_indices: np.ndarray | None = None,
    energy_epsilon: float = 1e-8,
) -> RepresentationBundle:
    """Build the exact eight adjacent-preprocessed and raw six-lag arms."""

    source = _video(values, minimum_frames=max(_V5_LAGS) + 1)
    original_dtype = str(np.asarray(values).dtype)
    preprocessed = causal_preprocess_common_input(
        source,
        config=preprocessing,
        source_frame_indices=source_frame_indices,
    )
    common = preprocessed.values
    indices = preprocessed.source_frame_indices
    native = {
        "raw": raw_representation(common, source_frame_indices=indices),
        "difference_signed": signed_difference_representation(
            common,
            source_frame_indices=indices,
        ),
        "difference_energy_normalized": energy_normalized_difference_representation(
            common,
            epsilon=energy_epsilon,
            source_frame_indices=indices,
        ),
        "difference_multilag_energy_normalized": multilag_energy_normalized_difference_representation(
            source,
            epsilon=energy_epsilon,
            source_frame_indices=indices,
        ),
        "pca_whitened_derivative": pca_whitened_derivative_representation(
            common,
            frozen_two_frame,
            source_frame_indices=indices,
        ),
        "pca_whitened_delay_total_energy": pca_whitened_delay_total_energy_representation(
            source,
            frozen_v5,
            source_frame_indices=indices,
        ),
        "cs_parzen_two_frame": frozen_two_frame_cs_parzen_representation(
            common,
            frozen_two_frame,
            source_frame_indices=indices,
        ),
        "cs_parzen_delay_residual": frozen_v5_residual_group_representation(
            source,
            frozen_v5,
            source_frame_indices=indices,
        ),
    }
    aligned, common_indices = align_representation_maps(native)
    diagnostics = {
        "input_shape": list(source.shape),
        "input_dtype": original_dtype,
        "arm_count": len(aligned),
        "raw_arm_definition": "common causal preprocessed input",
        "adjacent_frame_input_definition": "common causal Gaussian-plus-EMA input",
        "matched_six_lag_input_definition": (
            "acquisition-raw float32 before Gaussian-plus-EMA preprocessing"
        ),
        "preprocessing_applied_once": True,
        "preprocessing": dict(preprocessed.diagnostics["preprocessing"]),
        "common_alignment": _alignment_diagnostics(
            common_indices,
            history_frames=max(_V5_LAGS),
        ),
        "native_alignment": {
            name: dict(item.diagnostics["alignment"])
            for name, item in native.items()
        },
    }
    return RepresentationBundle(aligned, common_indices, diagnostics)


__all__ = [
    "CausalPreprocessingConfig",
    "RepresentationBundle",
    "RepresentationMap",
    "align_representation_maps",
    "build_representation_bundle",
    "causal_preprocess_common_input",
    "energy_normalized_difference_representation",
    "frozen_two_frame_cs_parzen_representation",
    "frozen_v5_residual_group_representation",
    "multilag_energy_normalized_difference_representation",
    "pca_whitened_delay_total_energy_representation",
    "pca_whitened_derivative_representation",
    "quiet_mad_standardized_difference_representation",
    "raw_representation",
    "signed_difference_representation",
]
