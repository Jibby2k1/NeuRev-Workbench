"""Device-resident Torch representations for the Gamma-LS difference ablation.

Adjacent-frame arms share the frozen causal input operator: spatial Gaussian
(sigma 1 pixel, reflect boundary, truncate 4) followed by an EMA with alpha
0.4. Matched six-lag arms consume acquisition-raw float32 frames because the
frozen v5 model was fit in that domain. Every output records its source frame.
This module only applies frozen models; it neither fits models nor reads labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as functional


SPATIAL_SIGMA_PX = 1.0
GAUSSIAN_TRUNCATE = 4.0
EMA_ALPHA = 0.4
V5_LAGS = (0, 1, 2, 4, 8, 16)
MULTILAG_DIFFERENCE_LAGS = V5_LAGS[1:]


@dataclass(frozen=True)
class DeviceRepresentationMap:
    """A float32 TYX representation and device-resident source-frame indices."""

    values: torch.Tensor
    source_frame_indices: torch.Tensor
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[0] < 1:
            raise ValueError("representation values must be non-empty TYX")
        if self.values.dtype != torch.float32:
            raise ValueError("representation values must have dtype torch.float32")
        indices = self.source_frame_indices
        if indices.ndim != 1 or indices.shape[0] != self.values.shape[0]:
            raise ValueError("one source-frame index is required per output frame")
        if indices.dtype != torch.int64:
            raise ValueError("source-frame indices must have dtype torch.int64")
        if indices.device != self.values.device:
            raise ValueError("values and source-frame indices must share a device")
        if indices.numel() > 1 and not bool(
            torch.all(indices[1:] - indices[:-1] == 1)
        ):
            raise ValueError("source-frame indices must be strictly contiguous")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("representation values must be finite")


@dataclass(frozen=True)
class DeviceRepresentationBundle:
    """Eight label-free representations aligned to one source-frame interval."""

    maps: Mapping[str, DeviceRepresentationMap]
    source_frame_indices: torch.Tensor
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.maps:
            raise ValueError("a representation bundle cannot be empty")
        for name, representation in self.maps.items():
            if (
                representation.source_frame_indices.device
                != self.source_frame_indices.device
            ):
                raise ValueError(f"representation {name!r} uses a different device")
            if not torch.equal(
                representation.source_frame_indices, self.source_frame_indices
            ):
                raise ValueError(f"representation {name!r} is not commonly aligned")


def _video(values: torch.Tensor, *, minimum_frames: int = 1) -> torch.Tensor:
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if values.dtype != torch.float32:
        raise ValueError("values must have dtype torch.float32")
    if values.ndim != 3 or values.numel() == 0 or values.shape[0] < minimum_frames:
        raise ValueError(
            f"values must be non-empty TYX with at least {minimum_frames} frames"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    return values


def _indices(
    count: int,
    source_frame_indices: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if source_frame_indices is None:
        return torch.arange(count, dtype=torch.int64, device=device)
    if not torch.is_tensor(source_frame_indices):
        raise TypeError("source_frame_indices must be a torch.Tensor when provided")
    if source_frame_indices.dtype != torch.int64:
        raise ValueError("source_frame_indices must have dtype torch.int64")
    if source_frame_indices.ndim != 1 or source_frame_indices.shape[0] != count:
        raise ValueError("source_frame_indices must have one entry per input frame")
    if source_frame_indices.device != device:
        raise ValueError("source_frame_indices must already be on the values device")
    result = source_frame_indices.clone()
    if result.numel() > 1 and not bool(torch.all(result[1:] - result[:-1] == 1)):
        raise ValueError("source_frame_indices must be strictly contiguous")
    return result


def _alignment_diagnostics(
    indices: torch.Tensor, *, history_frames: int
) -> dict[str, int]:
    return {
        "output_frame_count": int(indices.numel()),
        "source_frame_first": int(indices[0].item()),
        "source_frame_last": int(indices[-1].item()),
        "history_frames": int(history_frames),
    }


def _scipy_reflect_indices(
    length: int, radius: int, *, device: torch.device
) -> torch.Tensor:
    """Return half-sample-symmetric indices matching SciPy reflect."""

    coordinates = torch.arange(-radius, length + radius, device=device)
    period = 2 * length
    wrapped = torch.remainder(coordinates, period)
    return torch.where(wrapped < length, wrapped, period - 1 - wrapped).to(torch.int64)


def _spatial_gaussian_reflect(values: torch.Tensor) -> torch.Tensor:
    radius = int(GAUSSIAN_TRUNCATE * SPATIAL_SIGMA_PX + 0.5)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
        device=values.device,
    )
    kernel = torch.exp(-0.5 * (coordinates / SPATIAL_SIGMA_PX).square())
    kernel = kernel / kernel.sum()
    frames = values.unsqueeze(1)

    # SciPy reflect is half-sample symmetric. Torch reflection padding is
    # whole-sample symmetric, so explicit indices preserve NumPy semantics.
    y_indices = _scipy_reflect_indices(
        values.shape[-2], radius, device=values.device
    )
    spatial = functional.conv2d(
        torch.index_select(frames, -2, y_indices),
        kernel.reshape(1, 1, -1, 1),
    )
    x_indices = _scipy_reflect_indices(
        values.shape[-1], radius, device=values.device
    )
    return functional.conv2d(
        torch.index_select(spatial, -1, x_indices),
        kernel.reshape(1, 1, 1, -1),
    ).squeeze(1)


def causal_preprocess_common_input(
    values: torch.Tensor,
    *,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Apply the frozen sigma-1 spatial Gaussian and causal alpha-0.4 EMA."""

    source = _video(values)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    spatial = _spatial_gaussian_reflect(source)
    filtered_frames = [spatial[0]]
    for frame in range(1, spatial.shape[0]):
        filtered_frames.append(
            EMA_ALPHA * spatial[frame]
            + (1.0 - EMA_ALPHA) * filtered_frames[-1]
        )
    filtered = torch.stack(filtered_frames)
    diagnostics = {
        "representation": "common_causal_preprocessed_input",
        "preprocessing": {
            "spatial_filter": "gaussian_reflect",
            "spatial_sigma_px": SPATIAL_SIGMA_PX,
            "gaussian_truncate": GAUSSIAN_TRUNCATE,
            "temporal_filter": "causal_ema",
            "ema_alpha": EMA_ALPHA,
            "equivalent_ema_span_frames": 2.0 / EMA_ALPHA - 1.0,
        },
        "alignment": _alignment_diagnostics(indices, history_frames=0),
        "labels_used": False,
    }
    return DeviceRepresentationMap(filtered, indices, diagnostics)


def raw_representation(
    common_input: torch.Tensor,
    *,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Return the already-preprocessed input without another transform."""

    source = _video(common_input)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    diagnostics = {
        "representation": "raw_common_causal_input",
        "definition": "common causally preprocessed input (not acquisition raw)",
        "alignment": _alignment_diagnostics(indices, history_frames=0),
        "labels_used": False,
    }
    return DeviceRepresentationMap(source.clone(), indices, diagnostics)


def _lag(source: torch.Tensor, lag_frames: int) -> int:
    if isinstance(lag_frames, bool) or not isinstance(lag_frames, int):
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    if not 1 <= lag_frames < source.shape[0]:
        raise ValueError("lag_frames must be an integer in [1, T-1]")
    return lag_frames


def signed_difference_representation(
    common_input: torch.Tensor,
    *,
    lag_frames: int = 1,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Return P[t] - P[t-lag], aligned to current source frame t."""

    source = _video(common_input, minimum_frames=2)
    lag = _lag(source, lag_frames)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    output_indices = indices[lag:]
    diagnostics = {
        "representation": "signed_temporal_difference",
        "formula": "P[t] - P[t-lag]",
        "lag_frames": lag,
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag),
        "labels_used": False,
    }
    return DeviceRepresentationMap(
        source[lag:] - source[:-lag], output_indices.clone(), diagnostics
    )


def energy_normalized_difference_representation(
    common_input: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    lag_frames: int = 1,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Return the signed difference divided by two-frame signal energy."""

    import math

    source = _video(common_input, minimum_frames=2)
    lag = _lag(source, lag_frames)
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    previous = source[:-lag]
    current = source[lag:]
    normalized = (current - previous) / torch.sqrt(
        previous.square() + current.square() + float(epsilon)
    )
    output_indices = indices[lag:]
    diagnostics = {
        "representation": "energy_normalized_temporal_difference",
        "formula": "(P[t]-P[t-lag]) / sqrt(P[t-lag]^2 + P[t]^2 + epsilon)",
        "epsilon": float(epsilon),
        "lag_frames": lag,
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag),
        "labels_used": False,
    }
    return DeviceRepresentationMap(normalized, output_indices.clone(), diagnostics)


def _frozen_payload(frozen: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(frozen, Mapping):
        raise TypeError("a frozen model dictionary is required")
    payload = frozen.get("fit", frozen)
    if not isinstance(payload, Mapping):
        raise ValueError("frozen['fit'] must be a dictionary when present")
    return payload


def _matrix(
    payload: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"frozen model is missing {key!r}")
    raw = payload[key]
    if torch.is_tensor(raw):
        result = raw.detach().to(device=device, dtype=torch.float32)
    else:
        result = torch.as_tensor(raw, dtype=torch.float32, device=device)
    if result.shape != shape or not bool(torch.isfinite(result).all()):
        raise ValueError(f"frozen {key!r} must be finite with shape {shape}")
    return result


def _two_frame_stack(source: torch.Tensor, lag: int) -> torch.Tensor:
    return torch.stack((source[:-lag], source[lag:]), dim=0)


def _two_frame_whitening_tensors(
    frozen: Mapping[str, Any], *, device: torch.device
) -> tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]:
    payload = _frozen_payload(frozen)
    mean = _matrix(payload, "mean", (2,), device=device)
    whitening = _matrix(payload, "whitening", (2, 2), device=device)
    if int(torch.linalg.matrix_rank(whitening).item()) < 2:
        raise ValueError("frozen two-frame whitening must be full rank")
    return payload, mean, whitening


def _two_frame_fit_tensors(
    frozen: Mapping[str, Any], *, device: torch.device
) -> tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]:
    payload, mean, whitening = _two_frame_whitening_tensors(
        frozen, device=device
    )
    rotation_key = "demixing" if "demixing" in payload else "rotation"
    rotation = _matrix(payload, rotation_key, (2, 2), device=device)
    effective_demixing = rotation @ whitening
    if int(torch.linalg.matrix_rank(effective_demixing).item()) < 2:
        raise ValueError("frozen two-frame effective demixing must be full rank")
    return payload, mean, effective_demixing


def pca_whitened_derivative_representation(
    common_input: torch.Tensor,
    frozen_two_frame: Mapping[str, Any],
    *,
    lag_frames: int = 1,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Apply the frozen full-rank whitening row closest to [-1,+1]."""

    source = _video(common_input, minimum_frames=2)
    lag = _lag(source, lag_frames)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    _, mean, whitening = _two_frame_whitening_tensors(
        frozen_two_frame, device=source.device
    )
    norms = torch.linalg.vector_norm(whitening, dim=1)
    if bool(torch.any(norms <= torch.finfo(torch.float32).eps)):
        raise ValueError("frozen whitening rows must be non-zero")
    derivative_direction = torch.tensor(
        (-1.0, 1.0), dtype=torch.float32, device=source.device
    ) / (2.0**0.5)
    cosines = (whitening / norms[:, None]) @ derivative_direction
    component = int(torch.argmax(torch.abs(cosines)).item())
    sign = 1 if float(cosines[component].item()) >= 0.0 else -1
    stack = _two_frame_stack(source, lag)
    flat = (stack - mean[:, None, None, None]).reshape(2, -1)
    whitened = (whitening @ flat).reshape(
        2, source.shape[0] - lag, source.shape[1], source.shape[2]
    )
    output_indices = indices[lag:]
    diagnostics = {
        "representation": "frozen_pca_whitened_derivative",
        "fit_reused_without_refitting": True,
        "selection_rule": "whitening row closest to [-1,+1]/sqrt(2)",
        "selected_component": component,
        "selected_sign": sign,
        "lag_frames": lag,
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag),
        "labels_used": False,
    }
    return DeviceRepresentationMap(
        sign * whitened[component], output_indices.clone(), diagnostics
    )


def frozen_two_frame_cs_parzen_representation(
    common_input: torch.Tensor,
    frozen_two_frame: Mapping[str, Any],
    *,
    lag_frames: int = 1,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Apply a frozen CS-Parzen component using effective W @ Q."""

    source = _video(common_input, minimum_frames=2)
    lag = _lag(source, lag_frames)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    payload, mean, effective_demixing = _two_frame_fit_tensors(
        frozen_two_frame, device=source.device
    )
    if "activity_component" not in payload or "activity_sign" not in payload:
        raise ValueError(
            "frozen two-frame model must freeze activity_component and activity_sign"
        )
    component = int(payload["activity_component"])
    sign_value = float(payload["activity_sign"])
    if component not in (0, 1):
        raise ValueError("activity_component must be zero or one")
    if sign_value not in (-1.0, 1.0):
        raise ValueError("activity_sign must be -1 or +1")
    sign = int(sign_value)
    stack = _two_frame_stack(source, lag)
    flat = (stack - mean[:, None, None, None]).reshape(2, -1)
    outputs = (effective_demixing @ flat).reshape(
        2, source.shape[0] - lag, source.shape[1], source.shape[2]
    )
    output_indices = indices[lag:]
    diagnostics = {
        "representation": "frozen_two_frame_cs_parzen_activity",
        "operator": "effective_demixing_equals_W_matmul_Q",
        "fit_reused_without_refitting": True,
        "activity_component": component,
        "activity_sign": sign,
        "lag_frames": lag,
        "alignment": _alignment_diagnostics(output_indices, history_frames=lag),
        "labels_used": False,
    }
    return DeviceRepresentationMap(
        sign * outputs[component], output_indices.clone(), diagnostics
    )


def _integer_tuple(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    raw = payload.get(key, ())
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frozen {key!r} must be an integer sequence") from exc


def _delay_stack(
    source: torch.Tensor, lags: tuple[int, ...]
) -> tuple[torch.Tensor, int]:
    history = max(lags)
    length = source.shape[0] - history
    stack = torch.stack(
        [source[history - lag : history - lag + length] for lag in lags],
        dim=0,
    )
    return stack, history


def multilag_energy_normalized_difference_representation(
    acquisition_raw: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Pool five fixed energy-normalized lag differences at history 16."""

    import math

    source = _video(acquisition_raw, minimum_frames=max(V5_LAGS) + 1)
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    history = max(MULTILAG_DIFFERENCE_LAGS)
    current = source[history:]
    normalized_differences = []
    for lag in MULTILAG_DIFFERENCE_LAGS:
        previous = source[history - lag : source.shape[0] - lag]
        normalized_differences.append(
            (current - previous)
            / torch.sqrt(current.square() + previous.square() + float(epsilon))
        )
    energy = torch.sqrt(
        torch.sum(torch.stack(normalized_differences).square(), dim=0)
    )
    output_indices = indices[history:]
    diagnostics = {
        "representation": "multilag_energy_normalized_difference",
        "formula": (
            "sqrt(sum_lag(((I[t]-I[t-lag]) / "
            "sqrt(I[t]^2+I[t-lag]^2+epsilon))^2))"
        ),
        "input_domain": "acquisition_raw",
        "lags": list(MULTILAG_DIFFERENCE_LAGS),
        "epsilon": float(epsilon),
        "lag_reduction": "sqrt(sum(normalized_difference**2))",
        "alignment": _alignment_diagnostics(output_indices, history_frames=history),
        "labels_used": False,
    }
    return DeviceRepresentationMap(energy, output_indices.clone(), diagnostics)


def pca_whitened_delay_total_energy_representation(
    acquisition_raw: torch.Tensor,
    frozen_v5: Mapping[str, Any],
    *,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Apply frozen delay whitening before rotation and pool all coordinates."""

    source = _video(acquisition_raw, minimum_frames=max(V5_LAGS) + 1)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    payload = _frozen_payload(frozen_v5)
    if payload.get("formulation") != "delay_embedding":
        raise ValueError("frozen v5 model must use formulation='delay_embedding'")
    lags = _integer_tuple(payload, "lags")
    if lags != V5_LAGS:
        raise ValueError(f"frozen v5 lags must be exactly {V5_LAGS}")
    center = _matrix(payload, "center", (len(lags),), device=source.device)
    whitening = _matrix(
        payload, "whitening", (len(lags), len(lags)), device=source.device
    )
    if int(torch.linalg.matrix_rank(whitening).item()) < len(lags):
        raise ValueError("frozen v5 whitening must be full rank")
    stack, history = _delay_stack(source, lags)
    flat = (stack - center[:, None, None, None]).reshape(len(lags), -1)
    whitened = (whitening @ flat).reshape(
        len(lags), source.shape[0] - history, source.shape[1], source.shape[2]
    )
    total_energy = torch.sqrt(torch.sum(whitened.square(), dim=0))
    output_indices = indices[history:]
    diagnostics = {
        "representation": "frozen_v5_pca_whitened_delay_total_energy",
        "fit_reused_without_refitting": True,
        "operator": "frozen_whitening_Q_before_rotation",
        "lags": list(lags),
        "coordinate_count": len(lags),
        "full_rank_whitening": True,
        "ica_rotation_applied": False,
        "input_domain": "acquisition_raw",
        "reduction": "sqrt(sum(all_whitened_coordinates**2))",
        "alignment": _alignment_diagnostics(output_indices, history_frames=history),
        "labels_used": False,
    }
    return DeviceRepresentationMap(
        total_energy, output_indices.clone(), diagnostics
    )


def frozen_v5_residual_group_representation(
    acquisition_raw: torch.Tensor,
    frozen_v5: Mapping[str, Any],
    *,
    source_frame_indices: torch.Tensor | None = None,
) -> DeviceRepresentationMap:
    """Apply frozen six-lag v5 demixing and residual-subspace energy."""

    source = _video(acquisition_raw, minimum_frames=max(V5_LAGS) + 1)
    indices = _indices(source.shape[0], source_frame_indices, device=source.device)
    payload = _frozen_payload(frozen_v5)
    if payload.get("formulation") != "delay_embedding":
        raise ValueError("frozen v5 model must use formulation='delay_embedding'")
    lags = _integer_tuple(payload, "lags")
    if lags != V5_LAGS:
        raise ValueError(f"frozen v5 lags must be exactly {V5_LAGS}")
    center = _matrix(payload, "center", (len(lags),), device=source.device)
    demixing = _matrix(
        payload, "demixing", (len(lags), len(lags)), device=source.device
    )
    residual_indices = _integer_tuple(payload, "residual_indices")
    if not residual_indices or len(set(residual_indices)) != len(residual_indices):
        raise ValueError("frozen v5 residual_indices must be non-empty and unique")
    if min(residual_indices) < 0 or max(residual_indices) >= demixing.shape[0]:
        raise ValueError("frozen v5 residual_indices are out of range")

    stack, history = _delay_stack(source, lags)
    length = source.shape[0] - history
    flat = (stack - center[:, None, None, None]).reshape(len(lags), -1)
    outputs = (demixing @ flat).reshape(
        len(lags), length, source.shape[1], source.shape[2]
    )
    selected = torch.as_tensor(
        residual_indices, dtype=torch.int64, device=source.device
    )
    residual = torch.sqrt(
        torch.sum(torch.index_select(outputs, 0, selected).square(), dim=0)
    )
    output_indices = indices[history:]
    diagnostics = {
        "representation": "frozen_v5_delay_embedding_residual_group",
        "fit_reused_without_refitting": True,
        "lags": list(lags),
        "residual_indices": list(residual_indices),
        "residual_reduction": "sqrt(sum(component**2))",
        "input_domain": "acquisition_raw",
        "alignment": _alignment_diagnostics(output_indices, history_frames=history),
        "labels_used": False,
    }
    return DeviceRepresentationMap(residual, output_indices.clone(), diagnostics)


def align_representation_maps(
    maps: Mapping[str, DeviceRepresentationMap],
) -> tuple[dict[str, DeviceRepresentationMap], torch.Tensor]:
    """Trim device-resident maps to their exact source-frame intersection."""

    if not maps:
        raise ValueError("at least one representation is required")
    first_map = next(iter(maps.values()))
    device = first_map.values.device
    if any(item.values.device != device for item in maps.values()):
        raise ValueError("all representations must share a device")
    start = max(int(item.source_frame_indices[0].item()) for item in maps.values())
    stop = min(int(item.source_frame_indices[-1].item()) for item in maps.values())
    if start > stop:
        raise ValueError("representations have no shared source frames")
    common = torch.arange(start, stop + 1, dtype=torch.int64, device=device)
    aligned: dict[str, DeviceRepresentationMap] = {}
    for name, item in maps.items():
        first = start - int(item.source_frame_indices[0].item())
        last = first + common.numel()
        if first < 0 or last > item.values.shape[0]:
            raise ValueError(f"representation {name!r} does not cover alignment")
        selected_indices = item.source_frame_indices[first:last]
        if not torch.equal(selected_indices, common):
            raise ValueError(f"representation {name!r} does not cover alignment")
        diagnostics = dict(item.diagnostics)
        diagnostics["common_alignment"] = {
            **_alignment_diagnostics(
                common,
                history_frames=int(
                    item.diagnostics.get("alignment", {}).get("history_frames", 0)
                ),
            ),
            "trimmed_prefix_frames": first,
            "trimmed_suffix_frames": int(item.values.shape[0] - last),
        }
        aligned[name] = DeviceRepresentationMap(
            item.values[first:last].clone(), common.clone(), diagnostics
        )
    return aligned, common


def build_representation_bundle(
    values: torch.Tensor,
    *,
    frozen_two_frame: Mapping[str, Any],
    frozen_v5: Mapping[str, Any],
    source_frame_indices: torch.Tensor | None = None,
    energy_epsilon: float = 1e-8,
) -> DeviceRepresentationBundle:
    """Build adjacent-preprocessed and acquisition-raw six-lag arms."""

    source = _video(values, minimum_frames=max(V5_LAGS) + 1)
    preprocessed = causal_preprocess_common_input(
        source, source_frame_indices=source_frame_indices
    )
    common = preprocessed.values
    indices = preprocessed.source_frame_indices
    native = {
        "raw": raw_representation(common, source_frame_indices=indices),
        "difference_signed": signed_difference_representation(
            common, source_frame_indices=indices
        ),
        "difference_energy_normalized": energy_normalized_difference_representation(
            common, epsilon=energy_epsilon, source_frame_indices=indices
        ),
        "difference_multilag_energy_normalized": (
            multilag_energy_normalized_difference_representation(
                source, epsilon=energy_epsilon, source_frame_indices=indices
            )
        ),
        "pca_whitened_derivative": pca_whitened_derivative_representation(
            common, frozen_two_frame, source_frame_indices=indices
        ),
        "pca_whitened_delay_total_energy": (
            pca_whitened_delay_total_energy_representation(
                source, frozen_v5, source_frame_indices=indices
            )
        ),
        "cs_parzen_two_frame": frozen_two_frame_cs_parzen_representation(
            common, frozen_two_frame, source_frame_indices=indices
        ),
        "cs_parzen_delay_residual": frozen_v5_residual_group_representation(
            source, frozen_v5, source_frame_indices=indices
        ),
    }
    aligned, common_indices = align_representation_maps(native)
    diagnostics = {
        "input_shape": list(source.shape),
        "input_dtype": "float32",
        "device": str(source.device),
        "arm_count": len(aligned),
        "labels_used": False,
        "quiet_mad_included": False,
        "preprocessing_applied_once": True,
        "adjacent_frame_input_definition": "common causal Gaussian-plus-EMA input",
        "matched_six_lag_input_definition": (
            "acquisition-raw float32 before Gaussian-plus-EMA preprocessing"
        ),
        "preprocessing": dict(preprocessed.diagnostics["preprocessing"]),
        "common_alignment": _alignment_diagnostics(
            common_indices, history_frames=max(V5_LAGS)
        ),
    }
    return DeviceRepresentationBundle(aligned, common_indices, diagnostics)


__all__ = [
    "DeviceRepresentationBundle",
    "DeviceRepresentationMap",
    "EMA_ALPHA",
    "GAUSSIAN_TRUNCATE",
    "MULTILAG_DIFFERENCE_LAGS",
    "SPATIAL_SIGMA_PX",
    "V5_LAGS",
    "align_representation_maps",
    "build_representation_bundle",
    "causal_preprocess_common_input",
    "energy_normalized_difference_representation",
    "frozen_two_frame_cs_parzen_representation",
    "frozen_v5_residual_group_representation",
    "multilag_energy_normalized_difference_representation",
    "pca_whitened_derivative_representation",
    "pca_whitened_delay_total_energy_representation",
    "raw_representation",
    "signed_difference_representation",
]
