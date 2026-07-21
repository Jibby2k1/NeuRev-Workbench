"""Frozen CPU TemporalCNN transfer over one memmap-backed window at a time."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from neurobench.dynamics.models import TemporalCNNResidual

@dataclass(frozen=True)
class NormalizationBounds:
    """Frozen robust intensity bounds fitted only on quiet baseline frames."""
    lower: float
    upper: float
    lower_percentile: float
    upper_percentile: float
    sample_count: int
    baseline_frame_count: int
    baseline_first_index: int
    baseline_last_index: int
    frozen: bool = True
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class LoadedTemporalCNN:
    """A validated model plus a JSON-ready inference contract."""
    model: TemporalCNNResidual
    contract: dict[str, Any]
    @property
    def window_frames(self) -> int:
        return int(self.contract["window_frames"])
    @property
    def input_channels(self) -> int:
        return int(self.contract["input_channels"])
    @property
    def horizon_frames(self) -> int | None:
        return None if self.contract.get("horizon_frames") is None else int(self.contract["horizon_frames"])
    def predict_one(self, window: np.ndarray) -> np.ndarray:
        """Predict one CHW frame from exactly one TCHW window on the CPU."""
        array = np.asarray(window, dtype=np.float32)
        if array.ndim == 3 and self.input_channels == 1:
            array = array[:, None, :, :]
        expected = (self.window_frames, self.input_channels)
        if array.ndim != 4 or tuple(array.shape[:2]) != expected:
            raise ValueError(f"window must have shape ({self.window_frames}, {self.input_channels}, height, width); got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("window contains non-finite values.")
        tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        with torch.inference_mode():
            predicted = self.model(tensor)
        if predicted.device.type != "cpu" or predicted.shape[0] != 1:
            raise RuntimeError("Transfer model violated the CPU, batch-1 inference contract.")
        return predicted[0].detach().numpy().astype(np.float32, copy=False)

def load_temporal_cnn_checkpoint(
    path: str | Path,
    model_id: str = "",
    horizon_frames: int | None = None,
    cpu_threads: int = 1,
) -> LoadedTemporalCNN:
    """Load and validate a frozen ``temporal_cnn_pixel`` checkpoint on CPU."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TemporalCNN checkpoint does not exist: {checkpoint_path}")
    threads = _positive_int(cpu_threads, "cpu_threads")
    torch.set_num_threads(threads)
    payload = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("TemporalCNN checkpoint must contain a mapping.")
    architecture = str(payload.get("architecture") or "")
    if architecture != "temporal_cnn_pixel":
        raise ValueError(f"Expected checkpoint architecture 'temporal_cnn_pixel'; got {architecture!r}.")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("TemporalCNN checkpoint is missing mapping field 'model_state'.")
    input_channels = _positive_int(payload.get("input_channels"), "input_channels")
    window_frames = _positive_int(payload.get("window_frames"), "window_frames")
    hidden_channels = _positive_int(payload.get("hidden_channels"), "hidden_channels")
    num_layers = _positive_int(payload.get("num_layers"), "num_layers")
    residual_scale = _positive_float(payload.get("residual_scale"), "residual_scale")
    checkpoint_horizon = _checkpoint_horizon(payload)
    resolved_horizon = checkpoint_horizon if horizon_frames is None else _positive_int(horizon_frames, "horizon_frames")
    model = TemporalCNNResidual(input_channels, window_frames, hidden_channels, residual_scale, num_blocks=num_layers)
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    warnings: list[str] = []
    if not model_id and not payload.get("model_id") and not payload.get("experiment_id"):
        warnings.append("model_id provenance is absent from the checkpoint and loader arguments.")
    if checkpoint_horizon is None:
        warnings.append("prediction-horizon provenance is absent from the checkpoint.")
    elif horizon_frames is not None and int(horizon_frames) != checkpoint_horizon:
        warnings.append(f"loader horizon_frames={int(horizon_frames)} overrides checkpoint horizon {checkpoint_horizon}.")
    if not any(key in payload for key in ("dataset_path", "dataset_id", "training_dataset")):
        warnings.append("training-dataset provenance is absent from the checkpoint.")
    if not any(key in payload for key in ("normalization", "normalization_mode", "normalization_metadata")):
        warnings.append("normalization provenance is absent; transfer normalization must be reported separately.")
    if not any(key in payload for key in ("grid_size", "grid_shape", "grid_pooling")):
        warnings.append("grid geometry/pooling provenance is absent from the checkpoint.")
    provenance_keys = (
        "model_id experiment_id dataset_id dataset_path training_dataset normalization "
        "normalization_mode normalization_metadata grid_size grid_shape grid_pooling objective seed"
    ).split()
    provenance = {key: _json_safe(payload[key]) for key in provenance_keys if key in payload}
    contract = {
        "schema_version": 1,
        "model_id": str(model_id or payload.get("model_id") or payload.get("experiment_id") or ""),
        "checkpoint_path": str(checkpoint_path),
        "architecture": architecture,
        "input_channels": input_channels,
        "window_frames": window_frames,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "residual_scale": residual_scale,
        "horizon_frames": resolved_horizon,
        "checkpoint_horizon_frames": checkpoint_horizon,
        "device": "cpu",
        "cpu_threads": threads,
        "inference_batch_size": 1,
        "frozen": True,
        "provenance": provenance,
        "warnings": warnings,
    }
    return LoadedTemporalCNN(model=model, contract=contract)

def fit_robust_normalization_bounds(
    provider: Any,
    baseline_indices: Iterable[int],
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    max_samples: int = 1_000_000,
) -> NormalizationBounds:
    """Fit deterministic, memory-bounded bounds from baseline indices only."""
    indices = tuple(int(index) for index in baseline_indices)
    if not indices:
        raise ValueError("baseline_indices must not be empty.")
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("baseline_indices must contain unique non-negative indices.")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("normalization percentiles must satisfy 0 <= lower < upper <= 100.")
    budget = _positive_int(max_samples, "max_samples")
    per_frame = max(1, budget // len(indices))
    samples: list[np.ndarray] = []
    remaining = budget
    for index in indices:
        flat = np.asarray(provider[index]).reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            continue
        take = min(per_frame, remaining, int(finite.size))
        stride = max(1, int(np.ceil(finite.size / take)))
        sample = np.asarray(finite[::stride][:take], dtype=np.float32)
        samples.append(sample)
        remaining -= int(sample.size)
        if remaining <= 0:
            break
    if not samples:
        raise ValueError("baseline frames contain no finite intensity samples.")
    combined = np.concatenate(samples)
    lower, upper = np.percentile(combined, [lower_percentile, upper_percentile])
    if not np.isfinite(lower) or not np.isfinite(upper) or float(upper) <= float(lower):
        raise ValueError("baseline normalization bounds are degenerate.")
    return NormalizationBounds(
        float(lower), float(upper), float(lower_percentile), float(upper_percentile),
        int(combined.size), len(indices), min(indices), max(indices),
    )

def normalize_frame(frame: np.ndarray, bounds: NormalizationBounds) -> np.ndarray:
    """Apply already-frozen bounds without updating them from event data."""
    scale = float(bounds.upper - bounds.lower)
    if scale <= 0.0:
        raise ValueError("normalization bounds must have upper > lower.")
    normalized = (np.asarray(frame, dtype=np.float32) - float(bounds.lower)) / scale
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

def adaptive_max_pool_frame(frame: np.ndarray, output_shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Max-pool the complete field of view into a requested grid shape."""
    height, width = (_positive_int(v, "output_shape") for v in output_shape)
    array, squeeze = _frame_to_chw(frame)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    with torch.inference_mode():
        pooled = F.adaptive_max_pool2d(tensor, (height, width))[0].numpy()
    return pooled[0] if squeeze else pooled

def fixed_max_pool_frame(frame: np.ndarray, pool_size: int = 4) -> np.ndarray:
    """Apply a fixed square max-pool, trimming only bottom/right remainders."""
    size = _positive_int(pool_size, "pool_size")
    array, squeeze = _frame_to_chw(frame)
    _, height, width = array.shape
    trim_h, trim_w = height - height % size, width - width % size
    if trim_h == 0 or trim_w == 0:
        raise ValueError("frame is smaller than the fixed max-pool footprint.")
    cropped = array[:, :trim_h, :trim_w]
    pooled = cropped.reshape(array.shape[0], trim_h // size, size, trim_w // size, size).max(axis=(2, 4))
    pooled = np.asarray(pooled, dtype=np.float32)
    return pooled[0] if squeeze else pooled

def evaluate_temporal_cnn_transfer(
    loaded: LoadedTemporalCNN,
    provider: Any,
    target_indices: Iterable[int],
    *,
    horizon_frames: int | None = None,
    high_change_threshold: float = 0.05,
    frame_rate_hz: float = 50.0,
    decay_hz: float = 10.0,
    core_mask: np.ndarray | None = None,
    ring_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate target frames sequentially with a bounded rolling frame cache."""
    targets = tuple(int(index) for index in target_indices)
    if not targets:
        raise ValueError("target_indices must not be empty.")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("target_indices must be strictly increasing for streaming evaluation.")
    horizon = loaded.horizon_frames if horizon_frames is None else _positive_int(horizon_frames, "horizon_frames")
    if horizon is None:
        raise ValueError("horizon_frames is required when absent from checkpoint provenance.")
    threshold = _positive_float(high_change_threshold, "high_change_threshold")
    decay_factor = float(np.exp(-horizon * _positive_float(decay_hz, "decay_hz") / _positive_float(frame_rate_hz, "frame_rate_hz")))
    window_frames = loaded.window_frames
    first_input = targets[0] - horizon - window_frames + 1
    if first_input < 0:
        raise ValueError("target range does not have enough leading frames for this window/horizon.")
    try:
        provider_length = len(provider)
    except TypeError:
        provider_length = None
    if provider_length is not None and targets[-1] >= provider_length:
        raise IndexError(f"target index {targets[-1]} is outside provider length {provider_length}.")
    cache = _FrameCache(provider, capacity=window_frames + horizon + 2)
    global_acc = _ErrorAccumulator()
    high_acc = _ErrorAccumulator()
    decay_acc = _ErrorAccumulator()
    masked = {name: (_validated_mask(value), _ErrorAccumulator())
              for name, value in (("core", core_mask), ("ring", ring_mask)) if value is not None}
    correlation = _OnlineCorrelation()
    high_count = 0
    total_cells = 0
    shape: tuple[int, int, int] | None = None
    for target_index in targets:
        start = target_index - horizon - window_frames + 1
        stop = target_index - horizon + 1
        window = np.stack([_as_chw(cache.get(index)) for index in range(start, stop)])
        target = _as_chw(cache.get(target_index))
        if shape is None:
            shape = tuple(int(v) for v in target.shape)
            if shape[0] != loaded.input_channels:
                raise ValueError(f"provider has {shape[0]} channels; model expects {loaded.input_channels}.")
            for name, (mask, _acc) in masked.items():
                if mask.shape != shape[1:]:
                    raise ValueError(f"{name}_mask shape {mask.shape} does not match frame shape {shape[1:]}.")
        if tuple(target.shape) != shape or tuple(window.shape[1:]) != shape:
            raise ValueError("provider frame shapes changed during transfer evaluation.")
        prediction = loaded.predict_one(window)
        persistence = window[-1]
        pred_error = prediction - target
        persist_error = persistence - target
        global_acc.update(pred_error, persist_error)
        decay_acc.update(persistence * decay_factor - target, persist_error)
        correlation.update(np.maximum(target - persistence, 0.0), np.maximum(prediction - persistence, 0.0))
        high_mask = np.abs(target - persistence) >= threshold
        high_count += int(high_mask.sum())
        total_cells += int(target.size)
        high_acc.update(pred_error, persist_error, high_mask)
        for mask, accumulator in masked.values():
            accumulator.update(pred_error, persist_error, mask)
    metrics = global_acc.finalize()
    metrics.update(
        {
            "positive_change_correlation": correlation.finalize(),
            "high_change_threshold": threshold,
            "high_change_cell_count": high_count,
            "high_change_fraction": float(high_count / total_cells) if total_cells else 0.0,
            "high_change": high_acc.finalize(),
            "exponential_decay_sensitivity_control": {
                "label": "sensitivity_control_not_fluorescence_model",
                "decay_hz": float(decay_hz),
                "factor": decay_factor,
                **decay_acc.finalize(),
            },
            "masked": {name: accumulator.finalize() for name, (_mask, accumulator) in masked.items()},
        }
    )
    return {
        "schema_version": 1,
        "checkpoint": dict(loaded.contract),
        "evaluation": {
            "target_count": len(targets),
            "first_target_index": targets[0],
            "last_target_index": targets[-1],
            "window_frames": window_frames,
            "horizon_frames": horizon,
            "first_input_index": first_input,
            "last_input_index_for_first_target": targets[0] - horizon,
            "index_formula": "input=[t-horizon-window+1, t-horizon], target=t",
            "inference_batch_size": 1,
            "provider_frame_read_count": cache.read_count,
            "peak_cached_frames": cache.peak_size,
        },
        "metrics": metrics,
    }

class _FrameCache:
    def __init__(self, provider: Any, *, capacity: int):
        self.provider = provider
        self.capacity = int(capacity)
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self.read_count = 0
        self.peak_size = 0
    def get(self, index: int) -> np.ndarray:
        if index in self.frames:
            value = self.frames.pop(index)
            self.frames[index] = value
            return value
        value = np.asarray(self.provider[index])
        self.read_count += 1
        self.frames[index] = value
        while len(self.frames) > self.capacity:
            self.frames.popitem(last=False)
        self.peak_size = max(self.peak_size, len(self.frames))
        return value

class _ErrorAccumulator:
    def __init__(self):
        self.count = 0
        self.pred_sq = self.pred_abs = self.persist_sq = self.persist_abs = 0.0
    def update(self, pred: np.ndarray, persist: np.ndarray, mask: np.ndarray | None = None) -> None:
        pred_values = pred if mask is None else pred[..., mask]
        persist_values = persist if mask is None else persist[..., mask]
        self.count += int(pred_values.size)
        self.pred_sq += float(np.sum(pred_values * pred_values, dtype=np.float64))
        self.pred_abs += float(np.sum(np.abs(pred_values), dtype=np.float64))
        self.persist_sq += float(np.sum(persist_values * persist_values, dtype=np.float64))
        self.persist_abs += float(np.sum(np.abs(persist_values), dtype=np.float64))
    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "cell_count": 0,
                "prediction_mse": None,
                "prediction_mae": None,
                "persistence_mse": None,
                "persistence_mae": None,
                "improvement_over_persistence_mse": None,
            }
        return {
            "cell_count": self.count,
            "prediction_mse": self.pred_sq / self.count,
            "prediction_mae": self.pred_abs / self.count,
            "persistence_mse": self.persist_sq / self.count,
            "persistence_mae": self.persist_abs / self.count,
            "improvement_over_persistence_mse": (self.persist_sq - self.pred_sq) / self.count,
        }

class _OnlineCorrelation:
    def __init__(self):
        self.n = 0
        self.sum_x = self.sum_y = self.sum_x2 = self.sum_y2 = self.sum_xy = 0.0
    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        self.n += int(xv.size)
        self.sum_x += float(xv.sum())
        self.sum_y += float(yv.sum())
        self.sum_x2 += float(np.dot(xv, xv))
        self.sum_y2 += float(np.dot(yv, yv))
        self.sum_xy += float(np.dot(xv, yv))
    def finalize(self) -> float | None:
        if self.n < 2:
            return None
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denom_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        denom_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        denominator = float(np.sqrt(max(0.0, denom_x) * max(0.0, denom_y)))
        return None if denominator == 0.0 else float(numerator / denominator)

def _frame_to_chw(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(frame, dtype=np.float32)
    if array.ndim == 2:
        return array[None, :, :], True
    if array.ndim == 3:
        return array, False
    raise ValueError(f"frame must have shape (height, width) or (channels, height, width); got {array.shape}.")

def _as_chw(frame: np.ndarray) -> np.ndarray:
    array, _squeeze = _frame_to_chw(frame)
    if not np.all(np.isfinite(array)):
        raise ValueError("provider frame contains non-finite values.")
    return array

def _validated_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("core/ring masks must be two-dimensional spatial masks.")
    return array

def _checkpoint_horizon(payload: Mapping[str, Any]) -> int | None:
    for key in ("prediction_horizon_frames", "horizon_frames"):
        if payload.get(key) is not None:
            return _positive_int(payload[key], key)
    windowing = payload.get("windowing")
    if isinstance(windowing, Mapping) and windowing.get("prediction_horizon_frames") is not None:
        return _positive_int(windowing["prediction_horizon_frames"], "windowing.prediction_horizon_frames")
    return None

def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a positive integer.")
    return parsed

def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return parsed

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
