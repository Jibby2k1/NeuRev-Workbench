"""Deterministic construction and evaluation of the configured context bank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neurobench.algorithms.multiscale_local_normalization import (
    MSLNResult, SequentialSTContext, SpatialMSLNContext, TemporalMSLNContext,
    sequential_msln, spatial_msln, temporal_msln,
)
from neurobench.experiments.msln_msica.config import MSLNMSICAConfig


@dataclass(frozen=True)
class ContextDefinition:
    context_id: str
    kind: Literal["spatial", "temporal", "spatiotemporal"]
    spatial: SpatialMSLNContext | None = None
    temporal: TemporalMSLNContext | None = None
    sequential: SequentialSTContext | None = None


def ordered_contexts(config: MSLNMSICAConfig) -> tuple[ContextDefinition, ...]:
    result: list[ContextDefinition] = []
    spatial_by_width: dict[int, SpatialMSLNContext] = {}
    temporal_by_window: dict[int, TemporalMSLNContext] = {}
    if config.contexts.spatial.enabled:
        for outer, guard in zip(config.contexts.spatial.outer_widths_px, config.contexts.spatial.guard_widths_px):
            item = SpatialMSLNContext(f"spatial_{outer}_meanstd", outer, guard, config.contexts.spatial.primary_estimator, config.contexts.spatial.scale_floor_percentile)
            spatial_by_width[outer] = item
            result.append(ContextDefinition(item.context_id, "spatial", spatial=item))
    if config.contexts.temporal.enabled:
        for window in config.contexts.temporal.windows_frames:
            item = TemporalMSLNContext(f"temporal_{window}_meanstd", window, config.contexts.temporal.guard_frames, config.contexts.temporal.primary_estimator, config.contexts.temporal.causal, config.contexts.temporal.scale_floor_percentile)
            temporal_by_window[window] = item
            result.append(ContextDefinition(item.context_id, "temporal", temporal=item))
    if config.contexts.spatiotemporal.enabled:
        for pair in config.contexts.spatiotemporal.pairs:
            spatial = spatial_by_width[pair.spatial_outer_width_px]
            temporal = temporal_by_window[pair.temporal_window_frames]
            item = SequentialSTContext(f"st_t{pair.temporal_window_frames}_s{pair.spatial_outer_width_px}_meanstd", spatial.context_id, temporal.context_id, config.contexts.spatiotemporal.mode)
            result.append(ContextDefinition(item.context_id, "spatiotemporal", spatial=spatial, temporal=temporal, sequential=item))
    ids = [item.context_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("context IDs are not unique")
    return tuple(result)


def evaluate_context(video: np.ndarray, definition: ContextDefinition, *, quiet_mask: np.ndarray | None = None) -> MSLNResult:
    if definition.kind == "spatial":
        return spatial_msln(video, definition.spatial, quiet_mask=quiet_mask)  # type: ignore[arg-type]
    if definition.kind == "temporal":
        return temporal_msln(video, definition.temporal, quiet_mask=quiet_mask)  # type: ignore[arg-type]
    return sequential_msln(video, definition.sequential, spatial_context=definition.spatial, temporal_context=definition.temporal, quiet_mask=quiet_mask)  # type: ignore[arg-type]
