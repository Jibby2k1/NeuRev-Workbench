"""Compatibility-aware Cartesian cells with cell-specific Sobol interiors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np
from scipy.stats import qmc

from .config import ContinuousBounds, ICAWhiteningConfig


SPATIAL_WHITENERS = {
    "spatial", "spatial_then_temporal", "temporal_then_spatial",
}
TEMPORAL_WHITENERS = {
    "temporal", "spatial_then_temporal", "temporal_then_spatial",
}


@dataclass(frozen=True)
class DiscreteCell:
    family: str
    objective: str
    whitening_geometry: str
    covariance_scope: str
    causality: str
    rank: int
    spatial_width_px: int | None
    temporal_width_frames: int | None

    @property
    def feature_dimension(self) -> int:
        if self.family == "temporal":
            assert self.temporal_width_frames is not None
            return self.temporal_width_frames
        if self.family == "spatial":
            assert self.spatial_width_px is not None
            return self.spatial_width_px**2
        assert self.spatial_width_px is not None and self.temporal_width_frames is not None
        return self.spatial_width_px**2 * self.temporal_width_frames

    @property
    def cell_id(self) -> str:
        digest = hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.family[:3]}_{self.whitening_geometry[:4]}_{digest}"


def _needs_spatial(family: str, whitening: str) -> bool:
    return family in {"spatial", "joint_spatiotemporal"} or whitening in SPATIAL_WHITENERS or whitening == "joint_spatiotemporal"


def _needs_temporal(family: str, whitening: str) -> bool:
    return family in {"temporal", "joint_spatiotemporal"} or whitening in TEMPORAL_WHITENERS or whitening == "joint_spatiotemporal"


def enumerate_cells(config: ICAWhiteningConfig) -> tuple[DiscreteCell, ...]:
    """Return the complete valid Cartesian set in deterministic order."""
    design = config.design
    cells: list[DiscreteCell] = []
    for family, objective, whitening, scope, rank in itertools.product(
        design.families, design.objectives, design.whitening_geometries,
        design.covariance_scopes, design.ranks,
    ):
        needs_spatial = _needs_spatial(family, whitening)
        needs_temporal = _needs_temporal(family, whitening)
        spatial_values: tuple[int | None, ...] = design.spatial_widths_px if needs_spatial else (None,)
        temporal_values: tuple[int | None, ...] = design.temporal_widths_frames if needs_temporal else (None,)
        causalities = design.causalities if needs_temporal else ("centered",)
        for spatial_width, temporal_width, causality in itertools.product(
            spatial_values, temporal_values, causalities
        ):
            cell = DiscreteCell(
                family=family,
                objective=objective,
                whitening_geometry=whitening,
                covariance_scope=scope,
                causality=causality,
                rank=rank,
                spatial_width_px=spatial_width,
                temporal_width_frames=temporal_width,
            )
            if rank > cell.feature_dimension:
                continue
            if (
                family == "joint_spatiotemporal"
                and cell.feature_dimension > design.maximum_joint_dimension
            ):
                continue
            if whitening == "joint_spatiotemporal":
                assert spatial_width is not None and temporal_width is not None
                if spatial_width**2 * temporal_width > design.maximum_joint_dimension:
                    continue
            cells.append(cell)
    keyed = {cell.cell_id: cell for cell in cells}
    if len(keyed) != len(cells):
        raise RuntimeError("discrete cell identifier collision")
    return tuple(sorted(cells, key=lambda cell: cell.cell_id))


def _linear(value: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + value * (bounds[1] - bounds[0]))


def _log(value: float, bounds: tuple[float, float]) -> float:
    return float(math.exp(math.log(bounds[0]) + value * math.log(bounds[1] / bounds[0])))


def continuous_names(cell: DiscreteCell) -> tuple[str, ...]:
    names = ["raw_preserving_blend", "objective_scale"]
    if cell.whitening_geometry in SPATIAL_WHITENERS:
        names.extend(["spatial_exponent", "spatial_shrinkage", "spatial_eigen_floor_ratio"])
    if cell.whitening_geometry in TEMPORAL_WHITENERS:
        names.extend(["temporal_exponent", "temporal_shrinkage", "temporal_eigen_floor_ratio"])
    if cell.whitening_geometry == "joint_spatiotemporal":
        names.extend(["joint_exponent", "joint_shrinkage", "joint_eigen_floor_ratio"])
    return tuple(names)


def _map_point(
    names: tuple[str, ...], point: np.ndarray, bounds: ContinuousBounds
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in zip(names, np.asarray(point, dtype=float)):
        if name.endswith("_exponent"):
            result[name] = _linear(float(value), bounds.whitening_exponent)
        elif name.endswith("_shrinkage"):
            result[name] = _log(float(value), bounds.shrinkage)
        elif name.endswith("_eigen_floor_ratio"):
            result[name] = _log(float(value), bounds.eigen_floor_ratio)
        elif name == "raw_preserving_blend":
            result[name] = _linear(float(value), bounds.raw_preserving_blend)
        elif name == "objective_scale":
            result[name] = _log(float(value), bounds.objective_scale)
        else:  # pragma: no cover - guarded by continuous_names
            raise RuntimeError(f"unmapped continuous coordinate: {name}")
    return result


def _anchors(cell: DiscreteCell, bounds: ContinuousBounds) -> list[dict[str, float]]:
    names = continuous_names(cell)
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    center: dict[str, float] = {}
    for name in names:
        if name.endswith("_exponent"):
            interval = bounds.whitening_exponent
            low[name], high[name] = interval
            center[name] = 0.5 * sum(interval)
        elif name.endswith("_shrinkage"):
            interval = bounds.shrinkage
            low[name], high[name] = interval
            center[name] = math.sqrt(interval[0] * interval[1])
        elif name.endswith("_eigen_floor_ratio"):
            interval = bounds.eigen_floor_ratio
            low[name], high[name] = interval
            center[name] = math.sqrt(interval[0] * interval[1])
        elif name == "raw_preserving_blend":
            interval = bounds.raw_preserving_blend
            low[name], high[name] = interval
            center[name] = 0.5 * sum(interval)
        else:
            interval = bounds.objective_scale
            low[name], high[name] = interval
            center[name] = math.sqrt(interval[0] * interval[1])
    anchors = [center, low, high]
    if any(name.endswith("_exponent") for name in names):
        no_whitening = center.copy()
        full_whitening = center.copy()
        for name in names:
            if name.endswith("_exponent"):
                no_whitening[name] = 0.0
                full_whitening[name] = 1.0
        anchors.extend([no_whitening, full_whitening])
    if "raw_preserving_blend" in names:
        raw_only = center.copy()
        transformed_only = center.copy()
        raw_only["raw_preserving_blend"] = 0.0
        transformed_only["raw_preserving_blend"] = 1.0
        anchors.extend([raw_only, transformed_only])
    unique = {
        json.dumps(row, sort_keys=True, separators=(",", ":")): row for row in anchors
    }
    return [unique[key] for key in sorted(unique)]


def build_design(config: ICAWhiteningConfig) -> list[dict[str, Any]]:
    """Materialize every cell, boundary anchor, Sobol point, and paired seed."""
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(enumerate_cells(config)):
        names = continuous_names(cell)
        sobol = qmc.Sobol(
            d=len(names), scramble=True, seed=config.design.master_seed + cell_index
        ).random_base2(int(math.log2(config.design.sobol_points_per_cell)))
        points = [
            (f"anchor_{index:02d}", "anchor", values)
            for index, values in enumerate(_anchors(cell, config.continuous_bounds))
        ]
        points.extend(
            (
                f"sobol_{index:03d}", "sobol",
                _map_point(names, point, config.continuous_bounds),
            )
            for index, point in enumerate(sobol)
        )
        for point_id, point_kind, continuous in points:
            for seed in config.design.screen_seeds:
                base = {
                    **asdict(cell),
                    **continuous,
                    "cell_id": cell.cell_id,
                    "point_id": point_id,
                    "point_kind": point_kind,
                    "seed": int(seed),
                }
                digest = hashlib.sha256(
                    json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                rows.append({**base, "fit_id": f"icaw_{digest[:16]}"})
    if len({row["fit_id"] for row in rows}) != len(rows):
        raise RuntimeError("fit identifier collision")
    return rows


def design_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
