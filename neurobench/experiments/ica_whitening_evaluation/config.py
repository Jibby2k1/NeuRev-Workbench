"""Strict configuration for the ICA/whitening evaluation program."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


class ICAWhiteningConfigError(ValueError):
    """Raised when an evaluation manifest violates the frozen contract."""


def _strict(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ICAWhiteningConfigError(f"{context} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ICAWhiteningConfigError(
            f"{context} keys differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def _tuple(value: Any, context: str, cast=str) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise ICAWhiteningConfigError(f"{context} must be a non-empty list")
    try:
        result = tuple(cast(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ICAWhiteningConfigError(f"{context} contains an invalid value") from error
    if len(set(result)) != len(result):
        raise ICAWhiteningConfigError(f"{context} must not contain duplicates")
    return result


@dataclass(frozen=True)
class FrameConfig:
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    frame_period_ms: float


@dataclass(frozen=True)
class DesignConfig:
    master_seed: int
    sobol_points_per_cell: int
    families: tuple[str, ...]
    objectives: tuple[str, ...]
    whitening_geometries: tuple[str, ...]
    covariance_scopes: tuple[str, ...]
    causalities: tuple[str, ...]
    ranks: tuple[int, ...]
    spatial_widths_px: tuple[int, ...]
    temporal_widths_frames: tuple[int, ...]
    screen_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    maximum_joint_dimension: int


@dataclass(frozen=True)
class ContinuousBounds:
    whitening_exponent: tuple[float, float]
    shrinkage: tuple[float, float]
    eigen_floor_ratio: tuple[float, float]
    raw_preserving_blend: tuple[float, float]
    objective_scale: tuple[float, float]


@dataclass(frozen=True)
class EvaluationConfig:
    fixed_candidates_per_burst: int
    nms_distance_px: int
    match_radius_px: int
    temporal_pool: str


@dataclass(frozen=True)
class GateConfig:
    maximum_condition_number: float
    minimum_effective_rank_fraction: float
    minimum_converged_fraction: float
    minimum_truth_source_correlation: float
    maximum_truth_crosstalk: float
    minimum_trace_preservation: float
    minimum_macro_kpr_delta: float


@dataclass(frozen=True)
class ResourceConfig:
    device: str
    cpu_threads: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int
    maximum_fit_samples: int
    application_chunk_size: int


@dataclass(frozen=True)
class ScientificAuditConfig:
    enabled: bool
    opt_out_reason: str | None


@dataclass(frozen=True)
class ICAWhiteningConfig:
    manifest_path: Path
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    label_summary: Path
    output_dir: Path
    frames: FrameConfig
    design: DesignConfig
    continuous_bounds: ContinuousBounds
    evaluation: EvaluationConfig
    gates: GateConfig
    resources: ResourceConfig
    scientific_audit: ScientificAuditConfig

    @classmethod
    def from_json(cls, source: str | Path) -> "ICAWhiteningConfig":
        manifest = Path(source).expanduser().resolve()
        raw = _strict(
            json.loads(manifest.read_text(encoding="utf-8")),
            {
                "schema_version", "experiment_id", "source_video", "labels_tsv",
                "label_summary", "output_dir", "frames", "design",
                "continuous_bounds", "evaluation", "gates", "resources",
                "scientific_audit",
            },
            "top-level",
        )
        base = manifest.parent

        def resolve_path(value: Any) -> Path:
            text = str(value)
            if text.startswith("data://"):
                root = os.environ.get("NEUROBENCH_DATA_ROOT")
                if not root:
                    raise ICAWhiteningConfigError(
                        "data:// paths require NEUROBENCH_DATA_ROOT"
                    )
                return (Path(root).expanduser() / text.removeprefix("data://")).resolve()
            if text.startswith("repo://"):
                repository = manifest.parent.parent
                return (repository / text.removeprefix("repo://")).resolve()
            path = Path(text).expanduser()
            return path.resolve() if path.is_absolute() else (base / path).resolve()
        frame_raw = _strict(raw["frames"], set(FrameConfig.__dataclass_fields__), "frames")
        design_raw = _strict(raw["design"], set(DesignConfig.__dataclass_fields__), "design")
        bounds_raw = _strict(
            raw["continuous_bounds"], set(ContinuousBounds.__dataclass_fields__),
            "continuous_bounds",
        )
        evaluation_raw = _strict(
            raw["evaluation"], set(EvaluationConfig.__dataclass_fields__), "evaluation"
        )
        gate_raw = _strict(raw["gates"], set(GateConfig.__dataclass_fields__), "gates")
        resource_raw = _strict(
            raw["resources"], set(ResourceConfig.__dataclass_fields__), "resources"
        )
        audit_raw = _strict(
            raw["scientific_audit"], set(ScientificAuditConfig.__dataclass_fields__),
            "scientific_audit",
        )
        design = DesignConfig(
            master_seed=int(design_raw["master_seed"]),
            sobol_points_per_cell=int(design_raw["sobol_points_per_cell"]),
            families=_tuple(design_raw["families"], "design.families"),
            objectives=_tuple(design_raw["objectives"], "design.objectives"),
            whitening_geometries=_tuple(
                design_raw["whitening_geometries"], "design.whitening_geometries"
            ),
            covariance_scopes=_tuple(
                design_raw["covariance_scopes"], "design.covariance_scopes"
            ),
            causalities=_tuple(design_raw["causalities"], "design.causalities"),
            ranks=_tuple(design_raw["ranks"], "design.ranks", int),
            spatial_widths_px=_tuple(
                design_raw["spatial_widths_px"], "design.spatial_widths_px", int
            ),
            temporal_widths_frames=_tuple(
                design_raw["temporal_widths_frames"],
                "design.temporal_widths_frames", int,
            ),
            screen_seeds=_tuple(design_raw["screen_seeds"], "design.screen_seeds", int),
            confirmation_seeds=_tuple(
                design_raw["confirmation_seeds"], "design.confirmation_seeds", int
            ),
            maximum_joint_dimension=int(design_raw["maximum_joint_dimension"]),
        )

        def pair(name: str, *, positive: bool = False) -> tuple[float, float]:
            values = _tuple(bounds_raw[name], f"continuous_bounds.{name}", float)
            if len(values) != 2 or values[0] > values[1]:
                raise ICAWhiteningConfigError(f"{name} must be [low, high]")
            if positive and values[0] <= 0:
                raise ICAWhiteningConfigError(f"{name} must be strictly positive")
            return values

        config = cls(
            manifest_path=manifest,
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_video=resolve_path(raw["source_video"]),
            labels_tsv=resolve_path(raw["labels_tsv"]),
            label_summary=resolve_path(raw["label_summary"]),
            output_dir=resolve_path(raw["output_dir"]),
            frames=FrameConfig(**frame_raw),
            design=design,
            continuous_bounds=ContinuousBounds(
                whitening_exponent=pair("whitening_exponent"),
                shrinkage=pair("shrinkage", positive=True),
                eigen_floor_ratio=pair("eigen_floor_ratio", positive=True),
                raw_preserving_blend=pair("raw_preserving_blend"),
                objective_scale=pair("objective_scale", positive=True),
            ),
            evaluation=EvaluationConfig(**evaluation_raw),
            gates=GateConfig(**gate_raw),
            resources=ResourceConfig(**resource_raw),
            scientific_audit=ScientificAuditConfig(**audit_raw),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        def values(instance: Any) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name in instance.__dataclass_fields__:
                value = getattr(instance, name)
                result[name] = list(value) if isinstance(value, tuple) else value
            return result

        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "source_video": str(self.source_video),
            "labels_tsv": str(self.labels_tsv),
            "label_summary": str(self.label_summary),
            "output_dir": str(self.output_dir),
            "frames": values(self.frames),
            "design": values(self.design),
            "continuous_bounds": values(self.continuous_bounds),
            "evaluation": values(self.evaluation),
            "gates": values(self.gates),
            "resources": values(self.resources),
            "scientific_audit": values(self.scientific_audit),
        }

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise ICAWhiteningConfigError("schema_version must be 1 and experiment_id non-empty")
        allowed = {
            "families": {"temporal", "spatial", "joint_spatiotemporal"},
            "objectives": {"fastica_logcosh", "hsic_pairwise"},
            "whitening_geometries": {
                "none", "spatial", "temporal", "spatial_then_temporal",
                "temporal_then_spatial", "joint_spatiotemporal",
            },
            "covariance_scopes": {"global_quiet", "regional_quiet"},
            "causalities": {"centered", "causal"},
        }
        for name, permitted in allowed.items():
            values = set(getattr(self.design, name))
            if not values <= permitted:
                raise ICAWhiteningConfigError(f"unsupported {name}: {sorted(values-permitted)}")
        count = self.design.sobol_points_per_cell
        if count < 1 or count & (count - 1):
            raise ICAWhiteningConfigError("sobol_points_per_cell must be a power of two")
        if any(rank < 2 for rank in self.design.ranks):
            raise ICAWhiteningConfigError("ICA ranks must be at least two")
        for name, widths in (
            ("spatial_widths_px", self.design.spatial_widths_px),
            ("temporal_widths_frames", self.design.temporal_widths_frames),
        ):
            if any(width < 3 or width % 2 == 0 for width in widths):
                raise ICAWhiteningConfigError(f"{name} must contain odd values >=3")
        if self.design.maximum_joint_dimension < 27:
            raise ICAWhiteningConfigError("maximum_joint_dimension must be at least 27")
        f = self.frames
        if not (
            1 <= f.review_start_ui <= f.quiet_start_ui <= f.quiet_end_ui
            < f.review_end_ui and f.frame_period_ms > 0
        ):
            raise ICAWhiteningConfigError("frame intervals are invalid")
        b = self.continuous_bounds
        if not (0 <= b.whitening_exponent[0] <= b.whitening_exponent[1] <= 1):
            raise ICAWhiteningConfigError("whitening_exponent must lie in [0,1]")
        if not (0 <= b.raw_preserving_blend[0] <= b.raw_preserving_blend[1] <= 1):
            raise ICAWhiteningConfigError("raw_preserving_blend must lie in [0,1]")
        if self.scientific_audit.enabled and self.scientific_audit.opt_out_reason:
            raise ICAWhiteningConfigError("enabled scientific audit cannot have opt_out_reason")
        if not self.scientific_audit.enabled and not self.scientific_audit.opt_out_reason:
            raise ICAWhiteningConfigError("disabled scientific audit requires opt_out_reason")
        if self.resources.cpu_threads < 1 or self.resources.maximum_fit_samples < 32:
            raise ICAWhiteningConfigError("resource bounds are invalid")
