"""Strict manifest for the Stage-1 stochastic architecture visual comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .architecture_lanes import ARCHITECTURE_IDS


class ArchitectureVisualConfigError(ValueError):
    pass


def _strict(
    payload: Mapping[str, Any],
    allowed: set[str],
    scope: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ArchitectureVisualConfigError(f"{scope} must be an object")
    values = dict(payload)
    unknown = set(values) - allowed
    missing = allowed - set(values)
    if unknown:
        raise ArchitectureVisualConfigError(
            f"Unknown {scope} fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ArchitectureVisualConfigError(
            f"Missing {scope} fields: {', '.join(sorted(missing))}"
        )
    return values


@dataclass(frozen=True)
class ArchitectureVisualConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    output_dir: Path
    frames: dict[str, Any]
    stochastic: dict[str, Any]
    architectures: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ArchitectureVisualConfig":
        manifest = Path(path).resolve()
        raw = _strict(
            json.loads(manifest.read_text(encoding="utf-8")),
            {
                "schema_version",
                "experiment_id",
                "source_video",
                "output_dir",
                "frames",
                "stochastic",
                "architectures",
                "visualization",
                "resources",
            },
            "top-level",
        )
        frames = _strict(
            raw["frames"],
            {
                "review_start_ui",
                "review_end_ui",
                "quiet_start_ui",
                "quiet_end_ui",
                "frame_period_ms",
            },
            "frames",
        )
        stochastic = _strict(
            raw["stochastic"],
            {
                "fit_sample_pixels",
                "sample_seed",
                "covariance_mode",
                "eigenvalue_floor_ratio",
                "condition_number_max",
                "alpha_min",
                "alpha_max",
                "minimum_confidence_margin",
                "dictionary",
                "optimizer",
                "safety",
            },
            "stochastic",
        )
        stochastic["dictionary"] = _strict(
            stochastic["dictionary"],
            {
                "maximum_centers",
                "minimum_center_separation",
                "bandwidth",
                "bandwidth_min",
                "bandwidth_max",
                "update_rate",
                "replacement_policy",
                "warmup_samples",
                "seed",
            },
            "stochastic.dictionary",
        )
        stochastic["optimizer"] = _strict(
            stochastic["optimizer"],
            {
                "learning_rate",
                "gradient_clip",
                "maximum_angle_update_degrees",
                "batch_size",
                "maximum_iterations",
                "tolerance",
            },
            "stochastic.optimizer",
        )
        stochastic["safety"] = _strict(
            stochastic["safety"],
            {
                "maximum_previous_background_coefficient",
                "maximum_current_observation_coefficient",
                "maximum_reconstruction_operator_norm",
                "maximum_learned_fraction",
                "minimum_learned_fraction",
                "require_convergence_for_learned",
                "unsafe_policy",
            },
            "stochastic.safety",
        )
        architectures = _strict(
            raw["architectures"],
            {
                "ids",
                "quiet_background",
                "reference_half_life_seconds",
                "correction_fraction",
                "correction_clip_mad",
            },
            "architectures",
        )
        visualization = _strict(
            raw["visualization"],
            {
                "positive_lower_percentile",
                "positive_upper_percentile",
                "dynamics_absolute_percentile",
                "sample_frame_stride",
                "sample_row_stride",
                "sample_column_stride",
                "compression",
            },
            "visualization",
        )
        resources = _strict(
            raw["resources"],
            {
                "device",
                "cpu_threads",
                "max_ram_mib",
                "min_free_disk_mib",
                "max_output_mib",
            },
            "resources",
        )
        root = manifest.parent
        config = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / str(raw["source_video"])).resolve(),
            output_dir=(root / str(raw["output_dir"])).resolve(),
            frames=frames,
            stochastic=stochastic,
            architectures=architectures,
            visualization=visualization,
            resources=resources,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise ArchitectureVisualConfigError(
                "schema_version must be 1 and experiment_id non-empty"
            )
        frames = self.frames
        integer_frames = [
            int(frames[key])
            for key in (
                "review_start_ui",
                "review_end_ui",
                "quiet_start_ui",
                "quiet_end_ui",
            )
        ]
        if not (
            1 <= integer_frames[0] == integer_frames[2]
            <= integer_frames[3] < integer_frames[1]
            and float(frames["frame_period_ms"]) > 0
        ):
            raise ArchitectureVisualConfigError("invalid frame contract")
        stochastic = self.stochastic
        if stochastic["covariance_mode"] not in {"ordinary", "robust"}:
            raise ArchitectureVisualConfigError("invalid covariance mode")
        if not 16 <= int(stochastic["fit_sample_pixels"]) <= 65536:
            raise ArchitectureVisualConfigError("fit sample count is unbounded")
        dictionary = stochastic["dictionary"]
        if not (
            2 <= int(dictionary["maximum_centers"]) <= 1024
            and 2 <= int(dictionary["warmup_samples"]) <= 65536
            and 0 < float(dictionary["bandwidth_min"])
            <= float(dictionary["bandwidth"])
            <= float(dictionary["bandwidth_max"])
            and 0 <= float(dictionary["update_rate"]) <= 1
        ):
            raise ArchitectureVisualConfigError("invalid Parzen dictionary")
        optimizer = stochastic["optimizer"]
        if not (
            0 < float(optimizer["learning_rate"]) <= 0.1
            and 0 < float(optimizer["gradient_clip"]) <= 100
            and 0 < float(optimizer["maximum_angle_update_degrees"]) <= 10
            and 2 <= int(optimizer["batch_size"]) <= 65536
            and 1 <= int(optimizer["maximum_iterations"]) <= 10000
            and float(optimizer["tolerance"]) > 0
        ):
            raise ArchitectureVisualConfigError("invalid stochastic optimizer")
        safety = stochastic["safety"]
        if not (
            0 < float(safety["minimum_learned_fraction"])
            <= float(safety["maximum_learned_fraction"]) <= 1
            and bool(safety["require_convergence_for_learned"])
            and safety["unsafe_policy"] == "reference_fallback"
        ):
            raise ArchitectureVisualConfigError("invalid raw-lane safety")
        if float(safety["maximum_learned_fraction"]) != 1.0:
            raise ArchitectureVisualConfigError(
                "the raw stochastic comparison requires learned fraction 1"
            )
        ids = tuple(str(value) for value in self.architectures["ids"])
        if ids != ARCHITECTURE_IDS:
            raise ArchitectureVisualConfigError(
                "all four architecture IDs are required in canonical order"
            )
        architecture = self.architectures
        if architecture["quiet_background"] != "per_pixel_quiet_median":
            raise ArchitectureVisualConfigError(
                "quiet background must be the per-pixel quiet median"
            )
        if not (
            float(architecture["reference_half_life_seconds"]) > 0
            and 0 <= float(architecture["correction_fraction"]) <= 1
            and float(architecture["correction_clip_mad"]) > 0
        ):
            raise ArchitectureVisualConfigError(
                "invalid innovation regularization"
            )
        visual = self.visualization
        if not (
            0 <= float(visual["positive_lower_percentile"])
            < float(visual["positive_upper_percentile"]) <= 100
            and 50 < float(visual["dynamics_absolute_percentile"]) <= 100
            and min(
                int(visual["sample_frame_stride"]),
                int(visual["sample_row_stride"]),
                int(visual["sample_column_stride"]),
            ) >= 1
            and visual["compression"] == "zlib"
        ):
            raise ArchitectureVisualConfigError("invalid visualization controls")
        resources = self.resources
        if not (
            resources["device"] == "cpu"
            and 1 <= int(resources["cpu_threads"]) <= 8
            and int(resources["max_ram_mib"]) >= 1024
            and int(resources["min_free_disk_mib"]) > 0
            and int(resources["max_output_mib"]) > 0
        ):
            raise ArchitectureVisualConfigError("invalid CPU resource contract")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_video"] = str(self.source_video)
        payload["output_dir"] = str(self.output_dir)
        return payload
