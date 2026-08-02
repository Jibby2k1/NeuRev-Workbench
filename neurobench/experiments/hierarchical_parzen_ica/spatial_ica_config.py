"""Configuration for the Parzen-Innovation spatial ICA architecture screen."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class SpatialICAConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SpatialICAConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    architecture_manifest: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    input_lane: dict[str, Any]
    model: dict[str, Any]
    parzen: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "SpatialICAConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_video", "labels_tsv",
            "architecture_manifest", "output_dir", "preflight_dir", "frames",
            "input_lane", "model", "parzen", "evaluation", "visualization",
            "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise SpatialICAConfigError(
                f"top-level fields differ; missing={sorted(fields-set(raw))}, "
                f"unknown={sorted(set(raw)-fields)}"
            )
        root = manifest.parent
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            architecture_manifest=(root / raw["architecture_manifest"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            frames=dict(raw["frames"]),
            input_lane=dict(raw["input_lane"]),
            model=dict(raw["model"]),
            parzen=dict(raw["parzen"]),
            evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise SpatialICAConfigError("schema_version must be 1 and experiment_id non-empty")
        if set(self.frames) != {
            "review_start_ui", "review_end_ui", "quiet_start_ui",
            "quiet_end_ui", "frame_period_ms",
        } or not (
            1 <= int(self.frames["review_start_ui"])
            == int(self.frames["quiet_start_ui"])
            <= int(self.frames["quiet_end_ui"])
            < int(self.frames["review_end_ui"])
            and float(self.frames["frame_period_ms"]) > 0
        ):
            raise SpatialICAConfigError("invalid one-based inclusive frame contract")
        if set(self.input_lane) != {
            "id", "reference_half_life_seconds", "correction_fraction",
            "correction_clip_mad",
        } or self.input_lane["id"] != "reference_parzen_innovation":
            raise SpatialICAConfigError("input must be the accepted Parzen innovation lane")
        model_fields = {
            "patch_size", "rank", "sample_count", "seed",
            "fastica_max_iterations", "fastica_tolerance",
            "patch_lattice_stride", "wiener_lambda_z",
        }
        if set(self.model) != model_fields:
            raise SpatialICAConfigError("invalid model fields")
        patch = int(self.model["patch_size"])
        if not (
            5 <= patch <= 21 and patch % 2 == 1
            and 2 <= int(self.model["rank"]) <= min(32, patch * patch)
            and int(self.model["sample_count"]) >= 4 * patch * patch
            and 1 <= int(self.model["patch_lattice_stride"]) <= patch
            and 0 < float(self.model["wiener_lambda_z"]) <= 10
        ):
            raise SpatialICAConfigError("unsafe spatial ICA geometry or shrinkage")
        if set(self.parzen) != {
            "maximum_centers", "zero_fraction", "active_threshold_z",
            "bandwidth", "noise_variance", "lookup_points", "lookup_abs_z",
        } or not (
            8 <= int(self.parzen["maximum_centers"]) <= 256
            and 0 < float(self.parzen["zero_fraction"]) < 1
            and 0 < float(self.parzen["bandwidth"]) <= 5
            and 0 < float(self.parzen["noise_variance"]) <= 10
            and 512 <= int(self.parzen["lookup_points"]) <= 65536
        ):
            raise SpatialICAConfigError("invalid bounded Parzen posterior")
        if set(self.evaluation) != {
            "roi_radius_px", "temporal_pool_tau", "nms_distance_px",
            "match_radius_px", "quiet_false_peaks_per_map",
            "fixed_candidates_per_burst", "synthetic_seed", "synthetic_frames",
            "synthetic_size", "synthetic_snr_multipliers",
        }:
            raise SpatialICAConfigError("invalid evaluation fields")
        if set(self.visualization) != {
            "compression", "positive_upper_percentile",
            "remainder_absolute_percentile", "sample_frame_stride",
            "sample_row_stride", "sample_column_stride",
        } or self.visualization["compression"] != "zlib":
            raise SpatialICAConfigError("invalid visualization fields")
        if set(self.resources) != {
            "device", "cpu_threads", "frame_batch_size", "max_ram_mib",
            "max_gpu_memory_mib", "min_free_disk_mib", "max_output_mib",
        } or not (
            self.resources["device"] in {"cpu", "cuda"}
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and 1 <= int(self.resources["frame_batch_size"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 4096
        ):
            raise SpatialICAConfigError("invalid resource envelope")

    @property
    def variant_count(self) -> int:
        return 3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "source_video", "labels_tsv", "architecture_manifest",
            "output_dir", "preflight_dir",
        ):
            payload[field] = str(payload[field])
        return payload
