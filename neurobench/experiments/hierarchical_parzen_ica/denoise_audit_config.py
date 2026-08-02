"""Configuration for the sequential Parzen-Innovation denoising audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class DenoiseAuditConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DenoiseAuditConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    architecture_manifest: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    input_lane: dict[str, Any]
    methods: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "DenoiseAuditConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_video", "labels_tsv",
            "architecture_manifest", "output_dir", "preflight_dir", "frames",
            "input_lane", "methods", "evaluation", "visualization", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise DenoiseAuditConfigError(
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
            frames=dict(raw["frames"]), input_lane=dict(raw["input_lane"]),
            methods=dict(raw["methods"]), evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]), resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise DenoiseAuditConfigError("schema_version must be 1 and experiment_id non-empty")
        f = self.frames
        if set(f) != {
            "review_start_ui", "review_end_ui", "quiet_start_ui",
            "quiet_end_ui", "frame_period_ms",
        } or not (
            1 <= int(f["review_start_ui"]) == int(f["quiet_start_ui"])
            <= int(f["quiet_end_ui"]) < int(f["review_end_ui"])
            and float(f["frame_period_ms"]) > 0
        ):
            raise DenoiseAuditConfigError("invalid frame contract")
        if set(self.input_lane) != {
            "id", "reference_half_life_seconds", "correction_fraction",
            "correction_clip_mad",
        } or self.input_lane["id"] != "reference_parzen_innovation":
            raise DenoiseAuditConfigError("invalid input lane")
        required_methods = {
            "pointwise", "spatial_gate", "temporal_gate", "temporal_filters",
            "local_pca", "noise_normalized_pca", "component_parzen",
        }
        if set(self.methods) != required_methods:
            raise DenoiseAuditConfigError("all seven method families are required")
        pointwise = self.methods["pointwise"]
        if set(pointwise) != {"frame_gamma", "robust_gamma", "quiet_wiener"}:
            raise DenoiseAuditConfigError("invalid pointwise variants")
        local = self.methods["local_pca"]
        normalized = self.methods["noise_normalized_pca"]
        component = self.methods["component_parzen"]
        for value in (local, normalized, component):
            if not (
                8 <= int(value["patch_size"]) <= 32
                and 1 <= int(value["stride"]) < int(value["patch_size"])
                and 1 <= int(value["rank"]) <= 8
            ):
                raise DenoiseAuditConfigError("unsafe local method geometry")
        if set(self.evaluation) != {
            "roi_radius_px", "temporal_pool_tau", "nms_distance_px",
            "match_radius_px", "quiet_false_peaks_per_map",
            "fixed_candidates_per_burst", "synthetic_seed", "synthetic_frames",
            "synthetic_size", "synthetic_snr_multipliers",
        }:
            raise DenoiseAuditConfigError("invalid evaluation fields")
        if not (
            1 <= int(self.evaluation["roi_radius_px"]) <= 12
            and int(self.evaluation["nms_distance_px"]) == 6
            and float(self.evaluation["match_radius_px"]) == 6
            and int(self.evaluation["synthetic_frames"]) >= 64
            and int(self.evaluation["synthetic_size"]) >= 32
        ):
            raise DenoiseAuditConfigError("invalid evaluation bounds")
        if set(self.visualization) != {
            "compression", "positive_upper_percentile",
            "remainder_absolute_percentile", "sample_frame_stride",
            "sample_row_stride", "sample_column_stride",
        } or self.visualization["compression"] != "zlib":
            raise DenoiseAuditConfigError("invalid visualization")
        if set(self.resources) != {
            "device", "cpu_threads", "patch_batch_size", "max_ram_mib",
            "max_gpu_memory_mib", "min_free_disk_mib", "max_output_mib",
        } or not (
            self.resources["device"] in {"cpu", "cuda"}
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and 1 <= int(self.resources["patch_batch_size"]) <= 64
            and int(self.resources["max_ram_mib"]) >= 4096
        ):
            raise DenoiseAuditConfigError("invalid resources")

    @property
    def variant_count(self) -> int:
        return 11

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_video", "labels_tsv", "architecture_manifest",
            "output_dir", "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
