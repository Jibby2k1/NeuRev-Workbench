"""Frozen configuration for the Spon Ca Burst activity-feature utility study."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


FEATURE_IDS = (
    "derivative_positive_lag1",
    "derivative_negative_lag1",
    "derivative_absolute_lag1",
    "derivative_power1p5_lag1",
    "derivative_log_square_lag1",
    "derivative_huber_lag1",
    "derivative_square_lag1",
    "derivative_square_lag2",
    "derivative_square_lag4",
    "local_psd_signal",
    "local_psd_correction",
    "cross_scale_rank",
    "cross_scale_recall",
    "asymmetric_state",
    "asymmetric_innovation",
    "persistence_activity_gate",
    "persistent_artifact_score",
    "morphology_center_isolated",
    "morphology_membrane_isolated",
    "morphology_center_crowded",
    "morphology_membrane_crowded",
    "cfar_score",
    "cfar_background",
    "cfar_noise",
    "spatial_coherence",
)


class FeatureUtilityConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureUtilityConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    architecture_manifest: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    input_lane: dict[str, Any]
    shared_ica: dict[str, Any]
    feature_bank: dict[str, Any]
    fusion: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "FeatureUtilityConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version",
            "experiment_id",
            "source_video",
            "labels_tsv",
            "architecture_manifest",
            "output_dir",
            "preflight_dir",
            "frames",
            "input_lane",
            "shared_ica",
            "feature_bank",
            "fusion",
            "evaluation",
            "visualization",
            "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise FeatureUtilityConfigError(
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
            shared_ica=dict(raw["shared_ica"]),
            feature_bank=dict(raw["feature_bank"]),
            fusion=dict(raw["fusion"]),
            evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise FeatureUtilityConfigError("invalid schema version or experiment id")
        if set(self.frames) != {
            "review_start_ui",
            "review_end_ui",
            "quiet_start_ui",
            "quiet_end_ui",
            "frame_period_ms",
        } or not (
            1
            <= int(self.frames["review_start_ui"])
            == int(self.frames["quiet_start_ui"])
            <= int(self.frames["quiet_end_ui"])
            < int(self.frames["review_end_ui"])
        ):
            raise FeatureUtilityConfigError("invalid frame contract")
        if set(self.input_lane) != {
            "id",
            "reference_half_life_seconds",
            "correction_fraction",
            "correction_clip_mad",
        } or self.input_lane["id"] != "reference_parzen_innovation":
            raise FeatureUtilityConfigError("invalid carrier contract")
        if set(self.shared_ica) != {
            "patch_size",
            "rank",
            "sample_count",
            "seed",
            "fastica_max_iterations",
            "fastica_tolerance",
            "wiener_lambda_z",
            "parzen",
        }:
            raise FeatureUtilityConfigError("invalid shared ICA fields")
        required_bank = {
            "clip_z",
            "derivative",
            "local_psd",
            "cross_scale",
            "asymmetric",
            "persistence",
            "morphology",
            "cfar",
            "storage_dtype",
            "tiff_feature_ids",
        }
        if set(self.feature_bank) != required_bank:
            raise FeatureUtilityConfigError("invalid feature-bank fields")
        derivative = self.feature_bank["derivative"]
        if set(derivative) != {
            "spatial_sigma_px",
            "lags",
            "power",
            "energy_tau_z",
            "huber_delta_z",
        } or derivative["lags"] != [1, 2, 4]:
            raise FeatureUtilityConfigError("invalid derivative feature contract")
        if self.feature_bank["storage_dtype"] != "float16":
            raise FeatureUtilityConfigError("feature bank must use float16 storage")
        tiff_ids = self.feature_bank["tiff_feature_ids"]
        if (
            len(tiff_ids) != 10
            or len(set(tiff_ids)) != len(tiff_ids)
            or set(tiff_ids) - set(FEATURE_IDS)
        ):
            raise FeatureUtilityConfigError("exactly ten valid TIFF features required")
        if set(self.fusion) != {
            "boost_values",
            "gate_floors",
            "learning_rate",
            "epochs",
            "l2_to_carrier",
            "maximum_total_weight",
            "top_features_per_fold",
        } or not (
            self.fusion["boost_values"] == [0.25, 0.5, 1.0]
            and self.fusion["gate_floors"] == [0.5, 0.75, 0.9]
            and int(self.fusion["top_features_per_fold"]) == 6
            and 0 < float(self.fusion["maximum_total_weight"]) <= 1
        ):
            raise FeatureUtilityConfigError("invalid fusion contract")
        if set(self.evaluation) != {
            "roi_radius_px",
            "temporal_pool_tau",
            "nms_distance_px",
            "match_radius_px",
            "quiet_false_peaks_per_map",
            "fixed_candidates_per_burst",
            "synthetic_seed",
            "synthetic_frames",
            "synthetic_size",
            "synthetic_snr_multipliers",
        }:
            raise FeatureUtilityConfigError("invalid evaluation contract")
        if set(self.visualization) != {
            "compression",
            "upper_percentile",
            "sample_frame_stride",
            "sample_row_stride",
            "sample_column_stride",
        } or self.visualization["compression"] != "zlib":
            raise FeatureUtilityConfigError("invalid visualization contract")
        if set(self.resources) != {
            "device",
            "cpu_threads",
            "frame_batch_size",
            "max_ram_mib",
            "max_gpu_memory_mib",
            "min_free_disk_mib",
            "max_output_mib",
        } or not (
            self.resources["device"] in {"cpu", "cuda"}
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 8192
        ):
            raise FeatureUtilityConfigError("invalid resource contract")

    @property
    def feature_count(self) -> int:
        return len(FEATURE_IDS)

    @property
    def fixed_lane_count(self) -> int:
        return 1 + self.feature_count * (
            1
            + len(self.fusion["boost_values"])
            + len(self.fusion["gate_floors"])
        )

    @property
    def learned_scalar_fit_count(self) -> int:
        return self.feature_count * 4

    @property
    def multifeature_fit_count(self) -> int:
        return 4

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "source_video",
            "labels_tsv",
            "architecture_manifest",
            "output_dir",
            "preflight_dir",
        ):
            payload[field] = str(payload[field])
        return payload
