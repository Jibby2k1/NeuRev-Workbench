"""Strict manifest for the Spon acquisition/morphology/structure audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class ScientificAuditConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ScientificAuditConfig:
    schema_version: int
    experiment_id: str
    source_ranker_config: Path
    source_multiscale_root: Path
    source_video: Path
    carrier_path: Path
    labels_tsv: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    acquisition: dict[str, Any]
    noise: dict[str, Any]
    morphology: dict[str, Any]
    radial_information: dict[str, Any]
    propagation: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ScientificAuditConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_ranker_config",
            "source_multiscale_root", "source_video", "carrier_path",
            "labels_tsv", "output_dir", "preflight_dir", "frames",
            "acquisition", "noise", "morphology", "radial_information",
            "propagation", "evaluation", "visualization", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ScientificAuditConfigError(
                f"top-level fields differ; missing={sorted(fields-set(raw))}, "
                f"unknown={sorted(set(raw)-fields)}"
            )
        root = manifest.parent
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_ranker_config=(root / raw["source_ranker_config"]).resolve(),
            source_multiscale_root=(root / raw["source_multiscale_root"]).resolve(),
            source_video=(root / raw["source_video"]).resolve(),
            carrier_path=(root / raw["carrier_path"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            frames=dict(raw["frames"]), acquisition=dict(raw["acquisition"]),
            noise=dict(raw["noise"]), morphology=dict(raw["morphology"]),
            radial_information=dict(raw["radial_information"]),
            propagation=dict(raw["propagation"]),
            evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise ScientificAuditConfigError("invalid schema or experiment id")
        if self.frames != {
            "review_start_ui": 1800, "review_end_ui": 2359, "quiet_count": 100
        }:
            raise ScientificAuditConfigError("frame contract differs")
        if set(self.acquisition) != {
            "split_x_px", "saturation_value", "frame_period_ms",
            "frame_period_provenance",
        } or not (
            int(self.acquisition["split_x_px"]) == 286
            and int(self.acquisition["saturation_value"]) == 4095
            and float(self.acquisition["frame_period_ms"]) == 20.0
        ):
            raise ScientificAuditConfigError("acquisition contract differs")
        if self.noise != {
            "intensity_bins": 24, "drift_stride": 1,
            "spatial_diagnostic_sigma_px": 15.0,
        }:
            raise ScientificAuditConfigError("noise audit differs")
        if set(self.morphology) != {
            "template_size_px", "radii_px", "z_offsets_fraction",
            "membrane_thickness_px", "psf_sigmas_px", "crowd_sigma_px",
        } or not (
            self.morphology["template_size_px"] == 31
            and self.morphology["radii_px"] == [3.0, 4.5, 6.0]
            and self.morphology["z_offsets_fraction"] == [0.0, 0.5, 0.8, 0.93]
            and self.morphology["psf_sigmas_px"] == [0.75, 1.25]
        ):
            raise ScientificAuditConfigError("morphology grid differs")
        if self.radial_information != {
            "bin_centers_z": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
            "kernel_bandwidth_z": 0.5, "center_radius_px": 2.0,
            "shell_radius_px": 4.0, "outer_radius_px": 7.0,
            "context_authority": 0.5, "frame_batch_size": 4,
        }:
            raise ScientificAuditConfigError("radial-information grid differs")
        if self.propagation != {
            "coherence_windows_frames": [7, 15, 31],
            "lag_window_pairs": [[1, 15], [2, 15], [4, 31]],
            "spatial_sigma_px": 2.0,
        }:
            raise ScientificAuditConfigError("propagation grid differs")
        if set(self.evaluation) != {
            "budgets", "primary_budgets", "carrier_boosts",
        } or not (
            self.evaluation["budgets"] == [20, 40, 58, 80, 100]
            and self.evaluation["primary_budgets"] == [20, 40]
            and self.evaluation["carrier_boosts"] == [0.25, 0.5, 1.0]
        ):
            raise ScientificAuditConfigError("evaluation differs")
        if set(self.visualization) != {
            "compression", "upper_percentile", "feature_video_ids",
        }:
            raise ScientificAuditConfigError("visualization fields differ")
        if set(self.resources) != {
            "device", "cpu_threads", "max_ram_mib", "max_gpu_memory_mib",
            "min_free_disk_mib", "max_output_mib",
        } or not (
            self.resources["device"] == "cuda"
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 8192
        ):
            raise ScientificAuditConfigError("resource bounds differ")
        expected = {
            "radial_cs_center", "radial_cs_shell", "radial_cs_morph_max",
            "coherence_w15", "propagation_lag2_w15",
        }
        if set(self.visualization["feature_video_ids"]) != expected:
            raise ScientificAuditConfigError("representative video set differs")

    @property
    def feature_count(self) -> int:
        return 6 + 3 + 1 + 3 + 3

    @property
    def lane_count_per_field(self) -> int:
        return self.feature_count * (1 + len(self.evaluation["carrier_boosts"]))

    @property
    def evaluated_lane_count(self) -> int:
        return self.lane_count_per_field * 3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_ranker_config", "source_multiscale_root", "source_video",
            "carrier_path", "labels_tsv", "output_dir", "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
