"""Strict manifest for the multiscale Cauchy--Schwarz experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class MultiscaleInformationConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MultiscaleInformationConfig:
    schema_version: int
    experiment_id: str
    source_patch_config: Path
    source_patch_root: Path
    output_dir: Path
    preflight_dir: Path
    multiscale: dict[str, Any]
    fusions: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "MultiscaleInformationConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_patch_config",
            "source_patch_root", "output_dir", "preflight_dir", "multiscale",
            "fusions", "evaluation", "visualization", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise MultiscaleInformationConfigError(
                f"top-level fields differ; missing={sorted(fields-set(raw))}, "
                f"unknown={sorted(set(raw)-fields)}"
            )
        root = manifest.parent
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_patch_config=(root / raw["source_patch_config"]).resolve(),
            source_patch_root=(root / raw["source_patch_root"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            multiscale=dict(raw["multiscale"]),
            fusions=dict(raw["fusions"]),
            evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise MultiscaleInformationConfigError("invalid schema or experiment id")
        if set(self.multiscale) != {
            "patch_sizes_px", "kernel_bandwidths_z", "frame_batch_size",
            "center_surround_pairs",
        }:
            raise MultiscaleInformationConfigError("invalid multiscale fields")
        if (
            self.multiscale["patch_sizes_px"] != [5, 7, 9, 11, 13, 15]
            or self.multiscale["kernel_bandwidths_z"] != [0.5, 1.0]
            or self.multiscale["center_surround_pairs"]
            != [[5, 13], [7, 15], [9, 15]]
            or not 1 <= int(self.multiscale["frame_batch_size"]) <= 8
        ):
            raise MultiscaleInformationConfigError("frozen multiscale grid differs")
        if set(self.fusions) != {
            "softmax_temperatures", "contrast_pair", "contrast_authorities",
            "carrier_boosts",
        } or not (
            self.fusions["softmax_temperatures"] == [0.25, 0.5, 1.0]
            and self.fusions["contrast_pair"] == [7, 15]
            and self.fusions["contrast_authorities"] == [0.25, 0.5, 1.0]
            and self.fusions["carrier_boosts"] == [0.25, 0.5, 1.0]
        ):
            raise MultiscaleInformationConfigError("frozen fusion grid differs")
        if set(self.evaluation) != {
            "budgets", "primary_budgets", "oracle_source_budgets",
            "quota_carrier_fraction",
        } or not (
            self.evaluation["budgets"] == [20, 40, 58, 80, 100]
            and self.evaluation["primary_budgets"] == [20, 40]
            and self.evaluation["oracle_source_budgets"] == [20, 40, 58, 80, 100]
            and float(self.evaluation["quota_carrier_fraction"]) == 0.5
        ):
            raise MultiscaleInformationConfigError("frozen evaluation differs")
        if set(self.visualization) != {"tiff_feature_count", "compression"}:
            raise MultiscaleInformationConfigError("invalid visualization fields")
        if set(self.resources) != {
            "device", "cpu_threads", "max_ram_mib", "max_gpu_memory_mib",
            "min_free_disk_mib", "max_output_mib",
        } or not (
            self.resources["device"] == "cuda"
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 4096
        ):
            raise MultiscaleInformationConfigError("invalid resource bounds")

    @property
    def base_feature_count(self) -> int:
        return len(self.multiscale["patch_sizes_px"]) * len(
            self.multiscale["kernel_bandwidths_z"]
        )

    @property
    def fused_feature_count(self) -> int:
        scales = len(self.multiscale["patch_sizes_px"])
        bandwidths = len(self.multiscale["kernel_bandwidths_z"])
        return (
            bandwidths
            + bandwidths * len(self.fusions["softmax_temperatures"])
            + bandwidths * (scales - 1)
            + bandwidths * len(self.fusions["contrast_authorities"])
            + bandwidths * len(self.multiscale["center_surround_pairs"])
        )

    @property
    def feature_count(self) -> int:
        return self.base_feature_count + self.fused_feature_count

    @property
    def lane_count(self) -> int:
        return self.feature_count * (1 + len(self.fusions["carrier_boosts"]))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_patch_config", "source_patch_root", "output_dir",
            "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
