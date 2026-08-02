"""Strict manifest for the Principe-aligned patch-information experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


FAMILIES = (
    "renyi2_information_potential",
    "cs_quiet_divergence",
    "local_correntropy",
)


class PatchInformationConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PatchInformationConfig:
    schema_version: int
    experiment_id: str
    source_ranker_config: Path
    source_ranker_root: Path
    output_dir: Path
    preflight_dir: Path
    itl: dict[str, Any]
    screen: dict[str, Any]
    ranker: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PatchInformationConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_ranker_config",
            "source_ranker_root", "output_dir", "preflight_dir", "itl",
            "screen", "ranker", "visualization", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise PatchInformationConfigError(
                f"top-level fields differ; missing={sorted(fields-set(raw))}, "
                f"unknown={sorted(set(raw)-fields)}"
            )
        root = manifest.parent
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_ranker_config=(root / raw["source_ranker_config"]).resolve(),
            source_ranker_root=(root / raw["source_ranker_root"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            itl=dict(raw["itl"]), screen=dict(raw["screen"]),
            ranker=dict(raw["ranker"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise PatchInformationConfigError("invalid schema or experiment id")
        if set(self.itl) != {
            "bin_centers_z", "patch_sizes_px", "kernel_bandwidths_z",
            "families", "frame_batch_size",
        }:
            raise PatchInformationConfigError("invalid ITL fields")
        centers = [float(value) for value in self.itl["bin_centers_z"]]
        patches = [int(value) for value in self.itl["patch_sizes_px"]]
        bandwidths = [float(value) for value in self.itl["kernel_bandwidths_z"]]
        if (
            len(centers) != 13 or centers != sorted(set(centers))
            or patches != [7, 11, 15]
            or bandwidths != [0.5, 1.0, 2.0]
            or tuple(self.itl["families"]) != FAMILIES
            or not 1 <= int(self.itl["frame_batch_size"]) <= 16
        ):
            raise PatchInformationConfigError("frozen ITL grid differs")
        if set(self.screen) != {
            "carrier_boosts", "carrier_gate_floors", "budgets",
            "primary_budgets", "oracle_source_budgets",
        }:
            raise PatchInformationConfigError("invalid screen fields")
        if (
            self.screen["carrier_boosts"] != [0.1, 0.25, 0.5, 1.0]
            or self.screen["carrier_gate_floors"] != [0.5, 0.75, 0.9]
            or self.screen["budgets"] != [20, 40, 58, 80, 100]
            or self.screen["primary_budgets"] != [20, 40]
        ):
            raise PatchInformationConfigError("frozen screen grid differs")
        if set(self.ranker) != {
            "feature_sets", "learning_rates", "l2_values",
            "maximum_total_weights", "epochs",
        }:
            raise PatchInformationConfigError("invalid ranker fields")
        if tuple(self.ranker["feature_sets"]) != (
            "separation", "itl_anchor", "itl_all", "separation_itl_anchor"
        ):
            raise PatchInformationConfigError("frozen feature sets differ")
        if not (
            self.ranker["learning_rates"] == [0.003, 0.01, 0.03]
            and self.ranker["l2_values"] == [0.01, 0.1, 1.0]
            and self.ranker["maximum_total_weights"] == [0.25, 0.5]
            and int(self.ranker["epochs"]) == 300
        ):
            raise PatchInformationConfigError("frozen ranker grid differs")
        if set(self.visualization) != {"tiff_feature_count", "compression"}:
            raise PatchInformationConfigError("invalid visualization fields")
        if set(self.resources) != {
            "device", "cpu_threads", "max_ram_mib", "max_gpu_memory_mib",
            "min_free_disk_mib", "max_output_mib",
        }:
            raise PatchInformationConfigError("invalid resource fields")
        if (
            self.resources["device"] not in {"cpu", "cuda"}
            or not 1 <= int(self.resources["cpu_threads"]) <= 8
            or int(self.resources["max_ram_mib"]) < 4096
        ):
            raise PatchInformationConfigError("invalid resource bounds")

    @property
    def feature_count(self) -> int:
        return (
            len(self.itl["patch_sizes_px"])
            * len(self.itl["kernel_bandwidths_z"])
            * len(self.itl["families"])
        )

    @property
    def fixed_lane_count(self) -> int:
        return self.feature_count * (
            1 + len(self.screen["carrier_boosts"])
            + len(self.screen["carrier_gate_floors"])
        )

    @property
    def linear_config_count(self) -> int:
        return (
            len(self.ranker["feature_sets"])
            * len(self.ranker["learning_rates"])
            * len(self.ranker["l2_values"])
            * len(self.ranker["maximum_total_weights"])
        )

    @property
    def inner_fit_count(self) -> int:
        return self.linear_config_count * 4 * 3

    @property
    def outer_refit_count(self) -> int:
        return len(self.ranker["feature_sets"]) * 4

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_ranker_config", "source_ranker_root", "output_dir",
            "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
