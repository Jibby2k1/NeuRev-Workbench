"""Strict manifest for the bounded source-separation benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class InformationSeparationConfigError(ValueError):
    """Raised when a manifest differs from the versioned contract."""


CASES = (
    "isolated", "overlap", "synchronous", "correlated", "fast_onset",
    "slow_plateau", "similar_persistence", "illumination_drift",
    "motion_edge", "saturation", "heteroscedastic_noise", "pure_noise",
    "unresolved",
)


@dataclass(frozen=True)
class InformationSeparationConfig:
    schema_version: int
    experiment_id: str
    output_dir: Path
    source_video: Path
    generated: dict[str, Any]
    semi_synthetic: dict[str, Any]
    methods: dict[str, Any]
    selection: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "InformationSeparationConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        required = {
            "schema_version", "experiment_id", "output_dir", "source_video",
            "generated", "semi_synthetic", "methods", "selection", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            keys = set(raw) if isinstance(raw, dict) else set()
            raise InformationSeparationConfigError(
                f"top-level fields differ; missing={sorted(required-keys)}, "
                f"unknown={sorted(keys-required)}"
            )
        root = manifest.parent
        config = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            output_dir=(root / str(raw["output_dir"])).resolve(),
            source_video=(root / str(raw["source_video"])).resolve(),
            generated=dict(raw["generated"]),
            semi_synthetic=dict(raw["semi_synthetic"]),
            methods={key: dict(value) for key, value in raw["methods"].items()},
            selection=dict(raw["selection"]),
            resources=dict(raw["resources"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise InformationSeparationConfigError("invalid schema or experiment id")
        generated_fields = {
            "case_ids", "seeds", "frame_count", "height_px", "width_px",
            "snr_levels", "frame_period_ms",
        }
        if set(self.generated) != generated_fields:
            raise InformationSeparationConfigError("generated fields differ")
        case_ids = tuple(map(str, self.generated["case_ids"]))
        if len(case_ids) != len(set(case_ids)) or not case_ids or not set(case_ids).issubset(CASES):
            raise InformationSeparationConfigError("generated cases are invalid")
        seeds = [int(value) for value in self.generated["seeds"]]
        if len(seeds) != len(set(seeds)) or not seeds:
            raise InformationSeparationConfigError("generated seeds must be unique")
        if not 96 <= int(self.generated["frame_count"]) <= 4096:
            raise InformationSeparationConfigError("generated frame count outside [96,4096]")
        if not 8 <= int(self.generated["height_px"]) <= 64 or not 8 <= int(self.generated["width_px"]) <= 64:
            raise InformationSeparationConfigError("generated spatial shape outside [8,64]")
        snr = [float(value) for value in self.generated["snr_levels"]]
        if not snr or any(value <= 0 for value in snr):
            raise InformationSeparationConfigError("SNR levels must be positive")
        if float(self.generated["frame_period_ms"]) != 20.0:
            raise InformationSeparationConfigError("frame period must remain 20 ms")
        semi_fields = {
            "enabled", "quiet_start_ui", "quiet_end_ui", "crop_size_px",
            "crop_origins_xy", "injection_amplitudes",
        }
        if set(self.semi_synthetic) != semi_fields:
            raise InformationSeparationConfigError("semi-synthetic fields differ")
        if not 1 <= int(self.semi_synthetic["quiet_start_ui"]) <= int(self.semi_synthetic["quiet_end_ui"]):
            raise InformationSeparationConfigError("invalid quiet interval")
        crop = int(self.semi_synthetic["crop_size_px"])
        if not 8 <= crop <= 64:
            raise InformationSeparationConfigError("crop size outside [8,64]")
        origins = self.semi_synthetic["crop_origins_xy"]
        if not isinstance(origins, list) or any(
            not isinstance(value, list) or len(value) != 2 or min(map(int, value)) < 0
            for value in origins
        ):
            raise InformationSeparationConfigError("crop origins must be [x,y] pairs")
        if any(float(value) <= 0 for value in self.semi_synthetic["injection_amplitudes"]):
            raise InformationSeparationConfigError("injection amplitudes must be positive")
        method_ids = {
            "pca_reference", "multilag_sobi", "kernel_hsic_pairwise_rotation",
            "knn_mi_pairwise_rotation", "caiman_cnmf_reference_adapter",
            "group_energy_isa", "spatial_noisy_parzen_infomax",
        }
        if set(self.methods) != method_ids:
            raise InformationSeparationConfigError("method panel differs")
        common = {"enabled", "ranks"}
        if set(self.methods["pca_reference"]) != common:
            raise InformationSeparationConfigError("PCA fields differ")
        if set(self.methods["multilag_sobi"]) != common | {"lag_sets", "covariance_shrinkages"}:
            raise InformationSeparationConfigError("SOBI fields differ")
        if set(self.methods["kernel_hsic_pairwise_rotation"]) != common | {
            "bandwidth_scales", "angle_step_degrees", "max_sweeps", "max_fit_samples"
        }:
            raise InformationSeparationConfigError("HSIC fields differ")
        if set(self.methods["knn_mi_pairwise_rotation"]) != common | {
            "neighbors", "angle_step_degrees", "max_sweeps", "max_fit_samples"
        }:
            raise InformationSeparationConfigError("MI fields differ")
        adapter_fields = {"enabled", "backend", "required", "expected_version"}
        if set(self.methods["caiman_cnmf_reference_adapter"]) != adapter_fields:
            raise InformationSeparationConfigError("CNMF adapter fields differ")
        gated_fields = {"enabled", "status"}
        for method_id in ("group_energy_isa", "spatial_noisy_parzen_infomax"):
            if set(self.methods[method_id]) != gated_fields:
                raise InformationSeparationConfigError(f"{method_id} fields differ")
            if self.methods[method_id]["enabled"]:
                raise InformationSeparationConfigError(f"{method_id} is gated in schema v1")
        for method_id in (
            "pca_reference", "multilag_sobi", "kernel_hsic_pairwise_rotation",
            "knn_mi_pairwise_rotation",
        ):
            ranks = [int(value) for value in self.methods[method_id]["ranks"]]
            if not ranks or any(not 2 <= value <= 16 for value in ranks):
                raise InformationSeparationConfigError(f"invalid ranks for {method_id}")
        selection_fields = {
            "closure_tolerance", "minimum_peak_retention", "minimum_area_retention",
            "minimum_waveform_correlation", "maximum_timing_error_frames",
            "equivalence_margin", "require_correct_unresolved",
        }
        if set(self.selection) != selection_fields:
            raise InformationSeparationConfigError("selection fields differ")
        if not 0 < float(self.selection["closure_tolerance"]) <= 1e-3:
            raise InformationSeparationConfigError("closure tolerance is invalid")
        for field in (
            "minimum_peak_retention", "minimum_area_retention",
            "minimum_waveform_correlation",
        ):
            if not 0 < float(self.selection[field]) <= 1:
                raise InformationSeparationConfigError(f"invalid {field}")
        resource_fields = {
            "cpu_threads", "max_ram_mib", "min_free_disk_mib", "max_output_mib"
        }
        if set(self.resources) != resource_fields:
            raise InformationSeparationConfigError("resource fields differ")
        if not 1 <= int(self.resources["cpu_threads"]) <= 8:
            raise InformationSeparationConfigError("CPU thread count outside [1,8]")

    def generated_fixture_count(self) -> int:
        return (
            len(self.generated["case_ids"])
            * len(self.generated["seeds"])
            * len(self.generated["snr_levels"])
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["source_video"] = str(self.source_video)
        return payload
