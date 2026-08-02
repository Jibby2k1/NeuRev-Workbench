"""Strict configuration for the conclusive source-separation batch."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ConclusiveBatchConfigError(ValueError):
    """Raised when the conclusive-batch manifest differs from its contract."""


@dataclass(frozen=True)
class ConclusiveBatchConfig:
    schema_version: int
    experiment_id: str
    output_root: Path
    scientific_config_path: Path
    source_video: Path
    source_tiff: Path
    labels_tsv: Path
    caiman_python: Path
    frames: dict[str, Any]
    methods: tuple[dict[str, Any], ...]
    design: dict[str, Any]
    gates: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ConclusiveBatchConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        required = {
            "schema_version", "experiment_id", "output_root",
            "scientific_config", "source_video", "source_tiff", "labels_tsv",
            "caiman_python", "frames", "methods", "design", "gates",
            "resources",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            keys = set(raw) if isinstance(raw, dict) else set()
            raise ConclusiveBatchConfigError(
                f"top-level fields differ; missing={sorted(required-keys)}, "
                f"unknown={sorted(keys-required)}"
            )
        root = manifest.parent

        def resolve(value: Any) -> Path:
            return (root / str(value)).resolve()

        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            output_root=resolve(raw["output_root"]),
            scientific_config_path=resolve(raw["scientific_config"]),
            source_video=resolve(raw["source_video"]),
            source_tiff=resolve(raw["source_tiff"]),
            labels_tsv=resolve(raw["labels_tsv"]),
            caiman_python=Path(str(raw["caiman_python"])).expanduser().resolve(),
            frames=dict(raw["frames"]),
            methods=tuple(dict(value) for value in raw["methods"]),
            design=dict(raw["design"]),
            gates=dict(raw["gates"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise ConclusiveBatchConfigError("invalid schema or experiment id")
        if set(self.frames) != {
            "review_start_ui", "review_end_ui", "quiet_start_ui", "quiet_end_ui",
            "frame_period_ms",
        }:
            raise ConclusiveBatchConfigError("frame fields differ")
        review_start = int(self.frames["review_start_ui"])
        review_end = int(self.frames["review_end_ui"])
        quiet_start = int(self.frames["quiet_start_ui"])
        quiet_end = int(self.frames["quiet_end_ui"])
        if not 1 <= review_start <= quiet_start <= quiet_end <= review_end:
            raise ConclusiveBatchConfigError("invalid review/quiet frame contract")
        if float(self.frames["frame_period_ms"]) != 20.0:
            raise ConclusiveBatchConfigError("frame period must remain 20 ms")
        required_methods = {
            "raw_direct_reference", "amplitude_pca_reference", "multilag_sobi",
            "full_window_spatial_fastica_reference",
            "dense_patch_fastica_wiener_reference",
            "kernel_hsic_pairwise_rotation", "knn_mi_pairwise_rotation",
            "group_energy_hsic_isa", "spatial_noisy_parzen_infomax",
            "multistart_consensus", "caiman_cnmf", "caiman_cnmfe",
        }
        if not self.methods or any(set(item) != {"method_id", "enabled", "track", "configurations"} for item in self.methods):
            raise ConclusiveBatchConfigError("method entries differ")
        method_ids = [str(item["method_id"]) for item in self.methods]
        if set(method_ids) != required_methods or len(method_ids) != len(set(method_ids)):
            raise ConclusiveBatchConfigError("method panel differs")
        if any(item["track"] not in {"controlled_input", "native_best", "anchor"} for item in self.methods):
            raise ConclusiveBatchConfigError("invalid comparison track")
        if any(bool(item["enabled"]) and not isinstance(item["configurations"], list) for item in self.methods):
            raise ConclusiveBatchConfigError("enabled method configurations must be lists")
        if set(self.design) != {
            "development_fixture_count", "confirmation_fixture_count",
            "semi_synthetic_fixture_count", "confidence_perturbations",
            "fixed_candidate_budgets", "manual_review_rows",
        }:
            raise ConclusiveBatchConfigError("design fields differ")
        if not 24 <= int(self.design["development_fixture_count"]) <= 128:
            raise ConclusiveBatchConfigError("development fixture count is unbounded")
        if not 96 <= int(self.design["confirmation_fixture_count"]) <= 512:
            raise ConclusiveBatchConfigError("confirmation fixture count is unbounded")
        if not 24 <= int(self.design["semi_synthetic_fixture_count"]) <= 256:
            raise ConclusiveBatchConfigError("semi-synthetic fixture count is unbounded")
        if list(map(int, self.design["fixed_candidate_budgets"])) != [10, 20, 40, 58, 80, 100]:
            raise ConclusiveBatchConfigError("candidate budgets differ")
        if set(self.gates) != {
            "closure_tolerance", "maximum_false_resolution_count",
            "minimum_identifiable_coverage", "minimum_converged_fraction",
            "equivalence_margin", "minimum_peak_retention",
            "minimum_area_retention", "minimum_waveform_correlation",
            "maximum_timing_error_frames",
        }:
            raise ConclusiveBatchConfigError("gate fields differ")
        if int(self.gates["maximum_false_resolution_count"]) != 0:
            raise ConclusiveBatchConfigError("false-resolution count must remain zero")
        if not 0 < float(self.gates["closure_tolerance"]) <= 1e-3:
            raise ConclusiveBatchConfigError("invalid closure tolerance")
        if not 0.5 <= float(self.gates["minimum_identifiable_coverage"]) <= 1:
            raise ConclusiveBatchConfigError("invalid identifiable coverage")
        if not 0.9 <= float(self.gates["minimum_converged_fraction"]) <= 1:
            raise ConclusiveBatchConfigError("invalid convergence gate")
        resource_fields = {
            "general_cpu_workers", "maximum_caiman_processes", "worker_threads",
            "gpu_device", "gpu_allocation_cap_mib", "minimum_free_gpu_mib",
            "rss_soft_cap_mib", "rss_hard_stop_mib", "minimum_free_disk_mib",
            "maximum_output_mib", "heartbeat_seconds", "gpu_warning_c",
            "gpu_stop_c",
        }
        if set(self.resources) != resource_fields:
            raise ConclusiveBatchConfigError("resource fields differ")
        if not 1 <= int(self.resources["general_cpu_workers"]) <= 8:
            raise ConclusiveBatchConfigError("general CPU worker count outside [1,8]")
        if not 1 <= int(self.resources["maximum_caiman_processes"]) <= 12:
            raise ConclusiveBatchConfigError("CaImAn process count outside [1,12]")
        if int(self.resources["worker_threads"]) != 1:
            raise ConclusiveBatchConfigError("workers must freeze nested threads at one")
        if int(self.resources["rss_soft_cap_mib"]) >= int(self.resources["rss_hard_stop_mib"]):
            raise ConclusiveBatchConfigError("RSS soft cap must be below hard stop")
        if int(self.resources["gpu_warning_c"]) >= int(self.resources["gpu_stop_c"]):
            raise ConclusiveBatchConfigError("GPU warning temperature must be below stop")

    def enabled_configuration_count(self) -> int:
        return sum(
            len(item["configurations"])
            for item in self.methods
            if bool(item["enabled"])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "output_root": str(self.output_root),
            "scientific_config_path": str(self.scientific_config_path),
            "source_video": str(self.source_video),
            "source_tiff": str(self.source_tiff),
            "labels_tsv": str(self.labels_tsv),
            "caiman_python": str(self.caiman_python),
            "frames": self.frames,
            "methods": list(self.methods),
            "design": self.design,
            "gates": self.gates,
            "resources": self.resources,
        }
