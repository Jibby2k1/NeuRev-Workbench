"""Strict v1 manifest for dependent multiscale demixing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class DependentMultiscaleConfigError(ValueError):
    pass


_SECTIONS = {
    "input": {
        "observation_npy", "display_observation_npy", "scientific_carrier_npy",
        "provider_local_pca_metadata", "labels_tsv", "labels_role",
        "input_normalization_state",
    },
    "frames": {"review_start_ui", "review_end_ui", "quiet_count"},
    "geometry": {"analysis_patch_px", "stride_px", "overlap_window", "border_mode"},
    "views": {"supports_px", "operator_kind", "normalization_kind"},
    "factorization": {"source", "rank_rule", "rank_min", "rank_max", "fixed_rank_reference"},
    "model": {
        "groups", "allow_neural_within_group_dependence",
        "allow_cross_scale_residual_dependence", "population_drive_ablation",
    },
    "information": {
        "group_estimator", "residual_estimator", "matrix_kernel",
        "bandwidth_rule", "neural_background_weight", "neural_artifact_weight",
        "quiet_residual_weight",
    },
    "regularization": {
        "overlap_weight", "neural_compactness_weight", "neural_annular_ablation",
        "background_smoothness_weight", "artifact_motion_weight",
    },
    "optimization": {
        "mode", "max_iterations", "tolerance", "seeds", "gradient_clip", "fallback",
    },
    "evaluation": {
        "budgets", "primary_budgets", "match_radius_px", "write_label_projection",
    },
    "artifacts": {
        "write_dense_views", "write_dense_channels", "write_patch_factors", "write_tiffs",
    },
    "resources": {
        "device", "max_threads", "max_ram_mib", "max_gpu_memory_mib",
        "max_output_mib", "frame_batch_size",
    },
}


@dataclass(frozen=True)
class DependentMultiscaleConfig:
    schema_version: int
    experiment_id: str
    input: dict[str, Any]
    frames: dict[str, Any]
    geometry: dict[str, Any]
    views: dict[str, Any]
    factorization: dict[str, Any]
    model: dict[str, Any]
    information: dict[str, Any]
    regularization: dict[str, Any]
    optimization: dict[str, Any]
    evaluation: dict[str, Any]
    artifacts: dict[str, Any]
    resources: dict[str, Any]
    preflight_dir: Path
    output_dir: Path

    @classmethod
    def load(cls, path: str | Path) -> "DependentMultiscaleConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        top = {
            "schema_version", "experiment_id", *_SECTIONS,
            "preflight_dir", "output_dir",
        }
        if not isinstance(raw, dict) or set(raw) != top:
            raise DependentMultiscaleConfigError(
                f"top-level fields differ; missing={sorted(top-set(raw))}, "
                f"unknown={sorted(set(raw)-top)}"
            )
        for name, fields in _SECTIONS.items():
            if not isinstance(raw[name], dict) or set(raw[name]) != fields:
                actual = set(raw[name]) if isinstance(raw[name], dict) else set()
                raise DependentMultiscaleConfigError(
                    f"{name} fields differ; missing={sorted(fields-actual)}, "
                    f"unknown={sorted(actual-fields)}"
                )
        root = manifest.parent
        input_values = dict(raw["input"])
        for key in (
            "observation_npy", "display_observation_npy", "scientific_carrier_npy",
            "provider_local_pca_metadata", "labels_tsv",
        ):
            if input_values[key] is not None:
                input_values[key] = str((root / input_values[key]).resolve())
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            input=input_values,
            frames=dict(raw["frames"]),
            geometry=dict(raw["geometry"]),
            views=dict(raw["views"]),
            factorization=dict(raw["factorization"]),
            model=dict(raw["model"]),
            information=dict(raw["information"]),
            regularization=dict(raw["regularization"]),
            optimization=dict(raw["optimization"]),
            evaluation=dict(raw["evaluation"]),
            artifacts=dict(raw["artifacts"]),
            resources=dict(raw["resources"]),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise DependentMultiscaleConfigError("schema_version must be 1 and experiment_id non-empty")
        start = int(self.frames["review_start_ui"])
        stop = int(self.frames["review_end_ui"])
        quiet = int(self.frames["quiet_count"])
        if start < 1 or stop < start or not 2 <= quiet < stop - start + 1:
            raise DependentMultiscaleConfigError("invalid one-based inclusive frame contract")
        if self.input["labels_role"] != "evaluation_only":
            raise DependentMultiscaleConfigError("labels_role must be evaluation_only")
        if self.input["input_normalization_state"] not in {"raw", "calibrated", "quiet_standardized"}:
            raise DependentMultiscaleConfigError("input_normalization_state must be explicit")
        geometry = self.geometry
        if (
            int(geometry["analysis_patch_px"]) != 31
            or int(geometry["stride_px"]) != 8
            or geometry["overlap_window"] != "hann_floor_0p1"
            or geometry["border_mode"] != "reflect"
        ):
            raise DependentMultiscaleConfigError("frozen v1 geometry differs")
        if self.views["supports_px"] != [5, 7, 15]:
            raise DependentMultiscaleConfigError("frozen v1 supports must be [5,7,15]")
        if self.views["operator_kind"] not in {"quiet_normalized_local_support", "normalized_box_support"}:
            raise DependentMultiscaleConfigError("unsupported scale-view operator")
        if self.views["normalization_kind"] not in {"quiet_robust", "none"}:
            raise DependentMultiscaleConfigError("unsupported view normalization")
        if self.input["input_normalization_state"] == "quiet_standardized" and self.views["normalization_kind"] != "none":
            raise DependentMultiscaleConfigError("quiet-standardized inputs cannot be normalized twice")
        factor = self.factorization
        if factor["source"] not in {"neurev_local_pca", "provider_local_pca"}:
            raise DependentMultiscaleConfigError("unsupported factorization source")
        if factor["rank_rule"] not in {"quiet_null", "fixed_small_rank", "provider_metadata"}:
            raise DependentMultiscaleConfigError("unsupported label-free rank rule")
        rank_min, rank_max, reference = map(int, (factor["rank_min"], factor["rank_max"], factor["fixed_rank_reference"]))
        if not 1 <= rank_min <= reference <= rank_max <= 8:
            raise DependentMultiscaleConfigError("rank bounds must satisfy 1 <= min <= reference <= max <= 8")
        if factor["source"] == "provider_local_pca" and not self.input["provider_local_pca_metadata"]:
            raise DependentMultiscaleConfigError("provider factorization requires explicit metadata")
        if self.model["groups"] != ["neural", "background", "artifact"]:
            raise DependentMultiscaleConfigError("production v1 requires neural/background/artifact groups")
        if self.optimization["seeds"] != [7, 13, 19] or self.optimization["fallback"] != "orthogonal_shared_private":
            raise DependentMultiscaleConfigError("frozen seeds or fallback differ")
        if self.evaluation["budgets"] != [20, 40, 58, 80, 100] or self.evaluation["primary_budgets"] != [20, 40]:
            raise DependentMultiscaleConfigError("frozen evaluation budgets differ")
        resources = self.resources
        if resources["device"] != "cpu" or not 1 <= int(resources["max_threads"]) <= 8:
            raise DependentMultiscaleConfigError("initial wave is bounded CPU with 1-8 threads")
        if min(int(resources[key]) for key in ("max_ram_mib", "max_output_mib", "frame_batch_size")) <= 0:
            raise DependentMultiscaleConfigError("resource bounds must be positive")
        if self.preflight_dir == self.output_dir:
            raise DependentMultiscaleConfigError("preflight_dir and output_dir must differ")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["preflight_dir"] = str(self.preflight_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload
