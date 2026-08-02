"""Validated v3 design for bounded innovation denoising and Pareto mixtures."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


FAMILY_IDS = (
    "local_psd_wiener",
    "morphology_conditioned",
    "selected_nmf",
    "asymmetric_component_dynamics",
    "tempered_parzen_posterior",
    "graph_spatial_diffusion",
    "cross_scale_consensus",
    "blindspot_self_supervised",
)


class InnovationDenoisingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class InnovationDenoisingConfig:
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
    families: dict[str, Any]
    stages: dict[str, Any]
    mixture: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "InnovationDenoisingConfig":
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
            "families",
            "stages",
            "mixture",
            "evaluation",
            "visualization",
            "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise InnovationDenoisingConfigError(
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
            families=dict(raw["families"]),
            stages=dict(raw["stages"]),
            mixture=dict(raw["mixture"]),
            evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise InnovationDenoisingConfigError(
                "invalid schema_version or experiment_id"
            )
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
            raise InnovationDenoisingConfigError(
                "invalid one-based inclusive frame contract"
            )
        if set(self.input_lane) != {
            "id",
            "reference_half_life_seconds",
            "correction_fraction",
            "correction_clip_mad",
        } or self.input_lane["id"] != "reference_parzen_innovation":
            raise InnovationDenoisingConfigError("invalid input lane")
        if set(self.shared_ica) != {
            "patch_size",
            "rank",
            "sample_count",
            "seed",
            "fastica_max_iterations",
            "fastica_tolerance",
            "wiener_lambda_z",
            "parzen",
        } or not (
            5 <= int(self.shared_ica["patch_size"]) <= 21
            and int(self.shared_ica["patch_size"]) % 2 == 1
            and 2 <= int(self.shared_ica["rank"]) <= 32
        ):
            raise InnovationDenoisingConfigError("invalid shared ICA contract")
        if set(self.families) != set(FAMILY_IDS):
            raise InnovationDenoisingConfigError(
                "all eight innovation families are required"
            )
        counts = {
            family: len(payload.get("design_points", []))
            for family, payload in self.families.items()
        }
        if counts != {family: 12 for family in FAMILY_IDS}:
            raise InnovationDenoisingConfigError(
                f"frozen breadth counts changed: {counts}"
            )
        if set(self.stages) != {
            "stage_a_crop_margin_px",
            "stage_a_top_per_family",
            "stage_b_top_per_family",
            "write_family_finalist_tiffs",
            "write_top_mixture_tiffs",
            "confirmation_seeds",
            "confirmation_top_count",
        } or not (
            16 <= int(self.stages["stage_a_crop_margin_px"]) <= 64
            and int(self.stages["stage_a_top_per_family"]) == 2
            and int(self.stages["stage_b_top_per_family"]) == 1
            and int(self.stages["write_top_mixture_tiffs"]) == 2
            and len(self.stages["confirmation_seeds"]) == 3
            and int(self.stages["confirmation_top_count"]) == 2
        ):
            raise InnovationDenoisingConfigError("invalid staged contract")
        if set(self.mixture) != {
            "pareto_source_count",
            "pair_weight",
            "all_source_weights",
            "correction_limit_z",
            "maximize_objectives",
            "minimize_objectives",
        } or not (
            int(self.mixture["pareto_source_count"]) == 4
            and 0 < float(self.mixture["pair_weight"]) <= 0.5
            and len(self.mixture["all_source_weights"]) == 2
            and all(
                0 < float(value) <= 0.25
                for value in self.mixture["all_source_weights"]
            )
        ):
            raise InnovationDenoisingConfigError("invalid Pareto mixture contract")
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
            "minimum_peak_retention",
            "minimum_area_retention",
            "maximum_peak_frame_error",
            "minimum_synthetic_correlation",
            "minimum_fixed_budget_recall_gain",
            "maximum_candidate_multiplier",
        }:
            raise InnovationDenoisingConfigError("invalid evaluation contract")
        if set(self.visualization) != {
            "compression",
            "positive_upper_percentile",
            "remainder_absolute_percentile",
            "sample_frame_stride",
            "sample_row_stride",
            "sample_column_stride",
        } or self.visualization["compression"] != "zlib":
            raise InnovationDenoisingConfigError("invalid visualization contract")
        if set(self.resources) != {
            "device",
            "cpu_threads",
            "frame_batch_size",
            "max_ram_mib",
            "max_gpu_memory_mib",
            "min_free_disk_mib",
            "max_output_mib",
            "heartbeat_seconds",
        } or not (
            self.resources["device"] in {"cpu", "cuda"}
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and 1 <= int(self.resources["frame_batch_size"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 8192
        ):
            raise InnovationDenoisingConfigError("invalid resource envelope")

    def designs(self) -> dict[str, list[dict[str, Any]]]:
        return {
            family: [
                {"family_id": family, "variant_index": index, **dict(row)}
                for index, row in enumerate(
                    self.families[family]["design_points"], start=1
                )
            ]
            for family in FAMILY_IDS
        }

    @property
    def breadth_combination_count(self) -> int:
        return sum(len(rows) for rows in self.designs().values())

    @property
    def full_field_combination_count(self) -> int:
        return len(FAMILY_IDS) * int(self.stages["stage_a_top_per_family"])

    @property
    def family_finalist_count(self) -> int:
        return len(FAMILY_IDS) * int(self.stages["stage_b_top_per_family"])

    @property
    def mixture_combination_count(self) -> int:
        source_count = int(self.mixture["pareto_source_count"])
        pairs = source_count * (source_count - 1) // 2
        return pairs + len(self.mixture["all_source_weights"])

    @property
    def tiff_finalist_count(self) -> int:
        return self.family_finalist_count + int(
            self.stages["write_top_mixture_tiffs"]
        )


    @property
    def maximum_confirmation_refit_count(self) -> int:
        return int(self.stages["confirmation_top_count"]) * len(
            self.stages["confirmation_seeds"]
        )

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
