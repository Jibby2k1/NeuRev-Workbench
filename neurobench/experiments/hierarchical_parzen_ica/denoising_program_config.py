"""Configuration and exact design expansion for the broad denoising program."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any


FAMILY_IDS = (
    "skip_connected_parzen_ica",
    "per_component_parzen_ica",
    "multiscale_convolutional_ica",
    "bounded_ica_noise_subtraction",
    "noise_psd_wiener",
    "robust_low_rank_sparse",
    "nonnegative_factorization",
    "nonlocal_patch_denoising",
    "component_kalman",
    "undecimated_wavelet",
)


class DenoisingProgramConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DenoisingProgramConfig:
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
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "DenoisingProgramConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_video", "labels_tsv",
            "architecture_manifest", "output_dir", "preflight_dir", "frames",
            "input_lane", "shared_ica", "families", "stages", "evaluation",
            "visualization", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise DenoisingProgramConfigError(
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
            evaluation=dict(raw["evaluation"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise DenoisingProgramConfigError("invalid schema_version or experiment_id")
        if set(self.frames) != {
            "review_start_ui", "review_end_ui", "quiet_start_ui",
            "quiet_end_ui", "frame_period_ms",
        } or not (
            1 <= int(self.frames["review_start_ui"])
            == int(self.frames["quiet_start_ui"])
            <= int(self.frames["quiet_end_ui"])
            < int(self.frames["review_end_ui"])
        ):
            raise DenoisingProgramConfigError("invalid one-based inclusive frame contract")
        if set(self.input_lane) != {
            "id", "reference_half_life_seconds", "correction_fraction",
            "correction_clip_mad",
        } or self.input_lane["id"] != "reference_parzen_innovation":
            raise DenoisingProgramConfigError("invalid input lane")
        if set(self.shared_ica) != {
            "patch_sizes", "rank", "sample_count", "seed",
            "fastica_max_iterations", "fastica_tolerance",
            "wiener_lambda_z", "parzen",
        }:
            raise DenoisingProgramConfigError("invalid shared ICA fields")
        patch_sizes = [int(value) for value in self.shared_ica["patch_sizes"]]
        if (
            patch_sizes != sorted(set(patch_sizes))
            or any(value < 5 or value > 21 or value % 2 == 0 for value in patch_sizes)
            or 11 not in patch_sizes
            or not 2 <= int(self.shared_ica["rank"]) <= 32
        ):
            raise DenoisingProgramConfigError("invalid shared ICA geometry")
        if set(self.families) != set(FAMILY_IDS):
            raise DenoisingProgramConfigError("all ten denoising families are required")
        designs = self.designs()
        expected = {
            "skip_connected_parzen_ica": 5,
            "per_component_parzen_ica": 7,
            "multiscale_convolutional_ica": 7,
            "bounded_ica_noise_subtraction": 6,
            "noise_psd_wiener": 6,
            "robust_low_rank_sparse": 8,
            "nonnegative_factorization": 6,
            "nonlocal_patch_denoising": 6,
            "component_kalman": 9,
            "undecimated_wavelet": 9,
        }
        counts = {family: len(rows) for family, rows in designs.items()}
        if counts != expected:
            raise DenoisingProgramConfigError(
                f"frozen breadth counts changed: {counts} != {expected}"
            )
        if set(self.stages) != {
            "stage_a_crop_margin_px", "stage_a_top_per_family",
            "stage_b_top_per_family", "confirmation_seeds",
            "confirmation_holdout_bursts", "write_finalist_tiffs",
        } or not (
            16 <= int(self.stages["stage_a_crop_margin_px"]) <= 64
            and int(self.stages["stage_a_top_per_family"]) == 2
            and int(self.stages["stage_b_top_per_family"]) == 1
            and len(self.stages["confirmation_seeds"]) >= 3
            and self.stages["confirmation_holdout_bursts"] == [1, 2, 3, 4]
        ):
            raise DenoisingProgramConfigError("invalid staged selection contract")
        if set(self.evaluation) != {
            "roi_radius_px", "temporal_pool_tau", "nms_distance_px",
            "match_radius_px", "quiet_false_peaks_per_map",
            "fixed_candidates_per_burst", "synthetic_seed", "synthetic_frames",
            "synthetic_size", "synthetic_snr_multipliers",
            "minimum_peak_retention", "minimum_area_retention",
            "maximum_peak_frame_error", "minimum_synthetic_correlation",
        }:
            raise DenoisingProgramConfigError("invalid evaluation contract")
        if set(self.visualization) != {
            "compression", "positive_upper_percentile",
            "remainder_absolute_percentile", "sample_frame_stride",
            "sample_row_stride", "sample_column_stride",
        } or self.visualization["compression"] != "zlib":
            raise DenoisingProgramConfigError("invalid visualization contract")
        if set(self.resources) != {
            "device", "cpu_threads", "frame_batch_size",
            "max_ram_mib", "max_gpu_memory_mib", "min_free_disk_mib",
            "max_output_mib", "heartbeat_seconds",
        } or not (
            self.resources["device"] in {"cpu", "cuda"}
            and 1 <= int(self.resources["cpu_threads"]) <= 8
            and 1 <= int(self.resources["frame_batch_size"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 8192
        ):
            raise DenoisingProgramConfigError("invalid resource envelope")

    def designs(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for family in FAMILY_IDS:
            payload = self.families[family]
            if "design_points" in payload:
                rows = [dict(row) for row in payload["design_points"]]
            else:
                keys = sorted(payload)
                rows = [
                    dict(zip(keys, values))
                    for values in product(*(payload[key] for key in keys))
                ]
            result[family] = [
                {"family_id": family, "variant_index": index, **row}
                for index, row in enumerate(rows, start=1)
            ]
        return result

    @property
    def breadth_combination_count(self) -> int:
        return sum(len(rows) for rows in self.designs().values())

    @property
    def full_field_combination_count(self) -> int:
        return len(FAMILY_IDS) * int(self.stages["stage_a_top_per_family"])

    @property
    def finalist_count(self) -> int:
        return len(FAMILY_IDS) * int(self.stages["stage_b_top_per_family"])

    @property
    def confirmation_evaluation_count(self) -> int:
        return (
            self.finalist_count
            * len(self.stages["confirmation_seeds"])
            * len(self.stages["confirmation_holdout_bursts"])
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "source_video", "labels_tsv", "architecture_manifest",
            "output_dir", "preflight_dir",
        ):
            payload[field] = str(payload[field])
        return payload
