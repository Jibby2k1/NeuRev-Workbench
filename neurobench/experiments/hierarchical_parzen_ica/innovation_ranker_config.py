"""Frozen configuration for nested activity-proposal innovation experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


EXISTING_FEATURE_IDS = (
    "carrier_signed",
    "local_psd_signal",
    "asymmetric_state",
    "spatial_coherence",
    "cross_scale_rank",
    "cross_scale_recall",
    "cfar_score",
    "cfar_background",
    "cfar_noise",
    "derivative_positive_lag1",
    "derivative_negative_lag1",
    "persistence_activity_gate",
    "persistent_artifact_score",
)

GENERATED_FEATURE_IDS = (
    "signed_power_1p25",
    "signed_power_1p5",
    "signed_power_2",
    "shot_whitened_rho0p25",
    "shot_whitened_rho0p5",
    "shot_whitened_rho1",
    "onset_dominance",
    "derivative_energy",
    "artifact_attenuated_0p5",
    "artifact_attenuated_1",
    "cut_center_sigma1p5",
    "cut_center_sigma2p5",
    "cut_center_sigma3p5",
    "cut_ring_r3p0_t0p75",
    "cut_ring_r3p0_t1p25",
    "cut_ring_r4p5_t0p75",
    "cut_ring_r4p5_t1p25",
    "cut_ring_r6p0_t0p75",
    "cut_ring_r6p0_t1p25",
    "cut_crowd_context",
    "structure_context",
)

FEATURE_IDS = EXISTING_FEATURE_IDS + GENERATED_FEATURE_IDS

PROPOSAL_SOURCE_IDS = (
    "carrier_signed",
    "local_psd_signal",
    "asymmetric_state",
    "spatial_coherence",
    "cross_scale_rank",
    "cross_scale_recall",
    "cfar_score",
    "signed_power_1p25",
    "signed_power_1p5",
    "signed_power_2",
    "shot_whitened_rho0p25",
    "shot_whitened_rho0p5",
    "shot_whitened_rho1",
    "cut_center_sigma1p5",
    "cut_center_sigma2p5",
    "cut_center_sigma3p5",
    "cut_ring_r3p0_t0p75",
    "cut_ring_r3p0_t1p25",
    "cut_ring_r4p5_t0p75",
    "cut_ring_r4p5_t1p25",
    "cut_ring_r6p0_t0p75",
    "cut_ring_r6p0_t1p25",
)

FEATURE_SETS = {
    "separation": (
        "carrier_signed",
        "local_psd_signal",
        "asymmetric_state",
        "spatial_coherence",
        "cross_scale_rank",
        "cross_scale_recall",
        "cfar_score",
    ),
    "temporal": (
        "carrier_signed",
        "local_psd_signal",
        "asymmetric_state",
        "spatial_coherence",
        "cross_scale_rank",
        "cross_scale_recall",
        "signed_power_1p25",
        "signed_power_1p5",
        "signed_power_2",
        "derivative_positive_lag1",
        "derivative_negative_lag1",
        "onset_dominance",
        "derivative_energy",
    ),
    "cut_aware": (
        "carrier_signed",
        "local_psd_signal",
        "asymmetric_state",
        "spatial_coherence",
        "cross_scale_rank",
        "cross_scale_recall",
        "cut_center_sigma1p5",
        "cut_center_sigma2p5",
        "cut_center_sigma3p5",
        "cut_ring_r3p0_t0p75",
        "cut_ring_r3p0_t1p25",
        "cut_ring_r4p5_t0p75",
        "cut_ring_r4p5_t1p25",
        "cut_ring_r6p0_t0p75",
        "cut_ring_r6p0_t1p25",
        "cut_crowd_context",
    ),
    "calibrated": (
        "carrier_signed",
        "local_psd_signal",
        "asymmetric_state",
        "spatial_coherence",
        "cross_scale_rank",
        "cross_scale_recall",
        "cfar_score",
        "cfar_background",
        "cfar_noise",
        "shot_whitened_rho0p25",
        "shot_whitened_rho0p5",
        "shot_whitened_rho1",
        "persistence_activity_gate",
        "persistent_artifact_score",
        "artifact_attenuated_0p5",
        "artifact_attenuated_1",
        "structure_context",
    ),
    "full": FEATURE_IDS,
}

NEGATIVE_EVIDENCE_IDS = {
    "cfar_background",
    "cfar_noise",
    "persistent_artifact_score",
}


class InnovationRankerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class InnovationRankerConfig:
    schema_version: int
    experiment_id: str
    feature_root: Path
    feature_manifest: Path
    source_video: Path
    labels_tsv: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    map_generation: dict[str, Any]
    proposals: dict[str, Any]
    evaluation: dict[str, Any]
    linear_grid: dict[str, Any]
    mlp_grid: dict[str, Any]
    selection: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "InnovationRankerConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version",
            "experiment_id",
            "feature_root",
            "feature_manifest",
            "source_video",
            "labels_tsv",
            "output_dir",
            "preflight_dir",
            "frames",
            "map_generation",
            "proposals",
            "evaluation",
            "linear_grid",
            "mlp_grid",
            "selection",
            "visualization",
            "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise InnovationRankerConfigError(
                f"top-level fields differ; missing={sorted(fields-set(raw))}, "
                f"unknown={sorted(set(raw)-fields)}"
            )
        root = manifest.parent
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            feature_root=(root / raw["feature_root"]).resolve(),
            feature_manifest=(root / raw["feature_manifest"]).resolve(),
            source_video=(root / raw["source_video"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            frames=dict(raw["frames"]),
            map_generation=dict(raw["map_generation"]),
            proposals=dict(raw["proposals"]),
            evaluation=dict(raw["evaluation"]),
            linear_grid=dict(raw["linear_grid"]),
            mlp_grid=dict(raw["mlp_grid"]),
            selection=dict(raw["selection"]),
            visualization=dict(raw["visualization"]),
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise InnovationRankerConfigError("invalid schema or experiment id")
        if set(self.frames) != {
            "review_start_ui",
            "review_end_ui",
            "quiet_count",
        } or int(self.frames["quiet_count"]) != 100:
            raise InnovationRankerConfigError("invalid frame contract")
        if set(self.map_generation) != {
            "signed_powers",
            "whitening_rhos",
            "center_sigmas_px",
            "ring_specs",
            "crowd_sigma_px",
            "artifact_authorities",
            "normalization_clip",
        }:
            raise InnovationRankerConfigError("invalid map-generation fields")
        if (
            self.map_generation["signed_powers"] != [1.25, 1.5, 2.0]
            or self.map_generation["whitening_rhos"] != [0.25, 0.5, 1.0]
            or len(self.map_generation["ring_specs"]) != 6
        ):
            raise InnovationRankerConfigError("frozen map grid differs")
        if set(self.proposals) != {
            "source_ids",
            "nms_distance_px",
            "per_source_limit",
            "dedupe_radius_px",
        } or tuple(self.proposals["source_ids"]) != PROPOSAL_SOURCE_IDS:
            raise InnovationRankerConfigError("invalid proposal contract")
        if set(self.evaluation) != {
            "match_radius_px",
            "fixed_candidates_per_burst",
            "quiet_false_candidates_per_map",
            "oracle_source_budgets",
        }:
            raise InnovationRankerConfigError("invalid evaluation fields")
        if set(self.linear_grid) != {
            "feature_sets",
            "learning_rates",
            "l2_values",
            "maximum_total_weights",
            "epochs",
        } or tuple(self.linear_grid["feature_sets"]) != tuple(FEATURE_SETS):
            raise InnovationRankerConfigError("invalid linear fine-tuning grid")
        if set(self.mlp_grid) != {
            "feature_sets",
            "learning_rates",
            "weight_decays",
            "hidden_units",
            "maximum_residuals",
            "epochs",
            "inner_seeds",
            "confirmation_seeds",
        }:
            raise InnovationRankerConfigError("invalid MLP fine-tuning grid")
        if set(self.mlp_grid["feature_sets"]) - set(FEATURE_SETS):
            raise InnovationRankerConfigError("unknown MLP feature set")
        if set(self.selection) != {
            "minimum_weight",
            "mean_weight",
            "threshold_weight",
            "candidate_penalty",
        }:
            raise InnovationRankerConfigError("invalid selection fields")
        if set(self.visualization) != {
            "overlay_candidate_limit",
            "score_sigma_px",
            "compression",
        } or self.visualization["compression"] != "zlib":
            raise InnovationRankerConfigError("invalid visualization fields")
        if set(self.resources) != {
            "cpu_threads",
            "max_ram_mib",
            "min_free_disk_mib",
            "max_output_mib",
        } or not (
            1 <= int(self.resources["cpu_threads"]) <= 8
            and int(self.resources["max_ram_mib"]) >= 4096
        ):
            raise InnovationRankerConfigError("invalid resource fields")

    @property
    def linear_config_count(self) -> int:
        return (
            len(self.linear_grid["feature_sets"])
            * len(self.linear_grid["learning_rates"])
            * len(self.linear_grid["l2_values"])
            * len(self.linear_grid["maximum_total_weights"])
        )

    @property
    def mlp_config_count(self) -> int:
        return (
            len(self.mlp_grid["feature_sets"])
            * len(self.mlp_grid["learning_rates"])
            * len(self.mlp_grid["weight_decays"])
            * len(self.mlp_grid["hidden_units"])
            * len(self.mlp_grid["maximum_residuals"])
        )

    @property
    def inner_fit_count(self) -> int:
        training_pairs = 6
        return training_pairs * (
            self.linear_config_count
            + self.mlp_config_count * len(self.mlp_grid["inner_seeds"])
        )

    @property
    def outer_refit_count(self) -> int:
        return 4 + 4 * len(self.mlp_grid["confirmation_seeds"])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "feature_root",
            "feature_manifest",
            "source_video",
            "labels_tsv",
            "output_dir",
            "preflight_dir",
        ):
            payload[field] = str(payload[field])
        return payload
