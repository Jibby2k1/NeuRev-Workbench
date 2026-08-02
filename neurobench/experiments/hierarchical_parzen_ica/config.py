"""Strict version-1 manifest for hierarchical clean/noisy Parzen ICA."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class HierarchicalParzenConfigError(ValueError):
    pass


def _strict(payload: Mapping[str, Any], allowed: set[str], scope: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise HierarchicalParzenConfigError(f"{scope} must be an object")
    values = dict(payload)
    unknown = set(values) - allowed
    missing = allowed - set(values)
    if unknown:
        raise HierarchicalParzenConfigError(
            f"Unknown {scope} fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise HierarchicalParzenConfigError(
            f"Missing {scope} fields: {', '.join(sorted(missing))}"
        )
    return values


@dataclass(frozen=True)
class FrameConfig:
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    frame_period_ms: float


@dataclass(frozen=True)
class ResourceConfig:
    device: str
    cpu_threads: int
    projection_chunk_pixels: int
    patch_batch_size: int
    max_ram_mib: int
    max_gpu_memory_mib: int
    min_free_disk_mib: int
    max_output_mib: int


_PREPROCESSING_FIELDS = {
    "motion_mode", "gain_mode", "alpha_min", "alpha_max",
    "signed_stage1_residual", "quiet_scale_floor_percentile",
}
_STAGE1_FIELDS = {
    "lags", "methods", "fit_sample_pixels", "sample_seed", "covariance_mode",
    "eigenvalue_floor_ratio", "condition_number_max", "batch_cs_parzen",
    "stochastic_parzen", "tracking", "staticness", "safety",
    "subtraction_modes",
}
_STAGE1_BATCH_FIELDS = {
    "bandwidth", "kernel_block_rows", "screen_step_degrees",
    "refine_half_width_degrees", "refine_step_degrees",
}
_STAGE1_STOCHASTIC_FIELDS = {
    "dictionary_size", "dictionary_warmup_samples", "minimum_center_separation",
    "bandwidth", "bandwidth_min", "bandwidth_max", "learning_rate",
    "gradient_clip", "maximum_angle_update_degrees", "update_every_frames",
    "freeze_after_calibration", "replacement_policy", "seed",
}
_STAGE1_TRACKING_FIELDS = {
    "minimum_map_correlation", "minimum_mixing_cosine",
    "maximum_unresolved_frames", "fallback",
}
_STAGE1_STATICNESS_FIELDS = {
    "first_difference_weight", "second_difference_weight",
    "common_direction_weight", "spatial_high_frequency_weight",
    "global_intensity_weight", "minimum_confidence_margin", "allow_unresolved",
}
_STAGE1_SAFETY_FIELDS = {
    "maximum_previous_background_coefficient",
    "maximum_current_observation_coefficient",
    "maximum_reconstruction_operator_norm",
    "maximum_learned_fraction",
    "minimum_learned_fraction",
    "require_convergence_for_learned",
    "unsafe_policy",
}
_STAGE2_FIELDS = {
    "patch_height", "patch_width", "stride_y", "stride_x", "overlap_window",
    "noise_models", "subspace_methods", "rank_candidates",
    "minimum_signal_eigenvalue_ratio", "minimum_structured_energy_ratio",
    "maximum_rank", "methods", "ica", "dictionary",
    "projected_noise_variance_floor", "projected_noise_variance_ceiling",
    "stop_gradient_through_noise_variance", "qualification",
    "alternating_refinement_passes",
}
_STAGE2_ICA_FIELDS = {
    "maximum_iterations", "tolerance", "learning_rate", "gradient_clip",
    "decorrelation_floor", "seeds",
}
_STAGE2_DICTIONARY_FIELDS = {
    "maximum_centers", "warmup_samples", "minimum_center_separation",
    "bandwidth", "bandwidth_min", "bandwidth_max", "update_rate",
    "replacement_policy", "freeze_after_calibration", "seed",
}
_STAGE2_QUALIFICATION_FIELDS = {
    "enabled", "minimum_localization_score", "minimum_annularity_score",
    "minimum_temporal_coherence", "minimum_seed_stability",
    "maximum_motion_edge_correlation", "allow_unresolved",
}
_SYNTHETIC_FIELDS = {
    "seeds", "snr_db", "include_slow_ramps", "include_motion_edges",
    "include_poisson_gaussian_noise", "include_correlated_noise",
    "include_overlapping_annular_sources",
}
_EVALUATION_FIELDS = {
    "nms_distance_px", "match_radii_px", "primary_match_radius_px",
    "quiet_false_peaks_per_map", "fixed_candidates_per_burst",
    "temporal_pool_modes", "primary_temporal_pool", "manual_review_panel_size",
    "leakage_bootstrap_samples", "stability_temporal_blocks",
    "patch_offset_ablations",
}
_VISUALIZATION_FIELDS = {
    "write_dense_stage1_background", "write_dense_stage1_residual",
    "write_dense_structured_signal", "write_dense_structured_artifact",
    "write_dense_measurement_noise", "write_selected_tiffs",
    "selected_frames_ui", "component_gallery_count", "decomposition_sheet_count",
    "fixed_scales_across_frames",
}
_REALTIME_FIELDS = {
    "enabled_for_frozen_inference_only", "adaptation_enabled", "warmup_frames",
    "timed_frames", "frame_deadline_ms", "report_percentiles",
}


@dataclass(frozen=True)
class HierarchicalParzenICAConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    output_dir: Path
    frames: FrameConfig
    preprocessing: dict[str, Any]
    stage1: dict[str, Any]
    stage2: dict[str, Any]
    synthetic: dict[str, Any]
    evaluation: dict[str, Any]
    visualization: dict[str, Any]
    realtime: dict[str, Any]
    resources: ResourceConfig

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalParzenICAConfig":
        manifest = Path(path).resolve()
        raw = _strict(json.loads(manifest.read_text(encoding="utf-8")), {
            "schema_version", "experiment_id", "source_video", "labels_tsv",
            "output_dir", "frames", "preprocessing", "stage1", "stage2",
            "synthetic", "evaluation", "visualization", "realtime", "resources",
        }, "top-level")
        frames = _strict(raw["frames"], set(FrameConfig.__dataclass_fields__), "frames")
        preprocessing = _strict(raw["preprocessing"], _PREPROCESSING_FIELDS, "preprocessing")
        stage1 = _strict(raw["stage1"], _STAGE1_FIELDS, "stage1")
        stage1["batch_cs_parzen"] = _strict(
            stage1["batch_cs_parzen"], _STAGE1_BATCH_FIELDS, "stage1.batch_cs_parzen"
        )
        stage1["stochastic_parzen"] = _strict(
            stage1["stochastic_parzen"], _STAGE1_STOCHASTIC_FIELDS,
            "stage1.stochastic_parzen",
        )
        stage1["tracking"] = _strict(
            stage1["tracking"], _STAGE1_TRACKING_FIELDS, "stage1.tracking"
        )
        stage1["staticness"] = _strict(
            stage1["staticness"], _STAGE1_STATICNESS_FIELDS, "stage1.staticness"
        )
        stage1["safety"] = _strict(
            stage1["safety"], _STAGE1_SAFETY_FIELDS, "stage1.safety"
        )
        stage2 = _strict(raw["stage2"], _STAGE2_FIELDS, "stage2")
        stage2["ica"] = _strict(stage2["ica"], _STAGE2_ICA_FIELDS, "stage2.ica")
        stage2["dictionary"] = _strict(
            stage2["dictionary"], _STAGE2_DICTIONARY_FIELDS, "stage2.dictionary"
        )
        stage2["qualification"] = _strict(
            stage2["qualification"], _STAGE2_QUALIFICATION_FIELDS,
            "stage2.qualification",
        )
        synthetic = _strict(raw["synthetic"], _SYNTHETIC_FIELDS, "synthetic")
        evaluation = _strict(raw["evaluation"], _EVALUATION_FIELDS, "evaluation")
        visualization = _strict(
            raw["visualization"], _VISUALIZATION_FIELDS, "visualization"
        )
        realtime = _strict(raw["realtime"], _REALTIME_FIELDS, "realtime")
        resources = _strict(
            raw["resources"], set(ResourceConfig.__dataclass_fields__), "resources"
        )
        root = manifest.parent
        config = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            frames=FrameConfig(
                review_start_ui=int(frames["review_start_ui"]),
                review_end_ui=int(frames["review_end_ui"]),
                quiet_start_ui=int(frames["quiet_start_ui"]),
                quiet_end_ui=int(frames["quiet_end_ui"]),
                frame_period_ms=float(frames["frame_period_ms"]),
            ),
            preprocessing=preprocessing,
            stage1=stage1,
            stage2=stage2,
            synthetic=synthetic,
            evaluation=evaluation,
            visualization=visualization,
            realtime=realtime,
            resources=ResourceConfig(
                device=str(resources["device"]),
                cpu_threads=int(resources["cpu_threads"]),
                projection_chunk_pixels=int(resources["projection_chunk_pixels"]),
                patch_batch_size=int(resources["patch_batch_size"]),
                max_ram_mib=int(resources["max_ram_mib"]),
                max_gpu_memory_mib=int(resources["max_gpu_memory_mib"]),
                min_free_disk_mib=int(resources["min_free_disk_mib"]),
                max_output_mib=int(resources["max_output_mib"]),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise HierarchicalParzenConfigError("schema_version must be exactly 1")
        if not self.experiment_id.strip():
            raise HierarchicalParzenConfigError("experiment_id must be non-empty")
        f = self.frames
        if not (
            1 <= f.review_start_ui <= f.quiet_start_ui
            <= f.quiet_end_ui <= f.review_end_ui
        ) or f.frame_period_ms <= 0:
            raise HierarchicalParzenConfigError("invalid review/quiet frame contract")
        if f.quiet_start_ui != f.review_start_ui:
            raise HierarchicalParzenConfigError(
                "version 1 requires quiet_start_ui to equal review_start_ui"
            )
        p = self.preprocessing
        if p["motion_mode"] != "none" or p["gain_mode"] != "robust_pairwise":
            raise HierarchicalParzenConfigError(
                "version 1 requires no motion transform and robust_pairwise gain"
            )
        if not 0 < float(p["alpha_min"]) < float(p["alpha_max"]):
            raise HierarchicalParzenConfigError("invalid pairwise gain bounds")
        if not bool(p["signed_stage1_residual"]):
            raise HierarchicalParzenConfigError("Stage 1 residual must remain signed")

        s1 = self.stage1
        lags = tuple(int(value) for value in s1["lags"])
        quiet_frames = f.quiet_end_ui - f.quiet_start_ui + 1
        if not lags or len(set(lags)) != len(lags) or min(lags) < 1 or max(lags) >= quiet_frames:
            raise HierarchicalParzenConfigError("Stage 1 lags must be unique and bounded")
        required_stage1 = {
            "fixed_common_difference_reference", "adaptive_gain_common_difference",
            "batch_cs_parzen_pairwise", "stochastic_parzen_score_pairwise",
        }
        if set(s1["methods"]) != required_stage1:
            raise HierarchicalParzenConfigError(
                "version 1 requires all four declared Stage 1 reference lanes"
            )
        if s1["covariance_mode"] not in {"ordinary", "robust"}:
            raise HierarchicalParzenConfigError("invalid Stage 1 covariance mode")
        if not 16 <= int(s1["fit_sample_pixels"]) <= 65536:
            raise HierarchicalParzenConfigError("Stage 1 fit sample count is unbounded")
        if not 0 < float(s1["eigenvalue_floor_ratio"]) < 1:
            raise HierarchicalParzenConfigError("invalid Stage 1 eigenvalue floor")
        if float(s1["condition_number_max"]) <= 1:
            raise HierarchicalParzenConfigError("invalid Stage 1 condition bound")
        stochastic = s1["stochastic_parzen"]
        if not (
            2 <= int(stochastic["dictionary_size"]) <= 1024
            and 2 <= int(stochastic["dictionary_warmup_samples"]) <= 65536
            and 0 < float(stochastic["bandwidth_min"])
            <= float(stochastic["bandwidth"])
            <= float(stochastic["bandwidth_max"])
        ):
            raise HierarchicalParzenConfigError("invalid Stage 1 dictionary bounds")
        if stochastic["replacement_policy"] not in {
            "farthest_center", "deterministic_reservoir"
        }:
            raise HierarchicalParzenConfigError("invalid Stage 1 replacement policy")
        if set(s1["subtraction_modes"]) != {"exact", "confidence_weighted"}:
            raise HierarchicalParzenConfigError(
                "version 1 requires exact and confidence-weighted Stage 1 controls"
            )
        tracking = s1["tracking"]
        if not 0 <= float(tracking["minimum_map_correlation"]) <= 1:
            raise HierarchicalParzenConfigError("invalid map-correlation tracking bound")
        if not 0 <= float(tracking["minimum_mixing_cosine"]) <= 1:
            raise HierarchicalParzenConfigError("invalid mixing-cosine tracking bound")
        if tracking["fallback"] != "last_accepted":
            raise HierarchicalParzenConfigError("version 1 fallback must be last_accepted")
        safety = s1["safety"]
        if not (
            1 <= float(safety["maximum_previous_background_coefficient"]) <= 2
            and 0 < float(safety["maximum_current_observation_coefficient"]) <= 0.5
            and 1 <= float(safety["maximum_reconstruction_operator_norm"]) <= 4
            and 0 < float(safety["minimum_learned_fraction"])
            <= float(safety["maximum_learned_fraction"]) <= 0.25
        ):
            raise HierarchicalParzenConfigError(
                "invalid Stage 1 recursive safety bounds"
            )
        if not bool(safety["require_convergence_for_learned"]):
            raise HierarchicalParzenConfigError(
                "version 1 requires learned-lane convergence"
            )
        if safety["unsafe_policy"] != "reference_fallback":
            raise HierarchicalParzenConfigError(
                "version 1 unsafe learned fits must use reference_fallback"
            )

        s2 = self.stage2
        height, width = int(s2["patch_height"]), int(s2["patch_width"])
        stride_y, stride_x = int(s2["stride_y"]), int(s2["stride_x"])
        if not 8 <= height <= 64 or not 8 <= width <= 64:
            raise HierarchicalParzenConfigError("Stage 2 patch dimensions are unbounded")
        if not 1 <= stride_y < height or not 1 <= stride_x < width:
            raise HierarchicalParzenConfigError("Stage 2 requires overlapping positive strides")
        if s2["overlap_window"] not in {"hann", "tukey"}:
            raise HierarchicalParzenConfigError("invalid overlap-add window")
        if set(s2["noise_models"]) != {"diagonal_robust", "diagonal_shrinkage"}:
            raise HierarchicalParzenConfigError(
                "version 1 requires both bounded diagonal noise controls"
            )
        if set(s2["subspace_methods"]) != {"ordinary_pca", "noise_corrected_pca"}:
            raise HierarchicalParzenConfigError(
                "version 1 requires ordinary and noise-corrected subspace controls"
            )
        if set(s2["methods"]) != {"ordinary_parzen_ica", "noisy_parzen_ica_posterior"}:
            raise HierarchicalParzenConfigError(
                "version 1 requires ordinary and noisy posterior Parzen lanes"
            )
        ranks = tuple(int(value) for value in s2["rank_candidates"])
        maximum_rank = int(s2["maximum_rank"])
        if not ranks or min(ranks) < 1 or max(ranks) > maximum_rank or maximum_rank > min(height * width, 32):
            raise HierarchicalParzenConfigError("invalid bounded Stage 2 rank candidates")
        if int(s2["alternating_refinement_passes"]) not in {0, 1}:
            raise HierarchicalParzenConfigError("at most one refinement pass is permitted")
        dictionary = s2["dictionary"]
        if not (
            2 <= int(dictionary["maximum_centers"]) <= 1024
            and 2 <= int(dictionary["warmup_samples"]) <= 65536
            and 0 < float(dictionary["bandwidth_min"])
            <= float(dictionary["bandwidth"])
            <= float(dictionary["bandwidth_max"])
            and 0 <= float(dictionary["update_rate"]) <= 1
        ):
            raise HierarchicalParzenConfigError("invalid Stage 2 dictionary bounds")
        seeds = tuple(int(value) for value in s2["ica"]["seeds"])
        if not seeds or len(set(seeds)) != len(seeds):
            raise HierarchicalParzenConfigError("Stage 2 ICA seeds must be unique")
        if not 0 < float(s2["projected_noise_variance_floor"]) <= float(
            s2["projected_noise_variance_ceiling"]
        ):
            raise HierarchicalParzenConfigError("invalid projected-noise variance bounds")

        evaluation = self.evaluation
        if (
            int(evaluation["primary_match_radius_px"]) != 6
            or 6 not in tuple(int(value) for value in evaluation["match_radii_px"])
        ):
            raise HierarchicalParzenConfigError("primary match radius is frozen at six pixels")
        if int(evaluation["fixed_candidates_per_burst"]) != 58:
            raise HierarchicalParzenConfigError("fixed candidate budget is frozen at 58")
        if evaluation["primary_temporal_pool"] not in evaluation["temporal_pool_modes"]:
            raise HierarchicalParzenConfigError("primary temporal pool must be declared")
        if float(evaluation["quiet_false_peaks_per_map"]) <= 0:
            raise HierarchicalParzenConfigError("quiet candidate calibration must be positive")

        selected = tuple(int(value) for value in self.visualization["selected_frames_ui"])
        if len(set(selected)) != len(selected) or any(
            value < f.review_start_ui or value > f.review_end_ui for value in selected
        ):
            raise HierarchicalParzenConfigError("selected visual frames must be unique and in review")
        if not bool(self.visualization["fixed_scales_across_frames"]):
            raise HierarchicalParzenConfigError("visual scales must remain fixed across frames")
        if bool(self.realtime["adaptation_enabled"]):
            raise HierarchicalParzenConfigError("version 1 realtime benchmark freezes adaptation")
        if float(self.realtime["frame_deadline_ms"]) != f.frame_period_ms:
            raise HierarchicalParzenConfigError("realtime deadline must match frame period")
        if tuple(int(x) for x in self.realtime["report_percentiles"]) != (50, 95, 99):
            raise HierarchicalParzenConfigError("realtime percentiles must be 50/95/99")

        resources = self.resources
        if resources.device not in {"cpu", "cuda"} or not 1 <= resources.cpu_threads <= 8:
            raise HierarchicalParzenConfigError("invalid device or CPU thread bound")
        if min(
            resources.projection_chunk_pixels,
            resources.patch_batch_size,
            resources.max_ram_mib,
            resources.max_gpu_memory_mib,
            resources.min_free_disk_mib,
            resources.max_output_mib,
        ) <= 0:
            raise HierarchicalParzenConfigError("all resource caps must be positive")
        synthetic_seeds = tuple(int(value) for value in self.synthetic["seeds"])
        if not synthetic_seeds or len(set(synthetic_seeds)) != len(synthetic_seeds):
            raise HierarchicalParzenConfigError("synthetic seeds must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_video", "labels_tsv", "output_dir"):
            payload[key] = str(payload[key])
        return payload
