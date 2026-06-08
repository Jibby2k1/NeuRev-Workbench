"""Shared pipeline stage catalog and validation helpers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Iterable
from numbers import Real
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ParameterRange:
    """Inclusive numeric parameter bounds."""

    minimum: float | None = None
    maximum: float | None = None

    def validate(self, stage_id: str, param_name: str, value: Any) -> None:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"Pipeline stage '{stage_id}' parameter '{param_name}' must be numeric.")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(
                f"Pipeline stage '{stage_id}' parameter '{param_name}'={value} is below minimum {self.minimum}."
            )
        if self.maximum is not None and value > self.maximum:
            raise ValueError(
                f"Pipeline stage '{stage_id}' parameter '{param_name}'={value} is above maximum {self.maximum}."
            )

    def as_dict(self) -> dict[str, float]:
        data: dict[str, float] = {}
        if self.minimum is not None:
            data["minimum"] = self.minimum
        if self.maximum is not None:
            data["maximum"] = self.maximum
        return data


@dataclass(frozen=True)
class PipelineStage:
    """Catalog entry for a pipeline stage."""

    stage_id: str
    label: str
    order: int
    required_params: tuple[str, ...] = ()
    default_params: Mapping[str, Any] | None = None
    param_ranges: Mapping[str, ParameterRange] | None = None
    description: str = ""

    def merged_params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = deepcopy(dict(self.default_params or {}))
        merged.update(dict(params or {}))
        return merged

    def as_dict(self) -> dict[str, Any]:
        metadata = _stage_metadata(self.stage_id)
        return {
            "stage_id": self.stage_id,
            "label": self.label,
            "order": self.order,
            "availability": metadata["availability"],
            "ui_group": metadata["ui_group"],
            "type": metadata["type"],
            "input": metadata["input"],
            "output": metadata["output"],
            "expected_qc_outputs": metadata["expected_qc_outputs"],
            "required_params": list(self.required_params),
            "default_params": deepcopy(dict(self.default_params or {})),
            "param_ranges": {key: value.as_dict() for key, value in dict(self.param_ranges or {}).items()},
            "description": self.description,
            "why_use_it": metadata["why_use_it"],
            "parameter_docs": _parameter_docs(self.stage_id, self),
            "real_time_profile": metadata["real_time_profile"],
        }


_DEFAULT_REALTIME = {
    "mode": "unknown",
    "latency_budget_ms": None,
    "requires_gpu": False,
    "stateful": False,
    "adaptive": False,
    "closed_loop_candidate": False,
}

_DEFAULT_EXPECTED_QC_OUTPUTS: tuple[str, ...] = ()

LOCAL_RUNNER_STAGE_IDS = frozenset(
    {

        "video_manifest_build",
        "template_build_from_video",
        "template_register_video",
        "apply_video_registration",
        "grid_32x32_generate",
        "grid_state_extract",
        "grid_dynamics_dataset_build",
        "grid_autoencoder_train",
        "latent_rnn_train",
        "latent_classifier_train",
        "source_video_import",
        "temporal_highpass_gaussian",
        "event_preserving_noise_suppression",
        "spatial_gaussian",
        "rigid_shift_estimate",
        "robust_positive_local_z",
        "adaptive_ewma_z",
        "gamma_cfar",
        "adaptive_gamma_cfar",
        "candidate_event_pipeline",
        "component_filter",
        "local_background_ring",
        "trace_event_scoring",
        "robust_kalman_positive_innovation",
        "heuristic_priority_v1",
        "generate_neuron_review_app",
    }
)


_STAGE_METADATA: dict[str, dict[str, Any]] = {

    "video_manifest_build": {
        "availability": "implemented",
        "ui_group": "template_grid",
        "type": "manifest",
        "input": "",
        "output": "video_manifest",
        "expected_qc_outputs": ("video_manifest", "label_counts"),
        "why_use_it": "Parses filename-derived left/right/neutral labels and enforces video-level split units.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "template_build_from_video": {
        "availability": "implemented",
        "ui_group": "template_grid",
        "type": "template",
        "input": "video_manifest",
        "output": "template_spec",
        "expected_qc_outputs": ("template_projection", "outlier_frame_scores"),
        "why_use_it": "Builds a robust one-reference-video anatomical template for per-video registration.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "template_register_video": {
        "availability": "implemented",
        "ui_group": "template_grid",
        "type": "registration",
        "input": "template_spec",
        "output": "registration_results",
        "expected_qc_outputs": ("source_projection", "registered_projection", "registration_overlay"),
        "why_use_it": "Estimates one translation/rotation correction per video into template coordinates.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "apply_video_registration": {
        "availability": "implemented",
        "ui_group": "template_grid",
        "type": "registration",
        "input": "registration_results",
        "output": "registered_videos",
        "expected_qc_outputs": ("registered_video", "registered_projection"),
        "why_use_it": "Applies the per-video transform to every frame before grid pooling.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "grid_32x32_generate": {
        "availability": "implemented",
        "ui_group": "template_grid",
        "type": "grid",
        "input": "template_spec",
        "output": "grid_spec",
        "expected_qc_outputs": ("grid_overlay",),
        "why_use_it": "Creates deterministic rectangular template-coordinate regions for model states.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "grid_state_extract": {
        "availability": "implemented",
        "ui_group": "template_grid",
        "type": "grid",
        "input": "registered_videos",
        "output": "grid_states",
        "expected_qc_outputs": ("grid_preview", "grid_trace_summary"),
        "why_use_it": "Pools registered frames into 32x32 grid state sequences without detecting individual neurons.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "grid_dynamics_dataset_build": {
        "availability": "implemented",
        "ui_group": "dynamics",
        "type": "dataset",
        "input": "grid_states",
        "output": "dynamics_dataset",
        "expected_qc_outputs": ("split_manifest", "baseline_inputs"),
        "why_use_it": "Builds windows with train/validation/test splits by video to avoid frame leakage.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "grid_autoencoder_train": {
        "availability": "implemented",
        "ui_group": "dynamics",
        "type": "model_training",
        "input": "dynamics_dataset",
        "output": "autoencoder_run",
        "expected_qc_outputs": ("reconstruction_examples", "training_curve"),
        "why_use_it": "Learns compact grid-frame latent codes with a small CPU-safe CNN autoencoder.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None, "requires_gpu": False},
    },
    "latent_rnn_train": {
        "availability": "implemented",
        "ui_group": "dynamics",
        "type": "model_training",
        "input": "autoencoder_run",
        "output": "latent_rnn_run",
        "expected_qc_outputs": ("prediction_examples", "persistence_baseline"),
        "why_use_it": "Trains a GRU next-latent predictor and compares it with persistence.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None, "requires_gpu": False},
    },
    "latent_classifier_train": {
        "availability": "implemented",
        "ui_group": "dynamics",
        "type": "model_training",
        "input": "autoencoder_run",
        "output": "latent_classifier_run",
        "expected_qc_outputs": ("confusion_matrix", "per_video_predictions"),
        "why_use_it": "Classifies neutral/left/right from video-level latent summaries using filename labels.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None, "requires_gpu": False},
    },
    "source_video_import": {
        "availability": "implemented",
        "ui_group": "import",
        "type": "import",
        "input": "",
        "output": "raw_video",
        "expected_qc_outputs": ("raw_frame", "frame_mean_trace", "frame_max_trace"),
        "why_use_it": "Makes the source movie explicit so downstream artifacts and timing can be traced.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "closed_loop_candidate": True},
    },
    "review_data_import": {
        "availability": "implemented",
        "ui_group": "import",
        "type": "import",
        "input": "review_data",
        "output": "roi_candidates",
        "why_use_it": "Reuses an existing reviewed/candidate dataset as an Architecture Lab baseline.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "temporal_highpass_gaussian": {
        "availability": "implemented",
        "ui_group": "preprocessing",
        "type": "temporal_smoothing",
        "input": "raw_video",
        "output": "highpass_video",
        "expected_qc_outputs": ("highpass_frame", "temporal_baseline_trace"),
        "why_use_it": "Penalizes slow baseline drift while preserving fast calcium transients.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "event_preserving_noise_suppression": {
        "availability": "implemented",
        "ui_group": "denoising",
        "type": "denoising",
        "input": "highpass_video",
        "output": "denoised_video",
        "expected_qc_outputs": ("denoised_frame", "impulse_rejection_summary"),
        "why_use_it": "Reduces impulse-like noise before candidate extraction without intentionally blurring events away.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "spatial_gaussian": {
        "availability": "implemented",
        "ui_group": "preprocessing",
        "type": "spatial_smoothing",
        "input": "highpass_video",
        "output": "smoothed_video",
        "expected_qc_outputs": ("smoothed_frame", "smoothing_residual_frame"),
        "why_use_it": "Suppresses pixel-scale noise before CFAR or local-z scoring.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "closed_loop_candidate": True},
    },
    "rigid_shift_estimate": {
        "availability": "implemented",
        "ui_group": "motion",
        "type": "motion_correction",
        "input": "raw_video",
        "output": "registered_video",
        "expected_qc_outputs": ("registered_frame", "rigid_shift_trace"),
        "why_use_it": "Flags or corrects frame drift that can masquerade as neural activity.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 3.0, "stateful": True, "closed_loop_candidate": True},
    },
    "suite2p_import": {
        "availability": "external_import",
        "ui_group": "external",
        "type": "import",
        "input": "suite2p_output",
        "output": "roi_candidates",
        "why_use_it": "Benchmarks the current workflow against a widely used ROI extraction package.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None, "requires_gpu": False},
    },
    "pmd_import": {
        "availability": "external_import",
        "ui_group": "external",
        "type": "import",
        "input": "pmd_output",
        "output": "denoised_video",
        "why_use_it": "Compares against a low-rank denoising baseline without treating it as ground truth.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "oasis_import": {
        "availability": "external_import",
        "ui_group": "external",
        "type": "import",
        "input": "trace_array",
        "output": "deconvolved_events",
        "why_use_it": "Compares reviewed calcium events with an established deconvolution output.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "stateful": True, "closed_loop_candidate": True},
    },
    "pmd_denoised_video_import": {
        "availability": "external_import",
        "ui_group": "external",
        "type": "import",
        "input": "raw_video",
        "output": "highpass_video",
        "why_use_it": "Uses a denoised movie as candidate evidence while preserving raw-video provenance.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "robust_positive_local_z": {
        "availability": "implemented",
        "ui_group": "evidence",
        "type": "filtering",
        "input": "highpass_video",
        "output": "z_stack",
        "expected_qc_outputs": ("positive_z_frame", "max_z_projection"),
        "why_use_it": "Highlights positive local excursions using robust local scale estimates.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "adaptive_ewma_z": {
        "availability": "implemented",
        "ui_group": "evidence",
        "type": "filtering",
        "input": "raw_video",
        "output": "z_stack",
        "expected_qc_outputs": ("adaptive_z_frame", "active_fraction_trace"),
        "why_use_it": "Maintains streaming per-pixel baseline and variance estimates for 100 Hz candidate screening.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "gamma_cfar": {
        "availability": "implemented",
        "ui_group": "detection",
        "type": "filtering",
        "input": "smoothed_video",
        "output": "candidate_mask",
        "expected_qc_outputs": ("candidate_mask", "threshold_summary"),
        "why_use_it": "Adapts thresholds to local background so bright and dim regions are treated more fairly.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "adaptive": True, "closed_loop_candidate": True},
    },
    "adaptive_gamma_cfar": {
        "availability": "implemented",
        "ui_group": "detection",
        "type": "filtering",
        "input": "smoothed_video",
        "output": "candidate_mask",
        "expected_qc_outputs": ("candidate_mask", "adaptive_threshold_trace"),
        "why_use_it": "Uses a local training region and streaming update rate to keep CFAR thresholds responsive at high frame rates.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "candidate_event_pipeline": {
        "availability": "implemented",
        "ui_group": "detection",
        "type": "filtering",
        "input": "z_stack",
        "output": "candidate_events",
        "expected_qc_outputs": ("event_component_overlay", "event_count_trace"),
        "why_use_it": "Produces permissive event and discovery candidates for human triage.",
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "component_filter": {
        "availability": "implemented",
        "ui_group": "roi",
        "type": "trace_extraction",
        "input": "z_stack",
        "output": "roi_candidates",
        "expected_qc_outputs": ("roi_candidate_overlay", "roi_size_distribution"),
        "why_use_it": "Turns pixel evidence into object-level neuron candidates with size and temporal-support constraints.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "closed_loop_candidate": True},
    },
    "local_background_ring": {
        "availability": "implemented",
        "ui_group": "trace",
        "type": "background_correction",
        "input": "roi_candidates",
        "output": "roi_traces",
        "expected_qc_outputs": ("raw_roi_trace", "background_trace", "corrected_trace"),
        "why_use_it": "Subtracts nearby background/neuropil signal before event scoring.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "closed_loop_candidate": True},
    },
    "trace_event_scoring": {
        "availability": "implemented",
        "ui_group": "event_model",
        "type": "event_model",
        "input": "roi_traces",
        "output": "candidate_events",
        "expected_qc_outputs": ("event_score_trace", "candidate_event_markers"),
        "why_use_it": "Provides a simple thresholded trace event baseline.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "closed_loop_candidate": True},
    },
    "robust_kalman_positive_innovation": {
        "availability": "implemented",
        "ui_group": "event_model",
        "type": "event_model",
        "input": "roi_traces",
        "output": "candidate_events",
        "expected_qc_outputs": ("kalman_baseline_trace", "innovation_trace", "candidate_event_markers"),
        "why_use_it": "Tracks a robust baseline and calls positive innovations as candidate transients.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "oasis_deconvolution_import": {
        "availability": "external_import",
        "ui_group": "event_model",
        "type": "event_model",
        "input": "roi_traces",
        "output": "deconvolved_events",
        "why_use_it": "Attaches deconvolved activity estimates for comparison against reviewed event labels.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "stateful": True, "closed_loop_candidate": True},
    },
    "heuristic_priority_v1": {
        "availability": "implemented",
        "ui_group": "ranking",
        "type": "candidate_ranking",
        "input": "roi_candidates",
        "output": "ranked_candidates",
        "expected_qc_outputs": ("priority_score_distribution", "review_queue_preview"),
        "why_use_it": "Orders candidates for review using transparent feature weights rather than hidden labels.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "adaptive": False, "closed_loop_candidate": True},
    },
    "generate_neuron_review_app": {
        "availability": "implemented",
        "ui_group": "export",
        "type": "export",
        "input": "ranked_candidates",
        "output": "review_app",
        "why_use_it": "Builds the human review dashboard and exported summary artifacts.",
        "expected_qc_outputs": ("review_data", "dashboard_assets", "annotation_summary"),
        "real_time_profile": {"mode": "offline", "latency_budget_ms": None},
    },
    "flat_field_background": {
        "availability": "planned",
        "ui_group": "background",
        "type": "background_correction",
        "input": "raw_video",
        "output": "background_corrected_video",
        "expected_qc_outputs": ("background_map", "corrected_frame"),
        "why_use_it": "Compensates for uneven illumination and nonuniform background before detection.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "photobleach_correction": {
        "availability": "planned",
        "ui_group": "background",
        "type": "background_correction",
        "input": "raw_video",
        "output": "bleach_corrected_video",
        "expected_qc_outputs": ("photobleach_curve", "corrected_frame_mean_trace"),
        "why_use_it": "Removes slow global fluorescence decay before comparing activity evidence.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "temporal_hampel": {
        "availability": "planned",
        "ui_group": "denoising",
        "type": "denoising",
        "input": "raw_video",
        "output": "despiked_video",
        "expected_qc_outputs": ("impulse_rejection_count", "despiked_frame"),
        "why_use_it": "Suppresses single-frame impulse noise while preserving sparse calcium transients.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "stateful": True, "closed_loop_candidate": True},
    },
    "kalman_smoother": {
        "availability": "planned",
        "ui_group": "denoising",
        "type": "denoising",
        "input": "roi_traces",
        "output": "smoothed_traces",
        "expected_qc_outputs": ("smoothed_trace", "innovation_trace"),
        "why_use_it": "Smooths noisy traces while exposing innovations that may correspond to events.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "stateful": True, "adaptive": True, "closed_loop_candidate": True},
    },
    "local_temporal_correlation": {
        "availability": "planned",
        "ui_group": "evidence",
        "type": "evidence",
        "input": "raw_video",
        "output": "local_correlation_map",
        "expected_qc_outputs": ("local_correlation_map",),
        "why_use_it": "Highlights spatially coherent activity and downranks random impulse noise.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "event_triggered_footprint": {
        "availability": "planned",
        "ui_group": "evidence",
        "type": "evidence",
        "input": "candidate_events",
        "output": "event_triggered_footprints",
        "expected_qc_outputs": ("event_triggered_support_map", "footprint_movie"),
        "why_use_it": "Checks whether candidate events have compact multi-frame calcium footprints.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "background_heterogeneity_map": {
        "availability": "planned",
        "ui_group": "artifact",
        "type": "artifact_model",
        "input": "raw_video",
        "output": "background_heterogeneity_map",
        "expected_qc_outputs": ("background_heterogeneity_map",),
        "why_use_it": "Flags locally clustered background that can create false positives.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "saturation_blob_map": {
        "availability": "planned",
        "ui_group": "artifact",
        "type": "artifact_model",
        "input": "raw_video",
        "output": "saturation_blob_map",
        "expected_qc_outputs": ("saturation_map", "bright_blob_map"),
        "why_use_it": "Identifies saturated or persistent bright structures before ROI review.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "closed_loop_candidate": True},
    },
    "motion_sensitivity_map": {
        "availability": "planned",
        "ui_group": "motion",
        "type": "artifact_model",
        "input": "raw_video",
        "output": "motion_sensitivity_map",
        "expected_qc_outputs": ("motion_sensitivity_map", "shift_trace"),
        "why_use_it": "Flags candidates that track drift or frame motion rather than neural activity.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "log_dog_blob_candidates": {
        "availability": "planned",
        "ui_group": "roi",
        "type": "trace_extraction",
        "input": "z_stack",
        "output": "roi_candidates",
        "expected_qc_outputs": ("blob_scale_map", "candidate_count_by_scale"),
        "why_use_it": "Adds soma-scale blob candidates that may catch missed compact neurons.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 2.0, "closed_loop_candidate": True},
    },
    "watershed_split_merge": {
        "availability": "planned",
        "ui_group": "roi",
        "type": "trace_extraction",
        "input": "roi_candidates",
        "output": "split_merge_candidates",
        "expected_qc_outputs": ("split_merge_suggestions",),
        "why_use_it": "Suggests splits for merged clusters and merges for duplicate fragments.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "ensemble_union": {
        "availability": "planned",
        "ui_group": "ensemble",
        "type": "candidate_ranking",
        "input": "roi_candidates",
        "output": "ensemble_candidates",
        "expected_qc_outputs": ("method_agreement_counts",),
        "why_use_it": "Combines candidates from multiple methods while preserving provenance.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "consensus_stability_scoring": {
        "availability": "planned",
        "ui_group": "ensemble",
        "type": "candidate_ranking",
        "input": "ensemble_candidates",
        "output": "ranked_candidates",
        "expected_qc_outputs": ("consensus_score_distribution", "stability_score_distribution"),
        "why_use_it": "Prioritizes candidates that persist across methods or nearby parameters.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
    "artifact_classifier_v1": {
        "availability": "planned",
        "ui_group": "artifact",
        "type": "artifact_model",
        "input": "roi_candidates",
        "output": "artifact_scores",
        "expected_qc_outputs": ("artifact_score_distribution", "artifact_reason_counts"),
        "why_use_it": "Ranks likely vessels, borders, bright blobs, motion artifacts, and impulse noise for faster triage.",
        "real_time_profile": {"mode": "streaming", "latency_budget_ms": 1.0, "closed_loop_candidate": True},
    },
    "active_learning_ranker": {
        "availability": "planned",
        "ui_group": "ranking",
        "type": "candidate_ranking",
        "input": "artifact_scores",
        "output": "ranked_review_tasks",
        "expected_qc_outputs": ("ranker_confidence_distribution", "uncertainty_queue"),
        "why_use_it": "Uses reviewed labels to propose the next most useful candidates once enough annotations exist.",
        "real_time_profile": {"mode": "batch", "latency_budget_ms": None},
    },
}


_PARAMETER_DOCS: dict[str, dict[str, str]] = {

    "input_dir": "Directory containing labeled videos to scan.",
    "filename_regex": "Regex with named index and label groups for video filenames.",
    "reference_video_id": "Video ID used as the first anatomical template source.",
    "max_outlier_fraction": "Maximum fraction of frames removed while building the robust template projection.",
    "z_threshold": "Frame outlier z-score threshold for template construction.",
    "transform_model": "Registration model: translation, rigid, or similarity.",
    "rotation_range_deg": "Inclusive registration rotation search range in degrees.",
    "rotation_step_deg": "Registration rotation search step in degrees.",
    "allow_uniform_scale": "Whether registration may search a narrow uniform scale factor.",
    "rows": "Number of rectangular grid rows.",
    "cols": "Number of rectangular grid columns.",
    "features": "Grid feature channels to extract.",
    "normalization": "Grid-state normalization method.",
    "window_frames": "Number of frames in each recurrent model input window.",
    "prediction_horizon_frames": "Future frame offset to predict.",
    "split_unit": "Unit used for train/validation/test splits; must remain video.",
    "split_method": "Video-level split method.",
    "latent_dim": "Size of the learned grid latent code.",
    "hidden_dim": "Hidden dimension of the recurrent latent predictor.",
    "epochs": "Training epochs for CPU smoke model runs.",
    "batch_size": "Training batch size.",
    "learning_rate": "Optimizer learning rate.",
    "classifier": "Latent summary classifier family.",
    "source": "Path to the raw movie or frame stack. Keep this relative when possible for reproducibility.",
    "review_data": "Path to an existing review_data.json artifact to import as a baseline run.",
    "sigma_frames": "Temporal high-pass scale in frames. At 100 Hz, convert from seconds rather than copying 5 Hz values directly.",
    "spatial_sigma_px": "Spatial smoothing radius in pixels. Larger values suppress speckle but can merge nearby neurons.",
    "temporal_window_frames": "Temporal window used for event-preserving noise checks. Short windows are safer for sparse fast events.",
    "sigma_px": "Gaussian blur radius in pixels before spatial detection. Use the smallest value that reduces pixel noise.",
    "max_shift_px": "Maximum rigid x/y drift to search per frame. High values are slower and can overfit weak texture.",
    "reference": "Reference frame strategy for motion estimates. Use first for stable starts, mean/median for representative offline references.",
    "suite2p_dir": "Folder containing Suite2p outputs such as stat.npy, F.npy, and spks.npy.",
    "pmd_dir": "Folder containing PMD outputs for denoising/import comparison.",
    "traces": "Trace array path used by deconvolution importers.",
    "denoised_video": "Path to a denoised video artifact; keep raw-video provenance attached.",
    "local_radius_px": "Local neighborhood radius for robust z scoring. Should cover background around a soma without swallowing nearby cells.",
    "epsilon": "Small stabilizer added to the denominator to avoid noise blow-ups in low-variance regions.",
    "pfa": "Target false-alarm probability for CFAR-style adaptive thresholds. Lower values are more conservative.",
    "guard_px": "Pixels around the test point excluded from background estimation so the candidate does not train on itself.",
    "training_radius_px": "Outer local background radius used by adaptive CFAR training statistics.",
    "update_alpha": "Streaming update rate for adaptive background statistics. Lower values are slower but more stable.",
    "alpha": "EWMA update rate for online baseline and variance estimates.",
    "threshold_z": "Streaming z-score threshold for candidate activity masks.",
    "event_threshold_z": "Z-score threshold for candidate event calls. Lower values improve recall but increase review burden.",
    "min_area_px": "Smallest accepted component area in pixels. Use microscope resolution and expected soma size to set this.",
    "max_area_px": "Largest accepted component area in pixels. Helps flag merged clusters and broad artifacts.",
    "seed_z": "Higher threshold used to seed connected components from strong evidence peaks.",
    "grow_z": "Lower threshold used to grow components around seeds without swallowing background.",
    "projection_blob_z": "Threshold for persistent green projection evidence unioned into ROI candidates.",
    "sustained_z": "Trace-relative z-score threshold for marking sustained non-peak fluorescence intervals.",
    "tonic_z": "Trace-level tonic fluorescence score used to mark persistently active green ROIs.",
    "peak_window_frames": "Number of neighboring frames around event peaks excluded from sustained-activity intervals.",
    "outer_radius_px": "Outer radius of the local background/neuropil ring around an ROI.",
    "neuropil_weight": "Fraction of local background ring signal subtracted from the ROI trace.",
    "kalman_gain": "How quickly the baseline follows slow trace changes. Too high can absorb real calcium events.",
    "spike_gain": "How much positive innovation updates the event model. Use conservatively for sparse events.",
    "array_key": "Key inside an .npz file containing imported deconvolved activity.",
    "local_correlation_weight": "Priority contribution from local spatial/temporal consistency.",
    "event_support_weight": "Priority contribution from event evidence support.",
    "artifact_weight": "Negative priority contribution from artifact-like cues.",
    "include_discovery": "Whether the generated dashboard should include missed-neuron discovery suggestions.",
}


def _stage_metadata(stage_id: str) -> dict[str, Any]:
    metadata = deepcopy(_STAGE_METADATA.get(stage_id, {}))
    realtime = deepcopy(_DEFAULT_REALTIME)
    realtime.update(metadata.get("real_time_profile", {}))
    metadata["real_time_profile"] = realtime
    metadata.setdefault("availability", "implemented")
    metadata.setdefault("ui_group", metadata.get("type", "stage"))
    metadata.setdefault("expected_qc_outputs", _DEFAULT_EXPECTED_QC_OUTPUTS)
    metadata.setdefault("type", "stage")
    metadata.setdefault("input", "")
    metadata.setdefault("output", "")
    metadata.setdefault("why_use_it", "Use this stage when its output is needed by the following pipeline step.")
    return metadata


def _parameter_docs(stage_id: str, stage: PipelineStage) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    names = set(stage.required_params)
    names.update(dict(stage.default_params or {}))
    names.update(dict(stage.param_ranges or {}))
    for name in sorted(names):
        item: dict[str, Any] = {
            "meaning": _PARAMETER_DOCS.get(name, f"Parameter '{name}' for {stage.label}."),
            "default": deepcopy(dict(stage.default_params or {}).get(name)),
            "required": name in stage.required_params,
            "why": "Tune this parameter with sweeps when candidate recall, artifact burden, or 100 Hz latency changes.",
        }
        param_range = dict(stage.param_ranges or {}).get(name)
        if param_range is not None:
            item["range"] = param_range.as_dict()
        docs[name] = item
    return docs


STAGE_CATALOG: dict[str, PipelineStage] = {

    "video_manifest_build": PipelineStage(
        stage_id="video_manifest_build",
        label="Video manifest build",
        order=5,
        required_params=("input_dir",),
        default_params={"filename_regex": "^(?P<index>[1-9])_(?P<label>left|right|neutral)\\.(?:tif|tiff|npy)$", "split_unit": "video", "labels": ["left", "right", "neutral"]},
        description="Scan labeled zebrafish videos and write a video-level manifest.",
    ),
    "template_build_from_video": PipelineStage(
        stage_id="template_build_from_video",
        label="Template build from video",
        order=15,
        default_params={"reference_video_id": "1_neutral", "projection_kind": "mean_after_outlier_rejection", "outlier_rejection": True, "max_outlier_fraction": 0.05, "z_threshold": 3.5, "chunk_size_frames": 64},
        param_ranges={"max_outlier_fraction": ParameterRange(minimum=0.0, maximum=0.5), "z_threshold": ParameterRange(minimum=0.0, maximum=20.0), "chunk_size_frames": ParameterRange(minimum=1, maximum=10000)},
        description="Build a robust mean anatomical template from one reference video.",
    ),
    "template_register_video": PipelineStage(
        stage_id="template_register_video",
        label="Template register videos",
        order=20,
        default_params={"transform_model": "rigid", "rotation_range_deg": [-10.0, 10.0], "rotation_step_deg": 0.5, "allow_uniform_scale": False, "chunk_size_frames": 64},
        param_ranges={"rotation_step_deg": ParameterRange(minimum=0.1, maximum=10.0), "chunk_size_frames": ParameterRange(minimum=1, maximum=10000)},
        description="Estimate one source-to-template transform for each video.",
    ),
    "apply_video_registration": PipelineStage(
        stage_id="apply_video_registration",
        label="Apply video registration",
        order=25,
        default_params={"chunk_size_frames": 64, "output_dtype": "float32"},
        param_ranges={"chunk_size_frames": ParameterRange(minimum=1, maximum=10000)},
        description="Apply registration transforms to full videos.",
    ),
    "grid_32x32_generate": PipelineStage(
        stage_id="grid_32x32_generate",
        label="Generate 32x32 grid",
        order=30,
        default_params={"rows": 32, "cols": 32, "bounds": "full_template_image", "cell_policy": "rectangular_image_coordinates"},
        param_ranges={"rows": ParameterRange(minimum=1, maximum=256), "cols": ParameterRange(minimum=1, maximum=256)},
        description="Create deterministic rectangular regions in template coordinates.",
    ),
    "grid_state_extract": PipelineStage(
        stage_id="grid_state_extract",
        label="Extract grid states",
        order=35,
        default_params={"features": ["mean_intensity"], "normalization": "per_video_robust_percentile", "pooling": "area", "chunk_size_frames": 64, "max_grid_state_bytes": 1000000000},
        param_ranges={"chunk_size_frames": ParameterRange(minimum=1, maximum=10000), "max_grid_state_bytes": ParameterRange(minimum=1000000, maximum=100000000000)},
        description="Pool registered videos into model-ready grid state tensors.",
    ),
    "grid_dynamics_dataset_build": PipelineStage(
        stage_id="grid_dynamics_dataset_build",
        label="Build grid dynamics dataset",
        order=70,
        default_params={"window_frames": 8, "prediction_horizon_frames": 1, "split_unit": "video", "split_method": "stratified_by_label", "train_fraction": 0.7, "val_fraction": 0.15, "test_fraction": 0.15},
        param_ranges={"window_frames": ParameterRange(minimum=1, maximum=1000), "prediction_horizon_frames": ParameterRange(minimum=1, maximum=1000)},
        description="Build recurrent windows and video-level splits from grid states.",
    ),
    "grid_autoencoder_train": PipelineStage(
        stage_id="grid_autoencoder_train",
        label="Train grid autoencoder",
        order=75,
        default_params={"latent_dim": 32, "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "device": "cpu", "seed": 7},
        param_ranges={"latent_dim": ParameterRange(minimum=1, maximum=1024), "epochs": ParameterRange(minimum=1, maximum=10000), "batch_size": ParameterRange(minimum=1, maximum=10000), "learning_rate": ParameterRange(minimum=0.0, maximum=10.0)},
        description="Train a small CNN autoencoder on 32x32 grid frames.",
    ),
    "latent_rnn_train": PipelineStage(
        stage_id="latent_rnn_train",
        label="Train latent GRU predictor",
        order=80,
        default_params={"window_frames": 8, "hidden_dim": 64, "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "device": "cpu", "seed": 7},
        param_ranges={"window_frames": ParameterRange(minimum=1, maximum=1000), "hidden_dim": ParameterRange(minimum=1, maximum=4096), "epochs": ParameterRange(minimum=1, maximum=10000), "batch_size": ParameterRange(minimum=1, maximum=10000), "learning_rate": ParameterRange(minimum=0.0, maximum=10.0)},
        description="Train a GRU to predict the next latent grid code and decode the next state.",
    ),
    "latent_classifier_train": PipelineStage(
        stage_id="latent_classifier_train",
        label="Train latent classifier",
        order=85,
        default_params={"classifier": "logistic_regression", "split_unit": "video", "evaluation": "stratified_kfold"},
        description="Classify neutral/left/right from video-level latent summaries.",
    ),
    "source_video_import": PipelineStage(
        stage_id="source_video_import",
        label="Source video import",
        order=10,
        required_params=("source",),
        description="Resolve the source imaging video or frame stack used by later stages.",
    ),
    "review_data_import": PipelineStage(
        stage_id="review_data_import",
        label="Review data import",
        order=10,
        required_params=("review_data",),
        description="Import an existing Neurobench review_data.json artifact.",
    ),
    "temporal_highpass_gaussian": PipelineStage(
        stage_id="temporal_highpass_gaussian",
        label="Temporal high-pass Gaussian",
        order=20,
        default_params={"sigma_frames": 6.0},
        param_ranges={"sigma_frames": ParameterRange(minimum=0.0, maximum=120.0)},
        description="Remove slow temporal baseline drift with a Gaussian high-pass filter.",
    ),
    "event_preserving_noise_suppression": PipelineStage(
        stage_id="event_preserving_noise_suppression",
        label="Event-preserving noise suppression",
        order=30,
        default_params={"spatial_sigma_px": 1.0, "temporal_window_frames": 3},
        param_ranges={
            "spatial_sigma_px": ParameterRange(minimum=0.0, maximum=10.0),
            "temporal_window_frames": ParameterRange(minimum=1, maximum=101),
        },
        description="Suppress noise while retaining localized calcium transients.",
    ),
    "spatial_gaussian": PipelineStage(
        stage_id="spatial_gaussian",
        label="Spatial Gaussian smoothing",
        order=30,
        default_params={"sigma_px": 0.8},
        param_ranges={"sigma_px": ParameterRange(minimum=0.0, maximum=10.0)},
        description="Apply spatial Gaussian smoothing before local filtering or CFAR.",
    ),
    "rigid_shift_estimate": PipelineStage(
        stage_id="rigid_shift_estimate",
        label="Rigid drift estimate",
        order=35,
        default_params={"max_shift_px": 4, "reference": "first"},
        param_ranges={"max_shift_px": ParameterRange(minimum=1, maximum=50)},
        description="Estimate simple rigid x/y frame drift for QC or registration-aware comparison.",
    ),
    "suite2p_import": PipelineStage(
        stage_id="suite2p_import",
        label="Suite2p import",
        order=40,
        required_params=("suite2p_dir",),
        description="Import Suite2p ROI and trace outputs.",
    ),
    "pmd_import": PipelineStage(
        stage_id="pmd_import",
        label="PMD import",
        order=40,
        required_params=("pmd_dir",),
        description="Import penalized matrix decomposition outputs.",
    ),
    "oasis_import": PipelineStage(
        stage_id="oasis_import",
        label="OASIS import",
        order=40,
        required_params=("traces",),
        description="Import OASIS deconvolution outputs.",
    ),
    "pmd_denoised_video_import": PipelineStage(
        stage_id="pmd_denoised_video_import",
        label="PMD denoised video import",
        order=40,
        required_params=("denoised_video",),
        description="Attach a PMD-denoised video artifact for downstream candidate generation.",
    ),
    "robust_positive_local_z": PipelineStage(
        stage_id="robust_positive_local_z",
        label="Robust positive local-z",
        order=50,
        default_params={"local_radius_px": 11, "epsilon": 1.0},
        param_ranges={
            "local_radius_px": ParameterRange(minimum=1, maximum=101),
            "epsilon": ParameterRange(minimum=0.0, maximum=100.0),
        },
        description="Compute robust positive local z-score evidence from a filtered video.",
    ),
    "gamma_cfar": PipelineStage(
        stage_id="gamma_cfar",
        label="Gamma CFAR",
        order=50,
        default_params={"pfa": 0.001, "guard_px": 2, "training_radius_px": 11, "epsilon": 1e-6},
        param_ranges={
            "pfa": ParameterRange(minimum=0.0, maximum=1.0),
            "guard_px": ParameterRange(minimum=0, maximum=100),
            "training_radius_px": ParameterRange(minimum=1, maximum=101),
            "epsilon": ParameterRange(minimum=0.0, maximum=100.0),
        },
        description="Apply gamma CFAR thresholding to candidate evidence.",
    ),
    "adaptive_ewma_z": PipelineStage(
        stage_id="adaptive_ewma_z",
        label="Adaptive EWMA z-score",
        order=50,
        default_params={"alpha": 0.02, "threshold_z": 3.0, "epsilon": 1.0},
        param_ranges={
            "alpha": ParameterRange(minimum=0.0001, maximum=1.0),
            "threshold_z": ParameterRange(minimum=0.0, maximum=20.0),
            "epsilon": ParameterRange(minimum=0.0, maximum=100.0),
        },
        description="Maintain streaming per-pixel mean/variance estimates and emit positive z-score candidate masks.",
    ),
    "adaptive_gamma_cfar": PipelineStage(
        stage_id="adaptive_gamma_cfar",
        label="Adaptive Gamma CFAR",
        order=50,
        default_params={"pfa": 0.001, "guard_px": 2, "training_radius_px": 11, "update_alpha": 0.02},
        param_ranges={
            "pfa": ParameterRange(minimum=0.0, maximum=1.0),
            "guard_px": ParameterRange(minimum=0, maximum=100),
            "training_radius_px": ParameterRange(minimum=1, maximum=101),
            "update_alpha": ParameterRange(minimum=0.0001, maximum=1.0),
        },
        description="Plan a streaming CFAR detector with local training statistics and adaptive background updates.",
    ),
    "candidate_event_pipeline": PipelineStage(
        stage_id="candidate_event_pipeline",
        label="Candidate event pipeline",
        order=50,
        default_params={"event_threshold_z": 2.4, "min_area_px": 4},
        param_ranges={
            "event_threshold_z": ParameterRange(minimum=0.0, maximum=20.0),
            "min_area_px": ParameterRange(minimum=1, maximum=100000),
        },
        description="Detect candidate calcium events and ROI discovery suggestions.",
    ),
    "component_filter": PipelineStage(
        stage_id="component_filter",
        label="Component extraction",
        order=55,
        default_params={
            "seed_z": 2.0,
            "grow_z": 1.1,
            "min_area_px": 4,
            "max_area_px": 260,
            "support_min_frames": 1,
            "projection_blob_z": 0.0,
        },
        param_ranges={
            "seed_z": ParameterRange(minimum=0.0, maximum=20.0),
            "grow_z": ParameterRange(minimum=0.0, maximum=20.0),
            "min_area_px": ParameterRange(minimum=1, maximum=100000),
            "max_area_px": ParameterRange(minimum=1, maximum=100000),
            "support_min_frames": ParameterRange(minimum=1, maximum=100000),
            "projection_blob_z": ParameterRange(minimum=0.0, maximum=20.0),
        },
        description="Extract connected component ROI candidates from evidence maps, persistent green projections, or temporally supported candidate masks.",
    ),
    "local_background_ring": PipelineStage(
        stage_id="local_background_ring",
        label="Local background ring",
        order=58,
        default_params={"outer_radius_px": 15, "neuropil_weight": 0.7},
        param_ranges={
            "outer_radius_px": ParameterRange(minimum=1, maximum=1000),
            "neuropil_weight": ParameterRange(minimum=0.0, maximum=5.0),
        },
        description="Extract ROI traces with a local background or neuropil ring.",
    ),
    "trace_event_scoring": PipelineStage(
        stage_id="trace_event_scoring",
        label="Trace event scoring",
        order=60,
        required_params=("event_threshold_z",),
        default_params={"sustained_z": 1.2, "tonic_z": 2.0, "peak_window_frames": 1},
        param_ranges={
            "event_threshold_z": ParameterRange(minimum=0.0, maximum=20.0),
            "sustained_z": ParameterRange(minimum=0.0, maximum=20.0),
            "tonic_z": ParameterRange(minimum=0.0, maximum=20.0),
            "peak_window_frames": ParameterRange(minimum=0, maximum=1000),
        },
        description="Score candidate trace events and optional sustained or tonic activity states.",
    ),
    "robust_kalman_positive_innovation": PipelineStage(
        stage_id="robust_kalman_positive_innovation",
        label="Kalman positive innovation events",
        order=60,
        default_params={"event_threshold_z": 2.4, "kalman_gain": 0.06, "spike_gain": 0.008},
        param_ranges={
            "event_threshold_z": ParameterRange(minimum=0.0, maximum=20.0),
            "kalman_gain": ParameterRange(minimum=0.0, maximum=1.0),
            "spike_gain": ParameterRange(minimum=0.0, maximum=1.0),
        },
        description="Call candidate events from positive innovations over a robust baseline.",
    ),
    "oasis_deconvolution_import": PipelineStage(
        stage_id="oasis_deconvolution_import",
        label="OASIS deconvolution import",
        order=60,
        default_params={"array_key": "spikes"},
        description="Attach OASIS deconvolved traces or event evidence.",
    ),
    "heuristic_priority_v1": PipelineStage(
        stage_id="heuristic_priority_v1",
        label="Heuristic priority ranking",
        order=80,
        default_params={
            "local_correlation_weight": 0.2,
            "event_support_weight": 0.2,
            "artifact_weight": -0.15,
        },
        param_ranges={
            "local_correlation_weight": ParameterRange(minimum=-5.0, maximum=5.0),
            "event_support_weight": ParameterRange(minimum=-5.0, maximum=5.0),
            "artifact_weight": ParameterRange(minimum=-5.0, maximum=5.0),
        },
        description="Rank candidates for human review using transparent feature weights.",
    ),
    "generate_neuron_review_app": PipelineStage(
        stage_id="generate_neuron_review_app",
        label="Generate neuron review app",
        order=70,
        default_params={"include_discovery": True},
        description="Build review_data.json, summary tables, frames, and dashboard assets.",
    ),
    "flat_field_background": PipelineStage(
        stage_id="flat_field_background",
        label="Flat-field/background correction",
        order=18,
        default_params={"local_radius_px": 31, "update_alpha": 0.02},
        param_ranges={
            "local_radius_px": ParameterRange(minimum=3, maximum=301),
            "update_alpha": ParameterRange(minimum=0.0001, maximum=1.0),
        },
        description="Estimate and subtract uneven spatial background before candidate scoring.",
    ),
    "photobleach_correction": PipelineStage(
        stage_id="photobleach_correction",
        label="Photobleach correction",
        order=19,
        default_params={"update_alpha": 0.01},
        param_ranges={"update_alpha": ParameterRange(minimum=0.0001, maximum=1.0)},
        description="Track and compensate slow global fluorescence decay.",
    ),
    "temporal_hampel": PipelineStage(
        stage_id="temporal_hampel",
        label="Temporal Hampel despiking",
        order=29,
        default_params={"temporal_window_frames": 5, "threshold_z": 4.0},
        param_ranges={
            "temporal_window_frames": ParameterRange(minimum=3, maximum=101),
            "threshold_z": ParameterRange(minimum=0.0, maximum=20.0),
        },
        description="Reject isolated impulse noise using a local temporal median/MAD window.",
    ),
    "kalman_smoother": PipelineStage(
        stage_id="kalman_smoother",
        label="Kalman trace smoother",
        order=59,
        default_params={"kalman_gain": 0.06, "spike_gain": 0.008},
        param_ranges={
            "kalman_gain": ParameterRange(minimum=0.0, maximum=1.0),
            "spike_gain": ParameterRange(minimum=0.0, maximum=1.0),
        },
        description="Smooth ROI traces while preserving positive innovations for event review.",
    ),
    "local_temporal_correlation": PipelineStage(
        stage_id="local_temporal_correlation",
        label="Local temporal correlation evidence",
        order=61,
        default_params={"local_radius_px": 3},
        param_ranges={"local_radius_px": ParameterRange(minimum=1, maximum=25)},
        description="Score whether nearby pixels have coherent temporal activity.",
    ),
    "event_triggered_footprint": PipelineStage(
        stage_id="event_triggered_footprint",
        label="Event-triggered footprint evidence",
        order=62,
        default_params={"temporal_window_frames": 7},
        param_ranges={"temporal_window_frames": ParameterRange(minimum=1, maximum=101)},
        description="Average frames around candidate events to check for compact multi-frame footprints.",
    ),
    "background_heterogeneity_map": PipelineStage(
        stage_id="background_heterogeneity_map",
        label="Background heterogeneity map",
        order=47,
        default_params={"local_radius_px": 21},
        param_ranges={"local_radius_px": ParameterRange(minimum=3, maximum=301)},
        description="Map spatially clustered background that can explain false positives.",
    ),
    "saturation_blob_map": PipelineStage(
        stage_id="saturation_blob_map",
        label="Saturation and bright-blob map",
        order=48,
        default_params={"threshold_z": 6.0},
        param_ranges={"threshold_z": ParameterRange(minimum=0.0, maximum=50.0)},
        description="Detect persistent bright or saturated structures for artifact triage.",
    ),
    "motion_sensitivity_map": PipelineStage(
        stage_id="motion_sensitivity_map",
        label="Motion sensitivity map",
        order=49,
        default_params={"max_shift_px": 4},
        param_ranges={"max_shift_px": ParameterRange(minimum=1, maximum=50)},
        description="Estimate which candidate regions are sensitive to drift or frame motion.",
    ),
    "log_dog_blob_candidates": PipelineStage(
        stage_id="log_dog_blob_candidates",
        label="LoG/DoG soma-scale blob candidates",
        order=54,
        default_params={"sigma_px": 2.0, "threshold_z": 2.0},
        param_ranges={
            "sigma_px": ParameterRange(minimum=0.1, maximum=20.0),
            "threshold_z": ParameterRange(minimum=0.0, maximum=20.0),
        },
        description="Add blob-like candidate ROIs at expected soma scale.",
    ),
    "watershed_split_merge": PipelineStage(
        stage_id="watershed_split_merge",
        label="Watershed split/merge suggestions",
        order=56,
        default_params={"min_area_px": 8, "max_area_px": 260},
        param_ranges={
            "min_area_px": ParameterRange(minimum=1, maximum=100000),
            "max_area_px": ParameterRange(minimum=1, maximum=100000),
        },
        description="Suggest split and merge actions for overlapping or fragmented candidates.",
    ),
    "ensemble_union": PipelineStage(
        stage_id="ensemble_union",
        label="Candidate ensemble union",
        order=76,
        default_params={"include_discovery": True},
        description="Union candidates from multiple architecture runs while preserving provenance.",
    ),
    "consensus_stability_scoring": PipelineStage(
        stage_id="consensus_stability_scoring",
        label="Consensus and stability scoring",
        order=78,
        default_params={"local_correlation_weight": 0.25, "event_support_weight": 0.25},
        param_ranges={
            "local_correlation_weight": ParameterRange(minimum=-5.0, maximum=5.0),
            "event_support_weight": ParameterRange(minimum=-5.0, maximum=5.0),
        },
        description="Score candidates by method agreement and stability across nearby parameters.",
    ),
    "artifact_classifier_v1": PipelineStage(
        stage_id="artifact_classifier_v1",
        label="Artifact classifier v1",
        order=79,
        default_params={"artifact_weight": -0.2},
        param_ranges={"artifact_weight": ParameterRange(minimum=-5.0, maximum=5.0)},
        description="Rank likely vessels, borders, bright blobs, motion artifacts, and impulse noise.",
    ),
    "active_learning_ranker": PipelineStage(
        stage_id="active_learning_ranker",
        label="Active-learning review ranker",
        order=81,
        default_params={"local_correlation_weight": 0.2, "event_support_weight": 0.2, "artifact_weight": -0.15},
        param_ranges={
            "local_correlation_weight": ParameterRange(minimum=-5.0, maximum=5.0),
            "event_support_weight": ParameterRange(minimum=-5.0, maximum=5.0),
            "artifact_weight": ParameterRange(minimum=-5.0, maximum=5.0),
        },
        description="Use reviewed labels to prioritize the most informative remaining candidates.",
    ),
}


def catalog_as_dict(*, runner_stage_ids: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Return a JSON-serializable copy of the stage catalog."""

    runners = set(LOCAL_RUNNER_STAGE_IDS if runner_stage_ids is None else runner_stage_ids)
    catalog = {stage_id: stage.as_dict() for stage_id, stage in STAGE_CATALOG.items()}
    for stage_id, entry in catalog.items():
        entry["runner_available"] = stage_id in runners
        entry["locally_runnable"] = entry["availability"] == "implemented" and entry["runner_available"]
    return catalog


def stage_ids() -> tuple[str, ...]:
    return tuple(STAGE_CATALOG)


def get_stage(stage_id: str) -> PipelineStage:
    try:
        return STAGE_CATALOG[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline stage_id '{stage_id}'.") from exc


def is_structured_pipeline(pipeline: Sequence[Mapping[str, Any]] | None) -> bool:
    """Return True when any step uses the structured pipeline contract."""

    return any("id" in step or "stage_id" in step or "stage" in step for step in (pipeline or []))


def normalize_pipeline(
    pipeline: Sequence[Mapping[str, Any]] | None,
    *,
    require_structured: bool = False,
) -> list[dict[str, Any]]:
    """Validate a pipeline and merge catalog defaults into structured steps.

    Legacy architecture-run steps that only use ``name`` are preserved as-is unless
    ``require_structured`` is set. Structured steps must contain unique ``id``
    values, known ``stage_id`` values, required params, and nondecreasing catalog
    order.
    """

    if pipeline is None:
        if require_structured:
            raise ValueError("Pipeline is required.")
        return []
    if not isinstance(pipeline, Sequence) or isinstance(pipeline, (str, bytes, bytearray)):
        raise ValueError("Pipeline must be an array.")

    steps: list[dict[str, Any]] = []
    for index, step in enumerate(pipeline):
        if not isinstance(step, Mapping):
            raise ValueError(f"Pipeline step at index {index} must be an object.")
        steps.append(dict(step))
    structured = is_structured_pipeline(steps)
    if require_structured and not structured:
        raise ValueError("Pipeline must use structured steps with 'id' and 'stage_id'.")
    if not structured:
        return steps

    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    previous_order: int | None = None
    previous_stage_id = ""
    for index, step in enumerate(steps):
        step_id = step.get("id")
        stage_id = step.get("stage_id", step.get("stage"))
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"Pipeline step at index {index} is missing required string 'id'.")
        if step_id in seen_ids:
            raise ValueError(f"Duplicate pipeline step id '{step_id}'.")
        seen_ids.add(step_id)
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError(f"Pipeline step '{step_id}' is missing required string 'stage_id'.")

        stage = get_stage(stage_id)
        if previous_order is not None and stage.order < previous_order:
            raise ValueError(
                f"Pipeline stage '{stage_id}' is out of order after '{previous_stage_id}'."
            )

        raw_params = step.get("params")
        if raw_params is not None and not isinstance(raw_params, Mapping):
            raise ValueError(f"Pipeline step '{step_id}' params must be an object.")
        params = stage.merged_params(raw_params)
        for required_param in stage.required_params:
            if required_param not in params or params[required_param] is None:
                raise ValueError(f"Pipeline stage '{stage_id}' is missing required param '{required_param}'.")
        for param_name, param_range in dict(stage.param_ranges or {}).items():
            if param_name in params and params[param_name] is not None:
                param_range.validate(stage_id, param_name, params[param_name])

        normalized_step = dict(step)
        normalized_step["stage_id"] = stage_id
        normalized_step["params"] = params
        normalized.append(normalized_step)
        previous_order = stage.order
        previous_stage_id = stage_id
    return normalized


def validate_pipeline(
    pipeline: Sequence[Mapping[str, Any]] | None,
    *,
    require_structured: bool = False,
) -> list[dict[str, Any]]:
    """Validate and normalize a pipeline.

    This is an explicit validation entrypoint; it returns the same normalized
    structure as ``normalize_pipeline`` so callers can keep merged defaults.
    """

    return normalize_pipeline(pipeline, require_structured=require_structured)
