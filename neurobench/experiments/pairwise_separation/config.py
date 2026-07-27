"""Strict version-1 manifest for pairwise source separation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class PairwiseConfigError(ValueError):
    pass


def _strict(payload: Mapping[str, Any], allowed: set[str], scope: str) -> dict[str, Any]:
    values = dict(payload)
    unknown = set(values) - allowed
    if unknown:
        raise PairwiseConfigError(f"Unknown {scope} fields: {', '.join(sorted(unknown))}")
    return values


@dataclass(frozen=True)
class FrameConfig:
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    frame_period_ms: float


@dataclass(frozen=True)
class PreprocessingConfig:
    lag_frames: int
    spatial_sigma_px: float
    temporal_mode: str
    temporal_ema_span_frames: float
    motion_mode: str
    run_integer_shift_sensitivity: bool
    max_shift_px: int


@dataclass(frozen=True)
class SamplingConfig:
    primary_policy: str
    screen_samples: int
    confirm_samples: int
    screen_angle_step_degrees: float
    refine_half_width_degrees: float
    refine_angle_step_degrees: float
    pairwise_diagnostic_frames_ui: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class MethodConfig:
    fixed_binary_difference: dict[str, Any]
    adaptive_binary_difference: dict[str, Any]
    infomax_tanh_ica: dict[str, Any]
    cs_parzen_ica: dict[str, Any]
    shared_background_nmf: dict[str, Any]


@dataclass(frozen=True)
class ThresholdConfig:
    z_thresholds: tuple[float, ...]
    primary_z_threshold: float
    one_sided_positive: bool
    minimum_component_pixels: tuple[int, ...]
    write_binary_tiff: bool
    quiet_mad_floor_percentile: float


@dataclass(frozen=True)
class EvaluationConfig:
    binary_temporal_pool: str
    nms_distance_px: int
    primary_match_radius_px: int
    match_radii_px: tuple[int, ...]
    quiet_false_peaks_per_map: float
    capacity_reference_lane: str
    candidate_review_rows: int


@dataclass(frozen=True)
class ResourceConfig:
    cpu_threads: int
    frame_chunk: int
    kernel_block_rows: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int


@dataclass(frozen=True)
class PairwiseSeparationConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    source_tiff: Path
    labels_tsv: Path
    design_document: Path
    output_dir: Path
    frames: FrameConfig
    preprocessing: PreprocessingConfig
    sampling: SamplingConfig
    methods: MethodConfig
    thresholding: ThresholdConfig
    evaluation: EvaluationConfig
    resources: ResourceConfig

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseSeparationConfig":
        source = Path(path).resolve()
        raw = _strict(json.loads(source.read_text(encoding="utf-8")), {
            "schema_version", "experiment_id", "source_video", "source_tiff", "labels_tsv",
            "design_document", "output_dir", "frames", "preprocessing", "sampling", "methods",
            "thresholding", "evaluation", "resources",
        }, "top-level")
        root = source.parent
        frames = _strict(raw["frames"], set(FrameConfig.__dataclass_fields__), "frames")
        preprocessing = _strict(raw["preprocessing"], set(PreprocessingConfig.__dataclass_fields__), "preprocessing")
        sampling = _strict(raw["sampling"], set(SamplingConfig.__dataclass_fields__), "sampling")
        thresholding = _strict(raw["thresholding"], set(ThresholdConfig.__dataclass_fields__), "thresholding")
        evaluation = _strict(raw["evaluation"], set(EvaluationConfig.__dataclass_fields__), "evaluation")
        resources = _strict(raw["resources"], set(ResourceConfig.__dataclass_fields__), "resources")
        methods = _strict(raw["methods"], set(MethodConfig.__dataclass_fields__), "methods")
        method_fields = {
            "fixed_binary_difference": {"enabled"},
            "adaptive_binary_difference": {"enabled", "alpha_min", "alpha_max", "trim_fraction", "refinement_iterations"},
            "infomax_tanh_ica": {"enabled", "max_iterations", "learning_rate", "tolerance", "initial_angles_degrees"},
            "cs_parzen_ica": {"enabled", "bandwidth", "kernel_block_rows"},
            "shared_background_nmf": {"enabled", "activity_l1", "max_iterations", "tolerance"},
        }
        for key, allowed in method_fields.items():
            methods[key] = _strict(methods[key], allowed, f"methods.{key}")
        config = cls(
            schema_version=int(raw["schema_version"]), experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            source_tiff=(root / raw["source_tiff"]).resolve(),
            labels_tsv=(root / raw["labels_tsv"]).resolve(),
            design_document=(root / raw["design_document"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            frames=FrameConfig(**{k: (float(v) if k == "frame_period_ms" else int(v)) for k, v in frames.items()}),
            preprocessing=PreprocessingConfig(
                lag_frames=int(preprocessing["lag_frames"]), spatial_sigma_px=float(preprocessing["spatial_sigma_px"]),
                temporal_mode=str(preprocessing["temporal_mode"]), temporal_ema_span_frames=float(preprocessing["temporal_ema_span_frames"]),
                motion_mode=str(preprocessing["motion_mode"]), run_integer_shift_sensitivity=bool(preprocessing["run_integer_shift_sensitivity"]),
                max_shift_px=int(preprocessing["max_shift_px"]),
            ),
            sampling=SamplingConfig(
                primary_policy=str(sampling["primary_policy"]), screen_samples=int(sampling["screen_samples"]),
                confirm_samples=int(sampling["confirm_samples"]), screen_angle_step_degrees=float(sampling["screen_angle_step_degrees"]),
                refine_half_width_degrees=float(sampling["refine_half_width_degrees"]),
                refine_angle_step_degrees=float(sampling["refine_angle_step_degrees"]),
                pairwise_diagnostic_frames_ui=tuple(int(x) for x in sampling["pairwise_diagnostic_frames_ui"]),
                seed=int(sampling["seed"]),
            ),
            methods=MethodConfig(**methods),
            thresholding=ThresholdConfig(
                z_thresholds=tuple(float(x) for x in thresholding["z_thresholds"]),
                primary_z_threshold=float(thresholding["primary_z_threshold"]),
                one_sided_positive=bool(thresholding["one_sided_positive"]),
                minimum_component_pixels=tuple(int(x) for x in thresholding["minimum_component_pixels"]),
                write_binary_tiff=bool(thresholding["write_binary_tiff"]),
                quiet_mad_floor_percentile=float(thresholding["quiet_mad_floor_percentile"]),
            ),
            evaluation=EvaluationConfig(
                binary_temporal_pool=str(evaluation["binary_temporal_pool"]),
                nms_distance_px=int(evaluation["nms_distance_px"]),
                primary_match_radius_px=int(evaluation["primary_match_radius_px"]),
                match_radii_px=tuple(int(x) for x in evaluation["match_radii_px"]),
                quiet_false_peaks_per_map=float(evaluation["quiet_false_peaks_per_map"]),
                capacity_reference_lane=str(evaluation["capacity_reference_lane"]),
                candidate_review_rows=int(evaluation["candidate_review_rows"]),
            ),
            resources=ResourceConfig(**{k: int(v) for k, v in resources.items()}),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise PairwiseConfigError("schema_version must be exactly 1")
        f, p, s, t, e, r = self.frames, self.preprocessing, self.sampling, self.thresholding, self.evaluation, self.resources
        if not (1 <= f.review_start_ui <= f.quiet_start_ui <= f.quiet_end_ui <= f.review_end_ui) or f.frame_period_ms <= 0:
            raise PairwiseConfigError("invalid review/quiet frame contract")
        if f.quiet_start_ui != f.review_start_ui:
            raise PairwiseConfigError("version 1 requires the quiet interval to begin at review_start_ui")
        quiet_frames = f.quiet_end_ui - f.quiet_start_ui + 1
        if not 1 <= p.lag_frames < quiet_frames or quiet_frames - p.lag_frames < 50:
            raise PairwiseConfigError("quiet interval must contain at least 50 defined derivative frames")
        if p.temporal_mode != "causal_ema" or p.temporal_ema_span_frames < 1 or p.spatial_sigma_px < 0:
            raise PairwiseConfigError("version 1 requires causal_ema and nonnegative smoothing")
        if p.motion_mode != "none" or not 0 <= p.max_shift_px <= 8:
            raise PairwiseConfigError("primary version-1 motion_mode is none; max_shift_px must be bounded")
        if s.primary_policy != "uniform_anatomy" or not 16 <= s.screen_samples <= s.confirm_samples <= 16384:
            raise PairwiseConfigError("invalid bounded sampling contract")
        if 90 / s.screen_angle_step_degrees % 1 > 1e-9 or s.refine_angle_step_degrees <= 0 or s.refine_half_width_degrees <= 0:
            raise PairwiseConfigError("angle steps must tile/bound [0,90)")
        if len(set(s.pairwise_diagnostic_frames_ui)) != len(s.pairwise_diagnostic_frames_ui):
            raise PairwiseConfigError("diagnostic frames must be unique")
        if t.primary_z_threshold not in t.z_thresholds or not t.one_sided_positive:
            raise PairwiseConfigError("primary threshold must be declared and one-sided positive")
        if not t.minimum_component_pixels or min(t.minimum_component_pixels) != 1:
            raise PairwiseConfigError("minimum_component_pixels must include primary value 1")
        if e.primary_match_radius_px != 6 or 6 not in e.match_radii_px:
            raise PairwiseConfigError("Raw Direct primary match radius is frozen at six pixels")
        if e.nms_distance_px < 1 or e.quiet_false_peaks_per_map <= 0 or e.capacity_reference_lane != "raw_direct":
            raise PairwiseConfigError("invalid evaluation contract")
        if not 1 <= r.cpu_threads <= 8 or not 1 <= r.frame_chunk <= 128 or r.kernel_block_rows > s.confirm_samples:
            raise PairwiseConfigError("resource bounds exceed the experiment CLI contract")
        if min(r.max_ram_mib, r.min_free_disk_mib, r.max_output_mib) <= 0:
            raise PairwiseConfigError("resource caps must be positive")
        enabled = [getattr(self.methods, name).get("enabled") for name in MethodConfig.__dataclass_fields__]
        if not all(isinstance(x, bool) for x in enabled) or not any(enabled):
            raise PairwiseConfigError("method enabled flags must be booleans and at least one method enabled")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_video", "source_tiff", "labels_tsv", "design_document", "output_dir"):
            payload[key] = str(payload[key])
        return payload
