"""Strict, versioned configuration for the MSLN/MS-ICA experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class MSLNMSICAConfigError(ValueError):
    pass


def _strict(raw: Any, cls: type, scope: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MSLNMSICAConfigError(f"{scope} must be a mapping")
    values = dict(raw)
    allowed = set(cls.__dataclass_fields__)
    unknown, missing = set(values) - allowed, allowed - set(values)
    if unknown:
        raise MSLNMSICAConfigError(f"Unknown {scope} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise MSLNMSICAConfigError(f"Missing {scope} fields: {', '.join(sorted(missing))}")
    return values


def _tuple2(value: Any, scope: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MSLNMSICAConfigError(f"{scope} must contain exactly two integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class SourceConfig:
    movie_path: Path
    labels_path: Path
    baseline_evidence_dir: Path | None
    axes: str
    ui_one_based: bool
    review_interval_ui: tuple[int, int]
    quiet_interval_ui: tuple[int, int]
    burst_intervals_ui: dict[int, tuple[int, int]]


@dataclass(frozen=True)
class PreprocessingConfig:
    input_domain: str
    gaussian_sigma_px: float
    variance_stabilization: str
    motion_correction: str
    preserve_raw: bool


@dataclass(frozen=True)
class SpatialConfig:
    enabled: bool
    outer_widths_px: tuple[int, ...]
    guard_widths_px: tuple[int, ...]
    primary_estimator: str
    robust_ablation: bool
    scale_floor_percentile: float


@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool
    windows_frames: tuple[int, ...]
    guard_frames: int
    causal: bool
    primary_estimator: str
    scale_floor_percentile: float


@dataclass(frozen=True)
class STPairConfig:
    spatial_outer_width_px: int
    temporal_window_frames: int


@dataclass(frozen=True)
class SpatiotemporalConfig:
    enabled: bool
    mode: str
    pairs: tuple[STPairConfig, ...]


@dataclass(frozen=True)
class ContextsConfig:
    signed: bool
    spatial: SpatialConfig
    temporal: TemporalConfig
    spatiotemporal: SpatiotemporalConfig


@dataclass(frozen=True)
class SamplingConfig:
    seed: int
    per_context_screen_samples: int
    per_context_confirmation_samples: int
    cross_context_max_samples: int
    reuse_indices_across_baselines: bool
    time_block_length_frames: int
    bootstrap_replicates: int
    heldout_guard_frames: int


@dataclass(frozen=True)
class PerContextICAConfig:
    enabled: bool
    primary_objective: str
    run_fastica_ablation: bool
    parzen_bandwidth: float
    eigenvalue_floor_ratio: float
    angle_range_degrees: tuple[float, float]
    coarse_step_degrees: float
    refine_half_width_degrees: float
    refine_step_degrees: float
    kernel_block_rows: int


@dataclass(frozen=True)
class CrossContextConfig:
    modes: tuple[str, ...]
    cs_parzen_screen_only: bool
    max_components: int
    true_isa_enabled: bool
    groups: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class EnergyConfig:
    quiet_standardization: str
    mappings: tuple[str, ...]
    bounded_kappa_z: float
    tail_log_base: int
    tail_add_one_smoothing: bool
    spatial_pool_sigma_px: float


@dataclass(frozen=True)
class RoutingConfig:
    modes: tuple[str, ...]
    softmax_temperature: float
    complexity_penalty: float
    compact_minus_broad_weight: float


@dataclass(frozen=True)
class FusionConfig:
    primary_output: str
    evaluate_product_interaction: bool
    visualization_floor_beta: float
    bounded_gate_kappa: float
    linear_ranker_enabled: bool


@dataclass(frozen=True)
class EvaluationConfig:
    primary_protocol: str
    secondary_crossfit_selector: bool
    candidate_budgets: tuple[int, ...]
    primary_budgets: tuple[int, ...]
    nms_distance_px: int
    match_radius_px: int
    identical_proposal_evaluation: bool
    native_proposal_evaluation: bool
    synthetic_fixture_evaluation: bool


@dataclass(frozen=True)
class ComputeConfig:
    device: str
    cpu_threads: int
    max_worker_processes: int
    max_parallel_folds: int
    frame_chunk: int
    gpu_frame_batch: int
    context_batch: int
    kernel_dtype: str
    accumulator_dtype: str
    max_peak_ram_gb: float
    max_peak_vram_gb: float
    render_selected_videos_only: bool


@dataclass(frozen=True)
class OutputsConfig:
    root_dir: Path
    save_all_context_maps: bool
    save_all_latent_maps: bool
    selected_video_count: int
    representative_frames_ui: tuple[int, ...]


@dataclass(frozen=True)
class MSLNMSICAConfig:
    schema_version: int
    experiment_id: str
    source: SourceConfig
    preprocessing: PreprocessingConfig
    contexts: ContextsConfig
    sampling: SamplingConfig
    per_context_ica: PerContextICAConfig
    cross_context: CrossContextConfig
    energy: EnergyConfig
    routing: RoutingConfig
    fusion: FusionConfig
    evaluation: EvaluationConfig
    compute: ComputeConfig
    outputs: OutputsConfig
    fold_ids: tuple[int, ...]
    config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "MSLNMSICAConfig":
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise MSLNMSICAConfigError("configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__) - {"config_path"}
        unknown, missing = set(raw) - allowed, allowed - set(raw)
        if unknown:
            raise MSLNMSICAConfigError(f"Unknown top-level fields: {', '.join(sorted(unknown))}")
        if missing:
            raise MSLNMSICAConfigError(f"Missing top-level fields: {', '.join(sorted(missing))}")
        root = config_path.parent
        source = _strict(raw["source"], SourceConfig, "source")
        for key in ("movie_path", "labels_path"):
            source[key] = (root / Path(source[key])).resolve()
        evidence = source["baseline_evidence_dir"]
        source["baseline_evidence_dir"] = None if evidence is None else (root / Path(evidence)).resolve()
        source["review_interval_ui"] = _tuple2(source["review_interval_ui"], "review_interval_ui")
        source["quiet_interval_ui"] = _tuple2(source["quiet_interval_ui"], "quiet_interval_ui")
        source["burst_intervals_ui"] = {
            int(key): _tuple2(value, f"burst {key}") for key, value in source["burst_intervals_ui"].items()
        }
        preprocessing = PreprocessingConfig(**_strict(raw["preprocessing"], PreprocessingConfig, "preprocessing"))
        contexts_raw = _strict(raw["contexts"], ContextsConfig, "contexts")
        spatial_raw = _strict(contexts_raw["spatial"], SpatialConfig, "contexts.spatial")
        spatial_raw["outer_widths_px"] = tuple(int(x) for x in spatial_raw["outer_widths_px"])
        spatial_raw["guard_widths_px"] = tuple(int(x) for x in spatial_raw["guard_widths_px"])
        temporal_raw = _strict(contexts_raw["temporal"], TemporalConfig, "contexts.temporal")
        temporal_raw["windows_frames"] = tuple(int(x) for x in temporal_raw["windows_frames"])
        st_raw = _strict(contexts_raw["spatiotemporal"], SpatiotemporalConfig, "contexts.spatiotemporal")
        st_raw["pairs"] = tuple(STPairConfig(**_strict(item, STPairConfig, "spatiotemporal pair")) for item in st_raw["pairs"])
        contexts = ContextsConfig(
            signed=bool(contexts_raw["signed"]), spatial=SpatialConfig(**spatial_raw),
            temporal=TemporalConfig(**temporal_raw), spatiotemporal=SpatiotemporalConfig(**st_raw),
        )
        simple_types = {
            "sampling": SamplingConfig, "per_context_ica": PerContextICAConfig,
            "cross_context": CrossContextConfig, "energy": EnergyConfig,
            "routing": RoutingConfig, "fusion": FusionConfig,
            "evaluation": EvaluationConfig, "compute": ComputeConfig,
            "outputs": OutputsConfig,
        }
        sections: dict[str, Any] = {}
        for name, section_type in simple_types.items():
            values = _strict(raw[name], section_type, name)
            for key in ("modes", "mappings", "candidate_budgets", "primary_budgets", "representative_frames_ui"):
                if key in values:
                    values[key] = tuple(values[key])
            if name == "per_context_ica":
                values["angle_range_degrees"] = tuple(float(x) for x in values["angle_range_degrees"])
            if name == "cross_context":
                values["groups"] = {str(k): tuple(v) for k, v in values["groups"].items()}
            if name == "outputs":
                values["root_dir"] = (root / Path(values["root_dir"])).resolve()
            sections[name] = section_type(**values)
        config = cls(
            schema_version=int(raw["schema_version"]), experiment_id=str(raw["experiment_id"]),
            source=SourceConfig(**source), preprocessing=preprocessing, contexts=contexts,
            fold_ids=tuple(int(x) for x in raw["fold_ids"]), config_path=config_path,
            **sections,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id:
            raise MSLNMSICAConfigError("schema_version must be 1 and experiment_id non-empty")
        if self.source.axes != "TYX" or not self.source.ui_one_based:
            raise MSLNMSICAConfigError("v1 requires TYX axes and one-based UI intervals")
        if not self.preprocessing.preserve_raw or self.preprocessing.input_domain not in {"raw", "raw_smoothed"}:
            raise MSLNMSICAConfigError("raw preservation and a raw input domain are mandatory")
        spatial = self.contexts.spatial
        if len(spatial.outer_widths_px) != len(spatial.guard_widths_px) or len(set(spatial.outer_widths_px)) != len(spatial.outer_widths_px):
            raise MSLNMSICAConfigError("spatial outer/guard arrays must be paired and unique")
        for outer, guard in zip(spatial.outer_widths_px, spatial.guard_widths_px):
            if outer not in range(3, 32, 2) or guard < 1 or guard % 2 != 1 or guard >= outer:
                raise MSLNMSICAConfigError("invalid spatial support or guard")
        temporal = self.contexts.temporal
        if len(set(temporal.windows_frames)) != len(temporal.windows_frames) or any(w <= temporal.guard_frames for w in temporal.windows_frames):
            raise MSLNMSICAConfigError("temporal windows must be unique and exceed the guard")
        if not self.contexts.signed or not temporal.causal:
            raise MSLNMSICAConfigError("v1 requires signed contexts and causal temporal normalization")
        if spatial.primary_estimator != "mean_std" or temporal.primary_estimator != "mean_std":
            raise MSLNMSICAConfigError("mean/std is the frozen primary estimator")
        if self.contexts.spatiotemporal.enabled and (not spatial.enabled or not temporal.enabled):
            raise MSLNMSICAConfigError("spatiotemporal contexts require enabled spatial and temporal banks")
        if self.contexts.spatiotemporal.mode not in {"temporal_then_spatial", "spatial_then_temporal"}:
            raise MSLNMSICAConfigError("unknown spatiotemporal composition order")
        for pair in self.contexts.spatiotemporal.pairs:
            if pair.spatial_outer_width_px not in spatial.outer_widths_px or pair.temporal_window_frames not in temporal.windows_frames:
                raise MSLNMSICAConfigError("spatiotemporal pair references a disabled context")
        context_count = (len(spatial.outer_widths_px) if spatial.enabled else 0) + (len(temporal.windows_frames) if temporal.enabled else 0) + (len(self.contexts.spatiotemporal.pairs) if self.contexts.spatiotemporal.enabled else 0)
        if context_count < 1 or context_count > 8:
            raise MSLNMSICAConfigError("standard context count must be between 1 and 8")
        if self.cross_context.true_isa_enabled:
            raise MSLNMSICAConfigError("true ISA is stage-gated and disabled in v1")
        if set(self.cross_context.modes) - {"identity", "pca", "fastica", "group_energy"}:
            raise MSLNMSICAConfigError("unknown cross-context mode")
        if self.evaluation.primary_protocol != "fixed_unsupervised":
            raise MSLNMSICAConfigError("v1 primary protocol must be fixed_unsupervised")
        if self.fusion.linear_ranker_enabled:
            raise MSLNMSICAConfigError("label-trained linear routing is disabled in the primary track")
        if self.compute.cpu_threads < 1 or self.compute.cpu_threads > 4 or self.compute.max_worker_processes != 1 or self.compute.max_parallel_folds != 1:
            raise MSLNMSICAConfigError("standard compute profile is at most 4 threads, one worker, one fold")
        if self.compute.max_peak_ram_gb > 24 or self.compute.max_peak_vram_gb > 8 or self.compute.context_batch > 1:
            raise MSLNMSICAConfigError("standard RAM/VRAM/context caps cannot be raised")
        if self.compute.device not in {"cpu", "cuda", "auto"} or self.compute.kernel_dtype != "float32" or self.compute.accumulator_dtype != "float64":
            raise MSLNMSICAConfigError("invalid device or numerical dtype contract")
        if self.energy.quiet_standardization != "median_mad" or not self.energy.tail_add_one_smoothing:
            raise MSLNMSICAConfigError("quiet median/MAD and add-one tails are mandatory")
        if self.energy.tail_log_base not in {10} or self.energy.bounded_kappa_z <= 0:
            raise MSLNMSICAConfigError("invalid energy calibration")
        if self.fusion.primary_output != "separate_raw_and_evidence" or not 0 <= self.fusion.visualization_floor_beta <= 1:
            raise MSLNMSICAConfigError("raw and evidence must remain separate")
        if self.per_context_ica.primary_objective != "cs_parzen" or self.per_context_ica.angle_range_degrees != (0.0, 90.0):
            raise MSLNMSICAConfigError("primary per-context objective/range is frozen")
        if not 16 <= self.sampling.per_context_screen_samples <= self.sampling.per_context_confirmation_samples:
            raise MSLNMSICAConfigError("screen/confirmation samples must be ordered and at least 16")
        if self.sampling.cross_context_max_samples < 16 or self.sampling.time_block_length_frames < 2 or self.sampling.bootstrap_replicates < 1:
            raise MSLNMSICAConfigError("cross-context samples and bootstrap dimensions are invalid")
        if not self.sampling.reuse_indices_across_baselines:
            raise MSLNMSICAConfigError("identical sample IDs across baselines are mandatory")
        if not self.fold_ids or len(set(self.fold_ids)) != len(self.fold_ids):
            raise MSLNMSICAConfigError("fold_ids must be non-empty and unique")
        for interval in (self.source.review_interval_ui, self.source.quiet_interval_ui, *self.source.burst_intervals_ui.values()):
            if interval[0] < 1 or interval[1] < interval[0]:
                raise MSLNMSICAConfigError("UI intervals must be positive and inclusive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("config_path")
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value
        return convert(payload)
