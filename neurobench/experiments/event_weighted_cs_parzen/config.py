"""Versioned configuration for the event-balanced CS-Parzen ICA sweep."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class EventWeightedConfigError(ValueError):
    pass


def _strict(raw: Mapping[str, Any], allowed: set[str], scope: str) -> dict[str, Any]:
    values = dict(raw)
    unknown = set(values) - allowed
    if unknown:
        raise EventWeightedConfigError(
            f"Unknown {scope} fields: {', '.join(sorted(unknown))}"
        )
    missing = allowed - set(values)
    if missing:
        raise EventWeightedConfigError(
            f"Missing {scope} fields: {', '.join(sorted(missing))}"
        )
    return values


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
    gaussian_sigma_px: float
    ema_alpha: float
    motion_correction: str


@dataclass(frozen=True)
class SamplingConfig:
    seed: int
    screen_samples: int
    confirmation_samples: int
    heldout_guard_frames: int
    event_roi_radius_px: int
    event_screen_max_samples_per_event: int
    event_confirmation_max_samples_per_event: int
    reuse_sample_indices_across_alpha: bool
    merge_duplicate_indices: bool
    equal_mass_per_event: bool
    phase_balancing: bool
    bad_frames_ui: tuple[int, ...]


@dataclass(frozen=True)
class WeightingConfig:
    modes: tuple[str, ...]
    alpha_grid: tuple[float, ...]


@dataclass(frozen=True)
class WhiteningConfig:
    primary_mode: str
    run_weighted_ablation: bool
    weighted_ablation_alphas: tuple[float, ...]
    eigenvalue_floor_ratio: float


@dataclass(frozen=True)
class ParzenConfig:
    bandwidth: float
    kernel_block_rows: int
    kernel_dtype: str
    accumulator_dtype: str


@dataclass(frozen=True)
class AngleSearchConfig:
    range_degrees: tuple[float, float]
    coarse_step_degrees: float
    refine_half_width_degrees: float
    refine_step_degrees: float


@dataclass(frozen=True)
class ComputeConfig:
    device: str
    max_parallel_folds: int
    max_worker_processes: int
    max_peak_ram_gb: float
    max_peak_vram_gb: float


@dataclass(frozen=True)
class EvaluationConfig:
    primary_z_threshold: float
    quiet_mad_floor_percentile: float
    nms_distance_px: int
    primary_match_radius_px: int
    quiet_false_peaks_per_map: float


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    render_selected_videos_only: bool
    selected_video_alphas: tuple[float | str, ...]
    representative_frames_ui: tuple[int, ...]


@dataclass(frozen=True)
class EventWeightedCSParzenConfig:
    schema_version: int
    experiment_id: str
    source: SourceConfig
    preprocessing: PreprocessingConfig
    sampling: SamplingConfig
    weighting: WeightingConfig
    whitening: WhiteningConfig
    parzen: ParzenConfig
    angle_search: AngleSearchConfig
    compute: ComputeConfig
    evaluation: EvaluationConfig
    outputs: OutputConfig
    fold_ids: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path) -> "EventWeightedCSParzenConfig":
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise EventWeightedConfigError("configuration must be a YAML mapping")
        top = _strict(
            raw,
            {
                "schema_version", "experiment_id", "source", "preprocessing",
                "sampling", "weighting", "whitening", "parzen", "angle_search",
                "compute", "evaluation", "outputs", "fold_ids",
            },
            "top-level",
        )
        root = config_path.parent
        source = _strict(top["source"], set(SourceConfig.__dataclass_fields__), "source")
        source["movie_path"] = (root / source["movie_path"]).resolve()
        source["labels_path"] = (root / source["labels_path"]).resolve()
        evidence = source["baseline_evidence_dir"]
        source["baseline_evidence_dir"] = (
            None if evidence is None else (root / evidence).resolve()
        )
        source["review_interval_ui"] = tuple(int(x) for x in source["review_interval_ui"])
        source["quiet_interval_ui"] = tuple(int(x) for x in source["quiet_interval_ui"])
        source["burst_intervals_ui"] = {
            int(key): tuple(int(x) for x in value)
            for key, value in source["burst_intervals_ui"].items()
        }
        sections = {}
        section_types = {
            "preprocessing": PreprocessingConfig,
            "sampling": SamplingConfig,
            "weighting": WeightingConfig,
            "whitening": WhiteningConfig,
            "parzen": ParzenConfig,
            "angle_search": AngleSearchConfig,
            "compute": ComputeConfig,
            "evaluation": EvaluationConfig,
            "outputs": OutputConfig,
        }
        for name, section_type in section_types.items():
            sections[name] = _strict(
                top[name], set(section_type.__dataclass_fields__), name
            )
        sections["sampling"]["bad_frames_ui"] = tuple(
            int(x) for x in sections["sampling"]["bad_frames_ui"]
        )
        sections["weighting"]["modes"] = tuple(sections["weighting"]["modes"])
        sections["weighting"]["alpha_grid"] = tuple(
            float(x) for x in sections["weighting"]["alpha_grid"]
        )
        sections["whitening"]["weighted_ablation_alphas"] = tuple(
            float(x) for x in sections["whitening"]["weighted_ablation_alphas"]
        )
        sections["angle_search"]["range_degrees"] = tuple(
            float(x) for x in sections["angle_search"]["range_degrees"]
        )
        sections["outputs"]["root_dir"] = (
            root / sections["outputs"]["root_dir"]
        ).resolve()
        sections["outputs"]["selected_video_alphas"] = tuple(
            sections["outputs"]["selected_video_alphas"]
        )
        sections["outputs"]["representative_frames_ui"] = tuple(
            int(x) for x in sections["outputs"]["representative_frames_ui"]
        )
        config = cls(
            schema_version=int(top["schema_version"]),
            experiment_id=str(top["experiment_id"]),
            source=SourceConfig(**source),
            preprocessing=PreprocessingConfig(**sections["preprocessing"]),
            sampling=SamplingConfig(**sections["sampling"]),
            weighting=WeightingConfig(**sections["weighting"]),
            whitening=WhiteningConfig(**sections["whitening"]),
            parzen=ParzenConfig(**sections["parzen"]),
            angle_search=AngleSearchConfig(**sections["angle_search"]),
            compute=ComputeConfig(**sections["compute"]),
            evaluation=EvaluationConfig(**sections["evaluation"]),
            outputs=OutputConfig(**sections["outputs"]),
            fold_ids=tuple(int(x) for x in top["fold_ids"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        s, p = self.source, self.preprocessing
        sample, weight = self.sampling, self.weighting
        if self.schema_version != 1:
            raise EventWeightedConfigError("schema_version must be exactly 1")
        if s.axes != "TYX" or not s.ui_one_based:
            raise EventWeightedConfigError("source must use one-based TYX coordinates")
        if s.review_interval_ui[0] >= s.review_interval_ui[1]:
            raise EventWeightedConfigError("invalid review interval")
        if not (
            s.review_interval_ui[0] <= s.quiet_interval_ui[0]
            <= s.quiet_interval_ui[1] <= s.review_interval_ui[1]
        ):
            raise EventWeightedConfigError("quiet interval must be within review")
        if set(self.fold_ids) - set(s.burst_intervals_ui):
            raise EventWeightedConfigError("fold_ids must name declared bursts")
        if p.gaussian_sigma_px != 1.0 or p.ema_alpha != 0.4:
            raise EventWeightedConfigError("v1 freezes sigma=1.0 and EMA alpha=0.4")
        if p.motion_correction != "none":
            raise EventWeightedConfigError("v1 freezes motion correction to none")
        if not (
            16 <= sample.screen_samples <= sample.confirmation_samples <= 8192
        ):
            raise EventWeightedConfigError("invalid bounded sample counts")
        if sample.heldout_guard_frames < 0 or sample.event_roi_radius_px < 0:
            raise EventWeightedConfigError("guard and ROI radius must be nonnegative")
        if min(
            sample.event_screen_max_samples_per_event,
            sample.event_confirmation_max_samples_per_event,
        ) < 1:
            raise EventWeightedConfigError("event sample caps must be positive")
        if sample.event_screen_max_samples_per_event > sample.event_confirmation_max_samples_per_event:
            raise EventWeightedConfigError("event screen cap cannot exceed confirmation cap")
        if not (
            sample.reuse_sample_indices_across_alpha
            and sample.merge_duplicate_indices
            and sample.equal_mass_per_event
            and not sample.phase_balancing
        ):
            raise EventWeightedConfigError("v1 requires fixed, merged, equal-event support without phases")
        if not weight.modes or set(weight.modes) - {"frame_balanced", "roi_balanced"}:
            raise EventWeightedConfigError("invalid weighting modes")
        if (
            not weight.alpha_grid
            or weight.alpha_grid[0] != 0
            or tuple(sorted(set(weight.alpha_grid))) != weight.alpha_grid
            or any(not 0 <= x < 1 for x in weight.alpha_grid)
        ):
            raise EventWeightedConfigError(
                "alpha grid must be unique, increasing from zero, and within [0,1)"
            )
        if self.whitening.primary_mode != "natural_fixed":
            raise EventWeightedConfigError("primary whitening must be natural_fixed")
        if any(alpha not in weight.alpha_grid for alpha in self.whitening.weighted_ablation_alphas):
            raise EventWeightedConfigError("weighted ablation alphas must be in alpha_grid")
        if not 0 < self.whitening.eigenvalue_floor_ratio < 1:
            raise EventWeightedConfigError("invalid eigenvalue floor")
        if self.parzen.bandwidth != 0.35 or self.parzen.kernel_block_rows < 1:
            raise EventWeightedConfigError("v1 freezes bandwidth=0.35")
        if self.parzen.kernel_dtype != "float32" or self.parzen.accumulator_dtype != "float64":
            raise EventWeightedConfigError("standard dtype contract is float32/float64")
        if self.angle_search.range_degrees != (0.0, 90.0):
            raise EventWeightedConfigError("angle search must cover [0,90)")
        if 90 / self.angle_search.coarse_step_degrees % 1 > 1e-9:
            raise EventWeightedConfigError("coarse step must tile [0,90)")
        if self.compute.device != "cpu" or not 1 <= self.compute.max_parallel_folds <= 2:
            raise EventWeightedConfigError("v1 is CPU-only with at most two folds")
        if not 1 <= self.compute.max_worker_processes <= 8:
            raise EventWeightedConfigError("worker cap must be in [1,8]")
        if self.evaluation.primary_match_radius_px != 6:
            raise EventWeightedConfigError("primary match radius is frozen at six pixels")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"]["movie_path"] = str(self.source.movie_path)
        payload["source"]["labels_path"] = str(self.source.labels_path)
        payload["source"]["baseline_evidence_dir"] = (
            None
            if self.source.baseline_evidence_dir is None
            else str(self.source.baseline_evidence_dir)
        )
        payload["outputs"]["root_dir"] = str(self.outputs.root_dir)
        return payload
