"""Strict manifest parsing for latent-dynamics denoising experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


def _object(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {unknown}")
    return value


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


@dataclass(frozen=True)
class FrameConfig:
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    frame_period_ms: float


@dataclass(frozen=True)
class PreprocessingConfig:
    baseline_mode: str
    signed_residual: bool
    gain_mode: str
    motion_mode: str
    quiet_scale_floor_percentile: float


@dataclass(frozen=True)
class FitConfig:
    sample_pixels: int
    sample_seed: int
    temporal_validation_blocks: int
    stability_epsilon: float
    decay_time_ms_grid: tuple[float, ...]
    process_to_observation_grid: tuple[float, ...]
    parameter_mode: str


@dataclass(frozen=True)
class ApplicationConfig:
    tile_height: int
    tile_width: int
    write_filter_mean: bool
    write_smoother_mean: bool
    write_dense_residuals: bool


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...]
    write_dense_features: bool
    write_selected_tiffs: bool
    positive_views: bool


@dataclass(frozen=True)
class EvaluationConfig:
    primary_match_radius_px: float
    match_radii_px: tuple[float, ...]
    quiet_false_peaks_per_map: float
    capacity_reference_lane: str
    synthetic_seeds: tuple[int, ...]


@dataclass(frozen=True)
class ResourceConfig:
    cpu_threads: int
    max_ram_mib: int
    max_output_mib: int
    min_free_disk_mib: int


@dataclass(frozen=True)
class LatentDynamicsConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    output_dir: Path
    frames: FrameConfig
    preprocessing: PreprocessingConfig
    fit: FitConfig
    application: ApplicationConfig
    features: FeatureConfig
    evaluation: EvaluationConfig
    resources: ResourceConfig

    @classmethod
    def load(cls, path: str | Path) -> "LatentDynamicsConfig":
        source = Path(path).expanduser().resolve()
        raw = _object(json.loads(source.read_text(encoding="utf-8")), "manifest", {
            "schema_version", "experiment_id", "source_video", "labels_tsv",
            "output_dir", "frames", "preprocessing", "fit", "application",
            "features", "evaluation", "resources",
        })
        base = source.parent
        frames = _object(raw.get("frames"), "frames", {
            "review_start_ui", "review_end_ui", "quiet_start_ui", "quiet_end_ui", "frame_period_ms",
        })
        preprocessing = _object(raw.get("preprocessing"), "preprocessing", {
            "baseline_mode", "signed_residual", "gain_mode", "motion_mode",
            "quiet_scale_floor_percentile",
        })
        fit = _object(raw.get("fit"), "fit", {
            "sample_pixels", "sample_seed", "temporal_validation_blocks", "stability_epsilon",
            "decay_time_ms_grid", "process_to_observation_grid", "parameter_mode",
        })
        application = _object(raw.get("application"), "application", {
            "tile_height", "tile_width", "write_filter_mean", "write_smoother_mean",
            "write_dense_residuals",
        })
        features = _object(raw.get("features"), "features", {
            "lags", "write_dense_features", "write_selected_tiffs", "positive_views",
        })
        evaluation = _object(raw.get("evaluation"), "evaluation", {
            "primary_match_radius_px", "match_radii_px", "quiet_false_peaks_per_map",
            "capacity_reference_lane", "synthetic_seeds",
        })
        resources = _object(raw.get("resources"), "resources", {
            "cpu_threads", "max_ram_mib", "max_output_mib", "min_free_disk_mib",
        })
        config = cls(
            schema_version=raw.get("schema_version"), experiment_id=raw.get("experiment_id"),
            source_video=_path(base, raw.get("source_video"), "source_video"),
            labels_tsv=_path(base, raw.get("labels_tsv"), "labels_tsv"),
            output_dir=_path(base, raw.get("output_dir"), "output_dir"),
            frames=FrameConfig(**frames), preprocessing=PreprocessingConfig(**preprocessing),
            fit=FitConfig(
                **{key: value for key, value in fit.items() if key not in {"decay_time_ms_grid", "process_to_observation_grid"}},
                decay_time_ms_grid=tuple(fit.get("decay_time_ms_grid", ())),
                process_to_observation_grid=tuple(fit.get("process_to_observation_grid", ())),
            ),
            application=ApplicationConfig(**application),
            features=FeatureConfig(
                **{key: value for key, value in features.items() if key != "lags"},
                lags=tuple(features.get("lags", ())),
            ),
            evaluation=EvaluationConfig(
                **{key: value for key, value in evaluation.items() if key not in {"synthetic_seeds", "match_radii_px"}},
                match_radii_px=tuple(evaluation.get("match_radii_px", ())),
                synthetic_seeds=tuple(evaluation.get("synthetic_seeds", ())),
            ),
            resources=ResourceConfig(**resources),
        )
        config.validate()
        return config

    def validate(self) -> None:
        f, p, fit, app, feat, ev, res = (
            self.frames, self.preprocessing, self.fit, self.application,
            self.features, self.evaluation, self.resources,
        )
        if self.schema_version != 1 or not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("schema_version must be 1 and experiment_id must be non-empty")
        integers = (
            f.review_start_ui, f.review_end_ui, f.quiet_start_ui, f.quiet_end_ui,
            fit.sample_pixels, fit.sample_seed, fit.temporal_validation_blocks, app.tile_height, app.tile_width,
            res.cpu_threads, res.max_ram_mib, res.max_output_mib, res.min_free_disk_mib,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("Frame, count, tile, and resource fields must be integers")
        if not (1 <= f.review_start_ui <= f.quiet_start_ui <= f.quiet_end_ui <= f.review_end_ui):
            raise ValueError("Quiet interval must lie inside the one-based inclusive review interval")
        if f.quiet_end_ui - f.quiet_start_ui + 1 < 100:
            raise ValueError("Quiet interval must contain at least 100 frames for frozen calibration")
        if f.frame_period_ms <= 0 or fit.sample_pixels <= 0 or fit.temporal_validation_blocks < 1:
            raise ValueError("Frame period and fitting counts must be positive")
        if p.baseline_mode != "quiet_median" or not p.signed_residual:
            raise ValueError("Only signed quiet-median residuals are currently supported")
        if p.gain_mode != "none" or p.motion_mode != "none":
            raise ValueError("Gain and motion correction must remain 'none' in the initial contract")
        if not 0 <= p.quiet_scale_floor_percentile <= 100:
            raise ValueError("quiet_scale_floor_percentile must be in [0,100]")
        if fit.parameter_mode != "bounded_grid" or not 0 < fit.stability_epsilon < 1:
            raise ValueError("parameter_mode must be bounded_grid with epsilon in (0,1)")
        if not fit.decay_time_ms_grid or min(fit.decay_time_ms_grid) <= 0:
            raise ValueError("decay_time_ms_grid must contain positive values")
        if not fit.process_to_observation_grid or min(fit.process_to_observation_grid) <= 0:
            raise ValueError("process_to_observation_grid must contain positive values")
        if app.tile_height <= 0 or app.tile_width <= 0:
            raise ValueError("Application tiles must be positive")
        booleans = (p.signed_residual, app.write_filter_mean, app.write_smoother_mean,
                    app.write_dense_residuals, feat.write_dense_features,
                    feat.write_selected_tiffs, feat.positive_views)
        if any(not isinstance(value, bool) for value in booleans):
            raise ValueError("Boolean manifest fields must be JSON booleans")
        if app.write_dense_residuals or feat.write_dense_features:
            raise ValueError("Dense residual/features are disabled by the initial resource contract")
        if not feat.lags or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in feat.lags):
            raise ValueError("Feature lags must be positive integers")
        if ev.primary_match_radius_px <= 0 or not ev.match_radii_px or min(ev.match_radii_px) <= 0:
            raise ValueError("Evaluation radii must be positive and non-empty")
        if (ev.quiet_false_peaks_per_map <= 0 or ev.capacity_reference_lane != "raw_direct"
                or not ev.synthetic_seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in ev.synthetic_seeds)):
            raise ValueError("Evaluation capacity anchor and synthetic seeds must match the contract")
        if not 1 <= res.cpu_threads <= 24 or min(res.max_ram_mib, res.max_output_mib, res.min_free_disk_mib) <= 0:
            raise ValueError("Resource limits must be positive and cpu_threads in [1,24]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_video", "labels_tsv", "output_dir"):
            payload[key] = str(getattr(self, key))
        # Canonical JSON-compatible lists make reviewed config equality exact.
        return json.loads(json.dumps(payload))
