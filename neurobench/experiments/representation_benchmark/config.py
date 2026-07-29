"""Strict configuration for the Spon representation benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class RepresentationConfigError(ValueError):
    pass


def _strict(value: Mapping[str, Any], fields: set[str], scope: str) -> dict[str, Any]:
    payload = dict(value)
    unknown = set(payload) - fields
    missing = fields - set(payload)
    if unknown or missing:
        raise RepresentationConfigError(
            f"{scope}: unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    return payload


@dataclass(frozen=True)
class FrameConfig:
    review_start_ui: int
    review_end_ui: int
    quiet_start_ui: int
    quiet_end_ui: int
    frame_period_ms: float


@dataclass(frozen=True)
class PCAConfig:
    inputs: tuple[str, ...]
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class ICAConfig:
    inputs: tuple[str, ...]
    ranks: tuple[int, ...]
    seeds: tuple[int, ...]
    fit_sample_pixels: int
    max_iterations: int
    tolerance: float


@dataclass(frozen=True)
class AutoencoderConfig:
    enabled: bool
    inputs: tuple[str, ...]
    kinds: tuple[str, ...]
    ranks: tuple[int, ...]
    seeds: tuple[int, ...]
    train_pixels: int
    validation_pixels: int
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_width: int


@dataclass(frozen=True)
class UMAPConfig:
    enabled_if_available: bool
    neighbors: int
    min_dist: float
    sample_pixels: int
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    nms_distance_px: int
    match_radius_px: int
    quiet_false_peaks_per_map: float
    fixed_candidates_per_burst: int
    reconstruction_ranks: tuple[int, ...]
    representative_rank: int
    component_gallery_count: int
    write_representative_tiffs: bool


@dataclass(frozen=True)
class ResourceConfig:
    device: str
    cpu_threads: int
    projection_chunk_pixels: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int
    gpu_reserve_mib: int


@dataclass(frozen=True)
class RepresentationBenchmarkConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    output_dir: Path
    frames: FrameConfig
    pca: PCAConfig
    ica: ICAConfig
    autoencoder: AutoencoderConfig
    umap: UMAPConfig
    evaluation: EvaluationConfig
    resources: ResourceConfig

    @classmethod
    def load(cls, path: str | Path) -> "RepresentationBenchmarkConfig":
        source = Path(path).resolve()
        raw = _strict(json.loads(source.read_text(encoding="utf-8")), {
            "schema_version", "experiment_id", "source_video", "labels_tsv",
            "output_dir", "frames", "pca", "ica", "autoencoder", "umap",
            "evaluation", "resources",
        }, "top-level")
        root = source.parent
        frames = _strict(raw["frames"], set(FrameConfig.__dataclass_fields__), "frames")
        pca = _strict(raw["pca"], set(PCAConfig.__dataclass_fields__), "pca")
        ica = _strict(raw["ica"], set(ICAConfig.__dataclass_fields__), "ica")
        auto = _strict(raw["autoencoder"], set(AutoencoderConfig.__dataclass_fields__), "autoencoder")
        umap = _strict(raw["umap"], set(UMAPConfig.__dataclass_fields__), "umap")
        evaluation = _strict(raw["evaluation"], set(EvaluationConfig.__dataclass_fields__), "evaluation")
        resources = _strict(raw["resources"], set(ResourceConfig.__dataclass_fields__), "resources")
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
            pca=PCAConfig(
                inputs=tuple(str(x) for x in pca["inputs"]),
                ranks=tuple(int(x) for x in pca["ranks"]),
            ),
            ica=ICAConfig(
                inputs=tuple(str(x) for x in ica["inputs"]),
                ranks=tuple(int(x) for x in ica["ranks"]),
                seeds=tuple(int(x) for x in ica["seeds"]),
                fit_sample_pixels=int(ica["fit_sample_pixels"]),
                max_iterations=int(ica["max_iterations"]),
                tolerance=float(ica["tolerance"]),
            ),
            autoencoder=AutoencoderConfig(
                enabled=bool(auto["enabled"]),
                inputs=tuple(str(x) for x in auto["inputs"]),
                kinds=tuple(str(x) for x in auto["kinds"]),
                ranks=tuple(int(x) for x in auto["ranks"]),
                seeds=tuple(int(x) for x in auto["seeds"]),
                train_pixels=int(auto["train_pixels"]),
                validation_pixels=int(auto["validation_pixels"]),
                epochs=int(auto["epochs"]),
                batch_size=int(auto["batch_size"]),
                learning_rate=float(auto["learning_rate"]),
                hidden_width=int(auto["hidden_width"]),
            ),
            umap=UMAPConfig(
                enabled_if_available=bool(umap["enabled_if_available"]),
                neighbors=int(umap["neighbors"]),
                min_dist=float(umap["min_dist"]),
                sample_pixels=int(umap["sample_pixels"]),
                seed=int(umap["seed"]),
            ),
            evaluation=EvaluationConfig(
                nms_distance_px=int(evaluation["nms_distance_px"]),
                match_radius_px=int(evaluation["match_radius_px"]),
                quiet_false_peaks_per_map=float(evaluation["quiet_false_peaks_per_map"]),
                fixed_candidates_per_burst=int(evaluation["fixed_candidates_per_burst"]),
                reconstruction_ranks=tuple(int(x) for x in evaluation["reconstruction_ranks"]),
                representative_rank=int(evaluation["representative_rank"]),
                component_gallery_count=int(evaluation["component_gallery_count"]),
                write_representative_tiffs=bool(evaluation["write_representative_tiffs"]),
            ),
            resources=ResourceConfig(
                device=str(resources["device"]),
                cpu_threads=int(resources["cpu_threads"]),
                projection_chunk_pixels=int(resources["projection_chunk_pixels"]),
                max_ram_mib=int(resources["max_ram_mib"]),
                min_free_disk_mib=int(resources["min_free_disk_mib"]),
                max_output_mib=int(resources["max_output_mib"]),
                gpu_reserve_mib=int(resources["gpu_reserve_mib"]),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise RepresentationConfigError("schema_version must equal one")
        f = self.frames
        if not (1 <= f.review_start_ui <= f.quiet_start_ui <= f.quiet_end_ui <= f.review_end_ui):
            raise RepresentationConfigError("invalid inclusive frame contract")
        if f.quiet_start_ui != f.review_start_ui or f.frame_period_ms <= 0:
            raise RepresentationConfigError("quiet must begin at review start and frame period must be positive")
        allowed_inputs = {"amplitude", "quiet_residual"}
        for scope, inputs in (("pca", self.pca.inputs), ("ica", self.ica.inputs), ("autoencoder", self.autoencoder.inputs)):
            if not inputs or not set(inputs) <= allowed_inputs:
                raise RepresentationConfigError(f"{scope} inputs must use amplitude and/or quiet_residual")
        if tuple(sorted(set(self.pca.ranks))) != self.pca.ranks or not 1 <= min(self.pca.ranks) <= max(self.pca.ranks) <= 256:
            raise RepresentationConfigError("PCA ranks must be unique, sorted, and bounded by 256")
        if not set(self.ica.ranks) <= set(self.pca.ranks) or not self.ica.seeds:
            raise RepresentationConfigError("ICA ranks must be PCA ranks and seeds must be nonempty")
        if not 1024 <= self.ica.fit_sample_pixels <= 131072 or not 1 <= self.ica.max_iterations <= 1000:
            raise RepresentationConfigError("ICA fit bounds are invalid")
        a = self.autoencoder
        if a.enabled:
            if not set(a.kinds) <= {"linear", "nonlinear"} or not a.kinds or not a.ranks or not a.seeds:
                raise RepresentationConfigError("invalid autoencoder kinds/ranks/seeds")
            if a.train_pixels < 1024 or a.validation_pixels < 256 or a.epochs < 1 or a.batch_size < 32:
                raise RepresentationConfigError("autoencoder sampling/training bounds are invalid")
        e = self.evaluation
        if e.nms_distance_px != 6 or e.match_radius_px != 6:
            raise RepresentationConfigError("primary NMS and matching radius are frozen at six pixels")
        if not set(e.reconstruction_ranks) <= set(self.pca.ranks) or e.representative_rank not in self.pca.ranks:
            raise RepresentationConfigError("evaluation ranks must be declared PCA ranks")
        r = self.resources
        if r.device not in {"cpu", "cuda"} or not 1 <= r.cpu_threads <= 8:
            raise RepresentationConfigError("invalid device or CPU thread bound")
        if min(r.max_ram_mib, r.min_free_disk_mib, r.max_output_mib, r.projection_chunk_pixels) <= 0:
            raise RepresentationConfigError("resource bounds must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in ("source_video", "labels_tsv", "output_dir"):
            payload[field] = str(payload[field])
        return payload
