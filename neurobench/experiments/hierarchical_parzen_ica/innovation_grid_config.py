"""Strict configuration for the Spon Ca stochastic-state architecture grid."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class InnovationGridConfigError(ValueError):
    """Raised when an architecture-grid manifest violates its contract."""


def _mapping(raw: Any, required: set[str], scope: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InnovationGridConfigError(f"{scope} must be an object")
    unknown = set(raw) - required
    missing = required - set(raw)
    if unknown or missing:
        raise InnovationGridConfigError(
            f"{scope} fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return dict(raw)


def _positive_floats(values: Any, scope: str, *, allow_zero: bool = False) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise InnovationGridConfigError(f"{scope} must be a non-empty list")
    result = tuple(float(value) for value in values)
    lower_ok = all(value >= 0 for value in result) if allow_zero else all(value > 0 for value in result)
    if not lower_ok or len(set(result)) != len(result):
        raise InnovationGridConfigError(f"{scope} must contain unique bounded values")
    return result


@dataclass(frozen=True)
class InnovationGridConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    architecture_manifest: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    grid: dict[str, Any]
    screening: dict[str, Any]
    detection: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "InnovationGridConfig":
        manifest = Path(path).resolve()
        raw = _mapping(
            json.loads(manifest.read_text(encoding="utf-8")),
            {
                "schema_version", "experiment_id", "source_video", "labels_tsv",
                "architecture_manifest", "output_dir", "preflight_dir", "frames",
                "grid", "screening", "detection", "visualization", "resources",
            },
            "top-level",
        )
        frames = _mapping(
            raw["frames"],
            {"review_start_ui", "review_end_ui", "quiet_start_ui", "quiet_end_ui", "frame_period_ms"},
            "frames",
        )
        grid = _mapping(
            raw["grid"],
            {
                "innovation_half_life_seconds", "correction_fractions",
                "correction_clip_mad", "fixed_memory_half_life_seconds",
                "fixed_steady_state_observation_fractions",
            },
            "grid",
        )
        screening = _mapping(
            raw["screening"],
            {
                "roi_radius_px", "proxy_pixels_per_stratum", "random_seed",
                "finalists_per_family_per_fold", "global_visual_finalists",
                "bootstrap_samples", "hard_gates",
            },
            "screening",
        )
        screening["hard_gates"] = _mapping(
            screening["hard_gates"],
            {
                "minimum_peak_retention", "minimum_area_retention",
                "minimum_late_retention", "minimum_waveform_correlation",
                "maximum_quiet_rms_ratio", "maximum_artifact_dynamics_ratio",
            },
            "screening.hard_gates",
        )
        detection = _mapping(
            raw["detection"],
            {
                "temporal_pool_tau", "nms_distance_px", "primary_match_radius_px",
                "match_radii_px", "quiet_false_peaks_per_map",
                "froc_quiet_peaks_per_map", "candidate_cap_per_burst",
                "fixed_candidates_per_burst",
            },
            "detection",
        )
        visualization = _mapping(
            raw["visualization"],
            {"compression", "write_background", "write_dynamics", "write_detection_maps"},
            "visualization",
        )
        resources = _mapping(
            raw["resources"],
            {
                "device", "cpu_threads", "max_ram_mib", "min_free_disk_mib",
                "max_output_mib", "heartbeat_seconds",
            },
            "resources",
        )
        root = manifest.parent
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / str(raw["source_video"])).resolve(),
            labels_tsv=(root / str(raw["labels_tsv"])).resolve(),
            architecture_manifest=(root / str(raw["architecture_manifest"])).resolve(),
            output_dir=(root / str(raw["output_dir"])).resolve(),
            preflight_dir=(root / str(raw["preflight_dir"])).resolve(),
            frames=frames,
            grid=grid,
            screening=screening,
            detection=detection,
            visualization=visualization,
            resources=resources,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise InnovationGridConfigError("schema_version must be 1 and experiment_id non-empty")
        f = self.frames
        if not (
            1 <= int(f["review_start_ui"]) == int(f["quiet_start_ui"])
            <= int(f["quiet_end_ui"]) < int(f["review_end_ui"])
            and float(f["frame_period_ms"]) > 0
        ):
            raise InnovationGridConfigError("invalid inclusive UI frame contract")
        g = self.grid
        innovation_half_lives = _positive_floats(
            g["innovation_half_life_seconds"], "innovation half-lives"
        )
        fractions = _positive_floats(
            g["correction_fractions"], "correction fractions", allow_zero=True
        )
        clips = _positive_floats(g["correction_clip_mad"], "correction clips")
        fixed_half_lives = _positive_floats(
            g["fixed_memory_half_life_seconds"], "fixed half-lives"
        )
        steady = _positive_floats(
            g["fixed_steady_state_observation_fractions"],
            "fixed steady-state fractions",
            allow_zero=True,
        )
        if (
            max(innovation_half_lives + fixed_half_lives) > 120
            or max(fractions + steady) > 1
            or max(clips) > 64
            or 0.0 not in fractions
            or len(innovation_half_lives) * (1 + (len(fractions) - 1) * len(clips)) > 512
            or len(fixed_half_lives) * len(steady) > 512
        ):
            raise InnovationGridConfigError("grid is unsafe, redundant, or missing zero-correction controls")
        s = self.screening
        gates = s["hard_gates"]
        if not (
            1 <= int(s["roi_radius_px"]) <= 12
            and 128 <= int(s["proxy_pixels_per_stratum"]) <= 65536
            and 1 <= int(s["finalists_per_family_per_fold"]) <= 8
            and 1 <= int(s["global_visual_finalists"]) <= 8
            and 100 <= int(s["bootstrap_samples"]) <= 100000
            and all(float(value) >= 0 for value in gates.values())
        ):
            raise InnovationGridConfigError("invalid screening bounds")
        d = self.detection
        radii = tuple(float(value) for value in d["match_radii_px"])
        froc = _positive_floats(d["froc_quiet_peaks_per_map"], "FROC rates")
        if not (
            0 < float(d["temporal_pool_tau"]) <= 10
            and 1 <= int(d["nms_distance_px"]) <= 32
            and float(d["primary_match_radius_px"]) in radii
            and min(radii) > 0
            and 0 < float(d["quiet_false_peaks_per_map"]) <= 20
            and float(d["quiet_false_peaks_per_map"]) in froc
            and 10 <= int(d["candidate_cap_per_burst"]) <= 10000
            and 1 <= int(d["fixed_candidates_per_burst"]) <= int(d["candidate_cap_per_burst"])
        ):
            raise InnovationGridConfigError("invalid frozen detector contract")
        if self.visualization["compression"] != "zlib":
            raise InnovationGridConfigError("only deterministic zlib TIFF compression is supported")
        r = self.resources
        if not (
            r["device"] == "cpu"
            and 1 <= int(r["cpu_threads"]) <= 8
            and int(r["max_ram_mib"]) >= 2048
            and int(r["min_free_disk_mib"]) > 0
            and int(r["max_output_mib"]) > 0
            and 5 <= float(r["heartbeat_seconds"]) <= 300
        ):
            raise InnovationGridConfigError("invalid resource bounds")

    @property
    def innovation_specs(self) -> tuple[dict[str, float | str], ...]:
        result: list[dict[str, float | str]] = []
        fractions = tuple(float(x) for x in self.grid["correction_fractions"])
        clips = tuple(float(x) for x in self.grid["correction_clip_mad"])
        for half_life in (float(x) for x in self.grid["innovation_half_life_seconds"]):
            result.append({
                "lane_id": f"innovation_h{half_life:g}_e0",
                "family": "innovation", "half_life_seconds": half_life,
                "correction_fraction": 0.0, "correction_clip_mad": 0.0,
            })
            for fraction in fractions:
                if fraction == 0:
                    continue
                for clip in clips:
                    result.append({
                        "lane_id": f"innovation_h{half_life:g}_e{fraction:g}_c{clip:g}",
                        "family": "innovation", "half_life_seconds": half_life,
                        "correction_fraction": fraction, "correction_clip_mad": clip,
                    })
        return tuple(result)

    @property
    def fixed_specs(self) -> tuple[dict[str, float | str], ...]:
        period = float(self.frames["frame_period_ms"])
        result: list[dict[str, float | str]] = []
        zero_control_added = False
        for half_life in (float(x) for x in self.grid["fixed_memory_half_life_seconds"]):
            memory = 0.5 ** (period / (half_life * 1000.0))
            for steady in (
                float(x) for x in self.grid["fixed_steady_state_observation_fractions"]
            ):
                if steady == 0.0:
                    if zero_control_added:
                        continue
                    zero_control_added = True
                    lane_id = "fixed_static_quiet_reference"
                else:
                    lane_id = f"fixed_h{half_life:g}_s{steady:g}"
                result.append({
                    "lane_id": lane_id,
                    "family": "fixed_point", "half_life_seconds": half_life,
                    "memory_coefficient": memory,
                    "current_coefficient": (1.0 - memory) * steady,
                    "steady_state_observation_fraction": steady,
                })
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_video", "labels_tsv", "architecture_manifest",
            "output_dir", "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
