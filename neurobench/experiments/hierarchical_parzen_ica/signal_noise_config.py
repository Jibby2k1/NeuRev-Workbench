"""Strict manifest for the real-data noisy-Parzen signal/noise split."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class SignalNoiseConfigError(ValueError):
    pass


def _strict(value: Any, fields: set[str], scope: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SignalNoiseConfigError(f"{scope} must be an object")
    unknown, missing = set(value) - fields, fields - set(value)
    if unknown or missing:
        raise SignalNoiseConfigError(
            f"{scope} fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return dict(value)


@dataclass(frozen=True)
class SignalNoiseConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    labels_tsv: Path
    architecture_manifest: Path
    output_dir: Path
    preflight_dir: Path
    frames: dict[str, Any]
    input_lane: dict[str, Any]
    posterior: dict[str, Any]
    selection: dict[str, Any]
    visualization: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "SignalNoiseConfig":
        manifest = Path(path).resolve()
        raw = _strict(
            json.loads(manifest.read_text(encoding="utf-8")),
            {
                "schema_version", "experiment_id", "source_video", "labels_tsv",
                "architecture_manifest", "output_dir", "preflight_dir", "frames",
                "input_lane", "posterior", "selection", "visualization", "resources",
            },
            "top-level",
        )
        frames = _strict(
            raw["frames"],
            {"review_start_ui", "review_end_ui", "quiet_start_ui", "quiet_end_ui", "frame_period_ms"},
            "frames",
        )
        input_lane = _strict(
            raw["input_lane"],
            {"id", "reference_half_life_seconds", "correction_fraction", "correction_clip_mad"},
            "input_lane",
        )
        posterior = _strict(
            raw["posterior"],
            {
                "dictionary_centers", "dictionary_zero_mass_fraction",
                "dictionary_activation_abs_z", "dictionary_sample_pixels",
                "sample_seed", "bandwidths", "noise_variance_multipliers",
                "lookup_points", "lookup_abs_z", "quiet_scale_floor_percentile",
            },
            "posterior",
        )
        selection = _strict(
            raw["selection"],
            {
                "roi_radius_px", "minimum_peak_retention", "minimum_area_retention",
                "minimum_late_retention", "minimum_waveform_correlation",
                "maximum_quiet_signal_rms_ratio",
            },
            "selection",
        )
        visualization = _strict(
            raw["visualization"],
            {
                "compression", "signed_absolute_percentile",
                "positive_upper_percentile", "sample_frame_stride",
                "sample_row_stride", "sample_column_stride",
            },
            "visualization",
        )
        resources = _strict(
            raw["resources"],
            {
                "device", "cpu_threads", "max_ram_mib", "min_free_disk_mib",
                "max_output_mib",
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
            frames=frames, input_lane=input_lane, posterior=posterior,
            selection=selection, visualization=visualization, resources=resources,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise SignalNoiseConfigError("schema_version must be 1 and experiment_id non-empty")
        f = self.frames
        if not (
            1 <= int(f["review_start_ui"]) == int(f["quiet_start_ui"])
            <= int(f["quiet_end_ui"]) < int(f["review_end_ui"])
            and float(f["frame_period_ms"]) > 0
        ):
            raise SignalNoiseConfigError("invalid inclusive UI frame contract")
        lane = self.input_lane
        if not (
            lane["id"] == "reference_parzen_innovation"
            and float(lane["reference_half_life_seconds"]) > 0
            and 0 <= float(lane["correction_fraction"]) <= 1
            and float(lane["correction_clip_mad"]) > 0
        ):
            raise SignalNoiseConfigError("invalid Parzen Innovation input lane")
        p = self.posterior
        bandwidths = tuple(float(value) for value in p["bandwidths"])
        noise = tuple(float(value) for value in p["noise_variance_multipliers"])
        if not (
            4 <= int(p["dictionary_centers"]) <= 256
            and 0 < float(p["dictionary_zero_mass_fraction"]) < 1
            and 0 < float(p["dictionary_activation_abs_z"]) <= 20
            and 256 <= int(p["dictionary_sample_pixels"]) <= 65536
            and bandwidths and noise
            and len(set(bandwidths)) == len(bandwidths)
            and len(set(noise)) == len(noise)
            and min(bandwidths + noise) > 0
            and max(bandwidths) <= 10 and max(noise) <= 20
            and 1024 <= int(p["lookup_points"]) <= 262144
            and 3 <= float(p["lookup_abs_z"]) <= 50
            and 0 <= float(p["quiet_scale_floor_percentile"]) <= 100
            and len(bandwidths) * len(noise) <= 128
        ):
            raise SignalNoiseConfigError("invalid bounded posterior grid")
        s = self.selection
        if not (
            1 <= int(s["roi_radius_px"]) <= 12
            and all(
                0 <= float(s[key]) <= 2
                for key in (
                    "minimum_peak_retention", "minimum_area_retention",
                    "minimum_late_retention", "minimum_waveform_correlation",
                    "maximum_quiet_signal_rms_ratio",
                )
            )
        ):
            raise SignalNoiseConfigError("invalid selection gates")
        v = self.visualization
        if not (
            v["compression"] == "zlib"
            and 90 <= float(v["signed_absolute_percentile"]) <= 100
            and 90 <= float(v["positive_upper_percentile"]) <= 100
            and min(
                int(v["sample_frame_stride"]), int(v["sample_row_stride"]),
                int(v["sample_column_stride"]),
            ) >= 1
        ):
            raise SignalNoiseConfigError("invalid visualization contract")
        r = self.resources
        if not (
            r["device"] == "cpu" and 1 <= int(r["cpu_threads"]) <= 8
            and int(r["max_ram_mib"]) >= 2048
            and int(r["min_free_disk_mib"]) > 0
            and int(r["max_output_mib"]) > 0
        ):
            raise SignalNoiseConfigError("invalid resources")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_video", "labels_tsv", "architecture_manifest",
            "output_dir", "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
