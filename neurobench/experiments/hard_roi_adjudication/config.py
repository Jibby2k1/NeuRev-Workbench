"""Strict configuration for Spon Ca Burst hard-ROI adjudication."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class HardRoiAdjudicationConfigError(ValueError):
    """Raised when the adjudication manifest differs from its frozen contract."""


@dataclass(frozen=True)
class HardRoiAdjudicationConfig:
    schema_version: int
    experiment_id: str
    source_video: Path
    original_labels_tsv: Path
    source_ranker_root: Path
    source_scientific_audit_root: Path
    output_dir: Path
    preflight_dir: Path
    review: dict[str, Any]
    evaluation: dict[str, Any]
    frozen_panel: list[dict[str, Any]]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "HardRoiAdjudicationConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {
            "schema_version", "experiment_id", "source_video",
            "original_labels_tsv", "source_ranker_root",
            "source_scientific_audit_root", "output_dir", "preflight_dir",
            "review", "evaluation", "frozen_panel", "resources",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            keys = set(raw) if isinstance(raw, dict) else set()
            raise HardRoiAdjudicationConfigError(
                f"top-level fields differ; missing={sorted(fields-keys)}, "
                f"unknown={sorted(keys-fields)}"
            )
        root = manifest.parent
        frozen_panel = []
        for raw_row in raw["frozen_panel"]:
            row = dict(raw_row)
            row["path"] = str((root / str(row["path"])).resolve())
            frozen_panel.append(row)
        result = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_video=(root / raw["source_video"]).resolve(),
            original_labels_tsv=(root / raw["original_labels_tsv"]).resolve(),
            source_ranker_root=(root / raw["source_ranker_root"]).resolve(),
            source_scientific_audit_root=(
                root / raw["source_scientific_audit_root"]
            ).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            preflight_dir=(root / raw["preflight_dir"]).resolve(),
            review=dict(raw["review"]),
            evaluation=dict(raw["evaluation"]),
            frozen_panel=frozen_panel,
            resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id.strip():
            raise HardRoiAdjudicationConfigError("invalid schema or experiment id")
        review_fields = {
            "target_roi_ids", "pad_before_frames", "pad_after_frames",
            "crop_padding_px", "box_half_size_px", "playback_fps",
            "display_lower_percentile", "display_upper_percentile",
            "source_frame_rate_hz",
        }
        if set(self.review) != review_fields:
            raise HardRoiAdjudicationConfigError("review fields differ")
        targets = [str(value) for value in self.review["target_roi_ids"]]
        required = {
            "roi_007", "roi_010", "roi_014", "roi_015", "roi_019",
            "roi_020", "roi_023",
        }
        if len(targets) != len(set(targets)) or not required.issubset(targets):
            raise HardRoiAdjudicationConfigError(
                "target ROI list must be unique and contain the hard-ROI panel"
            )
        if not (5 <= int(self.review["pad_before_frames"]) <= 50):
            raise HardRoiAdjudicationConfigError("pad_before_frames outside [5, 50]")
        if not (5 <= int(self.review["pad_after_frames"]) <= 50):
            raise HardRoiAdjudicationConfigError("pad_after_frames outside [5, 50]")
        if not (16 <= int(self.review["crop_padding_px"]) <= 96):
            raise HardRoiAdjudicationConfigError("crop_padding_px outside [16, 96]")
        if not (3 <= int(self.review["box_half_size_px"]) <= 20):
            raise HardRoiAdjudicationConfigError("box_half_size_px outside [3, 20]")
        if not (1 <= float(self.review["playback_fps"]) <= 30):
            raise HardRoiAdjudicationConfigError("playback_fps outside [1, 30]")
        if float(self.review["source_frame_rate_hz"]) != 50.0:
            raise HardRoiAdjudicationConfigError("source frame rate must remain 50 Hz")
        lo = float(self.review["display_lower_percentile"])
        hi = float(self.review["display_upper_percentile"])
        if not (0 <= lo < hi <= 100):
            raise HardRoiAdjudicationConfigError("invalid display percentiles")
        evaluation_fields = {
            "budgets", "match_radius_px", "nms_distance_px",
            "relaxed_localization_radius_px", "temporal_pool_temperature",
        }
        if set(self.evaluation) != evaluation_fields:
            raise HardRoiAdjudicationConfigError("evaluation fields differ")
        budgets = [int(value) for value in self.evaluation["budgets"]]
        if budgets != [20, 40, 58, 80, 100]:
            raise HardRoiAdjudicationConfigError("frozen budgets differ")
        if not (
            float(self.evaluation["match_radius_px"]) == 6.0
            and int(self.evaluation["nms_distance_px"]) == 6
            and float(self.evaluation["relaxed_localization_radius_px"]) > 6.0
            and float(self.evaluation["temporal_pool_temperature"]) == 0.25
        ):
            raise HardRoiAdjudicationConfigError("frozen evaluation contract differs")
        panel_fields = {"feature_id", "path", "storage", "role"}
        panel_ids = []
        for row in self.frozen_panel:
            if set(row) != panel_fields:
                raise HardRoiAdjudicationConfigError("frozen-panel fields differ")
            if row["storage"] not in {"npy_float", "display_tiff_uint16"}:
                raise HardRoiAdjudicationConfigError("unsupported panel storage")
            panel_ids.append(str(row["feature_id"]))
        expected = {
            "carrier_signed", "coherence_w15", "propagation_lag2_w15",
            "radial_cs_shell", "noise_vst_residual",
        }
        if set(panel_ids) != expected or len(panel_ids) != len(set(panel_ids)):
            raise HardRoiAdjudicationConfigError("frozen panel differs")
        resource_fields = {
            "cpu_threads", "max_ram_mib", "min_free_disk_mib", "max_output_mib"
        }
        if set(self.resources) != resource_fields:
            raise HardRoiAdjudicationConfigError("resource fields differ")
        if not (1 <= int(self.resources["cpu_threads"]) <= 8):
            raise HardRoiAdjudicationConfigError("cpu_threads outside [1, 8]")
        if int(self.resources["max_ram_mib"]) < 1024:
            raise HardRoiAdjudicationConfigError("RAM cap is too small")

    def panel_paths(self) -> dict[str, Path]:
        return {
            str(row["feature_id"]): Path(str(row["path"]))
            for row in self.frozen_panel
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_video", "original_labels_tsv", "source_ranker_root",
            "source_scientific_audit_root", "output_dir", "preflight_dir",
        ):
            payload[key] = str(payload[key])
        return payload
