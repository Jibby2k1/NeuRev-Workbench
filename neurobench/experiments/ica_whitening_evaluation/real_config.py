"""Strict contract for the post-warning real-data ICA evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


class RealDataConfigError(ValueError):
    pass


def _strict(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealDataConfigError(f"{context} must be an object")
    missing, extra = fields - set(value), set(value) - fields
    if missing or extra:
        raise RealDataConfigError(
            f"{context} keys differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


@dataclass(frozen=True)
class ProposalContract:
    lanes: tuple[str, ...]
    temporal_pool: str
    per_lane_per_burst: int
    proposal_nms_distance_px: int
    union_separation_px: int
    evaluation_budgets: tuple[int, ...]
    match_radius_px: int


@dataclass(frozen=True)
class FitContract:
    eligibility: str
    maximum_fit_samples: int
    activity_fraction: float
    shard_count: int


@dataclass(frozen=True)
class RealDataConfig:
    manifest_path: Path
    schema_version: int
    experiment_id: str
    parent_config: Path
    synthetic_registry: Path
    decision_amendment: Path
    output_dir: Path
    proposals: ProposalContract
    fitting: FitContract
    controls: tuple[str, ...]
    synthetic_warning_non_blocking: bool

    @classmethod
    def from_json(cls, source: str | Path) -> "RealDataConfig":
        manifest = Path(source).expanduser().resolve()
        raw = _strict(json.loads(manifest.read_text(encoding="utf-8")), {
            "schema_version", "experiment_id", "parent_config",
            "synthetic_registry", "decision_amendment", "output_dir",
            "proposals", "fitting", "controls", "synthetic_warning_non_blocking",
        }, "top-level")

        def resolve(value: Any) -> Path:
            text = str(value)
            if text.startswith("repo://"):
                return (manifest.parent.parent / text.removeprefix("repo://")).resolve()
            if text.startswith("data://"):
                root = os.environ.get("NEUROBENCH_DATA_ROOT")
                if not root:
                    raise RealDataConfigError("data:// requires NEUROBENCH_DATA_ROOT")
                return (Path(root).expanduser() / text.removeprefix("data://")).resolve()
            path = Path(text).expanduser()
            return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()

        proposal_raw = _strict(raw["proposals"], {
            "lanes", "temporal_pool", "per_lane_per_burst",
            "proposal_nms_distance_px", "union_separation_px",
            "evaluation_budgets", "match_radius_px",
        }, "proposals")
        fit_raw = _strict(raw["fitting"], {
            "eligibility", "maximum_fit_samples", "activity_fraction", "shard_count",
        }, "fitting")
        config = cls(
            manifest_path=manifest,
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            parent_config=resolve(raw["parent_config"]),
            synthetic_registry=resolve(raw["synthetic_registry"]),
            decision_amendment=resolve(raw["decision_amendment"]),
            output_dir=resolve(raw["output_dir"]),
            proposals=ProposalContract(
                lanes=tuple(map(str, proposal_raw["lanes"])),
                temporal_pool=str(proposal_raw["temporal_pool"]),
                per_lane_per_burst=int(proposal_raw["per_lane_per_burst"]),
                proposal_nms_distance_px=int(proposal_raw["proposal_nms_distance_px"]),
                union_separation_px=int(proposal_raw["union_separation_px"]),
                evaluation_budgets=tuple(map(int, proposal_raw["evaluation_budgets"])),
                match_radius_px=int(proposal_raw["match_radius_px"]),
            ),
            fitting=FitContract(
                eligibility=str(fit_raw["eligibility"]),
                maximum_fit_samples=int(fit_raw["maximum_fit_samples"]),
                activity_fraction=float(fit_raw["activity_fraction"]),
                shard_count=int(fit_raw["shard_count"]),
            ),
            controls=tuple(map(str, raw["controls"])),
            synthetic_warning_non_blocking=bool(raw["synthetic_warning_non_blocking"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id:
            raise RealDataConfigError("schema_version must be 1 and experiment_id non-empty")
        if not self.synthetic_warning_non_blocking:
            raise RealDataConfigError("the investigator amendment must remain explicit")
        expected_lanes = {"raw_activity", "signed_temporal_difference", "spatial_highpass"}
        if set(self.proposals.lanes) != expected_lanes:
            raise RealDataConfigError(f"proposal lanes must be {sorted(expected_lanes)}")
        if tuple(sorted(self.proposals.evaluation_budgets)) != self.proposals.evaluation_budgets:
            raise RealDataConfigError("evaluation budgets must be strictly increasing")
        if len(set(self.proposals.evaluation_budgets)) != len(self.proposals.evaluation_budgets):
            raise RealDataConfigError("evaluation budgets must not repeat")
        numeric = (
            self.proposals.per_lane_per_burst,
            self.proposals.proposal_nms_distance_px,
            self.proposals.union_separation_px,
            self.proposals.match_radius_px,
            self.fitting.maximum_fit_samples,
            self.fitting.shard_count,
        )
        if any(value < 1 for value in numeric) or not 0 <= self.fitting.activity_fraction <= 1:
            raise RealDataConfigError("proposal and fitting bounds are invalid")
        if self.fitting.eligibility != "synthetic_all_numerically_resolved_only":
            raise RealDataConfigError("unsupported eligibility rule")
        required_controls = {"raw_direct", "signed_temporal_difference", "rank_matched_pca", "random_rotation", "separable_control"}
        if not required_controls <= set(self.controls):
            raise RealDataConfigError("matched controls are incomplete")
