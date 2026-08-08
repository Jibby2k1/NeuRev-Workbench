"""Default scientific-audit artifact policy for experiment workflows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping


class ScientificAuditError(ValueError):
    """Raised when the default scientific-audit contract is invalid or incomplete."""


@dataclass(frozen=True)
class ScientificAuditPolicy:
    enabled: bool = True
    opt_out_reason: str | None = None
    full_field_video: bool = True
    per_roi_closeup_videos: bool = True
    exact_pixel_time_series: bool = True
    occurrence_instant_montages: bool = True
    occurrence_detection_table: bool = True
    concise_report: bool = True
    separated_annotation_sections: bool = True
    sequential_model_stage_videos: bool = True
    comparison_figures_only: bool = True
    llm_context_index: bool = True
    marker_palette: str = "green_label_orange_candidate"
    roi_mode: str = "labels_or_candidate_surrogate"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ScientificAuditPolicy":
        block = config.get("scientific_audit", {})
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise ScientificAuditError("scientific_audit must be an object")
        known = set(cls.__dataclass_fields__)
        unknown = set(block) - known
        if unknown:
            raise ScientificAuditError(
                f"unknown scientific_audit fields: {sorted(unknown)}"
            )
        policy = cls(**dict(block))
        if not policy.enabled:
            reason = (policy.opt_out_reason or "").strip()
            if len(reason) < 12:
                raise ScientificAuditError(
                    "disabled scientific audit requires a specific opt_out_reason"
                )
            return policy
        if policy.opt_out_reason:
            raise ScientificAuditError(
                "opt_out_reason is only valid when scientific_audit.enabled is false"
            )
        required = (
            policy.full_field_video,
            policy.per_roi_closeup_videos,
            policy.exact_pixel_time_series,
            policy.occurrence_instant_montages,
            policy.occurrence_detection_table,
            policy.concise_report,
            policy.separated_annotation_sections,
            policy.sequential_model_stage_videos,
            policy.comparison_figures_only,
            policy.llm_context_index,
        )
        if not all(required):
            raise ScientificAuditError(
                "individual audit outputs cannot be disabled; opt out of the "
                "whole audit with an explicit reason"
            )
        if policy.marker_palette != "green_label_orange_candidate":
            raise ScientificAuditError(
                "marker_palette must preserve green labels and orange candidates"
            )
        if policy.roi_mode not in {
            "labels_or_candidate_surrogate",
            "labels",
            "candidate_surrogate",
        }:
            raise ScientificAuditError("unsupported scientific-audit roi_mode")
        return policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificAuditInventory:
    root: str
    full_field_videos: int
    closeup_videos: int
    trace_figures: int
    instant_figures: int
    occurrence_rows: int
    roi_metadata_files: int
    report_present: bool
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_scientific_audit(
    root: str | Path,
    *,
    expected_roi_count: int,
    expected_occurrence_count: int,
    policy: ScientificAuditPolicy | None = None,
) -> ScientificAuditInventory:
    selected = policy or ScientificAuditPolicy()
    target = Path(root)
    if not selected.enabled:
        return ScientificAuditInventory(
            root=str(target.resolve()),
            full_field_videos=0,
            closeup_videos=0,
            trace_figures=0,
            instant_figures=0,
            occurrence_rows=0,
            roi_metadata_files=0,
            report_present=False,
            complete=True,
        )
    if expected_roi_count < 1 or expected_occurrence_count < 1:
        raise ScientificAuditError("expected audit counts must be positive")
    full = list((target / "videos").glob("*full_field*.mp4"))
    closeups = list((target / "videos" / "closeups").glob("*.mp4"))
    traces = list((target / "figures" / "traces").glob("*.png"))
    instants = list((target / "figures" / "instants").glob("*.png"))
    metadata = list((target / "metadata").glob("roi_*.json"))
    table = target / "metadata" / "all_occurrences.csv"
    rows = 0
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
    report = (target / "REPORT.md").is_file()
    complete = (
        len(full) >= 1
        and len(closeups) >= expected_roi_count
        and len(traces) >= expected_roi_count
        and len(instants) >= expected_occurrence_count
        and rows == expected_occurrence_count
        and len(metadata) >= expected_roi_count
        and report
    )
    return ScientificAuditInventory(
        root=str(target.resolve()),
        full_field_videos=len(full),
        closeup_videos=len(closeups),
        trace_figures=len(traces),
        instant_figures=len(instants),
        occurrence_rows=rows,
        roi_metadata_files=len(metadata),
        report_present=report,
        complete=complete,
    )



@dataclass(frozen=True)
class ThreeSectionAuditInventory:
    root: str
    expert_full_field_videos: int
    expert_closeup_videos: int
    expert_trace_figures: int
    model_full_field_videos: int
    model_closeup_videos: int
    model_trace_figures: int
    comparison_trace_figures: int
    comparison_videos: int
    comparison_closeups: int
    llm_context_present: bool
    root_index_present: bool
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_three_section_scientific_audit(
    root: str | Path,
    *,
    expected_expert_roi_count: int,
    expected_model_roi_count: int,
    expected_expert_occurrence_count: int,
) -> ThreeSectionAuditInventory:
    """Inspect the finalized expert/model/comparison audit contract."""
    if min(
        expected_expert_roi_count,
        expected_model_roi_count,
        expected_expert_occurrence_count,
    ) < 1:
        raise ScientificAuditError("expected three-section audit counts must be positive")
    target = Path(root)
    expert = target / "1_Expert_Annotations"
    model = target / "2_Model_Annotations"
    comparison = target / "3_Comparison"
    expert_full = list((expert / "videos").glob("*full_field*.mp4"))
    expert_closeups = list((expert / "videos" / "closeups").glob("*.mp4"))
    expert_traces = list((expert / "figures" / "traces").glob("*.png"))
    model_full = list((model / "videos").glob("*full_field*.mp4"))
    model_closeups = list((model / "videos" / "closeups").glob("*.mp4"))
    model_traces = list((model / "figures" / "traces").glob("*.png"))
    comparison_traces = list((comparison / "trace_comparisons").glob("*.png"))
    comparison_videos = list(comparison.glob("**/*.mp4"))
    comparison_closeups = list(comparison.glob("**/closeups/*"))
    llm_path = target / "llm_context.json"
    llm_valid = False
    if llm_path.is_file():
        payload = json.loads(llm_path.read_text(encoding="utf-8"))
        llm_valid = (
            payload.get("annotation_separation") == "strict"
            and payload.get("comparison_spatial_panels")
            == ["Raw matched comparison", "MSICA + MSLN matched comparison"]
            and len(payload.get("model_stage_sequence", [])) >= 3
        )
    root_index = all(
        (target / name).is_file()
        for name in (
            "REPORT.md",
            "summary.json",
            "artifact_index.json",
            "validation.json",
        )
    )
    complete = (
        len(expert_full) >= 1
        and len(expert_closeups) >= expected_expert_roi_count
        and len(expert_traces) >= expected_expert_roi_count
        and len(model_full) >= 1
        and len(model_closeups) >= expected_model_roi_count
        and len(model_traces) >= expected_model_roi_count
        and len(comparison_traces) >= expected_expert_occurrence_count
        and not comparison_videos
        and not comparison_closeups
        and llm_valid
        and root_index
    )
    return ThreeSectionAuditInventory(
        root=str(target.resolve()),
        expert_full_field_videos=len(expert_full),
        expert_closeup_videos=len(expert_closeups),
        expert_trace_figures=len(expert_traces),
        model_full_field_videos=len(model_full),
        model_closeup_videos=len(model_closeups),
        model_trace_figures=len(model_traces),
        comparison_trace_figures=len(comparison_traces),
        comparison_videos=len(comparison_videos),
        comparison_closeups=len(comparison_closeups),
        llm_context_present=llm_valid,
        root_index_present=root_index,
        complete=complete,
    )


def require_three_section_scientific_audit(
    root: str | Path,
    *,
    expected_expert_roi_count: int,
    expected_model_roi_count: int,
    expected_expert_occurrence_count: int,
) -> ThreeSectionAuditInventory:
    inventory = inspect_three_section_scientific_audit(
        root,
        expected_expert_roi_count=expected_expert_roi_count,
        expected_model_roi_count=expected_model_roi_count,
        expected_expert_occurrence_count=expected_expert_occurrence_count,
    )
    if not inventory.complete:
        raise ScientificAuditError(
            "three-section scientific audit is incomplete: "
            + json.dumps(inventory.to_dict(), sort_keys=True)
        )
    return inventory

def require_scientific_audit(
    root: str | Path,
    *,
    expected_roi_count: int,
    expected_occurrence_count: int,
    policy: ScientificAuditPolicy | None = None,
) -> ScientificAuditInventory:
    inventory = inspect_scientific_audit(
        root,
        expected_roi_count=expected_roi_count,
        expected_occurrence_count=expected_occurrence_count,
        policy=policy,
    )
    if not inventory.complete:
        raise ScientificAuditError(
            "scientific audit is incomplete: "
            + json.dumps(inventory.to_dict(), sort_keys=True)
        )
    return inventory
