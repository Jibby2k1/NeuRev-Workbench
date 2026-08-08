import csv
import json

import pytest

from neurobench.reports.scientific_audit import (
    ScientificAuditError,
    ScientificAuditPolicy,
    inspect_scientific_audit,
    require_scientific_audit,
    inspect_three_section_scientific_audit,
    require_three_section_scientific_audit,
)


def test_scientific_audit_is_default_on() -> None:
    policy = ScientificAuditPolicy.from_config({})
    assert policy.enabled
    assert policy.roi_mode == "labels_or_candidate_surrogate"


def test_scientific_audit_opt_out_requires_specific_reason() -> None:
    with pytest.raises(ScientificAuditError, match="opt_out_reason"):
        ScientificAuditPolicy.from_config(
            {"scientific_audit": {"enabled": False, "opt_out_reason": "no"}}
        )
    policy = ScientificAuditPolicy.from_config(
        {"scientific_audit": {
            "enabled": False,
            "opt_out_reason": "User explicitly requested a metrics-only run.",
        }}
    )
    assert not policy.enabled


def test_scientific_audit_rejects_partial_output_suppression() -> None:
    with pytest.raises(ScientificAuditError, match="individual audit outputs"):
        ScientificAuditPolicy.from_config(
            {"scientific_audit": {"per_roi_closeup_videos": False}}
        )


def test_complete_scientific_audit_inventory(tmp_path) -> None:
    (tmp_path / "videos" / "closeups").mkdir(parents=True)
    (tmp_path / "figures" / "traces").mkdir(parents=True)
    (tmp_path / "figures" / "instants").mkdir(parents=True)
    (tmp_path / "metadata").mkdir(parents=True)
    (tmp_path / "videos" / "review_full_field.mp4").write_bytes(b"video")
    for roi in range(2):
        (tmp_path / "videos" / "closeups" / f"roi_{roi}.mp4").write_bytes(b"video")
        (tmp_path / "figures" / "traces" / f"roi_{roi}.png").write_bytes(b"png")
        (tmp_path / "metadata" / f"roi_{roi}.json").write_text("{}")
    for occurrence in range(3):
        path = tmp_path / "figures" / "instants" / f"event_{occurrence}.png"
        path.write_bytes(b"png")
    with (tmp_path / "metadata" / "all_occurrences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["roi_id", "frame"])
        writer.writeheader()
        writer.writerows([
            {"roi_id": 0, "frame": 1},
            {"roi_id": 0, "frame": 2},
            {"roi_id": 1, "frame": 3},
        ])
    (tmp_path / "REPORT.md").write_text("# Audit\n", encoding="utf-8")

    inventory = inspect_scientific_audit(
        tmp_path, expected_roi_count=2, expected_occurrence_count=3
    )
    assert inventory.complete
    assert require_scientific_audit(
        tmp_path, expected_roi_count=2, expected_occurrence_count=3
    ) == inventory


def test_incomplete_scientific_audit_fails_completion_gate(tmp_path) -> None:
    with pytest.raises(ScientificAuditError, match="incomplete"):
        require_scientific_audit(
            tmp_path, expected_roi_count=1, expected_occurrence_count=1
        )


def test_complete_three_section_audit_and_llm_index(tmp_path) -> None:
    for section in ("1_Expert_Annotations", "2_Model_Annotations"):
        (tmp_path / section / "videos" / "closeups").mkdir(parents=True)
        (tmp_path / section / "figures" / "traces").mkdir(parents=True)
        (tmp_path / section / "videos" / f"{section}_full_field.mp4").write_bytes(b"v")
    comparison = tmp_path / "3_Comparison"
    (comparison / "trace_comparisons").mkdir(parents=True)
    for index in range(2):
        (tmp_path / "1_Expert_Annotations" / "videos" / "closeups" / f"roi_{index}.mp4").write_bytes(b"v")
        (tmp_path / "1_Expert_Annotations" / "figures" / "traces" / f"roi_{index}.png").write_bytes(b"p")
    for index in range(3):
        (tmp_path / "2_Model_Annotations" / "videos" / "closeups" / f"model_roi_{index}.mp4").write_bytes(b"v")
        (tmp_path / "2_Model_Annotations" / "figures" / "traces" / f"model_roi_{index}.png").write_bytes(b"p")
    for index in range(4):
        (comparison / "trace_comparisons" / f"occurrence_{index}.png").write_bytes(b"p")
    for name in ("REPORT.md", "summary.json", "artifact_index.json", "validation.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "llm_context.json").write_text(json.dumps({
        "annotation_separation": "strict",
        "comparison_spatial_panels": ["Raw matched comparison", "MSICA + MSLN matched comparison"],
        "model_stage_sequence": ["Raw", "MSICA", "MSLN"],
    }))
    inventory = inspect_three_section_scientific_audit(
        tmp_path,
        expected_expert_roi_count=2,
        expected_model_roi_count=3,
        expected_expert_occurrence_count=4,
    )
    assert inventory.complete
    assert require_three_section_scientific_audit(
        tmp_path,
        expected_expert_roi_count=2,
        expected_model_roi_count=3,
        expected_expert_occurrence_count=4,
    ) == inventory


def test_three_section_comparison_rejects_videos(tmp_path) -> None:
    (tmp_path / "3_Comparison").mkdir()
    (tmp_path / "3_Comparison" / "mixed.mp4").write_bytes(b"v")
    inventory = inspect_three_section_scientific_audit(
        tmp_path,
        expected_expert_roi_count=1,
        expected_model_roi_count=1,
        expected_expert_occurrence_count=1,
    )
    assert inventory.comparison_videos == 1
    assert not inventory.complete
