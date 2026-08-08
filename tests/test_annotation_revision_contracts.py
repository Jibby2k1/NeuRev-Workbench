from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurobench.models.annotation_revision import (
    AnnotationOperation,
    AnnotationRevision,
    AnnotationViewContract,
)
from neurobench.workbench.annotation_revisions import (
    RevisionConflictError,
    append_revision_operation,
    fork_revision_root,
    initialize_revision_root,
    list_revision_roots,
    load_revision_root,
    publish_revision_root,
    revision_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "annotation_correction" / "tiny_views.json"


def revision_payload(*, revision_id: str = "ann_test", operation_count: int = 0) -> dict:
    return {
        "schema_version": 1,
        "revisionId": revision_id,
        "parentRevisionId": "ann_import_v1",
        "state": "draft",
        "reviewerId": "reviewer_local_1",
        "frozenRunId": "frozen_run_1",
        "sourceAnnotationsSha256": "a" * 64,
        "createdAt": "2026-08-06T14:30:12Z",
        "updatedAt": "2026-08-06T14:31:44Z",
        "revisionToken": operation_count,
        "operationCount": operation_count,
    }


def create_operation() -> dict:
    return {
        "schema_version": 1,
        "operationId": "op_001",
        "operationType": "create",
        "targetId": "E2",
        "before": None,
        "after": {"geometry": {"kind": "circle", "centroidXy": [2.0, 1.0], "radiusPx": 1.5}},
        "evidenceViewId": "msica",
        "uiFrame": 2,
        "sourceXy": [2.0, 1.0],
        "reviewerId": "reviewer_local_1",
        "timestamp": "2026-08-06T14:31:44Z",
        "expectedRevisionToken": 0,
    }


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def test_tiny_raw_msica_msln_fixture_is_aligned_and_valid() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    contracts = {
        contract.payload["view_id"]: contract
        for contract in (AnnotationViewContract.from_dict(item) for item in fixture["view_contracts"])
    }

    assert set(contracts) == {"raw", "msica", "msln"}
    for view_id, frames in fixture["arrays_tyx"].items():
        contract = contracts[view_id]
        assert len(frames) == contract.payload["shape_tyx"][0]
        assert all(len(frame) == contract.payload["shape_tyx"][1] for frame in frames)
        assert all(len(row) == contract.payload["shape_tyx"][2] for frame in frames for row in frame)
        assert contract.source_xy_to_view(2, 1) == (2.0, 1.0)

    assert fixture["arrays_tyx"]["raw"][1][1][2] == 9
    assert fixture["arrays_tyx"]["msica"][1][1][2] == 8
    assert fixture["arrays_tyx"]["msln"][1][1][2] == 6
    assert fixture["model_roi"]["status"] == "unknown"


def test_affine_view_transform_round_trips_source_coordinates() -> None:
    contract = AnnotationViewContract.from_dict(
        {
            "schema_version": 1,
            "view_id": "cropped_msln",
            "source_video_id": "fish",
            "shape_tyx": [3, 4, 5],
            "source_to_view": {
                "kind": "affine",
                "matrix_3x3": [[2.0, 0.0, -4.0], [0.0, 0.5, 3.0], [0.0, 0.0, 1.0]],
            },
            "frame_mapping": {"kind": "identity", "offset": 2},
            "intensity_semantics": "normalized_signed_visualization",
            "frame_pattern": "frames/msln/frame_%04d.png",
        }
    )

    view_xy = contract.source_xy_to_view(5.0, 8.0)

    assert view_xy == (6.0, 7.0)
    assert contract.view_xy_to_source(*view_xy) == pytest.approx((5.0, 8.0))
    assert contract.source_frame_to_view_index(4) == 6
    assert contract.view_frame_to_source_index(6) == 4
    assert contract.to_dict()["frame_pattern"] == "frames/msln/frame_%04d.png"


def test_view_contract_rejects_singular_or_non_affine_matrix() -> None:
    base = {
        "schema_version": 1,
        "view_id": "bad",
        "source_video_id": "fish",
        "shape_tyx": [3, 4, 5],
        "frame_mapping": {"kind": "identity", "offset": 0},
        "intensity_semantics": "raw_amplitude",
    }
    with pytest.raises(ValueError, match="invertible"):
        AnnotationViewContract.from_dict(
            {
                **base,
                "source_to_view": {
                    "kind": "affine",
                    "matrix_3x3": [[1, 2, 0], [2, 4, 0], [0, 0, 1]],
                },
            }
        )
    with pytest.raises(ValueError, match="bottom row"):
        AnnotationViewContract.from_dict(
            {
                **base,
                "source_to_view": {
                    "kind": "affine",
                    "matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
                },
            }
        )


def test_operation_semantics_reject_destructive_history_shapes() -> None:
    valid = AnnotationOperation.from_dict(create_operation())
    assert valid.to_dict()["operationType"] == "create"

    invalid_create = create_operation()
    invalid_create["before"] = {"id": "unexpected"}
    with pytest.raises(ValueError, match="before value must be null"):
        AnnotationOperation.from_dict(invalid_create)

    invalid_tombstone = create_operation()
    invalid_tombstone.update(
        {
            "operationType": "tombstone",
            "before": {"id": "E2"},
            "after": {"deleted": True},
        }
    )
    with pytest.raises(ValueError, match="after value must be null"):
        AnnotationOperation.from_dict(invalid_tombstone)

    invalid_move = create_operation()
    invalid_move.update({"operationType": "move", "before": {"id": "E2"}, "after": None})
    with pytest.raises(ValueError, match="requires an after value"):
        AnnotationOperation.from_dict(invalid_move)


def test_revision_contract_rejects_token_count_mismatch_or_self_parent() -> None:
    mismatched = revision_payload()
    mismatched["revisionToken"] = 1
    with pytest.raises(ValueError, match="must equal"):
        AnnotationRevision.from_dict(mismatched)

    self_parent = revision_payload()
    self_parent["parentRevisionId"] = self_parent["revisionId"]
    with pytest.raises(ValueError, match="must differ"):
        AnnotationRevision.from_dict(self_parent)


def test_revision_root_is_complete_loadable_and_collision_safe(tmp_path: Path) -> None:
    revision = revision_payload(operation_count=1)
    operation = create_operation()
    target = initialize_revision_root(
        tmp_path / "annotation_revisions",
        revision=revision,
        annotations={"rois": {"E2": {"cell_state": "accepted"}}},
        operations=[operation],
    )

    assert sorted(item.name for item in target.iterdir()) == [
        "annotations.json",
        "exports",
        "operations.jsonl",
        "revision.json",
    ]
    loaded_revision, loaded_annotations, loaded_operations = load_revision_root(target)
    assert loaded_revision.to_dict() == revision
    assert loaded_annotations.to_dict()["rois"]["E2"]["cell_state"] == "accepted"
    assert [item.to_dict() for item in loaded_operations] == [operation]

    before = tree_digest(target)
    with pytest.raises(FileExistsError, match="Refusing annotation revision collision"):
        initialize_revision_root(
            tmp_path / "annotation_revisions",
            revision=revision,
            annotations={"rois": {"E2": {"cell_state": "rejected"}}},
            operations=[operation],
        )
    assert tree_digest(target) == before
    assert not list((tmp_path / "annotation_revisions").glob(".ann_test.partial-*"))


def test_revision_root_rejects_noncontiguous_tokens_before_writing(tmp_path: Path) -> None:
    operation = create_operation()
    operation["expectedRevisionToken"] = 4

    with pytest.raises(ValueError, match="contiguous from zero"):
        initialize_revision_root(
            tmp_path / "annotation_revisions",
            revision=revision_payload(operation_count=1),
            annotations={},
            operations=[operation],
        )

    assert not (tmp_path / "annotation_revisions" / "ann_test").exists()


def test_draft_operation_updates_projection_and_token_atomically(tmp_path: Path) -> None:
    revisions = tmp_path / "annotation_revisions"
    before = {
        "source_xy": [2.0, 1.0],
        "geometry": {"kind": "circle", "radius_px": 1.5},
        "deleted": False,
    }
    root = initialize_revision_root(
        revisions,
        revision=revision_payload(),
        annotations={"rois": {"E2": before}},
    )
    before = load_revision_root(root)[1].to_dict()["rois"]["E2"]
    operation = create_operation()
    operation.update(
        {
            "operationId": "op_move_001",
            "operationType": "move",
            "before": before,
            "after": {**before, "source_xy": [3.0, 2.0]},
        }
    )

    snapshot = append_revision_operation(root, operation)

    assert snapshot["revision"]["revisionToken"] == 1
    assert snapshot["revision"]["operationCount"] == 1
    assert snapshot["annotations"]["rois"]["E2"]["source_xy"] == [3.0, 2.0]
    assert snapshot["operations"] == [operation]
    assert revision_snapshot(root) == snapshot
    assert [item["revisionId"] for item in list_revision_roots(revisions)] == ["ann_test"]
    assert not list(root.glob(".*.partial-*"))


def test_draft_operation_refuses_stale_token_without_changing_revision(tmp_path: Path) -> None:
    before = {"source_xy": [2.0, 1.0], "geometry": {"kind": "circle", "radius_px": 1.5}}
    root = initialize_revision_root(
        tmp_path / "annotation_revisions",
        revision=revision_payload(),
        annotations={"rois": {"E2": before}},
    )
    before = load_revision_root(root)[1].to_dict()["rois"]["E2"]
    first = create_operation()
    first.update(
        {
            "operationId": "op_move_001",
            "operationType": "move",
            "before": before,
            "after": {**before, "source_xy": [3.0, 2.0]},
        }
    )
    append_revision_operation(root, first)
    digest = tree_digest(root)
    stale = dict(first)
    stale["operationId"] = "op_move_stale"

    with pytest.raises(RevisionConflictError, match="revision token conflict"):
        append_revision_operation(root, stale)

    assert tree_digest(root) == digest


def test_tombstone_preserves_projection_and_restore_is_append_only(tmp_path: Path) -> None:
    before = {"source_xy": [2.0, 1.0], "geometry": {"kind": "circle", "radius_px": 1.5}}
    root = initialize_revision_root(
        tmp_path / "annotation_revisions",
        revision=revision_payload(),
        annotations={"rois": {"E2": before}},
    )
    before = load_revision_root(root)[1].to_dict()["rois"]["E2"]
    tombstone = create_operation()
    tombstone.update(
        {
            "operationId": "op_tombstone_001",
            "operationType": "tombstone",
            "before": before,
            "after": None,
        }
    )
    tombstoned = append_revision_operation(root, tombstone)
    tombstoned_projection = tombstoned["annotations"]["rois"]["E2"]
    restore = create_operation()
    restore.update(
        {
            "operationId": "op_restore_001",
            "operationType": "restore",
            "before": tombstoned_projection,
            "after": {**before, "deleted": False},
            "expectedRevisionToken": 1,
        }
    )

    restored = append_revision_operation(root, restore)

    assert restored["annotations"]["rois"]["E2"]["deleted"] is False
    assert [item["operationType"] for item in restored["operations"]] == ["tombstone", "restore"]


def test_link_event_interval_and_unlink_operations_remain_explicit(tmp_path: Path) -> None:
    root = initialize_revision_root(
        tmp_path / "annotation_revisions",
        revision=revision_payload(),
        annotations={"rois": {"E2": {"source_xy": [2.0, 1.0], "geometry": {"kind": "circle", "radius_px": 1.5}}}},
    )
    before = load_revision_root(root)[1].to_dict()["rois"]["E2"]
    link = create_operation()
    link.update(
        {
            "operationId": "op_link_001",
            "operationType": "link",
            "before": before,
            "after": {**before, "linked_model_id": "M2"},
        }
    )
    linked = append_revision_operation(root, link)
    before_event = linked["annotations"]["rois"]["E2"]
    event_edit = create_operation()
    event_edit.update(
        {
            "operationId": "op_event_001",
            "operationType": "edit-event-interval",
            "before": before_event,
            "after": {**before_event, "event_intervals": [[2, 4], [7, 7]], "events": [2, 7]},
            "expectedRevisionToken": 1,
        }
    )
    event_updated = append_revision_operation(root, event_edit)
    before_unlink = event_updated["annotations"]["rois"]["E2"]
    unlink = create_operation()
    unlink.update(
        {
            "operationId": "op_unlink_001",
            "operationType": "unlink",
            "before": before_unlink,
            "after": {**before_unlink, "linked_model_id": ""},
            "expectedRevisionToken": 2,
        }
    )

    snapshot = append_revision_operation(root, unlink)

    assert snapshot["revision"]["revisionToken"] == 3
    assert snapshot["annotations"]["rois"]["E2"]["linked_model_id"] == ""
    assert snapshot["annotations"]["rois"]["E2"]["event_intervals"] == [[2, 4], [7, 7]]
    assert [item["operationType"] for item in snapshot["operations"]] == [
        "link",
        "edit-event-interval",
        "unlink",
    ]


def test_promote_creates_new_expert_without_mutating_model_evidence(tmp_path: Path) -> None:
    root = initialize_revision_root(
        tmp_path / "annotation_revisions",
        revision=revision_payload(),
        annotations={},
    )
    proposal = {
        "proposal_id": "M2",
        "source_xy": [3.0, 2.0],
        "geometry": {"kind": "center"},
        "status": "unknown",
    }
    promoted = {
        "id": "E_promoted_M2",
        "annotation_correction_kind": "expert",
        "source_xy": [3.0, 2.0],
        "ui_frame": 2,
        "events": [2],
        "event_intervals": [[2, 2]],
        "geometry": {"kind": "circle", "radius_px": 0.75},
        "linked_model_id": "M2",
        "promoted_from_model_id": "M2",
        "review_state": "recently_edited",
        "deleted": False,
    }
    operation = create_operation()
    operation.update(
        {
            "operationId": "op_promote_001",
            "operationType": "promote",
            "targetId": "E_promoted_M2",
            "before": proposal,
            "after": promoted,
            "sourceXy": [3.0, 2.0],
        }
    )

    snapshot = append_revision_operation(root, operation)

    assert snapshot["annotations"]["rois"]["E_promoted_M2"]["promoted_from_model_id"] == "M2"
    assert snapshot["operations"][0]["before"] == proposal


def test_operation_semantics_reject_invalid_links_promotions_and_intervals() -> None:
    before = {"id": "E2", "linked_model_id": ""}
    invalid_link = create_operation()
    invalid_link.update({"operationType": "link", "before": before, "after": before})
    with pytest.raises(ValueError, match="requires linked_model_id"):
        AnnotationOperation.from_dict(invalid_link)

    invalid_promotion = create_operation()
    invalid_promotion.update(
        {
            "operationType": "promote",
            "before": {"proposal_id": "M2"},
            "after": {"promoted_from_model_id": "M1"},
        }
    )
    with pytest.raises(ValueError, match="retain its model proposal ID"):
        AnnotationOperation.from_dict(invalid_promotion)

    invalid_interval = create_operation()
    invalid_interval.update(
        {
            "operationType": "edit-event-interval",
            "before": before,
            "after": {**before, "event_intervals": [[0, 2]]},
        }
    )
    with pytest.raises(ValueError, match="one-based inclusive"):
        AnnotationOperation.from_dict(invalid_interval)


def test_fork_constructs_fresh_draft_from_current_projection(tmp_path: Path) -> None:
    revisions = tmp_path / "annotation_revisions"
    original = {"source_xy": [2.0, 1.0], "geometry": {"kind": "circle", "radius_px": 1.5}}
    source = initialize_revision_root(
        revisions,
        revision=revision_payload(revision_id="ann_source"),
        annotations={"rois": {"E2": original}},
    )
    original = load_revision_root(source)[1].to_dict()["rois"]["E2"]
    operation = create_operation()
    operation.update(
        {
            "operationId": "op_fork_move",
            "operationType": "move",
            "before": original,
            "after": {**original, "source_xy": [3.0, 2.0]},
        }
    )
    append_revision_operation(source, operation)
    source_digest = tree_digest(source)

    fork = fork_revision_root(
        source,
        revisions,
        revision_id="ann_fork",
        reviewer_id="reviewer_local_2",
        timestamp="2026-08-06T15:00:00Z",
    )
    snapshot = revision_snapshot(fork)

    assert snapshot["revision"]["parentRevisionId"] == "ann_source"
    assert snapshot["revision"]["state"] == "draft"
    assert snapshot["revision"]["reviewerId"] == "reviewer_local_2"
    assert snapshot["revision"]["revisionToken"] == 0
    assert snapshot["revision"]["operationCount"] == 0
    assert snapshot["operations"] == []
    assert snapshot["annotations"]["rois"]["E2"]["source_xy"] == [3.0, 2.0]
    assert tree_digest(source) == source_digest


def test_publish_creates_immutable_child_without_rewriting_draft(tmp_path: Path) -> None:
    revisions = tmp_path / "annotation_revisions"
    original = {"source_xy": [2.0, 1.0], "geometry": {"kind": "circle", "radius_px": 1.5}}
    draft = initialize_revision_root(
        revisions,
        revision=revision_payload(revision_id="ann_draft"),
        annotations={"rois": {"E2": original}},
    )
    original = load_revision_root(draft)[1].to_dict()["rois"]["E2"]
    operation = create_operation()
    operation.update(
        {
            "operationId": "op_publish_move",
            "operationType": "move",
            "before": original,
            "after": {**original, "source_xy": [3.0, 2.0]},
        }
    )
    append_revision_operation(draft, operation)
    draft_digest = tree_digest(draft)

    published = publish_revision_root(
        draft,
        revisions,
        revision_id="ann_published",
        expected_revision_token=1,
        timestamp="2026-08-06T15:05:00Z",
    )
    snapshot = revision_snapshot(published)

    assert snapshot["revision"]["parentRevisionId"] == "ann_draft"
    assert snapshot["revision"]["state"] == "published"
    assert snapshot["revision"]["revisionToken"] == 1
    assert snapshot["operations"] == [operation]
    assert snapshot["annotations"]["rois"]["E2"]["source_xy"] == [3.0, 2.0]
    assert tree_digest(draft) == draft_digest
    with pytest.raises(ValueError, match="immutable"):
        append_revision_operation(published, operation)
    with pytest.raises(FileExistsError, match="collision"):
        publish_revision_root(
            draft,
            revisions,
            revision_id="ann_published",
            expected_revision_token=1,
            timestamp="2026-08-06T15:06:00Z",
        )


def test_publish_refuses_stale_token_without_creating_child(tmp_path: Path) -> None:
    revisions = tmp_path / "annotation_revisions"
    draft = initialize_revision_root(
        revisions,
        revision=revision_payload(revision_id="ann_stale_draft"),
        annotations={"rois": {}},
    )

    with pytest.raises(RevisionConflictError, match="token conflict"):
        publish_revision_root(
            draft,
            revisions,
            revision_id="ann_stale_publish",
            expected_revision_token=1,
            timestamp="2026-08-06T15:10:00Z",
        )

    assert not (revisions / "ann_stale_publish").exists()
