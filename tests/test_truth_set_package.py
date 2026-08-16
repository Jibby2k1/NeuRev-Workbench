from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pytest

from neurobench.workbench.truth_set import audit_truth_set_root, build_truth_set_package, load_candidate_source_key, lock_raw_first_region, preflight_truth_set, publish_truth_set, record_adjudication, record_candidate_dispositions, record_second_review, reveal_candidate_assisted_payload, second_review_selection, unseal_protected


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "spon_ca_burst_truth_set_v1.example.json"


def test_preflight_is_read_only_and_validates_source_and_masks() -> None:
    before = hashlib.sha256(EXAMPLE.read_bytes()).hexdigest()
    result = preflight_truth_set(EXAMPLE)
    assert result["passed"]
    assert hashlib.sha256(EXAMPLE.read_bytes()).hexdigest() == before
    assert result["checks"][-1]["name"] == "protected_region_label_free_selection"


def test_package_is_collision_safe_and_private_key_is_not_in_browser_payload(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    result = build_truth_set_package(EXAMPLE, root)
    payload = json.loads((root / "review" / "review_payload.json").read_text())
    assert result["candidate_count"] == 4
    assert payload["review_pass"] == "raw_first" and payload["candidates"] == []
    assert (root / "private" / "candidate_source_key.json").is_file()
    assert "source_lane" not in json.dumps(payload).lower()
    with pytest.raises(FileExistsError, match="collision"):
        build_truth_set_package(EXAMPLE, root)


def test_raw_first_must_lock_before_candidates_are_revealed(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    with pytest.raises(ValueError, match="raw-first"):
        reveal_candidate_assisted_payload(root)
    manifest = json.loads((root / "truth_set_manifest.json").read_text())
    for region in manifest["regions"]:
        region["raw_first_complete"] = True
        region["review_status"] = "raw_first_locked"
    (root / "truth_set_manifest.json").write_text(json.dumps(manifest))
    payload = reveal_candidate_assisted_payload(root)
    assert payload["review_pass"] == "candidate_assisted"
    assert len(payload["candidates"]) == 4


def test_raw_first_can_add_detector_missed_object_and_private_key_stays_sealed(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    obj = {"object_id": "manual_missed", "region_id": "calibration_synthetic", "geometry": {"kind": "center", "x": 4, "y": 3}, "support_ui_frames": [2, 4], "disposition": "neuron", "morphology": "center", "context": "isolated", "visibility_confidence": "high", "quality_flags": {"motion": False}, "reviewer_id": "synthetic_reviewer_a", "source_provenance": "raw_first_manual", "annotation_revision_id": "synthetic_raw_first_publication"}
    event = {"event_id": "manual_event", "object_id": "manual_missed", "region_id": "calibration_synthetic", "onset_ui": 2, "peak_ui": 3, "end_ui": 4, "array_interval": [1, 4], "disposition": "event", "visibility_confidence": "high", "timing_confidence": "high", "quality_flags": {"motion": False}, "reviewer_id": "synthetic_reviewer_a"}
    lock_raw_first_region(root, region_id="calibration_synthetic", published_revision_id="synthetic_raw_first_publication", objects=[obj], events=[event])
    assert "manual_missed" in (root / "annotations" / "objects.tsv").read_text()
    with pytest.raises(PermissionError, match="sealed"):
        load_candidate_source_key(root)


def test_candidate_assisted_review_refuses_missing_frozen_lane_manifest(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    (root / "freeze" / "frozen_lane_manifest.json").unlink()
    with pytest.raises(ValueError, match="frozen lane"):
        reveal_candidate_assisted_payload(root)


def test_protected_unseal_refuses_frozen_manifest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    frozen_path = root / "freeze" / "frozen_lane_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["analysis_policy"]["nms_radius_px"] = 999
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="changed"):
        unseal_protected(root, reviewer_id="reviewer", reason="test")


def test_a0_initially_reports_exact_incomplete_blockers(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    result = audit_truth_set_root(root)
    assert result["decision"] == "incomplete"
    assert any("raw-first" in blocker for blocker in result["blockers"])
    assert any("undispositioned" in blocker for blocker in result["blockers"])
    assert result["meaning"].startswith("A0 advance")


def test_second_review_selection_is_written_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    dispositions = [{"subject_id": f"a{i}", "disposition": "accepted"} for i in range(5)] + [{"subject_id": "u", "disposition": "unresolved"}]
    first = second_review_selection(root, dispositions, seed=3)
    second = second_review_selection(root, dispositions, seed=3)
    assert first == second and "u" in first
    assert json.loads((root / "review" / "second_review_sample.json").read_text())["fraction"] == 0.20


def test_required_output_layout_is_present(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    required = ["truth_set_manifest.json", "resolved_manifest.json", "input_fingerprints.json", "candidate_panel/blinded_candidates.tsv", "annotations/objects.tsv", "annotations/events.tsv", "review/agreement.json", "freeze/protected_lock.json", "metrics/grouped_intervals.json", "artifact_index.json", "validation.json", "llm_context.json", "REPORT.md"]
    assert all((root / item).is_file() for item in required)


def test_cli_exposes_nested_truth_set_commands() -> None:
    from neurobench.cli.main import build_parser
    parser = build_parser(active_command="workbench")
    args = parser.parse_args(["workbench", "truth-set", "preflight", "--manifest", str(EXAMPLE), "--json"])
    assert args.truth_set_command == "preflight" and args.manifest == EXAMPLE


def test_server_reads_truth_set_payload_without_parallel_dashboard(tmp_path: Path) -> None:
    from neurobench.workbench.server import create_workbench_server
    app = tmp_path / "demo" / "app"
    app.mkdir(parents=True)
    (app / "review_data.json").write_text(json.dumps({"dataset": {"dataset_id": "demo"}, "video": {"name": "tiny", "width": 1, "height": 1, "frames": 1}, "rois": []}))
    build_truth_set_package(EXAMPLE, app / "truth_set")
    server, _ = create_workbench_server(app_dir=app, host="127.0.0.1", port=0, asset_mode="current")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with urlopen(f"http://{host}:{port}/api/truth-set", timeout=5) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert payload["mode"] == "truth_set" and payload["review_pass"] == "raw_first"


def test_tiny_end_to_end_fixture_reaches_a0_without_detector_claim(tmp_path: Path) -> None:
    root = tmp_path / "truth"
    build_truth_set_package(EXAMPLE, root)
    for index, region_id in enumerate(("calibration_synthetic", "protected_synthetic"), 1):
        object_id = f"manual_{index}"
        obj = {"object_id": object_id, "region_id": region_id, "geometry": {"kind": "center", "x": index, "y": index}, "support_ui_frames": [2, 3], "disposition": "neuron", "morphology": "center", "context": "isolated", "visibility_confidence": "high", "quality_flags": {"motion": False}, "reviewer_id": "synthetic_reviewer_a", "source_provenance": "raw_first_manual", "annotation_revision_id": "synthetic_raw_first_publication"}
        event = {"event_id": f"event_{index}", "object_id": object_id, "region_id": region_id, "onset_ui": 2, "peak_ui": 2, "end_ui": 3, "array_interval": [1, 3], "disposition": "event", "visibility_confidence": "high", "timing_confidence": "high", "quality_flags": {"motion": False}, "reviewer_id": "synthetic_reviewer_a"}
        lock_raw_first_region(root, region_id=region_id, published_revision_id="synthetic_raw_first_publication", objects=[obj], events=[event])
    payload = reveal_candidate_assisted_payload(root)
    dispositions = [{"candidate_id": item["candidate_id"], "subject_id": item["candidate_id"], "disposition": "neuron" if index % 2 == 0 else "artifact"} for index, item in enumerate(payload["candidates"])]
    record_candidate_dispositions(root, dispositions)
    sample = second_review_selection(root, [{"subject_id": item["candidate_id"], "disposition": "accepted" if item["disposition"] == "neuron" else "rejected"} for item in dispositions], seed=9)
    annotations = {"rois": {subject_id: {"cell_state": "accepted", "reviewer_id": "reviewer"} for subject_id in sample}, "events": {}, "suggestions": {}}
    record_second_review(root, selected_subject_ids=sample, reviewer_a=annotations, reviewer_b=annotations)
    record_adjudication(root, decisions=[])
    unseal_protected(root, reviewer_id="synthetic_adjudicator", reason="tiny fixture completed", timestamp="2026-08-16T01:00:00Z")
    result = audit_truth_set_root(root)
    assert result["decision"] == "advance"
    assert "not a detector pass" in result["meaning"]
    assert load_candidate_source_key(root)["sealed"] is True
    published = publish_truth_set(root, timestamp="2026-08-16T01:01:00Z")
    assert published["truth_set_id"] == "spon_ca_burst_synthetic_truth_set_v1"
    with pytest.raises(FileExistsError, match="already published"):
        publish_truth_set(root)
