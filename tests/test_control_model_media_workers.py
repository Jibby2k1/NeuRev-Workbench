"""Tiny original-renderer parity and fail-closed model checkpoint merging."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil

import pytest

from neurobench.experiments.gamma_ls_difference import control_audit as original
from neurobench.experiments.gamma_ls_difference import control_model_media_workers as workers


def test_pending_partitions_are_disjoint_and_preserve_original_inventory_order(tmp_path):
    plan = {"model_sites": [{"model_roi_id": f"model_{i:03d}"} for i in range(8)]}
    video, other = workers._roi_paths(tmp_path, "model_002")
    result = workers.partition_pending(tmp_path, plan, {str(video): {}}, {str(path): {} for path in other}, 3)
    assert result == [["model_000", "model_004", "model_007"], ["model_001", "model_005"], ["model_003", "model_006"]]
    assert len({name for part in result for name in part}) == 7
    with pytest.raises(ValueError, match="one to three"):
        workers.partition_pending(tmp_path, plan, {}, {}, 4)


def test_live_original_writer_and_duplicate_lock_are_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="must have exited"):
        workers._assert_stopped(os.getpid())
    with workers._lock(tmp_path / "worker.lock"):
        with pytest.raises(RuntimeError, match="owns"):
            with workers._lock(tmp_path / "worker.lock"):
                pytest.fail("Duplicate worker entered exclusive section")


def test_nondumpable_unrelated_process_is_recorded_but_relevant_renderer_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "run/audits/legacy_carrier"; workspace = tmp_path / "workers"
    workspace.mkdir(); proc = tmp_path / "proc"; process = proc / "3422"
    (process / "fd").mkdir(parents=True); (process / "fd/0").touch()
    (process / "stat").write_text("3422 (systemd) S 1 0 0\n")
    command = process / "cmdline"; command.write_bytes(b"/usr/lib/systemd/systemd\0--user\0")
    def denied_descriptor(path):
        raise PermissionError("nondumpable descriptor")
    monkeypatch.setattr(workers.os, "readlink", denied_descriptor)
    assert workers._external_writers(root, workspace, proc_root=proc) == [dict(pid=3422,
        unavailable_inspection=["descriptor_target_or_flags"], command_line_inspected=True,
        command_classification="unrelated_or_not_targeting_this_run")]
    command.write_bytes(f"python\0-m\0package.control_render_workers\0--output\0{root.parent.parent}\0".encode())
    with pytest.raises(RuntimeError, match="Relevant original renderer"):
        workers._external_writers(root, workspace, proc_root=proc)


def test_visible_writable_audit_descriptor_is_rejected_even_for_other_command(tmp_path, monkeypatch):
    root = tmp_path / "audit"; workspace = tmp_path / "workers"; workspace.mkdir()
    proc = tmp_path / "proc"; process = proc / "4444"
    (process / "fd").mkdir(parents=True); (process / "fd/1").touch()
    (process / "fdinfo").mkdir(); (process / "fdinfo/1").write_text("flags:\t0100001\n")
    (process / "stat").write_text("4444 (ffmpeg) S 1 0 0\n")
    (process / "cmdline").write_bytes(b"ffmpeg\0")
    monkeypatch.setattr(workers.os, "readlink", lambda path: str(root / "video.partial.mp4"))
    with pytest.raises(RuntimeError, match="open for writing"):
        workers._external_writers(root, workspace, proc_root=proc)
    assert workers._renderer_targets_audit(["python", "-m", "control_study", "audit", "--output", str(root)], root)
    assert not workers._renderer_targets_audit(["python", "-m", "control_study", "evaluate", "--output", str(root)], root)
    assert not workers._renderer_targets_audit(["python", "-m", "control_audit", "--output", "/other/run"], root)


def _fixture(tmp_path):
    import numpy as np
    frames = list(range(1800, 1808))
    raw = np.full((8, 48, 64), 4.125000000123, dtype=np.float64)
    score = np.zeros(raw.shape, dtype=np.float32)
    rows = []
    for index, (x, y) in enumerate(((12, 12), (30, 22), (50, 35))):
        for offset in (2, 4):
            value = float(2 + index + offset)
            score[offset, y, x] = value
            rows.append(dict(proposal_id=f"p{index}_{offset}", x_px=x, y_px=y,
                source_frame_ui=1800+offset, score=value, threshold=1, target_proposals_per_frame=1))
    paths = {}
    for name in ("Raw", "Input", "Score"):
        path = tmp_path / f"{name}.npy"
        np.save(path, score if name == "Score" else raw)
        paths[name] = path
    candidate_path = tmp_path / "candidates.tsv"
    labels_path = tmp_path / "experts.tsv"
    labels = [dict(canonical_roi_id="roi_001", observation_id="b1_roi1", burst_id=1,
        source_start_ui=1802, source_stop_ui=1805, x_px=12.25, y_px=12.25,
        temporal_extent_semantics="configured_burst_window_not_per_roi_onset")]
    for path, content in ((candidate_path, rows), (labels_path, labels)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(content)
    numeric = {}
    for q in (.25, .5, 1, 2, 5):
        path = tmp_path / f"q{q}.json"; path.write_text(json.dumps(dict(target_proposals_per_frame=q)))
        numeric[str(q)] = path
    return dict(root=tmp_path / "audit", stage_paths=paths, display_limits={"Raw": [0, 10], "Input": [0, 10], "Score": [-10, 10]},
        signed_stage_keys=["Score"], source_frames_ui=frames, candidates_path=candidate_path,
        expert_occurrences=labels_path, operating_point=dict(target_proposals_per_frame=1, threshold=1,
            application_source_start_ui=1802, application_source_stop_ui=1807),
        source_binding=dict(stage_sha256={name: original._digest(path) for name, path in paths.items()},
            candidates_sha256=original._digest(candidate_path), expert_occurrences_sha256=original._digest(labels_path)),
        fullnumeric_q_paths=numeric, context_frames=1)


def _stopped_pid(monkeypatch):
    # Deterministic terminated-writer fixture; active-writer refusal is separately tested.
    actual = workers._alive
    monkeypatch.setattr(workers, "_alive", lambda pid: False if pid == 123456789 else actual(pid))
    return 123456789


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required for exact media parity")
def test_worker_artifacts_equal_original_bytes_and_source_rgb_before_original_resume(tmp_path, monkeypatch):
    config = _fixture(tmp_path); root = config["root"]
    frozen_sources = {str(path): original._digest(path) for path in (Path(original.__file__), Path(original._base.__file__))}
    result = original.run_control_audit(**config)
    assert result["scientific_audit_complete"] and result["model_roi_count"] == 3
    plan = workers._read(root / "inventory_plan.json")
    original_videos = {row["path"]: row for row in workers._read(root / "video_manifest.json")["videos"]}
    expected = {}
    model_keys = [set(), set()]
    for location in plan["model_sites"]:
        video, others = workers._roi_paths(root, location["model_roi_id"])
        model_keys[0].add(str(video)); model_keys[1].update(map(str, others))
        for path in [video, video.with_suffix(".png"), *others]:
            expected[path] = path.read_bytes()
            path.unlink()
    # Simulate an interrupted original model loop, retaining its actual fullfield
    # and expert checkpoints. All changes are confined to this disposable fixture.
    for index, name in enumerate(workers.CHECKPOINTS):
        payload = workers._read(root / name)
        original._json(root / name, {key: value for key, value in payload.items() if key not in model_keys[index]})
    original._json(root / "status.json", dict(status="rendering", scientific_audit_complete=False))
    session = workers.prepare_workers(root, stopped_writer_pid=_stopped_pid(monkeypatch), workers=3)
    workspace = workers._workspace(root)
    assert session["pending_model_roi_count"] == 3 and list(map(len, session["partitions"])) == [1, 1, 1]
    canonical = {name: (root / name).read_bytes() for name in workers.CHECKPOINTS}
    with pytest.raises(FileNotFoundError):
        workers.merge_workers(root)
    for index in range(3):
        # Production starts separate processes: each worker must reproduce PNG
        # bytes from a newly constructed canvas, not one primed by the baseline.
        _, _, _, plt = original._load_deps()
        for canvas in original._TRACE_CANVASES.values():
            plt.close(canvas[0])
        original._TRACE_CANVASES.clear()
        assert workers.run_worker(root, index)["status"] == "COMPLETE"
    for name, value in canonical.items():
        assert (root / name).read_bytes() == value
    for path, value in expected.items():
        assert path.read_bytes() == value, f"Original/partitioned artifact bytes differ: {path}"
    for index in range(3):
        records = workers._read(workspace / f"worker_{index}/render_checkpoint.json")
        for identifier in session["partitions"][index]:
            video, _ = workers._roi_paths(root, identifier)
            record = records[str(video)]; baseline = original_videos[str(video)]
            assert record == baseline  # Includes all source RGB, decoded RGB, per-frame markers and geometry.
            assert record["all_frames_rgb_byte_exact"] and record["all_frames_marker_expectations_pass"]
    # Completed worker records cannot silently alter shared original keys.
    first = workspace / "worker_0"; checkpoint_path = first / "render_checkpoint.json"
    payload = workers._read(checkpoint_path); shared_key = next(key for key in payload if key not in model_keys[0])
    preserved = checkpoint_path.read_bytes(); payload[shared_key]["frame_count"] += 1
    original._json(checkpoint_path, payload)
    status_path = first / "status.json"; status = workers._read(status_path)
    for record in status["checkpoints"]:
        if record["path"] == str(checkpoint_path): record.update(workers._record(checkpoint_path))
    original._json(status_path, status)
    with pytest.raises(ValueError, match="Overlapping checkpoint"):
        workers.merge_workers(root)
    checkpoint_path.write_bytes(preserved)
    for record in status["checkpoints"]:
        if record["path"] == str(checkpoint_path): record.update(workers._record(checkpoint_path))
    original._json(status_path, status)
    merged = workers.merge_workers(root)
    assert merged["status"] == "MERGED" and merged["completed_pending_model_rois"] == 3
    assert not merged["scientific_audit_complete"]
    assert workers.merge_workers(root)["status"] == "MERGED"
    # Original full code, not the helper, performs scientific completion.
    assert original.run_control_audit(**config)["scientific_audit_complete"]
    for path, digest in frozen_sources.items():
        assert original._digest(Path(path)) == digest
