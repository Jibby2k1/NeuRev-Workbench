"""Partition only unfinished model media; the frozen audit remains the validator.

Stop the original audit writer before prepare, and do not restart it until merge
finishes. The frozen renderer cannot honor this helper's advisory locks. We also
require that writer PID to be gone, detect external writable descriptors, and
reject changes to canonical checkpoint snapshots. Each worker must run in its
own process; at most three workers, each with numerical and codec threads=1.
No detector, source array, original renderer, or scientific status is changed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from . import control_audit as original
from .control_audit import (
    ORANGE, SOURCE_TIME_SEMANTICS, _Progress, _artifact_cached,
    _canonical_digest, _checkpoint_artifact, _digest, _frozen_threshold,
    _json, _load_deps, _panel, _rows, _table, _trace_plot,
    _verify_candidate_score_binding, _write_video, audit_inventory_plan,
    identity_crop, panel_image_geometry,
)

CHECKPOINTS = ("render_checkpoint.json", "figure_checkpoint.json")


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _record(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=_digest(path), size_bytes=path.stat().st_size)


def _verify(record: Mapping[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["size_bytes"] or _digest(path) != record["sha256"]:
        raise ValueError(f"Changed bound file: {path}")


def _workspace(root: Path) -> Path:
    # Outside the original audit index: final validation must index audit media,
    # not mutable worker heartbeats, locks, or copies of full checkpoints.
    return root.parent / f".{root.name}_model_media_workers"


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _assert_stopped(pid: int) -> None:
    if pid <= 0 or _alive(pid):
        raise RuntimeError("The original audit writer PID must have exited before model workers run")


def _renderer_targets_audit(command: list[str], root: Path) -> bool:
    text = " ".join(command)
    renderer = any(name in text for name in ("control_render_workers", "control_audit"))
    renderer |= "control_study" in text and ("audit" in command or "audit(" in text)
    campaign = root.parent.parent if root.parent.name == "audits" else root
    return renderer and (str(root) in text or str(campaign) in text)


@contextlib.contextmanager
def _lock(path: Path, *, shared: bool = False):
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another prepare, merge, or worker owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _external_writers(root: Path, workspace: Path, *, proc_root: Path = Path("/proc")) -> list[dict]:
    """Reject relevant renderers/visible writers; report inaccessible proc data.

    Nondumpable desktop services are not evidence of an active audit writer.
    Their inaccessible descriptors are recorded, not silently assumed inspected.
    """
    allowed = {os.getpid()}
    for path in workspace.glob("worker_*/status.json"):
        status = _read(path)
        if status.get("status") == "RUNNING":
            allowed.add(int(status["pid"]))
    processes = {}; blind = {}
    def unavailable(pid, field):
        blind.setdefault(pid, set()).add(field)
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():
            continue
        pid = int(folder.name); parent = None; command = None
        try:
            fields = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
            parent = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unavailable(pid, "process_stat")
        try:
            command = [value.decode(errors="replace") for value in (folder / "cmdline").read_bytes().split(b"\0") if value]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unavailable(pid, "command_line")
        processes[pid] = (parent, folder, command)
    for _ in range(len(processes)):
        descendants = {pid for pid, (parent, _, _) in processes.items() if parent in allowed}
        if descendants <= allowed:
            break
        allowed |= descendants
    for pid, (_, folder, command) in processes.items():
        if pid in allowed:
            continue
        if command is not None and _renderer_targets_audit(command, root):
            raise RuntimeError(f"Relevant original renderer process {pid} is still targeting this run")
        try:
            descriptors = list((folder / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            unavailable(pid, "descriptor_inventory")
            continue
        for descriptor in descriptors:
            try:
                target = Path(os.readlink(descriptor))
                if not target.is_absolute() or not target.is_relative_to(root):
                    continue
                info = (folder / "fdinfo" / descriptor.name).read_text()
                flags = next(int(line.split()[1], 8) for line in info.splitlines() if line.startswith("flags:"))
                if flags & os.O_ACCMODE:
                    raise RuntimeError(f"External process {pid} has an audit file open for writing: {target}")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                unavailable(pid, "descriptor_target_or_flags")
                continue
    return [dict(pid=pid, unavailable_inspection=sorted(fields),
                 command_line_inspected=processes.get(pid, (None, None, None))[2] is not None,
                 command_classification="unrelated_or_not_targeting_this_run" if processes.get(pid, (None, None, None))[2] is not None else "command_unavailable")
            for pid, fields in sorted(blind.items()) if pid not in allowed]


def _roi_paths(root: Path, identifier: str) -> tuple[Path, list[Path]]:
    base = root / "2_Model_Annotations"
    return (base / "videos/closeups" / f"{identifier}.mp4", [
        base / "exact_pixel_traces" / f"{identifier}.csv",
        base / "figures/traces" / f"{identifier}.png",
        base / "metadata" / f"{identifier}.json"])


def _verify_checkpoints(render: Mapping, figures: Mapping) -> None:
    for key, record in render.items():
        if record["path"] != key or not record.get("all_frames_rgb_byte_exact") or not record.get("full_decode_pass"):
            raise ValueError("Video checkpoint is not a validated lossless source record")
        if not Path(key).is_file() or _digest(Path(key)) != record["sha256"]:
            raise ValueError(f"Video checkpoint bytes disagree: {key}")
        thumbnail = Path(record["thumbnail"])
        if not thumbnail.is_file() or _digest(thumbnail) != record["thumbnail_sha256"]:
            raise ValueError(f"Thumbnail checkpoint bytes disagree: {thumbnail}")
    for key in figures:
        if not _artifact_cached(Path(key), figures):
            raise ValueError(f"Figure/table checkpoint bytes disagree: {key}")


def partition_pending(root: Path, plan: Mapping, render: Mapping, figures: Mapping, workers: int) -> list[list[str]]:
    if not 1 <= workers <= 3:
        raise ValueError("Use one to three disjoint workers")
    identifiers = [row["model_roi_id"] for row in plan["model_sites"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate model ROI identifiers")
    pending = []
    for identifier in identifiers:
        video, other = _roi_paths(root, identifier)
        if str(video) not in render or any(str(path) not in figures for path in other):
            pending.append(identifier)
    # Stable round-robin assignment in the original inventory order; no labels.
    return [pending[index::workers] for index in range(workers)]


def _source_context(root: Path) -> tuple[dict, dict, list, list, list]:
    contract, plan, sources = (_read(root / name) for name in ("run_contract.json", "inventory_plan.json", "source_manifest.json"))
    expected = {"renderer_sha256": Path(original.__file__),
        "media_helper_sha256": Path(original._base.__file__),
        "evaluation_helper_sha256": Path(original.__file__).with_name("two_stencil_evaluation.py"),
        "inventory_validator_sha256": Path(original.__file__).resolve().parents[2] / "reports/scientific_audit.py"}
    code = []
    for key, path in expected.items():
        if _digest(path) != contract[key]:
            raise ValueError(f"Frozen renderer dependency changed: {key}")
        code.append(_record(path))
    by_stage = {row["stage"]: row for row in sources["sources"]}
    # The actual campaign provides both files. Their typed/string row order is
    # preserved exactly, rather than reconstructed from grouped identities.
    for name in ("candidates", "expert_occurrences"):
        _verify(by_stage[name])
    candidates, experts = (_rows(by_stage[name]["path"]) for name in ("candidates", "expert_occurrences"))
    if _canonical_digest(candidates) != contract["candidate_rows_sha256"] or _canonical_digest(experts) != contract["experts_sha256"]:
        raise ValueError("Candidate/annotation rows disagree with the original audit contract")
    rebuilt = audit_inventory_plan(candidates, experts, contract["source_frames_ui"], context_frames=contract["context_frames"], fullfield_step=contract["fullfield_step"])
    if _canonical_digest(rebuilt) != _canonical_digest(plan):
        raise ValueError("Saved full inventory disagrees with unchanged candidates and annotations")
    stages = []
    for name in contract["stage_sequence"]:
        row = by_stage[name]; path = Path(contract["stage_paths"][name]); stat = path.stat()
        if str(path) != row["path"] or stat.st_size != row["size_bytes"] or row["sha256"] != contract["source_binding"]["stage_sha256"][name]:
            raise ValueError("Stage source binding disagrees with the original audit")
        stages.append(dict(**row, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, bytes_rehashed=False))
    return contract, plan, candidates, stages, code


def prepare_workers(audit_root: str | Path, *, stopped_writer_pid: int, workers: int = 3) -> dict:
    root = Path(audit_root).resolve(); workspace = _workspace(root)
    _assert_stopped(stopped_writer_pid)
    if _read(root / "status.json").get("status") == "complete":
        raise ValueError("A completed original audit cannot be repartitioned")
    workspace.mkdir(exist_ok=True)
    with _lock(workspace / "phase.lock"):
        process_blind_spots = _external_writers(root, workspace)
        if (workspace / "session.json").exists():
            raise FileExistsError("A worker session already exists; resume its workers or merge it")
        contract, plan, candidates, stages, code = _source_context(root)
        checkpoints = [_read(root / name) if (root / name).exists() else {} for name in CHECKPOINTS]
        _verify_checkpoints(*checkpoints)
        partitions = partition_pending(root, plan, *checkpoints, workers)
        snapshots = []
        for name, payload in zip(CHECKPOINTS, checkpoints):
            path = root / name
            snapshot = workspace / f"original_{name}"
            # Preserve exact original bytes, including the last serialized
            # record's geometry-field presence; no synthetic cache entries.
            snapshot.write_bytes(path.read_bytes() if path.exists() else b"{}\n")
            snapshots.append(dict(name=name, original_exists=path.exists(), snapshot=_record(snapshot),
                                  original=_record(path) if path.exists() else None))
        records = [_record(root / name) for name in ("run_contract.json", "inventory_plan.json", "source_manifest.json")]
        source_rows = _read(root / "source_manifest.json")["sources"]
        records += [_record(Path(row["path"])) for row in source_rows if row["stage"] in ("candidates", "expert_occurrences")]
        session = dict(schema_version=1, audit_root=str(root), stopped_writer_pid=stopped_writer_pid,
            helper=_record(Path(__file__)), bindings=records+code, stage_bindings=stages,
            original_checkpoints=snapshots, worker_count=workers, partitions=partitions,
            original_model_roi_count=plan["model_roi_count"], pending_model_roi_count=sum(map(len, partitions)),
            cpu_threads_per_worker=1, detector_changed=False, scientific_audit_complete=False,
            process_inspection_blind_spots=process_blind_spots,
            process_inspection_scope="Read-only process scan; inaccessible descriptors remain uninspected. Metadata contains PIDs and command classifications only, never full command lines. Original stopped PID, visible relevant renderer commands, writable audit descriptors, locks and unchanged checkpoints are separate guards.",
            source_array_bytes_rehashed=False, source_array_validation="Inherited original source hashes; stat guard until original full audit rehashes and validates on resume",
            exclusive_execution="Original writer must stay stopped until merge; helper locks do not modify the frozen renderer")
        _json(workspace / "session.json", session)
        for index in range(workers):
            folder = workspace / f"worker_{index}"
            folder.mkdir(exist_ok=True)
            for snapshot in snapshots:
                (folder / snapshot["name"]).write_bytes(Path(snapshot["snapshot"]["path"]).read_bytes())
        return session


def _validate_session(root: Path, session: Mapping, *, merged: bool = False) -> None:
    _assert_stopped(int(session["stopped_writer_pid"]))
    _verify(session["helper"])
    for record in session["bindings"]:
        _verify(record)
    for row in session["stage_bindings"]:
        stat = Path(row["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) != (row["size_bytes"], row["mtime_ns"], row["ctime_ns"], row["inode"]):
            raise ValueError("Sealed source array changed during worker rendering")
    for record in session["original_checkpoints"]:
        _verify(record["snapshot"])
        path = root / record["name"]
        if not merged:
            if record["original_exists"]:
                _verify(record["original"])
            elif path.exists():
                raise ValueError("Canonical checkpoint appeared during worker rendering")


def _render_model_locations(root: Path, contract: Mapping, plan: Mapping, candidates: list, identifiers: list[str], folder: Path) -> None:
    """The frozen model branch, with only progress/checkpoint destinations split."""
    np, _, _, _ = _load_deps()
    arrays = {name: np.load(contract["stage_paths"][name], mmap_mode="r", allow_pickle=False) for name in contract["stage_sequence"]}
    frames = list(map(int, contract["source_frames_ui"])); frame_index = {frame: i for i, frame in enumerate(frames)}
    shape = arrays["Raw"].shape
    if len(shape) != 3 or shape[0] != len(frames) or any(a.shape != shape for a in arrays.values()):
        raise ValueError("Source array grid differs from the original full source interval")
    height, width = shape[1:]; display_limits = contract["display_limits"]; fps = contract["source_fps"]
    threshold = _frozen_threshold(candidates, contract["operating_point"])
    _verify_candidate_score_binding(candidates, arrays["Score"], frame_index, threshold, contract["operating_point"])
    checkpoint_path, figure_checkpoint_path = (folder / name for name in CHECKPOINTS)
    checkpoint, figure_checkpoint = (_read(path) for path in (checkpoint_path, figure_checkpoint_path))
    progress = _Progress(folder)
    candidates_by_frame = {}
    for row in candidates:
        candidates_by_frame.setdefault(int(row["source_frame_ui"]), []).append(row)
    def stages(frame):
        return {name: a[frame_index[frame]] for name, a in arrays.items()}
    sites = {row["model_roi_id"]: row for row in plan["model_sites"]}
    for completed, identifier in enumerate(identifiers, 1):
        location = sites[identifier]
        shown = plan["model_closeup_source_frames_ui"][identifier]
        relevant = [(f, f) for f in location["member_source_frames_ui"]]
        x, y = float(location["x_px"]), float(location["y_px"])
        bounds = identity_crop(location, width, height)
        member_ids = set(location.get("member_proposal_ids", []))
        def close_markers(frame, member_ids=member_ids):
            return [r for r in candidates_by_frame.get(frame, []) if r["proposal_id"] in member_ids]
        marker_index = next(i for i, frame in enumerate(shown) if close_markers(frame))
        video_path, (trace_path, figure_path, metadata_path) = _roi_paths(root, identifier)
        previous = copy.deepcopy(checkpoint.get(str(video_path)))
        record = _write_video(video_path, shown,
            lambda frame, mf=close_markers, crop=bounds: _panel(stages(frame), display_limits, source_ui=frame, source_fps=fps, markers=mf(frame), color=ORANGE, crop=crop, panel_width=144, panel_height=112),
            fps=fps, marker_sample_index=marker_index, annotation_section="model",
            checkpoint=checkpoint, checkpoint_path=checkpoint_path, progress=progress)
        closeup_geometry = panel_image_geometry(bounds[2]-bounds[0], bounds[3]-bounds[1], 144, 112)
        record["panel_image_geometry"] = closeup_geometry
        record["source_crop_xyxy_half_open"] = list(bounds)
        # Keep all original snapshot records exactly equal across workers. The
        # original writer can add geometry to its in-memory video manifest.
        if previous is not None:
            checkpoint[str(video_path)] = previous
        px, py = min(width-1, int(math.floor(x+.5))), min(height-1, int(math.floor(y+.5)))
        traces = {name: np.asarray(a[:, py, px]).copy() for name, a in arrays.items()}
        if any(not np.isfinite(values).all() for values in traces.values()):
            raise ValueError("Nonfinite exact-pixel trace")
        if not _artifact_cached(trace_path, figure_checkpoint):
            _table(trace_path, [{"source_frame_ui": frame, **{name: float(values[i]) for name, values in traces.items()}} for i, frame in enumerate(frames)])
            _checkpoint_artifact(trace_path, figure_checkpoint, figure_checkpoint_path)
        if not _artifact_cached(figure_path, figure_checkpoint):
            _trace_plot(figure_path, traces, frames, title=f"{identifier} | exact pixel ({px},{py}); all {len(frames)} source samples", spans=relevant, color="#ff9123", threshold=threshold)
            _checkpoint_artifact(figure_path, figure_checkpoint, figure_checkpoint_path)
        if not _artifact_cached(metadata_path, figure_checkpoint):
            _json(metadata_path, {**location, "exact_trace_pixel_xy": [px, py], "rounding": "floor(coordinate+0.5), clipped at last image pixel", "closeup_source_frames_ui": shown, "crop_xyxy_half_open": bounds, "panel_image_geometry": closeup_geometry, "all_source_samples_in_trace": True, "source_time_semantics": SOURCE_TIME_SEMANTICS})
            _checkpoint_artifact(metadata_path, figure_checkpoint, figure_checkpoint_path)
        _json(checkpoint_path, checkpoint)
        progress.emit("model_roi", force=True, roi=identifier, completed_rois=completed, expected_rois=len(identifiers))


def run_worker(audit_root: str | Path, worker_index: int) -> dict:
    root = Path(audit_root).resolve(); workspace = _workspace(root); session = _read(workspace / "session.json")
    if not 0 <= worker_index < session["worker_count"]:
        raise ValueError("Invalid worker index")
    folder = workspace / f"worker_{worker_index}"
    with _lock(workspace / "phase.lock", shared=True), _lock(folder / "worker.lock"):
        if (workspace / "merge.json").exists():
            raise ValueError("Checkpoint merge already started; do not restart a worker")
        _validate_session(root, session); process_blind_spots = _external_writers(root, workspace)
        contract, plan, candidates, _, _ = _source_context(root)
        status = dict(status="RUNNING", pid=os.getpid(), worker_index=worker_index, identifiers=session["partitions"][worker_index], session_sha256=_digest(workspace / "session.json"), process_inspection_blind_spots=process_blind_spots)
        _json(folder / "status.json", status)
        variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        prior = {key: os.environ.get(key) for key in variables}
        try:
            for key in variables:
                os.environ[key] = "1"
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                _render_model_locations(root, contract, plan, candidates, status["identifiers"], folder)
            _validate_session(root, session)
            result = dict(status, status="COMPLETE", checkpoints=[_record(folder / name) for name in CHECKPOINTS], scientific_audit_complete=False)
            _json(folder / "status.json", result)
            return result
        except BaseException as error:
            _json(folder / "status.json", dict(status, status="FAILED", error_type=type(error).__name__, error=str(error)))
            raise
        finally:
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def merge_workers(audit_root: str | Path) -> dict:
    root = Path(audit_root).resolve(); workspace = _workspace(root); session = _read(workspace / "session.json")
    with _lock(workspace / "phase.lock"):
        process_blind_spots = _external_writers(root, workspace)
        merge_path = workspace / "merge.json"
        previous_merge = _read(merge_path) if merge_path.exists() else None
        _validate_session(root, session, merged=previous_merge is not None)
        merged = [_read(Path(row["snapshot"]["path"])) for row in session["original_checkpoints"]]
        initial = copy.deepcopy(merged)
        statuses = []
        for index, identifiers in enumerate(session["partitions"]):
            folder = workspace / f"worker_{index}"; status = _read(folder / "status.json")
            if status.get("status") != "COMPLETE" or status["identifiers"] != identifiers or status["session_sha256"] != _digest(workspace / "session.json"):
                raise ValueError("Every disjoint worker must finish its complete declared ROI partition")
            if _alive(int(status["pid"])) and int(status["pid"]) != os.getpid():
                raise RuntimeError("Wait for the worker process to exit before merging")
            for record in status["checkpoints"]:
                _verify(record)
            statuses.append(_record(folder / "status.json"))
            allowed = [set(), set()]
            for identifier in identifiers:
                video, other = _roi_paths(root, identifier)
                allowed[0].add(str(video)); allowed[1].update(map(str, other))
            for kind, name in enumerate(CHECKPOINTS):
                payload = _read(folder / name)
                if not set(initial[kind]) <= set(payload):
                    raise ValueError("Worker dropped original checkpoint records")
                for key, record in payload.items():
                    if key in merged[kind] and merged[kind][key] != record:
                        raise ValueError(f"Overlapping checkpoint records disagree: {key}")
                    if key not in initial[kind] and key not in allowed[kind]:
                        raise ValueError("Worker wrote outside its disjoint model partition")
                    merged[kind][key] = record
                if not allowed[kind] <= set(payload):
                    raise ValueError("Worker checkpoint omits required model artifacts")
        _verify_checkpoints(*merged)
        if previous_merge:
            if previous_merge["merged_checkpoint_digests"] != [_canonical_digest(value) for value in merged]:
                raise ValueError("Prepared merge no longer agrees with worker checkpoint contents")
            for index, name in enumerate(CHECKPOINTS):
                current = _read(root / name) if (root / name).exists() else {}
                if current not in (initial[index], merged[index]):
                    raise ValueError("Canonical checkpoint changed outside the interrupted merge")
        result = dict(status="MERGING", session_sha256=_digest(workspace / "session.json"),
            process_inspection_blind_spots=process_blind_spots,
            worker_status_bindings=statuses, merged_checkpoint_digests=[_canonical_digest(value) for value in merged],
            model_roi_count=session["original_model_roi_count"], completed_pending_model_rois=session["pending_model_roi_count"],
            scientific_audit_complete=False, next_step="Resume original full audit for comparison, inventory, byte and geometry validation")
        _json(merge_path, result)
        for name, payload in zip(CHECKPOINTS, merged):
            _json(root / name, payload)
        result["status"] = "MERGED"
        result["canonical_checkpoints"] = [_record(root / name) for name in CHECKPOINTS]
        _json(merge_path, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "merge"))
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--stopped-writer-pid", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.stopped_writer_pid is None:
            parser.error("prepare requires --stopped-writer-pid")
        result = prepare_workers(args.audit_root, stopped_writer_pid=args.stopped_writer_pid, workers=args.workers)
    elif args.command == "worker":
        if args.worker_index is None:
            parser.error("worker requires --worker-index")
        result = run_worker(args.audit_root, args.worker_index)
    else:
        result = merge_workers(args.audit_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
