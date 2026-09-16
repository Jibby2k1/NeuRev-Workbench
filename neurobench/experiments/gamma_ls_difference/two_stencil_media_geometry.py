"""Post-hoc metadata validation of all completed encoding_v2 audit videos.

This standard-library-only validator does not change or reencode media. Actual
MP4 byte validation is inherited from the bound completed-audit evidence;
video-record hashes must agree with the bound artifact indexes. Probe geometry,
frame maps, rational FPS, and duration are checked independently here.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_STATE_COUNT = 25
EXPECTED_VIDEO_COUNT = 8176
AUDIT_REVISION = "encoding_v2"
STAGES = ["Raw", "X", "A", "M", "sigma", "contrast", "Z"]
DURATION_TOLERANCE = Fraction(1, 1_000_000)
FILE_VALIDATION_INHERITANCE = (
    "Actual MP4 bytes were validated by the bound completed-audit process. "
    "This validator does not rehash media: it checks each video-record SHA-256 "
    "against the sealed artifact-index SHA-256, plus current existence/size. "
    "The inherited byte-validation evidence is point-in-time, not a new media read."
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _binding(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        number = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be an integer") from error
    _require(number.denominator == 1 and number >= minimum, f"{label} must be an integer >= {minimum}")
    return int(number)


def _positive_fraction(value: Any, label: str) -> Fraction:
    try:
        number = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be a finite positive fraction") from error
    _require(number > 0, f"{label} must be positive")
    return number


def _frame_digest(frames: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(frames), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_video_geometry(video: Mapping[str, Any], *, contract: Mapping[str, Any],
                            expected_frames_ui: Sequence[int], kind: str,
                            annotation_section: str) -> dict[str, Any]:
    """Validate one sealed probe; duration is displayed samples/FPS, not span."""
    _require(kind in {"fullfield", "closeup"}, "Unknown video kind")
    _require(list(contract["stage_sequence"]) == STAGES, "Audit must declare all seven stages in order")
    expected = [_integer(frame, "expected source frame") for frame in expected_frames_ui]
    observed = [_integer(frame, "video source frame") for frame in video["source_frames_ui"]]
    _require(bool(expected) and all(b > a for a, b in zip(expected, expected[1:])), "Expected frame map must be nonempty and strictly increasing")
    _require(observed == expected, "Video source-frame map differs from its complete inventory")
    _require(video["source_frames_sha256"] == _frame_digest(expected), "Video frame-map digest mismatch")
    _require(_integer(video["frame_count"], "video frame count") == len(expected), "Video frame count mismatch")
    _require(video["annotation_section"] == annotation_section, "Video annotation section mismatch")
    probe = video["probe"]
    _require(_integer(probe["nb_frames"], "probe frame count") == len(expected), "Probe frame count mismatch")
    source_fps = _positive_fraction(contract["source_fps"], "source FPS")
    expected_fps = source_fps
    if kind == "fullfield":
        step = _integer(contract["fullfield_step"], "full-field step")
        expected_fps /= step
        _require(_positive_fraction(contract["fullfield_fps"], "contract full-field FPS") == expected_fps,
                 "Contract full-field FPS differs from source FPS/step")
    _require(_positive_fraction(video["fps"], "video FPS") == expected_fps, "Video FPS differs from declared FPS")
    _require(_positive_fraction(probe["r_frame_rate"], "probe FPS") == expected_fps, "Probe FPS differs from declared FPS")
    panel_key = "fullfield_panel_image_area_size_wh" if kind == "fullfield" else "closeup_panel_image_area_size_wh"
    panel_width, panel_height = (_integer(value, "panel dimension") for value in contract[panel_key])
    _require([panel_width, panel_height] == ([180, 136] if kind == "fullfield" else [144, 112]),
             "Panel dimensions differ from the frozen seven-stage audit layout")
    columns, header = 4, 30
    expected_width = columns * panel_width
    expected_height = ((len(STAGES) + columns - 1) // columns) * (panel_height + header)
    _require(_integer(probe["width"], "probe width") == expected_width and
             _integer(probe["height"], "probe height") == expected_height,
             "Probe canvas dimensions differ from the declared panel geometry")
    _require(video["panel_image_geometry"]["panel_image_area_size_wh"] == [panel_width, panel_height],
             "Video panel geometry differs from its contract")
    duration = _positive_fraction(probe["duration"], "probe duration")
    expected_duration = Fraction(len(expected), 1) / expected_fps
    _require(abs(duration - expected_duration) <= DURATION_TOLERANCE,
             "Probe duration differs from displayed frame count/FPS")
    _require(video.get("encoding_revision") == AUDIT_REVISION and video.get("all_frames_rgb_byte_exact") is True
             and video.get("all_frames_marker_expectations_pass") is True,
             "Video lacks completed encoding_v2 exact-frame validation")
    _require(_integer(video["decoded_frame_count"], "decoded frame count") == len(expected),
             "Exact decoded-frame inventory mismatch")
    return {"frame_count": len(expected), "canvas_width": expected_width,
            "canvas_height": expected_height, "fps_fraction": str(expected_fps),
            "expected_duration_fraction": str(expected_duration),
            "duration_error_seconds": float(abs(duration - expected_duration))}


def validate_state_geometry(videos: Sequence[Mapping[str, Any]], *, output_root: Path,
                            contract: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Require both full fields and every expert/model closeup, exactly once."""
    output_root = output_root.resolve()
    source = [_integer(frame, "contract source frame") for frame in contract["source_frames_ui"]]
    _require(bool(source) and all(b == a + 1 for a, b in zip(source, source[1:])),
             "Contract full-duration source grid must be consecutive")
    step = _integer(contract["fullfield_step"], "full-field step")
    full = source[::step]
    if full[-1] != source[-1]:
        full.append(source[-1])
    _require(plan["fullfield_source_frames_ui"] == full, "Full-field map omits a declared sample or inclusive final source frame")
    expected: dict[str, tuple[str, str, Sequence[int]]] = {}
    for section, directory in (("expert", "1_Expert_Annotations"), ("model", "2_Model_Annotations")):
        expected[f"{directory}/videos/{section}_sequential_full_field.mp4"] = ("fullfield", section, full)
        closeups = plan[f"{section}_closeup_source_frames_ui"]
        _require(len(closeups) == _integer(plan[f"{section}_roi_count"], "ROI count", minimum=0), "Closeup inventory disagrees with ROI count")
        for identifier, frames in closeups.items():
            _require(bool(frames) and set(frames) <= set(source), "Closeup source frames lie outside the audited source grid")
            expected[f"{directory}/videos/closeups/{identifier}.mp4"] = ("closeup", section, frames)
    _require(len(expected) == _integer(plan["expected_video_count"], "expected video count"), "Expected video inventory cardinality mismatch")
    observed: dict[str, Mapping[str, Any]] = {}
    for video in videos:
        path = Path(video["path"])
        path = path.resolve() if path.is_absolute() else (output_root / path).resolve()
        _require(path.is_relative_to(output_root), "Video path lies outside its audit state")
        relative = str(path.relative_to(output_root))
        _require(relative not in observed, "Duplicate video in media manifest")
        observed[relative] = video
    _require(set(observed) == set(expected), "Video manifest omits or adds an expected full field/ROI closeup")
    checked_frames = 0
    maximum_duration_error = 0.0
    for relative, (kind, section, frames) in expected.items():
        result = validate_video_geometry(observed[relative], contract=contract,
                                         expected_frames_ui=frames, kind=kind,
                                         annotation_section=section)
        checked_frames += result["frame_count"]
        maximum_duration_error = max(maximum_duration_error, result["duration_error_seconds"])
    return {"validated_video_count": len(expected), "validated_displayed_frame_count": checked_frames,
            "maximum_duration_error_seconds": maximum_duration_error}


def _validate_campaign(root: Path) -> dict[str, Any]:
    paths = {"protocol": root / "protocol.json", "results": root / "results.tsv",
             "audit_completion": root / "audit_completion.json",
             "audit_inventory_plan": root / f"audit_inventory_plan_{AUDIT_REVISION}.json"}
    bindings = {name: _binding(path) for name, path in paths.items()}
    protocol, completion, inventory = (_load(paths[key]) for key in ("protocol", "audit_completion", "audit_inventory_plan"))
    _require(completion.get("audit_revision") == AUDIT_REVISION and completion.get("q1_scientific_audit_complete") is True,
             "All encoding_v2 scientific audits must complete before final media validation")
    _require(completion.get("completed_states") == EXPECTED_STATE_COUNT and completion.get("expected_states") == EXPECTED_STATE_COUNT,
             "Campaign completion must contain all 25 states")
    _require(inventory.get("audit_revision") == AUDIT_REVISION and inventory.get("total_videos") == EXPECTED_VIDEO_COUNT,
             "Revision inventory must declare the expected 8,176 videos")
    protocol_ids = [row["cell_id"] for row in protocol["cells"]]
    completed_ids = [row["cell_id"] for row in completion["states"]]
    planned_ids = [row["cell_id"] for row in inventory["states"]]
    _require(len(protocol_ids) == len(set(protocol_ids)) == EXPECTED_STATE_COUNT
             and len(completed_ids) == len(set(completed_ids)) == EXPECTED_STATE_COUNT
             and len(planned_ids) == len(set(planned_ids)) == EXPECTED_STATE_COUNT
             and set(protocol_ids) == set(completed_ids) == set(planned_ids),
             "Protocol, completion and revision inventories must identify the same 25 unique states")
    planned = {row["cell_id"]: row for row in inventory["states"]}
    states = []
    for complete in completion["states"]:
        cell = complete["cell_id"]
        output = Path(complete["output_root"]).resolve()
        _require(output == root / "scientific_audits" / AUDIT_REVISION / cell,
                 "Completed output does not belong to the encoding_v2 state")
        _require(complete.get("completion_verification") in {"successful_renderer_return_this_invocation", "existing_complete_contract_artifacts_and_sources_reverified"},
                 "Completed state lacks inherited byte-validation evidence")
        names = {"video_manifest": "video_manifest.json", "artifact_index": "artifact_index.json",
                 "run_contract": "run_contract.json", "inventory_plan": "inventory_plan.json",
                 "source_manifest": "source_manifest.json", "summary": "summary.json", "status": "status.json"}
        state_bindings = {name: _binding(output / filename) for name, filename in names.items()}
        for key in ("artifact_index", "run_contract"):
            _require(state_bindings[key]["sha256"] == complete[f"{key}_sha256"],
                     f"{cell}: {key} differs from completed-audit binding")
        artifact_rows = _load(output / "artifact_index.json")["artifacts"]
        artifact_index = {row["path"]: row for row in artifact_rows}
        _require(len(artifact_index) == len(artifact_rows), f"{cell}: duplicate artifact-index path")
        for key, filename in names.items():
            if key != "artifact_index":
                _require(filename in artifact_index and artifact_index[filename]["sha256"] == state_bindings[key]["sha256"],
                         f"{cell}: sealed metadata hash mismatch for {filename}")
        status, summary = _load(output / "status.json"), _load(output / "summary.json")
        _require(status.get("status") == "complete" and status.get("scientific_audit_complete") is True
                 and summary.get("scientific_audit_complete") is True, f"{cell}: audit is not complete")
        contract = _load(output / "run_contract.json")
        _require(contract["source_binding"].get("audit_revision") == AUDIT_REVISION
                 and contract["source_binding"].get("cell_id") == cell, f"{cell}: contract identity/revision mismatch")
        plan = _load(output / "inventory_plan.json")
        videos = _load(output / "video_manifest.json")["videos"]
        result = validate_state_geometry(videos, output_root=output, contract=contract, plan=plan)
        _require(result["validated_video_count"] == planned[cell]["expected_video_count"] == complete["video_count"] == summary["video_count"],
                 f"{cell}: video inventory disagrees with campaign completion/plan")
        for video in videos:
            path = Path(video["path"])
            path = path.resolve() if path.is_absolute() else (output / path).resolve()
            relative = str(path.relative_to(output))
            _require(relative in artifact_index and video["sha256"] == artifact_index[relative]["sha256"],
                     f"{cell}: MP4 manifest/artifact-index digest mismatch")
            _require(path.is_file() and path.stat().st_size == artifact_index[relative]["size_bytes"],
                     f"{cell}: MP4 missing or changed size since inherited validation")
        states.append({"cell_id": cell, "output_root": str(output), **result, "bindings": state_bindings})
    total = sum(row["validated_video_count"] for row in states)
    _require(total == EXPECTED_VIDEO_COUNT, "Validated video total differs from the complete 8,176-video inventory")
    return {"schema_version": 1, "status": "PASS", "media_geometry_complete": True,
            "audit_revision": AUDIT_REVISION, "completed_state_count": len(states),
            "validated_video_count": total, "expected_state_count": EXPECTED_STATE_COUNT,
            "expected_video_count": EXPECTED_VIDEO_COUNT, "bindings": bindings, "states": states,
            "validator": _binding(Path(__file__)), "duration_tolerance_seconds": float(DURATION_TOLERANCE),
            "duration_semantics": "displayed sample count/FPS including the final sample duration; not elapsed source-frame span",
            "layout": {"stage_count": 7, "columns": 4, "header_height": 30,
                       "fullfield_canvas_wh": [720, 332], "closeup_canvas_wh": [576, 284]},
            "actual_media_bytes_rehashed": False,
            "inherited_file_validation": FILE_VALIDATION_INHERITANCE,
            "source_arrays_read": False, "media_reencoded": False}


def validate_audit_media_geometry(root: str | Path) -> dict[str, Any]:
    """Write the mandatory final validation artifact, replacing stale PASS on failure."""
    root = Path(root).resolve()
    output = root / "audit_media_geometry_validation.json"
    try:
        result = _validate_campaign(root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"schema_version": 1, "status": "FAIL", "media_geometry_complete": False,
                  "audit_revision": AUDIT_REVISION, "expected_state_count": EXPECTED_STATE_COUNT,
                  "expected_video_count": EXPECTED_VIDEO_COUNT, "failure": str(error),
                  "validator": _binding(Path(__file__))}
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        raise ValueError(f"Final media geometry validation failed: {error}") from error
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed campaign root")
    args = parser.parse_args()
    result = validate_audit_media_geometry(args.output)
    print(json.dumps({"status": result["status"], "media_geometry_complete": result["media_geometry_complete"],
                      "completed_state_count": result["completed_state_count"],
                      "validated_video_count": result["validated_video_count"],
                      "artifact": str(args.output.resolve() / "audit_media_geometry_validation.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
