"""Hash-verified reuse of complete media for unchanged saved q1 proposals.

This module performs no scoring, association, image loading, or rendering.
Every proposal maps to its ORIGINAL audit review location. An association
anchor at another member pixel does not acquire an exact-pixel trace here.
All old expert/model/comparison artifacts remain authoritative and unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import os
import tempfile
from typing import Any

from .followup_validate import FileVerifier, METADATA, require, verify_audit_files

SCHEMA = "saved_proposal_persistence_audit_reuse_v1"
MODEL = "2_Model_Annotations"
OP_FIELDS = ("threshold", "threshold_z", "threshold_id", "threshold_frozen_from_calibration_only",
             "setup_source_frames_ui", "scale_floor", "target_proposals_per_frame", "eligible_area_px",
             "application_source_start_ui", "application_source_stop_ui", "application_frame_count",
             "setup_proposal_budget", "setup_proposal_count", "setup_budget_per_reference_area_frame",
             "reference_area_px")
VIDEO_FLAGS = ("full_decode_pass", "all_frames_source_palette_pure", "all_frames_decoded_palette_pure",
               "all_frames_marker_expectations_pass", "all_frames_rgb_byte_exact",
               "encoded_marker_separation_pass")
COLUMNS = ("cell_id", "cohort", "case_id", "arm_id", "proposal_id", "source_frame_ui", "source_time_s",
           "application_frame_index_zero_based", "x_px", "y_px", "original_source_x_px", "original_source_y_px",
           "score", "threshold_z", "original_model_roi_id", "original_representative_proposal_id",
           "original_trace_x_px", "original_trace_y_px", "distance_to_original_trace_pixel_px",
           "candidate_anchor_equals_original_trace_pixel", "review_resource_id", "trace_csv", "trace_png",
           "closeup_video", "closeup_thumbnail", "closeup_frame_index_zero_based", "closeup_playback_time_s",
           "model_fullfield_video", "fullfield_frame_present", "fullfield_frame_index_zero_based",
           "fullfield_playback_time_s", "biological_interpretation")
SEMANTICS = {
    "scope": "Exact saved q1 rows only; no prefixes, rescoring, changed thresholds or additional detections.",
    "association": "Radius/gap groups are descriptive algorithmic associations, not biological events or identities.",
    "original_review_location": "Original frozen spatial review grouping; its trace is at its original representative pixel.",
    "anchor_trace": "Join each new anchor/member by (cell_id, proposal_id). A trace at a different original pixel is context only; no new anchor trace is claimed.",
    "media": "Original overlays show original proposals, not persistence-filtered or grouped output. All original media and comparisons are retained.",
    "time": "Source UI is one-based inclusive; source time=(UI-1)/source_fps. Playback index/fps is a separate seek coordinate.",
    "closeup_time": "Closeups preserve every member frame and fixed context; omitted gaps are compacted in playback. Use the bound source-frame map.",
    "fullfield_time": "Declared temporal decimation plus final source frame; absence in fullfield is explicit and never a missing proposal.",
    "truth": "Real proposals remain unknown; sparse known windows do not define false positives, precise onset, or biological persistence. Null source truth does not transfer to real data.",
    "validation": "Every source/artifact file is SHA-verified this pass, deduplicated by resolved file with stat/alias guards. Decode, palette and byte-exact RGB proofs are inherited from those verified records, not rerun.",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for v in value.values():
                yield from _records(v)
    elif isinstance(value, list):
        for v in value:
            yield from _records(v)


def _safe(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is not None
            and value not in (".", ".."), f"Unsafe {label}")
    return value


def _frames(value, label):
    require(isinstance(value, list) and bool(value) and all(type(v) is int and v > 0 for v in value)
            and value == sorted(set(value)), f"Invalid {label} source-frame map")
    return value


def _same(left, right):
    return Path(left["path"]).resolve() == Path(right["path"]).resolve() and left["sha256"] == right["sha256"]


class _Context:
    def __init__(self):
        self.v = FileVerifier()
        self.json_cache = {}
        self.completions = {}

    def read(self, path):
        binding = self.v.binding(path)
        key = binding["path"]
        if key not in self.json_cache:
            self.json_cache[key] = self.v.read_json(path)
        return self.json_cache[key]

    def completion(self, binding):
        b = self.v.verify(binding)
        if b["path"] not in self.completions:
            value = self.read(b["path"])
            numeric_complete = value.get("numerical_complete") is True or (
                value.get("schema") == "gamma_st_sensitivity_completion_v1"
                and "numerical_complete" not in value and value.get("numeric_complete") is True)
            require(value.get("status") == "PASS" and numeric_complete
                    and value.get("scientific_artifact_audit_complete") is True, "Input campaign is not complete")
            authority = {Path(r["path"]).resolve(): r for r in value["evidence_bindings"]}
            owner = Path(b["path"]).parent
            pp = (owner / "protocol.json").resolve()
            require(pp in authority, "Completion does not bind its protocol")
            self.v.verify(authority[pp])
            self.completions[b["path"]] = (b, authority, self.read(pp))
        return self.completions[b["path"]]

    def authoritative(self, record, completion, depth=0):
        require(depth <= 5, "Completion lineage is too deep or cyclic")
        _, authority, protocol = self.completion(completion)
        target = Path(record["path"]).resolve()
        if target in authority:
            require(_same(record, authority[target]), "Completion-bound input differs")
            return self.v.verify(authority[target])
        # Dataset metadata can belong to an explicitly bound completed ancestor.
        for key in ("prior_completion", "baseline_completion"):
            if key in protocol:
                try:
                    return self.authoritative(record, protocol[key], depth + 1)
                except _MissingAuthority:
                    pass
        raise _MissingAuthority(f"Input is not bound by the completed source lineage: {target}")


class _MissingAuthority(ValueError):
    pass


def _owner_record(cell, ctx):
    completion, authority, protocol = ctx.completion(cell["completion"])
    aggregate_path = (Path(completion["path"]).parent / "audit_complete.json").resolve()
    require(aggregate_path in authority, "Completion does not bind the audit aggregate")
    ctx.v.verify(authority[aggregate_path])
    aggregate = ctx.read(aggregate_path)
    rows = aggregate["audits"]
    keys = [(r["case_id"], r["arm_id"]) for r in rows]
    require(aggregate.get("status") == "PASS" and len(keys) == len(set(keys)) == aggregate["cells"],
            "Inherited audit aggregate inventory is incomplete or ambiguous")
    key = (cell["case_id"], cell["arm_id"])
    require(key in keys, "Selected cell is absent from its completed audit aggregate")
    record = rows[keys.index(key)]
    audit = Path(cell["audit_root"])
    require(audit.is_absolute(), "Audit root must be absolute")
    metadata = record["metadata_bindings"]
    if isinstance(metadata, dict):
        require(set(metadata) == set(METADATA), "Inherited named metadata inventory differs")
        for name, binding in metadata.items():
            require(Path(binding["path"]).name == name, "Inherited metadata name/path differs")
        metadata = list(metadata.values())
    require(isinstance(metadata, list), "Inherited metadata bindings must be a named mapping or list")
    bindings = {Path(b["path"]).resolve(): b for b in metadata}
    require(len(bindings) == len(METADATA), "Expected the exact eight inherited audit metadata bindings")
    for name in METADATA:
        path = (audit / name).resolve()
        require(path in bindings, f"Selected audit path is not owned by the completion: {name}")
        ctx.v.verify(bindings[path])
    require(ctx.read(audit / "summary.json") == record["summary"], "Inherited copied summary differs")
    return protocol, dict(completion=completion, audit_aggregate=ctx.v.binding(aggregate_path),
                          metadata_bindings=list(bindings.values()))


def _index(audit, ctx):
    index = ctx.read(audit / "artifact_index.json")
    result = {}
    for row in index["artifacts"]:
        path = (audit / row["path"]).resolve()
        require(path not in result, "Duplicate artifact-index path")
        result[path] = row
    def indexed(path):
        path = Path(path)
        require(path.resolve() in result, f"Required inherited artifact missing from sealed index: {path}")
        return ctx.v.verify(result[path.resolve()], root=audit)
    return result, indexed


def _video_records(audit, ctx, indexed, expected, source_frames):
    indexed(audit / "video_manifest.json")
    records = ctx.read(audit / "video_manifest.json")["videos"]
    require(len(records) == expected, "Video-manifest count differs from full required inventory")
    result = {}
    for row in records:
        path = Path(row["path"])
        require(path.is_absolute() and path.resolve() not in result, "Video path is relative or duplicated")
        actual = indexed(path)
        require(actual["sha256"] == row["sha256"], "Video manifest/index hash differs")
        thumb = indexed(Path(row["thumbnail"]))
        require(thumb["sha256"] == row["thumbnail_sha256"], "Video thumbnail hash differs")
        frames = _frames(row["source_frames_ui"], "video")
        require(set(frames) <= set(source_frames) and len(frames) == row["frame_count"] == row["decoded_frame_count"],
                "Video source/decode counts differ")
        require(all(row.get(k) is True for k in VIDEO_FLAGS), "Inherited encoded-media validation is incomplete")
        require(row["source_rgb_stream_sha256"] == row["decoded_rgb_stream_sha256"], "RGB stream proof differs")
        require(math.isfinite(row["fps"]) and row["fps"] > 0, "Invalid video playback rate")
        result[path.resolve()] = row
    return result


def _inventory(audit, ctx, indexed, candidate_count):
    summary = ctx.read(audit / "summary.json")
    status, valid, inv = (ctx.read(audit / name) for name in ("status.json", "validation.json", "inventory.json"))
    require(summary.get("scientific_audit_complete") is True and status.get("status") == "complete"
            and status.get("scientific_audit_complete") is True and valid.get("status") == "passed"
            and valid.get("scientific_audit_complete") is True and valid.get("failures") == []
            and inv.get("complete") is True, "Inherited audit is not complete")
    for name in ("inventory_plan.json", "coverage_manifest.json"):
        indexed(audit / name)
    plan, coverage = ctx.read(audit / "inventory_plan.json"), ctx.read(audit / "coverage_manifest.json")
    require(summary["model_proposal_count"] == candidate_count == plan["all_model_proposal_count"], "Proposal inventory differs")
    for key in ("expert_roi_count", "model_roi_count", "expert_occurrence_count"):
        require(summary[key] == plan[key], f"Required inventory differs: {key}")
        if key in inv:
            require(inv[key] == summary[key], f"Observed inventory differs: {key}")
    er, mr, eo = (summary[k] for k in ("expert_roi_count", "model_roi_count", "expert_occurrence_count"))
    require(summary["video_count"] == plan["expected_video_count"] == 2 + er + mr
            and summary["comparison_trace_count"] == eo
            and plan["expected_trace_figure_count"] == er + mr + eo, "Full media/trace cardinality differs")
    require(coverage.get("all_model_rois_rendered") is True
            and coverage.get("all_expert_occurrences_compared") is True
            and coverage.get("closeups_full_rate_all_relevant_source_frames") is True
            and coverage.get("model_sites_are_biological_identities") is False,
            "Inherited audit coverage is incomplete")
    require(len(plan["unique_experts"]) == er, "Expert identity inventory differs")
    for roi in plan["unique_experts"]:
        _safe(roi, "expert ROI ID")
        for suffix in (f"metadata/{roi}.json", f"exact_pixel_traces/{roi}.csv",
                       f"figures/traces/{roi}.png", f"videos/closeups/{roi}.mp4"):
            indexed(audit / "1_Expert_Annotations" / suffix)
    return summary, inv, plan, coverage


def _candidate_key(row):
    fields = ("source_frame_ui", "x_px", "y_px", "score", "threshold_z", "candidate_rank_within_frame")
    values = tuple(float(row[k]) for k in fields)
    require(all(math.isfinite(v) for v in values), "Nonfinite saved candidate field")
    require(all(v.is_integer() for v in values[:3]) and values[0] > 0 and min(values[1:3]) >= 0,
            "Saved candidate coordinates/frame must be nonnegative integer pixels and positive UI")
    return (str(row["proposal_id"]),) + values


def _cell(cell, ctx):
    v = ctx.v
    original_protocol, owner = _owner_record(cell, ctx)
    for name in ("candidates", "seal", "calibration", "metadata"):
        ctx.authoritative(cell[name], cell["completion"])
    seal, op, rows, metadata = (ctx.read(cell[name]["path"]) for name in ("seal", "calibration", "candidates", "metadata"))
    require(seal.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", "Input candidate seal is incomplete")
    require(_same(seal["audit_candidates"], cell["candidates"]) and _same(seal["calibration"], cell["calibration"]),
            "Selected q1 file/calibration differs from cell seal")
    require(op["threshold_id"] == "q1" and op["threshold_frozen_from_calibration_only"] is True,
            "Only unchanged setup-frozen q1 rows have inherited media coverage")
    require(isinstance(rows, list), "Candidate file is not a row list")
    ids = [str(row["proposal_id"]) for row in rows]
    require(len(ids) == len(set(ids)) and all(ids), "Duplicate or empty candidate ID")
    require(cell.get("input_proposal_count", len(rows)) == len(rows), "Frozen per-cell proposal count differs")
    candidates = dict(zip(ids, rows))
    audit = Path(cell["audit_root"])
    # Full large-file read is deliberately here, after small ownership checks.
    sources = verify_audit_files(audit, v)
    artifacts, indexed = _index(audit, ctx)
    summary, inv, plan, coverage = _inventory(audit, ctx, indexed, len(rows))
    contract = ctx.read(audit / "run_contract.json")
    require(contract["source_binding"] == sources["source_binding"], "Audit contract/source binding differs")
    require(contract["source_binding"]["candidates_sha256"] == cell["candidates"]["sha256"],
            "Inherited detector rows differ from requested saved q1 rows")
    for key in OP_FIELDS:
        require(contract["operating_point"][key] == op[key], f"Inherited q1 operating point differs: {key}")
    for record in seal["stages"].values():
        v.verify(record)
    for row in sources["sources"]:
        name = row.get("stage")
        if name in seal["stages"]:
            require(row["sha256"] == seal["stages"][name]["sha256"], f"Inherited scientific stage differs: {name}")
    for name, digest in contract["source_binding"]["stage_sha256"].items():
        require(seal["stages"][name]["sha256"] == digest, f"Displayed stage differs: {name}")
    for name, record in contract["source_binding"].get("numeric_stage_bindings", {}).items():
        require(seal["stages"][name]["sha256"] == record["sha256"], f"Linked numeric stage differs: {name}")
        v.verify(record)
    frames = _frames(contract["source_frames_ui"], "original audit")
    require(frames == metadata["source_frames_ui"] and len(frames) == plan["source_frame_count"]
            == coverage["full_duration_exact_pixel_trace_sample_count"] == summary["all_trace_sample_count"],
            "Exact trace source-sample inventory differs")
    app = _frames(metadata["application_source_frames_ui"], "application")
    require(app == list(range(op["application_source_start_ui"], op["application_source_stop_ui"] + 1))
            and len(app) == op["application_frame_count"] and set(app) <= set(frames), "Application exposure differs")
    fps, pixel = float(original_protocol["frame_rate_hz"]), float(original_protocol["pixel_size_um"])
    require(math.isfinite(fps) and fps > 0 and math.isfinite(pixel) and pixel > 0
            and fps == contract["source_fps"], "Physical time/pixel calibration differs")
    offset = metadata["original_source_offset_xy"]
    require(offset == contract["source_binding"]["original_source_offset_xy"], "Source coordinate offset differs")
    y0, x0, y1, x1 = metadata["evaluation_box_yxyx"]
    require(offset == [x0, y0] and x1 > x0 and y1 > y0, "Invalid crop geometry")
    w, h = x1 - x0, y1 - y0
    truth = contract["truth_semantics"]
    if cell["cohort"] == "real":
        require(metadata["truth_mode"] == truth["truth_mode"] == "sparse_real"
                and truth["real_precision_identified"] is False, "Real labels must retain sparse/unknown semantics")
    else:
        require(metadata["truth_mode"] == truth["truth_mode"] == "fully_synthetic"
                and inv.get("expert_applicable") is False
                and summary["expert_roi_count"] == summary["expert_occurrence_count"] == 0,
                "Null audit must explicitly retain empty Expert/Comparison inventory")
    videos = _video_records(audit, ctx, indexed, summary["video_count"], frames)
    mp4s = {p for p in artifacts if p.suffix == ".mp4"}
    require(mp4s == set(videos), "Full sealed MP4 inventory differs from video manifest")
    for section, prefix in (("1_Expert_Annotations", "expert"), (MODEL, "model")):
        full = (audit / section / "videos" / f"{prefix}_sequential_full_field.mp4").resolve()
        require(full in videos and videos[full]["annotation_section"] == prefix
                and videos[full]["source_frames_ui"] == plan["fullfield_source_frames_ui"]
                and videos[full]["fps"] == contract["fullfield_fps"], "Fullfield video/source map differs")
    full = (audit / MODEL / "videos/model_sequential_full_field.mp4").resolve()
    fullmap = {frame: index for index, frame in enumerate(videos[full]["source_frames_ui"])}
    for name in ("model_occurrences.csv", "review_sites.json"):
        indexed(audit / MODEL / name)
    with (audit / MODEL / "model_occurrences.csv").open(newline="") as stream:
        assignments = list(csv.DictReader(stream))
    a_ids = [str(row["proposal_id"]) for row in assignments]
    require(len(a_ids) == len(set(a_ids)) and set(a_ids) == set(ids), "Original model membership does not cover every proposal exactly once")
    assigned = dict(zip(a_ids, assignments))
    sites = ctx.read(audit / MODEL / "review_sites.json")
    site_ids = [r["model_roi_id"] for r in sites]
    require(len(site_ids) == len(set(site_ids)) == summary["model_roi_count"], "Original review-site inventory differs")
    require(sites == plan["model_sites"], "Review sites differ from original inventory plan")
    resources, ledger, seen = [], [], set()
    for site in sites:
        roi = _safe(site["model_roi_id"], "model review ROI ID")
        resource_id = f"{cell['cell_id']}::{roi}"
        rel = dict(metadata=f"{MODEL}/metadata/{roi}.json", trace_csv=f"{MODEL}/exact_pixel_traces/{roi}.csv",
                   trace_png=f"{MODEL}/figures/traces/{roi}.png", closeup_video=f"{MODEL}/videos/closeups/{roi}.mp4",
                   closeup_thumbnail=f"{MODEL}/videos/closeups/{roi}.png")
        bindings = {key: indexed(audit / value) for key, value in rel.items()}
        details = ctx.read(audit / rel["metadata"])
        members = site["member_proposal_ids"]
        require(members and len(members) == len(set(members)) == site["member_proposal_count"]
                and not (seen & set(members)) and set(members) <= set(ids), "Duplicate/missing site members")
        for key in ("model_roi_id", "representative_proposal_id", "member_proposal_ids", "member_source_frames_ui", "x_px", "y_px"):
            require(details[key] == site[key], f"Site metadata differs: {key}")
        require(site["representative_proposal_id"] in members, "Representative is not an actual member proposal")
        rep = candidates[site["representative_proposal_id"]]
        require((float(site["x_px"]), float(site["y_px"])) == (float(rep["x_px"]), float(rep["y_px"])),
                "Original representative coordinate differs from actual saved row")
        pixel_xy = details["exact_trace_pixel_xy"]
        expected_pixel = [min(w-1, math.floor(float(rep["x_px"])+.5)), min(h-1, math.floor(float(rep["y_px"])+.5))]
        require(pixel_xy == expected_pixel and details["all_source_samples_in_trace"] is True
                and details["original_source_offset_xy"] == offset, "Original exact trace pixel/provenance differs")
        close = videos[Path(bindings["closeup_video"]["path"])]
        close_frames = _frames(details["closeup_source_frames_ui"], "closeup")
        require(close["annotation_section"] == "model" and close["fps"] == fps
                and close_frames == close["source_frames_ui"] == plan["model_closeup_source_frames_ui"][roi],
                "Original closeup source map or rate differs")
        crop = details["crop_xyxy_half_open"]
        require(crop == close["source_crop_xyxy_half_open"] and 0 <= crop[0] < crop[2] <= w
                and 0 <= crop[1] < crop[3] <= h, "Original closeup crop differs")
        member_frames = sorted({int(candidates[pid]["source_frame_ui"]) for pid in members})
        require(site["member_source_frames_ui"] == member_frames, "Site member frame inventory differs")
        context = contract["context_frames"]
        expected_close = sorted(set(frames) & {j for f in member_frames for j in range(f-context, f+context+1)})
        require(close_frames == expected_close, "Closeup does not retain every member frame and fixed context")
        closemap = {frame: index for index, frame in enumerate(close_frames)}
        resources.append(dict(review_resource_id=resource_id, original_model_roi_id=roi,
                              exact_trace_pixel_xy=pixel_xy, member_proposal_ids=members, artifacts=bindings,
                              closeup_source_frames_ui=close_frames, closeup_fps=fps, crop_xyxy_half_open=crop,
                              full_trace_source_sample_count=len(frames), exact_trace_values_validation="inherited_hash_verified",
                              original_representative_proposal_id=site["representative_proposal_id"]))
        for pid in members:
            row, assignment = candidates[pid], assigned[pid]
            require(_candidate_key(row) == _candidate_key(assignment), "Original candidate coordinate/frame/score differs")
            require(assignment["model_roi_id"] == roi, "Original assignment/site membership differs")
            f, x, y = int(row["source_frame_ui"]), int(row["x_px"]), int(row["y_px"])
            require(f in app and f in closemap and crop[0] <= x < crop[2] and crop[1] <= y < crop[3],
                    "Candidate lies outside application or its original closeup")
            distance = math.hypot(x-pixel_xy[0], y-pixel_xy[1])
            require(math.isclose(float(assignment["distance_to_representative_px"]), distance, abs_tol=1e-10),
                    "Original recorded representative distance differs")
            ci, fi = closemap[f], fullmap.get(f)
            ledger.append(dict(cell_id=cell["cell_id"], cohort=cell["cohort"], case_id=cell["case_id"], arm_id=cell["arm_id"],
                proposal_id=pid, source_frame_ui=f, source_time_s=(f-1)/fps, application_frame_index_zero_based=f-app[0],
                x_px=x, y_px=y, original_source_x_px=x+offset[0], original_source_y_px=y+offset[1],
                score=float(row["score"]), threshold_z=float(row["threshold_z"]), original_model_roi_id=roi,
                original_representative_proposal_id=site["representative_proposal_id"], original_trace_x_px=pixel_xy[0],
                original_trace_y_px=pixel_xy[1], distance_to_original_trace_pixel_px=distance,
                candidate_anchor_equals_original_trace_pixel=distance == 0, review_resource_id=resource_id,
                trace_csv=rel["trace_csv"], trace_png=rel["trace_png"], closeup_video=rel["closeup_video"],
                closeup_thumbnail=rel["closeup_thumbnail"], closeup_frame_index_zero_based=ci, closeup_playback_time_s=ci/fps,
                model_fullfield_video=f"{MODEL}/videos/model_sequential_full_field.mp4", fullfield_frame_present=fi is not None,
                fullfield_frame_index_zero_based=fi, fullfield_playback_time_s=None if fi is None else fi/videos[full]["fps"],
                biological_interpretation="unknown_not_negative" if cell["cohort"] == "real" else "known_synthetic_null_source"))
        seen.update(members)
    require(seen == set(ids) and len(ledger) == len(rows), "Not all saved proposals have original media links")
    # Check trace/comparison figure cardinalities against indexed artifacts, including zero-row arms.
    def count(prefix, suffix):
        return sum(str(p.relative_to(audit.resolve())).startswith(prefix) and p.suffix == suffix for p in artifacts)
    for section, n in (("1_Expert_Annotations", summary["expert_roi_count"]), (MODEL, summary["model_roi_count"])):
        require(count(section+"/exact_pixel_traces/", ".csv") == n
                and count(section+"/figures/traces/", ".png") == n, "Full exact trace inventory differs")
    require(count("3_Comparison/trace_comparisons/", ".png") == summary["expert_occurrence_count"],
            "Per-occurrence comparison figure inventory differs")
    ledger.sort(key=lambda r: r["proposal_id"])
    record = dict(cell_id=cell["cell_id"], cohort=cell["cohort"], case_id=cell["case_id"], arm_id=cell["arm_id"],
        audit_root=str(audit.resolve()), source_inputs={k:cell[k] for k in ("candidates","seal","calibration","metadata","completion")},
        inherited_owner=owner, source_manifest=v.binding(audit/"source_manifest.json"), artifact_index=v.binding(audit/"artifact_index.json"),
        original_run_contract=v.binding(audit/"run_contract.json"), video_manifest=v.binding(audit/"video_manifest.json"),
        coverage_manifest=v.binding(audit/"coverage_manifest.json"), inventory_plan=v.binding(audit/"inventory_plan.json"),
        original_model_occurrences=v.binding(audit/MODEL/"model_occurrences.csv"), original_review_sites=v.binding(audit/MODEL/"review_sites.json"),
        source_frames_ui=frames, application_source_frames_ui=app, source_fps=fps, pixel_size_um=pixel,
        application_exposure_s=len(app)/fps, eligible_area_px=op["eligible_area_px"],
        eligible_area_um2=op["eligible_area_px"]*pixel*pixel, original_source_offset_xy=offset,
        evaluation_box_yxyx=metadata["evaluation_box_yxyx"], stage_sequence=contract["stage_sequence"],
        display_limits=contract["display_limits"], operating_point=op, original_summary=summary,
        source_candidate_rows=len(rows), mapped_candidate_rows=len(ledger), review_resources=resources,
        fullfield_source_frames_ui=videos[full]["source_frames_ui"], fullfield_fps=videos[full]["fps"],
        fullfield_video=v.binding(full), original_full_artifact_count=len(artifacts), all_original_artifacts_hash_verified=True)
    return record, ledger


def audit_sources(root, protocol, *, progress_callback=None):
    """Verify all source audits and publish an additive exhaustive proposal ledger.

    May resume before campaign completion: existing ledger bytes must match
    exactly; all input files are reverified. A completed output root is immutable.
    The canonical protocol digest binds caller-provided physical scope; no group
    radius/gap choice changes the media rows or the original trace coordinates.
    """
    root = Path(root).resolve()
    require(not (root/"completion_manifest.json").exists(), "Completed output roots are immutable")
    cells = protocol["cells"]
    require(isinstance(cells, list) and cells, "Empty persistence source matrix")
    keys = []
    for cell in cells:
        _safe(cell["cell_id"], "cell ID")
        require(cell["cohort"] in ("real", "null"), "Unknown persistence cohort")
        keys.append(cell["cell_id"])
    require(len(keys) == len(set(keys)) and len({(c["cohort"],c["case_id"],c["arm_id"]) for c in cells}) == len(cells),
            "Duplicated persistence source cell")
    ctx = _Context()
    protocol_digest = _digest(protocol)
    protocol_binding = None
    if (root/"protocol.json").exists():
        require(_digest(ctx.read(root/"protocol.json")) == protocol_digest, "Caller protocol differs from saved protocol")
        protocol_binding = ctx.v.binding(root/"protocol.json")
    code = [ctx.v.binding(Path(__file__)), ctx.v.binding(Path(__file__).with_name("followup_validate.py"))]
    for binding in protocol.get("code_bindings", []):
        ctx.v.verify(binding)
    result, ledger = [], []
    for index, cell in enumerate(cells):
        record, links = _cell(cell, ctx)
        result.append(record)
        ledger.extend(links)
        if progress_callback:
            progress_callback(dict(completed_cells=index+1, total_cells=len(cells), candidate_rows=len(ledger), cell_id=cell["cell_id"]))
    ledger.sort(key=lambda r: (r["cell_id"], r["proposal_id"]))
    require(len({(r["cell_id"],r["proposal_id"]) for r in ledger}) == len(ledger), "Ambiguous global proposal ledger")
    counts = {cohort: dict(cells=sum(c["cohort"]==cohort for c in result),
              candidate_rows=sum(c["source_candidate_rows"] for c in result if c["cohort"]==cohort)) for cohort in ("real","null")}
    expected_totals = protocol.get("input_proposal_totals", protocol.get("expected_candidate_rows_by_cohort", {}))
    for cohort, expected in expected_totals.items():
        require(counts[cohort]["candidate_rows"] == expected, f"Frozen {cohort} candidate total differs")
    require(protocol.get("expected_cells", len(cells)) == len(cells), "Frozen cell total differs")
    ctx.v.assert_unchanged()
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=SCHEMA, protocol_canonical_sha256=protocol_digest, semantics=SEMANTICS,
                   relative_media_path_base="cells[].audit_root in audit_reuse.json; review_resource_id resolves exact bindings",
                   row_count=len(ledger), rows=ledger)
    _immutable_json(root/"candidate_media_links.json", payload)
    tsv = root/"candidate_media_links.tsv"
    import io
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(ledger)
    _immutable_text(tsv, text.getvalue())
    manifest = dict(schema=SCHEMA, status="PASS", inherited_scientific_audit_complete=True,
                    new_media_rendered=False, no_new_anchor_exact_pixel_traces=True,
                    all_input_proposals_covered=True, cells=len(cells), candidate_rows=len(ledger), cohort_counts=counts,
                    inherited_video_count=sum(r["original_summary"]["video_count"] for r in result),
                    inherited_model_trace_count=sum(r["original_summary"]["model_roi_count"] for r in result),
                    inherited_expert_trace_count=sum(r["original_summary"]["expert_roi_count"] for r in result),
                    inherited_comparison_trace_count=sum(r["original_summary"]["expert_occurrence_count"] for r in result),
                    protocol=protocol_binding, protocol_canonical_sha256=protocol_digest, validation_code=code,
                    semantics=SEMANTICS, source_completion_lineage=[b for b,_,_ in ctx.completions.values()],
                    sources=result, artifacts=[ctx.v.binding(root/"candidate_media_links.json"),ctx.v.binding(tsv)])
    manifest.update(full_validation_unique_files=ctx.v.unique_file_count,
                    full_validation_bytes_hashed=ctx.v.bytes_hashed)
    ctx.v.assert_unchanged()
    _immutable_json(root/"audit_reuse.json", manifest)
    return manifest


def _immutable_text(path, text):
    if path.exists():
        require(path.read_text() == text, f"Refusing to overwrite different frozen reuse output: {path}")
    else:
        # Publish a complete file exclusively. A failed write leaves no partial
        # canonical evidence that would prevent an otherwise valid resume.
        fd, name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(name, path)
            except FileExistsError:
                require(path.read_text() == text, f"Concurrent frozen reuse output differs: {path}")
        finally:
            os.unlink(name)


def _immutable_json(path, value):
    _immutable_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def verify_reuse_receipt(root, protocol):
    """Read-only compact closure; full media/stage hashing is INHERITED.

    Recheck compact source ownership, every ledger member and the original
    metadata/index bindings. MP4/NPY/trace bytes are not rehashed here. Their
    complete file-validation evidence belongs to the unchanged audit_sources
    receipt; this helper must never be described as a fresh full-media pass.
    """
    root = Path(root).resolve()
    ctx = _Context()
    manifest = ctx.read(root/"audit_reuse.json")
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "PASS"
            and manifest.get("inherited_scientific_audit_complete") is True
            and manifest.get("all_input_proposals_covered") is True
            and manifest.get("new_media_rendered") is False, "Reuse receipt is incomplete")
    require(manifest["protocol_canonical_sha256"] == _digest(protocol), "Reuse receipt protocol differs")
    if manifest["protocol"] is not None:
        ctx.v.verify(manifest["protocol"])
        require(_digest(ctx.read(manifest["protocol"]["path"])) == _digest(protocol), "Saved protocol differs")
    expected_code = {Path(__file__).resolve(), Path(__file__).with_name("followup_validate.py").resolve()}
    require({Path(r["path"]).resolve() for r in manifest["validation_code"]} == expected_code,
            "Reuse validation code inventory differs")
    for record in manifest["validation_code"] + manifest["source_completion_lineage"] + manifest["artifacts"]:
        ctx.v.verify(record)
    required_outputs = {root/"candidate_media_links.json", root/"candidate_media_links.tsv"}
    require({Path(r["path"]).resolve() for r in manifest["artifacts"]} == required_outputs, "Reuse ledger inventory differs")
    payload = ctx.read(root/"candidate_media_links.json")
    rows = payload["rows"]
    require(payload["schema"] == SCHEMA and payload["protocol_canonical_sha256"] == _digest(protocol)
            and len(rows) == payload["row_count"] == manifest["candidate_rows"], "Proposal ledger count/protocol differs")
    lookup = {(r["cell_id"],r["proposal_id"]):r for r in rows}
    require(len(lookup) == len(rows), "Duplicated media ledger member")
    with (root/"candidate_media_links.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == list(COLUMNS), "Media TSV columns differ")
        count = 0
        for count, trow in enumerate(reader, 1):
            require(count <= len(rows) and all(trow[k] == ("" if rows[count-1][k] is None else str(rows[count-1][k]))
                                               for k in COLUMNS), "Media JSON/TSV row differs")
        require(count == len(rows), "Media JSON/TSV row count differs")
    originals = {r["cell_id"]:r for r in manifest["sources"]}
    require(len(originals) == len(manifest["sources"]) == manifest["cells"] == len(protocol["cells"])
            and set(originals) == {c["cell_id"] for c in protocol["cells"]}, "Reuse physical-cell inventory differs")
    seen = set()
    for cell in protocol["cells"]:
        record = originals[cell["cell_id"]]
        _, owner = _owner_record(cell, ctx)
        require(owner == record["inherited_owner"], "Reuse inherited owner differs")
        require(Path(record["audit_root"]).resolve() == Path(cell["audit_root"]).resolve(), "Reuse audit path differs")
        for name in ("candidates","seal","calibration","metadata","completion"):
            require(_same(record["source_inputs"][name], cell[name]), "Reuse input binding differs")
            ctx.authoritative(cell[name], cell["completion"]) if name != "completion" else ctx.v.verify(cell[name])
        for name in ("source_manifest","artifact_index","original_run_contract","video_manifest","coverage_manifest",
                     "inventory_plan","original_model_occurrences","original_review_sites"):
            ctx.v.verify(record[name])
        audit = Path(record["audit_root"])
        index = ctx.read(record["artifact_index"]["path"])["artifacts"]
        indexed = {(audit/r["path"]).resolve():r for r in index}
        require(len(indexed) == len(index), "Duplicate original artifact index")
        def index_agrees(binding):
            path = Path(binding["path"]).resolve()
            require(path in indexed and indexed[path]["sha256"] == binding["sha256"]
                    and indexed[path].get("size_bytes") == binding["size_bytes"], "Reuse resource/index binding differs")
        for name in ("video_manifest","coverage_manifest","inventory_plan","original_model_occurrences","original_review_sites"):
            index_agrees(record[name])
        require(record["original_summary"] == ctx.read(audit/"summary.json"), "Reuse summary differs")
        contract = ctx.read(record["original_run_contract"]["path"])
        metadata = ctx.read(cell["metadata"]["path"])
        require(record["source_frames_ui"] == contract["source_frames_ui"] == metadata["source_frames_ui"]
                and record["application_source_frames_ui"] == metadata["application_source_frames_ui"]
                and record["source_fps"] == contract["source_fps"]
                and record["display_limits"] == contract["display_limits"]
                and record["original_source_offset_xy"] == metadata["original_source_offset_xy"],
                "Reuse coordinate/time/display metadata differs")
        candidates = ctx.read(cell["candidates"]["path"])
        cids = [r["proposal_id"] for r in candidates]
        require(len(cids) == len(set(cids)) == record["source_candidate_rows"] == record["mapped_candidate_rows"],
                "Reuse candidate inventory differs")
        resources = {r["review_resource_id"]:r for r in record["review_resources"]}
        require(len(resources) == len(record["review_resources"]) == record["original_summary"]["model_roi_count"],
                "Reuse original ROI inventory differs")
        membership = {}
        for rid, resource in resources.items():
            for binding in resource["artifacts"].values():
                index_agrees(binding)
            ctx.v.verify(resource["artifacts"]["metadata"])
            detail = ctx.read(resource["artifacts"]["metadata"]["path"])
            require(detail["exact_trace_pixel_xy"] == resource["exact_trace_pixel_xy"]
                    and detail["member_proposal_ids"] == resource["member_proposal_ids"]
                    and detail["closeup_source_frames_ui"] == resource["closeup_source_frames_ui"],
                    "Reuse original ROI metadata differs")
            for pid in resource["member_proposal_ids"]:
                require(pid not in membership, "Reuse member belongs to multiple original ROIs")
                membership[pid] = rid
        require(set(membership) == set(cids), "Reuse resources do not cover every original proposal")
        for candidate in candidates:
            key = (cell["cell_id"],candidate["proposal_id"])
            require(key in lookup, "Missing proposal media link")
            row = lookup[key]; resource = resources[membership[candidate["proposal_id"]]]
            require(row["review_resource_id"] == resource["review_resource_id"]
                    and row["original_model_roi_id"] == resource["original_model_roi_id"]
                    and all(float(row[k]) == float(candidate[k]) for k in ("source_frame_ui","x_px","y_px","score","threshold_z")),
                    "Proposal media identity/coordinate/score differs")
            for name in ("trace_csv","trace_png","closeup_video","closeup_thumbnail"):
                require((audit/row[name]).resolve() == Path(resource["artifacts"][name]["path"]).resolve(),
                        "Proposal media link targets a different original artifact")
            xy = resource["exact_trace_pixel_xy"]
            distance = math.hypot(row["x_px"]-xy[0],row["y_px"]-xy[1])
            require([row["original_trace_x_px"],row["original_trace_y_px"]] == xy
                    and row["distance_to_original_trace_pixel_px"] == distance
                    and row["candidate_anchor_equals_original_trace_pixel"] == (distance == 0),
                    "New anchor was confused with original trace pixel")
            frame = row["source_frame_ui"]
            offset = record["original_source_offset_xy"]
            require(row["cohort"] == cell["cohort"] and row["case_id"] == cell["case_id"] and row["arm_id"] == cell["arm_id"]
                    and row["original_source_x_px"] == row["x_px"]+offset[0]
                    and row["original_source_y_px"] == row["y_px"]+offset[1]
                    and row["source_time_s"] == (frame-1)/record["source_fps"]
                    and row["application_frame_index_zero_based"] == frame-record["application_source_frames_ui"][0],
                    "Proposal source coordinate/time differs")
            ci = resource["closeup_source_frames_ui"].index(frame)
            fm = record["fullfield_source_frames_ui"]
            fi = fm.index(frame) if frame in fm else None
            require(row["closeup_frame_index_zero_based"] == ci and row["closeup_playback_time_s"] == ci/resource["closeup_fps"]
                    and row["fullfield_frame_present"] == (fi is not None) and row["fullfield_frame_index_zero_based"] == fi
                    and row["fullfield_playback_time_s"] == (None if fi is None else fi/record["fullfield_fps"]),
                    "Proposal media source/playback frame map differs")
            seen.add(key)
    require(seen == set(lookup), "Extra media ledger rows outside the physical source matrix")
    counts = {cohort:dict(cells=sum(c["cohort"]==cohort for c in protocol["cells"]),
                         candidate_rows=sum(row["cohort"]==cohort for row in rows)) for cohort in ("real","null")}
    require(counts == manifest["cohort_counts"], "Reuse cohort counts differ")
    for cohort, expected in protocol.get("input_proposal_totals", {}).items():
        require(counts[cohort]["candidate_rows"] == expected, "Reuse frozen proposal total differs")
    for output, field in (("inherited_video_count","video_count"),("inherited_model_trace_count","model_roi_count"),
                          ("inherited_expert_trace_count","expert_roi_count"),("inherited_comparison_trace_count","expert_occurrence_count")):
        require(manifest[output] == sum(r["original_summary"][field] for r in originals.values()),
                "Reuse aggregate original media inventory differs")
    ctx.v.assert_unchanged()
    return dict(status="PASS", scope="compact_receipt_sources_metadata_and_ledger_only",
                full_media_and_stage_file_validation="inherited_from_bound_audit_reuse_receipt_not_rehashed",
                audit_reuse=ctx.v.binding(root/"audit_reuse.json"), cells=len(originals), candidate_rows=len(rows),
                compact_unique_files_hashed=ctx.v.unique_file_count, compact_bytes_hashed=ctx.v.bytes_hashed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root/"protocol.json").read_text())
    result = audit_sources(args.root, protocol, progress_callback=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps({k:result[k] for k in ("status","cells","candidate_rows","inherited_video_count")}), flush=True)


if __name__ == "__main__":
    main()
