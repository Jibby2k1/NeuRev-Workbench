"""Build a lazy-media annotation-correction Workbench from a frozen MSICA->MSLN audit."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from neurobench.annotations import default_annotations_v3
from neurobench.algorithms.msln_msica_cuda import causal_joint_msln_cuda, cuda_device_summary
from neurobench.experiments.msln_msica.artifacts import atomic_json, sha256_file
from neurobench.experiments.msln_msica.multilag_program import (
    _aligned_outputs,
    _contexts,
    _fit_from_dict,
    _load,
    _source,
)
from neurobench.workbench.annotation_revisions import initialize_revision_root
from neurobench.workbench.builder import build_workbench

FROZEN_LANE = (
    "multilag_2d__normalized_hsic__short__uniform__bandwidth_scale-0p5"
    "::persistence::joint_s5_g1_t31_g1"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_occurrences(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[key])].append({
            "burst": int(row["burst"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "start_ui": int(row["start_ui"]),
            "end_ui": int(row["end_ui"]),
            **({"score": float(row["score"]), "rank": int(row["rank"]),
                "expert_supported": str(row["expert_supported"]).lower() == "true",
                "matched_expert_roi": str(row.get("matched_expert_roi") or ""),
                "match_distance_px": float(row["match_distance_px"]) if row.get("match_distance_px") else None}
               if key == "model_roi" else {}),
        })
    return dict(result)


def _identity_matches(model_groups: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], set[str]]:
    candidates: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    supported_models: set[str] = set()
    for model_id, members in model_groups.items():
        by_expert: dict[str, list[float]] = defaultdict(list)
        for member in members:
            expert_id = str(member.get("matched_expert_roi") or "")
            distance = member.get("match_distance_px")
            if expert_id and distance is not None:
                supported_models.add(model_id)
                by_expert[expert_id].append(float(distance))
        for expert_id, distances in by_expert.items():
            candidates[expert_id].append((len(distances), float(np.mean(distances)), model_id))
    matches = []
    used_models: set[str] = set()
    for expert_id in sorted(candidates):
        choices = sorted(candidates[expert_id], key=lambda item: (-item[0], item[1], item[2]))
        selected = next((item for item in choices if item[2] not in used_models), None)
        if selected is None:
            continue
        count, distance, model_id = selected
        used_models.add(model_id)
        matches.append({
            "expert_id": expert_id,
            "model_id": model_id,
            "distance_px": round(distance, 6),
            "identity_support_occurrences": count,
        })
    return matches, supported_models


def _trace(values: np.ndarray, x: float, y: float, radius: float) -> tuple[list[float], list[float]]:
    cx, cy = int(round(x)), int(round(y))
    cx = min(max(cx, 0), values.shape[2] - 1)
    cy = min(max(cy, 0), values.shape[1] - 1)
    yy, xx = np.ogrid[:values.shape[1], :values.shape[2]]
    mask = (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius) ** 2
    if not bool(mask.any()):
        mask[cy, cx] = True
    pixel = np.asarray(values[:, cy, cx], dtype=np.float32)
    mean = np.asarray(values[:, mask].mean(axis=1), dtype=np.float32)
    return np.round(pixel, 4).tolist(), np.round(mean, 4).tolist()


def _attach_traces(items: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    for index, item in enumerate(items, 1):
        radius = float(item["geometry"].get("radius_px", 3.0))
        item["traces"] = {}
        for view_id, values in arrays.items():
            pixel, mean = _trace(values, item["source_xy"][0], item["source_xy"][1], radius)
            item["traces"][view_id] = {"pixel": pixel, "roi_mean": mean}
        if index % 25 == 0:
            print(f"TRACES {index}/{len(items)}", flush=True)


def _uint8_unsigned(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((np.asarray(frame, dtype=np.float32) - low) * (255.0 / max(high - low, 1e-8)), 0, 255).astype(np.uint8)


def _uint8_signed(frame: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(127.5 + np.asarray(frame, dtype=np.float32) * (127.5 / max(limit, 1e-8)), 0, 255).astype(np.uint8)


def _render_frames(values: np.ndarray, root: Path, start_ui: int, *, signed: bool, resume: bool) -> dict[str, float]:
    root.mkdir(parents=True, exist_ok=True)
    sample = np.asarray(values[::max(1, len(values) // 32), ::4, ::4], dtype=np.float32)
    if signed:
        low, high = -float(np.percentile(np.abs(sample), 99.5)), float(np.percentile(np.abs(sample), 99.5))
    else:
        low, high = map(float, np.percentile(sample, [1.0, 99.8]))
    for index, frame in enumerate(values):
        ui_frame = start_ui + index
        target = root / f"frame_{ui_frame:04d}.png"
        if not (resume and target.is_file()):
            rendered = _uint8_signed(frame, high) if signed else _uint8_unsigned(frame, low, high)
            Image.fromarray(rendered, mode="L").save(target, compress_level=2)
        if (index + 1) % 40 == 0 or index + 1 == len(values):
            print(f"FRAMES {root.name} {index + 1}/{len(values)}", flush=True)
    return {"low": low, "high": high}


def _link_raw_frames(source: Path, target: Path, start_ui: int, stop_ui: int, resume: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for ui_frame in range(start_ui, stop_ui + 1):
        src = source / f"frame_{ui_frame:03d}.png"
        dst = target / f"frame_{ui_frame:04d}.png"
        if resume and dst.is_file():
            continue
        if not src.is_file():
            raise FileNotFoundError(src)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def _items(groups: dict[str, list[dict[str, Any]]], kind: str) -> list[dict[str, Any]]:
    result = []
    for identity, members in sorted(groups.items()):
        x = float(np.median([item["x"] for item in members]))
        y = float(np.median([item["y"] for item in members]))
        intervals = sorted([[item["start_ui"], item["end_ui"]] for item in members])
        result.append({
            "id": identity,
            "source_xy": [round(x, 3), round(y, 3)],
            "ui_frame": intervals[0][0],
            "events": [interval[0] for interval in intervals],
            "eventIntervals": intervals,
            "event_intervals": intervals,
            "geometry": {"kind": "circle", "radius_px": 5.0} if kind == "expert" else {"kind": "center"},
            "status": "unknown" if kind == "model" else "expert",
            "members": members,
        })
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.authorize_full_spon:
        raise PermissionError("real Spon dashboard generation requires --authorize-full-spon")
    config = _load(args.config)
    output = Path(args.output_root).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output root exists: {output}")
    audit = Path(args.audit_root).resolve()
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    if summary["frozen_lane"] != args.lane:
        raise ValueError("audit frozen lane does not match requested dashboard lane")
    required = [
        Path(config["source"]["movie_path"]),
        Path(config["source"]["labels_path"]),
        Path(config["outputs"]["root_dir"]) / "stage_a" / "surface.json",
        audit / "1_Expert_Annotations" / "expert_occurrences.csv",
        audit / "2_Model_Annotations" / "model_occurrences.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    disk = shutil.disk_usage(output.parent if output.parent.exists() else output.parent.parent)
    if disk.free < 5 * 2**30:
        raise RuntimeError("less than 5 GiB disk headroom")
    device = cuda_device_summary()
    cap = int(float(config["compute"]["max_peak_vram_gb"]) * 2**30)
    if device["free_bytes"] < cap:
        raise RuntimeError("free VRAM is below the frozen 8 GiB cap")
    output.mkdir(parents=True, exist_ok=True)
    app = output / "app"
    app.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "preflight.json", {
        "ready": True, "source_read_only": True, "output_collision_checked": True,
        "frozen_lane": args.lane, "gpu": device, "vram_cap_bytes": cap,
        "disk_free_bytes": disk.free, "created_at": now,
    })

    config_id, branch, context_id = args.lane.split("::")
    surface = json.loads((Path(config["outputs"]["root_dir"]) / "stage_a" / "surface.json").read_text(encoding="utf-8"))
    row = next((item for item in surface["expansion_rows"] if item["config_id"] == config_id), None)
    if row is None:
        raise ValueError(f"frozen config not found: {config_id}")
    context = next((item for item in _contexts(config) if item.context_id == context_id), None)
    if context is None:
        raise ValueError(f"MSLN context not found: {context_id}")
    values, quiet, global_history, msln_pre_roll = _source(config)
    fit = _fit_from_dict(row["fit"])
    outputs = _aligned_outputs(values, fit, config, global_history, msln_pre_roll)
    msica_with_preroll = outputs[branch]
    quiet_extended = np.concatenate((np.zeros(msln_pre_roll, dtype=bool), quiet))
    normalized = causal_joint_msln_cuda(
        msica_with_preroll, context, quiet_mask=quiet_extended,
        review_crop_frames=msln_pre_roll, max_vram_bytes=cap,
    )
    import cupy as cp
    msln = cp.asnumpy(normalized.values)
    cp.get_default_memory_pool().free_all_blocks()
    msica = np.asarray(msica_with_preroll[msln_pre_roll:], dtype=np.float32)
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    raw = movie[review_start - 1:review_stop]
    if raw.shape != msica.shape or raw.shape != msln.shape:
        raise RuntimeError(f"stage alignment mismatch: raw={raw.shape} msica={msica.shape} msln={msln.shape}")

    expert_groups = _group_occurrences(_rows(audit / "1_Expert_Annotations" / "expert_occurrences.csv"), "roi")
    model_groups = _group_occurrences(_rows(audit / "2_Model_Annotations" / "model_occurrences.csv"), "model_roi")
    experts = _items(expert_groups, "expert")
    models = _items(model_groups, "model")
    matches, supported_models = _identity_matches(model_groups)
    match_by_expert = {item["expert_id"]: item["model_id"] for item in matches}
    match_by_model = {item["model_id"]: item["expert_id"] for item in matches}
    for item in experts:
        item["linked_model_id"] = match_by_expert.get(item["id"], "")
    for item in models:
        item["linked_expert_id"] = match_by_model.get(item["id"], "")
        item["status"] = "matched" if item["linked_expert_id"] else "expert_supported_unlinked" if item["id"] in supported_models else "unknown"
    _attach_traces(experts + models, {"raw": raw, "msica": msica, "msln": msln})

    frames = app / "frames"
    raw_source = Path(args.raw_frames_root).resolve()
    _link_raw_frames(raw_source, frames / "raw", review_start, review_stop, args.resume)
    msica_range = _render_frames(msica, frames / "msica", review_start, signed=False, resume=args.resume)
    msln_range = _render_frames(msln, frames / "msln", review_start, signed=True, resume=args.resume)

    source_id = "spon_ca_burst_3_hindbrain_to_tail_488_20ms"
    revision_id = "spon_multilag_v5_correction_draft_v1"
    revision = {
        "schema_version": 1, "revisionId": revision_id, "parentRevisionId": "spon_labels_import_v1",
        "state": "draft", "reviewerId": "reviewer_local_1", "frozenRunId": args.lane,
        "sourceAnnotationsSha256": sha256_file(Path(config["source"]["labels_path"])),
        "createdAt": now, "updatedAt": now, "revisionToken": 0, "operationCount": 0,
    }
    contracts = [
        {"schema_version":1,"view_id":"raw","source_video_id":source_id,"shape_tyx":list(raw.shape),
         "source_to_view":{"kind":"identity"},"frame_mapping":{"kind":"identity","offset":review_start-1},
         "intensity_semantics":"raw_amplitude","frame_pattern":"frames/raw/frame_%04d.png"},
        {"schema_version":1,"view_id":"msica","source_video_id":source_id,"shape_tyx":list(msica.shape),
         "source_to_view":{"kind":"identity"},"frame_mapping":{"kind":"identity","offset":review_start-1},
         "intensity_semantics":"normalized_unsigned_visualization","frame_pattern":"frames/msica/frame_%04d.png"},
        {"schema_version":1,"view_id":"msln","source_video_id":source_id,"shape_tyx":list(msln.shape),
         "source_to_view":{"kind":"identity"},"frame_mapping":{"kind":"identity","offset":review_start-1},
         "intensity_semantics":"normalized_signed_visualization","frame_pattern":"frames/msln/frame_%04d.png"},
    ]
    review_data = {
        "dataset":{"dataset_id":source_id,"name":"Spon Ca Burst · MSICA→MSLN correction","frame_rate_hz":50.0},
        "video":{"name":source_id,"width":raw.shape[2],"height":raw.shape[1],"frames":raw.shape[0],"fps":50.0,
                 "framePattern":"frames/raw/frame_%04d.png"},
        "parameters":{"purpose":"single-reviewer real-data annotation correction","frozen_lane":args.lane},
        "rois":[],
        "annotationCorrection":{"schema_version":1,"source_video_id":source_id,"read_only":False,
            "revision":revision,"view_contracts":contracts,"expert_rois":experts,"model_rois":models,"matches":matches},
    }
    atomic_json(app / "review_data.json", review_data)
    build_workbench(app_dir=app, review_data_path=app / "review_data.json", dataset_id=source_id)
    annotations = default_annotations_v3()
    annotations["rois"] = {
        item["id"]:{"source_xy":item["source_xy"],"geometry":item["geometry"],
                    "event_intervals":item["eventIntervals"],"linked_model_id":item["linked_model_id"],
                    "cell_state":"accepted","reviewer_id":"reviewer_local_1","deleted":False,"notes":""}
        for item in experts
    }
    revision_root = app / "annotation_revisions" / revision_id
    if not revision_root.exists():
        initialize_revision_root(app / "annotation_revisions", revision=revision, annotations=annotations)
    shutil.copyfile(revision_root / "annotations.json", app / "annotations.json")
    validation = {
        "status":"complete","route":"#annotation-correction","layout_contract":"verified_50_50_slice4",
        "frame_interval_ui":[review_start,review_stop],"shape_tyx":list(raw.shape),
        "expert_identities":len(experts),"expert_occurrences":sum(map(len, expert_groups.values())),
        "model_identities":len(models),"model_occurrences":sum(map(len, model_groups.values())),
        "identity_matches":len(matches),"supported_unlinked_models":len(supported_models-set(match_by_model)),
        "lazy_frame_patterns":{item["view_id"]:item["frame_pattern"] for item in contracts},
        "display_ranges":{"msica":msica_range,"msln":msln_range},
        "msln_diagnostics":normalized.diagnostics,"review_data_bytes":(app/"review_data.json").stat().st_size,
        "arrays_embedded":False,"unmatched_candidates":"unknown",
    }
    atomic_json(output / "validation.json", validation)
    atomic_json(output / "status.json", {"status":"complete","scientific_status":"frozen_audit_attached","completed_at":datetime.now(timezone.utc).isoformat()})
    del outputs, msica_with_preroll, msln, msica
    gc.collect()
    return {"status":"complete","output_root":str(output),"app_dir":str(app),**validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/spon_ca_burst_multilag_msica_v5.example.json")
    parser.add_argument("--audit-root", default="Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_three_way_roi_audit_v3")
    parser.add_argument("--raw-frames-root", default="Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/app/frames_fiji_like")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lane", default=FROZEN_LANE)
    parser.add_argument("--authorize-full-spon", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
