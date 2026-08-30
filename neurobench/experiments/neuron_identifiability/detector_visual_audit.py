"""Three-section visual audit for the frozen compact detector lanes."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.hard_roi_adjudication.adjudication import label_view, load_tsv
from neurobench.experiments.hard_roi_adjudication.config import HardRoiAdjudicationConfig
from neurobench.experiments.hard_roi_adjudication.reevaluate import (
    REVIEW_START_UI, _candidate_records, _event_bounds, _frames, _load_feature,
    _match_spatiotemporal,
)
from neurobench.experiments.hierarchical_parzen_ica.patch_information_program import _pool_values
from neurobench.reports.scientific_audit import require_three_section_scientific_audit

from .contracts import atomic_json, atomic_text


GREEN = (75, 220, 130)
ORANGE = (255, 145, 40)
YELLOW = (245, 225, 120)
LANES = ("carrier_signed", "coherence_w15", "propagation_lag2_w15")
COMPACT_LANES = LANES[1:]
BUDGET = 20


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _limits(values: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(values[::23, ::7, ::7], dtype=np.float32)
    low, high = np.percentile(sample, [1.0, 99.8])
    return float(low), max(float(high), float(low) + 1e-7)


def _gray(values: np.ndarray, limits: tuple[float, float], size: tuple[int, int]) -> Image.Image:
    low, high = limits
    scaled = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").resize(size, Image.Resampling.BILINEAR).convert("RGB")


def _encode(path: Path, frames: Iterable[np.ndarray], size: tuple[int, int], fps: float = 20.0) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "29", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode() if process.stderr else ""
        code = process.wait()
    except BaseException:
        process.kill(); raise
    if code:
        raise RuntimeError(error[-2000:])
    partial.replace(path)


def _proposals(values: np.ndarray, labels: list[dict[str, Any]], config: HardRoiAdjudicationConfig, lane: str) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    events: dict[int, np.ndarray] = {}
    bounds: dict[int, tuple[int, int]] = {}
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        bounds[burst] = _event_bounds(labels, burst)
        events[burst] = _frames(values, *bounds[burst])
    maps = _pool_values(np.asarray(values[:100]), events, float(config.evaluation["temporal_pool_temperature"]))["events"]
    rows = []
    for burst, (start, end) in bounds.items():
        for candidate in _candidate_records(values, maps[burst], start, end, distance=6, limit=BUDGET):
            rows.append({"lane": lane, "burst_id": burst, "start_ui": start, "end_ui": end, **candidate})
    return maps, rows


def _consolidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for lane in COMPACT_LANES:
        groups: list[dict[str, Any]] = []
        for row in [value for value in rows if value["lane"] == lane]:
            group = next((g for g in groups if (row["x_px"] - g["x"]) ** 2 + (row["y_px"] - g["y"]) ** 2 <= 36), None)
            if group is None:
                group = {"x": float(row["x_px"]), "y": float(row["y_px"]), "members": []}; groups.append(group)
            group["members"].append(row)
        groups.sort(key=lambda g: (-len({m["burst_id"] for m in g["members"]}), -max(m["score"] for m in g["members"]), g["y"], g["x"]))
        for index, group in enumerate(groups, 1):
            model_id = f"{lane}__model_roi_{index:03d}"
            for member in group["members"]: member["model_roi"] = model_id
            models.append({"id": model_id, "lane": lane, "x": group["x"], "y": group["y"], "members": group["members"], "intervals": [[m["start_ui"], m["end_ui"]] for m in group["members"]]})
    return models


def _pooled_timeline(shape: tuple[int, ...], maps: dict[int, np.ndarray], labels: list[dict[str, Any]]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    for burst, score_map in maps.items():
        start, end = _event_bounds(labels, burst)
        result[start - REVIEW_START_UI:end - REVIEW_START_UI + 1] = score_map
    return result


def _best_lag_and_correlation(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> tuple[float | None, int | None]:
    best: tuple[float, int] | None = None
    for lag in range(-max_lag, max_lag + 1):
        left, right = (a[-lag:], b[:len(b) + lag]) if lag < 0 else ((a[:-lag], b[lag:]) if lag > 0 else (a, b))
        if len(left) < 3 or float(np.std(left)) == 0 or float(np.std(right)) == 0: continue
        value = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(value) and (best is None or value > best[0]): best = (value, lag)
    return (None, None) if best is None else best


def _crop(x: float, y: float, width: int, height: int, radius: int = 28) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 2 * radius - 1, int(round(x)) - radius))
    y0 = max(0, min(height - 2 * radius - 1, int(round(y)) - radius))
    return x0, y0, x0 + 2 * radius + 1, y0 + 2 * radius + 1


def _full_frames(arrays: list[np.ndarray], titles: list[str], markers: Any, color: tuple[int, int, int]) -> Iterable[np.ndarray]:
    panel, height, header = 286, 170, 42
    limits = [_limits(value) for value in arrays]; font = ImageFont.load_default()
    for index in range(len(arrays[0])):
        ui = REVIEW_START_UI + index
        canvas = Image.new("RGB", (panel * len(arrays), height + header), "black"); draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"Frozen B20 audit | UI {ui} | grayscale evidence", fill="white", font=font)
        for column, (values, title, limit) in enumerate(zip(arrays, titles, limits, strict=True)):
            canvas.paste(_gray(values[index], limit, (panel, height)), (column * panel, header)); draw.text((column * panel + 6, 24), title, fill="white", font=font)
            for marker in markers(ui):
                x = float(marker.get("x_px", marker.get("x"))); y = float(marker.get("y_px", marker.get("y")))
                cx = column * panel + x * panel / arrays[0].shape[2]; cy = header + y * height / arrays[0].shape[1]
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=color, width=2)
        yield np.asarray(canvas, np.uint8)


def _close_frames(arrays: list[np.ndarray], titles: list[str], item: dict[str, Any], color: tuple[int, int, int]) -> Iterable[np.ndarray]:
    panel, header = 128, 36; x = float(item.get("x_px", item.get("x"))); y = float(item.get("y_px", item.get("y")))
    x0, y0, x1, y1 = _crop(x, y, arrays[0].shape[2], arrays[0].shape[1]); limits = [_limits(value) for value in arrays]; font = ImageFont.load_default()
    for index in range(len(arrays[0])):
        canvas = Image.new("RGB", (panel * len(arrays), panel + header), "black"); draw = ImageDraw.Draw(canvas)
        draw.text((5, 4), f"{item['id']} | UI {REVIEW_START_UI + index}", fill="white", font=font)
        for column, (values, title, limit) in enumerate(zip(arrays, titles, limits, strict=True)):
            canvas.paste(_gray(values[index, y0:y1, x0:x1], limit, (panel, panel)), (column * panel, header)); draw.text((column * panel + 4, 20), title, fill="white", font=font)
            cx = column * panel + (x - x0) * panel / (x1 - x0); cy = header + (y - y0) * panel / (y1 - y0)
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=color, width=2)
        yield np.asarray(canvas, np.uint8)


def _trace(path: Path, arrays: list[np.ndarray], titles: list[str], item: dict[str, Any], color: str) -> None:
    if path.is_file(): return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = int(round(float(item.get("x_px", item.get("x"))))); y = int(round(float(item.get("y_px", item.get("y")))))
    fig, axes = plt.subplots(len(arrays), 1, figsize=(8, 2.0 * len(arrays)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, values, title in zip(axes, arrays, titles, strict=True):
        axis.plot(np.arange(len(values)) + REVIEW_START_UI, values[:, y, x], color=color, lw=.8); axis.set_ylabel(title); axis.grid(alpha=.2)
    axes[-1].set_xlabel("UI frame (one-based)"); fig.suptitle(f"{item['id']} | exact pixel x={x}, y={y}")
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


def _comparisons(root: Path, labels: list[dict[str, Any]], proposals: list[dict[str, Any]], raw: np.ndarray, values: dict[str, np.ndarray], maps: dict[str, dict[int, np.ndarray]], match_radius: float) -> list[dict[str, Any]]:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    comp = root / "3_Comparison"; (comp / "trace_comparisons").mkdir(parents=True, exist_ok=True); rows = []
    assigned: dict[tuple[str, int, str], bool] = {}
    for lane in COMPACT_LANES:
        for burst in sorted({int(row["burst_id"]) for row in labels}):
            burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
            burst_candidates = [row for row in proposals if row["lane"] == lane and row["burst_id"] == burst]
            matches, _ = _match_spatiotemporal(burst_candidates, burst_labels, match_radius)
            for label_index in range(len(burst_labels)):
                assigned[(lane, burst, burst_labels[label_index]["observation_id"])] = label_index in matches
    for label in labels:
        burst = int(label["burst_id"]); lane_nearest = {}
        for lane in COMPACT_LANES:
            candidates = [row for row in proposals if row["lane"] == lane and row["burst_id"] == burst]
            lane_nearest[lane] = min(candidates, key=lambda row: (row["x_px"] - float(label["x_px"])) ** 2 + (row["y_px"] - float(label["y_px"])) ** 2)
        nearest = min(lane_nearest.values(), key=lambda row: (row["x_px"] - float(label["x_px"])) ** 2 + (row["y_px"] - float(label["y_px"])) ** 2)
        distance = float(np.hypot(nearest["x_px"] - float(label["x_px"]), nearest["y_px"] - float(label["y_px"])))
        start = int(label["start_frame_ui"]); end = int(label["end_frame_ui"]); sl = slice(start - REVIEW_START_UI, end - REVIEW_START_UI + 1)
        ex, ey = int(round(float(label["x_px"]))), int(round(float(label["y_px"])))
        candidate_trace = values[nearest["lane"]][sl, int(nearest["y_px"]), int(nearest["x_px"])]
        expert_trace = values[nearest["lane"]][sl, ey, ex]
        corr, lag = _best_lag_and_correlation(expert_trace, candidate_trace)
        row = {"occurrence_id": label["observation_id"], "burst_id": burst, "expert_roi": label["roi_identity"], "nearest_lane": nearest["lane"], "model_roi": nearest["model_roi"], "distance_px": distance, "candidate_rank": nearest["rank"], "candidate_score": nearest["score"], "event_correlation": "" if corr is None else corr, "best_lag_frames": "" if lag is None else lag, "within_match_radius": distance <= match_radius, "one_to_one_match": assigned[(nearest["lane"], burst, label["observation_id"])], "interpretation": "known-positive nearest candidate; see one_to_one_match" if distance <= match_radius else "known-positive not recovered; unmatched candidates remain unknown"}; rows.append(row)
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
        frames = np.arange(start, end + 1); axes[0].plot(frames, raw[sl, ey, ex], color="#38a169", label="expert exact pixel")
        for axis, lane in zip(axes[1:], COMPACT_LANES, strict=True):
            candidate = lane_nearest[lane]; axis.plot(frames, values[lane][sl, ey, ex], color="#38a169", label="expert location"); axis.plot(frames, values[lane][sl, int(candidate["y_px"]), int(candidate["x_px"])], color="#dd6b20", ls="--", label=f"nearest B20 model, rank {candidate['rank']}"); axis.set_ylabel(lane); axis.legend(fontsize=7)
        axes[0].set_ylabel("raw"); axes[0].legend(fontsize=7); axes[-1].set_xlabel("UI frame"); fig.suptitle(f"{label['observation_id']} | nearest compact candidate {distance:.2f} px")
        fig.savefig(comp / "trace_comparisons" / f"{label['observation_id']}_comparison.png", dpi=110); plt.close(fig)
    _csv(comp / "expert_model_matches.csv", rows); _csv(comp / "nearest_roi_trace_metrics.csv", rows)
    for burst in sorted(maps[COMPACT_LANES[0]]):
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]; burst_proposals = [row for row in proposals if row["burst_id"] == burst and row["lane"] in COMPACT_LANES]
        start, end = _event_bounds(labels, burst); raw_map = np.max(_frames(raw, start, end), axis=0); compact_map = np.maximum(maps[COMPACT_LANES[0]][burst], maps[COMPACT_LANES[1]][burst])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2)); axes[0].imshow(raw_map, cmap="gray"); axes[0].set_title("Raw matched comparison"); axes[1].imshow(compact_map, cmap="gray"); axes[1].set_title("MSICA + MSLN matched comparison")
        for axis in axes:
            for label in burst_labels: axis.scatter(label["x_px"], label["y_px"], s=30, facecolors="none", edgecolors="#38a169", linewidths=1.2)
            for candidate in burst_proposals: axis.scatter(candidate["x_px"], candidate["y_px"], s=18, marker="s", facecolors="none", edgecolors="#dd6b20", linewidths=.8)
            axis.set_axis_off()
        fig.suptitle(f"Burst {burst}: green expert positives, orange frozen compact B20 candidates"); fig.tight_layout(); fig.savefig(comp / f"burst_{burst}_comparison.png", dpi=130); plt.close(fig)
    numeric = [row for row in rows if row["event_correlation"] != ""]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].hist([row["distance_px"] for row in rows], bins=12, color="#dd6b20", edgecolor="#333333"); axes[0].axvline(match_radius, color="#333333", ls="--"); axes[0].set(title="Nearest compact-candidate distance", xlabel="distance (px)", ylabel="expert occurrences")
    axes[1].scatter([row["distance_px"] for row in numeric], [float(row["event_correlation"]) for row in numeric], c="#dd6b20", s=18); axes[1].set(title="Distance versus event similarity", xlabel="distance (px)", ylabel="best-lag correlation")
    axes[2].scatter([row["candidate_rank"] for row in rows], [row["distance_px"] for row in rows], c="#dd6b20", s=18); axes[2].set(title="Rank versus distance", xlabel="candidate rank", ylabel="distance (px)")
    fig.savefig(comp / "aggregate_diagnostics.png", dpi=130); plt.close(fig)
    atomic_text(comp / "README.md", "# Comparison\n\nFigures and tables only. Green denotes expert positive labels; orange denotes frozen B20 model candidates; pale-yellow links are reserved for established matches. The two panel titles retain the repository audit-schema names, while the right panel contains the maximum of the two compact frozen maps. Unmatched candidates remain unknown.\n")
    return rows


def run_detector_visual_audit(data_root: Path, repo_root: Path, output: Path) -> dict[str, Any]:
    data_root, repo_root, output = data_root.resolve(), repo_root.resolve(), output.resolve()
    if output.exists(): raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    if partial.exists(): raise FileExistsError(partial)
    config = HardRoiAdjudicationConfig.load(repo_root / "examples/spon_ca_burst_hard_roi_adjudication_v1.example.json")
    label_path = data_root / "Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv"
    labels = label_view(load_tsv(label_path), "original", "original")
    raw_path = data_root / "Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
    raw_all = np.load(raw_path, mmap_mode="r", allow_pickle=False); raw = raw_all[REVIEW_START_UI - 1:REVIEW_START_UI - 1 + 560]
    values: dict[str, np.ndarray] = {}; feature_meta = {}
    feature_paths = {
        "carrier_signed": data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy",
        "coherence_w15": data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1/videos/coherence_w15.tif",
        "propagation_lag2_w15": data_root / "Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1/videos/propagation_lag2_w15.tif",
    }
    for row in config.frozen_panel:
        lane = str(row["feature_id"])
        if lane in LANES:
            values[lane], feature_meta[lane] = _load_feature({**row, "path": feature_paths[lane]})
    maps = {}; proposals = []
    for lane in LANES:
        maps[lane], rows = _proposals(values[lane], labels, config, lane); proposals.extend(rows)
    models = _consolidate(proposals); unique = {row["roi_identity"]: row for row in labels}
    for path in (partial / "1_Expert_Annotations/videos/closeups", partial / "1_Expert_Annotations/figures/traces", partial / "1_Expert_Annotations/metadata", partial / "2_Model_Annotations/videos/closeups", partial / "2_Model_Annotations/figures/traces", partial / "2_Model_Annotations/metadata"):
        path.mkdir(parents=True, exist_ok=True)
    expert = partial / "1_Expert_Annotations"; model = partial / "2_Model_Annotations"
    pooled = {lane: _pooled_timeline(raw.shape, maps[lane], labels) for lane in COMPACT_LANES}
    expert_arrays = [raw, values[COMPACT_LANES[0]], values[COMPACT_LANES[1]]]
    expert_titles = ["Raw", *COMPACT_LANES]
    _encode(expert / "videos/expert_annotations_full_field.mp4", _full_frames(expert_arrays, expert_titles, lambda ui: [row for row in labels if int(row["start_frame_ui"]) <= ui <= int(row["end_frame_ui"])], GREEN), (286 * 3, 212))
    model_arrays = [raw, values["carrier_signed"], values[COMPACT_LANES[0]], pooled[COMPACT_LANES[0]], values[COMPACT_LANES[1]], pooled[COMPACT_LANES[1]]]; model_titles = ["Raw", "carrier_signed", "coherence_w15", "coherence pooled rank map", "propagation_lag2_w15", "propagation pooled rank map"]
    _encode(model / "videos/model_annotations_sequential_full_field.mp4", _full_frames(model_arrays, model_titles, lambda ui: [item for item in models if any(a <= ui <= b for a, b in item["intervals"])], ORANGE), (286 * 6, 212))
    for roi_id, source in unique.items():
        item = {**source, "id": roi_id}; _encode(expert / f"videos/closeups/{roi_id}.mp4", _close_frames(expert_arrays, expert_titles, item, GREEN), (128 * 3, 164)); _trace(expert / f"figures/traces/{roi_id}.png", expert_arrays, expert_titles, item, "#38a169"); atomic_json(expert / f"metadata/roi_{roi_id}.json", {"roi_identity": roi_id, "x_px": item["x_px"], "y_px": item["y_px"], "coordinate_contract": "x=column,y=row"})
    for item in models:
        lane = item["lane"]; lane_arrays = [raw, values["carrier_signed"], values[lane], pooled[lane]]; lane_titles = ["Raw", "carrier_signed", lane, f"{lane} pooled rank map"]; _encode(model / f"videos/closeups/{item['id']}.mp4", _close_frames(lane_arrays, lane_titles, item, ORANGE), (128 * 4, 164)); _trace(model / f"figures/traces/{item['id']}.png", lane_arrays, lane_titles, item, "#dd6b20"); atomic_json(model / f"metadata/roi_{item['id']}.json", {"model_roi": item["id"], "lane": lane, "x_px": item["x"], "y_px": item["y"], "intervals": item["intervals"]})
    compact_proposals = [row for row in proposals if row["lane"] in COMPACT_LANES]
    _csv(expert / "expert_occurrences.csv", labels); _csv(model / "model_occurrences.csv", compact_proposals)
    comparisons = _comparisons(partial, labels, compact_proposals, raw, values, maps, float(config.evaluation["match_radius_px"]))
    matched = sum(row["within_match_radius"] for row in comparisons)
    atomic_text(expert / "README.md", "# Expert Annotations\n\nGreen expert-positive markers only, derived from the immutable original-workbook v1 cohort.\n")
    atomic_text(model / "README.md", "# Model Annotations\n\nOrange model-only markers at the frozen B20, NMS=6 operating point. Carrier, coherence, and lag are shown sequentially; candidates not matched to sparse-positive labels remain unknown.\n")
    summary = {"schema_version": 1, "status": "complete", "estimand": "visual reality check of frozen B20 compact candidates against 79 immutable known-positive occurrences", "expert_roi_identities": len(unique), "expert_occurrences": len(labels), "model_roi_identities": len(models), "model_occurrences": len(compact_proposals), "known_positive_matches_nearest_compact_union": matched, "known_positive_total": len(labels), "precision": "not identified", "unmatched_candidates": "unknown_not_negative", "operating_point": {"budget_per_burst_per_lane": BUDGET, "nms_distance_px": 6, "match_radius_px": 6}, "lanes": list(LANES)}
    atomic_json(partial / "summary.json", summary)
    atomic_json(partial / "llm_context.json", {"annotation_separation": "strict", "comparison_spatial_panels": ["Raw matched comparison", "MSICA + MSLN matched comparison"], "comparison_panel_semantic_alias": {"MSICA + MSLN matched comparison": "maximum of coherence_w15 and propagation_lag2_w15 frozen pooled maps"}, "model_stage_sequence": model_titles, "coordinate_convention": "x=column,y=row", "frame_convention": "UI one-based inclusive", "counts": summary, "primary_tables": ["3_Comparison/expert_model_matches.csv", "3_Comparison/nearest_roi_trace_metrics.csv"], "representative_artifacts": ["2_Model_Annotations/videos/model_annotations_sequential_full_field.mp4", "3_Comparison/burst_1_comparison.png", "3_Comparison/aggregate_diagnostics.png"]})
    atomic_text(partial / "REPORT.md", f"# Frozen compact-detector visual audit\n\nThis three-section packet audits the exact B20/NMS=6 operating point on the protected v1 sparse-positive cohort. It contains {len(unique)} expert identities and {len(labels)} expert occurrences, plus {len(models)} lane-specific consolidated compact-model locations from {len(compact_proposals)} burst-level proposals. The nearest compact-lane candidate lies within 6 px for {matched}/{len(labels)} known-positive occurrences. This is a visual and known-positive-recall audit only: the source labels were not an exhaustive field census, so precision, specificity, and false-positive rate remain unidentified.\n")
    atomic_json(partial / "validation.json", {"status": "pending_media_validation", "annotation_separation": "independent_render", "comparison_videos": 0, "source_hashes": {"raw": _sha(raw_path), "labels": _sha(label_path)}, "feature_metadata": feature_meta})
    atomic_json(partial / "artifact_index.json", {"status": "building"})
    inventory = require_three_section_scientific_audit(partial, expected_expert_roi_count=len(unique), expected_model_roi_count=len(models), expected_expert_occurrence_count=len(labels))
    artifacts = [{"path": str(path.relative_to(partial)), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in sorted(partial.rglob("*")) if path.is_file()]
    atomic_json(partial / "artifact_index.json", {"artifacts": artifacts})
    atomic_json(partial / "status.json", {"status": "complete_pending_visual_inspection", "inventory": inventory.to_dict()})
    partial.replace(output)
    return {"summary": summary, "inventory": inventory.to_dict(), "output": str(output)}


def finalize_visual_inspection(output: Path, *, passed: bool, inspection_note: str) -> dict[str, Any]:
    """Record bounded human/agent visual QA after all media have been rendered."""
    output = output.resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    inventory = require_three_section_scientific_audit(
        output,
        expected_expert_roi_count=int(summary["expert_roi_identities"]),
        expected_model_roi_count=int(summary["model_roi_identities"]),
        expected_expert_occurrence_count=int(summary["expert_occurrences"]),
    )
    videos = sorted(output.rglob("*.mp4")); video_failures = []
    for path in videos:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_frames", "-of", "json", str(path)], capture_output=True, text=True)
        if result.returncode or not json.loads(result.stdout).get("streams"): video_failures.append(str(path.relative_to(output)))
    image_failures = []
    for path in sorted(output.rglob("*.png")):
        try:
            with Image.open(path) as image: image.verify()
        except Exception: image_failures.append(str(path.relative_to(output)))
    validation = {"status": "passed" if passed and not video_failures and not image_failures else "failed", "inventory_complete": inventory.complete, "video_count": len(videos), "video_decode_failures": video_failures, "png_count": len(list(output.rglob("*.png"))), "png_decode_failures": image_failures, "visual_inspection": {"performed": True, "passed": bool(passed), "note": inspection_note, "checks": ["expert/model marker separation", "grayscale evidence backgrounds", "synchronized processed stages", "exact pooled ranking maps", "representative full-field and trace legibility"]}, "interpretation_guard": "known-positive visual audit only; unmatched candidates remain unknown"}
    atomic_json(output / "validation.json", validation)
    atomic_json(output / "status.json", {"status": "complete" if validation["status"] == "passed" else "failed_visual_validation", "inventory": {**inventory.to_dict(), "root": str(output)}, "validation": validation["status"]})
    artifacts = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_index.json"]
    atomic_json(output / "artifact_index.json", {"artifacts": artifacts})
    return validation
