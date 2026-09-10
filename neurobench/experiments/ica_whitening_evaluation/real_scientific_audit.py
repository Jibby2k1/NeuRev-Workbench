"""Three-section scientific visual audit for confirmed ICA finalists."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import correlate

from neurobench.experiments.frame_difference import _atomic_json
from neurobench.experiments.learned_operator_selection.data import load_and_validate_labels
from neurobench.reports.scientific_audit import require_three_section_scientific_audit

from .config import ICAWhiteningConfig
from .operators import apply_whitening
from .real_config import RealDataConfig
from .real_finalist_confirmation import _completed_factorial_rows
from .real_runner import _load_proposals, _signed_review


GREEN = (70, 220, 125)
ORANGE = (255, 145, 35)
COMPARISON_PANELS = ["Raw matched comparison", "ICA matched comparison"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode(path: Path, frames: Iterable[np.ndarray], size: tuple[int, int], fps: float) -> None:
    if path.is_file(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames: process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close(); error = process.stderr.read().decode(); code = process.wait()
    except BaseException:
        process.kill(); raise
    if code: raise RuntimeError(error[-2000:])
    partial.replace(path)


def _limits(values: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(values[::17, ::5, ::5], dtype=np.float32)
    low, high = np.percentile(sample, [1, 99.7])
    return float(low), max(float(high), float(low) + 1e-7)


def _gray(frame: np.ndarray, limits: tuple[float, float], size: tuple[int, int]) -> Image.Image:
    low, high = limits
    scaled = np.clip((np.asarray(frame, np.float32) - low) / (high - low), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").resize(
        size, Image.Resampling.BILINEAR).convert("RGB")


def _component_evidence_movie(
    whitened: np.ndarray, specification: dict[str, Any], model: dict[str, Any],
    quiet_frames: int,
) -> np.ndarray:
    family = specification["family"]
    sw = int(specification.get("spatial_width_px") or 1)
    tw = int(specification.get("temporal_width_frames") or 1)
    demixing = np.asarray(model["demixing"], dtype=np.float64)
    mean = np.asarray(model["internal_whitening"]["mean"] if "mean" in model["internal_whitening"]
                      else model.get("observation_mean", []), dtype=np.float64)
    # model_summary stores the mean in the full model's internal whitening only
    # in newer artifacts; legacy S4 artifacts expose it through model.whitening.
    if mean.size == 0:
        mean = np.asarray(model["whitening_mean"], dtype=np.float64)
    result = np.zeros_like(whitened, dtype=np.float32)
    if family in {"temporal", "joint_spatiotemporal"}:
        before, after = ((tw // 2, tw // 2) if specification["causality"] == "centered"
                         else (tw - 1, 0))
        padded = np.pad(whitened, ((before, after), (0, 0), (0, 0)), mode="reflect")
    for component, flat_kernel in enumerate(demixing):
        offset = float(flat_kernel @ mean)
        if family == "spatial":
            values = correlate(whitened, flat_kernel.reshape(1, sw, sw), mode="mirror") - offset
        elif family == "temporal":
            values = np.zeros_like(whitened, dtype=np.float32)
            for lag, weight in enumerate(flat_kernel):
                values += float(weight) * padded[lag:lag + len(whitened)]
            values -= offset
        else:
            kernel = flat_kernel.reshape(tw, sw, sw)
            values = np.zeros_like(whitened, dtype=np.float32)
            for lag in range(tw):
                values += correlate(padded[lag:lag + len(whitened)],
                                    kernel[lag][None], mode="mirror")
            values -= offset
        quiet = values[:quiet_frames]
        center = float(np.median(quiet)); scale = max(
            float(1.4826 * np.median(np.abs(quiet - center))), np.finfo(float).eps
        )
        np.maximum(result, np.abs((values - center) / scale), out=result)
    return result


def _panel_frames(
    arrays: list[np.ndarray], titles: list[str], marker_rows: list[list[dict[str, Any]]],
    colors: list[tuple[int, int, int]], *, start: int = 0, stop: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> Iterable[np.ndarray]:
    panel, height, header = 300, 190, 40
    stop = len(arrays[0]) if stop is None else stop
    limits = [_limits(array) for array in arrays]; font = ImageFont.load_default()
    for frame_index in range(start, stop):
        canvas = Image.new("RGB", (panel * len(arrays), height + header), "black")
        draw = ImageDraw.Draw(canvas); draw.text((6, 5), f"review frame {frame_index}", fill="white", font=font)
        for column, (array, title, limit) in enumerate(zip(arrays, titles, limits, strict=True)):
            frame = array[frame_index]
            if crop is not None: frame = frame[crop[1]:crop[3], crop[0]:crop[2]]
            canvas.paste(_gray(frame, limit, (panel, height)), (column * panel, header))
            draw.text((column * panel + 5, 22), title, fill="white", font=font)
            for rows, color in zip(marker_rows, colors, strict=True):
                for item in rows:
                    if not int(item.get("_start", 0)) <= frame_index < int(item.get("_stop", len(arrays[0]))):
                        continue
                    x, y = float(item["x_px"]), float(item["y_px"])
                    if crop is not None: x, y = x - crop[0], y - crop[1]
                    width = (crop[2] - crop[0]) if crop else arrays[0].shape[2]
                    full_height = (crop[3] - crop[1]) if crop else arrays[0].shape[1]
                    if not (0 <= x < width and 0 <= y < full_height): continue
                    cx = column * panel + x * panel / width; cy = header + y * height / full_height
                    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=color, width=2)
        yield np.asarray(canvas, dtype=np.uint8)


def _trace(path: Path, arrays: list[np.ndarray], titles: list[str], item: dict[str, Any]) -> None:
    if path.is_file(): return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x, y = int(round(float(item["x_px"]))), int(round(float(item["y_px"])))
    fig, axes = plt.subplots(len(arrays), 1, figsize=(8, 1.9 * len(arrays)), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, array, title in zip(axes, arrays, titles, strict=True):
        axis.plot(array[:, y, x], lw=.8); axis.set_ylabel(title); axis.grid(alpha=.2)
    axes[-1].set_xlabel("review-relative frame"); fig.suptitle(str(item["id"]))
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader(); writer.writerows(rows)


def _run_single_scientific_audit(
    real: RealDataConfig, *, preflight_dir: str | Path,
    selected_fit_id: str, audit_root: str | Path,
) -> dict[str, Any]:
    preflight = Path(preflight_dir).expanduser().resolve()
    s4 = real.output_dir / "stages" / "S4_FINALIST_CONFIRMATION"
    if json.loads((s4 / "validation.json").read_text()).get("status") != "pass":
        raise RuntimeError("validated S4 confirmation is required")
    s4_summary = json.loads((s4 / "summary.json").read_text())
    selected = str(selected_fit_id)
    if selected not in set(map(str, s4_summary["selected_fit_ids"])):
        raise RuntimeError("scientific-audit fit is not in the frozen finalist set")
    eligible_bursts = list(map(int, s4_summary["held_out_bursts_by_fit_id"][selected]))
    rows = _completed_factorial_rows(real); specification = rows[selected]
    original_seed = int(specification["seed"])
    parent = ICAWhiteningConfig.from_json(real.parent_config)
    source = np.load(parent.source_video, mmap_mode="r", allow_pickle=False)
    movie = _signed_review(source, parent)
    labels = load_and_validate_labels(parent.labels_tsv, parent.label_summary, tuple(source.shape[1:]))
    proposals = _load_proposals(preflight / "common_proposal_universe.tsv")
    intervals_raw = json.loads((preflight / "event_interval_contract.json").read_text())["intervals"]
    intervals = {int(key): (int(value[0]), int(value[1])) for key, value in intervals_raw.items()}
    quiet = parent.frames.quiet_end_ui - parent.frames.review_start_ui + 1
    whitened = apply_whitening(movie, quiet, specification,
                               maximum_samples=real.fitting.maximum_fit_samples).output
    evidence = np.zeros_like(movie, dtype=np.float32)
    ranked_candidates = []
    for burst in eligible_bursts:
        refit = json.loads((s4 / "refits"
                           / f"{selected}__seed_{original_seed}__heldout_{burst}.json").read_text())
        component = _component_evidence_movie(whitened, specification, refit["model"], quiet)
        start, stop = intervals[burst]; evidence[start:stop] = component[start:stop]
        ordered_ids = refit["top_200_candidate_ids_by_burst"][str(burst)][:58]
        rank_by_id = {candidate_id: rank for rank, candidate_id in enumerate(ordered_ids, 1)}
        ranked_candidates.extend([{**row, "id": row["candidate_id"],
                                   "audit_rank_within_burst": rank_by_id[row["candidate_id"]],
                                   "_start": start, "_stop": stop}
                                  for row in proposals if row["candidate_id"] in rank_by_id])
    eligible_labels = [row for row in labels if int(row["burst_id"]) in eligible_bursts]
    expert_rows = [{**row, "id": str(row.get("observation_id", f"expert_{index:03d}")),
                    "_start": intervals[int(row["burst_id"])][0],
                    "_stop": intervals[int(row["burst_id"])][1]}
                   for index, row in enumerate(eligible_labels)]
    # Consolidate repeated candidate coordinates into model identities.
    model_rows = []
    for candidate in ranked_candidates:
        existing = next((item for item in model_rows if
                         (item["x_px"] - candidate["x_px"]) ** 2
                         + (item["y_px"] - candidate["y_px"]) ** 2 <= 9), None)
        if existing is None:
            existing = {**candidate, "id": f"model_roi_{len(model_rows)+1:03d}",
                        "bursts": [], "best_audit_rank": candidate["audit_rank_within_burst"]}
            model_rows.append(existing)
        existing["bursts"].append(int(candidate["burst_id"]))
        existing["best_audit_rank"] = min(
            int(existing["best_audit_rank"]), int(candidate["audit_rank_within_burst"])
        )
    model_identity_count_before_closeup_sampling = len(model_rows)
    model_rows = sorted(model_rows, key=lambda item: (
        -len(set(item["bursts"])), int(item["best_audit_rank"]), item["id"],
    ))[:24]
    root = Path(audit_root).expanduser().resolve()
    expert = root / "1_Expert_Annotations"; model = root / "2_Model_Annotations"
    comparison = root / "3_Comparison"
    for path in (expert / "videos/closeups", expert / "figures/traces", expert / "metadata",
                 model / "videos/closeups", model / "figures/traces", model / "metadata",
                 comparison / "trace_comparisons"):
        path.mkdir(parents=True, exist_ok=True)
    fps = 1000.0 / parent.frames.frame_period_ms
    _encode(expert / "videos/expert_annotations_full_field.mp4",
            _panel_frames([movie], ["Raw expert-only"], [expert_rows], [GREEN]),
            (300, 230), fps)
    _encode(model / "videos/model_annotations_sequential_full_field.mp4",
            _panel_frames([movie, whitened, evidence], ["Raw", "External whitening", "ICA evidence"],
                          [ranked_candidates], [ORANGE]), (900, 230), fps)
    for item in expert_rows:
        x, y = int(item["x_px"]), int(item["y_px"]); crop = (
            max(0, x - 24), max(0, y - 24), min(movie.shape[2], x + 25), min(movie.shape[1], y + 25)
        )
        burst = int(item["burst_id"]); start, stop = intervals[burst]
        _encode(expert / f"videos/closeups/{item['id']}.mp4",
                _panel_frames([movie], ["Raw expert-only"], [[item]], [GREEN],
                              start=start, stop=stop, crop=crop), (300, 230), fps)
        _trace(expert / f"figures/traces/{item['id']}.png", [movie], ["Raw"], item)
        _atomic_json(expert / f"metadata/roi_{item['id']}.json", item)
    for item in model_rows:
        x, y = int(item["x_px"]), int(item["y_px"]); crop = (
            max(0, x - 24), max(0, y - 24), min(movie.shape[2], x + 25), min(movie.shape[1], y + 25)
        )
        start = min(intervals[burst][0] for burst in item["bursts"])
        stop = max(intervals[burst][1] for burst in item["bursts"])
        _encode(model / f"videos/closeups/{item['id']}.mp4",
                _panel_frames([movie, whitened, evidence], ["Raw", "Whitening", "ICA"],
                              [[item]], [ORANGE], start=start, stop=stop, crop=crop),
                (900, 230), fps)
        _trace(model / f"figures/traces/{item['id']}.png",
               [movie, whitened, evidence], ["Raw", "Whitening", "ICA"], item)
        _atomic_json(model / f"metadata/roi_{item['id']}.json", item)
    comparison_rows = []
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for item in expert_rows:
        burst = int(item["burst_id"]); candidates = [row for row in ranked_candidates
                                                     if int(row["burst_id"]) == burst]
        nearest = min(candidates, key=lambda row: (row["x_px"] - item["x_px"]) ** 2
                      + (row["y_px"] - item["y_px"]) ** 2)
        distance = float(np.hypot(nearest["x_px"] - item["x_px"], nearest["y_px"] - item["y_px"]))
        start, stop = intervals[burst]; frames = np.arange(start, stop)
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
        axes[0].plot(frames, movie[start:stop, int(item["y_px"]), int(item["x_px"])], color="#46dc7d")
        axes[0].set_ylabel("Raw expert pixel")
        axes[1].plot(frames, evidence[start:stop, int(nearest["y_px"]), int(nearest["x_px"])], color="#ff9123")
        axes[1].set_ylabel("ICA candidate"); axes[1].set_xlabel("review-relative frame")
        fig.suptitle(f"{item['id']} nearest frozen finalist candidate: {distance:.2f} px")
        fig.savefig(comparison / f"trace_comparisons/{item['id']}.png", dpi=110); plt.close(fig)
        comparison_rows.append({"expert_id": item["id"], "burst_id": burst,
                                "candidate_id": nearest["candidate_id"], "distance_px": distance,
                                "within_6px": distance <= 6})
    _csv(comparison / "expert_model_matches.csv", comparison_rows)
    _csv(expert / "expert_occurrences.csv", expert_rows)
    _csv(model / "model_occurrences.csv", ranked_candidates)
    summary = {
        "schema_version": 1, "status": "rendered_pending_visual_inspection",
        "source_fit_id": selected, "expert_roi_count": len(expert_rows),
        "expert_occurrence_count": len(expert_rows), "model_roi_count": len(model_rows),
        "model_occurrence_count": len(ranked_candidates),
        "model_identity_count_before_closeup_sampling": model_identity_count_before_closeup_sampling,
        "model_closeup_limit": 24,
        "model_closeup_sampling_rule": "repeated_across_bursts_then_best_rank_then_id",
        "known_positive_within_6px": sum(row["within_6px"] for row in comparison_rows),
        "precision": "not_identified", "unmatched_candidates": "unknown_not_negative",
        "protected_held_out_bursts": eligible_bursts,
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "llm_context.json", {
        "annotation_separation": "strict", "comparison_spatial_panels": COMPARISON_PANELS,
        "model_stage_sequence": ["Raw", "External whitening", "ICA evidence"],
        "coordinate_convention": "x=column,y=row", "counts": summary,
    })
    (root / "REPORT.md").write_text(
        "# ICA finalist scientific audit\n\nThree strictly separated sections show expert-only Raw evidence, "
        "model-only Raw/whitening/ICA evidence, and figure-only matched comparisons. Unmatched candidates "
        "remain unknown; this packet does not identify precision or biological source identity.\n",
        encoding="utf-8",
    )
    _atomic_json(root / "validation.json", {"status": "pending_visual_inspection"})
    _atomic_json(root / "artifact_index.json", {"status": "building"})
    inventory = require_three_section_scientific_audit(
        root, expected_expert_roi_count=len(expert_rows), expected_model_roi_count=len(model_rows),
        expected_expert_occurrence_count=len(expert_rows),
        expected_comparison_panels=COMPARISON_PANELS,
    )
    artifacts = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                  "sha256": _sha(path)} for path in sorted(root.rglob("*")) if path.is_file()]
    _atomic_json(root / "artifact_index.json", {"artifacts": artifacts})
    _atomic_json(root / "inventory.json", inventory.to_dict())
    return {"summary": summary, "inventory": inventory.to_dict()}


def run_scientific_audit(real: RealDataConfig, *, preflight_dir: str | Path) -> dict[str, Any]:
    """Render an independently valid three-section package for every frozen finalist."""
    s4 = real.output_dir / "stages" / "S4_FINALIST_CONFIRMATION"
    if json.loads((s4 / "validation.json").read_text()).get("status") != "pass":
        raise RuntimeError("validated S4 confirmation is required")
    selected = list(map(str, json.loads((s4 / "summary.json").read_text())["selected_fit_ids"]))
    root = real.output_dir / "stages" / "S5_SCIENTIFIC_AUDIT"
    root.mkdir(parents=True, exist_ok=True)
    packages = {}
    for fit_id in selected:
        package_root = root / "finalists" / fit_id
        packages[fit_id] = _run_single_scientific_audit(
            real, preflight_dir=preflight_dir, selected_fit_id=fit_id,
            audit_root=package_root,
        )
    summary = {
        "schema_version": 2, "status": "rendered_pending_visual_inspection",
        "finalist_count": len(selected), "selected_fit_ids": selected,
        "package_status": {fit_id: value["summary"]["status"]
                           for fit_id, value in packages.items()},
        "promotion_status": "pending_visual_inspection_for_every_finalist",
        "unmatched_candidates": "unknown_not_negative",
    }
    inventory = {
        "schema_version": 2, "status": "complete_pending_visual_inspection",
        "finalist_count": len(selected),
        "packages": {fit_id: value["inventory"] for fit_id, value in packages.items()},
    }
    _atomic_json(root / "summary.json", summary)
    _atomic_json(root / "inventory.json", inventory)
    _atomic_json(root / "validation.json", {
        "status": "pending_visual_inspection", "required_finalists": selected,
        "validated_finalists": [],
    })
    artifacts = [{
        "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
        "sha256": _sha(path),
    } for path in sorted((root / "finalists").rglob("*")) if path.is_file()]
    _atomic_json(root / "artifact_index.json", {"artifacts": artifacts})
    return {"summary": summary, "inventory": inventory}


def validate_all_scientific_audit_media(
    root: str | Path, *, visual_results: dict[str, tuple[bool, str]],
) -> dict[str, Any]:
    """Require explicit visual inspection and decode validation for every finalist."""
    target = Path(root).resolve()
    summary = json.loads((target / "summary.json").read_text())
    required = list(map(str, summary["selected_fit_ids"]))
    if set(visual_results) != set(required):
        raise ValueError("visual results must cover the exact frozen finalist set")
    results = {}
    for fit_id in required:
        passed, note = visual_results[fit_id]
        results[fit_id] = validate_scientific_audit_media(
            target / "finalists" / fit_id, visual_passed=passed,
            inspection_note=note,
        )
    passed = all(item["status"] == "pass" for item in results.values())
    validation = {
        "status": "pass" if passed else "fail", "required_finalists": required,
        "validated_finalists": [fit_id for fit_id, item in results.items()
                                if item["status"] == "pass"],
        "finalist_results": results,
        "promotion_allowed_by_audit": passed,
    }
    _atomic_json(target / "validation.json", validation)
    return validation


def validate_scientific_audit_media(root: str | Path, *, visual_passed: bool,
                                    inspection_note: str) -> dict[str, Any]:
    target = Path(root).resolve(); summary = json.loads((target / "summary.json").read_text())
    inventory = require_three_section_scientific_audit(
        target, expected_expert_roi_count=int(summary["expert_roi_count"]),
        expected_model_roi_count=int(summary["model_roi_count"]),
        expected_expert_occurrence_count=int(summary["expert_occurrence_count"]),
        expected_comparison_panels=COMPARISON_PANELS,
    )
    video_failures = []
    for path in target.rglob("*.mp4"):
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                                capture_output=True)
        if result.returncode: video_failures.append(str(path.relative_to(target)))
    image_failures = []
    for path in target.rglob("*.png"):
        try:
            with Image.open(path) as image: image.verify()
        except Exception: image_failures.append(str(path.relative_to(target)))
    passed = bool(visual_passed and inventory.complete and not video_failures and not image_failures)
    validation = {
        "status": "pass" if passed else "fail", "inventory_complete": inventory.complete,
        "visual_inspection": {"performed": True, "passed": bool(visual_passed),
                              "note": inspection_note},
        "video_decode_failures": video_failures, "png_decode_failures": image_failures,
        "promotion_allowed_by_audit": passed,
    }
    _atomic_json(target / "validation.json", validation)
    return validation
