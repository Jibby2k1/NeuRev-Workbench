"""Versioned audit adapter with spatially varying frozen setup cutoffs.

Copied from the immutable spatiotemporal renderer; original-score ranking is
preserved and this renderer never re-runs NMS. Each ROI trace shows its own
cutoff. Comparison traces label the primary expert cutoff explicitly.

Actual stage labels and the supplied final Score array are preserved. This
adapter reuses the frozen two-stencil inventory, geometry, exact lossless-video
writer, and source-bound checkpoints, but does not refit or rerun a detector.
Numerical score arrays must cover the complete consecutive source UI interval;
Other thresholds remain linked numerical evidence rather than a media claim.
Synthetic truth and sparse real annotations have distinct, explicit captions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import two_stencil_audit as _base
from .two_stencil_audit import (
    GREEN, ORANGE, PANEL_FIT_SEMANTICS, VIDEO_ENCODING,
    _Progress, _artifact_cached, _canonical_digest, _checkpoint_artifact,
    _correlation, _digest, _int, _json, _lag, _load_deps, _panel,
    _write_video, context_source_frames, identity_crop,
    panel_image_geometry, verify_completed_audit,
)

SOURCE_TIME_SEMANTICS = "source UI one-based inclusive; synthetic truth or sparse real windows as explicitly declared"
COMPARISON_PANELS = ["Raw matched comparison", "Score matched comparison"]
_TRACE_CANVASES: dict[tuple[str, ...], Any] = {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)) and Path(value).suffix.lower() == ".json":
        rows = json.loads(Path(value).read_text())
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError("Candidate/truth JSON must contain an exact list of row objects")
        return rows
    return _base._rows(value)


def audit_inventory_plan(candidate_rows: Any, expert_occurrences: Any, source_frames_ui: Sequence[int], *,
                         context_frames: int = 5, fullfield_step: int = 5) -> dict[str, Any]:
    """Retain the inherited inventory for labeled cases and explicit empty truth."""
    candidates,experts = _rows(candidate_rows),_rows(expert_occurrences)
    if experts:
        return _base.audit_inventory_plan(candidates,experts,source_frames_ui,context_frames=context_frames,fullfield_step=fullfield_step)
    frames = [_int(value,"source_frame_ui") for value in source_frames_ui]
    if not frames or frames[0] < 1 or any(b != a+1 for a,b in zip(frames,frames[1:])):
        raise ValueError("The full-duration source frame list must be consecutive")
    if fullfield_step < 1 or context_frames < 0:raise ValueError("Invalid video step or context")
    if any(_int(row["source_frame_ui"],"source_frame_ui") not in set(frames) for row in candidates):
        raise ValueError("Candidate frame is outside the audited source interval")
    sites,assignments = _base.spatial_review_sites(candidates)
    shown = frames[::fullfield_step]
    if shown[-1] != frames[-1]:shown.append(frames[-1])
    clips = {site["model_roi_id"]:context_source_frames([(f,f) for f in site["member_source_frames_ui"]],frames,context_frames) for site in sites}
    return {"expert_roi_count":0,"expert_occurrence_count":0,"expert_applicability":"N/A: no source/event truth rows declared",
        "model_roi_count":len(sites),"all_model_proposal_count":len(candidates),"source_frame_count":len(frames),
        "fullfield_source_frames_ui":shown,"fullfield_video_frame_count_each":len(shown),"expected_video_count":2+len(sites),
        "expected_trace_figure_count":len(sites),"closeup_video_frame_count_total":sum(map(len,clips.values())),
        "context_frames":context_frames,"fullfield_step":fullfield_step,"unique_experts":{},"model_sites":sites,
        "model_assignments":assignments,"expert_closeup_source_frames_ui":{},"model_closeup_source_frames_ui":clips,
        "no_model_anchor_sampling":True,"all_occurrences_covered":True}


def _truth_semantics(source_binding: Mapping[str, Any], experts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mode = source_binding.get("truth_mode")
    if mode not in {"fully_synthetic","sparse_real"}:
        raise ValueError("source_binding.truth_mode must be fully_synthetic or sparse_real")
    synthetic = mode == "fully_synthetic"
    return {"truth_mode":mode,"expert_applicable":bool(experts),
        "expert_label":"Synthetic source truth" if synthetic and experts else "Expert N/A: no declared source" if not experts else "Sparse real annotations",
        "comparison_scope":"descriptive window/site trace correspondence, not the primary framewise or event-assignment metric",
        "window_semantics":"simulated fluorescence event extent" if synthetic else "configured burst window, not per-ROI onset",
        "unmatched_interpretation":"synthetic truth classification belongs to linked primary metrics; this review grouping is not a TP/FP rule" if synthetic else "unknown_not_negative",
        "onset_semantics":"simulated onset exists in the sealed truth; trace-correlation lag is not detection delay" if synthetic else "no per-ROI onset truth",
        "real_precision_identified":False}


def _descriptive_evaluation(candidates: Sequence[Mapping[str, Any]], experts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare within each declared window, permitting overlapping simulations.

    A proposal may occur in multiple overlapping comparison windows. The joins
    are review diagnostics, never the primary synthetic TP/FP/event assignments.
    """
    from .two_stencil_evaluation import evaluate_occurrence_windows
    intervals = _burst_intervals(experts)
    result = {key:[] for key in ("occurrence_rows","site_rows","membership_rows","burst_summaries")}
    for burst,extent in sorted(intervals.items()):
        subset = [row for row in candidates if extent[0] <= int(row["source_frame_ui"]) <= extent[1]]
        positive = [row for row in experts if int(row["burst_id"]) == burst]
        current = evaluate_occurrence_windows(subset,positive,burst_intervals_ui={burst:extent})
        for key in result:result[key].extend(current[key])
    matched = sum(bool(row["matched"]) for row in result["occurrence_rows"])
    result["summary"] = {"emitted_frame_proposal_count":len(candidates),"declared_truth_occurrence_count":len(experts),
        "descriptive_window_site_matched_occurrence_count":matched,
        "descriptive_window_site_coverage":matched/len(experts) if experts else None,
        "primary_classification_metric":False,"precision_computed":False,"event_delay_computed":False,
        "scope":"separate window-local trace comparisons; overlapping windows may reuse proposals; primary metrics remain in numeric_threshold_paths"}
    return result


def _captioned_panel(*args: Any, caption: str, **kwargs: Any) -> Any:
    """Append a grayscale truth caption without changing source-image geometry."""
    pixels = _panel(*args,**kwargs)
    np,Image,ImageDraw,_ = _load_deps()
    canvas = Image.new("RGB",(pixels.shape[1],pixels.shape[0]+18),"black")
    canvas.paste(Image.fromarray(pixels),(0,0))
    ImageDraw.Draw(canvas).text((4,pixels.shape[0]+3),caption,fill="white")
    return np.asarray(canvas)


def _validate_video_geometry(video: Mapping[str, Any], *, stage_count: int, source_fps: float,
                             fullfield_step: int) -> None:
    """Check actual encoded stream geometry/timing, separately from RGB fidelity."""
    closeup = "closeups" in Path(video["path"]).parts
    panel_width,panel_height = (144,112) if closeup else (180,136)
    expected_wh = (4*panel_width,math.ceil(stage_count/4)*(panel_height+30)+18)
    probe = video["probe"]
    if (int(probe["width"]),int(probe["height"])) != expected_wh:
        raise ValueError("Encoded video canvas differs from stage-panel/footer geometry")
    frames = [_int(value,"source_frame_ui") for value in video["source_frames_ui"]]
    if not frames or frames[0] < 1 or any(b <= a for a,b in zip(frames,frames[1:])):
        raise ValueError("Video source-frame map must be nonempty and strictly increasing")
    expected_fps = Fraction(str(source_fps))/(1 if closeup else fullfield_step)
    if Fraction(str(probe["r_frame_rate"])) != expected_fps:
        raise ValueError("Encoded video frame rate differs from the declared source sampling")
    if int(probe["nb_frames"]) != len(frames):
        raise ValueError("Encoded video frame count differs from its source-frame map")
    expected_duration = len(frames)/float(expected_fps)
    if not math.isclose(float(probe["duration"]),expected_duration,rel_tol=0,abs_tol=1e-5):
        raise ValueError("Encoded video duration differs from frame count divided by playback FPS")


def _table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Commit tables atomically, just like the inherited JSON checkpoints."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    _base._table(temporary, rows)
    temporary.replace(path)


def _save_figure(figure: Any, path: Path, *, dpi: int) -> None:
    temporary = path.with_suffix(".partial.png")
    figure.savefig(temporary, dpi=dpi)
    temporary.replace(path)


def _threshold_at(threshold, x, y):
    if not isinstance(threshold, dict):
        return float(threshold)
    matches = [r for r in threshold["regions"]
               if r["box_yxyx"][1] <= x < r["box_yxyx"][3]
               and r["box_yxyx"][0] <= y < r["box_yxyx"][2]]
    if len(matches) != 1:
        raise ValueError("Trace/candidate must belong to exactly one calibration region")
    return float(matches[0]["threshold"])


def _frozen_threshold(candidates, operating_point):
    if operating_point.get("regional_thresholds") is not None:
        value = {"regions": operating_point["regional_thresholds"]}
        if not value["regions"] or any(r["threshold"] is None or not math.isfinite(float(r["threshold"])) for r in value["regions"]):
            raise ValueError("Every region needs a finite frozen threshold")
    else:
        value = operating_point.get("threshold", operating_point.get("threshold_z"))
        if value is None or not math.isfinite(float(value)):
            raise ValueError("A finite frozen threshold is required")
        value = float(value)
    for row in candidates:
        expected = _threshold_at(value, row["x_px"], row["y_px"])
        for key in ("threshold", "threshold_z"):
            if row.get(key) not in (None, "") and float(row[key]) != expected:
                raise ValueError("Candidate threshold differs from its frozen regional cutoff")
    return value


def _threshold_description(threshold):
    return "regional cutoffs (run_contract.json: operating_point)" if isinstance(threshold, dict) else f"{threshold:g}"


def _burst_intervals(experts: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, int]]:
    intervals: dict[int, tuple[int, int]] = {}
    for row in experts:
        burst = _int(row["burst_id"], "burst_id")
        extent = (_int(row["source_start_ui"], "source_start_ui"), _int(row["source_stop_ui"], "source_stop_ui"))
        if burst in intervals and intervals[burst] != extent:
            raise ValueError("Occurrences in one burst must share the declared inclusive interval")
        intervals[burst] = extent
    return intervals


def _verify_candidate_score_binding(candidates: Sequence[Mapping[str, Any]], score: Any,
                                    frame_index: Mapping[int, int], threshold: float,
                                    operating_point: Mapping[str, Any]) -> None:
    start = _int(operating_point.get("application_source_start_ui", min(frame_index)), "application_source_start_ui")
    stop = _int(operating_point.get("application_source_stop_ui", max(frame_index)), "application_source_stop_ui")
    if start > stop or start not in frame_index or stop not in frame_index:
        raise ValueError("Application interval must be contained in the audited source interval")
    for row in candidates:
        frame = _int(row["source_frame_ui"], "source_frame_ui")
        x, y = _int(row["x_px"], "x_px"), _int(row["y_px"], "y_px")
        if not start <= frame <= stop:
            raise ValueError("Candidate lies outside the declared application interval")
        saved = float(score[frame_index[frame], y, x])
        if not math.isfinite(saved) or float(row["score"]) != saved:
            raise ValueError("Candidate score differs from the supplied Score array")
        if not saved > _threshold_at(threshold, x, y):
            raise ValueError("Emitted candidate does not exceed the frozen score threshold")


def _trace_plot(path: Path, stage_traces: Mapping[str, Any], frames: Sequence[int], *, title: str, spans: Sequence[Sequence[int]], color: str, second: Mapping[str, Any] | None = None, assigned: Mapping[str, Any] | None = None, threshold: float | None = None) -> None:
    """Reuse fixed axes/lines; all samples and the original dimensions/DPI remain."""
    _, _, _, plt = _load_deps()
    keys = tuple(stage_traces)
    if keys not in _TRACE_CANVASES:
        fig, axes = plt.subplots(len(keys), 1, figsize=(9, 1.45 * len(keys)), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.095, right=.985, bottom=.12, top=.88, hspace=.19)
        lines = []
        for name, axis in zip(keys, axes[:, 0]):
            primary, = axis.plot([], [], linewidth=.7)
            neighbor, = axis.plot([], [], color="#ff9123", linewidth=.7)
            assignment, = axis.plot([], [], color="#8b7d43", linewidth=.7, linestyle=":")
            cutoff, = axis.plot([], [], color="black", linewidth=.7, linestyle="--")
            lines.append((primary, neighbor, assignment, cutoff))
            axis.set_ylabel(name)
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=8)
        axes[-1, 0].set_xlabel("source frame UI (one-based); all displayed samples; trace lag is not detection delay", fontsize=9)
        title_artist = fig.suptitle("", fontsize=9, y=.98)
        _TRACE_CANVASES[keys] = (fig, axes[:, 0], lines, title_artist)
    fig, axes, lines, title_artist = _TRACE_CANVASES[keys]
    for (name, values), axis, (primary, neighbor, assignment, cutoff) in zip(stage_traces.items(), axes, lines):
        primary.set_data(frames, values)
        primary.set_color(color)
        primary.set_label("truth/annotation center" if second is not None else "exact pixel")
        for line, data, label in ((neighbor, second, "nearest site"), (assignment, assigned, "assigned site")):
            line.set_visible(data is not None)
            line.set_data(frames if data is not None else [], data[name] if data is not None else [])
            line.set_label(label)
        show_cutoff = name == "Score" and threshold is not None
        cutoff.set_visible(show_cutoff)
        cutoff.set_data([frames[0], frames[-1]] if show_cutoff else [], [threshold, threshold] if show_cutoff else [])
        cutoff.set_label("frozen cutoff at primary trace pixel")
        for patch in list(axis.patches):
            patch.remove()
        for start, stop in spans:
            axis.axvspan(start, stop, color="#888888", alpha=.1)
        # relim(visible_only=True) prevents a previous card's hidden comparison
        # trace or threshold from determining this card's axes.
        axis.relim(visible_only=True)
        axis.autoscale_view(scalex=False, scaley=True)
        axis.set_xlim(frames[0], frames[-1] if len(frames) > 1 else frames[0]+1)
    previous_legend = axes[0].get_legend()
    if previous_legend is not None:
        previous_legend.remove()
    legend_handles = [line for line in lines[0][:3] if line.get_visible()]
    axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right")
    score_axis = axes[keys.index("Score")]
    score_legend = score_axis.get_legend()
    if score_legend is not None:
        score_legend.remove()
    score_cutoff = lines[keys.index("Score")][3]
    if score_cutoff.get_visible():
        score_axis.legend(handles=[score_cutoff], fontsize=7, loc="upper right")
    # Reserve actual text height, rather than assuming every title has one line.
    # Renderer text metrics avoid a second full-canvas draw for every ROI. Wrap
    # long source IDs at measured glyph widths while retaining all title text.
    renderer = fig.canvas.get_renderer()
    properties = title_artist.get_fontproperties()
    maximum_width = fig.bbox.width-24
    width_of = lambda value: renderer.get_text_width_height_descent(value,properties,ismath=False)[0]
    title_lines = []
    for paragraph in title.split("\n"):
        current = ""
        for token in paragraph.split():
            combined = current+" "+token if current else token
            if current and width_of(combined) > maximum_width:
                title_lines.append(current);current = ""
            while width_of(token) > maximum_width:
                lo,hi = 1,len(token)
                while lo < hi:
                    middle = (lo+hi+1)//2
                    if width_of(token[:middle]) <= maximum_width:lo = middle
                    else:hi = middle-1
                title_lines.append(token[:lo]);token = token[lo:]
            current = current+" "+token if current else token
        title_lines.append(current)
    title_artist.set_text("\n".join(title_lines))
    title_artist.set_position((.5,1-7/fig.bbox.height))
    title_bottom = title_artist.get_window_extent(renderer).y0
    top = (title_bottom-10)/fig.bbox.height
    bottom = 56/fig.bbox.height
    if top-bottom < .48:
        raise ValueError("Trace title leaves insufficient readable plotting area")
    fig.subplots_adjust(left=.095,right=.985,bottom=bottom,top=top,hspace=.19)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.png")
    fig.savefig(temporary, dpi=105)
    temporary.replace(path)


def _comparison_figure(path: Path, raw: Any, z: Any, experts: Sequence[Mapping[str, Any]], sites: Sequence[Mapping[str, Any]], occurrence_rows: Sequence[Mapping[str, Any]], limits: Mapping[str, Sequence[float]], title: str) -> None:
    _, _, _, plt = _load_deps()
    site_by_id = {row.get("site_id",row.get("model_roi_id")): row for row in sites}
    expert_by_id = {row["observation_id"]: row for row in experts}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for axis, data, name, key in zip(axes, (raw, z), COMPARISON_PANELS, ("Raw", "Score")):
        axis.imshow(data, cmap="gray", vmin=limits[key][0], vmax=limits[key][1])
        axis.scatter([float(r["x_px"]) for r in experts], [float(r["y_px"]) for r in experts], s=35, facecolors="none", edgecolors="#46dc7d", linewidths=.7)
        axis.scatter([float(r["x_px"]) for r in sites], [float(r["y_px"]) for r in sites], s=25, facecolors="none", edgecolors="#ff9123", marker="s", linewidths=.7)
        for row in occurrence_rows:
            if row["matched_site_id"]:
                site, expert = site_by_id[row["matched_site_id"]], expert_by_id[row["observation_id"]]
                axis.plot([float(expert["x_px"]), site["x_px"]], [float(expert["y_px"]), site["y_px"]], color="#f6e7a1", linewidth=.8)
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("x = column")
        axis.set_ylabel("y = row")
    fig.suptitle(title, fontsize=9)
    _save_figure(fig, path, dpi=115)
    plt.close(fig)


def _run_spatiotemporal_audit(output_root: str | Path, *, stage_paths: Mapping[str, str | Path], signed_stage_keys: Sequence[str], display_limits: Mapping[str, Sequence[float]], source_frames_ui: Sequence[int], source_binding: Mapping[str, Any], operating_point: Mapping[str, Any], numeric_threshold_paths: Mapping[str, str | Path], candidates_path: str | Path, expert_occurrences: Any, context_frames: int = 5, fullfield_step: int = 5, fps: float = 50.0, progress_callback: Any = None) -> dict[str, Any]:
    """Render one frozen state completely; never select, fit, or alter a detector.

    All stage arrays must have identical TYX shape, matching source_frames_ui.
    Every consolidated review location receives a closeup and exact-pixel trace.
    Named threshold-curve files are linked numerical evidence only.
    """
    if not numeric_threshold_paths or any(not isinstance(key,str) or not key.strip() for key in numeric_threshold_paths):
        raise ValueError("Numeric threshold paths require nonempty descriptive string keys")
    target = float(operating_point.get("target_proposals_per_frame",math.nan))
    if not math.isfinite(target) or target <= 0:
        raise ValueError("The frozen area-scaled proposal target must be finite and positive")
    candidate_input = Path(candidates_path).resolve()
    if not candidate_input.is_file():
        raise FileNotFoundError(candidate_input)
    if not source_binding.get("candidates_sha256"):
        raise ValueError("source_binding.candidates_sha256 must seal the candidate file")
    candidates, experts = _rows(candidate_input), _rows(expert_occurrences)
    for row in candidates:
        if row.get("target_proposals_per_frame") not in (None,"") and float(row["target_proposals_per_frame"]) != target:
            raise ValueError("Candidate target differs from the frozen area-scaled operating point")
    _frozen_threshold(candidates,operating_point)
    truth = _truth_semantics(source_binding,experts)
    plan = audit_inventory_plan(candidates, experts, source_frames_ui, context_frames=context_frames, fullfield_step=fullfield_step)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be finite and positive")
    if not {"Raw", "Input", "Score"} <= set(stage_paths) or not 3 <= len(stage_paths) <= 4:
        raise ValueError("Three or four actual stages including Raw/Input/Score are required")
    if any(not isinstance(name, str) or not name.strip() or "\n" in name for name in stage_paths):
        raise ValueError("Stage labels must be nonempty single-line strings")
    if list(stage_paths)[0] != "Raw" or list(stage_paths)[-1] != "Score":
        raise ValueError("Preserve pipeline order with Raw first and Score last")
    if set(source_binding.get("stage_sha256", {})) != set(stage_paths):
        raise ValueError("source_binding.stage_sha256 must seal every input stage before rendering")
    if set(display_limits) != set(stage_paths) or not set(signed_stage_keys) <= set(stage_paths):
        raise ValueError("Every stage requires a declared fixed display scale")
    for name, bounds in display_limits.items():
        low, high = map(float, bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Display limits must be finite and ordered")
        if name in signed_stage_keys and not math.isclose(low, -high, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Signed stage scales must be symmetric around zero")
    output = Path(output_root).resolve()
    paths = {name: Path(stage_paths[name]).resolve() for name in stage_paths}
    contract = {"schema_version": 1, "renderer_sha256": _digest(Path(__file__)),
                "evaluation_helper_sha256": _digest(Path(__file__).with_name("two_stencil_evaluation.py")),
                "media_helper_sha256": _digest(Path(_base.__file__)),
                "inventory_validator_sha256": _digest(Path(__file__).resolve().parents[2] / "reports" / "scientific_audit.py"),
                "score_stage": "Score", "cpu_thread_limit": 1,
                "stage_paths": {k: str(v) for k, v in paths.items()},
                "stage_sequence": list(paths), "signed_stage_keys": list(signed_stage_keys),
                "display_limits": dict(display_limits), "source_frames_ui": list(source_frames_ui),
                "source_binding": dict(source_binding), "operating_point": dict(operating_point),
                "truth_semantics":truth,"caption_footer_height_px":18,
                "candidate_rows_sha256": _canonical_digest(candidates), "experts_sha256": _canonical_digest(experts),
                "numeric_threshold_paths": {str(k): str(Path(v).resolve()) for k, v in numeric_threshold_paths.items()},
                "context_frames": context_frames, "fullfield_step": fullfield_step, "source_fps": fps,
                "fullfield_fps": fps / fullfield_step,
                "video_encoding": dict(VIDEO_ENCODING),
                "panel_fit_semantics": PANEL_FIT_SEMANTICS,
                "fullfield_panel_image_area_size_wh": [180, 136],
                "closeup_panel_image_area_size_wh": [144, 112],
                "fullfield_sampling": "every declared step plus final source frame; omitted source frames remain in closeups and full traces",
                "closeup_sampling": "all source samples in union of every occurrence/proposal frame plus fixed context",
                "no_roi_or_occurrence_sampling": True,
                "model_roi_interpretation": "spatial review location, not unique neuron or biological event",
                "numeric_curve_media_scope": "only this finite frozen threshold receives complete media; linked named curves are numerical evidence"}
    contract_digest = _canonical_digest(contract)
    if output.exists() and (output / "run_contract.json").exists():
        existing = json.loads((output / "run_contract.json").read_text())
        if _canonical_digest(existing) != contract_digest:
            raise ValueError("Cannot resume an audit under a changed source/configuration contract")
        if (output / "status.json").exists() and json.loads((output / "status.json").read_text()).get("status") == "complete":
            verify_completed_audit(output)
            return json.loads((output / "summary.json").read_text())
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an unrelated output root")
    output.mkdir(parents=True, exist_ok=True)
    progress = _Progress(output, progress_callback)
    _json(output / "run_contract.json", contract)
    _json(output / "status.json", {"status": "rendering", "scientific_audit_complete": False})
    _json(output / "inventory_plan.json", plan)
    progress.emit("audit_started", force=True, expert_rois=plan["expert_roi_count"], model_review_locations=plan["model_roi_count"], occurrences=plan["expert_occurrence_count"], expected_videos=plan["expected_video_count"])
    source_records = []
    expected_stage_hashes = dict(source_binding.get("stage_sha256", {}))
    for name, path in paths.items():
        digest = _digest(path)
        if name in expected_stage_hashes and digest != expected_stage_hashes[name]:
            raise ValueError(f"Stage {name} disagrees with its supplied sealed SHA-256")
        source_records.append({"stage": name, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
        progress.emit("source_hash_verified", force=True, stage_name=name, sha256=digest, source_path=str(path))
    for name,record in source_binding.get("numeric_stage_bindings",{}).items():
        path = Path(record["path"]).resolve()
        digest = _digest(path)
        if digest != record.get("sha256") or path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Numeric linked stage {name} differs from sealed bytes")
        source_records.append({"stage":name,"path":str(path),"sha256":digest,"size_bytes":path.stat().st_size,"media_scope":"numeric_stage_link"})
    for label, source in (("candidates", candidate_input), ("expert_occurrences", expert_occurrences)):
        expected_hash = source_binding.get(f"{label}_sha256")
        if isinstance(source, (str, Path)):
            path = Path(source).resolve()
            digest = _digest(path)
            if expected_hash and digest != expected_hash:
                raise ValueError(f"{label} differs from its sealed SHA-256")
            source_records.append({"stage": label, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
        elif expected_hash and _canonical_digest(_rows(source)) != expected_hash:
            raise ValueError(f"{label} rows differ from their sealed canonical SHA-256")
    numeric_sources = []
    for threshold_curve, source in numeric_threshold_paths.items():
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Declared numeric threshold evidence is missing: {path}")
        if path.is_dir() and (output.is_relative_to(path) or path.is_relative_to(output)):
            raise ValueError("Numeric source directories must not overlap the audit output")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".csv", ".tsv", ".json"})
        if not files:
            raise ValueError(f"Declared numeric threshold evidence contains no metadata: {path}")
        numeric_sources.extend({"threshold_curve":threshold_curve,"path":str(p),"sha256":_digest(p),"size_bytes":p.stat().st_size} for p in files)
    source_manifest = {"sources": source_records, "numeric_state_sources": numeric_sources, "source_binding": dict(source_binding), "renderer_sha256": _digest(Path(__file__))}
    if (output / "source_manifest.json").exists():
        prior_sources = json.loads((output / "source_manifest.json").read_text())
        for key in ("sources", "numeric_state_sources"):
            if _canonical_digest(prior_sources.get(key, [])) != _canonical_digest(source_manifest[key]):
                raise ValueError("Cannot resume media under changed source-file bytes")
    _json(output / "source_manifest.json", source_manifest)
    np, Image, _, plt = _load_deps()
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
    shape = arrays["Raw"].shape
    if len(shape) != 3 or shape[0] != len(source_frames_ui) or any(a.shape != shape for a in arrays.values()):
        raise ValueError("Every sealed array must align to the same full-duration TYX source grid")
    if any(a.dtype.kind not in "fui" for a in arrays.values()):
        raise ValueError("Stage arrays must be numeric")
    frames = list(map(int, source_frames_ui))
    frame_index = {frame: i for i, frame in enumerate(frames)}
    height, width = shape[1:]
    for row in [*experts, *candidates]:
        if not 0 <= float(row["x_px"]) < width or not 0 <= float(row["y_px"]) < height:
            raise ValueError("Annotation/proposal coordinates are outside the source image")
    threshold = _frozen_threshold(candidates, operating_point)
    _verify_candidate_score_binding(candidates, arrays["Score"], frame_index, threshold, operating_point)
    for name, array in arrays.items():
        for i, frame in enumerate(array):
            if not np.isfinite(frame).all():
                raise ValueError(f"Nonfinite stage {name} at source UI {frames[i]}")
        progress.emit("stage_values_verified", force=True, stage_name=name)
    intervals = _burst_intervals(experts)
    evaluation = _descriptive_evaluation(candidates,experts)
    expert_root, model_root, comparison = (output / name for name in ("1_Expert_Annotations", "2_Model_Annotations", "3_Comparison"))
    for root in (expert_root, model_root):
        for sub in ("videos/closeups", "figures/traces", "metadata", "exact_pixel_traces"):
            (root / sub).mkdir(parents=True, exist_ok=True)
    (comparison / "trace_comparisons").mkdir(parents=True, exist_ok=True)
    _table(expert_root / "expert_occurrences.csv", experts)
    _table(model_root / "model_occurrences.csv", plan["model_assignments"])
    _table(model_root / "review_sites.csv", plan["model_sites"])
    _json(model_root / "review_sites.json", plan["model_sites"])
    _table(comparison / "expert_model_matches.csv", evaluation["occurrence_rows"])
    _table(comparison / "burst_sites.csv", evaluation["site_rows"])
    _table(comparison / "site_memberships.csv", evaluation["membership_rows"])
    checkpoint_path = output / "render_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    figure_checkpoint_path = output / "figure_checkpoint.json"
    figure_checkpoint = json.loads(figure_checkpoint_path.read_text()) if figure_checkpoint_path.exists() else {}
    videos = []
    candidates_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        candidates_by_frame.setdefault(int(row["source_frame_ui"]), []).append(row)
    def stages(frame: int):
        index = frame_index[frame]
        return {name: a[index] for name, a in arrays.items()}
    def expert_markers(frame: int):
        return [row for row in experts if int(row["source_start_ui"]) <= frame <= int(row["source_stop_ui"])]
    def caption(section: str) -> str:
        if section == "expert":return truth["expert_label"] + " | source extent only"
        return ("Synthetic scene" if truth["truth_mode"] == "fully_synthetic" else "Sparse real recording") + " | model proposals; spatial review sites"
    for section, root, color, marker_function in (("expert", expert_root, GREEN, expert_markers), ("model", model_root, ORANGE, lambda f: candidates_by_frame.get(f, []))):
        shown = plan["fullfield_source_frames_ui"]
        marker_index = next((i for i, frame in enumerate(shown) if marker_function(frame)), None)
        videos.append(_write_video(root / "videos" / f"{section}_sequential_full_field.mp4", shown,
            lambda frame, c=color, mf=marker_function, sec=section: _captioned_panel(stages(frame), display_limits, source_ui=frame, source_fps=fps, markers=mf(frame), color=c,caption=caption(sec)),
            fps=fps/fullfield_step, marker_sample_index=marker_index, annotation_section=section,
            checkpoint=checkpoint, checkpoint_path=checkpoint_path, progress=progress))
        videos[-1]["panel_image_geometry"] = panel_image_geometry(width, height, 180, 136)
        videos[-1]["source_crop_xyxy_half_open"] = [0, 0, width, height]
        videos[-1]["truth_caption"] = caption(section)
        videos[-1]["caption_footer_height_px"] = 18
        progress.emit(f"{section}_fullfield_complete", force=True, completed_videos=len(videos), expected_videos=plan["expected_video_count"])
    trace_cache: dict[tuple[int, int], dict[str, Any]] = {}
    def exact_traces(x: float, y: float):
        # Explicit nearest pixel convention, avoiding NumPy's ties-to-even ambiguity.
        px, py = min(width-1, int(math.floor(x+.5))), min(height-1, int(math.floor(y+.5)))
        key = (px, py)
        if key not in trace_cache:
            values = {name: np.asarray(a[:, py, px]).copy() for name, a in arrays.items()}
            if any(not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Nonfinite exact-pixel trace")
            trace_cache[key] = values
        return trace_cache[key], px, py
    threshold = _frozen_threshold(candidates, operating_point)
    completed_rois = 0
    expected_rois = plan["expert_roi_count"] + plan["model_roi_count"]
    for section, root, locations, color in (("expert", expert_root, list(plan["unique_experts"].values()), GREEN), ("model", model_root, plan["model_sites"], ORANGE)):
        for location in locations:
            identifier = location["canonical_roi_id"] if section == "expert" else location["model_roi_id"]
            shown = plan[f"{section}_closeup_source_frames_ui"][identifier]
            relevant = location["intervals_ui"] if section == "expert" else [(f, f) for f in location["member_source_frames_ui"]]
            x, y = float(location["x_px"]), float(location["y_px"])
            bounds = identity_crop(location, width, height)
            def active(frame: int, intervals=relevant):
                return any(int(a) <= frame <= int(b) for a, b in intervals)
            # In model clips the member proposal coordinates, not a relocated marker, are shown.
            model_member_ids = set(location.get("member_proposal_ids", []))
            def close_markers(frame: int, loc=location, sec=section, member_ids=model_member_ids, af=active):
                if sec == "expert":
                    return [r for r in loc["occurrences"] if int(r["source_start_ui"]) <= frame <= int(r["source_stop_ui"])]
                return [r for r in candidates_by_frame.get(frame, []) if r["proposal_id"] in member_ids]
            marker_index = next(i for i, frame in enumerate(shown) if close_markers(frame))
            videos.append(_write_video(root / "videos/closeups" / f"{identifier}.mp4", shown,
                lambda frame, c=color, mf=close_markers, crop=bounds, sec=section: _captioned_panel(stages(frame), display_limits, source_ui=frame, source_fps=fps, markers=mf(frame), color=c, crop=crop, panel_width=144, panel_height=112,caption=caption(sec)),
                fps=fps, marker_sample_index=marker_index, annotation_section=section,
                checkpoint=checkpoint, checkpoint_path=checkpoint_path, progress=progress))
            closeup_geometry = panel_image_geometry(bounds[2] - bounds[0], bounds[3] - bounds[1], 144, 112)
            videos[-1]["panel_image_geometry"] = closeup_geometry
            videos[-1]["source_crop_xyxy_half_open"] = list(bounds)
            videos[-1]["truth_caption"] = caption(section)
            videos[-1]["caption_footer_height_px"] = 18
            traces, px, py = exact_traces(x, y)
            trace_path = root / "exact_pixel_traces" / f"{identifier}.csv"
            if not _artifact_cached(trace_path, figure_checkpoint):
                _table(trace_path, [{"source_frame_ui": frame, "frozen_local_cutoff": _threshold_at(threshold, px, py), **{name: float(values[i]) for name, values in traces.items()}} for i, frame in enumerate(frames)])
                _checkpoint_artifact(trace_path, figure_checkpoint, figure_checkpoint_path)
            figure_path = root / "figures/traces" / f"{identifier}.png"
            if not _artifact_cached(figure_path, figure_checkpoint):
                _trace_plot(figure_path, traces, frames, title=f"{caption(section)} | {identifier}\nExact pixel ({px},{py}); all {len(frames)} source samples", spans=relevant, color="#46dc7d" if section == "expert" else "#ff9123", threshold=_threshold_at(threshold, px, py))
                _checkpoint_artifact(figure_path, figure_checkpoint, figure_checkpoint_path)
            metadata_path = root / "metadata" / f"{identifier}.json"
            if not _artifact_cached(metadata_path, figure_checkpoint):
                _json(metadata_path, {**location, "exact_trace_pixel_xy": [px, py], "frozen_local_cutoff": _threshold_at(threshold, px, py), "rounding": "floor(coordinate+0.5), clipped at last image pixel", "closeup_source_frames_ui": shown, "crop_xyxy_half_open": bounds, "panel_image_geometry": closeup_geometry, "all_source_samples_in_trace": True, "source_time_semantics": SOURCE_TIME_SEMANTICS,
                    "original_source_offset_xy":source_binding.get("original_source_offset_xy",[0,0]),"truth_semantics":truth})
                _checkpoint_artifact(metadata_path, figure_checkpoint, figure_checkpoint_path)
            completed_rois += 1
            progress.emit(f"{section}_roi", force=completed_rois % 25 == 0 or completed_rois == expected_rois,
                          roi=identifier, completed_rois=completed_rois, expected_rois=expected_rois,
                          completed_videos=len(videos), expected_videos=plan["expected_video_count"])
    sites_by_id = {row["site_id"]: row for row in evaluation["site_rows"]}
    nearest_metrics = []
    for comparison_index, row in enumerate(evaluation["occurrence_rows"], start=1):
        observation = row["observation_id"]
        start, stop = int(row["window_start_frame_ui"]), int(row["window_stop_frame_ui"])
        expert_values, px, py = exact_traces(float(row["x_px"]), float(row["y_px"]))
        near = sites_by_id.get(row["nearest_site_id"])
        assigned = sites_by_id.get(row["matched_site_id"])
        near_values = exact_traces(float(near["x_px"]), float(near["y_px"]))[0] if near else None
        assigned_values = exact_traces(float(assigned["x_px"]), float(assigned["y_px"]))[0] if assigned and (not near or assigned["site_id"] != near["site_id"]) else None
        indices = slice(frame_index[start], frame_index[stop]+1)
        comparison_frames = context_source_frames([(start, stop)], frames, context_frames)
        comparison_slice = slice(frame_index[comparison_frames[0]], frame_index[comparison_frames[-1]]+1)
        correlation = _correlation(expert_values["Score"][indices], near_values["Score"][indices]) if near_values else None
        lag, lag_correlation = _lag(expert_values["Score"][indices], near_values["Score"][indices]) if near_values else (None, None)
        metric = {**row, "expert_pixel_cutoff": _threshold_at(threshold, px, py),
                  "nearest_pixel_cutoff": _threshold_at(threshold, int(math.floor(float(near["x_px"])+.5)), int(math.floor(float(near["y_px"])+.5))) if near else None,
                  "cutoff_line_scope": "primary expert pixel only; nearest cutoff is in this table",
                  "nearest_candidate_rank": near["site_rank"] if near else None,
                  "nearest_candidate_score": near["score"] if near else None,
                  "event_score_correlation": correlation, "best_lag_frames": lag,
                  "best_lag_correlation": lag_correlation, "best_lag_seconds": lag / fps if lag is not None else None,
                  "correlation_stage": "Score", "lag_search_radius_frames": 5,
                  "lag_sign": "positive means nearest-site trace lags truth/annotation center trace; not detection delay",
                  "nearest_assignment_computed_separately": True,
                  "nearest_equals_assigned": bool(near and assigned and near["site_id"] == assigned["site_id"]),
                  "event_window_semantics": truth["window_semantics"],"truth_semantics":truth,
                  "comparison_plot_scope": "declared_burst_plus_fixed_context; full_duration_ROI_traces_remain_separate",
                  "comparison_plot_context_frames": context_frames,
                  "comparison_plot_source_frames_ui": comparison_frames,
                  "comparison_plot_start_frame_ui": comparison_frames[0],
                  "comparison_plot_stop_frame_ui": comparison_frames[-1],
                  "correlation_and_lag_window_ui_inclusive": [start, stop],
                  "unmatched_candidate_interpretation": truth["unmatched_interpretation"],
                  "original_source_offset_xy":source_binding.get("original_source_offset_xy",[0,0])}
        nearest_metrics.append(metric)
        comparison_path = comparison / "trace_comparisons" / f"{observation}.png"
        if not _artifact_cached(comparison_path, figure_checkpoint):
            _trace_plot(comparison_path, {name: values[comparison_slice] for name, values in expert_values.items()}, comparison_frames,
                        title=f"{truth['expert_label']} | {observation} | window/site matched={row['matched']}\nUI {comparison_frames[0]}–{comparison_frames[-1]} (window {start}–{stop}, context ±{context_frames}); nearest {row['nearest_site_id']}",
                        spans=[(start, stop)], color="#46dc7d",
                        second={name: values[comparison_slice] for name, values in near_values.items()} if near_values is not None else None,
                        assigned={name: values[comparison_slice] for name, values in assigned_values.items()} if assigned_values is not None else None,
                        threshold=_threshold_at(threshold, px, py))
            _checkpoint_artifact(comparison_path, figure_checkpoint, figure_checkpoint_path)
        comparison_metadata = comparison / "trace_comparisons" / f"{observation}.json"
        if not _artifact_cached(comparison_metadata, figure_checkpoint):
            _json(comparison_metadata, metric)
            _checkpoint_artifact(comparison_metadata, figure_checkpoint, figure_checkpoint_path)
        progress.emit("occurrence_comparison", force=comparison_index % 25 == 0 or comparison_index == len(experts),
                      completed_occurrences=comparison_index, expected_occurrences=len(experts))
    _table(comparison / "nearest_roi_trace_metrics.csv", nearest_metrics)
    # Exact grayscale temporal means are accumulated without loading a whole movie.
    raw_overview = np.zeros((height, width), dtype=np.float64)
    z_overview = np.zeros((height, width), dtype=np.float64)
    all_burst_indices = []
    for burst, (start, stop) in sorted(intervals.items()):
        indices = list(range(frame_index[start], frame_index[stop]+1))
        all_burst_indices.extend(indices)
        raw_map = np.zeros((height, width), dtype=np.float64)
        z_map = np.zeros((height, width), dtype=np.float64)
        for i in indices:
            raw_map += arrays["Raw"][i]
            z_map += arrays["Score"][i]
        raw_overview += raw_map
        z_overview += z_map
        _comparison_figure(comparison / f"burst_{burst}_comparison.png", raw_map/len(indices), z_map/len(indices),
            [r for r in experts if int(r["burst_id"]) == burst], [r for r in evaluation["site_rows"] if r["burst_id"] == burst],
            [r for r in evaluation["occurrence_rows"] if r["burst_id"] == burst], display_limits,
            f"{truth['expert_label']}; window {burst}; grayscale temporal mean; descriptive site comparison")
    if not all_burst_indices:
        start = int(operating_point.get("application_source_start_ui",frames[0]))
        stop = int(operating_point.get("application_source_stop_ui",frames[-1]))
        all_burst_indices = list(range(frame_index[start],frame_index[stop]+1))
        for index in all_burst_indices:
            raw_overview += arrays["Raw"][index];z_overview += arrays["Score"][index]
        overview_sites = plan["model_sites"]
        overview_title = "Expert N/A: no declared source; application temporal means with all model review sites"
    else:
        # Deduplicate overlapping windows so each source frame weights the map once.
        all_burst_indices = sorted(set(all_burst_indices))
        raw_overview.fill(0);z_overview.fill(0)
        for index in all_burst_indices:
            raw_overview += arrays["Raw"][index];z_overview += arrays["Score"][index]
        overview_sites = evaluation["site_rows"]
        overview_title = truth["expert_label"]+"; union-of-windows means; review sites are not a pooled detector"
    _comparison_figure(comparison / "spatial_overview.png", raw_overview/len(all_burst_indices), z_overview/len(all_burst_indices), experts,
                       overview_sites,evaluation["occurrence_rows"],display_limits,overview_title)
    _json(comparison/"overview_metadata.json",{"source_frames_ui":[frames[index] for index in all_burst_indices],
        "truth_semantics":truth,"temporal_mean_weighting":"each included source frame once","expert_comparison_applicable":bool(experts)})
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    finite = [r for r in nearest_metrics if r["nearest_site_distance_px"] is not None]
    paired = [r for r in finite if r["event_score_correlation"] is not None]
    axes[0, 0].scatter([r["nearest_site_distance_px"] for r in paired], [r["event_score_correlation"] for r in paired], s=10, color="black")
    axes[0, 0].set(xlabel="Nearest site distance (px)", ylabel="Burst score correlation")
    axes[0, 1].scatter([r["nearest_site_distance_px"] for r in finite], [r["nearest_candidate_rank"] for r in finite], s=10, color="black")
    axes[0, 1].set(xlabel="Nearest site distance (px)", ylabel="Site rank")
    axes[1, 0].hist([r["nearest_site_distance_px"] for r in finite], bins=12, color="gray")
    axes[1, 0].set(xlabel="Nearest site distance (px)", ylabel="Known occurrences")
    axes[1, 1].hist([len(r["member_source_frames_ui"]) for r in plan["model_sites"]], bins=12, color="gray")
    axes[1, 1].set(xlabel="Member source frames per review site", ylabel="Spatial review sites")
    if not experts:
        for axis in axes.flat[:3]:
            axis.text(.5,.5,"Expert N/A: no declared source",transform=axis.transAxes,ha="center",fontsize=9)
    figure.suptitle(truth["expert_label"]+"; descriptive audit diagnostics",fontsize=10)
    _save_figure(figure, comparison / "aggregate_diagnostics.png", dpi=115)
    plt.close(figure)
    _json(output / "video_manifest.json", {"videos": videos})
    _json(output / "display_limits.json", {"limits": dict(display_limits), "signed_stage_keys": list(signed_stage_keys), "across_cell_source": source_binding.get("display_limits_source", "caller_supplied_common_frozen_limits")})
    coverage = {"audited_operating_point": dict(operating_point), "media_state": "frozen_operating_point",
                "numeric_threshold_evidence":{key:{"numeric_path":str(Path(path).resolve()),"scope":"linked curve; media binds only the declared frozen operating point"} for key,path in numeric_threshold_paths.items()},
                "truth_semantics":truth,
                "expert_roi_expected": plan["expert_roi_count"], "expert_occurrences_expected": plan["expert_occurrence_count"],
                "model_review_roi_expected": plan["model_roi_count"], "all_model_rois_rendered": True,
                "all_expert_occurrences_compared": True, "model_sites_are_biological_identities": False,
                "fullfield_step": fullfield_step, "fullfield_fps": fps/fullfield_step,
                "fullfield_omitted_source_frames": len(frames)-len(plan["fullfield_source_frames_ui"]),
                "closeups_full_rate_all_relevant_source_frames": True,
                "full_duration_exact_pixel_trace_sample_count": len(frames)}
    coverage["occurrence_comparison_plot_extent"] = f"exact declared burst plus {context_frames} source frames of context per side, clipped to source extent"
    _json(output / "coverage_manifest.json", coverage)
    llm = {"schema_version": 1, "entrypoint": "summary.json", "annotation_separation": "strict",
           "comparison_spatial_panels": COMPARISON_PANELS, "model_stage_sequence": list(arrays), "operating_point": dict(operating_point), "score_stage": "Score",
           "coordinate_convention": "x=column,y=row", "frame_convention": SOURCE_TIME_SEMANTICS,
           "coverage": coverage, "marker_semantics": {"green": truth["expert_label"], "orange": "model proposals", "pale_yellow": "window-local descriptive assignment links"},
           "primary_tables": ["1_Expert_Annotations/expert_occurrences.csv", "2_Model_Annotations/model_occurrences.csv", "3_Comparison/expert_model_matches.csv", "3_Comparison/nearest_roi_trace_metrics.csv"],
           "limitations": [truth["unmatched_interpretation"],"other thresholds numerical only",truth["onset_semantics"],truth["comparison_scope"],"review sites not neuron or event identities","fullfield video temporally decimated; all relevant closeup frames and trace samples retained"]}
    _json(output / "llm_context.json", llm)
    for root, description in ((expert_root, truth["expert_label"]+". Every declared source identity/occurrence is retained; zero rows means N/A, never invented truth."), (model_root, "Model-only markers, every consolidated review location; no biological identities."), (comparison, truth["comparison_scope"]+". Figures/tables only. Nearest-candidate similarity and window-local assignment are separate.")):
        (root / "README.md").write_text(description + "\n\nScientific video masters use lossless RGB H.264 (High 4:4:4 Predictive). Every decoded frame is byte-identical to its source composite, with exact source/decoded palette and marker-presence checks. This profile may require a software-capable video player; PNG previews are retained. No unchecked lossy proxy is a validated scientific master.\n")
    (output / "REPORT.md").write_text(
        "# Spatiotemporal scientific audit\n\n"
        f"{truth['expert_label']}. Full media audit of frozen threshold {_threshold_description(threshold)}, setup proposal target {target:g} per frame: {plan['expert_roi_count']} truth/annotation ROIs, {plan['expert_occurrence_count']} declared occurrences, "
        f"and every one of {plan['model_roi_count']} consolidated spatial review locations from {len(candidates)} emitted rows. "
        "No location or occurrence was omitted. Source arrays and input tables are hash-bound; scores and detector membership were not changed.\n\n"
        f"Full-field movies display every {fullfield_step}th source frame plus the last at {fps/fullfield_step:g} fps, with explicit source-frame maps. "
        "These videos omit intermediate frames and do not measure event timing accuracy. Closeups include all occurrence/proposal source frames plus fixed context; "
        f"all per-ROI exact-pixel traces contain {len(frames)} samples. Per-occurrence comparison plots show the exact declared burst plus {context_frames} source frames "
        "of context on each side, with their rendered source-frame lists recorded in metadata; correlation and lag still use only the declared burst. "
        "Movie panels preserve the source aspect ratio with black letterboxing and integer raster rounding; the complete source/crop extent is fitted, "
        "and marker coordinates follow the fitted image rectangle. Exact geometry and padding are recorded for every video. "
        "Scientific video masters use lossless RGB H.264 (High 4:4:4 Predictive): every decoded frame is byte-checked against its source composite, "
        "and exact annotation palette/presence is checked on every frame. This profile may require a software-capable video player; PNG previews are retained. "
        "No unchecked lossy proxy is a validated scientific master. "
        "Comparison maps are grayscale temporal means, not the detector's readout.\n\n"
        "Other thresholds have linked numerical evidence and are outside this operating point's media audit. "
        f"{truth['unmatched_interpretation']}. {truth['onset_semantics']}. "
        "Review sites are not unique events or neurons. Window/site trace joins are descriptive audit views; use linked primary metrics for synthetic TP/FP/FN and event timing. "
        "When source truth is empty, the Expert fullfield explicitly says N/A and retains the source movie; no expert ROI or event is invented. "
        "An18-pixel grayscale footer identifies truth mode; it does not crop or alter source-image panels.\n"
    )
    summary = {"status": "validation_pending", "scientific_audit_complete": False, "audited_media_state": "frozen_operating_point",
               "truth_semantics":truth,"operating_point":dict(operating_point),
               "expert_roi_count": plan["expert_roi_count"], "expert_occurrence_count": plan["expert_occurrence_count"],
               "model_roi_count": plan["model_roi_count"], "model_proposal_count": len(candidates),
               "video_count": len(videos), "comparison_trace_count": len(nearest_metrics),
               "all_trace_sample_count": len(frames), "metric_summary": evaluation["summary"],
               "claim_boundary": "scientific artifact fidelity for the frozen state; synthetic primary metrics are separate; no real precision/identity/onset or all-threshold-media claim"}
    _json(output / "summary.json", summary)
    failures = []
    for section, expected in ((expert_root, plan["expert_roi_count"]), (model_root, plan["model_roi_count"])):
        for folder, extension in (("videos/closeups", "mp4"), ("figures/traces", "png"), ("metadata", "json"), ("exact_pixel_traces", "csv")):
            if len(list((section / folder).glob(f"*.{extension}"))) != expected:
                failures.append(f"{section.name}/{folder} cardinality mismatch")
    if len(videos) != plan["expected_video_count"] or len(nearest_metrics) != len(experts):
        failures.append("video or occurrence cardinality mismatch")
    for video in videos:
        thumbnail = Path(video["thumbnail"])
        if not thumbnail.is_file() or _digest(thumbnail) != video.get("thumbnail_sha256"):
            failures.append(f"video thumbnail missing or changed: {thumbnail}")
    for video in videos:
        if not all(video.get(key) is True for key in (
            "full_decode_pass", "encoded_marker_separation_pass", "all_frames_rgb_byte_exact",
            "all_frames_marker_expectations_pass", "all_frames_source_palette_pure",
            "all_frames_decoded_palette_pure")):
            failures.append(f"Incomplete lossless video validation: {video['path']}")
        try:
            _validate_video_geometry(video,stage_count=len(paths),source_fps=fps,fullfield_step=fullfield_step)
        except (KeyError,TypeError,ValueError,ZeroDivisionError) as error:
            failures.append(f"Encoded geometry/timing validation failed: {video['path']}: {error}")
    pngs = list(output.rglob("*.png"))
    for path in pngs:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            failures.append(f"PNG decode failure: {path}: {error}")
    if list(comparison.rglob("*.mp4")):
        failures.append("comparison contains a video")
    validation = {"status": "passed" if not failures else "failed", "scientific_audit_complete": not failures,
                  "failures": failures, "inventory_complete": not failures, "all_videos_full_decode": all(r["full_decode_pass"] for r in videos),
                  "all_frames_rgb_byte_exact": all(r["all_frames_rgb_byte_exact"] for r in videos),
                  "all_frames_marker_expectations_pass": all(r["all_frames_marker_expectations_pass"] for r in videos),
                  "candidate_scores_equal_saved_score_array": True, "score_stage": "Score",
                  "all_video_geometry_timing_checked":True,
                  "encoded_marker_separation_pass": all(r["encoded_marker_separation_pass"] for r in videos),
                  "png_count": len(pngs), "source_hashes_computed": True, "no_new_detector_score": True}
    _json(output / "validation.json", validation)
    def index():
        _json(output / "artifact_index.json", {"artifacts": [{"path": str(p.relative_to(output)), "sha256": _digest(p), "size_bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_index.json"]})
    index()
    if plan["model_roi_count"] and plan["expert_roi_count"]:
        from neurobench.reports.scientific_audit import require_three_section_scientific_audit
        inventory = require_three_section_scientific_audit(output, expected_expert_roi_count=plan["expert_roi_count"], expected_model_roi_count=plan["model_roi_count"], expected_expert_occurrence_count=len(experts), expected_comparison_panels=COMPARISON_PANELS)
        _json(output / "inventory.json", inventory.to_dict())
    else:
        _json(output / "inventory.json", {"complete":not failures,"model_roi_count":plan["model_roi_count"],
            "expert_roi_count":plan["expert_roi_count"],"expert_occurrence_count":len(experts),"expert_applicable":bool(experts),
            "zero_count_semantics":"no invented ROI; both fullfields and all extant model/truth ROI artifacts retained; cardinalities and encoded media validated explicitly"})
    if failures:
        _json(output / "status.json", {"status": "failed_validation", "scientific_audit_complete": False})
        raise RuntimeError("; ".join(failures))
    summary.update(status="complete", scientific_audit_complete=True)
    _json(output / "summary.json", summary)
    _json(output / "status.json", {"status": "complete", "scientific_audit_complete": True})
    index()
    return summary



def run_spatiotemporal_audit(root: str | Path, *, stage_paths: Mapping[str, str | Path],
                      display_limits: Mapping[str, Sequence[float]], signed_stage_keys: Sequence[str],
                      source_frames_ui: Sequence[int], candidates_path: str | Path,
                      expert_occurrences: Any, operating_point: Mapping[str, Any],
                      source_binding: Mapping[str, Any], numeric_threshold_paths: Mapping[str, str | Path],
                      fps: float = 50.0, context_frames: int = 5, fullfield_step: int = 5,
                      progress_callback: Any = None) -> dict[str, Any]:
    """Audit every expert identity/occurrence and frozen model review location.

    stage_paths contains three or four stages, Raw first, Score last, and Input
    in between; labels are retained in media/CSV/metadata. stage_sha256 seals
    every stage, and candidates_sha256 seals the complete frozen proposal table.
    Source UI values are consecutive, one-based, inclusive. The full source
    interval is traced, while application bounds may exclude initialization.
    """
    # Set limits before optional numerical imports; restore the caller's env.
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {key: os.environ.get(key) for key in thread_vars}
    try:
        for key in thread_vars:
            os.environ[key] = "1"
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return _run_spatiotemporal_audit(root, stage_paths=stage_paths,
                display_limits=display_limits, signed_stage_keys=signed_stage_keys,
                source_frames_ui=source_frames_ui, candidates_path=candidates_path,
                expert_occurrences=expert_occurrences, operating_point=operating_point,
                source_binding=source_binding, numeric_threshold_paths=numeric_threshold_paths,
                fps=fps, context_frames=context_frames, fullfield_step=fullfield_step,
                progress_callback=progress_callback)
    except Exception as error:
        output = Path(root).resolve()
        status_path = output / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("status") != "complete":
                _json(status_path, {"status": "failed", "scientific_audit_complete": False,
                    "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.preflight:
        plan = audit_inventory_plan(config["candidates_path"], config["expert_occurrences"],
            config["source_frames_ui"], context_frames=config.get("context_frames", 5),
            fullfield_step=config.get("fullfield_step", 5))
        print(json.dumps({k: v for k, v in plan.items() if k not in {
            "unique_experts", "model_sites", "model_assignments", "expert_closeup_source_frames_ui",
            "model_closeup_source_frames_ui", "fullfield_source_frames_ui"}}, sort_keys=True))
    else:
        print(json.dumps(run_spatiotemporal_audit(**config), sort_keys=True))


if __name__ == "__main__":
    main()
