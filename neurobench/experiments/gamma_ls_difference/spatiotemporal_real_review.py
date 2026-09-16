"""Real-panel annotation preparation and calibration-only fluorescence descriptions.

No detector is run here. NumPy and movie pixels are loaded only by the explicit
descriptive pass or accepted-coverage validation, never on module import.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FRAME_REVIEW_FIELDS = ("panel_id","source_frame_ui","status","source_inventory_complete",
    "active_regions_complete","uncertainty_explicit","reviewer_id","reviewed_at","notes")
SOURCE_FIELDS = ("panel_id","source_id","presence_start_ui","presence_stop_ui","polygon_xy_json",
    "source_type","identity_status","reviewer_id","notes")
ACTIVE_FIELDS = ("panel_id","source_frame_ui","active_region_id","source_id","polygon_xy_json","reviewer_id","notes")
UNCERTAIN_FIELDS = ("panel_id","source_frame_ui","uncertainty_id","polygon_xy_json","reason","reviewer_id")
EVENT_FIELDS = ("panel_id","source_id","event_id","active_start_ui","active_stop_ui",
    "onset_lower_ui","onset_upper_ui","onset_status","reviewer_id","notes")
MUTABLE_REVIEW_FILES = ("frame_reviews.tsv","source_inventory.tsv","active_regions.tsv",
    "uncertain_areas.tsv","fluorescence_events.tsv","panel_characterization.tsv")


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path":str(path),"sha256":_sha(path),"size_bytes":path.stat().st_size}


def _verify(record: Mapping[str, Any]) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or _sha(path) != record["sha256"]:
        raise ValueError(f"Bound source changed: {path}")
    return path


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle,delimiter="\t"))


def _table(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def panel_geometry(height: int = 340, width: int = 573, side: int = 128) -> list[dict[str, Any]]:
    """Three adjacent interior crops centered as a group, using geometry only."""
    if side <= 0 or height < side+2 or width < 3*side+2:
        raise ValueError("Source geometry cannot contain three interior square panels")
    x_base,y0 = (width-3*side)//2,(height-side)//2
    return [{"panel_id":f"panel_{index+1:02d}","x0":x_base+index*side,"y0":y0,
             "x1":x_base+(index+1)*side,"y1":y0+side,"width":side,"height":side,
             "minimum_available_spatial_halo_px":min(x_base+index*side,y0,width-(x_base+(index+1)*side),height-(y0+side))}
            for index in range(3)]


def frame_grid(panels: Sequence[Mapping[str, Any]], evaluation_ui: Sequence[int]) -> list[dict[str, Any]]:
    first,last = map(int,evaluation_ui)
    if not 1 <= first <= last:
        raise ValueError("Invalid inclusive source frame interval")
    return [{"panel_id":panel["panel_id"],"source_frame_ui":frame,
             "source_numpy_index":frame-1,"evaluation_array_row":frame-first,
             **{key:int(panel[key]) for key in ("x0","y0","x1","y1")}}
            for panel in panels for frame in range(first,last+1)]


def _physical_metadata(frame_rate_hz: float, pixel_size_um: float, reported_bandwidth_hz: float) -> dict[str, Any]:
    if any(not math.isfinite(value) or value <= 0 for value in (frame_rate_hz,pixel_size_um,reported_bandwidth_hz)):
        raise ValueError("Physical metadata must be finite and positive")
    return {"frame_rate_hz":frame_rate_hz,"frame_interval_ms":1000/frame_rate_hz,"pixel_size_um":pixel_size_um,
        "metadata_provenance":"user confirmed acquisition50Hz and0.5um/pixel on2026-09-14; reported indicator bandwidth about30Hz",
        "reported_indicator_bandwidth_hz":reported_bandwidth_hz,"nyquist_hz":frame_rate_hz/2,
        "reported_bandwidth_exceeds_nyquist":reported_bandwidth_hz > frame_rate_hz/2,
        "indicator_identity":"unidentified","bandwidth_convention":"unidentified",
        "bandwidth_is_not_decay_time":True,"indicator_impulse_response_identified":False,
        "exposure_duration_independently_verified":False}


def _annotation_schema() -> dict[str, Any]:
    return {"schema_version":1,"coordinate_conventions":{
        "source_frames":"one-based inclusive; frame_grid.tsv is authoritative",
        "crop_bounds":"zero-based source columns/rows; half-open x0:x1,y0:y1",
        "expert_points":"x=column,y=row; numerical extraction rounds floor(coordinate+0.5)",
        "annotation_polygons":"global image-edge coordinates; pixel(column,row) has center(column+0.5,row+0.5); vertices may lie on crop edges",
        "polygon_xy_json":"JSON list of at least3 [x,y] vertices, no repeated closing vertex required"},
        "frame_reviews":{"fields":FRAME_REVIEW_FIELDS,"status_values":["pending","in_progress","complete"],
            "complete_meaning":"all sources and active fluorescence regions in the entire panel/frame inspected; uncertain areas explicitly masked"},
        "source_inventory":{"fields":SOURCE_FIELDS,"source_type_values":["cell_like","process","other_fluorescent_region","uncertain"],
            "identity_status_values":["accepted","uncertain"],"source_id_scope":"stable within a panel; not automatically a biological neuron"},
        "active_regions":{"fields":ACTIVE_FIELDS,"meaning":"one active fluorescent region per source per frame; source_id must exist in source inventory"},
        "uncertain_areas":{"fields":UNCERTAIN_FIELDS,"meaning":"exclude these pixels from evaluable material; never count them as negative"},
        "fluorescence_events":{"fields":EVENT_FIELDS,"onset_status_values":["bounded","left_censored","unidentified"],
            "meaning":"manual fluorescence timing only; distinct from spike onset and from algorithmically fitted calibration excursions"},
        "acceptance_required":{"accepted":True,"accepted_by":"human reviewer identifier","accepted_at":"timestamp",
            "complete_source_inventory":True,"complete_active_region_coverage":True,"uncertainty_explicit":True,
            "frame_grid_sha256":"SHA256 of immutable frame_grid.tsv",
            "annotation_file_sha256":"mapping of all mutable review filenames to their accepted bytes"},
        "real_precision_initially":None,"unmatched_sparse_candidates":"unknown_not_negative"}


def prepare_review_package(control_protocol_path: str | Path, output_dir: str | Path, *,
        calibration_ui: Sequence[int] = (1,1799), evaluation_ui: Sequence[int] = (1800,2359),
        detector_setup_ui: Sequence[int] = (1600,1799), conditioning_warmup_ui: Sequence[int] = (1400,1599),
        frame_rate_hz: float = 50.0, pixel_size_um: float = .5, reported_indicator_bandwidth_hz: float = 30.0) -> dict[str, Any]:
    """Create immutable review geometry and editable human annotation templates."""
    protocol_path,output = Path(control_protocol_path).resolve(),Path(output_dir).resolve()
    protocol = _read(protocol_path)
    source = _verify(protocol["source"])
    labels_path = _verify(protocol["labels"])
    experts = _rows(labels_path)
    if len(experts) != 79 or len({row["canonical_roi_id"] for row in experts}) != 26:
        raise ValueError("Expected the exact79-occurrence26-identity sparse cohort")
    cal_start,cal_stop = map(int,calibration_ui); eval_start,eval_stop = map(int,evaluation_ui)
    if not 1 <= cal_start <= cal_stop < eval_start <= eval_stop <= 2359:
        raise ValueError("Descriptive calibration must precede and not overlap evaluation")
    if not cal_start <= int(detector_setup_ui[0]) <= int(detector_setup_ui[1]) <= cal_stop:
        raise ValueError("Detector setup must be within pre-evaluation material")
    if not 1 <= int(conditioning_warmup_ui[0]) <= int(conditioning_warmup_ui[1]) < int(detector_setup_ui[0]):
        raise ValueError("Conditioning/history warmup must precede detector setup")
    panels = panel_geometry()
    physical = _physical_metadata(frame_rate_hz,pixel_size_um,reported_indicator_bandwidth_hz)
    for panel in panels:
        panel["area_um2"] = panel["width"]*panel["height"]*pixel_size_um**2
    contract = {"schema_version":1,"source_protocol":_binding(protocol_path),"source_movie":dict(protocol["source"]),
        "source_shape_tyx":[2359,340,573],"expert_source":dict(protocol["labels"]),"preparation_code":_binding(Path(__file__)),
        "physical":physical,"panels":panels,"descriptive_calibration_ui":[cal_start,cal_stop],
        "evaluation_ui":[eval_start,eval_stop],"detector_setup_ui":list(map(int,detector_setup_ui)),
        "conditioning_history_warmup_ui":list(map(int,conditioning_warmup_ui)),
        "panel_selection":"three adjacent128-square interiors centered from image geometry; no detector or label outcomes used",
        "filtering_contract":"parent processes full source field and available prior history, then evaluates these interiors; do not filter cropped panels",
        "history_sufficiency":"parent must compare realized maximum kernel lag/support to available warmup and spatial halo",
        "descriptive_scope":"known-center geometry is retrospective; fits use only descriptive calibration frames",
        "independent_confirmation":False,"accepted_exhaustive_annotation":False,"real_precision":None,
        "scientific_audit":"parent-owned complete expert/model/occurrence media; this preparation package does not replace it"}
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["contract"] != contract:
            raise ValueError("Review preparation contract changed; use a new output root")
        for record in manifest["immutable_artifacts"]:
            _verify(record)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an unrelated or incomplete review package")
    output.mkdir(parents=True,exist_ok=True)
    (output/"expert_occurrences.tsv").write_bytes(labels_path.read_bytes())
    grid = frame_grid(panels,evaluation_ui)
    _table(output/"panel_geometry.tsv",panels,list(panels[0]))
    _table(output/"frame_grid.tsv",grid,list(grid[0]))
    reviews = [{"panel_id":row["panel_id"],"source_frame_ui":row["source_frame_ui"],"status":"pending",
        "source_inventory_complete":False,"active_regions_complete":False,"uncertainty_explicit":False,
        "reviewer_id":"","reviewed_at":"","notes":""} for row in grid]
    _table(output/"frame_reviews.tsv",reviews,FRAME_REVIEW_FIELDS)
    for name,fields in (("source_inventory.tsv",SOURCE_FIELDS),("active_regions.tsv",ACTIVE_FIELDS),
                        ("uncertain_areas.tsv",UNCERTAIN_FIELDS),("fluorescence_events.tsv",EVENT_FIELDS)):
        _table(output/name,[],fields)
    character = [{"panel_id":p["panel_id"],"quiet_active_crowded_strata":"unreviewed","motion_or_artifact":"unreviewed","reviewer_id":"","notes":""} for p in panels]
    _table(output/"panel_characterization.tsv",character,list(character[0]))
    membership,locations = [],{}
    for expert in sorted(experts,key=lambda row:row["observation_id"]):
        x,y = float(expert["x_px"]),float(expert["y_px"])
        if not math.isfinite(x+y) or not 0 <= x < 573 or not 0 <= y < 340:
            raise ValueError("Known center lies outside the source image")
        coordinate = (x,y)
        if coordinate not in locations:
            locations[coordinate] = {"location_id":f"known_location_{len(locations)+1:03d}","x_px":x,"y_px":y,
                "anchor_observation_id":expert["observation_id"],"canonical_roi_ids":[],"observation_ids":[]}
        loc = locations[coordinate]
        loc["canonical_roi_ids"].append(expert["canonical_roi_id"]);loc["observation_ids"].append(expert["observation_id"])
        containing = [p["panel_id"] for p in panels if p["x0"] <= x < p["x1"] and p["y0"] <= y < p["y1"]]
        membership.append({"observation_id":expert["observation_id"],"canonical_roi_id":expert["canonical_roi_id"],
            "location_id":loc["location_id"],"x_px":x,"y_px":y,"panel_id":containing[0] if containing else "outside_fixed_panels",
            "sparse_label_not_exhaustive_truth":True})
    _table(output/"known_occurrence_panels.tsv",membership,list(membership[0]))
    _json(output/"known_center_locations.json",list(locations.values()))
    _json(output/"annotation_schema.json",_annotation_schema())
    _json(output/"annotation_acceptance.json",{"accepted":False,"accepted_by":"","accepted_at":"",
        "complete_source_inventory":False,"complete_active_region_coverage":False,"uncertainty_explicit":False,
        "frame_grid_sha256":_sha(output/"frame_grid.tsv"),"annotation_file_sha256":{}})
    instructions = ("# Exhaustive real-panel fluorescence review\n\n"
        "Inspect every panel/frame in frame_grid.tsv. The three panels were selected from geometry, not detector success. "
        "Prior sparse centers are hints, not a complete source list. Add every visible source and every active fluorescent region, "
        "including regions without an existing label. Keep uncertain identity, boundaries, and activity explicit in uncertain_areas.tsv.\n\n"
        "Use source_inventory.tsv for source presence/geometry, active_regions.tsv for per-frame active polygons, and "
        "fluorescence_events.tsv only for independently reviewed fluorescence timing. A source ID does not establish biological neuron identity. "
        "Polygon vertices use global image-edge coordinates; pixel centers are column+0.5,row+0.5. Raw point labels instead use x=column,y=row.\n\n"
        "Mark a frame complete only after inspecting all its material and recording every active region and uncertainty mask. "
        "Only accepted complete coverage can make the remaining reviewed material evaluable as negative. Pending or uncertain material stays unknown. "
        "A human acceptance record must bind the final annotation-file hashes, reviewer and timestamp. This module does not fill that approval automatically.\n\n"
        "Descriptive calibration uses UI1..1799; detector setup uses UI1600..1799 after conditioning/history warmup UI1400..1599. "
        "Evaluation uses UI1800..2359. The parent filters the full source field with prior history, then evaluates the interior crops. "
        "Check actual kernel support/history before calling those halos sufficient.\n\n"
        "50Hz and0.5um/pixel are user-confirmed. The reported approximately30Hz indicator bandwidth has an unidentified convention/indicator "
        "and exceeds the25Hz Nyquist frequency. It is not a decay constant. Descriptive raw-core-minus-background fits describe sampled "
        "fluorescence, not an indicator impulse response or spike onset. No real precision is available from the original79 sparse labels.\n")
    (output/"REVIEW_INSTRUCTIONS.md").write_text(instructions)
    immutable = ("expert_occurrences.tsv","panel_geometry.tsv","frame_grid.tsv","known_occurrence_panels.tsv",
                 "known_center_locations.json","annotation_schema.json","REVIEW_INSTRUCTIONS.md")
    manifest = {"status":"PREPARED_ANNOTATION_PENDING","contract":contract,
        "expected_panel_count":len(panels),"expected_panel_frame_tasks":len(grid),
        "known_occurrence_count":len(experts),"known_identity_count":26,"unique_known_coordinate_count":len(locations),
        "immutable_artifacts":[_binding(output/name) for name in immutable],
        "mutable_human_review_files":list(MUTABLE_REVIEW_FILES),"real_precision":None,
        "accepted_exhaustive_annotation":False,"scientific_audit_complete":False}
    _json(manifest_path,manifest)
    return manifest


def _truth(value: Any) -> bool:
    if value is True or value in ("True","true","1",1):return True
    if value is False or value in ("False","false","0",0,""):return False
    raise ValueError("Review flags must be explicit booleans")


def _polygon(value: str, panel: Mapping[str, Any]) -> list[list[float]]:
    vertices = json.loads(value)
    if not isinstance(vertices,list) or len(vertices) < 3:
        raise ValueError("An annotation polygon needs at least three vertices")
    points = []
    for vertex in vertices:
        if not isinstance(vertex,list) or len(vertex) != 2:
            raise ValueError("Polygon vertices must be [x,y] pairs")
        x,y = map(float,vertex)
        if not math.isfinite(x+y) or not panel["x0"] <= x <= panel["x1"] or not panel["y0"] <= y <= panel["y1"]:
            raise ValueError("Annotation polygon lies outside its declared panel")
        points.append([x,y])
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2
    if area <= 0:raise ValueError("Annotation polygon has zero area")
    return points


def _polygon_mask(points: Sequence[Sequence[float]], panel: Mapping[str, Any]) -> Any:
    import numpy as np
    yy,xx = np.mgrid[panel["y0"]:panel["y1"],panel["x0"]:panel["x1"]]
    xx,yy = xx+.5,yy+.5
    mask = np.zeros(xx.shape,dtype=bool)
    # Deterministic even-odd fill evaluated at the declared pixel centers.
    for first,second in zip(points,list(points[1:])+[points[0]]):
        x1,y1 = first;x2,y2 = second
        if y1 == y2:continue
        crossing = ((y1 > yy) != (y2 > yy)) & (xx < (x2-x1)*(yy-y1)/(y2-y1)+x1)
        mask ^= crossing
    return mask


def validate_annotation_coverage(package_root: str | Path) -> dict[str, Any]:
    """Validate human acceptance and coverage; never calculate real precision."""
    root = Path(package_root).resolve();manifest = _read(root/"manifest.json")
    for record in manifest["immutable_artifacts"]:_verify(record)
    panels = {p["panel_id"]:p for p in manifest["contract"]["panels"]}
    grid = _rows(root/"frame_grid.tsv");expected = {(r["panel_id"],int(r["source_frame_ui"])) for r in grid}
    reviews = _rows(root/"frame_reviews.tsv")
    by_frame = {(r["panel_id"],int(r["source_frame_ui"])):r for r in reviews}
    if len(by_frame) != len(reviews) or set(by_frame) != expected:
        raise ValueError("Review rows must cover the immutable panel/frame grid exactly once")
    complete = sum(row["status"] == "complete" and all(_truth(row[key]) for key in
        ("source_inventory_complete","active_regions_complete","uncertainty_explicit")) and bool(row["reviewer_id"].strip()) and bool(row["reviewed_at"].strip()) for row in reviews)
    acceptance = _read(root/"annotation_acceptance.json")
    base = {"panel_frame_tasks":len(grid),"complete_panel_frame_tasks":complete,
        "real_precision":None,"real_precision_computed":False,"framewise_classification_unit":"candidate versus active fluorescence region within a fully reviewed frame",
        "uncertain_material_is_negative":False,"independent_confirmation":False}
    if not _truth(acceptance.get("accepted",False)):
        return {**base,"status":"ANNOTATION_PENDING","accepted_exhaustive_coverage":False,"real_precision_eligible":False}
    if complete != len(grid) or not acceptance.get("accepted_by") or not acceptance.get("accepted_at"):
        raise ValueError("Accepted review needs complete frame coverage and a named, dated human acceptance")
    if not all(_truth(acceptance.get(key,False)) for key in ("complete_source_inventory","complete_active_region_coverage","uncertainty_explicit")):
        raise ValueError("Human acceptance does not attest complete source/activity/uncertainty review")
    if acceptance.get("frame_grid_sha256") != _sha(root/"frame_grid.tsv"):
        raise ValueError("Accepted frame grid differs")
    hashes = acceptance.get("annotation_file_sha256",{})
    if set(hashes) != set(MUTABLE_REVIEW_FILES) or any(_sha(root/name) != digest for name,digest in hashes.items()):
        raise ValueError("Accepted annotation file hashes differ; acceptance must be renewed")
    first,last = manifest["contract"]["evaluation_ui"]
    sources = {}
    for row in _rows(root/"source_inventory.tsv"):
        key = (row["panel_id"],row["source_id"])
        if key in sources or key[0] not in panels or not key[1] or not row["reviewer_id"]:
            raise ValueError("Invalid or duplicated source inventory record")
        if not first <= int(row["presence_start_ui"]) <= int(row["presence_stop_ui"]) <= last:
            raise ValueError("Source presence lies outside evaluation")
        if row["identity_status"] not in {"accepted","uncertain"}:
            raise ValueError("Source identity status must be explicit")
        _polygon(row["polygon_xy_json"],panels[key[0]]);sources[key] = row
    region_ids,source_frames = set(),set()
    for row in _rows(root/"active_regions.tsv"):
        panel,frame,source = row["panel_id"],int(row["source_frame_ui"]),row["source_id"]
        key = (panel,frame,row["active_region_id"]);source_key = (panel,source)
        if (panel,frame) not in expected or key in region_ids or source_key not in sources or not key[2] or not row["reviewer_id"]:
            raise ValueError("Active region lacks unique identity, reviewed frame, or source inventory")
        if (panel,frame,source) in source_frames:
            raise ValueError("One source cannot create duplicate active-region decisions in one frame")
        src = sources[source_key]
        if not int(src["presence_start_ui"]) <= frame <= int(src["presence_stop_ui"]):
            raise ValueError("Active region falls outside its source's presence interval")
        _polygon(row["polygon_xy_json"],panels[panel]);region_ids.add(key);source_frames.add((panel,frame,source))
    uncertainty,uncertain_ids = {},set()
    for row in _rows(root/"uncertain_areas.tsv"):
        key = (row["panel_id"],int(row["source_frame_ui"]))
        identifier = (*key,row["uncertainty_id"])
        if key not in expected or identifier in uncertain_ids or not identifier[-1] or not row["reason"] or not row["reviewer_id"]:
            raise ValueError("Invalid uncertainty annotation")
        uncertainty.setdefault(key,[]).append(_polygon(row["polygon_xy_json"],panels[key[0]]));uncertain_ids.add(identifier)
    import numpy as np
    excluded,total = 0,0
    for panel_id,frame in sorted(expected):
        panel = panels[panel_id];mask = np.zeros((panel["height"],panel["width"]),dtype=bool)
        for polygon in uncertainty.get((panel_id,frame),[]):mask |= _polygon_mask(polygon,panel)
        total += int(mask.size);excluded += int(mask.sum())
    events = _rows(root/"fluorescence_events.tsv");event_ids = set()
    bounded_onsets = 0
    for row in events:
        key = (row["panel_id"],row["source_id"]);identifier = (*key,row["event_id"])
        if key not in sources or identifier in event_ids or not identifier[-1] or not row["reviewer_id"]:
            raise ValueError("Invalid manual fluorescence-event record")
        if not first <= int(row["active_start_ui"]) <= int(row["active_stop_ui"]) <= last:
            raise ValueError("Fluorescence event lies outside review coverage")
        if row["onset_status"] == "bounded":
            if not first <= int(row["onset_lower_ui"]) <= int(row["onset_upper_ui"]) <= int(row["active_stop_ui"]):
                raise ValueError("Invalid independently annotated onset interval")
            bounded_onsets += 1
        elif row["onset_status"] not in {"left_censored","unidentified"}:
            raise ValueError("Fluorescence onset status must be explicit")
        event_ids.add(identifier)
    return {**base,"status":"ACCEPTED_EXHAUSTIVE_COVERAGE","accepted_exhaustive_coverage":True,
        "real_precision_eligible":total > excluded,"reviewed_pixel_frames":total,
        "uncertain_excluded_pixel_frames":excluded,"evaluable_pixel_frames":total-excluded,
        "active_region_frame_count":len(region_ids),"source_inventory_count":len(sources),
        "manually_bounded_fluorescence_onsets":bounded_onsets,"acceptance":_binding(root/"annotation_acceptance.json"),
        "metric_requirement":"downstream evaluator must exclude uncertainty masks and preserve one-to-one per-frame assignments"}


def _exponential_fit(values: Any, dt: float, kind: str) -> dict[str, Any]:
    """Bounded profile fit, used only after isolation/censoring checks."""
    import numpy as np
    if kind not in {"rise","decay"} or not math.isfinite(dt) or dt <= 0:
        raise ValueError("A rise/decay model and positive sampling interval are required")
    y = np.asarray(values,dtype=float);duration = (len(y)-1)*dt
    failed = {"status":"unidentified","tau_ms":None,"amplitude":None,"r_squared":None}
    if len(y) < 5 or duration < 4*dt or not np.isfinite(y).all() or y.max() <= 0:
        return {**failed,"reason":"insufficient_native_samples"}
    times = np.arange(len(y))*dt;taus = np.geomspace(dt/2,max(10*duration,4*dt),180)
    models = 1-np.exp(-times[None,:]/taus[:,None]) if kind == "rise" else np.exp(-times[None,:]/taus[:,None])
    amplitudes = (models@y)/np.sum(models*models,axis=1)
    losses = np.sum((amplitudes[:,None]*models-y)**2,axis=1)
    best = int(np.argmin(losses));variation = float(np.sum((y-y.mean())**2))
    r2 = 1-float(losses[best])/variation if variation > 0 else float("-inf")
    if best in (0,len(taus)-1) or taus[best] < dt:
        return {**failed,"reason":"subframe_or_parameter_boundary"}
    if r2 < .85 or not .8*float(y.max()) <= amplitudes[best] <= 2*float(y.max()):
        return {**failed,"reason":"poor_exponential_shape_or_unidentified_amplitude"}
    near = taus[losses <= losses[best]*1.1+np.finfo(float).eps]
    if len(near) and near[-1]/near[0] > 4:
        return {**failed,"reason":"broad_parameter_profile"}
    return {"status":"descriptive_fit","tau_ms":float(taus[best]*1000),"amplitude":float(amplitudes[best]),
        "r_squared":float(r2),"reason":"sampled_fluorescence_fit_not_indicator_irf",
        "model_assumptions":"fixed measured baseline and crossing/peak time; single exponential; no deconvolution or spike-time inference",
        "profile_10percent_loss_tau_range_ms":[float(near[0]*1000),float(near[-1]*1000)],"sample_count":len(y)}


def fit_isolated_transients(trace: Any, *, source_start_ui: int, frame_rate_hz: float = 50.0) -> dict[str, Any]:
    """Describe all excursions; do not manufacture fits when events are absent."""
    import numpy as np
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0 or source_start_ui < 1:
        raise ValueError("Positive source frame and frame rate are required")
    values = np.asarray(trace,dtype=float)
    if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("A finite calibration trace of at least20samples is required")
    baseline = float(np.median(values));differences = np.diff(values)
    spread = float(1.4826*np.median(np.abs(differences-np.median(differences)))/math.sqrt(2))
    common = {"baseline":baseline,"variability_proxy":spread,"proxy_definition":"MAD of first differences /sqrt2; not an independently measured sensor-noise model",
        "frame_interval_ms":1000/frame_rate_hz,"indicator_impulse_response_identified":False}
    if not spread > 0:
        return {**common,"status":"unidentified","reason":"zero_calibration_variability_or_no_resolved_signal","events":[]}
    signal = values-baseline;above = signal > 3*spread
    starts = np.flatnonzero(above & ~np.r_[False,above[:-1]])
    stops = np.flatnonzero(above & ~np.r_[above[1:],False])
    groups = [(int(start),int(stop)) for start,stop in zip(starts,stops) if float(signal[start:stop+1].max()) >= 5*spread]
    events = []
    for event_index,(start,stop) in enumerate(groups):
        peak = start+int(np.argmax(signal[start:stop+1]));amplitude = float(signal[peak])
        before = np.flatnonzero(signal[:start] <= 2*spread);after = np.flatnonzero(signal[peak+1:] <= 2*spread)
        left = int(before[-1]) if len(before) else None
        right = peak+1+int(after[0]) if len(after) else None
        left_censored = left is None or left < 5
        right_censored = right is None or right+5 >= len(signal)
        isolated = (left is not None and right is not None and
            (event_index == 0 or groups[event_index-1][1] < left-10) and
            (event_index == len(groups)-1 or groups[event_index+1][0] > right+10))
        quiet_before = left is not None and left >= 5 and bool(np.all(np.abs(signal[left-5:left]) <= 3*spread))
        row = {"event_id":f"cal_excursion_{event_index+1:03d}","threshold_start_ui":source_start_ui+start,
            "threshold_stop_ui":source_start_ui+stop,"peak_frame_ui":source_start_ui+peak,
            "left_baseline_frame_ui":source_start_ui+left if left is not None else None,
            "right_baseline_frame_ui":source_start_ui+right if right is not None else None,
            "peak_above_baseline":amplitude,"peak_over_variability":amplitude/spread,
            "left_censored":left_censored,"right_censored":right_censored,"isolation_pass":isolated and quiet_before,
            "event_selection":"all calibration excursions above3x variability with peak>=5x; no evaluation labels used",
            "true_onset_identified":False,"indicator_impulse_response_identified":False}
        failure = {"status":"unidentified","tau_ms":None,"reason":"censored_overlapping_or_no_stable_prebaseline"}
        row["rise_fit"],row["decay_fit"] = dict(failure),dict(failure)
        if isolated and quiet_before and not left_censored and not right_censored:
            rising = signal[left:peak+1]
            ten,ninety = np.flatnonzero(rising >= .1*amplitude),np.flatnonzero(rising >= .9*amplitude)
            bins = int(ninety[0]-ten[0]) if len(ten) and len(ninety) else None
            row["sampled_rise_10_90_ms"] = bins*1000/frame_rate_hz if bins is not None and bins > 1 else None
            row["fast_rise_unresolved"] = bins is None or bins <= 1
            row["rise_fit"] = ({"status":"unidentified","tau_ms":None,"reason":"rise_unresolved_at_native20ms_sampling"}
                               if row["fast_rise_unresolved"] else _exponential_fit(rising,1/frame_rate_hz,"rise"))
            decay = signal[peak:right+1]
            row["decay_fit"] = (_exponential_fit(decay,1/frame_rate_hz,"decay") if decay[-1] <= .25*amplitude else
                {"status":"unidentified","tau_ms":None,"reason":"decay_did_not_span_sufficient_amplitude"})
        else:
            row["sampled_rise_10_90_ms"] = None;row["fast_rise_unresolved"] = None
        events.append(row)
    return {**common,"status":"described" if events else "unidentified",
        "reason":"calibration_excursions_only" if events else "no_calibration_transient_passed_amplitude_guard","events":events}


def radial_profile(image: Any, *, center_x: float, center_y: float, pixel_size_um: float,
                   maximum_radius_px: int = 12, activity_image: bool = True) -> dict[str, Any]:
    """Azimuthal mean about the fixed known center; width is descriptive only."""
    import numpy as np
    values = np.asarray(image,dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or pixel_size_um <= 0:
        raise ValueError("A finite image and positive pixel size are required")
    yy,xx = np.indices(values.shape);radius = np.hypot(xx-center_x,yy-center_y)
    shells = []
    for bin_index in range(maximum_radius_px+1):
        mask = radius < .5 if bin_index == 0 else (radius >= bin_index-.5)&(radius < bin_index+.5)
        shells.append({"radius_px":bin_index,"radius_um":bin_index*pixel_size_um,"sample_count":int(mask.sum()),
                       "mean_native":float(values[mask].mean()) if mask.any() else None})
    summary = {"status":"unidentified","fwhm_um":None,"fwhm_px":None,
        "reason":"static_brightness_not_activity_footprint" if not activity_image else "no_positive_center_response",
        "profile_rows":shells,"interpretation":"radial fluorescence width about a known center; not soma diameter or a fitted source identity"}
    central = shells[0]["mean_native"]
    if not activity_image or central is None or central <= 0:return summary
    means = [row["mean_native"] for row in shells]
    if any(value is None for value in means):return {**summary,"reason":"spatially_censored_profile"}
    if int(np.argmax(means)) > 2:return {**summary,"reason":"peak_displaced_from_known_center"}
    half = central/2
    crossing = next((index for index in range(1,len(means)) if means[index] <= half),None)
    if crossing is None:return {**summary,"reason":"half_height_outside_profile_support"}
    left,right = means[crossing-1],means[crossing]
    radial_half = crossing-1+(left-half)/(left-right)
    if radial_half < 1:return {**summary,"reason":"width_insufficiently_sampled"}
    return {**summary,"status":"described","fwhm_px":float(2*radial_half),
        "fwhm_um":float(2*radial_half*pixel_size_um),"reason":"first_radial_half_height_crossing_linear_interpolation"}


def describe_calibration(package_root: str | Path) -> dict[str, Any]:
    """Read one small known-center patch at a time, strictly before evaluation.

    The full movie is memory-mapped only in this explicit pass. Every output trace
    retains every declared calibration sample; no application sample enters fits.
    """
    root = Path(package_root).resolve();prepared = _read(root/"manifest.json")
    for record in prepared["immutable_artifacts"]:_verify(record)
    contract = prepared["contract"]
    source = _verify(contract["source_movie"])
    _verify(contract["preparation_code"])
    output = root/"descriptives";manifest_path = output/"manifest.json"
    desc_contract = {"preparation_manifest":_binding(root/"manifest.json"),"code":_binding(Path(__file__)),
        "calibration_ui":contract["descriptive_calibration_ui"],"evaluation_ui":contract["evaluation_ui"],
        "source_movie":contract["source_movie"],"core_radius_px":2,"background_annulus_px":[8,12],
        "trace":"mean raw pixels at radius<=2 minus median raw pixels at8<=radius<=12 around rounded known center",
        "footprint_selection":"first chronological isolated uncensored calibration excursion; all excursions retained in kinetics JSON",
        "fit_guards":">=5 native samples; fitted tau>=1 sample; r2>=.85; bounded amplitude and parameter profile; uncensored isolated excursion with stable prebaseline",
        "indicator_impulse_response_identified":False,"spike_onset_identified":False,"independent_confirmation":False,
        "scientific_audit":"descriptive numeric supplement; parent retains complete scientific media audit"}
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["contract"] != desc_contract:raise ValueError("Descriptive contract changed; use a new package")
        for record in manifest["artifacts"]:_verify(record)
        return manifest
    import numpy as np
    movie = np.load(source,mmap_mode="r",allow_pickle=False)
    if list(movie.shape) != contract["source_shape_tyx"] or movie.ndim != 3:
        raise ValueError("Bound raw movie shape differs from the preparation contract")
    first,last = map(int,contract["descriptive_calibration_ui"])
    if last >= int(contract["evaluation_ui"][0]):raise ValueError("Descriptive fit overlaps evaluation")
    fps,pixel_um = (float(contract["physical"][key]) for key in ("frame_rate_hz","pixel_size_um"))
    output.mkdir(parents=True,exist_ok=True)
    summaries,artifacts = [],[]
    for location in _read(root/"known_center_locations.json"):
        x,y = (int(math.floor(float(location[key])+.5)) for key in ("x_px","y_px"))
        if not 0 <= x < movie.shape[2] or not 0 <= y < movie.shape[1]:raise ValueError("Rounded center outside movie")
        x0,x1,y0,y1 = max(0,x-12),min(movie.shape[2],x+13),max(0,y-12),min(movie.shape[1],y+13)
        patch = np.asarray(movie[first-1:last,y0:y1,x0:x1],dtype=np.float32)
        if not np.isfinite(patch).all():raise ValueError("Nonfinite raw calibration pixels")
        yy,xx = np.indices(patch.shape[1:]);radius = np.hypot(xx-(x-x0),yy-(y-y0))
        core,annulus = radius <= 2,(radius >= 8)&(radius <= 12)
        if not core.any() or not annulus.any():raise ValueError("Known-center extraction has empty core/background")
        core_mean = patch[:,core].mean(axis=1,dtype=np.float64)
        background = np.median(patch[:,annulus],axis=1).astype(float)
        trace = core_mean-background
        kinetics = fit_isolated_transients(trace,source_start_ui=first,frame_rate_hz=fps)
        spatially_censored = (x0 != x-12 or x1 != x+13 or y0 != y-12 or y1 != y+13)
        selected = next((event for event in kinetics["events"] if event["isolation_pass"] and
                         not event["left_censored"] and not event["right_censored"]),None)
        if selected is None:
            footprint_image = np.median(patch,axis=0)
            footprint_image = footprint_image-float(np.median(footprint_image[annulus]))
            footprint_frames = {"kind":"static_calibration_brightness_only","activity_event_id":None,
                                "calibration_ui":[first,last],"reason":"no_isolated_uncensored_calibration_excursion"}
        else:
            peak = int(selected["peak_frame_ui"])-first;left = int(selected["left_baseline_frame_ui"])-first
            peak_rows = list(range(max(0,peak-1),min(len(patch),peak+2)));baseline_rows = list(range(left-5,left))
            footprint_image = patch[peak_rows].mean(axis=0,dtype=np.float64)-patch[baseline_rows].mean(axis=0,dtype=np.float64)
            footprint_image -= float(np.median(footprint_image[annulus]))
            footprint_frames = {"kind":"calibration_transient_increment","activity_event_id":selected["event_id"],
                "peak_source_frames_ui":[first+row for row in peak_rows],"baseline_source_frames_ui":[first+row for row in baseline_rows]}
        profile = radial_profile(footprint_image,center_x=x-x0,center_y=y-y0,pixel_size_um=pixel_um,activity_image=selected is not None)
        if spatially_censored:profile.update(status="unidentified",fwhm_um=None,fwhm_px=None,reason="source_field_boundary_censors_extraction")
        location_dir = output/location["location_id"];location_dir.mkdir(exist_ok=True)
        trace_rows = [{"source_frame_ui":first+index,"source_numpy_index":first+index-1,
            "time_from_recording_start_s":(first+index-1)/fps,"raw_core_mean":float(core_mean[index]),
            "raw_annulus_median":float(background[index]),"raw_core_minus_background":float(trace[index])} for index in range(len(trace))]
        _table(location_dir/"calibration_trace.tsv",trace_rows,list(trace_rows[0]))
        _table(location_dir/"radial_profile.tsv",profile["profile_rows"],list(profile["profile_rows"][0]))
        _json(location_dir/"description.json",{"location":location,"rounded_center_xy":[x,y],"patch_xyxy":[x0,y0,x1,y1],
            "source_frames_ui":[first,last],"native_units":"raw acquisition fluorescence intensity; not calibrated photon counts",
            "spatially_censored":spatially_censored,"core_pixel_count":int(core.sum()),"background_pixel_count":int(annulus.sum()),
            "kinetics":kinetics,"footprint_frames":footprint_frames,"footprint":{key:value for key,value in profile.items() if key != "profile_rows"}})
        summaries.append({"location_id":location["location_id"],"anchor_observation_id":location["anchor_observation_id"],
            "x_px":location["x_px"],"y_px":location["y_px"],"rounded_x_px":x,"rounded_y_px":y,
            "calibration_start_ui":first,"calibration_stop_ui":last,"spatially_censored":spatially_censored,
            "calibration_excursion_count":len(kinetics["events"]),"rise_fit_count":sum(event["rise_fit"]["status"] == "descriptive_fit" for event in kinetics["events"]),
            "decay_fit_count":sum(event["decay_fit"]["status"] == "descriptive_fit" for event in kinetics["events"]),
            "unresolved_fast_rise_count":sum(event["fast_rise_unresolved"] is True for event in kinetics["events"]),
            "footprint_kind":footprint_frames["kind"],"footprint_status":profile["status"],"footprint_fwhm_um":profile["fwhm_um"],
            "indicator_impulse_response_identified":False})
        artifacts.extend(_binding(location_dir/name) for name in ("calibration_trace.tsv","radial_profile.tsv","description.json"))
    _table(output/"summary.tsv",summaries,list(summaries[0]));artifacts.append(_binding(output/"summary.tsv"))
    manifest = {"status":"CALIBRATION_DESCRIPTIVES_COMPLETE","contract":desc_contract,"artifacts":artifacts,
        "location_count":len(summaries),"all_known_coordinate_variants_retained":True,"sample_count_per_trace":last-first+1,
        "real_precision":None,"indicator_impulse_response_identified":False,"scientific_audit_complete":False}
    _json(manifest_path,manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command",required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--control-protocol",type=Path,required=True);prepare.add_argument("--output",type=Path,required=True)
    for command in ("describe","validate-annotations"):
        subparsers.add_parser(command).add_argument("--package-root",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "prepare":result = prepare_review_package(args.control_protocol,args.output)
    elif args.command == "describe":result = describe_calibration(args.package_root)
    else:result = validate_annotation_coverage(args.package_root)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":main()
