"""Deterministic A-versus-Z occurrence panels from a sealed control replay.

This is postprocessing only: it neither scores frames nor alters candidate
selection. Heavy dependencies and arrays are loaded only by ``run``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ARM_A = "s1_a0.4_difference_A"
ARM_Z = "s1_a0.4_difference_Z"
ARM_C = "s1_a0.4_difference_contrast"
OPERATOR = "s1_a0.4_difference__point__gamma__0__square"
STRATA = ("both_matched", "A_only", "Z_only", "neither_matched")
CONTEXT = 5


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _digest(path), "size_bytes": path.stat().st_size}


def _verify(record: Mapping[str, Any]) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or _digest(path) != record["sha256"]:
        raise ValueError(f"Sealed source or artifact changed: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    fields = list(fields or rows[0])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _boolean(value: Any) -> bool:
    if value is True or value in ("True", "true", "1", 1):
        return True
    if value is False or value in ("False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid matched value: {value!r}")


def _indexed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["observation_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate observation ID")
    return result


def stratify_occurrences(experts: Sequence[Mapping[str, Any]], a_rows: Sequence[Mapping[str, Any]],
                        z_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the full cohort; choose first observation ID in each outcome stratum."""
    truth, a_by_id, z_by_id = map(_indexed, (experts, a_rows, z_rows))
    if not truth or set(truth) != set(a_by_id) or set(truth) != set(z_by_id):
        raise ValueError("A, Z, and expert occurrence inventories must agree exactly")
    membership = []
    for observation in sorted(truth):
        expert, a, z = truth[observation], a_by_id[observation], z_by_id[observation]
        for outcome in (a, z):
            for key in ("canonical_roi_id", "burst_id", "x_px", "y_px"):
                if key in ("x_px", "y_px"):
                    equal = float(outcome[key]) == float(expert[key])
                else:
                    equal = str(outcome[key]) == str(expert[key])
                if not equal:
                    raise ValueError(f"Occurrence geometry/identity mismatch: {observation} {key}")
            if (int(outcome["window_start_frame_ui"]), int(outcome["window_stop_frame_ui"])) != (
                    int(expert["source_start_ui"]), int(expert["source_stop_ui"])):
                raise ValueError(f"Occurrence window mismatch: {observation}")
        am, zm = _boolean(a["matched"]), _boolean(z["matched"])
        stratum = "both_matched" if am and zm else "A_only" if am else "Z_only" if zm else "neither_matched"
        membership.append({"observation_id": observation, "canonical_roi_id": str(expert["canonical_roi_id"]),
            "burst_id": int(expert["burst_id"]), "x_px": float(expert["x_px"]), "y_px": float(expert["y_px"]),
            "window_start_frame_ui": int(expert["source_start_ui"]), "window_stop_frame_ui": int(expert["source_stop_ui"]),
            "A_matched": am, "Z_matched": zm, "stratum": stratum,
            "A_matched_site_id": a["matched_site_id"], "Z_matched_site_id": z["matched_site_id"],
            "A_nearest_site_id": a["nearest_site_id"], "Z_nearest_site_id": z["nearest_site_id"]})
    selection = []
    for stratum in STRATA:
        members = [row for row in membership if row["stratum"] == stratum]
        selection.append({"stratum": stratum, "occurrence_count": len(members),
            "selection_status": "selected" if members else "empty_stratum",
            "observation_id": members[0]["observation_id"] if members else ""})
    selected = {row["observation_id"] for row in selection if row["observation_id"]}
    for row in membership:
        row["selected_for_panel"] = row["observation_id"] in selected
    return membership, selection


def frame_window(start: int, stop: int, available_start: int, available_stop: int,
                 context: int = CONTEXT) -> list[int]:
    if context < 0 or not available_start <= start <= stop <= available_stop:
        raise ValueError("Declared burst must be fully inside the source interval")
    return list(range(max(available_start, start-context), min(available_stop, stop+context)+1))


def _compare_occurrence_table(saved: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> None:
    actual = _indexed(saved)
    if set(actual) != set(_indexed(expected)):
        raise ValueError("Saved occurrence table differs from the recomputed candidate join")
    for row in expected:
        other = actual[str(row["observation_id"])]
        if set(other) != set(row):
            raise ValueError("Saved occurrence schema differs from the bound evaluator")
        for key, value in row.items():
            found = other[key]
            if value is None:
                equal = found in (None, "")
            elif isinstance(value, bool):
                equal = _boolean(found) == value
            elif isinstance(value, (int, float)):
                equal = float(found) == value
            else:
                equal = str(found) == str(value)
            if not equal:
                raise ValueError(f"Saved occurrence table disagrees at {row['observation_id']} {key}")


def _threshold(calibration: Mapping[str, Any]) -> float:
    points = [row for row in calibration["operating_points"] if float(row["target_proposals_per_frame"]) == 1.0]
    if len(points) != 1 or not math.isfinite(float(points[0]["threshold_z"])):
        raise ValueError("Exactly one finite frozen q1 threshold is required")
    return float(points[0]["threshold_z"])


def _verify_panel_inventory(folder: Path, manifest: Mapping[str, Any], selection: Sequence[Mapping[str, Any]]) -> None:
    if [row["stratum"] for row in selection] != list(STRATA) or sum(int(row["occurrence_count"]) for row in selection) != 79:
        raise ValueError("The four strata must account for the full 79-occurrence inventory")
    if any((int(row["occurrence_count"]) > 0) != (row["selection_status"] == "selected") for row in selection):
        raise ValueError("Empty strata and selected examples disagree")
    selected = {row["observation_id"] for row in selection if row["selection_status"] == "selected"}
    if manifest.get("full_cohort_occurrence_count") != 79 or manifest.get("panel_count") != len(selected):
        raise ValueError("Completed panel inventory has incorrect cardinality")
    saved_ids = manifest.get("selected_occurrence_ids", [])
    if len(saved_ids) != len(selected) or set(saved_ids) != selected or manifest.get("selection") != list(selection):
        raise ValueError("Completed panel selection differs from the frozen strata")
    expected = {folder/name for name in ("panel_contract.json","all79_strata.tsv","selection.tsv","README.md")}
    for row in selection:
        if row["selection_status"] == "selected":
            expected.update(folder/f"{row['stratum']}__{row['observation_id']}{suffix}" for suffix in (".png",".json",".tsv"))
    records = manifest.get("artifacts", [])
    actual = {Path(record["path"]).resolve() for record in records}
    if len(records) != len(actual) or actual != {path.resolve() for path in expected}:
        raise ValueError("Completed panel artifact inventory is incomplete or unexpected")
    for record in records:
        _verify(record)


def _bound_record(records: Sequence[Mapping[str, Any]], path: Path) -> Mapping[str, Any]:
    matches = [record for record in records if Path(record["path"]).resolve() == path.resolve()]
    if len(matches) != 1:
        raise ValueError(f"Expected one upstream seal for {path}")
    _verify(matches[0])
    return matches[0]


def load_sources(root: Path) -> dict[str, Any]:
    """Verify source metadata and small tables before any dense array is loaded."""
    protocol_path, candidate_seal_path = root/"protocol.json", root/"candidate_seal.json"
    protocol, candidate_seal = _read_json(protocol_path), _read_json(candidate_seal_path)
    if protocol.get("source_frames_ui") != [1800,2359] or protocol.get("application_ui") != [1900,2359]:
        raise ValueError("This panel definition requires the frozen 560-frame control replay")
    completion_path = root/"evaluation_complete.json"
    completion = _read_json(completion_path)
    if completion.get("status") != "NUMERICAL_PASS" or completion.get("arm_count") != 28:
        raise ValueError("A numerically complete 28-arm root is required")
    for record in protocol["code_bindings"]:
        _verify(record)
    label_path = _verify(protocol["labels"])
    experts = _rows(label_path)
    if len(experts) != 79 or len({row["canonical_roi_id"] for row in experts}) != 26:
        raise ValueError("Expected all 79 occurrences and 26 expert identities")
    for record in candidate_seal["seals"]:
        _verify(record)
    sources = [_record(path) for path in (protocol_path,candidate_seal_path,completion_path,label_path)]
    seals = {}
    for arm in (ARM_A,ARM_Z,ARM_C):
        folder = root/"arms"/arm
        path = folder/"sealed.json"
        record = _bound_record(candidate_seal["seals"],path)
        sources.append(dict(record))
        seal = _read_json(path)
        if seal["arm"]["arm_id"] != arm:
            raise ValueError("Arm identity differs from its directory")
        # Verify every file in these three frozen seals, including all q rows.
        for record in seal["files"]:
            _verify(record)
            sources.append(dict(record))
        seals[arm] = seal
    operator_path = root/"operators"/OPERATOR/"complete.json"
    operator = _read_json(operator_path)
    sources.append(_record(operator_path))
    if (operator["spec"]["design"],operator["spec"]["reference_family"],operator["spec"]["guard_radius_px"],operator["spec"]["support_geometry"]) != ("point","gamma",0,"square"):
        raise ValueError("Unexpected spatial operator")
    stage_records = {"Raw": seals[ARM_A]["stages"]["Raw"], **operator["stages"]}
    if set(stage_records) != {"Raw","A","M","sigma","contrast","Z"}:
        raise ValueError("All six exact stage sources are required")
    for arm, readout in ((ARM_A,"A"),(ARM_Z,"Z"),(ARM_C,"contrast")):
        seal = seals[arm]
        if seal["stages"]["Score"] != stage_records[readout] or seal["stages"]["Mean"] != stage_records["M"]:
            raise ValueError("Shared operator stages do not agree with the frozen arm")
        if seal["stages"]["Raw"] != stage_records["Raw"] or seal["stages"]["Input"] != operator["source"]:
            raise ValueError("Arm input/source history differs")
        if float(seal["scale_floor"]) != float(operator["scale_floor"]):
            raise ValueError("Inconsistent local-spread floor")
    thresholds, outcomes = {}, {}
    from .two_stencil_evaluation import evaluate_occurrence_windows
    windows = {int(key): value for key,value in protocol["burst_windows"].items()}
    for arm in (ARM_A,ARM_Z):
        folder = root/"arms"/arm
        thresholds[arm] = _threshold(_read_json(folder/"calibration.json"))
        candidates = _rows(folder/"q1/candidates.tsv")
        result = evaluate_occurrence_windows(candidates,experts,burst_intervals_ui=windows)
        saved_path = folder/"q1/occurrence_rows.tsv"
        _compare_occurrence_table(_rows(saved_path),result["occurrence_rows"])
        summary_path = folder/"q1/summary.json"
        if _read_json(summary_path) != result["summary"]:
            raise ValueError("Saved q1 summary differs from the candidate join")
        sources.extend((_record(saved_path),_record(summary_path)))
        outcomes[arm] = result["occurrence_rows"]
    membership, selection = stratify_occurrences(experts,outcomes[ARM_A],outcomes[ARM_Z])
    return {"protocol":protocol,"sources":sources,"stage_records":stage_records,
        "operator":operator,"thresholds":thresholds,"membership":membership,"selection":selection}


def case_traces(arrays: Mapping[str, Any], row: Mapping[str, Any], *, source_start: int,
                source_stop: int, floor: float, epsilon: float) -> dict[str, Any]:
    import numpy as np
    frames = frame_window(int(row["window_start_frame_ui"]),int(row["window_stop_frame_ui"]),source_start,source_stop)
    x,y = float(row["x_px"]),float(row["y_px"])
    height,width = arrays["Raw"].shape[1:]
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        raise ValueError("Expert coordinate lies outside the source image")
    px,py = min(width-1,int(math.floor(x+0.5))),min(height-1,int(math.floor(y+0.5)))
    first,last = frames[0]-source_start,frames[-1]-source_start+1
    traces = {key:np.asarray(array[first:last,py,px]).copy() for key,array in arrays.items()}
    if any(len(values) != len(frames) or not np.isfinite(values).all() for values in traces.values()):
        raise ValueError("Invalid exact-center trace")
    if not np.allclose(traces["contrast"],traces["A"]-traces["M"],rtol=2e-6,atol=2e-6):
        raise ValueError("Contrast trace differs from A minus M")
    expected_z = traces["contrast"]/(np.maximum(traces["sigma"],floor)+epsilon)
    if np.any(traces["sigma"] < 0) or not np.allclose(traces["Z"],expected_z,rtol=2e-6,atol=2e-6):
        raise ValueError("Z trace differs from its declared contrast/spread/floor")
    return {"frames":frames,"pixel_xy":[px,py],"values":traces}


def _render(path: Path, row: Mapping[str, Any], traces: Mapping[str, Any], *, tau_a: float,
            tau_z: float, floor: float, frame_interval_ms: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    frames, values = traces["frames"],traces["values"]
    start,stop = int(row["window_start_frame_ui"]),int(row["window_stop_frame_ui"])
    time_ms = (np.asarray(frames)-start)*frame_interval_ms
    fig,axes = plt.subplots(4,1,figsize=(10.5,9.2),sharex=True)
    fig.subplots_adjust(left=.13,right=.98,top=.87,bottom=.13,hspace=.34)
    match = lambda key: "matched" if row[key] else "unmatched"
    fig.suptitle(f"{row['observation_id']} | A {match('A_matched')}; Z {match('Z_matched')}\n"
                 f"Exact labeled-center pixel {tuple(traces['pixel_xy'])}; source frames {frames[0]}–{frames[-1]}",fontsize=14,y=.97)
    axes[0].plot(time_ms,values["Raw"],color="#303030",label="Raw")
    axes[0].set_ylabel("Fluorescence\n(native units)")
    axes[0].set_title("Recorded fluorescence at the labeled center",loc="left",fontsize=11)
    axes[1].plot(time_ms,values["A"],color="#1763a6",label="Target A")
    axes[1].plot(time_ms,values["M"],color="#666666",label="Reference mean M")
    axes[1].axhline(tau_a,color="#1763a6",linestyle="--",label=f"A threshold {tau_a:.4g}")
    axes[1].set_ylabel("Fluorescence change\n(native units)")
    axes[1].set_title("Target and surrounding mean; A uses its own frozen threshold",loc="left",fontsize=11)
    axes[2].plot(time_ms,values["contrast"],color="#5e3c99",label="Contrast A − M")
    axes[2].plot(time_ms,values["sigma"],color="#8c5a2b",label="Reference standard deviation")
    axes[2].axhline(floor,color="#8c5a2b",linestyle=":",label=f"Spread floor {floor:.4g}")
    axes[2].set_ylabel("Fluorescence change\n(native units)")
    axes[2].set_title("Local subtraction and the spatial spread used to scale it",loc="left",fontsize=11)
    axes[3].plot(time_ms,values["Z"],color="#1763a6",label="Standardized Z")
    axes[3].axhline(tau_z,color="#1763a6",linestyle="--",label=f"Z threshold {tau_z:.4g}")
    axes[3].set_ylabel("Z\n(dimensionless)")
    axes[3].set_title("Standardized response; Z uses its own frozen threshold",loc="left",fontsize=11)
    axes[3].set_xlabel("Time from the declared window start (ms); window start is not a neural onset")
    for index,axis in enumerate(axes):
        axis.axvspan(0,(stop-start)*frame_interval_ms,color="#d9d9d9",alpha=.32,zorder=-5)
        if index:
            axis.axhline(0,color="#b0b0b0",linewidth=.6,zorder=-3)
        axis.grid(alpha=.18)
        axis.legend(fontsize=8,loc="upper right",ncol=2)
        axis.set_xlim(time_ms[0],time_ms[-1])
        axis.tick_params(labelsize=9)
    fig.text(.13,.035,"Shading: declared burst window. Curves: exact labeled-center samples.\n"
             "Match status: a separately assigned spatial representative within 6 pixels in that window; not a center crossing or onset measure.",fontsize=9)
    temporary = path.with_name(path.stem+".tmp.png")
    try:
        fig.savefig(temporary,dpi=150,facecolor="white")
        from PIL import Image
        with Image.open(temporary) as image:
            if image.size != (1575,1380):
                raise ValueError("Unexpected panel raster dimensions")
            image.verify()
        temporary.replace(path)
    finally:
        plt.close(fig)


def run(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    data = load_sources(root)
    # Hash full stages once; memmaps below access only four bounded trace windows.
    for record in data["stage_records"].values():
        _verify(record)
    contract = {"schema_version":1,"purpose":"deterministic illustrative exact-center stage traces",
        "run_root":str(root),"source_records":data["sources"],"stage_records":data["stage_records"],
        "renderer":_record(Path(__file__)),"thresholds":data["thresholds"],"context_frames":CONTEXT,
        "selection_rule":"first lexicographic observation_id in each A/Z q1 match stratum; empty strata explicit",
        "source_frames_ui":data["protocol"]["source_frames_ui"],"frame_interval_ms":data["protocol"]["frame_interval_ms"],
        "rounding":"floor(coordinate+0.5), clipped at last image pixel",
        "selection":data["selection"],"scale_floor":data["operator"]["scale_floor"],
        "epsilon":data["operator"]["spec"]["epsilon"],
        "match_semantics":"one-to-one spatial representative within6px in declared burst; separate from center trace",
        "original_full_recording_source":data["protocol"]["source"],
        "original_source_verification":"inherited from protocol; displayed Raw crop and all stages verified directly",
        "axis_semantics":"one axis per panel; A and Z thresholds stay in their own native/dimensionless units",
        "scientific_audit_replacement":False,"independent_confirmation":False,"onset_accuracy_identified":False}
    folder = root/"review/case_panels"
    path = folder/"panel_contract.json"
    if path.exists():
        if _read_json(path) != contract:
            raise ValueError("Cannot resume case panels with changed sources or selection")
        manifest_path = folder/"manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if manifest.get("status") == "COMPLETE":
                _verify_panel_inventory(folder,manifest,data["selection"])
                return manifest
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Refusing to overwrite unrelated case-panel outputs")
    folder.mkdir(parents=True,exist_ok=True)
    _write_json(path,contract)
    _write_rows(folder/"all79_strata.tsv",data["membership"])
    _write_rows(folder/"selection.tsv",data["selection"])
    import numpy as np
    arrays = {key:np.load(record["path"],mmap_mode="r",allow_pickle=False) for key,record in data["stage_records"].items()}
    if any(array.shape != (560,340,573) for array in arrays.values()):
        raise ValueError("Every stage must match the frozen560×340×573 source grid")
    if any(array.dtype.kind not in "fui" for array in arrays.values()):
        raise ValueError("Stage arrays must be numeric")
    selected = []
    for row in data["membership"]:
        if not row["selected_for_panel"]:
            continue
        traces = case_traces(arrays,row,source_start=1800,source_stop=2359,
                             floor=float(contract["scale_floor"]),epsilon=float(contract["epsilon"]))
        identifier = row["observation_id"]
        if not identifier or Path(identifier).name != identifier or identifier in {".",".."}:
            raise ValueError("Unsafe observation ID for output filename")
        prefix = folder/f"{row['stratum']}__{identifier}"
        plot_path = prefix.with_suffix(".png")
        _render(plot_path,row,traces,tau_a=data["thresholds"][ARM_A],tau_z=data["thresholds"][ARM_Z],
                floor=float(contract["scale_floor"]),frame_interval_ms=float(contract["frame_interval_ms"]))
        sample_rows = [{"source_frame_ui":frame,**{key:float(values[i]) for key,values in traces["values"].items()}}
                       for i,frame in enumerate(traces["frames"])]
        _write_rows(prefix.with_suffix(".tsv"),sample_rows)
        indices = [i for i,frame in enumerate(traces["frames"]) if row["window_start_frame_ui"] <= frame <= row["window_stop_frame_ui"]]
        center_counts = {key:int(np.sum(traces["values"][key][indices] > data["thresholds"][arm]))
                         for key,arm in (("A",ARM_A),("Z",ARM_Z))}
        metadata = {**row,"trace_pixel_xy":traces["pixel_xy"],"source_frames_ui":traces["frames"],
            "plot_window_ui_inclusive":[traces["frames"][0],traces["frames"][-1]],
            "threshold_A":data["thresholds"][ARM_A],"threshold_Z":data["thresholds"][ARM_Z],
            "scale_floor":contract["scale_floor"],"epsilon":contract["epsilon"],
            "panel_contract_sha256":_digest(path),"trace_stage_formula_check_pass":True,
            "trace_formula_check_rtol":2e-6,"trace_formula_check_atol":2e-6,
            "exact_center_above_threshold_frame_counts_in_burst":center_counts,
            "exact_center_counts_are_not_match_status":True,
            "trace_coordinate_role":"this occurrence's exact expert center, not the matched representative",
            "shaded_window_role":"broad annotation; not onset truth","render_dpi":150,
            "numerical_traces":str(prefix.with_suffix(".tsv")),"figure":str(plot_path)}
        _write_json(prefix.with_suffix(".json"),metadata)
        selected.append(identifier)
    readme = folder/"README.md"
    readme.write_text("# A versus Z: deterministic occurrence examples\n\n"
        "The full 79-occurrence stratum table determines the first observation ID in each of four A/Z q1 match categories. "
        "An empty category is explicitly recorded; no replacement example is selected. These outcome-stratified examples "
        "are illustrations, not a prevalence estimate or a replacement for the complete scientific audit.\n\n"
        "Each figure shows the occurrence's exact expert-center pixel over its declared burst plus five source frames of context. "
        "Match status instead refers to the separately computed one-to-one representative within 6 pixels in that burst. "
        "Raw, target A/reference M, contrast/reference standard deviation, and standardized Z have separate axes. "
        "A and Z each retain their own frozen q1 threshold; no dual-axis rescaling is used.\n\n"
        "Shading marks a broad known window, not neural onset. A decline or plateau need not produce a positive difference. "
        "Unknown proposals are not false positives. Numeric per-frame samples and source/selection hashes accompany every figure. "
        "This panel manifest does not certify the full study's media audit, independent confirmation, or real-time control.\n")
    artifacts = [_record(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != "manifest.json"]
    manifest = {"schema_version":1,"status":"COMPLETE","scope":"four-stratum case panels only",
        "full_cohort_occurrence_count":len(data["membership"]),"selected_occurrence_ids":selected,
        "selection":data["selection"],"panel_count":len(selected),"artifacts":artifacts,
        "panel_contract_sha256":_digest(path),"complete_scientific_audit_claim":False}
    _verify_panel_inventory(folder,manifest,data["selection"])
    _write_json(folder/"manifest.json",manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path,help="Numerically completed, sealed control-study root")
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps({key:result[key] for key in ("status","panel_count","full_cohort_occurrence_count")},sort_keys=True))


if __name__ == "__main__":
    main()
