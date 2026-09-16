"""Source-bound development report for regional calibration and crowding.

No scoring, matching, label changes, or dense arrays are used here. ``generate``
requires a new report directory; ``update`` only refreshes completion wording
and its manifest. Both refuse roots containing a completion_manifest.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Outputs/GammaLSFollowup/followup_20260914_r1"
READOUTS = ("level_X", "level_A", "level_C", "level_Z", "difference_Z")
DEADLINES = (0, 20, 40, 60, 100, 200, 500)
Q_VALUES = (0., .25, .5, 1., 2., 4., 8., 16.)
THRESHOLDS = tuple(f"q{q:g}" for q in Q_VALUES) + ("all_positive", "no_output")
AUDIT_METADATA = {"summary.json", "status.json", "validation.json", "inventory.json",
                  "artifact_index.json", "source_manifest.json", "run_contract.json", "llm_context.json"}
LABELS = {"level_X": "Level input", "level_A": "Level target", "level_C": "Level contrast",
          "level_Z": "Level Gamma-LS", "difference_Z": "Change Gamma-LS"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return dict(path=str(path), sha256=sha.hexdigest(), size_bytes=path.stat().st_size)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_tsv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def expected_cells(protocol):
    regional = [dict(case_id=case, arm_id=f"{arm}__{method}", base_arm=arm, study="regional",
                     calibration_method=method, input_mode=arm.split("_")[0], readout=arm[-1], window=13)
                for case in protocol["regional_cases"] for arm in READOUTS for method in ("global", "regional")]
    crowding = [dict(case_id=case["case_id"], arm_id=f"level_{score}__w{window}", base_arm=f"level_{score}",
                    study="crowding", calibration_method="global", input_mode="level", readout=score, window=window)
                for case in protocol["crowding_cases"] for score in ("A", "C", "Z") for window in (13, 3)]
    return regional + crowding


def curve_key(row):
    return row["case_id"], row["arm_id"], row["threshold_id"], float(row["match_radius_px"])


def validate_curve_inventory(rows, cells):
    expected = {(c["case_id"], c["arm_id"], threshold, radius)
                for c in cells for threshold in THRESHOLDS
                for radius in ((6., 2.) if c["study"] == "crowding" else (6.,))}
    keys = [curve_key(row) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete or duplicate cell/threshold/radius rows")


def dataset_bindings(seal, *, cell_seal=False):
    """Normalize the two actual preparation schemas and current cell seals.

    Reused necessity preparation stores a mixed small/array ``bindings`` list;
    new crowding preparation has named entries. Dense records are not read.
    Every current scored cell additionally seals the four small dataset files.
    """
    names = {"metadata.json", "experts.json", "active.json"} | ({"prepared.json"} if cell_seal else set())
    if cell_seal:
        records = seal["dataset_bindings"]
    elif "bindings" in seal:
        records = seal["bindings"]
    else:
        records = [seal[key] for key in ("metadata", "experts", "active")]
    selected = [r for r in records if Path(r["path"]).name in names]
    require(len(selected) == len(names) and {Path(r["path"]).name for r in selected} == names,
            "Missing or duplicate dataset metadata bindings")
    if cell_seal:
        require(len(records) == len(names), "Unexpected scored-cell dataset binding inventory")
    return {Path(r["path"]).name: r for r in selected}


def _load_inputs(root):
    """Verify saved small inputs and seals; inherit, never reread, dense stages."""
    root = Path(root).resolve()
    sources = {}

    def load(path):
        path = Path(path).resolve()
        sources[str(path)] = binding(path)
        return read(path)

    def verify(record, expected_path=None):
        path = Path(record["path"]).resolve()
        if expected_path is not None:
            require(path == Path(expected_path).resolve(), f"Wrong bound path: {path}")
        actual = binding(path)
        require(all(actual[key] == record[key] for key in ("sha256", "size_bytes")), f"Changed input: {path}")
        sources[str(path)] = actual

    protocol = load(root / "protocol.json")
    preflight = load(root / "preflight.json")
    require(preflight.get("status") == "PASS" and preflight["protocol_sha256"] == sources[str(root / "protocol.json")]["sha256"], "Protocol/preflight mismatch")
    for record in protocol["code_bindings"]:
        verify(record)
    for key in ("baseline_completion", "baseline_protocol", "real_annotation_acceptance"):
        verify(protocol[key])
    require(not read(protocol["real_annotation_acceptance"]["path"]).get("accepted"), "Accepted real labels need a new evaluation contract")
    require(len(protocol["regional_cases"]) == len(set(protocol["regional_cases"])) == 17 and "real" in protocol["regional_cases"], "Expected 16 pilot simulations plus one recording")
    require(len(protocol["crowding_cases"]) == 12 and protocol["expected_cells"] == 242, "Expected 12 paired crowding clips and 242 cells")
    require(protocol["regional"]["scores"] == list(READOUTS), "Regional readout contract changed")
    require(protocol["frame_rate_hz"] == 50 and protocol["pixel_size_um"] == .5, "Physical conversion contract changed")
    prepared = load(root / "datasets_complete.json")
    computation = load(root / "computation_complete.json")
    evaluation = load(root / "evaluation_complete.json")
    require(prepared.get("status") == "PASS" and prepared.get("datasets") == 29 and prepared.get("paired_setup_byte_equal") is True, "Preparation or paired setup incomplete")
    require(computation == dict(status="PASS", cells=242), "Computation incomplete")
    require(evaluation == dict(status="PASS", cells=242, curve_rows=3140), "Evaluation incomplete")
    replication = load(root / "baseline_replication.json")
    expected_replication = {(c, f"{a}__global") for c in protocol["regional_cases"] for a in READOUTS}
    replication_keys = [(r["case_id"], r["arm_id"]) for r in replication["cells"]]
    require(replication.get("status") == "PASS" and len(replication_keys) == len(set(replication_keys)) == 85
            and set(replication_keys) == expected_replication
            and all(r["all_thresholds_equal"] and r["metrics_equal"] for r in replication["cells"]), "Global-baseline replication incomplete")
    dataset_meta, experts, inherited = {}, {}, {}
    all_cases = protocol["regional_cases"] + [r["case_id"] for r in protocol["crowding_cases"]]
    require(len(all_cases) == len(set(all_cases)), "Duplicate dataset identity")
    for case in all_cases:
        folder = root / "datasets" / case
        seal = load(folder / "prepared.json")
        require(seal.get("status") == "PASS", "Dataset seal incomplete")
        for filename, record in dataset_bindings(seal).items():
            verify(record, folder / filename)
        dataset_meta[case] = load(folder / "metadata.json")
        experts[case] = load(folder / "experts.json")
    cells = expected_cells(protocol)
    rows = load(root / "all_curves.json")
    validate_curve_inventory(rows, cells)
    collected, events, operating, calibrations = [], [], {}, []
    for cell in cells:
        folder = root / "cells" / cell["case_id"] / cell["arm_id"]
        seal = load(folder / "sealed.json")
        require(seal.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", "Scored cell not sealed")
        for filename, record in dataset_bindings(seal, cell_seal=True).items():
            verify(record, root / "datasets" / cell["case_id"] / filename)
        done = load(folder / "evaluated.json")
        require(done.get("status") == "PASS", "Cell evaluation incomplete")
        verify(done["seal"], folder / "sealed.json")
        outputs = {Path(record["path"]).name: record for record in done["outputs"]}
        required = {"curves.json", "operating_metrics_r6.json"} | ({"operating_metrics_r2.json"} if cell["study"] == "crowding" else set())
        require(set(outputs) == required, "Unexpected or missing evaluation output binding")
        for name, record in outputs.items():
            verify(record, folder / name)
        for key in ("calibration", "threshold_plan", "audit_candidates"):
            verify(seal[key], folder / (key + ".json"))
        calibration = load(folder / "calibration.json")
        plan = load(folder / "threshold_plan.json")
        require({p["threshold_id"] for p in plan} == set(THRESHOLDS) and len(plan) == 10, "Threshold plan incomplete")
        require(calibration["threshold_id"] == "q1" and calibration["threshold_frozen_from_calibration_only"] is True, "Frozen operating point differs")
        require(all(calibration.get(key) == value for key, value in cell.items()), "Cell/calibration identity differs")
        candidates = load(folder / "audit_candidates.json")
        curves = load(folder / "curves.json")
        validate_curve_inventory(curves["curve_rows"], [cell])
        for row in curves["curve_rows"]:
            require(all(row.get(key) == value for key, value in cell.items()), "Curve/cell identity mismatch")
            setting = next(p for p in plan if p["threshold_id"] == row["threshold_id"])
            require(all(row.get(k) == v for k, v in setting.items()), "Curve cutoff differs from sealed calibration plan")
            require(row["truth_mode"] == dataset_meta[cell["case_id"]]["truth_mode"], "Curve truth mode differs")
            if row["truth_mode"] == "sparse_real":
                require(all(row.get(k) is None for k in ("precision", "false_positive_count", "framewise_sensitivity")), "Sparse real precision/sensitivity claim")
        collected.extend(curves["curve_rows"])
        for radius in ((6, 2) if cell["study"] == "crowding" else (6,)):
            fixed = load(folder / f"operating_metrics_r{radius}.json")
            summary = fixed["summary"]
            row = next(r for r in curves["curve_rows"] if r["threshold_id"] == "q1" and r["match_radius_px"] == radius)
            for key in ("proposal_count", "true_positive_count", "false_positive_count", "active_region_frame_count", "event_count", "recovered_event_count", "matched_known_positive_count"):
                if key in row and key in summary:
                    require(row[key] == summary[key], f"Operating/curve metric differs: {key}")
            require(row["proposal_count"] == len(candidates), "Operating candidate count differs")
            operating[(cell["case_id"], cell["arm_id"], float(radius))] = fixed
        for setting in plan:
            for region in setting.get("regional_thresholds", [dict(setting, region_id="global")]):
                calibrations.append(dict(cell, threshold_id=setting["threshold_id"],
                    setup_budget_per_reference_area_frame=setting["setup_budget_per_reference_area_frame"],
                    region_id=region["region_id"], box_yxyx=region.get("box_yxyx"), native_cutoff=region["threshold"],
                    setup_proposal_budget=region["setup_proposal_budget"], setup_proposal_count=region["setup_proposal_count"],
                    score_units="dimensionless" if cell["readout"] == "Z" else "native_fluorescence_units"))
        event_lookup = {str(e.get("event_id", e["observation_id"])): e for e in experts[cell["case_id"]]}
        if dataset_meta[cell["case_id"]]["truth_mode"] == "fully_synthetic":
            for radius in ((6, 2) if cell["study"] == "crowding" else (6,)):
                for threshold in THRESHOLDS:
                    group = [e for e in curves["event_rows"] if e["threshold_id"] == threshold and e["match_radius_px"] == radius]
                    require(len(group) == len(event_lookup) and {e["event_id"] for e in group} == set(event_lookup), "Incomplete event/threshold/radius rows")
                    for event in group:
                        truth = event_lookup[event["event_id"]]
                        require(event["canonical_roi_id"] == truth["canonical_roi_id"], "Event source identity mismatch")
                        require(int(event["source_start_ui"]) == int(truth["source_start_ui"]) and int(event["source_stop_ui"]) == int(truth["source_stop_ui"]), "Event support mismatch")
                        require(event["active_frame_count"] == int(truth["source_stop_ui"]) - int(truth["source_start_ui"]) + 1, "Event active-frame denominator differs")
                        require(0 <= event["matched_active_frame_count"] <= event["active_frame_count"], "Invalid source-frame count")
                        require(event["recovered"] == (event["first_delay_ms"] is not None), "Recovered/delay mismatch")
                        if cell["study"] == "crowding":
                            require(truth.get("source_role") in {"weak", "neighbor"}, "Crowding source role missing")
                        events.append(dict(event, **cell, source_role=truth.get("source_role", "declared_source"),
                                           seed=dataset_meta[cell["case_id"]].get("seed"),
                                           separation_px=dataset_meta[cell["case_id"]].get("neighbor_separation_px")))
        inherited[f"{cell['case_id']}/{cell['arm_id']}"] = {key: seal[key] for key in ("stages", "prefix", "setup_prefix")}
    require(sorted(rows, key=curve_key) == sorted(collected, key=curve_key), "Aggregate all_curves differs from sealed per-cell curves")
    return dict(root=root, protocol=protocol, rows=rows, events=events, operating=operating,
                calibrations=calibrations, metadata=dataset_meta, experts=experts,
                sources=sorted(sources.values(), key=lambda r: r["path"]), inherited=inherited)


def pool(rows):
    """Pool counts/exposures, following necessity_report.pool, not clip rates."""
    fields = ("true_positive_count", "false_positive_count", "active_region_frame_count", "event_count", "proposal_count", "recovered_event_count", "duplicate_near_active_region_count")
    total = {key: sum(row.get(key, 0) for row in rows) for key in fields}
    area_time = sum(row["eligible_area_px"] * .25 * row["exposure_seconds"] for row in rows)
    seconds = sum(row["exposure_seconds"] for row in rows)
    total.update(case_count=len(rows), precision=ratio(total["true_positive_count"], total["proposal_count"]),
        active_frame_coverage=ratio(total["true_positive_count"], total["active_region_frame_count"]),
        event_coverage=ratio(total["recovered_event_count"], total["event_count"]),
        false_proposals_per_10000um2_second=ratio(total["false_positive_count"] * 10000, area_time),
        proposals_per_second=ratio(total["proposal_count"], seconds))
    total["deadline_rows"] = [dict(deadline_ms=d, event_count=total["event_count"], recovered_by_deadline=sum(
        next(v["recovered_by_deadline"] for v in row["deadline_rows"] if v["deadline_ms"] == d) for row in rows)) for d in DEADLINES]
    total["prompt_200ms_coverage"] = ratio(next(v["recovered_by_deadline"] for v in total["deadline_rows"] if v["deadline_ms"] == 200), total["event_count"])
    return total


def weak_pool(events, clip_rows):
    require(len(events) == len(clip_rows) and len({e["case_id"] for e in events}) == len(events), "Need one weak event per paired clip")
    require(all(e["source_role"] == "weak" for e in events), "Neighbor entered weak-source pool")
    all_metrics = pool(clip_rows)
    total_active = sum(e["active_frame_count"] for e in events)
    matched_active = sum(e["matched_active_frame_count"] for e in events)
    deadlines = [dict(deadline_ms=d, event_count=len(events), recovered_by_deadline=sum(
        e["first_delay_ms"] is not None and e["first_delay_ms"] <= d + 1e-9 for e in events)) for d in DEADLINES]
    return dict(all_metrics, weak_event_count=len(events), weak_recovered_event_count=sum(e["recovered"] for e in events),
        weak_active_frame_count=total_active, weak_matched_active_frame_count=matched_active,
        weak_active_frame_coverage=ratio(matched_active, total_active),
        weak_event_coverage=ratio(sum(e["recovered"] for e in events), len(events)),
        weak_deadline_rows=deadlines, weak_prompt_200ms_coverage=ratio(next(v["recovered_by_deadline"] for v in deadlines if v["deadline_ms"] == 200), len(events)),
        false_proposal_scope="all sources and background in these clips; not assigned exclusively to weak source")


def make_tables(data):
    rows = data["rows"]
    regional = [r for r in rows if r["study"] == "regional"]
    crowding = [dict(r, seed=data["metadata"][r["case_id"]]["seed"],
                     separation_px=data["metadata"][r["case_id"]]["neighbor_separation_px"]) for r in rows if r["study"] == "crowding"]
    grouped = defaultdict(list)
    for row in regional:
        if row["truth_mode"] == "fully_synthetic":
            grouped[(row["base_arm"], row["calibration_method"], row["threshold_id"])].append(row)
    pooled = []
    for (arm, method, threshold), group in sorted(grouped.items()):
        require(len(group) == len({r["case_id"] for r in group}) == 16, "Incomplete regional synthetic pool")
        pooled.append(dict(base_arm=arm, calibration_method=method, threshold_id=threshold,
                           setup_budget_per_reference_area_frame=group[0]["setup_budget_per_reference_area_frame"], **pool(group)))
    real = [dict(r, known_window_coverage=ratio(r["matched_known_positive_count"], r["known_positive_count"])) for r in regional if r["truth_mode"] == "sparse_real"]
    require(len(real) == 100 and all(r["known_positive_count"] == 76 for r in real), "Real denominator changed")
    pairs = []
    lookup = {curve_key(r): r for r in regional}
    for case in data["protocol"]["regional_cases"]:
        for arm in READOUTS:
            for threshold in THRESHOLDS:
                a = lookup[(case, arm + "__global", threshold, 6.)]
                b = lookup[(case, arm + "__regional", threshold, 6.)]
                row = dict(case_id=case, base_arm=arm, threshold_id=threshold, direction="regional minus global",
                           seed=data["metadata"][case].get("seed"), truth_mode=a["truth_mode"])
                for key in ("proposal_count", "true_positive_count", "false_positive_count", "recovered_event_count", "matched_known_positive_count"):
                    row[key + "_change"] = b[key] - a[key] if a.get(key) is not None and b.get(key) is not None else None
                pairs.append(row)
    events = [e for e in data["events"] if e["study"] == "crowding"]
    weak = [e for e in events if e["source_role"] == "weak"]
    weak_groups = defaultdict(list)
    for e in weak:
        weak_groups[(e["readout"], e["window"], e["separation_px"], e["threshold_id"], float(e["match_radius_px"]))].append(e)
    crowd_lookup = {curve_key(r): r for r in crowding}
    pooled_weak = []
    for (score, window, separation, threshold, radius), group in sorted(weak_groups.items(), key=lambda item: str(item[0])):
        require(len(group) == 3 and len({e["seed"] for e in group}) == 3, "Incomplete three-seed weak-source pool")
        clips = [crowd_lookup[curve_key(e)] for e in group]
        pooled_weak.append(dict(readout=score, window=window, separation_px=separation, threshold_id=threshold,
            match_radius_px=radius, setup_budget_per_reference_area_frame=clips[0]["setup_budget_per_reference_area_frame"], **weak_pool(group, clips)))
    per_seed = []
    for event in weak:
        if event["threshold_id"] == "q1":
            clip = crowd_lookup[curve_key(event)]
            per_seed.append(dict(event, weak_active_frame_coverage=ratio(event["matched_active_frame_count"], event["active_frame_count"]),
                weak_recovered_by_200ms=event["first_delay_ms"] is not None and event["first_delay_ms"] <= 200 + 1e-9,
                clip_false_positive_count=clip["false_positive_count"], clip_proposal_count=clip["proposal_count"],
                clip_duplicate_proposal_count=clip["duplicate_near_active_region_count"], native_cutoff=clip["threshold"]))
    paired = []
    seed_lookup = {(e["seed"], e["separation_px"], e["readout"], e["window"], e["match_radius_px"]): e for e in per_seed}
    for seed in data["protocol"]["crowding"]["seeds"]:
        for separation in (None, 8, 12, 16):
            for score in ("A", "C", "Z"):
                for radius in (2., 6.):
                    a, b = [seed_lookup[(seed, separation, score, window, radius)] for window in (13, 3)]
                    paired.append(dict(seed=seed, separation_px=separation, readout=score, match_radius_px=radius,
                        direction="3x3 minus 13x13 selector", weak_matched_active_frame_change=b["matched_active_frame_count"]-a["matched_active_frame_count"],
                        weak_prompt_200ms_change=int(b["weak_recovered_by_200ms"])-int(a["weak_recovered_by_200ms"]),
                        clip_false_proposal_change=b["clip_false_positive_count"]-a["clip_false_positive_count"],
                        clip_proposal_change=b["clip_proposal_count"]-a["clip_proposal_count"],
                        cutoff_13=a["native_cutoff"], cutoff_3=b["native_cutoff"]))
    real_cutoffs = []
    for arm in READOUTS:
        selected = [r for r in data["calibrations"] if r["case_id"] == "real" and r["base_arm"] == arm and r["threshold_id"] == "q1"]
        global_row = next(r for r in selected if r["calibration_method"] == "global")
        regional_rows = sorted((r for r in selected if r["calibration_method"] == "regional"), key=lambda r: r["region_id"])
        require(len(regional_rows) == 6, "Need every geometry-fixed regional cutoff")
        for row in regional_rows:
            real_cutoffs.append(dict(row, global_native_cutoff=global_row["native_cutoff"],
                global_setup_proposal_budget=global_row["setup_proposal_budget"],
                global_setup_proposal_count=global_row["setup_proposal_count"]))
    return dict(regional_case_curves=regional, regional_pooled_curves=pooled, regional_real_curves=real,
        regional_paired_changes=pairs, calibration_thresholds=data["calibrations"], real_regional_q1_cutoffs=real_cutoffs,
        crowding_case_curves=crowding,
        crowding_event_curves=events, crowding_weak_event_curves=weak, crowding_pooled_weak_curves=pooled_weak,
        crowding_q1_per_seed=per_seed, crowding_selector_paired_q1=paired)


def _audit_status(root, protocol):
    path = root / "audit_complete.json"
    if not path.exists():
        return dict(status="NUMERICAL_ONLY_AUDITS_PENDING", scientific_audit_complete=False, source=None)
    aggregate = read(path)
    require(aggregate.get("status") == "PASS" and aggregate.get("cells") == 242, "Audit aggregate is present but not complete")
    require(aggregate.get("new_cells") == 157 and aggregate.get("reused_cells") == 85, "Audit reuse inventory differs")
    expected = {(c["case_id"], c["arm_id"]) for c in expected_cells(protocol)}
    keys = [(row["case_id"], row["arm_id"]) for row in aggregate["audits"]]
    require(len(keys) == len(set(keys)) == 242 and set(keys) == expected, "Audit aggregate lacks exact 242-cell coverage")
    require(sum(row.get("reused") is True for row in aggregate["audits"]) == 85,
            "Expected 85 explicitly identified reused global audits")
    bindings = []
    for row in aggregate["audits"]:
        require(set(row["metadata_bindings"]) == AUDIT_METADATA, "Audit metadata bindings incomplete")
        for name, record in row["metadata_bindings"].items():
            require(Path(record["path"]).name == name, "Audit metadata filename mismatch")
            actual = binding(record["path"])
            require(all(actual[k] == record[k] for k in ("sha256", "size_bytes")), "Audit metadata changed")
            bindings.append(actual)
        summary = read(row["metadata_bindings"]["summary.json"]["path"])
        validation = read(row["metadata_bindings"]["validation.json"]["path"])
        status = read(row["metadata_bindings"]["status.json"]["path"])
        require(summary == row["summary"] and summary.get("scientific_audit_complete") is True
                and status.get("status") == "complete" and status.get("scientific_audit_complete") is True
                and validation.get("status") == "passed" and not validation.get("failures"), "Audit completion fields disagree")
    return dict(status="AUDITS_VERIFIED_REPORT_QA_SEPARATE", scientific_audit_complete=True,
                source=binding(path), metadata_bindings=bindings,
                scope="Small aggregate metadata hashes verified; full movie/array validation inherited from the complete audit validator")


def _plot_figures(out, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    colors = {arm: plt.get_cmap("tab10")(i) for i, arm in enumerate(READOUTS)}
    score_colors = {s: plt.get_cmap("tab10")(i) for i, s in enumerate(("A", "C", "Z"))}
    figures = []
    def numeric(v): return float("nan") if v is None else v
    def save(fig, name, title, caption, handles=None, labels=None, bottom=.22):
        fig.suptitle(title, fontsize=14, y=.985)
        fig.subplots_adjust(left=.075, right=.975, top=.9, bottom=bottom, hspace=.36, wspace=.29)
        if handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.095), ncol=min(5,len(handles)), frameon=False, fontsize=8)
        fig.text(.055,.025,caption,fontsize=8,va="bottom",ha="left")
        fig.savefig(out/name,dpi=130,facecolor="white")
        plt.close(fig)
        figures.append(name)
    pooled = tables["regional_pooled_curves"]
    fig,axes = plt.subplots(2,3,figsize=(14,9))
    for arm in READOUTS:
        for method,style in (("global","-"),("regional","--")):
            rs = sorted([r for r in pooled if r["base_arm"]==arm and r["calibration_method"]==method and r["setup_budget_per_reference_area_frame"] is not None],key=lambda r:r["setup_budget_per_reference_area_frame"])
            for ax,key in zip(axes.flat,("precision","active_frame_coverage","prompt_200ms_coverage","false_proposals_per_10000um2_second","event_coverage","proposals_per_second")):
                ax.plot([r["setup_budget_per_reference_area_frame"] for r in rs],[numeric(r[key]) for r in rs],style,color=colors[arm],marker="o",ms=3,label=f"{LABELS[arm]} · {method}")
    for i,(ax,title) in enumerate(zip(axes.flat,("Frame-proposal precision","Active source/frame coverage","Events recovered within 200 ms","False proposals /10,000 µm² /s","Any-time event recovery","Application proposals /s"))):
        ax.set_title(title);ax.set_xlabel("Setup target q /reference area /frame");ax.set_xscale("symlog",linthresh=.5);ax.grid(alpha=.2)
        if i in (0,1,2,4):ax.set_ylim(-.03,1.03)
    save(fig,"regional_synthetic_budget_curves.png","Regional setup calibration: 16 paired pilot simulations",
         "Counts and area-time exposures pooled across 16 clips. All events, including misses, remain in prompt-recovery denominators.\nSolid: global; dashed: geometry-fixed 2×3 regions. Equal setup budgets do not guarantee equal application burden.",*axes[0,0].get_legend_handles_labels(),bottom=.26)
    real=tables["regional_real_curves"]
    fig,axes=plt.subplots(1,2,figsize=(12,6.2))
    for arm in READOUTS:
        for method,style in (("global","-"),("regional","--")):
            rs=sorted([r for r in real if r["base_arm"]==arm and r["calibration_method"]==method and r["setup_budget_per_reference_area_frame"] is not None],key=lambda r:r["setup_budget_per_reference_area_frame"])
            for ax,key in zip(axes,("matched_known_positive_count","proposal_count")):
                ax.plot([r["setup_budget_per_reference_area_frame"] for r in rs],[r[key] for r in rs],style,color=colors[arm],marker="o",ms=3,label=f"{LABELS[arm]} · {method}")
    for ax,title in zip(axes,("Known broad windows reached /76","Application frame/location proposals")):
        ax.set_title(title);ax.set_xlabel("Setup target q /reference area /frame");ax.set_xscale("symlog",linthresh=.5);ax.grid(alpha=.2)
    axes[0].set_ylim(-1,77)
    save(fig,"regional_real_budget_curves.png","Recording: known-window coverage and proposal burden",
         "The same 76 sparse known windows and 560 application frames are used in every arm. Unmatched proposals remain unknown.\nReal precision, exhaustive sensitivity and onset accuracy are unavailable; raw-panel rendering does not accept human labels.",*axes[0].get_legend_handles_labels(),bottom=.31)
    fig,ax=plt.subplots(figsize=(10.5,7))
    for arm in READOUTS:
        for method,style in (("global","-"),("regional","--")):
            rs=sorted([r for r in real if r["base_arm"]==arm and r["calibration_method"]==method and r["setup_budget_per_reference_area_frame"] is not None],key=lambda r:r["setup_budget_per_reference_area_frame"])
            ax.plot([r["proposal_count"] for r in rs],[r["matched_known_positive_count"] for r in rs],style,color=colors[arm],marker="o",ms=3,label=f"{LABELS[arm]} · {method}")
            point=next(r for r in rs if r["threshold_id"]=="q1")
            ax.scatter([point["proposal_count"]],[point["matched_known_positive_count"]],facecolors="none",edgecolors=[colors[arm]],s=65)
    ax.set_xlabel("Actual application frame/location proposals");ax.set_ylabel("Known broad windows reached /76");ax.set_ylim(-1,77);ax.grid(alpha=.2)
    save(fig,"regional_real_coverage_burden.png","Recording: paired coverage versus realized burden",
         "Points follow the eight fixed setup targets; open rings mark q=1. Curves do not select an application optimum.\nNative score units differ across readouts; no common numerical cutoff is implied.",*ax.get_legend_handles_labels(),bottom=.27)
    fig,axes=plt.subplots(1,5,figsize=(15,5.2))
    for ax,arm in zip(axes,READOUTS):
        rs=[r for r in tables["calibration_thresholds"] if r["case_id"]=="real" and r["base_arm"]==arm and r["threshold_id"]=="q1"]
        global_row=next(r for r in rs if r["calibration_method"]=="global")
        region_rows=sorted([r for r in rs if r["calibration_method"]=="regional"],key=lambda r:r["region_id"])
        ax.bar(range(6),[numeric(r["native_cutoff"]) for r in region_rows],color=colors[arm],alpha=.7)
        if global_row["native_cutoff"] is not None:ax.axhline(global_row["native_cutoff"],color="black",ls="--",label="Global")
        ax.set_xticks(range(6),[r["region_id"] for r in region_rows],rotation=60,ha="right");ax.set_title(LABELS[arm]);ax.set_ylabel("Dimensionless" if arm.endswith("Z") else "Fluorescence units");ax.grid(axis="y",alpha=.2)
    save(fig,"regional_real_native_cutoffs_q1.png","Native setup cutoffs at q=1: six regions versus one global population",
         "Each axis retains its own native score units; bar height is not comparable across readouts. Dashed line: global cutoff.\nRegion boundaries change calibration populations only; Gamma filtering retains its full context and uses no region masks.",bottom=.25)
    weak=tables["crowding_pooled_weak_curves"]
    conditions=(None,8,12,16);labels=("Alone","8 px","12 px","16 px")
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for score in ("A","C","Z"):
        for window,style in ((13,"-"),(3,"--")):
            rs=[next(r for r in weak if r["readout"]==score and r["window"]==window and r["separation_px"]==sep and r["threshold_id"]=="q1" and r["match_radius_px"]==2) for sep in conditions]
            for ax,key in zip(axes.flat,("weak_prompt_200ms_coverage","weak_active_frame_coverage","false_proposals_per_10000um2_second","weak_event_coverage")):
                ax.plot(range(4),[r[key] for r in rs],style,color=score_colors[score],marker="o",label=f"{score} · {window}×{window}")
    for i,(ax,title) in enumerate(zip(axes.flat,("Weak events recovered within 200 ms /3","Weak active source/frame coverage","Whole-clip false proposals /10,000 µm² /s","Weak events recovered at any time /3"))):
        ax.set_title(title);ax.set_xticks(range(4),labels);ax.set_xlabel("Neighbor center separation");ax.grid(alpha=.2)
        if i!=2:ax.set_ylim(-.03,1.03)
    save(fig,"crowding_weak_q1_r2.png","Crowding mechanism test: primary 2 px localization at q=1",
         "Three new paired seeds per condition; identical setup and noise within each seed. Weak-source misses stay in denominators.\nFalse proposals refer to the whole clip, not only the weak source. Each selector uses its own setup-frozen cutoff.",*axes[0,0].get_legend_handles_labels(),bottom=.25)
    fig,axes=plt.subplots(1,4,figsize=(15,5.7))
    for ax,sep,label in zip(axes,conditions,labels):
        for score in ("A","C","Z"):
            for window,style in ((13,"-"),(3,"--")):
                row=next(r for r in weak if r["readout"]==score and r["window"]==window and r["separation_px"]==sep and r["threshold_id"]=="q1" and r["match_radius_px"]==2)
                ax.plot(DEADLINES,[ratio(d["recovered_by_deadline"],d["event_count"]) for d in row["weak_deadline_rows"]],style,color=score_colors[score],marker="o",ms=3,label=f"{score} · {window}×{window}")
        ax.set_title(label);ax.set_ylim(-.03,1.03);ax.set_xlabel("Allowed sampled-onset delay (ms)");ax.grid(alpha=.2)
    axes[0].set_ylabel("Fraction of all three weak events recovered")
    save(fig,"crowding_weak_prompt_deadlines_r2.png","Weak-source prompt recovery: all events retained",
         "Primary 2 px matching; q=1. These are sampled fluorescence-onset delays, excluding computation and actuation.\nThe 200 ms reporting point is not an established fish-control requirement.",*axes[0].get_legend_handles_labels(),bottom=.29)
    fig,axes=plt.subplots(3,4,figsize=(15,11))
    metric_keys=("weak_prompt_200ms_coverage","weak_active_frame_coverage","false_proposals_per_10000um2_second")
    metric_titles=("Weak prompt recovery ≤200 ms","Weak active-frame coverage","Whole-clip false proposals /area /s")
    for column,(sep,label) in enumerate(zip(conditions,labels)):
        for score in ("A","C","Z"):
            for window,style in ((13,"-"),(3,"--")):
                rs=sorted([r for r in weak if r["readout"]==score and r["window"]==window and r["separation_px"]==sep and r["match_radius_px"]==2 and r["setup_budget_per_reference_area_frame"] is not None],key=lambda r:r["setup_budget_per_reference_area_frame"])
                for i,key in enumerate(metric_keys):axes[i,column].plot([r["setup_budget_per_reference_area_frame"] for r in rs],[r[key] for r in rs],style,color=score_colors[score],marker="o",ms=2,label=f"{score} · {window}×{window}")
        for i in range(3):
            ax=axes[i,column];ax.set_title(label if i==0 else "");ax.set_xscale("symlog",linthresh=.5);ax.set_xlabel("Setup target q");ax.grid(alpha=.2)
            if column==0:ax.set_ylabel(metric_titles[i])
            if i<2:ax.set_ylim(-.03,1.03)
    save(fig,"crowding_weak_budget_curves_r2.png","Weak-source sensitivity to the frozen setup budget",
         "Primary 2 px matching, three paired seeds. False-proposal area unit is 10,000 µm². Numeric tables retain both endpoints.\nCurves are descriptive across declared q values; no application-selected winner or significance test is reported.",*axes[0,0].get_legend_handles_labels(),bottom=.20)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for column,score in enumerate(("A","C","Z")):
        for radius,style in ((2,"-"),(6,":")):
            for window,color in ((13,"#2864a0"),(3,"#ae5b18")):
                rs=[next(r for r in weak if r["readout"]==score and r["window"]==window and r["separation_px"]==sep and r["threshold_id"]=="q1" and r["match_radius_px"]==radius) for sep in conditions]
                for ax,key in zip(axes[:,column],("weak_prompt_200ms_coverage","weak_active_frame_coverage")):
                    ax.plot(range(4),[r[key] for r in rs],style,color=color,marker="o",label=f"{window}×{window}, r={radius} px")
        axes[0,column].set_title(f"Level {score}")
        for ax in axes[:,column]:ax.set_xticks(range(4),labels);ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Weak prompt recovery ≤200 ms");axes[1,0].set_ylabel("Weak active-frame coverage")
    save(fig,"crowding_localization_r2_r6_q1.png","Primary 2 px localization versus legacy 6 px matching",
         "Solid: primary 2 px disjoint source disks. Dotted: legacy 6 px neighborhoods, which can overlap.\nMatched points under the legacy rule do not establish resolved biological sources. All comparisons retain the three paired seeds.",*axes[0,0].get_legend_handles_labels(),bottom=.24)
    return figures


def _pct(value):
    return "undefined" if value is None else f"{100*value:.1f}%"


def _report_text(tables, figures, audit_status):
    fixed=[r for r in tables["regional_pooled_curves"] if r["threshold_id"]=="q1"]
    real={(r["base_arm"],r["calibration_method"]):r for r in tables["regional_real_curves"] if r["threshold_id"]=="q1"}
    lines=["# Regional calibration and crowded-source follow-up", "",
        "This development study separates setup calibration from score construction, then tests whether spatial peak selection contributes to crowded-source misses. It does not select a new optimum or establish significance, universal Gamma-LS necessity, independent-recording generalization, neuronal onset timing, or feedback-control readiness.", "",
        "## Completion state", "", f"**{audit_status['status']}**. Numerical evidence covers 242 cells and 3,140 cell/threshold/radius rows. Scientific media covers only each frozen q=1 state; other thresholds are numerical evidence. Figure visual QA and final campaign closure are separate gates.", "",
        "## Regional calibration at q=1", "",
        "Five scores—activity-level input X, target A, local contrast A−M, Gamma-LS Z, and signed-change Z—use one global calibration population or a geometry-fixed 2×3 grid. The kernel and positive NMS stream stay fixed. Regional cutoffs filter that stream without score reranking; truth assignment is recomputed after filtering. The same integer setup budget is allocated by area with deterministic Hamilton rounding. This creates no masks in the Gamma stencil.", "",
        "| Score | Setup population | Synthetic precision | Active-frame coverage | Events ≤200 ms | False proposals | Real windows /76 | Real proposals |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in READOUTS:
        for method in ("global","regional"):
            row=next(r for r in fixed if r["base_arm"]==arm and r["calibration_method"]==method)
            d=next(v for v in row["deadline_rows"] if v["deadline_ms"]==200);recording=real[(arm,method)]
            lines.append(f"| {LABELS[arm]} | {method} | {_pct(row['precision'])} | {_pct(row['active_frame_coverage'])} | {d['recovered_by_deadline']}/{d['event_count']} | {row['false_positive_count']} | {recording['matched_known_positive_count']} | {recording['proposal_count']} |")
    lines += ["", "Synthetic totals pool 16 pilot clips, with per-case rows retained. Precision counts matched frame/location proposals; active-frame coverage counts matched source/frames over the full declared fluorescence support. Equal setup budgets do not imply equal application burden or a biological false-alarm probability. Null precision at no output is preserved. The numeric all-positive and no-output endpoints remain separate from the eight q targets.", "",
        "For the recording, all arms use the same 76 eligible broad windows during UI 1800–2359 (560 frames, 11.2 s). Three original windows were excluded by common crop geometry. Unmatched proposals are unknown; real precision, exhaustive sensitivity and per-ROI onset accuracy are unavailable. The prepared raw-panel review has no accepted exhaustive annotation at the frozen protocol. Rendering that review does not accept labels.", "",
        "## Recording setup cutoffs by region at q=1", "",
        "Each table retains one score's native units. Region IDs are row-major in the fixed 2×3 grid over the eligible interior; the same total integer setup budget is divided by eligible area. Counts are actual strict exceedances and may underfill because of tied scores. These boundaries never mask the Gamma kernels.", ""]
    for arm in READOUTS:
        rs=[r for r in tables["real_regional_q1_cutoffs"] if r["base_arm"]==arm]
        first=rs[0]
        cutoff="no output" if first["global_native_cutoff"] is None else f"{first['global_native_cutoff']:.9g}"
        lines += [f"**{LABELS[arm]} — {first['score_units'].replace('_',' ')}.** Global cutoff {cutoff}; global setup count/budget {first['global_setup_proposal_count']}/{first['global_setup_proposal_budget']}.", "",
                  "| Region | Native cutoff | Setup budget | Actual setup count |", "|---|---:|---:|---:|"]
        for row in rs:
            cutoff="no output" if row["native_cutoff"] is None else f"{row['native_cutoff']:.9g}"
            lines.append(f"| {row['region_id']} | {cutoff} | {row['setup_proposal_budget']} | {row['setup_proposal_count']} |")
        lines.append("")
    lines += ["## Crowding at primary 2 px localization", "",
        "Twelve new clips use three paired seeds, each with a weak source alone or a stronger neighbor 8, 12 or 16 pixels away. Setup and noise are identical within each seed. Activity-level A, A−M and Z are compared under 13×13 versus 3×3 local maxima, each with the same border, score ordering and strictly greater-than-6-pixel proposal separation. Each selector is separately calibrated. These fixed source widths, amplitudes and kinetics are technical stress settings, not measured neuron or indicator parameters.", "",
        "| Condition | Score | Selector | Weak events ≤200 ms /3 | Weak matched active frames | Whole-clip false proposals | Whole-clip proposals |",
        "|---|---|---|---:|---:|---:|---:|"]
    for separation in (None,8,12,16):
        for score in ("A","C","Z"):
            for window in (13,3):
                row=next(r for r in tables["crowding_pooled_weak_curves"] if r["threshold_id"]=="q1" and r["match_radius_px"]==2 and r["separation_px"]==separation and r["readout"]==score and r["window"]==window)
                d=next(v for v in row["weak_deadline_rows"] if v["deadline_ms"]==200)
                lines.append(f"| {'Alone' if separation is None else str(separation)+' px'} | {score} | {window}×{window} | {d['recovered_by_deadline']}/{d['event_count']} | {row['weak_matched_active_frame_count']}/{row['weak_active_frame_count']} | {row['false_positive_count']} | {row['proposal_count']} |")
    lines += ["", "The primary 2 px source disks are disjoint at every paired separation. The legacy 6 px results are retained separately; overlapping match neighborhoods cannot establish biological source separation. Weak-source roles come from the exact sealed event IDs, not event ordering or visual selection. Whole-clip false and duplicate proposals cannot be attributed exclusively to the weak source.", "",
        "Prompt-recovery curves retain every weak event, including misses. Delay is measured from sampled simulated fluorescence onset; computation, acquisition and actuation are excluded. The 200 ms point is a reporting convention, not an established fish-control deadline. A single matched frame can recover an event while most active frames remain uncovered. Per-seed counts and selector differences are provided; frames are not independent experimental replicates and no significance test is performed.", "",
        "## Figures", ""]
    lines += [f"- [{name}]({name})" for name in figures]
    lines += ["", "## Exact tables and provenance", "",
        "- [Regional per-case curves](regional_case_curves.tsv) and [paired regional-minus-global changes](regional_paired_changes.tsv).",
        "- [Recording q=1 regional cutoffs and actual setup counts](real_regional_q1_cutoffs.tsv); [all native cutoffs and regional budgets](calibration_thresholds.tsv).",
        "- [Crowding per-seed operating counts](crowding_q1_per_seed.tsv) and [paired selector changes](crowding_selector_paired_q1.tsv).",
        "- [All crowding source/event/threshold rows](crowding_event_curves.tsv), including neighbor roles and both localization radii.",
        "- [Frozen source manifest](sources_manifest.json) and [study protocol](../protocol.json).", "",
        "All plots compare declared setup q or native cutoffs within a score's units. X/A/contrast retain native fluorescence units; Z is dimensionless. Regional calibration has six cutoffs, not a meaningful single pooled native cutoff. The internal zero used to evaluate an already-filtered stream is not presented as its setup threshold.", "",
        "Stage-by-stage mechanism diagnostics are a separate parent-owned artifact. Configured-center pooled/per-center calibration and nuisance at an actual configured source remain deferred; this study does not answer those monitoring questions. The reused pilot material and new simulation seeds remain development evidence. A subsequent preferred configuration needs separate confirmation and a fresh runtime measurement.", ""]
    return "\n".join(lines)


def _mutable_root(root):
    root=Path(root).resolve()
    require(not (root/"completion_manifest.json").exists(), "Completed roots are immutable; create a new explicitly versioned report/root")
    return root


def generate(root=DEFAULT_ROOT):
    root=_mutable_root(root);out=root/"report"
    require(not out.exists() or not any(out.iterdir()), "Preserve existing or partial report; generation requires a new report directory")
    data=_load_inputs(root);tables=make_tables(data);audit=_audit_status(root,data["protocol"])
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        write_json(out/(name+".json"),rows);write_tsv(out/(name+".tsv"),rows)
    sources=dict(status="VERIFIED_NUMERICAL_SOURCES", reporter=binding(Path(__file__)), inputs=data["sources"],
                 inherited_large_sources=data["inherited"],
                 scope="Small numeric tables, code, source labels and seals verified; no arrays or large prefix ledgers read or rescored")
    write_json(out/"sources_manifest.json",sources)
    figures=_plot_figures(out,tables)
    (out/"REPORT.md").write_text(_report_text(tables,figures,audit))
    manifest=dict(status="GENERATED", report_stage=audit["status"], numerical_complete=True,
        scientific_audit_complete=audit["scientific_audit_complete"], audit_verification=audit,
        figure_count=len(figures), figures=figures, visual_qa_complete=False,
        table_row_counts={k:len(v) for k,v in tables.items()}, reporter=binding(Path(__file__)), inputs=data["sources"],
        artifacts=[dict(binding(p),path=str(p.relative_to(out))) for p in sorted(out.iterdir()) if p.is_file()])
    write_json(out/"manifest.json",manifest)
    print(json.dumps(dict(status=manifest["status"],report_stage=manifest["report_stage"],figures=len(figures))),flush=True)
    return manifest


def update(root=DEFAULT_ROOT):
    """Refresh completion text only before final closure; preserve numeric/PNG bytes."""
    root=_mutable_root(root);out=root/"report";manifest=read(out/"manifest.json")
    for record in [manifest["reporter"],*manifest["inputs"]]:
        actual=binding(record["path"])
        require(all(actual[k]==record[k] for k in ("sha256","size_bytes")), "Report source changed; update cannot regenerate evidence")
    for record in manifest["artifacts"]:
        actual=binding(out/record["path"])
        require(all(actual[k]==record[k] for k in ("sha256","size_bytes")), "Existing report artifact changed")
    audit=_audit_status(root,read(root/"protocol.json"))
    tables={key:read(out/(key+".json")) for key in manifest["table_row_counts"]}
    (out/"REPORT.md").write_text(_report_text(tables,manifest["figures"],audit))
    manifest.update(report_stage=audit["status"],scientific_audit_complete=audit["scientific_audit_complete"],audit_verification=audit)
    manifest["artifacts"]=[dict(binding(out/r["path"]),path=r["path"]) for r in manifest["artifacts"]]
    write_json(out/"manifest.json",manifest)
    print(json.dumps(dict(status="UPDATED_COMPLETION_TEXT_ONLY",report_stage=audit["status"],visual_qa_complete=manifest["visual_qa_complete"])),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("generate","update"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    globals()[args.command](args.root)


if __name__=="__main__":
    main()
