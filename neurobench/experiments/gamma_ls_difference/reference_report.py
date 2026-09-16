"""Standalone, source-bound figures for the nine-reference development study.

This postprocessor never predicts, recalibrates, reranks, or selects a winner.
Generation requires all 189 seals and completed evaluation. Status updates
preserve every scientific table/figure and refuse completed campaign roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "Outputs/GammaLSReference/reference_20260915_r1"
ARMS = tuple(f"mean{m}_n{n}" for m in ("2of3", "1", "4of3") for n in (3, 9, 15))
THRESHOLDS = ("q0", "q0.25", "q0.5", "q1", "q2", "q4", "q8", "q16", "all_positive", "no_output")
DEADLINES = (0, 20, 40, 60, 100, 200, 500)
SCENARIOS = ("alone", "sep8", "sep12", "sep16", "stationary", "variance_correlation", "shared_brightness_motion")
AUDIT_METADATA = {"summary.json", "status.json", "run_contract.json", "source_manifest.json",
                  "artifact_index.json", "llm_context.json", "validation.json", "inventory.json"}
ORDER_COLORS = {3: "#2471A3", 9: "#C57E12", 15: "#7E477F"}
MEAN_COLORS = {"2of3": "#2471A3", "1": "#C57E12", "4of3": "#7E477F"}
MEAN_STYLES = {"2of3": ":", "1": "-", "4of3": "--"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


class Sources:
    """One invocation's verified sources; changed files never reuse a hash."""
    def __init__(self):
        self.records, self.stats = {}, {}

    def bind(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if str(path) in self.records:
            require(stamp == self.stats[str(path)], f"Source changed during report: {path}")
            return self.records[str(path)]
        record = _binding(path)
        after = path.stat()
        require(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"Source changed while hashing: {path}")
        self.records[str(path)], self.stats[str(path)] = record, stamp
        return record

    def check(self, record, expected_path=None):
        if expected_path is not None:
            require(Path(record["path"]).resolve() == Path(expected_path).resolve(), "Unexpected evidence path")
        actual = self.bind(record["path"])
        require(all(actual[k] == record[k] for k in ("sha256", "size_bytes")), f"Changed evidence: {record['path']}")
        return actual

    def read(self, path):
        self.bind(path)
        result = _read(path)
        self.bind(path)
        return result

    def finish(self):
        for path in list(self.records):
            self.bind(path)
        return sorted(self.records.values(), key=lambda b: b["path"])


def _mutable(root):
    root = Path(root).resolve()
    require(not (root / "completion_manifest.json").exists(), "Completed roots are immutable")
    return root


def _matrix(protocol):
    cases = protocol["cases"]
    require(len(cases) == 21 and len({c["case_id"] for c in cases}) == 21, "Expected 21 unique cases")
    require(len(protocol["references"]) == 9 and {r["arm_id"] for r in protocol["references"]} == set(ARMS), "Expected exact nine references")
    require(protocol["expected_cells"] == 189 and protocol["expected_curve_rows"] == 3780, "Unexpected numerical matrix")
    return [(c["case_id"], a) for c in cases for a in ARMS]


def _seal_gate(root, protocol, sources):
    """Do this before opening any file containing activity truth or outcomes."""
    states = _matrix(protocol)
    require(all((root / "cells" / c / a / "sealed.json").is_file() for c, a in states), "All 189 candidate seals are required before activity data")
    seals = {}
    for case, arm in states:
        seal = sources.read(root / "cells" / case / arm / "sealed.json")
        require(seal.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", "Candidate seal is incomplete")
        seals[(case, arm)] = seal
    for case in {c for c, _ in states}:
        require(len({tuple(seals[(case,a)]["stages"][k]["sha256"] for k in ("Raw","Input","A")) for a in ARMS})==1,
                "Raw, conditioned input or fixed target differs across references")
    require(sources.read(root / "computation_complete.json") == dict(status="PASS", cells=189), "Scoring is incomplete")
    require(sources.read(root / "evaluation_complete.json") == dict(status="PASS", cells=189, curve_rows=3780), "All reference evaluations are required")
    return seals


def _scenario(case):
    return ("alone" if case.get("separation_px") is None else f"sep{case['separation_px']}") if case["kind"] == "crowding" else case["null_kind"]


def _key(row):
    return row["case_id"], row["arm_id"], row["threshold_id"], float(row["match_radius_px"])


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _area_time(row, pixel_um=.5):
    return row["eligible_area_px"] * pixel_um**2 * row["exposure_seconds"]


def validate_event_counts(curves, events):
    """Reconcile exported event and full-field counts without rematching rows."""
    grouped=defaultdict(list)
    for row in events:grouped[(row["threshold_id"],float(row["match_radius_px"]))].append(row)
    for row in curves:
        group=grouped[(row["threshold_id"],float(row["match_radius_px"]))]
        require(sum(r["active_frame_count"] for r in group)==row["active_region_frame_count"] and
                sum(r["matched_active_frame_count"] for r in group)==row["true_positive_count"] and
                len(group)==row["event_count"] and
                sum(r["first_delay_ms"] is not None for r in group)==row["recovered_event_count"],
                "Event and full-field metric counts disagree")
        require(row["false_positive_count"]==row["proposal_count"]-row["true_positive_count"],"Proposal partition differs")


def _load(root):
    sources = Sources()
    protocol = sources.read(root / "protocol.json")
    preflight = sources.read(root / "preflight.json")
    require(preflight["status"] == "PASS" and preflight["protocol_sha256"] == sources.bind(root / "protocol.json")["sha256"], "Protocol/preflight differ")
    for record in protocol["code_bindings"] + protocol["kernel_bindings"]:
        sources.check(record)
    require(protocol["frame_rate_hz"] == 50 and protocol["pixel_size_um"] == .5, "Physical units differ")
    seals = _seal_gate(root, protocol, sources)
    # No activity truth or metric files have been read above this line.
    for name in ("baseline_completion", "baseline_protocol"):
        sources.check(protocol[name])
    prep = sources.read(root / "datasets_complete.json")
    require(prep["status"] == "PASS" and prep["datasets"] == 21 and prep["paired_setup_byte_equal"], "Preparation incomplete")
    cases = {c["case_id"]: c for c in protocol["cases"]}
    require({(_scenario(c), c["seed"]) for c in cases.values()} == {(s, seed) for s in SCENARIOS for seed in (20260916,20260917,20260918)}, "Paired scenario/seed matrix differs")
    metadata, truth = {}, {}
    for case, c in cases.items():
        folder = root / "datasets" / case
        authority = {Path(b["path"]).resolve(): b for b in seals[(case, ARMS[0])]["dataset_bindings"]}
        for name in ("metadata.json", "experts.json", "active.json", "prepared.json"):
            sources.check(authority[(folder / name).resolve()], folder / name)
        metadata[case] = sources.read(folder / "metadata.json")
        events = sources.read(folder / "experts.json")
        truth[case] = {r["event_id"]: r for r in events}
        require(len(truth[case]) == len(events), "Duplicate synthetic event identity")
        meta = metadata[case]
        require(meta["truth_mode"] == "fully_synthetic" and meta["source_frames_ui"] == list(range(1,465)), "Unexpected truth or source-frame scope")
        require(meta["application_source_frames_ui"] == list(range(165,465)), "Application interval differs")
        if c["kind"] == "null":
            require(not events and meta["event_count"] == 0, "Null dataset contains neural truth")
        else:
            require({r["source_role"] for r in events} == ({"weak"} if c["separation_px"] is None else {"weak", "neighbor"}), "Crowding roles differ")
    curves, events, calibrations, inherited = [], [], [], []
    paired_calibration={}
    for (case, arm), seal in seals.items():
        folder = root / "cells" / case / arm
        evaluated = sources.read(folder / "evaluated.json")
        require(evaluated["status"] == "PASS", "Cell evaluation incomplete")
        sources.check(evaluated["seal"], folder / "sealed.json")
        for record in evaluated["outputs"]:
            sources.check(record)
        require((folder / "curves.json").resolve() in {Path(b["path"]).resolve() for b in evaluated["outputs"]}, "Unbound curves")
        for key in ("calibration", "threshold_plan"):
            sources.check(seal[key], folder / f"{key}.json")
        op = sources.read(folder / "calibration.json")
        plan = sources.read(folder / "threshold_plan.json")
        require({r["threshold_id"] for r in plan} == set(THRESHOLDS) and len(plan) == 10, "Cutoff grid differs")
        require(op["threshold_id"] == "q1" and op["threshold_frozen_from_calibration_only"] and op["window"] == 3, "Operating protocol differs")
        pair_key=(cases[case]["seed"],arm)
        calibrated=(plan,op["scale_floor"])
        require(pair_key not in paired_calibration or paired_calibration[pair_key]==calibrated,
                "Paired conditions do not share identical setup cutoffs and floor")
        paired_calibration[pair_key]=calibrated
        current = sources.read(folder / "curves.json")
        keys = [_key(r) for r in current["curve_rows"]]
        require(len(keys) == len(set(keys)) == 20 and set(keys) == {(case,arm,t,r) for t in THRESHOLDS for r in (2.,6.)}, "Cell curve inventory differs")
        settings = {r["threshold_id"]: r for r in plan}
        for r in current["curve_rows"]:
            require(r["threshold_z"] == settings[r["threshold_id"]]["threshold"], "Native threshold must not be replaced by post-filter zero")
        curves.extend(current["curve_rows"])
        for row in current["event_rows"]:
            require(row["event_id"] in truth[case], "Event identity missing from declared source truth")
            origin = truth[case][row["event_id"]]
            require(row["source_start_ui"] == origin["source_start_ui"] and row["active_frame_count"] == 246, "Weak/neighbor active interval differs")
            events.append(dict(row, case_id=case, arm_id=arm, source_role=origin["source_role"], seed=cases[case]["seed"], scenario=_scenario(cases[case])))
        expected_events = {(event,t,r) for event in truth[case] for t in THRESHOLDS for r in (2.,6.)}
        event_keys = [(r["event_id"],r["threshold_id"],float(r["match_radius_px"])) for r in current["event_rows"]]
        require(len(event_keys) == len(set(event_keys)) and set(event_keys) == expected_events, "Event curve inventory differs")
        validate_event_counts(current["curve_rows"], current["event_rows"])
        calibrations.extend(dict(r,case_id=case,arm_id=arm,scenario=_scenario(cases[case]),seed=cases[case]["seed"],scale_floor=op["scale_floor"]) for r in plan)
        inherited.append(dict(case_id=case,arm_id=arm,seal=sources.bind(folder/"sealed.json"),
                              large_sources={k:seal[k] for k in ("prefix","setup_prefix","audit_candidates")},stages=seal["stages"]))
    aggregate = sources.read(root / "all_curves.json")
    sources.bind(root / "all_curves.tsv")
    require(len(aggregate) == len(curves) == 3780 and {_key(r):r for r in aggregate} == {_key(r):r for r in curves}, "Aggregate differs from sealed cell curves")
    replication = sources.read(root / "baseline_replication.json")
    require(replication["status"] == "PASS" and len(replication["cells"]) == 12 and
            {(r["case_id"],r["arm_id"]) for r in replication["cells"]} == {(c,"mean1_n9") for c,v in cases.items() if v["kind"]=="crowding"} and
            all(all(r[k] for k in ("all_thresholds_equal","metrics_equal","stages_equal","candidates_equal")) for r in replication["cells"]), "Anchor replication incomplete")
    pairing=sources.read(root/"paired_setup_calibration_check.json")
    require(pairing["status"]=="PASS" and len(pairing["groups"])==27 and
            {(r["seed"],r["arm_id"]) for r in pairing["groups"]}==set(paired_calibration) and
            all(r["cases"]==7 and r["all_threshold_plans_exact"] and r["floors_exact"] and r["max_floor_difference"]==0 for r in pairing["groups"]),
            "Independent paired setup receipt differs")
    for row in curves:
        row.update(scenario=_scenario(cases[row["case_id"]]), seed=cases[row["case_id"]]["seed"],
                   false_proposals_per_10000_um2_s=10000*row["false_positive_count"]/_area_time(row))
    return dict(protocol=protocol,sources=sources,seals=seals,metadata=metadata,truth=truth,curves=curves,
                events=events,calibrations=calibrations,inherited=inherited)


def pool_curves(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"],row["arm_id"],row["threshold_id"],float(row["match_radius_px"]))].append(row)
    output=[]
    for (scenario,arm,threshold,radius), group in sorted(groups.items()):
        require(len({r["seed"] for r in group}) == len(group), "Duplicate seed within pooled curve")
        row=dict(scenario=scenario,arm_id=arm,threshold_id=threshold,match_radius_px=radius,
                 seed_count=len(group),seeds=sorted(r["seed"] for r in group))
        for key in ("proposal_count","true_positive_count","false_positive_count","active_region_frame_count",
                    "event_count","recovered_event_count","duplicate_near_active_region_count","application_frame_count"):
            row[key]=sum(r[key] for r in group)
        row.update(precision=_ratio(row["true_positive_count"],row["proposal_count"]),
                   framewise_sensitivity=_ratio(row["true_positive_count"],row["active_region_frame_count"]),
                   event_window_coverage=_ratio(row["recovered_event_count"],row["event_count"]),
                   false_proposals_per_10000_um2_s=10000*row["false_positive_count"]/sum(_area_time(r) for r in group))
        output.append(row)
    return output


def pool_events(rows, curves):
    groups=defaultdict(list)
    lookup={(r["scenario"],r["arm_id"],r["threshold_id"],r["match_radius_px"]):r for r in curves}
    for r in rows:
        groups[(r["scenario"],r["arm_id"],r["threshold_id"],float(r["match_radius_px"]),r["source_role"])].append(r)
    output=[]
    for (scenario,arm,threshold,radius,role), group in sorted(groups.items()):
        require(len({(r["case_id"],r["event_id"]) for r in group}) == len(group), "Duplicate event in pooled denominator")
        count=sum(r["active_frame_count"] for r in group); hits=sum(r["matched_active_frame_count"] for r in group)
        base=lookup[(scenario,arm,threshold,radius)]
        for deadline in DEADLINES:
            recovered=sum(r["first_delay_ms"] is not None and r["first_delay_ms"] <= deadline+1e-9 for r in group)
            output.append(dict(scenario=scenario,arm_id=arm,threshold_id=threshold,match_radius_px=radius,source_role=role,
                deadline_ms=deadline,event_count=len(group),recovered_by_deadline=recovered,
                deadline_recovery=recovered/len(group),active_frame_count=count,matched_active_frame_count=hits,
                active_frame_coverage=_ratio(hits,count),false_positive_count=base["false_positive_count"],
                proposal_count=base["proposal_count"],false_proposals_per_10000_um2_s=base["false_proposals_per_10000_um2_s"]))
    return output


def _geometry(data):
    import numpy as np
    summaries,profiles,mass=[],[],[]
    by_name={Path(b["path"]).stem:b for b in data["protocol"]["kernel_bindings"]}
    for r in data["protocol"]["references"]:
        arm=r["arm_id"];spec=r["spec"];w=np.load(data["sources"].check(by_name[arm])["path"])
        require(hashlib.sha256(w.tobytes(order="C")).hexdigest()==r["kernel_sha256_float64_c_order"], "Kernel payload differs from geometry metadata")
        half=w.shape[0]//2
        summaries.append(dict(arm_id=arm,mean_factor=spec["mean_factor"],reference_n=spec["reference_n"],
            requested_mean_radius_px=spec["requested_mean_radius_px"],realized_mean_radius_px=r["realized_mean_radius_px"],
            realized_radius_sd_px=r["realized_radius_sd_px"],reference_mu_per_px=spec["reference_mu_per_px"],
            per_pixel_mode_radius_px=spec["profile_mode_radius_px"],spatial_half_width_px=half,
            effective_sample_count=r["effective_sample_count"],retained_mass=r["retention"]["discrete_retained_mass_estimate"],
            preserved_anchor=spec["preserved_anchor"]))
        profiles.extend(dict(arm_id=arm,radius_px=i,normalized_pixel_weight=float(w[half,half+i])) for i in range(half+1))
        mass.extend(dict(arm_id=arm,radial_bin_low_px=i,radial_bin_high_px=i+1,normalized_annular_mass=v)
                    for i,v in enumerate(r["radial_bin_mass"]))
    return summaries,profiles,mass


def _traces(data):
    import numpy as np
    rows=[];cached={}
    cases=[c for c in data["protocol"]["cases"] if c["kind"]=="crowding" and c["seed"]==20260916 and c["separation_px"] in (None,8)]
    require(len(cases)==2,"Fixed trace pair absent")
    for case in cases:
        name=case["case_id"];truth=[r for r in data["truth"][name].values() if r["source_role"]=="weak"]
        require(len(truth)==1 and (truth[0]["x_px"],truth[0]["y_px"])==(68.,64.),"Fixed weak center differs")
        for arm in ARMS:
            seal=data["seals"][(name,arm)];op=data["sources"].read(seal["calibration"]["path"])
            values={}
            for stage in ("A","M","Spread","C","Z"):
                b=seal["stages"][stage];key=(str(Path(b["path"]).resolve()),b["sha256"])
                if key not in cached:
                    data["sources"].check(b)
                    a=np.load(b["path"],mmap_mode="r")
                    require(a.shape==(464,128,128),"Trace stage geometry differs")
                    cached[key]=np.asarray(a[:,64,68],dtype=float).copy()
                values[stage]=cached[key]
            for i,frame in enumerate(data["metadata"][name]["source_frames_ui"]):
                rows.append(dict(case_id=name,scenario=_scenario(case),arm_id=arm,seed=20260916,source_frame_ui=frame,
                    x_px=68,y_px=64,source_onset_ui=truth[0]["source_start_ui"],threshold_q1=op["threshold"],
                    scale_floor=op["scale_floor"],floor_active=bool(values["Spread"][i]<op["scale_floor"]),
                    **{k:float(v[i]) for k,v in values.items()}))
    return rows


def make_tables(data):
    pooled=pool_curves(data["curves"]); deadlines=pool_events(data["events"],pooled)
    summary,profile,mass=_geometry(data)
    return dict(curves=data["curves"],event_curves=data["events"],calibration=data["calibrations"],
        pooled_curves=pooled,pooled_deadlines=deadlines,kernel_summary=summary,kernel_profiles=profile,
        kernel_radial_mass=mass,mechanism_traces=_traces(data),
        kernel_source_overlap=[dict(arm_id=r["arm_id"],**v) for r in data["protocol"]["references"] for v in r["source_overlap"]],
        q1_source_summary=[r for r in deadlines if r["threshold_id"]=="q1" and r["deadline_ms"]==100],
        q1_event_replicates=[r for r in data["events"] if r["threshold_id"]=="q1"],
        q1_null_replicates=[r for r in data["curves"] if r["case_kind"]=="null" and r["threshold_id"]=="q1" and r["match_radius_px"]==2.])


def _tsv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t");writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _audit_status(root, protocol, sources):
    path=root/"audit_complete.json"
    if not path.exists():
        return dict(status="NUMERICAL_COMPLETE_AUDITS_PENDING",scientific_audit_complete=False,source=None)
    audit=sources.read(path)
    require(audit.get("status")=="PASS" and audit.get("cells")==189 and audit.get("new_cells")==177 and audit.get("reused_cells")==12,"Audit aggregate exists but is not complete")
    keys=[(r["case_id"],r["arm_id"]) for r in audit["audits"]]
    require(len(keys)==len(set(keys))==189 and set(keys)==set(_matrix(protocol)),"Audit matrix incomplete")
    require({Path(b["path"]).name for b in audit["validation_code"]} ==
            {"reference_validate.py","reference_media.py","followup_validate.py","followup_audit.py","two_stencil_audit.py"},"Audit validation code inventory differs")
    for b in audit["validation_code"]:sources.check(b)
    for key,name in (("protocol","protocol.json"),("evaluation","evaluation_complete.json"),("all_curves","all_curves.json")):
        sources.check(audit[key],root/name)
    metadata=[]
    for row in audit["audits"]:
        require(set(row["metadata_bindings"])==AUDIT_METADATA,"Audit metadata inventory differs")
        for name,b in row["metadata_bindings"].items():
            require(Path(b["path"]).name==name,"Audit metadata name differs")
            metadata.append(sources.check(b))
        summary=sources.read(row["metadata_bindings"]["summary.json"]["path"])
        status=sources.read(row["metadata_bindings"]["status.json"]["path"])
        validation=sources.read(row["metadata_bindings"]["validation.json"]["path"])
        require(summary==row["summary"] and summary["scientific_audit_complete"] is True and status["status"]=="complete" and
                status["scientific_audit_complete"] is True and validation["status"]=="passed" and not validation["failures"],"Audit completion fields disagree")
    return dict(status="AUDITS_VERIFIED_VISUAL_QA_SEPARATE",scientific_audit_complete=True,
                source=sources.bind(path),metadata_bindings=metadata,validation_code=audit["validation_code"],
                scope="Current compact audit metadata verified; complete indexed-file/media validation inherited from the bound aggregate validator.")


def _save(fig, out, stem, caption, tables):
    import matplotlib.pyplot as plt
    for ext in ("png","pdf"):
        fig.savefig(out/f"{stem}.{ext}",dpi=160,facecolor="white",bbox_inches="tight")
    plt.close(fig)
    return dict(id=stem,png=f"{stem}.png",pdf=f"{stem}.pdf",caption=caption,source_tables=tables)


def _style(axes):
    for ax in axes.flat:
        ax.grid(axis="y",color="#dddddd",lw=.5,zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)


def _null_ylim(rows):
    values=[float(r["false_proposals_per_10000_um2_s"]) for r in rows]
    require(values and all(math.isfinite(v) and v>=0 for v in values),"Invalid null burden extent")
    # A modest multiplicative margin is very small in symlog screen space.
    # 35% also leaves room for the full cross marker above the highest sample.
    return 0., max(1., 1.35*max(values))


def _plot(out,tables):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,"axes.labelsize":10,"pdf.fonttype":42})
    figures=[]; means=("2of3","1","4of3");orders=(3,9,15)
    geometry={r["arm_id"]:r for r in tables["kernel_summary"]}
    label=lambda m:f"Mean distance {geometry[f'mean{m}_n9']['realized_mean_radius_px']:.2f} px"
    order_handles=[Line2D([],[],color=ORDER_COLORS[n],label=f"Order {n}") for n in orders]
    mean_handles=[Line2D([],[],color=MEAN_COLORS[m],label=label(m)) for m in means]
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex="row",sharey="row");_style(axs)
    for col,m in enumerate(means):
        for n in orders:
            arm=f"mean{m}_n{n}";p=[r for r in tables["kernel_profiles"] if r["arm_id"]==arm];b=[r for r in tables["kernel_radial_mass"] if r["arm_id"]==arm]
            axs[0,col].plot([r["radius_px"] for r in p],[r["normalized_pixel_weight"] for r in p],color=ORDER_COLORS[n])
            axs[1,col].plot([r["radial_bin_low_px"]+.5 for r in b],[r["normalized_annular_mass"] for r in b],color=ORDER_COLORS[n])
        axs[0,col].set_title(label(m));axs[1,col].set_xlabel("Distance from target center (px)")
    axs[0,0].set_ylabel("Normalized weight per pixel\n(center-line samples)");axs[1,0].set_ylabel("Normalized mass in 1 px annulus")
    for ax in axs.flat:ax.set_xlim(left=0);ax.set_ylim(bottom=0)
    fig.legend(handles=order_handles,loc="upper center",ncol=3);fig.tight_layout(rect=(0,0,1,.94))
    figures.append(_save(fig,out,"reference_geometry","Matched mean distance is the pixel-mass-weighted radial mean, not the per-pixel profile mode. Annular mass includes all pixels in each 1 px radial bin. Effective sample counts and exact finite supports are in kernel_summary.tsv; they are geometric concentration measures, not independent samples.",["kernel_summary","kernel_profiles","kernel_radial_mass"]))
    pooled=tables["pooled_deadlines"];replicates=tables["q1_event_replicates"]
    for role,scenarios,stem in (("weak",SCENARIOS[:4],"weak_q1_recovery"),("neighbor",SCENARIOS[1:4],"neighbor_q1_recovery")):
        fig,axs=plt.subplots(2,3,figsize=(13,7.5),sharex=True,sharey="row");_style(axs)
        for col,n in enumerate(orders):
            for m in means:
                arm=f"mean{m}_n{n}";color=MEAN_COLORS[m]
                group=[r for r in pooled if r["arm_id"]==arm and r["threshold_id"]=="q1" and r["match_radius_px"]==2 and r["source_role"]==role and r["deadline_ms"]==100]
                by={r["scenario"]:r for r in group}
                for row,key in ((0,"active_frame_coverage"),(1,"deadline_recovery")):
                    axs[row,col].plot(range(len(scenarios)),[by[s][key] for s in scenarios],color=color,marker="o",lw=1.3)
                    for i,s in enumerate(scenarios):
                        seeds=sorted([r for r in replicates if r["arm_id"]==arm and r["match_radius_px"]==2 and r["source_role"]==role and r["scenario"]==s],key=lambda r:r["seed"])
                        vals=[r["matched_active_frame_count"]/r["active_frame_count"] if row==0 else float(r["first_delay_ms"] is not None and r["first_delay_ms"]<=100+1e-9) for r in seeds]
                        axs[row,col].scatter([i+j*.04 for j in (-1,0,1)],vals,color=color,alpha=.5,s=15,marker="x")
            axs[0,col].set_title(f"Reference order {n}")
            for row in range(2):
                axs[row,col].set_ylim(-.04,1.04);axs[row,col].yaxis.set_major_formatter(PercentFormatter(1))
                axs[row,col].set_xticks(range(len(scenarios)),["Alone" if s=="alone" else f"{s[3:]} px" for s in scenarios])
            axs[1,col].set_xlabel("Neighbor separation")
        axs[0,0].set_ylabel(f"{role.capitalize()} active frames matched");axs[1,0].set_ylabel(f"{role.capitalize()} events recovered ≤100 ms")
        fig.legend(handles=mean_handles,loc="upper center",ncol=3);fig.tight_layout(rect=(0,0,1,.94))
        figures.append(_save(fig,out,stem,"q = 1 setup target; inclusive 2 px matching. Lines pool counts over three seeds; crosses show each seed (small horizontal offsets only for visibility). Each source has 246 active frames per seed. All three events, including misses, remain in the deadline denominator. Frames are not independent replicates.",["q1_event_replicates","pooled_deadlines"]))
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),sharey=True);_style(axs)
    nulls=tables["q1_null_replicates"]
    for col,scenario in enumerate(SCENARIOS[4:]):
        for n in orders:
            points=[]
            for m in means:
                arm=f"mean{m}_n{n}";group=[r for r in nulls if r["arm_id"]==arm and r["scenario"]==scenario]
                x=geometry[arm]["realized_mean_radius_px"];vals=[r["false_proposals_per_10000_um2_s"] for r in group]
                axs[col].scatter([x]*len(vals),vals,color=ORDER_COLORS[n],marker="x",alpha=.55)
                points.append((x,10000*sum(r["false_positive_count"] for r in group)/sum(_area_time(r) for r in group)))
            axs[col].plot(*zip(*points),color=ORDER_COLORS[n],marker="o")
        axs[col].set_title({"stationary":"Stationary noise","variance_correlation":"Changing/correlated noise","shared_brightness_motion":"Brightness + moving artifact"}[scenario])
        axs[col].set_xlabel("Actual mean reference distance (px)");axs[col].set_yscale("symlog",linthresh=1)
    # Shared-axis limits must be finalized after every stressor is plotted.
    # Setting only the lower bound in the first panel disables later autoscale.
    null_limits=_null_ylim(nulls)
    for ax in axs:ax.set_ylim(*null_limits)
    axs[0].set_ylabel("False proposals / 10,000 µm² / s\n(symlog; linear below 1)")
    fig.legend(handles=order_handles,loc="upper center",ncol=3);fig.tight_layout(rect=(0,0,1,.92))
    figures.append(_save(fig,out,"null_q1_burden","Synthetic clips with no neural sources. Lines pool false counts divided by summed eligible area-time; crosses are the three seeds. Area = 116 × 116 pixels at 0.5 µm/pixel; application = 300 frames at 50 Hz. Null sensitivity/recovery is not applicable. Both nuisance clips combine declared stresses and do not isolate their ingredients.",["q1_null_replicates","pooled_curves"]))
    curve_lookup={(r["scenario"],r["arm_id"],r["threshold_id"]):r for r in tables["pooled_curves"] if r["match_radius_px"]==2}
    deadline_lookup={(r["scenario"],r["arm_id"],r["threshold_id"]):r for r in pooled if r["match_radius_px"]==2 and r["source_role"]=="weak" and r["deadline_ms"]==100}
    handles=order_handles+[Line2D([],[],color="#333333",ls=MEAN_STYLES[m],label=label(m)) for m in means]
    for kind,stem in (("precision","crowding_precision_sensitivity"),("deadline","weak_prompt_vs_false_burden")):
        fig,axs=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True);_style(axs)
        for ax,scenario in zip(axs.flat,SCENARIOS[:4]):
            for m in means:
                for n in orders:
                    arm=f"mean{m}_n{n}";selected=[];q1=None
                    for t in ("no_output",*THRESHOLDS[:-1]):
                        r=curve_lookup[(scenario,arm,t)]
                        point=(r["framewise_sensitivity"],r["precision"]) if kind=="precision" else (r["false_proposals_per_10000_um2_s"],deadline_lookup[(scenario,arm,t)]["deadline_recovery"])
                        if None not in point:selected.append(point)
                        if t=="q1":q1=point
                    if selected:ax.plot(*zip(*selected),color=ORDER_COLORS[n],ls=MEAN_STYLES[m],lw=1.2,alpha=.85)
                    if q1 is not None and None not in q1:ax.scatter(*q1,color=ORDER_COLORS[n],s=22,marker="o")
            ax.set_title("Weak source alone" if scenario=="alone" else f"Neighbor at {scenario[3:]} px")
            ax.set_ylim(-.02,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
            if kind=="precision":
                ax.set_xlim(-.02,1.02);ax.xaxis.set_major_formatter(PercentFormatter(1))
                ax.set_xlabel("All-source active-frame sensitivity");ax.set_ylabel("Precision of frame proposals")
            else:
                ax.set_xscale("symlog",linthresh=1);ax.set_xlim(left=0)
                ax.set_xlabel("Actual false proposals / 10,000 µm² / s\n(symlog; linear below 1)");ax.set_ylabel("Weak events recovered ≤100 ms")
        fig.legend(handles=handles,loc="upper center",ncol=3);fig.tight_layout(rect=(0,0,1,.91))
        figures.append(_save(fig,out,stem,"Inclusive 2 px matching; counts pooled over three paired seeds per condition. Lines connect the declared setup-budget cutoffs, including all-positive/no-output endpoints where the metric is defined; dots mark q = 1. Native numerical cutoffs differ between references. No application-label cutoff is chosen. False proposals include duplicates; precision is frame-proposal precision, not event precision.",["pooled_curves","pooled_deadlines","calibration"]))
    traces=tables["mechanism_traces"]
    for stages,stem in ((("A","M","Spread"),"paired_reference_stages"),(("C","Z"),"paired_contrast_score")):
        fig,axs=plt.subplots(len(stages),3,figsize=(14,3.2*len(stages)),sharex=True,sharey="row",squeeze=False);_style(axs)
        for col,m in enumerate(means):
            for row,stage in enumerate(stages):
                for n in orders:
                    if stage=="A" and n!=9:continue  # identical reused target for all orders
                    arm=f"mean{m}_n{n}"
                    for scenario,style in (("alone","--"),("sep8","-")):
                        selected=[r for r in traces if r["arm_id"]==arm and r["scenario"]==scenario]
                        ax=axs[row,col];color="#555555" if stage=="A" else ORDER_COLORS[n]
                        ax.plot([r["source_frame_ui"] for r in selected],[r[stage] for r in selected],color=color,ls=style,lw=1.15,alpha=.85)
                        if stage=="Z" and scenario=="alone":ax.axhline(selected[0]["threshold_q1"],color=color,lw=.75,ls=":")
                axs[row,col].axvline(185,color="#777777",lw=.7,ls="-.")
                axs[row,col].axvspan(65,164,color="#dddddd",alpha=.25)
                axs[row,col].set_xlim(1,464)
                if stage in ("C","Z"):axs[row,col].axhline(0,color="#bbbbbb",lw=.5)
                if stage=="Spread":axs[row,col].set_ylim(bottom=0)
                if col==0:axs[row,col].set_ylabel({"A":"Target response A","M":"Reference mean M","Spread":"Reference spread","C":"Target − reference","Z":"Local standardized score Z"}[stage]+("\n(dimensionless)" if stage=="Z" else "\n(native fluorescence units)"))
            axs[0,col].set_title(label(m));axs[-1,col].set_xlabel("Source frame (UI; 20 ms/sample)")
        trace_handles=order_handles+[Line2D([],[],color="#555555",ls="--",label="Weak alone"),Line2D([],[],color="#555555",ls="-",label="Neighbor 8 px")]
        if "Z" in stages:trace_handles.append(Line2D([],[],color="#555555",ls=":",label="Frozen q = 1 cutoff (Z)"))
        fig.legend(handles=trace_handles,loc="upper center",ncol=3);fig.tight_layout(rect=(0,0,1,.91))
        caption="Fixed seed 20260916, exact weak center (x, y) = (68, 64), all 464 source samples; three means × three orders. Shading marks setup UI 65–164; vertical line marks first simulated active sample UI 185. Alone/paired setup is identical. Stage scales are shared across all mean columns. "
        if "A" in stages:caption+="A is identical across references; the gray A curves therefore show it once. "
        caption+="Spread floors and all exact values are in mechanism_traces.tsv. These traces describe paired arithmetic and threshold relationships, not biological causes."
        figures.append(_save(fig,out,stem,caption,["mechanism_traces"]))
    require(len(figures)==8,"Expected eight scientific figures")
    return figures


def _text(tables, figures, audit):
    primary=[r for r in tables["pooled_deadlines"] if r["threshold_id"]=="q1" and r["source_role"]=="weak" and r["match_radius_px"]==2 and r["deadline_ms"]==100]
    lookup={(r["scenario"],r["arm_id"]):r for r in primary}
    lines=["# Reference mean-distance and order: development results", "",
        "All 189 numerical states and 3,780 cutoff/radius rows are complete. " + ("All 189 q = 1 scientific media audits passed aggregate validation." if audit["scientific_audit_complete"] else "Full q = 1 scientific media audits are pending; numerical completion is not media completion."), "",
        "Nine unmasked direct references cross three actual mean distances with orders 3, 9, and 15. The target response, conditioned activity-level input, 3×3 peak selector and 6 px separation remain fixed. Three paired noise seeds cover isolated/crowded sources and three synthetic null conditions. These reused development scenes do not establish biological precision, an optimum, independent validation, a constant application false-alarm probability, or feedback-control performance.", "",
        "## Weak-source results at q = 1", "", "Each entry is matched active frames out of 738 / events recovered within 100 ms out of 3. Inclusive 2 px matching; 246 active frames per seed. Missed events remain in the deadline denominator.", "",
        "|Reference|Alone|Neighbor 8 px|Neighbor 12 px|Neighbor 16 px|", "|---|---:|---:|---:|---:|"]
    geometry={r["arm_id"]:r for r in tables["kernel_summary"]}
    for arm in ARMS:
        cells=[f"{lookup[(s,arm)]['matched_active_frame_count']}/{lookup[(s,arm)]['active_frame_count']} / {lookup[(s,arm)]['recovered_by_deadline']}/3" for s in SCENARIOS[:4]]
        label=f"Mean {geometry[arm]['realized_mean_radius_px']:.2f} px; order {geometry[arm]['reference_n']:g}"
        lines.append("|"+label+"|"+"|".join(cells)+"|")
    lines.extend(["", "The central mean-distance/order-9 reference (mean1_n9) reproduces the 12 frozen crowding states. Every reference is reported; no setting is chosen from these outcomes. Inspect null burden and neighbor recovery alongside any weak-source gains. A late isolated match and prompt sustained coverage answer different questions.", "",
        "## Read the figures", ""])
    for r in figures:
        lines.extend([f"- [{r['id'].replace('_',' ')}]({r['png']}) ([PDF]({r['pdf']})): {r['caption']}"])
    lines.extend(["", "## Metrics and evidence", "",
        "Mean distance is the normalized pixel-weighted radial mean; order controls the shape while mean is matched discretely. The per-pixel peak radius, annular-mass mode, mean and finite support are different quantities. The geometric effective sample count is not a degrees-of-freedom estimate after spatial/temporal conditioning.", "",
        "The q label denotes the nominal setup proposal budget per 194,820 eligible pixels per frame. Here the eligible field is 116 × 116 pixels; q = 1 permits floor(100 × 13,456 / 194,820) = 6 setup proposals over 100 frames, subject to ties. Equal setup budgets do not guarantee equal application burden. Thresholds use strict native-score exceedance; q = 0 is a zero-setup-budget cutoff and is not the no-output endpoint.", "",
        "For each seed and reference, all seven source/null conditions have exactly identical setup floors and ten-cutoff plans. The report independently compares these values and binds the [paired setup check](../paired_setup_calibration_check.json). Thus the paired application outcomes are not produced by different calibrated settings within a seed/reference.", "",
        "Synthetic precision counts correctly assigned frame proposals divided by emitted proposals. Sensitivity counts assigned active-source frames divided by all active-source frames. False proposals include duplicates. Null clips have no neural truth, so sensitivity and event recovery are not applicable; all proposals are false under the declared simulation. False burden divides counts by eligible area-time, not by an invented number of true-negative opportunities. The 6 px results remain separate because their disks overlap at 8 px separation.", "",
        "Deadline recovery uses the first matched active fluorescence sample relative to the generator's first active sample; it is not inferred neuronal onset or acquisition/control latency. All three seeds are shown; no confidence interval is fabricated and active frames are not independent experimental replicates.", "",
        "Exact tables: [all curves](curves.tsv), [event curves](event_curves.tsv), [pooled curves](pooled_curves.tsv), [all deadlines](pooled_deadlines.tsv), [native setup cutoffs](calibration.tsv), [kernel geometry](kernel_summary.tsv), [unit-source reference overlaps](kernel_source_overlap.tsv), [all paired stage samples](mechanism_traces.tsv), and [weak/neighbor/null summary](findings_summary.json). Overlap values describe unit spatial footprints after conditioning; they are not observed variances or neural identities.", "",
        "Sources: [frozen protocol](../protocol.json), [numerical completion](../evaluation_complete.json), [anchor replication](../baseline_replication.json), [consumed-source bindings](sources_manifest.json), and [report manifest](manifest.json). Full q = 1 media are under ../audits; their completion evidence is separate from this report's visual QA. Model audit sites are review locations, not inferred neurons.", ""])
    if audit["scientific_audit_complete"]:lines.append("[Complete scientific audit certificate](../audit_complete.json).")
    return "\n".join(lines)


def _qa(out, manifest, sources):
    path=out/"visual_qa.json"
    if not path.exists():return dict(visual_qa_complete=False,visual_qa_binding=None)
    qa=sources.read(path)
    require(qa.get("status")=="PASS","Present report visual QA is not PASS")
    expected={(out/r["png"]).resolve() for r in manifest["figures"]}
    actual=[Path(r["path"]).resolve() for r in qa["figures"]]
    require(len(actual)==len(set(actual))==8 and set(actual)==expected,"Visual QA must cover exactly all eight current PNGs")
    for b in qa["figures"]:sources.check(b)
    return dict(visual_qa_complete=True,visual_qa_binding=sources.bind(path))


def report(root=ROOT):
    root=_mutable(root);out=root/"report"
    require(not out.exists() or not any(out.iterdir()),"Preserve existing report; generation requires an empty report directory")
    data=_load(root);tables=make_tables(data);out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():_write(out/f"{name}.json",rows);_tsv(out/f"{name}.tsv",rows)
    _write(out/"findings_summary.json",dict(
        weak_q1=[r for r in tables["q1_source_summary"] if r["source_role"]=="weak" and r["match_radius_px"]==2],
        neighbor_q1=[r for r in tables["q1_source_summary"] if r["source_role"]=="neighbor" and r["match_radius_px"]==2],
        null_q1=[r for r in tables["pooled_curves"] if r["scenario"] in SCENARIOS[4:] and r["threshold_id"]=="q1" and r["match_radius_px"]==2],
        kernel_source_overlap=tables["kernel_source_overlap"],
        interpretation="Complete predeclared cohorts and geometry-only overlaps; no setting selected. Overlaps describe unit spatial footprints, not observed variance or biological source identity."))
    figures=_plot(out,tables)
    numeric_sources=data["sources"].finish()
    audit_sources=Sources();audit=_audit_status(root,data["protocol"],audit_sources)
    _write(out/"sources_manifest.json",dict(status="VERIFIED_CONSUMED_SOURCES",inputs=numeric_sources,
        inherited_large_sources=data["inherited"],reporter=_binding(Path(__file__)),
        scope="All consumed numerical/geometry/truth files and the fixed trace arrays checked by SHA256. Complete positive-prefix and other stage bindings retained from cell seals; their full validation is owned by the aggregate scientific audit."))
    (out/"REPORT.md").write_text(_text(tables,figures,audit))
    manifest=dict(schema_version=1,status="GENERATED",numerical_complete=True,scientific_audit_complete=audit["scientific_audit_complete"],
        report_stage=audit["status"],audit_verification=audit,visual_qa_complete=False,visual_qa_binding=None,
        figure_count=len(figures),figures=figures,table_row_counts={k:len(v) for k,v in tables.items()},
        inputs=numeric_sources,reporter=_binding(Path(__file__)),
        artifacts=[dict(_binding(p),path=str(p.relative_to(out))) for p in sorted(out.iterdir()) if p.is_file()])
    _write(out/"manifest.json",manifest)
    print(dict(status="GENERATED",figures=len(figures),scientific_audit_complete=manifest["scientific_audit_complete"]),flush=True)
    return manifest


def update(root=ROOT):
    root=_mutable(root);out=root/"report";manifest=_read(out/"manifest.json");sources=Sources()
    sources.check(manifest["reporter"],Path(__file__))
    for b in manifest["inputs"]:sources.check(b)
    for b in manifest["artifacts"]:sources.check(dict(b,path=str(out/b["path"])))
    protocol=sources.read(root/"protocol.json");audit=_audit_status(root,protocol,sources)
    tables={name:_read(out/f"{name}.json") for name in manifest["table_row_counts"]}
    qa=_qa(out,manifest,sources);sources.finish()
    (out/"REPORT.md").write_text(_text(tables,manifest["figures"],audit))
    manifest.update(scientific_audit_complete=audit["scientific_audit_complete"],report_stage=audit["status"],audit_verification=audit,**qa)
    manifest["artifacts"]=[dict(_binding(out/b["path"]),path=b["path"]) for b in manifest["artifacts"]]
    _write(out/"manifest.json",manifest)
    print(dict(status="STATUS_UPDATED",scientific_audit_complete=manifest["scientific_audit_complete"],visual_qa_complete=manifest["visual_qa_complete"]),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("report","update"));parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args();globals()[args.command](args.root)


if __name__=="__main__":main()
