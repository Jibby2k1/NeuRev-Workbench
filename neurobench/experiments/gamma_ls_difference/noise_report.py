"""Source-bound null-noise factorial report; no prediction or cutoff selection.

All 81 candidate seals and evaluations precede diagnostic reads. Per-frame
statistics use the fixed eligible interior; temporal pairs are causal. Report
updates preserve numerical tables/figures and never mutate completed roots.
"""
from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path

from . import reference_report as common

ROOT = Path(__file__).resolve().parents[3] / "Outputs/GammaLSNoise/noise_20260915_r1"
ARMS = ("mean2of3_n3", "mean1_n9", "mean4of3_n9")
SEEDS = (20260916, 20260917, 20260918)
FACTORS = tuple(itertools.product((0, 1), repeat=3))
EFFECTS = ("V", "S", "T", "VS", "VT", "ST", "VST")
EPOCHS = {"setup": (65, 164), "early": (165, 264), "late": (265, 464)}
LABELS = ("Neither", "T", "S", "S + T", "V", "V + T", "V + S", "V + S + T")
ARM_LABELS = {"mean2of3_n3": "Compact reference\nmean 6.23 px; order 3",
              "mean1_n9": "Original reference\nmean 9.35 px; order 9",
              "mean4of3_n9": "Wider reference\nmean 12.46 px; order 9"}
RATE = "false_proposals_per_10000_um2_s"
STATS = ("Raw_residual_variance", "Input_residual_variance", "Spread_q50",
         "Z_q999", "Z_exceedance_fraction", "floor_active_fraction")
require, Sources = common.require, common.Sources
_read, _write, _binding, _tsv = common._read, common._write, common._binding, common._tsv


def _matrix(protocol):
    cases = protocol["cases"]
    require(len(cases) == len({c["case_id"] for c in cases}) == 27, "Expected 27 unique cases")
    require({r["arm_id"] for r in protocol["references"]} == set(ARMS) and
            len(protocol["references"]) == 3, "Expected exact three references")
    require(protocol["expected_cells"] == 81 and protocol["expected_curve_rows"] == 1620,
            "Unexpected noise matrix size")
    for seed in SEEDS:
        group = [c for c in cases if c["seed"] == seed]
        factorial = [c for c in group if c["kind"] == "factorial"]
        legacy = [c for c in group if c["kind"] == "legacy"]
        require(len(group) == 9 and len(factorial) == 8 and len(legacy) == 1 and
                {tuple(c[k] for k in "VST") for c in factorial} == set(FACTORS), "Incomplete seed factorial")
        require(all(legacy[0].get(k) is None for k in "VST"), "Legacy bridge cannot be a factorial cell")
    for c in cases:
        require(Path(c["case_id"]).name == c["case_id"] and c["case_id"] not in (".", ".."), "Unsafe case identifier")
    return [(c["case_id"], arm) for c in cases for arm in ARMS]


def _seal_gate(root, protocol, sources):
    states = _matrix(protocol)
    require(all((root / "cells" / c / a / "sealed.json").is_file() for c, a in states),
            "All 81 candidate seals are required before diagnostic/outcome reads")
    seals = {}
    for case, arm in states:
        s = sources.read(root / "cells" / case / arm / "sealed.json")
        require(s.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", "Incomplete candidate seal")
        seals[case, arm] = s
    for case in {c for c, _ in states}:
        require(len({tuple(seals[case, a]["stages"][k]["sha256"] for k in ("Raw", "Input", "A")) for a in ARMS}) == 1,
                "Input or target differs across references")
    require(sources.read(root / "computation_complete.json") == dict(status="PASS", cells=81), "Computation incomplete")
    require(sources.read(root / "evaluation_complete.json") == dict(status="PASS", cells=81, curve_rows=1620),
            "All 81 evaluations are required")
    return seals


def _load(root):
    sources = Sources(); p = sources.read(root / "protocol.json")
    preflight = sources.read(root / "preflight.json")
    require(preflight["status"] == "PASS" and preflight["protocol_sha256"] == sources.bind(root / "protocol.json")["sha256"],
            "Protocol/preflight differ")
    for b in p["code_bindings"] + p["kernel_bindings"]: sources.check(b)
    require(p["frame_rate_hz"] == 50 and p["pixel_size_um"] == .5, "Physical units differ")
    seals = _seal_gate(root, p, sources)
    # No diagnostic array, activity file, or outcome file has been read above.
    for key in ("baseline_completion", "baseline_protocol"): sources.check(p[key])
    if "paper_protocol" in p: sources.check(p["paper_protocol"])
    prep = sources.read(root / "datasets_complete.json")
    require(prep["status"] == "PASS" and prep["datasets"] == 27 and prep["paired_setup_byte_equal"] and
            prep["variance_pairs_early_byte_equal"], "Paired source preparation is incomplete")
    cases = {c["case_id"]: c for c in p["cases"]}
    curves, calibration, operations, inherited = [], [], {}, []
    pairing = {}
    for case in cases:
        files = {Path(b["path"]).name: b for b in seals[case, ARMS[0]]["dataset_bindings"]}
        for name in ("metadata.json", "experts.json", "active.json"):
            sources.check(files[name])
        meta = sources.read(files["metadata.json"]["path"])
        require(meta["truth_mode"] == "fully_synthetic" and meta["source_frames_ui"] == list(range(1, 465)) and
                meta["application_source_frames_ui"] == list(range(165, 465)), "Null dataset scope differs")
        require(meta["shape_tyx"] == [464, 226, 226] and meta["evaluation_shape_yx"] == [128, 128] and
                meta["original_source_offset_xy"] == [49, 49] and
                all(meta["background"][k] == v for k, v in
                    {"offset":100., "gradient_x_per_px":.03, "gradient_y_per_px":.02}.items()),
                "Known-background residual geometry differs")
        require(not sources.read(files["experts.json"]["path"]) and not sources.read(files["active.json"]["path"]),
                "Noise study requires genuinely empty synthetic truth")
    for (case, arm), seal in seals.items():
        folder = root / "cells" / case / arm
        evaluated = sources.read(folder / "evaluated.json")
        require(evaluated["status"] == "PASS", "Cell evaluation incomplete")
        sources.check(evaluated["seal"], folder / "sealed.json")
        for b in evaluated["outputs"]: sources.check(b)
        require((folder / "curves.json").resolve() in {Path(b["path"]).resolve() for b in evaluated["outputs"]}, "Unbound curves")
        for key in ("calibration", "threshold_plan"): sources.check(seal[key], folder / (key + ".json"))
        op = sources.read(folder / "calibration.json"); plan = sources.read(folder / "threshold_plan.json")
        require(op["threshold_id"] == "q1" and op["threshold_frozen_from_calibration_only"] and op["window"] == 3 and
                op["eligible_area_px"] == 13456 and op["setup_proposal_budget"] == 6, "Operating protocol differs")
        require(len(plan) == 10 and {r["threshold_id"] for r in plan} == set(common.THRESHOLDS), "Cutoff inventory differs")
        pair = (cases[case]["seed"], arm); value = (plan, op["scale_floor"])
        require(pair not in pairing or pairing[pair] == value, "Paired conditions have unequal setup cutoffs/floors")
        pairing[pair] = value; operations[case, arm] = op
        current = sources.read(folder / "curves.json")
        require(not current["event_rows"], "Null study contains event outcomes")
        cr = current["curve_rows"]; keys = [common._key(r) for r in cr]
        require(len(keys) == len(set(keys)) == 20 and set(keys) == {(case, arm, t, rad) for t in common.THRESHOLDS for rad in (2., 6.)},
                "Cell curve inventory differs")
        settings = {r["threshold_id"]: r for r in plan}
        for r in cr:
            require(r["threshold_z"] == settings[r["threshold_id"]]["threshold"], "Native threshold differs")
            require(r["event_count"] == r["active_region_frame_count"] == r["true_positive_count"] == 0 and
                    r["false_positive_count"] == r["proposal_count"] and r["framewise_sensitivity"] is None and
                    r["event_window_coverage"] is None, "Null outcome partition differs")
            require(r["eligible_area_px"] == 13456 and r["exposure_seconds"] == 6, "Burden denominator differs")
        curves.extend(cr)
        calibration.extend(dict(r, case_id=case, arm_id=arm, seed=cases[case]["seed"], case_kind=cases[case]["kind"],
                                scale_floor=op["scale_floor"]) for r in plan)
        inherited.append(dict(case_id=case, arm_id=arm, seal=sources.bind(folder / "sealed.json"),
                              stages=seal["stages"], prefix=seal["prefix"]))
    aggregate = sources.read(root / "all_curves.json"); sources.bind(root / "all_curves.tsv")
    require(len(aggregate) == len(curves) == 1620 and {common._key(r): r for r in aggregate} ==
            {common._key(r): r for r in curves}, "Aggregate differs from cell curves")
    replication = sources.read(root / "baseline_replication.json")
    expected_reuse = {(c["case_id"], a) for c in cases.values() if c["reused_dataset"] for a in ARMS}
    require(len(expected_reuse) == 18 and replication["status"] == "PASS" and len(replication["cells"]) == 18 and
            {(r["case_id"], r["arm_id"]) for r in replication["cells"]} == expected_reuse and
            all(all(r[k] for k in ("all_thresholds_equal", "metrics_equal", "stages_equal", "candidates_equal")) for r in replication["cells"]),
            "Legacy/stationary replication incomplete")
    receipt = sources.read(root / "paired_setup_calibration_check.json")
    require(receipt["status"] == "PASS" and len(receipt["groups"]) == 9 and
            {(r["seed"], r["arm_id"]) for r in receipt["groups"]} == set(pairing) and
            all(r["cases"] == 9 and r["thresholds_equal"] and r["floors_equal"] for r in receipt["groups"]),
            "Independent paired calibration receipt differs")
    return dict(protocol=p, sources=sources, seals=seals, cases=cases, operations=operations,
                curves=curves, calibration=calibration, inherited=inherited, replication=replication)


def epoch(frame):
    return next((name for name, (lo, hi) in EPOCHS.items() if lo <= frame <= hi), "warmup")


def _covariance(a, b):
    """Population covariance/correlation after separately centering each field."""
    import numpy as np
    a = a - np.mean(a, dtype=np.float64); b = b - np.mean(b, dtype=np.float64)
    cov = float(np.mean(a * b, dtype=np.float64))
    product = float(np.mean(a*a, dtype=np.float64) * np.mean(b*b, dtype=np.float64))
    return cov, cov / math.sqrt(product) if product > 0 else None


def residual_diagnostics(array, *, border=6, crop_origin=49, full_center=113):
    """Known-background residual statistics, bounded to one eligible frame.

    Spatial covariance averages horizontal and vertical lag-one pairs. Temporal
    pairs end at the reported frame, including a pair spanning an epoch boundary.
    Variance uses ddof=0 after removal of each residual field's spatial mean.
    """
    import numpy as np
    require(array.ndim == 3 and min(array.shape[1:]) > 2*border+1, "Invalid residual array geometry")
    y, x = np.mgrid[border:array.shape[1]-border, border:array.shape[2]-border]
    background = 100 + .03*(x+crop_origin-full_center) + .02*(y+crop_origin-full_center)
    output = []; previous = None
    for frame in array:
        u = np.asarray(frame[border:-border, border:-border], dtype=np.float64) - background
        require(bool(np.isfinite(u).all()), "Nonfinite residual stage")
        h = _covariance(u[:, :-1], u[:, 1:]); v = _covariance(u[:-1], u[1:])
        temporal = _covariance(previous, u) if previous is not None else (None, None)
        output.append(dict(residual_mean=float(np.mean(u)), residual_variance=float(np.var(u)),
                           spatial_lag1_covariance=(h[0]+v[0])/2,
                           spatial_lag1_correlation=(h[1]+v[1])/2 if h[1] is not None and v[1] is not None else None,
                           temporal_lag1_covariance=temporal[0], temporal_lag1_correlation=temporal[1]))
        previous = u
    return output


def score_diagnostics(array, *, stage, floor, threshold, border=6):
    import numpy as np
    output = []
    for frame in array:
        v = np.asarray(frame[border:-border, border:-border], dtype=np.float64).reshape(-1)
        require(v.size and bool(np.isfinite(v).all()), "Nonfinite/empty score stage")
        q = np.quantile(v, [.5, .95, .99, .999])
        row = dict(q50=float(q[0]), q95=float(q[1]), q99=float(q[2]), q999=float(q[3]), maximum=float(v.max()))
        if stage == "Spread":
            require(bool((v >= 0).all()), "Negative reference spread")
            row["floor_active_fraction"] = float(np.mean(v < floor))
        elif stage == "Z": row["exceedance_fraction"] = float(np.mean(v > threshold))
        else: raise ValueError(stage)
        output.append(row)
    return output


def proposal_frames(rows, *, first, last, threshold, already_filtered=False):
    """Count existing proposals only; never rerun NMS or change their ordering."""
    counts = {i: 0 for i in range(first, last+1)}; ids = set()
    for r in rows:
        frame = int(r["source_frame_ui"]); pid = r["proposal_id"]
        require(frame in counts and pid not in ids and math.isfinite(r["score"]), "Invalid/duplicate proposal row")
        require(6 <= r["x_px"] < 122 and 6 <= r["y_px"] < 122, "Proposal outside eligible interior")
        ids.add(pid)
        if already_filtered: require(r["score"] > threshold, "Frozen candidate does not exceed cutoff")
        if r["score"] > threshold: counts[frame] += 1
    return counts


def _per_frame(data):
    import numpy as np
    sources = data["sources"]; cache = {}; rows = []
    for index, ((case, arm), seal) in enumerate(data["seals"].items(), 1):
        c = data["cases"][case]; op = data["operations"][case, arm]; values = {}
        for stage in ("Raw", "Input", "Spread", "Z"):
            b = seal["stages"][stage]; key = (b["path"], b["sha256"], stage,
                                                op["scale_floor"] if stage == "Spread" else op["threshold"] if stage == "Z" else None)
            if key not in cache:
                sources.check(b); a = np.load(b["path"], mmap_mode="r")
                require(a.shape == (464, 128, 128), "Diagnostic stage geometry differs")
                cache[key] = residual_diagnostics(a) if stage in ("Raw", "Input") else score_diagnostics(
                    a, stage=stage, floor=op["scale_floor"], threshold=op["threshold"])
            values[stage] = cache[key]
        for name in ("audit_candidates", "setup_prefix"): sources.check(seal[name])
        application = proposal_frames(sources.read(seal["audit_candidates"]["path"]), first=165, last=464,
                                      threshold=op["threshold"], already_filtered=True)
        setup = proposal_frames(sources.read(seal["setup_prefix"]["path"]), first=65, last=164, threshold=op["threshold"])
        expected = next(r for r in data["curves"] if common._key(r) == (case, arm, "q1", 2.))
        require(sum(application.values()) == expected["proposal_count"] and sum(setup.values()) == op["setup_proposal_count"],
                "Saved q1 candidates/setup prefix disagree with numerical counts")
        for frame in range(65, 465):
            row = dict(case_id=case, arm_id=arm, seed=c["seed"], case_kind=c["kind"],
                       **{k: c.get(k) for k in "VST"}, source_frame_ui=frame, epoch=epoch(frame),
                       application_relative_seconds=(frame-165)/50, threshold=op["threshold"], scale_floor=op["scale_floor"],
                       proposal_count=(setup if frame < 165 else application)[frame])
            for stage in values:
                row.update({stage+"_"+k: v for k, v in values[stage][frame-1].items()})
            row["floor_active_fraction"] = row.pop("Spread_floor_active_fraction")
            rows.append(row)
        if index % 3 == 0:
            print(dict(status="DIAGNOSTIC_CELLS_COMPLETE", cells=index, total=len(data["seals"])), flush=True)
    return rows


def epoch_tables(rows):
    groups = defaultdict(list)
    for r in rows: groups[r["case_id"], r["arm_id"], r["epoch"]].append(r)
    output = []
    for (case, arm, ep), group in sorted(groups.items()):
        lo, hi = EPOCHS[ep]
        require(sorted(r["source_frame_ui"] for r in group) == list(range(lo, hi+1)), "Incomplete/duplicate epoch samples")
        row = {k: group[0][k] for k in ("case_id", "arm_id", "seed", "case_kind", "V", "S", "T", "threshold", "scale_floor")}
        row.update(epoch=ep, frame_count=len(group), eligible_area_px=13456, exposure_seconds=len(group)/50,
                   proposal_count=sum(r["proposal_count"] for r in group))
        row[RATE] = 10000*row["proposal_count"] / (13456*.25*row["exposure_seconds"])
        for k in group[0]:
            if k.startswith(("Raw_", "Input_", "Spread_", "Z_")) or k == "floor_active_fraction":
                v = [r[k] for r in group if r[k] is not None]
                row[k] = sum(v)/len(v) if v else None
        output.append(row)
    return output


def factorial_effects(rows, metrics=(RATE,)+STATS):
    """Paired difference scales: main effects/2-way/3-way are not regression betas."""
    groups = defaultdict(dict)
    for r in rows:
        if r["case_kind"] != "factorial": continue
        key = (r["seed"], r["arm_id"], r["epoch"]); factors = tuple(r[k] for k in "VST")
        require(factors not in groups[key], "Duplicate factorial cell")
        groups[key][factors] = r
    result = []
    for (seed, arm, ep), group in sorted(groups.items()):
        require(set(group) == set(FACTORS), "Incomplete factorial contrast")
        for effect in EFFECTS:
            ix = ["VST".index(k) for k in effect]
            for metric in metrics:
                require(all(r[metric] is not None and math.isfinite(r[metric]) for r in group.values()), "Undefined factorial statistic")
                value = sum(math.prod(2*f[i]-1 for i in ix)*r[metric] for f, r in group.items()) / 2**(3-len(ix))
                result.append(dict(seed=seed, arm_id=arm, epoch=ep, effect=effect, metric=metric, difference=value,
                                   paired_cell_count=8, contrast_definition="Mean paired difference" if len(ix)==1 else
                                   "Mean difference of differences" if len(ix)==2 else "Third difference"))
    return result


def variance_theory_rows(rows, theory):
    """Descriptive measured/theoretical comparison; no data-fitted normalization."""
    gaussian = theory["gaussian"]; ema = theory["ema"]
    require(gaussian["sigma_px"] == 1 and gaussian["truncate"] == 4 and ema["alpha"] == .4 and
            ema["raw_ar_rho"] == .8, "Theory does not describe the fixed conditioning")
    output = []
    for r in rows:
        row = {k:r[k] for k in ("case_id","arm_id","seed","case_kind","V","S","T","epoch")}
        active = r["epoch"] != "setup"
        if r["case_kind"] == "factorial":
            sigma2 = 25. if r["V"] and r["epoch"] == "late" else 4.
            spatial = gaussian["spatial_on_off_variance_ratio"] if active and r["S"] else 1.
            temporal = ema["temporal_on_off_variance_ratio"] if active and r["T"] else 1.
            predicted_input = sigma2 * gaussian["white_noise_energy"] * ema["white_noise_variance_factor"] * spatial * temporal
        elif not active:
            sigma2 = 4.; predicted_input = sigma2*gaussian["white_noise_energy"]*ema["white_noise_variance_factor"]
        else:
            lo,hi=EPOCHS[r["epoch"]]
            values=[]
            for frame in range(lo,hi+1):
                index=frame-165; sigma=2. if index<100 else 5.
                values.append(sigma*sigma+9*(1-.8**(2*(index+1)))+1.2*sigma)
            sigma2=sum(values)/len(values);predicted_input=None
        row.update(raw_pointwise_variance_theory=sigma2, conditioned_stationary_pointwise_variance_theory=predicted_input,
                   measured_Raw_residual_variance=r["Raw_residual_variance"],measured_Input_residual_variance=r["Input_residual_variance"],
                   Raw_measured_over_pointwise_theory=r["Raw_residual_variance"]/sigma2,
                   Input_measured_over_stationary_theory=r["Input_residual_variance"]/predicted_input if predicted_input else None)
        output.append(row)
    return output


def verify_early_variance_pairing(rows):
    groups = defaultdict(dict)
    for r in rows:
        if r["case_kind"] == "factorial" and r["source_frame_ui"] <= 264:
            key = (r["seed"], r["arm_id"], r["S"], r["T"], r["source_frame_ui"])
            require(r["V"] not in groups[key], "Duplicate early V pair")
            groups[key][r["V"]] = r
    for pair in groups.values():
        require(set(pair) == {0, 1}, "Incomplete early V pair")
        for k in pair[0]:
            if k.startswith(("Raw_", "Input_", "Spread_", "Z_")) or k in ("proposal_count", "threshold", "scale_floor", "floor_active_fraction"):
                require(pair[0][k] == pair[1][k], "V differs before its application variance step")
    return dict(status="PASS", compared_frame_pairs=len(groups), scope="Exact consumed diagnostics and q1 counts through UI264; no future samples used")


def time_bins(rows, width=10):
    groups = defaultdict(list)
    for r in rows: groups[r["case_id"], r["arm_id"], (r["source_frame_ui"]-65)//width].append(r)
    output = []
    for (_, _, index), group in sorted(groups.items()):
        first = 65+index*width
        require(sorted(r["source_frame_ui"] for r in group) == list(range(first, first+width)), "Incomplete time bin")
        row = {k: group[0][k] for k in ("case_id", "arm_id", "seed", "case_kind", "V", "S", "T")}
        count = sum(r["proposal_count"] for r in group)
        row.update(source_start_ui=first, source_stop_ui=first+width-1, time_mid_seconds=(first-165+(width-1)/2)/50,
                   frame_count=width, proposal_count=count, **{RATE:10000*count/(13456*.25*width/50)})
        output.append(row)
    return output


def _audit_status(root, protocol, sources):
    path = root / "audit_complete.json"
    if not path.exists(): return dict(status="NUMERICAL_COMPLETE_AUDITS_PENDING", scientific_audit_complete=False)
    audit = sources.read(path)
    require(audit.get("status") == "PASS" and audit.get("cells") == 81 and audit.get("new_cells") == 63 and
            audit.get("reused_cells") == 18, "Audit aggregate is not complete")
    keys = [(r["case_id"], r["arm_id"]) for r in audit["audits"]]
    require(len(keys) == len(set(keys)) == 81 and set(keys) == set(_matrix(protocol)), "Audit matrix differs")
    names = {Path(b["path"]).name for b in audit["validation_code"]}
    require({"noise_validate.py", "noise_media.py", "followup_validate.py", "followup_audit.py",
             "two_stencil_audit.py", "two_stencil_evaluation.py"} <= names, "Audit validation dependencies missing")
    for b in audit["validation_code"]: sources.check(b)
    for key, name in (("protocol", "protocol.json"), ("evaluation", "evaluation_complete.json"), ("all_curves", "all_curves.json"),
                      ("display_contract", "display_contract.json"), ("forecast", "audit_forecast.json"),
                      ("baseline_replication", "baseline_replication.json")):
        sources.check(audit[key], root / name)
    bindings = []
    for r in audit["audits"]:
        require(set(r["metadata_bindings"]) == common.AUDIT_METADATA, "Audit metadata inventory differs")
        for name, b in r["metadata_bindings"].items():
            require(Path(b["path"]).name == name, "Audit metadata filename differs"); bindings.append(sources.check(b))
        summary = sources.read(r["metadata_bindings"]["summary.json"]["path"])
        status = sources.read(r["metadata_bindings"]["status.json"]["path"])
        validation = sources.read(r["metadata_bindings"]["validation.json"]["path"])
        require(summary == r["summary"] and summary["scientific_audit_complete"] is True and status["status"] == "complete" and
                status["scientific_audit_complete"] is True and validation["status"] == "passed" and not validation["failures"],
                "Actual audit metadata is not complete")
    return dict(status="AUDITS_VERIFIED_VISUAL_QA_SEPARATE", scientific_audit_complete=True,
                aggregate=sources.bind(path), metadata_bindings=bindings, validation_code=audit["validation_code"])


def _limits(values, signed=False):
    require(values and all(math.isfinite(v) for v in values), "Invalid plot values")
    maximum = max(1., 1.4*max(abs(v) for v in values))
    require(signed or min(values) >= 0, "Negative value on nonnegative axis")
    return (-maximum if signed else 0., maximum)


def _plot(out, tables):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":11})
    figures = []; epochs = tables["epochs"]; factorial = [r for r in epochs if r["case_kind"] == "factorial"]
    colors = ("#2471A3", "#C57E12", "#7E477F"); phase_colors = dict(zip(EPOCHS, colors))
    factor_caption = "V: raw standard deviation increases from 2 to 5 at application 2 s; S: spatial correlation; T: temporal correlation. Dots are paired seeds, larger circles their descriptive mean. No confidence intervals or independent-frame interpretation."
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for j, arm in enumerate(ARMS):
        for i, ep in enumerate(("early", "late")):
            ax = axes[i, j]
            for x, f in enumerate(FACTORS):
                g = [r for r in factorial if r["arm_id"] == arm and r["epoch"] == ep and tuple(r[k] for k in "VST") == f]
                y = [r[RATE] for r in g]; require(len(y) == 3, "Missing plot seed")
                ax.scatter(x+np.linspace(-.12,.12,3), y, color=colors[j], alpha=.6, marker="x", s=22)
                ax.plot(x, np.mean(y), "o", color=colors[j], ms=5)
            ax.set_title(ARM_LABELS[arm] if i == 0 else "")
            ax.set_xticks(range(8), LABELS, rotation=30, ha="right"); ax.set_yscale("symlog", linthresh=1)
            if j == 0: ax.set_ylabel(("Early: first 2 s" if i == 0 else "Late: final 4 s")+"\nFalse proposals / 10,000 µm² / s")
    common._style(axes); axes[0,0].set_ylim(_limits([r[RATE] for r in factorial if r["epoch"] != "setup"]))
    fig.tight_layout(); figures.append(common._save(fig,out,"factorial_null_burden",factor_caption+" Common symlog axis includes zero and is linear below 1.",["epochs"]))

    effects = [r for r in tables["factorial_effects"] if r["metric"] == RATE and r["epoch"] != "setup"]
    fig, axes = plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for j,arm in enumerate(ARMS):
        for i,ep in enumerate(("early","late")):
            ax=axes[i,j]
            for x,effect in enumerate(EFFECTS):
                y=[r["difference"] for r in effects if r["arm_id"]==arm and r["epoch"]==ep and r["effect"]==effect]
                require(len(y)==3,"Missing contrast seed")
                ax.scatter(x+np.linspace(-.12,.12,3),y,color=colors[j],marker="x",s=22,alpha=.6);ax.plot(x,np.mean(y),"o",color=colors[j],ms=5)
            ax.axhline(0,color="black",lw=.7);ax.set_xticks(range(7),EFFECTS);ax.set_yscale("symlog",linthresh=1)
            if i==0:ax.set_title(ARM_LABELS[arm])
            if j==0:ax.set_ylabel(("Early" if i==0 else "Late")+" paired effect\nFalse proposals / 10,000 µm² / s")
    common._style(axes);axes[0,0].set_ylim(_limits([r["difference"] for r in effects],signed=True));fig.tight_layout()
    figures.append(common._save(fig,out,"paired_factorial_effects",factor_caption+" Main effects average four within-seed differences; two-way interactions average two differences of differences; the three-way term is a third difference. These are response contrasts, not proof of separate physical mechanisms.",["factorial_effects"]))

    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True); bins=[r for r in tables["time_bins"] if r["case_kind"]=="factorial"]
    st_colors=("#333333","#2471A3","#C57E12","#7E477F")
    for j,arm in enumerate(ARMS):
        for v in (0,1):
            ax=axes[v,j]
            for color,(s,t) in zip(st_colors,itertools.product((0,1),repeat=2)):
                group=[r for r in bins if r["arm_id"]==arm and (r["V"],r["S"],r["T"])==(v,s,t)]
                seed_rows=[sorted([r for r in group if r["seed"]==seed],key=lambda r:r["time_mid_seconds"]) for seed in SEEDS]
                x=[r["time_mid_seconds"] for r in seed_rows[0]]; y=np.array([[r[RATE] for r in g] for g in seed_rows])
                require(y.shape==(3,40),"Transient bin inventory differs")
                for row in y:ax.plot(x,row,color=color,alpha=.15,lw=.6)
                ax.plot(x,y.mean(axis=0),color=color,lw=1.3)
            ax.axvspan(-2,0,color="#dddddd",alpha=.35);ax.axvline(0,color="black",lw=.7);ax.axvline(2,color="black",lw=.7,ls=":")
            ax.set_yscale("symlog",linthresh=1)
            if v==0:ax.set_title(ARM_LABELS[arm])
            if j==0:ax.set_ylabel(("Variance fixed" if v==0 else "Variance step at 2 s")+"\nFalse proposals / 10,000 µm² / s")
            if v==1:ax.set_xlabel("Time from application start (s)")
    handles=[Line2D([0],[0],color=c,label=l) for c,l in zip(st_colors,("Neither correlation","Temporal only","Spatial only","Both correlations"))]
    fig.legend(handles=handles,loc="upper center",ncol=4);common._style(axes);axes[0,0].set_ylim(_limits([r[RATE] for r in bins]));fig.tight_layout(rect=(0,0,1,.94))
    figures.append(common._save(fig,out,"null_burden_transients","Fixed, nonoverlapping 10-frame (200 ms) bins; fine lines are seeds, bold lines their mean. Setup is shaded. Spatial/temporal factors start at 0 s; only V=1 changes raw variance at 2 s. Rates retain every proposal; binning is a display operation only.",["time_bins"]))

    statistic_sets=[("noise_and_spread",(("Raw_residual_variance","Raw residual variance\n(native units²)"),("Input_residual_variance","Conditioned residual variance\n(native units²)"),("Spread_q50","Mean frame median spread\n(native units)"))),
                    ("score_and_floor",(("Z_q999","Mean frame 99.9th score percentile"),("Z_exceedance_fraction","Fraction of pixels above cutoff"),("floor_active_fraction","Fraction of pixels below spread floor")))]
    for stem,stats in statistic_sets:
        fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,sharey="col")
        for i,arm in enumerate(ARMS):
            for j,(key,label) in enumerate(stats):
                ax=axes[i,j]
                for phase,color in phase_colors.items():
                    for x,f in enumerate(FACTORS):
                        g=[r for r in factorial if r["arm_id"]==arm and r["epoch"]==phase and tuple(r[k] for k in "VST")==f]
                        y=[r[key] for r in g];require(len(y)==3,"Missing diagnostic seed")
                        offset=(list(EPOCHS).index(phase)-1)*.2
                        ax.scatter(x+offset+np.linspace(-.045,.045,3),y,color=color,marker="x",s=12,alpha=.45)
                        ax.plot(x+offset,np.mean(y),"o",color=color,ms=4)
                if i==0:ax.set_title(label)
                if j==0:ax.set_ylabel(ARM_LABELS[arm])
                ax.set_xticks(range(8),LABELS,rotation=35,ha="right")
                if "fraction" not in key:ax.set_yscale("symlog",linthresh=1)
        for j,(key,_) in enumerate(stats):
            values=[r[key] for r in factorial]
            if "fraction" in key:
                require(all(0 <= value <= 1 for value in values), "Invalid probability fraction")
                axes[0,j].set_ylim(0., min(1.05, max(1e-6, 1.4*max(values))))
            else: axes[0,j].set_ylim(_limits(values,signed=key=="Z_q999" and min(values)<0))
        fig.legend(handles=[Line2D([0],[0],marker="o",ls="",color=c,label=l.title()) for l,c in phase_colors.items()],loc="upper center",ncol=3)
        common._style(axes);fig.tight_layout(rect=(0,0,1,.95))
        figures.append(common._save(fig,out,stem,factor_caption+" Known fixed linear background is removed before Raw/Input residual variance (per-frame spatial centering, ddof=0). Each epoch statistic is the mean of per-frame statistics; it is not a pooled pixel quantile. Spread is reference spatial variability, including residual background gradient. Cutoff and floor are frozen from setup.",["epochs","per_frame"]))
    return figures


def _text(tables, figures, audit):
    lines=["# Noise covariance control", "", "Numerical evaluation is complete for all 81 null-only cells and 1,620 cutoff/matching-radius rows.", "",
           "Scientific media audit: "+("complete; current metadata verified." if audit["scientific_audit_complete"] else "pending; numerical completion does not promote media."), "",
           "This is a source-free synthetic development study. Every emitted proposal is false under its generator. There is no source-event denominator, so sensitivity and recovery are not applicable; no precision or biological-performance claim is made.", "",
           "V increases raw noise standard deviation from 2 to 5 at UI265. S sums a 3×3 window of innovations and divides by 3, preserving unit variance before amplitude scaling. T adds causal AR(0.8) temporal correlation with innovation multiplier 0.6 and an independent initial state. S/T start at UI165. The first 164 warmup/setup frames are stationary and identical across conditions within a seed. All references use the same fixed 1 px Gaussian smoothing, causal EMA with coefficient 0.4, and target response.", "",
           "All eight V/S/T settings are crossed within three seeds and three fixed references. Nine exact older compound-noise bridge states are reported separately and excluded from factorial effects; their shared innovations change raw variance as well as covariance. The stationary and compound bridge states have an exact 18-cell replication check.", "",
           "q=1 permits at most six setup proposals over UI65–164 in the 116×116-pixel eligible field. Native score cutoffs and spread floors are separately calibrated by reference but exactly shared across conditions within each seed/reference. Equal setup allowance is not constant application false-alarm probability.", "",
           "Primary burden = emitted proposals / eligible area-time, scaled to 10,000 µm²·s (0.5 µm/pixel, 50 Hz). Setup and early application each contain 100 frames; late application contains 200. Early/late rates use their respective 2 s/4 s exposure; counts remain in every table. No negative opportunities or p-values are invented.", "",
           "Within each seed/reference/epoch, main effects average four paired differences; two-way terms average two differences of differences; VST is the third difference. Figures show the three seed values and their descriptive mean. Effects are on the absolute false-proposal-rate scale, and their interaction terms describe detector response rather than isolate a physical causal mechanism.", "",
           "Raw/Input residuals subtract the known fixed linear background in original coordinates. Variance is the ddof=0 spatial variance after removal of each residual frame's mean. Lag-one spatial covariance/correlation averages horizontal and vertical neighbor pairs; temporal pairs end at each reported frame, including epoch-boundary pairs. Quantile panels average per-frame quantiles, not a pooled pixel distribution. Correlation can change residual variance after the fixed Gaussian/EMA even though raw theoretical pointwise variance is matched. [Measured/theoretical variance values](variance_theory.tsv) use the frozen protocol's finite-filter calculations. The conditioned prediction is stationary; epoch averages include switching transients. Removing each finite field's spatial mean also lowers expected measured variance relative to pointwise variance under positive correlation, so ratios need not equal one exactly.", "",
           "Exact V-pair diagnostics and q1 frame counts through UI264 agree before the variance step. All diagnostics use existing stages/candidates after the complete seal/evaluation gate; no prediction, threshold, ranking or NMS decision is changed.", "",
           "| Reference | Late V effect | Late S effect | Late T effect |", "|---|---:|---:|---:|"]
    effects=tables["factorial_effects"]
    for arm in ARMS:
        values=[sum(r["difference"] for r in effects if r["arm_id"]==arm and r["epoch"]=="late" and r["effect"]==e and r["metric"]==RATE)/3 for e in "VST"]
        lines.append("| "+ARM_LABELS[arm].replace("\n","; ")+" | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    lines += ["", "Effects above use false proposals per 10,000 µm²·s. All interaction terms, setup/early contrasts and stage-statistic effects remain in the complete tables. These null controls cannot decide the recovery tradeoff or select a reference as optimal.", "",
              "[All source curves](curves.tsv) · [Epoch counts and diagnostics](epochs.tsv) · [Paired factorial effects](factorial_effects.tsv) · [Per-frame diagnostics](per_frame.tsv) · [Transient bins](time_bins.tsv) · [Legacy bridge](legacy_bridge.tsv) · [Native calibration](calibration.tsv)", ""]
    for f in figures:lines += ["!["+f["id"]+"]("+f["png"]+")", "", f["caption"], "", "[PDF export]("+f["pdf"]+")", ""]
    lines += ["[Protocol](../protocol.json) · [Evaluation](../evaluation_complete.json) · [Exact replication](../baseline_replication.json) · [Source bindings](sources_manifest.json) · [Report manifest](manifest.json)", "",
              "For paper placement: keep the local score and causal conditioning in architecture; put the V/S/T recipes, grid and setup/application intervals in experimental design. Pair the empirical false-proposal changes with residual covariance/spread diagnostics in Results. Describe nuisance robustness as a development boundary, with sensitivity and biological validation requiring separate evidence.", ""]
    if audit["scientific_audit_complete"]:lines.append("[Complete scientific audit](../audit_complete.json).")
    return "\n".join(lines)


def _qa(out, manifest, sources):
    path=out/"visual_qa.json"
    if not path.exists():return dict(visual_qa_complete=False,visual_qa_binding=None)
    qa=sources.read(path);require(qa["status"]=="PASS","Visual QA is not PASS")
    actual=[Path(b["path"]).resolve() for b in qa["figures"]]
    expected={(out/f["png"]).resolve() for f in manifest["figures"]}
    require(len(actual)==len(set(actual))==5 and set(actual)==expected,"QA must cover five current PNGs")
    for b in qa["figures"]:sources.check(b)
    if "pdf_companion_receipt" in qa:sources.check(qa["pdf_companion_receipt"])
    return dict(visual_qa_complete=True,visual_qa_binding=sources.bind(path))


def report(root=ROOT):
    root=common._mutable(root);out=root/"report"
    require(not out.exists() or not any(out.iterdir()),"Preserve previous reports; use an empty report directory")
    data=_load(root);per_frame=_per_frame(data);pairing=verify_early_variance_pairing(per_frame);epochs=epoch_tables(per_frame)
    tables=dict(curves=data["curves"],calibration=data["calibration"],per_frame=per_frame,epochs=epochs,
                factorial_effects=factorial_effects(epochs),time_bins=time_bins(per_frame),
                legacy_bridge=[r for r in epochs if r["case_kind"]=="legacy"],
                variance_theory=variance_theory_rows(epochs,data["protocol"]["theory"]))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():_write(out/(name+".json"),rows);_tsv(out/(name+".tsv"),rows)
    _write(out/"early_variance_pairing.json",pairing)
    figures=_plot(out,tables);inputs=data["sources"].finish();audit_sources=Sources()
    audit=_audit_status(root,data["protocol"],audit_sources);audit_sources.finish()
    dependencies=[_binding(Path(__file__)),_binding(Path(common.__file__))]
    _write(out/"sources_manifest.json",dict(status="VERIFIED_CONSUMED_SOURCES",inputs=inputs,reporting_code=dependencies,
        inherited_large_sources=data["inherited"],scope="All consumed Raw/Input/Spread/Z arrays and metadata checked by SHA256. Unconsumed stage/prefix bindings inherited from complete numerical seals; aggregate scientific audit verifies full artifacts."))
    (out/"REPORT.md").write_text(_text(tables,figures,audit))
    m=dict(schema_version=1,status="GENERATED",numerical_complete=True,scientific_audit_complete=audit["scientific_audit_complete"],
           report_stage=audit["status"],audit_verification=audit,visual_qa_complete=False,visual_qa_binding=None,
           figure_count=5,figures=figures,table_row_counts={k:len(v) for k,v in tables.items()},inputs=inputs,
           reporter=dependencies[0],reporting_code=dependencies,
           artifacts=[dict(_binding(p),path=p.name) for p in sorted(out.iterdir()) if p.is_file()])
    _write(out/"manifest.json",m);print(dict(status="GENERATED",figures=5,scientific_audit_complete=m["scientific_audit_complete"]),flush=True)
    return m


def update(root=ROOT):
    root=common._mutable(root);out=root/"report";m=_read(out/"manifest.json");sources=Sources()
    sources.check(m["reporter"],Path(__file__))
    for b in m["reporting_code"]+m["inputs"]:sources.check(b)
    for b in m["artifacts"]:sources.check(dict(b,path=str(out/b["path"])))
    p=sources.read(root/"protocol.json");audit=_audit_status(root,p,sources);qa=_qa(out,m,sources)
    tables={name:_read(out/(name+".json")) for name in m["table_row_counts"]};sources.finish()
    (out/"REPORT.md").write_text(_text(tables,m["figures"],audit))
    m.update(scientific_audit_complete=audit["scientific_audit_complete"],report_stage=audit["status"],audit_verification=audit,**qa)
    m["artifacts"]=[dict(_binding(out/b["path"]),path=b["path"]) for b in m["artifacts"]]
    _write(out/"manifest.json",m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("report","update"));parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args();globals()[args.command](args.root)


if __name__=="__main__":main()
