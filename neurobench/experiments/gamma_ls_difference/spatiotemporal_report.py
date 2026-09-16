"""Source-bound descriptive plots for the frozen spatial/causal-3D CFAR study.

Postprocessing only: no threshold, candidate, label, or operator is selected or
changed here. Partial reports explicitly retain missing-cell/media/annotation
gates. Synthetic active-frame precision and real sparse-window coverage are
never pooled together. Call run_report(root), or use --root CAMPAIGN_ROOT.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

STAGES = ("Raw", "Input", "A", "M", "Spread", "Score")
PRIMARY_METRICS = ("precision", "framewise_sensitivity", "event_window_coverage",
                   "proposals_per_frame", "first_delay_ms_mean_among_recovered")


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path), size_bytes=path.stat().st_size)


def _verify(record):
    if _binding(record["path"]) != record:
        raise ValueError(f"Changed sealed report input: {record['path']}")


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["no_rows"]
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict,list)) else value for key,value in row.items()} for row in rows)
    temporary.replace(path)


def _ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def _threshold_key(row):
    return "no_output" if row.get("threshold_z") is None else format(float(row["threshold_z"]), ".12g")


def _completion_paragraph(count, numeric_complete, audit_complete, audit_note):
    return (f"Numerical cells available: **{count}/357**. Numerical matrix: "
        f"**{'complete' if numeric_complete else 'pending'}**. Scientific media: "
        f"**{'complete according to the bound original audit aggregate' if audit_complete else 'pending'}**.\n\n{audit_note}")


def aggregate_synthetic(rows, *, strata="pooled"):
    """Pool counts/exposure, never ratios or conditional-delay medians."""
    groups = defaultdict(list)
    for row in rows:
        if row["truth_mode"] != "fully_synthetic":
            continue
        group = row["template_id"] if strata == "template" else "pooled"
        groups[(group, row["spec_id"], _threshold_key(row))].append(row)
    result = []
    for (group, spec_id, cutoff), values in sorted(groups.items()):
        tp = sum(row["true_positive_count"] for row in values)
        fp = sum(row["false_positive_count"] for row in values)
        fn = sum(row["false_negative_count"] for row in values)
        proposals = sum(row["proposal_count"] for row in values)
        frames = sum(row["application_frame_count"] for row in values)
        events = sum(row["event_count"] for row in values)
        recovered = sum(row["recovered_event_count"] for row in values)
        delay_n = sum(row["first_delay_recovered_denominator"] for row in values)
        delay_sum = sum((row["first_delay_ms_mean_among_recovered"] or 0)*row["first_delay_recovered_denominator"] for row in values)
        area_time = sum(row["eligible_area_px"]*.25*row["exposure_seconds"] for row in values)
        result.append(dict(stratum=group, spec_id=spec_id, threshold_label=cutoff,
            threshold_z=None if cutoff == "no_output" else float(cutoff), truth_mode="fully_synthetic",
            scene_seed_count=len(values), true_positive_count=tp, false_positive_count=fp,
            false_negative_count=fn, proposal_count=proposals, application_frame_count=frames,
            event_count=events, recovered_event_count=recovered, precision=_ratio(tp,tp+fp),
            framewise_sensitivity=_ratio(tp,tp+fn), event_window_coverage=_ratio(recovered,events),
            proposals_per_frame=_ratio(proposals,frames),
            false_proposals_per_10000_um2_s=_ratio(fp*10000,area_time),
            first_delay_ms_mean_among_recovered=_ratio(delay_sum,delay_n),
            first_delay_recovered_denominator=delay_n,
            pooling="sum counts and area-time; conditional mean delay weighted by recovered events"))
    return result


def _families(specs):
    primary = [s for s in specs if s["construction"] == "joint" and s["target_sigma_px"] == 1 and s["reference_n"] == 9]
    widths = [s for s in specs if s["construction"] == "joint" and s["radius_px"] == 7.5 and s["reference_n"] == 9 and s["time_scale_ms"] in (0,60)]
    shapes = [s for s in specs if s["construction"] == "joint" and s["radius_px"] == 7.5 and s["target_sigma_px"] == 1 and s["time_scale_ms"] in (0,60)]
    joint = [s for s in specs if s["radius_px"] == 7.5 and s["target_sigma_px"] == 1 and s["reference_n"] == 9 and (s["time_scale_ms"] == 60 or s["time_scale_ms"] == 0)]
    return [("radius_time", "Reference radius and causal time scale", primary),
            ("target_width", "Target-width controls", widths),
            ("reference_shape", "Reference-shape controls", shapes),
            ("joint_separable", "Joint geometry versus its product of marginals", joint)]


def _style(spec, family):
    colors = ["#3568b0", "#9461a8", "#ad6b32"]
    if family == "radius_time":
        color = colors[(5,7.5,10).index(spec["radius_px"])]
        label = f"R={spec['radius_px']*.5:g} µm, T={spec['time_scale_ms']:g} ms"
    elif family == "target_width":
        color = colors[(1,2,4).index(spec["target_sigma_px"])]
        label = f"Target σ={spec['target_sigma_px']*.5:g} µm, T={spec['time_scale_ms']:g} ms"
    elif family == "reference_shape":
        color = colors[(3,9,15).index(spec["reference_n"])]
        label = f"Shape n={spec['reference_n']:g}, T={spec['time_scale_ms']:g} ms"
    else:
        index = 0 if spec["time_scale_ms"] == 0 else 2 if spec["construction"] == "separable" else 1
        return dict(color=colors[index], linestyle=("-","--",":")[index],
                    label=("Spatial only", "Joint space-time", "Product of marginals")[index])
    return dict(color=color, linestyle={0:"-",20:"--",60:"-.",200:":"}[spec["time_scale_ms"]], label=label)


def _save(figure, path, plt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.png")
    figure.savefig(temporary, dpi=145)
    plt.close(figure); temporary.replace(path)


def _curves_plots(output, pooled, templated, families, plt):
    import numpy as np
    paths = []
    groups = [("pooled", pooled)] + [(name, [row for row in templated if row["stratum"] == name])
                for name in sorted({row["stratum"] for row in templated})]
    for stratum, rows in groups:
        for family, heading, specs in families:
            figure, axes = plt.subplots(2,2,figsize=(12,9))
            figure.subplots_adjust(bottom=.21,top=.86,hspace=.35,wspace=.26)
            any_values = [False]*4
            for spec in specs:
                curve = sorted((row for row in rows if row["spec_id"] == spec["spec_id"]), key=lambda row: math.inf if row["threshold_z"] is None else row["threshold_z"])
                style = _style(spec, family)
                for axis_index,(axis,xkey,ykey) in enumerate(zip(axes.flat,
                    ("threshold_z","threshold_z","framewise_sensitivity","false_proposals_per_10000_um2_s"),
                    ("precision","framewise_sensitivity","precision","framewise_sensitivity"))):
                    points = [(row.get(xkey),row.get(ykey)) for row in curve]
                    x = [np.nan if a is None else a for a,b in points]
                    y = [np.nan if b is None else b for a,b in points]
                    any_values[axis_index] |= any(math.isfinite(a) and math.isfinite(b) for a,b in zip(x,y))
                    axis.plot(x,y,linewidth=1.25,**style)
            labels = [("Score cutoff τ (strictly greater)","Framewise precision"),
                ("Score cutoff τ (strictly greater)","Active-frame sensitivity"),
                ("Active-frame sensitivity","Framewise precision"),
                ("False proposals / 10,000 µm² / second","Active-frame sensitivity")]
            for index,(axis,(xlabel,ylabel)) in enumerate(zip(axes.flat,labels)):
                axis.set(xlabel=xlabel,ylabel=ylabel,ylim=(-.02,1.02));axis.grid(alpha=.2)
                axis.set_xlim(left=0)
                if index in (0,1):
                    finite_cutoffs=[row["threshold_z"] for row in rows if row["threshold_z"] is not None]
                    axis.set_xlim(0,max(finite_cutoffs,default=1))
                if index == 2: axis.set_xlim(0,1.02)
                if index == 3: axis.set_xscale("symlog",linthresh=1)
                if not any_values[index]:
                    if index==3:
                        rates=[row["false_proposals_per_10000_um2_s"] for row in rows
                            if row["spec_id"] in {spec["spec_id"] for spec in specs} and row["false_proposals_per_10000_um2_s"] is not None]
                        axis.set_xlim(0,max(1,max(rates,default=0)*1.05))
                    axis.text(.5,.5,"Undefined: no active truth\nor no emitted proposals",transform=axis.transAxes,ha="center",va="center")
            handles, labels = axes[0,0].get_legend_handles_labels()
            figure.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.065),ncol=3,fontsize=8)
            figure.suptitle(f"{stratum.replace('_',' ')} — {heading}\nSynthetic truth only; counts pooled across available scene/seed replicates",fontsize=12)
            figure.text(.5,.015,"No-output precision is undefined. Curves are descriptive; no test cutoff is selected.\nFalse-proposal axis uses a symmetric-log scale with a linear interval near zero.",ha="center",fontsize=8)
            path = output/"curves"/f"{stratum}__{family}.png";_save(figure,path,plt);paths.append(path)
    return paths


def _fixed_rows(operating, datasets):
    rows = []
    for item in operating:
        summary = item["metrics"]["summary"]
        meta = datasets[item["case_id"]]
        row = {key:value for key,value in item.items() if key not in ("metrics","performance","setup_source_frames_ui")}
        row.update(summary)
        row.update(template_id=meta.get("template_id","real"), seed=meta.get("seed",0),
            exposure_seconds=item["application_frame_count"]*.02,
            fixed_calibration_threshold=item["threshold"], outcome_selected_cutoff=False)
        if row["truth_mode"] == "sparse_real":
            row["known_window_coverage"] = _ratio(summary["matched_known_positive_count"],summary["known_positive_count"])
        else:
            row["false_proposals_per_10000_um2_s"] = summary["false_positive_count"]*10000/(item["eligible_area_px"]*.25*row["exposure_seconds"])
        rows.append(row)
    return rows


def paired_fixed_rows(operating, specs):
    """Same scene and scale, each arm's own setup-frozen threshold; no test fit."""
    by_key = {(item["case_id"],item["spec_id"]):item for item in operating}
    spec_by_id = {spec["spec_id"]:spec for spec in specs}
    pairs = []
    for (case_id,spec_id), item in sorted(by_key.items()):
        spec = spec_by_id[spec_id]
        if spec["construction"] != "joint" or spec["time_scale_ms"] == 0:
            continue
        matches = [candidate for candidate in specs if candidate["construction"] == "joint" and candidate["time_scale_ms"] == 0 and
                   all(candidate[key] == spec[key] for key in ("radius_px","target_sigma_px","reference_n","dt_ms"))]
        if len(matches) != 1: raise ValueError("3D setting lacks a unique predeclared 2D counterpart")
        baseline = by_key.get((case_id,matches[0]["spec_id"]))
        if baseline is None: continue  # Partial report; missing comparisons remain pending.
        left,right = item["metrics"]["summary"],baseline["metrics"]["summary"]
        row = dict(case_id=case_id,three_d_spec_id=spec_id,two_d_spec_id=matches[0]["spec_id"],
            truth_mode=item["truth_mode"],threshold_3d=item["threshold"],threshold_2d=baseline["threshold"],
            comparison="3D minus 2D; same scene/seed and non-temporal parameters; separate setup-frozen cutoffs",
            proposals_3d=left["proposal_count"],proposals_2d=right["proposal_count"])
        for name in PRIMARY_METRICS:
            row[f"delta_{name}"] = None if left.get(name) is None or right.get(name) is None else left[name]-right[name]
        if item["truth_mode"] == "fully_synthetic":
            def active_hits(entry):
                return {(r["source_frame_ui"],r["matched_canonical_roi_id"]) for r in entry["metrics"]["proposal_rows"] if r["is_true_positive"]}
            a,b = active_hits(item),active_hits(baseline)
            truth_count = left["active_region_frame_count"]
            if truth_count != right["active_region_frame_count"]: raise ValueError("Paired active-truth denominators differ")
            row.update(active_frames_both=len(a&b),active_frames_gained=len(a-b),active_frames_lost=len(b-a),active_frames_neither=truth_count-len(a|b))
        else:
            a = {r["observation_id"] for r in item["metrics"]["occurrence_rows"] if r["matched"]}
            b = {r["observation_id"] for r in baseline["metrics"]["occurrence_rows"] if r["matched"]}
            truth_count = left["known_positive_count"]
            if truth_count != right["known_positive_count"]: raise ValueError("Paired known-window denominators differ")
            row.update(known_windows_both=len(a&b),known_windows_gained=len(a-b),known_windows_lost=len(b-a),known_windows_neither=truth_count-len(a|b),
                delta_known_window_coverage=(len(a)-len(b))/truth_count)
        pairs.append(row)
    return pairs


def _heatmaps(output, fixed, specs, plt):
    import numpy as np
    synthetic = [dict(row,threshold_z=0.,threshold_label="fixed") for row in fixed if row["truth_mode"] == "fully_synthetic"]
    pooled = aggregate_synthetic(synthetic)
    by_id = {row["spec_id"]:row for row in pooled}
    primary = [s for s in specs if s["construction"] == "joint" and s["target_sigma_px"] == 1 and s["reference_n"] == 9]
    metrics = [("precision","Precision",0,1),("framewise_sensitivity","Active-frame sensitivity",0,1),
        ("event_window_coverage","Event-window coverage",0,1),("proposals_per_frame","Proposals / application frame",0,None),
        ("first_delay_ms_mean_among_recovered","Mean first delay among recovered events (ms)",0,None)]
    figure, axes = plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for axis,(metric,title,low,high) in zip(axes.flat,metrics):
        values = np.full((3,4),np.nan)
        for spec in primary:
            value = by_id.get(spec["spec_id"],{}).get(metric)
            if value is not None: values[(5,7.5,10).index(spec["radius_px"]),(0,20,60,200).index(spec["time_scale_ms"])] = value
        image = axis.imshow(values,aspect="auto",vmin=low,vmax=high,cmap="viridis")
        axis.set(title=title,xticks=range(4),xticklabels=[0,20,60,200],yticks=range(3),yticklabels=[2.5,3.75,5],xlabel="Temporal reference scale T (ms)",ylabel="Reference profile mode R (µm)")
        for y in range(3):
            for x in range(4):
                value=values[y,x]
                ink="black" if np.isfinite(value) and image.norm(value)>.55 else "white"
                axis.text(x,y,"—" if not np.isfinite(value) else f"{value:.3g}",ha="center",va="center",color=ink,fontsize=9)
        figure.colorbar(image,ax=axis,shrink=.8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.95,"Each arm uses its setup-frozen cutoff.\nNo test-optimal threshold is selected.\n\nDelay excludes missed events; read it\nwith event-window coverage and counts.\n\nMain 12 settings only. Width, shape,\nand separable controls remain in\nthe full tables and curve panels.",va="top",fontsize=10)
    figure.suptitle("Fixed-calibration synthetic sensitivity — pooled counts across completed scenes",fontsize=13)
    path=output/"fixed_calibration_heatmaps.png";_save(figure,path,plt)
    return [path]


def _kernel_outputs(output, protocol, families, plt, inputs):
    import numpy as np
    from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec,build_kernels,support_convergence
    spec_fields = {field.name for field in fields(GammaSTSpec)}
    known = {item["spec_id"]:item for item in protocol["kernels"]}
    rows=[]; paths=[]
    for spec in protocol["specs"]:
        kernel = build_kernels(GammaSTSpec(**{key:value for key,value in spec.items() if key in spec_fields}))
        meta=kernel.metadata
        if meta != known[spec["spec_id"]]: raise ValueError("Rebuilt displayed kernel differs from the frozen protocol")
        row={key:value for key,value in spec.items()}
        row.update(spatial_half_width_px=meta["spatial_half_width_px"],maximum_history_ms=meta["maximum_history_ms"],maximum_lag_frames=meta["maximum_lag_frames"])
        for name in ("target","reference"):
            row.update({f"{name}_{key}":meta[name][key] for key in ("spatial_sigma_x_px","mean_lag_ms","lag_p90_ms","lag_p99_ms","effective_sample_count")})
        rows.append(row)
        figure,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for index,name in enumerate(("target","reference")):
            weights=getattr(kernel,name);h=weights.shape[1]//2
            if len(weights)==1:
                axes[0,index].imshow(weights[0]/weights[0].max(),origin="lower",cmap="gray",extent=[-(h+.5)*.5,(h+.5)*.5,-(h+.5)*.5,(h+.5)*.5])
                axes[0,index].set(xlabel="x offset (µm)",ylabel="y offset (µm)",title=f"{name.title()}: spatial slice, current frame")
            else:
                plane=weights[:,h,:]
                half_dt=float(spec["dt_ms"])/2
                axes[0,index].imshow(plane/plane.max(),origin="lower",aspect="auto",cmap="gray",extent=[-(h+.5)*.5,(h+.5)*.5,-half_dt,meta["maximum_history_ms"]+half_dt])
                axes[0,index].set(xlabel="x offset (µm)",ylabel="Past lag (ms)",title=f"{name.title()}: joint slice at y=0")
            marginal=meta[name]
            axes[0,2].plot(marginal["lag_ms"],marginal["lag_marginal"],marker="o",markersize=2,label=name.title())
            axes[1,0].plot(np.asarray(marginal["spatial_x_px"])*.5,marginal["spatial_x_marginal"],label=name.title())
            radial=weights[0,h,h:];axes[1,1].plot(np.arange(len(radial))*.5,radial/radial.max(),label=name.title())
        axes[0,2].set(title="Temporal marginal",xlabel="Past lag (ms)",ylabel="Normalized mass / sampled lag")
        axes[1,0].set(title="Spatial x marginal",xlabel="Offset (µm)",ylabel="Normalized mass / sampled x")
        axes[1,1].set(title="Current-frame radial amplitude",xlabel="Radius (µm)",ylabel="Fraction of each profile's peak")
        for axis in (axes[0,2],axes[1,0],axes[1,1]):axis.grid(alpha=.2);axis.legend(fontsize=8)
        axes[1,2].axis("off")
        text=["Actual normalized-kernel moments:"]
        for name in ("target","reference"):
            stats=meta[name];text.append(f"{name.title()}: σx={stats['spatial_sigma_x_px']*.5:.3g} µm\nmean lag={stats['mean_lag_ms']:.3g} ms; 90% lag={stats['lag_p90_ms']:g} ms")
        text += [f"Support: {meta['support_shape']}","Slice images are independently peak-normalized.\nMarginals retain normalized mass.\nT is not a calcium decay constant."]
        axes[1,2].text(0,.95,"\n\n".join(text),va="top",fontsize=9)
        figure.suptitle(f"{spec['spec_id']} — literal causal kernel; current and past samples only",fontsize=12)
        path=output/"kernels"/f"{spec['spec_id']}.png";_save(figure,path,plt);paths.append(path)
    convergence=[]
    checks=(GammaSTSpec(time_scale_ms=60),GammaSTSpec(radius_px=10,time_scale_ms=200),
            GammaSTSpec(reference_n=3,time_scale_ms=60),GammaSTSpec(target_sigma_px=4,time_scale_ms=60))
    for spec in checks:
        convergence.append(dict(spec_id=spec.spec_id,levels=support_convergence(spec)))
    _table(output/"kernel_summary.tsv",rows);_json(output/"support_convergence.json",convergence)
    return paths


def _real_curves(output, curves, fixed, families, plt):
    rows=[row for row in curves if row["truth_mode"]=="sparse_real"]
    if not rows:return []
    for row in rows:
        if any(row.get(key) is not None for key in ("precision","framewise_sensitivity","false_positive_count")):
            raise ValueError("Sparse real annotations must not supply precision or full sensitivity")
    paths=[]
    for family,title,specs in families:
        figure,axes=plt.subplots(1,2,figsize=(12,6));figure.subplots_adjust(bottom=.3,top=.82,wspace=.25)
        for spec in specs:
            points=sorted((row for row in rows if row["spec_id"]==spec["spec_id"]),key=lambda row: math.inf if row["threshold_z"] is None else row["threshold_z"])
            finite=[row for row in points if row["threshold_z"] is not None]
            axes[0].plot([row["threshold_z"] for row in finite],[_ratio(row["matched_known_positive_count"],row["known_positive_count"]) for row in finite],**_style(spec,family))
            axes[1].plot([row["proposal_count"] for row in points],[_ratio(row["matched_known_positive_count"],row["known_positive_count"]) for row in points],**_style(spec,family))
        for axis in axes:axis.set(ylim=(-.02,1.02),ylabel="Known-window coverage");axis.set_xlim(left=0);axis.grid(alpha=.2)
        axes[0].set_xlabel("Score cutoff τ");axes[1].set_xlabel("Actual emitted frame proposals")
        handles,labels=axes[0].get_legend_handles_labels();figure.legend(handles,labels,loc="lower center",ncol=3,fontsize=8)
        figure.suptitle(f"Sparse real recording — {title}\nKnown windows only; unmatched proposals remain unknown",fontsize=12)
        path=output/"real"/f"sparse_coverage__{family}.png";_save(figure,path,plt);paths.append(path)
    return paths


def _example_plots(root,output,datasets,cells,families,plt,dense_bindings,inputs):
    import numpy as np
    selected=[]; paths=[]; traces=[]
    for template in ("isolated_compact_fast","isolated_broad_slow","real"):
        cases=sorted(case for case,meta in datasets.items() if meta.get("template_id","real")==template)
        if not cases:continue
        case=cases[0];expert_path=root/"datasets"/case/"experts.json";inputs.append(_binding(expert_path))
        experts=sorted(_read(expert_path),key=lambda row:str(row["observation_id"]))
        if not experts:continue
        expert=experts[0];meta=datasets[case];frames=list(map(int,meta["source_frames_ui"]))
        selection=dict(case_id=case,observation_id=expert["observation_id"],rule="lexicographically first predeclared fast/slow scene, then first observation ID; real first observation ID; no outcome selection",
            x_px=expert["x_px"],y_px=expert["y_px"],source_start_ui=expert["source_start_ui"],source_stop_ui=expert["source_stop_ui"],truth_mode=meta["truth_mode"])
        selected.append(selection)
        x,y=int(math.floor(float(expert["x_px"])+.5)),int(math.floor(float(expert["y_px"])+.5))
        local={}
        selection["fixed_operating_outcomes"] = []
        for spec_id in sorted({s["spec_id"] for _,_,specs in families for s in specs}):
            item=cells.get((case,spec_id))
            if item is None:continue
            metrics=item["operating"]["metrics"]
            rows=metrics["event_rows"] if meta["truth_mode"] == "fully_synthetic" else metrics["occurrence_rows"]
            outcomes=[row for row in rows if str(row.get("observation_id",row.get("event_id"))) == str(expert["observation_id"])]
            if len(outcomes)!=1:raise ValueError("Fixed example lacks a unique saved event/window outcome")
            outcome=outcomes[0]
            selection["fixed_operating_outcomes"].append(dict(spec_id=spec_id,
                recovered=bool(outcome["recovered"] if meta["truth_mode"] == "fully_synthetic" else outcome["matched"]),
                outcome_grain="synthetic event window" if meta["truth_mode"] == "fully_synthetic" else "sparse real occurrence window",
                first_delay_ms=outcome.get("first_detection_delay_ms",outcome.get("first_delay_ms"))))
            stage_values={}
            for stage in STAGES:
                record=item["seal"]["stages"][stage];path=Path(record["path"]);stat=path.stat()
                if stat.st_size!=record["size_bytes"]:raise ValueError("Changed stage size before diagnostic plotting")
                array=np.load(path,mmap_mode="r",allow_pickle=False)
                if len(array)!=len(frames) or not (0<=x<array.shape[2] and 0<=y<array.shape[1]):raise ValueError("Example coordinate or source-frame map differs")
                stage_values[stage]=np.asarray(array[:,y,x],dtype=np.float64)
                dense_bindings.append(dict(**record,bytes_rehashed=False,observed_mtime_ns=stat.st_mtime_ns,selection=dict(case_id=case,spec_id=spec_id,stage=stage,x_px=x,y_px=y)))
            local[spec_id]=(stage_values,item["operating"]["threshold"])
            for i,frame in enumerate(frames):
                traces.append(dict(case_id=case,spec_id=spec_id,observation_id=expert["observation_id"],source_frame_ui=frame,x_px=x,y_px=y,
                    **{stage:float(stage_values[stage][i]) for stage in STAGES},frozen_threshold=item["operating"]["threshold"]))
        time=np.asarray(frames)*.02
        for family,title,specs in families:
            available=[spec for spec in specs if spec["spec_id"] in local]
            if not available:continue
            figure,axes=plt.subplots(3,2,figsize=(13,10),sharex=True);figure.subplots_adjust(bottom=.21,top=.88,hspace=.35,wspace=.22)
            for stage,axis in zip(STAGES,axes.flat):
                for spec in available:
                    values,cutoff=local[spec["spec_id"]];style=_style(spec,family)
                    axis.plot(time,values[stage],linewidth=.85,**style)
                    if stage=="Score":axis.axhline(cutoff,color=style["color"],linestyle=style["linestyle"],linewidth=.55,alpha=.45)
                    if stage in ("Raw","Input"):break  # These stages are identical supplied inputs.
                axis.axvspan(float(expert["source_start_ui"])*.02,float(expert["source_stop_ui"])*.02,color="gray",alpha=.14)
                axis.set_ylabel(stage);axis.grid(alpha=.2)
            for axis in axes[-1]:axis.set_xlabel("Source time (s; frame UI × 0.02 s)")
            handles,labels=axes[1,0].get_legend_handles_labels();figure.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.065),ncol=3,fontsize=8)
            figure.suptitle(f"{template.replace('_',' ')} | {expert['observation_id']} | exact pixel ({x},{y})\n{title}; {len(available)}/{len(specs)} completed settings",fontsize=11)
            figure.text(.5,.015,"Axes use stage-specific units; do not compare their heights. Faint Score lines are each setup-frozen cutoff.\nShading is synthetic active support or a broad real annotation window; traces are descriptive, not causal explanations or onset validation.",ha="center",fontsize=8)
            path=output/"examples"/f"{template}__{family}.png";_save(figure,path,plt);paths.append(path)
    _json(output/"example_selection.json",selected);_table(output/"example_traces.tsv",traces)
    return paths


def _load_inputs(root):
    protocol=_read(root/"protocol.json");inputs=[_binding(root/"protocol.json")]
    preflight=_read(root/"preflight.json");inputs.append(_binding(root/"preflight.json"))
    if preflight["protocol_sha256"]!=inputs[0]["sha256"]:raise ValueError("Frozen protocol changed")
    datasets={};cells={};curves=[];operating=[]
    for folder in sorted((root/"datasets").iterdir()):
        if not (folder/"prepared.json").exists():continue
        prepared=_read(folder/"prepared.json")
        if prepared.get("status")!="PASS":raise ValueError("Dataset preparation is incomplete")
        for key in ("metadata","experts","active"):_verify(prepared[key])
        meta=_read(folder/"metadata.json");datasets[folder.name]=meta
        inputs.extend([_binding(folder/"prepared.json"),_binding(folder/"metadata.json")])
        inputs.extend(prepared[key] for key in ("experts","active"))
        for spec in protocol["specs"]:
            cell=root/"cells"/folder.name/spec["spec_id"]
            if not (cell/"evaluated.json").exists():continue
            evaluated=_read(cell/"evaluated.json");seal=_read(cell/"sealed.json")
            if evaluated.get("status")!="PASS" or evaluated["candidate_seal_sha256"]!=_sha(cell/"sealed.json"):
                raise ValueError("Completed numerical cell has a changed original candidate seal")
            for key in ("prefix","calibration","audit_candidates"):_verify(seal[key])
            for name in ("evaluated.json","sealed.json","curves.json","operating_metrics.json","calibration.json","performance.json"):
                inputs.append(_binding(cell/name))
            inputs.extend(seal[key] for key in ("prefix","calibration","audit_candidates"))
            curve=_read(cell/"curves.json");op=_read(cell/"calibration.json");fixed=_read(cell/"operating_metrics.json")
            item={**spec,**op,"case_id":folder.name,"metrics":fixed,"performance":_read(cell/"performance.json")}
            operating.append(item);cells[(folder.name,spec["spec_id"])]=dict(seal=seal,operating=item,curve=curve)
            for row in curve["curve_rows"]:
                curves.append(dict(row,**spec,case_id=folder.name,scene_id=folder.name,template_id=meta.get("template_id","real"),seed=meta.get("seed",0),eligible_area_px=op["eligible_area_px"]))
    return protocol,datasets,cells,curves,operating,inputs


def _numerical_gate(root, count, inputs):
    checks=(("datasets_complete.json","case_count",17),
            ("computation_complete.json","cell_count",357),
            ("evaluation_complete.json","evaluated_cells",357))
    okay=count==357
    for name,key,expected in checks:
        path=root/name
        if not path.exists():okay=False;continue
        record=_read(path);inputs.append(_binding(path))
        okay &= record.get("status")=="PASS" and record.get(key)==expected
    for name in ("all_curves.json","all_curves.tsv","operating_points.json"):
        path=root/name
        if not path.exists():okay=False;continue
        inputs.append(_binding(path))
    return bool(okay)


def _descriptive_calibration(root, inputs):
    path=root/"real_review"/"descriptives"/"summary.tsv"
    if not path.exists():return None
    manifest_path=path.parent/"manifest.json";manifest=_read(manifest_path)
    if manifest.get("status")!="CALIBRATION_DESCRIPTIVES_COMPLETE":
        raise ValueError("Calibration descriptives lack their completed manifest")
    records=[row for row in manifest["artifacts"] if Path(row["path"]).resolve()==path.resolve()]
    if len(records)!=1:raise ValueError("Calibration summary lacks a unique manifest binding")
    _verify(records[0]);inputs.extend([_binding(manifest_path),records[0]])
    with path.open(newline="") as stream:rows=list(csv.DictReader(stream,delimiter="\t"))
    if len(rows)!=manifest["location_count"]:raise ValueError("Calibration location count differs from its manifest")
    return dict(location_count=len(rows),excursion_count=sum(int(r["calibration_excursion_count"]) for r in rows),
        rise_fit_count=sum(int(r["rise_fit_count"]) for r in rows),
        decay_fit_count=sum(int(r["decay_fit_count"]) for r in rows),
        unresolved_fast_rise_count=sum(int(r["unresolved_fast_rise_count"]) for r in rows),
        identified_footprint_count=sum(r["footprint_status"]=="described" for r in rows))


def _latency_rows(root, inputs):
    folder=root/"latency";path=folder/"summary.json"
    if not path.exists():return []
    index_path=folder/"artifact_index.json";index=_read(index_path)
    records={row["path"]:row for row in index["artifacts"]}
    if len(records)!=len(index["artifacts"]):raise ValueError("Duplicate latency artifact bindings")
    for row in records.values():
        if not Path(row["path"]).resolve().is_relative_to(folder.resolve()):raise ValueError("Latency artifact escapes its output root")
        _verify(row)
    for name in ("summary.json","contract.json","status.json"):
        if str((folder/name).resolve()) not in records:raise ValueError("Latency artifact index omits required metadata")
    summary=_read(path);contract=_read(folder/"contract.json");status=_read(folder/"status.json")
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    expected={(f"R7.5_T{time}_S1_n9_joint",rep) for time in (0,20,60,200) for rep in (1,2)}
    values=summary["results"]
    if (summary.get("status")!="PASS" or summary.get("complete") is not True or
        summary.get("all_stage_comparisons_pass") is not True or summary.get("all_nms_comparisons_pass") is not True or
        summary["contract_sha256"]!=digest or status.get("status")!="PASS" or status.get("complete") is not True or
        status["contract_sha256"]!=digest or summary["repetition_count"]!=8 or len(values)!=8 or
        {(row["spec_id"],row["repetition"]) for row in values}!=expected or
        Path(contract["study_root"]).resolve()!=root.resolve()):
        raise ValueError("Latency summary does not establish the declared eight parity-checked repetitions")
    result_records=[record for record in records.values() if Path(record["path"]).name=="result.json"]
    by_key={}
    for record in result_records:
        row=_read(record["path"]);key=(row["spec_id"],row["repetition"])
        if key in by_key:raise ValueError("Duplicate latency repetition results")
        by_key[key]=row
    rows=[]
    for value in values:
        key=(value["spec_id"],value["repetition"]);timing=value["timing"]
        if (by_key.get(key)!=value or value["status"]!="PASS" or value["contract_sha256"]!=digest or
            not value["stage_comparison"]["pass"] or not value["candidate_comparison"]["pass"] or
            timing["period_ms"]!=20. or timing["measured_frame_count"]!=96):
            raise ValueError("Latency repetition result differs from the saved parity/timing evidence")
        rows.append(dict(spec_id=value["spec_id"],time_scale_ms=value["operating_point"]["spec_id"].split("_T")[1].split("_")[0],
            repetition=value["repetition"],measured_frame_count=timing["measured_frame_count"],period_ms=timing["period_ms"],
            processing_p50_ms=timing["processing_ms"]["p50"],processing_p99_ms=timing["processing_ms"]["p99"],
            processing_max_ms=timing["processing_ms"]["max"],
            processing_deadline_exceedance_count=timing["processing_deadline_exceedance_count"],
            release_to_result_deadline_exceedance_count=timing["release_to_result_deadline_exceedance_count"],
            release_to_result_p99_ms=timing["release_to_result_ms"]["p99"],
            maximum_completion_lateness_ms=timing["completion_lateness_ms"]["max"],
            device=value["hardware"]["device_name"],maximum_history_ms=value["kernel_information_support"]["maximum_history_ms"],
            parity_pass=True,full_control_validation=False))
    inputs.extend([_binding(index_path),*records.values()])
    return sorted(rows,key=lambda row:(float(row["time_scale_ms"]),row["repetition"]))


def _audit_status(root, expected, inputs):
    path=root/"audit_complete.json"
    if not path.exists():return False,"Audit aggregate absent; scientific media completion pending."
    inputs.append(_binding(path));audit=_read(path);rows=audit.get("audits",[])
    keys={(row["case_id"],row["spec_id"]) for row in rows}
    okay=audit.get("status")=="PASS" and audit.get("cell_count")==len(expected) and len(rows)==len(keys) and keys==expected and all(row["summary"].get("scientific_audit_complete") for row in rows)
    required={"summary.json","status.json","run_contract.json","source_manifest.json",
              "artifact_index.json","llm_context.json","validation.json","inventory.json"}
    if okay:
        for row in rows:
            bindings=row.get("metadata_bindings",{})
            if not required.issubset(bindings):
                return False,"Audit aggregate lacks required per-cell metadata bindings; completion remains pending."
            folder=(root/"audits"/row["case_id"]/row["spec_id"]).resolve()
            if not folder.is_relative_to((root/"audits").resolve()):raise ValueError("Audit cell path escapes its campaign")
            for name in sorted(required):
                record=bindings[name]
                if Path(record["path"]).resolve()!=folder/name:raise ValueError("Audit metadata binding identifies a different cell")
                _verify(record);inputs.append(record)
            summary=_read(folder/"summary.json");status=_read(folder/"status.json");validation=_read(folder/"validation.json")
            if (summary!=row["summary"] or summary.get("scientific_audit_complete") is not True or
                status.get("status")!="complete" or status.get("scientific_audit_complete") is not True or
                validation.get("status")!="passed" or validation.get("scientific_audit_complete") is not True):
                return False,"Actual per-cell completion metadata disagrees with the original audit aggregate."
    return okay,"All declared cells and their eight bound completion metadata files are present in the original audit aggregate; media-byte verification is inherited from that completed validator." if okay else "Audit aggregate does not establish complete coverage of every declared cell."


def _report_text(protocol,datasets,operating,fixed,pairs,numeric_complete,audit_complete,audit_note,descriptives,latency,plots):
    lines=["# Spatial and causal spatiotemporal Gamma CFAR sensitivity", "",
        _completion_paragraph(len(operating),numeric_complete,audit_complete,audit_note),"",
        "This report describes a fixed sensitivity pilot; it selects no test-optimal kernel or cutoff. Synthetic truth and sparse real annotations remain separate. Exhaustive real-panel annotation is pending unless separately accepted and evaluated; no real precision or false-alarm rate is supplied here.","",
        "CFAR denotes the locally adaptive prescreening design here. The frozen setup rule limits an empirical proposal count; it does not establish a constant false-alarm probability. The locally standardized score is dimensionless, with no standard-normal interpretation assumed.","",
        "## What the metrics mean", "",
        "Synthetic precision is matched active-region/frame proposals divided by all emitted frame proposals. Active-frame sensitivity is matched pairs divided by all generated active-region/frame labels. Unmatched proposals, including same-frame duplicates, count as false proposals only under this exhaustive synthetic truth. No-output precision is undefined. Noise-only templates have no active-frame sensitivity denominator.","",
        "Event-window coverage asks whether an event has any matched active frame. First-detection delay is measured from the simulated fluorescence onset and is summarized only among recovered events; missed events remain in the coverage denominator. These event measures are separate from framewise precision. Real results instead use sparse broad-window matching and unknown unmatched candidates; they establish neither full sensitivity nor onset latency.","",
        "Pooled curves sum counts and exposure across scenes. Replicates are paired scene/seeds within fixed templates; frames are not independent recordings. This report does not invent confidence intervals from adjacent frames or select a maximum from the test curves.","",
        "## Read the results", "",
        "- [Main radius/time curves](curves/pooled__radius_time.png), [target widths](curves/pooled__target_width.png), [reference shapes](curves/pooled__reference_shape.png), and [joint versus separable](curves/pooled__joint_separable.png). Every available template has the same four panels in `curves/`.",
        "- [Fixed-calibration heatmaps](fixed_calibration_heatmaps.png), [all fixed operating points](fixed_calibration.tsv), and [paired same-scene 3D-minus-2D comparisons](paired_fixed_2d_3d.tsv). Each arm uses its own setup-frozen threshold; equal calibration targets do not force equal application burden.",
        "- [Pooled synthetic curves](pooled_synthetic_curves.tsv), [template curves](synthetic_template_curves.tsv), and [all case curves](all_case_curves.tsv). False proposals per 10,000 square micrometers per second use the stated evaluable area and exposure; this is not a true-negative-based false-positive rate.",
        "- [Sparse real coverage and burden](real/sparse_coverage__radius_time.png), [kernel widths and lags](kernel_summary.tsv), [support convergence](support_convergence.json), and [performance records](performance.tsv).",
        "- [Fixed example selection](example_selection.json) and [all example stage traces](example_traces.tsv). Fast, slow, and real examples are chosen by fixed scene/observation IDs, not success or failure; their outcomes are retained without relabeling them as representative prevalence.","",
        "## Fixed-calibration observations", ""]
    if not plots:
        lines.insert(4,"Graphical and example links are intentionally pending in this tables-only generation; no old figure is promoted by this report.\n")
    for time in (0,20,60,200):
        rows=[row for row in fixed if row["truth_mode"]=="fully_synthetic" and row["radius_px"]==7.5 and row["target_sigma_px"]==1 and row["reference_n"]==9 and row["time_scale_ms"]==time and row["construction"]=="joint"]
        if not rows:continue
        proposals=sum(row["proposal_count"] for row in rows);tp=sum(row["true_positive_count"] for row in rows);active=sum(row["active_region_frame_count"] for row in rows)
        recovered=sum(row["recovered_event_count"] for row in rows);events=sum(row["event_count"] for row in rows)
        lines.append(f"- Central radius, T={time:g} ms, {len(rows)} completed scenes: {tp}/{active} active-frame matches, {recovered}/{events} event windows recovered, {proposals} emitted frame proposals. These are fixed-setting observations, not a ranking.")
    real=[row for row in fixed if row["truth_mode"]=="sparse_real"]
    for row in real:
        if row["radius_px"]==7.5 and row["target_sigma_px"]==1 and row["reference_n"]==9 and row["construction"]=="joint":
            lines.append(f"- Real central radius, T={row['time_scale_ms']:g} ms: {row['matched_known_positive_count']}/{row['known_positive_count']} evaluable known windows matched with {row['proposal_count']} frame proposals; precision remains unidentified.")
    lines += ["","## Interpretation and remaining gates", "",
        "The literal third coordinate changes shell mass and realized spatial widths. Compare the saved target/reference marginals and the separable control before attributing changes to joint geometry. The reference includes a target's own past location. Its temporal scale is a kernel parameter, not a fitted calcium decay constant, a fixed output delay, or proof of biological causation.","",
        "Physical coordinates use 50 Hz and 0.5 micrometers per pixel. The reported approximately 30 Hz indicator bandwidth has an unknown definition/identity and exceeds the movie's 25 Hz Nyquist frequency; this study does not turn its reciprocal into a decay time.","",
        "Convolution timing is buffered batch throughput with declared chunk size, device, and minimum input-ring bytes. It does not establish single-arrival latency, queue stability, complete detector throughput, or feedback-control performance. Kernel history and buffered chunks must remain explicit when interpreting speed or delay.","",
        "Displayed exact-pixel stage traces use the sealed source-array paths and declared full-file digests, with sizes checked at use; dense arrays are not rehashed by this postprocessor. Full source/media validation belongs to the original audit. Plot review is a separate gate; generation alone does not mark figures visually approved.","",
        "[Original protocol](../protocol.json) · [Original aggregate curves](../all_curves.tsv) · [Original operating points](../operating_points.json) · [Scientific audits](../audits/) · [Audit aggregate](../audit_complete.json) · [Real annotation review instructions](../real_review/REVIEW_INSTRUCTIONS.md) · [Real annotation acceptance](../real_review/annotation_acceptance.json) · [Calibration descriptives](../real_review/descriptives/summary.tsv)."]
    real_meta=[meta for meta in datasets.values() if meta["truth_mode"]=="sparse_real"]
    for meta in real_meta:
        lines += ["",f"The common source halo and NMS interior retain {meta['known_occurrences_evaluable']}/{meta['known_occurrences_before_crop']} previously known real occurrence windows. The {len(meta['excluded_geometry_occurrences'])} excluded windows are geometry exclusions, not detector misses; their original coordinates and IDs are retained in [real dataset metadata](../datasets/real/metadata.json)."]
    if descriptives is not None:
        d=descriptives
        lines += ["",f"Separate pre-evaluation descriptions (source frames 1–1799; detector setup used frames 1600–1799) examined {d['location_count']} known coordinate variants and found {d['excursion_count']} candidate fluorescence excursions: {d['rise_fit_count']} accepted descriptive rise fits, {d['decay_fit_count']} accepted descriptive decay fits, and {d['identified_footprint_count']} identified activity-footprint widths; {d['unresolved_fast_rise_count']} rise was explicitly unresolved at the native sampling interval. These guarded descriptions do not identify the indicator impulse response or establish a physical match between kernel scales and neuronal kinetics."]
    if latency:
        lines += ["","## Measured one-frame processing latency","",
            "Two fresh-process repetitions measured 96 source frames each on the stated RTX 4070 SUPER, with one frame released every 20 ms and backlog retained. Every repetition passed stage and candidate parity against the sealed computation; that PASS denotes benchmark validity, not that every setting met its deadline.","",
            "| Temporal scale (ms) | Processing median, repetitions 1 / 2 (ms) | Processing 99th percentile, repetitions 1 / 2 (ms) | Frames exceeding 20 ms processing time |",
            "| ---: | ---: | ---: | ---: |"]
        for time in (0,20,60,200):
            values=[row for row in latency if float(row["time_scale_ms"])==time]
            medians=" / ".join(f"{row['processing_p50_ms']:.2f}" for row in values)
            p99=" / ".join(f"{row['processing_p99_ms']:.2f}" for row in values)
            lines.append(f"| {time} | {medians} | {p99} | {sum(row['processing_deadline_exceedance_count'] for row in values)}/{sum(row['measured_frame_count'] for row in values)} |")
        lines += ["","The measured interval includes one-frame causal convolution, transfers, valid-mass normalization, local score, and exact NMS. It excludes Gaussian/EMA/difference conditioning, capture, camera transport, decision/communication logic, actuator hardware, and biological delay. These short repetitions neither guarantee hard real-time behavior nor validate feedback control. Kernel history describes available past information separately from processing time. [Eight repetition rows](latency.tsv), [full latency results and scope](../latency/summary.json), and [hash-bound benchmark artifacts](../latency/artifact_index.json) retain release lateness, startup, parity, and hardware details."]
    return "\n".join(lines)+"\n"


def run_report(root: str | Path, *, plots: bool = True) -> dict[str,Any]:
    root=Path(root).resolve();protocol,datasets,cells,curves,operating,inputs=_load_inputs(root)
    if len(protocol["specs"])!=21 or len({s["spec_id"] for s in protocol["specs"]})!=21:
        raise ValueError("Expected the frozen 21-setting design with unique IDs")
    if not operating:raise ValueError("No completed numerical cells available for a report")
    expected={(case,spec["spec_id"]) for case in datasets for spec in protocol["specs"]}
    numeric_gate=_numerical_gate(root,len(cells),inputs)
    numeric_complete=len(datasets)==17 and len(cells)==357 and set(cells)==expected and numeric_gate
    if numeric_complete:
        # The report reconstructs cell tables for partial-run support; the final
        # reconstruction must agree with the original aggregate authority.
        aggregate=_read(root/"all_curves.json")
        compare=lambda rows: {(r["case_id"],r["spec_id"],_threshold_key(r)):
            {k:v for k,v in r.items() if k!="false_proposals_per_10000_um2_s"} for r in rows}
        if len(aggregate)!=len(curves) or compare(aggregate)!=compare(curves):
            raise ValueError("Reconstructed curves disagree with the original aggregate")
        actual=_read(root/"operating_points.json")
        if actual!=operating:raise ValueError("Reconstructed operating points disagree with the original aggregate")
    descriptives=_descriptive_calibration(root,inputs)
    latency=_latency_rows(root,inputs)
    audit_complete,audit_note=_audit_status(root,expected,inputs)
    audit_complete = audit_complete and numeric_complete
    output=root/"report"
    if output.exists() and any(output.iterdir()) and not (output/"manifest.json").exists():raise FileExistsError("Unrelated report directory already exists")
    output.mkdir(exist_ok=True)
    _json(output/"manifest.json",dict(status="WRITING",numeric_complete=False,scientific_audit_complete=False))
    pooled=aggregate_synthetic(curves);templated=aggregate_synthetic(curves,strata="template")
    fixed=_fixed_rows(operating,datasets);pairs=paired_fixed_rows(operating,protocol["specs"])
    _table(output/"all_case_curves.tsv",curves);_table(output/"pooled_synthetic_curves.tsv",pooled);_table(output/"synthetic_template_curves.tsv",templated)
    _table(output/"fixed_calibration.tsv",fixed);_table(output/"paired_fixed_2d_3d.tsv",pairs)
    _table(output/"performance.tsv",[dict(case_id=item["case_id"],spec_id=item["spec_id"],**item["performance"]) for item in operating])
    _table(output/"latency.tsv",latency)
    dense_bindings=[];figures=[];families=_families(protocol["specs"])
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures += _curves_plots(output,pooled,templated,families,plt)
        figures += _heatmaps(output,fixed,protocol["specs"],plt)
        figures += _kernel_outputs(output,protocol,families,plt,inputs)
        figures += _real_curves(output,curves,fixed,families,plt)
        figures += _example_plots(root,output,datasets,cells,families,plt,dense_bindings,inputs)
    text=_report_text(protocol,datasets,operating,fixed,pairs,numeric_complete,audit_complete,audit_note,descriptives,latency,plots)
    temporary=output/"REPORT.md.tmp";temporary.write_text(text);temporary.replace(output/"REPORT.md")
    completion_names={"protocol.json","preflight.json","datasets_complete.json","computation_complete.json",
                      "evaluation_complete.json","all_curves.json","all_curves.tsv","operating_points.json"}
    context=dict(schema_version=1,numerical_cell_count=len(cells),numeric_complete=numeric_complete,
        expected_cells=[dict(case_id=case,spec_id=spec) for case,spec in sorted(expected)],
        completion_inputs=[record for record in inputs if Path(record["path"]).parent==root and Path(record["path"]).name in completion_names],
        completion_paragraph=_completion_paragraph(len(cells),numeric_complete,audit_complete,audit_note))
    _json(output/"completion_context.json",context)
    generated=[output/name for name in ("REPORT.md","completion_context.json","all_case_curves.tsv",
        "pooled_synthetic_curves.tsv","synthetic_template_curves.tsv","fixed_calibration.tsv",
        "paired_fixed_2d_3d.tsv","performance.tsv","latency.tsv")]
    if plots:
        generated += figures+[output/name for name in ("kernel_summary.tsv","support_convergence.json","example_selection.json","example_traces.tsv")]
    artifacts=[dict(path=str(path.relative_to(output)),sha256=_sha(path),size_bytes=path.stat().st_size)
               for path in sorted(set(generated))]
    result=dict(schema_version=1,status="GENERATED",numerical_cell_count=len(cells),expected_cell_count=357,
        numeric_complete=numeric_complete,scientific_audit_complete=audit_complete,visual_qa_complete=False,
        real_precision_identified=False,selected_test_optimum=False,reporter=_binding(Path(__file__)),
        inputs=list({row["path"]:row for row in inputs}.values()),dense_stage_bindings=dense_bindings,
        dense_stage_bytes_rehashed=False,figure_count=len(figures),artifacts=artifacts,
        scope="descriptive synthetic framewise accuracy and separate sparse-real coverage; no kernel or test cutoff selected")
    _json(output/"manifest.json",result)
    return result


def refresh_completion(root: str | Path) -> dict[str,Any]:
    """Update completion prose only; retain plots, their bindings, and QA fields.

    This does not reread/replot dense data or regenerate numerical tables. The
    numerical aggregate inputs and every previously declared report artifact
    must still match. The original audit aggregate alone supplies media status;
    a partial numerical report can never be promoted by this operation.
    """
    root=Path(root).resolve();output=root/"report"
    manifest=_read(output/"manifest.json")
    if manifest.get("status")!="GENERATED":raise ValueError("Only a fully written report supports status refresh")
    artifacts={row["path"]:row for row in manifest["artifacts"]}
    if len(artifacts)!=len(manifest["artifacts"]):raise ValueError("Duplicate report artifact paths")
    for required in ("REPORT.md","completion_context.json"):
        if required not in artifacts:raise ValueError(f"Missing bound status input: {required}")
    for row in artifacts.values():
        path=(output/row["path"]).resolve()
        if not path.is_relative_to(output):raise ValueError("Report artifact path escapes its root")
        if _sha(path)!=row["sha256"] or path.stat().st_size!=row["size_bytes"]:
            raise ValueError(f"Changed report artifact before status refresh: {row['path']}")
    context=_read(output/"completion_context.json")
    for record in context["completion_inputs"]:_verify(record)
    expected={(row["case_id"],row["spec_id"]) for row in context["expected_cells"]}
    numeric_complete=context["numeric_complete"] is True and context["numerical_cell_count"]==357 and len(expected)==357
    if numeric_complete != manifest["numeric_complete"]:raise ValueError("Report numerical status disagrees with its bound context")
    audit_inputs=[];audit_complete,audit_note=_audit_status(root,expected,audit_inputs)
    audit_complete=bool(audit_complete and numeric_complete)
    old=context["completion_paragraph"]
    new=_completion_paragraph(context["numerical_cell_count"],numeric_complete,audit_complete,audit_note)
    text=(output/"REPORT.md").read_text()
    if text.count(old)!=1:raise ValueError("Expected exactly one original completion paragraph")
    if old!=new:
        temporary=output/"REPORT.md.tmp";temporary.write_text(text.replace(old,new,1));temporary.replace(output/"REPORT.md")
        context["completion_paragraph"]=new;_json(output/"completion_context.json",context)
    for name in ("REPORT.md","completion_context.json"):
        record=_binding(output/name);artifacts[name]=dict(record,path=name)
    manifest["artifacts"]=[artifacts[row["path"]] for row in manifest["artifacts"]]
    current={record["path"]:record for record in manifest["inputs"]}
    current.pop(str(root/"audit_complete.json"),None)
    current.update({record["path"]:record for record in audit_inputs})
    manifest.update(inputs=list(current.values()),scientific_audit_complete=audit_complete,
        completion_updater=_binding(Path(__file__)),completion_refresh_scope="Report completion paragraph and manifest only; original plots, numerical tables, dense-source bindings and visual-QA fields preserved")
    _json(output/"manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--no-plots",action="store_true",help="Produce numeric tables/report only; graphical review remains pending")
    parser.add_argument("--refresh-status",action="store_true",help="Refresh completion prose/manifest only; preserve existing figures and visual QA")
    args=parser.parse_args()
    if args.no_plots and args.refresh_status:parser.error("--no-plots and --refresh-status are mutually exclusive")
    result=refresh_completion(args.root) if args.refresh_status else run_report(args.root,plots=not args.no_plots)
    print(json.dumps({key:result[key] for key in ("status","numerical_cell_count","numeric_complete","scientific_audit_complete","figure_count")},sort_keys=True))


if __name__=="__main__":main()
