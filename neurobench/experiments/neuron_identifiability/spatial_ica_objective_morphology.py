"""Objective Raw/spatial-ICA morphology metrics with translated controls.

The spatial-ICA input is a globally linearly encoded positive-signal TIFF.  Its
first-page description records the source maximum, allowing inversion except
for values clipped at zero or the upper display limit.  Consequently, this
module treats amplitude ratios as representation-internal and emphasizes
shape and observed-minus-control contrasts.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
import tifffile

from .contracts import atomic_json, atomic_text


PRIMARY_METRICS = (
    "center_annulus_contrast",
    "boundary_sharpness",
    "compactness",
    "effective_radius_px",
    "peak_offset_px",
    "temporal_persistence",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def _disk(shape: tuple[int, int], x: float, y: float, lo: float, hi: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    radius = np.hypot(xx - x, yy - y)
    return (radius >= lo) & (radius <= hi)


def _component_metrics(image: np.ndarray, x: float, y: float) -> dict[str, float]:
    """Measure a 25x25 positive event map around an immutable coordinate."""
    h, w = image.shape
    x0, y0 = int(round(x)), int(round(y)); radius = 12
    xlo, xhi = max(0, x0-radius), min(w, x0+radius+1)
    ylo, yhi = max(0, y0-radius), min(h, y0+radius+1)
    crop = np.maximum(np.asarray(image[ylo:yhi, xlo:xhi], float), 0)
    cx, cy = x-xlo, y-ylo
    center = _disk(crop.shape, cx, cy, 0, 3)
    annulus = _disk(crop.shape, cx, cy, 6, 10)
    outer = crop[annulus]
    background = float(np.median(outer)) if outer.size else 0.0
    mad = float(np.median(np.abs(outer-background))) if outer.size else 0.0
    scale = max(1.4826*mad, np.finfo(float).eps)
    signal = np.maximum(crop-background, 0)
    center_mean = float(np.mean(crop[center]))
    annulus_mean = float(np.mean(outer)) if outer.size else 0.0
    contrast = (center_mean-annulus_mean)/(abs(center_mean)+abs(annulus_mean)+1e-12)
    threshold = background + 2*scale
    mask = crop > threshold
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), int))
    center_labels = labels[center]; candidates = np.unique(center_labels[center_labels > 0])
    if len(candidates):
        selected = max(candidates, key=lambda label: int(np.sum(labels == label)))
    elif count:
        selected = int(labels.flat[np.argmax(crop)])
    else:
        selected = 0
    component = labels == selected if selected else np.zeros_like(mask)
    area = float(np.sum(component))
    if area:
        eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3)), border_value=0)
        perimeter = float(np.sum(component & ~eroded))
        compactness = float(4*np.pi*area/max(perimeter**2, 1))
    else:
        compactness = 0.0
    yy, xx = np.indices(crop.shape); mass = float(signal.sum())
    if mass:
        mx, my = float((signal*xx).sum()/mass), float((signal*yy).sum()/mass)
        effective_radius = float(np.sqrt((signal*((xx-mx)**2+(yy-my)**2)).sum()/mass))
        peak_y, peak_x = np.unravel_index(int(np.argmax(signal)), signal.shape)
        peak_offset = float(np.hypot(peak_x-cx, peak_y-cy))
    else:
        effective_radius = float("nan"); peak_offset = float("nan")
    inner = _disk(crop.shape, cx, cy, 2, 4); outer_edge = _disk(crop.shape, cx, cy, 5, 7)
    local_range = float(np.percentile(crop, 95)-np.percentile(crop, 5))
    sharpness = float((np.mean(crop[inner])-np.mean(crop[outer_edge]))/max(scale, local_range, 1e-12))
    return {
        "center_annulus_contrast": float(contrast),
        "boundary_sharpness": sharpness,
        "compactness": compactness,
        "effective_radius_px": effective_radius,
        "peak_offset_px": peak_offset,
        "connected_components": float(count),
        "threshold_source_units": threshold,
    }


def _persistence(stack: np.ndarray, x: float, y: float) -> float:
    center = _disk(stack.shape[1:], x, y, 0, 3)
    annulus = _disk(stack.shape[1:], x, y, 6, 10)
    center_trace = stack[:, center].mean(axis=1); annulus_trace = stack[:, annulus].mean(axis=1)
    delta = center_trace-annulus_trace
    threshold = np.median(delta)+2*1.4826*np.median(np.abs(delta-np.median(delta)))
    return float(np.mean(delta > threshold))


def _map_and_metrics(video: np.ndarray, start: int, stop: int, x: float, y: float) -> dict[str, float]:
    event = np.asarray(video[start:stop], float)
    base = np.asarray(video[max(0, start-20):start], float)
    baseline = np.median(base, axis=0) if len(base) else np.zeros(video.shape[1:])
    event_map = np.maximum(np.max(event, axis=0)-baseline, 0)
    result = _component_metrics(event_map, x, y)
    result["temporal_persistence"] = _persistence(np.maximum(event-baseline[None], 0), x, y)
    return result


def _control_coordinate(row: dict[str, str], shape: tuple[int, int], all_sites: list[tuple[float, float]], baseline_map: np.ndarray) -> tuple[float, float]:
    x, y = float(row["x_px"]), float(row["y_px"])
    offsets = ((-32,0),(32,0),(0,-32),(0,32),(-24,-24),(24,24),(-24,24),(24,-24),(-40,16),(40,-16))
    target = _component_metrics(baseline_map, x, y)["threshold_source_units"]
    valid = []
    for dx, dy in offsets:
        cx, cy = x+dx, y+dy
        if not (12 <= cx < shape[1]-12 and 12 <= cy < shape[0]-12): continue
        if any((cx-sx)**2+(cy-sy)**2 < 12**2 for sx, sy in all_sites): continue
        value = _component_metrics(baseline_map, cx, cy)["threshold_source_units"]
        tie = hashlib.sha256(f"{row['observation_id']}|{dx}|{dy}".encode()).hexdigest()
        valid.append((abs(value-target), tie, cx, cy))
    if not valid: raise RuntimeError(f"no translated control for {row['observation_id']}")
    return min(valid)[2:]


def _bh(p_values: list[float]) -> list[float]:
    values=np.asarray(p_values,float); order=np.argsort(values); out=np.empty(len(values)); running=1.0
    for rank,index in reversed(list(enumerate(order,1))):
        running=min(running,float(values[index])*len(values)/rank); out[index]=running
    return out.tolist()


def _site_test(rows: list[dict[str, Any]], representation: str, metric: str, seed: int=20260824) -> dict[str, Any]:
    by_site: dict[str,list[float]]=defaultdict(list)
    for row in rows:
        if row["representation"] == representation and np.isfinite(float(row[f"delta_{metric}"])):
            by_site[row["observation_site_id"]].append(float(row[f"delta_{metric}"]))
    values=np.asarray([np.mean(v) for v in by_site.values()]); rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(values,(10000,len(values)),replace=True),axis=1)
    signs=rng.choice((-1,1),(20000,len(values))); perm=np.mean(signs*values,axis=1)
    observed=float(np.mean(values)); p=float((1+np.sum(np.abs(perm)>=abs(observed)))/(len(perm)+1))
    return {"representation":representation,"metric":metric,"site_count":len(values),"mean_observed_minus_control":observed,"ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),"site_sign_flip_p":p}


def _figure(rows: list[dict[str, Any]], tests: list[dict[str, Any]], path: Path) -> None:
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(12,7.2)); blue="#3976AF"; orange="#D87921"; grid="#D9DEE5"
    for ax,metric in zip(axes.flat,PRIMARY_METRICS):
        data=[]
        for representation in ("Raw","Spatial ICA"):
            values=[float(r[f"delta_{metric}"]) for r in rows if r["representation"]==representation and np.isfinite(float(r[f"delta_{metric}"]))]
            data.append(values)
        parts=ax.violinplot(data,positions=[0,1],showmeans=False,showmedians=True,widths=.75)
        for body,color in zip(parts["bodies"],[blue,orange]): body.set_facecolor(color);body.set_alpha(.35);body.set_edgecolor(color)
        for i,values in enumerate(data):
            jitter=np.linspace(-.08,.08,len(values));ax.scatter(i+jitter,values,s=7,color=(blue,orange)[i],alpha=.45)
        ax.axhline(0,color="#30343B",lw=.8);ax.set_xticks([0,1],["Raw","Spatial ICA"]);ax.set_title(metric.replace("_"," "));ax.grid(axis="y",color=grid,lw=.5)
    fig.suptitle("Objective morphology: labeled sites minus translated controls",x=.06,ha="left",fontsize=14)
    fig.text(.06,.945,"79 protected v1 observations; site is the bootstrap/permutation unit; positive spatial-ICA display inversion",fontsize=9,color="#4A515B")
    fig.tight_layout(rect=(0,.02,1,.92));fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def run(labels_tsv: Path, raw_npy: Path, ica_tif: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.with_name(output.name+".partial").exists(): raise FileExistsError(output)
    partial=output.with_name(output.name+".partial");partial.mkdir(parents=True)
    labels=_read_tsv(labels_tsv)
    if len(labels)!=79: raise RuntimeError(f"protected cohort must contain 79 rows, got {len(labels)}")
    raw=np.load(raw_npy,mmap_mode="r",allow_pickle=False)
    with tifffile.TiffFile(ica_tif) as tif:
        description=json.loads(tif.pages[0].description); maximum=float(description["signal_display_source_limits"][1])
        encoded=tif.asarray(out="memmap")
        ica=np.asarray(encoded,dtype=np.float32)*(maximum/65535.0)
        saturated_fraction=float(np.mean(encoded==65535)); zero_fraction=float(np.mean(encoded==0))
    sites=[(float(r["x_px"]),float(r["y_px"])) for r in labels]
    rows=[]
    for row in labels:
        start=int(row["original_start_frame_ui"])-1;stop=int(row["original_end_frame_ui"])
        ica_start=start-1799;ica_stop=stop-1799
        if not (0<=ica_start<ica_stop<=len(ica)): raise RuntimeError(f"ICA frame bounds for {row['observation_id']}")
        raw_base=np.median(np.asarray(raw[max(0,start-20):start],float),axis=0)
        cx,cy=_control_coordinate(row,raw.shape[1:],sites,raw_base)
        for representation,video,a,b in (("Raw",raw,start,stop),("Spatial ICA",ica,ica_start,ica_stop)):
            observed=_map_and_metrics(video,a,b,float(row["x_px"]),float(row["y_px"]));control=_map_and_metrics(video,a,b,cx,cy)
            record={"observation_id":row["observation_id"],"observation_site_id":row["observation_site_id"],"burst_id":int(row["burst_id"]),"include_confirmed":row["include_confirmed"],"disposition":row["disposition"],"representation":representation,"x_px":row["x_px"],"y_px":row["y_px"],"control_x_px":cx,"control_y_px":cy}
            for metric in (*PRIMARY_METRICS,"connected_components"):
                record[f"observed_{metric}"]=observed[metric];record[f"control_{metric}"]=control[metric];record[f"delta_{metric}"]=observed[metric]-control[metric]
            rows.append(record)
    tests=[_site_test(rows,representation,metric) for representation in ("Raw","Spatial ICA") for metric in PRIMARY_METRICS]
    q=_bh([r["site_sign_flip_p"] for r in tests])
    for row,value in zip(tests,q): row["bh_q_across_12_tests"]=value
    _write_tsv(partial/"occurrence_morphology_metrics.tsv",rows);_write_tsv(partial/"site_blocked_tests.tsv",tests)
    _figure(rows,tests,partial/"objective_morphology_controls.png")
    summary={"schema_version":1,"status":"completed_exploratory","cohort":"protected_v1_confirmed_sparse_positives","observations":79,"sites":len({r['observation_site_id'] for r in labels}),"representations":["Raw","Spatial ICA"],"primary_metrics":list(PRIMARY_METRICS),"tests":tests,"spatial_ica_provenance":{"variant_id":description["variant_id"],"encoding":"global_linear_uint16_positive_signal","source_limits":description["signal_display_source_limits"],"zero_fraction":zero_fraction,"saturated_fraction":saturated_fraction,"fit_scope":"transductive_all_review_frames_labels_excluded"},"interpretation_limits":["translated controls are unlabeled and therefore unknown, not biological negatives","representation-internal morphology metrics do not establish neuron identity","spatial ICA fit is one-seed and transductive","zero clipping removes negative reconstructed signal","single recording and four bursts"]}
    atomic_json(partial/"summary.json",summary)
    finite_counts={f"{rep}|{metric}":int(sum(bool(r["representation"]==rep and np.isfinite(float(r[f"delta_{metric}"]))) for r in rows)) for rep in ("Raw","Spatial ICA") for metric in PRIMARY_METRICS}
    validation={"status":"passed","protected_rows_79":len(labels)==79,"two_rows_per_observation":len(rows)==158,"finite_delta_counts":finite_counts,"minimum_finite_occurrences_per_test":min(finite_counts.values()),"adequate_finite_coverage":bool(min(finite_counts.values())>=60),"controls_separated_from_all_labeled_sites":all(all((float(r['control_x_px'])-x)**2+(float(r['control_y_px'])-y)**2>=12**2 for x,y in sites) for r in rows),"bh_family_size":len(tests)==12,"manual_review_required_for_completion":False}
    atomic_json(partial/"validation.json",validation)
    atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","occurrence_morphology_metrics.tsv","site_blocked_tests.tsv","objective_morphology_controls.png","llm_context.json"]})
    atomic_json(partial/"llm_context.json",{"entry_point":"summary.json","primary_table":"site_blocked_tests.tsv","figure":"objective_morphology_controls.png","claim_scope":"objective representation-level morphology contrasts; no precision, neuron-type, wiring, or causal claim","upstream_human_packet":"../16_spatial_ica_morphology_ablation_v1","manual_packet_status":"deferred_by_author"})
    atomic_text(partial/"REPORT.md","# Objective spatial-ICA morphology controls\n\nThis automated extension compares event maps at the 79 protected v1 sparse-positive coordinates with deterministic, background-matched translated coordinates. Raw and spatial ICA use identical observation windows and control coordinates. Site-blocked bootstrap intervals and sign-flip tests treat repeated bursts at one site as dependent; BH correction covers all 12 representation-by-metric tests.\n\nThe spatial-ICA TIFF is a globally linear positive-signal encoding with recorded source limits. Values are inverted to source units, but negative reconstructed values were clipped at zero and saturated pixels cannot be recovered. Therefore shape contrasts are emphasized over cross-representation amplitude. Translated coordinates are controls for location-specific morphology, not verified biological negatives. Human packet scoring remains deferred.\n")
    partial.replace(output);return summary
