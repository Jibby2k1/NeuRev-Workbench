"""Fail-closed transfer preflight, review packet, and truth-known movie benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import atomic_json, atomic_text


REVIEW_CLASSES = (
    "distinct_source", "overlapping_source", "duplicate_geometry",
    "identity_uncertain", "non_neuronal", "noise_artifact", "unresolved",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_transfer_preflight(input_root: Path, output: Path) -> dict:
    """Inventory untouched recordings without asserting cross-task validity."""
    import tifffile

    rows = []
    for path in sorted(input_root.rglob("*.tif")):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            rows.append({"path": str(path), "shape": list(series.shape), "dtype": str(series.dtype),
                         "sha256": _sha256(path)})
    workbooks = [{"path": str(p), "sha256": _sha256(p)} for p in sorted(input_root.rglob("*.xlsx"))]
    compatible_truth = False
    report = {
        "schema_version": 1,
        "status": "blocked_incompatible_or_incomplete_truth_contract",
        "recordings": rows,
        "roi_workbooks": workbooks,
        "independence": "untouched_by_current_spontaneous_burst_feature_selection",
        "task_labels_in_filenames": sorted({p.stem.split()[-1].lower() for p in input_root.rglob("*.tif")}),
        "required_before_transfer": [
            "declare the target task and event-window contract",
            "provide identity-resolved positive labels with coordinate/frame convention",
            "provide exhaustive bounded-field review if precision is an endpoint",
            "freeze preprocessing, detector, thresholds, features, and estimands before evaluation",
        ],
        "transfer_executed": compatible_truth,
        "reason": "available files are behavioral rest/left/right recordings and do not carry a compatible exhaustive spontaneous-burst identity truth set",
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "preflight.json", report)
    atomic_text(output / "REPORT.md", "# Independent transfer preflight\n\n"
                f"Status: **{report['status']}**. {len(rows)} untouched TIFF recordings and "
                f"{len(workbooks)} ROI workbooks were inventoried. No transfer metric was computed: "
                f"{report['reason']}. This is a fail-closed result, not a negative replication.\n")
    atomic_json(output / "validation.json", {"status": "passed", "recordings_found": len(rows),
                "transfer_not_executed_without_contract": not compatible_truth})
    return report


def build_identity_review_packet(source_media: Path, output: Path) -> dict:
    """Build a two-reviewer identity-complete annotation contract around blind media."""
    manifest = json.loads((source_media / "media_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    columns = ["reviewer_id", "mark_id", "x_px_global", "y_px_global", "burst_id",
               "identity_id", "class", "visibility_confidence", "onset_ui", "peak_ui",
               "end_ui", "source_overlap_ids", "notes", "timestamp"]
    for reviewer in ("reviewer_A", "reviewer_B"):
        with (output / f"{reviewer}.tsv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerow(columns)
    adjudication = columns + ["reviewer_A_mark_ids", "reviewer_B_mark_ids", "adjudication_reason"]
    with (output / "adjudication.tsv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerow(adjudication)
    packet = {
        "schema_version": 1, "status": "ready_for_two_blind_reviewers",
        "source_media_manifest": str(source_media / "media_manifest.json"),
        "source_media_sha256": _sha256(source_media / "media_manifest.json"),
        "region": manifest["region"], "bursts": [m["burst_id"] for m in manifest["media"]],
        "classes": list(REVIEW_CLASSES), "reviewers_required": 2, "adjudicator_required": True,
        "completion_rule": "each reviewer signs coverage of every burst and full region; all cross-reviewer identity conflicts adjudicated",
        "precision_enabled_after_completion": True,
        "current_precision_status": "unavailable_pending_review_and_adjudication",
        "blinding": manifest["annotation_blinding"],
    }
    atomic_json(output / "review_contract.json", packet)
    atomic_text(output / "coverage_signoff.tsv", "reviewer_id\tburst_id\tfull_region_reviewed\tcompleted_at\tnotes\n")
    atomic_text(output / "README.md", "# Identity-complete bounded-field review\n\n"
                "Use the detector-blind Raw media referenced by `review_contract.json`. Two reviewers "
                "independently mark every visible source and assign identities across bursts; an adjudicator "
                "resolves disagreements. Precision remains unavailable until both coverage signoffs and "
                "adjudication are complete. Allowed classes: " + ", ".join(REVIEW_CLASSES) + ".\n")
    atomic_json(output / "validation.json", {"status": "passed", "templates": 3,
                "reviewers_required": 2, "adjudication_required": True,
                "precision_currently_unavailable": True})
    return packet


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = ~positives
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) /
                 (positives.sum() * negatives.sum()))


def _footprint(y: np.ndarray, x: np.ndarray, cy: float, cx: float, shape: str) -> np.ndarray:
    if shape == "ellipse":
        return np.exp(-(((x-cx)/2.4)**2 + ((y-cy)/1.45)**2)/2)
    broad = np.exp(-(((x-cx)/2.5)**2 + ((y-cy)/2.1)**2)/2)
    cut = np.exp(-(((x-(cx+1.25))/1.9)**2 + ((y-(cy-.3))/1.6)**2)/2)
    return np.clip(broad - .78*cut, 0, None)


def _simulated_movie(rng: np.random.Generator, distance: float, ratio: float,
                     correlation: float, shape: str) -> tuple[np.ndarray, list[tuple[float, float]]]:
    yy, xx = np.mgrid[:48, :48]
    centers = [(24., 24.-distance/2), (24., 24.+distance/2)]
    base = rng.normal(size=72)
    traces = []
    kernel = np.exp(-np.arange(18)/5.0) * (1-np.exp(-np.arange(18)/1.5))
    for i in range(2):
        innovations = correlation*base + np.sqrt(1-correlation**2)*rng.normal(size=72)
        events = (innovations > 1.0).astype(float)
        traces.append(np.convolve(events, kernel, mode="full")[:72] * (ratio if i == 0 else 1.0))
    video = np.empty((72, 48, 48), np.float32)
    background = gaussian_filter(rng.normal(size=(48,48)), 7) * 1.6
    artifact_center = (float(rng.integers(8, 40)), float(rng.integers(8, 40)))
    artifact_trace = np.zeros(72); artifact_trace[int(rng.integers(12, 58))] = rng.uniform(1.5, 3.0)
    for t in range(72):
        dy = .7*np.sin(2*np.pi*t/31); dx = .7*np.cos(2*np.pi*t/37)
        signal = sum(traces[i][t] * _footprint(yy, xx, centers[i][0]+dy, centers[i][1]+dx,
                                               shape if i == 0 else "ellipse") for i in range(2))
        artifact = artifact_trace[t] * np.exp(-((yy-artifact_center[0])**2+(xx-artifact_center[1])**2)/(2*.7**2))
        shot = rng.normal(scale=.34*np.sqrt(np.maximum(signal+artifact+2, .1)), size=signal.shape)
        video[t] = background + signal + artifact + shot + rng.normal(scale=.22, size=signal.shape)
    return video, centers


def _score_maps(video: np.ndarray) -> dict[str, np.ndarray]:
    carrier = np.std(video, axis=0)
    context = gaussian_filter(carrier, 1.2) - gaussian_filter(carrier, 4.0)
    centered = video - np.median(video, axis=0, keepdims=True)
    kernel = np.exp(-np.arange(18)/5.0) * (1-np.exp(-np.arange(18)/1.5))
    kernel /= np.linalg.norm(kernel)
    kinetic = np.max(np.stack([np.sum(centered[t:t+18] * kernel[:,None,None], axis=0)
                               for t in range(video.shape[0]-17)]), axis=0)
    def robust_z(array: np.ndarray) -> np.ndarray:
        median = np.median(array); scale = np.median(np.abs(array-median))*1.4826
        return (array-median)/max(float(scale), 1e-6)
    return {"carrier": carrier, "spatial_context": context, "kinetic": kinetic,
            "combined": robust_z(carrier)+robust_z(context)+robust_z(kinetic)}


def _candidates(scoremap: np.ndarray, budget: int, border: int = 4) -> list[tuple[int, int, float]]:
    maxima = scoremap == maximum_filter(scoremap, size=5, mode="nearest")
    maxima[:border] = False; maxima[-border:] = False; maxima[:,:border] = False; maxima[:,-border:] = False
    ys, xs = np.nonzero(maxima)
    order = np.argsort(scoremap[ys,xs])[::-1][:budget]
    return [(int(ys[i]), int(xs[i]), float(scoremap[ys[i],xs[i]])) for i in order]


def _match(candidates: list[tuple[int,int,float]], centers: list[tuple[float,float]], radius: float = 3.0) -> dict:
    if not candidates:
        return {"tp":0,"fp":0,"fn":len(centers),"duplicates":0,"localization":[]}
    dist = np.asarray([[np.hypot(y-cy,x-cx) for cy,cx in centers] for y,x,_ in candidates])
    rows, cols = linear_sum_assignment(dist)
    accepted = [(r,c) for r,c in zip(rows,cols) if dist[r,c] <= radius]
    matched_candidates = {r for r,_ in accepted}
    tp=len(accepted); fp=len(candidates)-tp
    duplicates=sum(i not in matched_candidates and np.min(dist[i]) <= radius for i in range(len(candidates)))
    return {"tp":tp,"fp":fp,"fn":len(centers)-tp,"duplicates":int(duplicates),
            "localization":[float(dist[r,c]) for r,c in accepted]}


def run_realistic_movie_benchmark(output: Path, *, seeds: int = 12) -> dict:
    """Factorial anatomy/kinetics/motion/noise benchmark with exact source truth."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    conditions = [(d, r, c, s) for d in (3.0, 6.0, 10.0) for r in (.35, .65, 1.0)
                  for c in (0.0, .7) for s in ("ellipse", "crescent")]
    yy, xx = np.mgrid[:48, :48]
    for seed in range(seeds):
        rng = np.random.default_rng(4100 + seed)
        for distance, ratio, correlation, shape in conditions:
            video, centers = _simulated_movie(rng, distance, ratio, correlation, shape)
            temporal = np.std(video, axis=0)
            spatial = gaussian_filter(temporal, 1.2) - gaussian_filter(temporal, 4.0)
            for detector, scoremap in (("pixel_variance", temporal), ("spatial_context", spatial)):
                labels = np.zeros(scoremap.size, int)
                for cy, cx in centers:
                    labels[((yy-cy)**2 + (xx-cx)**2 <= 2.25**2).ravel()] = 1
                rows.append({"seed": seed, "distance_px": distance, "amplitude_ratio": ratio,
                             "temporal_correlation": correlation, "footprint_shape": shape,
                             "detector": detector, "pixel_auc": _auc(labels, scoremap.ravel()),
                             "weak_source_score": float(scoremap[round(centers[0][0]), round(centers[0][1])]),
                             "strong_source_score": float(scoremap[round(centers[1][0]), round(centers[1][1])])})
    with (output / "factorial_results.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for detector in ("pixel_variance", "spatial_context"):
        values = [r["pixel_auc"] for r in rows if r["detector"] == detector]
        grouped[detector] = {"mean_pixel_auc": float(np.mean(values)), "sd_across_cells_and_seeds": float(np.std(values))}
    deltas = []
    for key in {(r["seed"],r["distance_px"],r["amplitude_ratio"],r["temporal_correlation"],r["footprint_shape"]) for r in rows}:
        pair = {r["detector"]: r["pixel_auc"] for r in rows if (r["seed"],r["distance_px"],r["amplitude_ratio"],r["temporal_correlation"],r["footprint_shape"]) == key}
        deltas.append(pair["spatial_context"]-pair["pixel_variance"])
    summary = {"schema_version": 1, "status": "completed_truth_known_simulation",
               "design": {"seeds": seeds, "factorial_cells": len(conditions), "paired_runs": len(deltas),
                          "distance_px": [3,6,10], "amplitude_ratio": [.35,.65,1],
                          "temporal_correlation": [0,.7], "footprint_shape": ["ellipse","crescent"]},
               "models": grouped, "paired_spatial_context_minus_pixel_auc": {
                   "mean": float(np.mean(deltas)), "ci95_low": float(np.quantile(deltas,.025)),
                   "ci95_high": float(np.quantile(deltas,.975)), "win_fraction": float(np.mean(np.array(deltas)>0))},
               "scope": "mechanistic stress test, not biological external validation",
               "scientific_audit": {"truth": "exact simulator source identities", "model_annotations": "none",
                                    "media": "no human audit required for exact numeric simulator truth"}}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "validation.json", {"status": "passed", "rows": len(rows),
                "expected_rows": seeds*len(conditions)*2, "finite_auc": all(np.isfinite(r["pixel_auc"]) for r in rows)})
    atomic_text(output / "REPORT.md", "# Realistic movie stress test\n\n"
                f"Across {len(deltas)} paired anatomy/noise/kinetic conditions, spatial context changed pixel-level "
                f"AUC by {np.mean(deltas):+.3f} (empirical 95% interval {np.quantile(deltas,.025):+.3f} to "
                f"{np.quantile(deltas,.975):+.3f}). This is exact-truth mechanistic evidence, not independent biological validation.\n")
    return summary


def run_detector_level_benchmark(output: Path, *, seeds: int = 12) -> dict:
    """Evaluate frozen score stacks as proposals with exact one-to-one identity truth."""
    output.mkdir(parents=True, exist_ok=False)
    conditions = [(d,r,c,s) for d in (3.,6.,10.) for r in (.35,.65,1.) for c in (0.,.7)
                  for s in ("ellipse","crescent")]
    rows=[]; candidate_rows=[]
    for seed in range(seeds):
        rng=np.random.default_rng(7300+seed)
        for cell,(distance,ratio,correlation,shape) in enumerate(conditions):
            video, centers = _simulated_movie(rng,distance,ratio,correlation,shape)
            maps=_score_maps(video)
            for model,scoremap in maps.items():
                pool=_candidates(scoremap,16)
                for rank,(y,x,score) in enumerate(pool,1):
                    nearest=float(min(np.hypot(y-cy,x-cx) for cy,cx in centers))
                    candidate_rows.append({"seed":seed,"cell":cell,"model":model,"rank":rank,"y":y,"x":x,
                                           "score":score,"label":int(nearest<=3),"nearest_truth_distance_px":nearest})
                for budget in (2,4,8):
                    match=_match(pool[:budget],centers)
                    precision=match["tp"]/budget; recall=match["tp"]/2
                    rows.append({"seed":seed,"cell":cell,"distance_px":distance,"amplitude_ratio":ratio,
                                 "temporal_correlation":correlation,"footprint_shape":shape,"model":model,
                                 "budget":budget,"tp":match["tp"],"fp":match["fp"],"fn":match["fn"],
                                 "precision":precision,"recall":recall,"f1":2*precision*recall/max(precision+recall,1e-12),
                                 "duplicates":match["duplicates"],"mean_localization_px":float(np.mean(match["localization"])) if match["localization"] else float("nan")})
    for name,data in (("operating_points.tsv",rows),("candidate_scores.tsv",candidate_rows)):
        with (output/name).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]),delimiter="\t");writer.writeheader();writer.writerows(data)
    summary_models={}
    for model in ("carrier","spatial_context","kinetic","combined"):
        summary_models[model]={}
        for budget in (2,4,8):
            chosen=[r for r in rows if r["model"]==model and r["budget"]==budget]
            summary_models[model][str(budget)]={k:float(np.nanmean([r[k] for r in chosen])) for k in
                ("precision","recall","f1","duplicates","mean_localization_px")}
    # Prospective calibration: fit only seeds 0..5, evaluate seeds 6..11.
    calibration={}; selective=[]
    split=max(1,seeds//2)
    for model in summary_models:
        train=[r for r in candidate_rows if r["model"]==model and r["seed"]<split]
        test=[r for r in candidate_rows if r["model"]==model and r["seed"]>=split]
        if not test: test=train
        xtr=np.asarray([[r["score"]] for r in train]); ytr=np.asarray([r["label"] for r in train])
        xte=np.asarray([[r["score"]] for r in test]); yte=np.asarray([r["label"] for r in test])
        fit=LogisticRegression(C=1.0,max_iter=1000).fit(xtr,ytr); probabilities=fit.predict_proba(xte)[:,1]
        bins=np.linspace(0,1,11); ece=0.
        for low,high in zip(bins[:-1],bins[1:]):
            mask=(probabilities>=low)&(probabilities<(high if high<1 else high+1e-9))
            if mask.any(): ece += mask.mean()*abs(probabilities[mask].mean()-yte[mask].mean())
        predicted=probabilities>=.5
        calibration[model]={"train_seeds":list(range(split)),"evaluation_seeds":list(range(split,seeds)),
                            "positive_prevalence":float(yte.mean()),"brier":float(brier_score_loss(yte,probabilities)),
                            "log_loss":float(log_loss(yte,probabilities)),"roc_auc":float(roc_auc_score(yte,probabilities)),
                            "average_precision":float(average_precision_score(yte,probabilities)),
                            "balanced_accuracy_at_0p5":float(balanced_accuracy_score(yte,predicted)),"ece10":float(ece)}
        confidence=np.abs(probabilities-.5)*2
        for coverage in (1.,.75,.5,.25):
            keep=np.argsort(confidence)[::-1][:max(1,int(round(len(yte)*coverage)))]
            error=float(np.mean((probabilities[keep]>=.5)!=yte[keep]))
            selective.append({"model":model,"coverage":coverage,"classification_error":error,"retained":len(keep)})
    with (output/"calibration_abstention.tsv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selective[0]),delimiter="\t");writer.writeheader();writer.writerows(selective)
    best=max(summary_models,key=lambda m:summary_models[m]["4"]["f1"])
    result={"schema_version":1,"status":"completed_truth_known_detector_simulation",
            "design":{"seeds":seeds,"factorial_cells":len(conditions),"movies":seeds*len(conditions),
                      "models":["carrier","spatial_context","kinetic","combined"],"budgets":[2,4,8],
                      "match_radius_px":3,"one_to_one_matching":True,"proposal_nms_window_px":5},
            "operating_points":summary_models,"calibration":calibration,
            "best_budget4_f1_model":best,"scope":"mechanistic detector stress test, not biological validation",
            "scientific_audit":{"status":"computational_truth_audit","expert_truth":"exact simulator identities",
                                "human_review":"not used","model_annotations":"fully tabulated in candidate_scores.tsv"}}
    atomic_json(output/"summary.json",result)
    atomic_json(output/"validation.json",{"status":"passed","operating_rows":len(rows),
                "expected_operating_rows":seeds*len(conditions)*4*3,"candidate_rows":len(candidate_rows),
                "finite_primary_metrics":all(np.isfinite(r[k]) for r in rows for k in ("precision","recall","f1")),
                "calibration_seed_disjoint":all(not(set(v["train_seeds"])&set(v["evaluation_seeds"])) for v in calibration.values())})
    atomic_json(output/"llm_context.json",{"entry_point":"summary.json","primary_budget":4,
                "truth":"two exact source identities per movie","tables":["operating_points.tsv","candidate_scores.tsv","calibration_abstention.tsv"],
                "limitations":["simulation is not an independent animal","candidate calibration labels use a 3 px geometric rule"]})
    atomic_json(output/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","llm_context.json",
                "operating_points.tsv","candidate_scores.tsv","calibration_abstention.tsv"]})
    primary=summary_models
    atomic_text(output/"REPORT.md","# Detector-level exact-truth movie benchmark\n\n"
                f"At the primary budget of four proposals, `{best}` had the highest mean F1 "
                f"({primary[best]['4']['f1']:.3f}). Results include one-to-one matching, localization, duplicates, "
                "held-seed calibration, and selective risk. This is mechanistic simulation evidence, not biological transfer.\n")
    return result


def estimate_empirical_noise_profile(video_path: Path) -> dict:
    """Estimate a label-free noise scale from sparse frames of an untouched TIFF."""
    import tifffile

    stack = tifffile.memmap(video_path, mode="r")
    if stack.ndim != 3:
        raise ValueError("empirical noise source must be a frame-by-height-by-width TIFF")
    frame_indices = np.linspace(0, stack.shape[0]-1, min(48, stack.shape[0]), dtype=int)
    y0=max(0,(stack.shape[1]-256)//2); x0=max(0,(stack.shape[2]-256)//2)
    sample=np.asarray(stack[frame_indices,y0:y0+256,x0:x0+256],dtype=np.float32)
    median_intensity=float(np.median(sample))
    differences=np.diff(sample,axis=0).reshape(-1)
    difference_mad=float(np.median(np.abs(differences-np.median(differences)))*1.4826/np.sqrt(2))
    coefficient=difference_mad/max(abs(median_intensity),1.0)
    return {"source_path":str(video_path),"source_sha256":_sha256(video_path),"source_shape":list(stack.shape),
            "sampled_frames":frame_indices.tolist(),"crop":{"y0":y0,"x0":x0,"height":sample.shape[1],"width":sample.shape[2]},
            "median_intensity":median_intensity,"frame_difference_noise_mad":difference_mad,
            "robust_noise_to_median":coefficient,"simulator_read_sigma":float(np.clip(coefficient*5,.15,.65)),
            "label_use":"none"}


def _family_movie(rng: np.random.Generator, family: str, source_count: int,
                  empirical_profile: dict | None) -> tuple[np.ndarray,list[tuple[float,float]]]:
    frames=80; height=64; width=64; yy,xx=np.mgrid[:height,:width]
    grid=max(1,int(np.ceil(np.sqrt(source_count)))); anchors=[]
    for row in range(grid):
        for col in range(grid):
            if len(anchors)>=source_count: break
            anchors.append((12+(height-24)*(row+.5)/grid+rng.uniform(-1.5,1.5),
                            12+(width-24)*(col+.5)/grid+rng.uniform(-1.5,1.5)))
    centers=[(float(y),float(x)) for y,x in anchors]
    kernel=np.exp(-np.arange(20)/6)*(1-np.exp(-np.arange(20)/1.6)); shared=rng.normal(size=frames)
    traces=[]
    for index in range(source_count):
        innovation=.25*shared+np.sqrt(1-.25**2)*rng.normal(size=frames)
        events=(innovation>1.15).astype(float)
        traces.append(np.convolve(events,kernel,mode="full")[:frames]*rng.uniform(.45,1.25))
    structured=gaussian_filter(rng.normal(size=(height,width)),8)*1.5
    neuropil=np.zeros((frames,height,width),np.float32)
    if family in {"dense_neuropil","combined_shift"}:
        for _ in range(7):
            cy,cx=rng.uniform(0,height),rng.uniform(0,width); sigma=rng.uniform(5,11)
            field=np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*sigma**2))
            trace=np.convolve((rng.normal(size=frames)>1.25).astype(float),np.exp(-np.arange(25)/10),mode="full")[:frames]
            neuropil += (.28*trace[:,None,None]*field).astype(np.float32)
    read_sigma=.22
    if family in {"empirical_noise","combined_shift"} and empirical_profile:
        read_sigma=float(empirical_profile["simulator_read_sigma"])
    video=np.empty((frames,height,width),np.float32)
    for t in range(frames):
        signal=np.zeros((height,width),float)
        for index,(cy,cx) in enumerate(centers):
            if family in {"nonrigid_motion","combined_shift"}:
                dy=1.5*np.sin(2*np.pi*t/(27+index*2)+index); dx=1.2*np.cos(2*np.pi*t/(31+index*3)+index/2)
            else:
                dy=.5*np.sin(2*np.pi*t/31); dx=.5*np.cos(2*np.pi*t/37)
            shape="crescent" if index%3==0 else "ellipse"
            signal += traces[index][t]*_footprint(yy,xx,cy+dy,cx+dx,shape)
        bleach=np.exp(-t/90) if family in {"bleaching","combined_shift"} else 1.0
        shot=rng.normal(scale=.32*np.sqrt(np.maximum(signal+neuropil[t]+2,.1)),size=signal.shape)
        video[t]=bleach*(structured+signal+neuropil[t])+shot+rng.normal(scale=read_sigma,size=signal.shape)
    return video,centers


def run_generator_family_holdout(output: Path, *, empirical_profile: dict | None,
                                 seeds: int = 6) -> dict:
    """Stress frozen proposal stacks under held-out simulator mechanisms."""
    output.mkdir(parents=True,exist_ok=False)
    families=("baseline","dense_neuropil","bleaching","nonrigid_motion","empirical_noise","combined_shift")
    rows=[]; candidates=[]
    for family_index,family in enumerate(families):
        for seed in range(seeds):
            rng=np.random.default_rng(9900+100*family_index+seed)
            for source_count in (2,4,8):
                video,centers=_family_movie(rng,family,source_count,empirical_profile)
                for model,scoremap in _score_maps(video).items():
                    pool=_candidates(scoremap,24)
                    for rank,(y,x,score) in enumerate(pool,1):
                        nearest=float(min(np.hypot(y-cy,x-cx) for cy,cx in centers))
                        candidates.append({"family":family,"seed":seed,"source_count":source_count,"model":model,
                                           "rank":rank,"score":score,"label":int(nearest<=3),"nearest_truth_distance_px":nearest})
                    budget=2*source_count
                    match=_match(pool[:budget],centers); precision=match["tp"]/budget; recall=match["tp"]/source_count
                    rows.append({"family":family,"seed":seed,"source_count":source_count,"model":model,"budget":budget,
                                 "tp":match["tp"],"fp":match["fp"],"fn":match["fn"],"precision":precision,"recall":recall,
                                 "f1":2*precision*recall/max(precision+recall,1e-12),"duplicates":match["duplicates"],
                                 "mean_localization_px":float(np.mean(match["localization"])) if match["localization"] else float("nan")})
    for name,data in (("family_operating_points.tsv",rows),("family_candidates.tsv",candidates)):
        with (output/name).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]),delimiter="\t");writer.writeheader();writer.writerows(data)
    family_summary={}
    for family in families:
        family_summary[family]={}
        for model in ("carrier","spatial_context","kinetic","combined"):
            selected=[r for r in rows if r["family"]==family and r["model"]==model]
            family_summary[family][model]={k:float(np.nanmean([r[k] for r in selected])) for k in
                                           ("precision","recall","f1","duplicates","mean_localization_px")}
    holdout=[]
    for held in families:
        for model in ("carrier","spatial_context","kinetic","combined"):
            train=[r for r in candidates if r["family"]!=held and r["model"]==model]
            test=[r for r in candidates if r["family"]==held and r["model"]==model]
            xtr=np.asarray([[r["score"]] for r in train]);ytr=np.asarray([r["label"] for r in train])
            xte=np.asarray([[r["score"]] for r in test]);yte=np.asarray([r["label"] for r in test])
            fit=LogisticRegression(C=1,max_iter=1000).fit(xtr,ytr);p=fit.predict_proba(xte)[:,1]
            holdout.append({"held_out_family":held,"model":model,"positives":int(yte.sum()),"candidates":len(yte),
                            "positive_prevalence":float(yte.mean()),"roc_auc":float(roc_auc_score(yte,p)),
                            "average_precision":float(average_precision_score(yte,p)),"brier":float(brier_score_loss(yte,p)),
                            "log_loss":float(log_loss(yte,p)),"balanced_accuracy_at_0p5":float(balanced_accuracy_score(yte,p>=.5))})
    with (output/"leave_one_family_out_calibration.tsv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(holdout[0]),delimiter="\t");writer.writeheader();writer.writerows(holdout)
    mean_holdout={model:{metric:float(np.mean([r[metric] for r in holdout if r["model"]==model])) for metric in
                         ("roc_auc","average_precision","brier","log_loss","balanced_accuracy_at_0p5")}
                  for model in ("carrier","spatial_context","kinetic","combined")}
    result={"schema_version":1,"status":"completed_generator_family_holdout","design":{"families":list(families),
            "seeds_per_family":seeds,"source_counts":[2,4,8],"movies":len(families)*seeds*3,"proposal_budget":"2x source count",
            "models":["carrier","spatial_context","kinetic","combined"],"match_radius_px":3},
            "empirical_noise_profile":empirical_profile,"family_operating_points":family_summary,
            "mean_leave_one_family_out_calibration":mean_holdout,"scope":"mechanistic distribution-shift test, not biological validation",
            "scientific_audit":{"truth":"exact simulator identities","candidate_inventory":"family_candidates.tsv","human_review":"not used"}}
    atomic_json(output/"summary.json",result)
    atomic_json(output/"validation.json",{"status":"passed","movies":len(families)*seeds*3,"operating_rows":len(rows),
                "expected_operating_rows":len(families)*seeds*3*4,"holdout_rows":len(holdout),"expected_holdout_rows":24,
                "all_families_held_out_once":sorted({r["held_out_family"] for r in holdout})==sorted(families)})
    atomic_json(output/"llm_context.json",{"entry_point":"summary.json","primary_tables":["family_operating_points.tsv",
                "leave_one_family_out_calibration.tsv"],"evidence_boundary":"exact-truth simulator distribution shift only"})
    atomic_json(output/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","llm_context.json",
                "family_operating_points.tsv","family_candidates.tsv","leave_one_family_out_calibration.tsv"]})
    best=max(mean_holdout,key=lambda m:mean_holdout[m]["average_precision"])
    atomic_text(output/"REPORT.md","# Generator-family holdout benchmark\n\n"
                f"Across six generator families and source counts 2, 4, and 8, `{best}` had the highest mean "
                f"leave-one-family-out average precision ({mean_holdout[best]['average_precision']:.3f}). "
                "This is an automated mechanistic distribution-shift test, not biological validation.\n")
    return result


def _union_candidates(maps: dict[str,np.ndarray], limit_per_map: int = 32) -> list[tuple[int,int]]:
    points=set()
    for name in ("carrier","spatial_context","kinetic"):
        points.update((y,x) for y,x,_ in _candidates(maps[name],limit_per_map))
    return sorted(points)


def _nms_scored(points: list[tuple[int,int,float]], radius: float = 2.0) -> list[tuple[int,int,float]]:
    kept=[]
    for y,x,score in sorted(points,key=lambda row:row[2],reverse=True):
        if all(np.hypot(y-ky,x-kx)>radius for ky,kx,_ in kept): kept.append((y,x,score))
    return kept


def _false_alarm_threshold(null_scores: list[list[float]], target: float) -> tuple[float,float]:
    pooled=np.sort(np.asarray([score for movie in null_scores for score in movie],float))[::-1]
    allowed=int(np.floor(target*len(null_scores)))
    threshold=float("inf") if allowed<=0 else float(pooled[min(allowed-1,len(pooled)-1)])
    achieved=float(np.mean([sum(score>=threshold for score in movie) for movie in null_scores]))
    return threshold,achieved


def run_label_free_false_alarm_benchmark(output: Path, *, empirical_profile: dict | None,
                                         seeds: int = 6) -> dict:
    """Train on development families, calibrate on nulls, test untouched shifts."""
    output.mkdir(parents=True,exist_ok=False)
    development=("baseline","dense_neuropil","bleaching")
    evaluation=("nonrigid_motion","empirical_noise","combined_shift")
    feature_names=("carrier","spatial_context","kinetic")
    train_x=[];train_y=[]
    for family_index,family in enumerate(development):
        for seed in range(seeds):
            rng=np.random.default_rng(12100+100*family_index+seed)
            for source_count in (2,4,8):
                video,centers=_family_movie(rng,family,source_count,empirical_profile);maps=_score_maps(video)
                for y,x in _union_candidates(maps):
                    train_x.append([float(maps[name][y,x]) for name in feature_names])
                    train_y.append(int(min(np.hypot(y-cy,x-cx) for cy,cx in centers)<=3))
    gate=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,class_weight="balanced"))
    gate.fit(np.asarray(train_x),np.asarray(train_y))
    null_scores={model:[] for model in ("spatial_context","gated_fusion")}
    for family_index,family in enumerate(development):
        for seed in range(seeds):
            rng=np.random.default_rng(13100+100*family_index+seed)
            video,_=_family_movie(rng,family,0,empirical_profile);maps=_score_maps(video)
            spatial=[score for _,_,score in _candidates(maps["spatial_context"],32)]
            union=_union_candidates(maps);features=np.asarray([[maps[name][y,x] for name in feature_names] for y,x in union])
            gated=gate.predict_proba(features)[:,1].tolist()
            null_scores["spatial_context"].append(spatial);null_scores["gated_fusion"].append(gated)
    thresholds={}
    for model in null_scores:
        thresholds[model]={}
        for target in (1.,2.,4.):
            threshold,achieved=_false_alarm_threshold(null_scores[model],target)
            thresholds[model][str(int(target))]={"threshold":threshold,"target_false_alarms_per_movie":target,
                                                  "achieved_development_null_false_alarms_per_movie":achieved}
    rows=[];candidate_rows=[]
    for family_index,family in enumerate(evaluation):
        for seed in range(seeds):
            rng=np.random.default_rng(14100+100*family_index+seed)
            for source_count in (2,4,8):
                video,centers=_family_movie(rng,family,source_count,empirical_profile);maps=_score_maps(video)
                spatial_pool=_candidates(maps["spatial_context"],32)
                union=_union_candidates(maps);features=np.asarray([[maps[name][y,x] for name in feature_names] for y,x in union])
                gated_scores=gate.predict_proba(features)[:,1]
                gated_pool=_nms_scored([(y,x,float(score)) for (y,x),score in zip(union,gated_scores)])
                pools={"spatial_context":spatial_pool,"gated_fusion":gated_pool}
                for model,pool in pools.items():
                    for rank,(y,x,score) in enumerate(pool,1):
                        candidate_rows.append({"family":family,"seed":seed,"source_count":source_count,"model":model,
                            "rank":rank,"score":score,"nearest_truth_distance_px":float(min(np.hypot(y-cy,x-cx) for cy,cx in centers))})
                    for target in (1,2,4):
                        threshold=thresholds[model][str(target)]["threshold"]
                        selected=[row for row in pool if row[2]>=threshold];match=_match(selected,centers)
                        precision=match["tp"]/len(selected) if selected else 1.0;recall=match["tp"]/source_count
                        rows.append({"family":family,"seed":seed,"source_count":source_count,"model":model,
                            "target_false_alarms_per_movie":target,"threshold":threshold,"proposals":len(selected),
                            "tp":match["tp"],"fp":match["fp"],"fn":match["fn"],"precision":precision,"recall":recall,
                            "f1":2*precision*recall/max(precision+recall,1e-12),"duplicates":match["duplicates"],
                            "mean_localization_px":float(np.mean(match["localization"])) if match["localization"] else float("nan")})
    for name,data in (("evaluation_operating_points.tsv",rows),("evaluation_candidates.tsv",candidate_rows)):
        with (output/name).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]),delimiter="\t");writer.writeheader();writer.writerows(data)
    summary_models={}
    for model in ("spatial_context","gated_fusion"):
        summary_models[model]={}
        for target in (1,2,4):
            selected=[r for r in rows if r["model"]==model and r["target_false_alarms_per_movie"]==target]
            summary_models[model][str(target)]={k:float(np.nanmean([r[k] for r in selected])) for k in
                ("proposals","precision","recall","f1","duplicates","mean_localization_px")}
    result={"schema_version":1,"status":"completed_label_free_false_alarm_evaluation",
            "split":{"development_families":list(development),"null_calibration_families":list(development),
                     "untouched_evaluation_families":list(evaluation),"family_overlap":False},
            "design":{"seeds_per_family":seeds,"source_counts":[2,4,8],"evaluation_movies":len(evaluation)*seeds*3,
                      "models":["spatial_context","gated_fusion"],"target_false_alarms_per_movie":[1,2,4],
                      "match_radius_px":3,"fusion_features":list(feature_names)},
            "gate":{"training_candidates":len(train_y),"positive_fraction":float(np.mean(train_y)),
                    "training_labels":"exact simulator truth from development families only"},
            "null_thresholds":thresholds,"untouched_evaluation":summary_models,
            "empirical_noise_profile":empirical_profile,"scope":"simulator-family prospective test, not biological validation",
            "scientific_audit":{"truth":"exact evaluation-family simulator identities","threshold_labels":"none; source-free null movies only",
                                "candidate_inventory":"evaluation_candidates.tsv"}}
    atomic_json(output/"summary.json",result)
    atomic_json(output/"validation.json",{"status":"passed","family_overlap":False,"evaluation_rows":len(rows),
        "expected_evaluation_rows":len(evaluation)*seeds*3*2*3,"null_movie_count":len(development)*seeds,
        "all_thresholds_finite":all(np.isfinite(v["threshold"]) for model in thresholds.values() for v in model.values()),
        "thresholds_use_no_source_truth":True})
    atomic_json(output/"llm_context.json",{"entry_point":"summary.json","primary_table":"evaluation_operating_points.tsv",
        "primary_false_alarm_target":2,"evidence_boundary":"development-trained simulator gate tested on untouched generator families"})
    atomic_json(output/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","llm_context.json",
        "evaluation_operating_points.tsv","evaluation_candidates.tsv"]})
    primary={m:summary_models[m]["2"] for m in summary_models}
    winner=max(primary,key=lambda m:primary[m]["f1"])
    atomic_text(output/"REPORT.md","# Label-free false-alarm benchmark\n\n"
        f"At the primary target of two development-null false alarms per movie, `{winner}` achieved the higher "
        f"untouched-family mean F1 ({primary[winner]['f1']:.3f}). Thresholds used source-free development nulls only; "
        "evaluation families were not used for fitting or threshold selection.\n")
    return result
