"""Post-freeze class and certainty extensions for cross-neural networks."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, atomic_text


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def _site_classes(sites: list[dict[str, str]], detections: list[dict[str, str]], radius: float = 6.0) -> dict[tuple[str, int], int]:
    result = {}
    for site in sites:
        sx, sy = float(site["x_px"]), float(site["y_px"])
        for burst in range(1, 5):
            candidates = [d for d in detections if int(d["burst_id"]) == burst]
            ranked = sorted((float(np.hypot(sx-float(d["x_px"]), sy-float(d["y_px"]))), d) for d in candidates)
            if ranked and ranked[0][0] <= radius:
                result[(site["site_id"], burst)] = int(ranked[0][1]["class_id"])
    return result


def _assortativity(rows: list[dict[str, str]], classes: dict[tuple[str, int], int], permutations: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed); output=[]; summary={}
    representations=sorted({r["representation"] for r in rows})
    sites_by_burst={b:sorted({site for site, burst in classes if burst==b}) for b in range(1,5)}
    for representation in representations:
        observations=[]
        for row in rows:
            if row["representation"] != representation: continue
            for burst in range(1,5):
                a=(row["site_a"],burst); b=(row["site_b"],burst)
                if a not in classes or b not in classes: continue
                observations.append((burst,row["site_a"],row["site_b"],abs(float(row[f"burst_{burst}_correlation"])),classes[a],classes[b]))
        same=np.asarray([o[3] for o in observations if o[4]==o[5]]); different=np.asarray([o[3] for o in observations if o[4]!=o[5]])
        observed=float(np.mean(same)-np.mean(different)) if len(same) and len(different) else float("nan")
        exceed=0
        for _ in range(permutations):
            shuffled={}
            for burst, sites in sites_by_burst.items():
                labels=np.asarray([classes[(site,burst)] for site in sites]); rng.shuffle(labels)
                shuffled.update({(site,burst):int(label) for site,label in zip(sites,labels)})
            ps=[o[3] for o in observations if shuffled[(o[1],o[0])]==shuffled[(o[2],o[0])]]
            pd=[o[3] for o in observations if shuffled[(o[1],o[0])]!=shuffled[(o[2],o[0])]]
            value=float(np.mean(ps)-np.mean(pd)) if ps and pd else 0.0
            exceed += abs(value) >= abs(observed)
        p=(exceed+1)/(permutations+1)
        result={"representation":representation,"matched_pair_burst_observations":len(observations),"same_class_n":len(same),"different_class_n":len(different),"same_class_mean_absolute_correlation":float(np.mean(same)),"different_class_mean_absolute_correlation":float(np.mean(different)),"same_minus_different":observed,"burst_preserving_permutation_p_two_sided":p}
        output.append(result); summary[representation]=result
    return output,summary


def run_extension(pairwise: Path, coordinates: Path, detections: Path, v7: Path, output: Path, *, permutations: int=10000) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    partial=output.with_name(output.name+".partial"); partial.mkdir(parents=True,exist_ok=False)
    pairs,sites,detection_rows,v7_rows=_read(pairwise),_read(coordinates),_read(detections),_read(v7)
    classes=_site_classes(sites,detection_rows)
    class_rows,class_summary=_assortativity(pairs,classes,permutations,20260824)
    uncertain={r["original_roi_id"] for r in v7_rows if r["disposition"]=="activity_visible_identity_uncertain"}
    node_rows=[]
    for representation in sorted({r["representation"] for r in pairs}):
        relevant=[r for r in pairs if r["representation"]==representation]
        for site in sorted(s["site_id"] for s in sites):
            incident=[r for r in relevant if site in (r["site_a"],r["site_b"])]
            node_rows.append({"representation":representation,"site_id":site,"certainty_group":"any_identity_uncertain" if site in uncertain else "confirmed_only","stable_degree":sum(r["stable_edge"]=="True" for r in incident),"mean_absolute_correlation":float(np.mean([abs(float(r["median_correlation"])) for r in incident])),"class_matched_bursts":sum((site,b) in classes for b in range(1,5)),"dominant_matched_class":max(set(classes[(site,b)] for b in range(1,5) if (site,b) in classes),key=lambda c:sum(classes.get((site,b))==c for b in range(1,5))) if any((site,b) in classes for b in range(1,5)) else ""})
    _write(partial/"class_assortativity.tsv",class_rows); _write(partial/"node_profiles.tsv",node_rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    labels=[r["representation"].replace("_"," ") for r in class_rows]; x=np.arange(len(labels)); width=.34
    axes[0].bar(x-width/2,[r["same_class_mean_absolute_correlation"] for r in class_rows],width,label="same class",color="#3568a8")
    axes[0].bar(x+width/2,[r["different_class_mean_absolute_correlation"] for r in class_rows],width,label="different class",color="#d08328")
    axes[0].set(xticks=x,xticklabels=labels,ylabel="mean absolute pair correlation",title="Class-matched pair coactivity"); axes[0].tick_params(axis="x",rotation=20); axes[0].legend(); axes[0].grid(axis="y",alpha=.2)
    colors={"confirmed_only":"#3568a8","any_identity_uncertain":"#ad4d75"}
    for group in colors:
        subset=[r for r in node_rows if r["representation"]=="frozen_two_frame_ica" and r["certainty_group"]==group]
        axes[1].scatter([r["stable_degree"] for r in subset],[r["mean_absolute_correlation"] for r in subset],s=55,color=colors[group],label=group.replace("_"," "))
        for r in subset: axes[1].annotate(r["site_id"].replace("roi_",""),(r["stable_degree"],r["mean_absolute_correlation"]),xytext=(3,3),textcoords="offset points",fontsize=7)
    axes[1].set(xlabel="ICA stable degree",ylabel="mean absolute ICA correlation",title="ICA node profiles by v7 certainty history"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.suptitle("Post-freeze class and certainty extensions\nClass assignments were not refit; certainty panel is descriptive (2 uncertain-history sites)")
    fig.savefig(partial/"class_certainty_network_extensions.png",dpi=180); plt.close(fig)
    summary={"schema_version":1,"status":"complete_exploratory","class_matching":{"radius_px":6,"matched_site_burst_records":len(classes),"possible_site_burst_records":len(sites)*4,"frozen_classes_refit":False},"assortativity":class_summary,"certainty":{"uncertain_history_sites":sorted(uncertain & {s['site_id'] for s in sites}),"confirmed_only_sites":len(sites)-len(uncertain & {s['site_id'] for s in sites}),"inferential_test_performed":False,"reason":"only two complete sites have any identity-uncertain v7 occurrence"},"interpretation":"post-freeze within-recording association; candidate-assisted class matches and certainty are not independent validation"}
    atomic_json(partial/"summary.json",summary); atomic_json(partial/"validation.json",{"status":"passed_with_caveats","same_frozen_pair_table_all_representations":True,"class_labels_used_only_post_freeze":True,"certainty_inference_suppressed_for_sparse_group":True,"publication_claim_ready":False})
    atomic_text(partial/"REPORT.md","# Cross-neural class and certainty extensions\n\nFrozen detection classes were matched post hoc within six pixels and tested using burst-preserving class-label permutations. Certainty-linked node profiles are descriptive because only two complete sites have any identity-uncertain v7 occurrence. All four representation networks, including frozen ICA, use the same nodes and pair table.\n")
    atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","artifact_index.json","class_assortativity.tsv","class_certainty_network_extensions.png","node_profiles.tsv","summary.json","validation.json"]})
    partial.replace(output); return summary
