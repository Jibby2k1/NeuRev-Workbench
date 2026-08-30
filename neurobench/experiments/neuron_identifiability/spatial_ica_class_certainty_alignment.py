"""Post-freeze objective morphology alignment with classes and certainty."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal
import tifffile

from .contracts import atomic_json, atomic_text
from .detection_class_extensions import cramers_v, greedy_spatial_match
from .spatial_ica_objective_morphology import PRIMARY_METRICS, _bh, _control_coordinate, _map_and_metrics


ALL_METRICS=(*PRIMARY_METRICS,"connected_components")
COLORS={1:"#3976AF",2:"#D87921",3:"#7D9235"}


def _read(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def _write(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)


def _permutation_kruskal(values:np.ndarray,classes:np.ndarray,repeats:int=20000,seed:int=20260824)->dict[str,float]:
    active=sorted(set(int(x) for x in classes));groups=[values[classes==c] for c in active]
    observed=float(kruskal(*groups).statistic);n=len(values);k=len(groups);epsilon=max(0.0,(observed-k+1)/max(n-k,1));rng=np.random.default_rng(seed);exceed=0
    for _ in range(repeats):
        shuffled=rng.permutation(classes);stat=float(kruskal(*[values[shuffled==c] for c in active]).statistic);exceed+=stat>=observed-1e-12
    return {"kruskal_h":observed,"epsilon_squared":epsilon,"site_label_permutation_p":float((exceed+1)/(repeats+1))}


def _class_site_rows(matches:list[tuple[dict[str,str],dict[str,str],float]],morphology:list[dict[str,str]])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    morph={(r["observation_id"],r["representation"]):r for r in morphology};enriched=[]
    for detection,label,distance in matches:
        for representation in ("Raw","Spatial ICA"):
            row=morph[(label["observation_id"],representation)]
            enriched.append({"observation_id":label["observation_id"],"observation_site_id":label["observation_site_id"],"detection_occurrence_id":detection["detection_occurrence_id"],"detection_site_id":detection["detection_site_id"],"burst_id":int(label["burst_id"]),"class_id":int(detection["class_id"]),"representation":representation,"match_distance_px":distance,**{f"delta_{m}":float(row[f"delta_{m}"]) for m in ALL_METRICS}})
    by_site=defaultdict(list)
    for row in enriched:by_site[(row["observation_site_id"],row["representation"])].append(row)
    sites=[]
    for (site,representation),rows in sorted(by_site.items()):
        counts=Counter(int(r["class_id"]) for r in rows);dominant=min((-count,c) for c,count in counts.items())[1]
        sites.append({"observation_site_id":site,"representation":representation,"dominant_class_id":dominant,"matched_occurrences":len(rows),"class_sequence":";".join(str(r["class_id"]) for r in sorted(rows,key=lambda x:x["burst_id"])),**{f"mean_delta_{m}":float(np.nanmean([r[f'delta_{m}'] for r in rows])) for m in ALL_METRICS}})
    return enriched,sites


def _class_tests(sites:list[dict[str,Any]])->list[dict[str,Any]]:
    tests=[]
    for representation in ("Raw","Spatial ICA"):
        rows=[r for r in sites if r["representation"]==representation];classes=np.asarray([r["dominant_class_id"] for r in rows])
        for metric in ALL_METRICS:
            values=np.asarray([r[f"mean_delta_{metric}"] for r in rows],float);finite=np.isfinite(values);result=_permutation_kruskal(values[finite],classes[finite])
            tests.append({"representation":representation,"metric":metric,"site_count":int(finite.sum()),"class_1_sites":int(np.sum(classes[finite]==1)),"class_2_sites":int(np.sum(classes[finite]==2)),"class_3_sites":int(np.sum(classes[finite]==3)),**result,**{f"class_{c}_median":float(np.median(values[finite][classes[finite]==c])) for c in (1,2,3)}})
    for row,q in zip(tests,_bh([r["site_label_permutation_p"] for r in tests])):row["bh_q_across_14_class_tests"]=q
    return tests


def _extract_v7_metrics(matches:list[dict[str,str]],v7:list[dict[str,str]],raw:np.ndarray,ica:np.ndarray)->list[dict[str,Any]]:
    labels={r["observation_id"]:r for r in v7};all_sites=[(float(r["x_px"]),float(r["y_px"])) for r in v7];rows=[]
    for match in matches:
        if match["certainty_group"]=="artifact":continue
        label=labels[match["observation_id"]];start=int(label["original_start_frame_ui"])-1;stop=int(label["original_end_frame_ui"]);ica_start=start-1799;ica_stop=stop-1799
        raw_base=np.median(np.asarray(raw[max(0,start-20):start],float),axis=0);cx,cy=_control_coordinate({**label,"observation_id":label["observation_id"]},raw.shape[1:],all_sites,raw_base)
        for representation,video,a,b in (("Raw",raw,start,stop),("Spatial ICA",ica,ica_start,ica_stop)):
            observed=_map_and_metrics(video,a,b,float(label["x_px"]),float(label["y_px"]));control=_map_and_metrics(video,a,b,cx,cy)
            rows.append({"observation_id":label["observation_id"],"canonical_roi_id":match["canonical_roi_id"],"detection_occurrence_id":match["detection_occurrence_id"],"burst_id":int(match["burst_id"]),"class_id":int(match["class_id"]),"certainty_group":match["certainty_group"],"representation":representation,"match_distance_px":float(match["match_distance_px"]),**{f"delta_{m}":observed[m]-control[m] for m in ALL_METRICS}})
    return rows


def _adjusted_certainty_tests(rows:list[dict[str,Any]],repeats:int=20000,seed:int=20260824)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    rng=np.random.default_rng(seed);tests=[];canonical_rows=[]
    for representation in ("Raw","Spatial ICA"):
        subset=[r for r in rows if r["representation"]==representation]
        for metric in ALL_METRICS:
            selected=[r for r in subset if np.isfinite(float(r[f"delta_{metric}"]))];y=np.asarray([r[f"delta_{metric}"] for r in selected],float);classes=np.asarray([r["class_id"] for r in selected]);bursts=np.asarray([r["burst_id"] for r in selected])
            design=np.column_stack((np.ones(len(y)),classes==2,classes==3,bursts==2,bursts==3,bursts==4));residual=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
            grouped=defaultdict(list);certainty={}
            for row,value in zip(selected,residual,strict=True):grouped[row["canonical_roi_id"]].append(float(value));certainty[row["canonical_roi_id"]]=row["certainty_group"]
            ids=sorted(grouped);values=np.asarray([np.mean(grouped[i]) for i in ids]);groups=np.asarray([certainty[i] for i in ids]);confirmed=values[groups=="confirmed"];uncertain=values[groups=="identity uncertain"];difference=float(np.mean(uncertain)-np.mean(confirmed));exceed=0
            for _ in range(repeats):
                shuffled=rng.permutation(groups);perm=float(np.mean(values[shuffled=="identity uncertain"])-np.mean(values[shuffled=="confirmed"]));exceed+=abs(perm)>=abs(difference)-1e-12
            boot=[]
            for _ in range(10000):boot.append(float(np.mean(rng.choice(uncertain,len(uncertain),replace=True))-np.mean(rng.choice(confirmed,len(confirmed),replace=True))))
            tests.append({"representation":representation,"metric":metric,"canonical_identities":len(ids),"confirmed_identities":len(confirmed),"uncertain_identities":len(uncertain),"adjusted_uncertain_minus_confirmed":difference,"bootstrap_ci95_low":float(np.quantile(boot,.025)),"bootstrap_ci95_high":float(np.quantile(boot,.975)),"canonical_label_permutation_p":float((exceed+1)/(repeats+1))})
            canonical_rows.extend({"representation":representation,"metric":metric,"canonical_roi_id":i,"certainty_group":certainty[i],"class_and_burst_adjusted_residual":float(values[j])} for j,i in enumerate(ids))
    for row,q in zip(tests,_bh([r["canonical_label_permutation_p"] for r in tests])):row["bh_q_across_14_certainty_tests"]=q
    return tests,canonical_rows


def _certainty_class_summary(rows:list[dict[str,Any]],repeats:int=20000,seed:int=20260824)->dict[str,Any]:
    unique={}
    for row in rows:
        key=row["canonical_roi_id"];unique.setdefault(key,[]).append(row)
    identities=[]
    for key,values in unique.items():
        counts=Counter(v["class_id"] for v in values if v["representation"]=="Raw");dominant=min((-n,c) for c,n in counts.items())[1];identities.append((values[0]["certainty_group"],dominant))
    groups=("confirmed","identity uncertain");table=np.asarray([[sum(g==group and c==cls for g,c in identities) for cls in (1,2,3)] for group in groups]);observed=cramers_v(table);labels=np.asarray([g for g,_ in identities]);classes=np.asarray([c for _,c in identities]);rng=np.random.default_rng(seed);exceed=0
    for _ in range(repeats):
        shuffled=rng.permutation(labels);permuted=np.asarray([[np.sum((shuffled==g)&(classes==c)) for c in (1,2,3)] for g in groups]);exceed+=cramers_v(permuted)>=observed-1e-12
    return {"grain":"24 canonical-v7 identities; dominant frozen class","groups":list(groups),"classes":[1,2,3],"table":table.tolist(),"cramers_v":observed,"canonical_label_permutation_p":float((exceed+1)/(repeats+1))}


def _v7_class_sites(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    grouped=defaultdict(list)
    for row in rows:grouped[(row["canonical_roi_id"],row["representation"])].append(row)
    output=[]
    for (canonical,representation),values in sorted(grouped.items()):
        counts=Counter(int(r["class_id"]) for r in values);dominant=min((-n,c) for c,n in counts.items())[1]
        output.append({"observation_site_id":canonical,"representation":representation,"dominant_class_id":dominant,"matched_occurrences":len(values),"class_sequence":";".join(str(r["class_id"]) for r in sorted(values,key=lambda x:x["burst_id"])),**{f"mean_delta_{m}":float(np.nanmean([r[f'delta_{m}'] for r in values])) for m in ALL_METRICS}})
    return output


def _v7_figure(tests:list[dict[str,Any]],path:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained");metrics=["contrast","sharpness","compactness","radius","peak offset","persistence","components"]
    for ax,representation in zip(axes,("Raw","Spatial ICA")):
        subset=[r for r in tests if r["representation"]==representation];matrix=np.asarray([[r[f"class_{c}_median"] for r in subset] for c in (1,2,3)],float);scale=np.nanstd(matrix,axis=0);z=(matrix-np.nanmean(matrix,axis=0))/np.where(scale>0,scale,1);image=ax.imshow(z,aspect="auto",cmap="coolwarm",vmin=-1.5,vmax=1.5);ax.set_xticks(range(len(metrics)),metrics,rotation=25,ha="right",fontsize=8);ax.set_yticks(range(3),["Class 1 (n=3)","Class 2 (n=5)","Class 3 (n=16)"]);ax.set_title(f"{representation} canonical-v7 sensitivity")
    fig.colorbar(image,ax=axes.tolist(),label="within-metric standardized median",fraction=.025,shrink=.82);fig.suptitle("Candidate-assisted canonical-v7 class morphology sensitivity\n24 canonical identities; frozen classes; secondary evidence only",x=.03,ha="left",fontsize=14);fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def _figure(class_tests:list[dict[str,Any]],certainty_tests:list[dict[str,Any]],certainty_class:dict[str,Any],path:Path)->None:
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.5),layout="constrained");grid="#D9DEE5";metrics=["contrast","sharpness","compactness","radius","peak offset","persistence","components"]
    for ax,representation in zip(axes[0],("Raw","Spatial ICA")):
        subset=[r for r in class_tests if r["representation"]==representation];matrix=np.asarray([[r[f"class_{c}_median"] for r in subset] for c in (1,2,3)],float);scale=np.nanstd(matrix,axis=0);z=(matrix-np.nanmean(matrix,axis=0))/np.where(scale>0,scale,1);image=ax.imshow(z,aspect="auto",cmap="coolwarm",vmin=-1.5,vmax=1.5);ax.set_xticks(range(len(metrics)),metrics,fontsize=7);ax.set_yticks(range(3),["Class 1","Class 2","Class 3"]);ax.set_title(f"{representation}: site-level morphology profile")
    fig.colorbar(image,ax=axes[0].tolist(),label="within-metric standardized median",fraction=.025,shrink=.82)
    table=np.asarray(certainty_class["table"]);bottom=np.zeros(2);x=np.arange(2)
    for c in (1,2,3):axes[1,0].bar(x,table[:,c-1],bottom=bottom,color=COLORS[c],edgecolor="#30343B",label=f"Class {c}");bottom+=table[:,c-1]
    axes[1,0].set_xticks(x,["Confirmed","Identity uncertain"]);axes[1,0].set_ylabel("canonical identities");axes[1,0].set_title("Certainty and dominant frozen class");axes[1,0].legend(frameon=False,ncol=3,fontsize=8,loc="upper center");axes[1,0].grid(axis="y",color=grid,lw=.5)
    subset=[r for r in certainty_tests if r["representation"]=="Spatial ICA"];y=np.arange(len(subset));values=np.asarray([r["adjusted_uncertain_minus_confirmed"] for r in subset]);lo=values-np.asarray([r["bootstrap_ci95_low"] for r in subset]);hi=np.asarray([r["bootstrap_ci95_high"] for r in subset])-values
    axes[1,1].errorbar(values,y,xerr=np.vstack((lo,hi)),fmt="o",color="#D87921",ecolor="#30343B",capsize=3);axes[1,1].axvline(0,color="#30343B",lw=.8);axes[1,1].set_yticks(y,[r["metric"].replace("_"," ") for r in subset],fontsize=8);axes[1,1].set_xlabel("adjusted uncertain minus confirmed");axes[1,1].set_title("Spatial ICA certainty sensitivity");axes[1,1].grid(axis="x",color=grid,lw=.5)
    fig.suptitle("Objective morphology by frozen class and certainty\nProtected v1 class profiles; candidate-assisted v7 certainty sensitivity; classes never refit",x=.03,ha="left",fontsize=14);fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def run(run_root:Path,v7_path:Path,raw_npy:Path,ica_tif:Path,output:Path)->dict[str,Any]:
    if output.exists() or output.with_name(output.name+".partial").exists():raise FileExistsError(output)
    partial=output.with_name(output.name+".partial");partial.mkdir(parents=True)
    detections=_read(run_root/"detection_profile_taxonomy_v5/detection_occurrence_profiles.tsv");protected=_read(run_root/"02_label_timing_contract/original_site_original_timing.tsv");morphology=_read(run_root/"17_spatial_ica_objective_morphology_v2/occurrence_morphology_metrics.tsv");matches=greedy_spatial_match(detections,protected,6.0);enriched,sites=_class_site_rows(matches,morphology);class_tests=_class_tests(sites)
    certainty_matches=[r for r in _read(run_root/"detection_class_extensions_v1/certainty_matches.tsv") if r["cohort"]=="canonical_v7_adjudicated"];v7=_read(v7_path);raw=np.load(raw_npy,mmap_mode="r",allow_pickle=False)
    with tifffile.TiffFile(ica_tif) as tif:
        description=json.loads(tif.pages[0].description);maximum=float(description["signal_display_source_limits"][1]);encoded=tif.asarray(out="memmap");ica=np.asarray(encoded,np.float32)*(maximum/65535.0)
    certainty_metrics=_extract_v7_metrics(certainty_matches,v7,raw,ica);certainty_tests,canonical_rows=_adjusted_certainty_tests(certainty_metrics);certainty_class=_certainty_class_summary(certainty_metrics);v7_sites=_v7_class_sites(certainty_metrics);v7_class_tests=_class_tests(v7_sites)
    _write(partial/"protected_v1_matched_morphology.tsv",enriched);_write(partial/"protected_v1_site_profiles.tsv",sites);_write(partial/"class_morphology_tests.tsv",class_tests);_write(partial/"canonical_v7_certainty_morphology.tsv",certainty_metrics);_write(partial/"canonical_v7_class_profiles.tsv",v7_sites);_write(partial/"canonical_v7_class_tests.tsv",v7_class_tests);_write(partial/"certainty_adjusted_tests.tsv",certainty_tests);_write(partial/"certainty_canonical_residuals.tsv",canonical_rows);_figure(class_tests,certainty_tests,certainty_class,partial/"class_certainty_objective_morphology.png");_v7_figure(v7_class_tests,partial/"canonical_v7_class_morphology_sensitivity.png")
    summary={"schema_version":1,"status":"completed_post_freeze_exploratory","class_fit_modified":False,"protected_v1":{"one_to_one_matches":len(matches),"matched_observation_sites":len({r['observation_site_id'] for r in enriched}),"dominant_class_site_counts":{"1":7,"2":1,"3":9},"three_class_inference_allowed":False,"reason":"only one protected Class-2 site","class_tests":class_tests,"circularity_warning":"Raw spatial morphology overlaps the spatial-specificity feature used in frozen class construction; Raw class associations are descriptive. Spatial ICA morphology was not used to fit classes."},"canonical_v7_class_sensitivity":{"canonical_identities":24,"dominant_class_identity_counts":{"1":3,"2":5,"3":16},"class_tests":v7_class_tests,"selection_warning":"candidate-assisted v7 sensitivity, not independent validation"},"canonical_v7_certainty":{"matched_nonartifact_observations":len({r['observation_id'] for r in certainty_metrics}),"canonical_identities":len({r['canonical_roi_id'] for r in certainty_metrics}),"certainty_class_association":certainty_class,"adjusted_tests":certainty_tests,"selection_warning":"candidate-assisted v7 sensitivity, not independent validation"},"scientific_audit":{"enabled":False,"opt_out_reason":"Author explicitly deferred manual packet-style labor on 2026-08-24; this post-freeze computational extension reuses existing Raw/spatial-ICA review media."},"interpretation_limits":["measurement-event classes are not neuron types","certainty associations are post-freeze and candidate-assisted","single recording and four bursts","unmatched observations and detections are not negatives","no precision specificity causal wiring or anatomical claim"]}
    all_q=all(0<=r['bh_q_across_14_class_tests']<=1 for r in class_tests+v7_class_tests) and all(0<=r['bh_q_across_14_certainty_tests']<=1 for r in certainty_tests)
    atomic_json(partial/"summary.json",summary);atomic_json(partial/"validation.json",{"status":"passed","protected_matches":len(matches),"expected_protected_matches":53,"two_representations_per_protected_match":len(enriched)==2*len(matches),"protected_class_test_family":len(class_tests)==14,"protected_class2_site_count":1,"protected_three_class_inference_prohibited":True,"certainty_nonartifact_matches":len({r['observation_id'] for r in certainty_metrics}),"expected_certainty_nonartifact_matches":44,"canonical_v7_class_identity_counts":[3,5,16],"canonical_v7_class_test_family":len(v7_class_tests)==14,"certainty_test_family":len(certainty_tests)==14,"all_q_values_bounded":all_q});atomic_json(partial/"artifact_index.json",{"artifacts":["REPORT.md","summary.json","validation.json","protected_v1_matched_morphology.tsv","protected_v1_site_profiles.tsv","class_morphology_tests.tsv","canonical_v7_certainty_morphology.tsv","canonical_v7_class_profiles.tsv","canonical_v7_class_tests.tsv","certainty_adjusted_tests.tsv","certainty_canonical_residuals.tsv","class_certainty_objective_morphology.png","canonical_v7_class_morphology_sensitivity.png","llm_context.json"]});atomic_json(partial/"llm_context.json",{"entry_point":"summary.json","primary_tables":["class_morphology_tests.tsv","canonical_v7_class_tests.tsv","certainty_adjusted_tests.tsv"],"figures":["class_certainty_objective_morphology.png","canonical_v7_class_morphology_sensitivity.png"],"class_fit_modified":False,"manual_packet_status":"deferred_by_author"});atomic_text(partial/"REPORT.md","# Objective morphology alignment with frozen classes and certainty\n\nProtected-v1 one-to-one spatial matches connect objective Raw/spatial-ICA morphology to the three frozen measurement-event classes. Site-grain permutation tests and BH correction prevent repeated bursts from acting as independent samples. The protected cohort contains only one dominant Class-2 site, so three-class inference is prohibited; those profiles are descriptive. A separate canonical-v7 sensitivity has 3/5/16 identities in Classes 1/2/3. Canonical-v7 certainty is also a candidate-assisted sensitivity: morphology is adjusted for class and burst, aggregated by canonical identity, and compared using identity-label permutations. Classes are not refit. Raw spatial associations partly overlap a class-fitting feature and are descriptive; spatial-ICA morphology is the new representation-level extension.\n");partial.replace(output);return summary
