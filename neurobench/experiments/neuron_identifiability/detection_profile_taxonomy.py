"""Label-free profile metrics and stable classes for strict B20 detections."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from neurobench.algorithms.scientific_feature_audit import causal_local_correlation_feature
from neurobench.experiments.hard_roi_adjudication.adjudication import label_view, load_tsv
from neurobench.experiments.hard_roi_adjudication.config import HardRoiAdjudicationConfig
from neurobench.experiments.hard_roi_adjudication.reevaluate import REVIEW_START_UI, _candidate_records, _event_bounds, _frames
from neurobench.experiments.hierarchical_parzen_ica.patch_information_program import _pool_values
from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import _quiet_calibrate

from .contracts import atomic_json, atomic_text


LANES=("coherence_w15","propagation_lag2_w15"); BUDGET=20; RADIUS=6
FEATURES=("log_raw_peak","log_residual_peak","log_snr","spatial_specificity","annulus_correlation","event_area_scaled","time_to_peak_fraction","lane_agreement","mean_rank_fraction","recurrence_fraction","quiet_intensity_scaled")


def _tsv(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t"); writer.writeheader(); writer.writerows(rows)


def _disk(stack:np.ndarray,x:float,y:float,radius:int)->np.ndarray:
    xi,yi=int(round(x)),int(round(y)); yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]; mask=xx*xx+yy*yy<=radius*radius
    return np.asarray(stack[:,yi-radius:yi+radius+1,xi-radius:xi+radius+1],float)[:,mask].mean(axis=1)


def _ring(stack:np.ndarray,x:float,y:float,inner:int=3,outer:int=6)->np.ndarray:
    xi,yi=int(round(x)),int(round(y)); yy,xx=np.ogrid[-outer:outer+1,-outer:outer+1]; rr=xx*xx+yy*yy; mask=(rr>=inner*inner)&(rr<=outer*outer)
    return np.asarray(stack[:,yi-outer:yi+outer+1,xi-outer:xi+outer+1],float)[:,mask].mean(axis=1)


def _strict_candidates(values:np.ndarray,labels:list[dict[str,Any]],config:HardRoiAdjudicationConfig,lane:str)->list[dict[str,Any]]:
    events={}; bounds={}
    for burst in range(1,5): bounds[burst]=_event_bounds(labels,burst); events[burst]=_frames(values,*bounds[burst])
    maps=_pool_values(np.asarray(values[:100]),events,float(config.evaluation["temporal_pool_temperature"]))["events"]; rows=[]
    for burst,(start,end) in bounds.items():
        for row in _candidate_records(values,maps[burst],start,end,distance=RADIUS,limit=BUDGET,separated=True): rows.append({"lane":lane,"burst_id":burst,"start_ui":start,"end_ui":end,**row})
    return rows


def _union_occurrences(candidates:list[dict[str,Any]])->list[dict[str,Any]]:
    result=[]
    for burst in range(1,5):
        groups=[]; ordered=sorted([r for r in candidates if r["burst_id"]==burst],key=lambda r:(r["rank"],r["lane"],r["y_px"],r["x_px"]))
        for row in ordered:
            group=next((g for g in groups if (row["x_px"]-g["x"])**2+(row["y_px"]-g["y"])**2<=RADIUS**2),None)
            if group is None: group={"x":float(row["x_px"]),"y":float(row["y_px"]),"members":[]}; groups.append(group)
            group["members"].append(row)
        for index,g in enumerate(groups,1):
            best=min(g["members"],key=lambda r:(r["rank"],r["lane"])); lanes={r["lane"]:r for r in g["members"]}
            result.append({"detection_occurrence_id":f"b{burst:02d}__det_{index:03d}","burst_id":burst,"x_px":best["x_px"],"y_px":best["y_px"],"start_ui":best["start_ui"],"end_ui":best["end_ui"],"peak_frame_ui":best["peak_frame_ui"],"lane_agreement":len(lanes)/2,"mean_rank_fraction":float(np.mean([r["rank"]/BUDGET for r in lanes.values()])),"coherence_rank":lanes.get("coherence_w15",{}).get("rank",""),"propagation_rank":lanes.get("propagation_lag2_w15",{}).get("rank",""),"coherence_score":lanes.get("coherence_w15",{}).get("score",""),"propagation_score":lanes.get("propagation_lag2_w15",{}).get("score","")})
    return result


def _site_ids(rows:list[dict[str,Any]])->None:
    coords=np.asarray([[r["x_px"],r["y_px"]] for r in rows],float); cluster=fcluster(linkage(coords,method="complete"),t=RADIUS,criterion="distance")
    order={old:new for new,old in enumerate(sorted(set(cluster),key=lambda c:(np.mean(coords[cluster==c,1]),np.mean(coords[cluster==c,0]))),1)}
    counts={c:len({rows[i]["burst_id"] for i in np.flatnonzero(cluster==c)}) for c in set(cluster)}
    for row,c in zip(rows,cluster,strict=True): row["detection_site_id"]=f"dsite_{order[int(c)]:03d}"; row["recurrence_fraction"]=counts[int(c)]/4


def _metrics(rows:list[dict[str,Any]],raw:np.ndarray,labels:list[dict[str,Any]])->None:
    for row in rows:
        start,end=int(row["start_ui"]),int(row["end_ui"]); base_start=max(1,start-20); stack=np.asarray(raw[base_start-1:end]); split=start-base_start
        center=_disk(stack,row["x_px"],row["y_px"],2); annulus=_ring(stack,row["x_px"],row["y_px"]); base=center[:split]; ann_base=annulus[:split]; event=center[split:]; ann_event=annulus[split:]
        baseline=float(np.median(base)); delta=event-baseline; residual=(event-ann_event)-float(np.median(base-ann_base)); mad=float(np.median(np.abs(base-baseline))*1.4826); peak=float(np.max(delta)); ann_peak=float(np.max(ann_event-float(np.median(ann_base)))); denom=abs(peak)+abs(ann_peak)+np.finfo(float).eps
        corr=float(np.corrcoef(center,annulus)[0,1]) if np.std(center)>0 and np.std(annulus)>0 else 0.0
        row.update({"quiet_intensity":baseline,"raw_peak":peak,"residual_peak":float(np.max(residual)),"robust_snr":peak/max(mad,np.finfo(float).eps),"event_area":float(np.sum(delta)),"time_to_peak_frames":int(np.argmax(delta)),"time_to_peak_fraction":int(np.argmax(delta))/max(1,len(delta)-1),"spatial_specificity":(peak-ann_peak)/denom,"annulus_correlation":corr})
        burst_labels=[l for l in labels if int(l["burst_id"])==row["burst_id"]]; distances=[np.hypot(float(l["x_px"])-row["x_px"],float(l["y_px"])-row["y_px"]) for l in burst_labels]; nearest=int(np.argmin(distances)); row["nearest_known_positive_distance_px"]=float(distances[nearest]); row["known_positive_within_6px"]=bool(distances[nearest]<=RADIUS); row["nearest_known_positive_id"]=burst_labels[nearest]["observation_id"] if distances[nearest]<=RADIUS else ""
    _site_ids(rows)


def _matrix(rows:list[dict[str,Any]])->tuple[np.ndarray,dict[str,dict[str,float]]]:
    raw=np.asarray([[np.log1p(max(r["raw_peak"],0)),np.log1p(max(r["residual_peak"],0)),np.log1p(max(r["robust_snr"],0)),r["spatial_specificity"],r["annulus_correlation"],np.sign(r["event_area"])*np.log1p(abs(r["event_area"])),r["time_to_peak_fraction"],r["lane_agreement"],r["mean_rank_fraction"],r["recurrence_fraction"],np.log1p(max(r["quiet_intensity"],0))] for r in rows],float)
    z=np.empty_like(raw); scaling={}; within_burst={0,1,2,3,4,5,6,10}
    for column,name in enumerate(FEATURES):
        if column in within_burst:
            scaling[name]={"mode":"within_burst_median_iqr","standardized_clip":[-3.0,3.0],"bursts":{}}
            for burst in range(1,5):
                index=np.asarray([i for i,row in enumerate(rows) if row["burst_id"]==burst]); values=raw[index,column]; median=float(np.median(values)); scale=float(np.subtract(*np.percentile(values,[75,25]))); scale=scale if scale>1e-9 else 1.0; z[index,column]=(values-median)/scale; scaling[name]["bursts"][str(burst)]={"median":median,"iqr":scale}
        else:
            median=float(np.median(raw[:,column])); scale=float(np.subtract(*np.percentile(raw[:,column],[75,25]))); scale=scale if scale>1e-9 else 1.0; z[:,column]=(raw[:,column]-median)/scale; scaling[name]={"mode":"global_median_iqr","median":median,"iqr":scale,"standardized_clip":[-3.0,3.0]}
    z=np.clip(z,-3.0,3.0)
    for row,values in zip(rows,z,strict=True):
        for name,value in zip(FEATURES,values,strict=True): row["z_"+name]=float(value)
    return z,scaling


def _classes(rows:list[dict[str,Any]],z:np.ndarray)->dict[str,Any]:
    rng=np.random.default_rng(20260823); candidates=[]
    for k in range(2,7):
        model=KMeans(n_clusters=k,random_state=20260823,n_init=50).fit(z); silhouette=float(silhouette_score(z,model.labels_)); ari=[]
        for _ in range(100):
            sample=rng.integers(0,len(z),len(z)); boot=KMeans(n_clusters=k,random_state=int(rng.integers(1_000_000)),n_init=10).fit(z[sample]); ari.append(float(adjusted_rand_score(model.labels_,boot.predict(z))))
        sizes=np.bincount(model.labels_,minlength=k); burst_coverage=[len({rows[i]["burst_id"] for i in np.flatnonzero(model.labels_==label)}) for label in range(k)]; viable=bool(np.min(sizes)>=5 and min(burst_coverage)>=2)
        candidates.append({"k":k,"silhouette":silhouette,"bootstrap_mean_ari":float(np.mean(ari)),"minimum_class_size":int(np.min(sizes)),"minimum_burst_coverage":int(min(burst_coverage)),"viable":viable,"objective":silhouette*max(float(np.mean(ari)),0) if viable else -1.0})
    viable=[row for row in candidates if row["viable"]]
    if not viable: raise RuntimeError("no stable taxonomy satisfies the five-occurrence and two-burst class minima")
    chosen=max(viable,key=lambda r:(r["objective"],-r["k"])); model=KMeans(n_clusters=chosen["k"],random_state=20260823,n_init=100).fit(z); centers=model.cluster_centers_
    order=sorted(range(chosen["k"]),key=lambda c:(-centers[c,3],-centers[c,0],c)); remap={old:i+1 for i,old in enumerate(order)}
    for row,label in zip(rows,model.labels_,strict=True): row["class_id"]=remap[int(label)]
    overall={key:float(np.median([r[key] for r in rows])) for key in ("raw_peak","spatial_specificity","annulus_correlation","quiet_intensity")}; upper_signal=float(np.percentile([r["raw_peak"] for r in rows],75)); names={}
    for cid in sorted(remap.values()):
        group=[r for r in rows if r["class_id"]==cid]; med={key:float(np.median([r[key] for r in group])) for key in ("raw_peak","spatial_specificity","annulus_correlation","quiet_intensity","time_to_peak_fraction","event_area","lane_agreement","recurrence_fraction")}
        tags=["localized" if med["spatial_specificity"]>=overall["spatial_specificity"] else "broad-coupled",("high-signal" if med["raw_peak"]>=upper_signal else ("moderate-signal" if med["raw_peak"]>=overall["raw_peak"] else "low-signal")),("early-peaking" if med["time_to_peak_fraction"]<.2 else ("delayed-peaking" if med["time_to_peak_fraction"]>.45 else "mid-peaking")),("positive-area" if med["event_area"]>0 else "negative-area"),"lane-consensus" if med["lane_agreement"]>=.75 else "lane-selective","recurrent" if med["recurrence_fraction"]>=.75 else "episodic"]
        if med["annulus_correlation"]<overall["annulus_correlation"]: tags.append("annulus-decoupled")
        if med["quiet_intensity"]>overall["quiet_intensity"]: tags.append("bright-background")
        names[cid]=f"Class {cid}: "+", ".join(tags)
    for row in rows: row["class_name"]=names[row["class_id"]]
    return {"candidate_models":candidates,"selected_k":chosen["k"],"selection_rule":"maximum silhouette times bootstrap mean ARI; lowest k breaks ties","class_names":names,"standardized_centroids":{str(remap[old]):{name:float(value) for name,value in zip(FEATURES,centers[old],strict=True)} for old in order}}


def _figures(output:Path,rows:list[dict[str,Any]],classification:dict[str,Any])->None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    colors=["#3b6fb6","#d9791f","#7a8f35","#b6537a","#6b6b6b","#9a6cc1"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    for cid in sorted({r["class_id"] for r in rows}):
        group=[r for r in rows if r["class_id"]==cid]; axes[0].scatter([r["raw_peak"] for r in group],[r["spatial_specificity"] for r in group],s=28,alpha=.75,color=colors[cid-1],label=f"Class {cid}"); axes[1].scatter([r["x_px"] for r in group],[r["y_px"] for r in group],s=30,alpha=.75,color=colors[cid-1],label=f"Class {cid}")
    axes[0].set(xlabel="native peak amplitude",ylabel="spatial specificity",title="Detection profile space"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2); axes[1].invert_yaxis(); axes[1].set(xlabel="x (px)",ylabel="y (px)",title="Spatial distribution of classes"); axes[1].set_aspect("equal")
    ids=sorted({r["class_id"] for r in rows}); counts=[sum(r["class_id"]==i for r in rows) for i in ids]; known=[sum(r["class_id"]==i and r["known_positive_within_6px"] for r in rows) for i in ids]; axes[2].bar([str(i) for i in ids],counts,color=[colors[i-1] for i in ids],edgecolor="#333333",label="all detections"); axes[2].bar([str(i) for i in ids],known,color="none",edgecolor="#111111",hatch="//",label="near known positive"); axes[2].set(xlabel="class",ylabel="detection occurrences",title="Class counts and known-positive overlap"); axes[2].legend(fontsize=8)
    fig.savefig(output/"detection_class_overview.png",dpi=160); plt.close(fig)
    centers=classification["standardized_centroids"]; matrix=np.asarray([[centers[str(i)][name] for name in FEATURES] for i in ids]); fig,ax=plt.subplots(figsize=(12,3.8)); image=ax.imshow(matrix,cmap="coolwarm",vmin=-2,vmax=2,aspect="auto"); ax.set(yticks=np.arange(len(ids)),yticklabels=[f"Class {i}" for i in ids],xticks=np.arange(len(FEATURES)),xticklabels=[name.replace("_","\n") for name in FEATURES]); plt.setp(ax.get_xticklabels(),rotation=30,ha="right"); ax.set_title("Robust-standardized class profiles"); fig.colorbar(image,ax=ax,label="median/IQR units"); fig.tight_layout(); fig.savefig(output/"detection_class_profiles.png",dpi=160); plt.close(fig)


def _representatives(output:Path,rows:list[dict[str,Any]],raw:np.ndarray,classification:dict[str,Any])->list[dict[str,Any]]:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ids=sorted({r["class_id"] for r in rows}); selected=[]
    for cid in ids:
        center=np.asarray([classification["standardized_centroids"][str(cid)][name] for name in FEATURES]); group=[r for r in rows if r["class_id"]==cid]; selected.append(min(group,key=lambda r:float(np.sum((np.asarray([r["z_"+name] for name in FEATURES])-center)**2))))
    fig,axes=plt.subplots(2,len(selected),figsize=(3.2*len(selected),6),constrained_layout=True,squeeze=False)
    for column,item in enumerate(selected):
        x,y=int(item["x_px"]),int(item["y_px"]); start,end=int(item["start_ui"]),int(item["end_ui"]); base_start=max(1,start-20); baseline=np.mean(raw[base_start-1:start-1],axis=0); event=np.max(raw[start-1:end],axis=0); radius=18; crop=event[y-radius:y+radius+1,x-radius:x+radius+1]-baseline[y-radius:y+radius+1,x-radius:x+radius+1]
        axes[0,column].imshow(crop,cmap="gray"); axes[0,column].scatter([radius],[radius],s=70,facecolors="none",edgecolors="#dd6b20"); axes[0,column].set(title=f"Class {item['class_id']} representative\n{item['detection_occurrence_id']}"); axes[0,column].set_axis_off()
        trace=_disk(np.asarray(raw[base_start-1:end]),x,y,2); frames=np.arange(base_start,end+1); axes[1,column].plot(frames,trace,color="#333333",lw=1); axes[1,column].axvspan(start,end,color="#dd6b20",alpha=.18); axes[1,column].set(xlabel="UI frame",ylabel="native intensity"); axes[1,column].grid(alpha=.2)
    fig.savefig(output/"detection_class_representatives.png",dpi=160); plt.close(fig)
    return [{"class_id":r["class_id"],"class_name":r["class_name"],"detection_occurrence_id":r["detection_occurrence_id"],"burst_id":r["burst_id"],"x_px":r["x_px"],"y_px":r["y_px"]} for r in selected]


def run_detection_taxonomy(data_root:Path,repo_root:Path,output:Path)->dict[str,Any]:
    if output.exists(): raise FileExistsError(output)
    partial=output.with_name(output.name+".partial"); partial.mkdir(parents=True,exist_ok=False); config=HardRoiAdjudicationConfig.load(repo_root/"examples/spon_ca_burst_hard_roi_adjudication_v1.example.json")
    label_path=data_root/"Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv"; labels=label_view(load_tsv(label_path),"original","original"); carrier=np.load(data_root/"Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1/features/carrier_signed.npy",mmap_mode="r",allow_pickle=False)
    candidates=[]
    for lane,lag in (("coherence_w15",0),("propagation_lag2_w15",2)):
        values=_quiet_calibrate(causal_local_correlation_feature(carrier,window_frames=15,lag_frames=lag,spatial_sigma_px=2.0,activity_qualified=True),100); candidates.extend(_strict_candidates(values,labels,config,lane)); del values
    rows=_union_occurrences(candidates); raw=np.load(data_root/"Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",mmap_mode="r",allow_pickle=False); _metrics(rows,raw,labels); z,scaling=_matrix(rows); classification=_classes(rows,z); _figures(partial,rows,classification); representatives=_representatives(partial,rows,raw,classification)
    _tsv(partial/"detection_occurrence_profiles.tsv",rows); site_rows=[]
    for site in sorted({r["detection_site_id"] for r in rows}):
        group=[r for r in rows if r["detection_site_id"]==site]; site_rows.append({"detection_site_id":site,"occurrences":len(group),"bursts":len({r["burst_id"] for r in group}),"x_px":float(np.median([r["x_px"] for r in group])),"y_px":float(np.median([r["y_px"] for r in group])),"dominant_class":max({r["class_id"] for r in group},key=lambda c:sum(x["class_id"]==c for x in group)),"class_consistency":max(sum(x["class_id"]==c for x in group) for c in {r["class_id"] for r in group})/len(group),"known_positive_occurrences":sum(r["known_positive_within_6px"] for r in group),"median_raw_peak":float(np.median([r["raw_peak"] for r in group])),"median_spatial_specificity":float(np.median([r["spatial_specificity"] for r in group]))})
    _tsv(partial/"detection_site_profiles.tsv",site_rows); class_rows=[]
    for cid in sorted({r["class_id"] for r in rows}):
        group=[r for r in rows if r["class_id"]==cid]; class_rows.append({"class_id":cid,"class_name":group[0]["class_name"],"detection_occurrences":len(group),"detection_sites":len({r["detection_site_id"] for r in group}),"bursts":len({r["burst_id"] for r in group}),"median_raw_peak":float(np.median([r["raw_peak"] for r in group])),"median_residual_peak":float(np.median([r["residual_peak"] for r in group])),"median_spatial_specificity":float(np.median([r["spatial_specificity"] for r in group])),"median_recurrence_fraction":float(np.median([r["recurrence_fraction"] for r in group])),"known_positive_overlap":sum(r["known_positive_within_6px"] for r in group),"known_positive_overlap_interpretation":"descriptive only; not precision"})
    _tsv(partial/"class_summary.tsv",class_rows); summary={"schema_version":1,"status":"complete","candidate_source":"strict object-separated union of frozen coherence_w15 and propagation_lag2_w15 B20 proposals","fitting_labels_used":False,"known_labels_usage":"post-freeze class characterization only","detection_occurrences":len(rows),"detection_sites":len(site_rows),"features":list(FEATURES),"robust_scaling":scaling,"classification":classification,"class_counts":{str(cid):sum(r["class_id"]==cid for r in rows) for cid in sorted({r["class_id"] for r in rows})},"known_positive_overlap_by_class":{str(cid):sum(r["class_id"]==cid and r["known_positive_within_6px"] for r in rows) for cid in sorted({r["class_id"] for r in rows})},"representatives":representatives,"interpretation":"measurement/detection-event archetypes, not biological neuron types","prohibited":["precision","false-positive classes","cell types","population generalization"]}
    atomic_json(partial/"summary.json",summary); atomic_json(partial/"validation.json",{"status":"passed","finite_profile_matrix":bool(np.all(np.isfinite(z))),"labels_excluded_from_fit":True,"strict_nms_radius_px":RADIUS,"budget_per_burst_per_lane":BUDGET}); atomic_text(partial/"REPORT.md",f"# Detection profile taxonomy\n\nThe corrected strict-B20 union contains {len(rows)} detection occurrences consolidated into {len(site_rows)} spatial detection sites. A label-free robust profile model selected {classification['selected_k']} classes. These are measurement/detection archetypes, not neuron types. Known-positive overlap is reported only after class freezing and cannot identify precision.\n"); partial.replace(output); return summary
