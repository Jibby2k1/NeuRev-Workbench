"""Frozen within-recording retrieval comparison for contextual envelopes."""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from collections import defaultdict
from pathlib import Path
from typing import Any
os.environ.setdefault("MPLCONFIGDIR","/tmp/neurev-context-envelope-retrieval-mpl")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter,maximum_filter1d
from scipy.stats import spearmanr
from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import make_event_and_quiet_masks,occurrence_score

START_UI=1800;FRAMES=560;BOOTSTRAPS=5000;SEED=20260831;FEATURES=("A","A2","H_t3","H_t5","H_st5k3","H_t5_shuffle137","H_st5k3_displace11x17");SOURCES=("RAW","Z_CURRENT","T_ENVELOPE")
LABELS={"A":"instantaneous A","A2":"amplitude A2","H_t3":"H temporal 60 ms","H_t5":"H temporal 100 ms","H_st5k3":"H 100 ms plus 3x3","H_t5_shuffle137":"H temporal shuffled","H_st5k3_displace11x17":"H spatial displaced"}

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def atomic_json(path:Path,payload:Any)->None:
 q=path.with_suffix(path.suffix+".partial");q.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");q.replace(path)
def write_tsv(path:Path,rows:list[dict[str,Any]])->None:
 if not rows:raise ValueError(path)
 q=path.with_suffix(path.suffix+".partial")
 with q.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
 q.replace(path)
def trailing_max(a:np.ndarray,w:int)->np.ndarray:return maximum_filter1d(np.asarray(a,dtype=np.float32),size=w,axis=0,origin=(w-1)//2,mode="nearest")
def spatial_max(a:np.ndarray,k:int)->np.ndarray:return maximum_filter(np.asarray(a,dtype=np.float32),size=(1,k,k),mode="nearest")
def agreement_attenuated(a:np.ndarray,u:np.ndarray)->np.ndarray:return np.divide(a*a,np.maximum(u,1e-6),dtype=np.float32)
def feature_arrays(a:np.ndarray)->dict[str,np.ndarray]:
 u3=trailing_max(a,3);u5=trailing_max(a,5);ust=spatial_max(u5,3);shuffle=np.maximum(a,np.roll(u5,137,axis=0));displaced=np.maximum(a,np.roll(ust,(11,17),axis=(1,2)));return {"A":a,"A2":a*a,"H_t3":agreement_attenuated(a,u3),"H_t5":agreement_attenuated(a,u5),"H_st5k3":agreement_attenuated(a,ust),"H_t5_shuffle137":agreement_attenuated(a,shuffle),"H_st5k3_displace11x17":agreement_attenuated(a,displaced)}
def calibrate(x:np.ndarray)->tuple[np.ndarray,dict[str,float]]:
 q=np.asarray(x[:100,::4,::4],dtype=np.float64);center=float(np.median(q));hi=float(np.percentile(q,99.5));scale=max(hi-center,1e-6);return np.clip((x-center)/scale,0,1).astype(np.float32),{"quiet_center":center,"quiet_p99_5":hi,"positive_scale":scale}
def load_paths(order:Path)->dict[str,Path]:
 pre=json.loads((order/"preflight.json").read_text());m=json.loads((order/"array_manifest.json").read_text());return {"RAW":Path(pre["inputs"]["raw"]["path"]),"Z_CURRENT":Path(pre["inputs"]["recovery_ls"]["path"]),"T_ENVELOPE":order/m["orders"]["MP_LS_ICA"]["path"]}
def load_source(path:Path,name:str)->np.ndarray:
 x=np.load(path,mmap_mode="r",allow_pickle=False);return np.asarray(x[1799:2359] if name=="RAW" else x[:560],dtype=np.float32)
def bootstrap_site(rows,field,rng):
 g=defaultdict(list)
 for r in rows:g[r["site_id"]].append(float(r[field]))
 v=np.asarray([np.mean(g[k]) for k in sorted(g)]);d=np.asarray([np.mean(rng.choice(v,len(v),replace=True)) for _ in range(BOOTSTRAPS)]);return float(v.mean()),float(np.quantile(d,.025)),float(np.quantile(d,.975))
def paired(rows,feat,base,rng):
 o=defaultdict(dict);sites={}
 for r in rows:o[r["observation_id"]][r["feature_id"]]=float(r["event_localization_percentile"]);sites[r["observation_id"]]=r["site_id"]
 g=defaultdict(list)
 for oid,v in o.items():
  if feat in v and base in v:g[sites[oid]].append(v[feat]-v[base])
 x=np.asarray([np.mean(g[k]) for k in sorted(g)]);d=np.asarray([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(BOOTSTRAPS)]);return float(x.mean()),float(np.quantile(d,.025)),float(np.quantile(d,.975)),d
def summarize(rows):
 rng=np.random.default_rng(SEED);out=[];draws=[]
 for src in SOURCES:
  sr=[r for r in rows if r["source"]==src]
  for feat in FEATURES:
   q=[r for r in sr if r["feature_id"]==feat];mean,lo,hi=bootstrap_site(q,"event_localization_percentile",rng);d,dlo,dhi,db=paired(sr,feat,"A2",rng);control="H_t5_shuffle137" if feat in ("H_t3","H_t5") else "H_st5k3_displace11x17" if feat=="H_st5k3" else "A2";cd,clo,chi,cb=paired(sr,feat,control,rng);bursts={str(b):float(np.median([float(r["event_localization_percentile"]) for r in q if int(r["burst_id"])==b])) for b in range(1,5)};a2={str(b):float(np.median([float(r["event_localization_percentile"]) for r in sr if r["feature_id"]=="A2" and int(r["burst_id"])==b])) for b in range(1,5)};eligible=feat in ("H_t3","H_t5","H_st5k3");advance=bool(eligible and lo>.5 and min(bursts.values())>=.75 and dlo>0 and clo>0 and min(bursts[k]-a2[k] for k in bursts)>=-.02);out.append({"source":src,"feature_id":feat,"feature_label":LABELS[feat],"occurrences":len(q),"sites":len({r['site_id'] for r in q}),"mean_event_localization_percentile":mean,"ci95_low":lo,"ci95_high":hi,"paired_delta_vs_A2":d,"delta_A2_ci95_low":dlo,"delta_A2_ci95_high":dhi,"matched_control":control,"paired_delta_vs_control":cd,"delta_control_ci95_low":clo,"delta_control_ci95_high":chi,"median_event_quiet_effect":float(np.median([float(r["event_quiet_effect"]) for r in q])),"median_event_energy_fraction":float(np.median([float(r["event_energy_fraction"]) for r in q])),**{f"burst_{k}_median":v for k,v in bursts.items()},"minimum_burst_delta_vs_A2":min(bursts[k]-a2[k] for k in bursts),"advance_gate_passed":advance});
   for i,(x,y) in enumerate(zip(db,cb,strict=True)):draws.append({"source":src,"feature_id":feat,"draw":i,"delta_vs_A2":float(x),"delta_vs_control":float(y)})
 return out,draws
def repeatability(rows):
 out=[]
 for src in SOURCES:
  for feat in FEATURES:
   q=[r for r in rows if r["source"]==src and r["feature_id"]==feat];bb=defaultdict(dict)
   for r in q:bb[int(r["burst_id"])][r["site_id"]]=float(r["event_localization_percentile"])
   for a in range(1,5):
    for b in range(a+1,5):
     shared=sorted(set(bb[a])&set(bb[b]));x=np.asarray([bb[a][s] for s in shared]);y=np.asarray([bb[b][s] for s in shared]);rho=float(spearmanr(x,y).statistic) if len(shared)>=5 and np.ptp(x)>0 and np.ptp(y)>0 else float("nan");out.append({"source":src,"feature_id":feat,"burst_a":a,"burst_b":b,"shared_sites":len(shared),"spearman_rho":rho})
 return out
def trace_figure(path,item,traces):
 fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,constrained_layout=True);start=int(item["event_start_ui"])-START_UI;stop=int(item["event_end_ui"])-START_UI+1
 for ax,src in zip(axes,SOURCES,strict=True):
  for feat in FEATURES:ax.plot(np.arange(START_UI,START_UI+FRAMES),traces[src][feat],lw=1,label=LABELS[feat])
  ax.axvspan(START_UI+start,START_UI+stop-1,color="#f2c14e",alpha=.18);ax.set_ylabel(src);ax.grid(alpha=.18)
 axes[0].legend(ncol=4,fontsize=7);axes[-1].set_xlabel("UI frame");fig.suptitle(f"{item['observation_id']} | rank {int(item['priority_rank']):03d} | immutable center ({int(item['x_int'])},{int(item['y_int'])})");fig.savefig(path,dpi=120);plt.close(fig)
def summary_figure(path,summaries):
 fig,axes=plt.subplots(1,3,figsize=(18,5),sharey=True,constrained_layout=True)
 for ax,src in zip(axes,SOURCES,strict=True):
  q=[r for r in summaries if r["source"]==src];x=np.arange(len(q));m=np.asarray([r["mean_event_localization_percentile"] for r in q]);lo=np.asarray([r["ci95_low"] for r in q]);hi=np.asarray([r["ci95_high"] for r in q]);ax.errorbar(x,m,yerr=[m-lo,hi-m],fmt="o",capsize=3);ax.set_xticks(x,[r["feature_id"] for r in q],rotation=50,ha="right");ax.axhline(.5,color="gray",ls="--");ax.set_title(src);ax.grid(alpha=.2)
 axes[0].set_ylabel("site-bootstrap mean event localization percentile");fig.savefig(path,dpi=150);plt.close(fig)
def run(args):
 root=args.output_root.resolve();partial=root.with_name(root.name+".partial")
 if root.exists() or partial.exists():raise FileExistsError(root)
 partial.mkdir(parents=True);manifest=Path(args.manifest).resolve();order=Path(args.order_root).resolve();protocol=Path(args.protocol).resolve();paths=load_paths(order);pre={"status":"ready","protocol":{"path":str(protocol),"sha256":sha256(protocol)},"manifest":{"path":str(manifest),"sha256":sha256(manifest)},"inputs":{k:{"path":str(v),"sha256":sha256(v)} for k,v in paths.items()},"code":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())},"features":list(FEATURES),"bootstrap_seed":SEED,"bootstraps":BOOTSTRAPS,"scientific_claim_promoted":False};atomic_json(partial/"preflight.json",pre);atomic_json(partial/"status.json",{"status":"preflight_ready"})
 if args.preflight_only:return pre
 payload=json.loads(manifest.read_text());items=payload["items"];intervals=sorted({(int(i["event_start_ui"]),int(i["event_end_ui"])) for i in items});event_union,quiet=make_event_and_quiet_masks(FRAMES,intervals);sites={f"{i['original_roi_id']}@x{int(i['x_int'])}_y{int(i['y_int'])}":(int(i["x_int"]),int(i["y_int"])) for i in items};traces={s:{} for s in sites};cal={}
 for src,path in paths.items():
  x=load_source(path,src);A,cal[src]=calibrate(x);F=feature_arrays(A)
  for site,(xx,yy) in sites.items():traces[site][src]={f:np.asarray(F[f][:,yy,xx],dtype=np.float64) for f in FEATURES}
  del x,A,F;print(f"FEATURES {src} complete",flush=True)
 rows=[]
 for item in items:
  site=f"{item['original_roi_id']}@x{int(item['x_int'])}_y{int(item['y_int'])}"
  for src in SOURCES:
   for feat in FEATURES:rows.append({"observation_id":item["observation_id"],"site_id":site,"canonical_roi_id":item["canonical_roi_id"],"burst_id":int(item["burst_id"]),"source":src,"feature_id":feat,"event_start_ui":int(item["event_start_ui"]),"event_end_ui":int(item["event_end_ui"]),**occurrence_score(traces[site][src][feat],int(item["event_start_ui"]),int(item["event_end_ui"]),quiet,event_union)})
 summaries,draws=summarize(rows);repeats=repeatability(rows);one=partial/"1_Expert_Annotations";two=partial/"2_Model_Annotations";three=partial/"3_Comparison";one.mkdir();two.mkdir();(three/"trace_comparisons").mkdir(parents=True);(one/"README.md").write_text("# Expert Annotations\n\nReuses the validated canonical-v7 expert-only media identified in llm_context.json. No identity or coordinate changed.\n");(two/"README.md").write_text("# Model Annotations\n\nNot applicable: this experiment creates no detector, proposals, model coordinates, or candidate rankings.\n")
 write_tsv(three/"occurrence_feature_metrics.tsv",rows);write_tsv(three/"feature_summary.tsv",summaries);write_tsv(three/"paired_bootstrap_draws.tsv",draws);write_tsv(three/"burst_pair_repeatability.tsv",repeats);summary_figure(three/"feature_comparison.png",summaries)
 for n,item in enumerate(items,1):
  site=f"{item['original_roi_id']}@x{int(item['x_int'])}_y{int(item['y_int'])}";trace_figure(three/f"trace_comparisons/{item['observation_id']}.png",item,traces[site])
  if n%20==0:print(f"TRACE {n}/{len(items)}",flush=True)
 passed=[r for r in summaries if r["advance_gate_passed"]];summary={"status":"complete_exploratory","population":{"occurrences":len(items),"sites":len(sites)},"features_per_source":len(FEATURES),"occurrence_metric_rows":len(rows),"advance_gate_passes":[{"source":r["source"],"feature_id":r["feature_id"]} for r in passed],"scientific_claim_promoted":False,"calibration":cal};atomic_json(partial/"summary.json",summary);atomic_json(partial/"llm_context.json",{"experiment":"NREV-EXP-0032 contextual-envelope retrieval comparison","grain":"106 occurrences nested in 50 immutable sites","primary_metric":"site-bootstrap mean event localization percentile","unmatched_semantics":"unknown_not_negative","expert_annotations":"validated canonical-v7 upstream media","model_annotations":"not_applicable_no_new_detections","primary_tables":["3_Comparison/feature_summary.tsv","3_Comparison/occurrence_feature_metrics.tsv"],"primary_figure":"3_Comparison/feature_comparison.png","trace_comparisons":106,"limitations":["one previously analyzed recording","quiet windows not verified negatives","no precision specificity or identity claim"]});checks={"occurrence_rows":len(rows)==106*3*7,"summary_rows":len(summaries)==21,"trace_figures":len(list((three/"trace_comparisons").glob("*.png")))==106,"finite_metrics":all(np.isfinite(float(r["event_localization_percentile"])) for r in rows),"population":len(items)==106 and len(sites)==50,"figure":(three/"feature_comparison.png").is_file()};atomic_json(partial/"validation.json",{"status":"passed" if all(checks.values()) else "failed","checks":checks,"scientific_audit":{"mode":"analysis_only_reuses_validated_upstream_expert_media","expert":"upstream canonical-v7 media","model":"not_applicable_no_new_detections","comparison_trace_count":106}})
 report=["# Contextual-envelope retrieval comparison v1","",f"Advance-gate passes: {summary['advance_gate_passes'] or 'none'}.","","This is a within-recording exploratory comparison. Quiet windows are temporal references, unmatched activity remains unknown, and scientific promotion is false."];(partial/"REPORT.md").write_text("\n".join(report)+"\n");atomic_json(partial/"status.json",{"status":"complete_exploratory","validation":"passed","scientific_claim_promoted":False});files=sorted(p for p in partial.rglob("*") if p.is_file() and p.name!="artifact_index.json");atomic_json(partial/"artifact_index.json",{"artifacts":[{"path":str(p.relative_to(partial)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files]});partial.replace(root);return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--order-root",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--preflight-only",action="store_true");a=p.parse_args();print(json.dumps(run(a),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
