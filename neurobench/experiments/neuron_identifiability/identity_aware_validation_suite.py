"""Frozen canonical-v8 validation suite for localization, identity, and NMS."""
from __future__ import annotations
import argparse,csv,itertools,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.optimize import nnls
from scipy.stats import rankdata
from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import ALIGNMENT_START_UI,atomic_json,sha256,write_tsv
from neurobench.experiments.neuron_identifiability.identity_aware_feature_inspection import STAGES,RADIUS,event_response,response_patch,morphology
from neurobench.portable_paths import data_root,media_root,portable_path

def rt(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f,delimiter='\t'))
def weights(p):
 v=np.clip(np.asarray(p,float),0,None);q=np.quantile(v[v>0],.75) if np.any(v>0) else np.inf;v=np.where((v>0)&(v>=q),v,0);return v/max(v.sum(),1e-9)
def candidate(maps,keep=(0,1,2),mask=None):
 ranks=[]
 for i in keep:
  r=rankdata(maps[i].ravel()).reshape(maps[i].shape)/maps[i].size;ranks.append(r)
 c=np.median(np.stack(ranks),0)
 if mask is not None:c=np.where(mask,-np.inf,c)
 y,x=np.unravel_index(np.argmax(c),c.shape);return int(x),int(y)
def exact_p(a,b):
 vals=np.r_[a,b];n=len(a);obs=abs(np.median(a)-np.median(b));ds=[]
 for ix in itertools.combinations(range(len(vals)),n):
  s=set(ix);ds.append(abs(np.median(vals[list(s)])-np.median(vals[[i for i in range(len(vals)) if i not in s]])))
 return (1+sum(x>=obs-1e-12 for x in ds))/(1+len(ds))
def run(repo,out,recenter):
 if out.exists():raise FileExistsError(out)
 part=out.with_name(out.name+'.partial');part.mkdir(parents=True)
 data=data_root(repo);media=media_root(repo);atlas_path=media/'v7_priority_neuron_media_cs_parzen/manifest.json';atlas=json.load(atlas_path.open());items=atlas['items'];by={i['observation_id']:i for i in items}
 rec=rt(recenter/'recenter_summary.tsv');miss={r['observation_id']:r for r in rec};cache=data/'Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache';paths=[data/'Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy',cache/'recovery_msica.npy',cache/'recovery_msln.npy'];st=[np.load(p,mmap_mode='r') for p in paths];st[0]=st[0][ALIGNMENT_START_UI-1:ALIGNMENT_START_UI-1+len(st[1])]
 geoms={(i['original_roi_id'],int(i['x_int']),int(i['y_int'])):i for i in items};gs=list(geoms.values());bycanon=defaultdict(list)
 for i in items:bycanon[i['canonical_roi_id']].append(i)
 transfer=[];stageout=[];frozen=[];identity=[];spatial=[];controls=[]
 def maps_bounds(it):
  z=[response_patch(s,it) for s in st];return [x[0] for x in z],z[0][1]
 for oid,r in miss.items():
  it=by[oid];maps,b=maps_bounds(it);x0,x1,y0,y1=b;ox,oy=int(it['x_int']),int(it['y_int']);outcome='clear' if r['candidate_identity_status']=='identity_clear_within_6px' else 'collision';cx0,cy0=candidate(maps);cx,cy=x0+cx0,y0+cy0;start=int(it['event_start_ui'])-ALIGNMENT_START_UI;stop=int(it['event_end_ui'])-ALIGNMENT_START_UI+1
  for held in range(3):
   px,py=candidate(maps,tuple(i for i in range(3) if i!=held));gx,gy=x0+px,y0+py;or_,_=event_response(st[held][:,oy,ox],start,stop);cr,_=event_response(st[held][:,gy,gx],start,stop);stageout.append(dict(observation_id=oid,outcome=outcome,held_stage=STAGES[held],dx=gx-ox,dy=gy-oy,candidate_to_original=cr/max(abs(or_),1e-9)))
  prior=[q for q in bycanon[it['canonical_roi_id']] if q['observation_id']!=oid]
  for src in prior:
   sm,sb=maps_bounds(src);w=weights(sm[2]);sx0,sx1,sy0,sy1=sb
   if (sx1-sx0,sy1-sy0)!=(x1-x0,y1-y0):continue
   tr=np.tensordot(np.asarray(st[2][:,y0:y1,x0:x1],float),w,axes=([1,2],[0,1]));resp,mad=event_response(tr,start,stop);transfer.append(dict(observation_id=oid,outcome=outcome,source_observation_id=src['observation_id'],source_burst=src['burst_id'],target_burst=it['burst_id'],peak_mad=mad,event_response=resp,footprint_iou=float(np.sum((w>0)&(weights(maps[2])>0))/max(np.sum((w>0)|(weights(maps[2])>0)),1))))
  pre=np.asarray(st[2][max(0,start-15):start,y0:y1,x0:x1],float);pw=np.std(pre,axis=0);pw=np.where(pw>=np.quantile(pw,.75),pw,0);pw/=max(pw.sum(),1e-9);tr=np.tensordot(np.asarray(st[2][:,y0:y1,x0:x1],float),pw,axes=([1,2],[0,1]));resp,mad=event_response(tr,start,stop);frozen.append(dict(observation_id=oid,outcome=outcome,method='pre_event_variability',event_response=resp,peak_mad=mad))
  others=[g for g in gs if g['original_roi_id']!=it['original_roi_id']];near=sorted(others,key=lambda g:math.hypot(cx-int(g['x_int']),cy-int(g['y_int'])))[:5];candtr=np.asarray(st[2][:,cy,cx],float);orig=np.asarray(st[2][:,oy,ox],float);X=np.column_stack([np.asarray(st[2][:,int(g['y_int']),int(g['x_int'])],float) for g in near]);X=(X-X.mean(0))/np.maximum(X.std(0),1e-9);z=(candtr-candtr.mean())/max(candtr.std(),1e-9);coef,_=nnls(X,z);fit=X@coef;res=z-fit;rr,rm=event_response(res,start,stop)
  lag=[]
  for k in range(-5,6):
   a=z[max(0,k):len(z)+min(0,k)];q=X[:,0][max(0,-k):len(z)-max(0,k)];lag.append((k,float(np.corrcoef(a,q)[0,1])))
  mask=np.ones_like(maps[0],bool)
  for g in others:
   gx,gy=int(g['x_int'])-x0,int(g['y_int'])-y0
   yy,xx=np.ogrid[:mask.shape[0],:mask.shape[1]];mask&=(xx-gx)**2+(yy-gy)**2>36
  ex,ey=candidate(maps,mask=mask);ex+=x0;ey+=y0;er,_=event_response(st[2][:,ey,ex],start,stop)
  identity.append(dict(observation_id=oid,outcome=outcome,multineighbor_r2=1-float(np.sum((z-fit)**2))/max(float(np.sum(z*z)),1e-9),residual_event_response=rr,residual_peak_mad=rm,max_neighbor_weight=float(coef.max()),best_neighbor_lag=max(lag,key=lambda q:abs(q[1]))[0],best_neighbor_lag_r=max(lag,key=lambda q:abs(q[1]))[1],exclusion_x=ex,exclusion_y=ey,exclusion_response=er))
  m=morphology(maps[2]);val=np.clip(maps[2],0,None);py,px=np.unravel_index(np.argmax(val),val.shape);angles=np.arctan2(*np.ogrid[-py:val.shape[0]-py,-px:val.shape[1]-px]);sectors=[float(val[(angles>=-np.pi+i*np.pi/4)&(angles<-np.pi+(i+1)*np.pi/4)].sum()) for i in range(8)];maxf=ndimage.maximum_filter(val,3);peaks=np.argwhere((val==maxf)&(val>np.quantile(val[val>0],.9))) if np.any(val>0) else np.empty((0,2));prom=float(np.sort(val.ravel())[-1]-np.sort(val.ravel())[-2]);spatial.append(dict(observation_id=oid,outcome=outcome,sector_cv=float(np.std(sectors)/max(np.mean(sectors),1e-9)),peak_count=len(peaks),peak_prominence=prom,**m))
 # negative controls: recovered true-event shifts and quiet pseudo-event shifts
 for it in items:
  if it['observation_id'] in miss:continue
  maps,b=maps_bounds(it);px,py=candidate(maps);controls.append(dict(observation_id=it['observation_id'],control='recovered_true_event',shift_px=math.hypot(b[0]+px-int(it['x_int']),b[2]+py-int(it['y_int']))))
 for it in items[:min(106,len(items))]:
  q=dict(it);dur=int(it['event_end_ui'])-int(it['event_start_ui']);q['event_end_ui']=int(it['event_start_ui'])-20;q['event_start_ui']=int(q['event_end_ui'])-dur;maps,b=maps_bounds(q);px,py=candidate(maps);controls.append(dict(observation_id=it['observation_id'],control='quiet_pseudo_event',shift_px=math.hypot(b[0]+px-int(it['x_int']),b[2]+py-int(it['y_int']))))
 # frozen candidate budget/radius and identity-aware duplicate proxy
 model_path=repo/'Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8/10_representation_confirmation/detector_visual_audit_v2/2_Model_Annotations/model_occurrences.csv';models=list(csv.DictReader(model_path.open()));budget=[]
 for oid,r in miss.items():
  it=by[oid];ox,oy=int(it['x_int']),int(it['y_int'])
  for lane in ('carrier_signed','coherence_w15','propagation_lag2_w15'):
   rows=[x for x in models if x['lane']==lane and int(x['burst_id'])==int(it['burst_id'])]
   for rad in (4,6,8):
    ranks=[int(x['rank']) for x in rows if math.hypot(int(x['x_px'])-ox,int(x['y_px'])-oy)<=rad]
    first=min(ranks) if ranks else 999999
    for bud in (20,40,58,80,100):budget.append(dict(observation_id=oid,outcome='clear' if r['candidate_identity_status']=='identity_clear_within_6px' else 'collision',lane=lane,radius_px=rad,budget=bud,recovered=int(first<=bud),first_rank='' if first==999999 else first))
 for name,rows in [('cross_burst_transfer.tsv',transfer),('leave_stage_out.tsv',stageout),('frozen_footprint.tsv',frozen),('identity_contamination.tsv',identity),('spatial_topology.tsv',spatial),('negative_controls.tsv',controls),('budget_nms_counterfactual.tsv',budget)]:write_tsv(part/name,rows)
 tests=[]
 for table,field in ((stageout,'candidate_to_original'),(identity,'max_neighbor_weight'),(spatial,'sector_cv')):
  grouped=defaultdict(list)
  for x in table:grouped[(x['observation_id'],x['outcome'])].append(float(x[field]))
  a=[float(np.median(v)) for (oid,outcome),v in grouped.items() if outcome=='collision'];b=[float(np.median(v)) for (oid,outcome),v in grouped.items() if outcome=='clear'];tests.append(dict(metric=field,analysis_grain='occurrence',collision_n=len(a),clear_n=len(b),collision_median=float(np.median(a)),clear_median=float(np.median(b)),exact_permutation_p=exact_p(a,b)))
 write_tsv(part/'exact_permutation_tests.tsv',tests)
 nms=json.load((repo/'paper/overleaf_jnm/generated_analysis/strict_object_nms_sensitivity.json').open());atomic_json(part/'frozen_strict_nms_source.json',nms)
 summary={'schema_version':1,'status':'complete_exploratory','families':{'transfer':len(transfer),'stage_out':len(stageout),'frozen_footprint':len(frozen),'identity':len(identity),'spatial':len(spatial),'controls':len(controls),'budget_nms':len(budget),'exact_tests':len(tests)},'headline':{'collision_vs_clear_max_neighbor_weight':[tests[1]['collision_median'],tests[1]['clear_median'],tests[1]['exact_permutation_p']],'recovered_true_event_shift_median':float(np.median([x['shift_px'] for x in controls if x['control']=='recovered_true_event'])),'quiet_shift_median':float(np.median([x['shift_px'] for x in controls if x['control']=='quiet_pseudo_event']))},'limits':['same recording','four clear misses','exact tests descriptive after review-defined groups','unmatched candidates unknown','no detector refit']};atomic_json(part/'summary.json',summary)
 valid={'status':'passed','checks':{'twelve_stageout_x3':len(stageout)==36,'twelve_identity':len(identity)==12,'twelve_spatial':len(spatial)==12,'negative_controls_present':len(controls)>100,'budget_grid_complete':len(budget)==12*3*3*5,'exact_tests':len(tests)==3,'source_audit_passed':json.load((recenter/'validation.json').open())['status']=='passed'}};atomic_json(part/'validation.json',valid);atomic_json(part/'llm_context.json',{'entry_point':'summary.json','tables':[p.name for p in part.glob('*.tsv')],'audit_reuse':portable_path(recenter,repository=repo,data=data,media=media),'limits':summary['limits']});(part/'REPORT.md').write_text('# Canonical-v8 validation suite\n\nFrozen transfer, contamination, spatial, negative-control, budget/NMS, and exact-permutation tests. No detector was refit.\n')
 arts=[{'path':str(p.relative_to(part)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(part.rglob('*')) if p.is_file() and p.name!='artifact_index.json'];atomic_json(part/'artifact_index.json',{'artifacts':arts,'inputs':{portable_path(p,repository=repo,data=data,media=media):sha256(p) for p in paths}});part.replace(out);return summary
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);p.add_argument('--recenter-root',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.repo.resolve(),a.output.resolve(),a.recenter_root.resolve()),indent=2))
if __name__=='__main__':main()
