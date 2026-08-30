"""Expanded non-RL automated validation for canonical-v8 features."""
from __future__ import annotations
import argparse,csv,json,os
from collections import Counter,defaultdict
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/neurev-expanded-validation-mpl')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neurobench.experiments.neuron_identifiability.full_trace_feature_panel import atomic_json,sha256,write_tsv

SEED=20260828
FEATURES=['auc__raw_center','auc__ica_center','auc__ls_center','auc__raw_center_annulus','auc__vst_center_annulus','auc__matched_filter_ls','auc__carrier_signed','auc__coherence_w15','auc__propagation_lag2_w15','auc__representation_consensus','auc__multiscale_persistence']
def read(path):
 with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f,delimiter='\t'))
def coverage_risk(y,p,coverages=(.25,.5,.75,1.0)):
 confidence=np.abs(p-.5);order=np.argsort(-confidence);out=[]
 for c in coverages:
  n=max(1,int(np.ceil(len(y)*c)));ix=order[:n];out.append({'coverage':c,'n':n,'error_rate':float(np.mean((p[ix]>=.5)!=y[ix])),'brier':float(brier_score_loss(y[ix],p[ix]))})
 return out
def grouped_predictions(x,y,groups,cols):
 pred=np.full(len(y),np.nan);coef=[]
 for tr,te in GroupKFold(5).split(x[:,cols],y,groups):
  model=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',max_iter=5000,random_state=SEED));model.fit(x[tr][:,cols],y[tr]);pred[te]=model.predict_proba(x[te][:,cols])[:,1];coef.append(model[-1].coef_[0])
 return pred,np.asarray(coef)
def trace_stats(x):
 x=np.asarray(x,float);base=x[:40];event=x[40:80];mad=1.4826*np.median(np.abs(base-np.median(base)));return {'peak_mad':float((event.max()-np.median(base))/max(mad,1e-9)),'lag2':float(np.corrcoef(x[42:80],x[40:78])[0,1])}
def run(repo,out):
 if out.exists():raise FileExistsError(out)
 part=out.with_name(out.name+'.partial');part.mkdir(parents=True)
 source=repo/'Outputs/NeuronIdentifiability/spon_ca_burst_automated_feature_validation_v1/tables/recovery_model_predictions.tsv';rows=read(source);x=np.asarray([[float(r[c]) for c in FEATURES] for r in rows]);y=np.asarray([int(r['recovered_any_b58']) for r in rows]);groups=np.asarray([r['site_id'] for r in rows]);bursts=np.asarray([int(r['burst_id']) for r in rows]);rng=np.random.default_rng(SEED)
 # 1-2 identity-held-out prediction and baseline
 full_pred,coef=grouped_predictions(x,y,groups,list(range(len(FEATURES))));base_pred,_=grouped_predictions(x,y,groups,[FEATURES.index('auc__carrier_signed')]);identity_rows=[{'observation_id':r['observation_id'],'site_id':r['site_id'],'burst_id':r['burst_id'],'recovered':int(y[i]),'prediction_full':float(full_pred[i]),'prediction_carrier':float(base_pred[i])} for i,r in enumerate(rows)];write_tsv(part/'identity_grouped_predictions.tsv',identity_rows)
 # 3 stability selection: site bootstraps, L1 signs
 sites=np.unique(groups);stable=[];selected=Counter();positive=Counter();fits=100
 for k in range(fits):
  sampled=rng.choice(sites,len(sites),replace=True);ix=np.concatenate([np.where(groups==s)[0] for s in sampled]);m=make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',solver='liblinear',C=.2,class_weight='balanced',max_iter=3000,random_state=SEED+k));m.fit(x[ix],y[ix]);c=m[-1].coef_[0]
  for j,v in enumerate(c):selected[FEATURES[j]]+=abs(v)>1e-12;positive[FEATURES[j]]+=v>0
 for f in FEATURES:stable.append({'feature':f,'bootstrap_fits':fits,'selection_fraction':selected[f]/fits,'positive_fraction':positive[f]/fits});write_tsv(part/'stability_selection.tsv',stable)
 # 4 conditional importance by grouped drop-column and within-burst permutation
 base_loss=log_loss(y,full_pred);importance=[]
 for j,f in enumerate(FEATURES):
  drop,_=grouped_predictions(x,y,groups,[q for q in range(len(FEATURES)) if q!=j]);xp=x.copy()
  for b in np.unique(bursts):
   ix=np.where(bursts==b)[0];xp[ix,j]=rng.permutation(xp[ix,j])
  pp,_=grouped_predictions(xp,y,groups,list(range(len(FEATURES))))
  importance.append({'feature':f,'drop_column_log_loss_increase':float(log_loss(y,drop)-base_loss),'within_burst_permutation_loss_increase':float(log_loss(y,pp)-base_loss)})
 write_tsv(part/'conditional_importance.tsv',importance)
 # 5 prospective footprint and cross-burst evidence are reused, not recomputed
 prior=repo/'Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_validation_suite_v2';transfer=read(prior/'cross_burst_transfer.tsv');frozen=read(prior/'frozen_footprint.tsv');write_tsv(part/'prospective_transfer_reuse.tsv',[{'source':'cross_burst_transfer','rows':len(transfer),'clear_rows':sum(r['outcome']=='clear' for r in transfer)},{'source':'pre_event_frozen_footprint','rows':len(frozen),'clear_rows':sum(r['outcome']=='clear' for r in frozen)}])
 # 6 controlled mixtures with exact source truth
 mix=[];t=np.arange(120);target=np.exp(-np.maximum(t-40,0)/18)*(t>=40);target/=target.max();neighbor=np.exp(-np.maximum(t-45,0)/22)*(t>=45);neighbor/=neighbor.max()
 for corrmix in (0,.25,.5,.75,1):
  for amp in (.25,.5,1,2,4):
   for noise in (.1,.3,.6):
    z=target+amp*(corrmix*target+(1-corrmix)*neighbor)+rng.normal(0,noise,len(t));s=trace_stats(z);mix.append({'shared_fraction':corrmix,'neighbor_amplitude':amp,'noise_sd':noise,**s})
 write_tsv(part/'synthetic_source_mixtures.tsv',mix)
 # 7 metamorphic invariance/direction checks
 meta=[];fixed_noise=rng.normal(0,.2,len(t))
 for scale in (.5,1,2,4):
  z=scale*(target+fixed_noise);meta.append({'transform':'intensity_scale','level':scale,**trace_stats(z)})
 for common in (0,.25,.5,1,2):
  z=target+common*np.sin(t/15)+fixed_noise;meta.append({'transform':'common_mode','level':common,**trace_stats(z)})
 for shift in (-8,-4,0,4,8):meta.append({'transform':'temporal_shift','level':shift,**trace_stats(np.roll(target,shift)+fixed_noise)})
 write_tsv(part/'metamorphic_tests.tsv',meta)
 # 8 calibration and selective prediction
 calibration=coverage_risk(y,full_pred);write_tsv(part/'calibration_abstention.tsv',calibration)
 # 9 recovered-only unsupervised anomaly detectors
 scaler=StandardScaler().fit(x[y==1]);z=scaler.transform(x);mcd=MinCovDet(random_state=SEED).fit(z[y==1]);iso=IsolationForest(n_estimators=300,contamination='auto',random_state=SEED).fit(z[y==1]);mcds=mcd.mahalanobis(z);isos=-iso.score_samples(z);anomaly=[{'observation_id':r['observation_id'],'recovered':int(y[i]),'robust_distance':float(mcds[i]),'isolation_score':float(isos[i])} for i,r in enumerate(rows)];write_tsv(part/'unsupervised_anomaly.tsv',anomaly)
 # 10 spatial graph context from immutable site coordinates
 coords=np.asarray([[float(r['x_scaled']),float(r['y_scaled'])] for r in rows]);graph=[]
 for i,r in enumerate(rows):
  same=np.where(groups!=groups[i])[0];d=np.linalg.norm(coords[same]-coords[i],axis=1);near=same[np.argsort(d)[:5]];graph.append({'observation_id':r['observation_id'],'recovered':int(y[i]),'nearest_site_distance_scaled':float(d.min()),'neighbor5_recovery_fraction':float(y[near].mean()),'neighbor5_carrier_mean':float(x[near,FEATURES.index('auc__carrier_signed')].mean()),'carrier_minus_neighbor5':float(x[i,FEATURES.index('auc__carrier_signed')]-x[near,FEATURES.index('auc__carrier_signed')].mean())})
 write_tsv(part/'graph_crowding.tsv',graph)
 # compact figure
 fig,ax=plt.subplots(2,3,figsize=(14,8));top=sorted(stable,key=lambda r:-r['selection_fraction']);ax[0,0].barh([r['feature'].replace('auc__','') for r in top[:8]][::-1],[r['selection_fraction'] for r in top[:8]][::-1]);ax[0,0].set_title('Stability selection');imp=sorted(importance,key=lambda r:-r['within_burst_permutation_loss_increase']);ax[0,1].barh([r['feature'].replace('auc__','') for r in imp[:8]][::-1],[r['within_burst_permutation_loss_increase'] for r in imp[:8]][::-1]);ax[0,1].set_title('Conditional importance');ax[0,2].plot([r['coverage'] for r in calibration],[r['error_rate'] for r in calibration],marker='o');ax[0,2].set(title='Selective risk',xlabel='Coverage',ylabel='Error rate');ax[1,0].scatter(mcds,isos,c=['#52789c' if q else '#c65f45' for q in y]);ax[1,0].set(title='Recovered-only anomaly scores',xlabel='Robust distance',ylabel='Isolation score');ax[1,1].scatter([float(r['nearest_site_distance_scaled']) for r in graph],[float(r['carrier_minus_neighbor5']) for r in graph],c=['#52789c' if q else '#c65f45' for q in y]);ax[1,1].set(title='Graph crowding',xlabel='Nearest distance',ylabel='Carrier - neighbors');ax[1,2].scatter([r['neighbor_amplitude'] for r in mix],[r['peak_mad'] for r in mix],c=[r['shared_fraction'] for r in mix],cmap='viridis');ax[1,2].set(title='Synthetic mixtures',xlabel='Neighbor amplitude',ylabel='Peak/MAD');fig.suptitle('Expanded automated feature validation');fig.tight_layout();fig.savefig(part/'expanded_automated_validation.png',dpi=180);plt.close(fig)
 summary={'schema_version':1,'status':'complete_exploratory','population':{'occurrences':len(y),'sites':len(sites),'recovered':int(y.sum()),'missed':int((1-y).sum())},'identity_grouped':{'full_log_loss':float(base_loss),'carrier_log_loss':float(log_loss(y,base_pred)),'full_auc':float(roc_auc_score(y,full_pred)),'carrier_auc':float(roc_auc_score(y,base_pred))},'stability_top':top[:5],'importance_top':imp[:5],'calibration':calibration,'anomaly_auc':{'robust_distance':float(roc_auc_score(1-y,mcds)),'isolation':float(roc_auc_score(1-y,isos))},'graph_association':float(np.corrcoef(y,[float(r['neighbor5_recovery_fraction']) for r in graph])[0,1]),'prospective_transfer_boundary':{'cross_burst_clear_pairs':sum(r['outcome']=='clear' for r in transfer),'pre_event_clear':sum(r['outcome']=='clear' for r in frozen)},'limits':['same recording and reused labels','site grouped but not independent animal validation','only twelve misses','synthetic mixtures are one-dimensional','unmatched candidates remain unknown']};atomic_json(part/'summary.json',summary)
 checks={'ten_families_present':bool(all((part/n).is_file() for n in ['identity_grouped_predictions.tsv','stability_selection.tsv','conditional_importance.tsv','prospective_transfer_reuse.tsv','synthetic_source_mixtures.tsv','metamorphic_tests.tsv','calibration_abstention.tsv','unsupervised_anomaly.tsv','graph_crowding.tsv'])),'grouped_predictions_complete':bool(np.isfinite(full_pred).all()),'stability_100':bool(all(r['bootstrap_fits']==100 for r in stable)),'mixture_grid_75':bool(len(mix)==75),'metamorphic_14':bool(len(meta)==14),'source_audit_reused':bool((prior/'validation.json').is_file())};atomic_json(part/'validation.json',{'status':'passed' if all(checks.values()) else 'failed','checks':checks});atomic_json(part/'llm_context.json',{'entry_point':'summary.json','figure':'expanded_automated_validation.png','tables':[p.name for p in part.glob('*.tsv')],'scientific_audit':{'mode':'validated_source_audit_reuse','source':str(prior),'reason':'Frozen secondary analyses; no new candidates or labels.'},'limits':summary['limits']});(part/'REPORT.md').write_text('# Expanded automated validation\n\nTen non-RL test families spanning grouped prediction, stability, conditional importance, prospective transfer, synthetic mixtures, metamorphic tests, calibration, anomaly detection, and graph crowding.\n');arts=[{'path':str(p.relative_to(part)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(part.rglob('*')) if p.is_file() and p.name!='artifact_index.json'];atomic_json(part/'artifact_index.json',{'artifacts':arts,'source_sha256':sha256(source)});part.replace(out);return summary
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.repo.resolve(),a.output.resolve()),indent=2))
if __name__=='__main__':main()
