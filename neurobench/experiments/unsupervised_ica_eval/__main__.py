"""Bounded label-free Stage-A runner for the canonical two-frame experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .core import (analytic_baselines, distribution_summary, fit_two_frame_ica,
                   representation_fingerprint, stability_summary, temporal_block_shuffle)


def _write_json(path: Path, value: object) -> None:
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    temporary.replace(path)


def _rank(values: np.ndarray) -> np.ndarray:
    order=np.argsort(values,kind="mergesort"); ranks=np.empty(len(values),dtype=float); ranks[order]=np.arange(len(values))
    return ranks


def _comparison(a: np.ndarray,b: np.ndarray,top_fraction: float) -> dict[str,float]:
    pearson=float(np.corrcoef(a,b)[0,1]); spearman=float(np.corrcoef(_rank(a),_rank(b))[0,1])
    design=np.column_stack((np.ones(len(b)),b)); predicted=design@np.linalg.lstsq(design,a,rcond=None)[0]
    r2=1-float(np.sum((a-predicted)**2))/max(float(np.sum((a-a.mean())**2)),np.finfo(float).eps)
    k=max(1,int(len(a)*top_fraction)); ia=set(np.argpartition(np.abs(a),-k)[-k:]); ib=set(np.argpartition(np.abs(b),-k)[-k:])
    return {"pearson":pearson,"spearman":spearman,"r_squared":r2,"top_activation_jaccard":len(ia&ib)/len(ia|ib)}


def _draw(output: Path, fit, baselines: dict[str,np.ndarray], selected: int) -> None:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    directions=fit.effective_directions/np.linalg.norm(fit.effective_directions,axis=1,keepdims=True)
    for index,direction in enumerate(directions):
        axes[0].arrow(0,0,direction[0],direction[1],head_width=.04,length_includes_head=True,label=f"ICA {index}")
        axes[0].text(direction[0],direction[1],f" ICA {index}")
    axes[0].plot([0,-1/np.sqrt(2)],[0,1/np.sqrt(2)],"k--",label="difference")
    axes[0].set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),aspect="equal",title="Effective observation-space directions",xlabel="I(t)",ylabel="I(t+1)")
    response=fit.activations[:,selected]
    axes[1].hist(response,bins=80,density=True,alpha=.65,label=f"ICA {selected}")
    axes[1].hist(baselines["difference_standardized"],bins=80,density=True,histtype="step",label="standardized difference")
    axes[1].set(title="Response distributions",xlabel="response",ylabel="density"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output/"operator_identification.png",dpi=180); plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-npy",type=Path,required=True,help="Memory-mappable [time,y,x] video")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--start",type=int,default=0); parser.add_argument("--stop",type=int)
    parser.add_argument("--spatial-stride",type=int,default=4); parser.add_argument("--max-samples",type=int,default=250000)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4]); parser.add_argument("--top-fraction",type=float,default=.01)
    args=parser.parse_args()
    if args.output_dir.exists(): raise SystemExit(f"refusing existing output: {args.output_dir}")
    video=np.load(args.video_npy,mmap_mode="r",allow_pickle=False)
    stop=len(video) if args.stop is None else args.stop
    if video.ndim != 3 or not (0 <= args.start < stop-1 < len(video)): raise SystemExit("invalid [time,y,x] video or frame interval")
    pairs=np.stack((np.asarray(video[args.start:stop-1,args.spatial_stride//2::args.spatial_stride,args.spatial_stride//2::args.spatial_stride]).reshape(-1),
                    np.asarray(video[args.start+1:stop,args.spatial_stride//2::args.spatial_stride,args.spatial_stride//2::args.spatial_stride]).reshape(-1)),axis=1)
    if len(pairs)>args.max_samples:
        indexes=np.random.default_rng(20260820).choice(len(pairs),args.max_samples,replace=False); pairs=pairs[indexes]
    args.output_dir.mkdir(parents=True)
    fits=[fit_two_frame_ica(pairs,seed=seed) for seed in args.seeds]
    distributions=[distribution_summary(fits[0].activations[:,c]) for c in range(2)]
    derivative=np.asarray([-1.0,1.0])/np.sqrt(2.0)
    normalized_directions=fits[0].effective_directions/np.linalg.norm(fits[0].effective_directions,axis=1,keepdims=True)
    selected=int(np.argmax(np.abs(normalized_directions@derivative)))
    baselines=analytic_baselines(pairs,random_seed=20260820)
    null_pairs=temporal_block_shuffle(pairs,block_size=max(8,min(1024,len(pairs)//20)),seed=20260820)
    null_fit=fit_two_frame_ica(null_pairs,seed=args.seeds[0])
    selection={"criterion":"maximum_absolute_observation_space_cosine_to_signed_temporal_difference",
               "selected_component":selected,"component_set_frozen":True,"top_fraction":args.top_fraction,
               "labels_used":False,"frame_interval_zero_based_half_open":[args.start,stop],"spatial_stride":args.spatial_stride,"sample_count":len(pairs)}
    fingerprint=representation_fingerprint(fits[0],selection)
    summary={"schema_version":1,"status":"label_free_representation_frozen","source":str(args.video_npy.resolve()),
             "selection":selection,"representation_sha256":fingerprint,"fit":fits[0].frozen_dict(),
             "component_distributions":distributions,"stability":stability_summary(fits,top_fraction=args.top_fraction),
             "analytic_comparisons":{name:_comparison(fits[0].activations[:,selected],values,args.top_fraction) for name,values in baselines.items()},
             "null":{f"component_{c}":distribution_summary(null_fit.activations[:,c]) for c in range(2)},
             "interpretation":"Stage A operator-identification evidence only; no event labels or biological claims."}
    _write_json(args.output_dir/"summary.json",summary); _write_json(args.output_dir/"frozen_representation.json",{"sha256":fingerprint,"fit":fits[0].frozen_dict(),"selection":selection})
    _write_json(args.output_dir/"llm_context.json",{"status":summary["status"],"representation_sha256":fingerprint,"primary_metrics":"summary.json","primary_figure":"operator_identification.png","labels_used":False})
    _draw(args.output_dir,fits[0],baselines,selected)
    (args.output_dir/"REPORT.md").write_text(f"# Canonical two-frame ICA Stage A\n\nRepresentation `{fingerprint}` was frozen without labels. Component {selected} was identified by observation-space cosine to signed temporal difference. See `summary.json` and `operator_identification.png`. External label evaluation has not run.\n",encoding="utf-8")
    print(json.dumps({"output_dir":str(args.output_dir),"representation_sha256":fingerprint,"selected_component":selected},indent=2))


if __name__ == "__main__": main()
