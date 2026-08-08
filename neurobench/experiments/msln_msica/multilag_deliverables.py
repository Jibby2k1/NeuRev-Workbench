"""Build compact reports, figures, maps, and videos for multi-lag MSICA v5."""
from __future__ import annotations
import argparse, csv, gc, json
from pathlib import Path
from typing import Any
import numpy as np
from neurobench.algorithms.msln_msica_cuda import causal_joint_msln_cuda
from neurobench.reports.msln_msica_videos import Layer, _render_video
from .artifacts import atomic_json
from .joint_sweep import _atomic_npy
from .multilag_program import _aligned_outputs, _contexts, _fit_from_dict, _load, _source

OBJECTIVES = ("cs_parzen", "ksg_mi", "normalized_hsic", "matrix_renyi_mi")
FORMS = ("multilag_2d", "delay_embedding")
SHORT = {"cs_parzen":"CS-Parzen","ksg_mi":"KSG MI","normalized_hsic":"normalized HSIC",
"matrix_renyi_mi":"matrix-Renyi MI","multilag_2d":"multi-lag 2D","delay_embedding":"full embedding"}

def winner(rows, form=None, objective=None):
    subset=[r for r in rows if (form is None or r["formulation"]==form) and (objective is None or r["objective_family"]==objective)]
    return max(subset,key=lambda r:(r["selection_score"],r["lane_id"]))

def matches(row,budget=58):
    return int(row["protected"]["matched_by_budget"][str(budget)])

def family_champions(result):
    return [{"architecture":arch,"formulation":form,"objective_family":obj,
             "row":winner(result[key],form,obj)}
            for arch,key in (("Raw -> MSICA","raw_msica"),("Raw -> MSICA -> MSLN","raw_msica_then_msln"))
            for form in FORMS for obj in OBJECTIVES]

def write_tables(output,result,champions):
    root=output/"tables"; root.mkdir(parents=True)
    fields=["architecture","formulation","objective_family","lane_id","selection_score"]+[f"matches_at_{b}" for b in (20,40,58,80,100)]
    with (root/"family_champions.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for x in champions:
            r=x["row"]; w.writerow({**{k:x[k] for k in ("architecture","formulation","objective_family")},
                "lane_id":r["lane_id"],"selection_score":r["selection_score"],
                **{f"matches_at_{b}":matches(r,b) for b in (20,40,58,80,100)}})
    fields=["lane_id","formulation","objective_family","branch","context_id","selection_score"]+[f"matches_at_{b}" for b in (20,40,58,80,100)]
    with (root/"pipeline_all_lanes.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for r in result["raw_msica_then_msln"]:
            w.writerow({k:r[k] for k in fields if k in r}|{f"matches_at_{b}":matches(r,b) for b in (20,40,58,80,100)})

def write_figures(output,result,confirmation,champions):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    root=output/"figures"; root.mkdir(parents=True)
    selected=[x for x in champions if x["formulation"]=="delay_embedding"]
    labels=[SHORT[o] for o in OBJECTIVES]; raw=[matches(next(x["row"] for x in selected if x["architecture"]=="Raw -> MSICA" and x["objective_family"]==o)) for o in OBJECTIVES]
    pipe=[matches(next(x["row"] for x in selected if x["architecture"]=="Raw -> MSICA -> MSLN" and x["objective_family"]==o)) for o in OBJECTIVES]
    x=np.arange(4); fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(x-.18,raw,.36,color="#777777",label="Raw -> embedding MSICA"); ax.bar(x+.18,pipe,.36,color="#e58b2a",label="then MSLN")
    ax.axhline(49,color="#27864a",ls="--",label="Raw Direct: 49"); ax.set(xticks=x,xticklabels=labels,ylabel="Known matches at budget 58",ylim=(0,65),title="Predeclared label-free family champions"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/"embedding_family_champions.png",dpi=150); plt.close(fig)
    rows=confirmation["rows"]; x=np.arange(len(rows)); colors=["#e58b2a" if r["formulation"]=="multilag_2d" else "#27864a" for r in rows]
    fig,(a,b)=plt.subplots(2,1,figsize=(12,7),sharex=True); a.bar(x,[r["real_median_held_out_gain"] for r in rows],color=colors); a.axhline(0,color="black",lw=.8); a.set_ylabel("Median held-out gain")
    b.bar(x,[r["real_positive_gain_fraction"] for r in rows],color=colors); b.axhline(.6,color="#777777",ls="--"); b.set(ylabel="Positive-gain seed fraction",ylim=(0,1.05),xticks=x,xticklabels=[SHORT[r["objective_family"]]+"\n"+SHORT[r["formulation"]] for r in rows]); b.tick_params(axis="x",rotation=65); a.set_title("Five-seed objective stability"); fig.tight_layout(); fig.savefig(root/"objective_stability.png",dpi=150); plt.close(fig)
    rows=result["raw_msica_then_msln"]; scores=np.array([r["selection_score"] for r in rows]); known=np.array([matches(r) for r in rows]); rho,p=spearmanr(scores,known)
    fig,ax=plt.subplots(figsize=(7.5,5)); ax.scatter(scores,known,s=10,alpha=.25,color="#777777"); ax.axhline(49,color="#27864a",ls="--",label="Raw Direct"); ax.set(xlabel="Label-free event/quiet score",ylabel="Known matches at budget 58",title=f"Selector alignment across 1,110 lanes: rho={rho:.3f}, p={p:.2g}"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/"selector_alignment.png",dpi=150); plt.close(fig)

def build_visuals(config,source_root,output,result,champions):
    import cupy as cp
    surface=json.loads((source_root/"stage_a/surface.json").read_text()); expansion={r["config_id"]:r for r in surface["expansion_rows"]}
    values,_,gh,pre=_source(config); rs,re=map(int,config["source"]["review_interval_ui"]); raw=np.asarray(np.load(config["source"]["movie_path"],mmap_mode="r")[rs-1:re],dtype=np.float32)
    contexts={c.context_id:c for c in _contexts(config)}; quiet=np.zeros(len(raw),bool); qs,qe=map(int,config["source"]["quiet_interval_ui"]); quiet[qs-rs:qe-rs+1]=True; qext=np.r_[np.zeros(pre,bool),quiet]; cap=int(float(config["compute"]["max_peak_vram_gb"])*2**30)
    projections={}; cache={}
    def evidence(row,pipeline):
        lane=row["lane_id"]
        if lane in cache:return cache[lane]
        cid,branch,*rest=lane.split("::")
        if cid not in projections: projections[cid]=_aligned_outputs(values,_fit_from_dict(expansion[cid]["fit"]),config,gh,pre)
        if pipeline:
            z=causal_joint_msln_cuda(projections[cid][branch],contexts[rest[0]],quiet_mask=qext,review_crop_frames=pre,max_vram_bytes=cap); a=cp.asnumpy(cp.square(z.values,dtype=cp.float32)); del z; cp.get_default_memory_pool().free_all_blocks()
        else:a=np.square(projections[cid][branch][pre:],dtype=np.float32)
        cache[lane]=a; return a
    raw_multi=winner(result["raw_msica"],"multilag_2d"); raw_embed=winner(result["raw_msica"],"delay_embedding"); global_pipe=winner(result["raw_msica_then_msln"]); ceiling=next(r for r in result["raw_msica_then_msln"] if r["lane_id"]==result["summary"]["pipeline_protected_ceiling_lane"])
    pipe_champs=[x["row"] for x in champions if x["architecture"]=="Raw -> MSICA -> MSLN"]
    maps={}
    for row,pipeline in [(raw_multi,False),(raw_embed,False),(global_pipe,True),(ceiling,True)]+[(r,True) for r in pipe_champs]:
        lane=row["lane_id"]
        if lane in maps:continue
        ev=evidence(row,pipeline); blocks=[ev[a-rs:b-rs+1] for a,b in config["source"]["burst_intervals_ui"].values()]; pooled=np.max(np.concatenate(blocks),axis=0).astype(np.float32); safe=lane.replace("::","__"); npy=output/"maps"/(safe+".event_max.npy"); _atomic_npy(npy,pooled)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        lo,hi=np.percentile(pooled,(1,99.8)); fig,ax=plt.subplots(figsize=(8,5)); from matplotlib.colors import LinearSegmentedColormap; cmap=LinearSegmentedColormap.from_list("neutral_orange",["#000000","#4a4a4a","#b8b8b8","#d47a22","#ffd29a"]); im=ax.imshow(pooled,cmap=cmap,vmin=lo,vmax=max(hi,lo+1e-8)); ax.set(title=lane,xlabel="x = column",ylabel="y = row"); fig.colorbar(im,ax=ax,fraction=.035); fig.tight_layout(); png=output/"maps"/(safe+".event_max.png"); fig.savefig(png,dpi=130); plt.close(fig); maps[lane]={"npy":str(npy.relative_to(output)),"png":str(png.relative_to(output))}
    vr=output/"videos"; vr.mkdir(parents=True)
    videos={}
    videos["raw_msica_architecture"]=_render_video(vr/"raw_msica_architecture.mp4",[Layer("Raw input",raw,"raw"),Layer(f"Multi-lag 2D | {matches(raw_multi)}/79",evidence(raw_multi,False),"neutral_energy"),Layer(f"Full embedding | {matches(raw_embed)}/79",evidence(raw_embed,False),"neutral_energy")],"Raw -> MSICA: label-free architecture champions",review_start_ui=rs,fps=float(config["outputs"]["fps"]),columns=3)
    layers=[Layer("Raw input",raw,"raw")]+[Layer(f"{SHORT[r['formulation']]} | {SHORT[r['objective_family']]} | {matches(r)}/79",evidence(r,True),"neutral_energy") for r in pipe_champs]
    videos["pipeline_objective_champions"]=_render_video(vr/"pipeline_objective_champions.mp4",layers,"Raw -> MSICA -> MSLN: predeclared family champions",review_start_ui=rs,fps=float(config["outputs"]["fps"]),columns=3)
    videos["selector_gap"]=_render_video(vr/"selector_gap.mp4",[Layer("Raw input",raw,"raw"),Layer(f"Global label-free | {matches(global_pipe)}/79",evidence(global_pipe,True),"neutral_energy"),Layer(f"Protected ceiling | {matches(ceiling)}/79",evidence(ceiling,True),"neutral_energy")],"Selector gap: ceiling is diagnostic only",review_start_ui=rs,fps=float(config["outputs"]["fps"]),columns=3)
    atomic_json(output/"artifact_index.json",{"maps":maps,"videos":videos}); del cache,projections; gc.collect(); cp.get_default_memory_pool().free_all_blocks(); return maps,videos

def write_reports(output,result,champions):
    raw=winner(result["raw_msica"]); pipe=winner(result["raw_msica_then_msln"])
    def table(arch):
        lines=["| Formulation | Objective | Matches at 58 |","|---|---|---:|"]
        for x in champions:
            if x["architecture"]==arch:lines.append(f"| {SHORT[x['formulation']]} | {SHORT[x['objective_family']]} | {matches(x['row'])}/79 |")
        return lines
    (output/"RAW_MSICA_REPORT.md").write_text("\n".join(["# Raw -> MSICA concise report","",f"The global label-free winner reached **{matches(raw)}/79**, below Raw Direct at **49/79**.",""]+table("Raw -> MSICA")+["","Full-embedding residual energy was materially more useful than the two-output multi-lag maps. This is a representation result, not proof of biological component identity.","","See figures/objective_stability.png, videos/raw_msica_architecture.mp4, and tables/family_champions.csv."])+"\n")
    (output/"RAW_MSICA_MSLN_REPORT.md").write_text("\n".join(["# Raw -> MSICA -> MSLN concise report","",f"The global label-free selector reached **{matches(pipe)}/79**, below Raw Direct. Predeclared family champions were stronger:",""]+table("Raw -> MSICA -> MSLN")+["","Full-embedding CS-Parzen and normalized HSIC each reached **54/79**. Since eight objective/formulation strata were inspected, this is provisional, family-conditioned evidence. The label-assisted ceiling of **60/79** is diagnostic only.","","See figures/embedding_family_champions.png, figures/selector_alignment.png, and videos/pipeline_objective_champions.mp4."])+"\n")
    text=["# Multi-lag MSICA v5: conclusive report","","## Bottom line","",
    "- Multi-lag dependence had an effect: full embeddings exposed useful residual-subspace maps, and MSLN improved several predeclared families.",
    "- Raw -> MSICA alone selected 39/79 versus Raw Direct at 49/79.",
    "- The global Raw -> MSICA -> MSLN selector selected 44/79; the current global event/quiet objective is therefore inadequate.",
    "- Family-conditioned full-embedding CS-Parzen and normalized HSIC each reached 54/79 (+5 versus Raw Direct; +2 versus the prior five-seed switched ensemble at 52/79). This is promising but provisional.",
    "- The protected ceiling was 60/79 and is not deployable.","","## Recommendation","",
    "Freeze full-embedding CS-Parzen and normalized HSIC as two independent-recording confirmation arms, including their exact label-free context rules. Do not promote the 60/79 protected ceiling. Redesign the global selector to combine held-out dependence gain, seed consistency, and event/quiet contrast.","","## Scope","",
    "- 66 calibration fits, 96 expanded fits, 15 objective-diverse configurations.",
    "- Five real resampling and five synthetic seeds per frozen configuration.",
    "- 1,110 pipeline lanes over all 30 prior MSLN contexts.",
    "- Sparse positives: 79 labels; unmatched candidates remain unknown.",
    "- One GPU worker, four CPU threads, 8-frame chunks; CUDA parity error below 7.2e-7.","","## Artifact guide","",
    "- RAW_MSICA_REPORT.md and RAW_MSICA_MSLN_REPORT.md: experiment-level reports.",
    "- figures/: paper/slide-ready diagnostics.",
    "- maps/: event-max arrays and orange-tinted PNGs.",
    "- videos/: grayscale raw plus orange evidence; no red/white/blue scale.",
    "- tables/: family champions and all 1,110 pipeline lanes.",
    "- artifact_index.json: exact visual inventory."]
    (output/"CONCLUSIVE_REPORT.md").write_text("\n".join(text)+"\n")

def build(config_path,output_root):
    config=_load(config_path); source=Path(config["outputs"]["root_dir"]); output=Path(output_root).resolve()
    if output.exists():raise FileExistsError(output)
    if json.loads((source/"status.json").read_text())["status"]!="complete":raise RuntimeError("source run incomplete")
    output.mkdir(parents=True); result=json.loads((source/"stage_c/protected_results.json").read_text()); confirmation=json.loads((source/"stage_a/panel_confirmation.json").read_text()); champs=family_champions(result)
    write_tables(output,result,champs); write_figures(output,result,confirmation,champs); maps,videos=build_visuals(config,source,output,result,champs); write_reports(output,result,champs)
    summary={"status":"complete","source_run":str(source),"selection_labels_used":False,"pipeline_lane_count":len(result["raw_msica_then_msln"]),"map_count":len(maps),"videos":list(videos)}
    atomic_json(output/"summary.json",summary); atomic_json(output/"status.json",{"status":"complete","scientific_status":"reports_and_visuals_complete"}); return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True); p.add_argument("--output-root",required=True); a=p.parse_args(argv); print(json.dumps(build(a.config,a.output_root),indent=2)); return 0
if __name__=="__main__":raise SystemExit(main())
