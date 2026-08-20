"""Read-only guarded preflight for one learned-operator program stage."""
from __future__ import annotations
import json, os, shlex, shutil
from pathlib import Path
from typing import Any
import numpy as np
from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib
from .config import LearnedOperatorConfig
from .data import load_and_validate_labels, outer_burst_splits, sha256_file, ui_inclusive_to_zero_half_open
from .decisions import Stage, initial_program_state, prerequisite_decision

def _overlay(video: np.ndarray, labels: list[dict[str, Any]], path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image=np.asarray(video,dtype=np.float32).mean(axis=0); low,high=np.percentile(image,[1,99.5])
    fig,ax=plt.subplots(figsize=(9,5.5),dpi=140); ax.imshow(image,cmap="gray",vmin=low,vmax=high)
    ax.scatter([r["x_px"] for r in labels],[r["y_px"] for r in labels],s=32,facecolors="none",edgecolors="cyan",linewidths=1)
    ax.set(title="Learned-operator S0 label projection",xlabel="x = column",ylabel="y = row"); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def _gpu_status() -> dict[str,Any]:
    try:
        import torch
        if not torch.cuda.is_available(): return {"available":False}
        free,total=torch.cuda.mem_get_info(); return {"available":True,"name":torch.cuda.get_device_name(0),"free_mib":int(free//2**20),"total_mib":int(total//2**20)}
    except Exception as exc: return {"available":False,"error":repr(exc)}

def _active_processes() -> list[dict[str,Any]]:
    try:
        import psutil
        rows=[]
        for process in psutil.process_iter(("pid","name","cmdline")):
            command=" ".join(process.info.get("cmdline") or [])
            if any(token in command.lower() for token in ("neurobench","python","pytest")):
                rows.append({"pid":process.info["pid"],"name":process.info.get("name") or "","command":command[:240]})
        return rows[:32]
    except Exception as exc: return [{"inspection_error":repr(exc)}]

def preflight(config: LearnedOperatorConfig, *, artifact_dir: str|Path, stage: Stage=Stage.S0_BASELINE) -> dict[str,Any]:
    destination=Path(artifact_dir).expanduser().resolve()
    if destination.exists(): raise FileExistsError(f"preflight artifact directory exists: {destination}")
    missing=[str(path) for path in (config.source_video,config.labels_tsv,config.label_summary) if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    program_exists=config.output_dir.exists()
    if stage==Stage.S0_BASELINE and program_exists: raise FileExistsError(f"program output directory exists: {config.output_dir}")
    video=np.load(config.source_video,mmap_mode="r",allow_pickle=False)
    if video.ndim!=3: raise ValueError(f"source must be TYX, got {video.shape}")
    f=config.frames; start,stop=ui_inclusive_to_zero_half_open(f.review_start_ui,f.review_end_ui)
    if stop>len(video): raise ValueError("review interval exceeds source")
    labels=load_and_validate_labels(config.labels_tsv,config.label_summary,tuple(video.shape[1:]))
    if program_exists:
        state_path=config.output_dir/"program_state.json"
        if not state_path.is_file(): raise RuntimeError("existing program root lacks program_state.json")
        state=json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state=initial_program_state(config.experiment_id)
    prerequisite=prerequisite_decision(stage,state)
    review_bytes=(stop-start)*video.shape[1]*video.shape[2]*np.dtype(np.float32).itemsize
    estimated_peak_ram_mib=int(np.ceil(review_bytes*2.5/2**20)); estimated_output_mib=2
    probe=config.output_dir.parent
    while not probe.exists(): probe=probe.parent
    disk_free_mib=shutil.disk_usage(probe).free//2**20; ram_available_mib=_available_ram_mib(); gpu=_gpu_status()
    ready=bool(prerequisite["permitted"] and ram_available_mib>=config.resources.max_ram_mib and disk_free_mib>=config.resources.min_free_disk_mib+estimated_output_mib)
    exact_command=f".venv-neurobench/bin/python -m neurobench.cli.main experiment learned-operator run-stage --config {shlex.quote(config_path_hint(config))} --preflight-dir {shlex.quote(str(destination))} --stage {stage.value}"
    payload={"schema_version":1,"experiment_id":config.experiment_id,"stage":stage.value,"ready":ready,"source":{"shape":list(video.shape),"dtype":str(video.dtype),"axes":"TYX","sha256":sha256_file(config.source_video)},"labels":{"rows":len(labels),"unique_coordinates":len({r['roi_identity'] for r in labels}),"sha256":sha256_file(config.labels_tsv),"summary_sha256":sha256_file(config.label_summary),"coordinates":"x=column,y=row"},"frames":{"review_ui_inclusive":[f.review_start_ui,f.review_end_ui],"review_zero_half_open":[start,stop],"quiet_ui_inclusive":[f.quiet_start_ui,f.quiet_end_ui]},"splits":outer_burst_splits(labels),"design":{"total_master_points":config.design.master_size,"unique_points_requested":0,"expected_fit_count":1 if stage==Stage.S0_BASELINE else None,"expected_operator_cache_mib":0 if stage==Stage.S0_BASELINE else None},"resources":{"estimated_peak_ram_mib":estimated_peak_ram_mib,"ram_available_mib":ram_available_mib,"ram_guard_mib":config.resources.max_ram_mib,"peak_vram_estimate_mib":0 if stage==Stage.S0_BASELINE else None,"gpu":gpu,"output_estimate_mib":estimated_output_mib,"output_cap_mib":config.resources.max_output_mib,"disk_free_mib":disk_free_mib,"minimum_free_disk_mib":config.resources.min_free_disk_mib,"active_processes":_active_processes()},"collisions":{"artifact_dir":False,"program_dir":False},"scientific_audit":{"enabled":config.scientific_audit.enabled,"opt_out_reason":config.scientific_audit.opt_out_reason,"screening_only":True},"prerequisite":prerequisite,"diagnostic_rescue_available":True,"exact_run_command":exact_command,"current_video_limitation":"within-video evidence only; no cross-fish or cross-recording generalization"}
    destination.mkdir(parents=True,exist_ok=False); _overlay(video[f.quiet_start_ui-1:f.quiet_end_ui],labels,destination/"label_projection_overlay.png")
    _atomic_json(destination/"preflight.json",payload); _atomic_json(destination/"resolved_config.json",config.to_dict()); _atomic_json(destination/"program_state.json",state)
    if not ready: raise RuntimeError(f"learned-operator preflight not ready: {payload['resources']}")
    return payload

def config_path_hint(config: LearnedOperatorConfig) -> str:
    return "examples/spon_ca_burst_learned_operator_selection.example.json"

def matching_preflight(config: LearnedOperatorConfig, directory: str|Path, stage: Stage) -> dict[str,Any]:
    root=Path(directory).resolve(); payload=json.loads((root/"preflight.json").read_text()); resolved=json.loads((root/"resolved_config.json").read_text())
    if not payload.get("ready") or payload.get("stage")!=stage.value or resolved!=config.to_dict(): raise RuntimeError("run requires a ready preflight for the identical config and stage")
    for key,path in (("sha256",config.source_video),("sha256",config.labels_tsv),("summary_sha256",config.label_summary)):
        section="source" if path==config.source_video else "labels"
        if payload[section][key]!=sha256_file(path): raise RuntimeError(f"input fingerprint changed: {path}")
    return payload
