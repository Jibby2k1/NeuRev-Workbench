"""Collision-safe sequential runner; Milestones 1-2 implement S0 only."""
from __future__ import annotations
import hashlib, json, os, resource, subprocess, time
from pathlib import Path
from typing import Any
import numpy as np
from neurobench.experiments.frame_difference import _atomic_json
from .config import LearnedOperatorConfig
from .data import load_and_validate_labels, raw_direct_stack, sha256_file
from .decisions import Stage, initial_program_state, s0_decision
from .evaluation import evaluate_stack
from .preflight import matching_preflight
from .report import write_s0_artifacts

def _git_commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: return "unavailable"

def _config_hash(config: LearnedOperatorConfig) -> str:
    return hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def run_stage(config: LearnedOperatorConfig, *, preflight_dir: str|Path, stage: Stage, diagnostic_rescue: bool=False, **_: Any) -> dict[str,Any]:
    smoke=bool(_.get("smoke",False))
    if stage==Stage.S1_OPERATOR_SCREEN and smoke:
        return run_s1_synthetic(config,preflight_dir=preflight_dir)
    if stage==Stage.S1_OPERATOR_SCREEN:
        matching_preflight(config,preflight_dir,stage)
        from .screen import run_screen
        result=run_screen(config,prefix_size=int(_.get("prefix_size") or 8),diagnostic_rescue=diagnostic_rescue); state_path=config.output_dir/"program_state.json"; state=json.loads(state_path.read_text()); state["stages"][stage.value].update({"state":result["decision"]["decision"],"decision_path":str((Path(result["root"])/"decision.json").relative_to(config.output_dir)),"rescue_consumed":diagnostic_rescue}); _atomic_json(state_path,state); return {"stage":stage.value,"decision":result["decision"]["decision"],"artifact_dir":result["root"],"families":result["summary"]["families"]}
    if stage!=Stage.S0_BASELINE: raise NotImplementedError("full S1 response screening is not implemented until S1A and its reviewed preflight pass")
    matching_preflight(config,preflight_dir,stage)
    root=config.output_dir
    if root.exists(): raise FileExistsError(f"program output directory exists: {root}")
    root.mkdir(parents=True,exist_ok=False); started=time.monotonic(); state=initial_program_state(config.experiment_id); state["stages"][stage.value]["state"]="running"; _atomic_json(root/"program_state.json",state)
    try:
        source=np.load(config.source_video,mmap_mode="r",allow_pickle=False); labels=load_and_validate_labels(config.labels_tsv,config.label_summary,tuple(source.shape[1:]))
        stack,normalization=raw_direct_stack(source,config.frames.review_start_ui,config.frames.review_end_ui,config.frames.quiet_end_ui)
        metrics=evaluate_stack(stack,labels,config); repeated=evaluate_stack(stack,labels,config)
        e=config.evaluation; tolerance=e.exact_tolerance
        expected={"macro_kpr_at_58":e.expected_fixed_macro_kpr,"matches_at_58":e.expected_fixed_matches,"macro_kpr_q1":e.expected_q1_macro_kpr,"matches_q1":e.expected_q1_matches,"candidates_q1":e.expected_q1_candidates}
        invariants={"fixed_macro_exact":abs(metrics["macro_kpr_at_58"]-e.expected_fixed_macro_kpr)<=tolerance,"fixed_matches_exact":metrics["matches_at_58"]==e.expected_fixed_matches,"q1_macro_exact":abs(metrics["macro_kpr_q1"]-e.expected_q1_macro_kpr)<=tolerance,"q1_matches_exact":metrics["matches_q1"]==e.expected_q1_matches,"q1_candidates_exact":metrics["candidates_q1"]==e.expected_q1_candidates,"candidate_ranking_deterministic":metrics["top58_candidate_hash"]==repeated["top58_candidate_hash"],"finite_outputs":metrics["finite_output_fraction"]==1.0,"labels_exact":metrics["labels"]==79}
        summary={**metrics,"stage":stage.value,"method":"raw_direct","expected":expected,"invariants":invariants,"burst_delta_vector":[0.0]*4,"normalization":normalization,"trace_preserve":1.0,"numerical_health":{"finite_output_fraction":metrics["finite_output_fraction"],"max_condition_number":1.0,"effective_rank_fraction":1.0,"converged":True},"screening_audit_opt_out":config.scientific_audit.opt_out_reason,"current_video_limitation":"within-video evidence only"}
        decision=s0_decision(summary,rescue_consumed=diagnostic_rescue); runtime=time.monotonic()-started; peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        fingerprint=hashlib.sha256((sha256_file(config.source_video)+sha256_file(config.labels_tsv)).encode()).hexdigest(); commit=_git_commit(); chash=_config_hash(config)
        rows=[{"run_id":f"raw_direct_burst_{row['burst_id']}","stage":stage.value,"method":"raw_direct","operator_family":"raw_direct","design_id":"explicit_anchor","outer_fold":row["burst_id"],"inner_fold":"","seed":"","status":"completed","stop_reason":"","heldout_burst_id":row["burst_id"],"heldout_labels":row["labels"],"heldout_matches_at_58":row["matches_at_58"],"heldout_recall_at_58":row["recall_at_58"],"heldout_matches_q1":row["matches_q1"],"heldout_candidates_q1":row["candidates_q1"],"trace_preserve":1.0,"max_condition_number":1.0,"effective_rank_fraction":1.0,"converged":True,"best_epoch":"","runtime_seconds":runtime,"peak_rss_mib":peak,"peak_vram_mib":0,"parameter_file":"","config_hash":chash,"git_commit":commit,"data_fingerprint":fingerprint} for row in metrics["outer_folds"]]
        write_s0_artifacts(root,summary,decision,rows,config.to_dict()); state["stages"][stage.value].update({"state":decision["decision"],"rescue_consumed":diagnostic_rescue,"decision_path":"decision.json"}); _atomic_json(root/"program_state.json",state); _atomic_json(root/"resource_summary.json",{"runtime_seconds":runtime,"peak_rss_mib":peak,"peak_vram_mib":0})
        return {"stage":stage.value,"decision":decision["decision"],"program_dir":str(root),"macro_kpr_at_58":metrics["macro_kpr_at_58"],"macro_kpr_q1":metrics["macro_kpr_q1"]}
    except Exception as exc:
        state["stages"][stage.value]["state"]="failed_operationally"; _atomic_json(root/"program_state.json",state); _atomic_json(root/"failure.json",{"stage":stage.value,"error":repr(exc)}); raise

def run_s1_synthetic(config: LearnedOperatorConfig, *, preflight_dir: str|Path) -> dict[str,Any]:
    matching_preflight(config,preflight_dir,Stage.S1_OPERATOR_SCREEN)
    root=config.output_dir; state_path=root/"program_state.json"
    if not state_path.is_file(): raise RuntimeError("S1 requires the completed S0 program state")
    state=json.loads(state_path.read_text()); s0=state["stages"][Stage.S0_BASELINE.value]
    if s0.get("state") not in {"advance","advance_with_caution"}: raise RuntimeError("S0 decision does not permit S1")
    destination=root/"stages"/Stage.S1_OPERATOR_SCREEN.value/"synthetic"
    if (destination/"decision.json").is_file(): raise FileExistsError(f"completed S1A synthetic root exists: {destination}")
    from .operators import synthetic_validation
    result=synthetic_validation(); destination.mkdir(parents=True,exist_ok=True)
    decision={"stage":"S1A_SYNTHETIC","decision":"advance" if result["passed"] else "stop_program","numerical_health":"pass" if result["passed"] else "fail","reason_codes":[name for name,value in result["checks"].items() if not value],"next_allowed_stages":[Stage.S1_OPERATOR_SCREEN.value] if result["passed"] else []}
    _atomic_json(destination/"stage_summary.json",result); _atomic_json(destination/"decision.json",decision); _atomic_json(destination/"resolved_config.json",config.to_dict())
    state["stages"][Stage.S1_OPERATOR_SCREEN.value].update({"state":"preflight_ready" if result["passed"] else "stop_program","s1a_decision_path":str((destination/"decision.json").relative_to(root)),"rescue_consumed":False}); _atomic_json(state_path,state)
    return {"stage":"S1A_SYNTHETIC","decision":decision["decision"],"artifact_dir":str(destination),"checks":result["checks"]}

def status(program_dir: str|Path) -> dict[str,Any]:
    root=Path(program_dir).expanduser().resolve(); state=json.loads((root/"program_state.json").read_text()); result={"program_dir":str(root),"program_state":state}
    for name in ("decision.json","stage_summary.json","resource_summary.json"):
        if (root/name).is_file(): result[name.removesuffix(".json")]=json.loads((root/name).read_text())
    return result
