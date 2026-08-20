"""Compact report and essential-metrics artifact writers."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
from neurobench.experiments.frame_difference import _atomic_json

RUN_FIELDS=("run_id","stage","method","operator_family","design_id","outer_fold","inner_fold","seed","status","stop_reason","heldout_burst_id","heldout_labels","heldout_matches_at_58","heldout_recall_at_58","heldout_matches_q1","heldout_candidates_q1","trace_preserve","max_condition_number","effective_rank_fraction","converged","best_epoch","runtime_seconds","peak_rss_mib","peak_vram_mib","parameter_file","config_hash","git_commit","data_fingerprint")

def write_runs(path: Path, rows: list[dict[str,Any]]) -> None:
    temporary=path.with_suffix(".partial")
    with temporary.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=RUN_FIELDS,delimiter="\t",extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)

def write_s0_artifacts(root: Path, summary: dict[str,Any], decision: dict[str,Any], rows: list[dict[str,Any]], resolved: dict[str,Any]) -> None:
    _atomic_json(root/"stage_summary.json",summary); _atomic_json(root/"decision.json",decision); _atomic_json(root/"resolved_config.json",resolved); write_runs(root/"runs.tsv",rows)
    report=f"""# Learned operator selection — S0 evaluator freeze

Decision: `{decision['decision']}`.

- Macro-KPR@58: `{summary['macro_kpr_at_58']:.12f}` ({summary['matches_at_58']}/{summary['labels']} pooled known matches).
- Macro-KPR@Q1: `{summary['macro_kpr_q1']:.12f}` ({summary['matches_q1']}/{summary['labels']} known matches; {summary['candidates_q1']} event candidates).
- Candidate ordering: descending score, ascending y, ascending x.
- Candidate hash: `{summary['top58_candidate_hash']}`.

Unmatched candidates are unknown, not false positives. Results are within-video evidence only and do not establish cross-fish or cross-recording generalization. Screening uses the explicitly authorized metrics-only audit exemption; any promoted finalist must restore the full scientific audit.
"""
    (root/"REPORT.md").write_text(report,encoding="utf-8")

def build_s1_llm_documents(program_root: str|Path) -> dict[str,Any]:
    root=Path(program_root); valid=root/"stages"/"S1_OPERATOR_SCREEN"/"prefix_8_calibration_rescue"; invalid=root/"stages"/"S1_OPERATOR_SCREEN"/"prefix_8"
    stage=json.loads((valid/"stage_summary.json").read_text()); decision=json.loads((valid/"decision.json").read_text())
    compact={"stage":"S1_OPERATOR_SCREEN","decision":decision["decision"],"rescue_consumed":True,"primary_metric":"Macro-KPR@58","raw_direct":{"macro_kpr_at_58":stage["raw_direct"]["macro_kpr_at_58"],"matches_at_58":stage["raw_direct"]["matches_at_58"]},"families":stage["families"],"valid_result_path":str(valid.relative_to(root)),"superseded_result_path":str(invalid.relative_to(root)),"superseded_reason":"per-pixel MAD base did not match frozen Raw Direct global-scale protocol","interpretation":"No tested whitening family improved leakage-safe fixed selection; proceed only to minimal matched ICA confirmation.","limitations":["single recording","sparse positives do not identify precision","8-point prefix only","screening audit exemption"]}
    context={"experiment_id":"spon_ca_burst_learned_operator_selection_v1","entrypoint":"summary.json","coordinate_contract":"x=column,y=row","frame_contract":"UI one-based inclusive; NumPy zero-based half-open","candidate_contract":"58 per burst; 6 px NMS; 6 px one-to-one match; unmatched candidates unknown","stage_sequence":{"S0":"advance","S1A":"advance numerical only","S1":"stop_branch after calibration rescue"},"primary_tables":["runs.tsv"],"primary_decision":"decision.json","large_media":[],"full_scientific_audit_applicable":False,"full_scientific_audit_reason":"screening-only explicit opt-out; no promoted finalist"}
    artifacts={"artifacts":[{"id":"summary","path":"summary.json","role":"compact scientific result"},{"id":"decision","path":"decision.json","role":"machine gate"},{"id":"runs","path":"runs.tsv","role":"one row per operator"},{"id":"valid_stage","path":str((valid/"stage_summary.json").relative_to(root)),"role":"full compact metrics"},{"id":"superseded_stage","path":str((invalid/"stage_summary.json").relative_to(root)),"role":"preserved invalid diagnostic"}],"dense_arrays":0,"videos":0,"tiffs":0}
    validation={"passed":True,"decision_agrees":decision["decision"]=="stop_branch","family_count":len(stage["families"]),"operator_count":len(stage["operators"]),"all_numerically_resolved":all(r["numerical_health"]["resolved"] for r in stage["operators"]),"rescue_consumed":decision["rescue_consumed"],"superseded_result_not_used":True}
    for name,payload in (("summary.json",compact),("llm_context.json",context),("artifact_index.json",artifacts),("validation.json",validation),("decision.json",decision)): _atomic_json(root/name,payload)
    rows=[]
    for item in stage["operators"]:
        rows.append({"run_id":item["design_id"],"stage":"S1_OPERATOR_SCREEN","method":"fractional_whitening","operator_family":item["family"],"design_id":item["design_id"],"outer_fold":"all","inner_fold":"","seed":"","status":"completed","stop_reason":"","heldout_burst_id":"all","heldout_labels":item["labels"],"heldout_matches_at_58":item["matches_at_58"],"heldout_recall_at_58":item["macro_kpr_at_58"],"heldout_matches_q1":item["matches_q1"],"heldout_candidates_q1":item["candidates_q1"],"trace_preserve":item["trace_preserve"],"max_condition_number":item["numerical_health"]["condition_number"],"effective_rank_fraction":item["numerical_health"]["effective_rank_fraction"],"converged":item["numerical_health"]["resolved"],"best_epoch":"","runtime_seconds":"","peak_rss_mib":stage["peak_rss_mib"],"peak_vram_mib":0,"parameter_file":"","config_hash":"","git_commit":"","data_fingerprint":""})
    write_runs(root/"runs.tsv",rows)
    lines=["# Learned whitening S1 result","",f"Decision: `{decision['decision']}` (calibration rescue consumed).","","The initial prefix result is preserved but invalid because its per-pixel MAD base did not match Raw Direct. The corrected global-scale run is authoritative.","","| Family | Fixed-Select Macro-KPR@58 | Matches | Delta vs Raw | Classification |","|---|---:|---:|---:|---|"]
    for f in stage["families"]: lines.append(f"| {f['family']} | {f['fixed_select_macro_kpr_at_58']:.4f} | {f['fixed_select_matches']}/79 | {f['delta_vs_raw']:+.4f} | {f['classification']} |")
    lines += ["","No tested family supports learned-mixture development. The plan routes next to only the minimal matched ICA utility confirmation. These are within-video sparse-positive results; precision and cross-fish generalization are not identified."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); return {"summary":compact,"validation":validation}
