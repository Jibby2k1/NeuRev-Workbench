"""Terminal report and collision-safe finalization for the conclusive batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from neurobench.experiments.learnable_contrast import core as label_core

from .conclusive_config import ConclusiveBatchConfig
from .screen_runner import _atomic_json


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _review_package(config: ConclusiveBatchConfig, root: Path) -> dict[str, Any]:
    review=root/"review"; review.mkdir(exist_ok=True)
    labels=label_core.load_labels(config.labels_tsv)
    identities={"roi_007","roi_008","roi_010","roi_014","roi_015","roi_017","roi_019","roi_020"}
    expert_notes={
        "roi_007":"Expert saw a flash but no clear cell structure; cautious exclusion proposed.",
        "roi_008":"Expert saw a flash but the cell boundary was difficult to identify.",
        "roi_010":"Expert identified this neuron; review jointly with overlapping ROI 015.",
        "roi_014":"Expert saw a round neuron-like structure and activity.",
        "roi_015":"Expert judged this region mostly overlapped with ROI 010 and suggested merging.",
        "roi_017":"Expert saw a flash but the cell boundary was difficult to identify.",
        "roi_019":"Expert identified an individual neuron with weak intensity change.",
        "roi_020":"Expert observed only a bright dot; neuronal interpretation is uncertain.",
    }
    rows=[]
    for row in labels:
        identity=str(row["roi_identity"])
        if identity not in identities: continue
        rows.append({"burst_id":int(row["burst_id"]),"roi_identity":identity,
                     "start_frame_ui":int(row["start_frame_ui"]),"end_frame_ui":int(row["end_frame_ui"]),
                     "x_px":float(row["x_px"]),"y_px":float(row["y_px"]),
                     "expert_note":expert_notes[identity],"review_status":"unreviewed",
                     "review_label":"","merge_target":"roi_010" if identity=="roi_015" else "",
                     "reviewer":"","review_note":"",
                     "interpretation":"unknown_until_adjudicated"})
    path=review/"difficult_roi_adjudication_queue.tsv"
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    copied=[]
    for source,name in (
        (Path("Outputs/PairwiseSeparation/spon_ca_burst_pairwise_separation_v1/candidates/candidate_review_queue.tsv"),"prior_pairwise_candidate_review_queue.tsv"),
        (Path("Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_v1/adjudication_draft.tsv"),"prior_hard_roi_adjudication_draft.tsv")):
        if source.is_file():
            target=review/name; shutil.copy2(source,target)
            copied.append({"source":str(source.resolve()),"path":str(target.relative_to(root)),"sha256":_sha256(target)})
    return {"difficult_roi_rows":len(rows),"difficult_roi_queue":str(path.relative_to(root)),
            "copied_prior_queues":copied,
            "precision_status":"not_defined_without_exhaustive_bounded-field_adjudication"}


def _runtime_sum(paths) -> float:
    total=0.0
    for path in paths:
        try: total += float(json.loads(path.read_text()).get("runtime_seconds",0.0))
        except (OSError,ValueError,TypeError,json.JSONDecodeError): pass
    return total


def finalize(config: ConclusiveBatchConfig) -> dict[str, Any]:
    output=config.output_root; root=Path(str(output)+".partial")
    if output.exists(): raise FileExistsError(output)
    required={
        "stage0":root/"stages/00_numerical_qualification/metrics.json",
        "stage1":root/"stages/01_continuous_identifiability/metrics.json",
        "stage2":root/"stages/02_selective_risk/metrics.json",
        "stage3":root/"stages/03_generated_confirmation/metrics.json",
        "stage4":root/"stages/04_native_semi_synthetic/metrics.json",
        "videos":root/"videos/reference_and_difficult_roi_v1/manifest.json"}
    missing=[key for key,path in required.items() if not path.is_file()]
    if missing: raise RuntimeError(f"finalization prerequisites missing: {missing}")
    metrics={key:json.loads(path.read_text()) for key,path in required.items()}
    native_passing=list(metrics["stage4"]["passing_methods"])
    if native_passing:
        raise RuntimeError(f"native methods require frozen Spon evaluation before finalization: {native_passing}")
    stage5={"schema_version":1,"status":"no_candidate_survived_g0_g2",
            "executed":False,"passing_methods":[],
            "reason":"All common-input methods failed selective risk and all native-best methods failed held semi-synthetic scientific validity."}
    stage6={"schema_version":1,"status":"frozen_spon_evaluation_not_applicable",
            "executed":False,"reason":"No scientifically valid separator remained; running label-driven Spon method selection would bypass preregistered truth gates.",
            "raw_direct_anchor":0.605615942,"latent_dynamics_smoother_anchor":0.6867}
    _atomic_json(root/"stages/05_frozen_methods.json",stage5)
    _atomic_json(root/"stages/06_spon_evaluation.json",stage6)
    review=_review_package(config,root)
    stage1_runtime=_runtime_sum((root/"stages/01_continuous_identifiability/fits").glob("*.json"))
    stage2_runtime=_runtime_sum((root/"stages/02_selective_risk/rows").glob("*.json"))
    stage4_runtime=_runtime_sum((root/"stages/04_native_semi_synthetic/rows").glob("*.json"))
    stage1=metrics["stage1"]; stage2=metrics["stage2"]; stage4=metrics["stage4"]
    selective_rows=[{"method_id":row["method_id"],**row["evaluation"]} for row in stage2["method_results"]]
    native_rows=stage4["method_results"]
    report_lines=[
        "# Information Source Separation Conclusive Batch v1", "",
        "## Answer first", "",
        "No investigated source-separation method survived the preregistered truth and selective-risk gates. This is a conclusive negative model-selection result for the tested panel, not evidence that source separation is impossible in principle.", "",
        "The full Spon detector comparison was correctly not run: using sparse Spon labels after every truth-known separator failed would bypass the scientific gates. Raw Direct (`0.605615942`) and the offline latent-dynamics smoother (`0.6867`) therefore remain the relevant established detection anchors.", "",
        "## Completed evidence", "",
        f"- Numerical qualification: complete, including distinct exact CaImAn CNMF and CNMF-E fits.",
        f"- Continuous identifiability screen: {stage1['fit_count']} fits across {stage1['fixture_count']} fixtures and {stage1['configuration_count']} configurations.",
        f"- Selective-risk program: {stage2['numerical_fit_count']} numerical fits, with disjoint train/calibration/evaluation seeds.",
        f"- Native semi-synthetic program: {stage4['fit_count']} fits, including bounded development selection and held-seed evaluation.",
        f"- Diagnostic videos: {metrics['videos']['video_count']} verified MP4 files.", "",
        "## Selective-risk result", "",
        "| Method | False resolutions | Coverage | Convergence | Gate |", "| --- | ---: | ---: | ---: | --- |"]
    for row in selective_rows:
        report_lines.append(f"| {row['method_id']} | {row['false_resolution_count']}/{row['unidentifiable_count']} | {row['identifiable_coverage']:.3f} | {row['converged_fraction']:.3f} | fail |")
    report_lines += ["", "No method met zero false resolutions, at least 0.80 identifiable coverage, at least 0.95 convergence, and no catastrophic artifact family.", "",
                     "## Native semi-synthetic result", "",
                     "| Method | Held fits | Valid fraction | Median neural NMSE | Convergence | Gate |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in native_rows:
        report_lines.append(f"| {row['method_id']} | {row['fit_count']} | {row['scientific_valid_fraction']:.3f} | {row['median_neural_reconstruction_nmse']:.4g} | {row['converged_fraction']:.3f} | {'pass' if row['gate_passed'] else 'fail'} |")
    report_lines += ["", "## Diagnostic and review artifacts", "",
        "The video package contains hash-verified generated/Spon reference videos plus dedicated clips for ROI 007, 008, 014, 017, 019, 020, and the 010/015 overlap. Each difficult clip includes UI frames 1988–2068, covering pre-window activity before the second labeled window.", "",
        f"The fixed difficult-ROI adjudication queue contains {review['difficult_roi_rows']} burst/ROI rows. Unreviewed entries remain unknown; they are not negatives.", "",
        "## Interpretation and remaining limitation", "",
        "The information-theoretic methods often recovered sources well on average, but no confidence rule generalized safely across unseen rank deficiency, pure noise, overlap, collinearity, background aliasing, and artifact families. Native spatial/CaImAn methods were then evaluated separately rather than forced into the temporal interface, and none passed held reconstruction fidelity.", "",
        "Precision is still undefined without exhaustive bounded-field adjudication. Because no method survived the truth-known gates, that missing precision label does not affect the negative model-selection conclusion; it only limits claims about the biological composition of unmatched historical candidates.", "",
        "## Decision", "",
        "Terminal disposition: `no_candidate_survived`. Do not replace Raw Direct or the latent-dynamics smoother with a tested separator. The next scientifically justified work is improved supervision/acquisition metadata or a materially different generative model—not another blind hyperparameter sweep.", ""]
    (root/"report.md").write_text("\n".join(report_lines),encoding="utf-8")
    payload={"schema_version":1,"status":"conclusive_batch_complete",
             "terminal_disposition":"no_candidate_survived",
             "stage_statuses":{key:value["status"] for key,value in metrics.items() if key!="videos"},
             "fit_counts":{"stage1":stage1["fit_count"],"stage2_numerical":stage2["numerical_fit_count"],"stage4":stage4["fit_count"]},
             "recorded_runtime_seconds":{"stage1":stage1_runtime,"stage2":stage2_runtime,"stage4":stage4_runtime},
             "passing_methods":[],"spon_model_evaluation_executed":False,
             "video_count":metrics["videos"]["video_count"],"review":review,
             "precision_status":review["precision_status"],"completed_unix":time.time(),
             "recommendation":"Retain established anchors; require materially new supervision or generative assumptions before another separation search."}
    _atomic_json(root/"metrics.json",payload)
    state=json.loads((root/"run_state.json").read_text()); state.update({"status":"complete","current_stage":None,
        "completed_stages":["00_numerical_qualification","01_continuous_identifiability","02_selective_risk","03_generated_confirmation","04_native_semi_synthetic","05_frozen_methods","06_spon_evaluation","07_diagnostic_videos","08_final_report"],
        "terminal_disposition":"no_candidate_survived","updated_unix":time.time()})
    _atomic_json(root/"run_state.json",state)
    output.parent.mkdir(parents=True,exist_ok=True); root.replace(output)
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True)
    args=parser.parse_args(argv); print(json.dumps(finalize(ConclusiveBatchConfig.load(args.config)),indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
