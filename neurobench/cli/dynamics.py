"""Grid dynamics CLI commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from neurobench.dynamics.artifact_audit import build_grid128_artifact_audit
from neurobench.dynamics.baselines import write_baseline_metrics
from neurobench.dynamics.classifier import train_latent_classifier
from neurobench.dynamics.datasets import build_dynamics_dataset
from neurobench.dynamics.kinetics_baselines import evaluate_kinetics_baselines
from neurobench.dynamics.latent_interpretation import build_latent_interpretation_report, build_latent_objective_plan
from neurobench.dynamics.launch_readiness import build_grid128_stage_b_launch_readiness
from neurobench.dynamics.manual_annotations import evaluate_manual_roi_spikes_on_dataset, import_manual_roi_spikes, score_spatial_checkpoints_on_manual_roi_spikes, score_latent_sequence_runs_on_manual_roi_spikes
from neurobench.dynamics.multi_horizon import build_active_cell_rescue_plan, build_multi_horizon_report, build_shared_horizon_baseline_comparison, build_shared_horizon_neural_grid_plan, build_shared_horizon_neural_grid_status, build_shared_horizon_review_manifest
from neurobench.dynamics.multi_horizon_linear import evaluate_shared_multi_horizon_linear_latent
from neurobench.dynamics.multi_horizon_neural import train_shared_multi_horizon_latent_gru, train_shared_multi_horizon_latent_transformer
from neurobench.dynamics.planner import build_adaptive_sweep_plan
from neurobench.dynamics.report import build_dynamics_experiment_report
from neurobench.dynamics.concept_tests import backfill_spatial_prediction_examples
from neurobench.dynamics.supervisor import build_sweep_health_report, build_sweep_live_status, create_resume_script
from neurobench.dynamics.train import train_autoencoder, train_latent_rnn
from neurobench.dynamics.video_review import build_video_error_review
from neurobench.dynamics.sweep import run_latent_dynamics_sweep
from neurobench.dynamics.linear import evaluate_linear_latent_baseline
from neurobench.manifests import load_json
from neurobench.validation.schemas import validation_error_summary


def add_dynamics_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("dynamics", help="Build and train 32x32 grid latent dynamics models.")
    dyn = parser.add_subparsers(dest="dynamics_command", metavar="dynamics-command")
    build = dyn.add_parser("build-dataset", help="Build video-split dynamics arrays from grid states.")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--grid-states-dir", required=True, type=Path)
    build.add_argument("--split-unit", default="video")
    build.add_argument("--split-method", default="stratified_by_label")
    build.add_argument("--window-frames", type=int, default=8)
    build.add_argument("--prediction-horizon-frames", type=int, default=1)
    build.add_argument("--temporal-stride-frames", type=int, default=1)
    build.add_argument("--out-dir", required=True, type=Path)
    build.set_defaults(func=dynamics_build_dataset_command)

    baseline = dyn.add_parser("evaluate-baselines", help="Evaluate persistence and moving-average baselines.")
    baseline.add_argument("--dataset", required=True, type=Path)
    baseline.add_argument("--out", type=Path, default=None)
    baseline.set_defaults(func=dynamics_baseline_command)

    kinetics = dyn.add_parser("evaluate-kinetics-baselines", help="Evaluate calcium-kinetics-aware pixel baselines as sweep-compatible rows.")
    kinetics.add_argument("--dataset", action="append", required=True, type=Path, help="Dynamics dataset JSON. Can be passed multiple times.")
    kinetics.add_argument("--baseline-names", default="exponential_decay_10hz,exponential_decay_30hz,lowpass_10hz,lowpass_30hz,ar1_per_cell")
    kinetics.add_argument("--frame-rate-hz", type=float, default=None, help="Override dataset frame rate; defaults to dataset windowing metadata.")
    kinetics.add_argument("--out-dir", required=True, type=Path)
    kinetics.set_defaults(func=dynamics_kinetics_baseline_command)

    ae = dyn.add_parser("train-autoencoder", help="Train a tiny grid autoencoder smoke model.")
    ae.add_argument("--dataset", required=True, type=Path)
    ae.add_argument("--latent-dim", type=int, default=32)
    ae.add_argument("--base-channels", type=int, default=16)
    ae.add_argument("--epochs", type=int, default=10)
    ae.add_argument("--batch-size", type=int, default=32)
    ae.add_argument("--learning-rate", type=float, default=0.001)
    ae.add_argument("--device", default="cpu")
    ae.add_argument("--seed", type=int, default=7)
    ae.add_argument("--out-dir", required=True, type=Path)
    ae.set_defaults(func=dynamics_autoencoder_command)

    rnn = dyn.add_parser("train-latent-rnn", help="Train a tiny latent GRU next-state predictor.")
    rnn.add_argument("--dataset", required=True, type=Path)
    rnn.add_argument("--autoencoder-run", required=True, type=Path)
    rnn.add_argument("--window-frames", type=int, default=8)
    rnn.add_argument("--hidden-dim", type=int, default=64)
    rnn.add_argument("--epochs", type=int, default=10)
    rnn.add_argument("--batch-size", type=int, default=32)
    rnn.add_argument("--learning-rate", type=float, default=0.001)
    rnn.add_argument("--prediction-target", choices=["absolute", "delta"], default="absolute")
    rnn.add_argument("--device", default="cpu")
    rnn.add_argument("--seed", type=int, default=7)
    rnn.add_argument("--out-dir", required=True, type=Path)
    rnn.set_defaults(func=dynamics_latent_rnn_command)


    sweep = dyn.add_parser("sweep-latent-dynamics", help="Run a capped sequential AE + latent-GRU hyperparameter search.")
    sweep.add_argument("--dataset", required=True, type=Path)
    sweep.add_argument("--out-dir", required=True, type=Path)
    sweep.add_argument("--latent-dims", default="16,32,64")
    sweep.add_argument("--autoencoder-epochs", default="10,25")
    sweep.add_argument("--autoencoder-learning-rates", default="0.001,0.0003")
    sweep.add_argument("--autoencoder-batch-size", type=int, default=64)
    sweep.add_argument("--autoencoder-base-channels", default="16")
    sweep.add_argument("--rnn-hidden-dims", default="32,64,128")
    sweep.add_argument("--rnn-epochs", default="10,25")
    sweep.add_argument("--rnn-learning-rates", default="0.001,0.0003")
    sweep.add_argument("--rnn-batch-size", type=int, default=64)
    sweep.add_argument("--rnn-prediction-targets", default="absolute")
    sweep.add_argument("--max-autoencoders", type=int, default=6)
    sweep.add_argument("--max-rnn-runs", type=int, default=24)
    sweep.add_argument("--device", default="auto")
    sweep.add_argument("--seed", type=int, default=7)
    sweep.add_argument("--rerun-existing", action="store_true")
    sweep.set_defaults(func=dynamics_sweep_command)


    linear = dyn.add_parser("evaluate-linear-latent", help="Evaluate ridge/linear latent-window baselines.")
    linear.add_argument("--dataset", required=True, type=Path)
    linear.add_argument("--autoencoder-run", required=True, type=Path)
    linear.add_argument("--prediction-target", choices=["absolute", "delta"], default="absolute")
    linear.add_argument("--alphas", default="0,0.00001,0.0001,0.001,0.01,0.1,1")
    linear.add_argument("--batch-size", type=int, default=256)
    linear.add_argument("--device", default="cpu")
    linear.add_argument("--out-dir", required=True, type=Path)
    linear.set_defaults(func=dynamics_linear_command)

    shared_linear = dyn.add_parser("evaluate-shared-linear-horizons", help="Evaluate one horizon-conditioned linear latent baseline across multiple horizons.")
    shared_linear.add_argument("--dataset", action="append", required=True, type=Path, help="Dynamics dataset JSON. Can be passed multiple times.")
    shared_linear.add_argument("--autoencoder-run", required=True, type=Path)
    shared_linear.add_argument("--prediction-target", choices=["absolute", "delta"], default="delta")
    shared_linear.add_argument("--alphas", default="0,0.00001,0.0001,0.001,0.01,0.1,1")
    shared_linear.add_argument("--batch-size", type=int, default=256)
    shared_linear.add_argument("--device", default="cpu")
    shared_linear.add_argument("--out-dir", required=True, type=Path)
    shared_linear.set_defaults(func=dynamics_shared_linear_horizons_command)

    shared_gru = dyn.add_parser("train-shared-gru-horizons", help="Train one horizon-conditioned latent GRU across multiple dynamics horizons.")
    shared_gru.add_argument("--dataset", action="append", required=True, type=Path, help="Dynamics dataset JSON. Can be passed multiple times.")
    shared_gru.add_argument("--autoencoder-run", required=True, type=Path)
    shared_gru.add_argument("--hidden-dim", type=int, default=64)
    shared_gru.add_argument("--num-layers", type=int, default=1)
    shared_gru.add_argument("--epochs", type=int, default=25)
    shared_gru.add_argument("--batch-size", type=int, default=64)
    shared_gru.add_argument("--evaluation-batch-size", type=int, default=None, help="Decoded evaluation chunk size; defaults to --batch-size.")
    shared_gru.add_argument("--learning-rate", type=float, default=0.001)
    shared_gru.add_argument("--prediction-target", choices=["absolute", "delta"], default="delta")
    shared_gru.add_argument("--device", default="cpu")
    shared_gru.add_argument("--seed", type=int, default=7)
    shared_gru.add_argument("--progress-interval-epochs", type=int, default=1, help="Write/print a training heartbeat every N epochs.")
    shared_gru.add_argument("--quiet-progress", action="store_true", help="Write progress artifacts without printing heartbeat lines.")
    shared_gru.add_argument("--out-dir", required=True, type=Path)
    shared_gru.set_defaults(func=dynamics_shared_gru_horizons_command)

    shared_xfmr = dyn.add_parser("train-shared-transformer-horizons", help="Train one horizon-conditioned latent Transformer across multiple dynamics horizons.")
    shared_xfmr.add_argument("--dataset", action="append", required=True, type=Path, help="Dynamics dataset JSON. Can be passed multiple times.")
    shared_xfmr.add_argument("--autoencoder-run", required=True, type=Path)
    shared_xfmr.add_argument("--model-dim", type=int, default=64)
    shared_xfmr.add_argument("--num-heads", type=int, default=2)
    shared_xfmr.add_argument("--num-layers", type=int, default=1)
    shared_xfmr.add_argument("--dropout", type=float, default=0.1)
    shared_xfmr.add_argument("--epochs", type=int, default=25)
    shared_xfmr.add_argument("--batch-size", type=int, default=64)
    shared_xfmr.add_argument("--evaluation-batch-size", type=int, default=None, help="Decoded evaluation chunk size; defaults to --batch-size.")
    shared_xfmr.add_argument("--learning-rate", type=float, default=0.0003)
    shared_xfmr.add_argument("--prediction-target", choices=["absolute", "delta"], default="delta")
    shared_xfmr.add_argument("--device", default="cpu")
    shared_xfmr.add_argument("--seed", type=int, default=7)
    shared_xfmr.add_argument("--progress-interval-epochs", type=int, default=1, help="Write/print a training heartbeat every N epochs.")
    shared_xfmr.add_argument("--quiet-progress", action="store_true", help="Write progress artifacts without printing heartbeat lines.")
    shared_xfmr.add_argument("--out-dir", required=True, type=Path)
    shared_xfmr.set_defaults(func=dynamics_shared_transformer_horizons_command)

    clf = dyn.add_parser("train-latent-classifier", help="Train a video-level latent-code classifier.")
    clf.add_argument("--dataset", required=True, type=Path)
    clf.add_argument("--autoencoder-run", required=True, type=Path)
    clf.add_argument("--labels-from", default="manifest")
    clf.add_argument("--split-unit", default="video")
    clf.add_argument("--evaluation", default="stratified_kfold")
    clf.add_argument("--classifier", default="logistic_regression")
    clf.add_argument("--out-dir", required=True, type=Path)
    clf.set_defaults(func=dynamics_classifier_command)

    health = dyn.add_parser("sweep-health", help="Write a health report for a long dynamics sweep.")
    health.add_argument("--sweep-dir", required=True, type=Path)
    health.add_argument("--out", type=Path, default=None)
    health.add_argument("--stale-minutes", type=float, default=60.0)
    health.add_argument("--no-archives", action="store_true")
    health.add_argument("--resume-script", type=Path, default=None, help="Optional path for a generated detached resume script.")
    health.add_argument("--resume-batch-size", type=int, default=None, help="Batch size to use when generating --resume-script.")
    health.set_defaults(func=dynamics_sweep_health_command)

    live = dyn.add_parser("sweep-live-status", help="Write a compact live status report for a running dynamics sweep.")
    live.add_argument("--sweep-dir", required=True, type=Path)
    live.add_argument("--out", type=Path, default=None)
    live.add_argument("--pid", type=int, default=None, help="Optional known sweep runner PID; otherwise inferred from ps output.")
    live.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi checks.")
    live.set_defaults(func=dynamics_sweep_live_status_command)

    report = dyn.add_parser("report-sweep", help="Build a meeting-ready report from partial or completed dynamics sweeps.")
    report.add_argument("--sweep-dir", action="append", required=True, type=Path, help="Sweep directory to include. Can be passed multiple times.")
    report.add_argument("--comparison-dir", type=Path, default=None, help="Existing or generated comparison dashboard directory.")
    report.add_argument("--out-dir", required=True, type=Path)
    report.add_argument("--title", default="Grid Dynamics Experiment Report")
    report.add_argument("--refresh-dashboard", action="store_true", help="Rebuild comparison artifacts before writing the report.")
    report.set_defaults(func=dynamics_report_sweep_command)

    audit = dyn.add_parser("audit-grid128-artifacts", help="Audit key grid128 report, review, and planning artifacts.")
    audit.add_argument("--root", required=True, type=Path, help="Grid128 experiment root to audit.")
    audit.add_argument("--out-dir", required=True, type=Path, help="Directory for grid128_artifact_audit.md/json.")
    audit.add_argument("--title", default="Grid128 Artifact Audit")
    audit.add_argument("--fail-on-issues", action="store_true", help="Return a nonzero exit code when any audited artifact is missing, invalid, or has missing references.")
    audit.set_defaults(func=dynamics_audit_grid128_artifacts_command)

    readiness = dyn.add_parser("stage-b-launch-readiness", help="Write the grid128 Stage B launch-readiness handoff artifact.")
    readiness.add_argument("--root", required=True, type=Path, help="Grid128 experiment root.")
    readiness.add_argument("--out-dir", required=True, type=Path, help="Directory for stage_b_launch_readiness.md/json.")
    readiness.add_argument("--title", default="Grid128 Stage B Launch Readiness")
    readiness.set_defaults(func=dynamics_stage_b_launch_readiness_command)

    plan = dyn.add_parser("plan-next-sweep", help="Build an adaptive second-stage sweep plan from partial results.")
    plan.add_argument("--sweep-dir", required=True, type=Path, help="Source sweep directory containing sweep_manifest.json.")
    plan.add_argument("--comparison-dir", required=True, type=Path, help="Comparison directory containing results_intelligence.json.")
    plan.add_argument("--out-dir", required=True, type=Path, help="Directory for next_sweep_plan.md/json artifacts.")
    plan.add_argument("--max-experiments", type=int, default=160, help="Maximum planned experiment specs to keep.")
    plan.add_argument("--suggested-batch-size", type=int, default=4, help="Batch size shown in the suggested launch command.")
    plan.set_defaults(func=dynamics_plan_next_sweep_command)

    review = dyn.add_parser("review-video-errors", help="Build a static HTML review of saved prediction examples and error panels.")
    review.add_argument("--comparison-dir", required=True, type=Path, help="Comparison directory containing comparison_manifest.json.")
    review.add_argument("--out-dir", required=True, type=Path, help="Directory for video_error_review.html/json and panel PNGs.")
    review.add_argument("--selection-mode", choices=["best_by_family", "best_test", "best_val", "best_active_cell", "most_improved_video", "least_improved_video", "heldout_first", "worst_over_persistence"], default="best_by_family")
    review.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    review.add_argument("--max-models", type=int, default=5)
    review.add_argument("--example-index", type=int, default=0)
    review.add_argument("--dataset-key", default=None)
    review.add_argument("--title", default="Grid Dynamics Video Error Review")
    review.set_defaults(func=dynamics_review_video_errors_command)

    import_roi = dyn.add_parser("import-manual-roi-spikes", help="Import compact manual ROI/spike Excel workbooks into reusable annotation artifacts.")
    import_roi.add_argument("--input", action="append", required=True, type=Path, help="Manual ROI/spike .xlsx file. Can be passed multiple times.")
    import_roi.add_argument("--out-dir", required=True, type=Path)
    import_roi.add_argument("--grid-size", type=int, default=128)
    import_roi.add_argument("--crop-size", type=int, default=512)
    import_roi.add_argument("--frame-rate-hz", type=float, default=50.0)
    import_roi.add_argument("--title", default="Manual ROI Spike Annotations")
    import_roi.set_defaults(func=dynamics_import_manual_roi_spikes_command)

    eval_roi = dyn.add_parser("evaluate-manual-roi-spikes", help="Evaluate manual ROI/spike annotations on a dynamics dataset target/persistence baseline.")
    eval_roi.add_argument("--dataset", required=True, type=Path)
    eval_roi.add_argument("--annotations", required=True, type=Path, help="manual_roi_spike_annotations.json from import-manual-roi-spikes.")
    eval_roi.add_argument("--out-dir", required=True, type=Path)
    eval_roi.add_argument("--event-margin-frames", type=int, default=0)
    eval_roi.add_argument("--title", default="Manual ROI Spike Dataset Evaluation")
    eval_roi.set_defaults(func=dynamics_evaluate_manual_roi_spikes_command)

    score_roi = dyn.add_parser("score-manual-roi-spikes", help="Score spatial pixel checkpoints on manual ROI/spike event windows.")
    score_roi.add_argument("--dataset", required=True, type=Path)
    score_roi.add_argument("--annotations", required=True, type=Path)
    score_roi.add_argument("--run-dir", action="append", required=True, type=Path, help="Spatial checkpoint run directory containing concept_checkpoint.pt. Can be passed multiple times.")
    score_roi.add_argument("--out-dir", required=True, type=Path)
    score_roi.add_argument("--device", default="cuda")
    score_roi.add_argument("--batch-size", type=int, default=2)
    score_roi.add_argument("--event-margin-frames", type=int, default=0)
    score_roi.add_argument("--title", default="Manual ROI Spike Model Scores")
    score_roi.set_defaults(func=dynamics_score_manual_roi_spikes_command)

    score_latent_roi = dyn.add_parser("score-manual-roi-spikes-latent", help="Score latent/directional sequence checkpoints on manual ROI/spike event windows.")
    score_latent_roi.add_argument("--annotations", required=True, type=Path)
    score_latent_roi.add_argument("--latent-run", action="append", default=[], help="Latent RNN run as dataset_path=run_dir. Can be passed multiple times.")
    score_latent_roi.add_argument("--hybrid-run-dir", action="append", default=[], type=Path, help="Shared directional hybrid run directory. Can be passed multiple times.")
    score_latent_roi.add_argument("--left-dataset", type=Path, default=None)
    score_latent_roi.add_argument("--right-dataset", type=Path, default=None)
    score_latent_roi.add_argument("--out-dir", required=True, type=Path)
    score_latent_roi.add_argument("--device", default="cuda")
    score_latent_roi.add_argument("--batch-size", type=int, default=16)
    score_latent_roi.add_argument("--event-margin-frames", type=int, default=0)
    score_latent_roi.add_argument("--title", default="Manual ROI Spike Latent Sequence Scores")
    score_latent_roi.set_defaults(func=dynamics_score_manual_roi_spikes_latent_command)

    backfill = dyn.add_parser("backfill-concept-examples", help="Write prediction_examples.json for an existing spatial concept checkpoint.")
    backfill.add_argument("--dataset", required=True, type=Path, help="Dynamics dataset JSON used by the completed run.")
    backfill.add_argument("--run-dir", required=True, type=Path, help="Directory containing concept_checkpoint.pt and concept_metrics.json.")
    backfill.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path; defaults to <run-dir>/concept_checkpoint.pt.")
    backfill.add_argument("--metrics", type=Path, default=None, help="Optional metrics path; defaults to <run-dir>/concept_metrics.json.")
    backfill.add_argument("--out", type=Path, default=None, help="Optional output path; defaults to <run-dir>/prediction_examples.json.")
    backfill.add_argument("--batch-size", type=int, default=16)
    backfill.add_argument("--max-examples", type=int, default=3)
    backfill.add_argument("--device", default="cpu")
    backfill.add_argument("--no-update-metrics", action="store_true", help="Write examples without adding prediction_examples_path to metrics.")
    backfill.add_argument("--backfill-metrics", action="store_true", help="Also recompute full split, structured, and per-video diagnostics from the checkpoint.")
    backfill.add_argument("--dry-run", action="store_true", help="Validate checkpoint/dataset metadata and estimate work without writing examples or metrics.")
    backfill.add_argument("--json", action="store_true", help="Print the backfill summary as machine-readable JSON.")
    backfill.add_argument("--markdown-out", type=Path, default=None, help="Optional Markdown preflight summary path.")
    backfill.set_defaults(func=dynamics_backfill_concept_examples_command)

    latent = dyn.add_parser("interpret-latents", help="Build latent-state interpretation reports from autoencoder latent codes.")
    latent.add_argument("--autoencoder-run", required=True, type=Path, help="Path to autoencoder_run.json with latent_codes_path.")
    latent.add_argument("--out-dir", required=True, type=Path, help="Directory for latent interpretation JSON, Markdown, HTML, and PNG artifacts.")
    latent.add_argument("--max-frame-points", type=int, default=4000, help="Maximum sampled frame-level PCA points for previews and JSON.")
    latent.add_argument("--nearest-neighbors", type=int, default=3, help="Nearest video neighbors to report for each video.")
    latent.add_argument("--title", default="Latent State Interpretation Report")
    latent.set_defaults(func=dynamics_interpret_latents_command)

    latent_plan = dyn.add_parser("plan-latent-objectives", help="Plan supervised or contrastive latent follow-ups from an interpretation report.")
    latent_plan.add_argument("--interpretation-report", required=True, type=Path, help="Path to latent_interpretation_report.json.")
    latent_plan.add_argument("--out-dir", required=True, type=Path, help="Directory for latent_objective_plan.md/json.")
    latent_plan.add_argument("--title", default="Latent Objective Follow-Up Plan")
    latent_plan.set_defaults(func=dynamics_plan_latent_objectives_command)

    mh = dyn.add_parser("compare-horizons", help="Compare paired single-horizon runs and plan shared multi-horizon candidates.")
    mh.add_argument("--comparison-dir", required=True, type=Path, help="Comparison directory containing comparison_manifest.json.")
    mh.add_argument("--out-dir", required=True, type=Path, help="Directory for multi_horizon_report.md/json and plan manifest.")
    mh.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    mh.add_argument("--max-candidates", type=int, default=20)
    mh.add_argument("--title", default="Multi-Horizon Forecasting Report")
    mh.set_defaults(func=dynamics_compare_horizons_command)

    shgrid = dyn.add_parser("plan-shared-horizon-neural-grid", help="Write a small shared-horizon GRU/Transformer follow-up grid plan without launching runs.")
    shgrid.add_argument("--dataset", action="append", required=True, type=Path, help="Dynamics dataset JSON. Can be passed multiple times.")
    shgrid.add_argument("--autoencoder-run", required=True, type=Path)
    shgrid.add_argument("--out-dir", required=True, type=Path, help="Directory for shared_horizon_neural_grid_* artifacts.")
    shgrid.add_argument("--run-root", required=True, type=Path, help="Root directory where planned shared-GRU runs should write artifacts.")
    shgrid.add_argument("--device", default="cpu")
    shgrid.add_argument("--epochs", type=int, default=25)
    shgrid.add_argument("--batch-size", type=int, default=64)
    shgrid.add_argument("--evaluation-batch-size", type=int, default=16)
    shgrid.add_argument("--seeds", default="7,13")
    shgrid.add_argument("--max-gru-configs", type=int, default=16)
    shgrid.add_argument("--no-transformer-placeholders", action="store_true")
    shgrid.add_argument("--title", default="Shared-Horizon Neural Follow-Up Grid")
    shgrid.set_defaults(func=dynamics_plan_shared_horizon_neural_grid_command)

    shstatus = dyn.add_parser("status-shared-horizon-neural-grid", help="Summarize pending/completed metrics for a shared-horizon neural grid plan.")
    shstatus.add_argument("--manifest", required=True, type=Path, help="Path to shared_horizon_neural_grid_manifest.json.")
    shstatus.add_argument("--out-dir", type=Path, default=None, help="Directory for shared_horizon_neural_grid_status.md/json; defaults to manifest directory.")
    shstatus.add_argument("--title", default="Shared-Horizon Neural Grid Status")
    shstatus.set_defaults(func=dynamics_status_shared_horizon_neural_grid_command)

    shcompare = dyn.add_parser("compare-shared-horizon-runs", help="Compare completed shared-horizon metric artifacts across linear/GRU/Transformer runs.")
    shcompare.add_argument("--run", action="append", required=True, help="Metric artifact as label=path or just path. Can be passed multiple times.")
    shcompare.add_argument("--out-dir", required=True, type=Path, help="Directory for shared_horizon_baseline_comparison.md/json.")
    shcompare.add_argument("--title", default="Shared-Horizon Baseline Comparison")
    shcompare.set_defaults(func=dynamics_compare_shared_horizon_runs_command)

    shreview = dyn.add_parser("build-shared-horizon-review-input", help="Build a review-compatible comparison_manifest.json from shared-horizon metrics.")
    shreview.add_argument("--run", action="append", required=True, help="Metric artifact as label=path or just path. Can be passed multiple times.")
    shreview.add_argument("--out-dir", required=True, type=Path, help="Directory for comparison_manifest.json consumed by review-video-errors.")
    shreview.add_argument("--comparison-dir", type=Path, default=None, help="Optional comparison directory whose dataset metadata should be copied.")
    shreview.add_argument("--dataset", action="append", type=Path, default=None, help="Optional dynamics dataset JSON for dataset metadata. Can be passed multiple times.")
    shreview.add_argument("--title", default="Shared-Horizon Review Input")
    shreview.set_defaults(func=dynamics_build_shared_horizon_review_input_command)

    rescue = dyn.add_parser("plan-active-cell-rescue", help="Plan the next shared-horizon run after active-cell regressions.")
    rescue.add_argument("--comparison-report", required=True, type=Path, help="Path to shared_horizon_baseline_comparison.json.")
    rescue.add_argument("--grid-status", required=True, type=Path, help="Path to shared_horizon_neural_grid_status.json.")
    rescue.add_argument("--out-dir", required=True, type=Path, help="Directory for active_cell_rescue_plan.md/json.")
    rescue.add_argument("--title", default="Active-Cell Rescue Plan")
    rescue.set_defaults(func=dynamics_plan_active_cell_rescue_command)
    return parser


def dynamics_build_dataset_command(args: argparse.Namespace) -> int:
    if args.split_unit != "video":
        print("Dynamics dataset split-unit must be video", file=sys.stderr)
        return 1
    try:
        payload = build_dynamics_dataset(
            manifest=load_json(args.manifest),
            grid_states_dir=args.grid_states_dir,
            out_dir=args.out_dir,
            window_frames=args.window_frames,
            prediction_horizon_frames=args.prediction_horizon_frames,
            temporal_stride_frames=args.temporal_stride_frames,
            split_method=args.split_method,
        )
    except Exception as exc:
        print("Dynamics dataset build failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Dynamics dataset: {Path(args.out_dir) / 'dynamics_dataset.json'}")
    print(f"windows: {payload.get('extras', {}).get('window_count')}")
    return 0


def dynamics_baseline_command(args: argparse.Namespace) -> int:
    try:
        dataset = load_json(args.dataset)
        out = args.out or Path(args.dataset).with_name("baseline_metrics.json")
        metrics = write_baseline_metrics(dataset, out)
    except Exception as exc:
        print("Baseline evaluation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Baseline metrics: {out}")
    print(f"persistence_mse: {metrics['persistence']['mse']:.6g}")
    return 0


def dynamics_kinetics_baseline_command(args: argparse.Namespace) -> int:
    try:
        datasets = {Path(path).parent.name: load_json(path) for path in args.dataset}
        summary = evaluate_kinetics_baselines(
            datasets=datasets,
            out_dir=args.out_dir,
            baseline_names=_parse_str_list(args.baseline_names),
            frame_rate_hz=args.frame_rate_hz,
        )
    except Exception as exc:
        print("Kinetics baseline evaluation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Kinetics baseline summary: {Path(args.out_dir) / 'kinetics_baseline_summary.json'}")
    print(f"experiments: {summary['experiment_count']}")
    print(f"manifest: {summary['manifest_path']}")
    return 0


def dynamics_autoencoder_command(args: argparse.Namespace) -> int:
    try:
        run = train_autoencoder(dataset=load_json(args.dataset), out_dir=args.out_dir, latent_dim=args.latent_dim, base_channels=args.base_channels, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed, device=args.device)
    except Exception as exc:
        print("Autoencoder training failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Autoencoder run: {Path(args.out_dir) / 'autoencoder_run.json'}")
    print(f"checkpoint: {run['checkpoint_path']}")
    return 0


def dynamics_latent_rnn_command(args: argparse.Namespace) -> int:
    try:
        run = train_latent_rnn(dataset=load_json(args.dataset), autoencoder_run=load_json(args.autoencoder_run), out_dir=args.out_dir, window_frames=args.window_frames, hidden_dim=args.hidden_dim, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, prediction_target=args.prediction_target, seed=args.seed, device=args.device)
    except Exception as exc:
        print("Latent RNN training failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Latent RNN run: {Path(args.out_dir) / 'latent_rnn_run.json'}")
    print(f"baseline metrics: {run['baseline_metrics_path']}")
    return 0



def dynamics_sweep_command(args: argparse.Namespace) -> int:
    try:
        summary = run_latent_dynamics_sweep(
            dataset=load_json(args.dataset),
            out_dir=args.out_dir,
            latent_dims=_parse_int_list(args.latent_dims),
            autoencoder_epochs=_parse_int_list(args.autoencoder_epochs),
            autoencoder_learning_rates=_parse_float_list(args.autoencoder_learning_rates),
            autoencoder_batch_size=args.autoencoder_batch_size,
            autoencoder_base_channels=_parse_int_list(args.autoencoder_base_channels),
            rnn_hidden_dims=_parse_int_list(args.rnn_hidden_dims),
            rnn_epochs=_parse_int_list(args.rnn_epochs),
            rnn_learning_rates=_parse_float_list(args.rnn_learning_rates),
            rnn_batch_size=args.rnn_batch_size,
            rnn_prediction_targets=_parse_str_list(args.rnn_prediction_targets),
            max_autoencoders=args.max_autoencoders,
            max_rnn_runs=args.max_rnn_runs,
            device=args.device,
            seed=args.seed,
            skip_existing=not args.rerun_existing,
            progress=print,
        )
    except Exception as exc:
        print("Latent dynamics sweep failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    counts = summary.get("counts", {})
    best = summary.get("best", {}).get("latent_rnn_by_selection_latent_code_mse") or summary.get("best", {}).get("latent_rnn_by_latent_code_mse")
    print(f"Sweep summary: {Path(args.out_dir) / 'sweep_summary.json'}")
    print(f"completed: {counts.get('autoencoder_completed', 0)} autoencoders, {counts.get('latent_rnn_completed', 0)} latent RNNs")
    if best:
        print(f"best_selection_latent_code_mse: {best['value']:.6g} ({best['config_id']})")
    return 0


def _parse_int_list(text: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated integer list.")
    return [int(value) for value in values]


def _parse_float_list(text: str) -> list[float]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated float list.")
    return [float(value) for value in values]



def dynamics_linear_command(args: argparse.Namespace) -> int:
    try:
        run = evaluate_linear_latent_baseline(
            dataset=load_json(args.dataset),
            autoencoder_run=load_json(args.autoencoder_run),
            out_dir=args.out_dir,
            prediction_target=args.prediction_target,
            alphas=_parse_float_list(args.alphas),
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        print("Linear latent baseline failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    metrics = load_json(Path(run["metrics_path"]))
    print(f"Linear latent run: {Path(args.out_dir) / 'linear_latent_run.json'}")
    print(f"val_decoded_prediction_mse: {metrics.get('val_decoded_prediction_mse')}")
    return 0



def dynamics_shared_linear_horizons_command(args: argparse.Namespace) -> int:
    try:
        datasets = {Path(path).parent.name: load_json(path) for path in args.dataset}
        run = evaluate_shared_multi_horizon_linear_latent(
            datasets=datasets,
            autoencoder_run=load_json(args.autoencoder_run),
            out_dir=args.out_dir,
            prediction_target=args.prediction_target,
            alphas=_parse_float_list(args.alphas),
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        print("Shared multi-horizon linear latent baseline failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    metrics = load_json(Path(run["metrics_path"]))
    print(f"Shared multi-horizon linear run: {Path(args.out_dir) / 'multi_horizon_linear_run.json'}")
    print(f"horizons: {metrics.get('shared_horizons_frames')}")
    print(f"decoded_prediction_mse: {metrics.get('decoded_prediction_mse')}")
    return 0


def dynamics_shared_gru_horizons_command(args: argparse.Namespace) -> int:
    try:
        datasets = {Path(path).parent.name: load_json(path) for path in args.dataset}
        run = train_shared_multi_horizon_latent_gru(
            datasets=datasets,
            autoencoder_run=load_json(args.autoencoder_run),
            out_dir=args.out_dir,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            prediction_target=args.prediction_target,
            seed=args.seed,
            device=args.device,
            evaluation_batch_size=args.evaluation_batch_size,
            progress=None if args.quiet_progress else print,
            progress_interval_epochs=args.progress_interval_epochs,
        )
    except Exception as exc:
        print("Shared multi-horizon latent GRU training failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    metrics = load_json(Path(run["metrics_path"]))
    print(f"Shared multi-horizon GRU run: {Path(args.out_dir) / 'multi_horizon_gru_run.json'}")
    print(f"horizons: {metrics.get('shared_horizons_frames')}")
    print(f"selection_latent_code_mse: {metrics.get('selection_latent_code_mse')}")
    print(f"decoded_prediction_mse: {metrics.get('decoded_prediction_mse')}")
    return 0


def dynamics_shared_transformer_horizons_command(args: argparse.Namespace) -> int:
    try:
        datasets = {Path(path).parent.name: load_json(path) for path in args.dataset}
        run = train_shared_multi_horizon_latent_transformer(
            datasets=datasets,
            autoencoder_run=load_json(args.autoencoder_run),
            out_dir=args.out_dir,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            prediction_target=args.prediction_target,
            seed=args.seed,
            device=args.device,
            evaluation_batch_size=args.evaluation_batch_size,
            progress=None if args.quiet_progress else print,
            progress_interval_epochs=args.progress_interval_epochs,
        )
    except Exception as exc:
        print("Shared multi-horizon latent Transformer training failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    metrics = load_json(Path(run["metrics_path"]))
    print(f"Shared multi-horizon Transformer run: {Path(args.out_dir) / 'multi_horizon_transformer_run.json'}")
    print(f"horizons: {metrics.get('shared_horizons_frames')}")
    print(f"selection_latent_code_mse: {metrics.get('selection_latent_code_mse')}")
    print(f"decoded_prediction_mse: {metrics.get('decoded_prediction_mse')}")
    return 0


def _parse_str_list(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated string list.")
    return values

def dynamics_report_sweep_command(args: argparse.Namespace) -> int:
    try:
        report = build_dynamics_experiment_report(
            sweep_dirs=args.sweep_dir,
            comparison_dir=args.comparison_dir,
            out_dir=args.out_dir,
            title=args.title,
            refresh_dashboard=args.refresh_dashboard,
        )
    except Exception as exc:
        print("Dynamics sweep report failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Dynamics experiment report: {report['markdown_path']}")
    print(f"report json: {report['report_path']}")
    print(f"completed rows: {report['run_summary']['completed_metric_rows']}")
    return 0


def dynamics_sweep_health_command(args: argparse.Namespace) -> int:
    try:
        summary = build_sweep_health_report(
            sweep_dir=args.sweep_dir,
            out_path=args.out,
            include_archives=not args.no_archives,
            stale_minutes=args.stale_minutes,
        )
        if args.resume_script is not None:
            if args.resume_batch_size is None:
                raise ValueError("--resume-batch-size is required when --resume-script is provided.")
            script = create_resume_script(
                sweep_dir=args.sweep_dir,
                batch_size=args.resume_batch_size,
                script_path=args.resume_script,
            )
            summary["resume_script_path"] = str(script)
    except Exception as exc:
        print("Sweep health report failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Sweep health report: {summary['report_path']}")
    print(f"progress: {summary['current_index']}/{summary['experiment_count']}")
    print(f"status counts: {summary['status_counts']}")
    if summary.get("resume_script_path"):
        print(f"resume script: {summary['resume_script_path']}")
    return 0


def dynamics_sweep_live_status_command(args: argparse.Namespace) -> int:
    try:
        status = build_sweep_live_status(
            sweep_dir=args.sweep_dir,
            out_path=args.out,
            pid=args.pid,
            include_gpu=not args.no_gpu,
        )
    except Exception as exc:
        print("Sweep live status failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    active = status.get("active_status") or status.get("inferred_next_spec") or {}
    print(f"Sweep live status: {status['report_path']}")
    print(f"live state: {status['live_state']}")
    print(f"progress: {status['progress_index']}/{status['experiment_count']}")
    if active:
        print(f"active: {active.get('index')} {active.get('status', '')} {active.get('experiment_id')}")
    process = status.get("process") or {}
    if process.get("pid"):
        print(f"process: pid={process.get('pid')} stat={process.get('stat')} elapsed={process.get('elapsed')} cpu={process.get('cpu_percent')}")
    gpu = status.get("gpu") or {}
    if gpu.get("checked"):
        print(f"gpu: util={gpu.get('utilization_gpu_percent')} mem={gpu.get('process_used_memory_mib')}")
    return 0


def dynamics_audit_grid128_artifacts_command(args: argparse.Namespace) -> int:
    try:
        report = build_grid128_artifact_audit(
            root=args.root,
            out_dir=args.out_dir,
            title=args.title,
        )
    except Exception as exc:
        print("Grid128 artifact audit failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Grid128 artifact audit: {report['markdown_path']}")
    print(f"audit json: {report['json_path']}")
    print(f"status counts: {report['status_counts']}")
    if args.fail_on_issues and not report.get("ok"):
        return 2
    return 0


def dynamics_stage_b_launch_readiness_command(args: argparse.Namespace) -> int:
    try:
        readiness = build_grid128_stage_b_launch_readiness(
            root=args.root,
            out_dir=args.out_dir,
            title=args.title,
        )
    except Exception as exc:
        print("Stage B launch-readiness build failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Stage B launch readiness: {readiness['markdown_path']}")
    print(f"readiness json: {readiness['json_path']}")
    print(f"status: {readiness['status']}")
    stage_b = readiness.get("stage_b") or {}
    print(f"planned experiments: {stage_b.get('planned_experiment_count')}")
    print(f"dry run validated: {stage_b.get('dry_run_validated')}")
    return 0


def dynamics_plan_next_sweep_command(args: argparse.Namespace) -> int:
    try:
        summary = build_adaptive_sweep_plan(
            sweep_dir=args.sweep_dir,
            comparison_dir=args.comparison_dir,
            out_dir=args.out_dir,
            max_experiments=args.max_experiments,
            suggested_batch_size=args.suggested_batch_size,
        )
    except Exception as exc:
        print("Adaptive next-sweep planning failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Adaptive next-sweep plan: {summary['markdown_path']}")
    print(f"planning manifest: {summary['manifest_path']}")
    print(f"planned experiments: {summary['planned_experiment_count']}")
    print(f"selection counts: {summary['selection_counts']}")
    return 0


def dynamics_review_video_errors_command(args: argparse.Namespace) -> int:
    try:
        summary = build_video_error_review(
            comparison_dir=args.comparison_dir,
            out_dir=args.out_dir,
            selection_mode=args.selection_mode,
            split=args.split,
            max_models=args.max_models,
            example_index=args.example_index,
            dataset_key=args.dataset_key,
            title=args.title,
        )
    except Exception as exc:
        print("Video error review failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Video error review: {summary['html_path']}")
    print(f"review json: {summary['summary_path']}")
    print(f"models: {summary['selected_model_count']}")
    return 0


def dynamics_import_manual_roi_spikes_command(args: argparse.Namespace) -> int:
    try:
        manifest = import_manual_roi_spikes(
            inputs=args.input,
            out_dir=args.out_dir,
            grid_size=args.grid_size,
            crop_size=args.crop_size,
            frame_rate_hz=args.frame_rate_hz,
            title=args.title,
        )
    except Exception as exc:
        print("Manual ROI/spike import failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Manual ROI/spike annotations: {manifest['manifest_path']}")
    print(f"roi tsv: {manifest['roi_tsv_path']}")
    print(f"interval tsv: {manifest['interval_tsv_path']}")
    print(f"rois: {manifest['annotation_count']}")
    print(f"spike intervals: {manifest['spike_interval_count']}")
    print(f"warnings: {len(manifest.get('warnings', []))}")
    return 0


def dynamics_evaluate_manual_roi_spikes_command(args: argparse.Namespace) -> int:
    try:
        report = evaluate_manual_roi_spikes_on_dataset(
            dataset=load_json(args.dataset),
            annotations=load_json(args.annotations),
            out_dir=args.out_dir,
            event_margin_frames=args.event_margin_frames,
            title=args.title,
        )
    except Exception as exc:
        print("Manual ROI/spike dataset evaluation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Manual ROI/spike dataset evaluation: {report['markdown_path']}")
    print(f"evaluation json: {report['json_path']}")
    print(f"rois: {report['roi_count']}")
    for split, row in sorted((report.get('summary', {}).get('by_split', {}) or {}).items()):
        print(f"{split}: rois={row.get('roi_count', 0)} matched_event_windows={row.get('matched_event_window_count', 0)}")
    return 0


def dynamics_score_manual_roi_spikes_command(args: argparse.Namespace) -> int:
    try:
        report = score_spatial_checkpoints_on_manual_roi_spikes(
            dataset=load_json(args.dataset),
            annotations=load_json(args.annotations),
            run_dirs=args.run_dir,
            out_dir=args.out_dir,
            device=args.device,
            batch_size=args.batch_size,
            event_margin_frames=args.event_margin_frames,
            title=args.title,
        )
    except Exception as exc:
        print("Manual ROI/spike model scoring failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Manual ROI/spike model scores: {report['markdown_path']}")
    print(f"scores json: {report['json_path']}")
    print(f"runs: {report['run_count']}")
    print(f"union event windows: {report['union_event_window_count']}")
    for rank, row in enumerate(report.get('run_summaries', [])[:5], start=1):
        print(f"rank {rank}: {row.get('experiment_id')} improvement={row.get('mean_event_improvement_over_persistence_mse')}")
    return 0


def dynamics_score_manual_roi_spikes_latent_command(args: argparse.Namespace) -> int:
    try:
        latent_runs = []
        for item in args.latent_run or []:
            if "=" not in str(item):
                raise ValueError("--latent-run must be dataset_path=run_dir")
            dataset_path, run_dir = str(item).split("=", 1)
            latent_runs.append({"dataset": Path(dataset_path), "run_dir": Path(run_dir)})
        report = score_latent_sequence_runs_on_manual_roi_spikes(
            annotations=load_json(args.annotations),
            latent_runs=latent_runs,
            hybrid_runs=args.hybrid_run_dir,
            left_dataset=load_json(args.left_dataset) if args.left_dataset else None,
            right_dataset=load_json(args.right_dataset) if args.right_dataset else None,
            out_dir=args.out_dir,
            device=args.device,
            batch_size=args.batch_size,
            event_margin_frames=args.event_margin_frames,
            title=args.title,
        )
    except Exception as exc:
        print("Manual ROI/spike latent model scoring failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Manual ROI/spike latent scores: {report['markdown_path']}")
    print(f"scores json: {report['json_path']}")
    print(f"runs: {report['run_count']}")
    for rank, row in enumerate(report.get('run_summaries', [])[:8], start=1):
        print(f"rank {rank}: {row.get('experiment_id')} improvement={row.get('mean_event_improvement_over_persistence_mse')}")
    return 0


def dynamics_backfill_concept_examples_command(args: argparse.Namespace) -> int:
    try:
        summary = backfill_spatial_prediction_examples(
            dataset=load_json(args.dataset),
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            metrics_path=args.metrics,
            out_path=args.out,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            device=args.device,
            update_metrics=not args.no_update_metrics,
            backfill_metrics=args.backfill_metrics,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print("Concept prediction-example backfill failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    if args.markdown_out is not None:
        _write_backfill_preflight_markdown(args.markdown_out, summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"prediction examples: {summary.get('prediction_examples_path')}")
    if summary.get("dry_run"):
        print("dry run: true")
        print(f"dataset windows: {summary.get('dataset_window_count')}")
        print(f"estimated uncompressed bytes: {summary.get('estimated_uncompressed_bytes')}")
        print(f"estimated uncompressed GiB: {summary.get('estimated_uncompressed_gib')}")
        print(f"estimated compressed bytes: {summary.get('estimated_compressed_bytes')}")
        print(f"estimated compressed GiB: {summary.get('estimated_compressed_gib')}")
        print(f"requested batch size: {summary.get('requested_batch_size')}")
        print(f"estimated example batches: {summary.get('estimated_example_batches')}")
        print(f"estimated metric batches: {summary.get('estimated_metric_batches')}")
        would_write = summary.get("would_write_files") or []
        if would_write:
            print(f"would write files: {', '.join(str(path) for path in would_write)}")
        split_windows = summary.get("split_window_counts") or {}
        if split_windows:
            formatted = ", ".join(f"{key}={value}" for key, value in sorted(split_windows.items()))
            print(f"split windows: {formatted}")
        split_videos = summary.get("split_video_counts") or {}
        if split_videos:
            formatted = ", ".join(f"{key}={value}" for key, value in sorted(split_videos.items()))
            print(f"split videos: {formatted}")
        split_labels = summary.get("split_label_counts") or {}
        for split, labels in sorted(split_labels.items()):
            if labels:
                formatted = ", ".join(f"{key}={value}" for key, value in sorted(labels.items(), key=lambda item: (-item[1], item[0]))[:5])
                print(f"{split} labels: {formatted}")
        top_videos = summary.get("split_top_videos") or {}
        for split, rows in sorted(top_videos.items()):
            if rows:
                formatted = ", ".join(f"{item.get('video_id')}={item.get('window_count')}" for item in rows[:3])
                print(f"top {split} videos: {formatted}")
        preview = summary.get("example_preview") or []
        if preview:
            formatted = ", ".join(f"{item.get('index')}:{item.get('split') or 'unknown'}:{item.get('video_id')}" for item in preview[:5])
            print(f"example preview: {formatted}")
    print(f"metrics updated: {summary.get('metrics_updated')}")
    print(f"prediction metrics backfilled: {summary.get('prediction_metrics_backfilled')}")
    return 0


def _write_backfill_preflight_markdown(path: Path, summary: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_backfill_preflight_markdown(summary), encoding="utf-8")
    return path


def _backfill_preflight_markdown(summary: Mapping[str, Any]) -> str:
    def inline(value: Any) -> str:
        if value is None:
            return "None"
        return str(value)

    def split_counts_text(counts: Mapping[str, Any]) -> str:
        return ", ".join(f"{split}={count}" for split, count in sorted(counts.items()))

    lines = [
        "# Concept Prediction Backfill Preflight",
        "",
        f"Generated: `{inline(summary.get('created_at'))}`",
        f"Run directory: `{inline(summary.get('run_dir'))}`",
        f"Checkpoint: `{inline(summary.get('checkpoint_path'))}`",
        f"Metrics file: `{inline(summary.get('metrics_path'))}`",
        "",
        "## Scope",
        "",
        f"- Dry run: `{inline(summary.get('dry_run'))}`",
        f"- Architecture: `{inline(summary.get('architecture'))}`",
        f"- Dataset windows: `{inline(summary.get('dataset_window_count'))}`",
        f"- Estimated payload: `{inline(summary.get('estimated_uncompressed_gib'))}` GiB uncompressed, `{inline(summary.get('estimated_compressed_gib'))}` GiB compressed",
        f"- Requested batch size: `{inline(summary.get('requested_batch_size'))}`",
        f"- Estimated example batches: `{inline(summary.get('estimated_example_batches'))}`",
        f"- Estimated metric batches: `{inline(summary.get('estimated_metric_batches'))}`",
        f"- Would update metrics: `{inline(summary.get('would_update_metrics'))}`",
        f"- Would backfill metrics: `{inline(summary.get('would_backfill_metrics'))}`",
        "",
        "## Planned Writes",
        "",
    ]

    would_write = summary.get("would_write_files") or []
    if would_write:
        lines.extend(f"- `{path}`" for path in would_write)
    else:
        lines.append("- None")
    lines.extend(["", "## Split Coverage", ""])

    split_windows = summary.get("split_window_counts") or {}
    if isinstance(split_windows, Mapping) and split_windows:
        lines.append(f"- Split windows: {split_counts_text(split_windows)}")
    split_videos = summary.get("split_video_counts") or {}
    if isinstance(split_videos, Mapping) and split_videos:
        lines.append(f"- Split videos: {split_counts_text(split_videos)}")
    split_labels = summary.get("split_label_counts") or {}
    if isinstance(split_labels, Mapping):
        for split, labels in sorted(split_labels.items()):
            if isinstance(labels, Mapping) and labels:
                lines.append(f"- {split} labels: {split_counts_text(labels)}")
    top_videos = summary.get("split_top_videos") or {}
    if isinstance(top_videos, Mapping):
        for split, rows in sorted(top_videos.items()):
            if isinstance(rows, list) and rows:
                formatted = ", ".join(
                    f"{item.get('video_id')}={item.get('window_count')}"
                    for item in rows[:3]
                    if isinstance(item, Mapping)
                )
                if formatted:
                    lines.append(f"- Top {split} videos: {formatted}")

    lines.extend(["", "## Example Preview", ""])
    preview = summary.get("example_preview") or []
    if isinstance(preview, list) and preview:
        formatted = ", ".join(
            f"{item.get('index')}:{item.get('split') or 'unknown'}:{item.get('video_id')}"
            for item in preview[:10]
            if isinstance(item, Mapping)
        )
        lines.append(f"Selected examples: {formatted}")
    else:
        lines.append("Selected examples: None")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Do not run the full `--backfill-metrics` pass while the active CUDA sweep is resource-bound unless CPU/RAM headroom is intentionally available. Re-run this preflight if dataset, checkpoint, batch size, or metrics path changes.",
            "",
        ]
    )
    return "\n".join(lines)


def dynamics_interpret_latents_command(args: argparse.Namespace) -> int:
    try:
        report = build_latent_interpretation_report(
            autoencoder_run=args.autoencoder_run,
            out_dir=args.out_dir,
            max_frame_points=args.max_frame_points,
            nearest_neighbors=args.nearest_neighbors,
            title=args.title,
        )
    except Exception as exc:
        print("Latent interpretation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Latent interpretation report: {report['markdown_path']}")
    print(f"report html: {report['html_path']}")
    print(f"videos: {report['video_count']} frames: {report['frame_count']}")
    return 0


def dynamics_plan_latent_objectives_command(args: argparse.Namespace) -> int:
    try:
        plan = build_latent_objective_plan(
            interpretation_report=args.interpretation_report,
            out_dir=args.out_dir,
            title=args.title,
        )
    except Exception as exc:
        print("Latent objective planning failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Latent objective plan: {plan['markdown_path']}")
    print(f"plan json: {plan['plan_path']}")
    diagnosis = plan.get("diagnosis") or {}
    print(f"diagnosis: {diagnosis.get('status')}")
    return 0


def dynamics_compare_horizons_command(args: argparse.Namespace) -> int:
    try:
        report = build_multi_horizon_report(
            comparison_dir=args.comparison_dir,
            out_dir=args.out_dir,
            split=args.split,
            max_candidates=args.max_candidates,
            title=args.title,
        )
    except Exception as exc:
        print("Multi-horizon comparison failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Multi-horizon report: {report['markdown_path']}")
    print(f"report json: {report['report_path']}")
    print(f"paired groups: {report['paired_group_count']}")
    print(f"planned shared configs: {len(report['planned_shared_horizon_configs'])}")
    return 0


def dynamics_plan_shared_horizon_neural_grid_command(args: argparse.Namespace) -> int:
    try:
        plan = build_shared_horizon_neural_grid_plan(
            datasets=args.dataset,
            autoencoder_run=args.autoencoder_run,
            out_dir=args.out_dir,
            run_root=args.run_root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            seeds=_parse_int_list(args.seeds),
            max_gru_configs=args.max_gru_configs,
            include_transformer_placeholders=not args.no_transformer_placeholders,
            title=args.title,
        )
    except Exception as exc:
        print("Shared-horizon neural grid planning failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Shared-horizon neural grid plan: {plan['markdown_path']}")
    print(f"manifest: {plan['manifest_path']}")
    print(f"run script: {plan['script_path']}")
    print(f"ready configs: {plan['directly_executable_count']} placeholders: {plan['placeholder_count']}")
    return 0


def dynamics_status_shared_horizon_neural_grid_command(args: argparse.Namespace) -> int:
    try:
        status = build_shared_horizon_neural_grid_status(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            title=args.title,
        )
    except Exception as exc:
        print("Shared-horizon neural grid status failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Shared-horizon neural grid status: {status['markdown_path']}")
    print(f"status json: {status['status_path']}")
    print(f"status counts: {status['status_counts']}")
    return 0


def dynamics_compare_shared_horizon_runs_command(args: argparse.Namespace) -> int:
    try:
        report = build_shared_horizon_baseline_comparison(
            runs=args.run,
            out_dir=args.out_dir,
            title=args.title,
        )
    except Exception as exc:
        print("Shared-horizon baseline comparison failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Shared-horizon baseline comparison: {report['markdown_path']}")
    print(f"comparison json: {report['report_path']}")
    best = report.get("best_overall") or {}
    print(f"best overall: {best.get('label')} improve={best.get('improvement_over_persistence_mse')}")
    return 0


def dynamics_build_shared_horizon_review_input_command(args: argparse.Namespace) -> int:
    try:
        manifest = build_shared_horizon_review_manifest(
            runs=args.run,
            out_dir=args.out_dir,
            comparison_dir=args.comparison_dir,
            datasets=args.dataset,
            title=args.title,
        )
    except Exception as exc:
        print("Shared-horizon review input build failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Shared-horizon review manifest: {manifest['manifest_path']}")
    print(f"runs: {manifest['run_count']} rows: {manifest['row_count']}")
    return 0


def dynamics_plan_active_cell_rescue_command(args: argparse.Namespace) -> int:
    try:
        plan = build_active_cell_rescue_plan(
            comparison_report=args.comparison_report,
            grid_status=args.grid_status,
            out_dir=args.out_dir,
            title=args.title,
        )
    except Exception as exc:
        print("Active-cell rescue planning failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Active-cell rescue plan: {plan['markdown_path']}")
    print(f"plan json: {plan['plan_path']}")
    first = (plan.get("recommended_candidates") or [{}])[0]
    print(f"next candidate: {first.get('config_id')}")
    return 0


def dynamics_classifier_command(args: argparse.Namespace) -> int:
    if args.split_unit != "video":
        print("Classifier split-unit must be video", file=sys.stderr)
        return 1
    try:
        run = train_latent_classifier(dataset=load_json(args.dataset), autoencoder_run=load_json(args.autoencoder_run), out_dir=args.out_dir, classifier=args.classifier, split_method=args.evaluation)
    except Exception as exc:
        print("Latent classifier training failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Latent classifier run: {Path(args.out_dir) / 'latent_classifier_run.json'}")
    print(f"accuracy: {run['metrics']['accuracy']:.6g}")
    return 0
