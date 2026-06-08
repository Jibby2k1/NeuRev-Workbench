"""Grid dynamics CLI commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurobench.dynamics.baselines import write_baseline_metrics
from neurobench.dynamics.classifier import train_latent_classifier
from neurobench.dynamics.datasets import build_dynamics_dataset
from neurobench.dynamics.train import train_autoencoder, train_latent_rnn
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

    clf = dyn.add_parser("train-latent-classifier", help="Train a video-level latent-code classifier.")
    clf.add_argument("--dataset", required=True, type=Path)
    clf.add_argument("--autoencoder-run", required=True, type=Path)
    clf.add_argument("--labels-from", default="manifest")
    clf.add_argument("--split-unit", default="video")
    clf.add_argument("--evaluation", default="stratified_kfold")
    clf.add_argument("--classifier", default="logistic_regression")
    clf.add_argument("--out-dir", required=True, type=Path)
    clf.set_defaults(func=dynamics_classifier_command)
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



def _parse_str_list(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated string list.")
    return values

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
