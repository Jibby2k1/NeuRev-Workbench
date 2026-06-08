"""Reproducible overnight sweeps for grid latent dynamics experiments."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import gc
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from neurobench.dynamics.baselines import baseline_prediction, evaluate_baselines_from_arrays
from neurobench.dynamics.concept_tests import run_concept_tests
from neurobench.dynamics.linear import evaluate_linear_latent_baseline
from neurobench.dynamics.models import GridAutoencoder, LatentTransformerPredictor
from neurobench.dynamics.scalable import architecture_catalog, train_scalable_temporal_cnn
from neurobench.dynamics.train import (
    _checkpoint_latent_stats,
    _encode_frames_batched,
    _normalize_prediction_target,
    _prediction_split_metrics,
    _prepare_model_array,
    _promote_split_metrics,
    _split_mask,
    _torch,
    train_latent_rnn,
)


DEFAULT_DATASETS = {
    "w8_s3_h10": {
        "dataset": "Outputs/GridModel/060126/improvement_attempts_v1/datasets/w8_s3_h10/dynamics_dataset.json",
        "autoencoder_run": "Outputs/GridModel/060126/improvement_attempts_v1/models/autoencoder_s3_ld128_bc32_e75_lr0p001_v1/autoencoder_run.json",
        "window_frames": 8,
    },
    "w8_s1_h25": {
        "dataset": "Outputs/GridModel/060126/improvement_attempts_v1/datasets/w8_s1_h25/dynamics_dataset.json",
        "autoencoder_run": "Outputs/GridModel/060126/improvement_attempts_v1/models/autoencoder_s1_ld128_bc32_e75_lr0p001_v1/autoencoder_run.json",
        "window_frames": 8,
    },
    "w8_s1_h50": {
        "dataset": "Outputs/GridModel/060126/improvement_attempts_v1/datasets/w8_s1_h50/dynamics_dataset.json",
        "autoencoder_run": "Outputs/GridModel/060126/improvement_attempts_v1/models/autoencoder_s1_ld128_bc32_e75_lr0p001_v1/autoencoder_run.json",
        "window_frames": 8,
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    kind: str
    dataset_key: str
    seed: int
    params: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "kind": self.kind,
            "dataset_key": self.dataset_key,
            "seed": int(self.seed),
            "params": dict(self.params),
        }


def run_overnight_sweep(
    *,
    out_dir: str | Path,
    profile: str = "overnight",
    device: str = "cuda",
    seeds: tuple[int, ...] = (7, 13),
    batch_size: int = 64,
    epochs: int = 50,
    max_runs: int | None = None,
    time_limit_hours: float | None = None,
    datasets: Mapping[str, Mapping[str, Any]] | None = None,
    dry_run: bool = False,
    resume: bool = True,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    _set_conservative_env()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_map = {str(key): dict(value) for key, value in (datasets or DEFAULT_DATASETS).items()}
    specs = build_specs(profile=profile, seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=tuple(dataset_map.keys()))
    if max_runs is not None:
        specs = specs[: int(max_runs)]
    _validate_inputs(dataset_map)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "device": device,
        "seeds": [int(s) for s in seeds],
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "max_runs": max_runs,
        "time_limit_hours": time_limit_hours,
        "resume": bool(resume),
        "experiment_count": len(specs),
        "datasets": dataset_map,
        "experiments": [spec.to_json() for spec in specs],
    }
    (out / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        write_summary(out)
        return {"status": "dry_run", "experiment_count": len(specs), "manifest_path": str(out / "sweep_manifest.json")}

    start_time = time.monotonic()
    progress_path = out / "sweep_progress.jsonl"
    completed = 0
    skipped = 0
    failed = 0
    for index, spec in enumerate(specs, start=1):
        if time_limit_hours is not None:
            elapsed_hours = (time.monotonic() - start_time) / 3600.0
            if elapsed_hours >= float(time_limit_hours):
                break
        exp_out = out / spec.experiment_id
        status_record: dict[str, Any] = {
            "index": index,
            "experiment_count": len(specs),
            "experiment_id": spec.experiment_id,
            "kind": spec.kind,
            "dataset_key": spec.dataset_key,
            "seed": int(spec.seed),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path = expected_metrics_path(exp_out, spec)
        if resume and metrics_path.exists():
            skipped += 1
            status_record.update({"status": "skipped", "metrics_path": str(metrics_path)})
            _append_jsonl(progress_path, status_record)
            continue
        try:
            exp_start = time.monotonic()
            metrics_path = run_one(spec=spec, out_dir=exp_out, device=device, datasets=dataset_map)
            metrics = _load_json(metrics_path)
            status_record.update(
                {
                    "status": "completed",
                    "elapsed_seconds": time.monotonic() - exp_start,
                    "metrics_path": str(metrics_path),
                    "val_decoded_prediction_mse": metrics.get("val_decoded_prediction_mse"),
                    "val_persistence_mse": metrics.get("val_persistence_mse"),
                    "val_improvement_over_persistence_mse": metrics.get("val_improvement_over_persistence_mse"),
                    "test_decoded_prediction_mse": metrics.get("test_decoded_prediction_mse"),
                    "test_persistence_mse": metrics.get("test_persistence_mse"),
                    "test_improvement_over_persistence_mse": metrics.get("test_improvement_over_persistence_mse"),
                }
            )
            completed += 1
        except Exception as exc:  # noqa: BLE001 - this is a long-running batch runner.
            failed += 1
            status_record.update({"status": "failed", "error": repr(exc)})
            _append_jsonl(progress_path, status_record)
            _cleanup_torch()
            write_summary(out)
            if stop_on_error:
                raise
            continue
        _append_jsonl(progress_path, status_record)
        _cleanup_torch()
        write_summary(out)
    write_summary(out)
    return {
        "status": "finished",
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "manifest_path": str(out / "sweep_manifest.json"),
        "summary_tsv_path": str(out / "sweep_summary.tsv"),
        "summary_md_path": str(out / "sweep_summary.md"),
    }


def build_specs(*, profile: str, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...] = tuple(DEFAULT_DATASETS)) -> list[ExperimentSpec]:
    allowed_profiles = {"smoke", "overnight", "upgrade", "advanced", "advanced_big", "advanced_overnight", "cropped32_restricted", "cropped32_large", "highres_temporal_cnn_scalable"}
    if profile not in allowed_profiles:
        raise ValueError("profile must be one of: " + ", ".join(sorted(allowed_profiles)) + ".")
    if profile == "smoke":
        return [
            _residual_spec(dataset_keys[0], seed=seeds[0], hidden_dim=64, learning_rate=3e-4, residual_scale=0.10, epochs=1, batch_size=batch_size),
            _gru_spec(dataset_keys[0], seed=seeds[0], hidden_dim=64, learning_rate=3e-4, prediction_target="delta", epochs=1, batch_size=batch_size),
            _transformer_spec(dataset_keys[0], seed=seeds[0], model_dim=64, num_heads=2, num_layers=1, learning_rate=3e-4, prediction_target="delta", epochs=1, batch_size=batch_size),
        ]

    if profile == "upgrade":
        return _upgrade_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)
    if profile == "advanced":
        return _advanced_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)
    if profile == "advanced_big":
        return _advanced_big_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)
    if profile == "advanced_overnight":
        return _advanced_overnight_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)
    if profile == "cropped32_restricted":
        return _cropped32_restricted_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)
    if profile == "cropped32_large":
        return _cropped32_large_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)
    if profile == "highres_temporal_cnn_scalable":
        return _highres_temporal_cnn_scalable_specs(seeds=seeds, epochs=epochs, batch_size=batch_size, dataset_keys=dataset_keys)

    specs: list[ExperimentSpec] = []
    for dataset_key in dataset_keys:
        for seed in seeds:
            for hidden_dim in (64, 128, 256):
                for learning_rate in (3e-4, 1e-4, 3e-5):
                    for residual_scale in (0.10, 0.25):
                        specs.append(
                            _residual_spec(
                                dataset_key,
                                seed=seed,
                                hidden_dim=hidden_dim,
                                learning_rate=learning_rate,
                                residual_scale=residual_scale,
                                epochs=epochs,
                                batch_size=batch_size,
                            )
                        )
    for dataset_key in dataset_keys:
        for seed in seeds:
            for hidden_dim in (64, 128, 256):
                for learning_rate in (1e-3, 3e-4, 1e-4):
                    for prediction_target in ("absolute", "delta"):
                        specs.append(
                            _gru_spec(
                                dataset_key,
                                seed=seed,
                                hidden_dim=hidden_dim,
                                learning_rate=learning_rate,
                                prediction_target=prediction_target,
                                epochs=epochs,
                                batch_size=batch_size,
                            )
                        )
    for dataset_key in dataset_keys:
        for seed in seeds:
            for model_dim in (64, 128):
                for num_heads in (2, 4):
                    for num_layers in (1, 2):
                        for learning_rate in (3e-4, 1e-4):
                            for prediction_target in ("absolute", "delta"):
                                specs.append(
                                    _transformer_spec(
                                        dataset_key,
                                        seed=seed,
                                        model_dim=model_dim,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        learning_rate=learning_rate,
                                        prediction_target=prediction_target,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                    )
                                )
    return specs


def _upgrade_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for dataset_key in dataset_keys:
        for baseline_name in ("persistence", "moving_average", "linear_extrapolation", "mean_delta"):
            specs.append(_array_baseline_spec(dataset_key, baseline_name=baseline_name))
        for prediction_target in ("absolute", "delta"):
            specs.append(
                _linear_latent_spec(
                    dataset_key,
                    prediction_target=prediction_target,
                    batch_size=max(256, int(batch_size)),
                )
            )
    for dataset_key in dataset_keys:
        for seed in seeds:
            for variant in ("residual_pixel_mse", "residual_pixel_residual_mse", "residual_pixel_motion_weighted_mse", "residual_pixel_motion_weighted_huber"):
                for hidden_dim in (64, 128):
                    for learning_rate in (3e-4, 1e-4, 3e-5):
                        for residual_scale in (0.10, 0.25):
                            specs.append(
                                _residual_spec(
                                    dataset_key,
                                    seed=seed,
                                    hidden_dim=hidden_dim,
                                    learning_rate=learning_rate,
                                    residual_scale=residual_scale,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    variant=variant,
                                )
                            )
    for dataset_key in dataset_keys:
        for seed in seeds:
            for variant in ("convgru_pixel_mse", "convgru_pixel_residual_mse", "convgru_pixel_motion_weighted_mse", "convgru_pixel_motion_weighted_huber"):
                for hidden_channels in (16, 32, 64):
                    for learning_rate in (3e-4, 1e-4):
                        for residual_scale in (0.10, 0.25):
                            specs.append(
                                _convgru_spec(
                                    dataset_key,
                                    seed=seed,
                                    hidden_channels=hidden_channels,
                                    learning_rate=learning_rate,
                                    residual_scale=residual_scale,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    variant=variant,
                                )
                            )
    for dataset_key in dataset_keys:
        for seed in seeds:
            for hidden_dim in (64, 128, 256):
                for learning_rate in (1e-3, 3e-4, 1e-4):
                    for prediction_target in ("absolute", "delta"):
                        specs.append(
                            _gru_spec(
                                dataset_key,
                                seed=seed,
                                hidden_dim=hidden_dim,
                                learning_rate=learning_rate,
                                prediction_target=prediction_target,
                                epochs=epochs,
                                batch_size=batch_size,
                            )
                        )
    for dataset_key in dataset_keys:
        for seed in seeds:
            for model_dim in (64, 128):
                for num_heads in (2, 4):
                    for num_layers in (1, 2):
                        for learning_rate in (3e-4, 1e-4):
                            for prediction_target in ("absolute", "delta"):
                                specs.append(
                                    _transformer_spec(
                                        dataset_key,
                                        seed=seed,
                                        model_dim=model_dim,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        learning_rate=learning_rate,
                                        prediction_target=prediction_target,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                    )
                                )
    return specs


def _advanced_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for dataset_key in dataset_keys:
        for seed in seeds:
            for architecture in ("unet_convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"):
                architecture_depth = 2 if architecture == "temporal_cnn_pixel" else 1
                for loss_suffix in ("mse", "residual_mse", "motion_weighted_huber"):
                    variant = f"{architecture}_{loss_suffix}"
                    for hidden_channels in (16, 32):
                        for learning_rate in (3e-4, 1e-4):
                            specs.append(
                                _advanced_pixel_spec(
                                    dataset_key,
                                    architecture=architecture,
                                    variant=variant,
                                    seed=seed,
                                    hidden_channels=hidden_channels,
                                    num_layers=architecture_depth,
                                    learning_rate=learning_rate,
                                    residual_scale=0.10,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                )
                            )
    return specs


def _advanced_big_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    architecture_layers = {
        "convgru_pixel": (2,),
        "unet_convgru_pixel": (1,),
        "convlstm_pixel": (1, 2),
        "temporal_cnn_pixel": (2, 4),
    }
    for dataset_key in dataset_keys:
        for seed in seeds:
            for architecture, layer_options in architecture_layers.items():
                for loss_suffix in ("mse", "residual_mse", "motion_weighted_huber"):
                    variant = f"{architecture}_{loss_suffix}"
                    for num_layers in layer_options:
                        for hidden_channels in (16, 32, 64, 96):
                            for learning_rate in (3e-4, 1e-4, 3e-5):
                                for residual_scale in (0.05, 0.10):
                                    specs.append(
                                        _advanced_pixel_spec(
                                            dataset_key,
                                            architecture=architecture,
                                            variant=variant,
                                            seed=seed,
                                            hidden_channels=hidden_channels,
                                            num_layers=num_layers,
                                            learning_rate=learning_rate,
                                            residual_scale=residual_scale,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                        )
                                    )
    return specs


def _advanced_overnight_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    seen: set[str] = set()

    def add(spec: ExperimentSpec) -> None:
        if spec.experiment_id in seen:
            return
        seen.add(spec.experiment_id)
        specs.append(spec)

    primary_seed = seeds[0]
    confirmation_seeds = seeds[1:]

    for dataset_key in dataset_keys:
        for baseline_name in ("persistence", "moving_average", "linear_extrapolation", "mean_delta"):
            add(_array_baseline_spec(dataset_key, baseline_name=baseline_name))
        for prediction_target in ("absolute", "delta"):
            add(_linear_latent_spec(dataset_key, prediction_target=prediction_target, batch_size=max(256, int(batch_size))))

    scout_architectures = (
        ("convgru_pixel", 1),
        ("convgru_pixel", 2),
        ("unet_convgru_pixel", 1),
        ("convlstm_pixel", 1),
        ("temporal_cnn_pixel", 2),
        ("temporal_cnn_pixel", 4),
    )
    for hidden_channels in (16, 32):
        for loss_suffix in ("mse", "residual_mse", "motion_weighted_huber"):
            for architecture, num_layers in scout_architectures:
                for dataset_key in dataset_keys:
                    add(
                        _advanced_pixel_spec(
                            dataset_key,
                            architecture=architecture,
                            variant=f"{architecture}_{loss_suffix}",
                            seed=primary_seed,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            learning_rate=1e-4,
                            residual_scale=0.10,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

    wide_architectures = (
        ("convgru_pixel", 2),
        ("unet_convgru_pixel", 1),
        ("convlstm_pixel", 2),
        ("temporal_cnn_pixel", 4),
    )
    for hidden_channels in (64, 96):
        for loss_suffix in ("residual_mse", "motion_weighted_huber"):
            for architecture, num_layers in wide_architectures:
                for dataset_key in dataset_keys:
                    add(
                        _advanced_pixel_spec(
                            dataset_key,
                            architecture=architecture,
                            variant=f"{architecture}_{loss_suffix}",
                            seed=primary_seed,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            learning_rate=1e-4,
                            residual_scale=0.10,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

    for hidden_channels in (32, 64):
        for learning_rate in (3e-4, 3e-5):
            for residual_scale in (0.05, 0.10):
                for loss_suffix in ("residual_mse", "motion_weighted_huber"):
                    for architecture, num_layers in wide_architectures:
                        for dataset_key in dataset_keys:
                            add(
                                _advanced_pixel_spec(
                                    dataset_key,
                                    architecture=architecture,
                                    variant=f"{architecture}_{loss_suffix}",
                                    seed=primary_seed,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    learning_rate=learning_rate,
                                    residual_scale=residual_scale,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                )
                            )

    for seed in confirmation_seeds:
        for hidden_channels in (32, 64):
            for loss_suffix in ("residual_mse", "motion_weighted_huber"):
                for architecture, num_layers in wide_architectures:
                    for dataset_key in dataset_keys:
                        add(
                            _advanced_pixel_spec(
                                dataset_key,
                                architecture=architecture,
                                variant=f"{architecture}_{loss_suffix}",
                                seed=seed,
                                hidden_channels=hidden_channels,
                                num_layers=num_layers,
                                learning_rate=1e-4,
                                residual_scale=0.10,
                                epochs=epochs,
                                batch_size=batch_size,
                            )
                        )
    return specs


def _cropped32_restricted_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    seen: set[str] = set()

    def add(spec: ExperimentSpec) -> None:
        if spec.experiment_id in seen:
            return
        seen.add(spec.experiment_id)
        specs.append(spec)

    primary_seed = seeds[0]
    confirmation_seeds = seeds[1:2]
    pixel_architectures = (
        ("convgru_pixel", 1),
        ("unet_convgru_pixel", 1),
        ("convlstm_pixel", 1),
        ("temporal_cnn_pixel", 2),
    )
    wider_pixel_architectures = (
        ("convgru_pixel", 2),
        ("unet_convgru_pixel", 1),
        ("convlstm_pixel", 2),
        ("temporal_cnn_pixel", 4),
    )

    for dataset_key in dataset_keys:
        for baseline_name in ("persistence", "moving_average", "linear_extrapolation", "mean_delta"):
            add(_array_baseline_spec(dataset_key, baseline_name=baseline_name))
        for prediction_target in ("absolute", "delta"):
            add(_linear_latent_spec(dataset_key, prediction_target=prediction_target, batch_size=max(256, int(batch_size))))

        for variant in ("residual_pixel_residual_mse", "residual_pixel_motion_weighted_huber"):
            for hidden_dim in (32, 64):
                add(
                    _residual_spec(
                        dataset_key,
                        seed=primary_seed,
                        hidden_dim=hidden_dim,
                        learning_rate=1e-4,
                        residual_scale=0.10,
                        epochs=epochs,
                        batch_size=batch_size,
                        variant=variant,
                    )
                )

        for hidden_dim in (32, 64):
            for prediction_target in ("absolute", "delta"):
                add(
                    _gru_spec(
                        dataset_key,
                        seed=primary_seed,
                        hidden_dim=hidden_dim,
                        learning_rate=1e-4,
                        prediction_target=prediction_target,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                )

        for prediction_target in ("absolute", "delta"):
            add(
                _transformer_spec(
                    dataset_key,
                    seed=primary_seed,
                    model_dim=32,
                    num_heads=2,
                    num_layers=1,
                    learning_rate=1e-4,
                    prediction_target=prediction_target,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            )

        for loss_suffix in ("residual_mse", "motion_weighted_huber"):
            for architecture, num_layers in pixel_architectures:
                for hidden_channels in (8, 16):
                    add(
                        _advanced_pixel_spec(
                            dataset_key,
                            architecture=architecture,
                            variant=f"{architecture}_{loss_suffix}",
                            seed=primary_seed,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers,
                            learning_rate=1e-4,
                            residual_scale=0.10,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

        for loss_suffix in ("residual_mse", "motion_weighted_huber"):
            for architecture, num_layers in wider_pixel_architectures:
                for residual_scale in (0.05, 0.10):
                    add(
                        _advanced_pixel_spec(
                            dataset_key,
                            architecture=architecture,
                            variant=f"{architecture}_{loss_suffix}",
                            seed=primary_seed,
                            hidden_channels=32,
                            num_layers=num_layers,
                            learning_rate=1e-4,
                            residual_scale=residual_scale,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

        for variant in ("residual_pixel_residual_mse", "residual_pixel_motion_weighted_huber"):
            add(
                _residual_spec(
                    dataset_key,
                    seed=primary_seed,
                    hidden_dim=64,
                    learning_rate=1e-4,
                    residual_scale=0.05,
                    epochs=epochs,
                    batch_size=batch_size,
                    variant=variant,
                )
            )
        for prediction_target in ("absolute", "delta"):
            add(
                _gru_spec(
                    dataset_key,
                    seed=primary_seed,
                    hidden_dim=64,
                    learning_rate=3e-5,
                    prediction_target=prediction_target,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            )

        for seed in confirmation_seeds:
            add(
                _residual_spec(
                    dataset_key,
                    seed=seed,
                    hidden_dim=32,
                    learning_rate=1e-4,
                    residual_scale=0.10,
                    epochs=epochs,
                    batch_size=batch_size,
                    variant="residual_pixel_residual_mse",
                )
            )
            add(_gru_spec(dataset_key, seed=seed, hidden_dim=32, learning_rate=1e-4, prediction_target="delta", epochs=epochs, batch_size=batch_size))
            add(_transformer_spec(dataset_key, seed=seed, model_dim=32, num_heads=2, num_layers=1, learning_rate=1e-4, prediction_target="delta", epochs=epochs, batch_size=batch_size))
            for architecture, num_layers in pixel_architectures:
                add(
                    _advanced_pixel_spec(
                        dataset_key,
                        architecture=architecture,
                        variant=f"{architecture}_residual_mse",
                        seed=seed,
                        hidden_channels=16,
                        num_layers=num_layers,
                        learning_rate=1e-4,
                        residual_scale=0.10,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                )
    return specs


def _cropped32_large_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    seen: set[str] = set()

    def add(spec: ExperimentSpec) -> None:
        if spec.experiment_id in seen:
            return
        seen.add(spec.experiment_id)
        specs.append(spec)

    primary_seed = seeds[0]
    confirmation_seeds = seeds[1:2]
    pixel_architectures = (
        ("convgru_pixel", 1),
        ("convgru_pixel", 2),
        ("unet_convgru_pixel", 1),
        ("convlstm_pixel", 1),
        ("convlstm_pixel", 2),
        ("temporal_cnn_pixel", 2),
        ("temporal_cnn_pixel", 4),
        ("temporal_cnn_pixel", 6),
    )
    wide_pixel_architectures = (
        ("convgru_pixel", 2),
        ("unet_convgru_pixel", 1),
        ("convlstm_pixel", 2),
        ("temporal_cnn_pixel", 4),
        ("temporal_cnn_pixel", 6),
    )
    transformer_configs = (
        (32, 2, 1),
        (32, 2, 2),
        (64, 2, 1),
        (64, 4, 1),
        (64, 4, 2),
        (128, 4, 1),
        (128, 4, 2),
    )

    for dataset_key in dataset_keys:
        for baseline_name in ("persistence", "moving_average", "linear_extrapolation", "mean_delta"):
            add(_array_baseline_spec(dataset_key, baseline_name=baseline_name))
        for prediction_target in ("absolute", "delta"):
            add(_linear_latent_spec(dataset_key, prediction_target=prediction_target, batch_size=max(256, int(batch_size))))

        for variant in ("residual_pixel_residual_mse", "residual_pixel_motion_weighted_huber"):
            for hidden_dim in (32, 64, 128):
                for learning_rate in (3e-4, 1e-4, 3e-5):
                    for residual_scale in (0.025, 0.05, 0.10):
                        add(
                            _residual_spec(
                                dataset_key,
                                seed=primary_seed,
                                hidden_dim=hidden_dim,
                                learning_rate=learning_rate,
                                residual_scale=residual_scale,
                                epochs=epochs,
                                batch_size=batch_size,
                                variant=variant,
                            )
                        )
        for hidden_dim in (32, 64):
            for residual_scale in (0.05, 0.10):
                add(
                    _residual_spec(
                        dataset_key,
                        seed=primary_seed,
                        hidden_dim=hidden_dim,
                        learning_rate=1e-4,
                        residual_scale=residual_scale,
                        epochs=epochs,
                        batch_size=batch_size,
                        variant="residual_pixel_mse",
                    )
                )

        for hidden_dim in (32, 64, 128, 256):
            for learning_rate in (3e-4, 1e-4, 3e-5):
                for prediction_target in ("absolute", "delta"):
                    add(
                        _gru_spec(
                            dataset_key,
                            seed=primary_seed,
                            hidden_dim=hidden_dim,
                            learning_rate=learning_rate,
                            prediction_target=prediction_target,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

        for model_dim, num_heads, num_layers in transformer_configs:
            for learning_rate in (1e-4, 3e-5):
                for prediction_target in ("absolute", "delta"):
                    add(
                        _transformer_spec(
                            dataset_key,
                            seed=primary_seed,
                            model_dim=model_dim,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            learning_rate=learning_rate,
                            prediction_target=prediction_target,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

        for loss_suffix in ("residual_mse", "motion_weighted_huber"):
            for architecture, num_layers in pixel_architectures:
                for hidden_channels in (16, 32, 64):
                    for learning_rate in (1e-4, 3e-5):
                        for residual_scale in (0.05, 0.10):
                            add(
                                _advanced_pixel_spec(
                                    dataset_key,
                                    architecture=architecture,
                                    variant=f"{architecture}_{loss_suffix}",
                                    seed=primary_seed,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    learning_rate=learning_rate,
                                    residual_scale=residual_scale,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                )
                            )

        for architecture, num_layers in pixel_architectures:
            add(
                _advanced_pixel_spec(
                    dataset_key,
                    architecture=architecture,
                    variant=f"{architecture}_mse",
                    seed=primary_seed,
                    hidden_channels=32,
                    num_layers=num_layers,
                    learning_rate=1e-4,
                    residual_scale=0.10,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            )

        for loss_suffix in ("residual_mse", "motion_weighted_huber"):
            for architecture, num_layers in wide_pixel_architectures:
                for hidden_channels in (96, 128):
                    for residual_scale in (0.05, 0.10):
                        add(
                            _advanced_pixel_spec(
                                dataset_key,
                                architecture=architecture,
                                variant=f"{architecture}_{loss_suffix}",
                                seed=primary_seed,
                                hidden_channels=hidden_channels,
                                num_layers=num_layers,
                                learning_rate=1e-4,
                                residual_scale=residual_scale,
                                epochs=epochs,
                                batch_size=batch_size,
                            )
                        )

        for seed in confirmation_seeds:
            for variant in ("residual_pixel_residual_mse", "residual_pixel_motion_weighted_huber"):
                for residual_scale in (0.05, 0.10):
                    add(
                        _residual_spec(
                            dataset_key,
                            seed=seed,
                            hidden_dim=64,
                            learning_rate=1e-4,
                            residual_scale=residual_scale,
                            epochs=epochs,
                            batch_size=batch_size,
                            variant=variant,
                        )
                    )
            for hidden_dim in (64, 128):
                add(_gru_spec(dataset_key, seed=seed, hidden_dim=hidden_dim, learning_rate=1e-4, prediction_target="delta", epochs=epochs, batch_size=batch_size))
            for model_dim, num_heads, num_layers in ((64, 4, 1), (64, 4, 2)):
                add(_transformer_spec(dataset_key, seed=seed, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, learning_rate=1e-4, prediction_target="delta", epochs=epochs, batch_size=batch_size))
            for loss_suffix in ("residual_mse", "motion_weighted_huber"):
                for architecture, num_layers in wide_pixel_architectures:
                    for hidden_channels in (32, 64):
                        for residual_scale in (0.05, 0.10):
                            add(
                                _advanced_pixel_spec(
                                    dataset_key,
                                    architecture=architecture,
                                    variant=f"{architecture}_{loss_suffix}",
                                    seed=seed,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    learning_rate=1e-4,
                                    residual_scale=residual_scale,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                )
                            )
    return specs


def _highres_temporal_cnn_scalable_specs(*, seeds: tuple[int, ...], epochs: int, batch_size: int, dataset_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    seen: set[str] = set()

    def add(spec: ExperimentSpec) -> None:
        if spec.experiment_id in seen:
            return
        seen.add(spec.experiment_id)
        specs.append(spec)

    for dataset_key in dataset_keys:
        for seed in seeds:
            for architecture_spec in architecture_catalog():
                for loss_mode in ("residual_mse", "motion_weighted_huber"):
                    for residual_scale in (0.05, 0.10):
                        for learning_rate in (1e-4, 3e-5):
                            add(
                                _scalable_temporal_cnn_spec(
                                    dataset_key,
                                    architecture_spec=architecture_spec,
                                    seed=seed,
                                    loss_mode=loss_mode,
                                    learning_rate=learning_rate,
                                    residual_scale=residual_scale,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                )
                            )
    return specs


def _scalable_temporal_cnn_spec(
    dataset_key: str,
    *,
    architecture_spec: Mapping[str, Any],
    seed: int,
    loss_mode: str,
    learning_rate: float,
    residual_scale: float,
    epochs: int,
    batch_size: int,
) -> ExperimentSpec:
    spec = dict(architecture_spec)
    architecture_id = _slug_token(spec.get("architecture_id") or "scalable_tcnn")
    loss_slug = _slug_token(loss_mode)
    params = {
        "architecture_spec": spec,
        "architecture_id": architecture_id,
        "loss_mode": str(loss_mode),
        "learning_rate": float(learning_rate),
        "residual_scale": float(residual_scale),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "active_weight": 5.0,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
    }
    exp_id = (
        f"scalable_tcnn_{dataset_key}_{architecture_id}_{loss_slug}"
        f"_lr{_slug_float(learning_rate)}_rs{_slug_float(residual_scale)}_e{epochs}_s{seed}"
    )
    return ExperimentSpec(exp_id, "scalable_temporal_cnn_pixel", dataset_key, int(seed), params)


def _residual_spec(
    dataset_key: str,
    *,
    seed: int,
    hidden_dim: int,
    learning_rate: float,
    residual_scale: float,
    epochs: int,
    batch_size: int,
    variant: str = "residual_pixel_mse",
) -> ExperimentSpec:
    loss_mode = _loss_mode_for_variant(variant)
    params = {
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "residual_scale": float(residual_scale),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "variant": str(variant),
        "loss_mode": loss_mode,
    }
    variant_suffix = "" if variant == "residual_pixel_mse" else "_" + _slug_token(variant.removeprefix("residual_pixel_"))
    exp_id = f"residual_{dataset_key}{variant_suffix}_hd{hidden_dim}_lr{_slug_float(learning_rate)}_rs{_slug_float(residual_scale)}_e{epochs}_s{seed}"
    return ExperimentSpec(exp_id, "residual_pixel", dataset_key, int(seed), params)


def _convgru_spec(
    dataset_key: str,
    *,
    seed: int,
    hidden_channels: int,
    num_layers: int = 1,
    learning_rate: float,
    residual_scale: float,
    epochs: int,
    batch_size: int,
    variant: str = "convgru_pixel_mse",
) -> ExperimentSpec:
    loss_mode = _loss_mode_for_variant(variant)
    params = {
        "hidden_channels": int(hidden_channels),
        "num_layers": int(num_layers),
        "learning_rate": float(learning_rate),
        "residual_scale": float(residual_scale),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "variant": str(variant),
        "loss_mode": loss_mode,
    }
    variant_suffix = _slug_token(variant.removeprefix("convgru_pixel_"))
    layer_suffix = "" if int(num_layers) == 1 else f"_l{int(num_layers)}"
    exp_id = f"convgru_{dataset_key}_{variant_suffix}_hc{hidden_channels}{layer_suffix}_lr{_slug_float(learning_rate)}_rs{_slug_float(residual_scale)}_e{epochs}_s{seed}"
    return ExperimentSpec(exp_id, "convgru_pixel", dataset_key, int(seed), params)


def _advanced_pixel_spec(
    dataset_key: str,
    *,
    architecture: str,
    variant: str,
    seed: int,
    hidden_channels: int,
    num_layers: int = 1,
    learning_rate: float,
    residual_scale: float,
    epochs: int,
    batch_size: int,
) -> ExperimentSpec:
    loss_mode = _loss_mode_for_variant(variant)
    params = {
        "architecture": str(architecture),
        "variant": str(variant),
        "hidden_channels": int(hidden_channels),
        "num_layers": int(num_layers),
        "learning_rate": float(learning_rate),
        "residual_scale": float(residual_scale),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "loss_mode": loss_mode,
    }
    arch_slug = _slug_token(str(architecture).removesuffix("_pixel"))
    loss_slug = _slug_token(str(variant).removeprefix(str(architecture) + "_"))
    layer_suffix = "" if int(num_layers) == 1 else f"_l{int(num_layers)}"
    exp_id = f"{arch_slug}_{dataset_key}_{loss_slug}_hc{hidden_channels}{layer_suffix}_lr{_slug_float(learning_rate)}_rs{_slug_float(residual_scale)}_e{epochs}_s{seed}"
    return ExperimentSpec(exp_id, str(architecture), dataset_key, int(seed), params)


def _linear_latent_spec(dataset_key: str, *, prediction_target: str, batch_size: int) -> ExperimentSpec:
    params = {
        "prediction_target": str(prediction_target),
        "alphas": [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "batch_size": int(batch_size),
    }
    exp_id = f"linear_latent_{dataset_key}_{prediction_target}"
    return ExperimentSpec(exp_id, "linear_latent", dataset_key, 0, params)


def _array_baseline_spec(dataset_key: str, *, baseline_name: str) -> ExperimentSpec:
    params = {"baseline_name": str(baseline_name)}
    exp_id = f"array_baseline_{dataset_key}_{_slug_token(baseline_name)}"
    return ExperimentSpec(exp_id, "array_baseline", dataset_key, 0, params)


def _gru_spec(dataset_key: str, *, seed: int, hidden_dim: int, learning_rate: float, prediction_target: str, epochs: int, batch_size: int) -> ExperimentSpec:
    params = {
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "prediction_target": prediction_target,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }
    exp_id = f"gru_{dataset_key}_{prediction_target}_hd{hidden_dim}_lr{_slug_float(learning_rate)}_e{epochs}_s{seed}"
    return ExperimentSpec(exp_id, "latent_gru", dataset_key, int(seed), params)


def _transformer_spec(
    dataset_key: str,
    *,
    seed: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    learning_rate: float,
    prediction_target: str,
    epochs: int,
    batch_size: int,
) -> ExperimentSpec:
    params = {
        "model_dim": int(model_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout": 0.1,
        "learning_rate": float(learning_rate),
        "prediction_target": prediction_target,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }
    exp_id = (
        f"transformer_{dataset_key}_{prediction_target}_md{model_dim}_h{num_heads}_l{num_layers}"
        f"_lr{_slug_float(learning_rate)}_e{epochs}_s{seed}"
    )
    return ExperimentSpec(exp_id, "latent_transformer", dataset_key, int(seed), params)


def run_one(*, spec: ExperimentSpec, out_dir: Path, device: str, datasets: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_config.json").write_text(json.dumps(spec.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cfg = (datasets or DEFAULT_DATASETS)[spec.dataset_key]
    dataset = _load_json(cfg["dataset"])
    autoencoder_run_cache: dict[str, Any] | None = None

    def _require_autoencoder_run() -> dict[str, Any]:
        nonlocal autoencoder_run_cache
        if autoencoder_run_cache is None:
            path = cfg.get("autoencoder_run")
            if not path:
                raise ValueError(f"Experiment {spec.experiment_id} requires an autoencoder_run for dataset {spec.dataset_key}.")
            autoencoder_run_cache = _load_json(path)
        return autoencoder_run_cache
    if spec.kind == "residual_pixel":
        variant = str(spec.params.get("variant", "residual_pixel_mse"))
        run_concept_tests(
            dataset=dataset,
            autoencoder_run=_require_autoencoder_run(),
            out_dir=out_dir,
            variants=(variant,),
            hidden_dim=int(spec.params["hidden_dim"]),
            epochs=int(spec.params["epochs"]),
            batch_size=int(spec.params["batch_size"]),
            learning_rate=float(spec.params["learning_rate"]),
            residual_scale=float(spec.params["residual_scale"]),
            loss_mode=str(spec.params.get("loss_mode", "auto")),
            seed=int(spec.seed),
            device=device,
        )
        return out_dir / variant / "concept_metrics.json"
    if spec.kind == "convgru_pixel":
        variant = str(spec.params.get("variant", "convgru_pixel_mse"))
        run_concept_tests(
            dataset=dataset,
            autoencoder_run=_require_autoencoder_run(),
            out_dir=out_dir,
            variants=(variant,),
            hidden_dim=int(spec.params["hidden_channels"]),
            conv_hidden_channels=int(spec.params["hidden_channels"]),
            conv_layers=int(spec.params.get("num_layers", 1)),
            epochs=int(spec.params["epochs"]),
            batch_size=int(spec.params["batch_size"]),
            learning_rate=float(spec.params["learning_rate"]),
            residual_scale=float(spec.params["residual_scale"]),
            loss_mode=str(spec.params.get("loss_mode", "auto")),
            seed=int(spec.seed),
            device=device,
        )
        return out_dir / variant / "concept_metrics.json"
    if spec.kind in {"unet_convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}:
        variant = str(spec.params.get("variant", f"{spec.kind}_mse"))
        run_concept_tests(
            dataset=dataset,
            autoencoder_run=_require_autoencoder_run(),
            out_dir=out_dir,
            variants=(variant,),
            hidden_dim=int(spec.params["hidden_channels"]),
            conv_hidden_channels=int(spec.params["hidden_channels"]),
            conv_layers=int(spec.params.get("num_layers", 1)),
            epochs=int(spec.params["epochs"]),
            batch_size=int(spec.params["batch_size"]),
            learning_rate=float(spec.params["learning_rate"]),
            residual_scale=float(spec.params["residual_scale"]),
            loss_mode=str(spec.params.get("loss_mode", "auto")),
            seed=int(spec.seed),
            device=device,
        )
        return out_dir / variant / "concept_metrics.json"
    if spec.kind == "scalable_temporal_cnn_pixel":
        variant_out = out_dir / "scalable_temporal_cnn_pixel"
        train_scalable_temporal_cnn(
            dataset=dataset,
            out_dir=variant_out,
            architecture_spec=spec.params["architecture_spec"],
            epochs=int(spec.params["epochs"]),
            batch_size=int(spec.params["batch_size"]),
            learning_rate=float(spec.params["learning_rate"]),
            residual_scale=float(spec.params["residual_scale"]),
            loss_mode=str(spec.params.get("loss_mode", "residual_mse")),
            active_weight=float(spec.params.get("active_weight", 5.0)),
            weight_decay=float(spec.params.get("weight_decay", 1e-4)),
            gradient_clip_norm=float(spec.params.get("gradient_clip_norm", 1.0)),
            seed=int(spec.seed),
            device=device,
        )
        return variant_out / "concept_metrics.json"
    if spec.kind == "linear_latent":
        evaluate_linear_latent_baseline(
            dataset=dataset,
            autoencoder_run=_require_autoencoder_run(),
            out_dir=out_dir,
            prediction_target=str(spec.params.get("prediction_target", "delta")),
            alphas=tuple(float(v) for v in spec.params.get("alphas", (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0))),
            batch_size=int(spec.params.get("batch_size", 256)),
            device=device,
        )
        return out_dir / "linear_latent_metrics.json"
    if spec.kind == "array_baseline":
        return _write_array_baseline_metrics(dataset=dataset, out_dir=out_dir, baseline_name=str(spec.params["baseline_name"]))
    if spec.kind == "latent_gru":
        train_latent_rnn(
            dataset=dataset,
            autoencoder_run=_require_autoencoder_run(),
            out_dir=out_dir,
            window_frames=int(cfg["window_frames"]),
            hidden_dim=int(spec.params["hidden_dim"]),
            epochs=int(spec.params["epochs"]),
            batch_size=int(spec.params["batch_size"]),
            learning_rate=float(spec.params["learning_rate"]),
            prediction_target=str(spec.params["prediction_target"]),
            seed=int(spec.seed),
            device=device,
        )
        return out_dir / "latent_rnn_metrics.json"
    if spec.kind == "latent_transformer":
        train_latent_transformer(
            dataset=dataset,
            autoencoder_run=_require_autoencoder_run(),
            out_dir=out_dir,
            window_frames=int(cfg["window_frames"]),
            model_dim=int(spec.params["model_dim"]),
            num_heads=int(spec.params["num_heads"]),
            num_layers=int(spec.params["num_layers"]),
            dropout=float(spec.params["dropout"]),
            epochs=int(spec.params["epochs"]),
            batch_size=int(spec.params["batch_size"]),
            learning_rate=float(spec.params["learning_rate"]),
            prediction_target=str(spec.params["prediction_target"]),
            seed=int(spec.seed),
            device=device,
        )
        return out_dir / "latent_transformer_metrics.json"
    raise ValueError(f"Unsupported experiment kind: {spec.kind}")


def _write_array_baseline_metrics(*, dataset: Mapping[str, Any], out_dir: Path, baseline_name: str) -> Path:
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        video_ids = arrays["window_video_ids"].astype(str)
        pred = baseline_prediction(windows, baseline_name)
        persistence = baseline_prediction(windows, "persistence")
    diff = pred - targets
    persistence_diff = persistence - targets
    zero_latent = np.zeros((int(windows.shape[0]), 1), dtype=np.float32)
    split_metrics = _prediction_split_metrics(diff, zero_latent, zero_latent, persistence_diff, video_ids, dataset.get("splits"))
    metrics = {
        "schema_version": 1,
        "objective": f"array_{baseline_name}_baseline",
        "model_kind": "array_baseline",
        "model_family": "array_baseline",
        "baseline_name": str(baseline_name),
        "decoded_prediction_mse": float(np.mean(diff * diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(diff))),
        "persistence_mse": float(np.mean(persistence_diff * persistence_diff)),
        "persistence_mae": float(np.mean(np.abs(persistence_diff))),
        "persistence_baseline": {
            "mse": float(np.mean(persistence_diff * persistence_diff)),
            "mae": float(np.mean(np.abs(persistence_diff))),
            "count": int(windows.shape[0]),
        },
        "split_metrics": split_metrics,
        "training_window_count": 0,
        "evaluation_window_count": int(windows.shape[0]),
        "input_normalization": "finite_clipped_unit_interval",
        "decoded_output_normalization": "clipped_unit_interval",
    }
    _promote_split_metrics(metrics, split_metrics, ["decoded_prediction_mse", "decoded_prediction_mae", "persistence_mse", "window_count"])
    metrics["improvement_over_persistence_mse"] = float(metrics["persistence_mse"] - metrics["decoded_prediction_mse"])
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        if split.get("decoded_prediction_mse") is not None and split.get("persistence_mse") is not None:
            metrics[f"{split_name}_improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    metrics_path = out_dir / "array_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": out_dir.name or f"array_{baseline_name}_baseline",
        "model_kind": "array_baseline",
        "baseline_name": str(baseline_name),
        "source_dataset": str(dataset.get("array_path")),
        "metrics_path": str(metrics_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "array_baseline_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_path


def train_latent_transformer(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    window_frames: int = 8,
    model_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    prediction_target: str = "delta",
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(int(seed))
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prediction_target = _normalize_prediction_target(prediction_target)
    with np.load(dataset["array_path"], allow_pickle=False) as arrays:
        windows = _prepare_model_array(arrays["windows"])
        targets = _prepare_model_array(arrays["targets"])
        window_video_ids = arrays["window_video_ids"].astype(str)
        baseline = evaluate_baselines_from_arrays(arrays)
    train_mask = _split_mask(window_video_ids, dataset.get("splits"), "train", default_all=True)
    if not np.any(train_mask):
        raise ValueError("Latent Transformer training split is empty.")
    ckpt = torch.load(autoencoder_run["checkpoint_path"], map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ckpt, latent_dim)
    base_channels = int(ckpt.get("base_channels", 16))
    ae = GridAutoencoder(input_channels=int(windows.shape[2]), latent_dim=latent_dim, base_channels=base_channels, input_shape=tuple(ckpt.get("input_shape") or windows.shape[2:])).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()
    model = LatentTransformerPredictor(
        latent_dim=latent_dim,
        model_dim=int(model_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        dropout=float(dropout),
        max_window_frames=max(64, int(windows.shape[1])),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    xw = torch.from_numpy(windows).to(device)
    yt = torch.from_numpy(targets).to(device)
    train_indices = torch.nonzero(torch.as_tensor(train_mask, dtype=torch.bool, device=device), as_tuple=False).reshape(-1)
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    encode_batch_frames = max(int(batch_size) * int(window_frames), int(batch_size))
    with torch.no_grad():
        target_z_raw = _encode_frames_batched(ae, yt, batch_size=encode_batch_frames)
        target_z = (target_z_raw - latent_mean) / latent_std
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        model.train()
        perm = train_indices[torch.randperm(train_indices.shape[0], device=device)]
        epoch_losses = []
        for start in range(0, perm.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            batch = xw[idx]
            b, w, c, h, ww = batch.shape
            with torch.no_grad():
                z_window_raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
                z_window = (z_window_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_step = model(z_window)
            if prediction_target == "delta":
                target_step = target_z[idx] - z_window[:, -1, :]
            else:
                target_step = target_z[idx]
            loss = torch.mean((pred_step - target_step) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    model.eval()
    pred_z_chunks = []
    pred_z_raw_chunks = []
    pred_x_chunks = []
    with torch.no_grad():
        b, w, c, h, ww = xw.shape
        for start in range(0, b, int(batch_size)):
            batch = xw[start : start + int(batch_size)]
            bb = batch.shape[0]
            z_window_raw = ae.encode(batch.reshape(bb * w, c, h, ww)).reshape(bb, w, latent_dim)
            z_window = (z_window_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            pred_step_batch = model(z_window)
            if prediction_target == "delta":
                pred_z_batch = z_window[:, -1, :] + pred_step_batch
            else:
                pred_z_batch = pred_step_batch
            pred_z_raw_batch = pred_z_batch * latent_std + latent_mean
            pred_x_batch = _prepare_model_array(ae.decode(pred_z_raw_batch).detach().cpu().numpy())
            pred_z_chunks.append(pred_z_batch.detach().cpu().numpy().astype(np.float32))
            pred_z_raw_chunks.append(pred_z_raw_batch.detach().cpu().numpy().astype(np.float32))
            pred_x_chunks.append(pred_x_batch)
    pred_x = np.concatenate(pred_x_chunks, axis=0) if pred_x_chunks else np.zeros(targets.shape, dtype=np.float32)
    target_z_np = target_z.detach().cpu().numpy().astype(np.float32)
    pred_z_np = np.concatenate(pred_z_chunks, axis=0) if pred_z_chunks else np.zeros((0, latent_dim), dtype=np.float32)
    target_z_raw_np = target_z_raw.detach().cpu().numpy().astype(np.float32)
    pred_z_raw_np = np.concatenate(pred_z_raw_chunks, axis=0) if pred_z_raw_chunks else np.zeros((0, latent_dim), dtype=np.float32)
    diff = pred_x - targets
    latent_diff = pred_z_np - target_z_np
    latent_raw_diff = pred_z_raw_np - target_z_raw_np
    persistence_diff = windows[:, -1] - targets
    split_metrics = _prediction_split_metrics(diff, latent_diff, latent_raw_diff, persistence_diff, window_video_ids, dataset.get("splits"))
    metrics = {
        "schema_version": 1,
        "objective": "transformer_next_delta_code_mse" if prediction_target == "delta" else "transformer_next_code_mse",
        "prediction_target": prediction_target,
        "model_kind": "latent_transformer_predictor",
        "model_dim": int(model_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "training_loss": losses,
        "training_latent_mse": losses,
        "training_window_count": int(train_mask.sum()),
        "evaluation_window_count": int(windows.shape[0]),
        "latent_code_mse": float(np.mean(latent_diff * latent_diff)),
        "latent_code_mae": float(np.mean(np.abs(latent_diff))),
        "latent_code_raw_mse": float(np.mean(latent_raw_diff * latent_raw_diff)),
        "latent_code_raw_mae": float(np.mean(np.abs(latent_raw_diff))),
        "input_normalization": "finite_clipped_unit_interval",
        "decoded_output_normalization": "sigmoid_unit_interval",
        "latent_code_normalization": "standard_score_per_dimension",
        "decoded_prediction_mse": float(np.mean(diff * diff)),
        "decoded_prediction_mae": float(np.mean(np.abs(diff))),
        "prediction_mse": float(np.mean(diff * diff)),
        "prediction_mae": float(np.mean(np.abs(diff))),
        "persistence_baseline": baseline["persistence"],
        "split_metrics": split_metrics,
    }
    _promote_split_metrics(
        metrics,
        split_metrics,
        [
            "latent_code_mse",
            "latent_code_mae",
            "latent_code_raw_mse",
            "latent_code_raw_mae",
            "decoded_prediction_mse",
            "decoded_prediction_mae",
            "persistence_mse",
            "window_count",
        ],
    )
    metrics["improvement_over_persistence_mse"] = float(metrics["persistence_baseline"]["mse"] - metrics["decoded_prediction_mse"])
    for split_name in ("train", "val", "test"):
        split = split_metrics.get(split_name, {})
        if split.get("decoded_prediction_mse") is not None and split.get("persistence_mse") is not None:
            metrics[f"{split_name}_improvement_over_persistence_mse"] = float(split["persistence_mse"] - split["decoded_prediction_mse"])
    metrics_path = out / "latent_transformer_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = out / "latent_transformer_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_dim": latent_dim,
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "objective": metrics["objective"],
            "prediction_target": prediction_target,
            "latent_code_normalization": "standard_score_per_dimension",
        },
        checkpoint,
    )
    run = {
        "schema_version": 1,
        "run_id": out.name or "latent_transformer_v1",
        "model_kind": "latent_transformer_predictor",
        "latent_dim": latent_dim,
        "window_frames": int(window_frames),
        "prediction_horizon_frames": int(dataset.get("windowing", {}).get("prediction_horizon_frames", 1)),
        "model_dim": int(model_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "prediction_target": prediction_target,
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": 1e-4,
            "prediction_target": prediction_target,
            "latent_code_normalization": "standard_score_per_dimension",
        },
        "source_autoencoder_run": str(autoencoder_run.get("checkpoint_path")),
        "source_dataset": str(dataset.get("array_path")),
        "checkpoint_path": str(checkpoint),
        "metrics_path": str(metrics_path),
        "seed": int(seed),
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
    }
    (out / "latent_transformer_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run


def expected_metrics_path(exp_out: Path, spec: ExperimentSpec) -> Path:
    if spec.kind == "residual_pixel":
        return exp_out / str(spec.params.get("variant", "residual_pixel_mse")) / "concept_metrics.json"
    if spec.kind == "convgru_pixel":
        return exp_out / str(spec.params.get("variant", "convgru_pixel_mse")) / "concept_metrics.json"
    if spec.kind in {"unet_convgru_pixel", "convlstm_pixel", "temporal_cnn_pixel"}:
        return exp_out / str(spec.params.get("variant", f"{spec.kind}_mse")) / "concept_metrics.json"
    if spec.kind == "scalable_temporal_cnn_pixel":
        return exp_out / "scalable_temporal_cnn_pixel" / "concept_metrics.json"
    if spec.kind == "linear_latent":
        return exp_out / "linear_latent_metrics.json"
    if spec.kind == "array_baseline":
        return exp_out / "array_baseline_metrics.json"
    if spec.kind == "latent_gru":
        return exp_out / "latent_rnn_metrics.json"
    if spec.kind == "latent_transformer":
        return exp_out / "latent_transformer_metrics.json"
    raise ValueError(f"Unsupported experiment kind: {spec.kind}")


def write_summary(out_dir: str | Path) -> None:
    out = Path(out_dir)
    rows = collect_metric_rows(out)
    header = [
        "rank",
        "experiment_id",
        "kind",
        "dataset_key",
        "seed",
        "val_decoded_prediction_mse",
        "val_persistence_mse",
        "val_improvement_over_persistence_mse",
        "test_decoded_prediction_mse",
        "test_persistence_mse",
        "test_improvement_over_persistence_mse",
        "metrics_path",
    ]
    rows_sorted = sorted(rows, key=lambda r: (-(r.get("val_improvement_over_persistence_mse") or -999.0), r["experiment_id"]))
    lines = ["\t".join(header)]
    for rank, row in enumerate(rows_sorted, start=1):
        values = [str(rank)] + [_tsv(row.get(key)) for key in header[1:]]
        lines.append("\t".join(values))
    (out / "sweep_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md = [
        "# Overnight Dynamics Sweep Summary",
        "",
        f"Experiment root: `{out}`",
        "",
        "Positive improvement means the model beat split-aware persistence.",
        "",
        "| Rank | Experiment | Kind | Dataset | Val pred | Val persist | Val improve | Test pred | Test persist | Test improve |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows_sorted[:50], start=1):
        md.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"`{row['experiment_id']}`",
                    str(row.get("kind", "")),
                    str(row.get("dataset_key", "")),
                    _fmt(row.get("val_decoded_prediction_mse")),
                    _fmt(row.get("val_persistence_mse")),
                    _fmt(row.get("val_improvement_over_persistence_mse")),
                    _fmt(row.get("test_decoded_prediction_mse")),
                    _fmt(row.get("test_persistence_mse")),
                    _fmt(row.get("test_improvement_over_persistence_mse")),
                ]
            )
            + " |"
        )
    md.append("")
    md.append(f"Completed metric files: `{len(rows_sorted)}`")
    (out / "sweep_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def collect_metric_rows(out_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for config_path in sorted(out.glob("*/experiment_config.json")):
        config = _load_json(config_path)
        spec = ExperimentSpec(
            experiment_id=str(config["experiment_id"]),
            kind=str(config["kind"]),
            dataset_key=str(config["dataset_key"]),
            seed=int(config["seed"]),
            params=config.get("params", {}),
        )
        metrics_path = expected_metrics_path(config_path.parent, spec)
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "kind": spec.kind,
                "dataset_key": spec.dataset_key,
                "seed": spec.seed,
                "params": dict(spec.params),
                "objective": metrics.get("objective"),
                "model_kind": metrics.get("model_kind"),
                "model_family": metrics.get("model_family") or spec.kind,
                "loss_mode": metrics.get("loss_mode") or spec.params.get("loss_mode"),
                "val_decoded_prediction_mse": metrics.get("val_decoded_prediction_mse"),
                "val_persistence_mse": metrics.get("val_persistence_mse"),
                "val_improvement_over_persistence_mse": metrics.get("val_improvement_over_persistence_mse"),
                "test_decoded_prediction_mse": metrics.get("test_decoded_prediction_mse"),
                "test_persistence_mse": metrics.get("test_persistence_mse"),
                "test_improvement_over_persistence_mse": metrics.get("test_improvement_over_persistence_mse"),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _cleanup_torch() -> None:
    gc.collect()
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _validate_inputs(datasets: Mapping[str, Mapping[str, Any]]) -> None:
    missing: list[str] = []
    for cfg in datasets.values():
        dataset_path = cfg.get("dataset")
        if not dataset_path or not Path(dataset_path).exists():
            missing.append(str(dataset_path))
        autoencoder_path = cfg.get("autoencoder_run")
        if autoencoder_path and not Path(autoencoder_path).exists():
            missing.append(str(autoencoder_path))
    if missing:
        raise FileNotFoundError("Missing sweep inputs: " + ", ".join(missing))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _loss_mode_for_variant(variant: str) -> str:
    name = str(variant).strip().lower()
    if "residual_mse" in name or "delta_mse" in name:
        return "residual_mse"
    if "huber" in name:
        return "motion_weighted_huber"
    if "motion_weighted" in name or "weighted" in name:
        return "motion_weighted_mse"
    return "frame_mse"


def _slug_token(value: Any) -> str:
    text = str(value).strip().lower().replace(".", "p").replace("-", "m")
    return "_".join(part for part in text.replace("/", "_").replace(" ", "_").split("_") if part)


def _slug_float(value: float) -> str:
    return (f"{float(value):.0e}" if float(value) < 0.001 else f"{float(value):.4f}").replace(".", "p").replace("-", "m")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.9g}"


def _tsv(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ")


def _set_conservative_env() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible overnight grid dynamics sweep.")
    parser.add_argument("--out-dir", default="Outputs/GridModel/060126/overnight_sweep_v1")
    parser.add_argument("--profile", choices=("smoke", "overnight", "upgrade", "advanced", "advanced_big", "advanced_overnight", "cropped32_restricted", "cropped32_large", "highres_temporal_cnn_scalable"), default="overnight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_parse_seeds, default=(7, 13))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--time-limit-hours", type=float, default=None)
    parser.add_argument("--datasets-json", type=Path, default=None, help="Optional dataset mapping JSON with dataset, autoencoder_run, and window_frames per dataset key.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    summary = run_overnight_sweep(
        out_dir=args.out_dir,
        profile=args.profile,
        device=args.device,
        seeds=args.seeds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_runs=args.max_runs,
        time_limit_hours=args.time_limit_hours,
        datasets=_load_json(args.datasets_json) if args.datasets_json else None,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
