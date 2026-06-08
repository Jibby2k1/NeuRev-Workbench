"""Conservative sequential hyperparameter sweeps for latent grid dynamics."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import gc
import json
from itertools import islice, product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench.dynamics.train import train_autoencoder, train_latent_rnn

ProgressCallback = Callable[[str], None]


def run_latent_dynamics_sweep(
    *,
    dataset: Mapping[str, Any],
    out_dir: str | Path,
    latent_dims: Sequence[int] = (16, 32, 64),
    autoencoder_epochs: Sequence[int] = (10, 25),
    autoencoder_learning_rates: Sequence[float] = (1e-3, 3e-4),
    autoencoder_batch_size: int = 64,
    autoencoder_base_channels: Sequence[int] = (16,),
    rnn_hidden_dims: Sequence[int] = (32, 64, 128),
    rnn_epochs: Sequence[int] = (10, 25),
    rnn_learning_rates: Sequence[float] = (1e-3, 3e-4),
    rnn_batch_size: int = 64,
    rnn_prediction_targets: Sequence[str] = ("absolute",),
    max_autoencoders: int | None = 6,
    max_rnn_runs: int | None = 24,
    device: str = "auto",
    seed: int = 7,
    skip_existing: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a capped AE + latent-GRU grid search one candidate at a time.

    The search optimizes the GRU's real training objective first: standardized
    next latent-code MSE. Decoded next-frame MSE is retained as an evaluation
    metric and persistence comparison, but is not used as the training loss.
    """
    out = Path(out_dir)
    ae_root = out / "autoencoders"
    rnn_root = out / "latent_rnns"
    ae_root.mkdir(parents=True, exist_ok=True)
    rnn_root.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    created_at = datetime.now(timezone.utc).isoformat()
    dataset_window_frames = int(dataset.get("windowing", {}).get("window_frames", 0) or 0)
    dataset_info = {
        "dataset_id": str(dataset.get("dataset_id") or ""),
        "array_path": str(dataset.get("array_path") or ""),
        "input_shape": [int(v) for v in dataset.get("input_shape", [])],
        "window_frames": dataset_window_frames,
        "prediction_horizon_frames": int(dataset.get("windowing", {}).get("prediction_horizon_frames", 1)),
        "split_method": str(dataset.get("splits", {}).get("split_method", "")),
        "warnings": list(dataset.get("warnings", []) or []),
    }
    search_config = {
        "latent_dims": [int(v) for v in latent_dims],
        "autoencoder_epochs": [int(v) for v in autoencoder_epochs],
        "autoencoder_learning_rates": [float(v) for v in autoencoder_learning_rates],
        "autoencoder_batch_size": int(autoencoder_batch_size),
        "autoencoder_base_channels": [int(v) for v in autoencoder_base_channels],
        "rnn_hidden_dims": [int(v) for v in rnn_hidden_dims],
        "rnn_epochs": [int(v) for v in rnn_epochs],
        "rnn_learning_rates": [float(v) for v in rnn_learning_rates],
        "rnn_batch_size": int(rnn_batch_size),
        "rnn_prediction_targets": [str(v) for v in rnn_prediction_targets],
        "max_autoencoders": None if max_autoencoders is None else int(max_autoencoders),
        "max_rnn_runs": None if max_rnn_runs is None else int(max_rnn_runs),
        "device_requested": str(device),
        "device_resolved": resolved_device,
        "seed": int(seed),
        "skip_existing": bool(skip_existing),
        "ranking_primary_metric": "selection_latent_code_mse",
        "ranking_secondary_metric": "selection_decoded_prediction_mse",
        "window_frames_note": "RNN window length comes from the provided dynamics dataset; rebuild datasets to sweep this parameter.",
    }
    ae_configs = list(_limited(_autoencoder_configs(latent_dims, autoencoder_epochs, autoencoder_learning_rates, autoencoder_batch_size, autoencoder_base_channels), max_autoencoders))
    rnn_configs = list(_rnn_configs(rnn_hidden_dims, rnn_epochs, rnn_learning_rates, rnn_batch_size, rnn_prediction_targets))
    ae_records: list[dict[str, Any]] = []
    rnn_records: list[dict[str, Any]] = []

    def write_current() -> dict[str, Any]:
        summary = _summary_payload(
            out_dir=out,
            created_at=created_at,
            dataset_info=dataset_info,
            search_config=search_config,
            autoencoders=ae_records,
            latent_rnns=rnn_records,
        )
        _write_summary(out, summary)
        return summary

    for index, config in enumerate(ae_configs, start=1):
        run_dir = ae_root / config["config_id"]
        record = {"kind": "autoencoder", "index": index, "config": dict(config), "run_dir": str(run_dir), "status": "pending"}
        _emit(progress, f"autoencoder {index}/{len(ae_configs)} {config['config_id']}")
        try:
            run = _load_json_if_complete(run_dir / "autoencoder_run.json", run_dir / "autoencoder_metrics.json") if skip_existing else None
            if run is None:
                run = train_autoencoder(
                    dataset=dataset,
                    out_dir=run_dir,
                    latent_dim=int(config["latent_dim"]),
                    base_channels=int(config.get("base_channels", 16)),
                    epochs=int(config["epochs"]),
                    batch_size=int(config["batch_size"]),
                    learning_rate=float(config["learning_rate"]),
                    seed=int(seed),
                    device=resolved_device,
                )
            metrics = _load_json(Path(run["metrics_path"]))
            record.update(
                {
                    "status": "completed",
                    "run_path": str(run_dir / "autoencoder_run.json"),
                    "checkpoint_path": str(run.get("checkpoint_path", "")),
                    "metrics_path": str(run.get("metrics_path", "")),
                    "metrics": _autoencoder_metric_payload(metrics),
                }
            )
            _emit(progress, f"  mse={record['metrics']['mse']:.6g}")
        except Exception as exc:  # pragma: no cover - exercised through integration behavior.
            record.update({"status": "failed", "error": str(exc)})
            _emit(progress, f"  failed: {exc}")
        ae_records.append(record)
        _clear_device_cache(resolved_device)
        write_current()

    successful_aes = sorted((r for r in ae_records if r.get("status") == "completed"), key=lambda r: float(r.get("metrics", {}).get("mse", np.inf)))
    rnn_budget = len(successful_aes) * len(rnn_configs) if max_rnn_runs is None else max(0, int(max_rnn_runs))
    rnn_index = 0
    for ae_record in successful_aes:
        for config in rnn_configs:
            if rnn_index >= rnn_budget:
                break
            rnn_index += 1
            config = dict(config)
            config["autoencoder_config_id"] = ae_record["config"]["config_id"]
            config["dataset_window_frames"] = dataset_window_frames
            config_id = f"rnn_{ae_record['config']['config_id']}__{config['config_id']}"
            run_dir = rnn_root / config_id
            record = {"kind": "latent_rnn", "index": rnn_index, "config": config, "run_dir": str(run_dir), "status": "pending"}
            _emit(progress, f"latent-rnn {rnn_index}/{rnn_budget} {config_id}")
            try:
                run = _load_json_if_complete(run_dir / "latent_rnn_run.json", run_dir / "latent_rnn_metrics.json") if skip_existing else None
                if run is None:
                    run = train_latent_rnn(
                        dataset=dataset,
                        autoencoder_run=_load_json(Path(ae_record["run_path"])),
                        out_dir=run_dir,
                        window_frames=dataset_window_frames,
                        hidden_dim=int(config["hidden_dim"]),
                        epochs=int(config["epochs"]),
                        batch_size=int(config["batch_size"]),
                        learning_rate=float(config["learning_rate"]),
                        prediction_target=str(config.get("prediction_target", "absolute")),
                        seed=int(seed),
                        device=resolved_device,
                    )
                metrics = _load_json(Path(run["metrics_path"]))
                record.update(
                    {
                        "status": "completed",
                        "run_path": str(run_dir / "latent_rnn_run.json"),
                        "checkpoint_path": str(run.get("checkpoint_path", "")),
                        "metrics_path": str(run.get("metrics_path", "")),
                        "metrics": _rnn_metric_payload(metrics),
                    }
                )
                _emit(progress, f"  latent_mse={record['metrics']['latent_code_mse']:.6g} decoded_mse={record['metrics']['decoded_prediction_mse']:.6g}")
            except Exception as exc:  # pragma: no cover - exercised through integration behavior.
                record.update({"status": "failed", "error": str(exc)})
                _emit(progress, f"  failed: {exc}")
            rnn_records.append(record)
            _clear_device_cache(resolved_device)
            write_current()
        if rnn_index >= rnn_budget:
            break
    return write_current()


def _autoencoder_configs(latent_dims: Sequence[int], epochs: Sequence[int], learning_rates: Sequence[float], batch_size: int, base_channels: Sequence[int]):
    for epoch, lr, latent_dim, base in product(epochs, learning_rates, latent_dims, base_channels):
        yield {
            "config_id": f"ae_ld{int(latent_dim)}_bc{int(base)}_e{int(epoch)}_lr{_slug(lr)}_b{int(batch_size)}",
            "latent_dim": int(latent_dim),
            "base_channels": int(base),
            "epochs": int(epoch),
            "learning_rate": float(lr),
            "batch_size": int(batch_size),
        }


def _rnn_configs(hidden_dims: Sequence[int], epochs: Sequence[int], learning_rates: Sequence[float], batch_size: int, prediction_targets: Sequence[str]):
    for epoch, lr, hidden_dim, target in product(epochs, learning_rates, hidden_dims, prediction_targets):
        target = str(target)
        yield {
            "config_id": f"hd{int(hidden_dim)}_pt{target}_e{int(epoch)}_lr{_slug(lr)}_b{int(batch_size)}",
            "hidden_dim": int(hidden_dim),
            "prediction_target": target,
            "epochs": int(epoch),
            "learning_rate": float(lr),
            "batch_size": int(batch_size),
        }


def _limited(items, limit: int | None):
    if limit is None:
        yield from items
    else:
        yield from islice(items, max(0, int(limit)))


def _resolve_device(device: str) -> str:
    requested = str(device or "cpu")
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def _clear_device_cache(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            return


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_complete(run_path: Path, metrics_path: Path) -> dict[str, Any] | None:
    if run_path.is_file() and metrics_path.is_file():
        return _load_json(run_path)
    return None


def _autoencoder_metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mse": float(metrics.get("mse", np.inf)),
        "mae": float(metrics.get("mae", np.inf)),
        "input_normalization": str(metrics.get("input_normalization", "")),
        "output_normalization": str(metrics.get("output_normalization", "")),
        "latent_code_normalization": str(metrics.get("latent_code_normalization", "")),
        "training_loss_final": _last_float(metrics.get("training_loss")),
        "prediction_target": str(metrics.get("prediction_target", "")),
    }


def _rnn_metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    persistence = metrics.get("persistence_baseline", {}) if isinstance(metrics.get("persistence_baseline"), Mapping) else {}
    val_latent = _optional_metric(metrics.get("val_latent_code_mse"))
    val_decoded = _optional_metric(metrics.get("val_decoded_prediction_mse"))
    all_latent = _optional_metric(metrics.get("latent_code_mse"), default=np.inf)
    all_decoded = _optional_metric(metrics.get("decoded_prediction_mse"), default=np.inf)
    return {
        "selection_latent_code_mse": val_latent if np.isfinite(val_latent) else all_latent,
        "selection_decoded_prediction_mse": val_decoded if np.isfinite(val_decoded) else all_decoded,
        "latent_code_mse": all_latent,
        "latent_code_mae": _optional_metric(metrics.get("latent_code_mae"), default=np.inf),
        "latent_code_raw_mse": _optional_metric(metrics.get("latent_code_raw_mse"), default=np.inf),
        "decoded_prediction_mse": all_decoded,
        "decoded_prediction_mae": _optional_metric(metrics.get("decoded_prediction_mae"), default=np.inf),
        "train_latent_code_mse": _optional_metric(metrics.get("train_latent_code_mse")),
        "val_latent_code_mse": val_latent,
        "test_latent_code_mse": _optional_metric(metrics.get("test_latent_code_mse")),
        "train_decoded_prediction_mse": _optional_metric(metrics.get("train_decoded_prediction_mse")),
        "val_decoded_prediction_mse": val_decoded,
        "test_decoded_prediction_mse": _optional_metric(metrics.get("test_decoded_prediction_mse")),
        "persistence_mse": _optional_metric(persistence.get("mse"), default=np.inf),
        "val_persistence_mse": _optional_metric(metrics.get("val_persistence_mse")),
        "test_persistence_mse": _optional_metric(metrics.get("test_persistence_mse")),
        "improvement_over_persistence_mse": _optional_metric(metrics.get("improvement_over_persistence_mse"), default=np.nan),
        "val_improvement_over_persistence_mse": _optional_metric(metrics.get("val_improvement_over_persistence_mse")),
        "test_improvement_over_persistence_mse": _optional_metric(metrics.get("test_improvement_over_persistence_mse")),
        "latent_code_normalization": str(metrics.get("latent_code_normalization", "")),
        "decoded_output_normalization": str(metrics.get("decoded_output_normalization", "")),
        "training_loss_final": _last_float(metrics.get("training_loss")),
    }


def _optional_metric(value: Any, *, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _last_float(values: Any) -> float:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
        return float(values[-1])
    return float("nan")


def _summary_payload(
    *,
    out_dir: Path,
    created_at: str,
    dataset_info: Mapping[str, Any],
    search_config: Mapping[str, Any],
    autoencoders: list[dict[str, Any]],
    latent_rnns: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_ae = [r for r in autoencoders if r.get("status") == "completed"]
    completed_rnn = [r for r in latent_rnns if r.get("status") == "completed"]
    return {
        "schema_version": 1,
        "sweep_kind": "latent_dynamics_hyperparameter_search",
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "dataset": dict(dataset_info),
        "search_config": dict(search_config),
        "counts": {
            "autoencoder_candidates": len(autoencoders),
            "autoencoder_completed": len(completed_ae),
            "autoencoder_failed": sum(1 for r in autoencoders if r.get("status") == "failed"),
            "latent_rnn_candidates": len(latent_rnns),
            "latent_rnn_completed": len(completed_rnn),
            "latent_rnn_failed": sum(1 for r in latent_rnns if r.get("status") == "failed"),
        },
        "best": {
            "autoencoder_by_reconstruction_mse": _best_record(completed_ae, "mse"),
            "latent_rnn_by_selection_latent_code_mse": _best_record(completed_rnn, "selection_latent_code_mse"),
            "latent_rnn_by_latent_code_mse": _best_record(completed_rnn, "latent_code_mse"),
            "latent_rnn_by_decoded_prediction_mse": _best_record(completed_rnn, "decoded_prediction_mse"),
        },
        "autoencoders": autoencoders,
        "latent_rnns": latent_rnns,
    }


def _best_record(records: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if not records:
        return None
    best = min(records, key=lambda r: float(r.get("metrics", {}).get(metric, np.inf)))
    config = dict(best.get("config", {}))
    config_id = str(config.get("config_id", ""))
    if config.get("autoencoder_config_id"):
        config_id = f"{config['autoencoder_config_id']}__{config_id}"
    return {
        "config_id": config_id,
        "run_path": str(best.get("run_path", "")),
        "metrics_path": str(best.get("metrics_path", "")),
        "metric": metric,
        "value": float(best.get("metrics", {}).get(metric, np.inf)),
        "config": config,
    }


def _write_summary(out_dir: Path, summary: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(out_dir / "sweep_results.tsv", summary)


def _write_tsv(path: Path, summary: Mapping[str, Any]) -> None:
    rows = []
    for record in list(summary.get("autoencoders", []) or []) + list(summary.get("latent_rnns", []) or []):
        config = record.get("config", {}) if isinstance(record.get("config"), Mapping) else {}
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), Mapping) else {}
        rows.append(
            {
                "kind": record.get("kind", ""),
                "status": record.get("status", ""),
                "config_id": config.get("config_id", ""),
                "autoencoder_config_id": config.get("autoencoder_config_id", ""),
                "latent_dim": config.get("latent_dim", ""),
                "base_channels": config.get("base_channels", ""),
                "prediction_target": config.get("prediction_target", ""),
                "ae_epochs": config.get("epochs", "") if record.get("kind") == "autoencoder" else "",
                "ae_learning_rate": config.get("learning_rate", "") if record.get("kind") == "autoencoder" else "",
                "rnn_hidden_dim": config.get("hidden_dim", ""),
                "rnn_epochs": config.get("epochs", "") if record.get("kind") == "latent_rnn" else "",
                "rnn_learning_rate": config.get("learning_rate", "") if record.get("kind") == "latent_rnn" else "",
                "autoencoder_mse": metrics.get("mse", ""),
                "selection_latent_code_mse": metrics.get("selection_latent_code_mse", ""),
                "latent_code_mse": metrics.get("latent_code_mse", ""),
                "val_latent_code_mse": metrics.get("val_latent_code_mse", ""),
                "test_latent_code_mse": metrics.get("test_latent_code_mse", ""),
                "selection_decoded_prediction_mse": metrics.get("selection_decoded_prediction_mse", ""),
                "decoded_prediction_mse": metrics.get("decoded_prediction_mse", ""),
                "val_decoded_prediction_mse": metrics.get("val_decoded_prediction_mse", ""),
                "test_decoded_prediction_mse": metrics.get("test_decoded_prediction_mse", ""),
                "persistence_mse": metrics.get("persistence_mse", ""),
                "improvement_over_persistence_mse": metrics.get("improvement_over_persistence_mse", ""),
                "run_path": record.get("run_path", ""),
                "error": record.get("error", ""),
            }
        )
    fields = [
        "kind",
        "status",
        "config_id",
        "autoencoder_config_id",
        "latent_dim",
        "base_channels",
        "prediction_target",
        "ae_epochs",
        "ae_learning_rate",
        "rnn_hidden_dim",
        "rnn_epochs",
        "rnn_learning_rate",
        "autoencoder_mse",
        "selection_latent_code_mse",
        "latent_code_mse",
        "val_latent_code_mse",
        "test_latent_code_mse",
        "selection_decoded_prediction_mse",
        "decoded_prediction_mse",
        "val_decoded_prediction_mse",
        "test_decoded_prediction_mse",
        "persistence_mse",
        "improvement_over_persistence_mse",
        "run_path",
        "error",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(_tsv_cell(row.get(field, "")) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tsv_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _slug(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
