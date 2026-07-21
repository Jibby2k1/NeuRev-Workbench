#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurobench.dynamics.train import train_latent_rnn, _torch

ROOT = Path("Outputs/GridModel/060126_crop512_grid128_max_v1")
AE_RUN_PATH = ROOT / "models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json"
OUT_ROOT = ROOT / "directional_rnn_train_test_sweep_v2"
DATASET_KEYS = {
    "left": {
        "source": "w8_s1_h2_left_only_rnn_v1",
        "target": "w8_s1_h2_left_train_test_rnn_v2",
        "test_video_ids": ["8 left"],
    },
    "right": {
        "source": "w8_s1_h2_right_only_rnn_v1",
        "target": "w8_s1_h2_right_train_test_rnn_v2",
        "test_video_ids": ["5 right"],
    },
}

HIDDEN_DIMS = (64, 128, 256, 384)
LEARNING_RATES = (1e-3, 3e-4, 1e-4)
EPOCHS = (75, 150)
PREDICTION_TARGETS = ("delta", "absolute")
SEEDS = (7, 13, 29)
BATCH_SIZES = (32, 64)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def prepare_dataset(direction: str) -> Path:
    spec = DATASET_KEYS[direction]
    src = ROOT / "datasets" / spec["source"]
    dst = ROOT / "datasets" / spec["target"]
    dst.mkdir(parents=True, exist_ok=True)
    arrays_src = src / "dynamics_arrays.npz"
    arrays_dst = dst / "dynamics_arrays.npz"
    if not arrays_dst.exists():
        try:
            os.link(arrays_src, arrays_dst)
        except OSError:
            shutil.copy2(arrays_src, arrays_dst)
    dataset = read_json(src / "dynamics_dataset.json")
    videos = [str(v) for v in dataset.get("source_videos", [])]
    test_ids = [str(v) for v in spec["test_video_ids"]]
    train_ids = [v for v in videos if v not in set(test_ids)]
    if not train_ids or not test_ids:
        raise ValueError(f"Invalid {direction} train/test split: train={train_ids}, test={test_ids}")
    dataset["dataset_id"] = spec["target"]
    dataset["array_path"] = arrays_dst.as_posix()
    dataset["splits"] = {
        "split_unit": "video",
        "split_method": f"{direction}_train_test_only_no_resting_for_rnn_v2",
        "train_video_ids": train_ids,
        "val_video_ids": [],
        "test_video_ids": test_ids,
    }
    warnings = list(dataset.get("warnings") or [])
    note = "Resting videos excluded from RNN training; validation removed because directional sample count is sparse."
    if note not in warnings:
        warnings.append(note)
    dataset["warnings"] = warnings
    write_json(dst / "dynamics_dataset.json", dataset)
    write_json(dst / "split_manifest.json", dataset["splits"])
    return dst / "dynamics_dataset.json"


def config_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for hidden_dim, lr, epochs, target, seed, batch_size in product(HIDDEN_DIMS, LEARNING_RATES, EPOCHS, PREDICTION_TARGETS, SEEDS, BATCH_SIZES):
        configs.append({
            "hidden_dim": int(hidden_dim),
            "learning_rate": float(lr),
            "epochs": int(epochs),
            "prediction_target": str(target),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "config_id": f"hd{hidden_dim}_lr{slug_float(lr)}_e{epochs}_{target}_s{seed}_b{batch_size}",
        })
    return configs


def metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "decoded_prediction_mse",
        "persistence_mse",
        "improvement_over_persistence_mse",
        "train_decoded_prediction_mse",
        "train_persistence_mse",
        "train_improvement_over_persistence_mse",
        "test_decoded_prediction_mse",
        "test_persistence_mse",
        "test_improvement_over_persistence_mse",
        "test_active_cell_improvement_over_persistence_mse",
        "test_top_activity_improvement_over_persistence_mse",
        "test_high_change_improvement_over_persistence_mse",
        "latent_code_mse",
        "test_latent_code_mse",
    ]
    return {k: metrics.get(k) for k in keys if k in metrics}


def write_summary(out_dir: Path, records: list[dict[str, Any]], *, created_at: str, configs: list[dict[str, Any]], datasets: dict[str, str]) -> None:
    completed = [r for r in records if r.get("status") == "completed"]
    best = sorted(
        completed,
        key=lambda r: float(r.get("metrics", {}).get("test_improvement_over_persistence_mse", -1e9) if r.get("metrics", {}).get("test_improvement_over_persistence_mse") is not None else -1e9),
        reverse=True,
    )[:10]
    payload = {
        "schema_version": 1,
        "created_at": created_at,
        "updated_at": now(),
        "state": "running",
        "pid": os.getpid(),
        "datasets": datasets,
        "search_space": {
            "hidden_dims": list(HIDDEN_DIMS),
            "learning_rates": list(LEARNING_RATES),
            "epochs": list(EPOCHS),
            "prediction_targets": list(PREDICTION_TARGETS),
            "seeds": list(SEEDS),
            "batch_sizes": list(BATCH_SIZES),
            "configs_per_direction": len(configs),
            "ranking_metric": "test_improvement_over_persistence_mse",
        },
        "counts": dict(Counter(str(r.get("status")) for r in records)),
        "records": records,
        "best_by_test_improvement": best,
    }
    write_json(out_dir / "directional_rnn_sweep_summary.json", payload)
    fields = ["index", "direction", "status", "config_id", "hidden_dim", "learning_rate", "epochs", "prediction_target", "seed", "batch_size", "test_decoded_prediction_mse", "test_persistence_mse", "test_improvement_over_persistence_mse", "run_path", "error"]
    lines = ["\t".join(fields)]
    for r in records:
        c = r.get("config", {})
        m = r.get("metrics", {})
        row = {
            "index": r.get("index", ""),
            "direction": r.get("direction", ""),
            "status": r.get("status", ""),
            "config_id": c.get("config_id", ""),
            "hidden_dim": c.get("hidden_dim", ""),
            "learning_rate": c.get("learning_rate", ""),
            "epochs": c.get("epochs", ""),
            "prediction_target": c.get("prediction_target", ""),
            "seed": c.get("seed", ""),
            "batch_size": c.get("batch_size", ""),
            "test_decoded_prediction_mse": m.get("test_decoded_prediction_mse", ""),
            "test_persistence_mse": m.get("test_persistence_mse", ""),
            "test_improvement_over_persistence_mse": m.get("test_improvement_over_persistence_mse", ""),
            "run_path": r.get("run_path", ""),
            "error": r.get("error", ""),
        }
        lines.append("\t".join(str(row.get(f, "")).replace("\t", " ").replace("\n", " ") for f in fields))
    (out_dir / "directional_rnn_sweep_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", default="left,right")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-limit-hours", type=float, default=48.0)
    ap.add_argument("--max-runs", type=int, default=0, help="0 means no cap beyond time limit")
    ap.add_argument("--out-dir", default=OUT_ROOT.as_posix())
    args = ap.parse_args()

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    for d in directions:
        if d not in DATASET_KEYS:
            raise ValueError(f"Unknown direction {d!r}")
    _torch()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = now()
    configs = config_grid()
    datasets = {d: prepare_dataset(d).as_posix() for d in directions}
    autoencoder_run = read_json(AE_RUN_PATH)
    total = len(configs) * len(directions)
    records: list[dict[str, Any]] = []
    if (out_dir / "directional_rnn_sweep_summary.json").exists():
        prior = read_json(out_dir / "directional_rnn_sweep_summary.json")
        records = list(prior.get("records") or [])
        created_at = str(prior.get("created_at") or created_at)
    completed_ids = {(r.get("direction"), r.get("config", {}).get("config_id")) for r in records if r.get("status") == "completed"}
    start = time.time()
    run_count = 0
    progress_path = out_dir / "directional_rnn_sweep_progress.jsonl"
    write_summary(out_dir, records, created_at=created_at, configs=configs, datasets=datasets)
    for direction in directions:
        dataset = read_json(Path(datasets[direction]))
        window_frames = int(dataset.get("windowing", {}).get("window_frames") or 8)
        for cfg in configs:
            if args.max_runs and run_count >= int(args.max_runs):
                write_summary(out_dir, records, created_at=created_at, configs=configs, datasets=datasets)
                return 0
            elapsed_hours = (time.time() - start) / 3600.0
            if elapsed_hours >= float(args.time_limit_hours):
                write_summary(out_dir, records, created_at=created_at, configs=configs, datasets=datasets)
                return 0
            if (direction, cfg["config_id"]) in completed_ids:
                continue
            run_count += 1
            index = len(records) + 1
            run_dir = out_dir / direction / cfg["config_id"]
            record = {"index": index, "experiment_count": total, "direction": direction, "config": dict(cfg), "run_dir": run_dir.as_posix(), "status": "running", "started_at": now()}
            write_json(out_dir / "sweep_active.json", record)
            print(f"{now()} start {index}/{total} {direction} {cfg['config_id']}", flush=True)
            try:
                run = None
                if (run_dir / "latent_rnn_run.json").exists() and (run_dir / "latent_rnn_metrics.json").exists():
                    run = read_json(run_dir / "latent_rnn_run.json")
                else:
                    run = train_latent_rnn(
                        dataset=dataset,
                        autoencoder_run=autoencoder_run,
                        out_dir=run_dir,
                        window_frames=window_frames,
                        hidden_dim=int(cfg["hidden_dim"]),
                        epochs=int(cfg["epochs"]),
                        batch_size=int(cfg["batch_size"]),
                        learning_rate=float(cfg["learning_rate"]),
                        prediction_target=str(cfg["prediction_target"]),
                        seed=int(cfg["seed"]),
                        device=str(args.device),
                    )
                metrics = read_json(Path(run["metrics_path"]))
                record.update({"status": "completed", "completed_at": now(), "run_path": (run_dir / "latent_rnn_run.json").as_posix(), "metrics_path": str(run.get("metrics_path", "")), "checkpoint_path": str(run.get("checkpoint_path", "")), "metrics": metric_payload(metrics)})
                print(f"{now()} done {direction} {cfg['config_id']} test_improvement={record['metrics'].get('test_improvement_over_persistence_mse')}", flush=True)
            except Exception as exc:
                record.update({"status": "failed", "completed_at": now(), "error": repr(exc)})
                print(f"{now()} failed {direction} {cfg['config_id']}: {exc!r}", flush=True)
            records.append(record)
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            write_summary(out_dir, records, created_at=created_at, configs=configs, datasets=datasets)
    write_summary(out_dir, records, created_at=created_at, configs=configs, datasets=datasets)
    active = {"state": "finished", "completed_at": now(), "pid": os.getpid(), "records": len(records), "experiment_count": total}
    write_json(out_dir / "sweep_active.json", active)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
