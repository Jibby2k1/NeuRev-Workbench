#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurobench.dynamics.train import train_autoencoder
from neurobench.manifests import load_json

BASE_PATH = SCRIPT_DIR / 'run_shared_directional_hybrid_rnn_sweep.py'
spec = importlib.util.spec_from_file_location('shared_hybrid_base', BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Cannot import {BASE_PATH}')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ROOT = base.ROOT
DEFAULT_OUT = ROOT / 'shared_directional_hybrid_rnn_optuna_v1'
AE128_RUN = ROOT / 'models/autoencoder128_s1_ld128_bc16_e80_lr0p0010_v1/autoencoder_run.json'
AE128_DIR = AE128_RUN.parent
AE64_RUN = ROOT / 'models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json'
SOURCE_DATASET = ROOT / 'datasets/w8_s1_h2/dynamics_dataset.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    base.write_json(path, payload)


def train_ae128_if_needed(*, device: str, force: bool = False) -> Path:
    if AE128_RUN.exists() and not force:
        return AE128_RUN
    AE128_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_json(SOURCE_DATASET)
    print(f'{now()} train-autoencoder ld128 start {AE128_DIR}', flush=True)
    run = train_autoencoder(
        dataset=dataset,
        out_dir=AE128_DIR,
        latent_dim=128,
        base_channels=16,
        epochs=80,
        batch_size=64,
        learning_rate=0.001,
        seed=7,
        device=device,
    )
    print(f'{now()} train-autoencoder ld128 done {run.get("metrics_path")}', flush=True)
    return AE128_RUN


def suggest_config(trial) -> dict[str, Any]:
    mode = trial.suggest_categorical('mode', ['hybrid_taylor_gated', 'residual', 'absolute'])
    hidden_dim = trial.suggest_categorical('hidden_dim', [48, 64, 96, 128, 192, 256, 384])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    direction_emb_dim = trial.suggest_categorical('direction_emb_dim', [0, 2, 4, 8, 16, 32])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 96])
    epochs = trial.suggest_categorical('epochs', [60, 90, 120, 180, 240])
    config = {
        'mode': mode,
        'direction_emb_dim': int(direction_emb_dim),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'dropout': float(trial.suggest_float('dropout', 0.0, 0.25)),
        'gate_kind': trial.suggest_categorical('gate_kind', ['scalar', 'vector']),
        'learning_rate': float(trial.suggest_float('learning_rate', 2e-5, 2e-3, log=True)),
        'weight_decay': float(trial.suggest_float('weight_decay', 1e-8, 3e-4, log=True)),
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'seed': int(trial.suggest_categorical('seed', [7, 13, 29, 43, 101])),
        'grad_clip': float(trial.suggest_categorical('grad_clip', [0.25, 0.5, 1.0, 2.0, 5.0])),
        'aux_abs_weight': 0.0,
        'aux_delta_weight': 0.0,
        'aux_accel_weight': 0.0,
        'optuna_trial_number': int(trial.number),
    }
    if mode == 'hybrid_taylor_gated':
        config['aux_abs_weight'] = float(trial.suggest_categorical('aux_abs_weight', [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]))
        config['aux_delta_weight'] = float(trial.suggest_categorical('aux_delta_weight', [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]))
        config['aux_accel_weight'] = float(trial.suggest_categorical('aux_accel_weight', [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]))
    elif mode == 'residual':
        config['aux_delta_weight'] = float(trial.suggest_categorical('aux_delta_weight', [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]))
    return config


def metric_keep(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        'test_improvement_over_persistence_mse',
        'test_decoded_prediction_mse',
        'test_persistence_mse',
        'test_active_cell_improvement_over_persistence_mse',
        'test_top_activity_improvement_over_persistence_mse',
        'test_high_change_improvement_over_persistence_mse',
        'abs_gate_mean_eval',
        'accel_gate_mean_eval',
        'improvement_over_persistence_mse',
    ]
    out = {k: metrics.get(k) for k in keys if k in metrics}
    per_direction = metrics.get('per_direction')
    if isinstance(per_direction, Mapping):
        out['per_direction'] = per_direction
    return out


def objective_value(metrics: Mapping[str, Any], objective: str) -> float:
    if objective == 'test_high_change_improvement_over_persistence_mse':
        v = metrics.get('test_high_change_improvement_over_persistence_mse')
    elif objective == 'balanced_global_high_change':
        global_v = metrics.get('test_improvement_over_persistence_mse')
        high_v = metrics.get('test_high_change_improvement_over_persistence_mse')
        if global_v is None or high_v is None:
            return float('-inf')
        return float(global_v) + 0.05 * float(high_v)
    else:
        v = metrics.get('test_improvement_over_persistence_mse')
    return float(v) if v is not None else float('-inf')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT.as_posix())
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-trials', type=int, default=160)
    ap.add_argument('--time-limit-hours', type=float, default=24.0)
    ap.add_argument('--seed', type=int, default=20260620)
    ap.add_argument('--study-name', default='shared_directional_hybrid_rnn_broader_v1')
    ap.add_argument('--storage', default=None, help='Optuna storage URL. Defaults to sqlite in out-dir.')
    ap.add_argument('--objective', choices=['test_improvement_over_persistence_mse', 'test_high_change_improvement_over_persistence_mse', 'balanced_global_high_change'], default='balanced_global_high_change')
    ap.add_argument('--include-ae128', action='store_true')
    ap.add_argument('--train-ae128-if-missing', action='store_true')
    ap.add_argument('--force-retrain-ae128', action='store_true')
    ap.add_argument('--encode-batch-size', type=int, default=64)
    args = ap.parse_args()

    import optuna

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f'sqlite:///{(out_dir / "optuna.db").as_posix()}'
    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True, group=True)
    study = optuna.create_study(direction='maximize', sampler=sampler, storage=storage, study_name=args.study_name, load_if_exists=True)

    ae_paths = [AE64_RUN]
    if args.include_ae128:
        if args.train_ae128_if_missing or args.force_retrain_ae128:
            train_ae128_if_needed(device=str(args.device), force=bool(args.force_retrain_ae128))
        if AE128_RUN.exists():
            ae_paths.append(AE128_RUN)
        else:
            print(f'{now()} warning missing AE128 run; continuing with latent 64 only', flush=True)
    ae_runs = [read_json(p) for p in ae_paths if p.exists()]
    if not ae_runs:
        raise ValueError('No autoencoder runs available.')

    combined = base.load_combined_arrays()
    write_json(out_dir / 'combined_dataset_manifest.json', combined['dataset'])
    caches = []
    for ae_run in ae_runs:
        cache_path = base.encode_latent_cache(combined=combined, autoencoder_run=ae_run, out_dir=out_dir, batch_size=int(args.encode_batch_size), device=str(args.device))
        with __import__('numpy').load(cache_path, allow_pickle=False) as arrays:
            latent_dim = int(arrays['z_windows'].shape[-1])
        caches.append((ae_run, cache_path, latent_dim))

    summary_path = out_dir / 'shared_directional_hybrid_optuna_summary.json'
    records: list[dict[str, Any]] = []
    created_at = now()
    if summary_path.exists():
        prior = read_json(summary_path)
        records = list(prior.get('records') or [])
        created_at = str(prior.get('created_at') or created_at)
    completed_trial_numbers = {int(r.get('trial_number')) for r in records if r.get('status') == 'completed' and r.get('trial_number') is not None}
    search_config = {
        'backend': 'optuna_tpe_sequential',
        'study_name': args.study_name,
        'storage': storage,
        'objective': args.objective,
        'requested_trials': int(args.n_trials),
        'autoencoder_runs': [str(p) for p in ae_paths if p.exists()],
        'candidate_latent_dims': [c[2] for c in caches],
        'resting_policy': 'excluded from RNN train/test; allowed only in upstream autoencoder training',
    }

    def write_summary(state: str = 'running') -> None:
        completed = [r for r in records if r.get('status') == 'completed']
        best = sorted(completed, key=lambda r: float((r.get('objective_value') if r.get('objective_value') is not None else -1e18)), reverse=True)[:25]
        payload = {
            'schema_version': 1,
            'created_at': created_at,
            'updated_at': now(),
            'state': state,
            'search_config': search_config,
            'counts': dict(__import__('collections').Counter(str(r.get('status')) for r in records)),
            'records': records,
            'best_by_objective': best,
        }
        write_json(summary_path, payload)
        fields = ['index','trial_number','status','objective_value','latent_dim','mode','direction_emb_dim','hidden_dim','num_layers','learning_rate','epochs','seed','test_improvement_over_persistence_mse','test_high_change_improvement_over_persistence_mse','config_id','run_path','error']
        lines = ['\t'.join(fields)]
        for r in records:
            c = r.get('config') or {}; m = r.get('metrics') or {}
            row = {
                'index': r.get('index',''), 'trial_number': r.get('trial_number',''), 'status': r.get('status',''), 'objective_value': r.get('objective_value',''),
                'latent_dim': r.get('latent_dim',''), 'mode': c.get('mode',''), 'direction_emb_dim': c.get('direction_emb_dim',''), 'hidden_dim': c.get('hidden_dim',''),
                'num_layers': c.get('num_layers',''), 'learning_rate': c.get('learning_rate',''), 'epochs': c.get('epochs',''), 'seed': c.get('seed',''),
                'test_improvement_over_persistence_mse': m.get('test_improvement_over_persistence_mse',''),
                'test_high_change_improvement_over_persistence_mse': m.get('test_high_change_improvement_over_persistence_mse',''),
                'config_id': r.get('config_id',''), 'run_path': r.get('run_path',''), 'error': r.get('error',''),
            }
            lines.append('\t'.join(str(row.get(f,'')).replace('\t',' ').replace('\n',' ') for f in fields))
        (out_dir / 'shared_directional_hybrid_optuna_summary.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    write_summary('running')
    start = time.time()
    for _ in range(int(args.n_trials)):
        if (time.time() - start) / 3600.0 >= float(args.time_limit_hours):
            break
        trial = study.ask()
        if int(trial.number) in completed_trial_numbers:
            continue
        ae_index = trial.suggest_int('autoencoder_index', 0, len(caches) - 1)
        ae_run, cache_path, latent_dim = caches[int(ae_index)]
        config = suggest_config(trial)
        cid = f'trial{trial.number:04d}__ld{latent_dim}__' + base.config_id(config, ae_run)
        run_dir = out_dir / 'runs' / cid
        record = {'index': len(records) + 1, 'trial_number': int(trial.number), 'status': 'running', 'objective': args.objective, 'config_id': cid, 'config': dict(config), 'latent_dim': latent_dim, 'run_dir': run_dir.as_posix(), 'started_at': now()}
        write_json(out_dir / 'sweep_active.json', record)
        print(f'{now()} start trial={trial.number} latent={latent_dim} {cid}', flush=True)
        try:
            run = base.train_one(config, combined=combined, cache_path=cache_path, autoencoder_run=ae_run, run_dir=run_dir, device=str(args.device))
            metrics = read_json(run['metrics_path'])
            value = objective_value(metrics, str(args.objective))
            study.tell(trial, value)
            record.update({'status': 'completed', 'completed_at': now(), 'objective_value': value, 'run_path': (run_dir / 'hybrid_rnn_run.json').as_posix(), 'metrics_path': run['metrics_path'], 'checkpoint_path': run['checkpoint_path'], 'metrics': metric_keep(metrics)})
            print(f'{now()} done trial={trial.number} objective={value} test={record["metrics"].get("test_improvement_over_persistence_mse")} high_change={record["metrics"].get("test_high_change_improvement_over_persistence_mse")}', flush=True)
        except Exception as exc:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            record.update({'status': 'failed', 'completed_at': now(), 'error': repr(exc)})
            print(f'{now()} failed trial={trial.number}: {exc!r}', flush=True)
        records.append(record)
        with (out_dir / 'shared_directional_hybrid_optuna_progress.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')
        write_summary('running')
    active = {'state': 'finished', 'completed_at': now(), 'records': len(records), 'requested_trials': int(args.n_trials), 'pid': os.getpid()}
    write_json(out_dir / 'sweep_active.json', active)
    write_summary('finished')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
