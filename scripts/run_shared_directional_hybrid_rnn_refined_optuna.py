#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SWEEP = SCRIPT_DIR / 'run_shared_directional_hybrid_rnn_sweep.py'
BASE_OPTUNA = SCRIPT_DIR / 'run_shared_directional_hybrid_rnn_optuna.py'
for module_name, path in [('shared_hybrid_base', BASE_SWEEP), ('shared_hybrid_optuna_base', BASE_OPTUNA)]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals()[module_name] = mod
base = shared_hybrid_base
optbase = shared_hybrid_optuna_base

ROOT = base.ROOT
DEFAULT_OUT = ROOT / 'shared_directional_hybrid_rnn_refined_optuna_v1'
AE128_RUN = optbase.AE128_RUN


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    base.write_json(path, payload)


def suggest_refined_config(trial) -> dict[str, Any]:
    # Empirical neighborhood from broad run: latent 128, residual, hidden 96, low LR, batch 96.
    mode = trial.suggest_categorical('mode', ['residual', 'residual', 'hybrid_taylor_gated'])
    hidden_dim = trial.suggest_categorical('hidden_dim', [80, 96, 112, 128])
    direction_emb_dim = trial.suggest_categorical('direction_emb_dim', [0, 0, 2, 4, 8])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    batch_size = trial.suggest_categorical('batch_size', [64, 96, 128])
    epochs = trial.suggest_categorical('epochs', [90, 120, 180, 240, 320])
    config = {
        'mode': mode,
        'direction_emb_dim': int(direction_emb_dim),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'dropout': float(trial.suggest_float('dropout', 0.02, 0.24)),
        'gate_kind': trial.suggest_categorical('gate_kind', ['scalar', 'vector']),
        'learning_rate': float(trial.suggest_float('learning_rate', 1.2e-5, 8.0e-5, log=True)),
        'weight_decay': float(trial.suggest_float('weight_decay', 1e-8, 2e-5, log=True)),
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'seed': int(trial.suggest_categorical('seed', [7, 13, 43, 101, 211])),
        'grad_clip': float(trial.suggest_categorical('grad_clip', [0.5, 1.0, 2.0, 5.0])),
        'aux_abs_weight': 0.0,
        'aux_delta_weight': 0.0,
        'aux_accel_weight': 0.0,
        'optuna_trial_number': int(trial.number),
    }
    if mode == 'residual':
        config['aux_delta_weight'] = float(trial.suggest_categorical('aux_delta_weight', [0.0, 0.01, 0.02, 0.05, 0.1]))
    else:
        config['aux_abs_weight'] = float(trial.suggest_categorical('aux_abs_weight', [0.0, 0.01, 0.02, 0.05, 0.1]))
        config['aux_delta_weight'] = float(trial.suggest_categorical('aux_delta_weight', [0.0, 0.01, 0.02, 0.05, 0.1]))
        config['aux_accel_weight'] = float(trial.suggest_categorical('aux_accel_weight', [0.0, 0.005, 0.01, 0.02, 0.05]))
    return config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT.as_posix())
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-trials', type=int, default=120)
    ap.add_argument('--time-limit-hours', type=float, default=36.0)
    ap.add_argument('--seed', type=int, default=20260621)
    ap.add_argument('--study-name', default='shared_directional_hybrid_rnn_refined_v1')
    ap.add_argument('--storage', default=None)
    ap.add_argument('--objective', choices=['test_improvement_over_persistence_mse', 'test_high_change_improvement_over_persistence_mse', 'balanced_global_high_change'], default='balanced_global_high_change')
    ap.add_argument('--encode-batch-size', type=int, default=64)
    args = ap.parse_args()

    import numpy as np
    import optuna

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not AE128_RUN.exists():
        raise FileNotFoundError(f'Missing required 128-latent autoencoder: {AE128_RUN}')
    storage = args.storage or f'sqlite:///{(out_dir / "optuna.db").as_posix()}'
    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True, group=True, n_startup_trials=12)
    study = optuna.create_study(direction='maximize', sampler=sampler, storage=storage, study_name=args.study_name, load_if_exists=True)

    combined = base.load_combined_arrays()
    write_json(out_dir / 'combined_dataset_manifest.json', combined['dataset'])
    ae_run = read_json(AE128_RUN)
    cache_path = base.encode_latent_cache(combined=combined, autoencoder_run=ae_run, out_dir=out_dir, batch_size=int(args.encode_batch_size), device=str(args.device))
    with np.load(cache_path, allow_pickle=False) as arrays:
        latent_dim = int(arrays['z_windows'].shape[-1])
    if latent_dim != 128:
        raise ValueError(f'Expected latent_dim=128, got {latent_dim}')

    summary_path = out_dir / 'refined_optuna_summary.json'
    records: list[dict[str, Any]] = []
    created_at = now()
    if summary_path.exists():
        prior = read_json(summary_path)
        records = list(prior.get('records') or [])
        created_at = str(prior.get('created_at') or created_at)
    completed_trial_numbers = {int(r.get('trial_number')) for r in records if r.get('status') == 'completed' and r.get('trial_number') is not None}
    search_config = {
        'backend': 'optuna_tpe_refined_neighborhood',
        'study_name': args.study_name,
        'storage': storage,
        'objective': args.objective,
        'requested_trials': int(args.n_trials),
        'autoencoder_run': str(AE128_RUN),
        'latent_dim': latent_dim,
        'neighborhood_basis': 'broad optuna v1 top trials: residual, latent128, hidden96, low lr, batch96',
        'resting_policy': 'excluded from RNN train/test; allowed only in upstream autoencoder training',
    }

    def objective_value(metrics: Mapping[str, Any]) -> float:
        return optbase.objective_value(metrics, str(args.objective))

    def write_summary(state: str = 'running') -> None:
        completed = [r for r in records if r.get('status') == 'completed']
        best = sorted(completed, key=lambda r: float(r.get('objective_value') if r.get('objective_value') is not None else -1e18), reverse=True)[:30]
        payload = {
            'schema_version': 1,
            'created_at': created_at,
            'updated_at': now(),
            'state': state,
            'search_config': search_config,
            'counts': dict(Counter(str(r.get('status')) for r in records)),
            'records': records,
            'best_by_objective': best,
        }
        write_json(summary_path, payload)
        fields = ['index','trial_number','status','objective_value','mode','direction_emb_dim','hidden_dim','num_layers','learning_rate','weight_decay','dropout','epochs','batch_size','seed','test_improvement_over_persistence_mse','test_high_change_improvement_over_persistence_mse','config_id','run_path','error']
        lines = ['\t'.join(fields)]
        for r in records:
            c = r.get('config') or {}; m = r.get('metrics') or {}
            row = {
                'index': r.get('index',''), 'trial_number': r.get('trial_number',''), 'status': r.get('status',''), 'objective_value': r.get('objective_value',''),
                'mode': c.get('mode',''), 'direction_emb_dim': c.get('direction_emb_dim',''), 'hidden_dim': c.get('hidden_dim',''), 'num_layers': c.get('num_layers',''),
                'learning_rate': c.get('learning_rate',''), 'weight_decay': c.get('weight_decay',''), 'dropout': c.get('dropout',''), 'epochs': c.get('epochs',''), 'batch_size': c.get('batch_size',''), 'seed': c.get('seed',''),
                'test_improvement_over_persistence_mse': m.get('test_improvement_over_persistence_mse',''),
                'test_high_change_improvement_over_persistence_mse': m.get('test_high_change_improvement_over_persistence_mse',''),
                'config_id': r.get('config_id',''), 'run_path': r.get('run_path',''), 'error': r.get('error',''),
            }
            lines.append('\t'.join(str(row.get(f,'')).replace('\t',' ').replace('\n',' ') for f in fields))
        (out_dir / 'refined_optuna_summary.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    write_summary('running')
    start = time.time()
    for _ in range(int(args.n_trials)):
        if (time.time() - start) / 3600.0 >= float(args.time_limit_hours):
            break
        trial = study.ask()
        if int(trial.number) in completed_trial_numbers:
            continue
        config = suggest_refined_config(trial)
        cid = f'refined_trial{trial.number:04d}__ld128__' + base.config_id(config, ae_run)
        run_dir = out_dir / 'runs' / cid
        record = {'index': len(records) + 1, 'trial_number': int(trial.number), 'status': 'running', 'objective': args.objective, 'config_id': cid, 'config': dict(config), 'latent_dim': latent_dim, 'run_dir': run_dir.as_posix(), 'started_at': now()}
        write_json(out_dir / 'sweep_active.json', record)
        print(f'{now()} start trial={trial.number} {cid}', flush=True)
        try:
            run = base.train_one(config, combined=combined, cache_path=cache_path, autoencoder_run=ae_run, run_dir=run_dir, device=str(args.device))
            metrics = read_json(run['metrics_path'])
            value = objective_value(metrics)
            study.tell(trial, value)
            record.update({'status': 'completed', 'completed_at': now(), 'objective_value': value, 'run_path': (run_dir / 'hybrid_rnn_run.json').as_posix(), 'metrics_path': run['metrics_path'], 'checkpoint_path': run['checkpoint_path'], 'metrics': optbase.metric_keep(metrics)})
            print(f'{now()} done trial={trial.number} objective={value} test={record["metrics"].get("test_improvement_over_persistence_mse")} high_change={record["metrics"].get("test_high_change_improvement_over_persistence_mse")}', flush=True)
        except Exception as exc:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            record.update({'status': 'failed', 'completed_at': now(), 'error': repr(exc)})
            print(f'{now()} failed trial={trial.number}: {exc!r}', flush=True)
        records.append(record)
        with (out_dir / 'refined_optuna_progress.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')
        write_summary('running')
    active = {'state': 'finished', 'completed_at': now(), 'records': len(records), 'requested_trials': int(args.n_trials), 'pid': os.getpid()}
    write_json(out_dir / 'sweep_active.json', active)
    write_summary('finished')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
