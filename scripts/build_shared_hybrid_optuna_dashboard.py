#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def num(v: Any) -> float | None:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None


def fmt(v: Any) -> str:
    n = num(v)
    return 'n/a' if n is None else f'{n:.6g}'


def esc(v: Any) -> str:
    return html.escape(str(v))


def load_records(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in summary.get('records', []) if isinstance(r, Mapping) and r.get('status') == 'completed']
    return sorted(rows, key=lambda r: num(r.get('objective_value')) if num(r.get('objective_value')) is not None else float('-inf'), reverse=True)


def compact_record(r: Mapping[str, Any]) -> dict[str, Any]:
    c = r.get('config') or {}
    m = r.get('metrics') or {}
    return {
        'trial_number': r.get('trial_number'),
        'objective_value': r.get('objective_value'),
        'latent_dim': r.get('latent_dim'),
        'mode': c.get('mode'),
        'direction_emb_dim': c.get('direction_emb_dim'),
        'hidden_dim': c.get('hidden_dim'),
        'num_layers': c.get('num_layers'),
        'learning_rate': c.get('learning_rate'),
        'weight_decay': c.get('weight_decay'),
        'dropout': c.get('dropout'),
        'epochs': c.get('epochs'),
        'batch_size': c.get('batch_size'),
        'seed': c.get('seed'),
        'test_improvement_over_persistence_mse': m.get('test_improvement_over_persistence_mse'),
        'test_decoded_prediction_mse': m.get('test_decoded_prediction_mse'),
        'test_persistence_mse': m.get('test_persistence_mse'),
        'test_high_change_improvement_over_persistence_mse': m.get('test_high_change_improvement_over_persistence_mse'),
        'test_active_cell_improvement_over_persistence_mse': m.get('test_active_cell_improvement_over_persistence_mse'),
        'test_top_activity_improvement_over_persistence_mse': m.get('test_top_activity_improvement_over_persistence_mse'),
        'abs_gate_mean_eval': m.get('abs_gate_mean_eval'),
        'accel_gate_mean_eval': m.get('accel_gate_mean_eval'),
        'config_id': r.get('config_id'),
        'run_path': r.get('run_path'),
        'metrics_path': r.get('metrics_path'),
        'checkpoint_path': r.get('checkpoint_path'),
    }


def table_rows(records: list[Mapping[str, Any]], limit: int = 30) -> str:
    rows = []
    for rank, r in enumerate(records[:limit], start=1):
        c = r.get('config') or {}
        m = r.get('metrics') or {}
        cells = [
            rank, r.get('trial_number'), r.get('latent_dim'), c.get('mode'), c.get('direction_emb_dim'),
            c.get('hidden_dim'), c.get('num_layers'), fmt(c.get('learning_rate')), c.get('epochs'),
            fmt(r.get('objective_value')), fmt(m.get('test_improvement_over_persistence_mse')),
            fmt(m.get('test_high_change_improvement_over_persistence_mse')),
            f"<code>{esc(r.get('config_id'))}</code>", f"<code>{esc(r.get('run_path'))}</code>",
        ]
        rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    return '\n'.join(rows)


def group_summary(records: list[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        c = r.get('config') or {}
        key = str(r.get(field) if field == 'latent_dim' else c.get(field))
        groups.setdefault(key, []).append(r)
    out = []
    for key, rows in groups.items():
        best = max(rows, key=lambda r: num(r.get('objective_value')) if num(r.get('objective_value')) is not None else float('-inf'))
        vals = [num((r.get('metrics') or {}).get('test_improvement_over_persistence_mse')) for r in rows]
        vals = [v for v in vals if v is not None]
        out.append({
            'group': key,
            'count': len(rows),
            'best_objective': best.get('objective_value'),
            'best_test_improvement': (best.get('metrics') or {}).get('test_improvement_over_persistence_mse'),
            'mean_test_improvement': sum(vals) / len(vals) if vals else None,
        })
    return sorted(out, key=lambda x: num(x.get('best_objective')) or float('-inf'), reverse=True)


def group_table(groups: list[Mapping[str, Any]]) -> str:
    return '\n'.join(
        '<tr>' + ''.join([
            f'<td>{esc(g.get("group"))}</td>',
            f'<td>{g.get("count")}</td>',
            f'<td>{fmt(g.get("best_objective"))}</td>',
            f'<td>{fmt(g.get("best_test_improvement"))}</td>',
            f'<td>{fmt(g.get("mean_test_improvement"))}</td>',
        ]) + '</tr>'
        for g in groups
    )


def write_dashboard(summary_path: Path, out_dir: Path, title: str) -> None:
    summary = read_json(summary_path)
    records = load_records(summary)
    compact = [compact_record(r) for r in records]
    groups = {
        'mode': group_summary(records, 'mode'),
        'direction_emb_dim': group_summary(records, 'direction_emb_dim'),
        'hidden_dim': group_summary(records, 'hidden_dim'),
        'num_layers': group_summary(records, 'num_layers'),
        'latent_dim': group_summary(records, 'latent_dim'),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1,
        'created_at': now(),
        'title': title,
        'summary_path': summary_path.as_posix(),
        'search_config': summary.get('search_config'),
        'counts': summary.get('counts'),
        'records': compact,
        'best_by_objective': compact[:30],
        'groups': groups,
    }
    (out_dir / 'shared_hybrid_optuna_dashboard_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    best = compact[0] if compact else {}
    style = ":root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#101316;color:#e9edf0}body{margin:0;background:#101316}header{padding:18px 22px;border-bottom:1px solid #30363d}main{padding:18px 22px 32px}h1{margin:0 0 6px;font-size:22px}h2{margin:24px 0 10px;font-size:17px}.status{color:#aab3bc;font-size:13px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:16px 0}.metric{background:#181d22;border:1px solid #30363d;border-radius:6px;padding:10px}.metric span{display:block;color:#9ba6b0;font-size:12px;margin-bottom:4px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #2b3239;padding:7px 8px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#151a1f;color:#b8c3cc}code{font-size:12px;color:#c6dcff}.scroll{overflow:auto;border:1px solid #30363d;border-radius:6px}a{color:#9dccff}"
    html_text = "\n".join([
        '<!doctype html>',
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{esc(title)}</title><style>{style}</style></head>',
        '<body>',
        f'<header><h1>{esc(title)}</h1><div class="status">Generated {esc(manifest["created_at"])} - source <code>{esc(summary_path)}</code></div></header>',
        '<main>',
        '<section class="metrics">',
        f'<div class="metric"><span>Completed</span>{esc((summary.get("counts") or {}).get("completed", len(records)))}</div>',
        f'<div class="metric"><span>Best objective</span>{fmt(best.get("objective_value"))}</div>',
        f'<div class="metric"><span>Best test improvement</span>{fmt(best.get("test_improvement_over_persistence_mse"))}</div>',
        f'<div class="metric"><span>Best high-change improvement</span>{fmt(best.get("test_high_change_improvement_over_persistence_mse"))}</div>',
        '</section>',
        '<h2>Best Runs</h2><div class="scroll"><table><thead><tr><th>Rank</th><th>Trial</th><th>Latent</th><th>Mode</th><th>Token</th><th>Hidden</th><th>Layers</th><th>LR</th><th>Epochs</th><th>Objective</th><th>Test Improvement</th><th>High-Change Improvement</th><th>Config</th><th>Run</th></tr></thead><tbody>',
        table_rows(records, 40),
        '</tbody></table></div>',
        '<h2>By Mode</h2><div class="scroll"><table><thead><tr><th>Group</th><th>Count</th><th>Best Objective</th><th>Best Test Improvement</th><th>Mean Test Improvement</th></tr></thead><tbody>',
        group_table(groups['mode']),
        '</tbody></table></div>',
        '<h2>By Direction Token</h2><div class="scroll"><table><thead><tr><th>Group</th><th>Count</th><th>Best Objective</th><th>Best Test Improvement</th><th>Mean Test Improvement</th></tr></thead><tbody>',
        group_table(groups['direction_emb_dim']),
        '</tbody></table></div>',
        '<h2>By Hidden Dim</h2><div class="scroll"><table><thead><tr><th>Group</th><th>Count</th><th>Best Objective</th><th>Best Test Improvement</th><th>Mean Test Improvement</th></tr></thead><tbody>',
        group_table(groups['hidden_dim']),
        '</tbody></table></div>',
        '</main></body></html>',
    ])
    (out_dir / 'shared_hybrid_optuna_dashboard.html').write_text(html_text, encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--title', default='Shared Hybrid Optuna Dashboard')
    args = ap.parse_args()
    write_dashboard(args.summary, args.out_dir, args.title)
    print(args.out_dir / 'shared_hybrid_optuna_dashboard.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
