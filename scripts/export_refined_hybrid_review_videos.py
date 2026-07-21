#!/usr/bin/env python3
from __future__ import annotations

import argparse, html, importlib.util, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from neurobench.dynamics.train import _prepare_model_array, _torch
from neurobench.workbench.intermediates import write_png_gray8

spec = importlib.util.spec_from_file_location('shared_hybrid_base', SCRIPT_DIR / 'run_shared_directional_hybrid_rnn_sweep.py')
if spec is None or spec.loader is None:
    raise RuntimeError('Cannot import shared hybrid base')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ROOT = base.ROOT
DEPLOY = ROOT / 'deployed_dashboards' / 'results_inspection_v1'
SUMMARY = ROOT / 'shared_directional_hybrid_rnn_refined_optuna_v1/refined_optuna_summary.json'
OUT_NAME = 'refined_hybrid_review_v1'
OUT = DEPLOY / OUT_NAME
MANIFEST_PATH = OUT / 'refined_hybrid_review_manifest.json'
STATUS_PATH = OUT / 'status.json'
REPORT_PATH = DEPLOY / 'refined_hybrid_review.html'
FPS = 12
PANEL_GAP = 8
TMP_ROOT = Path('/tmp/neurobench_refined_hybrid_review')

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def status(**updates: Any) -> None:
    payload = {}
    if STATUS_PATH.exists():
        try: payload = read_json(STATUS_PATH)
        except Exception: payload = {}
    payload.update(updates); payload['updated_at'] = now(); write_json(STATUS_PATH, payload)

def slug(value: str) -> str:
    return '_'.join(''.join(ch if ch.isalnum() else '_' for ch in str(value)).split('_')) or 'item'

def esc(value: Any) -> str:
    return html.escape(str(value))

def fmt(value: Any) -> str:
    try: return f'{float(value):.6g}'
    except Exception: return 'n/a'

def as_gray(frame: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3: arr = arr[0]
    return np.round(np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)

def upscale2(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

def panel_image(window: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    panes = [upscale2(as_gray(window[-1])), upscale2(as_gray(target)), upscale2(as_gray(pred)), upscale2(as_gray(np.abs(np.asarray(pred[0]) - np.asarray(target[0])), 0.0, 0.25))]
    height = max(p.shape[0] for p in panes)
    width = sum(p.shape[1] for p in panes) + PANEL_GAP * (len(panes)-1)
    out = np.full((height, width), 18, dtype=np.uint8)
    x = 0
    for pane in panes:
        out[:, x:x+pane.shape[1]] = pane
        x += pane.shape[1] + PANEL_GAP
    return out

def encode_mp4(frame_dir: Path, mp4_path: Path) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp4_path.with_suffix('.tmp.mp4')
    if tmp.exists(): tmp.unlink()
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(FPS),'-i',str(frame_dir/'frame_%05d.png'),'-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)], check=True)
    os.replace(tmp, mp4_path)

def best_records(summary: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [r for r in summary.get('best_by_objective', []) if r.get('status') == 'completed']
    if not rows:
        rows = [r for r in summary.get('records', []) if r.get('status') == 'completed']
        rows.sort(key=lambda r: float(r.get('objective_value') or -1e18), reverse=True)
    return list(rows[:limit])

def predict_record(record: Mapping[str, Any], cache_path: Path, device: str, batch_size: int) -> np.ndarray:
    torch = _torch(); ckpt = torch.load(record['checkpoint_path'], map_location=device); cfg = ckpt['config']; latent_dim = int(ckpt['latent_dim'])
    model = base.DirectionalHybridGRU(latent_dim=latent_dim, hidden_dim=int(cfg['hidden_dim']), num_layers=int(cfg['num_layers']), direction_emb_dim=int(cfg['direction_emb_dim']), dropout=float(cfg['dropout']), mode=str(cfg['mode']), gate_kind=str(cfg['gate_kind'])).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval(); preds = []
    with np.load(cache_path, allow_pickle=False) as arrays:
        z_windows = arrays['z_windows'].astype(np.float32); direction_ids = arrays['direction_ids'].astype(np.int64)
    with torch.no_grad():
        for start in range(0, z_windows.shape[0], batch_size):
            zw = torch.as_tensor(z_windows[start:start+batch_size], dtype=torch.float32, device=device)
            dirs = torch.as_tensor(direction_ids[start:start+batch_size], dtype=torch.long, device=device)
            preds.append(model(zw, dirs)['pred'].detach().cpu().numpy().astype(np.float32))
    pred_z = np.concatenate(preds, axis=0) if preds else np.zeros((0, latent_dim), dtype=np.float32)
    run = read_json(Path(record['run_path']))
    return _prepare_model_array(base.decode_predictions({'checkpoint_path': run['source_autoencoder_run']}, pred_z, batch_size=max(16, batch_size), device=device))

def split_by_video(combined: Mapping[str, Any]) -> dict[str, str]:
    splits = combined['dataset']['splits']
    return {vid: split for split, vids in (('train', splits.get('train_video_ids') or []), ('test', splits.get('test_video_ids') or []), ('val', splits.get('val_video_ids') or [])) for vid in vids}

def write_static_html(manifest: Mapping[str, Any]) -> None:
    style = ':root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#101316;color:#e9edf0}body{margin:0;background:#101316}header{padding:18px 22px;border-bottom:1px solid #30363d}main{padding:18px 22px 32px}h1{margin:0 0 6px;font-size:22px}h2{margin:24px 0 10px;font-size:17px}.status{color:#aab3bc;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px}.card{border:1px solid #30363d;border-radius:6px;background:#181d22;padding:10px}.meta{color:#aab3bc;font-size:12px;line-height:1.4;margin:6px 0 8px}video{width:100%;background:#050607;border:1px solid #30363d}code{color:#c6dcff;font-size:12px}a{color:#9dccff}'
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Refined Hybrid Review</title>', f'<style>{style}</style></head><body>', '<header><h1>Refined Hybrid Review</h1><div class="status">Panes: newest input frame, target, prediction, absolute error. Use browser video controls to scrub frames. <a href="shared_hybrid_refined_dashboard/shared_hybrid_optuna_dashboard.html">Summary dashboard</a></div></header><main>']
    for model in manifest.get('models', []):
        parts.append(f'<h2>Rank {model.get("rank")} - trial {model.get("trial_number")} - test improvement {fmt(model.get("test_improvement_over_persistence_mse"))} - high-change {fmt(model.get("test_high_change_improvement_over_persistence_mse"))}</h2>')
        parts.append(f'<div class="status"><code>{esc(model.get("config_id"))}</code></div><div class="grid">')
        for video in model.get('videos', []):
            parts.append('<div class="card">')
            parts.append(f'<div><strong>{esc(video.get("split"))} - {esc(video.get("video_id"))}</strong></div>')
            parts.append(f'<div class="meta">frames {video.get("frame_count")} - MSE {fmt(video.get("decoded_prediction_mse"))} - persistence {fmt(video.get("persistence_mse"))}</div>')
            parts.append(f'<video controls playsinline preload="metadata" poster="{esc(video.get("poster_path"))}?v={esc(manifest.get("created_at"))}" src="{esc(video.get("video_path"))}?v={esc(manifest.get("created_at"))}"></video>')
            parts.append('</div>')
        parts.append('</div>')
    parts.append('</main></body></html>')
    REPORT_PATH.write_text('\n'.join(parts), encoding='utf-8')

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--summary', type=Path, default=SUMMARY); ap.add_argument('--top-n', type=int, default=5); ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=64); args = ap.parse_args()
    torch = _torch(); device = str(args.device)
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
    summary = read_json(args.summary); records = best_records(summary, int(args.top_n)); combined = base.load_combined_arrays()
    ae_run = read_json(ROOT / 'models/autoencoder128_s1_ld128_bc16_e80_lr0p0010_v1/autoencoder_run.json')
    cache_path = base.encode_latent_cache(combined=combined, autoencoder_run=ae_run, out_dir=ROOT/'shared_directional_hybrid_rnn_refined_optuna_v1', batch_size=int(args.batch_size), device=device)
    windows = np.asarray(combined['windows'], dtype=np.float32); targets = np.asarray(combined['targets'], dtype=np.float32); video_ids = np.asarray(combined['video_ids']).astype(str)
    split_lookup = split_by_video(combined); video_order = list(combined['dataset']['splits'].get('train_video_ids') or []) + list(combined['dataset']['splits'].get('test_video_ids') or [])
    OUT.mkdir(parents=True, exist_ok=True); manifest = {'schema_version':1,'state':'running','created_at':now(),'fps':FPS,'summary_path':args.summary.as_posix(),'models':[]}; write_json(MANIFEST_PATH, manifest); status(state='running', device=device, pid=os.getpid(), top_n=int(args.top_n))
    for rank, record in enumerate(records, start=1):
        metrics = record.get('metrics') or {}; cfg = record.get('config') or {}; model_slug = f"rank{rank:02d}_trial{int(record.get('trial_number') or 0):04d}"; print(f"{now()} predict {model_slug}", flush=True); preds = predict_record(record, cache_path, device, int(args.batch_size))
        model_item = {'rank':rank,'trial_number':record.get('trial_number'),'config_id':record.get('config_id'),'run_path':record.get('run_path'),'mode':cfg.get('mode'),'direction_emb_dim':cfg.get('direction_emb_dim'),'hidden_dim':cfg.get('hidden_dim'),'num_layers':cfg.get('num_layers'),'learning_rate':cfg.get('learning_rate'),'epochs':cfg.get('epochs'),'test_improvement_over_persistence_mse':metrics.get('test_improvement_over_persistence_mse'),'test_high_change_improvement_over_persistence_mse':metrics.get('test_high_change_improvement_over_persistence_mse'),'videos':[]}
        for vid in video_order:
            idxs = np.nonzero(video_ids == str(vid))[0]
            if not idxs.size: continue
            vslug = slug(str(vid)); frame_dir = TMP_ROOT / model_slug / vslug
            if frame_dir.exists(): shutil.rmtree(frame_dir)
            frame_dir.mkdir(parents=True, exist_ok=True); rel_video = Path(OUT_NAME)/'videos'/model_slug/f'{vslug}.mp4'; rel_poster = Path(OUT_NAME)/'posters'/model_slug/f'{vslug}.png'; mp4_path = DEPLOY/rel_video; poster_path = DEPLOY/rel_poster; started = time.time()
            for out_i, src_i in enumerate(idxs):
                panel = panel_image(windows[src_i], targets[src_i], preds[src_i]); write_png_gray8(frame_dir/f'frame_{out_i:05d}.png', int(panel.shape[1]), int(panel.shape[0]), panel.tobytes())
                if out_i == 0 or out_i + 1 == idxs.size or (out_i + 1) % 512 == 0:
                    status(state='running', current_model=model_slug, current_video=str(vid), current_frame=out_i+1, current_video_frames=int(idxs.size)); print(f"{now()} {model_slug} {vid}: {out_i+1}/{idxs.size}", flush=True)
            poster_path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(frame_dir/'frame_00000.png', poster_path); encode_mp4(frame_dir, mp4_path); shutil.rmtree(frame_dir)
            diff = preds[idxs] - targets[idxs]; persistence_diff = windows[idxs, -1] - targets[idxs]
            model_item['videos'].append({'video_id':str(vid),'split':split_lookup.get(str(vid),'unknown'),'frame_count':int(idxs.size),'fps':FPS,'duration_seconds':float(idxs.size)/float(FPS),'video_path':rel_video.as_posix(),'poster_path':rel_poster.as_posix(),'decoded_prediction_mse':float(np.mean(diff*diff)),'persistence_mse':float(np.mean(persistence_diff*persistence_diff)),'elapsed_seconds':time.time()-started})
        manifest['models'].append(model_item); manifest['video_count'] = sum(len(m.get('videos', [])) for m in manifest['models']); write_json(MANIFEST_PATH, manifest); write_static_html(manifest)
    manifest['state'] = 'complete'; manifest['completed_at'] = now(); manifest['video_count'] = sum(len(m.get('videos', [])) for m in manifest['models']); write_json(MANIFEST_PATH, manifest); write_static_html(manifest); status(state='complete', completed_at=now(), model_count=len(manifest['models']), video_count=manifest['video_count']); print(f"{now()} complete: {MANIFEST_PATH}", flush=True); return 0

if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        status(state='failed', error=repr(exc)); print(f"{now()} failed: {exc!r}", file=sys.stderr, flush=True); raise
