#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from neurobench.dynamics.train import GridAutoencoder, LatentGRUPredictor, _checkpoint_latent_stats, _prepare_model_array, _torch
from neurobench.workbench.intermediates import write_png_gray8

ROOT = Path("Outputs/GridModel/060126_crop512_grid128_max_v1")
DEPLOY = ROOT / "deployed_dashboards" / "results_inspection_v1"
OUT = DEPLOY / "directional_rnn_review_v1"
MANIFEST_PATH = OUT / "directional_rnn_review_manifest.json"
STATUS_PATH = OUT / "status.json"
REPORT_PATH = DEPLOY / "directional_rnn_review.html"
AE_RUN = ROOT / "models/autoencoder128_s1_ld64_bc16_e60_lr0p0010_v1/autoencoder_run.json"
RUN_ROOT = ROOT / "directional_rnn_v1"
FPS = 12
PANEL_GAP = 8
FRAME_GAP = 4
TMP_ROOT = Path("/tmp/neurobench_directional_rnn_review")

CONFIGS = [
    {
        "direction": "left",
        "dataset_key": "w8_s1_h2_left_only_rnn_v1",
        "run_dir": RUN_ROOT / "left_latent_gru_w8_s1_h2_delta_hd128_lr1em04_e50_s7",
    },
    {
        "direction": "right",
        "dataset_key": "w8_s1_h2_right_only_rnn_v1",
        "run_dir": RUN_ROOT / "right_latent_gru_w8_s1_h2_delta_hd128_lr1em04_e50_s7",
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return "_".join("".join(ch if ch.isalnum() else "_" for ch in str(value)).split("_")) or "item"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def status(**updates: Any) -> None:
    payload: dict[str, Any] = {}
    if STATUS_PATH.exists():
        try:
            payload = read_json(STATUS_PATH)
        except json.JSONDecodeError:
            payload = {}
    payload.update(updates)
    payload["updated_at"] = now()
    write_json(STATUS_PATH, payload)


def as_gray(frame: np.ndarray, *, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return np.round(np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)


def upscale2(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def latest_input_frame(window: np.ndarray) -> np.ndarray:
    return upscale2(as_gray(window[-1]))


def pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    out = np.full((height, frame.shape[1]), 18, dtype=np.uint8)
    y = max(0, (height - frame.shape[0]) // 2)
    out[y:y+frame.shape[0], :] = frame
    return out


def panel_image(window: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    input_panel = latest_input_frame(window)
    target_panel = upscale2(as_gray(target))
    pred_panel = upscale2(as_gray(pred))
    error_panel = upscale2(as_gray(np.abs(np.asarray(pred[0]) - np.asarray(target[0])), lo=0.0, hi=0.25))
    height = max(input_panel.shape[0], target_panel.shape[0], pred_panel.shape[0], error_panel.shape[0])
    panes = [pad_to_height(p, height) for p in (input_panel, target_panel, pred_panel, error_panel)]
    width = sum(p.shape[1] for p in panes) + PANEL_GAP * (len(panes) - 1)
    out = np.full((height, width), 18, dtype=np.uint8)
    x = 0
    for pane in panes:
        out[:, x:x+pane.shape[1]] = pane
        x += pane.shape[1] + PANEL_GAP
    return out


def encode_mp4(frame_dir: Path, mp4_path: Path) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp4_path.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)
    ], check=True)
    os.replace(tmp, mp4_path)


def load_models(config: dict[str, Any], device: str):
    torch = _torch()
    ae_run = read_json(AE_RUN)
    ae_ckpt = torch.load(ae_run["checkpoint_path"], map_location=device)
    latent_dim = int(ae_ckpt["latent_dim"])
    latent_mean_np, latent_std_np = _checkpoint_latent_stats(ae_ckpt, latent_dim)
    ae = GridAutoencoder(
        input_channels=1,
        latent_dim=latent_dim,
        base_channels=int(ae_ckpt.get("base_channels", 16)),
        input_shape=tuple(ae_ckpt.get("input_shape") or (1, 128, 128)),
    ).to(device)
    ae.load_state_dict(ae_ckpt["model_state"])
    ae.eval()
    rnn_run = read_json(Path(config["run_dir"]) / "latent_rnn_run.json")
    rnn_ckpt = torch.load(rnn_run["checkpoint_path"], map_location=device)
    rnn = LatentGRUPredictor(latent_dim=latent_dim, hidden_dim=int(rnn_ckpt["hidden_dim"])).to(device)
    rnn.load_state_dict(rnn_ckpt["model_state"])
    rnn.eval()
    return ae, rnn, rnn_run, latent_mean_np, latent_std_np


def predict_batches(ae, rnn, windows: np.ndarray, *, prediction_target: str, latent_mean_np: np.ndarray, latent_std_np: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    latent_dim = int(latent_mean_np.shape[0])
    latent_mean = torch.as_tensor(latent_mean_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    latent_std = torch.as_tensor(latent_std_np, dtype=torch.float32, device=device).reshape(1, latent_dim)
    preds = []
    with torch.no_grad():
        for start in range(0, int(windows.shape[0]), int(batch_size)):
            batch = torch.as_tensor(windows[start:start+int(batch_size)], dtype=torch.float32, device=device)
            b, w, c, h, ww = batch.shape
            z_raw = ae.encode(batch.reshape(b * w, c, h, ww)).reshape(b, w, latent_dim)
            z = (z_raw - latent_mean.reshape(1, 1, latent_dim)) / latent_std.reshape(1, 1, latent_dim)
            step = rnn(z)
            if prediction_target == "delta":
                pred_z = z[:, -1, :] + step
            else:
                pred_z = step
            pred_raw = pred_z * latent_std + latent_mean
            pred_x = _prepare_model_array(ae.decode(pred_raw).detach().cpu().numpy())
            preds.append(pred_x.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 1, 128, 128), dtype=np.float32)


def write_report_html() -> None:
    html = """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Directional RNN Review</title><style>:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#101316;color:#e9edf0}body{margin:0;background:#101316}header{padding:16px 20px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;gap:16px;align-items:end}h1{margin:0;font-size:20px}main{display:grid;grid-template-columns:320px minmax(0,1fr);min-height:calc(100vh - 70px)}aside{border-right:1px solid #30363d;padding:14px;overflow:auto}section{padding:16px 20px 24px;overflow:auto}label{display:block;color:#aab3bc;font-size:12px;margin:10px 0 5px}select,button{width:100%;background:#1a2026;color:#eef2f5;border:1px solid #3a424b;border-radius:6px;padding:8px 10px}button{text-align:left;cursor:pointer;margin:0 0 8px}button.active{border-color:#56a6ff;background:#172333}.top{display:grid;grid-template-columns:1fr 140px 140px;gap:10px;align-items:end;margin-bottom:12px}.scrubber{display:grid;grid-template-columns:78px 78px 1fr 88px;gap:10px;align-items:center;margin:0 0 12px}.scrubber button{margin:0;text-align:center}input[type=range]{width:100%;padding:0}video{display:block;width:min(100%,1316px);background:#050607;border:1px solid #30363d}.status,.hint{color:#aab3bc;font-size:13px;line-height:1.4}.meta{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin:12px 0}.metric{background:#181d22;border:1px solid #30363d;border-radius:6px;padding:9px 10px}.metric span{display:block;color:#99a3ad;font-size:12px;margin-bottom:4px}code{color:#c6dcff;font-size:12px}a{color:#9dccff}@media(max-width:860px){main,.top,.meta,.scrubber{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid #30363d}}</style></head>
<body><header><div><h1>Directional RNN Review</h1><div class=\"status\" id=\"status\">Loading...</div></div><div class=\"status\"><a href=\"index.html\">Dashboard index</a></div></header><main><aside><label for=\"directionSelect\">Direction</label><select id=\"directionSelect\"></select><label>Video</label><div id=\"videos\"></div></aside><section><div class=\"top\"><div class=\"hint\" id=\"selectionLabel\"></div><div><label for=\"rate\">Playback speed</label><select id=\"rate\"><option value=\"0.25\">0.25x</option><option value=\"0.5\">0.5x</option><option value=\"1\" selected>1x</option><option value=\"2\">2x</option><option value=\"4\">4x</option></select></div><div><label><input id=\"loop\" type=\"checkbox\" checked> Loop</label></div></div><div class=\"scrubber\"><button id=\"prevFrame\">-1 frame</button><button id=\"nextFrame\">+1 frame</button><input id=\"frameSlider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\"><div class=\"status\" id=\"frameLabel\">0 / 0</div></div><video id=\"video\" controls playsinline preload=\"metadata\"></video><div class=\"hint\">Panes left to right: newest input frame, target, prediction, absolute error.</div><div class=\"meta\" id=\"meta\"></div></section></main>
<script>let manifest=null,direction='left',videoIndex=0;const statusEl=document.getElementById('status'),directionSelect=document.getElementById('directionSelect'),videosEl=document.getElementById('videos'),videoEl=document.getElementById('video'),meta=document.getElementById('meta'),selectionLabel=document.getElementById('selectionLabel'),rate=document.getElementById('rate'),loop=document.getElementById('loop'),prevFrame=document.getElementById('prevFrame'),nextFrame=document.getElementById('nextFrame'),frameSlider=document.getElementById('frameSlider'),frameLabel=document.getElementById('frameLabel');const fmt=v=>v===null||v===undefined||Number.isNaN(Number(v))?'n/a':Number(v).toPrecision(6);async function loadJson(path){const r=await fetch(path+'?t='+Date.now());if(!r.ok)throw new Error(path+' '+r.status);return await r.json()}function directionVideos(){return (manifest?.videos||[]).filter(v=>v.direction===direction)}function currentRecord(){return directionVideos()[videoIndex]||null}function fpsValue(){return Number(currentRecord()?.fps||12)||12}function currentFrame(){return Math.max(0,Math.min(Number(frameSlider.max)||0,Math.round(videoEl.currentTime*fpsValue())))}function seekFrame(index){const max=Number(frameSlider.max)||0;const next=Math.max(0,Math.min(max,index));videoEl.pause();videoEl.currentTime=next/fpsValue();frameSlider.value=String(next);frameLabel.textContent=`${next+1} / ${max+1}`}function updateFrameControls(){const v=currentRecord();const max=Math.max(0,(v?.frame_count||1)-1);frameSlider.max=String(max);const frame=currentFrame();frameSlider.value=String(frame);frameLabel.textContent=`${frame+1} / ${max+1}`}async function refresh(){manifest=await loadJson('directional_rnn_review_v1/directional_rnn_review_manifest.json');render()}function render(){if(!manifest){return}statusEl.textContent=`${manifest.state||'unknown'} · ${manifest.videos?.length||0} videos · ${manifest.created_at||''}`;const dirs=[...new Set((manifest.videos||[]).map(v=>v.direction))];directionSelect.innerHTML=dirs.map(d=>`<option value=\"${d}\">${d}</option>`).join('');if(dirs.includes(direction))directionSelect.value=direction;const vids=directionVideos();if(videoIndex>=vids.length)videoIndex=0;videosEl.innerHTML=vids.map((v,i)=>`<button class=\"${i===videoIndex?'active':''}\" data-i=\"${i}\">${v.split} · ${v.video_id}<br><code>${v.run_id}</code><br>${v.frame_count} frames</button>`).join('');videosEl.querySelectorAll('button').forEach(b=>b.onclick=()=>{videoIndex=Number(b.dataset.i);render()});const v=vids[videoIndex];if(!v)return;selectionLabel.textContent=`${v.direction} RNN · ${v.split} · ${v.video_id}`;const cacheKey=encodeURIComponent(manifest.created_at||Date.now());const videoUrl=v.video_path+'?v='+cacheKey;const posterUrl=(v.poster_path||'')+'?v='+cacheKey;if(videoEl.getAttribute('src')!==videoUrl){videoEl.src=videoUrl;videoEl.poster=posterUrl;videoEl.playbackRate=Number(rate.value)||1;videoEl.loop=loop.checked;seekFrame(0)}meta.innerHTML=[['Direction',v.direction],['Split',v.split],['Video',v.video_id],['Input pane','newest frame'],['Frames',v.frame_count],['Duration',`${Math.round(v.duration_seconds)}s @ ${v.fps}fps`],['Model MSE',fmt(v.decoded_prediction_mse)],['Persistence MSE',fmt(v.persistence_mse)]].map(([k,val])=>`<div class=\"metric\"><span>${k}</span>${val}</div>`).join('')}directionSelect.onchange=()=>{direction=directionSelect.value;videoIndex=0;render()};rate.onchange=()=>{videoEl.playbackRate=Number(rate.value)||1};loop.onchange=()=>{videoEl.loop=loop.checked};prevFrame.onclick=()=>seekFrame(currentFrame()-1);nextFrame.onclick=()=>seekFrame(currentFrame()+1);frameSlider.oninput=()=>seekFrame(Number(frameSlider.value)||0);videoEl.ontimeupdate=updateFrameControls;videoEl.onloadedmetadata=updateFrameControls;refresh();</script></body></html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")


def main() -> int:
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(os.environ.get("NEUROBENCH_DIRECTIONAL_RNN_BATCH", "64" if device == "cuda" else "8"))
    OUT.mkdir(parents=True, exist_ok=True)
    write_report_html()
    manifest = {"schema_version": 1, "state": "running", "created_at": now(), "fps": FPS, "videos": []}
    write_json(MANIFEST_PATH, manifest)
    status(state="running", device=device, pid=os.getpid())
    for cfg in CONFIGS:
        dataset = read_json(ROOT / "datasets" / cfg["dataset_key"] / "dynamics_dataset.json")
        ae, rnn, rnn_run, latent_mean_np, latent_std_np = load_models(cfg, device)
        metrics = read_json(Path(rnn_run["metrics_path"]))
        with np.load(dataset["array_path"], allow_pickle=False) as arrays:
            windows = _prepare_model_array(arrays["windows"])
            targets = _prepare_model_array(arrays["targets"])
            video_ids = arrays["window_video_ids"].astype(str)
        preds = predict_batches(ae, rnn, windows, prediction_target=str(rnn_run["prediction_target"]), latent_mean_np=latent_mean_np, latent_std_np=latent_std_np, batch_size=batch_size, device=device)
        splits = dataset["splits"]
        split_by_video = {vid: split for split, vids in (("train", splits["train_video_ids"]), ("val", splits["val_video_ids"]), ("test", splits["test_video_ids"])) for vid in vids}
        for vid in dataset["source_videos"]:
            idxs = np.nonzero(video_ids == vid)[0]
            if not idxs.size:
                continue
            vslug = slug(vid)
            frame_dir = TMP_ROOT / cfg["direction"] / vslug
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)
            rel_video = Path("directional_rnn_review_v1") / "videos" / cfg["direction"] / f"{vslug}.mp4"
            rel_poster = Path("directional_rnn_review_v1") / "posters" / cfg["direction"] / f"{vslug}.png"
            mp4_path = DEPLOY / rel_video
            poster_path = DEPLOY / rel_poster
            started = time.time()
            for out_i, src_i in enumerate(idxs):
                panel = panel_image(windows[src_i], targets[src_i], preds[src_i])
                write_png_gray8(frame_dir / f"frame_{out_i:05d}.png", int(panel.shape[1]), int(panel.shape[0]), panel.tobytes())
                if out_i == 0 or out_i + 1 == idxs.size or (out_i + 1) % 256 == 0:
                    status(state="running", current_direction=cfg["direction"], current_video=vid, current_frame=out_i + 1, current_video_frames=int(idxs.size))
                    print(f"{now()} {cfg['direction']} {vid}: {out_i + 1}/{idxs.size}", flush=True)
            poster_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frame_dir / "frame_00000.png", poster_path)
            encode_mp4(frame_dir, mp4_path)
            shutil.rmtree(frame_dir)
            diff = preds[idxs] - targets[idxs]
            persistence_diff = windows[idxs, -1] - targets[idxs]
            record = {
                "direction": cfg["direction"],
                "dataset_key": cfg["dataset_key"],
                "video_id": vid,
                "split": split_by_video.get(vid, "unknown"),
                "run_id": Path(cfg["run_dir"]).name,
                "run_dir": Path(cfg["run_dir"]).as_posix(),
                "frame_count": int(idxs.size),
                "window_frames": int(windows.shape[1]),
                "fps": FPS,
                "duration_seconds": float(idxs.size) / float(FPS),
                "video_path": rel_video.as_posix(),
                "poster_path": rel_poster.as_posix(),
                "decoded_prediction_mse": float(np.mean(diff * diff)),
                "persistence_mse": float(np.mean(persistence_diff * persistence_diff)),
                "elapsed_seconds": time.time() - started,
            }
            manifest["videos"].append(record)
            write_json(MANIFEST_PATH, manifest)
    manifest["state"] = "complete"
    manifest["completed_at"] = now()
    write_json(MANIFEST_PATH, manifest)
    status(state="complete", completed_at=now(), video_count=len(manifest["videos"]))
    print(f"{now()} complete: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        status(state="failed", error=repr(exc))
        print(f"{now()} failed: {exc!r}", file=sys.stderr, flush=True)
        raise
