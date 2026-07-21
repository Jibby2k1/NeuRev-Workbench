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

from neurobench.dynamics.concept_tests import _build_spatial_pixel_model, _torch
from neurobench.workbench.intermediates import write_png_gray8

ROOT = Path("Outputs/GridModel/060126_crop512_grid128_max_v1")
DEPLOY = ROOT / "deployed_dashboards" / "results_inspection_v1"
EXPORT_NAME = os.environ.get("NEUROBENCH_FULL_VIDEO_EXPORT_NAME", "full_video_rnn_v1")
REPORT_NAME = os.environ.get("NEUROBENCH_FULL_VIDEO_REPORT_NAME", "full_video_rnn_reports.html")
OUT = DEPLOY / EXPORT_NAME
STATUS_PATH = OUT / "status.json"
MANIFEST_PATH = OUT / "full_video_top8_manifest.json"
COMPARISON = ROOT / "comparison_grid128_sequence_1day_v1" / "comparison_manifest.json"
DATASET = ROOT / "datasets" / "w8_s1_h2" / "dynamics_dataset.json"
GRID_STATES_ROOT = ROOT / "grid_states"
TOP_N = 8
PANEL_GAP = 4
FPS = 12
TMP_ROOT = Path("/tmp/neurobench_top8_video_reports")
RNN_PIXEL_FAMILIES = {"pixel_convgru", "convlstm_residual"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return "_".join("".join(ch if ch.isalnum() else "_" for ch in str(value)).split("_")) or "item"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_top_models() -> list[dict[str, Any]]:
    rows = []
    for row in read_json(COMPARISON).get("rows", []):
        if row.get("dataset_key") != "w8_s1_h2" or row.get("hyperparameter_group") != "pixel_sequence":
            continue
        if str(row.get("model_family") or "") not in RNN_PIXEL_FAMILIES:
            continue
        if row.get("primary_improvement_over_persistence_mse") is None:
            continue
        checkpoint = Path(str(row.get("metrics_path") or "")).parent / "concept_checkpoint.pt"
        if checkpoint.exists():
            item = dict(row)
            item["checkpoint_path"] = checkpoint.as_posix()
            rows.append(item)
    rows.sort(key=lambda r: float(r.get("primary_improvement_over_persistence_mse") or float("-inf")), reverse=True)
    return rows[:TOP_N]


def load_videos(dataset: dict[str, Any]) -> list[dict[str, str]]:
    splits = dataset.get("splits") or {}
    videos = []
    for split_name in ("train", "val", "test"):
        for video_id in splits.get(f"{split_name}_video_ids", []):
            path = GRID_STATES_ROOT / str(video_id) / "grid_states.npz"
            if path.exists():
                videos.append({"video_id": str(video_id), "split": split_name, "slug": slug(str(video_id)), "grid_states_path": path.as_posix()})
    return videos


def load_video_arrays(video_id: str, windowing: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    window_frames = int(windowing["window_frames"])
    horizon = int(windowing["prediction_horizon_frames"])
    stride = int(windowing.get("temporal_stride_frames") or windowing.get("stride_frames") or 1)
    with np.load(GRID_STATES_ROOT / video_id / "grid_states.npz", allow_pickle=False) as arrays:
        frames = np.asarray(arrays["grid_state"], dtype=np.float32)
    source_frame_count = int(frames.shape[0])
    frames = np.clip(np.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    n = int((frames.shape[0] - window_frames - horizon) // stride + 1)
    if n <= 0:
        raise ValueError(f"Video {video_id!r} has too few frames for window={window_frames}, horizon={horizon}.")
    windows = np.empty((n, window_frames, 1, frames.shape[1], frames.shape[2]), dtype=np.float32)
    targets = np.empty((n, 1, frames.shape[1], frames.shape[2]), dtype=np.float32)
    target_indices = np.empty((n,), dtype=np.int64)
    for i in range(n):
        start = i * stride
        target_index = start + window_frames + horizon - 1
        windows[i] = np.transpose(frames[start:start + window_frames], (0, 3, 1, 2))
        targets[i] = np.transpose(frames[target_index], (2, 0, 1))
        target_indices[i] = int(target_index)
    return windows, targets, target_indices, source_frame_count


def as_gray(frame: np.ndarray, *, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return np.round(np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)


def panel_image(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    target2 = np.asarray(target[0], dtype=np.float32)
    pred2 = np.asarray(pred[0], dtype=np.float32)
    err_model = np.abs(pred2 - target2)
    panes = [as_gray(target2), as_gray(pred2), as_gray(err_model, lo=0.0, hi=0.25)]
    h, w = panes[0].shape
    panel = np.full((h, w * len(panes) + PANEL_GAP * (len(panes) - 1)), 18, dtype=np.uint8)
    x = 0
    for pane in panes:
        panel[:, x:x + w] = pane
        x += w + PANEL_GAP
    return panel


def load_model(row: dict[str, Any], device: str):
    torch = _torch()
    ckpt = torch.load(Path(row["checkpoint_path"]), map_location=device)
    model = _build_spatial_pixel_model(architecture=str(ckpt["architecture"]), input_channels=int(ckpt.get("input_channels", 1)), window_frames=int(ckpt.get("window_frames", 8)), hidden_channels=int(ckpt["hidden_channels"]), num_layers=int(ckpt["num_layers"]), residual_scale=float(ckpt["residual_scale"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def encode_mp4(frame_dir: Path, mp4_path: Path) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp4_path.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], check=True)
    os.replace(tmp, mp4_path)


def make_model_record(row: dict[str, Any], rank: int, rel_video: Path, rel_poster: Path, frame_count: int, *, skipped: bool) -> dict[str, Any]:
    return {"rank": rank, "experiment_id": row.get("experiment_id"), "model_family": row.get("model_family"), "model_kind": row.get("model_kind"), "objective": row.get("objective"), "dataset_key": row.get("dataset_key"), "primary_improvement_over_persistence_mse": row.get("primary_improvement_over_persistence_mse"), "test_improvement_over_persistence_mse": row.get("test_improvement_over_persistence_mse"), "test_decoded_prediction_mse": row.get("test_decoded_prediction_mse"), "test_persistence_mse": row.get("test_persistence_mse"), "hyperparameter_summary": row.get("hyperparameter_summary"), "frame_count": int(frame_count), "fps": FPS, "duration_seconds": float(frame_count) / float(FPS), "video_path": rel_video.as_posix(), "poster_path": rel_poster.as_posix(), "skipped_existing": bool(skipped)}


def export_one(model, row: dict[str, Any], rank: int, video: dict[str, str], windows: np.ndarray, targets: np.ndarray, batch_size: int, device: str) -> dict[str, Any]:
    torch = _torch()
    exp_id = str(row["experiment_id"])
    vslug = video["slug"]
    mslug = f"model_{rank:02d}_{slug(exp_id)}"
    rel_video = Path(EXPORT_NAME) / "videos" / vslug / f"{mslug}.mp4"
    rel_poster = Path(EXPORT_NAME) / "posters" / vslug / f"{mslug}.png"
    mp4_path = DEPLOY / rel_video
    poster_path = DEPLOY / rel_poster
    if mp4_path.exists() and poster_path.exists():
        return make_model_record(row, rank, rel_video, rel_poster, int(windows.shape[0]), skipped=True)
    frame_dir = TMP_ROOT / vslug / mslug
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    n = int(windows.shape[0])
    started = time.time()
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            stop = min(start + int(batch_size), n)
            xb = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy().astype(np.float32)
            for offset, pred_frame in enumerate(pred):
                idx = start + offset
                panel = panel_image(targets[idx], pred_frame)
                write_png_gray8(frame_dir / f"frame_{idx:05d}.png", int(panel.shape[1]), int(panel.shape[0]), panel.tobytes())
            if start == 0 or stop == n or stop % max(int(batch_size) * 8, 1) == 0:
                status(current_video=video["video_id"], current_split=video["split"], current_model=exp_id, current_model_rank=rank, current_model_frames_done=stop, current_video_frames=n)
                print(f"{now()} {video['split']} {video['video_id']} model {rank}/{TOP_N}: {stop}/{n}", flush=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(frame_dir / "frame_00000.png", poster_path)
    encode_mp4(frame_dir, mp4_path)
    shutil.rmtree(frame_dir)
    record = make_model_record(row, rank, rel_video, rel_poster, n, skipped=False)
    record["elapsed_seconds"] = time.time() - started
    return record


def write_viewer_html() -> None:
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RNN Full-Video Reports</title><style>:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#101316;color:#e9edf0}}body{{margin:0;background:#101316}}header{{padding:16px 20px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;gap:16px;align-items:end}}h1{{margin:0;font-size:20px;letter-spacing:0}}main{{display:grid;grid-template-columns:320px minmax(0,1fr);min-height:calc(100vh - 70px)}}aside{{border-right:1px solid #30363d;padding:14px;overflow:auto}}section{{padding:16px 20px 24px;overflow:auto}}label{{display:block;color:#aab3bc;font-size:12px;margin:10px 0 5px}}select,button,input{{background:#1a2026;color:#eef2f5;border:1px solid #3a424b;border-radius:6px;padding:8px 10px}}select,button{{width:100%}}button{{text-align:left;cursor:pointer;margin:0 0 8px}}button.active{{border-color:#56a6ff;background:#172333}}.top{{display:grid;grid-template-columns:1fr 120px 120px 96px;gap:10px;align-items:end;margin-bottom:12px}}.scrubber{{display:grid;grid-template-columns:78px 78px 1fr 88px;gap:10px;align-items:center;margin:0 0 12px}}.scrubber button{{margin:0;text-align:center}}input[type=range]{{width:100%;padding:0}}video{{display:block;width:min(100%,780px);background:#050607;border:1px solid #30363d}}.status,.hint{{color:#aab3bc;font-size:13px;line-height:1.4}}.meta{{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin:12px 0}}.metric{{background:#181d22;border:1px solid #30363d;border-radius:6px;padding:9px 10px}}.metric span{{display:block;color:#99a3ad;font-size:12px;margin-bottom:4px}}code{{color:#c6dcff;font-size:12px}}a{{color:#9dccff}}@media(max-width:860px){{main,.top,.meta,.scrubber{{grid-template-columns:1fr}}aside{{border-right:0;border-bottom:1px solid #30363d}}}}</style></head>
<body><header><div><h1>RNN Full-Video Reports</h1><div class="status" id="status">Loading...</div></div><div class="status"><a href="index.html">Dashboard index</a></div></header><main><aside><label for="splitSelect">Split</label><select id="splitSelect"></select><label for="videoSelect">Video</label><select id="videoSelect"></select><label>Model</label><div id="models"></div></aside><section><div class="top"><div class="hint" id="selectionLabel"></div><div><label for="rate">Playback speed</label><select id="rate"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select></div><div><label><input id="loop" type="checkbox" checked> Loop</label></div><button id="playPause">Play</button></div><div class="scrubber"><button id="prevFrame">-1 frame</button><button id="nextFrame">+1 frame</button><input id="frameSlider" type="range" min="0" max="0" value="0"><div class="status" id="frameLabel">0 / 0</div></div><video id="video" controls playsinline preload="metadata"></video><div class="hint">Panes left to right: target frame, model prediction, model absolute error. Persistence is omitted because it is the last input frame carried forward.</div><div class="meta" id="meta"></div></section></main>
<script>let manifest=null,statusData=null,split='test',videoIndex=0,modelIndex=0,isSeekingFrame=false;const statusEl=document.getElementById('status'),splitSelect=document.getElementById('splitSelect'),videoSelect=document.getElementById('videoSelect'),modelsEl=document.getElementById('models'),videoEl=document.getElementById('video'),meta=document.getElementById('meta'),selectionLabel=document.getElementById('selectionLabel'),rate=document.getElementById('rate'),loop=document.getElementById('loop'),playPause=document.getElementById('playPause'),prevFrame=document.getElementById('prevFrame'),nextFrame=document.getElementById('nextFrame'),frameSlider=document.getElementById('frameSlider'),frameLabel=document.getElementById('frameLabel');const fmt=v=>v===null||v===undefined||Number.isNaN(Number(v))?'n/a':Number(v).toPrecision(6);async function loadJson(path){{const r=await fetch(path+'?t='+Date.now());if(!r.ok)throw new Error(path+' '+r.status);return await r.json()}}function videosForSplit(){{return (manifest?.videos||[]).filter(v=>v.split===split)}}function currentModel(){{const v=videosForSplit()[videoIndex];return v?.models?.[modelIndex]||null}}function fpsValue(){{return Number(currentModel()?.fps||{FPS})||{FPS}}}function currentFrame(){{return Math.max(0,Math.min(Number(frameSlider.max)||0,Math.round(videoEl.currentTime*fpsValue())))}}function seekFrame(index){{const max=Number(frameSlider.max)||0;const next=Math.max(0,Math.min(max,index));isSeekingFrame=true;videoEl.pause();playPause.textContent='Play';videoEl.currentTime=next/fpsValue();frameSlider.value=String(next);frameLabel.textContent=`${{next+1}} / ${{max+1}}`;setTimeout(()=>{{isSeekingFrame=false}},100)}}async function refresh(){{try{{statusData=await loadJson('{EXPORT_NAME}/status.json')}}catch(e){{}}try{{manifest=await loadJson('{EXPORT_NAME}/full_video_top8_manifest.json')}}catch(e){{}}render()}}function render(){{const st=statusData||{{}};statusEl.textContent=`${{st.state||'pending'}} · ${{st.completed_video_model_count||0}}/${{st.total_video_model_count||0}} model-videos · ${{st.current_split||''}} ${{st.current_video||''}}`;if(!manifest){{modelsEl.innerHTML='<div class="status">Reports are being generated. This page will update automatically.</div>';return}}const ready=(manifest.videos||[]).find(v=>(v.models||[]).length);if(ready&&!videosForSplit().some(v=>(v.models||[]).length)){{split=ready.split;videoIndex=(manifest.videos||[]).filter(v=>v.split===split).findIndex(v=>(v.models||[]).length);if(videoIndex<0)videoIndex=0;modelIndex=0}}const splits=[...new Set((manifest.videos||[]).map(v=>v.split))];splitSelect.innerHTML=splits.map(s=>`<option value="${{s}}">${{s}}</option>`).join('');if(splits.includes(split))splitSelect.value=split;const vids=videosForSplit();if(videoIndex>=vids.length)videoIndex=0;videoSelect.innerHTML=vids.map((v,i)=>`<option value="${{i}}">${{v.video_id}} (${{v.frame_count||0}} frames)</option>`).join('');videoSelect.value=String(videoIndex);const v=vids[videoIndex];const models=v?.models||[];if(modelIndex>=models.length)modelIndex=0;modelsEl.innerHTML=models.length?models.map((m,i)=>`<button class="${{i===modelIndex?'active':''}}" data-i="${{i}}">#${{m.rank}} ${{m.model_family}}<br><code>${{m.experiment_id}}</code><br>${{m.frame_count}} frames · ${{Math.round(m.duration_seconds)}}s</button>`).join(''):'<div class="status">This video is still pending.</div>';modelsEl.querySelectorAll('button').forEach(b=>b.onclick=()=>{{modelIndex=Number(b.dataset.i);setVideo(true);renderMeta();render()}});setVideo(false);renderMeta();updateFrameControls()}}function setVideo(force){{const v=videosForSplit()[videoIndex];const m=v?.models?.[modelIndex];selectionLabel.textContent=v?`${{v.split}} · ${{v.video_id}}`:'No video selected';if(m&&(force||videoEl.getAttribute('src')!==m.video_path)){{videoEl.src=m.video_path;videoEl.poster=m.poster_path||'';videoEl.playbackRate=Number(rate.value)||1;videoEl.loop=loop.checked;frameSlider.max=String(Math.max(0,(m.frame_count||1)-1));seekFrame(0)}}}}function updateFrameControls(){{const m=currentModel();const max=Math.max(0,(m?.frame_count||1)-1);frameSlider.max=String(max);const frame=currentFrame();if(!isSeekingFrame)frameSlider.value=String(frame);frameLabel.textContent=`${{frame+1}} / ${{max+1}}`}}function renderMeta(){{const v=videosForSplit()[videoIndex];const m=v?.models?.[modelIndex];if(!m){{meta.innerHTML='';return}}meta.innerHTML=[['Split',v.split],['Video',v.video_id],['Frames',m.frame_count],['Duration',`${{Math.round(m.duration_seconds)}}s @ ${{m.fps}}fps`],['Test improvement vs persistence',fmt(m.test_improvement_over_persistence_mse)],['Model MSE',fmt(m.test_decoded_prediction_mse)],['Persistence MSE',fmt(m.test_persistence_mse)],['Hparams',m.hyperparameter_summary||'n/a']].map(([k,val])=>`<div class="metric"><span>${{k}}</span>${{val}}</div>`).join('')}}splitSelect.onchange=()=>{{split=splitSelect.value;videoIndex=0;modelIndex=0;render()}};videoSelect.onchange=()=>{{videoIndex=Number(videoSelect.value);modelIndex=0;render()}};rate.onchange=()=>{{videoEl.playbackRate=Number(rate.value)||1}};loop.onchange=()=>{{videoEl.loop=loop.checked}};playPause.onclick=()=>{{if(videoEl.paused){{videoEl.play();playPause.textContent='Pause'}}else{{videoEl.pause();playPause.textContent='Play'}}}};prevFrame.onclick=()=>seekFrame(currentFrame()-1);nextFrame.onclick=()=>seekFrame(currentFrame()+1);frameSlider.oninput=()=>seekFrame(Number(frameSlider.value)||0);videoEl.ontimeupdate=updateFrameControls;videoEl.onloadedmetadata=updateFrameControls;videoEl.onplay=()=>{{playPause.textContent='Pause'}};videoEl.onpause=()=>{{playPause.textContent='Play'}};setInterval(refresh,5000);refresh();</script></body></html>"""
    (DEPLOY / REPORT_NAME).write_text(html, encoding="utf-8")

def write_manifest(videos: list[dict[str, Any]], windowing: dict[str, Any], device: str, state: str) -> None:
    write_json(MANIFEST_PATH, {"schema_version": 2, "created_at": now(), "state": state, "device": device, "fps": FPS, "windowing": windowing, "videos": videos})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_viewer_html()
    dataset = read_json(DATASET)
    windowing = dataset["windowing"]
    videos = load_videos(dataset)
    models = load_top_models()
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(os.environ.get("NEUROBENCH_FULL_VIDEO_BATCH", "32" if device == "cuda" else "4"))
    manifest_videos = []
    for v in videos:
        item = dict(v)
        item["models"] = []
        manifest_videos.append(item)
    total = len(videos) * len(models)
    done = 0
    status(state="running", pid=os.getpid(), device=device, started_at=now(), total_video_count=len(videos), selected_model_count=len(models), total_video_model_count=total, completed_video_model_count=0)
    write_manifest(manifest_videos, windowing, device, "running")
    loaded_models = [(rank, row, load_model(row, device)) for rank, row in enumerate(models, start=1)]
    try:
        for v_index, video in enumerate(videos):
            windows, targets, target_indices, source_frame_count = load_video_arrays(video["video_id"], windowing)
            manifest_videos[v_index]["frame_count"] = int(windows.shape[0])
            manifest_videos[v_index]["source_frame_count"] = int(source_frame_count)
            manifest_videos[v_index]["target_start_index"] = int(target_indices[0])
            manifest_videos[v_index]["target_end_index"] = int(target_indices[-1])
            for rank, row, model in loaded_models:
                status(current_split=video["split"], current_video=video["video_id"], current_model_rank=rank, current_model=row.get("experiment_id"), completed_video_model_count=done, total_video_model_count=total)
                manifest_videos[v_index]["models"].append(export_one(model, row, rank, video, windows, targets, batch_size, device))
                done += 1
                status(completed_video_model_count=done, total_video_model_count=total)
                write_manifest(manifest_videos, windowing, device, "running")
            del windows, targets
    finally:
        for _rank, _row, model in loaded_models:
            del model
    write_manifest(manifest_videos, windowing, device, "complete")
    status(state="complete", completed_at=now(), completed_video_model_count=done, total_video_model_count=total)
    print(f"{now()} complete: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        status(state="failed", error=repr(exc))
        print(f"{now()} failed: {exc!r}", file=sys.stderr, flush=True)
        raise
