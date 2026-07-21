#!/usr/bin/env python3
"""Export full-video review panels for the top grid128 pixel models."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
OUT = DEPLOY / "full_video_top8_v1"
STATUS_PATH = OUT / "status.json"
MANIFEST_PATH = OUT / "full_video_top8_manifest.json"
COMPARISON = ROOT / "comparison_grid128_sequence_1day_v1" / "comparison_manifest.json"
VIDEO_ID = "8 left"
GRID_STATES = ROOT / "grid_states" / VIDEO_ID / "grid_states.npz"
DATASET = ROOT / "datasets" / "w8_s1_h2" / "dynamics_dataset.json"
TOP_N = 8
PANEL_GAP = 4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def status(**updates: Any) -> None:
    payload: dict[str, Any] = {}
    if STATUS_PATH.exists():
        try:
            payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(updates)
    payload["updated_at"] = now()
    write_json(STATUS_PATH, payload)


def load_top_models() -> list[dict[str, Any]]:
    manifest = json.loads(COMPARISON.read_text(encoding="utf-8"))
    rows = []
    for row in manifest.get("rows", []):
        if row.get("dataset_key") != "w8_s1_h2":
            continue
        if row.get("hyperparameter_group") != "pixel_sequence":
            continue
        metric = row.get("primary_improvement_over_persistence_mse")
        if metric is None:
            continue
        metrics_path = Path(str(row.get("metrics_path") or ""))
        checkpoint = metrics_path.parent / "concept_checkpoint.pt"
        if checkpoint.exists():
            item = dict(row)
            item["checkpoint_path"] = checkpoint.as_posix()
            rows.append(item)
    rows.sort(key=lambda r: float(r.get("primary_improvement_over_persistence_mse") or float("-inf")), reverse=True)
    return rows[:TOP_N]


def load_video_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    windowing = dataset["windowing"]
    window_frames = int(windowing["window_frames"])
    horizon = int(windowing["prediction_horizon_frames"])
    stride = int(windowing.get("temporal_stride_frames") or windowing.get("stride_frames") or 1)
    with np.load(GRID_STATES, allow_pickle=False) as arrays:
        frames = np.asarray(arrays["grid_state"], dtype=np.float32)
    frames = np.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
    frames = np.clip(frames, 0.0, 1.0)
    n = int((frames.shape[0] - window_frames - horizon) // stride + 1)
    if n <= 0:
        raise ValueError(f"Video {VIDEO_ID!r} has too few frames for window={window_frames}, horizon={horizon}.")
    windows = np.empty((n, window_frames, 1, frames.shape[1], frames.shape[2]), dtype=np.float32)
    targets = np.empty((n, 1, frames.shape[1], frames.shape[2]), dtype=np.float32)
    source_indices = np.empty((n,), dtype=np.int64)
    for i in range(n):
        start = i * stride
        target_index = start + window_frames + horizon - 1
        windows[i] = np.transpose(frames[start : start + window_frames], (0, 3, 1, 2))
        targets[i] = np.transpose(frames[target_index], (2, 0, 1))
        source_indices[i] = int(target_index)
    return windows, targets, source_indices, windowing


def as_gray(frame: np.ndarray, *, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def panel_image(target: np.ndarray, pred: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    target2 = np.asarray(target[0], dtype=np.float32)
    pred2 = np.asarray(pred[0], dtype=np.float32)
    persist2 = np.asarray(persistence[0], dtype=np.float32)
    err_model = np.abs(pred2 - target2)
    err_persist = np.abs(persist2 - target2)
    improvement = err_persist - err_model
    panes = [
        as_gray(target2),
        as_gray(pred2),
        as_gray(persist2),
        as_gray(err_model, lo=0.0, hi=0.25),
        as_gray(err_persist, lo=0.0, hi=0.25),
        as_gray(improvement, lo=-0.15, hi=0.15),
    ]
    h, w = panes[0].shape
    panel = np.full((h, w * len(panes) + PANEL_GAP * (len(panes) - 1)), 18, dtype=np.uint8)
    x = 0
    for pane in panes:
        panel[:, x : x + w] = pane
        x += w + PANEL_GAP
    return panel


def export_model(model_row: dict[str, Any], rank: int, windows: np.ndarray, targets: np.ndarray, source_indices: np.ndarray, batch_size: int, device: str) -> dict[str, Any]:
    torch = _torch()
    checkpoint_path = Path(model_row["checkpoint_path"])
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = _build_spatial_pixel_model(
        architecture=str(ckpt["architecture"]),
        input_channels=int(ckpt.get("input_channels", 1)),
        window_frames=int(ckpt.get("window_frames", windows.shape[1])),
        hidden_channels=int(ckpt["hidden_channels"]),
        num_layers=int(ckpt["num_layers"]),
        residual_scale=float(ckpt["residual_scale"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    exp_id = str(model_row["experiment_id"])
    model_dir = OUT / "frames" / f"model_{rank:02d}_{exp_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    n = int(windows.shape[0])
    started = time.time()
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            stop = min(start + int(batch_size), n)
            xb = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy().astype(np.float32)
            for offset, pred_frame in enumerate(pred):
                idx = start + offset
                panel = panel_image(targets[idx], pred_frame, windows[idx, -1])
                rel = Path("full_video_top8_v1") / "frames" / f"model_{rank:02d}_{exp_id}" / f"frame_{idx:04d}.png"
                out_path = DEPLOY / rel
                write_png_gray8(out_path, int(panel.shape[1]), int(panel.shape[0]), panel.tobytes())
                frame_paths.append(rel.as_posix())
            if start == 0 or stop == n or stop % max(int(batch_size) * 4, 1) == 0:
                status(
                    state="running",
                    current_model_rank=rank,
                    current_model=exp_id,
                    current_model_frames_done=stop,
                    frames_per_model=n,
                    total_frames_done=(rank - 1) * n + stop,
                    total_frames_planned=TOP_N * n,
                )
                print(f"{now()} model {rank}/{TOP_N} {exp_id}: {stop}/{n} frames", flush=True)
    elapsed = time.time() - started
    return {
        "rank": rank,
        "experiment_id": exp_id,
        "model_family": model_row.get("model_family"),
        "model_kind": model_row.get("model_kind"),
        "objective": model_row.get("objective"),
        "dataset_key": model_row.get("dataset_key"),
        "primary_improvement_over_persistence_mse": model_row.get("primary_improvement_over_persistence_mse"),
        "test_improvement_over_persistence_mse": model_row.get("test_improvement_over_persistence_mse"),
        "test_decoded_prediction_mse": model_row.get("test_decoded_prediction_mse"),
        "test_persistence_mse": model_row.get("test_persistence_mse"),
        "hyperparameter_summary": model_row.get("hyperparameter_summary"),
        "checkpoint_path": checkpoint_path.as_posix(),
        "frame_count": len(frame_paths),
        "frame_paths": frame_paths,
        "elapsed_seconds": elapsed,
    }


def write_viewer_html() -> None:
    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top 8 Full-Video Review</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #111417; color: #e8ecef; }
    body { margin: 0; background: #111417; }
    header { padding: 18px 22px; border-bottom: 1px solid #30363d; display: flex; gap: 18px; align-items: end; justify-content: space-between; }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    main { display: grid; grid-template-columns: 330px 1fr; min-height: calc(100vh - 74px); }
    aside { border-right: 1px solid #30363d; padding: 14px; overflow: auto; }
    button, input { background: #1b2026; color: #e8ecef; border: 1px solid #3a424b; border-radius: 6px; padding: 8px 10px; }
    button { cursor: pointer; width: 100%; text-align: left; margin-bottom: 8px; }
    button.active { border-color: #56a6ff; background: #172333; }
    .viewer { padding: 16px 20px 24px; overflow: auto; }
    .controls { display: grid; grid-template-columns: auto 1fr 92px 110px; gap: 10px; align-items: center; margin-bottom: 12px; }
    .controls button { width: auto; margin: 0; text-align: center; }
    input[type=range] { width: 100%; padding: 0; }
    img { display: block; width: min(100%, 1182px); image-rendering: pixelated; border: 1px solid #30363d; background: #0b0d0f; }
    .meta { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin: 12px 0; }
    .metric { background: #181d22; border: 1px solid #30363d; border-radius: 6px; padding: 9px 10px; }
    .metric span { display: block; color: #99a3ad; font-size: 12px; margin-bottom: 4px; }
    .status { color: #b7c0c9; font-size: 13px; }
    .legend { color: #b7c0c9; font-size: 13px; margin-top: 8px; }
    code { color: #c6dcff; }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #30363d; } .controls, .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>Top 8 Full-Video Review</h1><div class="status" id="status">Loading status...</div></div>
    <div class="status"><a href="index.html" style="color:#9dccff">Dashboard index</a></div>
  </header>
  <main>
    <aside id="models"></aside>
    <section class="viewer">
      <div class="controls">
        <button id="play">Play</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <input id="fps" type="number" min="1" max="50" value="12">
        <div id="frameLabel" class="status">0 / 0</div>
      </div>
      <img id="frame" alt="Full-video diagnostic panel">
      <div class="legend">Panes left to right: target, prediction, persistence, model absolute error, persistence absolute error, persistence-minus-model error.</div>
      <div class="meta" id="meta"></div>
    </section>
  </main>
  <script>
    let manifest = null, statusData = null, modelIndex = 0, frameIndex = 0, timer = null;
    const statusEl = document.getElementById('status');
    const modelsEl = document.getElementById('models');
    const scrub = document.getElementById('scrub');
    const frame = document.getElementById('frame');
    const play = document.getElementById('play');
    const fps = document.getElementById('fps');
    const frameLabel = document.getElementById('frameLabel');
    const meta = document.getElementById('meta');
    const fmt = v => v === null || v === undefined || Number.isNaN(Number(v)) ? 'n/a' : Number(v).toPrecision(6);
    async function loadJson(path) { const r = await fetch(path + '?t=' + Date.now()); if (!r.ok) throw new Error(path + ' ' + r.status); return await r.json(); }
    async function refresh() {
      try { statusData = await loadJson('full_video_top8_v1/status.json'); } catch (e) {}
      try { manifest = await loadJson('full_video_top8_v1/full_video_top8_manifest.json'); } catch (e) {}
      render();
    }
    function render() {
      const st = statusData || {};
      statusEl.textContent = `${st.state || 'pending'} · ${st.total_frames_done || 0}/${st.total_frames_planned || 0} frames · updated ${st.updated_at || 'n/a'}`;
      if (!manifest || !manifest.models || !manifest.models.length) {
        modelsEl.innerHTML = '<div class="status">The GPU export is running. Models will appear as soon as the first manifest is written.</div>';
        return;
      }
      modelsEl.innerHTML = manifest.models.map((m, i) => `<button class="${i===modelIndex?'active':''}" data-i="${i}">#${m.rank} ${m.model_family}<br><code>${m.experiment_id}</code><br>frames ${m.frame_count}</button>`).join('');
      modelsEl.querySelectorAll('button').forEach(b => b.onclick = () => { modelIndex = Number(b.dataset.i); frameIndex = 0; render(); });
      const m = manifest.models[Math.min(modelIndex, manifest.models.length - 1)];
      const max = Math.max(0, (m.frame_paths || []).length - 1);
      scrub.max = String(max);
      frameIndex = Math.min(frameIndex, max);
      scrub.value = String(frameIndex);
      frameLabel.textContent = `${frameIndex + 1} / ${max + 1}`;
      if (m.frame_paths && m.frame_paths[frameIndex]) frame.src = m.frame_paths[frameIndex];
      meta.innerHTML = [
        ['Video', manifest.video_id],
        ['Test improvement', fmt(m.test_improvement_over_persistence_mse)],
        ['Model MSE', fmt(m.test_decoded_prediction_mse)],
        ['Persistence MSE', fmt(m.test_persistence_mse)],
        ['Frames', `${m.frame_count} / ${manifest.frames_per_model}`],
        ['Hparams', m.hyperparameter_summary || 'n/a'],
      ].map(([k,v]) => `<div class="metric"><span>${k}</span>${v}</div>`).join('');
    }
    scrub.oninput = () => { frameIndex = Number(scrub.value); render(); };
    play.onclick = () => {
      if (timer) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
      play.textContent = 'Pause';
      timer = setInterval(() => {
        const m = manifest && manifest.models && manifest.models[modelIndex];
        if (!m || !m.frame_paths || frameIndex >= m.frame_paths.length - 1) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
        frameIndex += 1; render();
      }, 1000 / Math.max(1, Number(fps.value) || 12));
    };
    setInterval(refresh, 5000);
    refresh();
  </script>
</body>
</html>
"""
    (DEPLOY / "full_video_top8.html").write_text(html, encoding="utf-8")


def write_partial_manifest(models: list[dict[str, Any]], windowing: dict[str, Any], source_indices: np.ndarray, device: str) -> None:
    payload = {
        "schema_version": 1,
        "created_at": now(),
        "state": "running",
        "device": device,
        "video_id": VIDEO_ID,
        "grid_states_path": GRID_STATES.as_posix(),
        "frames_per_model": int(source_indices.shape[0]),
        "source_target_indices": [int(v) for v in source_indices.tolist()],
        "windowing": windowing,
        "models": models,
    }
    write_json(MANIFEST_PATH, payload)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_viewer_html()
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(os.environ.get("NEUROBENCH_FULL_VIDEO_BATCH", "16" if device == "cuda" else "4"))
    status(state="starting", pid=os.getpid(), device=device, video_id=VIDEO_ID, started_at=now())
    if device != "cuda":
        print(f"{now()} WARNING: CUDA is not available; running on CPU.", flush=True)
    models = load_top_models()
    if len(models) < TOP_N:
        raise RuntimeError(f"Only found {len(models)} checkpointed pixel models; expected {TOP_N}.")
    windows, targets, source_indices, windowing = load_video_arrays()
    status(
        state="running",
        pid=os.getpid(),
        device=device,
        selected_model_count=len(models),
        frames_per_model=int(windows.shape[0]),
        total_frames_planned=int(windows.shape[0]) * len(models),
        total_frames_done=0,
    )
    exported: list[dict[str, Any]] = []
    write_partial_manifest(exported, windowing, source_indices, device)
    for rank, row in enumerate(models, start=1):
        exported.append(export_model(row, rank, windows, targets, source_indices, batch_size, device))
        write_partial_manifest(exported, windowing, source_indices, device)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["state"] = "complete"
    manifest["completed_at"] = now()
    write_json(MANIFEST_PATH, manifest)
    status(state="complete", completed_at=now(), total_frames_done=int(windows.shape[0]) * len(models))
    print(f"{now()} complete: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        status(state="failed", error=repr(exc))
        print(f"{now()} failed: {exc!r}", file=sys.stderr, flush=True)
        raise
