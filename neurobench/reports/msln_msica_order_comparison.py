"""Render restrained grayscale order-comparison atlases and videos."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_cascade_program_v4")
OUTPUT = Path("Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_order_visual_comparison_v1")
RAW = Path("Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy")
REVIEW = (1800, 2359)
QUIET_FRAMES = 100
REPRESENTATIVE = (1900, 2003, 2040, 2122, 2254)
FPS = 10.0


PALETTES = {
    "neutral": np.asarray([[5, 6, 6], [74, 77, 77], [153, 156, 154], [239, 238, 232]], dtype=np.float32),
    "orange": np.asarray([[6, 6, 5], [72, 70, 67], [151, 123, 91], [232, 197, 151]], dtype=np.float32),
    "green": np.asarray([[5, 7, 6], [67, 72, 69], [99, 145, 112], [198, 222, 202]], dtype=np.float32),
}


VIEWS = {
    "01_label_free_winners": [
        ("Raw amplitude", "raw", "neutral"),
        ("Original shallow · rank 1", "00_original_shallow/finalists/rank_01.npy", "orange"),
        ("Original deep · rank 1", "01_original_deep/finalists/rank_01.npy", "orange"),
        ("Switched deep · rank 1", "02_switched_per_branch/finalists/rank_01.npy", "green"),
        ("Switched 5-seed · rank 1", "03_switched_seed_ensemble/finalists/rank_01.npy", "green"),
    ],
    "02_matched_context_s5_g1_t31": [
        ("Raw amplitude", "raw", "neutral"),
        ("Original deep · S5/G1/T31", "01_original_deep/finalists/rank_03.npy", "orange"),
        ("Switched deep · S5/G1/T31", "02_switched_per_branch/finalists/rank_01.npy", "green"),
        ("Switched 5-seed · S5/G1/T31", "03_switched_seed_ensemble/finalists/rank_01.npy", "green"),
    ],
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _palette(values: np.ndarray, name: str) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    scaled = values * 3.0
    lower = np.minimum(scaled.astype(np.int32), 2)
    fraction = (scaled - lower)[..., None]
    colors = PALETTES[name]
    return np.clip(colors[lower] * (1.0 - fraction) + colors[lower + 1] * fraction, 0, 255).astype(np.uint8)


def _resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BILINEAR))


def _load() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not ROOT.is_dir() or json.loads((ROOT / "status.json").read_text())["status"] != "complete":
        raise RuntimeError("completed v4 source root is required")
    raw_source = np.load(RAW, mmap_mode="r", allow_pickle=False)
    raw = raw_source[REVIEW[0] - 1:REVIEW[1]]
    maps: dict[str, np.ndarray] = {}
    for lanes in VIEWS.values():
        for _, relative, _ in lanes:
            if relative == "raw" or relative in maps:
                continue
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != raw.shape or values.dtype != np.float32:
                raise ValueError(f"unexpected finalist map contract: {path}")
            maps[relative] = values
    return raw, maps


def _scales(raw: np.ndarray, maps: dict[str, np.ndarray]) -> dict[str, Any]:
    raw_sample = np.asarray(raw[::23, ::4, ::4], dtype=np.float32)
    raw_limits = tuple(float(item) for item in np.percentile(raw_sample, [1.0, 99.8]))
    quiet_p99: dict[str, float] = {}
    normalized_samples = []
    time_indices = np.unique(np.linspace(0, len(raw) - 1, 24).astype(np.int32))
    for relative, values in maps.items():
        quiet = np.square(np.asarray(values[:QUIET_FRAMES:2, ::4, ::4], dtype=np.float32))
        denominator = max(float(np.percentile(quiet, 99.0)), 1e-8)
        quiet_p99[relative] = denominator
        sample = np.square(np.asarray(values[time_indices, ::4, ::4], dtype=np.float32)) / denominator
        normalized_samples.append(sample.ravel())
    shared_cap = max(float(np.percentile(np.concatenate(normalized_samples), 99.8)), 1.0)
    return {"raw_limits": raw_limits, "quiet_p99_energy": quiet_p99, "shared_normalized_energy_cap": shared_cap}


def _frame_rgb(values: np.ndarray, index: int, relative: str, palette: str, scales: dict[str, Any]) -> np.ndarray:
    if relative == "raw":
        low, high = scales["raw_limits"]
        normalized = (np.asarray(values[index], dtype=np.float32) - low) / max(high - low, 1e-8)
    else:
        evidence = np.square(np.asarray(values[index], dtype=np.float32)) / scales["quiet_p99_energy"][relative]
        cap = scales["shared_normalized_energy_cap"]
        normalized = np.log1p(evidence) / np.log1p(cap)
    return _palette(normalized, palette)


def _compose(view: list[tuple[str, str, str]], raw: np.ndarray, maps: dict[str, np.ndarray], scales: dict[str, Any], frame_index: int, heading: str) -> np.ndarray:
    panel_width, panel_height, label_height, header_height = 320, 190, 28, 54
    columns = 3 if len(view) > 4 else 2
    rows = int(np.ceil(len(view) / columns))
    canvas = Image.new("RGB", (columns * panel_width, header_height + rows * (panel_height + label_height)), (10, 11, 10))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    draw.text((9, 8), f"{heading} · UI frame {REVIEW[0] + frame_index}", fill=(238, 237, 231), font=font)
    draw.text((9, 28), "Derived panels: quiet-p99 normalized energy · shared fixed scale · orange=original · green=switched", fill=(174, 177, 171), font=font)
    for lane_index, (title, relative, palette) in enumerate(view):
        row, column = divmod(lane_index, columns)
        x = column * panel_width; y = header_height + row * (panel_height + label_height)
        values = raw if relative == "raw" else maps[relative]
        rgb = _resize(_frame_rgb(values, frame_index, relative, palette, scales), (panel_width, panel_height))
        canvas.paste(Image.fromarray(rgb), (x, y + label_height))
        color = (220, 174, 116) if palette == "orange" else (143, 199, 156) if palette == "green" else (220, 220, 214)
        draw.text((x + 7, y + 8), title, fill=color, font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _render_video(path: Path, view: list[tuple[str, str, str]], raw: np.ndarray, maps: dict[str, np.ndarray], scales: dict[str, Any], heading: str) -> dict[str, Any]:
    first = _compose(view, raw, maps, scales, 0, heading)
    height, width = first.shape[:2]
    temporary = path.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    started = time.monotonic()
    for index in range(len(raw)):
        frame = first if index == 0 else _compose(view, raw, maps, scales, index, heading)
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
        if (index + 1) % 100 == 0 or index + 1 == len(raw):
            print(f"{path.name}: {index + 1}/{len(raw)}", flush=True)
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed ({code}): {stderr[-2000:]}")
    temporary.replace(path)
    return {"path": path.name, "frames": len(raw), "fps": FPS, "duration_seconds": len(raw) / FPS, "width": width, "height": height, "runtime_seconds": time.monotonic() - started}


def _atlas(path: Path, view: list[tuple[str, str, str]], raw: np.ndarray, maps: dict[str, np.ndarray], scales: dict[str, Any], heading: str) -> dict[str, Any]:
    tile_width, tile_height, label_width, header_height = 260, 154, 220, 58
    canvas = Image.new("RGB", (label_width + len(REPRESENTATIVE) * tile_width, header_height + len(view) * tile_height), (12, 13, 12))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    draw.text((10, 8), heading, fill=(238, 237, 231), font=font)
    draw.text((10, 28), "Shared fixed scale; energy normalized to each lane's quiet p99", fill=(170, 174, 168), font=font)
    for column, ui_frame in enumerate(REPRESENTATIVE):
        draw.text((label_width + column * tile_width + 8, 39), f"UI {ui_frame}", fill=(205, 205, 199), font=font)
    for row, (title, relative, palette) in enumerate(view):
        color = (220, 174, 116) if palette == "orange" else (143, 199, 156) if palette == "green" else (220, 220, 214)
        draw.text((10, header_height + row * tile_height + 66), title, fill=color, font=font)
        values = raw if relative == "raw" else maps[relative]
        for column, ui_frame in enumerate(REPRESENTATIVE):
            index = ui_frame - REVIEW[0]
            rgb = _resize(_frame_rgb(values, index, relative, palette, scales), (tile_width, tile_height))
            canvas.paste(Image.fromarray(rgb), (label_width + column * tile_width, header_height + row * tile_height))
    canvas.save(path, optimize=True)
    return {"path": path.name, "width": canvas.width, "height": canvas.height, "ui_frames": list(REPRESENTATIVE)}


def preflight() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    raw, maps = _load()
    free = shutil.disk_usage(OUTPUT.parent).free
    estimate = 2 * 100 * 2**20
    if free < estimate * 4:
        raise RuntimeError("insufficient disk headroom")
    payload = {"ready": True, "source_root": str(ROOT.resolve()), "output_root": str(OUTPUT.resolve()), "review_interval_ui": list(REVIEW), "shape": list(raw.shape), "maps": sorted(maps), "estimated_output_bytes": estimate, "disk_free_bytes": free, "cpu_threads": 4, "gpu_required": False, "palette_policy": "neutral grayscale plus restrained orange/green tint; no red-white-blue diverging scale", "scale_policy": "per-lane quiet-p99 energy normalization with one shared fixed display cap"}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "preflight.json", payload)
    _atomic_json(OUTPUT / "status.json", {"status": "preflight_ready"})
    return payload


def render() -> dict[str, Any]:
    if not (OUTPUT / "preflight.json").is_file():
        raise RuntimeError("preflight required")
    status = json.loads((OUTPUT / "status.json").read_text())
    if status["status"] == "complete":
        raise FileExistsError("completed comparison root cannot be overwritten")
    _atomic_json(OUTPUT / "status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    raw, maps = _load(); scales = _scales(raw, maps)
    _atomic_json(OUTPUT / "scaling.json", scales)
    artifacts = []
    for name, view in VIEWS.items():
        heading = "Label-free architecture winners" if name.startswith("01") else "Matched-context architecture comparison (S5/G1/T31)"
        artifacts.append({"view": name, "atlas": _atlas(OUTPUT / f"{name}_atlas.png", view, raw, maps, scales, heading), "video": _render_video(OUTPUT / f"{name}.mp4", view, raw, maps, scales, heading), "lanes": [{"title": title, "source": source, "palette": palette} for title, source, palette in view]})
    manifest = {"status": "complete", "representation": "quiet-p99 normalized energy; magnitude only; sign intentionally not encoded", "palette": {key: value.astype(int).tolist() for key, value in PALETTES.items()}, "artifacts": artifacts}
    _atomic_json(OUTPUT / "comparison_manifest.json", manifest)
    readme = """# Original-order versus switched-order visual comparison

Open `01_label_free_winners.mp4` for the operational comparison and `02_matched_context_s5_g1_t31.mp4` to reduce context confounding. The matching PNG atlases show five representative frames.

Raw amplitude is neutral grayscale. Original-order outputs use an orange-tinted grayscale; switched-order outputs use a green-tinted grayscale. Derived panels display squared magnitude normalized by that lane's quiet-period p99, followed by `log1p` compression and one shared fixed cap. Consequently the movies compare event evidence relative to each representation's own quiet baseline; they do not encode component sign or biological identity.

The label-free-winner view is the honest operational comparison. The matched-context view isolates ordering more closely, but its original-order lane is protected finalist rank 3 and therefore should be treated as a visual diagnostic rather than a deployable winner.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    _atomic_json(OUTPUT / "status.json", {"status": "complete", "video_count": 2, "atlas_count": 2, "completed_at": datetime.now(timezone.utc).isoformat()})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "render"))
    args = parser.parse_args()
    payload = preflight() if args.action == "preflight" else render()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
