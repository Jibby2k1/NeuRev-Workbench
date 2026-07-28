#!/usr/bin/env python3
"""Build synchronized, presentation-ready MP4 clips for Spon Ca Burst.

The source review window is UI frames 1800--2359 (NumPy 1799:2359).
Every clip uses the same 560 frames and the same fixed, per-method display
normalization.  Scientific arrays are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
UI_START = 1800
FRAME_COUNT = 560
FPS = 25
BURSTS = ((2003, 2026), (2040, 2063), (2122, 2149), (2254, 2300))


@dataclass(frozen=True)
class Clip:
    key: str
    title: str
    path: Path
    kind: str
    lo: float
    hi: float
    gamma: float


CLIPS = (
    Clip(
        "raw",
        "Raw fluorescence",
        ROOT
        / "Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/"
        "spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
        "raw",
        241,
        4095,
        0.72,
    ),
    Clip(
        "artifact_gate",
        "Causal artifact-gated amplitude",
        ROOT
        / "Outputs/FrameDifference/spon_ca_burst_activity_gate_v1/"
        "spon_ca_burst_artifact_gate.tif",
        "tiff",
        0,
        28607,
        0.72,
    ),
    Clip(
        "ica_activity",
        "InfoMax temporal activity",
        ROOT
        / "Outputs/PairwiseSeparation/spon_ca_burst_pairwise_separation_v1/"
        "methods/infomax_tanh_ica/positive_z.tif",
        "tiff",
        0,
        65535,
        0.55,
    ),
    Clip(
        "raw_plus_ica",
        "Raw + InfoMax auxiliary feature",
        ROOT
        / "Outputs/PairwiseSeparation/spon_ca_burst_pairwise_feature_fusion_v1/"
        "review_tiffs/raw_plus_infomax_lambda0p1.tif",
        "tiff",
        0,
        65535,
        0.72,
    ),
    Clip(
        "latent_smoother",
        "Offline latent-smoothed amplitude",
        ROOT
        / "Outputs/LatentDynamics/spon_ca_burst_latent_dynamics_v1/features/"
        "selected_review_tiffs/smoother_mean.tif",
        "tiff",
        0,
        446.25,
        0.72,
    ),
)


def _open(clip: Clip) -> np.ndarray:
    if clip.kind == "raw":
        source = np.load(clip.path, mmap_mode="r")
        return source[UI_START - 1 : UI_START - 1 + FRAME_COUNT]
    return tifffile.memmap(clip.path)


def _burst_label(ui_frame: int) -> str:
    for index, (start, end) in enumerate(BURSTS, start=1):
        if start <= ui_frame <= end:
            return f"ANNOTATED BURST {index}"
    return ""


def _normalize(frame: np.ndarray, clip: Clip) -> np.ndarray:
    scaled = np.clip(
        (np.asarray(frame, dtype=np.float32) - clip.lo) / (clip.hi - clip.lo),
        0,
        1,
    )
    scaled = np.power(scaled, clip.gamma)
    return np.rint(scaled * 255).astype(np.uint8)


def _compose(frame: np.ndarray, clip: Clip, index: int) -> np.ndarray:
    ui_frame = UI_START + index
    gray = _normalize(frame, clip)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    # H.264 yuv420p requires even dimensions; preserve all 573 source columns.
    canvas = Image.new("RGB", (rgb.shape[1] + 1, rgb.shape[0] + 42), (7, 13, 22))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, rgb.shape[0] + 7), clip.title, font=font, fill=(235, 242, 250))
    draw.text(
        (10, rgb.shape[0] + 23),
        f"UI frame {ui_frame}  |  50 Hz source; 25 fps playback",
        font=font,
        fill=(145, 163, 184),
    )
    burst = _burst_label(ui_frame)
    if burst:
        box = draw.textbbox((0, 0), burst, font=font)
        width = box[2] - box[0]
        draw.rounded_rectangle(
            (rgb.shape[1] - width - 22, rgb.shape[0] + 8,
             rgb.shape[1] - 8, rgb.shape[0] + 30),
            radius=5,
            fill=(14, 116, 144),
        )
        draw.text(
            (rgb.shape[1] - width - 15, rgb.shape[0] + 13),
            burst,
            font=font,
            fill=(255, 255, 255),
        )
    return np.asarray(canvas)


def _encode(clip: Clip, frames: np.ndarray, out_dir: Path) -> dict:
    if frames.shape != (FRAME_COUNT, 340, 573):
        raise ValueError(f"{clip.key}: unexpected shape {frames.shape}")
    out_path = out_dir / f"{clip.key}.mp4"
    partial = out_path.with_suffix(".partial.mp4")
    poster = out_dir / f"{clip.key}_poster.png"
    if out_path.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path} or {partial}")

    sample = _compose(frames[UI_START + 212 - UI_START], clip, 212)
    Image.fromarray(sample).save(poster)
    height, width = sample.shape[:2]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(FRAME_COUNT):
            process.stdin.write(_compose(frames[index], clip, index).tobytes())
            if index and index % 100 == 0:
                print(f"{clip.key}: {index}/{FRAME_COUNT}", flush=True)
    finally:
        process.stdin.close()
    result = process.wait()
    if result:
        raise RuntimeError(f"ffmpeg failed for {clip.key}: {result}")
    os.replace(partial, out_path)
    return {
        "key": clip.key,
        "title": clip.title,
        "video": out_path.name,
        "poster": poster.name,
        "frames": FRAME_COUNT,
        "ui_frames": [UI_START, UI_START + FRAME_COUNT - 1],
        "source_fps": 50,
        "playback_fps": FPS,
        "display": {
            "lo": clip.lo,
            "hi": clip.hi,
            "gamma": clip.gamma,
            "fixed_across_time": True,
        },
        "source": str(clip.path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out_dir = args.output.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out_dir}")
    out_dir.mkdir(parents=True)

    manifest = []
    for clip in CLIPS:
        if not clip.path.is_file():
            raise FileNotFoundError(clip.path)
        frames = _open(clip)
        manifest.append(_encode(clip, frames, out_dir))
    (out_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "Spon Ca Burst",
                "review_window_ui_inclusive": [1800, 2359],
                "burst_intervals_ui_inclusive": BURSTS,
                "clips": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} clips to {out_dir}")


if __name__ == "__main__":
    main()
