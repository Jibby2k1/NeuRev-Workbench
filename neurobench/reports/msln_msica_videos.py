"""Render synchronized, fixed-scale MSLN/MS-ICA layer-bank videos."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Kind = Literal["raw", "signed", "energy", "neutral_energy", "dominant", "interaction"]


@dataclass(frozen=True)
class Layer:
    title: str
    values: np.ndarray
    kind: Kind
    limits: tuple[float, float] | None = None


def _sample_limits(values: np.ndarray, kind: Kind) -> tuple[float, float] | None:
    if kind == "dominant":
        return None
    frame_indices = np.unique(
        np.linspace(0, len(values) - 1, min(24, len(values))).astype(int)
    )
    sample = np.asarray(values[frame_indices, ::4, ::4], dtype=np.float32)
    if kind in {"raw", "interaction"}:
        low, high = np.percentile(sample, [1.0, 99.8])
        return float(low), max(float(high), float(low) + 1e-6)
    if kind == "signed":
        limit = max(float(np.percentile(np.abs(sample), 99.5)), 1e-6)
        return -limit, limit
    return 0.0, max(float(np.percentile(sample, 99.5)), 1e-6)


def _rgb(values: np.ndarray, layer: Layer, width: int, height: int) -> np.ndarray:
    from PIL import Image
    from matplotlib import colormaps

    array = np.asarray(values, dtype=np.float32)
    if layer.kind == "dominant":
        normalized = np.clip(array / 7.0, 0.0, 1.0)
        rgb = (colormaps["tab10"](normalized)[..., :3] * 255).astype(np.uint8)
        resampling = Image.Resampling.NEAREST
    else:
        low, high = layer.limits or (0.0, 1.0)
        normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
        if layer.kind == "neutral_energy":
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                "neutral_orange", ["#000000", "#4a4a4a", "#b8b8b8", "#d47a22", "#ffd29a"]
            )
        else:
            cmap = (
                "gray" if layer.kind in {"raw", "interaction"}
                else "coolwarm" if layer.kind == "signed" else "inferno"
            )
        mapped = (
            colormaps[cmap](normalized)
            if isinstance(cmap, str)
            else cmap(normalized)
        )
        rgb = (mapped[..., :3] * 255).astype(np.uint8)
        resampling = Image.Resampling.BILINEAR
    return np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((width, height), resampling)
    )


def _compose(
    layers: list[Layer],
    frame_index: int,
    ui_frame: int,
    heading: str,
    *,
    columns: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    panel_width, image_height, label_height, header_height = 320, 190, 26, 36
    panel_height = image_height + label_height
    rows = int(np.ceil(len(layers) / columns))
    canvas = Image.new(
        "RGB", (columns * panel_width, header_height + rows * panel_height), "black"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 9),
        f"{heading}  |  UI frame {ui_frame}  |  fixed temporal scaling",
        fill="white",
        font=font,
    )
    for index, layer in enumerate(layers):
        row, column = divmod(index, columns)
        x, y = column * panel_width, header_height + row * panel_height
        panel = _rgb(layer.values[frame_index], layer, panel_width, image_height)
        canvas.paste(Image.fromarray(panel), (x, y + label_height))
        draw.text((x + 6, y + 7), layer.title, fill="white", font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _render_video(
    path: Path,
    layers: list[Layer],
    heading: str,
    *,
    review_start_ui: int,
    fps: float,
    columns: int,
) -> dict[str, object]:
    prepared = [
        Layer(layer.title, layer.values, layer.kind, _sample_limits(layer.values, layer.kind))
        for layer in layers
    ]
    first = _compose(prepared, 0, review_start_ui, heading, columns=columns)
    height, width = first.shape[:2]
    temporary = path.with_suffix(".partial.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(float(fps)), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    started = time.monotonic()
    try:
        for frame_index in range(len(prepared[0].values)):
            frame = (
                first
                if frame_index == 0
                else _compose(
                    prepared,
                    frame_index,
                    review_start_ui + frame_index,
                    heading,
                    columns=columns,
                )
            )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            if (frame_index + 1) % 50 == 0 or frame_index + 1 == len(prepared[0].values):
                print(
                    f"{path.name}: {frame_index + 1}/{len(prepared[0].values)} frames",
                    flush=True,
                )
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        return_code = process.wait()
    except Exception:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr[-2000:]}")
    temporary.replace(path)
    return {
        "path": path.name,
        "frames": len(prepared[0].values),
        "fps": float(fps),
        "duration_seconds": len(prepared[0].values) / float(fps),
        "width": width,
        "height": height,
        "runtime_seconds": time.monotonic() - started,
        "layers": [layer.title for layer in prepared],
        "scaling": {
            layer.title: None if layer.limits is None else list(layer.limits)
            for layer in prepared
        },
    }


def render(run_root: Path, output_root: Path, *, fps: float = 10.0) -> dict[str, object]:
    run = run_root.resolve()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError("source run is not complete")
    config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    review_start, review_stop = config["source"]["review_interval_ui"]
    raw_source = np.load(config["source"]["movie_path"], mmap_mode="r")
    raw = raw_source[review_start - 1:review_stop]
    contexts = [
        "spatial_5_meanstd", "spatial_7_meanstd", "spatial_15_meanstd",
        "temporal_5_meanstd", "temporal_15_meanstd", "temporal_31_meanstd",
        "st_t15_s5_meanstd", "st_t15_s7_meanstd",
    ]
    names = {
        "spatial_5_meanstd": "Spatial 5", "spatial_7_meanstd": "Spatial 7",
        "spatial_15_meanstd": "Spatial 15", "temporal_5_meanstd": "Temporal 5",
        "temporal_15_meanstd": "Temporal 15", "temporal_31_meanstd": "Temporal 31",
        "st_t15_s5_meanstd": "T15 -> S5", "st_t15_s7_meanstd": "T15 -> S7",
    }
    load = lambda relative: np.load(run / relative, mmap_mode="r")
    banks = {
        "01_signed_msln_context_bank.mp4": (
            "Signed MSLN context bank",
            [Layer("Raw amplitude (authority)", raw, "raw")]
            + [Layer(names[item], load(f"features/msln/{item}.npy"), "signed") for item in contexts],
            3,
        ),
        "02_ica_innovation_context_bank.mp4": (
            "Canonical ICA innovation context bank",
            [Layer("Raw amplitude (authority)", raw, "raw")]
            + [Layer(names[item], load(f"features/per_context_ica/{item}_innovation.npy"), "signed") for item in contexts],
            3,
        ),
        "03_quiet_tail_context_bank.mp4": (
            "Quiet-tail activity evidence context bank",
            [Layer("Raw amplitude (authority)", raw, "raw")]
            + [Layer(names[item], load(f"features/energy/{item}_quiet_tail.npy"), "energy") for item in contexts],
            3,
        ),
        "04_group_routing_and_interaction.mp4": (
            "Group energy, routing, and display interaction",
            [
                Layer("Raw amplitude (authority)", raw, "raw"),
                Layer("Compact group tail", load("features/energy/compact_spatial_quiet_tail_group_energy.npy"), "energy"),
                Layer("Causal group tail", load("features/energy/causal_dynamic_quiet_tail_group_energy.npy"), "energy"),
                Layer("Broad group tail", load("features/energy/broad_context_quiet_tail_group_energy.npy"), "energy"),
                Layer("Route: max", load("features/routing/max.npy"), "energy"),
                Layer("Route: compact agreement", load("features/routing/compact_agreement.npy"), "energy"),
                Layer("Route: compact - broad", load("features/routing/compact_minus_broad.npy"), "energy"),
                Layer("Route: softmax", load("features/routing/softmax.npy"), "energy"),
                Layer("Final activity evidence", load("features/routing/activity_evidence.npy"), "energy"),
                Layer("Dominant context (0-7)", load("features/routing/dominant_context.npy"), "dominant"),
                Layer("Raw x gate (display only)", load("features/interactions/raw_times_activity_gate_display_only.npy"), "interaction"),
            ],
            4,
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_run": str(run),
        "review_interval_ui": [review_start, review_stop],
        "representation_contract": (
            "Raw is authoritative; signed context/ICA maps, nonnegative evidence, "
            "categorical dominant context, and display-only interaction are labeled."
        ),
        "videos": [],
    }
    try:
        for filename, (heading, layers, columns) in banks.items():
            manifest["videos"].append(
                _render_video(
                    output / filename,
                    layers,
                    heading,
                    review_start_ui=review_start,
                    fps=fps,
                    columns=columns,
                )
            )
        (output / "video_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        links = "".join(
            f'<li><a href="{row["path"]}">{row["path"]}</a></li>'
            for row in manifest["videos"]
        )
        (output / "index.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>MSLN/MS-ICA videos</title>"
            "<style>body{font:17px system-ui;max-width:900px;margin:40px auto}li{margin:14px}</style>"
            "<h1>MSLN/MS-ICA full-review layer videos</h1>"
            "<p>UI frames 1800–2359 at 10 fps. Scaling is fixed across time within each panel.</p>"
            f"<ol>{links}</ol>",
            encoding="utf-8",
        )
    except Exception:
        (output / "status.json").write_text(
            json.dumps({"status": "partial"}, indent=2) + "\n", encoding="utf-8"
        )
        raise
    (output / "status.json").write_text(
        json.dumps({"status": "complete", "video_count": 4}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    payload = render(args.run_root, args.output_root, fps=args.fps)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
