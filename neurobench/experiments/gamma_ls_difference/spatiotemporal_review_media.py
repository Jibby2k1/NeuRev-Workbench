"""Raw-only review media and geometry projection; never fills human annotation.

Consumes the frozen real_review package and source bindings. Scientific master
videos preserve each generated RGB frame exactly. Browser proxies are explicitly
display-only; neither the display transform nor these media establish event
truth, real precision, review acceptance, or completion of the model audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return dict(path=str(path), sha256=digest(path), size_bytes=path.stat().st_size)


def verify(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if binding(path) != record:
        raise ValueError(f"Source binding changed: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def table(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    if result.stderr.strip():
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)["streams"][0]


def _read_exact(handle, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = handle.read(size - len(result))
        if not block:
            break
        result.extend(block)
    return bytes(result)


def _decode(path: Path, frame_hashes: list[str] | None, *, shape: tuple[int, int], frames: int) -> dict[str, Any]:
    height, width = shape
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"]
    stream_digest = hashlib.sha256()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        assert process.stdout is not None and process.stderr is not None
        for index in range(frames):
            frame = _read_exact(process.stdout, height * width * 3)
            if len(frame) != height * width * 3:
                process.kill()
                raise RuntimeError(f"Incomplete decoded frame {index}: {path}")
            if frame_hashes is not None and hashlib.sha256(frame).hexdigest() != frame_hashes[index]:
                process.kill()
                raise RuntimeError(f"Master RGB bytes changed at decoded frame {index}: {path}")
            stream_digest.update(frame)
        extra = process.stdout.read(1)
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0 or extra or error.strip():
            raise RuntimeError(f"Decode cardinality/error failure: {path}: {error}")
    metadata = probe(path)
    if (int(metadata["width"]), int(metadata["height"]), int(metadata["nb_frames"])) != (width, height, frames):
        raise RuntimeError(f"Video geometry or frame count mismatch: {path}")
    if metadata["avg_frame_rate"] != "50/1":
        raise RuntimeError(f"Video is not native 50 Hz: {path}")
    return {"full_decode_pass": True, "decoded_frame_count": frames,
            "decoded_rgb_stream_sha256": stream_digest.hexdigest(),
            "all_frames_rgb_byte_exact": frame_hashes is not None,
            "probe": metadata, "decode_command": command}


def render(review_root: Path, *, media_name: str = "raw_media_v1") -> Path:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    review_root = review_root.resolve()
    manifest_path = review_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["contract"]
    output = review_root / media_name
    if output.exists():
        raise FileExistsError(f"Preserve existing media: {output}")
    for record in manifest["immutable_artifacts"]:
        verify(record)
    movie_path = verify(contract["source_movie"])
    # Freeze the annotation files' initial bytes only to prove this renderer did
    # not change them. These are not accepted annotation or negative truth.
    annotation_before = {name: binding(review_root / name) for name in manifest["mutable_human_review_files"]}
    acceptance_before = binding(review_root / "annotation_acceptance.json")
    movie = np.load(movie_path, mmap_mode="r")
    if list(movie.shape) != contract["source_shape_tyx"]:
        raise ValueError("Movie geometry disagrees with frozen review contract")
    panels = rows(review_root / "panel_geometry.tsv")
    grid = rows(review_root / "frame_grid.tsv")
    experts = rows(review_root / "expert_occurrences.tsv")
    if len(panels) != 3 or len(experts) != 79:
        raise ValueError("Expected three fixed panels and 79 source occurrence rows")
    first, last = map(int, contract["evaluation_ui"])
    frames = list(range(first, last + 1))
    if frames != list(range(1800, 2360)):
        raise ValueError("Unexpected review source-frame inventory")
    setup_first, setup_last = map(int, contract["detector_setup_ui"])
    samples = []
    for panel in panels:
        x0, y0, x1, y1 = (int(panel[key]) for key in ("x0", "y0", "x1", "y1"))
        expected = [r for r in grid if r["panel_id"] == panel["panel_id"]]
        if [int(r["source_frame_ui"]) for r in expected] != frames or any(
            int(r["source_numpy_index"]) != int(r["source_frame_ui"]) - 1
            or [int(r[k]) for k in ("x0", "y0", "x1", "y1")] != [x0, y0, x1, y1]
            for r in expected
        ):
            raise ValueError("Frame-grid and panel coordinate bindings disagree")
        if (x1 - x0, y1 - y0) != (128, 128):
            raise ValueError("Expected native128x128 review panel")
        samples.append(np.asarray(movie[setup_first - 1:setup_last, y0:y1, x0:x1]).ravel())
    calibration = np.concatenate(samples)
    lo, hi = map(float, np.quantile(calibration, [.005, .999]))
    del calibration, samples
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Invalid calibration-only fixed display limits")
    output.mkdir(parents=True)
    masters = output / "masters"
    previews = output / "browser"
    masters.mkdir()
    previews.mkdir()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 16)
    small_font = ImageFont.truetype(str(font_path), 13)
    videos, display_rows = [], []
    height, width, scale = 488, 512, 3
    for panel in panels:
        name = panel["panel_id"]
        x0, y0, x1, y1 = (int(panel[key]) for key in ("x0", "y0", "x1", "y1"))
        path = masters / f"{name}_raw_50Hz_lossless.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "50", "-i", "-",
                   "-an", "-c:v", "libx264rgb", "-threads", "1", "-preset", "veryfast",
                   "-crf", "0", "-pix_fmt", "rgb24", "-movflags", "+faststart", str(path)]
        hashes, source_stream = [], hashlib.sha256()
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            assert process.stdin is not None and process.stderr is not None
            for frame in frames:
                raw = np.asarray(movie[frame - 1, y0:y1, x0:x1], dtype=np.float64)
                display = np.floor(np.clip((raw - lo) / (hi - lo), 0, 1) * 255 + .5).astype(np.uint8)
                canvas = Image.new("RGB", (width, height), "black")
                canvas.paste(Image.fromarray(display).resize((384, 384), Image.Resampling.NEAREST), (64, 80))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 8), f"{name} | RAW fluorescence | 50 Hz", font=font, fill="white")
                draw.text((16, 31), f"Source UI {frame} | {(frame - 1) / 50:.2f} s | review {(frame-first)/50:.2f} s", font=font, fill="white")
                draw.text((16, 56), f"Global x=[{x0},{x1}), y=[{y0},{y1}) | nearest 3x pixels", font=small_font, fill="white")
                draw.text((16, 468), f"Fixed setup display [{lo:.1f}, {hi:.1f}] | no detection overlays", font=small_font, fill="white")
                rgb = np.asarray(canvas)
                if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 1], rgb[..., 2]):
                    raise ValueError("A raw review frame contains a colored overlay")
                payload = rgb.tobytes()
                hashes.append(hashlib.sha256(payload).hexdigest())
                source_stream.update(payload)
                process.stdin.write(payload)
                display_rows.append(dict(panel_id=name, source_frame_ui=frame,
                    source_numpy_index=frame-1, video_frame_index=frame-first,
                    below_display_limit_pixel_count=int(np.count_nonzero(raw < lo)),
                    above_display_limit_pixel_count=int(np.count_nonzero(raw > hi)),
                    display_rgb_sha256=hashes[-1]))
                if frame in (first, 2014):
                    canvas.save(output / f"{name}_ui{frame}.png")
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0 or error.strip():
                raise RuntimeError(error)
        checked = _decode(path, hashes, shape=(height, width), frames=len(frames))
        if checked["decoded_rgb_stream_sha256"] != source_stream.hexdigest():
            raise RuntimeError("Master full-stream source and decoded hashes disagree")
        proxy = previews / f"{name}_raw_50Hz_browser.mp4"
        proxy_command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-an",
                         "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "18",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(proxy)]
        subprocess.run(proxy_command, check=True, capture_output=True)
        proxy_checked = _decode(proxy, None, shape=(height, width), frames=len(frames))
        videos.append(dict(panel_id=name, source_frames_ui=frames, source_shape_yx=[128, 128],
                           display_shape_yx=[height, width], source_box_xyxy=[x0, y0, x1, y1],
                           display_raw_origin_xy=[64, 80], integer_nearest_scale=scale,
                           master={**binding(path), **checked},
                           browser_proxy={**binding(proxy), **proxy_checked, "authority": "display_only_lossy_proxy"}))
        print(json.dumps(dict(status="RAW_PANEL_MEDIA_VALIDATED", panel_id=name, frames=len(frames))), flush=True)

    projection_start, projection_stop = map(int, contract["conditioning_history_warmup_ui"])
    if [projection_start, projection_stop] != [1400, 1599]:
        raise ValueError("Unexpected geometry projection frame interval")
    total = np.zeros(movie.shape[1:], dtype=np.float64)
    for frame in range(projection_start, projection_stop + 1):
        total += movie[frame - 1]
    mean = total / (projection_stop - projection_start + 1)
    np.save(output / "geometry_mean_raw_ui1400_1599.npy", mean)
    protocol_path = review_root.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    halo = int(protocol["common_spatial_halo_px"])
    if halo != 49:
        raise ValueError("Requested common49px geometry does not match the frozen protocol")
    fig, axis = plt.subplots(figsize=(12, 8))
    vmin, vmax = np.quantile(mean, [.005, .999])
    axis.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    axis.scatter([float(r["x_px"]) for r in experts], [float(r["y_px"]) for r in experts],
                 s=38, facecolors="none", edgecolors="#46dc7d", linewidths=1.0,
                 label="79 known occurrence points (repeated coordinates overlap)")
    axis.add_patch(Rectangle((halo-.5, halo-.5), movie.shape[2]-2*halo, movie.shape[1]-2*halo,
                             fill=False, edgecolor="white", linestyle="--", linewidth=1.3,
                             label="Common scored interior: 49 px source halo"))
    for index, panel in enumerate(panels):
        x0, y0, x1, y1 = (int(panel[key]) for key in ("x0", "y0", "x1", "y1"))
        axis.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False,
                                 edgecolor="#60d6ff", linewidth=1.6,
                                 label="Three fixed review panels" if index == 0 else None))
        axis.text((x0+x1-1)/2, y0-7, panel["panel_id"], color="#60d6ff", ha="center", fontsize=10,
                  bbox=dict(facecolor="black", alpha=.6, edgecolor="none", pad=2))
    axis.set(xlabel="Global x (column)", ylabel="Global y (row)",
             title="Geometry projection only | mean raw UI 1400–1599\nAll 79 sparse occurrences; geometry-selected review panels")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.10), fontsize=9, frameon=False)
    fig.text(.5, .018, "No detection overlays or label-dependent fitting. Sparse points do not establish exhaustive truth.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.18)
    projection_path = output / "geometry_projection.png"
    fig.savefig(projection_path, dpi=160)
    plt.close(fig)
    table(output / "display_frame_inventory.tsv", display_rows)
    intro = (
        "<h1>Raw fluorescence review — annotations pending</h1>"
        "<p>All three geometry-selected panels cover source UI1800–2359 at native50Hz (11.2s). "
        "Videos contain raw fluorescence only. Human review and acceptance remain pending; real precision is unavailable.</p>"
        f"<p>One grayscale transform is fixed across all frames and panels: [{lo:.3f}, {hi:.3f}], from the pooled "
        "0.5th and99.9th percentiles of raw panels during UI1600–1799. Integer3× nearest-pixel display; "
        "clipping counts and exact source frames are in the display inventory. Browser copies are lossy display proxies; "
        "the linked RGB-lossless masters have passed byte-exact decoding of every frame.</p>"
        "<p>Global point coordinates are x=column,y=row. Polygon image-edge coordinates place pixel centers at "
        "column+0.5,row+0.5. Clip crops and source frame IDs are burned into the margin. "
        "No detector output is shown. Use the frame buttons to pause and inspect individual source frames.</p>"
    )
    cards = []
    for panel in panels:
        name = html.escape(panel["panel_id"])
        cards.append(f'<section><h2>{name}</h2><video id="{name}" controls preload="metadata" poster="{name}_ui1800.png" '
                     f'src="browser/{name}_raw_50Hz_browser.mp4"></video><div><button onclick="step(\'{name}\',-1)">Previous frame</button> '
                     f'<button onclick="step(\'{name}\',1)">Next frame</button> <label>Source UI <input type="number" min="1800" max="2359" value="1800" '
                     f'onchange="seek(\'{name}\',this.value)"></label></div><p><a href="masters/{name}_raw_50Hz_lossless.mp4">RGB-lossless master</a></p></section>')
    links = ["REVIEW_INSTRUCTIONS.md", "annotation_schema.json", "frame_grid.tsv", *manifest["mutable_human_review_files"]]
    markup = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Raw panel review</title><style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.5 system-ui;background:#f5f5f3;color:#222}section{display:inline-block;vertical-align:top;width:510px;margin:12px 20px 18px 0}video{width:100%;background:black}button,input{font:inherit;padding:4px}input{width:90px}img{max-width:100%}a{color:#075c85}</style>'
    markup += intro + "".join(cards) + '<h2>Geometry check</h2><img src="geometry_projection.png" alt="Full-field raw mean, known points and fixed review boxes">'
    markup += "<h2>Review files</h2><ul>" + "".join(f'<li><a href="../{name}">{name}</a></li>' for name in links) + '</ul><p><a href="display_frame_inventory.tsv">Display/source frame inventory</a> · <a href="validation.json">Media validation</a></p>'
    markup += '<script>function seek(id,ui){let v=document.getElementById(id);v.pause();v.currentTime=(Math.max(1800,Math.min(2359,Number(ui)))-1800+.1)/50;}function step(id,d){let v=document.getElementById(id);v.pause();let f=Math.floor(v.currentTime*50+1e-5);v.currentTime=(Math.max(0,Math.min(559,f+d))+.1)/50;}</script></html>'
    (output / "index.html").write_text(markup)
    (output / "README.md").write_text(
        "# Raw review media\n\nOpen index.html for the three native50Hz raw clips. Browser files are display-only proxies; "
        "masters preserve every generated grayscale/caption RGB frame byte-exactly. Their intensity transform is a display transform, "
        "not the original uint16 fluorescence array. The source NPY remains the quantitative authority.\n\n"
        "Geometry projection uses raw UI1400–1599 and the79 existing occurrence coordinates. Repeated coordinates overlap. "
        "This is a geometry check only, without label-dependent fitting. It does not imply active fluorescence in the projection frames.\n\n"
        "No annotation file or acceptance record was changed. Review, exhaustive annotation, real precision and the parent model audit remain pending.\n")
    for record in annotation_before.values():
        verify(record)
    verify(acceptance_before)
    provenance = {
        "schema": "gamma_st_raw_real_review_media_v1", "source_manifest": binding(manifest_path),
        "source_movie": contract["source_movie"], "source_movie_fresh_hash_verified": True,
        "frozen_immutable_inputs": manifest["immutable_artifacts"], "parent_protocol": binding(protocol_path),
        "generator": binding(Path(__file__)), "font": binding(font_path),
        "display": {"source": "unconditioned raw fluorescence", "setup_ui": [setup_first, setup_last],
                    "quantiles": [.005, .999], "limits": [lo, hi], "common_scale_across_all_panels": True,
                    "transform": "floor(clip((raw-lo)/(hi-lo),0,1)*255+0.5)",
                    "resize": "integer3x nearest-neighbor, no smoothing", "detector_overlays": False},
        "projection": {"source_ui": [projection_start, projection_stop], "known_occurrence_points": 79,
                       "unique_coordinate_count": len({(r["x_px"],r["y_px"]) for r in experts}),
                       "purpose": "geometry validation only", "labels_used_to_fit": False,
                       "common_source_halo_px": halo, "panels": panels},
        "annotation_file_bindings_before_and_after": annotation_before,
        "acceptance_unchanged": acceptance_before,
        "review_status": "PREPARED_ANNOTATION_PENDING", "real_precision": None,
        "scientific_audit_complete": False, "videos": videos,
    }
    write_json(output / "provenance.json", provenance)
    artifacts = [binding(path) for path in sorted(output.rglob("*")) if path.is_file()]
    validation = {"status": "PASS", "scope": "raw review media encoding, geometry bindings and full decode only",
                  "master_count": 3, "browser_proxy_count": 3, "frames_per_video": 560,
                  "master_total_byte_exact_frames": 1680, "all_six_videos_fully_decoded": True,
                  "display_dimensions_wh": [width,height], "frame_rate_hz": 50,
                  "annotation_files_unchanged": True, "annotation_accepted": False,
                  "geometry_projection_visual_qa": "pending_separate_visual_inspection",
                  "review_complete": False, "real_precision": None, "artifacts": artifacts}
    write_json(output / "validation.json", validation)
    print(json.dumps(dict(status="RAW_REVIEW_MEDIA_PASS", output=str(output))), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("review_root", type=Path)
    parser.add_argument("--media-name", default="raw_media_v1")
    args = parser.parse_args()
    render(args.review_root, media_name=args.media_name)


if __name__ == "__main__":
    main()
