"""Grid generation and extraction CLI commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurobench.algorithms.grid_regions import write_grid_spec_artifacts, write_grid_state_artifacts, write_registered_grid_state_artifacts
from neurobench.manifests import load_json
from neurobench.validation.schemas import validation_error_summary


def add_grid_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("grid", help="Generate template-coordinate grids and grid states.")
    grid_subparsers = parser.add_subparsers(dest="grid_command", metavar="grid-command")
    generate = grid_subparsers.add_parser("generate", help="Generate a rectangular grid spec.")
    generate.add_argument("--template", required=True, type=Path)
    generate.add_argument("--rows", type=int, default=32)
    generate.add_argument("--cols", type=int, default=32)
    generate.add_argument("--out", required=True, type=Path)
    generate.set_defaults(func=grid_generate_command)

    extract = grid_subparsers.add_parser("extract-states", help="Pool registered videos into grid states.")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--registered-dir", required=True, type=Path)
    extract.add_argument("--grid", required=True, type=Path)
    extract.add_argument("--features", nargs="+", default=["mean_intensity"])
    extract.add_argument("--normalization", default="per_video_robust_percentile")
    extract.add_argument("--chunk-size-frames", type=int, default=64)
    extract.add_argument("--max-grid-state-bytes", type=int, default=1_000_000_000)
    extract.add_argument("--out-dir", required=True, type=Path)
    extract.set_defaults(func=grid_extract_command)

    streaming = grid_subparsers.add_parser(
        "extract-registered-states",
        help="Apply registration in memory and pool raw videos into grid states without registered_video.npy files.",
    )
    streaming.add_argument("--manifest", required=True, type=Path)
    streaming.add_argument("--registration-dir", required=True, type=Path)
    streaming.add_argument("--grid", required=True, type=Path)
    streaming.add_argument("--features", nargs="+", default=["mean_intensity"])
    streaming.add_argument("--normalization", default="per_video_robust_percentile")
    streaming.add_argument("--chunk-size-frames", type=int, default=64)
    streaming.add_argument("--max-grid-state-bytes", type=int, default=1_000_000_000)
    streaming.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    streaming.add_argument("--video-id", action="append", default=[], help="Limit extraction to one video ID; repeat for multiple IDs.")
    streaming.add_argument("--out-dir", required=True, type=Path)
    streaming.set_defaults(func=grid_extract_registered_command)
    return parser


def grid_generate_command(args: argparse.Namespace) -> int:
    try:
        spec = write_grid_spec_artifacts(template_spec=load_json(args.template), out_path=args.out, rows=args.rows, cols=args.cols)
    except Exception as exc:
        print("Grid spec generation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Grid spec: {args.out}")
    print(f"regions: {spec['region_count']}")
    return 0


def grid_extract_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(args.manifest)
        grid = load_json(args.grid)
        summaries = []
        for video in manifest.get("videos", []) or []:
            video_id = str(video["video_id"])
            registered = Path(args.registered_dir) / video_id / "registered_video.npy"
            summaries.append(
                write_grid_state_artifacts(
                    registered_video_path=registered,
                    grid_spec=grid,
                    out_dir=args.out_dir,
                    video_id=video_id,
                    label=str(video.get("label") or ""),
                    features=args.features,
                    normalization=args.normalization,
                    frame_rate_hz=video.get("frame_rate_hz"),
                    chunk_size_frames=args.chunk_size_frames,
                    max_grid_state_bytes=args.max_grid_state_bytes,
                )
            )
    except Exception as exc:
        print("Grid state extraction failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Grid states dir: {args.out_dir}")
    print(f"videos: {len(summaries)}")
    return 0


def grid_extract_registered_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(args.manifest)
        grid = load_json(args.grid)
        requested = {str(value) for value in args.video_id or []}
        videos = [video for video in manifest.get("videos", []) or [] if not requested or str(video.get("video_id")) in requested]
        if requested and len(videos) != len(requested):
            found = {str(video.get("video_id")) for video in videos}
            missing = sorted(requested - found)
            raise ValueError(f"video_id not found in manifest: {', '.join(missing)}")
        summaries = []
        for video in videos:
            video_id = str(video["video_id"])
            registration = load_json(Path(args.registration_dir) / video_id / "registration_result.json")
            summaries.append(
                write_registered_grid_state_artifacts(
                    video_path=video["path"],
                    registration_result=registration,
                    grid_spec=grid,
                    out_dir=args.out_dir,
                    video_id=video_id,
                    label=str(video.get("label") or ""),
                    features=args.features,
                    normalization=args.normalization,
                    frame_rate_hz=video.get("frame_rate_hz"),
                    chunk_size_frames=args.chunk_size_frames,
                    max_grid_state_bytes=args.max_grid_state_bytes,
                    device=args.device,
                )
            )
    except Exception as exc:
        print("Streaming grid state extraction failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    devices = sorted({summary.get("extras", {}).get("device", "unknown") for summary in summaries})
    print(f"Grid states dir: {args.out_dir}")
    print(f"videos: {len(summaries)}")
    print(f"devices: {', '.join(devices) if devices else 'none'}")
    return 0

