"""Video manifest CLI commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurobench.data.preflight import build_template_grid_preflight
from neurobench.data.video_manifest import DEFAULT_VIDEO_PATTERN, build_video_manifest
from neurobench.manifests import write_json
from neurobench.validation.schemas import validation_error_summary


def add_video_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("video", help="Build video manifests for template-grid workflows.")
    video_subparsers = parser.add_subparsers(dest="video_command", metavar="video-command")
    manifest = video_subparsers.add_parser("manifest", help="Parse labeled .tif/.tiff/.npy videos into a manifest.")
    manifest.add_argument("--input-dir", required=True, type=Path)
    manifest.add_argument("--pattern", default=DEFAULT_VIDEO_PATTERN)
    manifest.add_argument("--dataset-id", default="zebrafish_left_right_neutral_v1")
    manifest.add_argument("--labels", nargs="+", default=["left", "right", "neutral"])
    manifest.add_argument("--label-alias", action="append", default=[], metavar="RAW=CANONICAL")
    manifest.add_argument("--out", required=True, type=Path)
    manifest.add_argument("--frame-rate-hz", type=float, default=None)
    manifest.add_argument("--strict", action="store_true")
    manifest.add_argument("--json", action="store_true")
    manifest.set_defaults(func=video_manifest_command)

    preflight = video_subparsers.add_parser("preflight", help="Estimate template-grid real-data pilot disk/RAM requirements.")
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--out-dir", required=True, type=Path)
    preflight.add_argument("--out", required=True, type=Path)
    preflight.add_argument("--rows", type=int, default=32)
    preflight.add_argument("--cols", type=int, default=32)
    preflight.add_argument("--chunk-size-frames", type=int, default=64)
    preflight.add_argument("--registered-dtype", default="float32")
    preflight.add_argument("--expected-video-count", type=int, default=27)
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(func=video_preflight_command)
    return parser


def video_manifest_command(args: argparse.Namespace) -> int:
    try:
        payload = build_video_manifest(
            input_dir=args.input_dir,
            dataset_id=args.dataset_id,
            filename_regex=args.pattern,
            labels=args.labels,
            label_aliases=_parse_label_aliases(args.label_alias),
            frame_rate_hz=args.frame_rate_hz,
            strict=args.strict,
        )
        write_json(args.out, payload)
    except Exception as exc:
        print("Video manifest build failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Video manifest: {args.out}")
        print(f"videos: {len(payload.get('videos') or [])}")
        print(f"label_counts: {payload.get('label_counts')}")
        if payload.get("warnings"):
            print(f"warnings: {len(payload['warnings'])}")
    return 0


def video_preflight_command(args: argparse.Namespace) -> int:
    try:
        with args.manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        payload = build_template_grid_preflight(
            manifest=manifest,
            output_root=args.out_dir,
            rows=args.rows,
            cols=args.cols,
            chunk_size_frames=args.chunk_size_frames,
            registered_dtype=args.registered_dtype,
            expected_video_count=args.expected_video_count,
        )
        write_json(args.out, payload)
    except Exception as exc:
        print("Video preflight failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Video preflight: {args.out}")
        print(f"videos: {payload['video_count']}")
        print(f"estimated_output_total_bytes: {payload['estimated_output_total_bytes']}")
        print(f"warnings: {len(payload.get('warnings') or [])}")
    return 0



def _parse_label_aliases(items: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Label alias must be RAW=CANONICAL, got: {item}")
        raw, canonical = item.split("=", 1)
        raw = raw.strip()
        canonical = canonical.strip()
        if not raw or not canonical:
            raise ValueError(f"Label alias must be RAW=CANONICAL, got: {item}")
        aliases[raw] = canonical
    return aliases
