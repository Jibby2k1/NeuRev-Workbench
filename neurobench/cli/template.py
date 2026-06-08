"""Template construction and registration CLI commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurobench.algorithms.template_matching import write_registered_video_artifacts, write_registration_artifacts, write_template_artifacts
from neurobench.data.video_manifest import video_by_id
from neurobench.manifests import load_json
from neurobench.validation.schemas import validation_error_summary


def add_template_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("template", help="Build and apply anatomical video templates.")
    template_subparsers = parser.add_subparsers(dest="template_command", metavar="template-command")

    build = template_subparsers.add_parser("build-from-video", help="Build a template from one reference video.")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--reference-video-id", required=True)
    build.add_argument("--projection-kind", default="mean_after_outlier_rejection")
    build.add_argument("--max-outlier-fraction", type=float, default=0.05)
    build.add_argument("--z-threshold", type=float, default=3.5)
    build.add_argument("--chunk-size-frames", type=int, default=64)
    build.add_argument("--disable-outlier-rejection", action="store_true")
    build.add_argument("--out-dir", required=True, type=Path)
    build.set_defaults(func=template_build_command)

    register = template_subparsers.add_parser("register-videos", help="Register manifest videos into template coordinates.")
    register.add_argument("--manifest", required=True, type=Path)
    register.add_argument("--template", required=True, type=Path)
    register.add_argument("--transform-model", choices=["translation", "rigid", "similarity"], default="rigid")
    register.add_argument("--rotation-range-deg", type=float, nargs=2, default=(-10.0, 10.0))
    register.add_argument("--rotation-step-deg", type=float, default=0.5)
    register.add_argument("--allow-uniform-scale", action="store_true")
    register.add_argument("--out-dir", required=True, type=Path)
    register.add_argument("--chunk-size-frames", type=int, default=64)
    register.set_defaults(func=template_register_command)

    apply = template_subparsers.add_parser("apply-registration", help="Apply per-video registration transforms.")
    apply.add_argument("--manifest", required=True, type=Path)
    apply.add_argument("--template", required=True, type=Path)
    apply.add_argument("--registration-dir", required=True, type=Path)
    apply.add_argument("--out-dir", required=True, type=Path)
    apply.add_argument("--output-dtype", default="float32")
    apply.add_argument("--chunk-size-frames", type=int, default=64)
    apply.set_defaults(func=template_apply_command)
    return parser


def template_build_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(args.manifest)
        video = video_by_id(manifest, args.reference_video_id)
        spec = write_template_artifacts(
            video_path=video["path"],
            source_video_id=args.reference_video_id,
            out_dir=args.out_dir,
            outlier_rejection=not args.disable_outlier_rejection,
            max_outlier_fraction=args.max_outlier_fraction,
            z_threshold=args.z_threshold,
            chunk_size_frames=args.chunk_size_frames,
        )
    except Exception as exc:
        print("Template build failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Template spec: {Path(args.out_dir) / 'template_spec.json'}")
    print(f"removed frames: {len(spec.get('outlier_rejection', {}).get('removed_frame_indices') or [])}")
    return 0


def template_register_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(args.manifest)
        template = load_json(args.template)
        results = []
        for video in manifest.get("videos", []) or []:
            results.append(
                write_registration_artifacts(
                    video_path=video["path"],
                    video_id=str(video["video_id"]),
                    template_spec=template,
                    out_dir=args.out_dir,
                    transform_model=args.transform_model,
                    rotation_range_deg=(float(args.rotation_range_deg[0]), float(args.rotation_range_deg[1])),
                    rotation_step_deg=args.rotation_step_deg,
                    allow_uniform_scale=args.allow_uniform_scale,
                    chunk_size_frames=args.chunk_size_frames,
                )
            )
    except Exception as exc:
        print("Template registration failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Registration dir: {args.out_dir}")
    print(f"registered videos: {len(results)}")
    print(f"warnings: {sum(len(r.get('qc', {}).get('warnings') or []) for r in results)}")
    return 0


def template_apply_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(args.manifest)
        template = load_json(args.template)
        summaries = []
        for video in manifest.get("videos", []) or []:
            result_path = Path(args.registration_dir) / str(video["video_id"]) / "registration_result.json"
            summaries.append(
                write_registered_video_artifacts(
                    video_path=video["path"],
                    registration_result=load_json(result_path),
                    template_spec=template,
                    out_dir=args.out_dir,
                    output_dtype=args.output_dtype,
                    chunk_size_frames=args.chunk_size_frames,
                )
            )
    except Exception as exc:
        print("Apply registration failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Registered videos dir: {args.out_dir}")
    print(f"videos: {len(summaries)}")
    return 0
