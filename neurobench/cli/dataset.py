"""Dataset-related CLI commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurobench.data.catalog import discover_dataset_catalog, llm_catalog_context, query_dataset_catalog
from neurobench.data.intake import build_dataset_intake_manifest, dataset_intake_report
from neurobench.data.qc import compute_dataset_qc_from_manifest, render_dataset_qc_markdown
from neurobench.manifests import write_json
from neurobench.validation.schemas import validate_json, validation_error_summary


def add_dataset_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "dataset",
        help="Create, validate, and inspect dataset manifests.",
        description="Create, validate, and inspect dataset manifests.",
    )
    dataset_subparsers = parser.add_subparsers(dest="dataset_command", metavar="dataset-command")
    validate_parser = dataset_subparsers.add_parser("validate", help="Validate a dataset manifest JSON file.")
    validate_parser.add_argument("manifest", type=Path, help="Path to a dataset manifest JSON file.")
    validate_parser.set_defaults(func=validate_dataset_command)

    qc_parser = dataset_subparsers.add_parser("qc", help="Generate a dataset QC JSON and Markdown report.")
    qc_parser.add_argument("manifest", type=Path, help="Path to a dataset manifest JSON file.")
    qc_parser.add_argument("--output", required=True, type=Path, help="Output directory for qc_report.json and qc_report.md.")
    qc_parser.set_defaults(func=dataset_qc_command)

    intake_parser = dataset_subparsers.add_parser("intake", help="Create and check a metadata-only dataset intake manifest.")
    intake_parser.add_argument("--dataset-id", required=True)
    intake_parser.add_argument("--raw-video", required=True)
    intake_parser.add_argument("--out", required=True, type=Path)
    intake_parser.add_argument("--app-dir", type=Path, default=None)
    intake_parser.add_argument("--frame-rate-hz", type=float, default=None)
    intake_parser.add_argument("--pixel-size-microns", type=float, default=None)
    intake_parser.add_argument("--source-template", choices=["local", "dandi-nwb", "janelia-figshare"], default="local")
    intake_parser.add_argument("--name", default=None)
    intake_parser.add_argument("--modality", default="light_sheet_calcium")
    intake_parser.add_argument("--indicator", default="GCaMP")
    intake_parser.add_argument("--report-out", type=Path, default=None)
    intake_parser.set_defaults(func=dataset_intake_command)

    catalog_parser = dataset_subparsers.add_parser(
        "catalog",
        help="Discover and query review-ready datasets through one bounded catalog.",
    )
    catalog_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root containing Outputs/.")
    catalog_parser.add_argument("--search-root", action="append", default=None, help="Root under --root to scan; repeatable.")
    catalog_parser.add_argument("--max-depth", type=int, default=4, help="Maximum directory depth per search root.")
    catalog_parser.add_argument("--query", default="", help="Filter by dataset id, name, or declared path.")
    catalog_parser.add_argument("--llm", action="store_true", help="Emit the compact LLM-facing catalog contract.")
    catalog_parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact table.")
    catalog_parser.add_argument("--out", type=Path, default=None, help="Optionally write the JSON payload.")
    catalog_parser.set_defaults(func=dataset_catalog_command)
    return parser


def add_validate_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate",
        help="Validate public Neurobench artifacts.",
        description="Validate public Neurobench artifacts.",
    )
    validate_subparsers = parser.add_subparsers(dest="validate_command", metavar="artifact")
    dataset_parser = validate_subparsers.add_parser("dataset", help="Validate a dataset manifest JSON file.")
    dataset_parser.add_argument("manifest", type=Path, help="Path to a dataset manifest JSON file.")
    dataset_parser.set_defaults(func=validate_dataset_command)
    return parser


def validate_dataset_command(args: argparse.Namespace) -> int:
    try:
        payload = validate_json(args.manifest, "dataset")
    except Exception as exc:  # pragma: no cover - exact exception type is tested via subprocess behavior.
        print(f"Dataset manifest validation failed: {args.manifest}", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    dataset_id = payload.get("dataset_id", "(unknown)")
    print(f"Validated dataset manifest: {args.manifest} ({dataset_id})")
    return 0



def dataset_intake_command(args: argparse.Namespace) -> int:
    try:
        manifest = build_dataset_intake_manifest(
            dataset_id=args.dataset_id,
            raw_video=args.raw_video,
            app_dir=args.app_dir,
            frame_rate_hz=args.frame_rate_hz,
            pixel_size_microns=args.pixel_size_microns,
            source_template=args.source_template,
            name=args.name,
            modality=args.modality,
            indicator=args.indicator,
        )
        write_json(args.out, manifest)
        report = dataset_intake_report(manifest, base_dir=Path.cwd())
        if args.report_out:
            write_json(args.report_out, report)
    except Exception as exc:
        print("Dataset intake failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Dataset intake manifest: {args.out}")
    if args.report_out:
        print(f"Dataset intake report: {args.report_out}")
    print(f"ready: {'yes' if report.get('ready') else 'no'}")
    for check in report.get("checks", []):
        print(f"{check['status']}: {check['name']} - {check['detail']}")
    return 0

def dataset_qc_command(args: argparse.Namespace) -> int:
    try:
        validate_json(args.manifest, "dataset")
        qc = compute_dataset_qc_from_manifest(args.manifest)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "qc_report.json"
        markdown_path = out_dir / "qc_report.md"
        write_json(json_path, qc)
        markdown_path.write_text(render_dataset_qc_markdown(qc), encoding="utf-8")
    except Exception as exc:
        print(f"Dataset QC failed: {args.manifest}", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Dataset QC JSON: {json_path}")
    print(f"Dataset QC Markdown: {markdown_path}")
    print(f"warnings: {len(qc.get('warnings') or [])}")
    return 0


def dataset_catalog_command(args: argparse.Namespace) -> int:
    try:
        records = discover_dataset_catalog(
            args.root,
            search_roots=args.search_root,
            max_depth=args.max_depth,
        )
        records = query_dataset_catalog(records, args.query)
        payload: dict | list = llm_catalog_context(records) if args.llm else records
        if args.out:
            write_json(args.out, payload if isinstance(payload, dict) else {"schema_version": 1, "datasets": payload})
    except Exception as exc:
        print("Dataset catalog query failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    if args.json or args.llm:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print("dataset_id\tframes\tsize\treview\tCFAR\traw_video")
    for record in records:
        video = record.get("video") or {}
        paths = record.get("paths") or {}
        capabilities = record.get("capabilities") or {}
        size = f"{video.get('width', '?')}x{video.get('height', '?')}"
        print(
            f"{record.get('dataset_id', '')}\t{video.get('frames', '')}\t{size}\t"
            f"{'yes' if capabilities.get('review_app') else 'no'}\t"
            f"{'yes' if capabilities.get('cfar_annotation') else 'no'}\t{paths.get('raw_video', '')}"
        )
    return 0
