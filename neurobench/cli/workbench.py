"""Workbench CLI commands."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from neurobench.data.catalog import dataset_record_for_app, discover_dataset_catalog, query_dataset_catalog
from neurobench.workbench.intermediates import (
    add_attach_intermediates_arguments,
    add_export_intermediate_arguments,
    attach_intermediates_command,
    export_intermediate_command,
)


def _add_app_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app-dir", type=Path)
    group.add_argument("--dataset-id")
    parser.add_argument("--catalog-root", type=Path, default=Path.cwd())


def _catalog_app_dir(args: argparse.Namespace) -> Path:
    if args.app_dir:
        return args.app_dir.expanduser().resolve()
    records = discover_dataset_catalog(args.catalog_root)
    matches = [
        record
        for record in query_dataset_catalog(records, args.dataset_id)
        if str(record.get("dataset_id")) == str(args.dataset_id)
    ]
    if not matches:
        raise ValueError(f"Dataset '{args.dataset_id}' was not found under {args.catalog_root}")
    app_value = (matches[0].get("paths") or {}).get("app_dir")
    if not app_value:
        raise ValueError(f"Dataset '{args.dataset_id}' has no workbench app")
    path = Path(str(app_value)).expanduser()
    return path.resolve() if path.is_absolute() else (args.catalog_root / path).resolve()


def add_workbench_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "workbench",
        help="Build, serve, and prepare local review workbenches.",
        description="Build, serve, and prepare local review workbenches.",
    )
    workbench_subparsers = parser.add_subparsers(dest="workbench_command", metavar="workbench-command")

    build_parser = workbench_subparsers.add_parser(
        "build",
        help="Build or explicitly upgrade a review app while preserving annotations and attached runs.",
    )
    build_parser.add_argument("--app-dir", type=Path)
    build_parser.add_argument("--review-data", type=Path)
    build_parser.add_argument("--dataset-manifest", type=Path)
    build_parser.add_argument("--architecture-runs", type=Path)
    build_parser.add_argument("--json", action="store_true")
    build_parser.set_defaults(func=workbench_build_command)

    status_parser = workbench_subparsers.add_parser(
        "status",
        help="Report dataset identity, annotation capabilities, and app asset freshness.",
    )
    _add_app_selector(status_parser)
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=workbench_status_command)

    serve_parser = workbench_subparsers.add_parser(
        "serve",
        help="Serve one workbench with annotation and dataset APIs enabled.",
    )
    _add_app_selector(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.set_defaults(func=workbench_serve_command)

    export_parser = workbench_subparsers.add_parser(
        "export-intermediate",
        help="Export one stack as browser-readable Process Lab frames.",
    )
    add_export_intermediate_arguments(export_parser)
    export_parser.set_defaults(func=export_intermediate_command)

    attach_parser = workbench_subparsers.add_parser(
        "attach-intermediates",
        help="Attach frame-like pipeline_run artifacts to a workbench Process Lab run.",
    )
    add_attach_intermediates_arguments(attach_parser)
    attach_parser.set_defaults(func=attach_intermediates_command)
    return parser


def workbench_build_command(args: argparse.Namespace) -> int:
    from neurobench.workbench.builder import build_workbench, resolve_build_inputs

    if not args.dataset_manifest and not args.review_data:
        raise ValueError("workbench build requires --dataset-manifest or --review-data")
    default_review = args.review_data or Path("review_data.json")
    default_app = args.app_dir or Path(default_review).parent
    inputs = resolve_build_inputs(
        app_dir=args.app_dir,
        review_data=args.review_data,
        dataset_manifest=args.dataset_manifest,
        architecture_runs=args.architecture_runs,
        default_app_dir=default_app,
        default_review_data=default_review,
        default_dataset_id="dataset",
    )
    paths = build_workbench(
        app_dir=inputs["app_dir"],
        review_data_path=inputs["review_data_path"],
        dataset_id=inputs["dataset_id"],
        dataset_manifest=inputs["dataset_manifest"],
        architecture_runs_path=inputs["architecture_runs_path"],
    )
    payload = {key: str(value) for key, value in paths.items()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Workbench: {paths['index']}")
        print(f"Annotations preserved: {paths['annotations']}")
        print(f"Run catalog preserved: {paths['architecture_runs']}")
    return 0


def workbench_status_command(args: argparse.Namespace) -> int:
    from neurobench.workbench.builder import load_workbench_asset, workbench_asset_version

    app_dir = _catalog_app_dir(args)
    record = dataset_record_for_app(app_dir, workspace_root=args.catalog_root)
    installed_css = (app_dir / "workbench.css").read_text(encoding="utf-8") if (app_dir / "workbench.css").is_file() else ""
    installed_js = (app_dir / "workbench.js").read_text(encoding="utf-8") if (app_dir / "workbench.js").is_file() else ""
    installed_html = (app_dir / "index.html").read_text(encoding="utf-8") if (app_dir / "index.html").is_file() else ""
    version_match = re.search(
        r'<meta\s+name=["\']neurobench-workbench-asset-version["\']\s+content=["\']([0-9a-f]{12})["\']',
        installed_html,
        flags=re.IGNORECASE,
    )
    installed_version = version_match.group(1).lower() if version_match and installed_css and installed_js else ""
    packaged_version = workbench_asset_version()
    payload = {
        "dataset": record,
        "assets": {
            "installed_version": installed_version,
            "packaged_version": packaged_version,
            "current": bool(installed_version and installed_version == packaged_version),
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        caps = record.get("capabilities") or {}
        print(f"dataset: {record.get('dataset_id', '')}")
        print(f"app: {app_dir}")
        print(f"manual ROI: {'yes' if caps.get('manual_roi_annotation') else 'no'}")
        print(f"CFAR annotation: {'yes' if caps.get('cfar_annotation') else 'no'}")
        print(f"assets current: {'yes' if payload['assets']['current'] else 'no'} ({installed_version or 'missing'} -> {packaged_version})")
    return 0


def workbench_serve_command(args: argparse.Namespace) -> int:
    from neurobench.workbench.server import create_workbench_server

    app_dir = _catalog_app_dir(args)
    server, _ = create_workbench_server(app_dir=app_dir, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"Serving {app_dir} at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
