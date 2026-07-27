"""Workbench CLI commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    build_parser.add_argument(
        "--migrate-annotations",
        action="store_true",
        help="Explicitly migrate and rewrite annotations.json; otherwise existing bytes are preserved.",
    )
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
    serve_parser.add_argument(
        "--asset-mode",
        choices=("current", "installed"),
        default="current",
        help="Serve current packaged assets in memory, or the installed archived app assets.",
    )
    serve_parser.set_defaults(func=workbench_serve_command)

    baseline_parser = workbench_subparsers.add_parser(
        "baseline",
        help="Capture, verify, or diff Wave 0 preservation baselines without modifying app artifacts.",
    )
    baseline_parser.add_argument("--root", type=Path, default=Path.cwd())
    baseline_parser.add_argument("--app-dir", type=Path, action="append", default=[])
    baseline_action = baseline_parser.add_mutually_exclusive_group(required=True)
    baseline_action.add_argument("--output", type=Path, help="Capture a new baseline at this path.")
    baseline_action.add_argument("--verify", type=Path, help="Verify a stored baseline against --root.")
    baseline_action.add_argument(
        "--diff",
        type=Path,
        nargs=2,
        metavar=("EXPECTED", "ACTUAL"),
        help="Compare two stored baselines, ignoring capture timestamps.",
    )
    baseline_parser.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing baseline.")
    baseline_parser.add_argument("--json", action="store_true")
    baseline_parser.set_defaults(func=workbench_baseline_command)

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
        migrate_annotations=args.migrate_annotations,
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
    from neurobench.workbench.builder import workbench_asset_status

    app_dir = _catalog_app_dir(args)
    record = dataset_record_for_app(app_dir, workspace_root=args.catalog_root)
    assets = workbench_asset_status(app_dir)
    payload = {
        "dataset": record,
        "assets": assets,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        caps = record.get("capabilities") or {}
        print(f"dataset: {record.get('dataset_id', '')}")
        print(f"app: {app_dir}")
        print(f"manual ROI: {'yes' if caps.get('manual_roi_annotation') else 'no'}")
        print(f"CFAR annotation: {'yes' if caps.get('cfar_annotation') else 'no'}")
        print(
            f"assets current: {'yes' if assets['current'] else 'no'} "
            f"({assets['installed_version'] or 'missing'} -> {assets['packaged_version']}; "
            f"css={'ok' if assets['css_current'] else 'stale'}, "
            f"js={'ok' if assets['js_current'] else 'stale'})"
        )
    return 0


def workbench_serve_command(args: argparse.Namespace) -> int:
    from neurobench.workbench.server import create_workbench_server

    app_dir = _catalog_app_dir(args)
    if args.asset_mode == "installed":
        from neurobench.workbench.builder import workbench_asset_status

        assets = workbench_asset_status(app_dir)
        if not assets["current"]:
            print(
                "WARNING: serving stale or tampered installed workbench assets; "
                "use --asset-mode current to serve packaged assets without modifying the app.",
                file=sys.stderr,
            )
    server, _ = create_workbench_server(
        app_dir=app_dir,
        host=args.host,
        port=args.port,
        asset_mode=args.asset_mode,
    )
    host, port = server.server_address[:2]
    print(f"Serving {app_dir} at http://{host}:{port}/ ({args.asset_mode} assets)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _baseline_path(root: Path, value: Path) -> Path:
    path = value.expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Baseline path must be under --root {root}: {resolved}")
    return resolved


def workbench_baseline_command(args: argparse.Namespace) -> int:
    from neurobench.workbench.baseline import (
        capture_baseline,
        diff_baselines,
        load_baseline,
        verify_baseline,
        write_baseline,
    )

    root = args.root.expanduser().resolve()
    if args.output is not None:
        baseline = capture_baseline(root, app_dirs=args.app_dir or None)
        output = write_baseline(
            _baseline_path(root, args.output),
            baseline,
            overwrite=args.overwrite,
        )
        if args.json:
            print(json.dumps({"output": str(output), "baseline": baseline}, indent=2, sort_keys=True))
        else:
            print(f"Wave 0 baseline: {output}")
            print(f"Stable identity: {baseline['stable_identity']['sha256']}")
            print(f"Apps locked: {len(baseline.get('apps') or [])}")
            print(f"Catalog datasets: {len(baseline.get('catalog') or [])}")
        return 0

    if args.overwrite:
        raise ValueError("--overwrite is only valid with --output")
    if args.app_dir:
        raise ValueError("--app-dir is only valid with --output; verification uses the stored app set")

    if args.verify is not None:
        source = _baseline_path(root, args.verify)
        report = verify_baseline(root, load_baseline(source))
        label = f"Baseline verification: {'match' if report['match'] else 'DIFFERENT'}"
    else:
        expected_path, actual_path = (_baseline_path(root, value) for value in args.diff)
        report = diff_baselines(load_baseline(expected_path), load_baseline(actual_path))
        label = f"Baseline diff: {'match' if report['match'] else 'DIFFERENT'}"

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(label)
        print(f"Expected identity: {report['expected_identity']}")
        print(f"Actual identity:   {report['actual_identity']}")
        for change in report["changes"][:20]:
            print(f"changed: {change['path']}")
        if len(report["changes"]) > 20:
            print(f"... {len(report['changes']) - 20} additional changes")
    return 0 if report["match"] else 1
