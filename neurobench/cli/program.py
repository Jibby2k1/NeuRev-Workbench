"""CLI for stage-gated research-program planning."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def add_program_subcommands(subparsers) -> None:
    """Register lightweight program-audit commands."""
    root = subparsers.add_parser(
        "program",
        help="Audit stage-gated research programs without launching experiments.",
        description="Audit stage-gated research programs without launching experiments.",
    )
    programs = root.add_subparsers(dest="program_name", required=True)
    fish = programs.add_parser(
        "fish-control",
        help="Audit the neural-intent and inverse-control program.",
    )
    actions = fish.add_subparsers(dest="program_action", required=True)
    audit = actions.add_parser(
        "audit",
        help="Validate dependencies, gates, paths, resources, and experiment counts.",
    )
    audit.add_argument("--manifest", required=True, type=Path)
    audit.add_argument("--out-dir", type=Path, default=None)
    audit.add_argument(
        "--no-path-checks",
        action="store_true",
        help="Validate structure and counts without checking local artifact paths.",
    )
    audit.set_defaults(func=_run_fish_control_audit)


def _run_fish_control_audit(args) -> int:
    from neurobench.programs.fish_control import (
        audit_program_manifest,
        write_program_audit,
    )

    try:
        audit = audit_program_manifest(
            args.manifest,
            check_paths=not bool(args.no_path_checks),
        )
        written = (
            write_program_audit(audit, args.out_dir)
            if args.out_dir is not None
            else {}
        )
    except Exception as exc:
        print(f"Fish-control program audit failed: {args.manifest}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(audit, indent=2, sort_keys=True))
    for kind, path in written.items():
        print(f"{kind}: {path}", file=sys.stderr)
    return 0

