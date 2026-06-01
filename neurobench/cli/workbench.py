"""Workbench CLI commands."""
from __future__ import annotations

import argparse

from neurobench.workbench.intermediates import (
    add_attach_intermediates_arguments,
    add_export_intermediate_arguments,
    attach_intermediates_command,
    export_intermediate_command,
)


def add_workbench_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "workbench",
        help="Build, serve, and prepare local review workbenches.",
        description="Build, serve, and prepare local review workbenches.",
    )
    workbench_subparsers = parser.add_subparsers(dest="workbench_command", metavar="workbench-command")

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
