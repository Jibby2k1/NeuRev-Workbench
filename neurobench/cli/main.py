"""Minimal argparse command surface for Neurobench."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from neurobench.cli.dataset import add_dataset_subcommands, add_validate_subcommands
from neurobench.cli.dynamics import add_dynamics_subcommands
from neurobench.cli.llm import add_llm_subcommands
from neurobench.cli.report import add_report_subcommands
from neurobench.cli.video import add_video_subcommands
from neurobench.cli.template import add_template_subcommands
from neurobench.cli.grid import add_grid_subcommands
from neurobench.cli.run import add_run_subcommands
from neurobench.cli.workbench import add_workbench_subcommands


COMMAND_GROUPS = {
    "review": "Create review batches and reports.",
    "metrics": "Compute scientific metrics.",
    "report": "Generate human-readable reports.",
    "import": "Import external tool outputs.",
    "export": "Export annotations, traces, and reproducible bundles.",
    "benchmark": "Benchmark stages and processing paths.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurobench",
        description="Neurobench command-line tools for neuroimaging discovery, review, and reporting.",
    )
    parser.add_argument("--version", action="version", version="neurobench 0.1.0")
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    add_dataset_subcommands(subparsers)
    add_video_subcommands(subparsers)
    add_template_subcommands(subparsers)
    add_grid_subcommands(subparsers)
    add_dynamics_subcommands(subparsers)
    add_run_subcommands(subparsers)
    add_workbench_subcommands(subparsers)
    add_llm_subcommands(subparsers)
    add_report_subcommands(subparsers)
    add_validate_subcommands(subparsers)
    for name, help_text in COMMAND_GROUPS.items():
        if name in {"report"}:
            continue
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        subparser.set_defaults(command_name=name)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if hasattr(args, "func"):
        return int(args.func(args))
    parser.error(f"'{args.command}' command group is not implemented yet")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
