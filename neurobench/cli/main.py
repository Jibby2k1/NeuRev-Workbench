"""Minimal argparse command surface for Neurobench."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
import sys


COMMAND_GROUPS = {
    "review": "Create review batches and reports.",
    "metrics": "Compute scientific metrics.",
    "report": "Generate human-readable reports.",
    "import": "Import external tool outputs.",
    "export": "Export annotations, traces, and reproducible bundles.",
    "benchmark": "Benchmark stages and processing paths.",
}


COMMAND_REGISTRARS = (
    ("dataset", "neurobench.cli.dataset", "add_dataset_subcommands"),
    ("video", "neurobench.cli.video", "add_video_subcommands"),
    ("template", "neurobench.cli.template", "add_template_subcommands"),
    ("grid", "neurobench.cli.grid", "add_grid_subcommands"),
    ("dynamics", "neurobench.cli.dynamics", "add_dynamics_subcommands"),
    ("experiment", "neurobench.cli.experiment", "add_experiment_subcommands"),
    ("run", "neurobench.cli.run", "add_run_subcommands"),
    ("workbench", "neurobench.cli.workbench", "add_workbench_subcommands"),
    ("llm", "neurobench.cli.llm", "add_llm_subcommands"),
    ("report", "neurobench.cli.report", "add_report_subcommands"),
    ("validate", "neurobench.cli.dataset", "add_validate_subcommands"),
)
_SELECTIVE_COMMANDS = {"experiment"}


def build_parser(active_command: str | None = None) -> argparse.ArgumentParser:
    """Build the full CLI, or only one explicitly selected lightweight command."""
    parser = argparse.ArgumentParser(
        prog="neurobench",
        description="Neurobench command-line tools for neuroimaging discovery, review, and reporting.",
    )
    parser.add_argument("--version", action="version", version="neurobench 0.1.0")
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    registrations = (
        COMMAND_REGISTRARS
        if active_command is None
        else tuple(item for item in COMMAND_REGISTRARS if item[0] == active_command)
    )
    for _, module_name, function_name in registrations:
        module = importlib.import_module(module_name)
        getattr(module, function_name)(subparsers)
    for name, help_text in COMMAND_GROUPS.items():
        if name in {"report"}:
            continue
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        subparser.set_defaults(command_name=name)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    active_command = (
        arguments[0]
        if arguments and arguments[0] in _SELECTIVE_COMMANDS
        else None
    )
    parser = build_parser(active_command=active_command)
    args = parser.parse_args(arguments)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if hasattr(args, "func"):
        return int(args.func(args))
    parser.error(f"'{args.command}' command group is not implemented yet")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
