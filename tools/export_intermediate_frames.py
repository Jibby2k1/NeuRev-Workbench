#!/usr/bin/env python3
"""Compatibility wrapper for exporting Process Lab intermediate frame stacks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.workbench.intermediates import add_export_intermediate_arguments, export_intermediate_command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export intermediate stack frames for Process Lab.")
    return add_export_intermediate_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return export_intermediate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
