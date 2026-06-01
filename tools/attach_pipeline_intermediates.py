#!/usr/bin/env python3
"""Compatibility wrapper for attaching pipeline intermediates to a workbench run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.workbench.intermediates import add_attach_intermediates_arguments, attach_intermediates_command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_attach_intermediates_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return attach_intermediates_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
