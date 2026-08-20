#!/usr/bin/env python3
"""CLI for the staged center-whiten-ICA architecture benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from neurobench.experiments.msln_msica.center_whiten_ica_benchmark import preflight_screen
from neurobench.experiments.msln_msica.center_whiten_ica_screen import run_metrics_screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "screen"))
    parser.add_argument("config")
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight_screen(args.config), indent=2))
    elif args.command == "screen":
        print(json.dumps(run_metrics_screen(args.config), indent=2))


if __name__ == "__main__":
    main()
