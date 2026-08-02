"""Standalone bounded CLI for information source-separation development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import InformationSeparationConfig
from .preflight import audit
from .runner import run_tiny_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--output-dir", type=Path)
    smoke_parser = subparsers.add_parser("tiny-smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = InformationSeparationConfig.load(args.config)
    if args.action == "preflight":
        payload = audit(config, output_dir=args.output_dir)
    else:
        payload = run_tiny_smoke(config, output_dir=args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
