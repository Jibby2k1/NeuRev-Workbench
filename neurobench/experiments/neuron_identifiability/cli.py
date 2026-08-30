"""Command-line interface for the identifiability paper program."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import ProgramRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        item = sub.add_parser(command); item.add_argument("--config", type=Path, required=True)
        item.add_argument("--dry-run", action="store_true"); item.add_argument("--resume", action="store_true")
        item.add_argument("--from-stage"); item.add_argument("--through-stage"); item.add_argument("--cpu-only", action="store_true")
        item.add_argument("--max-workers", type=int); item.add_argument("--memory-limit-gib", type=float)
        item.add_argument("--allow-provisional-labels", action="store_true"); item.add_argument("--strict", action="store_true")
    export = sub.add_parser("paper-export"); export.add_argument("--run-root", type=Path, required=True); export.add_argument("--overleaf-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "paper-export":
        source = args.run_root / "manuscript"
        if not source.exists(): raise FileNotFoundError(source)
        args.overleaf_root.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file(): (args.overleaf_root / item.name).write_bytes(item.read_bytes())
        return
    config = load_config(args.config)
    runner = ProgramRunner(config, resume=args.resume, dry_run=args.dry_run)
    through = "00_preflight" if args.command == "preflight" else args.through_stage
    print(runner.run(args.from_stage, through))
