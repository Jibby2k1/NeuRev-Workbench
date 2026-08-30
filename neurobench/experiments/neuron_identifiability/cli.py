"""Command-line interface for the identifiability paper program."""
from __future__ import annotations

import argparse
import json
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
    jepa = sub.add_parser("jepa-pilot")
    jepa.add_argument("--config", type=Path, required=True)
    jepa.add_argument("--mode", choices=("preflight", "smoke", "screen"), default="preflight")
    jepa.add_argument("--repository-root", type=Path, default=Path.cwd())
    jepa.add_argument("--data-root", type=Path)
    jepa.add_argument("--output-root", type=Path)
    jepa.add_argument("--device", choices=("cpu", "cuda"))
    jepa.add_argument("--steps", type=int)
    jepa.add_argument("--train-clip-count", type=int)
    jepa.add_argument("--validation-clip-count", type=int)
    jepa.add_argument("--max-training-seeds", type=int, default=1)
    jepa.add_argument("--all-training-seeds", action="store_true")
    jepa.add_argument("--injection-cells-per-background", type=int, default=1)
    jepa.add_argument("--full-injection-grid", action="store_true")
    jepa.add_argument("--verify-hashes", action="store_true")
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
    if args.command == "jepa-pilot":
        from .jepa_pilot import run_jepa_pilot

        result = run_jepa_pilot(
            args.config,
            repository_root=args.repository_root,
            mode=args.mode,
            data_root=args.data_root,
            output_root=args.output_root,
            device=args.device,
            steps=args.steps,
            train_clip_count=args.train_clip_count,
            validation_clip_count=args.validation_clip_count,
            max_training_seeds=(None if args.all_training_seeds else args.max_training_seeds),
            injection_cells_per_background=(
                None if args.full_injection_grid else args.injection_cells_per_background
            ),
            verify_hashes=args.verify_hashes,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return
    config = load_config(args.config)
    runner = ProgramRunner(config, resume=args.resume, dry_run=args.dry_run)
    through = "00_preflight" if args.command == "preflight" else args.through_stage
    print(runner.run(args.from_stage, through))
