#!/usr/bin/env python3
"""Write a concise brief for a locally executed Gamma CFAR sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.reports.gamma_cfar_sweep import write_gamma_cfar_sweep_brief


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path, help="Directory containing sweep_summary.json and run folders.")
    parser.add_argument("--output", required=True, type=Path, help="Markdown brief output path.")
    parser.add_argument("--pixel-size-um", type=float, default=0.5, help="Microns per pixel for ROI size interpretation.")
    parser.add_argument("--size-mode", choices=["microns", "pixels"], default="microns", help="Report ROI sizes in microns when pixel size is known, or pixels when it is unknown.")
    parser.add_argument("--top-n", type=int, default=5, help="Top ranked candidates to include per recommended run.")
    args = parser.parse_args(argv)

    summary = write_gamma_cfar_sweep_brief(
        args.sweep_root,
        args.output,
        pixel_size_um=None if args.size_mode == "pixels" else args.pixel_size_um,
        top_n=args.top_n,
        size_mode=args.size_mode,
    )
    print(f"Gamma CFAR brief: {args.output}")
    print(f"JSON summary: {args.output.with_suffix('.json')}")
    print(f"completed: {summary['completed_count']}/{summary['run_count']}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
