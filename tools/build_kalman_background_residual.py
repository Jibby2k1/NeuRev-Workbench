#!/usr/bin/env python3
"""Build a Kalman-style positive residual stack from a video npy file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.algorithms.background import kalman_positive_residual_stack


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-init-frames", type=int, default=50)
    parser.add_argument("--kalman-gain", type=float, default=0.01)
    parser.add_argument("--positive-update-gain", type=float, default=0.002)
    parser.add_argument("--negative-update-gain", type=float, default=0.08)
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--write-baseline-stack", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = kalman_positive_residual_stack(
        args.source_npy,
        args.out_dir,
        baseline_init_frames=args.baseline_init_frames,
        kalman_gain=args.kalman_gain,
        positive_update_gain=args.positive_update_gain,
        negative_update_gain=args.negative_update_gain,
        chunk_frames=args.chunk_frames,
        write_baseline_stack=args.write_baseline_stack,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
