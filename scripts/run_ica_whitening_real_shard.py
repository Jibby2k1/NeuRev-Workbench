#!/usr/bin/env python3
"""Run one resumable real-data ICA whitening shard."""
from __future__ import annotations

import argparse

from neurobench.experiments.ica_whitening_evaluation.real_config import RealDataConfig
from neurobench.experiments.ica_whitening_evaluation.real_runner import (
    run_single_stage_whitening_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--geometry", required=True, choices=(
        "spatial", "temporal", "joint_spatiotemporal",
        "spatial_then_temporal", "temporal_then_spatial",
    ))
    parser.add_argument("--shard", required=True, type=int)
    args = parser.parse_args()
    config = RealDataConfig.from_json(args.config)
    summary = run_single_stage_whitening_shard(
        config, preflight_dir=args.preflight, shard_index=args.shard,
        whitening_geometry=args.geometry,
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
