#!/usr/bin/env python3
"""Durably sequence remaining real-data whitening geometries for one shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from neurobench.experiments.ica_whitening_evaluation.real_config import RealDataConfig
from neurobench.experiments.ica_whitening_evaluation.real_runner import (
    run_single_stage_whitening_shard,
)


def _validated(config: RealDataConfig, geometry: str, shard: int) -> bool:
    tag = geometry.upper()
    path = (config.output_dir / "stages"
            / f"S2C_{tag}_WHITENING_SHARD_{shard:02d}_OF_{config.fitting.shard_count:02d}"
            / "validation.json")
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    config = RealDataConfig.from_json(args.config)
    while not _validated(config, "spatial", args.shard):
        time.sleep(args.poll_seconds)
    for geometry in (
        "temporal", "joint_spatiotemporal",
        "spatial_then_temporal", "temporal_then_spatial",
    ):
        if _validated(config, geometry, args.shard):
            continue
        run_single_stage_whitening_shard(
            config, preflight_dir=Path(args.preflight), shard_index=args.shard,
            whitening_geometry=geometry,
        )


if __name__ == "__main__":
    main()
