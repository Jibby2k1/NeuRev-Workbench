"""Command-line entrypoint for the guarded ICA/whitening program."""
from __future__ import annotations

import argparse
import json

from .analysis import load_shard_rows, write_analysis
from .config import ICAWhiteningConfig
from .preflight import preflight
from .synthetic_runner import merge_synthetic_shards, run_synthetic_screen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--artifact-dir", required=True)
    synthetic_parser = subparsers.add_parser("synthetic")
    synthetic_parser.add_argument("--config", required=True)
    synthetic_parser.add_argument("--preflight-dir", required=True)
    synthetic_parser.add_argument("--limit", type=int)
    synthetic_parser.add_argument("--shard-index", type=int)
    synthetic_parser.add_argument("--shard-count", type=int)
    merge_parser = subparsers.add_parser("merge-synthetic")
    merge_parser.add_argument("--config", required=True)
    merge_parser.add_argument("--preflight-dir", required=True)
    merge_parser.add_argument("--shard-count", type=int, required=True)
    analysis_parser = subparsers.add_parser("analyze-synthetic")
    analysis_parser.add_argument("--config", required=True)
    analysis_parser.add_argument("--shard-count", type=int, required=True)
    analysis_parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    config = ICAWhiteningConfig.from_json(arguments.config)
    if arguments.command == "preflight":
        result = preflight(config, artifact_dir=arguments.artifact_dir)
    elif arguments.command == "synthetic":
        result = run_synthetic_screen(
            config, preflight_dir=arguments.preflight_dir, limit=arguments.limit,
            shard_index=arguments.shard_index, shard_count=arguments.shard_count,
        )
    elif arguments.command == "merge-synthetic":
        result = merge_synthetic_shards(
            config, preflight_dir=arguments.preflight_dir,
            shard_count=arguments.shard_count,
        )
    else:
        rows = load_shard_rows(config.output_dir / "stages", arguments.shard_count)
        result = write_analysis(rows, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
