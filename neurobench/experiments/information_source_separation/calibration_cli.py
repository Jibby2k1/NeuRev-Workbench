"""CLI for disjoint source-separation identifiability calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_config import CalibrationConfig
from .calibration_runner import audit_calibration, run_calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "resume"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = CalibrationConfig.load(args.config)
    payload = audit_calibration(config, output_dir=args.output_dir) if args.action == "preflight" else run_calibration(config, output_dir=args.output_dir, resume=args.action == "resume")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
