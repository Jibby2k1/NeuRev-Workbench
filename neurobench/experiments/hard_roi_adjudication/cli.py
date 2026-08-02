"""CLI for the hard-ROI adjudication workflow."""
from __future__ import annotations

import argparse
import json

from .config import HardRoiAdjudicationConfig
from .review_pack import create_review_pack, preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neurobench.experiments.hard_roi_adjudication",
        description="Versioned Spon Ca Burst hard-ROI adjudication.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    preflight_parser = actions.add_parser("preflight")
    preflight_parser.add_argument("--config", required=True)
    review_parser = actions.add_parser("review-pack")
    review_parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = HardRoiAdjudicationConfig.load(args.config)
    if args.action == "preflight":
        payload = preflight(config)
    elif args.action == "review-pack":
        payload = create_review_pack(config)
    else:  # pragma: no cover - argparse enforces this
        raise RuntimeError(args.action)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
