"""CLI for conservative hard-ROI timing review aids, version 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HardRoiAdjudicationConfig
from .review_checklist_v2 import generate_review_checklist_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adjudication-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = generate_review_checklist_v2(
        HardRoiAdjudicationConfig.load(args.config),
        adjudication_tsv=args.adjudication_tsv,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
