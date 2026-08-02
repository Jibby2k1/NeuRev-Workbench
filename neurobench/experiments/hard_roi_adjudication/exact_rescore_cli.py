"""CPU-only exact-causal hard-ROI frozen re-evaluation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HardRoiAdjudicationConfig
from .exact_reevaluate import reevaluate_exact_causal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adjudication-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args(argv)
    payload = reevaluate_exact_causal(
        HardRoiAdjudicationConfig.load(args.config),
        adjudication_tsv=args.adjudication_tsv,
        output_dir=args.output_dir,
        allow_provisional=bool(args.allow_provisional),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
