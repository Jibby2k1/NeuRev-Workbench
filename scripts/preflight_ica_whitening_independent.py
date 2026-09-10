#!/usr/bin/env python3
"""Freeze the independent-recording confirmation eligibility contract."""
from __future__ import annotations

import argparse
import json

from neurobench.experiments.ica_whitening_evaluation.independent_preflight import (
    run_independent_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = run_independent_preflight(
        dataset_root=args.dataset_root,
        dataset_dir=args.dataset_dir,
        destination=args.destination,
    )
    print(json.dumps({
        "status": result["status"],
        "eligible_recordings": result["eligible_recordings"],
    }, indent=2))


if __name__ == "__main__":
    main()
