#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from neurobench.experiments.ica_whitening_evaluation.independent_confirmation import (
    freeze_independent_confirmation_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = freeze_independent_confirmation_contract(
        independent_preflight=args.preflight, destination=args.destination,
    )
    print(json.dumps({"status": result["status"],
                      "eligible_recordings": result["eligible_recordings"]}, indent=2))


if __name__ == "__main__":
    main()
