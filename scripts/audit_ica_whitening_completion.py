#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from neurobench.experiments.ica_whitening_evaluation.completion_audit import (
    write_completion_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-factorial-fits", type=int, default=30_891)
    args = parser.parse_args()
    result = write_completion_audit(
        args.experiment_root, args.destination,
        expected_factorial_fits=args.expected_factorial_fits,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
