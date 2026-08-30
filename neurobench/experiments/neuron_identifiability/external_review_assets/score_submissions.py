#!/usr/bin/env python3
"""Private wrapper for repository scoring code; run from the repository checkout."""
from pathlib import Path
import argparse

from neurobench.experiments.neuron_identifiability.external_bounded_review import analyze_submissions

parser = argparse.ArgumentParser()
parser.add_argument("--private-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("submissions", nargs="+", type=Path)
args = parser.parse_args()
analyze_submissions(args.submissions, args.private_dir, args.output)
print(args.output)
