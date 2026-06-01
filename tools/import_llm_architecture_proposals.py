#!/usr/bin/env python3
"""Validate and import LLM-authored architecture proposals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.llm_planning import proposal_set_to_architecture_manifest, validate_proposal_set
from neurobench.manifests import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an LLM architecture proposal set into architecture_runs.json.")
    parser.add_argument("--proposal", type=Path, required=True, help="LLM proposal JSON.")
    parser.add_argument("--architecture-runs", type=Path, default=None, help="Existing architecture_runs.json to merge into.")
    parser.add_argument("--out", type=Path, required=True, help="Output architecture_runs.json.")
    parser.add_argument("--validation-report", type=Path, default=None, help="Optional validation report JSON.")
    parser.add_argument("--max-combinations", type=int, default=None, help="Override per-architecture sweep combination cap.")
    args = parser.parse_args()

    proposal = load_json(args.proposal)
    base = load_json(args.architecture_runs) if args.architecture_runs and args.architecture_runs.exists() else None
    validated = validate_proposal_set(proposal, max_combinations=args.max_combinations)
    manifest = proposal_set_to_architecture_manifest(
        validated,
        base_manifest=base,
        max_combinations=args.max_combinations,
    )
    write_json(args.out, manifest)
    if args.validation_report:
        write_json(args.validation_report, validated["validation_report"])
    print(
        f"Wrote {args.out} with {len(validated['proposals'])} LLM proposal"
        f"{'' if len(validated['proposals']) == 1 else 's'}"
    )


if __name__ == "__main__":
    main()
