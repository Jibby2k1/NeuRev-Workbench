#!/usr/bin/env python3
"""Run locally executable LLM architecture proposal experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.llm_experiments import execute_llm_proposal_experiments
from neurobench.manifests import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute locally supported LLM architecture proposal runs.")
    parser.add_argument("--proposal", type=Path, required=True, help="LLM proposal JSON.")
    parser.add_argument("--run-root", type=Path, required=True, help="Output root for executed proposal runs.")
    parser.add_argument("--max-combinations", type=int, default=None, help="Override per-architecture sweep combination cap.")
    parser.add_argument("--ground-truth-csv", type=Path, default=None, help="Optional object/event ground-truth CSV with ID, Start Frame, End Frame, X, Y columns.")
    parser.add_argument("--centroid-tolerance-px", type=float, default=4.0, help="Centroid tolerance for optional object matching.")
    parser.add_argument("--event-tolerance-frames", type=int, default=2, help="Frame tolerance for optional event-onset matching.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed run.")
    args = parser.parse_args()

    proposal = load_json(args.proposal)
    summary = execute_llm_proposal_experiments(
        proposal,
        run_root=args.run_root,
        max_combinations=args.max_combinations,
        ground_truth_csv=args.ground_truth_csv,
        centroid_tolerance_px=args.centroid_tolerance_px,
        event_tolerance_frames=args.event_tolerance_frames,
        stop_on_error=args.stop_on_error,
    )
    root = args.run_root
    print(f"Wrote {root / 'llm_experiment_summary.json'} with {len(summary.get('runs', []))} run records")


if __name__ == "__main__":
    main()
