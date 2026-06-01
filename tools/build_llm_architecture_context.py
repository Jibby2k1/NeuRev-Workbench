#!/usr/bin/env python3
"""Build provider-neutral LLM handoff context for Neurobench architecture planning."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.llm_planning import DEFAULT_MAX_COMBINATIONS, build_llm_context, render_llm_prompt
from neurobench.manifests import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an LLM handoff context for architecture proposals.")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="Dataset manifest JSON.")
    parser.add_argument("--architecture-runs", type=Path, default=None, help="Existing architecture_runs.json.")
    parser.add_argument("--lab-notes", type=Path, default=None, help="Optional free-text lab notes.")
    parser.add_argument("--objective", default="review_efficiency")
    parser.add_argument("--max-combinations", type=int, default=DEFAULT_MAX_COMBINATIONS)
    parser.add_argument("--out", type=Path, required=True, help="Output context JSON.")
    parser.add_argument("--prompt-out", type=Path, default=None, help="Optional Markdown prompt output.")
    args = parser.parse_args()

    dataset_manifest = load_json(args.dataset_manifest) if args.dataset_manifest else None
    architecture_runs = load_json(args.architecture_runs) if args.architecture_runs else None
    lab_notes = args.lab_notes.read_text(encoding="utf-8") if args.lab_notes else ""
    context = build_llm_context(
        dataset_manifest=dataset_manifest,
        architecture_runs=architecture_runs,
        objective=args.objective,
        max_combinations=args.max_combinations,
        lab_notes=lab_notes,
    )
    write_json(args.out, context)
    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(render_llm_prompt(context), encoding="utf-8")
    print(f"Wrote LLM architecture context to {args.out}")


if __name__ == "__main__":
    main()
