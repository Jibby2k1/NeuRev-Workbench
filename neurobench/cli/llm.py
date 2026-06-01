"""LLM-assisted architecture planning CLI commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurobench.llm_planning import build_llm_context, proposal_set_to_architecture_manifest, render_llm_prompt, validate_proposal_set
from neurobench.manifests import load_json, write_json
from neurobench.validation.schemas import validation_error_summary


def add_llm_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "llm",
        help="Build LLM handoffs, import proposals, and run local proposal experiments.",
        description="Build LLM handoffs, import proposals, and run local proposal experiments.",
    )
    llm_subparsers = parser.add_subparsers(dest="llm_command", metavar="llm-command")

    context_parser = llm_subparsers.add_parser("context", help="Build a provider-neutral LLM architecture context.")
    context_parser.add_argument("--dataset-manifest", type=Path, help="Optional dataset manifest JSON.")
    context_parser.add_argument("--architecture-runs", type=Path, help="Optional architecture_runs.json baseline.")
    context_parser.add_argument("--objective", default="review_efficiency", help="Optimization objective label.")
    context_parser.add_argument("--max-combinations", type=int, default=4096, help="Maximum combinations per proposed architecture.")
    context_parser.add_argument("--lab-notes", default="", help="Extra lab notes to include in the context payload.")
    context_parser.add_argument("--context-out", type=Path, help="Write context JSON to this path.")
    context_parser.add_argument("--prompt-out", type=Path, help="Write Markdown prompt to this path.")
    context_parser.add_argument("--json", action="store_true", help="Print context JSON instead of the prompt.")
    context_parser.set_defaults(func=llm_context_command)

    import_parser = llm_subparsers.add_parser("import-proposals", help="Validate and import an LLM proposal set.")
    import_parser.add_argument("proposal", type=Path, help="LLM proposal JSON.")
    import_parser.add_argument("--architecture-runs", type=Path, help="Optional existing architecture_runs.json to merge into.")
    import_parser.add_argument("--out", required=True, type=Path, help="Output architecture_runs.json path.")
    import_parser.add_argument("--validation-report", type=Path, help="Optional validation report JSON path.")
    import_parser.add_argument("--max-combinations", type=int, default=None, help="Override per-architecture sweep combination cap.")
    import_parser.add_argument("--json", action="store_true", help="Print import summary JSON.")
    import_parser.set_defaults(func=llm_import_proposals_command)

    run_parser = llm_subparsers.add_parser("run-proposals", help="Execute locally runnable LLM proposal runs.")
    run_parser.add_argument("proposal", type=Path, help="LLM proposal JSON.")
    run_parser.add_argument("--run-root", required=True, type=Path, help="Output root for executed proposal runs.")
    run_parser.add_argument("--max-combinations", type=int, default=None, help="Override per-architecture sweep combination cap.")
    run_parser.add_argument("--ground-truth-csv", type=Path, help="Optional ground-truth CSV for recall/burden metrics.")
    run_parser.add_argument("--centroid-tolerance-px", type=float, default=4.0, help="Centroid tolerance for optional object matching.")
    run_parser.add_argument("--event-tolerance-frames", type=int, default=2, help="Frame tolerance for optional event-onset matching.")
    run_parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed proposal run.")
    run_parser.add_argument("--json", action="store_true", help="Print run summary JSON.")
    run_parser.set_defaults(func=llm_run_proposals_command)
    return parser


def llm_context_command(args: argparse.Namespace) -> int:
    try:
        context = build_llm_context(
            dataset_manifest=load_json(args.dataset_manifest) if args.dataset_manifest else None,
            architecture_runs=load_json(args.architecture_runs) if args.architecture_runs else None,
            objective=args.objective,
            max_combinations=args.max_combinations,
            lab_notes=args.lab_notes,
        )
        prompt = render_llm_prompt(context)
        if args.context_out:
            write_json(args.context_out, context)
        if args.prompt_out:
            args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
            args.prompt_out.write_text(prompt, encoding="utf-8")
    except Exception as exc:
        print("LLM context generation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(context, indent=2, sort_keys=True))
    else:
        print(prompt, end="")
    return 0


def llm_import_proposals_command(args: argparse.Namespace) -> int:
    try:
        proposal = load_json(args.proposal)
        base = load_json(args.architecture_runs) if args.architecture_runs else None
        manifest = proposal_set_to_architecture_manifest(proposal, base_manifest=base, max_combinations=args.max_combinations)
        write_json(args.out, manifest)
        validation_report = validate_proposal_set(proposal, max_combinations=args.max_combinations).get("validation_report", {})
        if args.validation_report:
            write_json(args.validation_report, validation_report)
    except Exception as exc:
        print(f"LLM proposal import failed: {args.proposal}", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    summary = {
        "status": "imported",
        "proposal": str(args.proposal),
        "out": str(args.out),
        "proposal_count": validation_report.get("proposal_count", 0),
        "run_count": len(manifest.get("runs", [])),
        "saved_pipeline_count": len(manifest.get("saved_pipelines", [])),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Imported LLM proposals: {args.out}")
        print(f"proposals: {summary['proposal_count']}")
        print(f"planned runs: {summary['run_count']}")
    return 0


def llm_run_proposals_command(args: argparse.Namespace) -> int:
    try:
        from neurobench.llm_experiments import execute_llm_proposal_experiments

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
    except Exception as exc:
        print(f"LLM proposal execution failed: {args.proposal}", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"LLM experiment summary: {Path(args.run_root) / 'llm_experiment_summary.json'}")
        print(f"status: {summary['status']}")
        print(f"succeeded: {summary['succeeded']}")
        print(f"failed: {summary['failed']}")
    return 0 if summary.get("failed", 0) == 0 else 1
