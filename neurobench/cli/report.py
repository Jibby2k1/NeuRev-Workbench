"""Report CLI commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurobench.manifests import write_json
from neurobench.validation.schemas import validation_error_summary


def add_report_subcommands(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "report",
        help="Generate metrics and comparison reports.",
        description="Generate metrics and comparison reports.",
    )
    report_subparsers = parser.add_subparsers(dest="report_command", metavar="report-command")

    generate_parser = report_subparsers.add_parser("generate", help="Generate metrics_report.json and report.md.")
    generate_parser.add_argument("runs", nargs="+", type=Path, help="One or more pipeline_run.json files.")
    generate_parser.add_argument("--output", required=True, type=Path, help="Output directory for report files.")
    generate_parser.add_argument("--metrics-report-id", default=None, help="Optional stable metrics report ID.")
    generate_parser.set_defaults(func=generate_report_command)

    compare_parser = report_subparsers.add_parser("compare", help="Generate comparison_report.json and comparison_report.md.")
    compare_parser.add_argument("runs", nargs="+", type=Path, help="Two or more pipeline_run.json files.")
    compare_parser.add_argument("--output", required=True, type=Path, help="Output directory for comparison files.")
    compare_parser.set_defaults(func=compare_report_command)
    return parser


def generate_report_command(args: argparse.Namespace) -> int:
    try:
        from neurobench.reports import build_metrics_report_from_pipeline_runs, write_metrics_report_markdown

        report = build_metrics_report_from_pipeline_runs(
            args.runs,
            metrics_report_id=args.metrics_report_id,
        )
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics_report.json"
        markdown_path = out_dir / "report.md"
        report.write_json(metrics_path)
        write_metrics_report_markdown(report, markdown_path)
    except Exception as exc:
        print("Metrics report generation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Metrics report JSON: {metrics_path}")
    print(f"Metrics report Markdown: {markdown_path}")
    print(f"runs: {len(report.run_ids)}")
    return 0


def compare_report_command(args: argparse.Namespace) -> int:
    if len(args.runs) < 2:
        print("Run comparison requires at least two pipeline_run.json files.", file=sys.stderr)
        return 1
    try:
        from neurobench.reports import build_metrics_report_from_pipeline_runs, build_run_comparison_report, render_run_comparison_markdown

        reports = [build_metrics_report_from_pipeline_runs([path]) for path in args.runs]
        comparison = build_run_comparison_report(reports)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "comparison_report.json"
        markdown_path = out_dir / "comparison_report.md"
        write_json(json_path, comparison)
        markdown_path.write_text(render_run_comparison_markdown(comparison), encoding="utf-8")
    except Exception as exc:
        print("Run comparison report generation failed", file=sys.stderr)
        print(validation_error_summary(exc), file=sys.stderr)
        return 1
    print(f"Comparison report JSON: {json_path}")
    print(f"Comparison report Markdown: {markdown_path}")
    print(f"runs: {comparison['run_count']}")
    return 0
