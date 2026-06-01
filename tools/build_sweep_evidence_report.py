#!/usr/bin/env python3
"""Build automated sweep evidence reports for a workbench app."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.reports.sweep_evidence import write_sweep_evidence_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True, help="Workbench app directory containing architecture_runs.json.")
    parser.add_argument("--architecture-runs", type=Path, help="Architecture runs manifest. Defaults to app-dir/architecture_runs.json.")
    parser.add_argument("--annotations", type=Path, help="Annotations file with anatomy stencil. Defaults to app-dir/annotations.json.")
    parser.add_argument("--output", type=Path, help="JSON output path. Defaults to app-dir/sweep_evidence_report.json.")
    parser.add_argument("--markdown-output", type=Path, help="Markdown output path. Defaults to the JSON path with .md suffix.")
    parser.add_argument("--stability-radius-px", type=float, default=10.0, help="Spatial radius for cross-sweep candidate stability.")
    parser.add_argument("--stability-min-support-runs", type=int, default=2, help="Other runs required for a stable candidate.")
    parser.add_argument("--stencil-edge-margin-px", type=float, default=12.0, help="Pixels near stencil edge counted as in/near stencil.")
    parser.add_argument("--target-roi-min", type=int, default=40, help="Lower target ROI burden for score balancing.")
    parser.add_argument("--target-roi-max", type=int, default=180, help="Upper target ROI burden for score balancing.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of recommended runs to include.")
    parser.add_argument("--no-attach", action="store_true", help="Do not attach report paths to architecture_runs.json.")
    args = parser.parse_args(argv)

    report = write_sweep_evidence_report(
        args.app_dir,
        output=args.output,
        markdown_output=args.markdown_output,
        architecture_runs_path=args.architecture_runs,
        annotations_path=args.annotations,
        attach=not args.no_attach,
        stability_radius_px=args.stability_radius_px,
        stability_min_support_runs=args.stability_min_support_runs,
        stencil_edge_margin_px=args.stencil_edge_margin_px,
        target_roi_range=(args.target_roi_min, args.target_roi_max),
        top_n=args.top_n,
    )
    output = (args.output or args.app_dir / "sweep_evidence_report.json").resolve()
    markdown = (args.markdown_output or output.with_suffix(".md")).resolve()
    summary = report["summary"]
    print(f"Sweep evidence JSON: {output}")
    print(f"Sweep evidence Markdown: {markdown}")
    print(f"analyzed: {summary['analyzed_run_count']} runs")
    print(f"candidate-bearing: {summary['candidate_bearing_run_count']} runs")
    print(f"median stencil coverage: {summary['median_stencil_coverage_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
