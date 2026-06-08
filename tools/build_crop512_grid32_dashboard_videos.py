#!/usr/bin/env python3
"""Build cropped 32x32 grid-dynamics comparison videos and refresh the dashboard."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.dynamics.comparison import build_comparison_dashboard
from neurobench.dynamics.visualize_sweep import generate_sweep_visuals
from tools.build_grid64_dashboard_artifacts import build_video_selector, choose_models, load_json, load_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("Outputs/GridModel/060126_crop512_grid32_v1/cropped32_restricted_sweep_v1"))
    parser.add_argument("--mapping-json", type=Path, default=Path("Outputs/GridModel/060126_crop512_grid32_v1/datasets_cropped32_mapping.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("Outputs/GridModel/060126_crop512_grid32_v1/dashboard/app/grid_dynamics/crop512_grid32_comparison_v1"))
    parser.add_argument("--dashboard-prefix", default="grid_dynamics/crop512_grid32_comparison_v1")
    parser.add_argument("--title", default="Crop512 32x32 Grid Dynamics Comparison")
    parser.add_argument("--selected-count", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    summary_tsv = args.sweep_dir / "sweep_summary.tsv"
    if not summary_tsv.exists():
        raise FileNotFoundError(f"Missing sweep summary: {summary_tsv}")
    if not args.mapping_json.exists():
        raise FileNotFoundError(f"Missing dataset mapping: {args.mapping_json}")

    visuals_dir = args.sweep_dir / "visuals"
    generate_sweep_visuals(
        summary_tsv=summary_tsv,
        out_dir=visuals_dir,
        title=args.title,
        dashboard_prefix=args.dashboard_prefix,
        top_n=30,
    )
    rows = load_rows(summary_tsv)
    models = choose_models(rows, selected_count=args.selected_count)
    selector = build_video_selector(
        sweep_dir=args.sweep_dir,
        mapping=load_json(args.mapping_json),
        models=models,
        charts_dir=visuals_dir / "charts",
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        device=args.device,
        fps=args.fps,
        scale=args.scale,
        force=args.force,
    )

    dashboard_charts = args.out_dir.parent / args.sweep_dir.name / "charts"
    dashboard_charts.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(visuals_dir / "charts", dashboard_charts, dirs_exist_ok=True)

    comparison = build_comparison_dashboard(
        sweep_dirs=[args.sweep_dir],
        out_dir=args.out_dir,
        title=args.title,
        dashboard_prefix=args.dashboard_prefix,
        selected_count=args.selected_count,
    )
    result = {
        "status": "built",
        "models": [model.row["experiment_id"] for model in models],
        "input_options": len({item["video_id"] for item in selector["options"]}),
        "clip_options": len(selector["options"]),
        "mp4_count": len(list((visuals_dir / "charts").glob("*.mp4"))),
        "selector_json": str(visuals_dir / "charts" / "original_vs_reconstruction_selector.json"),
        "dashboard_charts": str(dashboard_charts),
        "comparison": comparison,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
