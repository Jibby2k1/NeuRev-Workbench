#!/usr/bin/env python3
"""Build grid-dynamics comparison videos and a multi-sweep static dashboard."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurobench.dynamics.comparison import build_comparison_dashboard
from neurobench.dynamics.visualize_sweep import generate_sweep_visuals
from tools.build_grid64_dashboard_artifacts import build_video_selector, choose_models, load_json, load_rows


def _mapping_grid_size(mapping: Mapping[str, Any]) -> int | None:
    for item in mapping.values():
        if isinstance(item, Mapping) and item.get("grid_size"):
            return int(item["grid_size"])
        if isinstance(item, Mapping) and item.get("dataset"):
            dataset_path = Path(str(item["dataset"]))
            if dataset_path.exists():
                dataset = load_json(dataset_path)
                array_path = Path(str(dataset.get("array_path", "")))
                if array_path.exists():
                    import numpy as np

                    with np.load(array_path, allow_pickle=False) as arrays:
                        return int(arrays["targets"].shape[-1])
    return None


def _default_scale(grid_size: int | None) -> int:
    if not grid_size:
        return 4
    if grid_size <= 32:
        return 8
    if grid_size <= 64:
        return 4
    return 2


def _mapping_for_index(mappings: list[Path], index: int) -> Path:
    if len(mappings) == 1:
        return mappings[0]
    if index >= len(mappings):
        raise ValueError("Pass either one --mapping-json for all sweeps or one --mapping-json per --sweep-dir.")
    return mappings[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True, help="Sweep directory. Can be passed multiple times.")
    parser.add_argument("--mapping-json", type=Path, action="append", required=True, help="Dataset mapping JSON. Pass once or once per sweep.")
    parser.add_argument("--out-dir", type=Path, default=Path("Outputs/GridModel/060126_crop512_highres_temporalcnn_v1/dashboard/app/grid_dynamics/crop512_highres_temporalcnn_v1"))
    parser.add_argument("--dashboard-prefix", default="grid_dynamics/crop512_highres_temporalcnn_v1")
    parser.add_argument("--title", default="Crop512 High-Resolution Temporal-CNN Comparison")
    parser.add_argument("--selected-count", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if len(args.mapping_json) not in {1, len(args.sweep_dir)}:
        raise ValueError("Pass either one --mapping-json for all sweeps or one --mapping-json per --sweep-dir.")

    built = []
    for index, sweep_dir in enumerate(args.sweep_dir):
        summary_tsv = sweep_dir / "sweep_summary.tsv"
        if not summary_tsv.exists():
            raise FileNotFoundError(f"Missing sweep summary: {summary_tsv}")
        mapping_path = _mapping_for_index(args.mapping_json, index)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing dataset mapping: {mapping_path}")
        mapping = load_json(mapping_path)
        grid_size = _mapping_grid_size(mapping)
        scale = int(args.scale) if args.scale is not None else _default_scale(grid_size)
        visuals_dir = sweep_dir / "visuals"
        generate_sweep_visuals(
            summary_tsv=summary_tsv,
            out_dir=visuals_dir,
            title=f"{args.title}: {sweep_dir.name}",
            dashboard_prefix=args.dashboard_prefix,
            top_n=30,
        )
        rows = load_rows(summary_tsv)
        models = choose_models(rows, selected_count=args.selected_count)
        selector = build_video_selector(
            sweep_dir=sweep_dir,
            mapping=mapping,
            models=models,
            charts_dir=visuals_dir / "charts",
            max_frames=args.max_frames,
            batch_size=args.batch_size,
            device=args.device,
            fps=args.fps,
            scale=scale,
            force=args.force,
        )
        dashboard_charts = args.out_dir.parent / sweep_dir.name / "charts"
        dashboard_charts.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(visuals_dir / "charts", dashboard_charts, dirs_exist_ok=True)
        built.append(
            {
                "sweep_dir": str(sweep_dir),
                "mapping_json": str(mapping_path),
                "grid_size": grid_size,
                "scale": scale,
                "models": [model.row["experiment_id"] for model in models],
                "input_options": len({item["video_id"] for item in selector["options"]}),
                "clip_options": len(selector["options"]),
                "mp4_count": len(list((visuals_dir / "charts").glob("*.mp4"))),
                "selector_json": str(visuals_dir / "charts" / "original_vs_reconstruction_selector.json"),
                "dashboard_charts": str(dashboard_charts),
            }
        )

    comparison = build_comparison_dashboard(
        sweep_dirs=args.sweep_dir,
        out_dir=args.out_dir,
        title=args.title,
        dashboard_prefix=args.dashboard_prefix,
        selected_count=args.selected_count,
    )
    result = {"status": "built", "sweeps": built, "comparison": comparison}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
