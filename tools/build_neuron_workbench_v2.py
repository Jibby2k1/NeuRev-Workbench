#!/usr/bin/env python3
"""Build the v2 interactive neuron annotation workbench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.workbench.builder import build_workbench, load_workbench_asset, resolve_build_inputs


DEFAULT_APP_DIR = PROJECT_ROOT / "Outputs/NeuronReview/calcium_video_2/app"
DEFAULT_DATA_PATH = DEFAULT_APP_DIR / "review_data.json"
ASSET_DIR = PROJECT_ROOT / "neurobench/workbench/assets"


CSS = load_workbench_asset("workbench.css")
JS = load_workbench_asset("workbench.js")
HTML_TEMPLATE = load_workbench_asset("workbench.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the interactive neuron annotation workbench.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--review-data", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--architecture-runs", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = resolve_build_inputs(
        app_dir=args.app_dir,
        review_data=args.review_data,
        dataset_manifest=args.dataset_manifest,
        architecture_runs=args.architecture_runs,
        default_app_dir=DEFAULT_APP_DIR,
        default_review_data=DEFAULT_DATA_PATH,
        default_dataset_id="calcium_video_2",
    )
    paths = build_workbench(
        app_dir=inputs["app_dir"],
        review_data_path=inputs["review_data_path"],
        dataset_id=inputs["dataset_id"],
        html_template=HTML_TEMPLATE,
        dataset_manifest=inputs["dataset_manifest"],
        architecture_runs_path=inputs["architecture_runs_path"],
        css_fallback=CSS,
        js_fallback=JS,
    )
    print(f"Wrote workbench to {paths['index']}")


if __name__ == "__main__":
    main()
