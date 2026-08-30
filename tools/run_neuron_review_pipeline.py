#!/usr/bin/env python3
"""Run the Fiji/Groovy neuron-review pipeline from a dataset manifest."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.manifests import load_dataset_manifest, manifest_path
from neurobench.pipeline_catalog import normalize_pipeline


DEFAULT_FIJI = Path(os.environ.get("NEUROBENCH_FIJI", shutil.which("fiji") or "fiji"))
DEFAULT_STAGES = [
    "high-pass",
    "event-denoise",
    "candidates",
    "temporal-scoring",
    "review-data",
    "proposal-analysis",
    "workbench",
    "index",
]
RUN_PARAM_ENV = {
    ("temporal_highpass_gaussian", "sigma_frames"): "NEUROBENCH_SIGMA_FRAMES",
    ("robust_positive_local_z", "local_radius_px"): "NEUROBENCH_LOCAL_RADIUS_PX",
    ("robust_positive_local_z", "epsilon"): "NEUROBENCH_EPSILON",
    ("component_filter", "seed_z"): "NEUROBENCH_COMPONENT_SEED_Z",
    ("component_filter", "grow_z"): "NEUROBENCH_COMPONENT_GROW_Z",
    ("component_filter", "min_area_px"): "NEUROBENCH_COMPONENT_MIN_AREA_PX",
    ("component_filter", "max_area_px"): "NEUROBENCH_COMPONENT_MAX_AREA_PX",
    ("local_background_ring", "outer_radius_px"): "NEUROBENCH_BACKGROUND_OUTER_RADIUS_PX",
    ("local_background_ring", "neuropil_weight"): "NEUROBENCH_NEUROPIL_WEIGHT",
    ("robust_kalman_positive_innovation", "event_threshold_z"): "NEUROBENCH_EVENT_THRESHOLD_Z",
    ("robust_kalman_positive_innovation", "kalman_gain"): "NEUROBENCH_KALMAN_GAIN",
    ("robust_kalman_positive_innovation", "spike_gain"): "NEUROBENCH_SPIKE_GAIN",
    ("trace_event_scoring", "event_threshold_z"): "NEUROBENCH_TRACE_EVENT_THRESHOLD_Z",
    ("candidate_event_pipeline", "event_threshold_z"): "NEUROBENCH_EVENT_THRESHOLD_Z",
    ("candidate_event_pipeline", "min_area_px"): "NEUROBENCH_COMPONENT_MIN_AREA_PX",
}
STAGE_ALIASES = {
    "all": DEFAULT_STAGES,
    "highpass": ["high-pass"],
    "denoise": ["event-denoise"],
    "event-preserving-denoise": ["event-denoise"],
    "candidate-event-pipeline": ["candidates"],
    "temporal": ["temporal-scoring"],
    "review": ["review-data"],
    "analysis": ["proposal-analysis"],
    "proposal": ["proposal-analysis"],
    "build": ["workbench"],
}


def quote_cmd(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def split_stages(value: str) -> list[str]:
    requested: list[str] = []
    for raw in value.split(","):
        stage = raw.strip()
        if not stage:
            continue
        requested.extend(STAGE_ALIASES.get(stage, [stage]))
    unknown = [stage for stage in requested if stage not in DEFAULT_STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}")
    return requested


def sigma_label(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric):02d}"
    return f"{int(round(numeric * 10)):03d}"


def threshold_tag(value: Any) -> str:
    return f"{round(float(value) * 10):03d}"


def preset_tag(overrides: Mapping[str, str]) -> str:
    if "NEUROBENCH_COMPONENT_PRESET_NAME" not in overrides:
        return "balanced_seed017_grow009_min3"
    seed = overrides.get("NEUROBENCH_COMPONENT_SEED_Z", "2.0")
    grow = overrides.get("NEUROBENCH_COMPONENT_GROW_Z", "1.1")
    min_area = int(float(overrides.get("NEUROBENCH_COMPONENT_MIN_AREA_PX", "4")))
    return f"{overrides['NEUROBENCH_COMPONENT_PRESET_NAME']}_seed{threshold_tag(seed)}_grow{threshold_tag(grow)}_min{min_area}"


def load_architecture_run(path: Path | None, run_id: str | None) -> dict[str, Any] | None:
    if path is None or run_id is None or not path.exists():
        return None
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    for run in manifest.get("runs", []):
        if run.get("run_id") == run_id:
            return dict(run)
    raise SystemExit(f"Run '{run_id}' was not found in {path}")


def run_env_overrides(run: Mapping[str, Any] | None) -> dict[str, str]:
    if not run:
        return {}
    overrides: dict[str, str] = {
        "NEUROBENCH_RUN_ID": str(run.get("run_id") or ""),
    }
    pipeline = normalize_pipeline(run.get("pipeline") or [])
    for stage in pipeline:
        stage_id = str(stage.get("stage_id") or stage.get("op") or stage.get("name") or "")
        params = dict(stage.get("params") or {})
        for (mapped_stage, param), env_name in RUN_PARAM_ENV.items():
            if mapped_stage == stage_id and param in params:
                overrides[env_name] = str(params[param])
    if "NEUROBENCH_SIGMA_FRAMES" in overrides:
        overrides["NEUROBENCH_SIGMA_LABEL"] = sigma_label(overrides["NEUROBENCH_SIGMA_FRAMES"])
    if any(key.startswith("NEUROBENCH_COMPONENT_") for key in overrides):
        overrides.setdefault("NEUROBENCH_COMPONENT_PRESET_NAME", "run")
    return {key: value for key, value in overrides.items() if value != ""}


def require_outputs(paths: list[Path], *, stage: str, started_at: float) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"{stage} did not produce expected output(s):\n  {joined}")
    if paths and not any(path.stat().st_mtime >= started_at - 2.0 for path in paths):
        joined = "\n  ".join(str(path) for path in paths)
        raise SystemExit(f"{stage} outputs were not refreshed; Fiji may have reported an internal script error:\n  {joined}")


def run(command: list[str], *, env: dict[str, str], dry_run: bool, expected: list[Path] | None = None, stage: str) -> None:
    print("+ " + quote_cmd(command), flush=True)
    if dry_run:
        return
    started_at = time.time()
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    if expected is not None:
        require_outputs(expected, stage=stage, started_at=started_at)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Neurobench Fiji/Groovy review pipeline.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Outputs")
    parser.add_argument("--stages", default="all", help="Comma-separated stage list, or all.")
    parser.add_argument("--architecture-runs", type=Path, default=None, help="Architecture-run manifest used for run-specific parameters.")
    parser.add_argument("--run-id", default=None, help="Run ID inside --architecture-runs to execute.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.dataset_manifest)
    dataset_id = str(manifest["dataset_id"])
    raw_video = manifest_path(manifest, "raw_video")
    app_dir = manifest_path(manifest, "app_dir") or (args.output_root / "NeuronReview" / dataset_id / "app")
    review_data = manifest_path(manifest, "review_data") or (app_dir / "review_data.json")
    annotations = manifest_path(manifest, "annotations") or (app_dir / "annotations.json")
    architecture_runs = manifest_path(manifest, "architecture_runs")
    if args.architecture_runs is not None:
        architecture_runs = args.architecture_runs
    architecture_run = load_architecture_run(architecture_runs, args.run_id)
    run_overrides = run_env_overrides(architecture_run)
    sigma = run_overrides.get("NEUROBENCH_SIGMA_LABEL", "06")
    sigma_name = f"sigma{sigma}"
    component_tag = preset_tag(run_overrides)
    source_z_stack = args.output_root / "CandidateEventPipeline" / dataset_id / f"{dataset_id}_sigma06_robust_positive_z_float32.tif"
    if "NEUROBENCH_SIGMA_LABEL" in run_overrides:
        source_z_stack = args.output_root / "CandidateEventPipeline" / dataset_id / f"{dataset_id}_sigma{sigma}_robust_positive_z_float32.tif"
    high_pass_dir = args.output_root / "HighPass" / dataset_id
    candidate_dir = args.output_root / "CandidateEventPipeline" / dataset_id
    temporal_dir = args.output_root / "TemporalCandidateScoring" / dataset_id

    if raw_video is None:
        raise SystemExit("Manifest is missing paths.raw_video")
    if not raw_video.exists():
        raise SystemExit(f"Raw video not found: {raw_video}")
    if not args.dry_run and not args.fiji.exists():
        raise SystemExit(f"Fiji executable not found: {args.fiji}")

    env = os.environ.copy()
    env.update(
        {
            "NEUROBENCH_PROJECT_ROOT": str(PROJECT_ROOT),
            "NEUROBENCH_DATASET_ID": dataset_id,
            "NEUROBENCH_RAW_VIDEO": str(raw_video),
            "NEUROBENCH_OUTPUT_ROOT": str(args.output_root.resolve()),
            "NEUROBENCH_APP_DIR": str(app_dir),
            "NEUROBENCH_SOURCE_Z_STACK": str(source_z_stack),
        }
    )

    macro_args = ",".join(
        [
            f"project_root={PROJECT_ROOT}",
            f"dataset_id={dataset_id}",
            f"input_path={raw_video}",
            f"output_root={args.output_root.resolve()}",
            *(
                [f"sigma_frames={run_overrides['NEUROBENCH_SIGMA_FRAMES']}", f"sigma_label={run_overrides['NEUROBENCH_SIGMA_LABEL']}"]
                if "NEUROBENCH_SIGMA_FRAMES" in run_overrides
                else []
            ),
        ]
    )
    commands = {
        "high-pass": [
            str(args.fiji),
            "--headless",
            "--console",
            "-macro",
            str(PROJECT_ROOT / "tools/temporal_highpass_gaussian.ijm"),
            macro_args,
        ],
        "event-denoise": [
            str(args.fiji),
            "--headless",
            "--run",
            str(PROJECT_ROOT / "tools/event_preserving_noise_suppression.groovy"),
        ],
        "candidates": [
            str(args.fiji),
            "--headless",
            "--run",
            str(PROJECT_ROOT / "tools/candidate_event_pipeline.groovy"),
        ],
        "temporal-scoring": [
            str(args.fiji),
            "--headless",
            "--run",
            str(PROJECT_ROOT / "tools/temporal_candidate_scoring.groovy"),
        ],
        "review-data": [
            str(args.fiji),
            "--headless",
            "--run",
            str(PROJECT_ROOT / "tools/generate_neuron_review_app.groovy"),
        ],
        "proposal-analysis": [
            sys.executable,
            str(PROJECT_ROOT / "tools/build_proposal_analysis.py"),
            "--review-data",
            str(review_data),
            "--annotations",
            str(annotations),
            "--run-id",
            str(args.run_id or "current_review_pipeline"),
        ],
        "workbench": [
            sys.executable,
            str(PROJECT_ROOT / "tools/build_neuron_workbench_v2.py"),
            "--dataset-manifest",
            str(args.dataset_manifest),
            "--app-dir",
            str(app_dir),
            "--review-data",
            str(review_data),
        ],
        "index": [
            sys.executable,
            str(PROJECT_ROOT / "tools/build_workbench_index.py"),
            "--root",
            str(args.output_root / "NeuronReview"),
            "--out",
            str(args.output_root / "NeuronReview/index.html"),
        ],
    }
    if architecture_runs is not None:
        commands["workbench"].extend(["--architecture-runs", str(architecture_runs)])
        commands["proposal-analysis"].extend(["--architecture-runs", str(architecture_runs)])
    env.update(run_overrides)

    expected_outputs = {
        "high-pass": [
            *(
                [high_pass_dir / f"{dataset_id}_hp_gaussian_sigma{sigma}f_float32.tif"]
                if "NEUROBENCH_SIGMA_LABEL" in run_overrides
                else [
                    high_pass_dir / f"{dataset_id}_hp_gaussian_sigma04f_float32.tif",
                    high_pass_dir / f"{dataset_id}_hp_gaussian_sigma06f_float32.tif",
                    high_pass_dir / f"{dataset_id}_hp_gaussian_sigma08f_float32.tif",
                ]
            ),
        ],
        "event-denoise": [
            args.output_root / "EventPreservingNoiseSuppression" / dataset_id / f"{dataset_id}_{sigma_name}_positive_local_z_float32.tif",
        ],
        "candidates": [
            source_z_stack,
            candidate_dir / "candidate_events.tsv",
        ],
        "temporal-scoring": [
            temporal_dir / f"{dataset_id}_{sigma_name}_{component_tag}_score_ge_050_mask.tif",
            temporal_dir / "temporal_candidates.tsv",
        ],
        "review-data": [
            review_data,
            app_dir / "roi_summary.tsv",
            app_dir / "frames" / "frame_001.png",
        ],
        "proposal-analysis": [
            app_dir / "analysis" / "proposal_analysis.json",
            app_dir / "analysis" / "artifact_classifier.tsv",
            app_dir / "analysis" / "missed_neuron_proposals.tsv",
        ],
        "workbench": [
            app_dir / "index.html",
            app_dir / "workbench.js",
            app_dir / "annotations.json",
        ],
        "index": [
            args.output_root / "NeuronReview/index.html",
        ],
    }

    stages = split_stages(args.stages)
    print(f"Dataset: {dataset_id}", flush=True)
    print(f"Raw video: {raw_video}", flush=True)
    print(f"App dir: {app_dir}", flush=True)
    for stage in stages:
        run(
            commands[stage],
            env=env,
            dry_run=args.dry_run,
            expected=expected_outputs.get(stage),
            stage=stage,
        )


if __name__ == "__main__":
    main()
