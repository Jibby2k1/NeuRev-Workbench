#!/usr/bin/env python3
"""Prepare crop512 grid-dynamics inputs at an arbitrary square grid size."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.algorithms.grid_regions import write_grid_spec_artifacts, write_registered_grid_state_artifacts
from neurobench.dynamics.datasets import build_dynamics_dataset
from neurobench.dynamics.scalable import write_architecture_catalog

DATASET_SPECS = (
    {"key": "w8_s1_h25", "window_frames": 8, "prediction_horizon_frames": 25, "temporal_stride_frames": 1},
    {"key": "w8_s1_h50", "window_frames": 8, "prediction_horizon_frames": 50, "temporal_stride_frames": 1},
    {"key": "w8_s3_h10", "window_frames": 8, "prediction_horizon_frames": 10, "temporal_stride_frames": 3},
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def validate_source_root(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    required = {
        "template_spec": root / "template" / "template_spec.json",
        "video_manifest": root / "manifest" / "video_manifest.json",
        "registration_summary": root / "registration" / "registration_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing crop512 source artifacts: " + ", ".join(missing))
    manifest = _load_json(required["video_manifest"])
    registration = _load_json(required["registration_summary"])
    videos = manifest.get("videos") or []
    registered = {str(item.get("video_id")) for item in registration.get("results") or []}
    missing_registrations = [str(video.get("video_id")) for video in videos if str(video.get("video_id")) not in registered]
    if missing_registrations:
        raise FileNotFoundError("Missing registration results for videos: " + ", ".join(missing_registrations))
    return {
        "schema_version": 1,
        "source_root": str(root),
        "video_count": len(videos),
        "template_spec_path": str(required["template_spec"]),
        "video_manifest_path": str(required["video_manifest"]),
        "registration_summary_path": str(required["registration_summary"]),
        "registration_count": len(registered),
    }


def prepare_crop512_grid_run(
    *,
    source_root: str | Path,
    out_dir: str | Path,
    grid_size: int,
    chunk_size_frames: int = 64,
    grid_device: str = "auto",
    max_grid_state_bytes: int | None = None,
) -> dict[str, Any]:
    source = Path(source_root)
    out = Path(out_dir)
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    source_summary = validate_source_root(source)
    manifest_dir = out / "manifest"
    template_dir = out / "template"
    registration_dir = out / "registration"
    grid_dir = out / "grid"
    grid_states_dir = out / "grid_states"
    datasets_dir = out / "datasets"

    template = _load_json(source / "template" / "template_spec.json")
    manifest = _load_json(source / "manifest" / "video_manifest.json")
    registration_summary = _load_json(source / "registration" / "registration_summary.json")
    manifest = dict(manifest)
    manifest["dataset_id"] = f"fish_060126_crop512_grid{grid_size}"

    _write_json(manifest_dir / "video_manifest.json", manifest)
    _copy_if_exists(source / "manifest" / "label_counts.json", manifest_dir / "label_counts.json")
    shutil.copytree(source / "template", template_dir, dirs_exist_ok=True)
    _write_json(registration_dir / "registration_summary.json", registration_summary)
    _copy_if_exists(source / "crop_manifest.json", out / "crop_manifest.json")

    grid_spec_path = grid_dir / f"grid_spec_{grid_size}x{grid_size}.json"
    grid_spec = write_grid_spec_artifacts(template_spec=template, out_path=grid_spec_path, rows=grid_size, cols=grid_size)

    registration_by_video = {str(item.get("video_id")): Path(str(item.get("result"))) for item in registration_summary.get("results") or []}
    max_bytes = int(max_grid_state_bytes or max(1_000_000_000, grid_size * grid_size * 11 * 4 * 4096))
    grid_summaries = []
    for video in manifest.get("videos", []) or []:
        video_id = str(video["video_id"])
        registration_result_path = registration_by_video[video_id]
        registration_result = _load_json(registration_result_path)
        grid_summaries.append(
            write_registered_grid_state_artifacts(
                video_path=video["path"],
                registration_result=registration_result,
                grid_spec=grid_spec,
                out_dir=grid_states_dir,
                video_id=video_id,
                label=str(video.get("label") or ""),
                features=("mean_intensity",),
                normalization="per_video_robust_percentile",
                frame_rate_hz=video.get("frame_rate_hz"),
                chunk_size_frames=chunk_size_frames,
                max_grid_state_bytes=max_bytes,
                device=grid_device,
            )
        )
    grid_summary = {
        "schema_version": 1,
        "grid_states_dir": str(grid_states_dir),
        "grid_size": int(grid_size),
        "video_count": len(grid_summaries),
        "source_registration_summary_path": str(source / "registration" / "registration_summary.json"),
        "videos": grid_summaries,
    }
    _write_json(grid_states_dir / "grid_states_summary.json", grid_summary)

    datasets: dict[str, dict[str, Any]] = {}
    for spec in DATASET_SPECS:
        dataset_out = datasets_dir / str(spec["key"])
        dataset = build_dynamics_dataset(
            manifest=manifest,
            grid_states_dir=grid_states_dir,
            out_dir=dataset_out,
            window_frames=int(spec["window_frames"]),
            prediction_horizon_frames=int(spec["prediction_horizon_frames"]),
            temporal_stride_frames=int(spec["temporal_stride_frames"]),
            split_method="stratified_by_label",
        )
        datasets[str(spec["key"])] = dataset

    mapping = {}
    for key, dataset in datasets.items():
        mapping[key] = {
            "dataset": str(datasets_dir / key / "dynamics_dataset.json"),
            "window_frames": int(dataset.get("windowing", {}).get("window_frames", 8)),
            "grid_size": int(grid_size),
        }
    mapping_path = out / f"datasets_crop512_grid{grid_size}_mapping.json"
    _write_json(mapping_path, mapping)
    write_architecture_catalog(out / "architecture_catalog.json", input_channels=1, window_frames=8, grid_sizes=(grid_size,))

    summary = {
        "schema_version": 1,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": source_summary,
        "source_root": str(source),
        "grid_size": int(grid_size),
        "video_manifest_path": str(manifest_dir / "video_manifest.json"),
        "template_spec_path": str(template_dir / "template_spec.json"),
        "registration_summary_path": str(registration_dir / "registration_summary.json"),
        "source_registration_summary_path": str(source / "registration" / "registration_summary.json"),
        "grid_spec_path": str(grid_spec_path),
        "grid_states_summary_path": str(grid_states_dir / "grid_states_summary.json"),
        "datasets_mapping_path": str(mapping_path),
        "dataset_keys": list(mapping),
        "architecture_catalog_path": str(out / "architecture_catalog.json"),
        "autoencoders_required": False,
        "chunk_size_frames": int(chunk_size_frames),
        "grid_device": str(grid_device),
        "max_grid_state_bytes": int(max_bytes),
    }
    _write_json(out / "preprocess_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("Outputs/GridModel/060126_crop512_grid32_v1"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--chunk-size-frames", type=int, default=64)
    parser.add_argument("--grid-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-grid-state-bytes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        summary = validate_source_root(args.source_root)
        summary["requested_grid_size"] = int(args.grid_size)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    summary = prepare_crop512_grid_run(
        source_root=args.source_root,
        out_dir=args.out_dir,
        grid_size=args.grid_size,
        chunk_size_frames=args.chunk_size_frames,
        grid_device=args.grid_device,
        max_grid_state_bytes=args.max_grid_state_bytes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
