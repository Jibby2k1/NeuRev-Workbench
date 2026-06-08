#!/usr/bin/env python3
"""Prepare cropped 512x512, 32x32-grid dynamics inputs for 060126 fish videos."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.algorithms.grid_regions import write_grid_spec_artifacts, write_registered_grid_state_artifacts
from neurobench.algorithms.template_matching import write_registration_artifacts, write_template_artifacts
from neurobench.data.crop import CropBox, crop_video_stack, validate_crop_box, write_crop_manifest
from neurobench.data.video import video_metadata
from neurobench.data.video_manifest import build_video_manifest, video_by_id
from neurobench.dynamics.datasets import build_dynamics_dataset
from neurobench.dynamics.train import train_autoencoder

DEFAULT_CROP = CropBox(x0=81, y0=115, x1=593, y1=627)
VIDEO_REGEX = r"^(?P<index>[0-9]+)\s+(?P<label>left|right|rest|resting|neutral)\.(?:tif|tiff|npy)$"
LABEL_ALIASES = {"rest": "neutral", "resting": "neutral"}
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


def _slug_float(value: float) -> str:
    return (f"{float(value):.0e}" if float(value) < 0.001 else f"{float(value):.4f}").replace(".", "p").replace("-", "m")


def _video_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".tif", ".tiff"})


def validate_inputs(*, input_dir: Path, crop: CropBox) -> dict[str, Any]:
    paths = _video_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No TIFF videos found in {input_dir}.")
    videos = []
    for path in paths:
        meta = video_metadata(path)
        validate_crop_box(crop, source_width=int(meta["width"]), source_height=int(meta["height"]))
        videos.append({"path": str(path), "shape": [int(v) for v in meta["shape"]], "dtype": str(meta["dtype"])})
    return {"schema_version": 1, "input_dir": str(input_dir), "crop_box": crop.as_dict(), "video_count": len(videos), "videos": videos}


def prepare_cropped_grid32_run(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    crop: CropBox = DEFAULT_CROP,
    reference_video_id: str = "1 resting",
    chunk_size_frames: int = 64,
    registration_device: str = "auto",
    autoencoder_device: str = "cuda",
    autoencoder_epochs: int = 60,
    autoencoder_batch_size: int = 128,
    autoencoder_learning_rate: float = 1e-3,
    skip_autoencoders: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_dir)
    out = Path(out_dir)
    cropped_dir = out / "cropped_videos"
    manifest_dir = out / "manifest"
    template_dir = out / "template"
    registration_dir = out / "registration"
    grid_dir = out / "grid"
    grid_states_dir = out / "grid_states"
    datasets_dir = out / "datasets"
    models_dir = out / "models"

    input_summary = validate_inputs(input_dir=input_root, crop=crop)
    crop_summaries = []
    for source in _video_paths(input_root):
        crop_summaries.append(
            crop_video_stack(
                source_path=source,
                output_path=cropped_dir / source.name,
                crop=crop,
                chunk_size_frames=chunk_size_frames,
            )
        )
    write_crop_manifest(summaries=crop_summaries, output_path=out / "crop_manifest.json", crop=crop)

    manifest = build_video_manifest(
        input_dir=cropped_dir,
        dataset_id="fish_060126_crop512_grid32",
        filename_regex=VIDEO_REGEX,
        label_aliases=LABEL_ALIASES,
        labels=("left", "right", "neutral"),
        strict=True,
    )
    manifest_path = manifest_dir / "video_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(manifest_dir / "label_counts.json", manifest.get("label_counts") or {})

    reference = video_by_id(manifest, reference_video_id)
    write_template_artifacts(
        video_path=reference["path"],
        source_video_id=reference_video_id,
        out_dir=template_dir,
        outlier_rejection=True,
        max_outlier_fraction=0.05,
        z_threshold=3.5,
        chunk_size_frames=chunk_size_frames,
    )
    template = _load_json(template_dir / "template_spec.json")

    registration_results = []
    for video in manifest.get("videos", []) or []:
        registration_results.append(
            write_registration_artifacts(
                video_path=video["path"],
                video_id=str(video["video_id"]),
                template_spec=template,
                out_dir=registration_dir,
                transform_model="rigid",
                rotation_range_deg=(-10.0, 10.0),
                rotation_step_deg=0.5,
                allow_uniform_scale=False,
                chunk_size_frames=chunk_size_frames,
            )
        )
    registration_summary = {
        "schema_version": 1,
        "registration_dir": str(registration_dir),
        "video_count": len(registration_results),
        "warnings": sum(len(item.get("qc", {}).get("warnings") or []) for item in registration_results),
        "results": [
            {
                "video_id": item["video_id"],
                "result": str(registration_dir / item["video_id"] / "registration_result.json"),
                "score": item.get("score"),
                "qc": item.get("qc"),
            }
            for item in registration_results
        ],
    }
    _write_json(registration_dir / "registration_summary.json", registration_summary)

    grid_spec_path = grid_dir / "grid_spec_32x32.json"
    grid_spec = write_grid_spec_artifacts(template_spec=template, out_path=grid_spec_path, rows=32, cols=32)

    grid_summaries = []
    for video in manifest.get("videos", []) or []:
        registration_result = _load_json(registration_dir / str(video["video_id"]) / "registration_result.json")
        grid_summaries.append(
            write_registered_grid_state_artifacts(
                video_path=video["path"],
                registration_result=registration_result,
                grid_spec=grid_spec,
                out_dir=grid_states_dir,
                video_id=str(video["video_id"]),
                label=str(video.get("label") or ""),
                features=("mean_intensity",),
                normalization="per_video_robust_percentile",
                frame_rate_hz=video.get("frame_rate_hz"),
                chunk_size_frames=chunk_size_frames,
                max_grid_state_bytes=1_000_000_000,
                device=registration_device,
            )
        )
    grid_summary = {"schema_version": 1, "grid_states_dir": str(grid_states_dir), "video_count": len(grid_summaries), "videos": grid_summaries}
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

    ae_runs: dict[str, dict[str, Any]] = {}
    if not skip_autoencoders:
        ae_s1_dir = models_dir / f"autoencoder32_s1_ld64_bc16_e{int(autoencoder_epochs)}_lr{_slug_float(autoencoder_learning_rate)}_v1"
        ae_s3_dir = models_dir / f"autoencoder32_s3_ld64_bc16_e{int(autoencoder_epochs)}_lr{_slug_float(autoencoder_learning_rate)}_v1"
        ae_runs["s1"] = train_autoencoder(
            dataset=datasets["w8_s1_h25"],
            out_dir=ae_s1_dir,
            latent_dim=64,
            base_channels=16,
            epochs=int(autoencoder_epochs),
            batch_size=int(autoencoder_batch_size),
            learning_rate=float(autoencoder_learning_rate),
            seed=7,
            device=autoencoder_device,
        )
        ae_runs["s3"] = train_autoencoder(
            dataset=datasets["w8_s3_h10"],
            out_dir=ae_s3_dir,
            latent_dim=64,
            base_channels=16,
            epochs=int(autoencoder_epochs),
            batch_size=int(autoencoder_batch_size),
            learning_rate=float(autoencoder_learning_rate),
            seed=7,
            device=autoencoder_device,
        )

    mapping = {}
    for key, dataset in datasets.items():
        ae_key = "s3" if key == "w8_s3_h10" else "s1"
        mapping[key] = {
            "dataset": str(datasets_dir / key / "dynamics_dataset.json"),
            "autoencoder_run": str(models_dir / (f"autoencoder32_{ae_key}_ld64_bc16_e{int(autoencoder_epochs)}_lr{_slug_float(autoencoder_learning_rate)}_v1") / "autoencoder_run.json"),
            "window_frames": int(dataset.get("windowing", {}).get("window_frames", 8)),
        }
    mapping_path = out / "datasets_cropped32_mapping.json"
    _write_json(mapping_path, mapping)

    summary = {
        "schema_version": 1,
        "status": "prepared",
        "input_summary": input_summary,
        "crop_manifest_path": str(out / "crop_manifest.json"),
        "video_manifest_path": str(manifest_path),
        "template_spec_path": str(template_dir / "template_spec.json"),
        "registration_summary_path": str(registration_dir / "registration_summary.json"),
        "grid_spec_path": str(grid_spec_path),
        "grid_states_summary_path": str(grid_states_dir / "grid_states_summary.json"),
        "datasets_mapping_path": str(mapping_path),
        "dataset_keys": list(mapping),
        "autoencoder_runs": {key: str(value.get("checkpoint_path")) for key, value in ae_runs.items()},
        "skip_autoencoders": bool(skip_autoencoders),
    }
    _write_json(out / "preprocess_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare cropped 512x512 32x32-grid dynamics artifacts.")
    parser.add_argument("--input-dir", type=Path, default=Path("Inputs/060126"))
    parser.add_argument("--out-dir", type=Path, default=Path("Outputs/GridModel/060126_crop512_grid32_v1"))
    parser.add_argument("--crop-x0", type=int, default=DEFAULT_CROP.x0)
    parser.add_argument("--crop-y0", type=int, default=DEFAULT_CROP.y0)
    parser.add_argument("--crop-x1", type=int, default=DEFAULT_CROP.x1)
    parser.add_argument("--crop-y1", type=int, default=DEFAULT_CROP.y1)
    parser.add_argument("--reference-video-id", default="1 resting")
    parser.add_argument("--chunk-size-frames", type=int, default=64)
    parser.add_argument("--registration-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--autoencoder-device", default="cuda")
    parser.add_argument("--autoencoder-epochs", type=int, default=60)
    parser.add_argument("--autoencoder-batch-size", type=int, default=128)
    parser.add_argument("--autoencoder-learning-rate", type=float, default=1e-3)
    parser.add_argument("--skip-autoencoders", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    crop = CropBox(args.crop_x0, args.crop_y0, args.crop_x1, args.crop_y1)
    if args.dry_run:
        print(json.dumps(validate_inputs(input_dir=args.input_dir, crop=crop), indent=2, sort_keys=True))
        return 0
    summary = prepare_cropped_grid32_run(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        crop=crop,
        reference_video_id=args.reference_video_id,
        chunk_size_frames=args.chunk_size_frames,
        registration_device=args.registration_device,
        autoencoder_device=args.autoencoder_device,
        autoencoder_epochs=args.autoencoder_epochs,
        autoencoder_batch_size=args.autoencoder_batch_size,
        autoencoder_learning_rate=args.autoencoder_learning_rate,
        skip_autoencoders=args.skip_autoencoders,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
