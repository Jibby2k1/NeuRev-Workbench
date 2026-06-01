#!/usr/bin/env python3
"""Export browser-ready Gamma CFAR contrast maps and attach them to workbench runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurobench.workbench.cfar_contrast_maps import export_cfar_contrast_frames
from neurobench.workbench.intermediates import safe_name


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact_path(run_root: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact.get("path") or ""))
    return path if path.is_absolute() else run_root / path


def artifact_by_id(pipeline_run: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any] | None:
    for artifact in pipeline_run.get("artifacts", []) or []:
        if artifact.get("artifact_id") == artifact_id:
            return artifact
    return None


def pipeline_run_index(sweep_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(sweep_root.glob("*/pipeline_run.json")):
        payload = load_json(path)
        run_id = str(payload.get("run_id") or "")
        if run_id:
            index[run_id] = (path, payload)
    return index


def stage_params(run: Mapping[str, Any], step_id: str) -> dict[str, Any]:
    for stage in run.get("pipeline", []) or []:
        if stage.get("id") == step_id or stage.get("step_id") == step_id:
            return dict(stage.get("params") or {})
    return {}


def cfar_params(
    run: Mapping[str, Any],
    pipeline_run: Mapping[str, Any],
    *,
    step_id: str,
    artifact_id: str,
) -> dict[str, Any]:
    params = stage_params(run, step_id)
    artifact = artifact_by_id(pipeline_run, artifact_id) or {}
    summary = dict(artifact.get("summary") or {})
    merged = dict(summary)
    merged.update(params)
    return {
        "guard_px": int(merged.get("guard_px", 0)),
        "training_radius_px": int(merged.get("training_radius_px", 0)),
        "epsilon": float(merged.get("epsilon", 1e-6)),
        "pfa": merged.get("pfa"),
    }


def shared_key(step_id: str, source_npy: Path, params: Mapping[str, Any]) -> str:
    source_key = safe_name(source_npy.stem)
    return safe_name(
        f"cfar_contrast_{step_id}_{source_key}_guard_{params['guard_px']}"
        f"_radius_{params['training_radius_px']}_eps_{params['epsilon']}"
    )


def contrast_record(
    *,
    step_id: str,
    label: str,
    description: str,
    out_dir: Path,
    app_dir: Path,
    source_npy: Path,
    params: Mapping[str, Any],
    summary: Mapping[str, Any],
    frame_pattern: str,
) -> dict[str, Any]:
    record_summary = dict(summary)
    if params.get("pfa") is not None:
        record_summary["pfa"] = params.get("pfa")
    return {
        "id": step_id,
        "label": label,
        "description": description,
        "stage_id": "gamma_cfar",
        "step_id": step_id,
        "media_type": "frame_sequence",
        "artifact_kind": "cfar_contrast_map",
        "frame_count": int(summary.get("frame_count") or 0),
        "frame_pattern": rel_to(out_dir / frame_pattern, app_dir),
        "source": str(source_npy),
        "summary": record_summary,
    }


def cfar_contrast_specs(run: Mapping[str, Any], pipeline_run: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    candidates = [
        (
            "green_single_cfar",
            "Green-excess single Gamma CFAR contrast",
            "Continuous local contrast score before thresholding the green-excess single Gamma CFAR pass.",
            "green_single_cfar_candidate_mask.v1",
        ),
        (
            "cfar_small_ref",
            "Small-reference Gamma CFAR contrast",
            "Continuous local contrast score before the small-reference Gamma CFAR threshold.",
            "cfar_small_ref_candidate_mask.v1",
        ),
        (
            "cfar_large_ref",
            "Large-reference Gamma CFAR contrast",
            "Continuous local contrast score before the large-reference Gamma CFAR threshold.",
            "cfar_large_ref_candidate_mask.v1",
        ),
    ]
    specs = []
    for step_id, label, description, artifact_id in candidates:
        if artifact_by_id(pipeline_run, artifact_id) or stage_params(run, step_id):
            specs.append((step_id, label, description, artifact_id))
    return specs


def source_smoothed_path(pipeline_run_path: Path, pipeline_run: Mapping[str, Any]) -> Path | None:
    artifact = artifact_by_id(pipeline_run, "smoothed_video.v1")
    if not artifact:
        return None
    path = artifact_path(pipeline_run_path.parent, artifact)
    return path if path.exists() else None


def attach_records(manifest: dict[str, Any], run_id: str, records: list[dict[str, Any]]) -> None:
    for run in manifest.get("runs", []) or []:
        if run.get("run_id") != run_id:
            continue
        artifacts = run.setdefault("artifacts", {})
        existing = list(artifacts.get("intermediates") or [])
        replace_ids = {record["id"] for record in records}
        artifacts["intermediates"] = [item for item in existing if item.get("id") not in replace_ids] + records
        return
    raise SystemExit(f"run_id not found in architecture runs manifest: {run_id}")


def selected_runs(args: argparse.Namespace, manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    requested = set(args.run_id or [])
    runs = list(manifest.get("runs", []) or [])
    if args.all_runs:
        return runs
    if requested:
        return [run for run in runs if run.get("run_id") in requested]
    raise SystemExit("Provide --all-runs or at least one --run-id.")


def export_and_attach(args: argparse.Namespace) -> dict[str, Any]:
    app_dir = args.app_dir.resolve()
    architecture_runs_path = (args.architecture_runs or app_dir / "architecture_runs.json").resolve()
    sweep_root = args.sweep_root.resolve()
    manifest = load_json(architecture_runs_path)
    pipeline_runs = pipeline_run_index(sweep_root)
    generated: dict[str, dict[str, Any]] = {}
    attached = 0
    skipped: list[dict[str, str]] = []

    for run in selected_runs(args, manifest):
        run_id = str(run.get("run_id") or "")
        entry = pipeline_runs.get(run_id)
        if not entry:
            skipped.append({"run_id": run_id, "reason": "pipeline_run.json not found under sweep root"})
            continue
        pipeline_path, pipeline_run = entry
        source = source_smoothed_path(pipeline_path, pipeline_run)
        if not source:
            skipped.append({"run_id": run_id, "reason": "smoothed_video.v1 artifact missing"})
            continue
        specs = cfar_contrast_specs(run, pipeline_run)
        records: list[dict[str, Any]] = []
        for step_id, label, description, artifact_id in specs:
            params = cfar_params(run, pipeline_run, step_id=step_id, artifact_id=artifact_id)
            if params["guard_px"] <= 0 or params["training_radius_px"] <= params["guard_px"]:
                skipped.append({"run_id": run_id, "reason": f"invalid {step_id} CFAR parameters"})
                continue
            key = shared_key(step_id, source, params)
            out_dir = app_dir / "generated_runs" / "_shared_intermediates" / key
            if key not in generated:
                summary = export_cfar_contrast_frames(
                    source_npy=source,
                    out_dir=out_dir,
                    guard_px=params["guard_px"],
                    training_radius_px=params["training_radius_px"],
                    epsilon=params["epsilon"],
                    chunk_frames=args.chunk_frames,
                    sample_stride=args.normalization_sample_stride,
                    frame_pattern=args.frame_pattern,
                    force=args.force,
                )
                generated[key] = {
                    "summary": summary,
                    "out_dir": str(out_dir),
                    "step_id": step_id,
                    "params": params,
                }
            summary = generated[key]["summary"]
            records.append(
                contrast_record(
                    step_id=step_id,
                    label=label,
                    description=description,
                    out_dir=out_dir,
                    app_dir=app_dir,
                    source_npy=source,
                    params=params,
                    summary=summary,
                    frame_pattern=args.frame_pattern,
                )
            )
        if records:
            attach_records(manifest, run_id, records)
            attached += 1

    write_json_atomic(architecture_runs_path, manifest)
    return {
        "architecture_runs": str(architecture_runs_path),
        "sweep_root": str(sweep_root),
        "attached_runs": attached,
        "generated_sequences": len(generated),
        "sequences": generated,
        "skipped": skipped,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True, help="Workbench app directory.")
    parser.add_argument("--sweep-root", type=Path, required=True, help="Gamma CFAR sweep root containing per-run pipeline_run.json files.")
    parser.add_argument("--architecture-runs", type=Path, help="Defaults to app-dir/architecture_runs.json.")
    parser.add_argument("--run-id", action="append", default=None, help="Run id to attach. May be repeated.")
    parser.add_argument("--all-runs", action="store_true", help="Attach contrast maps to all matching runs in architecture_runs.json.")
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--normalization-sample-stride", type=int, default=10)
    parser.add_argument("--frame-pattern", default="frame_%03d.png")
    parser.add_argument("--force", action="store_true", help="Regenerate frames even if an existing summary and first/last frames are present.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = export_and_attach(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
