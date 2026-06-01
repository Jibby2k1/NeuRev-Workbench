"""Helpers for exporting pipeline artifacts into Process Lab intermediates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib


FRAME_EXPORT_KINDS = {
    "highpass_video": "High-pass video",
    "denoised_video": "Denoised video",
    "z_stack": "Activity evidence z-stack",
    "smoothed_video": "Smoothed video",
    "registered_video": "Registered video",
    "candidate_mask": "Candidate mask",
}


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("NumPy is required to export intermediate frame stacks.") from exc
    return np


def normalize_array_frame(frame) -> bytes:
    """Normalize one 2-D frame to 8-bit grayscale PNG bytes."""
    np = _load_numpy()
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D frame, got shape {arr.shape}.")
    finite = np.isfinite(arr)
    if not np.any(finite):
        return bytes(arr.size)
    values = arr[finite]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        clipped = np.clip(np.where(finite, arr, lo), lo, hi)
        scaled = np.round((clipped - lo) * 255.0 / (hi - lo)).astype(np.uint8)
    return scaled.tobytes()


def write_png_gray8(path: Path, width: int, height: int, pixels: bytes) -> None:
    """Write an 8-bit grayscale PNG using only the Python standard library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    if len(pixels) != width * height:
        raise ValueError(f"Pixel buffer has {len(pixels)} bytes, expected {width * height}.")
    scanlines = b"".join(b"\x00" + pixels[row * width : (row + 1) * width] for row in range(height))
    path.write_bytes(
        b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
                chunk(b"IDAT", zlib.compress(scanlines, level=6)),
                chunk(b"IEND", b""),
            ]
        )
    )


def frame_output_path(out_dir: Path, pattern: str, index: int) -> Path:
    if "%" in pattern:
        try:
            return out_dir / (pattern % index)
        except TypeError:
            pass
    return out_dir / pattern.replace("%03d", f"{index:03d}").replace("{frame}", str(index)).replace("{frame:03d}", f"{index:03d}")


def relative_pattern(out_dir: Path, pattern: str, manifest_path: Path | None) -> str:
    pattern_path = out_dir / pattern
    if manifest_path is None:
        return pattern_path.as_posix()
    try:
        return pattern_path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    except ValueError:
        return pattern_path.resolve().as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value).strip("._-")
    return cleaned or "artifact"


def iter_npy_frames(path: Path):
    np = _load_numpy()
    stack = np.load(path)
    if stack.ndim == 2:
        yield stack
        return
    if stack.ndim in {3, 4}:
        for frame in stack:
            yield frame
        return
    raise ValueError(f"Expected a 2-D, 3-D, or 4-D .npy array, got shape {stack.shape}.")


def iter_tif_frames(path: Path):
    try:
        import tifffile
    except ModuleNotFoundError:
        tifffile = None
    if tifffile is not None:
        stack = tifffile.imread(path)
        if getattr(stack, "ndim", 0) == 2:
            yield stack
        else:
            for frame in stack:
                yield frame
        return

    try:
        from PIL import Image, ImageSequence
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TIFF export requires optional dependency 'tifffile' or 'Pillow'. "
            "Use --input-npy for local executor artifacts without extra dependencies."
        ) from exc
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            yield frame


def export_frames(input_path: Path, out_dir: Path, frame_pattern: str, *, input_kind: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter_npy_frames(input_path) if input_kind == "npy" else iter_tif_frames(input_path)
    count = 0
    for count, frame in enumerate(iterator, start=1):
        np = _load_numpy()
        arr = np.asarray(frame)
        height, width = arr.shape[:2]
        out = frame_output_path(out_dir, frame_pattern, count)
        write_png_gray8(out, int(width), int(height), normalize_array_frame(frame))
    return count


def attach_artifact(manifest: dict[str, Any], run_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    for run in manifest.get("runs", []):
        if run.get("run_id") != run_id:
            continue
        artifacts = run.setdefault("artifacts", {})
        items = list(artifacts.get("intermediates") or [])
        artifact_id = artifact.get("id")
        artifacts["intermediates"] = [item for item in items if item.get("id") != artifact_id] + [artifact]
        return manifest
    raise SystemExit(f"run_id not found in manifest: {run_id}")


def export_intermediate_stack(
    *,
    input_path: Path,
    input_kind: str,
    out_dir: Path,
    stage_id: str,
    step_id: str | None = None,
    label: str | None = None,
    description: str | None = None,
    run_id: str | None = None,
    architecture_runs_path: Path | None = None,
    frame_pattern: str = "frame_%03d.png",
) -> dict[str, Any]:
    count = export_frames(input_path, out_dir, frame_pattern, input_kind=input_kind)
    artifact = {
        "id": step_id or stage_id,
        "label": label or stage_id.replace("_", " "),
        "stage_id": stage_id,
        "step_id": step_id or stage_id,
        "media_type": "frame_sequence",
        "frame_count": count,
        "frame_pattern": relative_pattern(out_dir, frame_pattern, architecture_runs_path),
        "source": str(input_path),
    }
    if description:
        artifact["description"] = description
    if architecture_runs_path and run_id:
        manifest = attach_artifact(load_json(architecture_runs_path), run_id, artifact)
        write_json_atomic(architecture_runs_path, manifest)
    return artifact


def artifact_path(run_root: Path, artifact: dict[str, Any]) -> Path:
    path = Path(str(artifact.get("path") or ""))
    return path if path.is_absolute() else run_root / path


def label_for_artifact(artifact: dict[str, Any]) -> str:
    kind = str(artifact.get("kind") or "")
    return FRAME_EXPORT_KINDS.get(kind, kind.replace("_", " ").title() or str(artifact.get("producer_stage") or "Intermediate"))


def intermediate_record(
    *,
    artifact: dict[str, Any],
    out_dir: Path,
    frame_pattern: str,
    frame_count: int,
    architecture_runs: Path,
) -> dict[str, Any]:
    stage_id = str(artifact.get("producer_stage") or artifact.get("kind") or "intermediate")
    kind = str(artifact.get("kind") or stage_id)
    record: dict[str, Any] = {
        "id": safe_name(f"{stage_id}_{kind}"),
        "label": label_for_artifact(artifact),
        "stage_id": stage_id,
        "step_id": stage_id,
        "media_type": "frame_sequence",
        "frame_count": int(frame_count),
        "frame_pattern": relative_pattern(out_dir, frame_pattern, architecture_runs),
        "source": str(artifact.get("path") or ""),
        "description": f"Exported from pipeline artifact {artifact.get('artifact_id') or kind}.",
    }
    summary = artifact.get("summary")
    if isinstance(summary, dict) and summary:
        record["summary"] = summary
    return record


def attach_intermediates(architecture_runs: dict[str, Any], run_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    for run in architecture_runs.get("runs", []):
        if run.get("run_id") != run_id:
            continue
        artifacts = run.setdefault("artifacts", {})
        existing = list(artifacts.get("intermediates") or [])
        replace_ids = {record["id"] for record in records}
        artifacts["intermediates"] = [item for item in existing if item.get("id") not in replace_ids] + records
        return architecture_runs
    raise SystemExit(f"run_id not found in architecture runs manifest: {run_id}")


def attach_pipeline_intermediates(
    *,
    pipeline_run_path: Path,
    architecture_runs_path: Path,
    run_id: str | None = None,
    out_root: Path | None = None,
    frame_pattern: str = "frame_%03d.png",
    include_kinds: set[str] | None = None,
) -> dict[str, Any]:
    pipeline_run = load_json(pipeline_run_path)
    run_root = pipeline_run_path.parent
    target_run_id = run_id or str(pipeline_run.get("run_id") or "")
    if not target_run_id:
        raise SystemExit("run_id was not provided and pipeline_run.json has no run_id.")
    kinds = include_kinds or set(FRAME_EXPORT_KINDS)
    export_root = out_root or architecture_runs_path.parent / "generated_runs" / safe_name(target_run_id) / "intermediates"
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for artifact in pipeline_run.get("artifacts", []) or []:
        kind = str(artifact.get("kind") or "")
        source = artifact_path(run_root, artifact)
        if kind not in kinds:
            skipped.append({"kind": kind, "reason": "kind not selected"})
            continue
        if source.suffix.lower() != ".npy":
            skipped.append({"kind": kind, "reason": "not a .npy artifact", "path": str(source)})
            continue
        if not source.is_file():
            skipped.append({"kind": kind, "reason": "artifact path missing", "path": str(source)})
            continue
        stage_id = safe_name(str(artifact.get("producer_stage") or kind))
        out_dir = export_root / stage_id
        frame_count = export_frames(source, out_dir, frame_pattern, input_kind="npy")
        records.append(
            intermediate_record(
                artifact=artifact,
                out_dir=out_dir,
                frame_pattern=frame_pattern,
                frame_count=frame_count,
                architecture_runs=architecture_runs_path,
            )
        )

    if records:
        architecture_runs = attach_intermediates(load_json(architecture_runs_path), target_run_id, records)
        write_json_atomic(architecture_runs_path, architecture_runs)

    return {
        "pipeline_run": str(pipeline_run_path),
        "architecture_runs": str(architecture_runs_path),
        "run_id": target_run_id,
        "exported_count": len(records),
        "intermediates": records,
        "skipped": skipped,
    }


def add_export_intermediate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-tif", type=Path, help="Input TIFF stack. Requires optional tifffile or Pillow.")
    inputs.add_argument("--input-npy", type=Path, help="Input .npy array with shape HxW, TxHxW, or TxHxWxC.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--step-id", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--architecture-runs", type=Path, default=None)
    parser.add_argument("--frame-pattern", default="frame_%03d.png")
    return parser


def export_intermediate_command(args: argparse.Namespace) -> int:
    input_path = args.input_npy or args.input_tif
    input_kind = "npy" if args.input_npy else "tif"
    assert input_path is not None
    artifact = export_intermediate_stack(
        input_path=input_path,
        input_kind=input_kind,
        out_dir=args.out_dir,
        stage_id=args.stage_id,
        step_id=args.step_id,
        label=args.label,
        description=args.description,
        run_id=args.run_id,
        architecture_runs_path=args.architecture_runs,
        frame_pattern=args.frame_pattern,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def add_attach_intermediates_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--pipeline-run", type=Path, required=True, help="Path to pipeline_run.json from a local executor run.")
    parser.add_argument("--architecture-runs", type=Path, required=True, help="Workbench architecture_runs.json to update.")
    parser.add_argument("--run-id", default=None, help="Workbench run_id to attach to. Defaults to pipeline_run.json run_id.")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root for generated frame folders.")
    parser.add_argument("--frame-pattern", default="frame_%03d.png")
    parser.add_argument(
        "--include-kind",
        action="append",
        default=None,
        help="Artifact kind to export. May be repeated. Defaults to common frame-like artifact kinds.",
    )
    return parser


def attach_intermediates_command(args: argparse.Namespace) -> int:
    result = attach_pipeline_intermediates(
        pipeline_run_path=args.pipeline_run,
        architecture_runs_path=args.architecture_runs,
        run_id=args.run_id,
        out_root=args.out_root,
        frame_pattern=args.frame_pattern,
        include_kinds=set(args.include_kind or ()) or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
