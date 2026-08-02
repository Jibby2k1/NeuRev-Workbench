"""Integrity-check and finalize the gated source-separation program disposition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .screen_runner import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(calibration_root: Path, video_root: Path, output_root: Path) -> dict[str, Any]:
    calibration = calibration_root.resolve()
    videos = video_root.resolve()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(f"program disposition exists: {output}")
    calibration_path = calibration / "metrics.json"
    video_path = videos / "manifest.json"
    calibration_metrics = json.loads(calibration_path.read_text(encoding="utf-8"))
    video_manifest = json.loads(video_path.read_text(encoding="utf-8"))
    if calibration_metrics["status"] != "identifiability_calibration_complete":
        raise RuntimeError("calibration is not complete")
    if video_manifest["status"] != "diagnostic_video_suite_complete":
        raise RuntimeError("diagnostic suite is not complete")
    video_rows = video_manifest["generated_videos"] + video_manifest["spon_videos"]
    integrity = []
    for row in video_rows:
        path = videos / str(row["path"])
        digest = _sha256(path) if path.is_file() else None
        integrity.append({
            "path": str(path), "exists": path.is_file(),
            "bytes_match": path.is_file() and path.stat().st_size == int(row["bytes"]),
            "sha256_match": digest == row["sha256"],
            "even_dimensions": int(row["probe"]["width"]) % 2 == 0 and int(row["probe"]["height"]) % 2 == 0,
            "frame_count_positive": int(row["probe"]["frame_count"]) > 0,
        })
    video_integrity_passed = all(all(item[key] for key in ("exists", "bytes_match", "sha256_match", "even_dimensions", "frame_count_positive")) for item in integrity)
    passing = list(calibration_metrics["passing_methods"])
    gate_passed = bool(passing)
    stages = [
        {"stage": "repaired_identifiability_calibration", "status": "complete", "executed": True},
        {"stage": "generated_confirmation", "status": "gated_not_run" if not gate_passed else "selectable", "executed": False, "reason": "no method passed held-family identifiability" if not gate_passed else None},
        {"stage": "semi_synthetic_spon", "status": "gated_not_run", "executed": False, "reason": "requires a generated-confirmation survivor"},
        {"stage": "caiman_cnmf_fit", "status": "gated_not_run", "executed": False, "reason": "CaImAn is installed/verified, but fitting requires the semi-synthetic comparison gate"},
        {"stage": "full_spon_benchmark", "status": "gated_not_run", "executed": False, "reason": "requires G0-G2 survivors and exhaustive bounded-field labels"},
        {"stage": "diagnostic_video_suite", "status": "complete", "executed": True},
    ]
    payload = {
        "schema_version": 1, "status": "program_checkpoint_complete",
        "calibration_root": str(calibration), "video_root": str(videos),
        "calibration_metrics_sha256": _sha256(calibration_path),
        "video_manifest_sha256": _sha256(video_path),
        "passing_methods": passing, "advancement_gate_passed": gate_passed,
        "stages": stages, "video_count": len(video_rows),
        "generated_video_count": len(video_manifest["generated_videos"]),
        "spon_video_count": len(video_manifest["spon_videos"]),
        "video_integrity_passed": video_integrity_passed,
        "video_integrity": integrity,
        "interpretation": (
            "Every selected stage reached a terminal scientific state. Dependent fits "
            "were stopped by the frozen gate rather than bypassed. Gated-not-run is a "
            "completed disposition, not evidence of model performance."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "metrics.json", payload)
    return payload
