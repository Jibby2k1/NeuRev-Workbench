"""Validate and summarize the completed quantized-coactivity experiment."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .artifacts import atomic_json
from .quantized_coactivity import _load, _root


def _decode(path: Path) -> tuple[bool, str]:
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    return run.returncode == 0, run.stderr.strip()


def _montage(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    canvas.save(output)


def validate(config_path: str | Path) -> dict:
    config = _load(config_path)
    root = _root(config)
    audit = root / "scientific_audit"
    payload = json.loads((root / "results.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    lanes = payload["lanes"]

    table_path = root / "lane_metrics.csv"
    fields = ["lane_id", "family", "matched_at_58", "integrity_correlation",
              "label_free_score", "zero_fraction", "tie_fraction", "config_json"]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lane in lanes:
            metrics = lane["metrics"]
            writer.writerow({
                "lane_id": lane["lane_id"], "family": lane["family"],
                "matched_at_58": lane["expert"]["matched_by_budget"]["58"],
                "integrity_correlation": metrics["integrity_correlation"],
                "label_free_score": metrics["score"],
                "zero_fraction": metrics["zero_fraction"],
                "tie_fraction": metrics["sample_tie_fraction"],
                "config_json": json.dumps(lane["config"], sort_keys=True),
            })

    colors = {"control": "#4a5568", "quantization": "#3182ce",
              "neighborhood": "#38a169", "combined": "#dd6b20"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, color in colors.items():
        rows = [lane for lane in lanes if lane["family"] == family]
        ax.scatter([lane["metrics"]["integrity_correlation"] for lane in rows],
                   [lane["expert"]["matched_by_budget"]["58"] for lane in rows],
                   s=20, alpha=0.65, color=color, label=family)
    ax.axhline(summary["float_control_matches"], color="black", linestyle="--", linewidth=1)
    ax.axvline(0.95, color="black", linestyle=":", linewidth=1)
    ax.set(xlabel="Correlation with float GN", ylabel="Known matches at budget 58",
           title="Detection yield versus signal integrity")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(root / "integrity_vs_recall.png", dpi=150); plt.close(fig)

    names = ["Float GN", "Protected all", "Protected neighborhood", "Protected Q",
             "Protected integrity", "Frozen selector", "Post-hoc ceiling"]
    values = [summary["float_control_matches"], summary["protected"]["all"]["matched"],
              summary["protected"]["neighborhood"]["matched"],
              summary["protected"]["quantization"]["matched"],
              summary["protected"]["integrity_ge_0p95"]["matched"],
              summary["frozen_best_matches"], summary["posthoc_best_matches"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, values, color=["#4a5568", "#38a169", "#38a169", "#3182ce", "#68d391", "#c53030", "#dd6b20"])
    ax.bar_label(bars, padding=3); ax.axhline(54, color="black", linestyle="--", linewidth=1)
    ax.set(ylabel="Known matches / 79", title="Protected results and diagnostic bounds", ylim=(0, 65))
    ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(root / "protected_summary.png", dpi=150); plt.close(fig)

    comparison = audit / "3_Comparison"
    _montage([comparison / f"burst_{burst}_comparison.png" for burst in range(1, 5)], comparison / "spatial_overview.png")
    videos = sorted(audit.rglob("*.mp4")); failures = []
    for video in videos:
        passed, message = _decode(video)
        if not passed:
            failures.append({"path": str(video.relative_to(audit)), "error": message})
    validation = {
        "status": "passed" if not failures else "failed",
        "decoded_video_count": len(videos), "decode_failures": failures,
        "comparison_videos": 0,
        "annotation_separation": "expert-only and model-only overlays are separate",
        "comparison_contract": "figures and nearest-candidate traces only",
    }
    atomic_json(audit / "validation.json", validation)
    atomic_json(audit / "artifact_index.json", {"artifacts": [
        {"path": str(path.relative_to(audit)), "bytes": path.stat().st_size}
        for path in sorted(audit.rglob("*")) if path.is_file()]})
    status = json.loads((root / "status.json").read_text())
    status["scientific_audit"] = "complete" if not failures else "media_decode_failed"
    atomic_json(root / "status.json", status)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    result = validate(parser.parse_args().config); print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
