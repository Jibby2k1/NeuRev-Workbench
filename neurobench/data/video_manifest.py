"""Video manifest parsing for template-grid zebrafish workflows."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from neurobench.data.video import video_metadata

DEFAULT_VIDEO_PATTERN = r"^(?P<index>[1-9])_(?P<label>left|right|neutral)\.(?:tif|tiff|npy)$"
DEFAULT_LABELS = ("left", "right", "neutral")
SUPPORTED_SUFFIXES = {".tif", ".tiff", ".npy"}


def build_video_manifest(
    *,
    input_dir: str | Path | None = None,
    files: Iterable[str | Path] | None = None,
    dataset_id: str = "zebrafish_left_right_neutral_v1",
    filename_regex: str = DEFAULT_VIDEO_PATTERN,
    labels: Iterable[str] = DEFAULT_LABELS,
    label_aliases: Mapping[str, str] | None = None,
    frame_rate_hz: float | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Build a metadata manifest from filename-compatible videos."""
    root = Path(input_dir or ".").expanduser()
    if files is None:
        if not root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
        paths = sorted(path for path in root.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    else:
        paths = sorted(Path(path).expanduser() for path in files)
    pattern = re.compile(filename_regex)
    label_set = list(labels)
    aliases = {str(k): str(v) for k, v in dict(label_aliases or {}).items()}
    videos: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        name = path.name
        match = pattern.match(name)
        if not match:
            message = f"Ignoring video with unexpected filename: {name}"
            if strict:
                raise ValueError(message)
            warnings.append(message)
            continue
        groups = match.groupdict()
        raw_label = str(groups.get("label") or "")
        label = aliases.get(raw_label, raw_label)
        if label not in label_set:
            message = f"Ignoring video with unsupported label '{raw_label}': {name}"
            if strict:
                raise ValueError(message)
            warnings.append(message)
            continue
        meta: dict[str, Any]
        try:
            meta = video_metadata(path)
        except Exception as exc:
            meta = {"frames": None, "height": None, "width": None, "dtype": None}
            warnings.append(f"Could not inspect {name}: {exc}")
        video_id = path.stem
        videos.append(
            {
                "video_id": video_id,
                "path": str(path),
                "index": int(groups.get("index") or len(videos) + 1),
                "label": label,
                "raw_label": raw_label,
                "fish_id": video_id,
                "condition": raw_label,
                "frame_count": meta.get("frames"),
                "height": meta.get("height"),
                "width": meta.get("width"),
                "dtype": meta.get("dtype"),
                "frame_rate_hz": frame_rate_hz,
                "notes": "Each video is treated as a different fish.",
            }
        )
    counts = Counter(video["label"] for video in videos)
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "videos": videos,
        "label_set": label_set,
        "label_counts": {label: int(counts.get(label, 0)) for label in label_set},
        "split_policy": "by_video",
        "warnings": warnings,
        "extras": {"filename_regex": filename_regex, "label_aliases": aliases},
    }


def video_by_id(manifest: Mapping[str, Any], video_id: str) -> dict[str, Any]:
    for video in manifest.get("videos", []) or []:
        if str(video.get("video_id")) == str(video_id):
            return dict(video)
    raise KeyError(f"video_id not found in manifest: {video_id}")


def label_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = Counter(str(video.get("label") or "") for video in manifest.get("videos", []) or [])
    return {str(label): int(counts.get(str(label), 0)) for label in manifest.get("label_set", DEFAULT_LABELS)}
