"""Video-split grid dynamics dataset building."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def build_dynamics_dataset(
    *,
    manifest: Mapping[str, Any],
    grid_states_dir: str | Path,
    out_dir: str | Path,
    window_frames: int = 8,
    prediction_horizon_frames: int = 1,
    temporal_stride_frames: int = 1,
    split_method: str = "stratified_by_label",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid_root = Path(grid_states_dir)
    temporal_stride_frames = max(1, int(temporal_stride_frames))
    frames_all: list[np.ndarray] = []
    frame_video_ids: list[str] = []
    frame_labels: list[str] = []
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    window_video_ids: list[str] = []
    window_labels: list[str] = []
    source_videos: list[str] = []
    video_labels: dict[str, str] = {}
    grid_id = ""
    normalization = ""
    for video in manifest.get("videos", []) or []:
        video_id = str(video["video_id"])
        label = str(video.get("label") or "")
        npz_path = grid_root / video_id / "grid_states.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Grid states not found for {video_id}: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            grid = data["grid_state"].astype(np.float32)
            normalization = str(data["normalization"].item() if data["normalization"].shape == () else data["normalization"])
            if "grid_id" in data.files:
                grid_id = str(data["grid_id"].item() if data["grid_id"].shape == () else data["grid_id"])
        if temporal_stride_frames > 1:
            grid = grid[::temporal_stride_frames]
        if grid.ndim != 4:
            raise ValueError(f"grid_state for {video_id} must be [T,H,W,C], got {grid.shape}")
        if not grid_id:
            grid_id = f"grid_{int(grid.shape[1])}x{int(grid.shape[2])}"
        chw = np.moveaxis(grid, -1, 1).astype(np.float32)
        source_videos.append(video_id)
        video_labels[video_id] = label
        frames_all.append(chw)
        frame_video_ids.extend([video_id] * int(chw.shape[0]))
        frame_labels.extend([label] * int(chw.shape[0]))
        for end in range(int(window_frames) - 1, int(chw.shape[0]) - int(prediction_horizon_frames)):
            start = end - int(window_frames) + 1
            target_index = end + int(prediction_horizon_frames)
            windows.append(chw[start : end + 1])
            targets.append(chw[target_index])
            window_video_ids.append(video_id)
            window_labels.append(label)
    if not frames_all:
        raise ValueError("No grid states found for dynamics dataset")
    frames = np.concatenate(frames_all, axis=0).astype(np.float32)
    window_arr = np.stack(windows).astype(np.float32) if windows else np.zeros((0, int(window_frames), *frames.shape[1:]), dtype=np.float32)
    target_arr = np.stack(targets).astype(np.float32) if targets else np.zeros((0, *frames.shape[1:]), dtype=np.float32)
    split_method_name = str(split_method or "stratified_by_label")
    splits = split_video_ids(
        video_labels,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_method=split_method_name,
    )
    warnings: list[str] = []
    if split_method_name in {"train_all_smoke", "train_all", "smoke"}:
        warnings.append("train-all smoke split: all videos are used for training; do not interpret metrics as held-out generalization")
    array_path = out / "dynamics_arrays.npz"
    np.savez(
        array_path,
        frames=frames,
        frame_video_ids=np.asarray(frame_video_ids, dtype="U64"),
        frame_labels=np.asarray(frame_labels, dtype="U16"),
        windows=window_arr,
        targets=target_arr,
        window_video_ids=np.asarray(window_video_ids, dtype="U64"),
        window_labels=np.asarray(window_labels, dtype="U16"),
        train_video_ids=np.asarray(splits["train_video_ids"], dtype="U64"),
        val_video_ids=np.asarray(splits["val_video_ids"], dtype="U64"),
        test_video_ids=np.asarray(splits["test_video_ids"], dtype="U64"),
    )
    label_counts = Counter(video_labels.values())
    dataset = {
        "schema_version": 1,
        "dataset_id": str(manifest.get("dataset_id") or "zebrafish_grid32_v1"),
        "grid_id": grid_id or f"grid_{int(frames.shape[2])}x{int(frames.shape[3])}",
        "source_videos": source_videos,
        "array_path": str(array_path),
        "input_shape": [int(v) for v in frames.shape[1:]],
        "windowing": {"window_frames": int(window_frames), "prediction_horizon_frames": int(prediction_horizon_frames), "stride_frames": 1, "temporal_stride_frames": int(temporal_stride_frames)},
        "splits": {"split_unit": "video", "split_method": split_method_name, **splits},
        "normalization": normalization,
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "warnings": warnings,
        "extras": {"window_count": int(window_arr.shape[0]), "frame_count": int(frames.shape[0])},
    }
    (out / "dynamics_dataset.json").write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps(dataset["splits"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def split_video_ids(
    video_labels: Mapping[str, str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    split_method: str = "stratified_by_label",
) -> dict[str, list[str]]:
    method = str(split_method or "stratified_by_label")
    if method in {"train_all_smoke", "train_all", "smoke"}:
        return {"train_video_ids": sorted(str(vid) for vid in video_labels), "val_video_ids": [], "test_video_ids": []}
    groups: dict[str, list[str]] = defaultdict(list)
    for vid, label in sorted(video_labels.items()):
        groups[str(label)].append(str(vid))
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for _label, vids in sorted(groups.items()):
        n = len(vids)
        if n >= 3:
            n_train = max(1, int(round(n * train_fraction)))
            n_val = max(1, int(round(n * val_fraction)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train.extend(vids[:n_train])
            val.extend(vids[n_train : n_train + n_val])
            test.extend(vids[n_train + n_val :])
        elif n == 2:
            train.append(vids[0])
            test.append(vids[1])
        elif n == 1:
            train.append(vids[0])
    return {"train_video_ids": train, "val_video_ids": val, "test_video_ids": test}


def load_dynamics_arrays(dataset: Mapping[str, Any] | str | Path):
    if not isinstance(dataset, Mapping):
        dataset = json.loads(Path(dataset).read_text(encoding="utf-8"))
    return np.load(dataset["array_path"], allow_pickle=False)
