"""Atomic scientific artifacts for the pairwise experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from neurobench.experiments.frame_difference import _atomic_json


def _jsonable(value: Any) -> Any:
    """Convert NumPy-backed diagnostics to strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary=path.with_suffix(path.suffix+".partial")
    with temporary.open("wb") as stream: np.save(stream,array,allow_pickle=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary=path.with_suffix(path.suffix+".partial")
    with temporary.open("wb") as stream: np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def _write_review_tiff(
    path: Path,
    values: np.ndarray,
    *,
    signed: bool,
    method_id: str,
) -> None:
    """Write a robustly normalized uint16 review stack without a full copy."""
    sample = np.asarray(values[::4, ::4, ::4], dtype=np.float32)
    if signed:
        scale = max(float(np.percentile(np.abs(sample), 99.5)), 1e-6)
        normalization = {
            "mode": "symmetric_percentile",
            "percentile": 99.5,
            "source_limits": [-scale, scale],
            "display_zero": 32768,
            "interpretation": "negative=below-mid-gray, zero=mid-gray, positive=above-mid-gray",
        }
    else:
        scale = max(float(np.percentile(sample, 99.5)), 1e-6)
        normalization = {
            "mode": "positive_percentile",
            "percentile": 99.5,
            "source_limits": [0.0, scale],
            "display_zero": 0,
            "interpretation": "larger positive standardized activity is brighter",
        }
    description = json.dumps({
        "schema_version": 1,
        "method_id": method_id,
        "axes": "TYX",
        "viewer_dtype": "uint16",
        "normalization": normalization,
    })
    temporary = path.with_suffix(path.suffix + ".partial")
    with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
        for index, frame in enumerate(values):
            source = np.asarray(frame, dtype=np.float32)
            if signed:
                page = np.rint((np.clip(source, -scale, scale) / scale + 1.0) * 32767.5).astype(np.uint16)
            else:
                page = np.rint(np.clip(source, 0, scale) / scale * 65535.0).astype(np.uint16)
            writer.write(
                page,
                photometric="minisblack",
                contiguous=True,
                metadata=None,
                description=description if index == 0 else None,
            )
    temporary.replace(path)


def write_lane(root: Path, method_id: str, lane: dict[str,Any], *, write_tiff: bool, objective_rows=None) -> None:
    target=root/"methods"/method_id; target.mkdir(parents=True,exist_ok=False)
    _atomic_json(target/"fit.json",_jsonable(lane["fit"]))
    _atomic_json(target/"diagnostics.json",_jsonable(lane["diagnostics"]))
    _atomic_json(target/"timing.json",_jsonable(lane["timing"]))
    parameter_keys = (
        "mean", "covariance", "whitening", "demixing", "mixing",
        "alpha_quiet", "activity_l1", "objective_value",
    )
    parameters = {
        key: np.asarray(lane["fit"][key])
        for key in parameter_keys
        if lane["fit"].get(key) is not None
    }
    atomic_npz(target/"parameters.npz", **parameters)
    if lane.get("continuous") is not None:
        atomic_npy(target/"continuous_activity.npy",lane["continuous"].astype(np.float32))
        atomic_npz(target/"candidate_maps.npz",positive_z=lane["positive_z"].max(axis=0).astype(np.float32))
        if write_tiff:
            _write_review_tiff(
                target/"continuous_activity_signed.tif",
                lane["continuous"],
                signed=True,
                method_id=method_id,
            )
            _write_review_tiff(
                target/"positive_z.tif",
                lane["positive_z"],
                signed=False,
                method_id=method_id,
            )
    if lane.get("binary_mask") is not None:
        mask=lane["binary_mask"].astype(np.uint8); atomic_npy(target/"binary_mask.npy",mask)
        if write_tiff:
            temporary=target/"binary_mask.tif.partial"; tifffile.imwrite(temporary,mask*255,photometric="minisblack",bigtiff=True,
                description=json.dumps({"canonical_values":[0,1],"viewer_values":[0,255],"axes":"TYX"}))
            temporary.replace(target/"binary_mask.tif")
    if objective_rows:
        with (target/"objective_by_angle.tsv").open("w",newline="",encoding="utf-8") as stream:
            fields=sorted({key for row in objective_rows for key in row}); writer=csv.DictWriter(stream,fields,delimiter="\t"); writer.writeheader(); writer.writerows(objective_rows)


def write_candidates(path: Path, rows: list[dict[str,Any]]) -> None:
    fields=["candidate_id","lane","frame_or_burst_id","score","x_px","y_px","matched_known_label","nearest_known_label_px","source_stratum","review_status","review_label","review_note","interpretation"]
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fields,delimiter="\t"); writer.writeheader()
        for index,row in enumerate(rows,1): writer.writerow({"candidate_id":f"candidate_{index:05d}",**row})


def write_figures(root: Path, lanes: dict[str, dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    """Write the four bounded diagnostic figures required by the workflow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=False)

    def save(figure, name: str) -> None:
        temporary = destination / f"{name}.partial"
        figure.tight_layout()
        figure.savefig(temporary, format="png", dpi=120)
        plt.close(figure)
        temporary.replace(destination / name)

    resolved = [(name, lane) for name, lane in lanes.items() if lane.get("positive_z") is not None]
    columns = max(1, len(resolved))
    figure, axes = plt.subplots(1, columns, figsize=(3.2 * columns, 3.2), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (name, lane) in zip(axes.ravel(), resolved):
        axis.imshow(np.max(lane["positive_z"], axis=0), cmap="magma")
        axis.set_title(name.replace("_", " "), fontsize=8)
        axis.axis("off")
    save(figure, "method_comparison_montage.png")

    figure, axis = plt.subplots(figsize=(6, 4))
    found_objective = False
    for name, lane in lanes.items():
        rows = lane.get("fit", {}).get("diagnostics", {}).get("objective_by_angle", [])
        if rows:
            xs = [row.get("angle_degrees", row.get("angle", index)) for index, row in enumerate(rows)]
            ys = [row.get("objective", row.get("value", np.nan)) for row in rows]
            axis.plot(xs, ys, marker=".", label=name)
            found_objective = True
    if found_objective:
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "No angle-grid objective emitted", ha="center", va="center")
    axis.set(xlabel="Angle (degrees)", ylabel="Objective", title="ICA objective by angle")
    save(figure, "objective_by_angle.png")

    comparable = [row for row in metrics if "mean_recall" in row]
    figure, axis = plt.subplots(figsize=(6, 4))
    for row in comparable:
        axis.scatter(row["total_event_candidates"], row["mean_recall"])
        axis.annotate(row["lane"], (row["total_event_candidates"], row["mean_recall"]), fontsize=7)
    axis.set(xlabel="Event candidates", ylabel="Mean known-label recall", title="Recall and candidate burden")
    save(figure, "recall_candidate_tradeoff.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    names = [name for name, _ in resolved]
    occupancy = [float(lane["binary_mask"].mean()) for _, lane in resolved]
    axis.bar(range(len(names)), occupancy)
    axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], fontsize=7)
    axis.set(ylabel="Whole-stack nonzero fraction", title="Binary-mask occupancy diagnostic")
    save(figure, "quiet_burst_occupancy.png")
