"""Small artifact helpers for dependent-multiscale preflight."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_label_projection(
    quiet_frames: np.ndarray, labels_tsv: str | Path, destination: str | Path
) -> int:
    """Validate x/y coordinates and write the mandatory preflight overlay."""
    frames = np.asarray(quiet_frames, dtype=np.float32)
    if frames.ndim != 3 or not len(frames) or not np.isfinite(frames).all():
        raise ValueError("quiet_frames must be finite [T,Y,X]")
    with Path(labels_tsv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or any("x_px" not in row or "y_px" not in row for row in rows):
        raise ValueError("labels table lacks x_px/y_px rows")
    coordinates = [(float(row["x_px"]), float(row["y_px"])) for row in rows]
    if any(not (0 <= x < frames.shape[2] and 0 <= y < frames.shape[1]) for x, y in coordinates):
        raise ValueError("label coordinate lies outside observation")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = frames.mean(axis=0)
    low, high = np.percentile(image, [1, 99.5])
    figure, axis = plt.subplots(figsize=(10, 6), dpi=140)
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    axis.scatter(
        [item[0] for item in coordinates], [item[1] for item in coordinates],
        s=34, facecolors="none", edgecolors="cyan", linewidths=1,
    )
    axis.set(
        title="Dependent multiscale label projection (evaluation only)",
        xlabel="x = column", ylabel="y = row",
    )
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return len(rows)
