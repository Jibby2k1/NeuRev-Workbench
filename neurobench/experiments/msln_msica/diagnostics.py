"""Representative, explicitly typed diagnostic plots."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def render_diagnostics(root: Path, raw: np.ndarray, context_ids: tuple[str, ...], context_maps: list[np.ndarray], innovations: list[np.ndarray], evidence: np.ndarray, dominant: np.ndarray, angles: list[float]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = root / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    frame = len(raw) // 2
    panels = [("raw amplitude", raw[frame]), ("signed standardized evidence", context_maps[0][frame]), ("ICA innovation (signed)", innovations[0][frame]), ("final activity evidence (energy/surprise)", evidence[frame])]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3))
    for axis, (title, values) in zip(axes, panels):
        axis.imshow(values, cmap="gray")
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.tight_layout(); fig.savefig(destination / "context_maps_montage.tif", dpi=110); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 3)); axis.bar(np.arange(len(angles)), angles); axis.axhline(0, color="k", linewidth=.7); axis.set(xticks=np.arange(len(angles)), xticklabels=context_ids, ylabel="distance from derivative (degrees)", title="Per-context ICA innovation alignment"); axis.tick_params(axis="x", rotation=55, labelsize=7); fig.tight_layout(); fig.savefig(destination / "angle_by_context.png", dpi=120); plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4)); image = axis.imshow(dominant[frame], cmap="tab10", vmin=0, vmax=max(len(context_ids)-1, 1)); fig.colorbar(image, ax=axis, label="context index"); axis.set_title("Dominant context (categorical routing diagnostic)"); fig.tight_layout(); fig.savefig(destination / "dominant_context_maps.tif", dpi=110); plt.close(fig)
    return ["diagnostics/context_maps_montage.tif", "diagnostics/angle_by_context.png", "diagnostics/dominant_context_maps.tif"]
