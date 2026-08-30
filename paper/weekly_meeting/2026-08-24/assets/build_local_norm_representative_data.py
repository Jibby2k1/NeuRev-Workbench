"""Build a slide-ready Local Norm illustration from frozen NeuRev data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from neurobench.portable_paths import data_root, portable_path


ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = data_root(ROOT)
ATLAS = ROOT / "Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8/03_trace_atlas"
MOVIE = DATA_ROOT / "Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy"
OUT = ROOT / "paper/weekly_meeting/2026-08-24/assets/local_norm_representative_data_roi003_b01.png"
PROVENANCE = OUT.with_suffix(".json")

ORANGE = "#D97706"
TEAL = "#176B87"
GREEN = "#527A48"
INK = "#17212B"
GRAY = "#69737D"
GRID = "#DDE3E7"

site_id = "roi_003"
burst_id = 1
x_px, y_px = 405.575, 216.977
start_ui, end_ui, peak_ui = 2003, 2026, 2007
center_radius, annulus_inner, annulus_outer = 2, 3, 6
fps = 50.0

traces = np.load(ATLAS / "traces.npz", allow_pickle=False)
site_index = int(np.flatnonzero(traces["site_ids"] == site_id)[0])
raw_trace = traces["raw"][site_index]
annulus_trace = traces["annulus"][site_index]
residual_trace = traces["residual"][site_index]

movie = np.load(MOVIE, mmap_mode="r", allow_pickle=False)
frame = np.asarray(movie[peak_ui - 1], dtype=np.float32)
cx, cy = int(round(x_px)), int(round(y_px))
half = 14
crop = frame[cy-half:cy+half+1, cx-half:cx+half+1]
lo, hi = np.percentile(crop, [2, 99.5])

window_start_ui = start_ui - 20
window_end_ui = end_ui + 20
ui = np.arange(window_start_ui, window_end_ui + 1)
idx = ui - 1
time_s = (ui - start_ui) / fps

fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
grid = fig.add_gridspec(3, 2, width_ratios=[0.92, 1.5], hspace=0.22, wspace=0.18,
                        left=0.055, right=0.975, top=0.86, bottom=0.10)
ax_img = fig.add_subplot(grid[:, 0])
axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]

ax_img.imshow(crop, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
center = (half + (x_px - cx), half + (y_px - cy))
ax_img.add_patch(Circle(center, center_radius, fill=False, lw=3.0, ec=ORANGE))
ax_img.add_patch(Circle(center, annulus_inner, fill=False, lw=2.1, ec=TEAL, ls="--"))
ax_img.add_patch(Circle(center, annulus_outer, fill=False, lw=3.0, ec=TEAL))
ax_img.annotate("ROI  ·  radius 2 px", xy=(center[0]+1.5, center[1]-1.5), xytext=(1, 3),
                color=ORANGE, fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=2))
ax_img.annotate("Annulus  ·  3–6 px", xy=(center[0]+4.6, center[1]+3.2), xytext=(1, 25),
                color=TEAL, fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=2))
ax_img.set_title("Native frame at observed peak", fontsize=16, color=INK, fontweight="bold", pad=12)
ax_img.text(0.02, -0.055, "roi_003  ·  burst 1  ·  UI frame 2007", transform=ax_img.transAxes,
            fontsize=12, color=GRAY)
ax_img.set_xticks([]); ax_img.set_yticks([])
for spine in ax_img.spines.values(): spine.set_color("#AEB7BE"); spine.set_linewidth(1.0)

series = [
    (raw_trace[idx], "ROI signal", ORANGE, "native intensity"),
    (annulus_trace[idx], "Annulus signal", TEAL, "native intensity"),
    (residual_trace[idx], "ROI − annulus", GREEN, "residual intensity"),
]
for ax, (values, label, color, unit) in zip(axes, series):
    ax.plot(time_s, values, color=color, lw=2.5)
    ax.axvspan(0, (end_ui-start_ui)/fps, color=color, alpha=0.08, lw=0)
    ax.axvline(0, color=INK, lw=1.1, ls="--")
    ax.axvline((end_ui-start_ui)/fps, color=INK, lw=1.1, ls=":")
    ax.text(0.012, 0.88, label, transform=ax.transAxes, color=color,
            fontsize=15, fontweight="bold", va="top")
    ax.set_ylabel(unit, color=GRAY, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7BE"); ax.spines["bottom"].set_color("#AEB7BE")
    ax.tick_params(colors=GRAY, labelsize=9)
axes[-1].set_xlabel("Time relative to expert-identified burst onset (s)", color=INK, fontsize=12)
axes[0].text(0.47, 1.07, "expert-identified burst interval", transform=axes[0].transAxes,
             color=INK, fontsize=10, ha="center")

fig.suptitle("Representative Local Norm measurement | validated data",
             x=0.055, y=0.955, ha="left", fontsize=22, color=INK, fontweight="bold")
fig.text(0.055, 0.905,
         "Actual native frame and frozen trace-atlas outputs; colored geometry matches the analysis contract.",
         ha="left", fontsize=13, color=GRAY)
fig.text(0.975, 0.025,
         "Representative observation only · center disk r=2 px · annulus r=3–6 px · 50 Hz",
         ha="right", fontsize=10, color=GRAY)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, facecolor="white")
plt.close(fig)

PROVENANCE.write_text(json.dumps({
    "schema_version": 1,
    "figure": portable_path(OUT, repository=ROOT, data=DATA_ROOT),
    "representative_observation": "b01__roi_003",
    "site_id": site_id,
    "burst_id": burst_id,
    "x_px": x_px,
    "y_px": y_px,
    "event_frames_ui_inclusive": [start_ui, end_ui],
    "peak_frame_ui": peak_ui,
    "fps": fps,
    "geometry_px": {"center_radius": center_radius, "annulus_inner": annulus_inner, "annulus_outer": annulus_outer},
    "movie_source": portable_path(MOVIE, repository=ROOT, data=DATA_ROOT),
    "trace_source": portable_path(ATLAS / "traces.npz", repository=ROOT, data=DATA_ROOT),
    "channels": ["raw", "annulus", "residual"],
    "interpretation": "Measured representative observation, not a population summary."
}, indent=2) + "\n", encoding="utf-8")

print(OUT)
print(PROVENANCE)
