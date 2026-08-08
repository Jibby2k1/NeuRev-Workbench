"""Formula-explicit diagnostic package for the completed two-frame CS-Parzen ICA run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.algorithms.pairwise_separation import quiet_difference_stats
from neurobench.experiments.frame_difference import _atomic_json
from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_figures import (
    _AtomicMp4Writer,
)
from neurobench.experiments.pairwise_separation.sampling import causal_preprocess


SCHEMA_VERSION = 1
DEFAULT_REPRESENTATIVE_UI = (1900, 2005, 2048, 2130, 2270)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _sample(values: np.ndarray) -> np.ndarray:
    return np.asarray(values[:: max(1, len(values) // 70), ::4, ::4], dtype=np.float32)


def _bounds(values: np.ndarray, *, signed: bool, positive: bool = False) -> tuple[float, float]:
    sample = _sample(values)
    if positive:
        return 0.0, max(float(np.quantile(np.maximum(sample, 0), 0.995)), 1e-6)
    if signed:
        limit = max(float(np.quantile(np.abs(sample), 0.995)), 1e-6)
        return -limit, limit
    low, high = np.quantile(sample, (0.005, 0.995))
    return float(low), max(float(high), float(low) + 1e-6)


def _gray(values: np.ndarray, bounds: tuple[float, float], *, positive: bool = False) -> np.ndarray:
    frame = np.asarray(values, dtype=np.float32)
    if positive:
        frame = np.maximum(frame, 0)
    low, high = bounds
    return np.rint(255 * np.clip((frame - low) / (high - low), 0, 1)).astype(np.uint8)


def _fit_direction(fit: dict[str, Any]) -> dict[str, Any]:
    component = int(fit["activity_component"])
    sign = int(fit["activity_sign"])
    whitening = np.asarray(fit["whitening"], dtype=np.float64)
    demixing = np.asarray(fit["demixing"], dtype=np.float64)
    effective = sign * (demixing[component] @ whitening)
    normalized = effective / np.linalg.norm(effective)
    derivative = np.asarray([-1.0, 1.0]) / math.sqrt(2.0)
    common = np.asarray([1.0, 1.0]) / math.sqrt(2.0)
    return {
        "activity_component_zero_based": component,
        "activity_sign": sign,
        "effective_direction_observation_coordinates": effective.tolist(),
        "normalized_effective_direction": normalized.tolist(),
        "absolute_cosine_to_derivative": float(abs(normalized @ derivative)),
        "absolute_cosine_to_common_direction": float(abs(normalized @ common)),
    }


def _load_labels(path: Path) -> tuple[list[dict[str, Any]], dict[int, tuple[tuple[float, float], ...]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    normalized: list[dict[str, Any]] = []
    by_burst: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        item = {
            "burst_id": int(row["burst_id"]),
            "start_frame_ui": int(row["start_frame_ui"]),
            "end_frame_ui": int(row["end_frame_ui"]),
            "roi_identity": row["roi_identity"],
            "x_px": float(row["x_px"]),
            "y_px": float(row["y_px"]),
        }
        normalized.append(item)
        by_burst.setdefault(item["burst_id"], []).append((item["x_px"], item["y_px"]))
    return normalized, {key: tuple(dict.fromkeys(value)) for key, value in by_burst.items()}


def _active_burst(rows: list[dict[str, Any]], ui_frame: int) -> int | None:
    for row in rows:
        if row["start_frame_ui"] <= ui_frame <= row["end_frame_ui"]:
            return int(row["burst_id"])
    return None


def _draw_rings(
    canvas: Image.Image,
    points: Iterable[tuple[float, float]],
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    source_shape: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    sy, sx = source_shape
    for x, y in points:
        px = x0 + x * width / sx
        py = y0 + y * height / sy
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="black", width=4)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="white", width=2)


def _panel(
    canvas: Image.Image,
    values: np.ndarray,
    *,
    column: int,
    row: int,
    title: str,
    formula: str,
    detail: str,
    bounds: tuple[float, float],
    positive: bool = False,
    rings: Iterable[tuple[float, float]] = (),
) -> None:
    panel_width, image_width, image_height = 640, 620, 368
    header_height, row_height, title_height = 88, 466, 88
    x0 = column * panel_width + 10
    y0 = header_height + row * row_height
    draw = ImageDraw.Draw(canvas)
    draw.text((x0, y0 + 4), title, fill="white", font=_font(18, bold=True))
    draw.text((x0, y0 + 30), formula, fill=(220, 220, 220), font=_font(14))
    draw.text((x0, y0 + 52), detail, fill=(170, 170, 170), font=_font(13))
    image_y = y0 + title_height
    rendered = Image.fromarray(_gray(values, bounds, positive=positive), mode="L")
    rendered = rendered.resize((image_width, image_height), Image.Resampling.BILINEAR).convert("RGB")
    canvas.paste(rendered, (x0, image_y))
    if rings:
        _draw_rings(
            canvas,
            rings,
            x0=x0,
            y0=image_y,
            width=image_width,
            height=image_height,
            source_shape=values.shape,
        )


def _render_frame(
    *,
    raw: np.ndarray,
    filtered: np.ndarray,
    fixed: np.ndarray,
    parzen: np.ndarray,
    z_positive: np.ndarray,
    residual: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    ui_frame: int,
    review_start_ui: int,
    burst_id: int | None,
    rings: tuple[tuple[float, float], ...],
    beta: float,
    effective_direction: tuple[float, float],
    scale_floor: float,
) -> Image.Image:
    canvas = Image.new("RGB", (1920, 1080), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), "BASIC TWO-FRAME CS-PARZEN ICA — FORMULA-EXPLICIT REAL-DATA REVIEW", fill="white", font=_font(22, bold=True))
    burst_text = "quiet / no active label interval" if burst_id is None else f"annotated burst {burst_id}"
    draw.text(
        (14, 42),
        f"UI frame {ui_frame} | review offset {ui_frame-review_start_ui} | source cadence 20 ms | playback 5× slower | {burst_text}",
        fill=(205, 205, 205),
        font=_font(15),
    )
    _panel(
        canvas,
        raw,
        column=0,
        row=0,
        title="1. RAW OBSERVATION  R_t",
        formula="R_t(x,y) = original uint16 fluorescence intensity",
        detail=f"fixed display range [{bounds['raw'][0]:.3g}, {bounds['raw'][1]:.3g}]",
        bounds=bounds["raw"],
        rings=rings,
    )
    _panel(
        canvas,
        filtered,
        column=1,
        row=0,
        title="2. CAUSAL PREPROCESSED INPUT  P_t",
        formula="P_t = EMA_{a=0.4}(G_{sigma=1 px} * R_t)",
        detail=f"span=4 frames; fixed range [{bounds['filtered'][0]:.3g}, {bounds['filtered'][1]:.3g}]",
        bounds=bounds["filtered"],
    )
    _panel(
        canvas,
        fixed,
        column=2,
        row=0,
        title="3. FIXED TEMPORAL DERIVATIVE  D_t",
        formula="D_t = P_t - P_{t-1}",
        detail=f"signed: mid-gray=0; fixed range [{bounds['fixed'][0]:.3g}, {bounds['fixed'][1]:.3g}]",
        bounds=bounds["fixed"],
    )
    _panel(
        canvas,
        parzen,
        column=0,
        row=1,
        title="4. FITTED CS-PARZEN ACTIVITY  Y_t",
        formula="Y_t = s e_k^T W Q ([P_{t-1},P_t]^T - mu)",
        detail=f"s=-1, k=1 (zero-based); signed fixed range [{bounds['parzen'][0]:.3g}, {bounds['parzen'][1]:.3g}]",
        bounds=bounds["parzen"],
        rings=rings,
    )
    _panel(
        canvas,
        z_positive,
        column=1,
        row=1,
        title="5. POSITIVE QUIET-STANDARDIZED SCORE  Z_t+",
        formula="Z_t+ = max(0, (Y_t-med_Q) / max(1.4826 MAD_Q, floor))",
        detail=f"Q=UI 1800–1899; floor={scale_floor:.4g}; decision threshold Z>=3; not a probability",
        bounds=bounds["z_positive"],
        positive=True,
        rings=rings,
    )
    _panel(
        canvas,
        residual,
        column=2,
        row=1,
        title="6. PARZEN NON-DERIVATIVE RESIDUAL  E_t",
        formula="E_t = Y_t - beta D_t; beta = argmin_Q ||Y-beta D||_2^2",
        detail=f"beta={beta:.6g}; signed fixed range [{bounds['residual'][0]:.3g}, {bounds['residual'][1]:.3g}]",
        bounds=bounds["residual"],
    )
    draw.rectangle((0, 1020, 1920, 1080), fill=(15, 15, 15))
    draw.text(
        (14, 1029),
        "DISPLAY: raw/preprocessed black=low, white=high | signed panels black=negative, mid-gray=zero, white=positive | scales never change by frame",
        fill="white",
        font=_font(14),
    )
    draw.text(
        (14, 1053),
        f"LEARNED OBSERVATION AXIS: [{effective_direction[0]:+.6f}, {effective_direction[1]:+.6f}] | rings = known sparse-positive labels only during their burst; all other pixels are unknown",
        fill=(205, 205, 205),
        font=_font(13),
    )
    return canvas


def _direction_figure(path: Path, direction: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    vectors = {
        "common [1,1]": (np.asarray([1.0, 1.0]) / math.sqrt(2), "0.45"),
        "derivative [-1,1]": (np.asarray([-1.0, 1.0]) / math.sqrt(2), "0.15"),
        "learned CS-Parzen": (direction / np.linalg.norm(direction), "0.65"),
    }
    styles = {"common [1,1]": "--", "derivative [-1,1]": "-", "learned CS-Parzen": ":"}
    for label, (vector, color) in vectors.items():
        axis.arrow(0, 0, vector[0], vector[1], width=0.012, head_width=0.08, length_includes_head=True, color=color, linestyle=styles[label], label=label)
        axis.text(vector[0] * 1.08, vector[1] * 1.08, label, ha="center", fontsize=10)
    axis.axhline(0, color="0.8", linewidth=1)
    axis.axvline(0, color="0.8", linewidth=1)
    axis.set(xlim=(-1.25, 1.25), ylim=(-0.25, 1.25), xlabel="coefficient on P(t-1)", ylabel="coefficient on P(t)", title="Observation-space directions")
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _recall_figure(path: Path, metrics: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in metrics["lanes"] if "mean_recall" in row]
    labels = [row["lane"].replace("_", "\n") for row in rows]
    values = [row["mean_recall"] for row in rows]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(range(len(rows)), values, color="0.35")
    axis.set_xticks(range(len(rows)), labels, fontsize=8)
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("mean known-label recall across four bursts")
    axis.set_title("Real-data detection comparison (79 sparse-positive labels)")
    for index, value in enumerate(values):
        axis.text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_method_reference(path: Path, evidence: dict[str, Any]) -> None:
    d = evidence["direction"]
    p = evidence["preliminary_metrics"]
    text = f"""# Basic two-frame CS-Parzen ICA: method and interpretation reference

## Technical summary

This package studies a two-observation ICA model applied to adjacent, causally preprocessed fluorescence frames. The fit is numerically resolved and converged, but the recovered activity direction is almost exactly the temporal derivative. Its absolute cosine similarity to `[-1, 1] / sqrt(2)` is `{d['absolute_cosine_to_derivative']:.9f}`. The method is therefore best interpreted as a nonparametrically fitted temporal-change diagnostic, not as validated neural/background source separation.

## Data and analysis interval

- Source movie: Spon Ca Burst `3 hindbrain to tail 488 20ms.tif` and its memory-mapped uint16 cache.
- Full source shape: `2359 × 340 × 573` in `T,Y,X` order.
- Review interval: UI frames `1800–2359`, inclusive (`560` frames).
- Quiet calibration: UI frames `1800–1899`, inclusive (`100` frames).
- Source cadence: `20 ms/frame` (`50 Hz`). The video is encoded at `10 fps`, so playback is five times slower than acquisition.
- Labels: `79` burst-specific sparse-positive annotations. Unlabeled pixels are unknown, not negative.
- Coordinates: `x=column`, `y=row`; UI indices are one-based and inclusive.

## Step 1 — causal preprocessing

Let `R_t(x,y)` be the original uint16 fluorescence frame. The fitted method operated on

```text
P_t = EMA_a(G_sigma * R_t),    sigma = 1 pixel,    a = 2/(4+1) = 0.4.
```

`G_sigma` is a spatial Gaussian filter with reflect padding and truncation at `4 sigma`. The exponential moving average is causal:

```text
P_0 = G_sigma * R_0
P_t = 0.4 (G_sigma * R_t) + 0.6 P_(t-1).
```

No motion correction was applied.

## Step 2 — the two-observation ICA model

At every sampled pixel and time, the observation is

```text
x_t = [P_(t-1), P_t]^T.
```

The samples are centered by `mu` and whitened by `Q`:

```text
z_t = Q (x_t - mu).
```

The fitted rotation `W` produces two components:

```text
y_t = W z_t.
```

This model has only two observation dimensions. It cannot independently identify neural signal, background, motion, and measurement noise as four physical sources.

## Step 3 — the Parzen independence objective

For candidate outputs `(y_1,y_2)`, Gaussian Parzen kernels estimate their joint density and marginal densities. Independence is measured with the Cauchy–Schwarz divergence between the joint density `p(y_1,y_2)` and the product of marginals `p(y_1)p(y_2)`:

```text
D_CS(p,q) = -log( integral(p q) / sqrt(integral(p^2) integral(q^2)) ).
```

The implementation searches a bounded two-dimensional rotation angle and minimizes this divergence. It used bandwidth `0.35`, `1024` screen samples, `4096` confirmation samples, `3 degree` coarse steps, and `0.25 degree` refinement steps. Kernel computations were blocked in `256`-row chunks rather than constructing an all-pairs full-movie kernel matrix.

## Step 4 — selecting and orienting the activity component

The recovered component was selected using correlation with the fixed derivative, positive skewness, upper-tail occupancy, and low correlation with the common direction. The selected component is `k={d['activity_component_zero_based']}` with sign `s={d['activity_sign']}`:

```text
Y_t = s e_k^T W Q ([P_(t-1),P_t]^T - mu).
```

The effective observation-space direction is

```text
[{d['normalized_effective_direction'][0]:+.9f}, {d['normalized_effective_direction'][1]:+.9f}].
```

For comparison:

```text
common direction       = [ 0.707106781,  0.707106781]
temporal derivative    = [-0.707106781,  0.707106781].
```

The learned direction has absolute cosine similarity `{d['absolute_cosine_to_derivative']:.9f}` to the derivative and `{d['absolute_cosine_to_common_direction']:.9f}` to the common direction.

## Step 5 — quiet-standardized positive activity

For every pixel, the quiet interval supplies a robust center and scale:

```text
c(x,y) = median_Q Y_t(x,y)
s(x,y) = max(1.4826 median_Q |Y_t-c|, {p['quiet_scale_floor']:.9g})
Z_t+(x,y) = max(0, (Y_t(x,y)-c(x,y))/s(x,y)).
```

The binary decision used `Z_t+ >= 3`. This z-like score is not a probability.

## What the real-data result establishes

- CS-Parzen converged in `{p['fit_iterations']}` objective evaluations with covariance condition number `{p['condition_number']:.3f}`.
- Full-stack sampled correlation between CS-Parzen activity and the fixed derivative is `{p['parzen_derivative_correlation']:.9f}`.
- After quiet-fitted scale alignment, the residual `E_t = Y_t - beta D_t` has normalized RMS `{p['non_derivative_residual_nrms']:.9g}` with `beta={p['quiet_alignment_beta']:.9g}`.
- CS-Parzen known-label mean recall was `{p['cs_parzen_mean_recall']:.4f}` (`{p['cs_parzen_matches']}/79` matches, `{p['cs_parzen_candidates']}` event candidates), versus Raw Direct `{p['raw_direct_mean_recall']:.4f}` (`{p['raw_direct_matches']}/79`, `{p['raw_direct_candidates']}` candidates).

These measurements establish that the fitted output is essentially change evidence. They do not establish that detected changes are neural, that unmatched candidates are false, or that the residual is measurement noise.

## Why the derivative appears

Persistent intensity lies near `[1,1]`: it changes little between adjacent frames. The orthogonal direction is `[-1,1]`, which evaluates to `P_t-P_(t-1)`. With only two strongly correlated adjacent observations, whitening and independence optimization naturally expose a common-level coordinate and a change coordinate. The result is therefore a consequence of the observation geometry as well as the Parzen objective.

## Video panel dictionary

The full diagnostic video contains six formula-defined panels. All display limits are fixed for the full 560-frame interval.

1. `R_t`: original raw observation.
2. `P_t`: Gaussian-smoothed causal EMA input used by ICA.
3. `D_t=P_t-P_(t-1)`: explicit fixed derivative baseline.
4. `Y_t`: selected, oriented CS-Parzen component.
5. `Z_t+`: positive quiet-standardized activity; threshold `3` is shown but the panel remains continuous.
6. `E_t=Y_t-beta D_t`: what remains after fitting the derivative's scale on quiet frames.

Raw and preprocessed panels use black for low and white for high. Signed panels use black for negative, mid-gray for zero, and white for positive. Black-backed white rings appear only during the corresponding labeled burst and indicate sparse-positive known neurons; all other pixels remain unknown.

## Interpretation boundaries

The method is defensible as an interpretable temporal-change estimator and as evidence about identifiability. It is not defensible as a validated physical source decomposition or as a replacement for Raw Direct. Motion, illumination changes, neural onsets, and frame noise can all contribute to `D_t` and therefore to `Y_t`.

## Next questions justified by this package

1. Does the small non-derivative residual contain repeatable biological structure or only numerical/model mismatch?
2. Would longer temporal embeddings provide genuinely new identifiable directions, rather than reparameterizing derivatives?
3. Can measured motion or illumination covariates explain derivative energy before neural interpretation?
4. Can the continuous Parzen score help timing or ranking while Raw Direct retains amplitude and structure?
5. Which conclusions survive bounded, exhaustively annotated spatial review?
"""
    path.write_text(text, encoding="utf-8")


def _write_chatgpt_handoff(path: Path, evidence: dict[str, Any]) -> None:
    d = evidence["direction"]
    p = evidence["preliminary_metrics"]
    text = f"""# ChatGPT handoff: basic two-frame CS-Parzen ICA on Spon Ca Burst

## Requested role

Use this file and the accompanying figures as the empirical source packet for a rigorous methods document and discussion. Clearly distinguish:

1. facts measured from the local data;
2. mathematical consequences of the two-frame observation model;
3. plausible interpretations that remain hypotheses;
4. claims that the evidence does not support.

Do not describe the method as successful neural/background separation. Ask explicit questions when a biological prior, acceptable preservation tolerance, or intended scientific claim is missing.

## Local-data facts unavailable to a general ChatGPT session

- Movie: spontaneous calcium fluorescence, `2359 × 340 × 573` (`T,Y,X`), uint16, `20 ms/frame`.
- Analyzed interval: UI `1800–2359` inclusive, 560 frames.
- Quiet calibration: UI `1800–1899` inclusive, 100 frames.
- Four annotated burst intervals: `2003–2026`, `2040–2063`, `2122–2149`, `2254–2300`.
- Sparse-positive labels: 79 burst-specific rows. Unlabeled pixels are unknown; precision is not identifiable.
- Coordinates: x=column, y=row. UI frames are one-based/inclusive.
- No motion correction was used.

## Exact preprocessing and fitted model

```text
R_t = raw fluorescence
P_t = EMA_a(G_sigma * R_t), sigma=1 px, a=0.4 (span=4 frames)
x_t = [P_(t-1), P_t]^T
z_t = Q(x_t-mu)
y_t = W z_t
Y_t = s e_k^T y_t
```

Fit constants:

```text
mu = {evidence['fit']['mean']}
Q  = {evidence['fit']['whitening']}
W  = {evidence['fit']['demixing']}
k  = {d['activity_component_zero_based']} (zero-based)
s  = {d['activity_sign']}
normalized effective observation direction = [{d['normalized_effective_direction'][0]:+.9f}, {d['normalized_effective_direction'][1]:+.9f}]
```

Parzen objective:

```text
D_CS(p,q) = -log( integral(p q) / sqrt(integral(p^2) integral(q^2)) )
p = joint density of the two outputs
q = product of their marginal densities
```

Fit configuration: Gaussian kernel bandwidth 0.35; 1024 screen samples; 4096 confirmation samples; seed 20260727; 3-degree coarse angle grid; 0.25-degree refinement; 256-row kernel blocks.

## Preliminary measured results

```text
CS-Parzen converged: true
objective evaluations: {p['fit_iterations']}
confirmed objective: {p['objective_value']:.12g}
covariance condition number: {p['condition_number']:.9g}
cosine(learned direction, derivative): {d['absolute_cosine_to_derivative']:.12g}
cosine(learned direction, common): {d['absolute_cosine_to_common_direction']:.12g}
sampled correlation(Y_t, D_t): {p['parzen_derivative_correlation']:.12g}
quiet scale-alignment beta: {p['quiet_alignment_beta']:.12g}
normalized RMS of E_t=Y_t-beta D_t: {p['non_derivative_residual_nrms']:.12g}

Raw Direct: mean recall {p['raw_direct_mean_recall']:.9g}; {p['raw_direct_matches']}/79 known matches; {p['raw_direct_candidates']} event candidates
CS-Parzen: mean recall {p['cs_parzen_mean_recall']:.9g}; {p['cs_parzen_matches']}/79 known matches; {p['cs_parzen_candidates']} event candidates
```

The detection comparison uses the same four bursts and six-pixel primary match radius. Candidate yield is not precision because labels are not exhaustive.

## Primary interpretation to examine

The learned CS-Parzen direction is essentially `[-1,1]`, so the base method empirically rediscovers the temporal derivative. This is useful because it demonstrates that the optimizer is coherent and identifies a nonpersistent direction. It is limiting because neural onset, motion, illumination change, and measurement noise can all occupy that same direction.

The central research question is therefore not merely whether a richer independence loss can optimize better. It is what additional observations or priors are required to make biological source attribution identifiable.

## Accompanying evidence

- `METHOD_REFERENCE.md`: full method and interpretation reference.
- `preliminary_metrics.json`: machine-readable values quoted above.
- `fit.json`: exact fitted matrices and objective diagnostics.
- `real_data_metrics.json`: burst-level detection results.
- `figures/effective_direction_geometry.png`: learned direction versus common and derivative axes.
- `figures/objective_by_angle.png`: bounded CS-Parzen angle search.
- `figures/detection_comparison.png`: real-data known-label recall.
- `representative_frames/*.png`: exact frames rendered with the same definitions as the video.
- `VIDEO_GUIDE.md`: panel-by-panel definitions and display contract.

## Suggested document assignment

Produce a methods-first document with these sections:

1. motivating identifiability question;
2. data and preprocessing contract;
3. ICA model and whitening;
4. Parzen density intuition;
5. Cauchy–Schwarz independence objective;
6. bounded two-dimensional optimization;
7. activity-component selection and sign orientation;
8. quiet-standardized detection layer;
9. real-data results;
10. why the solution becomes a derivative;
11. supported versus unsupported claims;
12. concrete next experiments.

Use equations, but accompany every equation with a plain-language interpretation. Treat this package as preliminary evidence rather than a publication-ready validation study.
"""
    path.write_text(text, encoding="utf-8")


def _write_video_guide(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Basic CS-Parzen ICA diagnostic video guide",
        "",
        "## Playback and indexing",
        "",
        "- UI frames 1800–2359 inclusive; 560 video frames.",
        "- Source cadence 20 ms (50 Hz); MP4 playback 10 fps, five times slower than acquisition.",
        "- The first review frame has no previous frame inside the interval, so lagged panels are explicitly zero/undefined there.",
        "",
        "## Display contract",
        "",
        "- Every panel uses one fixed range for all 560 frames; there is no frame-wise contrast adjustment.",
        "- Raw/preprocessed panels: black=low, white=high.",
        "- Signed panels: black=negative, mid-gray=zero, white=positive.",
        "- Positive z-score panel: black=zero, white=the fixed upper display bound.",
        "- Black-backed white rings appear only during the labeled burst interval and mark sparse-positive known neurons. Unmarked pixels are unknown.",
        "",
        "## Panel formulas",
        "",
    ]
    for panel in manifest["panels"]:
        lines.extend([f"### {panel['title']}", "", f"`{panel['formula']}`", "", panel["interpretation"], ""])
    lines.extend(["## Fixed display bounds", "", "```json", json.dumps(manifest["display_bounds"], indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_package(
    *,
    completed_run: str | Path,
    raw_npy: str | Path,
    labels_tsv: str | Path,
    output_dir: str | Path,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Create a new atomic package from immutable completed-run artifacts."""
    source = Path(completed_run).resolve()
    raw_path = Path(raw_npy).resolve()
    labels_path = Path(labels_tsv).resolve()
    target = Path(output_dir).resolve()
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"Output or partial output already exists: {target}")
    required = (
        source / "config.resolved.json",
        source / "metrics.json",
        source / "methods/cs_parzen_ica/fit.json",
        source / "methods/cs_parzen_ica/continuous_activity.npy",
        source / "methods/fixed_binary_difference/continuous_activity.npy",
        raw_path,
        labels_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    partial.mkdir(parents=True)
    started = time.time()
    try:
        (partial / "videos").mkdir()
        (partial / "figures").mkdir()
        (partial / "representative_frames").mkdir()
        (partial / "source_evidence").mkdir()
        config = json.loads((source / "config.resolved.json").read_text())
        fit = json.loads((source / "methods/cs_parzen_ica/fit.json").read_text())
        metrics = json.loads((source / "metrics.json").read_text())
        frames = config["frames"]
        start_ui, end_ui = int(frames["review_start_ui"]), int(frames["review_end_ui"])
        quiet_count = int(frames["quiet_end_ui"] - frames["quiet_start_ui"] + 1)
        raw_source = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        raw = np.asarray(raw_source[start_ui - 1 : end_ui], dtype=np.float32)
        if raw.shape != (end_ui - start_ui + 1, 340, 573):
            raise ValueError(f"Unexpected aligned review shape: {raw.shape}")
        prep = config["preprocessing"]
        filtered = causal_preprocess(raw, float(prep["spatial_sigma_px"]), float(prep["temporal_ema_span_frames"]))
        parzen = np.load(source / "methods/cs_parzen_ica/continuous_activity.npy", mmap_mode="r", allow_pickle=False)
        fixed = np.load(source / "methods/fixed_binary_difference/continuous_activity.npy", mmap_mode="r", allow_pickle=False)
        if parzen.shape != raw.shape or fixed.shape != raw.shape:
            raise ValueError("Completed-run arrays are not aligned to the review interval")
        quiet_stats = quiet_difference_stats(parzen[1:quiet_count], floor_percentile=float(config["thresholding"]["quiet_mad_floor_percentile"]))
        quiet_y = np.asarray(parzen[1:quiet_count], dtype=np.float64)
        quiet_d = np.asarray(fixed[1:quiet_count], dtype=np.float64)
        beta = float(np.sum(quiet_y * quiet_d) / max(float(np.sum(quiet_d * quiet_d)), np.finfo(float).eps))
        sampled_y = _sample(parzen).ravel().astype(np.float64)
        sampled_d = _sample(fixed).ravel().astype(np.float64)
        sampled_e = sampled_y - beta * sampled_d
        correlation = float(np.corrcoef(sampled_y, sampled_d)[0, 1])
        residual_nrms = float(np.sqrt(np.mean(sampled_e**2)) / max(float(np.sqrt(np.mean(sampled_y**2))), np.finfo(float).eps))
        direction = _fit_direction(fit)
        normalized_direction = np.asarray(direction["normalized_effective_direction"], dtype=np.float64)
        rows, labels_by_burst = _load_labels(labels_path)
        raw_bounds = _bounds(raw, signed=False)
        filtered_bounds = _bounds(filtered, signed=False)
        fixed_bounds = _bounds(fixed, signed=True)
        parzen_bounds = _bounds(parzen, signed=True)
        z_sample = np.maximum((_sample(parzen) - quiet_stats.center[::4, ::4]) / quiet_stats.scale[::4, ::4], 0)
        z_bounds = (0.0, max(float(np.quantile(z_sample, 0.995)), 1e-6))
        residual_sample = sampled_e.reshape(_sample(parzen).shape)
        residual_limit = max(float(np.quantile(np.abs(residual_sample), 0.995)), 1e-6)
        bounds = {
            "raw": raw_bounds,
            "filtered": filtered_bounds,
            "fixed": fixed_bounds,
            "parzen": parzen_bounds,
            "z_positive": z_bounds,
            "residual": (-residual_limit, residual_limit),
        }
        video_path = partial / "videos/basic_two_frame_cs_parzen_ica_diagnostic.mp4"
        writer = _AtomicMp4Writer(video_path, (1080, 1920), fps)
        representative = set(DEFAULT_REPRESENTATIVE_UI)
        try:
            for index in range(len(raw)):
                ui_frame = start_ui + index
                burst_id = _active_burst(rows, ui_frame)
                rings = labels_by_burst.get(burst_id, ()) if burst_id is not None else ()
                if index == 0:
                    z_frame = np.zeros(raw.shape[1:], dtype=np.float32)
                    residual_frame = np.zeros(raw.shape[1:], dtype=np.float32)
                else:
                    z_frame = np.maximum((np.asarray(parzen[index]) - quiet_stats.center) / quiet_stats.scale, 0)
                    residual_frame = np.asarray(parzen[index]) - beta * np.asarray(fixed[index])
                canvas = _render_frame(
                    raw=raw[index], filtered=filtered[index], fixed=fixed[index], parzen=parzen[index],
                    z_positive=z_frame, residual=residual_frame, bounds=bounds, ui_frame=ui_frame,
                    review_start_ui=start_ui, burst_id=burst_id, rings=rings, beta=beta,
                    effective_direction=(float(normalized_direction[0]), float(normalized_direction[1])),
                    scale_floor=float(quiet_stats.scale_floor),
                )
                writer.write(np.asarray(canvas, dtype=np.uint8))
                if ui_frame in representative:
                    canvas.save(partial / "representative_frames" / f"ui_{ui_frame}.png")
            writer.close()
        except Exception:
            writer.abort()
            raise
        raw_lane = next(row for row in metrics["lanes"] if row["lane"] == "raw_direct")
        cs_lane = next(row for row in metrics["lanes"] if row["lane"] == "cs_parzen_ica")
        preliminary = {
            "condition_number": float(fit["condition_number"]),
            "fit_converged": bool(fit["converged"]),
            "fit_iterations": int(fit["iterations"]),
            "objective_value": float(fit["objective_value"]),
            "parzen_derivative_correlation": correlation,
            "quiet_alignment_beta": beta,
            "non_derivative_residual_nrms": residual_nrms,
            "quiet_scale_floor": float(quiet_stats.scale_floor),
            "raw_direct_mean_recall": float(raw_lane["mean_recall"]),
            "raw_direct_matches": int(raw_lane["total_matched"]),
            "raw_direct_candidates": int(raw_lane["total_event_candidates"]),
            "cs_parzen_mean_recall": float(cs_lane["mean_recall"]),
            "cs_parzen_matches": int(cs_lane["total_matched"]),
            "cs_parzen_candidates": int(cs_lane["total_event_candidates"]),
            "precision_identified": False,
        }
        evidence = {"fit": fit, "direction": direction, "preliminary_metrics": preliminary}
        _atomic_json(partial / "preliminary_metrics.json", evidence)
        shutil.copy2(source / "methods/cs_parzen_ica/fit.json", partial / "source_evidence/fit.json")
        shutil.copy2(source / "metrics.json", partial / "source_evidence/real_data_metrics.json")
        shutil.copy2(source / "methods/cs_parzen_ica/objective_by_angle.tsv", partial / "source_evidence/objective_by_angle.tsv")
        shutil.copy2(source / "figures/objective_by_angle.png", partial / "figures/objective_by_angle.png")
        _direction_figure(partial / "figures/effective_direction_geometry.png", normalized_direction)
        _recall_figure(partial / "figures/detection_comparison.png", metrics)
        panel_manifest = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": "videos/basic_two_frame_cs_parzen_ica_diagnostic.mp4",
                "frames": len(raw), "fps": fps, "duration_seconds": len(raw) / fps,
                "source_frame_period_ms": 20.0, "playback_slowdown": 50.0 / fps,
                "resolution_xy": [1920, 1080], "sha256": _sha256(video_path), "bytes": video_path.stat().st_size,
            },
            "review_interval_ui_inclusive": [start_ui, end_ui],
            "display_bounds": {key: [float(value[0]), float(value[1])] for key, value in bounds.items()},
            "panels": [
                {"title": "Raw observation R_t", "formula": "R_t(x,y) = original uint16 fluorescence intensity", "interpretation": "Anatomical/intensity context; not the direct ICA output."},
                {"title": "Causal preprocessed input P_t", "formula": "P_t = EMA_{a=0.4}(G_{sigma=1 px} * R_t)", "interpretation": "The actual current-frame observation supplied to the adjacent-frame ICA model."},
                {"title": "Fixed temporal derivative D_t", "formula": "D_t = P_t - P_{t-1}", "interpretation": "Declared non-learned background-null comparison."},
                {"title": "Fitted CS-Parzen activity Y_t", "formula": "Y_t = s e_k^T W Q ([P_{t-1},P_t]^T - mu)", "interpretation": "Selected and sign-oriented learned component; not automatically a physical neural source."},
                {"title": "Positive quiet-standardized score Z_t+", "formula": "Z_t+ = max(0,(Y_t-med_Q)/max(1.4826 MAD_Q,floor))", "interpretation": "Continuous one-sided activity score. The decision threshold is 3; the value is not a probability."},
                {"title": "Parzen non-derivative residual E_t", "formula": "E_t = Y_t-beta D_t; beta=argmin_Q ||Y-beta D||_2^2", "interpretation": "Displays only what remains after scaling the fixed derivative to the learned output using quiet frames."},
            ],
            "label_overlay": "black-backed white rings only within the matching annotated burst; sparse positives only; other pixels unknown",
            "scientific_status": "diagnostic_method_study_not_validated_source_separation",
        }
        _atomic_json(partial / "video_manifest.json", panel_manifest)
        _write_video_guide(partial / "VIDEO_GUIDE.md", panel_manifest)
        _write_method_reference(partial / "METHOD_REFERENCE.md", evidence)
        _write_chatgpt_handoff(partial / "CHATGPT_HANDOFF.md", evidence)
        readme = """# Basic two-frame CS-Parzen ICA diagnostic package

Start with `METHOD_REFERENCE.md` for the clean technical explanation and `VIDEO_GUIDE.md` before viewing the MP4. `CHATGPT_HANDOFF.md` is the self-contained context file to upload or paste into ChatGPT; `chatgpt_upload_bundle.zip` contains that handoff, method reference, machine-readable evidence, figures, and representative frames. The large MP4 remains beside the bundle.

Scientific status: the fit converged and essentially recovered the temporal derivative. This package is for method understanding and interpretation, not a validated neural/background separation claim.
"""
        (partial / "README.md").write_text(readme, encoding="utf-8")
        bundle = partial / "chatgpt_upload_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in (
                "CHATGPT_HANDOFF.md", "METHOD_REFERENCE.md", "VIDEO_GUIDE.md", "preliminary_metrics.json",
                "video_manifest.json", "source_evidence/fit.json", "source_evidence/real_data_metrics.json",
                "source_evidence/objective_by_angle.tsv", "figures/effective_direction_geometry.png",
                "figures/objective_by_angle.png", "figures/detection_comparison.png",
            ):
                archive.write(partial / relative, relative)
            for frame in sorted((partial / "representative_frames").glob("*.png")):
                archive.write(frame, frame.relative_to(partial))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "scientific_status": "diagnostic_method_study_not_validated_source_separation",
            "source_completed_run": str(source),
            "source_files": [
                {"role": "raw_cache", "path": str(raw_path), "sha256": _sha256(raw_path)},
                {"role": "labels", "path": str(labels_path), "sha256": _sha256(labels_path)},
                {"role": "completed_fit", "path": str(source / "methods/cs_parzen_ica/fit.json"), "sha256": _sha256(source / "methods/cs_parzen_ica/fit.json")},
                {"role": "completed_metrics", "path": str(source / "metrics.json"), "sha256": _sha256(source / "metrics.json")},
            ],
            "artifacts": {
                "video": panel_manifest["video"],
                "chatgpt_bundle": {"path": bundle.name, "bytes": bundle.stat().st_size, "sha256": _sha256(bundle)},
                "representative_frames": [path.name for path in sorted((partial / "representative_frames").glob("*.png"))],
            },
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(partial / "package_manifest.json", manifest)
        partial.replace(target)
        return manifest
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-run", required=True)
    parser.add_argument("--raw-npy", required=True)
    parser.add_argument("--labels-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    result = generate_package(
        completed_run=args.completed_run,
        raw_npy=args.raw_npy,
        labels_tsv=args.labels_tsv,
        output_dir=args.output_dir,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
