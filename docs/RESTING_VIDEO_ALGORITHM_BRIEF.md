# Resting Calcium Video Algorithm Brief

This workbench proposes candidate neurons and calcium events for
`calcium_rest_cropped.tif`. It is a local annotation/inspection tool, not yet a
validated detector with measured accuracy.

## Current Result

- Video: cropped resting hindbrain, `421 x 259` pixels, `628` frames
- Assumed sampling: `5 Hz`, `0.5 um/pixel`
- Candidate ROIs: `59`
- Missed-neuron suggestions: `62`
- Median ROI size: `47 px`, equivalent diameter `7.7 px` or about `3.85 um`

Expected hindbrain neurons are roughly `5-10 um` diameter (`10-20 px` at
`0.5 um/pixel`). The current median ROI is smaller, so these candidates are best
interpreted as active compact footprints rather than guaranteed full cell
bodies.

## Core Idea

The background is nonuniform and locally clustered, so one global threshold is
not appropriate. The current method uses local/adaptive measurements:

1. **Temporal high-pass:** estimate a slow baseline with Gaussian smoothing over
   `4`, `6`, and `8` frames, then compute `high_pass = raw - baseline`. At
   `5 Hz`, these windows are about `0.8`, `1.2`, and `1.6` seconds.
2. **Local background score:** remove frame-wide shifts and compute positive
   local z-scores using local median/MAD-style statistics, so each pixel is
   judged relative to nearby background.
3. **Candidate components:** propose compact connected components from
   permissive local-z seed/grow thresholds, then reject components that are too
   large, sparse, or elongated.
4. **ROI proposal:** aggregate robust-z evidence over time to form stable
   candidate footprints. For this run, data-derived ROI thresholds were peak
   `4.91`, grow floor `4.65`, and area `8-260 px`.
5. **Trace events:** extract each ROI trace, subtract a local background ring
   with neuropil weight `0.7`, estimate a slow robust Kalman-style baseline, and
   mark positive innovations above the baseline. The current displayed event
   threshold is `z >= 2.4`.

These values are baseline/review settings, not label-optimized parameters.

## Dashboard Interpretation

Selected-ROI trace colors:

- **Blue:** background-corrected dF/F-like ROI trace
- **Gray:** slow Kalman baseline
- **Orange:** positive-innovation event z-score
- **Yellow dots:** candidate event frames where event z is a local maximum above
  threshold

The yellow dots are not direct spikes. Calcium imaging measures indirect
calcium influx, and at `5 Hz` this video cannot reliably separate a single spike
from a burst. Treat them as candidate fluorescence transient onsets.

Discovery maps highlight uncovered regions with strong robust-z activity, local
contrast, or local temporal correlation. These are missed-neuron review targets,
not ground truth.

## What Has Not Been Tuned Yet

No label-driven hyperparameter optimization has been done. The current output is
a baseline proposal set built from heuristic defaults, data-derived thresholds,
and hand-tunable dashboard controls.

The next useful step is annotation-driven tuning: label likely neurons, false
positives, artifacts, and missed neurons, then tune against those labels.
