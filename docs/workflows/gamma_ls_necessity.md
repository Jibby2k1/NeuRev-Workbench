# Gamma-LS component necessity pilot

This study follows the completed spatial/temporal sensitivity pilot. It asks
which score components help discover simulated activity promptly and which help
monitor an already configured source. It does not assume Gamma-LS is necessary.

## Frozen comparison

Eight arms cross two inputs with four readouts. The inputs are the current
causal exponential average (activity level) and its signed adjacent-frame
difference (activity change). Both start with Gaussian spatial smoothing,
sigma 1 pixel, followed by EMA alpha 0.4. Initialization and all frames are
identical to the preceding pilot; reconstructed differences must match its
saved input byte for byte.

The readouts are X (conditioned input), A (centered target response), C=A-M
(local contrast), and Z=C/max(Spread,floor) (full Gamma-LS). M and Spread are
the direct reference-weighted mean and standard deviation of X. Target and
reference act on the same input. There is no explicit guard mask. The spatial
Gamma setting is fixed at reference radius 7.5 pixels, centered target nominal
width 1 pixel, and reference order 9. No temporal Gamma search or learning is
included. The completed difference A/M/Spread/Z arrays are reused by hash.

The same 16 synthetic clips and one recording segment are reused, with the
same halo, eligible spatial domain, warmup, setup and application frames.
Thus 8 arms x 17 datasets give 136 cells. Acquisition is 50 Hz and 0.5 um/pixel.
All results remain development evidence on these sources.

## Calibration and discovery

Scores have different units. Each arm receives its own threshold, using only
setup frames, at common target setup proposal burdens q=0,0.25,0.5,1,2,4,8,16
per 194820 pixels per frame. The integer budget is
floor(q * eligible_area/194820 * setup_frame_count). The strict order-statistic
cutoff never exceeds this setup count, including ties. Small areas quantize
several budgets to the same cutoff; those identities remain visible.

The numerical sweep also includes all positive proposals and an explicit
no-output endpoint. q=0 is a finite setup maximum, not the no-output endpoint.
The media audit uses q=1. Equal setup targets do not guarantee equal application
burden or false-alarm probability. Thresholds and complete positive NMS
prefixes for all 136 cells are sealed before joining activity truth.

Discovery uses the maintained deterministic spatial NMS and one-to-one
inclusive 6-pixel frame matching. Synthetic precision and active-frame
sensitivity are distinct from event-window recovery. Report recovery by
0,20,40,60,100,200,500 ms after the simulated fluorescence onset, retaining all
events, including misses, in the denominator. Report false frame proposals per
area and time, duplicate proposals, and total downstream burden. Neural spike
timing and controller performance are not established by these endpoints.

## Configured monitoring

A separate synthetic check samples the same scores at known source-center
pixels without NMS. Source coordinates are an oracle configuration supplied
from the fixture geometry before calibration; this does not demonstrate
automatic location discovery. Each configured center has a strict positive
setup cutoff with an exceedance budget floor(0.01 * setup_frame_count).
Thresholds use no activity times or application intensities. The unit is a
configured ROI/frame: report precision, sensitivity, and inactive ROI/frame
exceedance rate separately from discovery metrics. Nuisance-only clips have
no configured simulated source and contribute no monitoring opportunities;
they still contribute fully to the discovery false-proposal analysis.

The real panel annotation acceptance is false. Existing positive windows
support only known-window coverage and unknown proposal burden. Do not derive
real precision, exhaustive sensitivity, false-positive rates or exact onset
delays from those windows. Prepared review panels remain available in the
previous pilot; no labels are fabricated by this study.

## Execution and evidence

Use `.venv-neurobench/bin/python` from this checkout and a new output root.
The runner is `neurobench.experiments.gamma_ls_difference.necessity_study`;
commands are `preflight`, `prepare`, `run`, `evaluate`, `display`, `media`, and
`validate`. Media accepts `--workers 3 --worker 0` (also workers 1 and 2).
Set numerical and codec thread counts to one, use low priority, exclude CPUs
6 and 7, and verify live resources. CPU convolution uses bounded chunks.

The default root is `Outputs/GammaLSNecessity/necessity_20260914_r1`.
Completed preceding roots and sources are immutable. No GPU jobs, stimulation,
deployment, or publication are part of this study.

The full Scientific Audit Output Standard is enabled at q=1 for every cell:
separate expert/model full-field videos, every applicable ROI's close-up and
full-duration exact-pixel trace, every expert occurrence's comparison, tables,
context index, report, source hashes and full lossless decode/marker checks.
Nuisance Expert sections explicitly say not applicable. Media shows Raw,
chosen Input, optional A, and the exact selected Score. Other saved stages are
linked diagnostics, not hidden dependencies of the simpler scores. Input and
A share their scale within each input mode; C uses fluorescence units and Z is
dimensionless. Different unit families have explicitly different fixed display
scales. Numerical completion is not media-audit completion.
