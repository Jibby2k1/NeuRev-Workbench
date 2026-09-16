# Gamma-LS reference mean-distance and order study

This focused synthetic study investigates the reference-statistics failure seen
in the completed crowded-source controls. It changes only the spatial reference
weights. The exact centered target response, causal level conditioning, crop,
setup/application split and 3-by-3 peak selector stay fixed.

For each pixel at radial distance r, the reference weight is proportional to
`r^(n-1) exp(-mu r)`. The old radius parameter `(n-1)/mu` identifies the maximum
of this per-pixel profile. It is not the average distance of reference weight:
more pixels lie in larger annuli. In a continuous 2D field the radial-mass mean
is `(n+1)/mu`. Changing order at fixed old radius therefore changes both the
reference's distance and its concentration.

The new nine-kernel grid crosses orders 3, 9 and 15 with three **actual discrete
mean distances**, 2/3, 1 and 4/3 times the existing baseline's mean. These are
6.23066866, 9.34600299 and 12.46133732 pixels (3.1153, 4.6730 and 6.2307 micrometers).
The central order-9 stencil is byte-identical to the old R=7.5-pixel stencil.
Each other stencil solves mu on a fixed square; retention is checked against
converged larger squares. All retain at least 99.5% of estimated lattice mass,
and the largest reference plus Gaussian conditioning needs 39 pixels of halo,
within the existing 49. Metadata records realized means, standard deviations,
quantiles, old-style profile radii, support, effective weight count and source
footprint overlap. There is no explicit target or reference mask.

The 21 clips are the 12 existing paired crowded-source scenes (three seeds,
weak alone or with a neighbor at 8/12/16 pixels) plus nine source-free controls
(stationary noise, variance/correlation change, and shared-brightness/moving-
artifact stress, each with the same three seeds). All share byte-identical
warmup/setup for a given seed; nuisances begin only during application. The
compound stressors assess robustness, not attribution to individual ingredients.

Each clip has 64 warmup, 100 setup and 300 application frames at 50 Hz. Gaussian
sigma 1 pixel and causal EMA alpha .4 produce the level input. Direct reference
moments act on that same input and its square. The stored baseline target
response A is reused exactly across references, avoiding even tiny target-tail
changes caused by different reference supports. For each reference, compute
M, spread, C=A-M and Z=C/max(spread, floor), with the floor fitted from the
positive setup spread's 10th percentile. Every reference gets its own setup-only
thresholds. The peak selector keeps a 6-pixel border, score/y/x ordering and
greedy separation strictly greater than 6 pixels.

The primary operating point q=1 is a setup proposal budget, scaled by eligible
area relative to 194,820 pixels. It does not promise that application burden
will equal q. Retain q=0,.25,.5,1,2,4,8,16 and all-positive/no-output endpoints.
The q=0 setup-maximum cutoff is different from emitting no output. All 189
cells' scores, cutoffs and proposal streams must be sealed before activity
truth is joined. Synthetic truth is exhaustive; false proposals and precision
are defined only for this simulation. Source-free clips have no sensitivity
denominator; show false proposals per area-time rather than inventing recall.

Primary readouts are weak-source recovery within 100 ms and 2 pixels, including
misses in the event denominator; matched active-frame fraction; strong-neighbor
recovery; and source-free false-proposal burden. Keep 6-pixel matching separate.
Show every deadline/threshold curve, duplicates, localization and floor use.
The three seeds, not their many correlated frames, are the replicates. The
finite 1% decay tail contributes to active-frame sensitivity; prompt recovery
answers a different question. Simulated source widths and 100/1000-ms rise/decay
are development stress settings, not measured calcium kinetics.

Use `reference_study {preflight,prepare,run,evaluate}` under
`neurobench.experiments.gamma_ls_difference`. The noncolliding output root is
`Outputs/GammaLSReference/reference_20260915_r1`. Run numerical work with the
project virtual environment, one thread, low priority and affinity excluding
CPUs 6/7. Check active processes, GPU, available RAM and disk before a long run.
No GPU or controller is involved. Geometry-only setup projections are reviewed
before scoring. Completed roots and frozen source modules remain unchanged.

The full Scientific Audit Output Standard is enabled at q=1: every applicable
expert/model full field, ROI video and exact-pixel trace, and each expert
occurrence comparison, with fixed grayscale scales, separate annotation
colors, full decoding and marker validation. Twelve unchanged central audits
are reused only after score, cutoff, candidate and metric replication; 177
require new media. Empty synthetic truth is explicit. Numerical completion,
full audit completion and report visual review have separate receipts.

The remaining entry points in the same experiment package are
`reference_media forecast`, `reference_media media --worker 0 --workers 3`
(with disjoint workers 1 and 2), `reference_validate`, and
`reference_report report`. After the full aggregate audit and visual QA pass,
`reference_report update` changes report status without recomputing figures or
scientific tables. `reference_finalize --write` verifies the required gates,
retains a source capsule and seals the completion manifest. Each command takes
`--root`; use a new root for a new experiment rather than overwriting this one.

The optional `reference_shape_limits --root ...` helper is a separate analytic
diagnostic, not another detector evaluation. For a noiseless constant-background
image `X=c+b*u` with positive amplitude b, the center score tends to
`(a-m)/sqrt(v)`, where a is the target response of the conditioned shape u and
m and v are its reference-weighted mean and variance. The identity holds once
the fixed spread floor is inactive. Its 27-row table compares the actual
simulated spatial footprints using the frozen kernels. Fractional centers,
motion history, finite noise, gradients and local-maximum selection are outside
that derivation; the limits do not predict the observed frame counts.

A benefit limited to one neighbor distance supports a local mechanism result.
A better development tradeoff must also preserve isolated and strong-source
recovery and avoid greater nuisance burden. This study cannot select a general
optimum, establish biological precision, confirm statistical CFAR behavior in
new recordings, or demonstrate feedback control.
