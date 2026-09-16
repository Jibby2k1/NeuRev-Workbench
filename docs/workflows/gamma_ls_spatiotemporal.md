# Gamma CFAR spatial and causal temporal sensitivity study

This implements the approved September 14, 2026 proposal in the current paper's
editorial folder. The contribution under investigation is Gamma local
standardization inside CFAR prescreening. Conditioning is held fixed: spatial
Gaussian sigma 1 pixel, causal exponential average alpha 0.4, and signed adjacent
frame difference. No learning, stimulation, or controller experiment is included.

The acquisition metadata supplied by the user are 50 Hz and 0.5 micrometers per
pixel. The reported approximately 30 Hz indicator bandwidth is not converted
into a decay constant. The recording's Nyquist frequency is 25 Hz. Calibration
traces describe observed fluorescence kinetics, including neural drive and
measurement effects; they cannot identify the indicator impulse response.

## Frozen experiment

The 21 configurations comprise 12 radius/time combinations (R = 5, 7.5, 10
pixels; T = 0, 20, 60, 200 milliseconds), one separable marginal control at
R = 7.5 and T = 60, four target-width controls (sigma = 2 or 4 pixels; T = 0 or
60), and four reference-order controls (n = 3 or 15; T = 0 or 60). Remaining
settings are target sigma 1 and reference n = 9. T = 0 means spatial only.

Both target and reference operate directly on the same conditioned input.
There is no explicit guard mask. The target uses the centered exponential
profile; the reference uses q^(n-1) exp(-(n-1)q/R), where q is the radial
distance in x, y, and nonnegative past lag scaled by R/T. The temporal coordinate
is omitted for 2D. T describes the continuous reference profile peak on the
time axis, not a decay constant, mean lag, maximum history, or decision delay.
The actual discrete marginal moments, supports, and mass retention are saved.
The separable control preserves each joint kernel's own spatial and temporal
marginals; it is not exponential smoothing of the final standardized score.

The same 16 movies (eight stress templates, two fixed seeds) are used in every
configuration. Every movie has at least 64 warmup frames, then 100 setup frames,
then 300 application frames. A 128 by 128 pixel evaluation box is surrounded by
the maximum kernel radius plus the conditioning Gaussian support. Noise,
compact/broad footprints, rapid/slow fluorescence, persistence, repeated
activity, crowding, and nuisance-only changes are represented. These are
specified simulations, not estimates of this recording's biological truth.

The real recording is loaded from source UI 1400 through 2359. UI 1400–1599
warms the pipeline, UI 1600–1799 calibrates it, and UI 1800–2359 is the
application segment. All settings use the same field interior with complete
source halo and the same maintained six-pixel NMS border. Geometrically excluded
existing labels are counted explicitly. Descriptive trace calibration may use
UI 1–1799; it does not select detector parameters or application thresholds.

The denominator floor is the positive setup reference-standard-deviation tenth
percentile, with a minimum of 1e-6. Its application clipping fraction is reported.
The audit cutoff is the strict setup NMS order statistic satisfying a nominal
budget of one proposal per 194,820 pixels per frame, scaled to the actual
eligible area. Calibration starts after warmup and uses no application values
or annotations. It controls setup proposal burden, not application false-alarm
probability. A nonnegative threshold is required for the cached positive NMS
prefix. All thresholds from 0 through 10 in increments of 0.25, plus 12, 15,
20 and an explicit no-output endpoint, are evaluated numerically.

## Interpretation and output contract

Synthetic precision is the fraction of emitted frame/location proposals that
match a simulated active source. Sensitivity is the fraction of active
source/frame locations recovered. Matching is one-to-one within six pixels.
Separate outputs report event-window recovery, first-detection delay from the
simulated onset, duplicate proposals, and false proposals per eligible area-time.
Persistence can improve a fluorescence trace while reducing signed-difference
framewise sensitivity; the event-level readout exposes this distinction.

The existing real labels are sparse positive occurrence windows. Their coverage
and proposal burden are descriptive, and unmatched candidates remain unknown.
Real precision, sensitivity against exhaustive truth, and false-alarm rates
remain unavailable until all frames of three geometry-selected 128-pixel panels
are reviewed, uncertainty is masked, and the annotation coverage gate passes.
Panel selection is independent of detector scores. No independent recording or
closed-loop control validation is claimed.

The full scientific audit is enabled at the frozen area-scaled operating point
for every evaluated dataset/configuration. It includes section-pure expert and
model full-field videos, every expert and consolidated model location's closeup
and full trace, all expert-occurrence comparisons, metadata, a small context
index, a report, and complete lossless-video decoding and marker checks.
Nuisance-only clips explicitly mark Expert not applicable. Numerical threshold
curves do not imply that every threshold has a separate media audit. There is no
user opt-out. A numerical result is not audit-complete until its inventory and
media checks pass.

## Reproduction

Run from the substantive checkout with `.venv-neurobench/bin/python`:

```bash
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_study preflight
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_study prepare
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_study run --device cuda --chunk-frames 16
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_study evaluate
```

Use the repository Python path in place of `python` above. Set numerical and
codec thread counts to one, use low scheduling priority and the approved CPU
affinity, and verify current resources first. CUDA requires the CPU-oracle
parity checks to pass. The runner uses bounded chunks, atomic metadata, immutable
candidate seals, and new output roots. Timing from buffered chunks is throughput
evidence; streaming decision latency must be measured separately.

The current output root is
`Outputs/GammaLSST/sensitivity_20260914_r2`. Completed roots are never overwritten.
The r1 folder contains only a superseded preflight, with no prepared data or scores.
The prior two-stencil/control studies and manuscript remain separate evidence.

### Postprocessing and review

The original evaluation routine is exact but slow for dense low-threshold real
candidate streams. An optional spatial-hash backend preserves the original
representative ordering and inclusive matching radius. Its parity command must
pass before its resume command runs. Resume binds already completed numerical
files and verifies they remain unchanged:

```bash
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_sparse_fast parity --output Outputs/GammaLSST/sensitivity_20260914_r2/sparse_fast_parity_v1.json --real-prefix Outputs/GammaLSST/sensitivity_20260914_r2/cells/real/R7.5_T0_S1_n9_joint/prefix.json
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_sparse_fast resume --root Outputs/GammaLSST/sensitivity_20260914_r2 --parity-report Outputs/GammaLSST/sensitivity_20260914_r2/sparse_fast_parity_v1.json
```

These are reproduction commands; do not overwrite the completed parity record
or rerun an accepted backend amendment in an existing completed root. The
recorded r2 execution resumed after the original evaluator completed all
synthetic cells and six real cells. It changed no scientific matching rule,
threshold, candidate, or previously completed numerical output.

After numerical completion, create the common display contract and dispatch
disjoint media workers. The commands below are serial examples; each worker
selects its own third of the same 357-cell inventory. Use at most three workers,
each with one numerical/codec thread and a separate approved CPU affinity.

```bash
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_media display
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_media run --workers 3 --worker 0
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_media run --workers 3 --worker 1
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_media run --workers 3 --worker 2
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_media validate
python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_report --root Outputs/GammaLSST/sensitivity_20260914_r2
```

Visual review is required separately from plot generation. The initial audit
trace layout and initial report layouts were preserved after visual review
found overlapping or clipped labels. Rendering-only amendments preserve the
frozen scores, labels, thresholds, and numerical tables. The final media tree
uses the corrected renderer; `audits_layout_v1_superseded` retains the earlier
partial rendering history and its original absolute paths are historical.

If the report already exists and only audit completion has changed, use
`spatiotemporal_report --root Outputs/GammaLSST/sensitivity_20260914_r2
--refresh-status` to refresh completion prose and metadata while preserving
the reviewed figures. Generation or a numerical PASS alone is not a completed
scientific media audit.

The separate streaming benchmark is
`python -m neurobench.experiments.gamma_ls_difference.spatiotemporal_latency
--root Outputs/GammaLSST/sensitivity_20260914_r2`. It uses fresh-process
repetitions, one-frame arrival timing, exact stage/candidate parity, and
retained backlog. Its interval excludes acquisition, conditioning, communication,
actuation, and biological delay. A benchmark PASS means that its measurements
are valid, not that every arm meets the 20 ms deadline.

The raw recording review pack is under `real_review/raw_media_v1/index.html`.
The annotation acceptance record remains false until a human completes the
coverage and uncertainty review and binds the accepted annotation files. Empty
annotation tables do not provide negative labels.
