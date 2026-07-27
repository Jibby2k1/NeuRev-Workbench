# Spon Ca Burst multi-hypothesis CFAR

## Question

Can a morphology- and crowding-aware CFAR bank recover more of the 79 sparse
Spon Ca Burst labels than the frozen guarded CFAR without relaxing the
full-field quiet false-alarm budget?

The four motivating observations are represented as a factored model rather
than four unrelated detectors:

| Observation | Test region | Reference estimator |
| --- | --- | --- |
| bright center, dark surround | center | classic annulus |
| bright membrane, dark core/surround | shell | dark core plus classic annulus |
| bright center, illuminated neighbors | center | lowest two of four annular sectors |
| bright membrane, illuminated neighbors | shell | dark core plus lowest two annular sectors |

The screen crosses two morphologies, three radii (4, 6, and 8 pixels), two
reference estimators, and two temporal pools (LME and causal coherence): 24
fixed experts. Two label-free, quiet-calibrated fusions are predeclared, for 26
reported methods. Selection of the best single expert after viewing all bursts
is diagnostic only; it is not an unbiased held-out estimate.

## Scientific contracts

- The input is the raw stack after quiet-frame per-pixel median subtraction and
  positive residual scaling. The failed Kalman lane is not silently reused.
- Quiet frames 1800–1899 (UI convention) determine normalization, every expert
  threshold, every expert scale, and both fusion thresholds.
- Each threshold permits five pooled quiet-field NMS peaks across four
  duration-matched pseudo-events. Adding experts therefore cannot silently
  increase the false-alarm budget.
- Unlabeled event pixels are unknown. Precision is not claimed from the sparse
  positive workbook; candidate count at the fixed quiet false-alarm budget is
  reported as the available precision proxy.
- The labels do not yet identify center versus membrane or isolated versus
  crowded observations. Overall recall is measurable, but per-type claims wait
  for morphology annotations.
- UI frames are one-based and inclusive. Array intervals are zero-based and
  half-open. Coordinates are `x=column`, `y=row`.

## Checkpoints and gates

### C0 — read-only preflight

Verify every input and hash, a fresh output root, label bounds, projection
overlay, disk/RAM headroom, CUDA availability, and at least 9216 MiB free GPU
memory. Failure stops without creating the experiment output root.

### C1 — deterministic morphology/crowding screen

Run all 24 experts in bounded batches, calibrate on full-field quiet data, then
evaluate all four bursts. Also evaluate the two predeclared calibrated fusions.
Advance only if the better predeclared fusion exceeds frozen guarded CFAR mean
recall (`0.132763975`) by at least `0.03` and improves at least two of four
burst recalls. Otherwise stop for visual review and morphology annotation.

### C2 — full-field quiet hard-negative fusion

The initial v4 all-expert fusion gate did not pass: it diluted a strong center
expert with membrane branches that produced almost no known-label matches.
The revision first selects an expert using only the three training bursts. That
cross-fitted fixed procedure must exceed guarded CFAR by `0.03` mean recall and
win at least two bursts before fitting anything. Conditional on that check,
mine quiet NMS candidates from the expert-margin tensor and fit a bounded,
input-dependent soft gate using three bursts at a time. Its prior is initialized
from training-burst expert performance; only a per-expert contextual residual
bounded to 10% is tuned at learning rate `0.001`. The held-out burst never tunes
its gate, normalization, or stopping rule. Advance only if cross-fitted fusion
improves cross-fitted fixed selection by at least `0.02` mean recall and does
not lose more than one burst.

V5 exposed an initialization defect: standardizing all 24 training qualities
gave the best expert only 14–17% initial mass, so the model was still a broad
ensemble. It reached `0.21418` mean recall (18/79) and lost three bursts against
nested fixed selection. V6 restricts support to the top two training experts
and applies temperature `0.02`; the best expert starts with 60–81% mass across
folds. The same low learning rate and 10% contextual bound are retained.

V6 corrected the initialization and reached `0.32937` mean recall (27/79):
3/15, 5/20, 10/21, and 9/23, with 53 total candidates. Nested fixed selection
was `0.31584` (26/79) with 59 candidates. Although V6 is a small Pareto
improvement, its mean gain was only `0.01354`, below the predeclared `0.02` C2
gate, and one burst regressed. C3 therefore did not run.

The C3 bounded-kernel component is implemented in
`neurobench/experiments/learnable_contrast/bounded_residual.py`. It initializes
exactly from the fixed kernels, preserves every support mask and unit mass, and
limits multiplicative log-gain to 5% by default. Keeping the implementation
available does not authorize bypassing the failed C2 gate.

### C3 — bounded kernel residuals

Conditional on C2, initialize center, shell, and reference kernels from the C1
bank and permit only small, regularized radial residuals. Use a low learning
rate and full-field quiet hard negatives. Stop if kernel drift exceeds the
declared radius/mass bounds, the quiet calibration budget fails, or validation
does not improve. This stage is tuning, not unconstrained relearning.

### Long-term comparison

Raw Direct (`0.605615942` mean held-out recall) remains the external activation
baseline. A CFAR architecture can pass C1/C2 while remaining below Raw Direct;
it is then an informative component, not yet the preferred detector.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment learnable-contrast multi-cfar-preflight \
  --config examples/spon_ca_burst_multihypothesis_cfar.example.json \
  --artifact-dir Outputs/LearnableContrast/spon_ca_burst_multihypothesis_cfar_v4_preflight

.venv-neurobench/bin/python -m neurobench.cli.main experiment learnable-contrast multi-cfar \
  --config examples/spon_ca_burst_multihypothesis_cfar_v6.example.json
```

The runner writes atomic JSON manifests, fold metrics, resource telemetry, and
progress heartbeats. It never resumes into or overwrites an existing output
root.

## Standalone diagnostic videos

The video generator regenerates a recorded fixed expert, verifies its
quiet-calibrated threshold, and writes one raw-view MP4 per burst. It does not
make side-by-side comparison panels. The heat layer is the temporally pooled
event score and is therefore constant within a burst; the underlying raw frames
advance at the requested review rate. Cyan rings are known sparse-positive
labels, green crosses are matched detections, and orange crosses are unmatched
candidates whose truth remains unknown.

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment learnable-contrast multi-cfar-videos \
  --config examples/spon_ca_burst_multihypothesis_cfar_v6.example.json \
  --results-json Outputs/LearnableContrast/spon_ca_burst_multihypothesis_cfar_v6_sharp_gate/results.json \
  --output-dir Outputs/LearnableContrast/spon_ca_burst_multihypothesis_diagnostic_v1 \
  --expert-id center_r8_sector_censored_causal_coherence \
  --fps 10
```

The generated artifact manifest is
`Outputs/LearnableContrast/spon_ca_burst_multihypothesis_diagnostic_v1/manifest.json`.

## Completed v4 checkpoint

All 24 fixed experts and two label-free fusions completed. The strongest
post-hoc expert was `center_r8_sector_censored_causal_coherence` at `0.34084`
mean recall (28/79 pooled). The all-expert log-mean-exp fusion reached only
`0.15554`; max fusion reached `0.03468`. A leakage-safe nested check selected
experts on three bursts and achieved `0.31584` on the fourth (3/15, 4/20,
9/21, 10/23). This evidence justifies v5 selective gating, but does not justify
claiming that membrane experts fail biologically: morphology types are not yet
annotated.
