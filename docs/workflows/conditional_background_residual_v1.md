# Conditional-background residual screen v1

- Status: draft planning hold; bounded engineering screen only
- Program: `NREV-PRG-0001`
- Experiment: `NREV-EXP-0029`
- Planned run: `NREV-RUN-EXP-0029-SCREEN-20260830-A`
- Bounded engineering-screen execution: authorized by the user's 2026-08-30
  request
- Claim-bearing execution: not authorized
- Maintained configuration:
  `examples/conditional_background_residual_v1.example.json`
- Parent engineering run: `NREV-RUN-EXP-0028-SCREEN-20260829-B`

This protocol tests the concrete hypothesis that the frozen compact JEPA from
the prior screen may still contain a useful conditional background model even
though its direct latent-temporal-change head was weak. The experiment decodes
a background estimate in normalized pixel space, subtracts that estimate from
the observed movie, and applies the same frozen carrier/context/kinetic (HC)
score stack to the signed residual and to the raw comparator.

This is a new experiment, not a reinterpretation or mutation of
`NREV-EXP-0028`. It is self-supervised conditional background prediction, not
reinforcement learning. The bounded one-seed, 500-step run is an engineering
screen. It cannot formally pass or fail a preregistered scientific benefit
gate, resolve the parent experiment, support a claim, or create an evidence
capsule.

The maintained configuration separates authorization dimensions explicitly:
`engineering_screen_execution_authorized=true`, while general scientific and
claim-bearing execution remain false. The runner fails closed for smoke and
screen execution unless the bounded-engineering flag is true; that flag does
not broaden the authorized budget or scientific interpretation.

## Question and decision

### Question

Can a spatially blind pixel decoder attached to the frozen Screen-B JEPA
predict structured background while preserving an injected calcium source in
the signed residual well enough to improve exact-source recovery over the same
HC score stack applied to the observed movie?

### Decision this screen can inform

The screen may answer only whether a separately versioned, multi-seed
conditional-background experiment is worth designing. Advancement requires a
coherent combination of background suppression, signal retention, and source
recovery; no single metric is sufficient. Missing a descriptive threshold
weighs against designing the same follow-up unless a diagnosed implementation
defect justifies a new version; it does not formally fail either experiment.

Crossing the descriptive thresholds does not promote the screen to scientific
evidence. A claim-bearing follow-up must first satisfy the motion and
registration-residual dependency in `NREV-EXP-0025`, freeze a multi-seed design,
and complete the scientific-audit artifact contract.

### What a favorable screen would mean

A favorable screen would support this engineering statement only:

> Under one frozen non-scientific JEPA checkpoint and one 500-step pixel-decoder
> seed, conditional-background residualization retained the exact injected
> source and improved descriptive recovery over the paired raw HC endpoint on
> the registered empirical-background fixture grid.

It would not establish that the decoded movie is biological background, that
the residual is a neuron-only movie, or that the method generalizes beyond the
registered recordings and injections.

## Scientific identity of the residual

For an observed normalized movie

\[
x = b + s + \epsilon,
\]

the decoder produces a conditional prediction \(\hat b\), and the tested movie
is

\[
r = x - \hat b = s + \epsilon + (b - \hat b).
\]

The last term matters. Prediction error can contain background structure,
motion, acquisition artifact, decoder bias, and neural signal absorbed by the
predictor. Therefore the residual is called a **signed conditional residual**,
not "noise plus signal" and not a denoised movie without further validation.
No rectification occurs before the HC feature stack.

This interpretation follows the existing
[denoise-then-difference boundary](../research/DENOISE_THEN_DIFFERENCE.md): a
useful predictor must suppress nuisance while preserving event amplitude,
localization, and temporal structure.

## Frozen parent artifacts

The run reuses the exact normalized clip banks and frozen seed-1001 JEPA
checkpoint from `NREV-RUN-EXP-0028-SCREEN-20260829-B`. It never modifies the
parent output root.

| Parent artifact | Frozen contract |
| --- | --- |
| Artifact index | SHA-256 `6e6c7ef9d8c70c2c52fcb809ea1f4a65578987e558302a3a63b576cb9eb448bd` |
| Resolved parent configuration | SHA-256 `669dd51f1725fa52cac0e29932cbf9d7203795c223f974e5cc1f88ff81b3d046` |
| Data manifest | SHA-256 `732bc2a35dd8eb0d95e9c86cba71763126ee531d3be6d709f8d56ebb22fb0156` |
| Clip-bank manifest | SHA-256 `ca9386a89cc42683fb5962cb4b4839ea4b3fc0918ed8dbab6d3d3aefcde3ac35` |
| Background-window manifest | SHA-256 `ea3cd2a77ae34df3c1718b46aeb3b071c13734b15b33fd9e284a92e809350d25` |
| Paired-injection manifest | SHA-256 `6fcf2aa080d611c084719682605f3e5f4346ae4601d34da1d0c69f747c9f355a` |
| Seed-1001 frozen checkpoint | SHA-256 `1d9a1c261f5d49fc418e211cfa1f3250079ea08da716c72077feb34065c63a67` |
| 512-clip training cache | SHA-256 `d49e927e29044a5d7105d62375f678159fed59f93e7887fc856a435c3504c3bc` |
| 96-clip validation cache | SHA-256 `f253a35601034aa57b6c7bd13cc6aa0d1886f9e534db422014bcbc49d3c3dc38` |

The parent checkpoint is a one-seed, 500-step, explicitly non-scientific JEPA
screen checkpoint. Reuse does not make it validated or claim-bearing.

Before reading any parent array, preflight verifies the artifact index and each
required member hash, shape, and dtype. The training cache has shape
`512 x 32 x 64 x 64` and the validation cache has shape
`96 x 32 x 64 x 64`, both `float32`. The frozen normalization center is
`489.0`, its scale is `289.10699999999997`, and its contract hash is
`ec48d7f89bb0fbf3bce5305ad1ecebcff2b43ad15ef1af7910405df28ecdb4a4`.
No normalization is refitted.

The sanitized raw-video descriptor remains
`research/data-registry/jepa_raw_video_sources_v1.json`. Raw payloads, parent
outputs, private paths, and workstation-specific roots remain outside Git.

## Frozen implementation identity

The bounded run is tied to complete-file SHA-256 identities, not only symbol
names or a moving branch:

| Implementation | Frozen SHA-256 |
| --- | --- |
| Blind-halo decoder, training, tiling, and diagnostics (`jepa_background_residual.py`) | `2bf62a4508405433db96783ea97e9920206f91b3959ce268e405bf0b233af973` |
| Guarded preflight/smoke/screen runner (`jepa_residual_pilot.py`) | `846ec07a342effccdf149a129a0fb2381eb3a3cef8bfbfaabfae0db0e65be29d` |

The maintained configuration also pins every imported Screen-B JEPA data,
model, training, comparator, evaluation, and fixture-reconstruction source.
Any mismatch fails preflight and requires reviewed configuration versioning and
a new run ID; silently repinning a planned or completed run is forbidden.

## Conditional predictor

### Frozen representation provider

The primary provider is the frozen JEPA path

```text
masked normalized video
  -> frozen context encoder
  -> frozen JEPA predictor
  -> latent prediction
```

Every JEPA parameter is frozen before the first decoder step, excluded from the
optimizer, and checked by before/after parameter hashes. The target encoder is
not part of pixel prediction. Checkpoint selection, provider refitting, and
gradient flow into JEPA are forbidden.

The screen also requires the frozen Screen-B random encoder provider with an
independently trained decoder of identical architecture, initialization, and
batch/tile schedule. It is an architecture-specificity control: the primary
pair remains JEPA-residual HC versus raw HC, while random-residual versus raw
and JEPA-residual versus random-residual are required descriptive comparisons.

### Spatially blind geometry

The predictor must not see the target patch at any time. Each 32-frame by
64-by-64 clip is reflect-padded by one 8-pixel spatial token on every side,
forming a 32-by-80-by-80 movie and an 8-by-10-by-10 token grid. For each of the
64 original 8-by-8 spatial target patches:

1. mask a full-time 8-by-3-by-3 token halo, equivalent to all 32 frames over a
   24-by-24-pixel spatial region, intersected with the original movie;
2. apply that original-space mask **before** reflection padding so every
   reflected alias of a hidden halo voxel is also mask-valued;
3. reflect-pad the already masked movie and reapply the padded 3-by-3 token
   mask as an explicit fail-closed guard;
4. run the frozen provider on that masked padded movie;
5. decode the predicted latent grid;
6. write only the central 32-by-8-by-8 target tube; and
7. repeat until every original pixel has been written exactly once.

The one-token reflection pad makes the same 3-by-3 blind halo available at
boundaries. Masking must precede reflection: reflecting the unmasked movie and
then hiding only the nominal padded halo can leak an original halo voxel back
through a reflected alias outside that nominal mask. Preflight requires zero
reflected-alias mask-value mismatches. Padding does not count as an extra
target. A coverage count other than one for any original pixel aborts
preflight.

This full-time spatial blind tube is deliberate. A temporal mask alone could
let a persistent calcium source predict itself from adjacent times; the
full-time halo prevents the target patch and its immediate spatial neighbors
from entering the context at any frame.

### Geometry falsification tests

Every geometry, reflected-alias, fixture-footprint, coverage, and adversarial
invariance check in this section must pass before the first decoder optimizer
step. Repeating them after training is allowed but cannot substitute for the
pre-training fail-closed order.

The deterministic preflight must reproduce both registered source-isolation
bounds across every injected footprint used in the 108-cell fixture grid:

- worst footprint mass outside the 24-by-24 halo at most
  `0.004851305168298853`; and
- worst footprint energy outside the halo at most
  `0.0001163780028541624`.

These are the exact observed float64 reference maxima over all 252 normalized
registered footprints, computed by summing values and squared values outside
the footprint-peak-token-centered 24-by-24 spatial halo and dividing by total
footprint mass and energy, respectively. Preflight allows only `1e-12`
absolute numerical tolerance. The shorter values `0.00485` and `0.000116` are
not valid bounds because they truncate below the observed maxima.

It must also pass an adversarial invariance test at all 64 original target
tiles. Altering only each masked original-space halo must leave the provider
input and corresponding target prediction exactly unchanged after
mask-before-reflect processing, with maximum difference `0.0` and zero
reflected-alias mask-value mismatches. These tests establish input isolation;
they do not prove that the provider will avoid reconstructing a source from
correlated surrounding activity.

### Pixel decoder and training budget

The only trainable layer is one `ConvTranspose3d` mapping 64 latent channels to
one normalized pixel channel with kernel and stride `(4, 8, 8)` and a bias. Its
parameter count is exactly

\[
64 \times 1 \times 4 \times 8 \times 8 + 1 = 16{,}385.
\]

Training uses Smooth-L1 loss with fixed beta `1.0`, evaluated only on the
central 8-by-8 target patch. The fixed optimizer is AdamW with learning rate
`3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, gradient clip
norm `1.0`, and batch size four. CUDA bfloat16 autocast is allowed only when
supported; loss and metrics remain float32.

The decoder runs for exactly 500 steps with initialization seed `6201` and tile
schedule seed `6202`. There is no early stopping, checkpoint selection, or
validation-driven hyperparameter selection. The 96 validation clips are
hash-verified and reserved; they do not enter training, selection, or this
bounded screen's primary endpoint. The hard implementation cap is 2,000 steps,
but this screen is fixed at 500; increasing the budget requires a new run ID
and a reviewed configuration. Full-background inference batches eight spatial
target tubes at a time; this changes memory use only and not the 64-tile
prediction identity.

For each provider arm, the decoder core hashes the complete sequence of
zero-based step index, target-token flat index, and ordered training-clip
indices. The runner must record
`sampling_schedule_sha256_equal_between_provider_arms=true`; any hash mismatch
fails closed. Matching only the sparsely logged training-curve rows is not an
adequate fairness check.

## Evaluation

### Exact paired fixture grid

Evaluation reconstructs the same 12 background windows and nine injection
conditions per window as Screen B: three source counts (`1`, `2`, and `4`) by
three injection seeds (`3101`, `3102`, and `3103`). The total is 108 paired
source-off/source-on cells containing 252 exact injected sources.

The fixture implementation and constants remain those of
`native_background_calcium_injection_v1`. Every rebuilt cell must close under
the exact float32 injection contract and reproduce the frozen parent fixture
identity before scoring. `footprint_peak_centers_yx` is primary truth;
`placement_centers_yx` is a morphology sensitivity analysis.

The unit of analysis is recording by disjoint background window by injection
seed, with source count nested. Injected sources are not independent biological
replicates. Results may not be described as independent-animal validation.

### Frozen endpoint

Every endpoint uses the same `FrozenHandcraftedComparator`:

| Method | Movie supplied to HC |
| --- | --- |
| `raw_hc` | normalized observed movie |
| `jepa_conditional_residual_hc` | signed observed-minus-predicted residual |
| `random_conditional_residual_hc` | signed residual from the identically trained frozen-random-provider decoder |

JEPA residual minus raw is the primary descriptive pair. Random residual minus
raw tests whether the architecture and decoder alone reproduce an apparent
benefit. JEPA residual minus random residual tests whether any gain is specific
to the trained JEPA provider rather than the conditional-decoder machinery.

The HC carrier, spatial-context, and kinetic components are calibrated only
from each source-off movie through the structural
`fit_source_off(...) -> frozen_callable` interface. The callable is applied
unchanged to source-off and source-on. Source-on self-normalization, pair-wise
joint calibration, label fitting, and per-method threshold tuning are
forbidden.

Each score map returns up to four separated local maxima with minimum distance
two pixels, never synthetic fillers. Exact centers are matched one-to-one
within three pixels. Primary descriptive performance is macro source recall;
actual candidate count, intervention recall, source-pixel discrimination, and
placement-center sensitivity remain secondary diagnostics.

Before interpreting a residual comparison, `raw_hc` is regression-checked
against the `frozen_handcrafted_stack` anchor stored by Screen B. All 108 screen
fixtures must match exactly for `source_on_recovery` and
`intervention_recovery`, including recall and the complete serialized
`RecoveryResult`; numeric tolerance is zero. The smoke mode checks its exact
execution subset. Any mismatch ID or field fails closed and is written to
`raw_hc_cross_run_regression.json`.

### Paired signal-retention accounting

For every fixture define

\[
\Delta r = r_{on} - r_{off}, \qquad
\Delta \hat b = \hat b_{on} - \hat b_{off}, \qquad
s = x_{on} - x_{off}.
\]

The aligned retained gain is

\[
g_r = \frac{\langle \Delta r, s \rangle}{\langle s,s \rangle},
\]

and predictor absorption is

\[
g_b = \frac{\langle \Delta \hat b, s \rangle}{\langle s,s \rangle}.
\]

Because `r = x - prediction`, the projection closure is `g_r + g_b = 1` up
to floating-point error. The run separately checks
`maximum_projection_closure_absolute_error` for `|g_r + g_b - 1|` and
`maximum_voxelwise_pair_closure_absolute_error` for
`max|delta_r + delta_prediction - s|`; both must be at most `1e-5`. These are
engineering integrity requirements, not scientific benefit gates. Two
non-interchangeable error ratios are also reported:

\[
e_{total} = \frac{\lVert \Delta r-s \rVert_2}{\lVert s \rVert_2},
\qquad
e_{orth} = \frac{\lVert \Delta r-g_r s \rVert_2}{\lVert s \rVert_2}.
\]

`total_signal_error_ratio` includes both gain error and off-axis distortion;
`orthogonal_distortion_ratio` removes the aligned retained component before
measuring distortion. Neither may be silently substituted for the other.

Background suppression is measured on source-off movies as centered residual
RMS divided by centered input RMS and residual normal-consistent MAD divided by
centered input MAD. A separate dynamic-MAD ratio applies one-frame temporal
differences to both movies before the same robust scale. Lower ratios indicate
stronger suppression, but only when retention and recovery remain adequate.

### Descriptive uncertainty

The screen reports 1,000-draw hierarchical bootstraps over recording,
background window, and injection seed with seed `6203` for JEPA residual minus
raw, random residual minus raw, and JEPA residual minus random residual. These
intervals are descriptive stability summaries for the bounded fixture panel.
One upstream JEPA seed and one decoder seed per provider cannot establish
training-seed stability or formal scientific uncertainty.

## Descriptive advancement panel

The following values are engineering triage thresholds, not registered
scientific pass/fail gates:

| Quantity | Descriptive advancement value |
| --- | ---: |
| JEPA-residual HC minus raw HC macro recall | at least `+0.02` |
| JEPA-minus-raw grouped interval lower bound | greater than `0.0` |
| JEPA-residual HC minus random-residual HC observed mean | greater than `0.0` |
| JEPA-minus-random grouped interval lower bound | greater than `0.0` preferred stronger evidence |
| Median aligned retained gain | at least `0.90` |
| Median predictor absorption | at most `0.10` |
| Median background RMS ratio | at most `0.90` |
| Recording-wise recall direction | nonnegative in every recording |

Projection and voxelwise closure are intentionally absent from this benefit
table because they are mandatory engineering-integrity checks rather than
evidence of useful background separation.

The combined pattern determines whether a follow-up is worth designing:

- Suppression with low retained gain means the predictor absorbed injected
  signal and is scientifically unsafe.
- Retention without suppression means the decoder did not isolate useful
  background.
- Suppression and retention without recovery gain means the residual is not a
  better input to this HC endpoint.
- Recovery gain without closure or geometry validity is invalid.
- A JEPA-residual gain equaled or exceeded by the frozen-random residual does
  not support a JEPA-specific mechanism, even if both residuals beat raw.
- Crossing every descriptive value still requires a new multi-seed protocol;
  it does not resolve any formal experiment gate.

## Falsifiers and stop conditions

This screen is falsified as a clean test of the proposed mechanism if any of
the following occurs:

- parent artifacts, caches, checkpoint, or implementation hashes mismatch;
- any JEPA parameter changes or enters the decoder optimizer;
- the target-tube adversarial invariance difference is nonzero or a reflected
  alias of an original-space halo voxel is not mask-valued;
- the 64-tile reconstruction has a gap or overlapping write;
- registered footprint leakage exceeds either frozen geometry bound;
- source-on information enters calibration or decoder selection;
- projection closure or voxelwise pair closure exceeds `1e-5` absolute error;
- any required numeric output is missing, inconsistent, or nonfinite; or
- the run overwrites an existing output root.

Scientifically unfavorable descriptive values are reported as screen findings,
not mislabeled as formal preregistered gate failures.

## Execution modes

The configuration drives all modes; command-line overrides of frozen counts are
not allowed for the registered screen. Smoke and screen modes additionally
require the explicit bounded-engineering authorization flag; scientific and
claim-bearing authorization remain false.

| Mode | Contract | Status |
| --- | --- | --- |
| `preflight` | read-only hashes, geometry, device, resource, and collision checks | no output mutation |
| `smoke` | 8 train clips, 4 validation clips, 1 decoder step, 1 background, 1 fixture cell, CPU | implementation-only |
| `screen` | 512 train clips, 96 validation clips, 500 decoder steps, 12 backgrounds, 9 fixture cells each, CUDA | bounded non-claim-bearing screen |

The registered screen output root is

```text
Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/
  NREV-RUN-EXP-0029-SCREEN-20260830-A
```

It must be absent immediately before execution. Work is staged under a
non-colliding `.partial` path and promoted atomically only after numeric output
validation. Completed roots are never overwritten.

Before a CUDA screen, verify at least 20 GiB free disk, 8 GiB available RAM,
bfloat16 support, at least 8,192 MiB free CUDA memory, all input hashes, and the
output collision guard. Conflicting-process inspection is an explicitly
operator-observed external preflight; the runner must not claim to infer all
host processes from a sandboxed process view. That observation and its timing
must be recorded before screen execution. The run uses four CPU threads, atomic
metadata, and a progress heartbeat.

## Scientific audit and publication boundary

The [Scientific Audit Output Standard](SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) is
enabled by default. This bounded screen intentionally does not produce the
complete model-candidate video, close-up, trace, and standardized comparison
media required for audit completion. It must therefore write
`scientific_audit_status.json` with status `incomplete` and keep
`scientific_completion=false` and `scientific_promotion_allowed=false`.

Because injection truth supplies no expert biological annotations, the Expert
section is `not_applicable`; it is never silently omitted. A claim-bearing
follow-up would require a frozen candidate-surrogate Model section, its
full-field and close-up evidence, the standardized Comparison section, media
decode validation, and a passing inventory.

This screen creates no claim record and no evidence capsule. Public registry
records contain only sanitized portable metadata. Raw video, parent output
payloads, local absolute paths, credentials, reviewer identities, and private
material remain outside Git.

## Required small outputs

Agents inspect `llm_context.json`, `summary.json`, `artifact_index.json`, and
`validation.json` before any large artifact. The bounded screen must emit at
least:

```text
resolved_config.json
preflight.json
status.json
heartbeat.json
decoder_training_metrics.json
decoder_checkpoints_manifest.json
clip_reuse_manifest.json
background_window_manifest.json
blind_tube_geometry_validation.json
paired_injection_manifest.json
paired_injection_results.tsv
paired_injection_results.json
raw_hc_cross_run_regression.json
residual_diagnostics.tsv
residual_diagnostics.json
freeze_integrity.json
upstream_integrity.json
summary.json
validation.json
artifact_index.json
llm_context.json
scientific_audit_status.json
REPORT.md
```

The summary and report must state the exact 108-cell/252-source denominators,
actual proposal counts, provider and decoder seeds, parent and implementation
hashes, background suppression, retained gain, absorption, closure, total
signal error, orthogonal distortion, all three raw/JEPA/random recall
comparisons and grouped intervals, recording-wise direction, incomplete
scientific audit, unresolved motion dependency, and the non-claim-bearing
interpretation.

## Registered planning gates

The native registry retains mandatory safety and validity gates. They govern
whether the screen is interpretable; they do not convert the descriptive
advancement panel into a formal benefit gate.

| Gate | Stage | Requirement |
| --- | --- | --- |
| `output_collision` | preflight | The exact registered output root is absent. |
| `parent_artifact_identity` | preflight | Every required Screen-B artifact hash, cache shape, and dtype matches. |
| `motion_dependency` | preflight | `NREV-EXP-0025` must be evidence-backed before claim-bearing execution; the screen remains non-claim-bearing while unresolved. |
| `frozen_provider` | execution | All JEPA provider parameters remain frozen, optimizer-excluded, and hash-identical before and after training. |
| `screen_scope` | execution | Bounded-engineering authorization is true while scientific and claim-bearing authorization remain false; the run uses one upstream JEPA seed, matched JEPA/random decoder schedules, and 500 steps. |
| `blind_tube_geometry` | preflight | Exact 64-tile coverage, mask-before-reflect order, zero unmasked reflected aliases, full-time 3-by-3 halo, registered footprint leakage bounds, and zero adversarial target-tube influence at all 64 tiles pass. |
| `pair_safe_evaluation` | analysis | Calibration uses source-off only and exact off/on pairs, truth centers, proposal cap, and grouping remain frozen. |
| `retention_accounting` | analysis | Retention, absorption, total error, orthogonal distortion, suppression, projection closure, and voxelwise closure are finite; both closure errors are at most `1e-5`. |
| `scientific_audit` | review | Audit remains incomplete for this screen; scientific completion is forbidden until the full applicable package passes. |
| `publication_boundary` | publication | No claim or capsule is created from the engineering screen. |

## Current hold

`NREV-EXP-0025` remains draft and not evaluated, so nuisance-robust and
claim-bearing execution is blocked. The planned run is one frozen upstream JEPA
seed and one matched 500-step decoder schedule for each JEPA/random provider.
It is not a substitute for the parent three-seed, 5,000-step design and cannot
formally adjudicate that design. The planning decision remains `hold` until a
validated result justifies a separate next experiment.
