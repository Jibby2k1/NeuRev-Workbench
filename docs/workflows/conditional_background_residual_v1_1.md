# Conditional-background residual screen v1.1

- Status: bounded engineering screen complete; post-screen hold; non-claim-bearing
- Program: `NREV-PRG-0001`
- Experiment: `NREV-EXP-0029`
- Failed immutable run: `NREV-RUN-EXP-0029-SCREEN-20260830-A`
- Completed engineering run: `NREV-RUN-EXP-0029-SCREEN-20260830-B`
- Maintained configuration:
  `examples/conditional_background_residual_v1_1.example.json`
- Versioned runner:
  `neurobench/experiments/neuron_identifiability/jepa_residual_pilot_v1_1.py`
- Parent engineering run: `NREV-RUN-EXP-0028-SCREEN-20260829-B`
- Claim-bearing execution: not authorized

This document amends the frozen
[v1 protocol](conditional_background_residual_v1.md) after run A stopped at
its exact raw-HC cross-run guard. It does not erase, overwrite, resume, or
reinterpret run A. It changes one endpoint-domain mistake: the raw-HC anchor is
now scored in the same native `float32` raw units used by the immutable
EXP-0028 Screen-B parent. The JEPA and random-provider conditional residuals
remain in the frozen training-normalized pixel domain.

All other scientific boundaries remain those of v1. In particular, the
fixture grid, injected truth, normalization values, checkpoint, decoder,
seeds, training schedule, blind-tube geometry, source-off calibration,
proposal cap, matching radius, grouped analysis, descriptive thresholds, and
claim restrictions do not change. Run B is a fresh execution into a fresh
non-colliding output root, not a continuation of A.

## Why run A is retained as a failure

Run A passed its read-only preflight, hash checks, blind-tube geometry checks,
and completed the matched 500-step decoder training. It then failed closed at
`raw_hc_cross_run_regression` with the message:

> raw-HC cross-run anchor drifted from EXP-0028 Screen B

The v1 runner had supplied normalized pixels to the raw-HC comparator. Screen
B's frozen handcrafted endpoint had been fitted and applied directly in native
raw `float32` units. The raw HC implementation is source-off calibrated, but
the full discrete `RecoveryResult` is not guaranteed to be invariant to this
domain change because score ties and local-maximum ordering can change at
floating-point precision.

The guard therefore behaved correctly. Run A remains
`failed_nonresumable_partial`; its partial output is preserved and its small,
sanitized, byte-exact provenance is copied under
`research/run-provenance/NREV-RUN-EXP-0029-SCREEN-20260830-A/`. There is no
scientific evidence capsule and no residual-endpoint result from A.

### Exact run-A mismatch audit

An independent read-only reconstruction separated substantive recovery from
serialization drift:

- all 108 source-on `RecoveryResult` objects were exact;
- intervention recall, recovered-source identity, matched source/candidate
  indices, and localization-error arrays were exact for all 108 fixtures;
- intervention candidate coordinate lists differed in 53 of 108 fixtures;
- only four of 108 intervention results changed candidate cardinality and the
  derived unmatched-candidate count; and
- rerunning the comparator in native raw units reproduced all 216 parent
  source-on/intervention `RecoveryResult` objects exactly.

The four cardinality mismatches were:

| Fixture | Parent candidates / unmatched unknown | v1 normalized candidates / unmatched unknown | Recall | Localization error |
| --- | ---: | ---: | ---: | ---: |
| `060126_10_rest__window_2_median_mad__sources_1__seed_3102` | `2 / 1` | `3 / 2` | `1.0` in both | `1.0 px` in both |
| `060126_12_left__window_1_low_mad__sources_1__seed_3102` | `4 / 3` | `3 / 2` | `1.0` in both | `1.0 px` in both |
| `060126_15_right__window_3_high_mad__sources_1__seed_3102` | `3 / 2` | `4 / 3` | `1.0` in both | `0.0 px` in both |
| `spon_ca_burst_3_hindbrain_to_tail_488_20ms__window_1_low_mad__sources_1__seed_3103` | `4 / 3` | `3 / 2` | `1.0` in both | `1.0 px` in both |

For these four fixtures, source-on recall remained `0.0` and source-on
candidate cardinality remained four in both evaluations. The failed guard is
not described as a changed scientific recall or localization result. It is an
exact-continuity failure caused by an incorrect raw-arm input domain, detected
before residual comparisons were accepted.

## The v1.1 correction

The v1.1 endpoint domains are explicit:

| Arm | HC input domain | Normalization applications by v1.1 |
| --- | --- | ---: |
| `raw_hc` | native observed/source-off `float32` movie, exactly as EXP-0028 Screen B | `0` |
| `jepa_conditional_residual_hc` | signed normalized observation minus frozen-JEPA conditional prediction | `1` for each raw fixture movie before prediction |
| `random_conditional_residual_hc` | signed normalized observation minus matched frozen-random-provider conditional prediction | `1` for each raw fixture movie before prediction |

No cross-domain score magnitude is interpreted. The comparison remains at the
level of one-to-one exact-source recovery and macro source recall after each
arm's frozen source-off-only calibration.

The v1.1 runner first reconstructs the complete native raw-HC anchor, then
requires exact equality with both parent recovery objects per fixture:

```text
108 fixtures x (source-on RecoveryResult + intervention RecoveryResult)
  = 216 exactly equal serialized RecoveryResult objects
```

Only after this gate passes may the runner evaluate or interpret either
conditional-residual arm.

### The guard is not relaxed

Run B does not lower the standard that stopped A:

- numeric tolerance remains exactly `0.0`;
- complete candidate coordinates, cardinality, unmatched count, matched
  indices, recovery count, recall, and localization errors must all match;
- recall-only equivalence is explicitly insufficient;
- there is no exception for tie-related coordinate or candidate-count drift;
- all 108 fixtures and all 216 parent recovery objects are checked, including
  smoke mode;
- source counts, seeds, fixture identities, proposal cap, separation, border,
  match radius, and calibration API are unchanged; and
- any mismatch still aborts before residual-endpoint interpretation.

The correction changes the raw comparator input back to its frozen parent
domain; it does not ignore or tolerate the mismatches.

## Unchanged predictor and evaluation contract

Run B reuses the hash-verified EXP-0028 Screen-B seed-1001 non-scientific JEPA
checkpoint and the existing 512/96 normalized training/validation clip banks.
Each frozen JEPA/random provider receives the same masked context schedule. An
independent 16,385-parameter pixel decoder is trained for each provider with
the shared seed, batches, target tiles, optimizer, and 500-step cap.

The blind-tube contract is unchanged:

- each input is 32 by 64 by 64 pixels;
- a full-time 3-by-3 spatial-token halo is masked in original space before
  reflection and reapplied after reflect padding;
- only the central 32-by-8-by-8 target patch is decoded;
- 64 writes cover every output voxel exactly once;
- all 64 target-tube adversarial invariance checks must be bitwise exact; and
- the registered float64 footprint-leakage bounds over all 252 sources remain
  frozen.

The exact paired grid remains 12 background windows by three injection seeds
by source counts 1, 2, and 4: 108 cells containing 252 injected sources.
Calibration receives source-off only. The signed-residual identity remains

\[
r = x - \hat b = s + \epsilon + (b - \hat b),
\]

so the prediction is not assumed to be true biological background and the
residual is not assumed to be pure neural signal or denoised video.

## Execution-mode coverage

The word "smoke" applies only to decoder/residual work after the parent anchor
has passed:

| Mode | Raw-HC parent anchor | Decoder/residual subset | Scientific status |
| --- | --- | --- | --- |
| `preflight` | configuration, hashes, geometry, resources, and collision only | none | read-only |
| `smoke` | full 108 fixtures / 216 recovery objects | 8 train clips, 4 reserved validation clips, 1 decoder step, 1 background, 1 residual fixture | implementation-only |
| `screen` | full 108 fixtures / 216 recovery objects | 512 train clips, 96 reserved validation clips, 500 decoder steps, all 108 residual fixtures | bounded non-claim-bearing engineering screen |

The screen output root is:

```text
Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/
  NREV-RUN-EXP-0029-SCREEN-20260830-B
```

Both this root and its `.partial` sibling must be absent immediately before
execution. The A root and A `.partial` directory are immutable and are never
reused or removed by B.

## Coverage and required outputs

In addition to every v1 required output, v1.1 requires
`prediction_coverage.json`. It records the returned prediction coverage for
both JEPA and random-provider source-off windows and every evaluated source-on
fixture. A screen cannot complete if any prediction has a missing or repeated
tile write, a nonfinite value, or a shape inconsistent with the registered
32-by-64-by-64 movie. The complete screen must contain exactly 240 coverage
rows: 12 source-off windows by two provider arms, plus 108 source-on fixtures
by two provider arms. Every row must report 64 tiles and per-voxel minimum and
maximum coverage both equal to one.

The complete required small-output set is:

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
prediction_coverage.json
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

## Version and implementation identity

The maintained descriptor requires `schema_version=2`,
`runner_version="1.1"`, the exact B run ID, and the exact B output root. The
v1 runner accepts only schema 1, while the v1.1 runner accepts only schema 2
and run B; this binds each runner to its own descriptor in both directions. It
pins the new versioned runner as a complete file and retains the unchanged
hashes for every shared v1 data, model, training, comparator, evaluation,
fixture, and blind-halo source.

The v1 runner, v1 configuration, v1 protocol, and shared decoder/core files are
not edited by this amendment. Run A remains reproducible against their frozen
hashes. After 48 focused tests passed, the implementation owner declared the
v1.1 runner stable at SHA-256
`e3b4cb4cebe155b2a46c1c8b4ff810b28ec3bedb5415b5aed1e1097f01c531d9`.
A pending or mismatched hash fails preflight.

## Execution history and screen result

Run A remains the immutable safe failure described above. Run B used the
versioned schema-2 correction, reproduced all 108 raw-HC fixtures and all 216
parent recovery objects exactly, completed both 500-step decoder arms, and
evaluated the registered 108-fixture, 252-source grid. All 240 prediction rows
had finite, shape-exact, exactly-once coverage, both algebraic closure limits
passed, the frozen providers remained unchanged, and the 27 emitted artifacts
verified against their index.

The observed macro recalls were `0.1875` for native raw HC,
`0.09027777777777778` for JEPA-residual HC, and `0.06712962962962964` for the
random-provider residual. JEPA residual minus raw was
`-0.09722222222222222`, with grouped 95% interval
`[-0.24305555555555552, 0.027835648148148106]`. JEPA retained nearly all
aligned injected amplitude (`0.9990785812365571` median retained gain), but
its median background RMS, dynamic-MAD, and seam-to-interior ratios were
`1.7102741349511534`, `1.1093322124630185`, and `2.5786383127702`. The
registered background and seam diagnostics therefore increased rather than
showing the intended suppression, and the raw endpoint had the strongest
recall point estimate.

See the sanitized
[v1.1 screen results](../research/CONDITIONAL_BACKGROUND_RESIDUAL_V1_1_RESULTS.md)
for the paired intervals, recording-wise effects, signal accounting, and
portable provenance hashes.

## Interpretation and hold

The v1 descriptive advancement panel was engineering triage, not a formal
scientific gate. Its observed pattern does not justify escalating this exact
nonoverlapping tiled one-layer decoder unchanged. It does not reject the
broader conditional-background-learning hypothesis; a future attempt would
need a separately versioned design that directly addresses patch seams and
prediction-error amplification, plus multiple seeds.

The motion dependency `NREV-EXP-0025`, independent-recording generalization,
biological identity evidence, and complete scientific-audit media remain
unresolved. Numeric execution succeeded, but scientific outputs remain
partial because full-field videos, close-ups, full-duration traces, matched
comparison figures, and decode validation were not produced.

This amendment and screen create no claim and no evidence capsule. The
experiment stays `draft`, `not_evaluated`, and evidence tier `none`;
`scientific_completion=false` and `scientific_promotion_allowed=false` remain
mandatory.

## Post-screen derived diagnostics

Two integrity-checked frozen derived artifacts were added after Run B without
changing the frozen protocol or retroactively creating a gate. The rank
diagnostic exactly reconstructed all 324 Run-B recovery objects and separated
the operational intervention-recovered/source-on-missed pattern from sources
missed by both top-four maps. A source-off-only safety audit then applied
post-screen no-amplification thresholds and rejected all 12 JEPA and all 12
random-residual windows, falling back to raw for every fixture.

See the sanitized
[derived-diagnostic results](../research/CONDITIONAL_BACKGROUND_RESIDUAL_DERIVED_DIAGNOSTICS_V1_RESULTS.md)
for exact counts, grouped uncertainty, portable hashes, and the limitations of
the emitted category names. Rank F records complete frozen-input,
numerical-dependency, Git/runtime/command/timestamp provenance; safety v1.1 is
a fixed policy artifact rather than a run. These audits strengthen the
implementation hold, but neither is a registered scientific execution, and
they do not identify competitor biology, prove attenuation, complete the
scientific audit, or establish residual benefit.
