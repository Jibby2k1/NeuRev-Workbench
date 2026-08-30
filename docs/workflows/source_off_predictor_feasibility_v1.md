# Source-off conditional-background predictor feasibility v1

- Status: bounded engineering benchmark complete; no predictor admitted
- Program: `NREV-PRG-0001`
- Experiment: `NREV-EXP-0030`
- Canonical run: `NREV-RUN-EXP-0030-SCREEN-20260830-B`
- Maintained descriptor:
  `examples/source_off_predictor_feasibility_v1.example.json`
- Versioned runner:
  `neurobench/experiments/neuron_identifiability/source_off_predictor_feasibility.py`
- Claim-bearing execution: not authorized

This protocol asks a narrow question before another residual detector is
attempted: can any frozen background predictor reduce registered source-off
variation without worsening dynamics, seams, temporal whiteness, or
recording-level consistency? A pass would authorize only a separately
versioned injected-source detector screen. It would not establish denoising,
neuron preservation, biological background identity, or scientific benefit.

## Frozen data separation

The benchmark uses three roles with disjoint exact clip/window samples:

1. `512` normalized training clips fit the simple learned predictors.
2. All `96` recording-held validation clips evaluate every method, but never
   fit a parameter or select a hyperparameter.
3. The exact `12` registered quiet/source-off windows from EXP-0029 provide a
   second fixed evaluation panel.

No ROI labels, injected-source truth, detector scores, recovery results, or
biological annotations enter fitting, selection, or the gate. “Source-off”
means no synthetic source was injected; native neural activity in these
unlabeled clips remains unknown. The two evaluation panels are not independent
recording panels: nine of the 12 registered windows share recording IDs
`060126_10_rest`, `060126_12_left`, or `060126_15_right` with the held-validation
panel. Disjoint samples therefore do not imply independent recordings.

Every method is evaluated on the same zero-based frame interval `[8, 32)`, or
24 frames per clip. Raw source-off windows receive the frozen normalization
exactly once; already normalized clip banks are not normalized again.

The serialized `validation_not_used_for_fit_or_selection` field refers to
pre-gate fitting, tuning, candidate-panel selection, checkpoint selection,
threshold selection, and favorable-subset reporting. The evaluation panels
necessarily enter the predeclared eligibility gate; that gate selected zero
subtractors.

## Frozen predictor panel

The eight methods are:

- no subtraction, the identity reference;
- last-frame or causal zero-order hold;
- causal exponential moving average with `alpha=0.25`;
- causal temporal median with a seven-frame history;
- per-pixel AR(1), fitted on the 512 training clips;
- rank-eight low-rank spatial-basis/latent AR(1), fitted on those same clips;
- the frozen final-step EXP-0029 Run-B JEPA decoder; and
- its frozen matched random-provider decoder.

The causal and low-rank hyperparameters are constants in the versioned
runner, not validation-selected values. The frozen decoder checkpoints are
byte-verified and neither refit nor checkpoint-selected here.

## Predeclared feasibility gate

All 11 checks are required for one method to be eligible for a future
residual-detector experiment:

| Diagnostic | Held-validation check | Registered-window check |
| --- | --- | --- |
| Median centered RMS ratio | at most `0.90` | at most `0.90` |
| Median dynamic-MAD ratio | at most `1.00` | at most `1.00` |
| Median residual seam ratio | at most `1.25` | at most `1.25` |
| Spectral flatness | no worse than no subtraction | no worse than no subtraction |
| Absolute lag-1 autocorrelation | no worse than no subtraction | no worse than no subtraction |
| Recording consistency | one joint check requires centered RMS ratio at most `1.00` across all three held-validation and four registered-window recording rows | the same cross-panel check; counted once, not duplicated |

Centered RMS and MAD subtract each pixel's temporal median before measuring
scale. Dynamic MAD uses first differences. The seam ratio compares residual
jumps at the frozen eight-pixel decoder lattice with other adjacent-pixel
jumps. Spectral flatness and lagged autocorrelation are residual-whiteness
diagnostics; they are not neural-signal endpoints.

If no predictor passes all 11 checks, the frozen consequence is:

```text
do_not_launch_a_new_residual_detector_experiment_from_these_predictors
```

The identity reference is allowed to fail the `0.90` suppression requirement;
its purpose is to make a no-subtraction baseline explicit, not to manufacture
an eligible predictor.

## Execution and coverage contract

The complete benchmark requires:

- `8` methods;
- `96 + 12 = 108` evaluation clips per method;
- `864` exact per-clip metric rows;
- finite, shape-exact, complete predictions for every nonidentity method;
- unchanged EXP-0028 and EXP-0029 Run-B upstream snapshots;
- an exact artifact index and portable run-start provenance for the numerical
  execution path;
- applicable model-surrogate full-field videos, close-ups, traces, and
  comparison figures with full video-decode validation; and
- `scientific_completion=false`, `scientific_promotion_allowed=false`, and no
  claim or evidence capsule.

Run A is retained locally as a provisional package because its command record
contained a workstation path. It is not copied or registered. Run B uses a
fresh output root and records a portable module command with the runtime data
root redacted, plus a content-aware digest of the dirty run-start checkout.

Run B's explicit `ended_at` timestamp marks the end of numeric, media, and
upstream-integrity work. Report, provenance, status, index, and package
assembly followed; the final artifact index was serialized approximately 38
ms later. The timestamp is not report-completion or atomic-package-promotion
time.

Frozen numerical and media bytes are hash- and decode-verified, but the runtime
record omits SciPy for low-rank randomized SVD, Matplotlib and Pillow for
figures/frames, and the ffmpeg/ffprobe build for media. Exact low-rank numerical
and media-renderer replay environments are therefore incomplete even though
the frozen outputs and conclusions are integrity-checked.

## Result and scientific boundary

Run B completed all engineering and applicable audit checks, but none of the
eight methods passed the full feasibility gate; the identity reference is not
a predictor, so zero of seven subtractors were eligible. See the
[source-off predictor results](../research/SOURCE_OFF_PREDICTOR_FEASIBILITY_V1_RESULTS.md)
for the exact metrics and provenance.

This result does not evaluate source preservation. It does not identify any
prediction as biological background, any residual as neural signal, or any
unmatched native candidate as a negative. Independent-animal generalization,
physical pixel scale, frame cadence, and a claim-bearing detector endpoint
remain unresolved.
