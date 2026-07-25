# Fish Neural Intent and Inverse-Control Roadmap

## Technical summary

The program should be organized around the final causal question—how an action
changes neural intent and behavior—not around repeated optimization of a neuron
detector. Neuron activation detection is a measurement layer. It is necessary,
but it is not itself evidence of left/right intent or controllability.

Current evidence supports four conclusions:

1. Raw positive-residual scoring is substantially stronger than the learned
   guarded-contrast models on the sparse burst labels. The direct lane reached
   `0.6056` mean leave-one-burst-out recall. Learned contrast reached `0.1328`
   in v1 and `0.2051` in the best v2 cell; conservative direct-initialized
   learning in v3 preserved `0.6056` but did not improve any held-out burst.
2. Precision is not yet scientifically identified on the burst dataset because
   event labels are sparse and non-exhaustive. An unmatched peak is unknown,
   not a false positive.
3. Existing left/right/neutral latent evidence is weak. The video-level ridge
   head reached `0.3636` accuracy, equal to the majority baseline, across only
   11 videos. This does not establish intent discriminability.
4. Existing dynamics models are observational. They can support state
   representation and simulation work, but inverse control requires
   synchronized actions and behavior plus an action-conditioned forward model.

The recommended sequence is:

```text
validated imaging/behavior/action synchronization
  -> calibrated neural observations with uncertainty
  -> held-out-fish left/right/rest intent decoder
  -> randomized action-conditioned system identification
  -> constrained simulation and model-predictive control
  -> gated real closed-loop testing
```

## The problem is a chain of estimands

Each checkpoint asks a different question and needs its own labels, metrics,
and split:

| Checkpoint | Scientific question | Primary output |
| --- | --- | --- |
| Neural measurement | Which persistent neurons are active, when, and with what uncertainty? | Calibrated ROI/event observations |
| Intent discrimination | Does pre-movement neural state distinguish left, right, and rest on unseen fish/sessions? | Calibrated intent probabilities |
| System identification | How does a commanded intervention alter subsequent neural and behavioral state? | Action-conditioned transition model |
| Inverse control | Which safe action moves the current state toward a target intent/behavior state? | Constrained action sequence |

Success at an earlier checkpoint is necessary but not sufficient for the next.
In particular:

- detection recall does not imply useful intent information;
- intent association does not imply causal control;
- passive prediction does not identify an intervention response;
- a controller that succeeds in its training simulator is not yet safe or
  effective in a fish.

## Current evidence and its limits

### Activation detection preserves a strong direct lane but lacks precision truth

The v1-v3 experiments show that weight initialization was not the main reason
learnable CFAR underperformed. Stabilized optimization improved learned
contrast, and v3 parameters moved while loss decreased, but the labeled
detections did not improve. The likely limitations are:

- the contrast statistic suppresses information that the direct-amplitude lane
  uses;
- sparse point labels do not describe complete neuron footprints or all active
  cells;
- quiet negatives sampled at labeled coordinates do not match the full-field
  quiet peaks that set the evaluation threshold;
- NMS and thresholding are discontinuous post-processing decisions that are not
  aligned with the training loss;
- neuron localization, identity tracking, trace extraction, and activation
  inference are being treated as one detector problem.

The appropriate response is not a wider blind CFAR sweep. It is to build an
exhaustive evaluation subset, separate the measurement tasks, and optimize a
calibrated proposal-to-event pipeline.

### Left/right evidence currently measures video identity more than intent

The current left/right/neutral labels are primarily video-level. That permits a
weak separability smoke test, but it cannot determine whether a neural pattern
precedes motion, reflects ongoing motion, or comes from motion-correlated
imaging artifact. A scientifically useful intent label must be tied to
behavioral onset and must distinguish:

- pre-movement intent window;
- movement execution window;
- post-movement/recovery window;
- rest and ambiguous transitions.

The spatial hypothesis is plausible and testable, but it should be tested
against temporal and nuisance controls rather than assumed.

### Inverse control is blocked by the action variable, not by controller choice

The current forward predictors estimate future neural grid state from past
neural grid state. They do not contain an action variable. Therefore they
cannot estimate `p(next state | current state, intervention)` and should not be
inverted as though they were causal.

The control state should eventually combine:

```text
s_t = {
  neural observations and uncertainty,
  recent neural history,
  current pose/tail state,
  recent behavior history,
  anatomical registration confidence,
  recent actions and estimated latency
}
```

An action should record the commanded and measured intervention separately,
including spatial target, amplitude, duration, onset, latency, actuator limits,
and safety status.

## Stage gates

### Gate 0: Measurement and synchronization are auditable

Required:

- immutable fish/session/video identifiers;
- template-registration transform and confidence;
- imaging and behavior timestamps with explicit units;
- sync points and a passed alignment report;
- explicit annotation coverage: exhaustive, candidate-only, or sparse;
- action/stimulus log schema, even for no-action datasets.

Do not train an intent decoder on behavior labels until alignment is validated.

### Gate 1: Activation detection is calibrated for both recall and precision

Required evidence:

- exhaustively reviewed spatial-temporal tiles across fish, sessions,
  direction labels, intensity ranges, and artifact conditions;
- object-level neuron identity metrics and event-level activation metrics;
- precision-recall curves and average precision, not one threshold alone;
- FROC/false-candidate burden for sparse-label datasets;
- bootstrap intervals grouped by fish or ROI identity;
- calibration and abstention behavior;
- runtime p50/p95/p99 for any closed-loop candidate.

Advance when the pipeline improves over raw-direct and existing CFAR baselines
on held-out fish/sessions without trading an unacceptable amount of precision
for recall. The acceptable operating point should also be tested by how much
detector error degrades the downstream intent decoder relative to manual
neural observations.

### Gate 2: Left/right/rest intent is reproducible

Evaluate causal pre-onset windows and compare:

1. majority and class-prior baselines;
2. global population activity only;
3. spatial activity only;
4. temporal activity only;
5. full spatiotemporal activity;
6. coordinate-shuffled and time-shifted controls;
7. raw/grid features versus detected-ROI features.

Advance only if performance beats the majority and population-activity
baselines with group-bootstrap uncertainty, remains useful on held-out
fish/sessions, and survives movement-artifact controls. Report balanced
accuracy, macro F1, per-class recall/precision, calibration, abstention
coverage, and lead time before behavioral onset.

Spatial evidence is supported when spatial or registered-ROI features add
held-out value beyond population activity and temporal-only models, and that
value disappears under coordinate shuffling or anatomical misregistration.

### Gate 3: Intervention response is causally identifiable

Collect randomized or counterbalanced actions with no-action controls. Required
splits hold out fish/session and action patterns. Fit:

```text
s_(t+1:t+H) = f(s_(t-W+1:t), u_(t:t+H-1))
```

Advance only if the model beats:

- no-action persistence;
- an action-agnostic forward model;
- action-frequency or session-identity shortcuts;
- simple linear impulse-response baselines.

The action effect must be observable above uncertainty and not explained only
by stimulation artifact.

### Gate 4: A constrained controller succeeds in simulation

Use model-predictive control first. It supports hard action limits, latency,
uncertainty penalties, and receding-horizon correction. Compare against
no-control, random safe actions, and fixed heuristic policies.

Advance when the controller improves target-reaching success on held-out
sessions while respecting every hard safety constraint and remaining robust
across a model ensemble or uncertainty perturbations.

### Gate 5: Real closed-loop testing is deliberately limited

Begin with conservative action sets, explicit operator abort, dose budgets,
uncertainty gating, and shadow-mode recommendations before automatic action.
Real-fish evidence must be reported separately from simulator evidence.

## Program priorities

### Immediate: make neural measurement evaluable

1. Define exhaustive annotation tiles and reviewer coverage metadata.
2. Add candidate dispositions so precision is measurable.
3. Build full-field hard-negative mining and calibrated PR/FROC evaluation.
4. Separate persistent neuron identity from per-neuron activation inference.
5. Benchmark direct residual, fixed CFAR, learned components, and external
   proposal sources under one evaluator.

### Next: construct an intent dataset

1. Ingest tail/pose time series and behavioral onset labels.
2. Validate imaging-behavior synchronization.
3. Export causal pre-onset neural-state windows.
4. Run spatial, temporal, and spatiotemporal ablations.
5. Use group-held-out and leakage-control evaluations.

### Then: collect action-conditioned data

1. Freeze the action schema and safety limits before data collection.
2. Randomize safe interventions and retain no-action trials.
3. Measure commanded versus delivered action and latency.
4. Fit action-conditioned state transitions.
5. Build a simulation-first MPC harness.

## What should not happen next

- Do not treat unmatched candidates from sparse labels as false positives.
- Do not optimize only recall and infer precision from quiet peak counts.
- Do not use future movement frames to claim pre-movement intent.
- Do not interpret video-label classification as fish-general intent decoding.
- Do not invert a passive dynamics model without an action variable.
- Do not launch a controller on a real fish before action-conditioned
  validation and hard safety tests.
- Do not make one representation serve every checkpoint. Raw/grid and
  ROI/event representations should be compared at the intent gate.

## Supporting specifications

- [Activation Detection Robustness Plan](NEURAL_ACTIVATION_DETECTION_ROBUSTNESS.md)
- [Left/Right Intent and Inverse-Control Plan](LEFT_RIGHT_INTENT_AND_CONTROL_PLAN.md)
- [Fish-Control Tooling Roadmap](../developer/FISH_CONTROL_TOOLING_ROADMAP.md)
- [Inverse-Control Discussion Brief](../INVERSE_CONTROL_DISCUSSION_BRIEF.md)
- [Inverse-Dynamics Export](../INVERSE_DYNAMICS_EXPORT.md)
