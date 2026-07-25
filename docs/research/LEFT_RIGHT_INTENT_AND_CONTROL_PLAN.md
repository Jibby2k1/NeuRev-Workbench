# Left/Right Intent and Inverse-Control Plan

## Purpose

This document defines the experiments that connect reliable neural measurements
to the actual research objective: using the fish's current state and observed
neural activity to choose a stimulation command that produces a desired
left/right response.

The plan deliberately separates two questions:

1. **Intent decoding:** does pre-movement neural activity predict an upcoming
   left or right movement?
2. **Inverse control:** conditional on the current neural and behavioral state,
   which stimulation action changes the probability of the desired response?

A model can answer the first question without answering the second. Passive
left/right prediction is not evidence that stimulation is causal or
controllable.

## Current checkpoint

The existing latent-head smoke test is a useful negative control, not a
validated intent decoder. Across 11 leave-one-video-out folds, its accuracy was
`0.3636`, balanced accuracy `0.3611`, and macro F1 `0.3545`. Chance for three
balanced classes is approximately `0.3333` and the majority-class accuracy was
`0.3636`. This leaves no robust evidence that the current video-level latent
representation separates left, right, and neutral intent.

The present inverse-dynamics export and behavior-alignment tools establish
useful plumbing, but behavior and stimulation alignment are not populated by
default. Consequently, the repository currently supports passive forward
modeling and export preparation; it does not yet contain the action-conditioned
data needed to identify an inverse controller.

## Intent-decoding estimand

For a reference time `t0` before movement onset, predict:

`y ∈ {left, right, neutral}`

using only information available at or before `t0`. Every sample must include:

- fish, session, video, and trial identifiers;
- a behavior-defined movement onset and direction;
- a neural observation window ending before the leakage guard;
- registered neuron coordinates and coordinate confidence;
- trace/event values and their uncertainty or quality flags;
- behavior state before onset, including pose, heading, and velocity when
  available;
- annotation coverage and alignment quality;
- stimulation history, even when the value is explicitly “none.”

### Causal timing

The first benchmark should use several pre-onset windows, for example:

- `[-500, -250] ms`;
- `[-250, -100] ms`;
- `[-100, -40] ms`;
- a post-onset window used only as a leakage-positive diagnostic.

Exact windows should be derived from the acquisition rate and synchronization
quality. Frames inside the behavior-estimation uncertainty interval must not be
treated as clean pre-onset evidence.

## Spatial-versus-temporal experiment

The central question is whether direction is encoded primarily by *where*
activity occurs, *when* it occurs, or their interaction. Test that question
with a matched ablation ladder using identical samples and splits.

| ID | Representation | Question |
|---|---|---|
| I0 | class priors and pre-movement behavior only | What is the non-neural baseline? |
| I1 | population count/amplitude summaries | Is total activity sufficient? |
| I2 | registered spatial activity map aggregated over the window | Does spatial layout add signal? |
| I3 | coordinate-free temporal traces/events | Does temporal structure add signal? |
| I4 | registered neuron × time representation | Does the spatial-temporal interaction add signal? |
| I5 | I4 plus pre-movement behavior state | Does state resolve otherwise ambiguous neural activity? |

Use a simple regularized model first for each representation, followed by a
small nonlinear model only when the linear benchmark shows stable
above-baseline information. The comparison should not be confounded by
different split logic, training budgets, or label windows.

### Tests required to claim spatial coding

A spatial result is credible only if:

1. I2 or I4 improves held-out balanced accuracy, macro F1, and calibration
   relative to I0 and I1;
2. the improvement persists across held-out fish or, until enough fish exist,
   held-out sessions with uncertainty reported at the session level;
3. coordinate shuffling within a session removes the advantage;
4. deliberate registration perturbations degrade performance smoothly rather
   than unpredictably;
5. a matched model using population amount alone cannot explain the result;
6. the selected spatial regions are stable enough to reproduce across seeds
   and resampling.

### Tests required to claim temporal coding

A temporal result is credible only if:

1. I3 or I4 improves on the aggregated counterparts;
2. circular time shifts or event-time permutation remove the advantage;
3. performance rises as the window approaches movement onset without relying
   on post-onset frames;
4. the result survives controls for pre-existing motion, pose, and imaging
   artifacts.

## Split and evaluation contract

Random frame splits are prohibited because adjacent frames and repeated trials
leak session-specific information.

Preferred split order:

1. held-out fish;
2. if sample size is insufficient, held-out session;
3. held-out video only as an interim diagnostic.

Report:

- balanced accuracy and macro F1;
- per-class precision and recall;
- left-versus-right AUROC and average precision after excluding neutral, when
  scientifically appropriate;
- a confusion matrix;
- reliability/calibration error and Brier score;
- performance versus prediction lead time;
- group-bootstrap confidence intervals;
- result distributions across seeds, not only the best seed.

Model selection must use validation groups. The test groups remain untouched
until the model family, threshold, and lead-time window are fixed.

## Leakage and negative controls

Every intent experiment should run these controls automatically:

- label permutation within fish/session;
- coordinate permutation;
- temporal circular shift larger than the event autocorrelation window;
- post-onset-only diagnostic to quantify how easy overt movement is to detect;
- image-motion and behavior-only baselines;
- removal of high-motion or low-registration-quality trials;
- equalized activity-count analysis so left/right differences cannot be
  attributed only to more detected events;
- detector-version sensitivity analysis on a frozen intent test set.

The detector used to build neural features must be fit without intent-test
labels. Otherwise detector selection can indirectly tune to the downstream
test set.

## Action-conditioned system identification

Inverse control begins only after collecting randomized or otherwise
identifiable stimulation data. Each transition should represent:

`(state_t, action_t, state_{t+1}, outcome, quality)`

### State

- neural activations/traces and registered locations;
- detector confidence and missingness;
- recent neural history;
- pose, heading, velocity, and turn state;
- recent stimulation history;
- session/fish context that is available at deployment.

### Action

- requested command;
- measured delivery, not only the request;
- target/mask or stimulation location;
- amplitude, duration, waveform, and channel;
- command timestamp, hardware timestamp, and estimated latency;
- safety constraints and whether clipping/interlocks altered the command;
- an explicit no-stimulation action.

### Outcome

- left/right/neutral response probability;
- turn magnitude and latency;
- neural response;
- adverse or out-of-envelope state;
- alignment and observation-quality flags.

The initial dataset should use safe randomized stimulation coverage inside an
approved envelope. A policy-generated dataset without sufficient exploration
cannot distinguish action efficacy from the states in which the policy chose
the action.

## Model ladder for system identification

1. **Descriptive action-response table:** stratify outcomes by action and coarse
   pre-action state.
2. **Regularized action-conditioned classifier/regressor:** estimate response
   and uncertainty with explicit state-action interactions.
3. **Action-conditioned transition model:** predict neural and behavioral next
   state, including a no-action branch.
4. **Ensemble or probabilistic model:** expose epistemic uncertainty and reject
   out-of-distribution states.
5. **Optional representation model:** only after the tabular/simple transition
   models establish identifiable action effects.

The passive grid-dynamics models can provide representation and forecasting
ideas, but they cannot be “inverted” into a causal stimulation policy because
action is absent from their transitions.

## Controller checkpoint

The first controller should be constrained model-predictive control (MPC) in
simulation:

- optimize probability of desired direction and response latency;
- penalize stimulation energy, rapid action changes, and unsafe states;
- constrain every action to the approved envelope;
- abstain or fall back to no stimulation when uncertainty or distribution
  shift is high;
- compare with no stimulation, a fixed safe policy, and a randomized safe
  policy.

Only after simulator validation should the controller run in shadow mode on
live observations. Real stimulation should begin conservatively with logging,
interlocks, watchdogs, and a rollback/fallback policy.

## Stage gates

| Gate | Pass criterion | Failure response |
|---|---|---|
| Intent data | causal pre-onset labels, usable alignment, group IDs, adequate class counts | repair acquisition/synchronization before modeling |
| Intent signal | reproducible held-out gain over behavior/population baselines with leakage controls passed | revise representation or conclude signal is not measurable at this resolution |
| Action identifiability | safe action coverage, no-action controls, measured delivery, state overlap | collect designed stimulation data |
| Transition model | calibrated held-out action effects and useful counterfactual ranking in supported regions | simplify model or expand coverage |
| Simulator control | constrained MPC beats baselines across uncertainty and perturbation tests | improve transition model and safety envelope |
| Shadow deployment | latency, drift, abstention, and interlocks meet the operational contract | do not stimulate |
| Conservative closed loop | pre-registered safety and efficacy criteria pass | fall back and review |

## Immediate next work

1. Freeze an intent dataset schema and export pre-onset windows from aligned
   behavior and neural data.
2. Produce a data-readiness report: fish/session counts, class balance,
   alignment uncertainty, missing stimulation fields, and detector coverage.
3. Run I0–I4 with group splits and all leakage controls.
4. Use the result to decide whether spatial mapping, temporal resolution, or
   acquisition quality is the limiting factor.
5. In parallel, finalize the stimulation-action schema so new acquisitions are
   usable for system identification.

Related documents:

- [Program roadmap](FISH_INVERSE_CONTROL_ROADMAP.md)
- [Activation robustness plan](NEURAL_ACTIVATION_DETECTION_ROBUSTNESS.md)
- [Tooling roadmap](../developer/FISH_CONTROL_TOOLING_ROADMAP.md)
- [Inverse-dynamics export](../INVERSE_DYNAMICS_EXPORT.md)

