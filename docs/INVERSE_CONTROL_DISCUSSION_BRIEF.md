# Inverse-Control Discussion Brief

This brief is designed to be pasted into ChatGPT or used as meeting context when
designing an inverse-control solution from the Neurobench methods.

## Goal

Design a control system that can choose an intervention or stimulus sequence to
drive a zebrafish neural/behavioral state toward a desired target, using the
current Neurobench grid-dynamics and inverse-dynamics export methods.

The immediate objective should be a simulation-first controller. The current
models are observational forward predictors, not causal action-conditioned
control models.

## Current Evidence

We have:

- A template-aligned grid state representation from calcium/imaging videos.
- Video-level train/validation/test splits.
- Forward dynamics models that predict future grid states.
- Persistence baselines for every dynamics dataset.
- Dashboard videos showing target, horizon-shifted model prediction,
  horizon-shifted persistence, and lag-compensated absolute error.
- Export tooling for control-ready ROI traces and reviewed events.

Best current forward-model evidence:

- Grid32 large sweep:
  - 1,212 model rows.
  - 850/1,212 improved on held-out test videos.
  - Best test improvement over persistence: about 2.29e-4 MSE.
  - Best test models are temporal CNN pixel/grid predictors on `w8_s1_h50`.
- Grid128 scalable temporal-CNN sweep:
  - 32 model rows.
  - 31/32 improved on held-out test videos.
  - Best test improvement over persistence: about 3.22e-4 MSE.
  - Shorter 25-frame horizon appears promising.
- Prior overnight validation:
  - 8,227 command runs passed with 0 failures before the grid-dynamics expansion.
- Current local validation:
  - 389 passed, 4 failed, 2 skipped.
  - Open failures are API-reference/CLI-output/CLI-crash issues, not direct
    model math failures.

## State Representation

Use one of these state choices:

1. Grid state `x_t`
   A template-aligned image/grid tensor. This is most interpretable and matches
   the strongest current temporal-CNN results.

2. Latent state `z_t = E(x_t)`
   A compressed autoencoder representation. This may be easier for planning but
   needs careful reconstruction and behavior validation.

3. Behavior-proxy state `b_t = D(x_t)` or `D(z_t)`
   A decoder output such as left/right/rest probability. This is useful if the
   real target is behavior rather than neural-image similarity.

Recommended starting point:

Use grid128 or grid32 `x_t` for the forward model, then add a behavior decoder
only after behavior alignment is validated.

## Forward Model

Current model form:

```text
x_{t+H} = f_theta(x_{t-W+1:t})
```

Where:

- `W = 8` input frames in the current best workflows.
- `H = 25` or `50` future frames.
- `f_theta` is usually a temporal CNN over grid-state frames.
- Baseline is persistence: `x_{t+H} = x_t`.

Needed for inverse control:

```text
x_{t+H} = f_theta(x_{t-W+1:t}, u_{t:t+H-1})
```

The control/action sequence `u` is not yet represented in the current passive
models. It must be added from real interventions, simulated interventions, or a
carefully constrained proxy.

## Action Space

Define the action variable before designing a controller. Examples:

| Action type | Possible representation | Notes |
|---|---|---|
| Optogenetic stimulation | Spatial mask, amplitude, pulse duration, onset frame | Most direct neural control; needs safety constraints and latency model. |
| Visual stimulus | Direction, contrast, speed, timing | More behaviorally natural; action affects sensory pathway, not direct neural state. |
| Closed-loop perturbation | State-dependent pulse rule | Useful after a simulator exists. |
| No physical action yet | Latent intervention vector | Useful for planning experiments, not real control. |

Minimum action metadata:

- `action_id`
- `start_frame`
- `end_frame`
- `latency_frames`
- `spatial_mask` or target region
- amplitude/intensity
- actuator limits
- safety constraints
- whether the action was commanded, measured, or inferred

## Candidate Control Formulations

### 1. Model Predictive Control

At each time step:

1. Estimate current state from recent frames.
2. Optimize a candidate action sequence under the forward model.
3. Apply only the first action or short action prefix.
4. Observe new frames.
5. Repeat.

Objective:

```text
minimize_u  ||phi(x_{t+H}) - y_target||^2
          + lambda_action ||u||^2
          + lambda_smooth ||u_t - u_{t-1}||^2
          + lambda_risk R(x, u)
```

Where:

- `phi` maps neural grid state to the controlled output.
- `y_target` can be a target grid state, latent state, or behavior probability.
- `R` penalizes saturation, artifacts, unsafe stimulation, or out-of-distribution
  states.

Why this is the best first choice:

- It works with imperfect forward models.
- It can include hard constraints.
- It naturally handles latency and receding-horizon uncertainty.

### 2. Inverse Model

Learn:

```text
u_t = g_theta(x_{t-W+1:t}, y_target)
```

This is attractive but riskier. It requires intervention data and can produce
unsafe or non-identifiable actions if several actions produce similar observed
states.

Use only after an MPC-style simulator gives a reliable baseline.

### 3. Latent Planning

Plan in latent space:

```text
z_t = E(x_t)
z_{t+H} = f_z(z_{t-W+1:t}, u)
```

Then decode or evaluate:

```text
x_target_hat = D(z_target)
```

This can be computationally cheaper, but the latent must preserve behaviorally
important and controllable dimensions. Current latent models are useful but the
strongest evidence is still direct grid temporal CNN prediction.

### 4. Behavior-Targeted Control

Train a behavior decoder:

```text
p(left/right/rest | x_t)
```

Then optimize actions to shift behavior probability:

```text
maximize_u p(right | x_{t+H})
```

This is probably the most scientifically meaningful control target, but it
depends on frame-resolved behavior labels and synchronization.

## Recommended Architecture

Start with this stack:

```text
video frames
  -> crop/register/template alignment
  -> grid state x_t
  -> temporal CNN forward model f_theta
  -> optional behavior decoder phi
  -> MPC optimizer
  -> constrained action sequence u
```

First forward model candidate:

- Grid128 scalable temporal CNN.
- 8-frame input window.
- 25-frame horizon.
- Motion-weighted Huber and residual MSE variants.

Fallback forward model:

- Grid32 temporal CNN.
- 8-frame input window.
- 50-frame horizon.
- Residual MSE, 32-64 hidden channels, 4-6 layers.

## Experimental Design Needed Next

### Data Collection

Collect or construct datasets with:

- Frame-synchronized neural video.
- Frame-synchronized behavior.
- Explicit action/stimulus command log.
- Action latency estimates.
- Fish/session/day identifiers.
- Calibration for spatial alignment.
- No-action control periods.
- Randomized or counterbalanced stimulation policies.

### Splits

Use held-out splits that match the intended claim:

| Claim | Required split |
|---|---|
| Predict within same fish/session | held-out video segments are acceptable. |
| Predict new videos from same day | held-out videos by label. |
| Generalize to new fish | fish-held-out split. |
| Generalize across sessions | day/session-held-out split. |
| Control under interventions | intervention-policy-held-out or action-pattern-held-out split. |

### Metrics

Forward prediction:

- MSE/MAE against target grid state.
- Improvement over persistence.
- Motion-weighted error.
- Behavior-decoder consistency.
- Calibration/uncertainty.

Control:

- Target-reaching error.
- Time-to-target.
- Success rate.
- Energy/action cost.
- Constraint violations.
- Recovery from perturbations.
- No-control and persistence-control baselines.

Safety:

- Saturation fraction.
- Artifact score.
- Out-of-distribution score.
- Maximum stimulation dose.
- Spatial off-target penalty.

## Validation Checklist Before Real Closed-Loop Use

- [ ] Current test suite is green.
- [ ] Post-grid overnight soak passes.
- [ ] Absolute-error dashboard alignment is regression-tested.
- [ ] Behavior traces are synchronized and validated.
- [ ] Action schema exists and validates latency/limits.
- [ ] A forward model is trained with action-conditioned data.
- [ ] A no-action baseline is retained.
- [ ] Controller is evaluated in simulation.
- [ ] Controller beats no-control/persistence on held-out sessions.
- [ ] Uncertainty gating prevents unsafe actions.
- [ ] Safety constraints are hard-coded and tested.

## Questions To Ask ChatGPT

Use these prompts to structure the discussion.

### Prompt 1: Controller Design

```text
We have template-aligned zebrafish neural grid states x_t from calcium/imaging
videos. Current forward models predict x_{t+H} from the last 8 grid frames and
beat persistence on held-out videos. We want an inverse-control system, but we do
not yet have action-conditioned intervention data.

Design a simulation-first inverse-control architecture. Compare MPC, inverse
models, and latent planning. Specify objective functions, constraints, required
data fields, and failure modes.
```

### Prompt 2: Action-Conditioned Dataset Schema

```text
Design a dataset schema for action-conditioned zebrafish neural control. It must
store frame-synchronized neural grid states, behavior labels, action commands,
latency, actuator constraints, safety metadata, and train/validation/test splits.
The schema should support optogenetic and visual-stimulus actions.
```

### Prompt 3: Causality And Evaluation

```text
Given passive forward predictors that beat persistence, what experimental
protocol is required to claim causal controllability? Propose train/test splits,
baselines, ablations, and statistical tests for a closed-loop zebrafish neural
control experiment.
```

### Prompt 4: Model Selection

```text
We have grid32 and grid128 temporal CNN predictors. Grid32 large sweep has
850/1212 positive test improvements with best improvement about 2.29e-4 MSE.
Grid128 has 31/32 positive test improvements with best improvement about
3.22e-4 MSE but fewer configurations. Which model should be used first for MPC,
and what additional experiments would de-risk the choice?
```

### Prompt 5: Safety

```text
Design hard safety constraints and software tests for a neural inverse-control
system that can choose optogenetic or visual-stimulus actions. Include action
dose limits, spatial masks, uncertainty gating, saturation/artifact detection,
and emergency fallback behavior.
```

## Key Caveat

Do not let ChatGPT assume the current models are causal controllers. They are
forward predictors over observed grid dynamics. The inverse-control system must
add actions, causal validation, closed-loop simulation, and safety constraints
before any real intervention claim is justified.
