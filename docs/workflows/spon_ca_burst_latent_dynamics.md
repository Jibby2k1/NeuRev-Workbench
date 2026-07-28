# Spon Ca Burst Latent-Dynamics Workflow

Last updated: 2026-07-27.

## Scope and current authorization

This workflow implements the stable AR(1) reference from
`docs/developer/LATENT_DYNAMICS_DENOISING_IMPLEMENTATION_BRIEF.md`. It was first
validated on deterministic synthetic fixtures and tiny generated arrays. The
explicitly selected full Spon Ca Burst CPU run is now complete at
`Outputs/LatentDynamics/spon_ca_burst_latent_dynamics_v1`. No GPU was used.

The goal is not the smoothest movie. The goal is a stable, uncertainty-aware
latent trajectory that preserves event dynamics and can be evaluated as a
feature source under the frozen Raw Direct detector contract.

## Implemented model and signal names

The signed, quiet-normalized observation residual is modeled as

```text
state[t] = gamma * state[t-1] + dynamic_drive[t]
residual[t] = state[t] + observation_noise[t]
0 <= gamma <= 1 - stability_epsilon
```

The implementation deliberately keeps these quantities distinct:

- `latent_filter_mean`: causal Kalman posterior state;
- `latent_smoother_mean`: full-sequence, noncausal RTS posterior state;
- `state_difference_lag_k`: state at `t` minus state at `t-k`;
- `dynamic_drive`: state at `t` minus `gamma * state[t-1]`;
- `filter_innovation`: observation minus one-step state prediction;
- `smoother_residual`: observation minus the offline smoothed state.

Ordinary state differencing equals dynamic drive only in the explicit
`gamma=1` test fixture. The fitted model itself remains strictly stable and
never permits `gamma=1`.

## Interfaces and safety behavior

Reusable numerical code lives in `neurobench/algorithms/latent_dynamics.py` and
accepts `[T]` or `[T,N]` arrays. It uses float64 scalar recursions, Joseph-form
filter covariance, a shared scalar variance rather than pixel-by-pixel
covariance, finite-value checks, and float32 storage by default.

The experiment package is `neurobench/experiments/latent_dynamics/`. Its strict
manifest rejects unknown fields and resolves paths relative to the manifest.
Preflight:

- requires a new explicit artifact directory;
- refuses an existing output root;
- validates TYX shape, inclusive UI frames, label coordinates, output size,
  RAM, disk, and deterministic sample size;
- writes a label projection overlay;
- records that labels are unavailable to fitting; and
- does not authorize a real run merely because preflight passes.

The runner requires the identical reviewed resolved config, applies the model
in bounded pixel chunks, writes dense states through `.partial` memory maps,
validates them before atomic rename, removes partials on failure, and records
progress and resource summaries. Unlabeled candidates are `unknown`, never
negative.

## Commands

Run the synthetic falsification suite in a new output directory:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment latent-dynamics synthetic \
  --output-dir /tmp/neurev-latent-synthetic
```

Prepare the real-data preflight only after explicitly selecting this workflow:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment latent-dynamics preflight \
  --config examples/spon_ca_burst_latent_dynamics.example.json \
  --artifact-dir Outputs/LatentDynamics/preflight_spon_ca_burst_latent_dynamics_v1
```

After reviewing that directory, the separately selected run command is:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment latent-dynamics run \
  --config examples/spon_ca_burst_latent_dynamics.example.json \
  --preflight-dir Outputs/LatentDynamics/preflight_spon_ca_burst_latent_dynamics_v1
```

Read the already-produced benchmark result without new processing:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment latent-dynamics feature-benchmark \
  --run-dir Outputs/LatentDynamics/spon_ca_burst_latent_dynamics_v1
```

The example manifest is a specification. The completed output root is
collision-protected and must not be overwritten; use a new experiment ID for
any confirmation or perturbation run.

## Full Spon Ca Burst result

The completed run evaluated 13 single-feature lanes under one frozen detector
contract: temperature-`0.25` log-mean-exp temporal pooling, fifth-highest quiet
NMS peak calibration, six-pixel NMS and matching, a 500-candidate cap per
burst, and four labeled burst folds. Raw Direct reproduced the historical
baseline exactly.

| Lane | Macro recall | Known matches | Candidates | Wins vs Raw Direct |
| --- | ---: | ---: | ---: | ---: |
| Raw Direct | `0.6056` | `49/79` | 232 | reference |
| Causal filter amplitude | `0.6540` | `53/79` | 312 | 2/4 |
| Offline smoother amplitude | **`0.6867`** | **`55/79`** | 320 | **4/4** |
| Legacy asymmetric EMA | `0.3182` | `25/79` | 36 | 0/4 |
| Standardized filter innovation | `0.2161` | `18/79` | 23 | 0/4 |

Smoother-amplitude burst recalls were `0.6000`, `0.6500`, `0.7143`, and
`0.7826`, compared with Raw Direct `0.4667`, `0.5500`, `0.6667`, and `0.7391`.
It therefore improved macro recall by `+0.0811` and won all four bursts, passing
the preregistered C4 feature-value rule. Its candidate burden increased by 88
(`+37.9%`), so this is not evidence of improved precision. Unmatched candidates
remain unknown and require review.

The causal filter improved macro recall by `+0.0484`, but tied two bursts and
won only two; it does not pass C4. The RTS smoother is noncausal and cannot be a
real-time lane. Raw and latent differences, positive dynamic drive, and filter
innovation all underperformed substantially. The useful result is denoised
latent amplitude, not post-denoising differencing.

The selected shared model used `gamma=0.9844964`, decay time `1280 ms`, and
process-to-observation variance ratio `0.03`. The decay time landed at the upper
grid boundary, so temporal-block and bounded-parameter confirmation remains
important before treating this as a settled model choice.

## Current evidence and gates

- C0 implementation integrity: passed on the focused suite and tiny smoke run.
  The Raw Direct temporal-pooling anchor reproduced with zero absolute error,
  frame/coordinate contracts are explicit, and the artifact tree is
  collision-safe.
- C1 numerical stability: passed for the implemented scalar/vector algorithms,
  the 55-case synthetic CLI run (11 cases across five seeds), and tiny smoke
  runs. All outputs were finite and stable. The synthetic median NMSE was
  `0.3234` and median absolute peak error was three frames. The real preflight
  sample has not yet been selected or reviewed.
- C2 denoising validity: partially supported, not fully passed. Synthetic tests show the
  smoother improves a noisy transient reconstruction and preserves a ramp. In
  the fitted-filter suite, transient NMSE was `0.1565` and peak error ranged
  from two to five frames. Motion-edge reconstruction remained poor (median
  NMSE `0.8429`); outlier, drift, heteroscedastic, and model-mismatch fixtures
  remain falsification evidence rather than claims of robustness. The required
  real event-preservation and perturbation diagnostics remain incomplete.
- C3 real signal preservation: incomplete. Known-label recall improved, but
  amplitude, temporal-area, onset, temporal-block, and parameter-perturbation
  bounds have not all been passed.
- C4 feature usefulness: offline smoother amplitude passes conditionally with
  `+0.0811` macro recall and 4/4 burst wins. Causal filter amplitude does not
  pass. C4 does not waive the incomplete upstream C2/C3 checks.
- C5 real-time consideration: not passed. The winning smoother is noncausal;
  the causal filter lacks the required fold wins and latency measurements.

Completion of a run is not scientific success. Review denoising validity before
interpreting activation-detection metrics, and review single features before
authorizing any fusion initialized at Raw Direct.
