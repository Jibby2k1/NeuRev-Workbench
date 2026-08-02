# Spon Ca Burst stochastic-state architecture grid

## Purpose

This workflow tests whether the visually useful background/dynamics separation
from the Stage-1 stochastic-Parzen study also improves known-neuron detection.
It treats separation quality and detection quality as different endpoints.

The entry point is:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-architecture-grid preflight \
  --config examples/spon_ca_burst_stochastic_architecture_grid.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment parzen-architecture-grid run \
  --config examples/spon_ca_burst_stochastic_architecture_grid.example.json
```

The run requires a matching preflight and refuses completed or partial output
collisions.

## Architectures

Let \(I_t\) be the current frame, \(B_q\) the per-pixel quiet median, and
\(\widehat B_t^P=aI_{t-1}+bI_t+c\) the accepted affine reconstruction induced
by stochastic Parzen ICA.

The reference/Parzen innovation family is

\[
R_t=(1-\rho)R_{t-1}+\rho I_t,
\]

\[
\Delta_t=\widehat B_t^P-R_t-\operatorname{median}_{q}
(\widehat B_q^P-R_q),
\]

\[
B_t=R_t+\epsilon\,
\operatorname{clip}(\Delta_t,-k\,\mathrm{MAD}_q,k\,\mathrm{MAD}_q),
\qquad D_t=I_t-B_t.
\]

The grid spans five reference half-lives, six correction fractions, and five
clip levels. The \(\epsilon=0\) reference-only control is represented once per
half-life because its clip value has no effect: 130 effective innovation
operators.

The stable fixed-point family is

\[
B_t=B_q+m(B_{t-1}-B_q)+(1-m)s(I_t-B_q),
\qquad D_t=I_t-B_t,
\]

where \(m\) is determined by a memory half-life and \(s\in[0,1]\) is the
steady-state observation fraction. The \(s=0\) operator is a static quiet
reference for every memory half-life and is therefore canonicalized once:
55 effective fixed-point operators.

The maintained grid therefore screens 185 effective operators. The first
completed v1 audit contains 193 parameter rows because it retained nine
nominal half-life labels for the same \(s=0\) fixed operator; it contains 185
effective operators. Completed output was preserved rather than rewritten.

## Broad screen

Every operator is evaluated on:

- four-pixel disks around all 79 labeled observations;
- a bounded stable-bright artifact proxy;
- a stable anatomical-structure proxy;
- a stable lower-intensity background proxy;
- a high-difference active-unlabeled proxy; and
- a uniform calibration sample.

Proxy strata are deterministic algorithmic masks, not manually verified
truth. The screen measures peak, positive-area, late-window, and waveform
retention plus quiet, artifact, anatomy, background, and active-unlabeled
dynamics ratios.

For each held-out burst, finalists are selected with the other three bursts
and fixed proxy metrics. The held-out burst is not used for promotion. The
union of promoted lanes receives full-field evaluation.

## Frozen detection contract

Full-field finalists and controls use:

- per-pixel quiet-median centering;
- a quiet-derived global scale;
- positive residual evidence;
- temperature-0.25 log-mean-exp temporal pooling;
- four duration-matched quiet maps;
- six-pixel NMS and primary matching;
- one quiet peak per map as the primary threshold;
- FROC targets 0.25, 0.5, 1, 2, and 5;
- 58 candidates per burst as the fixed-budget comparison; and
- paired bootstrap intervals over 27 ROI identities.

Sparse labels support known-positive recall, FROC, candidate burden, and a
lower bound on known-label candidate yield. Unmatched event candidates remain
unknown, not false positives. TN, FP, and precision are not identified.

## Completed v1 result

Output:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_stochastic_architecture_grid_v1
```

The completed run used 1.79 GiB peak RAM and 73.33 seconds. Raw Direct
reproduced the frozen mean recall exactly at `0.605615942` (49/79 pooled;
232 candidates). The leave-one-burst-out screen selected a static quiet
reference in every fold. It reached only `0.364648` mean recall and 30/79
pooled matches, with an ROI-identity bootstrap difference of `-0.240506`
versus Raw Direct (95% percentile interval `[-0.357143, -0.138889]`;
probability of improvement `0.0`).

The current 10-second, 0.1-fraction, 4-MAD innovation lane reached `0.329969`
mean recall (27/79 pooled; 70 candidates). Its fixed-58-candidate recall was
`0.640580`, close to Raw Direct's `0.657246`. This is evidence that the
separation is visually selective and candidate-efficient, but not a
replacement detector at the frozen quiet threshold.

The primary implication is that ROI amplitude/retention screening is not a
sufficient surrogate for full-field spatial detection. The next useful audit
is manual review of the proxy masks and unmatched candidates, followed by
using the separated dynamics as a feature alongside Raw Direct rather than
classifying it alone.

## Important artifacts

- `REPORT.md`: concise scientific result.
- `metrics.json`: complete full-field and cross-fitted metrics.
- `screen_metrics.tsv`: all broad-screen rows.
- `roi_observation_metrics.tsv`: observation-level screen evidence.
- `selection.json`: fold-specific promotion record.
- `candidates.tsv`: known matches and truth-unknown candidates.
- `proxy_strata.json`: exact algorithmic proxy definitions and counts.
- `visuals/*/background.tif`: selected background estimates.
- `visuals/*/dynamics_noise.tif`: selected signed residuals.
- `visuals/*/detection_burst_maps.tif`: four temporally pooled burst maps.
