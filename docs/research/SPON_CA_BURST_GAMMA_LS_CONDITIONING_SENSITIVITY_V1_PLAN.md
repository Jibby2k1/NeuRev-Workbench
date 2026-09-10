# Spon Ca Burst Gamma-LS conditioning sensitivity v1

## Question

Do the historically fixed spatial Gaussian, causal EMA, and training-quiet
scale-floor settings materially improve the three nonlearned Gamma-LS lanes,
and is any improvement worth its online CUDA latency?

## Frozen nested design

The experiment crosses Gaussian sigma `{0, 0.5, 1.0, 1.5}` pixels, causal EMA
alpha `{1, 2/3, 0.4, 0.25}`, and training-quiet scale-floor percentile
`{5, 10, 20}` for each of `raw`, signed adjacent difference, and
energy-normalized adjacent difference. This is 48 settings per representation,
144 representation/settings, 576 outer-fold cells, and 1,152 reversed
quiet-role-swap rows.

There are four leave-one-burst-out outer folds. Each fold uses only its already
frozen radial `support_candidate_context` from the completed support screen.
Within a fold and representation, settings are compared using the mean of the
two event-minus-quiet q0.999 positive-tail contrasts, the minimum of those two
swap contrasts, and online CUDA runtime per frame. A setting is Pareto
nondominated when no other setting is at least as good in both contrasts and
runtime and strictly better in one. The deterministic selected point is the
nondominated setting with highest mean contrast, then highest minimum-swap
contrast, lowest runtime, and lexicographically smallest setting ID.

The selected setting and the historical anchor `(sigma=1, alpha=0.4,
percentile=10)` are both frozen per outer fold and representation. They alone
advance to a later protected candidate-budget (`B`) and NMS evaluation. No
protected coordinate or neuron-identity table may be opened by this screen.

## Raw-lane terminology

`raw` names a representation family, not necessarily unconditioned camera
values. Only `raw` at `sigma=0, alpha=1` is **acquisition raw float32**. The
historical `raw` arm at `sigma=1, alpha=0.4` is explicitly reported as the
**historically conditioned raw lane**. Other members are reported as
conditioned level lanes. This distinction is part of every grid and metric
row.

## Leakage and provenance gates

The executor reads the movie, base manifest, base preflight metadata, and the
indexed support-screen artifact. It rejects coordinate/identity-bearing fields
in the fold-context contract (apart from false-valued leakage attestations),
requires exactly four fold-local primary radial contexts, and verifies every
indexed file and implementation fingerprint before output mutation. It writes
a sensitivity-specific preflight so code, base-preflight, movie, support
artifact, and strict manifest hashes can be rechecked immediately before the
GPU run. Existing destinations are never overwritten.

The event windows make this a burst-window-supervised hyperparameter screen,
not a fully label-free analysis. Its results cannot establish protected recall,
proposal yield, neuron identity, or biological discovery. Those claims remain
pending the separately frozen B/NMS evaluation and scientific audit.

## Runtime boundary

Online Pareto runtime is the sum of acquisition transfer/conditioning,
representation formation, radial Gamma-LS moments, and frozen-floor score-map
application, each normalized per source frame. Scale-floor fitting and tail
metric calculation are calibration/evaluation costs and are reported
separately, not charged to online inference. CUDA-event synchronization is
required for device timing.

For grid efficiency, the executor makes four causal source/Gaussian passes
(one per sigma) and updates all four EMA states from each spatially filtered
chunk. This reuse shortens the physical grid run. A candidate's reported
deployment runtime still charges the full transfer and spatial-filter cost;
shared work is not divided by four when making the Pareto comparison.

## Execution

No GPU result is implied by this plan. After a fresh base preflight exists:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.conditioning_sensitivity preflight \
  --config examples/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1.example.json \
  --base-preflight Outputs/GammaLSDifference/<fresh-base-preflight> \
  --output Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1_preflight_<run>

.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.conditioning_sensitivity run \
  --config examples/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1.example.json \
  --sensitivity-preflight Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1_preflight_<run> \
  --output Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1_gpu_<run>
```
