# Spon Ca Burst Gamma-LS paper-success campaign

**Frozen on:** 2026-09-08, before the support-extension, protected, full-recording,
and sustained-streaming results were inspected.

## Decision target

Select the smallest justified causal architecture for the compact paper:

```text
acquired frame -> fixed or learned temporal representation
               -> signed radial Gamma-LS -> empirical CFAR threshold
               -> deterministic spatial NMS -> automated proposals
```

The paper retains ICA only if protected paired results justify its added fitting
and inference cost. Searching several Gamma contexts and selecting one is not
called multiscale inference. Max pooling is excluded from this campaign.

## Required evidence blocks

1. **Operator and implementation validity.** CPU/CUDA parity, signed-input,
   boundary, constant, impulse, chunking, legacy-anchor, and deterministic NMS
   tests must pass on the actual GPU runtime.
2. **Gamma-reference sufficiency.** Extend beyond the original half-widths
   7/11/15, keep sparse-positive coordinates sealed during selection, execute
   signed radial Gamma, signed square-annulus, and legacy-exact controls, and
   measure repeated one-frame plus batched CUDA latency and peak VRAM.
3. **Representation value.** Compare Raw, signed difference,
   energy-normalized difference, full-rank PCA-whitened derivative, and
   fold-fitted CS-Parzen ICA with one fold-local Gamma context. Fit the learned
   arms only on the three outer-training bursts plus guard exclusion. The
   six-lag comparison remains a separate matched-support level.
4. **Protected recovery.** Use the frozen v1 79-occurrence/26-identity population
   as primary. Join coordinates only after models, scores, thresholds, and
   candidate hashes are frozen. Use one-to-one spatial matching, B20/40/58/80/100
   per burst, the complete quiet-burden curve, NMS 4/6/8 sensitivity, and paired
   canonical-identity bootstrap intervals. The 106-row v7 confirmed population
   is descriptive sensitivity only.
5. **Automated proposal count.** After selecting the deployment arm, report
   frame-level proposal rows and proposals per eligible frame at every frozen
   threshold. Unmatched proposals are unknown, not false positives or confirmed
   neurons. A unique biological-event count is not inferred without a separate
   temporal-linking contract.
6. **Efficiency.** Separate fitting from inference. Report offline batch
   throughput for chunks 1/8/32/64 and at least 60 seconds of one-frame arrivals
   at a 1 ms cadence, including H2D, representation, Gamma-LS, threshold/NMS,
   D2H, p50/p95/p99/max, missed deadlines, backlog, RAM, and VRAM.
7. **External sensitivity.** If the selected arm is technically portable, score
   the already-eligible `15 right` recording under a frozen label-join order.
   Treat this as one-recording sparse-positive confirmation, not population
   generalization.
8. **Scientific audit and manuscript gate.** Produce source hashes, row-level
   tables, candidate overlays, expert-only/model-only/comparison media, an
   artifact index, validation report, concise result report, and explicit claim
   boundaries before changing the manuscript headline.

## Gamma support stopping rule

The support screen must expose a fold-local small/efficient candidate and a
larger-support comparator without sparse-positive coordinates. Protected labels
may evaluate those already-frozen choices but may not select the support.

A support is called *sufficient* only when all of the following hold:

- it is not the largest tested boundary;
- its training-window contrast is within the predeclared practical tolerance of
  the best eligible support for every fixed representation aggregate;
- its protected budget curve and B58 recall are not detectably worse than the
  larger frozen comparator under paired canonical-identity resampling; and
- the larger comparator does not justify its incremental synchronized latency or
  memory cost.

If the largest tested support is best or practically tied, the boundary remains
unresolved and the coordinate-free screen must expand before any sufficiency
claim. A fast-but-worse kernel and a strong-but-slower kernel are both retained
on the efficiency frontier; runtime alone does not select the scientific result.

## Claim vocabulary

- `automated proposal`: produced by frozen code without a human selecting that
  individual output;
- `known-positive match`: a proposal matched to a sparse-positive annotation;
- `unknown proposal`: an unmatched proposal under incomplete annotation;
- `protected result`: evaluated only after the complete upstream artifact was
  frozen;
- `burst-window-supervised`: allowed to use declared event time windows but not
  sparse-positive coordinates or identities;
- `1-kHz ready`: sustained p99 below 1 ms with no growing backlog under the
  frozen frame dimensions. Batch throughput does not establish this claim.

## Current status at freeze

- GPU runtime: RTX 4070 SUPER, NVIDIA 580.173.02, PyTorch CUDA 13.0 build.
- Focused CUDA test suite: 80 passed, zero skipped before campaign extensions.
- Real-data GPU smoke: complete.
- Original G1/G2 screen: complete; all four folds selected half-width 11, not
  the largest tested half-width 15. Three folds selected shape 9/mode fraction
  1.0; one selected shape 5/mode fraction 0.5.
- Larger-support, protected representation, full-recording, sustained streaming,
  external sensitivity, and final audit blocks: pending at this freeze.

