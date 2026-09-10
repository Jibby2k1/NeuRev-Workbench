# Compact Gamma-LS paper v2 outline

## Scope decision

The paper is an applied architecture paper about fast event-candidate
extraction for closed-loop optical experiments. The primary operator name is
**Gamma-weighted Local Standardization (Gamma-LS)**, followed by a calibrated
CFAR threshold and spatial NMS. The legacy name “Quadratic Gamma Detector” is
retired from the title, abstract, headings, figures, and primary claims; it may
appear once in historical attribution if needed to connect prior literature.

The architecture is reported in the order it executes:

```text
acquired frames
  -> selected representation input contract
       adjacent: common causal conditioning
       six-lag: acquisition raw
  -> temporal representation selected by the new ablation
  -> Gamma-LS
  -> CFAR threshold + NMS
  -> frame-level candidate stream
```

If the learned representation fails its advancement gate, “temporal
representation” becomes energy-normalized temporal difference. If it passes,
it becomes multi-lag CS-Parzen residual energy on the acquisition-raw domain,
and full-rank PCA whitening is explained inside the ICA fit rather than drawn
as an online stage.

Max pooling and fractional whitening are excluded from the compact main story.
They were useful diagnostics but currently lack promoted incremental detection
evidence. The old Evaluation Contracts section/table is removed; essential
population and claim boundaries remain in short prose and captions.

## Eight-page allocation before references

1. Abstract and index terms: 0.3 page.
2. Introduction and closed-loop motivation: 0.9 page.
3. System and operator methodology: 2.7 pages.
4. Experimental design: 1.0 page.
5. Essential results: 2.0 pages.
6. Discussion, limitations, and conclusion: 1.1 pages.

The Methods should explain only the deployed finalist plus the one decisive
representation ablation. Detailed grids, label-version histories, all
diagnostic controls, and scientific-audit inventories move to the supplement
or versioned workflow artifacts.

## Main figures

### Figure 1 — System architecture

Use distinct visual components rather than uniform boxes:

- a short image-strip glyph for acquired frames and causal buffering;
- paired/stacked frames plus a learned-axis or fixed-difference glyph for the
  representation;
- a target cell, guard disk, and Gamma-weighted reference ring for Gamma-LS;
- a threshold gauge and separated-peak map for CFAR/NMS;
- a timestamped candidate stream entering an inverse-control loop.

The diagram must not place a legacy feature detector downstream of ICA or show
PCA as a separately executed online block.

### Figure 2 — Operator anatomy

Show visually distinct supports at their true proportions:

- adjacent temporal difference and six-lag delay embedding on one time axis;
- the test pixel, explicit guard disk, radial Gamma reference weights, and
  boundary renormalization;
- small/medium/large Gamma contexts as genuinely different footprints;
- one selected context highlighted, with multi-context fusion labeled only if
  that arm is actually run.

Include the normalized radial kernel profile beside the 2-D stencil. Keep the
archived center-only, mode-outside-support kernel as a small diagnostic inset,
not the canonical operator.

### Figure 3 — Spon Ca Burst stage anatomy

Rebuild from current source arrays and one frozen frame sequence. Use the same
spatial crop and timestamps across panels:

1. acquired calcium frames;
2. common causal conditioning;
3. fixed difference, PCA-only, and learned representation outputs;
4. Gamma local mean/reference scale and signed Gamma-LS score;
5. thresholded/NMS candidates with sparse-positive overlays added only after
   the candidate artifact is frozen.

No panel is copied from the legacy voltage figure. Max-pooling panels remain
out unless a later promoted experiment changes that decision.

### Figure 4 — Accuracy/burden and efficiency

Use two answer-first panels:

- protected known-positive recall versus per-burst candidate budget and versus
  held-out quiet candidate burden, with paired intervals for the fixed and
  learned representations;
- batch throughput and causal one-frame latency (p50/p95/p99), including the
  1-ms voltage-imaging deadline and backlog/missed-deadline rate.

Report unmatched candidates per image/frame as unknown proposal burden, not
false-positive rate. Put the final automated full-recording candidate count in
the panel or a small adjacent table, with threshold and grouping grain stated.

## Essential results only

The main Results section should answer four questions:

1. Does learned ICA recover more known positives than energy-normalized
   differencing under the same frozen Gamma-LS/CFAR context?
2. What quiet candidate burden is required for that recovery?
3. How many candidates does the frozen system produce over a complete recording
   without annotated burst windows?
4. Does the frozen GPU implementation sustain the declared batch and 1-kHz
   streaming workloads?

Do not retain a large ledger of incompatible historical experiments in the
main text. Current negative evidence belongs in one sentence where it changes
a design choice; for example, max pooling is omitted because the exploratory
incremental ranking interval spans zero.

## Claim gate

The manuscript architecture is not frozen until the Gamma-LS difference/ICA
ablation and synchronized GPU timing pass their audits. Until then, figures
should be treated as design specifications rather than final result figures,
and no 1-kHz, multiscale-LS, end-to-end biological-detection, ICA-superiority,
precision, or false-positive-rate claim should be made.
