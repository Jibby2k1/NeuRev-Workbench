# Motion and registration-confound audit v2

- Status: bounded engineering screen complete; local fields not usable
- Program: `NREV-PRG-0001`
- Experiment: `NREV-EXP-0025`
- Canonical run: `NREV-RUN-EXP-0025-SCREEN-20260830-E`
- Versioned runner:
  `neurobench/experiments/neuron_identifiability/motion_registration_confound_audit.py`
- Parent endpoints: `NREV-RUN-EXP-0028-SCREEN-20260829-B` and
  `NREV-RUN-EXP-0029-SCREEN-20260830-B`
- Claim-bearing execution: not authorized

This protocol turns the migrated motion-audit queue item into a complete native
engineering design. It asks whether translation-like frame changes,
registration residuals, or stored acquisition rails are reliable enough to
interpret as nuisance fields and whether their window-level summaries are
associated with frozen EXP-0029 diagnostics.

The screen does not motion-correct a scientific endpoint. Phase correlation
cannot distinguish specimen motion from deformation, scan effects,
fluorescence activity, gain drift, or structured noise, so every estimate is
called translation-like rather than biological motion.

## Frozen scope

The primary panel is the exact 12 EXP-0029 source-off windows: three temporal
MAD strata from each of four recording groups. Every window has shape
`32 x 64 x 64`, producing `31` adjacent-frame pairs and `372` pairs total.
Raw-HC and JEPA/random residual diagnostics are read from the hash-verified
EXP-0029 Run-B tables without refitting or endpoint tuning.

The existing EXP-0028 acquisition audit supplies differently sampled
full-field context over 11 `060126` recordings. It is not pooled with the
12-window panel. Only the three shared `060126` recording identities receive a
local/full-field contextual ratio; endpoint associations remain limited to the
four primary recording groups.

## Translation and matched-support contract

For each adjacent pair, bidirectional phase correlation estimates the
correction applied to the moving frame. The search is bounded to eight pixels
per coordinate. A shift is valid only when the configured peak and extent
checks pass.

Raw and registered difference scale must use identical support. For every
valid shift, the runner forms one conservative symmetric interior by removing

```text
ceil(max(abs(dx), abs(dy))) + 2 pixels
```

from every edge. It computes both raw and registered MAD/RMS on exactly that
interior. Full-frame raw difference MAD remains a standalone change diagnostic
and never serves as the denominator of the registered/raw reduction ratio.
Invalid shifts have no matched-support fields.

This contract is fail-closed because an earlier provisional package compared
full-frame raw MAD with cropped registered MAD. That support mismatch altered
ratios, review triggers, and four association rows. The corrected Run E was
executed into a fresh output root; the earlier A–D roots are provisional and
unregistered.

## Reliability diagnostics

Reciprocal forward/reverse phase estimates are conjugate by construction. Their
closure is retained only as an implementation-symmetry check, not an
independent reliability measure. Local translation-like estimates are instead
reviewed using:

- independent two-step temporal cycle closure;
- valid-pair fraction;
- search-boundary concentration;
- disagreement between the `2 x 2` tile grid and the crop-global estimate
  (global only within the 64-by-64 local window);
- matched-support registered-difference reduction; and
- contextual disagreement with the differently sampled full-field audit.

A local window requires review if any frozen engineering trigger fires:

- valid global-pair fraction below `0.50`;
- more than `0.10` of valid estimates concentrated at a search boundary;
- tile/global p95 disagreement above `2.0` pixels; or
- median matched-support registered-difference reduction below `0.10`.

These thresholds are engineering review triggers, not validated biological
cut points. Passing them would permit further nuisance modeling, not identify a
motion field.

## Grouped association screen

Fourteen fixed predictor/outcome pairs connect translation/reliability/change
diagnostics to frozen raw-HC and JEPA residual summaries. The analysis reports
all-window Spearman correlation, within-recording ranks, exact within-recording
permutations, leave-one-recording-out ranges, and Benjamini–Hochberg correction
over the complete 14-test family.

With three windows per recording, the complete grouped permutation space is
`6^4 = 1,296`. Four recording groups are too few for a generalizable effect;
all associations remain descriptive even if a multiplicity threshold is met.

## Feature-support and sensor boundaries

The validated carrier, coherence, and recurrence trace lanes are not joined.
Their finite Spon support is UI frames 1800–2359, while all three frozen Spon
background windows end earlier; the `060126` recordings have no matched
exported feature traces. Forcing a join would change the estimand or create a
NaN-driven comparison.

Stored `uint16` low/high-code occupancy is measured exactly, but detector/ADC
bit depth, analog rails, gain, black level, and upstream clipping metadata are
absent. Digital-code occupancy cannot rule analog saturation in or out.

## Execution and scientific boundary

Run E requires exact raw hashes, parent indexes, input inventory, 372 pair
rows, 12 window rows, four recording rows, 14 association rows, matched-support
semantics, portable command/Git/runtime/timestamp provenance, and an exact
artifact index. It records a content-aware digest when the checkout is dirty.

The numeric package is not a complete scientific audit. Required full-field
videos, candidate-surrogate close-ups and full-duration traces, matched
figures, and decode/pixel validation are absent. Accordingly,
`scientific_completion=false`, `scientific_promotion_allowed=false`, and no
claim or evidence capsule is created.

See the
[motion and registration-confound results](../research/MOTION_REGISTRATION_CONFOUND_V2_RESULTS.md)
for Run E's reliability and association results.
