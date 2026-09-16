# Persistence of saved Gamma-LS proposals

This additive analysis follows the user's request to inspect all persistent
detections, with real neuronal/artifact identity left for review. No simulated
calcium events, image acquisition, detector rerun, threshold refit or biological
labels are introduced. The existing complete studies and their files stay frozen.

## Inputs and scope

The real panel is the eight level/difference by X/A/C/Z methods in
`Outputs/GammaLSNecessity/necessity_20260914_r1`, case `real`. Every emitted q1
proposal is included: 4,061 rows, including the two methods with zero rows.
Application is UI1800-2359 (560 frames, 11.2 seconds); setup was UI1600-1799
(200 frames, four seconds). Coordinates in the score crop become original-source
coordinates by adding [49,49] to [x,y]. The source is 50 Hz and 0.5 um/pixel.

For a known-noise comparison, include the 252 physical cells in
`Outputs/GammaLSBackground/background_20260915_r1`: 99,691 emitted q1 rows,
UI165-464 (300 frames, six seconds) each. These are previously generated
source-free controls, not new calcium simulations. Their 36 logical aliases
are not additional evidence or replicates. The real and null panels have
different fields, durations and setup windows; rates retain their denominators.

Only `audit_candidates.json` is consumed. All-positive prefixes contain points
that the frozen detector did not emit at q1 and are outside this analysis.
Current full-field videos, original review-site closeups and complete traces
already cover the exact emitted rows. Reuse is verified through completed
source manifests and complete candidate-to-original-media mappings.

## Association, episodes and recurrence

Freeze radii 2, 4 and 6 pixels (1, 2 and 3 um), crossed with zero or one allowed
missing frame. The primary description uses 4 pixels and one missing frame.
These are sensitivity settings, not estimates of neuron size or identity.

Process frames in order. Each location group has an immutable anchor at its
first proposal. At each frame, form edges to existing anchors within the radius;
use the documented nearest-first greedy assignment with deterministic ties and
at most one proposal per anchor per frame. Every unmatched proposal creates a
new anchor. Do not move anchors, merge groups through chains of nearby points,
discard singletons or give one proposal multiple group assignments. Record
ambiguous choices and conflicts. Existing anchors remain available for recurrence.

Split a location's assigned stream into episodes when its allowed gap is exceeded.
Report observed frame count, first-to-last elapsed time, inclusive frame span,
occupancy, longest uninterrupted run, episode count and boundary censoring.
Three, five and ten observed frames provide descriptive persistence cutoffs.
The main cutoff is three observations. With one missing frame allowed, three
observations need not be consecutive. A third-observation confirmation time is
causal; retrospectively attributing earlier members to a qualifying episode is
a descriptive summary, not an earlier detection trigger. No controller is run.

The same immutable anchor defines recurrence across separated episodes. An
algorithmic location group is not a neuron, and separate groups need not mean
separate neurons. Persistent or recurrent real locations remain unknown until
independent review. All proposals are false only in the exhaustive synthetic null.

## Setup calibration, in plain language

Setup calibration chooses the operating cutoff from an initial part of a
recording. It also chooses a small lower bound for the local spread denominator.
The kernel geometry and conditioning remain fixed. Local reference means and
spreads continue to be calculated on each new frame; they are not frozen images.
The score cutoff and denominator lower bound are held fixed during application.

The recent null control's two-second setup allowed at most six q1 proposals
over its small field. The real panel used four seconds and a larger field,
giving a budget of 109 proposals. These are setup operating budgets, not counts
of confirmed neurons, estimates of real false positives, or guarantees about
later proposal rates. A real setup segment may itself contain activity.

Comparing methods at matched burden means comparing retained information at
similar output workloads. It is not sufficient for one method to cover more
known observations simply by emitting many more proposals. In real data with
incomplete labels, matched *total* proposal burden and coverage of known windows
can be described; exhaustive false burden and sensitivity cannot be inferred.
Simulated calcium-event recovery is deferred under the user's current preference.

## Execution and audit

Use the repository Python environment, one numerical/codec thread, low priority,
and CPUs other than6/7. The noncolliding output root is
`Outputs/GammaLSPersistence/persistence_20260915_r2`. Pass this explicit root
to every command. The first attempt, `persistence_20260915_r1`, completed
association but stopped at an inherited metadata-format mismatch before its
media ledger was produced. Its protocol, results and source snapshot remain
preserved. The second attempt uses the same proposals and grouping rules with
an audit reader that supports the older filename-keyed metadata mapping.

Run `persistence_study preflight`, then `persistence_study run`,
`persistence_audit`, and `persistence_report`. Each takes `--root ROOT`.
The independent `persistence_validate` command is read-only by default;
`--write` adds a final source capsule and completion manifest only after all gates.

The [scientific audit standard](SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) stays enabled.
All exact original q1 audit media and metadata are verified and linked; none is
selectively disabled. New tables map every input proposal to an algorithmic
anchor/episode and to its original review-site media. Original trace-pixel
coordinates remain explicit and are never relabelled as new anchor coordinates.
Real spatial maps and a review table expose every primary real group. Source-free
Expert sections and sparse-real acceptance status retain their original meanings.
All four report figures and the review artifact receive presentation checks.
Independent validation verifies count conservation, radius/gap and time semantics,
source/coordinate mappings, JSON/TSV agreement and the complete source bindings.

Short clips support descriptive recurrence and persistence. They do not establish
rare long-duration false-trigger risk, biological identity, neuronal sensitivity
or feedback-control performance.
