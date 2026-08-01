# Spon Ca Burst hard-ROI adjudication and frozen re-evaluation

## Decision and scope

This workflow turns targeted expert feedback about repeatedly missed Spon Ca
Burst observations into a versioned review layer. It never edits the original
79 sparse-positive labels, tunes a detector, treats unlabeled pixels as
negatives, or authorizes a GPU/full sweep.

The engineering workflow and mechanics validation are complete. Scientific
completion requires a human reviewer to finalize the target observations and
then run the gated exact re-score.

## Immutable inputs and coordinate contract

- Labels: `Inputs/Spon Ca Burst/labels/labels_normalized.tsv`
- Movie cache:
  `Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy`
- Manifest: `examples/spon_ca_burst_hard_roi_adjudication_v1.example.json`

All manifest paths are resolved relative to the manifest, independent of the
calling directory. UI frames are one-based and inclusive; array intervals are
zero-based and half-open. Coordinates use `x=column`, `y=row`. New label-driven
runs write a projection overlay during preflight.

The original label digest recorded by the completed review pack is:

```text
8f008fede5771d83e949d805094292aae7516cc30353dd383da7ff1cf20b077e
```

## Completed review artifacts

The detector-blinded review pack is:

```text
Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_v1
```

It contains four padded clips, four projection overlays, a 79-row
`adjudication_draft.tsv`, `review_manifest.tsv`, input hashes, and preflight
evidence. The clips show raw intensity, fixed pseudo-color, and positive change
from a pre-window baseline. They contain no candidate, score, recovered/missed
status, or detector name.

The raw-trace review aid is:

```text
Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_review_checklist_v2
```

It contains 25 observation plots and `pending_review_checklist.tsv`. Timing
suggestions use only a local center-minus-annulus raw trace and are deliberately
conservative about lookback leakage. They are reviewer aids—not labels—and the
manifest records `automatic_adjudication_performed=false` and
`detector_outcomes_used=false`.

## Review schema and target panel

Each original observation has one row with original/canonical identity,
coordinates, original timing, optional event onset/peak/end, neuron and activity
confidence, morphology, context, disposition, confirmed/inclusive inclusion,
review status, reviewer provenance, source note, and merge reason.

The target identities are ROI 007, 008, 010, 014, 015, 017, 019, 020, and 023.
This includes the required hard panel plus uncertain burst-2 ROI 008 and 017.
The draft has 11 `provisional_expert_note` rows and 14 `pending` rows; neither
state is a final adjudication.

Allowed dispositions are `confirmed_neuron`,
`activity_visible_identity_uncertain`, `artifact`, `background`, `unresolved`,
and `pending_review`. Observation timing is either blank or a complete ordered
onset/peak/end triple. A final re-score refuses any target identity with a row
that is not `review_status=adjudicated`.

## Commands

The completed review roots must not be reused. For a new version, change the
manifest to new preflight and output roots, then run:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.hard_roi_adjudication \
  preflight \
  --config examples/spon_ca_burst_hard_roi_adjudication_v1.example.json

.venv-neurobench/bin/python -m neurobench.experiments.hard_roi_adjudication \
  review-pack \
  --config examples/spon_ca_burst_hard_roi_adjudication_v1.example.json
```

The promoted final evaluator is the exact CPU reconstruction path:

```bash
.venv-neurobench/bin/python \
  -m neurobench.experiments.hard_roi_adjudication.exact_rescore_cli \
  --config examples/spon_ca_burst_hard_roi_adjudication_v1.example.json \
  --adjudication-tsv /path/to/final_adjudication.tsv \
  --output-dir Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v1
```

Do not add `--allow-provisional` to a final run. That option exists only for
mechanics validation and stamps output `provisional_preview`.

## Frozen evaluation contract

The evaluator reports original, confirmed-only, and inclusive label views,
each with original and observation-specific timing, at budgets 20, 40, 58, 80,
and 100. At budget 58, each observation is assigned one of: matched, identity
conflict, ranking miss, temporal miss, localization miss, NMS suppressed, or
proposal miss. Unmatched candidates remain `unknown_not_negative`.

Quantitative lanes are the immutable `carrier_signed` tensor and deterministic
CPU reconstructions of `coherence_w15` and standalone
`propagation_lag2_w15`. `radial_cs_shell` and `noise_vst_residual` are
diagnostic-only because only display-clipped TIFFs were preserved.

The exact mechanics preview is:

```text
Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_exact_causal_preview_v1
```

It reproduced the following original-label/original-timing macro recall:

| Frozen lane | B20 | B40 | B58 | B80 | B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Carrier | 0.540864 | 0.657246 | 0.703080 | 0.715580 | 0.726449 |
| `coherence_w15` | 0.605305 | 0.676449 | 0.722283 | 0.722283 | 0.745057 |
| standalone `propagation_lag2_w15` | 0.621972 | 0.680616 | 0.749819 | 0.761724 | 0.772593 |

These are reproduction landmarks, not revised scientific results. The
standalone lag result is not interchangeable with the lower cross-fitted
all-family selector result from the scientific feature audit.

## Current provisional interpretation

- ROI 010/015 are 3.171 pixels apart, inside the frozen six-pixel match/NMS
  radius; bursts 1 and 2 therefore expose an identity conflict until the merge
  is anatomically adjudicated.
- ROI 014 and 019 show consistent relaxed-radius localization hypotheses at
  approximately 6.91 and 7.71 pixels, respectively.
- ROI 017 in burst 2 is a budget-58 ranking miss at rank 78.
- ROI 023 in bursts 3 and 4 has local evidence suppressed by six-pixel NMS.
- ROI 007 has mixed matched, localization, and NMS outcomes rather than one
  uniform proposal failure.

These statements describe frozen evaluator behavior and must not determine the
human labels.

## Human completion gate

The reviewer must confirm or reject ROI 010/015 per relevant burst; adjudicate
ROI 007 and 023 across remaining occurrences; decide the uncertain ROI 008,
017, and 020 observations; accept or replace timing where early onset is
claimed; record reviewer identity and timestamp; and save a new versioned final
TSV. Only then may the non-provisional exact re-score be run into a new output
root.

Targeted review is conditioned on known misses. It cannot estimate precision
or supply unbiased hard negatives; those require later exhaustive review of a
fixed bounded field.
