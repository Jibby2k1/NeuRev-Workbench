# Unlabeled Model Proposal Review

## Purpose

Use this workflow when a recording has frozen model candidates but no expert
annotations yet. It creates a model-only review dashboard, a model-assisted
proposal workbook, a blinded expert template, long-form proposal tables, and
explicit `not_applicable_pending_labels` audit states.

A model proposal is unknown until a reviewer acts on it. It is never an expert
annotation or a false positive by default.

## Required source contract

The packaging command consumes a completed Workbench app whose
`review_data.json` contains:

- `annotationCorrection.view_contracts` with synchronized Raw/processed frame
  mappings and browser-readable frame patterns;
- at least one frozen `annotationCorrection.model_rois` item;
- stable model IDs, source coordinates, event intervals or occurrence members,
  scores/ranks when available, and compact traces;
- a frozen run ID and immutable source media.

No expert ROI, match table, or label workbook is required. The package builder
removes expert ROIs, matches, links, match distances, and expert-support fields
even when the source app is a historical labeled proof fixture.

This command packages existing inference results; it does not run or tune the
scientific model.

## End-to-end inference for a new TIFF

When a TIFF has no candidate-bearing Workbench app yet, use the guarded
arbitrary-recording runner first. It applies the frozen v5
`Raw -> multi-lag MSICA persistence -> joint MSLN` architecture, proposes
event windows from robust peaks in framewise MSLN energy, freezes a bounded
candidate-surrogate panel, builds the source app, and invokes the package
builder. No labels are read.

The fixed lane is
`multilag_2d__normalized_hsic__short__uniform__bandwidth_scale-0p5::persistence::joint_s5_g1_t31_g1`.
MSICA uses lags `[0, 1, 2, 4]`; joint MSLN uses spatial outer/guard widths
`5/1` and temporal window/guard lengths `31/1`. Each event contributes at
most 58 NMS-separated occurrences. The review panel is capped at 32
consolidated identities ranked by recurrence and then peak evidence; the full
screening table remains available.

Run the mandatory collision/resource preflight, then its matching run:

```bash
.venv-neurobench/bin/python -m \
  neurobench.experiments.msln_msica.unlabeled_recording preflight \
  --input-tif <recording.tif> \
  --output-root <new-inference-root> \
  --review-root <new-review-root>

.venv-neurobench/bin/python -m \
  neurobench.experiments.msln_msica.unlabeled_recording run \
  --input-tif <recording.tif> \
  --output-root <matching-inference-root> \
  --review-root <matching-review-root>
```

The TIFF remains read-only and memory-mapped. One CUDA worker processes
96-frame chunks under an 8 GiB VRAM cap. A global scale floor calibrated from
eight separated quiet-rich pilot chunks stays fixed across MSLN chunks to
avoid seams. Raw/MSICA/MSLN alignment begins at UI frame 33 because one MSICA
history frame and 31 causal MSLN history frames are required.

If TIFF metadata lacks acquisition timing, the package records the rate as
unknown. The 10 fps MP4 setting is visualization playback only.

The Model section includes the five-stage full-field video, one full-duration
Raw/MSICA/MSLN close-up and exact-pixel trace per frozen identity, and one
five-stage instant montage per selected occurrence. Expert and Comparison are
explicitly `not_applicable_pending_labels`.

## Command

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench \
  model-proposal-package \
  --source-app-dir <completed-model-app> \
  --output-root <new-unused-output-root> \
  --event-source supplied \
  --json
```

Use `--event-source supplied` when event windows come from acquisition,
stimulus, or other external timing. Use `--event-source model_proposed` when
the detector also proposes the event windows. Never silently relabel
model-proposed timing as supplied timing.

The command refuses output collisions and preserves the source app. Frame assets
are hard-linked when possible and copied otherwise.

## Output contract

```text
<output_root>/
  REPORT.md
  proposal_manifest.json
  validation.json
  app/
    review_data.json
    annotations.json
    annotation_revisions/<model-proposal-draft>/
    frames/
  proposal_exports/
    MODEL_PROPOSALS_FOR_REVIEW.xlsx
    BLINDED_EXPERT_TEMPLATE.xlsx
    model_proposals_long.tsv
    model_proposal_identities.tsv
  audit/
    1_Expert_Annotations/status.json
    2_Model_Annotations/status.json
    3_Comparison/status.json
```

The model-assisted workbook contains:

1. an expert-compatible event-block layout with `proposal/rank, x, y`;
2. one row per model occurrence with a globally unique occurrence ID;
3. one row per stable model identity;
4. frozen-run and coordinate/frame provenance.

The blinded workbook contains event windows and empty point blocks only. Give
this version to experts first when independent annotation is required.

## Dashboard behavior

For a model-only payload, the correction workspace defaults to:

- queue: **Model proposals**;
- overlay: **Selected model only**;
- expert state: `not_applicable_pending_labels`;
- comparison state: `not_applicable_pending_labels`.

Raw and processed frames, close-ups, traces, scrubbing, and trace interaction
remain available. Promoting a proposal creates a separate expert ROI with
proposal provenance and never mutates the frozen model candidate.

## When labels arrive

Import the expert workbook as a new annotation revision. Keep the proposal
package and frozen run immutable. Then generate expert/model matching and a new
three-section scientific audit. Do not retrofit expert support fields into the
original model proposal table.

## Implemented proof

The current non-destructive proof is:

```text
Outputs/NeuronReview/spon_ca_burst_multilag_msica_v5_model_proposal_review_v2
```

It contains 156 model identities and 232 uniquely identified occurrences, zero
expert ROIs, zero expert/model matches, and an empty expert annotation revision.
It uses historical supplied event windows and existing frozen candidates, so it
validates the workflow and artifacts rather than serving as an independent
unlabeled-recording confirmation.

## Completed 072126 label-free runs

The first independent new-recording executions are:

```text
Outputs/NeuronReview/072126_6_left_4_multilag_msica_msln_model_proposal_v1
Outputs/NeuronReview/072126_6_right_2_multilag_msica_msln_model_proposal_v1
```

Both used eight model-proposed event windows and froze 32 identities. The left
package contains 80 selected occurrences from 464 screened event-level
candidates and 6,355 aligned frames. The right package contains 67 selected
occurrences from 464 screened candidates and 6,124 aligned frames. Both
model-only audit inventories passed. These counts describe review burden, not
precision or biological validity; all proposals remain unknown pending expert
review.
