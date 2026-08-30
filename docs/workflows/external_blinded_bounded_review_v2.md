# External blinded bounded review v2

## Purpose

This workflow packages one frozen enriched 192×192 field for independent team
annotation. It tests within-recording source visibility, identity, local
detection behavior, and processed-evidence interpretation without exposing
current labels or detector rankings.

It is independent annotation validation, not an independent biological
recording or histological confirmation.

## Generated package

The validated output is:

`Outputs/NeuronIdentifiability/external_blinded_bounded_review_v2/`

Share the archives in this order:

1. `NeuRev_external_review_v2_PHASE_A_RAW_FIRST.zip`
2. Receive and validate the locked Phase A JSON.
3. `NeuRev_external_review_v2_PHASE_B_ASSISTED.zip`

Never share `NeuRev_external_review_v2_PRIVATE_ADMIN.zip`.

## Phase A

Phase A presents four detector-blind Raw-only burst clips, mean and maximum
projections, and an interactive spatial overlay. Reviewers click every visible
source, assign a source class, confidence, timing, and cross-clip identity, then
sign off every clip and the complete region. Final export remains disabled
until coverage and reviewer provenance are complete.

## Phase B

Phase B presents 18 candidates under a new deterministic opaque-ID permutation.
The embedded source ID and coordinates are removed from the video header. Each
video contains Raw, CS-Parzen ICA, and local-standardized close-ups plus exact-
center complete traces. Reviewers record neuron likelihood, identity relation,
stage visibility, morphology, limitations, possible center offsets, confidence,
and notes.

Phase B is distributed only after Phase A is locked. This makes processed
evidence an assistance/adjudication phase rather than a contaminant of Raw-first
source enumeration.

## Reviewer instructions

Each reviewer receives a different opaque reviewer ID and works independently.
After extracting a ZIP, open `START_HERE.html`. If browser security prevents
local video playback, run:

```bash
python serve_review.py
```

Then open `http://127.0.0.1:8765/START_HERE.html`. Drafts autosave in that
browser and remain local until JSON export.

## Administrator workflow

1. Verify the archive SHA-256 values in `experiment_contract.json`.
2. Assign at least two opaque reviewer IDs.
3. Distribute Phase A only.
4. Retain both locked Phase A JSON files.
5. Distribute Phase B only after Phase A receipt.
6. Retain both locked Phase B JSON files.
7. Extract the private administrator archive inside the repository environment.
8. Run its `score_submissions.py` wrapper with all four submissions.
9. Resolve the generated adjudication queue with an independent adjudicator.
10. Freeze the adjudicated truth revision before computing detector metrics or
    fitting the uncertainty-aware learning program.

Automated analysis reports spatial matches, localization distances, matched
class agreement, Phase B call agreement, reviewer call distributions, and a
disagreement queue. It does not silently turn agreement into truth.

## Rebuild

The output root is collision-safe. Rebuild only into a new versioned path:

```bash
.venv-neurobench/bin/python \
  -m neurobench.experiments.neuron_identifiability.external_bounded_review \
  build \
  --source-run Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8 \
  --candidate-media Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1 \
  --output Outputs/NeuronIdentifiability/external_blinded_bounded_review_v3
```

## Validation contract

- Four Raw-first videos and 18 assisted videos exist and decode.
- All videos use 10 FPS; assisted videos contain 560 frames.
- Public ZIPs contain no detector-site IDs, ranks, scores, current decisions,
  randomization key, absolute source paths, or private reference labels.
- All ZIP entries are relative and traversal-safe.
- Phase A coordinate conversion is tested at boundaries.
- Submission validation fails closed on incomplete coverage, phase-order,
  reviewer provenance, unsupported labels, or unlocked exports.
- At least two reviewers are required for automated agreement analysis.
- Adjudication remains mandatory before truth or detector metrics.

## Permitted conclusions

After complete independent review and adjudication, the package can support
claims about the frozen enriched region: local source counts, local precision
and recall, false positives, misses, duplicates, localization, cross-burst
identity, reviewer agreement, and whether processed evidence changes visibility
or confidence.

It cannot establish whole-recording precision, independent-recording transfer,
histological neuronal identity, causal connectivity, or population-level
generalization.
