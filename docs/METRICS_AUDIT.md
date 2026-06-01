# Progress

The workbench Progress page is available at:

```text
http://127.0.0.1:8765/#progress
```

Legacy links to `#metrics` and `#audit` still open this page.

It summarizes the live annotation state from `annotations.json` and browser
memory. It is intended for review management, not final scientific evaluation.

Use this page before changing detector thresholds or exporting reviewed data.
It answers three practical questions:

1. Do we have enough reviewed examples to tune parameters less blindly?
2. Is the candidate generator producing too many false positives or missed
   candidates?
3. Are the labels complete enough to share or compare across reviewers?

Current summaries include:

- ROI state counts
- event state counts
- discovery suggestion outcomes
- trace-quality labels
- control-readiness labels
- triage category counts and queue IDs
- candidates per accepted ROI
- candidate events per accepted event
- review progress toward the first tuning milestone
- the recommended next annotation batch
- robustness examples that jump back to the Review page
- validation and 100 Hz real-time readiness for the active run
- local two-file adjudication for reviewer disagreements

The same summary can be generated from files:

```bash
python3 tools/compute_annotation_metrics.py \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_video_2/app/annotations.json \
  --out Outputs/NeuronReview/calcium_video_2/annotation_metrics.json
```

High candidate-per-accepted counts indicate that the detector is still too
permissive or that the review queue needs better ranking. Low control-ready
counts indicate that accepted neurons are not yet clean enough for downstream
inverse-dynamics work.

## Robustness Example Gallery

The gallery picks representative examples from the current review state and
embedded review data:

- accepted or strong neuron
- uncertain or weak-trace ROI
- artifact-like ROI
- large or merged-cluster candidate
- event-supported ROI
- missed-neuron suggestion

Each card jumps back to Review. Use this as a quick fixed sanity set whenever
you adjust thresholds, change overlays, generate a new preview, or compare two
runs. The gallery is not ground truth; it is a convenience layer for repeated
human inspection.

## Validation And Real-Time Readiness

The validation panel summarizes the active run's pipeline validation and
real-time metadata:

- target frame rate and per-frame budget
- estimated stage latency metadata
- GPU-sensitive stages
- offline/batch stages that are not closed-loop-ready
- synthetic latency smoke-test command
- local generation backend readiness

For 100 Hz samples, treat offline stages and unknown latency as warnings. They
may still be useful for offline analysis, but they need a streaming
implementation before closed-loop or real-time use.

## Adjudication Comparator

The browser-side comparator loads two `annotations.json` files and creates a
local disagreement table. It highlights missing labels and label conflicts for
ROIs, events, and suggestions. The comparison never uploads the files; it runs
inside the dashboard page.

Use it when two reviewers have labeled the same dataset and a final decision is
needed. Open each disagreement in Review, resolve the final label in the active
annotation file, then export the resolved annotations.

## Annotation-Driven Tuning Gate

The current detector settings should not be tuned as if the candidate output
were ground truth. The Metrics page now reports whether the session has reached
the first practical tuning gate:

- at least `20` reviewed ROIs
- at least `20` reviewed events

The threshold is intentionally modest. It is enough to make obvious parameter
comparisons less blind, but it is not a final validation set. The recommended
batch is also available from:

```bash
python3 tools/build_annotation_batch.py \
  --review-data Outputs/NeuronReview/calcium_rest_cropped/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_rest_cropped/app/annotations.json \
  --out Outputs/NeuronReview/calcium_rest_cropped/annotation_batch.json
```

## Triage Categories

File-generated annotation metrics include `triage_categories` and
`triage_queue_counts`. These fields are helper summaries for review queues, not
new scientific labels. They are derived from existing `annotations.json` and
`review_data.json` fields:

- `strong_neuron`: accepted ROI with good trace quality, yes/maybe control
  readiness, no artifact class, and low artifact score
- `possible_missed_neuron`: discovery suggestion marked missed or promoted
- `artifact_like`: ROI or suggestion with an artifact class/cue/state, or high
  artifact score
- `merged_cluster`: ROI marked for merge, ROI sharing an identity group with
  another ROI, or a virtual merge ROI
- `weak_trace`: ROI with weak, noisy, or unusable trace quality
- `needs_event_review`: unlabeled/unsure event, weak event, ambiguous timing, or
  slow-transient timing
