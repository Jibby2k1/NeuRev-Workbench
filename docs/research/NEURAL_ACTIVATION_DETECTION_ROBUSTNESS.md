# Neural Activation Detection Robustness Plan

## Decision

Develop activation measurement as a staged, calibrated pipeline:

```text
registration/artifact QC
  -> high-recall spatial proposals
  -> persistent neuron identities and footprints
  -> per-neuron trace extraction
  -> activation/event probabilities
  -> calibrated threshold or abstention
```

CFAR remains useful as a proposal/evidence channel. It should not be required to
perform localization, segmentation, identity tracking, trace inference, and
event classification through one local-contrast score.

## Why learnable CFAR did not generalize

The completed experiments do not support a simple initialization diagnosis:

| Experiment | Best learned mean held-out recall | Raw-direct comparator |
| --- | ---: | ---: |
| Guarded contrast v1 | 0.1328 | 0.6056 |
| Stabilized/jittered contrast v2 | 0.2051 | 0.6056 |
| Direct-initialized bounded tuning v3 | 0.6056 | 0.6056 |

In v3, every learned fit reduced its training objective and parameters moved
within the trust region. All nine configurations nevertheless tied direct and
won zero of four bursts. This means the local learning problem was active but
did not change the labeled decisions.

Likely causes:

1. **Incomplete supervision.** Sparse point labels identify known centers but
   not every active neuron, footprint, or event. They support recall/FROC-style
   evaluation, not ordinary precision.
2. **Training/evaluation mismatch.** Training quiet examples are sampled at
   known ROI coordinates. The deployed threshold is set by the strongest
   full-field quiet peaks after NMS.
3. **Statistic mismatch.** Guarded local contrast can suppress broad or
   spatially noncanonical activation that direct amplitude retains.
4. **Task conflation.** A transient pixel response is not the same object as a
   persistent neuron identity or a trace-level activation event.
5. **Post-processing mismatch.** NMS, thresholding, merging, and recurrence
   rules determine final candidates but are largely outside the learned loss.
6. **Domain shift.** Intensity, background, registration quality, fish/session,
   anatomical area, motion, and acquisition rate can change the score
   distribution.

## Precision requires an exhaustive evaluation subset

An unmatched candidate is only a false positive when a reviewer has declared
the relevant spatial-temporal region exhaustively annotated. Every benchmark
region should record:

- fish, session, video, direction/behavior state, and acquisition settings;
- spatial tile and frame interval;
- coverage mode: `exhaustive`, `candidate_review`, or `sparse_positive`;
- reviewer identity, confidence, and adjudication status;
- neuron footprints/centers and persistent identities;
- activation windows with rough/precise timing;
- artifact regions and uninterpretable intervals.

Use sparse-positive regions for recall/FROC and discovery. Use only exhaustive
regions for precision, average precision, false-positive rate, and
negative-predictive claims.

### Suggested annotation sampling

Construct a stratified tile set rather than exhaustively labeling every frame:

- different fish, days, sessions, left/right/rest videos;
- brain regions and image edges;
- quiet, weak, moderate, and strong population activity;
- motion-correction residuals, illumination shifts, vascular/background
  structure, impulse noise, and saturation;
- detector disagreement: direct-only, CFAR-only, external-tool-only, and
  unanimous proposals;
- random tiles independent of all detectors.

Hold out entire fish or sessions before annotation-derived tuning begins.

The workbench's Expert-mode `CFAR Foreground / Background` editor can supply
the per-ROI foreground/background masks for this sampling plan using free-form
brushes and bounded flood fill. These masks do not by themselves establish
exhaustive coverage: precision-oriented benchmarks must still record the tile,
frame interval, and `exhaustive` coverage declaration described above.

## Model the measurement tasks separately

### Task A: persistent neuron localization

Input: registered structural/temporal evidence.

Output: neuron identity, centroid, footprint, anatomical coordinate,
localization confidence, and provenance.

Candidate sources may include:

- raw-direct positive residual;
- fixed or learned CFAR;
- dark-soma/anatomical zones;
- local temporal correlation;
- Suite2p, CaImAn, or other imported proposal runs;
- manual workbench ROIs.

Combine proposals with one-to-one matching and auditable merge/split rules. A
proposal ensemble should optimize coverage; the later classifier/reviewer
determines whether a proposal is a neuron.

### Task B: trace extraction

For each persistent footprint, retain:

- raw and neuropil/background traces;
- dF/F and robust-z traces;
- baseline and noise estimates;
- saturation/motion/artifact features;
- missing-frame and registration-quality flags.

Geometry changes must invalidate materialized traces, matching the current
workbench behavior.

### Task C: activation/event inference

Input: causal trace history plus optional local image evidence.

Output: calibrated activation probability, event onset/peak/duration
distributions, amplitude, and uncertainty.

Compare:

- direct trace amplitude/change;
- robust local-z and fixed thresholds;
- OASIS-style deconvolution;
- small causal temporal CNN/GRU;
- hybrid models that use trace plus local residual image.

This task should not relearn the neuron footprint on every event.

## Training objective aligned to deployment

The next learnable detector experiment should use:

1. **Full-field quiet hard negatives.** Mine the highest NMS peaks from held-out
   quiet blocks, including recurring artifact locations.
2. **Candidate-level ranking.** Compare labeled event scores with hard-negative
   peak scores, not only same-coordinate quiet patches.
3. **Threshold-margin loss.** Emphasize positives and negatives close to the
   selected deployment threshold.
4. **Duplicate/merge penalties.** Penalize multiple proposals for one identity
   and one proposal covering multiple labeled identities.
5. **Calibration loss.** Fit probabilities or uncertainty after model
   selection using held-out calibration data.
6. **Abstention.** Permit `uncertain` predictions when imaging or registration
   quality is insufficient.

Keep raw-direct as a frozen additive or fallback lane. Learned components
should demonstrate incremental value, not replace the strongest baseline by
construction.

## Metrics

### Object/identity level

- precision, recall, F1, and average precision;
- recall at fixed precision and precision at fixed recall;
- centroid distance and footprint IoU;
- duplicate, split, and merge rates;
- recurrence/identity consistency across bursts or time;
- calibration error and abstention coverage.

Reuse `neurobench.metrics.detection` for one-to-one object matching. Extend the
evaluation layer with threshold sweeps, PR/AP, group bootstrap, and coverage
semantics.

### Event level

- event precision/recall and average precision;
- onset/peak timing error and duration error;
- recall by amplitude/SNR and recurrence;
- false events per neuron-minute and per field-minute;
- causal detection latency;
- calibration and abstention.

Reuse `neurobench.metrics.event_quality` for event matching and timing. Add
confidence sweeps and session/ROI-grouped uncertainty.

### Sparse-label/discovery level

- held-out known-center recall;
- masked-ROI Recall@K;
- FROC versus quiet peaks per event map;
- reviewer acceptance among top K novel candidates;
- candidate stability across seeds, sessions, and proposal sources.

Do not report ordinary precision from sparse labels.

### Downstream sufficiency

Measure intent decoding with:

1. manual/exhaustive neural observations;
2. detector observations;
3. detector observations plus uncertainty/abstention;
4. raw/grid features without explicit detections.

This quantifies whether detector errors materially limit the next checkpoint.

## Split and robustness policy

Use nested grouping:

- primary scientific split: held-out fish when available;
- fallback: held-out session/day;
- never split adjacent frames from one event across train and test;
- keep calibration data separate from model selection;
- report direction/rest, region, activity level, and artifact-condition slices;
- include leave-one-proposal-source-out and leave-one-acquisition-setting-out
  sensitivity where data permits.

## Prioritized experiments

### A0: Build the truth set

Create exhaustive stratified tiles plus random tiles and detector-disagreement
tiles. Gate: sufficient coverage and adjudication to compute precision.

### A1: Re-evaluate frozen baselines

Compare raw-direct, fixed CFAR, v3 components, and imported proposal sources
under identical object/event matching. Gate: reproducible PR/FROC reports with
group intervals.

### A2: Hard-negative direct tuning

Keep direct amplitude primary; learn bounded spatial/temporal/contrast terms
against full-field quiet peaks. Gate: improve AP or a predeclared
precision-recall operating point on held-out groups, not training loss alone.

### A3: Decouple identity and event inference

Freeze reviewed footprints, materialize traces, and compare trace-event models.
Gate: improved event AP/timing without worsening persistent identity metrics.

### A4: Domain robustness and calibration

Evaluate fish/session/acquisition groups and fit held-out calibration.
Gate: no subgroup collapse hidden by aggregate performance.

### A5: Streaming benchmark

Benchmark the selected causal pipeline end to end using p50/p95/p99 latency,
memory, missed-frame behavior, and fallback state. Attach frame-rate-specific
budgets rather than assuming every dataset is 100 Hz.

## Required artifacts

Each detection experiment should write:

- resolved manifest and input checksums;
- annotation coverage manifest and split manifest;
- candidate/ground-truth match table;
- PR, FROC, calibration, and subgroup tables;
- object/event error examples;
- latency and resource report;
- model/drift diagnostics;
- concise gate decision with unavailable metrics explicitly marked;
- workbench review payload for false positives, misses, merges, and splits.

## Stop conditions

Stop a detector line when:

- it improves surrogate loss but not held-out PR/FROC decisions;
- precision cannot be computed from the available annotation coverage;
- gains disappear on held-out fish/session;
- performance depends mainly on motion or acquisition artifacts;
- runtime violates the intended frame budget without a safe fallback;
- the same representation is worse than raw/grid inputs for intent decoding.
