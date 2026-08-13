# Spon Ca Burst next research program and PC execution guide

Date: 2026-08-12
Status: planning guideline; no experiment or GPU run is authorized by this
document.

## Executive decision

The next research step should be a measurement program, not another broad
model sweep. The existing studies already span temporal ICA, deterministic
smoothing, latent dynamics, spatial information features, source separation,
denoising, ranking, and quantized pooling. They repeatedly identify useful
signals, but sparse-positive labels still cannot determine whether unmatched
candidates are neurons, artifacts, or background. That makes biological
precision the limiting measurement.

The recommended order is therefore:

1. finish the already-prepared, detector-blinded review of the 25 hard
   observations;
2. create a bounded, exhaustively reviewed truth set with a calibration region
   and a protected region;
3. use stored scores to adjudicate the low-budget disagreement between
   `fullrank_ica_w17` and `exponential_w5` before fitting anything new;
4. only then run one compact, unified confirmation panel under identical
   candidate, NMS, matching, and audit contracts;
5. branch to an interpretable time-constant model, a morphology-conditional
   ranker, or a causal state model only if the corresponding frozen lane earns
   that branch;
6. require an independent recording before a detector is promoted as a
   general method, then benchmark the causal implementation at the 20 ms frame
   period.

This order spends reviewer time before GPU time and converts the present
precision ambiguity into a measured endpoint. Completion of any computational
stage remains distinct from scientific success.

## Evidence that constrains the plan

The values below use different carriers and operating-point contracts. They
are decision landmarks, not a cross-study leaderboard.

| Result | Current evidence | Planning consequence |
| --- | --- | --- |
| Raw Direct | Exact historical threshold anchor: 49/79 known matches, macro recall 0.6056, 232 candidates | Retain as an immutable reference and fallback. |
| Short temporal ICA | `fullrank_ica_w17` recovered 48/79, 55/79, and 56/79 at 20, 40, and 60 candidates per burst | Primary compact temporal candidate at scarce review budgets. |
| Deterministic temporal integration | `exponential_w5` recovered 55/79 at budget 60 and 64/79 at budget 100 | Primary simple temporal comparator; likely practical choice if precision is tied. |
| Long temporal ICA | Full-rank ICA had 0/15 convergence for every 51--201 tap length; frozen `dctica_w201_r8` recovered 36/79 with 700 candidates | Do not extend unconstrained or DCT long-history sweeps. |
| Local spatial context | Cross-fitted `coherence_w15` recovered 48/79 at budget 20 versus 43/79 for its carrier, with improvement in all four bursts | Strongest compact spatial confirmation candidate. |
| Lagged recurrence | Standalone `propagation_lag2_w15` is promising, but cross-family selection was less stable | Keep as one prespecified secondary lane; do not call it causal propagation. |
| Latent dynamics | Offline smoother amplitude recovered 55/79 with 320 candidates and won 4/4 bursts; upstream preservation gates remain incomplete | Noncausal ceiling and mechanistic clue, not a real-time detector. |
| Multi-lag MSICA | Family-frozen full-embedding CS-Parzen and normalized-HSIC arms reached 54/79 at 58 candidates per burst | Provisional, multiplicity-exposed arms for independent-recording confirmation only. |
| Quantized pooling | Protected panels did not beat float GN; hard max/min pooling collapsed spatial identities | Retire hard pooling and do not widen quantization. |
| Broad denoising and learned ranking | No denoiser passed all preservation gates; bounded linear ranking tied a deeper MLP and lost at budgets 20 and 40 | Preserve the scientific carrier; do not widen MLP or denoiser grids. |

The most coherent working model is that short temporal integration, robust
per-pixel quiet normalization, and causal spatial-neighborhood context carry
most of the useful information. Long freely learned filters, deeper rankers,
and aggressive denoising have not supplied stable incremental evidence.

## Questions and estimands

Every proposed study must name its estimand before scores are opened.

1. **Known-positive sensitivity:** among the 79 current sparse-positive
   occurrences, how many are recovered at a fixed candidate budget? This is
   not precision.
2. **Exhaustive-region detection:** within a region where every visible object
   and event has been reviewed, what are object/event precision, recall,
   average precision, calibration, duplicate/split/merge rates, and false
   events per field-minute?
3. **Candidate efficiency:** at budgets 20 and 40 per burst, which frozen lane
   ranks true activity earliest? Budget 58 is secondary and budgets 80/100 are
   screening endpoints.
4. **Signal integrity:** does the representation preserve peak amplitude,
   area, onset/peak timing, morphology, and trace shape even if it improves a
   ranking score?
5. **Generalization:** does the exact frozen method retain its behavior on a
   new recording, fish, day, or acquisition condition?
6. **Causal feasibility:** can the promoted path process a frame with bounded
   memory and p50/p95/p99 latency compatible with a 20 ms frame period?

Candidate burden remains a selectivity-pressure proxy until Question 2 has a
truth set. Unlabeled candidates remain unknown, never automatic negatives.

## Global design rules

- UI frame intervals are one-based and inclusive. Array intervals are
  zero-based and half-open. Coordinates use `x=column`, `y=row`.
- Every label-driven preflight writes and reviews a projection overlay.
- Freeze lane definitions, temporal windows, normalization, thresholds,
  candidate budgets, NMS radius, matching radius, ranking ties, and analysis
  code before protected labels are opened.
- Preserve raw scientific traces separately from ranking/display scores.
- Use one-to-one object matching. Record nearest-candidate identity separately
  from the assigned one-to-one match.
- Group uncertainty by recurring ROI identity and burst. Adjacent frames are
  never independent samples.
- Keep exploratory/post-hoc ceilings visually and machine-readably separate
  from frozen or cross-fitted estimates.
- A new output root must be collision-free. A completed output root is never
  overwritten.
- Scientific-audit output is default-on. Numerical completion is not
  audit-completion.
- No stage in this plan authorizes a full Spon, CUDA, Stage 2 Hierarchical
  Parzen, grid128 Stage B, or other long run. Each requires explicit user
  selection at the time of launch.

## G0: finish the existing hard-observation gate

### Purpose

Resolve known identity, timing, morphology, and NMS ambiguities before using
the current labels as a fixed reference. The engineering path is already
complete at:

```text
Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_v1
Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_review_checklist_v2
```

### Human task

Finalize all 25 target observations covering ROI 007, 008, 010, 014, 015,
017, 019, 020, and 023. The reviewer must decide canonical identity, visible
activity, morphology, context, inclusion view, and a complete onset/peak/end
triple when timing is changed. All target rows must be explicitly
`review_status=adjudicated`, with reviewer and timestamp provenance.

### Gate and expected result

Only after all 25 rows are final should the exact CPU re-score run into a new
root. The likely outcome is a cleaner decomposition of identity conflict,
ranking miss, localization miss, temporal miss, NMS suppression, and proposal
miss. This targeted review cannot estimate precision because it is conditioned
on known hard cases.

## A0: build a bounded exhaustive truth set

### Primary hypothesis

Differences currently described as candidate burden contain a mixture of
unlabeled neurons, motion/illumination artifacts, persistent anatomy, vascular
or membrane-like structure, and noise. Exhaustive review will change the
relative ordering of at least some current detectors.

### Region selection

Use a two-part design inside the anatomically relevant right field:

1. **Calibration region:** a morphology-rich bounded tile may deliberately
   include recurring hard identities. Record this enrichment and never report
   it as an unbiased field estimate.
2. **Protected region:** select a non-overlapping tile with a deterministic
   seed from a label-free tissue/quality mask. Detector scores and known-label
   outcomes must be unavailable during selection.

Start with a small raw-only pilot to measure reviewer minutes per burst. Freeze
the final dimensions using only anatomy, visibility, and workload—not detector
success. A 96--128 pixel square per region is a reasonable planning envelope,
but the preflight overlay and review-time pilot determine the exact coordinates.
If one protected tile is too small for stable uncertainty, add spatially
stratified protected tiles rather than expanding a score-selected region.

### Blinded two-pass review

1. **Raw-first discovery:** review synchronized raw, fixed positive-change,
   and trace views with neutral region markers and no detector identity,
   candidate color, score, or recovered/missed status. Mark every visible
   object/event.
2. **Candidate-assisted completeness:** reveal a randomized, detector-blinded
   union of frozen candidates to catch subtle omissions. Reviewers still do
   not see which lane proposed a candidate.
3. **Disagreement resolution:** re-review unresolved items and at least 20% of
   accepted/rejected items after a washout period or with a second reviewer.

### Annotation schema

Record a persistent object identity/footprint and an event row separately.
Required fields include neuron, artifact, background, or unresolved
disposition; center/membrane/other morphology; isolated/crowded context;
visibility confidence; onset/peak/end; motion/illumination/vascular flags;
reviewer identity and time; raw-source provenance; and explicit coverage masks.
Do not force unresolved examples into binary negatives.

### Splits and metrics

- Calibration data may support threshold and small-model development.
- The protected region remains untouched until all lane definitions are frozen.
- Report object- and event-level PR curves, AP, precision at fixed recall,
  recall at fixed precision, false events per field-minute, centroid/footprint
  error, merge/split/duplicate rates, and calibration/abstention coverage.
- Bootstrap recurring object identities and bursts as groups. Report interval
  widths and raw paired counts; four bursts cannot support a strong
  asymptotic-independence claim.

### Gate

A0 passes only when coverage is explicit, every candidate in the union has a
disposition, raw-first discovery is complete, unresolved cases are retained,
and held-out protected metrics can be computed. Until then, no lane may claim
ordinary precision.

## E1: stored temporal disagreement adjudication

### Why this comes before a new run

The completed long-history root already contains the scores needed to compare
`fullrank_ica_w17` and `exponential_w5`. Their key difference is review-budget
behavior, so fitting more models before reviewing their unique candidates
would add search multiplicity without resolving the decision.

### Frozen panel

- `fullrank_ica_w17`;
- `exponential_w5`;
- Raw Direct reference where coordinates can be reconstructed under the same
  geometry;
- the common candidate set, each lane's unique candidates, and a raw-discovery
  sample independent of either lane.

Use matched budgets 20, 40, 60, and 100 per burst for exact reproduction;
declare 20 and 40 primary. Randomize candidate order and blind lane identity.
No refitting or label-informed threshold changes are allowed.

### Primary endpoint and decision

On the exhaustive regions, compare AP and precision at matched recall. On the
legacy sparse-positive set, report paired known-positive recovery at fixed
budget only. A suggested preregistration target is an absolute AP gain of at
least 0.03 plus an absolute precision-at-matched-recall gain of at least 0.05,
with no material loss in three or more bursts. These values are planning
thresholds and must be frozen before protected review; effect intervals and
raw disagreements remain authoritative.

- If 17-tap ICA clears the target, retain it for a compact interpretable-
  temporal-model branch.
- If the lanes are practically equivalent, prefer the deterministic
  exponential because it is simpler, fully converged, and easier to stream.
- If unique candidates from either lane are mostly real unlabeled neurons,
  expand the truth set rather than calling the other lane more precise.
- If artifacts dominate both unique sets, shift effort to artifact and
  acquisition covariates rather than filter complexity.

## E2: unified compact confirmation

### Question

Under one exact carrier, normalization, temporal pooling, NMS, matching, and
candidate-budget contract, which small causal representation family gives the
best precision-aware ranking without damaging the scientific signal?

### Prespecified causal panel

1. Raw Direct.
2. The frozen quiet-standardized carrier.
3. `fullrank_ica_w17` plus positive per-pixel quiet normalization.
4. `exponential_w5` plus the identical normalization.
5. `coherence_w15` under its frozen carrier semantics.
6. Standalone `propagation_lag2_w15`, described as recurrence rather than
   biological propagation.
7. Causal latent-filter amplitude under the existing frozen state model.

The offline RTS smoother is reported only as a noncausal ceiling. Float GN may
remain the standard visualization transform, but the post-hoc alpha/top-5 lane
does not enter primary detector selection.

### Factorial mechanism check

If E1 leaves a meaningful temporal question, add a small prespecified
operator-by-normalization ablation:

- operator: identity, five-frame boxcar, five-frame exponential, frozen
  17-tap ICA;
- normalization: global robust scale or per-pixel quiet robust scale;
- sign handling for the two finalists: positive-only or two-sided diagnostic.

This 4-by-2 core isolates the value of the temporal operator from the value of
quiet normalization. It is not a new kernel search. Use exactly the same
candidate extraction for all lanes.

### Evaluation and gate

Use budgets 20, 40, 58, 80, and 100, with 20 and 40 primary. Test the
predeclared comparisons hierarchically: temporal ICA versus exponential first;
the surviving temporal lane versus the carrier second; then coherence and
recurrence additions. Report exhaustive-region AP/precision first and legacy
known-positive recall second.

Advancement requires all of the following:

- a practically meaningful protected-region precision/AP improvement, not
  only fewer candidates;
- at least four additional known matches across the 79 occurrences at budget
  20 or 40, with non-worse behavior in at least three bursts, unless the
  exhaustive endpoint provides a stronger predeclared decision;
- preserved peak and temporal area at the frozen tolerance, and median peak
  timing error no greater than one frame;
- stable results over ROI-grouped resampling and NMS/match-radius sensitivity;
- a complete, passed three-section scientific audit.

If no lane passes, preserve Raw Direct and the quiet-standardized carrier and
stop model expansion. A null compact benchmark is a useful result.

## Conditional E3 branches

Only one branch should run first, chosen by E1/E2 evidence.

### E3-T: interpretable time-constant bank

Run only if a temporal integrator wins. Replace one free coefficient per frame
with a small nonnegative bank of fixed causal exponentials spanning short to
long decay scales. Freeze a compact grid such as 40, 80, 160, 320, 640, and
1280 ms; compare individual filters and a simplex-constrained mixture. Include
`exponential_w5`, boxcar support controls, frozen 17-tap ICA, causal latent
amplitude, and the offline smoother ceiling.

The hypothesis is that two or three stable time constants can match the useful
calcium persistence while avoiding long-ICA non-identifiability. Reject the
branch if selected weights vary strongly across temporal folds, if gains occur
only at budget 80/100, or if timing shifts exceed the signal guard.

### E3-S: morphology-conditional monotone ranker

Run only after A0 supplies reviewed negatives and morphology. Retain the
standardized carrier as an immutable skip and use a very small feature set:
coherence, recurrence, center/annulus response, cross-scale agreement, crowd
context, and an artifact/motion flag. Compare a bounded linear model, a
monotone additive model, and at most one two-expert center-versus-membrane gate.

Train on the calibration region with burst and object-identity grouping; open
the protected region once. Optimize AP and top-of-list performance at budgets
20/40. Stop if the model merely improves budget 58, if the gate tracks
acquisition field rather than morphology, or if a single frozen feature is
equivalent.

### E3-D: causal rise/decay state model

Run only if the offline smoother remains a meaningful ceiling and the causal
filter approaches it. Test a compact rise/decay or innovation-gated state
model against the existing scalar AR(1), not a broad Kalman grid. Preserve the
raw/quiet-standardized trace as a skip. Require perturbation stability,
amplitude/area preservation, onset and peak error within one frame, and a
causal latency profile. The smoother can motivate the model but cannot be
promoted to real-time use.

## E4: independent-recording confirmation

This is the first stage that can support a general detector claim. Acquire or
identify at least one recording that was not used for fitting, family
selection, thresholding, reviewer training, or manuscript figure choice.
Before opening its labels, freeze:

- the exact finalist and two baselines;
- preprocessing, field boundaries, quiet-window rules, candidate budgets,
  NMS/matching, calibration, and abstention;
- acquisition subgroup checks and failure categories;
- the complete audit manifest.

Primary splitting should be by fish; session/day is the fallback. A held-out
interval from this same recording remains useful but is not an independent-
recording confirmation. A result that collapses under field, brightness,
motion, or morphology strata returns to acquisition/representation study, not
hyperparameter widening.

## E5: causal streaming checkpoint

After scientific promotion, implement a bounded frame-by-frame path and
measure p50/p95/p99 end-to-end latency, peak RSS/VRAM, missed-frame behavior,
warm-up, checkpoint recovery, and fallback output. The target is p99 below the
20 ms frame period, with an internal p95 target below 15 ms to retain headroom.
If the selected method cannot meet this on the intended machine, retain an
offline mode and a simpler causal fallback rather than weakening safety guards.

## PC-safe execution schedule

### Hardware snapshot used for planning

Read-only telemetry on 2026-08-12 showed an Intel i9-14900K (24 physical cores,
32 logical CPUs), 78 GiB RAM with 58 GiB available, 8 GiB unused swap, an RTX
4070 SUPER with 12,282 MiB VRAM, and about 1.8 TiB free disk. The GPU was not
idle (about 31% utilization and 2.2 GiB allocated), and a Gradle/Android process
held about 7 GiB RSS. This snapshot demonstrates capacity, not readiness; every
launch must repeat telemetry and should wait for an idle interactive window.

Measured planning anchors are:

- long-history numerical detection: 57 lanes in 675.9 s, 537 MiB final RSS,
  with a 4 GiB preflight estimate;
- scientific feature audit computation: 232.5 s and 4.4 GiB peak RSS;
- completed audited roots: roughly 0.8 GiB each;
- source TIFF and memory-mapped cache: roughly 0.86 GiB each.

Future durations are unknown until smoke calibration. Media rendering can
dominate computation because a complete audit may contain hundreds of videos.

### Serial queue

| Queue item | Dependency | Planning envelope | Default resource class | Completion marker |
| --- | --- | --- | --- | --- |
| Q0 freeze manifests, hashes, and analysis plan | none | 15--30 min | metadata only | reviewed pre-registration and new roots |
| Q1 finalize 25 hard observations | human availability | multiple short review sessions | no compute | every target row adjudicated |
| Q2 exhaustive-region pilot and frozen coverage | Q1 schema review | one short review/render session | 2 CPU threads, one encoder | coverage and reviewer-time report |
| Q3 assemble stored E1 candidate panel | A0 coordinates frozen | 15--45 min estimate | 2 CPU threads, low priority | atomic panel/index plus smoke validation |
| Q4 human E1/A0 review | Q3 | reviewer-limited | no concurrent heavy run | adjudicated candidate union and raw scan |
| Q5 exact metrics and compact E2 smoke | Q4 | 15--30 min estimate | 2 CPU threads; one bounded GPU process only if required | smoke gate and calibrated ETA |
| Q6 E2 numerical confirmation | Q5 passes and user explicitly selects | target chunks below 30 min; total target below 60 min before audit | serial CPU/GPU chunks | all burst/fold checkpoints complete |
| Q7 scientific-audit media | Q6 numerical gate passes | reserve 1--3 h and at least 5 GiB until smoke refines it | one encoder, no simultaneous fitting | inventory and media validation pass |
| Q8 conditional E3 branch | E2 selects exactly one branch | separately estimated after tiny smoke | same serial policy | branch-specific stop/advance gate |
| Q9 independent recording and streaming | data and scientific gate | separate campaign | new root and fresh preflight | independent audit plus latency report |

Do not overlap numerical fitting, audit encoding, another NeuroBench run, an
Android/Gradle build, or a heavy interactive GPU workload. Human review and
read-only report inspection may occur while the PC is otherwise idle, but the
serial queue remains the authority.

### Default resource classes

**CPU-bounded work**

- `.venv-neurobench/bin/python` only;
- two numerical-library threads, one process, low OS priority;
- chunk by burst, fold, and lane; checkpoint after each unit;
- warning at 12 GiB process RSS and stop at 20 GiB unless a reviewed manifest
  sets a lower bound;
- no multiprocessing fan-out.

**GPU-bounded work**

- one CUDA worker and at most two CPU numerical threads;
- default 4 GiB CUDA allocation cap and at most eight-frame projection chunks;
- CPU-backed output maps and atomic per-lane completion records;
- no concurrent encoder or other experiment; retry only after diagnosis, never
  by widening resource limits automatically.

**Audit media**

- one encoder;
- full-field video first, then independently resumable close-ups and traces;
- decode and annotation-color validation in bounded batches;
- retain partial metadata and resume; do not overwrite the numerical root.

### Mandatory launch preflight

Before every nontrivial run:

1. validate input path, SHA-256, array shape, frame/coordinate contract, label
   projection, manifest schema, and exact combination count;
2. prove the selected output root and every stage subroot do not exist;
3. record git SHA plus dirty-state summary without modifying user files;
4. confirm no active NeuroBench/Python experiment, renderer, or competing GPU
   compute process;
5. require at least 32 GiB host memory available, swap use below 0.5 GiB, and
   at least 100 GiB disk free plus four times the estimated artifact size;
6. for GPU work, require at least 8 GiB free VRAM and sustained low utilization
   before launch;
7. run a read-only preflight, then a tiny smoke that exercises one chunk and
   writes to a new disposable experiment root;
8. extrapolate time, RSS, VRAM, and disk from the smoke and re-review the queue
   before the full command;
9. start 30--60 s atomic heartbeats recording phase, lane, chunk, elapsed time,
   RSS, VRAM, and last successful checkpoint.

Stop the run if available RAM falls below 24 GiB, swap exceeds 1 GiB, GPU
temperature remains at or above 80 C for 60 s, an NVIDIA Xid/OOM appears, the
heartbeat stalls for five minutes, an input digest changes, an output collision
is detected, or desktop responsiveness degrades. A stopped run must remain
resumable and explicitly marked incomplete.

## Expected outcomes and decision responses

| Observation | Interpretation | Next action |
| --- | --- | --- |
| Exponential and 17-tap ICA have equivalent protected precision | ICA is not adding decision value beyond integration | Promote the deterministic exponential as temporal baseline; retire further ICA tuning. |
| 17-tap ICA has a reproducible low-budget precision edge | Its uneven compact kernel may encode useful temporal shape | Run E3-T only; test a compact time-constant basis against the frozen kernel. |
| `coherence_w15` wins at budget 20/40 and on exhaustive AP | Spatial neighborhood consistency adds information | Run E3-S after morphology labels; retain temporal lane as control. |
| Offline smoother wins but causal filter does not | Future-context denoising is useful but not deployable | Run the bounded causal E3-D model; keep smoother as ceiling. |
| Candidate-count gains disappear under exhaustive review | Surrogate selectivity was misleading | Freeze the simplest baseline and improve annotation/acquisition modeling. |
| Many unmatched candidates are accepted neurons | Existing labels are incomplete rather than detector-specific false positives | Version the expanded truth set and recompute all frozen baselines without calling earlier work wrong. |
| Artifacts dominate finalist-only candidates | Representation is tracking nuisance structure | Add motion/illumination/field covariates before any new ranker. |
| No compact lane passes | The null is scientifically useful | Stop local model search and prioritize a new recording plus acquisition controls. |

The expected—not guaranteed—outcome is that deterministic short integration
will explain much of the temporal ICA gain, while coherence or recurrence will
help at tight budgets. The exhaustive panel may also reveal real previously
unlabeled activity. The plan is deliberately valuable under all three results.

## Research lines to pause

Do not spend the next campaign on:

- full-rank temporal ICA beyond 33 taps or another 51--201 tap DCT-ICA sweep;
- promotion of unstable `K=16` temporal ICA or the label-assisted 60/79 MSICA
  ceiling;
- `exponential_w101` as a precision detector; it is only a review-expensive
  high-sensitivity screen;
- hard max/min coactivity pooling, a wider quantization grid, or another GN
  temperature search;
- wider residual-MLP, global-fusion, or denoiser grids;
- Hierarchical Parzen Stage 2 before the failed Stage-1 validity gate is fixed;
- the stopped grid128 Stage A sweep or its Stage B manifest without explicit
  user selection.

## Longer-horizon research after the detection gate

1. **Persistent identities and trace-first event inference.** Move from
   framewise peaks to reviewed footprints, persistent neuron identities,
   amplitude-preserving traces, event probabilities, and calibrated
   abstention.
2. **Acquisition and nuisance physics.** Model field boundaries, motion,
   saturation, illumination drift, z-plane morphology, and intensity-dependent
   noise as explicit covariates. Do not let detector score serve as a proxy for
   these quantities.
3. **Cross-fish and cross-session calibration.** Collect enough independent
   groups to report subgroup performance and domain drift rather than relying
   on four bursts from one recording.
4. **Uncertainty-aware active review.** Once a protected truth set exists, use
   model disagreement to prioritize additional review while retaining random
   tiles for unbiased evaluation.
5. **Genuine source-separation identifiability.** Revisit physical source
   separation only when multiple measured channels, views, or perturbations
   provide independent information. Temporal lags from one camera remain
   pseudo-channels.
6. **Intent and inverse control.** Begin left/right intent only after activation
   detection has a precision-audited, stable identity output. Causal action
   effects and control additionally require measured actions, safety
   interlocks, and shadow-mode evidence; passive activity is not an invertible
   controller.

## Required artifact contract for every future stage

Each experiment receives a new root and writes, atomically where practical:

- resolved manifest, input hashes, code/git provenance, and preflight overlay;
- exact candidate/fit count and label-access declaration;
- progress heartbeat, resume state, resource time series, and stop reason;
- frozen lane/config record and explicit post-hoc ceiling record;
- candidate and match tables with stable object/event identifiers;
- PR/FROC/calibration/subgroup tables when exhaustive labels exist;
- amplitude, area, timing, morphology, and NMS sensitivity diagnostics;
- `llm_context.json`, summary, artifact index, validation, and concise report;
- Expert-only and Model-only full-field videos, close-ups, and traces, plus the
  figure/table-only matched Comparison section required by the scientific-audit
  standard;
- a report-pass decision that says `advance`, `stop`, or `incomplete` and why.

The living Overleaf manuscript is
`docs/research/overleaf/quiet_normalized_multilag_temporal_ica/main.tex`. Its
research-program section mirrors this guideline while keeping the paper's
validated results separate from proposed work.
