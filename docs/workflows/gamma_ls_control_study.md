# Gamma-LS preprocessing and measurement study for causal fish control

This workflow specifies the user-authorized development replay begun on
2026-09-12. Its question is which preprocessing and readout components retain
useful neural-activity evidence at an acceptable computational and temporal
cost when settings can be configured before a control session. It evaluates
the measurement layer; it does not execute a controller or stimulation.

The source recording, labels, and earlier results have already been inspected.
The matrix is fixed for this execution, but it is not a preregistered study on
unseen data or independent confirmation. Report the complete matrix without
choosing a winning operator after inspecting its outcomes. The resolved,
hash-bound run manifest is the authority for exact numerical settings and arm
identifiers. This document states the comparison and interpretation contract.

## Questions and estimands

Keep the following outputs separate:

| Output | Question answered | What it does not identify |
| --- | --- | --- |
| Framewise spatial proposals | How many known labeled occurrence windows receive an assigned spatial representative, at a specified calibration operating point and measured application burden? | Biological precision, exhaustive recall, unique neurons/events, or onset accuracy |
| Configured-ROI activity | How do raw and processed signals behave at specified coordinates, and which known occurrence windows contain threshold activity there? | Automatic ROI discovery, a complete neuron footprint, or unbiased generalization to unknown ROIs |
| Retrospective historical bridge | How do changed history, peak selection, and burst aggregation affect older numerical readouts? | An online decision stream or an interchangeable version of the primary metric |
| Timing and causality | Which samples are available to each output, and how much computation or waiting does production require? | Successful closed-loop control or a verified neural-event-to-action delay |

The primary component contrasts are Gaussian smoothing present/absent, causal
EMA present/absent, current-frame versus signed adjacent difference, and
amplitude versus locally standardized evidence. Additional fixed controls
separate local subtraction, target pooling, reference weighting/support, a
historical slow background, and motion/global-offset conditioning. Compare
paired outcomes and actual emitted rows across the entire operating curve;
equal calibration targets do not guarantee equal application burden.

## Chronological replay and source coordinates

The measured recording was acquired at 50 Hz, with a 20 ms frame period. The
primary replay reads source UI frames 1800 through 2359 inclusive: 560 input
frames. Every recomputed arm starts at UI1800. No state is silently borrowed
from an earlier full-recording cache.

| Interval | Source UI frames, inclusive | Role |
| --- | --- | --- |
| Initialization input | 1800 | Common initial state; adjacent difference is ineligible |
| Calibration input | 1800..1899 | 100 inputs available before application |
| Eligible calibration scores | 1801..1899 | 99 scores for every primary arm, including current-frame arms |
| Application | 1900..2359 | 460 score frames, using only the current and preceding available inputs |

UI coordinates are one-based and inclusive. NumPy coordinates are zero-based
and half-open; spatial coordinates are `x=column`, `y=row`. A replay array's row
zero maps to source UI1800, not UI1. Store the exact `source_frames_ui` map;
use it in candidates, traces, annotation lookup, plots, and media. Distinguish
time relative to replay initialization from absolute recording time.

All parameters fitted from data use only the calibration interval. This covers
reference templates, numerical standard-deviation floors, empirical score
thresholds, and any quiet-reference statistics. Calling that interval a
calibration or quiet-reference interval does not certify it as biologically
event-free. The application labels cannot set a threshold or choose an arm.
Calibration outputs are setup diagnostics, not decisions asserted to have
been available before the setup batch ended. In particular, the historical
background initialization uses the median of the complete setup batch; its
early setup residuals therefore use later setup frames. The future-free
application contract begins at UI1900. Common replay origin does not require
different models to have identical initialization formulas.
Seal stages, calibration records, and candidate tables before the outcome join.
Since earlier labels/results were viewed, this chronology prevents future-data
use within the replay but does not make the study independent confirmation.

## Fixed score matrix

Let `R_t` be the input frame, `S_t` its optional Gaussian-smoothed version,
and `E_t = alpha*S_t + (1-alpha)*E_(t-1)` the causal EMA, initialized from
the first replay frame. With `alpha=1`, `E_t=S_t`. The two input representations
are current `X_t=E_t` and signed difference `X_t=E_t-E_(t-1)`; preserve the
sign of differences. Spatial Gaussian sigma is in pixels and temporal alpha
is dimensionless. Record padding and initialization explicitly.

For a local operator, `A` is its target response, `M` its reference mean,
`C=A-M` its contrast, and `Z=C/(max(sigma_ref, floor)+epsilon)` its standardized
readout. Here `sigma_ref` is local spatial variability, not the upstream
Gaussian smoothing sigma, a standard error, or a calibrated null probability.
With a point target, `A=X`. Every score has its own calibration-derived
threshold in its own units.

| Score family | Fixed comparison | Number of scores |
| --- | --- | ---: |
| Main upstream factorial | Spatial sigma 0/1 × EMA alpha 1/0.4 × current/difference × A/Z; point target, Gamma reference, guard 0, square support | 16 |
| Contrast controls | C for standard current and difference inputs | 2 |
| Historical residual controls | Native slow-background residual and its quiet per-pixel MAD-standardized carrier | 2 |
| Centered Gamma target controls | Direct A/C/Z and serial Z on standard signed input at the selected fixed target/reference construction | 4 |
| Reference-weight control | Uniform-reference Z | 1 |
| Deployed spatial geometry | Point-target Gamma-reference Z with guard 7 and disk support, recomputed from the common replay origin | 1 |
| Input nuisance controls | Motion-only and robust-global-offset-only correction, each before Gaussian sigma 1/EMA alpha 0.4/signed difference/point-Gamma guard-zero square Z | 2 |
| **Total** | **Each score evaluated at five calibration operating points** | **28** |

The standard upstream setting is spatial sigma 1 and EMA alpha 0.4. The main
factorial uses point-target Gamma guard-zero square support; it is distinct
from the deployed guard-seven disk control. The manifest must specify the
representation and readout of each additional control, discrete target and
reference weights, finite support, boundary normalization, floor rule, and
epsilon. Keep selected centered-Gamma settings fixed across this study rather
than searching their shape or support from the application outcomes.

The motion template is fitted from calibration frames only. Application
registration uses that template and the current frame, without a future-frame
average. Record estimated shifts, interpolation, edge treatment, and invalid
support. The global-offset control estimates a robust median from the current
frame; subtraction must not use a centered temporal window. These controls
test whether nuisance correction changes measurement behavior. A score change
alone does not prove that motion or illumination caused the original event.

The two historical scores retain the original quiet-only fit/settings. The
slow background has a 10-second reference half-life and innovation fraction
0.1 clipped at four quiet MADs. Its initial background is the median of the
100 setup frames. The standardized carrier uses calibration-derived per-pixel
residual median/MAD with a positive-MAD tenth-percentile floor. Preserve its
historical float16 carrier save and the float32 native residual as distinct
numeric representations, along with calibration arrays and metadata hashes.
The first residual corresponds to the first input at UI1800, without a
one-frame shift. These two controls do not establish the necessity of learned
preprocessing beyond the models actually evaluated.

## Online operating points and matching

All primary spatial-proposal arms use the same maintained local-maximum
prefilter, border exclusion, deterministic score/coordinate ordering, and
greedy Euclidean NMS rule with 6-pixel separation. Freeze this implementation
and its exact tie handling in the code seal. Changing NMS only in one arm would
confound the upstream component comparison.

For each score, fit thresholds at `q = 0.25, 0.5, 1, 2, 5` proposals per eligible
calibration frame. The 99-frame integer budgets are respectively 24, 49, 99,
198, and 495. Use exact NMS peak order statistics with a strict threshold
comparison; choose the densest attainable count at or below the budget.
Equal-score ties can leave a budget underfilled. Record target, attained
calibration count, threshold, ties, and any prefix-saturation proof. A bounded
prefix cannot silently stand in for an exact calibration count.

Apply frozen thresholds to all 460 application frames and retain every
emitted frame/coordinate/score row. Report count, candidates per application
frame, and distribution across frames. These are proposal-burden measures;
they are not biological false-alarm rates. The nominal `q` unit differs from
the older per-block calibration target.

Preserve the 79 known occurrence rows and the 26 canonical identity inventory.
Per-occurrence coordinate variants remain distinct. Within each declared
inclusive burst window, form spatial review representatives using the shared
deterministic radius-6 rule: sort actual proposals by decreasing score, then
stable coordinate/frame/proposal-ID ties; assign each row to the first accepted
representative within 6 pixels, otherwise accept that row as a new
representative. This grouping is nontransitive. Assign score-ranked
representatives to the nearest still-unassigned expert occurrence within
6 pixels, breaking distance ties by observation ID.

Report known-positive occurrence-window coverage, frame-proposal count, and
spatial-representative count separately. Representatives are review locations,
not inferred persistent neurons or unique biological events. An unmatched
known occurrence is a miss at this operating point. An unmatched candidate is
unknown. Broad burst windows do not identify the true onset of an event.

## Configured-ROI branch

The configured-ROI branch uses only the geometry of existing labeled centers
to specify extraction locations. Its setup is retrospective and label-informed
in that limited sense. It is not a test of automatic setup or discovery, and
its performance must be reported separately from framewise candidate search.

Retain every identity and occurrence coordinate variant. Where one fixed trace
anchor per identity is needed, use a deterministic actual occurrence coordinate,
record its source observation ID, and preserve the other variants; never
replace them with an averaged coordinate. Occurrence-level comparisons use
the occurrence's own coordinate.

Export exact-pixel raw and processed numerical traces with all 560 samples,
source frame IDs, signal units, calibration/application flags, and threshold
activity. Freeze the configured-ROI threshold rule from calibration only and
name its denominator separately from spatial NMS proposals per frame. Report
occurrence-window activity coverage and application threshold activity burden.
A binary threshold crossing is an activity readout, not a validated spike or
onset. Do not turn unannotated time points at these coordinates into confirmed
negative biological labels.

## Historical bridge

The cached canonical full-history stream is retained as a separately identified
provenance comparator. Its earlier state history makes it ineligible to serve
as a recomputed common-initialization arm. Bind its original code, calibration,
source map, stage hashes, and candidate definition; report differences without
rewriting its frozen result.

The bridge additionally contrasts the historical local-maximum selection with
the maintained Euclidean NMS and reproduces complete-known-burst LME/top-58
readouts. It uses the existing `_pool_values` transformation, including its
additional quiet-median and global-percentile normalization, and evaluates
both peak-selection policies at top 58. Record the normalization domain and
the exact pooled quantity alongside each bridge result. These are not simply
the primary scores with a different threshold. Those readouts aggregate over
a declared burst and may use frames
that are future data relative to an early decision. They are retrospective
diagnostics, even if their component frame scores were computed causally.
Keep aggregation, candidate unit, budget, and matching convention explicit.
Never pool their counts with framewise online rows or configured-ROI activity.

## Timing and future control interpretation

Report three distinct delays:

1. **Compute latency:** elapsed processing and transfer time after an input
   frame becomes available, including any queueing. Record hardware, workload,
   warmup, synchronization, p50/p95/p99, maxima, and missed deadlines when
   timing is measured; mean throughput cannot replace these quantities.
2. **Algorithmic waiting:** additional samples required before an output is
   final, including buffering, centered filters, inference lookahead, and any
   revision of previously reported events. Primary application updates use no
   future frames. A causal EMA can still reshape and delay a response without
   explicitly waiting for a future frame.
3. **Event-information delay:** time until the indicator and noisy observations
   provide sufficient evidence for a reliable biological decision. This needs
   suitable onset truth and an explicit decision policy; it is not measured
   by the current burst-window labels or by correlation-based best lag.

This is a replay of an acquired 50-Hz recording. Replaying its frames faster
does not create 1-kHz biological observations or validate a 1-kHz feedback
loop. Future control validation must measure acquisition-to-actuation timing,
tail latency/backlog under paced input, and performance with synchronized
behavior and delivered-action logs. The existing
[fish inverse-control program](../programs/fish_inverse_control/README.md)
keeps measurement, intent, system identification, and action execution as
separate gates.

Offline initialization followed by online extraction is an established design:
FIOLA estimates spatial footprints and parameters on an initial batch, then
extracts activity from incoming frames. Its authors' primary preprint describes
fixed spatial footprints for that fast extraction regime. This motivates the
configured-ROI baseline; it does not make source discovery unnecessary for
every task. [FIOLA paper](https://www.nature.com/articles/s41592-023-01964-2),
[authors' methods preprint](https://assets-eu.researchsquare.com/files/rs-800247/v1_covered.pdf).

Online motion correction is feasible in established pipelines, but its value
here is an empirical component question.
[NoRMCorre](https://doi.org/10.1016/j.jneumeth.2017.07.031).
Online deconvolution can explicitly trade future-sample lag for estimation
quality; fast processing alone is insufficient evidence of immediate event
decisions. [OASIS, limited-lag analysis](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423).

## Artifact and audit contract

The [Scientific Audit Output Standard](SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md)
is enabled, with no user opt-out. All 28 fixed `q=1` score states receive the
complete applicable media audit. Other `q` values retain all numeric results,
calibration records, and candidate rows and are explicitly outside the `q=1`
media scope. Selection of the media operating point is fixed across arms.

Each audited state contains expert-only and model-only full-field videos;
close-ups and exact-pixel full-duration traces for every expert identity and
every consolidated model review location; and figure/table comparisons for
every labeled occurrence. There is no outcome-based ROI cap. Expert green,
model orange, and pale-yellow match links remain exclusive annotation colors
on grayscale scientific backgrounds. Preserve the nearest-candidate comparison
separately from the one-to-one assignment.

Use three or four actual stage labels, including Raw, Input, and Score. Do not
call amplitude, contrast, or a historical residual Z. All arrays supplied to
the audit are TYX with rows mapped to exactly UI1800..2359. Display scales are
fixed and shared across scientifically comparable stages/arms and bound in the
audit configuration. Full-field videos use every fifth replay frame plus the
last at 10 fps; source frames and times remain explicit. Close-ups retain every
relevant occurrence/proposal frame plus five frames of context, clipped to
available input. Traces retain all 560 samples. Occurrence comparison plots
show the declared burst plus context; descriptive correlation and lag use
the exact burst only.

The scientific video master uses the existing lossless RGB composite encoder:
decoded source-composite pixels must agree exactly, in addition to section-color,
per-frame marker, count, duration, dimensions, and full-decode validation.
PNG previews remain available for players lacking the required H.264 profile.
RGB losslessness preserves the displayed composite, not the floating-point
array values before grayscale mapping, clipping, or spatial fitting. Numeric
traces and hash-bound arrays remain the numerical evidence.

The small indexes must expose configuration, stage order, source map, score
units, calibration/application intervals, operating point, expected/observed
counts, coordinate variants, match semantics, display scales, source hashes,
output hashes, and limitations. Bind every displayed stage and candidate file
to expected SHA-256 values; also bind all five numerical operating-point
tables and the expert occurrence rows or source file. Completion and resume
must verify those bindings and the full required artifact inventory. A failed
render leaves numeric results preserved and scientific audit explicitly
incomplete.

## Execution and success gates

Use the repository virtual environment, one bounded workload at a time,
explicit thread limits, and a new non-colliding output root. Before a long run,
check active CPU/GPU workloads, disk/RAM headroom, input hashes, source shape,
annotation geometry, and the expected 28-score/140-operating-point inventory.
Write the geometry-only projection overlay during preflight. Preserve prior
complete outputs and record technical amendments in a new revision when a
bound contract changes.

The implementation is divided among `control_study.py` (primary replay and
numeric study), `control_history.py` (history and retrospective bridge), and
`control_audit.py` (sealed-state media audit), in
`neurobench/experiments/gamma_ls_difference/`. The audit accepts a JSON
configuration and supports a read-only inventory preflight. The primary
runner provides `preflight`, `run`, `evaluate`, `audit`, and `report` stages.
The current execution root is
`Outputs/GammaLSControl/control_study_20260912_r1`; use a fresh root for a
different bound protocol. Run stages sequentially:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.control_study preflight --output Outputs/GammaLSControl/control_study_20260912_r1

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.control_study run --output Outputs/GammaLSControl/control_study_20260912_r1 --device cuda --chunk-frames 8

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.control_study evaluate --output Outputs/GammaLSControl/control_study_20260912_r1

env MPLCONFIGDIR=/tmp/neurobench-mpl-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.control_study audit --output Outputs/GammaLSControl/control_study_20260912_r1

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 31 ionice -c 3 nice -n 19 .venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.control_study report --output Outputs/GammaLSControl/control_study_20260912_r1
```

The host GPU is required for the CUDA command; a sandbox without GPU access
cannot validate that path. Use the host permission mechanism when necessary
and record the actual environment and command. The historical helper is
called by the primary runner and has no independent CLI. To inspect one
sealed audit configuration without rendering:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.gamma_ls_difference.control_audit --config PATH_TO_SEALED_AUDIT_CONFIG.json --preflight
```

Remove `--preflight` only for a scheduled serial renderer after numerical
sealing. Record actual commands in the run manifest; this workflow does not
infer success from a command's exit alone.

| Gate | Required evidence |
| --- | --- |
| Contract and causality | Exact frame maps, common replay origin, calibration-only fits, deterministic selection, and tests showing future application-input changes cannot alter earlier application outputs |
| Numerical completion | All 28 scores and 140 operating points, exact calibration counts, complete candidate rows, configured-ROI traces, historical bridge, and source/code/output seals |
| Scientific audit completion | All 28 q1 inventories, every applicable expert/model ROI and occurrence, media/trace/geometry validation, and matching small indexes |
| Development interpretation | Complete paired contrasts and application burdens, explicit unknowns and failures, no hand-selected winning arm or unmeasured latency/precision claim |
| Independent confirmation | New recording-level units, frozen setup and practical margins, synchronized onset/behavior truth where needed, and sufficient adjudication for the intended metric |

Passing execution and audit gates means the comparison is reviewable; it does
not establish that a component is necessary or that a method is superior.
Report a component as a candidate for further study only in relation to its
paired measurement benefit, computational cost, and supported endpoint. Any
practical benefit margin or selected deployment configuration needs an explicit
decision before independent evaluation.

Further biological event/onset claims require onset annotations or appropriate
simultaneous reference measurements. Precision requires an exhaustively
adjudicated bounded space-time set rather than relabeling unmatched proposals
as negative. Generalization needs multiple new acquisition/recording units,
with repeated occurrences from one recording kept dependent. Intent and
closed-loop action benefits require their own aligned data and experiments.
