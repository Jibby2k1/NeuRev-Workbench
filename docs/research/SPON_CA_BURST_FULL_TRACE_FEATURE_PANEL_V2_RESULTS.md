# Spon Ca Burst full-trace feature panel v2 results

## Answer first

The old and new analyses answer different questions and together support a
compact, role-specific feature stack rather than one universal score.

- At a known confirmed ROI center, event amplitude is already almost sufficient
  to locate the annotated burst within the 560-frame trace. The frozen carrier
  reached a site-weighted mean event-localization percentile of 0.994 (95%
  site-bootstrap interval 0.981--1.000), and 104 of 106 occurrence scores were
  exactly 1.0.
- Local coherence and lag-2 recurrence do not demonstrate a meaningful
  incremental gain in this ceiling-limited known-center task. Their means were
  0.994 and 0.996; the lag-2 paired difference from the carrier was +0.002
  (95% interval 0.000--0.006), which fails the prespecified strictly-positive
  interval gate.
- The earlier protected full-field audit nevertheless found that 15-frame
  local coherence improved known-positive recall at candidate budget 20 from
  0.5409 to 0.6053 in all four held-out bursts. Lagged recurrence reached
  0.5886. Neighborhood context therefore remains useful for proposal/ranking,
  even though it adds little once the correct ROI center is supplied.
- The prespecified Raw/ICA/LS consensus and multiscale-persistence extensions
  did not beat the carrier for localization. They did preserve cross-burst site
  ordering better than the saturated amplitude features: median available
  burst-pair Spearman correlations were 0.633 and 0.649. These are candidates
  for an observability/reliability profile, not replacements for amplitude.
- Center--annulus subtraction and variance stabilization were complementary
  nuisance diagnostics, not primary event scores in this recording. Their mean
  localization percentiles were 0.907 and 0.883, respectively.

## Data contract and estimand

The analysis contains 106 confirmed neuron--burst occurrences nested in 50
immutable coordinate-defined sites across four annotated bursts. For every
feature and occurrence, the maximum inside the one-based inclusive event
interval was compared with maxima from every same-duration window wholly inside
the guarded quiet set. The quiet set excludes all four burst intervals plus a
15-frame guard on each side. The primary statistic is the empirical percentile
of the event maximum in that quiet-window distribution.

The site, rather than the occurrence or frame, is the bootstrap unit. This
preserves dependence among repeated bursts at one geometry. Quiet windows are a
temporal reference distribution; they are not verified biological negatives.
The estimand is event localization at already confirmed centers, not full-field
precision, specificity, source identity, or spike inference.

## Frozen panel

The panel contains nine established or previously frozen features and two
prespecified extensions:

1. Raw center amplitude;
2. CS--Parzen ICA center amplitude;
3. local-standardization center amplitude;
4. Raw center-minus-annulus contrast;
5. variance-stabilized center-minus-annulus contrast;
6. an exponential matched filter on local-standardized amplitude;
7. the frozen amplitude carrier;
8. exact 15-frame local coherence;
9. exact lag-2, 15-frame recurrence;
10. the minimum quiet-calibrated percentile shared by Raw, ICA, and local
    standardization (representation consensus);
11. the geometric mean of causal 3-, 7-, and 15-frame positive local-
    standardization averages (multiscale persistence).

## Full-trace results

| Feature | Site-weighted mean percentile [95% CI] | Paired difference from carrier | Median burst-pair site-rank correlation |
| --- | ---: | ---: | ---: |
| Lag-2 recurrence, 15 frames | 0.996 [0.987, 1.000] | +0.002 [0.000, 0.006] | undefined at the ceiling |
| Frozen amplitude carrier | 0.994 [0.981, 1.000] | reference | undefined at the ceiling |
| Local coherence, 15 frames | 0.994 [0.981, 1.000] | 0.000 [0.000, 0.000] | undefined at the ceiling |
| Raw center | 0.991 [0.973, 1.000] | -0.003 [-0.008, 0.000] | undefined at the ceiling |
| Raw/ICA/LS consensus | 0.986 [0.963, 0.999] | -0.008 [-0.018, -0.001] | 0.633 |
| LS multiscale persistence | 0.972 [0.945, 0.992] | -0.022 [-0.040, -0.007] | 0.649 |
| CS--Parzen ICA center | 0.964 [0.927, 0.992] | -0.030 [-0.057, -0.008] | undefined at the ceiling |
| LS exponential matched filter | 0.958 [0.918, 0.990] | -0.036 [-0.071, -0.009] | undefined at the ceiling |
| Local-standardization center | 0.935 [0.894, 0.969] | -0.059 [-0.097, -0.028] | 0.524 |
| Raw center-minus-annulus | 0.907 [0.857, 0.949] | -0.086 [-0.134, -0.043] | 0.377 |
| VST center-minus-annulus | 0.883 [0.824, 0.934] | -0.111 [-0.171, -0.059] | 0.492 |

Every feature passed the deliberately broad localization gate, but none passed
the incremental-over-carrier gate. This is not evidence that all features are
equivalent. The primary percentile is saturated for the strongest features:
104/106 carrier scores, 104/106 coherence scores, and 105/106 lag scores equal
1.0. Feature selection must therefore use the task-specific evidence below
rather than the displayed rank alone.

## Reconciliation with the protected feature audit

The earlier audit evaluated full-field proposals and rankings under sparse
positive labels. At budget 20, carrier-only macro recall was 0.5409, compared
with 0.6053 for cross-fitted local coherence and 0.5886 for lagged recurrence.
Coherence won against the carrier in all four held-out bursts. On identical
proposals, the selected scientific feature produced gains at moderate budgets,
although most of that gain came from burst 2.

There is no contradiction with the new ceiling result:

- **Known-center event localization:** amplitude is the minimal strong feature.
- **Unknown-center proposal and scarce-budget ranking:** spatial coherence adds
  useful neighborhood evidence.
- **Temporal recurrence:** lagged correlation can consolidate activity, but it
  is descriptive and cannot establish causal propagation.
- **Stable site profiling:** consensus and multiscale persistence retain more
  cross-burst rank variation after amplitude percentiles saturate.
- **Nuisance diagnosis:** annulus contrast and variance stabilization quantify
  local contamination/noise behavior but should not replace amplitude by
  default.

## Relationship to established methods

This role-specific interpretation matches the modular design of established
calcium-imaging pipelines. PCA/ICA historically combined source separation with
cellular signal extraction; modern CNMF/CaImAn methods explicitly separate
localized spatial footprints, temporal traces, and background/neuropil terms.
CNMF-E uses local correlation and peak-to-noise information to initialize
candidate centers. FISSA shows why local neuropil estimation is valuable while
also documenting that simple subtraction can over-correct. OASIS and related
methods use autoregressive calcium dynamics for spike inference after spatial
shapes are known. Benchmark studies further show that algorithm performance is
recording- and indicator-dependent and must be tested against appropriate
ground truth.

The present matched filter is therefore best interpreted as a lightweight
kinetic comparator, not spike inference. The present coherence score is aligned
with established correlation/PNR seeding logic. The VST lane is aligned with
the measured shot-noise-like variance structure, but its weaker event
localization argues for retaining it as an auxiliary calibration or residual
feature. The two new consensus/persistence summaries remain exploratory because
they were evaluated on the same recording that motivated them.

## Feature-engineering decision

Freeze the next compact panel by role:

1. **Carrier amplitude** for event strength and trace-level localization.
2. **15-frame local coherence** for proposal generation and early-budget
   ranking.
3. **Lag-2 recurrence** as a secondary temporal-context score, labeled
   non-causally.
4. **Center--annulus contrast** as a nuisance/localization diagnostic rather
   than a universal subtraction.
5. **Multiscale persistence and representation consensus** as exploratory
   observability features for site reliability, never as established biological
   phenotypes.
6. **VST residuals** for acquisition/noise diagnostics, not as the default
   amplitude carrier.

Do not widen the feature grid on this recording. The decisive next tests are an
independent recording with the frozen definitions, and an exhaustively reviewed
bounded field that makes precision estimable.

## Artifacts and validation

The validated run is
`Outputs/NeuronIdentifiability/spon_ca_burst_full_trace_feature_panel_v2`.
It includes source hashes, 5,000-draw paired bootstrap distributions, all
occurrence-level scores, burst summaries, repeatability tables, an artifact
index, and a validation record. The manuscript import verifies every indexed
artifact before copying the figure and summary tables.

Primary literature used for research-state comparison:

- Mukamel EA, Nimmerjahn A, Schnitzer MJ. *Neuron* (2009),
  doi:10.1016/j.neuron.2009.08.009.
- Pnevmatikakis et al. *Neuron* (2016), doi:10.1016/j.neuron.2015.11.037.
- Zhou et al. *eLife* (2018), doi:10.7554/eLife.28728.
- Giovannucci et al. *eLife* (2019), doi:10.7554/eLife.38173.
- Keemink et al. *Scientific Reports* (2018),
  doi:10.1038/s41598-018-21640-2.
- Friedrich et al. *PLOS Computational Biology* (2017),
  doi:10.1371/journal.pcbi.1005423.
- Berens et al. *PLOS Computational Biology* (2018),
  doi:10.1371/journal.pcbi.1006157.
