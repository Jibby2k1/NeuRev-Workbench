# NeuRev: ICA and Unsupervised Representation Evaluation Plan

## Codex-Oriented Research and Implementation Specification

**Status:** Proposed experimental plan
**Primary goal:** Determine whether ICA and related learned operators discover reproducible, statistically meaningful structure in zebrafish neuroimaging video **without using event labels to train the representation**, and establish when it is justified to progress from the current two-time-step ICA sanity case to richer learned operators.

---

## 0. Codex Execution Directive

This document is both a research specification and an implementation contract. **Do not implement every longer-term branch at once.** Execute the plan sequentially and stop at the defined gates.

For the first implementation pass, Codex should:

1. audit the existing ICA/data/evaluation implementation without refactoring unrelated code;
2. implement the canonical two-frame ICA reproduction and analytic baselines;
3. add component-alignment, stability, and null-test utilities with tests;
4. generate the required interpretation figures;
5. freeze the selected unsupervised representation before external label evaluation;
6. generate paired **Raw vs Pipeline-output traces** for relevant detections using exactly matched spatial support and time windows;
7. test whether detected candidate neurons exhibit a reproducible **candidate activation signature** in the raw data and whether the pipeline preserves/enhances that signature;
8. produce one compact results summary assigning the experiment a Level 0--3 result from Section 12;
9. **stop before multi-time-step/spatial/spatiotemporal expansion unless the preceding gate is satisfied.**

Treat the existing repository as authoritative for exact data paths, current preprocessing conventions, and evaluation interfaces. Where this document proposes an equation or directory name that conflicts with the codebase, preserve the scientific intent but adapt the implementation to the repository rather than forcing a parallel system. Record any such deviations in the implementation audit.

Do not optimize against held-out event labels during the unsupervised fitting/model-selection stage. If labels are inspected during exploratory debugging, tag the resulting run as exploratory and do not use it as confirmatory evidence.

---

## 1. Motivation

The central scientific problem is not simply whether a model can improve a supervised detection metric after enough hyperparameter search. The more interesting question is:

> **Can the data itself reveal recurring statistical signatures that are plausibly related to neural activity, without defining those signatures through event labels?**

This matters because the project began in a regime where the exact morphology of a relevant neural event is not fully known a priori. Human raters can identify candidate events visually, but this is not the same as possessing a complete mechanistic description of the signal. An unsupervised or self-supervised representation is therefore valuable if it can expose stable structures for later scientific interpretation.

The current ICA work is a natural first case because ICA imposes a precise statistical bias: it seeks a representation with reduced statistical dependence / increased non-Gaussian structure after appropriate centering and whitening. However, demonstrating that an ICA component correlates with labeled events is not by itself enough. We need to determine:

1. **What operator did ICA actually learn?**
2. **Is that operator reproducible?**
3. **Is its behavior distinguishable from trivial baselines or null data?**
4. **Does the discovered representation enrich for labeled events even though labels were not used to fit it?**
5. **Does it reveal structure beyond a known analytic operation such as temporal differencing?**
6. **If the two-frame case collapses to a simple known operator, does increasing temporal/spatial context discover anything genuinely new?**

The purpose of the experiments below is to answer those questions sequentially and terminate unproductive directions early.

---

## 2. Scientific Framing

### 2.1 What we should claim

The desired claim is **not**:

> "ICA detects neural events."

A defensible initial claim is closer to:

> **ICA fitted without event labels discovers a stable non-Gaussian temporal contrast representation, and high activations of that representation are enriched for independently labeled events relative to matched nulls and simple analytic baselines.**

If later experiments show that richer ICA / learned operators discover structures not reducible to simple differencing, the claim can be strengthened to:

> **Statistical source-separation objectives discover reproducible spatiotemporal signatures in neuroimaging video that are not explicitly specified by human labels and that exhibit measurable enrichment for independently annotated neural events.**

The labels are then **external validation**, not the definition of the representation.

### 2.2 What we should not claim yet

Do not claim that a discovered component is definitively "neural" merely because it overlaps annotations. At the present stage, prefer terms such as:

- event-enriched component,
- candidate neural-event signature,
- transient component,
- statistically discovered representation,
- non-Gaussian temporal/spatiotemporal mode.

Biological interpretation should remain downstream of statistical validation.

---

## 3. Core Experimental Principle: Separate Discovery from Evaluation

The most important change from a broad supervised hyperparameter search is to separate:

### Discovery
Fit the representation **without event labels**.

### Model selection
Prefer selection using label-free criteria whenever the experiment is intended to support an unsupervised claim.

### External evaluation
After the representation/configuration is frozen, use annotations to ask whether the discovered structure is enriched for independently labeled events.

This avoids converting an ostensibly unsupervised method into supervised model selection by repeatedly checking labels and retaining whichever configuration performs best.

If labeled metrics must be inspected during development, mark those experiments explicitly as **exploratory** and reserve a separate held-out temporal region / video for confirmatory evaluation.

---

# Part I — Canonical Two-Time-Step ICA Sanity Case

## 4. Define the Canonical Experiment

Let a sample at spatial location \(p\) be the two-time-step vector

\[
\mathbf{x}_{p,t} =
\begin{bmatrix}
I(p,t) \\
I(p,t+\Delta t)
\end{bmatrix}.
\]

The canonical pipeline is:

\[
\mathbf{x}
\xrightarrow{\text{center}}
\mathbf{x}_c
\xrightarrow{\text{whiten}}
\mathbf{z}
\xrightarrow{\text{ICA}}
\mathbf{s} = W_{\mathrm{ICA}}\mathbf{z}.
\]

Because this case has only two input coordinates, it is unusually interpretable. A likely outcome is that the dominant directions resemble:

1. a **common/intensity mode**, approximately associated with the sum/common signal across adjacent frames, and
2. a **temporal contrast mode**, approximately associated with frame-to-frame change.

The exact learned operator should **not** be assumed in advance. Instead, measure whether the learned ICA response is empirically equivalent to candidate analytic operators.

---

## 5. Analytic Baselines for the Two-Frame Case

Implement the following baselines using identical data splits and evaluation code.

### B0 — Raw intensity
A scalar intensity baseline.

### B1 — Temporal difference

\[
D_t = I_{t+\Delta t} - I_t.
\]

Evaluate signed difference and absolute difference separately if useful.

### B2 — Standardized temporal difference

Estimate the appropriate background/empirical scale and use a normalized difference. Do not hard-code a normalization formula unless it matches the current pipeline.

### B3 — Energy-normalized difference candidate

For interpretability analysis, compare ICA output to a candidate such as

\[
D_{\mathrm{EN}} =
\frac{I_{t+\Delta t}-I_t}
{\sqrt{I_t^2 + I_{t+\Delta t}^2 + \epsilon}}.
\]

**Important:** this is an analytic surrogate to test against, not an assertion that ICA is mathematically identical to this nonlinear transform.

### B4 — PCA/whitened principal directions

Evaluate the whitened/PCA coordinates before ICA. This isolates whether the apparent benefit comes from:

- centering,
- variance conditioning / whitening,
- or the ICA rotation itself.

### B5 — Random orthogonal rotation after whitening

After whitening, apply random orthogonal rotations. This is a strong control because ICA in the whitened two-dimensional space differs from alternative solutions largely by rotation/sign/permutation.

---

# Part II — What Did ICA Learn?

## 6. Operator Identification Analysis

The first goal is not detection. It is **operator identification**.

For each learned ICA component:

1. Align component sign consistently.
2. Account for ICA permutation ambiguity.
3. Compare its output against each analytic baseline.
4. Report:
   - Pearson correlation,
   - Spearman correlation,
   - \(R^2\) from a simple linear fit,
   - rank agreement for top-activation samples,
   - cosine similarity between effective linear directions when such a comparison is mathematically valid.

### Interpretation

If an ICA component is almost perfectly explained by a simple temporal difference after whitening/standardization, then:

> The two-frame ICA experiment is a successful **sanity check**, but it is not yet evidence of a novel representation.

That result is still useful. It demonstrates that the statistical objective recovers an expected transient-sensitive direction from the data rather than from labels.

If the component cannot be explained well by the analytic baselines, inspect why before treating this as a positive result. Differences may arise from preprocessing, data-dependent whitening, scaling, saturation, or artifacts.

---

## 7. Distributional Characterization

For each component and each baseline, characterize the empirical response distribution.

Store/report at minimum:

- mean,
- standard deviation,
- skewness,
- excess kurtosis,
- selected quantiles,
- robust tail mass,
- optional negentropy approximation,
- optional fitted generalized-Gaussian shape parameter.

The important question is:

> Does the candidate component create a heavier-tailed / more non-Gaussian response than trivial alternatives, and are those tails associated with coherent video events rather than noise?

Do **not** equate high kurtosis or high non-Gaussianity with neural relevance automatically.

---

# Part III — Is the Representation Real?

## 8. Stability Analysis

A meaningful unsupervised representation should not depend strongly on arbitrary initialization or a narrow sample of the video.

### 8.1 Seed stability

Fit ICA using multiple random seeds.

For each pair of fits:

- resolve sign and permutation ambiguity,
- compare effective component directions,
- compare component activation maps,
- compare top-\(K\) activated samples.

Suggested summary metrics:

- mean aligned component cosine similarity,
- activation correlation,
- top-\(K\) Jaccard overlap.

### 8.2 Data-resampling stability

Repeat fitting on:

- temporal blocks,
- bootstrap samples,
- spatial subsamples,
- different fit/sample budgets.

Evaluate whether the same qualitative components reappear.

### Continue criterion

Proceed to richer ICA models only if the transient-sensitive representation is reasonably stable under seeds and moderate resampling.

### Stop / diagnose criterion

If component identity changes dramatically across small perturbations of the fitting set, diagnose conditioning, sample complexity, preprocessing, and degeneracy before scaling the architecture.

---

## 9. Null Models

A component that appears in real data should be compared against appropriately destroyed structure.

Implement at least two nulls.

### N1 — Temporal permutation / temporal block shuffle

Destroy local temporal structure while preserving marginal intensity statistics as much as practical.

Question:

> Does the transient-sensitive component or its event-like tail disappear when local temporal organization is destroyed?

### N2 — Spatially/temporally matched surrogate

Use a surrogate that approximately preserves low-order statistics while destroying coherent events. The exact implementation may depend on the data and should be documented.

Possible candidates:

- block permutation,
- phase-randomized surrogate when appropriate,
- local shuffle within matched background regions.

### Optional N3 — Random orthogonal post-whitening transform

This specifically asks whether ICA selects a statistically privileged rotation beyond arbitrary whitened directions.

### Null evaluation

Compare real vs null on:

- non-Gaussianity,
- stability,
- tail coherence,
- spatial/temporal contiguity of activations,
- event-label enrichment only in the final external evaluation stage.

---

# Part IV — External Label Evaluation Without Training on Labels

## 10. Labels as an Independent Scientific Assay

Once the representation and analysis procedure are frozen, annotations answer:

> Are the extreme activations of the discovered representation enriched for human-labeled events beyond what would be expected by chance or by trivial baselines?

This is substantially different from training directly on the labels.

---

## 11. Recommended Evaluation Metrics

Because neural events are sparse, avoid relying on raw accuracy.

### 11.1 Event enrichment

For a chosen activation quantile/top-\(K\) set:

\[
\text{Enrichment} =
\frac{P(\text{label} \mid \text{high activation})}
{P(\text{label})}.
\]

Also report an odds ratio or equivalent contingency-table statistic where appropriate.

### 11.2 Detection-oriented metrics

When converting activations into detections, report metrics consistent with the existing evaluation pipeline, such as:

- event-level recall / TPR,
- false positives per frame or per image (FPPI),
- precision,
- AUPRC where appropriate,
- localization overlap if spatial detections are produced.

### 11.3 Label-shift null

Circularly shift or otherwise misalign annotations relative to the video while preserving annotation frequency/structure.

Compare the true alignment against this null distribution.

This directly tests whether observed enrichment depends on correct temporal/spatial correspondence.

### 11.4 Paired Raw vs Pipeline-Output Traces

For every scientifically relevant detection category, generate **paired temporal traces at the same spatial support**:

1. **Raw trace:** literal unmodified video intensity sampled from the detected ROI / neuron support.
2. **Pipeline trace:** the corresponding scalar pipeline/operator output over the same ROI and time axis.
3. **Optional local-background trace:** a matched annulus or existing background estimator when this is already scientifically justified by the pipeline.
4. **Derived normalized traces:** baseline-centered / standardized copies for morphology comparison. These must never replace the literal raw trace.

The purpose is to determine what signal existed in the original video and what the pipeline did to it. A transformed trace alone is insufficient because normalization, whitening, differencing, or nonlinear processing may create visually compelling waveforms that are not obvious in the original intensity signal.

#### Spatial support

Use the **same ROI/support** for Raw and Pipeline traces. Do not choose a better-looking ROI independently for either representation.

Preferred hierarchy:

- existing neuron/ROI segmentation if the repository already provides it;
- detected connected component / object support;
- otherwise a fixed local patch centered on the detection.

If an object cannot be reliably identified as a neuron, call it a **candidate ROI** rather than a neuron.

#### Temporal support

For every event, extract an event-centered window with configurable pre-event and post-event context. The canonical window should be selected from the known event-duration / acquisition-rate characteristics of the current dataset and then frozen. Do not tune the trace window to maximize apparent waveform quality.

Also support a per-ROI full-video trace for a small set of representative detections when this remains computationally reasonable.

#### Required detection categories

After labels are allowed for external evaluation, trace panels should include at minimum:

- representative true positives;
- representative false positives;
- labeled false negatives sampled at the labeled ROI even though no detection was produced;
- high-activation unlabeled candidates;
- matched background / no-event windows.

Do not show only the best-looking examples. Define deterministic selection rules such as strongest, median-strength, and randomly seeded examples within each category.

#### Trace-level measurements

Where meaningful, compute:

- pre-event baseline median;
- robust baseline noise estimate;
- signed and absolute peak amplitude;
- peak SNR / contrast;
- time to peak;
- rise time;
- decay / half-decay time;
- full width at half maximum;
- signed and absolute area under the event window;
- Raw-to-Pipeline SNR gain;
- Raw-to-Pipeline peak-time offset;
- Raw-to-Pipeline width ratio;
- Raw-to-Pipeline trace correlation.

Not every quantity will be valid for every preprocessing branch. Explicitly mark non-applicable metrics rather than forcing them.

### 11.5 Candidate Neuron-Activation Signature Analysis

For the neurons / candidate ROIs that the pipeline detects, test whether there is a **reproducible temporal activation morphology** rather than merely isolated large activations.

The scientific hypothesis is:

> Detected event-associated ROIs contain a recurring temporal waveform in the Raw signal, and the pipeline preferentially enhances or exposes this waveform relative to matched non-event activity.

This should initially be called a **candidate activation signature**, not a definitive biological neuron signature.

#### A. Build event-centered Raw trace sets

For each detected ROI, collect event-centered Raw windows. Preserve two representations:

1. **amplitude-preserving trace** for physical/SNR interpretation;
2. **shape-normalized trace** for morphology comparison, using only a frozen baseline-centering and amplitude normalization rule.

Do not normalize each trace in a way that forces the same peak or artificially increases similarity.

#### B. Avoid repeated-event domination

If one neuron/ROI contributes many detected events, it must not dominate the population signature.

Compute both:

- event-level statistics; and
- neuron/ROI-level summaries, where each neuron contributes equal weight.

A useful hierarchy is:

\[
\text{events} \rightarrow \text{per-neuron template} \rightarrow \text{population template}.
\]

#### C. Event-triggered template

Generate robust event-aligned summaries:

- median trace;
- mean trace;
- bootstrap confidence band;
- per-neuron normalized heatmap ordered by peak timing or template correlation.

For confirmatory labeled events, prefer alignment to the externally defined event onset / interval when available. For label-free detections, align to the frozen detection time or operator peak. Do **not** independently shift traces to maximize template correlation in the primary analysis.

#### D. Quantify signature consistency

Report:

- pairwise trace correlation;
- correlation of each held-out trace with a template fit without that trace;
- leave-one-neuron-out template correlation;
- variance explained by the first temporal PCA mode;
- distribution of rise/decay/width parameters;
- sign consistency;
- peak-time dispersion.

A single averaged waveform is not enough. The signature must be supported by across-neuron consistency.

#### E. Test one signature before clustering

First test the hypothesis that one dominant activation morphology is sufficient.

Only if the population is clearly heterogeneous should Codex perform a small, interpretable clustering analysis over shape-normalized traces. Candidate methods include PCA followed by hierarchical clustering or a small Gaussian-mixture / k-means analysis. Avoid a broad clustering hyperparameter search.

For every proposed cluster require:

- minimum support across multiple neurons, not merely multiple events from one neuron;
- stability under resampling;
- a distinct medoid/template;
- better within-cluster similarity than matched null windows.

#### F. Null comparisons for the signature

Compare candidate activation signatures against:

1. temporally shifted windows from the **same detected neurons/ROIs**;
2. matched no-event windows from the same neurons/ROIs;
3. matched spatial background / non-detected ROIs;
4. false-positive detections;
5. label-shift nulls where labels are used.

This is essential because slow fluorescence drift, motion, photobleaching, and filtering can produce shared waveform shapes even without neural events.

#### G. Held-out generalization

If a candidate signature is found, construct its template on a subset of neurons/ROIs and score held-out neurons/ROIs using a frozen similarity metric.

A stronger result is:

> A template discovered from one set of detected neurons matches event-centered Raw traces from held-out neurons significantly better than matched null windows.

This is much more informative than showing an attractive population average.

#### H. Relationship to ICA / learned temporal operators

For multi-time-step ICA or another learned temporal operator, directly compare the learned temporal component to the empirically estimated Raw activation template.

Ask:

- Does the learned component resemble the candidate activation waveform?
- Does convolving/projecting Raw traces onto the component selectively increase the signature-to-background contrast?
- Does the pipeline preserve event timing and width, or does it systematically distort them?

This connects the **representation-learning result** to the **observed biological-signal morphology** rather than treating detection performance and signal interpretation as separate analyses.

### 11.6 Signature Evidence Levels

Track activation-signature evidence separately from the ICA Level 0--4 taxonomy.

- **S0 — No reproducible signature:** detected traces are heterogeneous or indistinguishable from matched null windows.
- **S1 — Within-ROI repeatability:** repeated events from the same neuron/ROI show a consistent shape.
- **S2 — Across-neuron signature:** a common template generalizes across multiple neurons/ROIs and exceeds matched null similarity.
- **S3 — Held-out signature:** a template frozen on one subset predicts the temporal morphology of events in held-out neurons/ROIs and remains enriched for independently labeled events.

Do not use the phrase **neuron activation signature** as a strong scientific claim unless at least S2 evidence is reached.

---

# Part V — What Counts as a Nontrivial Result?

## 12. Result Taxonomy

Use the following interpretation ladder.

### Level 0 — Failure

ICA is unstable, degenerate, or indistinguishable from random whitened rotations.

**Action:** stop and diagnose preprocessing/conditioning.

### Level 1 — Sanity-check recovery

ICA consistently recovers a direction approximately equivalent to temporal differencing or another simple expected operation.

**Meaning:** the objective is behaving sensibly, but the representation is not novel.

**Action:** use as a validated baseline and move to richer context.

### Level 2 — Statistically privileged representation

ICA produces a stable component that is measurably more non-Gaussian / less dependent than analytic or random-rotation baselines and behaves differently from null data.

**Meaning:** ICA identifies structure privileged by its statistical objective.

**Action:** perform external label enrichment analysis.

### Level 3 — Event-enriched unsupervised representation

A frozen ICA representation is significantly enriched for independent event labels relative to matched baselines and label-shift nulls.

**Meaning:** the representation is scientifically useful as an unsupervised candidate-event feature.

**Action:** investigate larger temporal/spatial context.

### Level 4 — Novel richer representation

Multi-time-step, local spatial, multiscale, or spatiotemporal ICA discovers stable structures that cannot be reduced to simple temporal difference / local normalization operators and that retain external event enrichment.

**Meaning:** this is the first level that strongly supports a nontrivial representation-learning contribution.

---

# Part VI — Progression Beyond Two-Frame ICA

## 13. Sequential Expansion Strategy

Do **not** immediately deploy a large combinatorial hyperparameter grid. Expand representation capacity only after the simpler case passes the relevant gates.

### Stage A — Two-frame ICA

Purpose: verify the entire statistical/evaluation framework.

Continue only if:

- the result is stable,
- real data differs meaningfully from nulls,
- and/or the representation shows label enrichment beyond simple baselines.

### Stage B — Multi-time-step temporal ICA

Use windows such as

\[
\mathbf{x}_{p,t} =
[I(p,t-k), \ldots, I(p,t), \ldots, I(p,t+k)]^\top.
\]

Questions:

- Does ICA discover temporal motifs rather than a single derivative-like direction?
- Do components resemble first derivative, second derivative, onset/offset, pulse, oscillatory, or longer-lived signatures?
- Are these components stable?
- Do multiple distinct components show different event enrichment profiles?

This is one of the most important next experiments because it gives ICA enough dimensionality to discover structure that cannot be represented by a single two-frame contrast.

### Stage C — Local spatial ICA

Fit over local patches.

Questions:

- Are discovered filters spatially coherent?
- Do they recover center-surround, edge-like, blob-like, or structured cellular patterns?
- Are the filters stable across locations and videos?
- Are they merely recapitulating local whitening/PCA?

### Stage D — Spatiotemporal ICA

Only proceed after temporal and spatial cases are understood independently.

Spatiotemporal patches have much higher dimensionality and potentially severe redundancy/sample-complexity problems.

Require explicit monitoring of:

- covariance conditioning,
- effective rank,
- sample-to-dimension ratio,
- component stability,
- degeneracy,
- computation/memory cost.

### Stage E — Multiscale / hierarchical ICA

Only justify multiscale or hierarchical variants if simpler ICA demonstrates scale-dependent structure or if a single receptive-field scale demonstrably conflates different phenomena.

---

# Part VII — Where a Learned Additive Operator Fits

## 14. Residual Learned Operator

A useful first learned-operator test is a residual form such as

\[
Y = X + \alpha\,\mathcal{O}_\theta(X),
\]

or a closely related additive composition.

This is a **good minimal test**, but additivity is not the scientific hypothesis.

The residual form is useful because it provides:

- an identity path,
- stable optimization,
- an interpretable correction term,
- a direct way to inspect what the learned operator adds/removes.

Backpropagation does **not** require additivity; any differentiable transform can participate in backpropagation. Therefore, richer non-additive transforms remain available later.

The important question is:

> What statistical transformation does \(\mathcal{O}_\theta\) learn, and is that transformation useful/stable/nontrivial?

---

## 15. Relationship to Whitening and Standardization

Whitening and standardization are not simply additive feature operations.

For centered data \(\mathbf{x}_c\), whitening takes the form

\[
\mathbf{z} = C^{-1/2}\mathbf{x}_c,
\]

which rescales and rotates the feature space based on covariance structure.

This suggests two distinct research directions:

### Direction 1 — Residual feature discovery

Learn a correction/operator and test whether it contributes useful structure.

### Direction 2 — Learned conditioning

Learn parameters that transform the geometry/statistics of the representation itself, e.g.:

- learned centering,
- learned scaling,
- learned local covariance correction,
- learned whitening-like transform,
- learned spatial/temporal conditioning.

These should not be conflated. A residual learned operator is the simplest probe; a learned conditioning transform is a richer later hypothesis.

---

# Part VIII — Entropy and ICA Objectives

## 16. Why Entropy Is Still Relevant

Entropy can remain central, but "maximize entropy" by itself is not an adequate objective.

Unconstrained differential entropy can increase because of scale, noise, or other undesirable behavior. ICA avoids this triviality through its particular formulation, commonly involving whitening, invertible transforms, nonlinear contrast functions, independence, or InfoMax-style objectives.

A useful conceptual framing is:

> **Use information-theoretic objectives to encode the statistical property we want, while separately constraining scale, covariance, reconstruction, or task-relevant information.**

---

## 17. Candidate Information-Theoretic Objectives

### 17.1 ICA / independence objective

For multiple learned components \(\mathbf{s}\), seek low dependence, e.g. low mutual information / total correlation.

Conceptually:

\[
\mathrm{TC}(\mathbf{s})
= D_{KL}\!\left(
 p(\mathbf{s})\;\|\;\prod_i p(s_i)
\right).
\]

Minimizing total correlation generalizes the independence motivation of ICA.

### 17.2 Negentropy / non-Gaussianity

Under controlled variance/whitening, encourage non-Gaussian components.

A conceptual negentropy is

\[
J(s)=H(s_{\mathrm{Gauss}})-H(s),
\]

where the Gaussian has matched variance.

In practice use stable approximations/contrast functions rather than direct entropy estimation unless there is a strong reason otherwise.

### 17.3 Redundancy reduction

For multiple channels/scales/features, penalize covariance or statistical dependence between outputs while preventing collapse.

This may be especially natural for multiscale learned operators.

### 17.4 Entropy-rate / temporal predictability objectives

Entropy rate may become relevant if the desired distinction is between:

- persistent/predictable background structure,
- transient innovations,
- noise.

However, **do not add entropy-rate machinery until a precise hypothesis is written**. Depending on sign and formulation, minimizing entropy rate can encourage predictable/static structure, while maximizing innovation-type entropy can emphasize unpredictable noise as well as events.

This direction therefore requires explicit controls for noise and should be treated as a later experiment.

---

## 18. Hybrid Supervised + Information-Theoretic Objective

If labels are used for the current learned-operator sanity test, a later hybrid objective can be tested:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{sup}}
+ \lambda_{\mathrm{info}}\mathcal{L}_{\mathrm{info}}
+ \lambda_{\mathrm{cond}}\mathcal{L}_{\mathrm{cond}}.
\]

Possible terms:

- \(\mathcal{L}_{\mathrm{sup}}\): event detection/localization objective,
- \(\mathcal{L}_{\mathrm{info}}\): independence, negentropy, or redundancy-reduction objective,
- \(\mathcal{L}_{\mathrm{cond}}\): variance/covariance/whitening constraint.

This experiment answers:

> Does an ICA-like information-theoretic bias produce a representation that is better behaved or more generalizable than supervision alone?

This is **not** the same as the unsupervised claim and should be reported separately.

---

# Part IX — Proposed Unsupervised Learned-Operator Research Direction

## 19. General Objective

After the ICA baseline is understood, consider a parameterized operator

\[
\mathbf{z}=\mathcal{O}_\theta(\mathbf{x})
\]

trained without event labels using constraints inspired by source separation and conditioning.

A generic experimental family is:

\[
\mathcal{L}_{\mathrm{unsup}}
=
\lambda_{\mathrm{dep}}\,\mathcal{L}_{\mathrm{dependence}}
+\lambda_{\mathrm{white}}\,\|\operatorname{Cov}(\mathbf{z})-I\|_F^2
+\lambda_{\mathrm{stab}}\,\mathcal{L}_{\mathrm{stability}}
+\lambda_{\mathrm{reg}}\,\mathcal{R}(\theta).
\]

Potential additions must be hypothesis-driven, not accumulated arbitrarily.

Examples:

- local/spatiotemporal dependence reduction,
- multiscale redundancy reduction,
- sparse/heavy-tailed component preference,
- temporal innovation extraction,
- invariance/equivariance to nuisance transforms,
- reconstruction or information-preservation constraints if collapse becomes possible.

The learned operator should then be evaluated using the **same stability, null, interpretability, and external-label enrichment framework** as ICA.

---

# Part X — Experiment Orchestration for Codex

## 20. Implementation Rules

Codex should follow these rules.

1. **Do not modify or remove existing experimental functionality unless required.**
2. Prefer new experiment modules/configuration files for this study.
3. Reuse existing data loading, annotation, and evaluation utilities where reliable.
4. Separate representation fitting from label evaluation in code.
5. Make it impossible to accidentally pass labels into the unsupervised fitting function.
6. Record exact random seeds and configuration hashes.
7. Keep run artifacts compact; store essential numerical summaries and only the visuals needed for interpretation.
8. Cache fitted transforms/components when possible so evaluation does not refit models.
9. Preserve ICA sign/permutation ambiguity explicitly in comparison utilities.
10. Add unit tests for all metric/alignment/null utilities before launching broad experiments.

---

## 21. Suggested Repository Structure

Adapt names to the existing repository rather than forcing this exact layout.

```text
experiments/
  ica_representation_eval/
    README.md
    configs/
      two_frame_canonical.yaml
      two_frame_nulls.yaml
      temporal_window.yaml
    run_fit.py
    run_analysis.py
    run_external_eval.py
    baselines.py
    nulls.py
    component_alignment.py
    metrics.py
    visualization.py
    schemas.py
    tests/
      test_component_alignment.py
      test_null_generation.py
      test_enrichment.py
      test_split_integrity.py
```

The critical architectural boundary is:

```text
UNLABELED VIDEO
      |
      v
representation fitting
      |
      v
frozen representation
      |
      +------------------------+
      |                        |
      v                        v
label-free analysis        held-out labels
(stability/nulls/etc.)     external evaluation only
```

---

## 22. Run Metadata

For every run, store at minimum:

```json
{
  "run_id": "...",
  "experiment": "two_frame_ica",
  "git_commit": "...",
  "config_hash": "...",
  "seed": 0,
  "fit_split": "...",
  "eval_split": "...",
  "n_fit_samples": 0,
  "status": "completed"
}
```

Do not save enormous intermediate arrays by default.

For trace analysis, preserve compact reproducible trace artifacts rather than rendered figures alone. At minimum, store:

- a trace manifest mapping `run_id`, `detection_id`, `roi_id/neuron_id`, spatial support, center frame, category, and label status when labels are permitted;
- event-centered Raw trace windows;
- matched Pipeline-output trace windows;
- optional local-background windows;
- the exact normalization parameters used to create morphology-only traces.

Use the repository's native efficient array format if one exists. Otherwise a compact NPZ/Zarr/HDF5 plus CSV/Parquet manifest is acceptable. Do not duplicate full-video arrays per run.

---

## 23. Essential Metrics Schema

Store only essential scalar metrics in the main run table / JSONL.

Suggested fields:

```text
# Representation
component_stability_cosine
activation_stability_corr
topk_stability_jaccard
component_kurtosis
component_skewness
component_negentropy_approx   # optional

# Analytic equivalence
corr_temporal_difference
corr_energy_norm_difference
r2_best_analytic_baseline

# Null comparison
real_vs_null_effect_size
real_vs_null_pvalue_or_empirical_quantile

# External labels
label_enrichment
label_odds_ratio
auprc                      # if appropriate
event_recall               # if thresholded
fppi                       # if thresholded
label_shift_empirical_p

# Raw/Pipeline trace fidelity
raw_peak_snr_median
pipeline_peak_snr_median
raw_to_pipeline_snr_gain
raw_pipeline_peak_offset_frames
raw_pipeline_width_ratio
raw_pipeline_trace_corr

# Candidate activation signature
signature_pairwise_corr_median
signature_leave_one_neuron_out_corr
signature_first_pc_variance_explained
signature_vs_null_effect_size
signature_vs_null_empirical_p
signature_evidence_level     # S0-S3

# Conditioning / diagnostics
cov_condition_number
cov_effective_rank
fit_samples_per_dimension
```

Not every metric applies to every stage. Missing values should be explicit rather than silently replaced with zero.

---

# Part XI — Required Visuals

## 24. Two-Frame ICA Visual Package

Generate the following first.

### V1 — Geometry of two-frame ICA

A 2D scatter plot in \((I_t, I_{t+1})\) or centered/whitened coordinates showing:

- data cloud,
- PCA/whitening axes,
- ICA axes,
- common-mode direction,
- temporal-difference direction.

This is the most important explanatory figure for the two-frame case.

### V2 — Learned vs analytic response

Scatter/hexbin plots:

- ICA component vs raw difference,
- ICA component vs standardized/energy-normalized difference.

Include correlation and \(R^2\).

### V3 — Response distributions

Overlay or separate histograms/density curves for:

- ICA transient component,
- temporal difference baseline,
- random whitened rotation,
- null-data ICA response.

### V4 — Top activation montage

Show the highest-activation examples in their spatial/temporal context.

The purpose is scientific interpretation, not merely qualitative aesthetics.

### V5 — Stability plot

Show aligned component direction/activation similarity across seeds or temporal blocks.

### V6 — Label enrichment curve

Plot label enrichment as a function of activation percentile / top-\(K\) budget.

Include analytic baselines and label-shift null confidence intervals.

### V7 — Paired Raw vs Pipeline trace panels

For deterministic representative examples, show Raw and Pipeline-output traces on the same time axis with:

- event/detection time marked;
- annotation interval overlaid only in the external-label stage;
- identical ROI support;
- literal Raw trace plus a clearly labeled normalized copy when normalization helps visual comparison.

Include at least representative true positives, false positives, false negatives, high-activation unlabeled candidates, and matched background windows.

### V8 — Event-aligned Raw/Pipeline population traces

Show event-aligned Raw and Pipeline population summaries with median/mean and bootstrap confidence bands. Also show neuron-balanced summaries so one repeatedly active neuron cannot dominate the figure.

### V9 — Candidate activation-signature heatmap and template

Display shape-normalized per-neuron/ROI traces as a heatmap together with the frozen population template. Keep amplitude-preserving statistics separate.

### V10 — Signature similarity vs null controls

Plot the frozen template-similarity score for:

- held-out detected/event-associated neurons;
- false positives;
- same-neuron shifted/no-event windows;
- matched background ROIs.

Include the corresponding effect size and empirical null significance.

### V11 — Pipeline fidelity / distortion

Summarize whether the pipeline enhances the event while preserving temporal morphology. Useful plots include:

- Raw SNR vs Pipeline SNR;
- Raw peak time vs Pipeline peak time;
- Raw width vs Pipeline width.

A pipeline that improves separability by severely shifting or narrowing events should be explicitly identified as such rather than described simply as improved.

---

# Part XII — Sequential Continue / Stop Gates

## 25. Gate 0 — Pipeline Integrity

**Required before scientific interpretation:**

- no label leakage into unsupervised fit,
- deterministic/reproducible split handling,
- ICA alignment code validated,
- baselines share identical evaluation data,
- null-generation tests pass.

If any fail: **STOP.**

---

## 26. Gate 1 — Two-Frame Interpretability

Questions:

- Is the learned ICA component stable?
- Can it be mapped to an understandable temporal contrast/common-mode structure?
- Is it better defined than random post-whitening rotations?

If no: **STOP and diagnose ICA/preprocessing.**

If yes: continue.

---

## 27. Gate 2 — Nontrivial Statistical Structure

Questions:

- Does real-data ICA differ meaningfully from nulls?
- Is its non-Gaussian structure reproducible?
- Are extreme activations coherent rather than obvious noise/artifact?

If no: treat the result as a limited sanity check and **do not scale to large ICA architectures yet**.

If yes: continue.

---

## 28. Gate 3 — Independent Event Enrichment

Freeze the representation before checking labels.

Question:

- Are high activations more enriched for true labels than analytic baselines and label-shift nulls?

If no: ICA may still characterize the video statistically, but the evidence for event relevance is weak. **Do not claim neural-event discovery.**

If yes: continue.

---

## 29. Gate 4 — Richer Temporal Context

Test modest temporal windows before large spatial/spatiotemporal models.

Proceed only if:

- new components are reproducible,
- at least one cannot be reduced to the known two-frame operators,
- null behavior remains favorable,
- external enrichment does not disappear.

If larger windows merely rediscover derivative-like components with no added information, stop increasing temporal dimensionality.

---

## 30. Gate 5 — Spatial / Spatiotemporal Scaling

Before every higher-dimensional fit, inspect:

- effective rank,
- covariance spectrum,
- condition number,
- sample/dimension ratio,
- component stability.

If conditioning becomes poor or solutions become unstable, prioritize dimensionality reduction, local whitening, regularization, or smaller receptive fields rather than brute-force fitting.

---

# Part XIII — Hyperparameter Search Philosophy

## 31. Avoid Label-Driven "Spray and Pray"

For experiments intended to demonstrate unsupervised discovery, do not select from hundreds/thousands of representations using the labeled detection metric.

Instead:

1. define a scientifically motivated range,
2. use unsupervised diagnostics for initial selection,
3. evaluate stability,
4. lock the configuration,
5. then inspect held-out label enrichment.

Sensitivity analysis is still useful, but it should answer:

> **Is the scientific conclusion robust across a reasonable parameter neighborhood?**

rather than:

> **Which arbitrary configuration maximizes the labels?**

If a broad search is needed for computational reasons, clearly separate:

- exploratory search,
- confirmatory frozen evaluation.

---

# Part XIV — Immediate Codex Work Order

## 32. Phase 1: Audit Existing Implementation

Codex should first inspect the current ICA implementation and document:

1. exact input tensor definition,
2. centering implementation,
3. whitening implementation,
4. ICA objective/solver,
5. normalization/scaling before and after ICA,
6. how components are selected/reordered,
7. current label usage,
8. current hyperparameters,
9. existing plots/metrics,
10. any data leakage risk.

Output:

```text
experiments/ica_representation_eval/IMPLEMENTATION_AUDIT.md
```

Do not refactor unrelated code during this audit.

---

## 33. Phase 2: Reproduce Canonical Two-Frame Result

Create one deterministic canonical configuration.

Required outputs:

- fitted centering/whitening transform,
- ICA directions,
- comparison to analytic baselines,
- V1–V5 visuals,
- paired Raw/Pipeline traces for deterministic top-activation and matched-background examples (without using labels for selection),
- compact metrics file.

No broad search yet.

---

## 34. Phase 3: Null + Stability Suite

Run a small but deliberate suite across:

- seeds,
- temporal resamples,
- at least two null constructions,
- random post-whitening rotations.

Create one summary table and one summary figure per diagnostic family.

Decision: assign result to Levels 0–2 from the taxonomy above.

---

## 35. Phase 4: Freeze Then Evaluate Labels

Before running the label evaluation:

- write the chosen representation/configuration to a frozen config,
- record its hash,
- do not alter it after observing label metrics without marking a new experiment as exploratory.

Then generate:

- enrichment-vs-percentile curve,
- label-shift null distribution,
- detection metrics if meaningful,
- representative true-positive / false-positive / high-activation examples,
- V7 paired Raw/Pipeline traces for true positives, false positives, false negatives, high-activation unlabeled candidates, and matched background windows,
- V8 event-aligned Raw/Pipeline population traces,
- V9 candidate activation-signature heatmap/template,
- V10 signature-vs-null comparison,
- V11 pipeline fidelity/distortion plots,
- an S0--S3 signature-evidence assignment.

Decision: determine whether Level 3 is reached and separately whether the data support an S0--S3 candidate activation signature.

---

## 36. Phase 5: Multi-Time-Step Pilot

Only if previous stages justify continuation.

Start with a **small number of temporal window sizes**, not a massive grid.

For each window:

- effective rank,
- covariance conditioning,
- component stability,
- component visualizations,
- analytic operator similarity,
- null comparison,
- frozen external-label enrichment.

The central success criterion is not a tiny detection improvement. It is the discovery of a **stable temporal component that contains structure not reducible to the two-frame baseline**.

For every promising multi-time-step temporal component, additionally compare its temporal shape against the frozen Raw candidate-activation template from Section 11.5. This tests whether the learned operator is converging toward a reproducible event morphology rather than merely maximizing a generic non-Gaussian statistic.

---

# Part XV — Questions the Final Analysis Must Answer

## 37. Mandatory Research Questions

Every experiment report should explicitly answer:

1. **What statistical assumption was imposed?**
2. **What representation was discovered?**
3. **Can that representation be reduced to a known analytic operator?**
4. **Is it stable across random initialization and data resampling?**
5. **Does the same structure appear in null data?**
6. **What are the highest-activation samples actually showing?**
7. **Are independent labels enriched in those activations?**
8. **Does the result generalize outside the exact data used to fit it?**
9. **What does the paired Raw trace show at each relevant detection, and what transformation did the Pipeline apply?**
10. **Is there a reproducible candidate activation signature across detected neurons/ROIs, and does it exceed same-neuron and matched-background nulls?**
11. **Does the Pipeline enhance the candidate signature while preserving its timing and temporal width?**
12. **What evidence would falsify the interpretation?**
13. **What specific result justifies increasing model complexity?**

If these questions cannot be answered, do not treat an improved scalar metric as sufficient evidence of scientific progress.

---

# Part XVI — Longer-Term Research Branches

## 38. Branch A — Learned Nonlinear ICA-Like Operator

Replace the linear unmixing transform with a constrained nonlinear operator while retaining independence / redundancy-reduction objectives.

Required controls:

- prevent collapse,
- control scaling,
- compare against equal-capacity autoencoder/self-supervised baselines,
- ensure the representation is not simply memorizing local intensity statistics.

---

## 39. Branch B — Learned Local Conditioning

Learn transforms that behave more like adaptive standardization/whitening than additive feature extraction.

Potential parameterizations:

- learned local mean subtraction,
- learned scale normalization,
- low-rank covariance correction,
- structured local whitening,
- separable spatial then temporal conditioning,
- multiscale covariance conditioning.

Evaluate whether these improve condition number and downstream unsupervised component stability before evaluating labels.

---

## 40. Branch C — Multiscale Redundancy Reduction

Construct features at multiple spatial/temporal scales and explicitly penalize redundant information between scales.

This may provide a stronger motivation for multiscale processing than merely concatenating multiple kernels.

Central question:

> Do different scales expose statistically distinct structures, or do they merely provide redundant versions of the same transient?

---

## 41. Branch D — Entropy-Rate / Innovation Decomposition

Investigate whether predictable background dynamics and transient innovations can be separated using temporal information-theoretic objectives.

Do not implement until a specific mathematical hypothesis and null model are written.

Possible framing:

\[
\text{video} = \text{predictable background} + \text{structured innovation} + \text{noise}.
\]

The challenge is that both true events and sensor noise can be unpredictable. A viable objective therefore needs an additional structural constraint such as spatial coherence, multiscale consistency, repeatability, or nontrivial temporal shape.

---

# Part XVII — Final Decision Logic

## 42. Compact Decision Tree

```text
Two-frame ICA reproducible?
    |
    +-- no --> diagnose preprocessing / conditioning / solver; stop expansion
    |
    +-- yes
          |
          v
Equivalent to temporal difference?
          |
          +-- yes --> treat as sanity baseline; test richer temporal context
          |
          +-- no --> characterize what differs; test nulls carefully

Real-data structure stronger/more stable than nulls?
    |
    +-- no --> weak evidence; stop or reformulate
    |
    +-- yes
          |
          v
Frozen representation enriched for independent labels?
          |
          +-- no --> statistically interesting but not yet event-relevant
          |
          +-- yes --> credible unsupervised candidate-event representation

Richer temporal/spatial context finds stable nontrivial new components?
    |
    +-- no --> retain simple model; complexity not justified
    |
    +-- yes --> pursue nonlinear / multiscale / learned-conditioning variants
```

---

# 43. Definition of Success for This Research Thread

The strongest near-term success is **not** simply a higher detection score.

A successful result is:

> A representation learned without event labels that is reproducible across fitting perturbations, statistically distinguished from appropriate nulls and trivial analytic transforms, interpretable through its activation structure, and independently enriched for human-annotated events.

The two-frame ICA experiment should establish the methodology for making that claim. If it merely rediscovers an expected normalized temporal difference, that is still useful: it provides a transparent sanity check and a validated baseline from which to ask whether additional temporal, spatial, multiscale, or nonlinear capacity discovers anything genuinely new.

The project should therefore progress from:

```text
"Does this configuration score well?"
```

 toward:

```text
"What statistical structure did the model discover,
why did it discover it,
is it reproducible,
is it absent under nulls,
and does independent biological annotation enrich for it?"
```

That shift is the core methodological contribution of this experimental plan.
